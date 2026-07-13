# orderflow-lm

Train an autoregressive model on raw market events (L2 book updates + trades), then
measure whether it works as a backtesting simulator — impact response, stylized facts,
and the replay-vs-simulator PnL gap for a fixed strategy.

Data: Tardis.dev free first-of-month datasets (incremental L2 + trades, microsecond
timestamps, no API key required for day-01 files). Current corpus: BTCUSDT perp on
Binance futures, 2026-01 through 2026-07 (922M raw L2 rows, 142M normalized events,
370M tokens).

## Pipeline

```
scripts/download.py        Tardis first-of-month fetcher -> data/raw/
src/oflm/reconstruct.py    raw L2 + trades -> normalized event parquet
src/oflm/tokenizer.py      normalized events -> uint16 token stream (.bin)
src/oflm/model.py          vanilla decoder-only GPT (~25M params), KV cache
src/oflm/generate.py       KV-cached, grammar-constrained sampling
scripts/train.py           training loop (cuda/mps/cpu)
scripts/sample.py          generate + decode to readable event log (--raw: unconstrained)
scripts/explore_raw.py     streaming stats over raw L2 files
```

## v0 data representation

Findings that drove it (BTC perp, June 2026):

- Binance L2 diffs arrive as fixed ~26ms frames (p10–p99 gap: 26–30ms), so book
  updates carry no arrival-time process; event-time structure lives in trades.
- A frame has median 25 / p99 200+ level changes; median change is 403 ticks from
  the touch. Only ~10–15% of changes land within trading-relevant distance.
- Book state is only consistent at frame boundaries (mid-frame it is transiently
  crossed ~10% of the time).

Representation: one chronological stream. Every event starts with a Δt token
(log2-bucketed µs since previous event), then either

```
trade:  [DT] [TRADE_SELL|TRADE_BUY] [price-offset] [size]
frame:  [DT] [FRAME] change+                         changes sorted by (side, offset)
change: [side+offset] [new-size]                     offset in -16..63
      | [JUMP_side] [hi] [lo] [new-size]             improvement beyond -16 ticks;
                                                     exact ticks = 17 + hi*32 + lo
```

- Offsets are ticks from the **pre-event** same-side best, so generated events can
  be applied forward without self-reference. In-band: −16..+63. Improvements beyond
  −16 use the jump escape (exact magnitude to 1,040 ticks, clamped above); changes
  deeper than +63 update the internal book during reconstruction but are not emitted.
- The jump escape exists because it binds often: 0.8% (calm day) to 3.2% (violent
  day) of frames improve the best by >16 ticks, worst observed 1,272 ticks ($127)
  in one 26ms frame — concentrated exactly in the stress periods a backtesting
  simulator needs most.
- Sizes: 32 log2 buckets over contracts; frame-size bucket 0 = level deleted.
- All buckets fixed a priori (nothing fit to data) — no leakage via the tokenizer.
- Vocab 384, ~2.6 tokens/event, ~65M tokens per market day.

Known v0.1 limitations (accepted, documented):

- Trades print against a book up to ~26ms stale (feed latency floor), so trade
  price offsets have a long right tail that the ±15/16 clamp truncates.
- Far-from-touch book depth is not modeled at all.
- Mid-file exchange snapshots (reconnects) reset the book without emitting a
  transition event; the model never sees these seams (~1–14 per day).

## Splits

Chronological by month: train = Jan–May (233M tokens), val = Jun (63M), test = Jul
(75M, untouched during development).
