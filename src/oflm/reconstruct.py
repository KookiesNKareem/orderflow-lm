"""Reconstruct a normalized event stream from raw Tardis L2 + trades files.

Emits one chronological parquet of events per day:
  kind=0 book-frame change: new absolute size at (side, offset from pre-frame
         same-side best, in ticks), offset in [-16, 63]; size 0 = level deleted.
         Price improvements beyond the band (offset < -16) are also emitted
         (any magnitude, size > 0 only) for jump-escape tokenization.
  kind=1 trade: side is the aggressor, offset from pre-trade same-side best
         (buy vs best ask, sell vs best bid)

Offsets are relative to the book state BEFORE the event so a generative model's
output can be applied forward without self-reference. Book state is only read
at frame boundaries (mid-frame state is transiently inconsistent). Changes
outside the offset band are applied to the internal book but not emitted.

Usage: python -m oflm.reconstruct <l2.csv.gz> <trades.csv.gz> <out.parquet> [max_l2_rows]
"""

import csv
import gzip
import heapq
import json
import math
import sys

import pyarrow as pa
import pyarrow.parquet as pq

TICK = 0.1  # fallback; per-file tick is inferred from the opening snapshot
OFF_MIN, OFF_MAX = -16, 63


def infer_tick(prices) -> float:
    """Tick = GCD of price gaps in the snapshot (prices sit on the tick grid)."""
    ps = sorted(set(prices))
    g = 0
    for a, b in zip(ps, ps[1:]):
        g = math.gcd(g, round((b - a) * 1e8))
    assert g > 0, "could not infer tick size"
    return g / 1e8
BATCH_ROWS = 2_000_000

SCHEMA = pa.schema([
    ("ts", pa.int64()),          # exchange timestamp, µs
    ("kind", pa.uint8()),        # 0 = frame change, 1 = trade
    ("new_frame", pa.bool_()),   # True on first change of each frame
    ("side", pa.uint8()),        # 0 = bid/sell-side book, 1 = ask; trades: 0 = sell, 1 = buy
    ("offset", pa.int32()),      # ticks from pre-event same-side best (2021 flash
                                 # moves exceed int16: observed |offset| > 34k ticks)
    ("size", pa.float32()),      # absolute new level size, or trade amount
    ("best_bid", pa.float64()),  # pre-event best (frame rows: post-frame best), for eval
    ("best_ask", pa.float64()),
])


class Book:
    """One side of the book with lazy-heap best tracking."""

    def __init__(self, is_bid: bool):
        self.levels: dict[float, float] = {}
        self.heap: list[float] = []
        self.sign = -1.0 if is_bid else 1.0  # bids: max-heap via negation

    def set(self, price: float, amount: float) -> None:
        if amount == 0.0:
            self.levels.pop(price, None)
        else:
            if price not in self.levels:
                heapq.heappush(self.heap, self.sign * price)
            self.levels[price] = amount

    def best(self) -> float | None:
        while self.heap:
            p = self.sign * self.heap[0]
            if p in self.levels:
                return p
            heapq.heappop(self.heap)
        return None

    def clear(self) -> None:
        self.levels.clear()
        self.heap.clear()


def trade_iter(path: str):
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            # exchange,symbol,timestamp,local_timestamp,id,side,price,amount
            yield int(row[2]), row[5], float(row[6]), float(row[7])


def reconstruct(l2_path: str, trades_path: str, out_path: str, max_l2_rows: int) -> None:
    bids, asks = Book(True), Book(False)
    tick = None
    snap_prices: list[float] = []
    trades = trade_iter(trades_path)
    next_trade = next(trades, None)

    writer = pq.ParquetWriter(out_path, SCHEMA, compression="zstd")
    cols: list[list] = [[] for _ in SCHEMA]
    n_out = n_frames = n_trades = n_resnap = 0

    def flush():
        nonlocal cols
        if cols[0]:
            writer.write_batch(pa.record_batch(
                [pa.array(c, type=f.type) for c, f in zip(cols, SCHEMA)], schema=SCHEMA))
            cols = [[] for _ in SCHEMA]

    def emit(ts, kind, new_frame, side, offset, size, bb, ba):
        nonlocal n_out
        for col, v in zip(cols, (ts, kind, new_frame, side, offset, size, bb, ba)):
            col.append(v)
        n_out += 1
        if len(cols[0]) >= BATCH_ROWS:
            flush()

    def emit_trades_until(ts: int) -> None:
        """Emit trades with ts <= given frame ts, against current (pre-frame) book."""
        nonlocal next_trade, n_trades
        bb, ba = bids.best(), asks.best()
        if bb is None or ba is None:
            while next_trade and next_trade[0] <= ts:
                next_trade = next(trades, None)
            return
        while next_trade and next_trade[0] <= ts:
            tts, tside, tprice, tamount = next_trade
            if tside == "buy":
                off = round((tprice - ba) / tick)
            else:
                off = round((bb - tprice) / tick)
            emit(tts, 1, False, 1 if tside == "buy" else 0, max(-128, min(127, off)),
                 tamount, bb, ba)
            n_trades += 1
            next_trade = next(trades, None)

    def close_frame(ts: int, changed: dict) -> None:
        """Frame rows fully applied; compute offsets vs pre-frame best, emit."""
        nonlocal n_frames
        pre_bb, pre_ba, post_bb, post_ba = changed.pop("_bests")
        if pre_bb is None or pre_ba is None:
            return
        out = []
        for (side, price), amount in changed.items():
            best = pre_bb if side == 0 else pre_ba
            off = round((best - price) / tick) if side == 0 else round((price - best) / tick)
            # in-band changes, plus price improvements beyond the band (off < OFF_MIN,
            # tokenized as jump-escape events); deeper-than-band changes stay dropped
            if off <= OFF_MAX and (off >= OFF_MIN or amount > 0):
                out.append((side, off, amount))
        if not out:
            return
        out.sort(key=lambda c: (c[0], c[1]))
        n_frames += 1
        first = True
        for side, off, amount in out:
            emit(ts, 0, first, side, off, amount, post_bb or 0.0, post_ba or 0.0)
            first = False

    with gzip.open(l2_path, "rt", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        in_snapshot = False
        cur_ts = None
        changed: dict = {}
        pre_bb = pre_ba = None
        n_l2 = 0
        for row in reader:
            n_l2 += 1
            if n_l2 > max_l2_rows:
                break
            # exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount
            ts = int(row[2])
            is_snap = row[4] == "true"
            side_bid = row[5] == "bid"
            price = float(row[6])
            amount = float(row[7])

            if is_snap:
                if not in_snapshot:  # snapshot begins: drop stale state, no diff emitted
                    in_snapshot = True
                    n_resnap += 1
                    if cur_ts is not None:
                        changed["_bests"] = (pre_bb, pre_ba, bids.best(), asks.best())
                        close_frame(cur_ts, changed)
                        cur_ts, changed = None, {}
                    bids.clear()
                    asks.clear()
                (bids if side_bid else asks).set(price, amount)
                if tick is None and len(snap_prices) < 8000:
                    snap_prices.append(price)
                continue
            in_snapshot = False
            if tick is None:
                tick = infer_tick(snap_prices)
                snap_prices = []

            if ts != cur_ts:
                if cur_ts is not None:
                    changed["_bests"] = (pre_bb, pre_ba, bids.best(), asks.best())
                    close_frame(cur_ts, changed)
                emit_trades_until(ts)
                cur_ts, changed = ts, {}
                pre_bb, pre_ba = bids.best(), asks.best()

            (bids if side_bid else asks).set(price, amount)
            changed[(0 if side_bid else 1, price)] = amount

        if cur_ts is not None:
            changed["_bests"] = (pre_bb, pre_ba, bids.best(), asks.best())
            close_frame(cur_ts, changed)

    flush()
    writer.close()
    with open(out_path + ".meta.json", "w") as f:
        json.dump({"tick": tick, "l2_rows": n_l2, "frames": n_frames,
                   "trades": n_trades, "events": n_out, "snapshots": n_resnap}, f)
    print(f"l2 rows: {n_l2:,}  frames emitted: {n_frames:,}  trades: {n_trades:,}  "
          f"events out: {n_out:,}  snapshots: {n_resnap}  tick: {tick}")


if __name__ == "__main__":
    reconstruct(sys.argv[1], sys.argv[2], sys.argv[3],
                int(sys.argv[4]) if len(sys.argv) > 4 else 10**12)
