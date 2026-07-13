"""Tokenize normalized event parquets into uint16 streams.

Grammar (every event begins with a Δt token, so token class ranges make the
stream parseable without state):

  trade:  [DT] [TRADE_SELL|TRADE_BUY] [TOFF] [TSIZE]
  frame:  [DT] [FRAME] change+              # changes sorted by (side, offset)
  change: [CHG] [FSIZE]                              # offset in -16..63
        | [JUMP_BID|JUMP_ASK] [HI] [LO] [FSIZE]      # improvement beyond -16;
                                                     # exact ticks = 17 + HI*32 + LO

Δt is µs since the previous event, log2-bucketed. Offsets are ticks from the
pre-event same-side best. Sizes are log2 buckets over contracts; FSIZE bucket 0
means the level was deleted. All buckets are fixed (not fit to data), so there
is no train/test leakage via the tokenizer.

Usage: python -m oflm.tokenizer <in.parquet>... --out <out.bin>
"""

import argparse
import json

import numpy as np
import polars as pl

PAD, BOS = 0, 1
DT0 = 2          # 32 Δt buckets: 2..33
TRADE_SELL, TRADE_BUY = 34, 35
TOFF0 = 36       # 32 trade-offset buckets (-15..16): 36..67
TSIZE0 = 68      # 32 trade-size buckets: 68..99
FRAME = 100
CHG_BID0 = 101   # bid change, offset -16..63: 101..180
CHG_ASK0 = 181   # ask change, offset -16..63: 181..260
FSIZE0 = 261     # 32 level-size buckets (0 = delete): 261..292
JUMP_BID, JUMP_ASK = 293, 294
DIGIT0 = 295     # 32 base-32 digits for jump magnitude: 295..326
JUMP_MAX = 17 + 1023  # exact magnitudes 17..1040 ticks; larger jumps clamp
SYM0 = 327       # 8 symbol-id slots: 327..334, emitted once after each BOS
VOCAB = 384

SYMBOLS = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}

SIZE_REF = 0.001  # contracts; one bucket per power of 2 above this


def size_bucket(size: np.ndarray) -> np.ndarray:
    b = np.zeros(len(size), dtype=np.int64)
    nz = size > 0
    b[nz] = np.clip(np.log2(size[nz] / SIZE_REF).astype(np.int64) + 1, 1, 31)
    return b


def dt_bucket(dt_us: np.ndarray) -> np.ndarray:
    b = np.zeros(len(dt_us), dtype=np.int64)
    nz = dt_us > 0
    b[nz] = np.clip(np.log2(dt_us[nz]).astype(np.int64) + 1, 1, 31)
    return b


def tokenize_day(df: pl.DataFrame, sym_id: int | None = None) -> np.ndarray:
    ts = df["ts"].to_numpy()
    kind = df["kind"].to_numpy()
    side = df["side"].to_numpy().astype(np.int64)
    offset = df["offset"].to_numpy().astype(np.int64)
    size = df["size"].to_numpy().astype(np.float64)
    new_frame = df["new_frame"].to_numpy()

    is_trade = kind == 1
    is_start = is_trade | new_frame                    # rows that begin an event
    is_esc = (kind == 0) & (offset < -16)              # jump-escape changes
    n = len(df)

    # per-row token counts: trades 4; frame changes 2 (+2 for [DT][FRAME] header
    # on the frame's first row, +2 for jump-escape encoding)
    length = np.where(is_trade, 4, 2 + 2 * new_frame + 2 * is_esc)
    hdr = 1 if sym_id is None else 2  # [BOS] or [BOS][SYM]
    pos = np.concatenate(([hdr], hdr + np.cumsum(length)[:-1]))
    total = hdr + int(length.sum())
    out = np.full(total, PAD, dtype=np.uint16)
    out[0] = BOS
    if sym_id is not None:
        out[1] = SYM0 + sym_id

    # Δt tokens at event starts (Δt from previous event-start row)
    start_idx = np.flatnonzero(is_start)
    dts = np.diff(ts[start_idx], prepend=ts[start_idx[0]])
    out[pos[start_idx]] = DT0 + dt_bucket(dts)

    # trades
    t = np.flatnonzero(is_trade)
    out[pos[t] + 1] = np.where(side[t] == 1, TRADE_BUY, TRADE_SELL)
    out[pos[t] + 2] = TOFF0 + np.clip(offset[t], -15, 16) + 15
    out[pos[t] + 3] = TSIZE0 + size_bucket(size[t])

    # frames
    fs = np.flatnonzero(new_frame)
    out[pos[fs] + 1] = FRAME
    fc = np.flatnonzero(kind == 0)
    chg_pos = pos[fc] + np.where(new_frame[fc], 2, 0)

    reg = fc[~is_esc[fc]]
    rp = chg_pos[~is_esc[fc]]
    out[rp] = np.where(side[reg] == 0, CHG_BID0, CHG_ASK0) + offset[reg] + 16
    out[rp + 1] = FSIZE0 + size_bucket(size[reg])

    esc = fc[is_esc[fc]]
    ep = chg_pos[is_esc[fc]]
    if len(esc):
        mag = np.minimum(-offset[esc], JUMP_MAX) - 17          # 0..1023
        out[ep] = np.where(side[esc] == 0, JUMP_BID, JUMP_ASK)
        out[ep + 1] = DIGIT0 + (mag >> 5)
        out[ep + 2] = DIGIT0 + (mag & 31)
        out[ep + 3] = FSIZE0 + size_bucket(size[esc])

    assert not (out[1:] == PAD).any(), "grammar hole: unwritten position"
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from pathlib import Path
    streams = []
    for path in args.inputs:
        df = pl.read_parquet(path)
        sym_id = SYMBOLS.get(Path(path).parent.name)  # symbol from directory layout
        toks = tokenize_day(df, sym_id)
        streams.append(toks)
        print(f"{path}: {len(df):,} events -> {len(toks):,} tokens (sym {sym_id})")
    all_toks = np.concatenate(streams)
    all_toks.tofile(args.out)
    with open(args.out + ".meta.json", "w") as f:
        json.dump({"vocab": VOCAB, "dtype": "uint16", "tokens": len(all_toks),
                   "inputs": args.inputs}, f, indent=1)
    print(f"{args.out}: {len(all_toks):,} tokens total")


if __name__ == "__main__":
    main()
