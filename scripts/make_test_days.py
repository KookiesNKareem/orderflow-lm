"""Tokenize held-out days into one token file per symbol-day.

Multi-day evaluation runs each injection experiment per day, so test days are
kept as separate files (<out>/<SYMBOL>_<date>.bin) instead of being appended
into a single split. Each file starts with a [BOS][SYM] header, which is what
find_regions and pilot_inject.py expect.

Usage: python scripts/make_test_days.py --norm data/normalized \
           --out data/test_days --dates 2026-08-01 2026-09-01 2026-10-01
"""

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm.tokenizer import SYMBOLS, VOCAB, tokenize_day  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--norm", required=True, help="normalized root with <SYMBOL>/<date>.parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--dates", nargs="+", required=True)
    p.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for sym in args.symbols:
        for date in args.dates:
            src = Path(args.norm) / sym / f"{date}.parquet"
            if not src.exists():
                print(f"missing {src}", flush=True)
                continue
            toks = tokenize_day(pl.read_parquet(src), SYMBOLS[sym])
            dest = out / f"{sym}_{date}.bin"
            toks.tofile(dest)
            with open(str(dest) + ".meta.json", "w") as f:
                json.dump({"vocab": VOCAB, "dtype": "uint16", "tokens": int(len(toks)),
                           "symbol": sym, "date": date, "source": str(src)}, f)
            print(f"{dest}: {len(toks):,} tokens", flush=True)


if __name__ == "__main__":
    main()
