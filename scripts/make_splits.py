"""Tokenize the corpus into walk-forward train, validation, and test splits."""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm.tokenizer import SYMBOLS, VOCAB, tokenize_day  # noqa: E402


def tokenize_one(job):
    path, tmp = job
    import polars as pl
    df = pl.read_parquet(path)
    sym_id = SYMBOLS.get(Path(path).parent.name)
    toks = tokenize_day(df, sym_id)
    toks.tofile(tmp)
    return tmp, len(toks)


def append_all(tmps, dest, half=None):
    """Ordered append of temp bins into dest. half='first'/'second' takes that
    half of each file (split at an even token index)."""
    total = 0
    with open(dest, "wb") as out:
        for tmp in tmps:
            arr = np.fromfile(tmp, dtype=np.uint16)
            if half == "first":
                arr = arr[:len(arr) // 2]
            elif half == "second":
                arr = arr[len(arr) // 2:]
            arr.tofile(out)
            total += len(arr)
    with open(str(dest) + ".meta.json", "w") as f:
        json.dump({"vocab": VOCAB, "dtype": "uint16", "tokens": total,
                   "files": len(tmps), "half": half}, f)
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--norm", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--holdout", required=True)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    norm, out = Path(args.norm), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmpdir = out / "tmp"
    tmpdir.mkdir(exist_ok=True)

    train_files, holdout_files = [], []
    for sym_dir in sorted(norm.iterdir()):
        if not sym_dir.is_dir():
            continue
        for pq in sorted(sym_dir.glob("*.parquet")):
            date = pq.stem
            if date <= args.train_end:
                train_files.append(pq)
            elif date == args.holdout:
                holdout_files.append(pq)
    print(f"train days: {len(train_files)}  holdout days: {len(holdout_files)}", flush=True)

    jobs = [(str(f), str(tmpdir / f"{f.parent.name}_{f.stem}.bin"))
            for f in train_files + holdout_files]
    with Pool(args.workers) as pool:
        for i, (tmp, n) in enumerate(pool.imap(tokenize_one, jobs), 1):
            print(f"[{i}/{len(jobs)}] {Path(tmp).name}: {n:,} tokens", flush=True)

    train_tmps = [str(tmpdir / f"{f.parent.name}_{f.stem}.bin") for f in train_files]
    hold_tmps = [str(tmpdir / f"{f.parent.name}_{f.stem}.bin") for f in holdout_files]
    n_train = append_all(train_tmps, out / "train.bin")
    n_val = append_all(hold_tmps, out / "val.bin", half="first")
    n_test = append_all(hold_tmps, out / "test.bin", half="second")
    print(f"train {n_train:,}  val {n_val:,}  test {n_test:,} tokens", flush=True)
    for t in tmpdir.iterdir():
        t.unlink()
    tmpdir.rmdir()


if __name__ == "__main__":
    main()
