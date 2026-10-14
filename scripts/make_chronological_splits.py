"""Tokenize normalized days into leak-free whole-day chronological splits."""

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm.tokenizer import SYMBOLS, VOCAB, tokenize_day  # noqa: E402


def tokenize_one(job: tuple[str, str, int]) -> tuple[str, str, int]:
    path, tmp, sym_id = job
    import polars as pl

    df = pl.read_parquet(path)
    if "reset" not in df.columns:
        raise RuntimeError(f"{path} predates reset-safe normalization")
    tokens = tokenize_day(df, sym_id)
    Path(tmp).parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(tmp)
    return path, tmp, len(tokens)


def append_split(name: str, rows: list[dict], out: Path) -> dict:
    dest = out / f"{name}.bin"
    tmp_dest = dest.with_suffix(".bin.part")
    total = 0
    with open(tmp_dest, "wb") as target:
        for row in rows:
            with open(row["tmp"], "rb") as source:
                shutil.copyfileobj(source, target, length=16 << 20)
            total += row["tokens"]
    os.replace(tmp_dest, dest)
    meta = {
        "vocab": VOCAB,
        "dtype": "uint16",
        "tokens": total,
        "days": len(rows),
        "dates": [row["date"] for row in rows],
        "inputs": [row["path"] for row in rows],
        "boundary_policy": "whole normalized days; source reset regions preserved as BOS",
    }
    Path(str(dest) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--norm", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-end", required=True, help="inclusive YYYY-MM-DD")
    p.add_argument("--val-end", required=True, help="inclusive YYYY-MM-DD")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.train_end >= args.val_end:
        raise ValueError("--train-end must be earlier than --val-end")
    if args.symbol not in SYMBOLS:
        raise ValueError(f"tokenizer has no symbol id for {args.symbol}")

    norm = Path(args.norm)
    files = sorted(norm.glob("*.parquet"))
    if not files:
        raise RuntimeError("no normalized Parquet days found")
    out = Path(args.out)
    tmp_dir = out / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in files:
        date = path.stem
        split = (
            "train"
            if date <= args.train_end
            else "val"
            if date <= args.val_end
            else "test"
        )
        rows.append(
            {
                "date": date,
                "split": split,
                "path": str(path),
                "tmp": str(tmp_dir / f"{date}.bin"),
            }
        )
    for name in ("train", "val", "test"):
        if not any(row["split"] == name for row in rows):
            raise RuntimeError(f"empty {name} split")

    jobs = [(row["path"], row["tmp"], SYMBOLS[args.symbol]) for row in rows]
    by_path = {row["path"]: row for row in rows}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(tokenize_one, job): job[0] for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            path, _, tokens = future.result()
            by_path[path]["tokens"] = tokens
            print(
                f"[{i}/{len(jobs)}] {Path(path).stem}: {tokens:,} tokens",
                flush=True,
            )

    summary = {
        name: append_split(
            name,
            [row for row in rows if row["split"] == name],
            out,
        )
        for name in ("train", "val", "test")
    }
    for row in rows:
        Path(row["tmp"]).unlink()
    tmp_dir.rmdir()
    (out / "splits.json").write_text(
        json.dumps(
            {
                "symbol": args.symbol,
                "train_end": args.train_end,
                "val_end": args.val_end,
                "splits": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
