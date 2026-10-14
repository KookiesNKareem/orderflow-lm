"""Reconstruct paired Tardis L2/trade days without deleting the raw archive."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm.reconstruct import reconstruct  # noqa: E402


def valid_output(path: Path) -> bool:
    meta_path = Path(str(path) + ".meta.json")
    if not path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("events", 0) > 0 and meta.get("resets") == meta.get("snapshots")


def process(job: tuple[str, str, str, str]) -> tuple[str, str, float]:
    date, l2_path, trades_path, out_path = job
    out = Path(out_path)
    if valid_output(out):
        return date, "skipped", 0.0

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.part")
    tmp_meta = Path(str(tmp) + ".meta.json")
    tmp.unlink(missing_ok=True)
    tmp_meta.unlink(missing_ok=True)
    started = time.time()
    try:
        reconstruct(l2_path, trades_path, str(tmp), 10**12)
        meta = json.loads(tmp_meta.read_text())
        if meta.get("resets") != meta.get("snapshots"):
            raise RuntimeError(
                f"snapshot/reset mismatch: {meta.get('snapshots')} != {meta.get('resets')}"
            )
        os.replace(tmp, out)
        os.replace(tmp_meta, Path(str(out) + ".meta.json"))
        return date, "done", time.time() - started
    except Exception:
        tmp.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)
        raise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="exchange raw root containing data-type dirs")
    p.add_argument("--symbol", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    raw = Path(args.raw)
    l2_dir = raw / "incremental_book_L2" / args.symbol
    trades_dir = raw / "trades" / args.symbol
    l2 = {p.stem.removesuffix(".csv"): p for p in l2_dir.glob("*.csv.gz")}
    trades = {p.stem.removesuffix(".csv"): p for p in trades_dir.glob("*.csv.gz")}
    dates = sorted(l2.keys() & trades.keys())
    missing_l2 = sorted(trades.keys() - l2.keys())
    missing_trades = sorted(l2.keys() - trades.keys())
    if missing_l2 or missing_trades:
        raise RuntimeError(
            f"unpaired raw days: missing_l2={missing_l2}, missing_trades={missing_trades}"
        )
    if not dates:
        raise RuntimeError("no paired raw days found")

    out = Path(args.out)
    jobs = [
        (date, str(l2[date]), str(trades[date]), str(out / f"{date}.parquet"))
        for date in dates
    ]
    print(
        json.dumps(
            {
                "paired_days": len(jobs),
                "first": dates[0],
                "last": dates[-1],
                "workers": args.workers,
            }
        ),
        flush=True,
    )
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, job): job[0] for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            date = futures[future]
            try:
                _, status, seconds = future.result()
                print(
                    f"[{i}/{len(jobs)}] {date}: {status} ({seconds:.1f}s)",
                    flush=True,
                )
            except Exception as exc:
                failures.append((date, str(exc)))
                print(f"[{i}/{len(jobs)}] {date}: FAILED: {exc}", flush=True)
    if failures:
        print(json.dumps({"failures": failures}, indent=2), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
