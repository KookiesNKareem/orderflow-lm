"""Fail-closed validation for reset-safe normalized order-book Parquet days."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


ZERO_FIELDS = (
    "bad_reset_boundary",
    "nonmonotonic_ts",
    "crossed_or_locked",
    "nonpositive_quotes",
    "nonfinite",
    "negative_size",
    "frame_offset_gt63",
    "trade_offset_oob",
)


def validate(path_str: str) -> dict:
    os.environ.setdefault("POLARS_MAX_THREADS", "2")
    import polars as pl

    path = Path(path_str)
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    q = pl.scan_parquet(path)
    result = q.select(
        pl.len().alias("rows"),
        ((pl.col("kind") == 0) & pl.col("new_frame")).sum().alias("frames"),
        (pl.col("kind") == 1).sum().alias("trades"),
        pl.col("reset").sum().alias("resets"),
        (
            pl.col("reset")
            & ~((pl.col("kind") == 1) | pl.col("new_frame"))
        ).sum().alias("bad_reset_boundary"),
        (pl.col("ts").diff() < 0).sum().alias("nonmonotonic_ts"),
        (
            (pl.col("best_bid") > 0)
            & (pl.col("best_ask") > 0)
            & (pl.col("best_bid") >= pl.col("best_ask"))
        ).sum().alias("crossed_or_locked"),
        (
            (pl.col("best_bid") <= 0) | (pl.col("best_ask") <= 0)
        ).sum().alias("nonpositive_quotes"),
        (
            ~pl.col("size").is_finite()
            | ~pl.col("best_bid").is_finite()
            | ~pl.col("best_ask").is_finite()
        ).sum().alias("nonfinite"),
        (pl.col("size") < 0).sum().alias("negative_size"),
        (
            (pl.col("kind") == 0) & (pl.col("offset") > 63)
        ).sum().alias("frame_offset_gt63"),
        (
            (pl.col("kind") == 1)
            & ((pl.col("offset") < -128) | (pl.col("offset") > 127))
        ).sum().alias("trade_offset_oob"),
        pl.col("ts").min().alias("ts_min"),
        pl.col("ts").max().alias("ts_max"),
    ).collect(engine="streaming").to_dicts()[0]
    result.update(
        date=path.stem,
        snapshots=meta["snapshots"],
        meta_rows_match=result["rows"] == meta["events"],
        meta_frames_match=result["frames"] == meta["frames"],
        meta_trades_match=result["trades"] == meta["trades"],
        meta_resets_match=(
            result["resets"] == meta["resets"] == meta["snapshots"]
        ),
    )
    result["valid"] = (
        all(result[name] == 0 for name in ZERO_FIELDS)
        and result["meta_rows_match"]
        and result["meta_frames_match"]
        and result["meta_trades_match"]
        and result["meta_resets_match"]
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--norm", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    paths = sorted(Path(args.norm).glob("*.parquet"))
    if not paths:
        raise RuntimeError("no normalized Parquet days found")

    days = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(validate, str(path)): path for path in paths}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            days.append(result)
            print(
                f"[{i}/{len(paths)}] {result['date']}: "
                f"{'ok' if result['valid'] else 'INVALID'}",
                flush=True,
            )
    days.sort(key=lambda row: row["date"])
    summary = {
        "days": len(days),
        "first": days[0]["date"],
        "last": days[-1]["date"],
        "rows": sum(day["rows"] for day in days),
        "frames": sum(day["frames"] for day in days),
        "trades": sum(day["trades"] for day in days),
        "snapshots": sum(day["snapshots"] for day in days),
        "invalid_days": [day["date"] for day in days if not day["valid"]],
    }
    payload = {"summary": summary, "days": days}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["invalid_days"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
