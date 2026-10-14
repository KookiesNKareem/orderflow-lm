"""Build a large event-state split in parallel at real BOS boundaries."""

import argparse
import bisect
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from oflm import tokenizer as tk  # noqa: E402


FIELDS = (
    "dt",
    "kind",
    "trade_off",
    "trade_size",
    "change_loc",
    "change_size",
    "n_changes",
    "overflow",
    "book",
    "token_length",
)


def bos_positions(tokens: np.memmap, chunk: int = 100_000_000) -> list[int]:
    positions = []
    for start in range(0, len(tokens), chunk):
        stop = min(len(tokens), start + chunk)
        found = np.flatnonzero(np.asarray(tokens[start:stop]) == tk.BOS)
        positions.extend((found + start).tolist())
    if not positions or positions[0] != 0:
        raise RuntimeError("token split must begin with BOS")
    return positions


def choose_boundaries(length: int, positions: list[int], shards: int) -> list[int]:
    effective_shards = min(shards, len(positions))
    cuts = [0]
    for i in range(1, effective_shards):
        target = i * length // effective_shards
        j = bisect.bisect_left(positions, target)
        if j < len(positions) and positions[j] not in cuts:
            cuts.append(positions[j])
    return sorted(cuts) + [length]


def valid_shard(
    path: Path, token_start: int, token_stop: int, topk: int, max_changes: int
) -> bool:
    meta_path = path / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        meta.get("source_token_start") == token_start
        and meta.get("source_token_stop") == token_stop
        and meta.get("book_topk_per_side") == topk
        and meta.get("max_changes") == max_changes
        and meta.get("events", 0) > 0
        and all((path / f"{name}.npy").exists() for name in FIELDS)
    )


def build_one(job: tuple) -> tuple[int, str]:
    (
        index,
        tokens,
        out,
        start,
        stop,
        topk,
        max_changes,
        log,
    ) = job
    out_path = Path(out)
    if valid_shard(out_path, start, stop, topk, max_changes):
        return index, "skipped"
    out_path.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "build_event_state_data.py"),
        "--tokens",
        tokens,
        "--out",
        out,
        "--topk",
        str(topk),
        "--max-changes",
        str(max_changes),
        "--start",
        str(start),
        "--stop",
        str(stop),
    ]
    with open(log, "w") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return index, "done"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shards", type=int, default=16)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--max-changes", type=int, default=128)
    p.add_argument("--cleanup-shards", action="store_true")
    args = p.parse_args()
    if args.shards < 1 or args.workers < 1:
        raise ValueError("--shards and --workers must be positive")

    token_path = Path(args.tokens)
    tokens = np.memmap(token_path, dtype=np.uint16, mode="r")
    positions = bos_positions(tokens)
    boundaries = choose_boundaries(len(tokens), positions, args.shards)
    shard_root = Path(str(Path(args.out)) + ".shards")
    log_root = shard_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    shard_paths = []
    for i, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
        shard = shard_root / f"{i:03d}"
        shard_paths.append(shard)
        jobs.append(
            (
                i,
                str(token_path),
                str(shard),
                start,
                stop,
                args.topk,
                args.max_changes,
                str(log_root / f"{i:03d}.log"),
            )
        )
    print(
        json.dumps(
            {
                "tokens": len(tokens),
                "bos_regions": len(positions),
                "shards": len(jobs),
                "boundaries": boundaries,
            }
        ),
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = {pool.submit(build_one, job): job[0] for job in jobs}
        for completed, future in enumerate(as_completed(futures), 1):
            index, status = future.result()
            print(
                f"[{completed}/{len(jobs)}] shard {index:03d}: {status}",
                flush=True,
            )

    command = [
        sys.executable,
        str(ROOT / "scripts" / "merge_event_state_data.py"),
        *[str(path) for path in shard_paths],
        "--out",
        args.out,
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    manifest = {
        "tokens": str(token_path),
        "token_count": len(tokens),
        "bos_regions": len(positions),
        "boundaries": boundaries,
        "shards": [str(path) for path in shard_paths],
        "topk": args.topk,
        "max_changes": args.max_changes,
    }
    out = Path(args.out)
    (out / "shard_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.cleanup_shards:
        shutil.rmtree(shard_root)
        print(f"removed merged temporary shards: {shard_root}", flush=True)


if __name__ == "__main__":
    main()
