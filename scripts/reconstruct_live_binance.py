"""CLI for converting Fly Binance recorder partitions to normalized events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from oflm.live_reconstruct import reconstruct_live_binance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconstruct_live_binance(args.raw, args.out), indent=2))


if __name__ == "__main__":
    main()
