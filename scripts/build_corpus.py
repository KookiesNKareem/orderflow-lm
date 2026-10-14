"""Build the multi-year, multi-symbol token corpus."""

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

BASE = "https://datasets.tardis.dev/v1/binance-futures"
RAW = ROOT / "data" / "raw"
NORM = ROOT / "data" / "normalized"


def month_firsts(start: str, end: str):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}-01"
        m += 1
        if m > 12:
            y, m = y + 1, 1


def fetch(data_type: str, symbol: str, date: str) -> Path | None:
    yyyy, mm, dd = date.split("-")
    url = f"{BASE}/{data_type}/{yyyy}/{mm}/{dd}/{symbol}.csv.gz"
    dest = RAW / data_type / symbol / f"{date}.csv.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(".part")
    for attempt in range(8):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                if r.status_code == 404:
                    return None
                if r.status_code == 429:
                    # parallel workers share the provider's rate limit; back off as told
                    retry_after = float(r.headers.get("retry-after", 30))
                    if retry_after > 600:
                        raise RuntimeError(
                            f"provider daily quota reached; retry after {retry_after:.0f}s")
                    time.sleep(retry_after)
                    continue
                r.raise_for_status()
                size = int(r.headers.get("content-length", 0))
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if size and tmp.stat().st_size != size:
                raise IOError("size mismatch")
            tmp.rename(dest)
            return dest
        except (requests.RequestException, IOError) as e:
            if attempt == 7:
                raise
            time.sleep(min(10 * (attempt + 1), 60))
    raise RuntimeError(f"still rate limited after retries: {url}")


def process(job) -> str:
    symbol, date = job
    out = NORM / symbol / f"{date}.parquet"
    if out.exists() and Path(str(out) + ".meta.json").exists():
        return f"skip {symbol} {date}"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        l2 = fetch("incremental_book_L2", symbol, date)
        tr = fetch("trades", symbol, date)
        if l2 is None or tr is None:
            return f"missing {symbol} {date}"
        from oflm.reconstruct import reconstruct
        reconstruct(str(l2), str(tr), str(out), 10**12)
        for p in (l2, tr):
            os.unlink(p)
        return f"done {symbol} {date}"
    except Exception as e:
        return f"FAILED {symbol} {date}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    jobs = [(s, d) for s in args.symbols for d in month_firsts(args.start, args.end)]
    print(f"{len(jobs)} symbol-days", flush=True)
    t0 = time.time()
    failed = 0
    with Pool(args.workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(process, jobs), 1):
            failed += msg.startswith("FAILED")
            print(f"[{i}/{len(jobs)} {time.time() - t0:6.0f}s] {msg}", flush=True)
    print(f"corpus build complete, {failed} failures", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
