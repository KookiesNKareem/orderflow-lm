"""Download free first-of-month market data from Tardis.dev."""

import argparse
import sys
import time
from pathlib import Path

import requests

BASE = "https://datasets.tardis.dev/v1"
ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
CHUNK = 1 << 20


def month_firsts(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    dates = []
    while (y, m) <= (ey, em):
        dates.append(f"{y:04d}-{m:02d}-01")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return dates


def download(exchange: str, data_type: str, symbol: str, date: str) -> str:
    yyyy, mm, dd = date.split("-")
    url = f"{BASE}/{exchange}/{data_type}/{yyyy}/{mm}/{dd}/{symbol}.csv.gz"
    dest = ROOT / exchange / data_type / symbol / f"{date}.csv.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # dest only exists post-rename, i.e. after a size-verified download
    if dest.exists():
        return "skipped"

    tmp = dest.with_suffix(".part")
    for attempt in range(6):
        try:
            # the datasets endpoint 404s on HEAD, so existence is probed via GET
            with requests.get(url, stream=True, timeout=60) as r:
                if r.status_code == 404:
                    return "missing"
                if r.status_code == 429:
                    retry_after = float(r.headers.get("retry-after", 30))
                    raise requests.RequestException(
                        f"rate limited; retry after {retry_after:.0f}s",
                        response=r,
                    )
                r.raise_for_status()
                remote_size = int(r.headers.get("content-length", 0))
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
            if remote_size and tmp.stat().st_size != remote_size:
                raise IOError(f"size mismatch: {tmp.stat().st_size} != {remote_size}")
            tmp.rename(dest)
            return f"downloaded {remote_size / 1e6:.0f}MB"
        except (requests.RequestException, IOError) as e:
            if attempt == 5:
                raise
            retry_after = 0.0
            if isinstance(e, requests.RequestException) and e.response is not None:
                retry_after = float(e.response.headers.get("retry-after", 0))
            if retry_after > 600:
                raise RuntimeError(
                    f"provider daily quota reached; retry after {retry_after:.0f}s"
                ) from e
            delay = min(max(retry_after, 10 * (2**attempt)), 60)
            print(
                f"  retry {attempt + 1} in {delay:.0f}s after error: {e}",
                flush=True,
            )
            time.sleep(delay)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exchange", required=True)
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--types", nargs="+", required=True)
    p.add_argument("--start", required=True, help="YYYY-MM")
    p.add_argument("--end", required=True, help="YYYY-MM")
    args = p.parse_args()

    jobs = [
        (args.exchange, t, s, d)
        for t in args.types
        for s in args.symbols
        for d in month_firsts(args.start, args.end)
    ]
    print(f"{len(jobs)} files", flush=True)
    failures = 0
    for i, job in enumerate(jobs, 1):
        t0 = time.time()
        try:
            status = download(*job)
        except Exception as e:
            status = f"FAILED: {e}"
            failures += 1
        print(f"[{i}/{len(jobs)}] {'/'.join(job)}: {status} ({time.time() - t0:.0f}s)", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
