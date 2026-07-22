"""Summarize raw incremental-book and trade feeds."""

import csv
import gzip
import sys
from collections import Counter

TICK = 0.1  # BTCUSDT perp price increment


def pct(sorted_vals, q):
    return sorted_vals[min(int(q * len(sorted_vals)), len(sorted_vals) - 1)]


def main(path: str, max_rows: int) -> None:
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    n = n_snap = n_del = n_ins = 0
    batch_sizes = Counter()
    gaps_us = []
    dist_ticks = []
    spreads = []
    cur_ts = None
    cur_batch = 0

    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            n += 1
            if n > max_rows:
                break
            _, _, ts, _, is_snap, side, price, amount = row
            ts = int(ts)
            price = float(price)
            amount = float(amount)
            book = bids if side == "bid" else asks

            if is_snap == "true":
                n_snap += 1
                book[price] = amount
                continue

            if ts != cur_ts:
                if cur_ts is not None:
                    batch_sizes[min(cur_batch, 200)] += 1
                    if ts > cur_ts:
                        gaps_us.append(ts - cur_ts)
                cur_ts, cur_batch = ts, 0
            cur_batch += 1

            # max/min over the full book is O(levels) — only pay it on sampled rows
            if n % 97 == 0 and bids and asks:
                best_bid, best_ask = max(bids), min(asks)
                spreads.append(round((best_ask - best_bid) / TICK))
                best = best_bid if side == "bid" else best_ask
                d = (best - price) if side == "bid" else (price - best)
                dist_ticks.append(round(d / TICK))

            if amount == 0:
                n_del += 1
                book.pop(price, None)
            else:
                if price not in book:
                    n_ins += 1
                book[price] = amount

    total_updates = n - n_snap - 1
    print(f"rows processed: {n:,} (snapshot rows: {n_snap:,})")
    print(f"deletes: {n_del / total_updates:.1%}  inserts(new level): {n_ins / total_updates:.1%}  "
          f"amends: {1 - (n_del + n_ins) / total_updates:.1%}")
    print(f"book levels now: {len(bids):,} bids / {len(asks):,} asks")

    gaps_us.sort()
    print("\ninter-batch gap (exchange ts, µs): "
          + "  ".join(f"p{int(q * 100)}={pct(gaps_us, q):,}" for q in (0.1, 0.5, 0.9, 0.99, 0.999)))

    sizes = sorted(batch_sizes.elements())
    print("batch size (updates per exchange ts): "
          + "  ".join(f"p{int(q * 100)}={pct(sizes, q)}" for q in (0.5, 0.9, 0.99)) + "  (capped at 200)")

    dist_ticks.sort()
    print("update distance from same-side best (ticks): "
          + "  ".join(f"p{int(q * 100)}={pct(dist_ticks, q):,}" for q in (0.1, 0.5, 0.9, 0.99, 0.999)))
    neg = sum(1 for d in dist_ticks if d < 0)
    print(f"  inside-spread/crossing share of sampled updates: {neg / len(dist_ticks):.2%}")

    spreads.sort()
    print("spread (ticks): "
          + "  ".join(f"p{int(q * 100)}={pct(spreads, q)}" for q in (0.1, 0.5, 0.9, 0.99)))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20_000_000)
