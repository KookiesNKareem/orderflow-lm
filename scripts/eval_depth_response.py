"""Does the model consume book depth after an injected trade?

Injected trades enter as trade prints only; the book is not edited. Binance
depth diffs arrive on a ~26 ms timer, so in real data a marketable buy shows
up first as a trade and then as a smaller or cleared best ask in the next
frame. This script asks whether the model emits that depletion itself.

For every window and arm, the book is replayed through the context (and the
injected trade), the best quotes are recorded, and the first generated frame
is applied. On the side the trade hits (ask for buys, bid for sells) we
measure the size removed at the pre-trade best price, whether that level was
cleared, and whether the best price moved away. Effects are paired against
the no-injection control under common random numbers; the untouched side is
a placebo that should show no effect.

The real-tape reference applies the same measurement to real at-the-touch
trades whose next event is a frame, minus the same quantities for frames not
preceded by a trade (background churn), by trade-size bucket. Sizes are the
tokenizer's bucket midpoints on both sides, so the comparison is lossy in the
same way for model and data.

Usage: python scripts/eval_depth_response.py --pilot eval_out/pilot \
           --symbol SOLUSDT [--bin data/tokens_v2/test_fixed.bin]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm.sim import Book, find_regions, iter_events, size_rep  # noqa: E402
from oflm.tokenizer import SYMBOLS  # noqa: E402

ARMS = ["buy_d1", "sell_d1", "buy_d2", "sell_d2", "ctrl"]
METRICS = ("removed", "cleared", "moved")


def touch(book, side):
    p = book.best(side)
    return p, (book.levels[side].get(p, 0.0) if p is not None else 0.0)


def frame_effect(book, side, p, q, changes):
    """Apply one frame; report the change at the pre-frame best price p."""
    book.apply_frame(changes)
    after = book.levels[side].get(p, 0.0)
    nb = book.best(side)
    moved = nb is not None and (nb > p if side == 1 else nb < p)
    return {"removed": q - after, "cleared": float(after == 0.0), "moved": float(moved)}


def first_frame_after(stream, n_ctx_events):
    """Book state after n_ctx_events events, then both sides' effects of the
    first later frame plus the number of trades emitted before it."""
    book = Book()
    events = iter_events(stream)
    for k, ev in enumerate(events):
        if k >= n_ctx_events:
            break
        if ev[0] == "frame":
            book.apply_frame(ev[2])
    else:
        return None
    # `ev` is the first post-context event; walk to the first frame
    gen_trades = 0
    while ev[0] != "frame":
        gen_trades += 1
        ev = next(events, None)
        if ev is None:
            return None
    pb, qb = touch(book, 0)
    pa, qa = touch(book, 1)
    if pb is None or pa is None:
        return None
    # both sides are measured against the pre-frame quotes captured above
    ask = frame_effect(book, 1, pa, qa, ev[2])
    after_bid = book.levels[0].get(pb, 0.0)
    nb = book.best(0)
    bid = {"removed": qb - after_bid, "cleared": float(after_bid == 0.0),
           "moved": float(nb is not None and nb < pb)}
    return {"ask": ask, "bid": bid, "gen_trades": gen_trades}


def model_effects(streams, seeds):
    n = len(seeds)
    res = {a: [None] * n for a in ARMS}
    sizes = {a: np.full(n, np.nan) for a in ARMS}
    for i in range(n):
        n_seed = sum(1 for _ in iter_events(seeds[i]))
        for arm in ARMS:
            injected = arm != "ctrl"
            st = streams[arm][i]
            if injected:
                inj = list(iter_events(st[len(seeds[i]):len(seeds[i]) + 4]))
                if not inj or inj[0][0] != "trade":
                    continue
                sizes[arm][i] = size_rep(int(inj[0][4]))
            res[arm][i] = first_frame_after(st, n_seed + injected)
    return res, sizes


def real_effects(data, sym_id, seg_len, segs):
    """Real at-touch trades immediately followed by a frame, and frames
    immediately following a frame, measured on both sides."""
    regions = [(s, e) for sid, s, e in find_regions(np.asarray(data)) if sid == sym_id]
    start, end = max(regions, key=lambda r: r[1] - r[0])
    seg_len = min(seg_len, (end - start - 1000) // segs)
    trades, background = [], []
    for s0 in np.linspace(start + 1000, end - seg_len - 1, segs).astype(int):
        book, prev, warm = Book(), None, 0
        for ev in iter_events(np.asarray(data[s0:s0 + seg_len])):
            if ev[0] == "trade":
                prev = ev
                continue
            if warm < 2000:  # let the anchor quotes wash out
                book.apply_frame(ev[2])
                warm += 1
                prev = ev
                continue
            pb, qb = touch(book, 0)
            pa, qa = touch(book, 1)
            ok = pb is not None and pa is not None
            if ok and prev is not None and (prev[0] == "frame" or prev[3] == 0):
                book.apply_frame(ev[2])
                a_after, b_after = book.levels[1].get(pa, 0.0), book.levels[0].get(pb, 0.0)
                na, nb = book.best(1), book.best(0)
                rec = {
                    "ask": {"removed": qa - a_after, "cleared": float(a_after == 0.0),
                            "moved": float(na is not None and na > pa)},
                    "bid": {"removed": qb - b_after, "cleared": float(b_after == 0.0),
                            "moved": float(nb is not None and nb < pb)}}
                if prev[0] == "frame":
                    background.append(rec)
                else:
                    rec["sign"], rec["szb"] = (1 if prev[2] == 1 else -1), int(prev[4])
                    trades.append(rec)
            else:
                book.apply_frame(ev[2])
            prev = ev
    return trades, background


def boot_mean(v, n_boot=2000, seed=37):
    v = np.asarray(v, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return {"mean": None, "lo": None, "hi": None, "n": int(len(v))}
    rng = np.random.default_rng(seed)
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return {"mean": float(v.mean()), "lo": float(np.percentile(b, 2.5)),
            "hi": float(np.percentile(b, 97.5)), "n": int(len(v))}


def fmt(r):
    if r["mean"] is None:
        return f"{'n/a':>26}"
    return f"{r['mean']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", default="eval_out/pilot")
    p.add_argument("--symbol", default="SOLUSDT")
    p.add_argument("--bin", default=None, help="real test tape for the reference")
    p.add_argument("--seg-len", type=int, default=2_000_000)
    p.add_argument("--segs", type=int, default=4)
    args = p.parse_args()
    pilot = Path(args.pilot)

    z = np.load(pilot / f"streams_{args.symbol}.npz")
    streams = {a: list(z[a]) for a in ARMS}
    seeds = list(z["seeds"])
    res, sizes = model_effects(streams, seeds)
    n = len(seeds)
    out = {"symbol": args.symbol, "n_windows": n, "model": {}, "real": None}

    def col(arm, side, metric):
        return np.array([r[side][metric] if r is not None else np.nan for r in res[arm]])

    print(f"=== depth response in the first generated frame, {args.symbol} ({n} windows) ===")
    ctrl_trades = np.array([r["gen_trades"] if r else np.nan for r in res["ctrl"]])
    print(f"ctrl: generated trades before first frame, mean {np.nanmean(ctrl_trades):.2f}")
    for dose in ("d1", "d2"):
        out["model"][dose] = {}
        for arm, hit, other in ((f"buy_{dose}", "ask", "bid"), (f"sell_{dose}", "bid", "ask")):
            q = sizes[arm]
            d = {}
            print(f"\n{arm}: injected size mean {np.nanmean(q):.4g}; effect vs ctrl "
                  f"(hit side = {hit}, placebo = {other})")
            for metric in METRICS:
                eff = col(arm, hit, metric) - col("ctrl", hit, metric)
                plc = col(arm, other, metric) - col("ctrl", other, metric)
                d[metric] = {"hit": boot_mean(eff), "placebo": boot_mean(plc)}
                print(f"  {metric:<8} hit {fmt(d[metric]['hit'])}   placebo {fmt(d[metric]['placebo'])}")
            frac = (col(arm, hit, "removed") - col("ctrl", hit, "removed")) / q
            d["removed_over_size"] = boot_mean(frac)
            print(f"  removed / injected size  {fmt(d['removed_over_size'])}")
            out["model"][dose][arm] = d

    if args.bin:
        data = np.memmap(args.bin, dtype=np.uint16, mode="r")
        trades, bg = real_effects(data, SYMBOLS[args.symbol], args.seg_len, args.segs)
        base = {side: {m: float(np.mean([r[side][m] for r in bg])) for m in METRICS}
                for side in ("ask", "bid")}
        print(f"\nreal tape: {len(trades):,} at-touch trades followed by a frame, "
              f"{len(bg):,} background frames")
        real = {"background": base, "by_size": {}}
        for szb in sorted({r["szb"] for r in trades}):
            sel = [r for r in trades if r["szb"] == szb]
            if len(sel) < 50:
                continue
            row = {}
            for metric in METRICS:
                hit = [r["ask" if r["sign"] > 0 else "bid"][metric]
                       - base["ask" if r["sign"] > 0 else "bid"][metric] for r in sel]
                row[metric] = boot_mean(hit)
            row["removed_over_size"] = boot_mean(
                [(r["ask" if r["sign"] > 0 else "bid"]["removed"]
                  - base["ask" if r["sign"] > 0 else "bid"]["removed"]) / size_rep(szb)
                 for r in sel])
            real["by_size"][szb] = row
            print(f"  size bucket {szb:>2} (n={len(sel):>6}): removed {fmt(row['removed'])}  "
                  f"cleared {fmt(row['cleared'])}  removed/size {fmt(row['removed_over_size'])}")
        out["real"] = real

    dest = pilot / f"depth_{args.symbol}.json"
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
