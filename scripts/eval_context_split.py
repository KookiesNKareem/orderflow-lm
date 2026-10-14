"""Split the injected response by the order flow in the context it lands in.

Hypothesis under test: the model reacts at the scale of the bare kernel G
rather than the dressed response R because a spliced trade arrives without
the correlated (metaorder) flow that dresses real trades. A model that has
learned conditional expectations should then respond more to an injection
that continues the recent signed flow (it looks like the next child of a
metaorder) than to one against it, and real trades should show the same
ordering in their dressed response.

For each window, imbalance = mean sign of the last K context trades. Each
injected arm's plateau response is signed in the trade's direction and taken
relative to the no-injection control on a common base: the last context mid
(replay indexes every injected trade to it) and post-context trade lags.
"With flow" is the arm whose side matches sign(imbalance). The paired
contrast (R_buy - R_sell)/2 is reported per bin as well.

The real-tape reference applies the same split to trades on the test tape:
plateau-window dressed response signed by the trade, grouped by whether the
trade agrees with the preceding K-trade imbalance, using the model's |imb|
bin edges.

Usage: python scripts/eval_context_split.py --pilot eval_out/pilot \
           --symbol SOLUSDT [--bin data/tokens_v2/test_fixed.bin]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm.sim import find_regions, replay  # noqa: E402
from oflm.tokenizer import SYMBOLS  # noqa: E402

ARMS = ["buy_d1", "sell_d1", "buy_d2", "sell_d2", "ctrl"]
PLATEAU = (16, 24)   # trade lags, inclusive; matches eval_causal_stats


def plateau_response(stream, seed_rep, arm):
    """Mean plateau-window move from the last context mid, in ticks.
    Lag l is the l-th trade after the injection point in every arm: the
    injected trade is lag 0 in injected arms; the control has no lag-0 trade.
    NaN if the rollout is too short or the injected trade is not in place."""
    rep = replay(stream, warmup_frames=0)
    idx, s = rep["trade_idx"], rep["trade_signs"]
    m2 = rep["mids2"].astype(np.float64)
    k0, t0 = len(seed_rep["trade_idx"]), seed_rep["frames"]
    if t0 < 1 or len(m2) < t0:
        return np.nan
    if arm == "ctrl":
        first = k0 - 1
    else:
        if k0 >= len(idx) or s[k0] != (1 if arm.startswith("buy") else -1):
            return np.nan
        first = k0
    lo, hi = PLATEAU
    if first + hi >= len(idx):
        return np.nan
    return float(np.mean((m2[idx[first + lo:first + hi + 1]] - m2[t0 - 1]) / 2.0))


def model_split(streams, seeds, K):
    n = len(seeds)
    imb = np.full(n, np.nan)
    resp = {a: np.full(n, np.nan) for a in ARMS}
    for i in range(n):
        srep = replay(seeds[i], warmup_frames=0)
        signs = srep["trade_signs"]
        if len(signs) >= K:
            imb[i] = signs[-K:].mean()
        for arm in ARMS:
            resp[arm][i] = plateau_response(streams[arm][i], srep, arm)
    return imb, resp


def real_split(data, sym_id, K, seg_len, segs):
    """Per real trade: preceding K-trade imbalance, trade sign, and signed
    plateau-window dressed response."""
    regions = [(s, e) for sid, s, e in find_regions(np.asarray(data)) if sid == sym_id]
    start, end = max(regions, key=lambda r: r[1] - r[0])
    seg_len = min(seg_len, (end - start - 1000) // segs)
    lo, hi = PLATEAU
    imbs, signs, resps = [], [], []
    for s0 in np.linspace(start + 1000, end - seg_len - 1, segs).astype(int):
        rep = replay(np.asarray(data[s0:s0 + seg_len]))
        idx, s = rep["trade_idx"], rep["trade_signs"].astype(np.float64)
        m2 = rep["mids2"].astype(np.float64)
        if len(idx) < K + hi + 2:
            continue
        csum = np.concatenate([[0.0], np.cumsum(s)])
        j = np.arange(K, len(idx) - hi)
        fut = np.mean([m2[idx[j + l]] for l in range(lo, hi + 1)], axis=0)
        imbs.append((csum[j] - csum[j - K]) / K)
        signs.append(s[j])
        resps.append(s[j] * (fut - m2[idx[j]]) / 2.0)
    return np.concatenate(imbs), np.concatenate(signs), np.concatenate(resps)


def boot_mean(v, n_boot=2000, seed=31):
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
        return f"{'n/a':>24}"
    return f"{r['mean']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", default="eval_out/pilot")
    p.add_argument("--symbol", default="SOLUSDT")
    p.add_argument("--bin", default=None, help="real test tape for the reference split")
    p.add_argument("--K", type=int, default=32, help="context trades in the imbalance")
    p.add_argument("--bins", type=int, default=3, help="|imbalance| quantile bins")
    p.add_argument("--seg-len", type=int, default=2_000_000)
    p.add_argument("--segs", type=int, default=4)
    args = p.parse_args()
    pilot = Path(args.pilot)

    z = np.load(pilot / f"streams_{args.symbol}.npz")
    streams = {a: list(z[a]) for a in ARMS}
    seeds = list(z["seeds"])
    imb, resp = model_split(streams, seeds, args.K)

    ok = ~np.isnan(imb) & (imb != 0)
    edges = np.quantile(np.abs(imb[ok]), np.linspace(0, 1, args.bins + 1))
    edges[-1] += 1e-9
    flow = np.sign(imb)
    out = {"symbol": args.symbol, "K": args.K, "plateau": PLATEAU,
           "bin_edges": edges.tolist(), "n_windows": len(seeds),
           "n_usable": int(ok.sum()), "model": {}, "real": None}

    print(f"=== context split, {args.symbol}: {ok.sum()}/{len(seeds)} windows, "
          f"K={args.K}, plateau trade lags {PLATEAU[0]}-{PLATEAU[1]} ===")
    for dose in ("d1", "d2"):
        b, s, c = resp[f"buy_{dose}"], resp[f"sell_{dose}"], resp["ctrl"]
        eff_buy, eff_sell = b - c, -(s - c)        # each signed in its trade's direction
        with_ = np.where(flow > 0, eff_buy, eff_sell)
        against = np.where(flow > 0, eff_sell, eff_buy)
        paired = (b - s) / 2.0
        rows = []
        print(f"\ndose {dose}   |imb| bin        with-flow                 against-flow"
              "              with - against            paired (buy-sell)/2")
        for k in range(args.bins):
            m = ok & (np.abs(imb) >= edges[k]) & (np.abs(imb) < edges[k + 1])
            row = {"bin": [float(edges[k]), float(edges[k + 1])],
                   "with": boot_mean(with_[m]), "against": boot_mean(against[m]),
                   "with_minus_against": boot_mean(with_[m] - against[m]),
                   "paired": boot_mean(paired[m])}
            rows.append(row)
            print(f"  [{edges[k]:.2f},{edges[k + 1]:.2f})  {fmt(row['with'])}  "
                  f"{fmt(row['against'])}  {fmt(row['with_minus_against'])}  "
                  f"{fmt(row['paired'])}")
        out["model"][dose] = rows
    out["ctrl_drift_with_flow"] = boot_mean(flow[ok] * resp["ctrl"][ok])
    print(f"\nctrl drift signed by context flow: {fmt(out['ctrl_drift_with_flow'])}")

    if args.bin:
        data = np.memmap(args.bin, dtype=np.uint16, mode="r")
        rimb, rsign, rresp = real_split(data, SYMBOLS[args.symbol], args.K,
                                        args.seg_len, args.segs)
        aligned = np.sign(rimb) == rsign
        rows = []
        print(f"\nreal tape: {len(rresp):,} trades   |imb| bin        with-flow"
              "                 against-flow")
        for k in range(args.bins):
            m = (rimb != 0) & (np.abs(rimb) >= edges[k]) & (np.abs(rimb) < edges[k + 1])
            row = {"bin": [float(edges[k]), float(edges[k + 1])],
                   "with": boot_mean(rresp[m & aligned]),
                   "against": boot_mean(rresp[m & ~aligned])}
            rows.append(row)
            print(f"  [{edges[k]:.2f},{edges[k + 1]:.2f})  {fmt(row['with'])}  "
                  f"{fmt(row['against'])}")
        out["real"] = rows

    dest = pilot / f"context_split_{args.symbol}.json"
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
