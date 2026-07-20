"""Size-conditioned response R(l | size bucket), real vs model, identical
estimator. Tests whether the model reacts to LARGE trades the way the real
market does, or under-reacts (the out-of-distribution hypothesis behind the
too-flat PnL-gap curve).

For each trade of size bucket b, R(l|b) = E[sign * (mid_{i+l} - mid_i)] / 2 in
ticks. Computed the same way on:
  real  - SOL segments from val.bin
  gen   - the pilot's control-arm rollouts (endogenous, unconstrained generation)
Herding contamination is present in both and cancels in the comparison (SOL sign
ACF already matches real). We report R at a fixed lag vs size, the log-log
scaling exponent for each, and overlay the CAUSAL injection impact (pnl_gap /
pilot) at the same sizes — if causal << conditional, that is direct evidence the
model under-reacts to off-policy large trades specifically.

Usage:
  python3 scripts/eval_size_impact.py --symbol SOLUSDT --buckets 6,9,12,15
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm.sim import find_regions, replay  # noqa: E402
from oflm.tokenizer import SYMBOLS  # noqa: E402

LAGS = [1, 2, 5, 10, 20, 50]
REF_LAG = 20


def cond_impact(reps, buckets, half=1):
    """R(l|b) pooled over reps, for each bucket b (± half buckets wide).
    Returns {b: {"R": [per-lag], "n": count}}."""
    out = {}
    for b in buckets:
        num = np.zeros(len(LAGS)); n = np.zeros(len(LAGS))
        raw = {l: [] for l in LAGS}  # per-trade values for bootstrap at REF_LAG
        for r in reps:
            idx, s, sz, m = (r["trade_idx"], r["trade_signs"],
                             r["trade_sizes"], r["mids2"].astype(np.float64))
            sel = np.abs(sz - b) <= half
            ii, ss = idx[sel], s[sel]
            for j, l in enumerate(LAGS):
                ok = ii + l < len(m)
                if ok.any():
                    v = ss[ok] * (m[ii[ok] + l] - m[ii[ok]]) / 2.0
                    num[j] += v.sum(); n[j] += len(v)
                    if l == REF_LAG:
                        raw[l].extend(v.tolist())
        out[b] = {"R": (num / np.maximum(n, 1)).tolist(), "n": int(n[LAGS.index(REF_LAG)]),
                  "raw_ref": raw[REF_LAG]}
    return out


def boot_mean(vals, n_boot=2000, seed=17):
    vals = np.array(vals)
    if len(vals) < 3:
        return [float(np.mean(vals)) if len(vals) else float("nan")] * 3
    rng = np.random.default_rng(seed)
    b = np.mean(vals[rng.integers(0, len(vals), (n_boot, len(vals)))], axis=1)
    return [float(vals.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def loglog_slope(buckets, R_ref):
    x = np.array(buckets, float) * np.log(2)  # bucket ~ log2(size), so size ~ 2^b
    y = np.array(R_ref, float)
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 2:
        return float("nan")
    # d log R / d log size ; log size = bucket*ln2 (up to const)
    return float(np.polyfit(x[ok], np.log(y[ok]), 1)[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", default="data/tokens_v2/val.bin")
    p.add_argument("--symbol", default="SOLUSDT")
    p.add_argument("--pilot", default="eval_out/pilot")
    p.add_argument("--buckets", default="6,9,12,15")
    p.add_argument("--seg-len", type=int, default=2_000_000)
    p.add_argument("--segs", type=int, default=6)
    p.add_argument("--out", default="eval_out/pnl")
    args = p.parse_args()
    buckets = [int(x) for x in args.buckets.split(",")]

    # real
    data = np.memmap(args.bin, dtype=np.uint16, mode="r")
    sym_id = SYMBOLS[args.symbol]
    regions = [(s, e) for sid, s, e in find_regions(np.asarray(data)) if sid == sym_id]
    start, end = max(regions, key=lambda r: r[1] - r[0])
    seg_starts = np.linspace(start + 1000, end - args.seg_len - 1, args.segs).astype(int)
    real = [replay(np.asarray(data[s:s + args.seg_len])) for s in seg_starts]
    print(f"real: {sum(len(r['trade_idx']) for r in real):,} trades", flush=True)
    R_real = cond_impact(real, buckets)

    # generated endogenous: pilot control-arm rollouts
    z = np.load(Path(args.pilot) / f"streams_{args.symbol}.npz")
    ctrl = [replay(np.asarray(s, dtype=np.uint16), warmup_frames=0) for s in z["ctrl"]]
    print(f"gen:  {sum(len(r['trade_idx']) for r in ctrl):,} trades", flush=True)
    R_gen = cond_impact(ctrl, buckets)

    # causal injection impact at these sizes, if available
    causal = {}
    pnl = Path(args.out) / f"pnlgap_{args.symbol}.json"
    if pnl.exists():
        pj = json.load(open(pnl))
        causal = {int(d): pj["impact_reactive"][str(d)][0] for d in pj["doses"]}

    ri = R_real[buckets[0]]["R"]  # for header sizing only
    print(f"\n=== size-conditioned response R(l={REF_LAG}), {args.symbol} ===")
    print("bucket | real R [CI]  n_real | gen R [CI]  n_gen | causal inj")
    result = {"symbol": args.symbol, "buckets": buckets, "lags": LAGS,
              "ref_lag": REF_LAG, "real": {}, "gen": {}, "causal": causal}
    for b in buckets:
        rr = boot_mean(R_real[b]["raw_ref"]); rg = boot_mean(R_gen[b]["raw_ref"])
        result["real"][b] = {"R_lags": R_real[b]["R"], "ref": rr, "n": R_real[b]["n"]}
        result["gen"][b] = {"R_lags": R_gen[b]["R"], "ref": rg, "n": R_gen[b]["n"]}
        c = causal.get(b)
        print(f"  {b:>4} | {rr[0]:+6.3f} [{rr[1]:+.2f},{rr[2]:+.2f}] {R_real[b]['n']:>6} | "
              f"{rg[0]:+6.3f} [{rg[1]:+.2f},{rg[2]:+.2f}] {R_gen[b]['n']:>6} | "
              f"{('%+.3f' % c) if c is not None else '   -'}")

    s_real = loglog_slope(buckets, [result["real"][b]["ref"][0] for b in buckets])
    s_gen = loglog_slope(buckets, [result["gen"][b]["ref"][0] for b in buckets])
    result["slope_real"], result["slope_gen"] = s_real, s_gen
    print(f"\nlog-log size-scaling exponent (R ~ size^beta):")
    print(f"  real {s_real:+.3f}   gen {s_gen:+.3f}   (sqrt law = 0.5)")
    print("  gen << real  => model under-reacts to large trades (OOD artifact)")
    print("  gen ~= real  => model's size-scaling is faithful")

    with open(Path(args.out) / f"sizeimpact_{args.symbol}.json", "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {args.out}/sizeimpact_{args.symbol}.json")


if __name__ == "__main__":
    main()
