"""Compute causal statistics for paired injection runs."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm.sim import replay  # noqa: E402

ARMS = ["buy_d1", "sell_d1", "buy_d2", "sell_d2", "ctrl"]
PLATEAU = slice(15, 24)  # trade lags 16..24
L = 32
MARGINS = (2.0, 1.5)  # predeclared equivalence factors: primary, secondary


def plateau_by_arm(streams, seeds):
    """Per-seed plateau-window mean response for every arm, measured from the
    first post-context trade (the injected trade in injected arms)."""
    n = len(seeds)
    out = {a: np.full(n, np.nan) for a in ARMS}
    for i in range(n):
        inj_pos = len(replay(seeds[i], warmup_frames=0)["trade_idx"])
        for arm in ARMS:
            rep = replay(streams[arm][i], warmup_frames=0)
            idx, s = rep["trade_idx"], rep["trade_signs"]
            if inj_pos >= len(idx):
                continue
            if arm != "ctrl":
                want = 1 if arm.startswith("buy") else -1
                if s[inj_pos] != want:
                    continue
            m2 = rep["mids2"].astype(np.float64)
            base = m2[idx[inj_pos]]
            lmax = min(L, len(idx) - 1 - inj_pos)
            resp = np.full(L, np.nan)
            for l in range(1, lmax + 1):
                resp[l - 1] = (m2[idx[inj_pos + l]] - base) / 2.0
            out[arm][i] = np.nanmean(resp[PLATEAU])
    return out


def boot(vals, n_boot=2000, seed=23, pcts=(2.5, 5, 95, 97.5)):
    vals = vals[~np.isnan(vals)]
    rng = np.random.default_rng(seed)
    b = np.nanmean(vals[rng.integers(0, len(vals), (n_boot, len(vals)))], axis=1)
    return float(vals.mean()), {str(p): float(np.percentile(b, p)) for p in pcts}, b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", default="eval_out/pilot")
    p.add_argument("--symbol", default="SOLUSDT")
    p.add_argument("--robust", default=None)
    args = p.parse_args()
    pilot = Path(args.pilot)

    z = np.load(pilot / f"streams_{args.symbol}.npz")
    streams = {a: list(z[a]) for a in ARMS}
    seeds = list(z["seeds"])
    P = plateau_by_arm(streams, seeds)

    G = float(np.mean(np.array(json.load(
        open(pilot / f"propagator_{args.symbol}.json"))["G_real"])[PLATEAU]))
    out = {"symbol": args.symbol, "n_seeds": len(seeds), "kernel_plateau": G,
           "margin_factors": list(MARGINS)}

    # 1. decomposition + symmetry
    for d in ("d1", "d2"):
        bc = P[f"buy_{d}"] - P["ctrl"]
        sc = P[f"sell_{d}"] - P["ctrl"]
        for name, v in [(f"buy_minus_ctrl_{d}", bc), (f"sell_minus_ctrl_{d}", sc),
                        (f"symmetry_{d}", bc + sc)]:
            m, ci, _ = boot(v)
            out[name] = {"mean": m, "ci95": [ci["2.5"], ci["97.5"]]}
            print(f"{name}: {m:+.4f} [{ci['2.5']:+.4f}, {ci['97.5']:+.4f}]")

    # 2. dose difference (paired within seed)
    paired = {d: (P[f"buy_{d}"] - P[f"sell_{d}"]) / 2.0 for d in ("d1", "d2")}

    # 2b. temporal-block bootstrap of the plateau (seeds are in window order, so
    # contiguous blocks are temporal blocks); sensitivity to serial dependence
    def block_boot(vals, B, n_boot=2000, seed=41):
        idx = np.flatnonzero(~np.isnan(vals))
        v = vals[idx]
        nb = max(1, len(v) // B)
        blocks = [v[j * B:(j + 1) * B] for j in range(nb)]
        rng = np.random.default_rng(seed)
        draws = [np.concatenate([blocks[j] for j in rng.integers(0, nb, nb)]).mean()
                 for _ in range(n_boot)]
        return np.percentile(draws, [5, 95])
    out["block_bootstrap"] = {}
    for d in ("d1", "d2"):
        for B in (16, 64):
            lo, hi = block_boot(paired[d], B)
            out["block_bootstrap"][f"{d}_B{B}"] = [float(lo), float(hi)]
            print(f"block_boot {d} B={B}: 90% CI [{lo:+.4f}, {hi:+.4f}]")
    m, ci, _ = boot(paired["d2"] - paired["d1"])
    out["dose_diff"] = {"mean": m, "ci95": [ci["2.5"], ci["97.5"]]}
    print(f"dose_diff (d2-d1): {m:+.4f} [{ci['2.5']:+.4f}, {ci['97.5']:+.4f}]")

    # 3. TOST equivalence vs kernel at each predeclared margin
    robust = None
    if args.robust:
        rp = Path(args.robust) / f"robust_{args.symbol}.json"
        if rp.exists():
            robust = np.array(json.load(open(rp))["boot_samples"])
            # recentre the kernel draws on the headline propagator-plateau kernel
            # so numerator and denominator reference the same estimator
            robust = robust - robust.mean() + G
    for d in ("d1", "d2"):
        m, ci, b = boot(paired[d])
        entry = {"mean": m, "ci90": [ci["5"], ci["95"]], "margins": {}}
        for MARGIN in MARGINS:
            lo_eq, hi_eq = G / MARGIN, G * MARGIN
            eq = lo_eq <= ci["5"] and ci["95"] <= hi_eq
            me = {"region": [lo_eq, hi_eq], "equivalent": bool(eq)}
            if robust is not None:
                # ratio bootstrap: response draws over kernel draws; equivalence
                # at margin M requires the 90% ratio CI inside [1/M, M]
                k = np.random.default_rng(29).choice(robust, size=len(b))
                rlo, rhi = np.percentile(b / k, [5, 95])
                me["ratio_ci90_kernel_prop"] = [float(rlo), float(rhi)]
                me["equivalent_kernel_prop"] = bool(
                    1.0 / MARGIN <= rlo and rhi <= MARGIN)
            entry["margins"][f"x{MARGIN:g}"] = me
            print(f"tost_{d} x{MARGIN:g}: mean {m:+.4f}, "
                  f"90% CI [{ci['5']:+.4f}, {ci['95']:+.4f}], "
                  f"region [{lo_eq:+.4f}, {hi_eq:+.4f}] -> "
                  f"{'EQUIVALENT' if eq else 'not established'}")
        out[f"tost_{d}"] = entry

    with open(pilot / f"causal_stats_{args.symbol}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {pilot / f'causal_stats_{args.symbol}.json'}")


if __name__ == "__main__":
    main()
