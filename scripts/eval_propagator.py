"""Bare-impact kernel G(l) via propagator deconvolution, vs injected response.

The measured response function R(l) = E[eps_i * (m_{i+l} - m_i)] / 2 conflates
the causal impact of trade i with the impact of the correlated trades around it
(sign ACF C(l)). Under the propagator model (impacts add linearly),

    m_i = sum_{j < i} G(i - j) * eps_j + noise
    R(l) = sum_{m >= 1} [C(|m - l|) - C(m)] * G(m)

so with R and C measured on the same trade-time grid, G is the solution of a
linear system. G(l) is the real-data estimate of what the injection pilot
measures by intervention: the price path caused by one uncorrelated trade.

Everything here runs in TRADE time (lag = trades, not frames), so the injected
paired response is recomputed on that clock for a like-for-like comparison.
Also measured: induced flow C_inj(l) — the net sign of the l-th generated trade
after a buy vs sell injection — i.e. how much herd the model reads into one
exogenous trade.

Usage:
  python3 scripts/eval_propagator.py --bin data/tokens_v2/val.bin --symbol SOLUSDT
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
L_REAL = 64          # trade lags for R/C/G on the real tape
L_INJ = 32           # trade lags measurable in 2500-token rollouts
SHOW = [1, 2, 3, 5, 8, 12, 16, 24, 32]


def trade_response(reps, L):
    """Pooled R(l) and C(l) on the trade clock across replay dicts."""
    R_num = np.zeros(L + 1)
    C_num = np.zeros(L + 1)
    n = np.zeros(L + 1)
    for rep in reps:
        m2 = rep["mids2"].astype(np.float64)
        idx, s = rep["trade_idx"], rep["trade_signs"].astype(np.float64)
        k = len(idx)
        for l in range(1, min(L, k - 1) + 1):
            d = (m2[idx[l:]] - m2[idx[:k - l]]) / 2.0
            R_num[l] += (s[:k - l] * d).sum()
            C_num[l] += (s[:k - l] * s[l:]).sum()
            n[l] += k - l
    with np.errstate(invalid="ignore"):
        return R_num / n, C_num / n, n


def deconvolve(R, C, L):
    """Solve R(l) = sum_m [C(|m-l|) - C(m)] G(m), m,l = 1..L (least squares).
    C(0) = 1 by definition. Truncation: G(m > L) ignored (C has decayed)."""
    Cfull = np.concatenate([[1.0], C[1:2 * L]]) if len(C) >= 2 * L else \
        np.concatenate([[1.0], C[1:], np.zeros(2 * L - len(C))])
    A = np.empty((L, L))
    for l in range(1, L + 1):
        for m in range(1, L + 1):
            A[l - 1, m - 1] = Cfull[abs(m - l)] - Cfull[m]
    G, *_ = np.linalg.lstsq(A, R[1:L + 1], rcond=None)
    recon = A @ G
    return G, float(np.sqrt(np.mean((recon - R[1:L + 1]) ** 2)))


def injection_trade_time(streams, seeds, L):
    """Per-seed injected response and follow-on signs on the trade clock.
    Injected trade = trade number n_seed_trades in each injected stream."""
    n = len(seeds)
    resp = {a: np.full((n, L), np.nan) for a in ARMS if a != "ctrl"}
    fsign = {a: np.full((n, L), np.nan) for a in ARMS if a != "ctrl"}
    mismatches = 0
    for i in range(n):
        inj_pos = len(replay(seeds[i], warmup_frames=0)["trade_idx"])
        for arm in resp:
            rep = replay(streams[arm][i], warmup_frames=0)
            idx, s = rep["trade_idx"], rep["trade_signs"]
            m2 = rep["mids2"].astype(np.float64)
            if inj_pos >= len(idx):
                continue
            want = 1 if arm.startswith("buy") else -1
            if s[inj_pos] != want:  # bookkeeping check: that trade IS the injection
                mismatches += 1
                continue
            base = m2[idx[inj_pos]]
            for l in range(1, min(L, len(idx) - 1 - inj_pos) + 1):
                resp[arm][i, l - 1] = (m2[idx[inj_pos + l]] - base) / 2.0
                fsign[arm][i, l - 1] = s[inj_pos + l]
    return resp, fsign, mismatches


def boot(vals, n_boot=2000, seed=11):
    rng = np.random.default_rng(seed)
    mean = np.nanmean(vals, axis=0)
    idx = rng.integers(0, vals.shape[0], (n_boot, vals.shape[0]))
    b = np.nanmean(vals[idx], axis=1)
    return mean, np.nanpercentile(b, 2.5, axis=0), np.nanpercentile(b, 97.5, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", default="data/tokens_v2/val.bin")
    p.add_argument("--symbol", default="SOLUSDT")
    p.add_argument("--pilot", default="eval_out/pilot")
    p.add_argument("--seg-len", type=int, default=2_000_000)
    p.add_argument("--segs", type=int, default=4)
    args = p.parse_args()

    # --- real tape: R, C, deconvolved G ---
    data = np.memmap(args.bin, dtype=np.uint16, mode="r")
    sym_id = SYMBOLS[args.symbol]
    regions = [(s, e) for sid, s, e in find_regions(np.asarray(data)) if sid == sym_id]
    start, end = max(regions, key=lambda r: r[1] - r[0])
    seg_starts = np.linspace(start + 1000, end - args.seg_len - 1, args.segs).astype(int)
    real = [replay(np.asarray(data[s:s + args.seg_len])) for s in seg_starts]
    print(f"real: {sum(len(r['trade_idx']) for r in real):,} trades "
          f"in {args.segs}x{args.seg_len / 1e6:.0f}M tokens", flush=True)
    R_real, C_real, _ = trade_response(real, 2 * L_REAL)
    G_real, rms = deconvolve(R_real, C_real, L_REAL)
    print(f"deconvolution fit rms {rms:.4f} ticks", flush=True)

    # per-segment G for a rough spread
    G_segs = []
    for rep in real:
        Rs, Cs, _ = trade_response([rep], 2 * L_REAL)
        G_segs.append(deconvolve(Rs, Cs, L_REAL)[0])
    G_sd = np.std(G_segs, axis=0)

    # --- generated endogenous (ctrl arm): same pipeline, small-sample ---
    z = np.load(Path(args.pilot) / f"streams_{args.symbol}.npz")
    streams = {a: list(z[a]) for a in ARMS}
    seeds = list(z["seeds"])
    ctrl_reps, gen_post = [], []
    for i, s in enumerate(seeds):
        inj_pos = len(replay(s, warmup_frames=0)["trade_idx"])
        rep = replay(streams["ctrl"][i], warmup_frames=0)
        gen_post.append({k: (rep[k][inj_pos:] if k in ("trade_idx", "trade_signs") else rep[k])
                         for k in ("mids2", "trade_idx", "trade_signs")})
    L_gen = 24
    R_gen, C_gen, n_gen = trade_response(gen_post, 2 * L_gen)
    G_gen, rms_g = deconvolve(R_gen, C_gen, L_gen)
    print(f"gen (ctrl): {int(n_gen[1])} post-seed trades, fit rms {rms_g:.4f}", flush=True)

    # --- injected response on the trade clock ---
    resp, fsign, mism = injection_trade_time(streams, seeds, L_INJ)
    if mism:
        print(f"WARNING: {mism} arm-streams failed the injected-sign check", flush=True)
    paired = {d: (resp[f"buy_{d}"] - resp[f"sell_{d}"]) / 2.0 for d in ("d1", "d2")}
    induced = {d: (fsign[f"buy_{d}"] - fsign[f"sell_{d}"]) / 2.0 for d in ("d1", "d2")}

    out = {"symbol": args.symbol, "L_real": L_REAL,
           "R_real": R_real[1:L_REAL + 1].tolist(), "C_real": C_real[1:2 * L_REAL].tolist(),
           "G_real": G_real.tolist(), "G_real_sd": G_sd.tolist(), "fit_rms": rms,
           "R_gen": R_gen[1:L_gen + 1].tolist(), "G_gen": G_gen.tolist(),
           "lags_inj": list(range(1, L_INJ + 1))}
    print(f"\n=== trade-clock comparison, {args.symbol} (lags in TRADES) ===")
    print("lag |  R_real  G_real(+-sd) | G_gen |  inj_d1 [CI]        inj_d2 [CI]      | induced_d1")
    for d in ("d1", "d2"):
        m, lo, hi = boot(paired[d])
        out[f"inj_{d}"] = {"mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
        im, ilo, ihi = boot(induced[d])
        out[f"induced_{d}"] = {"mean": im.tolist(), "lo": ilo.tolist(), "hi": ihi.tolist()}
    i1, i2 = out["inj_d1"], out["inj_d2"]
    ind1 = out["induced_d1"]
    for lag in SHOW:
        j = lag - 1
        g_gen = f"{G_gen[j]:+.3f}" if lag <= L_gen else "     "
        print(f" {lag:>3} | {R_real[lag]:+.3f}  {G_real[j]:+.3f}+-{G_sd[j]:.3f} | {g_gen} |"
              f" {i1['mean'][j]:+.3f} [{i1['lo'][j]:+.2f},{i1['hi'][j]:+.2f}]"
              f" {i2['mean'][j]:+.3f} [{i2['lo'][j]:+.2f},{i2['hi'][j]:+.2f}] |"
              f" {ind1['mean'][j]:+.3f}")
    print(f"\nsign ACF C(l) real: l=1 {C_real[1]:.3f}  l=8 {C_real[8]:.3f}  "
          f"l=32 {C_real[32]:.3f}   gen: l=1 {C_gen[1]:.3f}  l=8 {C_gen[8]:.3f}")

    with open(Path(args.pilot) / f"propagator_{args.symbol}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.pilot}/propagator_{args.symbol}.json")


if __name__ == "__main__":
    main()
