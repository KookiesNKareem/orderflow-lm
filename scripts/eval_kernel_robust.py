"""Evaluate impact-kernel sensitivity and uncertainty."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm.sim import find_regions, replay  # noqa: E402
from oflm.tokenizer import SYMBOLS  # noqa: E402

PLATEAU = slice(15, 24)  # trade lags 16..24


def response_sums(rep, L):
    """Per-segment R/C numerators and counts on the trade clock."""
    m2 = rep["mids2"].astype(np.float64)
    idx, s = rep["trade_idx"], rep["trade_signs"].astype(np.float64)
    k = len(idx)
    R_num, C_num, n = np.zeros(L + 1), np.zeros(L + 1), np.zeros(L + 1)
    for l in range(1, min(L, k - 1) + 1):
        d = (m2[idx[l:]] - m2[idx[:k - l]]) / 2.0
        R_num[l] = (s[:k - l] * d).sum()
        C_num[l] = (s[:k - l] * s[l:]).sum()
        n[l] = k - l
    return R_num, C_num, n


def pool(sums, which):
    R = sum(s[0] for s in sums) if which == "R" else sum(s[1] for s in sums)
    n = sum(s[2] for s in sums)
    with np.errstate(invalid="ignore"):
        return R / n


def deconvolve(R, C, L, lam=0.0):
    Cfull = np.concatenate([[1.0], C[1:2 * L]]) if len(C) >= 2 * L else \
        np.concatenate([[1.0], C[1:], np.zeros(2 * L - len(C))])
    A = np.empty((L, L))
    for l in range(1, L + 1):
        for m in range(1, L + 1):
            A[l - 1, m - 1] = Cfull[abs(m - l)] - Cfull[m]
    if lam > 0:
        mu = lam * np.mean(np.diag(A.T @ A))
        return np.linalg.solve(A.T @ A + mu * np.eye(L), A.T @ R[1:L + 1])
    return np.linalg.lstsq(A, R[1:L + 1], rcond=None)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", default="data/tokens_v2/val.bin")
    p.add_argument("--symbol", default="SOLUSDT")
    p.add_argument("--segs", type=int, default=8)
    p.add_argument("--seg-len", type=int, default=1_000_000)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", default="eval_out/robust")
    args = p.parse_args()

    data = np.memmap(args.bin, dtype=np.uint16, mode="r")
    sym_id = SYMBOLS[args.symbol]
    regions = [(s, e) for sid, s, e in find_regions(np.asarray(data)) if sid == sym_id]
    start, end = max(regions, key=lambda r: r[1] - r[0])
    seg_starts = np.linspace(start + 1000, end - args.seg_len - 1, args.segs).astype(int)

    Lmax = 64
    sums = []
    for i, s in enumerate(seg_starts):
        rep = replay(np.asarray(data[s:s + args.seg_len]))
        sums.append(response_sums(rep, 2 * Lmax))
        print(f"segment {i + 1}/{args.segs}: {len(rep['trade_idx']):,} trades", flush=True)

    R, C = pool(sums, "R"), pool(sums, "C")
    sweep = {}
    for L in (32, 40, 48, 64):
        for lam in (0.0, 1e-3, 1e-2, 1e-1):
            G = deconvolve(R, C, L, lam)
            sweep[f"L{L}_lam{lam:g}"] = float(np.mean(G[PLATEAU]))
    print("plateau sweep:", {k: round(v, 4) for k, v in sweep.items()})

    rng = np.random.default_rng(17)
    boots = []
    for _ in range(args.n_boot):
        pick = rng.integers(0, args.segs, args.segs)
        sub = [sums[j] for j in pick]
        G = deconvolve(pool(sub, "R"), pool(sub, "C"), Lmax, 0.0)
        boots.append(float(np.mean(G[PLATEAU])))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"bootstrap plateau: {boots.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"robust_{args.symbol}.json", "w") as f:
        json.dump({"symbol": args.symbol, "segs": args.segs, "seg_len": args.seg_len,
                   "plateau_sweep": sweep,
                   "boot_mean": float(boots.mean()), "boot_lo": float(lo),
                   "boot_hi": float(hi), "boot_samples": boots.tolist()}, f)
    print(f"wrote {outdir / f'robust_{args.symbol}.json'}")


if __name__ == "__main__":
    main()
