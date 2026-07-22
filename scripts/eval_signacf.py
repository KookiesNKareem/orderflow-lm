"""Measure order-sign memory in generated and real order flow."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm.sim import find_regions, iter_events  # noqa: E402
from oflm.tokenizer import SYMBOLS  # noqa: E402

SYM_NAMES = {v: k for k, v in SYMBOLS.items()}
LAGS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]


def signs_of(tokens) -> np.ndarray:
    return np.array([1 if ev[2] == 1 else -1
                     for ev in iter_events(tokens) if ev[0] == "trade"], dtype=np.float64)


def pooled_acf(seqs, lags):
    """ACF across sequences: demean per pool, sum lagged covariances."""
    allv = np.concatenate(seqs)
    m, var = allv.mean(), allv.var()
    out = []
    for lag in lags:
        s = n = 0.0
        for x in seqs:
            if len(x) > lag:
                d = x - m
                s += (d[:-lag] * d[lag:]).sum()
                n += len(x) - lag
        out.append(s / n / var if n > 0 else float("nan"))
    return np.array(out), m, int(sum(len(x) for x in seqs))


def fit_gamma(lags, acf, lo=5, hi=250):
    L, A = [], []
    for lag, a in zip(lags, acf):
        if lo <= lag <= hi and a > 0:
            L.append(np.log(lag))
            A.append(np.log(a))
    if len(L) < 3:
        return float("nan")
    return -np.polyfit(L, A, 1)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--new-tokens", type=int, default=16000)
    p.add_argument("--real-tokens", type=int, default=8_000_000)
    args = p.parse_args()

    import torch
    from oflm.generate import generate_batch
    from oflm.model import GPT, Config

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    block = model.c.block
    print(f"ckpt step {ckpt['step']} val {ckpt['val_loss']:.4f} device {device}", flush=True)

    data = np.memmap(args.bin, dtype=np.uint16, mode="r")
    regions = find_regions(np.asarray(data))
    rng = np.random.default_rng(1)

    for sym_id, start, end in regions:
        name = SYM_NAMES.get(sym_id, f"sym{sym_id}")
        mid = start + (end - start) // 2
        real_seq = signs_of(np.asarray(data[mid:mid + args.real_tokens]))

        seeds = []
        for _ in range(args.batch):
            at = int(rng.integers(start + 1000, end - block - 1000))
            seeds.append(np.asarray(data[at:at + block]))
        t0 = time.time()
        outs, gstats = generate_batch(model, seeds, args.new_tokens, 1.0, device)
        tput = args.batch * args.new_tokens / (time.time() - t0)
        gen_seqs = [signs_of(np.array(o, dtype=np.int64)) for o in outs]

        acf_r, m_r, n_r = pooled_acf([real_seq], LAGS)
        acf_g, m_g, n_g = pooled_acf(gen_seqs, LAGS)
        print(f"\n=== {name} ===  ({tput:,.0f} tok/s gen, illegal mass {gstats['illegal_mass_avg']:.4f})")
        print(f"  lags: {LAGS}")
        print("  real: " + " ".join(f"{a:6.3f}" for a in acf_r)
              + f"   (n={n_r:,}, mean sign {m_r:+.3f})")
        print("  gen : " + " ".join(f"{a:6.3f}" for a in acf_g)
              + f"   (n={n_g:,}, mean sign {m_g:+.3f})")
        print(f"  power-law gamma: real {fit_gamma(LAGS, acf_r):.3f}   "
              f"gen {fit_gamma(LAGS, acf_g, hi=100):.3f}  (real fit 5-250, gen 5-100)")


if __name__ == "__main__":
    main()
