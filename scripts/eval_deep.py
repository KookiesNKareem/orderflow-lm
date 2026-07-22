"""Evaluate distributional and temporal realism for one symbol."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm.sim import find_regions, ks, replay  # noqa: E402
from oflm.tokenizer import SYMBOLS  # noqa: E402

SYM_NAMES = {v: k for k, v in SYMBOLS.items()}
IMPACT_LAGS = [1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200]
ACF_LAGS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
RET_LAGS = list(range(1, 21))
VOL_LAGS = [1, 2, 5, 10, 20, 50, 100, 200, 500]
SIG_DELTAS = [1, 2, 5, 10, 20, 50, 100, 200]
DEPTH_BIN = 500  # frames per degradation bin


def pooled_acf(seqs, lags):
    seqs = [s for s in seqs if len(s) > 10]
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
        out.append(float(s / n / var) if n > 0 and var > 0 else float("nan"))
    return out


def fit_gamma(lags, acf, lo, hi):
    pts = [(np.log(l), np.log(a)) for l, a in zip(lags, acf) if lo <= l <= hi and a > 0]
    if len(pts) < 3:
        return float("nan")
    x, y = zip(*pts)
    return float(-np.polyfit(x, y, 1)[0])


def returns(runs, delta):
    out = []
    for r in runs:
        m = r["mids2"].astype(np.float64)
        if len(m) > delta:
            out.append((m[delta:] - m[:-delta]) / 2.0)
    return out


def impact(runs, lags):
    out, ns = [], []
    for lag in lags:
        s = n = 0.0
        for r in runs:
            idx, signs, m = r["trade_idx"], r["trade_signs"], r["mids2"]
            ok = idx + lag < len(m)
            if ok.any():
                s += (signs[ok] * (m[idx[ok] + lag] - m[idx[ok]]) / 2).sum()
                n += int(ok.sum())
        out.append(float(s / n) if n else float("nan"))
        ns.append(int(n))
    return out, ns


def sign_seqs(runs):
    return [r["trade_signs"].astype(np.float64) for r in runs]


def stylized(runs):
    r1 = [x for x in returns(runs, 1) if len(x) > 600]
    out = {
        "ret_acf": pooled_acf(r1, RET_LAGS),
        "vol_acf": pooled_acf([np.abs(x) for x in r1], VOL_LAGS),
        "kurtosis": {}, "signature": [],
    }
    for d in (1, 10, 50):
        pooled = np.concatenate(returns(runs, d))
        mu, sd = pooled.mean(), pooled.std()
        out["kurtosis"][str(d)] = float(((pooled - mu) ** 4).mean() / sd ** 4 - 3) if sd > 0 else float("nan")
    for d in SIG_DELTAS:
        pooled = np.concatenate(returns(runs, d))
        out["signature"].append(float(pooled.var() / d) if len(pooled) else float("nan"))
    return out


def degradation(runs, bin_frames=DEPTH_BIN):
    n_bins = max(len(r["spreads"]) for r in runs) // bin_frames
    med, p90 = [], []
    for b in range(n_bins):
        vals = np.concatenate([r["spreads"][b * bin_frames:(b + 1) * bin_frames] for r in runs])
        if len(vals) < 50:
            break
        med.append(float(np.median(vals)))
        p90.append(float(np.percentile(vals, 90)))
    return {"bin_frames": bin_frames, "median": med, "p90": p90}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--new-tokens", type=int, default=30000)
    p.add_argument("--seg-len", type=int, default=2_000_000)
    p.add_argument("--out", default="eval_out")
    args = p.parse_args()

    import torch
    from oflm.generate import generate_batch
    from oflm.model import GPT, Config

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    block = model.c.block
    Path(args.out).mkdir(exist_ok=True)
    print(f"ckpt step {ckpt['step']} val {ckpt['val_loss']:.4f} device {device}", flush=True)

    data = np.memmap(args.bin, dtype=np.uint16, mode="r")
    regions = find_regions(np.asarray(data))
    rng = np.random.default_rng(2)

    for sym_id, start, end in regions:
        name = SYM_NAMES.get(sym_id, f"sym{sym_id}")
        print(f"=== {name} ===", flush=True)

        seg_starts = np.linspace(start + 1000, end - args.seg_len - 1, 4).astype(int)
        real = [replay(np.asarray(data[s:s + args.seg_len])) for s in seg_starts]
        print("  real segments replayed", flush=True)

        seeds = []
        for _ in range(args.batch):
            at = int(rng.integers(start + 1000, end - block - 1000))
            seeds.append(np.asarray(data[at:at + block]))
        t0 = time.time()
        outs, gstats = generate_batch(model, seeds, args.new_tokens, 1.0, device)
        print(f"  generated {args.batch}x{args.new_tokens} at "
              f"{args.batch * args.new_tokens / (time.time() - t0):,.0f} tok/s", flush=True)
        gen = []
        for s, o in zip(seeds, outs):
            sf = replay(s, warmup_frames=0)["frames"]
            gen.append(replay(np.concatenate([s, np.array(o, dtype=np.uint16)]),
                              warmup_frames=sf))

        imp_r, imp_r_n = impact(real, IMPACT_LAGS)
        imp_g, imp_g_n = impact(gen, IMPACT_LAGS)
        acf_r = pooled_acf(sign_seqs(real), ACF_LAGS)
        acf_g = pooled_acf(sign_seqs(gen), ACF_LAGS)
        sp_r = np.concatenate([r["spreads"] for r in real])
        sp_g = np.concatenate([r["spreads"] for r in gen])
        dp_r = np.concatenate([r["depths"] for r in real])
        dp_g = np.concatenate([r["depths"] for r in gen])

        result = {
            "symbol": name,
            "illegal_mass": gstats["illegal_mass_avg"],
            "impact": {"lags": IMPACT_LAGS, "real": imp_r, "gen": imp_g,
                       "n_real": imp_r_n, "n_gen": imp_g_n},
            "sign_acf": {"lags": ACF_LAGS, "real": acf_r, "gen": acf_g,
                         "gamma_real": fit_gamma(ACF_LAGS, acf_r, 5, 250),
                         "gamma_gen": fit_gamma(ACF_LAGS, acf_g, 5, 150)},
            "stylized_real": stylized(real),
            "stylized_gen": stylized(gen),
            "ret_lags": RET_LAGS, "vol_lags": VOL_LAGS, "sig_deltas": SIG_DELTAS,
            "degradation_gen": degradation(gen),
            "spread_ref": {"real_med": float(np.median(sp_r)),
                           "real_p90": float(np.percentile(sp_r, 90))},
            "marginals": {
                "spread_ks": ks(sp_r, sp_g),
                "spread_floor": ks(np.concatenate([real[0]["spreads"], real[2]["spreads"]]),
                                   np.concatenate([real[1]["spreads"], real[3]["spreads"]])),
                "depth_ks": ks(dp_r, dp_g),
                "spread_p": {"real": np.percentile(sp_r, [50, 90, 99]).tolist(),
                             "gen": np.percentile(sp_g, [50, 90, 99]).tolist()},
                "rates": {"real_fps": sum(r["frames"] for r in real) / sum(r["sim_s"] for r in real),
                          "gen_fps": sum(r["frames"] for r in gen) / sum(r["sim_s"] for r in gen),
                          "real_tps": sum(r["trades"] for r in real) / sum(r["sim_s"] for r in real),
                          "gen_tps": sum(r["trades"] for r in gen) / sum(r["sim_s"] for r in gen)},
                "crossed_gen_kf": 1000 * sum(r["crossed"] for r in gen) / sum(r["frames"] for r in gen),
                "crossed_real_kf": 1000 * sum(r["crossed"] for r in real) / sum(r["frames"] for r in real),
            },
        }
        with open(f"{args.out}/deep_{name}.json", "w") as f:
            json.dump(result, f, indent=1)
        print(f"  impact real {imp_r[0]:.2f}->{imp_r[-1]:.2f}  gen {imp_g[0]:.2f}->{imp_g[-1]:.2f}")
        print(f"  sign gamma real {result['sign_acf']['gamma_real']:.3f}  "
              f"gen {result['sign_acf']['gamma_gen']:.3f}")
        print(f"  wrote {args.out}/deep_{name}.json", flush=True)


if __name__ == "__main__":
    main()
