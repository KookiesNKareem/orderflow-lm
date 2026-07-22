"""Evaluate generated order flow separately by symbol."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm.sim import find_regions, ks, replay  # noqa: E402
from oflm.tokenizer import SYMBOLS  # noqa: E402

SYM_NAMES = {v: k for k, v in SYMBOLS.items()}
LAGS = [1, 2, 5, 10, 20, 50, 100]
SIZE_LAG = 20


def impact(runs, lags):
    """Pooled mean response in ticks per lag across replay dicts."""
    out = []
    for lag in lags:
        s = n = 0
        for r in runs:
            idx, signs, m = r["trade_idx"], r["trade_signs"], r["mids2"]
            ok = idx + lag < len(m)
            if ok.any():
                s += (signs[ok] * (m[idx[ok] + lag] - m[idx[ok]]) / 2).sum()
                n += int(ok.sum())
        out.append((s / n if n else float("nan"), n))
    return out


def impact_by_size(runs, bounds, lag=SIZE_LAG):
    groups = [[] for _ in range(len(bounds) + 1)]
    for r in runs:
        idx, signs, sizes, m = r["trade_idx"], r["trade_signs"], r["trade_sizes"], r["mids2"]
        ok = idx + lag < len(m)
        resp = signs[ok] * (m[idx[ok] + lag] - m[idx[ok]]) / 2
        g = np.searchsorted(bounds, sizes[ok], side="right")
        for k in range(len(groups)):
            groups[k].extend(resp[g == k].tolist())
    return [(float(np.mean(g)) if g else float("nan"), len(g)) for g in groups]


def summarize(tag, pool):
    sp = np.concatenate([r["spreads"] for r in pool])
    dp = np.concatenate([r["depths"] for r in pool])
    frames = sum(r["frames"] for r in pool)
    trades = sum(r["trades"] for r in pool)
    sim_s = sum(r["sim_s"] for r in pool)
    crossed = sum(r["crossed"] for r in pool)
    print(f"  {tag:>5}: frames/s {frames / sim_s:6.1f}  trades/s {trades / sim_s:6.1f}  "
          f"spread p50/p90/p99 {np.percentile(sp, [50, 90, 99])}  "
          f"depth p50 {np.percentile(dp, 50):.3f}  crossed/kframe {1000 * crossed / frames:.2f}")
    return sp, dp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--new-tokens", type=int, default=12000)
    p.add_argument("--seg-len", type=int, default=1_200_000)
    args = p.parse_args()

    import torch
    from oflm.generate import generate
    from oflm.model import GPT, Config

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    block = model.c.block
    print(f"ckpt step {ckpt['step']} val {ckpt['val_loss']:.4f}", flush=True)

    data = np.memmap(args.bin, dtype=np.uint16, mode="r")
    regions = find_regions(np.asarray(data))
    rng = np.random.default_rng(0)

    for sym_id, start, end in regions:
        name = SYM_NAMES.get(sym_id, f"sym{sym_id}")
        print(f"\n=== {name} (tokens {start:,}..{end:,}) ===", flush=True)

        seg_starts = np.linspace(start + 1000, end - args.seg_len - 1, 4).astype(int)
        real = [replay(np.asarray(data[s:s + args.seg_len])) for s in seg_starts]
        gen = []
        for k in range(args.seeds):
            at = int(rng.integers(start + 1000, end - block - 1000))
            seed = np.asarray(data[at:at + block])
            toks, gstats = generate(model, seed, args.new_tokens, 1.0, device)
            seed_frames = replay(seed, warmup_frames=0)["frames"]
            gen.append(replay(np.concatenate([seed, np.array(toks, dtype=np.uint16)]),
                              warmup_frames=seed_frames))
            print(f"  seed {k + 1}/{args.seeds} done (illegal mass "
                  f"{gstats['illegal_mass_avg']:.4f})", flush=True)

        sp_r, dp_r = summarize("real", real)
        sp_g, dp_g = summarize("gen", gen)
        floor_sp = ks(np.concatenate([real[0]["spreads"], real[2]["spreads"]]),
                      np.concatenate([real[1]["spreads"], real[3]["spreads"]]))
        floor_dp = ks(np.concatenate([real[0]["depths"], real[2]["depths"]]),
                      np.concatenate([real[1]["depths"], real[3]["depths"]]))
        print(f"  spread KS {ks(sp_r, sp_g):.4f} (floor {floor_sp:.4f})   "
              f"depth KS {ks(dp_r, dp_g):.4f} (floor {floor_dp:.4f})")

        imp_r, imp_g = impact(real, LAGS), impact(gen, LAGS)
        print(f"  impact R(l) ticks   lags: {LAGS}")
        print("    real: " + "  ".join(f"{v:6.3f}" for v, _ in imp_r) + f"   (n={imp_r[0][1]:,})")
        print("    gen : " + "  ".join(f"{v:6.3f}" for v, _ in imp_g) + f"   (n={imp_g[0][1]:,})")

        all_sizes = np.concatenate([r["trade_sizes"] for r in real])
        bounds = np.percentile(all_sizes, [33.3, 66.7]).astype(np.int64)
        ibs_r = impact_by_size(real, bounds)
        ibs_g = impact_by_size(gen, bounds)
        print(f"  impact by size @ lag {SIZE_LAG} (bucket terciles, bounds {bounds.tolist()}):")
        print("    real: " + "  ".join(f"{v:6.3f}(n={n})" for v, n in ibs_r))
        print("    gen : " + "  ".join(f"{v:6.3f}(n={n})" for v, n in ibs_g))


if __name__ == "__main__":
    main()
