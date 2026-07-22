"""Run paired trade-injection experiments under common random numbers."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm import tokenizer as tk  # noqa: E402
from oflm.sim import find_regions, iter_events, replay  # noqa: E402
from oflm.tokenizer import SYMBOLS  # noqa: E402

LAGS = [1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200]
ARMS = ["buy_d1", "sell_d1", "buy_d2", "sell_d2", "ctrl"]
SEED_LEN = 1532          # +4 injected tokens = 1536 = 0.75 * block
DOSE_STEP = 3            # dose2 = dose1 + 3 buckets (~8x size)


def cut_seeds(data, start, end, n, seed_len):
    """Windows of exactly seed_len ending at an event boundary (a Δt token
    starts the next event; Δt ids 2..33 appear only as Δt, so boundaries are
    findable without grammar state). Windows containing BOS/SYM are skipped."""
    # n primary anchors spanning the full region; midpoint fallbacks absorb
    # rejections without biasing coverage toward the region's start.
    lo, hi = start + seed_len + 1000, end - 5000
    anchors = np.concatenate([np.linspace(lo, hi, n).astype(int),
                              np.linspace(lo, hi, 2 * n + 1).astype(int)[1::2]])
    seeds, used = [], []
    for a in anchors:
        if len(seeds) == n:
            break
        seg = np.asarray(data[a:a + 4000]).astype(np.int64)
        bounds = np.flatnonzero((seg >= tk.DT0) & (seg < tk.DT0 + 32))
        if len(bounds) == 0:
            continue
        e = a + int(bounds[0])
        w = np.asarray(data[e - seed_len:e]).astype(np.int64)
        if (w < tk.DT0).any() or (w >= tk.SYM0).any():  # PAD/BOS or SYM in window
            continue
        seeds.append(w)
        used.append(e)
    return seeds, used


def injection_params(seed):
    """(dt_bucket, size_bucket_dose1) from the seed's own trades."""
    dts, szs = [], []
    for ev in iter_events(seed):
        if ev[0] == "trade":
            dts.append(ev[1])
            szs.append(ev[4])
    dtb = int(np.median(dts)) if dts else 10
    szb = int(np.median(szs)) if szs else 8
    return dtb, szb


def inject_tokens(side, dtb, szb):
    head = tk.TRADE_BUY if side == "buy" else tk.TRADE_SELL
    return [tk.DT0 + dtb, head, tk.TOFF0 + 15, tk.TSIZE0 + szb]  # offset 0: at best


def arm_seed(seed, arm, dtb, szb):
    if arm == "ctrl":
        return list(seed)
    side, dose = arm.split("_")
    b = min(szb + (DOSE_STEP if dose == "d2" else 0), 31)
    return list(seed) + inject_tokens(side, dtb, b)


def run_generation(args, out_dir):
    import torch
    from oflm.generate import generate_batch
    from oflm.model import GPT, Config

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"ckpt step {ckpt['step']} val {ckpt['val_loss']:.4f} device {device}", flush=True)

    data = np.memmap(args.bin, dtype=np.uint16, mode="r")
    sym_id = SYMBOLS[args.symbol]
    regions = [(s, e) for sid, s, e in find_regions(np.asarray(data)) if sid == sym_id]
    region = max(regions, key=lambda r: r[1] - r[0])
    f0, f1 = (float(x) for x in args.span.split(","))
    s0, e0 = region
    region = (int(s0 + f0 * (e0 - s0)), int(s0 + f1 * (e0 - s0)))
    print(f"{args.symbol}: {len(regions)} region(s), using largest, "
          f"span [{f0},{f1}] -> {(region[1] - region[0]) / 1e6:.1f}M tokens", flush=True)
    seeds, used = cut_seeds(data, *region, args.seeds, SEED_LEN)
    if len(seeds) < args.seeds:
        print(f"WARNING: only {len(seeds)}/{args.seeds} usable seeds", flush=True)
    params = [injection_params(s) for s in seeds]
    print(f"{len(seeds)} seeds; median inj (dtb, szb): "
          f"{np.median([p[0] for p in params])}, {np.median([p[1] for p in params])}", flush=True)

    # per-batch parts on disk: a killed run resumes for free (seeds are
    # deterministic from args, and RNG is re-seeded per batch)
    parts = out_dir / f"parts_{args.symbol}"
    parts.mkdir(exist_ok=True)
    streams = {}  # arm -> list of full uint16 streams (seed [+inj] + generated)
    for arm in ARMS:
        streams[arm] = [None] * len(seeds)
        t0 = time.time()
        for b0 in range(0, len(seeds), args.batch):
            batch_idx = list(range(b0, min(b0 + args.batch, len(seeds))))
            part = parts / f"{arm}_{b0}.npy"
            if part.exists():
                cached = np.load(part)
                for pos, i in enumerate(batch_idx):
                    streams[arm][i] = cached[pos]
                print(f"  {arm} [{b0}:{batch_idx[-1] + 1}] cached", flush=True)
                t0 = time.time()
                continue
            batch = [arm_seed(seeds[i], arm, *params[i]) for i in batch_idx]
            # same seed across arms (CRN pairing); unique per batch and symbol
            torch.manual_seed(1000 + b0 + 1_000_000 * tk.SYMBOLS[args.symbol])
            row_seeds = None
            if args.pcrn:
                # positional CRN: per-window noise streams shared across arms
                row_seeds = [20_000_000 * (1 + tk.SYMBOLS[args.symbol]) + i
                             for i in batch_idx]
            outs, gstats = generate_batch(model, batch, args.new_tokens, 1.0,
                                          device, row_seeds=row_seeds)
            for pos, i in enumerate(batch_idx):
                streams[arm][i] = np.array(batch[pos] + outs[pos], dtype=np.uint16)
            np.save(part, np.stack([streams[arm][i] for i in batch_idx]))
            print(f"  {arm} [{b0}:{batch_idx[-1] + 1}] illegal {gstats['illegal_mass_avg']:.4f} "
                  f"{len(batch) * args.new_tokens / (time.time() - t0):,.0f} tok/s", flush=True)
            t0 = time.time()

    np.savez_compressed(
        out_dir / f"streams_{args.symbol}.npz",
        **{arm: np.stack(streams[arm]) for arm in ARMS},  # equal length within an arm
        seeds=np.stack([np.asarray(s, dtype=np.uint16) for s in seeds]))
    meta = {"symbol": args.symbol, "seed_len": SEED_LEN, "new_tokens": args.new_tokens,
            "n_seeds": len(seeds), "seed_ends": [int(u) for u in used],
            "inj_params": [[int(a), int(b)] for a, b in params], "dose_step": DOSE_STEP}
    with open(out_dir / f"meta_{args.symbol}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {out_dir}/streams_{args.symbol}.npz", flush=True)


def response_curves(streams, seeds, lags):
    """Per-seed per-arm R(l) = (mid(t0-1+l) - mid(t0-1)) / 2 ticks, NaN-masked
    where the rollout has too few frames. t0 = frames in the (uninjected) seed."""
    n = len(seeds)
    curves = {arm: np.full((n, len(lags)), np.nan) for arm in ARMS}
    stability = {arm: {"crossed_kf": [], "spread_p50": [], "spread_p90": []} for arm in ARMS}
    for i in range(n):
        seed_rep = replay(seeds[i], warmup_frames=0)
        t0 = seed_rep["frames"]
        for arm in ARMS:
            rep = replay(streams[arm][i], warmup_frames=0)
            m = rep["mids2"].astype(np.float64)
            if len(m) <= t0:
                continue
            base = m[t0 - 1]
            for j, lag in enumerate(lags):
                if t0 - 1 + lag < len(m):
                    curves[arm][i, j] = (m[t0 - 1 + lag] - base) / 2.0
            post = rep["spreads"][t0:]
            gen_frames = rep["frames"] - t0
            if gen_frames > 0 and len(post):
                stability[arm]["crossed_kf"].append(
                    1000 * (rep["crossed"] - seed_rep["crossed"]) / gen_frames)
                stability[arm]["spread_p50"].append(float(np.median(post)))
                stability[arm]["spread_p90"].append(float(np.percentile(post, 90)))
    return curves, stability


def boot_ci(vals, n_boot=2000, seed=7):
    """vals: (n_seeds, n_lags) with NaNs. Returns (mean, lo, hi) per lag."""
    rng = np.random.default_rng(seed)
    n = vals.shape[0]
    mean = np.nanmean(vals, axis=0)
    idx = rng.integers(0, n, (n_boot, n))
    boots = np.nanmean(vals[idx], axis=1)  # (n_boot, n_lags)
    return mean, np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)


def analyze(args, out_dir):
    z = np.load(out_dir / f"streams_{args.symbol}.npz")
    streams = {arm: list(z[arm]) for arm in ARMS}
    seeds = list(z["seeds"])
    curves, stability = response_curves(streams, seeds, LAGS)

    paired = {}
    for dose in ("d1", "d2"):
        paired[dose] = (curves[f"buy_{dose}"] - curves[f"sell_{dose}"]) / 2.0
    result = {"symbol": args.symbol, "lags": LAGS, "n_seeds": len(seeds)}
    print(f"\n=== paired-injection impact, {args.symbol} ({len(seeds)} seeds) ===")
    for dose in ("d1", "d2"):
        mean, lo, hi = boot_ci(paired[dose])
        result[f"paired_{dose}"] = {"mean": mean.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
        sig = ["*" if l > 0 else ("!" if h < 0 else " ") for l, h in zip(lo, hi)]
        print(f"\ndose {dose}  (R_buy - R_sell)/2, ticks   [* = CI>0, ! = CI<0]")
        for j, lag in enumerate(LAGS):
            print(f"  l={lag:>3}  {mean[j]:+7.3f}  [{lo[j]:+7.3f}, {hi[j]:+7.3f}] {sig[j]}")
    cm, cl, ch = boot_ci(curves["ctrl"])
    result["ctrl_drift"] = {"mean": cm.tolist(), "lo": cl.tolist(), "hi": ch.tolist()}
    print(f"\nctrl drift @ l=50: {cm[LAGS.index(50)]:+.3f} "
          f"[{cl[LAGS.index(50)]:+.3f}, {ch[LAGS.index(50)]:+.3f}]  (want ~0)")

    result["stability"] = {arm: {k: float(np.mean(v)) if v else None for k, v in d.items()}
                           for arm, d in stability.items()}
    print("\nstability (post-injection):  crossed/kf  spread_p50  spread_p90")
    for arm in ARMS:
        s = result["stability"][arm]
        print(f"  {arm:<8} {s['crossed_kf']:>10.2f} {s['spread_p50']:>11.2f} {s['spread_p90']:>11.2f}")

    ref_path = Path(args.out).parent / f"deep_{args.symbol}.json"
    if ref_path.exists():
        with open(ref_path) as f:
            deep = json.load(f)
        result["reference"] = deep["impact"]
        rl, rv = deep["impact"]["lags"], deep["impact"]["real"]
        print("\nreference R(l) real (endogenous):",
              "  ".join(f"l={l}:{v:.2f}" for l, v in zip(rl, rv) if l in (8, 20, 50, 200)))

    with open(out_dir / f"pilot_{args.symbol}.json", "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nwrote {out_dir}/pilot_{args.symbol}.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", default="data/tokens_v2/val.bin")
    p.add_argument("--ckpt", default="checkpoints/pod/v2.pt")
    p.add_argument("--symbol", default="SOLUSDT")
    p.add_argument("--seeds", type=int, default=48)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--new-tokens", type=int, default=2500)
    p.add_argument("--out", default="eval_out/pilot")
    p.add_argument("--span", default="0,1",
                   help="fractional sub-span of the region to sample, e.g. 0.5,1")
    p.add_argument("--pcrn", action="store_true",
                   help="positional common random numbers across arms (Gumbel)")
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        run_generation(args, out_dir)
    analyze(args, out_dir)


if __name__ == "__main__":
    main()
