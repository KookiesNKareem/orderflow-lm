"""Book-applier evaluation.

validate: replay real tokenized data through the Book applier and compare
  frame-end spread and mid-price changes against the reconstruction parquet
  (ground truth). The match rate measures how much book state the token
  representation actually carries (deep out-of-band changes are its known gap).

compare: spread/depth/event-rate distributions for model rollouts vs real
  data, with a real-vs-real noise floor. KS distances; the model "passes" a
  metric when real-vs-gen is close to the real-vs-real floor.

Usage:
  python scripts/eval_book.py validate --bin data/tokens/val.bin \
      --parquet data/normalized/BTCUSDT/2026-06-01.parquet --tokens 3000000
  python scripts/eval_book.py compare --bin data/tokens/val.bin \
      --ckpt checkpoints/v0.1.pt --seeds 12 --new-tokens 4000
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm.sim import Book, iter_events, dt_rep, TICK  # noqa: E402

WARMUP_FRAMES = 2000


def ks(a, b) -> float:
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    ca = np.searchsorted(a, grid, side="right") / len(a)
    cb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(ca - cb).max())


def replay(tokens, warmup_frames=WARMUP_FRAMES):
    """Apply a token stream; returns per-frame series + event rates."""
    book = Book()
    spreads, mids, depths = [], [], []
    n_frames = n_trades = 0
    sim_us = 0
    for ev in iter_events(tokens):
        sim_us += dt_rep(ev[1])
        if ev[0] == "trade":
            n_trades += 1
            continue
        book.apply_frame(ev[2])
        n_frames += 1
        if n_frames <= warmup_frames:
            continue
        s, bb, ba = book.spread_ticks(), book.best(0), book.best(1)
        if s is not None:
            spreads.append(s)
            mids.append(bb + ba)  # 2*mid in ticks; differences are anchor-free
            depths.append(book.depth_at_best())
    return {
        "spreads": np.array(spreads), "mids2": np.array(mids),
        "depths": np.array(depths), "frames": n_frames, "trades": n_trades,
        "sim_s": sim_us / 1e6, "crossed": book.crossed_frames,
    }


def validate(args):
    import polars as pl
    data = np.memmap(args.bin, dtype=np.uint16, mode="r")
    r = replay(np.asarray(data[:args.tokens]), warmup_frames=args.warmup)
    truth = (pl.scan_parquet(args.parquet)
             .filter(pl.col("kind") == 0, pl.col("new_frame"))
             .select("best_bid", "best_ask").head(r["frames"]).collect())
    tb = np.round(truth["best_bid"].to_numpy() / TICK).astype(np.int64)
    ta = np.round(truth["best_ask"].to_numpy() / TICK).astype(np.int64)
    t_spread = (ta - tb)[args.warmup:]
    t_mid2 = (ta + tb)[args.warmup:]
    n = min(len(t_spread), len(r["spreads"]))
    s_ok = r["spreads"][:n] == t_spread[:n]
    dm_ok = np.diff(r["mids2"][:n]) == np.diff(t_mid2[:n])
    print(f"frames compared: {n:,} (after {args.warmup} warmup)")
    print(f"spread exact-match:     {s_ok.mean():.2%}   "
          f"(|err| p50/p90/p99: {np.percentile(np.abs(r['spreads'][:n]-t_spread[:n]), [50,90,99])})")
    print(f"mid-change exact-match: {dm_ok.mean():.2%}")
    print(f"crossed frames (should be ~0 on real data): {r['crossed']}")
    for q, (lo, hi) in enumerate(zip(range(0, n, n // 4), range(n // 4, n + 1, n // 4))):
        print(f"  quartile {q + 1}: spread match {s_ok[lo:hi].mean():.2%}")


def summarize(name, r):
    sp = r["spreads"]
    print(f"{name:>12}: frames/s {r['frames'] / r['sim_s']:6.1f}  trades/s {r['trades'] / r['sim_s']:6.1f}  "
          f"spread p50/p90/p99 {np.percentile(sp, [50, 90, 99])}  "
          f"depth p50 {np.percentile(r['depths'], 50):.3f}  crossed/kframe {1000 * r['crossed'] / r['frames']:.2f}")


def compare(args):
    import torch
    from oflm.model import GPT, Config
    from oflm.generate import generate

    data = np.memmap(args.bin, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)

    # real segments, alternating halves for the noise floor
    seg_len = 600_000
    starts = np.linspace(0, len(data) - seg_len, 8).astype(int)
    real = [replay(np.asarray(data[s:s + seg_len])) for s in starts]
    real_a = np.concatenate([r["spreads"] for r in real[0::2]])
    real_b = np.concatenate([r["spreads"] for r in real[1::2]])

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])

    gen_runs = []
    for k in range(args.seeds):
        at = rng.integers(0, len(data) - 4096)
        seed = np.asarray(data[at:at + 1024])
        toks, _ = generate(model, seed, args.new_tokens, 1.0, device)
        # warm the book on the seed, then measure only the generated part
        run = replay(np.concatenate([seed, np.array(toks, dtype=np.uint16)]),
                     warmup_frames=0)
        seed_frames = replay(seed, warmup_frames=0)["frames"]
        for key in ("spreads", "mids2", "depths"):
            run[key] = run[key][seed_frames:]
        run["frames"] -= seed_frames
        gen_runs.append(run)
        print(f"  seed {k + 1}/{args.seeds} done", flush=True)
    gen = {
        "spreads": np.concatenate([r["spreads"] for r in gen_runs]),
        "depths": np.concatenate([r["depths"] for r in gen_runs]),
        "frames": sum(r["frames"] for r in gen_runs),
        "trades": sum(r["trades"] for r in gen_runs),
        "sim_s": sum(r["sim_s"] for r in gen_runs),
        "crossed": sum(r["crossed"] for r in gen_runs),
    }

    all_real = {
        "spreads": np.concatenate([r["spreads"] for r in real]),
        "depths": np.concatenate([r["depths"] for r in real]),
        "frames": sum(r["frames"] for r in real),
        "trades": sum(r["trades"] for r in real),
        "sim_s": sum(r["sim_s"] for r in real),
        "crossed": sum(r["crossed"] for r in real),
    }
    summarize("real", all_real)
    summarize("generated", gen)
    print(f"\nspread KS  real-vs-real (noise floor): {ks(real_a, real_b):.4f}")
    print(f"spread KS  real-vs-generated:          {ks(all_real['spreads'], gen['spreads']):.4f}")
    print(f"depth  KS  real-vs-generated:          {ks(all_real['depths'], gen['depths']):.4f}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--bin", required=True)
    v.add_argument("--parquet", required=True)
    v.add_argument("--tokens", type=int, default=3_000_000)
    v.add_argument("--warmup", type=int, default=WARMUP_FRAMES)
    c = sub.add_parser("compare")
    c.add_argument("--bin", required=True)
    c.add_argument("--ckpt", required=True)
    c.add_argument("--seeds", type=int, default=12)
    c.add_argument("--new-tokens", type=int, default=4000)
    args = p.parse_args()
    validate(args) if args.cmd == "validate" else compare(args)


if __name__ == "__main__":
    main()
