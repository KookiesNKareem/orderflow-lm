"""Measure execution cost in replay and the reactive simulator."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm import tokenizer as tk  # noqa: E402
from oflm.sim import find_regions, replay  # noqa: E402
from oflm.tokenizer import SYMBOLS  # noqa: E402

SEED_LEN = 1532


def cut_seeds(data, start, end, n, seed_len):
    """Windows of exactly seed_len ending at an event boundary, no BOS/SYM."""
    lo, hi = start + seed_len + 1000, end - 40000
    anchors = np.concatenate([np.linspace(lo, hi, n).astype(int),
                              np.linspace(lo, hi, 2 * n + 1).astype(int)[1::2]])
    seeds = []
    for a in anchors:
        if len(seeds) == n:
            break
        seg = np.asarray(data[a:a + 4000]).astype(np.int64)
        bounds = np.flatnonzero((seg >= tk.DT0) & (seg < tk.DT0 + 32))
        if len(bounds) == 0:
            continue
        e = a + int(bounds[0])
        w = np.asarray(data[e - seed_len:e]).astype(np.int64)
        if (w < tk.DT0).any() or (w >= tk.SYM0).any():
            continue
        seeds.append((w, e))
    return seeds


def last_dt(tokens):
    t = np.asarray(tokens, dtype=np.int64)
    dts = np.flatnonzero((t >= tk.DT0) & (t < tk.DT0 + 32))
    return int(dts[-1]) if len(dts) else len(tokens)


def common_trim(full_rows):
    """Trim every row to a shared length T = min last-Δt index, so the batch
    stays rectangular for generate_batch. The min-row ends exactly at a boundary;
    others may lose a partial trailing event (iter_events skips the desync). A
    spliced child (starting with Δt) then begins a clean event on every row."""
    T = min(last_dt(r) for r in full_rows)
    return [list(r[:T]) for r in full_rows], T


def child_tokens(dtb, szb):
    return [tk.DT0 + dtb, tk.TRADE_BUY, tk.TOFF0 + 15, tk.TSIZE0 + szb]  # buy at touch


def ask_ticks(stream):
    """Best ask (integer ticks) at the end of a stream, via warmup-free replay.
    ask = (mid2 + spread)/2 since mid2 = bid+ask, spread = ask-bid."""
    rep = replay(np.asarray(stream, dtype=np.uint16), warmup_frames=0)
    if not len(rep["mids2"]) or not len(rep["spreads"]):
        return None
    return (rep["mids2"][-1] + rep["spreads"][-1]) / 2.0


def mid_ticks(stream):
    rep = replay(np.asarray(stream, dtype=np.uint16), warmup_frames=0)
    return rep["mids2"][-1] / 2.0 if len(rep["mids2"]) else None


def median_inj(seed):
    from oflm.sim import iter_events
    dts = [ev[1] for ev in iter_events(seed) if ev[0] == "trade"]
    return int(np.median(dts)) if dts else 8


def run(args, out_dir):
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
    start, end = max(regions, key=lambda r: r[1] - r[0])
    f0, f1 = (float(x) for x in args.span.split(","))
    start, end = int(start + f0 * (end - start)), int(start + f1 * (end - start))
    seeds = cut_seeds(data, start, end, args.seeds, SEED_LEN)
    dtbs = [median_inj(s) for s, _ in seeds]
    doses = [int(x) for x in args.doses.split(",")]
    print(f"{len(seeds)} seeds, K={args.k_children}, spacing={args.spacing}, "
          f"doses={doses}", flush=True)

    def gen_segments(batch_seeds, inject_fn, b0):
        """K-segment generation. inject_fn(round, row) -> per-row child tokens or
        None. Returns per-row checkpoint asks and the arrival mid."""
        streams = [list(s) for s in batch_seeds]
        arrival = [mid_ticks(s) for s in streams]
        fills = [[] for _ in streams]
        for r in range(args.k_children):
            # CRN across arms and doses; unique per context batch
            torch.manual_seed(4000 + r + 1009 * b0)
            outs, _ = generate_batch(model, streams, args.spacing, 1.0, device)
            trimmed, _ = common_trim([streams[i] + outs[i] for i in range(len(streams))])
            for i in range(len(streams)):
                child = inject_fn(r, i)
                if child is not None:
                    fills[i].append(ask_ticks(trimmed[i]))  # fill = ask before child
                    trimmed[i] = trimmed[i] + child
                streams[i] = trimmed[i]
        return fills, arrival, streams

    result = {"symbol": args.symbol, "k": args.k_children, "spacing": args.spacing,
              "doses": doses, "impact_reactive": {},
              "shortfall_reactive": {}, "shortfall_ctrl": {}, "shortfall_naive": {}}

    for dose in doses:
        r_sf, c_sf, n_sf = [], [], []
        t0 = time.time()
        for b0 in range(0, len(seeds), args.batch):
            bi = list(range(b0, min(b0 + args.batch, len(seeds))))
            bseeds = [seeds[i][0] for i in bi]

            # reactive arm: inject child order every round
            rfill, rarr, _ = gen_segments(
                bseeds, lambda r, i: child_tokens(dtbs[bi[i]], dose), b0)
            # control arm: same generation, no injection (read ask at checkpoints)
            cfill, carr, _ = gen_segments_ctrl(
                bseeds, args, model, generate_batch, device, dtbs, bi)
            # naive arm: recorded future, real touch prices, no reaction
            nfill, narr = naive_arm(data, [seeds[i][1] for i in bi], args)

            def shortfall(fills, arr):
                good = [f for f in fills if f is not None]
                return (np.mean(good) - arr) if good and arr is not None else np.nan

            for k, i in enumerate(bi):
                r_sf.append(shortfall(rfill[k], rarr[k]))
                c_sf.append(shortfall(cfill[k], carr[k]))
                n_sf.append(shortfall([nfill[k]], narr[k]))
            print(f"  dose {dose} [{b0}:{bi[-1]+1}] "
                  f"{len(bi)*args.k_children*args.spacing/(time.time()-t0):,.0f} tok/s", flush=True)
            t0 = time.time()

        r_sf, c_sf, n_sf = map(np.array, (r_sf, c_sf, n_sf))
        k = str(dose)
        result["impact_reactive"][k] = boot(r_sf - c_sf)   # executor impact, drift removed
        result["shortfall_reactive"][k] = float(np.nanmean(r_sf))
        result["shortfall_ctrl"][k] = float(np.nanmean(c_sf))
        result["shortfall_naive"][k] = float(np.nanmean(n_sf))
        m, lo, hi = result["impact_reactive"][k]
        print(f"dose {dose}: reactive impact {m:+.3f} [{lo:+.3f},{hi:+.3f}] ticks/unit", flush=True)

    with open(out_dir / f"pnlgap_{args.symbol}.json", "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out_dir}/pnlgap_{args.symbol}.json")
    report(result)


def gen_segments_ctrl(bseeds, args, model, generate_batch, device, dtbs, bi):
    import torch
    streams = [list(s) for s in bseeds]
    arrival = [mid_ticks(s) for s in streams]
    fills = [[] for _ in streams]
    for r in range(args.k_children):
        torch.manual_seed(4000 + r + 1009 * bi[0])
        outs, _ = generate_batch(model, streams, args.spacing, 1.0, device)
        trimmed, _ = common_trim([streams[i] + outs[i] for i in range(len(streams))])
        for i in range(len(streams)):
            fills[i].append(ask_ticks(trimmed[i]))  # counterfactual fill, no trade placed
            streams[i] = trimmed[i]
    return fills, arrival, streams


def naive_arm(data, seed_ends, args):
    """Executor against the RECORDED future: fills at real touch, K checkpoints
    spaced ~spacing tokens apart. Market unaffected by our trading."""
    fills, arrival = [], []
    for e in seed_ends:
        arrival.append(mid_ticks(np.asarray(data[e - SEED_LEN:e])))
        asks = []
        for r in range(1, args.k_children + 1):
            pref = np.asarray(data[e - SEED_LEN:e + r * args.spacing])
            a = ask_ticks(pref)
            if a is not None:
                asks.append(a)
        fills.append(np.mean(asks) if asks else np.nan)
    return fills, arrival


def boot(vals, n_boot=2000, seed=13):
    vals = vals[~np.isnan(vals)]
    rng = np.random.default_rng(seed)
    if not len(vals):
        return [float("nan")] * 3
    b = np.mean(vals[rng.integers(0, len(vals), (n_boot, len(vals)))], axis=1)
    return [float(np.mean(vals)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def sqrt_fit(doses, impacts):
    """Impact ~ Y * sqrt(size); child size ~ SIZE_REF * 2^(bucket-0.5). Fit Y by LS."""
    size = tk.SIZE_REF * 2.0 ** (np.array(doses, dtype=float) - 0.5)
    q = np.sqrt(size)
    y = np.array([impacts[str(d)][0] for d in doses])
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return None, None
    Y = float(np.linalg.lstsq(q[ok, None], y[ok], rcond=None)[0][0])
    resid = y[ok] - Y * q[ok]
    ss = float(1 - (resid ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()) if ok.sum() > 2 else float("nan")
    return Y, ss


def report(result):
    doses = result["doses"]
    print(f"\n=== PnL gap, {result['symbol']} (K={result['k']} children) ===")
    print("dose | reactive impact [CI]      | naive | react/ctrl/naive shortfall (ticks)")
    for d in doses:
        m, lo, hi = result["impact_reactive"][str(d)]
        print(f" {d:>3} | {m:+7.3f} [{lo:+.3f},{hi:+.3f}] | {0.0:+.3f} | "
              f"{result['shortfall_reactive'][str(d)]:+.2f} / {result['shortfall_ctrl'][str(d)]:+.2f} / "
              f"{result['shortfall_naive'][str(d)]:+.2f}")
    Y, ss = sqrt_fit(doses, result["impact_reactive"])
    if Y is not None:
        print(f"\nsqrt-law fit: impact = {Y:.4f}*sqrt(size)  R^2={ss:.3f}")
    print("\nreading: naive impact = 0 (replay's blind spot); reactive impact is the "
          "\ncost replay hid. Gap = reactive - naive = reactive impact.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", default="data/tokens_v2/val.bin")
    p.add_argument("--ckpt", default="checkpoints/pod/v2.pt")
    p.add_argument("--symbol", default="SOLUSDT")
    p.add_argument("--seeds", type=int, default=128)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--k-children", type=int, default=8)
    p.add_argument("--spacing", type=int, default=250)
    p.add_argument("--doses", default="6,9,12,15")
    p.add_argument("--out", default="eval_out/pnl")
    p.add_argument("--span", default="0,1",
                   help="fractional sub-span of the region to sample, e.g. 0.5,1")
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        with open(out_dir / f"pnlgap_{args.symbol}.json") as f:
            report(json.load(f))
    else:
        run(args, out_dir)


if __name__ == "__main__":
    main()
