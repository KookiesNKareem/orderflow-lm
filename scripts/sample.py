"""Sample from a trained checkpoint and decode tokens back into events.

Seeds with a real context window from a tokens .bin, generates a continuation,
and prints both a human-readable event log and summary stats (event mix, grammar
violations, implied simulated wall-clock time).

Usage:
  python scripts/sample.py --ckpt checkpoints/v0.pt --seed-bin data/tokens/val.bin \
      --new-tokens 2000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm.model import GPT, Config  # noqa: E402
from oflm import tokenizer as tk  # noqa: E402
from oflm.generate import generate  # noqa: E402


def decode(tokens: list[int]) -> tuple[list[str], dict]:
    """Token stream -> readable event lines + stats. Tolerant of grammar errors."""
    lines = []
    stats = {"trades": 0, "frames": 0, "changes": 0, "violations": 0, "sim_us": 0}
    # the stream may begin mid-event (seed cut mid-frame): sync to the first Δt token
    i = 0
    n = len(tokens)
    while i < n and not (tk.DT0 <= tokens[i] < tk.DT0 + 32):
        i += 1
    stats["leading_partial"] = i

    def dt_mid(b):  # bucket -> representative µs (geometric mid)
        return 0 if b == 0 else int(2 ** (b - 0.5))

    while i < n:
        t = tokens[i]
        if t == tk.BOS or tk.SYM0 <= t < tk.SYM0 + 8:  # day/symbol separators
            i += 1
            continue
        if not (tk.DT0 <= t < tk.DT0 + 32):
            stats["violations"] += 1
            i += 1
            continue
        dt = dt_mid(t - tk.DT0)
        stats["sim_us"] += dt
        i += 1
        if i >= n:
            break
        head = tokens[i]
        if head in (tk.TRADE_SELL, tk.TRADE_BUY):
            if i + 2 >= n:
                break
            off = tokens[i + 1] - tk.TOFF0 - 15
            sz = tokens[i + 2] - tk.TSIZE0
            ok = (tk.TOFF0 <= tokens[i + 1] < tk.TOFF0 + 32
                  and tk.TSIZE0 <= tokens[i + 2] < tk.TSIZE0 + 32)
            if not ok:
                stats["violations"] += 1
            side = "BUY " if head == tk.TRADE_BUY else "SELL"
            lines.append(f"+{dt:>8}µs TRADE {side} off={off:+d} szb={sz}")
            stats["trades"] += 1
            i += 3
        elif head == tk.FRAME:
            i += 1
            stats["frames"] += 1
            chgs = []
            while i + 1 < n:
                c = tokens[i]
                if tk.CHG_BID0 <= c < tk.CHG_ASK0 + 80:
                    side, off = ("bid", c - tk.CHG_BID0 - 16) if c < tk.CHG_ASK0 else ("ask", c - tk.CHG_ASK0 - 16)
                    szb = tokens[i + 1] - tk.FSIZE0
                    if not (tk.FSIZE0 <= tokens[i + 1] < tk.FSIZE0 + 32):
                        stats["violations"] += 1
                    chgs.append(f"{side}{off:+d}={'DEL' if szb == 0 else szb}")
                    stats["changes"] += 1
                    i += 2
                elif c in (tk.JUMP_BID, tk.JUMP_ASK) and i + 3 < n:
                    hi, lo, szt = tokens[i + 1], tokens[i + 2], tokens[i + 3]
                    ok = all(tk.DIGIT0 <= d < tk.DIGIT0 + 32 for d in (hi, lo)) \
                        and tk.FSIZE0 <= szt < tk.FSIZE0 + 32
                    if not ok:
                        stats["violations"] += 1
                    mag = 17 + (hi - tk.DIGIT0) * 32 + (lo - tk.DIGIT0)
                    side = "bid" if c == tk.JUMP_BID else "ask"
                    chgs.append(f"{side}JUMP-{mag}={tokens[i + 3] - tk.FSIZE0}")
                    stats["jumps"] = stats.get("jumps", 0) + 1
                    i += 4
                else:
                    break
            if not chgs:
                stats["violations"] += 1
            lines.append(f"+{dt:>8}µs FRAME {' '.join(chgs)}")
        elif head == tk.BOS:
            lines.append("--- BOS ---")
            i += 1
        else:
            stats["violations"] += 1
            i += 1
    return lines, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/v0.pt")
    p.add_argument("--seed-bin", required=True)
    p.add_argument("--seed-at", type=int, default=-1, help="offset into bin; -1 = random")
    p.add_argument("--new-tokens", type=int, default=2000)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--show", type=int, default=40)
    p.add_argument("--raw", action="store_true",
                   help="disable grammar constraints (diagnostic: measures native violation rate)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = Config(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"ckpt step {ckpt['step']}  val_loss {ckpt['val_loss']:.4f}")

    data = np.memmap(args.seed_bin, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    at = rng.integers(0, len(data) - cfg.block) if args.seed_at < 0 else args.seed_at
    ctx = np.asarray(data[at:at + cfg.block])

    import time
    t0 = time.time()
    gen, gstats = generate(model, ctx, args.new_tokens, args.temp, device,
                           constrained=not args.raw)
    dt = time.time() - t0
    lines, stats = decode(gen)
    print(f"\ngenerated {args.new_tokens} tokens -> {len(lines)} events "
          f"({args.new_tokens / dt:,.0f} tok/s)")
    print(f"decode stats: {stats}  (sim time: {stats['sim_us'] / 1e6:.2f}s)")
    print(f"generation stats: {gstats}")
    print(f"\nfirst {args.show} events:")
    for ln in lines[:args.show]:
        print(" ", ln)


if __name__ == "__main__":
    main()
