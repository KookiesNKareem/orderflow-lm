"""Train the decoder-only order-flow transformer."""

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm.model import GPT, Config  # noqa: E402


def get_batch(data: np.ndarray, batch: int, block: int, device: str):
    ix = np.random.randint(0, len(data) - block - 1, batch)
    x = torch.from_numpy(np.stack([data[i:i + block] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1:i + block + 1] for i in ix]).astype(np.int64))
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, data, batch, block, device, amp, iters=40):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, batch, block, device)
        with amp():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--block", type=int, default=1024)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    # defaults are the tuned optima from the optim-study sweep (same hidden shapes)
    p.add_argument("--lr", type=float, default=None, help="adamw: 5e-4, muon: 4e-2")
    p.add_argument("--lr-aux", type=float, default=5e-4,
                   help="AdamW lr for embeddings/norms/biases when using muon")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--out", default="checkpoints/v0.pt")
    p.add_argument("--ckpt-every", type=int, default=500,
                   help="steps between full resumable checkpoints (<out>.last.pt)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>.last.pt if it exists (preemptible hosts)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = (lambda: torch.autocast(device_type=device, dtype=torch.bfloat16)) if args.bf16 \
        else contextlib.nullcontext
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data = np.memmap(args.train, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val, dtype=np.uint16, mode="r")
    print(f"device={device} train={len(train_data):,} val={len(val_data):,} tokens", flush=True)

    with open(args.train + ".meta.json") as f:
        vocab = json.load(f)["vocab"]
    cfg = Config(vocab=vocab, block=args.block, n_layer=args.n_layer,
                 n_head=args.n_head, d=args.d)
    model = GPT(cfg).to(device)
    print(f"params: {model.n_params() / 1e6:.1f}M  uniform-baseline loss: {math.log(cfg.vocab):.3f}", flush=True)

    if args.lr is None:
        args.lr = 4e-2 if args.optimizer == "muon" else 5e-4
    if args.optimizer == "muon":
        from oflm.muon import Muon
        is_hidden = lambda n, p: p.ndim == 2 and not any(k in n for k in ("tok", "pos", "head"))
        hidden = [p for n, p in model.named_parameters() if is_hidden(n, p)]
        aux = [p for n, p in model.named_parameters() if not is_hidden(n, p)]
        opts = [Muon(hidden, lr=args.lr, momentum=0.95),
                torch.optim.AdamW(aux, lr=args.lr_aux, betas=(0.9, 0.95), weight_decay=0.1)]
        print(f"muon on {sum(p.numel() for p in hidden)/1e6:.1f}M hidden params, "
              f"adamw on {sum(p.numel() for p in aux)/1e6:.1f}M aux params", flush=True)
    else:
        opts = [torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)]
    for o in opts:
        for g in o.param_groups:
            g["base_lr"] = g["lr"]

    def lr_factor(step):
        if step < args.warmup:
            return step / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    last_path = Path(args.out).with_suffix(".last.pt")
    best_val = float("inf")
    start = 1
    if args.resume and last_path.exists():
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        for o, so in zip(opts, st["opts"]):
            o.load_state_dict(so)
        best_val, start = st["best_val"], st["step"] + 1
        np.random.set_state(st["np_rng"])
        torch.set_rng_state(st["torch_rng"].cpu())
        if device == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(st["cuda_rng"].cpu())
        print(f"resumed from {last_path} at step {st['step']} (best val {best_val:.4f})", flush=True)

    def save_last(step):
        # write-then-rename so a preemption mid-save never corrupts the last good state
        tmp = last_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opts": [o.state_dict() for o in opts],
                    "config": cfg.__dict__, "step": step, "best_val": best_val,
                    "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
                    "args": vars(args)}, tmp)
        tmp.replace(last_path)

    t0 = time.time()
    for step in range(start, args.steps + 1):
        f = lr_factor(step)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * f
        x, y = get_batch(train_data, args.batch, args.block, device)
        with amp():
            _, loss = model(x, y)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()

        if step % 50 == 0:
            tps = (step - start + 1) * args.batch * args.block / (time.time() - t0)
            print(f"step {step}: train_loss {loss.item():.4f}  ({tps:,.0f} tok/s)", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            vl = eval_loss(model, val_data, args.batch, args.block, device, amp)
            print(f"step {step}: VAL {vl:.4f}", flush=True)
            if vl < best_val:
                best_val = vl
                torch.save({"model": model.state_dict(), "config": cfg.__dict__,
                            "step": step, "val_loss": vl}, args.out)
        if step % args.ckpt_every == 0 or step == args.steps:
            save_last(step)
    print(f"done. best val {best_val:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
