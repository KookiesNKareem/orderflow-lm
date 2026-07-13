"""KV-cached, grammar-constrained generation.

GrammarSampler tracks the event grammar as a state machine and returns, at each
step, the set of legal next tokens. Sampling masks logits to that set, so
generated streams are valid by construction: parseable, no duplicate levels
within a frame, frame changes in canonical (side, offset) order. The
probability mass the raw model puts on illegal tokens is accumulated as a
model-quality diagnostic (`illegal_mass`).

Frame ordering rule: change tokens are globally ordered (bid offsets -16..63 =
ids 101..180, then ask offsets = 181..260), and the tokenizer emits changes
sorted by (side, offset) with jump escapes (improvements beyond -16) first on
each side. Enforcing "next change id strictly greater than the last" therefore
yields sortedness, side ordering, and dedup in one rule.
"""

import torch

from . import tokenizer as tk

# grammar states (S_RESYNC: desynced by an illegal token in raw mode; wait for a Δt)
S_DT, S_HEAD, S_TOFF, S_TSIZE, S_CHG_FIRST, S_CHG_MORE, S_JHI, S_JLO, S_FSIZE, S_RESYNC = range(10)


def _mask(*ranges):
    m = torch.zeros(tk.VOCAB, dtype=torch.bool)
    for lo, hi in ranges:
        m[lo:hi] = True
    return m

M_DT = _mask((tk.DT0, tk.DT0 + 32))
M_HEAD = _mask((tk.TRADE_SELL, tk.TRADE_BUY + 1), (tk.FRAME, tk.FRAME + 1))
M_TOFF = _mask((tk.TOFF0, tk.TOFF0 + 32))
M_TSIZE = _mask((tk.TSIZE0, tk.TSIZE0 + 32))
M_FSIZE = _mask((tk.FSIZE0, tk.FSIZE0 + 32))
M_DIGIT = _mask((tk.DIGIT0, tk.DIGIT0 + 32))
CHG_LO, CHG_HI = tk.CHG_BID0, tk.CHG_ASK0 + 80


class GrammarSampler:
    def __init__(self):
        self.state = S_DT
        self.last_chg = 0        # last change-token id in current frame (ordering/dedup)
        self.jump_side = None    # side of a pending jump escape

    def allowed(self) -> torch.Tensor:
        s = self.state
        if s in (S_DT, S_RESYNC):
            return M_DT
        if s == S_HEAD:
            return M_HEAD
        if s == S_TOFF:
            return M_TOFF
        if s == S_TSIZE:
            return M_TSIZE
        if s in (S_CHG_FIRST, S_CHG_MORE):
            m = torch.zeros(tk.VOCAB, dtype=torch.bool)
            m[max(CHG_LO, self.last_chg + 1):CHG_HI] = True
            if self.last_chg < tk.CHG_BID0:          # no bid change yet
                m[tk.JUMP_BID] = True
            if self.last_chg < tk.CHG_ASK0:          # no ask change yet
                m[tk.JUMP_ASK] = True
            if s == S_CHG_MORE:                      # frame already non-empty: may end
                m |= M_DT
            return m
        if s in (S_JHI, S_JLO):
            return M_DIGIT
        return M_FSIZE

    def push(self, t: int) -> None:
        if not self.allowed()[t]:
            # only reachable in raw mode: drop to resync instead of corrupting state
            self.state = S_RESYNC
            return
        s = self.state
        if s in (S_DT, S_RESYNC):
            self.state = S_HEAD
        elif s == S_HEAD:
            if t == tk.FRAME:
                self.state = S_CHG_FIRST
                self.last_chg = 0
            else:
                self.state = S_TOFF
        elif s == S_TOFF:
            self.state = S_TSIZE
        elif s == S_TSIZE:
            self.state = S_DT
        elif s in (S_CHG_FIRST, S_CHG_MORE):
            if tk.DT0 <= t < tk.DT0 + 32:            # frame ended, new event's Δt
                self.state = S_HEAD
            elif t in (tk.JUMP_BID, tk.JUMP_ASK):
                self.jump_side = t
                self.state = S_JHI
            else:                                    # regular change
                self.last_chg = t
                self.state = S_FSIZE
        elif s == S_JHI:
            self.state = S_JLO
        elif s == S_JLO:
            self.state = S_FSIZE
        elif s == S_FSIZE:
            if self.jump_side is not None:
                # jump consumed; keep side ordering: a bid jump precedes bid changes
                if self.jump_side == tk.JUMP_ASK:
                    self.last_chg = max(self.last_chg, tk.CHG_ASK0 - 1)
                self.jump_side = None
            self.state = S_CHG_MORE

    def sync(self, tokens) -> None:
        """Establish state from a seed sequence (starts at its first Δt token)."""
        started = False
        for t in tokens:
            if not started:
                if tk.DT0 <= int(t) < tk.DT0 + 32:
                    started = True
                    self.state = S_DT
                else:
                    continue
            self.push(int(t))


@torch.no_grad()
def generate(model, ctx, n_new: int, temp: float, device: str,
             constrained: bool = True):
    """Returns (tokens, stats). ctx is a 1-D sequence of seed tokens."""
    model.eval()
    block = model.c.block
    sampler = GrammarSampler()
    sampler.sync(ctx.tolist())

    x = torch.tensor(ctx.tolist()[-block:], dtype=torch.long, device=device)[None]
    logits, _, past = model(x, use_cache=True)
    out = []
    illegal_mass = 0.0
    raw_violations = 0

    for _ in range(n_new):
        probs = torch.softmax(logits[0, -1] / temp, dim=-1)
        allowed = sampler.allowed().to(device)
        illegal_mass += probs[~allowed].sum().item()
        if constrained:
            probs = probs * allowed
            probs = probs / probs.sum()
        nxt = torch.multinomial(probs, 1).item()
        if not constrained and not allowed[nxt]:
            raw_violations += 1
        out.append(nxt)
        sampler.push(nxt)

        cache_len = past[0][0].size(2)
        if cache_len + 1 >= block:  # re-prefill on a half-window to keep positions valid
            tail = (ctx.tolist() + out)[-block // 2:]
            x = torch.tensor(tail, dtype=torch.long, device=device)[None]
            logits, _, past = model(x, use_cache=True)
        else:
            x = torch.tensor([[nxt]], dtype=torch.long, device=device)
            logits, _, past = model(x, past=past, use_cache=True)

    stats = {"illegal_mass_avg": illegal_mass / max(1, n_new)}
    if not constrained:
        stats["raw_violations"] = raw_violations
    return out, stats


@torch.no_grad()
def generate_batch(model, seeds, n_new: int, temp: float, device: str):
    """Constrained generation for B equal-length seeds at once. Returns
    (list of B token lists, stats). One GrammarSampler per row; masks are
    stacked so each row is constrained independently."""
    model.eval()
    block = model.c.block
    B = len(seeds)
    samplers = []
    for s in seeds:
        g = GrammarSampler()
        g.sync(list(s))
        samplers.append(g)

    hist = [list(s)[-block:] for s in seeds]
    x = torch.tensor(hist, dtype=torch.long, device=device)
    logits, _, past = model(x, use_cache=True)
    outs = [[] for _ in range(B)]
    illegal_mass = 0.0

    for _ in range(n_new):
        probs = torch.softmax(logits[:, -1] / temp, dim=-1)
        mask = torch.stack([g.allowed() for g in samplers]).to(device)
        illegal_mass += probs.masked_fill(mask, 0).sum(dim=-1).mean().item()
        probs = probs * mask
        probs = probs / probs.sum(dim=-1, keepdim=True)
        nxt = torch.multinomial(probs, 1).squeeze(-1)
        nxt_list = nxt.tolist()
        for b in range(B):
            outs[b].append(nxt_list[b])
            hist[b].append(nxt_list[b])
            samplers[b].push(nxt_list[b])

        if past[0][0].size(2) + 1 >= block:
            tails = [h[-block // 2:] for h in hist]
            x = torch.tensor(tails, dtype=torch.long, device=device)
            logits, _, past = model(x, use_cache=True)
        else:
            x = nxt.unsqueeze(-1)
            logits, _, past = model(x, past=past, use_cache=True)

    return outs, {"illegal_mass_avg": illegal_mass / max(1, n_new)}
