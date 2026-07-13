"""Apply a token stream (real or generated) to book state.

iter_events parses tokens into events (tolerantly — desyncs skip to next Δt).
Book maintains the near-touch book in integer ticks with the same semantics as
reconstruction: all changes in a frame are priced off the same pre-frame best;
jump escapes place a level `mag` ticks inside; size buckets are decoded to
their geometric midpoints (lossy, by design of the tokenization).

A frame that leaves the book crossed is resolved by pseudo-execution: crossing
levels consume each other smaller-first, mirroring what a matching engine
would have done. Occurrences are counted (`crossed_frames`) as a realism
diagnostic — real reconstructed frames never cross.
"""

import heapq

from . import tokenizer as tk

TICK = 0.1


def size_rep(bucket: int) -> float:
    return 0.0 if bucket == 0 else tk.SIZE_REF * 2 ** (bucket - 0.5)


def dt_rep(bucket: int) -> int:
    return 0 if bucket == 0 else int(2 ** (bucket - 0.5))


def iter_events(tokens):
    """Yield ('trade', dt_bucket, side, toff, size_bucket) and
    ('frame', dt_bucket, [(side, offset_ticks, size_bucket), ...]).
    side: 0=bid/sell, 1=ask/buy. Jump changes appear as offset < -16."""
    import numpy as np
    if isinstance(tokens, np.ndarray) and tokens.dtype != np.int64:
        tokens = tokens.astype(np.int64)  # uint16 arithmetic would wrap
    n = len(tokens)
    i = 0
    while i < n and not (tk.DT0 <= tokens[i] < tk.DT0 + 32):
        i += 1
    while i < n - 1:
        t = tokens[i]
        if not (tk.DT0 <= t < tk.DT0 + 32):
            i += 1
            continue
        dt = t - tk.DT0
        head = tokens[i + 1]
        if head in (tk.TRADE_SELL, tk.TRADE_BUY) and i + 3 < n:
            yield ("trade", dt, 1 if head == tk.TRADE_BUY else 0,
                   tokens[i + 2] - tk.TOFF0 - 15, tokens[i + 3] - tk.TSIZE0)
            i += 4
        elif head == tk.FRAME:
            i += 2
            changes = []
            while i < n:
                c = tokens[i]
                if tk.CHG_BID0 <= c < tk.CHG_ASK0 + 80 and i + 1 < n:
                    side, off = (0, c - tk.CHG_BID0 - 16) if c < tk.CHG_ASK0 else (1, c - tk.CHG_ASK0 - 16)
                    changes.append((side, off, tokens[i + 1] - tk.FSIZE0))
                    i += 2
                elif c in (tk.JUMP_BID, tk.JUMP_ASK) and i + 3 < n:
                    mag = 17 + (tokens[i + 1] - tk.DIGIT0) * 32 + (tokens[i + 2] - tk.DIGIT0)
                    changes.append((0 if c == tk.JUMP_BID else 1, -mag, tokens[i + 3] - tk.FSIZE0))
                    i += 4
                else:
                    break
            if changes:
                yield ("frame", dt, changes)
        else:
            i += 2  # desync: scan forward for next Δt


ANCHOR = 10**6  # arbitrary: the stream is offset-only, so absolute ticks are free


class Book:
    def __init__(self):
        self.levels = ({}, {})            # tick price -> size, [bid, ask]
        self.heaps = ([], [])             # lazy heaps: bid max (neg), ask min
        self.crossed_frames = 0
        # bootstrap anchor; churn at the touch replaces these within a few frames
        self._set(0, ANCHOR, 1e-9)
        self._set(1, ANCHOR + 1, 1e-9)

    def best(self, side: int):
        levels, heap = self.levels[side], self.heaps[side]
        while heap:
            p = -heap[0] if side == 0 else heap[0]
            if p in levels:
                return p
            heapq.heappop(heap)
        return None

    def _set(self, side: int, price: int, size: float):
        if size == 0.0:
            self.levels[side].pop(price, None)
        else:
            if price not in self.levels[side]:
                heapq.heappush(self.heaps[side], -price if side == 0 else price)
            self.levels[side][price] = size

    def apply_frame(self, changes) -> None:
        pre = (self.best(0), self.best(1))
        if pre[0] is None or pre[1] is None:  # one side emptied out entirely
            pre = (pre[0] or (pre[1] - 1 if pre[1] else ANCHOR),
                   pre[1] or (pre[0] + 1 if pre[0] else ANCHOR + 1))
        for side, off, szb in changes:
            base = pre[side]
            price = base - off if side == 0 else base + off
            self._set(side, price, size_rep(szb))
        # resolve crossed state by pseudo-execution, smaller level consumed first
        crossed = False
        while True:
            bb, ba = self.best(0), self.best(1)
            if bb is None or ba is None or bb < ba:
                break
            crossed = True
            if self.levels[0][bb] <= self.levels[1][ba]:
                self._set(0, bb, 0.0)
            else:
                self._set(1, ba, 0.0)
        self.crossed_frames += crossed

    def spread_ticks(self):
        bb, ba = self.best(0), self.best(1)
        return None if bb is None or ba is None else ba - bb

    def depth_at_best(self):
        bb, ba = self.best(0), self.best(1)
        if bb is None or ba is None:
            return None
        return self.levels[0][bb] + self.levels[1][ba]


def replay(tokens, warmup_frames: int = 2000):
    """Apply a token stream; per-frame series plus a trade log for impact
    analysis. Trades are indexed to the last recorded frame mid (pre-trade),
    so R(l) = sign * (mids2[i + l] - mids2[i]) / 2 in ticks."""
    import numpy as np
    book = Book()
    spreads, mids2, depths = [], [], []
    trade_idx, trade_signs, trade_sizes = [], [], []
    n_frames = n_trades = 0
    sim_us = 0
    last_mid2 = None
    for ev in iter_events(tokens):
        sim_us += dt_rep(ev[1])
        if ev[0] == "trade":
            n_trades += 1
            if n_frames > warmup_frames and last_mid2 is not None and mids2:
                trade_idx.append(len(mids2) - 1)
                trade_signs.append(1 if ev[2] == 1 else -1)
                trade_sizes.append(ev[4])
            continue
        book.apply_frame(ev[2])
        n_frames += 1
        if n_frames <= warmup_frames:
            continue
        bb, ba = book.best(0), book.best(1)
        if bb is not None and ba is not None:
            last_mid2 = bb + ba
            spreads.append(ba - bb)
            depths.append(book.depth_at_best())
        if last_mid2 is not None:
            mids2.append(last_mid2)  # carry-forward when a side is missing
    return {
        "spreads": np.array(spreads), "mids2": np.array(mids2, dtype=np.int64),
        "depths": np.array(depths), "frames": n_frames, "trades": n_trades,
        "sim_s": sim_us / 1e6, "crossed": book.crossed_frames,
        "trade_idx": np.array(trade_idx, dtype=np.int64),
        "trade_signs": np.array(trade_signs, dtype=np.int64),
        "trade_sizes": np.array(trade_sizes, dtype=np.int64),
    }


def ks(a, b) -> float:
    import numpy as np
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    ca = np.searchsorted(a, grid, side="right") / len(a)
    cb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(ca - cb).max())


def find_regions(tokens):
    """Locate [BOS][SYM] day headers; returns [(sym_id, start, end), ...]."""
    import numpy as np
    from . import tokenizer as tk
    bos = np.flatnonzero(tokens == tk.BOS)
    regions = []
    for i, b in enumerate(bos):
        if b + 1 < len(tokens) and tk.SYM0 <= tokens[b + 1] < tk.SYM0 + 8:
            end = bos[i + 1] if i + 1 < len(bos) else len(tokens)
            regions.append((int(tokens[b + 1]) - tk.SYM0, int(b), int(end)))
    return regions
