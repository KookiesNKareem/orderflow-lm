"""Decoder-only transformer for order-flow token streams. Deliberately vanilla
(pre-norm GPT, learned positions, SDPA) so eval results reflect the data
representation, not architecture novelty."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 384
    block: int = 1024
    n_layer: int = 8
    n_head: int = 8
    d: int = 512
    dropout: float = 0.0


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n_head = c.n_head
        self.qkv = nn.Linear(c.d, 3 * c.d)
        self.proj = nn.Linear(c.d, c.d)
        self.dropout = c.dropout

    def forward(self, x, past=None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (z.view(b, t, self.n_head, d // self.n_head).transpose(1, 2) for z in (q, k, v))
        if past is not None:
            # incremental decode: single new token attends to the whole cache
            assert t == 1, "cached forward is single-token; re-prefill for longer inputs"
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=past is None, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(b, t, d)), (k, v)


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d)
        self.attn = Attention(c)
        self.ln2 = nn.LayerNorm(c.d)
        self.mlp = nn.Sequential(
            nn.Linear(c.d, 4 * c.d), nn.GELU(), nn.Linear(4 * c.d, c.d), nn.Dropout(c.dropout))

    def forward(self, x, past=None):
        a, kv = self.attn(self.ln1(x), past)
        x = x + a
        return x + self.mlp(self.ln2(x)), kv


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.block, c.d)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, past=None, use_cache=False):
        b, t = idx.shape
        t_past = past[0][0].size(2) if past is not None else 0
        pos = torch.arange(t_past, t_past + t, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        new_past = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, past[i] if past is not None else None)
            new_past.append(kv)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.c.vocab), targets.reshape(-1))
        if use_cache:
            return logits, loss, new_past
        return logits, loss

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
