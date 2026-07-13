"""Muon: MomentUm Orthogonalized by Newton-schulz (Keller Jordan, 2024).

For each 2D weight, take the SGD-momentum update and replace it with the nearest
semi-orthogonal matrix (via a 5-step Newton-Schulz iteration) before applying it.
This is the standard reference implementation; we keep it faithful so the study
measures Muon, not our variant of it.
"""

import torch


def zeropower_via_newtonschulz5(G, steps: int = 5, eps: float = 1e-7):
    """Orthogonalize G (make its singular values ~1) via a quintic Newton-Schulz
    iteration, run in bfloat16. Coefficients tuned by Keller Jordan so the iteration
    pushes singular values toward 1 from a Frobenius-normalized start."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    # bf16 is a CUDA speed optimization; fp32 elsewhere (MPS bf16 matmuls are shaky)
    X = G.bfloat16() if G.is_cuda else G.float()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                # scale keeps the update's RMS comparable across matrix shapes
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(g.to(p.dtype), alpha=-lr * scale)
