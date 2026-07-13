"""Blog figures from eval_out/deep_*.json, styled to match the site's existing
matplotlib SVGs: transparent background, #8a929c ink, series green/amber."""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

INK = "#8a929c"
GEN = "#3fa96b"
REAL = "#6f7780"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

matplotlib.rcParams.update({
    "text.color": INK, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "path", "figure.facecolor": "none", "axes.facecolor": "none",
})

data = {s: json.load(open(f"eval_out/deep_{s}.json")) for s in SYMS}


def style(ax):
    ax.tick_params(length=3, width=0.8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)


# ---- Figure: impact response ----
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
for ax, s in zip(axes, SYMS):
    d = data[s]["impact"]
    ax.plot(d["lags"], d["real"], color=REAL, lw=2, marker="o", ms=3.5, label="real")
    ax.plot(d["lags"], d["gen"], color=GEN, lw=2, marker="o", ms=3.5, label="generated")
    ax.set_xscale("log")
    ax.set_title(s.replace("USDT", " perp"), color=INK, fontsize=11)
    ax.set_xlabel("lag (frames, ~26 ms)")
    style(ax)
axes[0].set_ylabel("impact R(ℓ), ticks")
axes[0].legend(frameon=False, loc="upper left", fontsize=9)
axes[0].annotate("generated impact\nkeeps climbing", xy=(150, 140), fontsize=8.5,
                 color=INK, style="italic", ha="right")
fig.tight_layout()
fig.savefig(OUT / "oflm-impact.svg", transparent=True)
print("wrote oflm-impact.svg")

# ---- Figure: order-sign ACF ----
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
for ax, s in zip(axes, SYMS):
    d = data[s]["sign_acf"]
    for key, col, lbl in (("real", REAL, "real"), ("gen", GEN, "generated")):
        pts = [(l, a) for l, a in zip(d["lags"], d[key]) if a > 0]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2,
                marker="o", ms=3.5, label=lbl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(s.replace("USDT", " perp"), color=INK, fontsize=11)
    ax.set_xlabel("lag (trades)")
    ax.text(0.05, 0.14, f"γ real {d['gamma_real']:.2f}", transform=ax.transAxes,
            fontsize=9, color=REAL)
    ax.text(0.05, 0.04, f"γ gen  {d['gamma_gen']:.2f}", transform=ax.transAxes,
            fontsize=9, color=GEN)
    style(ax)
axes[0].set_ylabel("sign autocorrelation")
axes[0].legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "oflm-signacf.svg", transparent=True)
print("wrote oflm-signacf.svg")
