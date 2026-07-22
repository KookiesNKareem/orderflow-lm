"""Build figures for the replay-gap experiment."""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

INK = "#8a929c"
GEN = "#3fa96b"
REAL = "#6f7780"
AMBER = "#d98c30"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

matplotlib.rcParams.update({
    "text.color": INK, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "path", "figure.facecolor": "none", "axes.facecolor": "none",
})


def style(ax):
    ax.tick_params(length=3, width=0.8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)


pnl = json.load(open("eval_out/pnl/pnlgap_SOLUSDT.json"))
siz = json.load(open("eval_out/pnl/sizeimpact_SOLUSDT.json"))
doses = pnl["doses"]
relsize = [2.0 ** (d - doses[0]) for d in doses]

# ---- Figure 1: the PnL gap — impact vs size, reactive vs naive ----
fig, ax = plt.subplots(figsize=(5.6, 3.6))
imp = [pnl["impact_reactive"][str(d)][0] for d in doses]
lo = [pnl["impact_reactive"][str(d)][1] for d in doses]
hi = [pnl["impact_reactive"][str(d)][2] for d in doses]
err = [np.subtract(imp, lo), np.subtract(hi, imp)]
ax.errorbar(relsize, imp, yerr=err, color=GEN, lw=2, marker="o", ms=5,
            capsize=3, label="reactive simulator")
ax.axhline(0, color=REAL, lw=2, ls="--", label="naive replay (assumes 0)")
# floor + sqrt fit overlay
xs = np.linspace(min(relsize), max(relsize), 100)
ax.plot(xs, 0.248 + 0.0174 * np.sqrt(xs), color=INK, lw=1, ls=":",
        label="floor + √ fit")
ax.set_xscale("log")
ax.set_xlabel("child order size (× median trade)")
ax.set_ylabel("execution impact, ticks per unit")
ax.set_title("Execution impact rises with order size", color=INK, fontsize=11)
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
ax.annotate("cost replay never charges", xy=(relsize[-1], imp[-1]),
            xytext=(relsize[1], 0.62), fontsize=8.5, color=INK, style="italic")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "oflm-pnl-gap.svg", transparent=True)
print("wrote oflm-pnl-gap.svg")

# ---- Figure 2: real vs generated size scaling (the validation) ----
fig, ax = plt.subplots(figsize=(5.6, 3.6))
b = siz["buckets"]
rsize = [2.0 ** (x - b[0]) for x in b]
rr = [siz["real"][str(x)]["ref"][0] for x in b]
rr_e = [[siz["real"][str(x)]["ref"][0] - siz["real"][str(x)]["ref"][1] for x in b],
        [siz["real"][str(x)]["ref"][2] - siz["real"][str(x)]["ref"][0] for x in b]]
gg = [siz["gen"][str(x)]["ref"][0] for x in b]
gg_e = [[siz["gen"][str(x)]["ref"][0] - siz["gen"][str(x)]["ref"][1] for x in b],
        [siz["gen"][str(x)]["ref"][2] - siz["gen"][str(x)]["ref"][0] for x in b]]
ax.errorbar(rsize, rr, yerr=rr_e, color=REAL, lw=2, marker="o", ms=5, capsize=3,
            label=f"real (β={siz['slope_real']:+.2f})")
ax.errorbar(rsize, gg, yerr=gg_e, color=GEN, lw=2, marker="o", ms=5, capsize=3,
            label=f"model (β={siz['slope_gen']:+.2f})")
ax.set_xscale("log")
ax.set_ylim(0, 1.5)
ax.set_xlabel("trade size (× smallest bucket)")
ax.set_ylabel(f"response R(ℓ={siz['ref_lag']}), ticks")
ax.set_title("Real markets are flat in size too", color=INK, fontsize=11)
ax.legend(frameon=False, loc="lower left", fontsize=8.5)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "oflm-size-scaling.svg", transparent=True)
print("wrote oflm-size-scaling.svg")
