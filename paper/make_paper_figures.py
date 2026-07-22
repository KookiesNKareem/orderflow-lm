"""Build publication figures from curated paper results."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "paper"
EVAL = RESULTS / "distributional"
PILOT = RESULTS / "injection"
PNL = RESULTS / "execution"
TWAP = RESULTS / "execution"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

REAL = "#1a1a1a"
GEN = "#0072B2"
ALT = "#E69F00"
CTRL = "#888888"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
LABEL = {"BTCUSDT": "BTC perp", "ETHUSDT": "ETH perp", "SOLUSDT": "SOL perp"}

matplotlib.rcParams.update({
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.4,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.facecolor": "white",
    "legend.edgecolor": "none",
    "pdf.fonttype": 42,
})

deep = {s: json.load(open(EVAL / f"deep_{s}.json")) for s in SYMS}


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# ---- Fig: impact response R(l), real vs generated, per symbol ----
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3), sharex=True)
for ax, s in zip(axes, SYMS):
    d = deep[s]["impact"]
    ax.plot(d["lags"], d["real"], color=REAL, marker="o", ms=2.8, label="real")
    ax.plot(d["lags"], d["gen"], color=GEN, marker="s", ms=2.8, label="generated")
    ax.set_xscale("log")
    ax.set_title(LABEL[s])
    ax.set_xlabel("lag $\\ell$ (frames)")
axes[0].set_ylabel("$R(\\ell)$ (ticks)")
axes[0].legend(loc="upper left")
save(fig, "impact.pdf")

# ---- Fig: order-sign autocorrelation, log-log ----
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3), sharex=True)
for ax, s in zip(axes, SYMS):
    d = deep[s]["sign_acf"]
    lags = np.asarray(d["lags"], float)
    ax.loglog(lags, d["real"], color=REAL, marker="o", ms=2.8, label="real")
    ax.loglog(lags, d["gen"], color=GEN, marker="s", ms=2.8, label="generated")
    ax.set_title(LABEL[s])
    ax.set_xlabel("lag (trades)")
    ax.text(0.04, 0.06,
            f"$\\gamma$ real {d['gamma_real']:.2f}, gen {d['gamma_gen']:.2f}",
            transform=ax.transAxes, fontsize=7.5)
axes[0].set_ylabel("sign ACF $C(\\ell)$")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, 1.0), fontsize=8)
save(fig, "signacf.pdf", rect=(0, 0, 1, 0.88))

# ---- Fig: injection response across the three symbols ----
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4))
for ax, s in zip(axes, ["SOLUSDT", "ETHUSDT", "BTCUSDT"]):
    p = json.load(open(PILOT / f"pilot_{s}.json"))
    lags = np.asarray(p["lags"], float)
    for key, color, lab in [("paired_d1", GEN, "injected, $1\\times$ size"),
                            ("paired_d2", ALT, "injected, $8\\times$ size")]:
        a = p[key]
        ax.plot(lags, a["mean"], color=color, marker="o", ms=2.8, label=lab)
        ax.fill_between(lags, a["lo"], a["hi"], color=color, alpha=0.16, lw=0)
    c = p["ctrl_drift"]
    ax.plot(lags, c["mean"], color=CTRL, ls="--", lw=1.1, label="no-injection control")
    ax.fill_between(lags, c["lo"], c["hi"], color=CTRL, alpha=0.13, lw=0)
    ax.axhline(0, color=REAL, lw=0.6, alpha=0.5)
    ax.set_xscale("log")
    ax.set_title(LABEL[s])
    ax.set_xlabel("lag $\\ell$ (frames)")
axes[0].set_ylabel("paired response $(R_{buy}-R_{sell})/2$ (ticks)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
save(fig, "injection.pdf", rect=(0, 0, 1, 0.86))

# ---- Fig: plateau-window injected response vs impact kernel, all symbols ----
plateau = {}
for line in open(PILOT / "plateau_window.txt"):
    parts = line.split()
    sym, dose = parts[0], parts[1].rstrip(":")
    m = float(parts[4])
    lo, hi = (float(x) for x in parts[5].strip("[]").split(","))
    plateau.setdefault(sym, {})[dose] = (m, lo, hi)
    plateau[sym]["kernel"] = float(parts[-1])

fig, ax = plt.subplots(figsize=(4.2, 2.7))
xs = np.arange(3)
for j, (dose, color, off, lab) in enumerate([("d1", GEN, -0.12, "injected, $1\\times$ size"),
                                             ("d2", ALT, 0.12, "injected, $8\\times$ size")]):
    m = np.array([plateau[s][dose] for s in SYMS])
    ax.errorbar(xs + off, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]],
                color=color, marker="o", ms=4, ls="none", capsize=2.5, lw=1.2, label=lab)
kern = [plateau[s]["kernel"] for s in SYMS]
for x, k in zip(xs, kern):
    ax.hlines(k, x - 0.3, x + 0.3, color=REAL, lw=1.4,
              label="impact kernel $G$ (real)" if x == 0 else None)
ax.axhline(0, color=REAL, lw=0.6, alpha=0.4)
ax.set_xticks(xs)
ax.set_xticklabels([LABEL[s] for s in SYMS])
ax.set_ylabel("plateau response, lags 16--24 (ticks)")
ax.legend(loc="upper left", fontsize=7.2)
save(fig, "plateau.pdf")

# ---- Fig: dressed response, impact kernel, and injected response (SOL) ----
pp = json.load(open(PILOT / "propagator_SOLUSDT.json"))
lags64 = np.arange(1, len(pp["G_real"]) + 1)
lags_inj = np.asarray(pp["lags_inj"], float)
g = np.asarray(pp["G_real"])
gsd = np.asarray(pp["G_real_sd"])
inj = pp["inj_d1"]

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
ax = axes[0]
ax.plot(lags64, pp["R_real"], color=REAL, label="dressed response $R(\\ell)$ (real)")
ax.plot(lags64[:48], g[:48], color=REAL, ls="--", lw=1.1,
        label="impact kernel $G(\\ell)$ (deconvolved)")
ax.plot(lags_inj, inj["mean"], color=GEN, marker="o", ms=2.6,
        label="injected response (model)")
ax.axhline(0, color=REAL, lw=0.6, alpha=0.5)
ax.set_xlabel("lag $\\ell$ (trades)")
ax.set_ylabel("response (ticks)")
ax.legend(loc="upper left", fontsize=7.2)

ax = axes[1]
ax.plot(lags64[:32], g[:32], color=REAL, ls="--", lw=1.1, label="impact kernel $G(\\ell)$")
ax.fill_between(lags64[:32], (g - gsd)[:32], (g + gsd)[:32], color=REAL, alpha=0.13, lw=0)
ax.plot(lags_inj, inj["mean"], color=GEN, marker="o", ms=2.6, label="injected response")
ax.fill_between(lags_inj, inj["lo"], inj["hi"], color=GEN, alpha=0.16, lw=0)
ax.axhline(0, color=REAL, lw=0.6, alpha=0.5)
ax.set_ylim(-0.06, 0.20)
ax.set_xlabel("lag $\\ell$ (trades)")
ax.legend(loc="upper left")
save(fig, "kernel.pdf")

# ---- Fig: replay gap — TWAP impact vs order size ----
pnl = json.load(open(TWAP / "pnlgap_SOLUSDT.json"))
rel = np.array([1, 8, 64, 512], float)
imp = np.array([pnl["impact_reactive"][str(d)] for d in pnl["doses"]])

fig, ax = plt.subplots(figsize=(3.6, 2.7))
ax.errorbar(rel, imp[:, 0], yerr=[imp[:, 0] - imp[:, 1], imp[:, 2] - imp[:, 0]],
            color=GEN, marker="o", ms=4, ls="none", capsize=2.5, lw=1.2,
            label="reactive simulator")
ax.axhline(0, color=REAL, lw=1.1, ls="--", label="naive replay (zero by construction)")
ax.set_xscale("log", base=2)
ax.set_xticks(rel)
ax.set_xticklabels(["$\\frac{1}{8}\\times$", "$1\\times$", "$8\\times$", "$64\\times$"])
ax.set_xlabel("order size (multiple of median trade)")
ax.set_ylabel("quote-response cost (ticks/unit)")
ax.legend(loc="upper left", fontsize=7.2)
save(fig, "pnlgap.pdf")

# ---- Fig: size-conditioned response, real vs model ----
si = json.load(open(PNL / "sizeimpact_SOLUSDT.json"))
fig, ax = plt.subplots(figsize=(3.6, 2.7))
for src, color, marker, lab in [("real", REAL, "o", "real trades"),
                                ("gen", GEN, "s", "model trades")]:
    m = np.array([si[src][str(b)]["ref"] for b in si["buckets"]])
    ax.errorbar(rel, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]],
                color=color, marker=marker, ms=4, ls="-", lw=1.2, capsize=2.5,
                label=lab)
ax.set_xscale("log", base=2)
ax.set_xticks(rel)
ax.set_xticklabels(["$\\frac{1}{8}\\times$", "$1\\times$", "$8\\times$", "$64\\times$"])
ax.set_ylim(0, 1.5)
ax.set_xlabel("trade size (multiple of median)")
ax.set_ylabel("$R(20\\,|\\,\\mathrm{size})$ (ticks)")
ax.text(0.04, 0.10, f"$\\beta$ real ${si['slope_real']:+.2f}$, model ${si['slope_gen']:+.2f}$",
        transform=ax.transAxes, fontsize=7.5)
ax.legend(loc="upper left", fontsize=7.5)
save(fig, "size.pdf")
