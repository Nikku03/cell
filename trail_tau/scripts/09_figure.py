#!/usr/bin/env python3
"""Figures. NOTE: the two figures the task requests (correlation decay; ratio vs gap) cannot be
produced, because tau was not measurable. Producing them would require inventing data. What is
produced instead is the design calculation that replaces them."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
F3 = os.path.join(ROOT, "data", "raw", "Codes for Figures", "Figure 3")

# --- Fig 1: measured state distributions (what IS in the data) ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, var, lab in zip(axes, ("pC80", "FLIP0"), ("procaspase-8 (pC8(0))", "c-FLIP (FLIP(0))")):
    for dose, c in zip(("25ng", "50ng", "100ng"), ("#4C72B0", "#DD8452", "#C44E52")):
        for grp, ls in (("R", "-"), ("S", "--")):
            k = f"{var}_values_{grp}_{dose}"
            v = sio.loadmat(os.path.join(F3, k + ".mat"))[k].ravel()
            ax.hist(np.log10(v), bins=30, histtype="step", density=True,
                    color=c, linestyle=ls, label=f"{grp} {dose}")
    ax.set_xlabel(f"log10 {lab}"); ax.set_ylabel("density")
    ax.set_title(f"{lab}\n(single value per cell; no time axis exists)")
axes[0].legend(fontsize=6, ncol=2)
fig.suptitle("What the deposited data contains: one static state per cell, 1324 cells", y=1.0)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_state_distributions.png"), dpi=150)

# --- Fig 2: required sample size (the design calculation) ---
n = np.array([200, 400, 800, 1600, 3200], float)
w95 = np.array([0.77615, 0.47664, 0.32093, 0.24191, 0.15682])
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(n, np.exp(w95), "o-", color="#4C72B0", label="simulated ML estimator")
sl = np.polyfit(np.log(n), np.log(w95), 1)[0]
nn = np.logspace(np.log10(40), np.log10(6000), 100)
ax.loglog(nn, np.exp(np.exp(np.polyval(np.polyfit(np.log(n), np.log(w95), 1), np.log(nn)))),
          "-", color="#999999", lw=1, label=f"power law, slope {sl:.2f}")
for tgt, c in ((5.0, "#C44E52"), (2.0, "#DD8452"), (1.5, "#55A868")):
    ax.axhline(tgt, ls=":", color=c, lw=1)
    ax.text(4200, tgt*1.02, f"factor {tgt:g}", color=c, fontsize=8, ha="right")
ax.set_xlabel("number of sister pairs"); ax.set_ylabel("width of 95% CI on tau (factor)")
ax.set_title("Route A design: sample size needed to MEASURE tau\n"
             "(design scaled in units of tau, so this holds without knowing tau)")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_required_sample_size.png"), dpi=150)
print("wrote", os.listdir(FIG))
