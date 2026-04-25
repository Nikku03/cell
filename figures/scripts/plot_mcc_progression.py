"""Plot the MCC trajectory across detector versions.

Reads figures/data/mcc_progression.csv and produces a publication-quality
line plot showing how MCC evolved as different detector strategies were
layered on. Two lines: one for small-panel (n=40) measurements, one for
full-panel (n=455). The Breuer 2019 FBA benchmark (MCC 0.59) is drawn as
a horizontal reference line.

Usage:
    python figures/scripts/plot_mcc_progression.py

Output:
    figures/output/mcc_progression.png  (created on first run)
    figures/output/mcc_progression.pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = REPO_ROOT / "figures/data/mcc_progression.csv"
OUT_DIR = REPO_ROOT / "figures/output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df_full = df[df["n_panel"] == 455].copy()
df_small = df[df["n_panel"] == 40].copy()

fig, ax = plt.subplots(figsize=(8.5, 5.0))

ax.plot(
    df_full["version"], df_full["mcc"],
    marker="o", linewidth=2.0, color="#1f77b4",
    label="Full Breuer panel (n=455)",
)
ax.plot(
    df_small["version"], df_small["mcc"],
    marker="s", linewidth=1.0, color="#888888",
    linestyle="--", alpha=0.7,
    label="Small panel (n=40, calibration only)",
)

# Breuer 2019 FBA benchmark
ax.axhline(
    y=0.59, color="#d62728", linestyle=":",
    linewidth=1.5, alpha=0.85,
    label="Breuer 2019 FBA benchmark (MCC 0.59)",
)

# Annotate the v15 endpoint
v15 = df_full[df_full["version"] == "v15"].iloc[0]
ax.annotate(
    f"v15: MCC {v15['mcc']:.3f}\n(287 TP / 3 FP / 69 TN / 96 FN)",
    xy=("v15", v15["mcc"]),
    xytext=(20, -50), textcoords="offset points",
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.7),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#1f77b4"),
)

ax.set_xlabel("Detector version", fontsize=11)
ax.set_ylabel("Matthews Correlation Coefficient", fontsize=11)
ax.set_title(
    "Essentiality prediction MCC across detector iterations\n"
    "(Syn3A vs Breuer 2019 experimental labels)",
    fontsize=12,
)
ax.set_ylim(0, 0.7)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig(OUT_DIR / "mcc_progression.png", dpi=150)
plt.savefig(OUT_DIR / "mcc_progression.pdf")
print(f"wrote {OUT_DIR}/mcc_progression.{{png,pdf}}")
