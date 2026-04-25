"""Plot the contribution of each detector to v15's MCC 0.537.

Reads figures/data/detector_contributions.csv and produces a bar chart
showing isolated MCC for each detector that has a measured isolated
number, plus an annotation for the composed v15 result.

The point of the figure: trajectory-only detectors (PerRule, Redundancy
Aware) hit a measured ceiling around MCC 0.13 on small panels;
knowledge-based detectors (ComplexAssembly, AnnotationClass) carry the
weight of the v15 result; the supervised-ML XGBoost stack on Tier-1
features (ESM-2 + ESMFold + MACE-OFF) does NOT improve on v15 — it
falsifies at MCC 0.443 union, 0.145 features-only.

Usage:
    python figures/scripts/plot_detector_contributions.py

Output:
    figures/output/detector_contributions.png
    figures/output/detector_contributions.pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = REPO_ROOT / "figures/data/detector_contributions.csv"
OUT_DIR = REPO_ROOT / "figures/output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
# Keep only rows with numeric isolated_mcc.
df["isolated_mcc"] = pd.to_numeric(df["isolated_mcc"], errors="coerce")
df = df.dropna(subset=["isolated_mcc"]).reset_index(drop=True)

# Color by role: trajectory = grey, knowledge = blue, composed = green,
# supervised ML = red (the falsified one).
def _color(row):
    role = str(row["role"])
    if "composed" in role or "union" in role:
        return "#2ca02c"
    if "knowledge" in role:
        return "#1f77b4"
    if "supervised" in role or "ML" in role:
        return "#d62728"
    return "#888888"

df["color"] = df.apply(_color, axis=1)

fig, ax = plt.subplots(figsize=(9.0, 5.0))
bars = ax.bar(
    df["detector_name"], df["isolated_mcc"],
    color=df["color"], edgecolor="black", linewidth=0.5,
)

# v15 baseline reference line at the headline MCC
ax.axhline(
    y=0.537, color="#2ca02c", linestyle=":",
    linewidth=1.5, alpha=0.85,
    label="v15 composed (MCC 0.537, n=455)",
)
# Breuer FBA benchmark
ax.axhline(
    y=0.59, color="#d62728", linestyle=":",
    linewidth=1.5, alpha=0.85,
    label="Breuer 2019 FBA (MCC 0.59)",
)

# Annotate each bar with its panel size
for bar, panel in zip(bars, df["isolated_panel"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.01,
        f"n={panel}",
        ha="center", fontsize=8, color="#444444",
    )

ax.set_xlabel("Detector", fontsize=11)
ax.set_ylabel("Isolated MCC", fontsize=11)
ax.set_title(
    "Isolated detector performance vs. composed v15\n"
    "(grey = trajectory-only; blue = knowledge-based; "
    "red = supervised ML, falsified)",
    fontsize=12,
)
ax.set_ylim(0, 0.7)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

plt.tight_layout()
plt.savefig(OUT_DIR / "detector_contributions.png", dpi=150)
plt.savefig(OUT_DIR / "detector_contributions.pdf")
print(f"wrote {OUT_DIR}/detector_contributions.{{png,pdf}}")
