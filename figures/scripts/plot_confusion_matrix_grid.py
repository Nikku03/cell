"""Plot the confusion matrix at each full-panel detector version.

Reads figures/data/confusion_matrices.csv and produces a 2x3 grid of
heatmaps showing how true positives grew (96 -> 287 over v10-v15) while
false positives stayed flat at 3 from v12 onwards. The grid makes the
"high-precision, growing recall" story visible at a glance.

Excludes the breuer_2019_FBA reference row since we don't have its
confusion matrix breakdown.

Usage:
    python figures/scripts/plot_confusion_matrix_grid.py

Output:
    figures/output/confusion_matrices.png
    figures/output/confusion_matrices.pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = REPO_ROOT / "figures/data/confusion_matrices.csv"
OUT_DIR = REPO_ROOT / "figures/output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df = df[df["tp"].notna()].reset_index(drop=True)

n_versions = len(df)
n_cols = 3
n_rows = (n_versions + n_cols - 1) // n_cols

fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(3.0 * n_cols, 3.2 * n_rows),
)
axes = np.array(axes).reshape(-1)

for idx, row in df.iterrows():
    ax = axes[idx]
    cm = np.array([
        [int(row["tn"]), int(row["fp"])],
        [int(row["fn"]), int(row["tp"])],
    ])
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() * 0.5 else "#333333"
            ax.text(
                j, i, f"{val}",
                ha="center", va="center",
                fontsize=12, color=color, fontweight="bold",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pred non", "pred ess"], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["actual non", "actual ess"], fontsize=9)
    ax.set_title(
        f"{row['version']}  MCC={row['mcc']:.3f}\n"
        f"prec={row['precision']:.3f}  rec={row['recall']:.3f}",
        fontsize=10,
    )

# Hide unused panels.
for k in range(len(df), len(axes)):
    axes[k].set_visible(False)

fig.suptitle(
    "Confusion matrices across detector versions "
    "(Syn3A, n=455 Breuer panel)",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrices.png", dpi=150)
plt.savefig(OUT_DIR / "confusion_matrices.pdf")
print(f"wrote {OUT_DIR}/confusion_matrices.{{png,pdf}}")
