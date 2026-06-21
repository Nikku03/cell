"""Plot the v1 BCE deployment Pareto: coverage vs precision.

Reads outputs/orphan/af_torch_preds_aug.npz (transformer p + brightness) and
outputs/orphan/smart_cooccur_results_aug.json (the GBM rescue tradeoff curve)
and overlays two curves on one plot:

  - Transformer alone:  sweep brightness threshold, report coverage at each
    threshold and the precision over the called set.
  - Combined (+GBM rescue): the per-tau curve already saved by smart_cooccur
    (transformer-confident block + GBM stacker rescues).

Marks the 90% precision floor + the locked operating point.
Output: outputs/orphan/v1_bce_pareto.png
"""
import json, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from af_common import OUT


def transformer_sweep(preds_path, n_steps=80):
    Z = np.load(preds_path, allow_pickle=True)
    p, br, y = Z["p"], Z["brightness"], Z["y"]
    ev = ~np.isnan(p); p = p[ev]; br = br[ev]; y = y[ev]
    N = len(y)
    call = (p > 0.5).astype(int)
    pts = []
    for tau in np.linspace(0.0, 1.0, n_steps):
        m = br >= tau
        n = int(m.sum())
        if n < 20:
            continue
        prec = float((call[m] == y[m]).mean())
        pts.append(dict(brightness_tau=float(tau),
                        coverage=n / N, precision=prec, n=n))
    return pts, N


def main(preds_path, cooccur_json, out_png, op_pt=None):
    preds_path = Path(preds_path); cooccur_json = Path(cooccur_json)
    transf, N = transformer_sweep(preds_path)
    combined = json.load(open(cooccur_json))["with_dnds"]["tradeoff_curve"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot([c["coverage"] for c in transf], [c["precision"] for c in transf],
            "o-", color="#2980B9", lw=2, ms=4, label="Transformer alone (sweep brightness)")
    ax.plot([c["combined_coverage"] for c in combined],
            [c["combined_precision"] for c in combined],
            "s-", color="#27AE60", lw=2, ms=5, label="+ GBM rescue (sweep stacker tau)")
    ax.axhline(0.90, color="red", ls="--", lw=1, label="precision floor 0.90")

    # operating point
    if op_pt is None:
        # default: take the locked deployment number from the saved cooccur summary
        s = json.load(open(cooccur_json))["with_dnds"]["smart_combined"]
        op_pt = (s["coverage"], s["precision"])
    ax.plot(*op_pt, marker="*", color="black", ms=22, zorder=10,
            label=f"deployment: {op_pt[0]:.3f} @ {op_pt[1]:.3f}")
    ax.annotate(f"  {op_pt[0]:.3f} @ {op_pt[1]:.3f}",
                xy=op_pt, xytext=(8, -6), textcoords="offset points",
                fontsize=10, fontweight="bold")

    ax.set_xlabel("combined coverage (fraction of genes called)")
    ax.set_ylabel("combined precision")
    ax.set_title(f"v1 BCE deployment Pareto -- {N:,} evaluated genes across 28 clades\n"
                 "(transformer brightness sweep vs +GBM rescue)")
    ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(0, 1.0); ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(OUT / "af_torch_preds_aug.npz"))
    ap.add_argument("--cooccur", default=str(OUT / "smart_cooccur_results_aug.json"))
    ap.add_argument("--out", default=str(OUT / "v1_bce_pareto.png"))
    a = ap.parse_args()
    main(a.preds, a.cooccur, a.out)
