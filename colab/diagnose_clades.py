"""Diagnose per-clade neCov@P90 variation on the v1 BCE run.

Why do burkholderia/magnetospirillum have decent AUC but neCov@P90 ~0.15, while
others reach 0.85? neCov@P90 is RANK-based (sort by 1-p, walk down, find the
largest prefix with >=90% non-essential precision), so it depends only on the
TOP of the non-essential ranking -- not on calibration. We decompose it into:

  (a) base difficulty   = the clade's essential rate (more essentials -> the
      non-essential precision tail is intrinsically squeezed),
  (b) top-prefix pollution = essentials the model confidently ranks as
      non-essential (the confidently-wrong set; = clade-specific/conditional
      essentials conservation mispredicts),

and we PROVE calibration is irrelevant by showing neCov@P90 is invariant to
temperature scaling.

Reads outputs/orphan/af_torch_preds_aug.npz. CPU only.
Output: diagnose_clades.json + diagnose_clades.png
"""
import json, argparse
from pathlib import Path
import numpy as np

from af_common import OUT, auc, auprc, cov_at_p


def running_precision_noness(p, y, fracs=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)):
    """Precision of the non-essential call at fixed coverage fractions."""
    s = 1 - p; o = np.argsort(-s); ny = (1 - y)[o]
    k = np.arange(1, len(ny) + 1); prec = np.cumsum(ny) / k
    out = {}
    for f in fracs:
        idx = max(0, int(f * len(ny)) - 1)
        out[str(f)] = round(float(prec[idx]), 3)
    return out


def main(preds_path, min_clade=2000):
    Z = np.load(preds_path, allow_pickle=True)
    p, br, y, clade = Z["p"], Z["brightness"], Z["y"], Z["clade"]
    ev = ~np.isnan(p); p, br, y, clade = p[ev], br[ev], y[ev], clade[ev]

    rows = []
    for cl in sorted(set(map(str, clade))):
        m = clade == cl
        if m.sum() < min_clade:
            continue
        pc, yc = p[m], y[m]
        ess = float(yc.mean())
        ne = cov_at_p(1 - pc, 1 - yc, 0.9)
        # top-decile confidently-non-essential: how many are actually essential?
        s = 1 - pc; o = np.argsort(-s); k = max(20, int(0.1 * len(o)))
        conf_wrong = float(yc[o[:k]].mean())          # essentials in the confident-NE prefix
        # confidently-wrong essentials overall: essentials with very low p
        cw_ess = float((pc[yc == 1] < 0.2).mean()) if (yc == 1).any() else 0.0
        rows.append(dict(clade=cl, n=int(m.sum()), ess_rate=round(ess, 3),
                         auc=round(auc(pc, yc), 3), auprc=round(auprc(pc, yc), 3),
                         neCov_P90=round(ne, 3),
                         top10_ess_pollution=round(conf_wrong, 3),
                         ess_called_confident_NE=round(cw_ess, 3),
                         prcurve=running_precision_noness(pc, yc)))

    rows.sort(key=lambda r: r["neCov_P90"])
    print(f"{'clade':<22}{'n':>7}{'ess%':>7}{'AUC':>7}{'AUPRC':>7}{'neCov':>7}"
          f"{'top10pol':>9}{'cwEss':>7}")
    print("-" * 80)
    for r in rows:
        print(f"{r['clade']:<22}{r['n']:>7}{r['ess_rate']*100:>6.1f}{r['auc']:>7.3f}"
              f"{r['auprc']:>7.3f}{r['neCov_P90']:>7.3f}{r['top10_ess_pollution']*100:>8.1f}%"
              f"{r['ess_called_confident_NE']*100:>6.1f}%")

    # ---- attribution: what predicts neCov across clades? ----
    A = np.array([[r["ess_rate"], r["top10_ess_pollution"], r["neCov_P90"]] for r in rows])
    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    print(f"\nacross clades:  corr(neCov, ess_rate)={corr(A[:,0],A[:,2]):+.2f}   "
          f"corr(neCov, top10_pollution)={corr(A[:,1],A[:,2]):+.2f}")

    # ---- prove calibration (temperature) does NOT change neCov ----
    cl = "burkholderia"
    if cl in set(map(str, clade)):
        m = clade == cl; pc, yc = p[m], y[m]
        logit = np.log(np.clip(pc, 1e-6, 1 - 1e-6) / np.clip(1 - pc, 1e-6, 1 - 1e-6))
        print(f"\ntemperature-scaling {cl} neCov@P90 (must be identical -> rank-invariant):")
        for T in (0.5, 1.0, 2.0, 5.0):
            pT = 1 / (1 + np.exp(-logit / T))
            print(f"  T={T:<4} neCov@P90={cov_at_p(1-pT,1-yc,0.9):.4f}")

    json.dump(rows, open(OUT / "diagnose_clades.json", "w"), indent=2)
    _plot(rows)
    print("\nwrote diagnose_clades.json + diagnose_clades.png")


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    # A: neCov vs ess_rate, colored by top10 pollution
    es = [r["ess_rate"] for r in rows]; ne = [r["neCov_P90"] for r in rows]
    pol = [r["top10_ess_pollution"] for r in rows]
    sc = ax[0].scatter(es, ne, c=pol, cmap="OrRd", s=80, edgecolor="grey", vmin=0, vmax=0.3)
    for r in rows:
        if r["neCov_P90"] < 0.3 or r["clade"] in ("burkholderia", "magnetospirillum"):
            ax[0].annotate(r["clade"], (r["ess_rate"], r["neCov_P90"]), fontsize=7)
    ax[0].set_xlabel("clade essential rate"); ax[0].set_ylabel("neCov@P90")
    ax[0].set_title("A. neCov@P90 vs essential rate\n(color = top-10% essential pollution)")
    fig.colorbar(sc, ax=ax[0], label="top-10% pollution")
    ax[0].grid(alpha=0.3)
    # B: running non-essential precision curves for weak vs strong clades
    weak = [r for r in rows if r["neCov_P90"] < 0.3][:4]
    strong = sorted(rows, key=lambda r: -r["neCov_P90"])[:3]
    fr = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    for r in strong:
        ax[1].plot(fr, [r["prcurve"][str(f)] for f in fr], "o-", color="green", alpha=0.5)
    for r in weak:
        ax[1].plot(fr, [r["prcurve"][str(f)] for f in fr], "s-", label=r["clade"])
    ax[1].axhline(0.9, color="red", ls="--", lw=1)
    ax[1].set_xlabel("coverage (top fraction of non-essential ranking)")
    ax[1].set_ylabel("non-essential precision")
    ax[1].set_title("B. Running precision down the ranking\n(green=strong clades; labeled=weak)")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "diagnose_clades.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(OUT / "af_torch_preds_aug.npz"))
    a = ap.parse_args()
    main(a.preds)
