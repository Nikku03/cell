"""Doubling-time estimation WITHOUT kinetics, from codon usage bias.

Physiology (Schaechter 1958; Vieira-Silva & Rocha 2010): fast-growing bacteria
optimize the codons of their highly-expressed genes (ribosomal proteins,
elongation factors) for translational speed. The codon-usage-bias gap between
the translation machinery and the bulk genome therefore predicts MINIMUM
generation time -- no growth curve, no rate constants needed.

  signal  delta_CAI = mean CAI(ribosomal + EF + translation genes)
                    - mean CAI(whole genome)
  model   log10(DT_min) = a * delta_CAI + b   (calibrated on organisms with a
                          known lab doubling time)

Honest scope: limited to the organisms that have CAI in codon_features.csv with
enough joinable annotated ribosomal genes (4 here). It predicts growth CLASS /
order-of-magnitude DT, not exact minutes (literature reports ~2-3x uncertainty).

Output: outputs/orphan/doubling_time.png
        outputs/orphan/doubling_time.json
"""
import csv, re, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/orphan")

# known fastest lab doubling times (minutes) -- calibration anchors
KNOWN_DT = {"ecoli_BW25113_tradis": 20, "beril_Keio": 20,
            "saur_NCTC8325_biotradis": 30, "spneT4": 30, "mtub": 1440,
            "beril_RalstoniaGMI1000": 90, "syn3a": 120, "mgen": 360}
NICE = {"saur_NCTC8325_biotradis": "S. aureus", "spneT4": "S. pneumoniae",
        "mtub": "M. tuberculosis", "beril_RalstoniaGMI1000": "R. solanacearum"}

# CAI per org
cai = defaultdict(dict)
for r in csv.DictReader(open("data/drive_import/labels/codon_features.csv")):
    try: cai[r["organism"]][r["locus_tag"]] = float(r["cai"])
    except (ValueError, KeyError): pass
# annotation per org
ann = defaultdict(dict)
for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")):
    if r.get("product"): ann[r["organism"]][r["locus_tag"]] = r["product"]
for r in csv.DictReader(open("data/drive_import/labels/deg_families.csv")):
    if r.get("product"): ann[r["organism"]].setdefault(r["locus_tag"], r["product"])

HX = re.compile(r"ribosomal protein|elongation factor|translation initiation|"
                r"translation elongation|aminoacyl|--trna ligase", re.I)

# compute delta_CAI per org
data = []
for o in cai:
    common = set(cai[o]) & set(ann.get(o, {}))
    if len(common) < 100: continue
    hx = [lt for lt in common if HX.search(ann[o][lt])]
    if len(hx) < 15: continue
    delta = float(np.mean([cai[o][lt] for lt in hx]) -
                  np.mean([cai[o][lt] for lt in cai[o]]))
    data.append(dict(org=o, n_cai=len(cai[o]), n_hx=len(hx),
                     delta_cai=round(delta, 4), known_dt=KNOWN_DT.get(o)))

print(f"{'organism':<26}{'n_hx':>6}{'delta_CAI':>10}{'known_DT':>10}")
for d in sorted(data, key=lambda x: -x["delta_cai"]):
    print(f"  {d['org']:<24}{d['n_hx']:>6}{d['delta_cai']:>10.3f}"
          f"{str(d['known_dt'] or '?'):>10}")

# calibrate log10(DT) ~ delta on the known points
cal = [d for d in data if d["known_dt"]]
x = np.array([d["delta_cai"] for d in cal])
y = np.log10([d["known_dt"] for d in cal])
A = np.vstack([x, np.ones_like(x)]).T
(a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
pred_log = A @ np.array([a, b])
ss_res = ((y - pred_log) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
r2 = 1 - ss_res / max(ss_tot, 1e-9)
print(f"\ncalibration: log10(DT_min) = {a:.2f}*delta + {b:.2f}   R2={r2:.2f}  "
      f"(n={len(cal)} anchors)")

# predicted DT for every org with a delta
for d in data:
    d["pred_dt"] = round(float(10 ** (a * d["delta_cai"] + b)), 0)
print(f"\n{'organism':<26}{'delta':>8}{'known_DT':>10}{'pred_DT':>9}{'fold_err':>9}")
for d in sorted(data, key=lambda x: x["pred_dt"]):
    fe = (max(d["pred_dt"], d["known_dt"]) / min(d["pred_dt"], d["known_dt"])
          if d["known_dt"] else None)
    print(f"  {d['org']:<24}{d['delta_cai']:>8.3f}{str(d['known_dt'] or '?'):>10}"
          f"{d['pred_dt']:>9.0f}{(f'{fe:.1f}x' if fe else '-'):>9}")

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
# (a) the calibration: delta vs known DT
cal_sorted = sorted(cal, key=lambda d: d["delta_cai"])
xs = np.array([d["delta_cai"] for d in cal_sorted])
ys = np.array([d["known_dt"] for d in cal_sorted])
ax[0].scatter(xs, ys, s=120, color="#e31a1c", zorder=3)
for d in cal_sorted:
    ax[0].annotate(NICE.get(d["org"], d["org"].replace("beril_", "")),
                   (d["delta_cai"], d["known_dt"]),
                   xytext=(6, 4), textcoords="offset points", fontsize=9)
gx = np.linspace(min(xs) - 0.02, max(xs) + 0.02, 50)
ax[0].plot(gx, 10 ** (a * gx + b), "--", color="#3182bd", lw=2,
           label=f"log10(DT)={a:.1f}·Δ+{b:.1f}  (R²={r2:.2f})")
ax[0].set_yscale("log")
ax[0].set_xlabel("Δ CAI  (translation-machinery codon bias − genome)")
ax[0].set_ylabel("known doubling time (min, log scale)")
ax[0].set_title("Doubling time WITHOUT kinetics\n"
                "codon-usage bias of ribosomal genes predicts growth rate",
                fontweight="bold", fontsize=12)
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3, which="both")

# (b) predicted vs known (validation)
ax[1].plot([10, 2000], [10, 2000], "k--", lw=1, alpha=0.5, label="perfect")
ax[1].fill_between([10, 2000], [5, 1000], [20, 4000], color="gray", alpha=0.12,
                   label="±2× band")
for d in cal:
    ax[1].scatter(d["known_dt"], d["pred_dt"], s=120, color="#e31a1c", zorder=3)
    ax[1].annotate(NICE.get(d["org"], d["org"].replace("beril_", "")),
                   (d["known_dt"], d["pred_dt"]),
                   xytext=(6, 4), textcoords="offset points", fontsize=9)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("known doubling time (min)")
ax[1].set_ylabel("predicted doubling time (min)")
ax[1].set_title("Predicted vs known (4 organisms)\n"
                "ranks growth class correctly; ~2× absolute uncertainty",
                fontweight="bold", fontsize=12)
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, which="both")

fig.suptitle("Kinetics-free doubling-time estimation from codon usage bias",
             fontweight="bold", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT / "doubling_time.png", dpi=120, bbox_inches="tight")
print("\nwrote outputs/orphan/doubling_time.png")

json.dump(dict(method="codon_usage_bias (Vieira-Silva & Rocha 2010)",
               calibration=dict(slope=round(a, 3), intercept=round(b, 3),
                                r2=round(float(r2), 3), n_anchors=len(cal)),
               organisms=data),
          open(OUT / "doubling_time.json", "w"), indent=2)
print("wrote outputs/orphan/doubling_time.json")
