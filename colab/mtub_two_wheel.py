"""Two-wheel system on M. tuberculosis -- the hardest real case.

Wheel 1 (transformer) is DEAD on mtub: leave-clade-out AUC 0.518 (random), and
its preds aren't even in the sandbox. So this is the demand-driven test: when
Wheel 1 is useless, what can Wheel 2 (systems + family conservation) do ALONE,
using only LOCAL data (no fetch, no GPU)?

Wheel 2 signal = cross-organism family essentiality. For each mtub gene mapped
to an orthogroup (via local mmseqs og_assignments), look up that OG's essential
rate across the 48 Beril organisms (mtub is NOT among them -> leak-clean by
construction). Predict mtub essentiality from that rate alone.

Validate vs mtub's 760 wet-lab essentials. Also build the functional cell from
the true essentials using the now-rich local annotation (98% products + COG)
and check completeness.

Output: outputs/orphan/mtub_two_wheel.png
        outputs/orphan/mtub_two_wheel.json
"""
import csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/orphan")
ORG = "mtub"

# ---- mtub OG assignments (local mmseqs) ----
mtub_og = {}
for r in csv.DictReader(open("colab/work/og_assignments.csv")):
    if r["organism"].startswith("deg_Mycobacterium_tuberculosis_H37Rv"):
        mtub_og.setdefault(r["locus_tag"], r["og_id"])

# ---- truth ----
truth = {}; gene_names = {}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"] == ORG:
        truth[r["locus_tag"]] = int(r["essential"])

# ---- Beril OG essential rate (leak-clean: mtub not in Beril) ----
ogof = {}
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"):
        ogof[(r["organism"], r["locus_tag"])] = r["og_id"]
og_n = defaultdict(int); og_e = defaultdict(int); og_orgs = defaultdict(set)
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"] == ORG:
        continue
    og = ogof.get((r["organism"], r["locus_tag"]))
    if og:
        og_n[og] += 1; og_e[og] += int(r["essential"]); og_orgs[og].add(r["organism"])
og_rate = {og: og_e[og] / og_n[og] for og in og_n}

# ---- annotation (local, now rich) + COG family ----
annot = {}
for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")):
    if r["organism"] == ORG and r.get("product"):
        annot[r["locus_tag"]] = r["product"]
cog = {}; func_class = {}
for r in csv.DictReader(open("data/drive_import/labels/deg_families.csv")):
    if r["organism"] == ORG:
        if r.get("cog"): cog[r["locus_tag"]] = r["cog"]
        if r.get("func_class"): func_class[r["locus_tag"]] = r["func_class"]

# ---- Wheel-2 prediction = OG essential rate ----
keys = [lt for lt in truth if lt in mtub_og and mtub_og[lt] in og_rate]
y = np.array([truth[lt] for lt in keys])
score = np.array([og_rate[mtub_og[lt]] for lt in keys])
breadth = np.array([len(og_orgs[mtub_og[lt]]) for lt in keys])
n, npos = len(keys), int(y.sum())
print(f"mtub: {len(truth)} genes, {sum(truth.values())} essential")
print(f"Wheel-2 predictable (OG + Beril rate): {n}  ({npos} essential)")

def auc(yt, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s))
    pos = yt == 1
    return (r[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / (pos.sum() * (~pos).sum())

A = auc(y, score)
print(f"\nWheel-2 (family-conservation) AUC on mtub: {A:.3f}")
print(f"   vs Wheel-1 transformer (leave-clade-out): AUC 0.518  (random)")

def cov_at_prec(s, yt, target=0.90, pos=True):
    sc = s if pos else -s; yy = yt if pos else 1 - yt
    o = np.argsort(-sc); ys = yy[o]; tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1); ok = prec >= target
    return float(tp[ok].max() / max(yy.sum(), 1)) if ok.any() else 0.0

essCov = cov_at_prec(score, y, 0.90, True)
neCov = cov_at_prec(score, y, 0.90, False)
print(f"   essCov@P90={essCov:.3f}  neCov@P90={neCov:.3f}")

# breadth-gated: only call when OG is broadly conserved (high confidence)
broad = breadth >= 20
print(f"\nbroadly-conserved OGs (>=20 orgs): {broad.sum()}  "
      f"ess_rate={y[broad].mean():.2f}  AUC={auc(y[broad],score[broad]):.3f}")

# ---- functional completeness of the TRUE essentials (rich local annotation) ----
SLOTS = [
    ("ribosomal", [r"ribosom"]), ("synthetase", [r"--trna lig", r"trna ligase", r"synthetase", r"aminoacyl"]),
    ("trna", [r"\btrna\b"]), ("transcription", [r"rna polymerase", r"transcription", r"sigma"]),
    ("replication", [r"dna polymerase", r"helicase", r"gyrase", r"primase", r"topoisomer", r"replicat", r"\bdna ligase\b"]),
    ("translation", [r"elongation factor", r"initiation factor", r"release factor", r"chaperon"]),
    ("cell_division", [r"cell division", r"septation", r"ftsz", r"\bftsz?\b"]),
    ("membrane", [r"membrane", r"secretion", r"transport", r"permease"]),
    ("energy", [r"atp synthase", r"dehydrogenase", r"cytochrome", r"nadh"]),
    ("lipid", [r"fatty acid", r"\bacp\b", r"lipid", r"mycolic"]),
    ("nucleotide", [r"purine", r"pyrimidine", r"nucleotide", r"ribonucleot"]),
    ("amino_acid", [r"amino acid", r"glutam", r"aspart"]),
    ("cofactor", [r"folate", r"thiamine", r"riboflavin", r"coenzyme", r"cofactor"]),
]
SP = [(nm, [re.compile(x, re.I) for x in pats]) for nm, pats in SLOTS]
def slot_of(lt):
    p = (annot.get(lt, "") or "").lower()
    for nm, rx in SP:
        if any(r.search(p) for r in rx): return nm
    return "other"
true_ess = [lt for lt, e in truth.items() if e == 1]
slot_counts = defaultdict(int)
for lt in true_ess: slot_counts[slot_of(lt)] += 1
REQ = {"ribosomal": 30, "synthetase": 12, "transcription": 6, "replication": 5,
       "translation": 5, "cell_division": 1}
complete = sum(1 for s in REQ if slot_counts[s] >= REQ[s])
print(f"\nmtub TRUE-essential cell ({len(true_ess)} genes) functional slots:")
for s in [s[0] for s in SLOTS]:
    flag = ""
    if s in REQ: flag = "OK" if slot_counts[s] >= REQ[s] else f"LOW (<{REQ[s]})"
    print(f"   {s:<14} {slot_counts[s]:>4}  {flag}")
print(f"core self-replication machinery: {complete}/{len(REQ)} complete")

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))

# (a) Wheel-2 vs dead Wheel-1
ax[0].bar([0, 1], [0.518, A], color=["#bbbbbb", "#e31a1c"], width=0.6)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["Wheel 1\n(transformer)", "Wheel 2\n(family conservation)"])
ax[0].axhline(0.5, color="k", ls=":", lw=1)
ax[0].set_ylabel("AUC on mtub (vs truth)"); ax[0].set_ylim(0, 1)
ax[0].set_title("When Wheel 1 is dead, Wheel 2 carries\n"
                f"AUC 0.52 (random) -> {A:.2f}", fontweight="bold", fontsize=11)
for i, v in enumerate([0.518, A]):
    ax[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

# (b) precision-coverage of Wheel 2
o = np.argsort(-score); ys = y[o]
tp = np.cumsum(ys); fp = np.cumsum(1 - ys); prec = tp / np.maximum(tp + fp, 1)
rec = tp / npos
ax[1].plot(rec, prec, "-", color="#3182bd", lw=2)
ax[1].axhline(0.90, color="#888", ls="--", lw=1)
ax[1].set_xlabel("recall (coverage of essentials)"); ax[1].set_ylabel("precision")
ax[1].set_title(f"Wheel-2 essential calling on mtub\nessCov@P90={essCov:.2f}",
                fontweight="bold", fontsize=11); ax[1].set_ylim(0, 1.02); ax[1].grid(alpha=0.3)

# (c) functional completeness
names = [s[0] for s in SLOTS]; vals = [slot_counts[s] for s in names]
ax[2].barh(range(len(names)), vals, color="#54278f")
for i, s in enumerate(names):
    if s in REQ:
        ax[2].plot([REQ[s], REQ[s]], [i - 0.4, i + 0.4], color="red", lw=2)
ax[2].set_yticks(range(len(names))); ax[2].set_yticklabels(names, fontsize=9)
ax[2].invert_yaxis(); ax[2].set_xlabel("# true essentials in slot")
ax[2].set_title(f"mtub true-essential cell: {complete}/{len(REQ)} core complete\n"
                "(red tick = viable minimum)", fontweight="bold", fontsize=11)

fig.suptitle("M. tuberculosis in the two-wheel system (sandbox, local data only) "
             "-- the demand-driven case", fontweight="bold", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT / "mtub_two_wheel.png", dpi=120, bbox_inches="tight")
print("\nwrote outputs/orphan/mtub_two_wheel.png")

json.dump(dict(
    organism=ORG, n_genes=len(truth), n_essential=sum(truth.values()),
    wheel2_predictable=n, wheel2_auc=round(float(A), 3),
    wheel1_auc=0.518, ess_cov_p90=round(essCov, 3), ne_cov_p90=round(neCov, 3),
    broad_n=int(broad.sum()), broad_auc=round(float(auc(y[broad], score[broad])), 3),
    slot_counts={s: slot_counts[s] for s in names},
    core_complete=f"{complete}/{len(REQ)}",
), open(OUT / "mtub_two_wheel.json", "w"), indent=2)
print("wrote outputs/orphan/mtub_two_wheel.json")
