"""Staged two-wheel run on S. aureus: W1 -> W2 -> W1 -> W2, every stage logged.

Wheel 1 = per-gene essentiality scorer. The trained transformer is DEAD on
S. aureus (leave-clade-out AUC ~0.50). Its deployable signal here is the
cross-organism family-conservation prior (OG essential rate over the 48 Beril
orgs; saur is not among them -> leak-clean). Labeled "Wheel 1 (conservation)".

Wheel 2 = cell assembler: lays the essentials into functional slots (rich local
annotation, 92% + COG families), audits for gaps vs a viable-cell requirements
table, and family-matches soup genes into the open slots.

Stages:
  1  W1  score all genes; take the high-precision confident-essential set.
  2  W2  build cell, find functional gaps, family-match soup -> fill gaps.
  3  W1  re-score: combine confident + Wheel-2 inserts; re-evaluate accuracy.
  4  W2  final cell: re-lay, final slots, completeness, final accuracy.

Output: outputs/orphan/saur_staged.png  (every stage + final cell)
        outputs/orphan/saur_staged.json  (all stage stats)
"""
import csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/orphan")
ORG = "saur_NCTC8325_biotradis"

# ---------- data ----------
truth = {}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"] == ORG:
        truth[r["locus_tag"]] = int(r["essential"])
sog = {}
for r in csv.DictReader(open("colab/work/og_assignments.csv")):
    if r["organism"].startswith("deg_Staphylococcus_aureus_subsp_aureus_NCTC_") or \
       r["organism"] == "deg_Staphylococcus_aureus_NCTC_8325":
        sog.setdefault(r["locus_tag"], r["og_id"])
ogof = {}
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"):
        ogof[(r["organism"], r["locus_tag"])] = r["og_id"]
ogn = defaultdict(int); oge = defaultdict(int)
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"] == ORG:
        continue
    og = ogof.get((r["organism"], r["locus_tag"]))
    if og:
        ogn[og] += 1; oge[og] += int(r["essential"])
og_rate = {og: oge[og] / ogn[og] for og in ogn}
annot = {}
for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")):
    if r["organism"] == ORG and r.get("product"):
        annot[r["locus_tag"]] = r["product"]
cogf = {}
for r in csv.DictReader(open("data/drive_import/labels/deg_families.csv")):
    if r["organism"] == ORG and r.get("cog"):
        cogf[r["locus_tag"]] = r["cog"]

TRUE_ESS = set(lt for lt, e in truth.items() if e == 1)
universe = [lt for lt in truth if lt in annot]           # cell works on annotated genes
def w1_score(lt):                                        # Wheel-1 conservation score
    og = sog.get(lt)
    return og_rate.get(og, np.nan) if og else np.nan
score = {lt: w1_score(lt) for lt in universe}

# ---------- slot classifier (rich annotation) ----------
SLOTS = [
    ("ribosomal", [r"ribosom"]),
    ("synthetase", [r"--trna lig", r"trna ligase", r"synthetase", r"aminoacyl"]),
    ("trna", [r"\btrna\b"]),
    ("transcription", [r"rna polymerase", r"transcription", r"sigma"]),
    ("replication", [r"dna polymerase", r"helicase", r"gyrase", r"primase", r"topoisomer", r"replicat", r"dna ligase"]),
    ("translation", [r"elongation factor", r"initiation factor", r"release factor", r"chaperon"]),
    ("cell_division", [r"cell division", r"septation", r"ftsz", r"\bftsz?\b", r"divisome"]),
    ("membrane", [r"membrane", r"secretion", r"transport", r"permease"]),
    ("energy", [r"atp synthase", r"dehydrogenase", r"cytochrome", r"nadh"]),
    ("lipid", [r"fatty acid", r"\bacp\b", r"lipid", r"teichoic"]),
    ("nucleotide", [r"purine", r"pyrimidine", r"nucleotide", r"ribonucleot"]),
    ("amino_acid", [r"amino acid", r"glutam", r"aspart"]),
    ("cofactor", [r"folate", r"thiamine", r"riboflavin", r"coenzyme", r"cofactor"]),
]
SP = [(nm, [re.compile(x, re.I) for x in pats]) for nm, pats in SLOTS]
def slot_of(lt):
    p = (annot.get(lt, "") or "").lower()
    for nm, rx in SP:
        if any(r.search(p) for r in rx):
            return nm
    return "other"
slot = {lt: slot_of(lt) for lt in universe}
REQ = {"ribosomal": 30, "synthetase": 12, "transcription": 6, "replication": 5,
       "translation": 5, "cell_division": 1}

def metrics(pred):
    tp = len(pred & TRUE_ESS)
    p = tp / max(len(pred), 1); r = tp / max(len(TRUE_ESS), 1)
    return dict(n=len(pred), tp=tp, precision=round(p, 3), recall=round(r, 3),
                f1=round(2 * p * r / max(p + r, 1e-9), 3))

stages = []

# ================= STAGE 1 -- WHEEL 1 (conservation scorer) =================
# transformer is dead (AUC ~0.50); use conservation prior, high-precision thr.
CONF = 0.75
confident = set(lt for lt in universe if not np.isnan(score[lt]) and score[lt] >= CONF)
m1 = metrics(confident)
filled1 = defaultdict(int)
for lt in confident: filled1[slot[lt]] += 1
gaps1 = {s: REQ[s] - filled1[s] for s in REQ if filled1[s] < REQ[s]}
stages.append(("S1  Wheel 1 (conservation, dead transformer)",
               dict(threshold=CONF, **m1, gaps=dict(gaps1))))
print(f"STAGE 1  WHEEL 1 (conservation; transformer dead AUC~0.50)")
print(f"  confident essentials @og_rate>={CONF}: {m1['n']}  "
      f"P={m1['precision']} R={m1['recall']} F1={m1['f1']}")
print(f"  functional gaps: {dict(gaps1)}")

# ================= STAGE 2 -- WHEEL 2 (assemble + family-match) =============
soup = [lt for lt in universe if lt not in confident]
inserts = defaultdict(list); chosen = set()
for s, need in gaps1.items():
    cands = [lt for lt in soup if slot[lt] == s and lt not in chosen]
    # rank by conservation score where available, else by COG presence
    cands.sort(key=lambda lt: -(score[lt] if not np.isnan(score[lt]) else 0.3))
    for lt in cands[:need]:
        inserts[s].append(lt); chosen.add(lt)
ins_all = [lt for v in inserts.values() for lt in v]
ins_prec = np.mean([lt in TRUE_ESS for lt in ins_all]) if ins_all else 0.0
assembled = confident | set(ins_all)
filled2 = defaultdict(int)
for lt in assembled: filled2[slot[lt]] += 1
gaps2 = {s: REQ[s] - filled2[s] for s in REQ if filled2[s] < REQ[s]}
m2 = metrics(assembled)
stages.append(("S2  Wheel 2 (assemble + family-match gaps)",
               dict(inserts=len(ins_all), insert_precision=round(float(ins_prec), 3),
                    gaps_after=dict(gaps2), **m2)))
print(f"\nSTAGE 2  WHEEL 2 (cell assembly + family-matched gap fill)")
print(f"  inserted {len(ins_all)} family-matched genes  precision={ins_prec:.2f}")
print(f"  gaps after fill: {dict(gaps2)}")
print(f"  assembled set: n={m2['n']} P={m2['precision']} R={m2['recall']} F1={m2['f1']}")

# ================= STAGE 3 -- WHEEL 1 (re-score combined) ===================
# Wheel 1 accepts Wheel 2's family-supported inserts as essential candidates and
# re-evaluates. Also promote any high-conservation soup gene the cell supports.
promoted = set(lt for lt in soup
               if (not np.isnan(score[lt]) and score[lt] >= 0.6)
               and slot[lt] in REQ)            # cell-supported core-function genes
combined = assembled | promoted
m3 = metrics(combined)
stages.append(("S3  Wheel 1 (re-score: accept inserts + cell-supported)",
               dict(promoted=len(promoted), **m3)))
print(f"\nSTAGE 3  WHEEL 1 (re-score with Wheel-2 feedback)")
print(f"  promoted {len(promoted)} cell-supported core genes")
print(f"  combined set: n={m3['n']} P={m3['precision']} R={m3['recall']} F1={m3['f1']}")

# ================= STAGE 4 -- WHEEL 2 (final cell) ==========================
final = combined
filled4 = defaultdict(int)
for lt in final: filled4[slot[lt]] += 1
complete = sum(1 for s in REQ if filled4[s] >= REQ[s])
m4 = metrics(final)
stages.append(("S4  Wheel 2 (final cell)",
               dict(core_complete=f"{complete}/{len(REQ)}", **m4)))
print(f"\nSTAGE 4  WHEEL 2 (final cell)")
print(f"  final cell: n={m4['n']} P={m4['precision']} R={m4['recall']} F1={m4['f1']}")
print(f"  core self-replication slots complete: {complete}/{len(REQ)}")

# accuracy of the final per-gene call over the predictable universe
# (essential if in final set, else non-essential), vs truth
pred_pos = final
acc_universe = [lt for lt in universe]
tp = sum(1 for lt in acc_universe if lt in pred_pos and truth[lt] == 1)
tn = sum(1 for lt in acc_universe if lt not in pred_pos and truth[lt] == 0)
accuracy = (tp + tn) / len(acc_universe)
print(f"\n=== FINAL ACCURACY (per-gene, over {len(acc_universe)} annotated genes): "
      f"{accuracy:.3f} ===")

# ---------- figure ----------
fig = plt.figure(figsize=(20, 6.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.05], wspace=0.22)

# (a) stage ledger -- every stage's stats as text
ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
ax.set_title("Every stage (W1->W2->W1->W2)", fontweight="bold", fontsize=12)
lines = []
for name, st in stages:
    lines.append(f"$\\bf{{{name.split('  ')[0]}}}$  {name.split('  ',1)[1]}")
    row = f"   n={st['n']}  P={st['precision']}  R={st['recall']}  F1={st['f1']}"
    extra = []
    if "gaps" in st and st["gaps"]: extra.append(f"gaps={st['gaps']}")
    if "inserts" in st: extra.append(f"inserts={st['inserts']} (P={st['insert_precision']})")
    if "promoted" in st: extra.append(f"promoted={st['promoted']}")
    if "core_complete" in st: extra.append(f"core={st['core_complete']}")
    lines.append(row)
    for e in extra: lines.append(f"      {e}")
    lines.append("")
ax.text(0.0, 0.98, "\n".join(lines), transform=ax.transAxes, va="top",
        fontsize=9.5, family="monospace")

# (b) metric trajectory across stages
ax = fig.add_subplot(gs[0, 1])
xs = ["S1\nW1", "S2\nW2", "S3\nW1", "S4\nW2"]
P = [s[1]["precision"] for s in stages]
R = [s[1]["recall"] for s in stages]
F = [s[1]["f1"] for s in stages]
ax.plot(xs, P, "o-", color="#3182bd", lw=2, label="precision")
ax.plot(xs, R, "s-", color="#e31a1c", lw=2, label="recall")
ax.plot(xs, F, "^-", color="#000", lw=2, label="F1")
ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax.set_title(f"Accuracy through the loop\nfinal per-gene accuracy = {accuracy:.2f}",
             fontweight="bold", fontsize=12)
for i in range(4):
    ax.annotate(f"{F[i]:.2f}", (i, F[i]), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")

# (c) the final cell -- slot bars with minimums
ax = fig.add_subplot(gs[0, 2])
names = [s[0] for s in SLOTS]
vals = [filled4[s] for s in names]
ax.barh(range(len(names)), vals, color="#54278f")
for i, s in enumerate(names):
    if s in REQ:
        ax.plot([REQ[s], REQ[s]], [i - 0.4, i + 0.4], color="red", lw=2)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis(); ax.set_xlabel("# essentials in slot (final cell)")
ax.set_title(f"Final cell: {complete}/{len(REQ)} core complete\n"
             f"(red tick = viable minimum)", fontweight="bold", fontsize=12)

fig.suptitle("S. aureus through the staged two-wheel loop "
             "(sandbox, local data only)", fontweight="bold", fontsize=14)
plt.savefig(OUT / "saur_staged.png", dpi=120, bbox_inches="tight")
print("\nwrote outputs/orphan/saur_staged.png")

json.dump(dict(organism=ORG, n_genes=len(truth), n_essential=len(TRUE_ESS),
               n_annotated=len(universe), transformer_auc=0.503,
               stages=[dict(stage=name, **st) for name, st in stages],
               final_accuracy=round(accuracy, 3),
               final_core_complete=f"{complete}/{len(REQ)}"),
          open(OUT / "saur_staged.json", "w"), indent=2)
print("wrote outputs/orphan/saur_staged.json")
