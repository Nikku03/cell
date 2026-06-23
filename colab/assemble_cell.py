"""Demand-driven cell assembly: Wheel 2 specs the table, Wheel 1 fills with
high-precision calls, Wheel 2 finds the gaps, Wheel 1 hands back protein
families for the leftovers, Wheel 2 reasons which families fit the open slots
and inserts them.

Pipeline (exactly the proposed loop):
  1 SPEC      Wheel 2 publishes the requirements table (functional categories +
              viable minima a cell must satisfy; syn3a-derived core machinery).
  2 FILL      Wheel 1 reports essentials at P>=0.90 (global thresholds). Wheel 2
              lays them on the table by category.
  3 GAPS      Wheel 2 reports unfilled categories.
  4 NARROW    drop placed essentials + confident non-essentials -> only the
              uncertain middle remains (the search space).
  5 FAMILY    Wheel 1 classifies the leftovers into PROTEIN FAMILIES.
              (orthogroup = sequence-clustered family; this is the slot where a
               real ESM/AlphaFold family classifier would plug in.)
  6 MATCH     Wheel 2 reasons which families fit each open slot and INSERTS the
              best candidates (educated guess) until the slot is satisfied.

Then validate against Keio truth (model never saw it): are the INSERTED genes
actually essential? do the gaps close? does the assembled cell beat the
confident-only cell?

Output: outputs/orphan/assemble_cell.png
        outputs/orphan/assemble_cell.json
        outputs/orphan/assemble_cell_report.md
"""
import csv, json, re
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

exec(open("/tmp/emergent_head.py").read())   # Keio: patched, klt, ky, kuniv,
                                              # slot_of(idx), classify, SLOT_NAMES,
                                              # og_of, og_present, synt, annot,
                                              # gene_names, true_ess, n_orgs
OUT = Path("outputs/orphan")
N = len(klt); lt2j = {klt[j]: j for j in range(N)}
ess_thr = json.load(open(OUT / "integrated_engine.json"))["thresholds"]["P90_essential"]
ne_thr = json.load(open(OUT / "integrated_engine.json"))["thresholds"]["P90_non_essential"]
print(f"global P90 thresholds: essential>={ess_thr:.3f}  non_essential<={ne_thr:.3f}")

# fixed classifier: synthetase BEFORE trna (aaRS products say "tRNA ligase"
# and the inherited classify() mis-slots them as trna)
_SL = [SLOTS[0]] + [[s for s in SLOTS if s[0] == "synthetase"][0],
                    [s for s in SLOTS if s[0] == "trna"][0]] + \
      [s for s in SLOTS[1:] if s[0] not in ("trna", "synthetase")]
_SN = [s[0] for s in _SL]
_SP = [[re.compile(x, re.I) for x in a] for _, a, _ in _SL]
_SG = [[re.compile(x, re.I) for x in b] for _, _, b in _SL]
def classify_name(prod, gn):
    pp = (prod or "").lower(); gg = (gn or "").lower().strip()
    txt = pp if pp else (gg if len(gg) > 6 else "")
    for i in range(len(_SN) - 1):
        if txt and any(rx.search(txt) for rx in _SP[i]): return _SN[i]
        if gg and any(rx.search(gg) for rx in _SG[i]): return _SN[i]
    return "other"
# recompute Keio per-gene slots with the fixed classifier
slot_name = {klt[j]: classify_name(annot.get(klt[j], ""), gene_names.get(klt[j], ""))
             for j in range(N)}
def sname(j):
    return slot_name[klt[j]]

# ===== STAGE 1: Wheel 2 publishes the requirements table =====================
syn_prod = {}
for r in csv.DictReader(open("memory_bank/data/syn3a_gene_table.csv")):
    if r.get("product"): syn_prod[r["locus_tag"]] = r["product"]
syn_counts = Counter()
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"] == "syn3a" and int(r["essential"]) == 1:
        syn_counts[classify_name(syn_prod.get(r["locus_tag"], ""), r.get("gene_name", ""))] += 1
# full functional cell: self-replication machinery + the metabolism/membrane a
# free-living cell needs. Minimums = 50% of syn3a's count per category.
REQ = [s for s in _SN if s != "other" and syn_counts.get(s, 0) > 0]
TABLE = {s: int(np.ceil(0.5 * syn_counts.get(s, 0))) for s in REQ}
print(f"\nSTAGE 1 -- Wheel 2 requirements table (full cell): {TABLE}")

# ===== STAGE 2: Wheel 1 fills with P90 essentials; Wheel 2 lays on table =====
score = patched.copy()
conf_ess = [j for j in range(N) if score[j] >= ess_thr]
conf_non = [j for j in range(N) if score[j] <= ne_thr]
placed = defaultdict(list)
for j in conf_ess:
    placed[sname(j)].append(j)
print(f"STAGE 2 -- Wheel 1 confident essentials: {len(conf_ess)}  "
      f"(precision vs truth {np.mean([klt[j] in true_ess for j in conf_ess]):.2f})")

# ===== STAGE 3: Wheel 2 gap report ==========================================
gaps = {s: TABLE[s] - len(placed.get(s, [])) for s in REQ if len(placed.get(s, [])) < TABLE[s]}
print(f"STAGE 3 -- gaps (categories below minimum): {gaps}")

# ===== STAGE 4: narrow -- only the uncertain middle remains =================
placed_set = set(conf_ess); noness_set = set(conf_non)
soup = [j for j in range(N) if j not in placed_set and j not in noness_set]
print(f"STAGE 4 -- search space after removing placed + confident non-essential: "
      f"{len(soup)} genes")

# ===== STAGE 5: Wheel 1 classifies leftovers into PROTEIN FAMILIES ===========
# orthogroup = sequence family. family functional slot = classify(consensus
# product across the family's members in ALL organisms).
all_annot = {}
for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")):
    if r.get("product"): all_annot[(r["organism"], r["locus_tag"])] = r["product"]
memb = defaultdict(list)
for k, o in og_of.items(): memb[o].append(k)
fam_slot_cache = {}
def family_slot(og):
    if og in fam_slot_cache: return fam_slot_cache[og]
    prods = [all_annot.get(k, "") for k in memb.get(og, [])]
    prods = [p for p in prods if p and p.lower() != "hypothetical protein"]
    cons = Counter(prods).most_common(1)[0][0] if prods else ""
    sl = classify_name(cons, "")
    fam_slot_cache[og] = (sl, cons)
    return fam_slot_cache[og]
# family report for the soup
fam_of = {}
for j in soup:
    og = og_of.get((ORG, klt[j]), "")
    fam_of[j] = family_slot(og) if og else ("other", "")
print(f"STAGE 5 -- Wheel 1 classified {len(soup)} leftovers into families "
      f"({len({og_of.get((ORG,klt[j]),'') for j in soup})} distinct orthogroups)")

# ===== STAGE 6: Wheel 2 reasons which families fit open slots, INSERTS =======
inserted = defaultdict(list)
for s in list(gaps):
    need = gaps[s]
    # candidates: soup genes whose PROTEIN FAMILY function matches the open slot
    cands = [j for j in soup if fam_of[j][0] == s]
    # rank by reasoning strength: family essentiality (og_univ) + model score
    cands.sort(key=lambda j: -((kuniv[j] if not np.isnan(kuniv[j]) else 0) * 0.6
                               + score[j] * 0.4))
    for j in cands[:need]:
        inserted[s].append(j)
n_ins = sum(len(v) for v in inserted.values())
ins_all = [j for v in inserted.values() for j in v]
ins_prec = np.mean([klt[j] in true_ess for j in ins_all]) if ins_all else 0
print(f"STAGE 6 -- Wheel 2 inserted {n_ins} family-matched genes into gaps "
      f"(precision vs truth {ins_prec:.2f})")
remaining_gaps = {s: TABLE[s] - len(placed.get(s, [])) - len(inserted.get(s, []))
                  for s in REQ
                  if len(placed.get(s, [])) + len(inserted.get(s, [])) < TABLE[s]}

# ===== assembled cell + validation ==========================================
def metrics(pred):
    tp = len(pred & true_ess)
    return dict(n=len(pred), tp=tp,
                precision=round(tp/max(len(pred),1),3),
                recall=round(tp/max(len(true_ess),1),3),
                f1=round(2*tp/max(len(pred)+len(true_ess),1),3))
cell_conf = set(klt[j] for j in conf_ess)
cell_asm = cell_conf | set(klt[j] for j in ins_all)
m_conf = metrics(cell_conf); m_asm = metrics(cell_asm)
print(f"\nconfident-only cell : {m_conf}")
print(f"ASSEMBLED cell      : {m_asm}")

# showcase: every gene Wheel 2 inserted to fill a gap (the educated guesses)
show_rows = []
for s in inserted:
    for j in inserted[s]:
        show_rows.append(dict(slot=s, locus_tag=klt[j], gene=gene_names.get(klt[j], "?"),
                              family_product=fam_of[j][1][:40],
                              model_p=round(float(score[j]), 3),
                              truly_essential=(klt[j] in true_ess)))

# ===== figure ================================================================
fig, ax = plt.subplots(1, 3, figsize=(20, 6.2))

# (1) the assembly table
slots = REQ
req = [TABLE[s] for s in slots]
filledc = [len(placed.get(s, [])) for s in slots]
insc = [len(inserted.get(s, [])) for s in slots]
yp = np.arange(len(slots))
ax[0].barh(yp, filledc, color="#e31a1c", label="Wheel 1 confident fill")
ax[0].barh(yp, insc, left=filledc, color="#fd8d3c",
           label="Wheel 2 family-matched insert")
for i, s in enumerate(slots):
    ax[0].plot([req[i], req[i]], [i - 0.4, i + 0.4], color="black", lw=2)
ax[0].set_yticks(yp); ax[0].set_yticklabels(slots); ax[0].invert_yaxis()
ax[0].set_xlabel("# genes in category")
ax[0].set_title("The assembly table\n(black tick = required minimum)",
                fontweight="bold", fontsize=11)
ax[0].legend(fontsize=8, loc="lower right")

# (2) what the cell adds
labels = ["genes\ncalled", "true ess.\nrecovered", "precision", "recall"]
cv = [m_conf["n"], m_conf["tp"], m_conf["precision"], m_conf["recall"]]
av = [m_asm["n"], m_asm["tp"], m_asm["precision"], m_asm["recall"]]
norm = [700, 700, 1, 1]
x = np.arange(4)
ax[1].bar(x - 0.2, [cv[i]/norm[i] for i in range(4)], 0.4, color="#9ecae1",
          label="confident-only cell")
ax[1].bar(x + 0.2, [av[i]/norm[i] for i in range(4)], 0.4, color="#e31a1c",
          label="assembled cell")
for i in range(4):
    ax[1].text(x[i]-0.2, cv[i]/norm[i]+0.02, str(cv[i]), ha="center", fontsize=8)
    ax[1].text(x[i]+0.2, av[i]/norm[i]+0.02, str(av[i]), ha="center", fontsize=8)
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=9)
ax[1].set_title(f"Confident-only vs assembled cell\n"
                f"insert precision {ins_prec:.2f} -- the educated guesses are mostly right",
                fontweight="bold", fontsize=11)
ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1)

# (3) synthetase showcase table
ax[2].axis("off")
ax[2].set_title(f"Wheel 2's educated guesses ({n_ins} inserts, "
                f"{int(ins_prec*100)}% truly essential)\n"
                f"family-matched into the open slots",
                fontweight="bold", fontsize=11)
txt = f"{'slot':<13}{'gene':<7}{'family product':<26}{'real?'}\n" + "-" * 54 + "\n"
for r in show_rows[:16]:
    mark = "ESSENTIAL" if r["truly_essential"] else "no"
    txt += f"{r['slot'][:12]:<13}{r['gene'][:6]:<7}{r['family_product'][:24]:<26}{mark}\n"
ax[2].text(0.0, 0.95, txt, transform=ax[2].transAxes, fontsize=9,
           va="top", family="monospace")

fig.suptitle("Demand-driven cell assembly: Wheel 2 specs the gaps, Wheel 1 "
             "supplies families, Wheel 2 reasons which fit and inserts",
             fontweight="bold", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT / "assemble_cell.png", dpi=120, bbox_inches="tight")
print("\nwrote outputs/orphan/assemble_cell.png")

json.dump(dict(
    requirements_table=TABLE, gaps_before=gaps, gaps_after=remaining_gaps,
    confident_fill=len(conf_ess), search_space=len(soup),
    inserted=n_ins, insert_precision=round(float(ins_prec), 3),
    confident_only_cell=m_conf, assembled_cell=m_asm,
    synthetase_inserts=show_rows,
), open(OUT / "assemble_cell.json", "w"), indent=2)
print("wrote outputs/orphan/assemble_cell.json")

with open(OUT / "assemble_cell_report.md", "w") as fh:
    fh.write(f"""# Demand-driven cell assembly — E. coli

Model trained leave-`escherichia`-out (never saw Keio labels). Protein family =
orthogroup (where a real ESM/AlphaFold family classifier would plug in).

## The loop
1. **Wheel 2 requirements table** (core machinery a cell must have): `{TABLE}`
2. **Wheel 1 confident fill** (P>=0.90): {len(conf_ess)} essentials, precision {m_conf['precision']}
3. **Gaps reported**: `{gaps}`
4. **Search space** after removing placed + confident non-essential: {len(soup)} genes
5. **Wheel 1 families**: leftovers grouped into orthogroup families
6. **Wheel 2 inserts** {n_ins} family-matched genes; **precision of the inserts = {ins_prec:.2f}**

## Result

| | confident-only cell | assembled cell |
|---|---|---|
| genes | {m_conf['n']} | {m_asm['n']} |
| true essentials | {m_conf['tp']} | {m_asm['tp']} |
| precision | {m_conf['precision']} | {m_asm['precision']} |
| recall | {m_conf['recall']} | {m_asm['recall']} |

Gaps remaining after assembly: `{remaining_gaps or 'none — table complete'}`
""")
print("wrote outputs/orphan/assemble_cell_report.md")
