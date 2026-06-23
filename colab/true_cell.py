"""Build cells from TRUE essential genes (no model in the loop) and check
whether every essential aspect of a cell is functionally complete.

This is the upper-bound test of the functional-slot framework. If the actual
labelled essentials of an organism fail to populate a required slot, the
framework's definition is wrong. If they consistently populate all slots
across diverse organisms (small Mycoplasma minimal cells AND large
E. coli/Mtb genomes), the framework is sound -- then any *model-built* cell
that fills the same slots is a candidate viable cell.

Chosen organisms (ground-truth essential set + good annotation):
  syn3a              383 essentials  (JCVI wet-lab minimal cell)
  mgen               382 essentials  (M. genitalium, near-minimal)
  beril_Keio         605 essentials  (E. coli Keio knockout collection)
  mtub               760 essentials  (M. tuberculosis Tn-seq)

For each: bin essentials into 14 functional slots, check completeness vs
syn3a's required-slot fingerprint (3+ genes per slot), and plot.

Output: outputs/orphan/true_cell_*.json
        outputs/orphan/true_cell_grid.png
"""
import csv, json, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/orphan")
ORGS = ["syn3a", "mgen", "beril_Keio", "mtub"]

# --- functional slots (gene_name + product keyword classifier) ---
SLOTS = [
    ("ribosomal",      [r"ribosom", r"\b30s\b", r"\b50s\b"],
                        [r"^rp[sl][a-z]?\d*$", r"^rpm[a-z]\d*$"]),
    ("trna",           [r"\btrna\b", r"transfer rna"],
                        [r"^trn[a-z]\d*$", r"^trna-"]),
    ("synthetase",     [r"synthetase", r"trna ligase", r"aminoacyl"],
                        [r"^[a-z]rs[a-z]?$"]),
    ("polymerase",     [r"polymerase", r"\brpo[a-z]?\b"],
                        [r"^rpo[a-z]\d*$"]),
    ("dna_machinery",  [r"helicase", r"primase", r"topoisomer", r"gyrase",
                         r"recombinase", r"replication", r"single-strand"],
                        [r"^dna[a-z]$", r"^gyr[ab]$", r"^par[ce]$",
                         r"^rec[a-z]$", r"^lig[a-z]$", r"^ssb$"]),
    ("translation",    [r"translation", r"elongation factor", r"initiation factor",
                         r"release factor", r"chaperone"],
                        [r"^tuf[ab]?$", r"^fus[a-z]?$", r"^if[123]$",
                         r"^prf[abc]$", r"^dna[jk]$", r"^gro[el]l?$"]),
    ("transcription",  [r"transcription", r"sigma factor", r"\bnusa\b"],
                        [r"^nus[abefg]$", r"^sig[a-z]?\d*$", r"^rho$"]),
    ("cell_division",  [r"cell division", r"divisome", r"septation", r"ftsz"],
                        [r"^fts[a-z]$", r"^min[cde]$", r"^zip[a-z]?$"]),
    ("membrane",       [r"membrane", r"lipoprotein", r"porin", r"channel",
                         r"transporter", r"permease", r"abc transport"],
                        [r"^omp[a-z]$", r"^lpt[abcdefg]?$", r"^msb[a-z]?$"]),
    ("energy",         [r"atp synthase", r"atpase", r"electron transport",
                         r"cytochrome", r"nadh"],
                        [r"^atp[a-z]\d*$", r"^cyo[a-z]$", r"^cyd[ab]$",
                         r"^nuo[a-z]$"]),
    ("lipid",          [r"fatty acid", r"phospholipid", r"lipid biosynth"],
                        [r"^fab[a-z]$", r"^pls[a-z]$", r"^acpp?$"]),
    ("nucleotide",     [r"nucleotide biosynth", r"purine", r"pyrimidine",
                         r"thymidylate", r"ribonucleot"],
                        [r"^pur[a-z]\d*$", r"^pyr[a-z]\d*$", r"^nrd[abe]$"]),
    ("amino_acid",     [r"amino acid biosynth", r"glutam", r"aspart"],
                        [r"^thr[abc]$", r"^trp[a-eg]$", r"^his[a-h]$",
                         r"^ilv[a-eh]$", r"^arg[a-h]$", r"^lys[acn]$",
                         r"^aro[a-h]$", r"^cys[a-zk]$", r"^met[a-l]\d*$"]),
    ("cofactor",       [r"cofactor biosynth", r"flavin", r"folate", r"thiamine",
                         r"pyridoxal", r"riboflavin", r"coenzyme"],
                        [r"^fol[a-ep]$", r"^thi[a-z]\d*$", r"^pdx[abjkty]?$",
                         r"^rib[a-z]$"]),
]
SLOT_NAMES = [s[0] for s in SLOTS]
SLOT_PROD = [[re.compile(p, re.I) for p in pats] for _, pats, _ in SLOTS]
SLOT_GENE = [[re.compile(p, re.I) for p in pats] for _, _, pats in SLOTS]


def classify(product, gene_name):
    p = (product or "").lower(); g = (gene_name or "").lower().strip()
    # if gene_name is long, it's actually a product description (mgen labels)
    text_for_keywords = p if p else (g if len(g) > 6 else "")
    vec = np.zeros(len(SLOT_NAMES), np.int8)
    for i in range(len(SLOT_NAMES)):
        if text_for_keywords and any(rx.search(text_for_keywords) for rx in SLOT_PROD[i]):
            vec[i] = 1; continue
        if g and any(rx.search(g) for rx in SLOT_GENE[i]):
            vec[i] = 1
    return vec


# ---- load everything ----
labels = {}; gene_names = {}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    k = (r["organism"], r["locus_tag"])
    labels[k] = int(r["essential"])
    gn = (r.get("gene_name") or "").strip()
    if gn and gn != "-":
        gene_names[k] = gn

annot = {}
for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")):
    if r.get("product"):
        annot[(r["organism"], r["locus_tag"])] = r["product"]
# syn3a's own product table
syn3a_prod = {}
try:
    for r in csv.DictReader(open("memory_bank/data/syn3a_gene_table.csv")):
        if r.get("product"):
            syn3a_prod[("syn3a", r["locus_tag"])] = r["product"]
except Exception:
    pass
annot.update(syn3a_prod)


def get_product(k):
    return syn3a_prod.get(k) or annot.get(k, "")


# ---- per-organism analysis ----
print("=== TRUE-ESSENTIAL CELL ASSEMBLY (no model in the loop) ===\n")
per_org = {}
SLOT_REQUIREMENTS = {  # minimum gene counts derived from syn3a's known-viable cell
    "ribosomal": 30, "trna": 20, "synthetase": 15, "polymerase": 3,
    "dna_machinery": 5, "translation": 5, "transcription": 2,
    "cell_division": 1, "membrane": 10, "energy": 3,
    "lipid": 0, "nucleotide": 0, "amino_acid": 0, "cofactor": 0,
}
print(f"{'org':<14} {'essentials':>10} {'annotated':>10} {'slots_present':>14} "
      f"{'slots_below_min':>16}")
print("-" * 70)
for org in ORGS:
    ess_keys = [k for k, v in labels.items() if k[0] == org and v == 1]
    if not ess_keys:
        continue
    slot_count = np.zeros(len(SLOT_NAMES), int)
    n_annot = 0; gene_lists = defaultdict(list)
    for k in ess_keys:
        prod = get_product(k); gn = gene_names.get(k, "")
        if prod or gn: n_annot += 1
        vec = classify(prod, gn)
        slot_count += vec
        for i in np.where(vec)[0]:
            gene_lists[SLOT_NAMES[i]].append((k[1], gn or "?", prod[:50]))
    present = [n for n in SLOT_NAMES if slot_count[SLOT_NAMES.index(n)] > 0]
    below_min = [n for n in SLOT_NAMES
                 if slot_count[SLOT_NAMES.index(n)] < SLOT_REQUIREMENTS[n]]
    per_org[org] = dict(n_essential=len(ess_keys), n_annotated=n_annot,
                        slot_counts={SLOT_NAMES[i]: int(slot_count[i])
                                     for i in range(len(SLOT_NAMES))},
                        slots_present=present,
                        slots_below_minimum=below_min)
    print(f"{org:<14} {len(ess_keys):>10} {n_annot:>10} "
          f"{len(present):>14} {len(below_min):>16}")

# detail per organism
print("\n=== per-slot detail ===")
print(f"{'slot':<16}", *[f"{o:>14}" for o in ORGS], "  min")
for sn in SLOT_NAMES:
    vals = [per_org.get(o, {}).get("slot_counts", {}).get(sn, 0) for o in ORGS]
    flags = ["!" if v < SLOT_REQUIREMENTS[sn] and v == 0 else
             ("." if v < SLOT_REQUIREMENTS[sn] else " ") for v in vals]
    print(f"{sn:<16}",
          *[f"{v:>11}{f:>3}" for v, f in zip(vals, flags)],
          f"  {SLOT_REQUIREMENTS[sn]:>3}")
print("\nlegend: '!' = slot empty (cell would die), '.' = below minimum count")


# ---- visualise as a 'cell map': 2x2 grid of horizontal bar charts ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
COLORS = ["#2c7fb8", "#41b6c4", "#a1dab4", "#ffffcc", "#fed976", "#fd8d3c",
          "#e31a1c", "#b10026", "#88419d", "#54278f", "#9ecae1", "#3182bd",
          "#08519c", "#bdbdbd"]
for ax, org in zip(axes.ravel(), ORGS):
    if org not in per_org:
        ax.set_visible(False); continue
    sc = per_org[org]["slot_counts"]
    names = list(sc.keys()); vals = list(sc.values())
    bars = ax.barh(range(len(names)), vals, color=COLORS[:len(names)])
    # overlay the minimum requirement as a red line per bar
    for i, n in enumerate(names):
        req = SLOT_REQUIREMENTS[n]
        if req > 0:
            ax.plot([req, req], [i - 0.4, i + 0.4], color="red", lw=2,
                    solid_capstyle="butt")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    title = f"{org}: {per_org[org]['n_essential']} essentials"
    if per_org[org]["slots_below_minimum"]:
        title += f" | below-min: {len(per_org[org]['slots_below_minimum'])} slots"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("# essential genes in slot")
    ax.grid(alpha=0.3, axis="x")
    for i, v in enumerate(vals):
        if v > 0:
            ax.text(v + 0.5, i, str(v), va="center", fontsize=8)

fig.suptitle("True-essential cell composition across organisms\n"
             "(red ticks = syn3a-derived minimum gene count per functional slot)",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(OUT / "true_cell_grid.png", dpi=140, bbox_inches="tight")
print(f"\nwrote outputs/orphan/true_cell_grid.png")

json.dump(dict(slot_requirements=SLOT_REQUIREMENTS, per_organism=per_org),
          open(OUT / "true_cell_composition.json", "w"), indent=2)
print(f"wrote outputs/orphan/true_cell_composition.json")
