"""Lay out the FULLY-FLEDGED E. coli cell -- spatial layout of all 3,585 genes
across functional modules, with operon edges, essential/non-essential coloring,
and universal-core marker. Same module-clustered layout we used for the
essentials-only network, but now extended to the whole genome.

Honest scope:
  - 3,073 of 3,585 Keio annotations are OG-PROPAGATED (no direct GFF for
    Keio's numeric locus_tags), so slot membership is correct at the FAMILY
    level but specific gene-level annotations may be lumped.
  - Layout is the same numpy-only deterministic module-clustered placement
    we used before; no force-directed run (we already showed that degenerates
    for ~600 nodes; 3,585 is worse).

Output: outputs/orphan/full_cell_layout.png
        outputs/orphan/full_cell_layout.json
"""
import csv, re, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/orphan")
ORG = "beril_Keio"

# ---- load ----
truth = {}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"] == ORG: truth[r["locus_tag"]] = int(r["essential"])
annot = {}
for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")):
    if r["organism"] == ORG and r.get("product"):
        annot[r["locus_tag"]] = r["product"]
reg = {}
for r in csv.DictReader(open("data/drive_import/labels/regulator_features.csv")):
    if r["organism"] == ORG:
        reg[r["locus_tag"]] = dict(regulator=int(r.get("is_regulator", 0) or 0),
                                    signaling=int(r.get("is_signaling", 0) or 0),
                                    transporter=int(r.get("is_transporter", 0) or 0))
synt = {}
for r in csv.DictReader(open("data/drive_import/labels/synteny_features.csv")):
    if r["organism"] == ORG:
        synt[r["locus_tag"]] = (r.get("prev_locus_tag", ""),
                                r.get("next_locus_tag", ""))
og_of = {}; og_present = defaultdict(set)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"):
        og_of[(r["organism"], r["locus_tag"])] = r["og_id"]
        og_present[r["og_id"]].add(r["organism"])
n_orgs = len({o for s in og_present.values() for o in s})

# ---- slot classifier (same as full_cell_keio.py) ----
SLOTS = [
    ("ribosomal",     [r"ribosom"]),
    ("synthetase",    [r"--trna lig", r"trna ligase", r"synthetase", r"aminoacyl"]),
    ("trna",          [r"\btrna\b"]),
    ("transcription", [r"rna polymerase", r"transcription", r"sigma"]),
    ("replication",   [r"dna polymerase", r"helicase", r"gyrase", r"primase",
                       r"topoisomer", r"replicat", r"dna ligase", r"recombin"]),
    ("translation",   [r"elongation factor", r"initiation factor",
                       r"release factor", r"chaperon"]),
    ("cell_division", [r"cell division", r"septation", r"ftsz", r"divisome"]),
    ("membrane",      [r"membrane", r"lipoprotein", r"porin", r"\bsec[a-z]\b"]),
    ("energy",        [r"atp synthase", r"atpase", r"cytochrome", r"nadh"]),
    ("lipid",         [r"fatty acid", r"\bacp\b", r"lipid", r"phospholipid"]),
    ("nucleotide",    [r"purine", r"pyrimidine", r"nucleotide", r"ribonucleot"]),
    ("amino_acid",    [r"amino acid", r"glutam", r"aspart"]),
    ("cofactor",      [r"folate", r"thiamine", r"riboflavin", r"coenzyme",
                       r"cofactor", r"biotin"]),
    ("metabolism",    [r"oxidoreductase", r"transferase", r"hydrolase",
                       r"isomerase", r"ligase", r"lyase", r"kinase",
                       r"dehydrogenase", r"reductase", r"phosphatase"]),
    ("stress",        [r"stress", r"sos", r"heat shock", r"oxidative"]),
    ("hypothetical",  [r"hypothetical", r"\bduf\d", r"unknown function"]),
]
SP = [(nm, [re.compile(x, re.I) for x in pats]) for nm, pats in SLOTS]
def slot_of(lt):
    rg = reg.get(lt, {})
    if rg.get("regulator"): return "regulation"
    if rg.get("transporter"): return "transport"
    if rg.get("signaling"): return "signaling"
    p = (annot.get(lt, "") or "").lower()
    for nm, rx in SP:
        if any(r.search(p) for r in rx): return nm
    return "other"
slot = {lt: slot_of(lt) for lt in truth}
all_slots = list({s for s in slot.values()})
# fixed order: information-processing core at top, periphery below
ORDER = ["ribosomal", "trna", "synthetase", "transcription", "replication",
         "translation", "cell_division", "membrane", "transport", "energy",
         "lipid", "nucleotide", "amino_acid", "cofactor", "metabolism",
         "regulation", "signaling", "stress", "hypothetical", "other"]
present_slots = [s for s in ORDER if s in all_slots]
# any slot not in ORDER -> append
for s in all_slots:
    if s not in present_slots: present_slots.append(s)

# ---- universal core marker ----
def is_universal(lt):
    og = og_of.get((ORG, lt))
    return bool(og) and len(og_present[og]) / n_orgs >= 0.9

# ---- module-clustered layout ----
N = len(truth)
genes = list(truth.keys())
idx = {lt: i for i, lt in enumerate(genes)}
node_mod = np.array([present_slots.index(slot[lt]) for lt in genes])
mod_size = np.array([(node_mod == m).sum() for m in range(len(present_slots))])

# circular module centers, radius scales with cluster size
ang = np.linspace(0, 2 * np.pi, len(present_slots), endpoint=False)
R = 7.0
mod_center = {m: np.array([R * np.cos(a), R * np.sin(a)])
              for m, a in zip(range(len(present_slots)), ang)}
rng = np.random.default_rng(0)
pos = np.zeros((N, 2))
for m in range(len(present_slots)):
    sel = np.where(node_mod == m)[0]
    if len(sel) == 0: continue
    rad = 0.40 * np.sqrt(max(len(sel), 1))     # cluster disc radius
    r = rad * np.sqrt(rng.random(len(sel)))
    th = rng.random(len(sel)) * 2 * np.pi
    pos[sel] = mod_center[m] + np.column_stack([r * np.cos(th), r * np.sin(th)])

# ---- operon edges (synteny prev/next) ----
edges = []
for lt in genes:
    p, nx = synt.get(lt, ("", ""))
    for nb in (p, nx):
        if nb in idx and nb != lt:
            a, b = idx[lt], idx[nb]
            if a < b: edges.append((a, b))
edges = list(set(edges))

# ---- figure ----
fig, ax = plt.subplots(figsize=(15, 15))
ax.set_facecolor("#fafafa")
# operon edges (light gray)
for a, b in edges:
    ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],
            color="#dddddd", lw=0.18, alpha=0.6, zorder=0)
# nodes: essential = red, non-essential = blue; universal-core = ring
ess_mask = np.array([truth[lt] == 1 for lt in genes])
univ_mask = np.array([is_universal(lt) for lt in genes])
# non-essential first (background), then essential (foreground)
sizes = np.where(univ_mask, 28, 12)
ax.scatter(pos[~ess_mask, 0], pos[~ess_mask, 1],
           s=sizes[~ess_mask], color="#74a9cf",
           edgecolor="none", alpha=0.55, zorder=1)
# essential, with universal ring
ax.scatter(pos[ess_mask & ~univ_mask, 0], pos[ess_mask & ~univ_mask, 1],
           s=20, color="#e31a1c", edgecolor="none", zorder=2)
ax.scatter(pos[ess_mask & univ_mask, 0], pos[ess_mask & univ_mask, 1],
           s=42, color="#e31a1c", edgecolor="#000000", linewidth=0.7, zorder=3)

# module labels at cluster centers (push outward)
for m, s in enumerate(present_slots):
    if mod_size[m] == 0: continue
    c = mod_center[m]
    out = c * 1.20
    ax.text(out[0], out[1], f"{s}\n({mod_size[m]})", ha="center", va="center",
            fontsize=10.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#666"),
            zorder=5)

ax.set_xlim(-12, 12); ax.set_ylim(-12, 12); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"E. coli FULLY-FLEDGED cell -- spatial layout\n"
             f"{N} genes across {len(present_slots)} subsystems  |  "
             f"{ess_mask.sum()} essential (red), "
             f"{(~ess_mask).sum()} non-essential (blue)  |  "
             f"black ring = universal-core ({univ_mask.sum()} genes)",
             fontweight="bold", fontsize=12, pad=20)

# legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elem = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#e31a1c',
           markersize=9, label=f'essential ({int(ess_mask.sum())})'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#e31a1c',
           markersize=11, markeredgecolor='black', markeredgewidth=1,
           label=f'essential + universal-core ({int((ess_mask&univ_mask).sum())})'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#74a9cf',
           markersize=8, label=f'non-essential ({int((~ess_mask).sum())})'),
    Line2D([0],[0], color='#cccccc', lw=2, label=f'operon adjacency ({len(edges)})'),
]
ax.legend(handles=legend_elem, loc="lower left", fontsize=9, framealpha=0.95)

plt.tight_layout()
plt.savefig(OUT / "full_cell_layout.png", dpi=120, bbox_inches="tight")
print(f"wrote outputs/orphan/full_cell_layout.png")
print(f"  {N} genes, {len(present_slots)} modules, {len(edges)} operon edges")
print(f"  essential: {int(ess_mask.sum())} | universal-core: {int(univ_mask.sum())}")

json.dump(dict(organism=ORG, n_genes=N, n_essential=int(ess_mask.sum()),
               n_universal_core=int(univ_mask.sum()),
               n_operon_edges=len(edges),
               module_sizes={present_slots[m]: int(mod_size[m])
                             for m in range(len(present_slots))}),
          open(OUT / "full_cell_layout.json", "w"), indent=2)
print("wrote outputs/orphan/full_cell_layout.json")
