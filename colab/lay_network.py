"""Lay the cellular network for E. coli's essential gene set.

Turns the 605 Keio essential genes from a parts-list into a wired system:

  NODES = essential genes, coloured by functional module (slot)
  EDGES =
    (a) OPERON adjacency  -- genes consecutive on the genome (synteny prev/next)
    (b) PHYLETIC COUPLING -- gene pairs whose orthogroups co-occur across the
        59 organisms with Jaccard >= 0.9 (co-inherited => functionally linked)

Then: connectivity (is the essential set one connected system or fragments?),
module-to-module wiring (does information flow link to metabolism link to
membrane?), and two figures -- gene-level graph + module-level system diagram.

Pure numpy (Fruchterman-Reingold layout); no networkx.
Output: outputs/orphan/cell_network_genes.png
        outputs/orphan/cell_network_modules.png
        outputs/orphan/cell_network.json
"""
import csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/orphan")
ORG = "beril_Keio"

# ---- functional modules (reuse the slot classifier) ----
SLOTS = [
    ("ribosomal",     [r"ribosom", r"\b30s\b", r"\b50s\b"], [r"^rp[sl][a-z]?\d*$", r"^rpm[a-z]\d*$"]),
    ("trna",          [r"\btrna\b", r"transfer rna"],        [r"^trn", r"^trna-"]),
    ("synthetase",    [r"synthetase", r"trna ligase", r"aminoacyl"], [r"^[a-z]rs[a-z]?$"]),
    ("transcription", [r"polymerase", r"transcription", r"sigma factor"], [r"^rpo[a-z]", r"^nus", r"^sig"]),
    ("replication",   [r"helicase", r"gyrase", r"primase", r"topoisomer", r"replication"],
                      [r"^dna[a-z]$", r"^gyr[ab]$", r"^par[ce]$", r"^lig[a-z]$", r"^ssb$", r"^hol[a-z]$"]),
    ("translation",   [r"elongation factor", r"initiation factor", r"release factor", r"chaperone"],
                      [r"^tuf", r"^fus", r"^if[123]", r"^prf", r"^dna[jk]$", r"^gro"]),
    ("cell_division", [r"cell division", r"septation", r"ftsz"], [r"^fts", r"^min[cde]$"]),
    ("membrane",      [r"membrane", r"lipoprotein", r"transporter", r"permease", r"porin"],
                      [r"^omp", r"^lpt", r"^msb", r"^sec[a-z]$", r"^lol[a-z]$"]),
    ("energy",        [r"atp synthase", r"atpase", r"cytochrome", r"nadh", r"electron transport"],
                      [r"^atp[a-z]", r"^nuo", r"^cyo", r"^cyd"]),
    ("lipid",         [r"fatty acid", r"phospholipid", r"lipid"], [r"^fab", r"^pls", r"^acp"]),
    ("nucleotide",    [r"purine", r"pyrimidine", r"nucleotide", r"ribonucleot"], [r"^pur", r"^pyr", r"^nrd"]),
    ("amino_acid",    [r"amino acid", r"glutam", r"aspart"], [r"^thr[abc]$", r"^trp", r"^his[a-h]$", r"^aro", r"^ilv"]),
    ("cofactor",      [r"cofactor", r"folate", r"thiamine", r"riboflavin", r"coenzyme"], [r"^fol", r"^thi", r"^rib", r"^pdx"]),
    ("other",         [], []),
]
SLOT_NAMES = [s[0] for s in SLOTS]
SLOT_PROD = [[re.compile(p, re.I) for p in pats] for _, pats, _ in SLOTS]
SLOT_GENE = [[re.compile(p, re.I) for p in pats] for _, _, pats in SLOTS]


def classify(product, gene_name):
    p = (product or "").lower(); g = (gene_name or "").lower().strip()
    txt = p if p else (g if len(g) > 6 else "")
    for i in range(len(SLOT_NAMES) - 1):
        if txt and any(rx.search(txt) for rx in SLOT_PROD[i]):
            return i
        if g and any(rx.search(g) for rx in SLOT_GENE[i]):
            return i
    return len(SLOT_NAMES) - 1   # "other"


# ---- load Keio essentials + annotation + synteny + OG presence ----
labels = {}; gene_names = {}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    labels[(r["organism"], r["locus_tag"])] = int(r["essential"])
    gn = (r.get("gene_name") or "").strip()
    if gn and gn != "-":
        gene_names[(r["organism"], r["locus_tag"])] = gn
annot = {}
for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")):
    if r["organism"] == ORG and r.get("product"):
        annot[(ORG, r["locus_tag"])] = r["product"]

ess = [lt for (o, lt) in labels if o == ORG and labels[(o, lt)] == 1]
ess_set = set(ess)
idx = {lt: i for i, lt in enumerate(ess)}
n = len(ess)
print(f"E. coli essential nodes: {n}")

# node module + label
node_mod = np.zeros(n, int); node_lbl = []
for lt in ess:
    prod = annot.get((ORG, lt), ""); gn = gene_names.get((ORG, lt), "")
    node_mod[idx[lt]] = classify(prod, gn)
    node_lbl.append(gn or lt)

# OG per gene + presence
og_by_gene = {}; og_present = defaultdict(set)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"):
        og_by_gene[(r["organism"], r["locus_tag"])] = r["og_id"]
        og_present[r["og_id"]].add(r["organism"])

# ---- EDGES (a) operon adjacency ----
edges = set()
synt = {}
for r in csv.DictReader(open("data/drive_import/labels/synteny_features.csv")):
    if r["organism"] == ORG:
        synt[r["locus_tag"]] = (r.get("prev_locus_tag", ""), r.get("next_locus_tag", ""))
n_operon = 0
for lt in ess:
    prev, nxt = synt.get(lt, ("", ""))
    for nb in (prev, nxt):
        if nb in ess_set and nb != lt:
            e = tuple(sorted((idx[lt], idx[nb])))
            if e not in edges:
                edges.add(e); n_operon += 1
print(f"operon edges: {n_operon}")

# ---- EDGES (b) phyletic coupling (PHYLOGENETIC PROFILING) ----
# Universal genes (present in ~all orgs) carry NO differential co-evolution
# signal: any two all-present profiles trivially have Jaccard~1, which would
# fuse the whole conserved core into one meaningless clique. Standard
# phylogenetic profiling therefore couples only genes with VARIABLE presence.
ess_ogs = [(lt, og_by_gene.get((ORG, lt))) for lt in ess]
present_vecs = {}; frac = {}
all_orgs = sorted({o for s in og_present.values() for o in s})
n_orgs = len(all_orgs)
oi = {o: k for k, o in enumerate(all_orgs)}
for lt, og in ess_ogs:
    if og:
        v = np.zeros(n_orgs, bool)
        for o in og_present[og]:
            v[oi[o]] = True
        present_vecs[lt] = v; frac[lt] = v.sum() / n_orgs
UNIV = 0.90                       # >=90% of genomes => "universal core"
universal = np.zeros(n, bool)
for lt in ess:
    if frac.get(lt, 0.0) >= UNIV:
        universal[idx[lt]] = True
n_univ = int(universal.sum())
# variable-presence genes only (informative profiles)
keys = [lt for lt, _ in ess_ogs
        if lt in present_vecs and 0.10 <= frac[lt] < UNIV]
n_phyl = 0
for a in range(len(keys)):
    va = present_vecs[keys[a]]; sa = va.sum()
    for b in range(a + 1, len(keys)):
        vb = present_vecs[keys[b]]
        inter = (va & vb).sum()
        if inter == 0:
            continue
        union = sa + vb.sum() - inter
        if union > 0 and inter / union >= 0.9:
            e = tuple(sorted((idx[keys[a]], idx[keys[b]])))
            if e not in edges:
                edges.add(e); n_phyl += 1
print(f"universal-core genes (>= {int(UNIV*100)}% of {n_orgs} genomes): {n_univ}")
print(f"phyletic-coupling edges (variable genes only): {n_phyl}")
print(f"total edges: {len(edges)}")

# ---- connectivity (union-find) ----
parent = list(range(n))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
for a, b in edges:
    union(a, b)
comp = defaultdict(list)
for i in range(n):
    comp[find(i)].append(i)
comps = sorted(comp.values(), key=len, reverse=True)
giant = comps[0]
# "wired" = in the biological giant component OR part of the universal core
univ_set = {idx[lt] for lt in ess if universal[idx[lt]]}
wired = set(giant) | univ_set
n_wired = len(wired)
print(f"\nconnected components (biological edges): {len(comps)}")
print(f"giant component: {len(giant)} nodes ({len(giant)/n*100:.0f}% of essentials)")
print(f"isolated (degree 0): {sum(1 for c in comps if len(c)==1)}")
print(f"wired (giant component + universal core): {n_wired}/{n} "
      f"({n_wired/n*100:.0f}%)")

# ---- deterministic module-clustered "cell map" layout ----
# Force-directed layout degenerates here (universal core has no phyletic edges
# => one ball; 200 edge-less genes => useless ring). Instead place each gene in
# its functional-module cluster arranged around a ring; cross-module edges then
# visibly trace the wiring between subsystems. Intra-module edges stay short.
adj = defaultdict(list)
for a, b in edges:
    adj[a].append(b); adj[b].append(a)
deg = np.array([len(adj[i]) for i in range(n)])
mod_size_all = np.array([(node_mod == m).sum() for m in range(len(SLOT_NAMES))])
present_mods = [m for m in range(len(SLOT_NAMES)) if mod_size_all[m] > 0]
ang = np.linspace(0, 2 * np.pi, len(present_mods), endpoint=False)
mod_center = {m: np.array([3.5 * np.cos(a), 3.5 * np.sin(a)])
              for m, a in zip(present_mods, ang)}
rng = np.random.default_rng(0)
pos = np.zeros((n, 2))
for m in present_mods:
    sel = np.where(node_mod == m)[0]
    rad = 0.30 * np.sqrt(max(len(sel), 1))      # disc grows with cluster size
    r = rad * np.sqrt(rng.random(len(sel)))     # uniform-in-disc
    th = rng.random(len(sel)) * 2 * np.pi
    pos[sel] = mod_center[m] + np.column_stack([r * np.cos(th), r * np.sin(th)])

# ---- gene-level figure ----
COLORS = plt.cm.tab20(np.linspace(0, 1, len(SLOT_NAMES)))
fig, ax = plt.subplots(figsize=(13, 13))
# edges: phyletic (faint), operon (darker); cross-module edges trace wiring
operon_set = set()
for lt in ess:
    prev, nxt = synt.get(lt, ("", ""))
    for nb in (prev, nxt):
        if nb in ess_set and nb != lt:
            operon_set.add(tuple(sorted((idx[lt], idx[nb]))))
for a, b in edges:
    if (a, b) in operon_set:
        ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],
                color="#d95f0e", lw=0.7, alpha=0.7, zorder=1)
    else:
        ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],
                color="#74a9cf", lw=0.25, alpha=0.4, zorder=0)
for m in range(len(SLOT_NAMES)):
    sel = node_mod == m
    if sel.any():
        ew = np.where(universal[sel], 1.4, 0.3)   # ring = universal core
        ax.scatter(pos[sel, 0], pos[sel, 1], s=22 + 9 * deg[sel],
                   color=COLORS[m], label=f"{SLOT_NAMES[m]} ({int(sel.sum())})",
                   edgecolor="k", linewidth=ew, zorder=2)
# module labels at each cluster center
for m in present_mods:
    c = mod_center[m]
    ax.text(c[0], c[1] + 0.30 * np.sqrt(mod_size_all[m]) + 0.15,
            SLOT_NAMES[m], ha="center", fontsize=11, fontweight="bold",
            color="#222", zorder=5)
ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.9,
          title="functional module")
ax.set_title(f"E. coli essential-gene network ({n} genes, {len(edges)} edges)\n"
             f"functional modules wired by operon adjacency (orange) + phyletic "
             f"coupling (blue)\n"
             f"black ring = {n_univ} universal-core genes  |  wired "
             f"{n_wired}/{n} ({n_wired/n*100:.0f}%)", fontweight="bold")
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(OUT / "cell_network_genes.png", dpi=130, bbox_inches="tight")
print("wrote cell_network_genes.png")

# ---- module-level system diagram ----
mod_edge = np.zeros((len(SLOT_NAMES), len(SLOT_NAMES)), int)
for a, b in edges:
    mod_edge[node_mod[a], node_mod[b]] += 1
    mod_edge[node_mod[b], node_mod[a]] += 1
mod_size = np.array([(node_mod == m).sum() for m in range(len(SLOT_NAMES))])
# circular layout of modules present
present_mods = [m for m in range(len(SLOT_NAMES)) if mod_size[m] > 0]
ang = np.linspace(0, 2 * np.pi, len(present_mods), endpoint=False)
mpos = {m: (np.cos(a), np.sin(a)) for m, a in zip(present_mods, ang)}
fig, ax = plt.subplots(figsize=(11, 11))
maxe = mod_edge.max() or 1
for ii, m1 in enumerate(present_mods):
    for m2 in present_mods[ii + 1:]:
        w = mod_edge[m1, m2]
        if w > 0:
            x1, y1 = mpos[m1]; x2, y2 = mpos[m2]
            ax.plot([x1, x2], [y1, y2], color="#888", lw=0.5 + 5 * w / maxe,
                    alpha=0.5, zorder=1)
            ax.text((x1+x2)/2, (y1+y2)/2, str(w), fontsize=7, color="#444")
for m in present_mods:
    x, y = mpos[m]
    ax.scatter([x], [y], s=200 + 40 * mod_size[m], color=COLORS[m],
               edgecolor="k", linewidth=1, zorder=2)
    ax.text(x, y - 0.12, f"{SLOT_NAMES[m]}\n({mod_size[m]})",
            ha="center", fontsize=9, fontweight="bold", zorder=3)
ax.set_title("E. coli minimal-cell module wiring\n"
             "(node size = #essential genes; edge width = #gene-gene links between modules)",
             fontweight="bold")
ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(OUT / "cell_network_modules.png", dpi=130, bbox_inches="tight")
print("wrote cell_network_modules.png")

# ---- save stats ----
json.dump(dict(
    organism=ORG, n_nodes=n, n_edges=len(edges),
    n_operon_edges=n_operon, n_phyletic_edges=n_phyl,
    n_universal_core=n_univ, n_genomes=n_orgs,
    n_components=len(comps), giant_component=len(giant),
    giant_fraction=round(len(giant) / n, 3),
    n_wired=n_wired, wired_fraction=round(n_wired / n, 3),
    n_isolated=sum(1 for c in comps if len(c) == 1),
    module_sizes={SLOT_NAMES[m]: int(mod_size[m]) for m in range(len(SLOT_NAMES))},
    top_hubs=[dict(gene=node_lbl[i], module=SLOT_NAMES[node_mod[i]], degree=int(deg[i]))
              for i in np.argsort(-deg)[:15]],
), open(OUT / "cell_network.json", "w"), indent=2)
print("wrote cell_network.json")
