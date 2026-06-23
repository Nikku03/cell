"""Build an essential-gene network for an organism with ZERO essentiality
labels -- using the patched model's predictions as nodes.

Premise: pretend E. coli (beril_Keio) has no labels. Use the cross-clade
model (escherichia clade was held out during model training, so it never saw
Keio labels) + the systems patch to predict essentials. Build the same
operon + phyletic-coupling network from those predicted essentials. Compare
against the truth network side-by-side.

This proves the network builder works on ANY genome with orthology +
synteny, even without essentiality data -- closing the loop between the
essentiality model and the systems view.

Output: outputs/orphan/lay_network_predicted.png
        outputs/orphan/lay_network_predicted.json
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
HELD_CLADE = "escherichia"

# ---------- model predictions + structural features (replicate patch setup) ----
P = np.load(OUT / "af_torch_preds_aug.npz", allow_pickle=True)
p = P["p"].astype(float); br = P["brightness"].astype(float)
y = P["y"].astype(float); clade = P["clade"].astype(str); lt = P["lt"].astype(str)
valid = ~(np.isnan(p) | np.isnan(br))

C = np.load(OUT / "af_msa_cache.npz", allow_pickle=True)
cache_orgs = [str(x) for x in C["orgs"]]; cm = C["meta_org"]; cc = C["clade"].astype(str)
org2clade = {cache_orgs[int(cm[i])]: str(cc[i]) for i in range(len(cm))}

feat = {}; orgs_by_lt = defaultdict(list)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    k = (r["organism"], r["locus_tag"])
    feat[k] = (r.get("og_id") or "",
               float(r.get("n_paralogs_in_genome") or 0),
               float(r.get("family_n_organisms") or 0),
               float(r.get("is_orphan") or 0))
    orgs_by_lt[r["locus_tag"]].append(r["organism"])

# resolve organism per prediction by clade-match on its locus_tag candidates
resolved_org = np.empty(len(p), object)
og = np.empty(len(p), object); paralog = np.zeros(len(p))
breadth = np.zeros(len(p)); is_orph = np.zeros(len(p))
for i in range(len(p)):
    cands = orgs_by_lt.get(lt[i], [])
    match = [o for o in cands if org2clade.get(o) == clade[i]]
    chosen = match[0] if len(match) == 1 else (cands[0] if len(cands) == 1 else None)
    if chosen is None:
        og[i] = ""; resolved_org[i] = ""; continue
    resolved_org[i] = chosen
    o_, np_, br_, or_ = feat[(chosen, lt[i])]
    og[i] = o_; paralog[i] = np_; breadth[i] = br_; is_orph[i] = or_

# ---------- leave-one-CLADE-out OG-universality (leak-clean) -----------------
g_sum = defaultdict(float); g_n = defaultdict(float)
c_sum = defaultdict(float); c_n = defaultdict(float)
for i in np.where(valid)[0]:
    o = og[i]
    if not o: continue
    g_sum[o] += y[i]; g_n[o] += 1
    c_sum[(o, clade[i])] += y[i]; c_n[(o, clade[i])] += 1
og_univ = np.full(len(p), np.nan)
for i in np.where(valid)[0]:
    o = og[i]
    if not o: continue
    n = g_n[o] - c_n[(o, clade[i])]
    if n > 0:
        og_univ[i] = (g_sum[o] - c_sum[(o, clade[i])]) / n

# ---------- train stacker LEAVE-ESCHERICHIA-OUT, predict for Keio ------------
def fit_logit(X, yy, l2=1.0, iters=500, lr=0.4):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]; w = np.zeros(Xs.shape[1])
    cb = np.bincount(yy.astype(int), minlength=2).astype(float)
    cw = (cb.sum() / np.maximum(cb, 1)) ** 0.5; sw = cw[yy.astype(int)]; sw /= sw.mean()
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
        g = Xs.T @ ((pr - yy) * sw) / len(X); g[1:] += l2 * w[1:] / len(X); w -= lr * g
    return w, mu, sd
def pred_logit(m, X):
    w, mu, sd = m
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), (X - mu) / sd] @ w, -30, 30)))

mask = valid & ~np.isnan(og_univ)
def feats(idxmask):
    return np.column_stack([
        p[idxmask], br[idxmask], og_univ[idxmask],
        np.log1p(paralog[idxmask]), np.log1p(breadth[idxmask]), is_orph[idxmask],
    ])
tr = mask & (clade != HELD_CLADE)
M = fit_logit(feats(tr), y[tr])
print(f"trained stacker on {tr.sum()} non-escherichia genes")

# Keio predictions: clade==escherichia AND resolved_org==beril_Keio
is_keio = (resolved_org == ORG) & mask
patched = np.full(len(p), np.nan)
patched[is_keio] = pred_logit(M, feats(is_keio))
print(f"predicted for {is_keio.sum()} Keio genes (model never saw their labels)")

# ---------- pick predicted-essential set: rank by patched, take top K --------
# K matches the cross-organism average essential fraction for genomes this size
# (~17% for E. coli-sized). For a truly unlabeled organism this is the
# realistic operating regime: rank + threshold without a held-out tuning set.
keio_idx = np.where(is_keio)[0]
keio_lt = lt[keio_idx]; keio_p = patched[keio_idx]; keio_truth = y[keio_idx]
true_ess_lt = set(keio_lt[keio_truth == 1])
K = int(round(0.17 * len(keio_idx)))
top = np.argsort(-keio_p)[:K]
pred_ess_lt = set(keio_lt[top])
both = pred_ess_lt & true_ess_lt
prec = len(both) / max(len(pred_ess_lt), 1)
rec = len(both) / max(len(true_ess_lt), 1)
print(f"\npredicted essential set (top {K}, ~17% of Keio): "
      f"precision={prec:.2f} recall={rec:.2f} overlap={len(both)}")

# ---------- gene-name + annotation lookups -----------------------------------
gene_names = {}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"] == ORG:
        gn = (r.get("gene_name") or "").strip()
        if gn and gn != "-":
            gene_names[r["locus_tag"]] = gn
annot = {}
for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")):
    if r["organism"] == ORG and r.get("product"):
        annot[r["locus_tag"]] = r["product"]

# ---------- network builder (same as lay_network.py) -------------------------
og_of = {}; og_present = defaultdict(set)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"):
        og_of[(r["organism"], r["locus_tag"])] = r["og_id"]
        og_present[r["og_id"]].add(r["organism"])
n_orgs = len({o for s in og_present.values() for o in s})
synt = {}
for r in csv.DictReader(open("data/drive_import/labels/synteny_features.csv")):
    if r["organism"] == ORG:
        synt[r["locus_tag"]] = (r.get("prev_locus_tag", ""), r.get("next_locus_tag", ""))

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
def classify(prod, gn):
    pp = (prod or "").lower(); gg = (gn or "").lower().strip()
    txt = pp if pp else (gg if len(gg) > 6 else "")
    for i in range(len(SLOT_NAMES) - 1):
        if txt and any(rx.search(txt) for rx in SLOT_PROD[i]): return i
        if gg and any(rx.search(gg) for rx in SLOT_GENE[i]): return i
    return len(SLOT_NAMES) - 1

def build_network(ess_lts):
    """build the operon + phyletic network on a given essential set."""
    ess = list(ess_lts); ess_set = set(ess)
    idx = {x: i for i, x in enumerate(ess)}; n = len(ess)
    node_mod = np.zeros(n, int); node_lbl = []
    for x in ess:
        node_mod[idx[x]] = classify(annot.get(x, ""), gene_names.get(x, ""))
        node_lbl.append(gene_names.get(x, x))
    edges = set()
    for x in ess:
        for nb in synt.get(x, ("", "")):
            if nb in ess_set and nb != x:
                edges.add(tuple(sorted((idx[x], idx[nb]))))
    frac = {}; pv = {}
    for x in ess:
        o = og_of.get((ORG, x))
        if o: pv[x] = og_present[o]; frac[x] = len(og_present[o]) / n_orgs
    keys = [x for x in ess if x in pv and 0.10 <= frac[x] < 0.90]
    for a in range(len(keys)):
        va = pv[keys[a]]; sa = len(va)
        for b in range(a + 1, len(keys)):
            vb = pv[keys[b]]; inter = len(va & vb)
            if inter and inter / (sa + len(vb) - inter) >= 0.9:
                edges.add(tuple(sorted((idx[keys[a]], idx[keys[b]]))))
    adj = defaultdict(set)
    for a, b in edges: adj[a].add(b); adj[b].add(a)
    # universal core flag
    univ = np.array([frac.get(x, 0) >= 0.90 for x in ess])
    return dict(ess=ess, n=n, idx=idx, edges=edges, adj=adj, node_mod=node_mod,
                node_lbl=node_lbl, universal=univ)

NET_T = build_network(sorted(true_ess_lt))
NET_P = build_network(sorted(pred_ess_lt))
print(f"\ntruth network    : {NET_T['n']} nodes, {len(NET_T['edges'])} edges")
print(f"predicted network: {NET_P['n']} nodes, {len(NET_P['edges'])} edges")

# ---------- layout (deterministic module-clustered) --------------------------
def layout(net, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    pos = np.zeros((net["n"], 2))
    mod_size = np.array([(net["node_mod"] == m).sum() for m in range(len(SLOT_NAMES))])
    pres = [m for m in range(len(SLOT_NAMES)) if mod_size[m] > 0]
    ang = np.linspace(0, 2 * np.pi, len(pres), endpoint=False)
    mcenter = {m: np.array([3.5 * np.cos(a), 3.5 * np.sin(a)])
               for m, a in zip(pres, ang)}
    for m in pres:
        sel = np.where(net["node_mod"] == m)[0]
        rad = 0.30 * np.sqrt(max(len(sel), 1))
        r = rad * np.sqrt(rng.random(len(sel)))
        th = rng.random(len(sel)) * 2 * np.pi
        pos[sel] = mcenter[m] + np.column_stack([r * np.cos(th), r * np.sin(th)])
    return pos, mcenter, pres, mod_size

# ---------- module composition agreement -------------------------------------
mod_truth = {SLOT_NAMES[m]: int((NET_T["node_mod"] == m).sum()) for m in range(len(SLOT_NAMES))}
mod_pred = {SLOT_NAMES[m]: int((NET_P["node_mod"] == m).sum()) for m in range(len(SLOT_NAMES))}

# ---------- figure -----------------------------------------------------------
COLORS = plt.cm.tab20(np.linspace(0, 1, len(SLOT_NAMES)))
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
for ax, (net, title) in zip(axes, [
    (NET_T, f"TRUTH (Keio labels) -- {NET_T['n']} essentials, {len(NET_T['edges'])} edges"),
    (NET_P, f"PREDICTED (model never saw Keio labels) -- {NET_P['n']} essentials, "
            f"{len(NET_P['edges'])} edges\nprec={prec:.2f}  rec={rec:.2f}  TP={len(both)}"),
]):
    pos, mcenter, pres, mod_size = layout(net)
    operon = set()
    for x in net["ess"]:
        for nb in synt.get(x, ("", "")):
            if nb in set(net["ess"]) and nb != x:
                operon.add(tuple(sorted((net["idx"][x], net["idx"][nb]))))
    for a, b in net["edges"]:
        col = "#d95f0e" if (a, b) in operon else "#74a9cf"
        lw = 0.6 if (a, b) in operon else 0.2
        al = 0.65 if (a, b) in operon else 0.35
        ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],
                color=col, lw=lw, alpha=al, zorder=0 if (a, b) not in operon else 1)
    deg = np.array([len(net["adj"][i]) for i in range(net["n"])])
    # mark FP/FN if this is the predicted network
    if title.startswith("PREDICTED"):
        is_truth = np.array([x in true_ess_lt for x in net["ess"]])
    else:
        is_truth = np.ones(net["n"], bool)
    for m in range(len(SLOT_NAMES)):
        sel = (net["node_mod"] == m)
        if not sel.any(): continue
        ew = np.where(net["universal"][sel], 1.4, 0.3)
        edge = ["#222" if t else "#cc0000" for t in is_truth[sel]]
        ax.scatter(pos[sel, 0], pos[sel, 1], s=20 + 9 * deg[sel],
                   color=COLORS[m], label=f"{SLOT_NAMES[m]} ({int(sel.sum())})",
                   edgecolor=edge, linewidth=ew, zorder=2)
    for m in pres:
        c = mcenter[m]
        ax.text(c[0], c[1] + 0.30 * np.sqrt(mod_size[m]) + 0.15,
                SLOT_NAMES[m], ha="center", fontsize=10, fontweight="bold",
                color="#222", zorder=5)
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.9, title="module")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

fig.suptitle("Same cell, built from labels (left) vs. from the patched model "
             "(right)\nred-outlined dots on the right = false-positive essentials",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "lay_network_predicted.png", dpi=120, bbox_inches="tight")
print("wrote outputs/orphan/lay_network_predicted.png")

json.dump(dict(
    held_clade=HELD_CLADE, organism=ORG,
    n_truth_essential=len(true_ess_lt),
    n_predicted_essential=K,
    overlap=len(both), precision=round(prec, 3), recall=round(rec, 3),
    truth_network=dict(nodes=NET_T['n'], edges=len(NET_T['edges'])),
    predicted_network=dict(nodes=NET_P['n'], edges=len(NET_P['edges'])),
    module_composition=dict(truth=mod_truth, predicted=mod_pred),
), open(OUT / "lay_network_predicted.json", "w"), indent=2)
print("wrote outputs/orphan/lay_network_predicted.json")
print(f"\nmodule composition:")
print(f"{'module':<16}{'truth':>8}{'predicted':>12}")
for m in SLOT_NAMES:
    print(f"  {m:<14}{mod_truth[m]:>8}{mod_pred[m]:>12}")
