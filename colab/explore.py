"""Exploration: what can the essential-gene system actually answer?

Three questions, three answers, one figure:

  Q1 Is essentiality intrinsic to a gene's function, or context-dependent?
     -> per-orthogroup essentiality rate across the 48 genomes is BIMODAL:
        a hard universal core + a dispensable conserved shell + a conditional
        middle.
  Q2 What is universally essential vs conditionally essential, by function?
     -> the universal core is the translation apparatus; metabolism dominates
        the conditional middle.
  Q3 How robust is the cell's essential network to gene loss?
     -> robust-yet-fragile: tolerant to random loss, collapses under targeted
        hub removal (scale-free signature).

Output: outputs/orphan/explore_findings.png
        outputs/orphan/explore_findings.json
"""
import csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/orphan")

# ---------- load ----------
labels = {}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    labels[(r["organism"], r["locus_tag"])] = int(r["essential"])
og_of = {}; og_present = defaultdict(set); og_ess = defaultdict(set)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"):
        og_of[(r["organism"], r["locus_tag"])] = r["og_id"]
for (o, lt), e in labels.items():
    og = og_of.get((o, lt))
    if og:
        og_present[og].add(o)
        if e == 1:
            og_ess[og].add(o)
annot = {}
for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")):
    if r.get("product"):
        annot[(r["organism"], r["locus_tag"])] = r["product"]
memb = defaultdict(list)
for k, og in og_of.items():
    memb[og].append(k)
def og_product(og):
    for k in memb[og]:
        if k in annot:
            return annot[k]
    return ""

# ---------- Q1: per-OG essentiality rate (>=20 orgs present) ----------
rate = []; broad = []
for og, present in og_present.items():
    if len(present) >= 20:
        rt = len(og_ess[og]) / len(present)
        rate.append(rt); broad.append((og, len(present), rt))
rate = np.array(rate)
bins = dict(
    universal=int((rate >= 0.999).sum()),
    high=int(((rate >= 0.8) & (rate < 0.999)).sum()),
    conditional=int(((rate >= 0.2) & (rate < 0.8)).sum()),
    rare=int(((rate > 0.001) & (rate < 0.2)).sum()),
    never=int((rate <= 0.001).sum()),
)

# ---------- Q2: functional composition universal vs conditional ----------
CATS = [
    ("translation/ribosome", [r"ribosom", r"trna ligase", r"trna", r"elongation factor",
                              r"release factor", r"initiation factor", r"aminoacyl", r"--trna"]),
    ("transcription",        [r"rna polymerase", r"transcription", r"sigma"]),
    ("replication/repair",   [r"dna polymerase", r"helicase", r"gyrase", r"primase",
                              r"topoisomer", r"replicat", r"ligase$"]),
    ("energy",               [r"atp synthase", r"dehydrogenase", r"cytochrome", r"nadh"]),
    ("nucleotide",           [r"purine", r"pyrimidine", r"nucleotide", r"ctp synthase",
                              r"ribonucleot", r"pyrophosphokinase"]),
    ("amino-acid",           [r"amino acid", r"glutam", r"aspart", r"synthase"]),
    ("lipid/membrane",       [r"fatty acid", r"acp", r"lipid", r"membrane", r"phospholipid"]),
    ("cofactor",             [r"folate", r"thiamine", r"riboflavin", r"coenzyme", r"cofactor"]),
    ("chaperone/other",      [r"chaperon", r"groel", r"protease", r"peptidase"]),
]
def categorize(prod):
    p = (prod or "").lower()
    for name, pats in CATS:
        if any(re.search(x, p) for x in pats):
            return name
    return "unclassified"
univ_ogs = [og for og, np_, rt in broad if rt >= 0.999]
cond_ogs = [og for og, np_, rt in broad if 0.2 <= rt < 0.8]
def comp(ogs):
    c = defaultdict(int)
    for og in ogs:
        c[categorize(og_product(og))] += 1
    return c
univ_comp = comp(univ_ogs); cond_comp = comp(cond_ogs)

# ---------- Q3: network robustness (E. coli Keio) ----------
ORG = "beril_Keio"
ess = [lt for (o, lt) in labels if o == ORG and labels[(o, lt)] == 1]
ess_set = set(ess); idx = {lt: i for i, lt in enumerate(ess)}; n = len(ess)
n_orgs = len({o for s in og_present.values() for o in s})
synt = {}
for r in csv.DictReader(open("data/drive_import/labels/synteny_features.csv")):
    if r["organism"] == ORG:
        synt[r["locus_tag"]] = (r.get("prev_locus_tag", ""), r.get("next_locus_tag", ""))
edges = set()
for lt in ess:
    for nb in synt.get(lt, ("", "")):
        if nb in ess_set and nb != lt:
            edges.add(tuple(sorted((idx[lt], idx[nb]))))
frac = {}; pv = {}
for lt in ess:
    og = og_of.get((ORG, lt))
    if og:
        pv[lt] = og_present[og]; frac[lt] = len(og_present[og]) / n_orgs
keys = [lt for lt in ess if lt in pv and 0.10 <= frac[lt] < 0.90]
for a in range(len(keys)):
    va = pv[keys[a]]
    for b in range(a + 1, len(keys)):
        vb = pv[keys[b]]; inter = len(va & vb)
        if inter and inter / (len(va) + len(vb) - inter) >= 0.9:
            edges.add(tuple(sorted((idx[keys[a]], idx[keys[b]]))))
adj = defaultdict(set)
for a, b in edges:
    adj[a].add(b); adj[b].add(a)
deg = np.array([len(adj[i]) for i in range(n)])
def giant(removed):
    seen = set(removed); best = 0
    for s in range(n):
        if s in seen:
            continue
        stack = [s]; seen.add(s); sz = 0
        while stack:
            u = stack.pop(); sz += 1
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        best = max(best, sz)
    return best
order = list(np.argsort(-deg))
rng = np.random.default_rng(0); rand = list(rng.permutation(n))
fracs = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
g_targ = [giant(set(order[:int(f * n)])) / n for f in fracs]
g_rand = [np.mean([giant(set(list(rng.permutation(n))[:int(f * n)])) for _ in range(3)]) / n
          for f in fracs]

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))

ax[0].hist(rate, bins=20, color="#3182bd", edgecolor="white")
ax[0].set_title("Q1  Essentiality is bimodal, not binary\n"
                f"(per-orthogroup essential-rate, {len(rate)} OGs in ≥20 genomes)",
                fontsize=11, fontweight="bold")
ax[0].set_xlabel("fraction of genomes where the orthogroup is essential")
ax[0].set_ylabel("# orthogroups")
ax[0].axvspan(0.999, 1.02, color="#e31a1c", alpha=0.12)
ax[0].axvspan(-0.02, 0.001, color="#888", alpha=0.12)
ax[0].text(0.985, ax[0].get_ylim()[1]*0.9, f"universal\ncore\n{bins['universal']}",
           ha="right", fontsize=8, color="#e31a1c", fontweight="bold")
ax[0].text(0.03, ax[0].get_ylim()[1]*0.9, f"never\nessential\n{bins['never']}",
           ha="left", fontsize=8, color="#555")

cats = [c for c, _ in CATS] + ["unclassified"]
uv = [univ_comp.get(c, 0) for c in cats]
cv = [cond_comp.get(c, 0) for c in cats]
uv_n = np.array(uv) / max(sum(uv), 1); cv_n = np.array(cv) / max(sum(cv), 1)
yp = np.arange(len(cats))
ax[1].barh(yp - 0.2, uv_n, height=0.4, color="#e31a1c", label=f"universal-essential ({sum(uv)})")
ax[1].barh(yp + 0.2, cv_n, height=0.4, color="#fd8d3c", label=f"conditional ({sum(cv)})")
ax[1].set_yticks(yp); ax[1].set_yticklabels(cats, fontsize=8.5); ax[1].invert_yaxis()
ax[1].set_xlabel("fraction of class")
ax[1].set_title("Q2  Universal core = translation;\nconditional = metabolism",
                fontsize=11, fontweight="bold")
ax[1].legend(fontsize=8, loc="lower right")

ax[2].plot([f*100 for f in fracs], g_targ, "o-", color="#e31a1c", lw=2,
           label="targeted (remove hubs)")
ax[2].plot([f*100 for f in fracs], g_rand, "s-", color="#3182bd", lw=2,
           label="random failure")
ax[2].set_title("Q3  Robust-yet-fragile cell network\n"
                "(scale-free signature)", fontsize=11, fontweight="bold")
ax[2].set_xlabel("% of essential genes removed")
ax[2].set_ylabel("giant component (fraction of network)")
ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "explore_findings.png", dpi=130, bbox_inches="tight")
print("wrote outputs/orphan/explore_findings.png")

json.dump(dict(
    q1_essentiality_distribution=bins,
    q1_total_broad_ogs=len(rate),
    q2_universal_composition=dict(univ_comp),
    q2_conditional_composition=dict(cond_comp),
    q2_universal_examples=[(og, len(og_present[og]), og_product(og))
                           for og in sorted(univ_ogs, key=lambda o: -len(og_present[o]))[:15]],
    q3_robustness=dict(removed_pct=[f*100 for f in fracs],
                       targeted_giant_frac=[round(x, 3) for x in g_targ],
                       random_giant_frac=[round(x, 3) for x in g_rand]),
), open(OUT / "explore_findings.json", "w"), indent=2)
print("wrote outputs/orphan/explore_findings.json")
print("\nQ1 bins:", bins)
print("Q2 universal:", dict(univ_comp))
print("Q3 targeted@20%:", round(g_targ[5],3), "random@20%:", round(g_rand[5],3))
