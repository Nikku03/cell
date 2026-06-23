"""Everything the essential-gene system can give -- one master figure + JSON.

Six panels, six answers:
  1 Law of essential genome size  (how small can a cell be?)
  2 Essentiality is bimodal       (intrinsic vs contextual)
  3 The universal irreducible core (what every cell must have)
  4 Duplication buffers essentiality (where the AND gate softens)
  5 The AND gate of viability     (remove one essential -> death)
  6 Robust-yet-fragile genome     (random vs targeted loss)

Output: outputs/orphan/complete_findings.png
        outputs/orphan/complete_findings.json
"""
import csv, json, re
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/orphan")

# ---------- load ----------
labels = {}; per_org = defaultdict(lambda: [0, 0])
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    k = (r["organism"], r["locus_tag"]); e = int(r["essential"])
    labels[k] = e; per_org[r["organism"]][0] += 1; per_org[r["organism"]][1] += e
og_of = {}; og_present = defaultdict(set); og_ess = defaultdict(set)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"):
        og_of[(r["organism"], r["locus_tag"])] = r["og_id"]
for k, e in labels.items():
    og = og_of.get(k)
    if og:
        og_present[og].add(k[0])
        if e: og_ess[og].add(k[0])
annot = {}
for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")):
    if r.get("product"): annot[(r["organism"], r["locus_tag"])] = r["product"]
memb = defaultdict(list)
for k, og in og_of.items(): memb[og].append(k)
def og_product(og):
    ps = [annot.get(k, "") for k in memb[og]]
    ps = [p for p in ps if p and p.lower() != "hypothetical protein"]
    return Counter(ps).most_common(1)[0][0] if ps else ""

# ---------- 1: law of essential genome size ----------
# minimal cells (mostly-essential genomes) define the floor
MINIMAL = {"syn3a", "mgen", "mpne"}
gx, gy, gc_, glab = [], [], [], []
for o, (tot, e) in per_org.items():
    if e == 0: continue
    gx.append(tot); gy.append(e)
    gc_.append("#e31a1c" if o in MINIMAL else "#3182bd")
    glab.append(o)

# ---------- 2: bimodal essentiality ----------
rate = np.array([len(og_ess[og]) / len(og_present[og])
                 for og in og_present if len(og_present[og]) >= 20])

# ---------- 3: universal core composition ----------
CATS = [
    ("translation", [r"ribosom", r"--trna", r"trna ligase", r"elongation factor",
                     r"release factor", r"initiation factor", r"aminoacyl", r"\btrna\b"]),
    ("transcription", [r"rna polymerase", r"transcription", r"sigma"]),
    ("replication", [r"dna polymerase", r"helicase", r"gyrase", r"primase", r"topoisomer", r"replicat"]),
    ("energy", [r"atp synthase", r"dehydrogenase", r"cytochrome", r"nadh"]),
    ("nucleotide", [r"purine", r"pyrimidine", r"nucleotide", r"ctp synthase", r"ribonucleot", r"pyrophosphokinase"]),
    ("amino-acid", [r"amino acid", r"glutam", r"aspart"]),
    ("lipid/membr", [r"fatty acid", r"\bacp\b", r"lipid", r"membrane", r"phospholipid"]),
    ("cofactor", [r"folate", r"thiamine", r"riboflavin", r"coenzyme", r"cofactor", r"prenyl"]),
    ("chaperone", [r"chaperon", r"groel"]),
]
def categorize(p):
    p = (p or "").lower()
    for nm, pats in CATS:
        if any(re.search(x, p) for x in pats): return nm
    return "other/metab"
univ = [og for og in og_present if len(og_present[og]) >= 25 and len(og_ess[og]) == len(og_present[og])]
ucomp = Counter(categorize(og_product(og)) for og in univ)

# ---------- 4: paralog redundancy ----------
by_org_og = defaultdict(list)
for (o, lt), og in og_of.items(): by_org_og[(o, og)].append(lt)
para_total = para_all_non = 0
para_ess_dist = Counter()
for (o, og), m in by_org_og.items():
    if len(m) >= 2:
        para_total += 1
        ne = sum(labels.get((o, lt), 0) for lt in m)
        para_ess_dist[ne] += 1
        if ne == 0: para_all_non += 1

# ---------- 5: AND gate (E. coli BW25113) ----------
ORG = "beril_Keio"
gl = [g for (o, g) in labels if o == ORG]
E = sum(labels[(ORG, g)] for g in gl); Ntot = len(gl); NE = Ntot - E
def p_viable(k):
    if k > NE: return 0.0
    lp = sum(np.log((NE - i) / (Ntot - i)) for i in range(k)); return float(np.exp(lp))
ks = np.arange(0, 40); viab = [p_viable(int(k)) for k in ks]
half = next(int(k) for k in ks if viab[k] <= 0.5)

# ---------- 6: robust-yet-fragile (whole genome) ----------
ess_mask = np.array([labels[(ORG, g)] == 1 for g in gl])
rng = np.random.default_rng(0)
fr = np.linspace(0, 0.30, 13)
ess_lost_rand = []
for f in fr:
    k = int(f * Ntot)
    ess_lost_rand.append(np.mean([ess_mask[rng.choice(Ntot, k, replace=False)].sum() / E
                                  for _ in range(150)]) if k else 0)
ess_lost_targ = [min(1.0, (f * Ntot) / E) for f in fr]

# ---------- figure ----------
fig = plt.figure(figsize=(19, 11))
gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.26)

ax = fig.add_subplot(gs[0, 0])
ax.scatter(gx, gy, c=gc_, s=28, alpha=0.8, edgecolor="white", linewidth=0.4)
for o, x, y in zip(glab, gx, gy):
    if o in MINIMAL or o in ("beril_Keio", "mtub", "reut_full"):
        ax.annotate(o.replace("beril_", ""), (x, y), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
ax.axhspan(370, 460, color="#e31a1c", alpha=0.08)
ax.text(max(gx)*0.98, 415, "minimal-cell floor\n~380-460 genes",
        ha="right", fontsize=8, color="#e31a1c")
ax.set_xlabel("genome size (annotated genes)"); ax.set_ylabel("# essential genes")
ax.set_title("1. Law of essential genome size\n(red = wet-lab minimal cells)",
             fontweight="bold", fontsize=11); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
ax.hist(rate, bins=20, color="#3182bd", edgecolor="white")
ax.axvspan(0.999, 1.02, color="#e31a1c", alpha=0.12)
ax.axvspan(-0.02, 0.001, color="#888", alpha=0.12)
ax.set_xlabel("fraction of genomes where OG is essential")
ax.set_ylabel("# orthogroups")
ax.set_title("2. Essentiality is bimodal\n(hard core + dispensable shell + conditional middle)",
             fontweight="bold", fontsize=11)

ax = fig.add_subplot(gs[0, 2])
items = ucomp.most_common()
ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color="#e31a1c")
ax.set_xlabel("# orthogroups")
ax.set_title(f"3. The universal irreducible core ({len(univ)} OGs)\n"
             "essential in EVERY genome -> translation dominates",
             fontweight="bold", fontsize=11)

ax = fig.add_subplot(gs[1, 0])
xs = sorted(para_ess_dist)
ax.bar([str(x) for x in xs], [para_ess_dist[x] for x in xs], color="#41b6c4",
       edgecolor="white")
ax.set_xlabel("# essential copies in a paralog group")
ax.set_ylabel("# paralog groups")
ax.set_title(f"4. Duplication buffers essentiality\n"
             f"{100*para_all_non//para_total}% of {para_total:,} paralog groups fully dispensable",
             fontweight="bold", fontsize=11)

ax = fig.add_subplot(gs[1, 1])
ax.plot(ks, viab, color="#e31a1c", lw=2.4)
ax.fill_between(ks, viab, alpha=0.18, color="#e31a1c")
ax.axvline(half, color="#888", ls="--", lw=0.8)
ax.text(half + 0.4, 0.55, f"50% dead\nat k={half}", fontsize=9)
ax.set_xlabel("# random gene knockouts"); ax.set_ylabel("P(cell viable)")
ax.set_title("5. The AND gate of viability\nremove ONE essential -> cell dies",
             fontweight="bold", fontsize=11); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 2])
ax.plot(fr*100, ess_lost_rand, "o-", color="#3182bd", lw=2, label="random over genome")
ax.plot(fr*100, ess_lost_targ, "s-", color="#e31a1c", lw=2, label="targeted (essentials first)")
ax.set_xlabel("% of genome removed"); ax.set_ylabel("fraction of essentials lost")
ax.set_title("6. Robust-yet-fragile\n(genome tolerates random loss, not targeted)",
             fontweight="bold", fontsize=11); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.text(15, 0.04, "any value > 0 = dead cell", fontsize=8, color="#444")

fig.suptitle("The bacterial essential-gene system -- complete findings",
             fontweight="bold", fontsize=15, y=0.98)
plt.savefig(OUT / "complete_findings.png", dpi=120, bbox_inches="tight")
print("wrote outputs/orphan/complete_findings.png")

json.dump(dict(
    law_min_floor=[per_org[o][1] for o in MINIMAL if o in per_org],
    n_organisms=len(per_org),
    bimodal=dict(universal=int((rate >= 0.999).sum()),
                 never=int((rate <= 0.001).sum()),
                 conditional=int(((rate >= 0.2) & (rate < 0.8)).sum()),
                 total=len(rate)),
    universal_core_n=len(univ), universal_core_composition=dict(ucomp),
    paralog_groups=para_total, paralog_fully_dispensable=para_all_non,
    paralog_pct_dispensable=round(100*para_all_non/para_total, 1),
    and_gate_half_dead_k=half,
    ecoli=dict(genome=Ntot, essential=E, p_ess=round(E/Ntot, 3)),
), open(OUT / "complete_findings.json", "w"), indent=2)
print("wrote outputs/orphan/complete_findings.json")
print(f"\nuniversal core: {len(univ)} OGs | paralog dispensable: "
      f"{100*para_all_non//para_total}% | AND-gate 50% dead at k={half}")
