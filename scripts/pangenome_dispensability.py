"""Pangenome presence/absence as a POSITIVE dispensability signal (eliminator D).

The orthogonal-eliminator test showed no $0 genome feature positively marks
dispensability -- everything reduces to inverted conservation, which murders the
rogue zone. The one untested positive-evidence channel is WITHIN-SPECIES
pangenome presence/absence: if gene X's ortholog is ABSENT in a viable sister
strain, that strain proves the function is loseable at the organism level.

GMI1000 has 3 sisters in the orthology table: BSBF1503, PSI07, UW163.
For each GMI1000 gene we compute how many sisters carry its OG.

Two competing readings, and the data decides:
  (a) DISPENSABILITY: absent-in-sister => loseable => should enrich non-essential
      AND (the hope) spare rogue essentials, since rogue genes might still be
      conserved among close Ralstonia even if globally rare.
  (b) CONDITIONALITY trap: absent-in-sister could instead flag CONDITIONAL
      essentials (essential in GMI1000 precisely because of GMI1000-specific
      context; the sister simply lacks the gene). Then it would HIT the rogue
      zone, not spare it.

Metrics vs the earlier eliminators A/B/C, plus the rogue-sparing number.

Outputs:
  outputs/orphan/pangenome_dispensability.png
  outputs/orphan/pangenome_dispensability_GMI1000.csv
  outputs/orphan/pangenome_dispensability_summary.json
"""
import csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/user/cell/outputs/orphan")
DATA = Path("/home/user/cell/data/drive_import")
ORG = "beril_RalstoniaGMI1000"
SISTERS = ["beril_RalstoniaBSBF1503", "beril_RalstoniaPSI07", "beril_RalstoniaUW163"]

# OG -> set of organisms; and GMI1000 gene -> og
og_orgs = defaultdict(set); gene_og = {}
for r in csv.DictReader(open(DATA / "labels" / "orthology_features.csv")):
    if r["og_id"]:
        og_orgs[r["og_id"]].add(r["organism"])
        if r["organism"] == ORG:
            gene_og[r["locus_tag"]] = r["og_id"]

# pull GMI1000 feature table (has essential, conservation, paralogs, length)
df = pd.read_csv(OUT / "cross_org_coverage_scores.csv")
df = df[df.organism == ORG].copy().reset_index(drop=True)
df["og"] = df.locus_tag.map(gene_og)
df["n_sisters_present"] = df.og.map(
    lambda o: sum(1 for s in SISTERS if o in og_orgs and s in og_orgs[o]) if isinstance(o, str) else 0)
df["absent_in_a_sister"] = df.n_sisters_present < len(SISTERS)
df["absent_in_all_sisters"] = df.n_sisters_present == 0
df["rogue"] = (df.essential == 1) & (df.conservation < 0.1)

base = df.essential.mean()
rogue_total = int(df.rogue.sum())
print(f"GMI1000: {len(df)} genes, base ess {base:.3f}, {rogue_total} rogue, "
      f"{df.og.notna().sum()} with an OG\n")

# ---- core test: essential rate by sister-presence ----
print("essential rate by number of sisters carrying the ortholog:")
for k in range(len(SISTERS)+1):
    sub = df[df.n_sisters_present == k]
    if len(sub) == 0: continue
    print(f"  present in {k}/3 sisters: n={len(sub):<5} ess_rate={sub.essential.mean():.3f}  "
          f"rogue_frac={sub.rogue.mean():.3f}")

# ---- eliminator D: kill genes ABSENT in >=1 sister as 'dispensable' ----
def stats(mask, label):
    killed = int(mask.sum())
    if killed == 0:
        print(f"  {label}: kills 0"); return None
    kp = float((df.essential[mask] == 0).mean())     # kill precision (truly non-ess)
    ess_killed = int((mask & (df.essential == 1)).sum())
    rogue_killed = int((mask & df.rogue).sum())
    d = dict(killed=killed, kill_frac=round(killed/len(df), 3),
             kill_precision=round(kp, 3),
             essentials_killed=ess_killed,
             essentials_killed_frac=round(ess_killed/int(df.essential.sum()), 3),
             rogue_killed=rogue_killed,
             rogue_spared=rogue_total - rogue_killed,
             rogue_spared_frac=round((rogue_total - rogue_killed)/rogue_total, 3))
    print(f"  {label}: kills {killed} ({killed/len(df):.0%})  kill-prec {kp:.3f}  "
          f"ess killed {ess_killed} ({ess_killed/int(df.essential.sum()):.1%})  "
          f"rogue spared {d['rogue_spared']}/{rogue_total} ({d['rogue_spared_frac']:.0%})")
    return d

print("\neliminator D variants (kill = 'dispensable: absent in sister'):")
D_any = stats(df.absent_in_a_sister, "D1 absent in >=1 sister")
D_all = stats(df.absent_in_all_sisters, "D2 absent in ALL 3 sisters")

# the verdict: is absent-in-sister dispensability or conditionality?
abs_sub = df[df.absent_in_a_sister]
print(f"\nVERDICT CHECK: among genes absent-in-a-sister (n={len(abs_sub)}):")
print(f"  essential rate {abs_sub.essential.mean():.3f} (vs base {base:.3f})")
print(f"  of these essentials, rogue fraction {abs_sub[abs_sub.essential==1].rogue.mean():.3f} "
      f"(vs {df[df.essential==1].rogue.mean():.3f} overall)")

# save per-gene
atlas = pd.read_csv(OUT / "gene_dot_atlas_GMI1000.csv")[["locus_tag", "product"]]
out = df.merge(atlas, on="locus_tag", how="left")
out[["locus_tag", "product", "essential", "conservation", "n_paralogs",
     "gene_len_aa", "n_sisters_present", "absent_in_a_sister", "rogue"]].to_csv(
    OUT / "pangenome_dispensability_GMI1000.csv", index=False)

# ---- figure ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# A: essential rate by sister presence
ax = axes[0]
ks = list(range(len(SISTERS)+1))
er = [df[df.n_sisters_present == k].essential.mean() for k in ks]
ns = [int((df.n_sisters_present == k).sum()) for k in ks]
ax.bar(ks, er, color="#C0392B", alpha=0.85)
for i,(e,n) in enumerate(zip(er,ns)):
    ax.text(i, e, f"{e:.2f}\nn={n}", ha="center", va="bottom", fontsize=8)
ax.axhline(base, color="gray", ls="--", lw=0.8, label=f"base {base:.2f}")
ax.set_xlabel("# sister Ralstonia strains carrying the ortholog (of 3)")
ax.set_ylabel("essential rate (GMI1000)"); ax.set_ylim(0, max(er)*1.3)
ax.set_title("A. Sister-presence vs essentiality\n(absent in sisters => less essential?)")
ax.legend(fontsize=8)

# B: rogue fraction by sister presence (the conditionality trap check)
ax = axes[1]
rf = [df[df.n_sisters_present == k].rogue.mean() for k in ks]
ax.bar(ks, rf, color="#8E44AD", alpha=0.85)
for i,r in enumerate(rf):
    ax.text(i, r, f"{r:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xlabel("# sisters carrying the ortholog")
ax.set_ylabel("rogue fraction (essential & cons<0.1)")
ax.set_title("B. Does absent-in-sister flag ROGUE essentials?\n(high bar at 0 = conditionality trap)")

# C: eliminator comparison incl. earlier A/B/C
ax = axes[2]
# load earlier summary for A/B/C
try:
    prev = json.loads((OUT / "elim_orthogonal_summary.json").read_text())
    abc = [("A essentiality", prev["A_essentiality_axis"]["kill_frac"], prev["A_essentiality_axis"]["rogue_spared_frac"]),
           ("B dispensability", prev["B_dispensability"]["kill_frac"], prev["B_dispensability"]["rogue_spared_frac"]),
           ("C conservation", prev["C_conservation_only"]["kill_frac"], prev["C_conservation_only"]["rogue_spared_frac"])]
except Exception:
    abc = []
pts = abc[:]
if D_any: pts.append(("D1 absent>=1 sister", D_any["kill_frac"], D_any["rogue_spared_frac"]))
if D_all: pts.append(("D2 absent all sisters", D_all["kill_frac"], D_all["rogue_spared_frac"]))
cols = ["#C0392B","#27AE60","#E67E22","#2980B9","#16A085"]
for i,(lab,kf,rs) in enumerate(pts):
    ax.scatter(kf, rs, s=170, c=cols[i % len(cols)])
    ax.annotate(lab, (kf, rs), fontsize=8, xytext=(5,4), textcoords="offset points")
ax.set_xlabel("coverage: fraction of genes eliminated")
ax.set_ylabel("rogue essentials spared (fraction)")
ax.set_title("C. All eliminators: the kill-vs-spare-rogue tradeoff\n(top-right = ideal)")
ax.set_xlim(-0.02, 0.7); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

fig.suptitle("Pangenome presence/absence as positive dispensability evidence (eliminator D)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(OUT / "pangenome_dispensability.png", dpi=140, bbox_inches="tight")

(OUT / "pangenome_dispensability_summary.json").write_text(json.dumps({
    "organism": ORG, "sisters": SISTERS, "n_genes": int(len(df)),
    "base_ess": round(float(base), 3), "rogue_total": rogue_total,
    "ess_rate_by_n_sisters": {str(k): round(float(df[df.n_sisters_present==k].essential.mean()), 3)
                              for k in ks if (df.n_sisters_present==k).any()},
    "rogue_frac_by_n_sisters": {str(k): round(float(df[df.n_sisters_present==k].rogue.mean()), 3)
                                for k in ks if (df.n_sisters_present==k).any()},
    "D1_absent_in_any_sister": D_any,
    "D2_absent_in_all_sisters": D_all,
    "absent_in_sister_ess_rate": round(float(abs_sub.essential.mean()), 3),
}, indent=2))
print("\nwrote pangenome_dispensability.png + csv + summary")
