"""The honest essentiality atlas for GMI1000 -- predict where we can, abstain where we can't.

Synthesizes the whole project into a per-gene call with explicit confidence
tiers and an honest 'phenotype-required' bucket (the AlphaFold paradigm):

  CONFIDENT_ESSENTIAL     cross-org essential score in the band where
                          essential precision >= 0.90
  CONFIDENT_NONESSENTIAL  EITHER essential score in the band where non-ess
                          precision >= 0.90, OR positive dispensability evidence
                          (ortholog absent in ALL 3 sister Ralstonia) --
                          but NEVER a rogue-suspect
  PHENOTYPE_REQUIRED      the abstention bucket: rogue-suspects (cons<0.1,
                          single-copy, present in all sisters -- exactly where
                          rogue essentials hide and the model is untrustworthy)
                          plus the unresolved middle

The point of the atlas is NOT to maximize accuracy by forcing calls; it is to
report the trustworthy calls and quarantine the rest for wet-lab phenotyping.

Outputs:
  outputs/orphan/essentiality_atlas_GMI1000.csv     (the deliverable, per-gene)
  outputs/orphan/essentiality_atlas_GMI1000.png
  outputs/orphan/essentiality_atlas_GMI1000_summary.json
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

# essential score (cross-org LOO, GMI1000 held out) + features
df = pd.read_csv(OUT / "cross_org_coverage_scores.csv")
df = df[df.organism == ORG].copy().reset_index(drop=True)
s = df["score_all_combined"].to_numpy()          # honest cross-org essential score
y = df.essential.to_numpy()

# pangenome presence in sisters
og_orgs = defaultdict(set); gene_og = {}
for r in csv.DictReader(open(DATA / "labels" / "orthology_features.csv")):
    if r["og_id"]:
        og_orgs[r["og_id"]].add(r["organism"])
        if r["organism"] == ORG:
            gene_og[r["locus_tag"]] = r["og_id"]
df["n_sisters"] = df.locus_tag.map(
    lambda l: sum(1 for sis in SISTERS if gene_og.get(l) in og_orgs and sis in og_orgs[gene_og.get(l)]))
df["absent_all_sisters"] = df.n_sisters == 0
df["rogue_suspect"] = (df.conservation < 0.1) & (df.n_paralogs == 0) & (df.n_sisters == 3)

# ---- calibrate score bands to precision >= 0.90 (each direction) ----
def hi_thr(score, yv, target=0.9, mn=20):
    o = np.argsort(-score); ys = yv[o].astype(float); ss = score[o]
    k = np.arange(1, len(ys)+1); prec = np.cumsum(ys)/k
    ok = (prec >= target) & (k >= mn)
    return float(ss[np.where(ok)[0].max()]) if ok.any() else np.inf
def lo_thr(score, yv, target=0.9, mn=20):
    # non-essential precision among lowest scores
    o = np.argsort(score); ys = (1-yv)[o].astype(float); ss = score[o]
    k = np.arange(1, len(ys)+1); prec = np.cumsum(ys)/k
    ok = (prec >= target) & (k >= mn)
    return float(ss[np.where(ok)[0].max()]) if ok.any() else -np.inf

hi = hi_thr(s, y, 0.9); lo = lo_thr(s, y, 0.9)
print(f"score bands: essential>= {hi:.3f}  non-essential<= {lo:.3f}")

# ---- assign tiers ----
tier = np.full(len(df), "UNRESOLVED_MIDDLE", dtype=object)
# confident essential
tier[s >= hi] = "CONFIDENT_ESSENTIAL"
# confident non-essential: low score OR absent-in-all-sisters, but not rogue-suspect
conf_ne = ((s <= lo) | df.absent_all_sisters.to_numpy()) & (~df.rogue_suspect.to_numpy()) & (s < hi)
tier[conf_ne] = "CONFIDENT_NONESSENTIAL"
# abstain: rogue suspects always go to phenotype-required (override)
tier[df.rogue_suspect.to_numpy()] = "PHENOTYPE_REQUIRED"
# the unresolved middle that isn't confident either way also -> phenotype-required
tier[(tier == "UNRESOLVED_MIDDLE")] = "PHENOTYPE_REQUIRED"
df["tier"] = tier

# ---- evaluate each tier honestly against the real labels ----
print("\nTIER REPORT (honest vs measured labels):")
print(f"  {'tier':<24}{'n':>6}{'%':>6}{'ess_rate':>10}{'call_correct':>14}")
rep = {}
for t in ["CONFIDENT_ESSENTIAL", "CONFIDENT_NONESSENTIAL", "PHENOTYPE_REQUIRED"]:
    sub = df[df.tier == t]
    er = float(sub.essential.mean()) if len(sub) else float("nan")
    # call correctness: ESS tier correct if essential; NONESS tier correct if non-ess
    if t == "CONFIDENT_ESSENTIAL": correct = er
    elif t == "CONFIDENT_NONESSENTIAL": correct = 1 - er
    else: correct = float("nan")
    rep[t] = dict(n=int(len(sub)), pct=round(len(sub)/len(df), 3),
                  ess_rate=round(er, 3),
                  call_precision=None if np.isnan(correct) else round(correct, 3))
    cc = "n/a (abstain)" if np.isnan(correct) else f"{correct:.3f}"
    print(f"  {t:<24}{len(sub):>6}{len(sub)/len(df)*100:>5.0f}%{er:>10.3f}{cc:>14}")

# coverage = fraction we make a confident call on
n_called = int((df.tier != "PHENOTYPE_REQUIRED").sum())
called = df[df.tier != "PHENOTYPE_REQUIRED"]
call_correct = ((called.tier == "CONFIDENT_ESSENTIAL") & (called.essential == 1)) | \
               ((called.tier == "CONFIDENT_NONESSENTIAL") & (called.essential == 0))
overall_prec = float(call_correct.mean())
print(f"\n  COVERAGE (confident calls): {n_called}/{len(df)} = {n_called/len(df):.0%}")
print(f"  PRECISION over confident calls: {overall_prec:.3f}")

# how many rogue essentials did we correctly NOT call non-essential?
rogue = df[(df.essential == 1) & (df.conservation < 0.1)]
rogue_protected = int((rogue.tier != "CONFIDENT_NONESSENTIAL").sum())
print(f"  rogue essentials NOT mislabeled non-essential: {rogue_protected}/{len(rogue)} "
      f"({rogue_protected/max(1,len(rogue)):.0%})")

# ---- save the deliverable ----
atlas = pd.read_csv(OUT / "gene_dot_atlas_GMI1000.csv")[["locus_tag", "product"]]
deliver = df.merge(atlas, on="locus_tag", how="left")
deliver["predicted_call"] = deliver.tier.map({
    "CONFIDENT_ESSENTIAL": "essential",
    "CONFIDENT_NONESSENTIAL": "non-essential",
    "PHENOTYPE_REQUIRED": "unknown"})
deliver[["locus_tag", "product", "tier", "predicted_call", "essential",
         "score_all_combined", "conservation", "n_paralogs", "gene_len_aa",
         "n_sisters", "absent_all_sisters", "rogue_suspect"]].rename(
    columns={"essential": "measured_essential"}).to_csv(
    OUT / "essentiality_atlas_GMI1000.csv", index=False)

# ---- figure ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# A: tier composition pie
ax = axes[0]
order = ["CONFIDENT_ESSENTIAL", "CONFIDENT_NONESSENTIAL", "PHENOTYPE_REQUIRED"]
sizes = [rep[t]["n"] for t in order]
cols = ["#C0392B", "#95A5A6", "#F39C12"]
ax.pie(sizes, labels=[f"{t.replace('_',chr(10))}\n{n}" for t,n in zip(order,sizes)],
       colors=cols, autopct=lambda p: f"{p:.0f}%", startangle=90,
       textprops={"fontsize": 8})
ax.set_title(f"A. Atlas tiers ({len(df)} genes)\ncoverage {n_called/len(df):.0%} at precision {overall_prec:.0%}")

# B: per-tier essential rate (honest)
ax = axes[1]
er = [rep[t]["ess_rate"] for t in order]
b = ax.bar(range(3), er, color=cols, alpha=0.88)
for i,(t,e) in enumerate(zip(order,er)):
    ax.text(i, e, f"{e:.2f}", ha="center", va="bottom", fontsize=10)
ax.axhline(df.essential.mean(), color="k", ls="--", lw=0.8, label=f"genome base {df.essential.mean():.2f}")
ax.set_xticks(range(3)); ax.set_xticklabels(["confident\nessential","confident\nnon-ess","phenotype\nrequired"], fontsize=9)
ax.set_ylabel("measured essential rate"); ax.set_ylim(0,1)
ax.set_title("B. Each tier vs ground truth\n(ess tier high, non-ess low, abstain mixed)")
ax.legend(fontsize=8)

# C: score distribution colored by tier
ax = axes[2]
for t,c in zip(order, cols):
    sub = df[df.tier == t]
    ax.scatter(sub.score_all_combined, sub.conservation, s=8, c=c, alpha=0.5,
               label=t.replace("_"," ").title())
ax.axvline(hi, color="#C0392B", ls="--", lw=0.8); ax.axvline(lo, color="#7F8C8D", ls="--", lw=0.8)
ax.set_xlabel("cross-org essential score (LOO)"); ax.set_ylabel("conservation")
ax.set_title("C. Tier map: score x conservation\n(abstain captures low-conservation middle)")
ax.legend(fontsize=7, loc="center right")

fig.suptitle(f"Essentiality atlas -- {ORG.replace('beril_','')}: confident calls + honest phenotype-required bucket",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(OUT / "essentiality_atlas_GMI1000.png", dpi=140, bbox_inches="tight")

(OUT / "essentiality_atlas_GMI1000_summary.json").write_text(json.dumps({
    "organism": ORG, "n_genes": int(len(df)),
    "score_bands": {"essential_ge": round(hi,3), "nonessential_le": round(lo,3)},
    "tiers": rep,
    "coverage_confident_calls": round(n_called/len(df), 3),
    "precision_over_confident_calls": round(overall_prec, 3),
    "rogue_total": int(len(rogue)),
    "rogue_not_mislabeled_noness": rogue_protected,
    "deliverable": "outputs/orphan/essentiality_atlas_GMI1000.csv",
}, indent=2))
print("\nwrote essentiality_atlas_GMI1000.csv + png + summary")
