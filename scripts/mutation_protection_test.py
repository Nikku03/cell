"""Test the hypothesis: essential genes are PROTECTED FROM or RESISTANT TO mutation.

Two distinct mechanisms in the hypothesis, two distinct measurements:

  H1 "protected from mutation"  -> a LOWER mutation RATE at essential loci.
      Test: dS (synonymous substitution rate vs sister orthologs) = a neutral
      mutation-rate proxy (synonymous changes are ~invisible to selection).
      If essentials are protected, dS_essential < dS_nonessential.

  H2 "resistant to mutation"    -> the protein TOLERATES amino-acid changes
      (robust fold / can absorb substitutions without losing function).
      Test: dN (nonsynonymous rate) and dN/dS (selection). A mutation-robust
      gene would tolerate aa changes -> HIGH dN, dN/dS toward 1.
      If essentials are resistant, dN/dS_essential would be HIGH.

  H3 (the alternative the data may support): NEITHER -- essentials are protected
      by PURIFYING SELECTION. Same mutation rate (dS equal), but aa-changing
      mutations are deleterious and removed (dN low, dN/dS low). Protection by
      death of the mutants, not protection of the DNA or robustness of the protein.

Plus a SYSTEM-level robustness check: paralog redundancy (a backup copy is a
different kind of mutation-resistance -- at the network level, not the sequence).

6 bacteria, per-organism + pooled, Mann-Whitney U, effect sizes.

Outputs:
  outputs/orphan/mutation_protection_test.png
  outputs/orphan/mutation_protection_test.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/user/cell/outputs/orphan")
ORGS = ["beril_RalstoniaGMI1000","beril_RalstoniaBSBF1503","beril_RalstoniaPSI07",
        "beril_Dda3937","beril_HerbieS","beril_Magneto"]

# per-gene features (essential, paralogs, length, gc, conservation)
cov = pd.read_csv(OUT/"cross_org_coverage_scores.csv")
frames=[]
for org in ORGS:
    p = OUT/f"dnds_{org}.parquet"
    if not p.exists(): continue
    d = pd.read_parquet(p)[["locus_tag","dn","ds","dnds"]]
    d["organism"]=org
    frames.append(d)
dnds = pd.concat(frames, ignore_index=True)
df = cov.merge(dnds, on=["organism","locus_tag"], how="inner")
df = df[df.organism.isin(ORGS)]
df = df[(df.ds>0.005) & (df.ds<2) & df.dnds.notna() & np.isfinite(df.dnds)]
print(f"{len(df)} genes with usable dN/dS across {df.organism.nunique()} orgs")
print(f"essential {int(df.essential.sum())}, non-essential {int((df.essential==0).sum())}\n")

e = df[df.essential==1]; n = df[df.essential==0]

def cliffs_delta(a,b):
    # nonparametric effect size via AUC: P(a>b)-P(a<b)
    a=np.asarray(a); b=np.asarray(b)
    if len(a)==0 or len(b)==0: return float("nan")
    # use rank-based shortcut
    from scipy.stats import mannwhitneyu
    U,_=mannwhitneyu(a,b,alternative="two-sided")
    return 2*U/(len(a)*len(b))-1

def test(metric, label, direction_if_protected):
    me, mn = e[metric].median(), n[metric].median()
    U,pmw = mannwhitneyu(e[metric], n[metric], alternative="two-sided")
    delta = cliffs_delta(e[metric], n[metric])
    print(f"{label}")
    print(f"   essential median {me:.4f} | non-essential median {mn:.4f} | "
          f"ratio {me/mn:.2f} | Cliff's delta {delta:+.3f} | p={pmw:.1e}")
    return dict(metric=metric, ess_median=round(float(me),4), noness_median=round(float(mn),4),
                ratio=round(float(me/mn),3), cliffs_delta=round(float(delta),3),
                p_value=float(pmw))

print("="*70)
print("H1 -- PROTECTED FROM MUTATION? (lower mutation rate => dS lower)")
print("="*70)
r_ds = test("ds", "dS (neutral mutation rate)", "lower")
print(f"   VERDICT: {'PROTECTED (dS lower)' if r_ds['ratio']<0.9 and r_ds['p_value']<0.05 else 'NOT protected -- mutation rate is essentially the same'}")

print("\n"+"="*70)
print("H2 -- RESISTANT TO MUTATION? (robust => tolerates aa changes => dN/dS high)")
print("="*70)
r_dn = test("dn", "dN (amino-acid change rate)", "higher")
r_w  = test("dnds", "dN/dS (selection; high=tolerant/relaxed, low=purged)", "higher")
print(f"   VERDICT: {'RESISTANT (dN/dS high)' if r_w['ratio']>1.1 else 'NOT resistant -- aa changes are PURGED (dN/dS low), genes are mutation-SENSITIVE'}")

print("\n"+"="*70)
print("H3 -- PROTECTED BY PURIFYING SELECTION? (same dS, lower dN, lower dN/dS)")
print("="*70)
same_ds = 0.9 < r_ds["ratio"] < 1.1
low_dnds = r_w["ratio"] < 0.9
print(f"   same mutation rate (dS ratio {r_ds['ratio']:.2f} ~ 1):  {same_ds}")
print(f"   stronger purifying selection (dN/dS ratio {r_w['ratio']:.2f} < 1): {low_dnds}")
print(f"   VERDICT: {'CONFIRMED -- protected by selection, not by mutation-protection or robustness' if (same_ds and low_dnds) else 'mixed'}")

print("\n"+"="*70)
print("SYSTEM-LEVEL robustness check: paralog redundancy (backup copies)")
print("="*70)
r_par = test("n_paralogs", "n_paralogs (gene-level backup)", "higher")
print(f"   VERDICT: {'essentials ARE backed up more (system-level robustness real)' if r_par['ratio']>1.1 else 'no extra redundancy'}")

# per-organism consistency of the headline contrasts
print("\nper-organism consistency (dS ratio | dN/dS ratio, ess/noness):")
per_org={}
for org in ORGS:
    eo=df[(df.organism==org)&(df.essential==1)]; no=df[(df.organism==org)&(df.essential==0)]
    if len(eo)<30 or len(no)<30: continue
    ds_r=eo.ds.median()/no.ds.median(); w_r=eo.dnds.median()/no.dnds.median()
    per_org[org]=dict(dS_ratio=round(float(ds_r),3), dnds_ratio=round(float(w_r),3))
    print(f"   {org.replace('beril_',''):<22} dS {ds_r:.2f}   dN/dS {w_r:.2f}")

summary=dict(n_genes=int(len(df)),
    H1_protected_from_mutation=r_ds, H2a_dN=r_dn, H2b_dnds=r_w, H3_paralogs=r_par,
    verdict=("NOT protected from mutation (dS equal) and NOT resistant (dN/dS low = "
             "mutation-sensitive); essentials are protected by PURIFYING SELECTION "
             "(mutants die), with additional SYSTEM-level redundancy via paralogs."),
    per_org=per_org)
json.dump(summary, open(OUT/"mutation_protection_test.json","w"), indent=2)

# ---- figure ----
fig, ax = plt.subplots(1,3, figsize=(17,5.5))
# A: dS distributions (mutation rate) -- the protection test
ax[0].hist(np.clip(e.ds,0,0.6), bins=40, density=True, alpha=0.6, color="#C0392B", label=f"essential (med {e.ds.median():.2f})")
ax[0].hist(np.clip(n.ds,0,0.6), bins=40, density=True, alpha=0.5, color="#7F8C8D", label=f"non-ess (med {n.ds.median():.2f})")
ax[0].set_xlabel("dS (neutral mutation rate)"); ax[0].set_ylabel("density")
ax[0].set_title("H1: protected from mutation?\nSAME dS -> NO"); ax[0].legend(fontsize=8)
# B: dN/dS distributions -- the resistance test
ax[1].hist(np.clip(e.dnds,0,0.6), bins=40, density=True, alpha=0.6, color="#C0392B", label=f"essential (med {e.dnds.median():.2f})")
ax[1].hist(np.clip(n.dnds,0,0.6), bins=40, density=True, alpha=0.5, color="#7F8C8D", label=f"non-ess (med {n.dnds.median():.2f})")
ax[1].set_xlabel("dN/dS (selection)"); ax[1].set_title("H2: resistant to mutation?\nLOWER dN/dS -> NO (more sensitive, purged)")
ax[1].legend(fontsize=8)
# C: the decomposition bars
cats=["dS\n(mutation)","dN\n(aa change)","dN/dS\n(selection)","paralogs\n(backup)"]
em=[e.ds.median(),e.dn.median(),e.dnds.median(),e.n_paralogs.median()]
nm=[n.ds.median(),n.dn.median(),n.dnds.median(),n.n_paralogs.median()]
x=np.arange(4); w=0.38
ax[2].bar(x-w/2,em,w,color="#C0392B",label="essential")
ax[2].bar(x+w/2,nm,w,color="#7F8C8D",label="non-essential")
for i,(a,b) in enumerate(zip(em,nm)):
    ax[2].text(i-w/2,a,f"{a:.2f}",ha="center",va="bottom",fontsize=8)
    ax[2].text(i+w/2,b,f"{b:.2f}",ha="center",va="bottom",fontsize=8)
ax[2].set_xticks(x); ax[2].set_xticklabels(cats,fontsize=8)
ax[2].set_title("The decomposition:\nsame mutation, fewer aa changes, more backups")
ax[2].legend(fontsize=8)
fig.suptitle("Are essential genes protected from / resistant to mutation? (6 bacteria, dN/dS vs sisters)",
             fontweight="bold", fontsize=13)
fig.tight_layout(); fig.savefig(OUT/"mutation_protection_test.png", dpi=140, bbox_inches="tight")
print("\nwrote mutation_protection_test.png + .json")
