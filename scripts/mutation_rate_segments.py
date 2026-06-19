"""Mutation rate across chromosomal segments vs sister Ralstonia (predecessors).

Uses the per-gene dN/dS table (computed against sister orthologs, mostly PSI07):
  dS    synonymous substitution rate  -> proxy for the NEUTRAL local mutation rate
        (accumulates regardless of selection: 'mutation rate vs predecessors')
  dN    nonsynonymous rate
  dN/dS selection (low = purifying)

Five questions:
  Q1 SEGMENT MAP   does the neutral mutation rate (dS) vary ALONG the chromosome?
  Q2 REPLICATION-TIMING GRADIENT  does dS rise from oriC to terminus
                   (later-replicating DNA mutates more)?
  Q3 STRAND ASYMMETRY  leading vs lagging dS (replication-strand mutation bias)
  Q4 SELECTION vs MUTATION  essential genes: is it LOWER dN/dS (more selection)
                   while dS (raw mutation) is similar? -- i.e. essentials are
                   protected by SELECTION, not by a lower local mutation rate
  Q5 BY ATLAS TIER  dN/dS and dS across core-essential / non-essential /
                   rogue / conditional tiers

Outputs:
  outputs/orphan/mutation_rate_segments.png
  outputs/orphan/mutation_rate_segments_summary.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/user/cell/outputs/orphan")

# dN/dS table + positions/essential/tier
dnds = pd.read_parquet(OUT / "dnds_beril_RalstoniaGMI1000.parquet")
atlas = pd.read_csv(OUT / "atlas_multi" / "beril_RalstoniaGMI1000.csv")
dot = pd.read_csv(OUT / "gene_dot_atlas_GMI1000.csv")[["locus_tag","mid","leading","oriC_frac"]]

df = dnds.merge(atlas[["locus_tag","tier","measured_essential","conservation","n_paralogs"]],
                on="locus_tag", how="inner").merge(dot, on="locus_tag", how="inner")
df = df[df.ds.notna() & (df.ds > 0) & (df.dnds.notna())].copy()
# clip extreme dS (saturation / alignment artifacts)
df = df[df.ds < 2.0]
print(f"{len(df)} genes with usable dS/dN against sister orthologs")
print(f"  essential {int(df.measured_essential.sum())}, "
      f"non-essential {int((df.measured_essential==0).sum())}")

# ---- Q4 selection vs mutation: essential vs non-essential ----
e = df[df.measured_essential==1]; n = df[df.measured_essential==0]
print("\nQ4 SELECTION vs MUTATION (essential vs non-essential):")
print(f"  dS  (neutral mutation):  ess {e.ds.median():.3f}  non-ess {n.ds.median():.3f}")
print(f"  dN  (protein change):    ess {e.dn.median():.3f}  non-ess {n.dn.median():.3f}")
print(f"  dN/dS (selection):       ess {e.dnds.median():.3f}  non-ess {n.dnds.median():.3f}")

# ---- Q2 replication-timing gradient: dS vs oriC distance ----
df["ori_bin"] = pd.cut(df.oriC_frac, bins=np.linspace(0,1,11), labels=False,
                       include_lowest=True)
grad = df.groupby("ori_bin").agg(n=("ds","size"), ds_med=("ds","median"),
                                 dnds_med=("dnds","median")).reset_index()
print("\nQ2 REPLICATION-TIMING GRADIENT (dS by oriC distance):")
for _,r in grad.iterrows():
    print(f"  ori_frac {r['ori_bin']*0.1:.1f}-{(r['ori_bin']+1)*0.1:.1f}: "
          f"n={int(r['n']):<4} dS_med={r['ds_med']:.3f}  dN/dS_med={r['dnds_med']:.3f}")
# linear trend
from numpy.polynomial import polynomial as P
xb = df.oriC_frac.to_numpy(); yb = df.ds.to_numpy()
slope = np.polyfit(xb, yb, 1)[0]
r_ds_ori = np.corrcoef(xb, yb)[0,1]
print(f"  dS vs oriC_frac: slope {slope:+.3f}, Pearson r {r_ds_ori:+.3f}")

# ---- Q3 strand asymmetry ----
lead = df[df.leading==1]; lag = df[df.leading==0]
print(f"\nQ3 STRAND ASYMMETRY: leading dS {lead.ds.median():.3f}  "
      f"lagging dS {lag.ds.median():.3f}")

# ---- Q5 by atlas tier ----
print("\nQ5 BY ATLAS TIER (median dS / dN/dS):")
tier_stats = {}
for t in ["CONFIDENT_ESSENTIAL","CONFIDENT_NONESSENTIAL","ROGUE_SUSPECT",
          "CONDITIONAL_SUSPECT","UNRESOLVED"]:
    s = df[df.tier==t]
    if len(s)<5: continue
    tier_stats[t]=dict(n=int(len(s)), ds=round(float(s.ds.median()),3),
                       dnds=round(float(s.dnds.median()),3))
    print(f"  {t:<24} n={len(s):<5} dS={s.ds.median():.3f}  dN/dS={s.dnds.median():.3f}")

# ---- Q1 segment map (windowed dS along chromosome) ----
df = df.sort_values("mid")
L = 3_716_413
WIN=100_000; STEP=20_000
centers=np.arange(WIN//2, L-WIN//2, STEP)
ds_track=[]; dnds_track=[]
for c in centers:
    m=(df.mid>=c-WIN//2)&(df.mid<c+WIN//2)
    ds_track.append(df.ds[m].median() if m.sum()>=5 else np.nan)
    dnds_track.append(df.dnds[m].median() if m.sum()>=5 else np.nan)
ds_track=np.array(ds_track); dnds_track=np.array(dnds_track)

# ---- figure ----
fig, axes = plt.subplots(2, 3, figsize=(19, 11))

# A: dS segment map along chromosome
ax=axes[0,0]
xmb=centers/1e6
ax.plot(xmb, ds_track, color="#2980B9", lw=1.4, label="dS (neutral mutation)")
ax.fill_between(xmb, np.nanmin(ds_track), ds_track, color="#2980B9", alpha=0.15)
ax.set_xlabel("position (Mb)"); ax.set_ylabel("median dS (100 kb window)")
ax.set_title("Q1. Neutral mutation rate along the chromosome")
ax.grid(alpha=0.3)

# B: replication-timing gradient
ax=axes[0,1]
xs=(grad.ori_bin.astype(float)+0.5)*0.1
ax.plot(xs, grad.ds_med, "o-", color="#2980B9", lw=2, label="dS (mutation)")
z=np.polyfit(xs, grad.ds_med,1); ax.plot(xs, np.polyval(z,xs), "--", color="#1B4F72", lw=1, alpha=0.7)
ax.set_xlabel("distance from oriC (0=origin, 1=terminus)")
ax.set_ylabel("median dS"); ax.set_title(f"Q2. Replication-timing gradient\nr={r_ds_ori:+.2f}")
ax.grid(alpha=0.3); ax.legend(fontsize=8)

# C: selection vs mutation, ess vs non-ess
ax=axes[0,2]
cats=["dS\n(mutation)","dN\n(protein)","dN/dS\n(selection)"]
emed=[e.ds.median(), e.dn.median(), e.dnds.median()]
nmed=[n.ds.median(), n.dn.median(), n.dnds.median()]
x=np.arange(3); w=0.36
ax.bar(x-w/2, emed, w, color="#C0392B", label="essential")
ax.bar(x+w/2, nmed, w, color="#7F8C8D", label="non-essential")
for i,(a,b) in enumerate(zip(emed,nmed)):
    ax.text(i-w/2,a,f"{a:.2f}",ha="center",va="bottom",fontsize=8)
    ax.text(i+w/2,b,f"{b:.2f}",ha="center",va="bottom",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
ax.set_title("Q4. Selection vs mutation\n(essentials: low dN/dS, SAME dS)")
ax.legend(fontsize=8)

# D: dN/dS by tier
ax=axes[1,0]
ts=list(tier_stats)
dnds_t=[tier_stats[t]["dnds"] for t in ts]
ds_t=[tier_stats[t]["ds"] for t in ts]
cols={"CONFIDENT_ESSENTIAL":"#C0392B","CONFIDENT_NONESSENTIAL":"#7F8C8D",
      "ROGUE_SUSPECT":"#F39C12","CONDITIONAL_SUSPECT":"#E67E22","UNRESOLVED":"#BDC3C7"}
ax.bar(range(len(ts)), dnds_t, color=[cols[t] for t in ts], alpha=0.88)
for i,v in enumerate(dnds_t): ax.text(i,v,f"{v:.2f}",ha="center",va="bottom",fontsize=8)
ax.set_xticks(range(len(ts))); ax.set_xticklabels([t.replace("_","\n") for t in ts], fontsize=7)
ax.set_ylabel("median dN/dS"); ax.set_title("Q5. Selection (dN/dS) by atlas tier")

# E: dS by tier (mutation rate -- should be flatter than selection)
ax=axes[1,1]
ax.bar(range(len(ts)), ds_t, color=[cols[t] for t in ts], alpha=0.88)
for i,v in enumerate(ds_t): ax.text(i,v,f"{v:.2f}",ha="center",va="bottom",fontsize=8)
ax.set_xticks(range(len(ts))); ax.set_xticklabels([t.replace("_","\n") for t in ts], fontsize=7)
ax.set_ylabel("median dS"); ax.set_title("Q5. Mutation rate (dS) by atlas tier\n(flat => same mutation, different selection)")

# F: strand asymmetry + scatter
ax=axes[1,2]
ax.scatter(lead.oriC_frac, lead.ds, s=5, c="#27AE60", alpha=0.3, label=f"leading (med {lead.ds.median():.2f})")
ax.scatter(lag.oriC_frac, lag.ds, s=5, c="#C0392B", alpha=0.3, label=f"lagging (med {lag.ds.median():.2f})")
ax.set_xlabel("distance from oriC"); ax.set_ylabel("dS")
ax.set_ylim(0, np.percentile(df.ds,98))
ax.set_title("Q3. Strand asymmetry of mutation rate")
ax.legend(fontsize=8)

fig.suptitle("Mutation rate across chromosomal segments vs sister Ralstonia (dS=mutation, dN/dS=selection)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT / "mutation_rate_segments.png", dpi=140, bbox_inches="tight")

(OUT / "mutation_rate_segments_summary.json").write_text(json.dumps({
    "n_genes": int(len(df)),
    "Q4_essential_vs_nonessential": {
        "dS_ess": round(float(e.ds.median()),3), "dS_noness": round(float(n.ds.median()),3),
        "dN_ess": round(float(e.dn.median()),3), "dN_noness": round(float(n.dn.median()),3),
        "dnds_ess": round(float(e.dnds.median()),3), "dnds_noness": round(float(n.dnds.median()),3)},
    "Q2_dS_vs_oriC": {"slope": round(float(slope),4), "pearson_r": round(float(r_ds_ori),3),
        "by_bin": [dict(ori_frac=round(float(b)*0.1+0.05,2), n=int(nn), ds=round(float(d),3))
                   for b,nn,d in zip(grad.ori_bin, grad.n, grad.ds_med)]},
    "Q3_strand": {"leading_dS": round(float(lead.ds.median()),3),
                  "lagging_dS": round(float(lag.ds.median()),3)},
    "Q5_by_tier": tier_stats,
}, indent=2))
print("\nwrote mutation_rate_segments.png + summary")
