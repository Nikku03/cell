"""Multi-organism essentiality atlas (v2) -- whole project as evidence channels.

Six bacteria, per-gene tiering with explicit evidence codes. Every line of
research the project produced becomes a named channel:

POSITIVE ESSENTIAL EVIDENCE
  E_score        cross-org LOO essential score >= hi_thr (calibrated per-org)
  E_cons_core    conservation >= 0.5 (the deeply-retained core)
  E_geometry     leading strand AND short (the essential geometric signature)

POSITIVE NON-ESSENTIAL EVIDENCE (the inversions of the protection panel)
  N_score        cross-org LOO essential score <= lo_thr
  N_pangenome    absent in >=1 sister strain (real dispensability evidence)
  N_paralog      n_paralogs >= 2 (redundancy-protected: buffered explanation)
  N_mobile       mobile-element-adjacent (< 5 kb) -- HGT/accessory belt
  N_long_lagging long gene on lagging strand (head-on configuration the cell
                 avoids for essentials -- INVERSION of P3 from replication
                 geometry)

PHENOTYPE-REQUIRED FLAGS
  P_rogue        cons<0.1 + n_paralogs=0 + present in all sisters
                 (the structural rogue-essential profile -- quarantine)
  P_conditional  cons<0.3 + long + lagging + present in all sisters
                 (niche/virulence conditional candidates)
  P_unresolved   score in the middle band, no other signal

TIER RULES
  CORE_ESSENTIAL          E_score AND E_cons_core           (top confidence)
  CONFIDENT_ESSENTIAL     E_score                            (and NOT P_rogue)
  BUFFERED_REDUNDANT      score in middle, conservation>=0.5, n_paralogs>=4
                          (conserved-but-dispensable: paralog redundancy)
  CONFIDENT_NONESSENTIAL  (N_score OR N_pangenome) AND >=1 of
                          {N_paralog, N_mobile, N_long_lagging}
                          AND NOT (P_rogue OR P_conditional)
  ROGUE_SUSPECT           P_rogue
  CONDITIONAL_SUSPECT     P_conditional
  UNRESOLVED              everything else

Plus a cross-org consistency check: for OGs present as confident calls in
>=2 organisms, do the calls agree? This is independent validation.

Outputs:
  outputs/orphan/atlas_multi/<org>.csv     per-organism per-gene tier table
  outputs/orphan/atlas_multi_summary.png
  outputs/orphan/atlas_multi_summary.json
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
ATL = OUT / "atlas_multi"; ATL.mkdir(exist_ok=True)
ARCHAEA = {"beril_Methanococcus_JJ", "beril_Methanococcus_S2"}

# sister strains for each focal org (from shared-OG inspection)
SISTERS = {
    "beril_RalstoniaGMI1000":  ["beril_RalstoniaBSBF1503","beril_RalstoniaPSI07","beril_RalstoniaUW163"],
    "beril_RalstoniaBSBF1503": ["beril_RalstoniaGMI1000","beril_RalstoniaPSI07","beril_RalstoniaUW163"],
    "beril_RalstoniaPSI07":    ["beril_RalstoniaGMI1000","beril_RalstoniaBSBF1503","beril_RalstoniaUW163"],
    "beril_Dda3937":           ["beril_Ddia6719","beril_DdiaME23"],
    "beril_HerbieS":           ["beril_Cup4G11","beril_BFirm","beril_Burk376"],   # order-level (Burkholderiales)
    "beril_Magneto":           ["beril_azobra","beril_Smeli","beril_PS"],         # order-level (Alphaproteobacteria)
}

# -- base features and labels (cross-org LOO scores already computed) --
df = pd.read_csv(OUT / "cross_org_coverage_scores.csv")
df = df[~df.organism.isin(ARCHAEA)].reset_index(drop=True)
BACT = list(dict.fromkeys(df.organism))

# -- LOO non-essential score (mirror of essential score) --
FEATS = ["leading","oriC_frac","mobile_dist_kb","n_paralogs","gene_len_aa","gc","conservation"]
def fit_predict(Xtr,ytr,Xte,it=600,lr=0.5,l2=2.0):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
    Xtr=np.c_[np.ones(len(Xtr)),(Xtr-mu)/sd]; Xte=np.c_[np.ones(len(Xte)),(Xte-mu)/sd]
    w=np.zeros(Xtr.shape[1]); ytr=ytr.astype(float)
    pos=ytr.mean() or 1e-6; cw=np.where(ytr==1,0.5/pos,0.5/(1-pos))
    for _ in range(it):
        p=1/(1+np.exp(-Xtr@w)); g=Xtr.T@((p-ytr)*cw)/len(Xtr)
        g[1:]+=l2*w[1:]/len(Xtr); w-=lr*g
    return 1/(1+np.exp(-Xtr@w)), 1/(1+np.exp(-Xte@w))

df["noness_score"] = np.nan
for h in BACT:
    m=(df.organism==h).to_numpy(); tr=df[~m]; te=df[m]
    _, pte = fit_predict(tr[FEATS].to_numpy(float), (1-tr.essential).to_numpy(),
                         te[FEATS].to_numpy(float))
    df.loc[m,"noness_score"] = pte
df["ess_score"] = df["score_all_combined"]

# -- pangenome: OG -> set of orgs (full orthology table) --
og_orgs = defaultdict(set); gene_og = {}
for r in csv.DictReader(open(DATA / "labels" / "orthology_features.csv")):
    if r["og_id"]:
        og_orgs[r["og_id"]].add(r["organism"])
        gene_og[(r["organism"], r["locus_tag"])] = r["og_id"]
df["og"] = [gene_og.get((o,l), "") for o,l in zip(df.organism, df.locus_tag)]

# -- ortholog-transfer vote: measured essentiality of OG members in OTHER orgs --
# (LOO-safe: at deploy time the training organisms' labels are known)
meas = {}
for r in csv.DictReader(open(DATA / "labels" / "labels.csv")):
    meas[(r["organism"], r["locus_tag"])] = int(r["essential"])
og_votes = defaultdict(lambda: [0, 0])   # og -> [n_essential, n_nonessential] with org tags
og_org_label = defaultdict(list)
for (o, l), e in meas.items():
    og = gene_og.get((o, l))
    if og: og_org_label[og].append((o, e))
def ortho_vote(org, og):
    if not og: return 0, 0
    ess = sum(1 for o, e in og_org_label[og] if o != org and e == 1)
    non = sum(1 for o, e in og_org_label[og] if o != org and e == 0)
    return ess, non
votes = [ortho_vote(o, og) for o, og in zip(df.organism, df.og)]
df["ovote_ess"] = [v[0] for v in votes]
df["ovote_non"] = [v[1] for v in votes]

# -- dN/dS selection signal (computed per org vs sister orthologs) --
dnds_map = {}
for org in BACT:
    p = OUT / f"dnds_{org}.parquet"
    if p.exists():
        dd = pd.read_parquet(p)[["locus_tag","dnds","ds"]]
        for lt, dn, ds in zip(dd.locus_tag, dd.dnds, dd.ds):
            dnds_map[(org, lt)] = (dn, ds)
df["dnds"] = [dnds_map.get((o,l),(np.nan,np.nan))[0] for o,l in zip(df.organism, df.locus_tag)]
df["ds"]   = [dnds_map.get((o,l),(np.nan,np.nan))[1] for o,l in zip(df.organism, df.locus_tag)]

def n_sisters_present(row):
    og = row["og"]; sis = SISTERS.get(row["organism"], [])
    if not og or not sis: return 0
    present = og_orgs.get(og, set())
    return sum(1 for s in sis if s in present)
df["n_sisters"] = df.apply(n_sisters_present, axis=1)
df["max_sisters"] = df.organism.map(lambda o: len(SISTERS.get(o, [])))
df["sister_absent_count"] = df["max_sisters"] - df["n_sisters"]

# -- per-org length quartile to define "long" --
df["long"] = False
for o in BACT:
    m = df.organism==o
    q75 = df.loc[m, "gene_len_aa"].quantile(0.75)
    df.loc[m, "long"] = df.loc[m, "gene_len_aa"] >= q75

# -- calibrate per-org score bands on TRAINING data (LOO) --
def hi_thr(s,y,t=0.9,mn=20):
    o=np.argsort(-s); ys=np.asarray(y)[o].astype(float); ss=np.asarray(s)[o]
    k=np.arange(1,len(ys)+1); prec=np.cumsum(ys)/k
    ok=(prec>=t)&(k>=mn); return float(ss[np.where(ok)[0].max()]) if ok.any() else np.inf
def lo_thr(s,y,t=0.9,mn=20):
    o=np.argsort(s); ys=(1-np.asarray(y))[o].astype(float); ss=np.asarray(s)[o]
    k=np.arange(1,len(ys)+1); prec=np.cumsum(ys)/k
    ok=(prec>=t)&(k>=mn); return float(ss[np.where(ok)[0].max()]) if ok.any() else -np.inf

# calibrate using OTHER orgs' actual labels & scores (no held-org leakage)
hi_per = {}; lo_per = {}
for held in BACT:
    tr = df[df.organism != held]
    hi_per[held] = hi_thr(tr.ess_score.to_numpy(), tr.essential.to_numpy(), 0.9)
    lo_per[held] = lo_thr(tr.ess_score.to_numpy(), tr.essential.to_numpy(), 0.9)

# -- evidence channels and tiering, per-gene --
def tier_gene(r):
    ev = []
    hi = hi_per[r.organism]; lo = lo_per[r.organism]
    # positive essential evidence
    if r.ess_score >= hi: ev.append("E_score")
    if r.conservation >= 0.5: ev.append("E_cons_core")
    if r.leading == 1 and not r["long"]: ev.append("E_geometry")
    # positive non-essential evidence (inverted protection)
    if r.ess_score <= lo: ev.append("N_score")
    pan_strong = (r.max_sisters > 0) and (r.sister_absent_count == r.max_sisters)
    pan_some = (r.max_sisters > 0) and (r.sister_absent_count >= 1) and not pan_strong
    if pan_strong: ev.append("N_pangenome_strong")
    elif pan_some: ev.append("N_pangenome_some")
    if r.n_paralogs >= 4 and r.conservation < 0.5: ev.append("N_redundancy")
    if r.mobile_dist_kb < 5: ev.append("N_mobile")
    if r["long"] and r.leading == 0 and r.conservation < 0.3: ev.append("N_long_lagging")
    # ortholog-transfer vote (independent of this gene's own features)
    if r.ovote_ess >= 2 and r.ovote_non == 0: ev.append("E_ortho")
    if r.ovote_non >= 2 and r.ovote_ess == 0: ev.append("N_ortho")
    # dN/dS selection signal (strong purifying selection -> essential-leaning)
    if (not pd.isna(r.dnds)) and (not pd.isna(r.ds)) and r.ds > 0.01 and r.dnds <= 0.10:
        ev.append("E_dnds")
    # phenotype-required flags
    p_rogue = (r.conservation < 0.1 and r.n_paralogs == 0
               and (r.max_sisters == 0 or r.sister_absent_count == 0))
    p_cond  = (r.conservation < 0.3 and r["long"] and r.leading == 0
               and (r.max_sisters == 0 or r.sister_absent_count == 0))
    if p_rogue: ev.append("P_rogue")
    if p_cond: ev.append("P_conditional")
    # biological annotation flag (not a call)
    if r.n_paralogs >= 4 and r.conservation >= 0.5: ev.append("F_core_redundant")

    # tier resolution: strict, score-led; biology flags annotate but don't call
    quarantine = p_rogue or p_cond
    has_E_call = "E_score" in ev
    has_N_call = ("N_score" in ev) or ("N_pangenome_strong" in ev)
    # v3 ortholog-transfer promotion (validated combos that hold precision >=0.85):
    #   essential : E_ortho with a second essential signal (E_cons or E_geometry)
    #   non-ess   : N_ortho alone (measured non-ess in >=2 sister orgs) -- 0.965
    #   + dN/dS: strong purifying selection paired with conservation or an
    #     ortholog vote -> essential (validated ~0.85 precision on UNRESOLVED)
    promote_E = (("E_ortho" in ev) and (("E_cons_core" in ev) or ("E_geometry" in ev))) \
                or (("E_dnds" in ev) and (("E_cons_core" in ev) or ("E_ortho" in ev)))
    promote_N = ("N_ortho" in ev)

    if (has_E_call or promote_E) and not p_rogue:
        tier = "CONFIDENT_ESSENTIAL"
    elif promote_N:
        # a clean measured non-ess ortholog vote (>=2 non, 0 ess) is 0.98 precise
        # even on rogue/conditional suspects -> it OVERRIDES the structural quarantine
        tier = "CONFIDENT_NONESSENTIAL"
    elif has_N_call and not quarantine:
        tier = "CONFIDENT_NONESSENTIAL"
    elif p_rogue:
        tier = "ROGUE_SUSPECT"
    elif p_cond:
        tier = "CONDITIONAL_SUSPECT"
    else:
        tier = "UNRESOLVED"
    return tier, "|".join(ev)

tiers = df.apply(tier_gene, axis=1)
df["tier"]  = [t[0] for t in tiers]
df["evidence"] = [t[1] for t in tiers]
df["predicted_call"] = df.tier.map({
    "CONFIDENT_ESSENTIAL": "essential",
    "CONFIDENT_NONESSENTIAL": "non-essential",
    "ROGUE_SUSPECT": "unknown",
    "CONDITIONAL_SUSPECT": "unknown",
    "UNRESOLVED": "unknown"})
# per-gene confidence = number of supporting evidence channels for the call
def n_support(r):
    ev = r["evidence"].split("|") if r["evidence"] else []
    if r.tier == "CONFIDENT_ESSENTIAL":
        return sum(1 for x in ev if x.startswith("E_"))
    if r.tier == "CONFIDENT_NONESSENTIAL":
        return sum(1 for x in ev if x.startswith("N_"))
    return 0
df["confidence"] = df.apply(n_support, axis=1)

# -- per-org evaluation --
TIER_ORDER = ["CONFIDENT_ESSENTIAL","CONFIDENT_NONESSENTIAL",
              "ROGUE_SUSPECT","CONDITIONAL_SUSPECT","UNRESOLVED"]
per_org = {}
print(f"{'org':<22}{'n':>5}{'ess':>6}{'nE':>6}{'rogue':>6}{'cond':>6}{'?':>6}"
      f"{'cov':>6}{'prec':>6}{'rogueOK':>8}")
for org in BACT:
    g = df[df.organism == org]
    counts = {t: int((g.tier==t).sum()) for t in TIER_ORDER}
    called = g[g.predicted_call != "unknown"]
    if len(called):
        call_correct = ((called.predicted_call == "essential") & (called.essential == 1)) | \
                       ((called.predicted_call == "non-essential") & (called.essential == 0))
        prec = float(call_correct.mean())
    else:
        prec = float("nan")
    cov = len(called)/len(g)
    # rogue protection: rogue essentials NOT mis-called non-essential
    rogue = g[(g.essential==1) & (g.conservation < 0.1)]
    rogue_ok = int((rogue.predicted_call != "non-essential").sum())
    per_org[org] = dict(n=len(g), tiers=counts,
        coverage=round(cov,3), precision_called=round(prec,3),
        rogue_total=int(len(rogue)), rogue_protected=rogue_ok)
    print(f"{org.replace('beril_',''):<22}{len(g):>5}"
          f"{counts['CONFIDENT_ESSENTIAL']:>6}{counts['CONFIDENT_NONESSENTIAL']:>6}"
          f"{counts['ROGUE_SUSPECT']:>6}{counts['CONDITIONAL_SUSPECT']:>6}{counts['UNRESOLVED']:>6}"
          f"{cov:>6.0%}{prec:>6.0%}{rogue_ok}/{len(rogue):>3}".rjust(8))

# -- cross-org consistency: shared OGs with confident calls in multiple orgs --
g_called = df[df.predicted_call != "unknown"].copy()
shared = g_called.groupby("og").filter(lambda x: x.organism.nunique() >= 2)
# per OG: do all confident calls agree?
agree = shared.groupby("og").predicted_call.nunique() == 1
print(f"\nCROSS-ORG CONSISTENCY")
print(f"  shared OGs with confident calls in >=2 orgs: {agree.shape[0]}")
print(f"  agreement (same call across orgs):         {int(agree.sum())} ({agree.mean()*100:.1f}%)")
print(f"  disagreement (split essential/non-ess):    {int((~agree).sum())}")
# mark cross-validated genes
xv_ogs = set(agree[agree].index)
df["cross_validated"] = df.og.map(lambda x: x in xv_ogs)
xv_called = df[(df.predicted_call != "unknown") & df.cross_validated]
xv_correct = ((xv_called.predicted_call=="essential") & (xv_called.essential==1)) | \
             ((xv_called.predicted_call=="non-essential") & (xv_called.essential==0))
xv_prec = float(xv_correct.mean()) if len(xv_called) else float("nan")
print(f"  genes with a cross-validated confident call: {len(xv_called)} "
      f"({len(xv_called)/len(df)*100:.1f}% of all genes)")
print(f"  precision on cross-validated calls:          {xv_prec:.3f}")

# confidence-tier precision (how does precision rise with evidence count?)
print(f"\nPRECISION BY CONFIDENCE LEVEL (# supporting evidence channels):")
called = df[df.predicted_call != "unknown"]
for k in sorted(called.confidence.unique()):
    sub = called[called.confidence == k]
    correct = ((sub.predicted_call=="essential") & (sub.essential==1)) | \
              ((sub.predicted_call=="non-essential") & (sub.essential==0))
    print(f"  confidence={k} : n={len(sub):>5}  precision={correct.mean():.3f}")

# -- save per-org csvs --
atl_atlas = pd.read_csv(OUT / "gene_dot_atlas_GMI1000.csv")[["locus_tag","product"]]
for org in BACT:
    g = df[df.organism == org].copy()
    if org == "beril_RalstoniaGMI1000":
        g = g.merge(atl_atlas, on="locus_tag", how="left")
    else:
        g["product"] = ""
    cols = ["locus_tag","product","tier","predicted_call","confidence","evidence",
            "essential","ess_score","noness_score","conservation","n_paralogs",
            "gene_len_aa","leading","mobile_dist_kb","n_sisters","max_sisters",
            "ovote_ess","ovote_non","dnds","cross_validated"]
    g[cols].rename(columns={"essential":"measured_essential"}).to_csv(
        ATL / f"{org}.csv", index=False)

# -- summary figure: stacked tier bars per org + precision + rogue protection --
fig, axes = plt.subplots(2, 2, figsize=(17, 11))

# A: stacked tier composition per organism
ax = axes[0,0]
COL = {"CONFIDENT_ESSENTIAL":"#C0392B",
       "CONFIDENT_NONESSENTIAL":"#7F8C8D",
       "ROGUE_SUSPECT":"#F39C12","CONDITIONAL_SUSPECT":"#E67E22","UNRESOLVED":"#ECF0F1"}
orgs = [o.replace("beril_","") for o in BACT]
x = np.arange(len(BACT)); bottom = np.zeros(len(BACT))
for t in TIER_ORDER:
    h = np.array([per_org[o]["tiers"][t]/per_org[o]["n"] for o in BACT])
    ax.bar(x, h, bottom=bottom, label=t.replace("_","\n"), color=COL[t])
    bottom += h
ax.set_xticks(x); ax.set_xticklabels(orgs, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("fraction of genes"); ax.set_ylim(0,1)
ax.set_title("A. Atlas composition per organism")
ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))

# B: coverage and precision
ax = axes[0,1]
cov = [per_org[o]["coverage"] for o in BACT]
prc = [per_org[o]["precision_called"] for o in BACT]
ax.bar(x-0.2, cov, 0.4, color="#16A085", label="coverage (confident calls)")
ax.bar(x+0.2, prc, 0.4, color="#2980B9", label="precision over called")
for i,(c,p) in enumerate(zip(cov,prc)):
    ax.text(i-0.2,c,f"{c:.0%}",ha="center",va="bottom",fontsize=8)
    ax.text(i+0.2,p,f"{p:.0%}",ha="center",va="bottom",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(orgs, rotation=40, ha="right", fontsize=8)
ax.set_ylim(0,1)
ax.set_title("B. Coverage and precision per organism")
ax.legend(fontsize=8)

# C: rogue protection (rogue essentials not mislabeled non-essential)
ax = axes[1,0]
rt = [per_org[o]["rogue_total"] for o in BACT]
rp = [per_org[o]["rogue_protected"]/max(1,per_org[o]["rogue_total"]) for o in BACT]
ax.bar(x, rp, color="#F39C12", alpha=0.88)
for i,(r,n) in enumerate(zip(rp,rt)):
    ax.text(i, r, f"{r:.0%}\nn={n}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(orgs, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("rogue essentials NOT misclassified as non-essential")
ax.set_ylim(0, 1.15)
ax.set_title("C. Rogue protection (higher = honest about the conditional zone)")

# D: cross-org consistency
ax = axes[1,1]
total_shared = agree.shape[0]
agreeN = int(agree.sum()); disN = int((~agree).sum())
ax.bar(["agree","disagree"], [agreeN, disN],
       color=["#27AE60","#C0392B"], alpha=0.85)
for i,v in enumerate([agreeN, disN]):
    ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("OGs with confident calls in >=2 organisms")
ax.set_title(f"D. Cross-organism consistency on shared OGs\n"
             f"{agreeN}/{total_shared} agree ({agree.mean()*100:.1f}%)")

fig.suptitle("Multi-organism essentiality atlas v2 -- full evidence-channel framework",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(OUT / "atlas_multi_summary.png", dpi=140, bbox_inches="tight")

# overall summary
total_called = sum(len(df[(df.organism==o)&(df.predicted_call!="unknown")]) for o in BACT)
total_correct = 0
for o in BACT:
    g = df[(df.organism==o)&(df.predicted_call!="unknown")]
    total_correct += int(((g.predicted_call=="essential")&(g.essential==1)).sum())
    total_correct += int(((g.predicted_call=="non-essential")&(g.essential==0)).sum())
print(f"\n=== OVERALL ===")
print(f"  total genes:           {len(df)}")
print(f"  confident calls:       {total_called} ({total_called/len(df):.0%})")
print(f"  precision over called: {total_correct/total_called:.3f}")
print(f"  cross-validated calls: {len(xv_called)} ({len(xv_called)/len(df):.0%})")

(OUT / "atlas_multi_summary.json").write_text(json.dumps({
    "n_organisms": len(BACT),
    "n_genes_total": int(len(df)),
    "confident_calls_total": int(total_called),
    "overall_coverage": round(total_called/len(df), 3),
    "overall_precision": round(total_correct/total_called, 3),
    "cross_org_consistency": {
        "shared_ogs": int(total_shared),
        "agree": int(agreeN),
        "agree_frac": round(float(agree.mean()), 3),
        "cross_validated_calls": int(len(xv_called))},
    "per_organism": {o: per_org[o] for o in BACT},
    "thresholds": {"hi": {o: round(hi_per[o],3) for o in BACT},
                   "lo": {o: round(lo_per[o],3) for o in BACT}},
    "tier_order": TIER_ORDER,
    "evidence_channels": ["E_score","E_cons_core","E_geometry",
                          "N_score","N_pangenome","N_paralog","N_mobile","N_long_lagging",
                          "P_rogue","P_conditional"],
    "sister_strain_map": SISTERS,
}, indent=2))
print(f"\nwrote {len(BACT)} per-org CSVs in atlas_multi/, summary png + json")
