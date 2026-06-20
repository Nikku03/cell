"""Three flavors of co-occurrence as essentiality predictors:

  (1) PRESENCE CO-OCCURRENCE: target OG essential when partner OG present.
      Find partner whose PRESENCE pattern is POSITIVELY correlated with target's
      ESSENTIALITY pattern. Predict: ess iff partner present in held org.
      (The positive analog of Rosconi backup mining.)

  (2) CO-ESSENTIALITY: target essential when partner OG essential elsewhere.
      Find partner whose ESSENTIALITY across orgs is positively correlated with
      target's. Predict held using the partner's essentiality in held org.

  (3) PHYLETIC-PROFILE COOCCUR FEATURE (prebuilt):
      cooccur_essential_neighbor_frac from labels/cooccurrence_features.csv --
      what fraction of the gene's phyletic-profile neighbors are essential.
      Use directly as a per-gene score, no per-fold training.

All leave-one-organism-out, vectorized.
Output: outputs/orphan/cooccurrence_test.json
"""
import csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")
ORGS = ["beril_RalstoniaGMI1000","beril_RalstoniaBSBF1503","beril_RalstoniaPSI07",
        "beril_Dda3937","beril_HerbieS","beril_Magneto"]

# ---- load orthology + labels + co-occur prebuilt ----
gene_og={}; og_present=defaultdict(set)
for r in csv.DictReader(open(DATA/"labels"/"orthology_features.csv")):
    if r["og_id"]:
        gene_og[(r["organism"],r["locus_tag"])]=r["og_id"]
        og_present[r["og_id"]].add(r["organism"])
labels={(r["organism"],r["locus_tag"]):int(r["essential"])
        for r in csv.DictReader(open(DATA/"labels"/"labels.csv"))}
all_orgs=sorted({o for og in og_present.values() for o in og})
O=len(all_orgs); org_idx={o:i for i,o in enumerate(all_orgs)}
all_ogs=sorted(og_present)
G=len(all_ogs); og_idx={og:j for j,og in enumerate(all_ogs)}

Y=np.full((O,G), np.nan, np.float32)
for (o,l),e in labels.items():
    og=gene_og.get((o,l))
    if og: Y[org_idx[o], og_idx[og]]=e
P=np.zeros((O,G), np.float32)
for og,orgs_p in og_present.items():
    j=og_idx[og]
    for o in orgs_p: P[org_idx[o],j]=1.0
print(f"O={O} orgs, G={G} OGs")

# (3) prebuilt phyletic-cooccur feature: cooccur_essential_neighbor_frac per gene
cofeat={}
for r in csv.DictReader(open(DATA/"labels"/"cooccurrence_features.csv")):
    try:
        cofeat[(r["organism"],r["locus_tag"])]=float(r["cooccur_essential_neighbor_frac"])
    except: pass
print(f"prebuilt cooccur feature: {len(cofeat)} genes")

def auc(s,y):
    y=np.asarray(y); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return float("nan")
    r=np.argsort(np.argsort(s))+1; return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

# Candidate partner OGs (informative variation in presence)
p_sum=P.sum(0); cand_p_mask=(p_sum>=5)&(p_sum<=O-5)
cand_p_idx=np.where(cand_p_mask)[0]
# Candidate partner OGs by essentiality (need enough labeled orgs)
n_lab=(~np.isnan(Y)).sum(0); mean_e=np.nanmean(Y, axis=0)
cand_e_mask=(n_lab>=10) & (mean_e>0.05) & (mean_e<0.95)
cand_e_idx=np.where(cand_e_mask)[0]
print(f"candidate presence-partners: {len(cand_p_idx)}; essentiality-partners: {len(cand_e_idx)}")

# Standardize once
Pc=P[:,cand_p_idx] - P[:,cand_p_idx].mean(0)
Pn=Pc/(Pc.std(0)+1e-9)                 # [O, n_cand_p]

# ---------------- Targets: variable-essentiality OGs ----------------
target_mask=(n_lab>=10)&(mean_e>0.1)&(mean_e<0.9)
target_idx=np.where(target_mask)[0]
import random; random.seed(0)
sample_targets=random.sample(list(target_idx), min(500, len(target_idx)))
print(f"target OGs sampled: {len(sample_targets)}")

# ---------------- Tests (1) & (2) ----------------
results_pres=[]; results_coess=[]
for held in ORGS:
    if held not in org_idx: continue
    hi=org_idx[held]
    mask=np.ones(O, bool); mask[hi]=False

    # Standardized presence on training orgs
    Pn_tr = Pn[mask]
    n_tr=mask.sum()
    # For co-essentiality, build essentiality matrix on training orgs for candidate-e partners
    Y_tr=Y[mask][:, cand_e_idx]   # [O_tr, n_cand_e], NaN where unlabeled
    # Mean-impute for centering (per column)
    Y_mean = np.nanmean(Y_tr, axis=0); Y_mean[np.isnan(Y_mean)]=0.5
    Y_tr_imp = np.where(np.isnan(Y_tr), Y_mean[None,:], Y_tr)
    Y_tr_c = Y_tr_imp - Y_tr_imp.mean(0)
    Y_tr_std = Y_tr_c.std(0)+1e-9
    Y_tr_z = Y_tr_c / Y_tr_std

    for j in sample_targets:
        e_full=Y[:,j]; e_tr=e_full[mask]
        v=~np.isnan(e_tr)
        if v.sum()<8: continue
        et=e_tr[v]
        if et.std()<0.1: continue
        et_z=(et-et.mean())/(et.std()+1e-9)

        # ---- (1) presence partner: positive correlation ----
        Pv=Pn_tr[v]; Pvc=Pv-Pv.mean(0); Pvz=Pvc/(Pvc.std(0)+1e-9)
        corr_p=(et_z @ Pvz)/len(et_z)
        # ban self
        if j in cand_p_idx:
            pos=np.searchsorted(cand_p_idx, j)
            if pos<len(cand_p_idx) and cand_p_idx[pos]==j: corr_p[pos]=-2.0
        bp=int(np.argmax(corr_p)); sc=float(corr_p[bp])
        if sc>=0.4:
            partner_og=cand_p_idx[bp]
            partner_present_held = int(P[hi, partner_og]==1.0)
            pred = partner_present_held   # essential iff partner present
            truth=e_full[hi]
            if not np.isnan(truth):
                results_pres.append((pred, int(truth), sc))

        # ---- (2) essentiality partner ----
        # corr of target essentiality with partner essentiality across training orgs
        # only use the labeled-subset positions (v); restrict Y_tr_z to v as well
        Yv=Y_tr_z[v]
        Yvc=Yv-Yv.mean(0); Yvz=Yvc/(Yvc.std(0)+1e-9)
        corr_e=(et_z @ Yvz)/len(et_z)
        # ban self
        if j in cand_e_idx:
            pos=np.searchsorted(cand_e_idx, j)
            if pos<len(cand_e_idx) and cand_e_idx[pos]==j: corr_e[pos]=-2.0
        bp2=int(np.argmax(corr_e)); sc2=float(corr_e[bp2])
        if sc2>=0.5:
            partner_og2=cand_e_idx[bp2]
            partner_ess_held=Y[hi, partner_og2]
            if not np.isnan(partner_ess_held):
                truth=e_full[hi]
                if not np.isnan(truth):
                    pred = int(partner_ess_held>=0.5)
                    results_coess.append((pred, int(truth), sc2))

# ---------------- Report (1) and (2) ----------------
print("\n=== (1) PRESENCE CO-OCCURRENCE (essential iff partner present) ===")
if results_pres:
    R=np.array(results_pres); pr=R[:,0].astype(int); tr=R[:,1].astype(int); sc=R[:,2]
    base=tr.mean(); acc=(pr==tr).mean()
    print(f"  predictions: {len(R)}  base rate {base:.3f}  accuracy {acc:.3f}")
    for thr in (0.5,0.6,0.7,0.8):
        m=sc>=thr
        if m.sum()>=20:
            print(f"  corr>={thr}: n={m.sum()} acc={(pr[m]==tr[m]).mean():.3f}  truth-ess-rate {tr[m].mean():.2f}")
else: print("  no predictions")

print("\n=== (2) CO-ESSENTIALITY (partner essentiality) ===")
if results_coess:
    R=np.array(results_coess); pr=R[:,0].astype(int); tr=R[:,1].astype(int); sc=R[:,2]
    base=tr.mean(); acc=(pr==tr).mean()
    print(f"  predictions: {len(R)}  base rate {base:.3f}  accuracy {acc:.3f}")
    for thr in (0.6,0.7,0.8,0.9):
        m=sc>=thr
        if m.sum()>=20:
            print(f"  corr>={thr}: n={m.sum()} acc={(pr[m]==tr[m]).mean():.3f}  truth-ess-rate {tr[m].mean():.2f}")
else: print("  no predictions")

# ---------------- (3) Pre-built cooccur_essential_neighbor_frac ----------------
print("\n=== (3) PREBUILT cooccur_essential_neighbor_frac ===")
# Pool all atlas-6 genes; evaluate as a predictor of essentiality
xs=[]; ys=[]
for o in ORGS:
    for (oo,l),e in labels.items():
        if oo!=o: continue
        v=cofeat.get((oo,l))
        if v is None: continue
        xs.append(v); ys.append(e)
xs=np.array(xs); ys=np.array(ys)
print(f"  n={len(xs)} atlas-6 genes with feature; ess rate {ys.mean():.3f}")
print(f"  AUC of cooccur_essential_neighbor_frac vs essentiality: {auc(xs, ys):.3f}")
# top-decile precision
o=np.argsort(-xs); k=int(0.1*len(xs))
print(f"  top-10% by feature: ess rate {ys[o][:k].mean():.3f} ({ys[o][:k].mean()/ys.mean():.2f}x lift)")
# bottom-decile (low feature -> low essentiality?)
print(f"  bottom-10%: ess rate {ys[o][-k:].mean():.3f} ({ys[o][-k:].mean()/ys.mean():.2f}x lift)")

# Save
out=dict()
if results_pres:
    R=np.array(results_pres); sc=R[:,2]; pr=R[:,0].astype(int); tr=R[:,1].astype(int)
    out["presence_cooccurrence"]=dict(n=int(len(R)), base_rate=round(float(tr.mean()),3),
        accuracy=round(float((pr==tr).mean()),3),
        thresholds={str(t):{"n":int((sc>=t).sum()),
                            "acc":round(float((pr[sc>=t]==tr[sc>=t]).mean()),3) if (sc>=t).sum()>=20 else None}
                    for t in (0.5,0.6,0.7,0.8)})
if results_coess:
    R=np.array(results_coess); sc=R[:,2]; pr=R[:,0].astype(int); tr=R[:,1].astype(int)
    out["coessentiality"]=dict(n=int(len(R)), base_rate=round(float(tr.mean()),3),
        accuracy=round(float((pr==tr).mean()),3),
        thresholds={str(t):{"n":int((sc>=t).sum()),
                            "acc":round(float((pr[sc>=t]==tr[sc>=t]).mean()),3) if (sc>=t).sum()>=20 else None}
                    for t in (0.6,0.7,0.8,0.9)})
out["prebuilt_phyletic_cooccur"]=dict(n=int(len(xs)),
    auc=round(auc(xs,ys),3),
    top10_ess_rate=round(float(ys[o][:k].mean()),3),
    bottom10_ess_rate=round(float(ys[o][-k:].mean()),3),
    base_rate=round(float(ys.mean()),3))
json.dump(out, open(OUT/"cooccurrence_test.json","w"), indent=2)
print("\nwrote cooccurrence_test.json")
