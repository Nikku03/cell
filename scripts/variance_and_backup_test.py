"""Vectorized test of two literature-driven moves (sandbox, no GPU/network):

  (A) ESSENTIALITY-VARIANCE FLAG: per OG, std of essentiality across labeled
      orgs (LOO-org, leak-free). Predicts which genes will disagree with their
      family majority -- = which genes the atlas should abstain on.

  (B) PAN-GENOME BACKUP MINING (Rosconi 2022 frame): for each test OG with
      variable essentiality, find the backup OG whose presence is most
      anti-correlated with its essentiality across organisms. Predict held-out
      essentiality from backup presence.

Both vectorized via numpy presence/essentiality matrices.
"""
import csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")
ORGS = ["beril_RalstoniaGMI1000","beril_RalstoniaBSBF1503","beril_RalstoniaPSI07",
        "beril_Dda3937","beril_HerbieS","beril_Magneto"]

# --- load orthology + labels ---
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
print(f"orgs={O}, OGs={G}")

# essentiality matrix Y[O,G]: 0/1 if labeled, NaN if not
Y=np.full((O,G), np.nan, np.float32)
for (o,l),e in labels.items():
    og=gene_og.get((o,l))
    if og: Y[org_idx[o], og_idx[og]] = e
# presence matrix P[O,G]: 1 if org has the OG, 0 otherwise
P=np.zeros((O,G), np.float32)
for og,orgs_p in og_present.items():
    j=og_idx[og]
    for o in orgs_p: P[org_idx[o],j]=1.0
print(f"essentiality entries: {int((~np.isnan(Y)).sum()):,}; presence-1 entries: {int(P.sum()):,}")

def auc(s,y):
    y=np.asarray(y); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return float("nan")
    r=np.argsort(np.argsort(s))+1; return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

# ---------------------- (A) VARIANCE FLAG ----------------------
print("\n=== (A) essentiality-variance flag ===")
# Per OG: vector of essentialities across orgs (NaN where unlabeled)
# variance excluding org i = std of Y[~i, j] over non-NaN entries
# Compute once per held-org via vectorized masking
results=[]
for held in ORGS:
    hi=org_idx[held]
    mask=np.ones(O, bool); mask[hi]=False
    Y_tr=Y[mask]                # [O-1, G]
    # per OG: count + sum
    valid=~np.isnan(Y_tr)
    n=valid.sum(0)              # [G]
    s=np.where(valid, Y_tr, 0).sum(0)
    s2=np.where(valid, Y_tr*Y_tr, 0).sum(0)
    mean = np.where(n>0, s/np.maximum(n,1), 0)
    var  = np.where(n>0, s2/np.maximum(n,1) - mean**2, 0)
    std  = np.sqrt(np.maximum(var,0))
    maj  = (mean>=0.5).astype(np.float32)
    # for genes in held organism with labels, look up
    for (o,l),e in labels.items():
        if o!=held: continue
        og=gene_og.get((o,l))
        if og is None: continue
        j=og_idx[og]
        if n[j]<3: continue
        results.append((std[j], float(maj[j]), float(mean[j]), e))
results=np.array(results)
stds=results[:,0]; majs=results[:,1]; means=results[:,2]; ys=results[:,3].astype(int)
disagrees=(ys!=majs.astype(int)).astype(int)
n_total=len(ys); n_dis=disagrees.sum()
print(f"  atlas-6 genes with family-majority computable: {n_total}, disagreers: {n_dis} ({n_dis/n_total:.1%})")
print(f"  AUC of variance-flag predicting disagreement: {auc(stds, disagrees):.3f}")
o=np.argsort(-stds); k=int(0.1*n_total)
top10_lift=disagrees[o][:k].mean()/(n_dis/n_total)
print(f"  top-10% by variance: disagreement rate {disagrees[o][:k].mean():.3f} (lift {top10_lift:.2f}x)")
# combined with majority
Z=np.c_[np.ones(n_total), (means-means.mean())/(means.std()+1e-9), (stds-stds.mean())/(stds.std()+1e-9)]
w=np.zeros(3); yf=ys.astype(float)
for _ in range(400):
    p=1/(1+np.exp(-Z@w)); g=Z.T@(p-yf)/n_total; w-=0.5*g
auc_maj=auc(means, ys); auc_combo=auc(1/(1+np.exp(-Z@w)), ys)
print(f"  majority-vote (mean) AUC: {auc_maj:.3f}")
print(f"  majority + variance combined AUC: {auc_combo:.3f}  (variance coef {w[2]:+.3f})")

# ---------------------- (B) PAN-GENOME BACKUP MINING ----------------------
print("\n=== (B) pan-genome backup mining ===")
# Find target OGs with variable essentiality (>=10 labeled orgs, mean in 0.1-0.9)
n_lab=(~np.isnan(Y)).sum(0)            # [G]
mean_ess=np.nanmean(Y, axis=0)
target_mask = (n_lab>=10) & (mean_ess>0.1) & (mean_ess<0.9)
target_idx=np.where(target_mask)[0]
print(f"  target OGs (>=10 labeled, variable ess): {len(target_idx)}")
# Restrict candidate backups to OGs present in 5-43 orgs (informative variation)
p_sum=P.sum(0)
cand_mask = (p_sum>=5) & (p_sum<=O-5)
cand_idx=np.where(cand_mask)[0]
print(f"  candidate backups (informative presence variation): {len(cand_idx)}")
# Standardize P columns once
Pc=P[:,cand_idx] - P[:,cand_idx].mean(0)
Pc_std=Pc.std(0)+1e-9
Pn=Pc/Pc_std                            # [O, n_cand]

results_b=[]
import random; random.seed(0)
sample_targets=random.sample(list(target_idx), min(500, len(target_idx)))
print(f"  vectorized search over {len(sample_targets)} targets x {len(cand_idx)} candidates ...")

for held in ORGS:
    hi=org_idx[held]; mask=np.ones(O, bool); mask[hi]=False
    Pn_tr=Pn[mask]                       # [O-1, n_cand]
    n_tr=mask.sum()
    for j in sample_targets:
        e_full=Y[:,j]
        # exclude held org
        e_tr=e_full[mask]
        valid=~np.isnan(e_tr)
        if valid.sum()<8: continue
        et=e_tr[valid]
        if et.std()<0.1: continue
        et_z=(et-et.mean())/(et.std()+1e-9)
        # anti-correlation = -pearson( et_z, presence_z ) = -dot/n_valid
        # Pn_tr[valid] : [n_valid, n_cand]
        Pv=Pn_tr[valid]
        # to get true pearson with the SUBSAMPLE we need to re-standardize Pv
        Pv_c = Pv - Pv.mean(0)
        Pv_std = Pv_c.std(0)+1e-9
        Pv_z = Pv_c/Pv_std
        corr = (et_z @ Pv_z) / len(et_z)   # [n_cand]
        anticorr = -corr
        best_pos = int(np.argmax(anticorr))
        score = float(anticorr[best_pos])
        if score < 0.4: continue
        backup_og = all_ogs[cand_idx[best_pos]]
        # predict: essential iff backup absent in held
        backup_present_held = int(P[hi, cand_idx[best_pos]]==1.0)
        pred_ess = 1 - backup_present_held
        truth = e_full[hi]
        if np.isnan(truth): continue
        results_b.append((pred_ess, int(truth), score))

if results_b:
    R=np.array(results_b)
    pr=R[:,0].astype(int); tr=R[:,1].astype(int); sc=R[:,2]
    acc=(pr==tr).mean(); base=tr.mean()
    print(f"  predictions made (anti-corr>=0.4): {len(R)}")
    print(f"  backup-mining accuracy: {acc:.3f}  (base rate {base:.3f})")
    print(f"  mean anti-corr score: {sc.mean():.3f}")
    for thr in (0.5, 0.6, 0.7, 0.8):
        m=sc>=thr
        if m.sum()>=20:
            print(f"  anti-corr>={thr}: n={m.sum()} accuracy={(pr[m]==tr[m]).mean():.3f}  ess-rate-truth={tr[m].mean():.2f}")
else:
    print("  no qualifying predictions"); acc=base=float("nan")

json.dump(dict(
  variance_flag=dict(
    n=int(n_total),
    auc_predicting_family_disagreement=round(auc(stds,disagrees),3),
    top10pct_variance_lift=round(top10_lift,2),
    majority_vote_auc=round(auc_maj,3),
    majority_plus_variance_auc=round(auc_combo,3),
    variance_coef_standardized=round(float(w[2]),3)),
  backup_mining=dict(
    target_ogs_searched=len(sample_targets),
    candidate_backup_ogs=int(cand_mask.sum()),
    predictions_made=int(len(results_b)),
    accuracy=round(float((np.array(results_b)[:,0]==np.array(results_b)[:,1]).mean()),3) if results_b else None,
    base_rate=round(float(np.array(results_b)[:,1].mean()),3) if results_b else None,
    high_conf_thresholds={
      str(t): {"n": int((np.array(results_b)[:,2]>=t).sum()),
               "acc": round(float((np.array(results_b)[(np.array(results_b)[:,2]>=t),0]==np.array(results_b)[(np.array(results_b)[:,2]>=t),1]).mean()),3)
                     if (np.array(results_b)[:,2]>=t).sum()>=20 else None}
      for t in (0.5,0.6,0.7,0.8)} if results_b else {})
), open(OUT/"variance_and_backup_test.json","w"), indent=2)
print("\nwrote variance_and_backup_test.json")
