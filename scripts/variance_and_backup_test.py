"""Test two literature-driven moves cheaply, sandbox-only:

  (A) ESSENTIALITY-VARIANCE FLAG: per OG, compute the variance of measured
      essentiality across labeled organisms. High-variance families are
      context-dependent; the feature predicts UNPREDICTABILITY (= which
      genes to abstain on). Leak-free: variance computed from OTHER orgs.

  (B) PAN-GENOME BACKUP MINING: for each gene, find candidate backups -- OGs
      whose PRESENCE is anti-correlated with our gene's essentiality across
      strains. If gene X is essential exactly in the strains lacking gene Y,
      Y is X's backup, and we can predict X's essentiality FROM Y's presence.

For each: leave-one-organism-out, measure AUC + abstention precision.
Output: outputs/orphan/variance_and_backup_test.json
"""
import csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")
ORGS = ["beril_RalstoniaGMI1000","beril_RalstoniaBSBF1503","beril_RalstoniaPSI07",
        "beril_Dda3937","beril_HerbieS","beril_Magneto"]

# gene -> og ; og -> {org: essentiality} ; og -> set of orgs present
gene_og={}; og_org_ess=defaultdict(dict); og_present=defaultdict(set)
for r in csv.DictReader(open(DATA/"labels"/"orthology_features.csv")):
    if r["og_id"]:
        gene_og[(r["organism"],r["locus_tag"])]=r["og_id"]
        og_present[r["og_id"]].add(r["organism"])
labels={(r["organism"],r["locus_tag"]):int(r["essential"])
        for r in csv.DictReader(open(DATA/"labels"/"labels.csv"))}
for (o,l),e in labels.items():
    og=gene_og.get((o,l))
    if og: og_org_ess[og][o]=e
all_orgs=sorted({o for og in og_present.values() for o in og})
print(f"labeled orgs in orthology: {len(all_orgs)}; OGs with >=1 labeled member: {len(og_org_ess)}")

# Build per-gene set used by both tests
rows=[]
for (o,l),e in labels.items():
    og=gene_og.get((o,l))
    if not og or o not in ORGS: continue
    rows.append(dict(org=o, lt=l, ess=e, og=og))
print(f"atlas-6 genes with OG: {len(rows)}")

# ---------------------- (A) ESSENTIALITY-VARIANCE FLAG ----------------------
def auc(s,y):
    y=np.asarray(y); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return float("nan")
    r=np.argsort(np.argsort(s))+1; return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def variance_feature(held):
    """OG -> std of essentiality across labeled orgs EXCEPT the held org."""
    out={}
    for og,d in og_org_ess.items():
        vals=[v for o,v in d.items() if o!=held]
        if len(vals)>=3: out[og]=float(np.std(vals))
    return out

# predict UNPREDICTABILITY = |pred - actual| would be high.
# Operational: identify genes that DISAGREE with the family majority -- those are
# the context-flipping ones. variance flag should be enriched in disagreers.
print("\n=== (A) essentiality-variance flag ===")
disagrees=[]; var_flag=[]; ess=[]
for row in rows:
    vmap = variance_feature(row["org"])
    v = vmap.get(row["og"], np.nan)
    if np.isnan(v): continue
    family_ess=og_org_ess[row["og"]]
    # family majority (excluding held org)
    others=[v2 for o2,v2 in family_ess.items() if o2!=row["org"]]
    if len(others)<3: continue
    fam_majority = 1 if np.mean(others)>=0.5 else 0
    disagree = int(row["ess"]!=fam_majority)
    disagrees.append(disagree); var_flag.append(v); ess.append(row["ess"])
disagrees=np.array(disagrees); var_flag=np.array(var_flag); ess=np.array(ess)
n=len(disagrees); n_dis=disagrees.sum()
print(f"  n={n}, family-majority disagreers: {n_dis} ({n_dis/n:.1%})")
print(f"  AUC of variance-flag predicting disagreement: {auc(var_flag, disagrees):.3f}")
print(f"     (i.e. can we predict which genes will buck their family?)")
# precision in the top-decile by variance
o=np.argsort(-var_flag); k=int(0.1*n)
print(f"  top-10% by variance: disagreement rate {disagrees[o][:k].mean():.3f} vs base {n_dis/n:.3f} "
      f"({disagrees[o][:k].mean()/(n_dis/n):.2f}x lift)")
# does variance flag improve essentiality prediction when COMBINED w/ majority vote?
maj_vote=[]; vf=[]; y=[]
for row in rows:
    family_ess=og_org_ess[row["og"]]
    others=[v2 for o2,v2 in family_ess.items() if o2!=row["org"]]
    if len(others)<3: continue
    maj_vote.append(np.mean(others))   # = family_frac
    v=variance_feature(row["org"]).get(row["og"], 0.0)
    vf.append(v); y.append(row["ess"])
maj_vote=np.array(maj_vote); vf=np.array(vf); y=np.array(y)
# logistic of [majority, -variance]: penalize high variance (less confident)
mu0,sd0=maj_vote.mean(),maj_vote.std()+1e-9
mu1,sd1=vf.mean(),vf.std()+1e-9
Z=np.c_[np.ones(len(y)),(maj_vote-mu0)/sd0,(vf-mu1)/sd1]
w=np.zeros(Z.shape[1])
for _ in range(400):
    p=1/(1+np.exp(-Z@w)); g=Z.T@(p-y.astype(float))/len(y); w-=0.5*g
print(f"  majority-vote AUC: {auc(maj_vote, y):.3f}")
print(f"  majority + variance combined AUC: {auc(1/(1+np.exp(-Z@w)), y):.3f}")
print(f"  variance coefficient (standardized): {w[2]:+.3f}  (negative = high variance lowers conf)")

# ---------------------- (B) PAN-GENOME BACKUP MINING ----------------------
print("\n=== (B) pan-genome backup mining ===")
# For each test OG, find a CANDIDATE BACKUP: another OG whose PRESENCE pattern
# (across labeled organisms) is anti-correlated with the test OG's ESSENTIALITY pattern.
# Anti-corr: OG_X is essential IFF backup_Y is ABSENT.
# Leak-clean: training uses only OTHER organisms; predict held org's essentiality
# = (1 - present(held, backup)).

# Build presence matrix per organism (which OGs are present)
present_by_org={o:set() for o in all_orgs}
for og,orgs_p in og_present.items():
    for o in orgs_p: present_by_org[o].add(og)
# binary essentiality matrix per (og,org) only where labeled
ess_dict={(og,o):v for og,d in og_org_ess.items() for o,v in d.items()}

# Find candidate backups for OGs with sufficient essentiality variance
candidates={}
ess_vec_cache={}
def vec(og, orgs_subset):
    if (og,tuple(orgs_subset)) in ess_vec_cache: return ess_vec_cache[(og,tuple(orgs_subset))]
    out=np.array([ess_dict.get((og,o),-1) for o in orgs_subset])
    ess_vec_cache[(og,tuple(orgs_subset))]=out
    return out

# Restrict search: OGs with enough variance and enough labeled orgs (>=10)
target_ogs=[og for og,d in og_org_ess.items() if len(d)>=10 and 0.1<np.mean(list(d.values()))<0.9]
print(f"  candidate target OGs (variable essentiality, >=10 labeled orgs): {len(target_ogs)}")

# Build a presence vector per OG over the labeled orgs (cheap)
# anti-correlation: high score when essential(target,org)=1 AND absent(backup,org)=0
# Cap search per target to top-100 backup candidates by negative pearson
def best_backup(target_og, train_orgs):
    e_target = np.array([ess_dict.get((target_og,o),-1) for o in train_orgs], float)
    mask = e_target>=0
    if mask.sum()<5: return None,0.0
    et=e_target[mask]
    if et.std()<0.1: return None,0.0
    best=None; best_score=-1
    # Iterate over candidate OGs that have at least some presence variation
    for cand,orgs_with in og_present.items():
        if cand==target_og: continue
        present_vec=np.array([1.0 if o in orgs_with else 0.0 for o in train_orgs])[mask]
        if present_vec.std()<0.1: continue
        # anti-correlation: when target ess high, backup absent (low)
        corr = -np.corrcoef(et, present_vec)[0,1]
        if corr>best_score:
            best_score=corr; best=cand
    return best, best_score

# Run on a small sample (full search is O(target_ogs * all_OGs) which is slow)
import random
random.seed(0)
sample = random.sample(target_ogs, min(200, len(target_ogs)))
predictions=[]; truths=[]; corrs=[]
for held in ORGS:
    train_orgs=[o for o in all_orgs if o!=held]
    for og in sample:
        if og not in og_org_ess: continue
        truth=og_org_ess[og].get(held)
        if truth is None: continue
        backup,score=best_backup(og, train_orgs)
        if backup is None or score<0.4: continue   # require non-trivial anti-corr
        # predict: essential iff backup ABSENT in held
        pred_ess = 1 if (held not in og_present[backup]) else 0
        predictions.append(pred_ess); truths.append(truth); corrs.append(score)
predictions=np.array(predictions); truths=np.array(truths); corrs=np.array(corrs)
print(f"  predictions made (anti-corr>=0.4): {len(predictions)}")
if len(predictions):
    acc=(predictions==truths).mean()
    print(f"  backup-mining accuracy: {acc:.3f} (base rate {truths.mean():.3f})")
    # high-confidence subset (anti-corr >= 0.7)
    hi=corrs>=0.7
    if hi.sum()>=20:
        print(f"  high-confidence (anti-corr>=0.7, n={hi.sum()}): acc {(predictions[hi]==truths[hi]).mean():.3f}")
    print(f"  mean anti-corr score: {corrs.mean():.3f}")
else:
    print("  no qualifying backup predictions")

json.dump(dict(
  variance_flag=dict(
    auc_predicting_family_disagreement=round(auc(var_flag,disagrees),3),
    top10pct_variance_lift=round(disagrees[o][:k].mean()/(n_dis/n),2),
    majority_vote_auc=round(auc(maj_vote,y),3),
    majority_plus_variance_auc=round(auc(1/(1+np.exp(-Z@w)),y),3),
    variance_coef_standardized=round(float(w[2]),3)),
  backup_mining=dict(
    target_ogs_searched=len(sample),
    predictions_made=int(len(predictions)),
    accuracy=round(float((predictions==truths).mean()),3) if len(predictions) else None,
    base_rate=round(float(truths.mean()),3) if len(predictions) else None,
    high_conf_n=int((corrs>=0.7).sum()) if len(predictions) else 0,
    high_conf_acc=round(float((predictions[corrs>=0.7]==truths[corrs>=0.7]).mean()),3) if (len(predictions) and (corrs>=0.7).sum()>=20) else None,
    mean_anti_corr=round(float(corrs.mean()),3) if len(predictions) else None),
), open(OUT/"variance_and_backup_test.json","w"), indent=2)
print("\nwrote variance_and_backup_test.json")
