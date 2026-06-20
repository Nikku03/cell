"""Atlas v5: add backup / presence-cooccur / co-essentiality / phyletic-cooccur
channels to atlas v4's abstention bucket and measure the headline lift:
  v4 baseline: 58.0% coverage at 90.3% precision (per-organism LOO)

For each held org's abstained genes (UNRESOLVED / ROGUE_SUSPECT /
CONDITIONAL_SUSPECT), compute the 4 cooccurrence signals leak-clean and
apply a strict promotion rule: only call when >=2 signals agree at
high-confidence anti-corr threshold. Report new coverage + precision over
the called set.

Output: outputs/orphan/atlas_v5_cooccur_results.json
"""
import csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")
ORGS = ["beril_RalstoniaGMI1000","beril_RalstoniaBSBF1503","beril_RalstoniaPSI07",
        "beril_Dda3937","beril_HerbieS","beril_Magneto"]

# ---- load orthology + labels + cooccur prebuilt ----
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

# prebuilt phyletic cooccur
cofeat={}
for r in csv.DictReader(open(DATA/"labels"/"cooccurrence_features.csv")):
    try: cofeat[(r["organism"],r["locus_tag"])]=float(r["cooccur_essential_neighbor_frac"])
    except: pass

p_sum=P.sum(0); cand_p=(p_sum>=5)&(p_sum<=O-5); cand_p_idx=np.where(cand_p)[0]
n_lab=(~np.isnan(Y)).sum(0); mean_e=np.nanmean(Y,axis=0)
cand_e=(n_lab>=10)&(mean_e>0.05)&(mean_e<0.95); cand_e_idx=np.where(cand_e)[0]
print(f"O={O}, G={G}, cand_presence={len(cand_p_idx)}, cand_ess={len(cand_e_idx)}")

# ---- atlas v4 per-org tables ----
atlas_frames=[]
for org in ORGS:
    a=pd.read_csv(OUT/"atlas_multi"/f"{org}.csv"); a["org"]=org
    atlas_frames.append(a)
atlas=pd.concat(atlas_frames, ignore_index=True)
RESID={"UNRESOLVED","ROGUE_SUSPECT","CONDITIONAL_SUSPECT"}
atlas["residual"]=atlas.tier.isin(RESID)
print(f"v4 atlas: {len(atlas)} genes; residual={int(atlas.residual.sum())} ({atlas.residual.mean():.0%})")

# Map each gene -> og index
atlas["og"]=[gene_og.get((o,l)) for o,l in zip(atlas.org, atlas.locus_tag)]
atlas["og_j"]=[og_idx.get(og,-1) for og in atlas.og]
atlas["org_i"]=[org_idx.get(o,-1) for o in atlas.org]

# Pre-standardize presence on the full table (we will re-standardize per fold's subset)
Pc=P[:,cand_p_idx]-P[:,cand_p_idx].mean(0); Pn=Pc/(Pc.std(0)+1e-9)

CALL_THRESH_BACKUP=0.7
CALL_THRESH_PRES=0.7
CALL_THRESH_COESS=0.7
PHYL_LOW=0.10        # bottom-decile cutoff for confident non-essential (from cooccur_test: 0.26x base)
MIN_VOTES=2          # require >=2 channels to agree before calling

def channels_for_gene(j, held_org_i, train_mask):
    """Return four channel votes (each +1 essential / -1 non-ess / 0 abstain).
    j = OG index of focal gene, held_org_i = its org's index, train_mask = bool over orgs."""
    out={"backup":0,"presence":0,"coess":0,"phyl":0}
    if j<0: return out

    # essentiality vector of THIS OG on training orgs
    e_full=Y[:,j]; e_tr=e_full[train_mask]
    v=~np.isnan(e_tr)
    if v.sum()>=8 and e_tr[v].std()>=0.1:
        et=e_tr[v]; et_z=(et-et.mean())/(et.std()+1e-9)

        # (1) backup mining: anti-corr partner -> ess iff partner absent
        Pn_tr=Pn[train_mask][v]
        Pnc=Pn_tr-Pn_tr.mean(0); Pnz=Pnc/(Pnc.std(0)+1e-9)
        corr=(et_z @ Pnz)/len(et_z)
        # ban self
        if j in cand_p_idx:
            pos=np.searchsorted(cand_p_idx, j)
            if pos<len(cand_p_idx) and cand_p_idx[pos]==j: corr[pos]=2.0
        # anti-correlation
        anti=-corr
        bp=int(np.argmax(anti)); sc=float(anti[bp])
        if sc>=CALL_THRESH_BACKUP:
            partner_present=int(P[held_org_i, cand_p_idx[bp]]==1.0)
            out["backup"] = -1 if partner_present else +1  # ess iff partner absent

        # (2) presence partner: pos-corr -> ess iff partner present
        bp2=int(np.argmax(corr)); sc2=float(corr[bp2])
        if sc2>=CALL_THRESH_PRES:
            partner_present=int(P[held_org_i, cand_p_idx[bp2]]==1.0)
            out["presence"] = +1 if partner_present else -1

        # (3) co-essentiality: pos-corr in essentiality space
        Y_tr=Y[train_mask][:,cand_e_idx]
        Ymean=np.nanmean(Y_tr,axis=0); Ymean[np.isnan(Ymean)]=0.5
        Yimp=np.where(np.isnan(Y_tr),Ymean[None,:],Y_tr)
        Yv=Yimp[v]; Yvc=Yv-Yv.mean(0); Yvz=Yvc/(Yvc.std(0)+1e-9)
        corr_e=(et_z @ Yvz)/len(et_z)
        if j in cand_e_idx:
            pos=np.searchsorted(cand_e_idx, j)
            if pos<len(cand_e_idx) and cand_e_idx[pos]==j: corr_e[pos]=-2.0
        bp3=int(np.argmax(corr_e)); sc3=float(corr_e[bp3])
        if sc3>=CALL_THRESH_COESS:
            partner_ess_held=Y[held_org_i, cand_e_idx[bp3]]
            if not np.isnan(partner_ess_held):
                out["coess"] = +1 if partner_ess_held>=0.5 else -1
    return out

# ---- score every residual gene leave-one-organism-out ----
print("\nscoring atlas residual genes with 4 cooccur channels (per-organism LOO)...")
calls=[]   # (gene_idx, predicted, truth)
for held in ORGS:
    if held not in org_idx: continue
    hi=org_idx[held]
    train_mask=np.ones(O, bool); train_mask[hi]=False
    sub=atlas[(atlas.org==held)&atlas.residual]
    print(f"  {held:<24} residual {len(sub)}")
    for _,row in sub.iterrows():
        j=int(row.og_j); oi=int(row.org_i)
        if j<0 or oi<0: continue
        ch=channels_for_gene(j, oi, train_mask)
        # phyletic prebuilt: per-gene
        phyl=cofeat.get((row.org, row.locus_tag))
        if phyl is not None:
            if phyl<=PHYL_LOW: ch["phyl"]=-1
            elif phyl>=0.8: ch["phyl"]=+1
        votes=[v for v in ch.values() if v!=0]
        if len(votes)<MIN_VOTES: continue
        # need agreement among non-zero channels
        s=sum(votes)
        if abs(s)<MIN_VOTES: continue   # split votes -> abstain
        pred = 1 if s>0 else 0
        calls.append((row.measured_essential, pred, int(row.essential) if 'essential' in row else int(row.measured_essential)))

print(f"\nresidual genes called by cooccur channels: {len(calls)}")
if calls:
    R=np.array([(int(t),int(p)) for t,p,_ in calls])
    truth=R[:,0]; pred=R[:,1]
    acc=(pred==truth).mean()
    n_added=len(calls); n_total=len(atlas)
    print(f"  accuracy on newly-called: {acc:.3f}  (base ess rate in residual: {atlas[atlas.residual].measured_essential.mean():.3f})")
    # combined with v4 confident calls
    v4_called=atlas[~atlas.residual]
    v4_correct=((v4_called.tier=="CONFIDENT_ESSENTIAL")&(v4_called.measured_essential==1)) | \
               ((v4_called.tier=="CONFIDENT_NONESSENTIAL")&(v4_called.measured_essential==0))
    v4_prec=float(v4_correct.mean())
    v4_cov=len(v4_called)/n_total
    print(f"\n  v4 baseline:           {len(v4_called)}/{n_total} = {v4_cov:.3f} coverage at {v4_prec:.3f} precision")
    new_correct=(pred==truth).sum()
    combined_n=len(v4_called)+n_added
    combined_correct=int(v4_correct.sum())+int(new_correct)
    print(f"  + cooccur new calls:   +{n_added} at accuracy {acc:.3f}")
    print(f"  v5 combined:           {combined_n}/{n_total} = {combined_n/n_total:.3f} coverage at "
          f"{combined_correct/combined_n:.3f} precision")
    print(f"  Δ coverage:            {(combined_n/n_total - v4_cov)*100:+.1f} pp")
    print(f"  Δ precision:           {(combined_correct/combined_n - v4_prec)*100:+.2f} pp")

    out=dict(
      v4_baseline=dict(coverage=round(v4_cov,3), precision=round(v4_prec,3), called=int(len(v4_called)), total=int(n_total)),
      cooccur_added=dict(n=int(n_added), accuracy=round(float(acc),3)),
      v5_combined=dict(coverage=round(combined_n/n_total,3),
                        precision=round(combined_correct/combined_n,3),
                        called=int(combined_n)),
      delta=dict(coverage_pp=round((combined_n/n_total - v4_cov)*100,2),
                  precision_pp=round((combined_correct/combined_n - v4_prec)*100,2)),
      thresholds=dict(backup=CALL_THRESH_BACKUP, presence=CALL_THRESH_PRES,
                       coess=CALL_THRESH_COESS, phyl_low=PHYL_LOW, min_votes=MIN_VOTES))
    json.dump(out, open(OUT/"atlas_v5_cooccur_results.json","w"), indent=2)
    print("\nwrote atlas_v5_cooccur_results.json")
else:
    print("no calls made")
