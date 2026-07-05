"""STAGE-1 TEST — does the LINCS-trained dynamics transformer beat 0.25 on a dense external time course?

The transformer was trained on LINCS (perturbation x time -> per-gene response). This is the real proof:
query it at the EGF course's minute-timepoints (GSE6783, 20-240 min) and compare its predicted TEMPORAL
PROFILE to the observed one, vs our best prior (GBM 0.25 / mechanistic 0.29). EGF is a LINCS ligand, so this
tests interpolation to a FINER timescale than training (LINCS EGF is sampled in hours).

Metric is scale-invariant fraction-of-change-complete (LINCS Level-5 is a moderated z-score, not LFC, so we
compare SHAPE not magnitude): frac(t) = response(t)/response(t_last). Self-contained model rebuild from the
checkpoint (dim/layers/n_pert/n_cell) + lincs_train.npz (name->index maps, dose/time normalization).
-> eval_transformer_egf.json
"""
import os, json, csv, math
from pathlib import Path
from collections import Counter
import numpy as np
OUT=Path("outputs/orphan"); LN2=math.log(2)
TIMES_MIN=[20,40,60,120,240,480]; MID=[20,40,60,120,240]

def _hl(name):
    m={}; f=OUT/name
    if f.exists():
        for r in csv.DictReader(open(f),delimiter="\t"):
            try: m[r["gene"]]=math.log10(float(r["half_life_h"]))
            except: pass
    return m

def main():
    ck=OUT/"dynamics_transformer.pt"; npz=OUT/"lincs_train.npz"; egf=OUT/"timecourse_egf.tsv"
    if not (ck.exists() and npz.exists() and egf.exists()):
        print("eval needs dynamics_transformer.pt + lincs_train.npz + timecourse_egf.tsv "
              "(train the transformer + fetch_timecourse first)"); return
    try: import torch, torch.nn as nn
    except Exception as e: print("needs torch:",repr(e)[:80]); return
    dev="cuda" if torch.cuda.is_available() else "cpu"
    C=torch.load(ck, map_location=dev, weights_only=False)
    DIM=C["dim"]; LAY=C["layers"]; n_pert=C["n_pert"]; n_cell=C["n_cell"]; genes=list(C["genes"]); n_gene=len(genes)
    D=np.load(npz, allow_pickle=True)
    pert_names=list(D["pert_names"]); cell_names=list(D["cell_names"]); pert_of_row=D["pert_of_row"]
    dose=D["dose"]; time_h=D["time_h"]; cell_row=D["cell"]
    # same normalization the model trained with (recomputed from the npz)
    dmu,dsd=np.nanmean(dose),np.nanstd(dose)+1e-6; tmu,tsd=np.nanmean(time_h),np.nanstd(time_h)+1e-6
    # physics features for the checkpoint's gene order
    mrna=_hl("mrna_halflife.tsv"); prot=_hl("protein_halflife.tsv")
    mm=np.array([mrna.get(g,np.nan) for g in genes]); pp=np.array([prot.get(g,np.nan) for g in genes])
    for a in (mm,pp): a[np.isnan(a)]=np.nanmedian(a) if np.isfinite(np.nanmedian(a)) else 0.0
    gfeat=torch.tensor(np.stack([mm,pp],1),dtype=torch.float32,device=dev)
    class Model(nn.Module):
        def __init__(s):
            super().__init__()
            s.gid=nn.Embedding(n_gene,DIM); s.gp=nn.Linear(2,DIM)
            s.pe=nn.Embedding(n_pert,DIM); s.ce=nn.Embedding(n_cell,DIM); s.cx=nn.Linear(2,DIM)
            s.film=nn.Linear(DIM*3,DIM*2)
            enc=nn.TransformerEncoderLayer(DIM,4,DIM*2,0.1,batch_first=True)
            s.tr=nn.TransformerEncoder(enc,LAY); s.head=nn.Linear(DIM,1); s.gidx=torch.arange(n_gene,device=dev)
        def forward(s,p,c,x):
            B=p.shape[0]; g=(s.gid(s.gidx)+s.gp(gfeat)).unsqueeze(0).expand(B,-1,-1)
            gamma,beta=s.film(torch.cat([s.pe(p),s.ce(c),s.cx(x)],-1)).chunk(2,-1)
            g=g*(1+torch.tanh(gamma.unsqueeze(1)))+beta.unsqueeze(1); return s.head(s.tr(g)).squeeze(-1)
    model=Model().to(dev); model.load_state_dict(C["state"]); model.eval()
    # locate EGF + its most common cell line in LINCS
    if "EGF" not in pert_names: print("EGF not among LINCS perturbations -> cannot run this transfer test"); return
    pidx=pert_names.index("EGF")
    egf_cells=[int(c) for c,p in zip(cell_row,pert_of_row) if p=="EGF"]
    cidx=Counter(egf_cells).most_common(1)[0][0] if egf_cells else 0
    egf_dose=np.nanmedian([d for d,p in zip(dose,pert_of_row) if p=="EGF"]) if any(pert_of_row=="EGF") else dmu
    # query the model along the EGF timepoints (minutes -> hours)
    def predict(t_h):
        x=np.array([[ (egf_dose-dmu)/dsd, (t_h-tmu)/tsd ]],dtype=np.float32)
        with torch.no_grad():
            z=model(torch.tensor([pidx],device=dev),torch.tensor([cidx],device=dev),torch.tensor(x,device=dev))
        return z[0].cpu().numpy()                                   # predicted response per gene at time t
    zt={t: predict(t/60.0) for t in TIMES_MIN}
    gpos={g:i for i,g in enumerate(genes)}
    # observed EGF course (landmark genes only)
    obs={}
    for r in csv.DictReader(open(egf),delimiter="\t"):
        g=r["gene"]
        if g not in gpos: continue
        try: obs[g]=[float(r[f"t{t}"]) for t in [0]+TIMES_MIN]
        except: pass
    def frac_complete(tau_min,t): return 1.0-math.exp(-t/tau_min)
    e_tf=[]; e_mech=[]; e_step=[]; n=0
    for g,vals in obs.items():
        R0,Rf=vals[0],vals[-1]
        if R0<=0 or Rf<=0 or abs(math.log2(Rf/R0))<1.0: continue
        fr=[(t,(R-R0)/(Rf-R0)) for t,R in zip([0]+TIMES_MIN,vals) if t in MID]
        if any(fo<-0.25 or fo>1.25 for _,fo in fr): continue
        i=gpos[g]; zfin=zt[480][i]
        if abs(zfin)<1e-6: continue
        n+=1; hl=mrna.get(g); mech_tau=(10**hl)*60/LN2 if hl else None
        for t,fo in fr:
            fp=zt[t][i]/zfin                                       # transformer temporal profile (scale-invariant)
            e_tf.append(abs(fp-fo)); e_step.append(abs(1.0-fo))
            if mech_tau: e_mech.append(abs(frac_complete(mech_tau,t)-fo))
    def md(x): return round(float(np.median(x)),4) if x else None
    res=dict(n_genes=n, egf_pert_index=pidx, egf_cell=cell_names[cidx] if cidx<len(cell_names) else "?",
             median_abs_fraction_error=dict(transformer=md(e_tf), mechanistic_halflife=md(e_mech),
                 instant_step=md(e_step)),
             prior_gbm_baseline=0.25, oracle_floor=0.17)
    json.dump(res, open(OUT/"eval_transformer_egf.json","w"))
    e=res["median_abs_fraction_error"]
    print(f"STAGE-1 EGF transfer test on {n} genes | EGF@{res['egf_cell']}")
    print(f"  median |fraction error|: TRANSFORMER {e['transformer']} | mechanistic {e['mechanistic_halflife']} | "
          f"instant-step {e['instant_step']} | (prior GBM 0.25, oracle 0.17)")
    if e['transformer'] is not None:
        v=("BEATS the 0.25 prior — LINCS dynamics transfer to fine timescales!" if e['transformer']<0.25
           else "does NOT beat 0.25 -> the hours->minutes timescale gap is real (go to Stage 2: dense time-course data)")
        print(f"  >>> VERDICT: {v}")

if __name__=="__main__":
    main()
