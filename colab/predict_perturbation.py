"""PREDICT PERTURBATION RESPONSE — the LINCS dynamics transformer, repurposed for what it's good at.

The transformer learned perturbation x time x cell -> per-gene response from LINCS (in-distribution r~0.18,
beats gene-mean). It does NOT do minute-scale trajectories (proven: the EGF transfer failed), but it DOES
give a useful empirical answer to "what does compound / gene-perturbation X do to the cell at ~hours?" --
complementing the mechanistic knockout-cascade route (which is network-topology-based, not data-driven).

  predict("vorinostat")            -> top up/down genes for that drug (default cell/time)
  predict("trichostatin-a","MCF7",24)
CLI: python predict_perturbation.py <pert_name> [cell] [time_h]
Needs dynamics_transformer.pt + lincs_train.npz + mrna/protein half-lives (same physics features as training).
"""
import os, sys, math, csv
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan"); LN2=math.log(2)

def _hl(name):
    m={}; f=OUT/name
    if f.exists():
        for r in csv.DictReader(open(f),delimiter="\t"):
            try: m[r["gene"]]=math.log10(float(r["half_life_h"]))
            except: pass
    return m

_CACHE={}
def _load():
    if _CACHE: return _CACHE
    import torch, torch.nn as nn
    dev="cuda" if torch.cuda.is_available() else "cpu"
    C=torch.load(OUT/"dynamics_transformer.pt", map_location=dev, weights_only=False)
    DIM=C["dim"]; LAY=C["layers"]; n_pert=C["n_pert"]; n_cell=C["n_cell"]; genes=list(C["genes"]); n_gene=len(genes)
    D=np.load(OUT/"lincs_train.npz", allow_pickle=True)
    pert_names=list(D["pert_names"]); cell_names=list(D["cell_names"]); pert_of_row=D["pert_of_row"]
    dose=D["dose"]; time_h=D["time_h"]; cell_row=D["cell"]
    dmu,dsd=float(np.nanmean(dose)),float(np.nanstd(dose)+1e-6); tmu,tsd=float(np.nanmean(time_h)),float(np.nanstd(time_h)+1e-6)
    mrna=_hl("mrna_halflife.tsv"); prot=_hl("protein_halflife.tsv")
    mm=np.array([mrna.get(g,np.nan) for g in genes]); pp=np.array([prot.get(g,np.nan) for g in genes])
    for a in (mm,pp): a[np.isnan(a)]=np.nanmedian(a) if np.isfinite(np.nanmedian(a)) else 0.0
    gfeat=torch.tensor(np.stack([mm,pp],1),dtype=torch.float32,device=dev)
    class Model(nn.Module):
        def __init__(s):
            super().__init__()
            s.gid=nn.Embedding(n_gene,DIM); s.gp=nn.Linear(2,DIM); s.pe=nn.Embedding(n_pert,DIM)
            s.ce=nn.Embedding(n_cell,DIM); s.cx=nn.Linear(2,DIM); s.film=nn.Linear(DIM*3,DIM*2)
            enc=nn.TransformerEncoderLayer(DIM,4,DIM*2,0.1,batch_first=True)
            s.tr=nn.TransformerEncoder(enc,LAY); s.head=nn.Linear(DIM,1); s.gidx=torch.arange(n_gene,device=dev)
        def forward(s,p,c,x):
            B=p.shape[0]; g=(s.gid(s.gidx)+s.gp(gfeat)).unsqueeze(0).expand(B,-1,-1)
            gamma,beta=s.film(torch.cat([s.pe(p),s.ce(c),s.cx(x)],-1)).chunk(2,-1)
            g=g*(1+torch.tanh(gamma.unsqueeze(1)))+beta.unsqueeze(1); return s.head(s.tr(g)).squeeze(-1)
    model=Model().to(dev); model.load_state_dict(C["state"]); model.eval()
    _CACHE.update(dict(torch=torch, model=model, dev=dev, genes=genes, pert_names=pert_names,
        cell_names=cell_names, pert_of_row=pert_of_row, cell_row=cell_row, dose=dose,
        dmu=dmu,dsd=dsd,tmu=tmu,tsd=tsd))
    return _CACHE

def predict(pert, cell=None, time_h=24.0, top=15):
    if not (OUT/"dynamics_transformer.pt").exists():
        return {"error":"dynamics_transformer.pt absent — train it (transformer_dynamics.ipynb) / restore from Drive"}
    S=_load(); torch=S["torch"]
    pn=S["pert_names"]
    if pert not in pn:
        from difflib import get_close_matches
        near=get_close_matches(pert, pn, n=6, cutoff=0.6)
        return {"error":f"'{pert}' not in LINCS perturbations", "did_you_mean":near}
    pidx=pn.index(pert)
    from collections import Counter
    rows_for=[int(c) for c,p in zip(S["cell_row"],S["pert_of_row"]) if p==pert]
    if cell and cell in S["cell_names"]: cidx=S["cell_names"].index(cell)
    else: cidx=Counter(rows_for).most_common(1)[0][0] if rows_for else 0
    egf_dose=float(np.nanmedian([d for d,p in zip(S["dose"],S["pert_of_row"]) if p==pert])) if rows_for else S["dmu"]
    x=np.array([[(egf_dose-S["dmu"])/S["dsd"], (time_h-S["tmu"])/S["tsd"]]],dtype=np.float32)
    with torch.no_grad():
        z=S["model"](torch.tensor([pidx],device=S["dev"]),torch.tensor([cidx],device=S["dev"]),
                     torch.tensor(x,device=S["dev"]))[0].cpu().numpy()
    order=np.argsort(z); genes=S["genes"]
    up=[(genes[i],round(float(z[i]),2)) for i in order[::-1][:top]]
    dn=[(genes[i],round(float(z[i]),2)) for i in order[:top]]
    return dict(perturbation=pert, cell=S["cell_names"][cidx], time_h=time_h,
                up_genes=up, down_genes=dn, n_cells_measured=len(set(rows_for)))

def main():
    if len(sys.argv)<2:
        print("usage: python predict_perturbation.py <pert_name> [cell] [time_h]"); return
    pert=sys.argv[1]; cell=sys.argv[2] if len(sys.argv)>2 else None
    time_h=float(sys.argv[3]) if len(sys.argv)>3 else 24.0
    r=predict(pert, cell, time_h)
    if "error" in r:
        print(r["error"]);
        if r.get("did_you_mean"): print("  did you mean:", ", ".join(r["did_you_mean"]))
        return
    print(f"PREDICTED response to {r['perturbation']} in {r['cell']} at {r['time_h']}h "
          f"(learned from LINCS, {r['n_cells_measured']} cell lines):")
    print("  UP  :", ", ".join(f"{g}({v:+})" for g,v in r["up_genes"]))
    print("  DOWN:", ", ".join(f"{g}({v:+})" for g,v in r["down_genes"]))
    print("  [empirical LINCS-learned signature at ~hours; complements the network knockout-cascade route]")

if __name__=="__main__":
    main()
