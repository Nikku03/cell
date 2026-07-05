"""DYNAMICS TRANSFORMER — learn perturbation x time -> per-gene response, physics-informed.

A compact gene-transformer: 978 landmark genes are tokens (gene embedding + our PHYSICS features: mRNA/protein
half-life), modulated by a context (perturbation embedding + cell-line embedding + dose + TIME) via FiLM, then
a few self-attention layers let genes co-regulate, and a head predicts each gene's log-fold-change. Trained on
LINCS (lincs_train.npz), held out BY PERTURBATION so the test measures generalization to unseen perturbagens.

Time is a conditioning input (LINCS gives 6 h & 24 h), so the trained model is queryable at any t -- the basis
for beating the mechanistic 0.29 on dense external time courses (next step). Saves a checkpoint for reuse.
Env: DTF_EPOCHS, DTF_DIM, DTF_LAYERS, DTF_BATCH, DTF_HOLDOUT_FRAC. Skips gracefully without torch/data.
-> dynamics_transformer.pt (checkpoint) + dynamics_transformer_eval.json (held-out metrics vs baselines)
"""
import os, json, csv, math
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan")

def _halflives():
    m={}
    f=OUT/"mrna_halflife.tsv"
    if f.exists():
        for r in csv.DictReader(open(f),delimiter="\t"):
            try: m.setdefault(r["gene"],{})["mrna"]=math.log10(float(r["half_life_h"]))
            except: pass
    f=OUT/"protein_halflife.tsv"
    if f.exists():
        for r in csv.DictReader(open(f),delimiter="\t"):
            try: m.setdefault(r["gene"],{})["prot"]=math.log10(float(r["half_life_h"]))
            except: pass
    return m

def main():
    npz=OUT/"lincs_train.npz"
    if not npz.exists(): print("dynamics-transformer: lincs_train.npz absent -> run fetch_lincs first"); return
    try:
        import torch, torch.nn as nn
    except Exception as e:
        print("dynamics-transformer needs PyTorch (GPU runtime):",repr(e)[:80]); return
    dev="cuda" if torch.cuda.is_available() else "cpu"
    D=np.load(npz, allow_pickle=True)
    lfc=D["lfc"]; pert=D["pert"].astype(np.int64); cell=D["cell"].astype(np.int64)
    dose=D["dose"]; time_h=D["time_h"]; syms=list(D["gene_symbols"]); pert_of_row=D["pert_of_row"]
    n_sig, n_gene = lfc.shape
    n_pert=int(pert.max())+1; n_cell=int(cell.max())+1
    # per-gene physics features (log half-lives, median-imputed) -> [n_gene, 2]
    hl=_halflives(); mm=np.array([hl.get(s,{}).get("mrna",np.nan) for s in syms])
    pp=np.array([hl.get(s,{}).get("prot",np.nan) for s in syms])
    for a in (mm,pp): a[np.isnan(a)]=np.nanmedian(a) if np.isfinite(np.nanmedian(a)) else 0.0
    gfeat=torch.tensor(np.stack([mm,pp],1),dtype=torch.float32,device=dev)
    # normalize dose/time
    dz=(dose-np.nanmean(dose))/(np.nanstd(dose)+1e-6); tz=(time_h-np.nanmean(time_h))/(np.nanstd(time_h)+1e-6)
    ctx=np.stack([np.nan_to_num(dz), np.nan_to_num(tz)],1).astype(np.float32)
    # held-out split BY PERTURBATION
    hold=float(os.environ.get("DTF_HOLDOUT_FRAC","0.15"))
    uperts=sorted(set(pert_of_row.tolist())); rng=np.random.RandomState(0); rng.shuffle(uperts)
    test_perts=set(uperts[:max(1,int(len(uperts)*hold))])
    is_test=np.array([p in test_perts for p in pert_of_row])
    tr=np.where(~is_test)[0]; te=np.where(is_test)[0]
    print(f"  {n_sig} sigs, {n_gene} genes, {n_pert} perts, {n_cell} cells | train {len(tr)} / test {len(te)} (held-out perts) | dev={dev}")

    DIM=int(os.environ.get("DTF_DIM","128")); LAY=int(os.environ.get("DTF_LAYERS","3"))
    class Model(nn.Module):
        def __init__(s):
            super().__init__()
            s.gid=nn.Embedding(n_gene,DIM); s.gp=nn.Linear(2,DIM)
            s.pe=nn.Embedding(n_pert,DIM); s.ce=nn.Embedding(n_cell,DIM); s.cx=nn.Linear(2,DIM)
            s.film=nn.Linear(DIM*3,DIM*2)
            enc=nn.TransformerEncoderLayer(DIM,4,DIM*2,0.1,batch_first=True)
            s.tr=nn.TransformerEncoder(enc,LAY); s.head=nn.Linear(DIM,1)
            s.gidx=torch.arange(n_gene,device=dev)
        def forward(s, p,c,x):
            B=p.shape[0]
            g=s.gid(s.gidx)+s.gp(gfeat)                     # [n_gene,DIM] shared gene tokens
            g=g.unsqueeze(0).expand(B,-1,-1)                # [B,n_gene,DIM]
            ctxv=torch.cat([s.pe(p),s.ce(c),s.cx(x)],-1)    # [B,3*DIM]
            gamma,beta=s.film(ctxv).chunk(2,-1)             # FiLM condition on perturbation+cell+dose+time
            g=g*(1+gamma.unsqueeze(1))+beta.unsqueeze(1)
            return s.head(s.tr(g)).squeeze(-1)              # [B,n_gene] predicted LFC
    model=Model().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    lossf=nn.MSELoss()
    Y=torch.tensor(lfc,device=dev); P=torch.tensor(pert,device=dev); C=torch.tensor(cell,device=dev); Xc=torch.tensor(ctx,device=dev)
    EP=int(os.environ.get("DTF_EPOCHS","12")); BS=int(os.environ.get("DTF_BATCH","256"))
    import time as _t; amp=(dev=="cuda")                          # bf16 autocast on GPU -> ~2-3x faster on L4
    for ep in range(EP):
        model.train(); perm=np.random.permutation(tr); tot=0.0; t0=_t.time()
        for i in range(0,len(perm),BS):
            b=perm[i:i+BS]; bi=torch.tensor(b,device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                pred=model(P[bi],C[bi],Xc[bi]); loss=lossf(pred,Y[bi])
            opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()*len(b)
        print(f"  epoch {ep+1}/{EP}  train MSE {tot/len(perm):.4f}  ({_t.time()-t0:.0f}s)", flush=True)
    # ---- held-out eval: per-signature Pearson r (pred vs true across genes) + MSE, vs baselines ----
    model.eval()
    gene_mean=torch.tensor(lfc[tr].mean(0),device=dev)                # per-gene bias baseline (from train)
    def pear(a,b):
        a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/(a.norm()*b.norm()+1e-8))
    r_model=[]; r_mean=[]; mse_model=[]; mse_zero=[]; mse_mean=[]
    with torch.no_grad():
        for i in range(0,len(te),BS):
            b=te[i:i+BS]; bi=torch.tensor(b,device=dev)
            pred=model(P[bi],C[bi],Xc[bi]); true=Y[bi]
            for j in range(len(b)):
                r_model.append(pear(pred[j],true[j])); r_mean.append(pear(gene_mean,true[j]))
                mse_model.append(float(((pred[j]-true[j])**2).mean())); mse_zero.append(float((true[j]**2).mean()))
                mse_mean.append(float(((gene_mean-true[j])**2).mean()))
    res=dict(n_test_sigs=len(te), n_train_sigs=len(tr), n_perts=n_pert, n_cells=n_cell, device=dev,
             median_pearson_r=dict(model=round(float(np.median(r_model)),4), gene_mean_baseline=round(float(np.median(r_mean)),4)),
             mean_mse=dict(model=round(float(np.mean(mse_model)),4), predict_zero=round(float(np.mean(mse_zero)),4),
                           gene_mean_baseline=round(float(np.mean(mse_mean)),4)))
    json.dump(res, open(OUT/"dynamics_transformer_eval.json","w"))
    torch.save({"state":model.state_dict(),"n_pert":n_pert,"n_cell":n_cell,"dim":DIM,"layers":LAY,
                "genes":syms}, OUT/"dynamics_transformer.pt")
    r=res["median_pearson_r"]; m=res["mean_mse"]
    print(f"dynamics-transformer: held-out (unseen perturbations) median gene-wise Pearson r = {r['model']} "
          f"(gene-mean baseline {r['gene_mean_baseline']})")
    print(f"  MSE model {m['model']} vs predict-zero {m['predict_zero']}, gene-mean {m['gene_mean_baseline']} "
          f"-> {'LEARNS perturbation-specific response' if m['model']<m['gene_mean_baseline']-0.001 else 'not beating gene-mean (needs more epochs/data)'}")
    print(f"  checkpoint -> dynamics_transformer.pt (query at any time t for the trajectory)")

if __name__=="__main__":
    main()
