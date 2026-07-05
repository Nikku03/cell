"""DEFINITIVE reverse-inference test — recover a LINCS-knocked-down gene from its REAL signature.

The honest, non-circular validation: LINCS shRNA (trt_sh) and overexpression (trt_oe) signatures are REAL
experimental readouts of "gene X was perturbed -> the cell's response". We feed only the signature (978
landmark gene changes) to reverse_infer and ask: does it recover X as the causal source? The signature is
independent experimental data; the inference uses only the model's network. Recall@K vs the 1/N random rate.
Needs lincs_train.npz (from fetch_lincs / Drive). -> lincs_reverse_test.json
"""
import json, os
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan")

def main():
    from compute_reverse_inference import load_model, reverse_infer
    npz=OUT/"lincs_train.npz"
    if not npz.exists(): print("lincs_train.npz absent -> run fetch_lincs / restore from Drive"); return
    M=load_model(); nmset=set(M["nm"]); N=M["N"]
    D=np.load(npz, allow_pickle=True)
    lfc=D["lfc"]; syms=list(D["gene_symbols"]); pert=D["pert_of_row"]
    # keep signatures whose perturbed gene (shRNA/oe target = the pert name) is a node in the model
    idxs=[i for i in range(len(pert)) if str(pert[i]) in nmset]
    if len(idxs)>2000:                                          # cap for runtime; one signature per perturbed gene
        seen=set(); keep=[]
        for i in idxs:
            if pert[i] not in seen: seen.add(pert[i]); keep.append(i)
        idxs=keep[:800]
    ranks=[]; ranks_tf=[]; ranks_ntf=[]
    reg=M["regulon"]; is_tf=set(M["G"][i]["name"] for i in reg)
    for c,i in enumerate(idxs):
        sig={syms[j]:float(lfc[i][j]) for j in range(len(syms)) if abs(lfc[i][j])>0.5}   # differential genes
        if len(sig)<20: continue
        res=reverse_infer(sig, M, top=100)
        gene=str(pert[i]); r=next((k+1 for k,x in enumerate(res) if x["gene"]==gene), 999)
        ranks.append(r); (ranks_tf if gene in is_tf else ranks_ntf).append(r)
    ranks=np.array(ranks)
    def rec(a,k): return round(float((np.array(a)<=k).mean()),3) if len(a) else None
    res=dict(n_tested=len(ranks), random_recall_10=round(100/N,4),
             recall_1=rec(ranks,1), recall_10=rec(ranks,10), recall_30=rec(ranks,30),
             median_rank=int(np.median(ranks)) if len(ranks) else None,
             tf_recall_10=rec(ranks_tf,10), nontf_recall_10=rec(ranks_ntf,10),
             n_tf=len(ranks_tf), n_nontf=len(ranks_ntf))
    json.dump(res, open(OUT/"lincs_reverse_test.json","w"))
    print(f"LINCS REVERSE TEST — recover the perturbed gene from its real signature ({res['n_tested']} genes)")
    print(f"  recall@1 {res['recall_1']} | recall@10 {res['recall_10']} | recall@30 {res['recall_30']} "
          f"| median rank {res['median_rank']} (random recall@10 = {res['random_recall_10']})")
    print(f"  by type: TF recall@10 {res['tf_recall_10']} (n={res['n_tf']}) | non-TF {res['nontf_recall_10']} (n={res['n_nontf']})")
    v=res['recall_10'] or 0
    print(f"  >>> {'RECOVERS THE CAUSE from real signatures — %dx over random'%round(v/(100/N)) if v>3*(100/N) else 'weak recovery on real data — needs a different inference strategy'}")

if __name__=="__main__":
    main()
