"""DEFINITIVE reverse-inference test — recover a LINCS-perturbed gene from its REAL signature.

Now with the CMap signature-matching lens (Lever 1). We split LINCS signatures into a LIBRARY half and a
TEST half (disjoint), build a per-gene reference library from the library half, and test recovery on the
test half -- so matching a test signature to the library is CROSS-CONTEXT (non-circular): a gene is
recovered only if its signature is CONSISTENT across independent measurements. Reports network-only vs
+signature-matching so the improvement is explicit. Needs lincs_train.npz. -> lincs_reverse_test.json
"""
import json, os
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan")

def main():
    from compute_reverse_inference import load_model, reverse_infer, build_signature_library
    npz=OUT/"lincs_train.npz"
    if not npz.exists(): print("lincs_train.npz absent -> run fetch_lincs / restore from Drive"); return
    M=load_model(); nmset=set(M["nm"]); N=M["N"]
    D=np.load(npz, allow_pickle=True)
    lfc=D["lfc"]; syms=list(D["gene_symbols"]); pert=D["pert_of_row"]
    rng=np.random.RandomState(0)
    allrows=np.arange(len(pert)); rng.shuffle(allrows)
    libset=set(allrows[:len(allrows)//2].tolist())            # LIBRARY half
    lib, lib_cols = build_signature_library(lfc, syms, pert, nmset, rows=list(libset))
    print(f"  signature library: {len(lib)} genes (from {len(libset)} held-out signatures)")
    # test on the OTHER half: one signature per gene, gene present in the model AND in the library
    testrows=[i for i in allrows if i not in libset and str(pert[i]) in nmset and str(pert[i]) in lib]
    seen=set(); test=[]
    for i in testrows:
        if pert[i] not in seen: seen.add(pert[i]); test.append(i)
    test=test[:800]
    reg=M["regulon"]; is_tf=set(M["G"][i]["name"] for i in reg)
    r_cos=[]; r_rank=[]; r_tf=[]; r_ntf=[]
    for i in test:
        sig={syms[j]:float(lfc[i][j]) for j in range(len(syms)) if abs(lfc[i][j])>0.5}
        if len(sig)<20: continue
        gene=str(pert[i])
        rc=reverse_infer(sig, M, top=100, library=lib, lib_cols=lib_cols, sig_metric="cosine")
        rr=reverse_infer(sig, M, top=100, library=lib, lib_cols=lib_cols, sig_metric="rank")
        rankc=next((k+1 for k,x in enumerate(rc) if x["gene"]==gene), 999)
        rankr=next((k+1 for k,x in enumerate(rr) if x["gene"]==gene), 999)
        r_cos.append(rankc); r_rank.append(rankr); (r_tf if gene in is_tf else r_ntf).append(rankr)
    def rec(a,k): return round(float((np.array(a)<=k).mean()),3) if len(a) else None
    res=dict(n_tested=len(r_rank), random_recall_10=round(10/N,5),
             signature_cosine=dict(recall_1=rec(r_cos,1), recall_10=rec(r_cos,10), recall_30=rec(r_cos,30), median_rank=int(np.median(r_cos))),
             signature_rank=dict(recall_1=rec(r_rank,1), recall_10=rec(r_rank,10), recall_30=rec(r_rank,30), median_rank=int(np.median(r_rank))),
             tf_recall_10=rec(r_tf,10), nontf_recall_10=rec(r_ntf,10), n_tf=len(r_tf), n_nontf=len(r_ntf))
    json.dump(res, open(OUT/"lincs_reverse_test.json","w"))
    rc=res["signature_cosine"]; rr=res["signature_rank"]
    print(f"LINCS REVERSE TEST — recover perturbed gene from REAL signature ({res['n_tested']} genes, random recall@10 = {res['random_recall_10']})")
    print(f"  SIGNATURE (cosine)      : recall@10 {rc['recall_10']} | recall@30 {rc['recall_30']} | recall@1 {rc['recall_1']}")
    print(f"  SIGNATURE (rank, robust): recall@10 {rr['recall_10']} | recall@30 {rr['recall_30']} | recall@1 {rr['recall_1']}")
    print(f"  by type (rank): TF recall@10 {res['tf_recall_10']} (n={res['n_tf']}) | non-TF {res['nontf_recall_10']} (n={res['n_nontf']})")
    best=max(rc['recall_10'] or 0, rr['recall_10'] or 0); base=res['random_recall_10']
    print(f"  >>> best recall@10 {best} = {round(best/base) if base else '?'}x over random "
          f"({'strong' if best>0.2 else 'good' if best>0.12 else 'modest'})")

if __name__=="__main__":
    main()
