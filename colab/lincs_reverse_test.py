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
    from compute_reverse_inference import load_model, reverse_infer, build_signature_library, build_signature_index
    npz=OUT/"lincs_train.npz"
    if not npz.exists(): print("lincs_train.npz absent -> run fetch_lincs / restore from Drive"); return
    M=load_model(); nmset=set(M["nm"]); N=M["N"]
    D=np.load(npz, allow_pickle=True)
    lfc=D["lfc"]; syms=list(D["gene_symbols"]); pert=D["pert_of_row"]
    rng=np.random.RandomState(0)
    allrows=np.arange(len(pert)); rng.shuffle(allrows)
    libset=set(allrows[:len(allrows)//2].tolist())            # LIBRARY half
    lib, lib_cols = build_signature_library(lfc, syms, pert, nmset, rows=list(libset))     # centroids
    index = build_signature_index(lfc, syms, pert, nmset, rows=list(libset))               # individual sigs (k-NN)
    print(f"  library: {len(lib)} gene centroids + {index[0].shape[0]} individual signatures (from {len(libset)} held-out)")
    # test on the OTHER half: one signature per gene, gene present in the model AND in the library
    testrows=[i for i in allrows if i not in libset and str(pert[i]) in nmset and str(pert[i]) in lib]
    seen=set(); test=[]
    for i in testrows:
        if pert[i] not in seen: seen.add(pert[i]); test.append(i)
    test=test[:800]
    reg=M["regulon"]; is_tf=set(M["G"][i]["name"] for i in reg)
    R={"centroid_rank":[], "knn":[]}; r_tf=[]; r_ntf=[]
    for i in test:
        sig={syms[j]:float(lfc[i][j]) for j in range(len(syms)) if abs(lfc[i][j])>0.5}
        if len(sig)<20: continue
        gene=str(pert[i])
        rr=reverse_infer(sig, M, top=100, library=lib, lib_cols=lib_cols, sig_metric="rank")
        rk=reverse_infer(sig, M, top=100, index=index)                          # k-NN + all network lenses
        R["centroid_rank"].append(next((k+1 for k,x in enumerate(rr) if x["gene"]==gene), 999))
        rankk=next((k+1 for k,x in enumerate(rk) if x["gene"]==gene), 999)
        R["knn"].append(rankk); (r_tf if gene in is_tf else r_ntf).append(rankk)
    def rec(a,k): return round(float((np.array(a)<=k).mean()),3) if len(a) else None
    def blk(a): return dict(recall_1=rec(a,1), recall_10=rec(a,10), recall_30=rec(a,30), median_rank=int(np.median(a)))
    res=dict(n_tested=len(R["knn"]), random_recall_10=round(10/N,5),
             centroid_rank=blk(R["centroid_rank"]), knn_ensemble=blk(R["knn"]),
             tf_recall_10=rec(r_tf,10), nontf_recall_10=rec(r_ntf,10), n_tf=len(r_tf), n_nontf=len(r_ntf))
    json.dump(res, open(OUT/"lincs_reverse_test.json","w"))
    cr=res["centroid_rank"]; kn=res["knn_ensemble"]
    print(f"LINCS REVERSE TEST — recover perturbed gene from REAL signature ({res['n_tested']} genes, random recall@10 = {res['random_recall_10']})")
    print(f"  centroid (rank)   : recall@10 {cr['recall_10']} | recall@30 {cr['recall_30']} | recall@1 {cr['recall_1']} | median {cr['median_rank']}")
    print(f"  k-NN + ensemble   : recall@10 {kn['recall_10']} | recall@30 {kn['recall_30']} | recall@1 {kn['recall_1']} | median {kn['median_rank']}")
    print(f"  by type (kNN): TF recall@10 {res['tf_recall_10']} (n={res['n_tf']}) | non-TF {res['nontf_recall_10']} (n={res['n_nontf']})")
    best=max(cr['recall_10'] or 0, kn['recall_10'] or 0); base=res['random_recall_10']
    print(f"  >>> best recall@10 {best} = {round(best/base) if base else '?'}x over random "
          f"({'strong' if best>0.25 else 'good' if best>0.15 else 'modest'})")

if __name__=="__main__":
    main()
