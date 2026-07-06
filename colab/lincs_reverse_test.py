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
    r_net=[]; r_full=[]; r_tf=[]; r_ntf=[]
    for i in test:
        sig={syms[j]:float(lfc[i][j]) for j in range(len(syms)) if abs(lfc[i][j])>0.5}
        if len(sig)<20: continue
        gene=str(pert[i])
        res_net=reverse_infer(sig, M, top=100)                                 # network-only
        res_full=reverse_infer(sig, M, top=100, library=lib, lib_cols=lib_cols) # + signature matching
        rn=next((k+1 for k,x in enumerate(res_net) if x["gene"]==gene), 999)
        rf=next((k+1 for k,x in enumerate(res_full) if x["gene"]==gene), 999)
        r_net.append(rn); r_full.append(rf); (r_tf if gene in is_tf else r_ntf).append(rf)
    def rec(a,k): return round(float((np.array(a)<=k).mean()),3) if len(a) else None
    res=dict(n_tested=len(r_full), random_recall_10=round(10/N,5),
             network_only=dict(recall_1=rec(r_net,1), recall_10=rec(r_net,10), recall_30=rec(r_net,30), median_rank=int(np.median(r_net))),
             with_signature_matching=dict(recall_1=rec(r_full,1), recall_10=rec(r_full,10), recall_30=rec(r_full,30), median_rank=int(np.median(r_full))),
             tf_recall_10=rec(r_tf,10), nontf_recall_10=rec(r_ntf,10), n_tf=len(r_tf), n_nontf=len(r_ntf))
    json.dump(res, open(OUT/"lincs_reverse_test.json","w"))
    rn=res["network_only"]; rf=res["with_signature_matching"]
    print(f"LINCS REVERSE TEST — recover perturbed gene from REAL signature ({res['n_tested']} genes, random recall@10 = {res['random_recall_10']})")
    print(f"  NETWORK ONLY        : recall@10 {rn['recall_10']} | recall@30 {rn['recall_30']} | median rank {rn['median_rank']}")
    print(f"  + SIGNATURE MATCHING: recall@10 {rf['recall_10']} | recall@30 {rf['recall_30']} | median rank {rf['median_rank']}")
    print(f"  by type (full): TF recall@10 {res['tf_recall_10']} (n={res['n_tf']}) | non-TF {res['nontf_recall_10']} (n={res['n_nontf']})")
    v=rf['recall_10'] or 0; base=res['random_recall_10']
    print(f"  >>> signature matching: {round(v/base) if base else '?'}x over random "
          f"({'strong recovery' if v>0.15 else 'improved but still modest' if v>0.05 else 'still weak — need cleaner perturbation data / learned retrieval'})")

if __name__=="__main__":
    main()
