"""VALIDATE reverse inference — can it recover the SOURCE from the phenotype, honestly?

Two honesty controls beyond the trivial self-check:
  (A) NOISE + TRUNCATION robustness: build a seed's signature via the signed cascade, add Gaussian noise and
      keep only the top-K differentially-'expressed' genes (like a real experiment), then recover the seed.
      Report recall@1/5/10 and median rank across many seeds, vs the 1/N random baseline.
  (B) PER-LENS recovery: how well does EACH lens alone (master-reg / diffusion / cascade) recover the seed?
      The diffusion lens rides on PPI+co-essentiality, which are INDEPENDENT of the regulatory cascade that
      generated the signature -- so if diffusion recovers the seed, that is non-circular evidence the method
      finds the cause, not just its own inverse.
The definitive real-data test (recover a LINCS-knocked-down gene from its signature) is lincs_reverse_test.py
(run on Colab where LINCS lives). -> reverse_validation.json
"""
import json, math, os
from collections import defaultdict
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan")

def main():
    from compute_reverse_inference import (load_model, reverse_infer, master_regulators, diffuse,
                                            cascade_match, codep_coherence, _sig_array, _z)
    M=load_model(); G=M["G"]; reg=M["regulon"]; N=M["N"]
    seeds=[i for i in reg if len(reg[i])>=25]
    rng=np.random.RandomState(0); rng.shuffle(seeds); seeds=seeds[:60]
    def make_sig(seed, noise, keepK):
        sig=defaultdict(float)
        for t,sg in reg[seed]:
            sig[t]+=sg*2.0
            for u,sg2 in reg.get(t,[])[:10]: sig[u]+=0.5*sg*sg2
        arr=np.zeros(N)
        for t,v in sig.items(): arr[t]=v
        arr=arr+rng.normal(0,noise,N)*(np.abs(arr)>0)          # noise only on responding genes
        arr+=rng.normal(0,noise*0.3,N)                          # + background noise everywhere
        if keepK:                                               # keep only the top-K |differential| genes
            keep=set(np.argsort(-np.abs(arr))[:keepK].tolist()); arr=np.array([arr[i] if i in keep else 0 for i in range(N)])
        return {G[i]["name"]:arr[i] for i in range(N) if arr[i]!=0}
    def rank_of(res, seed): return next((k+1 for k,r in enumerate(res) if r["gene"]==G[seed]["name"]), 999)
    # (A) noise robustness (full pipeline)
    rob={}
    for noise in [0.0,1.0,2.0]:
        ranks=[]
        for seed in seeds:
            sig=make_sig(seed, noise, keepK=300)
            res=reverse_infer(sig, M, top=50); ranks.append(rank_of(res,seed))
        ranks=np.array(ranks)
        rob[f"noise_{noise}"]=dict(recall_1=round(float((ranks<=1).mean()),3), recall_5=round(float((ranks<=5).mean()),3),
                                   recall_10=round(float((ranks<=10).mean()),3), median_rank=int(np.median(ranks)))
    # (B) per-lens recovery at moderate noise (which lens carries the signal?)
    perlens={"master_reg":[], "diffusion":[], "cascade":[], "codep(independent)":[], "combined":[]}
    for seed in seeds:
        sig=make_sig(seed, 1.0, keepK=300); s=_sig_array(sig,M)
        mr={g:abs(v) for g,v in master_regulators(s,M).items()}
        h=diffuse(s,M); diff={i:h[i] for i in range(N)}
        cset=set(np.argsort(-h)[:300]).union(mr)
        casc=cascade_match(cset, s, M); cdp=codep_coherence(cset, s, M)
        def rk(d): return int(np.where(np.argsort(-np.array([d.get(i,-1e9) for i in range(N)]))==seed)[0][0])+1
        perlens["master_reg"].append(rk(mr)); perlens["diffusion"].append(rk(diff))
        perlens["cascade"].append(rk(casc)); perlens["codep(independent)"].append(rk(cdp))
        res=reverse_infer(sig,M,top=N); perlens["combined"].append(rank_of(res,seed))
    lens_rec={k:dict(recall_10=round(float((np.array(v)<=10).mean()),3), median_rank=int(np.median(v))) for k,v in perlens.items()}
    payload=dict(n_seeds=len(seeds), random_baseline_recall_10=round(10/N,4),
                 noise_robustness=rob, per_lens_recovery=lens_rec,
                 note="seed-recovery from cascade-generated signatures with noise+truncation. Per-lens shows the "
                      "diffusion lens (PPI+co-essentiality, independent of the reg cascade) as non-circular evidence. "
                      "Definitive test = LINCS knockdown recovery (lincs_reverse_test.py, Colab).")
    json.dump(payload, open(OUT/"reverse_validation.json","w"))
    print(f"REVERSE-INFERENCE VALIDATION ({len(seeds)} seeds, random recall@10 = {payload['random_baseline_recall_10']})")
    print("  noise robustness (full pipeline):")
    for k,v in rob.items():
        print(f"    {k}: recall@1 {v['recall_1']} recall@5 {v['recall_5']} recall@10 {v['recall_10']} (median rank {v['median_rank']})")
    print("  per-lens recall@10 (which lens finds the source; diffusion = non-circular):")
    for k,v in lens_rec.items(): print(f"    {k:11}: recall@10 {v['recall_10']} (median rank {v['median_rank']})")

if __name__=="__main__":
    main()
