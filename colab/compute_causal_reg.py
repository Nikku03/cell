"""PHASE 3 — causal regulome: upgrade TF binding to TF regulation.

ReMap says a TF *binds* near a gene (candidate); Perturb-seq says knocking out the TF *actually moves*
that gene (response). An edge supported by BOTH is causally-supported regulation — including edges in
no curated database. Sign: if the gene goes DOWN when the TF is knocked out, the TF normally ACTIVATES
it (+1); if UP, the TF REPRESSES it (-1). Optional 3rd line: GTEx trans-eQTL agreement.

Inputs (skips gracefully if absent): remap_tf_targets.tsv (Phase-4/ReMap), perturbseq_targets.json
(from compute_perturbseq), gtex_trans_edges.tsv (optional). -> causal_reg.tsv + causal_reg.json
(loaded by build_cell_complete as high-confidence, signed regulatory edges).
"""
import json, csv
from collections import defaultdict
from pathlib import Path
OUT=Path("outputs/orphan"); H=Path("data/external_data/human")

def load_remap():
    tf2t=defaultdict(dict)   # TF -> {target: peak_support}   (TF/target uppercased to match Perturb-seq HGNC)
    p=OUT/"remap_tf_targets.tsv"
    if p.exists():
        for l in open(p):
            r=l.rstrip("\n").split("\t")
            if len(r)>=2:
                tf=r[0].strip().upper(); g=r[1].strip().upper()
                try: tf2t[tf][g]=int(r[2]) if len(r)>2 else 1
                except: tf2t[tf][g]=1
    return tf2t

def load_pert_targets():
    p=OUT/"perturbseq_targets.json"
    if not p.exists(): return {}
    raw=json.load(open(p))
    # {TF: {gene: ko_sign}} — uppercase both the perturbed-gene key and the readout genes so they
    # share ReMap's identifier space (some Perturb-seq labels arrive lower/mixed case).
    return {str(tf).strip().upper():{str(g).strip().upper():s for g,s in tl} for tf,tl in raw.items()}

def load_gtex():
    pairs=set(); p=OUT/"gtex_trans_edges.tsv"
    if p.exists():
        for l in open(p):
            r=l.rstrip("\n").split("\t")
            if len(r)>=2: pairs.add((r[0],r[1]))
    return pairs

def main():
    remap=load_remap(); pert=load_pert_targets(); gtex=load_gtex()
    both_tf=set(remap)&set(pert)
    diag=dict(n_remap_tfs=len(remap), n_pert_genes=len(pert),
              remap_file_exists=(OUT/"remap_tf_targets.tsv").exists(),
              pert_file_exists=(OUT/"perturbseq_targets.json").exists(),
              tf_overlap=len(both_tf),
              remap_tf_sample=sorted(remap)[:12], pert_key_sample=sorted(pert)[:12])
    if not remap or not pert:
        print(f"causal regulome: ReMap TFs={len(remap)} (file={diag['remap_file_exists']}), "
              f"Perturb-seq target-genes={len(pert)} (file={diag['pert_file_exists']}) -> skipping (one input empty)")
        json.dump(dict(n_edges=0, n_tfs=0, n_triple=0, edges=[], diagnostics=diag), open(OUT/"causal_reg.json","w"))
        return
    if not both_tf:   # both inputs present but their identifier spaces don't intersect -> surface it loudly
        print(f"causal regulome: remap TFs={len(remap)}, perturb keys={len(pert)}, but ZERO overlap. "
              f"remap sample={diag['remap_tf_sample'][:6]} | pert sample={diag['pert_key_sample'][:6]} "
              f"-> no causal edges (identifier mismatch or one screen lacks TFs)")
        json.dump(dict(n_edges=0, n_tfs=0, n_triple=0, edges=[], diagnostics=diag), open(OUT/"causal_reg.json","w"))
        return
    edges=[]
    for tf in both_tf:
        bound=remap[tf]; responders=pert[tf]
        for g in set(bound)&set(responders):
            ko_sign=responders[g]                       # -1 = gene went down on KO
            reg_sign=-ko_sign                           # TF activates (+) if KO lowers the gene
            support=["remap_binding","perturbseq_response"]
            if (tf,g) in gtex: support.append("gtex_eqtl")
            edges.append(dict(tf=tf,target=g,sign=reg_sign,
                              peaks=bound[g],n_support=len(support),support=support))
    edges.sort(key=lambda e:(-e["n_support"],-e["peaks"]))
    with open(OUT/"causal_reg.tsv","w") as o:
        for e in edges: o.write(f"{e['tf']}\t{e['target']}\t{e['sign']}\n")
    diag["n_edges"]=len(edges)
    json.dump(dict(n_edges=len(edges),n_tfs=len(both_tf),
                   n_triple=sum(1 for e in edges if e["n_support"]>=3),
                   edges=edges[:5000], diagnostics=diag), open(OUT/"causal_reg.json","w"))
    print(f"causal regulome: {len(edges)} causally-supported TF->target edges from {len(both_tf)} TFs "
          f"(binding x response); {sum(1 for e in edges if e['n_support']>=3)} also GTEx-supported")

if __name__=="__main__":
    main()
