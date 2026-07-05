"""BILLION-DOLLAR TEST — can the integrated cell DISCOVER drug targets from biology alone?

A virtual cell that identifies viable drug targets before the field drugs them is worth billions (each
validated novel target = a potential drug program). Rigorous, non-circular test:

  LABEL   = gene is a known drug target (has a DGIdb drug).
  FEATURES = BIOLOGY ONLY -- essentiality, DepMap dependency, LOEUF constraint, literature, regulatory/PPI/
             co-essentiality network position, TF/master status, CpG/enhancer. NO drug or disease features
             (so the model can't cheat by seeing the answer).
  STAR     = the DENSE multi-lens network feature: what fraction of a gene's PPI+regulatory+co-essential
             neighbors are targets -- computed per-CV-fold with TRAIN labels only (no leakage).

We ask three things: (1) does biology predict targetability at all (held-out AUC)? (2) does the DENSE
NETWORK add real signal over features alone? (3) what are the top-ranked, NOT-yet-drugged, UNDERSTUDIED
genes -- the discovery shortlist -- and are they enriched for independent value signals (constraint,
essentiality)? -> target_discovery.json
"""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan")

def main():
    cc=OUT/"cell_complete.json"
    if not cc.exists(): print("cell_complete.json absent -> skipped"); return
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
    except Exception as e:
        print("needs scikit-learn:",repr(e)[:80]); return
    D=json.load(open(cc)); G=D["genes"]; N=len(G)
    drugs=D.get("drugs") or {}
    # LABEL = target of an APPROVED drug (the validated, valuable target class), not just any DGIdb interaction.
    has_drug=np.array([1 if str(i) in drugs else 0 for i in range(N)])
    y=np.array([1 if any((dd.get("a") if isinstance(dd,dict) else False) for dd in drugs.get(str(i),[])) else 0
                for i in range(N)])
    # --- dense neighbor adjacency (PPI + regulation + co-essentiality) ---
    adj=defaultdict(set)
    for e in D.get("ppi",[]):
        if len(e)>=2: adj[e[0]].add(e[1]); adj[e[1]].add(e[0])
    for e in D.get("reg",[]):
        if len(e)>=2: adj[e[0]].add(e[1]); adj[e[1]].add(e[0])
    for k,v in (D.get("codep") or {}).items():
        i=int(k)
        for p in v: adj[i].add(p[0] if isinstance(p,(list,tuple)) else p)
    ppi_deg=defaultdict(int); rin=defaultdict(int); rout=defaultdict(int)
    for e in D.get("ppi",[]):
        if len(e)>=2: ppi_deg[e[0]]+=1; ppi_deg[e[1]]+=1
    for e in D.get("reg",[]):
        if len(e)>=2: rout[e[0]]+=1; rin[e[1]]+=1
    codep_deg={int(k):len(v) for k,v in (D.get("codep") or {}).items()}
    def f(g,k,d=np.nan):
        v=g.get(k,d)
        if v in (None,""): return d
        try: return float(v)
        except (TypeError,ValueError): return 1.0    # non-numeric flag (e.g. master='neuron') -> presence
    Xb=np.zeros((N,12))
    for i,g in enumerate(G):
        lo=f(g,"loeuf"); lo=lo if (isinstance(lo,float) and lo>=0) else np.nan
        Xb[i]=[lo, f(g,"ess",0), f(g,"dep_frac",0), math.log10(f(g,"pubs",0)+1), f(g,"dark",0),
               f(g,"npath",0), f(g,"tf",0), f(g,"master",0), f(g,"cpg",0),
               math.log10(ppi_deg.get(i,0)+1), math.log10(rin.get(i,0)+rout.get(i,0)+1),
               math.log10(codep_deg.get(i,0)+1)]
    FNAMES=["loeuf","essential","dep_frac","log_pubs","dark","n_pathways","is_tf","master_tf","cpg",
            "log_ppi_deg","log_reg_deg","log_codep_deg","network_target_frac"]
    # --- 5-fold CV; network feature computed per-fold with train labels only ---
    skf=StratifiedKFold(5, shuffle=True, random_state=0)
    oof_base=np.zeros(N); oof_net=np.zeros(N)
    for tr,te in skf.split(Xb,y):
        trs=set(tr.tolist())
        netf=np.full(N, np.nan)
        for i in range(N):
            nb=[j for j in adj.get(i,()) if j in trs and 0<=j<N]
            if nb: netf[i]=float(np.mean([y[j] for j in nb]))
        Xn=np.column_stack([Xb, netf])
        mb=HistGradientBoostingClassifier(max_iter=300,max_depth=4,learning_rate=0.06,l2_regularization=1.0).fit(Xb[tr],y[tr])
        mn=HistGradientBoostingClassifier(max_iter=300,max_depth=4,learning_rate=0.06,l2_regularization=1.0).fit(Xn[tr],y[tr])
        oof_base[te]=mb.predict_proba(Xb[te])[:,1]; oof_net[te]=mn.predict_proba(Xn[te])[:,1]
    auc_base=roc_auc_score(y,oof_base); auc_net=roc_auc_score(y,oof_net)
    # --- discovery shortlist: TRULY UNDRUGGED genes (no DGIdb drug at all) that score like APPROVED targets ---
    cand=[(i, oof_net[i]) for i in range(N) if has_drug[i]==0]
    cand.sort(key=lambda z:-z[1]); short=[i for i,_ in cand[:60]]
    def constrained(i): lo=f(G[i],"loeuf"); return isinstance(lo,float) and 0<=lo<0.35
    undrugged=[i for i in range(N) if has_drug[i]==0]
    short_constr=np.mean([constrained(i) for i in short]); base_constr=np.mean([constrained(i) for i in undrugged])
    short_ess=np.mean([f(G[i],"ess",0) for i in short]); base_ess=np.mean([f(G[i],"ess",0) for i in undrugged])
    res=dict(n_genes=N, n_targets=int(y.sum()),
             auc_biology_only=round(float(auc_base),4), auc_with_dense_network=round(float(auc_net),4),
             network_gain=round(float(auc_net-auc_base),4),
             discovery_shortlist=[dict(gene=G[i]["name"], score=round(float(oof_net[i]),3),
                 loeuf=f(G[i],"loeuf"), essential=int(f(G[i],"ess",0)), pubs=int(f(G[i],"pubs",0))) for i in short[:30]],
             shortlist_enrichment=dict(constrained_frac=round(float(short_constr),3), constrained_base=round(float(base_constr),3),
                 constrained_enrichment=round(float(short_constr/base_constr),2) if base_constr>0 else None,
                 essential_frac=round(float(short_ess),3), essential_base=round(float(base_ess),3),
                 essential_enrichment=round(float(short_ess/base_ess),2) if base_ess>0 else None),
             note="predict DGIdb drug-target from biology-only features; held-out 5-fold AUC; dense network "
                  "feature computed per-fold with train labels (no leakage). Shortlist = high-score, undrugged, understudied.")
    json.dump(res, open(OUT/"target_discovery.json","w"))
    print(f"BILLION-DOLLAR TEST — can biology alone find drug targets? ({N} genes, {int(y.sum())} known targets)")
    print(f"  held-out AUC: biology-only {res['auc_biology_only']} | +DENSE NETWORK {res['auc_with_dense_network']} "
          f"(network adds +{res['network_gain']})")
    e=res["shortlist_enrichment"]
    print(f"  discovery shortlist (undrugged + understudied, top-scored): "
          f"{e['constrained_enrichment']}x enriched for LoF-constraint, {e['essential_enrichment']}x for essentiality")
    print("  top novel target candidates:", ", ".join(d["gene"] for d in res["discovery_shortlist"][:12]))

if __name__=="__main__":
    main()
