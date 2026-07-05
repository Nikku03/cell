"""TIME RESOLUTION — the LEARNED predictor (does ML beat the mechanistic 0.29?).

The mechanistic relaxation equation (compute_timeresolved.py) hit ~0.29 median fraction-error on the EGF
held-out test; the oracle single-time-constant floor is ~0.17. The gap is per-gene EFFECTIVE-rate
heterogeneity the closed form can't see. This script asks the decisive question: given ALL our per-gene
features (half-lives, network position, expression, magnitude, cell-cycle, essentiality, constraint), can
a LEARNED model predict each gene's trajectory better than the equation?

Two models, both evaluated on held-out genes:
  A) features -> effective tau, then the equation (interpretable; bounded by the 0.17 single-tau oracle).
  B) (features, timepoint) -> fraction-complete DIRECTLY (learns arbitrary shape; can beat the oracle).
Architecture note: for one perturbation + a fixed per-gene feature vector, gradient boosting is the correct
learned model (a from-scratch transformer would overfit ~600 train genes). A transformer earns its keep on
SEQUENCE data across MANY perturbations (LINCS) -- the next step if learning helps here.

-> timeresolved_ml.json (held-out errors for every method + feature importances)
"""
import json, math, csv
from statistics import median
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan"); LN2=math.log(2)
TIMES=[0,20,40,60,120,240,480]; MID=[20,40,60,120,240]

def frac_complete(tau_min, t): return 1.0-math.exp(-t/tau_min)
def _best_tau(fr):
    best=None
    for tau in [10,20,30,45,60,90,120,180,240,360,480,720,1000,1500,2500,4000]:
        e=sum((frac_complete(tau,t)-fo)**2 for t,fo in fr)
        if best is None or e<best[1]: best=(tau,e)
    return best[0]

def _num(name,col):
    f=OUT/name; m={}
    if not f.exists(): return m
    for r in csv.DictReader(open(f),delimiter="\t"):
        try: m[r["gene"]]=float(r[col])
        except (KeyError,ValueError,TypeError): pass
    return m
def _str(name,col):
    f=OUT/name; m={}
    if not f.exists(): return m
    for r in csv.DictReader(open(f),delimiter="\t"):
        if r.get("gene") and r.get(col): m[r["gene"]]=r[col]
    return m

def main():
    from compute_timeresolved import load_reg_depth, IEG
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except Exception as e:
        print("timeresolved-ml needs scikit-learn:",repr(e)[:80]); return
    mrna=_num("mrna_halflife.tsv","half_life_h"); prot=_num("protein_halflife.tsv","half_life_h")
    phase=_str("cellcycle_phase.tsv","phase")
    tc={}
    f=OUT/"timecourse_egf.tsv"
    if f.exists():
        for r in csv.DictReader(open(f),delimiter="\t"):
            try: tc[r["gene"]]=[float(r[f"t{t}"]) for t in TIMES]
            except (KeyError,ValueError): pass
    if not (mrna and tc): print("timeresolved-ml: need mrna_halflife + timecourse_egf -> skipped"); return
    depth=load_reg_depth(IEG)
    # gene-level features from cell_complete (essentiality, LOEUF, TF/regulatory degree)
    ess={}; loeuf={}; indeg={}; outdeg={}; istf=set()
    p=OUT/"cell_complete.json"
    if p.exists():
        D=json.load(open(p)); G=D.get("genes",[]); nm=[g["name"] for g in G]
        for g in G: ess[g["name"]]=g.get("ess",0); loeuf[g["name"]]=g.get("loeuf",-1)
        for e in D.get("reg",[]):
            if len(e)<2: continue
            s=nm[e[0]] if isinstance(e[0],int) and e[0]<len(nm) else e[0]
            t=nm[e[1]] if isinstance(e[1],int) and e[1]<len(nm) else e[1]
            if s: outdeg[s]=outdeg.get(s,0)+1; istf.add(s)
            if t: indeg[t]=indeg.get(t,0)+1
    PH=["G1/S","S","G2","G2/M","M/G1"]
    FEATS=["log_mrna_hl","log_prot_hl","cascade_depth","log_t0","log2_fold","fold_sign",
           "essential","loeuf","log_indeg","log_outdeg","is_tf","is_ieg"]+["ph_"+p for p in PH]
    rows=[]; taus=[]; frac_rows=[]; keys=[]
    for g,vals in tc.items():
        hl=mrna.get(g)
        if not hl: continue
        R0,Rf=vals[0],vals[-1]
        if R0<=0 or Rf<=0 or abs(math.log2(Rf/R0))<1.0: continue
        fr=[(t,(R-R0)/(Rf-R0)) for t,R in zip(TIMES,vals) if t in MID]
        if any(fo<-0.25 or fo>1.25 for _,fo in fr): continue
        d=depth.get(g)
        feat=[math.log10(hl), math.log10(prot[g]) if g in prot else np.nan,
              d if d is not None else np.nan, math.log10(max(R0,1e-3)),
              abs(math.log2(Rf/R0)), 1.0 if Rf>R0 else -1.0,
              float(ess.get(g,0)), loeuf.get(g,-1) if loeuf.get(g,-1)>=0 else np.nan,
              math.log10(indeg.get(g,0)+1), math.log10(outdeg.get(g,0)+1),
              1.0 if g in istf else 0.0, 1.0 if g in IEG else 0.0]
        feat+=[1.0 if phase.get(g)==p else 0.0 for p in PH]
        rows.append(feat); taus.append(math.log10(_best_tau(fr))); frac_rows.append(fr); keys.append(g)
    n=len(rows)
    if n<40: print(f"timeresolved-ml: only {n} testable genes -> too few to train; skipped"); return
    X=np.array(rows); ytau=np.array(taus)
    rng=np.arange(n); tr=rng%2==0; te=~tr                         # deterministic held-out split
    tau_med_min=median([10**t for t in ytau])                    # global baseline

    # ---- Model A: features -> effective tau ----
    A=HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05, l2_regularization=1.0)
    A.fit(X[tr], ytau[tr]); tau_pred=10**A.predict(X)

    # ---- Model B: (features, timepoint) -> fraction-complete directly ----
    Xb_tr=[]; yb_tr=[];
    for i in np.where(tr)[0]:
        for t,fo in frac_rows[i]: Xb_tr.append(list(X[i])+[t/480.0, math.log10(t)]); yb_tr.append(fo)
    B=HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.05, l2_regularization=1.0)
    B.fit(np.array(Xb_tr), np.array(yb_tr))

    # ---- evaluate all methods on the TEST genes ----
    e_step=[]; e_mech=[]; e_glob=[]; e_oracle=[]; e_mlA=[]; e_mlB=[]
    for i in np.where(te)[0]:
        mech_tau=mrna[keys[i]]*60/LN2                            # measured half-life -> tau (the mechanistic baseline)
        orc_tau=10**ytau[i]
        for t,fo in frac_rows[i]:
            e_step.append(abs(1.0-fo))
            e_mech.append(abs(frac_complete(mech_tau,t)-fo))
            e_glob.append(abs(frac_complete(tau_med_min,t)-fo))
            e_oracle.append(abs(frac_complete(orc_tau,t)-fo))
            e_mlA.append(abs(frac_complete(tau_pred[i],t)-fo))
            pb=float(B.predict(np.array([list(X[i])+[t/480.0, math.log10(t)]]))[0])
            e_mlB.append(abs(min(max(pb,0.0),1.5)-fo))
    def md(x): return round(float(median(x)),4) if x else None
    # permutation importance (HistGBR has no native feature_importances_): which features drive effective tau
    imp=[]
    try:
        from sklearn.inspection import permutation_importance
        pi=permutation_importance(A, X[te], ytau[te], n_repeats=10, random_state=0, scoring="neg_mean_squared_error")
        imp=sorted(zip(FEATS, pi.importances_mean), key=lambda z:-z[1])
    except Exception: pass
    res=dict(n_test_genes=int(te.sum()), n_train_genes=int(tr.sum()),
             median_abs_fraction_error=dict(
                 instant_step=md(e_step), mechanistic_measured_halflife=md(e_mech),
                 global_single_tau=md(e_glob), ml_A_features_to_tau=md(e_mlA),
                 ml_B_features_to_fraction=md(e_mlB), oracle_single_tau_floor=md(e_oracle)),
             top_features=[[f,round(float(v),3)] for f,v in imp[:8]])
    json.dump(res, open(OUT/"timeresolved_ml.json","w"))
    e=res["median_abs_fraction_error"]
    print(f"time-resolution LEARNED predictor | train {res['n_train_genes']} / test {res['n_test_genes']} genes (held-out)")
    print(f"  median |fraction error|: instant-step {e['instant_step']} | mechanistic {e['mechanistic_measured_halflife']} | "
          f"global {e['global_single_tau']} | ML-A(tau) {e['ml_A_features_to_tau']} | ML-B(direct) {e['ml_B_features_to_fraction']} "
          f"| oracle-floor {e['oracle_single_tau_floor']}")
    best_ml=min(e['ml_A_features_to_tau'], e['ml_B_features_to_fraction'])
    verdict=("LEARNING HELPS" if best_ml < e['mechanistic_measured_halflife']-0.005 else
             "learning does NOT beat the equation (info not in static features -> need measured-input data)")
    print(f"  >>> VERDICT: {verdict} (best ML {best_ml} vs mechanistic {e['mechanistic_measured_halflife']}, oracle {e['oracle_single_tau_floor']})")
    if imp: print("  what predicts response speed:", ", ".join(f"{f}={v:.2f}" for f,v in imp[:5]))

if __name__=="__main__":
    main()
