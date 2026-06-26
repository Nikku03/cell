"""Run the 3-wheel system on FULL-COVERAGE data only. No proxies, no transfer.

A gene enters the test universe only if it has, in this organism:
  - a Keio/TnSeq-style essentiality TRUTH label
  - W1 (transformer prediction) for its clade and locus
  - W2 (leak-clean conservation rate) i.e. has an orthogroup with >=5 cross-
    clade observations
  - W3 = REAL FBA single-gene-deletion (not the FBA-free proxy, not transferred
    across OGs). Requires the organism has its OWN BiGG model AND the gene maps
    into that model.

Restricts naturally to the 4 organisms with native metabolic models: Putida,
Keio, mtub, Koxy. Genes outside the model, or with no OG, or with no W1 score
are EXCLUDED -- not given a default value.

Then measure exactly what the system delivers on that complete-data core:
  precision/recall/coverage of essential and non-essential calls, plus the
  consensus tiers. No tuning across the set.
"""
import csv, json, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
cobra.Configuration().solver="glpk"
OUT=Path("outputs/orphan"); MODELS=Path("data/external_data/bigg_models")

RICH=["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L","gly","his__L",
      "ile__L","leu__L","lys__L","met__L","phe__L","pro__L","ser__L","thr__L","trp__L",
      "tyr__L","val__L","ade","gua","ura","thymd","cytd","ins","adn","gsn","uri","thm",
      "ribflv","nac","pnto__R","pydx","4abz","btn","fol"]

# load data
ALL=list(csv.DictReader(open("data/drive_import/labels/labels.csv")))
clade_of={r["organism"]:r["clade"] for r in csv.DictReader(open("data/drive_import/labels/clade_splits.csv"))}
og_of={(r["organism"],r["locus_tag"]):r["og_id"]
       for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")) if r.get("og_id")}
truth_by={(r["organism"],r["locus_tag"]):int(r["essential"]) for r in ALL}
P=np.load(OUT/"af_torch_preds_aug.npz", allow_pickle=True)
_cl,_lt,_p=P["clade"],P["lt"],P["p"]
w1={(str(_cl[i]),str(_lt[i])): float(_p[i]) for i in range(len(_p)) if not np.isnan(_p[i])}

def cons_rate(held):
    n=defaultdict(int); e=defaultdict(int)
    for r in ALL:
        if clade_of.get(r["organism"])==held: continue
        og=og_of.get((r["organism"], r["locus_tag"]))
        if og: n[og]+=1; e[og]+=int(r["essential"])
    return {og: e[og]/n[og] for og in n if n[og]>=5}

# bridges to model gene ids
b_to_keio={r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
keio_to_b={v:k for k,v in b_to_keio.items()}

CONFIG=[("beril_Putida","iJN1463","native"),
        ("beril_Keio","iML1515","keio"),
        ("mtub","iEK1008","native"),
        ("beril_Koxy","iYL1228","koxy")]

def koxy_bridge(M):
    ka={r["locus_tag"]:r["product"] for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv"))
        if r["organism"]=="beril_Koxy" and r.get("product")}
    tok=defaultdict(list)
    for lt,prod in ka.items():
        for t in re.findall(r"[A-Za-z0-9]{4,}",prod): tok[t.lower()].append(lt)
    out={}
    for g in M.genes:
        gn=(g.name or "").strip()
        if gn and g.id!=gn:
            mm=tok.get(gn.lower(),[])
            if len(mm)==1: out[mm[0]]=g.id
    return out

def fba_run(model_id):
    M=cobra.io.read_sbml_model(str(MODELS/f"{model_id}.xml.gz"))
    for x in RICH:
        rid=f"EX_{x}_e"
        if rid in M.reactions: M.reactions.get_by_id(rid).lower_bound=-10
    g0=M.slim_optimize()
    de=single_gene_deletion(M)
    fba={}
    for _,row in de.iterrows():
        gid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
        fba[gid]=int(row["growth"]<0.01*g0 or np.isnan(row["growth"]))
    return fba, M

def auc(scores, ys):
    s=np.array(scores); y=np.array(ys); o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s))
    pos=y==1
    if pos.sum()==0 or (~pos).sum()==0: return 0.5
    return float((r[pos].sum()-pos.sum()*(pos.sum()-1)/2)/(pos.sum()*(~pos).sum()))

def cov_at_p(scores, ys, target=0.90, min_calls=10, ascending=False):
    s=np.array(scores); y=np.array(ys)
    if ascending:
        o=np.argsort(s); y=y; pos_class=0
    else:
        o=np.argsort(-s); pos_class=1
    yo=y[o]; n=np.arange(1,len(yo)+1)
    if pos_class==1:
        tp=np.cumsum(yo); prec=tp/n; pos=(y==1).sum()
    else:
        tp=np.cumsum(yo==0); prec=tp/n; pos=(y==0).sum()
    valid=(prec>=target)&(n>=min_calls)
    if not valid.any(): return (0.0, 0.0, 0)
    k=np.max(np.where(valid)[0])
    return (float(prec[k]), float(tp[k]/max(pos,1)), int(n[k]))

print("Running FULL-COVERAGE 3-wheel test on the 4 modeled organisms.", flush=True)
print("Every gene in the test universe has REAL: W1 + W2 + W3 (native FBA) + truth.\n", flush=True)

rows_all=[]
per_org=[]
t0=time.time()
for our_org, model_id, bk in CONFIG:
    fba, M = fba_run(model_id)
    model_ids={g.id for g in M.genes}
    # our locus -> model gene
    if bk=="native":
        loc2mod={lt:lt for (o,lt) in truth_by if o==our_org and lt in model_ids}
    elif bk=="keio":
        loc2mod={lt:keio_to_b[lt] for (o,lt) in truth_by if o==our_org and lt in keio_to_b and keio_to_b[lt] in model_ids}
    elif bk=="koxy":
        loc2mod=koxy_bridge(M)
    held=clade_of.get(our_org,"?")
    rate=cons_rate(held)
    # build per-gene rows for THIS org with FULL coverage
    rows=[]
    n_truth=n_w1=n_w2=n_w3=n_all=0
    for (o,lt),tv in truth_by.items():
        if o!=our_org: continue
        n_truth+=1
        og=og_of.get((o,lt))
        w1v=w1.get((held,lt), None)
        w2v=rate.get(og, None) if og else None
        mg=loc2mod.get(lt)
        w3v=fba.get(mg, None) if mg else None
        if w1v is not None: n_w1+=1
        if w2v is not None: n_w2+=1
        if w3v is not None: n_w3+=1
        # FULL COVERAGE means ALL present
        if w1v is not None and w2v is not None and w3v is not None:
            n_all+=1
            rows.append((lt, tv, w1v, w2v, float(w3v)))
            rows_all.append(dict(org=our_org, lt=lt, truth=tv, w1=w1v, w2=w2v, w3=w3v))
    if not rows: continue
    ys=[r[1] for r in rows]; pos=sum(ys); N=len(rows); base=pos/N
    # individual wheel AUC
    a1=auc([r[2] for r in rows], ys)
    a2=auc([r[3] for r in rows], ys)
    a3=auc([r[4] for r in rows], ys)
    # combine: mean of 3 (no per-org tuning)
    s_mean=[(r[2]+r[3]+r[4])/3 for r in rows]
    a_mean=auc(s_mean, ys)
    # essCov @ P>=0.90 (consensus mean score)
    pE,covE,nE=cov_at_p(s_mean, ys, 0.90)
    # neCov @ P>=0.90 (call non-essential when LOW score; sort ascending)
    s_inv=[-x for x in s_mean]
    pN,covN,nN=cov_at_p(s_inv, [1-y for y in ys], 0.90)
    # consensus TTT tier (all 3 wheels >=0.5)
    ttt=[(r[2]>=0.5 and r[3]>=0.5 and r[4]>=0.5) for r in rows]
    n_ttt=sum(ttt); tp_ttt=sum(1 for r,t in zip(rows,ttt) if t and r[1]==1)
    nnn=[(r[2]<0.5 and r[3]<0.5 and r[4]<0.5) for r in rows]
    n_nnn=sum(nnn); tn_nnn=sum(1 for r,t in zip(rows,nnn) if t and r[1]==0)
    rec=dict(organism=our_org, total_keio_truth=n_truth,
             with_w1=n_w1, with_w2=n_w2, with_w3=n_w3, full_coverage=n_all,
             base_essential=round(base,3),
             auc_W1=round(a1,3), auc_W2=round(a2,3), auc_W3=round(a3,3),
             auc_consensus=round(a_mean,3),
             ess_calls_at_P090=nE, ess_precision=round(pE,3), ess_coverage=round(covE,3),
             ne_calls_at_P090=nN, ne_precision=round(pN,3), ne_coverage=round(covN,3),
             TTT_calls=n_ttt, TTT_precision=round(tp_ttt/max(n_ttt,1),3),
             NNN_calls=n_nnn, NNN_precision=round(tn_nnn/max(n_nnn,1),3))
    per_org.append(rec)
    print(f"  {our_org:<14} truth={n_truth:<5} W1={n_w1} W2={n_w2} W3={n_w3} -> FULL={n_all}")
    print(f"     wheels AUC W1={a1:.2f} W2={a2:.2f} W3={a3:.2f}  consensus={a_mean:.2f}")
    print(f"     ess@P0.90: P={pE:.2f} cov={covE:.2f} (n={nE})  |  ne@P0.90: P={pN:.2f} cov={covN:.2f} (n={nN})")
    print(f"     TTT (all-3-essential): n={n_ttt} P={tp_ttt/max(n_ttt,1):.2f}  |  NNN (all-3-non-ess): n={n_nnn} P={tn_nnn/max(n_nnn,1):.2f}\n")

# aggregate (excluding Koxy if its bridge is too small)
big=[r for r in per_org if r["full_coverage"]>=500]
mE=np.mean([r["ess_coverage"] for r in big])
mN=np.mean([r["ne_coverage"] for r in big])
mP=np.mean([r["ess_precision"] for r in big])
mPN=np.mean([r["ne_precision"] for r in big])
mTTT=np.mean([r["TTT_precision"] for r in big])
mNNN=np.mean([r["NNN_precision"] for r in big])
N_all=sum(r["full_coverage"] for r in per_org)
all_truth=sum(r["total_keio_truth"] for r in per_org)
print(f"=== AGGREGATE: full-data core only (no proxies, no transfer) ===")
print(f"  total Keio-truth genes across 4 orgs: {all_truth}")
print(f"  full-coverage subset (all 3 wheels present): {N_all}  ({100*N_all/all_truth:.0f}%)")
print(f"  mean ess@P>=0.90 across {len(big)} large orgs: P={mP:.3f}  coverage={mE:.3f}")
print(f"  mean ne@P>=0.90:                                 P={mPN:.3f}  coverage={mN:.3f}")
print(f"  mean TTT consensus precision (all 3 say ess):    P={mTTT:.3f}")
print(f"  mean NNN consensus precision (all 3 say non-ess):P={mNNN:.3f}")

# Honest scope statement
excluded_orgs=set(truth_by.keys()) - set((r["organism"], None) for r in per_org)
print(f"\n  EXCLUDED from this test: every gene without a native FBA model and every")
print(f"  organism without one. That is {all_truth - N_all} genes ({100*(all_truth-N_all)/all_truth:.0f}%)")
print(f"  of our truth-labelled gene pool -- they remain in the 2-wheel regime.")

json.dump(dict(per_org=per_org, n_full=N_all, n_truth_total=all_truth,
   mean_ess_P=float(mP), mean_ess_coverage=float(mE),
   mean_ne_P=float(mPN), mean_ne_coverage=float(mN),
   mean_TTT_precision=float(mTTT), mean_NNN_precision=float(mNNN),
   excluded_genes=all_truth-N_all),
   open(OUT/"full_coverage_test.json","w"), indent=2)
print(f"\nwrote outputs/orphan/full_coverage_test.json ({time.time()-t0:.0f}s)")
