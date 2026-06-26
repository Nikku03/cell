"""Honest stats for the 81% of genes WITHOUT native FBA data.

These genes have only TWO real signals: W1 (transformer) and W2 (leak-clean
conservation). No proxy W3, no OG-transferred W3, no defaults. We report what
the 2-wheel system delivers on exactly this set.

A gene is in the '81% no-FBA' set if:
  - it is in an organism with NO metabolic model (all 56 such organisms), OR
  - it is in a modeled organism but does NOT map into that model.

For each such gene we require W1 and W2 to be real to make a 2-wheel call;
genes missing even one are reported separately (thinner-than-2-wheel).
"""
import csv, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
OUT=Path("outputs/orphan"); MODELS=Path("data/external_data/bigg_models")

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

# ---- determine which genes HAVE native FBA (to EXCLUDE them) ----
b_to_keio={r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
keio_to_b={v:k for k,v in b_to_keio.items()}
def koxy_bridge(M):
    ka={r["locus_tag"]:r["product"] for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv"))
        if r["organism"]=="beril_Koxy" and r.get("product")}
    tok=defaultdict(list)
    for lt,prod in ka.items():
        for t in re.findall(r"[A-Za-z0-9]{4,}",prod): tok[t.lower()].append(lt)
    out=set()
    for g in M.genes:
        gn=(g.name or "").strip()
        if gn and g.id!=gn:
            mm=tok.get(gn.lower(),[])
            if len(mm)==1: out.add(mm[0])
    return out

has_fba=set()  # (org, lt) that map into a native model
print("identifying genes WITH native FBA (to exclude)...", flush=True)
for our_org, model_id, bk in [("beril_Putida","iJN1463","native"),("beril_Keio","iML1515","keio"),
                              ("mtub","iEK1008","native"),("beril_Koxy","iYL1228","koxy")]:
    M=cobra.io.read_sbml_model(str(MODELS/f"{model_id}.xml.gz"))
    mids={g.id for g in M.genes}
    for (o,lt) in truth_by:
        if o!=our_org: continue
        if bk=="native" and lt in mids: has_fba.add((o,lt))
        elif bk=="keio" and keio_to_b.get(lt) in mids: has_fba.add((o,lt))
    if bk=="koxy":
        kb=koxy_bridge(M)
        for lt in kb: has_fba.add(("beril_Koxy",lt))
print(f"  genes with native FBA: {len(has_fba)}\n", flush=True)

# ---- the 81% = everything else; run 2-wheel ----
def cov_at_p(scores, ys, target=0.90, min_calls=10, ne=False):
    s=np.array(scores); y=np.array(ys)
    # essential: rank by HIGH score first; non-essential: rank by LOW score first
    o=np.argsort(s) if ne else np.argsort(-s)
    yo=y[o]; n=np.arange(1,len(yo)+1)
    cls = (yo==0) if ne else (yo==1)
    tp=np.cumsum(cls); prec=tp/n; pos=(y==(0 if ne else 1)).sum()
    valid=(prec>=target)&(n>=min_calls)
    if not valid.any(): return (0.0,0.0,0)
    k=np.max(np.where(valid)[0]); return (float(prec[k]), float(tp[k]/max(pos,1)), int(n[k]))

orgs=sorted({o for (o,_) in truth_by})
rows=[]
tot_genes=both=only_w1=neither=0
for org in orgs:
    held=clade_of.get(org,"?"); rate=cons_rate(held)
    recs=[]
    n_both=n_w1only=n_neither=0
    for (o,lt),tv in truth_by.items():
        if o!=org: continue
        if (o,lt) in has_fba: continue   # exclude native-FBA genes
        tot_genes+=1
        w1v=w1.get((held,lt), None)
        og=og_of.get((o,lt)); w2v=rate.get(og, None) if og else None
        if w1v is not None and w2v is not None:
            n_both+=1; recs.append((tv, w1v, w2v))
        elif w1v is not None:
            n_w1only+=1
        else:
            n_neither+=1
    both+=n_both; only_w1+=n_w1only; neither+=n_neither
    if len(recs)<50: continue
    ys=[r[0] for r in recs]; pos=sum(ys); N=len(recs)
    if pos<10 or (N-pos)<10: continue
    s=[(r[1]+r[2])/2 for r in recs]
    pE,covE,nE=cov_at_p(s,ys,0.90,ne=False)
    pN,covN,nN=cov_at_p(s,ys,0.90,ne=True)
    rows.append(dict(org=org, n2wheel=N, base=round(pos/N,3),
                     essP=round(pE,3),essCov=round(covE,3),
                     neP=round(pN,3),neCov=round(covN,3)))

rows.sort(key=lambda r:-r["n2wheel"])
print(f"{'organism':<24}{'n(2-wheel)':>11}{'base':>6}{'essCov@P0.9':>12}{'neCov@P0.9':>12}")
for r in rows[:25]:
    print(f"  {r['org']:<22}{r['n2wheel']:>11}{r['base']:>6.2f}{r['essCov']:>12.2f}{r['neCov']:>12.2f}")

import numpy as np
essC=np.array([r["essCov"] for r in rows]); neC=np.array([r["neCov"] for r in rows])
print(f"\n=== 2-WHEEL stats on the no-FBA genes ({len(rows)} organisms with >=50 callable) ===")
print(f"  gene accounting (all no-FBA genes):")
print(f"    total no-FBA genes:        {tot_genes}")
print(f"    with W1+W2 (2-wheel):      {both}  ({100*both/tot_genes:.0f}%)")
print(f"    with W1 only (no OG -> 1-wheel): {only_w1}  ({100*only_w1/tot_genes:.0f}%)")
print(f"    with neither:              {neither}  ({100*neither/tot_genes:.0f}%)")
print(f"\n  essCov @ P>=0.90 : mean {essC.mean():.3f}  median {np.median(essC):.3f}  range {essC.min():.2f}-{essC.max():.2f}")
print(f"  neCov  @ P>=0.90 : mean {neC.mean():.3f}  median {np.median(neC):.3f}  range {neC.min():.2f}-{neC.max():.2f}")
print(f"  (precision held at >=0.90 by construction)")

import json
json.dump(dict(total_nofba_genes=tot_genes, with_2wheel=both, w1_only=only_w1, neither=neither,
   n_orgs=len(rows), mean_essCov=float(essC.mean()), median_essCov=float(np.median(essC)),
   mean_neCov=float(neC.mean()), median_neCov=float(np.median(neC)),
   essCov_range=[float(essC.min()),float(essC.max())], per_org=rows),
   open(OUT/"two_wheel_81pct.json","w"), indent=2)
print("\nwrote outputs/orphan/two_wheel_81pct.json")
