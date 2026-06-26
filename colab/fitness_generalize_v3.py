"""Fitness generalization v3 - CORRECT pipeline.

IDs verified: our truth.locus_tag == feba.db Gene.locusId == GeneFitness.locusId
(100% for matched orgs). v1/v2 mistakes documented; this is the clean run.

Setup:
  - For each organism, compute per-gene MIN FITNESS over all its experiments.
  - SCORE = -min_fit (more negative fit -> higher score -> predicted essential).
  - Restrict to genes WITH fitness data (no absent-bonus heuristic).
  - SELF-TEST AUC: score predicts the org's own essentiality truth.
  - CROSS-ORG TRANSFER AUC: project SOURCE's per-gene score onto TARGET via
    feba's Ortholog table, validate against TARGET truth.

Output:
  outputs/orphan/fitness_generalize_v3.json
"""
import sqlite3, csv, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db")
cur=con.cursor()
OUT=Path("/home/user/cell/outputs/orphan")

fb_orgs={r[0]:r[1:] for r in cur.execute("SELECT orgId,division,genus,species,strain FROM Organism")}
ours=set()
for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv")): ours.add(r["organism"])
bridge=[(o, o.replace("beril_","")) for o in ours]
bridge=[(o,fb) for (o,fb) in bridge if fb in fb_orgs]
truth=defaultdict(dict)
for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv")):
    truth[r["organism"]][r["locus_tag"]] = int(r["essential"])
print(f"matched orgs: {len(bridge)}", flush=True)

# 1. per-gene MIN fit per organism (over all experiments)
print("computing per-gene min_fit per org...", flush=True); t0=time.time()
min_fit={}
for our, fb in bridge:
    mf={}
    for lid, fit in cur.execute("SELECT locusId, fit FROM GeneFitness WHERE orgId=?", (fb,)):
        if fit is None: continue
        if lid not in mf or fit<mf[lid]: mf[lid]=fit
    min_fit[fb]=mf
print(f"  ({time.time()-t0:.0f}s)", flush=True)

def auc(scores, ys):
    s=np.array(scores); y=np.array(ys)
    if (y==1).sum()==0 or (y==0).sum()==0: return float("nan")
    o=np.argsort(-s); yo=y[o]; pos=(y==1).sum()
    r=np.empty(len(s)); r[o]=np.arange(len(s)); p=y==1
    return float((r[p].sum()-pos*(pos-1)/2)/(pos*(~p).sum()))

def cov_at_p(scores, ys, target=0.90, ne=False, min_calls=10):
    s=np.array(scores); y=np.array(ys)
    o=np.argsort(s) if ne else np.argsort(-s)
    yo=y[o]; n=np.arange(1,len(yo)+1)
    cls=(yo==0) if ne else (yo==1); tp=np.cumsum(cls); prec=tp/n
    pos=(y==(0 if ne else 1)).sum(); valid=(prec>=target)&(n>=min_calls)
    if not valid.any(): return (0.0,0.0,0)
    k=np.max(np.where(valid)[0]); return (float(prec[k]),float(tp[k]/max(pos,1)),int(n[k]))

# 2. SELF-TEST: only genes with fitness data
print("\n=== SELF-TEST v3 (genes with fitness data only) ===")
print(f"{'organism':<24}{'n_with_fit':>12}{'ess_in_test':>13}{'AUC':>8}{'essCov@P0.9':>13}")
self_auc={}
for our, fb in bridge:
    tr=truth.get(our,{})
    sc=[]; ys=[]
    for lt,tv in tr.items():
        v=min_fit[fb].get(lt)
        if v is None: continue
        sc.append(-v); ys.append(tv)
    if len(ys)<200: continue
    if sum(ys)<10 or (len(ys)-sum(ys))<10: continue
    a=auc(sc,ys)
    pE,covE,_=cov_at_p(sc,ys,0.90,ne=False)
    self_auc[fb]=(our,a,sum(ys),len(ys),covE)
    print(f"  {our:<22}{len(ys):>12}{sum(ys):>13}{a:>8.3f}{covE:>13.3f}")
print(f"\nSELF-TEST v3 mean AUC: {np.mean([v[1] for v in self_auc.values()]):.3f}  "
      f"mean essCov@P0.90: {np.mean([v[4] for v in self_auc.values()]):.3f}")

# 3. CROSS-ORG TRANSFER via Ortholog
print("\n=== CROSS-ORG TRANSFER v3 ===")
sources=sorted([fb for fb,v in self_auc.items() if v[1]>=0.70], key=lambda fb:-self_auc[fb][1])[:8]
print(f"top self-AUC sources used: {sources}\n")
rows=[]
for src in sources:
    src_div=fb_orgs[src][0]
    orth=defaultdict(dict)
    for r in cur.execute("SELECT orgId2,locusId2,locusId1 FROM Ortholog WHERE orgId1=?", (src,)):
        orth[r[0]][r[1]] = r[2]
    for tgt_our, tgt_fb in bridge:
        if tgt_fb==src or tgt_fb not in orth: continue
        tr=truth.get(tgt_our,{})
        sc=[]; ys=[]
        for lt,tv in tr.items():
            src_l=orth[tgt_fb].get(lt)
            if src_l is None: continue
            v=min_fit[src].get(src_l)
            if v is None: continue
            sc.append(-v); ys.append(tv)
        if len(ys)<200 or sum(ys)<10 or (len(ys)-sum(ys))<10: continue
        a=auc(sc,ys); _,covE,_=cov_at_p(sc,ys,0.90,ne=False)
        tgt_div=fb_orgs[tgt_fb][0]
        rows.append(dict(source=src,source_division=src_div,
            target=tgt_fb,target_division=tgt_div,same_division=(tgt_div==src_div),
            n=len(ys),ess=sum(ys),auc=round(a,3),essCov=round(covE,3)))

# group by source -> distance class
print(f"{'source':<14}{'-> target':<28}{'div':<14}{'sameDiv':>9}{'n':>5}{'AUC':>7}{'covE':>7}")
for src in sources:
    src_div=fb_orgs[src][0]
    src_self=self_auc[src][1]
    print(f"  {src} (self-AUC {src_self:.2f}, {src_div}):")
    for r in [x for x in rows if x["source"]==src]:
        print(f"    -> {r['target']:<24}{r['target_division'][:13]:<14}{('SAME' if r['same_division'] else 'cross'):>9}{r['n']:>5}{r['auc']:>7.3f}{r['essCov']:>7.3f}")

same=[r['auc'] for r in rows if r['same_division']]
cross=[r['auc'] for r in rows if not r['same_division']]
print(f"\n=== AUC by phylogenetic relationship (v3) ===")
print(f"  SELF (upper bound)       : mean {np.mean([v[1] for v in self_auc.values()]):.3f}")
print(f"  SAME DIVISION (n={len(same):>3}) : mean {np.mean(same) if same else 0:.3f}  median {np.median(same) if same else 0:.3f}")
print(f"  CROSS DIVISION (n={len(cross):>3}): mean {np.mean(cross) if cross else 0:.3f}  median {np.median(cross) if cross else 0:.3f}")

# essCov gain over current 2-wheel baseline (~0.50)
mean_xfer_essCov_same=np.mean([r['essCov'] for r in rows if r['same_division']])
mean_xfer_essCov_cross=np.mean([r['essCov'] for r in rows if not r['same_division']])
print(f"\n  essCov@P0.9: same-div transfer mean {mean_xfer_essCov_same:.3f}  cross-div {mean_xfer_essCov_cross:.3f}")
print(f"  for reference, our 2-wheel essCov@P0.9 mean ~0.50")

json.dump(dict(
   self_test={k:dict(our=v[0],auc=v[1],ess=v[2],n=v[3],essCov_P090=v[4]) for k,v in self_auc.items()},
   self_mean_auc=float(np.mean([v[1] for v in self_auc.values()])),
   self_mean_essCov=float(np.mean([v[4] for v in self_auc.values()])),
   transfer_same_division_auc=float(np.mean(same)) if same else None,
   transfer_cross_division_auc=float(np.mean(cross)) if cross else None,
   transfer_same_essCov=float(mean_xfer_essCov_same) if rows else None,
   transfer_cross_essCov=float(mean_xfer_essCov_cross) if rows else None,
   transfer_rows=rows),
   open(OUT/"fitness_generalize_v3.json","w"), indent=2)
print("\nwrote outputs/orphan/fitness_generalize_v3.json")
