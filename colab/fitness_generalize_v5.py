"""Fitness generalization v5: CORRECTED AUC formula + best scoring + cross-org.

v1-v4 had an inverted AUC formula (argsort(-s) version). v5 uses the verified
correct formula and applies it to:
  1. SELF-TEST on mtub with the INDEPENDENT DeJesus 2017 truth (best clean test)
  2. SELF-TEST on every matched organism (with our family-derived truth for the
     other orgs -- still imperfect but the only label we have)
  3. CROSS-ORG TRANSFER: project a high-self-AUC source's fitness data via
     feba.db's Ortholog table onto every other organism, validate against
     target truth, slice by phylogenetic same/cross-division.

Scoring (best from v4): combined_absent_plus_min_fit
  - if gene is absent from GeneFitness for that org -> score = ABSENT_BONUS
  - else if gene has any significant fit -> score = -min(fit) over (|t|>4 entries)
  - else (measured but never significant) -> score = 0
"""
import sqlite3, csv, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db")
cur=con.cursor()
OUT=Path("/home/user/cell/outputs/orphan")

# CORRECT AUC (ascending sort + ascending ranks); verified against pairwise
def auc(s, y):
    s=np.array(s); y=np.array(y)
    if (y==1).sum()==0 or (y==0).sum()==0: return float("nan")
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s))
    p=y==1; pos=p.sum()
    return float((r[p].sum() - pos*(pos-1)/2) / (pos*(~p).sum()))

def cov_at_p(scores, ys, target=0.90, ne=False, min_calls=10):
    s=np.array(scores); y=np.array(ys)
    o=np.argsort(s) if ne else np.argsort(-s)
    yo=y[o]; n=np.arange(1,len(yo)+1)
    cls=(yo==0) if ne else (yo==1); tp=np.cumsum(cls); prec=tp/n
    pos=(y==(0 if ne else 1)).sum(); valid=(prec>=target)&(n>=min_calls)
    if not valid.any(): return (0.0,0.0,0)
    k=np.max(np.where(valid)[0]); return (float(prec[k]),float(tp[k]/max(pos,1)),int(n[k]))

# 1. organisms & truth
fb_orgs={r[0]:r[1:] for r in cur.execute("SELECT orgId,division,genus,species,strain FROM Organism")}
bridge=[]
for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv")):
    bridge.append(r["organism"])
bridge=sorted(set(bridge))
MANUAL={"mtub":"MycoTube"}   # our id -> feba orgId where the name differs
bridge=[(o, MANUAL.get(o, o.replace("beril_",""))) for o in bridge]
bridge=[(o,fb) for (o,fb) in bridge if fb in fb_orgs]
truth=defaultdict(dict)
for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv")):
    truth[r["organism"]][r["locus_tag"]] = int(r["essential"])
mtub_truth={lt:tv for lt,tv in truth["mtub"].items()}   # DeJesus 2017
print(f"matched orgs: {len(bridge)}  |  mtub independent truth: {len(mtub_truth)} ({sum(mtub_truth.values())} ess)", flush=True)

# 2. per-org fitness aggregates with |t|>4 significance filter
print("computing per-gene min-fit (|t|>4) per org...", flush=True); t0=time.time()
sig_min={}; any_seen={}
for our, fb in bridge:
    sm={}; sn={}
    for lid, fit, t in cur.execute("SELECT locusId, fit, t FROM GeneFitness WHERE orgId=?", (fb,)):
        if fit is None: continue
        sn[lid]=True
        if t is not None and abs(t)>4:
            if lid not in sm or fit<sm[lid]: sm[lid]=fit
    sig_min[fb]=sm; any_seen[fb]=sn
print(f"  ({time.time()-t0:.0f}s)", flush=True)

ABSENT_BONUS=10.0
def score(fb, lt):
    """combined: absent -> ABSENT_BONUS; sig-min -> -min_fit; measured-but-no-sig -> 0"""
    if lt not in any_seen[fb]: return ABSENT_BONUS  # not in fitness table
    if lt in sig_min[fb]: return -sig_min[fb][lt]
    return 0.0

# 3. mtub SELF-TEST with the INDEPENDENT DeJesus 2017 truth
print("\n=== (A) mtub SELF-TEST: feba.db MycoTube -> DeJesus 2017 truth ===")
sc=[]; ys=[]; abs_count=0
for lt,tv in mtub_truth.items():
    sc.append(score("MycoTube",lt)); ys.append(tv)
    if lt not in any_seen["MycoTube"]: abs_count+=1
a=auc(sc,ys); pE,covE,nE=cov_at_p(sc,ys,0.90)
print(f"  n={len(ys)}  ess={sum(ys)}  absent={abs_count} ({100*abs_count/len(ys):.0f}%)")
print(f"  AUC = {a:.3f}")
print(f"  ess@P>=0.90: precision {pE:.3f}, coverage {covE:.3f}, n calls {nE}")
# absent-only baseline
sc_abs=[1.0 if score("MycoTube",lt)==ABSENT_BONUS else 0.0 for lt in mtub_truth]
a_abs=auc(sc_abs, list(mtub_truth.values()))
print(f"  ABSENT-ONLY baseline AUC = {a_abs:.3f}  (just 'no mutants recovered' as a binary signal)")

# 4. SELF-TEST on every matched org (family-derived truth, less clean)
print("\n=== (B) SELF-TEST on every matched org (most truths are family-derived, take with caveat) ===")
self_auc={}
for our, fb in bridge:
    tr=truth.get(our,{})
    sc=[]; ys=[]
    for lt,tv in tr.items():
        sc.append(score(fb,lt)); ys.append(tv)
    if len(ys)<200 or sum(ys)<10 or (len(ys)-sum(ys))<10: continue
    a=auc(sc,ys); pE,covE,nE=cov_at_p(sc,ys,0.90)
    self_auc[fb]=dict(our=our, auc=a, essCov=covE, n=len(ys), ess=sum(ys))
print(f"{'organism':<24}{'n':>6}{'ess':>5}{'AUC':>7}{'essCov@P0.9':>14}")
for fb,d in sorted(self_auc.items(), key=lambda kv:-kv[1]["auc"])[:25]:
    print(f"  {d['our']:<22}{d['n']:>6}{d['ess']:>5}{d['auc']:>7.3f}{d['essCov']:>14.3f}")
print(f"\nSELF-TEST mean AUC (family-derived truths): {np.mean([v['auc'] for v in self_auc.values()]):.3f}")
print(f"SELF-TEST mean essCov@P0.9                : {np.mean([v['essCov'] for v in self_auc.values()]):.3f}")

# 5. CROSS-ORG TRANSFER (the actual generalization question)
print("\n=== (C) CROSS-ORG TRANSFER via feba.db Ortholog ===")
# use mtub (DeJesus truth, clean self-AUC) + top 5 family-derived sources as sources
sources=["MycoTube"] + sorted([fb for fb,v in self_auc.items() if v["auc"]>=0.65 and fb!="MycoTube"],
                              key=lambda fb:-self_auc[fb]["auc"])[:5]
print(f"sources (in order): {sources}")
rows=[]
for src in sources:
    src_div=fb_orgs[src][0]
    orth=defaultdict(dict)
    for r in cur.execute("SELECT orgId2,locusId2,locusId1 FROM Ortholog WHERE orgId1=?", (src,)):
        orth[r[0]][r[1]] = r[2]
    print(f"\n  source = {src} ({src_div}):")
    for tgt_our, tgt_fb in bridge:
        if tgt_fb==src or tgt_fb not in orth: continue
        tr=truth.get(tgt_our,{})
        sc=[]; ys=[]
        for lt,tv in tr.items():
            src_l=orth[tgt_fb].get(lt)
            if src_l is None: continue
            sc.append(score(src,src_l)); ys.append(tv)
        if len(ys)<200 or sum(ys)<10 or (len(ys)-sum(ys))<10: continue
        a=auc(sc,ys); pE,covE,_=cov_at_p(sc,ys,0.90)
        tgt_div=fb_orgs[tgt_fb][0]
        rows.append(dict(source=src,source_division=src_div,target=tgt_fb,target_division=tgt_div,
                         same_division=(tgt_div==src_div),n=len(ys),ess=sum(ys),
                         auc=round(a,3),essCov=round(covE,3)))
    # print top + bottom for this source
    src_rows=[r for r in rows if r["source"]==src]
    src_rows.sort(key=lambda r:-r["auc"])
    for r in src_rows[:5]:
        mark = "SAME" if r["same_division"] else "cross"
        print(f"    -> {r['target']:<24}{r['target_division'][:14]:<14} {mark}  n={r['n']:>4}  AUC={r['auc']:.3f}  covE={r['essCov']:.3f}")
    if len(src_rows)>5: print(f"    ... and {len(src_rows)-5} more")

same=[r['auc'] for r in rows if r['same_division']]
cross=[r['auc'] for r in rows if not r['same_division']]
print(f"\n=== AUC by phylogenetic relationship (v5 CORRECTED) ===")
print(f"  SELF mean (all orgs)        : {np.mean([v['auc'] for v in self_auc.values()]):.3f}")
print(f"  SAME DIVISION transfer (n={len(same):>3}) : mean {np.mean(same) if same else 0:.3f}  median {np.median(same) if same else 0:.3f}")
print(f"  CROSS DIVISION transfer (n={len(cross):>3}): mean {np.mean(cross) if cross else 0:.3f}  median {np.median(cross) if cross else 0:.3f}")

json.dump(dict(
   mtub_self=dict(auc=a, essCov=covE, absent_only_auc=a_abs, n=len(mtub_truth), ess=sum(mtub_truth.values())),
   self_test_all=self_auc,
   self_mean_auc=float(np.mean([v['auc'] for v in self_auc.values()])),
   self_mean_essCov=float(np.mean([v['essCov'] for v in self_auc.values()])),
   transfer_rows=rows,
   transfer_same_mean=float(np.mean(same)) if same else None,
   transfer_cross_mean=float(np.mean(cross)) if cross else None),
   open(OUT/"fitness_generalize_v5.json","w"), indent=2)
print("\nwrote outputs/orphan/fitness_generalize_v5.json")
