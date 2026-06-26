"""Fitness generalization test on REAL feba.db data.

Question: does TnSeq fitness data transfer across organisms via orthologs?
If we measure fitness in org A, how well does it predict essentiality in org B
as a function of phylogenetic distance?

Method:
  - For each matched (our_org <-> feba_orgId) pair, compute per-gene 'fitness
    essentiality score' = number of conditions in which the gene's fit < -3.0
    (a standard severe-defect threshold). Higher = more conditionally essential.
  - Use feba.db's own Ortholog table (3.6M pre-computed orthologs) to project
    these scores from a SOURCE organism to a TARGET organism.
  - Validate against the TARGET's Keio-style essentiality truth.
  - Compute AUC of source-projected scores vs target truth, sliced by:
       (a) within-clade vs across-clade
       (b) phylogenetic distance (via Organism table divisions)
       (c) how well the source's own labels predict its own truth (self-test)

Output: outputs/orphan/fitness_generalize.json
"""
import sqlite3, csv, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db")
cur=con.cursor()
OUT=Path("/home/user/cell/outputs/orphan")

# 1. orgs and our-panel bridge
fb_orgs={r[0]:r[1:] for r in cur.execute("SELECT orgId,division,genus,species,strain FROM Organism")}
ours=set()
clade_of={}
for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv")):
    ours.add(r["organism"])
for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/clade_splits.csv")):
    clade_of[r["organism"]]=r["clade"]
bridge=[(o, o.replace("beril_","")) for o in ours]
bridge=[(o,fb) for (o,fb) in bridge if fb in fb_orgs]
print(f"matched orgs: {len(bridge)}")

# 2. truth labels
truth=defaultdict(dict)
for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv")):
    truth[r["organism"]][r["locus_tag"]] = int(r["essential"])

# 3. for each matched org: compute per-gene fitness-essentiality from feba.db
print("computing per-org fitness-essentiality (conditions with fit<-3)...", flush=True)
t0=time.time()
fit_ess={}    # feba_orgId -> {locusId: n_severe_conditions}
fit_total={}  # feba_orgId -> {locusId: n_conditions_measured}
for our, fb in bridge:
    rows=cur.execute("SELECT locusId, fit FROM GeneFitness WHERE orgId=?", (fb,))
    se=defaultdict(int); tot=defaultdict(int)
    for lid, fit in rows:
        tot[lid]+=1
        if fit is not None and fit < -3.0: se[lid]+=1
    fit_ess[fb]=se; fit_total[fb]=tot
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# 4. SELF-TEST: does fitness predict essentiality within an organism?
def auc(scores, ys):
    s=np.array(scores); y=np.array(ys)
    o=np.argsort(-s); yo=y[o]; pos=(y==1).sum()
    if pos==0 or pos==len(y): return 0.5
    r=np.empty(len(s)); r[o]=np.arange(len(s)); p=y==1
    return float((r[p].sum()-pos*(pos-1)/2)/(pos*(~p).sum()))

print("\n=== SELF-TEST: feba fitness -> own essentiality truth ===")
print(f"{'organism':<24}{'n':>6}{'ess':>5}{'with_fit':>10}{'AUC':>7}")
self_auc={}
for our, fb in bridge:
    tr=truth.get(our,{})
    if not tr: continue
    # try: locusId may be either our_locus or sysName -- start with direct match
    sc=[]; ys=[]
    for lt,tv in tr.items():
        n=fit_ess[fb].get(lt, None)
        if n is None: continue
        sc.append(n); ys.append(tv)
    if len(ys)<200: continue
    a=auc(sc,ys)
    self_auc[fb]=(our, a, sum(ys), len(ys))
    print(f"  {our:<22}{len(ys):>6}{sum(ys):>5}{len(ys):>10}{a:>7.3f}")
print(f"\nSELF-TEST mean AUC: {np.mean([v[1] for v in self_auc.values()]):.3f}  "
      f"({len(self_auc)} orgs with >=200 joinable genes)")

# 5. CROSS-ORG: project fitness from source -> target via Ortholog table
print("\n=== CROSS-ORG TRANSFER via feba.db Ortholog table ===")
print("(predict TARGET essentiality using SOURCE's fitness profile)")
# pick a small set of high-quality sources
sources=sorted([fb for fb,v in self_auc.items() if v[1]>=0.6], key=lambda fb:-self_auc[fb][1])[:6]
print(f"top {len(sources)} self-test sources: {sources}")

transfer_rows=[]
for src in sources:
    src_our, src_self_auc, _, _ = self_auc[src]
    src_div = fb_orgs[src][0]
    # fetch orthologs FROM src
    print(f"\n  source={src} (own AUC={src_self_auc:.2f}, {src_div})", flush=True)
    orth=defaultdict(dict)   # orth[target_orgId][target_locusId] = src_locusId
    for r in cur.execute("SELECT orgId2,locusId2,locusId1 FROM Ortholog WHERE orgId1=?", (src,)):
        orth[r[0]][r[1]] = r[2]
    for tgt_our, tgt_fb in bridge:
        if tgt_fb==src: continue
        if tgt_fb not in orth: continue
        tr=truth.get(tgt_our,{})
        sc=[]; ys=[]
        for lt,tv in tr.items():
            src_l = orth[tgt_fb].get(lt)
            if src_l is None: continue
            n = fit_ess[src].get(src_l, None)
            if n is None: continue
            sc.append(n); ys.append(tv)
        if len(ys)<200: continue
        a=auc(sc,ys)
        tgt_div=fb_orgs[tgt_fb][0]
        same_div = (tgt_div==src_div)
        transfer_rows.append(dict(source=src, source_division=src_div, source_self_auc=src_self_auc,
                                   target=tgt_fb, target_division=tgt_div,
                                   same_division=same_div, n_joinable=len(ys),
                                   ess_in_target=sum(ys), auc=round(a,3)))
        print(f"    -> {tgt_fb:<26} ({tgt_div[:14]:<14}, {'SAME' if same_div else 'cross'})  "
              f"n={len(ys):>4}  AUC={a:.3f}")

# 6. summarize transfer curve
print("\n=== AUC by phylogenetic relationship ===")
same=[r['auc'] for r in transfer_rows if r['same_division']]
cross=[r['auc'] for r in transfer_rows if not r['same_division']]
print(f"  SAME division (same class): n={len(same)}  mean AUC = {np.mean(same):.3f}  median {np.median(same):.3f}")
print(f"  CROSS division (different class): n={len(cross)}  mean AUC = {np.mean(cross):.3f}  median {np.median(cross):.3f}")
print(f"\n  SELF-TEST mean (upper bound): {np.mean([v[1] for v in self_auc.values()]):.3f}")

json.dump(dict(
   n_matched_orgs=len(bridge),
   self_test={k:dict(our=v[0], auc=v[1], ess=v[2], n=v[3]) for k,v in self_auc.items()},
   self_test_mean_auc=float(np.mean([v[1] for v in self_auc.values()])),
   transfer_same_division_mean=float(np.mean(same)) if same else None,
   transfer_cross_division_mean=float(np.mean(cross)) if cross else None,
   transfer_rows=transfer_rows),
   open(OUT/"fitness_generalize.json","w"), indent=2)
print("\nwrote outputs/orphan/fitness_generalize.json")
