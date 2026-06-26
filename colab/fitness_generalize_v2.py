"""Fitness generalization v2: correct scoring.

v1 used "n severe conditions" as score and got self-AUC ~0.52. Reason: genes
with NO fitness data (no recovered mutants = absolute essentials) were silently
excluded, removing the strongest essentiality signal.

v2 fixes this with two changes:
  1. SCORE = max(0, -minFit) where minFit is the per-gene minimum over all
     conditions. Plus a saturation bonus for genes ABSENT from GeneFitness
     (interpreted as "could not be measured = likely essential").
  2. Test on the full Keio-truth gene set including the genes absent from
     GeneFitness (most informative ones).

Then re-run self-test + cross-organism transfer.
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

# 1. per-org: min_fit per gene (lowest = most essential)
print("computing per-gene MIN FITNESS per organism...", flush=True); t0=time.time()
min_fit={}      # fb -> {locusId: min_fit}
n_meas={}       # fb -> {locusId: n_conditions}
absent_essential_rate={}  # gene-absent prior P(essential | absent)
for our, fb in bridge:
    mf={}; nm=defaultdict(int)
    for lid, fit in cur.execute("SELECT locusId, fit FROM GeneFitness WHERE orgId=?", (fb,)):
        if fit is None: continue
        nm[lid]+=1
        if lid not in mf or fit<mf[lid]: mf[lid]=fit
    min_fit[fb]=mf; n_meas[fb]=nm
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# 2. SCORE per gene:
#    base = max(0, -min_fit)  (clip negative min_fit to a positive essentiality score)
#    + bonus if gene is absent from GeneFitness entirely (signal of "no mutants recovered")
def score_for(fb, lt, ABSENT_BONUS=8.0):
    mf=min_fit[fb].get(lt)
    if mf is None: return ABSENT_BONUS
    return max(0.0, -mf)

def auc(scores, ys):
    s=np.array(scores); y=np.array(ys)
    o=np.argsort(-s); yo=y[o]; pos=(y==1).sum()
    if pos==0 or pos==len(y): return 0.5
    r=np.empty(len(s)); r[o]=np.arange(len(s)); p=y==1
    return float((r[p].sum()-pos*(pos-1)/2)/(pos*(~p).sum()))

# 3. SELF-TEST v2 (now including absent genes)
print("\n=== SELF-TEST v2 (includes ABSENT-from-fitness signal) ===")
print(f"{'organism':<24}{'n_tested':>10}{'ess':>5}{'%absent':>9}{'AUC':>7}")
self_auc={}
for our, fb in bridge:
    tr=truth.get(our,{})
    if not tr: continue
    sc=[]; ys=[]; abs_count=0
    for lt,tv in tr.items():
        s=score_for(fb,lt)
        sc.append(s); ys.append(tv)
        if min_fit[fb].get(lt) is None: abs_count+=1
    if len(ys)<200: continue
    a=auc(sc,ys)
    self_auc[fb]=(our,a,sum(ys),len(ys),abs_count)
    print(f"  {our:<22}{len(ys):>10}{sum(ys):>5}{100*abs_count/len(ys):>8.1f}%{a:>7.3f}")
print(f"\nSELF-TEST v2 mean AUC: {np.mean([v[1] for v in self_auc.values()]):.3f}  "
      f"(n={len(self_auc)} orgs)")

# 4. ABSENT-ALONE baseline (no fit scores, just "is gene measured?")
print("\n=== ABSENT-FROM-FITNESS ALONE (no quantitative score, just the binary signal) ===")
absent_only_aucs=[]
for our, fb in bridge:
    tr=truth.get(our,{})
    if not tr: continue
    sc=[1.0 if min_fit[fb].get(lt) is None else 0.0 for lt in tr]
    ys=list(tr.values())
    if len(ys)<200: continue
    a=auc(sc, ys)
    absent_only_aucs.append((fb,a))
print(f"  mean AUC (binary absent vs present): {np.mean([a for _,a in absent_only_aucs]):.3f}")

# 5. CROSS-ORG TRANSFER v2 via Ortholog
print("\n=== CROSS-ORG TRANSFER v2 (project min_fit + absent-bonus via orthologs) ===")
sources=sorted([fb for fb,v in self_auc.items() if v[1]>=0.75], key=lambda fb:-self_auc[fb][1])[:6]
print(f"top self-AUC sources used: {sources}")
transfer_rows=[]
for src in sources:
    src_div=fb_orgs[src][0]
    orth=defaultdict(dict)
    for r in cur.execute("SELECT orgId2,locusId2,locusId1 FROM Ortholog WHERE orgId1=?", (src,)):
        orth[r[0]][r[1]] = r[2]
    print(f"\n  src={src} ({src_div})")
    for tgt_our, tgt_fb in bridge:
        if tgt_fb==src or tgt_fb not in orth: continue
        tr=truth.get(tgt_our,{})
        sc=[]; ys=[]
        for lt,tv in tr.items():
            src_l=orth[tgt_fb].get(lt)
            if src_l is None: continue
            sc.append(score_for(src, src_l)); ys.append(tv)
        if len(ys)<200: continue
        a=auc(sc,ys)
        tgt_div=fb_orgs[tgt_fb][0]
        transfer_rows.append(dict(source=src,source_division=src_div,
            target=tgt_fb,target_division=tgt_div,same_division=(tgt_div==src_div),
            n=len(ys),ess=sum(ys),auc=round(a,3)))
        print(f"    -> {tgt_fb:<26} ({tgt_div[:14]:<14}, {'SAME' if tgt_div==src_div else 'cross'})  n={len(ys):>4} AUC={a:.3f}")

same=[r['auc'] for r in transfer_rows if r['same_division']]
cross=[r['auc'] for r in transfer_rows if not r['same_division']]
print("\n=== AUC by phylogenetic relationship (v2) ===")
print(f"  SAME class: n={len(same)}  mean {np.mean(same) if same else 0:.3f}  median {np.median(same) if same else 0:.3f}")
print(f"  CROSS class: n={len(cross)}  mean {np.mean(cross) if cross else 0:.3f}  median {np.median(cross) if cross else 0:.3f}")
print(f"  SELF-TEST mean (upper bound): {np.mean([v[1] for v in self_auc.values()]):.3f}")

json.dump(dict(self_test={k:dict(our=v[0],auc=v[1],ess=v[2],n=v[3],absent=v[4]) for k,v in self_auc.items()},
   self_test_mean=float(np.mean([v[1] for v in self_auc.values()])),
   absent_only_mean=float(np.mean([a for _,a in absent_only_aucs])),
   transfer_same_division_mean=float(np.mean(same)) if same else None,
   transfer_cross_division_mean=float(np.mean(cross)) if cross else None,
   transfer_rows=transfer_rows),
   open(OUT/"fitness_generalize_v2.json","w"), indent=2)
print("\nwrote outputs/orphan/fitness_generalize_v2.json")
