"""Fitness generalization v4 -- the ONE clean test we can do.

Realization from diagnostic:
  - 96% of our truth labels are 'BERIL/FitnessBrowser via essential_families'
    -- DERIVED from family conservation, not per-organism TnSeq. So asking
    "does per-org fit predict per-org family-essentiality" compares apples
    to oranges by construction.
  - mtub is the ONE organism with an INDEPENDENT truth source: DeJesus 2017
    (MTBseq). Direct per-organism TnSeq, not feba-derived.

v4 does the one clean test:
  feba.db MycoTube fitness data -> predict DeJesus 2017 essentiality (mtub)
  with proper |t| > 4 filter, multiple scoring options.

If self-AUC is high here, the fitness signal is real -- and the failures on
the other orgs were truth-label mismatches, not data weakness. Then we can
defensibly use mtub fitness to project to OTHER orgs via Ortholog.

If self-AUC is also low here, something is genuinely wrong with the data or
the join.
"""
import sqlite3, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db")
cur=con.cursor()
OUT=Path("/home/user/cell/outputs/orphan")

# 1. mtub truth (DeJesus 2017, independent)
truth={}
for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv")):
    if r["organism"]=="mtub" and r["source"]=="DeJesus 2017 (MTBseq)":
        truth[r["locus_tag"]] = int(r["essential"])
print(f"mtub DeJesus 2017 truth: {len(truth)} genes, {sum(truth.values())} essential")

# 2. feba.db MycoTube: which orgId?
# from the table list, FitByExp_MycoTube is there. let's find its orgId.
orgs=list(cur.execute("SELECT orgId, division, genus, species, strain FROM Organism WHERE genus LIKE 'Mycobac%'"))
print("Mycobacterium organisms in feba.db:", orgs)
fb="MycoTube"  # assumed table-name match

# 3. confirm ID join: do mtub locus_tags appear as Gene.locusId for MycoTube?
mt_locus=set(r[0] for r in cur.execute("SELECT locusId FROM Gene WHERE orgId=?", (fb,)))
mt_sysname=set(r[0] for r in cur.execute("SELECT sysName FROM Gene WHERE orgId=?", (fb,)))
print(f"\njoin sanity:")
print(f"  truth has {len(truth)} loci")
print(f"  intersect Gene.locusId  : {len(set(truth)&mt_locus)}")
print(f"  intersect Gene.sysName  : {len(set(truth)&mt_sysname)}")
# show first 3 truth locus_tags + first 3 Gene rows
print(f"  sample truth locus_tags: {list(truth)[:5]}")
print(f"  sample Gene rows: {list(cur.execute('SELECT locusId, sysName FROM Gene WHERE orgId=? LIMIT 5',(fb,)))}")

# 4. fetch per-gene fit array (filter |t|>4 for significance, then take MIN)
print(f"\nfetching MycoTube GeneFitness with |t|>4 filter ...")
sig_fits=defaultdict(list)
all_fits=defaultdict(list)
n_total=n_sig=0
for lid, fit, t in cur.execute("SELECT locusId, fit, t FROM GeneFitness WHERE orgId=?", (fb,)):
    if fit is None: continue
    n_total+=1
    all_fits[lid].append(fit)
    if t is not None and abs(t)>4:
        sig_fits[lid].append(fit); n_sig+=1
print(f"  total GeneFitness rows: {n_total}  with |t|>4: {n_sig} ({100*n_sig/n_total:.1f}%)")
print(f"  genes with any sig fit: {len(sig_fits)}; with any fit: {len(all_fits)}")

# 5. SELF-TEST with different scoring options
def auc(s, y):
    s=np.array(s); y=np.array(y)
    if (y==1).sum()==0 or (y==0).sum()==0: return float("nan")
    o=np.argsort(-s); yo=y[o]; pos=(y==1).sum()
    r=np.empty(len(s)); r[o]=np.arange(len(s)); p=y==1
    return float((r[p].sum()-pos*(pos-1)/2)/(pos*(~p).sum()))

# try a few scoring approaches
schemes={
  "min_fit_all (no t-filter)": lambda lt: -min(all_fits[lt]) if all_fits.get(lt) else None,
  "min_fit_sig (|t|>4)": lambda lt: -min(sig_fits[lt]) if sig_fits.get(lt) else None,
  "n_severe_sig (count fit<-3, |t|>4)": lambda lt: sum(1 for f in sig_fits.get(lt,[]) if f<-3),
  "median_fit_sig": lambda lt: -np.median(sig_fits[lt]) if sig_fits.get(lt) else None,
  "absent_from_GeneFitness (binary)": lambda lt: 1.0 if not all_fits.get(lt) else 0.0,
}

print(f"\n=== mtub SELF-TEST with various scores ===")
print(f"{'scheme':<42}{'n_tested':>10}{'n_ess':>8}{'AUC':>8}")
results={}
for name, fn in schemes.items():
    sc=[]; ys=[]
    for lt,tv in truth.items():
        v=fn(lt)
        if v is None: continue
        sc.append(v); ys.append(tv)
    if not sc: continue
    a=auc(sc,ys)
    results[name]=dict(n=len(sc), n_ess=sum(ys), auc=round(a,3))
    print(f"  {name:<40}{len(sc):>10}{sum(ys):>8}{a:>8.3f}")

# 6. combined: use absent as a positive class + min_fit_sig for present
print(f"\n=== COMBINED: absent => high score; present => -min_fit_sig ===")
def combined(lt, bonus=10.0):
    if not all_fits.get(lt): return bonus  # absent = essential
    if sig_fits.get(lt): return -min(sig_fits[lt])
    return 0.0  # measured but no significant fit
sc=[]; ys=[]
for lt,tv in truth.items():
    sc.append(combined(lt)); ys.append(tv)
a=auc(sc,ys)
n_abs=sum(1 for lt in truth if not all_fits.get(lt))
print(f"  n={len(sc)} ess={sum(ys)} absent={n_abs}  AUC={a:.3f}")
results["combined_absent+min_fit_sig"]=dict(n=len(sc), n_ess=sum(ys), auc=round(a,3), n_absent=n_abs)

json.dump(dict(truth_source="DeJesus 2017 (MTBseq)", feba_org="MycoTube",
   schemes=results), open(OUT/"fitness_generalize_v4_mtub.json","w"), indent=2)
print("\nwrote outputs/orphan/fitness_generalize_v4_mtub.json")
