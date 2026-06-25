"""Structural 'trucks we don't need' filter -> confident NON-essential calls.

Idea (yours): instead of finding which soup gene fills a hole, FILTER OUT the
genes the cell clearly doesn't need -> those are non-essential. Three pure-graph
flags (NO FBA):
  - redundant : every reaction the gene catalyses has another gene (isozyme
                backup) -> removing it disables nothing.
  - disconnected: the gene only touches reactions that are already blocked
                (involve a dead-end metabolite) -> carries no flux.
  - orphans-nothing: removing the gene orphans no non-currency metabolite
                (complement of the sole-connection 'candidate essential' set).
A gene flagged 'not needed' is called NON-ESSENTIAL.

Measure on iML1515 vs Keio truth:
  - precision of each flag + their union
  - non-essential COVERAGE gain when we add the structural calls on top of W1
    (transformer) at a high precision target.

Output: outputs/orphan/structural_noness.json
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
cobra.Configuration().solver="glpk"
OUT=Path("outputs/orphan"); MODELS=Path("data/external_data/bigg_models"); MODEL="iML1515"
CURRENCY={"h2o","h","pi","ppi","atp","adp","amp","gtp","gdp","gmp","utp","udp","ump","ctp",
          "cdp","cmp","nad","nadh","nadp","nadph","fad","fadh2","fmn","fmnh2","coa","co2","o2",
          "nh4","hco3","so4","h2s","na1","k","cl","mg2","mn2","fe2","fe3","ca2","cobalt2","cu2",
          "zn2","mobd","h2","q8","q8h2","mqn8","mql8","2dmmql8","2dmmq8","h2o2","amet","ahcys",
          "accoa","acp","thf","mlthf","methf","5mthf","10fthf","dhf","glu__L","akg","pep","pyr"}
def base(mid): return mid.rsplit("_",1)[0]

print(f"loading {MODEL} (graph only)...", flush=True)
M=cobra.io.read_sbml_model(str(MODELS/f"{MODEL}.xml.gz"))

producers=defaultdict(set); consumers=defaultdict(set)
for R in M.reactions:
    if R.id.startswith(("EX_","DM_","SK_")): continue
    for m,c in R.metabolites.items():
        if R.reversibility: producers[m.id].add(R.id); consumers[m.id].add(R.id)
        elif c>0: producers[m.id].add(R.id)
        else: consumers[m.id].add(R.id)
# dead-end metabolites (pre-existing dangling): produced-but-not-consumed or vice-versa
deadend={mid for mid in set(list(producers)+list(consumers))
         if base(mid) not in CURRENCY and (not producers[mid] or not consumers[mid])}

gene_rxns=defaultdict(set)
for R in M.reactions:
    if R.id.startswith(("EX_","DM_","SK_")) or "BIOMASS" in R.id.upper(): continue
    for g in R.genes: gene_rxns[g.id].add(R.id)

flags={}   # gene -> dict of flags
for gid,rxns in gene_rxns.items():
    Rs=[M.reactions.get_by_id(r) for r in rxns]
    redundant = bool(rxns) and all(len(R.genes)>=2 for R in Rs)
    # sole-connection (candidate essential): sole producer/consumer of a non-currency met
    sole=False
    for R in Rs:
        for m in R.metabolites:
            mid=m.id
            if base(mid) in CURRENCY: continue
            if producers[mid] and consumers[mid] and (not (producers[mid]-rxns) or not (consumers[mid]-rxns)):
                sole=True; break
        if sole: break
    # disconnected: every reaction touches a dead-end metabolite (can't carry flux)
    disconnected = bool(rxns) and all(any(m.id in deadend for m in R.metabolites) for R in Rs)
    flags[gid]=dict(redundant=redundant, sole=sole, disconnected=disconnected,
                    orphans_nothing=(not sole))

# ---- truth + W1 ----
b_to_keio={r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
truth={r["locus_tag"]:int(r["essential"]) for r in csv.DictReader(open("data/drive_import/labels/labels.csv")) if r["organism"]=="beril_Keio"}
clade_of={r["organism"]:r["clade"] for r in csv.DictReader(open("data/drive_import/labels/clade_splits.csv"))}
kclade=clade_of.get("beril_Keio","escherichia")
P=np.load(OUT/"af_torch_preds_aug.npz",allow_pickle=True)
_cl,_lt,_p=P["clade"],P["lt"],P["p"]
w1={}
for i in range(len(_p)):
    if str(_cl[i])==kclade and not np.isnan(_p[i]): w1[str(_lt[i])]=float(_p[i])

# universe: model genes joinable to truth AND with a W1 score
rows=[]
for gid in gene_rxns:
    lt=b_to_keio.get(gid)
    if lt is None or lt not in truth or lt not in w1: continue
    rows.append(dict(gene=gid, lt=lt, truth=truth[lt], w1=w1[lt], **flags[gid]))
N=len(rows); pos_ess=sum(r["truth"] for r in rows); pos_ne=N-pos_ess
print(f"\n  test universe (model gene ∩ truth ∩ W1): {N} genes "
      f"({pos_ess} essential, {pos_ne} non-essential)", flush=True)

def flagstats(name, key):
    called=[r for r in rows if r[key]]
    tn=sum(1 for r in called if r["truth"]==0)
    P_=tn/max(len(called),1); R_=tn/max(pos_ne,1)
    print(f"  {name:<22} calls={len(called):<5} ne-precision={P_:.3f}  ne-coverage(recall)={R_:.3f}")
    return dict(flag=name, calls=len(called), precision=round(P_,3), recall=round(R_,3))

print("\n=== each structural flag as a NON-ESSENTIAL call ===")
fs=[flagstats("redundant", "redundant"),
    flagstats("disconnected", "disconnected"),
    flagstats("orphans-nothing", "orphans_nothing")]
# union: redundant OR disconnected (the two high-confidence flags)
for r in rows: r["struct_strict"]=r["redundant"] or r["disconnected"]
for r in rows: r["struct_loose"]=r["redundant"] or r["disconnected"] or r["orphans_nothing"]
fs.append(flagstats("strict (red|disc)", "struct_strict"))
fs.append(flagstats("loose (+orphans)", "struct_loose"))

# ---- coverage gain over W1 at a high non-essential precision target ----
def necov_at_precision(callfn, target):
    """callfn(r, t)->bool (called non-essential). sweep t, max recall with precision>=target."""
    best=(0.0,0.0,None)
    for t in np.linspace(0.01,0.99,99):
        called=[r for r in rows if callfn(r,t)]
        if len(called)<10: continue
        tn=sum(1 for r in called if r["truth"]==0)
        P_=tn/len(called); R_=tn/max(pos_ne,1)
        if P_>=target and R_>best[1]: best=(P_,R_,round(t,2))
    return best

for target in (0.95,0.90):
    w1_only=necov_at_precision(lambda r,t: r["w1"]<t, target)
    combo  =necov_at_precision(lambda r,t: (r["w1"]<t) or r["struct_strict"], target)
    combo_l=necov_at_precision(lambda r,t: (r["w1"]<t) or r["struct_loose"], target)
    print(f"\n=== non-essential COVERAGE @ precision>={target} (metabolic test universe) ===")
    print(f"  W1 alone           : neCov={w1_only[1]:.3f} (P={w1_only[0]:.3f}, t={w1_only[2]})")
    print(f"  W1 + strict struct : neCov={combo[1]:.3f} (P={combo[0]:.3f})   gain={(combo[1]-w1_only[1])*100:+.1f}pp")
    print(f"  W1 + loose struct  : neCov={combo_l[1]:.3f} (P={combo_l[0]:.3f})   gain={(combo_l[1]-w1_only[1])*100:+.1f}pp")
    if target==0.95:
        save=dict(target=target, w1_neCov=w1_only[1], w1_P=w1_only[0],
                  combo_strict_neCov=combo[1], combo_strict_P=combo[0], gain_strict_pp=(combo[1]-w1_only[1])*100,
                  combo_loose_neCov=combo_l[1], combo_loose_P=combo_l[0], gain_loose_pp=(combo_l[1]-w1_only[1])*100)

json.dump(dict(model=MODEL, universe=N, essential=pos_ess, nonessential=pos_ne,
               flags=fs, coverage=save), open(OUT/"structural_noness.json","w"), indent=2)
print(f"\nwrote outputs/orphan/structural_noness.json", flush=True)
