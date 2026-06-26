"""Valve-graph wheel: the cell as pipes + valves, propagation not optimization.

NO FBA, NO LP. We combine three layers:
  - METABOLISM (from iML1515): reaction = pipe; substrates required, products made
  - TRANSPORT: same, just cross compartment (already in iML1515)
  - REGULATION (RegulonDB): TF activates/represses target gene transcription;
                            sigma factor must be present for transcription

Propagation rule (fixed-point):
  - a metabolite is AVAILABLE if it is in the medium OR produced by an active rxn
  - a gene is EXPRESSED if NOT knocked out AND (no required activator/sigma is
    blocking it) AND (no repressor is active solo). Conservatively for unknown
    regulation, default expressed. Conservative for known regulation: if any
    listed activator/sigma is necessary, the gene needs at least one alive.
  - a reaction FIRES if all its GPR genes are expressed AND its substrates are
    available
  - iterate until stable

Knockout = remove that gene from EXPRESSED set, re-propagate, and check whether
any biomass precursor that was reachable in the intact cell is now unreachable.
Gene called essential by the valve graph.

Validate on iML1515 + RegulonDB vs Keio truth:
  - precision / recall overall
  - broken down by gene CATEGORY (metabolic / transporter / regulator / sigma /
    other-non-metabolic)
  - compare to FBA baseline single-gene-deletion
  - and to W1 (transformer) precision on the same universe
"""
import csv, json, time, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
cobra.Configuration().solver="glpk"
OUT=Path("outputs/orphan"); MODELS=Path("data/external_data/bigg_models"); REG=Path("data/external_data/regulon")
MODEL="iML1515"

CURRENCY={"h2o","h","pi","ppi","atp","adp","amp","gtp","gdp","gmp","utp","udp","ump","ctp","cdp","cmp",
          "nad","nadh","nadp","nadph","fad","fadh2","fmn","fmnh2","coa","co2","o2","nh4","hco3","so4",
          "h2s","na1","k","cl","mg2","mn2","fe2","fe3","ca2","cobalt2","cu2","zn2","mobd","h2","q8",
          "q8h2","mqn8","mql8","2dmmql8","2dmmq8","h2o2","amet","ahcys","accoa","acp","thf","mlthf",
          "methf","5mthf","10fthf","dhf","glu__L","akg","pep","pyr"}
def base(mid): return mid.rsplit("_",1)[0]

# rich medium
RICH=["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L","gly","his__L","ile__L","leu__L",
      "lys__L","met__L","phe__L","pro__L","ser__L","thr__L","trp__L","tyr__L","val__L","ade","gua","ura",
      "thymd","cytd","ins","adn","gsn","uri","thm","ribflv","nac","pnto__R","pydx","4abz","btn","fol"]
print(f"loading {MODEL}...", flush=True); t0=time.time()
M=cobra.io.read_sbml_model(str(MODELS/f"{MODEL}.xml.gz"))
for x in RICH:
    rid=f"EX_{x}_e"
    if rid in M.reactions: M.reactions.get_by_id(rid).lower_bound=-10

# ---- gene name -> b-number map (RegulonDB uses gene names; model uses b-numbers) ----
name_to_b={}
for g in M.genes:
    nm=(g.name or "").strip()
    if nm: name_to_b.setdefault(nm.lower(), g.id)
# RegulonDB lists genes as standard E. coli names; b-numbers also accepted
def name_to_bnum(n):
    if not n: return None
    n=n.strip()
    if n.startswith("b") and len(n)==5 and n[1:].isdigit(): return n
    return name_to_b.get(n.lower())

# ---- load RegulonDB edges (TF + sigma -> target gene) ----
def load_edges(path):
    out=[]
    for l in open(path):
        if not l.strip() or l.startswith("#"): continue
        p=l.rstrip("\n").split("\t")
        if len(p)<3: continue
        tf, tgt, eff = p[0], p[1], p[2].lower()
        if eff not in ("activator","repressor","dual"): continue
        out.append((tf.lower(), tgt, eff))
    return out
tf_edges=load_edges(REG/"network_tf_gene.txt")
sigma_edges=load_edges(REG/"network_sigma_gene.txt")
# map TF gene name to b-number too (so a TF knockout = removing its b-number gene)
tf_to_b=defaultdict(lambda: None)
for tf,_,_ in tf_edges+sigma_edges: tf_to_b[tf]=name_to_bnum(tf)
tf_set={tf for tf,_,_ in tf_edges+sigma_edges if tf_to_b[tf]}
# build target -> activators / repressors  (also include sigma as activator class)
activators=defaultdict(set); repressors=defaultdict(set); sigmas=defaultdict(set)
for tf,tgt,eff in tf_edges:
    b=tf_to_b[tf]
    if not b: continue
    if eff in ("activator","dual"): activators[tgt.lower()].add(b)
    if eff in ("repressor","dual"): repressors[tgt.lower()].add(b)
for tf,tgt,eff in sigma_edges:
    b=tf_to_b[tf]
    if not b: continue
    sigmas[tgt.lower()].add(b)
n_reg_targets = len(set(list(activators)+list(repressors)+list(sigmas)))
n_tf_bnums = len({b for s in (activators,repressors,sigmas) for vs in s.values() for b in vs})
print(f"  RegulonDB: {len(tf_edges)} TF-edges + {len(sigma_edges)} sigma-edges; "
      f"{n_reg_targets} regulated genes; {n_tf_bnums} TF/sigma genes mapped to b-numbers", flush=True)

# ---- build per-gene 'expression' rule and per-reaction firing rule ----
gene_name_lower={g.id: (g.name or "").strip().lower() for g in M.genes}
def gene_expressed(b, alive_set):
    """A gene is expressed if (no required sigma rule violated) AND (not solely repressed).
    Conservative interpretation: if the gene has any sigma in our list, require
    at least one alive sigma (matches biology -- sigma factors are non-redundant
    when listed). If the gene has activators, require at least one alive activator.
    If only repressors, expressed if no repressor is alive (knockout of derepressor
    is OK -- gene comes on)."""
    if b not in alive_set: return False
    nm = gene_name_lower.get(b,"")
    sigs = sigmas.get(nm, set())
    acts = activators.get(nm, set())
    reps = repressors.get(nm, set())
    if sigs and not (sigs & alive_set): return False
    if acts and not (acts & alive_set): return False
    # if only repressors, the gene is expressed unless ALL listed repressors are
    # active -- which actually means "fully repressed". This is the common case:
    # remove any one repressor and the gene comes on. So default expressed when
    # at least one repressor is dead (always true unless all reps removed).
    if reps and not acts and not sigs:
        # all reps alive -> repressed; otherwise expressed (default)
        if reps <= alive_set:
            return False
    return True

# ---- baseline FBA + reach computation ----
g0=M.slim_optimize()
print(f"  FBA rich growth g0={g0:.3f}", flush=True)

# eval cobra GPR with an expressed-set
from cobra.core.gene import eval_gpr, parse_gpr
gpr_cache={}
for R in M.reactions:
    if R.gene_reaction_rule:
        try: gpr_cache[R.id]=parse_gpr(R.gene_reaction_rule)[0]
        except: gpr_cache[R.id]=None
    else: gpr_cache[R.id]=None

def reaction_fires(R, expressed):
    """GPR-aware: rxn fires unless GPR evaluates to False when given 'expressed' set."""
    tree=gpr_cache.get(R.id)
    if tree is None: return True  # no GPR = spontaneous, always active
    knocked={g.id for g in R.genes if g.id not in expressed}
    try: return eval_gpr(tree, knocked)
    except: return True

# ---- propagation: build reachable metabolite set given an expressed gene set ----
biomass_rxns=[R for R in M.reactions if "BIOMASS" in R.id.upper()]
bio_pre=set()
for R in biomass_rxns:
    for m,c in R.metabolites.items():
        if c<0 and base(m.id) not in CURRENCY: bio_pre.add(m.id)
print(f"  non-currency biomass precursors: {len(bio_pre)}", flush=True)

# seed = currency + medium uptake products
seed=set(m.id for m in M.metabolites if base(m.id) in CURRENCY)
for R in M.reactions:
    if R.id.startswith("EX_") and R.lower_bound<0:
        for m in R.metabolites: seed.add(m.id)

# precompute reaction tuples
dirs=[]   # (subs:frozenset, prods:tuple, rid)
for R in M.reactions:
    if R.id.startswith(("EX_","DM_","SK_")) or "BIOMASS" in R.id.upper(): continue
    subs=tuple(m.id for m,c in R.metabolites.items() if c<0)
    pros=tuple(m.id for m,c in R.metabolites.items() if c>0)
    if subs and pros:
        dirs.append((frozenset(subs), pros, R.id))
        if R.reversibility: dirs.append((frozenset(pros), subs, R.id))

rxn_by_id={r.id:r for r in M.reactions}
def reach(expressed_genes):
    avail=set(seed)
    # which reactions are FIRING-capable (gene-wise)
    firing_cap={rid for rid,(subs,pros,_) in [(d[2],d) for d in dirs]}
    firing_cap={rid for rid in {d[2] for d in dirs} if reaction_fires(rxn_by_id[rid], expressed_genes)}
    changed=True
    while changed:
        changed=False
        for subs,pros,rid in dirs:
            if rid not in firing_cap: continue
            if subs<=avail:
                for p in pros:
                    if p not in avail: avail.add(p); changed=True
    return avail

# ---- propagate the 'expressed' set itself (regulatory closure) ----
def expressed_closure(knocked):
    """Iterate: a gene is expressed if not knocked AND its regulators (in the
    expressed set) satisfy the rule. Most regs are themselves regulated."""
    all_bnums={g.id for g in M.genes} | set().union(*activators.values(), *repressors.values(), *sigmas.values())
    alive=set(all_bnums) - set(knocked)
    for _ in range(8):   # few rounds suffice; not infinite to avoid pathological
        new=set()
        for b in alive:
            if gene_expressed(b, alive): new.add(b)
        if new==alive: break
        alive=new
    return alive

# ---- intact baseline ----
intact_expressed=expressed_closure(set())
print(f"  intact expressed genes: {len(intact_expressed)} / {len(M.genes)}  "
      f"(TF genes alive: {sum(1 for b in intact_expressed if b in {tf_to_b[t] for t in tf_set})})", flush=True)
intact_reach=reach(intact_expressed)
intact_bio_reachable={p for p in bio_pre if p in intact_reach}
print(f"  intact reachable biomass precursors: {len(intact_bio_reachable)}/{len(bio_pre)}", flush=True)

# Universe of genes to test: model genes + TF/sigma b-numbers
b_to_keio={r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
truth={r["locus_tag"]:int(r["essential"]) for r in csv.DictReader(open("data/drive_import/labels/labels.csv")) if r["organism"]=="beril_Keio"}
clade_of={r["organism"]:r["clade"] for r in csv.DictReader(open("data/drive_import/labels/clade_splits.csv"))}
kclade=clade_of.get("beril_Keio","escherichia")
P=np.load(OUT/"af_torch_preds_aug.npz",allow_pickle=True)
_cl,_lt,_p=P["clade"],P["lt"],P["p"]
w1={}
for i in range(len(_p)):
    if str(_cl[i])==kclade and not np.isnan(_p[i]): w1[str(_lt[i])]=float(_p[i])

tf_bnums={tf_to_b[t] for t in tf_set if tf_to_b[t]}
test_bnums={g.id for g in M.genes} | tf_bnums
test_genes=[(b, b_to_keio.get(b), truth.get(b_to_keio.get(b)), w1.get(b_to_keio.get(b)))
            for b in test_bnums]
test_genes=[(b,lt,t,w) for (b,lt,t,w) in test_genes if lt is not None and t is not None]
print(f"\n  test universe: {len(test_genes)} genes joinable to Keio truth", flush=True)

# ---- run knockouts (valve-graph) ----
print("  running valve-graph knockouts...", flush=True)
t1=time.time()
valve_ess=set()
why={}
n_done=0
for b,lt,t,w in test_genes:
    ex=expressed_closure({b})
    if b not in ex: pass  # consistent
    rk=reach(ex)
    lost=[p for p in intact_bio_reachable if p not in rk]
    if lost:
        valve_ess.add(b); why[b]=M.metabolites.get_by_id(lost[0]).name
    n_done+=1
    if n_done%200==0: print(f"    {n_done}/{len(test_genes)} ({time.time()-t1:.0f}s)", flush=True)
print(f"  valve-essential genes: {len(valve_ess)}  ({time.time()-t1:.0f}s)", flush=True)

# ---- FBA baseline ----
print("\n  FBA baseline (for comparison)...", flush=True)
fba_ess=set()
sd=single_gene_deletion(M)
for _,row in sd.iterrows():
    gid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
    if row["growth"]<0.01*g0 or np.isnan(row["growth"]): fba_ess.add(gid)
print(f"  FBA-essential: {len(fba_ess)}", flush=True)

# ---- categorize genes and score ----
transp_genes=set()
for R in M.reactions:
    if R.id.startswith(("EX_","DM_","SK_")) or "BIOMASS" in R.id.upper(): continue
    # transport if any reactant in compartment != any product compartment
    comps=set()
    for m in R.metabolites: comps.add(m.id.rsplit("_",1)[-1])
    if len(comps)>1:
        for g in R.genes: transp_genes.add(g.id)
def category(b):
    if b in tf_bnums: return "regulator"
    if b in transp_genes: return "transporter"
    if b in {g.id for g in M.genes}: return "metabolic"
    return "other"

cats=defaultdict(lambda: dict(n=0,ess=0,fba_call=0,fba_tp=0,valve_call=0,valve_tp=0))
for (b,lt,t,w) in test_genes:
    c=category(b)
    d=cats[c]; d["n"]+=1; d["ess"]+=(t==1)
    if b in fba_ess:
        d["fba_call"]+=1; d["fba_tp"]+=(t==1)
    if b in valve_ess:
        d["valve_call"]+=1; d["valve_tp"]+=(t==1)

print("\n=== precision per category ===")
print(f"{'category':<14}{'n':>6}{'ess':>6}  | {'FBA: call P R':<22} | {'VALVE: call P R':<22}")
for c in ("metabolic","transporter","regulator","other"):
    d=cats[c]
    fP=d["fba_tp"]/max(d["fba_call"],1); fR=d["fba_tp"]/max(d["ess"],1)
    vP=d["valve_tp"]/max(d["valve_call"],1); vR=d["valve_tp"]/max(d["ess"],1)
    print(f"  {c:<12}{d['n']:>6}{d['ess']:>6}  | {d['fba_call']:>4} {fP:.2f} {fR:.2f}        "
          f"| {d['valve_call']:>4} {vP:.2f} {vR:.2f}")

# overall + by category JSON
all_b=[b for b,_,_,_ in test_genes]; all_t={b:t for b,lt,t,_ in test_genes}
def stats(callset):
    tp=sum(1 for b in callset if all_t.get(b)==1); called=sum(1 for b in callset if b in all_t)
    pos=sum(1 for v in all_t.values() if v==1)
    return dict(called=called, tp=tp, P=round(tp/max(called,1),3), R=round(tp/max(pos,1),3))

overall_fba=stats(fba_ess); overall_valve=stats(valve_ess)
print(f"\n=== OVERALL on {len(test_genes)} genes ===")
print(f"  FBA   : called={overall_fba['called']} P={overall_fba['P']} R={overall_fba['R']}")
print(f"  VALVE : called={overall_valve['called']} P={overall_valve['P']} R={overall_valve['R']}")
print(f"  valve∩FBA={len(valve_ess&fba_ess)}  valve-only={len(valve_ess-fba_ess)}  FBA-only={len(fba_ess-valve_ess)}")

# new essentials valve catches that FBA misses (broken down)
new_calls=valve_ess - fba_ess
new_in_truth=[b for b in new_calls if all_t.get(b)==1]
print(f"\n  valve-only true positives: {len(new_in_truth)} (of {len(new_calls)} valve-only calls)")
by_cat=defaultdict(int); by_cat_tp=defaultdict(int)
for b in new_calls:
    c=category(b); by_cat[c]+=1
    if all_t.get(b)==1: by_cat_tp[c]+=1
for c in ("metabolic","transporter","regulator","other"):
    print(f"    {c:<14} valve-only calls={by_cat[c]:<4} TP={by_cat_tp[c]:<4} P={by_cat_tp[c]/max(by_cat[c],1):.2f}")

# examples of true new catches by category
print("\n  example valve-only TRUE essentials:")
for c in ("regulator","transporter"):
    examples=[b for b in new_in_truth if category(b)==c][:5]
    for b in examples:
        nm=gene_name_lower.get(b,"?")
        print(f"    [{c}] {b} ({nm}) -> lost: {why.get(b,'?')}")

json.dump(dict(model=MODEL, n_genes=len(test_genes),
   tf_count=len(tf_set), tf_mapped=len(tf_bnums),
   fba_overall=overall_fba, valve_overall=overall_valve,
   per_category={c: dict(n=cats[c]["n"], ess=cats[c]["ess"],
                         fba_call=cats[c]["fba_call"], fba_tp=cats[c]["fba_tp"],
                         valve_call=cats[c]["valve_call"], valve_tp=cats[c]["valve_tp"])
                 for c in ("metabolic","transporter","regulator","other")},
   valve_only_calls=len(new_calls), valve_only_tp=len(new_in_truth),
   valve_only_by_cat={c: dict(calls=by_cat[c], tp=by_cat_tp[c]) for c in by_cat}),
   open(OUT/"valve_graph.json","w"),indent=2)
print(f"\nwrote outputs/orphan/valve_graph.json ({time.time()-t0:.0f}s total)", flush=True)
