"""Regulatory-FBA hybrid: regulators as valves, FBA as the downstream oracle.

Your idea: a gene's product is a valve; knocking it out stops downstream flow
only if there's no parallel path. v1 failed for two reasons (both fixed here):
  - TF->b-number mapping was broken (TFs aren't in the metabolic model). FIXED:
    use RegulonDB GeneProductSet.txt -> genome-wide symbol->b-number.
  - pure-graph reachability can't bootstrap cofactors. FIXED: use FBA as the
    'did the cell still work' oracle (the one irreducible piece).

Mechanism for a knockout of gene g:
  1. Regulatory closure: start {g}. A target gene becomes OFF if ALL its
     activators are OFF, or its sigma factor is OFF, or it is fully repressed-
     -derepressed logic ignored (we only add genes that LOSE required activation).
     Iterate (a TF can regulate another TF).
  2. Map the OFF set to the metabolic model and knock those reactions out (GPR).
  3. FBA: growth < 1% of baseline -> g is ESSENTIAL (directly or via its regulon).

This lets a regulator be called essential when its regulon contains essential
metabolic genes with no alternate activator -- exactly the genes plain single-
gene FBA (TF not in model) and conservation both miss.

Validate vs Keio truth, by category (metabolic/transporter/regulator), vs plain
single-gene FBA. Output: outputs/orphan/regulatory_fba.json
"""
import csv, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
cobra.Configuration().solver="glpk"
OUT=Path("outputs/orphan"); MODELS=Path("data/external_data/bigg_models"); REG=Path("data/external_data/regulon")
MODEL="iML1515"

RICH=["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L","gly","his__L","ile__L","leu__L",
      "lys__L","met__L","phe__L","pro__L","ser__L","thr__L","trp__L","tyr__L","val__L","ade","gua","ura",
      "thymd","cytd","ins","adn","gsn","uri","thm","ribflv","nac","pnto__R","pydx","4abz","btn","fol"]
print(f"loading {MODEL}...", flush=True); t0=time.time()
M=cobra.io.read_sbml_model(str(MODELS/f"{MODEL}.xml.gz"))
for x in RICH:
    rid=f"EX_{x}_e"
    if rid in M.reactions: M.reactions.get_by_id(rid).lower_bound=-10
g0=M.slim_optimize()
print(f"  rich growth g0={g0:.3f}", flush=True)

# ---- genome-wide symbol -> b-number (GeneProductSet: col1 name, col2 bnum) ----
sym2b={}
for l in open(REG/"GeneProductSet.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t")
    if len(p)>=3 and p[2].startswith("b") and p[1].strip():
        sym2b[p[1].strip().lower()]=p[2].strip()
print(f"  symbol->bnumber map: {len(sym2b)} genes", flush=True)

# ---- RegulonDB edges -> per-target activators/repressors/sigmas (as b-numbers) ----
def load(path):
    out=[]
    for l in open(path):
        if not l.strip() or l.startswith("#"): continue
        p=l.rstrip("\n").split("\t")
        if len(p)>=3: out.append((p[0], p[1], p[2].lower()))
    return out
tf_edges=load(REG/"network_tf_gene.txt"); sig_edges=load(REG/"network_sigma_gene.txt")
activators=defaultdict(set); repressors=defaultdict(set); sigmas=defaultdict(set)
mapped_tf=set()
for tf,tgt,eff in tf_edges:
    tb=sym2b.get(tf.lower()); gb=sym2b.get(tgt.lower())
    if not tb or not gb: continue
    mapped_tf.add(tb)
    if eff in ("activator","dual"): activators[gb].add(tb)
    if eff in ("repressor","dual"): repressors[gb].add(tb)
for tf,tgt,eff in sig_edges:
    tb=sym2b.get(tf.lower()); gb=sym2b.get(tgt.lower())
    if not tb or not gb: continue
    sigmas[gb].add(tb)
regulated=set(activators)|set(repressors)|set(sigmas)
print(f"  RegulonDB mapped: {len(mapped_tf)} TF/sigma genes; {len(regulated)} regulated target genes "
      f"({len(tf_edges)+len(sig_edges)} raw edges)", flush=True)

# ---- regulatory closure: genes turned OFF by a knockout ----
def reg_off(seed):
    off=set(seed)
    for _ in range(10):
        new=set(off)
        for tgt in regulated:
            if tgt in new: continue
            acts=activators.get(tgt,set()); sigs=sigmas.get(tgt,set())
            # OFF if it HAS activators and all are off, OR has sigma and all off
            if acts and acts<=off: new.add(tgt); continue
            if sigs and sigs<=off: new.add(tgt); continue
        if new==off: break
        off=new
    return off

# ---- baseline single-gene FBA (oracle) ----
print("  baseline single-gene FBA...", flush=True)
sd=single_gene_deletion(M)
fba_ess=set();
for _,row in sd.iterrows():
    gid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
    if row["growth"]<0.01*g0 or np.isnan(row["growth"]): fba_ess.add(gid)
print(f"  FBA-essential (single gene): {len(fba_ess)}", flush=True)
model_genes={g.id for g in M.genes}

# ---- test universe: model genes + mapped TFs, joinable to truth ----
b_to_keio={r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
truth={r["locus_tag"]:int(r["essential"]) for r in csv.DictReader(open("data/drive_import/labels/labels.csv")) if r["organism"]=="beril_Keio"}
def tr(b):
    lt=b_to_keio.get(b); return truth.get(lt) if lt else None
test=sorted((model_genes|mapped_tf))
test=[b for b in test if tr(b) is not None]
print(f"  test universe: {len(test)} genes joinable to truth "
      f"({sum(1 for b in test if b in mapped_tf and b not in model_genes)} are regulators not in model)", flush=True)

# ---- regulatory-FBA knockout per gene ----
print("  running regulatory-FBA knockouts...", flush=True)
t1=time.time()
reg_fba_ess=set(); reg_regulon_size={}
n=0
for b in test:
    off = reg_off({b})
    off_in_model = off & model_genes
    if not off_in_model:
        # gene not in model and regulates no model gene -> FBA can't see it
        n+=1; continue
    with M:
        for gid in off_in_model:
            try: M.genes.get_by_id(gid).knock_out()
            except KeyError: pass
        gr=M.slim_optimize()
    if gr is None or gr<0.01*g0 or np.isnan(gr):
        reg_fba_ess.add(b); reg_regulon_size[b]=len(off_in_model)
    n+=1
    if n%200==0: print(f"    {n}/{len(test)} ({time.time()-t1:.0f}s)", flush=True)
print(f"  regulatory-FBA essential: {len(reg_fba_ess)}  ({time.time()-t1:.0f}s)", flush=True)

# ---- categorize ----
transp=set()
for R in M.reactions:
    if R.id.startswith(("EX_","DM_","SK_")) or "BIOMASS" in R.id.upper(): continue
    if len({m.id.rsplit("_",1)[-1] for m in R.metabolites})>1:
        for g in R.genes: transp.add(g.id)
def cat(b):
    if b in mapped_tf and b not in model_genes: return "regulator(off-model)"
    if b in mapped_tf: return "regulator(in-model)"
    if b in transp: return "transporter"
    if b in model_genes: return "metabolic"
    return "other"

def stats(callset, universe):
    tp=sum(1 for b in callset if tr(b)==1 and b in universe)
    called=sum(1 for b in callset if b in universe)
    pos=sum(1 for b in universe if tr(b)==1)
    return dict(called=called, tp=tp, P=round(tp/max(called,1),3), R=round(tp/max(pos,1),3), pos=pos)

U=set(test)
o_fba=stats(fba_ess,U); o_reg=stats(reg_fba_ess,U)
print(f"\n=== OVERALL ({len(test)} genes) ===")
print(f"  plain FBA      : called={o_fba['called']} P={o_fba['P']} R={o_fba['R']}")
print(f"  regulatory-FBA : called={o_reg['called']} P={o_reg['P']} R={o_reg['R']}")
new=reg_fba_ess - fba_ess
new_tp=[b for b in new if tr(b)==1]
print(f"  reg-only calls: {len(new)}  true-positive: {len(new_tp)}  (P={len(new_tp)/max(len(new),1):.2f})")

print(f"\n=== by category (regulatory-FBA only NEW calls vs plain FBA) ===")
bycat=defaultdict(lambda:[0,0])
for b in new:
    c=cat(b); bycat[c][0]+=1; bycat[c][1]+=(tr(b)==1)
for c,(cc,tp) in sorted(bycat.items(), key=lambda x:-x[1][0]):
    print(f"  {c:<22} new_calls={cc:<4} TP={tp:<4} P={tp/max(cc,1):.2f}")

print(f"\n  example regulator essentials newly caught (reg-FBA, truth-essential):")
shown=0
for b in new_tp:
    if "regulator" in cat(b):
        # name
        nm=next((k for k,v in sym2b.items() if v==b), b)
        print(f"    {b} ({nm})  regulon-in-model={reg_regulon_size.get(b,0)}  cat={cat(b)}")
        shown+=1
        if shown>=10: break
if shown==0: print("    (none -- regulators did not become essential via their regulon)")

json.dump(dict(model=MODEL, n_test=len(test),
   symbol_map=len(sym2b), tf_mapped=len(mapped_tf), regulated_genes=len(regulated),
   plain_fba=o_fba, regulatory_fba=o_reg,
   reg_only_calls=len(new), reg_only_tp=len(new_tp),
   by_category={c:dict(new_calls=v[0],tp=v[1]) for c,v in bycat.items()}),
   open(OUT/"regulatory_fba.json","w"),indent=2)
print(f"\nwrote outputs/orphan/regulatory_fba.json ({time.time()-t0:.0f}s)", flush=True)
