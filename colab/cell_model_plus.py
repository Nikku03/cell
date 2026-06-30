"""'Better than a simulator': one perturbation propagates through ALL layers.
  ESSENTIAL CORE : non-metabolic essentials (ribosome/RNAP/replication/tRNA-synth)
                   -> catches what FBA misses (recall up)
  REGULATORY->METABOLIC : KO a TF -> its sole-activated targets are forced OFF
                   (fixpoint cascade through TF->TF) -> those reactions blocked in FBA
  METABOLIC      : FBA on iJO1366, condition-dependent
Verdict = lethal if (FBA-dead with forced-off genes blocked) OR (KO hits essential core).
Validated vs Keio; head-to-head with the FBA-only simulator.
"""
import csv, json, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
warnings.filterwarnings("ignore"); cobra.Configuration().solver="glpk"
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan"); MODELS=Path("data/external_data/bigg_models")
m=cobra.io.read_sbml_model(str(MODELS/"iJO1366.xml.gz")); model_b={g.id for g in m.genes}
CARBON={"glucose":"EX_glc__D_e","arabinose":"EX_arab__L_e","glycerol":"EX_glyc_e","maltose":"EX_malt_e","succinate":"EX_succ_e","acetate":"EX_ac_e"}
exset={r.id for r in m.reactions}

name2b={};
genes_prod={}
for r in csv.DictReader(open(REG/"TRN.csv")):
    if r.get("gene_name") and r.get("gene_id"): name2b[r["gene_name"].lower()]=r["gene_id"]
for l in open(REG/"GeneProductSet.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=3: genes_prod[p[1].lower()]=p[2].lower()
# regulatory network + activators map
activators=defaultdict(set); repressors=defaultdict(set); is_tf=set()
for l in open(REG/"network_tf_gene.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.split("\t"); tf,tg,s=p[0].lower(),p[1].lower(),p[2].lower(); is_tf.add(tf)
    if s=="activator": activators[tg].add(tf)
    elif s=="repressor": repressors[tg].add(tf)
# ESSENTIAL CORE (informational/universal machinery) by product keyword + name prefix
KW=["ribosomal protein","rna polymerase","dna polymerase","trna ligase","aminoacyl","dna gyrase",
    "dna topoisomerase","primase","replicative dna helicase","elongation factor","initiation factor",
    "peptide chain release","rrna","trna ","ribosome","dna primase","cell division protein fts"]
PFX=("rpl","rps","rpm","rpo","tuf","fus","frr","infa","infb","infc","prf","dnaa","dnab","dnae","dnag","dnan","dnax","gyra","gyrb","nusa","nusg","lig")
core_names=set()
for g,prod in genes_prod.items():
    if any(k in prod for k in KW) or g.startswith(PFX): core_names.add(g)
core_b=set(name2b[g] for g in core_names if g in name2b)
print(f"essential core (informational machinery): {len(core_names)} genes, {len(core_b)} mapped to b-numbers")

def set_medium(mm,carbon):
    med=mm.medium.copy()
    for cs,ex in CARBON.items():
        if ex in exset: med[ex]= (10.0 if cs==carbon else 0.0)
    mm.medium=med
def tf_active(tf,carbon):
    """condition-dependent activity for global regulators (conserved effector logic).
    None = unknown -> we do NOT force-off from it (avoid over-prediction)."""
    if tf=="crp": return carbon!="glucose"      # CRP active only when glucose absent
    if tf=="fnr": return False                  # aerobic assumption
    return None
def forced_off(ko_names,carbon):
    """seed only from KO'd TFs that were ACTIVE in this condition; then fixpoint:
    a gene is OFF if it has activators and ALL of them are OFF."""
    seed=set(g for g in ko_names if tf_active(g,carbon) is True)
    off=set(seed); changed=True
    while changed:
        changed=False
        for g,acts in activators.items():
            if g not in off and acts and acts<=off:
                off.add(g); changed=True
    return off
def simulate(knockouts=(), carbon="glucose", mutate=None):
    ko_names=[g.lower() for g in knockouts]
    off=forced_off(ko_names,carbon)               # regulatory cascade -> forced-off genes
    block_b=set()
    for g in off:
        b=name2b.get(g)
        if b in model_b: block_b.add(b)
    for g in knockouts:
        b=g if g in model_b else name2b.get(g.lower())
        if b in model_b: block_b.add(b)
    with m as mm:
        set_medium(mm,carbon); wt=mm.slim_optimize()
        for b in block_b:
            try: mm.genes.get_by_id(b).knock_out()
            except KeyError: pass
        if mutate:
            for g,frac in mutate.items():
                b=g if g in model_b else name2b.get(g.lower())
                if b in model_b:
                    for rx in mm.genes.get_by_id(b).reactions: rx.lower_bound*=frac; rx.upper_bound*=frac
        gr=mm.slim_optimize(); gr=float(gr) if gr and gr==gr else 0.0
    core_hit=[g for g in knockouts if name2b.get(g.lower()) in core_b]
    fba_dead=gr<0.01*wt
    lethal = fba_dead or bool(core_hit)
    reason = ("essential-core:"+",".join(core_hit)) if core_hit else ("metabolic" if fba_dead else "viable")
    return dict(carbon=carbon,knockouts=list(knockouts),growth=round(gr,3),viable=not lethal,
                reason=reason, reg_forced_off=len(off)-len(ko_names),
                metabolic_off=sorted(block_b-{name2b.get(g.lower()) for g in knockouts})[:8])

# ===== VALIDATION: head-to-head FBA-only vs +essential-core, vs Keio =====
b_of={r["locus_tag"]:r["b_number"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
ess_b={}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"]=="beril_Keio":
        b=b_of.get(r["locus_tag"]);
        if b: ess_b[b]=int(r["essential"])
with m as mm:
    set_medium(mm,"glucose"); wt=mm.slim_optimize(); df=single_gene_deletion(mm,processes=1)
fba_lethal=set()
for _,row in df.iterrows():
    ids=row["ids"]; b=list(ids)[0] if isinstance(ids,(set,frozenset)) else ids
    gr=row["growth"]
    if gr is None or (isinstance(gr,float) and np.isnan(gr)) or gr<0.01*wt: fba_lethal.add(b)
uni=list(ess_b); truth=set(b for b in uni if ess_b[b]==1)   # FULL genome with Keio truth
def metr(pred):
    P=pred&set(uni); tp=len(P&truth); fp=len(P-truth); fn=len(truth-P)
    return dict(precision=round(tp/max(tp+fp,1),3),recall=round(tp/max(tp+fn,1),3),
               accuracy=round(sum(1 for b in uni if (b in pred)==(b in truth))/len(uni),3))
print("\nVALIDATION vs Keio (single-gene KO, glucose):")
print(f"  FBA only            : {metr(fba_lethal)}")
print(f"  FBA + essential core: {metr(fba_lethal|core_b)}")

print("\n=== worked perturbations (showing the upgrades) ===")
def show(r):
    v="VIABLE" if r["viable"] else "*** LETHAL ***"
    extra=f"  [reg cascade forced off {r['reg_forced_off']} genes; metabolic-off e.g. {r['metabolic_off'][:4]}]" if r['reg_forced_off'] else ""
    print(f"  KO {r['knockouts']} on {r['carbon']}: growth {r['growth']} -> {v}  ({r['reason']}){extra}")
show(simulate(["rpsL"]))                        # ribosomal -> now LETHAL via core (FBA-only missed)
show(simulate(["pyrB"]))                         # metabolic essential
show(simulate(["crp"],carbon="arabinose"))      # TF KO -> forced-off araBAD -> LETHAL (FBA-only said viable)
show(simulate(["crp"],carbon="glucose"))        # CRP KO on glucose -> still viable
show(simulate(["sdhA"],carbon="succinate"))     # conditional
show(simulate(["dnaA"]))                         # replication TF -> core lethal
json.dump(dict(fba_only=metr(fba_lethal),fba_plus_core=metr(fba_lethal|core_b),
               n_core=len(core_b)),open(OUT/"cell_model_plus.json","w"),indent=2)
print(f"\nwrote {OUT}/cell_model_plus.json")
