"""Gap-directed Wheel 2 for UNMODELED organisms.

Unmodeled orgs have NO metabolic model and NO EC numbers in their annotations,
but 100% orthogroup coverage. So the gap-fill transfers via the orthogroup:

  1. From the 4 native models, find HOLES (essential reactions). Each hole's
     true gene belongs to an orthogroup OG -> that OG is a "hole-OG", and it
     carries the hole's DIMENSIONS (EC, cofactor, direction, substrate size).
  2. An unmodeled organism's gene sitting in a hole-OG is the ORTHOLOG of an
     enzyme that fills an essential metabolic hole -> predicted essential, with
     its dimensions inherited from the model (no local EC needed).
  3. MEASURE precision + coverage of this hole-OG call, and -- the crucial
     honesty check -- whether it ADDS anything over conservation (Wheel 2),
     or just re-finds the conserved core (as metabolic_transfer.py did).
  4. Report the experimental product: per organism, how many essential
     reactions get a UNIQUE ortholog pinpointed (the gap-directed shortlist),
     and how many holes have NO ortholog = candidate non-orthologous
     displacement (where a different gene family fills the same hole).

Output: outputs/orphan/wheel2_unmodeled.json
"""
import csv, json, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_reaction_deletion
cobra.Configuration().solver = "glpk"
OUT = Path("outputs/orphan"); MODELS = Path("data/external_data/bigg_models")

RICH=["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L","gly","his__L",
      "ile__L","leu__L","lys__L","met__L","phe__L","pro__L","ser__L","thr__L","trp__L",
      "tyr__L","val__L","ade","gua","ura","thymd","cytd","ins","adn","gsn","uri","thm",
      "ribflv","nac","pnto__R","pydx","4abz","btn","fol"]
CURRENCY={"h2o","h","pi","ppi","atp","adp","amp","gtp","gdp","gmp","utp","udp","ump",
          "ctp","cdp","cmp","nad","nadh","nadp","nadph","fad","fadh2","fmn","fmnh2",
          "coa","co2","o2","nh4","hco3","so4","h2s","na1","k","cl","mg2","mn2","fe2",
          "fe3","ca2","cobalt2","cu2","zn2","mobd","h2","q8","q8h2","mqn8","mql8",
          "2dmmql8","2dmmq8","thf","mlthf","methf","5mthf","10fthf","dhf","h2o2","amet","ahcys"}
REDUCED={"nadh","nadph","fadh2","fmnh2","q8h2","mql8","2dmmql8"}
def base(mid): return mid.rsplit("_",1)[0]
def carbons(m):
    f=m.formula or ""; mt=re.search(r"C(\d+)",f); return int(mt.group(1)) if mt else 0
def cofactor(bs):
    if {"nadh","nad"}&bs: return "NAD"
    if {"nadph","nadp"}&bs: return "NADP"
    if {"fadh2","fad","fmn","fmnh2"}&bs: return "FAD/FMN"
    if {"atp","adp","amp","gtp","gdp"}&bs: return "ATP/NTP"
    if {"coa"}&bs: return "CoA"
    return "none"
def direction(R):
    if R.reversibility: return "reversible"
    for m,c in R.metabolites.items():
        if base(m.id) in REDUCED: return "oxidation" if c>0 else "reduction"
    return "n/a"
def size_bucket(R):
    subs=[carbons(m) for m in R.metabolites if base(m.id) not in CURRENCY]
    mx=max(subs) if subs else 0
    return "small" if mx<=4 else "med" if mx<=10 else "large"
def ec_of(R):
    e=R.annotation.get("ec-code")
    return (e[0] if isinstance(e,list) else e) if e else None
def ec3(R):
    e=ec_of(R)
    if not e: return None
    p=e.split("."); return ".".join(p[:3]) if len(p)>=3 else e
def shape(R):
    bs={base(m.id) for m in R.metabolites}
    return dict(ec=ec_of(R), ec3=ec3(R), cof=cofactor(bs), dirn=direction(R), size=size_bucket(R))

# ---- shared label data ----
ALL=list(csv.DictReader(open("data/drive_import/labels/labels.csv")))
clade_of={r["organism"]:r["clade"] for r in csv.DictReader(open("data/drive_import/labels/clade_splits.csv"))}
og_of={(r["organism"],r["locus_tag"]):r["og_id"]
       for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")) if r.get("og_id")}
truth_by={(r["organism"],r["locus_tag"]):int(r["essential"]) for r in ALL}

# bridges model-gene -> our locus
b_to_keio={r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
mtub_og={}
for r in csv.DictReader(open("colab/work/og_assignments.csv")):
    if r["organism"].startswith("deg_Mycobacterium_tuberculosis_H37Rv"):
        mtub_og.setdefault(r["locus_tag"], r["og_id"])

def model_gene_to_og(model_id, M):
    """returns dict model_gene_id -> og_id for the 4 native models."""
    out={}
    if model_id=="iJN1463":
        for g in M.genes:
            o=og_of.get(("beril_Putida", g.id))
            if o: out[g.id]=o
    elif model_id=="iML1515":
        for g in M.genes:
            lt=b_to_keio.get(g.id); o=og_of.get(("beril_Keio", lt)) if lt else None
            if o: out[g.id]=o
    elif model_id=="iEK1008":
        for g in M.genes:
            o=mtub_og.get(g.id) or og_of.get(("mtub", g.id))
            if o: out[g.id]=o
    elif model_id=="iYL1228":
        # gene-name bridge
        koxy_annot={r["locus_tag"]:r["product"]
                    for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv"))
                    if r["organism"]=="beril_Koxy" and r.get("product")}
        tok=defaultdict(list)
        for lt,prod in koxy_annot.items():
            for t in re.findall(r"[A-Za-z0-9]{4,}",prod): tok[t.lower()].append(lt)
        for g in M.genes:
            gn=(g.name or "").strip()
            if gn and g.id!=gn:
                mm=tok.get(gn.lower(),[])
                if len(mm)==1:
                    o=og_of.get(("beril_Koxy", mm[0]))
                    if o: out[g.id]=o
    return out

# ---- Step 1: build the hole catalog (OG -> dimensions) from 4 models ----
print("Step 1: FBA holes -> hole-OG catalog", flush=True); t0=time.time()
hole_og=set(); og_dims={}     # og -> representative dims (first seen)
og_hole_models=defaultdict(set)
for model_id in ["iJN1463","iML1515","iEK1008","iYL1228"]:
    M=cobra.io.read_sbml_model(str(MODELS/f"{model_id}.xml.gz"))
    for x in RICH:
        rid=f"EX_{x}_e"
        if rid in M.reactions: M.reactions.get_by_id(rid).lower_bound=-10
    g0=M.slim_optimize()
    g2og=model_gene_to_og(model_id, M)
    metabolic=[R for R in M.reactions
               if not R.id.startswith(("EX_","DM_","SK_")) and "BIOMASS" not in R.id.upper() and R.genes]
    de=single_reaction_deletion(M, reaction_list=[R.id for R in metabolic])
    ess=set()
    for _,row in de.iterrows():
        rid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
        if row["growth"]<0.01*g0 or np.isnan(row["growth"]): ess.add(rid)
    n_holes=0
    for R in metabolic:
        if R.id not in ess: continue
        s=shape(R)
        for g in R.genes:
            o=g2og.get(g.id)
            if o:
                hole_og.add(o); og_hole_models[o].add(model_id)
                if o not in og_dims: og_dims[o]=s
                n_holes+=1
    print(f"  {model_id}: g0={g0:.2f} ess_rxn={len(ess)} hole-OG hits={n_holes} "
          f"(total hole-OGs so far={len(hole_og)})", flush=True)
print(f"  hole-OG catalog: {len(hole_og)} orthogroups ({time.time()-t0:.0f}s)\n", flush=True)

# ---- conservation (leak-clean, per held clade) ----
def cons_rate(held):
    n=defaultdict(int); e=defaultdict(int)
    for r in ALL:
        if clade_of.get(r["organism"])==held: continue
        o=og_of.get((r["organism"],r["locus_tag"]))
        if o: n[o]+=1; e[o]+=int(r["essential"])
    return {o:e[o]/n[o] for o in n if n[o]>=5}

def cov_at_p(scores, ys, target=0.90, min_calls=10):
    s=np.array(scores); y=np.array(ys); od=np.argsort(-s); yo=y[od]
    tp=np.cumsum(yo); n=np.arange(1,len(yo)+1); prec=tp/n; pos=y.sum()
    valid=(prec>=target)&(n>=min_calls)
    if not valid.any(): return (0.0,0.0,0)
    k=np.max(np.where(valid)[0]); return (float(prec[k]), float(tp[k]/max(pos,1)), int(n[k]))

# ---- Step 2-4: per unmodeled organism ----
MODELED15={"beril_Putida","beril_Keio","ecoli_BW25113_tradis","mtub","beril_Koxy",
           "beril_WCS417","beril_PS","beril_psRCH2","beril_SyringaeB728a",
           "beril_SyringaeB728a_mexBdelta","beril_pseudo1_N1B4","beril_pseudo3_N2E3",
           "beril_pseudo5_N2C3_1","beril_pseudo6_N2E2","beril_pseudo13_GW456_L13"}
orgs=sorted({o for (o,_) in truth_by} - MODELED15)
pergene_dump=[]   # (organism, locus_tag, truth, og, w2, hole) for independent verification
print("Step 2-4: hole-OG transfer on unmodeled organisms", flush=True)
print(f"{'organism':<22}{'N':>6}{'holeOG':>10}{'P_hole':>8}{'covHole':>9}"
      f"{'W2cov':>9}{'best+':>9}{'dCov':>7}{'P_redu':>8}{'P_orth':>8}{'n_orth':>7}", flush=True)
rows=[]
for org in orgs:
    held=clade_of.get(org,"?"); rate=cons_rate(held)
    genes=[(lt) for (o,lt) in truth_by if o==org]
    if len(genes)<200: continue
    # per-gene: truth, og, w2, hole?
    recs=[]
    for lt in genes:
        og=og_of.get((org,lt))
        if not og: continue
        recs.append(dict(lt=lt, truth=truth_by[(org,lt)], og=og,
                         w2=rate.get(og, 0.0), hole=(og in hole_og)))
    if len(recs)<200: continue
    for r in recs:
        pergene_dump.append((org, r["lt"], r["truth"], r["og"], round(r["w2"],4), int(r["hole"])))
    N=len(recs); pos=sum(r["truth"] for r in recs)
    ys=[r["truth"] for r in recs]
    # hole-OG call alone
    hole_pos=[r for r in recs if r["hole"]]
    P_hole=sum(r["truth"] for r in hole_pos)/max(len(hole_pos),1)
    cov_hole=sum(r["truth"] for r in hole_pos)/max(pos,1)
    # W2 alone essCov@P0.90
    _,w2cov,_=cov_at_p([r["w2"] for r in recs],ys)
    # FAIR combine: soft bonus b added to hole genes; sweep b, take BEST essCov@P0.90.
    # (a hard override is wrong because hole alone is only ~0.58 precision)
    best_cov=w2cov; best_b=0.0
    for b in [0.0,0.05,0.1,0.15,0.2,0.3,0.5]:
        comb=[r["w2"]+(b if r["hole"] else 0.0) for r in recs]
        pC,covC,_=cov_at_p(comb,ys)
        if covC>best_cov: best_cov=covC; best_b=b
    # HONEST: a single FIXED global bonus (no per-org selection) avoids in-sample bias
    comb_fixed=[r["w2"]+(0.10 if r["hole"] else 0.0) for r in recs]
    _,fixed_cov,_=cov_at_p(comb_fixed,ys)
    # redundancy: precision of hole genes that conservation ALREADY ranks high vs the ones it misses
    hole_hi=[r for r in hole_pos if r["w2"]>=0.5]
    hole_lo=[r for r in hole_pos if r["w2"]<0.5]
    P_hole_hi=sum(r["truth"] for r in hole_hi)/max(len(hole_hi),1)   # redundant part
    P_hole_lo=sum(r["truth"] for r in hole_lo)/max(len(hole_lo),1)   # orthogonal part
    rows.append(dict(organism=org, N=N, ess=pos, holeOG_hits=len(hole_pos),
                     P_hole=round(P_hole,3), cov_hole=round(cov_hole,3),
                     w2_essCov=round(w2cov,3), best_combine_essCov=round(best_cov,3),
                     fixed_combine_essCov=round(fixed_cov,3), dCov_fixed=round(fixed_cov-w2cov,3),
                     best_bonus=best_b, dCov=round(best_cov-w2cov,3),
                     P_hole_redundant=round(P_hole_hi,3), n_redundant=len(hole_hi),
                     P_hole_orthogonal=round(P_hole_lo,3), n_orthogonal=len(hole_lo)))
    print(f"  {org:<20}{N:>6}{len(hole_pos):>10}{P_hole:>8.3f}{cov_hole:>9.3f}"
          f"{w2cov:>9.3f}{best_cov:>9.3f}{best_cov-w2cov:>+7.3f}"
          f"{P_hole_hi:>8.3f}{P_hole_lo:>8.3f}{len(hole_lo):>7}", flush=True)

mP=np.mean([r["P_hole"] for r in rows]); mCov=np.mean([r["cov_hole"] for r in rows])
mW2=np.mean([r["w2_essCov"] for r in rows]); mC=np.mean([r["best_combine_essCov"] for r in rows])
mFixed=np.mean([r["fixed_combine_essCov"] for r in rows])
mRedu=np.mean([r["P_hole_redundant"] for r in rows]); mOrth=np.mean([r["P_hole_orthogonal"] for r in rows])
mOrthN=np.mean([r["n_orthogonal"] for r in rows])
n_help=sum(1 for r in rows if r["dCov"]>0.005)
n_help_fixed=sum(1 for r in rows if r["dCov_fixed"]>0.005)
n_hurt_fixed=sum(1 for r in rows if r["dCov_fixed"]<-0.005)
print(f"\n=== MEAN over {len(rows)} unmodeled orgs ===")
print(f"  hole-OG call alone:   precision={mP:.3f} (vs ~0.32 base = {mP/0.32:.1f}x)  catches {mCov:.1%} of essentials")
print(f"  essCov@P0.90 W2-only={mW2:.3f}")
print(f"    HONEST fixed-bonus(0.10) combine={mFixed:.3f}  dCov={(mFixed-mW2)*100:+.1f}pp  "
      f"({n_help_fixed} up / {n_hurt_fixed} down / {len(rows)-n_help_fixed-n_hurt_fixed} flat)")
print(f"    (optimistic per-org-tuned bonus={mC:.3f}  dCov={(mC-mW2)*100:+.1f}pp -- in-sample upper bound)")
print(f"     ({n_help}/{len(rows)} orgs improved at all; rest: hole-OG adds nothing at P>=0.90)")
print(f"  redundancy: hole genes W2 ALREADY ranks high -> precision={mRedu:.3f} (the conserved core)")
print(f"  orthogonal: hole genes W2 MISSES (w2<0.5)   -> precision={mOrth:.3f}, ~{mOrthN:.0f}/org")
print(f"  => orthogonal part is {'USEFUL' if mOrth>0.5 else 'too noisy ('+format(mOrth,'.2f')+') to add at P>=0.90'}")

json.dump(dict(n_hole_ogs=len(hole_og), per_org=rows,
               mean_hole_precision=float(mP), mean_hole_coverage=float(mCov),
               mean_w2_essCov=float(mW2),
               mean_fixed_combine_essCov=float(mFixed), mean_dCov_fixed=float(mFixed-mW2),
               n_orgs_up_fixed=n_help_fixed, n_orgs_down_fixed=n_hurt_fixed,
               mean_best_combine_essCov=float(mC), mean_dCov_optimistic=float(mC-mW2),
               mean_P_redundant=float(mRedu), mean_P_orthogonal=float(mOrth)),
          open(OUT/"wheel2_unmodeled.json","w"), indent=2)
print(f"\nwrote outputs/orphan/wheel2_unmodeled.json", flush=True)
import csv as _csv
with open(OUT/"wheel2_unmodeled_pergene.csv","w",newline="") as f:
    w=_csv.writer(f); w.writerow(["organism","locus_tag","truth","og","w2","hole"])
    w.writerows(pergene_dump)
print(f"wrote outputs/orphan/wheel2_unmodeled_pergene.csv ({len(pergene_dump)} genes)", flush=True)
