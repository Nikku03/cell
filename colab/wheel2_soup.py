"""Wheel 2's REAL job: rescue the conservation-blind soup.

The 23% soup = non-conserved essentials, orphan essentials, and look-alike
non-essentials that conservation (Wheel 2-score) cannot separate. The question
your design poses: inside that soup -- where conservation has NO signal by
construction -- can a gene-LOCAL necessity call separate the essentials?

Gene-local necessity (NOT orthology): does this gene's OWN product fill a
non-redundant hole in the cell? Operationally = the gene is FBA single-gene-
deletion essential in the organism's OWN model (rich medium). This is the
gene's own reaction killing growth, independent of how rare/orphan it is across
organisms.

Done on the 4 organisms with a native model (iJN1463/iML1515/iEK1008/iYL1228).
For each:
  soup = genes where leak-clean conservation w2 < 0.30  OR  orphan (no OG)
  within the soup:
    - base essential rate
    - gene-local necessity precision / coverage / lift
    - of the soup essentials: how many are REACHABLE (in the metabolic model)
      and recovered by necessity, vs UNREACHABLE (orphan / non-metabolic)
    - characterize both buckets by product annotation

Output: outputs/orphan/wheel2_soup.json
"""
import csv, json, re, time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
cobra.Configuration().solver = "glpk"
OUT = Path("outputs/orphan"); MODELS = Path("data/external_data/bigg_models")

RICH=["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L","gly","his__L",
      "ile__L","leu__L","lys__L","met__L","phe__L","pro__L","ser__L","thr__L","trp__L",
      "tyr__L","val__L","ade","gua","ura","thymd","cytd","ins","adn","gsn","uri","thm",
      "ribflv","nac","pnto__R","pydx","4abz","btn","fol"]

ALL=list(csv.DictReader(open("data/drive_import/labels/labels.csv")))
clade_of={r["organism"]:r["clade"] for r in csv.DictReader(open("data/drive_import/labels/clade_splits.csv"))}
og_of={(r["organism"],r["locus_tag"]):r["og_id"]
       for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")) if r.get("og_id")}
annot={(r["organism"],r["locus_tag"]):(r.get("product") or "")
       for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv"))}
truth_by={(r["organism"],r["locus_tag"]):int(r["essential"]) for r in ALL}

def cons_rate(held):
    n=defaultdict(int); e=defaultdict(int)
    for r in ALL:
        if clade_of.get(r["organism"])==held: continue
        o=og_of.get((r["organism"],r["locus_tag"]))
        if o: n[o]+=1; e[o]+=int(r["essential"])
    return {o:e[o]/n[o] for o in n if n[o]>=5}

# bridges: our locus -> model gene id
b_to_keio={r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
keio_to_b={v:k for k,v in b_to_keio.items()}

def fba_essential(model_id):
    M=cobra.io.read_sbml_model(str(MODELS/f"{model_id}.xml.gz"))
    for x in RICH:
        rid=f"EX_{x}_e"
        if rid in M.reactions: M.reactions.get_by_id(rid).lower_bound=-10
    g0=M.slim_optimize()
    de=single_gene_deletion(M)
    out={}
    for _,row in de.iterrows():
        gid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
        out[gid]=int(row["growth"]<0.01*g0 or np.isnan(row["growth"]))
    return out, M

# org config: (our_org, model_id, bridge to map our locus -> model gene)
def koxy_bridge(M):
    ka={r["locus_tag"]:r["product"] for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv"))
        if r["organism"]=="beril_Koxy" and r.get("product")}
    tok=defaultdict(list)
    for lt,prod in ka.items():
        for t in re.findall(r"[A-Za-z0-9]{4,}",prod): tok[t.lower()].append(lt)
    out={}
    for g in M.genes:
        gn=(g.name or "").strip()
        if gn and g.id!=gn:
            mm=tok.get(gn.lower(),[])
            if len(mm)==1: out[mm[0]]=g.id   # our_locus -> model gene
    return out

CONFIG=[("beril_Putida","iJN1463","native"),
        ("beril_Keio","iML1515","keio"),
        ("mtub","iEK1008","native"),
        ("beril_Koxy","iYL1228","koxy")]

print("Running...", flush=True); t0=time.time()
results=[]
agg_reach=Counter(); agg_unreach=Counter()
for our, model_id, bk in CONFIG:
    fba, M = fba_essential(model_id)
    model_ids={g.id for g in M.genes}
    # our locus -> model gene
    if bk=="native":
        loc2mod={lt:lt for (o,lt) in truth_by if o==our and lt in model_ids}
    elif bk=="keio":
        loc2mod={lt:keio_to_b[lt] for (o,lt) in truth_by if o==our and lt in keio_to_b and keio_to_b[lt] in model_ids}
    elif bk=="koxy":
        loc2mod=koxy_bridge(M)
    held=clade_of.get(our,"?"); rate=cons_rate(held)
    # build soup
    soup=[]
    for (o,lt),tv in truth_by.items():
        if o!=our: continue
        og=og_of.get((o,lt))
        w2=rate.get(og, np.nan) if og else np.nan
        is_orphan = (og is None) or (og not in rate)
        w2v = 0.0 if np.isnan(w2) else w2
        in_soup = (w2v < 0.30) or is_orphan
        if not in_soup: continue
        mg=loc2mod.get(lt)
        fba_ess = fba.get(mg, None) if mg else None   # None = not in model (unreachable)
        soup.append(dict(lt=lt, truth=tv, w2=w2v, orphan=is_orphan,
                         in_model=(mg is not None), fba=fba_ess,
                         product=annot.get((o,lt),"")))
    n=len(soup); pos=sum(s["truth"] for s in soup)
    base=pos/max(n,1)
    # gene-local necessity call = FBA-essential (only defined for in-model genes)
    called=[s for s in soup if s["fba"]==1]
    P_nec=sum(s["truth"] for s in called)/max(len(called),1)
    cov_nec=sum(s["truth"] for s in called)/max(pos,1)
    # reachability of the soup ESSENTIALS
    ess=[s for s in soup if s["truth"]==1]
    ess_in_model=[s for s in ess if s["in_model"]]
    ess_recovered=[s for s in ess if s["fba"]==1]
    ess_unreach=[s for s in ess if not s["in_model"]]
    results.append(dict(organism=our, model=model_id, soup_n=n, soup_ess=pos,
        soup_base=round(base,3),
        necessity_called=len(called), necessity_P=round(P_nec,3), necessity_cov=round(cov_nec,3),
        lift=round(P_nec/max(base,1e-6),2),
        ess_in_model=len(ess_in_model), ess_recovered=len(ess_recovered),
        ess_unreachable=len(ess_unreach),
        frac_ess_reachable=round(len(ess_in_model)/max(pos,1),3)))
    # characterize buckets
    for s in ess_recovered:
        for w in re.findall(r"[A-Za-z]{4,}", s["product"].lower()): agg_reach[w]+=1
    for s in ess_unreach:
        for w in re.findall(r"[A-Za-z]{4,}", s["product"].lower()): agg_unreach[w]+=1
    print(f"  {our:<14} soup n={n:<5} ess={pos:<4} base={base:.2f} | "
          f"necessity P={P_nec:.2f} cov={cov_nec:.2f} lift={P_nec/max(base,1e-6):.1f}x | "
          f"ess reachable {len(ess_in_model)}/{pos} ({len(ess_in_model)/max(pos,1)*100:.0f}%) "
          f"recovered={len(ess_recovered)} unreachable={len(ess_unreach)}", flush=True)
print(f"  ({time.time()-t0:.0f}s)\n", flush=True)

# aggregate
TN=sum(r["soup_n"] for r in results); TE=sum(r["soup_ess"] for r in results)
TC=sum(r["necessity_called"] for r in results)
TCR=sum(r["ess_recovered"] for r in results); TIM=sum(r["ess_in_model"] for r in results)
TUN=sum(r["ess_unreachable"] for r in results)
# pooled necessity precision
all_called_correct=sum(r["necessity_P"]*r["necessity_called"] for r in results)
print("=== AGGREGATE over 4 modeled organisms (conservation-blind soup) ===")
print(f"  soup size: {TN} genes, {TE} essential (base rate {TE/TN:.3f})")
print(f"  gene-local necessity: {TC} called, ~{all_called_correct/max(TC,1):.3f} precision, "
      f"recovers {TCR}/{TE} soup essentials ({TCR/TE*100:.0f}%)")
print(f"  reachability of soup essentials: {TIM}/{TE} in metabolic model ({TIM/TE*100:.0f}%); "
      f"{TUN}/{TE} UNREACHABLE (orphan/non-metabolic, {TUN/TE*100:.0f}%)")
print(f"\n  recovered soup-essential products (top words): "
      + ", ".join(f"{w}({c})" for w,c in agg_reach.most_common(12)))
print(f"  UNREACHABLE soup-essential products (top words): "
      + ", ".join(f"{w}({c})" for w,c in agg_unreach.most_common(12)))

json.dump(dict(per_org=results,
               soup_n=TN, soup_ess=TE, soup_base=TE/TN,
               necessity_called=TC, necessity_precision=all_called_correct/max(TC,1),
               necessity_recovered=TCR, recovered_frac=TCR/TE,
               ess_reachable=TIM, ess_reachable_frac=TIM/TE, ess_unreachable=TUN,
               recovered_top_words=agg_reach.most_common(25),
               unreachable_top_words=agg_unreach.most_common(25)),
          open(OUT/"wheel2_soup.json","w"), indent=2)
print(f"\nwrote outputs/orphan/wheel2_soup.json", flush=True)
