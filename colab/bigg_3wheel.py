"""Wire the new BiGG models into the 3-wheel system.

For each organism with a usable bridge, run single-gene-deletion FBA (rich
medium, matching wet-lab screen conditions) and compute the 3-source consensus
precision: model + conservation + FBA.

Bridge types:
  NATIVE      : BiGG gene IDs match our truth locus_tags directly
                 -> beril_Putida (iJN1463, 97%), mtub (iEK1008, 98%)
  KEIO BRIDGE : pre-built b-number <-> Keio numeric tag bridge
                 -> beril_Keio (iML1515)
  GENE NAME   : BiGG's gene.name == truth's gene_name field
                 -> ecoli_BW25113_tradis (iML1515; ~3500 mapped)

For each: FBA essential set, soup definition, 3-way agreement vs truth,
and compare to the 2-wheel baseline.
"""
import csv, time, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
cobra.Configuration().solver = "glpk"

OUT = Path("outputs/orphan")
MODELS = Path("data/external_data/bigg_models")

# ------- rich-medium helper --------
RICH_AA = ["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L",
           "gly","his__L","ile__L","leu__L","lys__L","met__L","phe__L",
           "pro__L","ser__L","thr__L","trp__L","tyr__L","val__L",
           "ade","gua","ura","thymd","cytd","ins","adn","gsn","uri",
           "thm","ribflv","nac","pnto__R","pydx","4abz","btn","fol"]
def to_rich(M, lb=-10.0):
    n=0
    for x in RICH_AA:
        rid=f"EX_{x}_e"
        if rid in M.reactions:
            M.reactions.get_by_id(rid).lower_bound=lb; n+=1
    return n

# ------- run FBA gene deletion -------
def fba_essentials(M, label):
    g0=M.slim_optimize()
    print(f"  {label}: rich growth = {g0:.3f}/h  ...", flush=True)
    de=single_gene_deletion(M)
    out={}
    for _,row in de.iterrows():
        gid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
        out[gid]=(row["growth"]<0.01*g0 or np.isnan(row["growth"]))
    return out, g0

# ------- data shared across orgs --------
ALL_LABELS = list(csv.DictReader(open("data/drive_import/labels/labels.csv")))
clade_of = {r["organism"]:r["clade"] for r in csv.DictReader(open("data/drive_import/labels/clade_splits.csv"))}
og_of = {}
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"): og_of[(r["organism"],r["locus_tag"])]=r["og_id"]

# preds + brightness per (clade, lt) — extract arrays ONCE (npz re-extraction
# in a dict-comp is ~100x slower than indexing local names)
P=np.load(OUT/"af_torch_preds_aug.npz", allow_pickle=True)
_cl = P["clade"]; _lt = P["lt"]; _p = P["p"]; _br = P["brightness"]
preds = {(_cl[i], _lt[i]):(float(_p[i]), float(_br[i]))
         for i in range(len(_p)) if not np.isnan(_p[i])}
del _cl, _lt, _p, _br
print(f"[preamble] preds dict built: {len(preds)} entries", flush=True)

def cons_rate_loco(held_clade):
    """leak-clean OG essential rate, excluding held_clade entirely."""
    n=defaultdict(int); e=defaultdict(int)
    for r in ALL_LABELS:
        if clade_of.get(r["organism"])==held_clade: continue
        og=og_of.get((r["organism"], r["locus_tag"]))
        if og: n[og]+=1; e[og]+=int(r["essential"])
    return {og: e[og]/n[og] for og in n if n[og]>=5}

# ------- per-organism config -------
ORGS = [
    dict(our="beril_Putida", model="iJN1463", clade="pseudomonas", bridge="native"),
    dict(our="mtub",         model="iEK1008", clade="mycobacterium", bridge="native"),
    dict(our="beril_Keio",   model="iML1515", clade="escherichia",
         bridge="keio_b_num"),
    dict(our="ecoli_BW25113_tradis", model="iML1515", clade="escherichia",
         bridge="gene_name"),
]

# Keio numeric -> b-number bridge
keio_bridge={r["locus_tag"]:r["b_number"]
             for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))
             if r["source"] in ("anchor_product","anchor_product_round2","synteny_walk")}

def join_truth_to_model(org, model, bridge_kind):
    """returns dict: our_locus_tag -> model_gene_id (when bridgeable)."""
    truth = {r["locus_tag"]:int(r["essential"]) for r in ALL_LABELS if r["organism"]==org["our"]}
    gene_names = {r["locus_tag"]: (r.get("gene_name") or "").strip()
                  for r in ALL_LABELS if r["organism"]==org["our"]}
    model_ids = {g.id for g in model.genes}
    name_to_id = {g.name: g.id for g in model.genes if g.name}
    bridge = {}
    if bridge_kind == "native":
        bridge = {lt:lt for lt in truth if lt in model_ids}
    elif bridge_kind == "keio_b_num":
        bridge = {lt:keio_bridge[lt] for lt in truth
                  if lt in keio_bridge and keio_bridge[lt] in model_ids}
    elif bridge_kind == "gene_name":
        bridge = {lt:name_to_id[gn] for lt,gn in gene_names.items()
                  if gn and gn in name_to_id}
    return truth, bridge

results = []
for org in ORGS:
    t0=time.time()
    print(f"\n=== {org['our']} <- {org['model']} ({org['bridge']}) ===", flush=True)
    path = MODELS / f"{org['model']}.xml.gz"
    M = cobra.io.read_sbml_model(str(path))
    opened = to_rich(M)
    truth, bridge = join_truth_to_model(org, M, org['bridge'])
    if len(bridge) < 50:
        print(f"  bridge too small ({len(bridge)}) -- skipping", flush=True); continue
    n_ess = sum(truth[lt] for lt in bridge)
    print(f"  bridge: {len(bridge)} genes ({n_ess} essential) | opened {opened} uptake rxns", flush=True)
    fba_ess, growth = fba_essentials(M, org['model'])
    fba_mapped = {lt:fba_ess.get(bid,False) for lt,bid in bridge.items()}

    # conservation per gene
    rate = cons_rate_loco(org['clade'])
    def cons(lt):
        og = og_of.get((org['our'], lt))
        return rate.get(og, np.nan) if og else np.nan
    # model pred per gene (use clade-matched record)
    def model_p(lt):
        return preds.get((org['clade'], lt), (np.nan, np.nan))[0]

    # 3-source on the bridge subset
    table=[]
    for lt in bridge:
        table.append(dict(lt=lt,
                          truth=truth[lt],
                          w1=int(model_p(lt) >= 0.5) if not np.isnan(model_p(lt)) else 0,
                          w2=int(cons(lt) >= 0.5) if not np.isnan(cons(lt)) else 0,
                          w3=int(fba_mapped[lt])))
    ys=[r["truth"] for r in table]
    def prec(g, t):
        return sum(1 for r in g if r["truth"]==t)/max(len(g),1)
    ttt=[r for r in table if r["w1"]==r["w2"]==r["w3"]==1]
    nnn=[r for r in table if r["w1"]==r["w2"]==r["w3"]==0]
    w1w2=[r for r in table if r["w1"]==1 and r["w2"]==1]
    fba_only_ess=[r for r in table if r["w3"]==1 and r["w1"]==0 and r["w2"]==0 and r["truth"]==1]
    metrics=dict(
        n_joinable=len(table),
        true_essential=sum(ys),
        ttt_n=len(ttt), ttt_precision=round(prec(ttt,1),3),
        nnn_n=len(nnn), nnn_precision=round(prec(nnn,0),3),
        w1w2_n=len(w1w2), w1w2_precision=round(prec(w1w2,1),3),
        delta_w3_pp=round((prec(ttt,1)-prec(w1w2,1))*100,1),
        fba_only_essentials_caught=len(fba_only_ess),
    )
    elapsed=time.time()-t0
    print(f"  3-source: TTT n={metrics['ttt_n']} P={metrics['ttt_precision']:.3f} "
          f"vs W1+W2 P={metrics['w1w2_precision']:.3f}  "
          f"(delta {metrics['delta_w3_pp']:+.1f}pp)", flush=True)
    print(f"  FBA-only essentials caught (W1+W2 missed): {metrics['fba_only_essentials_caught']} "
          f"| runtime {elapsed:.1f}s", flush=True)
    results.append(dict(organism=org['our'], model=org['model'],
                        bridge=org['bridge'], **metrics,
                        runtime_sec=round(elapsed,1)))

json.dump(results, open(OUT/"bigg_3wheel_results.json","w"), indent=2)
print(f"\nwrote outputs/orphan/bigg_3wheel_results.json")
print(f"\n=== SUMMARY across {len(results)} organisms ===")
print(f"{'organism':<26}{'n_join':>8}{'TTT_n':>7}{'TTT_P':>8}{'W1+W2_P':>9}{'delta':>7}{'FBA_only':>10}")
for r in results:
    print(f"  {r['organism']:<24}{r['n_joinable']:>8}{r['ttt_n']:>7}"
          f"{r['ttt_precision']:>8.3f}{r['w1w2_precision']:>9.3f}"
          f"{r['delta_w3_pp']:>6.1f}pp{r['fba_only_essentials_caught']:>10}")
