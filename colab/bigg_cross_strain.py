"""Cross-strain / cross-species ortholog bridges for the new BiGG models.

The 4-org wiring already in (Putida, mtub, Keio, ecoli_BW25113) used native
ID matches or gene-name bridges. The remaining 13 organisms in
bigg_manifest.csv use proxy models where IDs don't share format:

  - 10 Pseudomonas spp. proxied via iJN1463 (P. putida): bridge via ORTHOGROUP
      (PP_X -> OG_Y, then target_org gene with OG_Y)
  - saur_NCTC8325 via iSB619 (S. aureus N315): cross-strain, try gene-name
      bridge (model.gene.name -> our gene product or symbol)
  - beril_Koxy via iYL1228 (K. pneumoniae): cross-species, try gene-name
  - beril_SynE via iJB785 (Synechococcus): RefSeq-version mismatch; try OG

Run single_gene_deletion ONCE per model (iJN1463 is the only repeat: reused
across all 10 Pseudomonas targets) -- so 4 FBA runs cover 13 organisms.
"""
import csv, time, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
cobra.Configuration().solver = "glpk"

OUT = Path("outputs/orphan")
MODELS = Path("data/external_data/bigg_models")

# ---- rich medium ----
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

def run_fba(M, label):
    g0=M.slim_optimize()
    print(f"  [{label}] rich growth = {g0:.3f}/h  running single_gene_deletion...", flush=True)
    de=single_gene_deletion(M)
    out={}
    for _,row in de.iterrows():
        gid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
        out[gid]=(row["growth"]<0.01*g0 or np.isnan(row["growth"]))
    return out

# ---- shared loading ----
ALL_LABELS = list(csv.DictReader(open("data/drive_import/labels/labels.csv")))
clade_of = {r["organism"]:r["clade"] for r in csv.DictReader(open("data/drive_import/labels/clade_splits.csv"))}
og_of = {}; og_in_org = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"):
        og_of[(r["organism"], r["locus_tag"])] = r["og_id"]
        og_in_org[r["og_id"]][r["organism"]].append(r["locus_tag"])
print(f"orthology: {len(og_of)} (org,lt) -> og; {len(og_in_org)} unique OGs", flush=True)

P=np.load(OUT/"af_torch_preds_aug.npz", allow_pickle=True)
_cl, _lt, _p, _br = P["clade"], P["lt"], P["p"], P["brightness"]
preds = {(_cl[i], _lt[i]):(float(_p[i]), float(_br[i]))
         for i in range(len(_p)) if not np.isnan(_p[i])}
del _cl, _lt, _p, _br
print(f"preds: {len(preds)} entries", flush=True)

def cons_rate_loco(held_clade):
    n=defaultdict(int); e=defaultdict(int)
    for r in ALL_LABELS:
        if clade_of.get(r["organism"])==held_clade: continue
        og=og_of.get((r["organism"], r["locus_tag"]))
        if og: n[og]+=1; e[og]+=int(r["essential"])
    return {og: e[og]/n[og] for og in n if n[og]>=5}

# ---- 3-source compute helper ----
def consensus(target_org, bridge_dict, fba_dict):
    truth_local = {r["locus_tag"]:int(r["essential"]) for r in ALL_LABELS if r["organism"]==target_org}
    clade = clade_of.get(target_org, "?")
    rate = cons_rate_loco(clade)
    table=[]
    for lt, model_gene in bridge_dict.items():
        if lt not in truth_local: continue
        og = og_of.get((target_org, lt))
        c = rate.get(og, np.nan) if og else np.nan
        m = preds.get((clade, lt), (np.nan, np.nan))[0]
        table.append(dict(lt=lt,
                          truth=truth_local[lt],
                          w1=int(m >= 0.5) if not np.isnan(m) else 0,
                          w2=int(c >= 0.5) if not np.isnan(c) else 0,
                          w3=int(fba_dict.get(model_gene, False))))
    def prec(g, t):
        return sum(1 for r in g if r["truth"]==t)/max(len(g),1)
    ttt=[r for r in table if r["w1"]==r["w2"]==r["w3"]==1]
    nnn=[r for r in table if r["w1"]==r["w2"]==r["w3"]==0]
    w1w2=[r for r in table if r["w1"]==1 and r["w2"]==1]
    fba_only=[r for r in table if r["w3"]==1 and r["w1"]==0 and r["w2"]==0 and r["truth"]==1]
    return dict(n_joinable=len(table),
                true_essential=sum(r["truth"] for r in table),
                ttt_n=len(ttt), ttt_precision=round(prec(ttt,1),3),
                nnn_n=len(nnn), nnn_precision=round(prec(nnn,0),3),
                w1w2_n=len(w1w2), w1w2_precision=round(prec(w1w2,1),3),
                delta_w3_pp=round((prec(ttt,1)-prec(w1w2,1))*100,1),
                fba_only_essentials_caught=len(fba_only))

# ============== PHASE 1: Pseudomonas proxies via iJN1463 + OG bridge ==============
print("\n=== PHASE 1: Pseudomonas proxies via iJN1463 (OG bridge) ===", flush=True)
M = cobra.io.read_sbml_model(str(MODELS/"iJN1463.xml.gz"))
to_rich(M)
fba_putida = run_fba(M, "iJN1463")

PSEUDO_TARGETS = [
    "beril_WCS417","beril_PS","beril_psRCH2","beril_SyringaeB728a",
    "beril_SyringaeB728a_mexBdelta","beril_pseudo1_N1B4","beril_pseudo3_N2E3",
    "beril_pseudo5_N2C3_1","beril_pseudo6_N2E2","beril_pseudo13_GW456_L13",
]
# build PP_gene -> OG (cached)
pp_to_og = {lt: og for (org, lt), og in og_of.items() if org == "beril_Putida"}
print(f"\nPutida (beril_Putida) genes with OG: {len(pp_to_og)}", flush=True)

results = []
for tgt in PSEUDO_TARGETS:
    # bridge: target_lt -> PP_gene via shared OG (only 1:1 OG mappings)
    bridge = {}
    ambiguous = 0
    for pp_gene, og in pp_to_og.items():
        if pp_gene not in fba_putida: continue        # only consider genes in the model
        tgt_lts = og_in_org[og].get(tgt, [])
        if len(tgt_lts) == 1:
            bridge[tgt_lts[0]] = pp_gene
        elif len(tgt_lts) >= 2:
            ambiguous += 1
    print(f"\n  {tgt}: bridge {len(bridge)} (1:1 OG), {ambiguous} ambiguous", flush=True)
    if len(bridge) < 50: print(f"    too small, skipping"); continue
    m = consensus(tgt, bridge, fba_putida)
    print(f"    TTT n={m['ttt_n']} P={m['ttt_precision']:.3f} | W1+W2 P={m['w1w2_precision']:.3f} "
          f"({m['delta_w3_pp']:+.1f}pp) | FBA-only ess: {m['fba_only_essentials_caught']}", flush=True)
    results.append(dict(organism=tgt, model="iJN1463", bridge="og_proxy", **m))

# ============== PHASE 2: S. aureus N315 model -> NCTC8325 (gene_name attempt) ==============
print("\n=== PHASE 2: S. aureus iSB619 -> saur_NCTC8325 (gene-name bridge) ===", flush=True)
M2 = cobra.io.read_sbml_model(str(MODELS/"iSB619.xml.gz"))
to_rich(M2)
fba_saur = run_fba(M2, "iSB619")
# build bridge: model.gene.name -> SAOUHSC_X via product match
import re
saur_annot = {r["locus_tag"]:r["product"]
              for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv"))
              if r["organism"]=="saur_NCTC8325_biotradis" and r.get("product")}
# For each model gene, try to find a SAOUHSC gene whose product contains its name
bridge_saur = {}
model_names = [(g.id, (g.name or "").strip()) for g in M2.genes]
# index annotations by token
import re as _re
saur_tokens = defaultdict(list)
for lt, prod in saur_annot.items():
    for tok in _re.findall(r"[A-Za-z0-9]{4,}", prod):
        saur_tokens[tok.lower()].append(lt)
for gid, gname in model_names:
    if not gname or gid == gname: continue
    matches = saur_tokens.get(gname.lower(), [])
    if len(matches) == 1: bridge_saur[matches[0]] = gid
print(f"  saur bridge via product-keyword: {len(bridge_saur)}", flush=True)
if len(bridge_saur) >= 50:
    m = consensus("saur_NCTC8325_biotradis", bridge_saur, fba_saur)
    print(f"  TTT n={m['ttt_n']} P={m['ttt_precision']:.3f} | W1+W2 P={m['w1w2_precision']:.3f} "
          f"({m['delta_w3_pp']:+.1f}pp) | FBA-only ess: {m['fba_only_essentials_caught']}", flush=True)
    results.append(dict(organism="saur_NCTC8325_biotradis", model="iSB619",
                        bridge="gene_name_token", **m))
else:
    print(f"  bridge too small, skipping", flush=True)

# ============== PHASE 3: Klebsiella iYL1228 -> beril_Koxy (gene_name) ==============
print("\n=== PHASE 3: Klebsiella iYL1228 -> beril_Koxy (gene-name bridge) ===", flush=True)
M3 = cobra.io.read_sbml_model(str(MODELS/"iYL1228.xml.gz"))
to_rich(M3)
fba_koxy = run_fba(M3, "iYL1228")
koxy_annot = {r["locus_tag"]:r["product"]
              for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv"))
              if r["organism"]=="beril_Koxy" and r.get("product")}
koxy_tokens = defaultdict(list)
for lt, prod in koxy_annot.items():
    for tok in _re.findall(r"[A-Za-z0-9]{4,}", prod):
        koxy_tokens[tok.lower()].append(lt)
bridge_koxy = {}
for g in M3.genes:
    gname = (g.name or "").strip()
    if not gname or g.id == gname: continue
    matches = koxy_tokens.get(gname.lower(), [])
    if len(matches) == 1: bridge_koxy[matches[0]] = g.id
print(f"  Koxy bridge via product-keyword: {len(bridge_koxy)}", flush=True)
if len(bridge_koxy) >= 50:
    m = consensus("beril_Koxy", bridge_koxy, fba_koxy)
    print(f"  TTT n={m['ttt_n']} P={m['ttt_precision']:.3f} | W1+W2 P={m['w1w2_precision']:.3f} "
          f"({m['delta_w3_pp']:+.1f}pp) | FBA-only ess: {m['fba_only_essentials_caught']}", flush=True)
    results.append(dict(organism="beril_Koxy", model="iYL1228",
                        bridge="gene_name_token", **m))
else:
    print(f"  bridge too small, skipping", flush=True)

# ============== PHASE 4: Synechococcus iJB785 -> beril_SynE (OG via gene name) =====
# iJB785 uses SYNPCC7942_RS* in BOTH id and name, but our SynE uses Synpcc7942_*.
# Try OG bridge: if any beril_SynE gene shares an OG with a Synechococcus
# in the orthology table, use that as the link. But iJB785 genes aren't in
# our orthology table directly. Pass for now -- documented limit.
print("\n=== PHASE 4: Synechococcus iJB785 -> beril_SynE ===", flush=True)
print("  iJB785 uses SYNPCC7942_RS* (newer RefSeq) and our SynE uses Synpcc7942_*", flush=True)
print("  (older Cyanobase). No locally available cross-version map; skipping.", flush=True)

# ============== save + summary ==============
json.dump(results, open(OUT/"bigg_cross_strain_results.json","w"), indent=2)
print(f"\nwrote outputs/orphan/bigg_cross_strain_results.json", flush=True)
print(f"\n=== SUMMARY across {len(results)} organisms ===")
print(f"{'organism':<32}{'model':<10}{'n_join':>8}{'TTT_n':>7}{'TTT_P':>8}{'W12_P':>8}{'delta':>8}{'FBA_only':>10}")
for r in results:
    print(f"  {r['organism']:<30}{r['model']:<10}{r['n_joinable']:>8}{r['ttt_n']:>7}"
          f"{r['ttt_precision']:>8.3f}{r['w1w2_precision']:>8.3f}"
          f"{r['delta_w3_pp']:>7.1f}pp{r['fba_only_essentials_caught']:>10}")
