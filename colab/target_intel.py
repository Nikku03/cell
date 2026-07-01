"""Target-intelligence layer — the "scientific completeness" pass that turns the
cell-map engine from a target-HYPOTHESIS generator into a de-risked, prioritized
antibacterial TARGET list. Seven dimensions, real local data where possible:

  1 host selectivity   BLASTP vs human proteome            (host_selectivity.json)
  2 pan-strain ess.    feba orthologs x cross-org fitness  (real)
  3 ESKAPE coverage    FBA on iCN718/iYL1228/STM/iEK1008   (real pathogen models)
  4 in-host conditions FBA under iron-lim / anaerobic / C-lim (real)
  5 escape difficulty  breadth of essentiality across conditions (real)
  6 druggability       target-class precedent prior         (PROXY, labeled)
  7 calibrated conf.   reliability-binned P(essential)       (real)

+ credibility benchmark: does the ranking recover KNOWN antibiotic targets?
Outputs a ranked target_intel.csv + report.
"""
import csv, json, sqlite3, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion, single_reaction_deletion
warnings.filterwarnings("ignore"); cobra.Configuration().solver="glpk"
OUT=Path("outputs/orphan"); REG=Path("data/external_data/regulon"); MODELS=Path("data/external_data/bigg_models")
D=json.load(open(OUT/"cell_app_data.json")); GENES=D["genes"]
b2g={g["b"]:g for g in GENES if g["b"]}
con=sqlite3.connect("data/external_data/feba/feba.db"); cur=con.cursor()

# ---------- 7. calibrated confidence (reliability binning of pred_prob vs Keio) ----------
tru=[(g["pred_prob"],g["essential"]) for g in GENES if g["essential"] is not None]
tp=np.array([p for p,_ in tru]); ty=np.array([y for _,y in tru])
order=np.argsort(tp); bins=np.array_split(order,12); centers=[];rates=[]
for bi in bins:
    centers.append(tp[bi].mean()); rates.append(ty[bi].mean())
centers=np.array(centers); rates=np.maximum.accumulate(np.array(rates))  # monotone
def calib(p): return float(np.interp(p,centers,rates))
print(f"[7] calibration: {len(tru)} labeled genes, 12 bins (monotone)")

# ---------- 2. pan-strain essentiality (feba orthologs x cross-org fitness) ----------
print("[2] pan-strain essentiality from feba orthologs ...",flush=True)
b2kl={sn:l for l,sn in cur.execute("SELECT locusId,sysName FROM Gene WHERE orgId='Keio' AND sysName LIKE 'b%'")}
ortho=defaultdict(list)
for l1,o2,l2 in cur.execute("SELECT locusId1,orgId2,locusId2 FROM Ortholog WHERE orgId1='Keio'"):
    ortho[l1].append((o2,l2))
minfit={}
for o,l,mf in cur.execute("SELECT orgId,locusId,MIN(fit) FROM GeneFitness GROUP BY orgId,locusId"):
    if mf is not None: minfit[(o,l)]=mf
paness={}
for b,kl in b2kl.items():
    orgs=ortho.get(kl,[]); dat=[minfit[(o,l)] for o,l in orgs if (o,l) in minfit]
    if dat:
        ess=sum(1 for m in dat if m< -2.0)
        paness[b]=dict(n_orgs=len(dat), frac_ess=round(ess/len(dat),3))
print(f"    pan-strain fitness for {len(paness)} genes (median orgs/gene "
      f"{int(np.median([v['n_orgs'] for v in paness.values()]))})")

# ---------- 4+5. in-host conditions + escape difficulty (iJO1366) ----------
print("[4/5] in-host-condition essentiality (iJO1366) ...",flush=True)
m=cobra.io.read_sbml_model(str(MODELS/"iJO1366.xml.gz")); MG={g.id for g in m.genes}
def ess_under(setup):
    with m as mm:
        med=mm.medium.copy(); setup(med); mm.medium=med
        wt=mm.slim_optimize()
        if not wt or wt<1e-6: return set(),0.0
        df=single_gene_deletion(mm,processes=1); es=set()
        for _,r in df.iterrows():
            i=r["ids"]; b=list(i)[0] if isinstance(i,(set,frozenset)) else i
            gr=r["growth"]; gr=0.0 if gr is None or gr!=gr else gr
            if gr<0.01*wt: es.add(b)
        return es,float(wt)
def s_rich(med): pass
def s_iron(med):                       # iron-LIMITED (host nutritional immunity), not iron-free
    for k in list(med):
        if "fe2" in k or "fe3" in k: med[k]=min(med[k],0.1)
def s_anaer(med):
    for k in list(med):
        if k=="EX_o2_e": med[k]=0.0
def s_clim(med):
    if "EX_glc__D_e" in med: med["EX_glc__D_e"]=1.0
CONDS={"rich":s_rich,"iron_limited":s_iron,"anaerobic":s_anaer,"carbon_limited":s_clim}
cond_ess={};
for nm,fn in CONDS.items():
    es,wt=ess_under(fn); cond_ess[nm]=es; print(f"    {nm:14s}: wt={wt:.2f}  essential={len(es)}")
rich=cond_ess["rich"]
in_host_only={} ; breadth={}
allm=set().union(*cond_ess.values())
for b in allm:
    c_in=[nm for nm in CONDS if b in cond_ess[nm]]
    breadth[b]=round(len(c_in)/len(CONDS),3)
    if b not in rich and any(b in cond_ess[nm] for nm in ("iron_limited","anaerobic","carbon_limited")):
        in_host_only[b]=c_in
print(f"    in-host-CONDITIONAL targets (essential in a host niche, not rich): {len(in_host_only)}")

# ---------- 3. ESKAPE pathogen coverage (essential REACTIONS, intersect by BiGG id) ----------
print("[3] ESKAPE/pathogen essential-reaction sets ...",flush=True)
PATH={"A.baumannii(iCN718)":"iCN718.xml.gz","K.pneumoniae(iYL1228)":"iYL1228.xml.gz",
      "Salmonella(STM)":"STM_v1_0.xml.gz","M.tuberculosis(iEK1008)":"iEK1008.xml.gz"}
path_ess_rxn={}
for nm,fn in PATH.items():
    try:
        pm=cobra.io.read_sbml_model(str(MODELS/fn)); wt=pm.slim_optimize()
        if not wt or wt<1e-6: print(f"    {nm}: no growth on default medium, skip"); continue
        df=single_reaction_deletion(pm,processes=1); es=set()
        for _,r in df.iterrows():
            i=r["ids"]; rid=list(i)[0] if isinstance(i,(set,frozenset)) else i
            gr=r["growth"]; gr=0.0 if gr is None or gr!=gr else gr
            if gr<0.01*wt: es.add(rid)
        path_ess_rxn[nm]=es; print(f"    {nm:24s}: wt={wt:.2f}  essential rxns={len(es)}")
    except Exception as e: print(f"    {nm}: FAILED {e}")
# broad-spectrum: reactions essential in ALL pathogen models that contain them
allrxn=defaultdict(int); esrxn=defaultdict(int)
mods=[cobra.io.read_sbml_model(str(MODELS/fn)) for fn in PATH.values()]
present={nm:{r.id for r in mm.reactions} for nm,mm in zip(PATH,mods)}
broad=[]
for nm,es in path_ess_rxn.items():
    for rid in es: esrxn[rid]+=1
for rid in esrxn:
    have=sum(1 for nm in path_ess_rxn if rid in present[nm])
    if have>=2 and esrxn[rid]==have: broad.append((rid,have))
broad.sort(key=lambda x:-x[1])
print(f"    broad-spectrum essential reactions (essential in ALL models having them, >=2): {len(broad)}")

# ---------- 1. host selectivity (from background blastp) ----------
hs=OUT/"host_selectivity.json"
host=json.load(open(hs)) if hs.exists() else {}
print(f"[1] host selectivity: {'loaded '+str(len(host))+' human homologs' if host else 'PENDING (blastp not finished)'}")
def selectivity(b):
    h=host.get(b)
    if not host: return None            # unknown yet
    if h is None: return 1.0            # no human homolog -> fully selective
    return round(max(0.0,1.0-h["human_pident"]/60.0),3)  # >=60% identity -> 0

# ---------- 6. druggability (target-class precedent PROXY, from flags + iJO1366 subsystem) ----------
# product field is a b-number (no description), so classify from the flags we DO have:
# metabolic enzymes = most druggable (small-molecule precedent); TFs least.
b2sub={}
for rx in m.reactions:
    sub=(rx.subsystem or "").strip()
    for gg in rx.genes:
        if sub and gg.id not in b2sub: b2sub[gg.id]=sub
def druggability(g):
    b=g["b"]
    if g.get("is_tf"): return 0.2,"regulator"
    if g.get("metabolic"):                 # enzyme/transport in the metabolic model
        sub=b2sub.get(b,"").lower()
        if any(k in sub for k in("transport","membrane")): return 0.6,"transporter"
        return 0.9,"enzyme"
    if g.get("core"): return 0.7,"translation/replication"
    return 0.4,"other"
def subsystem(b): return b2sub.get(b,"")

# ---------- assemble the ranked target table ----------
rows=[]
for g in GENES:
    b=g["b"]
    if not b: continue
    ce=calib(g["pred_prob"])
    sel=selectivity(b); pan=paness.get(b,{}).get("frac_ess"); brd=breadth.get(b)
    drug,cls=druggability(g)
    inhost=b in in_host_only
    # composite: gate on essentiality & selectivity; reward pan-strain, druggable, hard-escape
    s_sel = 1.0 if sel is None else sel
    s_pan = 0.5 if pan is None else pan
    s_brd = 0.3 if brd is None else brd
    score = ce * s_sel * (0.4+0.6*drug) * (0.5+0.5*s_pan) * (0.6+0.4*s_brd)
    rows.append(dict(gene=g["name"],b=b,subsystem=subsystem(b)[:40],
        keio_essential=g["essential"], calib_ess=round(ce,3),
        host_selectivity=sel, pan_strain_ess=pan, cond_breadth=brd,
        in_host_conditional=int(inhost), druggability=drug, target_class=cls,
        target_score=round(score,4)))
rows.sort(key=lambda r:-r["target_score"])
for i,r in enumerate(rows): r["rank"]=i+1
with open(OUT/"target_intel.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# ---------- credibility benchmark: recover KNOWN antibiotic targets ----------
KNOWN={"gyra","gyrb","parc","pare","rpob","rpoc","fola","folp","folc","fabi","fabb",
    "mura","murb","murc","murd","mure","murf","ddlb","alr","ftsi","mrda","mrcb","lpxc","lpxa",
    "lpxd","def","metg","iles","rpsl","accd","acpp","kasa","efp","glmu","waaa"}
rank_of={r["gene"].lower():r["rank"] for r in rows}
N=len(rows); hits=[(g,rank_of[g]) for g in KNOWN if g in rank_of]
top10pct=[g for g,rk in hits if rk<=0.10*N]; top25pct=[g for g,rk in hits if rk<=0.25*N]
print(f"\n[benchmark] known antibiotic targets found: {len(hits)}; "
      f"in top10%={len(top10pct)} top25%={len(top25pct)}")
print("  examples:",sorted(hits,key=lambda x:x[1])[:12])

summary=dict(n_targets=N, calibrated=True,
    pan_strain_genes=len(paness), in_host_conditional=len(in_host_only),
    pathogen_models=list(path_ess_rxn), broad_spectrum_rxns=len(broad),
    broad_spectrum_examples=[r for r,_ in broad[:20]],
    host_selectivity_loaded=bool(host), known_target_recovery=dict(
        found=len(hits),top10pct=len(top10pct),top25pct=len(top25pct),
        ranks={g:rk for g,rk in sorted(hits,key=lambda x:x[1])}),
    top20_targets=[{k:r[k] for k in("rank","gene","subsystem","target_score","calib_ess",
        "host_selectivity","pan_strain_ess","druggability","target_class")} for r in rows[:20]])
json.dump(summary,open(OUT/"target_intel.json","w"),indent=2)
print("\n=== TOP 15 TARGETS ===")
for r in rows[:15]:
    print(f"  {r['rank']:4d} {r['gene']:6s} score={r['target_score']:.3f} ess={r['calib_ess']:.2f} "
          f"sel={r['host_selectivity']} pan={r['pan_strain_ess']} drug={r['druggability']} {r['target_class']:11s} {r['subsystem'][:34]}")
print("\nwrote target_intel.csv + target_intel.json")
