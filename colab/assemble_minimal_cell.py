"""Wheel 3 as an ASSEMBLY ENGINE: lay out the essential minimal cell in layers,
from the genome, with provenance labels. Pilots: Putida (data-rich) + mtub
(independent DeJesus truth).

Layers:
  L1 SCAFFOLD (universal core)  - ribosome, tRNA-synthetases, replication,
       transcription core, translation factors, division, envelope, ATP synthase.
       Tagged from product annotation. Generalizes to ANY bacterium.
  L2 METABOLIC SKELETON         - native-model FBA single-gene-deletion essentials.
       (curated BiGG model here; CarveMe auto-builds this for a novel genome.)
  L3 REGULATORS                 - TF genes by family + effector class (curated
       family->effector table). Generalizes from genome.
  L4 REGULATORY EDGES + MOTIFS  - local adjacency TF->target edges; FFL,
       single-input motifs. Partial (local only) by the sequence ceiling.

Per layer: how many true essentials explained (+provenance), cumulative coverage,
and what's left = needs-measurement.
"""
import sqlite3, csv, json, re
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
cobra.Configuration().solver="glpk"
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db"); cur=con.cursor()
OUT=Path("/home/user/cell/outputs/orphan")

CORE={
 "ribosome":      ["ribosomal protein","ribosomal subunit","ribosome","rRNA"],
 "tRNA_system":   ["trna ligase","trna synthetase","aminoacyl-trna","trna-"],
 "replication":   ["dna polymerase","dna primase","dna gyrase","dna helicase","dna ligase",
                   "replicative","replication","topoisomerase","chromosomal replication","dnaa","single-strand"],
 "transcription": ["rna polymerase","sigma factor","transcription termination","transcription antitermination"],
 "translation":   ["elongation factor","initiation factor","release factor","translation factor",
                   "peptide chain release","ribosome recycling","peptidyl-trna"],
 "cell_division": ["cell division","ftsz","ftsa","ftsk","ftsw","ftsq","septum","divisome"],
 "envelope":      ["peptidoglycan","cell wall","lipid a","lipopolysaccharide","mura","murb","murc",
                   "murd","mure","murf","d-alanine","lipoprotein signal","outer membrane assembly"],
 "energy":        ["atp synthase","f0f1","proton-translocating"],
}
def core_cat(prod):
    p=(prod or "").lower()
    for cat,kws in CORE.items():
        for k in kws:
            if k in p: return cat
    return None

TF_FAMILY={
 "lysr":(["lysr","lttr"],          "aromatic/amino-acid intermediate"),
 "tetr":(["tetr"],                  "lipophilic drug / acyl-CoA"),
 "gntr":(["gntr","fadr","hutc"],    "carbon acid / fatty-acyl-CoA"),
 "marr":(["marr","slya"],           "phenolic / oxidative-stress signal"),
 "arac":(["arac","xyls"],           "sugar / aromatic effector"),
 "ompr":(["ompr","phob","narl"],    "phospho-relay (two-component)"),
 "lacl":(["laci","galr","purr"],    "sugar / nucleotide"),
 "merr":(["merr","soxr"],           "metal ion / redox"),
 "arsr":(["arsr","smtb"],           "heavy-metal ion"),
 "asnc":(["asnc","lrp"],            "amino acid (leucine class)"),
 "iclr":(["iclr"],                  "glyoxylate / acetate"),
 "deor":(["deor","fucr"],           "sugar-phosphate / nucleoside"),
 "crp":(["crp","fnr","camp"],       "cAMP / O2 redox"),
 "rrf2":(["rrf2","iscr","nsrr"],    "Fe-S / NO"),
 "sigma":(["sigma factor","rpod","rpoh","rpos","rpoe","rpon"], "sigma (condition switch)"),
}
def tf_family(prod):
    p=(prod or "").lower().replace("-","")
    for fam,(kws,eff) in TF_FAMILY.items():
        for k in kws:
            if k.replace("-","") in p: return fam,eff
    return None,None

def load_org(our, fb, model_id, truth_filter=None):
    print(f"\n{'='*60}\nASSEMBLING: {our}  (feba={fb}, model={model_id})\n{'='*60}", flush=True)
    prod={}
    for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/gene_annotations.csv")):
        if r["organism"]==our and r.get("product"): prod[r["locus_tag"]]=r["product"]
    truth={}
    for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv")):
        if r["organism"]==our:
            if truth_filter and r["source"]!=truth_filter: continue
            truth[r["locus_tag"]]=int(r["essential"])
    coords={}
    for lid,begin,end,strand in cur.execute("SELECT locusId,begin,end,strand FROM Gene WHERE orgId=? AND type=1",(fb,)):
        coords[lid]=(begin,end,strand)
    M=cobra.io.read_sbml_model(f"/home/user/cell/data/external_data/bigg_models/{model_id}.xml.gz")
    RICH=["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L","gly","his__L","ile__L","leu__L",
          "lys__L","met__L","phe__L","pro__L","ser__L","thr__L","trp__L","tyr__L","val__L","ade","gua","ura",
          "thymd","cytd","ins","adn","gsn","uri","thm","ribflv","nac","pnto__R","pydx","4abz","btn","fol"]
    for x in RICH:
        rid=f"EX_{x}_e"
        if rid in M.reactions: M.reactions.get_by_id(rid).lower_bound=-10
    g0=M.slim_optimize(); de=single_gene_deletion(M)
    fba_ess=set()
    for _,row in de.iterrows():
        gid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
        if row["growth"]<0.01*g0 or np.isnan(row["growth"]): fba_ess.add(gid)
    model_genes={g.id for g in M.genes}

    genes=set(truth); POS=sum(truth.values())
    print(f"genes with truth: {len(genes)}  essential: {POS}", flush=True)

    L1=defaultdict(set)
    for lt in genes:
        c=core_cat(prod.get(lt,""))
        if c: L1[c].add(lt)
    L1_all=set().union(*L1.values()) if L1 else set()
    L2=set(lt for lt in genes if lt in model_genes and lt in fba_ess)
    TFs={}; eff_count=Counter()
    for lt in genes:
        fam,eff=tf_family(prod.get(lt,""))
        if fam: TFs[lt]=(fam,eff); eff_count[fam]+=1
    ordered=sorted([lt for lt in genes if lt in coords], key=lambda lt:coords[lt][0])
    idx={lt:i for i,lt in enumerate(ordered)}
    edges=[]
    for tf in TFs:
        if tf not in idx: continue
        i=idx[tf]
        for j in (i-1,i+1):
            if 0<=j<len(ordered): edges.append((tf,ordered[j]))
    tf_set=set(TFs); out_adj=defaultdict(set)
    for a,b in edges: out_adj[a].add(b)
    ffl=sum(1 for a in tf_set for b in out_adj[a] if b in tf_set for c in out_adj[b] if c in out_adj[a])
    sim=sum(1 for a in tf_set if len(out_adj[a])>=2)

    def cov(s):
        es=sum(1 for g in s if truth.get(g)==1); return len(s),es,(es/max(len(s),1))
    n1,e1,p1=cov(L1_all); n2,e2,p2=cov(L2)
    union=L1_all|L2; nu,eu,pu=cov(union)
    n3,e3,p3=cov(set(TFs)); rem=POS-eu

    print(f"\n  L1 SCAFFOLD (universal core): {n1} genes, {e1} essential (precision {p1:.2f})")
    for c,s in sorted(L1.items(),key=lambda kv:-len(kv[1])):
        ce=sum(1 for g in s if truth.get(g)==1)
        print(f"       {c:<15}{len(s):>4} genes {ce:>4} essential")
    print(f"  L2 METABOLIC SKELETON (FBA): {n2} genes, {e2} essential (precision {p2:.2f})")
    print(f"  L1 U L2 STRUCTURAL CORE: {nu} genes -> covers {eu}/{POS} essentials = {eu/POS:.1%}")
    print(f"  L3 REGULATORS: {n3} TF genes ({e3} essential)")
    print(f"       effector families: {dict(eff_count.most_common())}")
    print(f"  L4 LOCAL EDGES: {len(set(edges))} | FFL: {ffl} | single-input(>=2): {sim}")
    print(f"  REMAINING essentials (not in core): {rem}/{POS} = {rem/POS:.1%} -> needs W4/measurement")

    return dict(organism=our, n_genes=len(genes), n_essential=POS,
        L1_scaffold=dict(n=n1,essential=e1,precision=round(p1,3),
            categories={c:[len(s),sum(1 for g in s if truth.get(g)==1)] for c,s in L1.items()}),
        L2_metabolic=dict(n=n2,essential=e2,precision=round(p2,3)),
        structural_core=dict(n=nu,covers_essentials=eu,coverage_frac=round(eu/POS,3)),
        L3_regulators=dict(n_tf=n3,essential=e3,effector_families=dict(eff_count)),
        L4_edges=dict(n_local_edges=len(set(edges)),ffl_motifs=ffl,single_input=sim),
        remaining_essentials=rem, remaining_frac=round(rem/POS,3))

results=[load_org("beril_Putida","Putida","iJN1463"),
         load_org("mtub","MycoTube","iEK1008", truth_filter="DeJesus 2017 (MTBseq)")]
json.dump(results, open(OUT/"assemble_minimal_cell.json","w"), indent=2)
print(f"\nwrote outputs/orphan/assemble_minimal_cell.json")
