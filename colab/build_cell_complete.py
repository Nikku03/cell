"""Assemble the COMPLETE cell data model: every localized protein with its compartment,
cellular PROCESS (transcription/translation/trafficking/transport/metabolism/replication/
signaling/degradation/...), pathway, all our layers, the regulatory + PPI networks (for
perturbation propagation), HIV hijack map, and dark-gene flags. -> cell_complete.json
(consumed by the interactive perturbable cell app)."""
import csv, json, gzip, re
from collections import defaultdict
from pathlib import Path
OUT=Path("outputs/orphan"); H=Path("data/external_data/human")
loc=json.load(open(H/"gene_compartment.json"))
# Entrez -> symbol (for HIV host mapping)
entrez2sym={}
with gzip.open(H/"gene_info.gz","rt") as f:
    next(f)
    for l in f:
        p=l.split("\t")
        if len(p)>2 and p[0]=="9606": entrez2sym[p[1]]=p[2]
# ENSG -> symbol; gene -> pathways
ensp2sym={}; ensp2ensg={}
for l in gzip.open(H/"string_aliases.txt.gz","rt"):
    p=l.rstrip("\n").split("\t")
    if len(p)<3: continue
    if p[2]=="Ensembl_HGNC_symbol": ensp2sym[p[0]]=p[1]
    elif p[2]=="Ensembl_HGNC_ensembl_gene_id": ensp2ensg[p[0]]=p[1]
ensg2sym={ensp2ensg[e]:ensp2sym[e] for e in ensp2sym if e in ensp2ensg}
GEN={"Metabolism","Signal Transduction","Immune System","Disease","Gene expression (Transcription)","Metabolism of proteins","Developmental Biology","Homeostasis","Metabolism of RNA"}
paths=defaultdict(list)
for l in open(H/"reactome_human.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)<4 or not p[0].startswith("ENSG"): continue
    s=ensg2sym.get(p[0])
    if s: paths[s].append(p[3])
def toppath(g):
    ps=[x for x in paths.get(g,[]) if x not in GEN]
    return min(ps,key=len) if ps else (paths.get(g,[None])[0] or "")
# PROCESS assignment
def process(g,comp,istf):
    t=" ".join(paths.get(g,[])).lower()
    if istf or "transcription" in t or "rna polymerase ii" in t or "chromatin" in t: return "transcription"
    if "translation" in t or "ribosom" in t or "trna aminoacyl" in t or "rrna" in t or "elongation" in t: return "translation"
    if "dna replication" in t or "cell cycle" in t or "mitotic" in t or "s phase" in t or "dna repair" in t: return "replication/repair"
    if "slc" in t or ("transport" in t) or "ion channel" in t or "aquaporin" in t: return "transport/uptake"
    if any(k in t for k in["glycolysis","citric acid","oxidative phosphorylation","fatty acid","amino acid metab","nucleotide metab","pentose","cholesterol","gluconeogen","beta-oxidation","biosynthesis of","metabolism of"]): return "metabolism"
    if "proteasome" in t or "ubiquitin" in t or "autophagy" in t or "lysosom" in t or "degradation" in t: return "degradation"
    if "secret" in t or comp in("ER","Golgi") or "vesicle" in t or "exocyt" in t or "endocyt" in t: return "trafficking/secretion"
    if "signal" in t or "receptor" in t or "kinase" in t or "gpcr" in t or "mapk" in t: return "signaling"
    if "immune" in t or "interferon" in t or "cytokine" in t or "antigen" in t or "complement" in t: return "immune"
    if "apopto" in t or "cell death" in t or "pyropto" in t: return "cell-death"
    if comp=="mitochondrion": return "metabolism"
    if comp in("plasma membrane","membrane"): return "transport/uptake"
    if comp=="extracellular": return "trafficking/secretion"
    if comp=="cytoskeleton": return "structure/cytoskeleton"
    return "other"
# integrated map (localized proteins)
rows=[r for r in csv.DictReader(open(OUT/"integrated_cell_human.csv")) if r["gene"] and loc.get(r["gene"],"unknown")!="unknown"]
idx={r["gene"]:i for i,r in enumerate(rows)}
def fv(v,d=-1.0):
    try:return float(v)
    except:return d
G=[]
for r in rows:
    g=r["gene"]; comp=loc.get(g,"unknown"); istf=r["is_tf"]=="1"
    G.append(dict(name=g,comp=comp,proc=process(g,comp,istf),
        ess=1 if r["essential"]=="1" else (0 if r["essential"]=="0" else -1),
        ess_src="measured" if r["essential"] in("0","1") else "none",
        loeuf=round(fv(r["loeuf"]),3),tf=int(istf),ppi=int(fv(r["ppi_degree"],0)),
        ndis=int(fv(r["n_diseases"],0)),master=r["lineage_master"],npath=int(fv(r["n_pathways"],0)),
        chrom=r.get("chrom","") or "",tss=r.get("tss","") or "",
        cpg=1 if r.get("cpg_promoter")=="1" else 0,enh=int(fv(r.get("enhancers"),0)),
        path=toppath(g)[:44]))
# === DepMap MEASURED essentiality (1,100 cancer cell lines) — the ground truth, overrides all ===
dep=OUT/"depmap_essentiality.csv"
if dep.exists():
    n=0
    for r in csv.DictReader(open(dep)):
        i=idx.get(r["gene"])
        if i is not None:
            G[i]["ess"]=1 if r["essential"]=="1" else 0
            G[i]["ess_src"]="measured"; G[i]["dep_frac"]=round(fv(r["frac_dep"],0),2); n+=1
    print("DepMap measured essentiality: set",n,"genes (overrides predictions)")
else:
    print("DepMap absent -> essentiality from Hart labels + model")
# === MODEL 1 enrichment: our trained essentiality model fills the UNLABELED genes ===
# (produced by the notebook -> outputs/orphan/predicted_essentiality.csv: gene,pred,prob)
m1=OUT/"predicted_essentiality.csv"
if m1.exists():
    n_fill=0
    for r in csv.DictReader(open(m1)):
        i=idx.get(r["gene"])
        if i is not None and G[i]["ess"]==-1:
            G[i]["ess"]=1 if r["pred"]=="1" else 0; G[i]["ess_src"]="model1"; G[i]["ess_prob"]=round(fv(r["prob"],0),3); n_fill+=1
    print("Model 1 (our essentiality model): filled",n_fill,"unlabeled genes")
else:
    print("Model 1: predicted_essentiality.csv absent -> using measured labels only")
# networks for perturbation propagation
reg=[];
for r in csv.DictReader(open(H/"collectri.tsv"),delimiter="\t"):
    a=idx.get(r["source_genesymbol"]); b=idx.get(r["target_genesymbol"])
    if a is not None and b is not None and a!=b:
        sg=1 if r.get("is_stimulation")=="True" else (-1 if r.get("is_inhibition")=="True" else 0)
        reg.append([a,b,sg])
ppi=[]
for l in gzip.open(H/"string_physical.txt.gz","rt"):
    if l.startswith("protein1"): continue
    a,b,s=l.split()
    if int(s)<700: continue
    ga=ensp2sym.get(a); gb=ensp2sym.get(b)
    if ga in idx and gb in idx and ga!=gb: ppi.append([idx[ga],idx[gb]])
# HIV hijack map
def normhiv(n):
    n=n.lower()
    for k,v in {"tat":"Tat","rev":"Rev","nef":"Nef","vif":"Vif","vpr":"Vpr","vpu":"Vpu","retropepsin":"Protease","protease":"Protease","integrase":"Integrase","reverse transcriptase":"RT","gp120":"gp120(Env)","gp41":"gp41(Env)","envelope":"Env","gag":"Gag","capsid":"Capsid(Gag)","matrix":"Matrix(Gag)","nucleocapsid":"Gag"}.items():
        if k in n: return v
    return n.split()[0].title()
hiv=defaultdict(list); hivhost=defaultdict(set)
for l in open(H/"hiv_interactions"):
    if l.startswith("#"): continue
    p=l.rstrip("\n").split("\t")
    if len(p)<9 or p[0]!="11676": continue
    hp=normhiv(p[3]); host=entrez2sym.get(p[6]); typ=p[4]
    if host and host in idx:
        hiv[hp].append([idx[host],typ]); hivhost[host].add(hp)
# HIV weak points: host dependency factors = host genes HIV binds/needs that are essential or hubs
DEP=["binds","complexes with","interacts with","incorporates","requires","activated by","enhanced by"]
def is_dep(host):
    i=idx[host]
    for hp in hivhost[host]:
        for i2,t in hiv[hp]:
            if i2==i and any(dep in t for dep in DEP): return True
    return False
weak=[]
for host,hps in hivhost.items():
    g=G[idx[host]]
    if (g["ess"]==1 or g["ppi"]>=25) and is_dep(host):
        weak.append(dict(gene=host,by=sorted(hps),ess=g["ess"],ppi=g["ppi"],comp=g["comp"]))
weak.sort(key=lambda x:-x["ppi"])
# dark genes: no pathway, no disease, low PPI, unknown-ish (function frontier)
dark=[i for i,g in enumerate(G) if g["npath"]==0 and g["ndis"]==0 and g["path"]=="" ]
darkset=set(dark)
for i,g in enumerate(G): g["dark"]=1 if i in darkset else 0
# curated core metabolic reactions (enzyme genes -> substrate -> product), so the cell can
# show REACTIONS, not just proteins. Only genes present in the map are kept per reaction.
RAW_RX=[
 ("glycolysis","HK1","glucose","glucose-6-P"),("glycolysis","GPI","glucose-6-P","fructose-6-P"),
 ("glycolysis","PFKL","fructose-6-P","fructose-1,6-bisP"),("glycolysis","ALDOA","fructose-1,6-bisP","G3P + DHAP"),
 ("glycolysis","GAPDH","G3P","1,3-bisphosphoglycerate"),("glycolysis","PGK1","1,3-BPG","3-phosphoglycerate"),
 ("glycolysis","ENO1","2-phosphoglycerate","phosphoenolpyruvate"),("glycolysis","PKM","phosphoenolpyruvate","pyruvate"),
 ("pyruvate->acetyl-CoA","PDHA1","pyruvate","acetyl-CoA"),("pyruvate->acetyl-CoA","LDHA","pyruvate","lactate"),
 ("TCA cycle","CS","acetyl-CoA + OAA","citrate"),("TCA cycle","ACO2","citrate","isocitrate"),
 ("TCA cycle","IDH2","isocitrate","alpha-ketoglutarate"),("TCA cycle","OGDH","alpha-KG","succinyl-CoA"),
 ("TCA cycle","SUCLA2","succinyl-CoA","succinate"),("TCA cycle","SDHA","succinate","fumarate"),
 ("TCA cycle","FH","fumarate","malate"),("TCA cycle","MDH2","malate","oxaloacetate"),
 ("oxidative phosphorylation","NDUFS1","NADH","NAD+ + H+ (Complex I)"),("oxidative phosphorylation","SDHB","FADH2","FAD (Complex II)"),
 ("oxidative phosphorylation","UQCRC1","ubiquinol","cytochrome c (Complex III)"),("oxidative phosphorylation","MT-CO1","cytochrome c","H2O (Complex IV)"),
 ("oxidative phosphorylation","ATP5F1A","ADP + Pi","ATP (Complex V)"),
 ("pentose phosphate","G6PD","glucose-6-P","6-phosphogluconolactone + NADPH"),
 ("fatty acid synthesis","FASN","acetyl-CoA + malonyl-CoA","palmitate"),("fatty acid oxidation","CPT1A","fatty acyl-CoA","acylcarnitine (into mito)"),
 ("urea cycle","OTC","ornithine + carbamoyl-P","citrulline"),("urea cycle","ASS1","citrulline + aspartate","argininosuccinate"),
 ("urea cycle","ARG1","arginine","urea + ornithine"),
 ("nucleotide synthesis","IMPDH1","IMP","XMP -> GMP"),("nucleotide synthesis","TYMS","dUMP","dTMP"),
 ("serine/glycine","PHGDH","3-phosphoglycerate","3-phosphohydroxypyruvate"),("folate","MTHFR","5,10-methylene-THF","5-methyl-THF"),
]
reactions=[]
for pw,enz,sub,prod in RAW_RX:
    if enz in idx: reactions.append(dict(enz=enz,i=idx[enz],sub=sub,prod=prod,pathway=pw))
# genome-scale metabolism from Human-GEM: real reaction (substrate=>product) per enzyme gene
generxn=defaultdict(list); ENSG=re.compile(r"ENSG\d+")
hg=H/"human_gem.txt"
if hg.exists():
    seen=set()
    with open(hg) as f:
        next(f)
        for l in f:
            p=l.split("\t")
            if len(p)<3: continue
            formula=p[1].strip(); gpr=p[2]
            if not formula or "=>" not in formula and "<=>" not in formula: continue
            eq=formula.replace(" <=> "," ⇌ ").replace(" => "," → ")
            if len(eq)>90: eq=eq[:88]+"…"
            for ensg in set(ENSG.findall(gpr)):
                s=ensg2sym.get(ensg)
                i=idx.get(s) if s else None
                if i is None: continue
                key=(i,eq)
                if key in seen or len(generxn[i])>=8: continue
                seen.add(key); generxn[i].append(eq)
    print("Human-GEM: reactions attached to",len(generxn),"enzyme genes")
else:
    print("Human-GEM absent -> only",len(reactions),"curated reactions")
# === MODEL 2 enrichment: cell-type master TFs derived from the atlas (Tabula Sapiens) ===
# (produced by the notebook -> outputs/orphan/celltype_masters.json: {cell_type:[TF,...]})
celltypes={"hepatocyte":["HNF4A","HNF1A","FOXA2"],"cardiac muscle cell":["NKX2-5","GATA4","TBX5"],
     "natural killer cell":["EOMES","TBX21"],"macrophage":["SPI1","CEBPB"],"endothelial cell":["ERG","FLI1"],
     "T cell":["GATA3","TCF7"],"neuron":["NEUROG2","NEUROD1"],"CD4 T cell(HIV target)":["GATA3","TCF7","CD4"]}
m2=OUT/"celltype_masters.json"; ct_src="curated defaults"
if m2.exists():
    dd=json.load(open(m2))
    if dd: celltypes={k:v for k,v in dd.items()}; ct_src="atlas-derived (Model 2)"
print("Model 2 (cell-type layer):",len(celltypes),"cell types from",ct_src)
# === MODEL 3 enrichment: Geneformer in-silico perturbation direction ===
# (produced by the notebook -> outputs/orphan/gf_perturb.json: {gene:[downstream_gene,...]})
gf={}
m3=OUT/"gf_perturb.json"
if m3.exists():
    raw=json.load(open(m3))
    for g,ds in raw.items():
        if g in idx: gf[idx[g]]=[idx[d] for d in ds if d in idx][:40]
    print("Model 3 (Geneformer perturbation):",len(gf),"genes with in-silico downstream targets")
else:
    print("Model 3: gf_perturb.json absent -> cascade uses measured reg+PPI graph only")
# === mutation -> structure effect (curated) + Open Targets disease/druggability ===
struct={}
try:
    sm=json.load(open(OUT/"structure_mutations.json"))["per_gene"]
    struct={g:v for g,v in sm.items() if g in idx}
except Exception as e: print("no structure_mutations:",e)
fold={}
try:
    for x in json.load(open(OUT/"fold_before_after.json")):
        if x["gene"] in idx: fold[x["gene"]]=x
except Exception as e: print("no fold_before_after:",e)
otdis={}
try:
    for t in json.load(open(OUT/"ot_disease_expansion.json"))["per_tf"]:
        if t["gene"] in idx: otdis[t["gene"]]=dict(top=t.get("top",[])[:3],druggable=t.get("druggable",False),
            ndis=t.get("n_diseases",0),ev=t.get("dominant_evidence",""))
except Exception as e: print("no ot_disease:",e)
print("structure genes:",len(struct),"| fold examples:",len(fold),"| OT disease genes:",len(otdis))
# pathway -> member gene indices (from each gene's displayed top pathway); keep sizeable ones
pathmembers=defaultdict(list)
for i,g in enumerate(G):
    if g["path"]: pathmembers[g["path"]].append(i)
pathmembers={k:v for k,v in pathmembers.items() if 3<=len(v)<=400}
pathlist=sorted(pathmembers,key=lambda k:-len(pathmembers[k]))[:60]  # 60 biggest for the selector
pathsel={k:pathmembers[k] for k in pathlist}
print("pathways indexed:",len(pathmembers),"| offered in selector:",len(pathsel))
# === EXTERNAL LAYERS: UniProt PTMs + acc->sym, SIGNOR signaling, Complex Portal, DGIdb drugs,
#     CellPhoneDB ligand-receptor, Reactome cell-cycle phase ===
acc2sym={}; ptm={}
up=H/"uniprot_acc_ptm.tsv"
if up.exists():
    for l in open(up,encoding="utf-8",errors="ignore"):
        p=l.rstrip("\n").split("\t")
        if len(p)<2 or p[0]=="Entry": continue
        acc,sym=p[0],p[1]; mod=p[2] if len(p)>2 else ""
        if sym: acc2sym[acc]=sym
        if mod and sym in idx:
            n=mod.count("MOD_RES")
            cats=[kw for kw in["Phospho","Acetyl","Methyl","Ubiquitin","Sumoyl","Hydroxy","GlcNAc","Nitros","Malonyl","Succinyl","Palmit"] if kw.lower() in mod.lower()]
            if n: ptm[idx[sym]]=dict(n=n,c=cats[:5])
print("UniProt: acc->sym",len(acc2sym),"| PTM-annotated genes",len(ptm))
sig=[]; sgf=H/"signor.tsv"
if sgf.exists():
    seen=set()
    for l in open(sgf,encoding="utf-8",errors="ignore"):
        p=l.split("\t")
        if len(p)<9: continue
        a=idx.get(p[0]); b=idx.get(p[4])
        if a is None or b is None or a==b: continue
        eff=p[8].lower(); s=1 if "up-regulat" in eff else(-1 if "down-regulat" in eff else 0)
        k=(a,b)
        if k in seen: continue
        seen.add(k); sig.append([a,b,s])
print("SIGNOR signaling edges:",len(sig))
complexes={}; gene2cplx=defaultdict(list); cpf=H/"complexportal.tsv"
if cpf.exists():
    rd=csv.reader(open(cpf),delimiter="\t"); next(rd,None)
    for row in rd:
        if len(row)<5: continue
        name=row[1]; syms=set()
        for tok in row[4].split("|"):
            acc=tok.split("(")[0].split("-")[0]
            s=acc2sym.get(acc)
            if s and s in idx: syms.add(s)
        if len(syms)>=2:
            complexes[name]=[idx[s] for s in sorted(syms)]
            for s in syms:
                if len(gene2cplx[idx[s]])<4: gene2cplx[idx[s]].append(name)
print("Complex Portal complexes:",len(complexes))
drugs=defaultdict(list); dgf=H/"dgidb.tsv"
if dgf.exists():
    rows_dg=list(csv.reader(open(dgf),delimiter="\t"))[1:]
    # prefer approved, named drugs (skip raw ChEMBL/ID-like names)
    rows_dg.sort(key=lambda r:0 if (len(r)>10 and r[10] in("TRUE","True")) else 1)
    seen=set()
    for row in rows_dg:
        if len(row)<10: continue
        i=idx.get(row[2]); drug=(row[9] or "").strip()
        if i is None or not drug or drug.lower().startswith("chembl"): continue
        typ=row[5] if row[5] not in("NULL","","N/A") else ""
        k=(i,drug.lower())
        if k in seen or len(drugs[i])>=8: continue
        seen.add(k); drugs[i].append(dict(d=drug.title(),t=typ,a=(len(row)>10 and row[10] in("TRUE","True"))))
print("DGIdb drug-targeted genes:",len(drugs))
lr=[]; cpdb=H/"cellphonedb.csv"
if cpdb.exists():
    rd=csv.DictReader(open(cpdb)); seen=set()
    for row in rd:
        a=acc2sym.get((row.get("partner_a") or "").strip()); b=acc2sym.get((row.get("partner_b") or "").strip())
        ia=idx.get(a) if a else None; ib=idx.get(b) if b else None
        if ia is not None and ib is not None and ia!=ib and (ia,ib) not in seen:
            seen.add((ia,ib)); lr.append([ia,ib])
print("CellPhoneDB ligand-receptor pairs:",len(lr))
cellcycle={}
for i,g in enumerate(G):
    ps=" ".join(paths.get(g["name"],[]))
    if "Cell Cycle" in ps or "Mitotic" in ps or "M Phase" in ps:
        ph=next((k for k in["G1/S Transition","G2/M","S Phase","M Phase","Mitotic","G1","G2"] if k in ps),"cell cycle")
        cellcycle[i]=ph
print("Reactome cell-cycle genes:",len(cellcycle))
# === MODEL 2: abundance + cell-type expression (drives abundance, cell-type wiring, differentiation) ===
# celltype_expression.csv = cell types (rows) x genes (cols); log1p CP10k means from the atlas.
ctnames=[]; abund={}; emask={}
cte=OUT/"celltype_expression.csv"
if cte.exists():
    rd=csv.reader(open(cte)); header=next(rd)
    gcols=header[1:]; mat=[]
    for row in rd:
        ctnames.append(row[0]); mat.append([float(x) if x else 0.0 for x in row[1:]])
    T=len(ctnames); THR=1.0    # "expressed" if log1p CP10k > 1 in that cell type
    for ci,gname in enumerate(gcols):
        gi=idx.get(gname)
        if gi is None: continue
        vals=[mat[t][ci] for t in range(T)]
        mx=max(vals) if vals else 0.0
        if mx<=0: continue
        abund[gi]=min(15,int(round(mx*2)))          # baseline abundance 0-15
        m=0
        for t,v in enumerate(vals):
            if v>THR: m|=(1<<t)
        if m: emask[gi]=str(m)                        # bitmask over cell types (string: can exceed 2^53)
    print("Model 2 expression: abundance for",len(abund),"genes across",T,"cell types")
else:
    print("Model 2: no celltype_expression.csv -> no abundance / cell-type wiring / differentiation")
DATA=dict(genes=G,reg=reg,ppi=ppi,reactions=reactions,generxn={k:v for k,v in generxn.items()},gf_perturb=gf,
    struct=struct,fold=fold,otdis=otdis,pathways=pathsel,
    sig=sig,complexes=complexes,gene2cplx={k:v for k,v in gene2cplx.items()},
    drugs={k:v for k,v in drugs.items()},lr=lr,ptm=ptm,cellcycle=cellcycle,
    ctnames=ctnames,abund=abund,emask=emask,
    procs=sorted(set(g["proc"] for g in G)),comps=sorted(set(g["comp"] for g in G)),
    hiv={k:v for k,v in hiv.items()},hiv_targets={k:len(v) for k,v in hiv.items()},
    hiv_weakpoints=weak[:40],dark_count=len(dark),celltypes=celltypes)
json.dump(DATA,open(OUT/"cell_complete.json","w"),separators=(",",":"))
from collections import Counter
print("proteins:",len(G),"| reg edges:",len(reg),"| ppi edges:",len(ppi))
print("processes:",dict(Counter(g["proc"] for g in G).most_common()))
print("HIV proteins mapped:",len(hiv),"-> host targets:",{k:len(v) for k,v in sorted(hiv.items(),key=lambda x:-len(x[1]))[:8]})
print("HIV host-dependency weak points:",len(weak),"e.g.",[w["gene"] for w in weak[:10]])
print("dark genes (no pathway/disease):",len(dark))
print("curated reactions present:",len(reactions),"across",len(set(r["pathway"] for r in reactions)),"pathways")
print("wrote cell_complete.json (%d KB)"%(len(json.dumps(DATA))//1024))
