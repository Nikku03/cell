"""Assemble the COMPLETE cell data model: every localized protein with its compartment,
cellular PROCESS (transcription/translation/trafficking/transport/metabolism/replication/
signaling/degradation/...), pathway, all our layers, the regulatory + PPI networks (for
perturbation propagation), HIV hijack map, and dark-gene flags. -> cell_complete.json
(consumed by the interactive perturbable cell app)."""
import csv, json, gzip, re
from collections import defaultdict, Counter
from pathlib import Path
OUT=Path("outputs/orphan"); H=Path("data/external_data/human")
loc=json.load(open(H/"gene_compartment.json"))
# Entrez -> symbol (for HIV host mapping)
entrez2sym={}; ensg2sym={}
with gzip.open(H/"gene_info.gz","rt") as f:
    next(f)
    for l in f:
        p=l.split("\t")
        if len(p)>2 and p[0]=="9606":
            entrez2sym[p[1]]=p[2]
            if len(p)>5 and "Ensembl:" in p[5]:              # dbXrefs -> ENSG for HuRI mapping
                for x in p[5].split("|"):
                    if x.startswith("Ensembl:ENSG"): ensg2sym[x.split(":")[1]]=p[2]
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
# DoRothEA + TRRUST curated TF->target (union with CollecTRI, dedup, keep a sign if any source has one)
regsign={(e[0],e[1]):e for e in reg}
def add_reg(a,b,sg):
    if a is None or b is None or a==b: return
    if (a,b) in regsign:
        if sg and not regsign[(a,b)][2]: regsign[(a,b)][2]=sg
    else:
        e=[a,b,sg]; reg.append(e); regsign[(a,b)]=e
if (H/"dorothea.tsv").exists():
    for r in csv.DictReader(open(H/"dorothea.tsv"),delimiter="\t"):
        sg=1 if r.get("is_stimulation")=="True" else (-1 if r.get("is_inhibition")=="True" else 0)
        add_reg(idx.get(r.get("source_genesymbol")),idx.get(r.get("target_genesymbol")),sg)
if (H/"trrust_human.tsv").exists():
    for l in open(H/"trrust_human.tsv"):
        p=l.rstrip("\n").split("\t")
        if len(p)>=3: add_reg(idx.get(p[0]),idx.get(p[1]),1 if p[2]=="Activation" else (-1 if p[2]=="Repression" else 0))
nCollec=len(reg)
# ReMap ChIP-seq TF->target (measured binding) — unsigned candidate regulation
if (OUT/"remap_tf_targets.tsv").exists():
    regset={(e[0],e[1]) for e in reg}
    for l in open(OUT/"remap_tf_targets.tsv"):
        p=l.rstrip("\n").split("\t"); a=idx.get(p[0]); b=idx.get(p[1]) if len(p)>1 else None
        if a is not None and b is not None and a!=b and (a,b) not in regset:
            regset.add((a,b)); reg.append([a,b,0])
# GTEx trans-eQTL source->eGene — unsigned candidate cross-gene regulation
if (OUT/"gtex_trans_edges.tsv").exists():
    regset={(e[0],e[1]) for e in reg}
    for l in open(OUT/"gtex_trans_edges.tsv"):
        p=l.rstrip("\n").split("\t"); a=idx.get(p[0]); b=idx.get(p[1]) if len(p)>1 else None
        if a is not None and b is not None and a!=b and (a,b) not in regset:
            regset.add((a,b)); reg.append([a,b,0])
# Causal regulome (Phase 3): ReMap-binding x Perturb-seq-response -> SIGNED, high-confidence edges
n_causal=0
if (OUT/"causal_reg.tsv").exists():
    regmap={(e[0],e[1]):e for e in reg}
    for l in open(OUT/"causal_reg.tsv"):
        p=l.rstrip("\n").split("\t")
        if len(p)<3: continue
        a=idx.get(p[0]); b=idx.get(p[1])
        if a is None or b is None or a==b: continue
        sg=int(p[2])
        if (a,b) in regmap: regmap[(a,b)][2]=sg          # upgrade the candidate edge with a measured sign
        else: e=[a,b,sg]; reg.append(e); regmap[(a,b)]=e
        n_causal+=1
print(f"regulatory edges: CollecTRI {nCollec} + ReMap/GTEx {len(reg)-nCollec-0} total {len(reg)} | causal (signed, binding x response) {n_causal}")
ppi=[]; ppiset=set()
def add_ppi(ga,gb):
    a=idx.get(ga); b=idx.get(gb)
    if a is None or b is None or a==b: return
    k=(a,b) if a<b else (b,a)
    if k not in ppiset: ppiset.add(k); ppi.append([a,b])
for l in gzip.open(H/"string_physical.txt.gz","rt"):        # STRING physical >=700
    if l.startswith("protein1"): continue
    a,b,s=l.split()
    if int(s)<700: continue
    add_ppi(ensp2sym.get(a),ensp2sym.get(b))
nS=len(ppi)
# MEASURED interactome (AP-MS): BioPlex 3.0 high-confidence + OpenCell proximity-labeling
if (H/"bioplex.tsv").exists():
    for r in csv.DictReader(open(H/"bioplex.tsv"),delimiter="\t"):
        try:
            if float(r["pInt"])>=0.75: add_ppi(r["SymbolA"],r["SymbolB"])
        except: pass
nB=len(ppi)
if (H/"opencell.csv").exists():
    for r in csv.DictReader(open(H/"opencell.csv")):
        add_ppi(r.get("target_gene_name"),r.get("interactor_gene_name"))
nO=len(ppi)
# HuRI — systematic, unbiased Y2H binary interactome (orthogonal to AP-MS above); ENSG-ENSG edges
if (H/"huri.tsv").exists():
    for l in open(H/"huri.tsv"):
        p=l.replace(",","\t").split("\t")
        if len(p)>=2:
            ga=ensg2sym.get(p[0].strip()); gb=ensg2sym.get(p[1].strip())
            if ga and gb: add_ppi(ga,gb)
print(f"PPI edges: STRING {nS} + BioPlex {nB-nS} + OpenCell {nO-nB} + HuRI {len(ppi)-nO} = {len(ppi)} total measured interactions")
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
# celltype_expression.csv is log1p CP10k means. Orientation varies (fresh census = celltypes x genes;
# a restored file may be genes x celltypes), so auto-detect which axis is genes by matching our index.
ctnames=[]; abund={}; emask={}
cte=OUT/"celltype_expression.csv"
if cte.exists():
    rd=csv.reader(open(cte)); header=next(rd)
    rows=[r for r in rd if r]
    hdr_hits=sum(1 for h in header[1:] if h in idx)                 # genes-as-columns?
    row_hits=sum(1 for r in rows if r and r[0] in idx)              # genes-as-rows?
    if row_hits>hdr_hits:                                           # file is genes x celltypes -> transpose logic
        ctnames=header[1:]; T=len(ctnames); THR=1.0
        for r in rows:
            gi=idx.get(r[0])
            if gi is None: continue
            vals=[float(x) if x else 0.0 for x in r[1:1+T]]
            mx=max(vals) if vals else 0.0
            if mx<=0: continue
            abund[gi]=min(15,int(round(mx*2)))
            m=0
            for t,v in enumerate(vals):
                if v>THR: m|=(1<<t)
            if m: emask[gi]=str(m)
    else:                                                           # file is celltypes x genes (fresh census)
        gcols=header[1:]; mat=[]
        for row in rows:
            ctnames.append(row[0]); mat.append([float(x) if x else 0.0 for x in row[1:]])
        T=len(ctnames); THR=1.0
        for ci,gname in enumerate(gcols):
            gi=idx.get(gname)
            if gi is None: continue
            vals=[mat[t][ci] for t in range(T) if ci<len(mat[t])]
            mx=max(vals) if vals else 0.0
            if mx<=0: continue
            abund[gi]=min(15,int(round(mx*2)))
            m=0
            for t,v in enumerate(vals):
                if v>THR: m|=(1<<t)
            if m: emask[gi]=str(m)
    print(f"Model 2 expression: abundance for {len(abund)} genes across {len(ctnames)} cell types "
          f"(orientation: {'genes x celltypes' if row_hits>hdr_hits else 'celltypes x genes'})")
else:
    print("Model 2: no celltype_expression.csv -> no abundance / cell-type wiring / differentiation")
# === BUCKET 1: co-essentiality + synthetic lethality (DepMap), 3D loops, ensemble confidence ===
codep={}; sl=[]
cdp=OUT/"depmap_codep.json"
if cdp.exists():
    raw=json.load(open(cdp))
    for g,parts in raw.items():
        i=idx.get(g)
        if i is None: continue
        codep[i]=[[idx[p],r] for p,r in parts if p in idx][:8]
    codep={k:v for k,v in codep.items() if v}
    print("DepMap co-essential partners:",len(codep),"genes")
slp=OUT/"depmap_sl.json"
if slp.exists():
    for a,b,r in json.load(open(slp)):
        if a in idx and b in idx: sl.append([idx[a],idx[b],r])
    print("DepMap synthetic-lethal candidate pairs:",len(sl))
# 3D chromatin loops: map each gene TSS to a loop anchor -> distal looped region(s)
loops3d={}; lf=H/"hiccups_loops.txt.gz"
if lf.exists():
    by_chr=defaultdict(list)
    with gzip.open(lf,"rt") as f:
        next(f)
        for l in f:
            p=l.split("\t")
            if len(p)<6: continue
            try:
                c1,x1,x2,c2,y1,y2=p[0],int(p[1]),int(p[2]),p[3],int(p[4]),int(p[5])
            except: continue
            by_chr[c1].append((x1,x2,f"{c2}:{y1//1000}kb")); by_chr[c2].append((y1,y2,f"{c1}:{x1//1000}kb"))
    for i,g in enumerate(G):
        c=g["chrom"].replace("chr","");
        try: tss=int(g["tss"])
        except: continue
        hits=[d for (s,e,d) in by_chr.get(c,[]) if s<=tss<=e]
        if hits: loops3d[i]=hits[:5]
    print("3D chromatin loops mapped to",len(loops3d),"genes")
# ensemble confidence: agreement across measured (DepMap) + constraint (LOEUF) + model (M1)
n_conf=0; n_flag=0
for i,g in enumerate(G):
    votes=[]
    df=g.get("dep_frac")
    if df is not None: votes.append(1 if df>=0.5 else 0)
    if g["loeuf"]>=0: votes.append(1 if g["loeuf"]<0.35 else 0)
    if g["ess_src"]=="model1": votes.append(g["ess"])
    if len(votes)>=2:
        frac=sum(votes)/len(votes)
        g["conf"]="high" if frac in (0.0,1.0) else "split"; n_conf+=1
        if df is not None and df>=0.5 and g["loeuf"]>=0.7:
            g["flag"]="cancer-dependency: essential in cancer lines, loss-of-function tolerated in population"; n_flag+=1
        elif df is not None and df<0.1 and 0<=g["loeuf"]<0.2:
            g["flag"]="germline-constrained yet cancer-dispensable"; n_flag+=1
print("ensemble confidence set on",n_conf,"genes |",n_flag,"disagreement flags (novelty candidates)")
# === DARK-GENE FUNCTION from MEASURED neighbors (Perturb-seq + co-essentiality + PPI) ===
# Perturb-seq functional neighbors (measured perturbation-response similarity)
psn=defaultdict(list)
psf=OUT/"perturbseq_neighbors.json"
if psf.exists():
    raw=json.load(open(psf))
    for g,parts in raw.items():
        i=idx.get(g)
        if i is not None: psn[i]=[idx[p] for p,r in parts if p in idx][:8]
    print("Perturb-seq: measured functional neighbors for",len(psn),"genes")
# LINCS L1000 perturbation neighbors — merged into the same measured-neighbor layer
lcf=OUT/"lincs_neighbors.json"
if lcf.exists():
    raw=json.load(open(lcf)); nadd=0
    for g,parts in raw.items():
        i=idx.get(g)
        if i is None: continue
        have=set(psn.get(i,[]))
        for p,r in parts:
            j=idx.get(p)
            if j is not None and j not in have: psn[i].append(j); have.add(j)
        psn[i]=psn[i][:12]; nadd+=1
    print("LINCS L1000: merged perturbation neighbors, genes with neighbors now",len(psn))
# ARCHS4 co-expression neighbors (Phase 1) — an independent lens; kept as its own layer AND merged into darkfn pool
coexpr={}
cxf=OUT/"coexpr_neighbors.json"
if cxf.exists():
    raw=json.load(open(cxf))
    for g,parts in raw.items():
        i=idx.get(g)
        if i is None: continue
        coexpr[i]=[[idx[p],r] for p,r in parts if p in idx][:12]
        have=set(psn.get(i,[]))
        for p,r in parts:
            j=idx.get(p)
            if j is not None and j not in have and len(psn[i])<16: psn[i].append(j); have.add(j)
    print("ARCHS4 co-expression: neighbors for",len(coexpr),"genes (independent lens)")
# PPI adjacency for neighbor lookup
ppiadj=defaultdict(list)
for a,b in ppi: ppiadj[a].append(b); ppiadj[b].append(a)
darkfn={}
for i,g in enumerate(G):
    if not g["dark"]: continue
    # gather MEASURED functional neighbors: Perturb-seq (best) + co-essential + top PPI
    neigh=list(psn.get(i,[]))+[j for j,r in codep.get(i,[])]+ppiadj.get(i,[])[:6]
    if not neigh: continue
    votes=Counter(); ev=[]
    for j in neigh:
        p=G[j]["path"] or G[j]["proc"]
        if p and p!="other": votes[p]+=1; ev.append(G[j]["name"])
    if votes:
        pred,cnt=votes.most_common(1)[0]
        src="Perturb-seq+co-essentiality" if i in psn else ("co-essentiality" if i in codep else "interaction")
        darkfn[i]=dict(pred=pred,ev=ev[:5],n=len(neigh),conf="high" if cnt>=3 else "low",src=src)
print("dark genes with a predicted function (from measured neighbors):",len(darkfn),"of",len(dark))
# merge STRUCTURE (Foldseek) + DOMAIN (Pfam/InterPro) lenses — reach dark genes with no neighbor signal,
# and upgrade confidence where an independent lens agrees
for fn,tag in [("structure_function.json","structure"),("domain_function.json","domain")]:
    p=OUT/fn
    if not p.exists(): continue
    raw=json.load(open(p)); added=0; agreed=0
    for g,v in raw.items():
        i=idx.get(g)
        if i is None or not G[i]["dark"]: continue
        if i not in darkfn:
            darkfn[i]=dict(pred=v["pred"],ev=v.get("ev",[])[:5],n=v.get("n_hits",0),
                           conf=v.get("conf","low"),src=v.get("src",tag)); added+=1
        elif darkfn[i]["pred"]==v["pred"]:                 # independent lens agrees -> promote to high
            darkfn[i]["conf"]="high"; darkfn[i]["src"]+="+"+tag; agreed+=1
    print(f"  {tag} lens: +{added} newly-covered dark genes, {agreed} confirmed by agreement")
print("dark genes with a predicted function (all lenses):",len(darkfn),"of",len(dark))
# === MODEL 4: predicted perturbation response (trained on Perturb-seq, held-out validated) ===
model4={}; model4_meta={}
m4f=OUT/"model4_predictions.json"
if m4f.exists():
    raw=json.load(open(m4f)); model4_meta=raw.get("_meta",{})
    for g,v in raw.get("predictions",{}).items():
        i=idx.get(g)
        if i is not None: model4[str(i)]=v
    print("Model 4: predicted perturbation response for",len(model4),"never-perturbed genes | held-out lift %.1fx"%(model4_meta.get("lift",0)))
# ncRNA -> target regulatory layer (Phase 5): miRTarBase + LncTarD
ncrna={}
ncf=OUT/"ncrna_targets.json"
if ncf.exists():
    raw=json.load(open(ncf)); g2n=raw.get("gene2ncrna",{})
    for g,ncs in g2n.items():
        i=idx.get(g)
        if i is not None: ncrna[str(i)]=ncs[:20]
    print("ncRNA regulation: regulating ncRNAs for",len(ncrna),"genes")
# === TARGET INTELLIGENCE: literature coverage + ranked target-priority / white-space tables ===
lit={}
lp2=OUT/"literature_counts.json"
if lp2.exists(): lit=json.load(open(lp2))
for i,g in enumerate(G): g["pubs"]=lit.get(g["name"],0)
ti_priority=[]; ti_ws=[]
tpf=OUT/"ti_target_priority.json"
if tpf.exists(): ti_priority=[t for t in json.load(open(tpf)) if t["gene"] in idx][:200]
wsf=OUT/"ti_whitespace.json"
if wsf.exists(): ti_ws=[w for w in json.load(open(wsf)) if w["gene"] in idx][:200]
pri=set(t["gene"] for t in ti_priority); wss=set(w["gene"] for w in ti_ws)
for i,g in enumerate(G):
    if g["name"] in pri: g["ti"]="priority"
    elif g["name"] in wss: g["ti"]="whitespace"
print("Target Intelligence: literature on",len(lit),"genes | priority table",len(ti_priority),"| white-space",len(ti_ws))
# biomarkers of sensitivity (#4/#5): per-target expression + mutation correlates (CCLE)
biomarkers={}
bmf=OUT/"biomarkers.json"
if bmf.exists():
    raw=json.load(open(bmf)); biomarkers={g:v for g,v in raw.items() if g in idx}
    print("biomarkers (CCLE expression + mutation) for",len(biomarkers),"targets")
DATA=dict(genes=G,reg=reg,ppi=ppi,reactions=reactions,generxn={k:v for k,v in generxn.items()},gf_perturb=gf,
    codep=codep,sl=sl,loops3d=loops3d,ti_priority=ti_priority,ti_whitespace=ti_ws,biomarkers=biomarkers,
    struct=struct,fold=fold,otdis=otdis,pathways=pathsel,
    sig=sig,complexes=complexes,gene2cplx={k:v for k,v in gene2cplx.items()},
    drugs={k:v for k,v in drugs.items()},lr=lr,ptm=ptm,cellcycle=cellcycle,
    ctnames=ctnames,abund=abund,emask=emask,
    procs=sorted(set(g["proc"] for g in G)),comps=sorted(set(g["comp"] for g in G)),
    hiv={k:v for k,v in hiv.items()},hiv_targets={k:len(v) for k,v in hiv.items()},
    hiv_weakpoints=weak[:40],dark_count=len(dark),celltypes=celltypes,
    darkfn={str(k):v for k,v in darkfn.items()},
    model4=model4,model4_meta=model4_meta,
    coexpr={str(k):v for k,v in coexpr.items()},
    ncrna=ncrna)
json.dump(DATA,open(OUT/"cell_complete.json","w"),separators=(",",":"))
print("proteins:",len(G),"| reg edges:",len(reg),"| ppi edges:",len(ppi))
print("processes:",dict(Counter(g["proc"] for g in G).most_common()))
print("HIV proteins mapped:",len(hiv),"-> host targets:",{k:len(v) for k,v in sorted(hiv.items(),key=lambda x:-len(x[1]))[:8]})
print("HIV host-dependency weak points:",len(weak),"e.g.",[w["gene"] for w in weak[:10]])
print("dark genes (no pathway/disease):",len(dark))
print("curated reactions present:",len(reactions),"across",len(set(r["pathway"] for r in reactions)),"pathways")
print("wrote cell_complete.json (%d KB)"%(len(json.dumps(DATA))//1024))
