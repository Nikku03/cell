"""Validate the model's PREDICTIONS against independent gold standards (no train/test hiding):
co-essentiality vs known complexes/pathways; guilt-by-association pathway recovery (dark-gene
proxy); synthetic-lethal functional relatedness; biomarker method on textbook cases (direct).
Each vs a random baseline. -> validation_report.json"""
import json, gzip, csv, random
from collections import defaultdict
from pathlib import Path
random.seed(0)
OUT=Path("outputs/orphan"); H=Path("data/external_data/human")
ensp2sym={};ensp2ensg={}
for l in gzip.open(H/"string_aliases.txt.gz","rt"):
    p=l.rstrip("\n").split("\t")
    if len(p)<3: continue
    if p[2]=="Ensembl_HGNC_symbol": ensp2sym[p[0]]=p[1]
    elif p[2]=="Ensembl_HGNC_ensembl_gene_id": ensp2ensg[p[0]]=p[1]
ensg2sym={ensp2ensg[e]:ensp2sym[e] for e in ensp2sym if e in ensp2ensg}
pw=defaultdict(set)
for l in open(H/"reactome_human.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)<4 or not p[0].startswith("ENSG"): continue
    s=ensg2sym.get(p[0])
    if s: pw[s].add(p[3])
D=json.load(open(OUT/"cell_complete.json")); G=D["genes"]
cxpart=defaultdict(set)
for name,members in D["complexes"].items():
    ms=[G[i]["name"] for i in members]
    for a in ms:
        for b in ms:
            if a!=b: cxpart[a].add(b)
def related(a,b): return (b in cxpart.get(a,())) or bool(pw.get(a,set())&pw.get(b,set()))
allg=list(set(list(pw)+list(cxpart)))
def baseline(n):
    h=0
    for _ in range(n):
        a,b=random.sample(allg,2)
        if related(a,b): h+=1
    return h/n
base=baseline(5000)
codep=json.load(open(OUT/"depmap_codep.json")); tot=hit=0
for a,parts in codep.items():
    for b,r in parts[:5]:
        tot+=1
        if related(a,b): hit+=1
p1=hit/tot
tried=corr=0
for a,parts in codep.items():
    if not pw.get(a): continue
    votes=defaultdict(float)
    for b,r in parts[:8]:
        for p in pw.get(b,()): votes[p]+=1
    if not votes: continue
    tried+=1
    if max(votes,key=votes.get) in pw[a]: corr+=1
sl=json.load(open(OUT/"depmap_sl.json"))
sh=sum(1 for a,b,r in sl if related(a,b))
def load(fn):
    import numpy as np
    with open(H/fn) as f:
        rd=csv.reader(f); genes=[g.split(" (")[0] for g in next(rd)[1:]]; lines=[];M=[]
        for r in rd: lines.append(r[0]);M.append([float(x) if x else np.nan for x in r[1:]])
    return lines,genes,np.array(M,dtype=np.float32)
bio=[]
try:
    import numpy as np
    dl,dg,Dm=load("CRISPRGeneEffect.csv"); ml,mg,MU=load("CCLE_mutations_damaging.csv")
    di={a:i for i,a in enumerate(dl)};dgi={g:i for i,g in enumerate(dg)};mgi={g:i for i,g in enumerate(mg)}
    shm=[a for a in ml if a in di];mix=[ml.index(a) for a in shm];dmut=[di[a] for a in shm]
    Mb=np.nan_to_num(MU[mix])>0
    def delta(t,m):
        if t not in dgi or m not in mgi: return None
        d=Dm[dmut,dgi[t]];mk=Mb[:,mgi[m]];ok=~np.isnan(d)
        return float(np.mean(d[mk&ok])-np.mean(d[(~mk)&ok]))
    for t,m,exp in [("MDM2","TP53","resistant"),("CTNNB1","APC","sensitive")]:
        dv=delta(t,m); got="resistant" if (dv or 0)>0.1 else "sensitive" if (dv or 0)<-0.1 else "none"
        bio.append(dict(target=t,marker=m,expected=exp,delta=round(dv,3) if dv is not None else None,got=got,correct=got==exp))
except Exception as e:
    bio=[{"error":repr(e)[:120]}]
rep=dict(random_baseline=round(base,3),
 coessentiality=dict(share_complex_or_pathway=round(p1,3),enrichment_vs_random=round(p1/base,1),pairs=tot),
 guilt_by_association=dict(pathway_recovery_top1=round(corr/tried,3),genes_tested=tried),
 synthetic_lethal=dict(functionally_related=round(sh/len(sl),3),enrichment_vs_random=round((sh/len(sl))/base,1),pairs=len(sl)),
 biomarker_method=bio)
json.dump(rep,open(OUT/"validation_report.json","w"),indent=1)
print("=== VALIDATION vs gold standards (random baseline %.1f%%) ==="%(base*100))
print(f"1. Co-essential partners share a complex/pathway: {p1*100:.0f}%  ({p1/base:.1f}x random)  [n={tot}]")
print(f"2. Guilt-by-association recovers pathway (top-1):  {corr/tried*100:.0f}%  [n={tried}]  (dark-gene proxy)")
print(f"3. Synthetic-lethal pairs functionally related:   {sh/len(sl)*100:.0f}%  ({(sh/len(sl))/base:.1f}x random)  [n={len(sl)}]")
print("4. Biomarker method (textbook cases, direct):")
for c in bio:
    if "error" in c: print("   skipped:",c["error"]);break
    print(f"   {c['target']} x {c['marker']}: delta {c['delta']} -> {c['got']} (expect {c['expected']}) {'OK' if c['correct'] else 'x'}")
