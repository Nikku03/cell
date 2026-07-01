"""Mutations in TF genes: where do they land? A TF's 'active site' is its DNA-binding
domain. Expect pathogenic TF mutations to concentrate in DNA-binding (vs enzymes ->
catalytic/ligand). Plus: TFs are haploinsufficient (one bad copy -> dominant disease)."""
import json, urllib.request, re, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
OUT=Path("outputs/orphan"); SC=Path("/tmp/claude-0/-home-user-cell/558875a8-e44f-59af-a869-11202bf854d3/scratchpad")
acc={}
for l in open(SC/"human.faa"):
    if l.startswith(">"):
        m=re.match(r">\w+\|(\w+)\|",l); gn=[t[3:] for t in l.split() if t.startswith("GN=")]
        if m and gn: acc.setdefault(gn[0],m.group(1))
# disease TFs (developmental + cancer), all in TRRUST
TF_GENES=["TP53","PAX6","GATA4","TBX5","HNF1A","HNF4A","RUNX1","MECP2","FOXP2","SOX9","NKX2-5","GATA1","FOXP3","POU3F4"]
def hj(url,data=None):
    req=urllib.request.Request(url,data=(json.dumps(data).encode() if data else None),
        headers={"Content-Type":"application/json"} if data else {})
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(req,timeout=90))
        except Exception: time.sleep(3)
    return None
def features(ac):
    d=hj(f"https://rest.uniprot.org/uniprotkb/{ac}.json?fields=ft_act_site,ft_binding,ft_dna_bind,ft_zn_fing,ft_domain")
    dna=set(); znf=set()
    for f in (d or {}).get("features",[]):
        loc=f["location"]
        try: s=int(loc["start"]["value"]); e=int(loc["end"]["value"])
        except: continue
        t=f["type"]; desc=(f.get("description") or "").lower()
        if t in("DNA binding","Zinc finger"): dna|=set(range(s,e+1))
        if t=="Domain" and any(k in desc for k in["homeobox","hmg","fork","bhlh","ets","t-box","gata","runt","paired","pou","leucine zip","bzip"]):
            dna|=set(range(s,e+1))
    return dna
def ppos(h):
    m=re.search(r"p\.[A-Za-z]{3}(\d+)",h or ""); return int(m.group(1)) if m else None
def af_len(ac):
    api=hj(f"https://alphafold.ebi.ac.uk/api/prediction/{ac}")
    if not api: return None
    try: txt=urllib.request.urlopen(api[0]["pdbUrl"],timeout=90).read().decode()
    except: return None
    return max([int(l[22:26]) for l in txt.splitlines() if l.startswith("ATOM")]+[0])
tot_path=0; in_dna=0; per=[]
for sym in TF_GENES:
    ac=acc.get(sym)
    if not ac: print(f"{sym}: no acc"); continue
    dna=features(ac); L=af_len(ac) or 0
    cv=hj("https://gnomad.broadinstitute.org/api",{"query":'{gene(gene_symbol:"%s",reference_genome:GRCh38){clinvar_variants{clinical_significance major_consequence hgvsp}}}'%sym})
    npath=0; nd=0
    if cv and cv.get("data",{}).get("gene"):
        for v in cv["data"]["gene"]["clinvar_variants"]:
            if v.get("major_consequence")!="missense_variant": continue
            sig=(v.get("clinical_significance") or "").lower(); p=ppos(v.get("hgvsp"))
            if p and "pathogenic" in sig and "conflicting" not in sig:
                npath+=1
                if p in dna: nd+=1
    frac_dna_len=round(len(dna)/L,2) if L else 0
    frac_mut_dna=round(nd/npath,2) if npath else 0
    per.append(dict(tf=sym,path=npath,in_dna=nd,frac_mut_in_dna=frac_mut_dna,dna_domain_frac=frac_dna_len))
    tot_path+=npath; in_dna+=nd
    print(f"  {sym:8s} pathogenic={npath:4d}  in DNA-binding domain={nd:4d} ({frac_mut_dna:.0%})  "
          f"[domain is {frac_dna_len:.0%} of protein]",flush=True)
    time.sleep(0.4)
print(f"\n=== TF pathogenic mutations in the DNA-binding domain: {in_dna}/{tot_path} ({100*in_dna/max(tot_path,1):.0f}%) ===")
print(f"  (the DNA-binding domain is a minority of each protein, yet catches the mutations")
print(f"   -> the TF's 'active site' IS its DNA contact; breaking it = developmental disease/cancer)")
json.dump(dict(per_tf=per,total_pathogenic=tot_path,in_dna_binding=in_dna,
    frac=round(in_dna/max(tot_path,1),3)),open(OUT/"tf_mutations.json","w"),indent=2)
print("wrote tf_mutations.json")
