"""What PART of the protein does each mutation hit — active site, ligand/metal binding,
DNA-binding, disulfide, buried core, or surface? UniProt functional features + AlphaFold
3D distance to the nearest catalytic/binding residue. Compare pathogenic vs tolerated vs
adaptive."""
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
DISEASE=["LDLR","PAH","TP53","CFTR","VHL","RB1","MLH1","PTEN","HBB","G6PD"]
ADAPT=["SLC24A5","SLC45A2","MC1R","EPAS1","ACKR1"]; GENES=DISEASE+ADAPT
METALS={"zn","fe","mg","mn","cu","ca","ni","co","k","na","mo","cd"}
def hj(url,data=None):
    req=urllib.request.Request(url,data=(json.dumps(data).encode() if data else None),
        headers={"Content-Type":"application/json"} if data else {})
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(req,timeout=90))
        except Exception: time.sleep(3)
    return None
def af(ac):
    api=hj(f"https://alphafold.ebi.ac.uk/api/prediction/{ac}")
    if not api: return None
    try: txt=urllib.request.urlopen(api[0]["pdbUrl"],timeout=90).read().decode()
    except Exception: return None
    ca={}
    for ln in txt.splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip()=="CA":
            ca[int(ln[22:26])]=(float(ln[30:38]),float(ln[38:46]),float(ln[46:54]),float(ln[60:66]))
    return ca
def features(ac):
    d=hj(f"https://rest.uniprot.org/uniprotkb/{ac}.json?fields=ft_act_site,ft_binding,ft_site,ft_disulfid,ft_dna_bind,ft_transmem,ft_mod_res")
    cat=set(); bind=set(); metal=set(); dna=set(); ss=set(); tm=set(); anchor=set()
    for f in (d or {}).get("features",[]):
        loc=f["location"]; 
        try: s=int(loc["start"]["value"]); e=int(loc["end"]["value"])
        except: continue
        rng=range(s,e+1); t=f["type"]
        lig=((f.get("ligand") or {}).get("name","") or "").lower()
        if t=="Active site": cat|=set(rng); anchor|=set(rng)
        elif t=="Binding site":
            bind|=set(rng); anchor|=set(rng)
            if any(mm in lig for mm in METALS): metal|=set(rng)
        elif t=="DNA binding": dna|=set(rng); anchor|=set(rng)
        elif t=="Disulfide bond": ss|={s,e}
        elif t=="Transmembrane": tm|=set(rng)
    return dict(cat=cat,binding=bind,metal=metal,dna=dna,disulfide=ss,transmem=tm,anchor=anchor)
def ppos(h):
    m=re.search(r"p\.[A-Za-z]{3}(\d+)",h or ""); return int(m.group(1)) if m else None
def gnomad(sym):
    cv=hj("https://gnomad.broadinstitute.org/api",{"query":'{gene(gene_symbol:"%s",reference_genome:GRCh38){clinvar_variants{clinical_significance major_consequence hgvsp}}}'%sym})
    mm=hj("https://gnomad.broadinstitute.org/api",{"query":'{gene(gene_symbol:"%s",reference_genome:GRCh37){variants(dataset:gnomad_r2_1){consequence hgvsp exome{ac an}}}}'%sym})
    return cv,mm
pc=json.load(open(OUT/"population_constraint.json"))
adapt_pos={g["sym"]:ppos(g["top_diff_variant"]["hgvsp"]) for g in pc["genes"] if g["category"]=="adaptation" and g.get("top_diff_variant")}

def classify(pos,ca,ft,contacts):
    if pos in ft["cat"]: return "active_site"
    if pos in ft["metal"]: return "metal_binding"
    if pos in ft["binding"]: return "ligand_binding"
    if pos in ft["dna"]: return "dna_binding"
    if pos in ft["disulfide"]: return "disulfide"
    # 3D proximity to nearest anchor
    if ft["anchor"] and pos in ca:
        p=np.array(ca[pos][:3])
        dmin=min(np.sqrt(((p-np.array(ca[a][:3]))**2).sum()) for a in ft["anchor"] if a in ca) if any(a in ca for a in ft["anchor"]) else 99
        if dmin<=8: return "near_active_site"
    if pos in ft["transmem"]: return "transmembrane"
    return "buried_core" if contacts>=16 else "surface"

classes={"pathogenic":[], "common_benign":[], "adaptive":[]}
for sym in GENES:
    ac=acc.get(sym); 
    if not ac: continue
    ca=af(ac); ft=features(ac)
    if not ca: print(f"{sym}: no structure"); continue
    rn=sorted(ca); xyz=np.array([ca[r][:3] for r in rn])
    contacts={r:(int((np.sqrt(((xyz-xyz[i])**2).sum(1))<10).sum())-1) for i,r in enumerate(rn)}
    cv,mm=gnomad(sym); np_=nc=0
    if cv and cv.get("data",{}).get("gene"):
        for v in cv["data"]["gene"]["clinvar_variants"]:
            if v.get("major_consequence")!="missense_variant": continue
            sig=(v.get("clinical_significance") or "").lower(); p=ppos(v.get("hgvsp"))
            if p in ca and "pathogenic" in sig and "conflicting" not in sig:
                classes["pathogenic"].append(classify(p,ca,ft,contacts[p])); np_+=1
    if mm and mm.get("data",{}).get("gene"):
        for v in mm["data"]["gene"]["variants"]:
            if v.get("consequence")!="missense_variant" or not v.get("exome") or not v["exome"].get("an"): continue
            af_=v["exome"]["ac"]/v["exome"]["an"]; p=ppos(v.get("hgvsp"))
            if p in ca and af_>0.001: classes["common_benign"].append(classify(p,ca,ft,contacts[p])); nc+=1
    if sym in adapt_pos and adapt_pos[sym] in ca:
        p=adapt_pos[sym]; classes["adaptive"].append(classify(p,ca,ft,contacts[p]))
    print(f"  {sym:8s} path={np_} common={nc} feats(cat={len(ft['cat'])} bind={len(ft['binding'])} metal={len(ft['metal'])} dna={len(ft['dna'])} ss={len(ft['disulfide'])})",flush=True)
    time.sleep(0.4)
CATS=["active_site","metal_binding","ligand_binding","dna_binding","disulfide","near_active_site","transmembrane","buried_core","surface"]
from collections import Counter
print("\n=== what part of the protein does each mutation class hit (%) ===")
print(f"{'part':18s} "+" ".join(f"{k:>12s}" for k in classes))
out={}
for cat in CATS:
    row={}
    line=f"{cat:18s} "
    for k in classes:
        n=len(classes[k]); c=classes[k].count(cat); pct=100*c/n if n else 0; row[k]=round(pct,1)
        line+=f"{c:>4d}/{n:<3d}({pct:>4.0f}%)".rjust(13)
    out[cat]=row; print(line)
func=["active_site","metal_binding","ligand_binding","dna_binding","disulfide","near_active_site"]
print("\n=== AT or NEAR a functional site (active/binding/DNA/disulfide/near) ===")
for k in classes:
    n=len(classes[k]); f=sum(classes[k].count(c) for c in func); print(f"  {k:16s}: {100*f/n if n else 0:.0f}%  ({f}/{n})")
json.dump(dict(by_part=out,n=({k:len(v) for k,v in classes.items()})),open(OUT/"mutation_sites.json","w"),indent=2)
print("\nwrote mutation_sites.json")
