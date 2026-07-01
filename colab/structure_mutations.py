"""Mutations on protein STRUCTURE: are DISEASE mutations in the buried, conserved core
(function-breaking) while TOLERATED/ADAPTIVE variants are on the flexible surface? The
'unmutated core' is the important part; a mutation matters only if it lands there.

For each gene: AlphaFold structure (burial = contact number; pLDDT = order/flexibility) x
gnomAD clinvar (pathogenic) + common missense (population-tolerated) + adaptation variants.
"""
import json, urllib.request, re, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
OUT=Path("outputs/orphan"); SC=Path("/tmp/claude-0/-home-user-cell/558875a8-e44f-59af-a869-11202bf854d3/scratchpad")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# gene -> UniProt ACC (from human.faa headers)
acc={}
for l in open(SC/"human.faa"):
    if l.startswith(">"):
        m=re.match(r">\w+\|(\w+)\|",l); gn=[t[3:] for t in l.split() if t.startswith("GN=")]
        if m and gn: acc.setdefault(gn[0],m.group(1))
DISEASE=["LDLR","PAH","TP53","CFTR","VHL","RB1","MLH1","PTEN","HBB","G6PD"]
ADAPT=["SLC24A5","SLC45A2","MC1R","EPAS1","ACKR1"]
GENES=DISEASE+ADAPT

def http_json(url,data=None):
    req=urllib.request.Request(url,data=(json.dumps(data).encode() if data else None),
        headers={"Content-Type":"application/json"} if data else {})
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(req,timeout=90))
        except Exception: time.sleep(3)
    return None
def af_structure(ac):
    api=http_json(f"https://alphafold.ebi.ac.uk/api/prediction/{ac}")
    if not api: return None
    url=api[0]["pdbUrl"]
    try: txt=urllib.request.urlopen(url,timeout=90).read().decode()
    except Exception: return None
    ca={}  # resnum -> (x,y,z,plddt)
    for line in txt.splitlines():
        if line.startswith("ATOM") and line[12:16].strip()=="CA":
            rn=int(line[22:26]); x=float(line[30:38]); y=float(line[38:46]); z=float(line[46:54]); b=float(line[60:66])
            ca[rn]=(x,y,z,b)
    return ca
def burial_map(ca):
    rn=sorted(ca); xyz=np.array([ca[r][:3] for r in rn]); out={}
    for i,r in enumerate(rn):
        d=np.sqrt(((xyz-xyz[i])**2).sum(1)); out[r]=(int((d<10).sum())-1, ca[r][3])  # contacts, pLDDT
    return out
def gnomad_gene(sym):
    # clinvar (GRCh38) + common missense (GRCh37 r2_1); protein positions are build-independent
    cv=http_json("https://gnomad.broadinstitute.org/api",
        {"query":'{gene(gene_symbol:"%s",reference_genome:GRCh38){clinvar_variants{clinical_significance major_consequence hgvsp}}}'%sym})
    mm=http_json("https://gnomad.broadinstitute.org/api",
        {"query":'{gene(gene_symbol:"%s",reference_genome:GRCh37){variants(dataset:gnomad_r2_1){consequence hgvsp exome{ac an}}}}'%sym})
    return cv,mm
def ppos(h):
    m=re.search(r"p\.[A-Za-z]{3}(\d+)",h or ""); return int(m.group(1)) if m else None

pc=json.load(open(OUT/"population_constraint.json"))
adapt_pos={g["sym"]:ppos(g["top_diff_variant"]["hgvsp"]) for g in pc["genes"]
           if g["category"]=="adaptation" and g.get("top_diff_variant")}

data={"pathogenic":[], "common_benign":[], "adaptive":[]}
per_gene={}
for sym in GENES:
    ac=acc.get(sym)
    if not ac: print(f"{sym}: no ACC"); continue
    ca=af_structure(ac)
    if not ca: print(f"{sym}: no structure"); continue
    bm=burial_map(ca)
    cv,mm=gnomad_gene(sym)
    npath=0; ncommon=0
    if cv and cv.get("data",{}).get("gene"):
        for v in cv["data"]["gene"]["clinvar_variants"]:
            if v.get("major_consequence")!="missense_variant": continue
            sig=(v.get("clinical_significance") or "").lower()
            p=ppos(v.get("hgvsp"))
            if p in bm and "pathogenic" in sig and "conflicting" not in sig:
                data["pathogenic"].append(bm[p]); npath+=1
    if mm and mm.get("data",{}).get("gene"):
        for v in mm["data"]["gene"]["variants"]:
            if v.get("consequence")!="missense_variant" or not v.get("exome") or not v["exome"].get("an"): continue
            af=v["exome"]["ac"]/v["exome"]["an"]; p=ppos(v.get("hgvsp"))
            if p in bm and af>0.001: data["common_benign"].append(bm[p]); ncommon+=1
    if sym in adapt_pos and adapt_pos[sym] in bm:
        data["adaptive"].append(bm[adapt_pos[sym]])
    per_gene[sym]=dict(acc=ac,residues=len(ca),pathogenic=npath,common=ncommon)
    print(f"  {sym:8s} ({ac}) res={len(ca)} pathogenic={npath} common={ncommon}",flush=True)
    time.sleep(0.5)

def stats(key):
    a=np.array(data[key])
    if len(a)==0: return None
    return dict(n=len(a), mean_contacts=round(float(a[:,0].mean()),1), mean_plddt=round(float(a[:,1].mean()),1),
        frac_buried=round(float((a[:,0]>=16).mean()),2), frac_ordered=round(float((a[:,1]>=70).mean()),2))
print("\n=== mutations on structure: buried/ordered vs surface/flexible ===")
print(f"{'class':16s} {'n':>5s} {'contacts':>9s} {'pLDDT':>7s} {'%buried':>8s} {'%ordered':>9s}")
res={}
for k in ["pathogenic","common_benign","adaptive"]:
    s=stats(k); res[k]=s
    if s: print(f"{k:16s} {s['n']:>5d} {s['mean_contacts']:>9.1f} {s['mean_plddt']:>7.1f} {s['frac_buried']:>8.2f} {s['frac_ordered']:>9.2f}")

# figure
fig,ax=plt.subplots(figsize=(8,5.5))
col={"pathogenic":"#c0392b","common_benign":"#2980b9","adaptive":"#27ae60"}
for k in ["common_benign","pathogenic","adaptive"]:
    a=np.array(data[k])
    if len(a): ax.scatter(a[:,0],a[:,1],s=18,c=col[k],alpha=.45,edgecolor="none",label=f"{k} (n={len(a)})")
ax.set_xlabel("burial (contacts within 10A)  ->  buried/core"); ax.set_ylabel("pLDDT (structural order)")
ax.set_title("Where mutations sit on the protein: disease=core, tolerated=surface")
ax.legend(); ax.grid(alpha=.25)
plt.tight_layout(); plt.savefig(OUT/"structure_mutations.png",dpi=130)
json.dump(dict(per_gene=per_gene,by_class=res,
    hypothesis="pathogenic mutations sit in buried/ordered core; common+adaptive on flexible surface"),
    open(OUT/"structure_mutations.json","w"),indent=2)
print("\nwrote structure_mutations.png + .json")
