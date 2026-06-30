"""Lay out a cell: E. coli K-12 MG1655, integrating every layer we built.
Per-gene blueprint + whole-cell composition + worked examples through all layers.
"""
import csv, json, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cobra, cobra.io
warnings.filterwarnings("ignore"); cobra.Configuration().solver="glpk"
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan"); MODELS=Path("data/external_data/bigg_models")

# --- gene parts list (name -> product, coords) ---
genes={}; product={}
for l in open(REG/"GeneProductSet.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=6 and p[3].isdigit() and p[4].isdigit() and p[5] in ("forward","reverse"):
        genes[p[1].lower()]=(int(p[3]),int(p[4]),p[5]); product[p[1].lower()]=p[2] if len(p)>2 else ""
name2b={r["gene_name"].lower():r["gene_id"] for r in csv.DictReader(open(REG/"TRN.csv")) if r.get("gene_name") and r.get("gene_id")}
# --- essentiality (Keio truth) ---
b_of={r["locus_tag"]:r["b_number"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
ess={}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"]=="beril_Keio":
        b=b_of.get(r["locus_tag"])
        if b: ess[b]=int(r["essential"])
# --- metabolic (iJO1366 genes) ---
m=cobra.io.read_sbml_model(str(MODELS/"iJO1366.xml.gz")); metab_b={g.id for g in m.genes}
# --- regulatory: TFs, regulon size, class, autoreg ---
SIGN={"activator":1,"repressor":-1}
edges=[]; tf_t=defaultdict(set)
for l in open(REG/"network_tf_gene.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.split("\t"); edges.append((p[0].lower(),p[1].lower(),SIGN.get(p[2].lower(),0))); tf_t[p[0].lower()].add(p[1].lower())
regs=set(tf_t); auto={}
for tf,tg,s in edges:
    if tg==tf: auto[tf]=s
regsize={t:len(tf_t[t]) for t in regs}
def tf_class(t):
    n=regsize[t]
    if n>=80: return "global"
    if n>=12: return "specific"
    return "minor"

# --- assemble per-gene blueprint ---
rows=[]
for g in genes:
    b=name2b.get(g,"")
    rows.append(dict(gene=g,b=b,product=product.get(g,"")[:40],
        essential=ess.get(b,""),
        metabolic=int(b in metab_b) if b else 0,
        is_TF=int(g in regs),
        tf_class=tf_class(g) if g in regs else "",
        regulon_size=regsize.get(g,0) if g in regs else 0,
        autoreg={1:"pos",-1:"neg"}.get(auto.get(g),"") if g in regs else ""))
with open(OUT/"cell_layout.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# --- whole-cell composition ---
N=len(rows)
n_ess=sum(1 for r in rows if r["essential"]==1)
n_metab=sum(r["metabolic"] for r in rows)
n_tf=sum(r["is_TF"] for r in rows)
n_glob=sum(1 for r in rows if r["tf_class"]=="global"); n_spec=sum(1 for r in rows if r["tf_class"]=="specific")
n_autoreg=sum(1 for r in rows if r["autoreg"]); n_negauto=sum(1 for r in rows if r["autoreg"]=="neg")
ess_metab=sum(1 for r in rows if r["essential"]==1 and r["metabolic"])
print(f"=== E. coli K-12 cell layout ===")
print(f"  total protein-coding genes laid out : {N}")
print(f"  essential (Keio)                    : {n_ess}")
print(f"  metabolic (iJO1366)                 : {n_metab}  (of which essential: {ess_metab})")
print(f"  transcription factors               : {n_tf}  (global {n_glob}, specific {n_spec})")
print(f"  autoregulated TFs ([TF] computable) : {n_autoreg}  (neg-autoreg {n_negauto})")

# --- worked examples through all layers ---
def show(g):
    r=next((x for x in rows if x["gene"]==g),None)
    if not r: return f"{g}: not found"
    tags=[]
    tags.append("essential" if r["essential"]==1 else ("non-ess" if r["essential"]==0 else "ess:?"))
    if r["metabolic"]: tags.append("metabolic(iJO1366)")
    if r["is_TF"]: tags.append(f"TF:{r['tf_class']}(regulon {r['regulon_size']}, autoreg {r['autoreg'] or 'none'})")
    return f"  {g:<6} {r['b']:<7} {r['product'][:32]:<33} -> {', '.join(tags)}"
print("\nworked examples through the layers:")
for g in ["rpoB","crp","trpr","sdha","gnt r".replace(' ',''),"lacz","fnr","dnaa"]:
    print(show(g))

# --- figure: cell composition ---
fig,ax=plt.subplots(figsize=(9,5))
cats=["all genes","essential\ncore","metabolic\n(iJO1366)","TFs\n(total)","TFs global","TFs specific","autoreg TFs\n([TF] computable)"]
vals=[N,n_ess,n_metab,n_tf,n_glob,n_spec,n_autoreg]
colors=["#94a3b8","#ef4444","#3b82f6","#a855f7","#7c3aed","#c084fc","#22c55e"]
ax.bar(range(len(vals)),vals,color=colors)
for i,v in enumerate(vals): ax.text(i,v+30,str(v),ha="center",fontsize=9)
ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats,fontsize=8)
ax.set_ylabel("genes"); ax.set_title("E. coli K-12 cell layout — laid-out components (genome -> blueprint)")
plt.tight_layout(); fig.savefig(OUT/"cell_layout.png",dpi=110)
json.dump(dict(total=N,essential=n_ess,metabolic=n_metab,essential_metabolic=ess_metab,
               TFs=n_tf,global_TF=n_glob,specific_TF=n_spec,autoreg_TF=n_autoreg,neg_autoreg=n_negauto),
          open(OUT/"cell_layout.json","w"),indent=2)
print(f"\nwrote {OUT}/cell_layout.csv, .json, .png")
