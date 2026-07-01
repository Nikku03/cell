"""Does human POPULATION CONSTRAINT (gnomAD pLI / LOEUF / missense-z) predict gene
essentiality (Hart CEG/NEG) — where the FBA metabolic wheel failed? This is the human
analog of the bacterial conservation+fitness signal: 'can't take mutations across
people' = essential. Key test: does it cover the non-metabolic 83% FBA can't see?
"""
import gzip, csv, json, warnings
from pathlib import Path
import numpy as np
import cobra, cobra.io
warnings.filterwarnings("ignore"); cobra.Configuration().solver="glpk"
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
def readset(fn):
    s=set()
    for i,l in enumerate(open(H/fn)):
        if i==0: continue
        g=l.split("\t")[0].strip().strip('"')
        if g: s.add(g)
    return s
CEG=readset("CEGv2.txt"); NEG=readset("NEGv1.txt")
# gnomAD constraint
con={}
with gzip.open(H/"gnomad_constraint.txt.bgz","rt") as f:
    hdr=f.readline().rstrip("\n").split("\t")
    gi,pi,li,mi=hdr.index("gene"),hdr.index("pLI"),hdr.index("oe_lof_upper"),hdr.index("mis_z")
    for l in f:
        p=l.rstrip("\n").split("\t"); g=p[gi]
        def fv(x):
            try: return float(x)
            except: return None
        rec=dict(pLI=fv(p[pi]),loeuf=fv(p[li]),mis_z=fv(p[mi]))
        # keep the most-constrained transcript per gene (lowest LOEUF)
        if g not in con or (rec["loeuf"] is not None and (con[g]["loeuf"] is None or rec["loeuf"]<con[g]["loeuf"])):
            con[g]=rec
print(f"gnomAD constraint for {len(con)} genes; truth CEG {len(CEG)} NEG {len(NEG)}")

def auc(score,y):
    s=np.asarray(score,float); y=np.asarray(y)
    m=~np.isnan(s); s=s[m]; y=y[m]
    if y.sum()==0 or y.sum()==len(y): return float('nan'),0
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); mk=y==1
    return float((r[mk].sum()-mk.sum()*(mk.sum()-1)/2)/(mk.sum()*(~mk).sum())), int(m.sum())

def bench(genes_ceg,genes_neg,label):
    rows=[(g,1) for g in genes_ceg]+[(g,0) for g in genes_neg]
    rows=[(g,y) for g,y in rows if g in con]
    y=[y for _,y in rows]
    cov_c=sum(1 for g in genes_ceg if g in con); cov_n=sum(1 for g in genes_neg if g in con)
    def sc(key,sign): return [sign*(con[g][key] if con[g][key] is not None else np.nan) for g,_ in rows]
    a_pli,_=auc(sc("pLI",1),y)          # higher pLI = essential
    a_loeuf,n=auc(sc("loeuf",-1),y)     # lower LOEUF = essential
    a_mis,_=auc(sc("mis_z",1),y)        # higher missense-z = essential
    print(f"\n--- {label} (CEG cov {cov_c}/{len(genes_ceg)}, NEG cov {cov_n}/{len(genes_neg)}, n_eval {n}) ---")
    print(f"    pLI     AUC = {a_pli:.3f}")
    print(f"    LOEUF   AUC = {a_loeuf:.3f}   (the recommended constraint metric)")
    print(f"    mis_z   AUC = {a_mis:.3f}")
    return dict(label=label,ceg_cov=cov_c,ceg_total=len(genes_ceg),neg_cov=cov_n,neg_total=len(genes_neg),
                n_eval=n,auc_pLI=round(a_pli,3),auc_LOEUF=round(a_loeuf,3),auc_mis_z=round(a_mis,3))

res=[]
res.append(bench(CEG,NEG,"GENOME-WIDE (all essential vs non-essential)"))

# split by metabolic vs non-metabolic using Human-GEM
print("\nloading Human-GEM to split metabolic vs non-metabolic ...",flush=True)
ensg2sym={}
for r in csv.DictReader(open(H/"genes.tsv"),delimiter="\t"):
    if r.get("genes") and r.get("geneSymbols"): ensg2sym[r["genes"]]=r["geneSymbols"].split(";")[0]
m=cobra.io.read_sbml_model(str(H/"Human-GEM.xml"))
metab_syms={ensg2sym.get(g.id,g.name) for g in m.genes}
ceg_met=CEG&metab_syms; ceg_non=CEG-metab_syms
neg_met=NEG&metab_syms; neg_non=NEG-metab_syms
res.append(bench(ceg_met,neg_met|neg_non,"METABOLIC essentials (the 17% FBA can see)"))
res.append(bench(ceg_non,neg_non|neg_met,"NON-METABOLIC essentials (the 83% FBA CANNOT see)"))

# headline: constraint coverage of the non-metabolic essentials + how many are strongly constrained
strong=[g for g in ceg_non if g in con and con[g]["loeuf"] is not None and con[g]["loeuf"]<0.35]
print(f"\n=== HEADLINE ===")
print(f"  non-metabolic core-essential genes: {len(ceg_non)}")
print(f"  ... with gnomAD constraint data: {sum(1 for g in ceg_non if g in con)} "
      f"({100*sum(1 for g in ceg_non if g in con)/max(len(ceg_non),1):.0f}% coverage vs FBA's 0%)")
print(f"  ... strongly LoF-constrained (LOEUF<0.35): {len(strong)} "
      f"({100*len(strong)/max(len(ceg_non),1):.0f}%)")

summary=dict(source="gnomAD v2.1.1 constraint", genes_with_constraint=len(con),
    benchmarks=res,
    non_metabolic_constraint_coverage=round(sum(1 for g in ceg_non if g in con)/max(len(ceg_non),1),3),
    fba_comparison="FBA metabolic wheel: 17% coverage, 0/26 drug targets; constraint: genome-wide")
json.dump(summary,open(OUT/"human_constraint_benchmark.json","w"),indent=2)
print("\nwrote human_constraint_benchmark.json")
