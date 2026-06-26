"""Plot the real E. coli transcriptional regulatory data we have (RegulonDB):
every TF -> target gene -> effect (activate/repress). 211 TFs, 4670 edges.
Also relate TF connectivity to essentiality using our Keio truth.
"""
import csv, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=Path("outputs/orphan"); REG=Path("data/external_data/regulon")

rows=[l.rstrip("\n").split("\t") for l in open(REG/"network_tf_gene.txt") if l.strip() and not l.startswith("#")]
# rows: TF, target, effect, evidence, strength
tf_targets=defaultdict(set); tf_eff=defaultdict(Counter); effect=Counter()
for tf,tgt,eff,*_ in rows:
    tf_targets[tf].add(tgt); tf_eff[tf][eff]+=1; effect[eff]+=1
n_tf=len(tf_targets); n_tgt=len({r[1] for r in rows})
out_deg={tf:len(s) for tf,s in tf_targets.items()}
# in-degree: how many TFs regulate each gene
in_deg=Counter(r[1] for r in rows)

# symbol -> b-number, and truth, to relate connectivity<->essentiality
sym2b={}
for l in open(REG/"GeneProductSet.txt"):
    if l.strip() and not l.startswith("#"):
        p=l.rstrip("\n").split("\t")
        if len(p)>=3 and p[2].startswith("b") and p[1].strip(): sym2b[p[1].strip().lower()]=p[2]
b_to_keio={r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
truth={r["locus_tag"]:int(r["essential"]) for r in csv.DictReader(open("data/drive_import/labels/labels.csv")) if r["organism"]=="beril_Keio"}
def ess_of(sym):
    b=sym2b.get(sym.lower()); lt=b_to_keio.get(b) if b else None
    return truth.get(lt) if lt else None

fig=plt.figure(figsize=(16,11))

# (1) top 20 global regulators, stacked activator/repressor
ax=fig.add_subplot(2,2,1)
top=sorted(out_deg, key=out_deg.get, reverse=True)[:20]
act=[tf_eff[t].get("activator",0) for t in top]
rep=[tf_eff[t].get("repressor",0) for t in top]
y=np.arange(len(top))
ax.barh(y, act, color="#2c7fb8", label="activates")
ax.barh(y, rep, left=act, color="#d95f0e", label="represses")
ax.set_yticks(y); ax.set_yticklabels(top, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("# target genes"); ax.set_title("Top 20 global regulators (E. coli, RegulonDB)")
ax.legend(fontsize=8)

# (2) out-degree distribution (log-log) -- scale-free signature
ax=fig.add_subplot(2,2,2)
deg=np.array(sorted(out_deg.values(), reverse=True))
ax.loglog(np.arange(1,len(deg)+1), deg, "o", ms=3, color="#333")
ax.set_xlabel("TF rank"); ax.set_ylabel("# targets (out-degree)")
ax.set_title(f"TF out-degree (rank plot): {n_tf} TFs, heavy-tailed\nmedian={int(np.median(deg))}, max={deg.max()} (CRP)")

# (3) in-degree: how many TFs converge on one gene
ax=fig.add_subplot(2,2,3)
ind=np.array(list(in_deg.values()))
ax.hist(ind, bins=range(1,ind.max()+2), color="#41ab5d", edgecolor="white")
ax.set_xlabel("# TFs regulating a gene (in-degree)"); ax.set_ylabel("# genes")
ax.set_title(f"Regulatory convergence: {n_tgt} target genes\nmean inputs/gene={ind.mean():.2f}, max={ind.max()}")
ax.set_yscale("log")

# (4) essentiality vs TF connectivity (do hub regulators matter more?)
ax=fig.add_subplot(2,2,4)
bins=[(1,2),(3,5),(6,15),(16,50),(51,1000)]
labels=[]; rates=[]; ns=[]
for lo,hi in bins:
    tfs=[t for t in out_deg if lo<=out_deg[t]<=hi]
    es=[ess_of(t) for t in tfs]; es=[e for e in es if e is not None]
    if es:
        labels.append(f"{lo}-{hi if hi<1000 else '+'}\n(n={len(es)})")
        rates.append(np.mean(es)); ns.append(len(es))
# baseline genome essential rate
base=np.mean([v for v in truth.values()])
ax.bar(range(len(rates)), rates, color="#756bb1")
ax.axhline(base, ls="--", color="red", label=f"genome base rate {base:.2f}")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("fraction of TFs essential (Keio)")
ax.set_title("Is a regulator essential? vs its # targets")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT/"regulon_plots.png", dpi=130)
print("wrote outputs/orphan/regulon_plots.png")

# textual summary
summ=dict(n_tf=n_tf, n_targets=n_tgt, n_edges=len(rows),
          effect_split=dict(effect),
          top_regulators={t:out_deg[t] for t in top},
          median_targets=int(np.median(list(out_deg.values()))),
          mean_inputs_per_gene=round(float(ind.mean()),2), max_inputs=int(ind.max()),
          genome_base_essential=round(float(base),3),
          essential_rate_by_connectivity={lab:round(r,3) for lab,r in zip(labels,rates)})
json.dump(summ, open(OUT/"regulon_plots.json","w"), indent=2)
print(json.dumps(summ, indent=2))
