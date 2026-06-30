"""TF network MOTIF layer: detect motifs in the EDGE graph (not from sequence).

Motifs are topological patterns in the TF->gene network. We HAVE that graph
(RegulonDB measured; co-expression+co-fitness inferred elsewhere). Detect the
canonical motifs and test over-representation vs a degree-preserving random null
(Milo 2002 method). Validate against the known result: negative autoregulation,
feed-forward loops, and bi-fans are strongly over-represented in E. coli.

Output: counts + Z-scores per motif; the regulatory backbone to feed the
Gillespie dynamics engine.
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan")
rng=np.random.default_rng(0)
SIGN={"activator":1,"repressor":-1,"+":1,"-":-1,"dual":0,"+-":0,"-+":0,"unknown":0}
edges=[]
for l in open(REG/"network_tf_gene.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.rstrip("\n").split("\t")
    tf,tg=p[0].lower(),p[1].lower(); s=SIGN.get(p[2].lower(),0) if len(p)>2 else 0
    edges.append((tf,tg,s))
regs=set(e[0] for e in edges)                      # TF nodes
nodes=set(e[0] for e in edges)|set(e[1] for e in edges)
print(f"network: {len(nodes)} genes, {len(regs)} TFs, {len(edges)} regulatory edges")

def build(edge_list):
    out=defaultdict(dict)                            # tf -> {target: sign}
    for tf,tg,s in edge_list: out[tf][tg]=s
    return out
def count_motifs(edge_list):
    out=build(edge_list); reg=set(out)
    # autoregulation
    auto_neg=sum(1 for t in out if t in out[t] and out[t][t]==-1)
    auto_pos=sum(1 for t in out if t in out[t] and out[t][t]==1)
    # feed-forward loops: X->Y (Y is TF), X->Z, Y->Z
    ffl=0; coh=0; inc=0
    for X in out:
        NX=out[X]
        for Y in NX:
            if Y not in reg or Y==X: continue
            NY=out[Y]
            common=set(NX)&set(NY); common.discard(X); common.discard(Y)
            for Z in common:
                ffl+=1
                sxy,syz,sxz=NX.get(Y,0),NY.get(Z,0),NX.get(Z,0)
                if sxy and syz and sxz:
                    if sxy*syz==sxz: coh+=1
                    else: inc+=1
    # bi-fan: X,Y -> Z,W (X!=Y TFs, Z!=W common targets, count pairs)
    bifan=0; reglist=[r for r in out]
    tgtsets={r:set(out[r]) for r in out}
    for i in range(len(reglist)):
        for j in range(i+1,len(reglist)):
            common=tgtsets[reglist[i]]&tgtsets[reglist[j]]
            c=len(common)
            if c>=2: bifan+=c*(c-1)//2
    # SIM: TF is the SOLE regulator of >=4 targets
    indeg=defaultdict(list)
    for tf,tg,s in edge_list: indeg[tg].append(tf)
    sim=0
    for r in out:
        sole=[tg for tg in out[r] if len(indeg[tg])==1]
        if len(sole)>=4: sim+=1
    return dict(auto_neg=auto_neg,auto_pos=auto_pos,ffl=ffl,ffl_coherent=coh,ffl_incoherent=inc,bifan=bifan,sim=sim)

obs=count_motifs(edges)
print("\nobserved motif counts:")
for k,v in obs.items(): print(f"  {k:<16} {v}")

# degree-preserving null via directed double-edge swap
def swap(edge_list,nswap):
    el=[list(e) for e in edge_list]; m=len(el)
    done=0; tries=0
    while done<nswap and tries<nswap*20:
        tries+=1
        a,b=rng.integers(m),rng.integers(m)
        if a==b: continue
        # swap targets, preserve signs with their edges
        if el[a][1]==el[b][1] or el[a][0]==el[b][0]: continue
        el[a][1],el[b][1]=el[b][1],el[a][1]
        done+=1
    return [tuple(e) for e in el]

NR=60
keys=list(obs); rand=defaultdict(list)
for _ in range(NR):
    rc=count_motifs(swap(edges,len(edges)*5))
    for k in keys: rand[k].append(rc[k])
print(f"\nover-representation vs degree-preserving null ({NR} randomizations):")
print(f"{'motif':<16}{'observed':>9}{'null_mean':>11}{'Zscore':>9}")
res={}
for k in keys:
    mu=float(np.mean(rand[k])); sd=float(np.std(rand[k]) or 1)
    z=(obs[k]-mu)/sd; res[k]=dict(observed=obs[k],null_mean=round(mu,1),z=round(z,1))
    print(f"  {k:<16}{obs[k]:>9}{mu:>11.1f}{z:>9.1f}")
json.dump(dict(n_genes=len(nodes),n_tf=len(regs),n_edges=len(edges),motifs=res),
          open(OUT/"motif_detect.json","w"),indent=2)
print(f"\nwrote {OUT}/motif_detect.json")
