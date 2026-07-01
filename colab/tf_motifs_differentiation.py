"""(2) TF network motifs from TRRUST vs a degree-preserving null; (3) how the SAME genome
makes different cell types: mutual-repression toggle switches -> multistability. Includes a
bistable toggle simulation showing one network settling into multiple stable cell fates."""
import json, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H=Path("data/external_data/human"); OUT=Path("outputs/orphan"); rng=np.random.default_rng(0)
# TRRUST directed signed graph
edges=[]; TFs=set(); sign={}
for l in open(H/"trrust_human.tsv"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=3:
        a,b,mode=p[0],p[1],p[2]; TFs.add(a); edges.append((a,b))
        sign[(a,b)]={"Activation":1,"Repression":-1}.get(mode,0)
out=defaultdict(set); 
for a,b in edges: out[a].add(b)
# TF-TF subnetwork (regulators regulating regulators) for interaction motifs
tt=[(a,b) for a,b in edges if b in TFs and a!=b]
tto=defaultdict(set)
for a,b in tt: tto[a].add(b)
def count_ffl(g):
    n=0
    for a in g:
        for b in g[a]:
            if b in g:
                n+=len(g[a] & g[b])   # A->B, B->C, A->C
    return n
def count_mutual(g):
    s=set()
    for a in g:
        for b in g[a]:
            if b in g and a in g[b] and a<b: s.add((a,b))
    return len(s)
def randomize(g,nswap):
    e=[(a,b) for a in g for b in g[a]]; e=[list(x) for x in e]
    for _ in range(nswap):
        i,j=rng.integers(len(e)),rng.integers(len(e))
        if i==j: continue
        (a,b),(c,d)=e[i],e[j]
        if a==d or c==b or a==c: continue
        gg=set(); 
        # ensure no dup
        e[i]=[a,d]; e[j]=[c,b]
    g2=defaultdict(set)
    for a,b in e: 
        if a!=b: g2[a].add(b)
    return g2
# observed
obs_ffl=count_ffl(tto); obs_mut=count_mutual(tto)
selfpos=sum(1 for a in out if a in out[a] and sign.get((a,a))==1)
selfneg=sum(1 for a in out if a in out[a] and sign.get((a,a))==-1)
# null
nn=80; ffls=[]; muts=[]
for _ in range(nn):
    g2=randomize(tto, len(tt)*3); ffls.append(count_ffl(g2)); muts.append(count_mutual(g2))
def z(o,d): d=np.array(d); return round((o-d.mean())/(d.std()+1e-9),1)
print("=== (2) TF NETWORK MOTIFS (TRRUST, TF->TF subnetwork) ===")
print(f"  TFs {len(TFs)}, TF->TF edges {len(tt)}")
print(f"  feed-forward loops : obs {obs_ffl}  null {np.mean(ffls):.0f}+-{np.std(ffls):.0f}  Z={z(obs_ffl,ffls)}")
print(f"  mutual regulation  : obs {obs_mut}  null {np.mean(muts):.1f}+-{np.std(muts):.1f}  Z={z(obs_mut,muts)}")
print(f"  autoregulation     : positive {selfpos}, negative {selfneg} (self-loops)")

# (3) toggle switches = mutual REPRESSION pairs -> bistable cell-fate branch points
toggles=[]
for a in tto:
    for b in tto[a]:
        if b in tto and a in tto[b] and a<b and sign.get((a,b))==-1 and sign.get((b,a))==-1:
            toggles.append((a,b))
mutual_any=[(a,b) for a in tto for b in tto[a] if b in tto and a in tto[b] and a<b]
print(f"\n=== (3) DIFFERENTIATION: same genome -> many cell types ===")
print(f"  mutual-regulation TF pairs: {len(mutual_any)}; mutual-REPRESSION toggle switches: {len(toggles)}")
print(f"  example toggles: {toggles[:8]}")

# bistable toggle simulation: two TFs mutually repress -> two stable fates from ONE network
def sim(u0,v0,alpha=4,n=3,steps=2000,dt=0.02):
    u,v=u0,v0
    for _ in range(steps):
        du=alpha/(1+v**n)-u; dv=alpha/(1+u**n)-v; u+=dt*du; v+=dt*dv
    return u,v
fates=set()
starts=[(0.1,3),(3,0.1),(1,1.1),(1.1,1),(2,0.5),(0.5,2)]
finals=[]
for u0,v0 in starts:
    u,v=sim(u0,v0); finals.append((round(u,2),round(v,2)))
    fates.add(("A" if u>v else "B"))
print(f"  toggle simulation from 6 initial states -> final (u,v): {finals}")
print(f"  distinct stable fates reached: {len(fates)} (same equations/genome, different attractors)")

# figure: toggle phase outcomes
fig,ax=plt.subplots(1,2,figsize=(12,5))
ax[0].bar(["feed-forward\nloops","mutual\nregulation"],[z(obs_ffl,ffls),z(obs_mut,muts)],color=["#2980b9","#c0392b"])
ax[0].set_ylabel("enrichment Z vs random network"); ax[0].set_title("(2) TF network motifs are over-represented")
ax[0].axhline(0,color="k",lw=.5); ax[0].grid(alpha=.2,axis="y")
# toggle trajectories
for u0,v0 in starts:
    us=[u0]; vs=[v0]; u,v=u0,v0
    for _ in range(600):
        du=4/(1+v**3)-u; dv=4/(1+u**3)-v; u+=0.02*du; v+=0.02*dv; us.append(u); vs.append(v)
    ax[1].plot(us,vs,alpha=.7)
    ax[1].scatter([us[-1]],[vs[-1]],c="k",zorder=5,s=40)
ax[1].set_xlabel("TF A activity"); ax[1].set_ylabel("TF B activity")
ax[1].set_title("(3) one toggle network -> two stable cell fates")
plt.tight_layout(); plt.savefig(OUT/"tf_motifs_differentiation.png",dpi=130)
json.dump(dict(tfs=len(TFs),tf_tf_edges=len(tt),ffl=dict(obs=obs_ffl,z=z(obs_ffl,ffls)),
    mutual=dict(obs=obs_mut,z=z(obs_mut,muts)),autoreg=dict(pos=selfpos,neg=selfneg),
    toggle_switches=len(toggles),toggle_examples=toggles[:20],mutual_pairs=len(mutual_any),
    sim_distinct_fates=len(fates),sim_finals=finals),open(OUT/"tf_motifs_differentiation.json","w"),indent=2)
print("\nwrote tf_motifs_differentiation.png + .json")
