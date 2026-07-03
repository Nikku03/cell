"""TRANSFORMER prep — assemble the heterogeneous typed KG edge list + node features (Tower A input).

Collapses every layer of the cell model into one typed edge list the gene-tower (GNN) trains on, plus
a per-node feature matrix (the cold-start-free node init, pending ESM2). -> kg_edges.tsv, kg_nodes.npz
"""
import json, numpy as np
from collections import defaultdict
from pathlib import Path
OUT=Path("outputs/orphan")

def main():
    D=json.load(open(OUT/"cell_complete.json")); G=D["genes"]; N=len(G); idx={g["name"]:i for i,g in enumerate(G)}
    edges=[]   # (src, dst, type, weight, sign)
    def add(a,b,t,w=1.0,s=0):
        if a is not None and b is not None and a!=b: edges.append((a,b,t,w,s))
    for e in D.get("reg",[]): add(e[0],e[1],"reg",1.0,e[2] if len(e)>2 else 0)
    for a,b in D.get("ppi",[]): add(a,b,"ppi")
    for k,v in D.get("codep",{}).items():
        i=int(k)
        for j,r in v: add(i,j,"codep",float(r))
    g2c=defaultdict(list)
    for k,v in D.get("gene2cplx",{}).items():
        for c in (v if isinstance(v,list) else [v]): g2c[c].append(int(k))
    for c,mem in g2c.items():
        for x in range(len(mem)):
            for y in range(x+1,len(mem)): add(mem[x],mem[y],"complex")
    # optional analysis-derived edge types (present after their phases run)
    for fn,t in [("coexpr_neighbors.json","coexpr"),("convergence.json","convergence")]:
        p=OUT/fn
        if not p.exists(): continue
        raw=json.load(open(p))
        if fn.startswith("coexpr"):
            for g,parts in raw.items():
                i=idx.get(g)
                for entry in parts:
                    add(i, idx.get(entry[0]), "coexpr", float(entry[1]))
        else:
            for s in raw.get("novel",[])+raw.get("known_control",[]):
                add(idx.get(s["a"]), idx.get(s["b"]), "convergence", float(s.get("score",1)))
    # node features (cold-start init; ESM2 to be concatenated later)
    comps=sorted(set(g["comp"] for g in G)); procs=sorted(set(g["proc"] for g in G))
    ci={c:k for k,c in enumerate(comps)}; pi={p:k for k,p in enumerate(procs)}
    rout=np.zeros(N); rin=np.zeros(N)
    for a,b,t,w,s in edges:
        if t=="reg": rout[a]+=1; rin[b]+=1
    med=np.median([g["loeuf"] for g in G if g["loeuf"]>=0] or [1.0])
    F=[]
    for i,g in enumerate(G):
        num=[g["loeuf"] if g["loeuf"]>=0 else med, g["tf"], np.log1p(g["ppi"]), g["ndis"], g["npath"],
             g.get("dep_frac",0) or 0, np.log1p(g["pubs"]), 1.0 if g["ess"]==1 else 0.0,
             np.log1p(rout[i]), np.log1p(rin[i]), 1.0 if g["dark"] else 0.0]
        oh=[0.0]*(len(comps)+len(procs)); oh[ci[g["comp"]]]=1; oh[len(comps)+pi[g["proc"]]]=1
        F.append(num+oh)
    F=np.array(F,dtype=np.float32)
    with open(OUT/"kg_edges.tsv","w") as o:
        for a,b,t,w,s in edges: o.write(f"{a}\t{b}\t{t}\t{w:.3f}\t{s}\n")
    np.savez(OUT/"kg_nodes.npz", features=F, names=np.array([g["name"] for g in G]))
    from collections import Counter
    print(f"KG: {N} nodes, {len(edges)} typed edges -> kg_edges.tsv | node features {F.shape} -> kg_nodes.npz")
    print("edge types:", dict(Counter(t for _,_,t,_,_ in edges)))

if __name__=="__main__":
    main()
