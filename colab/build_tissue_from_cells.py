"""TISSUE T1 — instantiate the single-cell model per cell type, then wire instances into a tissue.

The user's vision: don't wire raw HPA expression — wire N instances of the full cell model. Each cell
type gets a CELL INSTANCE = the cell model (`cell_complete.json`) masked to the genes that cell type
actually expresses, so its active regulatory/PPI subnetwork, active TFs, and active reactions are the
real modelled network (not just an expression vector). Then cell-cell communication propagates signals
through each receiver's *active* network.

Reads cell_complete.json (networks) + per-cell-type expression (celltype_expression.csv, Model 2; or
the emask/celltypes already baked into cell_complete). -> cell_instances.json.
"""
import json, csv
from collections import defaultdict
from pathlib import Path
OUT=Path("outputs/orphan")
THR=float(__import__("os").environ.get("TISSUE_EXPR_THR","1.0"))

def load_celltype_expression(idx):
    """Return {celltype: set(active gene idx)} from celltype_expression.csv (genes x celltypes) or
    from cell_complete's own emask/celltypes."""
    active=defaultdict(set)
    f=OUT/"celltype_expression.csv"
    if f.exists():
        rd=csv.reader(open(f)); hdr=next(rd)
        cts=hdr[1:]
        for row in rd:
            g=row[0]; i=idx.get(g)
            if i is None: continue
            for c,val in zip(cts,row[1:]):
                try:
                    if float(val)>=THR: active[c].add(i)
                except: pass
    return active

def main():
    cc=OUT/"cell_complete.json"
    if not cc.exists():
        print("cell_complete.json absent -> skipping tissue-from-cells"); return
    D=json.load(open(cc)); G=D["genes"]; idx={g["name"]:i for i,g in enumerate(G)}
    active=load_celltype_expression(idx)
    # fall back to cell_complete's baked-in per-cell-type expression (emask/celltypes) if no csv
    if not active and D.get("emask") and D.get("celltypes"):
        cts=D["celltypes"]; em=D["emask"]
        # emask: {celltype: [gene idx expressed]} or {gene: [celltype idx]} — handle both
        if isinstance(em,dict):
            for k,v in em.items():
                if k in (cts if isinstance(cts,list) else []): active[k]=set(v)
                else:
                    try: gi=int(k)
                    except: continue
                    for c in v: active[(cts[c] if isinstance(cts,list) and c<len(cts) else str(c))].add(gi)
    if not active:
        print("no per-cell-type expression available -> skipping (needs Model 2 celltype_expression.csv)"); return
    # networks
    reg=D.get("reg",[]); ppi=D.get("ppi",[])
    tf=set(i for i,g in enumerate(G) if g.get("tf"))
    instances={}
    for ct,act in active.items():
        act_reg=[[a,b,s] for a,b,s in ((e[0],e[1],e[2] if len(e)>2 else 0) for e in reg) if a in act and b in act]
        act_ppi=sum(1 for a,b in ppi if a in act and b in act)
        act_tf=[G[i]["name"] for i in (act & tf)]
        # master TFs of this instance = active TFs by out-degree in the active reg subnetwork
        outdeg=defaultdict(int)
        for a,b,s in act_reg: outdeg[a]+=1
        masters=[G[i]["name"] for i in sorted((act&tf),key=lambda i:-outdeg[i])[:15]]
        instances[ct]=dict(n_active=len(act), n_reg_edges=len(act_reg), n_ppi_edges=act_ppi,
                           n_tf=len(act_tf), masters=masters)
    json.dump({ct:v for ct,v in instances.items()}, open(OUT/"cell_instances.json","w"))
    print(f"tissue T1: {len(instances)} cell-type instances of the cell model")
    for ct,v in list(instances.items())[:8]:
        print(f"  {ct[:28]:28} active={v['n_active']:5} reg={v['n_reg_edges']:6} ppi={v['n_ppi_edges']:6} masters={v['masters'][:4]}")

if __name__=="__main__":
    main()
