"""PROPAGATION-THERAPY circuit finder — find REAL self-amplifying secreted circuits in the model.

The simulation (simulate_propagation.py) showed a propagating cure needs a signal that, in a neighbour,
RE-INDUCES its own secretion (a closed loop L -> R -> ... -> L) so the wave sustains. Here we search the
model's actual layers for such circuits:

  ligand L (secreted / surfaceome)  --lr-->  receptor R  --signaling/reg-->  TF  --reg-->  L   (loop closed)

A closed loop = self-propagation (a treated cell's secreted L hits R on a neighbour, whose response makes
MORE L -> the neighbour now broadcasts too). We also flag circuits whose downstream hits an EFFECT program
(apoptosis / the disease driver) so the propagating signal is therapeutic, not just self-sustaining. Ranks
candidate circuits by loop closure + receptor breadth + effect match. Needs the ligand-receptor + signaling
layers (full build). -> propagation_circuits.json
"""
import json
from collections import defaultdict
from pathlib import Path
OUT=Path("outputs/orphan")
APOPTOSIS={"CASP3","CASP8","CASP9","BAX","BAK1","BBC3","PMAIP1","FAS","FASLG","TNFSF10","TP53","BID","APAF1"}

def main():
    cc=OUT/"cell_complete.json"
    if not cc.exists(): print("cell_complete.json absent -> skipped"); return
    D=json.load(open(cc)); G=D["genes"]; nm=[g["name"] for g in G]; ni={n:i for i,n in enumerate(nm)}
    lr=D.get("lr") or []                                     # ligand-receptor pairs (indices or names)
    if not lr:
        print("propagation-therapy: ligand-receptor layer empty -> run the full build (OmniPath ligrec / CellPhoneDB). "
              "Logic ready; circuits will be found once lr is populated."); return
    # ligand -> receptors
    lig2rec=defaultdict(set)
    for e in lr:
        a=e[0]; b=e[1]
        a=nm[a] if isinstance(a,int) and a<len(nm) else a; b=nm[b] if isinstance(b,int) and b<len(nm) else b
        if a and b: lig2rec[a].add(b)
    # regulator -> targets (signed) ; and target -> regulators
    regulon=defaultdict(list)
    for e in D.get("reg",[]):
        if len(e)>=2 and isinstance(e[0],int): regulon[nm[e[0]]].append(nm[e[1]])
    # signaling adjacency (receptor -> downstream): use reg + sig as a directed proxy
    down=defaultdict(set)
    for r,ts in regulon.items():
        for t in ts: down[r].add(t)
    for e in D.get("sig",[]):
        if isinstance(e,(list,tuple)) and len(e)>=2 and isinstance(e[0],int): down[nm[e[0]]].add(nm[e[1]])
    # secretome (space.json) if present, else any ligand in lr is 'secretable'
    secretome=set((json.load(open(OUT/"space.json")).get("secretome",[])) if (OUT/"space.json").exists() else [])
    circuits=[]
    for L,recs in lig2rec.items():
        if secretome and L not in secretome: continue        # L must be secretable
        for R in recs:
            # downstream of R within 2 hops -> does it reach a regulator of L (loop) or an effect gene?
            d1=down.get(R,set()); d2=set().union(*[down.get(x,set()) for x in list(d1)[:40]]) if d1 else set()
            reach=d1|d2
            makers_of_L=set(r for r in regulon if L in regulon[r])   # TFs that drive L
            loop=reach & makers_of_L                           # closes L->R->...->TF->L
            effect=reach & APOPTOSIS
            if loop or effect:
                circuits.append(dict(ligand=L, receptor=R, loop_via=sorted(loop)[:3],
                                     effect_genes=sorted(effect)[:4], self_propagating=bool(loop),
                                     therapeutic=bool(effect),
                                     score=2*len(loop)+len(effect)+0.1*len(recs)))
    circuits.sort(key=lambda c:-c["score"])
    n_loop=sum(1 for c in circuits if c["self_propagating"])
    payload=dict(n_circuits=len(circuits), n_self_propagating=n_loop,
                 n_therapeutic=sum(1 for c in circuits if c["therapeutic"]),
                 top_circuits=circuits[:40],
                 note="ligand L secreted by a treated cell -> receptor R on neighbour -> downstream re-induces L "
                      "(self-propagating loop) and/or an apoptosis/effect program (therapeutic). A closed loop above "
                      "the percolation threshold (simulate_propagation.py) spreads the cure from a small seed.")
    json.dump(payload, open(OUT/"propagation_circuits.json","w"))
    print(f"PROPAGATION CIRCUITS: {len(circuits)} candidate ligand->receptor circuits "
          f"({n_loop} self-propagating loops, {payload['n_therapeutic']} therapeutic)")
    for c in circuits[:10]:
        tag=("LOOP" if c["self_propagating"] else "")+("+APOPTOSIS" if c["therapeutic"] else "")
        print(f"  {c['ligand']} -> {c['receptor']} [{tag}] loop_via {c['loop_via']} effect {c['effect_genes']}")

if __name__=="__main__":
    main()
