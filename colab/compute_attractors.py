"""ATTRACTOR STATES (Problem 1, kinetics-free dynamics).

Cells sit in deep basins of the regulatory network -- quiescent, proliferating, apoptotic, ... -- and
'dynamics' is largely hopping between these stable states. We compute them WITHOUT any rate constants:
treat the signed TF->TF regulatory network as a threshold-Boolean dynamical system,
    tf_j(t+1) = sign( SUM_i W_ij * tf_i(t) ),   W_ij = +1 activation / -1 repression,
iterate biologically-seeded states (condition regulons) to their fixed points = ATTRACTORS, then find
which single-TF perturbations flip one attractor's basin into another = the TRANSITION map + its drivers.
This answers 'route from state A to state B' as energetic/basin routing, not a step-by-step kinetic path.

Reads cell_complete.json (signed reg edges) + conditions.json (seeds). -> attractors.json
{attractors:[{id,label,active_tfs,size,seeds}], transitions:[{from,to,driver,...}]}
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan")
MAXIT=60

def iterate(state, W_in, order, clamp=None):
    """threshold-Boolean update to a fixed point. clamp TFs (the condition's driver sensors) are held
    fixed so distinct conditions settle into distinct basins instead of collapsing to one global state."""
    s=dict(state); clamp=clamp or {}
    for _ in range(MAXIT):
        nxt=dict(s)
        for j in order:
            if j in clamp: nxt[j]=clamp[j]; continue
            net=sum(sgn*s.get(i,0) for i,sgn in W_in.get(j,()))
            if net>0: nxt[j]=1
            elif net<0: nxt[j]=-1
        if nxt==s: break
        s=nxt
    return s

def active(state): return frozenset(t for t,v in state.items() if v>0)

def main():
    cc=OUT/"cell_complete.json"
    if not cc.exists(): print("cell_complete.json absent -> attractors skipped"); return
    D=json.load(open(cc)); G=D["genes"]; idx={g["name"]:i for i,g in enumerate(G)}
    tfset=set(i for i,g in enumerate(G) if g.get("tf"))
    # signed TF->TF adjacency (regulatory core); W_in[j] = list of (regulator i, sign) for target TF j
    W_in=defaultdict(list); tfs=set()
    for e in D.get("reg",[]):
        a,b=e[0],e[1]; s=e[2] if len(e)>2 else 1
        if a in tfset and b in tfset:
            W_in[b].append((a, 1 if s>=0 else -1)); tfs.add(a); tfs.add(b)
    if not tfs: print("no signed TF->TF edges -> attractors skipped"); return
    order=sorted(tfs)
    name={i:G[i]["name"] for i in tfs}
    # seeds: each condition seeds its sensor TFs ON + their up-regulon TFs ON; sensors are CLAMPED so the
    # basin reflects "this condition sustained". -> distinct attractor per condition.
    seeds={}; clamps={}
    cond=json.load(open(OUT/"conditions.json")) if (OUT/"conditions.json").exists() else {}
    for cname,cv in cond.items():
        sensors=set(idx[s] for s in cv.get("sensors",[]) if idx.get(s) in tfs)
        up=set(idx[g] for g in cv.get("up",[]) if idx.get(g) in tfs)
        on=sensors|up
        if sensors:
            seeds[cname]={i:(1 if i in on else -1) for i in tfs}
            clamps[cname]={i:1 for i in sensors}          # hold the condition's drivers ON
    seeds["_baseline"]={i:-1 for i in tfs}; clamps["_baseline"]={}
    # find attractors
    attractors=[]; seed_of=defaultdict(list)
    for label,seed in seeds.items():
        fp=iterate(seed, W_in, order, clamps.get(label)); act=active(fp)
        # dedupe by Jaccard >= 0.9 with an existing attractor
        hit=None
        for k,(a_act,_) in enumerate(attractors):
            u=len(act|a_act); j=len(act&a_act)/u if u else 1.0
            if j>=0.9: hit=k; break
        if hit is None: attractors.append((act, fp)); hit=len(attractors)-1
        seed_of[hit].append(label)
    # transition map: from attractor A, activate attractor B's driver sensors -> does the network settle
    # into B? If so, that driver set is the route A->B (e.g. reach hypoxia by activating HIF1A).
    def jac(a,b): u=len(a|b); return len(a&b)/u if u else 1.0
    transitions=[]
    for kb,(b_act,b_fp) in enumerate(attractors):
        blabel=(seed_of[kb] or [None])[0]; bclamp=clamps.get(blabel,{})
        if not bclamp: continue                            # baseline / undriven -> no driver route
        drivers=[name[i] for i in bclamp]
        for ka,(a_act,a_fp) in enumerate(attractors):
            if ka==kb: continue
            landed=active(iterate(a_fp, W_in, order, bclamp))
            if jac(landed, b_act)>=0.85:
                transitions.append(dict(**{"from":ka}, to=kb, drivers=drivers, driver_flip="activate",
                                        barrier=len(drivers)))
    out_att=[]
    for k,(a_act,fp) in enumerate(attractors):
        tops=sorted((name[i] for i in a_act), key=lambda x:x)[:15]
        out_att.append(dict(id=k, label="/".join(seed_of[k][:3]) or f"attractor{k}",
                            active_tfs=tops, size=len(a_act), seeds=seed_of[k]))
    json.dump(dict(attractors=out_att, transitions=transitions,
                   summary=dict(n_attractors=len(attractors), n_transitions=len(transitions),
                                tf_core=len(tfs))), open(OUT/"attractors.json","w"))
    print(f"attractors: {len(attractors)} stable states in the {len(tfs)}-TF regulatory core | "
          f"{len(transitions)} single-TF transitions between basins")
    for a in out_att[:8]:
        print(f"  [{a['id']}] {a['label']:34} {a['size']} active TFs e.g. {', '.join(a['active_tfs'][:5])}")

if __name__=="__main__":
    main()
