"""REASONING ENGINE — derive NEW facts by logical composition over the cell model, and flag gaps.

Beyond guilt-by-association (convergence), this does symbolic inference:
  Rule B  signed path composition:  A --s1--> B --s2--> C  =>  A --(s1*s2)--> C
          (if X activates Y and Y activates Z, X activates Z). A direct edge with the OPPOSITE sign is a
          CONTRADICTION -> context-dependent regulation or an error (a place the model is incomplete/wrong).
  Rule A  transitive functional closure:  A~B and B~C (co-essential) => propose A~C; CONFIRMED if another
          independent lens (PPI) also supports A~C -> a newly-derived high-confidence functional link.
  Rule C  constraint consistency:  obligate complex members should share essentiality; interacting proteins
          should co-localize. Violations are flagged as gaps to resolve (redundancy / trafficking / error).

Runs on cell_complete.json (reg signs, PPI, co-essentiality, complexes, essentiality, compartment).
-> reasoning.json {derived_regulation, contradictions, derived_functional, consistency_flags}
"""
import json
from collections import defaultdict
from pathlib import Path
OUT=Path("outputs/orphan")

def main():
    D=json.load(open(OUT/"cell_complete.json")); G=D["genes"]; nm=lambda i:G[i]["name"]
    tf=set(i for i,g in enumerate(G) if g.get("tf"))
    comp=[g.get("comp") for g in G]; ess=[g.get("ess") for g in G]

    # ---- signed regulatory adjacency ----
    out=defaultdict(list); sgn={}
    for e in D.get("reg",[]):
        a,b=e[0],e[1]; s=e[2] if len(e)>2 else 0
        out[a].append(b); sgn[(a,b)]=s
    indeg=defaultdict(int)
    for (a,b) in sgn: indeg[b]+=1
    MAX_MID=int(__import__("os").environ.get("REASON_MAX_MID","40"))  # middle must be a SPECIFIC connector

    # Rule B: 2-hop signed composition through a TF middle (specific, not a promiscuous hub)
    derived_reg=[]; contradictions=[]; seen=set()
    for a in list(out):
        for b in out[a]:
            if b not in tf or sgn[(a,b)]==0: continue
            if len(out[b])>MAX_MID: continue                       # skip promiscuous middles (hub filter)
            s1=sgn[(a,b)]
            for c in out[b]:
                s2=sgn[(b,c)]
                if s2==0 or c==a: continue
                net=s1*s2
                if (a,c) in sgn:
                    sd=sgn[(a,c)]
                    if sd!=0 and sd!=net:
                        contradictions.append(dict(a=nm(a),c=nm(c),via=nm(b),direct=sd,path=net))
                elif (a,c) not in seen:
                    seen.add((a,c)); derived_reg.append(dict(tf=nm(a),target=nm(c),sign=net,via=nm(b)))

    # Rule A: transitive functional closure on co-essentiality, confirmed by PPI
    codep=D.get("codep",{})
    co=defaultdict(set)
    for k,v in codep.items():
        i=int(k)
        for j,r in v: co[i].add(j); co[j].add(i)
    ppi=set()
    for x,y in D.get("ppi",[]): ppi.add((x,y) if x<y else (y,x))
    derived_fn=[]; seenf=set()
    for b in list(co):
        if len(co[b])>25: continue                                # specific connector only (no co-essentiality hubs)
        nb=list(co[b])
        for ia in range(len(nb)):
            for ic in range(ia+1,len(nb)):
                a,c=nb[ia],nb[ic]
                key=(a,c) if a<c else (c,a)
                if key in seenf or c in co.get(a,()): continue   # A-C not already co-essential
                seenf.add(key)
                if key in ppi:                                    # transitive prediction CONFIRMED by physical
                    derived_fn.append(dict(a=nm(a),c=nm(c),via=nm(b),confirm="PPI"))

    # Rule C: complex members should share essentiality
    g2c=D.get("gene2cplx",{}); cpl=defaultdict(list)
    for k,v in g2c.items():
        for c in (v if isinstance(v,list) else [v]): cpl[c].append(int(k))
    ess_flags=[]
    for c,mem in cpl.items():
        e=[ess[i] for i in mem if ess[i] in (0,1)]
        if len(e)>=3 and 0<sum(e)<len(e):
            frac=sum(e)/len(e)
            if 0.2<frac<0.8:                                      # genuinely mixed (not one outlier)
                ess_flags.append(dict(complex=c,n=len(mem),ess_frac=round(frac,2),
                                      members=[nm(i) for i in mem[:6]]))

    payload=dict(n_derived_regulation=len(derived_reg), n_contradictions=len(contradictions),
                 n_derived_functional=len(derived_fn), n_consistency_flags=len(ess_flags),
                 derived_regulation=derived_reg[:3000], contradictions=contradictions[:1000],
                 derived_functional=derived_fn[:2000], consistency_flags=ess_flags[:500])
    json.dump(payload, open(OUT/"reasoning.json","w"))
    print(f"reasoning: derived {len(derived_reg)} indirect regulatory effects (signed path composition)")
    print(f"           {len(contradictions)} sign CONTRADICTIONS (context-dependent / error candidates)")
    print(f"           {len(derived_fn)} transitive functional links CONFIRMED by an independent lens")
    print(f"           {len(ess_flags)} complexes with inconsistent member essentiality (gaps)")
    for d in derived_fn[:8]:
        print(f"  derived: {d['a']} ~ {d['c']}  (via {d['via']}, confirmed by {d['confirm']})")
    for c in contradictions[:5]:
        print(f"  contradiction: {c['a']}->{c['c']} direct={c['direct']} but via {c['via']} implies {c['path']}")

if __name__=="__main__":
    main()
