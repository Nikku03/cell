"""PROSPECTIVE VALIDATION (DENSE) — does 'consult every dataset + integrate' beat the single-lens model?

The base test (validate_prospective.py) predicted a dark gene's function from ONE thin slice of neighbors
and beat chance 2.18x. Here we let the model GIVE ITS ALL: for each dark gene we build a DENSE evidence
network across every layer (PPI + regulation + co-essentiality + signaling + co-expression + Perturb-seq,
whichever are present) and weight each candidate partner by HOW MANY INDEPENDENT LENSES connect it. The
hypothesis: partners on which many lenses AGREE are far better functional bets.

Test (same future-truth as before): for the dense evidence set at each support level (>=1, >=2, >=3 lenses),
does the formerly-dark gene share a current GO BP term with it, vs random same-size sets -> enrichment.
If enrichment RISES with lens-agreement, dense integration measurably improves the hypotheses.
-> prospective_validation_dense.json
"""
import json, os, csv, re, random, math, urllib.request
from collections import defaultdict
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
CAP=20                                                       # cap neighbors per lens per gene (tame hubs)

def fetch_go_bp():
    f=H/"uniprot_go_bp.tsv"
    if not (f.exists() and f.stat().st_size>100000):
        url=("https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true"
             "&fields=gene_primary,go_p&format=tsv")
        print("  fetching current GO biological-process annotations (UniProt) ..."); urllib.request.urlretrieve(url,f)
    go={}
    for r in csv.DictReader(open(f), delimiter="\t"):
        g=(r.get("Gene Names (primary)") or "").strip()
        if not g: continue
        ids=set(re.findall(r"GO:\d+", r.get("Gene Ontology (biological process)") or ""))
        if ids: go[g]=ids
    return go

def build_lenses(D, G):
    """name -> {lens: set(neighbor names)} across every available evidence layer."""
    nm=[g["name"] for g in G]; N=len(nm)
    def nn(i): return nm[i] if isinstance(i,int) and 0<=i<N else None
    lens=defaultdict(lambda: defaultdict(set))
    # PPI (physical)
    deg=defaultdict(int)
    for e in D.get("ppi",[]):
        if len(e)<2: continue
        a,b=nn(e[0]),nn(e[1])
        if a and b and deg[a]<CAP and deg[b]<CAP:
            lens[a]["ppi"].add(b); lens[b]["ppi"].add(a); deg[a]+=1; deg[b]+=1
    # regulation (both directions)
    for e in D.get("reg",[]):
        if len(e)<2: continue
        a,b=nn(e[0]),nn(e[1])
        if a and b: lens[a]["reg"].add(b); lens[b]["reg"].add(a)
    # co-essentiality (top partners)
    for k,v in (D.get("codep") or {}).items():
        a=nn(int(k))
        if not a: continue
        for pair in v[:CAP]:
            b=nn(pair[0] if isinstance(pair,(list,tuple)) else pair)
            if b: lens[a]["codep"].add(b)
    # signaling (if present as index pairs)
    for e in D.get("sig",[]):
        if isinstance(e,(list,tuple)) and len(e)>=2 and isinstance(e[0],int):
            a,b=nn(e[0]),nn(e[1])
            if a and b: lens[a]["sig"].add(b); lens[b]["sig"].add(a)
    # co-expression + Perturb-seq predictor (present in a full build; name- or index-keyed)
    for key,tag in [("coexpr","coexpr"),("model4","perturbseq")]:
        v=D.get(key) or {}
        if isinstance(v,dict):
            for k,val in v.items():
                a=k if k in nm else nn(int(k)) if str(k).isdigit() else None
                if not a: continue
                nbrs=val.get("neighbors",val) if isinstance(val,dict) else val
                for x in (nbrs or [])[:CAP]:
                    b=x if x in nm else nn(x) if isinstance(x,int) else (nn(x[0]) if isinstance(x,(list,tuple)) else None)
                    if b: lens[a][tag].add(b)
    return lens

def main():
    cc=OUT/"cell_complete.json"
    if not cc.exists(): print("cell_complete.json absent -> skipped"); return
    D=json.load(open(cc)); G=D["genes"]; darkfn={int(k):v for k,v in D.get("darkfn",{}).items()}
    if not darkfn: print("no darkfn -> skipped"); return
    go=fetch_go_bp()
    if not go: print("GO unavailable -> skipped"); return
    pool=[g for g in go]; lens=build_lenses(D,G)
    active=sorted({L for v in lens.values() for L in v}); print(f"  active evidence lenses: {active}")
    def overlap(genes,xgo):
        u=set()
        for e in genes: u|=go.get(e,set())
        return len(u & xgo)>0
    rnd=random.Random(0); R=20
    # per support level: hit rate (model) and random baseline
    levels=[1,2,3]; res={}
    counts={}
    for L in levels:
        mh=[]; rh=[]
        for i,v in darkfn.items():
            if "GO(annotation)" in v.get("src",""): continue
            if i>=len(G): continue
            name=G[i]["name"]; xgo=go.get(name)
            if not xgo: continue
            w=defaultdict(int)
            for Lname,nbrs in lens.get(name,{}).items():
                for b in nbrs:
                    if b!=name and b in go: w[b]+=1
            ev=[b for b,c in sorted(w.items(),key=lambda z:-z[1]) if c>=L][:12]
            if len(ev)<2: continue
            mh.append(overlap(ev,xgo))
            for _ in range(R): rh.append(overlap(rnd.sample(pool,len(ev)),xgo))
        n=len(mh); mr=sum(mh)/n if n else 0; rr=(sum(rh)/len(rh)) if rh else 0
        se=math.sqrt(rr*(1-rr)/n) if n and 0<rr<1 else 0
        res[f">={L}_lenses"]=dict(n=n, model_hit_rate=round(mr,3), random_rate=round(rr,3),
                                  enrichment=round(mr/rr,2) if rr>0 else None,
                                  z=round((mr-rr)/se,1) if se>0 else None)
    payload=dict(single_lens_baseline_enrichment=2.18, by_lens_support=res, active_lenses=active,
                 note="dense multi-lens evidence network; evidence = partners supported by >=L independent "
                      "lenses. Enrichment rising with L => consulting+integrating more datasets improves the "
                      "hypotheses. Random same-size gene-set null.")
    json.dump(payload, open(OUT/"prospective_validation_dense.json","w"))
    print("PROSPECTIVE VALIDATION (DENSE multi-lens) — dark-gene function vs current GO")
    print(f"  single-lens baseline: 2.18x enrichment")
    for L in levels:
        r=res[f">={L}_lenses"]
        print(f"  >= {L} lens{'es' if L>1 else ' '} agree: {r['model_hit_rate']} vs {r['random_rate']} "
              f"-> {r['enrichment']}x  (n={r['n']}, z={r['z']})")
    e1=res[">=1_lenses"]["enrichment"]; e3=res.get(">=3_lenses",{}).get("enrichment") or res[">=2_lenses"]["enrichment"]
    print(f"  >>> {'DENSE INTEGRATION HELPS — enrichment rises with lens-agreement (%.2fx -> %.2fx)'%(e1,e3) if (e3 and e1 and e3>e1) else 'dense integration does not clearly raise enrichment here'}")

if __name__=="__main__":
    main()
