"""PHASE 2 — convergence engine: novel functional links where INDEPENDENT lenses agree.

Principle (DISCOVERY_ENGINES.md): a gene pair supported by multiple *independent* assays
(physical PPI, genetic co-essentiality, transcriptional Perturb-seq, co-expression) but NOT already
connected by annotation (same complex / same pathway) is a novel, self-validating functional link —
the KIDINS220-XPR1 pattern, at scale.

Reads whatever lenses are present (cell_complete.json for PPI/co-essentiality/complex/pathway/process;
optional coexpr_neighbors.json + perturbseq_neighbors.json). Scores every candidate pair by how many
independent lenses support it, filters to the novel ones (convergent but unannotated), ranks them,
and writes convergence.json. Positive control: known same-complex pairs must score high.
"""
import json
from collections import defaultdict
from pathlib import Path
OUT=Path("outputs/orphan")
LENS_W={"ppi":1.0,"codep":1.0,"perturbseq":0.9,"coexpr":0.6}   # reliability weights

def load():
    D=json.load(open(OUT/"cell_complete.json"))
    G=D["genes"]; N=len(G); idx={g["name"]:i for i,g in enumerate(G)}
    edges=defaultdict(set)   # lens -> set of frozenset{i,j}
    def add(lens,i,j):
        if i!=j and i is not None and j is not None: edges[lens].add((i,j) if i<j else (j,i))
    for a,b in D.get("ppi",[]): add("ppi",a,b)
    for k,v in D.get("codep",{}).items():
        i=int(k)
        for j,r in v: add("codep",i,j)
    # optional measured lenses from their own files (names -> idx)
    for fn,lens in [("coexpr_neighbors.json","coexpr"),("perturbseq_neighbors.json","perturbseq")]:
        p=OUT/fn
        if p.exists():
            raw=json.load(open(p))
            for g,parts in raw.items():
                i=idx.get(g)
                if i is None: continue
                for entry in parts:
                    nb=entry[0] if isinstance(entry,list) else entry
                    add(lens, i, idx.get(nb))
    # annotation lenses (to EXCLUDE — these are already-known links)
    same_complex=set()
    g2c=D.get("gene2cplx",{})
    cpl2genes=defaultdict(list)
    for k,v in g2c.items():
        for c in (v if isinstance(v,list) else [v]): cpl2genes[c].append(int(k))
    for c,mem in cpl2genes.items():
        for a in range(len(mem)):
            for b in range(a+1,len(mem)):
                i,j=mem[a],mem[b]; same_complex.add((i,j) if i<j else (j,i))
    path=[ (g.get("path") or "").strip() for g in G ]
    proc=[ g.get("proc") or "other" for g in G ]
    return D,G,N,idx,edges,same_complex,path,proc

def main():
    D,G,N,idx,edges,same_complex,path,proc=load()
    lenses=[l for l in LENS_W if edges.get(l)]
    print("convergence lenses available:",{l:len(edges[l]) for l in lenses})
    # candidate pairs = union of all measured-lens edges
    allpairs=set()
    for l in lenses: allpairs|=edges[l]
    # score each pair by independent-lens support
    scored=[]
    for (i,j) in allpairs:
        support=[l for l in lenses if (i,j) in edges[l]]
        if len(support)<2: continue                      # need >=2 INDEPENDENT lenses to be convergent
        samep = path[i] and path[j] and path[i]==path[j]
        samec = (i,j) in same_complex
        score=round(sum(LENS_W[l] for l in support),2)
        scored.append(dict(a=G[i]["name"],b=G[j]["name"],ai=i,bi=j,
                           lenses=support,n_lens=len(support),score=score,
                           same_complex=bool(samec),same_pathway=bool(samep),
                           cross_process=proc[i]!=proc[j],
                           pa=path[i] or proc[i], pb=path[j] or proc[j]))
    scored.sort(key=lambda x:(-x["n_lens"],-x["score"]))
    # NOVEL = convergent but NOT already annotated together (not same complex, not same pathway)
    novel=[s for s in scored if not s["same_complex"] and not s["same_pathway"]]
    known=[s for s in scored if s["same_complex"]]        # positive control
    payload=dict(lenses=lenses, n_convergent=len(scored),
                 n_novel=len(novel), n_known_control=len(known),
                 novel=novel[:500], known_control=known[:50])
    # optional: PubMed co-occurrence check to rank the top novel pairs by literature novelty (NCBI eutils)
    import os
    if os.environ.get("CONV_PUBMED","0")=="1":
        import urllib.request, urllib.parse, time
        def npapers(term):
            q=urllib.parse.urlencode({"db":"pubmed","term":term,"rettype":"count","retmode":"json"})
            try: return int(json.load(urllib.request.urlopen(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{q}",timeout=30))["esearchresult"]["count"])
            except: return -1
        K=int(os.environ.get("CONV_PUBMED_TOPN","40"))
        for s in novel[:K]:
            co=npapers(f'"{s["a"]}"[tiab] AND "{s["b"]}"[tiab]'); time.sleep(0.34)
            s["pubmed_cooccur"]=co
        novel[:K]=sorted(novel[:K], key=lambda s:(s.get("pubmed_cooccur",0), -s["score"]))  # fewest papers first
        payload["novel"]=novel[:500]
        print("PubMed co-occurrence checked on top",K,"novel pairs")
    json.dump(payload,open(OUT/"convergence.json","w"))
    print(f"convergent pairs (>=2 lenses): {len(scored)} | novel (unannotated): {len(novel)} | known-complex control: {len(known)}")
    print("top novel convergent links:")
    for s in novel[:15]:
        cop=f" | PubMed co-occ={s['pubmed_cooccur']}" if "pubmed_cooccur" in s else ""
        print(f"  {s['a']:10} x {s['b']:10}  lenses={'+'.join(s['lenses'])}  [{s['pa'][:20]} | {s['pb'][:20]}]{cop}")

if __name__=="__main__":
    main()
