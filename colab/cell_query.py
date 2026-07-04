"""ASK THE CELL — a unified query/reasoning engine over the whole model.

Routes any question to the right layer and returns an answer + CONFIDENCE + provenance, and — crucially —
an honest "not knowable" for quantitative/dynamic/kinetic questions the data can't support. This is the
breadth+reasoning axis where a broad cell model can lead (see docs/BILLION_DOLLAR_BENCHMARK.md).

Usage:  python colab/cell_query.py "what does TP53 do"
        from cell_query import answer; answer("what interacts with MDM2")
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
OUT=Path("outputs/orphan")

class Cell:
    def __init__(self):
        self.D=json.load(open(OUT/"cell_complete.json")); self.G=self.D["genes"]
        self.idx={g["name"]:i for i,g in enumerate(self.G)}
        self.upper={g["name"].upper():g["name"] for g in self.G}
        self.OUTr=defaultdict(list); self.INr=defaultdict(list); self.sgn={}
        for e in self.D.get("reg",[]):
            self.OUTr[e[0]].append(e[1]); self.INr[e[1]].append(e[0]); self.sgn[(e[0],e[1])]=e[2] if len(e)>2 else 0
        self.ppi=defaultdict(list)
        for a,b in self.D.get("ppi",[]): self.ppi[a].append(b); self.ppi[b].append(a)
        self.codep={int(k):v for k,v in self.D.get("codep",{}).items()}
        self.darkfn={int(k):v for k,v in self.D.get("darkfn",{}).items()}
        self.ppm={int(k):v for k,v in self.D.get("ppm",{}).items()}
        self.go=self.D.get("go",{}); self.drugs=self.D.get("drugs",{})
        self.g2c=self.D.get("gene2cplx",{})
        self.conv=self._load("convergence.json"); self.cond=self._load("conditions.json")
        self.reason=self._load("reasoning.json"); self.cal=self._load("calibration.json")
    def _load(self,fn):
        p=OUT/fn
        return json.load(open(p)) if p.exists() else None
    def gene(self,q):
        for tok in re.findall(r"[A-Za-z0-9\-]+", q):
            if tok.upper() in self.upper: return self.upper[tok.upper()]
        return None

def _conf(n, hi=3): return "high" if n>=hi else ("medium" if n>=1 else "low")

def answer(q, C=None):
    C=C or Cell(); ql=q.lower(); g=C.gene(q)
    # honest refusal: quantitative / dynamic / kinetic questions the data cannot support
    if re.search(r"\b(rate|kinetic|kcat|km|concentration|how fast|flux|velocity|over time|time.?course|dynamics|at t=|minutes|seconds)\b", ql):
        return dict(q=q, answer="NOT KNOWABLE from this model — rates/concentrations/dynamics need kinetics "
                    "data that is <10% measured for human (see KINETICS_ASSESSMENT.md). The model is "
                    "structural/relational, not a dynamic simulator.", confidence="n/a", source="honest-limit")
    if g is None:
        return dict(q=q, answer="no gene from the model recognized in the question.", confidence="n/a", source="parser")
    i=C.idx[g]; gg=C.G[i]
    def A(ans,conf,src,**kw): return dict(q=q, gene=g, answer=ans, confidence=conf, source=src, **kw)

    # --- route by intent ---
    if re.search(r"what (does|is)|function|role|do\b|does\b", ql) and not re.search(r"interact|bind|regulat|essential|drug|express|abund|where|locali", ql):
        if not gg["dark"]:
            ann=(C.go.get(g,{}) or {}).get("F",[])
            return A(f"{g}: {gg.get('path') or gg['proc']}"+(f"; GO molecular function: {ann[0]}" if ann else ""),
                     "high","annotation", process=gg["proc"], compartment=gg["comp"])
        d=C.darkfn.get(i)
        if d:
            acc=""
            if C.cal and "dark_gene_function" in C.cal:
                f=C.cal["dark_gene_function"]
                rate=f["high_conf_accuracy"] if d.get("conf")=="high" else f["low_conf_accuracy"]
                acc=f" [measured accuracy of {d.get('conf','low')}-confidence dark-gene predictions: ~{round(rate*100)}% exact pathway, ~{round(f['same_process']*100)}% right process]"
            return A(f"{g} is a DARK gene; predicted function: {d['pred']} (evidence: {', '.join(d.get('ev',[])[:3])}){acc}",
                     d.get("conf","low"), "prediction:"+d.get("src","?"), tag="predicted, not measured")
        return A(f"{g} is a dark gene with no confident functional prediction.","low","dark")
    if re.search(r"synthetic leth|co-?essential|co-?depend|buffer|synthetic|sl partner|sl pair", ql):
        cd=[C.G[j]["name"] for j,r in C.codep.get(i,[])[:8]]
        return A(f"{g} co-essential / candidate synthetic-lethal partners: {', '.join(cd) or '(none)'}",
                 _conf(len(cd)),"DepMap co-essentiality")
    if re.search(r"interact|bind|partner|complex", ql):
        pp=[C.G[j]["name"] for j in C.ppi.get(i,[])[:12]]
        cx=C.g2c.get(str(i),[]);
        return A(f"{g} physically interacts with: {', '.join(pp) or '(none mapped)'}"+
                 (f"; in complex(es): {', '.join(cx[:4])}" if cx else ""),
                 _conf(len(pp)),"PPI(STRING/BioPlex/OpenCell/HuRI)+complexes", n_ppi=len(C.ppi.get(i,[])))
    if re.search(r"essential|dependen|fitness", ql):
        return A(f"{g}: essential={gg['ess']} (source {gg['ess_src']}"+
                 (f", dependent in {round(gg.get('dep_frac',0)*100)}% of cancer lines" if gg.get('dep_frac') else "")+")",
                 "high" if gg["ess_src"]=="measured" else "medium","DepMap")
    if re.search(r"knock ?out|knock ?down|remove|delete|what happens if|perturb", ql):
        # SIGNED multi-hop cascade: removing X removes its output. If X activates T (+), T goes DOWN on KO;
        # if X represses T (-), T goes UP. Propagate the composed sign 2 hops. (effect on X itself = loss)
        eff={}  # gene -> net sign of change (-1 down, +1 up)
        for t in C.OUTr.get(i,[]):
            s=C.sgn.get((i,t),0)
            if s==0: continue
            eff[t]=eff.get(t,0) - s                         # KO removes X's activation -> -s
        first=dict(eff)
        for t,st in first.items():
            for u in C.OUTr.get(t,[])[:15]:
                s2=C.sgn.get((t,u),0)
                if s2 and u!=i: eff[u]=eff.get(u,0)+ (1 if st*s2>0 else -1)
        up=[C.G[t]["name"] for t,s in sorted(eff.items(),key=lambda x:-x[1]) if s>0][:12]
        dn=[C.G[t]["name"] for t,s in sorted(eff.items(),key=lambda x:x[1]) if s<0][:12]
        binders=[C.G[t]["name"] for t in C.ppi.get(i,[])[:8]]
        return A(f"knocking out {g}: predicted DOWN {', '.join(dn) or '-'}; predicted UP {', '.join(up) or '-'}; "
                 f"complexes/binders disrupted: {', '.join(binders) or '-'}",
                 _conf(len(eff),hi=5),"signed 2-hop cascade over the regulatory network",
                 n_affected=len(eff), caveat="directional prediction, not quantitative; unvalidated vs measured KO")
    if re.search(r"regulat|transcription factor|\btf\b|target of|upstream|downstream", ql):
        tg=[C.G[t]["name"] for t in C.OUTr.get(i,[])[:12]]; up=[C.G[t]["name"] for t in C.INr.get(i,[])[:8]]
        return A(f"{g} regulates: {', '.join(tg) or '-'}; regulated by: {', '.join(up) or '-'}",
                 _conf(len(tg)+len(up)),"CollecTRI/DoRothEA/TRRUST/ReMap/GTEx")
    if re.search(r"novel|new (link|interaction|function)|undiscover|convergen", ql):
        if C.conv:
            hits=[s for s in C.conv.get("novel",[]) if g in (s["a"],s["b"])][:8]
            parts=[(s["b"] if s["a"]==g else s["a"])+"["+"+".join(s["lenses"])+"]" for s in hits]
            return A(f"novel convergent links for {g}: "+(", ".join(parts) or "(none in top set)"),
                     "medium" if hits else "low","convergence engine (multi-lens agreement)")
        return A("convergence.json not present (run compute_convergence).","n/a","missing")
    if re.search(r"how much|abundance|expression level|copies|ppm|quantit", ql):
        v=C.ppm.get(i)
        return (A(f"{g} protein abundance: {v} ppm (PaxDb integrated)","high","PaxDb") if v is not None
                else A(f"no measured abundance for {g} (PaxDb layer absent or unmapped).","low","PaxDb"))
    if re.search(r"drug|inhibitor|therap|druggable", ql):
        dr=C.drugs.get(str(i),[]) or C.drugs.get(i,[])
        names=[(x.get("d") if isinstance(x,dict) else str(x)) for x in dr]
        names=[n for n in names if n]
        return A(f"{g} drugs/ligands (DGIdb): {', '.join(names[:6]) or '(no known drug)'}",
                 _conf(len(names)),"DGIdb")
    if re.search(r"where|locali|compartment|located", ql):
        return A(f"{g} localizes to: {gg['comp']} (process: {gg['proc']})","high","UniProt/HPA localization")
    if re.search(r"condition|heat|temperature|\bph\b|acid|pressure|hypoxia|stress|inflam|dna damage|damage|oxidat|osmotic|starv|xenobiotic|\bunder\b", ql) and C.cond:
        hit=[(c,v) for c,v in C.cond.items() if g in (v.get("up",[])+v.get("down",[]))]
        return A(f"{g} responds to condition(s): "+(", ".join(f"{c} ({'up' if g in C.cond[c].get('up',[]) else 'down'})" for c,_ in hit[:5]) or "(not in a curated condition response set)"),
                 _conf(len(hit)),"condition ontology (sensor regulons)")
    # default: a compact profile
    return A(f"{g}: {gg.get('path') or gg['proc']} | {gg['comp']} | essential={gg['ess']} | "
             f"{len(C.ppi.get(i,[]))} interactors | ask about function/interactions/knockout/novel/condition/abundance/drugs.",
             "high","profile")

if __name__=="__main__":
    q=" ".join(sys.argv[1:]) or "what does TP53 do"
    import pprint; pprint.pprint(answer(q))
