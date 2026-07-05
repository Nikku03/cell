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
        self.nichenet=self.D.get("nichenet",{})            # ligand -> downstream target genes (NicheNet)
        self.metab=(self._load("metabolites.json") or {}).get("metabolites",{})   # first-class metabolites
        _at=self._load("attractors.json") or {}
        self.attractors=_at.get("attractors",[]); self.transitions=_at.get("transitions",[])
        _sp=self._load("space.json") or {}
        self.space=_sp.get("locations",{})                                          # HPA subcellular localization
        self.surfaceome=set(_sp.get("surfaceome",[])); self.secretome=set(_sp.get("secretome",[]))
        _var=self._load("variants.json") or {}
        self.variants=_var.get("variants",{})                                       # ClinVar + constraint vulnerability
        self.validated_targets={t["gene"] for t in _var.get("genetically_validated_targets",[])}  # disease variant + drug
        self.unmet_targets={t["gene"] for t in _var.get("unmet_need_targets",[])}   # disease/essential + constrained + undrugged
        self.dyn=(self._load("dynamics.json") or {}).get("dynamics",{})             # turnover half-lives -> response times
        self.metab_lower={k.lower():k for k in self.metab}
        self._metab_keys=sorted(self.metab_lower, key=len, reverse=True)   # longest-first for matching
        self.conv=self._load("convergence.json"); self.cond=self._load("conditions.json")
        self.reason=self._load("reasoning.json"); self.cal=self._load("calibration.json")
        self.kin=(self._load("kinetics.json") or {}).get("kinetics",{})
        self.kinref=(self._load("kinetics_refined.json") or {}).get("kinetics_refined",{})  # best: measured>flux-deconv>CatPred
        self.ivk=(self._load("invivo_kcat.json") or {}).get("invivo_kcat",{})   # in-vivo apparent kcat (flux/[E])
        self.conc=(self._load("concentration.json") or {}).get("concentration",{})
        fx=self._load("flux.json") or {}; self.flux=fx.get("flux",{}); self.flux_val=fx.get("validation")
        self.flux_units=fx.get("summary",{}).get("flux_units","relative")
        self.gene2flux=defaultdict(list)                    # gene -> [(rxn_id, rec)] from ecFBA solution
        for rid,rec in self.flux.items():
            for g in rec.get("genes",[]): self.gene2flux[g].append((rid,rec))
    def _load(self,fn):
        p=OUT/fn
        return json.load(open(p)) if p.exists() else None
    def gene(self,q):
        for tok in re.findall(r"[A-Za-z0-9\-]+", q):
            if tok.upper() in self.upper: return self.upper[tok.upper()]
        return None
    def metabolite(self,q):
        """longest metabolite name (>=3 chars) appearing in the question -> its canonical name."""
        ql=q.lower()
        for k in self._metab_keys:
            if len(k)>=3 and re.search(r"(?<![a-z0-9])"+re.escape(k)+r"(?![a-z0-9])", ql):
                return self.metab_lower[k]
        return None

def _conf(n, hi=3): return "high" if n>=hi else ("medium" if n>=1 else "low")

def answer(q, C=None):
    C=C or Cell(); ql=q.lower(); g=C.gene(q)
    # --- DYNAMICS (turnover / linear-response level): measured half-lives -> response time, synthesis rate, cell-cycle phase ---
    if C.dyn and re.search(r"half.?life|turn ?over|response time|how fast.*(respond|turn|deplet|chang)|synthesis rate|"
                           r"degrad|how long.*(knock|deplet|last|take)|cell.?cycle|which phase|when.*(express|peak)|"
                           r"rapid|synthesis.?vulnerab", ql):
        gg2=C.gene(q); d=C.dyn.get(gg2) if gg2 else None
        if d:
            parts=[]
            if d.get("protein_half_life_h"): parts.append(f"protein t½ {d['protein_half_life_h']}h (response time {d['response_time_h']}h, {d.get('turnover_class','')}-turnover)")
            if d.get("mrna_half_life_h"): parts.append(f"mRNA t½ {d['mrna_half_life_h']}h")
            if d.get("synthesis_rate_nM_per_h"): parts.append(f"needs ~{d['synthesis_rate_nM_per_h']} nM/h synthesis to hold steady state")
            if d.get("cell_cycle_phase"): parts.append(f"peaks in {d['cell_cycle_phase']} phase")
            if d.get("essential") and d.get("turnover_class")=="fast": parts.append("ESSENTIAL + fast → synthesis-vulnerable (translation-block depletes it fastest)")
            return dict(q=q, gene=gg2, answer=f"{gg2}: "+"; ".join(parts or ["no turnover data"]),
                        confidence="medium", source="dynamics (measured half-lives: Mathieson/RNADecayCafe + Whitfield cell-cycle)",
                        caveat="turnover/linear-response level (response time = t½/ln2); not a full timed simulation")
        if re.search(r"synthesis.?vulnerab|fast.?turnover|rapid respond", ql):     # landscape query
            fv=[gn for gn,r in C.dyn.items() if r.get("essential") and r.get("turnover_class")=="fast"][:20]
            return dict(q=q, answer="synthesis-vulnerable (essential + fast-turnover) genes: "+", ".join(fv),
                        confidence="medium", source="dynamics x essentiality")
    # truly-not-knowable: a full TIME-RESOLVED trajectory/simulation (we have turnover response times, not trajectories)
    if re.search(r"\b(over time|time.?course|at t=|after \d+\s*(min|sec|hour)|trajectory|simulate the|per second changes)\b", ql):
        return dict(q=q, answer="NOT KNOWABLE at full resolution — a time-resolved TRAJECTORY needs a kinetic simulation "
                    "the data can't support. I CAN give per-gene response time / half-life (dynamics layer), static "
                    "concentration, velocity, and TF occupancy — just not the second-by-second trajectory.",
                    confidence="n/a", source="honest-limit")
    # --- ATTRACTOR STATES / basin transitions (kinetics-free dynamics) ---
    if C.attractors and re.search(r"attractor|stable state|cell state|basin|transition|switch|how (to|do).*(get|move|reach)|driver|reprogram", ql):
        labels={a["id"]:a["label"] for a in C.attractors}
        # transition drivers into/out of a state whose label matches a token in the query
        tgt=next((a for a in C.attractors if any(w and w in ql for w in a["label"].lower().split("/"))), None)
        if tgt and re.search(r"transition|switch|how|driver|reprogram|reach|into|to ", ql):
            drv=sorted([t for t in C.transitions if t["to"]==tgt["id"]], key=lambda t:t["barrier"])
            routes=[f"from {labels.get(t['from'],t['from'])} (barrier {t['barrier']}): activate {', '.join(t.get('activate',[])[:4])}"
                    +(f" / repress {', '.join(t.get('repress',[])[:3])}" if t.get('repress') else "") for t in drv[:5]]
            return dict(q=q, answer=f"reprogram to '{tgt['label']}': "+("; ".join(routes) or "no route found"),
                        confidence="medium", source="attractor reprogramming map (regulatory basins)")
        # otherwise list the attractor landscape
        parts=[f"[{a['id']}] {a['label']} ({a['size']} active TFs)" for a in C.attractors[:8]]
        return dict(q=q, answer=f"{len(C.attractors)} regulatory attractor states: "+"; ".join(parts)+
                    f" | {len(C.transitions)} single-TF transitions between basins",
                    confidence="medium", source="attractor landscape (Boolean regulatory network)",
                    caveat="kinetics-free basin analysis, not a timed trajectory")
    # --- METABOLITE as a first-class species (production/consumption/concentration/turnover) ---
    mm=C.metabolite(q) if C.metab else None
    if mm and re.search(r"produc|make|synthesi|form\b|consum|degrad|break|reaction|turnover|flux through|"
                        r"concentration|how much|level|\bhub\b|currency|what is|what'?s", ql):
        r=C.metab[mm]
        if re.search(r"concentration|how much|level|how many", ql):
            if "concentration_uM" in r:
                return dict(q=q, metabolite=mm, source="metabolite concentration",
                            answer=f"{mm} ≈ {r['concentration_uM']} µM intracellular ({r['conc_tier']})",
                            confidence="medium" if r["conc_tier"]=="measured" else "low",
                            caveat="order-of-magnitude central-metabolome value")
            return dict(q=q, metabolite=mm, confidence="n/a", source="metabolite",
                        answer=f"{mm}: present in the network but no mapped concentration (measured metabolomics absent).")
        prod=r.get("produced_by",[]); cons=r.get("consumed_by",[])
        extra=f" | flux turnover {r['turnover']}" if "turnover" in r else ""
        cc=f" ≈ {r['concentration_uM']} µM" if "concentration_uM" in r else ""
        return dict(q=q, metabolite=mm, source="metabolite network (Human-GEM)", confidence="high",
                    answer=f"{mm} [{', '.join(r['compartments']) or 'metabolite'}]{cc}: produced by {r['n_producers']} "
                           f"reaction(s) ({', '.join(prod[:6])}); consumed by {r['n_consumers']} ({', '.join(cons[:6])})"
                           f"{extra}"+(" [hub/currency metabolite]" if r["hub"] else ""))
    # static concentration / occupancy / velocity (the concentration layer — E. coli method ported to human)
    if re.search(r"\b(concentration|how much|abundance|copies|\bnM\b|occupancy|expression level|velocity|how many molecules|\[.*\])\b", ql):
        gg2=C.gene(q); r=C.conc.get(gg2) if gg2 else None
        if r:
            occ=f"; TF occupancy {r['occupancy']} (fraction above its Kd)" if "occupancy" in r else ""
            vel=f"; est. reaction velocity {r['velocity_rel_per_s']}/s (kcat x [E])" if "velocity_rel_per_s" in r else ""
            condc=""
            if re.search(r"hypox|heat|acid|oxidat|inflam|stress|starv|damage|under|condition", ql) and r.get("per_condition"):
                condc="; per-condition (nM): "+", ".join(f"{k}={v}" for k,v in list(r["per_condition"].items())[:4])
            return dict(q=q, gene=gg2,
                        answer=f"{gg2} ≈ {r['baseline_nM']} nM ({r['tier']}){occ}{vel}{condc}",
                        confidence="medium" if "PaxDb" in r["tier"] else "low", source="concentration layer",
                        caveat="absolute conversion is order-of-magnitude (cell-count/volume anchors)")
        return dict(q=q, answer=f"no concentration estimate for this gene (abundance layer absent or unmapped).",
                    confidence="n/a", source="concentration")
    # metabolic FLUX through an enzyme's reaction(s) — ecFBA on Human-GEM (validated vs 13C-MFA).
    # Checked before kinetics so "rate-limiting"/"bottleneck" route here, not to kcat.
    if re.search(r"\b(flux|throughput|carbon flow|rate.?limiting|bottleneck|carries|flowing through)\b", ql):
        gg2=C.gene(q); fl=C.gene2flux.get(gg2) if gg2 else None
        if fl:
            fl=sorted(fl,key=lambda x:-abs(x[1]["v"]))[:4]
            parts=[f"{rid} v={rec['v']} [{rec['tier']}]" for rid,rec in fl]
            vnote=(f" | model validated vs 13C-MFA at {C.flux_val['median_fold_error']}x median fold-error, "
                   f"within-2x {C.flux_val['within_2x']:.0%}" if C.flux_val else "")
            return dict(q=q, gene=gg2, answer=f"{gg2} carries flux through: "+"; ".join(parts)+f" ({C.flux_units})"+vnote,
                        confidence="medium", source="enzyme-constrained FBA (Human-GEM)",
                        caveat=("absolute, anchored to measured exchange fluxes" if "absolute" in C.flux_units
                                else "relative flux (add measured exchange data for absolute mmol/gDW/hr)"))
        if C.flux:
            return dict(q=q, answer=f"{gg2 or 'that gene'} carries ~no flux in the current ecFBA solution (inactive "
                        "reaction, or not enzyme-associated).", confidence="low", source="enzyme-constrained FBA")
        return dict(q=q, answer="flux layer absent (run compute_flux on Colab: needs Human-GEM + kinetics + concentration).",
                    confidence="n/a", source="flux")
    # kcat / turnover / reaction-rate of an ENZYME -> tiered ESTIMATE (measured with conditions, else imputed)
    if re.search(r"\b(rate|kcat|turnover|kinetic|km|how fast|catalyt|vmax|reaction speed)\b", ql):
        gg2=C.gene(q); k=C.kin.get(gg2) if gg2 else None
        iv=C.ivk.get(gg2) if gg2 else None
        kr=C.kinref.get(gg2) if gg2 else None
        # the refined value is the best: measured > flux-deconvolved > CatPred(calibrated)+prior, with Km/saturation
        if kr and kr.get("kcat_per_s"):
            km=f"; Km {round(kr['km_uM'],3)} µM" if kr.get("km_uM") else ""
            ki=f"; measured Ki {round(kr['measured_ki_uM'],4)} µM" if kr.get("measured_ki_uM") else ""
            sat=f"; saturation {kr['saturation']}" if kr.get("saturation") is not None else ""
            cx=f"; Davidi in-vivo max {kr['davidi_kcat_max_per_s']}/s" if kr.get("davidi_kcat_max_per_s") else ""
            return dict(q=q, gene=gg2, answer=f"{gg2} kcat ≈ {kr['kcat_per_s']} /s [{kr['tier']}]{km}{ki}{sat}{cx}",
                        confidence="high" if kr["tier"]=="measured" else "medium",
                        source="refined kinetics ("+kr["tier"]+")", tier=kr["tier"])
        # PREFER in-vivo apparent kcat (flux/[E]) when the in-vitro value is only imputed -- it is
        # data-derived (from the flux + abundance layers) rather than an EC-median/family guess.
        if iv and iv.get("kcat_app_per_s") and (not k or k.get("tier") not in ("measured","EC-measured")):
            note=f"IN-VIVO apparent (flux/[E]); {iv.get('interpretation','')}"
            comp=f"; in-vitro {iv['kcat_invitro_per_s']}/s [{iv.get('kcat_invitro_tier')}], saturation {iv.get('saturation_ratio')}" if iv.get("kcat_invitro_per_s") else ""
            return dict(q=q, gene=gg2, answer=f"{gg2} kcat ≈ {iv['kcat_app_per_s']} /s [{note}]{comp}",
                        confidence="medium", source="in-vivo kcat (flux/abundance)", tier="in-vivo")
        if k:
            cond=(f" at pH {k.get('pH')}, {k.get('temp')}C ({k.get('source')}{', in-vitro' if k.get('in_vitro') else ''})"
                  if k.get('tier')=="measured" else "")
            note={"measured":"MEASURED (fact, this protein)",
                  "EC-measured":"real measured human kcat for this enzyme's EC class (DLKcat, in-vitro)",
                  "family-prior":"ESTIMATE from enzyme family (not measured)",
                  "network-propagated":"ESTIMATE propagated from network neighbours (weak, not measured)",
                  "global-prior":"crude default (no family data)"}.get(k["tier"],k["tier"])
            vm=f"; relative in-vivo capacity Vmax(log10)={k['vmax_rel']} (kcat x PaxDb abundance)" if k.get("vmax_rel") is not None else ""
            return dict(q=q, gene=gg2, answer=f"{gg2} kcat ≈ {k['kcat_per_s']} /s [{note}]{cond}{vm}",
                        confidence="high" if k["tier"]=="measured" else ("medium" if k["tier"] in ("EC-measured","family-prior") else "low"),
                        source="kinetics ("+k["tier"]+")", tier=k["tier"])
        return dict(q=q, answer="no kinetic estimate for this gene (not an annotated enzyme, or kinetics layer "
                    "absent). Measured human kcat exists for <10% of enzymes.", confidence="n/a", source="kinetics")
    if g is None:
        return dict(q=q, answer="no gene from the model recognized in the question.", confidence="n/a", source="parser")
    i=C.idx[g]; gg=C.G[i]
    def A(ans,conf,src,**kw): return dict(q=q, gene=g, answer=ans, confidence=conf, source=src, **kw)

    # --- route by intent ---
    # NicheNet ligand->target signaling (checked early so "signal/ligand" beats the generic function/regulation routes)
    if re.search(r"signal|ligand|paracrine|secret|downstream target", ql) and C.nichenet:
        if g in C.nichenet:                                        # g is a ligand -> its downstream targets
            tg=C.nichenet[g]
            return A(f"{g} (ligand) signals downstream to: {', '.join(tg[:15])}"+(" ..." if len(tg)>15 else ""),
                     _conf(len(tg)),"NicheNet ligand->target regulatory potential")
        upstream=[lig for lig,ts in C.nichenet.items() if g in ts]  # g is a target -> which ligands move it
        if upstream:
            return A(f"{g} is a downstream target of ligand(s): {', '.join(upstream[:15])}",
                     _conf(len(upstream)),"NicheNet ligand->target regulatory potential")
        return A(f"{g} is not a NicheNet ligand and no modeled ligand is predicted to move it.","low","NicheNet")
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
    if re.search(r"where|locali|compartment|located|subcellular", ql):
        sp=C.space.get(g)
        if sp:
            multi=" [multi-localized]" if sp.get("multi") else ""
            acc=(" [SURFACEOME — plasma-membrane, drug-accessible]" if g in C.surfaceome
                 else " [SECRETOME — secreted]" if g in C.secretome else "")
            return A(f"{g} localizes to: {', '.join(sp['locations'])}{multi}{acc} (HPA {sp.get('reliability','')}; "
                     f"main {', '.join(sp.get('main',[])) or '-'})","high","HPA immunofluorescence (subcellular)")
        return A(f"{g} localizes to: {gg['comp']} (process: {gg['proc']})","medium","model compartment tag")
    if re.search(r"variant|mutation|pathogenic|clinvar|vulnerab|disease.?gene|missense|target|druggable", ql):
        v=C.variants.get(g)
        if v:
            cv=(f"; ClinVar: {v['clinvar_pathogenic']} pathogenic / {v['clinvar_benign']} benign / {v['clinvar_vus']} VUS "
                f"({v.get('review_stars',0)}★)" if "clinvar_pathogenic" in v else "")
            eg=f"; e.g. {', '.join(v.get('example_pathogenic',[])[:2])}" if v.get("example_pathogenic") else ""
            tgt=(" ⚑ GENETICALLY-VALIDATED druggable target (disease variants + a drug)" if g in C.validated_targets
                 else " ⚑ UNMET-NEED target (disease/essential + LoF-constrained + undrugged)" if g in C.unmet_targets else "")
            dr=f"; drugs: {', '.join(v['drugs'][:4])}" if v.get("drugs") else ""
            return A(f"{g} variant vulnerability: {v['vulnerability_tier']} (LOEUF {v['loeuf']}, "
                     f"essential={v['essential']}){cv}{eg}{dr}{tgt}",
                     "high" if v.get("review_stars",0)>=2 else "medium", "ClinVar + gnomAD constraint + DGIdb/OpenTargets")
        return A(f"no variant profile for {g} (variants layer absent or gene unmapped).","n/a","variants")
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
