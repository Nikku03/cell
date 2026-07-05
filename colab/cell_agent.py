"""CELL AGENT — a grounded orchestrator that turns MACRO questions into cited, multi-hop answers.

The query engine (cell_query) answers ATOMIC questions. Macro questions ("here's a diseased-cell profile,
where's the target and will disabling it work?") need a CHAIN of tools + synthesis. This agent does that:

  TOOLS      : every capability wrapped as a grounded function returning {value, source, confidence}.
  RECIPES    : deterministic multi-hop chains for the macro-capabilities (target dossier, disease->target
               reverse inference, mutation->consequence, perturbation effect).
  ROUTER     : maps a question to a recipe (keyword rules; or an LLM if a key is set).
  SYNTHESIS  : assembles a cited, confidence-tagged answer. The LLM, IF present, only ROUTES and PHRASES --
               it never supplies facts; every fact comes from a tool. Fully works with no LLM.

Usage:  from cell_agent import CellAgent; a=CellAgent(); print(a.ask("is MCL1 a good target"))
        a.ask("where does this disease live", signature={"FOS":2.1,"JUN":1.8, ...})
"""
import os, json, math
from pathlib import Path
OUT=Path("outputs/orphan")

class CellAgent:
    def __init__(self):
        import sys; sys.path.insert(0, str(Path(__file__).parent))
        from cell_query import Cell
        self.C=Cell(); self.M=None                      # reverse-inference model (lazy)
        self.tools={  # name -> (description, callable)
            "function":("what the gene does", self.t_function),
            "efficacy":("does disabling it kill the disease cell (DepMap)", self.t_efficacy),
            "safety":("toxicity / therapeutic window (LoF constraint)", self.t_safety),
            "mechanism":("downstream cascade when disabled", self.t_cascade),
            "combinations":("synthetic-lethal partners to co-target", self.t_combos),
            "druggability":("known drugs / target status", self.t_drugs),
            "kinetics":("kcat/Km/Ki if enzyme", self.t_kinetics),
            "dynamics":("response time / turnover", self.t_dynamics),
            "localization":("compartment / surface accessibility", self.t_localize),
            "variants":("disease variants", self.t_variants),
            "reverse_infer":("phenotype signature -> candidate causal genes", self.t_reverse),
        }

    # ---------- grounded tools (each returns {value, source, confidence} or None) ----------
    def _g(self, name):
        i=self.C.idx.get(name); return (i, self.C.G[i]) if i is not None else (None,None)
    def t_function(self, name):
        i,g=self._g(name)
        if g is None: return None
        return dict(value=f"{g.get('path') or g.get('proc')} [{g.get('comp','?')}]", source="annotation", confidence="high")
    def t_efficacy(self, name):
        i,g=self._g(name)
        if g is None: return None
        df=g.get("dep_frac"); ess=g.get("ess")
        if df is not None:
            return dict(value=f"disabling kills {round(df*100)}% of cancer lines (DepMap {'common-essential' if ess else 'selective'})",
                        source="DepMap CRISPR (measured)", confidence="high", num=df)
        return dict(value=f"essential={ess} (source {g.get('ess_src','?')})", source="DepMap/model", confidence="medium", num=float(ess or 0))
    def t_safety(self, name):
        i,g=self._g(name)
        if g is None: return None
        lo=g.get("loeuf",-1)
        if isinstance(lo,(int,float)) and lo>=0:
            w="narrow (constrained → on-target toxicity risk)" if lo<0.35 else ("wide (LoF-tolerant → safer window)" if lo>1 else "moderate")
            return dict(value=f"LOEUF {lo} → {w}", source="gnomAD constraint", confidence="high", num=lo)
        return None
    def t_cascade(self, name):
        i,g=self._g(name)
        if i is None: return None
        C=self.C; eff={}
        for t in C.OUTr.get(i,[]):
            s=C.sgn.get((i,t),0)
            if s: eff[t]=eff.get(t,0)-s
        dn=[C.G[t]["name"] for t,s in sorted(eff.items(),key=lambda x:x[1]) if s<0][:8]
        up=[C.G[t]["name"] for t,s in sorted(eff.items(),key=lambda x:-x[1]) if s>0][:8]
        if not (dn or up): return None
        return dict(value=f"predicted DOWN {', '.join(dn) or '-'}; UP {', '.join(up) or '-'}",
                    source="signed regulatory cascade", confidence="medium")
    def t_combos(self, name):
        i,g=self._g(name)
        if i is None: return None
        cd=[self.C.G[j]["name"] for j,r in self.C.codep.get(i,[])[:6]]
        return dict(value=", ".join(cd) or "(none)", source="DepMap co-essentiality", confidence="medium") if cd else None
    def t_drugs(self, name):
        i,g=self._g(name)
        if i is None: return None
        dr=self.C.drugs.get(str(i),[]) or self.C.drugs.get(i,[])
        names=[x.get("d") if isinstance(x,dict) else str(x) for x in dr]; names=[n for n in names if n]
        return dict(value=", ".join(names[:6]) if names else "no known drug (undrugged)",
                    source="DGIdb", confidence="high", drugged=bool(names))
    def t_kinetics(self, name):
        r=self.C.kinref.get(name) if self.C.kinref else None
        if r and r.get("kcat_per_s"):
            extra=(f", Km {round(r['km_uM'],2)}µM" if r.get("km_uM") else "")+(f", measured Ki {r['measured_ki_uM']}µM" if r.get("measured_ki_uM") else "")
            return dict(value=f"kcat {r['kcat_per_s']}/s [{r['tier']}]{extra}", source="refined kinetics", confidence="high" if r["tier"]=="measured" else "medium")
        return None
    def t_dynamics(self, name):
        d=self.C.dyn.get(name) if self.C.dyn else None
        if d:
            parts=[]
            if d.get("protein_half_life_h"): parts.append(f"protein t½ {d['protein_half_life_h']}h (response {d.get('response_time_h','?')}h)")
            if d.get("cell_cycle_phase"): parts.append(f"peaks {d['cell_cycle_phase']}")
            return dict(value="; ".join(parts), source="measured half-lives", confidence="medium") if parts else None
        return None
    def t_localize(self, name):
        sp=self.C.space.get(name) if self.C.space else None
        acc=" [SURFACE — drug-accessible]" if name in getattr(self.C,"surfaceome",set()) else ""
        if sp: return dict(value=f"{', '.join(sp.get('locations',[])[:4])}{acc}", source="HPA", confidence="high")
        i,g=self._g(name)
        return dict(value=f"{g.get('comp','?')}{acc}", source="model compartment", confidence="medium") if g else None
    def t_variants(self, name):
        v=self.C.variants.get(name) if self.C.variants else None
        if v and "clinvar_pathogenic" in v:
            tag=" ⚑ genetically-validated target" if name in getattr(self.C,"validated_targets",set()) else ""
            return dict(value=f"{v['clinvar_pathogenic']} pathogenic ClinVar variants (vulnerability {v.get('vulnerability_tier','?')}){tag}",
                        source="ClinVar+gnomAD", confidence="high" if v.get("review_stars",0)>=2 else "medium")
        return None
    def t_reverse(self, signature):
        if self.M is None:
            from compute_reverse_inference import load_model; self.M=load_model()
        from compute_reverse_inference import reverse_infer
        return reverse_infer(signature, self.M, top=10)

    # ---------- recipes (multi-hop macro-capabilities) ----------
    def target_dossier(self, name):
        name=self.C.upper.get(name.upper(), name)
        if name not in self.C.idx: return dict(error=f"'{name}' not recognized")
        d={k:self.tools[k][1](name) for k in ["function","efficacy","safety","mechanism","combinations",
                                              "druggability","kinetics","dynamics","localization","variants"]}
        # verdict: efficacy (disabling kills) x safety window x druggability
        eff=(d["efficacy"] or {}).get("num",0); lo=(d["safety"] or {}).get("num",1.0)
        drugged=(d["druggability"] or {}).get("drugged",False)
        verdict="promising" if (eff and eff>0.3) else ("context-dependent" if eff and eff>0.05 else "weak/undetermined")
        window="wider safety window" if (isinstance(lo,(int,float)) and lo>0.6) else "narrow window — watch on-target toxicity"
        return dict(gene=name, dossier=d, verdict=verdict, window=window,
                    novel="already drugged" if drugged else "undrugged (discovery)")

    def disease_to_target(self, signature, top=3):
        cands=self.t_reverse(signature)
        out=[]
        for c in (cands or [])[:top]:
            out.append(dict(candidate=c, dossier=self.target_dossier(c["gene"])))
        return dict(reverse_inference=cands, top_targets=out)

    def perturbation_effect(self, name):
        return dict(gene=name, cascade=self.t_cascade(name), timing=self.t_dynamics(name),
                    viability=self.t_efficacy(name))

    # ---------- router + synthesis ----------
    def ask(self, question, signature=None):
        ql=question.lower()
        if signature or any(w in ql for w in ["where does","disease live","phenotype","causal","reverse","find the target","source of"]):
            if not signature: return self._fmt_atomic(question)   # no signature -> can't reverse-infer
            return self._synth_disease(self.disease_to_target(signature), question)
        g=self.C.gene(question)
        if g and any(w in ql for w in ["target","will it work","disable","disabl","dossier","drug","develop","worth","inhibit"]):
            return self._synth_dossier(self.target_dossier(g), question)
        if g and any(w in ql for w in ["knock","perturb","what happens","overexpress","effect of"]):
            return self._synth_pert(self.perturbation_effect(g), question)
        return self._fmt_atomic(question)               # fall back to the atomic query engine

    def _fmt_atomic(self, q):
        from cell_query import answer
        a=answer(q, self.C); return f"{a['answer']}\n   [{a.get('confidence','?')} · {a.get('source','?')}]"

    def _line(self, label, r):
        return f"  • {label}: {r['value']}  [{r['confidence']} · {r['source']}]" if r else None
    def _synth_dossier(self, D, q):
        if D.get("error"): return D["error"]
        d=D["dossier"]; L=[f"TARGET DOSSIER — {D['gene']}  (verdict: disabling looks {D['verdict']}; {D['window']}; {D['novel']})"]
        for k,lab in [("function","Function"),("efficacy","Will disabling it work"),("safety","Safety window"),
                      ("mechanism","Downstream effect"),("combinations","Combine with (synthetic-lethal)"),
                      ("druggability","Drugs"),("kinetics","Kinetics"),("dynamics","Dynamics"),
                      ("localization","Location"),("variants","Genetic validation")]:
            ln=self._line(lab, d.get(k))
            if ln: L.append(ln)
        L.append("  (efficacy = DepMap measured; everything carries a source + confidence; no molecule/PK/clinical claim)")
        return "\n".join(L)
    def _synth_pert(self, P, q):
        L=[f"PERTURBATION EFFECT — {P['gene']}"]
        for lab,r in [("Downstream cascade",P["cascade"]),("Timing",P["timing"]),("Viability impact",P["viability"])]:
            ln=self._line(lab,r);
            if ln: L.append(ln)
        return "\n".join(L)
    def _synth_disease(self, R, q):
        L=["REVERSE INFERENCE — from the phenotype, the disease likely originates at:"]
        for c in (R.get("reverse_inference") or [])[:5]:
            L.append(f"  • {c['gene']} (score {c['score']}, {c['direction']}; MR {c['master_reg']}, cascade {c['cascade_match']})")
        if R.get("top_targets"):
            best=R["top_targets"][0]
            L.append("\n  → top candidate dossier:"); L.append(self._synth_dossier(best["dossier"], q))
        L.append("  [reverse-inference accuracy pending the LINCS real-data test — treat as hypothesis]")
        return "\n".join(L)

    # ---------- optional LLM layer: route free-form questions + phrase grounded facts (never invents) ----------
    def _llm(self, system, user, max_tokens=700):
        """call an LLM IF a key is set (Anthropic then OpenAI); else None. Used only to route + phrase."""
        try:
            key=os.environ.get("ANTHROPIC_API_KEY")
            if key:
                import anthropic
                m=anthropic.Anthropic(api_key=key).messages.create(
                    model=os.environ.get("AGENT_MODEL","claude-sonnet-5"), max_tokens=max_tokens,
                    system=system, messages=[{"role":"user","content":user}])
                return "".join(b.text for b in m.content if getattr(b,"type","")=="text")
        except Exception: pass
        try:
            key=os.environ.get("OPENAI_API_KEY")
            if key:
                import openai
                r=openai.OpenAI(api_key=key).chat.completions.create(
                    model=os.environ.get("AGENT_MODEL","gpt-4o"), max_tokens=max_tokens,
                    messages=[{"role":"system","content":system},{"role":"user","content":user}])
                return r.choices[0].message.content
        except Exception: pass
        return None

    def chat(self, question, signature=None):
        """LLM-fronted: the model gets the GROUNDED tool output and writes the answer. No key -> deterministic ask()."""
        grounded=self.ask(question, signature)                # every fact, sourced, from tools
        sys_p=("You are a cautious drug-discovery analyst. You are given GROUNDED facts from a virtual-cell "
               "model, each with a source and confidence. Write a clear, honest answer to the user's question "
               "USING ONLY these facts. Never add biology not in the facts. Keep the sources/confidence. If the "
               "facts don't answer it, say so. Note that the model gives cellular target-level evidence, not a "
               "clinical or molecule-design claim.")
        out=self._llm(sys_p, f"Question: {question}\n\nGrounded facts:\n{grounded}")
        return out or grounded                                # graceful fallback

def main():
    import sys
    a=CellAgent()
    q=" ".join(sys.argv[1:]) or "is MCL1 a good target to develop"
    print(a.ask(q))

if __name__=="__main__":
    main()
