"""reasoning_chain — ONE connected chain, not separate departments. Each step's OUTPUT conditions the next, so
the whole reasons end-to-end from a mutation to a targetable conclusion.

HONEST division of labour (this is the corrected design; see colab/reasoning_chain_test.py for why):
  - "does the variant MATTER?" is answered by ESM + ΔΔG + recurrence ONLY. These are the pathogenicity signals.
  - "IF it matters, HOW?" (which interaction breaks) is answered by interface localisation. This is a MECHANISM
    signal, NOT a pathogenicity signal — a blind ClinVar test on VHL showed interface membership is identical in
    pathogenic (34%) and benign (33%) variants (Fisher p=1.0). VHL is a tiny all-interface adaptor, so "at an
    interface" carries almost no information about whether a variant is damaging. Therefore localisation NEVER
    rescues significance on its own; it only sets the MODE that conditions how an already-significant break spreads.

The real, surviving value is coupling (b): localisation changes WHAT gets propagated. If a significant mutation
breaks a SPECIFIC contact (VHL–HIF), release ONLY that substrate through the signed network — VHL still binds
ElonginC, only HIF escapes. If instead it is a whole-product loss (destabilising / not a partner contact), propagate
from the gene node (all edges). Same gene, two mutations, two different chains — driven by validated mechanism
attribution, not by a significance claim the data does not support.

  STEP 1  does it matter?        ESM + ΔΔG + recurrence               -> the ONLY significance gate
  STEP 2  what breaks, exactly?  interface_analysis on the complex     -> MECHANISM (a specific contact / whole
                                                                          product); conditional on step 1, never a
                                                                          significance rescue
  --- GATE: prune if step 1 is neutral. If step 1 neutral but at a known interface -> CONDITIONAL hypothesis (low
      confidence), explicitly not a call. ---
  STEP 3  so what downstream?    propagate THAT break through the signed network
             interaction-specific -> inject on the released substrate(s) only (the gene's other arms untouched)
             whole-product LOF    -> inject -1 on the gene node (all edges)
             GOF activation       -> inject +1 on the gene node
  STEP 4  therefore phenotype?   read the up/down programme as a hallmark
  STEP 5  therefore target?      the released driver / the gene's context dependency

Confidence is honest: significance drives it; a variant ESM/ΔΔG miss but that sits at an interface is returned as a
low-confidence conditional hypothesis, never a confident call.
-> reason(...) returns the connected trace + conclusion + confidence.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
import molecular_engine as me
import cancer_cell_map as ccm

_C = {"cell": None}
_STRENGTH = {"high": 1.0, "medium": 0.6, "low": 0.3}


def _cell():
    if _C["cell"] is None:
        from complete_cell import CompleteCell
        _C["cell"] = CompleteCell()
    return _C["cell"]


def _edge_sign(out, i, j):
    for w, s in out.get(i, ()):
        if w == j:
            return s
    return 0.0


def _agg_conf(evidence):
    """evidence = [(name, significant_bool, strength_str)]. Honest aggregation:
    - no significant signal        -> ~0.2 (passenger)
    - one significant signal       -> its strength, discounted (uncorroborated)
    - two+ significant signals     -> best strength, boosted (corroboration)"""
    sig = [_STRENGTH[s] for _, ok, s in evidence if ok]
    if not sig:
        return 0.2
    best = max(sig)
    if len(sig) == 1:
        return round(best * 0.8, 2)
    return round(min(1.0, best + 0.15), 2)


def reason(gene, pos, wt, mut, acc=None, recurrent=False,
           complex_pdb=None, complex_chain=None, chain_labels=None, partner_map=None):
    C = _cell()
    trace, evidence, notes = [], [], []
    i = C.idx.get(gene)
    role_gof = gene in getattr(ccm, "ONCOGENE", set())
    role_lof = gene in getattr(ccm, "SUPPRESSOR", set())

    # ---- STEP 1: does it MATTER? molecular signal (ESM + ΔΔG) + independent recurrence prior ----
    # These are the ONLY significance signals. Interface localisation is deliberately NOT one of them: a blind test
    # on VHL (colab/reasoning_chain_test.py) showed interface membership does NOT separate pathogenic from benign
    # (34% vs 33%, Fisher p=1.0) -- it is a MECHANISM signal, not a pathogenicity signal, so it must never rescue
    # significance on its own or the chain manufactures confident calls from a non-discriminative feature.
    ve = me.variant_effect(gene, pos, wt, mut, acc=acc, recurrent=recurrent)
    fs = ve["functional_score"]
    mol_sig = (fs is not None and fs >= 0.35)
    destabilising = (ve["ddg"] is not None and ve["ddg"] > 1.5)
    trace.append({"step": "1 molecular", "finding": ve["call"], "significant": bool(mol_sig or destabilising),
                  "detail": f"ESM func {fs}, ddG {ve['ddg']}", "confidence": ve["confidence"]})
    evidence.append(("molecular", bool(mol_sig or destabilising),
                     "high" if (fs and fs >= 0.6) else "medium" if mol_sig else "low"))
    if recurrent:                                              # recurrence = independent statistical evidence it matters
        evidence.append(("recurrence", True, "medium"))
        trace.append({"step": "1b prior", "finding": "recurrent somatic hotspot (independent significance signal)",
                      "significant": True, "confidence": "medium"})
    significant = mol_sig or destabilising or recurrent

    # ---- STEP 2: localisation — MECHANISM attribution (WHICH interaction breaks), conditional on step 1 ----
    # Not a significance signal (see above). It sets the MODE that conditions how the break propagates.
    broken_partners, mode, at_interface = [], None, False
    if complex_pdb and complex_chain:
        import interface_analysis as ia
        r = ia.analyze(complex_pdb, complex_chain, pos, chain_labels)
        at_interface = bool(r["at_interface"])
        if at_interface:
            broken_partners = list(r["contacts"]); mode = "interaction-specific"
            trace.append({"step": "2 localisation", "significant": False, "confidence": "high",
                          "finding": f"mechanism: AT the {'/'.join(broken_partners)} interface (exp. {complex_pdb}, "
                                     f"{min(r['contacts'].values())} A) — mechanism, not evidence of pathogenicity",
                          "detail": r["contacts"]})
        else:
            mode = "whole-product"
            trace.append({"step": "2 localisation", "significant": False, "confidence": "medium",
                          "finding": "mechanism: not at any partner interface -> intramolecular "
                                     "(stability/allostery), whole-product effect", "detail": r["verdict"]})
    else:
        import structural_context as sc
        b, _ = sc.gof_hint(acc, pos)
        buried = (b is not None and b >= 135)
        if destabilising:
            mode = "whole-product"
            trace.append({"step": "2 localisation", "significant": False, "confidence": "medium",
                          "finding": "mechanism: destabilising (ΔΔG) -> whole product lost", "detail": f"burial {b}"})
        elif recurrent and mol_sig and not destabilising:
            mode = "gof-activation"
            trace.append({"step": "2 localisation", "significant": False, "confidence": "medium",
                          "finding": "mechanism: recurrent, functional, stability-neutral -> activating "
                                     "(exact interface needs AF-Multimer)", "detail": f"burial {b}"})
        else:
            mode = "whole-product" if (mol_sig and (role_lof or not role_gof)) else "interface-unresolved"
            trace.append({"step": "2 localisation", "significant": False, "confidence": "low",
                          "finding": f"mechanism: {'buried' if buried else 'surface'}, no complex structure "
                                     f"-> interface unresolved", "detail": f"burial {b}"})

    # ---- GATE: significance is decided by step 1 ONLY. Structure never rescues it. ----
    if not significant:
        if at_interface:                                       # honest: unconfirmed, but mechanism is localised
            notes.append("significance UNCONFIRMED — ESM/ΔΔG do not flag this and interface membership is NOT itself "
                         "evidence of pathogenicity (validated null on VHL, Fisher p=1.0). Reported as a CONDITIONAL "
                         "mechanism hypothesis, not a call: needs an independent signal (clinical recurrence, assay).")
            return {"gene": gene, "variant": f"{wt}{pos}{mut}", "mode": "significance-uncertain",
                    "conclusion": f"{gene} {wt}{pos}{mut}: cannot confirm it matters (ESM/ΔΔG neutral). IF pathogenic, "
                                  f"mechanism = {'/'.join(broken_partners)} interface loss (conditional hypothesis)",
                    "confidence": 0.3, "notes": notes, "chain": trace}
        return {"gene": gene, "variant": f"{wt}{pos}{mut}", "mode": "tolerated",
                "conclusion": "tolerated passenger — ESM/ΔΔG neutral, no recurrence; chain stops",
                "confidence": _agg_conf(evidence), "notes": notes, "chain": trace}

    if i is None:
        trace.append({"step": "3 propagation", "finding": "gene not in network", "confidence": "low"})
        return {"gene": gene, "variant": f"{wt}{pos}{mut}", "mode": mode,
                "conclusion": "chain breaks: gene absent from network",
                "confidence": _agg_conf(evidence), "notes": notes, "chain": trace}

    # ---- decide GOF vs LOF (direction of the gene-level effect) ----
    is_gof = (mode == "gof-activation") or ve["call"].startswith("GOF") or (role_gof and mode != "whole-product"
                                                                            and recurrent and mol_sig)

    # ---- STEP 3: propagate the SPECIFIC break (localisation decides WHAT is injected) ----
    out = ccm._signed_out(C)
    released = []
    if mode == "interaction-specific":
        inj = {}
        for lbl in broken_partners:
            pg = (partner_map or {}).get(lbl, str(lbl).upper())
            j = C.idx.get(pg)
            if j is None:
                continue
            e = _edge_sign(out, i, j)
            if e:                                              # losing a (repressive) interaction flips the partner
                inj[j] = -1.0 if e > 0 else 1.0
                released.append(pg)
        if not inj:                                            # at interface but partner has no modelled edge
            inj = {i: -1.0}
            notes.append(f"interface partner(s) {broken_partners} have no modelled downstream edge; "
                         f"fell back to gene-level LOF")
            mode = "interaction-specific (unmodelled partner)"
    elif is_gof:
        inj = {i: 1.0}
    else:                                                      # whole-product LOF / interface-unresolved
        inj = {i: -1.0}

    pert = ccm.propagate_signed(out, inj)
    paths = ccm._pathways(C)
    up, down = ccm.pathway_delta(pert, paths)
    top_up = [p["pathway"] for p in up[:3]]
    top_dn = [p["pathway"] for p in down[:3]]
    net_up = sum(inj.values()) >= 0
    drives_up = is_gof or net_up                               # released substrate (net +) drives its programme UP
    trace.append({"step": "3 propagation", "mode": mode, "confidence": "medium",
                  "finding": (f"release {released} -> " if released else "") +
                             ("drives UP" if drives_up else "propagates loss of") + " downstream programme",
                  "inject": {list(C.name)[k] if k < len(C.name) else k: round(v, 2) for k, v in inj.items()},
                  "up": top_up, "lost": top_dn})

    # ---- STEP 4: phenotype (read the driven programme) ----
    prog = top_up if drives_up else top_dn
    trace.append({"step": "4 phenotype", "confidence": "medium",
                  "finding": " / ".join(prog[:2]) or "no clear programme"})

    # ---- STEP 5: target ----
    cdp = getattr(C, "context_dep", {}).get(i) or {}
    if released:
        target = f"target the released driver(s): {'/'.join(released)} (e.g. HIF axis)"
    elif is_gof:
        target = f"revert {gene} (the activated driver)"
    elif cdp.get("addiction_context"):
        target = f"synthetic-lethal / context dependency (addiction: {cdp['addiction_context'][0][0]})"
    elif cdp.get("orphan"):
        target = (f"{gene} is a lost suppressor; its selective dependency (rank #{cdp.get('selective_rank')}) "
                  f"is orphan -> needs lesion attribution")
    else:
        target = f"{gene} lost -> downstream / synthetic-lethal target (no direct drug)"
    trace.append({"step": "5 target", "finding": target, "confidence": "medium"})

    head = ("breaks " + "/".join(released or broken_partners)) if broken_partners else mode
    concl = (f"{gene} {wt}{pos}{mut} [{ve['call']}] {head} -> "
             f"{'activates' if drives_up else 'loses'} {prog[0] if prog else 'programme'} -> {target}")
    return {"gene": gene, "variant": f"{wt}{pos}{mut}", "mode": mode, "conclusion": concl,
            "confidence": _agg_conf(evidence), "notes": notes, "chain": trace}


def run(cases=None):
    import json
    labels = {"B": "ElonginB", "C": "ElonginC", "V": "VHL", "H": "HIF1a"}
    pmap = {"HIF1a": "HIF1A", "ElonginC": "ELOC", "ElonginB": "ELOB"}
    cases = cases or [
        dict(gene="VHL", pos=115, wt="Y", mut="H", acc="P40337", complex_pdb="1LM8", complex_chain="V",
             chain_labels=labels, partner_map=pmap),
        dict(gene="VHL", pos=167, wt="R", mut="W", acc="P40337", complex_pdb="1LM8", complex_chain="V",
             chain_labels=labels, partner_map=pmap),
        dict(gene="BRAF", pos=600, wt="V", mut="E", acc="P15056", recurrent=True),
        dict(gene="TP53", pos=175, wt="R", mut="H", acc="P04637"),
    ]
    out = []
    for c in cases:
        r = reason(**c)
        out.append(r)
        print("=" * 88)
        print(f"REASONING CHAIN — {r['gene']} {r['variant']}   mode={r['mode']}   (confidence {r['confidence']})")
        print("=" * 88)
        for s in r["chain"]:
            extra = ""
            if "up" in s:
                extra = f"\n{'':20}UP:{[p[:24] for p in s['up']]}  LOST:{[p[:24] for p in s['lost']]}"
            sig = "*" if s.get("significant") else " "
            print(f" {sig}{s['step']:16} {str(s['finding'])[:64]}{extra}")
        for n in r.get("notes", []):
            print(f"  !note: {n}")
        print(f"  => {r['conclusion'][:170]}")
        print()
    os.makedirs("outputs/orphan", exist_ok=True)
    json.dump(out, open("outputs/orphan/reasoning_chain.json", "w"), indent=2, default=str)
    return out


if __name__ == "__main__":
    run()
