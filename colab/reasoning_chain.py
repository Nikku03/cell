"""reasoning_chain — ONE connected chain, not separate departments. Each step's OUTPUT conditions the next, so
the whole reasons end-to-end from a mutation to a targetable conclusion. The value is the COUPLING, and two
couplings in particular:

  (a) a later, better-grounded step can RESCUE a weaker earlier one. ESM (sequence) misses VHL Y115H (it scores
      it tolerated). But the experimental complex says residue 115 sits 3.2 A from HIF-1a. The chain does not let
      the noisy sequence model veto the structure — it INTEGRATES them, and the structure carries the call.
  (b) localisation changes WHAT gets propagated. If the mutation breaks a SPECIFIC contact (VHL-HIF), we release
      ONLY that substrate through the signed network — VHL still binds ElonginC, only HIF escapes. If instead the
      residue is functionally important but NOT a partner contact (VHL R167W: ESM strong, not at any interface),
      the whole product is compromised and we propagate from the gene node (all its edges). Same gene, two
      mutations, two different chains, two different mechanistic conclusions.

  STEP 1  does it matter?        molecular_engine (ESM + ddG)          -> one piece of evidence, NOT a veto
  STEP 2  what breaks, exactly?  interface_analysis on the complex     -> a SPECIFIC contact, or 'whole product'
             (can rescue step 1)                                          <-- localisation
  --- JOINT GATE: prune as passenger only if NO step found it significant (molecular OR structural) ---
  STEP 3  so what downstream?    propagate THAT break through the signed network
             interaction-specific -> inject on the released substrate(s) only (VHL's other arms untouched)
             whole-product LOF    -> inject -1 on the gene node (all edges)
             GOF activation       -> inject +1 on the gene node
  STEP 4  therefore phenotype?   read the up/down programme as a hallmark
  STEP 5  therefore target?      the released driver / the gene's context dependency

Confidence accumulates HONESTLY: two corroborating steps raise it; a single uncorroborated step is discounted;
a contradiction (one step significant, the other not) is surfaced as a note, not silently resolved.
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

    # ---- STEP 1: molecular signal (evidence, NOT a veto) ----
    ve = me.variant_effect(gene, pos, wt, mut, acc=acc, recurrent=recurrent)
    fs = ve["functional_score"]
    mol_sig = (fs is not None and fs >= 0.35)
    destabilising = (ve["ddg"] is not None and ve["ddg"] > 1.5)
    trace.append({"step": "1 molecular", "finding": ve["call"], "significant": bool(mol_sig or destabilising),
                  "detail": f"ESM func {fs}, ddG {ve['ddg']}", "confidence": ve["confidence"]})
    evidence.append(("molecular", bool(mol_sig or destabilising),
                     "high" if (fs and fs >= 0.6) else "medium" if mol_sig else "low"))

    # ---- STEP 2: localisation (structural fact; runs regardless of step 1 so it can RESCUE it) ----
    broken_partners, mode = [], None
    if complex_pdb and complex_chain:
        import interface_analysis as ia
        r = ia.analyze(complex_pdb, complex_chain, pos, chain_labels)
        if r["at_interface"]:
            broken_partners = list(r["contacts"]); mode = "interaction-specific"
            trace.append({"step": "2 localisation", "significant": True, "confidence": "high",
                          "finding": f"AT the {'/'.join(broken_partners)} interface (exp. {complex_pdb}, "
                                     f"{min(r['contacts'].values())} A)", "detail": r["contacts"]})
            evidence.append(("localisation", True, "high"))
            if not mol_sig:
                notes.append("sequence model (ESM) scored this tolerated, but the experimental complex places it "
                             "on a partner interface -> structure carries the call")
        else:
            mode = "whole-product"
            trace.append({"step": "2 localisation", "significant": bool(destabilising or mol_sig),
                          "confidence": "medium", "detail": r["verdict"],
                          "finding": "not at any partner interface -> intramolecular (stability/allostery), "
                                     "whole-product effect"})
            evidence.append(("localisation", bool(destabilising), "medium"))
    else:
        import structural_context as sc
        b, _ = sc.gof_hint(acc, pos)
        buried = (b is not None and b >= 135)
        if destabilising:
            mode = "whole-product"
            trace.append({"step": "2 localisation", "significant": True, "confidence": "medium",
                          "finding": "destabilising (ddG) -> whole product lost", "detail": f"burial {b}"})
            evidence.append(("localisation", True, "medium"))
        elif recurrent and mol_sig and not destabilising:
            mode = "gof-activation"
            trace.append({"step": "2 localisation", "significant": True, "confidence": "medium",
                          "finding": "recurrent, functional, stability-neutral -> activating "
                                     "(exact interface needs AF-Multimer)", "detail": f"burial {b}"})
            evidence.append(("localisation", True, "low"))
        else:
            mode = "whole-product" if (mol_sig and (role_lof or not role_gof)) else "interface-unresolved"
            trace.append({"step": "2 localisation", "significant": False, "confidence": "low",
                          "finding": f"{'buried' if buried else 'surface'}, no complex structure "
                                     f"-> interface unresolved", "detail": f"burial {b}"})
            evidence.append(("localisation", False, "low"))

    # ---- JOINT GATE: prune only if NEITHER molecular NOR structural evidence was significant ----
    if not any(ok for _, ok, _ in evidence):
        return {"gene": gene, "variant": f"{wt}{pos}{mut}", "mode": "tolerated",
                "conclusion": "tolerated passenger — no molecular, structural or interface evidence; chain stops",
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
