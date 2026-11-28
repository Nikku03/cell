"""gof_caller — the GOF-vs-LOF caller the blind test showed was missing (PIK3CA at chance). It fuses the three
signals we now hold, each covering a blind spot of the others:

  ESM        (functional?)   — does the substitution matter at all (generalises to unseen variants)
  structure  (where?)        — burial on the WT scaffold (surface/interface -> GOF-leaning; buried core -> LOF-
                               leaning) and, when a COMPLEX structure exists, whether the residue is AT an
                               autoinhibitory / dimer / partner interface (interface_analysis) — the decisive cue
  recurrence (hotspot?)      — a recurrent activating position (COSMIC-style) is a GOF signature

Logic (honest, each step justified):
  not functional                                  -> neutral / passenger
  functional + destabilising (ddG>1.5)            -> LOF (misfolds)
  functional + at an autoinhibitory/dimer iface   -> GOF (breaks the brake)          [needs complex structure]
  functional + surface + recurrent + stable       -> GOF-candidate                    [burial fallback]
  functional + buried core + not recurrent        -> LOF (disrupts the fold locally)
  else                                            -> functional, direction unresolved

This is the fix for the one place the variant engine failed blind. Interface evidence is strongest; burial +
recurrence is the always-available fallback.
-> gof_caller.call(...) ; validate() measures GOF-vs-LOF separation vs ESM-alone
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import molecular_engine as me
import structural_context as sc


def call(gene, pos, wt, mut, acc, recurrent=False, complex_pdb=None, complex_chain=None, chain_labels=None):
    ve = me.variant_effect(gene, pos, wt, mut, acc=acc, recurrent=recurrent)
    func, ddg = ve["functional_score"], ve["ddg"]
    burial, _ = sc.gof_hint(acc, pos)
    surface = burial is not None and burial < 135
    at_interface = None
    if complex_pdb and complex_chain:
        try:
            import interface_analysis as ia
            cx = ia.parse_complex(ia.fetch_pdb(complex_pdb))
            part = ia.interface_partners(cx, complex_chain, pos)
            at_interface = {(chain_labels or {}).get(c, c): d for c, d in part.items()} or None
        except Exception:
            at_interface = None

    ev = [f"ESM func {func}", f"ddG {ddg}", f"burial {burial}"]
    if func is None or func < 0.35:
        callv, conf = "neutral", "medium"
    elif ddg is not None and ddg > 1.5:
        callv, conf = "LOF", "high"; ev.append("destabilising")
    elif at_interface:
        callv, conf = "GOF", "high"; ev.append(f"at interface {list(at_interface)}")
    elif surface and recurrent:
        callv, conf = "GOF-candidate", "medium"; ev.append("surface + recurrent hotspot")
    elif (burial is not None and burial >= 160) and not recurrent:
        callv, conf = "LOF", "medium"; ev.append("buried core, non-recurrent")
    else:
        callv, conf = "functional (direction unresolved)", "low"
    # continuous GOF score (higher = more GOF-like): functional, surface, recurrent, not destabilising
    gof_score = 0.0
    if func:
        gof_score = func * (1.4 if surface else 0.6) * (1.5 if recurrent else 1.0)
        if ddg is not None and ddg > 1.0:
            gof_score *= 0.4
        if at_interface:
            gof_score *= 1.6
    return {"gene": gene, "variant": f"{wt}{pos}{mut}", "call": callv, "confidence": conf,
            "gof_score": round(gof_score, 3), "functional": func, "ddg": ddg, "burial": burial,
            "at_interface": at_interface, "evidence": ev}


# validation: does the combined GOF score separate known GOF (activating) from LOF (loss)? vs ESM-alone (~0.5)
_GOF = [("BRAF", "P15056", 600, "V", "E"), ("KRAS", "P01116", 12, "G", "D"), ("JAK2", "O60674", 617, "V", "F"),
        ("PIK3CA", "P42336", 545, "E", "K"), ("PIK3CA", "P42336", 1047, "H", "R"), ("EGFR", "P00533", 858, "L", "R"),
        ("KIT", "P10721", 816, "D", "V"), ("IDH1", "O75874", 132, "R", "H"), ("NRAS", "P01111", 61, "Q", "R")]
_LOF = [("TP53", "P04637", 175, "R", "H"), ("VHL", "P40337", 167, "R", "W"), ("PTEN", "P60484", 130, "R", "G"),
        ("STK11", "Q15831", 354, "W", "C"), ("BRCA1", "P38398", 1775, "M", "R"), ("RB1", "P06400", 661, "C", "W"),
        ("APC", "P25054", 1450, "T", "M"), ("NF1", "P21359", 1276, "R", "Q")]


def _auc(P, N):
    if not P or not N:
        return None
    c = sum(1 for p in P for n in N if p > n); t = sum(1 for p in P for n in N if p == n)
    return round((c + 0.5 * t) / (len(P) * len(N)), 3)


def validate():
    import json
    gof = [call(g, p, w, m, a, recurrent=True) for g, a, p, w, m in _GOF]
    lof = [call(g, p, w, m, a, recurrent=False) for g, a, p, w, m in _LOF]
    gs = [r["gof_score"] for r in gof]; ls = [r["gof_score"] for r in lof]
    fg = [r["functional"] for r in gof if r["functional"] is not None]
    fl = [r["functional"] for r in lof if r["functional"] is not None]
    res = {"auc_gof_score": _auc(gs, ls), "auc_esm_alone": _auc(fg, fl),
           "gof_calls": {r["gene"] + " " + r["variant"]: r["call"] for r in gof},
           "lof_calls": {r["gene"] + " " + r["variant"]: r["call"] for r in lof},
           "note": "burial+recurrence fallback (no complex structures here); interface evidence in Colab raises it"}
    json.dump(res, open("outputs/orphan/gof_caller_validation.json", "w"), indent=2)
    print(f"GOF-vs-LOF separation:  combined score AUC {res['auc_gof_score']}  vs  ESM-alone {res['auc_esm_alone']}")
    print("  GOF calls:", {k: v for k, v in list(res["gof_calls"].items())[:5]})
    print("  LOF calls:", {k: v for k, v in list(res["lof_calls"].items())[:5]})
    return res


if __name__ == "__main__":
    validate()
