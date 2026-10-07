"""disease_reason — a SECOND reasoning engine on the same reasoner_core: is a gene DISEASE-associated? This is a
different axis from essentiality (disease and cell-fitness overlap only 11% vs 9% base — nearly orthogonal), so it
is a real new engine, not a relabel. Ground truth: ndis > 0 (disease-association count, from disease databases).

Evidence lines chosen to be INDEPENDENT of the disease databases (no circularity — and NO literature/pub count,
which would be circular since disease genes are studied more):
  loeuf      — evolutionary constraint (dominant-disease genes are intolerant to loss)   [gnomAD]
  complex    — member of a physical machine                                              [CORUM]
  hub        — regulatory out-degree                                                     [SIGNOR]
  pleiotropy — sits in many pathways                                                     [Reactome]
  loc_surface— secreted / cell-membrane (receptors & secreted = classic disease/target)  [UniProt]
  loc_nuclear— nuclear (TFs / chromatin disease)                                         [UniProt]

Tests the same two things: does fusing beat the best single line, and is confidence calibrated?
-> outputs/orphan/disease_reason.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


def build_lines(C):
    N = len(C.genes)
    L = {k: np.full(N, np.nan) for k in ("loeuf", "complex", "hub", "pleiotropy", "loc_surface", "loc_nuclear")}
    g2c = C.D.get("gene2cplx", {}) or {}
    cplx = {int(x) for x in g2c}
    loc = json.load(open(f"{OUT}/localization.json")).get("labels", {}) if os.path.exists(f"{OUT}/localization.json") else {}
    npaths = [C.genes[i].get("npath", 0) or 0 for i in range(N)]
    med_np = np.median([n for n in npaths if n > 0]) if any(npaths) else 1
    for i in range(N):
        g = C.genes[i]
        lo = g.get("loeuf")
        if isinstance(lo, (int, float)):
            L["loeuf"][i] = float(lo < 0.35)
        L["complex"][i] = float(i in cplx)
        L["hub"][i] = float(len(C.causal_out.get(i, []) or []) >= 20)
        L["pleiotropy"][i] = float((g.get("npath", 0) or 0) >= med_np)
        comps = loc.get(g.get("name"))
        if comps:
            L["loc_surface"][i] = float(any(c in ("Secreted", "Cell membrane") for c in comps))
            L["loc_nuclear"][i] = float("Nucleus" in comps)
    return L


def main():
    import reasoner_core as rc
    from complete_cell import CompleteCell
    C = CompleteCell(); N = len(C.genes)
    ndis = np.array([C.genes[i].get("ndis", 0) or 0 for i in range(N)], float)
    truth = (ndis > 0).astype(float)                       # disease-associated
    L = build_lines(C)
    rep = rc.reason_over(L, truth, min_lines=3)
    print("=" * 82)
    print("DISEASE-GENE REASONING — a second engine on reasoner_core (truth: ndis>0)")
    print("=" * 82)
    rc.print_report(rep, entity="gene", truth_name="disease-gene")
    save = {k: rep[k] for k in ("n", "n_truth_pos", "lines", "singles", "convergence_auc",
                                "weighted_reasoning_auc", "best_single", "calibration", "verdict")}
    save["axis_note"] = "disease != essentiality (11% vs 9% overlap); independent lines, no literature (circular)"
    # per-gene calibrated disease prior (by # agreeing lines), for the 'disease' syscall
    calib = {int(k): v["p_truth"] for k, v in rep["calibration"].items()}
    agree, avail = rep["agree"], rep["avail"]
    priors = {}
    for i in range(N):
        if avail[i] >= 3:
            na = int(round(agree[i]))
            priors[C.genes[i].get("name")] = {"n_agree": na, "n_lines": int(avail[i]),
                                              "p_disease": calib.get(na), "known_disease": bool(truth[i])}
    save["priors"] = priors
    json.dump(save, open(f"{OUT}/disease_reason.json", "w"), indent=1)
    return rep


if __name__ == "__main__":
    main()
