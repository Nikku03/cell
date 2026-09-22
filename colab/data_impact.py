"""data_impact — evaluate a new data layer across ALL engines/metrics, not just one. Built after over-indexing on
the essentiality-reasoning AUC: a layer can lift one engine strongly and do nothing for the others (localization &
STRING lift essentiality a lot but the disease axis barely, because they are essentiality-correlated signals). This
scores each science-run layer against every engine so the honest, full impact is visible — a win on one metric is
NOT a win on the cell.

Metrics scored per layer:
  - essentiality-reason AUC lift (add the layer's line to the core 5)
  - disease-reason AUC lift (a different axis)
  - pathway-tier propagation (undirected network only): can neighbors recover a gene's tier, and reach
-> outputs/orphan/data_impact.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")
CORE = ("shock", "fba", "loeuf", "complex", "hub")     # the pre-science essentiality lines


def _layers(C):
    """binary essentiality-leaning line per science layer."""
    N = len(C.genes); name = lambda i: C.genes[i].get("name")
    out = {}
    loc = json.load(open(f"{OUT}/localization.json")).get("labels", {}) if os.path.exists(f"{OUT}/localization.json") else {}
    if loc:
        v = np.full(N, np.nan)
        for i in range(N):
            c = loc.get(name(i))
            if c:
                if c[0] in ("Nucleus", "Mitochondrion"):
                    v[i] = 1.0
                elif c[0] in ("Secreted", "Cell membrane", "Membrane"):
                    v[i] = 0.0
        out["localization"] = v
    sdj = json.load(open(f"{OUT}/string_degree.json")).get("layer", {}) if os.path.exists(f"{OUT}/string_degree.json") else {}
    if sdj:
        degs = np.array([sdj.get(name(i), {}).get("physical_degree", 0) for i in range(N)], float)
        thr = np.percentile(degs[degs > 0], 75)
        out["string_ppi"] = np.where(degs > 0, (degs >= thr).astype(float), np.nan)
    return out, sdj


def main():
    import reason as R, reasoner_core as rc, disease_reason as DR
    from complete_cell import CompleteCell
    from scipy.stats import spearmanr
    C = CompleteCell(); N = len(C.genes); name = lambda i: C.genes[i].get("name")
    ess = np.array([float(C.genes[i].get("dep_frac")) if str(C.genes[i].get("dep_frac")).replace(".", "", 1)
                    .replace("-", "", 1).isdigit() else np.nan for i in range(N)])
    dis = (np.array([C.genes[i].get("ndis", 0) or 0 for i in range(N)], float) > 0).astype(float)
    layers, sdj = _layers(C)

    allL = R.evidence_lines(C)
    core = {k: allL[k] for k in CORE}
    dcore = {k: v for k, v in DR.build_lines(C).items() if not k.startswith("loc")}

    ess_base = rc.reason_over(core, ess)["weighted_reasoning_auc"]
    dis_base = rc.reason_over(dcore, dis, min_lines=2)["weighted_reasoning_auc"]

    # pathway-tier network propagation (undirected → only guilt-by-association)
    lv = json.load(open(f"{OUT}/cell_levels.json"))["labels"]
    known = {g: r["level"] for g, r in lv.items() if r["status"] == "signaling-tier" and r.get("level") is not None}

    matrix = {}
    for lname, line in layers.items():
        e = rc.reason_over({**core, lname: line}, ess)["weighted_reasoning_auc"]
        d = rc.reason_over({**dcore, lname: line}, dis, min_lines=2)["weighted_reasoning_auc"]
        row = {"essentiality_reason": {"base": round(ess_base, 3), "with": round(e, 3), "lift": round(e - ess_base, 3)},
               "disease_reason": {"base": round(dis_base, 3), "with": round(d, 3), "lift": round(d - dis_base, 3)}}
        if lname == "string_ppi" and sdj:
            pt, tt = [], []
            for g, lvl in known.items():
                nb = [known[p["gene"]] for p in sdj.get(g, {}).get("partners", []) if p["gene"] in known]
                if len(nb) >= 3:
                    pt.append(np.mean(nb)); tt.append(lvl)
            reach = sum(1 for g, r in lv.items() if r["status"] == "no-level"
                        and sum(1 for p in sdj.get(g, {}).get("partners", []) if p["gene"] in known) >= 3)
            row["pathway_tier_propagation"] = {"spearman": round(float(spearmanr(tt, pt).statistic), 2),
                                               "reach_genes": reach, "note": "undirected → weak; not wired as trustworthy"}
        matrix[lname] = row

    print("=" * 90)
    print("DATA IMPACT MATRIX — each new layer scored across ALL engines (not just one)")
    print("=" * 90)
    for lname, row in matrix.items():
        print(f"\n  {lname}:")
        print(f"    essentiality-reason: {row['essentiality_reason']['base']} → {row['essentiality_reason']['with']}  "
              f"(lift {row['essentiality_reason']['lift']:+.3f})")
        print(f"    disease-reason:      {row['disease_reason']['base']} → {row['disease_reason']['with']}  "
              f"(lift {row['disease_reason']['lift']:+.3f})")
        if "pathway_tier_propagation" in row:
            pp = row["pathway_tier_propagation"]
            print(f"    pathway-tier (undirected propagation): Spearman {pp['spearman']}, reaches {pp['reach_genes']} "
                  f"no-context genes — WEAK, not wired")
    verdict = ("A layer that wins one engine is NOT a win for the cell. localization & STRING lift ESSENTIALITY "
               "reasoning strongly but the DISEASE axis barely (they are essentiality-correlated), and STRING's "
               "undirected network recovers pathway tier only weakly. Score every new layer across all engines.")
    print("\n" + "=" * 90 + f"\nVERDICT: {verdict}\n" + "=" * 90)
    json.dump({"matrix": matrix, "core_lines": list(CORE), "verdict": verdict},
              open(f"{OUT}/data_impact.json", "w"), indent=1)
    return verdict


if __name__ == "__main__":
    main()
