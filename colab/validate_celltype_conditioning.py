"""Validate cell-type conditioning HONESTLY: what the emask gate does and does NOT add.

The scorecard axis gates on the one claim that is robust and non-circular: the emask GATE is cell-type-specific
— canonical EXTERNAL (textbook, not model-derived) lineage markers are expressed in their own cell type far
above background and near-zero in unrelated lineages. That is what makes cell-type-conditioned answers correct.

It ALSO measures and records, without gating on them, the honest negatives — the versions of conditioning that
do NOT robustly beat the global model — so the capability's scope is documented, not oversold. -> outputs/orphan/celltype_conditioning_validation.json
"""
import os
import json, random, statistics as st
from pathlib import Path
from celltype_conditioning import CellTypeConditioner, _popcount

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))

# EXTERNAL canonical markers (textbook lineage markers, NOT derived from the model) + their cell-type name.
PANEL = [
    ("erythroid",  "erythroid progenitor",   ["GYPA", "SLC4A1", "ALAS2", "GATA1", "TAL1", "KLF1"]),
    ("monocyte",   "classical monocyte",     ["CD14", "CSF1R", "LYZ", "FCN1", "VCAN", "S100A8"]),
    ("B cell",     "B cell",                 ["MS4A1", "CD79A", "CD79B", "CD19", "PAX5", "BANK1"]),
    ("Treg",       "regulatory T cell",      ["FOXP3", "IL2RA", "CTLA4", "IKZF2"]),
    ("NK",         "natural killer cell",    ["NKG7", "GNLY", "KLRD1", "NCAM1", "NCR1"]),
    ("hepatocyte", "hepatocyte",             ["ALB", "APOB", "TTR", "SERPINA1", "CYP3A4", "ASGR1"]),
]

# curated-program TFs (small, curated target sets) vs their native lineage — the enrichment that DOES hold.
PROGRAMS = [("FOXP3", "regulatory T cell"), ("PAX5", "B cell"), ("SPI1", "classical monocyte")]


def gate_specificity(c):
    """robust claim: external markers land in their own lineage far above other lineages."""
    ct_idx = [c.resolve_celltype(nm) for _, nm, _ in PANEL]
    rows = []
    for (lab, nm, markers), t in zip(PANEL, ct_idx):
        if t is None:
            continue
        mk = [m for m in markers if m in c.idx]
        nat = sum(1 for m in mk if c.is_expressed(c.idx[m], t)) / max(1, len(mk))
        ctrl = [sum(1 for m in mk if c.is_expressed(c.idx[m], u)) / max(1, len(mk))
                for u in ct_idx if u is not None and u != t]
        br = c.base_rate(t)
        rows.append(dict(lineage=lab, n_markers=len(mk), native_expr=round(nat, 3),
                         control_mean=round(st.mean(ctrl), 3) if ctrl else None,
                         background=round(br, 3),
                         specificity=round(nat - (st.mean(ctrl) if ctrl else 0), 3),
                         fold_over_background=round(nat / br, 1) if br else None))
    return rows


def program_enrichment(c):
    out = []
    for tf, nm in PROGRAMS:
        r = c.program_enrichment(tf, nm)
        if r:
            out.append(dict(tf=tf, **r))
    return out


def coexpr_ppi_confound(c, n=15000, seed=0):
    """honest negative: emask co-expression predicts PPI, but mostly a hub artifact — weak once
    expression-degree is matched."""
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None
    rng = random.Random(seed)
    emk = list(c.emask)
    ppi = [(a, b) for a, b in ((e[0], e[1]) for e in c.D.get("ppi", []))
           if a in c.emask and b in c.emask][:n]
    if len(ppi) < 1000:
        return None
    shared = lambda a, b: _popcount(c.emask[a] & c.emask[b])
    # random negatives
    neg = []
    while len(neg) < len(ppi):
        a, b = rng.choice(emk), rng.choice(emk)
        if a != b:
            neg.append((a, b))
    y = [1] * len(ppi) + [0] * len(neg)
    auc_rand = roc_auc_score(y, [shared(a, b) for a, b in ppi + neg])
    # expression-degree-matched negatives (match each endpoint's popcount bucket)
    from collections import defaultdict
    byb = defaultdict(list)
    for g in emk:
        byb[_popcount(c.emask[g]) // 10].append(g)
    neg2 = []
    for a, b in ppi:
        ca = rng.choice(byb.get(_popcount(c.emask[a]) // 10, emk))
        cb = rng.choice(byb.get(_popcount(c.emask[b]) // 10, emk))
        neg2.append((ca, cb))
    auc_matched = roc_auc_score([1] * len(ppi) + [0] * len(neg2),
                                [shared(a, b) for a, b in ppi + neg2])
    return dict(auc_random_neg=round(float(auc_rand), 3), auc_degree_matched=round(float(auc_matched), 3))


def main():
    c = CellTypeConditioner()
    gate = gate_specificity(c)
    prog = program_enrichment(c)
    conf = coexpr_ppi_confound(c)

    min_spec = min(r["specificity"] for r in gate)
    mean_spec = round(st.mean(r["specificity"] for r in gate), 3)
    mean_fold = round(st.mean(r["fold_over_background"] for r in gate if r["fold_over_background"]), 1)
    passed = (len(gate) >= 5 and min_spec >= 0.4 and mean_fold >= 4.0)

    res = dict(
        passed=bool(passed),
        gate_specificity=gate,
        summary=dict(n_lineages=len(gate), min_specificity=min_spec, mean_specificity=mean_spec,
                     mean_fold_over_background=mean_fold),
        program_enrichment=prog,
        honest_negatives=dict(
            coexpr_ppi=conf,
            note=("conditioning is a correct-gate CAPABILITY, not a predictive lift over the global model: "
                  "co-expression PPI AUC collapses from ~0.72 (random neg) to ~0.57 (expression-degree matched, "
                  "a hub artifact); reachability specificity saturates on the dense directed graph (a random TF "
                  "reaches the same markers); only small CURATED regulatory programs (e.g. FOXP3) show cell-type "
                  "enrichment, bulk-inferred target sets do not.")),
        bar="external canonical markers: min specificity>=0.4 across >=5 lineages & mean >=4x background",
    )
    json.dump(res, open(OUT / "celltype_conditioning_validation.json", "w"), indent=2)

    print("=" * 78)
    print("CELL-TYPE CONDITIONING — emask gate specificity  —  %s" % ("PASS" if passed else "FAIL"))
    print("=" * 78)
    print("  robust claim: canonical EXTERNAL lineage markers land in their own cell type")
    for r in gate:
        print(f"   {r['lineage']:11} native {r['native_expr']:.2f}  vs control {r['control_mean']:.2f}  "
              f"({r['fold_over_background']}x background)  specificity {r['specificity']:+.2f}")
    print(f"\n  min specificity {min_spec:+.2f} | mean {mean_spec:+.2f} | mean {mean_fold}x over background")
    print("\n  curated-program enrichment (native lineage):")
    for p in prog:
        print(f"   {p['tf']:6} in {p['celltype'][:22]:22} enrichment {p['enrichment']}x")
    if conf:
        print(f"\n  honest negative — co-expression as PPI predictor: "
              f"AUC {conf['auc_random_neg']} (random neg) -> {conf['auc_degree_matched']} (degree-matched)")
    print("\n-> outputs/orphan/celltype_conditioning_validation.json")
    return res


if __name__ == "__main__":
    main()
