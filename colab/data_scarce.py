"""DOES CONTACT MATTER MORE WHEN THE 1D DATA IS POOR? The one live objection to the chromatin verdict.

THE OBJECTION, which is a good one. Every negative result in the contact ledger -- analytic Rouse +0.005,
stiffness sweep +0.0005, KMC extrusion + Langevin -0.0027, measured Hi-C -0.0031 -- was measured on a model
carrying 311 measured K562 TF ChIP tracks. K562 is ENCODE Tier 1. Almost no other cell type, and no patient
tissue, has anything like that. If the TF tracks are what made contact redundant, then the verdict is
"contact is redundant IN K562" rather than "contact is redundant", and the whole programme is alive again
for every cell type anyone actually cares about.

THE PREDICTION BEING TESTED, stated before running so it cannot be re-fitted afterwards: strip the TF
tracks and the baseline falls to ~0.55-0.58, and 3D contact then pulls it back toward 0.65+. That is a
claim about the MARGINAL VALUE OF CONTACT GROWING as 1D features are removed. It is not enough for the
baseline to fall -- of course it falls. The delta from contact has to GROW.

THE DESIGN. Three nested feature tiers, poorest first, and at each tier the same three contact arms:

    tier A   log_dist + ATAC only                  the data-scarce regime: what a new cell type has
    tier B   + H3K27ac, PolII, PROcap, promoter, expression    a well-characterised but not Tier 1 line
    tier C   + 311 TF ChIP tracks                  K562, where every previous result was measured

    arm 0    baseline
    arm 1    + CTCF-count contact  (the cheap proxy)
    arm 2    + measured Hi-C loops and domains  (the ceiling -- no simulation can exceed a measurement)

If contact's marginal value is flat across tiers, the redundancy is with DISTANCE, which every tier has,
and the verdict generalises. If it grows as tiers are stripped, the objection is right and the chromatin
programme should be reopened for data-poor cell types.

WHY MEASURED HI-C RATHER THAN THE SIMULATION. The question is whether contact information helps, not
whether our simulation of it does. Measured Hi-C is the ceiling: anything a polymer model could compute is
an approximation to it. A null here is a null for every possible contact model.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
INV = OUT / "invivo"
SEEDS = (0, 1, 2, 3, 4)

TIERS = [
    ("A  distance + ATAC only", ["log_dist", "atac_enh"], False),
    ("B  + full epigenetics", ["log_dist", "atac_enh", "h3k27ac_enh", "polr2a_enh", "procap_enh",
                               "promoter_atac", "promoter_polii", "gene_expr"], False),
    ("C  + 311 TF ChIP tracks", ["log_dist", "atac_enh", "h3k27ac_enh", "polr2a_enh", "procap_enh",
                                 "promoter_atac", "promoter_polii", "gene_expr"], True),
]


def main():
    from crispr_gate import _cv_auprc, _seeded_folds
    from contact_gate import load_ctcf, contact_features
    from hic_gate import read_bedpe, read_domains, hic_features, LOOPS, DOMAINS

    df = pd.read_csv(OUT / "crispr_features_compendium.csv")
    comp = json.load(open(INV / "compendium_tf.json"))
    et, tfl = comp["element_tfs"], comp["tf_list"]
    tss = {}
    with open(INV / "crispr_egpairs.tsv") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ix = {k: hdr.index(k) for k in ("chrom", "chromStart", "chromEnd", "startTSS", "measuredGeneSymbol")}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(hdr):
                continue
            try:
                tss[(f"{p[ix['chrom']]}:{p[ix['chromStart']]}-{p[ix['chromEnd']]}",
                     p[ix["measuredGeneSymbol"]])] = (
                    p[ix["chrom"]], (int(p[ix["chromStart"]]) + int(p[ix["chromEnd"]])) // 2,
                    int(p[ix["startTSS"]]))
            except ValueError:
                continue
    rows = [tss[(e, g)] for e, g in zip(df["element"], df["gene"])]
    Cr = contact_features(rows, load_ctcf())
    H = hic_features(rows, read_bedpe(LOOPS["intact"]), read_domains(DOMAINS["intact"]))
    tfmat = np.zeros((len(df), len(tfl)), np.int8)
    for a, ek in enumerate(df["element"].values):
        for ti in et.get(ek, []):
            tfmat[a, ti] = 1
    y, ch = df["crispr_hit"].values, df["chromosome"].values

    print("=" * 100)
    print("DOES CONTACT MATTER MORE WHEN THE 1D DATA IS POOR?")
    print("=" * 100)
    print(f"  {len(y):,} pairs, {int(y.sum())} positives, base rate {y.mean():.4f}")
    print("  PRE-REGISTERED: the objection is right only if the contact DELTA GROWS as tiers are stripped.")
    print("  A falling baseline alone proves nothing.\n")

    def sc(X):
        return np.array([_cv_auprc(X, y, _seeded_folds(ch, s)) for s in SEEDS])

    R = {"base_rate": float(y.mean()), "tiers": {}}
    print(f"  {'tier':28s} {'baseline':>9s} {'+CTCF':>9s} {'d_CTCF':>9s} {'+Hi-C':>9s} {'d_HiC':>9s}")
    print("  " + "-" * 78)
    for name, cols, use_tf in TIERS:
        X0 = df[cols].values
        if use_tf:
            X0 = np.hstack([X0, tfmat])
        b = sc(X0)
        c = sc(np.hstack([X0, Cr]))
        h = sc(np.hstack([X0, H]))
        dc, dh = (c - b), (h - b)
        print(f"  {name:28s} {b.mean():9.4f} {c.mean():9.4f} {dc.mean():+9.4f} "
              f"{h.mean():9.4f} {dh.mean():+9.4f}")
        R["tiers"][name.strip()] = {
            "n_features": int(X0.shape[1]), "baseline": float(b.mean()),
            "plus_ctcf": float(c.mean()), "delta_ctcf": float(dc.mean()),
            "delta_ctcf_per_seed": dc.tolist(),
            "plus_hic": float(h.mean()), "delta_hic": float(dh.mean()),
            "delta_hic_per_seed": dh.tolist()}

    ks = list(R["tiers"])
    dA_c, dC_c = R["tiers"][ks[0]]["delta_ctcf"], R["tiers"][ks[-1]]["delta_ctcf"]
    dA_h, dC_h = R["tiers"][ks[0]]["delta_hic"], R["tiers"][ks[-1]]["delta_hic"]
    print(f"\n  baseline falls {R['tiers'][ks[-1]]['baseline']:.4f} -> {R['tiers'][ks[0]]['baseline']:.4f} "
          f"as tiers are stripped ({R['tiers'][ks[-1]]['baseline']-R['tiers'][ks[0]]['baseline']:+.4f})")
    print(f"  CTCF-count delta   tier C {dC_c:+.4f}  ->  tier A {dA_c:+.4f}   "
          f"({'GROWS' if dA_c > dC_c + 0.01 else 'does NOT grow'})")
    print(f"  measured Hi-C delta tier C {dC_h:+.4f}  ->  tier A {dA_h:+.4f}   "
          f"({'GROWS' if dA_h > dC_h + 0.01 else 'does NOT grow'})")
    R["ctcf_delta_grows"] = bool(dA_c > dC_c + 0.01)
    R["hic_delta_grows"] = bool(dA_h > dC_h + 0.01)
    R["verdict"] = (
        "OBJECTION UPHELD -- contact is worth materially more when the 1D data is poor; the chromatin "
        "verdict is K562-specific and should be reopened for data-scarce cell types."
        if (R["ctcf_delta_grows"] or R["hic_delta_grows"]) else
        "OBJECTION REFUTED -- contact's marginal value does not grow as 1D features are stripped. The "
        "redundancy is with DISTANCE, which every tier retains, not with the 311 TF tracks.")
    print(f"\n  VERDICT: {R['verdict']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "data_scarce.json", "w"), indent=1)
    print(f"\n  -> {OUT/'data_scarce.json'}")


if __name__ == "__main__":
    main()
