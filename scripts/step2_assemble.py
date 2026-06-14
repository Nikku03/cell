"""STEP 2a: assemble the condition-conditional training frame.

Joins the CES consensus label (Paper-2's target) to our per-gene features.

THE JOIN (verified clean, 100%): the consensus is keyed by feba
(orgId, locusId); our feature tables are keyed by (organism, locus_tag).
Our beril_* organisms ARE the feba organisms (strip "beril_" -> feba orgId),
and our `locus_tag` IS feba's internal `locusId` (3887/3887 match on ANA3).
So  organism = "beril_" + consensus.organism  AND  locus_tag = consensus.locusId
joins with zero mapping loss.

Output (outputs/step2_training_frame.parquet), one row per
(organism, locus_tag, condition_cluster):
  TARGETS
    consensus_fitness        continuous (the thing we regress)
    consensus_essential      cor12-weighted fraction in [0,1]
  CONSERVATION BASELINE (the bar to beat)
    family_frac              leak-free cross-organism essential fraction
    prevalence_og            family_n_organisms / 48
    n_paralogs, is_orphan
  CONDITION (the axis that makes it condition-conditional)
    condition_cluster        (kept as id; embedding built in 2b)
  EXTRA per-gene features
    cai, gc, gc3, cds_length (codon_features)
    cooccur_* , is_regulator/transporter, fba_essential   (where available)
  ABSTENTION / CALIBRATION signals (conservative bias)
    n_screens, mean_cor12, total_weight, within_disagreement
  SPLIT
    clade                    for leave-one-clade-out evaluation

--smoke : validate the join + frame assembly on synthetic inputs (no Colab data)
--real  : read the real consensus parquet + feature CSVs and write the frame
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LAB = REPO / "data" / "drive_import" / "labels"
OUT = REPO / "outputs"
# consensus parquet location: outputs/ (this session) or Drive (persisted)
CONSENSUS_CANDIDATES = [
    OUT / "ces_consensus.parquet",
    Path("/content/drive/MyDrive/ces_consensus.parquet"),
]


def _find_consensus(explicit=None):
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"--consensus {explicit} not found")
    for p in CONSENSUS_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"ces_consensus.parquet not found in {CONSENSUS_CANDIDATES}; "
        f"run ces_consensus.py --real first, or pass --consensus PATH")


def assemble(consensus, orth, codon, cooccur, reg, fba, clades):
    """Join consensus -> features. All our feature tables key on
    (organism, locus_tag); consensus keys on (organism=feba orgId, locusId).
    Bridge: organism='beril_'+consensus.organism, locus_tag=consensus.locusId."""
    import pandas as pd
    c = consensus.copy()
    c["organism"] = "beril_" + c["organism"].astype(str)
    c = c.rename(columns={"locusId": "locus_tag"})
    c["locus_tag"] = c["locus_tag"].astype(str)

    n0 = len(c)
    # conservation baseline
    orth = orth.copy()
    orth["prevalence_og"] = orth["family_n_organisms"] / 48.0
    keep_orth = ["organism", "locus_tag", "family_frac_essential_leakfree",
                 "prevalence_og", "n_paralogs_in_genome", "is_orphan"]
    df = c.merge(orth[keep_orth], on=["organism", "locus_tag"], how="left")
    df = df.rename(columns={"family_frac_essential_leakfree": "family_frac"})
    cons_join = df.family_frac.notna().mean()
    print(f"  consensus rows: {n0:,};  family_frac join: {cons_join:.1%}")

    # codon features
    if codon is not None:
        df = df.merge(codon[["organism", "locus_tag", "cai", "gc", "gc3",
                             "cds_length"]], on=["organism", "locus_tag"],
                      how="left")
    # cooccurrence
    if cooccur is not None:
        cc = [col for col in ["cooccur_max_jaccard", "cooccur_n_neighbors_50",
                              "cooccur_essential_neighbor_frac"]
              if col in cooccur.columns]
        df = df.merge(cooccur[["organism", "locus_tag"] + cc],
                      on=["organism", "locus_tag"], how="left")
    # regulator
    if reg is not None:
        rc = [col for col in ["is_regulator", "is_transporter", "is_signaling",
                              "is_conditional"] if col in reg.columns]
        df = df.merge(reg[["organism", "locus_tag"] + rc],
                      on=["organism", "locus_tag"], how="left")
    # fba
    if fba is not None and "fba_essential" in fba.columns:
        df = df.merge(fba[["organism", "locus_tag", "fba_essential"]],
                      on=["organism", "locus_tag"], how="left")
    # clade for LOCO splits
    df = df.merge(clades[["organism", "clade"]], on="organism", how="left")
    clade_join = df.clade.notna().mean()
    print(f"  clade join: {clade_join:.1%}")
    return df


def run_real(args):
    import pandas as pd
    print("=" * 64)
    print("STEP 2a -- assemble condition-conditional training frame")
    print("=" * 64)
    cpath = _find_consensus(args.consensus)
    print(f"  consensus: {cpath}")
    consensus = pd.read_parquet(cpath)
    orth = pd.read_csv(LAB / "orthology_features.csv")
    codon = _try(LAB / "codon_features.csv")
    cooccur = _try(LAB / "cooccurrence_features.csv")
    reg = _try(LAB / "regulator_features.csv")
    fba = _try(LAB / "fba_features.csv")
    clades = pd.read_csv(LAB / "clade_splits.csv")

    df = assemble(consensus, orth, codon, cooccur, reg, fba, clades)
    # keep rows with the conservation baseline present (needed to compare)
    df = df.dropna(subset=["family_frac", "consensus_fitness", "clade"])
    OUT.mkdir(exist_ok=True)
    out = OUT / "step2_training_frame.parquet"
    df.to_parquet(out, index=False)

    print(f"\n  TRAINING FRAME: {len(df):,} rows  "
          f"{df.organism.nunique()} orgs  "
          f"{df.condition_cluster.nunique()} conditions  "
          f"{df.clade.nunique()} clades")
    print(f"  essential (consensus>=0.5): {int((df.consensus_essential>=0.5).sum()):,}")
    print(f"  multi-screen rows: {int((df.n_screens>=2).sum()):,} "
          f"({100*(df.n_screens>=2).mean():.0f}%)")
    feat_cols = [c for c in df.columns if c not in
                 ("organism", "locus_tag", "condition_cluster",
                  "consensus_fitness", "consensus_essential", "clade")]
    print(f"  feature columns ({len(feat_cols)}): {feat_cols}")
    # feature coverage
    print("  feature non-null coverage:")
    for c in feat_cols:
        print(f"    {c:<28s} {df[c].notna().mean():.1%}")
    (OUT / "step2_frame_summary.json").write_text(json.dumps({
        "n_rows": int(len(df)), "n_orgs": int(df.organism.nunique()),
        "n_conditions": int(df.condition_cluster.nunique()),
        "n_clades": int(df.clade.nunique()),
        "n_essential": int((df.consensus_essential >= 0.5).sum()),
        "pct_multiscreen": round(100*(df.n_screens >= 2).mean(), 1),
        "feature_cols": feat_cols,
    }, indent=2))
    print(f"\nwrote {out} + summary")
    return 0


def _try(path):
    import pandas as pd
    return pd.read_csv(path) if path.exists() else None


def run_smoke():
    import pandas as pd, numpy as np
    print("=" * 64)
    print("STEP 2a SMOKE -- join + frame assembly on synthetic data")
    print("=" * 64)
    # consensus keyed by feba (orgId, locusId)
    consensus = pd.DataFrame({
        "organism": ["ANA3", "ANA3", "MR1"],
        "locusId": ["11530891", "11530892", "999"],
        "condition_cluster": ["LB|aerobic|none", "M9|aerobic|glc",
                              "LB|aerobic|none"],
        "consensus_fitness": [-3.0, 0.1, -2.0],
        "consensus_essential": [1.0, 0.0, 0.8],
        "n_screens": [3, 1, 2], "mean_cor12": [0.5, 0.2, 0.3],
        "total_weight": [1.5, 0.2, 0.6], "within_disagreement": [0.4, np.nan, 0.2],
    })
    # our features keyed by (organism='beril_'+orgId, locus_tag=locusId)
    orth = pd.DataFrame({
        "organism": ["beril_ANA3", "beril_ANA3", "beril_MR1"],
        "locus_tag": ["11530891", "11530892", "999"],
        "family_frac_essential_leakfree": [0.9, 0.1, 0.7],
        "family_n_organisms": [40, 5, 30],
        "n_paralogs_in_genome": [0, 1, 0], "is_orphan": [0, 0, 0]})
    codon = pd.DataFrame({
        "organism": ["beril_ANA3", "beril_ANA3", "beril_MR1"],
        "locus_tag": ["11530891", "11530892", "999"],
        "cai": [0.7, 0.4, 0.6], "gc": [0.5, 0.5, 0.5],
        "gc3": [0.6, 0.6, 0.6], "cds_length": [900, 600, 1200]})
    clades = pd.DataFrame({"organism": ["beril_ANA3", "beril_MR1"],
                           "clade": ["shewanella", "shewanella"]})

    df = assemble(consensus, orth, codon, None, None, None, clades)
    print(df[["organism", "locus_tag", "condition_cluster", "consensus_fitness",
              "family_frac", "prevalence_og", "cai", "clade",
              "total_weight"]].to_string(index=False))
    assert df.family_frac.notna().all(), "join lost rows -- bridge broken"
    assert (df.organism == "beril_" + consensus.organism.values).all()
    assert df.clade.notna().all(), "clade join failed"
    assert abs(df.prevalence_og.iloc[0] - 40/48) < 1e-9
    print("\n=== STEP 2a SMOKE PASS ===")
    print("  beril_-prefix + locusId->locus_tag bridge joins cleanly;")
    print("  conservation + codon + clade attached. Run --real on Colab.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--consensus", default=None,
                    help="path to ces_consensus.parquet (default: auto-find)")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
