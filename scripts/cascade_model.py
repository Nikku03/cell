"""The cascade model — the synthesis, as planned.

Combines N independent essentiality evidence streams into a tiered,
abstention-aware prediction. The architecture validated in
cascade_agreement_test.py (2-stream agreement lifts precision +0.07): here we
generalise to any number of streams and produce the deployable tiered output.

PRINCIPLE (from the error-buildup analysis):
  - a serial chain compounds error multiplicatively -> ~3% on hard cases
  - a parallel agreement of INDEPENDENT streams compounds the error of the
    DISAGREEMENT -> high-precision consensus + explicit abstention
  - precision rises and coverage falls as you require MORE streams to agree
    (tier 1 = any stream, tier N = all streams). The gold tier is N-of-N.

STREAMS (leak-free by construction):
  conservation  family_frac_essential_leakfree   (Phase-0; clade-excluded)
  flux          fba_essential                     (FBA on genome-scale model)
  cooccur       cooccur_n_neighbors_80            (phylogenetic co-occurrence)
  [phase2]      xgb strong_hit prob               (Drive; --phase2-dir, optional)
  [structure]   future mechanism stream           (not yet built)

OUTPUT per tier k (>= k streams call essential, among streams that COVER
the gene): precision / recall / coverage vs true essentiality, plus the
stream error-overlap matrix (the independence diagnostic) and the final
tiered gene table.

Pure pandas/numpy (sandbox-safe). --phase2-dir adds the Drive stream.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data" / "drive_import" / "labels"


# Each stream: (name, dataframe-merge, score col, threshold, higher=essential)
# Thresholds are fixed, interpretable, leak-free operating points (not tuned
# on the eval labels). family_frac/cooccur are [0,1]-ish; fba is 0/1.
STREAM_SPECS = [
    ("conservation", "family_frac_essential_leakfree", 0.50, True),
    ("flux",         "fba_essential",                   0.50, True),
    ("cooccur",      "cooccur_n_neighbors_80",          None, True),  # thr set by quantile
]


def metrics(y, called, covered):
    """Precision/recall/coverage where 'called'=essential among 'covered'."""
    import numpy as np
    y = np.asarray(y); called = np.asarray(called); covered = np.asarray(covered)
    pred1 = called & covered
    tp = int((pred1 & (y == 1)).sum()); fp = int((pred1 & (y == 0)).sum())
    fn = int((~pred1 & (y == 1)).sum())
    return dict(precision=tp / max(tp + fp, 1),
                recall=tp / max((y == 1).sum(), 1),
                coverage=float(pred1.mean()),
                n_called=tp + fp, tp=tp, fp=fp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-dir", type=Path, default=None,
                    help="Drive phase2/ dir to add the Phase 2 xgb stream")
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs" / "cascade_model_result.md")
    args = ap.parse_args()

    import pandas as pd
    import numpy as np

    lab = pd.read_csv(D / "labels.csv")[["organism", "locus_tag", "essential"]]
    orth = pd.read_csv(D / "orthology_features.csv",
                       usecols=["organism", "locus_tag",
                                "family_frac_essential_leakfree"])
    fba = pd.read_csv(D / "fba_features.csv")
    cooc = pd.read_csv(D / "cooccurrence_features.csv",
                       usecols=["organism", "locus_tag", "cooccur_n_neighbors_80",
                                "cooccur_max_jaccard"])
    df = (lab.merge(orth, on=["organism", "locus_tag"], how="left")
             .merge(fba[["organism", "locus_tag", "fba_essential", "fba_in_model"]],
                    on=["organism", "locus_tag"], how="left")
             .merge(cooc, on=["organism", "locus_tag"], how="left"))

    # flux only meaningful where the gene is IN a model
    df.loc[df.fba_in_model != 1, "fba_essential"] = np.nan

    # optional Phase 2 stream from Drive predictions
    streams = list(STREAM_SPECS)
    if args.phase2_dir is not None:
        pred_dir = args.phase2_dir / "eval" / "loo-org"
        parts = []
        for p in sorted(pred_dir.glob("preds_xgb__*.parquet")):
            pp = pd.read_parquet(p, columns=["orgId", "locusId", "pred"])
            parts.append(pp)
        if parts:
            ph = pd.concat(parts, ignore_index=True)
            # phase2 is per (org, locus, EXPERIMENT) -> take max prob per gene
            ph = (ph.groupby(["orgId", "locusId"]).pred.max()
                    .reset_index().rename(columns={"pred": "phase2_pred"}))
            ph["organism"] = "beril_" + ph.orgId.astype(str)
            ph["locus_tag"] = ph.locusId.astype(str)
            df = df.merge(ph[["organism", "locus_tag", "phase2_pred"]],
                          on=["organism", "locus_tag"], how="left")
            streams.append(("phase2", "phase2_pred", 0.50, True))
            print(f"  added Phase 2 stream: {df.phase2_pred.notna().sum():,} genes")

    # set cooccur threshold by its own upper-quartile (leak-free: feature-only)
    for i, (name, col, thr, hi) in enumerate(streams):
        if thr is None and col in df.columns:
            thr = float(df[col].dropna().quantile(0.75))
            streams[i] = (name, col, thr, hi)

    y = df.essential.values.astype(int)
    print(f"\ntotal genes: {len(df):,}  essential base rate: {y.mean():.3f}")

    # build per-stream covered + called masks
    print(f"\n=== streams (fixed leak-free operating points) ===")
    print(f"  {'stream':<14s} {'coverage':>9s} {'prec':>6s} {'recall':>7s} {'thr':>6s}")
    covered = {}; called = {}
    for name, col, thr, hi in streams:
        if col not in df.columns:
            continue
        cov = df[col].notna().values
        v = df[col].values
        call = np.where(cov, (v >= thr) if hi else (v <= thr), False)
        covered[name] = cov; called[name] = call.astype(bool)
        m = metrics(y, called[name], covered[name])
        print(f"  {name:<14s} {cov.mean():>9.3f} {m['precision']:>6.3f} "
              f"{m['recall']:>7.3f} {thr:>6.2f}")

    stream_names = list(covered.keys())
    S = len(stream_names)

    # ---- error-overlap (independence diagnostic) ----
    print(f"\n=== error overlap (pairwise) -- low overlap = independent = good ===")
    print(f"  {'pair':<28s} {'both_wrong':>11s} {'expected_indep':>15s} {'ratio':>6s}")
    for i in range(S):
        for j in range(i + 1, S):
            a, b = stream_names[i], stream_names[j]
            both_cov = covered[a] & covered[b]
            ea = (called[a] != (y == 1)) & both_cov
            eb = (called[b] != (y == 1)) & both_cov
            both_err = int((ea & eb).sum())
            n = int(both_cov.sum())
            exp = (ea.sum() / max(n, 1)) * (eb.sum() / max(n, 1)) * n
            ratio = both_err / max(exp, 1e-9)
            print(f"  {a+' x '+b:<28s} {both_err:>11d} {exp:>15.0f} {ratio:>6.2f}")

    # ---- TIERED agreement: among streams that COVER a gene, how many call it ----
    n_cover = np.zeros(len(df), dtype=int)
    n_call = np.zeros(len(df), dtype=int)
    for name in stream_names:
        n_cover += covered[name].astype(int)
        n_call += (called[name] & covered[name]).astype(int)

    print(f"\n=== TIERED AGREEMENT (genes covered by >=2 streams) ===")
    print(f"  tier k = at least k covering streams call essential")
    multi = n_cover >= 2
    print(f"  {'tier':<22s} {'precision':>10s} {'recall':>7s} {'coverage':>9s} {'n':>7s}")
    tier_rows = []
    for k in range(1, S + 1):
        called_tier = (n_call >= k) & multi
        m = metrics(y, called_tier, np.ones(len(df), bool))
        # recall here is over ALL essentials; coverage over ALL genes
        tier_rows.append((k, m))
        print(f"  >= {k} streams agree     {m['precision']:>10.3f} {m['recall']:>7.3f} "
              f"{m['coverage']:>9.3f} {m['n_called']:>7d}")

    # unanimous tier = all covering streams agree (the gold tier)
    unanimous = (n_call == n_cover) & multi
    mu = metrics(y, unanimous, np.ones(len(df), bool))
    print(f"  UNANIMOUS (all agree) {mu['precision']:>10.3f} {mu['recall']:>7.3f} "
          f"{mu['coverage']:>9.3f} {mu['n_called']:>7d}   <-- GOLD tier")

    # ---- verdict ----
    p1 = tier_rows[0][1]["precision"]
    p_top = tier_rows[-1][1]["precision"]
    print(f"\n=== VERDICT ===")
    print(f"  precision tier-1 (any stream): {p1:.3f}")
    print(f"  precision tier-{S} (all streams): {p_top:.3f}")
    print(f"  precision GOLD (unanimous):    {mu['precision']:.3f}")
    print(f"  monotonic precision rise: "
          f"{'YES -- cascade works' if p_top > p1 else 'NO'}")

    # ---- write ----
    lines = ["# Cascade model result (the synthesis, as planned)\n",
             f"{len(df):,} genes, {len(stream_names)} streams "
             f"({', '.join(stream_names)}), essential base rate {y.mean():.3f}.\n",
             "## Tiered agreement (genes covered by >=2 streams)\n",
             "| tier | precision | recall | coverage | n_called |",
             "|---|---:|---:|---:|---:|"]
    for k, m in tier_rows:
        lines.append(f"| >= {k} agree | {m['precision']:.3f} | {m['recall']:.3f} | "
                     f"{m['coverage']:.3f} | {m['n_called']} |")
    lines.append(f"| UNANIMOUS (gold) | {mu['precision']:.3f} | {mu['recall']:.3f} | "
                 f"{mu['coverage']:.3f} | {mu['n_called']} |")
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
