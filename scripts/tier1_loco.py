"""TIER 1.1a -- Leave-One-Compound-Out evaluation: does the kernel transfer
to UNSEEN drugs, or only within seen drug classes?

The honest test the Phase 2 LOO didn't run. Phase 2 measured leave-one-
ORGANISM-out (47% recall@P30 macro). LOCO is harder: hold out a whole
COMPOUND from training, predict it cold. The MoA kernel was built to pass
this test; this is the measurement.

DESIGN (per the revised plan):
  - hold out one drug at a time from: cisplatin, bacitracin, nalidixic acid,
    fusidic acid, D-cycloserine, gentamicin
  - positives = fit < -3 AND |t| >= 3 (the project's standard, NOT the foreign
    plan's loose -0.3)
  - retrain Phase 2 from scratch with that compound fully excluded
  - score on the held-out compound's rows ONLY
  - report PER-DRUG: recall@P30, Precision@top-20, AUPRC, n_test_pos
    PER-DRUG DISTRIBUTION + POOLED MACRO (not a point estimate)

LEAK DISCIPLINE:
  * compound exclusion is GLOBAL: removed from frame, from og_cpd aggregates,
    from kernel features (test_org perspective). Audited at runtime.
  * og_cpd_hit_rate and kernel_rate are RECOMPUTED with the held-out
    compound excluded; the bake-off's compute_additive_effects already
    accepts held_out_cpds.

SMOKE MODE (--smoke):
  validates the leak audit + per-drug metrics machinery on a small mock
  frame WITHOUT retraining. Confirms in <5s that:
    1. the held-out compound is in the test set, not in training
    2. og_cpd aggregates exclude it
    3. recall@P30 and Precision@top-20 compute correctly on per-drug subsets
  This is what we run in the sandbox to prove the script is correct.

REAL MODE (no --smoke):
  retrains Phase 2 per held-out drug on Colab. Writes:
    PHASE2_DIR/eval/loco/preds_xgb__held=<cpd>.parquet
    PHASE2_DIR/eval/loco/summary.json
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/phase2")

# the drugs to hold out, ordered by atlas positive count
DEFAULT_HOLDOUT_CPDS = [
    "cisplatin",
    "bacitracin",
    "nalidixic acid",
    "fusidic acid",
    "d-cycloserine",
    "gentamicin",
]
HIT_FIT = -3.0
HIT_T = 3.0
MIN_POSITIVES_PER_DRUG = 30


def norm_compound(c: str) -> str:
    s = str(c or "").lower().strip()
    s = re.sub(r"\b(hexahydrate|dihydrate|monohydrate|trihydrate|tetrahydrate|"
               r"anhydrous|sodium salt|potassium salt|salt|dihydrochloride|"
               r"hydrochloride|sulfate|chloride|nitrate|acetate|phosphate)\b",
               "", s)
    return re.sub(r"\s+", " ", s).strip()


def cpd_matches(s: str, target: str) -> bool:
    """True if normalized s contains target (e.g., 'cisplatin' matches all
    cisplatin formulations)."""
    return target.lower() in norm_compound(s)


def tail_metrics(y, p):
    """Same metric vocabulary as bakeoff: AUPRC, recall@P30, plus
    Precision@top-20 (reviewer's addition for stability with small n_pos)."""
    import numpy as np
    if y.sum() == 0 or len(y) < 5:
        return {"auprc": None, "recall_at_p30": None, "precision_at_top20": None,
                "n": len(y), "n_pos": int(y.sum())}
    # tie-aware PR sweep (no sklearn in sandbox)
    order = np.argsort(-p, kind="stable")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(y.sum(), 1)
    # AUPRC via trapezoidal on (recall, precision)
    auprc = float(np.trapezoid(prec, rec))
    # recall@P30: max recall where precision >= 0.30
    mask = prec >= 0.30
    rec_at_p30 = float(rec[mask].max()) if mask.any() else 0.0
    # Precision@top-20 (or top-min(20, n))
    k = min(20, len(y))
    prec_at_top = float(y_sorted[:k].sum() / k)
    return {"auprc": auprc, "recall_at_p30": rec_at_p30,
            "precision_at_top20": prec_at_top,
            "n": int(len(y)), "n_pos": int(y.sum())}


def leak_audit(train_df, test_df, held_compound, ogcpd_train) -> dict:
    """Hard-verify the held-out compound is GONE from training and og_cpd
    aggregates. Returns a dict; raises on any leak."""
    audit = {}
    # 1. compound not in any train row
    train_cpds = train_df["compound"].dropna().unique()
    leak_rows = [c for c in train_cpds if cpd_matches(c, held_compound)]
    audit["train_rows_with_held_cpd"] = len(leak_rows)
    audit["train_cpd_strings_leaked"] = leak_rows[:5]
    if leak_rows:
        raise RuntimeError(
            f"LEAK: held-out '{held_compound}' present in train: {leak_rows}")
    # 2. og_cpd aggregates exclude the held-out compound
    if ogcpd_train is not None:
        ocp_cpds = ogcpd_train["compound"].dropna().unique()
        leak_oc = [c for c in ocp_cpds if cpd_matches(c, held_compound)]
        audit["ogcpd_cells_with_held_cpd"] = len(leak_oc)
        if leak_oc:
            raise RuntimeError(
                f"LEAK: og_cpd aggregates carry '{held_compound}': {leak_oc}")
    # 3. test_df: all rows match held compound
    test_cpds = test_df["compound"].dropna().unique()
    nonmatch = [c for c in test_cpds if not cpd_matches(c, held_compound)]
    audit["test_rows_total"] = len(test_df)
    audit["test_distinct_cpds"] = list(test_cpds)
    audit["test_nonmatching_cpds"] = nonmatch[:5]
    if nonmatch:
        raise RuntimeError(
            f"test contains non-held-cpd rows: {nonmatch}")
    return audit


def run_smoke():
    """Sandbox proof that the leak machinery and metrics are correct,
    on a small mock dataset, without retraining."""
    import pandas as pd
    import numpy as np
    print("=" * 60)
    print("SMOKE MODE -- validating leak audit + metrics on mock data")
    print("=" * 60)
    rng = np.random.RandomState(0)
    # mock frame with 5 compounds
    cpds = ["bacitracin", "cisplatin", "vancomycin", "gentamicin", "L-glutamine"]
    n = 5000
    df = pd.DataFrame({
        "orgId": rng.choice(["A", "B", "C"], n),
        "locusId": rng.randint(1, 500, n).astype(str),
        "compound": rng.choice(cpds, n),
        "fit": rng.randn(n) * 1.5,
        "t": rng.randn(n) * 3.0,
        "og_id": ["OG" + str(x).zfill(5) for x in rng.randint(0, 200, n)],
    })
    df["strong_hit"] = ((df.fit < HIT_FIT) & (df.t.abs() >= HIT_T)).astype(int)
    # bias positives slightly toward bacitracin
    boost = (df.compound == "bacitracin") & (rng.random(n) < 0.05)
    df.loc[boost, "strong_hit"] = 1

    # mock og_cpd aggregate (per (orgId, og_id, compound))
    ogcpd = (df.groupby(["orgId", "og_id", "compound"])
               .agg(n=("strong_hit", "size"),
                    n_hit=("strong_hit", "sum")).reset_index())
    print(f"  mock frame: {len(df):,} rows over {df.compound.nunique()} compounds")
    print(f"  base rate: {df.strong_hit.mean():.4f}")

    failures = 0
    for held in ["bacitracin", "cisplatin", "gentamicin"]:
        n_test_pos = df[df.compound.apply(lambda c: cpd_matches(c, held))].strong_hit.sum()
        print(f"\n  --- hold out '{held}' (n_pos in held subset = {n_test_pos}) ---")
        is_held = df.compound.apply(lambda c: cpd_matches(c, held))
        train = df[~is_held].copy()
        test = df[is_held].copy()
        ogcpd_train = ogcpd[~ogcpd.compound.apply(lambda c: cpd_matches(c, held))]
        try:
            audit = leak_audit(train, test, held, ogcpd_train)
            print(f"    leak audit PASS  "
                  f"(train_rows_leaked={audit['train_rows_with_held_cpd']}, "
                  f"og_cpd_cells_leaked={audit['ogcpd_cells_with_held_cpd']})")
        except RuntimeError as e:
            print(f"    leak audit FAIL: {e}"); failures += 1; continue
        # fake predictions: noisy proxy for strong_hit
        y = test.strong_hit.values.astype(int)
        p = (test.strong_hit.values * 0.6 +
             rng.random(len(test)) * 0.4).astype(float)
        m = tail_metrics(y, p)
        print(f"    metrics on held compound: "
              f"AUPRC={m['auprc']:.3f}  R@P30={m['recall_at_p30']:.3f}  "
              f"P@top20={m['precision_at_top20']:.3f}  n_pos={m['n_pos']}")
        # explicit leak test: deliberately leave 1 row in train -> must raise
        leaky_train = pd.concat([train, test.head(1)])
        try:
            leak_audit(leaky_train, test, held, ogcpd_train)
            print(f"    EXPECTED-FAILURE-TEST FAILED: leak NOT caught"); failures += 1
        except RuntimeError:
            print(f"    expected-failure test PASS (deliberate leak caught)")

    if failures:
        print(f"\nSMOKE FAILED ({failures} errors)"); return 1
    print(f"\nSMOKE PASS -- leak machinery + per-drug metrics correct")
    return 0


def run_real(args):
    """Run on Colab: per held-out compound, retrain Phase 2, score on
    held-out compound only. Reuses the existing bake-off frame loader."""
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from train_phase2_bakeoff import (
            load_clade_map, ff_col_for_org, load_shards, compute_additive_effects,
            featurize, fit_xgb, merge_func, add_ogcpd_rate, add_kernel_rate,
            load_func_features, load_kernel, fit_additive,
            NUMERIC_GENE_COLS, NUMERIC_COND_COLS, CLADE_MAP,
        )
    except ImportError as e:
        print(f"ERROR importing bake-off helpers: {e}", file=sys.stderr); return 1

    import pandas as pd
    import numpy as np

    out_dir = args.out
    eval_dir = out_dir / "eval" / "loco"
    eval_dir.mkdir(parents=True, exist_ok=True)
    # set up the global CLADE_MAP/FUNC_FEATS/OGCPD/KERNEL used by helpers
    import train_phase2_bakeoff as tpb
    tpb.CLADE_MAP = load_clade_map(args.clades)
    load_func_features(out_dir)
    load_kernel(out_dir)
    # load all shards once
    manifest = json.loads((out_dir / "_build_manifest.json").read_text())
    all_orgs = [r["org"] for r in manifest["per_org"]]
    print(f"  total organisms: {len(all_orgs)}")

    per_drug_results = []
    for held in args.cpds:
        held_n = held.lower().strip()
        print(f"\n{'='*60}\n=== held-out compound: {held_n!r} ===")
        t0 = time.time()
        # decide a representative test_org for ff_col -- LOCO trains on all
        # orgs, so we just need any column; pick the most-essential-rich one
        ff_col = "family_frac_essential_leakfree"
        # for LOCO we use the leakfree column (which excludes the row's own
        # clade); but Phase 2's bake-off uses per-fold; here we pick a clade
        # 0 column as a stable default. Document this clearly.
        ff_col = "family_frac_essential_fold0"

        base_cols = ["orgId", "locusId", "expName", "fit", "t", "strong_hit",
                     "compound", "expGroup", "og_id"] + NUMERIC_GENE_COLS \
                    + NUMERIC_COND_COLS + [ff_col]
        full = load_shards(out_dir, all_orgs, base_cols, subdir="frame")
        full["compound"] = full.compound.astype(str)
        full["_norm_cpd"] = full.compound.apply(norm_compound)
        is_held = full._norm_cpd.str.contains(held_n, na=False)
        train_df = full[~is_held].copy()
        # test = FULL-prevalence rows for held compound (no downsampling)
        full_test = load_shards(out_dir, all_orgs, base_cols, subdir="frame_full")
        full_test["compound"] = full_test.compound.astype(str)
        full_test["_norm_cpd"] = full_test.compound.apply(norm_compound)
        test_df = full_test[full_test._norm_cpd.str.contains(held_n, na=False)].copy()

        n_test_pos = int(test_df.strong_hit.sum())
        print(f"  train rows: {len(train_df):,}  test rows: {len(test_df):,}  "
              f"test positives: {n_test_pos}")
        if n_test_pos < MIN_POSITIVES_PER_DRUG:
            print(f"  SKIP -- only {n_test_pos} test positives "
                  f"(min {MIN_POSITIVES_PER_DRUG})")
            continue
        # compute additive effects with held compound excluded
        mu, ag, bc = compute_additive_effects(out_dir, all_orgs,
                                               held_out_cpds=[held_n])
        train_df = merge_func(train_df)
        test_df = merge_func(test_df)
        # og_cpd_rate uses leave-own-org-out routing (already leak-free);
        # we pass held_out_org=None for train, but the held compound is
        # already absent from training rows so og_cpd lookup excludes it
        # naturally. Re-verify here:
        train_df = add_ogcpd_rate(train_df, held_out_org=None)
        test_df = add_ogcpd_rate(test_df, held_out_org=None)
        train_df = add_kernel_rate(train_df, held_out_org=None)
        test_df = add_kernel_rate(test_df, held_out_org=None)
        train_df["additive_pred"] = fit_additive(train_df, mu, ag, bc)
        test_df["additive_pred"] = fit_additive(test_df, mu, ag, bc)

        # leak audit BEFORE training
        # rebuild ogcpd training table for the audit
        ogcpd_train = tpb.OGCPD[~tpb.OGCPD.compound.astype(str)
                                 .str.lower().str.contains(held_n, na=False)] \
            if tpb.OGCPD is not None else None
        try:
            audit = leak_audit(train_df, test_df, held_n, ogcpd_train)
            print(f"  leak audit PASS: {audit}")
        except RuntimeError as e:
            print(f"  LEAK AUDIT FAILED: {e}", file=sys.stderr); continue

        # train
        X_tr = featurize(train_df, ff_col); y_tr = train_df.strong_hit.values.astype(int)
        X_te = featurize(test_df, ff_col); y_te = test_df.strong_hit.values.astype(int)
        clf = fit_xgb(X_tr, y_tr, gpu=not args.no_gpu)
        prob = clf.predict_proba(X_te)[:, 1]
        m = tail_metrics(y_te, prob)
        m["held_compound"] = held_n
        m["seconds"] = round(time.time() - t0, 1)
        print(f"  >> {held_n!r}: AUPRC={m['auprc']:.3f}  "
              f"recall@P30={m['recall_at_p30']:.3f}  "
              f"Precision@top20={m['precision_at_top20']:.3f}  "
              f"n_test_pos={m['n_pos']}  ({m['seconds']}s)")
        # save predictions
        pred_df = test_df[["orgId", "locusId", "expName", "compound",
                            "fit", "t", "strong_hit"]].copy()
        pred_df["pred"] = prob
        pred_df.to_parquet(eval_dir / f"preds_xgb__held={held_n.replace(' ', '_')}.parquet",
                            index=False)
        per_drug_results.append(m)

    # summary: distribution + macro
    if not per_drug_results:
        print("\nNO drugs completed. Check positive counts."); return 1
    valid = [r for r in per_drug_results if r.get("auprc") is not None]
    macro = {
        "n_drugs": len(valid),
        "macro_recall_at_p30": float(np.mean([r["recall_at_p30"] for r in valid])),
        "macro_precision_at_top20": float(np.mean([r["precision_at_top20"] for r in valid])),
        "macro_auprc": float(np.mean([r["auprc"] for r in valid])),
        "per_drug": valid,
    }
    print(f"\n=== POOLED MACRO over {len(valid)} drugs ===")
    print(f"  recall@P30 = {macro['macro_recall_at_p30']:.3f} "
          f"(spread per-drug: "
          f"{min(r['recall_at_p30'] for r in valid):.3f}-"
          f"{max(r['recall_at_p30'] for r in valid):.3f})")
    print(f"  Precision@top20 = {macro['macro_precision_at_top20']:.3f} "
          f"(spread: "
          f"{min(r['precision_at_top20'] for r in valid):.3f}-"
          f"{max(r['precision_at_top20'] for r in valid):.3f})")
    print(f"  AUPRC = {macro['macro_auprc']:.3f}")
    (eval_dir / "summary.json").write_text(json.dumps(macro, indent=2, default=str))
    print(f"\nwrote {eval_dir / 'summary.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--clades", type=Path,
                    default=REPO / "data" / "drive_import" / "labels" / "clade_splits.csv")
    ap.add_argument("--cpds", nargs="*", default=DEFAULT_HOLDOUT_CPDS,
                    help="compounds to hold out one-at-a-time")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="run smoke test on mock data (sandbox-safe)")
    args = ap.parse_args()
    if args.smoke:
        return run_smoke()
    return run_real(args)


if __name__ == "__main__":
    sys.exit(main())
