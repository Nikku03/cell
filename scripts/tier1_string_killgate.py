"""TIER 1.3 -- STRING centrality (degree) kill-gate.

The review predicted: STRING degree probably correlates with family_frac
strongly (conserved essential genes are also hubs), so the error-overlap
will be >2x and the cascade gain will be near-zero. Build the CHEAP test
first: degree only (pandas edge-count), test error-overlap vs family_frac.
If overlap > ~1.5x, drop without building betweenness.

Input (real mode):
  STRING per-organism file 'protein.links.detailed.v12.0.txt' (downloaded
  on Colab; ~10 MB per org) + a mapping STRING-ID -> our locus_tag (via
  protein.info.v12.0.txt which has the alias column).

For each organism in --orgs:
  1. parse links file -> degree per protein
  2. join to our labels via locus_tag
  3. compare to family_frac as predictors of essentiality
  4. error-overlap test:
       conservation_err & string_err / expected_independent
       <1.5x = streams err on DIFFERENT genes (good, add it)
       >2.0x = streams err together (drop it, no lift possible)

SMOKE MODE (--smoke):
  builds a mock STRING-like edge list + mock labels + family_frac in sandbox,
  runs the error-overlap calculation, validates the kill-gate logic.

REAL MODE:
  python scripts/tier1_string_killgate.py \\
    --string-dir /content/drive/.../string \\
    --orgs beril_Keio beril_Putida ...
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KILL_THRESHOLD = 1.5    # error-overlap ratio; > this -> drop the stream
DROP_THRESHOLD = 2.0    # > this -> definitely drop


def compute_degree(links_df, score_col="combined_score", min_score=400):
    """Build symmetric degree from a STRING links file."""
    import pandas as pd
    f = links_df[links_df[score_col] >= min_score]
    a = f.groupby("protein1").size()
    b = f.groupby("protein2").size()
    deg = a.add(b, fill_value=0).rename("degree").reset_index()
    deg.columns = ["protein", "degree"]
    return deg


def error_overlap_test(y, stream_a, stream_b):
    """Returns ratio = observed-both-wrong / expected-if-independent.
    <1.5 = streams err on different genes (agreement helps);
    >2.0 = streams err together (agreement buys nothing)."""
    import numpy as np
    a_err = (stream_a != (y == 1))
    b_err = (stream_b != (y == 1))
    n = len(y)
    both_err = int((a_err & b_err).sum())
    exp = (a_err.sum() / n) * (b_err.sum() / n) * n
    ratio = both_err / max(exp, 1e-9)
    return both_err, float(exp), float(ratio)


def killgate_verdict(ratio):
    if ratio > DROP_THRESHOLD:
        return ("DROP", "errors overlap heavily -- redundant with conservation")
    if ratio > KILL_THRESHOLD:
        return ("WEAK", "small overlap reduction -- marginal lift")
    return ("KEEP", "errors uncorrelated -- genuinely independent voter")


def run_smoke():
    import pandas as pd
    import numpy as np
    print("=" * 60)
    print("SMOKE MODE -- STRING degree kill-gate on mock data")
    print("=" * 60)
    rng = np.random.RandomState(0)
    n_genes = 1500
    # essential: 25%
    y = (rng.random(n_genes) < 0.25).astype(int)
    # family_frac: realistic predictor with REAL errors (AUC ~0.80)
    family_frac = np.clip(y * 0.25 + rng.normal(0.25, 0.20, n_genes), 0, 1)
    # build TWO scenarios for degree:
    print(f"\n  SCENARIO A: degree errs in SAME pattern as family_frac (REDUNDANT)")
    # construct degree directly from family_frac + tiny noise -> overlapping errors
    degree_a = (family_frac * 30 + rng.normal(0, 2, n_genes)).astype(int)
    print(f"  SCENARIO B: degree errs INDEPENDENTLY of family_frac (COMPLEMENTARY)")
    # degree predicts y from a different noise source
    degree_b = (y * 15 + rng.normal(5, 8, n_genes)).astype(int)
    # binary predictions at thresholds
    pred_ff = (family_frac >= 0.5).astype(int)
    pred_a = (degree_a >= np.quantile(degree_a, 0.75)).astype(int)
    pred_b = (degree_b >= np.quantile(degree_b, 0.75)).astype(int)
    # run error-overlap
    bo_a, exp_a, r_a = error_overlap_test(y, pred_ff, pred_a)
    bo_b, exp_b, r_b = error_overlap_test(y, pred_ff, pred_b)
    print(f"\n  scenario A (degree~y): both_wrong={bo_a} expected_indep={exp_a:.1f}  "
          f"ratio={r_a:.2f}  -> {killgate_verdict(r_a)}")
    print(f"  scenario B (degree~rand): both_wrong={bo_b} expected_indep={exp_b:.1f}  "
          f"ratio={r_b:.2f}  -> {killgate_verdict(r_b)}")
    # validate kill-gate logic
    if killgate_verdict(r_a)[0] not in ("DROP", "WEAK"):
        print("  FAIL: scenario A should DROP/WEAK"); return 1
    if killgate_verdict(r_b)[0] != "KEEP":
        print("  FAIL: scenario B should KEEP"); return 1
    print(f"\n  PREDICTION for real STRING (review): scenario A is the likely "
          f"outcome -- DROP without building betweenness")
    print(f"\nSMOKE PASS -- kill-gate logic correct, predictions ready")
    return 0


def run_real(args):
    import pandas as pd
    print(f"Loading labels + family_frac ...")
    lab = pd.read_csv(REPO / "data" / "drive_import" / "labels" / "labels.csv")
    orth = pd.read_csv(REPO / "data" / "drive_import" / "labels" / "orthology_features.csv",
                       usecols=["organism", "locus_tag",
                                "family_frac_essential_leakfree"])
    df = lab.merge(orth, on=["organism", "locus_tag"], how="left")

    results = []
    for org in args.orgs:
        links_path = args.string_dir / f"{org}_links.txt"
        info_path = args.string_dir / f"{org}_info.txt"
        if not links_path.exists():
            print(f"  {org}: links file missing at {links_path} -- skip"); continue
        print(f"\n--- {org} ---")
        links = pd.read_csv(links_path, sep=" ")
        info = pd.read_csv(info_path, sep="\t") if info_path.exists() else None
        deg = compute_degree(links)
        # map STRING id -> our locus_tag (info file's preferred_name or alias)
        if info is not None:
            id2locus = dict(zip(info["#string_protein_id"],
                                  info.get("preferred_name", info.iloc[:, 1])))
            deg["locus_tag"] = deg.protein.map(id2locus)
        else:
            deg["locus_tag"] = deg.protein
        sub = df[df.organism == org].merge(deg, on="locus_tag", how="left")
        sub = sub.dropna(subset=["family_frac_essential_leakfree", "degree"])
        if len(sub) < 200:
            print(f"  {org}: only {len(sub)} joined rows, skip"); continue
        y = sub.essential.values.astype(int)
        pred_ff = (sub.family_frac_essential_leakfree.values >= 0.5).astype(int)
        # threshold degree at its own 75th percentile (matches cooccur pattern)
        deg_thr = float(sub.degree.quantile(0.75))
        pred_deg = (sub.degree.values >= deg_thr).astype(int)
        both_err, exp_indep, ratio = error_overlap_test(y, pred_ff, pred_deg)
        verdict = killgate_verdict(ratio)
        from numpy import corrcoef
        corr = float(corrcoef(sub.family_frac_essential_leakfree.values,
                                sub.degree.values)[0, 1])
        print(f"  joined: {len(sub):,}  corr(family_frac, degree)={corr:+.3f}")
        print(f"  error-overlap: both_wrong={both_err} expected={exp_indep:.1f} "
              f"ratio={ratio:.2f} -> {verdict[0]} ({verdict[1]})")
        results.append({"organism": org, "n": len(sub), "corr": corr,
                        "ratio": ratio, "verdict": verdict[0]})

    if not results:
        print("\nNo organisms produced results."); return 1
    print(f"\n=== AGGREGATE VERDICT ===")
    keeps = sum(1 for r in results if r["verdict"] == "KEEP")
    print(f"  KEEP: {keeps}/{len(results)} organisms")
    if keeps < len(results) // 2:
        print(f"  *** STRING degree fails the kill-gate -- DROP from cascade ***")
        print(f"  Don't bother computing betweenness.")
    else:
        print(f"  STRING degree passes -- proceed to betweenness compute")
    out = REPO / "outputs" / "tier1_string_killgate.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--string-dir", type=Path,
                    default=Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/string"),
                    help="directory with <org>_links.txt + <org>_info.txt")
    ap.add_argument("--orgs", nargs="*",
                    default=["beril_Keio", "beril_Putida", "mtub", "beril_BFirm"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    return run_smoke() if args.smoke else run_real(args)


if __name__ == "__main__":
    sys.exit(main())
