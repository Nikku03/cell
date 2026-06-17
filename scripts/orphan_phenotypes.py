"""PATH-ORPHAN, STEP 6 -- Price-style specific-phenotype miner.

Replicates Price 2018's main de-orphaning move: for each gene, find conditions
where the knockout causes a strong, specific fitness defect (fit < -3 AND |t|>4,
AND the gene is fitness-NEUTRAL in OTHER conditions). The (gene, condition)
pairs ARE the conditional essentiality we DO have data for -- no prediction.

LEAK NOTE: this is a READ of the perturbation experiment, not a prediction; the
leak rules don't apply. The output is direct experimental evidence per gene.

INPUT (Colab):
  feba.db (7.3 GB on Drive)
  fb_org resolved as args.org.replace('beril_', '') (the FB orgId convention
  we established in orphan_cofit)
OUTPUT:
  outputs/orphan/phenotypes_<org>.parquet
    locus_tag, condition, fit, t, specificity_score, n_conditions

--smoke : validate specific-phenotype filter + specificity score on synthetic
--real  : pull GeneFitness from feba.db, emit specific phenotypes per gene
"""
from __future__ import annotations
import argparse, sys, sqlite3, json, re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "orphan"
FIT_DEFECT = -3.0       # Price 2018 cutoff for "strong fitness defect"
T_DEFECT = 4.0          # |t| cutoff
NEUTRAL_BAND = 1.0      # |fit| < NEUTRAL_BAND elsewhere -> specific


def specific_phenotypes(fits_by_gene):
    """fits_by_gene: dict locusId -> list of (condition, fit, t).
    Returns list of (locusId, condition, fit, t, specificity_score, n_conditions).
    A condition is specific iff fit < FIT_DEFECT AND |t| > T_DEFECT AND the gene
    is fitness-neutral elsewhere (median |fit| < NEUTRAL_BAND across the rest).
    """
    out = []
    for g, rows in fits_by_gene.items():
        n = len(rows)
        if n < 5:           # need a baseline distribution
            continue
        for i, (c, f, t) in enumerate(rows):
            if not (f < FIT_DEFECT and abs(t) > T_DEFECT):
                continue
            others = [abs(rr[1]) for j, rr in enumerate(rows) if j != i]
            med_other = sorted(others)[len(others) // 2] if others else 0
            if med_other > NEUTRAL_BAND:
                continue          # not specific -- the gene hurts in many cond
            # specificity: gap between this hit and the gene's median |fit|
            spec = abs(f) - med_other
            out.append((g, c, f, t, spec, n))
    return out


def run_real(args):
    import pandas as pd
    fb_org = args.fb_org or args.org.replace("beril_", "")
    if not Path(args.feba).exists():
        print(f"ERROR: feba.db not found at {args.feba} (run on Colab)"); return 2
    con = sqlite3.connect(args.feba)
    gf = pd.read_sql(
        "SELECT locusId, expName, fit, t FROM GeneFitness WHERE orgId=?",
        con, params=(fb_org,))
    exp = pd.read_sql(
        "SELECT expName, condition_1 FROM Experiment WHERE orgId=?",
        con, params=(fb_org,))
    con.close()
    print("=" * 70)
    print(f"PHENOTYPES (Price 2018-style) -- {args.org}  (FB orgId {fb_org!r})")
    print("=" * 70)
    print(f"  GeneFitness rows: {len(gf):,}  experiments: {gf.expName.nunique()}")
    if len(gf) == 0:
        print("  no fitness data for this org -- defer"); return 1
    gf["fit"] = pd.to_numeric(gf["fit"], errors="coerce")
    gf["t"] = pd.to_numeric(gf["t"], errors="coerce")
    gf = gf.dropna()
    gf = gf.merge(exp, on="expName", how="left")
    gf["condition_1"] = gf["condition_1"].fillna(gf["expName"])
    fits_by_gene = {}
    for r in gf.itertuples():
        fits_by_gene.setdefault(r.locusId, []).append(
            (str(r.condition_1), float(r.fit), float(r.t)))

    rows = specific_phenotypes(fits_by_gene)
    print(f"  specific (gene,condition) hits: {len(rows):,}  "
          f"across {len({r[0] for r in rows}):,} genes")
    # bridge FB locusId -> our RefSeq locus_tag (reuse the coinherit bridge)
    sys.path.insert(0, str(REPO / "scripts"))
    from orphan_cofit import _locus_bridge, _manifest_acc
    acc = _manifest_acc(args.org)
    lb = _locus_bridge(acc) if acc else {}
    rs_rows = []
    for g, c, f, t, spec, n in rows:
        lt = lb.get(g, g)
        rs_rows.append((lt, c, f, t, round(spec, 2), n))
    df = pd.DataFrame(rs_rows, columns=["locus_tag", "condition", "fit", "t",
                                         "specificity_score", "n_conditions"])
    df = df.sort_values("specificity_score", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"phenotypes_{args.org}.parquet", index=False)
    # summarize: top conditions, top genes
    print("\n  top 8 specific phenotypes:")
    for r in df.head(8).itertuples():
        print(f"    {r.locus_tag:<13} fit={r.fit:>6.2f}  t={r.t:>5.1f}  "
              f"on '{str(r.condition)[:40]}'")
    (OUT / f"phenotypes_{args.org}_summary.json").write_text(json.dumps({
        "org": args.org, "fb_org": fb_org,
        "specific_hits": len(df),
        "unique_genes_with_hit": int(df.locus_tag.nunique()),
        "unique_conditions": int(df.condition.nunique()),
    }, indent=2))
    print(f"\n  wrote phenotypes_{args.org}.parquet + summary")
    return 0


def run_smoke():
    print("=" * 70)
    print("PHENOTYPES SMOKE -- Price-style specific-defect filter")
    print("=" * 70)
    # g1: clear specific defect on one condition (fit -5, t=8), neutral elsewhere
    # g2: chronically sick (fit ~ -4 in many) -> NOT specific
    # g3: weak hit (fit -2.5) -> below cutoff
    # g4: too few conditions -> dropped
    fits = {
        "g1": [("cA", 0.1, 0.5), ("cB", -5.0, 8.0), ("cC", 0.0, 0.2),
               ("cD", -0.2, 0.3), ("cE", 0.1, 0.1), ("cF", -0.1, 0.2)],
        "g2": [("cA", -3.5, 6.0), ("cB", -4.2, 7.0), ("cC", -3.8, 5.5),
               ("cD", -4.0, 6.5), ("cE", -3.6, 5.0)],
        "g3": [("cA", -2.5, 3.5), ("cB", 0.0, 0.2), ("cC", 0.1, 0.1),
               ("cD", 0.0, 0.0), ("cE", -0.2, 0.3)],
        "g4": [("cA", -5.0, 8.0), ("cB", 0.0, 0.2)],  # too few cond
    }
    out = specific_phenotypes(fits)
    print(f"  hits: {len(out)}")
    for r in out: print(f"    {r}")
    assert len(out) == 1, "only g1 should be specific"
    assert out[0][0] == "g1" and out[0][1] == "cB"
    print("\n=== PHENOTYPES SMOKE PASS ===")
    print("  specific-defect filter + neutrality requirement validated.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--org", default="beril_RalstoniaGMI1000")
    ap.add_argument("--fb_org", default="")
    ap.add_argument("--feba", default="/content/feba.db")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
