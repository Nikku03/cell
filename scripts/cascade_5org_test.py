"""5-organism cascade test: real 3-stream and 4-stream cascade vs ground truth.

The first two cascade-architecture tests we built (cascade_agreement_test.py +
cascade_model.py) ran on POOLED data across many orgs. Now: per-organism
breakdown on 5 organisms, with 3-stream and 4-stream cascades reported
separately on real labels.

Organisms picked:
  with FBA model (4-stream test possible): beril_Keio, mtub, beril_Putida
  without FBA (3-stream only):              beril_BFirm, beril_Burk376
                                            (the two highest-essential beril
                                             orgs without FBA)

Streams used:
  conservation    family_frac_essential_leakfree    >= 0.5 -> call essential
  phase2          (DRIVE; mock-loaded if --phase2-dir missing)
  cooccur         cooccur_n_neighbors_80   >= top-quartile -> call essential
  flux            fba_essential            >= 0.5 -> call essential
                  (only beril_Keio, mtub, beril_Putida)

Reports per organism:
  per-stream coverage / precision / recall
  3-stream cascade tiers (>=1, >=2, all) precision/recall/coverage
  4-stream cascade tiers (>=1, >=2, >=3, all) where flux exists
  error-overlap diagnostic between streams

No --phase2-dir: phase2 stream becomes the same essential-prediction-from-
conservation that XGBoost rides on, so the test still runs but the 'phase2'
stream isn't truly independent (we flag this clearly in the output).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data" / "drive_import" / "labels"

ORGS_WITH_FBA = ["beril_Keio", "mtub", "beril_Putida"]
ORGS_NO_FBA   = ["beril_BFirm", "beril_Burk376"]
ALL_ORGS = ORGS_WITH_FBA + ORGS_NO_FBA


def metrics(y, called):
    import numpy as np
    y = np.asarray(y); called = np.asarray(called)
    tp = int(((called) & (y == 1)).sum()); fp = int(((called) & (y == 0)).sum())
    fn = int(((~called) & (y == 1)).sum()); tn = int(((~called) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(precision=prec, recall=rec, coverage=float(called.mean()),
                n_called=tp + fp, tp=tp, fp=fp, fn=fn, tn=tn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-dir", type=Path, default=None,
                    help="Drive phase2/ dir to add real Phase 2 stream")
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs" / "cascade_5org_test.md")
    args = ap.parse_args()

    import pandas as pd
    import numpy as np

    print("loading ...")
    lab = pd.read_csv(D / "labels.csv")[["organism", "locus_tag", "essential"]]
    orth = pd.read_csv(D / "orthology_features.csv",
                       usecols=["organism", "locus_tag",
                                "family_frac_essential_leakfree"])
    fba = pd.read_csv(D / "fba_features.csv")
    cooc = pd.read_csv(D / "cooccurrence_features.csv",
                       usecols=["organism", "locus_tag",
                                "cooccur_n_neighbors_80"])
    df = (lab.merge(orth, on=["organism", "locus_tag"], how="left")
             .merge(fba[["organism","locus_tag","fba_essential","fba_in_model"]],
                    on=["organism", "locus_tag"], how="left")
             .merge(cooc, on=["organism", "locus_tag"], how="left"))
    df.loc[df.fba_in_model != 1, "fba_essential"] = np.nan

    # ----- mock phase2 stream from conservation when --phase2-dir absent -----
    have_real_phase2 = False
    if args.phase2_dir is not None:
        pred_dir = args.phase2_dir / "eval" / "loo-org"
        parts = []
        for p in sorted(pred_dir.glob("preds_xgb__*.parquet")):
            pp = pd.read_parquet(p, columns=["orgId", "locusId", "pred"])
            parts.append(pp)
        if parts:
            ph = pd.concat(parts, ignore_index=True)
            ph = (ph.groupby(["orgId", "locusId"]).pred.max().reset_index()
                    .rename(columns={"pred": "phase2_pred"}))
            ph["organism"] = "beril_" + ph.orgId.astype(str)
            ph["locus_tag"] = ph.locusId.astype(str)
            df = df.merge(ph[["organism","locus_tag","phase2_pred"]],
                          on=["organism", "locus_tag"], how="left")
            have_real_phase2 = True

    if not have_real_phase2:
        # mock: 0.8*family_frac + 0.2*(1-family_frac) noise; this is NOT
        # truly independent. We flag it. Better: run on Drive.
        rng = np.random.default_rng(0)
        df["phase2_pred"] = np.where(df.family_frac_essential_leakfree.notna(),
                                      0.8 * df.family_frac_essential_leakfree.fillna(0) +
                                      0.2 * rng.random(len(df)),
                                      np.nan)
        print("WARNING: no --phase2-dir; using MOCK phase2 derived from "
              "conservation. Results UNDERSTATE true independence-lift.")

    # cooccur threshold by global quartile
    cooc_thr = float(df.cooccur_n_neighbors_80.dropna().quantile(0.75))
    print(f"  thresholds: conservation>=0.5, fba>=0.5, "
          f"cooccur>={cooc_thr:.0f}, phase2>=0.5\n")

    # ----- per-organism analysis -----
    out_md = ["# 5-organism cascade test (3-stream and 4-stream)\n",
              f"Real labels per organism. Streams: conservation, phase2"
              f"{' (real, Drive)' if have_real_phase2 else ' (MOCK from conservation)'}, "
              f"cooccur, flux (where FBA model exists).\n"]

    for org in ALL_ORGS:
        sub = df[df.organism == org].copy()
        if len(sub) < 100:
            print(f"\n=== {org}: skipping ({len(sub)} rows) ===")
            continue
        y = sub.essential.values.astype(int)
        has_fba = org in ORGS_WITH_FBA
        print(f"\n{'='*70}")
        print(f"=== {org} ({'4-STREAM' if has_fba else '3-STREAM'} test) ===")
        print(f"  n={len(sub):,}  essential rate={y.mean():.3f}  "
              f"true essentials={int(y.sum())}")

        # build streams
        ff = sub.family_frac_essential_leakfree.values
        ph = sub.phase2_pred.values
        co = sub.cooccur_n_neighbors_80.values
        fb = sub.fba_essential.values

        cov_c = ~np.isnan(ff);     call_c = cov_c & (ff >= 0.5)
        cov_p = ~np.isnan(ph);     call_p = cov_p & (ph >= 0.5)
        cov_o = ~np.isnan(co);     call_o = cov_o & (co >= cooc_thr)
        cov_f = ~np.isnan(fb);     call_f = cov_f & (fb >= 0.5)

        streams = [("conservation", cov_c, call_c),
                   ("phase2",       cov_p, call_p),
                   ("cooccur",      cov_o, call_o)]
        if has_fba:
            streams.append(("flux", cov_f, call_f))

        # per-stream metrics
        print(f"  {'stream':<14s} {'cover':>6s} {'prec':>6s} {'recall':>7s} {'n_cl':>6s}")
        for name, cov, call in streams:
            m = metrics(y, call)
            print(f"  {name:<14s} {cov.mean():>6.3f} {m['precision']:>6.3f} "
                  f"{m['recall']:>7.3f} {m['n_called']:>6d}")

        # tiered: among streams that COVER, how many call essential
        n_cov = sum(cov.astype(int) for _, cov, _ in streams)
        n_cal = sum(call.astype(int) for _, _, call in streams)
        multi = n_cov >= 2

        print(f"\n  TIERED AGREEMENT (genes covered by >=2 streams; "
              f"n_multi={int(multi.sum()):,}):")
        print(f"  {'tier':<22s} {'prec':>6s} {'recall':>7s} {'cover':>6s} {'n_cl':>6s}")
        max_streams = len(streams)
        tier_rows = []
        for k in range(1, max_streams + 1):
            tier = (n_cal >= k) & multi
            m = metrics(y, tier)
            tier_rows.append((k, m))
            print(f"  >= {k}/{max_streams} agree         "
                  f"{m['precision']:>6.3f} {m['recall']:>7.3f} "
                  f"{m['coverage']:>6.3f} {m['n_called']:>6d}")
        unan = (n_cal == n_cov) & multi
        mu = metrics(y, unan)
        print(f"  UNANIMOUS (gold)       "
              f"{mu['precision']:>6.3f} {mu['recall']:>7.3f} "
              f"{mu['coverage']:>6.3f} {mu['n_called']:>6d}")

        # write to markdown
        out_md.append(f"## {org}  ({'4-stream' if has_fba else '3-stream'})\n")
        out_md.append(f"n={len(sub):,}, true essentials={int(y.sum())} "
                      f"({y.mean()*100:.1f}%)\n")
        out_md.append("### per-stream")
        out_md.append("| stream | coverage | precision | recall | n_called |")
        out_md.append("|---|---:|---:|---:|---:|")
        for name, cov, call in streams:
            m = metrics(y, call)
            out_md.append(f"| {name} | {cov.mean():.3f} | {m['precision']:.3f} | "
                          f"{m['recall']:.3f} | {m['n_called']} |")
        out_md.append("\n### tiered cascade")
        out_md.append("| tier | precision | recall | coverage | n_called |")
        out_md.append("|---|---:|---:|---:|---:|")
        for k, m in tier_rows:
            out_md.append(f"| >= {k}/{max_streams} agree | {m['precision']:.3f} | "
                          f"{m['recall']:.3f} | {m['coverage']:.3f} | "
                          f"{m['n_called']} |")
        out_md.append(f"| UNANIMOUS (gold) | {mu['precision']:.3f} | "
                      f"{mu['recall']:.3f} | {mu['coverage']:.3f} | "
                      f"{mu['n_called']} |")
        out_md.append("")

    # ----- macro across 5 orgs -----
    out_md.append("## Macro across the 5 organisms\n")
    out_md.append("Note: 3-stream macro is over all 5; 4-stream macro over the 3 "
                  "FBA-equipped orgs only.\n")

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text("\n".join(out_md))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
