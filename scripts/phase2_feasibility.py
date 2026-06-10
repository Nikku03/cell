"""Phase 2 feasibility — measure the 4 gating numbers BEFORE building any model.

Per PHASE2_DESIGN.md, four numbers decide the architecture and scope. This
script measures them against the unfiltered Fitness Browser data (feba.db):

  Q1  NOISE CEILING (replicate agreement)
      Same (orgId, normalized compound, ~same concentration), different
      experiment. Correlate fit across shared genes. Also: of strong hits
      (fit<-3) in replicate A, what fraction reproduce (<-2) in replicate B?
      -> the UPPER BOUND on any model's achievable accuracy.

  Q2  ADDITIVE BASELINE (the floor to beat)
      Per organism, fit ~ mu + alpha_gene + beta_cond. Variance explained
      and tail-recall of fit<-3. Gap from Q2 to Q1 = interaction headroom.
      Also: does additive recover the bacitracin cluster? (should NOT --
      proves we need an interaction model.)

  Q3  CONDITION-FEATURE TRANSFERABILITY (gates LOCO)
      How many distinct compounds map to a known structure/MoA via the
      Compounds table? How many compounds appear in >=N organisms? If
      conditions are effectively one-hot, leave-one-compound-out is a
      cold-start and impossible -> Phase 2 restricts to LOO-organism.

  Q4  HOLDOUT GROUP STRUCTURE (leakage-safe splitting)
      Count (organism x compound) cells and replicate structure so the
      triple-stratified holdout (seen-cpd/unseen-org, unseen-cpd/seen-org,
      unseen-both) is designed without replicate leakage.

PERFORMANCE NOTE: feba.db is ~7.3 GB on Drive FUSE. Copy it to local Colab
disk first for ~10-50x faster queries:
    !cp /content/drive/.../feba.db /content/feba.db
then run with --db /content/feba.db.

Output: outputs/phase2_feasibility.json + a console verdict.
"""
from __future__ import annotations
import argparse, json, re, sqlite3, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DB = Path("/content/feba.db")  # local copy preferred; fall back to Drive

# the validated phase-1 cluster, as a litmus inside Q2
BACITRACIN_CLUSTER = [
    ("envZ", "PV4",  "5211222"), ("envZ", "SB2B", "6939419"),
    ("ompR", "PV4",  "5211221"), ("ompR", "SB2B", "6939418"),
    ("pspB", "MR1",  "200970"),  ("pspB", "SB2B", "6937067"),
]
# organisms to sample for the expensive replicate/additive computations.
# include the three Shewanella (our litmus) + a spread of others.
SAMPLE_ORGS = ["MR1", "PV4", "SB2B", "Keio", "Putida", "DvH", "Phaeo", "ANA3"]


def norm_compound(c: str) -> str:
    s = str(c or "").lower().strip()
    s = re.sub(r"\b(hexahydrate|dihydrate|monohydrate|trihydrate|tetrahydrate|"
               r"anhydrous|sodium salt|potassium salt|salt|dihydrochloride|"
               r"hydrochloride|sulfate|chloride|nitrate|acetate|phosphate)\b",
               "", s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--orgs", nargs="*", default=SAMPLE_ORGS,
                    help="organisms to sample for Q1/Q2 (heavy queries)")
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs" / "phase2_feasibility.json")
    args = ap.parse_args()

    if not args.db.exists():
        alt = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/"
                   "fitness_browser/feba.db")
        if alt.exists():
            print(f"  --db not found; using Drive copy {alt}")
            print(f"  (TIP: cp it to /content/feba.db for much faster queries)")
            args.db = alt
        else:
            print(f"ERROR: feba.db not at {args.db} or {alt}", file=sys.stderr)
            return 1

    import pandas as pd
    import numpy as np
    con = sqlite3.connect(str(args.db))
    out: dict = {"db": str(args.db), "sample_orgs": args.orgs}

    # ---- schema sanity ----
    cur = con.cursor()
    tabs = {r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    print(f"=== schema: {len(tabs)} tables ===")
    for t in ("Compounds", "MediaComponents"):
        if t in tabs:
            cols = [r[1] for r in cur.execute(f"PRAGMA table_info({t})").fetchall()]
            print(f"  {t}: {cols}")

    # =========================================================
    # Q3 (cheap, run first): condition-feature transferability
    # =========================================================
    print("\n=== Q3: condition-feature transferability (gates LOCO) ===")
    exps = pd.read_sql(
        "SELECT orgId, expName, expGroup, condition_1, units_1, "
        "concentration_1, media, pH, temperature, aerobic FROM Experiment", con)
    exps["compound"] = exps.condition_1.apply(norm_compound)
    real = exps[exps.compound != ""].copy()
    n_compounds = real.compound.nunique()
    cpd_orgs = real.groupby("compound").orgId.nunique()
    print(f"  experiments: {len(exps):,}   with a named compound: {len(real):,}")
    print(f"  distinct compounds (normalized): {n_compounds}")
    print(f"  compounds in >=2 orgs: {(cpd_orgs>=2).sum()}   "
          f">=3 orgs: {(cpd_orgs>=3).sum()}   >=5 orgs: {(cpd_orgs>=5).sum()}")
    print(f"  expGroup vocabulary (coarse MoA proxy): "
          f"{sorted(real.expGroup.dropna().unique())}")

    cpd_mapped = None
    if "Compounds" in tabs:
        comp = pd.read_sql("SELECT * FROM Compounds", con)
        print(f"  Compounds table: {len(comp)} rows, cols={comp.columns.tolist()}")
        # try to estimate how many experiment-compounds map to a structure/id
        key = None
        for c in comp.columns:
            if c.lower() in ("compound", "name", "compoundname"):
                key = c; break
        if key:
            comp["_n"] = comp[key].apply(norm_compound)
            mapped = real.compound.isin(set(comp["_n"]))
            cpd_mapped = float(mapped.mean())
            print(f"  fraction of experiment-compounds found in Compounds "
                  f"table: {cpd_mapped:.2f}")
            id_cols = [c for c in comp.columns
                       if any(k in c.lower() for k in
                              ("cas","pubchem","cid","smiles","inchi","kegg"))]
            print(f"  structure/id columns in Compounds: {id_cols} "
                  f"(-> feed external MoA/fingerprint)")
    out["Q3_condition_features"] = {
        "n_experiments": int(len(exps)),
        "n_with_compound": int(len(real)),
        "n_distinct_compounds": int(n_compounds),
        "compounds_ge2_orgs": int((cpd_orgs >= 2).sum()),
        "compounds_ge3_orgs": int((cpd_orgs >= 3).sum()),
        "compounds_ge5_orgs": int((cpd_orgs >= 5).sum()),
        "expgroups": sorted(real.expGroup.dropna().unique().tolist()),
        "frac_compounds_in_Compounds_table": cpd_mapped,
    }
    loco_ok = (cpd_orgs >= 3).sum() >= 30
    print(f"  -> LOCO feasibility: {'PLAUSIBLE' if loco_ok else 'WEAK'} "
          f"({(cpd_orgs>=3).sum()} compounds in >=3 orgs; "
          f"transfer needs MoA/structure features, not one-hot)")

    # =========================================================
    # Q4: holdout group structure
    # =========================================================
    print("\n=== Q4: holdout group structure (organism x compound) ===")
    cells = real.groupby(["orgId", "compound"]).expName.nunique().reset_index(
        name="n_replicate_exps")
    print(f"  (organism x compound) cells: {len(cells):,}")
    print(f"  cells with >=2 replicate experiments: "
          f"{(cells.n_replicate_exps>=2).sum():,} "
          f"(these are the replicate-leakage hazard -> split by compound)")
    out["Q4_holdout"] = {
        "n_org_compound_cells": int(len(cells)),
        "cells_with_replicates": int((cells.n_replicate_exps >= 2).sum()),
        "n_organisms": int(real.orgId.nunique()),
    }

    # =========================================================
    # Q1: noise ceiling (replicate agreement) — heavy, sampled
    # =========================================================
    print(f"\n=== Q1: noise ceiling (replicate agreement) — orgs={args.orgs} ===")
    rep_corrs, rep_tail_corrs, hit_reprod = [], [], []
    for org in args.orgs:
        e = real[real.orgId == org]
        # replicate pairs: same compound, >=2 experiments
        for cpd, grp in e.groupby("compound"):
            exp_names = grp.expName.tolist()
            if len(exp_names) < 2:
                continue
            # pull fitness for these experiments
            qmarks = ",".join("?" * len(exp_names))
            gf = pd.read_sql(
                f"SELECT locusId, expName, fit FROM GeneFitness "
                f"WHERE orgId=? AND expName IN ({qmarks})",
                con, params=[org] + exp_names)
            if gf.empty:
                continue
            wide = gf.pivot_table(index="locusId", columns="expName",
                                  values="fit")
            cols = wide.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    a, b = wide[cols[i]], wide[cols[j]]
                    m = a.notna() & b.notna()
                    if m.sum() < 100:
                        continue
                    r = float(np.corrcoef(a[m], b[m])[0, 1])
                    rep_corrs.append(r)
                    tail = m & ((a.abs() > 2) | (b.abs() > 2))
                    if tail.sum() >= 30:
                        rt = float(np.corrcoef(a[tail], b[tail])[0, 1])
                        rep_tail_corrs.append(rt)
                    # strong-hit reproducibility
                    strongA = m & (a < -3)
                    if strongA.sum() >= 5:
                        hit_reprod.append(float((b[strongA] < -2).mean()))
    def _summ(xs):
        if not xs: return None
        a = np.array(xs)
        return {"n": len(a), "median": float(np.median(a)),
                "mean": float(a.mean()), "p25": float(np.percentile(a, 25)),
                "p75": float(np.percentile(a, 75))}
    print(f"  replicate pairs found: {len(rep_corrs)}")
    print(f"  full fit corr:  {_summ(rep_corrs)}")
    print(f"  TAIL fit corr:  {_summ(rep_tail_corrs)}  <- the ceiling that matters")
    print(f"  strong-hit reproducibility (frac of fit<-3 that re-hit <-2): "
          f"{_summ(hit_reprod)}")
    out["Q1_noise_ceiling"] = {
        "replicate_fit_corr": _summ(rep_corrs),
        "replicate_tail_corr": _summ(rep_tail_corrs),
        "strong_hit_reproducibility": _summ(hit_reprod),
    }

    # =========================================================
    # Q2: additive baseline + bacitracin litmus — sampled
    # =========================================================
    print(f"\n=== Q2: additive baseline (mu + alpha_gene + beta_cond) ===")
    add_rows = []
    for org in args.orgs:
        e = real[real.orgId == org]
        exp_names = e.expName.tolist()
        if len(exp_names) < 10:
            continue
        qmarks = ",".join("?" * len(exp_names))
        gf = pd.read_sql(
            f"SELECT locusId, expName, fit FROM GeneFitness "
            f"WHERE orgId=? AND expName IN ({qmarks})",
            con, params=[org] + exp_names)
        if gf.empty:
            continue
        gf = gf.merge(e[["expName", "compound"]], on="expName", how="left")
        mu = gf.fit.mean()
        alpha = gf.groupby("locusId").fit.mean() - mu          # gene main-effect
        beta = gf.groupby("expName").fit.mean() - mu           # cond main-effect
        gf["pred_add"] = mu + gf.locusId.map(alpha).fillna(0) \
                            + gf.expName.map(beta).fillna(0)
        gf["resid"] = gf.fit - gf.pred_add
        ss_tot = float(((gf.fit - mu) ** 2).sum())
        ss_res = float((gf.resid ** 2).sum())
        r2_add = 1 - ss_res / ss_tot if ss_tot else float("nan")
        # tail recall of additive
        strong = gf.fit < -3
        add_pred_strong = gf.pred_add < -3
        rec = float((strong & add_pred_strong).sum() / max(strong.sum(), 1))
        add_rows.append({"org": org, "n": len(gf), "r2_additive": r2_add,
                         "strong_n": int(strong.sum()),
                         "additive_tail_recall": rec,
                         "resid_var_frac": ss_res / ss_tot if ss_tot else None})
        print(f"  {org:6s} n={len(gf):>7d}  R2_add={r2_add:+.3f}  "
              f"strong={int(strong.sum()):>5d}  add_tail_recall={rec:.3f}  "
              f"resid(interaction)_var={ss_res/ss_tot if ss_tot else 0:.2f}")
    out["Q2_additive_baseline"] = add_rows
    if add_rows:
        mean_resid = float(np.mean([r["resid_var_frac"] for r in add_rows
                                    if r["resid_var_frac"] is not None]))
        mean_rec = float(np.mean([r["additive_tail_recall"] for r in add_rows]))
        print(f"  MEAN interaction (residual) variance: {mean_resid:.2f}  "
              f"<- the headroom the model fights for")
        print(f"  MEAN additive tail-recall: {mean_rec:.3f}  "
              f"<- the floor the model must beat")
        out["Q2_summary"] = {"mean_residual_var_frac": mean_resid,
                             "mean_additive_tail_recall": mean_rec}

    # bacitracin litmus: additive should MISS it
    print(f"\n  --- bacitracin litmus: does additive predict the cluster? ---")
    lit = []
    for gene, org, locus in BACITRACIN_CLUSTER:
        e = real[real.orgId == org]
        if e.empty: continue
        exp_names = e.expName.tolist()
        qmarks = ",".join("?" * len(exp_names))
        gf = pd.read_sql(
            f"SELECT locusId, expName, fit FROM GeneFitness "
            f"WHERE orgId=? AND expName IN ({qmarks})",
            con, params=[org] + exp_names)
        if gf.empty: continue
        gf = gf.merge(e[["expName", "compound"]], on="expName", how="left")
        mu = gf.fit.mean()
        alpha = gf.groupby("locusId").fit.mean() - mu
        beta = gf.groupby("expName").fit.mean() - mu
        sub = gf[(gf.locusId.astype(str) == locus) &
                 (gf.compound.str.contains("bacitracin", na=False))]
        for _, r in sub.iterrows():
            pred = mu + (alpha.get(r.locusId, 0)) + (beta.get(r.expName, 0))
            lit.append({"gene": gene, "org": org, "actual": float(r.fit),
                        "additive_pred": float(pred),
                        "interaction_missed": float(r.fit - pred)})
    for r in lit:
        print(f"    {r['gene']:5s} {r['org']:5s}  actual={r['actual']:+.2f}  "
              f"additive_pred={r['additive_pred']:+.2f}  "
              f"interaction_missed={r['interaction_missed']:+.2f}")
    out["Q2_bacitracin_litmus"] = lit
    if lit:
        avg_missed = float(np.mean([abs(r["interaction_missed"]) for r in lit]))
        print(f"  -> additive misses the cluster by {avg_missed:.2f} fit-units "
              f"on average. THIS is the interaction signal a model must capture.")

    con.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {args.out}")

    # ---- VERDICT ----
    print("\n" + "=" * 60)
    print("VERDICT (drives PHASE2_DESIGN architecture choice)")
    print("=" * 60)
    tail = out["Q1_noise_ceiling"].get("replicate_tail_corr")
    if tail:
        print(f"  Q1 noise ceiling (tail corr): {tail['median']:.2f} "
              f"-> do NOT chase accuracy above this.")
    if out.get("Q2_summary"):
        print(f"  Q2 interaction headroom: "
              f"{out['Q2_summary']['mean_residual_var_frac']:.2f} of variance "
              f"is interaction; additive tail-recall floor = "
              f"{out['Q2_summary']['mean_additive_tail_recall']:.3f}.")
    print(f"  Q3 LOCO: {'in scope' if loco_ok else 'RESTRICT to LOO-organism'} "
          f"(needs MoA/structure features; check Compounds id columns above).")
    print(f"  Q4: split by COMPOUND not experiment "
          f"({out['Q4_holdout']['cells_with_replicates']} replicate cells are "
          f"the leakage hazard).")
    print("  Next: build training frame on FULL GeneFitness (incl. neutral),")
    print("  then bake-off additive vs XGBoost-concat vs FiLM with tail metrics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
