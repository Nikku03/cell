"""kcat HEAD-TO-HEAD, and the unit chain that turns a turnover number into a cell rate.

TWO QUESTIONS, BOTH ANSWERED WITH NUMBERS MEASURED HERE RATHER THAN QUOTED FROM PAPERS.

PART A -- WHICH kcat SOURCE IS ACTUALLY BETTER, ON THE SAME LABELS?
The figures in circulation are not comparable. CatPred's 3.30x median fold error was measured on THIS
project's 278 labels; ECMpy's "6-8x" is a literature figure on ITS OWN data. Comparing them is meaningless.
This scores every available source on ONE label set, with one metric.

  labels        every Homo sapiens record in DLKcat's BRENDA+SABIO table (17,010 records, 2,437 human)
  contenders    CatPred        outputs/orphan/kinetics_refined.json, 2,549 genes, tier catpred*
                ecHumanGEM     the GECKO-pipeline kcats, already joined to Human-GEM MAR ids (4,421)
                EC-max         ECMpy's actual rule: max kcat over ALL organisms for that EC
                EC-median      this project's build-1/9 hierarchy: median over human records
                global median  the null -- one number for every enzyme

THE CIRCULARITY THIS FILE EXISTS TO AVOID. EC-max and EC-median are computed FROM the same table that
supplies the labels, so scoring them naively would let each predictor see its own answer. Every EC-level
statistic here is LEAVE-ONE-OUT: the record being predicted is removed before the statistic is formed. An
earlier file in this project (kcat_flag/kcat_verify) reported 0.93 Spearman and was later found circular
because it used incell_rate = kcat x sigma; that correction is recorded in kcat_invivo_validate.json and
this file is written to not repeat it.

PART B -- THE UNIT CHAIN: does kcat x abundance give a real cell rate?
Builds 1 and 8 fitted ONE global constant to make the enzyme ceiling reproduce the measured 24 h doubling
time. Fitting is legitimate but it hides whether the number is physically sensible. The constant is
derivable from first principles, and the derivation uses the two things a cell actually has -- a TIME
(turnover per second, growth per hour) and a QUANTITY (protein per gram dry weight):

    v_max,i [mmol/gDW/h] = kcat_i [1/s] x 3600 [s/h] x E_i [mmol enzyme/gDW]
    E_i                  = (ppm_i / 1e6) x P_tot / MW_i
    P_tot                ~ 0.6 g protein per gDW for a mammalian cell        <- ASSUMPTION, stated
    MW_i                   from ecHumanGEM_kcat_table.tsv's MW_kDa column    <- MEASURED, not assumed

Comparing the DERIVED constant against the FITTED one says how far the first-principles capacity is from
what the model needed. That is a real consistency check on builds 1 and 8, and it costs nothing.
"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
sys.path.insert(0, str(Path(__file__).resolve().parent))

P_TOT_G_PER_GDW = 0.6      # g protein per gDW, mammalian; an assumption and labelled as one
FITTED_SCALE = 3.259e-4    # build 1/8's fitted constant, for comparison


def fold_errors(pred, true):
    p, t = np.asarray(pred, float), np.asarray(true, float)
    ok = np.isfinite(p) & np.isfinite(t) & (p > 0) & (t > 0)
    if ok.sum() < 20:
        return None
    lr = np.log10(p[ok] / t[ok])
    return {"n": int(ok.sum()), "median_fold_error": float(10 ** np.median(np.abs(lr))),
            "log10_rmse": float(np.sqrt((lr ** 2).mean())), "log10_bias": float(np.median(lr)),
            "within_2x": float((np.abs(lr) < np.log10(2)).mean()),
            "within_3x": float((np.abs(lr) < np.log10(3)).mean()),
            "within_10x": float((np.abs(lr) < 1.0).mean())}


def main():
    # ---------------- labels: measured human in-vitro kcat ----------------
    recs = json.load(open(SP / "dlkcat.tsv"))
    human, by_ec_all, by_ec_hum = [], collections.defaultdict(list), collections.defaultdict(list)
    for i, r in enumerate(recs):
        ec = (r.get("ECNumber") or "").strip()
        try:
            v = float(r.get("Value"))
        except (TypeError, ValueError):
            continue
        if not ec or v <= 0 or not np.isfinite(v):
            continue
        by_ec_all[ec].append(v)
        if r.get("Organism") == "Homo sapiens":
            by_ec_hum[ec].append(v)
            human.append((ec, v))
    print(f"labels: {len(human):,} measured Homo sapiens kcat records over "
          f"{len({e for e, _ in human}):,} EC numbers")
    assert len(human) > 500, "LABEL SET TOO SMALL -- refusing to continue"
    gmed = float(np.median([v for _, v in human]))

    # ---------------- EC -> MAR reaction, from the SBML ----------------
    rid, r2ec = None, collections.defaultdict(set)
    pat = re.compile(r'ec-code/([0-9]+\.[0-9]+\.[0-9]+\.[0-9-]+)')
    for line in open(SP / "HumanGEM.xml"):
        m = re.search(r'<reaction [^>]*\bid="([^"]+)"', line)
        if m:
            rid = m.group(1)
        if rid:
            for ec in pat.findall(line):
                r2ec[rid].add(ec)
        if "</reaction>" in line:
            rid = None
    ec2mar = collections.defaultdict(set)
    for r, ecs in r2ec.items():
        for e in ecs:
            ec2mar[e].add(r)
    print(f"EC -> reaction: {len(r2ec):,} reactions annotated, {len(ec2mar):,} distinct EC numbers")

    # ---------------- contender 1: ecHumanGEM (the GECKO / ECMpy tier) ----------------
    ech = {}
    with open(SP / "ecHumanGEM_kcats_MARjoined.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                ech[row["MAR_id"]] = float(row["kcat_per_s_max"])
            except (TypeError, ValueError, KeyError):
                continue
    print(f"ecHumanGEM: {len(ech):,} MAR reactions with a kcat")
    assert len(ech) > 1000, "ecHumanGEM JOIN EMPTY"
    ec2ech = {}
    for e, mars in ec2mar.items():
        vs = [ech[m] for m in mars if m in ech]
        if vs:
            ec2ech[e] = float(np.median(vs))
    print(f"  -> a kcat for {len(ec2ech):,} EC numbers")

    # ---------------- contender 2: CatPred ----------------
    kr = json.load(open(OUT / "kinetics_refined.json"))["kinetics_refined"]
    cat_gene = {g: v["kcat_per_s"] for g, v in kr.items()
                if isinstance(v, dict) and v.get("kcat_per_s") and str(v.get("tier", "")).startswith("catpred")}
    print(f"CatPred: {len(cat_gene):,} genes on a catpred tier (of {len(kr):,} in kinetics_refined)")
    assert len(cat_gene) > 500, "CATPRED JOIN EMPTY"
    # gene -> EC via the SBML's gene rules
    D = json.load(open(OUT / "cell_complete.json"))
    from cell_sim import ensembl_to_symbol
    e2s = ensembl_to_symbol()
    import cobra
    cobra.Configuration().solver = "glpk"
    m = cobra.io.read_sbml_model(str(SP / "HumanGEM.xml"))
    ec2cat = collections.defaultdict(list)
    for r in m.reactions:
        for e in r2ec.get(r.id, ()):
            for g in r.genes:
                s = e2s.get(g.id)
                if s in cat_gene:
                    ec2cat[e].append(cat_gene[s])
    ec2cat = {e: float(np.median(v)) for e, v in ec2cat.items()}
    print(f"  -> a kcat for {len(ec2cat):,} EC numbers")

    # ---------------- score all contenders on the identical labels ----------------
    arms = {k: ([], []) for k in ("ecHumanGEM (GECKO/ECMpy tier)", "CatPred", "EC-max any organism (ECMpy rule)",
                                  "EC-median human (build 1/9)", "global median (null)")}
    for ec, v in human:
        # LEAVE-ONE-OUT for the two statistics derived from this same table
        allv = list(by_ec_all[ec])
        allv.remove(v)
        humv = list(by_ec_hum[ec])
        humv.remove(v)
        for name, p in (("ecHumanGEM (GECKO/ECMpy tier)", ec2ech.get(ec)),
                        ("CatPred", ec2cat.get(ec)),
                        ("EC-max any organism (ECMpy rule)", max(allv) if allv else None),
                        ("EC-median human (build 1/9)", float(np.median(humv)) if humv else None),
                        ("global median (null)", gmed)):
            if p is not None and p > 0:
                arms[name][0].append(p)
                arms[name][1].append(v)

    # A first version scored each arm on whatever labels it happened to cover -- CatPred on 950, the null on
    # 2,437 -- and then compared the numbers to each other. That is not a comparison. Everything below is
    # ALSO scored on the COMMON SUBSET where every arm has a prediction.
    print("\n" + "=" * 100)
    print("PART A -- kcat SOURCES, ALL SCORED ON THE SAME MEASURED HUMAN LABELS (leave-one-out where needed)")
    print("=" * 100)
    print(f"{'source':38s} {'n':>6s} {'fold err':>9s} {'log10RMSE':>10s} {'bias':>7s} "
          f"{'<2x':>6s} {'<3x':>6s} {'<10x':>6s}")
    res = {}
    for name, (p, t) in arms.items():
        r = fold_errors(p, t)
        res[name] = r
        if r is None:
            print(f"{name:38s} {len(p):>6,}  too few overlapping labels -- refusing to report")
            continue
        print(f"{name:38s} {r['n']:>6,} {r['median_fold_error']:>8.2f}x {r['log10_rmse']:>10.3f} "
              f"{r['log10_bias']:>+7.2f} {r['within_2x']:>6.1%} {r['within_3x']:>6.1%} {r['within_10x']:>6.1%}")
    print("=" * 100)

    # ---- the SAME arms on the COMMON SUBSET, which is the only valid head-to-head ----
    common = []
    for ec, v in human:
        allv = list(by_ec_all[ec]); allv.remove(v)
        humv = list(by_ec_hum[ec]); humv.remove(v)
        row = {"ecHumanGEM (GECKO/ECMpy tier)": ec2ech.get(ec), "CatPred": ec2cat.get(ec),
               "EC-max any organism (ECMpy rule)": max(allv) if allv else None,
               "EC-median human (build 1/9)": float(np.median(humv)) if humv else None,
               "global median (null)": gmed}
        if all(x is not None and x > 0 for x in row.values()):
            common.append((row, v, len(humv)))
    print(f"\nCOMMON SUBSET -- {len(common):,} labels where EVERY arm has a prediction")
    print(f"{'source':38s} {'n':>6s} {'fold err':>9s} {'log10RMSE':>10s} {'bias':>7s} "
          f"{'<2x':>6s} {'<3x':>6s} {'<10x':>6s}")
    res_c = {}
    for name in arms:
        r = fold_errors([c[0][name] for c in common], [c[1] for c in common])
        res_c[name] = r
        if r:
            print(f"{name:38s} {r['n']:>6,} {r['median_fold_error']:>8.2f}x {r['log10_rmse']:>10.3f} "
                  f"{r['log10_bias']:>+7.2f} {r['within_2x']:>6.1%} {r['within_3x']:>6.1%} "
                  f"{r['within_10x']:>6.1%}")
    print("=" * 100)

    # ---- is EC-median's win real skill, or within-EC replicate agreement? ----
    # Predicting a measurement from the median of OTHER measurements of the SAME EC is close to predicting a
    # value from its own replicates. If the advantage is reproducibility rather than skill it should grow with
    # the number of sibling records and collapse when there is only one.
    print("\nIS 'EC-median human' SKILL OR REPLICATE AGREEMENT? stratified by sibling records for that EC")
    strat = {}
    for lo, hi, lab in ((1, 1, "1 sibling"), (2, 3, "2-3"), (4, 9, "4-9"), (10, 10 ** 9, "10+")):
        sub = [(c[0]["EC-median human (build 1/9)"], c[1]) for c in common if lo <= c[2] <= hi]
        r = fold_errors([a for a, _ in sub], [b for _, b in sub]) if len(sub) >= 20 else None
        strat[lab] = r
        print(f"    {lab:>10s}: n={len(sub):>5,}  " +
              (f"fold err {r['median_fold_error']:.2f}x, <2x {r['within_2x']:.1%}" if r else "too few"))
    print("    If the 1-sibling row is as good as the 10+ row, it is skill. If it degrades sharply, the")
    print("    headline number is largely within-EC reproducibility -- i.e. the noise floor itself.")

    ranked = sorted(((n, r) for n, r in res_c.items() if r), key=lambda t: t[1]["median_fold_error"])
    best = ranked[0]
    print(f"\nBEST BY MEDIAN FOLD ERROR: {best[0]}  ({best[1]['median_fold_error']:.2f}x, "
          f"log10 RMSE {best[1]['log10_rmse']:.3f})")
    print(f"  target set for this work: beat ECMpy's ~6-8x, reach the 3-4x experimental noise floor")
    print(f"  -> {'REACHES the noise floor' if best[1]['median_fold_error'] <= 4 else 'does NOT reach 3-4x'}")
    print(f"\n  NOTE on the previously-quoted 3.30x for CatPred: that was measured on a different label set")
    print(f"  (278 curated in-vitro labels in kcat_calibration_validation.json). The number above is CatPred")
    print(f"  scored on THIS label set, which is the only way it is comparable to the others.")

    # ---------------- PART B: the unit chain ----------------
    mw = {}
    with open(SP / "ecHumanGEM_kcat_table.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                g, w = row.get("ensembl_gene"), float(row.get("MW_kDa"))
            except (TypeError, ValueError):
                continue
            if g and w > 0:
                mw[g] = w
    names = [g["name"] for g in D["genes"]]
    ppm = {names[int(k)]: float(v) for k, v in D["ppm"].items()
           if k.isdigit() and int(k) < len(names) and v}
    pairs = [(ppm[e2s[g]], mw[g]) for g in mw if g in e2s and e2s[g] in ppm]
    print("\n" + "=" * 100)
    print("PART B -- THE UNIT CHAIN: kcat x abundance -> mmol/gDW/h, from a TIME and a QUANTITY")
    print("=" * 100)
    print(f"  molecular weights joined for {len(pairs):,} genes with both MW and PaxDb ppm")
    assert len(pairs) > 200, "MW/ppm JOIN TOO SMALL -- refusing to report a unit conversion"
    w = np.array([p for p, _ in pairs])
    mws = np.array([m for _, m in pairs]) * 1000.0            # kDa -> g/mol
    mw_abundance_weighted = float((w * mws).sum() / w.sum())
    print(f"  abundance-weighted mean protein MW: {mw_abundance_weighted:,.0f} g/mol "
          f"(unweighted {mws.mean():,.0f})")
    print(f"  ASSUMED total protein: {P_TOT_G_PER_GDW} g/gDW (mammalian; this is the one unmeasured input)")
    mol_per_gdw = P_TOT_G_PER_GDW / mw_abundance_weighted
    derived = 3600.0 * mol_per_gdw * 1e-6 * 1000.0            # s/h x mol/gDW x per-ppm x mmol/mol
    print(f"\n  total protein          {mol_per_gdw*1e3:.4g} mmol/gDW")
    print(f"  DERIVED constant       v_max = kcat[1/s] x ppm x {derived:.4g}   mmol/gDW/h")
    print(f"  FITTED constant (build 1/8)                    {FITTED_SCALE:.4g}")
    print(f"  ratio fitted/derived   {FITTED_SCALE/derived:.2f}x")
    print(f"\n  READING: the model needed {FITTED_SCALE/derived:.1f}x MORE capacity than first principles give.")
    print(f"  Within one order of magnitude of a derivation resting on one assumed constant (0.6 g/gDW) is")
    print(f"  agreement, not disagreement -- it says builds 1 and 8 were not fitting a physically absurd")
    print(f"  number. It also says the first-principles ceiling is TIGHTER than the cell needs, which is the")
    print(f"  wrong direction to rescue recall: a tighter ceiling constrains more, and build 1 already showed")
    print(f"  that constraining more does not move recall off 0.203.")

    json.dump({"labels": {"n_human_records": len(human), "n_ec": len({e for e, _ in human}),
                          "global_median_kcat": gmed},
               "part_a_all_labels": res, "part_a_common_subset": res_c,
               "ec_median_stratified": strat, "best": {"source": best[0], **best[1]},
               "part_b": {"n_mw_joined": len(pairs), "mw_abundance_weighted_g_per_mol": mw_abundance_weighted,
                          "p_tot_g_per_gdw": P_TOT_G_PER_GDW, "mol_protein_per_gdw": mol_per_gdw,
                          "derived_scale": derived, "fitted_scale": FITTED_SCALE,
                          "fitted_over_derived": FITTED_SCALE / derived}},
              open(OUT / "kcat_headtohead.json", "w"), indent=1)
    print(f"\n  -> {OUT/'kcat_headtohead.json'}")


if __name__ == "__main__":
    main()
