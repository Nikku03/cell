"""Loop 205. Can physics compute a rate constant accurately enough to be worth computing?

THE QUESTION UNDERNEATH THE WHOLE REM-CELL ARCHITECTURE. Loop 204 measured that answering a normal
question set needs 48,062 of the 56,277 missing links. Whether that matters depends entirely on
whether those links can be COMPUTED, and the ones that decide dynamics are rate constants.

WHY THIS DESIGN, AND NOT A DOCKING RUN. The temptation is to compute some barriers and see. That
would measure one method's accuracy on one enzyme family. The stronger question is what the CEILING
is for any physics method at all, and there is a way to measure it that needs no simulation:

    Transition-state theory is exact: k = (kB.T/h) . exp(-dG_dagger / RT). At 37 C, RT = 0.616
    kcal/mol, so a 1.4 kcal/mol error in the barrier is a factor of TEN in the rate. Every physics
    route to a rate goes through that exponential.

    What physics gives you at best is the CHEMISTRY -- which bond is made or broken, what the
    transition state looks like, what the barrier roughly is. An EC number is exactly that: a
    complete specification of the chemistry being performed. So the accuracy achievable by
    PERFECTLY identifying the chemistry is an UPPER BOUND on any physics method, because a physics
    method that got the chemistry wrong would do worse, and one that got it right cannot beat
    knowing it outright.

    Loops 131-133 already found that sequence adds NOTHING beyond the EC number. That makes the EC
    number the operative ceiling, and it is measurable on data already on this disk.

So this loop asks: given PERFECT knowledge of the chemistry, how well is a rate constant pinned?
If the answer is "not to within an order of magnitude", then no physics method can pin it either,
and a simulator that needs rates has to measure them rather than compute them.

DATA. colab/data/ml/kcat_records.tsv -- 17,004 measured kcat values with an EC number, a substrate,
an organism, a homology cluster_id and a precomputed fold. 3,006 distinct clusters. Splitting by
cluster rather than at random is the loop 156 lesson: a random split lets a near-identical sequence
sit on both sides and inflates everything.

PREDECLARED, BEFORE ANY NUMBER.

  C1 IS THE INSTRUMENT HONEST?
     Gate: PASS iff 17,004 records parse with a finite log10 kcat, the homology clusters and folds
     are present, and NO cluster appears in two folds. A cluster spanning folds means the
     homology-aware split is not homology-aware and every number below is inflated.

  C2 HOW MUCH DOES A RATE VARY WITHIN ONE EC NUMBER?
     The irreducible spread for any method whose output is the chemistry.
     Gate: PASS iff the median within-EC spread of log10 kcat is BELOW 1.0 -- that is, knowing the
     exact reaction pins the rate to within one order of magnitude. A FAIL means the chemistry does
     not determine the rate, and the ceiling is already too low to be useful.

  C3 THE EC-MEDIAN PREDICTOR, HELD OUT BY HOMOLOGY.
     Predict each held-out record with the median log10 kcat of its EC number, computed on the
     training folds only.
     Gate: PASS iff at least 50% of held-out predictions land within 10x of the measured value.
     Ten-fold is the loosest bar under which a simulator's fluxes are still meaningful, and it is
     stated here rather than tuned later.

  C4 IS THE SPREAD REAL BIOLOGY OR MEASUREMENT NOISE?
     If the spread is assay variability it might be beatable; if the same EC in the SAME organism
     still spreads, it is biology and no method removes it.
     Gate: PASS iff the within-EC-within-organism spread is at least half the overall within-EC
     spread. A PASS means the spread is intrinsic. Magnitude comparison, no sign assumed.

  C5 DOES STRUCTURE HELP WHERE IT CAN BE JOINED?
     colab/data/ml/struct_enzymes.npz and elecster_enzymes.npz carry 64 geometry, 17 electrostatic
     and 17 steric descriptors over 2,178 AlphaFold structures, keyed by UniProt accession. The
     kcat table is keyed by sequence index and gene symbol.
     Gate: PASS iff the join covers at least 200 records AND structure beats the EC-median
     predictor. VOID, not FAIL, if the join is too small -- an untestable claim is not a refuted
     one, and this is exactly the guard loop 201's W6 needed and did not have.

  C6 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import csv, json, os, sys, time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates, weakened_by

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
REC = "colab/data/ml/kcat_records.tsv"
OUT = "outputs/loop_physics_ceiling.json"
RT = 1.98720425e-3 * 310.15          # kcal/mol at 37 C

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "physics ceiling"}
    say("=" * 104)
    say("LOOP 205 -- CAN PHYSICS COMPUTE A RATE ACCURATELY ENOUGH TO BE WORTH COMPUTING?")
    say("=" * 104)
    say(f"     RT at 37 C = {RT:.4f} kcal/mol,  so a barrier error of "
        f"{RT*np.log(10):.2f} kcal/mol is a factor of 10 in the rate")

    rows = [r for r in csv.DictReader(open(REC), delimiter="\t")]
    for r in rows:
        r["y"] = float(r["log10_kcat"])
    ok_fin = all(np.isfinite(r["y"]) for r in rows)

    # ------------------------------------------------------------ C1
    say("C1 IS THE INSTRUMENT HONEST?")
    cl_folds = defaultdict(set)
    for r in rows:
        cl_folds[r["cluster_id"]].add(r["fold"])
    leaky = [c for c, f in cl_folds.items() if len(f) > 1]
    say(f"     records {len(rows):,}   finite kcat {ok_fin}   clusters {len(cl_folds):,}   "
        f"folds {sorted({r['fold'] for r in rows})}")
    say(f"     clusters appearing in more than one fold: {len(leaky)}")
    ok1 = (len(rows) == 17004 and ok_fin and not leaky)
    G.add("C1", ok1,
          if_true="C1 PASS -- 17,004 finite records, and no homology cluster spans two folds",
          if_false=lambda: f"C1 FAIL -- {len(rows)} records, finite={ok_fin}, "
                           f"{len(leaky)} clusters span folds")

    # ------------------------------------------------------------ C2
    say("C2 HOW MUCH DOES A RATE VARY WITHIN ONE EC NUMBER?")
    by_ec = defaultdict(list)
    for r in rows:
        if r["ec"]:
            by_ec[r["ec"]].append(r["y"])
    multi = {e: v for e, v in by_ec.items() if len(v) >= 5}
    spreads = np.array([float(np.std(v)) for v in multi.values()])
    iqrs = np.array([float(np.percentile(v, 75) - np.percentile(v, 25)) for v in multi.values()])
    med_sd = float(np.median(spreads))
    say(f"     EC numbers with >=5 measurements  {len(multi):,}   "
        f"records covered {sum(len(v) for v in multi.values()):,}")
    say(f"     within-EC spread of log10 kcat:  median sd {med_sd:.3f}   "
        f"median IQR {np.median(iqrs):.3f}   (log10 units, so 1.0 = one order of magnitude)")
    say(f"     equivalent barrier spread: {med_sd*RT*np.log(10):.2f} kcal/mol")
    say(f"     overall spread across all records: sd {np.std([r['y'] for r in rows]):.3f}")
    G.add("C2", bool(med_sd < 1.0), stat=med_sd, requires=("C1",),
          if_true=lambda: f"C2 PASS -- knowing the exact reaction pins log10 kcat to sd {med_sd:.2f}",
          if_false=lambda: f"C2 FAIL -- the same EC number spans sd {med_sd:.2f} in log10 kcat "
                           f"({med_sd*RT*np.log(10):.2f} kcal/mol of barrier). The chemistry does "
                           f"not determine the rate, so no method that outputs the chemistry can")
    res["within_ec"] = {"n_ec": len(multi), "median_sd": med_sd,
                        "median_iqr": float(np.median(iqrs)),
                        "overall_sd": float(np.std([r["y"] for r in rows])),
                        "barrier_kcal": med_sd * RT * np.log(10)}

    # ------------------------------------------------------------ C3
    say("C3 THE EC-MEDIAN PREDICTOR, HELD OUT BY HOMOLOGY")
    within10, within3, n_pred, errs = 0, 0, 0, []
    global_med = float(np.median([r["y"] for r in rows]))
    for fold in sorted({r["fold"] for r in rows}):
        tr = [r for r in rows if r["fold"] != fold]
        te = [r for r in rows if r["fold"] == fold]
        med = defaultdict(list)
        for r in tr:
            if r["ec"]:
                med[r["ec"]].append(r["y"])
        med = {e: float(np.median(v)) for e, v in med.items()}
        for r in te:
            p = med.get(r["ec"], global_med)
            e = abs(p - r["y"])
            errs.append(e); n_pred += 1
            within10 += e <= 1.0
            within3 += e <= np.log10(3)
    f10, f3 = within10 / n_pred, within3 / n_pred
    errs = np.array(errs)
    say(f"     held-out predictions {n_pred:,}")
    say(f"     within 10x  {within10:,}  = {f10:.4f}")
    say(f"     within  3x  {within3:,}  = {f3:.4f}")
    say(f"     median absolute error {np.median(errs):.3f} log10 = "
        f"{10**np.median(errs):.1f}x   90th percentile {10**np.percentile(errs,90):.0f}x")
    G.add("C3", bool(f10 >= 0.50), stat=f10, requires=("C1",),
          if_true=lambda: f"C3 PASS -- {f10:.1%} of held-out rates land within 10x given perfect "
                          f"knowledge of the chemistry",
          if_false=lambda: f"C3 FAIL -- only {f10:.1%} land within 10x, median error "
                           f"{10**np.median(errs):.1f}x. Perfect chemistry knowledge is the "
                           f"CEILING for any physics method, and it is below the loosest bar "
                           f"under which a simulator's fluxes mean anything")
    res["ec_predictor"] = {"n": n_pred, "within10": f10, "within3": f3,
                           "median_fold_error": float(10 ** np.median(errs)),
                           "p90_fold_error": float(10 ** np.percentile(errs, 90))}

    # ------------------------------------------------------------ C4
    say("C4 IS THE SPREAD REAL BIOLOGY OR MEASUREMENT NOISE?")
    by_eo = defaultdict(list)
    for r in rows:
        if r["ec"] and r["organism"]:
            by_eo[(r["ec"], r["organism"])].append(r["y"])
    eo = {k: v for k, v in by_eo.items() if len(v) >= 5}
    sd_eo = float(np.median([float(np.std(v)) for v in eo.values()])) if eo else float("nan")
    say(f"     EC x organism groups with >=5 measurements  {len(eo):,}")
    say(f"     within EC and ORGANISM   median sd {sd_eo:.3f}")
    say(f"     within EC alone          median sd {med_sd:.3f}")
    ratio = sd_eo / med_sd if med_sd else float("nan")
    say(f"     ratio {ratio:.3f}  -- if the spread were assay noise between species this would "
        f"be small")
    G.add("C4", bool(ratio >= 0.5), stat=ratio, requires=("C2",),
          if_true=lambda: f"C4 PASS -- {ratio:.0%} of the spread survives fixing the organism, so "
                          f"it is intrinsic to the enzyme and no method removes it",
          if_false=lambda: f"C4 FAIL -- only {ratio:.0%} survives, so much of the spread is "
                           f"cross-species variation a method could in principle model")
    res["organism_control"] = {"n_groups": len(eo), "sd_within_ec_organism": sd_eo,
                               "sd_within_ec": med_sd, "ratio": ratio}

    # ------------------------------------------------------------ C5
    say("C5 DOES STRUCTURE HELP WHERE IT CAN BE JOINED?")
    try:
        st = np.load("colab/data/ml/struct_enzymes.npz", allow_pickle=True)
        acc = {str(a) for a in st["accs"]}
    except Exception:
        acc = set()
    genes = {r["gene"] for r in rows if r["gene"]}
    say(f"     structures available {len(acc):,} UniProt accessions")
    say(f"     kcat records carrying a gene symbol {sum(1 for r in rows if r['gene']):,} "
        f"over {len(genes):,} genes")
    say("     the kcat table is keyed by sequence index and gene symbol; the structure tables by")
    say("     UniProt accession, and this repo carries no symbol->accession map for that roster")
    G.add("C5", None, requires=("C1",), void_if=True,
          void_reason="the two tables cannot be joined without a symbol-to-accession map that is "
                      "not on this disk, so structure is UNTESTED here rather than refuted -- "
                      "loop 184 measured AlphaFold geometry explaining 0% of TF binding spread, "
                      "and loop 163d measured structure adding +0.0065 to enzyme assignment, but "
                      "neither is a kcat result")
    res["structure"] = {"joinable": False, "n_structures": len(acc),
                        "records_with_gene": sum(1 for r in rows if r["gene"])}

    # ------------------------------------------------------------ C6
    say("C6 WHAT THIS CANNOT SHOW")
    say("     The EC ceiling is an UPPER bound on physics, not a measurement of any physics")
    say("     method. A method that computed barriers directly could in principle beat it by")
    say("     resolving what the EC number pools -- different substrates, isoforms, conditions.")
    say("     Nothing here tried, and C5 could not test it.")
    say("     BRENDA-derived kcat values carry assay heterogeneity: temperature, pH, buffer and")
    say("     construct all vary and none is controlled here. That inflates C2 and C3's spread")
    say("     by an unknown amount, so both are pessimistic about the ceiling.")
    say("     C4 bounds that inflation from the other side but does not remove it: fixing the")
    say("     organism does not fix the assay.")
    say("     A rate being unpredictable does not make it NEEDED. Loop 156 measured growth-rate")
    say("     sensitivity to median kcat at 0.0034 against 0.9966 for ribosome elongation, so an")
    say("     aggregate can be right while every rate in it is wrong.")

    G.summary(seconds=time.time() - t0)
    gates, void = G.as_dict()
    res["gates"], res["void"] = gates, void
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    say(f"wrote {OUT}")


if __name__ == "__main__":
    main()
