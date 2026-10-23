"""LOOP 134 -- B4 WAS THE STOPPING RULE, AND B4 IS WRONG.

improver.py, run on the ml_kcat track, named loop 133's B4 as the rule that decides whether four
hours of 650M embedding get spent. Then it printed its own indictment: THE CHECK REJECTED 0 OF 5
ITEMS. A theory check that accepts everything is not a theory check, and the first thing to do
about that is not to tighten the check -- it is to look at the number the check was told to trust.

B4 REPORTED: "the sequence ADDS NOTHING beyond the EC number", from a refit on the EC-median
residual scoring RMSE 1.5386 against a residual sd of 1.4849, R2 -0.0737.

THE SAME JSON REFUTES IT. loop 132 recorded ec_rmse = 1.5015, the out-of-fold EC-median predictor
scored on the RAW target, against the model's 1.3768. The model beats EC-alone by 0.1247, which is
2.6x the 0.0488 paired interval. And the residual sd of 1.4849 against a constant's 1.5091 means
the EC number removes 1 - (1.4849/1.5091)^2 = 3.2% of the variance. A quantity that explains 3.2%
of the variance cannot be the thing that "everything the model knows" is already inside.

So B4's conclusion and loop 132's ec_rmse cannot both be right, and B4 is the one with a defect:

  DEFECT 1, THE TRAINING TARGET LEAKS. resid[te] for fold k is built from medians over folds != k,
  which is correct for the TEST rows. But cv_reg then TRAINS on resid[tr], and those training
  residuals were each built from a complement that CONTAINS fold k. The target the model fits was
  constructed using the labels of the fold it is about to be scored on. This is loop 129's error
  wearing a different hat, and T5 of the improver exists precisely to catch it -- but T5 was
  checking the PLAN, and nothing was checking the number the plan was built on.

  DEFECT 2, THE BASELINE IS IN-SAMPLE. `rmse(resid, pr) < resid.std()` compares an out-of-fold
  prediction against a spread computed over the whole vector with its own global mean. The honest
  baseline is the out-of-fold mean of the TRAINING residuals, which is what a constant predictor
  would actually have available.

  DEFECT 3, AND THE REAL ONE: THE RESIDUAL TEST IS THE WRONG TEST. Subtracting a per-class median
  changes the target, and a model that beats a constant on target A can lose to a constant on
  target A-minus-something without that meaning anything about class lookup. The question "does the
  sequence add anything beyond the EC number" does not need a new target at all. It needs the SAME
  target and a control that destroys the sequence while PRESERVING the EC number. That is a
  within-class permutation, and this repository already owns the machinery for it.

PREDECLARED, and note that C3 is written so it can return either verdict:

  C1 HOW MUCH DOES THE EC NUMBER ACTUALLY EXPLAIN?
       out-of-fold EC-median predictor on the raw target, against a constant. Gate: report the
       variance share. If it is small, B4's premise is dead regardless of what C3 finds.

  C2 DOES B4's RESIDUAL LEAK, AND HOW MUCH IS IT WORTH?
       build the residual BOTH ways -- loop 133's, and a clean one where a training row's residual
       uses only rows from ITS OWN side of the split -- and score both. Gate: report both numbers.
       If they differ materially the leak is real and loop 133's B4 line gets struck.

  C3 THE TEST B4 SHOULD HAVE BEEN: WITHIN-CLASS PERMUTATION.        THE DECISIVE ONE.
       same target, same folds, same features, but the ESM embedding is permuted among records
       SHARING AN EC NUMBER. EC is preserved exactly; sequence identity is destroyed. Gate: if
       permuting costs more than the paired interval of 0.0488, the sequence carries information
       the EC number does not, and B4's headline is REFUTED. If it costs less, B4 was right for the
       wrong reason and the headline SURVIVES on better evidence.

  C4 IS THE PERMUTATION CONTROL EVEN CAPABLE OF MOVING?             THE VACUOUS-GATE CHECK.
       loop 92 found twelve gates that fired while measuring nothing, and one family was the null
       that cannot move. A within-class permutation cannot move a record that is alone in its EC
       class. Gate: report the fraction of records that actually get a different embedding, and
       require it to clear gate_guard's achievable bound. A control that cannot move must not be
       reported as a control that did not move.

  C5 WHAT THE SEQUENCE ADDS, PRICED AGAINST WHAT EC ADDS.
       four nested feature sets on the same folds: constant, EC only, sequence only, both. Gate:
       report all four. This is the decomposition B4 was reaching for and it needs no residual.

  C6 THE GROUNDED MAXIMA THE IMPROVER ASKED FOR.
       every predicted_gain in improver.py's plan is currently an assertion of mine. Gate: derive
       the LARGEST gain each proposed change could possibly deliver, from a recorded number, so
       that a promise exceeding its own maximum can be rejected by arithmetic rather than taste.

-> outputs/loop_b4_fix.json
"""
import collections
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(1 << 30)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import gate_guard as GG  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
ML = Path("colab/data/ml")
SEED = 13400
N_FOLDS = 5
PAIRED_CI = 0.0488          # loop 132 A3, the interval on a paired model-vs-model difference

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 134 -- B4 was the stopping rule, and B4 is wrong")
    say("=" * 100)
    say()

    rows = list(csv.DictReader(open(ML / "kcat_records.tsv"), delimiter="\t"))
    y = np.array([float(r["log10_kcat"]) for r in rows])
    seq_id = np.array([int(r["seq_id"]) for r in rows])
    smi_id = np.array([int(r["smiles_id"]) for r in rows])
    fold = np.array([int(r["fold"]) for r in rows])
    ec = np.array([r["ec"] for r in rows])
    SF = np.load(ML / "seq_features.npy")
    FP = np.unpackbits(np.load(ML / "substrate_ecfp.npy"), axis=1).astype(np.float32)
    E = np.load(ML / "esm2_8M_mean.npy").astype(np.float32)
    say(f"  {len(rows):,} records | ESM {E.shape} | paired interval {PAIRED_CI}")
    say()

    import xgboost as xgb

    def mk():
        return xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.06,
                                subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0,
                                n_jobs=4, random_state=SEED, verbosity=0)

    def cv(X, t):
        p = np.zeros(len(t))
        for k in range(N_FOLDS):
            te, tr = fold == k, fold != k
            m = mk()
            m.fit(X[tr], t[tr])
            p[te] = m.predict(X[te])
        return p

    gates, res = {}, {}

    # ------------------------------------------------------------------ C1
    say("C1 HOW MUCH DOES THE EC NUMBER ACTUALLY EXPLAIN?")
    pec = np.zeros(len(y))
    pconst = np.zeros(len(y))
    for k in range(N_FOLDS):
        te, tr = fold == k, fold != k
        med = collections.defaultdict(list)
        for i in np.flatnonzero(tr):
            if ec[i]:
                med[ec[i]].append(y[i])
        gm = float(np.median(y[tr]))
        pec[te] = [np.median(med[ec[i]]) if ec[i] in med else gm for i in np.flatnonzero(te)]
        pconst[te] = float(np.mean(y[tr]))
    r_const, r_ec = rmse(y, pconst), rmse(y, pec)
    share = 1.0 - (r_ec / r_const) ** 2
    say(f"     out-of-fold constant         RMSE {r_const:.4f}")
    say(f"     out-of-fold EC-median        RMSE {r_ec:.4f}")
    say(f"     the EC number explains {share * 100:.1f}% of the variance in log10 kcat")
    say(f"     loop 133's B4 asserted that everything the model knows is already in the EC number.")
    say(f"     A predictor worth {share * 100:.1f}% of the variance cannot contain a model that beats it.")
    gates["C1"] = bool(np.isfinite(share))
    res["c1"] = {"constant": r_const, "ec_median": r_ec, "variance_share": float(share)}
    say(f"     C1 PASS -- measured")
    say()

    # ------------------------------------------------------------------ C2
    say("C2 DOES B4's RESIDUAL LEAK, AND HOW MUCH IS IT WORTH?")
    X = np.hstack([SF[seq_id], E[seq_id], FP[smi_id]])

    # loop 133's construction: every row's residual uses the complement of ITS OWN fold, so a
    # training row for fold k was built from a set containing fold k.
    resid_leaky = y - pec
    p_leaky = cv(X, resid_leaky)
    base_leaky = float(resid_leaky.std())

    # clean: for a given test fold k, training rows get residuals built ONLY from training rows,
    # via an inner split, so nothing in the target ever saw fold k.
    p_clean = np.zeros(len(y))
    resid_clean_te = np.zeros(len(y))
    base_clean = np.zeros(len(y))
    for k in range(N_FOLDS):
        te, tr = fold == k, fold != k
        tri = np.flatnonzero(tr)
        rc = np.zeros(len(y))
        inner = fold[tr]
        for j in np.unique(inner):                      # inner leave-one-fold-out inside training
            ite, itr = inner == j, inner != j
            med = collections.defaultdict(list)
            for i in tri[itr]:
                if ec[i]:
                    med[ec[i]].append(y[i])
            gm = float(np.median(y[tri[itr]]))
            rc[tri[ite]] = y[tri[ite]] - np.array(
                [np.median(med[ec[i]]) if ec[i] in med else gm for i in tri[ite]])
        med = collections.defaultdict(list)             # test rows: medians from training only
        for i in tri:
            if ec[i]:
                med[ec[i]].append(y[i])
        gm = float(np.median(y[tri]))
        resid_clean_te[te] = y[te] - np.array(
            [np.median(med[ec[i]]) if ec[i] in med else gm for i in np.flatnonzero(te)])
        base_clean[te] = float(np.mean(rc[tri]))        # the constant a real predictor would have
        m = mk()
        m.fit(X[tr], rc[tr])
        p_clean[te] = m.predict(X[te])
    r_leaky = rmse(resid_leaky, p_leaky)
    r_clean = rmse(resid_clean_te, p_clean)
    r_clean_base = rmse(resid_clean_te, base_clean)
    say(f"     loop 133's residual, loop 133's baseline   model {r_leaky:.4f}  vs sd {base_leaky:.4f}"
        f"   -> {'beats' if r_leaky < base_leaky else 'LOSES'}")
    say(f"     clean residual, out-of-fold baseline       model {r_clean:.4f}  vs const "
        f"{r_clean_base:.4f}   -> {'beats' if r_clean < r_clean_base else 'LOSES'}")
    say(f"     the leak was worth {r_leaky - r_clean:+.4f} and the in-sample baseline "
        f"{base_leaky - r_clean_base:+.4f}")
    gates["C2"] = True
    res["c2"] = {"leaky_model": r_leaky, "leaky_baseline": base_leaky, "clean_model": r_clean,
                 "clean_baseline": r_clean_base}
    say(f"     C2 PASS -- both constructions reported")
    say()

    # ------------------------------------------------------------------ C4 (before C3: the control
    # must be shown capable of moving BEFORE its result is allowed to mean anything)
    say("C4 IS THE PERMUTATION CONTROL EVEN CAPABLE OF MOVING?")
    by_ec = collections.defaultdict(list)
    for i, e in enumerate(ec):
        by_ec[e].append(i)
    perm = np.arange(len(y))
    for e, idx in by_ec.items():
        if len(idx) > 1:
            perm[idx] = rng.permutation(idx)
    moved = float(np.mean(seq_id[perm] != seq_id))
    alone = float(np.mean([len(by_ec[e]) == 1 for e in ec]))
    ach = GG.achievable_change(seq_id)
    say(f"     records whose EC class has only one record (cannot move): {alone * 100:.1f}%")
    say(f"     records that actually received a DIFFERENT sequence: {moved * 100:.1f}%")
    say(f"     gate_guard achievable bound for this vector: {ach:.4f}")
    gates["C4"] = bool(moved >= 0.5 * ach)
    res["c4"] = {"moved": moved, "alone_in_class": alone, "achievable": ach}
    say(f"     C4 {'PASS' if gates['C4'] else 'FAIL'} -- the control "
        f"{'can move' if gates['C4'] else 'CANNOT MOVE and its result means nothing'}")
    say()

    # ------------------------------------------------------------------ C3
    say("C3 THE TEST B4 SHOULD HAVE BEEN: WITHIN-CLASS PERMUTATION")
    p_real = cv(X, y)
    Xp = np.hstack([SF[seq_id[perm]], E[seq_id[perm]], FP[smi_id]])
    p_perm = cv(Xp, y)
    r_real, r_perm = rmse(y, p_real), rmse(y, p_perm)
    cost = r_perm - r_real
    say(f"     real embedding                        RMSE {r_real:.4f}")
    say(f"     embedding permuted WITHIN EC class    RMSE {r_perm:.4f}")
    say(f"     destroying sequence identity while keeping EC costs {cost:+.4f}")
    say(f"     the paired interval is {PAIRED_CI:.4f}")
    if gates["C4"]:
        gates["C3"] = bool(cost > PAIRED_CI)
        verdict = ("CARRIES information the EC number does not; B4 REFUTED" if gates["C3"]
                   else "adds nothing beyond EC; B4 SURVIVES on better evidence")
        say(f"     C3 {'PASS' if gates['C3'] else 'FAIL'} -- the sequence {verdict}")
    else:
        gates["C3"] = False
        say(f"     C3 FAIL -- C4 says the control cannot move, so this number is not evidence")
    res["c3"] = {"real": r_real, "permuted": r_perm, "cost": cost, "paired_ci": PAIRED_CI}
    say()

    # ------------------------------------------------------------------ C5
    say("C5 WHAT THE SEQUENCE ADDS, PRICED AGAINST WHAT EC ADDS")
    ecs = sorted({e for e in ec if e})
    eci = {e: i for i, e in enumerate(ecs)}
    ECX = np.zeros((len(y), 1), dtype=np.float32)
    ECX[:, 0] = [eci.get(e, -1) for e in ec]
    sets = {
        "constant": None,
        "EC only": ECX,
        "sequence only": np.hstack([SF[seq_id], E[seq_id]]),
        "sequence + substrate": X,
        "sequence + substrate + EC": np.hstack([X, ECX]),
    }
    tab = {}
    for name, Xs in sets.items():
        tab[name] = r_const if Xs is None else rmse(y, cv(Xs, y))
        say(f"     {name:<28} RMSE {tab[name]:.4f}")
    gates["C5"] = True
    res["c5"] = tab
    say(f"     C5 PASS -- the nested decomposition B4 was reaching for, with no residual involved")
    say()

    # ------------------------------------------------------------------ C6
    say("C6 THE GROUNDED MAXIMA THE IMPROVER ASKED FOR")
    cur = tab["sequence + substrate"]
    probe = json.load(open(OUT / "loop_ml_probe.json"))
    mi = probe["b1"]["irreducible_rmse"]

    def cap(reduce_rmse):
        """largest gain if a change removed that entire variance component"""
        return cur - max(cur ** 2 - reduce_rmse ** 2, 0.0) ** 0.5

    maxima = {
        "P1_active_site_pooling": {
            "from": "loop 133 B1 irreducible_rmse", "value": mi, "max_gain": cap(mi),
            "argument": "the ABSOLUTE ceiling if active-site pooling resolved every point-mutant "
                        "pair perfectly. It will not; this is the number a promise may not exceed"},
        "P4_explicit_EC": {
            "from": "C5, this loop", "value": tab["sequence + substrate + EC"],
            "max_gain": cur - tab["sequence + substrate + EC"],
            "argument": "not a ceiling but the MEASURED value, because C5 just ran it"},
        "P2_mutant_flag": {
            "from": "loop 133 B1", "value": mi, "max_gain": cap(mi),
            "argument": "shares P1's ceiling: a flag and a pooling change attack the SAME variance "
                        "component, so their gains may not be added"},
    }
    for k, v in maxima.items():
        say(f"     {k:<26} max gain {v['max_gain']:+.4f}   ({v['from']})")
    say(f"     P2 and P1 cite the same deficit, so the improver's bundle arithmetic must treat")
    say(f"     them as OVERLAPPING rather than independent.")
    gates["C6"] = True
    res["c6"] = maxima
    say()

    say("=" * 100)
    for k in ("C1", "C2", "C3", "C4", "C5", "C6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[ML / "kcat_records.tsv", ML / "esm2_8M_mean.npy",
                              ML / "substrate_ecfp.npy"],
                      available=len(rows), used=len(rows), selection="all", seed=SEED,
                      controls=["EC-median residual built BOTH leaky and clean, both scored",
                                "embedding permuted within EC class -- destroys sequence, keeps EC",
                                "the permutation itself checked against gate_guard's achievable "
                                "bound before its result is allowed to mean anything",
                                "five nested feature sets on identical folds"],
                      note="loop 133's B4 concluded the sequence adds nothing beyond the EC "
                           "number, and improver.py made that the stopping rule for a 4-hour "
                           "run. This retests it without a residual.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 134 -- B4 corrected", "manifest": man, "gates": gates,
               **res, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_b4_fix.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_b4_fix.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
