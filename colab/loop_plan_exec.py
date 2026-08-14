"""LOOP 135 -- EXECUTE THE PART OF THE PLAN THAT CAN ACTUALLY RUN.

improver.py proposed five changes. T8 -- do the files exist -- invalidates two of them outright:
the UniProt active-site annotations P1 needs were never successfully fetched, and the temperature
and pH that P3's Q10 normalisation needs are not on disk in any form, despite Q6 of the first run
describing them as "free, already on disk". P5 is BLOCKED behind P1. That leaves P2 and P4, and
the improver's own upgrade step said to run them as ONE change because each alone predicts less
than the 0.0488 paired interval.

SO THIS LOOP IS SMALL ON PURPOSE. It is what remains after the checks have taken everything that
could not be justified, and reporting that honestly is the result.

THE TWO CHANGES:

  P2 A MUTANT FLAG AND A SUBSTITUTION COUNT. loop 133 B1 found 18,595 point-mutant pairs whose
  kcat differs by a median 4.5x. A mean-pooled embedding cannot see which residue changed. It can,
  however, be told THAT a record is an engineered variant and by how many residues, both of which
  are computable from sequences.json alone. The ceiling is the whole 0.947 mutant component; the
  claim is 12% of it, because a flag recovers a mean offset and not a per-variant effect.

  P4 THE EC NUMBER AS AN EXPLICIT FEATURE. loop 134 C5 measured this directly.

WHAT WOULD LEAK, AND IS THEREFORE NOT DONE. The obvious mutant feature is the wild type's own
kcat. It is forbidden: it is the label of another record, unavailable at prediction time, and the
grouped split puts a WT and its variants in the SAME fold, so it would be read straight out of the
test set. This is loop 129's error and the temptation here is stronger, so it is written down.

PREDECLARED:

  E1 CAN THE MUTANT DETECTOR EVEN FIRE?                              THE CAPABILITY CHECK.
       fraction of records receiving is_mutant = 1, and the achievable-change bound for that
       vector. Gate: the flag must be non-degenerate by gate_guard's standard. A feature that is
       constant cannot help and must not be reported as a feature that did not help.

  E2 P2 ALONE.
       baseline features plus is_mutant and n_substitutions, same folds. Gate: report the paired
       difference and a CLUSTER-level bootstrap interval on it. Predeclared expectation: the gain
       is smaller than the 0.0488 interval, i.e. NOT individually measurable. Confirming that is a
       pass, because the improver already said so and a plan whose predictions fail is worth more
       than one whose predictions are never checked.

  E3 P4 ALONE.
       baseline plus the EC index. Gate: same treatment.

  E4 THE BUNDLE, TESTED AS ONE CHANGE.                               THE DECISIVE ONE.
       both feature blocks together against the baseline. Gate: the bundle's paired interval must
       exclude zero. If it does not, the improver's bundle arithmetic over-promised and the record
       says so.

  E5 THE NEGATIVE CONTROL.
       the mutant flag and substitution count PERMUTED across records, everything else identical.
       Gate: the permuted gain must be smaller than the real gain. If a shuffled flag helps as
       much as a real one, the gain is capacity and not information.

  E6 DID THE IMPROVER'S PREDICTION HOLD?
       predicted combined gain against measured. Gate: report both and the signed error. This is
       the number that feeds the next turn of the loop, and it is the only way the predictions in
       improver.py ever acquire a track record.

-> outputs/loop_plan_exec.json
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
SEED = 13500
N_FOLDS = 5
NEAR_J = 0.95
MUT_MAX = 5
N_BOOT = 400

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def kmers(s, k=5):
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 135 -- execute the part of the plan that can actually run")
    say("=" * 100)
    say()

    rows = list(csv.DictReader(open(ML / "kcat_records.tsv"), delimiter="\t"))
    y = np.array([float(r["log10_kcat"]) for r in rows])
    seq_id = np.array([int(r["seq_id"]) for r in rows])
    smi_id = np.array([int(r["smiles_id"]) for r in rows])
    fold = np.array([int(r["fold"]) for r in rows])
    clu = np.array([int(r["cluster_id"]) for r in rows])
    ec = np.array([r["ec"] for r in rows])
    seqs = json.load(open(ML / "sequences.json"))
    SF = np.load(ML / "seq_features.npy")
    FP = np.unpackbits(np.load(ML / "substrate_ecfp.npy"), axis=1).astype(np.float32)
    E = np.load(ML / "esm2_8M_mean.npy").astype(np.float32)
    say(f"  {len(rows):,} records | {len(seqs):,} sequences | ESM {E.shape}")
    say()

    import xgboost as xgb

    def cv(X):
        p = np.zeros(len(y))
        for k in range(N_FOLDS):
            te, tr = fold == k, fold != k
            m = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.06,
                                 subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0,
                                 n_jobs=4, random_state=SEED, verbosity=0)
            m.fit(X[tr], y[tr])
            p[te] = m.predict(X[te])
        return p

    def paired(pa, pb):
        """cluster-level bootstrap on the PAIRED per-record squared-error difference. Resampling
        clusters and not records is the only honest unit here: records inside a cluster are near
        copies, so a record bootstrap would report an interval several times too tight."""
        ea, eb = (y - pa) ** 2, (y - pb) ** 2
        cl = np.unique(clu)
        idx = {c: np.flatnonzero(clu == c) for c in cl}
        d = []
        for _ in range(N_BOOT):
            pick = rng.choice(cl, size=len(cl), replace=True)
            sel = np.concatenate([idx[c] for c in pick])
            d.append(np.sqrt(ea[sel].mean()) - np.sqrt(eb[sel].mean()))
        d = np.array(d)
        return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    gates, res = {}, {}

    # ------------------------------------------------------------------ mutant detection
    say("BUILDING THE MUTANT FEATURES (P2)")
    by_clu = collections.defaultdict(list)
    seq_clu = {}
    for i in range(len(rows)):
        seq_clu[seq_id[i]] = clu[i]
    for sid, c in seq_clu.items():
        by_clu[c].append(sid)
    is_mut = np.zeros(len(seqs), dtype=np.float32)
    n_sub = np.zeros(len(seqs), dtype=np.float32)
    n_near = np.zeros(len(seqs), dtype=np.float32)
    for c, sids in by_clu.items():
        if len(sids) < 2:
            continue
        km = {s: kmers(seqs[s]) for s in sids}
        for ii in range(len(sids)):
            for jj in range(ii + 1, len(sids)):
                a, b = sids[ii], sids[jj]
                sa, sb = seqs[a], seqs[b]
                if abs(len(sa) - len(sb)) > 0:
                    continue
                ka, kb = km[a], km[b]
                u = len(ka | kb)
                if not u or len(ka & kb) / u < NEAR_J:
                    continue
                nd = sum(1 for x, z in zip(sa, sb) if x != z)
                if 1 <= nd <= MUT_MAX:
                    for s in (a, b):
                        is_mut[s] = 1.0
                        n_near[s] += 1
                    n_sub[a] = nd if n_sub[a] == 0 else min(n_sub[a], nd)
                    n_sub[b] = nd if n_sub[b] == 0 else min(n_sub[b], nd)
    MUT = np.stack([is_mut[seq_id], n_sub[seq_id], n_near[seq_id]], axis=1)
    say(f"     sequences flagged as members of a point-mutant pair: {int(is_mut.sum()):,}")
    say(f"     records carrying the flag: {int(MUT[:, 0].sum()):,} of {len(rows):,}")
    say()

    # ------------------------------------------------------------------ E1
    say("E1 CAN THE MUTANT DETECTOR EVEN FIRE?")
    frac = float(MUT[:, 0].mean())
    ach = GG.achievable_change(MUT[:, 0])
    say(f"     fraction of records flagged: {frac:.4f}")
    say(f"     gate_guard achievable-change bound for that binary vector: {ach:.4f}")
    say(f"     distinct substitution counts present: {sorted(set(n_sub[n_sub > 0].tolist()))[:8]}")
    gates["E1"] = bool(ach >= 0.02)
    say(f"     E1 {'PASS' if gates['E1'] else 'FAIL'} -- the flag is "
        f"{'non-degenerate' if gates['E1'] else 'DEGENERATE and cannot be reported as a feature'}")
    res["e1"] = {"flagged_fraction": frac, "achievable": ach,
                 "n_flagged_records": int(MUT[:, 0].sum())}
    say()

    # ------------------------------------------------------------------ feature sets
    ecs = sorted({e for e in ec if e})
    eci = {e: i for i, e in enumerate(ecs)}
    ECX = np.array([[eci.get(e, -1)] for e in ec], dtype=np.float32)
    BASE = np.hstack([SF[seq_id], E[seq_id], FP[smi_id]])
    p_base = cv(BASE)
    r_base = rmse(y, p_base)
    say(f"  BASELINE (loop 132's feature set, same folds): RMSE {r_base:.4f}")
    say()

    # ------------------------------------------------------------------ E2
    say("E2 P2 ALONE -- the mutant flag and substitution count")
    p2 = cv(np.hstack([BASE, MUT]))
    r2_ = rmse(y, p2)
    g2, lo2, hi2 = paired(p_base, p2)
    say(f"     RMSE {r2_:.4f}   gain {r_base - r2_:+.4f}")
    say(f"     cluster bootstrap on the paired difference: {g2:+.4f} [{lo2:+.4f}, {hi2:+.4f}]")
    gates["E2"] = True
    res["e2"] = {"rmse": r2_, "gain": r_base - r2_, "boot": [g2, lo2, hi2]}
    say(f"     E2 PASS -- measured with an interval; individually "
        f"{'SIGNIFICANT' if lo2 > 0 else 'not distinguishable from zero, as predicted'}")
    say()

    # ------------------------------------------------------------------ E3
    say("E3 P4 ALONE -- the EC number as an explicit feature")
    p4 = cv(np.hstack([BASE, ECX]))
    r4_ = rmse(y, p4)
    g4, lo4, hi4 = paired(p_base, p4)
    say(f"     RMSE {r4_:.4f}   gain {r_base - r4_:+.4f}")
    say(f"     cluster bootstrap on the paired difference: {g4:+.4f} [{lo4:+.4f}, {hi4:+.4f}]")
    gates["E3"] = True
    res["e3"] = {"rmse": r4_, "gain": r_base - r4_, "boot": [g4, lo4, hi4]}
    say(f"     E3 PASS -- measured with an interval; individually "
        f"{'SIGNIFICANT' if lo4 > 0 else 'not distinguishable from zero, as predicted'}")
    say()

    # ------------------------------------------------------------------ E4
    say("E4 THE BUNDLE, TESTED AS ONE CHANGE")
    pb = cv(np.hstack([BASE, MUT, ECX]))
    rb_ = rmse(y, pb)
    gb, lob, hib = paired(p_base, pb)
    say(f"     RMSE {rb_:.4f}   gain {r_base - rb_:+.4f}")
    say(f"     cluster bootstrap on the paired difference: {gb:+.4f} [{lob:+.4f}, {hib:+.4f}]")
    gates["E4"] = bool(lob > 0)
    say(f"     E4 {'PASS' if gates['E4'] else 'FAIL'} -- the bundle "
        f"{'is measurably better than the baseline' if gates['E4'] else 'DOES NOT clear zero; the improver over-promised'}")
    res["e4"] = {"rmse": rb_, "gain": r_base - rb_, "boot": [gb, lob, hib]}
    say()

    # ------------------------------------------------------------------ E5
    say("E5 THE NEGATIVE CONTROL -- the same features, shuffled")
    sh = rng.permutation(len(rows))
    pp = cv(np.hstack([BASE, MUT[sh], ECX[sh]]))
    rp_ = rmse(y, pp)
    say(f"     permuted-feature RMSE {rp_:.4f}   'gain' {r_base - rp_:+.4f}")
    say(f"     real bundle gain {r_base - rb_:+.4f}")
    gates["E5"] = bool((r_base - rb_) > (r_base - rp_))
    say(f"     E5 {'PASS' if gates['E5'] else 'FAIL'} -- "
        f"{'the real features beat their own shuffle' if gates['E5'] else 'A SHUFFLE DOES AS WELL; the gain is capacity, not information'}")
    res["e5"] = {"permuted_rmse": rp_, "permuted_gain": r_base - rp_}
    say()

    # ------------------------------------------------------------------ E6
    say("E6 DID THE IMPROVER'S PREDICTION HOLD?")
    pred = None
    ip = OUT / "improver_ml_kcat.json"
    if ip.exists():
        d = json.load(open(ip))
        bd = d.get("bundle") or {}
        pred = bd.get("combined_gain")
    meas = r_base - rb_
    say(f"     improver predicted combined gain: "
        f"{('%+.4f' % pred) if pred is not None else 'not recorded'}")
    say(f"     measured                        : {meas:+.4f}")
    if pred is not None:
        say(f"     signed error                    : {meas - pred:+.4f}   "
            f"({'OVER-promised' if meas < pred else 'UNDER-promised'})")
    gates["E6"] = True
    res["e6"] = {"predicted": pred, "measured": meas,
                 "error": (meas - pred) if pred is not None else None}
    say(f"     E6 PASS -- recorded, so the predictions in improver.py now have a track record")
    say()

    say("=" * 100)
    for k in ("E1", "E2", "E3", "E4", "E5", "E6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[ML / "kcat_records.tsv", ML / "sequences.json",
                              ML / "esm2_8M_mean.npy"],
                      available=len(rows), used=len(rows), selection="all", seed=SEED,
                      controls=["mutant flag checked against gate_guard's achievable bound "
                                "BEFORE its effect is measured",
                                "every gain carries a CLUSTER-level bootstrap interval",
                                "the added features permuted, as a capacity control",
                                "the improver's own prediction scored against the measurement"],
                      note="P1 and P3 were removed by T8 (inputs absent) and P5 is BLOCKED behind "
                           "P1, so this is the whole executable plan and not a selection from it.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 135 -- executable plan", "manifest": man, "gates": gates,
               "baseline_rmse": r_base, **res, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_plan_exec.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_plan_exec.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
