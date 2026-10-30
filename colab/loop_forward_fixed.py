"""Loop 223. The forward test, with the defect removed and run on the intervals that carry signal.

WHAT WAS WRONG WITH LOOP 222's V7, IN ITS OWN WORDS FROM THE COMMIT. V7 scored each arm against
ITS OWN persistence baseline and compared the two lifts. Patched lift +1.03487 beat unpatched
+0.55058 and the gate said PASS. But the patched model's absolute R2 was WORSE, -0.08898 against
-0.04997, and the lift only grew because patching dragged its own baseline from -0.60054 to
-1.12385. A gate that can be won by damaging its own denominator is the Family One fault wearing a
different hat, and it passed.

THE FIX IS STRUCTURAL, NOT A TIGHTER BAR. Three things are now nailed down and shared by every arm:

    THE TARGET is replicate 4's RAW per-interval change. It is never patched, by anything, in any
    arm. A patch can therefore never make its own target easier.

    THE BASELINE is persistence -- the raw replicate-1-2-3 mean change at the preceding interval.
    One baseline, one number, shared. No arm gets its own.

    THE PATCH APPLIES ONLY TO THE PREDICTOR SIDE. This is the whole point. The honest question is
    whether correcting the INPUT helps predict an UNCORRECTED future, and a correction that
    destroys information now loses instead of winning.

W1 checks all of this mechanically rather than asserting it: the target vector and the baseline
vector must be bit-identical across arms, max absolute difference exactly 0.0.

AND THE RUN MOVES TO THE INTERVALS THAT CARRY SIGNAL. Loop 222 measured within-replicate-1-2-3
agreement per interval and the spread is nearly twenty-fold:

    30->60   +0.202      240->420  +0.317
    60->120  +0.292      420->480  +0.145
    120->180 +0.036      480->600  +0.127
    180->240 +0.014      600->720  +0.000

Loops 215 and 215b scored short-horizon prediction on 180->240 -- reliability +0.014, the second
worst interval in the series. The 75.3% directional accuracy this project has been carrying was
measured where the replicates barely agree at all.

EVALUATION IS LEAVE-ONE-INTERVAL-OUT. For each testable interval the model trains on every other
interval and is scored on that one, so no interval is ever fit to itself and all seven get a
held-out number instead of two.

THREE ARMS, differing only in the predictor:
    RAW      the replicate-1-2-3 mean change, uncorrected.
    RUV      the same, with loop 222's k=4 control-gene directions regressed out. k and the
             directions are re-estimated here on replicates 1, 2, 3 only.
    RUV+W    the same predictor, with training rows weighted by their interval's measured
             reliability -- the statistically correct use of a reliability estimate, rather than
             loop 222's rescaling of both sides.

PREDECLARED, BEFORE ANY NUMBER.

  W1 IS THE COMPARISON WELL-POSED?  -- the fix to V7, checked and not asserted
     Gate: PASS iff the target vector and the persistence vector are bit-identical across all
     three arms, max absolute difference exactly 0.0. A FAIL means the defect is still present
     and W2 onward must not be read.

  W2 DOES ANY ARM BEAT PERSISTENCE ON THE SHARED TARGET?
     Pooled over all seven held-out intervals, R2 against the single shared persistence baseline.
     Gate: PASS iff at least one arm's pooled R2 exceeds persistence's pooled R2.

  W3 DOES PER-INTERVAL SKILL TRACK PER-INTERVAL RELIABILITY?
     Spearman across the seven testable intervals between within-{1,2,3} reliability and
     held-out R2 against replicate 4. These are different quantities measured on different
     replicates -- reliability never sees replicate 4 -- so this is not tautological.
     Gate: PASS iff Spearman >= +0.75 AND the permutation p is below 0.05. Both are required
     because with seven points a Spearman of +0.6 has p near 0.12; the coefficient alone would
     not be evidence and stating that here is cheaper than discovering it afterwards.

  W4 ON THE GOOD INTERVALS ONLY, IS ANYTHING PREDICTABLE?
     Restrict to 60->120 and 240->420, the two intervals above +0.29 reliability.
     Gate: PASS iff the best arm beats persistence on BOTH of them separately. One out of two is
     a coin flip and will be reported as such.

  W5 CONTROL: DOES W3 SURVIVE PERMUTING THE RELIABILITY LABELS?
     1,000 permutations of the reliability vector against fixed held-out R2 values.
     Gate: PASS iff the real Spearman sits outside the 95th percentile of the null. Requires W3
     to have produced a defined coefficient, not to have passed -- a refuted W3 still needs its
     null to be interpretable.

  W6 WHAT THIS CANNOT SHOW -- written before the run.
     Seven intervals is seven points. W3 and W5 are testing a monotone relationship on a sample
     that small, and a single interval moving rank can flip the coefficient by 0.2 or more. If W3
     passes it is suggestive of reliability being the right axis, not a demonstration.
     Beating persistence on a per-interval change says nothing about whether the predicted change
     is biologically correct; replicate 4 is one measurement, not the truth.
     RUV's control genes are the lowest-|plateau| fifth of the roster, so if the contaminating
     component lives mainly in responders rather than in quiet genes, RUV cannot see it and a
     FAIL would not mean the component is absent.
"""
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SP = L191.SP
OUT = "outputs/loop_forward_fixed.json"
GRID = [30, 60, 120, 180, 240, 420, 480, 600, 720]
MIN_TPM, SEED, RIDGE = 1.0, 223223, 1.0
K_RUV, CTRL_FRAC = 4, 0.20
GOOD = [(60, 120), (240, 420)]
SPEAR_BAR, NPERM = 0.75, 1000

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear(a, b):
    a = np.asarray(a, float).ravel() - np.mean(a); b = np.asarray(b, float).ravel() - np.mean(b)
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def r2s(y, p):
    ss = float(np.sum((y - p) ** 2)); tt = float(np.sum((y - np.mean(y)) ** 2))
    return 1 - ss / tt if tt > 0 else float("nan")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return pear(ra, rb)


def ruv_remove(D_, dirs):
    if dirs is None or len(dirs) == 0:
        return D_
    B = np.asarray(dirs, float).T
    return D_ - (B @ np.linalg.pinv(B.T @ B) @ B.T) @ D_


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "forward prediction with the V7 defect removed"}
    rng = np.random.default_rng(SEED)
    say("=" * 104)
    say("LOOP 223 -- THE FORWARD TEST, DEFECT REMOVED, RUN WHERE THE SIGNAL IS")
    say("=" * 104)
    say("     Loop 222's V7 compared each arm against its own persistence baseline, so patching")
    say("     could win by wrecking its own denominator -- and did: patched lift +1.0349 beat")
    say("     unpatched +0.5506 while the patched model's absolute R2 was WORSE, -0.0890 vs")
    say("     -0.0500. Here the target and the baseline are shared and the patch touches only")
    say("     the predictor.")

    z = np.load(SP / "grtc" / "rna.npz", allow_pickle=True)
    tpm, mins, reps = z["tpm"], z["mins"].astype(int), z["reps"].astype(int)
    g = np.array(GRID, float)
    base = {r: tpm[(mins == 30) & (reps == r)].mean(0) for r in (1, 2, 3, 4)}
    sel = np.where(np.all([base[r] >= MIN_TPM for r in (1, 2, 3, 4)], axis=0))[0]
    ngen = len(sel)
    V = {}
    for r in (1, 2, 3, 4):
        Mi, _ = L191.rep_trajectories(tpm, mins, reps, (r,), g)
        V[r] = Mi[:, sel]
    D = {r: np.array([V[r][j] - V[r][j - 1] for j in range(1, len(g))]) for r in (1, 2, 3, 4)}
    n_iv = len(g) - 1
    P_raw = np.mean([D[r] for r in (1, 2, 3)], axis=0)
    T = D[4]                                                  # the target, never patched
    say(f"     {ngen:,} genes, {n_iv} intervals, predictor = mean of replicates 1,2,3, "
        f"target = replicate 4 RAW")

    rel = np.array([np.mean([pear(D[a][j], D[b][j]) for a, b in ((1, 2), (1, 3), (2, 3))])
                    for j in range(n_iv)], float)
    say("     within-123 reliability per interval (re-measured here):")
    say("       " + "  ".join(f"{GRID[j]}->{GRID[j+1]} {rel[j]:+.3f}" for j in range(n_iv)))

    plateau = np.mean([V[r] for r in (1, 2, 3)], axis=0)[-3:].mean(0)
    ctrl = np.argsort(np.abs(plateau))[: int(ngen * CTRL_FRAC)]
    Dc = P_raw[:, ctrl]
    Uc, _, _ = np.linalg.svd(Dc - Dc.mean(1, keepdims=True), full_matrices=False)
    dirs = Uc[:, :K_RUV].T
    P_ruv = ruv_remove(P_raw, dirs)
    say(f"     RUV: k={K_RUV} directions from the {len(ctrl):,} lowest-|plateau| genes, "
        f"estimated on replicates 1,2,3 only")

    ARMS = {"RAW": (P_raw, False), "RUV": (P_ruv, False), "RUV+W": (P_ruv, True)}
    per_iv, pooled, tgt_store, pers_store = {a: {} for a in ARMS}, {}, {}, {}

    for arm, (P, use_w) in ARMS.items():
        lvl = np.cumsum(P, axis=0)
        rows = {}
        for j in range(1, n_iv):
            X = np.column_stack([P[j - 1], lvl[j - 1], np.ones(ngen)])
            rows[j] = (X, T[j], P_raw[j - 1])                 # baseline is RAW for every arm
        yh_all, y_all, pe_all, w_all = [], [], [], []
        for j in sorted(rows):
            Xtr = np.vstack([rows[k][0] for k in rows if k != j])
            ytr = np.concatenate([rows[k][1] for k in rows if k != j])
            if use_w:
                wtr = np.concatenate([np.full(ngen, max(rel[k], 0.0)) for k in rows if k != j])
                Wd = wtr[:, None]
                A = (Xtr * Wd).T @ Xtr + RIDGE * np.eye(Xtr.shape[1])
                b = np.linalg.solve(A, (Xtr * Wd).T @ ytr)
            else:
                A = Xtr.T @ Xtr + RIDGE * np.eye(Xtr.shape[1])
                b = np.linalg.solve(A, Xtr.T @ ytr)
            Xte, yte, pe = rows[j]
            yh = Xte @ b
            per_iv[arm][j] = {"r2": r2s(yte, yh), "r2_pers": r2s(yte, pe), "r": pear(yte, yh),
                              "rel": float(rel[j])}
            yh_all.append(yh); y_all.append(yte); pe_all.append(pe)
        y_all = np.concatenate(y_all); yh_all = np.concatenate(yh_all)
        pe_all = np.concatenate(pe_all)
        pooled[arm] = {"r2": r2s(y_all, yh_all), "r": pear(y_all, yh_all)}
        tgt_store[arm] = y_all; pers_store[arm] = pe_all
    pooled_pers = r2s(tgt_store["RAW"], pers_store["RAW"])

    # ---------------------------------------------------------------- W1
    say("W1 IS THE COMPARISON WELL-POSED?")
    dt = max(float(np.max(np.abs(tgt_store[a] - tgt_store["RAW"]))) for a in ARMS)
    dp = max(float(np.max(np.abs(pers_store[a] - pers_store["RAW"]))) for a in ARMS)
    say(f"     target vectors across arms, max absolute difference      {dt:.1e}")
    say(f"     persistence vectors across arms, max absolute difference {dp:.1e}")
    say(f"     shared persistence baseline, pooled over 7 held-out intervals: R2 {pooled_pers:+.5f}")
    G.add("W1", bool(dt == 0.0 and dp == 0.0), stat=float(dt + dp),
          if_true=lambda: "W1 PASS -- every arm is scored against a bit-identical target and a "
                          "bit-identical baseline; loop 222's V7 defect cannot recur here",
          if_false=lambda: f"W1 FAIL -- arms differ on target by {dt:.1e} or baseline by {dp:.1e}")
    res["wellposed"] = {"target_maxdiff": dt, "pers_maxdiff": dp, "pooled_persistence": pooled_pers}

    # ---------------------------------------------------------------- W2
    say("W2 DOES ANY ARM BEAT PERSISTENCE ON THE SHARED TARGET?")
    for a in ARMS:
        say(f"       {a:<6} pooled R2 {pooled[a]['r2']:+.5f}   Pearson {pooled[a]['r']:+.5f}")
    say(f"       {'PERSIST':<6} pooled R2 {pooled_pers:+.5f}")
    best = max(pooled, key=lambda a: pooled[a]["r2"])
    G.add("W2", bool(pooled[best]["r2"] > pooled_pers), stat=float(pooled[best]["r2"]),
          requires=("W1",),
          if_true=lambda: f"W2 PASS -- {best} reaches {pooled[best]['r2']:+.4f} against "
                          f"persistence {pooled_pers:+.4f}",
          if_false=lambda: f"W2 FAIL -- best arm {best} at {pooled[best]['r2']:+.4f} does not "
                           f"beat persistence {pooled_pers:+.4f}")
    res["pooled"] = pooled; res["best_arm"] = best

    # ---------------------------------------------------------------- W3
    say("W3 DOES PER-INTERVAL SKILL TRACK PER-INTERVAL RELIABILITY?")
    js = sorted(per_iv[best])
    rr = np.array([per_iv[best][j]["rel"] for j in js])
    sk = np.array([per_iv[best][j]["r2"] - per_iv[best][j]["r2_pers"] for j in js])
    for j in js:
        d = per_iv[best][j]
        say(f"       {GRID[j]:>3}->{GRID[j+1]:<3}  reliability {d['rel']:+.3f}   "
            f"held-out R2 {d['r2']:+.5f}   persistence {d['r2_pers']:+.5f}   "
            f"lift {d['r2']-d['r2_pers']:+.5f}")
    rho = spearman(rr, sk)
    null = np.array([spearman(rng.permutation(rr), sk) for _ in range(NPERM)])
    pv = float((np.sum(null >= rho) + 1) / (NPERM + 1))
    say(f"     Spearman(reliability, lift) over {len(js)} intervals = {rho:+.4f}   "
        f"permutation p = {pv:.4f}")
    G.add("W3", bool(rho >= SPEAR_BAR and pv < 0.05), stat=float(rho), requires=("W1",),
          if_true=lambda: f"W3 PASS -- rho {rho:+.3f}, p {pv:.3f}; intervals the replicates agree "
                          f"on are the intervals that are predictable",
          if_false=lambda: f"W3 FAIL -- rho {rho:+.3f} against a {SPEAR_BAR:+.2f} bar, p {pv:.3f}")
    res["per_interval"] = {a: {str(j): per_iv[a][j] for j in per_iv[a]} for a in ARMS}
    res["spearman"] = {"rho": float(rho), "p": pv, "n": len(js)}

    # ---------------------------------------------------------------- W4
    say("W4 ON THE GOOD INTERVALS ONLY, IS ANYTHING PREDICTABLE?")
    gj = [j for j in js if (GRID[j], GRID[j + 1]) in GOOD]
    wins = []
    for j in gj:
        d = per_iv[best][j]
        w = d["r2"] > d["r2_pers"]
        wins.append(w)
        say(f"       {GRID[j]:>3}->{GRID[j+1]:<3}  {best} R2 {d['r2']:+.5f}   persistence "
            f"{d['r2_pers']:+.5f}   {'beats' if w else 'does not beat'} persistence")
    G.add("W4", bool(len(wins) == len(GOOD) and all(wins)),
          stat=float(sum(wins)) if wins else None, requires=("W1",),
          if_true=lambda: f"W4 PASS -- {best} beats persistence on both high-reliability intervals",
          if_false=lambda: f"W4 FAIL -- {best} beats persistence on {sum(wins)} of "
                           f"{len(GOOD)} high-reliability intervals")
    res["good_intervals"] = {f"{GRID[j]}->{GRID[j+1]}": per_iv[best][j] for j in gj}

    # ---------------------------------------------------------------- W5
    say("W5 CONTROL: DOES W3 SURVIVE PERMUTING THE RELIABILITY LABELS?")
    say(f"     null over {NPERM} permutations: mean {null.mean():+.4f}, sd {null.std():.4f}, "
        f"95th percentile {np.percentile(null,95):+.4f}")
    say(f"     real {rho:+.4f}")
    G.add("W5", bool(rho > np.percentile(null, 95)), stat=float(rho),
          requires=("W1",),
          if_true=lambda: f"W5 PASS -- {rho:+.3f} is outside the permuted 95th percentile "
                          f"{np.percentile(null,95):+.3f}",
          if_false=lambda: f"W5 FAIL -- {rho:+.3f} is inside the permuted 95th percentile "
                           f"{np.percentile(null,95):+.3f}; with {len(js)} points the coefficient "
                           f"alone was never going to be evidence")
    res["null"] = {"mean": float(null.mean()), "sd": float(null.std()),
                   "p95": float(np.percentile(null, 95))}

    # ---------------------------------------------------------------- W6
    say("W6 WHAT THIS CANNOT SHOW")
    say("     Seven intervals is seven points. W3 and W5 test a monotone relationship on a sample")
    say("     that small, and one interval moving rank can shift the coefficient by 0.2 or more.")
    say("     Beating persistence on a per-interval change says nothing about whether the change")
    say("     predicted is biologically correct: replicate 4 is one measurement, not the truth.")
    say("     RUV's controls are the lowest-|plateau| fifth of the roster, so if the contaminating")
    say("     component lives in responders rather than quiet genes, RUV cannot see it and a FAIL")
    say("     here would not mean the component is absent.")

    res["reliability"] = [float(x) for x in rel]
    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary()
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
