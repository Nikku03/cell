"""Loop 215b. What is the accuracy at 180-240 minutes, in units a person can check?

WHY THIS EXISTS. Loop 215's H3 reported held-out R2 for six one-step splits, and R2 is the right
statistic for "does this beat predicting no change" but a poor answer to "how accurate is it". The
180->240 minute interval is the one where the relaxation actually beat persistence (-0.07647
against -0.12980), so it is the fairest place to ask what that margin buys in real terms.

This is a REPORTING loop. It fits nothing new: it takes loop 215's exact split -- train on the
intervals ending at 60, 120 and 180 minutes, score the single interval 180->240 -- and the exact
persisted three-block set point, and re-expresses the same predictions as error, direction and
correlation against the measured data.

PREDECLARED, BEFORE ANY NUMBER.

  J1 IS THIS LOOP 215's SPLIT AND MODEL?
     Gate: PASS iff the relaxation reproduces R2 -0.07647 and persistence -0.12980 to four
     decimals on the 180->240 interval. FAIL means this is describing a different fit.

  J2 HOW BIG IS THE THING BEING PREDICTED?
     The measured change over 180->240 minutes, in log2 units, as a distribution.
     Not scored -- it is the denominator every other number here should be read against.

  J3 WHAT IS THE TYPICAL ERROR?
     Median and 90th-percentile absolute error for the relaxation and for persistence.
     Gate: PASS iff the relaxation's median absolute error is below persistence's. Persistence
     predicts zero change, so its error IS the size of the real change, and a model that cannot
     beat that is not reducing error at all.

  J4 DOES IT GET THE DIRECTION RIGHT?
     Fraction of genes whose predicted sign matches the measured sign, over genes whose measured
     change exceeds the replicate noise floor.
     Gate: PASS iff directional accuracy exceeds 0.55. Fifty per cent is a coin; persistence has
     no direction at all, so this is the model's own bar.

  J5 DOES IT CORRELATE WITH THE MEASURED CHANGE?
     Pearson and Spearman between predicted and measured change.
     Gate: PASS iff Pearson exceeds 0.20.

  J6 HOW MANY GENES ARE PREDICTED WELL?
     Fraction of genes whose predicted change lands within 0.25 log2 of the measured one, against
     the fraction persistence lands within the same window.
     Gate: PASS iff the model's fraction exceeds persistence's.

  J7 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import json, os, pickle, sys, time, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from loop_setpoint_physics import gene_set
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SP = L191.SP
MODEL = ROOT / "colab" / "models" / "setpoint_stack_v1.pkl"
OUT = "outputs/loop_240_accuracy.json"
REPS = (1, 2, 3)

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def r2s(y, p):
    ss = float(np.sum((y - p) ** 2)); tt = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss / tt if tt > 0 else float("nan")


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "accuracy at 180-240 min"}
    say("=" * 104)
    say("LOOP 215b -- WHAT IS THE ACCURACY AT 180-240 MINUTES, IN UNITS A PERSON CAN CHECK?")
    say("=" * 104)

    grid, M, A9, sym, keep, tssb = gene_set()
    gi = np.where(keep)[0]
    allg = [sym[i] for i in gi]
    pos = {s: k for k, s in enumerate(allg)}
    art = pickle.load(open(MODEL, "rb"))
    names = art["genes"]
    S = np.array(art["stack_prediction"])
    idx = gi[np.array([pos[s] for s in names])]
    j_te = int(np.where(grid == 240)[0][0])
    tr_js = [j for j in range(1, j_te)]
    say(f"     train on intervals ending {[int(grid[j]) for j in tr_js]} min")
    say(f"     score the single interval {int(grid[j_te-1])}->{int(grid[j_te])} min, "
        f"{len(names):,} genes")

    def gather(js):
        L_, Y_, D_, Gg = [], [], [], []
        for j in js:
            L_.append(M[j - 1, idx]); Y_.append(M[j, idx] - M[j - 1, idx])
            D_.append(np.full(len(idx), grid[j] - grid[j - 1])); Gg.append(np.arange(len(idx)))
        return (np.concatenate(L_), np.concatenate(Y_), np.concatenate(D_), np.concatenate(Gg))
    Ltr, Ytr, Dtr, Gtr = gather(tr_js)
    Lte, Yte, Dte, Gte = gather([j_te])

    # ---------------------------------------------------------------- J1
    say("J1 IS THIS LOOP 215's SPLIT AND MODEL?")
    d_tr = Dtr * (S[Gtr] - Ltr); d_te = Dte * (S[Gte] - Lte)
    lam = float(d_tr @ Ytr / (d_tr @ d_tr))
    pred = lam * d_te
    r2_rel, r2_per = r2s(Yte, pred), r2s(Yte, np.zeros_like(Yte))
    say(f"     relaxation R2 {r2_rel:+.5f}   loop 215 recorded -0.07647")
    say(f"     persistence R2 {r2_per:+.5f}   loop 215 recorded -0.12980")
    ok1 = abs(r2_rel + 0.07647) < 5e-5 and abs(r2_per + 0.12980) < 5e-5
    G.add("J1", ok1,
          if_true="J1 PASS -- same split, same model, same numbers",
          if_false=lambda: f"J1 FAIL -- {r2_rel:.5f} / {r2_per:.5f}")

    # ---------------------------------------------------------------- J2
    say("J2 HOW BIG IS THE THING BEING PREDICTED?")
    q = np.percentile(np.abs(Yte), [50, 75, 90, 99])
    say(f"     measured |change| over 180->240 min, log2 units:")
    say(f"       median {q[0]:.4f}   75th {q[1]:.4f}   90th {q[2]:.4f}   99th {q[3]:.4f}")
    say(f"       sd {np.std(Yte):.4f}   fraction moving more than 0.25 log2 "
        f"{float(np.mean(np.abs(Yte)>0.25)):.1%}")
    # replicate noise floor on the same interval
    per_rep = []
    z = np.load(SP / "grtc" / "rna.npz", allow_pickle=True)
    tpm, mins, reps = z["tpm"], z["mins"].astype(int), z["reps"].astype(int)
    for rp in REPS:
        Mi, _ = L191.rep_trajectories(tpm, mins, reps, (rp,), grid)
        per_rep.append(Mi[j_te, idx] - Mi[j_te - 1, idx])
    noise = float(np.median([np.median(np.abs(a - b)) for a, b in
                             ((0, 1), (0, 2), (1, 2))
                             for a, b in [(per_rep[a], per_rep[b])]]))
    say(f"     replicate noise floor on this interval: median |rep1-rep2| {noise:.4f} log2")
    res["target"] = {"median_abs": q[0], "p75": q[1], "p90": q[2], "p99": q[3],
                     "sd": float(np.std(Yte)), "noise_floor": noise}

    # ---------------------------------------------------------------- J3
    say("J3 WHAT IS THE TYPICAL ERROR?")
    e_rel, e_per = np.abs(pred - Yte), np.abs(Yte)
    say(f"       relaxation   median abs error {np.median(e_rel):.4f}   "
        f"90th {np.percentile(e_rel,90):.4f} log2")
    say(f"       persistence  median abs error {np.median(e_per):.4f}   "
        f"90th {np.percentile(e_per,90):.4f} log2")
    say(f"       in fold-change terms: typical miss {2**np.median(e_rel):.3f}x against "
        f"{2**np.median(e_per):.3f}x")
    G.add("J3", bool(np.median(e_rel) < np.median(e_per)), stat=float(np.median(e_rel)),
          requires=("J1",),
          if_true=lambda: f"J3 PASS -- median error {np.median(e_rel):.4f} against persistence's "
                          f"{np.median(e_per):.4f}",
          if_false=lambda: f"J3 FAIL -- median error {np.median(e_rel):.4f} is not below "
                           f"persistence's {np.median(e_per):.4f}. The model does not reduce error")
    res["error"] = {"rel_median": float(np.median(e_rel)), "rel_p90": float(np.percentile(e_rel, 90)),
                    "per_median": float(np.median(e_per)), "per_p90": float(np.percentile(e_per, 90))}

    # ---------------------------------------------------------------- J4
    say("J4 DOES IT GET THE DIRECTION RIGHT?")
    mv = np.abs(Yte) > noise
    acc = float(np.mean(np.sign(pred[mv]) == np.sign(Yte[mv])))
    say(f"     genes moving more than the noise floor: {int(mv.sum()):,} of {len(Yte):,} "
        f"({mv.mean():.1%})")
    say(f"     directional accuracy on those {acc:.4f}")
    say(f"     (persistence predicts zero change and has no direction)")
    G.add("J4", bool(acc > 0.55), stat=acc, requires=("J1",),
          if_true=lambda: f"J4 PASS -- calls the direction right {acc:.1%} of the time",
          if_false=lambda: f"J4 FAIL -- {acc:.1%}, against a coin at 50% and a bar of 55%")
    res["direction"] = {"n_moving": int(mv.sum()), "accuracy": acc}

    # ---------------------------------------------------------------- J5
    say("J5 DOES IT CORRELATE WITH THE MEASURED CHANGE?")
    rp = pear(pred, Yte)
    rs = pear(np.argsort(np.argsort(pred)), np.argsort(np.argsort(Yte)))
    say(f"     Pearson  {rp:+.4f}     Spearman {rs:+.4f}")
    G.add("J5", bool(rp > 0.20), stat=rp, requires=("J1",),
          if_true=lambda: f"J5 PASS -- Pearson {rp:+.4f}",
          if_false=lambda: f"J5 FAIL -- Pearson {rp:+.4f} against a bar of 0.20")
    res["correlation"] = {"pearson": rp, "spearman": rs}

    # ---------------------------------------------------------------- J6
    say("J6 HOW MANY GENES ARE PREDICTED WELL?")
    for w in (0.10, 0.25, 0.50):
        a = float(np.mean(np.abs(pred - Yte) <= w)); b = float(np.mean(np.abs(Yte) <= w))
        say(f"       within {w:.2f} log2 ({2**w:.2f}x):  model {a:.1%}   persistence {b:.1%}")
    a25 = float(np.mean(np.abs(pred - Yte) <= 0.25)); b25 = float(np.mean(np.abs(Yte) <= 0.25))
    G.add("J6", bool(a25 > b25), stat=a25, requires=("J1",),
          if_true=lambda: f"J6 PASS -- {a25:.1%} within 0.25 log2 against persistence's {b25:.1%}",
          if_false=lambda: f"J6 FAIL -- {a25:.1%} against persistence's {b25:.1%}")
    res["within"] = {"model_0.25": a25, "persistence_0.25": b25}

    say("J7 WHAT THIS CANNOT SHOW")
    say("     This is ONE interval of ONE course in ONE cell line, chosen because it is the")
    say("     interval where the model beat persistence. Loop 215 measured that it loses on 4 of")
    say("     the other 5 one-step intervals, so this is the model's best case and not its")
    say("     expected performance.")
    say("     The set point was fitted with these genes' own later timepoints in the training")
    say("     folds for the SET POINT model, though not for the relaxation rate. That is loop")
    say("     213's design and it makes the set point optimistic here.")
    say("     Directional accuracy is computed only on genes moving more than the replicate")
    say("     noise floor. Including the rest would raise the number by adding coin flips on")
    say("     changes too small to call.")

    G.summary(seconds=time.time() - t0)
    gates, void = G.as_dict()
    res["gates"], res["void"] = gates, void
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)
    say(f"wrote {OUT}")


if __name__ == "__main__":
    main()
