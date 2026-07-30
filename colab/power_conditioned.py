"""A POWER-CONDITIONED PROTOCOL FOR EFFECT-SIZE PREDICTION. Turning a confound into a usable benchmark.

THE PROBLEM THIS SOLVES. `effectsize.py` found that |EffectSize| on the 820 causal pairs is not predictable
from biology -- every biological arm was net-negative -- while adding DETECTION POWER as a feature reached
R2 +0.0881. So how hard an enhancer appears to act is substantially a function of how well the experiment could
see it. The marginal correlation of |EffectSize| with power is only -0.058, so the confound is invisible to the
obvious check and was only caught by running the arm.

That is not fixable with a better model, and it is not a reason to abandon quantitative effect-size
prediction. It is a reason to change the TARGET. Three protocols are defined and measured here, each removing
the confound a different way, so that a future model can be scored on a quantity that is not mostly an assay
property:

 P1 STRATIFIED. Score within narrow power strata and macro-average. Inside a stratum power is near-constant,
    so it cannot be the predictor. Costs sample size and is the least assumption-laden.

 P2 RESIDUALISED. Regress |EffectSize| on power alone (flexibly, via power deciles) and predict the RESIDUAL.
    Keeps all 820 pairs. Assumes the power-effect relation is estimable, which the stratified arm checks.

 P3 RANK-WITHIN-STRATUM. Convert |EffectSize| to its rank inside its own power stratum, then pool the ranks.
    Distribution-free, keeps all pairs, and is the version most robust to the shape of the power relation.

WHAT SUCCESS AND FAILURE BOTH MEAN. If biology predicts the power-conditioned target, quantitative E-G
prediction is benchmarkable and the earlier null was a confound artefact. If biology still fails, the honest
conclusion is stronger than before: effect magnitude is not predictable from these features at all, and the
+0.0881 was purely the assay. Either way the protocol is the deliverable -- it is what a later model should be
scored against.

THE NAIVE ARM IS KEPT for contrast, and the power-as-feature arm is kept as the confound demonstration. Every
arm has a permuted-target control, held out by chromosome, ridge rather than trees at 820 rows.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SEEDS = (0, 1, 2, 3, 4)
NSTRAT = int(os.environ.get("PC_NSTRAT", 4))


def main():
    from scipy import stats
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from ranking_gate import load, build, folds_chrom, fnum, ACT, CTCFC
    print("=" * 100)
    print("POWER-CONDITIONED EFFECT-SIZE PROTOCOL -- making magnitude benchmarkable")
    print("=" * 100)
    rows = load()
    d = build(rows)
    sig = d["y"] == 1
    eff = np.array([fnum(r, "EffectSize") for r in rows])[sig]
    ae = np.abs(eff)
    ch = d["ch"][sig]
    X = np.hstack([d["D"], d["A"], d["C"], d["P"]])[sig]
    # power at several effect sizes: the confound is "could this experiment have seen an effect", and the
    # benchmark reports it at five thresholds. Using all five describes the detection regime better than one.
    PWC = [f"PowerAtEffectSize{k}" for k in (10, 15, 20, 25, 50)]
    P = np.column_stack([[fnum(r, c) for r in rows] for c in PWC])[sig]
    pw = P[:, PWC.index("PowerAtEffectSize25")]
    print(f"  {len(ae)} causal pairs; power columns {PWC}")
    print(f"  |EffectSize| median {np.median(ae):.4f}; marginal Spearman with power@ES25 "
          f"{stats.spearmanr(ae, pw)[0]:+.4f}  <- the check that misses the confound")

    def loco(Xa, t, chs):
        sc = []
        for s in SEEDS:
            f = folds_chrom(chs, s)
            pred = np.full(len(t), np.nan)
            for k in range(5):
                te = f == k
                if te.sum() < 8 or (~te).sum() < 30:
                    continue
                mu, sd = Xa[~te].mean(0), Xa[~te].std(0) + 1e-9
                m = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit((Xa[~te] - mu) / sd, t[~te])
                pred[te] = m.predict((Xa[te] - mu) / sd)
            ok = np.isfinite(pred)
            if ok.sum() > 40:
                sc.append(r2_score(t[ok], pred[ok]))
        return float(np.mean(sc)) if sc else np.nan

    rng = np.random.default_rng(0)

    def arm(Xa, t, chs, label, indent=4):
        r2 = loco(Xa, t, chs)
        perm = np.mean([loco(Xa, t[rng.permutation(len(t))], chs) for _ in range(3)])
        print(f"{' '*indent}{label:34s} R2 {r2:+.4f}   permuted {perm:+.4f}   net {r2-perm:+.4f}")
        return {"r2": r2, "permuted": float(perm), "net": r2 - float(perm)}

    R = {"n": int(len(ae)), "power_cols": PWC,
         "marginal_spearman_ae_power": float(stats.spearmanr(ae, pw)[0]), "protocols": {}}

    # ---- the two reference points ----
    print("\n  REFERENCE POINTS (raw |EffectSize|, the target that failed)")
    R["naive"] = arm(X, ae, ch, "biology on raw |EffectSize|")
    R["with_power"] = arm(np.hstack([X, P]), ae, ch, "biology + power (the confound)")
    print(f"    -> the gap {R['with_power']['net'] - R['naive']['net']:+.4f} is what the assay contributes")

    # ---- P1 stratified ----
    q = np.quantile(pw, np.linspace(0, 1, NSTRAT + 1))
    st = np.clip(np.digitize(pw, q[1:-1]), 0, NSTRAT - 1)
    print(f"\n  P1 STRATIFIED -- score inside {NSTRAT} power strata, macro-average")
    per, wts = [], []
    for s_ in range(NSTRAT):
        m = st == s_
        if m.sum() < 60 or len(set(ch[m])) < 5:
            print(f"    stratum {s_}: n={int(m.sum())} too small to score")
            continue
        r2 = loco(X[m], ae[m], ch[m])
        pm = np.mean([loco(X[m], ae[m][rng.permutation(int(m.sum()))], ch[m]) for _ in range(3)])
        print(f"    stratum {s_}: n={int(m.sum()):4d}  power {pw[m].min():.3f}-{pw[m].max():.3f}  "
              f"R2 {r2:+.4f}  permuted {pm:+.4f}  net {r2-pm:+.4f}")
        per.append(r2 - pm); wts.append(int(m.sum()))
    if per:
        macro = float(np.mean(per))
        print(f"    MACRO-AVERAGED net across strata: {macro:+.4f}")
        R["protocols"]["P1 stratified"] = {"per_stratum_net": per, "n_per": wts, "macro_net": macro}

    # ---- P2 residualised ----
    print(f"\n  P2 RESIDUALISED -- predict |EffectSize| minus its power-decile mean")
    dec = np.clip(np.digitize(pw, np.quantile(pw, np.linspace(0, 1, 11))[1:-1]), 0, 9)
    resid = ae.copy()
    for b in range(10):
        m = dec == b
        if m.any():
            resid[m] = ae[m] - ae[m].mean()
    print(f"    residual sd {resid.std():.4f} vs raw sd {ae.std():.4f}; residual-vs-power Spearman "
          f"{stats.spearmanr(resid, pw)[0]:+.4f}")
    R["protocols"]["P2 residualised"] = arm(X, resid, ch, "biology on the power residual")
    R["protocols"]["P2 residualised"]["resid_vs_power"] = float(stats.spearmanr(resid, pw)[0])

    # ---- P3 rank within stratum ----
    print(f"\n  P3 RANK-WITHIN-STRATUM -- distribution-free")
    rk = np.zeros(len(ae))
    for s_ in range(NSTRAT):
        m = st == s_
        if m.sum() > 1:
            rk[m] = stats.rankdata(ae[m]) / m.sum()
    print(f"    rank-vs-power Spearman {stats.spearmanr(rk, pw)[0]:+.4f}")
    R["protocols"]["P3 rank-in-stratum"] = arm(X, rk, ch, "biology on within-stratum rank")
    R["protocols"]["P3 rank-in-stratum"]["rank_vs_power"] = float(stats.spearmanr(rk, pw)[0])

    # ---- does adding power still help under each protocol? the check that the fix worked ----
    print(f"\n  DID THE FIX WORK? adding power back should now buy ~nothing")
    for nm, t in (("P2 residualised", resid), ("P3 rank-in-stratum", rk)):
        a = arm(np.hstack([X, P]), t, ch, f"{nm}: biology + power", indent=4)
        base = R["protocols"][nm]["net"]
        print(f"        power still adds {a['net']-base:+.4f} under this protocol")
        R["protocols"][nm]["power_still_adds"] = a["net"] - base

    # ---- verdict ----
    print("\n" + "=" * 100)
    best = max(R["protocols"], key=lambda k: R["protocols"][k].get("macro_net",
                                                                   R["protocols"][k].get("net", -9)))
    bv = R["protocols"][best]
    bn = bv.get("macro_net", bv.get("net"))
    leak = max([abs(R["protocols"][k].get("power_still_adds", 0)) for k in R["protocols"]] or [0])
    if bn > 0.02:
        v = (f"MAGNITUDE IS BENCHMARKABLE ONCE POWER IS CONDITIONED OUT. Best protocol {best} gives net "
             f"{bn:+.4f} where the raw target gave {R['naive']['net']:+.4f}, and power adds at most "
             f"{leak:+.4f} on top. So the earlier null was partly a confound artefact and this protocol is "
             f"the target a later model should be scored on.")
    else:
        v = (f"MAGNITUDE IS NOT PREDICTABLE, CONFOUND OR NO CONFOUND. Every power-conditioned protocol still "
             f"gives net <= {bn:+.4f} (best {best}), against {R['naive']['net']:+.4f} raw. Conditioning "
             f"removes the assay signal -- power now adds at most {leak:+.4f} -- and nothing biological "
             f"appears in its place. That is a STRONGER negative than before: the +0.0881 was purely "
             f"detection power, and these features carry no information about how hard an element acts. The "
             f"protocol is still the deliverable: it is how a future model should be scored, and it now has "
             f"a measured floor of ~0.")
    R["verdict"] = v
    R["recommended_protocol"] = best
    print(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "power_conditioned.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'power_conditioned.json'}")


if __name__ == "__main__":
    main()
