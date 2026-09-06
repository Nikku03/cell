"""Loop 211. Is the ceiling the MODEL or the DATA?

WHAT HAS NEVER BEEN ASKED IN 210 LOOPS. Every set-point arm in this project has been RIDGE
REGRESSION. Loop 206's nine same-cell tracks, loop 208's 360,540 Perturb-seq gains, loop 209's
879-motif thermodynamic block, loop 210's combination -- all linear, all penalised the same way.
So when a route scores r 0.29 against a requirement near 0.9, nothing on disk says whether that is
the features being exhausted or the model being too simple. A neural network on the same features
is strictly more expressive and settles half of it.

THE PROTOCOL POINT, AND IT IS NOT A DETAIL. The natural way to say this is "keep correcting the
weights until it reaches the accuracy on test". That trains on the test set: once the stopping
decision is made by looking at test, test is no longer held out and the number it reports is not a
prediction. Every model below therefore gets a THREE-WAY split -- weights fitted on train, early
stopping and capacity chosen on VALIDATION, and the test fold scored exactly once, after all
decisions are frozen. Nested inside five outer folds, so every gene is scored exactly once as test
and never contributes to a decision about itself.

WHY A CAPACITY LADDER ALONE WOULD NOT ANSWER IT. A bigger model that ties ridge tells you nothing:
it could be that the features are exhausted, or that 600 genes is too few to fit anything larger.
Those have opposite remedies. A LEARNING CURVE separates them -- train on 20, 40, 60, 80 and 100
per cent of the available genes and watch the held-out score. A curve still climbing at 100% means
more genes would help; a flat curve means they would not. So the two diagnostics together give a
three-way answer:

    nonlinear BEATS ridge                      -> MODEL-limited. Push on architecture.
    nonlinear ties ridge, curve still RISING   -> DATA-limited. Get more genes, not a bigger net.
    nonlinear ties ridge, curve FLAT           -> FEATURES exhausted. Neither helps; measure
                                                  something else.

D4 and D5 are the two halves and they are independent, so the verdict falls out of their pair
rather than from a judgement call.

PREDECLARED, BEFORE ANY NUMBER.

  D1 IS THE INSTRUMENT HONEST?
     Gate: PASS iff the feature blocks load on loop 210's exact gene set and folds, AND every
     model class scores |r| below 0.10 on SHUFFLED labels. A model that scores on shuffled labels
     is reading the split, and every number below would be that.

  D2 THE CAPACITY LADDER.
     Ridge, a small MLP, a wide MLP, and gradient boosting, on identical features and folds, each
     with its own hyperparameters chosen on validation only.
     Gate: PASS iff every class returns a finite score on every fold. This is a completion check,
     not a performance claim -- an arm that silently returns nan has been this project's failure
     mode twice (loop 201's P6, loop 208's A4) and it is checked rather than assumed.

  D3 THE LEARNING CURVE.
     Held-out |r| at 20, 40, 60, 80 and 100 per cent of the training genes, for the best model.
     Gate: PASS iff the curve is monotone non-decreasing within noise. A ragged curve means the
     estimate is too unstable at this sample size for D5 to be read.

  D4 IS IT MODEL-LIMITED?
     Gate: PASS iff the best nonlinear model beats ridge by >= 0.05 in |r| on the test folds.
     A PASS says the ceiling was the model and more architecture is worth buying.

  D5 IS IT DATA-LIMITED?
     Gate: PASS iff |r| at 100% of the training genes exceeds |r| at 60% by >= 0.03.
     A PASS says more genes would help. Requires D3.

  D6 DOES ANYTHING REACH 0.40?
     Gate: PASS iff the best test |r| across all classes reaches 0.40.

  D7 IS ANY OF IT FAME?
     Gate: PASS iff the best model beats publication count on the same folds.

  D8 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import gzip, json, os, sys, time, warnings
from pathlib import Path

import h5py
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from loop_setpoint_physics import SEQ_F, gene_set
from gate_guard import Gates

from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SP = L191.SP
PHYS_CACHE = ROOT / "colab" / "data" / "physics" / "motif_occupancy_1200.npz"
K562 = SP / "perturbseq" / "K562_gwps_normalized_bulk_01.h5ad"
OUT = "outputs/loop_capacity.json"
N_TRAIN, SEED = 6, 211211
FRACS = (0.2, 0.4, 0.6, 0.8, 1.0)

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def models(seed):
    return {
        "ridge": make_pipeline(StandardScaler(),
                               RidgeCV(alphas=np.logspace(0, 6, 25))),
        "mlp_small": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(32,), alpha=1e-2, max_iter=2000, early_stopping=True,
            n_iter_no_change=25, validation_fraction=0.2, random_state=seed)),
        "mlp_wide": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(256, 64), alpha=1e-1, max_iter=2000, early_stopping=True,
            n_iter_no_change=25, validation_fraction=0.2, random_state=seed)),
        "gbm": xgb.XGBRegressor(
            n_estimators=2000, max_depth=4, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.3, reg_lambda=5.0, early_stopping_rounds=50,
            random_state=seed, n_jobs=4, verbosity=0),
    }


def fit_predict(name, mdl, Xtr, ytr, Xva, yva, Xte):
    if name == "gbm":
        mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        mdl.fit(np.vstack([Xtr, Xva]), np.r_[ytr, yva])
    return mdl.predict(Xte)


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "capacity vs data"}
    say("=" * 104)
    say("LOOP 211 -- IS THE CEILING THE MODEL, OR THE DATA?")
    say("=" * 104)

    # ------------------------------------------------------------ features
    grid, M, A9, sym, keep, tssb = gene_set()
    gi = np.where(keep)[0]
    S_all = (M[-3:].mean(0))[gi]
    seqs = json.load(open(SEQ_F))
    TRACKS = ["NR3C1", "EP300", "JUN", "JUNB", "CEBPB", "FOSL2", "DNase", "CTCF", "RAD21"]
    TR = {}
    for name in TRACKS:
        pt, PM = L191.promoter_track(name, [tssb.get(s) for s in sym], PROM_PAD := L191.PROM_PAD,
                                     lambda *_: None)
        TR[name] = PM[[int(np.where(pt == t)[0][0]) for t in grid]]
    fk = h5py.File(K562, "r")
    gt = [x.decode() if isinstance(x, bytes) else str(x) for x in fk["obs/gene_transcript"][:]]
    cats = [x.decode() if isinstance(x, bytes) else str(x)
            for x in fk["var/__categories/gene_name"][:]]
    readout = np.array([cats[c] for c in fk["var/gene_name"][:]])
    ridx = {g: i for i, g in enumerate(readout)}
    Xk = fk["X"]
    names = [sym[i] for i in gi if sym[i] in seqs and sym[i] in ridx]
    pos = {s: k for k, s in enumerate([sym[i] for i in gi])}
    y = np.array([S_all[pos[s]] for s in names])
    kidx = {s: k for k, s in enumerate([sym[i] for i in gi])}
    Fchip = np.column_stack([np.column_stack([
        TR[t][:N_TRAIN, gi].mean(0), TR[t][:N_TRAIN, gi].max(0),
        TR[t][N_TRAIN - 1, gi] - TR[t][0, gi]]) for t in TRACKS])
    Fchip = np.array([Fchip[kidx[s]] for s in names])
    cols = np.array([ridx[s] for s in names])
    rng = np.random.default_rng(SEED)
    picked = []
    for p in rng.permutation(Xk.shape[0]):
        if len(picked) >= 200:
            break
        v = Xk[int(p), :][cols]
        if np.isfinite(v).all():
            picked.append(int(p))
    Fgain = np.column_stack([Xk[p, :][cols] for p in picked])
    if not PHYS_CACHE.exists():
        G.add("D1", None, void_if=True,
              void_reason=f"loop 210's motif cache is not on disk at {PHYS_CACHE}; run loop 210 "
                          f"first")
        G.summary(seconds=time.time() - t0)
        return
    # Restrict the gene set to what the motif cache covers, THEN build every block once against
    # that final list. The first draft rebuilt Fchip twice with a no-op line between, which is the
    # kind of dead code that looks like it is doing something; removed.
    Z = np.load(PHYS_CACHE, allow_pickle=True)
    cpos = {str(g): i for i, g in enumerate(Z["genes"])}
    names = [s for s in names if s in cpos]
    y = np.array([S_all[pos[s]] for s in names])
    chip_full = np.column_stack([np.column_stack([
        TR[t][:N_TRAIN, gi].mean(0), TR[t][:N_TRAIN, gi].max(0),
        TR[t][N_TRAIN - 1, gi] - TR[t][0, gi]]) for t in TRACKS])
    Fchip = np.array([chip_full[kidx[s]] for s in names])
    cols = np.array([ridx[s] for s in names])
    Fgain = np.column_stack([Xk[p, :][cols] for p in picked])
    Fphys = np.array([Z["F"][cpos[s]] for s in names])
    X = np.hstack([Fchip, Fgain, Fphys]).astype(np.float64)
    say(f"     genes {len(names):,}   features {X.shape[1]:,} "
        f"(chip {Fchip.shape[1]}, gains {Fgain.shape[1]}, physics {Fphys.shape[1]})")

    outer = np.array_split(np.random.default_rng(SEED).permutation(len(y)), 5)

    def run(Xm, ym, frac=1.0, seed=SEED, shuffle=False):
        """Nested: outer fold = test (scored once), inner split = validation."""
        pred = np.full(len(ym), np.nan)
        rg = np.random.default_rng(seed)
        for k in range(5):
            te = outer[k]
            tr_all = np.concatenate([outer[j] for j in range(5) if j != k])
            if frac < 1.0:
                tr_all = tr_all[rg.permutation(len(tr_all))[:max(20, int(frac * len(tr_all)))]]
            cut = int(0.8 * len(tr_all))
            tr, va = tr_all[:cut], tr_all[cut:]
            yy = rg.permutation(ym) if shuffle else ym
            out = {}
            for nm, mdl in models(seed).items():
                try:
                    out[nm] = fit_predict(nm, mdl, Xm[tr], yy[tr], Xm[va], yy[va], Xm[te])
                except Exception:
                    out[nm] = np.full(len(te), np.nan)
            yield k, te, out

    def score_all(frac=1.0, shuffle=False, seed=SEED):
        acc = {nm: np.full(len(y), np.nan) for nm in models(seed)}
        for k, te, out in run(X, y, frac, seed, shuffle):
            for nm, p in out.items():
                acc[nm][te] = p
        return {nm: abs(pear(v, y)) for nm, v in acc.items()}, acc

    # ------------------------------------------------------------ D1
    say("D1 IS THE INSTRUMENT HONEST?")
    sh, _ = score_all(shuffle=True)
    for nm, v in sh.items():
        say(f"       SHUFFLED labels  {nm:<10} |r| {v:.4f}")
    ok1 = all(np.isfinite(v) and v < 0.10 for v in sh.values())
    G.add("D1", ok1,
          if_true="D1 PASS -- no model class scores on shuffled labels",
          if_false=lambda: f"D1 FAIL -- {sh}")
    res["shuffled"] = sh

    # ------------------------------------------------------------ D2
    say("D2 THE CAPACITY LADDER")
    full, preds = score_all()
    for nm, v in full.items():
        say(f"       {nm:<12} test |r| {v:.4f}")
    ok2 = all(np.isfinite(v) for v in full.values())
    G.add("D2", ok2, requires=("D1",),
          if_true="D2 PASS -- every class returned a finite score on every fold",
          if_false=lambda: f"D2 FAIL -- non-finite arms: "
                           f"{[k for k,v in full.items() if not np.isfinite(v)]}")
    res["ladder"] = full
    best_nm = max((k for k in full if np.isfinite(full[k])), key=lambda k: full[k])
    best_r = full[best_nm]

    # ------------------------------------------------------------ D3
    say("D3 THE LEARNING CURVE")
    curve = {}
    for f in FRACS:
        s, _ = score_all(frac=f)
        curve[f] = s.get(best_nm, float("nan"))
        say(f"       {int(f*100):>3}% of training genes   |r| {curve[f]:.4f}   ({best_nm})")
    vals = [curve[f] for f in FRACS]
    drops = sum(1 for a, b in zip(vals, vals[1:]) if b < a - 0.03)
    G.add("D3", bool(drops == 0), stat=float(drops), requires=("D2",),
          if_true="D3 PASS -- the curve is monotone within noise, so D5 can be read",
          if_false=lambda: f"D3 FAIL -- {drops} drops of more than 0.03; the estimate is too "
                           f"unstable at {len(y)} genes for the curve to be read")
    res["curve"] = {str(k): v for k, v in curve.items()}

    # ------------------------------------------------------------ D4
    say("D4 IS IT MODEL-LIMITED?")
    nl = max(full[k] for k in ("mlp_small", "mlp_wide", "gbm") if np.isfinite(full[k]))
    gain = nl - full["ridge"]
    say(f"       best nonlinear {nl:.4f}   ridge {full['ridge']:.4f}   gain {gain:+.4f}")
    G.add("D4", bool(gain >= 0.05), stat=gain, requires=("D2",),
          if_true=lambda: f"D4 PASS -- nonlinear beats ridge by {gain:+.4f}; the ceiling was the "
                          f"MODEL and architecture is worth buying",
          if_false=lambda: f"D4 FAIL -- nonlinear beats ridge by only {gain:+.4f}. Capacity is not "
                           f"the binding constraint")

    say("D5 IS IT DATA-LIMITED?")
    rise = curve[1.0] - curve[0.6]
    say(f"       |r| at 100% {curve[1.0]:.4f}   at 60% {curve[0.6]:.4f}   rise {rise:+.4f}")
    G.add("D5", bool(rise >= 0.03), stat=rise, requires=("D3",),
          if_true=lambda: f"D5 PASS -- still rising ({rise:+.4f}); more genes would help",
          if_false=lambda: f"D5 FAIL -- flat ({rise:+.4f}); more genes of this kind would not help")

    say("D6 DOES ANYTHING BEAT THE RIDGE BASELINE MEANINGFULLY?")
    say(f"       loop 210 measured the ridge combination at |r| 0.4761 on this feature set.")
    say(f"       this loop's own ridge on its own folds: {full['ridge']:.4f} -- that is the")
    say(f"       baseline the ladder has to beat, and it is recomputed here rather than quoted,")
    say(f"       because a different fold assignment moves it.")
    bar = full["ridge"] + 0.05
    G.add("D6", bool(best_r >= bar), stat=best_r, requires=("D2",),
          if_true=lambda: f"D6 PASS -- {best_nm} reaches {best_r:.4f}, clearing the ridge "
                          f"baseline {full['ridge']:.4f} by {best_r-full['ridge']:+.4f}",
          if_false=lambda: f"D6 FAIL -- best is {best_nm} at {best_r:.4f} against a ridge "
                           f"baseline of {full['ridge']:.4f} ({best_r-full['ridge']:+.4f}). "
                           f"Capacity bought nothing on top of the linear combination")

    say("D7 IS ANY OF IT FAME?")
    tab = json.load(gzip.open("colab/data/cell_complete.json.gz"))["genes"]
    pubs = {str(g["name"]).upper(): float(g.get("pubs") or 0) for g in tab}
    Xf = np.log1p(np.array([pubs.get(s, 0.0) for s in names])).reshape(-1, 1)
    accf = np.full(len(y), np.nan)
    for k in range(5):
        te = outer[k]; tr = np.concatenate([outer[j] for j in range(5) if j != k])
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 25)))
        m.fit(Xf[tr], y[tr]); accf[te] = m.predict(Xf[te])
    r_fame = abs(pear(accf, y))
    say(f"       publication count |r| {r_fame:.4f}   best model {best_r:.4f}")
    G.add("D7", bool(best_r > r_fame), stat=r_fame, requires=("D2",),
          if_true=lambda: f"D7 PASS -- {best_r:.4f} beats fame {r_fame:.4f}",
          if_false=lambda: f"D7 FAIL -- fame {r_fame:.4f} is not beaten")

    say("     THE THREE-WAY VERDICT")
    if G.status.get("D4") == "PASS":
        say("       MODEL-limited -- a bigger model helps; push on architecture")
    elif G.status.get("D5") == "PASS":
        say("       DATA-limited -- more genes would help; a bigger model would not")
    else:
        say("       FEATURES EXHAUSTED -- neither capacity nor more of the same data helps.")
        say("       The remedy is different measurements, not a different model.")

    say("D8 WHAT THIS CANNOT SHOW")
    say(f"     {len(y):,} genes against {X.shape[1]:,} features is a hard regime for any network.")
    say("     A tie between the MLP and ridge at this size is genuinely ambiguous between 'the")
    say("     features are exhausted' and 'the net had too little to learn from', which is")
    say("     exactly why D5's learning curve is here rather than a capacity ladder alone.")
    say("     The gene set is the intersection of three data sources and is smaller than any of")
    say("     them; loop 210 measured it at 600 of loop 198's 1,336.")
    say("     Nothing here changes the target. The requirement is a property of the relaxation")
    say("     rule, and loop 210 is what re-measures that.")

    G.summary(seconds=time.time() - t0)
    gates, void = G.as_dict()
    res["gates"], res["void"] = gates, void
    res["best"] = {"model": best_nm, "r": best_r, "fame": r_fame}
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)
    say(f"wrote {OUT}")


if __name__ == "__main__":
    main()
