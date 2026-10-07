"""mc_outofgame -- Monte Carlo for "is this gene out of the game?", built only where sampling can actually add something.

THERE ARE TWO DIFFERENT MONTE CARLOS ON OFFER AND ONLY ONE IS WORTH RUNNING. This module builds the second and PROVES
the first is decorative rather than asserting it, because the first is the one that sounds most appealing.

  MC-1, MOLECULAR NOISE (the appealing one). Simulate transcription and translation as a stochastic birth-death
  process, get the distribution of protein copies per cell, read off the fraction of cells below a functional
  threshold. It is the textbook move and it is useless HERE for a reason that is arithmetic, not biological:
  P(copies < threshold) is a MONOTONE DECREASING function of the mean copies remaining, and every ranking metric in
  this project -- Spearman, AUPRC, precision@k, recall@k -- is invariant under monotone transforms of the score. So an
  elaborate stochastic simulation ranks knockouts and genes EXACTLY as the one-line deterministic mean does. Part A
  measures that equivalence instead of trusting the argument.

  And the mean it is equivalent to already loses. Measured on 1,482 knockouts against the number of specific movers:
  fraction-remaining x expression level scores Spearman +0.110, while EXPRESSION LEVEL ALONE scores +0.146. Knockdown
  DEPTH alone scores -0.063, and once expression level is partialled out its sign FLIPS to +0.132 -- so the marginal
  depth effect was confounded, not weak. The whole simulation is bounded by a quantity that loses to one of its inputs.

  Its parameters are also not there. A two-stage model needs k_syn, k_mdeg, k_transl, k_pdeg. Only k_mdeg is measured
  (RNAdecayCafe). k_syn is derived and agrees with a direct TT-seq measurement at only r = 0.361 (rate_ground_truth).
  NO human protein degradation rate exists on disk at all, and k_transl is not merely missing but NON-IDENTIFIABLE:
  steady state gives only the ratio k_transl/k_pdeg. Nor can bursting be fitted -- the only candidate variance field,
  gwps var/clean_cv, has a Fano factor of 0.012, below the 1.0 that any birth-death process must exceed, so it is a
  spread across pseudobulk knockout profiles rather than across cells. A simulation with invented parameters cannot be
  validated, and this repo already owns one: cell_sim's Gillespie model reports std_mcc = 0.0 across three seeds,
  meaning its answers come from hand-added priors, not from the stochastic dynamics it runs.

  MC-2, UNCERTAINTY OVER WHAT WE DO NOT KNOW (built here). The chain in markov_outofgame must commit to things nobody
  measured: which regulatory edges are real, what sign the unsigned 91% carry, how many steps to propagate, and how
  deep the knockdown truly went. Sampling THOSE turns a point prediction into a PROBABILITY -- and probability is
  exactly what rank-invariance does not flatten, since two models can rank identically and calibrate very differently.

THE CONTROL THAT DECIDES IT. A Monte Carlo is only worth its cost if it beats simply CALIBRATING THE DETERMINISTIC
SCORE, so the deterministic chain is Platt-scaled on training knockouts and scored identically. If sampling does not
beat that, the uncertainty was already implicit in the point estimate. Second baseline: generic responsiveness, which
is a calibrated probability BY CONSTRUCTION and therefore very hard to beat on Brier score -- and which already beat
the mechanism by nine sigma in markov_outofgame. Both scored on held-out knockouts."""
import json, collections
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import os
from scipy import sparse

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
T = 4.2
TIDE_FRAC = 0.05
OUT_THR = 0.05
NREP = 60               # Monte Carlo replicates
NKO_MC = 500            # knockouts carried through the MC (cost scales linearly)
UNSIGNED_P_ACT = 0.6
EDGE_KEEP = 0.8


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def ece(p, y, nbin=12):
    """Expected calibration error over QUANTILE bins -- equal-width bins are useless when nearly every prediction
    sits near zero, which is the case at a 1.6e-04 base rate.

    Returns nan when the predictor is so concentrated that fewer than three distinct quantile edges exist. That is a
    PROPERTY OF THE PREDICTOR, not a failure, and it must be reported as such: an earlier version printed a bare
    'nan' next to a rival's 0.2514 as though the rival had lost."""
    qs = np.unique(np.quantile(p, np.linspace(0, 1, nbin + 1)))
    if len(qs) < 3:
        return float("nan"), []
    b = np.clip(np.digitize(p, qs[1:-1]), 0, len(qs) - 2)
    tot = 0.0; curve = []
    for k in range(len(qs) - 1):
        m = b == k
        if m.sum() < 10:
            continue
        pm, ym = float(p[m].mean()), float(y[m].mean())
        tot += m.mean() * abs(pm - ym)
        curve.append({"bin": k, "n": int(m.sum()), "pred": pm, "obs": ym})
    return float(tot), curve


def fmt(x):
    return "n/a (too concentrated to bin)" if not np.isfinite(x) else f"{x:.4f}"


def auprc(scores, labels):
    o = np.argsort(-scores); y = labels[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(y.sum(), 1)
    ap = 0.0; pr = 0.0
    for i in np.where(y > 0)[0]:
        ap += prec[i] * (rec[i] - pr); pr = rec[i]
    return float(ap)


def part_a_rank_equivalence(seed_rem, level, rng):
    """MEASURE the claim that a mean-field stochastic model cannot out-rank its own mean.

    Two-stage steady state: mRNA ~ Poisson(lam) with lam proportional to level x fraction remaining; protein tracks
    mRNA. P(out of game) = P(copies <= threshold), SIMULATED, so the equivalence is shown on samples rather than
    argued from the formula. The copy scale and threshold are illustrative BY NECESSITY -- which is exactly why the
    result is reported as a rank equivalence, a statement invariant to both."""
    from scipy.stats import spearmanr
    SCALE = 40.0
    lam = np.clip(level * seed_rem * SCALE, 1e-3, None)
    nsim = 400
    p_below = np.array([(rng.poisson(l, nsim) <= 3.0).mean() for l in lam])
    p2 = np.array([(rng.poisson(l, nsim) <= 12.0).mean() for l in lam])
    return {"spearman_MCprob_vs_mean": float(spearmanr(p_below, lam)[0]),
            "spearman_MCprob_vs_deterministic": float(spearmanr(p_below, seed_rem * level)[0]),
            "spearman_across_two_thresholds": float(spearmanr(p_below, p2)[0]), "n": int(len(lam))}


def main():
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression

    df = pd.read_parquet(SP / "repl_k562_zscores.parquet")
    kos = [str(k) for k in df.index]; genes = [str(g) for g in df.columns]
    Z = df.values.astype(np.float32)
    ok = ~((np.abs(Z) >= T).mean(0) >= TIDE_FRAC)
    f = h5py.File(SP / "gwps.h5ad", "r")
    dec = lambda x: x.decode() if isinstance(x, bytes) else str(x)
    cats = [dec(x) for x in f["var"]["__categories"]["gene_name"][:]]
    vn = [cats[c] for c in f["var"]["gene_name"][:]]
    vidx = {}
    for i, n in enumerate(vn):
        vidx.setdefault(n, i)
    gt = [dec(x) for x in f["obs"]["gene_transcript"][:]]
    rowof = {}
    for i, t in enumerate(gt):
        p = t.split("_")
        if len(p) >= 2:
            rowof.setdefault(p[1], i)
    gcol = np.array([vidx.get(g, -1) for g in genes]); have = gcol >= 0
    cmean = np.asarray(f["var"]["clean_mean"][:], float)

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; nidx = {n: i for i, n in enumerate(names)}; N = len(names)
    E = []
    for e in D.get("reg") or []:
        try:
            a, b = int(e[0]), int(e[1]); s = int(e[2]) if len(e) > 2 else 0
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            E.append((a, b, s))
    Er = np.array(E, dtype=np.int32)
    print(f"{len(Er)} regulatory edges; {N} network nodes", flush=True)

    X = f["X"]
    gmask = have & ok & np.array([g in nidx for g in genes])
    gcols = np.where(gmask)[0]; gnet = np.array([nidx[genes[j]] for j in gcols])
    kidx = [i for i, k in enumerate(kos) if k in rowof and k in nidx]
    rng = np.random.RandomState(0)
    kidx = list(rng.permutation(kidx))[:NKO_MC]

    seeds, Y, lev = [], [], []
    for i in kidx:
        k = kos[i]
        xr = np.asarray(X[rowof[k]], dtype=np.float32)
        rem = 1.0 + xr[gcol[gcols]]
        j = vidx.get(k, -1)
        sr = 1.0 + float(xr[j]) if j >= 0 and np.isfinite(xr[j]) else np.nan
        if not np.isfinite(sr):
            continue
        seeds.append((nidx[k], float(np.clip(sr, 0.0, 1.0))))
        Y.append((Z[i, gcols] <= -T) & (rem <= OUT_THR))
        lev.append(float(cmean[j]) if j >= 0 else np.nan)
    Y = np.array(Y); K = len(Y)
    sn = np.array([s[0] for s in seeds]); sl = np.array([s[1] for s in seeds], np.float32)
    lev = np.array(lev)
    print(f"{K} knockouts x {len(gcols)} genes; {int(Y.sum())} out-of-game positives, "
          f"base rate {Y.mean():.3e}\n", flush=True)

    print("  PART A -- can a molecular-noise Monte Carlo out-RANK its own mean? Measured, not assumed.")
    a = part_a_rank_equivalence(sl, np.nan_to_num(lev, nan=float(np.nanmedian(lev))), rng)
    print(f"    Spearman(MC P(out of game), mean copies)           = {a['spearman_MCprob_vs_mean']:+.4f}")
    print(f"    Spearman(MC P(out of game), deterministic rem*lvl) = {a['spearman_MCprob_vs_deterministic']:+.4f}")
    print(f"    Spearman(MC at threshold 3, MC at threshold 12)    = {a['spearman_across_two_thresholds']:+.4f}")
    print(f"    -> the simulation reproduces the ordering of its own mean, and the arbitrary threshold does not")
    print(f"       change that ordering either. For ANY rank-based metric the simulation is decorative.\n", flush=True)

    def build_P(keep, sgn_unsigned):
        vals = np.where(Er[keep, 2] > 0, 1.0, np.where(Er[keep, 2] < 0, -1.0, sgn_unsigned))
        A = sparse.coo_matrix((vals.astype(np.float32), (Er[keep, 0], Er[keep, 1])), shape=(N, N)).tocsr()
        mag = np.asarray(abs(A).sum(1)).ravel()
        inv = np.zeros(N, np.float32); nz = mag > 0; inv[nz] = 1.0 / mag[nz]
        return sparse.diags(inv) @ A

    def chain(Pm, steps, alpha, seed_loss):
        L0 = np.zeros((N, K), np.float32); L0[sn, np.arange(K)] = seed_loss
        PT = Pm.T.tocsr(); cur = L0; tot = np.zeros_like(L0)
        for kk in range(1, steps + 1):
            cur = PT @ cur
            np.clip(cur, -1.0, 1.0, out=cur)
            tot = tot + (alpha ** kk) * cur
        return tot[gnet, :].T

    print(f"  PART B -- Monte Carlo over what the chain has to guess: which edges are real (keep {EDGE_KEEP:.0%} per")
    print(f"           replicate), the sign of the unsigned edges, the depth, the damping, and the true knockdown")
    print(f"           depth. {NREP} replicates over {K} knockouts.", flush=True)
    acc = np.zeros((K, len(gcols)), np.float32)
    for r in range(NREP):
        keep = rng.rand(len(Er)) < EDGE_KEEP
        su = np.where(rng.rand(int(keep.sum())) < UNSIGNED_P_ACT, 1.0, -1.0) * 0.5
        Pm = build_P(keep, su)
        steps = int(rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20]))
        alpha = float(rng.uniform(0.2, 0.7))
        sj = np.clip(sl + rng.normal(0, 0.08, K), 0.0, 1.0).astype(np.float32)
        S = chain(Pm, steps, alpha, sj)
        cut = np.quantile(S, 1.0 - 10.0 ** rng.uniform(-3.2, -2.2))
        acc += (S >= cut)
        if (r + 1) % 15 == 0:
            print(f"    replicate {r+1}/{NREP}", flush=True)
    Pmc = acc / NREP

    Pdet = build_P(np.ones(len(Er), bool), np.full(len(Er), 0.5, np.float32))
    Sdet = chain(Pdet, 2, 0.5, sl)

    rows = []
    for fi, (tr, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros(K), np.zeros(K), np.arange(K))):
        yte = Y[te].ravel().astype(float)
        gen = Y[tr].mean(0)
        Pgen = np.tile(gen, (len(te), 1)).ravel()
        lr = LogisticRegression(max_iter=400).fit(Sdet[tr].ravel().reshape(-1, 1), Y[tr].ravel().astype(int))
        Pplatt = lr.predict_proba(Sdet[te].ravel().reshape(-1, 1))[:, 1]
        pmc = Pmc[te].ravel()
        # THE RAW MC IS MISCALIBRATED BY CONSTRUCTION AND THAT IS MY KNOB, NOT THE METHOD'S FAULT. Each replicate
        # calls positive above a sampled quantile drawn from 10^-3.2..10^-2.2, i.e. a call rate one to two orders of
        # magnitude ABOVE the 2.4e-04 base rate, so the averaged probabilities sit near 6e-02 and Brier is dominated
        # by that prior. Comparing raw MC against a PLATT-SCALED rival would be measuring my threshold prior, not
        # whether sampling helps. So the MC gets the same one-parameter calibration its rival gets, fitted on train.
        lrm = LogisticRegression(max_iter=400).fit(Pmc[tr].ravel().reshape(-1, 1), Y[tr].ravel().astype(int))
        pmc_cal = lrm.predict_proba(Pmc[te].ravel().reshape(-1, 1))[:, 1]
        e_cal, _ = ece(pmc_cal, yte)
        e_mc, curve = ece(pmc, yte); e_pl, _ = ece(Pplatt, yte); e_gen, _ = ece(Pgen, yte)
        rows.append({"fold": fi, "n_pos": int(yte.sum()),
                     "brier_mc": brier(pmc, yte), "brier_mc_calibrated": brier(pmc_cal, yte),
                     "brier_platt": brier(Pplatt, yte),
                     "brier_generic": brier(Pgen, yte), "brier_base": brier(np.full_like(yte, yte.mean()), yte),
                     "ece_mc": e_mc, "ece_mc_calibrated": e_cal, "ece_platt": e_pl, "ece_generic": e_gen,
                     "auprc_mc": auprc(pmc, yte.astype(int)),
                     "auprc_mc_calibrated": auprc(pmc_cal, yte.astype(int)),
                     "auprc_platt": auprc(Pplatt, yte.astype(int)),
                     "auprc_generic": auprc(Pgen, yte.astype(int)),
                     "curve": curve if fi == 0 else None})
        print(f"    fold {fi}: Brier MC {rows[-1]['brier_mc']:.3e}  Platt {rows[-1]['brier_platt']:.3e}  "
              f"generic {rows[-1]['brier_generic']:.3e}", flush=True)

    agg = {k: float(np.mean([r[k] for r in rows])) for k in
           ["brier_mc", "brier_mc_calibrated", "brier_platt", "brier_generic", "brier_base",
            "ece_mc", "ece_mc_calibrated", "ece_platt", "ece_generic",
            "auprc_mc", "auprc_mc_calibrated", "auprc_platt", "auprc_generic"]}
    print(f"\n  HELD-OUT ({K} knockouts, 5 folds). Lower Brier/ECE better; higher AUPRC better.")
    print(f"    {'model':28s} {'Brier':>11s} {'ECE':>9s} {'AUPRC':>9s}")
    for lab, b, e, ap in [("MC raw (uncalibrated)", "brier_mc", "ece_mc", "auprc_mc"),
                          ("MC + same 1-param calibration", "brier_mc_calibrated", "ece_mc_calibrated",
                           "auprc_mc_calibrated"),
                          ("Platt-scaled deterministic", "brier_platt", "ece_platt", "auprc_platt"),
                          ("generic responsiveness", "brier_generic", "ece_generic", "auprc_generic")]:
        print(f"    {lab:30s} {agg[b]:11.4e} {fmt(agg[e]):>28s} {agg[ap]:9.4f}")
    print(f"    {'predict the base rate':30s} {agg['brier_base']:11.4e} {'-':>28s} {'-':>9s}")
    print(f"\n    ECE is n/a where a predictor is too concentrated for 3 distinct quantile edges -- a property of")
    print(f"    that predictor, not a loss. AUPRC is threshold-free and is the comparison that means something here.")

    mc_beats_platt = agg["brier_mc_calibrated"] < agg["brier_platt"]
    mc_beats_generic = agg["auprc_mc_calibrated"] > agg["auprc_generic"]

    verdict = (
        "MONTE CARLO FOR 'OUT OF THE GAME', built only where sampling can add something. "
        "PART A settles the appealing version first. A molecular-noise simulation -- birth-death mRNA and protein, read "
        "off the fraction of cells below a functional threshold -- CANNOT out-rank its own mean, because "
        "P(copies < threshold) is a monotone function of that mean and every ranking metric here is invariant under "
        f"monotone transforms. Measured rather than asserted: Spearman between the simulated probability and the mean is "
        f"{a['spearman_MCprob_vs_mean']:+.4f}, against the deterministic remaining x level product "
        f"{a['spearman_MCprob_vs_deterministic']:+.4f}, and between two very different functional thresholds "
        f"{a['spearman_across_two_thresholds']:+.4f}. The simulation is therefore decorative for AUPRC, precision@k and "
        "Spearman alike. It is also bounded by a losing quantity: on 1,482 knockouts, fraction-remaining x expression "
        "level scores Spearman +0.110 against the number of specific movers while expression level ALONE scores +0.146, "
        "and knockdown depth alone scores -0.063 whose sign FLIPS to +0.132 once expression is partialled out, meaning "
        "the marginal depth effect was confounded rather than weak. And its parameters are absent: only k_mdeg is "
        "measured, k_syn agrees with direct TT-seq at r = 0.361, no human protein degradation rate is on disk at all, "
        "k_transl is non-identifiable from steady state, and bursting cannot be fitted because the only candidate "
        "variance field has a Fano factor of 0.012 -- below the 1.0 any birth-death process must exceed. "
        "PART B builds the version that can pay: sampling over which edges are real, the sign of the unsigned 91%, the "
        "propagation depth, the damping, and the true knockdown depth, turning a point prediction into a probability. "
        f"A REPORTING TRAP HAD TO BE DISARMED FIRST: the raw MC is miscalibrated BY CONSTRUCTION, because each replicate "
        f"calls positive above a sampled quantile drawn from 10^-3.2..10^-2.2 -- a call rate one to two orders of "
        f"magnitude above the {float(Y.mean()):.1e} base rate -- so its raw Brier ({agg['brier_mc']:.3e}) measures that "
        f"threshold prior, which is my knob, and not whether sampling helps. Giving the MC the SAME one-parameter "
        f"calibration its rival gets makes the comparison like-for-like. "
        f"RESULT on held-out knockouts: Brier {agg['brier_mc_calibrated']:.3e} for the calibrated Monte Carlo versus "
        f"{agg['brier_platt']:.3e} for the Platt-scaled deterministic chain and {agg['brier_generic']:.3e} for generic "
        f"responsiveness (predicting the base rate alone scores {agg['brier_base']:.3e}, so Brier barely separates "
        f"anything at this sparsity). The threshold-free comparison is the informative one: AUPRC "
        f"{agg['auprc_mc_calibrated']:.4f} for the MC, {agg['auprc_platt']:.4f} for the deterministic chain, and "
        f"{agg['auprc_generic']:.4f} for generic responsiveness. "
        + ("SAMPLING EARNS ITS COST on calibration: even matched one-for-one on post-hoc calibration, the Monte Carlo "
           "beats the deterministic chain, so the uncertainty was not already implicit in the point estimate. "
           if mc_beats_platt else
           "SAMPLING DOES NOT EARN ITS COST: matched one-for-one on calibration, the single deterministic chain does as "
           "well or better, so the uncertainty the Monte Carlo samples over was already implicit in the point estimate "
           "and 60 replicates bought nothing a one-parameter squashing function did not. ")
        + ("It also beats generic responsiveness, a hard baseline because that null is calibrated by construction. "
           if mc_beats_generic else
           "It does NOT beat generic responsiveness -- the per-gene rate of being driven out of the game, fitted on "
           "training knockouts, calibrated by construction, and already nine sigma ahead of the mechanism in "
           "markov_outofgame. That remains the thing to beat. ")
        + "WHAT THIS CANNOT SAY: 'out of the game' is defined transcriptionally at <= 5% of control mRNA, which is not "
        "protein loss or functional failure; knockdown_dose established this screen cannot locate the level at which a "
        "gene stops working, so the threshold is a definition, not a finding. Part A's copy scale (40 per normalised "
        "expression unit) and functional threshold are illustrative, which is precisely why its result is reported as a "
        "rank equivalence -- a statement invariant to both. Deterministic given seed 0.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"n_knockouts": K, "n_genes": int(len(gcols)), "n_positives": int(Y.sum()),
               "base_rate": float(Y.mean()), "n_replicates": NREP,
               "part_a_rank_equivalence": a, "held_out": agg, "folds": rows,
               "mc_beats_platt": bool(mc_beats_platt), "mc_beats_generic": bool(mc_beats_generic),
               "verdict": verdict, "note": verdict}, open(OUT / "mc_outofgame.json", "w"), indent=1)
    print("\n  -> outputs/orphan/mc_outofgame.json")


if __name__ == "__main__":
    main()
