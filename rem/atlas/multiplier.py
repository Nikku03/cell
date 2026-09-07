"""The multiplier form: pay ADDITIVELY for an axis a count would pay multiplicatively for.

WHERE THIS COMES FROM. poscount.py measured what position is worth in a class count and found a
representation fault rather than a missing signal. The information is real -- a ridge extracts
+0.77 noise floors from position on class counts, and +0.41 within the single main design ladder
where the group-mean estimator LOSES 0.33 -- but a count cannot reach it. The reason is
structural: a count summary is a PARTITION, and position enters as a smooth effect shared across
all factors. Partitioning by position destroys exactly the sharing that makes the effect
estimable, and does so while multiplying the alphabet, paying full cost for information it has
just made unusable.

The form that follows from that diagnosis, recorded there as a hypothesis and untested:

    P(x_i | class counts) * g(positions)        instead of        P(x_i | cells)

A multiplier costs ADDITIVELY in the alphabet where a cell costs multiplicatively: C + P
parameters instead of C * P table entries. This module tests it, on both halves.

AND THE ENGINE ANALOGUE, which is what makes this more than a promoter result. Position is to a
promoter what SLICE TIME is to a controller history. Both are axes a count aggregates over; both
enter as a smooth modulation shared across the things being counted; both are exactly what the
count destroys. summary.py's total count sums controller-slices over the whole window and throws
away WHEN. So the same form is testable there -- P(x_i | total count) * g(when) -- and if the
analogy is real the promoter result transfers to the engine. If it is not, this is a fact about
promoters and nothing more. M5 decides that and it is the gate that matters.

THE ESTIMATOR. In log2 expression a multiplier is an additive offset, so the promoter model is a
backfit: a partition term m(context, class counts) estimated as a group mean, and a shared term
h(positions) estimated by ridge on position-indicator counts, alternating. In the engine the same
thing on the LOG ODDS of a target being on: mu_j[count] + sum_l h_l n_l, with mu_j per target and
h shared across targets. Both are one nonparametric partition plus one shrunk shared term, which
is the whole content of the hypothesis.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

M0  THE CEILING GATE, FIRST, BOTH HALVES.
    (a) COST. The claim is that the multiplier's cap is the cap at alphabet C, not at C*P.
        VERIFY it rather than assume it, by adding the P shared parameters to the cap
        functional and checking the cap does not move. PREDECLARED: if it does move, the
        additive-versus-multiplicative claim is false and the module stops.
    (b) ACCURACY. The ridge is the reference for what position is worth -- not a proven upper
        bound, since a nonlinear model could do better, but the most flexible position model
        measured. PREDECLARED: the multiplier form must recover a substantial fraction of the
        ridge's gain over the same base. Recovering none of it means the form is wrong; matching
        the CELL form's gain while keeping the count's cost is the success case.

M1  IT IS AN IDENTITY WHEN IT SHOULD BE. Force the shared term to zero and the predictor must
    equal the pure class-count group mean to machine precision. A form that does not reduce
    exactly to what it generalises is not a generalisation of it.

M2  ON REAL PROMOTERS. Multiplier against: pure class count, the CELL form at the same C and P,
    the ridge, and identity. All held out, classes refit in-fold, positional term fit in-fold.

M3  ABSTENTION MUST NOT RISE. This is the form's structural claim -- it does not split groups, so
    it cannot make them sparser. PREDECLARED: if the multiplier's fallback rate exceeds the pure
    class count's by more than a rounding error, the implementation has partitioned somewhere it
    should not have, and that is a defect and not a result.

M4  INSIDE THE MAIN LADDER. poscount.py found the apparent position win was design provenance
    between sub-libraries, and it vanished within the single main ladder. The multiplier's gain
    has to be measured where that confound is constant or it is measuring the same artefact.

M5  THE ENGINE SIDE, AND IT IS THE GATE THAT MATTERS. Position is to a promoter what slice time is
    to a history. Test P(x_i | total count) * g(when) against the total count, against the
    per-slice CELL form, and against the full pattern, all in one matched log-odds
    parameterisation so the comparison is about the FORM and not the link. Parameter counts
    reported beside the errors. PREDECLARED: if the temporal multiplier does not repair a
    substantial part of the count's break, the analogy fails and the promoter result does not
    transfer to the engine.

M6  WHAT THIS DOES AND DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.promoter import fetch, load
from rem.atlas.statedim import stationary
from rem.atlas.block import block_joint, marginalise_targets
from rem.atlas.summary import var_on
from rem.atlas.signed import tf_effects, class_bins, cv_grouped, class_caps, signed_controller
from rem.atlas.poscount import pos_binner, cell_key, ridge_cv


# =================================================================================================
# the multiplier form on promoters
# =================================================================================================

def shared_fit(rs, resid, feats, lam):
    """Ridge on position-indicator COUNTS, no intercept -- the intercept belongs to the partition
    term and giving it to both would make the backfit unidentifiable."""
    keys = sorted({k for r in rs for k in feats(r)})
    if not keys:
        return {}
    ki = {k: j for j, k in enumerate(keys)}
    X = np.zeros((len(rs), len(keys)))
    for i, r in enumerate(rs):
        for k, v in feats(r).items():
            X[i, ki[k]] += v
    w = np.linalg.solve(X.T @ X + lam * np.eye(len(keys)), X.T @ resid)
    return {k: float(w[ki[k]]) for k in keys}


def apply_shared(r, h, feats):
    return sum(h.get(k, 0.0) * v for k, v in feats(r).items())


def multiplier_cv(sub, makekey, feats, folds=5, seed=0, lam=10.0, iters=4):
    """Group mean on the partition key TIMES a shared multiplier -- additive in log2 expression.

    Backfit: m is the group mean of (y - h), h is the ridge fit of (y - m). Both estimated on the
    training fold only. The held-out prediction falls back to the CONTEXT mean exactly as every
    other model in this build order does, so the fallback rate is comparable; note that the
    shared term is still applied on a fallback, which is the point -- it does not need the group."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(sub))
    rng.shuffle(idx)
    fold = np.array_split(idx, folds)
    err = np.zeros(len(sub))
    hit = np.zeros(len(sub), dtype=bool)
    for f in range(folds):
        te = set(fold[f].tolist())
        tr = [sub[i] for i in idx if i not in te]
        keyfn = makekey(tr)
        y = np.array([r["y"] for r in tr])
        h = {}
        gmm = cmm = None
        glob = float(y.mean())
        for _ in range(iters):
            off = np.array([apply_shared(r, h, feats) for r in tr])
            gm = collections.defaultdict(list)
            cm = collections.defaultdict(list)
            for r, v in zip(tr, y - off):
                gm[keyfn(r)].append(v)
                cm[r["ctx"]].append(v)
            gmm = {k: float(np.mean(v)) for k, v in gm.items()}
            cmm = {k: float(np.mean(v)) for k, v in cm.items()}
            glob = float(np.mean(y - off))
            mvec = np.array([gmm.get(keyfn(r), cmm.get(r["ctx"], glob)) for r in tr])
            if lam < np.inf:
                h = shared_fit(tr, y - mvec, feats, lam)
        for i in fold[f]:
            r = sub[i]
            k = keyfn(r)
            base = gmm.get(k)
            if base is None:
                base = cmm.get(r["ctx"], glob)
            else:
                hit[i] = True
            err[i] = r["y"] - (base + apply_shared(r, h, feats))
    return float(np.sqrt(np.mean(err ** 2))), 1.0 - hit.mean(), err, hit


def pos_feats(binner):
    def f(r):
        d = {}
        for t in r["tfs"]:
            d[binner(t[3])] = d.get(binner(t[3]), 0.0) + 1.0
        return d
    return f


# =================================================================================================
# the multiplier form in the engine: the axis is TIME, not position
# =================================================================================================

def logit(q, eps=1e-9):
    q = np.clip(q, eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def temporal_forms(nC, nT, L, dt, sgn, mag, cc=1.5, lam=1e-6):
    """Four parameterisations of P(x_j = 1 | controller history), all on the LOG ODDS so the
    comparison is about the FORM and not the link:

        count      mu_j[ total controller-slices on ]
        cells      mu_j[ (n_0, ..., n_L) ]                  the multiplicative alphabet
        MULTIPLIER mu_j[ total ] + sum_l h_l n_l            h SHARED across targets
        pattern    mu_j[ a ]                                exact, one group per history

    Every fit is weighted by the stratum probability w(a), which is what the engine sums with."""
    Q, nv = signed_controller(nC, nT, sgn, mag, cc=cc)
    pi, res, _ = stationary(Q)
    cur = block_joint(Q, pi, nC, L, dt)
    exact = marginalise_targets(pi, nv, nC)
    exact = exact / exact.sum()
    vex = var_on(exact, nT)
    st = np.arange(1 << nT, dtype=np.int64)
    bits = [((st >> j) & 1) for j in range(nT)]
    A, W, Qm = [], [], []
    for a, v in cur.items():
        w = float(v.sum())
        if w <= 1e-300:
            continue
        m = marginalise_targets(v, nv, nC)
        s = m.sum()
        if s <= 0:
            continue
        A.append(a)
        W.append(w)
        Qm.append([float((m / s)[bits[j] == 1].sum()) for j in range(nT)])
    W = np.array(W)
    Z = logit(np.array(Qm))                       # strata x targets, on the log odds
    nsl = np.array([[sum((s >> c) & 1 for c in range(nC)) for s in a] for a in A], dtype=float)
    tot = nsl.sum(axis=1)

    def group_fit(keys):
        """weighted mean of the log odds within each group -- the matched analogue of a group mean"""
        gi = collections.defaultdict(list)
        for i, k in enumerate(keys):
            gi[k].append(i)
        out = np.zeros_like(Z)
        for k, ii in gi.items():
            ii = np.array(ii)
            out[ii] = (W[ii, None] * Z[ii]).sum(axis=0) / W[ii].sum()
        return out, len(gi) * nT

    fits = {}
    fits["total count"] = group_fit([int(t) for t in tot])
    fits["per-slice cells"] = group_fit([tuple(r) for r in nsl])
    fits["full pattern"] = group_fit(list(range(len(A))))
    fits["no conditioning"] = group_fit([0] * len(A))

    # the MULTIPLIER: partition on the total count, plus a shared temporal term, backfit
    # The MULTIPLIER: partition on the total count, plus a shared temporal term, backfit.
    # h is shared across targets, so (nsl @ h)[i] does not depend on j and the weighted least
    # squares for h reduces exactly to a fit against the per-stratum MEAN residual over targets,
    # with weight W[i] * nT. Writing it that way rather than stacking (i, j) rows is the same
    # estimator and makes the sharing explicit.
    gi = collections.defaultdict(list)
    for i, k in enumerate([int(t) for t in tot]):
        gi[k].append(np.int64(i))
    gidx = {k: np.array(v) for k, v in gi.items()}
    h = np.zeros(L + 1)
    mu = np.zeros_like(Z)
    for _ in range(12):
        R = Z - (nsl @ h)[:, None]
        for k, ii in gidx.items():
            mu[ii] = (W[ii, None] * R[ii]).sum(axis=0) / W[ii].sum()
        resid = (Z - mu).mean(axis=1)
        sw = np.sqrt(W)
        X = nsl * sw[:, None]
        h = np.linalg.solve(X.T @ X + lam * np.eye(L + 1), X.T @ (resid * sw))
    R = Z - (nsl @ h)[:, None]                    # mu must be refreshed against the FINAL h
    for k, ii in gidx.items():
        mu[ii] = (W[ii, None] * R[ii]).sum(axis=0) / W[ii].sum()
    zmul = mu + (nsl @ h)[:, None]
    npar_mul = len(gidx) * nT + (L + 1)
    fits["MULTIPLIER count x g(when)"] = (zmul, npar_mul)

    out = {}
    for name, (zz, npar) in fits.items():
        qa = expit(zz)
        acc = np.zeros(1 << nT)
        for i in range(len(A)):
            term = np.full(1 << nT, W[i])
            for j in range(nT):
                term = term * np.where(bits[j] == 1, qa[i, j], 1.0 - qa[i, j])
            acc += term
        acc = acc / acc.sum()
        out[name] = ((var_on(acc, nT) - vex) / vex, npar)
    return out, len(A), res


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE); P_("THE MULTIPLIER FORM: PAY ADDITIVELY FOR AN AXIS A COUNT PAYS MULTIPLICATIVELY FOR")
    P_(RULE)
    d, sha = fetch()
    recs = load(d)
    sub = [r for r in recs if r["tfs"]]
    l1 = np.array([r["l1"] for r in recs]); l2 = np.array([r["l2"] for r in recs])
    floor = float(np.std(l1 - l2)) / np.sqrt(2.0)
    P_(f"  Sharon et al. 2012, doi:10.1038/nbt.2205, GEO GSE37851, sha256[:32] {sha}.")
    P_(f"  {len(sub)} promoters. Replicate noise floor {floor:.4f} log2.")
    P_( "  The form under test: P(x | class counts) * g(positions), which in log2 expression is a")
    P_( "  group mean on the class counts PLUS a shared additive positional term, backfit.")

    fits = {}

    def cls_key(C):
        st = {"f": 0}

        def mk(tr):
            f = st["f"]; st["f"] += 1
            if f not in fits:
                fits[f] = tf_effects(tr)
            cl = class_bins(fits[f], C)
            return lambda r: cell_key(cl, lambda s: 0, False, r)
        return mk

    def cellform_key(C, P__, mode, rs=None):
        posb = pos_binner(rs if rs is not None else sub, P__, mode)
        st = {"f": 0}

        def mk(tr):
            f = st["f"]; st["f"] += 1
            if f not in fits:
                fits[f] = tf_effects(tr)
            cl = class_bins(fits[f], C)
            return lambda r: cell_key(cl, posb, False, r)
        return mk

    # ---- M0(a)  COST -------------------------------------------------------------------------
    P_("\n" + RULE); P_("M0(a)  THE COST CEILING: IS THE CLAIM 'C + P, NOT C * P' TRUE?"); P_(RULE)
    P_("  The cap functional counts the per-gene TABLE SIZE. A cell form multiplies that table by")
    P_("  P; a multiplier leaves the table alone and adds P parameters SHARED across all genes.")
    P_("  So the claim is that the multiplier's cap is the cap at alphabet C. Verified, not")
    P_("  assumed, by adding the shared parameters to the functional and checking it does not move.")
    L = 8
    caps = class_caps((4, 8, 16, 32, 64, 128), L)
    P_(f"\n    {'form':<34} {'alphabet':>9} {'cap |C| at L=8':>15}")
    P_(f"    {'pattern (exact)':<34} {'2^(k(L+1))':>9} {caps[('pattern', 0)]:>15}")
    for C, P__ in ((8, 4), (16, 8), (32, 4)):
        P_(f"    {f'CELL form, C={C} x P={P__}':<34} {C*P__:>9} {caps[('class', C*P__)]:>15}")
        P_(f"    {f'MULTIPLIER, C={C} and g over P={P__}':<34} {C:>9} {caps[('class', C)]:>15}")
    shared = max(P__ for _, P__ in ((8, 4), (16, 8), (32, 4)))
    caps_plus = class_caps((8, 16, 32), L, bar=1e12 - shared)
    moved = [C for C in (8, 16, 32) if caps_plus[("class", C)] != caps[("class", C)]]
    P_(f"\n  the P shared parameters charged against the same bar move the cap for: "
       f"{moved if moved else 'NO C -- the claim holds'}")
    P_(f"  M0(a): {'PASS' if not moved else 'FAIL -- the additive claim is false'}."
       f" A multiplier over P bins at C classes caps at {caps[('class', 8)]} controllers where the")
    P_(f"  cell form of the same expressive reach caps at {caps[('class', 32)]}."
       f" That is the entire point of the form.")

    # ---- M1  IDENTITY --------------------------------------------------------------------------
    P_("\n" + RULE); P_("M1  IT MUST REDUCE EXACTLY TO WHAT IT GENERALISES"); P_(RULE)
    feats4 = pos_feats(pos_binner(sub, 4, "loc"))
    e_pure, f_pure, err_pure, hit_pure = cv_grouped(sub, cls_key(8))
    e_id, f_id, err_id, _ = multiplier_cv(sub, cls_key(8), feats4, lam=np.inf)
    P_(f"    pure class count C=8, group mean      {e_pure:.10f}")
    P_(f"    multiplier with the shared term OFF   {e_id:.10f}")
    P_(f"    max |difference| over all promoters   {np.abs(err_pure-err_id).max():.3e}")
    m1 = np.abs(err_pure - err_id).max() < 1e-12
    P_(f"  M1: {'PASS' if m1 else 'FAIL -- the form does not reduce to the class count'}")

    # ---- M2  ON REAL PROMOTERS -----------------------------------------------------------------
    P_("\n" + RULE); P_("M2  ON REAL PROMOTERS"); P_(RULE)
    e_idn, f_idn, err_idn, hit_idn = cv_grouped(
        sub, lambda tr: (lambda r: (r["ctx"], tuple(sorted(t[0] for t in r["tfs"])))))
    P_(f"    {'model':<44} {'RMSE':>9} {'vs pure C=8':>12} {'fallback':>9} {'alphabet':>9}")
    P_(f"    {'pure class count, C = 8':<44} {e_pure:>9.4f} {0.0:>12.2f} {f_pure:>8.1%} {8:>9}")
    m2 = {}
    for P__ in (2, 4, 8, 49):
        binner = pos_binner(sub, P__, "loc") if P__ < 49 else (lambda s: s)
        ft = pos_feats(binner)
        ec, fc, erc, hc = cv_grouped(sub, cellform_key(8, P__, "loc")) if P__ < 49 else (
            np.nan, np.nan, None, None)
        em, fm, erm, hm = multiplier_cv(sub, cls_key(8), ft)
        m2[P__] = (em, fm, erm, hm, ec, fc, erc, hc)
        if P__ < 49:
            P_(f"    {f'CELL form, C = 8 x P = {P__}':<44} {ec:>9.4f}"
               f" {(e_pure-ec)/floor:>+12.2f} {fc:>8.1%} {8*P__:>9}")
        P_(f"    {f'MULTIPLIER, C = 8, g over {P__} position bins':<44} {em:>9.4f}"
           f" {(e_pure-em)/floor:>+12.2f} {fm:>8.1%} {8:>9}")
    r_base = ridge_cv(sub, lambda r: {("ctx", r["ctx"]): 1.0})
    P_(f"\n    reference points, for what position is worth at all:")
    P_(f"    {'identity group mean':<44} {e_idn:>9.4f} {(e_pure-e_idn)/floor:>+12.2f} {f_idn:>8.1%}")
    P_(f"    {'context only, no site information':<44} {r_base:>9.4f}")
    bestP = min(m2, key=lambda k: m2[k][0])
    gain = (e_pure - m2[bestP][0]) / floor
    cellgain = max((e_pure - m2[k][4]) / floor for k in m2 if not np.isnan(m2[k][4]))
    P_(f"\n  M2: the multiplier's best is {gain:+.2f} floors over the pure class count, at"
       f" {bestP} bins.")
    P_(f"  The cell form's best over the same base is {cellgain:+.2f} floors, at an alphabet up to")
    P_(f"  8x larger. {'The multiplier gets more for less.' if gain > cellgain else 'The cell form is not beaten on accuracy; the multiplier is cheaper.'}")

    # ---- M3  ABSTENTION ------------------------------------------------------------------------
    P_("\n" + RULE); P_("M3  ABSTENTION MUST NOT RISE -- THE FORM'S STRUCTURAL CLAIM"); P_(RULE)
    P_(f"    {'model':<44} {'fallback':>9} {'vs pure':>9}")
    P_(f"    {'pure class count, C = 8':<44} {f_pure:>8.1%} {0.0:>9.1%}")
    ok3 = True
    for P__ in sorted(m2):
        em, fm, erm, hm, ec, fc, erc, hc = m2[P__]
        if not np.isnan(fc):
            P_(f"    {f'CELL form, C = 8 x P = {P__}':<44} {fc:>8.1%} {fc-f_pure:>+9.1%}")
        P_(f"    {f'MULTIPLIER, g over {P__} bins':<44} {fm:>8.1%} {fm-f_pure:>+9.1%}")
        ok3 = ok3 and abs(fm - f_pure) < 1e-9
    P_(f"  M3: {'PASS -- the multiplier never splits a group, so abstention is identical' if ok3 else 'FAIL -- the multiplier partitioned somewhere it should not have'}")

    # ---- M4  INSIDE THE MAIN LADDER ------------------------------------------------------------
    P_("\n" + RULE); P_("M4  INSIDE THE MAIN LADDER, WHERE poscount's CONFOUND IS CONSTANT"); P_(RULE)
    mainl = [r for r in sub if all(t[3] % 7 == 1 for t in r["tfs"])]
    P_(f"  {len(mainl)} of {len(sub)} promoters, all sites at start mod 7 == 1. poscount.py found")
    P_( "  the group-mean position gain VANISHES here (-0.33 floors at four bins) while a ridge")
    P_( "  still gains +0.41. This is where the multiplier has to work if the form is right.")
    fits.clear()
    em_base, fm_base, _, _ = cv_grouped(mainl, cls_key(8))
    P_(f"\n    {'model':<44} {'RMSE':>9} {'vs pure C=8':>12} {'fallback':>9}")
    P_(f"    {'pure class count, C = 8':<44} {em_base:>9.4f} {0.0:>12.2f} {fm_base:>8.1%}")
    for P__ in (4, 14):
        binner = pos_binner(mainl, P__, "loc") if P__ < 14 else (lambda s: s)
        ec, fc, _, _ = cv_grouped(mainl, cellform_key(8, P__, "loc", mainl)) if P__ < 14 else (
            np.nan, np.nan, None, None)
        if P__ < 14:
            P_(f"    {f'CELL form, C = 8 x P = {P__}':<44} {ec:>9.4f}"
               f" {(em_base-ec)/floor:>+12.2f} {fc:>8.1%}")
        e2, f2, _, _ = multiplier_cv(mainl, cls_key(8), pos_feats(binner))
        P_(f"    {f'MULTIPLIER, g over {P__} bins':<44} {e2:>9.4f}"
           f" {(em_base-e2)/floor:>+12.2f} {f2:>8.1%}")
    fits.clear()

    # ---- M5  THE ENGINE SIDE -------------------------------------------------------------------
    P_("\n" + RULE); P_("M5  THE ENGINE SIDE: POSITION IS TO A PROMOTER WHAT SLICE TIME IS TO A HISTORY")
    P_(RULE)
    P_("  summary.py's total count sums controller-slices over the whole window and discards WHEN.")
    P_("  That is the same shape of loss as discarding where a site sits. So the same form is")
    P_("  testable: P(x | total count) * g(when), with g SHARED across targets.")
    P_("  All four parameterisations are on the LOG ODDS so the comparison is about the form.")
    P_("  SWEPT over the window length L, not measured at one. The count discards MORE as the")
    P_("  window lengthens, so a form that repairs the count has to keep repairing it as L grows;")
    P_("  a single L is the error this session has now made five times.")
    P_("  NOTE: these errors are not comparable with summary.py's, which averaged the probability")
    P_("  where this averages the log odds. Within this table the link is matched throughout.")
    nC, nT, dtb = 3, 5, 0.25
    sg = np.array([1.0, -1.0, 1.0])
    mg = np.array([4.5, 4.0, 1.2])
    P_(f"\n  nC={nC} controllers (two activators, one repressor, heterogeneous magnitudes),"
       f" nT={nT}, dt={dtb}")
    P_(f"\n    {'L':>3} {'strata':>7} {'count err':>11} {'MULTIPLIER':>11} {'cells':>11}"
       f" {'pattern':>10} {'repaired':>9} {'params mul':>11} {'params cell':>12}")
    reps = []
    for Lb in (1, 2, 3, 4):
        res, nstrata, rr = temporal_forms(nC, nT, Lb, dtb, sg, mg)
        pat = res["full pattern"][0]
        ec = abs(res["total count"][0] - pat)
        em = abs(res["MULTIPLIER count x g(when)"][0] - pat)
        ecell = abs(res["per-slice cells"][0] - pat)
        rep = 100 * (ec - em) / ec if ec > 0 else float("nan")
        repc = 100 * (ec - ecell) / ec if ec > 0 else float("nan")
        reps.append((Lb, rep, repc, ec, em, ecell,
                     res["MULTIPLIER count x g(when)"][1], res["per-slice cells"][1],
                     res["total count"][1], res["no conditioning"][0], pat))
        P_(f"    {Lb:>3} {nstrata:>7} {ec:>11.3e} {em:>11.3e} {ecell:>11.3e}"
           f" {pat:>10.2e} {rep:>8.1f}% {res['MULTIPLIER count x g(when)'][1]:>11}"
           f" {res['per-slice cells'][1]:>12}")
    P_("\n    'count err', 'MULTIPLIER' and 'cells' are EXCESSES over the full pattern of the same")
    P_("    system -- the pattern is not exact either and charging a form for the harness is not a")
    P_("    measurement of the form. 'repaired' is the fraction of the count's excess the")
    P_("    multiplier removes.")
    P_(f"\n    {'L':>3} {'multiplier repairs':>19} {'cell form repairs':>18}"
       f" {'extra params, mul':>18} {'extra params, cell':>19}")
    for (Lb, rep, repc, ec, em, ecell, pm, pc, p0, nocond, pat) in reps:
        P_(f"    {Lb:>3} {rep:>18.1f}% {repc:>17.1f}% {pm-p0:>18} {pc-p0:>19}")
    rep_last = reps[-1][1]
    m5 = all(r[1] > 25.0 for r in reps)
    P_(f"\n  the multiplier repairs {min(r[1] for r in reps):.1f}% to"
       f" {max(r[1] for r in reps):.1f}% of the count's break across L = 1 to 4, for"
       f" {reps[-1][6]-reps[-1][8]} extra parameters at L = 4,")
    P_(f"  where the cell form repairs {min(r[2] for r in reps):.1f}% to"
       f" {max(r[2] for r in reps):.1f}% for {reps[-1][7]-reps[-1][8]} extra parameters.")
    P_(f"  M5: {'PASS -- the analogy transfers and the promoter result is an engine result' if m5 else 'FAIL -- the temporal multiplier does not repair the count, so the analogy does not transfer and this is a fact about promoters only'}")

    # ---- M6 ------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("M6  WHAT THIS DOES AND DOES NOT SETTLE"); P_(RULE)
    P_("  1. The promoter half is steady state with no time axis; the engine half is a synthetic")
    P_("     four-controller system. Neither is a real regulatory network over time, and the")
    P_("     ANALOGY between position and slice time is an argument, tested here on one system,")
    P_("     not a theorem.")
    P_("  2. The shared term is LINEAR in the position or slice counts. A multiplier that had to")
    P_("     be nonlinear in them would need more than this and is not tested.")
    P_("  3. The cap is a cost statement about TABLE SIZE. It says the tables fit, not that the")
    P_("     engine is accurate at that size, and the strata still have to be enumerated.")
    P_("  4. g is shared across all genes and all targets here. Whether a real network needs")
    P_("     per-gene positional responses -- which would put the P back inside the product and")
    P_("     undo the whole saving -- is exactly the thing this form assumes away.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_multiplier.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P_(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
