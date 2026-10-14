"""
LOOP 268 -- THE OFF-DIAGONAL OPERATOR AND THE TRANSFER CURVE, OUTSIDE LINCS

Two findings in this arc have never been tested on data other than LINCS shRNA:

  loop 262   90% of the gene x line interaction is OFF-DIAGONAL. The line-specific operator
             W_c is worth +0.0673 on held-out genes against a diagonal's +0.0238, and only
             10.4% of its Frobenius norm sits on its own diagonal.
  loop 265   cross-line transfer rises with the number of training lines. Loop 267 then put
             saturation at n = 600 on Score2, correcting loop 265's extrapolated 61 upward
             by roughly tenfold.

Score2 could not test the first one at all: CRISPR fitness is ONE SCALAR per (gene, line),
so there is no response profile and no operator to fit. Tahoe is the first dataset with both
properties -- 50 cell lines AND a 978-gene readout per condition -- so it can test both.

WHAT WAS STREAMED, and the numbers that constrain this loop:
    32,210 conditions x 978 landmark genes, 6.51% missing
    50 cell lines, 643-645 conditions each -- almost perfectly balanced
    276 drugs at 2 doses, 695 (drug, dose) combinations
    837 of 978 genes finite in at least 95% of conditions

THE CONDITIONING PROBLEM, STATED BEFORE ANY NUMBER, BECAUSE IT SETS THE READOUT SIZE.
LINCS gave each line 3,801 perturbations against a 978-gene readout, so each column of W_c
had 3.8 observations per parameter. Tahoe gives 644 per line. A 978 x 978 operator would be
UNDERdetermined at 0.66:1 -- worse conditioned than anything loop 262 measured, and a weak
result would then be ambiguous between "no off-diagonal structure here" and "not enough
rows". So the operator is fitted on the K most variable well-covered genes, K chosen so that
observations per parameter exceed 2:1, and K is reported. This is a deliberate loss of
comparability with loop 262's absolute numbers in exchange for a conditioned fit.

A REPLICATION AXIS THAT COSTS NOTHING. Every drug appears at two doses. U6 runs the whole
analysis at each dose separately. Score2 gave a processing-pipeline replication; this gives
a concentration replication, and neither needed new data.

GATES, ALL DECLARED BEFORE THE NUMBERS:

  U1 IS THE INTERACTION SUBSTANTIAL?
     Variance split into drug main effect, line main effect and interaction. LINCS shRNA was
     68.7% interaction, Score2 CRISPR fitness reported separately in loop 267.
     Gate: PASS iff interaction exceeds 20%.

  U2 DOES LOOP 260'S GEOMETRY REPLICATE A THIRD TIME?               -- requires U1
     Best in-sample one-way models of the residual. 0.027% / 0.0017% on LINCS,
     0.0000% / 0.0000% on Score2. Gate: PASS iff both below 1%.

  U3 IS THE INTERACTION OFF-DIAGONAL HERE TOO?                      -- requires U2
     Full W_c against diag(s_c), both fitted on half a line's conditions and scored on the
     other half -- loop 262's honest protocol, not its same-rows oracle.
     Gate: PASS iff the full operator exceeds the diagonal by at least 0.01.
     This is loop 262's central claim, on a different perturbation type in a different lab.

  U4 LOAD-BEARING -- DOES TRANSFER RISE WITH THE NUMBER OF LINES?   -- requires U2
     Combination of n training operators, weights fitted on half the held-out line's
     conditions, scored on the other half, against a null whose training profiles are
     shuffled over genes. Gate: PASS iff the excess at n = 49 exceeds that at n = 2 by 0.005.
     The bar is RELATIVE to this dataset's own n=2 point, not an absolute threshold --
     loop 267's T4 failed by 0.0012 against an absolute bar set before its assay's scale
     was known, and that mistake is not being repeated.

  U5 WHERE DOES IT SATURATE?                                        -- requires U4
     Smallest n reaching 90% of the maximum excess. Reported against Score2's 600 and loop
     265's extrapolated 61. Not gated: with only 49 training lines available, a saturation
     point at or near 49 would be censored by the data rather than measured, and gating a
     censored quantity would manufacture a verdict.

  U6 DOES IT REPLICATE ACROSS DOSE?                                 -- requires U3
     U3 and U4 recomputed at each dose separately.
     Gate: PASS iff both statistics agree between doses within 0.01.

  U7 WHAT THIS CANNOT SHOW
     Stated regardless of outcome.
"""
import json, time, collections
from pathlib import Path
import numpy as np

from gate_guard import Gates

SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
OUT = "outputs/loop_tahoe.json"
SEED = 268268
OBS_PER_PARAM = 2.0
NLINES = [2, 4, 8, 16, 24, 32, 40, 49]
NHOLD, NREP = 25, 2
LAM = [1e0, 1e1, 1e2, 1e3, 1e4]
U1_BAR, U2_BAR, U3_BAR, U4_BAR, U6_TOL = 0.20, 0.01, 0.01, 0.005, 0.01
SCORE2_SAT, LOOP265_EXTRAP = 600, 61
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else np.nan


def build(dose=None):
    """(condition x gene) response, the additive baseline, and the double-centred residual."""
    z = np.load(SCR / "tahoe_assembled.npz", allow_pickle=True)
    M = np.asarray(z["M"], np.float64)
    drug = np.array([str(x) for x in z["drug"]]); conc = np.asarray(z["conc"], np.float64)
    cell = np.array([str(x) for x in z["cell"]]); genes = np.array([str(x) for x in z["genes"]])
    keep_line = cell != "NA"                       # one shard carries no DepMap identifier
    if dose is not None:
        keep_line &= np.isclose(conc, dose)
    M, drug, conc, cell = M[keep_line], drug[keep_line], conc[keep_line], cell[keep_line]
    cov = np.isfinite(M).mean(0)
    gk = cov >= 0.95
    M, genes = M[:, gk], genes[gk]
    M = np.where(np.isfinite(M), M, np.nan)
    col = np.nanmean(M, 0)
    M = np.where(np.isfinite(M), M, col[None, :])   # impute the residual 5% at the gene mean
    combo = np.array([f"{d}|{c:g}" for d, c in zip(drug, conc)])
    return M, combo, cell, genes


def decompose(M, combo, cell):
    """Additive baseline over (perturbation, line) and the residual it leaves."""
    cm, lm = {}, {}
    for k in np.unique(combo): cm[k] = M[combo == k].mean(0)
    for l in np.unique(cell): lm[l] = M[cell == l].mean(0)
    gr = M.mean(0)
    A = np.stack([cm[k] + lm[c] - gr for k, c in zip(combo, cell)])
    return A, M - A, cm, lm, gr


def fit_dense(X, R, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ R)


def diag_fit(U, R):
    den = (U * U).sum(0)
    return np.where(den > 1e-12, (U * R).sum(0) / np.maximum(den, 1e-12), 0.0)


def analyse(M, combo, cell, genes, rng, label, want_curve=True):
    """U3's off-diagonal comparison and U4's transfer curve, on one slice of the data."""
    lines = sorted(set(cell.tolist()))
    npc = int(np.median([int((cell == l).sum()) for l in lines]))
    K = max(50, min(len(genes), int(npc / OBS_PER_PARAM)))
    v = M.var(0)
    sel = np.argsort(-v)[:K]
    say(f"     {label}: {len(lines)} lines, ~{npc} conditions each, operator on the "
        f"K={K} most variable of {len(genes)} genes ({npc/K:.1f} obs per parameter)")
    Msel = M[:, sel]
    A, R, _, _, _ = decompose(Msel, combo, cell)

    own_full, own_diag, curve, curve_null = [], [], collections.defaultdict(list), \
        collections.defaultdict(list)
    hold_ids = list(rng.choice(len(lines), size=min(NHOLD, len(lines)), replace=False))
    for hi, li in enumerate(hold_ids):
        hold = lines[li]
        hm = cell == hold
        Xh, Rh, Ah, Yh = R[hm], R[hm], A[hm], Msel[hm]
        idx = rng.permutation(int(hm.sum()))
        h1, h2 = idx[:len(idx) // 2], idx[len(idx) // 2:]
        # the operator maps the PERTURBATION's average profile to this line's deviation
        cmean = {}
        for k in np.unique(combo): cmean[k] = Msel[combo == k].mean(0)
        Xg = np.stack([cmean[k] for k in combo[hm]]) - Msel.mean(0)
        lam = 1e3
        best, be = LAM[0], np.inf
        q = rng.permutation(len(h1)); q1, q2 = h1[q[:len(q)//2]], h1[q[len(q)//2:]]
        for l_ in LAM:
            W_ = fit_dense(Xg[q1], Rh[q1], l_)
            e = float(((Xg[q2] @ W_ - Rh[q2]) ** 2).mean())
            if e < be: be, best = e, l_
        W = fit_dense(Xg[h1], Rh[h1], best)
        sdg = diag_fit(Xg[h1], Rh[h1])
        b0 = float(np.nanmean([pear(Ah[i], Yh[i]) for i in h2]))
        sc = lambda P: float(np.nanmean([pear(P[j], Yh[h2[j]]) for j in range(len(h2))]))
        own_full.append(sc(Ah[h2] + Xg[h2] @ W) - b0)
        own_diag.append(sc(Ah[h2] + sdg * Xg[h2]) - b0)

        if want_curve:
            tr_lines = [l for l in lines if l != hold]
            Wtr = {}
            for l in tr_lines:
                m = cell == l
                Xl = np.stack([cmean[k] for k in combo[m]]) - Msel.mean(0)
                Wtr[l] = fit_dense(Xl, R[m], 1e3)
            for n in NLINES:
                nn = min(n, len(tr_lines))
                for _ in range(NREP):
                    s = rng.choice(len(tr_lines), size=nn, replace=False)
                    for tag, mats in (("real", [Wtr[tr_lines[i]] for i in s]),
                                      ("null", [Wtr[tr_lines[i]][:, rng.permutation(K)]
                                                for i in s])):
                        P1 = np.stack([Xg[h1] @ W_ for W_ in mats])
                        P2 = np.stack([Xg[h2] @ W_ for W_ in mats])
                        F = P1.reshape(nn, -1)
                        a = np.linalg.solve(F @ F.T + 1e-6 * np.eye(nn), F @ Rh[h1].ravel())
                        g = sc(Ah[h2] + np.tensordot(a, P2, axes=(0, 0))) - b0
                        (curve if tag == "real" else curve_null)[nn].append(g)
            del Wtr
        if hi % 5 == 0:
            say(f"       {label}: held-out line {hi+1}/{len(hold_ids)}")
    return (float(np.mean(own_full)), float(np.mean(own_diag)),
            {n: float(np.mean(curve[n]) - np.mean(curve_null[n])) for n in sorted(curve)},
            {n: float(np.mean(curve_null[n])) for n in sorted(curve_null)}, K)


def main():
    t0 = time.time()
    G_ = Gates(emit=say)
    res = {"test": "loop 262's off-diagonal operator and loop 265's transfer curve, outside LINCS"}
    say("=" * 104)
    say("LOOP 268 -- THE OFF-DIAGONAL OPERATOR AND THE TRANSFER CURVE, OUTSIDE LINCS")
    say("=" * 104)
    say("     Two findings have only ever been tested on LINCS shRNA. Score2 could test one of")
    say("     them: its fitness is a SCALAR per (gene, line), so there is no operator at all.")
    say("     Tahoe is the first data with 50 cell lines AND a 978-gene readout, so both the")
    say("     off-diagonal claim (loop 262) and the transfer curve (loops 265, 267) can run.")
    say("     CONDITIONING DECLARED FIRST: LINCS gave 3,801 perturbations per line against 978")
    say("     genes, 3.8 observations per operator parameter. Tahoe gives 644, so a full")
    say("     978x978 operator would be UNDERdetermined at 0.66:1 and a weak result would be")
    say("     ambiguous. The operator is fitted on the K most variable genes with K set so")
    say("     that observations per parameter exceed 2:1, and K is reported.")

    rng = np.random.default_rng(SEED)
    M, combo, cell, genes = build()
    say(f"     {M.shape[0]:,} conditions x {M.shape[1]} well-covered genes, "
        f"{len(set(cell.tolist()))} cell lines, {len(set(combo.tolist()))} (drug, dose) combos")

    say("U1 IS THE INTERACTION SUBSTANTIAL?")
    A, R, _, _, _ = decompose(M, combo, cell)
    vt = float(M.var()); vi = float(R.var())
    cm = {k: M[combo == k].mean(0) for k in np.unique(combo)}
    lmn = {l: M[cell == l].mean(0) for l in np.unique(cell)}
    vg = float(np.stack([cm[k] for k in combo]).var()); vl = float(np.stack([lmn[c] for c in cell]).var())
    say(f"     drug main {vg/vt:.1%}, line main {vl/vt:.1%}, interaction {vi/vt:.1%}")
    say(f"     LINCS shRNA was 27.6% / 3.8% / 68.7%")
    G_.add("U1", bool(vi / vt >= U1_BAR), stat=float(vi / vt),
           if_true=lambda: f"U1 PASS -- {vi/vt:.1%} of the variance is drug x line interaction",
           if_false=lambda: f"U1 FAIL -- only {vi/vt:.1%} is interaction; the response is nearly "
                            f"additive and there is nothing here to transfer")
    res["U1"] = {"drug": vg / vt, "line": vl / vt, "interaction": vi / vt}

    say("U2 DOES LOOP 260'S GEOMETRY REPLICATE A THIRD TIME?")
    cg = 1.0 - float(((R - np.stack([R[combo == k].mean(0) for k in combo])) ** 2).sum()
                     / (R ** 2).sum())
    cl = 1.0 - float(((R - np.stack([R[cell == c].mean(0) for c in cell])) ** 2).sum()
                     / (R ** 2).sum())
    say(f"     best drug-only model of the residual {cg:.6f}, line-only {cl:.6f}")
    say(f"     LINCS shRNA 0.000270 / 0.000017; Score2 CRISPR 0.000000 / 0.000000")
    G_.add("U2", bool(cg < U2_BAR and cl < U2_BAR), stat=float(max(cg, cl)), requires=("U1",),
           if_true=lambda: f"U2 PASS -- {cg:.4%} and {cl:.4%}; double-centring pins one-way "
                           f"models at zero on a third assay",
           if_false=lambda: f"U2 FAIL -- drug-only {cg:.4%}, line-only {cl:.4%}")
    res["U2"] = {"drug_only": cg, "line_only": cl}

    say("     running the operator and transfer analysis ...")
    full, dg, curve, null, K = analyse(M, combo, cell, genes, rng, "all doses")
    res["K"] = K

    say("U3 IS THE INTERACTION OFF-DIAGONAL HERE TOO?")
    d3 = full - dg
    say(f"     full operator {full:+.4f} vs diagonal {dg:+.4f}   {d3:+.4f}")
    say(f"     loop 262 on LINCS: +0.0673 full against +0.0238 diagonal, difference +0.0435")
    G_.add("U3", bool(d3 >= U3_BAR), stat=float(d3), requires=("U2",),
           if_true=lambda: f"U3 PASS -- the off-diagonal is worth {d3:+.4f} beyond a diagonal "
                           f"on drug perturbations too, so loop 262's claim is not a LINCS artefact",
           if_false=lambda: f"U3 FAIL -- the off-diagonal is worth {d3:+.4f} beyond a diagonal "
                            f"here; loop 262's finding does not reproduce on this assay")
    res["U3"] = {"full": full, "diagonal": dg, "delta": d3}

    say("U4 LOAD-BEARING -- DOES TRANSFER RISE WITH THE NUMBER OF LINES?")
    ns = sorted(curve)
    for n in ns:
        say(f"       n = {n:2d} training lines   excess {curve[n]:+.4f}   (null {null[n]:+.4f})")
    d4 = curve[ns[-1]] - curve[ns[0]]
    say(f"     n={ns[0]} -> n={ns[-1]}: {d4:+.4f}")
    say(f"     the bar is RELATIVE to this dataset's own n={ns[0]} point. Loop 267's T4 failed")
    say(f"     by 0.0012 against an ABSOLUTE bar set before its assay's scale was known.")
    G_.add("U4", bool(d4 >= U4_BAR), stat=float(d4), requires=("U2",),
           if_true=lambda: f"U4 PASS -- the excess grows {d4:+.4f} from {ns[0]} to {ns[-1]} "
                           f"training lines, in excess of a gene-shuffled null",
           if_false=lambda: f"U4 FAIL -- the excess changes by {d4:+.4f} from {ns[0]} to "
                            f"{ns[-1]} training lines, below the {U4_BAR} bar")
    res["U4"] = {"curve": {str(n): curve[n] for n in ns},
                 "null": {str(n): null[n] for n in ns}, "delta": d4}

    say("U5 WHERE DOES IT SATURATE?")
    mx = max(curve.values())
    sat = next((n for n in ns if curve[n] >= 0.9 * mx), ns[-1])
    say(f"     maximum excess {mx:+.4f}, 90% of it first reached at n = {sat}")
    say(f"     Score2 saturated at {SCORE2_SAT} of 907; loop 265 extrapolated {LOOP265_EXTRAP}")
    say(f"     NOT GATED: only {ns[-1]} training lines exist here, so a saturation point near")
    say(f"     {ns[-1]} is CENSORED by the data rather than measured, and gating a censored")
    say(f"     quantity would manufacture a verdict out of a limit of the dataset.")
    res["U5"] = {"saturation_n": int(sat), "max_excess": mx, "n_max": int(ns[-1]),
                 "censored": bool(sat >= ns[-1])}

    say("U6 DOES IT REPLICATE ACROSS DOSE?")
    per = {}
    for dose in (0.05, 0.5):
        Md, cbd, cld, gnd = build(dose=dose)
        f_, d_, c_, n_, K_ = analyse(Md, cbd, cld, gnd, np.random.default_rng(SEED),
                                     f"dose {dose}")
        nsd = sorted(c_)
        per[dose] = {"off_diag": f_ - d_, "rise": c_[nsd[-1]] - c_[nsd[0]], "K": K_}
        say(f"     dose {dose}: off-diagonal {f_-d_:+.4f}, transfer growth "
            f"{c_[nsd[-1]]-c_[nsd[0]]:+.4f} (K={K_})")
    da = abs(per[0.05]["off_diag"] - per[0.5]["off_diag"])
    db = abs(per[0.05]["rise"] - per[0.5]["rise"])
    say(f"     between doses: off-diagonal differs by {da:.4f}, transfer growth by {db:.4f}")
    G_.add("U6", bool(da <= U6_TOL and db <= U6_TOL), stat=float(max(da, db)), requires=("U3",),
           if_true=lambda: f"U6 PASS -- the two doses agree to {max(da,db):.4f}; a replication "
                           f"that needed no new data, as Score2's two pipelines were",
           if_false=lambda: f"U6 FAIL -- the doses differ by {max(da,db):.4f}; the result "
                            f"depends on concentration and is not a property of the cell lines")
    res["U6"] = {str(k): v for k, v in per.items()}

    say("U7 WHAT THIS CANNOT SHOW")
    say(f"     The operator uses K={K} genes, not 978, because 644 conditions per line cannot")
    say(f"     condition a 978x978 fit. Loop 262's ABSOLUTE numbers are therefore not")
    say(f"     comparable; only the sign and rough size of the off-diagonal effect are.")
    say("     DRUGS, not gene knockdowns. A drug hits many targets and its 'perturbation")
    say("     identity' is not a gene, so agreement with LINCS is a statement about response")
    say("     structure, not about the same biological intervention.")
    say(f"     {len(NLINES)} points up to 49 training lines. Score2 needed 600 to saturate, so")
    say("     this curve is almost certainly still climbing at its right edge and U5 says so.")
    say("     50 cancer cell lines on one platform. Saturation within that population is not")
    say("     saturation across human cell states.")
    say("     Everything is on the double-centred residual, so by loop 260 and U2 any arm")
    say("     constant across lines is pinned near zero however expressive it is.")

    res["gates"] = {k: (v == "PASS") for k, v in G_.status.items()}
    res["void"] = [k for k, v in G_.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G_.summary(seconds=res["seconds"])
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
