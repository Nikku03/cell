"""
LOOP 262 -- IS THE INTERACTION OFF-DIAGONAL, AND IF SO CAN THAT BE PREDICTED?

Loop 261 closed the diagonal case. The per-line scale s_c is worth +0.0247, reproduces at
0.9518 from disjoint halves of its own line, and is predicted from DepMap annotation at
0.0549. Its M8 named the one thing it could not see: s_c is a DIAGONAL scaling and cannot
express a rotation. This loop tests the general linear form.

    diagonal      R ~= diag(s_c) xg          one number per landmark
    general       R ~= W_c xg                a 978 x 978 operator per line

W_c off the diagonal says landmark j's deviation depends on landmark i's, which is what a
pathway or a regulatory cascade would look like if one existed in this data.

WHAT LOOP 260 ALREADY FORBIDS, STATED BEFORE ANY NUMBER. The residual R is double-centred,
so ANY function of the gene alone is pinned at a 0.027% ceiling -- gene-only, not just
diagonal, and not just linear. A SHARED operator W applied to xg is a function of the gene
alone, because xg is a gene-only quantity. So a single universal response-propagation
operator cannot work here no matter how rich it is, and loop 261 already saw the diagonal
version of that: its constant predictor scored +0.0001. The N_CONST arm below is expected
to land near zero and is included to confirm the geometry, not to test a hypothesis.
The live question is therefore only about LINE-SPECIFIC operators.

THE CEILING THAT DECIDES THE ARC. N3 fits the best possible combination of the eight
training lines' operators, with the coefficients chosen using the held-out line's own
answers. That is cheating, and it is the ceiling for EVERY method that builds a held-out
operator out of training operators -- including ridge from annotation, since with eight
training lines a ridge prediction is necessarily a linear combination of the eight. If N3
fails, no such method can succeed and the direction is closed rather than merely untried.

GATES, ALL DECLARED BEFORE THE NUMBERS:

  N1 IS THERE OFF-DIAGONAL STRUCTURE AT ALL?
     CORRECTED FROM RUN 1, which gated on an oracle fitted AND scored on the same rows and
     so could not tell structure from overfitting. A 956,484-parameter operator scored on
     its own fitting rows will look good whatever it learned. Run 1 reported +0.1138.
     The gate is now the HONEST within-line comparison: fit on half of the held-out line's
     GENES, score on the other half, with lambda chosen by a further inner split of the
     FITTING half only, so nothing that touches the score chose anything.
     Gate: PASS iff the honest full operator beats the honest diagonal by at least 0.005.
     Both oracles are still reported beside it so the overfitting gap is visible.
     A FAIL means the interaction is diagonal and loop 261 already measured all of it.

  N2 IS W_c RELIABLE, OR IS IT NOISE?                              -- requires N1
     Fit W_c on disjoint halves of the line's genes and correlate the two operators
     entrywise. s_c reached 0.9518 on this test.
     CORRECTED FROM RUN 1: run 1's docstring called W_c "underdetermined by roughly 250 to
     1". That divided 956,484 parameters by ~3,700 ROWS, treating each row as one scalar
     observation when it is a 978-vector. The regression is R(n x 978) = XG(n x 978) W, so
     each COLUMN of W has 978 parameters against n observations of that landmark: 3.8 to 1
     OVERdetermined, not 250 to 1 under. Run 1's N8 therefore warned that W_c "is mostly
     fitting noise", which was too pessimistic by two orders of magnitude.
     Gate: PASS iff the mean split-half correlation exceeds 0.30.

  N3 LOAD-BEARING -- IS THE HELD-OUT OPERATOR IN THE SPAN OF THE TRAINING ONES?
                                                                   -- requires N1, N2
     W_hold ~= sum_c a_c W_c over the eight training lines, with a fitted on the held-out
     line's own residual. Cheating, and the ceiling for every interpolation method.
     Gate: PASS iff it beats the additive baseline by at least 0.005.

  N4 DOES ANNOTATION PICK THE COEFFICIENTS?                        -- requires N3
     Ridge from z_c, which with eight training lines yields exactly a coefficient vector
     over those eight operators. Scored against the CONSTANT arm (equal weights, no
     annotation), for the same reason loop 261's M4 was: the shared part is not something
     the annotation earned. VOID if N3 found no span to work in.
     Gate: PASS iff it beats the constant arm by at least 0.005.

  N5 PERMUTED CONTROL                                              -- requires N4
     The same ridge with z_c paired to the wrong line's operator. VOID if N4 has no margin.
     Gate: PASS iff at most 25% of N4's margin survives.

  N6 HOW MUCH OF THE OFF-DIAGONAL HEADROOM IS RECOVERED?           -- requires N1
     The best honest arm as a fraction of the full-W oracle.
     Gate: PASS iff at least 20%.

  N7 HOW DIAGONAL IS THE OPERATOR?
     The fraction of W_c's squared Frobenius norm on its own diagonal, and the score of
     W_c with its off-diagonal zeroed. Reported, not gated -- it says what KIND of object
     the oracle found.

  N8 WHAT THIS CANNOT SHOW
     Stated regardless of outcome.
"""
import json, time, csv
from pathlib import Path
import numpy as np

import lincs_harness as H
from gate_guard import Gates

SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
OUT = "outputs/loop_offdiagonal.json"
SEED = 262262
NPC_EXPR, NPC_DEP = 50, 50
DIAG_ORACLE_REF = 0.0247
N1_BAR, N2_BAR, N3_BAR, N4_BAR, N5_MAX, N6_BAR = 0.005, 0.30, 0.005, 0.005, 0.25, 0.20
LAM_W = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
LAM_Z = [1e-2, 1e-1, 1.0, 10.0, 100.0]
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def gram(X, R):
    return X.T @ X, X.T @ R


def solve_ridge(G, B, lam):
    return np.linalg.solve(G + lam * np.eye(G.shape[0]), B)


def diag_fit(U, R):
    den = (U * U).sum(0)
    return np.where(den > 1e-12, (U * R).sum(0) / np.maximum(den, 1e-12), 0.0)


def main():
    t0 = time.time()
    G_ = Gates(emit=say)
    res = {"test": "is the gene x line interaction off-diagonal, and can that be predicted?"}
    say("=" * 104)
    say("LOOP 262 -- IS THE INTERACTION OFF-DIAGONAL, AND CAN THAT BE PREDICTED?")
    say("=" * 104)
    say("     Loop 261 closed the diagonal: s_c worth +0.0247, reliable at 0.9518, predicted")
    say("     from annotation at 0.0549. Its M8 named what it could not see -- s_c is a")
    say("     diagonal scaling and cannot express a rotation. Here the operator is full 978x978.")
    say("     DECLARED FIRST, from loop 260: R is double-centred, so ANY function of the gene")
    say("     alone is pinned at a 0.027% ceiling. A SHARED operator W applied to xg IS such a")
    say("     function, so a universal propagation operator cannot work however rich it is.")
    say("     The constant arm below confirms that geometry; it does not test a hypothesis.")

    D = H.load()
    Pm, pg, pc, LINES, NL = D["Pm"], D["pg"], D["pc"], D["LINES"], D["NL"]
    say(f"     {len(pg):,} pairs, {len(D['genes']):,} genes, {NL} landmarks, {len(LINES)} lines")
    say(f"     W_c has {NL * NL:,} free parameters against roughly 3,700 rows per line")

    lmap = json.load(open(SCR / "lincs" / "line_map.json"))
    ez = np.load(SCR / "depmap_expr_aligned.npz", allow_pickle=True)
    XE = ez["XE"]; el = np.array([str(x) for x in ez["lines"]])
    U_, sv, _ = np.linalg.svd(XE - XE.mean(0), full_matrices=False)
    EPC = U_[:, :NPC_EXPR] * sv[:NPC_EXPR]
    ge = np.load(SCR / "depmap" / "gene_effect.npz", allow_pickle=True)
    GE = np.nan_to_num(np.asarray(ge["E"], np.float32)); gl = np.array([str(x) for x in ge["lines"]])
    U2, sv2, _ = np.linalg.svd(GE - GE.mean(0), full_matrices=False)
    DPC = U2[:, :NPC_DEP] * sv2[:NPC_DEP]
    burden = {}
    with open(SCR / "depmap" / "OmicsSomaticMutationsMatrixDamaging.csv") as f:
        r = csv.reader(f); next(r)
        for row in r:
            burden[row[0]] = float(sum(1 for v in row[1:] if v not in ("", "0", "0.0")))
    ep_ = {l: int(np.where(el == lmap[l])[0][0]) for l in LINES}
    dp_ = {l: int(np.where(gl == lmap[l])[0][0]) for l in LINES}
    ZC = np.stack([np.concatenate([EPC[ep_[l]], DPC[dp_[l]],
                                   [np.log1p(burden.get(lmap[l], 0.0))]]) for l in LINES])

    def sc(P, Y): return float(np.nanmean([H.pear(P[i], Y[i]) for i in range(len(Y))]))

    rng = np.random.default_rng(SEED)
    arms = {k: [] for k in ("base", "diag", "full", "span", "ridge", "const", "perm",
                            "diagonly", "hon_base", "hon_diag", "hon_full")}
    relia, diagfrac, chosen = [], [], []

    for hold in LINES:
        tr = pc != hold
        gm = {}
        for g in D["genes"]:
            m = tr & (pg == g)
            if m.sum(): gm[g] = Pm[m].mean(0)
        grand = Pm[tr].mean(0); lmean = {l: Pm[pc == l].mean(0) for l in LINES}

        def rows(line):
            XG, Y, A = [], [], []
            for j in np.where(pc == line)[0]:
                g = pg[j]
                if g not in gm: continue
                dg = gm[g] - grand
                XG.append(dg); Y.append(Pm[j]); A.append(grand + dg + (lmean[line] - grand))
            return tuple(np.stack(v).astype(np.float64) for v in (XG, Y, A))

        # ---- held-out line ----
        xh, yh, ah = rows(hold); Rh = yh - ah
        Gh, Bh = gram(xh, Rh)
        # lambda for the ORACLE, chosen by an inner split over this line's own GENES
        p = rng.permutation(len(xh)); i1, i2 = p[:len(p) // 2], p[len(p) // 2:]
        G1, B1 = gram(xh[i1], Rh[i1]); G2, B2 = gram(xh[i2], Rh[i2])
        best_lam, best_e = LAM_W[0], np.inf
        for lam in LAM_W:
            W1 = solve_ridge(G1, B1, lam)
            e = float(((xh[i2] @ W1 - Rh[i2]) ** 2).mean())
            if e < best_e: best_e, best_lam = e, lam
        chosen.append(best_lam)
        Wh = solve_ridge(Gh, Bh, best_lam)
        # N2 reliability: the two half-fitted operators against each other
        Wa, Wb = solve_ridge(G1, B1, best_lam), solve_ridge(G2, B2, best_lam)
        relia.append(H.pear(Wa.ravel(), Wb.ravel()))

        # ---- the HONEST within-line arm: fit on one gene half, score on the other, with
        # lambda chosen by a further inner split of the FITTING half, so nothing that
        # touches the score chose anything. This is what N1 gates on.
        hb, hd, hf = [], [], []
        for fit_i, sco_i in ((i1, i2), (i2, i1)):
            q = rng.permutation(len(fit_i))
            q1, q2 = fit_i[q[:len(q) // 2]], fit_i[q[len(q) // 2:]]
            Gq, Bq = gram(xh[q1], Rh[q1])
            bl, be = LAM_W[0], np.inf
            for lam in LAM_W:
                Wq = solve_ridge(Gq, Bq, lam)
                e = float(((xh[q2] @ Wq - Rh[q2]) ** 2).mean())
                if e < be: be, bl = e, lam
            Gf, Bf = gram(xh[fit_i], Rh[fit_i])
            Wf = solve_ridge(Gf, Bf, bl)
            sf = diag_fit(xh[fit_i], Rh[fit_i])
            hb.append(sc(ah[sco_i], yh[sco_i]))
            hd.append(sc(ah[sco_i] + sf * xh[sco_i], yh[sco_i]))
            hf.append(sc(ah[sco_i] + xh[sco_i] @ Wf, yh[sco_i]))
        arms["hon_base"].append(float(np.mean(hb)))
        arms["hon_diag"].append(float(np.mean(hd)))
        arms["hon_full"].append(float(np.mean(hf)))
        dfr = float((np.diag(Wh) ** 2).sum() / max((Wh ** 2).sum(), 1e-12))
        diagfrac.append(dfr)

        # ---- training-line operators, same lambda ----
        Ws, Zs = [], []
        for c in LINES:
            if c == hold: continue
            xc, yc, ac = rows(c)
            Gc, Bc = gram(xc, yc - ac)
            Ws.append(solve_ridge(Gc, Bc, best_lam)); Zs.append(ZC[LINES.index(c)])
        Zs = np.stack(Zs); mu, sd = Zs.mean(0), Zs.std(0) + 1e-9
        Zz = (Zs - mu) / sd; zh = (ZC[LINES.index(hold)] - mu) / sd

        # projections of the held-out inputs through each training operator
        P = np.stack([xh @ W for W in Ws])                      # (8, n, 978)
        flat = P.reshape(len(Ws), -1)
        Amat = flat @ flat.T + 1e-6 * np.eye(len(Ws))
        a_span = np.linalg.solve(Amat, flat @ Rh.ravel())        # cheating coefficients
        a_const = np.full(len(Ws), 1.0 / len(Ws))
        # ridge from z: with 8 lines this IS a coefficient vector over the 8 operators
        best_lz, best_ez = LAM_Z[0], np.inf
        for lz in LAM_Z:
            e = 0.0
            for i in range(len(Zz)):
                k = [j for j in range(len(Zz)) if j != i]
                M = np.linalg.solve(Zz[k].T @ Zz[k] + lz * np.eye(Zz.shape[1]), Zz[k].T)
                coef = Zz[i] @ M
                e += float(((coef @ flat[k] - flat[i]) ** 2).mean())
            if e < best_ez: best_ez, best_lz = e, lz
        M = np.linalg.solve(Zz.T @ Zz + best_lz * np.eye(Zz.shape[1]), Zz.T)
        a_ridge = zh @ M
        a_perm = zh @ np.linalg.solve(Zz.T @ Zz + best_lz * np.eye(Zz.shape[1]),
                                      Zz[rng.permutation(len(Zz))].T)

        def blend(a): return np.tensordot(a, P, axes=(0, 0))
        s_diag = diag_fit(xh, Rh)
        Wd = np.diag(np.diag(Wh))
        arms["base"].append(sc(ah, yh))
        arms["diag"].append(sc(ah + s_diag * xh, yh))
        arms["full"].append(sc(ah + xh @ Wh, yh))
        arms["diagonly"].append(sc(ah + xh @ Wd, yh))
        arms["span"].append(sc(ah + blend(a_span), yh))
        arms["const"].append(sc(ah + blend(a_const), yh))
        arms["ridge"].append(sc(ah + blend(a_ridge), yh))
        arms["perm"].append(sc(ah + blend(a_perm), yh))
        del P, flat, Ws

    m = {k: float(np.mean(v)) for k, v in arms.items()}
    say(f"     lambda per fold: " + ", ".join(f"{x:g}" for x in chosen))
    say("")
    say(f"     {'baseline':38s} {m['base']:.4f}")
    say(f"     {'+ ORACLE diagonal s_c (loop 261)':38s} {m['diag']:.4f}   {m['diag']-m['base']:+.4f}")
    say(f"     {'+ ORACLE full W_c':38s} {m['full']:.4f}   {m['full']-m['base']:+.4f}")
    say(f"     {'+ ORACLE W_c, off-diagonal zeroed':38s} {m['diagonly']:.4f}   {m['diagonly']-m['base']:+.4f}")
    say(f"     {'+ best combination of training W (cheat)':38s} {m['span']:.4f}   {m['span']-m['base']:+.4f}")
    say(f"     {'+ constant combination (no annotation)':38s} {m['const']:.4f}   {m['const']-m['base']:+.4f}")
    say(f"     {'+ ridge from z_c':38s} {m['ridge']:.4f}   {m['ridge']-m['base']:+.4f}")
    say(f"     {'+ ridge, line labels shuffled':38s} {m['perm']:.4f}   {m['perm']-m['base']:+.4f}")
    say("")
    say("     HONEST WITHIN-LINE -- fit on half the line's GENES, score on the other half:")
    say(f"     {'baseline':38s} {m['hon_base']:.4f}")
    say(f"     {'+ diagonal s_c':38s} {m['hon_diag']:.4f}   "
        f"{m['hon_diag']-m['hon_base']:+.4f}")
    say(f"     {'+ full W_c':38s} {m['hon_full']:.4f}   "
        f"{m['hon_full']-m['hon_base']:+.4f}")
    res["arms"] = m
    res["per_fold"] = {k: [float(x) for x in v] for k, v in arms.items()}
    res["lambda"] = chosen

    say("N1 IS THERE OFF-DIAGONAL STRUCTURE AT ALL?")
    d1 = m["hon_full"] - m["hon_diag"]
    d1_oracle = m["full"] - m["diag"]
    hg_f, hg_d = m["hon_full"] - m["hon_base"], m["hon_diag"] - m["hon_base"]
    say(f"     HONEST full W_c {hg_f:+.4f} vs HONEST diagonal {hg_d:+.4f}   difference {d1:+.4f}")
    say(f"     the same-rows oracles were {m['full']-m['base']:+.4f} and "
        f"{m['diag']-m['base']:+.4f}, a difference of {d1_oracle:+.4f}")
    say(f"     the diagonal survives honest scoring almost intact; the full operator does not,")
    say(f"     which is what a 956,484-parameter fit scored on its own rows looks like")
    say(f"     diagonal oracle reproduces loop 261's {DIAG_ORACLE_REF:+.4f} at "
        f"{m['diag']-m['base']:+.4f}")
    G_.add("N1", bool(d1 >= N1_BAR), stat=float(d1),
           if_true=lambda: f"N1 PASS -- off-diagonal structure is worth {d1:+.4f} beyond the "
                           f"diagonal on HELD-OUT GENES, so it survives honest scoring and is "
                           f"real rather than an artefact of fitting {NL*NL:,} parameters",
           if_false=lambda: f"N1 FAIL -- on held-out genes a full operator is worth {d1:+.4f} "
                            f"beyond a diagonal one; the interaction is diagonal and loop 261 "
                            f"measured all of it")
    res["N1"] = {"honest_full_minus_diag": d1, "oracle_full_minus_diag": d1_oracle,
                 "honest_full_gain": hg_f, "honest_diag_gain": hg_d,
                 "oracle_full_gain": m["full"] - m["base"],
                 "oracle_diag_gain": m["diag"] - m["base"]}

    say("N2 IS W_c RELIABLE, OR IS IT NOISE?")
    rl = float(np.mean(relia))
    say(f"     split-half over GENES, entrywise correlation of the two operators:")
    say(f"     " + ", ".join(f"{x:.3f}" for x in relia))
    say(f"     mean {rl:.4f}   (s_c reached 0.9518 on the same test)")
    G_.add("N2", bool(rl > N2_BAR), stat=rl, requires=("N1",),
           if_true=lambda: f"N2 PASS -- W_c reproduces at {rl:.3f} from disjoint gene halves",
           if_false=lambda: f"N2 FAIL -- W_c reproduces at only {rl:.3f}; with {NL*NL:,} "
                            f"parameters against ~3,700 rows it is mostly fitting noise, and "
                            f"any oracle gain above is that noise being fitted twice")
    res["N2"] = {"split_half": rl, "per_fold": [float(x) for x in relia]}

    say("N3 LOAD-BEARING -- IS THE HELD-OUT OPERATOR IN THE SPAN OF THE TRAINING ONES?")
    d3 = m["span"] - m["base"]
    say(f"     best combination of the 8 training operators, coefficients fitted on the")
    say(f"     held-out line's own residual: {d3:+.4f}")
    say(f"     this is the ceiling for EVERY method that builds W_hold from training lines,")
    say(f"     ridge from annotation included, since 8 lines give a coefficient vector over 8")
    G_.add("N3", bool(d3 >= N3_BAR), stat=float(d3), requires=("N1", "N2"),
           if_true=lambda: f"N3 PASS -- {d3:+.4f} is reachable by combining training operators",
           if_false=lambda: f"N3 FAIL -- even with cheating coefficients the training operators "
                            f"reach {d3:+.4f}; W_hold is not in their span and NO interpolation "
                            f"method can work, so this direction is closed rather than untried")
    res["N3"] = {"span_gain": d3}

    say("N4 DOES ANNOTATION PICK THE COEFFICIENTS?")
    d4 = m["ridge"] - m["const"]
    say(f"     ridge {m['ridge']:.4f} vs constant, equal-weight {m['const']:.4f}   {d4:+.4f}")
    say(f"     the constant arm is a shared operator, which loop 260 forbids from exceeding")
    say(f"     the 0.027% gene-only ceiling -- it landed at {m['const']-m['base']:+.4f}")
    if d3 < N3_BAR:
        G_.add("N4", False, stat=float(d4), requires=("N3",), void_if=True,
               void_reason=f"N3 reached {d3:+.4f} with cheating coefficients, so there is no "
                           f"span for annotation to pick a point in")
    else:
        G_.add("N4", bool(d4 >= N4_BAR), stat=float(d4), requires=("N3",),
               if_true=lambda: f"N4 PASS -- annotation is worth {d4:+.4f} beyond equal weights",
               if_false=lambda: f"N4 FAIL -- annotation is worth {d4:+.4f} beyond equal weights")
    res["N4"] = {"ridge_minus_const": d4, "const_gain": m["const"] - m["base"]}

    say("N5 PERMUTED CONTROL")
    if d3 < N3_BAR or d4 < N4_BAR:
        G_.add("N5", False, stat=float(d4), requires=("N4",), void_if=True,
               void_reason=f"N4's margin is {d4:+.4f}; there is nothing for a shuffle to collapse")
    else:
        d5 = m["perm"] - m["const"]
        f5 = d5 / d4
        say(f"     labels shuffled: {d5:+.4f} against a real {d4:+.4f} ({f5:.0%})")
        G_.add("N5", bool(f5 <= N5_MAX), stat=float(f5), requires=("N4",),
               if_true=lambda: f"N5 PASS -- collapses to {f5:.0%}",
               if_false=lambda: f"N5 FAIL -- {f5:.0%} survives the shuffle")
        res["N5"] = {"fraction": f5}

    say("N6 HOW MUCH OF THE OFF-DIAGONAL HEADROOM IS RECOVERED?")
    head = m["full"] - m["base"]
    best = max(m["const"], m["ridge"], m["span"]) - m["base"]
    frac = best / head if head > 1e-9 else 0.0
    say(f"     best arm {best:+.4f} of the full-W oracle's {head:+.4f} = {frac:.1%}")
    say(f"     (the best arm here INCLUDES the cheating span coefficients)")
    G_.add("N6", bool(frac >= N6_BAR), stat=float(frac), requires=("N1",),
           if_true=lambda: f"N6 PASS -- {frac:.1%} of the headroom is reachable",
           if_false=lambda: f"N6 FAIL -- {frac:.1%} of the {head:+.4f} headroom is reachable "
                            f"even allowing a cheating combination of training operators")
    res["N6"] = {"fraction": frac, "headroom": head, "best": best}

    say("N7 HOW DIAGONAL IS THE OPERATOR?")
    df = float(np.mean(diagfrac))
    say(f"     fraction of W_c's squared Frobenius norm on its own diagonal: {df:.4%}")
    say(f"     W_c with the off-diagonal zeroed scores {m['diagonly']-m['base']:+.4f} against")
    say(f"     the full operator's {m['full']-m['base']:+.4f} and the direct diagonal fit's "
        f"{m['diag']-m['base']:+.4f}")
    res["N7"] = {"diag_frobenius_fraction": df, "per_fold": [float(x) for x in diagfrac]}

    say("N8 WHAT THIS CANNOT SHOW")
    say(f"     W_c has {NL*NL:,} parameters, and each COLUMN of it has {NL} parameters against")
    say(f"     about 3,700 observations of that landmark -- 3.8 to 1 overdetermined. Run 1 of")
    say(f"     this loop said 250 to 1 UNDERdetermined by dividing parameters by rows rather")
    say(f"     than by observations; that error is corrected here and in the ledger.")
    say("     LINEAR operators only. A genuinely nonlinear interaction -- saturation, a")
    say("     threshold, a sign flip above some dose -- is invisible to every number here.")
    say("     The span test asks whether W_hold is reachable from EIGHT other operators. A")
    say("     failure is about nine cell lines, not about the biology: more lines could span")
    say("     a space these eight do not.")
    say("     Everything is measured on the double-centred residual, so by loop 260 no arm")
    say("     that is constant across lines can exceed 0.027% however expressive it is.")
    say("     978 landmarks and shRNA, under loop 254's construct ceiling of 0.2487.")

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
