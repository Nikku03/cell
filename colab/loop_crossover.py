"""
LOOP 266 -- THE CROSSOVER: HOW MANY GENES MUST YOU MEASURE IN A NEW CELL LINE?

Loop 265 found the first transferable signal in this arc. Other cell lines' OPERATORS carry
information about an unseen line -- +0.0115 above a permuted null at eleven training lines,
rising +0.00094 per line added, 9.0 se. Fourteen annotation sources had failed to carry
anything; measured behaviour in other lines does.

But loop 265's R5 also failed: given HALF a line's genes measured, that line's own operator
scores +0.0587 and blending in all eleven others is worth -0.0076. Read alone that says
borrowing is useless. Read carefully it says something narrower, because half a line's genes
is the budget most favourable to the own-operator arm:

    a line's OWN operator      978 x 978 = 956,484 parameters to estimate
    BORROWING from 11 lines            11 coefficients to estimate

At half the genes there is ample data for both, so the richer model wins. At 25 genes the
own operator is estimated from 25 rows and cannot possibly be worth anything, while eleven
coefficients from 25 x 978 = 24,450 observations are perfectly well determined. The two arms
must therefore cross somewhere, and WHERE they cross is the number that turns loop 265's
finding into a protocol: measure this many genes in your new cell line, then borrow the rest.

DESIGN. The evaluation set is FIXED at half of each held-out line's genes and never varies,
so every budget is scored on identical genes and the curves are directly comparable. The
measurement budget m is subsampled from the OTHER half, from 25 genes up to the whole of it.
Lambda is chosen by an inner split of the measured genes only, separately for each arm and
each budget, so no arm is handed a regularisation strength the others were denied.

THE CONTROL IS CARRIED AT EVERY BUDGET, NOT JUST AT THE END. Loop 265's first run was
withheld precisely because adding basis vectors to a least-squares fit improves it almost for
free, and the permuted-operator null quantified that at +0.0007 against a real +0.0122. The
same null runs at every budget here, because a crossover that appears only where the null
also rises would be an artefact of the fitting, not a finding about cell lines.

GATES, ALL DECLARED BEFORE THE NUMBERS:

  S1 DOES THE FULL BUDGET REPRODUCE LOOP 265?
     At the largest budget, the own operator against loop 265's +0.0587 and the borrowed
     combination against its +0.0122. Gate: PASS iff both within 0.015.

  S2 IS THERE A CROSSOVER AT ALL?                                    -- requires S1
     Gate: PASS iff at the smallest budget the borrowed arm exceeds the own arm by at
     least 0.005. A FAIL means the own operator wins even at 25 genes and there is no
     regime where borrowing is the right choice.

  S3 LOAD-BEARING -- IS THE BORROWED ARM REAL AT THE CROSSOVER?      -- requires S2
     The borrowed arm against the permuted-operator null at the same budget.
     Gate: PASS iff it exceeds the null by at least 0.005.
     A FAIL means the crossover is the fitting procedure, not the cell lines.

  S4 WHERE IS THE CROSSOVER, AND IS IT PRACTICAL?                    -- requires S2, S3
     The budget at which the own operator overtakes the borrowed one.
     Gate: PASS iff it is at least 100 genes, which is the point at which the answer is
     useful rather than trivial -- a crossover below that would mean borrowing helps only
     in a regime nobody operates in.

  S5 DOES BLENDING BEAT BOTH?                                        -- requires S2
     Own and borrowed together, coefficients fitted on the same measured genes.
     Gate: PASS iff the blend is at least as good as the better of the two at every
     budget, within 0.002. This is the gate that decides whether the protocol needs a
     switch at the crossover or can simply always blend.

  S6 WHAT THIS CANNOT SHOW
     Stated regardless of outcome.
"""
import json, time, collections
from pathlib import Path
import numpy as np

import lincs_harness as H
from gate_guard import Gates
from loop_more_lines import load_extended, fit_dense, sc

OUT = "outputs/loop_crossover.json"
SEED = 266266
BUDGETS = [25, 50, 100, 200, 400, 800, 1600, None]
NREP = 4
LAM = [1e2, 1e3, 1e4, 1e5]
LOOP265_OWN, LOOP265_BORROW = 0.0587, 0.0122
S1_TOL, S2_BAR, S3_BAR, S4_MIN, S5_TOL = 0.015, 0.005, 0.005, 100, 0.002
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def pick_lam_dense(X, R, NL, rng):
    """Lambda by an inner split of the MEASURED genes only."""
    if len(X) < 8: return LAM[-1]
    p = rng.permutation(len(X)); a, b = p[:len(p) // 2], p[len(p) // 2:]
    G = X[a].T @ X[a]; B = X[a].T @ R[a]
    best, be = LAM[-1], np.inf
    for lam in LAM:
        W = np.linalg.solve(G + lam * np.eye(NL), B)
        e = float(((X[b] @ W - R[b]) ** 2).mean())
        if e < be: be, best = e, lam
    return best


def main():
    t0 = time.time()
    G_ = Gates(emit=say)
    res = {"test": "how many genes must be measured in a new line before its own operator wins"}
    say("=" * 104)
    say("LOOP 266 -- THE CROSSOVER: HOW MANY GENES MUST YOU MEASURE IN A NEW CELL LINE?")
    say("=" * 104)
    say("     Loop 265: other lines' operators carry +0.0115 above a permuted null at 11 lines.")
    say("     Loop 265's R5: given HALF a line's genes, its own operator (+0.0587) beats them.")
    say("     Those are consistent, because half a line's genes is the budget most favourable")
    say("     to the own arm: 956,484 parameters to estimate against 11 for borrowing.")
    say("     At 25 genes the own operator cannot be worth anything and 11 coefficients from")
    say("     24,450 observations are well determined. WHERE they cross is the protocol.")
    say("     The permuted-operator null runs at EVERY budget, not just at the end.")

    E = load_extended()
    LINES, Pm, pg, pc, NL = E["LINES"], E["Pm"], E["pg"], E["pc"], E["NL"]
    say(f"     {len(LINES)} cell lines, {len(E['genes']):,} genes, {NL} landmarks")
    rng = np.random.default_rng(SEED)

    def fold(hold):
        tr = pc != hold
        gm = {}
        for g in E["genes"]:
            m = tr & (pg == g)
            if m.sum(): gm[g] = Pm[m].mean(0)
        grand = Pm[tr].mean(0); lmean = {l: Pm[pc == l].mean(0) for l in LINES}
        def rows(line):
            XG, Y, A = [], [], []
            for j in np.where(pc == line)[0]:
                g = pg[j]
                if g not in gm: continue
                dg = gm[g] - grand
                XG.append(dg); Y.append(Pm[j]); A.append(grand + dg + lmean[line] - grand)
            return tuple(np.stack(v).astype(np.float64) for v in (XG, Y, A))
        return rows

    curves = {k: collections.defaultdict(list) for k in ("own", "borrow", "blend", "null")}
    for hold in LINES:
        rows = fold(hold)
        XG, Y, A = rows(hold)
        R = Y - A
        p = rng.permutation(len(XG))
        ev = p[:len(p) // 2]                      # FIXED evaluation half, never varies
        pool = p[len(p) // 2:]
        b0 = sc(A[ev], Y[ev])

        Wtr, Wnull = [], []
        for c in LINES:
            if c == hold: continue
            Xc, Yc, Ac = rows(c)
            Rc = Yc - Ac
            Wtr.append(fit_dense(Xc, Rc, 1e3, NL))
            Wnull.append(fit_dense(Xc, Rc[rng.permutation(len(Rc))], 1e3, NL))
        Pev = np.stack([XG[ev] @ W for W in Wtr])
        Qev = np.stack([XG[ev] @ W for W in Wnull])

        for bud in BUDGETS:
            m = len(pool) if bud is None else min(bud, len(pool))
            reps = {k: [] for k in curves}
            for _ in range(NREP if bud is not None else 1):
                s1 = rng.choice(pool, size=m, replace=False)
                Xm, Rm = XG[s1], R[s1]
                lam = pick_lam_dense(Xm, Rm, NL, rng)
                Wown = fit_dense(Xm, Rm, lam, NL)
                own_ev = XG[ev] @ Wown
                Pm1 = np.stack([Xm @ W for W in Wtr])
                Qm1 = np.stack([Xm @ W for W in Wnull])

                def combo(F1, F2):
                    Fl = F1.reshape(len(F1), -1)
                    a = np.linalg.solve(Fl @ Fl.T + 1e-6 * np.eye(len(Fl)), Fl @ Rm.ravel())
                    return np.tensordot(a, F2, axes=(0, 0))
                bor_ev = combo(Pm1, Pev)
                nul_ev = combo(Qm1, Qev)
                F1 = np.concatenate([(Xm @ Wown).ravel()[None, :], Pm1.reshape(len(Pm1), -1)], 0)
                a = np.linalg.solve(F1 @ F1.T + 1e-6 * np.eye(len(F1)), F1 @ Rm.ravel())
                bl_ev = a[0] * own_ev + np.tensordot(a[1:], Pev, axes=(0, 0))

                reps["own"].append(sc(A[ev] + own_ev, Y[ev]) - b0)
                reps["borrow"].append(sc(A[ev] + bor_ev, Y[ev]) - b0)
                reps["null"].append(sc(A[ev] + nul_ev, Y[ev]) - b0)
                reps["blend"].append(sc(A[ev] + bl_ev, Y[ev]) - b0)
            for k in curves: curves[k][m if bud is not None else len(pool)].append(
                float(np.mean(reps[k])))
        del Wtr, Wnull, Pev, Qev
        say(f"     {hold:9s} done   [{time.time()-t0:.0f}s]")

    ms = sorted(curves["own"])
    M = {k: np.array([np.mean(curves[k][m]) for m in ms]) for k in curves}
    say("")
    say(f"     {'genes measured':>15s} {'OWN operator':>13s} {'BORROW 11':>11s} "
        f"{'BLEND':>9s} {'perm null':>10s}")
    for i, m in enumerate(ms):
        say(f"     {m:15d} {M['own'][i]:+13.4f} {M['borrow'][i]:+11.4f} "
            f"{M['blend'][i]:+9.4f} {M['null'][i]:+10.4f}")
    res["budgets"] = [int(x) for x in ms]
    res["curves"] = {k: [float(x) for x in v] for k, v in M.items()}

    say("S1 DOES THE FULL BUDGET REPRODUCE LOOP 265?")
    o_full, b_full = float(M["own"][-1]), float(M["borrow"][-1])
    say(f"     at {ms[-1]} genes: own {o_full:+.4f} against loop 265's {LOOP265_OWN:+.4f}, "
        f"borrow {b_full:+.4f} against {LOOP265_BORROW:+.4f}")
    ok1 = abs(o_full - LOOP265_OWN) <= S1_TOL and abs(b_full - LOOP265_BORROW) <= S1_TOL
    G_.add("S1", bool(ok1), stat=float(o_full),
           if_true=lambda: f"S1 PASS -- both arms reproduce loop 265 at the full budget",
           if_false=lambda: f"S1 FAIL -- own {o_full:+.4f} vs {LOOP265_OWN:+.4f}, borrow "
                            f"{b_full:+.4f} vs {LOOP265_BORROW:+.4f}")
    res["S1"] = {"own_full": o_full, "borrow_full": b_full}

    say("S2 IS THERE A CROSSOVER AT ALL?")
    d2 = float(M["borrow"][0] - M["own"][0])
    say(f"     at the smallest budget ({ms[0]} genes): borrow {M['borrow'][0]:+.4f} vs own "
        f"{M['own'][0]:+.4f}   {d2:+.4f}")
    G_.add("S2", bool(d2 >= S2_BAR), stat=d2, requires=("S1",),
           if_true=lambda: f"S2 PASS -- borrowing is worth {d2:+.4f} over the line's own "
                           f"operator at {ms[0]} genes, so a crossover regime exists",
           if_false=lambda: f"S2 FAIL -- borrowing is worth {d2:+.4f} over the own operator "
                            f"even at {ms[0]} genes; there is no regime where borrowing wins")
    res["S2"] = {"delta_at_min": d2, "min_budget": int(ms[0])}

    say("S3 LOAD-BEARING -- IS THE BORROWED ARM REAL AT THE CROSSOVER?")
    d3 = float(M["borrow"][0] - M["null"][0])
    say(f"     at {ms[0]} genes: borrow {M['borrow'][0]:+.4f} vs permuted null "
        f"{M['null'][0]:+.4f}   {d3:+.4f}")
    say(f"     the null uses operators fitted on ROW-PERMUTED residuals of the same lines --")
    say(f"     identical procedure, identical count, gene-to-response pairing destroyed")
    G_.add("S3", bool(d3 >= S3_BAR), stat=d3, requires=("S2",),
           if_true=lambda: f"S3 PASS -- the borrowed arm is worth {d3:+.4f} over its own "
                           f"permutation, so the crossover is about cell lines",
           if_false=lambda: f"S3 FAIL -- the borrowed arm is worth {d3:+.4f} over its own "
                            f"permutation; the crossover is the fitting procedure, not the "
                            f"cell lines")
    res["S3"] = {"borrow_minus_null_at_min": d3}

    say("S4 WHERE IS THE CROSSOVER, AND IS IT PRACTICAL?")
    cross = None
    for i, m in enumerate(ms):
        if M["own"][i] >= M["borrow"][i]:
            cross = m
            break
    say(f"     own overtakes borrow at {cross if cross else 'no budget tested'} genes")
    if cross is None:
        G_.add("S4", False, stat=float(ms[-1]), requires=("S2", "S3"), void_if=True,
               void_reason=f"the own operator never overtakes borrowing up to {ms[-1]} genes, "
                           f"so no crossover exists in the tested range to locate")
    else:
        G_.add("S4", bool(cross >= S4_MIN), stat=float(cross), requires=("S2", "S3"),
               if_true=lambda: f"S4 PASS -- the crossover is at {cross} genes, so borrowing is "
                               f"the right choice below that and measuring above it",
               if_false=lambda: f"S4 FAIL -- the crossover is at {cross} genes, below the "
                                f"{S4_MIN} bar; borrowing wins only in a regime too small to "
                                f"be useful")
        res["S4"] = {"crossover_genes": int(cross)}

    say("S5 DOES BLENDING BEAT BOTH?")
    best = np.maximum(M["own"], M["borrow"])
    worst_gap = float((M["blend"] - best).min())
    say(f"     blend against the better of the two arms, worst budget: {worst_gap:+.4f}")
    for i, m in enumerate(ms):
        say(f"       {m:5d} genes: blend {M['blend'][i]:+.4f} vs best-of-two "
            f"{best[i]:+.4f}   {M['blend'][i]-best[i]:+.4f}")
    G_.add("S5", bool(worst_gap >= -S5_TOL), stat=worst_gap, requires=("S2",),
           if_true=lambda: f"S5 PASS -- blending is within {S5_TOL} of the better arm at every "
                           f"budget, so the protocol can always blend and needs no switch",
           if_false=lambda: f"S5 FAIL -- blending is {worst_gap:+.4f} behind the better arm at "
                            f"its worst budget, so the protocol needs an explicit switch at "
                            f"the crossover rather than a single blended model")
    res["S5"] = {"worst_gap": worst_gap}

    say("S6 WHAT THIS CANNOT SHOW")
    say("     The measured genes are a RANDOM subset. A real experiment would choose which")
    say("     genes to perturb, and a chosen subset could cross far earlier than a random one.")
    say("     Nothing here says which genes to pick, and that is the obvious next question.")
    say("     Eleven training lines. Loop 265 measured the borrowed arm rising +0.00094 per")
    say("     added line, so every number in the borrow column is a floor that more lines")
    say("     would raise, and the crossover would move RIGHT with more lines.")
    say("     The evaluation half is fixed per line but the measured subset is drawn from the")
    say("     other half only, so the largest budget is half a line's genes and no budget")
    say("     above that is tested.")
    say("     Everything is on the double-centred residual, so by loop 260 any arm constant")
    say("     across lines is capped at 0.027%.")
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
