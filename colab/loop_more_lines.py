"""
LOOP 265 -- DOES CROSS-LINE TRANSFER IMPROVE WITH MORE CELL LINES?

Every attempt to predict a new cell line's operator from something OTHER than that line has
failed: fourteen annotation sources, three neural architectures, and now loop 264's pathway
constraint, which raised the operator's reliability from 0.448 to 0.611 and bought exactly
-0.0004 of transfer. That last one matters for this loop, because it ruled out the
comfortable explanation. A more reliable operator that transfers WORSE means the
line-specificity is real rather than measurement noise.

Which leaves one hypothesis standing, and it is about the DATA rather than the model: nine
cell lines may simply be too few to span the space an unseen line lives in. Loop 262's N8
said exactly that -- "a failure is about nine cell lines, not about the biology" -- and it
has been the open door ever since.

This loop opens it. The extracted shRNA block holds 20 cell lines and only 9 were ever used,
because the other 11 have fewer signatures. Three of them are substantial, and two are NOT
cancer lines: NPC (neural progenitor) and ASC (adipose stem) are far more distant from the
nine cancer lines than those nine are from each other, which is precisely what a spanning
test needs.

THE TRAP, NAMED BEFORE ANY NUMBER, AND IT ALMOST CAUGHT ME. The obvious design is loop 262's
span test with more basis operators: fit coefficients on the held-out line's own residual
and see whether the fit improves with n. That measurement is WORTHLESS. With cheating
coefficients the span is a projection onto an n-dimensional subspace, so it is monotonically
non-decreasing in n BY CONSTRUCTION. The curve would rise no matter what and mean nothing.

So the coefficients here are fitted honestly, on HALF the held-out line's genes, and scored
on the other half. That is not monotone in n -- more basis operators means more coefficients
estimated from the same fixed amount of data, so the curve can and should turn over if the
extra lines add nothing. It is also the practically relevant question: you have measured
part of a new cell line, and you want to know whether knowing other lines helps you predict
the rest of it.

GATES, ALL DECLARED BEFORE THE NUMBERS:

  R1 IS THE ORIGINAL HARNESS UNTOUCHED?
     The dense within-line operator on the ORIGINAL nine-line configuration, against loop
     262's +0.0670 and loop 264's +0.0670. Gate: PASS iff within 0.015.
     This loop adds a SEPARATE extended loader and must not perturb load().

  R2 HOW MUCH DOES THE EXTENSION ACTUALLY ADD?                       -- requires R1
     Lines and genes gained. Gate: PASS iff at least 2 new lines survive with enough genes
     to fit an operator at all.

  R3 DO THE NEW LINES SUPPORT AN OPERATOR?                           -- requires R2
     Each new line's own dense operator, fitted on half its genes, scored on the other half.
     Gate: PASS iff the new lines average at least +0.02.
     A FAIL means they are too thin to carry an operator and they cannot inform R4 either.

  R4 LOAD-BEARING -- DOES TRANSFER IMPROVE WITH THE NUMBER OF LINES?  -- requires R3
     Combination of n training operators, coefficients fitted on HALF the held-out line's
     genes, scored on the other half, for n from 2 upward.
     Gate: PASS iff the slope over n is positive by at least 2 standard errors.
     A FAIL says more cell lines of this kind will not close the gap, and the direction is
     closed for reasons that are no longer about sample size.

  R5 THE PRACTICAL QUESTION -- DOES IT ADD TO THE LINE'S OWN OPERATOR? -- requires R3
     Own operator alone, against own operator blended with the training combination, both
     with coefficients fitted on the same half.
     Gate: PASS iff the blend adds at least 0.005.

  R6 EXTRAPOLATION                                                    -- requires R4
     At the measured slope, how many cell lines would reach the own-operator's +0.0670.
     VOID if R4 found no positive slope, because extrapolating a flat line is meaningless.

  R7 WHAT THIS CANNOT SHOW
     Stated regardless of outcome.
"""
import json, time, collections, gzip
from pathlib import Path
import numpy as np

import lincs_harness as H
from gate_guard import Gates

SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
LX = SCR / "lincs"
OUT = "outputs/loop_more_lines.json"
SEED = 265265
MIN_GENES_LINE = 500
MIN_LINES_GENE = 6
NSUB = 6
LOOP262_DENSE = 0.0670
R1_TOL, R3_BAR, R5_BAR = 0.015, 0.02, 0.005
LAM = [1e2, 1e3, 1e4, 1e5]
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def load_extended(min_genes_line=MIN_GENES_LINE, min_lines_gene=MIN_LINES_GENE):
    """A SEPARATE loader. lincs_harness.load() is left exactly as it is, so every earlier
    loop's numbers remain reproducible."""
    X = np.load(LX / "shrna_landmark.npy", mmap_mode="r")
    S = np.load(LX / "select2.npz", allow_pickle=True)
    gene = np.array([str(x) for x in S["gene"]]); cell = np.array([str(x) for x in S["cell"]])
    ngene = {c: len(set(gene[cell == c])) for c in set(cell.tolist())}
    lines = sorted([c for c, n in ngene.items() if n >= min_genes_line],
                   key=lambda c: -ngene[c])
    keep = np.isin(cell, lines)
    Xk, gk, ck = np.asarray(X[keep]), gene[keep], cell[keep]
    idxs = collections.defaultdict(list)
    for i, (g, c) in enumerate(zip(gk, ck)): idxs[(g, c)].append(i)
    pairs = sorted(idxs)
    nlc = collections.Counter(g for g, c in pairs)
    pairs = [p for p in pairs if nlc[p[0]] >= min_lines_gene]
    Pm = np.stack([Xk[idxs[p]].mean(0) for p in pairs])
    pg = np.array([p[0] for p in pairs]); pc = np.array([p[1] for p in pairs])
    return dict(Pm=Pm, pg=pg, pc=pc, LINES=lines, NL=Pm.shape[1],
                genes=sorted(set(pg.tolist())), ngene=ngene)


def sc(P, Y): return float(np.nanmean([H.pear(P[i], Y[i]) for i in range(len(Y))]))


def fit_dense(X, R, lam, NL):
    return np.linalg.solve(X.T @ X + lam * np.eye(NL), X.T @ R)


def main():
    t0 = time.time()
    G_ = Gates(emit=say)
    res = {"test": "does cross-line transfer improve with more cell lines?"}
    say("=" * 104)
    say("LOOP 265 -- DOES CROSS-LINE TRANSFER IMPROVE WITH MORE CELL LINES?")
    say("=" * 104)
    say("     Loop 264 ruled out the comfortable explanation: constraining the operator raised")
    say("     its reliability 0.448 -> 0.611 and bought -0.0004 of transfer. A MORE reliable")
    say("     operator transferring WORSE means the line-specificity is real, not noise.")
    say("     One hypothesis is left and it is about DATA: nine lines may be too few to span")
    say("     the space an unseen line lives in. 20 lines were extracted; 9 were ever used.")
    say("     TRAP DECLARED FIRST: loop 262's span test fits coefficients on the held-out")
    say("     line's OWN residual, so it is monotone in n BY CONSTRUCTION and a rising curve")
    say("     would mean nothing. Coefficients here are fitted on HALF the held-out line's")
    say("     genes and scored on the other half, which can and should turn over.")

    say("R1 IS THE ORIGINAL HARNESS UNTOUCHED?")
    D9 = H.load()
    rng = np.random.default_rng(SEED)
    g9 = []
    for hold in D9["LINES"]:
        tr = D9["pc"] != hold
        gm = {}
        for g in D9["genes"]:
            m = tr & (D9["pg"] == g)
            if m.sum(): gm[g] = D9["Pm"][m].mean(0)
        grand = D9["Pm"][tr].mean(0); lmn = D9["Pm"][D9["pc"] == hold].mean(0)
        XG, Y, A = [], [], []
        for j in np.where(D9["pc"] == hold)[0]:
            g = D9["pg"][j]
            if g not in gm: continue
            dg = gm[g] - grand
            XG.append(dg); Y.append(D9["Pm"][j]); A.append(grand + dg + lmn - grand)
        XG, Y, A = (np.stack(v).astype(np.float64) for v in (XG, Y, A))
        R = Y - A
        p = rng.permutation(len(XG)); h1, h2 = p[:len(p)//2], p[len(p)//2:]
        W = fit_dense(XG[h1], R[h1], 1e3, D9["NL"])
        g9.append(sc(A[h2] + XG[h2] @ W, Y[h2]) - sc(A[h2], Y[h2]))
    d1 = float(np.mean(g9))
    say(f"     nine-line dense operator, half the genes: {d1:+.4f} against loop 262's "
        f"{LOOP262_DENSE:+.4f}")
    G_.add("R1", bool(abs(d1 - LOOP262_DENSE) <= R1_TOL), stat=d1,
           if_true=lambda: f"R1 PASS -- reproduces to {abs(d1-LOOP262_DENSE):.4f}; load() is "
                           f"untouched and every earlier loop's numbers still stand",
           if_false=lambda: f"R1 FAIL -- {d1:+.4f} against {LOOP262_DENSE:+.4f}")
    res["R1"] = {"nine_line_dense": d1}

    say("R2 HOW MUCH DOES THE EXTENSION ACTUALLY ADD?")
    E = load_extended()
    LINES, Pm, pg, pc, NL = E["LINES"], E["Pm"], E["pg"], E["pc"], E["NL"]
    new = [l for l in LINES if l not in D9["LINES"]]
    say(f"     lines with >= {MIN_GENES_LINE} distinct genes: {len(LINES)} "
        f"({len(D9['LINES'])} original + {len(new)} new)")
    for l in LINES:
        n = int((pc == l).sum())
        tag = "  <-- NEW" if l in new else ""
        say(f"       {l:9s} {n:5,d} (gene, line) pairs, {E['ngene'][l]:5,d} genes in the "
            f"raw block{tag}")
    say(f"     genes present in >= {MIN_LINES_GENE} of these lines: {len(E['genes']):,} "
        f"(nine-line set was {len(D9['genes']):,})")
    G_.add("R2", bool(len(new) >= 2), stat=float(len(new)), requires=("R1",),
           if_true=lambda: f"R2 PASS -- {len(new)} new lines survive: {', '.join(new)}",
           if_false=lambda: f"R2 FAIL -- only {len(new)} new lines clear "
                            f"{MIN_GENES_LINE} genes")
    res["R2"] = {"lines": LINES, "new": new, "n_genes": len(E["genes"]),
                 "ngene": {k: int(v) for k, v in E["ngene"].items()}}

    # ---------- per-line residual construction on the EXTENDED set ----------
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

    say("R3 DO THE NEW LINES SUPPORT AN OPERATOR?")
    own, curve, blend_gain, base_all = {}, collections.defaultdict(list), [], {}
    curve_null = collections.defaultdict(list)
    for hold in LINES:
        rows = fold(hold)
        XG, Y, A = rows(hold)
        R = Y - A
        p = rng.permutation(len(XG)); h1, h2 = p[:len(p)//2], p[len(p)//2:]
        Wown = fit_dense(XG[h1], R[h1], 1e3, NL)
        b = sc(A[h2], Y[h2])
        base_all[hold] = b
        own[hold] = sc(A[h2] + XG[h2] @ Wown, Y[h2]) - b

        # training operators, each fitted on its own line's full data
        Wtr, tn = [], []
        for c in LINES:
            if c == hold: continue
            Xc, Yc, Ac = rows(c)
            Wtr.append(fit_dense(Xc, Yc - Ac, 1e3, NL)); tn.append(c)
        # projections of the held-out line's inputs through each training operator
        P1 = np.stack([XG[h1] @ W for W in Wtr])
        P2 = np.stack([XG[h2] @ W for W in Wtr])
        # NULL: operators fitted on ROW-PERMUTED residuals of the same training lines.
        # Identical fitting procedure, identical scale, identical count -- the gene-to-
        # response pairing destroyed. Adding basis vectors to a least-squares fit helps
        # almost automatically, so the REAL curve must beat this one to mean anything.
        Wnull = []
        for c in LINES:
            if c == hold: continue
            Xc, Yc, Ac = rows(c)
            Rc = Yc - Ac
            Wnull.append(fit_dense(Xc, Rc[rng.permutation(len(Rc))], 1e3, NL))
        Q1_ = np.stack([XG[h1] @ W for W in Wnull])
        Q2_ = np.stack([XG[h2] @ W for W in Wnull])
        del Wnull
        own1, own2 = XG[h1] @ Wown, XG[h2] @ Wown

        for n in range(2, len(Wtr) + 1):
            for store, PA, PB in ((curve, P1, P2), (curve_null, Q1_, Q2_)):
                gains = []
                for _ in range(NSUB):
                    s = rng.choice(len(PA), size=n, replace=False)
                    F = PA[s].reshape(n, -1)
                    a = np.linalg.solve(F @ F.T + 1e-6 * np.eye(n), F @ R[h1].ravel())
                    gains.append(sc(A[h2] + np.tensordot(a, PB[s], axes=(0, 0)), Y[h2]) - b)
                store[n].append(float(np.mean(gains)))
        # R5: own operator blended with ALL training operators, weights on the same half
        F = np.concatenate([own1.ravel()[None, :], P1.reshape(len(Wtr), -1)], 0)
        a = np.linalg.solve(F @ F.T + 1e-6 * np.eye(len(F)), F @ R[h1].ravel())
        pred = a[0] * own2 + np.tensordot(a[1:], P2, axes=(0, 0))
        blend_gain.append(sc(A[h2] + pred, Y[h2]) - b - own[hold])
        del P1, P2, Q1_, Q2_, Wtr

    newg = [own[l] for l in new]; oldg = [own[l] for l in LINES if l not in new]
    say(f"     own dense operator, fit on half the line's genes, scored on the other half:")
    for l in LINES:
        say(f"       {l:9s} {own[l]:+.4f}{'   <-- NEW' if l in new else ''}")
    d3 = float(np.mean(newg))
    say(f"     new lines mean {d3:+.4f}   original nine mean {np.mean(oldg):+.4f}")
    G_.add("R3", bool(d3 >= R3_BAR), stat=d3, requires=("R2",),
           if_true=lambda: f"R3 PASS -- the new lines carry an operator at {d3:+.4f}",
           if_false=lambda: f"R3 FAIL -- the new lines reach only {d3:+.4f}; too thin to carry "
                            f"an operator, so they cannot inform R4 either")
    res["R3"] = {"own_gain": {k: float(v) for k, v in own.items()},
                 "new_mean": d3, "old_mean": float(np.mean(oldg))}

    say("R4 LOAD-BEARING -- DOES TRANSFER IMPROVE WITH THE NUMBER OF LINES?")
    ns = sorted(curve)
    means = np.array([np.mean(curve[n]) for n in ns])
    say(f"     combination of n training operators, weights fitted on half the held-out")
    say(f"     line's genes and scored on the other half:")
    mnull = np.array([np.mean(curve_null[n]) for n in ns])
    say(f"       {'n':>4s}  {'real':>9s}  {'PERMUTED null':>14s}  {'difference':>11s}")
    for n, mv, nv in zip(ns, means, mnull):
        say(f"       {n:4d}  {mv:+9.4f}  {nv:+14.4f}  {mv-nv:+11.4f}")
    xs = np.array(ns, float)
    per = np.array([[curve[n][i] - curve_null[n][i] for n in ns] for i in range(len(LINES))])
    slopes = np.array([np.polyfit(xs, per[i], 1)[0] for i in range(len(LINES))])
    sl, se = float(slopes.mean()), float(slopes.std(ddof=1) / np.sqrt(len(slopes)))
    raw = np.array([[curve[n][i] for n in ns] for i in range(len(LINES))])
    rs = float(np.mean([np.polyfit(xs, raw[i], 1)[0] for i in range(len(LINES))]))
    say(f"     RAW slope per added line {rs:+.5f} -- but adding basis vectors to a least")
    say(f"     squares fit helps almost automatically, which is why the null curve exists.")
    say(f"     EXCESS slope over the permuted null: {sl:+.5f} +/- {se:.5f} "
        f"({sl/max(se,1e-12):+.1f} se)")
    G_.add("R4", bool(sl > 2 * se), stat=sl, requires=("R3",),
           if_true=lambda: f"R4 PASS -- transfer rises {sl:+.5f} per added line IN EXCESS of "
                           f"the permuted null, {sl/max(se,1e-12):.1f} se from zero",
           if_false=lambda: f"R4 FAIL -- the excess slope over the permuted null is {sl:+.5f} "
                            f"+/- {se:.5f}; the raw slope of {rs:+.5f} per line is what "
                            f"adding basis vectors to a least-squares fit buys for free")
    res["R4"] = {"n": ns, "mean_gain": [float(x) for x in means],
                 "null_gain": [float(x) for x in mnull],
                 "excess_slope": sl, "slope_se": se, "raw_slope": rs}

    say("R5 THE PRACTICAL QUESTION -- DOES IT ADD TO THE LINE'S OWN OPERATOR?")
    d5 = float(np.mean(blend_gain))
    say(f"     own operator {np.mean(list(own.values())):+.4f}, "
        f"blended with all {len(LINES)-1} training operators: {d5:+.4f} on top")
    G_.add("R5", bool(d5 >= R5_BAR), stat=d5, requires=("R3",),
           if_true=lambda: f"R5 PASS -- other lines add {d5:+.4f} to a line's own operator",
           if_false=lambda: f"R5 FAIL -- other lines add {d5:+.4f} to a line's own operator; "
                            f"once you have measured half a line, the other lines are surplus")
    res["R5"] = {"blend_gain": d5, "own_mean": float(np.mean(list(own.values())))}

    say("R6 EXTRAPOLATION")
    if sl <= 2 * se:
        G_.add("R6", False, stat=sl, requires=("R4",), void_if=True,
               void_reason=f"R4's slope is {sl:+.5f} +/- {se:.5f}; extrapolating a flat line "
                           f"would invent a number")
    else:
        need = (np.mean(list(own.values())) - (means[-1] - mnull[-1])) / sl + ns[-1]
        say(f"     at {sl:+.5f} per line, reaching the own-operator's "
            f"{np.mean(list(own.values())):+.4f} needs about {need:.0f} cell lines")
        G_.add("R6", bool(need <= 200), stat=float(need), requires=("R4",),
               if_true=lambda: f"R6 PASS -- about {need:.0f} lines would suffice, which is a "
                               f"reachable experiment",
               if_false=lambda: f"R6 FAIL -- about {need:.0f} lines would be needed")
        res["R6"] = {"lines_needed": float(need)}

    say("R7 WHAT THIS CANNOT SHOW")
    say(f"     The new lines are thin. NPC and ASC carry about a quarter of the genes the")
    say(f"     original nine do, so their operators are estimated from fewer rows and R3 is")
    say(f"     the gate that says whether that mattered.")
    say("     R4's slope is measured over n up to the lines available here. A curve that is")
    say("     flat from 2 to 11 could still rise at 50; the honest claim is only that lines")
    say("     OF THIS KIND, added at this rate, do not help.")
    say("     NPC and ASC are not cancer lines, which is what makes them a real spanning test,")
    say("     but it also means a failure could be distance rather than count -- they may be")
    say("     too far from the nine to interpolate rather than too few.")
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
