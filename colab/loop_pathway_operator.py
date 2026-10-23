"""
LOOP 264 -- PATHWAYS AS A CONSTRAINT ON THE OPERATOR, NOT AS A FEATURE

Thirteen annotation sources have now returned nothing: OmniPath arrows, BioGRID, paralogy,
co-dependency, and loop 255's nine. Loop 260 explains why in one line of algebra rather
than thirteen empirical failures. The target R = Y - (gene_mean + line_mean - grand) is
double-centred, so ANY function of the gene alone is capped at 0.027% of it. A pathway
membership, a GO term, a curated function -- all are properties of the GENE, identical in
every cell line, hence gene-only, hence capped. Every paper ever written about what genes
do enters as a feature and hits that ceiling by arithmetic.

So literature enters here in a different place. Not as an input to the model, but as a
RESTRICTION ON THE PARAMETER SPACE of the one object carrying real signal.

Loop 262's line-specific coupling operator W_c is 978 x 978 = 956,484 free parameters and
reproduces at only 0.448 from disjoint halves of its own line. That noise is a candidate
explanation for why it does not transfer: the span of eight training operators reached only
+0.0097 of the held-out line. Group the landmarks by pathway and the operator becomes

    W_c = M Theta_c M^T      M: 978 x K membership,  Theta_c: K x K

At K = 143 (Reactome pathways with at least 20 landmark members, covering 76% of the 978)
that is 20,449 parameters instead of 956,484 -- 47x fewer. This is NOT a gene-only function,
so loop 260's ceiling does not apply to it. The mechanism under test is explicit: fewer
parameters -> higher reliability -> a real chance of transferring across cell lines.

THE CONTROL THAT DECIDES IT, DECLARED FIRST. Loop 263 taught this the hard way. Its
neighbour predictor gained +0.0055, and a RANDOM graph of the same size gained +0.0054 --
99% of it. The gain was pooling, not the biology. The identical trap is here: projecting
978 landmarks onto 143 groups is a low-rank smoother, and low-rank smoothing helps whether
or not the groups mean anything. So Q3 compares the real pathway grouping against a RANDOM
grouping with the SAME SIZE DISTRIBUTION over the SAME landmarks. If Reactome does not beat
shuffled blocks, the answer is that the rank helped and the pathways did not, and that is
what gets recorded.

Theta_c is fitted in closed form. With U = X M,
    Theta = (U^T U + lam I)^-1 U^T R M (M^T M + mu I)^-1
so there is no gradient descent anywhere in this loop.

GATES, ALL DECLARED BEFORE THE NUMBERS:

  Q1 DOES THE HARNESS REPRODUCE THE DENSE OPERATOR?
     Dense W_c fitted on half a line's genes and scored on the other half, against loop
     262's honest +0.0673. Gate: PASS iff within 0.015.

  Q2 DOES THE PATHWAY-BLOCKED OPERATOR WORK AT ALL?                  -- requires Q1
     Blocked plus a diagonal term, since the blocks cover only 76% of landmarks and the
     diagonal is 978 cheap parameters that cover the rest.
     Gate: PASS iff it beats the additive baseline by at least 0.02 on held-out genes.

  Q3 LOAD-BEARING -- DO REAL PATHWAYS BEAT RANDOM BLOCKS?            -- requires Q2
     The same construction with membership shuffled: identical K, identical size
     distribution, identical landmark coverage, meaningless groups.
     Gate: PASS iff Reactome exceeds the shuffled grouping by at least 0.005.
     A FAIL means the low-rank projection did the work and the biology did not, which is
     loop 263's result in a new costume and is recorded as such.

  Q4 IS THETA MORE RELIABLE THAN THE DENSE OPERATOR?                 -- requires Q2
     Split each line's genes in half, fit Theta on each, correlate entrywise. Dense W_c
     scored 0.448 on this test and s_c scored 0.9518.
     Gate: PASS iff Theta exceeds the dense operator's reliability by at least 0.05.
     This is the MECHANISM claim -- if fewer parameters do not buy reliability, the reason
     to expect better transfer is gone even if Q5 somehow passes.

  Q5 THE POINT -- DOES IT TRANSFER ACROSS LINES BETTER THAN DENSE?   -- requires Q2
     Best combination of the eight training operators with coefficients fitted on the
     held-out line's own residual, for blocked and for dense. Cheating for both, so the
     comparison is fair, and it is the ceiling for every interpolation method.
     Gate: PASS iff blocked exceeds dense by at least 0.005.

  Q6 WITHIN-LINE, BLOCKED AGAINST DENSE
     Reported, not gated. Says what the 47x parameter reduction costs where data is
     plentiful, which is the price paid for whatever Q5 buys.

  Q7 WHAT THIS CANNOT SHOW
     Stated regardless of outcome.
"""
import json, time
from pathlib import Path
import numpy as np

import lincs_harness as H
from gate_guard import Gates

SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
OUT = "outputs/loop_pathway_operator.json"
SEED = 264264
MIN_PATH = 20
LOOP262_DENSE_HONEST = 0.0673
LOOP262_DENSE_SPAN = 0.0097
LOOP262_DENSE_RELIA = 0.4479
Q1_TOL, Q2_BAR, Q3_BAR, Q4_BAR, Q5_BAR = 0.015, 0.02, 0.005, 0.05, 0.005
LAM = [1e1, 1e2, 1e3, 1e4, 1e5]
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def fit_theta(X, R, M, lam, mu=1e-3):
    """Closed form for R ~= X M Theta M^T."""
    U = X @ M
    A = U.T @ U + lam * np.eye(U.shape[1])
    Bm = M.T @ M + mu * np.eye(M.shape[1])
    return np.linalg.solve(A, U.T @ R @ M) @ np.linalg.inv(Bm)


def diag_fit(U, R):
    den = (U * U).sum(0)
    return np.where(den > 1e-12, (U * R).sum(0) / np.maximum(den, 1e-12), 0.0)


def main():
    t0 = time.time()
    G_ = Gates(emit=say)
    res = {"test": "pathways as a constraint on the operator's parameter space"}
    say("=" * 104)
    say("LOOP 264 -- PATHWAYS AS A CONSTRAINT ON THE OPERATOR, NOT AS A FEATURE")
    say("=" * 104)
    say("     13 annotation sources have returned nothing, and loop 260 says why in one line:")
    say("     the residual is double-centred, so ANY function of the gene alone is capped at")
    say("     0.027%. A pathway membership IS such a function. So it enters here not as an")
    say("     input but as a restriction on the parameter space of the line-specific operator.")
    say("     W_c = M Theta_c M^T. Loop 262's dense W_c is 956,484 parameters at 0.448")
    say("     reliability; this is ~20,000 at K=143. Mechanism under test: fewer parameters ->")
    say("     more reliable -> transfers. CONTROL DECLARED FIRST, from loop 263's lesson: a")
    say("     random grouping of the same sizes is also a low-rank smoother. Q3 decides.")

    D = H.load()
    Pm, pg, pc, LINES, NL = D["Pm"], D["pg"], D["pc"], D["LINES"], D["NL"]
    lm = list(D["lmsym"]); lmi = {g: i for i, g in enumerate(lm)}

    raw = {}
    for line in open(SCR / "ReactomePathways.gmt"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 3: continue
        mem = [g for g in p[2:] if g in lmi]
        if len(mem) >= MIN_PATH: raw[p[0]] = sorted(set(mem))
    names = sorted(raw)
    K = len(names)
    cov = sorted(set(g for v in raw.values() for g in v))
    say(f"     {K} Reactome pathways with >= {MIN_PATH} landmark members, covering "
        f"{len(cov)}/{NL} landmarks ({len(cov)/NL:.0%})")
    say(f"     dense operator {NL*NL:,} parameters -> blocked {K*K:,}   "
        f"({NL*NL/(K*K):.0f}x fewer)")

    def membership(groups):
        M = np.zeros((NL, len(groups)))
        for k, mem in enumerate(groups):
            for g in mem: M[lmi[g], k] = 1.0
        s = M.sum(0); s[s == 0] = 1.0
        return M / np.sqrt(s)

    rng = np.random.default_rng(SEED)
    Mreal = membership([raw[n] for n in names])
    sizes = [len(raw[n]) for n in names]
    pool = list(cov)
    shuf = []
    for sz in sizes:
        shuf.append(list(rng.choice(pool, size=sz, replace=False)))
    Mrand = membership(shuf)
    say(f"     random control: {K} groups, identical size distribution, same "
        f"{len(pool)} landmarks")

    def sc(P, Y): return float(np.nanmean([H.pear(P[i], Y[i]) for i in range(len(Y))]))

    per_line = {}
    for hold in LINES:
        tr = pc != hold
        gm = {}
        for g in D["genes"]:
            m = tr & (pg == g)
            if m.sum(): gm[g] = Pm[m].mean(0)
        grand = Pm[tr].mean(0); lmn = Pm[pc == hold].mean(0)
        XG, Y, A = [], [], []
        for j in np.where(pc == hold)[0]:
            g = pg[j]
            if g not in gm: continue
            dg = gm[g] - grand
            XG.append(dg); Y.append(Pm[j]); A.append(grand + dg + lmn - grand)
        per_line[hold] = tuple(np.stack(v).astype(np.float64) for v in (XG, Y, A))
    say(f"     residuals built for {len(LINES)} lines   [{time.time()-t0:.0f}s]")

    def pick_lam(X, R, fit, apply_, i1, i2):
        best, be = LAM[0], np.inf
        for lam in LAM:
            P = fit(X[i1], R[i1], lam)
            e = float(((apply_(X[i2], P) - R[i2]) ** 2).mean())
            if e < be: be, best = e, lam
        return best

    fit_dense = lambda X, R, lam: np.linalg.solve(X.T @ X + lam * np.eye(NL), X.T @ R)
    app_dense = lambda X, W: X @ W

    def make_blocked(M):
        return (lambda X, R, lam: fit_theta(X, R, M, lam),
                lambda X, T: (X @ M) @ T @ M.T)

    arms = {k: [] for k in ("base", "dense", "block", "blockrand", "blockdiag",
                            "blockranddiag", "span_dense", "span_block")}
    relia = {"dense": [], "block": []}

    for hold in LINES:
        XG, Y, A = per_line[hold]
        R = Y - A
        p = rng.permutation(len(XG)); h1, h2 = p[:len(p)//2], p[len(p)//2:]
        q = rng.permutation(len(h1)); q1, q2 = h1[q[:len(q)//2]], h1[q[len(q)//2:]]
        fb, ab = make_blocked(Mreal)
        fr, ar = make_blocked(Mrand)

        ld = pick_lam(XG, R, fit_dense, app_dense, q1, q2)
        lb = pick_lam(XG, R, fb, ab, q1, q2)
        lr = pick_lam(XG, R, fr, ar, q1, q2)

        W = fit_dense(XG[h1], R[h1], ld)
        Tb = fb(XG[h1], R[h1], lb)
        Tr = fr(XG[h1], R[h1], lr)
        sd = diag_fit(XG[h1], R[h1] - ab(XG[h1], Tb))
        sr = diag_fit(XG[h1], R[h1] - ar(XG[h1], Tr))

        arms["base"].append(sc(A[h2], Y[h2]))
        arms["dense"].append(sc(A[h2] + app_dense(XG[h2], W), Y[h2]))
        arms["block"].append(sc(A[h2] + ab(XG[h2], Tb), Y[h2]))
        arms["blockrand"].append(sc(A[h2] + ar(XG[h2], Tr), Y[h2]))
        arms["blockdiag"].append(sc(A[h2] + ab(XG[h2], Tb) + sd * XG[h2], Y[h2]))
        arms["blockranddiag"].append(sc(A[h2] + ar(XG[h2], Tr) + sr * XG[h2], Y[h2]))

        Wa, Wb = fit_dense(XG[h1], R[h1], ld), fit_dense(XG[h2], R[h2], ld)
        Ta, Tb2 = fb(XG[h1], R[h1], lb), fb(XG[h2], R[h2], lb)
        relia["dense"].append(H.pear(Wa.ravel(), Wb.ravel()))
        relia["block"].append(H.pear(Ta.ravel(), Tb2.ravel()))
        del Wa, Wb

        # ---- cross-line span, blocked vs dense, both with cheating coefficients ----
        Wtr, Ttr = [], []
        for c in LINES:
            if c == hold: continue
            Xc, Yc, Ac = per_line[c]
            Rc = Yc - Ac
            Wtr.append(fit_dense(Xc, Rc, ld)); Ttr.append(fb(Xc, Rc, lb))
        for store, mats, apply_ in (("span_dense", Wtr, app_dense), ("span_block", Ttr, ab)):
            P = np.stack([apply_(XG, m) for m in mats])
            flat = P.reshape(len(mats), -1)
            a = np.linalg.solve(flat @ flat.T + 1e-6 * np.eye(len(mats)), flat @ R.ravel())
            arms[store].append(sc(A + np.tensordot(a, P, axes=(0, 0)), Y))
        del Wtr, Ttr

    m = {k: float(np.mean(v)) for k, v in arms.items()}
    rl = {k: float(np.mean(v)) for k, v in relia.items()}
    say("")
    say(f"     WITHIN-LINE, fit on half a line's genes, scored on the other half:")
    say(f"       {'baseline':34s} {m['base']:.4f}")
    say(f"       {'dense W_c (956,484 params)':34s} {m['dense']:.4f}   {m['dense']-m['base']:+.4f}")
    say(f"       {'pathway blocks alone':34s} {m['block']:.4f}   {m['block']-m['base']:+.4f}")
    say(f"       {'RANDOM blocks alone':34s} {m['blockrand']:.4f}   {m['blockrand']-m['base']:+.4f}")
    say(f"       {'pathway blocks + diagonal':34s} {m['blockdiag']:.4f}   {m['blockdiag']-m['base']:+.4f}")
    say(f"       {'RANDOM blocks + diagonal':34s} {m['blockranddiag']:.4f}   "
        f"{m['blockranddiag']-m['base']:+.4f}")
    say(f"     CROSS-LINE, best combination of the 8 training operators (cheating):")
    say(f"       {'dense':34s} {m['span_dense']-m['base']:+.4f}")
    say(f"       {'pathway-blocked':34s} {m['span_block']-m['base']:+.4f}")
    say(f"     reliability from disjoint gene halves: dense {rl['dense']:.4f}, "
        f"blocked {rl['block']:.4f}")
    res["arms"] = m; res["reliability"] = rl

    say("Q1 DOES THE HARNESS REPRODUCE THE DENSE OPERATOR?")
    d1 = m["dense"] - m["base"]
    say(f"     dense {d1:+.4f} against loop 262's honest {LOOP262_DENSE_HONEST:+.4f}")
    G_.add("Q1", bool(abs(d1 - LOOP262_DENSE_HONEST) <= Q1_TOL), stat=float(d1),
           if_true=lambda: f"Q1 PASS -- reproduces to {abs(d1-LOOP262_DENSE_HONEST):.4f}",
           if_false=lambda: f"Q1 FAIL -- {d1:+.4f} against {LOOP262_DENSE_HONEST:+.4f}")
    res["Q1"] = {"dense_gain": d1}

    say("Q2 DOES THE PATHWAY-BLOCKED OPERATOR WORK AT ALL?")
    d2 = m["blockdiag"] - m["base"]
    say(f"     pathway blocks + diagonal: {d2:+.4f} on held-out genes")
    G_.add("Q2", bool(d2 >= Q2_BAR), stat=float(d2), requires=("Q1",),
           if_true=lambda: f"Q2 PASS -- the blocked operator is worth {d2:+.4f}",
           if_false=lambda: f"Q2 FAIL -- the blocked operator is worth {d2:+.4f}; {K*K:,} "
                            f"parameters over {len(cov)} covered landmarks is too coarse to "
                            f"express the interaction at all")
    res["Q2"] = {"blockdiag_gain": d2}

    say("Q3 LOAD-BEARING -- DO REAL PATHWAYS BEAT RANDOM BLOCKS?")
    d3 = m["blockdiag"] - m["blockranddiag"]
    d3b = m["block"] - m["blockrand"]
    say(f"     Reactome {m['blockdiag']:.4f} vs shuffled blocks {m['blockranddiag']:.4f}   "
        f"{d3:+.4f}")
    say(f"     without the diagonal term: {d3b:+.4f}")
    say(f"     loop 263 found 99% of its neighbour gain survived a shuffled graph; this is")
    say(f"     the same trap in a new costume, and this gate is the thing that catches it")
    G_.add("Q3", bool(d3 >= Q3_BAR), stat=float(d3), requires=("Q2",),
           if_true=lambda: f"Q3 PASS -- real pathways are worth {d3:+.4f} over random blocks "
                           f"of identical size, so the biology and not the rank did the work",
           if_false=lambda: f"Q3 FAIL -- real pathways are worth {d3:+.4f} against random "
                            f"blocks of identical size; the low-rank projection did the work "
                            f"and Reactome did not")
    res["Q3"] = {"delta": d3, "delta_no_diag": d3b,
                 "real": m["blockdiag"], "random": m["blockranddiag"]}

    say("Q4 IS THETA MORE RELIABLE THAN THE DENSE OPERATOR?")
    d4 = rl["block"] - rl["dense"]
    say(f"     blocked {rl['block']:.4f} vs dense {rl['dense']:.4f}   {d4:+.4f}")
    say(f"     (loop 262 measured dense at {LOOP262_DENSE_RELIA:.4f}; s_c reached 0.9518)")
    G_.add("Q4", bool(d4 >= Q4_BAR), stat=float(d4), requires=("Q2",),
           if_true=lambda: f"Q4 PASS -- {NL*NL//(K*K)}x fewer parameters buys {d4:+.4f} of "
                           f"reliability, which is the mechanism this loop is built on",
           if_false=lambda: f"Q4 FAIL -- {NL*NL//(K*K)}x fewer parameters buys {d4:+.4f} of "
                            f"reliability; the reason to expect better transfer is gone")
    res["Q4"] = {"blocked": rl["block"], "dense": rl["dense"], "delta": d4}

    say("Q5 THE POINT -- DOES IT TRANSFER ACROSS LINES BETTER THAN DENSE?")
    sb, sd_ = m["span_block"] - m["base"], m["span_dense"] - m["base"]
    d5 = sb - sd_
    say(f"     blocked span {sb:+.4f} vs dense span {sd_:+.4f}   {d5:+.4f}")
    say(f"     loop 262's dense span was {LOOP262_DENSE_SPAN:+.4f}; both arms cheat equally")
    G_.add("Q5", bool(d5 >= Q5_BAR), stat=float(d5), requires=("Q2",),
           if_true=lambda: f"Q5 PASS -- constraining by pathway buys {d5:+.4f} of cross-line "
                           f"transfer over the dense operator",
           if_false=lambda: f"Q5 FAIL -- constraining by pathway buys {d5:+.4f} of cross-line "
                            f"transfer; the operator stays line-specific however few "
                            f"parameters it has")
    res["Q5"] = {"span_blocked": sb, "span_dense": sd_, "delta": d5}

    say("Q6 WITHIN-LINE, BLOCKED AGAINST DENSE")
    say(f"     dense {m['dense']-m['base']:+.4f} vs blocked+diagonal {d2:+.4f}   "
        f"{d2-(m['dense']-m['base']):+.4f}")
    say(f"     that difference is what the {NL*NL//(K*K)}x parameter reduction COSTS where")
    say(f"     data is plentiful, and it is the price paid for whatever Q5 buys.")
    res["Q6"] = {"dense": m["dense"] - m["base"], "blocked": d2}

    say("Q7 WHAT THIS CANNOT SHOW")
    say(f"     Reactome pathways cover {len(cov)}/{NL} landmarks ({len(cov)/NL:.0%}). The other")
    say(f"     {NL-len(cov)} are reachable only through the diagonal term, so a coupling that")
    say("     lives entirely among uncovered landmarks is invisible to the blocked arm.")
    say("     M Theta M^T is symmetric in its grouping but the blocks are NOT disjoint -- a")
    say("     landmark in several pathways is counted in each, so Theta's entries are not")
    say("     independent and a large Theta does not mean a large pathway-to-pathway effect.")
    say("     Q5's span test asks whether the held-out operator is reachable from EIGHT others.")
    say("     That is a fact about nine cell lines, not about pathways.")
    say("     Everything is on the double-centred residual, so by loop 260 any arm constant")
    say("     across lines is capped at 0.027% however expressive it is.")
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
