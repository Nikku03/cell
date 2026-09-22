"""
LOOP 261 -- CAN THE PER-LINE SCALE VECTOR s_c BE PREDICTED FROM ANNOTATION?

Loop 259 was designed and then NOT run, because three of its five branches turned out to
span the same function space and closed-form solutions predicted the whole thing in
minutes. What that analysis left behind is a sharply posed target.

For a FIXED cell line, every branch of the form "scale the gene's mean profile per
landmark" collapses to the same object:

    B1 = g * xg          B2 = (W2 z_c) * xg          B4 = (c * xlm) * xg
                    all of them are      s_c * xg

and the oracle numbers confirmed it empirically -- the per-line-scale oracle and the
triple-product oracle both scored 0.4762, identical to four decimals. So there is ONE
quantity worth predicting, not five: a 978-vector per cell line.

MEASURED ALREADY, and this loop is built on it:
    oracle s_c (fitted on the held-out line's OWN answers, i.e. cheating)   +0.0247
    every out-of-sample attempt at it so far                                 0.0000
      B1 gene-only closed form   +0.0001
      B4 triple product          -0.0003
      B3 rank-64 gated operator  -0.0152   (loop 257)
      loop 258's aided network   -0.0001   (and its alpha could not agree on a SIGN:
                                            +0.957 to -0.405 across nine folds)

So +0.0247 of headroom exists and nothing has taken any of it. This loop asks the question
directly instead of through a network: extract s_c* for every line, regress the cell
annotation z_c onto it, and see whether the map transfers to a line it never saw.

s_c* is defined per landmark by least squares on that line's own rows:
    s_c*[l] = sum_i xg[i,l] R[i,l] / sum_i xg[i,l]^2
with gene means, grand mean and the residual R all computed from the OTHER lines, so the
only thing the held-out line contributes to its own target is the fit of s.

THE NULL THAT MATTERS, DECLARED FIRST: the s_c vectors may share a large common component.
A constant predictor -- the MEAN of the training lines' s_c, using no annotation at all --
would then transfer perfectly well and buy most of the headroom. That would be a real
finding about the data and NOT evidence that annotation predicts anything. M3 measures it
and M4 is scored against M3, not against the baseline.

GATES, ALL DECLARED BEFORE THE NUMBERS:

  M1 DOES THE ORACLE REPRODUCE +0.0247?
     Sanity: s_c* fitted on the held-out line's own answers, applied to that line.
     Gate: PASS iff within 0.005 of the +0.0247 already measured.

  M2 IS s_c ITSELF RELIABLE, OR IS IT NOISE?                       -- requires M1
     Split the line's GENES in half, fit s_c on each half, correlate the two 978-vectors.
     If s_c cannot be reproduced from disjoint halves of its own line, nothing can predict
     it and every gate below is measuring noise.
     Gate: PASS iff the mean split-half correlation exceeds 0.30.

  M3 DOES A CONSTANT, ANNOTATION-FREE SCALE ALREADY TRANSFER?      -- requires M1, M2
     Predict the held-out line's scale as the MEAN of the training lines' s_c. No z_c, no
     model, no line-specific information whatsoever.
     Gate: PASS iff it beats the additive baseline by at least 0.005.
     A PASS here means the scale is mostly SHARED across lines, and any annotation model
     must be scored against this, not against the baseline.

  M4 LOAD-BEARING -- DOES ANNOTATION PREDICT THE LINE-SPECIFIC PART?   -- requires M1, M2
     Ridge from z_c to s_c*, fitted on the 8 training lines with lambda chosen by an inner
     leave-one-line-out among those 8, then applied to the held-out line.
     Scored against M3's constant predictor, because that is the part annotation has to
     earn. Gate: PASS iff the ridge exceeds the constant predictor by at least 0.005.

  M5 PERMUTED CONTROL                                              -- requires M4
     The same ridge with the mapping between z_c and s_c* shuffled among training lines.
     VOID if M4 found no margin, because there is then nothing to collapse.
     Gate: PASS iff at most 25% of M4's margin survives the shuffle.

  M6 HOW MUCH OF THE HEADROOM IS RECOVERED?                        -- requires M1
     The best out-of-sample arm as a fraction of the oracle's +0.0247. Reported and gated
     at 20%, so that a technically-significant but negligible recovery cannot be read as
     success.

  M7 HOW WELL IS s_c PREDICTED AS A VECTOR?
     Direct leave-one-line-out correlation between predicted and true s_c, for the ridge
     and for the constant predictor. Reported, not gated -- it is the diagnostic that says
     WHY M4 landed where it did.

  M8 WHAT THIS CANNOT SHOW
     Stated regardless of outcome.
"""
import json, time, csv
from pathlib import Path
import numpy as np

import lincs_harness as H
from gate_guard import Gates

SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
OUT = "outputs/loop_scale_probe.json"
SEED = 261261
NPC_EXPR, NPC_DEP = 50, 50
ORACLE_REF = 0.0247
M1_TOL, M2_BAR, M3_BAR, M4_BAR, M5_MAX, M6_BAR = 0.005, 0.30, 0.005, 0.005, 0.25, 0.20
LAMBDAS = [1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 1e4]
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def diag_fit(U, R):
    """s[l] = sum_i U[i,l] R[i,l] / sum_i U[i,l]^2 -- the least-squares per-landmark scale."""
    den = (U * U).sum(0)
    return np.where(den > 1e-12, (U * R).sum(0) / np.maximum(den, 1e-12), 0.0)


def ridge_fit(Z, S, lam):
    """Z: (n, d) annotations, S: (n, 978) targets. Returns W with S ~= Z @ W."""
    d = Z.shape[1]
    return np.linalg.solve(Z.T @ Z + lam * np.eye(d), Z.T @ S)


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "can the per-line landmark scale s_c be predicted from cell annotation?"}
    say("=" * 104)
    say("LOOP 261 -- CAN THE PER-LINE SCALE VECTOR s_c BE PREDICTED FROM ANNOTATION?")
    say("=" * 104)
    say("     Loop 259's branches collapsed: for a FIXED line, B1, B2 and B4 are all s_c * xg,")
    say("     and their oracles matched to four decimals (0.4762 both). One 978-vector per line")
    say("     is the whole target. Oracle s_c is worth +0.0247; every out-of-sample attempt so")
    say("     far has taken 0.0000 of it (B1 +0.0001, B4 -0.0003, B3 -0.0152, loop 258 -0.0001).")
    say("     DECLARED FIRST: if the s_c share a common component, a CONSTANT predictor using no")
    say("     annotation transfers and buys the headroom. M4 is scored against that, not the")
    say("     baseline, because the shared part is not something annotation earned.")

    D = H.load()
    Pm, pg, pc, LINES, NL = D["Pm"], D["pg"], D["pc"], D["LINES"], D["NL"]
    say(f"     {len(pg):,} (gene, line) pairs, {len(D['genes']):,} genes, {NL} landmarks, "
        f"{len(LINES)} cell lines")

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
    say(f"     annotation z_c: {ZC.shape[1]} dims per line "
        f"({NPC_EXPR} expression PCs + {NPC_DEP} dependency PCs + mutation burden)")

    def fold(hold):
        """Everything -- gene means, grand, line means -- from the OTHER lines."""
        tr = pc != hold
        gm = {}
        for g in D["genes"]:
            m = tr & (pg == g)
            if m.sum(): gm[g] = Pm[m].mean(0)
        grand = Pm[tr].mean(0); lmean = {l: Pm[pc == l].mean(0) for l in LINES}

        def rows(line):
            XG, Y, A, GN = [], [], [], []
            for j in np.where(pc == line)[0]:
                g = pg[j]
                if g not in gm: continue
                dg = gm[g] - grand
                XG.append(dg); Y.append(Pm[j]); A.append(grand + dg + (lmean[line] - grand))
                GN.append(g)
            return (*(np.stack(v).astype(np.float64) for v in (XG, Y, A)), np.array(GN))
        return rows

    def sc(P, Y): return float(np.nanmean([H.pear(P[i], Y[i]) for i in range(len(Y))]))

    rng = np.random.default_rng(SEED)
    arms = {k: [] for k in ("base", "oracle", "const", "ridge", "perm")}
    splithalf, probe_r, chosen = [], {"ridge": [], "const": []}, []
    S_all = {}

    for hold in LINES:
        rows = fold(hold)
        # --- target matrix S over the 8 training lines, all computed under THIS fold ---
        Str, Ztr, names = [], [], []
        for c in LINES:
            if c == hold: continue
            xg, y, a, gn = rows(c)
            Str.append(diag_fit(xg, y - a)); Ztr.append(ZC[LINES.index(c)]); names.append(c)
        Str = np.stack(Str); Ztr = np.stack(Ztr)
        mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
        Ztr_z = (Ztr - mu) / sd
        zh = (ZC[LINES.index(hold)] - mu) / sd

        xg, y, a, gn = rows(hold)
        s_true = diag_fit(xg, y - a)
        S_all[hold] = s_true

        # --- M2: split the held-out line's GENES in half and refit s on each half ---
        perm = rng.permutation(len(xg)); h1, h2 = perm[:len(perm) // 2], perm[len(perm) // 2:]
        sa, sb = diag_fit(xg[h1], (y - a)[h1]), diag_fit(xg[h2], (y - a)[h2])
        splithalf.append(H.pear(sa, sb))

        # --- lambda by inner leave-one-line-out among the 8 training lines ---
        best_lam, best_err = LAMBDAS[0], np.inf
        for lam in LAMBDAS:
            err = 0.0
            for i in range(len(Str)):
                k = [j for j in range(len(Str)) if j != i]
                W = ridge_fit(Ztr_z[k], Str[k], lam)
                err += float(((Ztr_z[i] @ W - Str[i]) ** 2).mean())
            if err < best_err: best_err, best_lam = err, lam
        chosen.append(best_lam)

        W = ridge_fit(Ztr_z, Str, best_lam)
        s_ridge = zh @ W
        s_const = Str.mean(0)
        Wp = ridge_fit(Ztr_z, Str[rng.permutation(len(Str))], best_lam)
        s_perm = zh @ Wp

        arms["base"].append(sc(a, y))
        arms["oracle"].append(sc(a + s_true * xg, y))
        arms["const"].append(sc(a + s_const * xg, y))
        arms["ridge"].append(sc(a + s_ridge * xg, y))
        arms["perm"].append(sc(a + s_perm * xg, y))
        probe_r["ridge"].append(H.pear(s_ridge, s_true))
        probe_r["const"].append(H.pear(s_const, s_true))

    m = {k: float(np.mean(v)) for k, v in arms.items()}
    say(f"     lambda chosen per fold: " + ", ".join(f"{x:g}" for x in chosen))
    say("")
    say(f"     {'baseline':34s} {m['base']:.4f}")
    say(f"     {'+ ORACLE s_c (cheating)':34s} {m['oracle']:.4f}   {m['oracle']-m['base']:+.4f}")
    say(f"     {'+ constant s (no annotation)':34s} {m['const']:.4f}   {m['const']-m['base']:+.4f}")
    say(f"     {'+ ridge from z_c':34s} {m['ridge']:.4f}   {m['ridge']-m['base']:+.4f}")
    say(f"     {'+ ridge, line labels shuffled':34s} {m['perm']:.4f}   {m['perm']-m['base']:+.4f}")
    res["arms"] = m
    res["per_fold"] = {k: [float(x) for x in v] for k, v in arms.items()}
    res["lambda"] = chosen

    say("M1 DOES THE ORACLE REPRODUCE +0.0247?")
    d1 = m["oracle"] - m["base"]
    say(f"     oracle s_c fitted on the held-out line's own answers: {d1:+.4f} "
        f"against the {ORACLE_REF:+.4f} already measured")
    G.add("M1", bool(abs(d1 - ORACLE_REF) <= M1_TOL), stat=float(d1),
          if_true=lambda: f"M1 PASS -- reproduces to {abs(d1 - ORACLE_REF):.4f}; there is "
                          f"{d1:+.4f} of headroom and the target is correctly extracted",
          if_false=lambda: f"M1 FAIL -- {d1:+.4f} against {ORACLE_REF:+.4f}; the target is not "
                           f"the same quantity the earlier analysis measured")
    res["M1"] = {"oracle_gain": d1, "reference": ORACLE_REF}

    say("M2 IS s_c ITSELF RELIABLE, OR IS IT NOISE?")
    sh = float(np.mean(splithalf))
    say(f"     split-half over GENES within each line, correlation of the two s_c estimates:")
    say(f"     " + ", ".join(f"{x:.3f}" for x in splithalf))
    say(f"     mean {sh:.4f}")
    G.add("M2", bool(sh > M2_BAR), stat=sh, requires=("M1",),
          if_true=lambda: f"M2 PASS -- s_c reproduces at {sh:.3f} from disjoint halves of its "
                          f"own line, so it is a real quantity and worth trying to predict",
          if_false=lambda: f"M2 FAIL -- s_c reproduces at only {sh:.3f} from disjoint gene "
                           f"halves; it is largely noise and every gate below measures noise")
    res["M2"] = {"split_half": sh, "per_fold": [float(x) for x in splithalf]}

    say("M3 DOES A CONSTANT, ANNOTATION-FREE SCALE ALREADY TRANSFER?")
    d3 = m["const"] - m["base"]
    say(f"     mean of the 8 training lines' s_c, applied to the held-out line: {d3:+.4f}")
    say(f"     this uses NO annotation and no line-specific information at all")
    G.add("M3", bool(d3 >= M3_BAR), stat=float(d3), requires=("M1", "M2"),
          if_true=lambda: f"M3 PASS -- a shared, line-independent scale is worth {d3:+.4f}. The "
                          f"scale is mostly COMMON across lines, and annotation must beat this.",
          if_false=lambda: f"M3 FAIL -- a shared scale is worth {d3:+.4f}; there is no "
                           f"transferable common component, so the whole {d1:+.4f} is "
                           f"line-specific")
    res["M3"] = {"delta_vs_baseline": d3}

    say("M4 LOAD-BEARING -- DOES ANNOTATION PREDICT THE LINE-SPECIFIC PART?")
    d4 = m["ridge"] - m["const"]
    say(f"     ridge from z_c {m['ridge']:.4f} vs constant predictor {m['const']:.4f}   "
        f"{d4:+.4f}")
    say(f"     scored against the CONSTANT, not the baseline: the shared component is not")
    say(f"     something the annotation earned, and crediting it there would be double-counting")
    G.add("M4", bool(d4 >= M4_BAR), stat=float(d4), requires=("M1", "M2"),
          if_true=lambda: f"M4 PASS -- annotation is worth {d4:+.4f} beyond a constant scale",
          if_false=lambda: f"M4 FAIL -- annotation is worth {d4:+.4f} beyond a constant scale; "
                           f"z_c does not predict which line gets which scaling")
    res["M4"] = {"delta_vs_constant": d4}

    say("M5 PERMUTED CONTROL")
    if d4 < M4_BAR:
        G.add("M5", False, stat=float(d4), requires=("M4",), void_if=True,
              void_reason=f"M4's margin is {d4:+.4f}; there is nothing for a shuffle to collapse")
    else:
        d5 = m["perm"] - m["const"]
        f5 = d5 / d4
        say(f"     line labels shuffled: {d5:+.4f} against a real {d4:+.4f} ({f5:.0%})")
        G.add("M5", bool(f5 <= M5_MAX), stat=float(f5), requires=("M4",),
              if_true=lambda: f"M5 PASS -- collapses to {f5:.0%} when z_c is paired with the "
                              f"wrong line's scale",
              if_false=lambda: f"M5 FAIL -- {f5:.0%} survives shuffling which line is which")
        res["M5"] = {"shuffled": d5, "fraction": f5}

    say("M6 HOW MUCH OF THE HEADROOM IS RECOVERED?")
    best = max(m["const"], m["ridge"]) - m["base"]
    frac = best / d1 if d1 > 1e-9 else 0.0
    say(f"     best out-of-sample arm {best:+.4f} of the oracle's {d1:+.4f} = {frac:.1%}")
    G.add("M6", bool(frac >= M6_BAR), stat=float(frac), requires=("M1",),
          if_true=lambda: f"M6 PASS -- {frac:.1%} of the headroom is recovered out of sample",
          if_false=lambda: f"M6 FAIL -- only {frac:.1%} of the {d1:+.4f} headroom is recovered; "
                           f"the scale exists and remains unpredictable")
    res["M6"] = {"fraction_recovered": frac, "best_out_of_sample": best}

    say("M7 HOW WELL IS s_c PREDICTED AS A VECTOR?")
    pr, pc_ = float(np.mean(probe_r["ridge"])), float(np.mean(probe_r["const"]))
    say(f"     leave-one-line-out correlation between predicted and true s_c:")
    say(f"       ridge from z_c      {pr:+.4f}")
    say(f"       constant predictor  {pc_:+.4f}")
    say(f"     split-half reliability of s_c itself was {sh:.4f}, which is the ceiling both")
    say(f"     of these live under. Reported, not gated.")
    res["M7"] = {"ridge_r": pr, "const_r": pc_, "reliability": sh,
                 "ridge_per_fold": [float(x) for x in probe_r["ridge"]],
                 "const_per_fold": [float(x) for x in probe_r["const"]]}

    say("M8 WHAT THIS CANNOT SHOW")
    say("     Nine cell lines. The ridge fits 101 annotation dims from EIGHT training lines, so")
    say("     it is underdetermined by construction and lambda is doing most of the work; the")
    say("     inner leave-one-line-out picks it honestly but cannot manufacture degrees of")
    say("     freedom. A FAIL here does not prove annotation is uninformative in general -- it")
    say("     proves it is not learnable from eight examples in this form.")
    say("     s_c is a DIAGONAL scaling. It is the exact object loop 259's branches spanned, but")
    say("     it cannot express a rotation, and a genuinely off-diagonal interaction would be")
    say("     invisible to every number here.")
    say("     s_c* for the held-out line is fitted on that line's own residual, so the oracle")
    say("     is an upper bound that no honest model can reach, not a target to aim at.")
    say("     978 landmarks and shRNA, not a transcriptome and not a clean knockout, and every")
    say("     number lives under loop 254's construct ceiling.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary(seconds=res["seconds"])
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
