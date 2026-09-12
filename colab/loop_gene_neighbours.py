"""
LOOP 263 -- DO FUNCTIONALLY RELATED GENES DEVIATE TOGETHER WITHIN A CELL LINE?

The arc has two axes for sharing information and they behave oppositely:

    across CELL LINES,  9 examples      three loops, all <= 0     (258 J5, 261 M4, 262 N4)
    across GENES within one line, 3,801 examples   +0.0673        (262, held-out genes)

Loop 262's working number IS across-gene pooling: one coupling operator W_c learned from
half a line's genes and applied to genes it never saw. But that operator treats all 3,801
genes as exchangeable -- it learns a single map and applies it uniformly. It has no notion
that two genes might be in the same complex and therefore deviate the same way in the same
cell. This loop tests whether that notion adds anything.

    W_c prediction     R_B ~= x_B W_c            uses the gene's own mean profile
    neighbour          R_B ~= mean of R_A        over measured genes A related to B

The neighbour predictor is only possible because 941 of 978 landmarks are themselves
perturbed here, so relatedness among perturbed genes and relatedness among readouts are the
same relation. Three sources of relatedness are tested separately: DepMap co-dependency
(correlation of gene_effect across 1,178 lines), paralogy, and BioGRID physical interaction.

THE CONFOUND, NAMED BEFORE ANY NUMBER. Related genes have similar MEAN profiles x_A ~ x_B.
Any operator maps similar inputs to similar outputs, so related genes will co-deviate for a
reason that has nothing to do with the interconnection being real. P2 measures the raw
effect and P3 is the gate that matters -- the same comparison after matching pairs on
profile similarity. A P2 PASS with a P3 FAIL means the whole effect was the confound.

WHY THIS IS THE RIGHT AXIS TO PUSH. Loop 262 established the practical shape of the answer:
measure part of a new cell line and the rest becomes predictable. Every attempt to avoid
measuring anything has failed. So the question worth asking is no longer "can we skip the
experiment" but "given a partial experiment, how much can we squeeze from it" -- and that
is a question about structure among GENES, where there are 3,801 examples rather than 9.

PROTOCOL. Gene means come from the OTHER eight lines, so a line never defines its own x_g.
Within the held-out line the genes are split three ways: A fits the operator and supplies
the neighbours, B fits the two blend weights, C is scored and never touched.

GATES, ALL DECLARED BEFORE THE NUMBERS:

  P1 DOES THE OPERATOR ALONE STILL WORK ON A THIRD OF THE GENES?
     Loop 262 got +0.0673 fitting on half a line's genes. This fits on a third.
     Gate: PASS iff the operator alone beats the additive baseline by at least 0.02.

  P2 DO RELATED GENES CO-DEVIATE MORE THAN RANDOM PAIRS?             -- requires P1
     Correlation of R_A and R_B within a line, related pairs against random pairs.
     Gate: PASS iff related exceeds random by at least 0.02.

  P3 DOES THAT SURVIVE MATCHING ON PROFILE SIMILARITY?               -- requires P2
     The same comparison with pairs binned by corr(x_A, x_B) and compared inside bins.
     Gate: PASS iff the matched difference is at least 0.01.
     A P2 PASS with a P3 FAIL means related genes co-deviate only because they look alike
     to begin with, and the interconnection added nothing.

  P4 LOAD-BEARING -- DOES THE NEIGHBOUR PREDICTOR ADD OVER THE OPERATOR?
                                                                     -- requires P1
     Operator + neighbour term against operator alone, on genes neither fitted nor blended.
     Gate: PASS iff the neighbour term adds at least 0.005.

  P5 CONTROL: A SHUFFLED RELATEDNESS GRAPH                           -- requires P4
     The same neighbour predictor with each gene's neighbour list replaced by a random one
     of the same size. VOID if P4 found no margin.
     Gate: PASS iff at most 25% of P4's margin survives.

  P6 WHICH SOURCE OF RELATEDNESS, IF ANY?                            -- requires P1
     Co-dependency, paralogy and BioGRID scored separately.
     Reported, and gated only on whether any source clears P4's bar.

  P7 WHAT THIS CANNOT SHOW
     Stated regardless of outcome.
"""
import json, time, gzip, csv
from pathlib import Path
import numpy as np

import lincs_harness as H
from gate_guard import Gates

SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
OUT = "outputs/loop_gene_neighbours.json"
SEED = 263263
TOPK = 25
LAM_W = [1e2, 1e3, 1e4, 1e5]
P1_BAR, P2_BAR, P3_BAR, P4_BAR, P5_MAX = 0.02, 0.02, 0.01, 0.005, 0.25
NPAIR = 40000
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def zrow(M):
    M = M - M.mean(1, keepdims=True)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def main():
    t0 = time.time()
    G_ = Gates(emit=say)
    res = {"test": "do functionally related genes deviate together within a cell line?"}
    say("=" * 104)
    say("LOOP 263 -- DO FUNCTIONALLY RELATED GENES DEVIATE TOGETHER WITHIN A CELL LINE?")
    say("=" * 104)
    say("     Across CELL LINES sharing has failed three times (258 J5, 261 M4, 262 N4).")
    say("     Across GENES within a line it WORKS: loop 262's +0.0673 is exactly that -- one")
    say("     operator learned from half a line's genes, applied to genes it never saw.")
    say("     But that operator treats all 3,801 genes as exchangeable. This asks whether")
    say("     knowing WHICH genes are functionally related adds anything on top.")
    say("     CONFOUND DECLARED FIRST: related genes have similar mean profiles, and any")
    say("     operator maps similar inputs to similar outputs. P3, not P2, is the real test.")

    D = H.load()
    Pm, pg, pc, LINES, NL = D["Pm"], D["pg"], D["pc"], D["LINES"], D["NL"]
    gset = list(D["genes"]); gidx = {g: i for i, g in enumerate(gset)}
    say(f"     {len(pg):,} pairs, {len(gset):,} genes, {NL} landmarks, {len(LINES)} lines")

    # ---------------- relatedness sources ----------------
    ge = np.load(SCR / "depmap" / "gene_effect.npz", allow_pickle=True)
    GE = np.nan_to_num(np.asarray(ge["E"], np.float32))
    gn = np.array([str(x) for x in ge["genes"]])
    keep = [i for i, g in enumerate(gn) if g in gidx]
    sub = zrow(GE[:, keep].T.astype(np.float64))
    codep_genes = gn[keep]
    CD = sub @ sub.T
    np.fill_diagonal(CD, -np.inf)
    say(f"     co-dependency matrix over {len(keep):,} genes shared with LINCS")

    nbrs = {"codep": {}, "paralog": {}, "biogrid": {}}
    order = np.argsort(-CD, axis=1)[:, :TOPK]
    for i, g in enumerate(codep_genes):
        nbrs["codep"][g] = [codep_genes[j] for j in order[i]]
    with open(SCR / "para" / "paralogs.tsv") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2 or not p[0] or not p[1]: continue
            if p[0] in gidx and p[1] in gidx:
                nbrs["paralog"].setdefault(p[0], []).append(p[1])
                nbrs["paralog"].setdefault(p[1], []).append(p[0])
    with gzip.open(SCR / "biogrid_hs_edges.tsv.gz", "rt") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2: continue
            if p[0] in gidx and p[1] in gidx and p[0] != p[1]:
                nbrs["biogrid"].setdefault(p[0], []).append(p[1])
                nbrs["biogrid"].setdefault(p[1], []).append(p[0])
    for k in nbrs:
        nbrs[k] = {g: list(dict.fromkeys(v))[:TOPK] for g, v in nbrs[k].items() if v}
        say(f"     {k:9s} {len(nbrs[k]):,} genes with neighbours, median degree "
            f"{int(np.median([len(v) for v in nbrs[k].values()])) if nbrs[k] else 0}")
    res["sources"] = {k: len(v) for k, v in nbrs.items()}

    rng = np.random.default_rng(SEED)

    def sc(P, Y): return float(np.nanmean([H.pear(P[i], Y[i]) for i in range(len(Y))]))

    # ---------------- per-line residuals with gene means from the OTHER lines -------------
    per_line = {}
    for hold in LINES:
        tr = pc != hold
        gm = {}
        for g in gset:
            m = tr & (pg == g)
            if m.sum(): gm[g] = Pm[m].mean(0)
        grand = Pm[tr].mean(0); lmn = Pm[pc == hold].mean(0)
        XG, Y, A, GN = [], [], [], []
        for j in np.where(pc == hold)[0]:
            g = pg[j]
            if g not in gm: continue
            dg = gm[g] - grand
            XG.append(dg); Y.append(Pm[j]); A.append(grand + dg + lmn - grand); GN.append(g)
        per_line[hold] = tuple(np.stack(v).astype(np.float64) for v in (XG, Y, A)) + (np.array(GN),)
    say(f"     residuals built for {len(per_line)} lines   [{time.time()-t0:.0f}s]")

    # ---------------- P2 / P3: pairwise co-deviation, raw and profile-matched -------------
    say("     sampling gene pairs for the co-deviation test ...")
    rel_cd, rel_xs, ran_cd, ran_xs = [], [], [], []
    for hold in LINES:
        XG, Y, A, GN = per_line[hold]
        R = zrow(Y - A); X = zrow(XG)
        pos = {g: i for i, g in enumerate(GN)}
        cand = [(a, b) for a, vs in nbrs["codep"].items() if a in pos
                for b in vs[:5] if b in pos]
        if not cand: continue
        pick = rng.choice(len(cand), size=min(NPAIR // len(LINES), len(cand)), replace=False)
        for k in pick:
            a, b = cand[k]
            i, j = pos[a], pos[b]
            rel_cd.append(float(R[i] @ R[j])); rel_xs.append(float(X[i] @ X[j]))
        n = len(pick)
        ii = rng.integers(0, len(GN), n); jj = rng.integers(0, len(GN), n)
        for i, j in zip(ii, jj):
            if i == j: continue
            ran_cd.append(float(R[i] @ R[j])); ran_xs.append(float(X[i] @ X[j]))
    rel_cd, rel_xs = np.array(rel_cd), np.array(rel_xs)
    ran_cd, ran_xs = np.array(ran_cd), np.array(ran_xs)
    say(f"     {len(rel_cd):,} related pairs, {len(ran_cd):,} random pairs")

    say("P1 DOES THE OPERATOR ALONE STILL WORK ON A THIRD OF THE GENES?")
    # ---------------- P1 / P4 / P5: the predictive test ----------------
    def evaluate(source, shuffle=False):
        base, wonly, comb = [], [], []
        for hold in LINES:
            XG, Y, A, GN = per_line[hold]
            R = Y - A
            p = rng.permutation(len(GN)); t = len(p) // 3
            Ai, Bi, Ci = p[:t], p[t:2 * t], p[2 * t:]
            G1 = XG[Ai].T @ XG[Ai]; B1 = XG[Ai].T @ R[Ai]
            q = rng.permutation(len(Ai)); q1, q2 = Ai[q[:len(q)//2]], Ai[q[len(q)//2:]]
            Gq = XG[q1].T @ XG[q1]; Bq = XG[q1].T @ R[q1]
            best, be = LAM_W[0], np.inf
            for lam in LAM_W:
                Wq = np.linalg.solve(Gq + lam * np.eye(NL), Bq)
                e = float(((XG[q2] @ Wq - R[q2]) ** 2).mean())
                if e < be: be, best = e, lam
            W = np.linalg.solve(G1 + best * np.eye(NL), B1)
            posA = {}
            for k in Ai: posA.setdefault(GN[k], []).append(k)
            names = list(posA)
            def nbr_pred(idx):
                out = np.zeros((len(idx), NL))
                for r, k in enumerate(idx):
                    g = GN[k]
                    nb = nbrs[source].get(g, [])
                    if shuffle:
                        nb = [names[x] for x in rng.integers(0, len(names), len(nb))] if names else []
                    rowsk = [m for h in nb if h != g for m in posA.get(h, [])]
                    if rowsk: out[r] = R[rowsk].mean(0)
                return out
            PB_w, PB_n = XG[Bi] @ W, nbr_pred(Bi)
            Z = np.stack([PB_w.ravel(), PB_n.ravel()], 1)
            ab = np.linalg.solve(Z.T @ Z + 1e-6 * np.eye(2), Z.T @ R[Bi].ravel())
            PC_w, PC_n = XG[Ci] @ W, nbr_pred(Ci)
            base.append(sc(A[Ci], Y[Ci]))
            wonly.append(sc(A[Ci] + PC_w, Y[Ci]))
            comb.append(sc(A[Ci] + ab[0] * PC_w + ab[1] * PC_n, Y[Ci]))
        return float(np.mean(base)), float(np.mean(wonly)), float(np.mean(comb))

    b0, w0, c0 = evaluate("codep")
    say(f"     baseline {b0:.4f}   operator alone {w0:.4f}   {w0-b0:+.4f}")
    say(f"     loop 262 got +0.0673 fitting on HALF the genes; this fits on a third")
    G_.add("P1", bool(w0 - b0 >= P1_BAR), stat=float(w0 - b0),
           if_true=lambda: f"P1 PASS -- the operator alone is worth {w0-b0:+.4f} on genes it "
                           f"never saw, so the across-gene axis still carries the signal",
           if_false=lambda: f"P1 FAIL -- {w0-b0:+.4f} on a third of the genes")
    res["P1"] = {"base": b0, "w_only": w0, "gain": w0 - b0}

    say("P2 DO RELATED GENES CO-DEVIATE MORE THAN RANDOM PAIRS?")
    d2 = float(rel_cd.mean() - ran_cd.mean())
    say(f"     related pairs  {rel_cd.mean():+.4f}   random pairs {ran_cd.mean():+.4f}   "
        f"{d2:+.4f}")
    say(f"     their MEAN PROFILES: related {rel_xs.mean():+.4f}, random {ran_xs.mean():+.4f}")
    G_.add("P2", bool(d2 >= P2_BAR), stat=d2, requires=("P1",),
           if_true=lambda: f"P2 PASS -- related genes co-deviate {d2:+.4f} more than random",
           if_false=lambda: f"P2 FAIL -- related genes co-deviate {d2:+.4f} more than random")
    res["P2"] = {"related": float(rel_cd.mean()), "random": float(ran_cd.mean()), "delta": d2,
                 "x_related": float(rel_xs.mean()), "x_random": float(ran_xs.mean())}

    say("P3 DOES THAT SURVIVE MATCHING ON PROFILE SIMILARITY?")
    edges = np.quantile(np.concatenate([rel_xs, ran_xs]), np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    diffs, wts = [], []
    for k in range(10):
        mr = (rel_xs >= edges[k]) & (rel_xs < edges[k + 1])
        mn = (ran_xs >= edges[k]) & (ran_xs < edges[k + 1])
        if mr.sum() >= 30 and mn.sum() >= 30:
            diffs.append(rel_cd[mr].mean() - ran_cd[mn].mean()); wts.append(mr.sum())
    d3 = float(np.average(diffs, weights=wts)) if diffs else 0.0
    say(f"     matched inside {len(diffs)} bins of corr(x_A, x_B): {d3:+.4f}")
    say(f"     raw was {d2:+.4f}, so profile similarity accounts for "
        f"{(1 - d3/d2)*100 if abs(d2) > 1e-9 else 0:.0f}% of it")
    G_.add("P3", bool(d3 >= P3_BAR), stat=d3, requires=("P2",),
           if_true=lambda: f"P3 PASS -- {d3:+.4f} survives matching on profile similarity",
           if_false=lambda: f"P3 FAIL -- only {d3:+.4f} survives matching; related genes "
                            f"co-deviate because they LOOK alike, not because of the relation")
    res["P3"] = {"matched_delta": d3, "raw_delta": d2, "bins": len(diffs)}

    say("P4 LOAD-BEARING -- DOES THE NEIGHBOUR PREDICTOR ADD OVER THE OPERATOR?")
    d4 = c0 - w0
    say(f"     operator alone {w0:.4f}   operator + co-dependency neighbours {c0:.4f}   {d4:+.4f}")
    say(f"     scored on genes used neither to fit the operator nor to fit the blend weights")
    G_.add("P4", bool(d4 >= P4_BAR), stat=float(d4), requires=("P1",),
           if_true=lambda: f"P4 PASS -- knowing which genes are related is worth {d4:+.4f} "
                           f"beyond a uniform operator",
           if_false=lambda: f"P4 FAIL -- the neighbour term is worth {d4:+.4f} beyond a uniform "
                            f"operator; the operator already captures what relatedness offers")
    res["P4"] = {"combined": c0, "w_only": w0, "delta": d4}

    say("P5 CONTROL: A SHUFFLED RELATEDNESS GRAPH")
    if d4 < P4_BAR:
        G_.add("P5", False, stat=float(d4), requires=("P4",), void_if=True,
               void_reason=f"P4's margin is {d4:+.4f}; there is nothing for a shuffle to collapse")
    else:
        _, _, cs = evaluate("codep", shuffle=True)
        d5 = cs - w0
        f5 = d5 / d4
        say(f"     random neighbours of the same count: {d5:+.4f} against a real {d4:+.4f} "
            f"({f5:.0%})")
        G_.add("P5", bool(f5 <= P5_MAX), stat=float(f5), requires=("P4",),
               if_true=lambda: f"P5 PASS -- collapses to {f5:.0%} on a shuffled graph",
               if_false=lambda: f"P5 FAIL -- {f5:.0%} survives a shuffled graph, so the gain is "
                                f"from averaging other genes at all, not from WHICH genes")
        res["P5"] = {"shuffled": d5, "fraction": f5}

    say("P6 WHICH SOURCE OF RELATEDNESS, IF ANY?")
    per_src = {"codep": d4}
    for s in ("paralog", "biogrid"):
        _, ws, cs = evaluate(s)
        per_src[s] = cs - ws
        say(f"     {s:9s} operator {ws:.4f} -> combined {cs:.4f}   {cs-ws:+.4f}")
    say(f"     {'codep':9s} {d4:+.4f}")
    best_src = max(per_src, key=per_src.get)
    spread = max(per_src.values()) - min(per_src.values())
    say(f"     spread across the three sources: {spread:.4f}. Three unrelated graphs agreeing")
    say(f"     to within {spread:.4f} is what 'the graph does not matter' looks like.")
    # DEFECT C: this gate asked WHICH source wins. If P5 showed a shuffled graph does just as
    # well, then no source "wins" in any sense the question meant, and a PASS here would be a
    # true sentence about a meaningless comparison. It must see P5's verdict.
    G_.add("P6", bool(per_src[best_src] >= P4_BAR), stat=float(per_src[best_src]),
           requires=("P1", "P5"),
           if_true=lambda: f"P6 PASS -- {best_src} clears the bar at {per_src[best_src]:+.4f} "
                           f"and P5 confirmed a shuffled graph does NOT",
           if_false=lambda: f"P6 FAIL -- the best source is {best_src} at "
                            f"{per_src[best_src]:+.4f}; no relatedness graph on disk adds to a "
                            f"uniform operator")
    res["P6"] = {k: float(v) for k, v in per_src.items()}

    say("P7 WHAT THIS CANNOT SHOW")
    say("     The neighbour term is a MEAN over related genes. A gene whose partners deviate in")
    say("     opposing directions averages to nothing, so a real but sign-structured relation")
    say("     would read as a null here.")
    say("     Relatedness is taken from DepMap, Ensembl paralogy and BioGRID, all of which are")
    say("     cell-line agnostic. If the relation that matters is itself line-specific, none of")
    say("     these three can express it -- and loop 262 found the operator IS line-specific.")
    say("     Co-dependency is computed from the same DepMap gene_effect matrix that supplied")
    say("     the annotation which failed in loops 258, 261 and 262. It is a different USE of")
    say("     that data -- gene-to-gene rather than line-to-line -- but not independent of it.")
    say("     P1 fits the operator on a THIRD of each line's genes rather than a half, so its")
    say("     number is not directly comparable to loop 262's +0.0673.")
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
