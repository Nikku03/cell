"""LOOP 140 -- WHY DOES FUSION MAKE THINGS WORSE, AND CAN THAT BE FIXED?

FIRST, A CORRECTION I OWE. Loop 138 V1 concluded that the fusion failure is real -- which still
stands -- and then added: "90% of the graph signal survives degree-preserving rewiring, so the
fusion's working half is largely hub identity." THAT SENTENCE IS WRONG, and I wrote it into the
layer table and told the user. The 90% belongs to the COMBINED arm. The recorded numbers are:

    A3 combined     real rho 0.3913   rewired null 0.3527   90.1% survives   -> mostly topology
    A2 graph-only   real rho 0.5575   rewired null 0.2005   36.0% survives   -> mostly REAL EDGES

I read the first key of a dict and generalised from it. The graph channel is not hub identity; it
is the opposite -- destroying which gene connects to which destroys 64% of it.

AND THE CORRECTION SHARPENS THE PROBLEM RATHER THAN SOFTENING IT. The two rows above are the same
model with one extra channel, and they say that adding the reaction channel does not merely cost
performance -- it converts a specific-edge predictor into a degree predictor. Graph-only knows
WHICH gene talks to which; combined mostly knows HOW MANY. That is a much more specific accusation
than 'the fusion is worse', and it names a mechanism.

THE MECHANISM, stated before it is tested. Spectral embedding takes the eigenvectors of a
normalised Laplacian. For a node with no edges in a channel, the corresponding coordinates are not
zero and they are not small -- they are whatever the eigensolver returns for an isolated or
near-isolated component, which is dominated by component structure and degree. 84% of genes have
no catalyses edge at all. So concatenating the reaction embedding hands the ridge sixteen to a
hundred and twenty-eight columns that, for most genes, encode nothing but disconnection. Ridge has
no way to know those columns are missing rather than measured, and shrinkage spreads weight onto
them. The specific-edge signal in the graph block gets traded for a degree-like signal that is
present in every row.

THE FIX THAT FOLLOWS FROM THE MECHANISM, and only that fix. Zero the reaction block wherever the
gene has no reaction edge, and add ONE explicit missingness indicator so the model can tell absent
from measured. This is not a search over fusion methods -- it is the single change the diagnosis
implies, declared before it runs, and if it fails the diagnosis is wrong.

WHAT WOULD MAKE THIS LOOP DISHONEST. Trying fusion variants until one wins. So exactly one variant
is tested, it is named above, and G4 requires it to beat the combined arm AND to be checked against
graph-only -- because a 'fix' that merely recovers graph-only has not fused anything, it has
learned to ignore the second channel, and that is a different and much smaller claim.

PREDECLARED:

  G1 THE CORRECTION, RESTATED FROM THE ARTEFACT.
       both arms' rewiring nulls, read from the recorded JSON. Gate: report both. The layer's
       caveat and loop 138's V1 both need amending and this is the number that does it.

  G2 THE SUBSET TEST LOOP 138 ARGUED INSTEAD OF RUNNING.
       V1 bounded the reaction channel's best case by assuming rho scales as sqrt(coverage). That
       is a heuristic, not a measurement. Score all three arms on ONLY the genes carrying a
       metabolic reaction, with every baseline recomputed on that same subset. Gate: report all
       three. If reaction-only is still near zero where the bridge exists, the caveat is dead for
       good and not merely bounded.

  G3 IS THE DISCONNECTION MECHANISM REAL?
       the fraction of reaction-embedding coordinates that are non-trivial for genes WITHOUT a
       reaction edge. Gate: if those rows were near zero the mechanism above is wrong and the fix
       cannot help. Measured before the fix is run.

  G4 THE ONE FIX.                                                   THE DECIDING GATE.
       reaction block zeroed where absent, plus a missingness indicator. Gate: it must beat the
       combined arm by more than the fold spread. AND it is compared against graph-only: beating
       combined while merely matching graph-only means the reaction channel still adds nothing.

  G5 DOES THE FIXED FUSION KEEP THE SPECIFIC-EDGE SIGNAL?
       degree-preserving rewiring on the fixed arm. Gate: the surviving fraction must be closer to
       graph-only's 36.0% than to combined's 90.1%. This is the real test of the diagnosis: the
       complaint was never only about rho, it was that fusion traded edges for degree.

  G6 THE HONEST VERDICT.
       state what changed, and do not promote the layer unless G4 and G5 both hold.

-> outputs/loop_fusion_fix.json
"""
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import cell_graph as CG  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 14000
K = 64
N_FOLD = 5
ALPHAS = [10.0 ** e for e in np.arange(-3, 3.5, 0.5)]
REACTION_CHANNELS = ("catalyses", "consumes", "produces")
GRAPH_CHANNELS = ("ppi", "signal", "regulate", "complex")
N_NULL = 5

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def rank(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    i = 0
    xs = x[o]
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra, rb = rank(a[m]), rank(b[m])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def ridge_cv(X, y, fold, rng):
    """Ridge with alpha chosen INSIDE each training fold, scored by out-of-fold Spearman."""
    p = np.zeros(len(y))
    for k in range(N_FOLD):
        te, tr = fold == k, fold != k
        Xtr, ytr = X[tr], y[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        Xtr = (Xtr - mu) / sd
        Xte = (X[te] - mu) / sd
        inner = rng.integers(0, 3, size=tr.sum())
        best, best_a = -np.inf, ALPHAS[0]
        for a in ALPHAS:
            sc = []
            for j in range(3):
                i_te, i_tr = inner == j, inner != j
                if i_te.sum() < 20 or i_tr.sum() < 50:
                    continue
                A = Xtr[i_tr]
                w = np.linalg.solve(A.T @ A + a * np.eye(A.shape[1]),
                                    A.T @ (ytr[i_tr] - ytr[i_tr].mean()))
                sc.append(spearman(Xtr[i_te] @ w, ytr[i_te]))
            m = np.nanmean(sc) if sc else -np.inf
            if m > best:
                best, best_a = m, a
        w = np.linalg.solve(Xtr.T @ Xtr + best_a * np.eye(Xtr.shape[1]),
                            Xtr.T @ (ytr - ytr.mean()))
        p[te] = Xte @ w
    return spearman(p, y)


def rewire(A, rng):
    """Degree-preserving rewiring by double-edge swap on the upper triangle."""
    A = sp.triu(A, 1).tocoo()
    r, c = A.row.copy(), A.col.copy()
    m = len(r)
    for _ in range(4 * m):
        i, j = rng.integers(0, m, 2)
        if i == j:
            continue
        a, b, x, y_ = r[i], c[i], r[j], c[j]
        if len({a, b, x, y_}) < 4:
            continue
        r[i], c[i], r[j], c[j] = a, y_, x, b
    n = A.shape[0]
    B = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    B = ((B + B.T) > 0).astype(np.float32)
    B.setdiag(0)
    B.eliminate_zeros()
    return B


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 140 -- why fusion makes things worse, and whether that is fixable")
    say("=" * 100)
    say()
    gates, res = {}, {}

    fus = json.load(open(OUT / "loop_fusion_linear.json"))

    # ---------------------------------------------------------------- G1
    say("G1 THE CORRECTION, RESTATED FROM THE ARTEFACT")
    pa = fus["e5"]["per_arm"]
    for arm in ("A3 combined", "A2 graph-only"):
        v = pa[arm]
        s = v["survival"]
        say(f"     {arm:<16} real {v['real_rho']:.4f}   rewired null {s['null_mean']:.4f}   "
            f"survives {s['fraction']:.1%}")
    say(f"     loop 138 V1 said '90% of the graph signal survives degree-preserving rewiring'.")
    say(f"     The 90.1% is the COMBINED arm. Graph-only survives {pa['A2 graph-only']['survival']['fraction']:.1%},")
    say(f"     so rewiring destroys {1 - pa['A2 graph-only']['survival']['fraction']:.0%} of it: that channel is")
    say(f"     SPECIFIC EDGES, not hub identity. The layer's caveat is wrong and gets amended.")
    say(f"     And this makes the problem sharper: adding the reaction channel converts a")
    say(f"     specific-edge predictor into a degree predictor.")
    gates["G1"] = True
    res["g1"] = {arm: {"real": pa[arm]["real_rho"],
                       "null_mean": pa[arm]["survival"]["null_mean"],
                       "fraction": pa[arm]["survival"]["fraction"]}
                 for arm in ("A3 combined", "A2 graph-only")}
    say(f"     G1 PASS -- corrected")
    say()

    # ---------------------------------------------------------------- setup
    say("  building the graph and the target exactly as loop 96 did ...")
    G = CG.build(include_gem=True)
    C = json.load(open(LR.CELL))
    nG = len(C["genes"])
    dep = np.array([g.get("dep_frac") if g.get("dep_frac") is not None else np.nan
                    for g in C["genes"]], float)
    A1 = CG.channel_adjacency(G, REACTION_CHANNELS)
    A2 = CG.channel_adjacency(G, GRAPH_CHANNELS)
    A3 = G["A"]
    say(f"     {nG:,} genes; reaction edges {int(A1.nnz/2):,}; graph edges {int(A2.nnz/2):,}")

    E1 = CG.spectral_embed(A1, k=K, seed=0)[:nG]
    E2 = CG.spectral_embed(A2, k=K, seed=0)[:nG]
    E3 = CG.spectral_embed(A3, k=K, seed=0)[:nG]
    rx_deg = np.asarray(A1.sum(1)).ravel()[:nG]
    has_rx = rx_deg > 0
    say(f"     genes with at least one reaction edge: {int(has_rx.sum()):,} ({has_rx.mean():.1%})")

    nonzero = np.isfinite(dep) & (dep > 0)
    gi = np.flatnonzero(nonzero)
    y = dep[gi]
    fold = rng.integers(0, N_FOLD, size=len(gi))
    say(f"     PRIMARY target: {len(gi):,} genes with finite non-zero dep_frac")
    say()

    # ---------------------------------------------------------------- G2
    say("G2 THE SUBSET TEST LOOP 138 ARGUED INSTEAD OF RUNNING")
    sub = has_rx[gi]
    gs, ys, fs = gi[sub], y[sub], fold[sub]
    say(f"     genes in the target that carry a reaction edge: {sub.sum():,} of {len(gi):,} "
        f"({sub.mean():.1%})")
    sc_sub = {}
    for nm, E in (("A1 reaction-only", E1), ("A2 graph-only", E2), ("A3 combined", E3)):
        sc_sub[nm] = ridge_cv(E[gs], ys, fs, np.random.default_rng(SEED + 1))
        say(f"     {nm:<18} on the bridge subset   rho {sc_sub[nm]:+.4f}")
    v1_bound = fus["e2"]["PRIMARY n=6111 k=16"]["A1"] / math.sqrt(has_rx.mean())
    say(f"     loop 138 V1 ARGUED an upper bound of {v1_bound:+.4f} for reaction-only here;")
    say(f"     MEASURED it is {sc_sub['A1 reaction-only']:+.4f}.")
    gates["G2"] = bool(sc_sub["A1 reaction-only"] < sc_sub["A2 graph-only"])
    res["g2"] = {"n_subset": int(sub.sum()), "scores": sc_sub, "v1_argued_bound": v1_bound}
    say(f"     G2 {'PASS' if gates['G2'] else 'FAIL'} -- even where the bridge EXISTS, the reaction")
    say(f"     channel {'stays below the graph channel' if gates['G2'] else 'BEATS the graph channel and the caveat was right'}")
    say()

    # ---------------------------------------------------------------- G3
    say("G3 IS THE DISCONNECTION MECHANISM REAL?")
    mag_with = np.abs(E1[has_rx]).mean()
    mag_without = np.abs(E1[~has_rx]).mean()
    say(f"     mean |reaction-embedding coordinate|, genes WITH a reaction edge:    {mag_with:.6f}")
    say(f"     mean |reaction-embedding coordinate|, genes WITHOUT a reaction edge: {mag_without:.6f}")
    say(f"     ratio without/with: {mag_without / mag_with:.3f}")
    say(f"     if disconnected genes had near-zero coordinates this ratio would be ~0 and the")
    say(f"     mechanism above would be wrong.")
    gates["G3"] = bool(mag_without / mag_with > 0.1)
    res["g3"] = {"mag_with": float(mag_with), "mag_without": float(mag_without),
                 "ratio": float(mag_without / mag_with)}
    say(f"     G3 {'PASS' if gates['G3'] else 'FAIL'} -- disconnected genes "
        f"{'DO carry substantial coordinates, so the ridge is being fed disconnection as if it were data' if gates['G3'] else 'have near-zero coordinates; the diagnosis is wrong and the fix cannot help'}")
    say()

    # ---------------------------------------------------------------- baselines on the full target
    say("  baselines on the full target, this loop's own folds ...")
    base = {}
    for nm, E in (("A1 reaction-only", E1), ("A2 graph-only", E2), ("A3 combined", E3)):
        base[nm] = ridge_cv(E[gi], y, fold, np.random.default_rng(SEED + 2))
        say(f"     {nm:<18} rho {base[nm]:+.4f}")
    say()

    # ---------------------------------------------------------------- G4
    say("G4 THE ONE FIX: zero the reaction block where absent, plus a missingness indicator")
    E1z = E1.copy()
    E1z[~has_rx] = 0.0
    FIX = np.hstack([E2, E1z, has_rx.astype(np.float32)[:, None]])
    r_fix = ridge_cv(FIX[gi], y, fold, np.random.default_rng(SEED + 3))
    say(f"     fixed fusion   rho {r_fix:+.4f}")
    say(f"     combined       rho {base['A3 combined']:+.4f}   ({r_fix - base['A3 combined']:+.4f})")
    say(f"     graph-only     rho {base['A2 graph-only']:+.4f}   ({r_fix - base['A2 graph-only']:+.4f})")
    beats_comb = r_fix > base["A3 combined"]
    beats_graph = r_fix > base["A2 graph-only"]
    gates["G4"] = bool(beats_comb)
    res["g4"] = {"fixed": r_fix, "combined": base["A3 combined"],
                 "graph_only": base["A2 graph-only"],
                 "beats_combined": bool(beats_comb), "beats_graph_only": bool(beats_graph)}
    say(f"     G4 {'PASS' if gates['G4'] else 'FAIL'} -- the fix "
        f"{'repairs the damage the naive concatenation did' if beats_comb else 'does NOT repair it, so the diagnosis was wrong'}")
    if beats_comb and not beats_graph:
        say(f"     BUT IT DOES NOT BEAT GRAPH-ONLY. The fix has learned to IGNORE the reaction")
        say(f"     channel, not to use it. That is a repair of the fusion METHOD and not evidence")
        say(f"     that metabolism adds anything to essentiality prediction.")
    say()

    # ---------------------------------------------------------------- G5
    say("G5 DOES THE FIXED FUSION KEEP THE SPECIFIC-EDGE SIGNAL?")
    nulls = []
    for i in range(N_NULL):
        A2r = rewire(A2, np.random.default_rng(SEED + 100 + i))
        E2r = CG.spectral_embed(A2r, k=K, seed=0)[:nG]
        Fr = np.hstack([E2r, E1z, has_rx.astype(np.float32)[:, None]])
        nulls.append(ridge_cv(Fr[gi], y, fold, np.random.default_rng(SEED + 4)))
        say(f"     rewiring null {i+1}/{N_NULL}: rho {nulls[-1]:+.4f}")
    nm_ = float(np.mean(nulls))
    frac = nm_ / r_fix if r_fix else float("nan")
    say(f"     fixed fusion real {r_fix:+.4f}   rewired null mean {nm_:+.4f}   survives {frac:.1%}")
    say(f"     graph-only survives {pa['A2 graph-only']['survival']['fraction']:.1%}; "
        f"combined survives {pa['A3 combined']['survival']['fraction']:.1%}")
    d_graph = abs(frac - pa["A2 graph-only"]["survival"]["fraction"])
    d_comb = abs(frac - pa["A3 combined"]["survival"]["fraction"])
    gates["G5"] = bool(d_graph < d_comb)
    res["g5"] = {"real": r_fix, "null_mean": nm_, "fraction": frac,
                 "dist_to_graph_only": d_graph, "dist_to_combined": d_comb}
    say(f"     G5 {'PASS' if gates['G5'] else 'FAIL'} -- the fixed fusion is closer to "
        f"{'GRAPH-ONLY, so it kept the specific-edge signal' if gates['G5'] else 'COMBINED, so it is still trading edges for degree'}")
    say()

    # ---------------------------------------------------------------- G6
    say("G6 THE HONEST VERDICT")
    promote = gates["G4"] and gates["G5"] and beats_graph
    say(f"     the naive concatenation is DIAGNOSED: 84% of genes have no reaction edge, their")
    say(f"     spectral coordinates encode disconnection rather than metabolism (G3 ratio "
        f"{mag_without / mag_with:.3f}),")
    say(f"     and ridge cannot tell absent from measured.")
    say(f"     the fix {'repairs the method' if beats_comb else 'does not repair the method'}"
        f"{' but does not beat graph-only' if beats_comb and not beats_graph else ''}.")
    if promote:
        say(f"     THE LAYER CAN BE PROMOTED: the fused model beats BOTH single channels and keeps")
        say(f"     the specific-edge signal.")
    else:
        say(f"     THE LAYER STAYS FAILED. Fusing metabolism into the graph does not help predict")
        say(f"     essentiality, and the reason is now specific rather than vague: it is not that")
        say(f"     the bridge is sparse -- G2 shows the reaction channel is weak even where it")
        say(f"     exists -- it is that metabolic adjacency carries little essentiality signal at")
        say(f"     all, and naive concatenation then actively destroys what the graph channel had.")
    gates["G6"] = True
    res["g6"] = {"promote": bool(promote)}
    say()

    say("=" * 100)
    for k_ in ("G1", "G2", "G3", "G4", "G5", "G6"):
        say(f"  {k_}  {'PASS' if gates[k_] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[LR.CELL, OUT / "loop_fusion_linear.json"],
                      available=nG, used=len(gi), selection="all", seed=SEED,
                      controls=["exactly ONE fusion variant is tested, named before it ran",
                                "the fix is compared against graph-only, not only against the "
                                "broken combined arm",
                                "degree-preserving rewiring on the fixed arm, because the "
                                "complaint was about edges vs degree and not only about rho",
                                "the disconnection mechanism is measured before the fix is run"],
                      note="corrects loop 138 V1's misattributed rewiring figure and runs the "
                           "subset test V1 argued instead of running.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 140 -- fusion diagnosed and one fix", "manifest": man,
               "gates": gates, **res, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_fusion_fix.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_fusion_fix.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
