"""LOOP 141 -- IS THE +0.0100 REAL? AND WHAT IS THE FIX ACTUALLY DOING?

Loop 140 printed "THE LAYER CAN BE PROMOTED" on the strength of a late-fused model scoring 0.5309
against graph-only's 0.5208. That margin is +0.0100 AND IT HAS NO CONFIDENCE INTERVAL. Loop 120
passed two gates on margins smaller than their own spread and this repository has been correcting
that mistake ever since; promoting a layer on a bare +0.0100 would repeat it exactly.

AND LOOP 140's STATED MECHANISM WAS FALSE, which changes what is even being claimed. G3 predicted
that genes with no reaction edge would carry spectral coordinates encoding disconnection. They
carry EXACTLY ZERO -- ratio 0.000. So zeroing them was a no-op and the "fix" was never about
missingness. The only remaining difference between the two arms is:

    A3 combined   embed(A_graph + A_reaction)          ONE embedding of the MERGED adjacency
    fixed         [embed(A_graph), embed(A_reaction)]  TWO embeddings, CONCATENATED

which is early fusion against late fusion. The 0.3997 -> 0.5309 gain is that, and nothing else.

THE CORRECTED MECHANISM, and G5 already supports it. Merging adjacencies puts metabolite-mediated
edges into the same Laplacian as protein interactions. One common metabolite links hundreds of
genes, so the merged graph acquires dense hub structure that dominates the leading eigenvectors.
The embedding becomes a hub detector: combined survives degree-preserving rewiring at 90.1% while
late fusion survives 34.2%, against graph-only's 36.0%. Early fusion did not dilute the signal --
it REPLACED a specific-edge signal with a degree signal.

SO THERE ARE TWO SEPARATE CLAIMS AND THEY DESERVE DIFFERENT TREATMENT:

    CLAIM 1  late fusion beats early fusion, +0.1312       large, and the rewiring nulls
                                                            independently corroborate it
    CLAIM 2  late fusion beats graph-only, +0.0100         tiny, uncorroborated, and it is the
                                                            ONLY claim that would mean metabolism
                                                            adds anything to essentiality

Claim 1 is a finding about method. Claim 2 is a finding about biology, and it is the one the layer's
verdict depends on. This loop measures claim 2 properly and accepts whatever comes back.

PREDECLARED:

  M1 THE PAIRED INTERVAL ON CLAIM 2.                                 THE DECIDING GATE.
       both arms scored on identical folds, retaining per-gene out-of-fold predictions, then a
       bootstrap over GENES on the paired difference in Spearman rho. Gate: the interval must
       exclude zero. If it does not, the layer is NOT promoted and metabolism has not been shown
       to add anything.

  M2 THE SAME QUESTION WITH THE REACTION BLOCK REPLACED BY NOISE.    THE CONTROL.
       concatenate a RANDOM matrix of the same shape and sparsity pattern instead of the reaction
       embedding. Gate: if random columns buy a similar margin, the +0.0100 is capacity and not
       metabolism. This is loop 136's lesson -- there, active-site pooling beat mean pooling until
       random positions did too.

  M3 CLAIM 1, WITH ITS OWN INTERVAL.
       late versus early fusion, same treatment. Gate: report the interval. This is expected to be
       large and the point is to show the two claims are not the same size.

  M4 IS THE MARGIN CARRIED BY THE BRIDGE GENES?
       claim 2's difference computed separately on the 1,008 genes with a reaction edge and the
       5,103 without. Gate: if metabolism is adding anything it must be adding it where metabolism
       is measured. A margin that lives in the genes with NO reaction edge is an artefact.

  M5 THE VERDICT, WITH THE LAYER'S STATUS AS THE OUTPUT.
       promote only if M1 excludes zero AND M2 shows random columns do not, AND M4 puts the effect
       on the bridge genes. Any other combination leaves the layer FAILED with a sharper record.

-> outputs/loop_fusion_margin.json
"""
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import cell_graph as CG  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 14100
K = 64
N_FOLD = 5
N_BOOT = 2000
ALPHAS = [10.0 ** e for e in np.arange(-3, 3.5, 0.5)]
REACTION_CHANNELS = ("catalyses", "consumes", "produces")
GRAPH_CHANNELS = ("ppi", "signal", "regulate", "complex")

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


def ridge_oof(X, y, fold, rng):
    """Identical to loop 140's ridge_cv but RETURNS the out-of-fold predictions, because a paired
    interval cannot be formed from a scalar."""
    p = np.zeros(len(y))
    for k in range(N_FOLD):
        te, tr = fold == k, fold != k
        Xtr, ytr = X[tr], y[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        Xtr = (Xtr - mu) / sd
        Xte = (X[te] - mu) / sd
        inner = rng.integers(0, 3, size=int(tr.sum()))
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
    return p


def boot_diff(y, pa, pb, rng, idx=None, n=N_BOOT):
    """Bootstrap over GENES on the paired difference rho(pb) - rho(pa)."""
    if idx is None:
        idx = np.arange(len(y))
    y, pa, pb = y[idx], pa[idx], pb[idx]
    d = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        d.append(spearman(pb[s], y[s]) - spearman(pa[s], y[s]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 141 -- is the +0.0100 real, and what is the fix actually doing?")
    say("=" * 100)
    say()
    gates, res = {}, {}

    say("  rebuilding loop 140's arms on identical folds ...")
    G = CG.build(include_gem=True)
    C = json.load(open(LR.CELL))
    nG = len(C["genes"])
    dep = np.array([g.get("dep_frac") if g.get("dep_frac") is not None else np.nan
                    for g in C["genes"]], float)
    A1 = CG.channel_adjacency(G, REACTION_CHANNELS)
    A2 = CG.channel_adjacency(G, GRAPH_CHANNELS)
    A3 = G["A"]
    E1 = CG.spectral_embed(A1, k=K, seed=0)[:nG]
    E2 = CG.spectral_embed(A2, k=K, seed=0)[:nG]
    E3 = CG.spectral_embed(A3, k=K, seed=0)[:nG]
    has_rx = np.asarray(A1.sum(1)).ravel()[:nG] > 0
    nonzero = np.isfinite(dep) & (dep > 0)
    gi = np.flatnonzero(nonzero)
    y = dep[gi]
    fold = np.random.default_rng(SEED).integers(0, N_FOLD, size=len(gi))
    say(f"     {len(gi):,} target genes; {int(has_rx[gi].sum()):,} carry a reaction edge")
    say()

    IND = has_rx.astype(np.float32)[:, None]
    X_graph = E2[gi]
    X_late = np.hstack([E2, E1, IND])[gi]
    X_early = E3[gi]

    say("  scoring three arms, identical folds, retaining out-of-fold predictions ...")
    p_graph = ridge_oof(X_graph, y, fold, np.random.default_rng(SEED + 1))
    p_late = ridge_oof(X_late, y, fold, np.random.default_rng(SEED + 1))
    p_early = ridge_oof(X_early, y, fold, np.random.default_rng(SEED + 1))
    r_graph, r_late, r_early = (spearman(p_graph, y), spearman(p_late, y), spearman(p_early, y))
    say(f"     graph-only  {r_graph:+.4f}")
    say(f"     late fusion {r_late:+.4f}")
    say(f"     early fusion (merged adjacency) {r_early:+.4f}")
    say()

    # ---------------------------------------------------------------- M1
    say("M1 THE PAIRED INTERVAL ON CLAIM 2 (late fusion vs graph-only)")
    m, lo, hi = boot_diff(y, p_graph, p_late, np.random.default_rng(SEED + 10))
    say(f"     difference {r_late - r_graph:+.4f}")
    say(f"     bootstrap over genes: {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    gates["M1"] = bool(lo > 0)
    res["m1"] = {"graph": r_graph, "late": r_late, "diff": r_late - r_graph,
                 "boot": [m, lo, hi]}
    say(f"     M1 {'PASS' if gates['M1'] else 'FAIL'} -- the interval "
        f"{'EXCLUDES zero' if gates['M1'] else 'INCLUDES zero, so the margin is not distinguishable from noise'}")
    say()

    # ---------------------------------------------------------------- M2
    say("M2 THE CONTROL: the reaction block replaced by noise of the same shape")
    rg = np.random.default_rng(SEED + 20)
    E1n = np.zeros_like(E1)
    src = E1[has_rx]
    E1n[has_rx] = rg.normal(loc=0.0, scale=src.std(), size=src.shape).astype(np.float32)
    X_noise = np.hstack([E2, E1n, IND])[gi]
    p_noise = ridge_oof(X_noise, y, fold, np.random.default_rng(SEED + 1))
    r_noise = spearman(p_noise, y)
    mn, lon, hin = boot_diff(y, p_graph, p_noise, np.random.default_rng(SEED + 21))
    say(f"     random block, same shape and scale, same nonzero rows: rho {r_noise:+.4f}")
    say(f"     versus graph-only: {mn:+.4f}  [{lon:+.4f}, {hin:+.4f}]")
    say(f"     the REAL reaction block bought {m:+.4f}; noise buys {mn:+.4f}")
    gates["M2"] = bool((r_late - r_graph) > (r_noise - r_graph))
    res["m2"] = {"noise": r_noise, "boot_vs_graph": [mn, lon, hin]}
    say(f"     M2 {'PASS' if gates['M2'] else 'FAIL'} -- the real block "
        f"{'beats its own noise control' if gates['M2'] else 'DOES NOT beat noise; the margin is capacity, exactly as random residue positions were in loop 136'}")
    say()

    # ---------------------------------------------------------------- M3
    say("M3 CLAIM 1, WITH ITS OWN INTERVAL (late vs early fusion)")
    m3, lo3, hi3 = boot_diff(y, p_early, p_late, np.random.default_rng(SEED + 30))
    say(f"     difference {r_late - r_early:+.4f}")
    say(f"     bootstrap: {m3:+.4f}  [{lo3:+.4f}, {hi3:+.4f}]")
    say(f"     claim 1 is {abs(m3 / m) if m else float('nan'):.0f}x the size of claim 2.")
    gates["M3"] = bool(lo3 > 0)
    res["m3"] = {"early": r_early, "late": r_late, "diff": r_late - r_early,
                 "boot": [m3, lo3, hi3]}
    say(f"     M3 {'PASS' if gates['M3'] else 'FAIL'} -- late fusion beats early fusion")
    say()

    # ---------------------------------------------------------------- M4
    say("M4 IS THE MARGIN CARRIED BY THE BRIDGE GENES?")
    sub = has_rx[gi]
    mb, lob, hib = boot_diff(y, p_graph, p_late, np.random.default_rng(SEED + 40),
                             idx=np.flatnonzero(sub))
    mo, loo, hio = boot_diff(y, p_graph, p_late, np.random.default_rng(SEED + 41),
                             idx=np.flatnonzero(~sub))
    say(f"     on the {int(sub.sum()):,} genes WITH a reaction edge:    {mb:+.4f} "
        f"[{lob:+.4f}, {hib:+.4f}]")
    say(f"     on the {int((~sub).sum()):,} genes WITHOUT one:           {mo:+.4f} "
        f"[{loo:+.4f}, {hio:+.4f}]")
    say(f"     if metabolism is adding anything it must add it where metabolism is measured.")
    gates["M4"] = bool(mb > mo)
    res["m4"] = {"bridge": [mb, lob, hib], "nonbridge": [mo, loo, hio],
                 "n_bridge": int(sub.sum()), "n_nonbridge": int((~sub).sum())}
    say(f"     M4 {'PASS' if gates['M4'] else 'FAIL'} -- the effect is "
        f"{'larger on the bridge genes, as it must be' if gates['M4'] else 'NOT larger on the bridge genes, which makes it an artefact'}")
    say()

    # ---------------------------------------------------------------- M5
    say("M5 THE VERDICT")
    promote = gates["M1"] and gates["M2"] and gates["M4"]
    say(f"     M1 interval excludes zero: {gates['M1']}")
    say(f"     M2 beats its noise control: {gates['M2']}")
    say(f"     M4 effect sits on the bridge genes: {gates['M4']}")
    if promote:
        say(f"     PROMOTE. Metabolic adjacency adds {m:+.4f} [{lo:+.4f}, {hi:+.4f}] to essentiality")
        say(f"     prediction over the protein graph alone, it beats its own noise control, and the")
        say(f"     effect sits where metabolism is measured.")
    else:
        say(f"     DO NOT PROMOTE. The layer stays FAILED. What IS established is claim 1 -- late")
        say(f"     fusion beats early fusion by {m3:+.4f} [{lo3:+.4f}, {hi3:+.4f}] -- which is a")
        say(f"     finding about METHOD, not about biology. Merging adjacencies lets")
        say(f"     metabolite-mediated hubs dominate the spectrum and turns a specific-edge")
        say(f"     predictor into a degree detector; embedding channels separately does not.")
        say(f"     That is worth recording and it is not the same as 'the fusion works'.")
    gates["M5"] = True
    res["m5"] = {"promote": bool(promote)}
    say()

    say("=" * 100)
    for k_ in ("M1", "M2", "M3", "M4", "M5"):
        say(f"  {k_}  {'PASS' if gates[k_] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[LR.CELL, OUT / "loop_fusion_fix.json"],
                      available=nG, used=len(gi), selection="all", seed=SEED,
                      controls=["a bootstrap interval on the margin the promotion rests on",
                                "the reaction block replaced by noise of the same shape and scale",
                                "the margin split by whether the gene has a reaction edge at all",
                                "the two claims separated, because one is method and one is biology"],
                      note="loop 140 printed a promotion on a bare +0.0100 with no interval. Loop "
                           "120 passed two gates on margins inside their own spread and this is "
                           "the check that prevents a third.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 141 -- the fusion margin", "manifest": man, "gates": gates, **res,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_fusion_margin.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_fusion_margin.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
