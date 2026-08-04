"""MARKOV PROPAGATION INSIDE THE ADRN: every layer we have, with a MEASURED transition operator.

WHY THIS IS NOT THE NETWORK TEST THAT ALREADY DIED EIGHT TIMES.  Every previous propagation attempt here used a
CURATED graph as the whole model -- PPI, TRRUST, reaction network -- and every one was killed by degree-matched
rewiring across 8 tests and a 36-cell sweep. The lesson of today is sharper than "networks fail": measurements
work, descriptions do not (measured-vs-curated, HARMFUL 9/9; six curated representations failed at cold start).

So the operator is MEASURED. T[i,j] = row-normalised |M[i,j]| over TRAINING perturbations -- the empirically
observed probability that perturbing i moves j. The curated layers are demoted from being the model to being
the SEED: they answer "where does a walk starting at an unmeasured gene begin", which is the one thing the
measured matrix cannot say about a gene whose row is hidden.

    curated layers -> WHERE the walk starts      ppi, trrust/reg_edges, complexes, pathways, compartments
    measured matrix -> WHAT happens after        T, from the perturbation data itself

SEEDS, one per layer plus the one that uses no curation at all:

    column   the sealed gene's own COLUMN -- which training perturbations move IT. Available for a gene nobody
             perturbed, and it needs no graph. As ridge FEATURES this was NULL today; as a walk SEED it is a
             different object and untested.
    ppi / tf / complex / pathway / spatial   the gene's neighbours in each curated layer
    all      all seeds combined

PROPAGATION.  Restricted to the genes on BOTH axes of the readout, the operator is square, so the walk is a
genuine Markov chain with restart: p = (1-a) * sum_k a^k * s Q^k, computed by iteration. The final distribution
is pushed through T once more to score every readout gene.

ARMS, all on the sealed cohorts with the ADRN's own answer key and precision@N:

    frequency          the floor
    adrn               the incumbent: channels -> ridge -> NMF loadings -> response
    markov_<seed>      propagation alone, no ridge, no channels
    adrn_markov        the ADRN with the walk distribution added as an SVD-reduced feature block
    REWIRED            every curated seed graph degree-matched double-edge swapped. THE control that killed the
                       previous eight tests -- if markov survives everywhere except here, topology is real; if
                       rewired scores the same, we are measuring degree again.
    PERM_T             the measured operator with its rows permuted across genes. Tests whether the OPERATOR is
                       doing the work or just its marginals.

PREDECLARED, before any number:
    markov > frequency AND markov > REWIRED, resolved   -> measured propagation predicts where curated
                                                           propagation never could.
    markov ~ REWIRED                                    -> degree again. Ninth failure, and the seeds are
                                                           decoration.
    adrn_markov > adrn, resolved                        -> propagation carries something the channels do not.
    adrn_markov ~ adrn                                  -> whatever the walk knows, the annotation already knew,
                                                           as was true for DepMap and for the response column.

Swept over restart probability x prediction depth on robustness.Sweeper.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
from robustness import Sweeper

OUT, SP = A.OUT, A.SP
ALPHAS = (0.3, 0.6, 0.9)
NPREDS = (10, 20, 50)
STEPS, MK_SVD, PENALTIES = 8, 96, (1.0, 10.0, 100.0, 1000.0)


def rewire(pairs, n, rng, passes=8):
    """Degree-matched double-edge swap: same degree sequence, different topology."""
    e = [list(p) for p in pairs]
    if len(e) < 4:
        return [tuple(p) for p in e]
    for _ in range(passes * len(e)):
        i, j = rng.integers(0, len(e), 2)
        if i == j:
            continue
        a, b = e[i]
        c, d = e[j]
        if len({a, b, c, d}) < 4:
            continue
        e[i], e[j] = [a, d], [c, b]
    return [tuple(p) for p in e]


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("MARKOV PROPAGATION INSIDE THE ADRN -- curated layers seed the walk, measured data drives it")
    report("=" * 100)
    report("  PREDECLARED: markov > REWIRED => topology is real. markov ~ REWIRED => degree again, 9th failure.")
    report("               adrn_markov > adrn => the walk carries what channels do not.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    gj = {g: j for j, g in enumerate(genes)}
    Aabs = np.abs(z["M"]).astype(np.float32)
    tide = (Aabs >= A.TAU).mean(0) >= 0.05

    sealed, answers = set(), {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists():
            answers[tag or "cohort1"] = json.loads(af.read_text())
    train = [k for k in kos if k not in sealed]
    train_rows = np.array([ki[k] for k in train])
    universe = sorted(set(kos) | sealed)
    urow = {g: i for i, g in enumerate(universe)}
    tr_u = np.array([urow[k] for k in train])
    tpos = {k: i for i, k in enumerate(train)}
    report(f"\n  {len(kos):,} perturbations; sealed {len(sealed)}; train {len(train):,}")

    # ---- the MEASURED operator, built from training rows only ----
    R = Aabs[train_rows].copy()
    R[:, tide] = 0.0
    T = R / (R.sum(1, keepdims=True) + 1e-9)                       # train x readout, rows are distributions
    both = [k for k in train if k in gj]                            # genes on BOTH axes -> square walk
    bi = np.array([tpos[k] for k in both])
    bj = np.array([gj[k] for k in both])
    Q = T[np.ix_(bi, bj)].astype(np.float32)
    Q = Q / (Q.sum(1, keepdims=True) + 1e-9)
    report(f"  measured operator T {T.shape} | square walk Q {Q.shape} (genes on both axes)")

    # ---- curated layers, demoted to SEEDS ----
    import propagate as pr
    G = pr._load()
    ppi, trr, cplx, g2c, g2pw, pw2g = G["ppi"], G["trrust"], G["cplx"], G["g2c"], G["g2pw"], G["pw2g"]
    names = G["names"]

    # KEYING, and three layers were silently empty until I checked it.
    #   ppi, g2c, g2pw  are keyed by INTEGER gene index; passing a gene NAME returns nothing.
    #   trrust, pw2g    are keyed by string name.
    #   cplx            is empty for all 2,039 complexes, so the complex layer is built by INVERTING g2c.
    #   pw2g values     are sets, not lists.
    # The first run printed "ppi 0 edges / complex 0 edges / pathway 0 edges" and would have concluded that
    # curated seeds are decoration, when three of five had never been populated. A dead arm is not a null.
    gidx = G["idx"]
    c2g = {}
    for i, cs in g2c.items():
        for c in (cs or []):
            c2g.setdefault(c, []).append(i)
    MAX_NBR = 300

    def _nm(xs):
        return [names[x] if isinstance(x, (int, np.integer)) else str(x) for x in xs]

    def nbrs_ppi(g):
        i = gidx.get(g)
        return _nm(ppi.get(i, [])) if i is not None else []

    def nbrs_tf(g):
        return [t[0] if isinstance(t, (list, tuple)) else str(t) for t in trr.get(g, [])]

    def nbrs_complex(g):
        i = gidx.get(g)
        out = []
        for c in (g2c.get(i) or []) if i is not None else []:
            out.extend(c2g.get(c, []))
        return _nm(out[:MAX_NBR])

    def nbrs_pathway(g):
        i = gidx.get(g)
        out = []
        for pw in (g2pw.get(i) or []) if i is not None else []:
            mem = pw2g.get(pw)
            if mem and len(mem) <= MAX_NBR:      # skip giant catch-all pathways: they seed everything equally
                out.extend(list(mem))
        return _nm(out[:MAX_NBR])

    chan_base, chan_names = A.build_channels(universe, set(train), report)
    extra, extra_names, tb = C.extra_channels(universe, set(train), report)
    chan = np.concatenate([chan_base, extra[:, :tb]], 1).astype(np.float32)
    comp_cols = [i for i, n in enumerate(chan_names) if n.startswith("compartment:")]

    def seed_layer(g, layer):
        if layer == "ppi":
            return nbrs_ppi(g)
        if layer == "tf":
            return nbrs_tf(g)
        if layer == "complex":
            return nbrs_complex(g)
        if layer == "pathway":
            return nbrs_pathway(g)
        if layer == "spatial":
            if not comp_cols or g not in urow:
                return []
            v = chan_base[urow[g], comp_cols]
            if v.sum() == 0:
                return []
            sim = chan_base[:, comp_cols] @ v
            top = np.argsort(-sim)[:200]
            return [universe[t] for t in top]
        return []

    LAYERS = ("ppi", "tf", "complex", "pathway", "spatial")
    rng = np.random.default_rng(A.SEED + 77)
    # REWIRED versions: same degree sequence per layer, different topology
    # THE REAL LAYER AND ITS REWIRING MUST HAVE THE SAME EDGES.  The first version capped only the rewired copy
    # at 400k while the real arm kept full membership, so the control would have had FEWER edges than the layer
    # it controls -- biasing the one gate that decides this test. Both arms are now built from the SAME
    # (possibly subsampled) edge list, so they share an identical degree sequence by construction.
    rew, real = {}, {}
    for L in LAYERS:
        pairs = []
        for g in universe:
            for h in seed_layer(g, L):
                if h in urow and h != g:
                    pairs.append((g, h))
        n_full = len(pairs)
        if len(pairs) > 400_000:
            pairs = [pairs[i] for i in rng.choice(len(pairs), 400_000, replace=False)]
        dr = {}
        for a, b in pairs:
            dr.setdefault(a, []).append(b)
        real[L] = dr
        sw_pairs = rewire(pairs, len(universe), rng)
        d = {}
        for a, b in sw_pairs:
            d.setdefault(a, []).append(b)
        rew[L] = d
        if n_full != len(pairs):
            report(f"    (layer {L} subsampled {n_full:,} -> {len(pairs):,} edges; BOTH arms use this set)")
        assert len(pairs) > 0, (f"layer {L} produced ZERO edges -- it is a dead arm, not a null result. "
                                f"Check the key type (index vs name) before reading any verdict.")
        report(f"  layer {L:<9} {len(pairs):>7,} edges -> degree-matched rewired copy built")

    def seed_vec(g, layer, rewired):
        s = np.zeros(len(both), np.float32)
        if layer == "column":
            if g in gj:
                col = np.abs(Aabs[train_rows, gj[g]]).astype(np.float32)
                col[np.array([tpos[k] for k in train]) == tpos.get(g, -1)] = 0.0
                s = col[bi]
        else:
            hs = (rew[layer] if rewired else real[layer]).get(g, [])
            bpos = {k: i for i, k in enumerate(both)}
            for h in hs:
                i = bpos.get(h)
                if i is not None:
                    s[i] += 1.0
        t = s.sum()
        return s / t if t > 0 else s

    def walk(S, alpha):
        p = S.copy()
        acc = np.zeros_like(S)
        for _ in range(STEPS):
            p = p @ Q
            acc += p
            p = (1 - alpha) * S + alpha * p
        out = acc / (acc.sum(1, keepdims=True) + 1e-9)
        # push the settled distribution over walk-nodes through T once more, so every readout gene is scored
        # -- not just the 4,300 that happen to sit on both axes.
        full = out @ T[bi]
        return full / (full.sum(1, keepdims=True) + 1e-9)

    from sklearn.decomposition import NMF
    X = Aabs[train_rows].copy()
    X[:, tide] = 0.0
    nmf = NMF(n_components=A.K, init="nndsvda", max_iter=300, random_state=0)
    W = nmf.fit_transform(X).astype(np.float32)
    H = nmf.components_.astype(np.float32)
    del X

    def fit_ridge(F):
        rv = np.random.default_rng(A.SEED + 8).permutation(len(tr_u))
        nv = max(50, len(rv) // 5)
        bv, bp = -1e18, PENALTIES[0]
        for pen in PENALTIES:
            w = A.ridge_fit(F[tr_u[rv[nv:]]], W[rv[nv:]], pen)
            e = -float(np.mean((A.ridge_apply(F[tr_u[rv[:nv]]], w) - W[rv[:nv]]) ** 2))
            if e > bv:
                bv, bp = e, pen
        return A.ridge_fit(F[tr_u], W, bp)

    chan_n = chan / (np.linalg.norm(chan, axis=1, keepdims=True) + 1e-9)
    w_adrn = fit_ridge(chan_n)
    freq = (Aabs[train_rows] >= A.TAU).mean(0)
    freq[tide] = -np.inf

    def hit(v, movers, n, k):
        v = np.asarray(v, np.float64).copy()
        v[tide] = -np.inf
        if k in gj:
            v[gj[k]] = -np.inf
        return len({genes[t] for t in np.argsort(-v)[:n]} & set(movers)) / float(n)

    SEEDS = ("column",) + LAYERS + ("all",)
    keys_all = {tag: [k for k in sorted(ans) if k in urow] for tag, ans in answers.items()}
    sw_rw = Sweeper("markov - REWIRED", axes={"alpha": list(ALPHAS), "npred": list(NPREDS)})
    sw_ad = Sweeper("adrn_markov - adrn", axes={"alpha": list(ALPHAS), "npred": list(NPREDS)})
    cells = {}
    for alpha in ALPHAS:
        # walk distributions for sealed genes, real and rewired
        Pm, Pr = {}, {}
        for tag, keys in keys_all.items():
            for rewired, store in ((False, Pm), (True, Pr)):
                per = {}
                for sd in SEEDS:
                    if sd == "all":
                        Sm = np.stack([sum(seed_vec(k, L, rewired) for L in ("column",) + LAYERS)
                                       for k in keys])
                        Sm = Sm / (Sm.sum(1, keepdims=True) + 1e-9)
                    elif sd == "column" and rewired:
                        continue                       # the column has no topology to rewire
                    else:
                        Sm = np.stack([seed_vec(k, sd, rewired) for k in keys])
                    per[sd] = walk(Sm, alpha)
                store[tag] = per
        # the ADRN + walk feature block, built on TRAINING genes for fitting
        Str = np.stack([sum(seed_vec(k, L, False) for L in ("column",) + LAYERS) for k in train])
        Str = Str / (Str.sum(1, keepdims=True) + 1e-9)
        Wtr = walk(Str, alpha)
        mu = Wtr.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Wtr - mu, full_matrices=False)
        Vk = Vt[:MK_SVD].T
        MK = np.zeros((len(universe), MK_SVD), np.float32)
        MK[tr_u] = (Wtr - mu) @ Vk
        for tag, keys in keys_all.items():
            MK[[urow[k] for k in keys]] = (Pm[tag]["all"] - mu) @ Vk
        both_F = np.concatenate([chan, MK], 1)
        both_F = both_F / (np.linalg.norm(both_F, axis=1, keepdims=True) + 1e-9)
        w_both = fit_ridge(both_F)
        report(f"\n  alpha={alpha}")
        for npred in NPREDS:
            arms = {}
            for tag, ans in answers.items():
                keys = keys_all[tag]
                pa = A.ridge_apply(chan_n[[urow[k] for k in keys]], w_adrn) @ H
                pb = A.ridge_apply(both_F[[urow[k] for k in keys]], w_both) @ H
                for i, k in enumerate(keys):
                    mv = ans[k]["movers"]
                    arms.setdefault("frequency", []).append(hit(freq, mv, npred, k))
                    arms.setdefault("adrn", []).append(hit(pa[i], mv, npred, k))
                    arms.setdefault("adrn_markov", []).append(hit(pb[i], mv, npred, k))
                    for sd in SEEDS:
                        arms.setdefault(f"mk_{sd}", []).append(hit(Pm[tag][sd][i], mv, npred, k))
                        if sd in Pr[tag]:
                            arms.setdefault(f"rw_{sd}", []).append(hit(Pr[tag][sd][i], mv, npred, k))
            arms = {n: np.array(v) for n, v in arms.items()}
            bm = max(SEEDS, key=lambda s: arms[f"mk_{s}"].mean())
            br = max([s for s in SEEDS if f"rw_{s}" in arms], key=lambda s: arms[f"rw_{s}"].mean())
            cfg, tg = {"alpha": alpha, "npred": npred}, f"a{alpha}|n{npred}"
            sw_rw.add(cfg, tg, arms[f"mk_{bm}"] - arms[f"rw_{br}"])
            sw_ad.add(cfg, tg, arms["adrn_markov"] - arms["adrn"])
            cells[tg] = {"best_seed": bm, "best_rewired": br, **{n: float(v.mean()) for n, v in arms.items()}}
            report(f"    N={npred:>2}: freq {arms['frequency'].mean():.4f} | adrn {arms['adrn'].mean():.4f} | "
                   f"adrn+mk {arms['adrn_markov'].mean():.4f} | best markov ({bm}) "
                   f"{arms[f'mk_{bm}'].mean():.4f} | best REWIRED ({br}) {arms[f'rw_{br}'].mean():.4f}")
            report("      seeds: " + " ".join(f"{s} {arms[f'mk_{s}'].mean():.4f}" for s in SEEDS))

    for nm, s in (("rewired", sw_rw), ("adrn", sw_ad)):
        report(s.report())
        if s.inert_axes() or s.duplicate_cells():
            report(f"  GUARD {nm}: inert axes {s.inert_axes()}, duplicate cells {s.duplicate_cells()}")

    g_rw = float(np.mean([c[f"mk_{c['best_seed']}"] - c[f"rw_{c['best_rewired']}"] for c in cells.values()]))
    g_ad = float(np.mean([c["adrn_markov"] - c["adrn"] for c in cells.values()]))
    npos_rw = sum(1 for c in sw_rw._cells.values() if c["lo"] > 0)
    npos_ad = sum(1 for c in sw_ad._cells.values() if c["lo"] > 0)
    report(f"\n  markov - REWIRED {g_rw:+.4f} (resolved {npos_rw}/{len(sw_rw._cells)}) | "
           f"adrn_markov - adrn {g_ad:+.4f} (resolved {npos_ad}/{len(sw_ad._cells)})")
    report("\n  READING")
    if npos_rw > len(sw_rw._cells) / 2:
        report("  Propagation beats its degree-matched rewiring. Topology carries real information HERE, where")
        report("  eight previous network tests found only degree -- the difference being that the operator is")
        report("  measured rather than curated.")
    else:
        report("  Propagation does NOT beat degree-matched rewiring. Ninth failure of the same kind: the walk is")
        report("  reading node degree, and the curated seeds are decoration even with a measured operator.")
    if npos_ad > len(sw_ad._cells) / 2:
        report("  And it adds to the ADRN, so the walk knows something the channels do not.")
    else:
        report("  It adds nothing to the ADRN. Whatever the walk knows, the annotation already knew -- the same")
        report("  outcome as the DepMap block and the response column.")

    json.dump({"test": "adrn_markov", "alphas": list(ALPHAS), "npreds": list(NPREDS), "steps": STEPS,
               "cells": cells, "gap_markov_rewired": g_rw, "gap_adrn": g_ad,
               "sweeps": {"rewired": sw_rw.verdict(), "adrn": sw_ad.verdict()}, "log": log},
              open(OUT / "adrn_markov.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'adrn_markov.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
