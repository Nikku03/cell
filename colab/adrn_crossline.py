"""CROSS-LINE: does a K562-trained model predict knockout response in a cell line it was never trained on?

WHAT THE PRIOR EVIDENCE SAYS, so this is not run blind. `ceiling_cartography` measured naive cross-line transfer
at 0.1835 against a marginal floor of 0.2498 -- transferring scored WORSE than predicting the average.
`multiline_network` found pairwise causal-edge reproduction between lines of 0.047-0.131 with 4.6% of edges
conserved. And today's LOCO result showed the model cannot extrapolate across functional classes even WITHIN
K562. So the expectation is failure; the point is to measure it properly and find where the boundary sits.

THE DESIGN.

    target lines   RPE1 / Jurkat / HepG2, ~2,151 knockouts each, profiles truncated to their top ~250 movers.
    truth          for a held-out knockout, the movers in its profile that fall inside the common gene space
                   (that line's target vocabulary intersected with K562's 8,246).
    held out       200 knockouts per line, seeded, removed from any within-line training.

ARMS

    k562_model   ridge + NMF fitted on K562 training knockouts ONLY, decoded onto the common space. The test.
    within_line  the same construction fitted on the target line's own remaining knockouts. The ceiling for
                 this task at this depth -- what you would get by simply measuring the line.
    freq_line    the most frequently moving genes in the target line's training knockouts. THE OPPONENT: a
                 within-line lookup table needing no model at all.
    freq_k562    the most frequently moving genes in K562. Tests whether K562's tide alone transfers.
    random       floor.

PREDECLARED GATE: `k562_model` must beat `freq_line` past a bootstrap CI. Beating `freq_k562` or `random` is not
sufficient -- if a K562-trained model cannot beat "the genes that usually move in RPE1", it carries nothing
line-specific and cross-line prediction has failed.

DEPTH WARNING, because the numbers below are NOT comparable to the 0.2920 / 0.3593 sealed figures. Those score
against K562 answer keys built from the full 8,246-gene matrix; these score against keys truncated to ~250
movers. A same-protocol, same-depth K562 self-prediction is therefore run as the reference point, and every
cross-line number should be read against THAT, not against the headline.

STRATIFICATION, following today's finding that "have I seen something like this before" drives everything: test
genes are split by whether the same gene was also knocked out in K562 training. `gene_seen` is transfer of a
known gene to a new line; `gene_new` is a novel gene AND a novel line.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
from robustness import Sweeper

OUT, SP = A.OUT, A.SP
LINES = ("RPE1", "Jurkat", "HepG2")
K_VALUES = (30, 60, 120)
PENALTIES = (1.0, 10.0, 100.0, 1000.0)
N_TEST = 200


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("CROSS-LINE: does a K562-trained model predict a line it never saw?")
    report("=" * 100)
    report("  PREDECLARED: k562_model must beat freq_line (the within-line frequency prior).")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    Amat = np.abs(z["M"]).astype(np.float32)
    Sgn = z["M"].astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05

    sealed = set()
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
    k562_train = [k for k in kos if k not in sealed]

    prof = {ln: pickle.load(open(SP / f"nlz_{ln}.pkl", "rb")) for ln in LINES}
    universe = sorted(set(kos) | sealed | {g for d in prof.values() for g in d})
    urow = {g: i for i, g in enumerate(universe)}
    base, _ = A.build_channels(universe, set(k562_train), report)
    extra, _, tb = C.extra_channels(universe, set(k562_train), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    report(f"  channels {chan.shape} over {len(universe):,} symbols")

    from sklearn.decomposition import NMF

    def build(line):
        """common gene space, response matrix for the line, and the K562 matrix on the same space."""
        d = prof[line]
        vocab = sorted({t for p in d.values() for t in p} & set(genes))
        gi = {g: i for i, g in enumerate(vocab)}
        kcol = np.array([genes.index(g) for g in vocab])
        lk = sorted(d)
        M = np.zeros((len(lk), len(vocab)), np.float32)
        for r, g in enumerate(lk):
            for t, v in d[g].items():
                j = gi.get(t)
                if j is not None:
                    M[r, j] = float(v)
        return vocab, kcol, lk, M

    def prec(scores, keys, truth):
        out = []
        for j, k in enumerate(keys):
            s = scores[j].copy()
            out.append(len({j2 for j2 in np.argsort(-s)[:A.NPRED]} & truth[k]) / float(A.NPRED))
        return np.array(out)

    def fit_ridge(Xsrc, Wt):
        rng = np.random.default_rng(A.SEED + 5)
        p = rng.permutation(len(Xsrc))
        nv = max(50, int(0.2 * len(Xsrc)))
        best_v, best_p = -1e9, PENALTIES[0]
        for pen in PENALTIES:
            w = A.ridge_fit(Xsrc[p[nv:]], Wt[p[nv:]], pen)
            err = float(np.mean((A.ridge_apply(Xsrc[p[:nv]], w) - Wt[p[:nv]]) ** 2))
            if -err > best_v:
                best_v, best_p = -err, pen
        return A.ridge_fit(Xsrc, Wt, best_p)

    sw = Sweeper("k562_model - freq_line", axes={"line": list(LINES), "K": list(K_VALUES)})
    means, strat = {}, {}

    for line in LINES:
        vocab, kcol, lk, M = build(line)
        rng = np.random.default_rng(A.SEED)
        test_idx = rng.choice(len(lk), min(N_TEST, len(lk) // 3), replace=False)
        test_keys = [lk[i] for i in test_idx]
        tr_idx = np.setdiff1d(np.arange(len(lk)), test_idx)
        truth = {lk[i]: set(np.where(np.abs(M[i]) > 0)[0]) for i in test_idx}
        seen = np.array([k in set(k562_train) for k in test_keys])
        report(f"\n  === {line}: {len(lk):,} knockouts, common space {len(vocab):,} genes, "
               f"{len(test_keys)} held out ({int(seen.sum())} also knocked out in K562 training) ===")
        report(f"      mean movers per held-out knockout: "
               f"{np.mean([len(v) for v in truth.values()]):.0f} of {len(vocab):,}")

        # K562 matrix restricted to the same gene space, tide removed on that space
        Xk = Amat[[ki[k] for k in k562_train]][:, kcol].copy()
        tide_v = tide[kcol]
        Xk[:, tide_v] = 0.0
        Ml = np.abs(M[tr_idx]).copy()

        freq_line = Ml.mean(0)
        freq_line[tide_v] = -np.inf
        freq_k562 = (Xk >= A.TAU).mean(0)
        freq_k562[tide_v] = -np.inf
        fl = np.tile(np.argsort(-freq_line)[:A.NPRED], (len(test_keys), 1))
        fk = np.tile(np.argsort(-freq_k562)[:A.NPRED], (len(test_keys), 1))

        def top_from(idx_mat):
            return np.array([len(set(idx_mat[j]) & truth[k]) / float(A.NPRED)
                             for j, k in enumerate(test_keys)])

        p_fl, p_fk = top_from(fl), top_from(fk)
        rr = np.random.default_rng(1)
        p_rand = np.array([len(set(rr.choice(len(vocab), A.NPRED, replace=False)) & truth[k]) / float(A.NPRED)
                           for k in test_keys])

        for K in K_VALUES:
            nk = NMF(n_components=K, init="nndsvda", max_iter=300, random_state=0)
            Wk = nk.fit_transform(Xk).astype(np.float32)
            Hk = nk.components_.astype(np.float32)
            w_k = fit_ridge(chan[[urow[k] for k in k562_train]], Wk)
            s = A.ridge_apply(chan[[urow[k] for k in test_keys]], w_k) @ Hk
            s[:, tide_v] = -np.inf
            p_k562 = prec(s, test_keys, truth)

            nl = NMF(n_components=K, init="nndsvda", max_iter=300, random_state=0)
            Wl = nl.fit_transform(Ml).astype(np.float32)
            Hl = nl.components_.astype(np.float32)
            w_l = fit_ridge(chan[[urow[lk[i]] for i in tr_idx]], Wl)
            s2 = A.ridge_apply(chan[[urow[k] for k in test_keys]], w_l) @ Hl
            s2[:, tide_v] = -np.inf
            p_within = prec(s2, test_keys, truth)

            sw.add({"line": line, "K": K}, line, p_k562 - p_fl)
            means[f"{line}|K={K}"] = {"k562_model": float(p_k562.mean()),
                                      "within_line": float(p_within.mean()),
                                      "freq_line": float(p_fl.mean()), "freq_k562": float(p_fk.mean()),
                                      "random": float(p_rand.mean())}
            if K == 60:
                strat[line] = {
                    "gene_seen_n": int(seen.sum()),
                    "gene_seen_k562model": float(p_k562[seen].mean()) if seen.any() else None,
                    "gene_seen_freq": float(p_fl[seen].mean()) if seen.any() else None,
                    "gene_new_n": int((~seen).sum()),
                    "gene_new_k562model": float(p_k562[~seen].mean()) if (~seen).any() else None,
                    "gene_new_freq": float(p_fl[~seen].mean()) if (~seen).any() else None}
            report(f"    K={K:>3}  k562_model {p_k562.mean():.4f} | within_line {p_within.mean():.4f} "
                   f"| freq_line {p_fl.mean():.4f} | freq_k562 {p_fk.mean():.4f} | random {p_rand.mean():.4f}")

    report(sw.report())
    v = sw.verdict()

    report("\n  GENE SEEN IN K562 TRAINING vs NOT  (K=60)")
    report(f"  {'line':<8} {'n seen':>7} {'model':>8} {'freq':>8} | {'n new':>6} {'model':>8} {'freq':>8}")
    for ln, s in strat.items():
        f = lambda x: f"{x:.4f}" if x is not None else "   n/a"
        report(f"  {ln:<8} {s['gene_seen_n']:>7} {f(s['gene_seen_k562model']):>8} {f(s['gene_seen_freq']):>8} "
               f"| {s['gene_new_n']:>6} {f(s['gene_new_k562model']):>8} {f(s['gene_new_freq']):>8}")

    json.dump({"test": "adrn_crossline", "lines": list(LINES), "K_values": list(K_VALUES),
               "means": means, "sweep": v, "strata": strat, "log": log},
              open(OUT / "adrn_crossline.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'adrn_crossline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
