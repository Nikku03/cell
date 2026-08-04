"""CAN WE LEARN THE SIMILARITY THE ANNOTATION CANNOT EXPRESS?

WHERE THIS COMES FROM.  adrn_is_it_lookup measured two things on the sealed cohorts:

    ridge - nearest-neighbour     +0.0077, FRAGILE, sign inconsistent   -> the ADRN IS a lookup
    ORACLE - nearest-neighbour    +0.2128, ROBUST 6/6                   -> and a much better answer is SITTING IN
                                                                           THE TRAINING SET, unreachable

The oracle picks the training gene whose TRUE response is closest. It cannot be run in practice, but it proves
the answer exists among genes already measured. Annotation-cosine retrieval finds it +0.2128 worse. That is not
a data problem -- the data has the answer -- and not a capacity problem -- a 42k ridge beats a 630k transformer.
It is a problem with the SIMILARITY FUNCTION.

THE QUESTION.  Annotation cosine is a hand-chosen metric: it treats every channel as equally important and
assumes "shares many labels" means "responds alike". Nobody ever checked that. So learn the metric instead:
train a function on TRAINING PAIRS ONLY to predict how similar two genes' RESPONSES are from their annotations,
then retrieve with it.

    metric_linear   a learned low-rank projection P, so similarity is cos(P.chan_i, P.chan_j). This is exactly
                    annotation cosine with the channels re-weighted and rotated -- the minimal change that could
                    close the gap.
    metric_pair     a ridge on the ELEMENTWISE PRODUCT of two genes' channel vectors. For binary channels that
                    product is the set of labels they SHARE, so this learns which shared labels predict a shared
                    response and which are noise.

ARMS AND CONTROLS
    frequency          the floor.
    nn_annot           annotation cosine, k in {1,5,10}. The incumbent retrieval and the thing to beat.
    nn_scorable        annotation cosine restricted to training genes that HAVE a scorable phenotype. This
                       separates "a better metric" from "a better pool" -- retrieving a silent gene returns a
                       useless answer, and fixing that needs no learning at all.
    metric_linear      learned projection, same k.
    metric_pair        learned pair ridge, same k.
    nn_ORACLE          true-response similarity. The denominator: what any retrieval could reach.

LEAK CONTROL.  The target -- response similarity -- is computed between TRAINING genes only. No sealed gene's
response enters the fit, the pair sampling, or the validation split. At test time the learned metric sees a
sealed gene's ANNOTATION and every training gene's ANNOTATION, never a sealed response. The moment a sealed
response touched the metric it would become the oracle and mean nothing.

PREDECLARED, before any number:
    learned > nn_annot, resolved      -> the descriptors DID contain findable signal and annotation cosine was
                                         throwing it away. The fix is math, and we have it.
    learned ~ nn_annot                -> annotation cannot express response similarity at all. The +0.2128 gap
                                         is unreachable from these features, and closing it needs NEW
                                         MEASUREMENTS rather than a better model. That is the honest end of the
                                         road for data already on disk.

Reported as the FRACTION OF THE ORACLE GAP CLOSED, because the absolute gain matters less than whether the
approach can reach the thing we proved is there. Swept over prediction depth x embedding rank.
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
NPREDS = (10, 20, 50)
RANKS = (32, 64)
KNN = (1, 5, 10)
N_PAIRS, EPOCHS, MIN_MOVERS = 400_000, 120, 5


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("CAN WE LEARN THE SIMILARITY ANNOTATION CANNOT EXPRESS?")
    report("=" * 100)
    report("  PREDECLARED: learned > nn_annot => the descriptors had findable signal, the fix is math.")
    report("               learned ~ nn_annot => annotation cannot express it; only new MEASUREMENTS close it.")

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

    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tb = C.extra_channels(universe, set(train), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1).astype(np.float32)

    R = Aabs.copy()
    R[:, tide] = 0.0
    Rtr = R[train_rows]
    nmov = (Aabs[train_rows] >= A.TAU).sum(1)
    scorable = np.where(nmov >= MIN_MOVERS)[0]
    report(f"\n  training pool {len(train):,}; scorable {len(scorable):,}; sealed {len(sealed)}")

    Rn = Rtr / (np.linalg.norm(Rtr, axis=1, keepdims=True) + 1e-9)
    Cn = chan[tr_u] / (np.linalg.norm(chan[tr_u], axis=1, keepdims=True) + 1e-9)

    # PAIRS drawn from scorable training genes only: a pair of silent genes has a response similarity that is
    # pure noise and would teach the metric to match noise.
    rng = np.random.default_rng(A.SEED)
    pi = rng.choice(scorable, N_PAIRS)
    pj = rng.choice(scorable, N_PAIRS)
    keep = pi != pj
    pi, pj = pi[keep], pj[keep]
    y = (Rn[pi] * Rn[pj]).sum(1).astype(np.float32)
    report(f"  {len(pi):,} training pairs; target response-cosine mean {y.mean():.4f} sd {y.std():.4f}")

    # validation split by GENE, not by pair -- splitting pairs would put a gene in both halves
    gperm = rng.permutation(scorable)
    vg = set(gperm[: max(50, len(gperm) // 5)].tolist())
    vmask = np.array([(a in vg) or (b in vg) for a, b in zip(pi, pj)])
    report(f"  validation: {len(vg)} held-out genes -> {int(vmask.sum()):,} pairs excluded from fitting")

    import torch
    import torch.nn as nn
    torch.set_num_threads(4)
    Ct = torch.tensor(Cn)
    yt = torch.tensor(y)
    pit, pjt = torch.tensor(pi), torch.tensor(pj)
    trm = torch.tensor(~vmask)
    vam = torch.tensor(vmask)

    def fit_linear(rank, seed):
        torch.manual_seed(seed)
        P = nn.Linear(Cn.shape[1], rank, bias=False)
        opt = torch.optim.AdamW(P.parameters(), lr=3e-3, weight_decay=1e-4)
        idx_tr = torch.where(trm)[0]
        idx_va = torch.where(vam)[0]
        best, bad, state = 1e18, 0, None
        for ep in range(EPOCHS):
            perm = idx_tr[torch.randperm(len(idx_tr))]
            for s in range(0, len(perm), 8192):
                b = perm[s:s + 8192]
                opt.zero_grad()
                e = P(Ct)
                e = e / (e.norm(dim=1, keepdim=True) + 1e-9)
                pred = (e[pit[b]] * e[pjt[b]]).sum(1)
                loss = ((pred - yt[b]) ** 2).mean()
                loss.backward()
                opt.step()
            with torch.no_grad():
                e = P(Ct)
                e = e / (e.norm(dim=1, keepdim=True) + 1e-9)
                v = float((((e[pit[idx_va]] * e[pjt[idx_va]]).sum(1) - yt[idx_va]) ** 2).mean())
            if v < best - 1e-7:
                best, bad, state = v, 0, {k: t.clone() for k, t in P.state_dict().items()}
            else:
                bad += 1
                if bad >= 15:
                    break
        P.load_state_dict(state)
        with torch.no_grad():
            W = P.weight.detach().numpy().T.astype(np.float32)
        return W, best

    def pair_ridge():
        """ridge on the elementwise product = which SHARED labels predict a shared response."""
        Xp = (Cn[pi] * Cn[pj])[~vmask]
        yp = y[~vmask]
        g = Xp.T @ Xp + 10.0 * np.eye(Xp.shape[1], dtype=np.float32)
        return np.linalg.solve(g, Xp.T @ yp).astype(np.float32)

    wpair = pair_ridge()
    report(f"  pair ridge fitted on {int((~vmask).sum()):,} pairs")

    freq = (Aabs[train_rows] >= A.TAU).mean(0)
    freq[tide] = -np.inf

    def hit(v, movers, n, k):
        v = v.copy()
        v[tide] = -np.inf
        if k in gj:
            v[gj[k]] = -np.inf
        return len({genes[t] for t in np.argsort(-v)[:n]} & set(movers)) / float(n)

    sw = Sweeper("best learned metric - best annotation nn", axes={"npred": list(NPREDS), "rank": list(RANKS)})
    cells = {}
    for rank in RANKS:
        Wp, vloss = fit_linear(rank, 0)
        report(f"\n  rank {rank}: linear metric fitted, val MSE {vloss:.5f}")
        E = chan @ Wp
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        for npred in NPREDS:
            arms = {}
            for tag, ans in answers.items():
                keys = [k for k in sorted(ans) if k in urow]
                Q = chan[[urow[k] for k in keys]]
                Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
                S_ann = Qn @ Cn.T
                S_lin = E[[urow[k] for k in keys]] @ E[tr_u].T
                S_pair = (Qn[:, None, :] * Cn[None, :, :]).reshape(-1, Cn.shape[1]) @ wpair \
                    if len(keys) * len(Cn) < 4_000_000 else None
                if S_pair is not None:
                    S_pair = S_pair.reshape(len(keys), len(Cn))
                Qr = np.array([R[ki[k]] for k in keys], np.float32)
                Qr = Qr / (np.linalg.norm(Qr, axis=1, keepdims=True) + 1e-9)
                S_or = Qr @ Rn.T
                sc_mask = np.full(len(Cn), -np.inf, np.float32)
                sc_mask[scorable] = 0.0
                for i, k in enumerate(keys):
                    mv = ans[k]["movers"]
                    arms.setdefault("frequency", []).append(hit(freq.copy(), mv, npred, k))
                    for kk in KNN:
                        for nm, S in (("nn_annot", S_ann), ("metric_linear", S_lin),
                                      ("nn_ORACLE", S_or)):
                            idx = np.argsort(-S[i])[:kk]
                            arms.setdefault(f"{nm}{kk}", []).append(hit(Rtr[idx].mean(0), mv, npred, k))
                        idx = np.argsort(-(S_ann[i] + sc_mask))[:kk]
                        arms.setdefault(f"nn_scorable{kk}", []).append(hit(Rtr[idx].mean(0), mv, npred, k))
                        if S_pair is not None:
                            idx = np.argsort(-S_pair[i])[:kk]
                            arms.setdefault(f"metric_pair{kk}", []).append(hit(Rtr[idx].mean(0), mv, npred, k))
            arms = {n: np.array(v) for n, v in arms.items()}

            def best(prefix):
                cand = [f"{prefix}{kk}" for kk in KNN if f"{prefix}{kk}" in arms]
                return max(cand, key=lambda n: arms[n].mean()) if cand else None
            b_ann, b_sc, b_lin = best("nn_annot"), best("nn_scorable"), best("metric_linear")
            b_pair, b_or = best("metric_pair"), best("nn_ORACLE")
            learned = max([n for n in (b_lin, b_pair) if n], key=lambda n: arms[n].mean())
            cfg, tg = {"npred": npred, "rank": rank}, f"n{npred}|r{rank}"
            sw.add(cfg, tg, arms[learned] - arms[b_ann])
            cells[tg] = {"best_learned": learned, **{n: float(v.mean()) for n, v in arms.items()}}
            gap = arms[b_or].mean() - arms[b_ann].mean()
            closed = (arms[learned].mean() - arms[b_ann].mean()) / gap if gap > 1e-9 else 0.0
            cells[tg]["oracle_gap"] = float(gap)
            cells[tg]["fraction_closed"] = float(closed)
            report(f"    N={npred:>2}: freq {arms['frequency'].mean():.4f} | annot {arms[b_ann].mean():.4f} | "
                   f"scorable {arms[b_sc].mean():.4f} | linear {arms[b_lin].mean():.4f} | "
                   f"pair {arms[b_pair].mean():.4f} | ORACLE {arms[b_or].mean():.4f} | "
                   f"closed {closed:+.1%}")

    report(sw.report())
    v = sw.verdict()
    if sw.inert_axes() or sw.duplicate_cells():
        report(f"  GUARD: inert axes {sw.inert_axes()}, duplicate cells {sw.duplicate_cells()}")

    mg = float(np.mean([c["fraction_closed"] for c in cells.values()]))
    md = float(np.mean([c[c["best_learned"]] - c[[n for n in c if n.startswith("nn_annot")][0]]
                        for c in cells.values()]))
    npos = sum(1 for c in sw._cells.values() if c["lo"] > 0)
    report(f"\n  MEAN fraction of the oracle gap closed by a learned metric: {mg:+.1%}")
    report("\n  READING")
    if npos > len(sw._cells) / 2:
        report(f"  A learned metric beats annotation cosine, resolved in {npos} of {len(sw._cells)} cells, and")
        report(f"  closes {mg:.1%} of the gap to the retrieval oracle. The descriptors DID contain findable")
        report("  signal that annotation cosine was discarding. The fix is math and we now have some of it.")
    else:
        report("  A learned metric does NOT beat annotation cosine. Re-weighting and rotating the channels, and")
        report("  learning which SHARED labels matter, both fail to find the neighbour the oracle finds.")
        report("  So the +0.2128 is not recoverable from these descriptors by any similarity function of them:")
        report("  the information is ABSENT from the annotation, not merely badly weighted. Closing it needs")
        report("  new MEASUREMENTS of the genes, not new mathematics over the ones we have. That is the honest")
        report("  end of the road for data already on disk.")

    json.dump({"test": "adrn_learned_metric", "npreds": list(NPREDS), "ranks": list(RANKS), "knn": list(KNN),
               "n_pairs": int(len(pi)), "cells": cells, "mean_fraction_closed": mg,
               "sweep": v, "log": log}, open(OUT / "adrn_learned_metric.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'adrn_learned_metric.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
