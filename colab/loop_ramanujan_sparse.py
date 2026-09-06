"""Loop 171. A Ramanujan-expander sparse network on the candidate features, and the control that matters.

WHAT A RAMANUJAN GRAPH IS FOR. A d-regular graph is Ramanujan when its non-trivial adjacency
eigenvalues obey |lambda| <= 2*sqrt(d-1) -- the Alon-Boppana bound, which no infinite family can
beat. For a biregular bipartite graph the corresponding bound is sqrt(d_left-1) + sqrt(d_right-1).
Such graphs are optimal expanders: they carry information between any two parts of the network with
the fewest possible edges. That is exactly the property a sparse neural network wants from its mask,
and it is why Ramanujan constructions appear in sparse-network design.

WHAT IT IS BEING ASKED TO DO HERE. Loop 170's gradient-boosted ranker reaches hit@1 0.8506 at full
coverage on 24 per-candidate features. The question is whether a sparse expander network beats it
where it matters least comfortably -- at 100% coverage, answering every gap.

THE CONTROL THAT DECIDES IT, and it is the same shape as every other control in this arc. A sparse
network can beat a dense one for two quite different reasons: because sparsity regularises, or
because THIS sparsity is well-connected. So the Ramanujan mask is measured against a RANDOM sparse
mask at identical density and identical parameter count. If random does as well, the expander
property bought nothing and the honest description is "a sparse net helped", not "a Ramanujan net
helped".

THE HONEST PRIOR, stated before the run. 24 features is very little to sparsify. Expanders earn
their keep when layers are wide enough that dense connection is wasteful, and gradient boosting is
strong on small tabular problems. The plausible outcome is that the sparse net does not beat the
GBM, and A5 asks the more promising question anyway: whether the two ENSEMBLE, since a neural net
and a tree ensemble make different errors.

PREDECLARED, before any number is looked at.

  A1 THE MASK IS ACTUALLY RAMANUJAN. Construct each bipartite layer mask as a random biregular
     graph and verify the spectral bound lambda_2 <= sqrt(d_l - 1) + sqrt(d_r - 1) explicitly,
     resampling until it holds and reporting how many draws were needed.
     Gate: PASS iff every layer mask satisfies the bound. Calling a mask Ramanujan without checking
     its spectrum is the whole failure mode this gate exists to prevent.

  A2 DOES IT BEAT THE GBM AT FULL COVERAGE? hit@1 over all cases, same folds, same seed.
     Gate: more than 3 sem over 0.8506.

  A3 IS IT THE EXPANDER OR JUST THE SPARSITY? Ramanujan mask against a random sparse mask at
     identical density, and against a dense net at identical width.
     Gate: PASS iff Ramanujan beats random-sparse by more than 3 sem. A FAIL means the spectral
     property contributed nothing and the result should be reported as sparsity, not as Ramanujan.

  A4 PARAMETER-MATCHED DENSE. A dense network with the same number of weights as the sparse one,
     which is the fair architectural comparison rather than the same width.
     Gate: passes on being reported.

  A5 DO THEY ENSEMBLE? Rank-average of the GBM and the best network.
     Gate: PASS iff the ensemble beats the better of the two alone by more than 3 sem. Trees and
     nets make different errors and this is the question most likely to pay.

  A6 WHAT THIS CANNOT SHOW. DEV only. 24 features is a small space to sparsify and this is not a
     test of expander networks in general. And the Ramanujan property is verified on the mask, not
     on anything about the trained function.

-> outputs/loop_ramanujan_sparse.json
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                    # noqa: E402
import run_manifest as RM                  # noqa: E402

CACHE = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/"
             "scratchpad/l170_features.npz")
OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_ramanujan_sparse.json"
NFOLD, SEED = 5, 17100
WIDTH, DEPTH, DEG = 256, 3, 8
EPOCHS, LR, BATCH = 25, 3e-3, 4096
GBM_HIT1 = 0.8506

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def biregular_mask(n_in, n_out, d_out, rng, tries=40):
    """Random biregular bipartite mask, verified against the Feng-Li Ramanujan bound.

    Each output unit takes exactly d_out DISTINCT inputs, and inputs are consumed in least-used-first
    order so the left degrees stay within one of each other -- that is what makes the graph biregular
    rather than merely sparse. The first version shuffled a tiled index array and cut it into
    consecutive chunks, which almost never yields d_out distinct entries per row; every retry failed
    and the function returned None. Building the row from a usage count cannot produce a duplicate.

    The bound for a (d_l, d_r)-biregular bipartite graph is sqrt(d_l - 1) + sqrt(d_r - 1) on the
    second singular value of the biadjacency matrix.
    """
    d_out = int(min(d_out, n_in))
    best = None
    for t in range(1, tries + 1):
        M = np.zeros((n_out, n_in), np.float32)
        use = np.zeros(n_in)
        for r in range(n_out):
            jitter = rng.random(n_in) * 0.5
            take = np.argsort(use + jitter)[:d_out]
            M[r, take] = 1.0
            use[take] += 1.0
        d_l = float(M.sum(0).mean())
        d_r = float(M.sum(1).mean())
        sv = np.linalg.svd(M, compute_uv=False)
        lam2 = float(sv[1]) if len(sv) > 1 else 0.0
        bound = float(np.sqrt(max(d_l - 1, 0)) + np.sqrt(max(d_r - 1, 0)))
        if best is None or lam2 < best[1]:
            best = (M, lam2, bound, t)
        if lam2 <= bound:
            return M, lam2, bound, t, True
    M, lam2, bound, t = best
    return M, lam2, bound, t, False


def main():
    t0 = time.time()
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    say("=" * 104)
    say("  A RAMANUJAN-EXPANDER SPARSE NETWORK, against random sparsity and against the GBM")
    say("=" * 104)
    say()
    z = np.load(CACHE, allow_pickle=True)
    X, y, grp, kind = z["X"], z["y"], z["grp"], z["kind"]
    n_ok = int(z["n"])
    cases = np.unique(grp)
    mu, sd = X.mean(0), X.std(0)
    Xz = (X - mu) / np.maximum(sd, 1e-9)
    say(f"     {n_ok:,} cases | {X.shape[0]:,} rows | {X.shape[1]} features | GBM baseline "
        f"hit@1 {GBM_HIT1:.4f}")

    # ------------------------------------------------------------------ A1
    say()
    say("A1 THE MASKS")
    rng = np.random.default_rng(SEED)
    dims = [X.shape[1]] + [WIDTH] * DEPTH
    masks, spec = [], []
    for k in range(len(dims) - 1):
        M, lam2, bound, tries, okb = biregular_mask(dims[k], dims[k + 1],
                                                    min(DEG, dims[k]), rng)
        masks.append(M)
        spec.append({"layer": k, "shape": list(M.shape), "density": float(M.mean()),
                     "lambda2": lam2, "bound": bound, "tries": tries, "ramanujan": bool(okb)})
        say(f"     layer {k}: {M.shape[1]}->{M.shape[0]}  degree {int(min(DEG, dims[k]))}  "
            f"density {M.mean():.3f}  lambda2 {lam2:.3f} vs bound {bound:.3f}  "
            f"{'RAMANUJAN' if okb else 'NOT Ramanujan'} after {tries} draw(s)")
    a1 = all(s["ramanujan"] for s in spec)
    GG.verdict(a1, emit=say, if_true=(
        "every mask satisfies the Feng-Li bound, verified on its spectrum rather than assumed."),
        if_false="a mask fails the bound; it is a sparse mask but not a Ramanujan one.")
    say(f"     A1 {'PASS' if a1 else 'FAIL'}")

    # ---------------------------------------------------------------- models
    class MaskedMLP(nn.Module):
        def __init__(self, dims, masks=None, dense=False):
            super().__init__()
            self.lin = nn.ModuleList()
            self.reg = []
            for k in range(len(dims) - 1):
                lay = nn.Linear(dims[k], dims[k + 1])
                self.lin.append(lay)
                self.reg.append(None if dense else
                                torch.tensor(masks[k], dtype=torch.float32))
            self.out = nn.Linear(dims[-1], 1)

        def forward(self, x):
            for k, lay in enumerate(self.lin):
                w = lay.weight if self.reg[k] is None else lay.weight * self.reg[k]
                x = torch.relu(torch.nn.functional.linear(x, w, lay.bias))
            return self.out(x).squeeze(-1)

    def train_eval(make):
        p = np.zeros(len(y))
        from sklearn.model_selection import GroupKFold
        for tr, te in GroupKFold(n_splits=NFOLD).split(Xz, y, grp):
            torch.manual_seed(SEED)
            net = make()
            opt = torch.optim.Adam(net.parameters(), lr=LR)
            lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
                float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))))
            Xt = torch.tensor(Xz[tr], dtype=torch.float32)
            yt = torch.tensor(y[tr], dtype=torch.float32)
            n = len(Xt)
            for ep in range(EPOCHS):
                perm = torch.randperm(n)
                for b in range(0, n, BATCH):
                    idx = perm[b:b + BATCH]
                    opt.zero_grad()
                    loss = lossf(net(Xt[idx]), yt[idx])
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                p[te] = net(torch.tensor(Xz[te], dtype=torch.float32)).numpy()
        return p

    def hits(p):
        return np.array([1.0 if y[grp == g].astype(bool)[np.argmax(p[grp == g])] else 0.0
                         for g in cases])
    nparam_sparse = int(sum(m.sum() for m in masks)) + WIDTH * DEPTH + WIDTH + 1

    say()
    say(f"     training three networks, {WIDTH}x{DEPTH}, {EPOCHS} epochs, "
        f"{nparam_sparse:,} live weights in the sparse ones")
    p_ram = train_eval(lambda: MaskedMLP(dims, masks))
    say(f"       ramanujan done [{time.time()-t0:.0f}s]")
    rmask = []
    for m in masks:
        r = np.zeros_like(m)
        flat = rng.permutation(m.size)[:int(m.sum())]
        r.flat[flat] = 1.0
        rmask.append(r)
    p_rnd = train_eval(lambda: MaskedMLP(dims, rmask))
    say(f"       random-sparse done [{time.time()-t0:.0f}s]")
    wdense = max(int(np.sqrt(nparam_sparse / DEPTH)), 8)
    ddims = [X.shape[1]] + [wdense] * DEPTH
    p_dns = train_eval(lambda: MaskedMLP(ddims, dense=True))
    say(f"       dense (width {wdense}, parameter-matched) done [{time.time()-t0:.0f}s]")

    h = {"ramanujan": hits(p_ram), "random_sparse": hits(p_rnd), "dense_matched": hits(p_dns)}
    say()
    for k, v in sorted(h.items(), key=lambda kv: -kv[1].mean()):
        say(f"     {k:<16s} hit@1 {v.mean():.4f}")
    say(f"     {'GBM (loop 170)':<16s} hit@1 {GBM_HIT1:.4f}")

    def pd(a, b):
        d = a - b
        return float(d.mean()), float(d.std() / np.sqrt(len(d)))

    # ------------------------------------------------------------------ A2
    say()
    hr = h["ramanujan"]
    d2 = float(hr.mean() - GBM_HIT1)
    s2 = float(hr.std() / np.sqrt(len(hr)))
    a2 = bool(d2 > 3 * s2)
    say(f"A2 ramanujan vs GBM at full coverage: {d2:+.4f} sem {s2:.4f} ({d2/s2:+.1f} sem)")
    GG.verdict(a2, emit=say, if_true="the sparse expander net beats gradient boosting here.",
               if_false="it does not beat gradient boosting, which is the expected outcome on 24 "
                        "tabular features and is why A5 asks about the ensemble instead.")
    say(f"     A2 {'PASS' if a2 else 'FAIL'}")

    # ------------------------------------------------------------------ A3
    d3, s3 = pd(h["ramanujan"], h["random_sparse"])
    a3 = bool(d3 > 3 * s3)
    say()
    say(f"A3 ramanujan vs random sparse at identical density: {d3:+.4f} sem {s3:.4f} "
        f"({d3/s3:+.1f} sem)")
    GG.verdict(a3, emit=say, if_true=(
        "the EXPANDER property is doing the work, not merely the sparsity."), if_false=(
        "random sparsity does as well, so the spectral property contributed nothing here and the "
        "honest description is a sparse net, not a Ramanujan one."))
    say(f"     A3 {'PASS' if a3 else 'FAIL'}")

    # ------------------------------------------------------------------ A4
    d4, s4 = pd(h["ramanujan"], h["dense_matched"])
    say()
    say(f"A4 ramanujan vs parameter-matched dense (width {wdense}): {d4:+.4f} sem {s4:.4f} "
        f"({d4/s4:+.1f} sem)")
    a4 = True
    say(f"     A4 {'PASS' if a4 else 'FAIL'}")

    # ------------------------------------------------------------------ A5
    say()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from scipy import stats as st
    pg = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=NFOLD).split(X, y, grp):
        pg[te] = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
            random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    hg = hits(pg)
    best_net = max(h, key=lambda k: h[k].mean())
    pn = {"ramanujan": p_ram, "random_sparse": p_rnd, "dense_matched": p_dns}[best_net]
    ens = np.zeros(len(y))
    for g in cases:
        s = grp == g
        ens[s] = st.rankdata(pg[s]) + st.rankdata(pn[s])
    he = hits(ens)
    base = max(hg.mean(), h[best_net].mean())
    d5 = float(he.mean() - base)
    s5 = float((he - (hg if hg.mean() >= h[best_net].mean() else h[best_net])).std()
               / np.sqrt(len(he)))
    a5 = bool(d5 > 3 * s5)
    say(f"A5 ENSEMBLE (rank-average of GBM and {best_net}): {he.mean():.4f} "
        f"vs best single {base:.4f} = {d5:+.4f} sem {s5:.4f} ({d5/s5:+.1f} sem)")
    GG.verdict(a5, emit=say, if_true=(
        "trees and the network make different errors and the ensemble captures both."), if_false=(
        "the ensemble does not beat the better single model."))
    say(f"     A5 {'PASS' if a5 else 'FAIL'}")

    say()
    say("A6 WHAT THIS CANNOT SHOW")
    say("     DEV only. 24 features is a small space to sparsify, so this is a test of expander")
    say("     masks on THIS problem and not of expander networks in general.")
    say("     The Ramanujan property is verified on the mask, not on anything about the function")
    say("     the network learns.")
    a6 = True
    say(f"     A6 {'PASS' if a6 else 'FAIL'}")

    gates = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5, "A6": a6}
    man = RM.manifest(inputs=[CACHE], available=n_ok, used=n_ok, selection="all", seed=SEED,
                      controls=["the Ramanujan bound verified on each mask's spectrum, not assumed",
                                "random sparse mask at IDENTICAL density as the control for A3",
                                "a parameter-matched dense net, not merely a width-matched one",
                                "identical folds and seed as the GBM arm",
                                "the ensemble asked separately, since trees and nets differ"],
                      note="Ramanujan-expander sparse network on the loop 170 feature set")
    out = {"test": "ramanujan sparse network", "gates": gates, "n": n_ok,
           "spectra": spec, "hit1": {k: float(v.mean()) for k, v in h.items()},
           "gbm": float(hg.mean()), "ensemble": float(he.mean()),
           "a2": [d2, s2], "a3": [d3, s3], "a4": [d4, s4], "a5": [d5, s5],
           "n_param_sparse": nparam_sparse, "dense_width": wdense,
           "manifest": man, "seconds": time.time() - t0, "log": log}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'PASS' if v else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [{time.time()-t0:.0f}s]")
    say("=" * 104)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
