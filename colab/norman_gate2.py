"""GATE 2: is the NON-ADDITIVE part of a pair's response predictable from the two genes' annotations?

Gate 1 established that 47.1% of pair-effect energy is non-additive and 1.69x the split-half noise floor. That
makes a pair screen non-redundant. It does not make it LEARNABLE. If the interaction term cannot be predicted for
a pair the model has never seen, a pair screen buys coverage of the ~190M pair space one pair at a time, which is
not a route to anything. This decides which.

TARGET.  resid_AB = Delta_AB - (Delta_A + Delta_B), projected onto a K-component SVD basis fitted on TRAINING
pairs only. Predicting the code, then decoding, is the same construction as the programme model.

FEATURES.  Pairs are unordered, so the pair representation must be symmetric in A and B:
    [ x_A + x_B ,  |x_A - x_B| ,  x_A * x_B ]
built from the 696 named channels. The product term is the only place a genuine interaction can live: sum and
absolute-difference are symmetric functions that a purely additive model could also use.

TWO SPLITS, and the second is the real one.
    pair    hold out whole pairs at random. A test pair may still share a gene with training pairs.
    gene    hold out whole GENES: no test pair shares either gene with any training pair. This is the setting
            that matters, because the 190M unmeasured pairs are pairs of genes, and most involve genes with no
            measured pair at all.

ARMS.  ridge / gbm / knn on the symmetric pair features; marginal (the mean training residual, no pair-specific
information -- the floor); shuffled (pair features permuted across pairs -- must lose).

PREDECLARED.  Gate 2 passes only if ridge or gbm beats marginal past a bootstrap CI under the GENE split. Passing
only under the pair split would mean the model recognises the genes, not the interaction.

POWER, stated up front: 131 pairs. Five folds leave ~26 test pairs each. This test can detect a large effect and
cannot detect a small one; the MDE is reported alongside every contrast.
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/norman")
OUT = A.OUT
KC, FOLDS, BOOT, MIN_CELLS = 16, 5, 10_000, 40


def main():
    log = []
    def report(t):
        print(t, flush=True); log.append(t)
    report("=" * 100)
    report("GATE 2: is the non-additive interaction predictable from annotation?")
    report("=" * 100)
    report("  PREDECLARED: passes only if ridge or gbm beats marginal under the GENE split.")

    z = np.load(SP / "pseudobulk.npz", allow_pickle=True)
    groups = [str(g) for g in z["groups"]]; genes = [str(g) for g in z["genes"]]
    prof, ncell, ph = z["profile"], z["ncell"], z["half"]
    idx = {g: i for i, g in enumerate(groups)}
    c = idx["control"]
    singles = {g: idx[g] for g in groups if "_" not in g and g != "control"}
    pairs = {}
    for g in groups:
        if "_" in g and "NegCtrl" not in g:
            a, b = g.split("_", 1)
            if a in singles and b in singles and ncell[idx[g]] >= MIN_CELLS:
                pairs[g] = (idx[g], a, b)
    keys = sorted(pairs)
    report(f"  {len(singles)} singles, {len(keys)} pairs")

    D = prof - prof[c][None, :]
    R = np.stack([D[pairs[k][0]] - D[singles[pairs[k][1]]] - D[singles[pairs[k][2]]] for k in keys])
    report(f"  residual matrix {R.shape}")

    universe = sorted(set(singles) | {"control"})
    # channels must be built over a universe containing the Norman genes
    zk = np.load(A.SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in zk["kos"]]
    uni = sorted(set(kos) | set(singles))
    base, _ = A.build_channels(uni, set(kos), report)
    extra, _, tb = C.extra_channels(uni, set(kos), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    urow = {g: i for i, g in enumerate(uni)}
    covered = sum(1 for g in singles if g in urow)
    report(f"  channels {chan.shape[1]}; {covered}/{len(singles)} Norman genes have annotation")

    def feat(k):
        _, a, b = pairs[k]
        xa = chan[urow[a]] if a in urow else np.zeros(chan.shape[1], np.float32)
        xb = chan[urow[b]] if b in urow else np.zeros(chan.shape[1], np.float32)
        return np.concatenate([xa + xb, np.abs(xa - xb), xa * xb])
    X = np.stack([feat(k) for k in keys]).astype(np.float32)
    report(f"  symmetric pair features: {X.shape[1]} columns (sum | absdiff | product)")

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    rng = np.random.default_rng(A.SEED)
    Xs = X[rng.permutation(len(X))]
    summary = {}
    for split in ("pair", "gene"):
        if split == "pair":
            fold_of = np.array([i % FOLDS for i in rng.permutation(len(keys))])
        else:
            gs = sorted({g for k in keys for g in pairs[k][1:]})
            asg = {g: i % FOLDS for i, g in enumerate(rng.permutation(len(gs)))}
            asg = {gs[int(u)]: i % FOLDS for i, u in enumerate(rng.permutation(len(gs)))}
            fold_of = np.array([max(asg[pairs[k][1]], asg[pairs[k][2]]) for k in keys])
        per = {a: [] for a in ("ridge", "gbm", "knn", "marginal", "shuffled")}
        for f in range(FOLDS):
            te = np.where(fold_of == f)[0]
            if split == "gene":
                bad = {g for k in [keys[i] for i in te] for g in pairs[k][1:]}
                tr = np.array([i for i in range(len(keys)) if fold_of[i] != f
                               and pairs[keys[i]][1] not in bad and pairs[keys[i]][2] not in bad])
            else:
                tr = np.where(fold_of != f)[0]
            if len(tr) < 20 or not len(te): continue
            u, s, vt = np.linalg.svd(R[tr] - R[tr].mean(0), full_matrices=False)
            B = vt[:KC].astype(np.float32); mu = R[tr].mean(0)
            Y = (R[tr] - mu) @ B.T
            pred = {"ridge": A.ridge_apply(X[te], A.ridge_fit(X[tr], Y, 10.0)),
                    "knn": KNeighborsRegressor(n_neighbors=min(5, len(tr))).fit(X[tr], Y).predict(X[te]),
                    "marginal": np.zeros((len(te), KC), np.float32),
                    "shuffled": A.ridge_apply(Xs[te], A.ridge_fit(Xs[tr], Y, 10.0))}
            gb = np.zeros((len(te), KC), np.float32)
            for j in range(KC):
                gb[:, j] = HistGradientBoostingRegressor(max_iter=60, random_state=0)\
                    .fit(X[tr], Y[:, j]).predict(X[te])
            pred["gbm"] = gb
            for arm, P in pred.items():
                for r, i in enumerate(te):
                    hat = P[r] @ B + mu
                    t = R[i]
                    per[arm].append(float(np.corrcoef(hat, t)[0, 1]) if hat.std() > 0 else 0.0)
        report(f"\n### {split} split  ({len(per['ridge'])} held-out pairs scored)")
        vals = {a: np.array(v) for a, v in per.items()}
        for a in ("gbm", "ridge", "knn", "marginal", "shuffled"):
            report(f"  {a:<10} mean corr(pred, true residual) {np.nanmean(vals[a]):+.4f}")
        out = {}
        for nm, a_, b_ in (("ridge - marginal", "ridge", "marginal"),
                           ("gbm - marginal", "gbm", "marginal"),
                           ("ridge - shuffled", "ridge", "shuffled")):
            d = vals[a_] - vals[b_]; d = d[np.isfinite(d)]
            br = np.random.default_rng(0)
            bs = d[br.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
            lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            report(f"  {nm:<20} {d.mean():>+8.4f} [{lo:+.4f}, {hi:+.4f}]  "
                   f"{'RESOLVED' if (lo>0 or hi<0) else 'within noise'}   (n={len(d)})")
            out[nm] = {"gap": float(d.mean()), "ci": [lo, hi], "n": int(len(d))}
        summary[split] = {"means": {a: float(np.nanmean(v)) for a, v in vals.items()},
                          "contrasts": out}
    json.dump({"test": "norman_gate2", "n_pairs": len(keys), "K": KC,
               "summary": summary, "log": log}, open(OUT / "norman_gate2.json", "w"), indent=2)
    report(f"\n  -> {OUT/'norman_gate2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
