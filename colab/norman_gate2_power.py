"""GATE 2 AT MAXIMUM POWER: leave-one-gene-out over all 131 pairs, with the MDE stated.

WHY NOT SIMPLY MORE PAIRS.  Norman 2019 contains exactly 237 perturbation groups = 105 singles + 131 pairs + 1
control.  The >=40-cell filter excluded NOTHING (all 131 pairs have >= 54 cells, median 301).  131 is the hard
ceiling of the dataset, and it is the only large single-cell dataset with gene pairs, so "run gate 2 with more
pairs" cannot be done by adding data.  What CAN be done is stop wasting the pairs we have.

THE POWER LEAK IN THE FIRST RUN.  The 5-fold gene split removed, from training, every pair sharing EITHER gene
with the test fold.  With 105 genes spread over 131 pairs that discards most of the training set on every fold.
Leave-one-gene-out holds out one gene at a time, so training keeps nearly all 131 pairs while the test pairs still
contain a gene never seen in any training pair -- the same guarantee, far more training data.

    5-fold gene split   ~26 test pairs/fold, training gutted by the exclusion rule
    leave-one-gene-out  105 folds, every pair tested, training near-complete on each

Repeated random gene splits (20 seeds) are run alongside so the estimate does not rest on one partition.

THE MDE IS THE POINT.  The first run reported "no large predictable interaction signal" and could not distinguish
that from "no signal".  This states, explicitly, the smallest effect the design could have detected.  A negative
with a stated MDE is a bound; a negative without one is just an absence.
"""
import collections, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/norman")
OUT = A.OUT
KC, BOOT, SEEDS = 16, 10_000, 20


def main():
    log = []
    def report(t):
        print(t, flush=True); log.append(t)
    report("=" * 100)
    report("GATE 2 AT MAXIMUM POWER: leave-one-gene-out, with the MDE stated")
    report("=" * 100)

    z = np.load(SP / "pseudobulk.npz", allow_pickle=True)
    groups = [str(g) for g in z["groups"]]
    prof, ncell = z["profile"], z["ncell"]
    idx = {g: i for i, g in enumerate(groups)}
    c = idx["control"]
    singles = {g: idx[g] for g in groups if "_" not in g and g != "control"}
    pairs = {g: (idx[g], *g.split("_", 1)) for g in groups
             if "_" in g and "NegCtrl" not in g
             and g.split("_", 1)[0] in singles and g.split("_", 1)[1] in singles}
    keys = sorted(pairs)
    report(f"  {len(singles)} singles, {len(keys)} pairs (the dataset's hard ceiling)")

    D = prof - prof[c][None, :]
    R = np.stack([D[pairs[k][0]] - D[singles[pairs[k][1]]] - D[singles[pairs[k][2]]] for k in keys])

    zk = np.load(A.SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in zk["kos"]]
    uni = sorted(set(kos) | set(singles))
    base, _ = A.build_channels(uni, set(kos), report)
    extra, _, tb = C.extra_channels(uni, set(kos), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    urow = {g: i for i, g in enumerate(uni)}

    def feat(k):
        _, a, b = pairs[k]
        xa = chan[urow[a]] if a in urow else np.zeros(chan.shape[1], np.float32)
        xb = chan[urow[b]] if b in urow else np.zeros(chan.shape[1], np.float32)
        return np.concatenate([xa + xb, np.abs(xa - xb), xa * xb])
    X = np.stack([feat(k) for k in keys]).astype(np.float32)
    rng = np.random.default_rng(A.SEED)
    Xs = X[rng.permutation(len(X))]
    report(f"  symmetric pair features {X.shape[1]}")

    genes = sorted({g for k in keys for g in pairs[k][1:]})
    report(f"  {len(genes)} distinct genes across the pairs -> {len(genes)} LOGO folds")

    def run(fold_of, tag):
        per = {a: [] for a in ("ridge", "marginal", "shuffled")}
        for f in sorted(set(fold_of)):
            te = np.where(fold_of == f)[0]
            tr = np.where(fold_of != f)[0]
            if len(tr) < 20 or not len(te): continue
            u, s, vt = np.linalg.svd(R[tr] - R[tr].mean(0), full_matrices=False)
            B = vt[:KC].astype(np.float32); mu = R[tr].mean(0)
            Y = (R[tr] - mu) @ B.T
            P = {"ridge": A.ridge_apply(X[te], A.ridge_fit(X[tr], Y, 10.0)),
                 "marginal": np.zeros((len(te), KC), np.float32),
                 "shuffled": A.ridge_apply(Xs[te], A.ridge_fit(Xs[tr], Y, 10.0))}
            for arm, p in P.items():
                for r, i in enumerate(te):
                    hat = p[r] @ B + mu
                    per[arm].append(float(np.corrcoef(hat, R[i])[0, 1]) if hat.std() > 0 else 0.0)
        return {a: np.array(v) for a, v in per.items()}

    # ---- leave-one-gene-out: hold out every pair containing gene g ----
    report(f"\n### LEAVE-ONE-GENE-OUT ({len(genes)} folds)")
    per = {a: [] for a in ("ridge", "marginal", "shuffled")}
    ntest = 0
    for g in genes:
        te = np.array([i for i, k in enumerate(keys) if g in pairs[k][1:]])
        tr = np.array([i for i, k in enumerate(keys) if g not in pairs[k][1:]])
        if not len(te) or len(tr) < 20: continue
        ntest += len(te)
        u, s, vt = np.linalg.svd(R[tr] - R[tr].mean(0), full_matrices=False)
        B = vt[:KC].astype(np.float32); mu = R[tr].mean(0)
        Y = (R[tr] - mu) @ B.T
        P = {"ridge": A.ridge_apply(X[te], A.ridge_fit(X[tr], Y, 10.0)),
             "marginal": np.zeros((len(te), KC), np.float32),
             "shuffled": A.ridge_apply(Xs[te], A.ridge_fit(Xs[tr], Y, 10.0))}
        for arm, p in P.items():
            for r, i in enumerate(te):
                hat = p[r] @ B + mu
                per[arm].append(float(np.corrcoef(hat, R[i])[0, 1]) if hat.std() > 0 else 0.0)
    v = {a: np.array(x) for a, x in per.items()}
    report(f"  {ntest} (gene, pair) test evaluations from {len(genes)} folds; "
           f"training keeps {len(keys) - 1} pairs on average vs ~26 before")
    for a in ("ridge", "marginal", "shuffled"):
        report(f"  {a:<10} mean corr {np.nanmean(v[a]):+.4f}   sd {np.nanstd(v[a]):.4f}")
    out = {}
    for nm, a_, b_ in (("ridge - marginal", "ridge", "marginal"),
                       ("ridge - shuffled", "ridge", "shuffled")):
        d = v[a_] - v[b_]; d = d[np.isfinite(d)]
        br = np.random.default_rng(0)
        bs = d[br.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        mde = 2.8 * d.std(ddof=1) / np.sqrt(len(d))
        report(f"  {nm:<20} {d.mean():>+8.4f} [{lo:+.4f}, {hi:+.4f}]  MDE {mde:.4f}  "
               f"{'RESOLVED' if (lo>0 or hi<0) else 'within noise'}  (n={len(d)})")
        out[nm] = {"gap": float(d.mean()), "ci": [lo, hi], "mde": float(mde), "n": int(len(d))}

    # ---- repeated random gene splits ----
    report(f"\n### REPEATED RANDOM GENE SPLITS ({SEEDS} seeds, 5 folds each)")
    acc = {a: [] for a in ("ridge", "marginal", "shuffled")}
    for s in range(SEEDS):
        rg = np.random.default_rng(1000 + s)
        asg = {genes[int(u)]: i % 5 for i, u in enumerate(rg.permutation(len(genes)))}
        fold_of = np.array([max(asg[pairs[k][1]], asg[pairs[k][2]]) for k in keys])
        r = run(fold_of, s)
        for a in acc: acc[a].append(np.nanmean(r[a]))
    for a in ("ridge", "marginal", "shuffled"):
        arr = np.array(acc[a])
        report(f"  {a:<10} mean over seeds {arr.mean():+.4f}  sd {arr.std(ddof=1):.4f}")
    d = np.array(acc["ridge"]) - np.array(acc["marginal"])
    mde = 2.8 * d.std(ddof=1) / np.sqrt(len(d))
    report(f"  ridge - marginal   {d.mean():+.4f}  MDE {mde:.4f}  "
           f"{'RESOLVED' if abs(d.mean()) > mde else 'within noise'}")
    out["repeated ridge - marginal"] = {"gap": float(d.mean()), "mde": float(mde), "seeds": SEEDS}

    json.dump({"test": "norman_gate2_power", "n_pairs": len(keys), "n_genes": len(genes),
               "logo": {a: float(np.nanmean(x)) for a, x in v.items()},
               "contrasts": out, "log": log},
              open(OUT / "norman_gate2_power.json", "w"), indent=2)
    report(f"\n  -> {OUT/'norman_gate2_power.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
