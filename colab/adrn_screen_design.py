"""UNCERTAINTY-DRIVEN SCREEN DESIGN: does choosing WHICH knockouts to measure beat measuring at random?

THE ARGUMENT.  The scaling curve says the next tranche of perturbations buys +0.024 per doubling IF CHOSEN
RANDOMLY.  But the best-validated capability of this model is calibrated abstention -- it knows which knockouts it
cannot call (top-25% confidence reach recall@50 0.62 against 0.49 at full coverage).  Those two facts compose:
rank the unmeasured genes by how badly the model predicts them and measure THOSE.  If model uncertainty is a
useful acquisition signal, a chosen tranche should beat a random tranche of the same size.

THE SIMULATION.  Take the training pool as the "unmeasured" world.  Start from a small seed of measured
knockouts, then grow the training set by n more, chosen by each strategy, refit everything, and score on the SAME
sealed cohorts.  The only thing that varies is WHICH knockouts were added.

    random          uniform draw.  What the scaling curve assumed and the honest control.
    uncertainty     the knockouts the current model predicts worst (lowest retrieval confidence).
    diversity       the knockouts furthest from the current measured set in annotation space -- maximise coverage
                    of the channel space rather than target the model's weakness.
    high_signal     the knockouts with the largest measured response.  This is an ORACLE (it reads the answer to
                    decide what to measure) and is reported as a denominator, not a strategy anyone can run.

PREDECLARED: an acquisition strategy is worth using only if it beats `random` past a bootstrap CI at more than one
budget.  If uncertainty ties random, then model confidence carries no information about what is worth measuring
and screens should be designed on biology, not on the model.

HONEST SCOPE: this simulates acquisition inside an already-measured pool, so the candidate genes are ones the
screen already covered.  It bounds the value of the acquisition RULE, not of a real prospective screen.
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

OUT, SP = A.OUT, A.SP
SEED_N, BUDGETS, REPEATS, BOOT = 600, (600, 1200, 2400), 3, 10_000


def main():
    log = []
    def report(t):
        print(t, flush=True); log.append(t)
    report("=" * 100)
    report("UNCERTAINTY-DRIVEN SCREEN DESIGN: does choosing what to measure beat measuring at random?")
    report("=" * 100)
    report("  PREDECLARED: a strategy is worth using only if it beats `random` at >1 budget.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]; genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05
    cix = np.where(~tide)[0]

    sealed, answers = set(), {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists(): sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists(): answers[tag] = json.loads(af.read_text())
    train = [k for k in kos if k not in sealed]
    universe = sorted(set(kos) | sealed)
    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tb = C.extra_channels(universe, set(train), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}
    rows = np.array([urow[k] for k in train])
    report(f"  candidate pool {len(train):,} knockouts; seed {SEED_N}; budgets {BUDGETS}")

    from sklearn.decomposition import NMF
    def fit_and_score(sub):
        X = Amat[[ki[train[i]] for i in sub]].copy(); X[:, tide] = 0.0
        k_comp = min(A.K, max(5, len(sub) // 8))
        nmf = NMF(n_components=k_comp, init="nndsvda", max_iter=300, random_state=0)
        W = nmf.fit_transform(X).astype(np.float32); H = nmf.components_.astype(np.float32)
        w = A.ridge_fit(chan[rows[sub]], W, A.PENALTY)
        out = {}
        for tag in ("", "_holdout"):
            ans = answers.get(tag)
            if not ans: continue
            cohort = sorted(ans)
            mix = A.ridge_apply(chan[[urow[k] for k in cohort]], w)
            hits = []
            for j, k in enumerate(cohort):
                s = (mix[j] @ H).copy(); s[tide] = -np.inf
                gi_ = genes.index(k) if k in genes else -1
                if gi_ >= 0: s[gi_] = -np.inf
                top = np.argsort(-s)[:A.NPRED]
                hits.append(len({genes[t] for t in top} & set(ans[k]["movers"])) / A.NPRED)
            out[tag or "cohort1"] = np.array(hits)
        return out, w, H

    # response magnitude, used only by the oracle strategy
    mag = np.array([(Amat[ki[k]][cix] >= A.TAU).sum() for k in train], float)

    summary = {}
    for rep in range(REPEATS):
        rng = np.random.default_rng(A.SEED + 100 * rep)
        seed_idx = rng.choice(len(train), SEED_N, replace=False)
        _, w0, H0 = fit_and_score(seed_idx)
        # current-model confidence for every unmeasured candidate: retrieval margin in channel space
        Cs = chan[rows[seed_idx]]
        Cs = Cs / (np.linalg.norm(Cs, axis=1, keepdims=True) + 1e-9)
        Ca = chan[rows]; Ca = Ca / (np.linalg.norm(Ca, axis=1, keepdims=True) + 1e-9)
        sim = Ca @ Cs.T
        conf = np.sort(sim, axis=1)[:, -5:].mean(1)      # similarity to nearest measured knockouts
        rest = np.setdiff1d(np.arange(len(train)), seed_idx)

        # DIVERSITY, done properly.  The first version of this arm sorted `1 - conf` descending, which is the
        # same permutation as sorting `conf` ascending -- it picked a byte-identical set to `uncertainty` and
        # tested nothing.  Real farthest-first maximises the minimum distance to everything ALREADY HELD,
        # including the picks made so far, which is exactly what `uncertainty` cannot do: uncertainty scores
        # each candidate against the seed alone and will happily buy 600 near-duplicates of each other so long
        # as all 600 are far from the seed.  The greedy order is nested, so it is computed once to the largest
        # budget and sliced.
        Cr = Ca[rest]
        mind = 1.0 - sim[rest].max(1)                     # distance to the nearest measured knockout
        order, taken = [], np.zeros(len(rest), bool)
        for _ in range(max(BUDGETS)):
            j = int(np.argmax(np.where(taken, -np.inf, mind)))
            order.append(j); taken[j] = True
            mind = np.minimum(mind, 1.0 - Cr @ Cr[j])
        order = np.array(order)

        for b in BUDGETS:
            if b > len(rest): continue
            picks = {
                "random": rng.choice(rest, b, replace=False),
                "uncertainty": rest[np.argsort(conf[rest])[:b]],
                "diversity": rest[order[:b]],
                "high_signal_ORACLE": rest[np.argsort(-mag[rest])[:b]],
            }
            # LIVENESS: an arm that selects the same genes as another arm is a switched-off mechanism, not a
            # null result.  This is the assertion that would have caught the bug described above.
            assert set(picks["diversity"].tolist()) != set(picks["uncertainty"].tolist()), \
                f"diversity and uncertainty picked identical sets at budget {b} -- dead arm"
            for name, pk in picks.items():
                if pk is None: continue
                sub = np.concatenate([seed_idx, pk])
                res, _, _ = fit_and_score(sub)
                for tag, v in res.items():
                    summary.setdefault((b, name, tag), []).append(v)
        report(f"  repeat {rep+1}/{REPEATS} done")

    report(f"\n  {'budget':>7} {'strategy':<20} {'cohort1':>9} {'cohort2':>9}")
    means = {}
    for (b, name, tag), vs in sorted(summary.items()):
        m = float(np.mean([v.mean() for v in vs]))
        means[(b, name, tag)] = m
    for b in BUDGETS:
        for name in ("random", "uncertainty", "diversity", "high_signal_ORACLE"):
            c1 = means.get((b, name, "cohort1")); c2 = means.get((b, name, "_holdout"))
            if c1 is None: continue
            report(f"  {b:>7,} {name:<20} {c1:>9.4f} {c2:>9.4f}")
        report("")
    report(f"  {'budget':>7} {'contrast':<26} {'cohort1':>10} {'cohort2':>10}  verdict")
    out = {}
    for b in BUDGETS:
        for name in ("uncertainty", "diversity", "high_signal_ORACLE"):
            row = []
            for tag in ("cohort1", "_holdout"):
                a = summary.get((b, name, tag)); r = summary.get((b, "random", tag))
                if not a or not r: row.append(None); continue
                d = np.concatenate([x for x in a]) - np.concatenate([x for x in r])
                br = np.random.default_rng(0)
                bs = d[br.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
                lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
                row.append((float(d.mean()), lo, hi))
            if row[0] and row[1]:
                v1, v2 = row
                res = ("RESOLVED both" if (v1[1] > 0 or v1[2] < 0) and (v2[1] > 0 or v2[2] < 0)
                       else "not both")
                report(f"  {b:>7,} {name + ' - random':<26} {v1[0]:>+10.4f} {v2[0]:>+10.4f}  {res}")
                out[f"{b}|{name}"] = {"c1": v1, "c2": v2}
    json.dump({"test": "adrn_screen_design", "seed_n": SEED_N, "budgets": list(BUDGETS),
               "repeats": REPEATS, "means": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in means.items()},
               "contrasts": out, "log": log},
              open(OUT / "adrn_screen_design.json", "w"), indent=2)
    report(f"\n  -> {OUT/'adrn_screen_design.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
