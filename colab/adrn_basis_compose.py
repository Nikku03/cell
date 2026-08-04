"""DOES THE RESPONSE SPACE COMPOSE ACROSS FUNCTIONAL CLASSES? The go/no-go before building anything.

WHY THIS TEST EXISTS.  Three independent results now say the same thing -- LOCO HARMFUL 18/18 (hold out a
functional class and the model collapses to the frequency baseline), cross-line HARMFUL 9/9, and the depth test
(the model captures a shrinking fraction of what is measurable: 85% of the ceiling at 10 cells per perturbation,
51% at 60). The diagnosis is representational, and the only lever left is "a model that composes function".

Before building that model, one thing has to be true, and it is cheap to check: **the responses of a functional
class have to be expressible as combinations of the responses of OTHER classes.** If they are not, no
compositional model exists for this data, however it is built, and the right answer is to say so.

THE MEASUREMENT.  Fit a response basis (NMF, rank K) on every perturbation EXCEPT a held-out set. Then project
each held-out response onto that basis by non-negative least squares and score how well the reconstruction
recovers that perturbation's own top-20 movers.

    The projection uses the TRUE response to fit the coefficients. That makes it an ORACLE, and it is meant to
    be: it is the UPPER BOUND on what any gene -> coefficient model could achieve given this basis. A denominator,
    not a competitor. If the oracle cannot reconstruct a held-out class, no mapping from annotations to
    coefficients can rescue it.

THE CONTROL THAT DECIDES IT.  Reconstruction being imperfect proves nothing on its own -- a basis fitted without
ANY set of perturbations reconstructs that set worse than one fitted with it. So the class holdout is compared
against a RANDOM holdout OF THE SAME SIZE, drawn from all classes. The two arms differ in exactly one thing:
whether the removed perturbations share a function.

    class-holdout ~ random-holdout   the response space composes ACROSS functional boundaries. Whatever a class
                                     does is spanned by what other classes do, and the entire failure lives in
                                     the gene -> coefficient map. BUILD THE MODEL.
    class-holdout << random-holdout   each functional class has private response directions. Nothing can compose
                                     them from the others; the constraint is coverage of function space, not
                                     architecture. DO NOT BUILD IT.

A third arm is reported as the ceiling for the measure: reconstruction of perturbations that WERE in the basis
fit. Any conclusion has to be read against how well this works at all.

PREDECLARED, before the numbers: judged by robustness.Sweeper on (class-holdout - random-holdout) at matched
size, swept over basis rank x cluster count. A resolved-negative majority means classes are private. NULL or
near-zero means they compose. Note the sign convention matches adrn_loco.py and is the REVERSE of most sweeps
here: a gap at or above zero is the GOOD outcome, so a "ROBUST" label would mean class-holdout BEAT random and
should be reported as composition, not as a win.

CLASSES are k-means on the channel matrix, the same construction adrn_loco.py uses -- the annotation geometry the
model actually sees, not an external taxonomy it may not use. Cluster count is swept because "how coarse is a
function" is an undeliberated default this project has been burned by.
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
N_CLUSTERS = (8, 16, 32)
K_VALUES = (30, 60, 120)
MIN_CLASS, MIN_MOVERS = 25, 5


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("DOES THE RESPONSE SPACE COMPOSE ACROSS FUNCTIONAL CLASSES?  (go/no-go, oracle upper bound)")
    report("=" * 100)
    report("  PREDECLARED: class-holdout ~ random-holdout => composes, build it.")
    report("               class-holdout << random-holdout => classes are private, do not build it.")
    report("  Sign convention as in adrn_loco.py: a gap at or above zero is the GOOD outcome.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    M = np.abs(z["M"]).astype(np.float32)
    tide = (M >= A.TAU).mean(0) >= 0.05
    M[:, tide] = 0.0
    report(f"  {M.shape[0]:,} perturbations x {M.shape[1]:,} genes; {int(tide.sum())} tide genes zeroed")

    # scorable perturbations only: a profile with fewer than MIN_MOVERS specific movers has no answer key
    nmov = (M >= A.TAU).sum(1)
    keep = np.where(nmov >= MIN_MOVERS)[0]
    report(f"  scorable (>= {MIN_MOVERS} specific movers): {len(keep):,}")
    M = M[keep]
    kos = [kos[i] for i in keep]

    universe = sorted(set(kos))
    base, _ = A.build_channels(universe, set(kos), report)
    extra, _, tb = C.extra_channels(universe, set(kos), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}
    Ck = chan[[urow[k] for k in kos]]

    truth = [set(np.argsort(-M[i])[:A.NPRED].tolist()) for i in range(len(kos))]

    from sklearn.cluster import KMeans
    from sklearn.decomposition import NMF
    from scipy.optimize import nnls

    def reconstruct(fit_rows, test_rows, K):
        """Fit the basis on fit_rows, project test_rows onto it with the TRUE response (oracle), score top-20."""
        nmf = NMF(n_components=K, init="nndsvda", max_iter=300, random_state=0)
        nmf.fit(M[fit_rows])
        H = nmf.components_.astype(np.float64)
        HT = H.T
        out = []
        for i in test_rows:
            w, _ = nnls(HT, M[i].astype(np.float64))
            rec = w @ H
            out.append(len(set(np.argsort(-rec)[:A.NPRED].tolist()) & truth[i]) / float(A.NPRED))
        return np.array(out)

    sw = Sweeper("class-holdout - random-holdout (>=0 is composition)",
                 axes={"clusters": list(N_CLUSTERS), "K": list(K_VALUES)})
    detail = {}
    for ncl in N_CLUSTERS:
        km = KMeans(n_clusters=ncl, n_init=4, random_state=0).fit(Ck)
        lab = km.labels_
        big = [c for c in range(ncl) if (lab == c).sum() >= MIN_CLASS]
        report(f"\n  clusters={ncl}: {len(big)} classes with >= {MIN_CLASS} members "
               f"(sizes {sorted((lab == c).sum() for c in big)})")
        for K in K_VALUES:
            cls_all, rnd_all, inb_all = [], [], []
            rng = np.random.default_rng(A.SEED + 7 * ncl + K)
            for c in big:
                test = np.where(lab == c)[0]
                fit = np.where(lab != c)[0]
                if len(fit) < K * 2:
                    continue
                cls_all.append(reconstruct(fit, test, K))

                # MATCHED RANDOM HOLDOUT: same number removed, drawn from every class
                rtest = rng.choice(len(kos), len(test), replace=False)
                rfit = np.setdiff1d(np.arange(len(kos)), rtest)
                rnd_all.append(reconstruct(rfit, rtest, K))

                # CEILING for the measure: perturbations that WERE in the basis fit
                inb = rng.choice(fit, min(len(test), len(fit)), replace=False)
                inb_all.append(reconstruct(fit, inb, K))
            if not cls_all:
                continue
            cl = np.concatenate(cls_all)
            rn = np.concatenate(rnd_all)
            ib = np.concatenate(inb_all)
            n = min(len(cl), len(rn))
            sw.add({"clusters": ncl, "K": K}, f"ncl{ncl}|K{K}", cl[:n] - rn[:n])
            detail[f"ncl={ncl}|K={K}"] = {"n": int(n), "class_holdout": float(cl.mean()),
                                          "random_holdout": float(rn.mean()), "in_basis": float(ib.mean()),
                                          "gap": float(cl[:n].mean() - rn[:n].mean())}
            report(f"    K={K:>3}: class-holdout {cl.mean():.4f} | random-holdout {rn.mean():.4f} | "
                   f"in-basis {ib.mean():.4f} | gap {cl[:n].mean() - rn[:n].mean():+.4f}")

    report(sw.report())
    v = sw.verdict()
    if sw.inert_axes() or sw.duplicate_cells():
        report(f"  GUARD: inert axes {sw.inert_axes()}, duplicate cells {sw.duplicate_cells()}")

    gaps = [d["gap"] for d in detail.values()]
    mg = float(np.mean(gaps)) if gaps else 0.0
    mc = float(np.mean([d["class_holdout"] for d in detail.values()])) if detail else 0.0
    mr = float(np.mean([d["random_holdout"] for d in detail.values()])) if detail else 0.0
    mi = float(np.mean([d["in_basis"] for d in detail.values()])) if detail else 0.0
    report(f"\n  MEANS  class-holdout {mc:.4f} | random-holdout {mr:.4f} | in-basis ceiling {mi:.4f} | "
           f"gap {mg:+.4f}")
    report("\n  READING")
    resolved_neg = sum(1 for c in sw._cells.values() if c["hi"] < 0)
    if resolved_neg > len(sw._cells) / 2:
        report("  Removing a whole FUNCTION costs more than removing the same number of perturbations at random.")
        report("  Each functional class carries response directions the other classes do not span. An oracle with")
        report("  the true response in hand cannot rebuild a held-out class from the rest -- so no mapping from")
        report("  annotations to coefficients can either. A compositional model does NOT exist for this data;")
        report("  the constraint is coverage of function space, not architecture.")
    elif abs(mg) < 0.01:
        report("  Removing a whole function costs no more than removing the same number at random. The response")
        report("  space COMPOSES across functional boundaries: what a class does is spanned by what other classes")
        report("  do. The entire failure therefore lives in the gene -> coefficient map, which is a model problem")
        report("  and is worth building. The oracle number above is the ceiling that model would be chasing.")
    else:
        report(f"  Gap {mg:+.4f} is neither resolved-negative nor near zero. Read the per-cell signs before")
        report("  concluding anything; this is not a result yet.")

    json.dump({"test": "adrn_basis_compose", "clusters": list(N_CLUSTERS), "K_values": list(K_VALUES),
               "detail": detail, "means": {"class_holdout": mc, "random_holdout": mr, "in_basis": mi, "gap": mg},
               "sweep": v, "log": log}, open(OUT / "adrn_basis_compose.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'adrn_basis_compose.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
