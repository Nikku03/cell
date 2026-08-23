"""Loop 172. Molecular structure descriptors on the candidates, and the annotation-bias gate.

THE HYPOTHESIS THIS TESTS. Loop 169 found the ranker's remaining errors concentrated in isomer
ambiguity: 85.5% of the single-new-product errors have a median of 4 chemically distinct molecules
that all balance the residual exactly, because Human-GEM stores formulas and a formula cannot tell
a ring oxygen from a side-chain one. 5,095 metabolites (60.2%) have now been resolved to a structure
and 5,087 carry 14 RDKit descriptors -- real H-bond donors and acceptors, TPSA, logP, molar
refractivity, rotatable bonds, aromatic rings, Bertz complexity. Those separate isomers by
construction. The question is whether they separate the RIGHT one.

THE RISK THAT COULD FAKE THE WHOLE RESULT, and why B1 and B4 exist. Coverage is not random. The
true answer is resolvable in 72.1% of the ambiguity cases while overall candidate coverage is 60.2%,
so positives are BETTER ANNOTATED than negatives. A tree model handles missing values natively, and
"this candidate has a structure at all" is therefore available to it as a feature in everything but
name. A gain driven by that is an artefact of who bothered to annotate what, not chemistry.

    B1 measures the bias directly: coverage among positives against coverage among negatives.
    B4 removes it: the same comparison restricted to cases where EVERY candidate is resolvable, so
       missingness carries no information at all. B4 is the gate that decides the claim.

PREDECLARED, before any number is looked at.

  B1 HOW BIG IS THE ANNOTATION BIAS? Coverage among true products against coverage among decoys, on
     the shortlist rows.
     Gate: passes on being reported. Its purpose is that B2's number is read knowing the size of the
     confound rather than after it.

  B2 DOES IT HELP OVERALL? hit@1 with 14 descriptors added, against loop 170's 0.8506 on identical
     folds and seed.
     Gate: more than 3 sem.

  B3 DOES IT HELP THE BUCKET IT TARGETS? hit@1 on the isomer-ambiguity cases specifically -- single
     new product, two or more chemically distinct competitors that all balance exactly.
     Gate: more than 3 sem. Separated from B2 so a general drift cannot be reported as the fix.

  B4 IS THE GAIN CHEMISTRY OR ANNOTATION? The same comparison on the subset where every shortlist
     candidate is resolvable, so the presence or absence of a structure says nothing.
     Gate: PASS iff the gain survives at more than 3 sem on that subset. A FAIL means the model
     learned who is well documented, and the result is withdrawn rather than qualified.

  B5 WHICH DESCRIPTOR CARRIES IT. Permutation importance over the new features.
     Gate: passes on being reported.

  B6 WHAT THIS CANNOT SHOW. DEV only. 39.8% of metabolites have no structure and they are the
     unusual ones -- in loop 169's worked example the unannotated molecules were the estradiol
     quinones, including the answer. Descriptors are 2D: no conformer, no 3D shape, no tautomer
     handling.

-> outputs/loop_structure_rerank.json
"""
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse, stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                    # noqa: E402
import run_manifest as RM                  # noqa: E402
from rem.harness import REM, auc_of        # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_structure_rerank.json"
L170 = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/"
            "scratchpad/l170_features.npz")
CACHE = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/"
             "scratchpad/l172_features.npz")
DESC = Path("colab/data/ml/mol_descriptors.npz")
CMAX, HASH_SEED = 6, 90210
BLOCK_SCALE, ALPHA, NITER = 0.9, 0.15, 60
TOPN, NFOLD, SEED, DEG_CLIP = 20, 5, 17200, 1
L170_HIT1 = 0.8506

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    say("=" * 104)
    say("  MOLECULAR STRUCTURE DESCRIPTORS, and the annotation-bias gate")
    say("=" * 104)
    say()
    R = REM()
    D = np.load(DESC, allow_pickle=False)
    dnames = list(map(str, D["names"]))
    dmap = {a: i for i, a in enumerate(map(str, D["accs"]))}
    DX = D["X"]
    # per-candidate descriptor matrix, NaN where unresolved
    CD = np.full((len(R.noncur), len(dnames)), np.nan, np.float32)
    for k, i in enumerate(R.noncur):
        sp = R.species[int(i)]
        if sp in dmap:
            CD[k] = DX[dmap[sp]]
    cov = float(np.isfinite(CD[:, 0]).mean())
    say(f"     {len(dnames)} descriptors | {cov:.1%} of the {len(R.noncur):,} non-currency "
        f"candidates resolved")

    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        X, y, grp, kind, amb, allcov = (z["X"], z["y"], z["grp"], z["kind"], z["amb"], z["allcov"])
        n_ok = int(z["n"])
        say(f"     features from cache: {X.shape}")
    else:
        z0 = np.load(L170, allow_pickle=True)
        Xb, yb, grpb, kindb = z0["X"], z0["y"], z0["grp"], z0["kind"]
        n_ok = int(z0["n"])
        say(f"     loop 170 base features: {Xb.shape}; recovering shortlist identity")
        Ei = np.rint(R.Enc).astype(np.int64)
        nz = (Ei != 0).any(1)
        els = list(map(str, R.elements))
        rngh = np.random.default_rng(HASH_SEED)
        h1 = rngh.integers(-(2 ** 62), 2 ** 62, Ei.shape[1], dtype=np.int64)
        h2 = rngh.integers(-(2 ** 62), 2 ** 62, Ei.shape[1], dtype=np.int64)

        def keys(V, h):
            with np.errstate(over="ignore"):
                return (V.astype(np.int64) * h).sum(1)
        own1, own2 = Counter(), Counter()
        for c in range(1, CMAX + 1):
            for a, b in zip(keys(c * Ei[nz], h1), keys(c * Ei[nz], h2)):
                own1[int(a)] += 1
                own2[int(b)] += 1

        def tier_parts(cs):
            r0 = cs["residual"]
            r = np.rint(r0).astype(np.int64)
            solo = np.zeros(len(Ei))
            pair = np.zeros(len(Ei))
            feas = np.zeros(len(Ei))
            if np.abs(r0 - r).max() > 1e-6 or (r < 0).any():
                return solo, pair, feas
            d1, d2 = Counter(), Counter()
            for s in cs["seeds"]:
                k = R.ncmap[int(s)]
                if not nz[k]:
                    continue
                for c in range(1, CMAX + 1):
                    d1[int(keys((c * Ei[k])[None, :], h1)[0])] += 1
                    d2[int(keys((c * Ei[k])[None, :], h2)[0])] += 1
            feas = (((r - Ei) >= 0).all(1) & nz).astype(float)
            for c in range(1, CMAX + 1):
                L = r - c * Ei
                ok = (L >= 0).all(1) & nz
                solo[ok & (L == 0).all(1)] = 1.0
                if not ok.any():
                    continue
                k1, k2 = keys(L, h1), keys(L, h2)
                for i in np.where(ok)[0]:
                    a, b = int(k1[i]), int(k2[i])
                    if own1[a] - d1.get(a, 0) > 0 and own2[b] - d2.get(b, 0) > 0:
                        pair[i] = 1.0
            return solo, pair, feas

        def r01(v):
            return (stats.rankdata(v, "average") - 1) / max(len(v) - 1, 1)
        rows, ambl, covl = [], [], []
        gi = 0
        for j in R.dev:
            cs = R.case(j)
            if cs is None:
                continue
            m = ~cs["excl"]
            if cs["pos"][m].sum() == 0 or (~cs["pos"][m]).sum() == 0:
                continue
            solo, pair, feas = tier_parts(cs)
            T = 8 * solo + 4 * pair + 1 * feas
            w = R.walk(R.operator(cs["j"]), cs["seeds"])[:R.NS][R.noncur]
            zw = np.zeros(len(Ei))
            mm = w > 0
            nn = int(mm.sum())
            if nn == 1:
                zw[mm] = 1.0
            elif nn > 1:
                zw[mm] = 0.001 + 0.999 * (stats.rankdata(w[mm], "average") - 1) / (nn - 1)
            rb = r01(R.balance_score(cs["residual"]))
            blend = T + BLOCK_SCALE * np.maximum(zw, rb) + 1e-6 * rb
            blend[cs["excl"]] = -np.inf
            cols = np.argsort(-blend, kind="stable")[:TOPN]
            sd = CD[cols]
            # seed-relative deltas, using the seeds that are themselves resolved
            sidx = [R.ncmap[int(s)] for s in cs["seeds"] if int(s) in R.ncmap]
            sref = np.nanmean(CD[sidx], 0) if sidx else np.full(len(dnames), np.nan)
            rows.append(np.hstack([sd, sd - sref]))
            # bucket labels
            names = {R.sp_name[int(R.noncur[k])] for k in np.where(solo & ~cs["excl"])[0]}
            is_amb = (not ({R.sp_name[i] for i in cs["seeds"]}
                           & {R.sp_name[i] for i in cs["targets"]})
                      and len(cs["targets"]) == 1 and len(names) >= 2)
            ambl.append(1.0 if is_amb else 0.0)
            covl.append(1.0 if np.isfinite(sd[:, 0]).all() else 0.0)
            gi += 1
            if gi % 1000 == 0:
                say(f"     {gi:,}/{n_ok:,} [{time.time()-t0:.0f}s]")
        SD = np.vstack(rows)
        X = np.hstack([Xb, SD])
        y, grp, kind = yb, grpb, kindb
        amb, allcov = np.array(ambl), np.array(covl)
        np.savez_compressed(CACHE, X=X, y=y, grp=grp, kind=kind, amb=amb,
                            allcov=allcov, n=n_ok)
        say(f"     built and cached: {X.shape} [{time.time()-t0:.0f}s]")

    NB = X.shape[1] - 2 * len(dnames)
    NEW = [f"d_{n}" for n in dnames] + [f"dd_{n}" for n in dnames]
    cases = np.unique(grp)
    say(f"     {n_ok:,} cases | {NB} base + {len(NEW)} structure features")

    # ------------------------------------------------------------------ B1
    say()
    say("B1 THE ANNOTATION BIAS")
    has = np.isfinite(X[:, NB])
    cp = float(has[y == 1].mean())
    cn = float(has[y == 0].mean())
    say(f"     shortlist rows with a structure: positives {cp:.1%}, decoys {cn:.1%}, "
        f"difference {cp-cn:+.1%}")
    say(f"     lone AUC of the has-structure indicator: {auc_of(has.astype(float), y.astype(bool)):.4f}")
    say(f"     cases where EVERY candidate is resolved: {allcov.mean():.1%} "
        f"({int(allcov.sum()):,} of {n_ok:,})")
    say(f"     isomer-ambiguity cases: {amb.mean():.1%} ({int(amb.sum()):,})")
    b1 = True
    say(f"     B1 {'PASS' if b1 else 'FAIL'}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    def oof(cols):
        p = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=NFOLD).split(X, y, grp):
            p[te] = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                random_state=0).fit(X[tr][:, cols], y[tr]).predict_proba(X[te][:, cols])[:, 1]
        return p
    p_base = oof(list(range(NB)))
    p_all = oof(list(range(X.shape[1])))

    def hits(p, sel=None):
        cs_ = cases if sel is None else cases[sel]
        return np.array([1.0 if y[grp == g].astype(bool)[np.argmax(p[grp == g])] else 0.0
                         for g in cs_])
    hb, ha = hits(p_base), hits(p_all)

    def pd(a, b):
        d = a - b
        return float(d.mean()), float(d.std() / np.sqrt(len(d)))

    # ------------------------------------------------------------------ B2
    d2, s2 = pd(ha, hb)
    b2 = bool(d2 > 3 * s2)
    say()
    say(f"B2 OVERALL hit@1: {hb.mean():.4f} -> {ha.mean():.4f} = {d2:+.4f} sem {s2:.4f} "
        f"({d2/s2:+.1f} sem)   [loop 170: {L170_HIT1:.4f}]")
    GG.verdict(b2, emit=say, if_true="structure descriptors help overall.",
               if_false="structure descriptors do not move the overall number.")
    say(f"     B2 {'PASS' if b2 else 'FAIL'}")

    # ------------------------------------------------------------------ B3
    sel = amb.astype(bool)
    d3, s3 = pd(hits(p_all, sel), hits(p_base, sel))
    b3 = bool(d3 > 3 * s3)
    say()
    say(f"B3 ISOMER-AMBIGUITY cases (n={int(sel.sum()):,}): "
        f"{hits(p_base, sel).mean():.4f} -> {hits(p_all, sel).mean():.4f} = {d3:+.4f} "
        f"sem {s3:.4f} ({d3/s3:+.1f} sem)")
    GG.verdict(b3, emit=say, if_true="the targeted bucket improves.",
               if_false="the bucket structure descriptors were added for does not improve.")
    say(f"     B3 {'PASS' if b3 else 'FAIL'}")

    # ------------------------------------------------------------------ B4
    sel4 = allcov.astype(bool)
    d4, s4 = pd(hits(p_all, sel4), hits(p_base, sel4))
    b4 = bool(d4 > 3 * s4)
    say()
    say(f"B4 FULLY-RESOLVED cases only (n={int(sel4.sum()):,}), where missingness says nothing: "
        f"{hits(p_base, sel4).mean():.4f} -> {hits(p_all, sel4).mean():.4f} = {d4:+.4f} "
        f"sem {s4:.4f} ({d4/s4:+.1f} sem)")
    GG.verdict(b4, emit=say, if_true=(
        "the gain survives where every candidate has a structure, so it is chemistry and not a "
        "record of who bothered to annotate what."), if_false=(
        "the gain does not survive once missingness carries no information. The model learned "
        "which molecules are well documented, and the result is WITHDRAWN rather than qualified."))
    say(f"     B4 {'PASS' if b4 else 'FAIL'}")

    # ------------------------------------------------------------------ B5
    say()
    say("B5 WHICH DESCRIPTOR")
    rng = np.random.default_rng(SEED)
    tr, te = next(iter(GroupKFold(n_splits=NFOLD).split(X, y, grp)))
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                         min_samples_leaf=40, random_state=0).fit(X[tr], y[tr])

    def h_of(P):
        return float(np.mean([1.0 if y[te][grp[te] == g].astype(bool)[np.argmax(P[grp[te] == g])]
                              else 0.0 for g in np.unique(grp[te])]))
    b0 = h_of(clf.predict_proba(X[te])[:, 1])
    imp = {}
    for k in range(NB, X.shape[1]):
        Xp = X[te].copy()
        Xp[:, k] = rng.permutation(Xp[:, k])
        imp[NEW[k - NB]] = b0 - h_of(clf.predict_proba(Xp)[:, 1])
    for f, v in sorted(imp.items(), key=lambda kv: -kv[1])[:8]:
        say(f"     {f:<18s} {v:+.4f}")
    b5 = True
    say(f"     B5 {'PASS' if b5 else 'FAIL'}")

    say()
    say("B6 WHAT THIS CANNOT SHOW")
    say(f"     DEV only. {1-cov:.1%} of candidates have no structure and they are the unusual ones;")
    say("     in loop 169's worked example the unannotated molecules were the estradiol quinones,")
    say("     including the right answer.")
    say("     Descriptors are 2D: no conformer, no 3D shape, no tautomer handling.")
    b6 = True
    say(f"     B6 {'PASS' if b6 else 'FAIL'}")

    gates = {"B1": b1, "B2": b2, "B3": b3, "B4": b4, "B5": b5, "B6": b6}
    man = RM.manifest(inputs=[DESC, L170], available=n_ok, used=n_ok, selection="all", seed=SEED,
                      controls=["annotation bias measured at B1 before any performance number",
                                "the targeted bucket gated separately from the overall number",
                                "B4 restricts to fully-resolved cases so missingness carries nothing",
                                "identical folds and seed as the no-structure arm",
                                "permutation importance over the new features only"],
                      note="RDKit 2D descriptors added to the shortlist ranker, with the annotation-bias control")
    out = {"test": "structure rerank", "gates": gates, "n": n_ok, "coverage": cov,
           "bias": {"pos": cp, "neg": cn}, "hit1_base": float(hb.mean()),
           "hit1_struct": float(ha.mean()), "b2": [d2, s2], "b3": [d3, s3], "b4": [d4, s4],
           "n_amb": int(amb.sum()), "n_allcov": int(allcov.sum()), "importance": imp,
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
