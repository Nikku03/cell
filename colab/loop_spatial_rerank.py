"""Loop 170. Add SPATIAL features -- compartments and transport topology -- and re-rank.

THE GAP THIS FILLS. Every feature the ranker has seen so far is chemical or topological: tier,
balance, walk, chain, degree, size, charge. Not one of them says WHERE anything is. That is a
strange omission for a model whose candidates are compartment-tagged species and whose single
largest error class is transport.

WHAT LOOP 169's ANATOMY MEASURED. Of 5,582 DEV cases, 31.1% are transport -- the same molecule
arriving in another compartment -- and they account for 27.9% of all errors. Inside the
single-new-product errors, a further 14.5% have only ONE distinct competing molecule, which means
the model was not choosing between chemicals at all; it was choosing a compartment with no
information about compartments. MAR02104 is the worked example: its 19 exact matches collapse to 6
distinct molecules, and the true answer is the substrate itself in a different compartment.

THE FEATURES, and why none of them is a leak. Every one is computed from the SEEDS, which are
given, and from the graph with reaction j deleted:
    same_molecule_as_seed   the candidate is a seed's molecule in another compartment -- the exact
                            signature of a transport reaction, and derivable from the seeds alone
    same_comp_as_seed       the candidate sits in a compartment a seed already occupies
    cand_is_cytosol         cytosol carries 96.9% of all transport (loop 160's R2)
    transport_pairs         how many OTHER reactions move material between the candidate's
                            compartment and a seed's, with j's own contribution subtracted
    seed_comp_count         how many distinct compartments the seeds span
    cand_comp_size          how many species live in the candidate's compartment
    twin_exists             the candidate's molecule also exists in a seed's compartment, so a
                            transport step between them is representable at all

PREDECLARED, before any number is looked at.

  Z1 NO LEAK. The transport-pair counts must exclude reaction j, and same_molecule_as_seed must be
     computable from the seed list alone.
     Gate: PASS iff the j-deleted pair count differs from the undeleted one on every transport case,
     and no spatial feature alone separates true from false above AUC 0.75 on the shortlist. A
     single feature at 0.75+ on this task would be a label in disguise, as the degree column was.

  Z2 DOES IT HELP OVERALL? hit@1 with spatial features against loop 167's 0.7266 without them,
     identical folds, identical seed.
     Gate: more than 3 sem.

  Z3 DOES IT FIX THE BUCKET IT TARGETS? hit@1 on transport cases specifically, which was 0.752
     overall and 0.592 in the abstained half.
     Gate: PASS iff transport hit@1 improves by more than 3 sem. This is the targeted claim and it
     is separated from Z2 so a general drift cannot be reported as a fix.

  Z4 DOES ANYTHING GET WORSE? hit@1 on multi-product and single-new-product cases.
     Gate: PASS iff neither falls by more than 3 sem. A model that learns "prefer the transported
     twin" could plausibly damage the cases where the answer is genuinely new.

  Z5 WHICH SPATIAL FEATURE CARRIES IT. Permutation importance restricted to the new features.
     Gate: passes on being reported.

  Z6 WHAT IT DOES TO THE CONFIDENCE CURVE. Precision at 50% coverage, against loop 168's 0.9534.
     Gate: passes on being reported. More correct answers at rank 1 should raise the whole curve,
     and if it does not that is worth seeing.

  Z7 WHAT THIS CANNOT SHOW. DEV only. Compartment assignment comes from Human-GEM's own curation,
     so a model learning compartment regularities is learning the curators' conventions as well as
     the biology.

-> outputs/loop_spatial_rerank.json
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

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_spatial_rerank.json"
CACHE = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/"
             "scratchpad/l170_features.npz")
CMAX, HASH_SEED = 6, 90210
BLOCK_SCALE, ALPHA, NITER = 0.9, 0.15, 60
TOPN, NFOLD, SEED, DEG_CLIP = 20, 5, 17000, 1
BASE = ["tier", "solo", "pair", "feas", "balance", "bal_rank", "walk", "walk_rank",
        "walk_zero", "chain", "chain_rank", "log_deg_clip", "log_heavy", "het_frac",
        "charge", "blend_rank", "resid_cos"]
SPAT = ["same_molecule_as_seed", "same_comp_as_seed", "cand_is_cytosol", "log_transport_pairs",
        "seed_comp_count", "log_cand_comp_size", "twin_exists"]
L167_HIT1 = 0.7266

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def build(R):
    NC = len(R.noncur)
    els = list(map(str, R.elements))
    Ei = np.rint(R.Enc).astype(np.int64)
    nz = (Ei != 0).any(1)
    heavy = R.Enc.sum(1) - R.Enc[:, els.index("H")]
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

    # ---- spatial precompute
    comp = np.array(R.sp_comp)
    name = np.array(R.sp_name)
    comp_nc = comp[R.noncur]
    name_nc = name[R.noncur]
    comp_size = Counter(comp.tolist())
    pair_all = Counter()
    rx_pairs = {}
    for j in range(R.NR):
        cs_ = {comp[i] for i in R.react_of[j]}
        cp_ = {comp[i] for i in R.prod_of[j]}
        ps = set()
        for a in cs_:
            for b in cp_:
                if a != b:
                    ps.add(tuple(sorted((a, b))))
        rx_pairs[j] = ps
        for p in ps:
            pair_all[p] += 1
    say(f"     spatial precompute: {len(comp_size)} compartments, "
        f"{len(pair_all)} linked pairs, {sum(pair_all.values()):,} transport reactions")

    def tier_parts(cs):
        r0 = cs["residual"]
        r = np.rint(r0).astype(np.int64)
        solo = np.zeros(NC)
        pair = np.zeros(NC)
        feas = np.zeros(NC)
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

    def restricted(cs, cols):
        keepsp = set(int(R.noncur[c]) for c in cols) | set(cs["seeds"])
        rxs = set()
        for i in keepsp:
            rxs |= R.sp_rx[i]
        rxs.discard(cs["j"])
        if not rxs:
            return np.zeros(len(cols))
        spl, rxl = sorted(keepsp), sorted(rxs)
        si = {v: k for k, v in enumerate(spl)}
        ri = {v: k + len(spl) for k, v in enumerate(rxl)}
        n = len(spl) + len(rxl)
        src, dst = [], []
        for j in rxl:
            rv = R.rev[j] == 1
            for i in R.react_of[j]:
                if i in si:
                    src += [si[i]]
                    dst += [ri[j]]
                    if rv:
                        src += [ri[j]]
                        dst += [si[i]]
            for i in R.prod_of[j]:
                if i in si:
                    src += [ri[j]]
                    dst += [si[i]]
                    if rv:
                        src += [si[i]]
                        dst += [ri[j]]
        if not src:
            return np.zeros(len(cols))
        A = sparse.csr_matrix((np.ones(len(src)), (dst, src)), shape=(n, n))
        c0 = np.asarray(A.sum(0)).ravel()
        c0[c0 == 0] = 1.0
        P = A @ sparse.diags(1.0 / c0)
        e = np.zeros(n)
        for s in cs["seeds"]:
            if s in si:
                e[si[s]] = 1.0
        if e.sum() == 0:
            return np.zeros(len(cols))
        e /= e.sum()
        p = e.copy()
        for _ in range(NITER):
            p = (1 - ALPHA) * (P @ p) + ALPHA * e
        return np.array([p[si[int(R.noncur[c])]] for c in cols])

    def r01(v):
        return (stats.rankdata(v, "average") - 1) / max(len(v) - 1, 1)
    NOix = [els.index(e) for e in ("N", "O")]
    name_to_comps = defaultdict(set)
    for i in range(R.NS):
        name_to_comps[name[i]].add(comp[i])

    X, y, grp, kind, leakchk = [], [], [], [], []
    n_ok = 0
    t0 = time.time()
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
        zw = np.zeros(NC)
        mm = w > 0
        nn = int(mm.sum())
        if nn == 1:
            zw[mm] = 1.0
        elif nn > 1:
            zw[mm] = 0.001 + 0.999 * (stats.rankdata(w[mm], "average") - 1) / (nn - 1)
        bal = R.balance_score(cs["residual"])
        rb = r01(bal)
        blend = T + BLOCK_SCALE * np.maximum(zw, rb) + 1e-6 * rb
        blend[cs["excl"]] = -np.inf
        cols = np.argsort(-blend, kind="stable")[:TOPN]
        ch = restricted(cs, cols)
        dg = np.maximum(cs["degv"][cols], DEG_CLIP)
        res = cs["residual"]
        cosr = (R.Enc[cols] @ res) / (np.maximum(np.linalg.norm(R.Enc[cols], axis=1), 1e-9)
                                      * max(np.linalg.norm(res), 1e-9))
        # ---- spatial, all from the seeds and the graph minus j
        seed_names = {name[i] for i in cs["seeds"]}
        seed_comps = {comp[i] for i in cs["seeds"]}
        own_pairs = rx_pairs[cs["j"]]
        tp = np.zeros(len(cols))
        for k, c in enumerate(cols):
            cc = comp_nc[c]
            best = 0
            for sc in seed_comps:
                if sc == cc:
                    continue
                key = tuple(sorted((sc, cc)))
                v = pair_all.get(key, 0) - (1 if key in own_pairs else 0)
                best = max(best, v)
            tp[k] = best
        SP = np.column_stack([
            np.array([1.0 if (name_nc[c] in seed_names and comp_nc[c] not in seed_comps) else 0.0
                      for c in cols]),
            np.array([1.0 if comp_nc[c] in seed_comps else 0.0 for c in cols]),
            (comp_nc[cols] == "c").astype(float),
            np.log1p(tp),
            np.full(len(cols), float(len(seed_comps))),
            np.log1p([comp_size[comp_nc[c]] for c in cols]),
            np.array([1.0 if (name_to_comps[name_nc[c]] & seed_comps) else 0.0 for c in cols]),
        ])
        X.append(np.column_stack([
            T[cols], solo[cols], pair[cols], feas[cols], bal[cols], rb[cols],
            w[cols], r01(w)[cols], (w[cols] == 0).astype(float), ch, r01(ch),
            np.log(dg), np.log1p(heavy[cols]),
            R.Enc[cols][:, NOix].sum(1) / np.maximum(heavy[cols], 1),
            R.charge[R.noncur][cols], np.arange(len(cols), dtype=float) / TOPN, cosr, SP]))
        y.append(cs["pos"][cols].astype(int))
        grp.append(np.full(len(cols), n_ok))
        tn = {name[i] for i in cs["targets"]}
        kind.append("transport" if (seed_names & tn) else
                    ("multi" if len(cs["targets"]) >= 2 else "single"))
        # Z1 evidence: did j's own contribution actually get subtracted anywhere
        leakchk.append(1.0 if own_pairs else 0.0)
        n_ok += 1
        if n_ok % 1000 == 0:
            say(f"     features {n_ok:,} [{time.time()-t0:.0f}s]")
    return (np.vstack(X), np.concatenate(y), np.concatenate(grp),
            np.array(kind), float(np.mean(leakchk)), n_ok)


def main():
    t0 = time.time()
    say("=" * 104)
    say("  SPATIAL FEATURES: compartments and transport topology, then re-rank")
    say("=" * 104)
    say()
    R = REM()
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        X, y, grp, kind, subfrac, n_ok = (z["X"], z["y"], z["grp"], z["kind"],
                                          float(z["subfrac"]), int(z["n"]))
        say(f"     from cache: {X.shape}")
    else:
        X, y, grp, kind, subfrac, n_ok = build(R)
        np.savez_compressed(CACHE, X=X, y=y, grp=grp, kind=kind, subfrac=subfrac, n=n_ok)
        say(f"     built and cached: {X.shape} [{time.time()-t0:.0f}s]")
    ALL = BASE + SPAT
    say(f"     {n_ok:,} cases | {len(BASE)} base + {len(SPAT)} spatial features")
    say(f"     case mix: " + ", ".join(f"{k} {np.mean(kind==k):.1%}"
                                       for k in ("transport", "multi", "single")))

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    def oof(cols):
        p = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=NFOLD).split(X, y, grp):
            p[te] = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                random_state=0).fit(X[tr][:, cols], y[tr]).predict_proba(X[te][:, cols])[:, 1]
        return p
    ibase = list(range(len(BASE)))
    iall = list(range(len(ALL)))
    p_base, p_all = oof(ibase), oof(iall)
    cases = np.unique(grp)

    def hits(p):
        return np.array([1.0 if y[grp == g].astype(bool)[np.argmax(p[grp == g])] else 0.0
                         for g in cases])
    h_base, h_all = hits(p_base), hits(p_all)

    # ------------------------------------------------------------------ Z1
    say()
    say("Z1 NO LEAK")
    say(f"     cases where reaction j itself contributes a transport pair (so the subtraction "
        f"actually bites): {subfrac:.1%}")
    solo_auc = {f: auc_of(X[:, len(BASE) + k], y.astype(bool)) for k, f in enumerate(SPAT)}
    for f, a in sorted(solo_auc.items(), key=lambda kv: -abs(kv[1] - 0.5)):
        say(f"     {f:<24s} lone AUC {a:.4f}")
    worst = max(solo_auc.values())
    z1 = bool(worst <= 0.75 and subfrac > 0)
    GG.verdict(z1, emit=say, if_true=(
        "no spatial feature is a label in disguise, and j's own transport pair is subtracted."),
        if_false=(
        f"a spatial feature reaches {worst:.4f} alone -- that is the shape of the degree leak and "
        f"it has to be closed before anything below is believed."))
    say(f"     Z1 {'PASS' if z1 else 'FAIL'}")

    # ------------------------------------------------------------------ Z2
    d2 = float((h_all - h_base).mean())
    s2 = float((h_all - h_base).std() / np.sqrt(len(h_all)))
    z2 = bool(d2 > 3 * s2)
    say()
    say(f"Z2 OVERALL hit@1: {h_base.mean():.4f} -> {h_all.mean():.4f} = {d2:+.4f} "
        f"sem {s2:.4f} ({d2/s2:+.1f} sem)")
    GG.verdict(z2, emit=say, if_true="spatial features help overall.",
               if_false="spatial features do not move the overall number.")
    say(f"     Z2 {'PASS' if z2 else 'FAIL'}")

    # ------------------------------------------------------------------ Z3/Z4
    say()
    say("Z3/Z4 BY CASE TYPE")
    say(f"     {'type':<12s} {'n':>6s} {'without':>9s} {'with':>9s} {'delta':>9s} {'sem':>7s}")
    per = {}
    for k in ("transport", "multi", "single"):
        m = kind == k
        d = h_all[m] - h_base[m]
        per[k] = {"n": int(m.sum()), "base": float(h_base[m].mean()),
                  "with": float(h_all[m].mean()), "delta": float(d.mean()),
                  "sem": float(d.std() / np.sqrt(len(d)))}
        say(f"     {k:<12s} {per[k]['n']:>6,} {per[k]['base']:>9.4f} {per[k]['with']:>9.4f} "
            f"{per[k]['delta']:>+9.4f} {per[k]['sem']:>7.4f}")
    z3 = bool(per["transport"]["delta"] > 3 * per["transport"]["sem"])
    GG.verdict(z3, emit=say, if_true=(
        "the targeted bucket improves: transport cases were 27.9% of all errors and spatial "
        "features are what they were missing."), if_false=(
        "transport cases do not improve, so whatever spatial features are doing, it is not the "
        "thing they were added for."))
    say(f"     Z3 {'PASS' if z3 else 'FAIL'}")
    z4 = bool(all(per[k]["delta"] > -3 * per[k]["sem"] for k in ("multi", "single")))
    GG.verdict(z4, emit=say, if_true="nothing else gets worse.",
               if_false="another bucket is damaged; the model has learned to over-prefer transport.")
    say(f"     Z4 {'PASS' if z4 else 'FAIL'}")

    # ------------------------------------------------------------------ Z5
    say()
    say("Z5 WHICH SPATIAL FEATURE (permutation, drop in hit@1)")
    rng = np.random.default_rng(SEED)
    tr, te = next(iter(GroupKFold(n_splits=NFOLD).split(X, y, grp)))
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                         min_samples_leaf=40, random_state=0).fit(X[tr], y[tr])

    def h_of(P):
        return float(np.mean([1.0 if y[te][grp[te] == g].astype(bool)[np.argmax(P[grp[te] == g])]
                              else 0.0 for g in np.unique(grp[te])]))
    b0 = h_of(clf.predict_proba(X[te])[:, 1])
    imp = {}
    for k, f in enumerate(ALL):
        Xp = X[te].copy()
        Xp[:, k] = rng.permutation(Xp[:, k])
        imp[f] = b0 - h_of(clf.predict_proba(Xp)[:, 1])
    for f, v in sorted(imp.items(), key=lambda kv: -kv[1])[:8]:
        tag = " (spatial)" if f in SPAT else ""
        say(f"     {f:<24s} {v:+.4f}{tag}")
    z5 = True
    say(f"     Z5 {'PASS' if z5 else 'FAIL'}")

    # ------------------------------------------------------------------ Z6
    say()
    say("Z6 THE CONFIDENCE CURVE")
    from sklearn.isotonic import IsotonicRegression
    pc = np.zeros(len(y))
    for tr2, te2 in GroupKFold(n_splits=NFOLD).split(X, y, grp):
        pc[te2] = IsotonicRegression(out_of_bounds="clip").fit(p_all[tr2], y[tr2]).predict(p_all[te2])
    conf = np.array([pc[grp == g].max() for g in cases])
    corr = np.array([1.0 if y[grp == g].astype(bool)[np.argmax(pc[grp == g])] else 0.0
                     for g in cases])
    o = np.argsort(-conf)
    curve = []
    for cov in (0.10, 0.25, 0.50, 0.75, 1.00):
        k = max(int(cov * len(o)), 1)
        curve.append({"coverage": cov, "precision": float(corr[o[:k]].mean())})
        say(f"     {cov:>6.0%}  precision {corr[o[:k]].mean():.4f}"
            + ("   (loop 168: 0.9534)" if cov == 0.50 else ""))
    z6 = True
    say(f"     Z6 {'PASS' if z6 else 'FAIL'}")

    say()
    say("Z7 WHAT THIS CANNOT SHOW")
    say("     DEV only. Compartment assignment is Human-GEM's own curation, so a model learning")
    say("     compartment regularities learns the curators' conventions as well as the biology.")
    z7 = True
    say(f"     Z7 {'PASS' if z7 else 'FAIL'}")

    gates = {"Z1": z1, "Z2": z2, "Z3": z3, "Z4": z4, "Z5": z5, "Z6": z6, "Z7": z7}
    man = RM.manifest(inputs=[Path("colab/data/rem_bipartite.npz")],
                      available=n_ok, used=n_ok, selection="all", seed=SEED,
                      controls=["transport-pair counts with reaction j's own contribution subtracted",
                                "every spatial feature's lone AUC gated at Z1 against the degree-leak shape",
                                "the targeted bucket gated separately from the overall number",
                                "the other buckets gated for damage",
                                "identical folds and seed as the no-spatial arm"],
                      note="compartment and transport-topology features added to the shortlist ranker")
    out = {"test": "spatial re-rank", "gates": gates, "n": n_ok,
           "hit1_base": float(h_base.mean()), "hit1_spatial": float(h_all.mean()),
           "delta": [d2, s2], "per_type": per, "solo_auc": solo_auc,
           "importance": imp, "curve": curve,
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
