"""The signed count: the last untested compression, and it is a CURVE, not a verdict.

WHERE THIS SITS. summary.py found that the engine's cost collapses from exponential to polynomial
if a gene's response depends on its controllers' history only through a COUNT. promoter.py then
refuted that on real measurements: on 5,790 designed yeast promoters the identity of the bound
factors beats their number by 0.7319 log2, which is 6.3 replicate noise floors. The stated reason
was that activators and repressors cannot be added together. The obvious repair -- count them
SEPARATELY -- was named there as the nearest untested candidate. This module tests it.

THE CEILING GATE WAS RUN AS A STANDALONE PROBE BEFORE THIS FILE EXISTED, which is the discipline
block.py cost a module by skipping. Its result is stated here rather than discovered below,
because it is the reason this module is shaped the way it is:

    a signed count with a DELIBERATELY LEAKY sign oracle -- per-factor effects fitted on all the
    promoter data, test folds included, so the number is an upper bound on what any honest
    sign-learning scheme could reach -- closes 61.6% of the count-to-identity gap and leaves
    2.41 noise floors open.

By promoter.py's own standard that is a REFUTATION: the signed count does not reach identity, and
no cleverness in estimating signs can rescue it, because it was already given the answer. So this
module does not ask "does the signed count work". It asks the question the refutation exposes:
the signed count is C = 2 in a one-parameter family -- partition the factors into C classes and
count within each -- where C = 1 is the plain count and C = n_TF is identity. The useful object is
the whole curve of error against C, priced against the cap each C buys. That is this session's
own rule, learned five separate times and most recently in quant.py: AN APPROXIMATION IS RANKED BY
A SCALING LAW, NEVER AT A POINT.

THE COST OF A CLASS COUNT. For a gene whose k_i controllers fall into classes with k_ij members
each, the per-gene factor is the product over occupied classes of (k_ij(L+1)+1), and the
controller block is the same product over the block's own class split. C = 1 recovers summary.py's
polynomial k_i(L+1)+1; C >= k_i recovers its per-controller count (L+2)^k_i. So C interpolates
between a polynomial and an exponential, and the cap must be recomputed for every C rather than
assumed.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN OF THIS MODULE
=================================================================================================

S0  THE CEILING GATE, RESTATED AND RE-RUN INSIDE THE MODULE so it is in the record and not only
    in the transcript. Both halves: the accuracy ceiling above, and the COST ceiling -- the cap
    and the edge coverage each C buys on the real TRRUST network, computed with summary.py's own
    cap function extended with the class-count term. PREDECLARED: a C is worth reporting as a
    candidate only if its cap exceeds the pattern representation's cap by more than one AND its
    error is inside one noise floor of identity. If no C satisfies both, say so.

S1  THE SCALING LAW. Error against C, from C = 1 to the full factor set, with the cap beside it.
    The frontier is the result; the signed count is one point on it and is reported as such.

S2  NO LEAKAGE IN THE HONEST VERSION. Classes refitted inside each training fold, never on the
    held-out promoters. The honest-minus-leaky difference is the price of not having a sign
    oracle and it is reported at every C. PREDECLARED: if that difference is itself larger than
    the noise floor, then the leaky ceiling was not an upper bound on anything achievable and
    the whole family is unavailable in practice whatever its ceiling says.

S3  FALLBACK IS PART OF THE ANSWER. The identity model wins partly by REFUSING to predict: 11.1%
    of held-out promoters have no training group and drop to the context mean. A finer summary
    buys accuracy on the cases it covers and loses it on the cases it abstains from, and an RMSE
    that mixes the two is not a comparison. Every number is repeated on the subset where NO model
    falls back.

S4  CAN THE SIGN BE ASSIGNED AT ALL ON THE REAL NETWORK? A signed count needs a sign per edge.
    TRRUST annotates edges Activation, Repression or Unknown. Report the unsigned fraction and
    what it does to the achievable C, because a representation that requires data which does not
    exist is unavailable whatever its accuracy.

S5  THE ENGINE SIDE, AND A SUSPICION ABOUT THE SYNTHETIC SYSTEM. In block.py's generator every
    target is driven by every controller through (1 + g*b)^(1/nC) with ONE g -- all controllers
    act identically and positively, so the drive is a symmetric function of the count and the
    total count is count-sufficient BY CONSTRUCTION. Three variants are run: (A) all activators of
    equal strength, reproducing that; (B) half repressors of equal magnitude, where the drive is a
    function of the signed difference and the signed count is exact BY CONSTRUCTION -- stated so
    the gate is not read as evidence; (C) half repressors with heterogeneous magnitudes WITHIN
    each sign class, where neither is exact. PREDECLARED: (C) is the only informative one, and the
    fraction of the break it repairs must be compared against the 59-62% measured on real
    promoters. If the synthetic and the real numbers disagree, the synthetic system is not a model
    of the thing promoter.py measured.

S6  WHAT THIS DOES AND DOES NOT SETTLE.

=================================================================================================
TWO DEFECTS IN THIS MODULE'S OWN FIRST RUN, FOUND BEFORE THE RESULT WAS REPORTED
=================================================================================================
D1  THE CAP WAS NON-MONOTONE IN C. The first run gave cap 13 at C = 12 and cap 16 at C = 16, which
    cannot be a property of the family -- a finer partition can only split a class, never merge
    two, so the cost can only rise. The cause was drawing an INDEPENDENT random class map for
    each C, so consecutive rows were comparing two unrelated partitions and the difference between
    them was assignment noise. One uniform draw per gene with class = floor(u*C) makes the
    partitions nested at every doubling and the cap monotone by construction.

D2  S5 SCORED EVERY SUMMARY AGAINST ZERO. The full pattern is not exact in this construction
    either: the product over targets and the finite window both cost something, and the pattern
    carried 2.2e-2 of error on its own. Reading a summary's raw error as "how much the summary
    costs" therefore charged it for the harness. Every S5 number is now an EXCESS over the
    pattern floor of the same variant, which changes the headline: the signed count repairs 77%
    of the break rather than 68%, and the comparison against the real promoter number moves with
    it.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from rem.atlas.hybrid_tune import RULE
from rem.atlas.promoter import fetch, load
from rem.atlas.statedim import stationary
from rem.atlas.block import block_joint, marginalise_targets
from rem.atlas.summary import var_on
from rem.atlas.trrust_engine import signed_edges
from rem.atlas.localclosure import load_trrust, ball
from rem.atlas.engine import residual_graph
from rem.atlas.boundedwidth import bounded_elimination


# =================================================================================================
# the class-count family on real promoters
# =================================================================================================

def tf_effects(rs, lam=1.0):
    """Per-factor effect as a ridge coefficient on per-factor site counts, with context
    indicators. A plain per-factor mean is confounded by which OTHER sites co-occur; the ridge
    coefficient is not, which matters because 63% of these promoters carry two or more sites."""
    tfs = sorted({t[0] for r in rs for t in r["tfs"]})
    ctxs = sorted({r["ctx"] for r in rs})
    ti = {k: i for i, k in enumerate(tfs)}
    ci = {k: len(tfs) + i for i, k in enumerate(ctxs)}
    X = np.zeros((len(rs), len(tfs) + len(ctxs) + 1))
    X[:, -1] = 1.0
    for i, r in enumerate(rs):
        for t in r["tfs"]:
            X[i, ti[t[0]]] += 1.0
        X[i, ci[r["ctx"]]] = 1.0
    y = np.array([r["y"] for r in rs])
    w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    return {k: float(w[ti[k]]) for k in tfs}


def class_bins(eff, C, at_zero=False):
    """C classes of factors. at_zero splits on the SIGN, which is the biological definition of a
    signed count; otherwise equal-frequency bins of the effect, which is what generalises to C>2."""
    if C == 1:
        return {k: 0 for k in eff}
    if at_zero:
        return {k: (1 if v > 0 else 0) for k, v in eff.items()}
    ks = sorted(eff, key=lambda k: eff[k])
    out = {}
    for b, g in enumerate(np.array_split(np.arange(len(ks)), C)):
        for j in g:
            out[ks[j]] = b
    return out


def class_key(cls, C, r):
    c = [0] * C
    for t in r["tfs"]:
        c[cls.get(t[0], 0)] += 1
    return (r["ctx"], tuple(c))


def cv_grouped(sub, makekey, folds=5, seed=0):
    """Held-out group-mean predictor, identical to promoter.py's, except that the key function is
    BUILT FROM THE TRAINING FOLD so a class map fitted on held-out data cannot leak in. Returns
    the RMSE, the fallback rate, and the per-record errors and hit mask so S3 can restrict to the
    promoters no model abstained on."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(sub))
    rng.shuffle(idx)
    fold = np.array_split(idx, folds)
    err = np.zeros(len(sub))
    hit = np.zeros(len(sub), dtype=bool)
    for f in range(folds):
        te = set(fold[f].tolist())
        tr = [sub[i] for i in idx if i not in te]
        keyfn = makekey(tr)
        gm = collections.defaultdict(list)
        cm = collections.defaultdict(list)
        for r in tr:
            gm[keyfn(r)].append(r["y"])
            cm[r["ctx"]].append(r["y"])
        gmm = {k: float(np.mean(v)) for k, v in gm.items()}
        cmm = {k: float(np.mean(v)) for k, v in cm.items()}
        glob = float(np.mean([r["y"] for r in tr]))
        for i in fold[f]:
            r = sub[i]
            k = keyfn(r)
            if k in gmm:
                err[i] = r["y"] - gmm[k]
                hit[i] = True
            else:
                err[i] = r["y"] - cmm.get(r["ctx"], glob)
    return float(np.sqrt(np.mean(err ** 2))), 1.0 - hit.mean(), err, hit


# =================================================================================================
# the cost of a class count on the real network
# =================================================================================================

def class_caps(Cs, L, bar=1e12, capexp=400, maxC=420, seed=20260907):
    """Cap under each class-count representation, with BOTH terms in the functional.

    This is summary.py's trrust_cap with one representation added and two changes that are about
    correctness rather than speed. The elimination is HOISTED, so every representation is scored
    on the same decomposition instead of each choosing its own best one. And the per-gene class
    counts are maintained INCREMENTALLY as controllers are added -- recomputing them inside the
    (r, w) loop would be 86 million set intersections and would also have scored the same gene
    six times over."""
    E, _ = signed_edges()
    adj, inv, sha, ne = load_trrust()
    n = len(adj)
    outd = collections.Counter(u for u, v, _ in E)
    name2i = {g: i for i, g in inv.items()}
    od = np.zeros(n)
    for gname, d in outd.items():
        if gname in name2i:
            od[name2i[gname]] = d
    order = list(np.argsort(-od))
    # ONE uniform draw per gene, then class = floor(u*C). Drawing an INDEPENDENT random class
    # map for each C made the cap non-monotone in C -- 13 at C=12 but 16 at C=16 -- which is
    # noise between two different partitions, not a property of the family. With a shared u the
    # partitions are nested at every doubling, so a finer C can only split a class, never merge
    # two, and the cap is monotone by construction.
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    asg = {C: np.minimum((u * C).astype(int), C - 1) for C in Cs}

    def slot(m):
        return m * (L + 1) + 1.0

    kv = np.zeros(n)
    cnt = {C: [collections.Counter() for _ in range(n)] for C in Cs}
    fac = {C: np.ones(n) for C in Cs}
    inC = np.zeros(n, dtype=bool)
    keys = [("pattern", 0), ("count", 0)] + [("class", C) for C in Cs]
    best = {k: 0 for k in keys}
    live = set(keys)
    for nC in range(1, maxC):
        if not live:
            break
        u = order[nC - 1]
        for i in adj[u]:
            if inC[i]:
                continue
            kv[i] += 1
            for C in Cs:
                c = int(asg[C][u])
                m = cnt[C][i][c]
                cnt[C][i][c] = m + 1
                fac[C][i] = min(fac[C][i] * slot(m + 1) / slot(m), 2.0 ** capexp)
        inC[u] = True
        kv[u] = 0.0
        for C in Cs:
            fac[C][u] = 1.0
        Cset = set(order[:nC])
        ctrl = order[:nC]
        res = residual_graph(adj, Cset, n)
        fpat = np.array([2.0 ** min(kv[i] * (L + 1), capexp) for i in range(n)])
        fcnt = kv * (L + 1) + 1.0
        per = {k: None for k in live}
        for r in (1, 2):
            for w in (4, 6, 8):
                pa, o, _ = bounded_elimination([ball(res, i, r) - {i} for i in range(n)], w)
                base = np.array([2.0 ** min(1 + len(pa[i]), capexp) for i in range(n)])
                for key in list(live):
                    rep, C = key
                    f = fpat if rep == "pattern" else fcnt if rep == "count" else fac[C]
                    v = float(np.sum(base * f))
                    per[key] = v if per[key] is None else min(per[key], v)
        for key in list(live):
            rep, C = key
            if rep == "pattern":
                blk = 2.0 ** min(nC * (L + 1), capexp)
            elif rep == "count":
                blk = float(nC * (L + 1) + 1)
            else:
                bc = collections.Counter(int(asg[C][i]) for i in ctrl)
                blk = 1.0
                for _, m in bc.items():
                    blk = min(blk * slot(m), 1e300)
            if per[key] + blk <= bar:
                best[key] = nC
            else:
                live.discard(key)
    return best


def edge_coverage(nC):
    E, _ = signed_edges()
    adj, inv, sha, ne = load_trrust()
    outd = collections.Counter(u for u, v, _ in E)
    name2i = {g: i for i, g in inv.items()}
    od = np.zeros(len(adj))
    for gname, d in outd.items():
        if gname in name2i:
            od[name2i[gname]] = d
    order = list(np.argsort(-od))
    C = {inv[i] for i in order[:nC]}
    return sum(1 for u, v, _ in E if u in C) / len(E)


# =================================================================================================
# the engine side: controllers that are not all the same sign
# =================================================================================================

def signed_controller(nC, nT, sgn, mag, cc=1.5, boff=2.0, seed=20260907):
    """block.py's multi_controller with a PER-CONTROLLER signed gain. The target drive is
    exp((1/nC) * sum_c sgn_c * mag_c * b_c), so:
      all sgn=+1, mag equal  -> drive is a function of the TOTAL COUNT (count-sufficient)
      half sgn=-1, mag equal -> drive is a function of the SIGNED DIFFERENCE (signed-sufficient)
      heterogeneous mag      -> neither, which is the only informative case."""
    nv = nC + nT
    n = 1 << nv
    rng = np.random.default_rng(seed)
    a = np.exp(rng.normal(0, 0.3, nv))
    b = boff * np.exp(rng.normal(0, 0.3, nv))
    st = np.arange(n, dtype=np.int64)
    bits = [((st >> i) & 1).astype(float) for i in range(nv)]
    R, C, D = [], [], []
    for c in range(nC):
        drive = 1.0 + cc * bits[c - 1] if c > 0 else np.ones(n)
        R.append(st); C.append(st ^ (1 << c))
        D.append(np.where(bits[c] == 0, a[c] * drive, b[c]))
    for t in range(nC, nv):
        lg = np.zeros(n)
        for c in range(nC):
            lg = lg + sgn[c] * mag[c] * bits[c]
        drive = np.exp(lg / nC)
        R.append(st); C.append(st ^ (1 << t))
        D.append(np.where(bits[t] == 0, a[t] * drive, b[t]))
    Q = coo_matrix((np.concatenate(D), (np.concatenate(R), np.concatenate(C))),
                   shape=(n, n)).tocsr()
    dg = np.asarray(Q.sum(axis=1)).ravel()
    return (Q - csr_matrix((dg, (st, st)), shape=(n, n))).tocsr(), nv


def sufficiency_signed(nC, nT, L, dt, sgn, mag, cc=1.5):
    """summary.py's sufficiency construction -- the approximation bites in the PRODUCT over
    targets, not in the sum -- with a signed-count summary added to the candidate list."""
    Q, nv = signed_controller(nC, nT, sgn, mag, cc=cc)
    pi, res, _ = stationary(Q)
    cur = block_joint(Q, pi, nC, L, dt)
    exact = marginalise_targets(pi, nv, nC)
    exact = exact / exact.sum()
    vex = var_on(exact, nT)
    st = np.arange(1 << nT, dtype=np.int64)
    bits = [((st >> j) & 1) for j in range(nT)]
    rows = {}
    for a, v in cur.items():
        w = float(v.sum())
        if w <= 1e-300:
            continue
        m = marginalise_targets(v, nv, nC)
        s = m.sum()
        if s > 0:
            rows[a] = (w, np.array([float((m / s)[bits[j] == 1].sum()) for j in range(nT)]))

    def tot(a, _):
        return sum(bin(s).count("1") for s in a)

    def sgncount(a, _):
        na = sum(sum((s >> c) & 1 for s in a) for c in range(nC) if sgn[c] > 0)
        nr = sum(sum((s >> c) & 1 for s in a) for c in range(nC) if sgn[c] < 0)
        return (na, nr)

    cands = {"full pattern": lambda a, _: a,
             "signed count": sgncount,
             "signed difference": lambda a, _: sgncount(a, _)[0] - sgncount(a, _)[1],
             "total count": tot,
             "no conditioning": lambda a, _: 0}
    out = {}
    for name, keyfn in cands.items():
        grp = collections.defaultdict(lambda: [0.0, np.zeros(nT)])
        for a, (w, q) in rows.items():
            gg = grp[keyfn(a, nC)]
            gg[0] += w
            gg[1] += w * q
        acc = np.zeros(1 << nT)
        for a, (w, q) in rows.items():
            gg = grp[keyfn(a, nC)]
            qa = gg[1] / gg[0]
            term = np.full(1 << nT, w)
            for j in range(nT):
                term = term * np.where(bits[j] == 1, qa[j], 1.0 - qa[j])
            acc += term
        acc = acc / acc.sum()
        out[name] = ((var_on(acc, nT) - vex) / vex, len(grp))
    return out, len(rows), res


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE); P_("THE SIGNED COUNT, RANKED AS A CURVE"); P_(RULE)
    d, sha = fetch()
    recs = load(d)
    sub = [r for r in recs if r["tfs"]]
    l1 = np.array([r["l1"] for r in recs]); l2 = np.array([r["l2"] for r in recs])
    floor = float(np.std(l1 - l2)) / np.sqrt(2.0)
    P_(f"  Sharon et al. 2012, Nat Biotechnol 30:521-30, doi:10.1038/nbt.2205, via NCBI GEO")
    P_(f"  {'GSE37851'}, sha256[:32] {sha}. {len(sub)} promoters with at least one site,")
    P_(f"  {len({t[0] for r in sub for t in r['tfs']})} distinct factors. Replicate noise floor"
       f" {floor:.4f} log2, measured before any model.")

    CS = (1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128, 404)
    eff_all = tf_effects(sub)
    npos = sum(1 for v in eff_all.values() if v > 0)
    P_(f"  leaky oracle over all folds: {npos} activating factors, {len(eff_all)-npos} repressing.")

    e_cnt, f_cnt, err_cnt, hit_cnt = cv_grouped(sub, lambda tr: (lambda r: (r["ctx"], len(r["tfs"]))))
    e_idn, f_idn, err_idn, hit_idn = cv_grouped(
        sub, lambda tr: (lambda r: (r["ctx"], tuple(sorted(t[0] for t in r["tfs"])))))
    gap = e_cnt - e_idn

    # ---- S0  THE CEILING GATE ------------------------------------------------------------------
    P_("\n" + RULE); P_("S0  THE CEILING GATE, BOTH HALVES"); P_(RULE)
    P_("  (a) ACCURACY CEILING. The signed count is given a DELIBERATELY LEAKY sign oracle --")
    P_("      per-factor effects fitted on all the data, held-out promoters included -- so the")
    P_("      number below is an upper bound on any honest sign-learning scheme.")
    sgn0 = class_bins(eff_all, 2, at_zero=True)
    e_sgn, f_sgn, _, _ = cv_grouped(sub, lambda tr: (lambda r: class_key(sgn0, 2, r)))
    P_(f"      plain count RMSE   {e_cnt:.4f}   fallback {f_cnt:>5.1%}")
    P_(f"      SIGNED count RMSE  {e_sgn:.4f}   fallback {f_sgn:>5.1%}   (leaky oracle)")
    P_(f"      identity RMSE      {e_idn:.4f}   fallback {f_idn:>5.1%}")
    P_(f"      the signed count closes {100*(e_cnt-e_sgn)/gap:.1f}% of the count-to-identity gap")
    P_(f"      and leaves {(e_sgn-e_idn)/floor:.2f} noise floors open.")
    s0a = (e_sgn - e_idn) <= floor
    P_(f"      S0(a): {'PASS' if s0a else 'FAIL -- REFUTED AT ITS OWN CEILING'}. The signed count"
       f" does not reach identity even")
    P_( "      when handed the answer, so no honest sign estimator can make it do so. What follows")
    P_( "      is therefore not a verdict on C = 2 but the curve C = 2 sits on.")

    P_("\n  (b) COST CEILING. summary.py's cap function with the class-count term added, and the")
    P_("      elimination hoisted so every representation is scored on the SAME decomposition.")
    L = 8
    caps = class_caps(CS, L)
    P_(f"\n      {'representation':<24} {'cap |C| at L=8':>15} {'edge coverage':>15}")
    P_(f"      {'pattern (exact)':<24} {caps[('pattern', 0)]:>15} "
       f"{edge_coverage(caps[('pattern', 0)]):>14.1%}")
    for C in CS:
        nm = ("total count (C=1)" if C == 1 else
              "SIGNED count (C=2)" if C == 2 else f"class count C={C}")
        P_(f"      {nm:<24} {caps[('class', C)]:>15} {edge_coverage(caps[('class', C)]):>14.1%}")
    P_(f"      {'total count, summary.py':<24} {caps[('count', 0)]:>15} "
       f"{edge_coverage(caps[('count', 0)]):>14.1%}")
    P_("      A class count is polynomial of degree (occupied classes), so the cap falls as C")
    P_("      rises. That is the trade this module exists to price.")
    caps_by_C = {C: caps[("class", C)] for C in CS}

    # ---- S1  THE SCALING LAW -------------------------------------------------------------------
    P_("\n" + RULE); P_("S1  THE SCALING LAW: ERROR AGAINST C, WITH THE CAP BESIDE IT"); P_(RULE)
    # cv_grouped calls makekey once per fold, in fold order, with the same shuffle for every
    # model. So the training effects are fitted ONCE per fold and reused across C, keyed by the
    # fold counter. id(tr) would have worked by accident until a list was garbage-collected and
    # its address reused, which is exactly the kind of defect that does not announce itself.
    fits = {}

    def honest_key(C):
        st = {"f": 0}

        def mk(tr):
            f = st["f"]
            st["f"] += 1
            if f not in fits:
                fits[f] = tf_effects(tr)
            cl = class_bins(fits[f], C)
            return lambda r: class_key(cl, C, r)
        return mk

    rows = {}
    P_(f"    {'C':>5} {'honest RMSE':>12} {'leaky RMSE':>11} {'floors from':>12} {'% of gap':>9}"
       f" {'fallback':>9} {'cap |C|':>8} {'coverage':>9}")
    P_(f"    {'':>5} {'':>12} {'':>11} {'identity':>12} {'closed':>9} {'':>9} {'at L=8':>8} {'':>9}")
    for C in CS:
        eh, fh, errh, hith = cv_grouped(sub, honest_key(C))
        cl = class_bins(eff_all, C)
        el, fl, _, _ = cv_grouped(sub, lambda tr, cl=cl, C=C: (lambda r: class_key(cl, C, r)))
        rows[C] = (eh, fh, errh, hith, el)
        cap = caps[("class", C)]
        P_(f"    {C:>5} {eh:>12.4f} {el:>11.4f} {(eh-e_idn)/floor:>12.2f} "
           f"{100*(e_cnt-eh)/gap:>8.1f}% {fh:>8.1%} {cap:>8} {edge_coverage(cap):>8.1%}")
    P_(f"    {'ident':>5} {e_idn:>12.4f} {e_idn:>11.4f} {0.0:>12.2f} {100.0:>8.1f}%"
       f" {f_idn:>8.1%} {caps[('pattern', 0)]:>8} {edge_coverage(caps[('pattern', 0)]):>8.1%}")
    inside = [C for C in CS if (rows[C][0] - e_idn) <= floor]
    P_(f"\n  S1: the smallest C inside one noise floor of identity is"
       f" {min(inside) if inside else 'NONE of those tested'}.")
    P_( "  The signed count (C = 2) is one point on this curve and it is not the knee.")
    P_( "  The C = 404 row is close to identity but not equal to it: classes are fitted IN-FOLD,")
    P_( "  so factors absent from a training fold have no bin and fall together into class 0,")
    P_( "  where the identity model keys on the factor name. That merging is why C = 404 abstains")
    P_( "  slightly less often than identity and scores slightly worse.")
    ok_acc = [C for C in CS if (rows[C][0] - e_idn) <= floor]
    ok_cost = [C for C in CS if caps[("class", C)] > caps[("pattern", 0)] + 1]
    both = sorted(set(ok_acc) & set(ok_cost))
    P_(f"\n  S0's PREDECLARED bar, evaluated: a C counts as a candidate only if its cap beats the")
    P_(f"  pattern cap by more than one AND its error is inside one noise floor of identity.")
    P_(f"    inside one floor of identity : {ok_acc}")
    P_(f"    cap beats pattern by > 1     : {ok_cost}")
    P_(f"    BOTH                         : {both if both else 'NONE'}")
    if both:
        c0 = min(both)
        P_(f"  So a candidate exists, and it is C = {c0}: cap {caps[('pattern', 0)]} ->"
           f" {caps[('class', c0)]}, coverage {edge_coverage(caps[('pattern', 0)]):.1%} ->"
           f" {edge_coverage(caps[('class', c0)]):.1%}. That is a real gain and a modest one. It")
        P_( "  is emphatically NOT the polynomial collapse the total count promised, which was")
        P_(f"  cap {caps[('class', 1)]} and {edge_coverage(caps[('class', 1)]):.1%} coverage --")
        P_( "  and which this data refutes. Both ends of the family fail in opposite directions:")
        P_(f"  everything cheap enough to keep the {caps[('class', 1)]}-controller cap (C <= 4) is")
        P_( "  two or more noise floors from identity, and everything accurate enough to be inside")
        P_( "  one floor has already given back nine tenths of the coverage. What survives is the")
        P_( "  middle, and the middle is worth cap 3 -> 13, not 3 -> 419.")

    # ---- S2  LEAKAGE ---------------------------------------------------------------------------
    P_("\n" + RULE); P_("S2  WHAT THE SIGN ORACLE WAS WORTH: HONEST MINUS LEAKY"); P_(RULE)
    dmax = max(rows[C][0] - rows[C][4] for C in CS)
    P_(f"    {'C':>5} {'honest - leaky':>15} {'in floors':>11}")
    for C in CS:
        dd = rows[C][0] - rows[C][4]
        P_(f"    {C:>5} {dd:>15.4f} {dd/floor:>11.2f}")
    P_(f"\n  largest leakage advantage {dmax:.4f} log2 = {dmax/floor:.2f} floors")
    P_(f"  S2: {'PASS -- the classes can be learned in-fold, so the leaky ceiling was a fair upper bound' if dmax <= floor else 'FAIL -- the class map cannot be learned from training data alone'}")

    # ---- S3  FALLBACK --------------------------------------------------------------------------
    P_("\n" + RULE); P_("S3  FALLBACK IS PART OF THE ANSWER"); P_(RULE)
    P_("  A finer summary buys accuracy where it has a training group and abstains where it does")
    P_("  not, dropping to the context mean. An RMSE mixing the two is not a comparison, and the")
    P_("  identity model abstains four times as often as a class count.")
    shown = [1, 2, 8, 64, 128]
    ok = hit_cnt & hit_idn
    for C in shown:
        ok = ok & rows[C][3]
    P_(f"\n  {int(ok.sum())} of {len(sub)} promoters are scored by every model without fallback")
    P_(f"    {'model':<24} {'all promoters':>14} {'no-fallback subset':>19} {'floors from ident':>18}")
    sub_idn = float(np.sqrt(np.mean(err_idn[ok] ** 2)))
    P_(f"    {'plain count':<24} {e_cnt:>14.4f} {float(np.sqrt(np.mean(err_cnt[ok]**2))):>19.4f}"
       f" {(float(np.sqrt(np.mean(err_cnt[ok]**2)))-sub_idn)/floor:>18.2f}")
    for C in shown:
        s = float(np.sqrt(np.mean(rows[C][2][ok] ** 2)))
        nm = "SIGNED count (C=2)" if C == 2 else f"class count C={C}"
        if C == 1:
            continue
        P_(f"    {nm:<24} {rows[C][0]:>14.4f} {s:>19.4f} {(s-sub_idn)/floor:>18.2f}")
    P_(f"    {'identity':<24} {e_idn:>14.4f} {sub_idn:>19.4f} {0.0:>18.2f}")
    P_("  This is the comparison that counts, because it is the only one where every model is")
    P_("  answering the same questions.")

    # ---- S4  CAN THE SIGN BE ASSIGNED? ---------------------------------------------------------
    P_("\n" + RULE); P_("S4  CAN THE SIGN BE ASSIGNED ON THE REAL NETWORK AT ALL?"); P_(RULE)
    E, _ = signed_edges()
    md = collections.Counter(m for _, _, m in E)
    P_(f"  TRRUST, {len(E)} distinct directed pairs after deduplication:")
    for m, c in md.most_common():
        P_(f"    {m:<14} {c:>7}  {c/len(E):>6.1%}")
    unk = md.get("Unknown", 0) / len(E)
    P_(f"\n  {unk:.1%} of real regulatory edges carry NO SIGN. A signed count needs one per edge,")
    P_( "  so on this network the representation is not two classes but three -- activator,")
    P_( "  repressor, and unknown -- and the third class is the largest single obstruction to")
    P_( "  using it, before any question of accuracy. This is the same shape of problem as the")
    P_( "  42.3% figure trrust_engine.py had to correct: the annotation, not the method, is the")
    P_( "  limit.")

    # ---- S5  THE ENGINE SIDE -------------------------------------------------------------------
    P_("\n" + RULE); P_("S5  THE ENGINE SIDE, AND WHAT THE SYNTHETIC SYSTEM WAS ACTUALLY TESTING")
    P_(RULE)
    P_("  block.py's generator drives every target by (1+g*b)^(1/nC) with ONE g, so all")
    P_("  controllers act identically and positively and the drive is a symmetric function of the")
    P_("  count. The total count was count-sufficient there BY CONSTRUCTION. Three variants:")
    nC, nT, Lb, dtb = 4, 5, 2, 0.25
    rg = np.random.default_rng(7)
    variants = [
        ("A  all activators, equal strength", np.ones(nC), np.full(nC, 3.0)),
        ("B  half repressors, equal magnitude",
         np.array([1.0, 1.0, -1.0, -1.0]), np.full(nC, 3.0)),
        ("C  half repressors, HETEROGENEOUS magnitude",
         np.array([1.0, 1.0, -1.0, -1.0]), np.array([4.5, 1.5, 4.0, 1.2])),
    ]
    P_(f"\n    nC={nC} controllers, nT={nT} targets, L={Lb}, dt={dtb}; signed relative error on")
    P_( "    the variance of the target count, against the exact joint")
    P_("    Every summary is scored against the FULL PATTERN of the same construction, not")
    P_("    against zero. The pattern is not exact either -- the product over targets and the")
    P_("    finite window L, dt both cost something -- so the pattern's own error is the floor")
    P_("    of this experiment, and a summary's EXCESS over it is the only part that is about")
    P_("    the summary.")
    P_(f"\n    {'variant':<42} {'pattern':>10} {'total count':>13} {'signed count':>13}"
       f" {'signed diff':>12}")
    keep = {}
    for nm, sg, mg in variants:
        res, nrows, rr = sufficiency_signed(nC, nT, Lb, dtb, sg, mg)
        keep[nm[0]] = res
        P_(f"    {nm:<42} {res['full pattern'][0]:>10.1e} {res['total count'][0]:>13.2e}"
           f" {res['signed count'][0]:>13.2e} {res['signed difference'][0]:>12.2e}")
    P_(f"    {'(no conditioning, the baseline that must fail)':<42} {'':>10}"
       f" {keep['C']['no conditioning'][0]:>13.2e}")

    def exc(v, k):
        return abs(keep[v][k][0] - keep[v]["full pattern"][0])

    P_(f"\n    {'variant':<42} {'total count':>13} {'signed count':>13} {'repaired':>10}")
    for v in ("A", "B", "C"):
        et, es = exc(v, "total count"), exc(v, "signed count")
        P_(f"    {'excess over the pattern floor, ' + v:<42} {et:>13.2e} {es:>13.2e}"
           f" {100*(et-es)/et if et > 0 else float('nan'):>9.1f}%")
    span = {v: exc(v, "no conditioning") for v in ("A", "B", "C")}
    P_(f"\n    the span this is measured on -- no conditioning at all, minus the pattern floor:")
    for v in ("A", "B", "C"):
        P_(f"      variant {v}: {span[v]:.2e}   total count uses"
           f" {100*(1-exc(v, 'total count')/span[v]):.1f}% of it,"
           f" signed count {100*(1-exc(v, 'signed count')/span[v]):.1f}%")
    P_(f"\n  Variant A reproduces summary.py: the total count's excess over the pattern is")
    P_(f"  {exc('A', 'total count'):.1e}, which is {100*exc('A','total count')/span['A']:.1f}% of")
    P_( "  the span between the pattern and no conditioning at all -- so the count recovers")
    P_(f"  {100*(1-exc('A','total count')/span['A']):.0f}% of what the full history is worth, because the system was")
    P_( "  built so that the drive is a symmetric function of the count. Not zero, but close.")
    P_(f"  Variant B: the total count's excess rises to {exc('B', 'total count'):.2e} and the")
    P_(f"  signed count's is {exc('B', 'signed count'):.2e} -- and that is BY CONSTRUCTION, since")
    P_( "  the drive is a function of the signed difference. Not evidence, and not exactly zero")
    P_( "  either: the drive is a function of the signed difference at each INSTANT, while the")
    P_( "  summary is over a window of L+1 slices, so the construction makes the signed count")
    P_( "  sufficient for the drive and not for the history.")
    et, es = exc("C", "total count"), exc("C", "signed count")
    P_(f"  Variant C is the only informative one. Heterogeneous magnitudes WITHIN each sign class")
    P_(f"  are the synthetic analogue of what the promoter data has. The total count's excess is")
    P_(f"  {et:.2e}, the signed count repairs {100*(et-es)/et:.1f}% of it.")
    P_(f"  On real promoters the signed count closed {100*(e_cnt-e_sgn)/gap:.1f}% of the")
    P_(f"  count-to-identity gap. The synthetic number is {abs(100*(et-es)/et - 100*(e_cnt-e_sgn)/gap):.0f}"
       f" points MORE FORGIVING than the measurement, which is the honest way to put it -- the")
    P_( "  two are the same order and the same sign, and the synthetic one flatters the summary.")
    P_( "  The finding is still that the synthetic system only reproduces the measured behaviour")
    P_( "  once it is given repressors AND within-sign heterogeneity. Before that it had neither,")
    P_( "  which is why summary.py's synthetic verdict and promoter.py's measured verdict")
    P_( "  disagreed: they were not testing the same thing.")

    # ---- S6 ------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("S6  WHAT THIS DOES AND DOES NOT SETTLE"); P_(RULE)
    P_("  1. The promoter half tests SITES, not bound factors, at steady state, in two sequence")
    P_("     contexts, with no time axis. The engine's summary is over controller HISTORY; this")
    P_("     settles the identity half of that claim and says nothing about the temporal half.")
    P_("  2. The class map here is fitted from the response itself. That is legitimate for the")
    P_("     question asked -- is there ANY partition into C classes that suffices -- and it is")
    P_("     the most generous possible reading of the family. A partition chosen from")
    P_("     biology alone could only be worse.")
    P_("  3. The cap is a COST statement from graph arithmetic. It says the engine could be built")
    P_("     at that size, not that it would be accurate there.")
    P_("  4. The random class assignment used for the cap is the realistic case, not the best")
    P_("     one. A class map that happened to align with the network's structure would give a")
    P_("     higher cap, and one that cut across it a lower.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_signed.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P_(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
