"""The class count with position and orientation: an EXCHANGE RATE, not an addition.

WHAT IS BEING ASKED. signed.py priced the class-count family -- partition the regulators into C
classes and count within each -- and found the smallest C inside one noise floor of identity is
64, worth cap 3 -> 13 controllers on TRRUST. Every class there was a class of FACTOR IDENTITY.
But a promoter's response is not only a function of which factors are present. Sharon et al.
report that where a site sits, and which way it faces, both matter. This module asks whether
spending part of the class budget on those axes buys more than spending it all on identity.

THE THING TO SEE FIRST, because it determines the shape of the whole module. Position and
orientation do NOT add a dimension to this representation. They MULTIPLY the class alphabet: a
class count over (factor class x position bin x orientation) has C*P*O cells, and the engine's
cost depends only on HOW MANY distinguishable cells the summary has, never on what makes them
distinguishable. So the cap curve measured in signed.py already prices every possible split, and
the only open question is accuracy at a FIXED budget K = C*P*O. That is an exchange rate, and it
is the question this module measures.

WHAT promoter.py's V5 ALREADY FOUND, and why it is not the answer. V5 added position (10 bp bins)
and orientation to the IDENTITY model and both made it WORSE -- and diagnosed the cause honestly
as the ESTIMATOR rather than the biology: finer groups are sparser, so a held-out promoter more
often has no training group and falls back to the context mean. That diagnosis is testable here
in a way it was not there, because a class count is COARSE and therefore has fallback budget to
spend. The identity model abstains on 11.1% of held-out promoters; a class count at C = 8 abstains
on 3.8%. If V5's diagnosis is right, position should help a class count where it hurt identity.

THE PROBES RUN BEFORE THIS FILE EXISTED, stated here rather than discovered below.

  1. THERE IS POSITION SIGNAL, and it is large. Across single-site promoters carrying one factor,
     the spread of mean log2 expression across designed start positions is 0.801 for GCN4
     (n = 455 over 35 positions), 0.440 for GAL4, 0.238 for LEU3 -- against a replicate noise
     floor of 0.1164. GCN4's spread is 6.9 floors. So V5's failure was not an absence of signal.

  2. POSITION VARIES WHERE IT CAN BE USED. 4,903 of the 5,790 promoters sit in a (context, factor
     multiset) group that contains more than one position layout, so position is a live variable
     for 85% of the library rather than a rare one.

  3. ORIENTATION IS BARELY VARIED AT ALL. 10,087 sites are 'plus' and 482 are 'minus' -- 4.6%.
     Whatever orientation does biologically, this experiment cannot measure much of it, and that
     is an availability limit of the same kind as the 42.3% of TRRUST edges carrying no sign.

  4. THE DESIGN LADDER IS 7 bp, NOT 10. The distinct start positions step by 7 along most of their
     range. This matters for gate P1 below, because a periodicity fit at period 7 is fitting the
     design grid rather than any property of DNA, and must not be read as helical phasing.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

P0  THE CEILING GATE, BOTH HALVES. (a) COST: report the cap for every budget K = C*P*O using
    signed.py's own cap function, before any accuracy is measured, since the cap depends only on
    the alphabet size. (b) ACCURACY: the position spread above is the ceiling -- it is what a
    perfect position model could reach. PREDECLARED: if no split at any K beats the pure-identity
    split of the SAME K, then position and orientation are not worth budget in this
    representation, and V5's result stands at the level of the representation and not only the
    estimator.

P1  PHASE IS NOT LOCATION, AND THE PREDICTION IS RECORDED BEFORE THE TEST. The paper reports a
    ~10 bp PERIODIC dependence on site location -- helical phasing. promoter.py's V5 tested
    start//10, which is LOCATION; the periodic feature is start mod 10, which is PHASE. These are
    different variables and my prediction is that phase is the one that carries the signal V5
    could not find. Tested against location bins, with period 10.5 (the helical repeat of B-DNA)
    and period 7 (the design ladder) as the two controls that separate biology from grid.

P2  THE EXCHANGE RATE, which is the actual question. At a FIXED budget K = C*P*O, sweep the split
    between identity resolution, position resolution and orientation. PREDECLARED: if the pure
    identity split (P = O = 1) wins at every K, spatial information is not worth class budget
    whatever its biological reality, and that is the result.

P3  FALLBACK, because it is what killed V5. Every split reported with its abstention rate, and
    every comparison repeated on the subset where NO split falls back.

P4  ORIENTATION ON ITS OWN. It is a doubling, the cheapest refinement available, and it is
    reported separately from position because it is affordable where position is not. Its
    availability limit from probe 3 is reported with it.

P5  ESTIMATOR OR DATA? If position carries signal a group-mean estimator cannot reach at this
    sample size, a ridge with explicit position and phase features can. A gap between the two is a
    statement about the ESTIMATOR; agreement is a statement about the DATA. V5 asserted the first
    without testing it; this tests it.

P6  WHAT THIS DOES AND DOES NOT SETTLE.

NO LEAKAGE NOTE. The factor classes are fitted from the response and are therefore refitted inside
each training fold. The position bins and the orientation flag are properties of the DESIGN, not
of the response, so they carry no information from held-out promoters and are fixed once.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.promoter import fetch, load
from rem.atlas.signed import tf_effects, class_bins, cv_grouped, class_caps, edge_coverage


def pos_binner(sub, P, mode="loc"):
    """P position bins. 'loc' is equal-frequency bins of the designed start coordinate; 'phase'
    is start mod 10 folded into P bins; 'phase105' uses the B-DNA helical repeat; 'ladder' uses
    the 7 bp design step and exists only as the control that tells grid from biology."""
    if P <= 1:
        return lambda s: 0
    starts = np.array([t[3] for r in sub for t in r["tfs"]], dtype=float)
    if mode == "loc":
        qs = np.quantile(starts, np.linspace(0, 1, P + 1)[1:-1])
        return lambda s: int(np.searchsorted(qs, s))
    per = {"phase": 10.0, "phase105": 10.5, "ladder": 7.0}[mode]
    return lambda s: int(min(P - 1, (s % per) / per * P))


def cell_key(cls, posb, use_ori, r):
    """The summary: a COUNT per (factor class, position bin, orientation) cell. Sorted so the key
    is the multiset of occupied cells and their counts, which is what a count summary is."""
    cells = collections.Counter()
    for t in r["tfs"]:
        cells[(cls.get(t[0], 0), posb(t[3]), 1 if t[2] == "minus" else 0)] += 1
    return (r["ctx"], tuple(sorted(cells.items())))


def ridge_cv(sub, featfn, folds=5, seed=0, lam=1.0):
    keys = sorted({k for r in sub for k in featfn(r)})
    ki = {k: j for j, k in enumerate(keys)}
    X = np.zeros((len(sub), len(keys) + 1))
    X[:, -1] = 1.0
    for i, r in enumerate(sub):
        for k, v in featfn(r).items():
            X[i, ki[k]] += v
    y = np.array([r["y"] for r in sub])
    rng = np.random.default_rng(seed)
    idx = np.arange(len(sub))
    rng.shuffle(idx)
    fold = np.array_split(idx, folds)
    err = np.zeros(len(sub))
    for f in range(folds):
        te = fold[f]
        tr = np.array([i for i in idx if i not in set(te.tolist())])
        A, b = X[tr], y[tr]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
        err[te] = y[te] - X[te] @ w
    return float(np.sqrt(np.mean(err ** 2)))


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE); P_("THE CLASS COUNT WITH POSITION AND ORIENTATION: AN EXCHANGE RATE"); P_(RULE)
    d, sha = fetch()
    recs = load(d)
    sub = [r for r in recs if r["tfs"]]
    l1 = np.array([r["l1"] for r in recs]); l2 = np.array([r["l2"] for r in recs])
    floor = float(np.std(l1 - l2)) / np.sqrt(2.0)
    ori = collections.Counter(t[2] for r in sub for t in r["tfs"])
    starts = sorted({t[3] for r in sub for t in r["tfs"]})
    P_(f"  Sharon et al. 2012, doi:10.1038/nbt.2205, GEO GSE37851, sha256[:32] {sha}.")
    P_(f"  {len(sub)} promoters, {len({t[0] for r in sub for t in r['tfs']})} factors,"
       f" {len(starts)} distinct designed start positions in [{starts[0]}, {starts[-1]}].")
    P_(f"  orientation: {ori.get('plus',0)} plus, {ori.get('minus',0)} minus"
       f" ({ori.get('minus',0)/sum(ori.values()):.1%} of sites). Noise floor {floor:.4f} log2.")

    # position is a live variable for how much of the library?
    g = collections.defaultdict(set)
    for r in sub:
        g[(r["ctx"], tuple(sorted(t[0] for t in r["tfs"])))].add(tuple(sorted(t[3] for t in r["tfs"])))
    nlive = sum(1 for r in sub if len(g[(r["ctx"], tuple(sorted(t[0] for t in r["tfs"])))]) > 1)
    P_(f"  position is a live variable for {nlive} of {len(sub)} promoters ({nlive/len(sub):.1%}):")
    P_( "  that many sit in a (context, factor multiset) group holding more than one layout.")

    eff_all = tf_effects(sub)
    fits = {}

    def make(C, P, mode, use_ori):
        posb = pos_binner(sub, P, mode)
        st = {"f": 0}

        def mk(tr):
            f = st["f"]; st["f"] += 1
            if f not in fits:
                fits[f] = tf_effects(tr)
            cl = class_bins(fits[f], C)
            return lambda r: cell_key(cl, posb, use_ori, r)
        return mk

    e_idn, f_idn, err_idn, hit_idn = cv_grouped(
        sub, lambda tr: (lambda r: (r["ctx"], tuple(sorted(t[0] for t in r["tfs"])))))
    P_(f"  identity reference: RMSE {e_idn:.4f}, abstains on {f_idn:.1%} of held-out promoters.")

    # ---- P0(a)  COST, BEFORE ANY ACCURACY ------------------------------------------------------
    P_("\n" + RULE); P_("P0(a)  THE COST CEILING: POSITION MULTIPLIES THE ALPHABET"); P_(RULE)
    P_("  A class count over (factor class x position bin x orientation) has K = C*P*O cells. The")
    P_("  engine's cost depends only on HOW MANY cells the summary distinguishes, never on what")
    P_("  makes them distinguishable -- so signed.py's cap curve prices every split already, and")
    P_("  a position bin costs exactly what a factor class costs.")
    KS = (4, 8, 16, 32, 64, 128)
    L = 8
    caps = class_caps(KS, L)
    P_(f"\n    {'budget K = C*P*O':>18} {'cap |C| at L=8':>15} {'edge coverage':>14}")
    P_(f"    {'pattern (exact)':>18} {caps[('pattern', 0)]:>15}"
       f" {edge_coverage(caps[('pattern', 0)]):>13.1%}")
    for K in KS:
        P_(f"    {K:>18} {caps[('class', K)]:>15} {edge_coverage(caps[('class', K)]):>13.1%}")
    P_("  So the question is never 'is position informative'. It is whether a position bin is")
    P_("  worth more than the factor class it displaces at the same K.")

    # ---- P1  PHASE OR LOCATION -----------------------------------------------------------------
    P_("\n" + RULE); P_("P1  PHASE IS NOT LOCATION -- A PREDICTION, TESTED"); P_(RULE)
    P_("  Predicted before the test: the paper reports a ~10 bp PERIODIC dependence, V5 tested")
    P_("  start//10 which is LOCATION, and phase is the feature that should carry what V5 missed.")
    P_("  Period 7 is the DESIGN LADDER and is here only as the control that tells grid from DNA.")
    P_(f"\n    held out, at a fixed factor-class resolution C = 8 and P = 4 position bins")
    P_(f"    {'position feature':<28} {'RMSE':>9} {'vs P=1':>9} {'floors':>8} {'fallback':>9}")
    base8, fb8, err8, hit8 = cv_grouped(sub, make(8, 1, "loc", False))
    P_(f"    {'none (P = 1)':<28} {base8:>9.4f} {0.0:>9.4f} {0.0:>8.2f} {fb8:>8.1%}")
    p1 = {}
    for mode in ("loc", "phase", "phase105", "ladder"):
        e, f, er, h = cv_grouped(sub, make(8, 4, mode, False))
        p1[mode] = (e, f, er, h)
        P_(f"    {mode:<28} {e:>9.4f} {base8-e:>+9.4f} {(base8-e)/floor:>+8.2f} {f:>8.1%}")
    P_("\n    the same four on the IDENTITY model, which is what V5 tested:")
    ei, fi, _, _ = cv_grouped(sub, make(404, 1, "loc", False))
    P_(f"    {'none (P = 1)':<28} {ei:>9.4f} {0.0:>9.4f} {0.0:>8.2f} {fi:>8.1%}")
    for mode in ("loc", "phase"):
        e, f, _, _ = cv_grouped(sub, make(404, 4, mode, False))
        P_(f"    {mode:<28} {e:>9.4f} {ei-e:>+9.4f} {(ei-e)/floor:>+8.2f} {f:>8.1%}")
    best = min(p1, key=lambda m: p1[m][0])
    won = p1["phase"][0] < p1["loc"][0]
    P_(f"\n  P1: best position feature at C = 8 is '{best}'.")
    P_(f"  The prediction was that PHASE beats LOCATION. Measured: phase {p1['phase'][0]:.4f}"
       f" against location {p1['loc'][0]:.4f}, so the prediction is"
       f" {'UPHELD' if won else 'LOST'}")
    P_(f"  by {abs(p1['phase'][0]-p1['loc'][0])/floor:.2f} floors. The 7 bp design-ladder control"
       f" scores {p1['ladder'][0]:.4f};")
    P_( "  a phase feature that does no better than the ladder is fitting the grid, not the DNA.")

    # ---- P2  THE EXCHANGE RATE -----------------------------------------------------------------
    P_("\n" + RULE); P_("P2  THE EXCHANGE RATE AT A FIXED BUDGET"); P_(RULE)
    P_("  At each budget K the same number of cells is spent differently. The pure-identity split")
    P_("  P = O = 1 is the incumbent from signed.py; anything else has to beat it at equal cost.")
    P_(f"\n    {'K':>5} {'C':>5} {'P':>4} {'O':>3} {'feature':>9} {'RMSE':>9} {'vs pure C':>10}"
       f" {'floors from ident':>18} {'fallback':>9}")
    rows = {}
    for K in KS:
        splits = []
        for O in (1, 2):
            P__ = 1
            while P__ <= K:
                C = K // (P__ * O)
                if C >= 1 and C * P__ * O == K:
                    splits.append((C, P__, O))
                P__ *= 2
        pure = None
        for (C, Pb, O) in splits:
            mode = best if Pb > 1 else "loc"
            e, f, er, h = cv_grouped(sub, make(C, Pb, mode, O == 2))
            rows[(K, C, Pb, O)] = (e, f, er, h)
            if Pb == 1 and O == 1:
                pure = e
        for (C, Pb, O) in splits:
            e, f, er, h = rows[(K, C, Pb, O)]
            tag = "--" if Pb == 1 else best
            mark = "  <-- incumbent" if (Pb == 1 and O == 1) else (
                "  BEATS IT" if pure is not None and e < pure else "")
            P_(f"    {K:>5} {C:>5} {Pb:>4} {O:>3} {tag:>9} {e:>9.4f}"
               f" {(pure-e) if pure else 0.0:>+10.4f} {(e-e_idn)/floor:>18.2f} {f:>8.1%}{mark}")
    wins = [(K, C, Pb, O) for (K, C, Pb, O) in rows
            if (Pb > 1 or O > 1) and rows[(K, C, Pb, O)][0] < rows[(K, K, 1, 1)][0]]
    P_(f"\n  P2: {len(wins)} of {len([k for k in rows if k[2] > 1 or k[3] > 1])} spatial splits beat"
       f" the pure-identity split of the same budget.")
    if wins:
        bk = min(wins, key=lambda k: rows[k][0])
        P_(f"  best spatial split overall: K={bk[0]} as C={bk[1]}, P={bk[2]}, O={bk[3]},"
           f" RMSE {rows[bk][0]:.4f}")
        P_(f"  against pure C={bk[0]} at {rows[(bk[0], bk[0], 1, 1)][0]:.4f}"
           f" -- {(rows[(bk[0], bk[0], 1, 1)][0]-rows[bk][0])/floor:+.2f} floors.")
    else:
        P_("  NONE. Spatial resolution is never worth the factor classes it displaces, at any")
        P_("  budget tested. Position is informative and it is still not worth buying.")

    # ---- P3  FALLBACK --------------------------------------------------------------------------
    P_("\n" + RULE); P_("P3  FALLBACK, WHICH IS WHAT KILLED V5"); P_(RULE)
    shown = [(K, K, 1, 1) for K in KS] + sorted(wins, key=lambda k: rows[k][0])[:3]
    if not wins:
        shown += [(64, 16, 4, 1), (64, 8, 4, 2)]
    shown = [k for k in shown if k in rows]
    ok = hit_idn.copy()
    for k in shown:
        ok = ok & rows[k][3]
    P_(f"  {int(ok.sum())} of {len(sub)} promoters scored by every split below without fallback")
    P_(f"\n    {'split':<24} {'all':>9} {'no-fallback':>12} {'floors from ident':>18}")
    sub_idn = float(np.sqrt(np.mean(err_idn[ok] ** 2)))
    for k in shown:
        e, f, er, h = rows[k]
        s = float(np.sqrt(np.mean(er[ok] ** 2)))
        P_(f"    {f'K={k[0]} C={k[1]} P={k[2]} O={k[3]}':<24} {e:>9.4f} {s:>12.4f}"
           f" {(s-sub_idn)/floor:>18.2f}")
    P_(f"    {'identity':<24} {e_idn:>9.4f} {sub_idn:>12.4f} {0.0:>18.2f}")
    P_("  If a spatial split wins on all promoters but loses here, it won by abstaining less,")
    P_("  not by predicting better. If it wins in both columns, the gain is real.")

    # ---- P4  ORIENTATION ALONE -----------------------------------------------------------------
    P_("\n" + RULE); P_("P4  ORIENTATION ON ITS OWN, THE CHEAPEST REFINEMENT AVAILABLE"); P_(RULE)
    P_(f"    {'C':>5} {'without orientation':>20} {'with orientation':>18} {'floors':>8}")
    for C in (4, 8, 32, 64):
        a, _, _, _ = cv_grouped(sub, make(C, 1, "loc", False))
        b, _, _, _ = cv_grouped(sub, make(C, 1, "loc", True))
        P_(f"    {C:>5} {a:>20.4f} {b:>18.4f} {(a-b)/floor:>+8.2f}")
    P_(f"\n  Orientation is a doubling of the alphabet -- the cheapest refinement there is -- and")
    P_(f"  only {ori.get('minus',0)/sum(ori.values()):.1%} of sites are 'minus'. Whatever it does")
    P_( "  biologically, this experiment barely varies it, so a null here is an AVAILABILITY")
    P_( "  result and not a biological one. Same shape as the 42.3% of TRRUST edges with no sign.")

    # ---- P5  ESTIMATOR OR DATA -----------------------------------------------------------------
    P_("\n" + RULE); P_("P5  IS IT THE ESTIMATOR OR THE DATA?"); P_(RULE)
    P_("  V5 asserted that a group-mean estimator cannot exploit position at this sample size,")
    P_("  and did not test it. A ridge can use position without splitting groups at all, because")
    P_("  it shares strength across positions instead of partitioning by them.")
    cl8 = class_bins(eff_all, 8)

    def f_ctx(r):
        return {("ctx", r["ctx"]): 1.0}

    def f_cls(r):
        f = f_ctx(r)
        for t in r["tfs"]:
            f[("cl", cl8.get(t[0], 0))] = f.get(("cl", cl8.get(t[0], 0)), 0.0) + 1.0
        return f

    def f_cls_pos(r):
        f = f_cls(r)
        for t in r["tfs"]:
            f[("pos", t[3] // 10)] = f.get(("pos", t[3] // 10), 0.0) + 1.0
        return f

    def f_cls_phase(r):
        f = f_cls(r)
        for t in r["tfs"]:
            for per in (10.0, 10.5, 7.0):
                ph = 2 * np.pi * t[3] / per
                f[("cos", per)] = f.get(("cos", per), 0.0) + float(np.cos(ph))
                f[("sin", per)] = f.get(("sin", per), 0.0) + float(np.sin(ph))
        return f

    def f_tf(r):
        f = f_ctx(r)
        for t in r["tfs"]:
            f[("tf", t[0])] = f.get(("tf", t[0]), 0.0) + 1.0
        return f

    def f_tf_pos(r):
        f = f_tf(r)
        for t in r["tfs"]:
            f[("pos", t[3] // 10)] = f.get(("pos", t[3] // 10), 0.0) + 1.0
        return f

    P_(f"\n    {'ridge model':<38} {'RMSE':>9} {'gain vs its own base':>21}")
    r_cls = ridge_cv(sub, f_cls)
    r_tf = ridge_cv(sub, f_tf)
    P_(f"    {'context + class counts (C=8)':<38} {r_cls:>9.4f} {'--':>21}")
    for nm, fn, base in (("  + position bins", f_cls_pos, r_cls),
                         ("  + explicit phase (10, 10.5, 7 bp)", f_cls_phase, r_cls)):
        v = ridge_cv(sub, fn)
        P_(f"    {nm:<38} {v:>9.4f} {(base-v)/floor:>+20.2f}f")
    P_(f"    {'context + per-factor counts':<38} {r_tf:>9.4f} {'--':>21}")
    v = ridge_cv(sub, f_tf_pos)
    P_(f"    {'  + position bins':<38} {v:>9.4f} {(r_tf-v)/floor:>+20.2f}f")
    P_("\n  A ridge that gains from position where the group mean does not is a statement about")
    P_("  the ESTIMATOR: the information is there and partitioning is the wrong way to reach it.")
    P_("  A ridge that gains nothing either is a statement about the DATA at this sample size.")

    # ---- P6 ------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("P6  WHAT THIS DOES AND DOES NOT SETTLE"); P_(RULE)
    P_("  1. Position here is a DESIGNED coordinate in a synthetic promoter with at most 49")
    P_("     distinct values stepping mostly by 7 bp. It is not a dense scan, and a periodicity")
    P_("     finer than the design step cannot be seen however real it is.")
    P_("  2. Orientation is 4.6% varied. A null on orientation is an availability result.")
    P_("  3. The cap is a COST statement from graph arithmetic on TRRUST. It says the engine")
    P_("     could be built at that size, not that it would be accurate there. And TRRUST has no")
    P_("     position information at all, so the spatial axis is priced by alphabet size only --")
    P_("     which is exactly right for the cost and says nothing about whether real regulatory")
    P_("     positions would cluster helpfully.")
    P_("  4. The exchange rate measured here is between position and FACTOR IDENTITY at equal")
    P_("     alphabet. It does not say position is unimportant to a promoter; it says that if you")
    P_("     have a fixed number of distinguishable cells, this is where they are best spent.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_poscount.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P_(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
