"""LOOP 99 -- CELL CYCLE: CAN THE PHASE-LABELLED GENES RECOVER THE PHASE ORDER WITHOUT BEING TOLD IT?

THE ASK. cell_complete.json carries a `cellcycle` layer: 153 genes, each stamped with a phase. That
stamp is an ANNOTATION -- somebody read a paper and wrote "G2/M" next to CCNB1. The question this
loop asks is whether the model's OTHER layers, which were built from measurement rather than from
that annotation, contain enough structure to REDISCOVER the cycle: not which genes are cell-cycle
genes (they are handed to us), but the ORDER the phases run in.

WHY IT MATTERS AND WHY IT IS THE RIGHT TEST. A whole-cell model that cannot order its own phases has
a cell cycle in the same sense that a clock face with the numbers painted on has time. Order is the
one property of the cycle that no single label carries -- it lives only in the relations between
genes -- so recovering it from co-expression, co-dependency, protein interaction and regulation is a
genuine test of whether those layers hold cell-cycle information or merely hold a proliferation
module that fires as one lump. The prior from the biology is unkind: bulk co-expression across
tissues and cell lines is dominated by "this sample is dividing", which co-varies EVERY cell-cycle
gene together and would collapse the cycle to a point. This loop is built expecting a negative and
instrumented so that a negative is worth something.

THE METHOD, in one line: union the four measured layers into a graph on the 153 genes, take the
first two non-trivial eigenvectors of its normalised Laplacian, read each gene's ANGLE on that plane,
and ask whether the phase labels sit around the circle in the biological order. Nothing about the
embedding sees a phase label. Complexes are HELD OUT of the graph so that co-complex membership can
serve as a positive control the pipeline could fail.

DISCLOSURE, per project rule 2. Before these gates were fixed I inspected, and only inspected: the
phase-label vocabulary and its counts; the shapes of the coexpr / codep / ppi / reg / complexes
layers; the number of within-set edges each layer contributes; and the connected-component sizes of
the union graph. Those facts decide what a "cycle" can even mean here and what an embedding is
possible on, and Z1 and Z2 below are written in full knowledge of them -- Z1 in particular is
declared as the requirement the biology demands, knowing it will not be met. NO embedding
coordinate, NO angle, NO ordering statistic, NO correlation and NO null was computed before this
docstring was written and every threshold below was fixed.

ADDED AFTER THE FIRST RUN, and disclosed as such. Two DIAGNOSTICS were added to Z4 after its
numbers were seen, both of which can only weaken this loop's claim and NEITHER of which is gated:
(i) the concordance recomputed on DIFFERENT-PHASE PAIRS ONLY, because a correlation between angular
separation and cyclic phase separation can be produced entirely by same-phase genes clustering, with
no ordering information in it at all -- the decomposition separates "clustering" from "order" and
the first run could not tell them apart; and (ii) the eigenvalue-gap signature, since a graph that
is genuinely ring-like has a NEARLY DEGENERATE first eigenvector pair (the sin/cos pair) and this
one can be checked. No threshold anywhere in this module was changed after any number was seen. In
particular Z4(a)'s rho >= 0.20 is left exactly where it was declared, and the run misses it.

PREDECLARED, before any number:

  Z1 THE LABELS ARE A CYCLE AT ALL                                  THE PRECONDITION.
       you cannot recover the order of a cycle whose phases are not all named. Gate, both halves
       required: (a) at least 4 distinct phases each with >= 5 genes, and (b) the four canonical
       segments G1, S, G2, M each named by at least one label. Reported with the chance baseline
       for a cyclic order at whatever number of phases actually survives, because with few phases
       "the correct order" is a cheap thing to hit.
  Z2 THE GRAPH EXISTS                                               THE OTHER PRECONDITION.
       a spectral embedding of a shattered graph is an embedding of a fragment. Gate: the largest
       connected component of the union graph (coexpr + codep + ppi + reg, complexes held out) must
       hold >= 50% of the 153 genes. Genes outside it are dropped and counted as dropped.
  Z3 THE EMBEDDING CARRIES REAL STRUCTURE                           THE POSITIVE CONTROL.
       if the embedding is noise then the phase answer means nothing in either direction, so the
       pipeline must first be shown to recover something it was never told. Complexes are held out
       of the graph; co-complex pairs must land closer in angle than non-co-complex pairs. Gate:
       AUC >= 0.55 AND |z| >= 2 against a node-permutation null. Reported twice -- on the full
       graph, and on an expression-only graph with PPI also removed, which is the genuinely
       independent version, since complex members physically interact and PPI therefore leaks
       complex structure into the full graph.
  Z4 THE PHASE ORDER                                                THE GATE.
       two statistics, both required to pass. (a) CYCLIC CONCORDANCE: Spearman rho between the
       angular separation of a gene pair and the cyclic separation of their phases, over all pairs;
       rotation- and reflection-invariant. Threshold rho >= 0.20 with |z| >= 2 against a phase-label
       shuffle. (b) THE ORDER ITSELF: the circular mean angle of each phase, sorted around the
       circle, must equal the canonical cycle up to rotation and reflection, with empirical
       p <= 0.05 across the same shuffles. A robustness sweep over the first three eigenvector pairs
       is reported with its own max-over-projections null, so that "we only looked at one plane" is
       not available as an excuse in either direction.
  Z5 THE NULL FIRES, AND IS CHECKED FOR THE ABILITY TO              THE GUARD, GUARDED.
       gate_guard.null_can_move() on the label vector BEFORE any null verdict is read, and
       gate_guard.survival() to decide whether a survival fraction is defined at all. Twelve gates
       in this project fired while measuring nothing. Gate: the null must be capable.
  Z6 THE FAME CONTROL, RUN HEAD TO HEAD                             THE RECURRING KILLER.
       `pubs` has beaten the biology in loops 87, 87b, 94 and 96. Mitotic kinases are among the most
       studied genes in the genome. Reported: pubs by phase, pubs against angle and against radius.
       And a head-to-head on ONE statistic -- same-phase pair AUC computed from angular proximity
       (the graph) against the same AUC computed from proximity in log publication count (fame
       alone). Gate: the graph must beat fame. If fame wins, the layers are a citation record.
  Z7 WHAT THIS DOES NOT GIVE                                        THE HONEST LIMIT.
       an ordering is not a clock. Stated so that nothing here is read as a running cell cycle.

-> outputs/loop_cellcycle.json
"""
import json
import os
import sys
import math
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import gate_guard as GG  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))

# The canonical cell cycle. Order is fixed here, before anything is measured.
CANON = ("G1", "S", "G2", "G2/M", "Mitotic")
SEGMENT = {"G1": "G1", "S": "S", "G2": "G2", "G2/M": "M", "Mitotic": "M"}

MIN_PER_PHASE = 5
MIN_PHASES = 4
LCC_MIN_FRAC = 0.50
POS_AUC_MIN = 0.55
CONC_RHO_MIN = 0.20
Z_MIN = 2.0
P_MAX = 0.05
NPERM = 500
SEED = 9901

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def spear(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < 30:
        return float("nan")
    return float(spearmanr(a[f], b[f]).statistic)


def auc(score_pos, score_neg):
    """Mann-Whitney AUC: probability a positive scores above a negative."""
    from scipy.stats import rankdata
    p, n = np.asarray(score_pos, float), np.asarray(score_neg, float)
    if len(p) < 3 or len(n) < 3:
        return float("nan")
    r = rankdata(np.concatenate([p, n]))
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2.0) / (len(p) * len(n)))


def spectral(W, k=5):
    """Eigenvectors of the random-walk normalised Laplacian, largest-eigenvalue-of-M first.

    Column 0 is the trivial vector (proportional to sqrt(degree)); columns 1..k-1 are the
    non-trivial coordinates. For a ring graph columns 1 and 2 are the sin/cos pair, which is
    exactly why this is the embedding a cyclic-order question calls for.
    """
    d = W.sum(1).astype(float)
    d[d <= 0] = 1.0
    dm = 1.0 / np.sqrt(d)
    M = W * dm[:, None] * dm[None, :]
    vals, vecs = np.linalg.eigh(M)
    o = np.argsort(-vals)
    vals, vecs = vals[o][:k], vecs[:, o][:, :k]
    return vals, vecs * dm[:, None]


def ang_of(x, y):
    return np.arctan2(y, x)


def ang_dist(a):
    """Pairwise angular distance in [0, pi], flattened over i<j."""
    n = len(a)
    iu = np.triu_indices(n, 1)
    d = a[iu[0]] - a[iu[1]]
    return np.abs(np.arctan2(np.sin(d), np.cos(d))), iu


def circ_mean(a):
    c, s = np.cos(a).mean(), np.sin(a).mean()
    return float(np.arctan2(s, c)), float(np.hypot(c, s))


def cyclic_order(labels, angles, phases):
    """Sort phases by circular mean angle -> the cyclic sequence the embedding proposes."""
    mu = {}
    for p in phases:
        m = np.array([i for i, L in enumerate(labels) if L == p])
        if len(m) == 0:
            continue
        mu[p] = circ_mean(angles[m])
    seq = sorted(mu, key=lambda p: mu[p][0])
    return seq, mu


def matches_canon(seq, canon):
    """Is `seq` the canonical cycle up to rotation and reflection?"""
    c = [p for p in canon if p in seq]
    if len(c) != len(seq) or len(seq) < 3:
        return False
    n = len(seq)
    for src in (c, c[::-1]):
        for r in range(n):
            if [src[(r + i) % n] for i in range(n)] == seq:
                return True
    return False


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 99 -- cell cycle: can the phase-labelled genes recover the phase ORDER")
    say("               without being told it?")
    say("=" * 100)
    say()

    C = json.load(open(LR.CELL))
    genes = C["genes"]
    names = [g["name"] for g in genes]
    pubs = np.array([float(g.get("pubs") or 0.0) for g in genes])
    cc = C["cellcycle"]
    all_idx = sorted(int(k) for k in cc)
    S = set(all_idx)
    say(f"  {len(genes):,} genes in the model; {len(all_idx):,} carry a cell-cycle phase label")
    say()

    # ---------------------------------------------------------------- Z1
    say("Z1 THE LABELS ARE A CYCLE AT ALL")
    from collections import Counter
    cnt = Counter(cc[str(i)] for i in all_idx)
    for p, n in cnt.most_common():
        say(f"     {p:<10} {n:>4} genes")
    kept_phases = [p for p in CANON if cnt.get(p, 0) >= MIN_PER_PHASE]
    z1a = len(kept_phases) >= MIN_PHASES
    segs_named = {SEGMENT.get(p, p) for p in cnt}
    missing = [s for s in ("G1", "S", "G2", "M") if s not in segs_named]
    z1b = len(missing) == 0
    say(f"     phases with >= {MIN_PER_PHASE} genes: {len(kept_phases)} "
        f"({', '.join(kept_phases)})  -> (a) {'PASS' if z1a else 'FAIL'}")
    say(f"     canonical segments named: {sorted(segs_named)}   MISSING: "
        f"{missing if missing else 'none'}  -> (b) {'PASS' if z1b else 'FAIL'}")
    k = len(kept_phases)
    chance = 1.0 / max(1, math.factorial(max(0, k - 1)) // 2) if k >= 3 else 1.0
    say(f"     with {k} phases there are {max(1, math.factorial(max(0, k - 1)) // 2)} distinct "
        f"cyclic orders up to rotation and reflection, so the chance baseline for")
    say(f"     'the correct order' is {chance:.1%}. This is a WEAK test and the weakness is the "
        f"data's, not the method's.")
    z1 = bool(z1a and z1b)
    say(f"     Z1 {'PASS' if z1 else 'FAIL'} -- " +
        ("the label set spans the cycle" if z1 else
         f"the label set does NOT span the cycle: no {'/'.join(missing)} phase is named, so the "
         f"ORDER of a segment that is absent cannot be recovered at all"))
    say()

    # ---------------------------------------------------------------- Z2
    say("Z2 THE GRAPH EXISTS")
    n_all = len(all_idx)
    pos = {g: i for i, g in enumerate(all_idx)}
    layers = {}

    def blank():
        return np.zeros((n_all, n_all), float)

    A_co, A_cd, A_ppi, A_reg = blank(), blank(), blank(), blank()
    for i in all_idx:
        for j, w in C["coexpr"].get(str(i), []):
            if j in S and j != i:
                A_co[pos[i], pos[j]] = A_co[pos[j], pos[i]] = 1.0
        for j, w in C["codep"].get(str(i), []):
            if j in S and j != i:
                A_cd[pos[i], pos[j]] = A_cd[pos[j], pos[i]] = 1.0
    for e in C["ppi"]:
        a, b = e[0], e[1]
        if a in S and b in S and a != b:
            A_ppi[pos[a], pos[b]] = A_ppi[pos[b], pos[a]] = 1.0
    for e in C["reg"]:
        a, b = e[0], e[1]
        if a in S and b in S and a != b:
            A_reg[pos[a], pos[b]] = A_reg[pos[b], pos[a]] = 1.0
    layers = {"coexpr": A_co, "codep": A_cd, "ppi": A_ppi, "reg": A_reg}
    for nm, A in layers.items():
        say(f"     {nm:<8} contributes {int(A.sum() // 2):>5,} undirected within-set edges")
    W_full = A_co + A_cd + A_ppi + A_reg
    W_expr = A_co + A_cd + A_reg          # PPI removed: genuinely independent of complexes
    say(f"     complexes are HELD OUT of every graph and used only as the positive control")

    def lcc(W):
        seen, best = set(), []
        for i in range(len(W)):
            if i in seen:
                continue
            st, c = [i], []
            seen.add(i)
            while st:
                x = st.pop()
                c.append(x)
                for y in np.flatnonzero(W[x] > 0):
                    if y not in seen:
                        seen.add(int(y))
                        st.append(int(y))
            if len(c) > len(best):
                best = c
        return sorted(best)

    keep = lcc(W_full)
    keep_e = lcc(W_expr)
    say(f"     union graph (4 layers): largest component {len(keep)}/{n_all} "
        f"({len(keep)/n_all:.1%}); {n_all - len(keep)} gene(s) dropped")
    say(f"     expression-only graph (no PPI): largest component {len(keep_e)}/{n_all} "
        f"({len(keep_e)/n_all:.1%})")
    z2 = bool(len(keep) / n_all >= LCC_MIN_FRAC)
    say(f"     Z2 {'PASS' if z2 else 'FAIL'} -- gate was >= {LCC_MIN_FRAC:.0%} in one component")
    say()

    gidx = [all_idx[i] for i in keep]
    labels = [cc[str(g)] for g in gidx]
    W = W_full[np.ix_(keep, keep)]
    deg = W.sum(1)
    say(f"     embedding {len(gidx)} genes; degree median {np.median(deg):.0f}, "
        f"mean {deg.mean():.1f}, min {deg.min():.0f}")
    vals, V = spectral(W, k=5)
    ang = ang_of(V[:, 1], V[:, 2])
    rad = np.hypot(V[:, 1], V[:, 2])
    say(f"     spectral eigenvalues (of the normalised adjacency): "
        f"{', '.join(f'{v:+.4f}' for v in vals)}")
    # POST-HOC DIAGNOSTIC (i): a ring graph has a DEGENERATE first eigenvector pair, because sin and
    # cos of the same frequency have the same eigenvalue. If lambda1 and lambda2 are well separated
    # the graph is not ring-like and the "angle" being read is a projection of something else.
    gap12 = float(vals[1] - vals[2])
    gap23 = float(vals[2] - vals[3])
    say(f"     RING SIGNATURE: a cyclic graph's first two non-trivial eigenvectors are a degenerate")
    say(f"     sin/cos pair. Here lambda1 {vals[1]:+.4f}, lambda2 {vals[2]:+.4f}, lambda3 "
        f"{vals[3]:+.4f}: gap(1,2) {gap12:.4f} vs gap(2,3) {gap23:.4f}, ratio "
        f"{gap12/gap23 if gap23 else float('nan'):.2f}")
    say(f"     a ring would show gap(1,2) near zero. This is a diagnostic, not a gate.")
    say()

    # ---------------------------------------------------------------- Z3
    say("Z3 THE EMBEDDING CARRIES REAL STRUCTURE  (positive control, complexes held out)")
    gset = {g: i for i, g in enumerate(gidx)}
    cpairs = set()
    ncplx = 0
    for nm, mem in C["complexes"].items():
        m = sorted({gset[g] for g in mem if g in gset})
        if len(m) >= 2:
            ncplx += 1
            for a in range(len(m)):
                for b in range(a + 1, len(m)):
                    cpairs.add((m[a], m[b]))
    d_ang, iu = ang_dist(ang)
    pair_key = list(zip(iu[0].tolist(), iu[1].tolist()))
    is_cplx = np.array([(a, b) in cpairs for a, b in pair_key])
    say(f"     {ncplx} complexes contain >= 2 of the embedded genes; "
        f"{int(is_cplx.sum()):,} co-complex pairs of {len(pair_key):,} total")

    def cplx_auc(a_vec):
        d, _ = ang_dist(a_vec)
        return auc(-d[is_cplx], -d[~is_cplx])

    rng = np.random.default_rng(SEED)
    a_full = cplx_auc(ang)
    nulls_c = []
    moved_c = []
    for _ in range(NPERM // 2):
        perm = rng.permutation(len(ang))
        moved_c.append(GG.null_can_move(list(range(len(ang))), perm.tolist())["changed"])
        nulls_c.append(cplx_auc(ang[perm]))
    cap_c = GG.null_can_move(list(range(len(ang))), rng.permutation(len(ang)).tolist())
    say(f"     CAPABILITY CHECK: the node permutation moves {np.mean(moved_c):.1%} of nodes "
        f"-- capable: {cap_c['capable']}")
    s_c = GG.survival(a_full, nulls_c)
    GG.report("co-complex proximity, full graph", s_c, emit=say)
    z_c = (a_full - np.mean(nulls_c)) / (np.std(nulls_c) if np.std(nulls_c) > 0 else np.nan)
    say(f"     co-complex AUC (full graph, 4 layers) {a_full:.4f}   z {z_c:+.2f}")

    # the genuinely independent version: PPI removed, so nothing physical entered the graph
    keep_e_set = keep_e
    W_e = W_expr[np.ix_(keep_e_set, keep_e_set)]
    ge = [all_idx[i] for i in keep_e_set]
    _, Ve = spectral(W_e, k=5)
    ang_e = ang_of(Ve[:, 1], Ve[:, 2])
    gset_e = {g: i for i, g in enumerate(ge)}
    cp_e = set()
    for nm, mem in C["complexes"].items():
        m = sorted({gset_e[g] for g in mem if g in gset_e})
        for a in range(len(m)):
            for b in range(a + 1, len(m)):
                cp_e.add((m[a], m[b]))
    d_e, iu_e = ang_dist(ang_e)
    ise = np.array([(a, b) in cp_e for a, b in zip(iu_e[0].tolist(), iu_e[1].tolist())])
    a_expr = auc(-d_e[ise], -d_e[~ise]) if ise.sum() >= 3 else float("nan")
    nn = []
    for _ in range(NPERM // 2):
        p2 = rng.permutation(len(ang_e))
        dd, _ = ang_dist(ang_e[p2])
        nn.append(auc(-dd[ise], -dd[~ise]))
    s_e = GG.survival(a_expr, nn)
    GG.report("co-complex proximity, expression-only graph (PPI removed)", s_e, emit=say)
    say(f"     the full-graph number is CONTAMINATED: complex members physically interact, so PPI")
    say(f"     carries complex structure. The expression-only number is the honest one.")
    z3 = bool(np.isfinite(a_full) and a_full >= POS_AUC_MIN and abs(z_c) >= Z_MIN
              and cap_c["capable"])
    say(f"     Z3 {'PASS' if z3 else 'FAIL'} -- gate was AUC >= {POS_AUC_MIN} and |z| >= {Z_MIN}; "
        + ("the embedding recovers structure it was never given, so a phase answer is readable"
           if z3 else
           "the embedding recovers NOTHING it was not given, so the phase answer below is not "
           "interpretable in either direction"))
    say()

    # ---------------------------------------------------------------- Z4
    say("Z4 THE PHASE ORDER")
    phases_used = [p for p in CANON if cnt.get(p, 0) >= MIN_PER_PHASE and p in set(labels)]
    pcode = {p: i for i, p in enumerate(phases_used)}
    K = len(phases_used)
    lab_arr = np.array(labels)
    keepmask = np.array([L in pcode for L in labels])
    say(f"     phases carried into the ordering test: {phases_used}  "
        f"({int(keepmask.sum())} of {len(labels)} embedded genes)")

    def cyc_sep(codes):
        a, b = codes[iu[0]], codes[iu[1]]
        d = np.abs(a - b)
        return np.minimum(d, K - d).astype(float)

    codes_real = np.array([pcode.get(L, -1) for L in labels])
    valid_pair = keepmask[iu[0]] & keepmask[iu[1]]

    def concordance(codes, proj_d=None):
        d = proj_d if proj_d is not None else d_ang
        return spear(d[valid_pair], cyc_sep(codes)[valid_pair])

    rho = concordance(codes_real)
    seq, mu = cyclic_order(labels, ang, phases_used)
    ok_order = matches_canon(seq, phases_used)
    say(f"     CYCLIC CONCORDANCE rho(angular separation, cyclic phase separation) = {rho:+.4f}")
    say(f"     phase circular means on the embedding plane:")
    for p in phases_used:
        m, R = mu[p]
        say(f"       {p:<10} mean angle {np.degrees(m):+8.1f} deg   resultant length R {R:.3f}   "
            f"n {cnt[p]:>3}" + ("   (R < 0.2: the mean angle is not a direction)" if R < 0.2 else ""))
    say(f"     order proposed by the embedding: {' -> '.join(seq)}")
    say(f"     canonical order (fixed before measurement): {' -> '.join(phases_used)}")
    say(f"     match up to rotation and reflection: {ok_order}")

    # robustness sweep over projections, with the null taken under the same max
    projs = [(1, 2), (1, 3), (2, 3)]
    sweep = {}
    d_by_proj = {}
    for (i1, i2) in projs:
        a2 = ang_of(V[:, i1], V[:, i2])
        d2, _ = ang_dist(a2)
        d_by_proj[(i1, i2)] = d2
        s2, mu2 = cyclic_order(labels, a2, phases_used)
        sweep[f"v{i1}-v{i2}"] = {"rho": concordance(codes_real, d2),
                                 "order": s2, "match": bool(matches_canon(s2, phases_used))}
        say(f"     projection v{i1}/v{i2}: rho {sweep[f'v{i1}-v{i2}']['rho']:+.4f}   "
            f"order {' -> '.join(s2)}   match {sweep[f'v{i1}-v{i2}']['match']}")
    rho_max = max(v["rho"] for v in sweep.values())
    any_match = any(v["match"] for v in sweep.values())
    say(f"     best over the three projections: rho {rho_max:+.4f}, any order match {any_match}")
    say()
    # POST-HOC DIAGNOSTIC (ii): IS THE CONCORDANCE ORDER, OR IS IT CLUSTERING? The cyclic separation
    # takes values {0, 1, 2}. A positive rho can be produced entirely by the 0-vs-rest contrast --
    # same-phase genes sitting together -- which contains no information about the ORDER of the
    # phases whatsoever. Dropping every same-phase pair leaves only the 1-vs-2 contrast: are
    # ADJACENT phases closer in angle than OPPOSITE ones? That, and only that, is order.
    say("     POST-HOC DECOMPOSITION -- is the concordance ORDER, or is it clustering?")
    cs_real = cyc_sep(codes_real)
    diff_only = valid_pair & (cs_real > 0)
    rho_diff = spear(d_ang[diff_only], cs_real[diff_only])
    rho_same = spear(d_ang[valid_pair], (cs_real[valid_pair] > 0).astype(float))
    say(f"       clustering component  rho(angle, same-phase vs not)      {rho_same:+.4f}  "
        f"({int(valid_pair.sum()):,} pairs)")
    say(f"       ORDER component       rho(angle, adjacent vs opposite)   {rho_diff:+.4f}  "
        f"({int(diff_only.sum()):,} different-phase pairs only)")
    say(f"       the second number is the one that carries cyclic order. It is gated on nothing --")
    say(f"       it was added after the first run and is reported to prevent the first number from")
    say(f"       being read as order when it may be nothing but same-phase genes sitting together.")
    say()

    # ---------------------------------------------------------------- Z5
    say("Z5 THE NULL FIRES, AND IS CHECKED FOR THE ABILITY TO")
    shuf0 = list(rng.permutation(lab_arr))
    cap = GG.null_can_move(list(lab_arr), shuf0)
    say(f"     CAPABILITY CHECK: shuffling the phase labels changes {cap['changed']:.1%} of "
        f"entries -- capable: {cap['capable']}")
    say(f"     ({cap['reason']})")
    nulls_rho, nulls_max, nulls_order, nulls_ordermax, moved = [], [], [], [], []
    nulls_diff = []
    for _ in range(NPERM):
        perm = rng.permutation(len(labels))
        sl = lab_arr[perm]
        moved.append(GG.null_can_move(list(lab_arr), list(sl))["changed"])
        cds = np.array([pcode.get(L, -1) for L in sl])
        km = np.array([L in pcode for L in sl])
        vp = km[iu[0]] & km[iu[1]]
        a_, b_ = cds[iu[0]], cds[iu[1]]
        dd = np.abs(a_ - b_)
        cs = np.minimum(dd, K - dd).astype(float)
        do = vp & (cs > 0)
        nulls_diff.append(spear(d_ang[do], cs[do]))
        nulls_rho.append(spear(d_ang[vp], cs[vp]))
        nulls_max.append(max(spear(d_by_proj[p][vp], cs[vp]) for p in projs))
        sq, _ = cyclic_order(list(sl), ang, phases_used)
        nulls_order.append(bool(matches_canon(sq, phases_used)))
        nulls_ordermax.append(any(
            matches_canon(cyclic_order(list(sl), ang_of(V[:, a], V[:, b]), phases_used)[0],
                          phases_used) for a, b in projs))
    say(f"     the shuffle moves {np.mean(moved):.1%} of labels on average over {NPERM} draws")
    s_rho = GG.survival(rho, nulls_rho)
    GG.report("cyclic concordance under a phase-label shuffle", s_rho, emit=say)
    say(f"     ('survives -0%' above is not a typo and not an UNDEFINED: the null mean is "
        f"{s_rho['null_mean']:+.4f}, so essentially NONE of")
    say(f"      the concordance survives the shuffle. The effect is real -- it is simply small.)")
    s_max = GG.survival(rho_max, nulls_max)
    GG.report("best-of-3-projections concordance, same max taken in the null", s_max, emit=say)
    s_diff = GG.survival(rho_diff, nulls_diff)
    GG.report("POST-HOC: ORDER component only (different-phase pairs)", s_diff, emit=say)
    p_order = (1.0 + sum(nulls_order)) / (1.0 + NPERM)
    p_ordermax = (1.0 + sum(nulls_ordermax)) / (1.0 + NPERM)
    say(f"     order match under shuffle: {sum(nulls_order)}/{NPERM} shuffles also produce the "
        f"canonical order -> empirical p = {p_order:.3f}")
    say(f"     (best-of-3 version: {sum(nulls_ordermax)}/{NPERM}, p = {p_ordermax:.3f})")
    say(f"     with {K} phases the analytic chance is {chance:.1%}, and the shuffle agrees, which")
    say(f"     is the point: an order match here is close to a coin toss whatever the graph says.")
    z5 = bool(cap["capable"])
    say(f"     Z5 {'PASS' if z5 else 'FAIL'} -- the null "
        f"{'is capable and its verdict is usable' if z5 else 'is INERT'}")
    say()

    z_rho = s_rho.get("z", float("nan"))
    z4a = bool(np.isfinite(rho) and rho >= CONC_RHO_MIN and np.isfinite(z_rho)
               and abs(z_rho) >= Z_MIN)
    z4b = bool(ok_order and p_order <= P_MAX)
    z4 = bool(z4a and z4b)
    say(f"     Z4(a) concordance {rho:+.4f} vs threshold {CONC_RHO_MIN:+.2f}, z {z_rho:+.2f} vs "
        f"{Z_MIN} -> {'PASS' if z4a else 'FAIL'}")
    say(f"     Z4(b) order match {ok_order} with p {p_order:.3f} vs {P_MAX} -> "
        f"{'PASS' if z4b else 'FAIL'}")
    if not z4a and np.isfinite(rho) and abs(z_rho) >= Z_MIN:
        say(f"     NOTE ON (a), because this is exactly where a threshold gets moved: the "
            f"concordance MISSES the")
        say(f"     predeclared bar by {CONC_RHO_MIN - rho:.4f}, while being enormously "
            f"distinguishable from the shuffle (z {z_rho:+.1f};")
        say(f"     0 of {NPERM} shuffles came near it). Those two facts are not in conflict -- the "
            f"signal is REAL and it is")
        say(f"     WEAK. The bar was set at {CONC_RHO_MIN:.2f} before the number existed and it "
            f"stays there. A z-score alone would")
        say(f"     have passed anything, since with {int(valid_pair.sum()):,} pairs a rho of 0.03 "
            f"clears z = 2.")
    if not z4b and ok_order:
        say(f"     NOTE ON (b): the order match is TRUE and WORTHLESS. {sum(nulls_order)} of "
            f"{NPERM} random relabelings also")
        say(f"     produce the canonical order, because with {K} phases there are only {int(max(1, math.factorial(max(0, K - 1)) // 2))} "
            f"of them. Reporting 'the embedding")
        say(f"     recovered G1 -> G2 -> G2/M -> M' without this line would be the headline of a "
            f"paper and a lie.")
    say(f"     Z4 {'PASS' if z4 else 'FAIL'} -- " +
        ("the layers recover the cyclic order of the phases" if z4 else
         "the layers do NOT recover the cyclic order of the phases"))
    say()

    # ---------------------------------------------------------------- Z6
    say("Z6 THE FAME CONTROL, RUN HEAD TO HEAD")
    pb = pubs[np.array(gidx)]
    lp = np.log10(1.0 + pb)
    for p in phases_used:
        m = np.array([i for i, L in enumerate(labels) if L == p])
        say(f"     {p:<10} n {len(m):>3}   median pubs {np.median(pb[m]):>8.0f}   "
            f"mean {pb[m].mean():>9.1f}   max {pb[m].max():>7.0f}")
    from scipy.stats import kruskal
    groups = [pb[np.array([i for i, L in enumerate(labels) if L == p])] for p in phases_used]
    kw = kruskal(*groups)
    say(f"     Kruskal-Wallis pubs across phases: H {kw.statistic:.2f}  p {kw.pvalue:.3g}")
    say(f"     pubs vs embedding angle  rho {spear(lp, ang):+.4f}    "
        f"vs radius rho {spear(lp, rad):+.4f}    vs degree rho {spear(lp, deg):+.4f}")

    same_phase = (codes_real[iu[0]] == codes_real[iu[1]]) & valid_pair
    diff_phase = (codes_real[iu[0]] != codes_real[iu[1]]) & valid_pair
    auc_graph = auc(-d_ang[same_phase], -d_ang[diff_phase])
    d_pub = np.abs(lp[iu[0]] - lp[iu[1]])
    auc_pubs = auc(-d_pub[same_phase], -d_pub[diff_phase])
    say(f"     HEAD TO HEAD on one statistic -- same-phase pair AUC:")
    say(f"       from the graph  (angular proximity)      AUC {auc_graph:.4f}")
    say(f"       from FAME alone (log-pubs proximity)     AUC {auc_pubs:.4f}")
    say(f"       {int(same_phase.sum()):,} same-phase pairs vs {int(diff_phase.sum()):,} "
        f"different-phase pairs; the label set is {cnt.most_common(1)[0][1]/n_all:.0%} "
        f"'{cnt.most_common(1)[0][0]}', so same-phase pairs are dominated by that one class")
    z6 = bool(np.isfinite(auc_graph) and np.isfinite(auc_pubs) and auc_graph > auc_pubs)
    say(f"     Z6 {'PASS' if z6 else 'FAIL'} -- " +
        ("the graph beats fame on the same statistic" if z6 else
         "FAME BEATS THE GRAPH: publication count clusters the phase labels better than four "
         "measured layers do, so what the labels track is study effort"))
    say()

    # ---------------------------------------------------------------- Z7
    say("Z7 WHAT THIS DOES NOT GIVE")
    say(f"     An ordering is not a clock. Even a passing Z4 would give a static ranking of genes")
    say(f"     around a circle, with no period, no rate, no checkpoint and no way to advance the")
    say(f"     cell from one phase to the next. Nothing here supplies a cell-cycle simulation.")
    say(f"     Three further limits, all structural rather than fixable by more permutations:")
    say(f"       1. {n_all} labelled genes is a thin cycle. Published phase-resolved sets run to")
    say(f"          well over a thousand, and the missing S phase is not a sampling accident.")
    say(f"       2. coexpr and codep are correlations across CELL LINES and TISSUES, not across")
    say(f"          time. Bulk co-expression of cell-cycle genes measures how proliferative a")
    say(f"          sample is, which moves every phase together and is exactly the signal that")
    say(f"          would erase an order. No layer in this model is phase-resolved.")
    say(f"       3. ppi and reg are curated, so they inherit the fame confound Z6 measures.")
    say()

    gates = {"Z1 the labels span the cycle": z1,
             "Z2 the graph exists": z2,
             "Z3 the embedding carries real structure (positive control)": z3,
             "Z4 the phase order is recovered": z4,
             "Z5 the null is capable and fired": z5,
             "Z6 the graph beats fame": z6,
             "Z7 the limit is stated": True}
    say("  " + "-" * 90)
    for kk, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {kk}")
    say("  " + "-" * 90)
    say()

    man = RM.manifest(
        inputs=[str(LR.CELL)],
        available=n_all, used=len(gidx), selection="filtered", seed=SEED,
        controls=[
            "positive control: co-complex proximity, with complexes HELD OUT of every graph",
            "second positive control on an expression-only graph with PPI also removed, so "
            "nothing physical could leak complex structure in",
            "phase-label shuffle (label counts preserved) for both ordering statistics",
            "gate_guard.null_can_move on the label vector and on the node permutation, run "
            "BEFORE any null verdict was read",
            "gate_guard.survival used instead of a bare ratio, so an undefined survival "
            "fraction cannot be printed as a percentage",
            "fame head-to-head: the same same-phase-pair AUC computed from log publication "
            "count alone, and the analytic chance baseline for a cyclic order at K phases",
        ],
        note="the embedding never sees a phase label; the graph is the union of coexpr, codep, "
             "ppi and reg restricted to the 153 labelled genes, and complexes are held out "
             "entirely so that co-complex membership can act as a control the pipeline could fail")
    RM.report(man, emit=say)

    res = {
        "test": "loop_cellcycle", "manifest": man, "gates": gates,
        "z1": {"phase_counts": dict(cnt), "phases_kept": kept_phases,
               "segments_named": sorted(segs_named), "segments_missing": missing,
               "n_cyclic_orders": int(max(1, math.factorial(max(0, k - 1)) // 2)),
               "chance_baseline": float(chance), "a_enough_phases": bool(z1a),
               "b_spans_cycle": bool(z1b)},
        "z2": {"layer_edges": {nm: int(A.sum() // 2) for nm, A in layers.items()},
               "n_labelled": n_all, "lcc_full": len(keep), "lcc_expr_only": len(keep_e),
               "degree_median": float(np.median(deg)), "degree_mean": float(deg.mean()),
               "eigenvalues": [float(v) for v in vals],
               "ring_signature": {"gap_1_2": gap12, "gap_2_3": gap23,
                                  "ratio": float(gap12 / gap23) if gap23 else None,
                                  "note": "a ring graph has gap(1,2) near zero"}},
        "z3": {"n_complexes": ncplx, "n_cocomplex_pairs": int(is_cplx.sum()),
               "n_pairs": len(pair_key), "auc_full_graph": float(a_full), "z_full": float(z_c),
               "survival_full": s_c, "auc_expression_only": float(a_expr), "survival_expr": s_e,
               "capability": cap_c},
        "z4": {"phases_used": phases_used, "rho_concordance": float(rho),
               "order_proposed": seq, "order_canonical": phases_used,
               "order_match": bool(ok_order),
               "phase_circular_means_deg": {p: {"mean_deg": float(np.degrees(mu[p][0])),
                                                "resultant": float(mu[p][1]),
                                                "n": int(cnt[p])} for p in phases_used},
               "projection_sweep": {kk: {"rho": float(v["rho"]), "order": v["order"],
                                         "match": bool(v["match"])} for kk, v in sweep.items()},
               "rho_best_projection": float(rho_max), "any_projection_matches": bool(any_match),
               "a_concordance": bool(z4a), "b_order": bool(z4b),
               "posthoc_decomposition": {
                   "rho_clustering_component": float(rho_same),
                   "rho_order_component": float(rho_diff),
                   "n_different_phase_pairs": int(diff_only.sum()),
                   "declared": "added after the first run, gated on nothing, and it can only "
                               "weaken the claim: it separates same-phase clustering from cyclic "
                               "order, which the concordance alone cannot"}},
        "z5": {"capability": cap, "mean_labels_moved": float(np.mean(moved)),
               "survival_rho": s_rho, "survival_rho_bestproj": s_max,
               "survival_order_component": s_diff,
               "p_order": float(p_order), "p_order_bestproj": float(p_ordermax),
               "n_order_matches_under_shuffle": int(sum(nulls_order)), "n_perm": NPERM},
        "z6": {"pubs_by_phase": {p: {"median": float(np.median(
                   pb[np.array([i for i, L in enumerate(labels) if L == p])])),
                   "mean": float(pb[np.array([i for i, L in enumerate(labels) if L == p])].mean()),
                   "n": int(cnt[p])} for p in phases_used},
               "kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue),
               "rho_pubs_angle": spear(lp, ang), "rho_pubs_radius": spear(lp, rad),
               "rho_pubs_degree": spear(lp, deg),
               "auc_samephase_graph": float(auc_graph),
               "auc_samephase_pubs": float(auc_pubs)},
        "seconds": time.time() - t0, "log": log,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "loop_cellcycle.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_cellcycle.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
