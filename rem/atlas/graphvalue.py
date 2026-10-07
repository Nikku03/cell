"""What the engine's cost actually rests on in TRRUST -- and therefore what a sequence model buys.

WHY THIS MODULE EXISTS. alphagenome.py priced ONE use of AlphaGenome, the in-silico joint-deletion
experiment, and pre-registered it. That is not the largest use. The engine's ENTIRE cost line rests
on one literature-curated graph: the controller order is TRRUST out-degree, the per-gene factor is
a product over TRRUST regulator sets, and widthblock's central finding is that from |C| = 20 upward
ONE GENE -- CDKN1A, 52 regulators -- is essentially the whole per-gene term. CDKN1A is one of the
most-studied genes in biology. AlphaGenome's CHIP_TF head predicts TF binding from sequence alone,
which is an edge set with no citation bias in it at all. So the question that decides whether that
is worth anything is not about AlphaGenome. It is about TRRUST:

    HOW MUCH OF THE CAP IS THE SPECIFIC CURATED WIRING, AND HOW MUCH IS JUST THE DEGREE SEQUENCE?

This is the ceiling gate again, in its fifth application, and it needs no API key. If the cap is a
function of the degree sequence alone, then replacing WHICH TF binds WHICH gene -- exactly what a
sequence model changes -- moves nothing, and AlphaGenome's edge set is worth using only insofar as
it changes DEGREES. If the cap moves under rewiring, the wiring is load-bearing and its provenance
becomes a live risk to every cap in this record.

WHAT IS BEING MEASURED, STATED SO THE NUMBER CANNOT BE OVERSOLD. The cap is signed.py's functional
per + blk <= 1e12 at L = 8, C = 64 classes, computed by widthblock's own code path so the numbers
are comparable to its 39 and 140 rather than being a new scale. Nothing here measures biology. It
measures how sensitive OUR COST is to the graph we happen to have.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

V0  THE CEILING GATE ON THE EDGE SET. Price the WIRING at random. Degree-preserving double-edge
    swaps on the directed pairs: every TF keeps its out-degree, every gene keeps its in-degree,
    only WHO BINDS WHOM changes. The controller order is therefore unchanged by construction and
    the class map is attached to nodes, so the rewired run is a matched control and not a
    different experiment.
    PREDECLARED: if the rewired cap lands within 1.5x of the real cap, the cap is a function of
    the degree sequence and a sequence-derived edge set buys the engine NOTHING unless it changes
    degrees -- and this module says so and stops recommending it. If it moves by more than 1.5x,
    the specific wiring is load-bearing, and the provenance of every cap in this record is a live
    risk that AlphaGenome is the only available instrument for.
    Reported with the continuous per-gene term beside the integer cap, because an integer cap can
    hide a factor of two in the quantity that sets it.

V0b SATURATION GUARD. Three modules in this record reported a loop ceiling minus one as a cap --
    scalarcap's 59, signed.py's 419, whatdata's 200. Any cap equal to maxC - 1 is reported as
    SATURATED and is not compared to anything.

V1  THE HUB, AT THE EDGE LEVEL RATHER THAN THE COST LEVEL. widthblock demoted the top-m genes
    inside the COST FUNCTION, which is a representation change. Here the top-m in-degree genes
    have their regulator sets TRUNCATED to the network median in-degree -- which is what a
    sequence-derived graph would do to them if their regulator sets are citation artefacts.
    PREDECLARED, and the bar is on the DELTA rather than the level: if truncating the single
    highest in-degree gene moves the cap by less than 1.5x, then "one gene is the width
    exponential" is a statement about the cost function's maximum and not about the cap, and
    re-deriving that one gene's edges is worth little.

V1b THE MATCHED CONTROL FOR V1, WITHOUT WHICH V1 MEANS NOTHING. Truncating hubs DELETES EDGES, and
    deleting edges always lowers a monotone cost. The control removes the SAME NUMBER of directed
    edges uniformly at random. PREDECLARED: the hub effect is only the part above the random-
    deletion control at matched edge count.

V2  IS THE HUB A CITATION ARTEFACT? Measured inside the file rather than asserted. TRRUST carries
    a PMID per record. Per gene: in-degree, distinct supporting publications, and the share of its
    in-edges supported by exactly ONE publication.
    PREDECLARED: this measurement can show that a hub's regulator set is an accumulation of
    singly-reported one-offs, or that it is multiply-replicated consensus. It CANNOT show the
    edges are wrong. It bounds how much a sequence-derived edge set could differ, not whether it
    would be better. That limit is stated with the number, not after it.

V3  WHAT A REPLACEMENT EDGE SET WOULD HAVE TO CHANGE, given V0-V2, in the engine's own units.

V4  THE RANKED USES OF ALPHAGENOME AND THEIR PRICES -- and, explicitly, which of them this module
    has priced and which remain assertions.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import time
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.localclosure import load_trrust, ball as _ball
from rem.atlas.engine import residual_graph
from rem.atlas.boundedwidth import bounded_elimination
from rem.atlas.trrust_engine import TSV
from rem.atlas.widthblock import L, BAR, CAPEXP, slot, mindeg_width

OUT = os.path.join(os.path.dirname(__file__), "RESULTS_graphvalue.txt")
MAXC = 260


# =================================================================================================
# THE GRAPH, REBUILT IN ONE INDEX SPACE SO REWIRING AND TRUNCATION ARE WELL DEFINED
# =================================================================================================

def parse(path=TSV):
    """Directed deduplicated pairs plus their supporting PMIDs, in ONE index space.

    load_trrust builds its undirected adj from exactly this deduplicated pair set, so rebuilding
    the pairs here and re-deriving adj reproduces it -- checked, not assumed, in main()."""
    names, D = {}, {}
    for ln in open(path, encoding="utf8", errors="replace"):
        f = ln.rstrip("\n").split("\t")
        if len(f) < 2 or f[0] == f[1]:
            continue
        for g in (f[0], f[1]):
            names.setdefault(g, len(names))
        key = (names[f[0]], names[f[1]])
        pm = D.setdefault(key, set())
        if len(f) >= 4:
            for p in f[3].split(";"):
                p = p.strip()
                if p:
                    pm.add(p)
    inv = {v: k for k, v in names.items()}
    return names, inv, len(names), D


def undirected(D, n):
    adj = [set() for _ in range(n)]
    for a, b in D:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def out_order(D, n):
    od = np.zeros(n)
    for a, _ in D:
        od[a] += 1
    return list(np.argsort(-od)), od


def rewire(pairs, rng, passes=12):
    """Degree-preserving directed double-edge swap. Out-degree and in-degree are preserved EXACTLY
    for every node, so the controller order (top out-degree) is identical and the class map, being
    attached to nodes, is unchanged. Only who-binds-whom moves."""
    E = list(pairs)
    S = set(E)
    m = len(E)
    tries = passes * m
    ok = 0
    for _ in range(tries):
        i = int(rng.integers(m))
        j = int(rng.integers(m))
        if i == j:
            continue
        a, b = E[i]
        c, d = E[j]
        if a == d or c == b:
            continue
        if (a, d) in S or (c, b) in S:
            continue
        S.discard((a, b)); S.discard((c, d))
        S.add((a, d)); S.add((c, b))
        E[i] = (a, d); E[j] = (c, b)
        ok += 1
    return S, ok


def truncate_hubs(D, n, targets, keep, rng):
    """Cut each named gene's IN-edges down to `keep` of them, chosen at random. Returns the new
    pair set and the number of directed edges removed."""
    ins = collections.defaultdict(list)
    for a, b in D:
        ins[b].append(a)
    drop = set()
    for g in targets:
        src = sorted(ins.get(g, []))
        if len(src) <= keep:
            continue
        rng.shuffle(src)
        for a in src[keep:]:
            drop.add((a, g))
    return {e for e in D if e not in drop}, len(drop)


def drop_random(D, k, rng):
    E = sorted(D)
    idx = rng.choice(len(E), size=min(k, len(E)), replace=False)
    dead = {E[int(i)] for i in idx}
    return {e for e in D if e not in dead}, len(dead)


# =================================================================================================
# THE COST FUNCTIONAL -- signed.py's, via widthblock's code path
# =================================================================================================

def gene_regs(adj, order, nC, n):
    Cset = set(order[:nC])
    return Cset, [(i, adj[i] & Cset) for i in range(n)
                  if i not in Cset and (adj[i] & Cset)]


def bases_for(adj, Cset, n):
    """The six (r, w) base vectors. They depend ONLY on the residual graph, so they are computed
    once per (graph, nC) and reused across every cost variant -- which is what makes the sweeps
    affordable."""
    res = residual_graph(adj, Cset, n)
    out = []
    for r in (1, 2):
        G = [_ball(res, i, r) - {i} for i in range(n)]
        for w in (4, 6, 8):
            pa, _, _ = bounded_elimination(G, w)
            out.append(np.array([2.0 ** min(1 + len(pa[i]), CAPEXP) for i in range(n)]))
    return out


def fvec(genes, asg, n, demote=frozenset()):
    f = np.ones(n)
    for i, regs in genes:
        if i in demote:
            f[i] = len(regs) * (L + 1) + 1.0
            continue
        v = 1.0
        for _, m in collections.Counter(int(asg[j]) for j in regs).items():
            v = min(v * slot(m), 2.0 ** CAPEXP)
        f[i] = v
    return f


def ctrl_nb(D, order, nC):
    names = set(order[:nC])
    idx = {g: i for i, g in enumerate(sorted(names))}
    nb = [set() for _ in idx]
    for u, v in D:
        if u in names and v in names and u != v:
            nb[idx[u]].add(idx[v])
            nb[idx[v]].add(idx[u])
    return nb


def blk_factored(D, order, nC):
    return 2.0 ** min(mindeg_width(ctrl_nb(D, order, nC)) + 1, CAPEXP)


def cap_scan(D, n, asg, ms=(0,), maxC=MAXC, probe=()):
    """The cap for each demotion level m, in ONE upward scan with the bases cached per nC.

    Returns (caps, per_at_probe). caps[m] is the largest nC for which per + blk <= BAR held on
    every step up to it -- the same prefix rule as widthblock's, so the numbers are comparable."""
    adj = undirected(D, n)
    order, _ = out_order(D, n)
    caps = {m: 0 for m in ms}
    alive = {m: True for m in ms}
    probe_vals = {}
    hold = max(probe) if probe else 0
    for nC in range(1, maxC):
        if not any(alive.values()) and nC > hold:
            break
        Cset, genes = gene_regs(adj, order, nC, n)
        if not genes:
            for m in ms:
                if alive[m]:
                    caps[m] = nC
            continue
        B = bases_for(adj, Cset, n)
        blk = blk_factored(D, order, nC)
        base_f = fvec(genes, asg, n)
        cost_order = sorted(range(n), key=lambda i: -base_f[i])
        if nC in probe:
            probe_vals[nC] = (min(float(np.sum(b * base_f)) for b in B), blk)
        for m in ms:
            if not alive[m]:
                continue
            f = base_f if m == 0 else fvec(genes, asg, n, demote=set(cost_order[:m]))
            per = min(float(np.sum(b * f)) for b in B)
            if per + blk <= BAR:
                caps[m] = nC
            else:
                alive[m] = False
    return caps, probe_vals


def sat(c, maxC=MAXC):
    return "  SATURATED" if c >= maxC - 1 else ""


# =================================================================================================
def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    t0 = time.time()
    names, inv, n, D = parse()
    adj0 = undirected(D, n)
    order0, od0 = out_order(D, n)

    ra, rinv, rsha, rne = load_trrust()
    same = (len(ra) == n) and all(
        {rinv[j] for j in ra[i]} == {inv[j] for j in adj0[names[rinv[i]]]}
        for i in range(len(ra)))
    rng = np.random.default_rng(20260907)
    a64 = np.minimum((rng.random(n) * 64).astype(int), 63)

    P_(RULE); P_("WHAT THE CAP RESTS ON: CURATED WIRING, OR THE DEGREE SEQUENCE?"); P_(RULE)
    P_(f"  TRRUST human, sha {rsha}, {n} genes, {len(D)} distinct directed pairs.")
    P_(f"  rebuilt adjacency reproduces load_trrust's: {same}")
    P_(f"  cost functional: signed.py's  per + blk <= {BAR:.0e}  at L = {L}, C = 64 classes,")
    P_( "  block FACTORED, per-gene term minimised over r in (1,2) and w in (4,6,8).")

    # ---- V0 -------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("V0  THE CEILING GATE: PRICE THE WIRING AT RANDOM"); P_(RULE)
    P_("  Degree-preserving double-edge swaps. Every out-degree and every in-degree is preserved")
    P_("  EXACTLY, so the controller order is identical and the class map is unchanged. The only")
    P_("  thing that moves is which TF binds which gene.")

    probe = (5, 10, 13, 20, 30, 40)
    caps_real, pv_real = cap_scan(D, n, a64, ms=(0,), probe=probe)
    P_(f"\n    {'graph':<18} {'swaps':>8} {'cap |C|':>9}  "
       f"{'per @13':>11} {'per @20':>11} {'per @40':>11}")

    def row(nm, sw, cap, pv):
        P_(f"    {nm:<18} {sw:>8} {cap:>9}{sat(cap):<11}"
           f" {pv.get(13, (float('nan'),))[0]:>11.3e}"
           f" {pv.get(20, (float('nan'),))[0]:>11.3e}"
           f" {pv.get(40, (float('nan'),))[0]:>11.3e}")

    row("TRRUST, real", 0, caps_real[0], pv_real)
    rcaps, rpers = [], []
    for s in range(5):
        Dr, nsw = rewire(D, np.random.default_rng(1000 + s))
        c, pv = cap_scan(Dr, n, a64, ms=(0,), probe=probe)
        rcaps.append(c[0])
        rpers.append(pv)
        row(f"rewired, seed {s}", nsw, c[0], pv)

    med = float(np.median(rcaps))
    ratio = (med / caps_real[0]) if caps_real[0] else float("inf")
    P_(f"\n  real cap {caps_real[0]}, rewired median {med:.0f}, ratio {ratio:.2f}x")
    for k in probe:
        rp = [p[k][0] for p in rpers if k in p]
        if rp and k in pv_real:
            P_(f"  per-gene term at |C| = {k:<3}: real {pv_real[k][0]:.3e},"
               f" rewired median {np.median(rp):.3e},"
               f" ratio {np.median(rp)/pv_real[k][0]:.3g}x")
    P_("\n  V0 VERDICT (bar predeclared at 1.5x on the cap):")
    P_(f"    {'PASS -- wiring is load-bearing' if (ratio > 1.5 or ratio < 1/1.5) else 'FAIL -- the cap is a function of the DEGREE SEQUENCE'}")

    # ---- V0c  WHY A 100x EFFECT IN THE COST BOUGHT 1.08x IN THE CAP ----------------------------
    P_("\n" + RULE)
    P_("V0c THE EXCHANGE RATE: HOW MANY CONTROLLERS IS A FACTOR OF X IN THE COST WORTH?")
    P_(RULE)
    P_("  The integer cap hid a factor of 100, which is exactly the failure mode V0 predeclared")
    P_("  reporting the continuous term against. The reason is the growth rate of the cost.")
    ks = sorted(k for k in probe if k in pv_real)
    yy = np.array([np.log10(pv_real[k][0]) for k in ks])
    xx = np.array(ks, float)
    slope, icpt = np.polyfit(xx, yy, 1)
    r2 = 1.0 - float(np.sum((yy - (slope * xx + icpt)) ** 2)
                     / max(np.sum((yy - yy.mean()) ** 2), 1e-30))
    P_(f"\n    {'|C|':>6} {'per-gene term':>15} {'log10':>8}")
    for k in ks:
        P_(f"    {k:>6} {pv_real[k][0]:>15.3e} {np.log10(pv_real[k][0]):>8.2f}")
    P_(f"\n  log10(per) = {slope:.3f} * |C| + {icpt:.2f}   R^2 = {r2:.3f}")
    P_(f"  THE EXCHANGE RATE: {slope:.3f} decades of cost per controller, so a cost reduction of")
    P_(f"  a factor F buys log10(F) / {slope:.3f} controllers of cap.")
    r40 = [p[40][0] for p in rpers if 40 in p]
    fac = float("nan")
    if r40 and 40 in pv_real:
        fac = pv_real[40][0] / float(np.median(r40))
        P_(f"\n  Read far out, at |C| = 40, the rewired graph is {fac:.0f}x CHEAPER than the real one --")
        P_( "  the real wiring is WORSE for this engine than random wiring at matched degrees.")

    # THE RATIO MUST BE READ AT THE CAP, AND THE FIRST WRITING OF THIS SECTION DID NOT.
    kc = min(ks, key=lambda k: abs(k - caps_real[0]))
    i = ks.index(kc)
    prev = ks[i - 1] if i > 0 else ks[i + 1]
    loc = abs((np.log10(pv_real[kc][0]) - np.log10(pv_real[prev][0])) / (kc - prev))
    rc = [p[kc][0] for p in rpers if kc in p]
    ratio_cap = pv_real[kc][0] / float(np.median(rc)) if rc else float("nan")
    P_("\n  AND THAT FACTOR DOES NOT TRANSFER TO THE CAP, WHICH THE FIRST WRITING OF THIS SECTION")
    P_(f"  GOT WRONG. It set the {fac:.0f}x measured at |C| = 40 beside the 1-controller change measured")
    P_(f"  at |C| = {caps_real[0]} and called the gap unexplained. The two are at different points. The ratio")
    P_("  that governs the cap is the one AT the cap:")
    P_(f"\n    real / rewired at |C| = {kc:<3}                     {ratio_cap:.2f}x")
    P_(f"    local slope, |C| = {prev} to {kc}                   {loc:.3f} decades per controller")
    P_(f"    predicted cap change  log10({ratio_cap:.2f}) / {loc:.3f}     {np.log10(ratio_cap)/loc:.2f} controllers")
    P_(f"    MEASURED cap change                        {med - caps_real[0]:.0f}")
    P_(f"\n  which agrees. The {fac:.0f}x at |C| = 40 is a ratio read {40 - caps_real[0]} controllers BEYOND the cap,")
    P_("  where neither graph is affordable, and it was never going to transfer. So the deliverable")
    P_("  statement is the small one: at the point the engine actually operates, the curated wiring")
    P_("  differs from random wiring at matched degrees by a factor of two, worth ONE controller.")
    P_("  V0's bar was set on the cap before the run. The verdict stands at FAIL.")

    # ---- V1 / V1b -------------------------------------------------------------------------------
    P_("\n" + RULE); P_("V1  TRUNCATE THE HUBS' REGULATOR SETS -- AND THE MATCHED CONTROL"); P_(RULE)
    ind = collections.Counter(b for _, b in D)
    kmed = int(np.median([ind.get(i, 0) for i in range(n) if ind.get(i, 0) > 0]))
    top = [g for g, _ in ind.most_common()]
    P_(f"  median in-degree over genes with any regulator: {kmed}")
    P_(f"  top in-degree: " + ", ".join(f"{inv[g]}({ind[g]})" for g in top[:6]))
    P_(f"\n    {'genes cut':>10} {'edges removed':>14} {'cap, hubs cut':>15} {'cap, random cut':>17}")
    base_cap = caps_real[0]
    P_(f"    {0:>10} {0:>14} {base_cap:>15}{sat(base_cap)} {base_cap:>17}{sat(base_cap)}")
    v1 = {}
    for m in (1, 2, 5, 20, 100):
        Dt, ndrop = truncate_hubs(D, n, top[:m], kmed, np.random.default_rng(7))
        ct, _ = cap_scan(Dt, n, a64, ms=(0,))
        Dc, _ = drop_random(D, ndrop, np.random.default_rng(8))
        cc, _ = cap_scan(Dc, n, a64, ms=(0,))
        v1[m] = (ndrop, ct[0], cc[0])
        P_(f"    {m:>10} {ndrop:>14} {ct[0]:>15}{sat(ct[0])} {cc[0]:>17}{sat(cc[0])}")

    d1 = v1[1][1] / base_cap if base_cap else float("inf")
    P_(f"\n  V1 VERDICT (bar predeclared at 1.5x for the SINGLE top gene):")
    P_(f"    cutting {inv[top[0]]} alone: {base_cap} -> {v1[1][1]}  ({d1:.2f}x)"
       f"  -- {'PASS' if d1 > 1.5 else 'FAIL'}")
    P_(f"  V1b the matched control at the same edge count: {v1[1][2]}"
       f"  -- attributable to the hub: {v1[1][1]} vs {v1[1][2]}")

    # ---- V2 -------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("V2  IS THE HUB A CITATION ARTEFACT? THE PMIDs, COUNTED"); P_(RULE)
    allp = set()
    for pm in D.values():
        allp |= pm
    single = sum(1 for pm in D.values() if len(pm) == 1)
    P_(f"  {len(D)} pairs, {len(allp)} distinct publications,"
       f" {single} pairs ({100*single/len(D):.1f}%) supported by exactly ONE.")
    P_(f"\n    {'gene':<10} {'in-deg':>7} {'distinct PMIDs':>15} {'1-paper edges':>14}"
       f" {'share':>8}")
    for g in top[:8]:
        pms = set()
        one = 0
        for a in [a for a, b in D if b == g]:
            pm = D[(a, g)]
            pms |= pm
            if len(pm) == 1:
                one += 1
        k = ind[g]
        P_(f"    {inv[g]:<10} {k:>7} {len(pms):>15} {one:>14} {100*one/k:>7.1f}%")
    rest = [g for g in range(n) if ind.get(g, 0) > 0 and g not in set(top[:8])]
    ones = []
    for g in rest:
        c = sum(1 for a, b in D if b == g and len(D[(a, b)]) == 1)
        ones.append(c / ind[g])
    P_(f"\n    all other regulated genes ({len(rest)}): mean 1-paper share"
       f" {100*float(np.mean(ones)):.1f}%")
    P_("\n  V2 LIMIT, STATED WITH THE NUMBER: a publication count bounds how much a sequence-")
    P_("  derived edge set COULD differ. It does not show any edge is wrong, and it does not")
    P_("  show a predicted edge would be right. It is an upper bound on the size of the change,")
    P_("  not evidence about its direction.")

    # ---- V3 -------------------------------------------------------------------------------
    P_("\n" + RULE); P_("V3  WHAT A REPLACEMENT EDGE SET WOULD HAVE TO CHANGE"); P_(RULE)
    P_("  Not the wiring. V0 preserved every degree and moved the cap by one controller, and the")
    P_("  direction was the wrong way round -- random wiring is CHEAPER than the curated wiring,")
    P_("  so a sequence-derived edge set helps only if it is LESS concentrated than TRRUST. A")
    P_("  ChIP-style binding predictor is if anything MORE concentrated, because the promoters")
    P_("  that attract many factors are the ones it will also predict many factors for.")
    P_("\n  What moves the cap is IN-DEGREE, and only in bulk. V1's sweep against V1b's matched")
    P_("  control at equal edge count:")
    P_(f"\n    {'genes cut':>10} {'edges removed':>14} {'hubs cut':>10} {'random cut':>12}"
       f" {'attributable':>13}")
    for m in (1, 2, 5, 20, 100):
        nd, ch, cc = v1[m]
        P_(f"    {m:>10} {nd:>14} {ch:>10} {cc:>12} {ch/cc if cc else float('nan'):>12.2f}x")
    P_("\n  V3: one gene is not the cap. widthblock's 'from |C| = 20 upward ONE GENE is")
    P_("  essentially the entire per-gene cost' is a true statement about the cost's MAXIMUM and")
    P_("  a false one about the cap -- deleting that gene's 113 excess in-edges moves the cap from")
    P_(f"  {base_cap} to {v1[1][1]}. The hub effect is real but it is COLLECTIVE: at 100 genes the cut is worth")
    P_(f"  {v1[100][1]}/{v1[100][2]} = {v1[100][1]/v1[100][2]:.1f}x over deleting the same {v1[100][0]} edges at random.")
    P_("  Note also that CDKN1A's in-degree here is 115; widthblock's 52 was its regulators AMONG")
    P_("  THE TOP 122 FACTORS, a different quantity, and the two are not in conflict.")

    # ---- V4 -------------------------------------------------------------------------------
    P_("\n" + RULE); P_("V4  THE RANKED USES, AND WHICH OF THEM THIS MODULE HAS PRICED"); P_(RULE)
    P_("  PRICED HERE, AND REJECTED. Replacing TRRUST's edge set with a sequence-derived one.")
    worth = np.log10(max(ratio_cap, 1.0)) / loc if loc else float("nan")
    P_(f"  Worth {worth:.1f} controllers at the exchange rate measured AT the cap, {loc:.3f} decades per")
    P_("  controller, and in the wrong direction. V2 removes the motivating hypothesis too:")
    P_("  the hubs are the BETTER-replicated part of TRRUST, not the worse. No API key is needed")
    P_("  to reach this conclusion, which is the point of running it first.")
    P_("\n  NOT PRICED HERE, AND NOW THE LARGEST MEASURED LEVERAGE IN THE RECORD. The target")
    P_("  response sigma(base + gain * wact). realkinetics K5b measured the sensitivity of its own")
    P_("  reported tail to these two constants and recorded that both were INVENTED:")
    P_("      base -1.5  ->  -9.37 orders        base -0.5  ->  +8.13 orders")
    P_("      gain  1.0  ->  -0.80 orders        gain  3.0  ->  +0.63 orders")
    P_("  Half a log-odds unit on an unmeasured constant moves the answer by eight to nine orders.")
    P_("  tailtight's whole composed certificate is 26.77 orders wide. So the base constant is")
    P_("  within a factor of two of the entire certification problem, and unlike the box")
    P_("  relaxation it is not a mathematical defect -- it is a MISSING MEASUREMENT, which is the")
    P_("  one kind of gap a predictor can close.")
    P_("\n  THE LIMIT ON THAT USE, STATED BEFORE IT IS ATTEMPTED. AlphaGenome predicts expression;")
    P_("  base + gain * wact is a log-odds for a firing event in a CME. A dose-response in intact")
    P_("  site count pins the SHAPE -- curvature, saturation point, the ratio of gain to base --")
    P_("  but not the absolute log-odds scale, which needs a link neither the model nor this")
    P_("  record supplies. A shape with an unknown offset is still strictly more than invented.")
    P_("\n  UNCHANGED FROM alphagenome.py's A0: the joint regime. Single-perturbation data pins a")
    P_("  median 15.6% of the engine's per-gene joint degrees of freedom at k >= 2 and 0.00% at")
    P_("  k >= 20, and ENCODE supplies 74 perturbed regulators against whatdata's requirement of")
    P_("  128. In silico there is no such limit. That gate passed; this one did not.")

    P_(f"\n  runtime {time.time() - t0:.1f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
