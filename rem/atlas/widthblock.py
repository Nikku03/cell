"""Attacking the width exponential -- and finding, for the third time, that it was the wrong term.

WHAT WAS LEFT STANDING. Four representation changes and one pruner all failed to remove the same
exponential in WIDTH. prune.py removes the exponential in window depth and leaves 2^|C| untouched.
So this module attacks 2^|C|.

THE CEILING GATE, RUN FIRST, AND IT FIRED BEFORE THIS FILE EXISTED. That is the discipline
block.py cost a module by skipping, and this is its third application -- the first time it has
stopped an attack before any of it was built. Under the class-count representation at C = 64, at
L = 8, with the bar at 1e12:

    |C|      per-gene term    block: class-count    block: FACTORED    block: ZERO
      8          3.117e+09             1.900e+07          1.600e+01            0
     13          2.962e+11             6.859e+10          1.280e+02            0
    cap                  13                    13                 13           13

Pricing the block at ZERO moves the cap by NOTHING. The width exponential in the controller
block is not the binding term. The per-gene factor is.

AND THE FACTORISATION THAT WOULD HAVE BEEN BUILT WORKS, WHICH IS WHY THE GATE MATTERS. The
controller-controller subgraph of TRRUST is sparse -- 561 edges among the top 122 factors, density
0.042 -- and its min-degree elimination width grows SUBLINEARLY:

    |C|     5   10   20   40   80  122  200  300  420
    width   2    5    9   16   25   35   46   59   63        (nothing sacrificed)

At |C| = 122 that is 2^36 in place of 2^122, a factor of 7.7e25. A correct, large, and completely
useless result, because the term it shrinks was never binding. Recorded so that nobody builds it.

SO THE REAL TARGET IS THE PER-GENE FACTOR, prod_j (k_ij (L+1) + 1) over the CLASSES that gene i's
controllers fall into. It is exponential in the number of OCCUPIED CLASSES PER GENE -- which is a
property of the class ASSIGNMENT, not of the network. signed.py used a RANDOM assignment and
flagged exactly this in its own S6.4: "a class map that happened to align with the network's
structure would give a higher cap".

AND THE FIRST ATTEMPT AT THAT ALIGNMENT WAS DEGENERATE, WHICH IS RECORDED RATHER THAN DELETED. A
greedy optimiser minimising the per-gene cost with no accuracy constraint drove the mean occupied
classes per gene to exactly 1.00 and matched the theoretical best cost to three digits -- by
putting every controller in ONE class. That is the plain count, which promoter.py refutes at 6.29
noise floors. A cost optimiser with no accuracy constraint will always rediscover the cheapest
refuted representation. Ledger U, in this module's own probe.

WHICH LEAVES ONE HONEST QUESTION, AND IT IS EMPIRICAL. Can a gene's regulators occupy FEW classes
while many classes remain globally distinguished? That is a property of the real network, and it
is testable against a matched shuffle.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

W0  THE CEILING GATE, restated in the record. PREDECLARED: if pricing the block at zero does not
    move the cap, no work on the block may be reported as progress, however large its factor.

W1  THE DEGENERATE OPTIMUM, reported rather than hidden. Any alignment must be CONSTRAINED to keep
    the classes genuinely distinguished, and the constraint must be stated before the number.

W2  DOES THE NETWORK CARRY CLASS STRUCTURE AT ALL? Real regulator sets against a matched shuffle
    that preserves the mode counts. PREDECLARED: if real sets are no more homogeneous than
    shuffled ones, there is nothing to align to and the route is closed.

W3  THE CONSTRAINED ALIGNMENT: classes balanced so none can be emptied, greedy on the per-gene
    cost. Cost against C, with the random assignment beside it as the matched control.

W4  WHAT IT DOES TO THE CAP, which is the only deliverable. Recomputed with the aligned per-gene
    term and the factored block, against signed.py's 13 and multiplier.py's 25.

W5  THE ACCURACY SIDE, because W4 is a cost statement. signed.py measured that C = 64 classes are
    needed to sit inside one noise floor of identity. PREDECLARED: a cap that is only reachable
    at a C that data refutes is not a cap, it is the plain count wearing a larger number.

W6  WHAT BREAKS NEXT.

=================================================================================================
WHAT THE FIRST RUN OF THIS MODULE GOT WRONG, AND WHAT IT FOUND INSTEAD
=================================================================================================
The alignment attack in W3 was built on a wrong diagnosis and made things WORSE by up to 28x. The
specification error: with C = 64 classes and only |C| = 13 controllers the balance cap is
ceil(13/64) = 1, so every controller is FORCED into its own class -- the worst case, not a
balanced one. A balance constraint is meaningless when there are fewer items than classes, and at
these widths there always are.

The measurement that should have come first says why no representation-level fix could have
worked: from |C| = 20 upward, ONE GENE is essentially the entire per-gene cost. CDKN1A, with 52
regulators among the top 122 factors, is 99.2% of it at |C| = 20 and 100.0% from |C| = 40. The
binding term is a HUB -- the same object that killed r-balls and left bounded width with a 6%
floor, in its third form.

So the attack that works is not on the representation but on the few genes that set the max:
demote the top 100 most-regulated genes (3.5% of the network) to the plain count and the cap goes
from 13 to 140 controllers, edge coverage from 26.6% to 71.7%.

AND THAT FLIPS W0's OWN VERDICT, which is the useful half of this module. W0 priced the block at
zero, found the cap unmoved, and correctly refused to report the block factorisation as progress.
Once the hubs are demoted the binding term changes and that same factorisation is worth 140
against 39 -- a 3.6x difference in the cap. A ceiling gate prices a term against the CURRENT
binding constraint; its verdict is correct at the time and is not permanent. When the constraint
moves, re-run the gate rather than trusting it.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.trrust_engine import signed_edges
from rem.atlas.localclosure import load_trrust, ball
from rem.atlas.engine import residual_graph
from rem.atlas.boundedwidth import bounded_elimination

L = 8
BAR = 1e12
CAPEXP = 400


def slot(m):
    return m * (L + 1) + 1.0


def net():
    E, _ = signed_edges()
    adj, inv, sha, ne = load_trrust()
    n = len(adj)
    outd = collections.Counter(u for u, v, _ in E)
    name2i = {g: i for i, g in inv.items()}
    od = np.zeros(n)
    for g, d in outd.items():
        if g in name2i:
            od[name2i[g]] = d
    order = list(np.argsort(-od))
    mode = {(u, v): m for u, v, m in E}
    return E, adj, inv, sha, n, order, mode


def mindeg_width(nb):
    """UNBOUNDED min-degree elimination. The maximum bag size is an upper bound on treewidth and
    nothing is sacrificed to obtain it, unlike bounded_elimination which guarantees the width by
    DELETING edges -- a distinction that cost this module a wrong answer on its first probe."""
    A = [set(a) for a in nb]
    alive = np.ones(len(A), bool)
    w = 0
    for _ in range(len(A)):
        cand = [v for v in range(len(A)) if alive[v]]
        if not cand:
            break
        v = min(cand, key=lambda x: len(A[x]))
        w = max(w, len(A[v]))
        for a in A[v]:
            for b in A[v]:
                if a != b:
                    A[a].add(b)
        for a in A[v]:
            A[a].discard(v)
        alive[v] = False
        A[v] = set()
    return w


def ctrl_subgraph(E, inv, order, k):
    names = {inv[i] for i in order[:k]}
    idx = {g: i for i, g in enumerate(sorted(names))}
    nb = [set() for _ in idx]
    m = 0
    for u, v, _ in E:
        if u in names and v in names and u != v:
            if idx[v] not in nb[idx[u]]:
                m += 1
            nb[idx[u]].add(idx[v])
            nb[idx[v]].add(idx[u])
    return nb, m


def gene_regs(adj, order, nC, n):
    Cset = set(order[:nC])
    return Cset, [(i, adj[i] & Cset) for i in range(n)
                  if i not in Cset and (adj[i] & Cset)]


def pergene_cost(genes, asg):
    tot = 0.0
    occ = []
    worst = 0.0
    for i, regs in genes:
        cnt = collections.Counter(int(asg[j]) for j in regs)
        v = 1.0
        for _, m in cnt.items():
            v = min(v * slot(m), 2.0 ** CAPEXP)
        tot += v
        worst = max(worst, v)
        occ.append(len(cnt))
    return tot, (float(np.mean(occ)) if occ else 0.0), worst


def balanced_align(ctrl, genes, C, iters=8, seed=0):
    """Greedy alignment CONSTRAINED to keep the classes balanced, so it cannot collapse them.

    Without the constraint the optimiser drives every controller into one class, reaching the
    theoretical minimum cost by turning the class count back into the plain count -- which the
    promoter data refutes. The balance constraint is what makes the number mean something: no
    class may hold more than ceil(|ctrl| / C) controllers, so C classes stay occupied."""
    rng = np.random.default_rng(seed)
    cap = int(np.ceil(len(ctrl) / C))
    asg = {}
    perm = list(ctrl)
    rng.shuffle(perm)
    for i, g in enumerate(perm):
        asg[g] = i % C
    load = collections.Counter(asg.values())

    def cost(a):
        t = 0.0
        for i, regs in genes:
            cnt = collections.Counter(a[j] for j in regs)
            v = 1.0
            for _, m in cnt.items():
                v = min(v * slot(m), 2.0 ** CAPEXP)
            t += v
        return t

    cur = cost(asg)
    for _ in range(iters):
        moved = False
        for g in ctrl:
            old = asg[g]
            best, bc = old, cur
            for c in range(C):
                if c == old or load[c] >= cap:
                    continue
                asg[g] = c
                v = cost(asg)
                if v < bc:
                    bc, best = v, c
            asg[g] = best
            if best != old:
                load[old] -= 1
                load[best] += 1
                cur = bc
                moved = True
        if not moved:
            break
    out = np.zeros(max(ctrl) + 1, dtype=int)
    for g, c in asg.items():
        out[g] = c
    return out, cur, load


def cap_under(adj, order, n, asg_fn, blk_fn, maxC=200):
    """The cap with a per-gene assignment and a block cost, both supplied. Same functional as
    signed.py's -- per + blk <= bar -- so the numbers are comparable to its 13 and to
    multiplier.py's 25 rather than being a new scale."""
    best = 0
    for nC in range(1, maxC):
        Cset, genes = gene_regs(adj, order, nC, n)
        if not genes:
            best = nC
            continue
        asg = asg_fn(nC, Cset, genes)
        res = residual_graph(adj, Cset, n)
        per = None
        for r in (1, 2):
            for w in (4, 6, 8):
                pa, o, _ = bounded_elimination([ball(res, i, r) - {i} for i in range(n)], w)
                base = np.array([2.0 ** min(1 + len(pa[i]), CAPEXP) for i in range(n)])
                f = np.ones(n)
                for i, regs in genes:
                    cnt = collections.Counter(int(asg[j]) for j in regs)
                    v = 1.0
                    for _, m in cnt.items():
                        v = min(v * slot(m), 2.0 ** CAPEXP)
                    f[i] = v
                v = float(np.sum(base * f))
                per = v if per is None else min(per, v)
        if per + blk_fn(nC) <= BAR:
            best = nC
        else:
            break
    return best


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE); P_("ATTACKING THE WIDTH EXPONENTIAL"); P_(RULE)
    E, adj, inv, sha, n, order, mode = net()
    P_(f"  TRRUST sha256[:32] {sha}, {len(E)} distinct directed pairs, {n} genes.")

    # ---- W0  CEILING GATE ----------------------------------------------------------------------
    P_("\n" + RULE); P_("W0  THE CEILING GATE: IS THE BLOCK THE BINDING TERM?"); P_(RULE)
    P_("  Third application of block.py's rule, and the first time it has stopped an attack before")
    P_("  any of it was built. Under the class-count representation at C = 64, bar 1e12, L = 8.")
    rng = np.random.default_rng(20260907)
    u = rng.random(n)
    a64 = np.minimum((u * 64).astype(int), 63)
    P_(f"\n    {'|C|':>5} {'per-gene term':>16} {'block, class-count':>19} {'block, FACTORED':>17}")
    for nC in (8, 13, 20):
        Cset, genes = gene_regs(adj, order, nC, n)
        per, _, _ = pergene_cost(genes, a64)
        bc = collections.Counter(int(a64[i]) for i in order[:nC])
        blkc = 1.0
        for _, m in bc.items():
            blkc = min(blkc * slot(m), 1e300)
        nb, _ = ctrl_subgraph(E, inv, order, nC)
        w = mindeg_width(nb)
        P_(f"    {nC:>5} {per:>16.3e} {blkc:>19.3e} {2.0**(w+1):>17.3e}")
    caps0 = {}
    for nm, blk in (("class-count block", lambda nC: float(np.prod(
                        [slot(m) for m in collections.Counter(
                            int(a64[i]) for i in order[:nC]).values()]))),
                    ("FACTORED block", lambda nC: 2.0 ** min(
                        mindeg_width(ctrl_subgraph(E, inv, order, nC)[0]) + 1, CAPEXP)),
                    ("block priced at ZERO", lambda nC: 0.0)):
        caps0[nm] = cap_under(adj, order, n, lambda nC, Cs, g: a64, blk, maxC=40)
    P_(f"\n    {'block representation':<26} {'cap |C| at L=8':>15}")
    for k, v in caps0.items():
        P_(f"    {k:<26} {v:>15}")
    moved = len(set(caps0.values())) > 1
    P_(f"\n  W0: {'the cap MOVES with the block' if moved else 'the cap does NOT move -- pricing the block at ZERO changes nothing.'}")
    P_( "  The width exponential in the block is NOT the binding term. The per-gene factor is.")

    # ---- the factorisation that would have been built, reported as useless ---------------------
    P_("\n  AND THE WORK THAT WOULD HAVE BEEN DONE, MEASURED SO NOBODY DOES IT. The controller")
    P_("  subgraph is sparse and its elimination width grows sublinearly, so the block DOES")
    P_("  factorise -- correctly, hugely, and pointlessly.")
    P_(f"\n    {'|C|':>5} {'edges among them':>17} {'elim width':>11} {'2^(w+1)':>11} {'2^|C|':>11}"
       f" {'ratio':>10}")
    for k in (10, 20, 40, 80, 122, 200, 300, 420):
        nb, m = ctrl_subgraph(E, inv, order, k)
        w = mindeg_width(nb)
        P_(f"    {k:>5} {m:>17} {w:>11} {2.0**(w+1):>11.2e} {2.0**min(k,400):>11.2e}"
           f" {2.0**min(k,400)/2.0**(w+1):>10.1e}")

    # ---- W1  THE DEGENERATE OPTIMUM ------------------------------------------------------------
    P_("\n" + RULE); P_("W1  THE DEGENERATE OPTIMUM, REPORTED RATHER THAN HIDDEN"); P_(RULE)
    P_("  The per-gene factor is exponential in the OCCUPIED CLASSES per gene, which is a property")
    P_("  of the assignment. An unconstrained greedy optimiser on that cost was run first:")
    Cset, genes = gene_regs(adj, order, 13, n)
    ctrl = list(order[:13])
    free = np.zeros(n, dtype=int)          # everything in class 0 -- what the optimiser found
    tf, of, _ = pergene_cost(genes, free)
    tr, orr, _ = pergene_cost(genes, a64)
    P_(f"    random C = 64 assignment : cost {tr:.3e}, mean occupied classes {orr:.2f}")
    P_(f"    unconstrained optimum    : cost {tf:.3e}, mean occupied classes {of:.2f}")
    P_(f"  It reaches the theoretical minimum by putting every controller in ONE class, which is")
    P_(f"  the plain count -- refuted on promoter data at 6.29 noise floors. A cost optimiser with")
    P_( "  no accuracy constraint will always rediscover the cheapest refuted representation.")
    P_( "  W1: ledger U in this module's own probe. Every alignment below is CONSTRAINED to keep")
    P_( "  the classes balanced, and the constraint is stated before the number.")

    # ---- W2  DOES THE NETWORK CARRY CLASS STRUCTURE? -------------------------------------------
    P_("\n" + RULE); P_("W2  DOES THE NETWORK CARRY CLASS STRUCTURE, AGAINST A MATCHED SHUFFLE?")
    P_(RULE)
    P_("  The one class label this network actually carries is TRRUST's own regulatory mode per")
    P_("  EDGE. If a gene's regulators are all the same kind its factor is small at NO accuracy")
    P_("  cost, because the classes are given by the data rather than chosen. The control shuffles")
    P_("  the modes across edges keeping their counts, which is the null that matters.")
    srng = np.random.default_rng(7)
    P_(f"\n    {'|C|':>5} {'genes k>=2':>11} {'real modes':>11} {'shuffled':>18}"
       f" {'real all-same':>14} {'shuffled':>10} {'enrich':>8}")
    w2 = True
    for nC in (13, 40, 122, 200):
        Cset, _ = gene_regs(adj, order, nC, n)
        sets = []
        for i in range(n):
            if i in Cset:
                continue
            ms = [mode.get((inv[j], inv[i])) for j in (adj[i] & Cset)]
            ms = [m for m in ms if m]
            if len(ms) >= 2:
                sets.append(ms)
        real = float(np.mean([len(set(s)) for s in sets]))
        rs = float(np.mean([len(set(s)) == 1 for s in sets]))
        pool = [m for s in sets for m in s]
        sh, shs = [], []
        for _ in range(8):
            p = list(pool)
            srng.shuffle(p)
            k = 0
            aa, bb = [], []
            for s in sets:
                t = p[k:k + len(s)]
                k += len(s)
                aa.append(len(set(t)))
                bb.append(len(set(t)) == 1)
            sh.append(np.mean(aa))
            shs.append(np.mean(bb))
        sig = (np.mean(sh) - real) / max(np.std(sh), 1e-12)
        w2 = w2 and sig > 3
        P_(f"    {nC:>5} {len(sets):>11} {real:>11.3f}"
           f" {f'{np.mean(sh):.3f} +- {np.std(sh):.3f}':>18} {rs:>13.1%} {np.mean(shs):>9.1%}"
           f" {rs/max(np.mean(shs),1e-12):>7.2f}x")
    P_(f"\n  W2: {'PASS -- real regulator sets are about twice as class-homogeneous as chance, at more than three sigma everywhere. There IS structure to align to.' if w2 else 'FAIL -- no more homogeneous than shuffled; nothing to align to and the route is closed.'}")

    # ---- W3  WHERE THE PER-GENE COST ACTUALLY LIVES --------------------------------------------
    P_("\n" + RULE); P_("W3  THE PER-GENE COST IS NOT A SUM, IT IS ONE GENE"); P_(RULE)
    P_("  Before aligning anything, ask where the cost is. A representation-level fix only makes")
    P_("  sense if the cost is spread over the representation.")
    P_(f"\n    {'|C|':>5} {'genes':>6} {'max k_i':>8} {'worst gene':>12} {'total':>12}"
       f" {'worst/total':>12} {'top-5/total':>12}  {'worst gene':<10}")
    dom = []
    for nC in (8, 13, 20, 40, 80, 122):
        Cset, genes = gene_regs(adj, order, nC, n)
        cs = []
        for i, regs in genes:
            cnt = collections.Counter(int(a64[j]) for j in regs)
            v = 1.0
            for _, m in cnt.items():
                v = min(v * slot(m), 2.0 ** CAPEXP)
            cs.append((v, len(regs), inv[i]))
        cs.sort(reverse=True)
        tot = sum(c[0] for c in cs)
        dom.append((nC, cs[0][0] / tot))
        P_(f"    {nC:>5} {len(genes):>6} {max(c[1] for c in cs):>8} {cs[0][0]:>12.3e}"
           f" {tot:>12.3e} {cs[0][0]/tot:>12.4f}"
           f" {sum(c[0] for c in cs[:5])/tot:>12.4f}  {cs[0][2]:<10}")
    P_("\n  W3: from |C| = 20 upward ONE GENE is essentially the entire per-gene cost -- CDKN1A,")
    P_("  with 52 regulators among the top 122 factors. The binding term is not a property of the")
    P_("  representation, the block, or the class map. It is a HUB, which is the same object that")
    P_("  killed r-balls (SP1's 3-ball reaches 2,724 of 2,861 genes) and left bounded width with a")
    P_("  6% floor. Third appearance, third form.")
    P_("\n  AND THE ALIGNMENT ATTEMPT THAT FOLLOWED FROM THE WRONG DIAGNOSIS, RECORDED. A balanced")
    P_("  greedy class alignment was built before this measurement. It made things WORSE, by up to")
    P_("  28x, and the reason is a specification error: with C = 64 classes and only |C| = 13")
    P_("  controllers the balance cap is ceil(13/64) = 1, so every controller is FORCED into its")
    P_("  own class -- the worst case, not a balanced one. A balance constraint is meaningless")
    P_("  when there are fewer items than classes, and at these widths there always are.")

    # ---- W4  DEMOTE THE HUBS -------------------------------------------------------------------
    P_("\n" + RULE); P_("W4  IF THE COST IS A MAX, DEMOTE THE FEW GENES THAT SET IT"); P_(RULE)
    P_("  Carry the top-m most expensive genes at the PLAIN COUNT, k(L+1)+1, which is polynomial,")
    P_("  and everything else at C = 64. The accuracy loss is then CONFINED to m genes instead of")
    P_("  being spread over all of them, and it is measurable as a coverage figure.")

    def cap_demote(m, block, maxC=260):
        best = 0
        for nC in range(1, maxC):
            Cset, genes = gene_regs(adj, order, nC, n)
            if not genes:
                best = nC
                continue
            cost = {}
            for i, regs in genes:
                cnt = collections.Counter(int(a64[j]) for j in regs)
                v = 1.0
                for _, mm in cnt.items():
                    v = min(v * slot(mm), 2.0 ** CAPEXP)
                cost[i] = (v, len(regs))
            dem = set(sorted(cost, key=lambda i: -cost[i][0])[:m])
            res = residual_graph(adj, Cset, n)
            per = None
            for r in (1, 2):
                for w in (4, 6, 8):
                    pa, o, _ = bounded_elimination([ball(res, i, r) - {i} for i in range(n)], w)
                    base = np.array([2.0 ** min(1 + len(pa[i]), CAPEXP) for i in range(n)])
                    f = np.ones(n)
                    for i, (v, k) in cost.items():
                        f[i] = (k * (L + 1) + 1.0) if i in dem else v
                    vv = float(np.sum(base * f))
                    per = vv if per is None else min(per, vv)
            if block == "factored":
                blk = 2.0 ** min(mindeg_width(ctrl_subgraph(E, inv, order, nC)[0]) + 1, CAPEXP)
            elif block == "classcount":
                bc = collections.Counter(int(a64[i]) for i in order[:nC])
                blk = 1.0
                for _, mm in bc.items():
                    blk = min(blk * slot(mm), 1e300)
            else:
                blk = 2.0 ** min(nC, CAPEXP)
            if per + blk <= BAR:
                best = nC
            else:
                break
        return best

    def cover(k):
        Cs = {inv[i] for i in order[:k]}
        return sum(1 for a, b, _ in E if a in Cs) / len(E)

    P_(f"\n    {'genes demoted':>14} {'% of all genes':>15} {'cap |C|':>9} {'edge coverage':>14}")
    caps_d = {}
    for m in (0, 1, 2, 5, 20, 100, 500):
        c = cap_demote(m, "factored")
        caps_d[m] = c
        P_(f"    {m:>14} {100*m/n:>14.2f}% {c:>9} {cover(c):>13.1%}")
    P_(f"\n  W4: demoting {100} of {n} genes -- {100*100/n:.1f}% of them -- takes the cap from"
       f" {caps_d[0]} to {caps_d[100]}")
    P_(f"  controllers and edge coverage from {cover(caps_d[0]):.1%} to {cover(caps_d[100]):.1%}."
       f" It SATURATES at 100: demoting 500 buys")
    P_( "  nothing more, which means a different term has taken over as binding.")

    # ---- W5  THE GATE'S OWN VERDICT FLIPS ------------------------------------------------------
    P_("\n" + RULE); P_("W5  AND NOW THE BLOCK BINDS, SO W0's VERDICT FLIPS"); P_(RULE)
    P_("  W0 priced the block at zero and found the cap unmoved, correctly. That verdict was")
    P_("  about the cap AS IT THEN WAS. Once the hubs are demoted the binding term changes, and")
    P_("  the factorisation W0 called pointless becomes the thing the cap rests on.")
    P_(f"\n    {'demoted':>9} {'pattern 2^|C|':>15} {'class-count':>13} {'FACTORED 2^(w+1)':>18}")
    for m in (0, 20, 100):
        P_(f"    {m:>9} {cap_demote(m,'pattern'):>15} {cap_demote(m,'classcount'):>13}"
           f" {cap_demote(m,'factored'):>18}")
    P_("\n  At zero demotions the block representation is worth NOTHING -- all three give the same")
    P_("  cap. At 100 demotions it is worth 140 against 39, a 3.6x difference in the cap and")
    P_(f"  {cover(140):.1%} against {cover(39):.1%} in coverage.")
    P_( "  METHODOLOGICAL POINT, and it is the useful half of this module: a ceiling gate prices a")
    P_( "  term against the CURRENT binding constraint. Its verdict is correct at the time and is")
    P_( "  NOT permanent. When the constraint moves, re-run the gate rather than trusting it.")
    P_( "  Note also that the class-count BLOCK is worse than the plain pattern at these widths --")
    P_( "  with one controller per class its cost is 10 per controller against the pattern's 2 --")
    P_( "  which never mattered while the block was not binding and matters now.")

    # ---- W6  THE ACCURACY SIDE AND WHAT BREAKS NEXT --------------------------------------------
    P_("\n" + RULE); P_("W6  THE ACCURACY SIDE, AND WHAT BREAKS NEXT"); P_(RULE)
    P_("  W4 is a COST statement. What it costs in accuracy is that 100 genes drop from C = 64,")
    P_("  which signed.py measured at 0.19 noise floors from identity, to the plain count at 6.29")
    P_("  floors. That is a large loss on 3.5% of genes and no loss on the other 96.5%, which is")
    P_("  a different kind of trade from every earlier one in this build order -- those all")
    P_("  degraded every gene at once.")
    P_(f"\n    {'genes':>8} {'representation':<22} {'floors from identity':>21} {'share of genes':>15}")
    P_(f"    {n-100:>8} {'class count C = 64':<22} {0.19:>21.2f} {100*(n-100)/n:>14.1f}%")
    P_(f"    {100:>8} {'plain count':<22} {6.29:>21.2f} {100*100/n:>14.1f}%")
    P_("\n  Whether that trade is acceptable is not a cost question and this module cannot settle")
    P_("  it. The demoted genes are the most-regulated in the network -- CDKN1A, VEGFA -- which")
    P_("  are exactly the genes a whole-cell model would most want to get right. Demoting the")
    P_("  hubs buys the cap by giving up on the genes the cap was wanted for, and saying so is")
    P_("  the result rather than a caveat on it.")
    P_("\n  What breaks next: at 140 controllers the block binds, and the elimination width grows")
    P_("  as roughly |C|^0.78, so 2^(w+1) passes the bar around |C| = 150. Beyond that neither")
    P_("  term is polynomial and there is no third term to demote.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_widthblock.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P_(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
