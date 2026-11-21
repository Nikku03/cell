"""Adding the per-gene scalar to the engine, and rerunning the caps.

WHAT genegroups LEFT. The context-dependence that damages every compression here is mostly ONE
OBSERVABLE SCALAR PER GENE -- the gene's own baseline strength -- rather than an irreducible
per-gene function. Sharing classes across groups of genes was tested and lost to that scalar in
both cell types, at 3 to 32 times fewer parameters. The scalar is therefore the repair, and this
puts it into the engine and recomputes what it is worth.

THE CLAIM THAT HAS TO BE CHECKED IN CODE RATHER THAN ASSERTED. genegroups' J4 said in prose that
the cap does not move: a per-gene covariate adds one global coefficient and changes no gene's
alphabet, so the per-gene factor prod_j (k_ij(L+1)+1) is untouched. That reasoning is almost
certainly right and it is exactly the kind of reasoning block.py's E4 and E7 got wrong -- they
were STRING ARITHMETIC, never executed, and the conclusion they supported was false by a wide
margin. So V0 runs it.

AND THE QUESTION HIDING BEHIND IT, WHICH IS THE ONE THAT MATTERS. The scalar does not change cost
AT FIXED C. But C is what drives the cap: signed.py measured cap 419 at C <= 4, 25 at C = 8, 15 at
C = 16 and 13 at C = 64, and the reason C = 64 was needed was accuracy. If the scalar absorbs the
context variance, a COARSER class map may reach the same accuracy -- and a smaller C is worth an
enormous amount of cap. That is the real experiment.

THE SCALAR IS FREE IN THE ENGINE, WHICH IS WHY THIS IS WORTH DOING AT ALL. The observable is the
gene's own baseline expression level. The engine already carries x_i's distribution; the scalar is
a functional of state it already has, not a new variable to track. It costs one global coefficient
shared by every gene.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

V0  THE CEILING GATE, IN EXECUTED CODE. At fixed C, compute the cap with and without the scalar
    term in the functional. PREDECLARED: they must be identical. If they are not, the accounting
    in genegroups' J4 is wrong and everything below it is suspect.

V1  DOES THE SCALAR BUY A SMALLER C? On the human MPRA data, held out -- fit on replicate 1, score
    on replicate 2 -- sweep the number of motif classes with and without the per-gene scalar, and
    find the smallest C reaching a matched accuracy target in each case. The target is signed.py's
    own: within one noise floor of the finest model.

V2  THE CAP AT THOSE C's, through signed.py's own cap function, so the numbers are comparable to
    its 13 and to widthblock's 140 rather than being a new scale.

V3  THE TRANSFER IS BY FRACTION OF THE ALPHABET, NOT BY C ITSELF. The MPRA has 136 motif classes
    and TRRUST has 404 regulators; a C measured on one is only meaningful on the other as a
    fraction. PREDECLARED: the fraction is what is carried across, and it is stated as an
    assumption rather than a measurement, because it is one.

V4  WHAT THIS DOES AND DOES NOT SETTLE.

=================================================================================================
V1 WAS DEGENERATE AND V2 IS WITHDRAWN
=================================================================================================
V1 asked for the smallest class count within one noise floor of the finest model, which is
signed.py's own criterion. On this data the ENTIRE ladder is narrower than the floor -- 0.06 log2
against 0.39 in HepG2 -- so every C qualifies, C = 1 included, and the criterion returns 1
whatever the truth is. That is the flat ladder mpra_human already reported, meeting a bar built
for yeast's, which separated its ends by 8.6 floors. Ledger U, in this module's own gate.

V2 was to carry that class count to TRRUST as a fraction of the alphabet and rerun the caps. Run
as written it produced "human data says C = 3 suffices, so the cap is 140 at 71.7% coverage" --
a spectacular claim resting entirely on the degenerate statistic above. It is withdrawn.

A THIRD THING THE FIRST RUN GOT WRONG, quieter but worth the same treatment: V0 fixed the block
representation to the factored one, which put the C = 4 cap at 59 against signed.py's 419, because
the factored block is the WORSE choice at low C and large |C|. The engine would obviously use the
cheapest available block, so the functional now takes the minimum over the three.

WHAT SURVIVES IS THE THING THAT WAS ASKED FOR. The per-gene scalar is FREE -- it adds one value
per gene where the per-gene factor is a PRODUCT over classes, so it cannot multiply anything -- and
the cap is identical with and without it at every class count tested, in executed code rather than
prose. It is a free ACCURACY gain, worth +0.64 noise floors in HepG2 and +0.13 to +0.25 in K562,
flat in C. The class count that sets the cap remains the yeast-derived C = 64, still unvalidated
for human, exactly as humantransfer left it.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.genegroups import panel, motif_classes
from rem.atlas.widthblock import (net, gene_regs, slot, mindeg_width, ctrl_subgraph,
                                  BAR, CAPEXP, L)
from rem.atlas.localclosure import ball
from rem.atlas.engine import residual_graph
from rem.atlas.boundedwidth import bounded_elimination


def cap_with_scalar(nC_classes, scalar, maxC=200, seed=20260907):
    """signed.py's cap functional, with an optional PER-GENE SCALAR term.

    The scalar is one global coefficient plus an observable the engine already carries, so it
    enters the functional as an ADDITIVE constant per gene rather than as a factor on the gene's
    table. Whether that leaves the cap untouched is the thing V0 measures rather than assumes."""
    E, adj, inv, sha, n, order, mode = net()
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    asg = np.minimum((u * nC_classes).astype(int), nC_classes - 1)
    best = 0
    for nCtrl in range(1, maxC):
        Cset, genes = gene_regs(adj, order, nCtrl, n)
        if not genes:
            best = nCtrl
            continue
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
                    # the scalar adds ONE value per gene, it does not multiply the table
                    f[i] = (v + 1.0) if scalar else v
                per = float(np.sum(base * f)) if per is None else min(
                    per, float(np.sum(base * f)))
        # the engine would use the CHEAPEST available block, so take the minimum rather than
        # fixing one. Using only the factored block put the C = 4 cap at 59 against signed.py's
        # 419, because the factored block is the worse choice at low C and large |C|.
        bc = collections.Counter(int(asg[i]) for i in order[:nCtrl])
        blk_cls = 1.0
        for _, m in bc.items():
            blk_cls = min(blk_cls * slot(m), 1e300)
        blk = min(2.0 ** min(mindeg_width(ctrl_subgraph(E, inv, order, nCtrl)[0]) + 1, CAPEXP),
                  blk_cls, 2.0 ** min(nCtrl, CAPEXP))
        if per + blk + (1.0 if scalar else 0.0) <= BAR:
            best = nCtrl
        else:
            break
    return best


def coverage(k):
    E, adj, inv, sha, n, order, mode = net()
    Cs = {inv[i] for i in order[:k]}
    return sum(1 for a, b, _ in E if a in Cs) / len(E)


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE); P_("THE PER-GENE SCALAR IN THE ENGINE, AND THE CAPS RERUN"); P_(RULE)
    P_("  genegroups found the context-dependence that damages every compression here is mostly")
    P_("  ONE OBSERVABLE SCALAR PER GENE, and that sharing classes across groups loses to it. The")
    P_("  scalar is free in the engine: the observable is the gene's own baseline expression, a")
    P_("  functional of state the engine already carries, plus one global coefficient.")

    # ---- V0  THE CEILING GATE, IN CODE ---------------------------------------------------------
    P_("\n" + RULE); P_("V0  DOES THE SCALAR MOVE THE CAP AT FIXED C? (IN EXECUTED CODE)"); P_(RULE)
    P_("  genegroups asserted in prose that it does not. block.py's E4 and E7 asserted their")
    P_("  arithmetic in prose too, were never executed, and were false by a wide margin. So this")
    P_("  is run rather than argued.")
    P_(f"\n    {'C classes':>10} {'cap without scalar':>20} {'cap WITH scalar':>17} {'same?':>7}")
    v0 = True
    for C in (4, 8, 16, 64):
        a = cap_with_scalar(C, False, maxC=60)
        b = cap_with_scalar(C, True, maxC=60)
        v0 = v0 and (a == b)
        P_(f"    {C:>10} {a:>20} {b:>17} {str(a == b):>7}")
    P_(f"\n  V0: {'PASS -- the scalar changes no cap. It adds one value per gene where the table is a PRODUCT over classes, so it cannot multiply anything.' if v0 else 'FAIL -- the cap moves, so the accounting in genegroups J4 is wrong'}")

    # ---- V1  DOES THE SCALAR BUY A SMALLER C? --------------------------------------------------
    P_("\n" + RULE); P_("V1  DOES THE SCALAR BUY A SMALLER CLASS COUNT?"); P_(RULE)
    P_("  This is the question that matters, because C is what drives the cap. Held out on the")
    P_("  human MPRA: fit on replicate 1, score on replicate 2, sweep C with and without the")
    P_("  scalar, and find the smallest C within one noise floor of the finest model -- which is")
    P_("  signed.py's own criterion, transplanted.")
    need = {}
    for cell in ("HepG2", "K562"):
        W1, W2, E1, E2, M, sha = panel(cell)
        lab, ncl = motif_classes(M)
        sg = float(np.std(E1 - E2)) / np.sqrt(2.0)
        eff = np.array([E1[lab == c].mean() for c in range(ncl)])
        order_c = np.argsort(eff)
        CS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, ncl]
        rows = []
        for C in CS:
            sup = np.zeros(ncl, dtype=int)
            for b, grp in enumerate(np.array_split(order_c, min(C, ncl))):
                sup[grp] = b
            sl = sup[lab]
            cm = {c: float(E1[sl == c].mean()) for c in set(sl.tolist())}
            g0 = np.array([cm[c] for c in sl])
            e_plain = float(np.sqrt(np.mean((E2 - g0) ** 2)))
            A = np.c_[g0, W1, np.ones(len(E1))]
            b2, _, _, _ = np.linalg.lstsq(A, E2, rcond=None)
            e_scal = float(np.sqrt(np.mean((E2 - A @ b2) ** 2)))
            rows.append((C, e_plain, e_scal))
        fine_p = rows[-1][1]
        fine_s = rows[-1][2]
        okp = [C for C, p, s in rows if p - fine_p <= sg]
        oks = [C for C, p, s in rows if s - fine_s <= sg]
        span_p = max(p for _, p, _ in rows) - min(p for _, p, _ in rows)
        span_s = max(s for _, _, s in rows) - min(s for _, _, s in rows)
        degenerate = span_p <= sg and span_s <= sg
        need[cell] = (min(okp), min(oks), ncl, sg, degenerate, span_p, span_s)
        P_(f"\n  {cell}: {ncl} motif classes, floor {sg:.4f}")
        P_(f"    {'C':>6} {'no scalar':>11} {'with scalar':>13} {'gain in floors':>16}")
        for C, p, s in rows:
            P_(f"    {C:>6} {p:>11.4f} {s:>13.4f} {(p-s)/sg:>16.2f}")
        P_(f"    the WHOLE ladder spans {span_p:.4f} without the scalar and {span_s:.4f} with it,")
        P_(f"    against a noise floor of {sg:.4f}.")
        if degenerate:
            P_( "    THE CRITERION IS DEGENERATE HERE. Every C sits within one floor of the finest")
            P_(f"    model, C = 1 included, because the entire ladder is narrower than the floor.")
            P_( "    'Smallest C within one floor' therefore returns 1 whatever the truth is, and")
            P_( "    NO class count may be derived from this data. That is the flat ladder")
            P_( "    mpra_human already reported, meeting a criterion built for yeast's, which")
            P_( "    separated its ends by 8.6 floors. Ledger U -- a gate passing on a saturated")
            P_( "    quantity -- and it is this module's own gate.")
        else:
            P_(f"    smallest C within one floor: {min(okp)} without, {min(oks)} WITH the scalar")
        P_(f"\n    WHAT IS MEASURABLE HERE: the scalar's gain, which is real and flat in C --")
        P_(f"    {(rows[0][1]-rows[0][2])/sg:+.2f} floors at C = 1 and"
           f" {(rows[-1][1]-rows[-1][2])/sg:+.2f} floors at C = {ncl}. On this data the per-gene")
        P_( "    scalar does essentially all the work and the class map almost none, but the class")
        P_( "    map's flatness is the data's limitation and not a finding about class maps.")

    # ---- V2 / V3  THE CAPS ---------------------------------------------------------------------
    P_("\n" + RULE); P_("V2/V3  THE CAPS, AND WHY THEY DO NOT MOVE"); P_(RULE)
    P_("  The plan was to take the class count V1 measured on human data, carry it to TRRUST as a")
    P_("  fraction of the alphabet, and rerun the caps there. V1 CANNOT SUPPLY THAT NUMBER: its")
    P_("  ladder is narrower than its own noise floor, so the criterion returns C = 1 regardless")
    P_("  of the truth. Deriving a cap from it would have produced 'human data says C = 3 suffices,")
    P_("  so the cap is 140 at 71.7% coverage' -- a spectacular claim resting entirely on a")
    P_("  degenerate statistic. It is withdrawn rather than reported.")
    P_("\n  WHAT SURVIVES IS V0, AND IT IS WHAT WAS ASKED FOR. The per-gene scalar is FREE: it adds")
    P_("  one value per gene where the per-gene factor is a PRODUCT over classes, so it cannot")
    P_("  multiply anything, and the cap is identical with and without it at every class count")
    P_("  tested. The caps therefore stand exactly as they were:")
    P_(f"\n    {'representation':<40} {'cap':>6} {'coverage':>10}")
    for nm, cp in (("pattern (exact)", 3), ("class count C = 64", 13),
                   ("+ hub demotion + factored block", 140)):
        P_(f"    {nm:<40} {cp:>6} {coverage(cp):>9.1%}")
    P_("    + per-gene scalar                          unchanged, at whichever row it is added to")
    P_("\n  So the scalar is a free ACCURACY gain, not a cap gain, and the class count that sets")
    P_("  the cap remains the yeast-derived C = 64 -- still unvalidated for human, exactly as")
    P_("  humantransfer left it. Nothing here changes that, and nothing here pretended to.")

    # ---- V4 ------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("V4  WHAT THIS DOES AND DOES NOT SETTLE"); P_(RULE)
    P_("  1. The fraction transfer is an assumption, not a measurement. A class count measured on")
    P_("     136 motif classes in a reporter assay is being carried to 404 regulators in a")
    P_("     curated network, and nothing here validates that step.")
    P_("  2. The scalar is fitted and tested on SINGLE-SITE contexts, because that is all the open")
    P_("     human data has. Whether one scalar still absorbs the context when several sites are")
    P_("     present is the copy-number question, still unasked of human data.")
    P_("  3. V0 is a statement about the COST functional only. It says the scalar is free, not")
    P_("     that it is sufficient.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_scalarcap.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P_(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
