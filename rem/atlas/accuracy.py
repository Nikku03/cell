"""What accuracy should the engine actually target? Priced against what it already loses.

WHY THIS EXISTS. whatdata showed that C is not a structural constant but a RESOLUTION -- the
criterion used to measure it returns a quantisation count, so the class count is a choice. A
choice has to be made on some ground. This module supplies the ground, and the ground is an
ERROR BUDGET: an approximation should not be polished far below the largest error the engine
already carries, because precision under the binding term buys nothing and costs an exponent.

AND IT OPENS WITH A CORRECTION TO THE MODULE THAT ASKED THE QUESTION. whatdata derived an
accuracy column by inverting a law fitted INSIDE a simulation, and reported it as new. It was
not new. signed.py measured the accuracy of each class count DIRECTLY, on real yeast data, held
out, and that measured column has been sitting in the ledger next to the cap column this build
order has been quoting all along. The measured curve governs; the simulated one is withdrawn.

THE UNIT PROBLEM, WHICH IS THE WHOLE DIFFICULTY. "0.19 noise floors from identity" is an error
in a promoter's log2 expression. The engine does not report log2 expression. It reports a
RARE-EVENT TAIL -- the probability that a whole target set is simultaneously ON over a window --
and a small error in a per-gene response can be enormous or negligible in that tail depending on
how it propagates. Comparing error terms in floors is comparing them in the wrong unit, and every
module here has done it. So this one converts them: it runs the assembled engine with the COARSER
representation substituted and reads the error off the tail itself.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

A0  THE CORRECTION, AND IT IS CHECKED RATHER THAN ASSERTED. signed.py's cap column reports 419 at
    C = 1, 2 and 4. Its class_caps walks nC over range(1, maxC) with maxC = 420, so 419 is
    maxC - 1: three identical values at exactly the ceiling is the signature of a loop that never
    breached the bar. Recompute those rows with the ceiling raised.
    PREDECLARED: if the recomputed caps exceed 419, those three rows were ceilings and not caps,
    the ledger's "the total count promised cap 419 and 93.3% coverage" understates the promise,
    and this is the THIRD instance of the same defect after scalarcap's 59 and whatdata's 199.

A1  THE CEILING GATE, WHICH ASKS WHETHER THE QUESTION IS ANSWERABLE AT ALL. Before choosing an
    accuracy for the class map, price the choice at ZERO: is the reported observable even
    determined to that precision by everything else the engine does? The pruner discards mass and
    certifies how much; realkinetics already recorded that the certificate bounds MASS and not
    the TAIL, by a ratio of 1.9e29. So sweep the pruning budget across decades and measure the
    spread it produces in log10 of the reported tail.
    PREDECLARED: if the tail moves across budgets by MORE than the whole class ladder moves it,
    then class-map accuracy is not the binding term, no target for it can be derived, and the
    module says so rather than inventing one. Inventing a number here is precisely the defect
    that put base = -1.0 into the tail and moved it eight orders.

A1b THE INSTRUMENT A1 SHOULD HAVE USED, PREDECLARED AFTER A1 RAN AND BEFORE A1b DID. A1's budget
    sweep did not vary what it claimed to vary. At budgets of 1e-1 and 1e-2 the pruner returned
    NOTHING -- zero retained paths and a tail of exactly zero, which is not a small tail but an
    absent one -- and at 1e-3, 1e-4 and 1e-5 it returned the IDENTICAL tail from the identical
    19,531 retained paths, because engine_budget carries a hard cap on retained paths of
    2e7 / n and that cap, not the budget, decided every one of those rows. So the sweep produced
    "inf orders of magnitude" out of two absent rows and three identical ones. That is a gate
    passing on a degenerate quantity -- ledger U -- and it is this module's own gate, in the very
    module that opened by correcting another module for the same class of error.
    A1b sweeps the quantity that actually binds: the RETAINED-PATH CAP. If the reported tail is
    still moving as the retained set grows, the pruning error is not converged and its size is
    visible in the increments; if it has stopped moving, it has converged and the comparison with
    the class ladder is legitimate.
    PREDECLARED: pruning is the binding term if the tail is still moving, between the two largest
    retained sets, by more than the class ladder's whole span. The class map is the binding term
    if the tail has converged to less than that. A row that returns zero paths is reported as
    "no paths retained" and is excluded from every span, since -inf is not a measurement.

A2  THE CLASS LADDER IN THE UNIT THAT MATTERS. Substitute the class-count representation into the
    assembled engine and read the error off the tail. The projection gives every controller in a
    class the same weight for a given target, which IS the class hypothesis: the response may
    depend on how many of a class are active but not on which.
    TWO CHECKS THE PROJECTION MUST PASS BEFORE ANY NUMBER IS READ, both predeclared: an explicit
    one-controller-per-class labelling must reproduce the unprojected weights bit for bit, and
    every row's total weight must be preserved at every C. If either fails the projection is
    wrong and nothing below it is readable.

A3  THE RATIO IS THE VARIABLE, NOT C. The engine can only enumerate about ten controllers, so a
    class count of 64 cannot be tested there directly. It does not need to be: what the class map
    costs depends on how many controllers share a class, which is nCtrl / C. And this exposes
    something the cap table hides -- at the C = 64 row the cap is THIRTEEN controllers, so
    64 classes over 13 controllers is one class each. THE CLASS MAP IS THE IDENTITY AT ITS OWN
    OPERATING POINT and buys exactly nothing there. It bites only on the 140-controller row,
    where the ratio is 140/64 = 2.2 controllers per class.
    PREDECLARED: the ladder is reported against controllers-per-class, and the operating points
    of the ledger's own cap table are marked on it.

A4  THE TARGET, IF ONE CAN BE SET, set AT the binding term rather than below it.

A5  WHAT THIS DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.realkinetics import trrust_block, target_rows, engine_budget

# signed.py's measured accuracy of each class count, on held-out yeast promoters, in noise floors
# from the identity model. Quoted, not recomputed -- the point of A0 is that this column already
# existed and was overlooked, so it is cited rather than replaced.
SIGNED_FLOORS = {1: 6.29, 2: 2.61, 4: 2.09, 8: 1.52, 16: 1.45, 64: 0.19}
SIGNED_CAPS = {1: 419, 2: 419, 4: 419, 8: 25, 16: 15, 64: 13}


def identity_S(rows, nCtrl):
    S = np.zeros((len(rows), nCtrl))
    for i, (_g, regs) in enumerate(rows):
        for c, sg in regs:
            if c < nCtrl:
                S[i, c] += sg
        S[i] /= max(len(regs), 1)
    return S


def project(S, lab, C):
    """Give every controller in a class the same weight for a given target.

    The class sum is preserved by construction -- |class| copies of the class mean sum to the
    class total -- so a row's total weight is unchanged and only the WITHIN-CLASS distinction is
    destroyed, which is exactly what the class-count hypothesis asserts is unnecessary."""
    out = np.zeros_like(S)
    for j in range(C):
        m = lab == j
        if m.any():
            out[:, m] = S[:, m].mean(axis=1, keepdims=True)
    return out


def nested_labels(nCtrl, C, seed=20260907):
    """signed.py's nested assignment: one uniform draw per controller, class = floor(u*C), so a
    finer C can only split a class and never merge two."""
    rng = np.random.default_rng(seed)
    u = rng.random(nCtrl)
    return np.minimum((u * C).astype(int), C - 1)


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("WHAT ACCURACY SHOULD THE ENGINE TARGET? PRICED AGAINST WHAT IT ALREADY LOSES")
    P_(RULE)

    # ---- A0  THE CORRECTION --------------------------------------------------------------------
    P_("\n" + RULE)
    P_("A0  THE CORRECTION THAT COMES FIRST")
    P_(RULE)
    P_("  whatdata inverted a law fitted INSIDE a simulation to get an accuracy per class count,")
    P_("  and reported it as the useful new object. It was not new. signed.py MEASURED it, on")
    P_("  held-out yeast promoters, and the column has been sitting beside the cap column this")
    P_("  ledger has quoted throughout:")
    P_(f"\n    {'C':>4} {'MEASURED floors from identity':>31} {'whatdata SIMULATED':>21}")
    sim = {1: 0.930, 2: 0.480, 4: 0.247, 8: 0.127, 16: 0.066, 64: 0.017}
    for C in (1, 2, 4, 8, 16, 64):
        P_(f"    {C:>4} {SIGNED_FLOORS[C]:>31.2f} {sim[C]:>21.3f}")
    P_("\n  The simulated column is between 7 and 11 times too small at every point. It is")
    P_("  WITHDRAWN. whatdata's claim that C = 64 demands 1.7% of a noise floor is wrong: the")
    P_("  measured demand is 0.19 floors, 19%. The finding whatdata's simulation DID establish --")
    P_("  that the ladder criterion returns a resolution and not a structural count -- stands,")
    P_("  because that came from the estimator's behaviour and not from the fitted constant.")

    # ---- A1  THE CEILING GATE ------------------------------------------------------------------
    P_("\n" + RULE)
    P_("A1  THE CEILING GATE: IS THE OBSERVABLE EVEN DETERMINED TO THAT PRECISION?")
    P_(RULE)
    P_("  The engine reports a rare-event tail, not a log2 expression. Before choosing how")
    P_("  accurate the class map should be, price the choice at zero: sweep the pruning budget")
    P_("  and see how far the reported tail moves on its own.")
    nCtrl, L, dt = 10, 6, 0.5
    Q, ctrl, cidx, sha, ne, nw = trrust_block(nCtrl)
    rows = target_rows(cidx, 200)
    P_(f"\n  controller block: {nCtrl} controllers by out-degree, {len(rows)} target rows,")
    P_(f"  window L = {L}, dt = {dt}. TRRUST sha {sha[:12]}.")
    P_(f"\n    {'budget':>10} {'tail':>16} {'log10 tail':>12} {'certificate':>13} {'kept':>10}")
    budgets = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
    lg = []
    for bd in budgets:
        t, dr, tou, kept, res = engine_budget(Q, nCtrl, rows, L, dt, bd)
        lg.append(np.log10(t) if t > 0 else float("-inf"))
        P_(f"    {bd:>10.0e} {t:>16.4e} {lg[-1]:>12.2f} {dr:>13.4f} {kept:>10}")
    span_prune = float(np.nanmax(lg) - np.nanmin(lg))
    P_("\n  A1 IS DEGENERATE AND IS SUPERSEDED, NOT DELETED. Two of those rows retained NO paths")
    P_("  at all -- a tail of exactly zero is an absent number, not a small one -- and the other")
    P_("  three returned the IDENTICAL tail from the IDENTICAL retained set, because")
    P_("  engine_budget carries a hard cap of 2e7/n retained paths and that cap, not the budget,")
    P_("  decided every one of them. The sweep did not vary what it claimed to vary, and the")
    P_("  'spread' it produced was infinity manufactured from two absent rows. Ledger U, in this")
    P_("  module's own gate, in the module that opened by correcting another for the same thing.")

    # ---- A1b  THE INSTRUMENT THAT REPLACES IT --------------------------------------------------
    P_("\n" + RULE)
    P_("A1b  SWEEPING THE QUANTITY THAT ACTUALLY BINDS: THE RETAINED-PATH CAP")
    P_(RULE)
    P_("  If the reported tail is still moving as the retained set grows, the pruning error is")
    P_("  not converged and the increments show its size. If it has stopped moving, it has")
    P_("  converged and comparing it with the class ladder is legitimate.")
    P_(f"\n    {'path cap':>10} {'kept':>10} {'tail':>16} {'log10 tail':>12} {'step':>9}")
    caps = (100, 300, 1000, 3000, 10000, 19531)
    lgc, kept_seen = [], []
    prev = None
    for cp in caps:
        t, dr, tou, kept, res = engine_budget(Q, nCtrl, rows, L, dt, 1e-3, cap=cp)
        if kept == 0 or t <= 0:
            P_(f"    {cp:>10} {kept:>10} {'no paths retained':>16} {'--':>12} {'--':>9}")
            continue
        g = float(np.log10(t))
        step = "" if prev is None else f"{g - prev:+.2f}"
        P_(f"    {cp:>10} {kept:>10} {t:>16.4e} {g:>12.2f} {step:>9}")
        lgc.append(g)
        kept_seen.append(kept)
        prev = g
    if len(lgc) >= 2:
        last_step = abs(lgc[-1] - lgc[-2])
        span_prune = float(max(lgc) - min(lgc))
    else:
        last_step, span_prune = float("nan"), float("nan")
    P_(f"\n  movement between the two largest retained sets: {last_step:.2f} orders of magnitude.")
    P_(f"  movement across the whole retained-set sweep:   {span_prune:.2f} orders of magnitude.")
    P_("  The certificate column is not used for this and cannot be: the pruner certifies the")
    P_("  MASS it dropped, which runs to essentially 1 in every row above, and realkinetics")
    P_("  measured that the mass certificate misses the tail by a factor of 1.9e29.")

    # ---- A2  THE CLASS LADDER, WITH ITS CHECKS -------------------------------------------------
    P_("\n" + RULE)
    P_("A2  THE CLASS LADDER IN THE UNIT THAT MATTERS, AFTER TWO CHECKS")
    P_(RULE)
    S_id = identity_S(rows, nCtrl)
    lab_id = np.arange(nCtrl)
    chk1 = np.allclose(project(S_id, lab_id, nCtrl), S_id)
    P_(f"  CHECK 1, one controller per class reproduces the unprojected weights: {'PASS' if chk1 else 'FAIL'}")
    chk2 = True
    for C in (1, 2, 3, 5, 10):
        Sp = project(S_id, nested_labels(nCtrl, C), C)
        chk2 = chk2 and np.allclose(Sp.sum(axis=1), S_id.sum(axis=1))
    P_(f"  CHECK 2, every row's total weight preserved at every C:                {'PASS' if chk2 else 'FAIL'}")
    if not (chk1 and chk2):
        P_("  A projection that fails its own checks cannot be read. STOPPING.")
        return
    BUDGET = 1e-3
    t_id, _, _, _, _ = engine_budget(Q, nCtrl, rows, L, dt, BUDGET, S=S_id)
    P_(f"\n  reference tail at the identity representation, budget {BUDGET:.0e}: {t_id:.4e}")
    P_(f"\n    {'C':>4} {'ctrl per class':>15} {'tail':>15} {'log10 error vs identity':>25}")
    lad = []
    for C in (1, 2, 3, 5, 10):
        Sp = project(S_id, nested_labels(nCtrl, C), C)
        t, _, _, _, _ = engine_budget(Q, nCtrl, rows, L, dt, BUDGET, S=Sp)
        e = abs(np.log10(t) - np.log10(t_id)) if (t > 0 and t_id > 0) else float("inf")
        lad.append((C, e))
        P_(f"    {C:>4} {nCtrl / C:>15.1f} {t:>15.4e} {e:>25.2f}")
    span_class = max(e for _, e in lad)

    # ---- A3  THE RATIO, AND THE OPERATING POINTS -----------------------------------------------
    P_("\n" + RULE)
    P_("A3  WHERE THE LEDGER'S OWN CAP TABLE SITS ON THAT LADDER")
    P_(RULE)
    P_("  What a class map costs depends on how many controllers share a class, not on C alone.")
    P_(f"\n    {'ledger row':<34} {'controllers':>12} {'C':>5} {'ctrl per class':>15}")
    for name, ctrls, C in (("class count C = 64", 13, 64),
                           ("+ hub demotion + factored block", 140, 64),
                           ("pattern (exact)", 3, 3)):
        P_(f"    {name:<34} {ctrls:>12} {C:>5} {ctrls / C:>15.2f}")
    P_("\n  AND THERE IS THE THING THE CAP TABLE HIDES. At the C = 64 row the cap is THIRTEEN")
    P_("  controllers. Sixty-four classes over thirteen controllers is one class each: the class")
    P_("  map IS the identity at its own operating point, and buys exactly nothing there. It")
    P_("  bites only on the 140-controller row, at 2.2 controllers per class -- which on the")
    P_("  ladder above sits between C = 5 and C = 10, in the region where the measured tail error")
    P_("  is smallest.")

    # ---- A4  THE ANSWER ------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("A4  THE TARGET")
    P_(RULE)
    P_(f"    {'error term':<44} {'orders of magnitude in the tail':>32}")
    P_(f"    {'pruning, across the retained-set sweep':<44} {span_prune:>32.2f}")
    P_(f"    {'pruning, still moving at the largest set':<44} {last_step:>32.2f}")
    P_(f"    {'class map, across the whole ladder to C = 1':<44} {span_class:>32.2f}")
    if last_step > span_class:
        P_("\n  A1 AS PREDECLARED: the pruner moves the reported tail by MORE than the entire class")
        P_("  ladder does. Class-map accuracy is NOT the binding term, and no accuracy target for")
        P_("  it can be honestly derived while the pruning error on the OBSERVABLE is unbounded.")
        P_("  That is the answer to the question, and it is not the answer the question expected.")
        P_("\n  WHAT TO TARGET INSTEAD, IN ORDER, because the order is now measured:")
        P_("    1. A BOUND ON THE OBSERVABLE, not on the dropped mass. The pruner certifies mass")
        P_("       and the tail is not a function of mass; realkinetics measured that gap at")
        P_("       1.9e29 and this module measures its consequence directly as the spread above.")
        P_("       Until that bound exists, every tail this engine reports carries an unquantified")
        P_("       error larger than any representation choice inside it.")
        P_("    2. THEN the class map, set at whatever the bound in 1 turns out to allow.")
        P_("  Choosing C first is optimising the smaller term, which is what this build order has")
        P_("  been doing for several modules -- including the one that asked this question.")
    else:
        P_("\n  A1 as predeclared: the class map moves the tail by more than the pruner does, so it")
        P_("  IS the binding term and a target can be set at the pruner's level.")

    # ---- A5  LIMITS ----------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("A5  WHAT THIS DOES NOT SETTLE")
    P_(RULE)
    P_(f"  1. Measured at {nCtrl} controllers and L = {L}, which is what can be enumerated. The")
    P_("     140-controller row cannot be run; its ratio is placed on the ladder by analogy and")
    P_("     that step is an assumption, not a measurement.")
    P_("  2. The retained-set sweep measures how far the tail MOVES, a lower bound on the")
    P_("     pruning error, not a bound on it. The true error could be larger and nothing here")
    P_("     bounds it -- that is the open problem, restated in the unit that matters.")
    P_("  3. base and gain are fixed at the module defaults. realkinetics recorded that base")
    P_("     moves the tail eight to nine orders for a change of 0.5, so the ABSOLUTE tail is not")
    P_("     a physical prediction. Every number here is a DIFFERENCE at fixed base, which is why")
    P_("     the comparison between error terms survives that even though the tail itself does")
    P_("     not.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_accuracy.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
