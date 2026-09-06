"""Loop 158. Does the interaction profiler actually SAVE evaluations on the nexus objective?

Loop 157 and the nexus profile (outputs/loop_nexus_catalyst_interactions.json) both used the
profiler on a design space small enough to ENUMERATE, so the profiler's probe cost was free: every
configuration it asked for was already computed. That is a comfortable place to stand and it hides
the question the tool exists to answer -- on a space you cannot enumerate, does profiling cost less
than the thing it replaces?

The nexus run left that question open in the worst way. N3 FAILED: at the default n_references=3
the profiler missed 23 of 56 order-2/3 groups against the exact Moebius residual maximised over
references, and needed n_references=12 to reach zero misses. Reference count is LINEAR in cost. So
the setting that makes the tool CORRECT on this objective is 4x the setting that makes it cheap,
and nobody has counted what that costs.

This loop counts it. The two enumerated tables are cached in the nexus artefact, so the objective
here is a dictionary lookup and the measurement is of the PROBE SCHEDULE, which is the only thing
that varies. Every configuration the profiler requests is recorded; the budget is the number of
DISTINCT configurations, because a caller with an expensive objective would memoise.

PREDECLARED, before any number is looked at.

  P1 INSTRUMENT. Every configuration the profiler requests during every sweep below is present in
     the cached enumerated table -- no lookup falls outside it, so the counter is counting the real
     schedule and not a truncation of it.
     Gate: 0 misses, in both spaces, at every n_references swept.

  P2 SPACE A -- IS THERE A SAVING AT THE CORRECT SETTING? Sweep n_references and, at each, record
     both the number of distinct configurations consumed and whether the profile agrees with the
     exact decomposition. Let R* be the smallest setting with zero missed groups.
     Gate: PASS iff distinct configurations at R* < 128, the full state space. This gate is
     written to be able to fail, and if it fails the honest headline is that on this objective the
     profiler is not a search economy -- it is an analysis of a table you already had.

  P3 SPACE B -- the same question on the 8-block, 256-configuration docking space, where the nexus
     run found the profiler already exact at n_references=3.
     Gate: PASS iff distinct configurations at R* < 256.

  P4 WHERE IS THE CROSSOVER? The profiler's schedule grows polynomially and enumeration grows as
     2^n, so a saving must exist for large enough n whatever happens at n=7. Measure the crossover
     directly: build synthetic objectives with a DECLARED interaction structure at two edge
     densities -- 0.76, matching the density space A actually has (16 interacting pairs of 21), and
     0.15, sparse -- and count distinct configurations at n = 7,8,10,12,14,16,18,20, at both the
     default n_references=3 and at the R* space A needed.
     Gate: PASS iff a crossover exists at n <= 20 for both densities and both settings, and the
     crossover n is reported. FAIL means the tool does not pay for itself in any space a person
     would plausibly build.

  P5 WHAT THAT IS WORTH IN SECONDS, HERE. The space-A objective is a 5-fold pair-head fit over 3
     torch seeds; the nexus run enumerated 128 of them in 11 s of the 82.25 s loop. Convert P2's
     configuration count into wall clock at that measured per-configuration cost, and set it
     against the cost of the thing the profiler is actually competing with on the nexus arm, which
     is not enumeration of the design space but the DOCKING STAGE: 23,304.7 s of a 25,209.5 s
     pipeline (N6, measured).
     Gate: PASS iff the seconds the profiler saves on the design space are reported next to the
     seconds the ablation it recommends saves on the pipeline, and the two are compared honestly.

  P6 THE ONE-SIDEDNESS, RESTATED AS A DECISION RULE. N3 found the profiler's error is one-sided:
     it never invented an interaction, it only missed some. If that holds across the sweep then a
     reported interaction can be trusted at any n_references, and only ABSENCE needs the expensive
     setting -- which is a different and much cheaper way to use the tool than running it at R*.
     Gate: PASS iff false positives are 0 at every setting in both spaces, so the asymmetry is a
     property of the schedule here and not a coincidence of one setting.

-> outputs/loop_profiler_budget.json
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "standalone"))
import gate_guard as GG                                  # noqa: E402
from interaction_profiler import profile_objective       # noqa: E402

NEXUS = Path("outputs/loop_nexus_catalyst_interactions.json")
OUT = Path("outputs/loop_profiler_budget.json")
REFS = (3, 6, 12, 24, 48)
SEED = 15901

# measured in the nexus run, N2 and N6 -- quoted here, not re-derived
SEC_ENUM_A = 11.0          # seconds to enumerate all 128 space-A configurations
N_A, N_B = 128, 256
SEC_DOCK = 23304.7         # docking stage, measured
SEC_PIPE = 25209.5         # whole feature pipeline, measured


class Counter:
    """Wraps a table lookup and records the schedule."""

    def __init__(self, table, n):
        self.table, self.n = table, n
        self.calls, self.seen, self.misses = 0, set(), 0

    def __call__(self, cfg):
        key = "".join(str(int(cfg[i])) for i in range(self.n))
        self.calls += 1
        self.seen.add(key)
        if key not in self.table:
            self.misses += 1
            return 0.0
        return self.table[key]


def exact_delta_maxref(table, group, n):
    """Exact inclusion-exclusion residual, maximised over every reference setting of the other
    variables. This is the quantity profile_objective estimates from n_references random draws."""
    others = [i for i in range(n) if i not in group]
    best = 0.0
    for rmask in range(1 << len(others)):
        base = [0] * n
        for b, i in enumerate(others):
            base[i] = (rmask >> b) & 1
        tot = 0.0
        for r in range(len(group) + 1):
            for sub in combinations(group, r):
                key = list(base)
                for i in group:
                    key[i] = 0
                for i in sub:
                    key[i] = 1
                tot += ((-1) ** (len(group) - r)) * table["".join(str(x) for x in key)]
        best = max(best, abs(tot))
    return best


def truth_set(table, n, tau):
    """Which order-2/3 groups genuinely interact, by the exact max-over-references residual."""
    return {g for order in (2, 3) for g in combinations(range(n), order)
            if exact_delta_maxref(table, g, n) > tau}


def sweep(table, n, tau, label, say):
    """Profile at each n_references; count the schedule and score it against the exact table."""
    truth = {}
    for order in (2, 3):
        for g in combinations(range(n), order):
            truth[g] = exact_delta_maxref(table, g, n) > tau
    n_true = sum(truth.values())
    say(f"     {label}: {len(truth)} groups of order 2 and 3, {n_true} genuinely interacting "
        f"at tau={tau:.4g}; full state space {2 ** n}")
    rows = []
    for R in REFS:
        c = Counter(table, n)
        t0 = time.time()
        rep = profile_objective(c, list(range(n)), 2, tau=tau, max_order=3,
                                n_references=R, adaptive=True, seed=SEED)
        missed = sum(1 for g, t in truth.items() if t and not rep.strengths.get(g, 0.0) > tau)
        fp = sum(1 for g, t in truth.items() if (not t) and rep.strengths.get(g, 0.0) > tau)
        rows.append({"n_references": R, "calls": c.calls, "distinct": len(c.seen),
                     "misses_outside_table": c.misses, "missed": missed, "false_positives": fp,
                     "frac_of_space": len(c.seen) / 2 ** n, "seconds": time.time() - t0})
        say(f"       R={R:<3d} calls {c.calls:>6,}  DISTINCT {len(c.seen):>4,} of {2 ** n:,} "
            f"({len(c.seen) / 2 ** n:6.1%})   missed {missed:>2d}  false-pos {fp}")
    star = next((r for r in rows if r["missed"] == 0), None)
    return {"n_true": n_true, "n_groups": len(truth), "rows": rows, "rstar": star}


def synthetic(n, density, rng):
    """An objective with a DECLARED structure: unary terms everywhere, pairwise terms on a random
    graph at the given density, and a third of those pairs promoted to a triple."""
    u = rng.normal(0, 1, n)
    pairs = [p for p in combinations(range(n), 2) if rng.random() < density]
    w = {p: rng.normal(0, 1) for p in pairs}
    tri = {}
    for p in pairs[::3]:
        cands = [k for k in range(n) if k not in p and ((min(p[0], k), max(p[0], k)) in w)]
        if cands:
            k = int(rng.choice(cands))
            tri[tuple(sorted(p + (k,)))] = rng.normal(0, 1)

    def f(cfg):
        v = sum(u[i] * cfg[i] for i in range(n))
        v += sum(c * cfg[i] * cfg[j] for (i, j), c in w.items())
        v += sum(c * cfg[i] * cfg[j] * cfg[k] for (i, j, k), c in tri.items())
        return float(v)

    return f


def main():
    t0 = time.time()
    log = []

    def say(s=""):
        print(s, flush=True)
        log.append(s)

    say("=" * 104)
    say("  DOES THE PROFILER SAVE EVALUATIONS? -- the nexus design spaces, counted")
    say("=" * 104)

    nx = json.load(open(NEXUS))
    ta, tb = nx["table_a"], nx["table_b"]
    tau_a, tau_b = nx["n1_noise"]["tau_used"], nx["n5"]["tau_b"]
    blocks_a, blocks_b = nx["blocks_a"], nx["blocks_b"]
    say(f"     tables from {NEXUS}: space A {len(ta)} configs (tau {tau_a}), "
        f"space B {len(tb)} configs (tau {tau_b:.4f})")
    say(f"     A blocks: {', '.join(blocks_a)}")
    say(f"     B blocks: {', '.join(blocks_b)}")
    say()

    truth_a = truth_set(ta, 7, tau_a)
    say("P1/P2/P6  SPACE A -- the 7 ESM/sequence blocks, 2,231 reactions")
    A = sweep(ta, 7, tau_a, "space A", say)
    say()
    say("P3        SPACE B -- the 8 docking/sequence blocks, 58 dockable reactions")
    B = sweep(tb, 8, tau_b, "space B", say)
    say()

    p1 = all(r["misses_outside_table"] == 0 for r in A["rows"] + B["rows"])
    GG.verdict(p1, emit=say, if_true=(
        "every configuration the profiler asked for was in the enumerated table: "
        "the counter counts the real schedule."), if_false=(
        "the profiler asked for configurations outside the enumerated table -- the "
        "count below is not the real schedule and P2/P3 are void."))
    say(f"     P1 {'PASS' if p1 else 'FAIL'}")
    say()

    a_star = A["rstar"]
    p2 = bool(a_star) and a_star["distinct"] < N_A
    ra = a_star["n_references"] if a_star else None
    da = a_star["distinct"] if a_star else None
    say(f"     space A reaches zero missed groups first at n_references={ra}, consuming "
        f"{da:,} distinct configurations of {N_A}")
    GG.verdict(p2, emit=say, if_true=(
        f"that is a real saving: {N_A - da} configurations never evaluated."), if_false=(
        f"that is the ENTIRE state space. On this objective the profiler at the setting that "
        f"makes it correct evaluates everything enumeration would have evaluated, so it saves "
        f"nothing -- it is an analysis of a table you already had, not a way to avoid building "
        f"one."))
    say(f"     P2 {'PASS' if p2 else 'FAIL'}")
    say()
    say("     NOT A GATE -- a diagnostic added after this run's R* disagreed with the nexus run's.")
    say("     The nexus sweep (seed 15900) reported zero misses from n_references=12; this run")
    say("     (seed 15901) still misses 6 at 12. The references are drawn at random, so R* is")
    say("     itself a random variable. Re-drawing it 8 ways to see how wide:")
    seed_rstar = []
    for sd in range(8):
        row = []
        for R in REFS:
            c = Counter(ta, 7)
            rep = profile_objective(c, list(range(7)), 2, tau=tau_a, max_order=3,
                                    n_references=R, adaptive=True, seed=1000 + sd)
            miss = sum(1 for g in truth_a if not rep.strengths.get(g, 0.0) > tau_a)
            row.append((R, miss, len(c.seen)))
        first = next((r for r, m, _ in row if m == 0), None)
        seed_rstar.append({"seed": 1000 + sd, "rstar": first,
                           "misses": {r: m for r, m, _ in row}})
        say(f"       seed {1000 + sd}: misses by R " +
            " ".join(f"{r}:{m}" for r, m, _ in row) + f"   -> R* = {first}")
    got = [d["rstar"] for d in seed_rstar]
    say(f"     R* across 8 seeds: {got}  -- the cheap setting cannot tell you which one you are on.")
    say()

    b_star = B["rstar"]
    p3 = bool(b_star) and b_star["distinct"] < N_B
    rb = b_star["n_references"] if b_star else None
    db = b_star["distinct"] if b_star else None
    say(f"     space B reaches zero missed groups first at n_references={rb}, consuming "
        f"{db:,} distinct configurations of {N_B}")
    GG.verdict(p3, emit=say, if_true=(
        f"a saving of {N_B - db} configurations."), if_false=(
        "no saving: the whole space is consumed."))
    say(f"     P3 {'PASS' if p3 else 'FAIL'}")
    say()

    say("P4        WHERE DOES THE SCHEDULE BEAT ENUMERATION? synthetic objectives, declared structure")
    cross = {}
    for dens in (0.76, 0.15):
        for R in (3, ra or 12):
            key = f"density {dens:.2f}, R={R}"
            first = None
            row = []
            for n in (7, 8, 10, 12, 14, 16, 18, 20):
                rng = np.random.default_rng(SEED + n)
                f = synthetic(n, dens, rng)
                c = Counter({}, n)

                def obj(cfg, _f=f, _c=c):
                    _c.calls += 1
                    _c.seen.add("".join(str(int(cfg[i])) for i in range(_c.n)))
                    return _f(cfg)

                profile_objective(obj, list(range(n)), 2, tau=1e-9, max_order=3,
                                  n_references=R, adaptive=True, seed=SEED)
                row.append({"n": n, "distinct": len(c.seen), "space": 2 ** n})
                if first is None and len(c.seen) < 2 ** n:
                    first = n
            cross[key] = {"crossover_n": first, "rows": row}
            shown = "  ".join(f"n={r['n']}:{r['distinct']:,}/{r['space']:,}" for r in row)
            say(f"     {key:<22s} crossover at n={first}   {shown}")
    p4 = all(v["crossover_n"] is not None and v["crossover_n"] <= 20 for v in cross.values())
    GG.verdict(p4, emit=say, if_true=(
        "the schedule does pay for itself once the space is bigger than the nexus arm's -- "
        "the tool is not useless, it is being used on a space too small for it."), if_false=(
        "no crossover at n<=20: the schedule does not beat enumeration in any space a person "
        "would plausibly build here."))
    say(f"     P4 {'PASS' if p4 else 'FAIL'}")
    say()

    say("P5        SECONDS, AT THE MEASURED COST OF THIS OBJECTIVE")
    per_cfg = SEC_ENUM_A / N_A
    saved_design = (N_A - da) * per_cfg
    say(f"     space-A objective: {SEC_ENUM_A:.0f} s for {N_A} configurations = {per_cfg:.3f} s each")
    say(f"     profiling at R={ra} instead of enumerating saves {saved_design:.2f} s")
    say(f"     the ablation the profile RECOMMENDS -- drop every docking block, regret 0.0000 (N5) --")
    say(f"       saves {SEC_DOCK:,.0f} s of a {SEC_PIPE:,.0f} s pipeline: "
        f"{SEC_DOCK / SEC_PIPE:.1%}, a {SEC_PIPE / (SEC_PIPE - SEC_DOCK):.2f}x reduction")
    ratio = (SEC_DOCK / saved_design) if saved_design > 0 else float("inf")
    say(f"     the recommendation is worth {ratio:,.0f}x what the search economy is"
        if saved_design > 0 else
        "     the search economy is worth exactly nothing; the recommendation is worth 23,304.7 s")
    p5 = True
    say("     P5 PASS  (both figures reported against each other)")
    say()

    p6 = all(r["false_positives"] == 0 for r in A["rows"] + B["rows"])
    GG.verdict(p6, emit=say, if_true=(
        "one-sided at every setting in both spaces: a REPORTED interaction can be believed\n"
        "     at n_references=3, and only a reported ABSENCE needs the expensive setting. That is\n"
        "     the cheap way to use this tool -- profile at 3, and pay for the expensive setting\n"
        "     only where you intend to act on a null."), if_false=(
        "false positives appear, so the error is two-sided and neither direction can be\n"
        "     trusted at the cheap setting."))
    say(f"     P6 {'PASS' if p6 else 'FAIL'}")
    say()

    gates = {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5, "P6": p6}
    res = {"test": "does the interaction profiler save evaluations on the nexus design spaces",
           "gates": gates, "space_a": A, "space_b": B, "crossover": cross,
           "seconds_per_config_a": per_cfg, "seconds_saved_design": saved_design,
           "seconds_saved_by_recommendation": SEC_DOCK, "pipeline_seconds": SEC_PIPE,
           "rstar_by_seed": seed_rstar,
           "seconds": time.time() - t0, "log": log}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'PASS' if v else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 104)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
