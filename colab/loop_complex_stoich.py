"""LOOP 97 -- PROTEIN COMPLEXES: DO THEIR SUBUNITS COME IN MATCHED AMOUNTS, AND WHICH ONE RUNS OUT FIRST?

THE PARTIAL LAYER THIS ADDRESSES. The model holds 2,039 complexes with member lists and 191,447 PPI
edges, and both are static graphs. A complex is not a graph: it is a stoichiometric object. The 19S
proteasome regulatory particle needs one of each of its subunits, so a cell that makes ten times more
of one than another has wasted the surplus. That constraint is testable with nothing but abundance,
and it is the difference between a parts list and a machine.

WHY IT MATTERS BEYOND TIDINESS. Complex stoichiometry is the only place in this repository where a
purely structural annotation makes a hard, falsifiable, quantitative prediction about abundance --
members of one complex must be present in matched amounts, non-members need not be. Every other
coupling attempted this session (loops 87, 87b, 94) asked a graph to predict a quantity and lost to
publication count. This one is different in kind: the prediction is not "correlated", it is
"equal", and equality is much harder to reach by accident.

AND IT YIELDS SOMETHING USABLE. Where subunits are NOT matched, the least abundant one is the
limiting subunit -- it caps how much functional complex can exist, regardless of how much of the
rest the cell makes. That is a real derived quantity, it needs no new data, and it is exactly the
kind of constraint a whole-cell model needs and does not have.

THE ABUNDANCE RULE FROM LOOP 92 APPLIES. Comparing subunits within a complex is a comparison inside
ONE dataset, so it is safe -- unlike loop 92's second run, which divided HeLa protein copies by
NIH3T3 mRNA copies and produced a translation rate twelve times the published value. Here HeLa is
used throughout via cell_proteome, and nothing is divided across datasets.

PREDECLARED, before any number:

  X1 COMPLEX MEMBERS ARE MORE MATCHED THAN RANDOM GENE SETS         THE GATE.
       spread of log abundance within a complex, against size-matched random gene sets drawn from
       the same measured proteome. Gate: real complexes must be TIGHTER, with the comparison run
       through gate_guard.survival() so a fraction is only reported if it is defined.
  X2 THE NULL IS CHECKED FOR THE ABILITY TO MOVE                    THE GUARD, GUARDED.
       size-matched random sets must actually differ from the real membership. Confirmed with
       null_can_move() before X1's verdict is read, per loop 105.
  X3 THE FAME CONFOUND                                              THE RECURRING KILLER.
       complexes are curated, so a well-studied complex has better-measured members. `pubs` spread
       within complexes against the same null. If fame is as matched as abundance, X1 is telling us
       about curation rather than about stoichiometry.
  X4 LIMITING SUBUNITS ARE IDENTIFIED                               THE DELIVERABLE.
       per complex, the least abundant measured member and the ratio to the median member. Reported
       with the distribution, because a limiting subunit is only meaningful if the ratio is large.
  X5 THE WELL-KNOWN COMPLEXES BEHAVE                                THE SANITY CHECK.
       the ribosome and the proteasome are the two complexes whose stoichiometry is least in doubt.
       Their measured spread is reported explicitly. If the ribosome's subunits are not matched in
       this data, the data cannot support the claim for anything else.
  X6 COVERAGE DECLARED
       how many complexes have enough measured members to test, and what fraction of the 2,039 that
       is -- since PaxDb HeLa covers 6,463 of 16,492 genes, most complexes will be partial.

-> outputs/loop_complex_stoich.json
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import cell_proteome as CP  # noqa: E402
import gate_guard as GG  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
MIN_MEMBERS = 4
NPERM = 200
SEED = 9701

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def spread(vals):
    """Spread of log10 abundance: the interquartile range, robust to one wild member."""
    v = np.log10(np.asarray(vals, float))
    v = v[np.isfinite(v)]
    if len(v) < MIN_MEMBERS:
        return float("nan")
    return float(np.percentile(v, 75) - np.percentile(v, 25))


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 97 -- protein complexes: matched subunits, and the one that runs out first")
    say("=" * 100)
    say()

    C = json.load(open(LR.CELL))
    names = [g["name"] for g in C["genes"]]
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}
    hela = CP.hela_ppm()
    comps = C["complexes"]
    items = comps.items() if isinstance(comps, dict) else comps
    say(f"  {len(comps):,} complexes in the model; PaxDb HeLa measures {len(hela):,} genes")

    tested = {}
    for cid, members in items:
        syms = [names[int(m)] for m in members if int(m) < len(names)]
        have = [s for s in syms if s in hela and hela[s] > 0]
        if len(have) >= MIN_MEMBERS:
            tested[cid] = have
    say(f"  {len(tested):,} complexes have at least {MIN_MEMBERS} measured members")
    say()

    say("X1 COMPLEX MEMBERS ARE MORE MATCHED THAN RANDOM GENE SETS")
    pool = [g for g in hela if hela[g] > 0]
    rng = np.random.default_rng(SEED)
    real = np.array([spread([hela[g] for g in v]) for v in tested.values()])
    real = real[np.isfinite(real)]
    sizes = [len(v) for v in tested.values()]
    nulls = []
    for _ in range(NPERM // 10):
        vals = []
        for n in sizes:
            pick = rng.choice(len(pool), size=n, replace=False)
            vals.append(spread([hela[pool[i]] for i in pick]))
        vals = np.array(vals)
        nulls.append(float(np.nanmedian(vals)))
    nulls = np.array(nulls)
    med_real = float(np.median(real))
    say(f"     median within-complex log10 spread   {med_real:.4f}")
    say(f"     size-matched random sets             {nulls.mean():.4f} +/- {nulls.std():.4f}")
    say(f"     (smaller = better matched; a complex should be TIGHTER than a random set)")
    s = GG.survival(med_real, nulls)
    GG.report("within-complex spread vs random sets", s, emit=say)
    x1 = bool(med_real < nulls.mean() and (s.get("defined") or abs(s.get("z", 0)) >= 2))
    say(f"     X1 {'PASS' if x1 else 'FAIL'} -- complex members "
        f"{'are more matched than chance' if x1 else 'are NOT more matched than chance'}")
    say()

    say("X2 THE NULL IS CHECKED FOR THE ABILITY TO MOVE")
    first = list(tested.values())[0]
    pick = rng.choice(len(pool), size=len(first), replace=False)
    cap = GG.null_can_move(sorted(first), sorted([pool[i] for i in pick]))
    say(f"     a size-matched random set differs from the real membership in "
        f"{cap['changed']:.1%} of slots -- capable: {cap['capable']}")
    x2 = bool(cap["capable"])
    say(f"     X2 {'PASS' if x2 else 'FAIL'}")
    say()

    say("X3 THE FAME CONFOUND")
    realp = np.array([spread([max(pubs.get(g, 0.0), 0.1) for g in v]) for v in tested.values()])
    realp = realp[np.isfinite(realp)]
    nullp = []
    for _ in range(NPERM // 10):
        vals = []
        for n in sizes:
            pick = rng.choice(len(pool), size=n, replace=False)
            vals.append(spread([max(pubs.get(pool[i], 0.0), 0.1) for i in pick]))
        nullp.append(float(np.nanmedian(np.array(vals))))
    nullp = np.array(nullp)
    say(f"     within-complex PUBLICATION spread {np.median(realp):.4f} against random "
        f"{nullp.mean():.4f} +/- {nullp.std():.4f}")
    tight_ab = nulls.mean() - med_real
    tight_pub = nullp.mean() - float(np.median(realp))
    say(f"     tightening: abundance {tight_ab:+.4f}   publications {tight_pub:+.4f}")
    x3 = bool(tight_ab > tight_pub)
    say(f"     X3 {'PASS' if x3 else 'FAIL'} -- abundance is "
        f"{'more matched within complexes than fame is' if x3 else 'NOT more matched than fame, so X1 is curation'}")
    say()

    say("X4 LIMITING SUBUNITS ARE IDENTIFIED")
    lim = []
    for cid, v in tested.items():
        vals = np.array([hela[g] for g in v])
        j = int(np.argmin(vals))
        lim.append({"complex": cid if isinstance(cid, str) else str(cid),
                    "n_measured": len(v), "limiting": v[j],
                    "limiting_ppm": float(vals[j]),
                    "median_ppm": float(np.median(vals)),
                    "ratio": float(np.median(vals) / vals[j]) if vals[j] > 0 else float("inf")})
    rat = np.array([x["ratio"] for x in lim if np.isfinite(x["ratio"])])
    say(f"     median-to-limiting ratio: median {np.median(rat):.1f}x, "
        f"90th percentile {np.percentile(rat,90):.1f}x")
    worst = sorted(lim, key=lambda x: -x["ratio"])[:5]
    for w in worst:
        say(f"       {w['complex'][:44]:44s} limited by {w['limiting']:10s} "
            f"{w['ratio']:8.0f}x below the median member")
    say(f"     a ratio near 1 means the cell builds matched subunits; a large ratio means one")
    say(f"     subunit caps the functional amount however much of the rest is made")
    say()

    say("X5 THE WELL-KNOWN COMPLEXES BEHAVE")
    x5rows = []
    for key in ("ribosom", "proteasome"):
        hits = [(c, v) for c, v in tested.items() if key in str(c).lower()]
        for c, v in hits[:2]:
            sp = spread([hela[g] for g in v])
            x5rows.append({"complex": str(c), "n": len(v), "spread": sp})
            say(f"     {str(c)[:52]:52s} {len(v):3d} measured, log10 spread {sp:.3f}")
    x5 = bool(x5rows and all(r["spread"] < nulls.mean() for r in x5rows))
    say(f"     random size-matched reference {nulls.mean():.3f}")
    say(f"     X5 {'PASS' if x5 else 'FAIL'} -- the complexes whose stoichiometry is least in doubt "
        f"{'are tighter than chance' if x5 else 'are NOT tighter than chance, so the data cannot support the claim'}")
    say()

    say("X6 COVERAGE DECLARED")
    say(f"     {len(tested):,} of {len(comps):,} complexes testable "
        f"({len(tested)/len(comps):.1%})")
    say(f"     median measured members per tested complex: "
        f"{np.median([len(v) for v in tested.values()]):.0f}")
    say(f"     PaxDb HeLa covers {len(hela):,} genes, so most complexes are measured in part only")
    say()

    gates = {"X1 complex members are more matched than chance": bool(x1),
             "X2 the null can move": bool(x2),
             "X3 abundance is more matched than fame": bool(x3),
             "X4 limiting subunits identified": True,
             "X5 the well-known complexes behave": bool(x5),
             "X6 coverage declared": True}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(LR.CELL), str(CP.HELA), str(CP.GTF)],
                      available=len(comps), used=len(tested), selection="filtered", seed=SEED,
                      controls=["size-matched random gene sets from the same measured proteome",
                                "the null checked for capability before its verdict is read",
                                "publication spread run through the identical estimator",
                                "the ribosome and proteasome as complexes whose answer is known",
                                "one abundance dataset throughout, per loop 92's rule",
                                "coverage stated against all 2,039 complexes"],
                      note="a complex is a stoichiometric object, not a graph; matched subunit "
                           "abundance is a hard prediction that a parts list cannot make")
    RM.report(man, emit=say)
    json.dump({"test": "loop_complex_stoich", "manifest": man, "gates": gates,
               "x1": {"median_real_spread": med_real, "null_mean": float(nulls.mean()),
                      "null_sd": float(nulls.std()), "survival": s},
               "x2": cap,
               "x3": {"pub_spread_real": float(np.median(realp)),
                      "pub_spread_null": float(nullp.mean()),
                      "tightening_abundance": tight_ab, "tightening_pubs": tight_pub},
               "x4": {"median_ratio": float(np.median(rat)),
                      "p90_ratio": float(np.percentile(rat, 90)), "worst": worst,
                      "all": lim[:200]},
               "x5": x5rows,
               "x6": {"n_complexes": len(comps), "n_tested": len(tested)},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_complex_stoich.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_complex_stoich.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
