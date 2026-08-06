"""HOW GOOD MUST A STRUCTURAL DISCRIMINATOR BE TO FIND A CATALYST?  The arithmetic that decides the design,
computed from numbers this project has already measured.

THE PROPOSAL.  Use NEXUS the way it was used for interactions: cover the structures in spheres, model the
effect of binding on the whole structure, introduce a third protein, and let the physics say which one is the
catalyst. The mechanism is right -- it asks "does this enzyme act on this substrate", which is the actual
question, unlike co-dependency (viability covariance) or co-expression (shared transcriptional control), both
of which were just measured to add nothing.

THE THING TO CHECK FIRST, because it decides whether this is a SEARCH or a TIE-BREAK.  Finding a catalyst
from scratch means screening the candidate enzyme vocabulary: 5,282 genes. A screen is a ranking problem, and
this project has measured its ranking ability repeatedly:

    NEXUS dual sensor, folding x binding, "does it still work"      AUC 0.755
    dMaSIF-style surface descriptor, unbound                        AUC 0.714
    best docking rescore on interface point pairs                   AUC 0.814
    GeoConv on interface point pairs                                AUC 0.617

For a ranker with AUC A, a positive scored against N negatives sits at expected rank (1-A)*N + 1. That single
formula is what decides the architecture, so it is computed here across the range of A this project has
actually achieved and the range of N the task could present.

WHAT COMES OUT OF IT, stated before the table so it cannot be read as hindsight: to put the true catalyst in
the top 10 of 5,282 candidates you need AUC ~0.998. Nothing in structural biology does that, and nothing in
this project comes within 0.18 of it. So NEXUS cannot BE the search.

But the same arithmetic run at small N says something much more useful, and it is the reason to build this
rather than abandon it: on a shortlist of 10, an AUC of 0.755 is worth real accuracy. And this session already
produced a shortlist generator -- closed-book naming reaches 0.463 top-1 on exactly the obscure enzymes that
make up the gap, and its top-k will be far above that.

So the composition the numbers support is:

    LLM proposes a shortlist  ->  NEXUS discriminates within it

which is the user's instinct applied at the N where the physics can actually pay. This module computes what
top-k accuracy the pair would need, and what it would deliver, so the pilot is designed against a number
instead of a hope.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A

OUT = A.OUT

# every one of these was measured in this project, not quoted from a paper
MEASURED = (("GeoConv, interface point pairs", 0.617),
            ("dMaSIF surface descriptor, unbound", 0.714),
            ("NEXUS dual sensor (fold x bind)", 0.755),
            ("best docking rescore", 0.814))
CANDIDATE_SPACE = 5282          # distinct catalyst genes in the network
SHORTLISTS = (5, 10, 20, 50, 200, 1000, 5282)


def expected_rank(auc, n):
    """A positive scored against n negatives by a ranker of the given AUC sits, in expectation, at
    (1-auc)*n + 1. AUC IS the probability a random positive outranks a random negative, so the expected
    number of negatives placed above it is exactly (1-auc)*n."""
    return (1.0 - auc) * n + 1.0


def auc_needed(n, target_rank):
    """the AUC required to land the positive within target_rank of n negatives, in expectation"""
    return 1.0 - (target_rank - 1.0) / n


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("CAN A STRUCTURAL DISCRIMINATOR FIND A CATALYST? -- the arithmetic that decides the design")
    report("=" * 100)
    report("  The mechanism is right: unlike co-dependency and co-expression, a binding calculation asks")
    report("  'does this enzyme act on this substrate', which IS the question. So the issue is not whether")
    report("  the signal is relevant. It is whether the signal is SHARP ENOUGH for the size of the search.")
    report(f"  Candidate enzyme vocabulary: {CANDIDATE_SPACE:,} genes.")

    report(f"\n  WHERE THE TRUE CATALYST LANDS, by discriminator quality and shortlist size")
    report(f"  (expected rank of the true catalyst; a screen is only useful if this is single digits)")
    report(f"    {'discriminator':<36}{'AUC':>7}" + "".join(f"{('N=' + str(n)):>10}" for n in SHORTLISTS))
    rows = {}
    for nm, a in MEASURED:
        r = [expected_rank(a, n - 1) for n in SHORTLISTS]
        rows[nm] = {"auc": a, "ranks": dict(zip(map(str, SHORTLISTS), r))}
        report(f"    {nm:<36}{a:>7.3f}" + "".join(f"{x:>10.1f}" for x in r))

    report(f"\n  THE AUC A FULL SCREEN WOULD REQUIRE")
    need = {}
    for k in (1, 5, 10, 50):
        v = auc_needed(CANDIDATE_SPACE - 1, k)
        need[str(k)] = v
        report(f"    to place the true catalyst in the top {k:<3} of {CANDIDATE_SPACE:,}:  AUC >= {v:.4f}")
    best_nm, best_a = max(MEASURED, key=lambda kv: kv[1])
    gap = need["10"] - best_a
    report(f"    best measured in this project: {best_a:.3f} ({best_nm}) -- short by {gap:.3f}")

    report("\n  READING")
    report(f"  NEXUS CANNOT BE THE SEARCH. Screening {CANDIDATE_SPACE:,} enzymes needs AUC "
           f"{need['10']:.4f} to reach a")
    report(f"  top-10; the best structural number this project has measured is {best_a:.3f}, which leaves the")
    report(f"  true catalyst near rank {expected_rank(best_a, CANDIDATE_SPACE-1):.0f}. That is the same wall the")
    report("  docking line already hit seven different ways -- generation worked, ranking did not -- and")
    report("  nothing about spheres or a third protein changes the size of the candidate space.")
    report("")
    report("  NEXUS AS A TIE-BREAK IS A DIFFERENT PROPOSITION, and the arithmetic supports it. On a shortlist")
    for n in (5, 10, 20):
        report(f"    of {n:>2}: AUC {best_a:.3f} puts the true catalyst at expected rank "
               f"{expected_rank(best_a, n-1):.1f}, and the NEXUS sensor at "
               f"{expected_rank(0.755, n-1):.1f}")
    report("  which is exactly the regime where a physics score can convert a near-miss into a hit.")
    report("")
    report("  AND THIS SESSION ALREADY BUILT THE SHORTLIST GENERATOR. Closed-book naming reaches 0.463 top-1")
    report("  on the obscure enzymes that make up the gap; its top-10 will be well above that. So the")
    report("  composition the numbers support is:")
    report("      LLM proposes ~10 candidates  ->  structure/NEXUS discriminates within them")
    report("  That is the original instinct applied at the N where the physics can actually pay, and it is")
    report("  the design the pilot should test. What it needs measured next, in order:")
    report("     1. top-10 accuracy of the closed-book arm (is the answer even in the shortlist?)")
    report("     2. AUC of the binding score on true-catalyst vs 9 decoys, with AlphaFold structures")
    report("     3. whether 1 and 2 compose, against the 0.463 top-1 the LLM already delivers alone")
    report("  Step 2 is only worth running if step 1 clears roughly 0.7 -- otherwise the discriminator is")
    report("  sorting a list that does not contain the answer, which is the 0.505 ceiling all over again.")

    report(f"\n  SCOPE, measured separately: of the 9,186 enzyme-shaped gaps, 2,143 (23%) have a protein or")
    report("  complex participant to dock against. The other 7,043 are metabolite-in-a-pocket, a different")
    report("  calculation than the protein-protein one NEXUS performs.")

    json.dump({"test": "nexus_screen_feasibility", "candidate_space": CANDIDATE_SPACE,
               "shortlists": list(SHORTLISTS), "rows": rows, "auc_needed": need,
               "protein_shaped_gap": 2143, "total_enzyme_gap": 9186, "log": log},
              open(OUT / "nexus_screen_feasibility.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'nexus_screen_feasibility.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
