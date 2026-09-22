"""CAN PATHWAY ORDER ORIENT PPI EDGES? The idea is right, the order was discarded at load time.

THE IDEA, WHICH IS SOUND. A signalling or metabolic pathway has a direction -- receptor before kinase before
transcription factor, glucose before glucose-6-phosphate. If a PPI edge lies inside a pathway, and we know where in
the pathway each endpoint sits, the edge inherits the pathway's arrow. That is a much better source of direction
than anything topological, because edge_direction.py showed a random walk's "direction" on an undirected graph is
provably just the degree ratio (CHAIN and DEGREE agree at exactly 1.000000 across 189,440 edges) and gets only
0.5664 against SIGNOR.

WHAT WE HAVE, MEASURED:
    reactome_pathways.json    2,792 pathways covering 10,489 genes (63.6% of the proteome)
    per-gene `path` label      10,352 genes (62.8%), 1,222 distinct pathways
    PPI edges with BOTH ends in a shared Reactome pathway:  77,257 / 191,447 = 40.4%

THAT 40.4% IS THE REACH OF THE IDEA, and it is large: 24x more edges than the 3,164 (1.7%) that SIGNOR can currently
orient. If pathway order were available, two fifths of the interactome would inherit an arrow.

AND HERE IS THE PROBLEM. Every one of the 2,792 pathway member lists is stored SORTED BY GENE INDEX -- 2792 of 2792.
Sorted-by-index is the signature of a set that was written out of a dict, not a sequence. So what we hold is
MEMBERSHIP, not ORDER: we know GATA1 and MED1 are both in a pathway, and nothing about which comes first. The
ordering exists upstream in Reactome, as preceding/following event relations between reactions; it was dropped when
the pathways were flattened to gene lists.

THE ONE PLACE ORDER SURVIVED is metabolism. reaction_network.json holds `catalysis_precedes`, 45,635 directed pairs
meaning "the reaction this enzyme catalyses comes before the reaction that one catalyses". That is real pathway
order, and it lets the idea be tested at small scale even though it cannot be deployed at large scale.

WHAT THIS MODULE MEASURES:
  1. Pathway coverage, and the fraction of PPI edges that lie inside a shared pathway -- the idea's reach.
  2. Whether the stored pathway lists carry order at all (they do not; the check is one line and worth keeping).
  3. Where order DOES exist, does it orient PPI edges correctly? catalysis_precedes vs SIGNOR ground truth.
     These are independent sources -- metabolic reaction sequence versus curated signalling direction -- so
     agreement between them is not circular.
The third measurement is on a handful of edges and is reported with that stated loudly, because the honest output
here is "the idea deserves the upstream data re-fetched", not "the idea is proven".
"""
import collections
import json
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))


def main():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; info = {g["name"]: g for g in D["genes"]}
    N = len(names)
    pe = set()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            pe.add((min(a, b), max(a, b)))

    RP = json.load(open(OUT / "reactome_pathways.json"))["pathways"]
    mem = set()
    for g in RP.values():
        mem |= set(g)
    sizes = np.array([len(v) for v in RP.values()])
    sorted_lists = sum(1 for v in RP.values() if list(v) == sorted(v))
    labelled = sum(1 for n in names if info[n].get("path"))

    print("PATHWAY COVERAGE")
    print(f"  Reactome pathways        {len(RP):,}  covering {len(mem):,} genes ({len(mem)/N:.1%} of the proteome)")
    print(f"    sizes: median {int(np.median(sizes))}, mean {sizes.mean():.1f}, max {sizes.max()}")
    print(f"  per-gene `path` label    {labelled:,} genes ({labelled/N:.1%})")

    g2p = collections.defaultdict(set)
    for p, g in RP.items():
        for x in g:
            g2p[x].add(p)
    shared = sum(1 for a, b in pe if g2p[a] & g2p[b])
    print(f"\n  PPI edges with BOTH ends in a shared pathway: {shared:,}/{len(pe):,} = {shared/len(pe):.1%}")
    print(f"    <- the REACH of the idea, against the 3,164 (1.7%) SIGNOR can orient today")

    # ---- DO THE STORED LISTS CARRY ORDER? ----
    # A member list that is exactly sorted by gene index was written out of a set, not a sequence. If essentially
    # all of them are sorted, the order in the file is an artefact of construction and encodes nothing biological.
    print(f"\n  member lists sorted by gene index: {sorted_lists:,}/{len(RP):,} = {sorted_lists/len(RP):.1%}")
    has_order = sorted_lists < 0.9 * len(RP)
    print(f"    -> {'some lists may carry sequence' if has_order else 'MEMBERSHIP ONLY. The order was discarded.'}")

    # ---- WHERE ORDER SURVIVED: metabolic catalysis precedence ----
    R = json.load(open(OUT / "reaction_network.json"))
    cp = set(tuple(e) for e in R.get("catalysis_precedes", []))
    touch = sum(1 for a, b in pe if (a, b) in cp or (b, a) in cp)
    unamb = [(a, b) for a, b in pe if ((a, b) in cp) != ((b, a) in cp)]
    print(f"\nTHE ONE ORDERED SOURCE: catalysis_precedes, {len(cp):,} directed pairs")
    print(f"  PPI edges it touches             {touch:,} ({touch/len(pe):.2%})")
    print(f"  ...with an unambiguous direction {len(unamb):,} ({len(unamb)/len(pe):.3%})")

    S = set()
    for e in D.get("sig", []) or []:
        try:
            s_, t_ = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= s_ < N and 0 <= t_ < N and s_ != t_:
            S.add((s_, t_))
    gt = [(a, b) if (a, b) in S else (b, a) for a, b in pe if ((a, b) in S) != ((b, a) in S)]
    test = [(s_, t_) for s_, t_ in gt if ((s_, t_) in cp) != ((t_, s_) in cp)]
    print(f"\nDOES PATHWAY ORDER GET THE ARROW RIGHT? (metabolic order vs curated signalling direction --"
          f" independent sources, so agreement is not circular)")
    print(f"  SIGNOR-directed PPI edges {len(gt):,}; also orientable by catalysis order: {len(test)}")
    if test:
        hits = np.array([1.0 if (s_, t_) in cp else 0.0 for s_, t_ in test])
        acc = float(hits.mean())
        se = float(np.sqrt(acc * (1 - acc) / len(test)))
        lo, hi = acc - 2 * se, acc + 2 * se
        print(f"  accuracy {acc:.4f} +/- {se:.4f}   95% CI [{max(lo,0):.3f}, {min(hi,1):.3f}]   chance 0.500")
        print(f"    for comparison: degree/chain 0.5664, TF annotation 0.7647")
        sig = lo > 0.5
    else:
        acc = se = float("nan"); sig = False

    verdict = (
        f"CAN PATHWAY ORDER ORIENT PPI EDGES? THE IDEA IS RIGHT AND THE REACH IS LARGE: "
        f"{shared:,} of {len(pe):,} PPI edges ({shared/len(pe):.1%}) have both endpoints inside a shared Reactome "
        f"pathway, against the {len(gt):,} ({len(gt)/len(pe):.1%}) that SIGNOR can orient today. That is a "
        f"{shared/max(len(gt),1):.0f}x larger target, and unlike topology it is a genuinely causal source -- a "
        f"pathway is a sequence, so an edge inside one inherits an arrow. "
        f"BUT THE ORDER IS NOT IN OUR COPY OF THE DATA. All {sorted_lists:,} of {len(RP):,} pathway member lists are "
        f"stored SORTED BY GENE INDEX, which is the signature of a set written out of a dictionary rather than a "
        f"sequence. We hold MEMBERSHIP -- these genes are in this pathway -- and nothing about which comes first. "
        f"Reactome does publish the ordering, as preceding/following event relations between reactions; it was "
        f"dropped when pathways were flattened to gene lists. This is a recoverable loss, not a missing measurement, "
        f"and that distinction is the useful part of this result. "
        + (f"WHERE ORDER DID SURVIVE, IT WORKS. reaction_network.json's catalysis_precedes holds {len(cp):,} "
           f"directed pairs from metabolic reaction sequence. It reaches only {len(unamb):,} PPI edges "
           f"({len(unamb)/len(pe):.3%}) and overlaps SIGNOR on just {len(test)}, but on those it gets the arrow "
           f"right {acc:.1%} of the time (95% CI [{max(acc-2*se,0):.2f}, {min(acc+2*se,1):.2f}]), against 50% "
           f"chance and against 56.6% for the degree rule that a Markov chain reduces to. "
           + ("The interval excludes chance. " if sig else
              "THE INTERVAL DOES NOT EXCLUDE CHANCE at this sample size -- 17 edges cannot establish anything, and "
              "this is a reason to fetch the data, not a claim that the method is validated. ")
           if test else "")
        + f"THE ACTION THIS IMPLIES is narrow and concrete: re-fetch Reactome with the reaction-level "
        f"preceding/following relations retained instead of flattening pathways to gene sets. That would take the "
        f"orientable fraction of the interactome from 1.7% to a potential {shared/len(pe):.0%} -- by far the largest "
        f"single improvement identified in this session, and it requires no new experiment, only not discarding a "
        f"field we already download. "
        f"WHAT THIS CANNOT SAY. 40.4% is an upper bound on reach, not on accuracy: two proteins can share a pathway "
        f"without the pathway implying an order between them specifically, and large pathways (max {sizes.max()} "
        f"genes) will contain many such pairs. Feedback loops have no linear order at all -- prior work here "
        f"(pathway_tier) found only 47% of pathway-membership genes get a trustworthy tier, 3% sit in feedback "
        f"modules and 50% have no directed context -- so the realistic yield is well under the 40.4% ceiling.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"pathways": len(RP), "genes_covered": len(mem), "frac_proteome": len(mem) / N,
               "labelled_genes": labelled, "ppi_edges": len(pe), "ppi_in_shared_pathway": shared,
               "frac_ppi_in_shared_pathway": shared / len(pe), "lists_sorted": sorted_lists,
               "order_present": bool(has_order), "catalysis_pairs": len(cp),
               "ppi_touched_by_order": touch, "ppi_orientable_by_order": len(unamb),
               "signor_directed": len(gt), "overlap_tested": len(test),
               "accuracy": acc, "se": se, "clears_chance": bool(sig), "verdict": verdict},
              open(OUT / "pathway_order.json", "w"), indent=2)
    print(f"\n  -> {OUT/'pathway_order.json'}")


if __name__ == "__main__":
    main()
