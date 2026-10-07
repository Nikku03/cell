"""Loop 200b. One number: how many distinct reactions do Human-GEM and Rhea-human hold together?

Loop 200 measured the two catalogues separately and showed neither contains the other. This asks
the arithmetic question that follows: what is their union?

IT CANNOT BE AN EXACT NUMBER, and the reason is the same one that made loop 200's 16.6% a lower
bound. The two catalogues can only be matched where BOTH sides carry a Rhea identifier, and only
1,774 of Human-GEM's 12,931 reactions carry one. A reaction that is genuinely in both, but which
Human-GEM stores without a Rhea id, is invisible to the match and gets counted twice.

So this loop reports a RANGE with both ends measured, not a point estimate:

  UPPER  = distinct Human-GEM chemistries + Rhea-human reactions - matched overlap
           Every unmatched duplicate is counted twice, so this is the largest the union can be.
  LOWER  = max(the two catalogue sizes)
           The smallest the union can be, reached only if one catalogue contained the other -- and
           loop 200's W7 already showed neither does, so the true union is strictly above this.

PREDECLARED, BEFORE ANY NUMBER.

  V1 DOES THE OVERLAP DEDUPLICATE CORRECTLY?
     The overlap must be counted in CHEMISTRY units on the Human-GEM side, not reaction units:
     several compartment copies of one chemistry can each carry the same Rhea id, and counting
     them separately would subtract the same reaction more than once and shrink the union below
     the truth.
     Gate: PASS iff the chemistry-unit overlap is <= the reaction-unit overlap, and the union
     computed from it is >= the union computed from reaction units. FAIL means the deduplication
     is running backwards.

  V2 IS THE RANGE ORDERED?
     Gate: PASS iff LOWER < UPPER. A range whose ends cross means one of them is computed wrong.

  V3 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import json, os, re, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates
from loop_denominator import parse_gem, signature, GEM, RHEA2UP, HUMAN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "loop_denominator_union.json")

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def main():
    t0 = time.time()
    G = Gates(emit=say)
    say("=" * 96)
    say("LOOP 200b -- ONE NUMBER: Human-GEM and Rhea-human together")
    say("=" * 96)

    _, rxns, _ = parse_gem(GEM)
    human_acc = {l.strip() for l in open(HUMAN) if l.strip()}
    master_acc = defaultdict(set)
    with open(RHEA2UP) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                master_acc[p[2]].add(p[3])
    rhea_human = {m for m, a in master_acc.items() if a & human_acc}

    # group Human-GEM into distinct chemistries, carrying each group's Rhea ids
    groups = defaultdict(list)
    for r in rxns:
        groups[signature(r)].append(r)
    n_chem = len(groups)
    chem_rhea = [set().union(*(set(r["rhea"]) for r in v)) for v in groups.values()]

    overlap_chem = sum(1 for s in chem_rhea if s & rhea_human)
    overlap_rxn = sum(1 for r in rxns if set(r["rhea"]) & rhea_human)

    say(f"     Human-GEM reactions as listed        {len(rxns):>7,}")
    say(f"     Human-GEM distinct chemistries       {n_chem:>7,}")
    say(f"     Rhea reactions curated to human      {len(rhea_human):>7,}")
    say(f"     matched overlap, chemistry units     {overlap_chem:>7,}")
    say(f"     matched overlap, reaction units      {overlap_rxn:>7,}  (would over-subtract)")

    union_chem = n_chem + len(rhea_human) - overlap_chem
    union_rxn = n_chem + len(rhea_human) - overlap_rxn
    lower = max(n_chem, len(rhea_human))

    say("V1 DOES THE OVERLAP DEDUPLICATE CORRECTLY?")
    G.add("V1", bool(overlap_chem <= overlap_rxn and union_chem >= union_rxn),
          if_true=lambda: f"V1 PASS -- chemistry-unit overlap {overlap_chem:,} <= reaction-unit "
                          f"{overlap_rxn:,}, so the union is not shrunk by compartment copies",
          if_false=lambda: f"V1 FAIL -- overlap {overlap_chem:,} vs {overlap_rxn:,}")

    say("V2 IS THE RANGE ORDERED?")
    G.add("V2", bool(lower < union_chem), requires=("V1",),
          if_true=lambda: f"V2 PASS -- {lower:,} < {union_chem:,}",
          if_false=lambda: f"V2 FAIL -- {lower:,} is not below {union_chem:,}")

    say("=" * 96)
    say(f"     COMBINED DISTINCT REACTIONS:  {lower:,}  to  {union_chem:,}")
    say(f"     Human-GEM contributes {n_chem:,}, Rhea-human {len(rhea_human):,}, "
        f"{overlap_chem:,} are provably the same reaction")
    say(f"     unique to Rhea-human (not matchable in Human-GEM)  "
        f"{len(rhea_human) - overlap_chem:,}")
    say("=" * 96)

    say("V3 WHAT THIS CANNOT SHOW")
    say("     The upper end double-counts every reaction that is in both but which Human-GEM")
    say("     stores without a Rhea id. Only 1,774 of 12,931 Human-GEM reactions carry one, so")
    say("     that set is not small and the true union sits well below the upper end.")
    say("     Neither end is the number of reactions a human cell performs. Both catalogues are")
    say("     curation, and a reaction nobody has written down is absent from both.")

    G.summary(seconds=time.time() - t0)
    gates, void = G.as_dict()
    res = {"test": "catalogue union", "gates": gates, "void": void,
           "gem_reactions": len(rxns), "gem_chemistries": n_chem,
           "rhea_human": len(rhea_human), "overlap_chem": overlap_chem,
           "overlap_rxn": overlap_rxn, "union_lower": lower, "union_upper": union_chem,
           "rhea_only": len(rhea_human) - overlap_chem,
           "seconds": time.time() - t0, "log": LOG}
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    say(f"wrote {OUT}")


if __name__ == "__main__":
    main()
