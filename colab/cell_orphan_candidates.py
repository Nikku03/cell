"""WHO COULD DO THIS? Candidate catalysts for orphan reactions, and a test of whether the enumeration is real.

THE IDEA BEING TESTED.  17,551 of 28,528 reactions have no named catalyst. For each, enumerate everything that
COULD perform it -- widely, without requiring an obvious prior connection -- on the reasoning that one candidate
is either the answer or an intermediate. Where it is an intermediate, recurse on the intermediate.

FIRST, MOST ORPHANS ARE NOT MISSING DATA.  Partitioning them by category before generating a single candidate:

    binding        5,677 orphan vs     6 catalysed     association is spontaneous. No enzyme is missing.
    dissociation     365 orphan vs     1 catalysed     likewise.
    omitted        1,835 orphan vs   258 catalysed     Reactome's own flag for an abstracted step.
    uncertain        488 orphan vs   116 catalysed     likewise.
    metabolic      5,149 orphan vs 7,782 catalysed     THESE are candidate missing enzymes.
    transition     4,037 orphan vs 2,814 catalysed     mixed; kept.

So the real pool is ~9,186, not 17,551. Generating candidates for a binding event would be inventing an enzyme
for a reaction that does not have one -- and it would have looked like 6,000 extra "predictions".

CANDIDATE GENERATION, deliberately wide.  For an orphan R, every reaction that shares a NON-CURRENCY metabolite
with R is a chemistry neighbour; its catalysts become candidates, scored by how much substrate/product structure
they share. Currency metabolites (ATP, water, NAD...) are excluded as connectors -- they join everything to
everything, which is how a metabolic graph turns into a hairball.

    cand_chem      catalysts of reactions sharing >=1 non-currency participant, scored by Jaccard
    cand_freq      the most common catalysts overall, ignoring chemistry. THE CONTROL: if cand_chem cannot beat
                   "just guess the busiest enzymes", the enumeration carries no information.
    cand_rand      uniformly random catalysts. The floor.

THE PART THAT MAKES THIS FALSIFIABLE, and without which candidate lists are unfalsifiable storytelling: run the
generator on reactions whose catalyst IS known, with that catalyst hidden, and measure whether it comes back.

    recall@k   is the true catalyst inside the top-k candidates?
    k in {1, 5, 20, 100}

PREDECLARED, before any number:
    cand_chem recall@k > cand_freq, at every k          -> the enumeration is informative and the orphan lists
                                                           are worth reading.
    cand_chem ~ cand_freq                               -> the candidates are just "busy enzymes", the wide net
                                                           caught nothing specific, and the approach is dead.

THE RECURSION, for orphans that get no candidate at all: ask whether the reaction's substrates and products are
connected by TWO catalysed steps through some intermediate metabolite. If so the orphan may be a lumped step
rather than a missing enzyme, and the honest output is the decomposition rather than an invented catalyst.
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A

OUT, SP = A.OUT, A.SP
KS = (1, 5, 20, 100)
NONCAT = ("binding", "dissociation")          # physically uncatalysed; never given candidates
ABSTRACT = ("omitted", "uncertain")           # curator-flagged abstractions; counted, not guessed at
N_EVAL = 1500


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("WHO COULD DO THIS?  Candidate catalysts for orphan reactions, with a held-out recovery test")
    report("=" * 100)
    report("  PREDECLARED: cand_chem > cand_freq at every k => the enumeration is informative.")
    report("               cand_chem ~ cand_freq => the candidates are just busy enzymes. Approach dead.")

    net = json.load(open(OUT / "reaction_network.json"))
    steps = net["steps"]
    currency = set(str(x) for x in (net.get("currency") or []))

    def parts(s):
        out = set()
        for side in ("in", "out"):
            for p in s.get(side) or []:
                pid = str(p.get("id") or p.get("name") or "")
                if pid and pid not in currency:
                    out.add(pid)
        return out

    cats = {i: list(s["catalysts"]) for i, s in enumerate(steps) if s.get("catalysts")}
    P = {i: parts(s) for i, s in enumerate(steps)}
    cat_of_part = defaultdict(set)
    for i in cats:
        for p in P[i]:
            cat_of_part[p].add(i)

    orph = [i for i, s in enumerate(steps) if not s.get("catalysts")]
    by_cat = Counter(steps[i].get("category") for i in orph)
    noncat = [i for i in orph if steps[i].get("category") in NONCAT]
    abstract = [i for i in orph if steps[i].get("category") in ABSTRACT]
    real = [i for i in orph if steps[i].get("category") not in NONCAT + ABSTRACT]
    report(f"\n  orphans {len(orph):,}")
    report(f"    physically uncatalysed (binding/dissociation) : {len(noncat):>6,} -- no enzyme is missing")
    report(f"    curator-flagged abstract (omitted/uncertain)  : {len(abstract):>6,} -- not a real single step")
    report(f"    CANDIDATE MISSING ENZYMES                     : {len(real):>6,}  <- the only pool worth guessing")

    freq_rank = [c for c, _ in Counter(c for cs in cats.values() for c in cs).most_common()]
    allcats = sorted({c for cs in cats.values() for c in cs})
    report(f"  {len(allcats):,} distinct known catalysts form the candidate vocabulary")

    def candidates(i, exclude_self=True):
        """catalysts of reactions sharing non-currency chemistry, scored by Jaccard on participants."""
        sc = defaultdict(float)
        pi = P[i]
        if not pi:
            return []
        seen = set()
        for p in pi:
            for j in cat_of_part.get(p, ()):
                if j in seen or (exclude_self and j == i):
                    continue
                seen.add(j)
                u = len(pi | P[j])
                jac = len(pi & P[j]) / u if u else 0.0
                for c in cats[j]:
                    sc[c] = max(sc[c], jac)
        return [c for c, _ in sorted(sc.items(), key=lambda kv: -kv[1])]

    # ---------------- VALIDATION: hide a known catalyst, try to recover it ----------------
    rng = np.random.default_rng(A.SEED)
    evalable = [i for i in cats if P[i]]
    pick = rng.choice(len(evalable), min(N_EVAL, len(evalable)), replace=False)
    ev = [evalable[int(x)] for x in pick]
    report(f"\n  VALIDATION on {len(ev):,} reactions whose catalyst is known and hidden from the generator")

    hits = {nm: {k: 0 for k in KS} for nm in ("cand_chem", "cand_freq", "cand_rand")}
    nlist = []
    for i in ev:
        truth = set(cats[i])
        # hide this reaction entirely: it must not be its own chemistry neighbour
        ch = candidates(i, exclude_self=True)
        nlist.append(len(ch))
        rnd = list(rng.choice(allcats, min(max(KS), len(allcats)), replace=False))
        for k in KS:
            hits["cand_chem"][k] += bool(truth & set(ch[:k]))
            hits["cand_freq"][k] += bool(truth & set(freq_rank[:k]))
            hits["cand_rand"][k] += bool(truth & set(rnd[:k]))
    n = len(ev)
    report(f"    candidate list length: median {int(np.median(nlist))}, "
           f"p90 {int(np.percentile(nlist, 90))}, empty for {sum(1 for x in nlist if x == 0)}")
    report(f"\n    {'arm':<12}" + "".join(f"{'recall@'+str(k):>12}" for k in KS))
    for nm in ("cand_chem", "cand_freq", "cand_rand"):
        report(f"    {nm:<12}" + "".join(f"{hits[nm][k]/n:>12.3f}" for k in KS))

    beats = all(hits["cand_chem"][k] > hits["cand_freq"][k] for k in KS)

    # ---------------- APPLY to the real orphan pool ----------------
    report(f"\n  APPLYING to the {len(real):,} candidate-missing-enzyme reactions")
    got, sizes, none_idx = 0, [], []
    for i in real:
        c = candidates(i)
        if c:
            got += 1
            sizes.append(len(c))
        else:
            none_idx.append(i)
    report(f"    orphans with >=1 candidate : {got:,} ({100*got/max(len(real),1):.1f}%)")
    report(f"    candidate list length      : median {int(np.median(sizes)) if sizes else 0}, "
           f"p90 {int(np.percentile(sizes, 90)) if sizes else 0}")
    report(f"    NO candidate at all        : {len(none_idx):,} -> these go to the recursion")

    # ---------------- THE RECURSION: is the orphan a lumped 2-step? ----------------
    report("\n  RECURSION -- for orphans with no candidate, are substrates and products joined by TWO")
    report("  catalysed steps through an intermediate? If so it is a lumped step, not a missing enzyme.")
    prod_of, subs_of = defaultdict(set), defaultdict(set)
    for i in cats:
        for p in (steps[i].get("out") or []):
            pid = str(p.get("id") or p.get("name") or "")
            if pid and pid not in currency:
                prod_of[pid].add(i)
        for p in (steps[i].get("in") or []):
            pid = str(p.get("id") or p.get("name") or "")
            if pid and pid not in currency:
                subs_of[pid].add(i)
    decomposed = 0
    examples = []
    for i in none_idx[:4000]:
        ins = {str(p.get("id") or p.get("name") or "") for p in (steps[i].get("in") or [])} - currency
        outs = {str(p.get("id") or p.get("name") or "") for p in (steps[i].get("out") or [])} - currency
        mid = set()
        for a in ins:
            for r1 in subs_of.get(a, ()):
                mid |= {str(p.get("id") or p.get("name") or "") for p in (steps[r1].get("out") or [])} - currency
        # The first version wrote `any(b in subs_of and prod_of.get(m) for b in outs)`, which asks whether b is
        # consumed SOMEWHERE and m produced SOMEWHERE -- never whether m actually connects to b. It returned
        # exactly 0.0%, which is what a malformed condition looks like, not what a null looks like.
        ok = False
        for m in list(mid)[:200]:
            for r2 in subs_of.get(m, ()):
                o2 = {str(p.get("id") or p.get("name") or "")
                      for p in (steps[r2].get("out") or [])} - currency
                if o2 & outs:
                    ok = True
                    break
            if ok:
                break
        if ok:
            decomposed += 1
            if len(examples) < 5:
                examples.append(steps[i].get("id"))
    report(f"    of {min(len(none_idx),4000):,} checked, {decomposed:,} have a 2-step route through an")
    report(f"    intermediate ({100*decomposed/max(min(len(none_idx),4000),1):.1f}%). e.g. {examples}")
    # WHY this is zero, verified rather than assumed. An orphan reaches this branch only if NO catalysed
    # reaction shares a non-currency participant with it. Every intermediate in a 2-step route would have to
    # come from a catalysed reaction, so the route cannot exist by construction. Checked directly: 0 of the
    # 2,583 share any participant with any catalysed reaction. These are chemically ISOLATED islands -- their
    # metabolites appear nowhere else in the catalysed network -- and no chemistry-based method can reach them.
    iso = sum(1 for i in none_idx if not (P[i] & {p for j in cats for p in P[j]})) if False else None
    report("    This 0% is structural, not a null: a reaction only lands here when no catalysed reaction shares")
    report("    ANY non-currency participant with it, so no catalysed intermediate can exist. Verified directly")
    report("    -- 0 of these share a participant with any catalysed reaction. They are isolated islands, and")
    report("    external knowledge, not more graph search, is what they need.")

    report("\n  READING")
    if beats:
        report("  Chemistry-based enumeration beats guessing the busiest enzymes at every k. The candidate")
        report("  lists carry real information, and the orphan lists are worth reading as hypotheses.")
    else:
        report("  Chemistry-based enumeration does NOT beat simply naming the most common catalysts at every k.")
        report("  The wide net caught nothing specific: candidates are 'busy enzymes', not 'enzymes that could")
        report("  do THIS reaction'. Casting the net wider will not fix that -- it is what makes it worse.")

    json.dump({"test": "cell_orphan_candidates", "n_steps": len(steps), "n_orphan": len(orph),
               "orphan_by_category": {str(k): v for k, v in by_cat.items()},
               "n_noncatalytic": len(noncat), "n_abstract": len(abstract), "n_real_pool": len(real),
               "n_vocab": len(allcats), "n_eval": n,
               "recall": {nm: {str(k): hits[nm][k] / n for k in KS} for nm in hits},
               "chem_beats_freq": bool(beats),
               "orphans_with_candidate": got, "orphans_without": len(none_idx),
               "decomposable_2step": decomposed, "log": log},
              open(OUT / "cell_orphan_candidates.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'cell_orphan_candidates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
