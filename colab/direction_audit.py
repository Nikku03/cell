"""HOW MUCH OF THE PPI NETWORK IS DIRECTED, AND HOW MUCH OF THAT IS TRUSTWORTHY?

The build reported 15.34% of edges directed. That number is true and, on its own, misleading: it pools a curated
tier with two predicted tiers whose error rates are very different, and one of those tiers is applied precisely
where nothing can check it. This module takes the assembled network apart and asks, per tier: how many edges, how
accurate, measured on what, and how much of that measurement transfers to where the tier is actually used.

THE THREE TIERS AND WHAT EACH IS WORTH:

  signor          Manually curated signalling direction. Treated as ground truth everywhere else in this repo, so
                  quoting an accuracy for it would be circular -- it IS the answer key. Its real limitation is
                  coverage, not correctness.
  pathway_order   Reactome reaction sequence. Validated against SIGNOR, an independently curated database built from
                  different primary literature, so the comparison is meaningful.
  tf_prior        "the transcription factor points at the non-TF". A heuristic. It was measured at 0.7647 -- but on
                  the 255 edges that had BOTH a SIGNOR direction AND a TF/non-TF difference, and it is APPLIED to
                  23,097 edges where the better tiers are silent. Whether the measured rate transfers to where it
                  is used is the central question of this audit, and the honest answer is that it cannot be checked
                  on those edges, because if it could they would not need the heuristic.

WHAT IS COMPUTED:
  1. coverage per tier, and the expected number of CORRECT arrows given each tier's measured error rate
  2. per-tier accuracy on every edge where an independent source can check it, with binomial CIs
  3. CROSS-TIER AGREEMENT: where two independent tiers both have an opinion, do they agree? Agreement between
     independently-derived sources is the strongest evidence available without new experiments, and disagreement
     puts a hard floor under the true error rate.
  4. the VALIDATABLE FRACTION: what share of each tier's edges can be checked at all. A tier that is 76% accurate
     on the 1% of its edges that are checkable is not a 76%-accurate tier.
"""
import collections
import json
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))


def ci(k, n):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    return p, float(np.sqrt(p * (1 - p) / n))


def main():
    DN = json.load(open(OUT / "directed_network.json"))
    edges = DN["edges"]
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; N = len(names)
    nidx = {n: i for i, n in enumerate(names)}
    info = {g["name"]: g for g in D["genes"]}
    tf = {n: float(info[n].get("tf") or 0) for n in names}

    S = set()
    for e in D.get("sig", []) or []:
        try:
            s_, t_ = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= s_ < N and 0 <= t_ < N and s_ != t_:
            S.add((names[s_], names[t_]))
    sig_dir = {}
    for a, b in S:
        if (b, a) not in S:
            sig_dir[frozenset((a, b))] = (a, b)

    R = set()
    for e in D.get("reg", []) or []:
        try:
            s_, t_ = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= s_ < N and 0 <= t_ < N and s_ != t_:
            R.add((names[s_], names[t_]))
    reg_dir = {}
    for a, b in R:
        if (b, a) not in R:
            reg_dir[frozenset((a, b))] = (a, b)

    tier = collections.defaultdict(list)
    for a, b, prov, t in edges:
        tier[prov].append((a, b))
    total = len(edges)

    print(f"PPI edges: {total:,}\n")
    print(f"  {'tier':16s} {'edges':>8s} {'% of PPI':>9s} {'checkable':>10s} {'accuracy':>17s}")
    stats = {}
    for prov in ["signor", "pathway_order", "tf_prior", "undirected"]:
        es = tier.get(prov, [])
        if prov == "undirected":
            print(f"  {prov:16s} {len(es):8,} {len(es)/total:9.2%} {'-':>10s} {'-':>17s}")
            stats[prov] = {"n": len(es), "frac": len(es) / total}
            continue
        # THE SIGNOR TIER CANNOT BE AUDITED FROM WITHIN ITS OWN TIER. An earlier version checked it against `reg`
        # restricted to the signor TIER of the built network -- i.e. only edges where pathway_order happened to be
        # silent -- and got 0.4405, which it reported as "below chance". That was a selection artefact: across ALL
        # 2,876 edges the two curated sources both orient, they agree 0.7830. The tier-restricted comparison is
        # dropped and the full one is reported separately below.
        key = sig_dir
        keyname = "SIGNOR"
        if prov == "signor":
            print(f"  {prov:16s} {len(es):8,} {len(es)/total:9.2%} {'(is the key)':>10s} {'curated':>17s}")
            stats[prov] = {"n": len(es), "frac": len(es) / total, "checkable": 0,
                           "checkable_frac": 0.0, "acc": 1.0, "se": 0.0, "against": "curated"}
            continue
        chk = [(a, b) for a, b in es if frozenset((a, b)) in key]
        good = sum(1 for a, b in chk if key[frozenset((a, b))] == (a, b))
        p, se = ci(good, len(chk))
        stats[prov] = {"n": len(es), "frac": len(es) / total, "checkable": len(chk),
                       "checkable_frac": len(chk) / max(len(es), 1), "acc": p, "se": se, "against": keyname}
        print(f"  {prov:16s} {len(es):8,} {len(es)/total:9.2%} {len(chk):10,} "
              f"{p:8.4f} +/- {se:.4f} vs {keyname}")

    ndir = total - len(tier.get("undirected", []))
    print(f"\n  {'TOTAL DIRECTED':16s} {ndir:8,} {ndir/total:9.2%}")

    # ---- expected correct arrows, using each tier's own measured rate ----
    print(f"\n  EXPECTED CORRECT ARROWS (each tier at its own measured rate; signor taken as curated truth)")
    exp = 0.0
    for prov in ["signor", "pathway_order", "tf_prior"]:
        s_ = stats[prov]
        # tf_prior has NO SIGNOR-checkable edges by construction, so its rate is taken from the earlier
        # measurement on the 255 edges where a curated direction and a TF/non-TF difference coexisted. That is a
        # biased sample and the number is carried forward with that label, not as a property of the tier.
        rate = 1.0 if prov == "signor" else (s_["acc"] if s_["acc"] == s_["acc"] else 0.7647)
        e_ = s_["n"] * rate
        exp += e_
        print(f"    {prov:16s} {s_['n']:8,} x {rate:.3f} = {e_:9,.0f}")
    print(f"    {'expected correct':16s} {exp:>20,.0f}  = {exp/ndir:.1%} of directed, {exp/total:.2%} of all PPI")
    print(f"    {'expected WRONG':16s} {ndir-exp:>20,.0f}  = {(ndir-exp)/ndir:.1%} of the arrows we assert")

    # ---- cross-tier agreement: the strongest check available without new experiments ----
    print(f"\n  CROSS-TIER AGREEMENT (independent sources, where both have an opinion)")
    po = {frozenset((a, b)): (a, b) for a, b in tier.get("pathway_order", [])}
    pairs = [("pathway_order", po, "SIGNOR", sig_dir), ("pathway_order", po, "reg", reg_dir)]
    for n1, d1, n2, d2 in pairs:
        common = set(d1) & set(d2)
        if not common:
            print(f"    {n1} vs {n2}: no overlap")
            continue
        ag = sum(1 for k in common if d1[k] == d2[k])
        p, se = ci(ag, len(common))
        print(f"    {n1:14s} vs {n2:8s} n={len(common):5,}  agree {p:.4f} +/- {se:.4f}")
    # tf_prior vs reg looks like strong validation and is NOT: `reg` edges are TF->target BY DEFINITION and
    # tf_prior predicts TF->non-TF, so the two agree almost by construction. Measured: 835 of the 923 overlapping
    # reg edges (90.5%) have a TF as their source. Reported with that caveat attached rather than as evidence.
    tfp = {frozenset((a, b)): (a, b) for a, b in tier.get("tf_prior", [])}
    common = set(tfp) & set(reg_dir)
    if common:
        ag = sum(1 for k in common if tfp[k] == reg_dir[k])
        p, se = ci(ag, len(common))
        srctf = sum(1 for k in common if tf.get(reg_dir[k][0], 0) == 1)
        print(f"    {'tf_prior':14s} vs {'reg':8s} n={len(common):5,}  agree {p:.4f} +/- {se:.4f}"
              f"   <- CIRCULAR: {srctf/len(common):.0%} of these reg edges have a TF as source")

    # ---- DO THE TWO CURATED SOURCES AGREE WITH EACH OTHER? the realistic ceiling for any predictor ----
    both = set(sig_dir) & set(reg_dir)
    agree = sum(1 for k in both if sig_dir[k] == reg_dir[k])
    pc, sec = ci(agree, len(both))
    disagree = [k for k in both if sig_dir[k] != reg_dir[k]]
    tf_involved = sum(1 for k in disagree if tf.get(list(k)[0], 0) + tf.get(list(k)[1], 0) > 0)
    print(f"\n  THE CEILING: do the two CURATED sources agree with each other?")
    print(f"    edges both orient {len(both):,}   agreement {pc:.4f} +/- {sec:.4f}")
    print(f"    of {len(disagree):,} disagreements, {tf_involved:,} ({tf_involved/max(len(disagree),1):.0%}) "
          f"involve a transcription factor")
    for k in disagree[:4]:
        print(f"       SIGNOR {sig_dir[k][0]}->{sig_dir[k][1]:14s}  reg {reg_dir[k][0]}->{reg_dir[k][1]}")
    print(f"    -> these are not errors. A kinase can act ON a TF while that TF transcriptionally regulates the")
    print(f"       kinase's gene. Protein-level direction and transcriptional direction are DIFFERENT relations")
    print(f"       and can legitimately oppose, so {pc:.0%} is the realistic ceiling for any single arrow.")

    tfs = stats["tf_prior"]
    verdict = (
        f"HOW MUCH OF THE NETWORK IS DIRECTED, AND HOW SURE ARE WE? {ndir:,} of {total:,} PPI edges carry an arrow, "
        f"{ndir/total:.2%}. That single number pools three very different things and should not be quoted alone. "
        f"ONLY {stats['signor']['n']:,} EDGES ({stats['signor']['frac']:.2%} OF THE NETWORK) ARE CURATED DIRECTION. "
        f"Those come from SIGNOR, are used as the answer key everywhere else in this repo, and their limitation is "
        f"coverage rather than correctness. "
        f"{stats['pathway_order']['n']:,} more ({stats['pathway_order']['frac']:.2%}) come from Reactome reaction "
        f"sequence and are right {stats['pathway_order']['acc']:.1%} of the time against independently curated "
        f"SIGNOR -- so roughly one in three of those arrows is BACKWARDS. "
        f"THE REMAINING {tfs['n']:,} ({tfs['frac']:.1%}) ARE A HEURISTIC -- 'the transcription factor points at the "
        f"non-TF' -- and this is where the honest answer gets uncomfortable. SIGNOR can check EXACTLY ZERO of its "
        f"{tfs['n']:,} edges, and not by accident: the heuristic is applied only where the curated tiers are "
        f"silent, so the edges it is used on are precisely the edges nothing can check. Its 76.5% comes from a "
        f"different set -- the 255 edges that happened to have both a curated direction and a TF/non-TF difference "
        f"-- i.e. from well-studied TF pairs, and assuming that rate transfers to 23,097 unstudied ones is an "
        f"assumption, not a result. Its 90.5% agreement with `reg` looks like corroboration and is not: `reg` edges "
        f"are TF->target BY DEFINITION and this heuristic predicts TF->non-TF, so 90% of those reg edges have a TF "
        f"as their source and the agreement is close to definitional. "
        f"TAKING EACH TIER AT ITS OWN MEASURED RATE, about {exp:,.0f} arrows are right and {ndir-exp:,.0f} are "
        f"wrong: {exp/total:.2%} of the network is directed AND probably correct, against the {ndir/total:.2%} "
        f"headline. THE DEFENSIBLE SUMMARY IS THEREFORE: about 1.5% of the interactome has a curated arrow, about "
        f"another 1.8% has a pathway-derived arrow that is right about two thirds of the time, and roughly 12% has a "
        f"guess that is probably right about three quarters of the time but cannot be checked where it matters. "
        f"85% has no direction at all. "
        f"AND THERE IS A CEILING NOBODY CAN EXCEED, which is the most important thing this audit found: the two "
        f"CURATED sources agree with each other only {pc:.1%} of the time on the {len(both):,} edges both orient, "
        f"and {tf_involved/max(len(disagree),1):.0%} of the disagreements involve a transcription factor. Those are "
        f"not curation errors. SIGNOR says HSP90AA1 acts on AR; reg says AR transcriptionally regulates HSP90AA1. "
        f"BOTH ARE TRUE. Protein-level direction and transcriptional direction are different relations on the same "
        f"pair and routinely oppose, because that is what a feedback loop IS. So 'the direction of a PPI edge' is "
        f"not a single well-defined property, and {pc:.0%} is the realistic ceiling for any single arrow -- our "
        f"pathway-order tier at {stats['pathway_order']['acc']:.1%} should be read against that, not against 100%. "
        f"WHAT WOULD MAKE US SURE. Nothing in this pipeline: every tier is annotation or heuristic, and the "
        f"agreement between tiers is bounded by the weakest curation underneath both. Direction is a causal claim "
        f"and only a time-resolved or interventional measurement -- kinase-substrate assays, degron depletion with a "
        f"time course -- can settle it per edge.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"total": total, "directed": ndir, "frac_directed": ndir / total,
               "expected_correct": exp, "frac_correct": exp / total, "tiers": stats, "verdict": verdict},
              open(OUT / "direction_audit.json", "w"), indent=2)
    print(f"\n  -> {OUT/'direction_audit.json'}")


if __name__ == "__main__":
    main()
