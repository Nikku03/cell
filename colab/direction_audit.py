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
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")


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
        # check against SIGNOR, and for the signor tier itself against reg (a different, independent source)
        key = reg_dir if prov == "signor" else sig_dir
        keyname = "reg" if prov == "signor" else "SIGNOR"
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
        rate = 1.0 if prov == "signor" else s_["acc"]
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
    # tf_prior checked against BOTH curated sources
    tfp = {frozenset((a, b)): (a, b) for a, b in tier.get("tf_prior", [])}
    for n2, d2 in [("SIGNOR", sig_dir), ("reg", reg_dir)]:
        common = set(tfp) & set(d2)
        if common:
            ag = sum(1 for k in common if tfp[k] == d2[k])
            p, se = ci(ag, len(common))
            print(f"    {'tf_prior':14s} vs {n2:8s} n={len(common):5,}  agree {p:.4f} +/- {se:.4f}")

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
        f"non-TF' -- and this is where the honest answer gets uncomfortable. It measures {tfs['acc']:.1%} on the "
        f"{tfs['checkable']:,} of its edges a curated source can check, which is {tfs['checkable_frac']:.1%} of the "
        f"tier. The other {100*(1-tfs['checkable_frac']):.1f}% are unvalidatable BY CONSTRUCTION: the heuristic is "
        f"applied exactly where the curated sources are silent, so the edges where it is used are precisely the "
        f"edges where it cannot be checked. Its measured rate comes from a biased sample -- TF pairs well studied "
        f"enough to be curated -- and assuming it transfers is an assumption, not a result. "
        f"TAKING EACH TIER AT ITS OWN MEASURED RATE, about {exp:,.0f} arrows are right and {ndir-exp:,.0f} are "
        f"wrong: {exp/total:.2%} of the network is directed AND probably correct, against the {ndir/total:.2%} "
        f"headline. THE DEFENSIBLE SUMMARY IS THEREFORE: about 1.5% of the interactome has a curated arrow, about "
        f"another 1.8% has a pathway-derived arrow that is right about two thirds of the time, and roughly 12% has a "
        f"guess that is probably right about three quarters of the time but cannot be checked where it matters. "
        f"85% has no direction at all. "
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
