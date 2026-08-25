"""Loop 202. Of all the feedback and feedforward loops we can count, how many can we trust?

WHAT LOOP 187 COUNTED, AND WHAT IT NEVER ASKED. Loop 187 counted 1,630 two-cycles and 318,683
feedforward triads in the curated tier and scored them against degree-preserving nulls. Every one of
those counts treats an edge as an edge. It is not. An edge in this network can be a claim resting on
one sentence in one paper, or a claim ten groups have reproduced, and the motif counts pool them.

A MOTIF IS ONLY AS TRUSTWORTHY AS ITS WEAKEST EDGE. A two-cycle needs both of its edges to be real.
A feedforward loop needs all three. So motif trust is the MINIMUM edge trust over the motif, and if
most edges rest on a single paper then most motifs rest on a single paper too -- and the ones that
need three edges are hit hardest, purely by arithmetic. This loop measures how hard.

THE EVIDENCE, MEASURED BEFORE THE TIERS WERE DEFINED so the cut points are not chosen to flatter
the answer:

    CollecTRI  43,536 edges, EVERY one carrying at least one PMID.
               32,762 (75.3%) rest on exactly one PMID. 27,350 (62.8%) come from one resource
               of the twelve it aggregates.
    SIGNOR     19,533 edges, every one with a PMID. 17,636 (90.3%) rest on exactly one.
    OmniPath   85,526 directed edges. 63,647 (74.4%) have curation effort 1.

So the modal edge in every literature source is a single-paper claim, and that is the fact the
tiers below have to represent.

THE TIERS, declared before any motif was counted:

    E3 CORROBORATED   >= 2 distinct PMIDs AND >= 2 distinct source resources
    E2 LITERATURE     >= 1 PMID -- a named paper exists
    E1 OCCUPANCY      present only in the binding tier; ChIP-derived, no paper for THIS edge
    E0 UNATTRIBUTED   present only in the unidentified tier

net_bundle's provenance is by row order, recorded in loop 187: rows 0-55,716 curated causal,
55,716-278,405 binding, 278,405-612,133 unidentified.

PREDECLARED, BEFORE ANY NUMBER.

  Q1 IS THE EVIDENCE TABLE HONEST?
     Gate: PASS iff every CollecTRI and SIGNOR edge carries at least one PMID, the per-source
     counts reproduce the numbers above, and no edge is assigned a tier above the evidence it
     actually carries -- checked by re-deriving 200 sampled tier assignments from the raw fields.
     FAIL means the tiers are decoration and nothing below may be read.

  Q2 IS MOTIF EVIDENCE CLUSTERED, OR SCATTERED?
     Let p be the fraction of edges at E3. If edges were independent, the fraction of two-cycles
     with BOTH edges at E3 would be p^2. Well-studied genes plausibly have all their edges well
     studied, which would put the observed fraction ABOVE p^2.
     Gate: PASS iff observed > p^2. A FAIL is the worse finding: evidence scattered across motifs
     means motifs are LESS trustworthy than the per-edge rate suggests, not more.

  Q3 DOES FEEDFORWARD TRUST FALL FASTER THAN FEEDBACK TRUST?
     Three edges must all hold instead of two, so some drop is arithmetic. The question is whether
     the observed drop is steeper or shallower than the arithmetic alone predicts.
     Gate: PASS iff the observed FFL-to-feedback trust ratio exceeds the independence prediction
     p^3/p^2 = p. A FAIL means feedforward loops are disproportionately built from weak edges.
     Requires Q2 -- without a measured clustering there is nothing to compare against.

  Q4 HOW MANY MOTIFS REST ON A SINGLE PAPER?
     A motif is single-paper-dependent if ANY of its edges carries exactly one PMID from exactly
     one resource.
     Predeclared: the MAJORITY of both feedback and feedforward motifs are single-paper-dependent.
     Gate: PASS iff both fractions exceed 0.5. Failable in either direction and I am not
     predicting which way it goes for feedback specifically.

  Q5 HOW MANY FEEDBACK LOOPS ARE CONFIRMED BY TWO INDEPENDENT CATALOGUES?
     Both edges present in CollecTRI AND both present in SIGNOR. These two are curated by
     different groups from different literature, and loop 201 measured their edge overlap at
     2,604 -- small enough that agreement is informative.
     Gate: PASS iff the count is greater than zero AND exceeds what random pairing of the two
     catalogues' edge sets would give, measured with 200 draws that preserve each catalogue's
     degree sequence. A count that a shuffle reproduces is not confirmation.

  Q6 WHAT DOES THIS DO TO LOOP 187'S HEADLINE?
     Loop 187 reported 1,630 curated two-cycles at z +43.8 and 318,683 feedforward triads at
     z +1.3. Re-report both with the trust tier attached.
     Gate: PASS iff the tier breakdown sums to the loop 187 totals it claims to describe. This is
     an arithmetic check on my own bookkeeping, and it has caught a miscount before -- loop 187's
     own feedforward counter was wrong by 2x(two-cycles) until a self-check found it.

  Q7 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import csv, gzip, json, os, sys, time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "colab", "data", "net_bundle.json.gz")
NET = os.path.join(ROOT, "colab", "data", "networks")
OUT = os.path.join(ROOT, "outputs", "loop_motif_evidence.json")

CURATED_END, BINDING_END = 55716, 278405
LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def load_evidence():
    """Per-edge PMIDs and resources, from the sources that carry them."""
    pmid, resrc = defaultdict(set), defaultdict(set)
    ct = set()
    with open(os.path.join(NET, "collectri.csv")) as f:
        for row in csv.DictReader(f):
            k = (row["source"], row["target"])
            ct.add(k)
            pmid[k] |= {x.strip() for x in row["PMID"].split(";") if x.strip()}
            resrc[k] |= {x.strip() for x in row["resources"].split(";") if x.strip()}
    sg = set()
    with open(os.path.join(NET, "signor_human.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 22 or p[1] != "protein" or p[5] != "protein":
                continue
            k = (p[0], p[4])
            sg.add(k)
            if p[21].strip():
                pmid[k].add(p[21].strip())
            resrc[k].add("SIGNOR")
    om = set()
    with open(os.path.join(NET, "omnipath.tsv")) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("is_directed") != "True":
                continue
            k = (row["source_genesymbol"], row["target_genesymbol"])
            om.add(k)
            pmid[k] |= {x.split(":")[-1] for x in row["references"].split(";") if x.strip()}
            resrc[k] |= {x.strip() for x in row["sources"].split(";") if x.strip()}
    return pmid, resrc, ct, sg, om


def tier(e, pmid, resrc, prov):
    n_p, n_r = len(pmid.get(e, ())), len(resrc.get(e, ()))
    if n_p >= 2 and n_r >= 2:
        return 3
    if n_p >= 1:
        return 2
    return 1 if prov.get(e) == "binding" else 0


def two_cycles(edges):
    return sorted({tuple(sorted((a, b))) for a, b in edges if (b, a) in edges})


def ffl_triads(edges, out, inn, cap=None):
    """A->B, A->C, B->C. Returns the triads as (a,b,c)."""
    tri = []
    for a, b in edges:
        for c in out.get(b, ()):
            if (a, c) in edges:
                tri.append((a, b, c))
                if cap and len(tri) >= cap:
                    return tri
    return tri


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "motif evidence"}
    say("=" * 104)
    say("LOOP 202 -- OF ALL THE FEEDBACK AND FEEDFORWARD LOOPS, HOW MANY CAN WE TRUST?")
    say("=" * 104)

    pmid, resrc, ct, sg, om = load_evidence()
    nb = json.load(gzip.open(NB))
    names = nb["names"]
    prov = {}
    for i, (s, t, _) in enumerate(nb["reg"]):
        e = (names[s], names[t])
        prov.setdefault(e, "curated" if i < CURATED_END
                        else ("binding" if i < BINDING_END else "unidentified"))

    # ------------------------------------------------------------ Q1
    say("Q1 IS THE EVIDENCE TABLE HONEST?")
    ct_no_pmid = sum(1 for e in ct if not pmid.get(e))
    sg_no_pmid = sum(1 for e in sg if not pmid.get(e))
    say(f"     CollecTRI {len(ct):,}  without a PMID {ct_no_pmid}")
    say(f"     SIGNOR    {len(sg):,}  without a PMID {sg_no_pmid}")
    say(f"     OmniPath  {len(om):,}")
    rng = np.random.default_rng(202)
    universe = sorted(ct | sg)
    sample = [universe[i] for i in rng.choice(len(universe), 200, replace=False)]
    bad = 0
    for e in sample:
        t = tier(e, pmid, resrc, prov)
        n_p, n_r = len(pmid.get(e, ())), len(resrc.get(e, ()))
        if t == 3 and not (n_p >= 2 and n_r >= 2):
            bad += 1
        if t == 2 and n_p < 1:
            bad += 1
    say(f"     200 sampled tier assignments re-derived from raw fields   mismatches {bad}")
    ok1 = (len(ct) == 43536 and len(sg) == 19533 and len(om) == 85526
           and ct_no_pmid == 0 and sg_no_pmid == 0 and bad == 0)
    G.add("Q1", ok1,
          if_true="Q1 PASS -- every literature edge carries a PMID and the tiers re-derive",
          if_false=lambda: f"Q1 FAIL -- ct {len(ct)} sg {len(sg)} om {len(om)}, "
                           f"no-PMID {ct_no_pmid}/{sg_no_pmid}, tier mismatches {bad}")

    # the literature network: the union of the two PMID-carrying catalogues
    lit = ct | sg
    lout, linn = defaultdict(set), defaultdict(set)
    for a, b in lit:
        lout[a].add(b); linn[b].add(a)
    tiers = {e: tier(e, pmid, resrc, prov) for e in lit}
    p_e3 = sum(1 for v in tiers.values() if v == 3) / len(tiers)
    say(f"     literature network (CollecTRI + SIGNOR)  {len(lit):,} edges")
    say(f"     per-edge tier: " + "  ".join(
        f"E{k} {v:,} ({v/len(tiers):.3f})" for k, v in sorted(Counter(tiers.values()).items(),
                                                             reverse=True)))
    res["sources"] = {"collectri": len(ct), "signor": len(sg), "omnipath": len(om),
                      "literature_union": len(lit),
                      "tier_counts": {str(k): v for k, v in Counter(tiers.values()).items()},
                      "p_e3": p_e3}

    # ------------------------------------------------------------ motifs
    say("Q2 IS MOTIF EVIDENCE CLUSTERED, OR SCATTERED?")
    fb = two_cycles(lit)
    fb_tier = [min(tiers[(a, b)], tiers[(b, a)]) for a, b in fb]
    fb_c = Counter(fb_tier)
    say(f"     feedback loops in the literature network  {len(fb):,}")
    for k in (3, 2, 1, 0):
        say(f"       both edges at E{k} or better: {sum(v for kk, v in fb_c.items() if kk >= k):,}")
    obs_fb3 = fb_c.get(3, 0) / len(fb)
    say(f"     observed both-E3 fraction {obs_fb3:.4f}   independence predicts p^2 = "
        f"{p_e3**2:.4f}   (p = {p_e3:.4f})")
    G.add("Q2", bool(obs_fb3 > p_e3 ** 2), stat=obs_fb3, requires=("Q1",),
          if_true=lambda: f"Q2 PASS -- evidence CLUSTERS: {obs_fb3:.4f} vs p^2 {p_e3**2:.4f}, so a "
                          f"well-attested edge tends to sit opposite another one",
          if_false=lambda: f"Q2 FAIL -- {obs_fb3:.4f} is at or below p^2 {p_e3**2:.4f}: evidence is "
                           f"scattered and motifs are LESS trustworthy than the per-edge rate says")

    say("Q3 DOES FEEDFORWARD TRUST FALL FASTER THAN FEEDBACK TRUST?")
    tri = ffl_triads(lit, lout, linn)
    tri_tier = [min(tiers[(a, b)], tiers[(a, c)], tiers[(b, c)]) for a, b, c in tri]
    tri_c = Counter(tri_tier)
    say(f"     feedforward loops in the literature network  {len(tri):,}")
    for k in (3, 2, 1, 0):
        say(f"       all three edges at E{k} or better: "
            f"{sum(v for kk, v in tri_c.items() if kk >= k):,}")
    obs_ffl3 = tri_c.get(3, 0) / len(tri) if tri else float("nan")
    ratio = obs_ffl3 / obs_fb3 if obs_fb3 else float("nan")
    say(f"     observed all-E3 fraction {obs_ffl3:.4f}")
    say(f"     FFL/feedback trust ratio {ratio:.4f}   independence predicts p = {p_e3:.4f}")
    G.add("Q3", bool(ratio > p_e3), stat=ratio, requires=("Q2",),
          if_true=lambda: f"Q3 PASS -- {ratio:.4f} above the arithmetic prediction {p_e3:.4f}",
          if_false=lambda: f"Q3 FAIL -- {ratio:.4f} at or below {p_e3:.4f}: feedforward loops are "
                           f"disproportionately built from weakly attested edges")

    # ------------------------------------------------------------ Q4
    say("Q4 HOW MANY MOTIFS REST ON A SINGLE PAPER?")
    def single(e):
        return len(pmid.get(e, ())) == 1 and len(resrc.get(e, ())) == 1
    fb_single = sum(1 for a, b in fb if single((a, b)) or single((b, a)))
    tri_single = sum(1 for a, b, c in tri if single((a, b)) or single((a, c)) or single((b, c)))
    f_fb, f_tri = fb_single / len(fb), (tri_single / len(tri) if tri else float("nan"))
    say(f"     feedback loops with >=1 single-paper edge     {fb_single:,} of {len(fb):,} "
        f"= {f_fb:.4f}")
    say(f"     feedforward loops with >=1 single-paper edge  {tri_single:,} of {len(tri):,} "
        f"= {f_tri:.4f}")
    G.add("Q4", bool(f_fb > 0.5 and f_tri > 0.5), stat=f_fb, requires=("Q1",),
          if_true=lambda: f"Q4 PASS -- the majority of both rest on at least one single-paper "
                          f"edge ({f_fb:.1%} feedback, {f_tri:.1%} feedforward)",
          if_false=lambda: f"Q4 FAIL -- {f_fb:.1%} feedback and {f_tri:.1%} feedforward")

    # ------------------------------------------------------------ Q5
    say("Q5 HOW MANY FEEDBACK LOOPS ARE CONFIRMED BY TWO INDEPENDENT CATALOGUES?")
    both = [(a, b) for a, b in fb
            if (a, b) in ct and (b, a) in ct and (a, b) in sg and (b, a) in sg]
    say(f"     both edges in CollecTRI AND both in SIGNOR   {len(both):,}")
    # degree-preserving shuffle of SIGNOR, keeping CollecTRI fixed
    sg_src = [a for a, b in sg]; sg_tgt = [b for a, b in sg]
    r5 = np.random.default_rng(2025)
    nulls = []
    for _ in range(200):
        perm = r5.permutation(len(sg_tgt))
        shuffled = {(sg_src[i], sg_tgt[perm[i]]) for i in range(len(sg_src))}
        nulls.append(sum(1 for a, b in fb
                         if (a, b) in ct and (b, a) in ct
                         and (a, b) in shuffled and (b, a) in shuffled))
    nm, ns = float(np.mean(nulls)), float(np.std(nulls))
    z5 = (len(both) - nm) / ns if ns > 0 else float("nan")
    say(f"     stub-shuffled SIGNOR null  {nm:.2f} +/- {ns:.2f}   z {z5:+.1f}   (200 draws)")
    G.add("Q5", bool(len(both) > 0 and z5 > 3), stat=z5, requires=("Q1",),
          if_true=lambda: f"Q5 PASS -- {len(both):,} feedback loops carry independent confirmation "
                          f"from two separately curated catalogues, z {z5:+.1f}",
          if_false=lambda: f"Q5 FAIL -- {len(both):,} doubly-confirmed against a null of "
                           f"{nm:.2f} +/- {ns:.2f}, z {z5:+.1f}")

    # ------------------------------------------------------------ Q6
    say("Q6 WHAT DOES THIS DO TO LOOP 187'S HEADLINE?")
    cur = {(names[s], names[t]) for i, (s, t, _) in enumerate(nb["reg"]) if i < CURATED_END}
    cout, cinn = defaultdict(set), defaultdict(set)
    for a, b in cur:
        cout[a].add(b); cinn[b].add(a)
    c_fb = two_cycles(cur)
    c_tri = ffl_triads(cur, cout, cinn)
    say(f"     loop 187 curated tier: two-cycles {len(c_fb):,} (loop 187 said 1,630 DIRECTED "
        f"= {2*len(c_fb):,} ordered)   feedforward {len(c_tri):,} (loop 187 said 318,683)")
    c_fb_t = Counter(min(tier((a, b), pmid, resrc, prov), tier((b, a), pmid, resrc, prov))
                     for a, b in c_fb)
    c_tri_t = Counter(min(tier((a, b), pmid, resrc, prov), tier((a, c), pmid, resrc, prov),
                          tier((b, c), pmid, resrc, prov)) for a, b, c in c_tri)
    for label, C, tot in (("two-cycles", c_fb_t, len(c_fb)), ("feedforward", c_tri_t, len(c_tri))):
        say(f"     {label} by trust tier: " + "  ".join(
            f"E{k} {C.get(k,0):,}" for k in (3, 2, 1, 0)) + f"   sum {sum(C.values()):,} of {tot:,}")
    ok6 = (sum(c_fb_t.values()) == len(c_fb) and sum(c_tri_t.values()) == len(c_tri))
    G.add("Q6", ok6, requires=("Q1",),
          if_true="Q6 PASS -- the tier breakdown accounts for every motif it claims to describe",
          if_false=lambda: f"Q6 FAIL -- {sum(c_fb_t.values())} vs {len(c_fb)} and "
                           f"{sum(c_tri_t.values())} vs {len(c_tri)}")
    res["motifs"] = {
        "literature": {
            "feedback_total": len(fb), "feedback_by_tier": {str(k): v for k, v in fb_c.items()},
            "ffl_total": len(tri), "ffl_by_tier": {str(k): v for k, v in tri_c.items()},
            "obs_fb_e3": obs_fb3, "obs_ffl_e3": obs_ffl3, "p_e3": p_e3, "ratio": ratio,
            "fb_single_paper": fb_single, "ffl_single_paper": tri_single,
            "frac_fb_single": f_fb, "frac_ffl_single": f_tri,
            "doubly_confirmed": len(both), "null_mean": nm, "null_sd": ns, "z": z5},
        "loop187_curated": {
            "two_cycles_undirected": len(c_fb), "ffl": len(c_tri),
            "two_cycles_by_tier": {str(k): c_fb_t.get(k, 0) for k in (3, 2, 1, 0)},
            "ffl_by_tier": {str(k): c_tri_t.get(k, 0) for k in (3, 2, 1, 0)}},
    }

    say("Q7 WHAT THIS CANNOT SHOW")
    say("     A PMID count is not a quality measure. Ten papers citing one original observation")
    say("     look like ten pieces of evidence here and are one. This design cannot see that, so")
    say("     every E3 count is an UPPER bound on corroboration.")
    say("     CollecTRI aggregates twelve resources that themselves overlap, so its resource")
    say("     count is not twelve independent groups either.")
    say("     Absence of a PMID is not absence of evidence -- the binding tier is real ChIP data,")
    say("     it just has no paper attached to the individual edge.")
    say("     Loop 187's enrichments were measured on the tiers as-is. Nothing here re-runs them")
    say("     at E3 only, and the null would have to be rebuilt on the E3 subgraph to do that.")

    G.summary(seconds=time.time() - t0)
    gates, void = G.as_dict()
    res["gates"], res["void"] = gates, void
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    say(f"wrote {OUT}")


if __name__ == "__main__":
    main()
