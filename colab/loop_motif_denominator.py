"""Loop 203. Out of how many feedback and feedforward loops a cell actually has?

LOOP 202 ANSWERED THE NUMERATOR AND CALLED IT AN ANSWER. It counted 2,280 feedback loops and
226,062 feedforward loops in the literature network, and reported how many were corroborated. Every
one of those fractions has the OBSERVED motif count as its denominator. The question underneath is
what fraction of the cell's actual motifs those are, and loop 202 did not ask it.

WHY THIS IS HARDER THAN LOOP 200'S REACTION DENOMINATOR, and the reason is arithmetic rather than
biology. Reactions are counted one at a time: miss 40% of the reactions and you miss 40% of the
reactions. MOTIFS ARE NOT LINEAR IN EDGES. A feedback loop needs two edges to both be present and a
feedforward loop needs three, so an incomplete edge list loses motifs FASTER than it loses edges,
and the three-edge motif is lost fastest. Missing half the edges does not cost half the feedforward
loops -- if the count went as the cube it would cost seven eighths of them.

So the answer needs two things this project has never measured together:

    (1) HOW STEEPLY motif count scales with edge count -- measured by subsampling, not assumed
    (2) HOW MANY EDGES THERE ACTUALLY ARE -- estimated by capture-recapture on independently
        curated catalogues

CAPTURE-RECAPTURE, AND WHY IT IS THE RIGHT INSTRUMENT HERE. Two curation efforts that sample the
same literature independently give an estimate of the total population from their overlap:
N = |A| x |B| / |A and B|. It is the standard ecological estimator, and curated databases are a
legitimate application: each is a "capture" of the underlying set of real regulatory interactions.
Its one assumption is INDEPENDENCE, and that assumption is testable here because there are three
catalogues, so there are three pairwise estimates that must agree if independence holds. R3 tests
exactly that and is allowed to fail.

PREDECLARED, BEFORE ANY NUMBER.

  R1 IS THE SUBSAMPLING INSTRUMENT SOUND?
     Gate: PASS iff counting motifs on the full edge set through the subsampling code reproduces
     loop 202's counts exactly -- 2,280 feedback and 226,062 feedforward on the literature union.
     FAIL means the scaling below is measured with a different counter than the numerator was, and
     the ratio would be meaningless.

  R2 HOW STEEPLY DOES MOTIF COUNT SCALE WITH EDGE COUNT?
     Subsample the literature network at 20%..100% of its edges, count both motifs, fit
     log(count) = a + k*log(edges).
     Gate: PASS iff both fits reach R^2 >= 0.98 AND the feedforward exponent exceeds the feedback
     exponent. The second half is the real test: three edges must be present instead of two, so if
     the exponents come out equal the counter or the subsampling is wrong.

  R3 DO THE CATALOGUES BEHAVE AS INDEPENDENT SAMPLES?
     Three pairwise Lincoln-Petersen estimates from CollecTRI, SIGNOR and OmniPath.
     Gate: PASS iff the largest and smallest agree within a factor of 2. Independence is what makes
     the estimator valid, and three mutually inconsistent estimates are proof it does not hold.
     A FAIL means no single edge total can be quoted, and R4 must carry a range instead.

  R4 WHAT IS THE MOTIF DENOMINATOR?
     Apply R2's exponents to R3's edge estimates.
     Gate: PASS iff the resulting motif estimate spans less than one order of magnitude. If R3
     passed this is a narrow range and a usable number; if R3 failed it will not be, and saying so
     with the width attached is the finding.
     Requires R2 -- without a measured exponent there is nothing to extrapolate with.

  R5 HOW MUCH OF THE POSSIBLE SPACE IS EVEN OCCUPIED?
     ~1,600 human transcription factors x 16,492 genes bounds the directed edge space from above.
     Gate: PASS iff every R3 estimate falls below that bound. An estimate above it would prove the
     estimator broken, and this is the cheapest available check that it is not.

  R6 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import csv, gzip, json, os, sys, time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(ROOT, "colab", "data", "networks")
OUT = os.path.join(ROOT, "outputs", "loop_motif_denominator.json")

FRACTIONS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0)
DRAWS = 5
N_TF = 1600
N_GENE = 16492

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def load():
    ct, sg, om = set(), set(), set()
    with open(os.path.join(NET, "collectri.csv")) as f:
        for r in csv.DictReader(f):
            ct.add((r["source"], r["target"]))
    with open(os.path.join(NET, "signor_human.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 10 and p[1] == "protein" and p[5] == "protein":
                sg.add((p[0], p[4]))
    with open(os.path.join(NET, "omnipath.tsv")) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("is_directed") == "True":
                om.add((r["source_genesymbol"], r["target_genesymbol"]))
    return ct, sg, om


def count_motifs(edge_list):
    """Undirected two-cycles and directed feedforward triads, on one edge list."""
    E = set(edge_list)
    out = defaultdict(set)
    for a, b in E:
        out[a].add(b)
    fb = len({tuple(sorted((a, b))) for a, b in E if (b, a) in E})
    ffl = 0
    for a, b in E:
        oa = out.get(a)
        if not oa:
            continue
        for c in out.get(b, ()):
            if c in oa:
                ffl += 1
    return fb, ffl


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "motif denominator"}
    say("=" * 104)
    say("LOOP 203 -- OUT OF HOW MANY FEEDBACK AND FEEDFORWARD LOOPS DOES A CELL ACTUALLY HAVE?")
    say("=" * 104)

    ct, sg, om = load()
    lit = sorted(ct | sg)
    say(f"     literature union (CollecTRI + SIGNOR)  {len(lit):,} edges")

    # ------------------------------------------------------------ R1
    say("R1 IS THE SUBSAMPLING INSTRUMENT SOUND?")
    fb_full, ffl_full = count_motifs(lit)
    say(f"     full-set recount   feedback {fb_full:,}   feedforward {ffl_full:,}")
    say(f"     loop 202 reported  feedback 2,280        feedforward 226,062")
    ok1 = (fb_full == 2280 and ffl_full == 226062)
    G.add("R1", ok1,
          if_true="R1 PASS -- the same counter reproduces loop 202 exactly",
          if_false=lambda: f"R1 FAIL -- {fb_full:,}/{ffl_full:,} against 2,280/226,062")

    # ------------------------------------------------------------ R2
    say("R2 HOW STEEPLY DOES MOTIF COUNT SCALE WITH EDGE COUNT?")
    rng = np.random.default_rng(203)
    xs, fbs, ffls = [], [], []
    say("       frac      edges     feedback   feedforward")
    for f in FRACTIONS:
        n = int(round(f * len(lit)))
        a_fb, a_ffl = [], []
        for _ in range(1 if f == 1.0 else DRAWS):
            idx = rng.choice(len(lit), size=n, replace=False)
            sub = [lit[i] for i in idx]
            x, y = count_motifs(sub)
            a_fb.append(x); a_ffl.append(y)
        mfb, mffl = float(np.mean(a_fb)), float(np.mean(a_ffl))
        say(f"       {f:.2f}   {n:>8,}   {mfb:>10,.0f}   {mffl:>11,.0f}")
        xs.append(n); fbs.append(mfb); ffls.append(mffl)

    def powfit(x, y):
        lx, ly = np.log(np.array(x, float)), np.log(np.array(y, float) + 1e-9)
        k, a = np.polyfit(lx, ly, 1)
        pred = a + k * lx
        ss_res = float(np.sum((ly - pred) ** 2))
        ss_tot = float(np.sum((ly - ly.mean()) ** 2))
        return float(k), float(a), 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    k_fb, a_fb_c, r2_fb = powfit(xs, fbs)
    k_ffl, a_ffl_c, r2_ffl = powfit(xs, ffls)
    say(f"     feedback     count ~ E^{k_fb:.3f}   R^2 {r2_fb:.4f}")
    say(f"     feedforward  count ~ E^{k_ffl:.3f}   R^2 {r2_ffl:.4f}")
    ok2 = bool(r2_fb >= 0.98 and r2_ffl >= 0.98 and k_ffl > k_fb)
    G.add("R2", ok2, stat=k_ffl, requires=("R1",),
          if_true=lambda: f"R2 PASS -- feedback scales as E^{k_fb:.2f}, feedforward as "
                          f"E^{k_ffl:.2f}; the three-edge motif is lost faster, as it must be",
          if_false=lambda: f"R2 FAIL -- exponents {k_fb:.3f}/{k_ffl:.3f}, R^2 {r2_fb:.4f}/"
                           f"{r2_ffl:.4f}")
    res["scaling"] = {"edges": xs, "feedback": fbs, "feedforward": ffls,
                      "k_feedback": k_fb, "k_feedforward": k_ffl,
                      "r2_feedback": r2_fb, "r2_feedforward": r2_ffl}

    # ------------------------------------------------------------ R3
    say("R3 DO THE CATALOGUES BEHAVE AS INDEPENDENT SAMPLES?")
    ests = {}
    for na, A, nb, B in (("CollecTRI", ct, "SIGNOR", sg), ("CollecTRI", ct, "OmniPath", om),
                         ("SIGNOR", sg, "OmniPath", om)):
        m = len(A & B)
        N = len(A) * len(B) / m if m else float("inf")
        ests[f"{na}x{nb}"] = N
        say(f"     {na:<10} x {nb:<10} |A| {len(A):>6,}  |B| {len(B):>6,}  overlap {m:>6,}"
            f"   ->  N = {N:>12,.0f}")
    lo, hi = min(ests.values()), max(ests.values())
    spread = hi / lo
    say(f"     spread between the three estimates  {spread:.1f}x")
    G.add("R3", bool(spread <= 2.0), stat=spread,
          if_true=lambda: f"R3 PASS -- the three agree within {spread:.2f}x, so independence holds "
                          f"and a single edge total can be quoted",
          if_false=lambda: f"R3 FAIL -- the three estimates span {spread:.1f}x "
                           f"({lo:,.0f} to {hi:,.0f}). The catalogues are NOT independent samples "
                           f"-- SIGNOR is largely inside OmniPath -- so capture-recapture cannot "
                           f"pin the edge total and no single denominator can be quoted")
    res["capture_recapture"] = {"estimates": ests, "low": lo, "high": hi, "spread": spread}

    # ------------------------------------------------------------ R4
    say("R4 WHAT IS THE MOTIF DENOMINATOR?")
    say(f"     observed on {len(lit):,} edges:  feedback {fb_full:,}   feedforward {ffl_full:,}")
    proj = {}
    for name, N in sorted(ests.items(), key=lambda kv: kv[1]):
        r = N / len(lit)
        pfb, pffl = fb_full * r ** k_fb, ffl_full * r ** k_ffl
        proj[name] = {"N_edges": N, "ratio": r, "feedback": pfb, "feedforward": pffl,
                      "frac_fb_seen": fb_full / pfb, "frac_ffl_seen": ffl_full / pffl}
        say(f"     if there are {N:>11,.0f} edges ({r:>5.1f}x what we have):")
        say(f"         feedback    ~ {pfb:>14,.0f}   we have seen {fb_full/pfb:>7.2%}")
        say(f"         feedforward ~ {pffl:>14,.0f}   we have seen {ffl_full/pffl:>7.2%}")
    fb_lo = min(v["feedback"] for v in proj.values())
    fb_hi = max(v["feedback"] for v in proj.values())
    ffl_lo = min(v["feedforward"] for v in proj.values())
    ffl_hi = max(v["feedforward"] for v in proj.values())
    span = max(fb_hi / fb_lo, ffl_hi / ffl_lo)
    say(f"     feedback denominator    {fb_lo:,.0f} to {fb_hi:,.0f}")
    say(f"     feedforward denominator {ffl_lo:,.0f} to {ffl_hi:,.0f}")
    say(f"     widest span {span:,.0f}x")
    G.add("R4", bool(span < 10), stat=span, requires=("R2",),
          if_true=lambda: f"R4 PASS -- the denominator is pinned to within {span:.1f}x",
          if_false=lambda: f"R4 FAIL -- the motif denominator spans {span:,.0f}x. The edge total is "
                           f"uncertain by {spread:.1f}x and the exponent AMPLIFIES that, so the "
                           f"question 'out of how many' has no answer at this resolution")
    res["projection"] = {"observed_fb": fb_full, "observed_ffl": ffl_full,
                         "by_estimate": proj, "fb_range": [fb_lo, fb_hi],
                         "ffl_range": [ffl_lo, ffl_hi], "span": span}

    # ------------------------------------------------------------ R5
    say("R5 HOW MUCH OF THE POSSIBLE SPACE IS EVEN OCCUPIED?")
    bound = N_TF * N_GENE
    say(f"     upper bound on directed edges: {N_TF:,} TFs x {N_GENE:,} genes = {bound:,}")
    for k, N in sorted(ests.items(), key=lambda kv: kv[1]):
        say(f"     {k:<22} {N:>12,.0f}   = {N/bound:.4%} of the possible space")
    G.add("R5", bool(all(N < bound for N in ests.values())), stat=hi,
          if_true=lambda: f"R5 PASS -- every estimate sits below the {bound:,} bound, the highest "
                          f"at {hi/bound:.2%} of the possible space",
          if_false=lambda: f"R5 FAIL -- an estimate exceeds the {bound:,} bound, which would prove "
                           f"the estimator broken")
    res["bound"] = {"n_tf": N_TF, "n_gene": N_GENE, "possible_edges": bound,
                    "occupancy": {k: N / bound for k, N in ests.items()}}

    # ------------------------------------------------------------ R6
    say("R6 WHAT THIS CANNOT SHOW")
    say("     Capture-recapture assumes every edge is equally catchable. Curation is biased to")
    say("     well-studied genes, so rarely-studied edges are under-caught in EVERY catalogue at")
    say("     once. That inflates the overlap and DEFLATES N, so all three estimates are lower")
    say("     bounds and the true edge count is above even the highest of them.")
    say("     The extrapolation assumes missing edges attach like observed ones. They do not:")
    say("     curated edges concentrate on studied hubs, and loop 202 measured that evidence")
    say("     CLUSTERS (both-E3 at 0.2215 against p^2 of 0.0686). Clustered missing edges would")
    say("     make fewer new motifs than the exponent predicts, so the projections above are")
    say("     upper bounds in that direction while being lower bounds in the direction above.")
    say("     Those two biases point opposite ways and this design cannot net them out.")
    say("     A cell does not run all its edges at once. These counts are of a static wiring")
    say("     diagram pooled over cell types; loop 187's curated and binding tiers already")
    say("     disagreed on feedforward by 22 sigma, which is what pooling looks like.")

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
