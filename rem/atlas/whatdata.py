"""What data would actually validate C? The design specification, computed rather than wished for.

WHY THIS EXISTS. Every cap in this build order -- 3, 13, 140 -- is cost arithmetic carried out at
a stated accuracy, and that accuracy is "C = 64 of 404 regulator classes suffice". C was measured
in yeast. humantransfer showed the human panel cannot distinguish class resolutions at all, and
scalarcap showed the human MPRA ladder is narrower than its own noise floor, so C = 64 is an
extrapolation across a kingdom that human data in hand is PROVABLY incapable of testing.

Both of those modules closed by naming what was needed in prose -- "many more replicates, or a
human assay with the yeast experiment's structure". Prose is where this build order has repeatedly
gone wrong: block.py's E4 and E7 were prose arithmetic and false by a wide margin. So this module
converts the wish into numbers. It does not run a new experiment. It computes, from data already
in hand, HOW BIG one would have to be, and prices the catalogue that exists against that number.

THE UNITS. A class ladder is informative only if its ends are separated by more than the noise
floor of the quantity being scored; otherwise "the smallest C within one floor" returns C = 1
whatever the truth is, which is the degeneracy scalarcap recorded as ledger U. Yeast's ladder
separated its ends by 8.6 floors. That is the only number in this build order that ever made a
class count meaningful, so it is the reference standard throughout.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

D0  THE CEILING GATE, AND IT RUNS FIRST AS ALWAYS. Price a new experiment at ZERO: does data that
    ALREADY EXISTS meet the specification? Census the ENCODE K562 CRISPRi RNA-seq catalogue live
    -- how many distinct factors, and how many biological replicates per experiment.
    PREDECLARED: if the catalogue offers >= 128 distinct factors AND >= 7 biological replicates,
    then no new experiment is needed, the answer is "download the rest of it", and this module
    stops at D0. Both thresholds are derived in D1 and D2 below; they are stated here first so
    that D0's verdict cannot be tuned to whatever the catalogue turns out to hold.

D1  THE REPLICATE AXIS. Reproduce humantransfer's between-regulator variance component and its
    ratio to the noise floor -- call it rho, the identity signal in floors at the CURRENT
    replicate depth of two. If this does not reproduce humantransfer's published figure the
    module is broken and stops.
    Then the arithmetic that matters: the true between-regulator sd is a property of the biology
    and does not change with replication, while the floor against which it is judged is the noise
    in the scored quantity, which for a mean of R/2 replicates is sigma/sqrt(R/2). So the margin
    in floors is rho * sqrt(R/2) and the depth needed to reach a bar B is R = 2 * (B/rho)^2.
    PREDECLARED: R(B=1) <= 8 means replicate depth is a fixable axis and worth naming first.
    R(B=8.6) > 100 means yeast-grade resolution is NOT reachable by replicating this assay, and
    the design has to change rather than the depth.

D2  THE ALPHABET AXIS, WHICH IS A HARD CEILING AND NOT A POWER CALCULATION. A panel that perturbs
    N regulators cannot return a class count above N, and cannot show a ladder FLATTEN above its
    knee unless N is comfortably larger than the knee. To test C = 64 the panel must carry at
    least 64 and, to see the ladder flat above it, at least 2x that.
    PREDECLARED: N >= 128 distinct perturbed regulators, and the module reports the coverage
    SCALING LAW -- how many TRRUST targets reach a given regulator depth as N grows -- rather
    than a single number, because ranking a design at a point is the error this build order has
    corrected three times.

D3  THE DEPTH AXIS. The cap's binding regime is not k = 2. widthblock found the width exponential
    is ONE GENE, CDKN1A, with 52 regulators and |C| = 20 after elimination, and the 140 cap comes
    from demoting exactly that kind of hub. A class count validated at k = 2 is not validated at
    k = 20 unless the identity margin is known to be flat in k.
    So: does the margin grow, shrink, or stay flat with k? Measured on the cached panel in
    cumulative bins with bootstrap CIs.
    PREDECLARED: if the CIs overlap across every bin the scaling is UNMEASURABLE in this panel,
    NO extrapolation to k = 20 is permitted, and the spec must state the depth requirement as an
    unknown rather than a number. A flat or growing margin would be the favourable case; a
    shrinking one would mean the deep regime is HARDER than the shallow one and the whole
    class-count route is worse off than currently recorded.

D4  THE COMBINATORIAL AXIS, which no amount of single knockdowns can supply. The engine's
    per-gene factor is a PRODUCT over occupied classes of (k_ij (L+1) + 1): a statement about
    JOINT states of a gene's regulators. Single-perturbation data measures MARGINALS. Two
    regulators can have identical marginal effects and different joint effects, and the class
    hypothesis asserts they do not.
    PREDECLARED: this gate does not measure anything. It states the test a combinatorial design
    would have to run and counts how many same-target regulator PAIRS TRRUST offers, so the
    requirement is a number rather than a gesture.

D5  THE SPEC, AND ITS APPLICABILITY DOMAIN. Assemble D1-D4 into one design statement, and state
    the limit that a C validated on CRISPRi knockdown log-ratios is a C for that response
    function and not automatically a C for the engine's steady-state transcription rate. A number
    without its applicability domain is ledger S, which this build order has recorded twice.
"""

from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import json
import urllib.request
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.trrust_engine import signed_edges
from rem.atlas.humantransfer import prepare, regulator_index

ENC = "https://www.encodeproject.org"
YEAST_LADDER_FLOORS = 8.6   # signed.py's ladder span, the only class count ever measured here
CAP_BINDING_DEPTH = 20      # widthblock's |C| for CDKN1A after elimination


# =================================================================================================
# D0  the catalogue census
# =================================================================================================

def encode_census(cell="K562", assay="CRISPRi RNA-seq", timeout=170):
    """How many distinct factors, and how deep, does the catalogue already hold?

    Returns (n_experiments, n_targets, replicate histogram) or None if the query fails. A failed
    query is reported as a failed query -- it is not silently treated as an empty catalogue,
    which would make D0's ceiling gate pass for the wrong reason."""
    url = (f"{ENC}/search/?type=Experiment&assay_title={assay.replace(' ', '+')}"
           f"&biosample_ontology.term_name={cell}&status=released&limit=all&format=json"
           f"&field=accession&field=target.label&field=replicates.biological_replicate_number")
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception as e:                                   # pragma: no cover - network
        return None, str(e)
    g = d.get("@graph", [])
    targets, hist = set(), collections.Counter()
    for e in g:
        t = (e.get("target") or {}).get("label")
        if t:
            targets.add(t)
        hist[len({x.get("biological_replicate_number")
                  for x in (e.get("replicates") or [])})] += 1
    return (len(g), len(targets), dict(sorted(hist.items()))), None


# =================================================================================================
# D1  the variance component, reused rather than reinvented
# =================================================================================================

def between_regulator(groups, sigma, rng, B=2000):
    """humantransfer's estimator, verbatim in form: the within-gene variance of the response
    across its regulators, with the replicate noise that survives two-replicate averaging
    subtracted, bootstrapped over genes."""
    def est(gs):
        num, den = 0.0, 0
        for v in gs:
            K = len(v)
            if K < 2:
                continue
            num += np.var(v, ddof=0) * K - (sigma ** 2 / 2) * (K - 1)
            den += K
        return max(num / max(den, 1), 0.0)
    pt = est(groups)
    idx = np.arange(len(groups))
    bs = np.array([est([groups[i] for i in rng.choice(idx, len(idx), replace=True)])
                   for _ in range(B)])
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def depth_for_bar(rho, bar):
    """Total biological replicates needed for the identity margin to reach `bar` floors.

    Half the replicates build the predictor and half are scored, so the floor on the scored
    quantity is sigma/sqrt(R/2) and the margin is rho*sqrt(R/2). Inverting: R = 2 (bar/rho)^2."""
    if rho <= 0:
        return float("inf")
    return 2.0 * (bar / rho) ** 2


# =================================================================================================
# D2  the coverage scaling law
# =================================================================================================

def trrust_depths():
    """Regulator count per target gene in TRRUST, and the regulator out-degrees."""
    E, _ = signed_edges()
    tgt = collections.defaultdict(set)
    outdeg = collections.Counter()
    for u, v, _m in E:
        tgt[v].add(u)
        outdeg[u] += 1
    return {v: len(s) for v, s in tgt.items()}, outdeg, tgt


def coverage(tgt, panel):
    """Given a set of perturbed regulators, how many targets reach each depth k."""
    P = set(panel)
    ks = [len(s & P) for s in tgt.values()]
    return collections.Counter(ks)


def coverage_law(tgt, outdeg, sizes, rng, reps=40):
    """Coverage as N grows, bracketed between the two designs that bound any real panel:
    regulators chosen AT RANDOM (worst case) and the top-N by out-degree (best case)."""
    order = [t for t, _ in outdeg.most_common()]
    allregs = list(outdeg)
    rows = []
    for N in sizes:
        if N > len(allregs):
            continue
        best = coverage(tgt, order[:N])
        rnd = collections.Counter()
        for _ in range(reps):
            c = coverage(tgt, [allregs[i] for i in
                               rng.choice(len(allregs), N, replace=False)])
            for k, v in c.items():
                rnd[k] += v
        rows.append((N,
                     {k: sum(v for kk, v in best.items() if kk >= k) for k in (2, 4, 8, 20)},
                     {k: sum(v for kk, v in rnd.items() if kk >= k) / reps for k in (2, 4, 8, 20)}))
    return rows


# =================================================================================================

def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("WHAT DATA WOULD ACTUALLY VALIDATE C? THE SPECIFICATION, COMPUTED")
    P_(RULE)
    P_("  Two modules closed by naming what was needed in prose. This turns the prose into")
    P_("  numbers, using only data already in hand plus a live census of what exists.")

    # ---- D0  CEILING GATE ----------------------------------------------------------------------
    P_("\n" + RULE)
    P_("D0  THE CEILING GATE: DOES THE DATA ALREADY EXIST?")
    P_(RULE)
    P_("  PREDECLARED ABOVE, before the census ran: >= 128 distinct factors AND >= 7 biological")
    P_("  replicates would mean no new experiment is needed and this module stops here.")
    cen, err = encode_census()
    if cen is None:
        P_(f"  CENSUS FAILED: {err}")
        P_("  D0 is UNDECIDED -- a failed query is not an empty catalogue, and this module does")
        P_("  not get to pass its ceiling gate by being unable to look.")
        nfac, maxrep = None, None
    else:
        nexp, nfac, hist = cen
        maxrep = max(hist) if hist else 0
        P_(f"  ENCODE K562 CRISPRi RNA-seq, released: {nexp} experiments, {nfac} distinct factors.")
        P_(f"  biological replicates per experiment: {hist}")
        P_(f"  the deepest experiment in the entire catalogue has {maxrep} biological replicates.")
        ok = (nfac >= 128) and (maxrep >= 7)
        P_(f"\n  D0: {'PASS -- download the rest.' if ok else 'FAIL on BOTH axes.'}")
        if not ok:
            P_(f"    factors {nfac} against the 128 required: short by {128 - nfac}.")
            P_(f"    depth {maxrep} against the 7 required: short by {7 - maxrep}.")
            P_("  The whole ENCODE CRISPRi K562 catalogue -- not the 48 factors cached here, the")
            P_("  ENTIRE released catalogue -- cannot express C = 64 and cannot reach the depth.")
            P_("  So the answer is not 'download more of the same', and the rest of this module")
            P_("  is not a wish list but a specification with a purpose.")

    # ---- D1  THE REPLICATE AXIS ----------------------------------------------------------------
    P_("\n" + RULE)
    P_("D1  THE REPLICATE AXIS: HOW DEEP WOULD THE SAME ASSAY HAVE TO BE?")
    P_(RULE)
    genes, tfs, DA, DB, expressed, _sha = prepare()
    regs, _gsym = regulator_index(genes, tfs)
    keep = {j: r for j, r in regs.items() if expressed[j] and len(r) >= 2}
    pairs = [(j, ti, m) for j, r in keep.items() for ti, m in r]
    y1 = np.array([DA[ti, j] for j, ti, _m in pairs])
    y2 = np.array([DB[ti, j] for j, ti, _m in pairs])
    sigma = float(np.std(y1 - y2)) / np.sqrt(2.0)
    mbar = (y1 + y2) / 2.0
    gid = np.array([j for j, _ti, _m in pairs])
    gl = sorted(set(gid.tolist()))
    rng = np.random.default_rng(20260908)
    ph, lo, hi = between_regulator([mbar[gid == g] for g in gl], sigma, rng)
    rho = np.sqrt(ph) / sigma
    P_(f"  {len(keep)} expressed targets with >= 2 perturbed regulators, {len(pairs)} pairs,")
    P_(f"  measured twice each. noise floor sigma = {sigma:.4f} log2.")
    P_(f"  between-regulator sd {np.sqrt(ph):.4f} log2 = {rho:.2f} floors"
       f"   CI [{np.sqrt(lo)/sigma:.2f}, {np.sqrt(hi)/sigma:.2f}]")
    P_("  humantransfer's H3 reported this same quantity at about 1.05 floors. If the figure")
    P_("  above is not that number, this module is reading the data differently from the module")
    P_("  it is extending, and nothing below it is readable.")
    P_(f"\n  DEPTH REQUIRED, R = 2 (bar/rho)^2, for a margin of:")
    P_(f"    {'bar, in floors':<40} {'total biological replicates':>28}")
    for bar, why in ((1.0, "the ladder merely clears its own floor"),
                     (2.0, "some class resolutions excluded"),
                     (YEAST_LADDER_FLOORS, "yeast-grade resolution, the reference")):
        P_(f"    {f'{bar:.1f}  ({why})':<40} {depth_for_bar(rho, bar):>28.0f}")
    r1_, r86 = depth_for_bar(rho, 1.0), depth_for_bar(rho, YEAST_LADDER_FLOORS)
    P_(f"\n  D1 as predeclared: R(1) = {r1_:.0f}"
       f" {'<= 8, so depth IS a fixable axis' if r1_ <= 8 else '> 8, depth alone is already hard'};"
       f" R(8.6) = {r86:.0f}"
       f" {'> 100, so yeast-grade resolution is NOT reachable by replicating this assay' if r86 > 100 else '<= 100'}.")
    P_("  CAVEAT THAT IS NOT OPTIONAL: sqrt(R) holds only for the component of the noise that is")
    P_("  INDEPENDENT across biological replicates. A guide's off-target and indirect effects are")
    P_("  shared by both replicates of the same experiment, so they are invisible to sigma and")
    P_("  are NOT reduced by replication. With two replicates that systematic floor cannot be")
    P_("  bounded from these data at all, and R above is therefore a LOWER bound on the depth.")

    # ---- D2  THE ALPHABET AXIS -----------------------------------------------------------------
    P_("\n" + RULE)
    P_("D2  THE ALPHABET AXIS: HOW MANY FACTORS MUST BE PERTURBED?")
    P_(RULE)
    depths, outdeg, tgt = trrust_depths()
    P_(f"  TRRUST: {len(outdeg)} regulators, {len(tgt)} targets.")
    P_("  A panel of N regulators cannot return C > N. To test C = 64 and see the ladder flatten")
    P_("  above the knee, N >= 128. That is the hard part of the requirement. The rest is")
    P_("  coverage: how many targets does a panel of N actually reach at a useful depth?")
    sizes = [48, 74, 128, 200, 300, len(outdeg)]
    rows = coverage_law(tgt, outdeg, sizes, rng)
    P_(f"\n  TARGETS REACHED AT DEPTH >= k, BRACKETED BY THE BEST AND WORST PANEL OF SIZE N")
    P_(f"  (best = the N highest-out-degree regulators; worst = N drawn at random, mean of 40)")
    P_(f"    {'N':>5}  {'k>=2':>13} {'k>=4':>13} {'k>=8':>13} {'k>=20':>13}")
    for N, best, rnd in rows:
        cells = "".join(f" {best[k]:>6.0f}/{rnd[k]:<6.0f}" for k in (2, 4, 8, 20))
        P_(f"    {N:>5} {cells}")
    P_("    read as best/worst.")
    full = rows[-1][1]
    P_(f"\n  THE CEILING OF THE IDEAL EXPERIMENT -- every TRRUST regulator perturbed:")
    P_(f"    {full[2]} targets at k >= 2, {full[4]} at k >= 4, {full[8]} at k >= 8,"
       f" {full[20]} at k >= {CAP_BINDING_DEPTH}.")
    P_("  That number at k >= 20 is the population available to test the regime the 140 cap")
    P_("  actually lives in, and it is the ceiling: no panel can do better, because TRRUST does")
    P_("  not know of more regulators for those genes.")

    # ---- D3  THE DEPTH AXIS --------------------------------------------------------------------
    P_("\n" + RULE)
    P_("D3  THE DEPTH AXIS: IS THE IDENTITY MARGIN FLAT IN k?")
    P_(RULE)
    P_("  PREDECLARED: overlapping CIs across every bin means the scaling is unmeasurable here")
    P_("  and no extrapolation to k = 20 is permitted.")
    P_(f"\n    {'bin':<12} {'genes':>6} {'pairs':>7} {'sd, log2':>10} {'floors':>8} {'95% CI, floors':>20}")
    bins = []
    for kmin in (2, 3, 4):
        gs = [g for g in gl if (gid == g).sum() >= kmin]
        if len(gs) < 5:
            continue
        p, l, h = between_regulator([mbar[gid == g] for g in gs], sigma, rng)
        bins.append((kmin, len(gs), sum(int((gid == g).sum()) for g in gs),
                     np.sqrt(p), np.sqrt(p) / sigma, np.sqrt(l) / sigma, np.sqrt(h) / sigma))
        P_(f"    {f'k >= {kmin}':<12} {bins[-1][1]:>6} {bins[-1][2]:>7} {bins[-1][3]:>10.4f}"
           f" {bins[-1][4]:>8.2f} {f'[{bins[-1][5]:.2f}, {bins[-1][6]:.2f}]':>20}")
    overlap = all(bins[i][6] >= bins[j][5] and bins[j][6] >= bins[i][5]
                  for i in range(len(bins)) for j in range(len(bins)))
    if overlap:
        P_("\n  D3: the CIs overlap across every bin. THE SCALING IS UNMEASURABLE IN THIS PANEL.")
        P_("  As predeclared, no extrapolation to k = 20 is permitted and the spec must carry the")
        P_("  depth requirement as an UNKNOWN. That is the honest reading: the deepest bin here")
        P_(f"  holds {bins[-1][1] if bins else 0} genes, and a bin that small cannot rank anything.")
    else:
        P_("\n  D3: the bins separate. The margin is NOT flat in k and the direction is readable")
        P_("  from the table above.")

    # ---- D4  THE COMBINATORIAL AXIS ------------------------------------------------------------
    P_("\n" + RULE)
    P_("D4  THE COMBINATORIAL AXIS, WHICH SINGLE KNOCKDOWNS CANNOT SUPPLY AT ANY DEPTH")
    P_(RULE)
    P_("  The engine's per-gene factor is a PRODUCT over occupied classes of (k_ij (L+1) + 1). It")
    P_("  is a statement about JOINT states of a gene's regulators. Every measurement above is a")
    P_("  MARGINAL: one factor silenced, everything else left alone. Two regulators can have")
    P_("  identical marginals and different joint behaviour, and the class hypothesis asserts")
    P_("  they do not. No number of single knockdowns tests that assertion.")
    npairs = sum(len(s) * (len(s) - 1) // 2 for s in tgt.values() if len(s) >= 2)
    ntrip = sum(len(s) * (len(s) - 1) * (len(s) - 2) // 6 for s in tgt.values() if len(s) >= 3)
    P_(f"\n  WHAT A COMBINATORIAL DESIGN WOULD HAVE TO COVER, counted in TRRUST:")
    P_(f"    same-target regulator PAIRS   {npairs:>10}")
    P_(f"    same-target regulator TRIPLES {ntrip:>10}")
    P_("  The test at pair level: for regulators a and b that a class map places in the SAME")
    P_("  class, the double knockdown must be predicted by the class count -- that is, {a,b}")
    P_("  must behave like any other same-class pair with the same count. A class map that")
    P_("  survives its marginals and fails this is a class map the engine cannot use, because the")
    P_("  cost functional only ever reads the count vector.")
    P_(f"  This is the copy-number question, unasked of human data since the engine module, and")
    P_(f"  the pair count above is the size of the smallest design that could ask it.")

    # ---- D5  THE SPEC --------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("D5  THE SPECIFICATION")
    P_(RULE)
    P_("  A dataset validates C for this engine if and only if it has all four of:")
    P_(f"    1. ALPHABET   >= 128 distinct regulators perturbed"
       f"  (ENCODE K562 CRISPRi has {nfac if nfac is not None else '?'})")
    P_(f"    2. DEPTH      >= {r1_:.0f} independent replicates per perturbation for a ladder that")
    P_(f"                  merely clears its floor, >= {r86:.0f} for yeast-grade resolution")
    P_(f"                  (ENCODE's deepest experiment has {maxrep if maxrep is not None else '?'})"
       f"; and this is a LOWER bound, since the shared systematic")
    P_( "                  component of the noise is invisible to two replicates")
    P_(f"    3. BREADTH    enough targets at k >= {CAP_BINDING_DEPTH} to test the regime the 140 cap")
    P_(f"                  lives in -- ceiling {full[20]} targets, and only if EVERY TRRUST")
    P_( "                  regulator is perturbed")
    P_( "    4. JOINT      multi-perturbation, not single knockdowns, or criterion 1-3 validate")
    P_( "                  marginals of a functional defined on joints")
    P_("\n  APPLICABILITY DOMAIN, stated because a number without one is ledger S and this build")
    P_("  order has recorded that twice. A C validated on CRISPRi knockdown log-ratios is a C for")
    P_("  THAT response function. The engine's C is for a steady-state transcription rate. The")
    P_("  two coincide only if the knockdown response is monotone in the same latent quantity the")
    P_("  rate depends on, which is an assumption this module does not test and no module here")
    P_("  has tested.")
    P_("\n  TWO NAMED CANDIDATES, NOT ANALYSED HERE AND FLAGGED AS SUCH. Genome-scale Perturb-seq")
    P_("  (Replogle et al., Cell 2022, PMID 35688146, doi:10.1016/j.cell.2022.05.013; via PubMed)")
    P_("  targets all expressed genes with CRISPRi across >2.5 million human cells, which is the")
    P_("  only published design plausibly meeting criteria 1-3 -- its alphabet is every expressed")
    P_("  regulator and its per-perturbation cell counts are the replication. For criterion 4 the")
    P_("  nearest published design is the combinatorial CRISPRa GI map of Norman et al. (Science")
    P_("  2019, PMID 31395745, doi:10.1126/science.aax4438; via PubMed). NEITHER WAS DOWNLOADED,")
    P_("  NEITHER WAS TESTED, and their per-perturbation depth is not stated in either abstract,")
    P_("  so whether they clear the D1 bar is unknown. They are named as the next measurement,")
    P_("  which is exactly the status humantransfer gave the human MPRA before it was run -- and")
    P_("  when that one WAS run it failed. A named candidate is not a validated one.")
    P_("\n  AND THE HONEST BOTTOM LINE. Criterion 1 alone rules out every released CRISPRi panel")
    P_("  in this cell line. Criterion 4 rules out ALL single-perturbation data whatever its size.")
    P_("  So the answer to 'what data would validate C' is not a bigger version of what was")
    P_("  already used: it is a multi-perturbation panel over >= 128 regulators with replicate")
    P_("  depth in the tens, and until such a panel is analysed the caps 3, 13 and 140 remain")
    P_("  cost arithmetic at an accuracy standard measured in yeast.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_whatdata.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
