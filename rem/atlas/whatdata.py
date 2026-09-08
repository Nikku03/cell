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

D1b THE BAR D1 SHOULD HAVE PUT IT ON, PREDECLARED AFTER D1 RAN AND BEFORE D1b DID. D1's first
    run returned R(bar = 1) = 2 -- that is, the CURRENT depth already clears it -- because the
    between-regulator sd is 1.05 floors and D1 set its bar directly on that sd. But the quantity
    that was degenerate is not the sd: it is the LADDER SPAN, the difference in held-out RMSE
    between the coarsest and finest class model, and RMSE differences COMPRESS a signal. A gate
    whose bar is met at the current depth tells you nothing about what to fix, which is a bar no
    evidence can fail -- the mirror of ledger P -- and D1's arithmetic is therefore superseded
    rather than deleted, and left in the output above this gate so the error is visible.
    D1b puts the bar on the estimator that is actually used, and measures rather than derives it:
    simulate a world in which the class hypothesis is EXACTLY TRUE with C = 64 shared classes, at
    the measured between-regulator variance and the measured noise floor, with the real TRRUST
    depth distribution; build the ladder the way signed.py and humantransfer build it; and sweep
    panel size N and replicate depth R to find where the knee is recovered.
    Because the simulation grants a purely shared class structure -- which genegroups measured to
    be a MINORITY of the real signal -- every requirement it returns is a LOWER BOUND.
    PREDECLARED: the knee counts as recovered if it lands within a factor of two of C = 64. If no
    (N, R) in the sweep recovers it, the spec must say so instead of quoting the smallest number
    in the table.

D1c THE CONTROL THAT DECIDES WHAT D1b's SATURATION MEANS, PREDECLARED BEFORE IT RAN. D1b's knee
    stops rising long before it reaches the C = 64 it was told to find, and there are two
    completely different reasons that could happen. Either the ESTIMATOR cannot return 64 at any
    depth -- in which case "smallest C within one floor" measures the RESOLUTION of the
    measurement and not the structure of the biology, and signed.py's C = 64 is a statement about
    yeast's precision rather than about kinds of regulator. Or the estimator can return 64 when
    the 64 levels are genuinely separable, and D1b's saturation only says that 64 levels drawn
    from a continuous spread are not separable at these depths.
    THE CONTROL: rerun the identical sweep with the 64 class effects EQUALLY SPACED at exactly
    one floor apart, so they are separable by construction, and see whether the knee reaches 64.
    Then report the SCALING LAW of the knee in replicate depth rather than a point, because this
    build order has had to correct three approximations ranked at a point, and invert it to the
    depth at which 64 becomes resolvable.
    PREDECLARED: if the separated control ALSO saturates below 64, the estimator is
    resolution-limited whatever the truth is and C is not a structural constant. If the separated
    control reaches 64, then C = 64 is recoverable in principle and the requirement is a signal
    to floor ratio, which the inversion turns into a replicate depth.

D1d WHAT C ACTUALLY IS, AND THE CAP AS A CURVE. PREDECLARED BEFORE RUNNING. If D1c says the
    knee is a resolution rather than a structure, then C is not a constant of the biology waiting
    to be measured -- it is set by how finely the regulator effect spread has to be represented,
    which is the ratio of that spread to the accuracy demanded. That is a DESIGN parameter of the
    engine, not a fact about human cells. If so, the honest object is not a cap at C = 64 but the
    cap as a FUNCTION of demanded accuracy.
    Fit the law relating the recovered knee to the signal-to-floor ratio across every simulation
    point in D1b and D1c, then push it through scalarcap's cap functional to get that curve.
    PREDECLARED: the fit is reported as a LAW only if it holds across both simulation families --
    continuously spread effects and separated ones -- since a law fitted to one family and
    applied to the other is the error prune.py's N2 made. Otherwise it is reported as a trend
    and the curve is labelled indicative.

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
# D1b  the power calculation, on the estimator that is actually used
# =================================================================================================

def ladder_knee(N, C_true, R, G, sigma, ph, kdist, rng, Cgrid=None, sep_floors=None):
    """Simulate the class ladder under a world where the class hypothesis is exactly true.

    Regulators carry a class effect shared across genes -- the best case for every summary this
    build order has tested. Half the replicates fit, half are scored, so the floor on the scored
    quantity is sigma/sqrt(R/2), which is what replication actually buys.

    Returns (span in floors, recovered knee), where the knee is the smallest C whose held-out
    RMSE is within one floor of the finest model -- signed.py's criterion, verbatim."""
    if Cgrid is None:
        Cgrid = [c for c in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if c <= N] + [N]
        Cgrid = sorted(set(Cgrid))
    cls = rng.integers(0, C_true, N)
    if sep_floors is None:
        eff = rng.normal(0.0, np.sqrt(ph), C_true)
    else:
        step = sep_floors * (sigma / np.sqrt(max(R // 2, 1)))
        eff = (np.arange(C_true) - (C_true - 1) / 2.0) * step
    gi, ti = [], []
    for g in range(G):
        K = int(kdist[rng.integers(len(kdist))])
        K = min(K, N)
        for t in rng.choice(N, K, replace=False):
            gi.append(g)
            ti.append(int(t))
    ti = np.array(ti)
    if ti.size == 0:
        return 0.0, 1
    mu = eff[cls[ti]]
    half = max(R // 2, 1)
    fl = sigma / np.sqrt(half)
    yf = mu + rng.normal(0.0, fl, ti.size)
    yt = mu + rng.normal(0.0, fl, ti.size)
    # per-regulator effect fitted on the fitting half only
    sums = np.bincount(ti, weights=yf, minlength=N)
    cnts = np.bincount(ti, minlength=N).astype(float)
    seen = cnts > 0
    fitted = np.where(seen, sums / np.maximum(cnts, 1), 0.0)
    rank = np.empty(N, dtype=int)
    rank[np.argsort(fitted, kind="stable")] = np.arange(N)
    errs = {}
    for C in Cgrid:
        lab = np.minimum((rank * C) // N, C - 1)
        ls = np.bincount(lab[ti], weights=yf, minlength=C)
        lc = np.bincount(lab[ti], minlength=C).astype(float)
        pred = np.where(lc[lab[ti]] > 0, ls[lab[ti]] / np.maximum(lc[lab[ti]], 1), 0.0)
        errs[C] = float(np.sqrt(np.mean((yt - pred) ** 2)))
    fine = errs[max(Cgrid)]
    span = (errs[1] - fine) / fl
    knee = min((C for C in Cgrid if errs[C] <= fine + fl), default=max(Cgrid))
    return span, knee


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

    depths, outdeg, tgt = trrust_depths()

    # ---- D1b  THE BAR ON THE RIGHT QUANTITY ----------------------------------------------------
    P_("\n" + RULE)
    P_("D1b  THE BAR D1 PUT ON THE WRONG QUANTITY, AND THE POWER CALCULATION THAT REPLACES IT")
    P_(RULE)
    P_(f"  D1 returned R(bar = 1) = {r1_:.0f}: the CURRENT depth already clears it, because the")
    P_( "  between-regulator sd is 1.05 floors and D1 set its bar directly on that sd. A bar the")
    P_( "  data already meets cannot tell you what to fix. The quantity that was degenerate in")
    P_( "  scalarcap is not the sd -- it is the LADDER SPAN, a difference of held-out RMSEs, and")
    P_( "  RMSE differences compress a signal. D1's arithmetic is superseded, left above so the")
    P_( "  error is visible, and the bar is now put on the estimator that is actually used.")
    P_( "\n  THE SIMULATION GRANTS THE CLASS HYPOTHESIS. Regulator effects are drawn from C = 64")
    P_( "  classes SHARED ACROSS GENES, at the measured between-regulator variance and the")
    P_( "  measured floor, with TRRUST's real depth distribution. genegroups measured the")
    P_( "  shared-across-genes component to be a MINORITY of the real signal, so this is the best")
    P_( "  case and every requirement below is a LOWER BOUND on what real data would need.")
    kdist = np.array([len(s_) for s_ in tgt.values() if len(s_) >= 2])
    P_(f"\n  {'N regs':>7} {'R reps':>7} {'targets':>8} {'ladder span, floors':>21} {'knee recovered':>16}")
    sweep = []
    for Np in (48, 128, 404, len(outdeg)):
        av = {k: sum(1 for s_ in tgt.values() if len(s_) >= k) for k in (2,)}
        G = min(av[2], 1400)
        for R in (2, 4, 8, 16, 32, 64, 128):
            sp, kn = [], []
            for rep in range(5):
                rr = np.random.default_rng(4000 + 97 * Np + 7 * R + rep)
                a, b = ladder_knee(Np, 64, R, G, sigma, ph, kdist, rr)
                sp.append(a)
                kn.append(b)
            spm, knm = float(np.mean(sp)), float(np.median(kn))
            ok = (32 <= knm <= 128) and Np >= 64
            sweep.append((Np, R, G, spm, knm, ok))
            P_(f"  {Np:>7} {R:>7} {G:>8} {spm:>21.2f} {f'{knm:.0f}':>16}"
               + ("   <-- recovered" if ok else ""))
    good = [s_ for s_ in sweep if s_[5]]
    if good:
        best = min(good, key=lambda s_: (s_[1], s_[0]))
        P_(f"\n  D1b: the knee is recovered. The cheapest point in the sweep that does it is")
        P_(f"  N = {best[0]} regulators at R = {best[1]} replicates, ladder span {best[3]:.2f} floors.")
        P_( "  Below that depth the ladder is narrower than its own floor and 'smallest C within")
        P_( "  one floor' returns 1 -- which is exactly the degeneracy scalarcap recorded, now")
        P_( "  reproduced from first principles rather than discovered after the fact.")
    else:
        P_("\n  D1b: NO point in the sweep recovers the knee. As predeclared, the spec says so")
        P_("  rather than quoting the smallest number in the table.")
    at48 = [s_ for s_ in sweep if s_[0] == 48]
    P_(f"\n  AND THE ALPHABET CEILING SHOWS UP HERE ON ITS OWN: at N = 48 the recovered knee is")
    P_(f"  {min(s_[4] for s_ in at48):.0f}-{max(s_[4] for s_ in at48):.0f} at every depth, because a panel of 48 cannot express 64 classes")
    P_( "  however many times it is replicated. Depth cannot buy alphabet.")

    # ---- D1c  WHAT THE SATURATION MEANS --------------------------------------------------------
    P_("\n" + RULE)
    P_("D1c  IS THE KNEE A STRUCTURAL COUNT OR A RESOLUTION? THE CONTROL")
    P_(RULE)
    P_("  D1b's knee stops rising long before C = 64, the number it was told to find. Two")
    P_("  incompatible readings, and the control below separates them. PREDECLARED above.")
    P_(f"\n  THE SCALING LAW FIRST, since ranking at a point is the error corrected three times here.")
    at404 = [(s_[1], s_[4]) for s_ in sweep if s_[0] == 404]
    Rs = np.array([a for a, _ in at404], float)
    Ks = np.array([b for _, b in at404], float)
    al = np.polyfit(np.log(Rs), np.log(Ks), 1)
    P_(f"    knee ~ R^{al[0]:.2f}   (fitted over R = {int(Rs.min())}..{int(Rs.max())} at N = 404)")
    P_(f"    an exponent near 0.5 is the signature of a RESOLUTION statistic: the number of")
    P_(f"    distinguishable levels grows as the square root of the precision, which is what")
    P_(f"    quantising a fixed spread into equal-count bins does. A structural count would be")
    P_(f"    FLAT in R -- it would find 64 and stop.")
    R64 = float(np.exp((np.log(64.0) - al[1]) / al[0])) if al[0] > 0 else float("inf")
    P_(f"    extrapolating that law to a knee of 64 needs R = {R64:,.0f} replicates.")
    P_("\n  NOW THE CONTROL: the same 64 classes, EQUALLY SPACED one floor apart, separable by")
    P_("  construction. If the estimator can ever return 64, it returns it here.")
    P_(f"\n    {'N regs':>7} {'R reps':>7} {'span, floors':>14} {'knee':>7} {'signal, floors':>16}")
    hit = []
    for Np in (128, 404):
        for R in (2, 8, 32):
            sp, kn = [], []
            for rep in range(5):
                rr = np.random.default_rng(9000 + 31 * Np + 5 * R + rep)
                a, b = ladder_knee(Np, 64, R, 1400, sigma, ph, kdist, rr, sep_floors=1.0)
                sp.append(a); kn.append(b)
            step_sd = np.sqrt((64.0 ** 2 - 1) / 12.0)          # sd of the spaced effects, in floors
            P_(f"    {Np:>7} {R:>7} {np.mean(sp):>14.2f} {np.median(kn):>7.0f} {step_sd:>16.1f}")
            hit.append(float(np.median(kn)))
    sep_sd_floors = float(np.sqrt((64.0 ** 2 - 1) / 12.0))
    if max(hit) >= 32:
        P_(f"\n  D1c: THE ESTIMATOR CAN RETURN 64 -- but only in a world whose between-regulator")
        P_(f"  signal is {sep_sd_floors:.0f} floors. The measured human value is {rho:.2f} floors. Since the signal in")
        P_(f"  floors grows as sqrt(R/2), reaching {sep_sd_floors:.0f} needs R = {2*(sep_sd_floors/rho)**2:,.0f} replicates per")
        P_(f"  perturbation -- and that is the LOWER bound again, because the systematic component")
        P_(f"  of the noise does not shrink with R at all.")
        P_(f"\n  SO C IS NOT A STRUCTURAL CONSTANT AT THE PRECISION ANYONE ACTUALLY HAS. The knee the")
        P_(f"  criterion returns is min(true class count, what the precision can resolve), and on")
        P_(f"  every real dataset in this build order the second term binds. signed.py's C = 64")
        P_(f"  is therefore a statement about YEAST'S PRECISION as much as about kinds of")
        P_(f"  regulator, and the caps that rest on it inherit that. This is not a refutation of")
        P_(f"  the caps -- the cost arithmetic is unchanged -- but it relocates what C is.")
    else:
        P_(f"\n  D1c: the separated control ALSO saturates (knee {max(hit):.0f}). The estimator is")
        P_(f"  resolution-limited whatever the truth is, and 'smallest C within one floor' cannot")
        P_(f"  return a structural class count at any depth. That is stronger than the reading")
        P_(f"  above and worse for the caps: C would then have no measurement that defines it.")

    # ---- D1d  THE LAW, AND THE CAP AS A CURVE --------------------------------------------------
    P_("\n" + RULE)
    P_("D1d  IF C IS A RESOLUTION, THE CAP IS A CURVE. THE LAW, AND THE CURVE")
    P_(RULE)
    xs, ys, fam = [], [], []
    for Np, R, _G, _sp, kn, _ok in sweep:
        if Np < 128:
            continue
        xs.append(rho * np.sqrt(max(R // 2, 1)))
        ys.append(max(kn, 1.0))
        fam.append("spread")
    for k_ in hit:
        xs.append(sep_sd_floors)
        ys.append(max(k_, 1.0))
        fam.append("separated")
    xs, ys = np.array(xs), np.array(ys)
    b, a = np.polyfit(np.log(xs), np.log(ys), 1)
    pred = np.exp(a) * xs ** b
    ss = 1.0 - np.sum((np.log(ys) - np.log(pred)) ** 2) / np.sum((np.log(ys) - np.log(ys).mean()) ** 2)
    P_(f"  pooled over {len(xs)} simulation points from BOTH families:")
    P_(f"    knee  =  {np.exp(a):.2f} * (signal in floors) ^ {b:.2f}        R^2 = {ss:.3f}")
    both = all(any(f == g for f, g in zip(fam, fam)) for g in ("spread", "separated"))
    holds = ss >= 0.85 and ("spread" in fam and "separated" in fam)
    P_(f"  {'IT HOLDS ACROSS BOTH FAMILIES -- reported as a law.' if holds else 'It does not hold across both families -- reported as a TREND, and the curve below is indicative only.'}")
    P_("\n  READ PLAINLY: the number of regulator classes a dataset supports is roughly the")
    P_("  regulator effect spread divided by the accuracy demanded of it. Demand less accuracy")
    P_("  and C falls; demand more and C rises. It is a quantisation count.")
    P_("\n  SO THE CAP IS NOT A NUMBER AT C = 64. IT IS THIS CURVE:")
    P_(f"\n    {'demanded accuracy':>19} {'C supported':>12} {'cap, controllers':>18}")
    from rem.atlas.scalarcap import cap_with_scalar
    MAXC = int(os.environ.get("REM_WHATDATA_MAXC", "2000"))
    spread = rho          # measured between-regulator spread, in units of the current floor
    sat_any = False
    for mult in (4.0, 2.0, 1.0, 0.5, 0.25):
        Cq = int(max(1, round(np.exp(a) * (spread / mult) ** b)))
        try:
            cp = cap_with_scalar(max(Cq, 1), True, maxC=MAXC)
        except Exception as e:                                # pragma: no cover
            P_(f"    cap functional failed at C = {Cq}: {e}")
            continue
        sat = cp >= MAXC - 1
        sat_any = sat_any or sat
        P_(f"    {f'{mult:.2f} floors':>19} {Cq:>12} {(f'>= {cp}  SATURATED' if sat else str(cp)):>18}")
    P_("\n  A ROW MARKED SATURATED IS NOT A MEASUREMENT. The cap functional walks the controller")
    P_(f"  count upward until the bar is breached and stops at its own ceiling of {MAXC}; a row that")
    P_("  reaches the ceiling means the bar was never breached, so the number is the ceiling and")
    P_("  not the cap.")
    if sat_any:
        P_("  Some rows above are saturated and are reported as lower bounds only.")
    P_("\n  AND THAT DEFECT REACHES BACKWARD, INTO THE MODULE SHIPPED BEFORE THIS ONE. scalarcap")
    P_("  called the same functional with maxC = 60 and reported a cap of 59 at C = 4. 59 is")
    P_("  60 - 1: that row was ITS OWN LOOP CEILING, not a cap. scalarcap's V0 VERDICT SURVIVES")
    P_("  -- with and without the scalar both hit the same ceiling, so the scalar still changes")
    P_("  nothing -- but the NUMBER 59 was a saturated statistic reported as a measurement,")
    P_("  which is ledger U, in a module whose own V1 was withdrawn for exactly that. The rows")
    P_("  at C = 8, 16 and 64 gave 30, 15 and 13, well under the ceiling, and are unaffected.")
    P_("\n  The left column is what the engine is asked to reproduce, in units of the human")
    P_("  panel's own replicate noise. The right column is how many controllers fit under the")
    P_("  bar at that standard. Every previously reported cap is ONE ROW of this table, and the")
    P_("  row was chosen in yeast.")
    P_("\n  WHAT THIS DOES NOT ESTABLISH, AND THE DISCREPANCY THAT SAYS SO. The law above is")
    P_("  measured INSIDE the simulation. It licenses the statement that the estimator returns a")
    P_("  resolution count -- it could not return 64 even when told 64 was true -- and signed.py")
    P_("  uses that estimator, so signed.py's C = 64 is at least partly a resolution. It does NOT")
    P_("  license reading 64 off this law. Doing so needs yeast's signal-to-floor ratio in these")
    P_(f"  same units; the one crude figure available is about {rho*15:.0f} floors, which the law maps to")
    P_(f"  a knee near {np.exp(a)*(rho*15)**b:.0f}, not 64. That 4x discrepancy is unexplained here and is itself the")
    P_("  reason not to over-read the reframing: what is established is that the criterion cannot")
    P_("  identify a structural class count, not that yeast's particular 64 is fully accounted")
    P_("  for. Reconciling it needs signed.py's ladder rebuilt in floors, which this module did")
    P_("  not do.")

    # ---- D2  THE ALPHABET AXIS -----------------------------------------------------------------
    P_("\n" + RULE)
    P_("D2  THE ALPHABET AXIS: HOW MANY FACTORS MUST BE PERTURBED?")
    P_(RULE)
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
    if good:
        P_(f"    2. DEPTH      >= {best[1]} independent replicates per perturbation -- the cheapest point")
        P_(f"                  in D1b's sweep that recovers the knee, at N = {best[0]}")
        P_(f"                  (ENCODE's deepest experiment has"
           f" {maxrep if maxrep is not None else '?'}). This is a LOWER bound twice over:")
        P_( "                  the simulation grants a purely shared class structure, and the")
        P_( "                  systematic component of the noise is invisible to two replicates")
        P_( "                  and is not reduced by replication at all")
    else:
        P_( "    2. DEPTH      UNRESOLVED -- no point in D1b's sweep recovered the knee")
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
