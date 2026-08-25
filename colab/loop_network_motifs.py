"""Loop 187. Feedforward and feedback loops in the K562 regulatory network -- are they there beyond
chance, and is their sign logic coherent?

WHY THIS QUESTION, AND WHY IT IS ASKED STRUCTURALLY FIRST. A feedforward loop is three edges: A
regulates B, and both A and B regulate C. A feedback loop is a cycle. Both are the classic network
motifs (Shen-Orr, Milo & Alon, Nat Genet 2002), and the claim attached to them is functional --
coherent feedforward loops act as sign-sensitive delays and persistence detectors, incoherent ones
as pulse generators, negative feedback speeds a response and positive feedback makes it switch-like
(Alon, Nat Rev Genet 2007). None of that can be tested here today: the Perturb-seq matrices this
project used for such things are not on disk after the container reset, and re-fetching the raw
single-cell data is a multi-gigabyte download against 2.7 GB of headroom. So this loop does the half
that IS fully answerable from what is here -- whether the motifs exist above what the network's own
degree sequence forces, and whether their signs are coherent -- and says plainly that the functional
half is not attempted.

WHAT IS USED. This project's assembled TF network: 612,133 signed edges over 16,492 gene symbols,
partitioned by row order into three tiers that must not be pooled (the partition and that warning
are both recorded in cell_tfnet.json):

    curated causal   rows      0- 55,716   CollecTRI, literature-curated with a direction and a sign
    binding          rows 55,716-278,405   DoRothEA A-D, largely occupancy-derived
    unidentified     rows 278,405-612,133  provenance recorded as unknown, best guess ENCODE ChIP

THE CONTROL THAT DECIDES EVERYTHING HERE, and it is not optional. Feedforward triangles appear in
ANY network with a broad out-degree distribution, purely by arithmetic: a regulator with 370 targets
and another with 300 will share targets whether or not anything circuit-like is happening. So every
count is compared against DEGREE-PRESERVING RANDOMISATIONS -- double-edge swaps that hold every
node's in-degree and out-degree exactly -- and the z-score against that ensemble is the result. A
raw triangle count is not evidence of anything.

AND ONE THAT DECIDES WHAT AN ENRICHMENT MEANS. The binding tier is occupancy-derived: two factors
binding the same promoter create A->C and B->C automatically, and A->B as soon as one of them binds
the other's promoter. Feedforward "enrichment" in that tier is what promoter co-occupancy looks like
when it is written down as a graph, not a circuit. Loop 183's W4 learned this the hard way on the
same network, so B3 requires the curated tier to out-enrich the binding tier before any circuit
claim is made.

PREDECLARED, BEFORE ANY NUMBER.

  B1 IS THE NETWORK USABLE FOR THIS? Per-tier node and edge counts, degree distributions, and what
     fraction of curated edges carry a usable sign.
     Gate: PASS iff at least half the curated-tier edges carry an activation or repression sign.
     Without signs the coherence question cannot be asked and B4 would be measuring nothing.

  B2 ARE FEEDFORWARD LOOPS ENRICHED? Triads A->B, A->C, B->C counted per tier against 100
     degree-preserving randomisations.
     Gate: PASS iff the curated tier's z-score exceeds 3.

  B3 IS THE ENRICHMENT A CO-BINDING ARTEFACT? The curated tier's z-score against the binding
     tier's.
     Gate: PASS iff the curated z exceeds the binding z. A FAIL means the motif signal lives where
     the edges are occupancy, and the honest reading is promoter co-occupancy rather than circuitry.

  B4 ARE FEEDFORWARD LOOPS COHERENT? An FFL is coherent when the sign of the direct edge A->C
     matches the product of the indirect path, sign(A->B) x sign(B->C). E. coli's answer is about
     85% coherent (Mangan & Alon, PNAS 2003).
     Gate: PASS iff the coherent fraction exceeds what a SIGN-SHUFFLED null gives -- the same
     topology with the signs permuted across edges -- by more than 3 standard deviations. The
     topology is held fixed so this tests the sign logic and nothing else.

  B5 ARE FEEDBACK LOOPS ENRICHED? Two-cycles (A->B and B->A) and three-cycles, against the same
     degree-preserving ensemble, with the sign product reported so positive and negative feedback
     are separated.
     Gate: PASS iff the curated tier's two-cycle z-score exceeds 3.

  B6 AUTOREGULATION AGAINST CHANCE. Loop 175 found 24 self-loops among 795 curated regulators and
     reported 3.0% as a small number against the plan's expected 50%. It never asked what chance
     would give. With 55,716 edges spread over ~800 regulators and ~15,000 targets, chance gives
     very few, so 3.0% may be a large enrichment reported as a small fraction.
     Gate: PASS iff the self-loop count exceeds the degree-preserving null by more than 3 sd. This
     gate exists to correct a framing in loop 175's record, whichever way it comes out.

  B7 WHAT THIS CANNOT SHOW.

  WHY THIS IS A RERUN. The first run reached B1 and the curated tier's ensemble -- 95.3% of curated
  edges signed, FFL 315,423 against a null of 312,531.9 +/- 3,617.0 for z = +0.8, two-cycle
  z = +43.8, three-cycle z = +3.8 -- and then the container was reclaimed during the binding tier's
  randomisations, so B2 through B7 were never scored. Partial ensembles are now checkpointed per
  tier so a second reclaim costs minutes rather than the whole run.

  AND WHAT THE SELF-CHECK FOUND, which is the reason the first run's FFL numbers do NOT stand. The
  matrix counter and the nested-loop counter disagreed on random graphs, always by exactly twice
  the two-cycle count. The loop counter was the wrong one. It intended to exclude the degenerate
  triads c == a and c == b, and for c == b it tested the right thing; for c == a it tested only
  whether b -> a exists. But c == a needs c to be in out(a) as well, which means a -> a -- a
  self-loop. With no self-loops anywhere in this network, that subtraction should never fire, and
  instead it fired once for every ordered pair joined in both directions. So every feedforward
  count in the first run, observed AND null, was low by 2 x (two-cycles), and the curated tier's
  1,630 two-cycles put the observed count 3,260 low while the null's ~714 put the null ~1,428 low.
  The corrected counts are what this run reports, and the first run's 315,423 is recorded here as
  wrong rather than quietly replaced. The direction of B2 is not expected to change -- the error
  was larger in the observation than in the null, so correcting it can only raise the z-score, and
  the raw gap it has to close is +0.8 against a bar of 3.0 -- but the number itself changes and the
  z-score is stated freshly rather than carried over.

  A NOTE ON THE NULLS. Both tiers now get the SAME null: 100 degree-preserving randomisations with
  every motif counted. The first attempt at this loop did not, and the reason it did not was
  arithmetic rather than principle -- the three-cycle count written as nested Python loops is about
  10^8 operations per call on the binding tier, so the binding tier was given 25 draws without
  three-cycles and the log said so. That is a bad position to argue B3 from, because B3 compares
  the two tiers' z-scores and an unequal null makes the comparison partly about the null. All four
  counts are now linear algebra on the adjacency matrix -- feedforward triads are sum(A * A^2),
  two-cycles are sum(A * A^T)/2, three-cycles are trace(A^3)/3 -- which BLAS does in under a second
  where the loops took minutes, and the weaker null is no longer needed for anything.

  THE SELF-CHECK THAT GUARDS THAT SUBSTITUTION. A fast reimplementation of a count is a chance to
  silently change what is being counted. So the original nested-loop counter is KEPT in the file as
  stats_ref, and before any gate is scored the two implementations are run against each other on
  the observed curated graph, on a randomised curated graph, and on the observed binding graph;
  the coherence enumeration is checked the same way. Any disagreement aborts the loop rather than
  being reported. The same applies to the sign-shuffled null, which now enumerates the feedforward
  triads once as edge indices and permutes signs into that fixed array, because the topology is
  held constant across that null by construction.

-> outputs/loop_network_motifs.json
"""
import gzip
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_network_motifs.json"
BUNDLE = Path("colab/data/net_bundle.json.gz")
TIERS = {"curated": (0, 55716), "binding": (55716, 278405), "unidentified": (278405, None)}
N_RAND = 100             # both tiers, identically -- see the note on the nulls in the docstring
SWAP_FACTOR = 10
# np.savez APPENDS .npz to any path not already ending in it, so the name must end in .npz
# or the file lands somewhere ck.exists() never looks and resume silently never fires.
CKPT = Path(os.environ.get("CELL_OUT", "outputs")) / "l187_null_{}.npz"
MIN_SIGNED = 0.50
Z_BAR = 3.0
SEED = 187187

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def load_tier(reg, lo, hi):
    e = reg[lo:(hi if hi is not None else len(reg))]
    return [(int(r[0]), int(r[1]), int(r[2]) if len(r) > 2 else 0) for r in e]


def stats_ref(edges, cycles3=True):
    """The nested-loop reference counter: FFL triads, 2-cycles, 3-cycles and self-loops.

    This is the ORIGINAL implementation and it is kept only to check the fast one against. It is
    too slow to build an ensemble with -- the three-cycle count is O(sum_a sum_{b in out(a)}
    |out(b)|), about 10^8 Python operations per call on the binding tier -- which is exactly why
    stats_dense exists. `cycles3` switches the three-cycle count off so the reference can still be
    run once on the dense tier for the self-check without costing an hour."""
    out = defaultdict(set)
    for a, b, s in edges:
        out[a].add(b)
    selfl = sum(1 for a, b, s in edges if a == b)
    ffl = 0
    reg = set(out)
    for a in reg:
        ta = out[a]
        for b in ta:
            if b == a or b not in out:
                continue
            # the two subtractions remove c == a and c == b. BOTH require c to actually be in
            # the intersection, which is what the first of them originally failed to check: it
            # fired on b -> a alone, and c == a additionally needs a -> a. See the docstring.
            ffl += (len(ta & out[b])
                    - (1 if (a in ta and a in out[b]) else 0)
                    - (1 if (b in ta and b in out[b]) else 0))
    two = 0
    for a in reg:
        for b in out[a]:
            if b != a and b in out and a in out[b] and a < b:
                two += 1
    three = 0
    for a in (reg if cycles3 else ()):
        for b in out[a]:
            if b == a or b not in out:
                continue
            for c in out[b]:
                if c == a or c == b or c not in out:
                    continue
                if a in out[c] and a < b and a < c:
                    three += 1
    return dict(ffl=ffl, two_cycle=two, three_cycle=three, self_loop=selfl,
                n_reg=len(reg), n_edge=len(edges))


class DenseIndex:
    """Fixed row and column maps for one tier, built once and reused for every randomisation.

    This reuse is legitimate for exactly one reason: the double-edge swap preserves every node's
    in-degree and out-degree EXACTLY, so the set of nodes with an out-edge and the set with an
    in-edge are identical in the observed graph and in every randomisation of it. If the swap ever
    stopped being degree-preserving this cache would silently lie, which is one more thing the
    self-check against stats_ref would catch."""

    def __init__(self, edges):
        regs = sorted({a for a, b, s in edges})
        nodes = sorted({x for a, b, s in edges for x in (a, b)})
        hi = max(nodes) + 1
        self.row_of = np.full(hi, -1, dtype=np.int64)
        self.col_of = np.full(hi, -1, dtype=np.int64)
        for i, a in enumerate(regs):
            self.row_of[a] = i
        for i, n in enumerate(nodes):
            self.col_of[n] = i
        self.rcols = self.col_of[np.array(regs, dtype=np.int64)]
        self.nr, self.nn = len(regs), len(nodes)

    def matrix(self, edges):
        e = np.asarray(edges, dtype=np.int64)
        a, b = e[:, 0], e[:, 1]
        keep = a != b                                    # self-loops are counted separately
        M = np.zeros((self.nr, self.nn), dtype=np.float32)
        M[self.row_of[a[keep]], self.col_of[b[keep]]] = 1.0
        return M


def stats_dense(edges, ix, cycles3=True):
    """The same four counts as stats_ref, as linear algebra.

    With the diagonal zeroed, a feedforward triad is a pair (a,c) joined both directly and by one
    intermediate, so the count is sum(A * A^2) elementwise; a two-cycle is sum(A * A^T)/2; a
    three-cycle is trace(A^3)/3, written here as sum(A^2 * A^T) so the third matrix product is
    never formed. Only rows with an out-edge can start a path, so A is stored as the regulator
    rows only, and A^2 restricted to those rows is S @ M where S is the regulator-by-regulator
    block. Every intermediate is a non-negative integer count below 2^24, so float32 products are
    exact; the outer sums are accumulated in float64 because they are not."""
    M = ix.matrix(edges)
    S = M[:, ix.rcols]
    ffl = float(np.sum(M * (S @ M), dtype=np.float64))
    two = float(np.sum(S * S.T, dtype=np.float64)) / 2.0
    three = float(np.sum((S @ S) * S.T, dtype=np.float64)) / 3.0 if cycles3 else 0.0
    selfl = sum(1 for a, b, s in edges if a == b)
    return dict(ffl=int(round(ffl)), two_cycle=int(round(two)),
                three_cycle=int(round(three)), self_loop=selfl,
                n_reg=ix.nr, n_edge=len(edges))


def randomise(edges, rng, factor=SWAP_FACTOR):
    """Double-edge swap: (a1->b1, a2->b2) becomes (a1->b2, a2->b1). Every node's in-degree and
    out-degree is preserved EXACTLY, which is the whole point -- a triangle count is trivially
    predicted by the degree sequence and only the excess over it means anything."""
    e = [(a, b) for a, b, s in edges]
    m = len(e)
    have = set(e)
    n_try = factor * m
    for _ in range(n_try):
        i, j = rng.integers(0, m), rng.integers(0, m)
        if i == j:
            continue
        a1, b1 = e[i]
        a2, b2 = e[j]
        if b1 == b2 or a1 == b2 or a2 == b1:
            continue
        if (a1, b2) in have or (a2, b1) in have:
            continue
        have.discard((a1, b1))
        have.discard((a2, b2))
        have.add((a1, b2))
        have.add((a2, b1))
        e[i] = (a1, b2)
        e[j] = (a2, b1)
    return [(a, b, 0) for a, b in e]


KEYS = ("ffl", "two_cycle", "three_cycle", "self_loop")


def null_ensemble(edges, ix, n=N_RAND, seed=SEED, report=print, label="", cycles3=True):
    """Degree-preserving ensemble, checkpointed every 10 draws.

    The first attempt at this loop lost a completed 274-second curated ensemble when the container
    was reclaimed mid-way through the binding tier. The draws are deterministic given the seed, so
    a checkpoint is simply a prefix of the same sequence and resuming from it is not an
    approximation -- but the rng has to be advanced through the draws already banked, which is what
    the replay loop below does, and the checkpoint records the edge count it was built from so a
    stale file cannot be silently adopted."""
    ck = Path(str(CKPT).format(label))
    acc = {k: [] for k in KEYS}
    done = 0
    if ck.exists():
        try:
            z = np.load(ck)
            if int(z["n_edge"]) == len(edges) and bool(z["cycles3"]) == cycles3:
                for k in KEYS:
                    acc[k] = list(z[k])
                done = len(acc["ffl"])
                report(f"      {label}: resuming from checkpoint at {done}/{n}")
        except Exception as exc:                                     # noqa: BLE001
            report(f"      {label}: checkpoint unreadable ({exc}); starting over")
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for k in range(n):
        e = randomise(edges, rng)                # replayed even when banked, to keep the rng state
        if k < done:
            continue
        st = stats_dense(e, ix, cycles3=cycles3)
        for key in KEYS:
            acc[key].append(st[key])
        if (k + 1) % 10 == 0 or k + 1 == n:
            ck.parent.mkdir(parents=True, exist_ok=True)
            np.savez(ck, n_edge=len(edges), cycles3=cycles3,
                     **{key: np.array(acc[key], dtype=float) for key in KEYS})
        if (k + 1) % 25 == 0:
            report(f"      {label} randomisation {k+1}/{n}  [{time.time()-t0:.0f}s]")
    return {k: np.array(v, dtype=float) for k, v in acc.items()}


def z_of(obs, null):
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    return (obs - mu) / sd if sd > 0 else float("nan"), mu, sd


def coherence(edges):
    """Classify each FFL by whether sign(A->C) equals sign(A->B) x sign(B->C)."""
    out = defaultdict(dict)
    for a, b, s in edges:
        out[a][b] = s
    coh = inc = unsigned = 0
    for a in list(out):
        ta = out[a]
        for b in list(ta):
            if b == a or b not in out:
                continue
            for c in out[b]:
                if c == a or c == b or c not in ta:
                    continue
                s1, s2, s3 = ta[b], out[b][c], ta[c]
                if s1 == 0 or s2 == 0 or s3 == 0:
                    unsigned += 1
                elif s1 * s2 == s3:
                    coh += 1
                else:
                    inc += 1
    return coh, inc, unsigned


def ffl_triples(edges):
    """Every feedforward triad as three EDGE indices (a->b, b->c, a->c), enumerated once.

    The sign-shuffled null holds the topology fixed and permutes signs across edges, so the triads
    it scores are the same triads every time. Enumerating them once turns each of the 40 draws from
    a re-traversal of the graph into three array lookups. Duplicate (a,b) rows collapse to the last
    one, matching the reference, which stores edges in a dict keyed by (a,b)."""
    out = defaultdict(dict)
    for i, (a, b, s) in enumerate(edges):
        if a != b:
            out[a][b] = i
    tri = []
    for a, ta in out.items():
        for b, iab in ta.items():
            if b not in out:
                continue
            for c, ibc in out[b].items():
                if c == a or c == b or c not in ta:
                    continue
                tri.append((iab, ibc, ta[c]))
    return np.array(tri, dtype=np.int64).reshape(-1, 3)


def coh_frac(tri, signs):
    s1, s2, s3 = signs[tri[:, 0]], signs[tri[:, 1]], signs[tri[:, 2]]
    signed = (s1 != 0) & (s2 != 0) & (s3 != 0)
    coh = int(((s1 * s2 == s3) & signed).sum())
    return coh, int(signed.sum()) - coh, int((~signed).sum())


def sign_shuffle_null(tri, signs, n, seed, report=print):
    rng = np.random.default_rng(seed)
    fr = []
    for k in range(n):
        c, i2, u = coh_frac(tri, signs[rng.permutation(len(signs))])
        fr.append(c / max(c + i2, 1))
    return np.array(fr)


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 187  FEEDFORWARD AND FEEDBACK LOOPS: are they there beyond the degree sequence?")
    say("=" * 104)
    say(f"  PREDECLARED: at least {MIN_SIGNED:.0%} of curated edges must carry a sign; every motif")
    say(f"  count judged against {N_RAND} DEGREE-PRESERVING randomisations at z > {Z_BAR}; the")
    say("  curated tier must out-enrich the binding tier before any circuit claim is made, because")
    say("  two factors on one promoter make a feedforward triangle without a circuit; and coherence")
    say("  is judged against a SIGN-SHUFFLED null on the identical topology.")
    say("  NOT ATTEMPTED: the functional claims. The Perturb-seq matrices are not on disk after the")
    say("  container reset and re-fetching the raw single-cell data is a multi-gigabyte download")
    say("  against 2.7 GB of headroom. This loop is structure only and says so.")
    say()

    nb = json.load(gzip.open(BUNDLE))
    names, reg = nb["names"], nb["reg"]
    say(f"    network: {len(names):,} symbols, {len(reg):,} edges")

    # ---- B1 ------------------------------------------------------------------------------------
    say()
    say("B1 IS THE NETWORK USABLE FOR THIS?")
    tier = {}
    for t, (lo, hi) in TIERS.items():
        e = load_tier(reg, lo, hi)
        tier[t] = e
        sg = sum(1 for a, b, s in e if s != 0)
        outd = Counter(a for a, b, s in e)
        ind = Counter(b for a, b, s in e)
        say(f"     {t:13} {len(e):7,} edges, {len(outd):5,} regulators, {len(ind):6,} targets, "
            f"signed {sg/max(len(e),1):.1%}, out-degree median "
            f"{np.median(list(outd.values())):.0f} max {max(outd.values()):,}")
    frac_signed = sum(1 for a, b, s in tier["curated"] if s != 0) / max(len(tier["curated"]), 1)
    b1 = bool(frac_signed >= MIN_SIGNED)
    GG.verdict(b1, emit=say,
               if_true=f"B1 PASS -- {frac_signed:.1%} of curated edges carry a sign, so coherence "
                       f"can be asked",
               if_false=f"B1 FAIL -- only {frac_signed:.1%} of curated edges carry a sign; B4 "
                        f"would be measuring the unsigned majority")

    # ---- the self-check that guards the fast counter ---------------------------------------------
    say()
    say("   SELF-CHECK: the matrix counter against the nested-loop reference")
    ix = {t: DenseIndex(tier[t]) for t in ("curated", "binding")}
    checks = []
    for lbl, e, t, c3 in (("curated observed", tier["curated"], "curated", True),
                          ("curated randomised", randomise(tier["curated"],
                                                           np.random.default_rng(1)),
                           "curated", True),
                          ("binding observed", tier["binding"], "binding", False)):
        a = stats_ref(e, cycles3=c3)
        b = stats_dense(e, ix[t], cycles3=c3)
        keys = KEYS if c3 else ("ffl", "two_cycle", "self_loop")
        same = all(a[k] == b[k] for k in keys)
        checks.append(same)
        say(f"     {lbl:20} " + "  ".join(f"{k} {a[k]:,}" for k in keys)
            + ("   AGREE" if same else "   DISAGREE " + str({k: (a[k], b[k]) for k in keys})))
    tri = ffl_triples(tier["curated"])
    signs = np.array([s for a, b, s in tier["curated"]], dtype=np.int64)
    cr = coherence(tier["curated"])
    cd = coh_frac(tri, signs)
    checks.append(cr == cd)
    say(f"     coherence enumeration  reference {cr}  vectorised {cd}   "
        + ("AGREE" if cr == cd else "DISAGREE"))
    if not all(checks):
        say("     ABORT -- the fast counter does not reproduce the reference, so no gate below "
            "would be measuring what it says it measures")
        raise SystemExit(2)
    say(f"     all {len(checks)} checks agree")

    # ---- B2, B3, B5, B6 ------------------------------------------------------------------------
    say()
    say(f"   counting motifs and building {N_RAND} degree-preserving nulls per tier, "
        f"identically for both")
    obs, nulls, zs = {}, {}, {}
    for t in ("curated", "binding"):
        e = tier[t]
        o = stats_dense(e, ix[t])
        obs[t] = o
        say(f"     {t}: FFL {o['ffl']:,}  2-cycles {o['two_cycle']:,}  "
            f"3-cycles {o['three_cycle']:,}  self-loops {o['self_loop']}")
        nulls[t] = null_ensemble(e, ix[t], n=N_RAND, report=say, label=t)
        zs[t] = {}
        for key in KEYS:
            z, mu, sd = z_of(o[key], nulls[t][key])
            zs[t][key] = dict(obs=o[key], null_mean=mu, null_sd=sd, z=z)
            say(f"       {key:12} observed {o[key]:9,}  null {mu:11,.1f} +/- {sd:8,.1f}  "
                f"z {z:+9.1f}")

    say()
    say("B2 ARE FEEDFORWARD LOOPS ENRICHED?")
    zc = zs["curated"]["ffl"]["z"]
    say(f"     curated tier FFL z = {zc:+.1f} against {N_RAND} degree-preserving randomisations")
    b2 = bool(np.isfinite(zc) and zc > Z_BAR)
    GG.verdict(b2, emit=say,
               if_true=f"B2 PASS -- the curated network carries {obs['curated']['ffl']:,} "
                       f"feedforward triads against {zs['curated']['ffl']['null_mean']:,.0f} "
                       f"expected from its own degree sequence",
               if_false=f"B2 FAIL -- z {zc:+.1f}; the feedforward count is what the degree "
                        f"sequence alone predicts")

    say()
    say("B3 IS THE ENRICHMENT A CO-BINDING ARTEFACT?")
    zb = zs["binding"]["ffl"]["z"]
    say(f"     curated z {zc:+.1f} against binding z {zb:+.1f}, both from {N_RAND} "
        f"randomisations with every motif counted")
    b3 = bool(np.isfinite(zc) and np.isfinite(zb) and zc > zb)
    GG.verdict(b3, emit=say,
               if_true="B3 PASS -- the curated causal tier is more enriched than the occupancy "
                       "tier, so this is not simply two factors sharing a promoter",
               if_false="B3 FAIL -- the occupancy tier is at least as enriched, which is what "
                        "promoter co-occupancy looks like written as a graph; no circuit claim "
                        "survives this")

    # ---- B4 ------------------------------------------------------------------------------------
    say()
    say("B4 ARE FEEDFORWARD LOOPS COHERENT?")
    coh, inc, uns = coh_frac(tri, signs)
    tot = coh + inc
    frac = coh / max(tot, 1)
    say(f"     curated FFLs with all three edges signed: {tot:,} "
        f"({uns:,} had an unsigned edge and are excluded)")
    say(f"     coherent {coh:,} ({frac:.1%}), incoherent {inc:,} ({1-frac:.1%}); "
        f"E. coli's published answer is about 85% coherent")
    null_fr = sign_shuffle_null(tri, signs, 40, SEED, say)
    z4 = (frac - null_fr.mean()) / (null_fr.std(ddof=1) if null_fr.std(ddof=1) > 0 else np.nan)
    say(f"     sign-shuffled null on the identical topology: {null_fr.mean():.1%} "
        f"+/- {null_fr.std(ddof=1):.1%}  ->  z {z4:+.1f}")
    b4 = bool(np.isfinite(z4) and z4 > Z_BAR)
    GG.verdict(b4, emit=say,
               if_true="B4 PASS -- the sign logic is coherent beyond what the network's own sign "
                       "composition forces",
               if_false="B4 FAIL -- shuffling the signs across the same edges reproduces the "
                        "coherent fraction, so the coherence is a composition effect")

    # ---- B5 ------------------------------------------------------------------------------------
    say()
    say("B5 ARE FEEDBACK LOOPS ENRICHED?")
    z5 = zs["curated"]["two_cycle"]["z"]
    say(f"     curated 2-cycles {obs['curated']['two_cycle']:,}, "
        f"null {zs['curated']['two_cycle']['null_mean']:,.1f}, z {z5:+.1f}")
    say(f"     curated 3-cycles {obs['curated']['three_cycle']:,}, "
        f"null {zs['curated']['three_cycle']['null_mean']:,.1f}, "
        f"z {zs['curated']['three_cycle']['z']:+.1f}")
    out2 = defaultdict(dict)
    for a, b, s in tier["curated"]:
        out2[a][b] = s
    ppos = pneg = punk = 0
    for a in list(out2):
        for b in out2[a]:
            if b != a and b in out2 and a in out2[b] and a < b:
                s1, s2 = out2[a][b], out2[b][a]
                if s1 == 0 or s2 == 0:
                    punk += 1
                elif s1 * s2 > 0:
                    ppos += 1
                else:
                    pneg += 1
    say(f"     of the signed 2-cycles: {ppos:,} POSITIVE feedback, {pneg:,} NEGATIVE "
        f"({punk:,} with an unsigned edge)")
    b5 = bool(np.isfinite(z5) and z5 > Z_BAR)
    GG.verdict(b5, emit=say,
               if_true="B5 PASS -- mutual regulation is far commoner than the degree sequence "
                       "predicts",
               if_false=f"B5 FAIL -- z {z5:+.1f}; two-cycles are at chance")

    # ---- B6 ------------------------------------------------------------------------------------
    say()
    say("B6 AUTOREGULATION AGAINST CHANCE")
    z6 = zs["curated"]["self_loop"]["z"]
    n_reg = obs["curated"]["n_reg"]
    say(f"     curated self-loops {obs['curated']['self_loop']} over {n_reg:,} regulators "
        f"({obs['curated']['self_loop']/max(n_reg,1):.1%})")
    say(f"     degree-preserving null {zs['curated']['self_loop']['null_mean']:.2f} "
        f"+/- {zs['curated']['self_loop']['null_sd']:.2f}  ->  z {z6:+.1f}")
    say("     loop 175 reported this as '3.0%, against the plan's expected 50%' without asking "
        "what chance gives")
    b6 = bool(np.isfinite(z6) and z6 > Z_BAR)
    GG.verdict(b6, emit=say,
               if_true=f"B6 PASS -- self-regulation is {obs['curated']['self_loop'] / max(zs['curated']['self_loop']['null_mean'], 1e-9):.0f}x "
                       f"the chance rate. Loop 175's 3.0% was a small FRACTION and a large "
                       f"ENRICHMENT, and reporting only the fraction understated it",
               if_false=f"B6 FAIL -- z {z6:+.1f}; the self-loops are what chance gives and loop "
                        f"175's framing stands")

    say()
    say("B7 WHAT THIS CANNOT SHOW")
    say("     Nothing here is functional. Whether a coherent feedforward loop actually delays a")
    say("     response, or a negative feedback actually speeds one, needs perturbation time-course")
    say("     data that is not on this disk, and this loop does not gesture at those claims.")
    say("     CollecTRI is literature-curated, so its edges are enriched for the interactions")
    say("     people looked for. A motif can be over-represented because it is over-studied, and")
    say("     no degree-preserving null can see that.")
    say("     The double-edge swap preserves in-degree and out-degree exactly but not higher-order")
    say("     structure, so an enrichment could still reflect modularity rather than circuitry.")
    say("     Signs are Activation/Repression/Unknown as curated; a context-dependent factor that")
    say("     activates in one setting and represses in another is recorded as one or the other.")
    say("     B3 compares two z-scores from two DIFFERENT graphs. Equalising the nulls removes one")
    say("     confound and not the others: the tiers differ in density, in how signs were assigned")
    say("     and in what an edge means, and a z-score is not a quantity those differences leave")
    say("     alone. B3 is a direction-of-difference test and is not read as a ratio.")
    b7 = True
    say(f"     B7 {'PASS' if b7 else 'FAIL'}")

    gates = {"B1": b1, "B2": b2, "B3": b3, "B4": b4, "B5": b5, "B6": b6, "B7": b7}
    man = RM.manifest(inputs=[BUNDLE],
                      available=len(reg), used=len(tier["curated"]) + len(tier["binding"]),
                      selection="curated and binding tiers, kept separate", seed=SEED,
                      controls=[f"{N_RAND} degree-preserving double-edge-swap randomisations, "
                                f"the same null for both tiers",
                                "the matrix counter checked against the nested-loop reference "
                                "on three graphs before any gate is scored",
                                "the curated tier required to out-enrich the occupancy tier",
                                "a sign-shuffled null on the identical topology for coherence"],
                      note="feedforward and feedback motif enrichment in the K562 TF network")
    out = dict(test="network motifs", gates=gates,
               tiers={t: dict(edges=len(tier[t]),
                              regulators=len({a for a, b, s in tier[t]}),
                              signed=sum(1 for a, b, s in tier[t] if s != 0)) for t in tier},
               observed={t: obs[t] for t in obs},
               z={t: zs[t] for t in zs},
               coherence=dict(coherent=coh, incoherent=inc, unsigned=uns, fraction=frac,
                              null_mean=float(null_fr.mean()), null_sd=float(null_fr.std(ddof=1)),
                              z=float(z4)),
               feedback_signs=dict(positive=ppos, negative=pneg, unsigned=punk),
               manifest=man, seconds=time.time() - t0, log=log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'PASS' if v else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [{time.time()-t0:.0f}s]")
    say("=" * 104)
    out["log"] = log
    json.dump(out, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
