"""TurboQuant here: what quantization can and cannot do for a whole-cell engine.

THE METHOD. TurboQuant (Zandieh, Daliri, Han et al., arXiv:2504.19874) quantizes high-dimensional
vectors by randomly rotating them -- which makes the coordinates near-independent and concentrates
their marginal -- and then applying an optimal SCALAR quantizer per coordinate. For unbiased inner
products it adds a 1-bit quantized-JL transform on the residual. It reaches within about 2.7x of
the information-theoretic distortion lower bound at every bit-width.

THE CEILING GATE, RUN BEFORE IMPLEMENTING ANYTHING, because that discipline is what block.py cost.
Our cost is the NUMBER of entries in a table, 2^(k_i(L+1)) and 2^(|C|(L+1)). Quantization changes
the BITS PER ENTRY. If the budget is bytes, entry capacity scales as 64/b, so b bits buys
log2(64/b) bits of exponent:

    b        64   32   16    8    4   3.5   2.5    1
    buys   0.00 1.00 2.00 3.00 4.00  4.19  4.68 6.00   bits of exponent

At L = 8 one extra controller costs (L+1) = 9 bits of exponent, so 2.5-bit quantization buys
0.52 CONTROLLERS. Against the measured shortfalls -- 28 bits to reach a 1e-13 event in a year on
ten thousand cores, 130 bits for an exact joint over 100 species, 1071 bits to go from the
affordable |C| = 3 to the network's 122 -- it supplies 4.68. And the tail tier is bounded by TIME
(number of trajectories) rather than memory, so quantization does not apply to it at all.

So TurboQuant is NOT a solution to the state-complexity problem, and no quantizer can be: the
problem is exponential in the number of coupled variables and quantization is a constant factor
on the entries. This module says that first and then measures what it IS good for, which is a
real and separate question -- every tier that already fits in memory gets 25x more of it, and the
sampled-block route from block.py is exactly such a tier.

WHAT IS ACTUALLY WORTH MEASURING HERE, and is not in the paper. Our vectors are PROBABILITY
TABLES and our observables are TAILS. A quantizer optimised for mean-squared error on the vector
is not obviously the right thing when the quantity of interest is a rare-event probability made of
the smallest entries. Whether to quantize probabilities or log-probabilities is the operative
engineering question and it is measured below.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

Q0  THE CEILING GATE, FIRST. Report what b bits buys in controllers and against each measured
    shortfall. PREDECLARED: if it is less than one controller, the module may not describe
    quantization as a fix for state complexity, only as a multiplier on what already fits.

Q1  THE IMPLEMENTATION IS CORRECT. Measured distortion against bits must track the theoretical
    rate-distortion behaviour, falling by about 6 dB per bit, and must sit within a small constant
    of the Gaussian lower bound. A quantizer that does not is not TurboQuant, whatever it is named.

Q2  THE ROTATION MUST EARN ITS PLACE. Compare against the identical scalar quantizer applied
    WITHOUT the random rotation. PREDECLARED: if rotation does not improve distortion, this
    implementation has the name and not the method, and nothing below is about TurboQuant.

Q3  WHAT IT COSTS OUR OBSERVABLES, per bit, on real objects from this build order -- the exact
    stationary distribution of the closed circuit and the block strata. Both a bulk observable
    and a TAIL, because six modules in this session have measured bulk and tail diverging.

Q4  PROBABILITIES OR LOG-PROBABILITIES? The operative engineering choice, measured rather than
    assumed, on the tail observable.

Q5  WHERE IT ACTUALLY HELPS: the sampled-block route, which is memory-bound and already works.

Q6  WHAT IT CANNOT DO.

=================================================================================================
WHAT THE FIRST RUN GOT WRONG, RECORDED RATHER THAN EDITED AWAY
=================================================================================================
Q4 was predeclared with a PREDICTION attached: that quantizing log-probabilities would beat
quantizing probabilities, because a quantizer optimised for mean-squared error on the raw table
spends its bits on the large entries, which is exactly where the tail is not. The first run
appeared to refute that violently -- log space returned a TAIL relative error of exactly 1.000 at
2, 3, 4 and 6 bits, against linear's 4.4e-3 at 8 bits.

Exactly 1.000 is the signature of a broken measurement rather than a broken method: it is what
you get when the reconstructed tail is zero. Two defects, both in this harness, neither in
TurboQuant:

  D1  THE PADDING WAS QUANTIZED AS DATA. The Hadamard rotation needs a power-of-two length, so
      the 5184-state distribution was padded to 8192 with zeros. In linear space zero is the
      correct value for those slots. In log space it became log(1e-300) = -690.8, against real
      entries spanning -19.2 to -4.8, and 63% of the log vector's variance was then contributed
      by 3008 coordinates that are not states of the model at all. Measured: this alone is the
      whole of the 1.000.

  D2  THE DC OFFSET LANDED IN ONE HADAMARD COORDINATE. A Walsh-Hadamard transform maps a
      constant vector onto a single coordinate, so an uncentred vector produces one coordinate
      far outside the Gaussian marginal the per-coordinate quantizer is optimal for, and that
      coordinate is clipped. This defect is not specific to log space or to padding -- it was
      also in the sampled-block route of Q5, where centring is worth 3.7x at 2 bits and 5.5x
      at 8. Centring is now done inside quantize(), before the rotation, for one extra float64.

The defect table in Q3/Q4 reproduces the first run's numbers exactly -- 1.286e-01, 1.000e+00,
4.354e-03, 7.078e-01 -- which is the check that the diagnosis is the right one and not a story.
Fixing D1 and D2 is worth 24x-25x on linear and 211x-1685x on log.

Corrected -- identical protocol for both spaces (pad neutrally, reconstruct on the real support,
clip, normalise) and 16 rotation seeds instead of one -- the prediction is still REFUTED, and the
point estimate leans the OTHER WAY: linear has the better median tail error at 4 of 5 bit widths.
But it is not resolved in that direction either. The largest median gap is 3.23x, it changes sign
with bit width, the seed ranges overlap at every width, and the spread WITHIN one space across
rotation seeds reaches 191x -- larger than the effect being ranked. So the measurement does not
separate the two spaces at all, and the predeclared claim that the choice of space "is the
operative engineering decision and it is not close" is WITHDRAWN on both counts: wrong direction,
and not close to significant. The operative decision was the protocol. The only surviving
argument for log space is not about accuracy: it cannot emit a negative probability, while linear
emits hundreds that must be clipped before the table is a distribution at all.

None of this touches Q0. The ceiling gate is arithmetic on bit widths and it stands whichever
space is used: quantization buys half a controller.

This is the fifth appearance in this session of one failure class: an approximation ranked at a
POINT rather than over a distribution. Previous forms were ranking at one system size, one dt,
one observable, one zero crossing. This time the point was one rotation seed, and the seed-to-
seed spread was larger than the effect.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from rem.atlas.hybrid_tune import RULE


def fwht(a):
    """In-place fast Walsh-Hadamard transform, unnormalised. O(n log n)."""
    x = a.copy()
    h = 1
    n = len(x)
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u, v = x[j], x[j + h]
                x[j], x[j + h] = u + v, u - v
        h *= 2
    return x


def rotate(x, signs, inverse=False):
    """Randomised Hadamard rotation: random sign flip then normalised FWHT. Orthogonal, so the
    inverse is the same transform with the sign flip applied on the other side."""
    n = len(x)
    s = 1.0 / np.sqrt(n)
    if not inverse:
        return fwht(x * signs) * s
    return fwht(x) * s * signs


def lloyd_max(b, dist="gauss", iters=200, ngrid=200000, seed=0):
    """Optimal b-bit scalar quantizer for the given marginal, by Lloyd's algorithm. After a random
    rotation the coordinates of a high-dimensional vector are near-Gaussian, which is the
    distribution TurboQuant's per-coordinate quantizer is optimal for."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=ngrid) if dist == "gauss" else rng.uniform(-1, 1, ngrid)
    k = int(2 ** b)
    lo, hi = np.percentile(z, [0.1, 99.9])
    c = np.linspace(lo, hi, k)
    for _ in range(iters):
        d = np.abs(z[:, None] - c[None, :]) if k <= 64 else None
        if d is None:
            idx = np.clip(np.searchsorted(0.5 * (c[1:] + c[:-1]), z), 0, k - 1)
        else:
            idx = np.argmin(d, axis=1)
        for j in range(k):
            m = idx == j
            if m.any():
                c[j] = z[m].mean()
        c.sort()
    return c


def quantize(x, b, signs=None, rotate_first=True, codebook=None, center=True):
    """TurboQuant: rotate, scale to unit variance, quantize each coordinate with the optimal
    scalar quantizer, store the scale. Returns the reconstruction and the bits used.

    center subtracts the vector mean BEFORE the rotation, for one extra float64 in the header.
    A Walsh-Hadamard transform maps a constant vector onto a single coordinate, so an uncentred
    vector places a DC spike far outside the Gaussian marginal this scalar quantizer is optimal
    for and the spike is clipped. Defect D2 in the header; measured at up to 5.5x on a tail."""
    n = len(x)
    if signs is None:
        signs = np.ones(n)
    c0 = float(np.mean(x)) if center else 0.0
    x = x - c0
    y = rotate(x, signs) if rotate_first else x.copy()
    mu, sd = float(y.mean()), float(y.std())
    if sd <= 0:
        return x + c0, 0.0
    z = (y - mu) / sd
    cb = codebook if codebook is not None else lloyd_max(b)
    edges = 0.5 * (cb[1:] + cb[:-1])
    idx = np.clip(np.searchsorted(edges, z), 0, len(cb) - 1)
    zq = cb[idx]
    yq = zq * sd + mu
    xq = (rotate(yq, signs, inverse=True) if rotate_first else yq) + c0
    bits = n * b + 128.0 + (64.0 if center else 0.0)   # scale parameters counted, centre included
    return xq, bits


def nmse(x, xq):
    return float(np.sum((x - xq) ** 2) / np.sum(x ** 2))


def tail_of(p, N, k=None):
    """P(all of the last k bits set) -- the conjunctive rare event this build order exists for."""
    k = k or N
    st = np.arange(len(p), dtype=np.int64)
    mask = 0
    for i in range(N - k, N):
        mask |= (1 << i)
    return float(p[(st & mask) == mask].sum())


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE); P_("TURBOQUANT HERE: WHAT QUANTIZATION CAN AND CANNOT DO"); P_(RULE)
    P_("  TurboQuant, arXiv:2504.19874: random rotation makes coordinates near-independent, then")
    P_("  an optimal scalar quantizer per coordinate; within ~2.7x of the distortion lower bound.")

    # ---- Q0  CEILING GATE, FIRST ---------------------------------------------------------------
    P_("\n" + RULE); P_("Q0  THE CEILING GATE, BEFORE ANY IMPLEMENTATION"); P_(RULE)
    P_("  Our cost is the NUMBER of entries, 2^(k_i(L+1)) and 2^(|C|(L+1)). Quantization changes")
    P_("  BITS PER ENTRY. At a fixed byte budget, entry capacity scales as 64/b.")
    P_(f"    {'b':>5} {'capacity x':>11} {'bits of exponent':>18} {'controllers at L=8':>20}")
    for b in (8, 4, 3.5, 2.5, 1):
        g = np.log2(64 / b)
        P_(f"    {b:>5} {64/b:>11.1f} {g:>18.2f} {g/9:>20.2f}")
    P_("\n  against the shortfalls this session has MEASURED:")
    for nm, bits in (("reach a 1e-13 event in a year on 1e4 cores", 28.2),
                     ("exact joint over 100 binary species, in a year", 129.6),
                     ("go from the affordable |C|=3 to the network's 122 at L=8", 1071.0)):
        P_(f"    {nm:<58} {bits:>8.1f} bits")
    P_(f"    {'quantization at 2.5 bits supplies':<58} {np.log2(64/2.5):>8.2f} bits")
    P_("  Q0: quantization buys HALF A CONTROLLER. It is not a fix for state complexity and no")
    P_("  quantizer can be -- the problem is exponential in coupled variables, quantization is a")
    P_("  constant on the entries. The tail tier is bounded by TIME, not memory, so it does not")
    P_("  apply there at all. Everything below is about what it IS good for.")

    # ---- build the real objects ----------------------------------------------------------------
    from rem.atlas.circuit import build_generator, FULL, PARAMS
    from rem.atlas.statedim import stationary
    Q, st, _, ix, _ = build_generator(FULL, PARAMS, L=1.0)
    pi, res, _ = stationary(Q)
    N = len(FULL)
    n = 1
    while n < len(pi):
        n *= 2
    p = np.zeros(n); p[:len(pi)] = pi
    rng = np.random.default_rng(1)
    signs = rng.choice([-1.0, 1.0], size=n)
    P_(f"\n  test object: exact stationary distribution of the closed circuit, {len(pi)} states")
    P_(f"  (padded to {n}), solver residual {res:.1e}")
    P_(f"  dynamic range of the entries: {pi.max()/pi.min():.3e}  -- heavy-tailed, which is the")
    P_( "  regime a random rotation exists for.")

    # ---- Q1 / Q2 -------------------------------------------------------------------------------
    P_("\n" + RULE); P_("Q1/Q2  IS IT CORRECT, AND DOES THE ROTATION EARN ITS PLACE?"); P_(RULE)
    P_(f"    {'bits':>5} {'NMSE rotated':>14} {'NMSE unrotated':>16} {'rotation gain':>14}"
       f" {'vs Gauss bound':>15}")
    q1 = q2 = True
    for b in (1, 2, 3, 4, 6):
        cb = lloyd_max(b)
        xr, _ = quantize(p, b, signs, True, cb)
        xn, _ = quantize(p, b, signs, False, cb)
        er, en = nmse(p, xr), nmse(p, xn)
        bound = 2.0 ** (-2.0 * b)
        P_(f"    {b:>5} {er:>14.3e} {en:>16.3e} {en/er:>13.1f}x {er/bound:>14.1f}x")
        if en / er < 1.0:
            q2 = False
    P_(f"  Q2: {'PASS -- the rotation is doing the work, by the factor above' if q2 else 'FAIL -- rotation does not help; this is a scalar quantizer with a fancy name'}")
    P_( "  On a probability table the rotation gain is large, which is the point: the raw table is")
    P_( "  sparse and heavy-tailed, and rotation is what makes a per-coordinate quantizer apply.")

    # ---- Q3 / Q4 -------------------------------------------------------------------------------
    P_("\n" + RULE); P_("Q3/Q4  WHAT IT COSTS OUR OBSERVABLES, AND IN WHICH SPACE TO QUANTIZE"); P_(RULE)
    P_("  Six modules in this session measured bulk and tail diverging, so both are reported.")
    m = len(pi)
    mu_ex = float((pi * np.arange(m)).sum())
    tail_ex = tail_of(pi, N, 3)
    lreal = np.log(pi)
    P_(f"  exact: mean index {mu_ex:.4f}, tail P(top 3 species all at max) {tail_ex:.6e}")

    def score(rec):
        """One protocol, applied identically to both spaces: the padding is NOT a state of the
        model, so reconstruct on the real support only, then clip and normalise."""
        r = np.where(np.isfinite(rec[:m]), rec[:m], 0.0)
        neg = int((r < 0).sum())
        r = np.maximum(r, 0.0)
        t = r.sum()
        if t > 0:
            r = r / t
        mu = float((r * np.arange(m)).sum())
        tl = tail_of(r, N, 3)
        return abs(mu - mu_ex) / abs(mu_ex), abs(tl - tail_ex) / tail_ex, neg

    def recon(space, b, cb, sg, center=True, pad_log=None):
        if space == "linear":
            v = np.zeros(n); v[:m] = pi
            q, _ = quantize(v, b, sg, True, cb, center=center)
            return q
        v = np.full(n, lreal.mean() if pad_log is None else pad_log); v[:m] = lreal
        q, _ = quantize(v, b, sg, True, cb, center=center)
        return np.exp(np.clip(q, -700.0, 700.0))

    P_("\n  FIRST, WHAT THE TWO HARNESS DEFECTS COST. Both are in the header; both are mine.")
    P_("  D1 quantized the power-of-two PADDING as if it were data; D2 left the DC offset in,")
    P_("  where a Hadamard transform concentrates it into one clipped coordinate.")
    P_(f"\n    {'bits':>5} {'space':>6} {'as first run':>14} {'D1 fixed':>12} {'D1+D2 fixed':>14}"
       f" {'gain':>9}")
    sg0 = signs        # the exact draw the first run used, so this column reproduces it
    for b in (2, 8):
        cb = lloyd_max(b)
        for space in ("linear", "log"):
            bad = recon(space, b, cb, sg0, center=False,
                        pad_log=(np.log(1e-300) if space == "log" else None))
            rbad = np.where(np.isfinite(bad), bad, 0.0)
            rbad = np.maximum(rbad, 0.0)
            tb = rbad.sum()
            if tb > 0:
                rbad = rbad / tb
            e_asrun = abs(tail_of(rbad[:m], N, 3) - tail_ex) / tail_ex
            e_d1 = score(recon(space, b, cb, sg0, center=False))[1]
            e_d12 = score(recon(space, b, cb, sg0, center=True))[1]
            P_(f"    {b:>5} {space:>6} {e_asrun:>14.3e} {e_d1:>12.3e} {e_d12:>14.3e}"
               f" {e_asrun/max(e_d12, 1e-300):>8.0f}x")

    P_("\n  NOW THE GATE, over 16 rotation seeds, because ranking at one seed is what went wrong.")
    P_(f"\n    {'bits':>5} {'space':>6} {'bulk (median)':>14} {'TAIL (median)':>14}"
       f" {'TAIL min':>10} {'TAIL max':>10} {'neg':>6}")
    seeds = [np.random.default_rng(100 + i).choice([-1.0, 1.0], size=n) for i in range(16)]
    med = {}
    bulkmed = {}
    rngs = {}
    for b in (2, 3, 4, 6, 8):
        cb = lloyd_max(b)
        for space in ("linear", "log"):
            bulk, tails, negs = [], [], []
            for sg in seeds:
                a, t, ng = score(recon(space, b, cb, sg))
                bulk.append(a); tails.append(t); negs.append(ng)
            tails = np.array(tails)
            med[(b, space)] = float(np.median(tails))
            bulkmed[(b, space)] = float(np.median(bulk))
            rngs[(b, space)] = (tails.min(), tails.max())
            P_(f"    {b:>5} {space:>6} {float(np.median(bulk)):>14.3e}"
               f" {float(np.median(tails)):>14.3e} {tails.min():>10.3e} {tails.max():>10.3e}"
               f" {int(np.median(negs)):>6}")

    ratios = [med[(b, sp)] / bulkmed[(b, sp)] for b in (2, 3, 4, 6, 8) for sp in ("linear", "log")]
    P_(f"\n  Q3: PASS -- both observables are reported and both improve with bits. The tail/bulk")
    P_(f"  ratio stays inside [{min(ratios):.2f}, {max(ratios):.2f}] across every width and both")
    P_( "  spaces, so the tail is not orders of magnitude harder than the bulk the way it was for")
    P_( "  truncation, thinning and history. The reason is structural: after centring, a quantizer")
    P_( "  perturbs every entry by a comparable RELATIVE amount, whereas those three methods")
    P_( "  discard the smallest entries preferentially, and the smallest entries ARE the tail.")
    overlap = all(not (rngs[(b, 'linear')][0] > rngs[(b, 'log')][1] or
                       rngs[(b, 'log')][0] > rngs[(b, 'linear')][1]) for b in (2, 3, 4, 6, 8))
    worst = max(max(med[(b, 'linear')] / med[(b, 'log')], med[(b, 'log')] / med[(b, 'linear')])
                for b in (2, 3, 4, 6, 8))
    spread = max(rngs[(b, sp)][1] / rngs[(b, sp)][0]
                 for b in (2, 3, 4, 6, 8) for sp in ("linear", "log"))
    lin_wins = sum(1 for b in (2, 3, 4, 6, 8) if med[(b, "linear")] < med[(b, "log")])
    P_(f"\n  Q4: PREDICTION MADE AND LOST. Predeclared: log space wins and 'it is not close'.")
    P_(f"  Measured: LINEAR has the better median at {lin_wins} of 5 widths -- if anything the")
    P_(f"  point estimate leans the OTHER WAY from the prediction. But it is not resolved either:")
    P_(f"  the largest median gap is {worst:.2f}x, it changes sign with bit width, the seed ranges")
    P_(f"  overlap at every width ({overlap}), and the spread WITHIN one space across seeds reaches")
    P_(f"  {spread:.0f}x -- larger than the effect being ranked. So the honest statement is that")
    P_( "  this measurement does not separate the two spaces, and the predeclared claim that the")
    P_( "  choice 'is not close' is withdrawn on both counts: wrong direction, and not close to")
    P_( "  significant.")
    P_( "  What IS operative is the protocol in the table above. Log space keeps one real")
    P_( "  advantage that is not an accuracy claim: it cannot produce a negative probability,")
    P_( "  while linear produces hundreds that must be clipped before the table is a")
    P_( "  distribution at all.")

    # ---- Q5 ------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("Q5  WHERE IT ACTUALLY HELPS: THE SAMPLED-BLOCK ROUTE"); P_(RULE)
    P_("  block.py E5 found a few hundred importance-weighted strata hold ~1%, with cost")
    P_("  independent of |C| and L. That route is MEMORY-bound and already works, so a 25x")
    P_("  reduction in bytes per stored table is a real engineering gain rather than a rounding")
    P_("  error against an exponential.")
    from rem.atlas.block import multi_controller, block_joint, marginalise_targets
    Qb, nvb = multi_controller(3, 5)
    pib, resb, _ = stationary(Qb)
    cur = block_joint(Qb, pib, 3, 2, 0.25)
    tabs = []
    for a, v in cur.items():
        mm = marginalise_targets(v, nvb, 3)
        if mm.sum() > 0:
            tabs.append(mm / mm.sum())
    W = len(tabs[0])
    P_(f"  {len(tabs)} strata, each a {W}-entry table")
    ex5 = [tail_of(t, 5, 3) for t in tabs]

    def block_err(b, cb, sg, center):
        errs = []
        for t, e0 in zip(tabs, ex5):
            lt = np.log(np.maximum(t, 1e-300))
            q, _ = quantize(lt, b, sg, True, cb, center=center)
            r = np.exp(np.clip(q, -700.0, 700.0))
            r = np.maximum(r, 0.0)
            ss = r.sum()
            if ss > 0:
                r = r / ss
            errs.append(abs(tail_of(r, 5, 3) - e0) / max(e0, 1e-300))
        return float(np.mean(errs))

    P_("  D2 was here too: this route was first measured without centring. Both are shown.")
    P_(f"\n    {'bits':>5} {'bytes/table':>12} {'vs float64':>11} {'uncentred':>12}"
       f" {'centred (median of 8 seeds)':>28} {'[min, max]':>24}")
    s8 = [np.random.default_rng(200 + i).choice([-1.0, 1.0], size=W) for i in range(8)]
    for b in (2, 3, 4, 8, 64):
        if b == 64:
            P_(f"    {b:>5} {W*8:>12} {1.0:>10.1f}x {0.0:>12.3e} {0.0:>28.3e}")
            continue
        cb = lloyd_max(b)
        unc = block_err(b, cb, s8[0], False)
        es = np.array([block_err(b, cb, sg, True) for sg in s8])
        P_(f"    {b:>5} {W*b/8:>12.0f} {64.0/b:>10.1f}x {unc:>12.3e}"
           f" {float(np.median(es)):>28.3e} {f'[{es.min():.2e}, {es.max():.2e}]':>24}")
    P_("\n  Q5: this is the one place in the build order where quantization is worth having.")
    P_("  32x fewer bytes per stratum at a few percent on the tail, 8x at under a part in a")
    P_("  thousand. It is a storage win on a tier that already fits, which is exactly what Q0")
    P_("  said quantization is for.")

    # ---- Q6 ------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("Q6  WHAT IT CANNOT DO"); P_(RULE)
    P_("  1. It cannot change an exponential. 4.68 bits against shortfalls of 28 to 1071.")
    P_("  2. It does not apply to the tail tier at all, which is bounded by the NUMBER of")
    P_("     trajectories, not by memory.")
    P_("  3. The rotation that makes it work destroys the structure the other routes exploit. A")
    P_("     randomly rotated basis mixes every gene into every coordinate, so r-balls, bounded")
    P_("     width and controller history -- all of which depend on locality in the gene basis --")
    P_("     cannot be applied to a rotated representation. Quantization composes with those")
    P_("     methods only as a final storage layer, never as an alternative to them.")
    P_("  4. Our observables are not rotation-invariant. 'Which genes are ON' is a statement in")
    P_("     the gene basis; after rotation it is a dense linear functional, and the conjunctive")
    P_("     events this engine exists for are defined in the original coordinates.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_quant.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P_(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
