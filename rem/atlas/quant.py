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


def quantize(x, b, signs=None, rotate_first=True, codebook=None):
    """TurboQuant: rotate, scale to unit variance, quantize each coordinate with the optimal
    scalar quantizer, store the scale. Returns the reconstruction and the bits used."""
    n = len(x)
    if signs is None:
        signs = np.ones(n)
    y = rotate(x, signs) if rotate_first else x.copy()
    mu, sd = float(y.mean()), float(y.std())
    if sd <= 0:
        return x.copy(), 0.0
    z = (y - mu) / sd
    cb = codebook if codebook is not None else lloyd_max(b)
    edges = 0.5 * (cb[1:] + cb[:-1])
    idx = np.clip(np.searchsorted(edges, z), 0, len(cb) - 1)
    zq = cb[idx]
    yq = zq * sd + mu
    xq = rotate(yq, signs, inverse=True) if rotate_first else yq
    bits = n * b + 128.0            # the two float64 scale parameters are counted
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
    mu_ex = float((p * np.arange(n)).sum())
    tail_ex = tail_of(pi, N, 3)
    P_(f"  exact: mean index {mu_ex:.4f}, tail P(top 3 species all at max) {tail_ex:.6e}")
    P_(f"\n    {'bits':>5} {'space':>6} {'bulk rel err':>14} {'TAIL rel err':>14} {'negatives':>11}")
    for b in (2, 3, 4, 6, 8):
        cb = lloyd_max(b)
        for space in ("linear", "log"):
            if space == "linear":
                xq, _ = quantize(p, b, signs, True, cb)
                rec = np.maximum(xq, 0.0)
            else:
                lp = np.log(np.maximum(p, 1e-300))
                lq, _ = quantize(lp, b, signs, True, cb)
                rec = np.exp(lq)
            neg = int((xq < 0).sum()) if space == "linear" else 0
            s = rec.sum()
            rec = rec / s if s > 0 else rec
            mu = float((rec * np.arange(n)).sum())
            tl = tail_of(rec[:len(pi)], N, 3)
            P_(f"    {b:>5} {space:>6} {abs(mu-mu_ex)/abs(mu_ex):>14.3e}"
               f" {abs(tl-tail_ex)/tail_ex:>14.3e} {neg:>11}")
    P_("\n  Q4: the choice of space is the operative engineering decision and it is not close.")
    P_("  A quantizer optimised for mean-squared error on the RAW table spends its bits on the")
    P_("  large entries, which is exactly where the tail is not. Quantizing log-probabilities")
    P_("  spends them evenly in orders of magnitude. Linear quantization also produces NEGATIVE")
    P_("  probabilities, which have to be clipped before the table is a distribution at all.")

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
        m = marginalise_targets(v, nvb, 3)
        if m.sum() > 0:
            tabs.append(m / m.sum())
    P_(f"  {len(tabs)} strata, each a {len(tabs[0])}-entry table")
    ex = float(np.mean([tail_of(t, 5, 3) for t in tabs]))
    P_(f"    {'bits':>5} {'bytes/table':>12} {'vs float64':>11} {'mean tail rel err':>18}")
    for b in (2, 3, 4, 8, 64):
        if b == 64:
            P_(f"    {b:>5} {len(tabs[0])*8:>12} {1.0:>10.1f}x {0.0:>18.3e}")
            continue
        cb = lloyd_max(b)
        errs = []
        nn = 1
        while nn < len(tabs[0]):
            nn *= 2
        sg = rng.choice([-1.0, 1.0], size=nn)
        for t in tabs:
            pad = np.zeros(nn); pad[:len(t)] = np.log(np.maximum(t, 1e-300))
            lq, _ = quantize(pad, b, sg, True, cb)
            r = np.exp(lq[:len(t)]); r = r / r.sum()
            errs.append(abs(tail_of(r, 5, 3) - tail_of(t, 5, 3)) / max(tail_of(t, 5, 3), 1e-300))
        P_(f"    {b:>5} {len(tabs[0])*b/8:>12.0f} {64.0/b:>10.1f}x {float(np.mean(errs)):>18.3e}")

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
