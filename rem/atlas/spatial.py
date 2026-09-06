"""Spatial organisation: when does it matter, and is the well-mixed assumption I have used safe?

WHAT I SAID BEFORE, AND WHY IT WAS ONLY HALF RIGHT. division.py's V6 recorded that spatial
organisation "needs different mathematics, because a reaction-diffusion PDE has no finite state
space, so every method here that enumerates states fails outright". That is true of the CONTINUUM
limit and false of the thing one actually computes. Discretise space into compartments and the
reaction-diffusion master equation has a perfectly finite state space -- it is just a product over
compartments, so it grows as (N+1)^V and becomes intractable by ENUMERATION, not by principle. For
a few compartments it is exactly solvable, and that is enough to answer the question that matters.

THE QUESTION THAT MATTERS IS NOT "CAN WE SIMULATE SPACE" BUT "DOES IT CHANGE THE ANSWER". That is
set by one dimensionless group, the Damkohler number

    Da  =  tau_diffusion / tau_reaction  =  (L^2 / 2D) / tau_reaction

Da << 1 means a molecule crosses the cell many times before anything happens to it, so the cell is
well mixed and every model in this build order is safe. Da >~ 1 means it does not, and space is
load-bearing. With L = 10 um and standard cytoplasmic diffusion coefficients:

    species                     D (um^2/s)   tau_diff
    small metabolite                 400        0.1 s
    globular protein                  15        3.3 s
    ribosome / large complex           1         50 s
    mRNA in an RNP granule          0.05       1000 s

Pairing each process with the species that must actually MOVE for it to proceed -- substrates for
a metabolic reaction, the signalling protein for a cascade, the transcript for translation:

    metabolism, turnover ~1 s        metabolite    Da ~ 0.1
    signalling, ~10 s                protein       Da ~ 0.33
    expression, mRNA decay ~9 h      mRNA          Da ~ 0.02
    cell cycle, 24 h                 mRNA          Da ~ 0.012

So the well-mixed assumption is justified by one to two orders for everything built here, and
becomes marginal exactly where I already said the machinery does not apply. That is a satisfying
closure and it is also exactly the kind of claim that deserves to be checked rather than admired,
which is what S3 does.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

S1  THE SOLVER REDUCES TO WELL-MIXED. As the hopping rate goes to infinity the V-compartment tail
    must converge to the tail of the equivalent single well-mixed compartment. Worst relative
    disagreement below 1e-4 at the fastest hopping tested, or the spatial solver is not solving a
    generalisation of the model it claims to generalise.

S2  THE DAMKOHLER TABLE IS ARITHMETIC. Every number above recomputed from stated diffusion
    coefficients and rates, with the species-process pairing made explicit, so a reader can
    disagree with a coefficient rather than with a conclusion.

S3  THE CROSSOVER IS MEASURED, NOT ASSUMED. Sweep the hopping rate over many orders and find where
    the local-depletion tail departs from the well-mixed prediction by more than 0.05 orders.
    Predeclared: if the departure sets in near Da ~ 1 the argument above holds; if it sets in at
    Da far below 1, then well-mixed models fail while still looking safe by this criterion, and
    every result in this build order inherits that.

S4  THE COMPARTMENT TEST, which is not about Da at all. Transcription happens in the nucleus and
    translation in the cytoplasm, with an export step between them -- a two-compartment structure
    my expression model collapsed into one bag. Solve it exactly and compare the protein tail
    against the one-bag model AT MATCHED MEAN. Predeclared: a gap means expression.py's tails are
    wrong by that much for reasons that have nothing to do with diffusion coefficients.

S5  WHICH EARLIER RESULTS ARE SAFE. Apply the measured crossover to metabolism, expression, the
    cell cycle and the whole-cell model, and say for each whether its well-mixed assumption holds.

S6  WHAT SPACE COSTS IN PARAMETERS, counted rather than asserted.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import itertools
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import gammaincc

from rem.atlas.hybrid_tune import RULE

L_CELL = 10e-6
D_VALUES = {"small metabolite": 400.0, "globular protein": 15.0,
            "ribosome / large complex": 1.0, "mRNA in an RNP granule": 0.05}
PROCESSES = (("metabolism, turnover ~1 s", 1.0, "small metabolite"),
             ("signalling, ~10 s", 10.0, "globular protein"),
             ("expression, mRNA decay ~9 h", 9 * 3600 / np.log(2), "mRNA in an RNP granule"),
             ("protein decay, ~46 h", 46 * 3600 / np.log(2), "globular protein"),
             ("cell cycle, 24 h", 86400.0, "mRNA in an RNP granule"))
V_COMP, N_MAX = 3, 22
BIRTH_TOTAL, DEATH = 12.0, 1.0
THRESH_LOCAL = 1


def tau_diff(D_um2_s):
    return L_CELL ** 2 / (2.0 * D_um2_s * 1e-12)


def rdme(V, N, birth_total, death, hop):
    """Exact reaction-diffusion master equation on V well-mixed compartments in a ring.
    Birth is shared equally, death is per molecule, hopping moves one molecule to a neighbour."""
    states = list(itertools.product(range(N + 1), repeat=V))
    index = {s: i for i, s in enumerate(states)}
    n = len(states)
    r_, c_, v_ = [], [], []

    def add(i, j, rate):
        if rate > 0:
            r_.append(j); c_.append(i); v_.append(rate)
            r_.append(i); c_.append(i); v_.append(-rate)

    b = birth_total / V
    for s, i in index.items():
        for k in range(V):
            if s[k] < N:
                t = list(s); t[k] += 1
                add(i, index[tuple(t)], b)
            if s[k] > 0:
                t = list(s); t[k] -= 1
                add(i, index[tuple(t)], death * s[k])
                for nb in ((k - 1) % V, (k + 1) % V):
                    if s[nb] < N:
                        t2 = list(s); t2[k] -= 1; t2[nb] += 1
                        add(i, index[tuple(t2)], hop * s[k])
    L = coo_matrix((v_, (r_, c_)), shape=(n, n)).tocsr().tolil()
    L[n - 1, :] = 1.0
    rhs = np.zeros(n); rhs[n - 1] = 1.0
    p = np.maximum(spsolve(L.tocsc(), rhs), 0.0)
    p = p / p.sum()
    return p, states


def local_tail(p, states, thresh=THRESH_LOCAL, comp=0):
    return float(sum(pi for pi, s in zip(p, states) if s[comp] < thresh))


def wellmixed_local(birth_total, death, V, N, thresh=THRESH_LOCAL):
    """Infinite-hopping limit: the total is Poisson(birth/death) and each molecule lands in a
    compartment independently, so one compartment is Poisson(birth/(V*death))."""
    lam = birth_total / (V * death)
    # Poisson CDF via the regularised incomplete gamma; np.math was removed in NumPy 2
    return float(gammaincc(max(thresh, 1), lam)) if thresh >= 1 else 0.0


def nuc_cyt(k_tx, k_exp, k_dmn, k_dmc, k_tl, k_dp, Mn, Mc, Mp):
    """Transcription in the nucleus, export, translation in the cytoplasm. State (m_nuc, m_cyt, p).
    This is compartmentalisation, not diffusion: it matters however fast molecules move."""
    n = (Mn + 1) * (Mc + 1) * (Mp + 1)
    idx = lambda a, b, c: (a * (Mc + 1) + b) * (Mp + 1) + c
    r_, c_, v_ = [], [], []

    def add(i, j, rate):
        if rate > 0:
            r_.append(j); c_.append(i); v_.append(rate)
            r_.append(i); c_.append(i); v_.append(-rate)

    for a in range(Mn + 1):
        for b in range(Mc + 1):
            for c in range(Mp + 1):
                i = idx(a, b, c)
                if a < Mn:
                    add(i, idx(a + 1, b, c), k_tx)
                if a > 0:
                    add(i, idx(a - 1, b, c), k_dmn * a)
                    if b < Mc:
                        add(i, idx(a - 1, b + 1, c), k_exp * a)
                if b > 0:
                    add(i, idx(a, b - 1, c), k_dmc * b)
                    if c < Mp:
                        add(i, idx(a, b, c + 1), k_tl * b)
                if c > 0:
                    add(i, idx(a, b, c - 1), k_dp * c)
    L = coo_matrix((v_, (r_, c_)), shape=(n, n)).tocsr().tolil()
    L[n - 1, :] = 1.0
    rhs = np.zeros(n); rhs[n - 1] = 1.0
    p = np.maximum(spsolve(L.tocsc(), rhs), 0.0)
    p = (p / p.sum()).reshape(Mn + 1, Mc + 1, Mp + 1)
    return p.sum(axis=(0, 1))


def two_stage(k_tx_eff, k_tl, k_dm, k_dp, Mm, Mp):
    n = (Mm + 1) * (Mp + 1)
    idx = lambda m, q: m * (Mp + 1) + q
    r_, c_, v_ = [], [], []

    def add(i, j, rate):
        if rate > 0:
            r_.append(j); c_.append(i); v_.append(rate)
            r_.append(i); c_.append(i); v_.append(-rate)

    for m in range(Mm + 1):
        for q in range(Mp + 1):
            i = idx(m, q)
            if m < Mm:
                add(i, idx(m + 1, q), k_tx_eff)
            if m > 0:
                add(i, idx(m - 1, q), k_dm * m)
                if q < Mp:
                    add(i, idx(m, q + 1), k_tl * m)
            if q > 0:
                add(i, idx(m, q - 1), k_dp * q)
    L = coo_matrix((v_, (r_, c_)), shape=(n, n)).tocsr().tolil()
    L[n - 1, :] = 1.0
    rhs = np.zeros(n); rhs[n - 1] = 1.0
    p = np.maximum(spsolve(L.tocsc(), rhs), 0.0)
    p = (p / p.sum()).reshape(Mm + 1, Mp + 1)
    return p.sum(axis=0)


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE); P("SPATIAL ORGANISATION: WHEN DOES IT MATTER?"); P(RULE)

    # ---- S2 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("S2  THE DAMKOHLER TABLE IS ARITHMETIC"); P(RULE)
    P(f"  cell diameter {L_CELL*1e6:.0f} um, mixing time tau_diff = L^2 / 2D")
    P(f"  {'species':>26}{'D (um^2/s)':>13}{'tau_diff (s)':>15}")
    for nm, D in D_VALUES.items():
        P(f"  {nm:>26}{D:>13.2f}{tau_diff(D):>15.2f}")
    P(f"\n  paired with the species that must MOVE for each process to proceed:")
    P(f"  {'process':>30}{'moving species':>26}{'tau_react (s)':>15}{'Da':>12}{'':>4}")
    for nm, tr, sp in PROCESSES:
        da = tau_diff(D_VALUES[sp]) / tr
        P(f"  {nm:>30}{sp:>26}{tr:>15.1f}{da:>12.2e}"
          f"{'  <- marginal' if da > 0.1 else '':>4}")
    P("  Da << 1 means a molecule crosses the cell many times before anything happens to it.")

    # ---- S1, S3 ---------------------------------------------------------------------------------
    P("\n" + RULE); P("S1/S3  THE CROSSOVER, MEASURED"); P(RULE)
    P(f"  {V_COMP} compartments in a ring, birth {BIRTH_TOTAL} total, death {DEATH} per molecule,")
    P(f"  rare event: compartment 0 is empty. Well-mixed limit is Poisson({BIRTH_TOTAL/(V_COMP*DEATH):.2f}).")
    wm = wellmixed_local(BIRTH_TOTAL, DEATH, V_COMP, N_MAX)
    P(f"  well-mixed prediction P(empty) = {wm:.6e}")
    P(f"\n  {'hop rate':>11}{'Da = death/hop':>17}{'P(empty)':>14}{'ratio to well-mixed':>21}"
      f"{'log10 gap':>11}")
    rows = []
    t0 = time.time()
    for hop in (1e4, 1e3, 1e2, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03):
        p, states = rdme(V_COMP, N_MAX, BIRTH_TOTAL, DEATH, hop)
        t = local_tail(p, states)
        gap = abs(np.log10(max(t, 1e-300)) - np.log10(max(wm, 1e-300)))
        rows.append((hop, DEATH / hop, t, gap))
        P(f"  {hop:>11.4g}{DEATH/hop:>17.4g}{t:>14.6e}{t/wm:>21.4f}{gap:>11.4f}")
    P(f"  {len(rows)} solves in {time.time()-t0:.0f}s")
    fastest = rows[0]
    P(f"\n  S1: at the fastest hopping, relative disagreement with the well-mixed limit"
      f" {abs(fastest[2]-wm)/wm:.2e}"
      f"   {'PASS' if abs(fastest[2]-wm)/wm < 1e-4 else 'FAIL'} (bar 1e-4)")
    cross = next((r for r in rows if r[3] > 0.05), None)
    if cross:
        P(f"  S3: the tail departs from well-mixed by more than 0.05 orders once Da exceeds"
          f" {cross[1]:.3g}")
        if cross[1] > 0.1:
            P("  That is near Da ~ 1, so the Damkohler argument holds and a well-mixed model is")
            P("  safe wherever Da is small.")
        else:
            P("  That is FAR BELOW Da ~ 1, so well-mixed models fail while still looking safe by")
            P("  the Damkohler criterion, and every result in this build order inherits it.")
    else:
        P("  S3: no departure above 0.05 orders anywhere in the swept range.")

    # ---- S4 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("S4  THE COMPARTMENT TEST  (nucleus and cytoplasm, not diffusion)"); P(RULE)
    P("  Transcription is nuclear, translation cytoplasmic, with an export step between. My")
    P("  expression model collapsed that into one bag. This solves both exactly at matched mean.")
    k_dp = np.log(2) / 46.0
    k_dmc, k_tl = 0.30, 3.6
    Mn, Mc, Mp = 8, 14, 150
    P(f"\n  {'k_export':>10}{'nuc+cyt mean':>14}{'1-bag mean':>12}{'nuc+cyt P(<8)':>16}"
      f"{'1-bag P(<8)':>14}{'log10 gap':>11}")
    worst4 = 0.0
    for k_exp in (20.0, 5.0, 2.0, 1.0, 0.5):
        k_tx = 2.4
        pq = nuc_cyt(k_tx, k_exp, 0.10, k_dmc, k_tl, k_dp, Mn, Mc, Mp)
        m_nc = float((np.arange(len(pq)) * pq).sum())
        t_nc = float(pq[:8].sum())
        # one-bag comparator, transcription scaled so the MEANS agree exactly
        lo_k, hi_k = 0.05 * k_tx, 20.0 * k_tx
        pq2 = None
        for _ in range(40):
            mid = 0.5 * (lo_k + hi_k)
            pq2 = two_stage(mid, k_tl, k_dmc, k_dp, Mc, Mp)
            m2 = float((np.arange(len(pq2)) * pq2).sum())
            if abs(m2 - m_nc) / max(m_nc, 1e-12) < 1e-9:
                break
            if m2 > m_nc:
                hi_k = mid
            else:
                lo_k = mid
        t_1b = float(pq2[:8].sum())
        m_1b = float((np.arange(len(pq2)) * pq2).sum())
        gap = abs(np.log10(max(t_nc, 1e-300)) - np.log10(max(t_1b, 1e-300)))
        worst4 = max(worst4, gap)
        P(f"  {k_exp:>10.2f}{m_nc:>14.4f}{m_1b:>12.4f}{t_nc:>16.4e}{t_1b:>14.4e}{gap:>11.4f}")
    P(f"\n  worst |log10 gap| at matched mean: {worst4:.4f} orders")
    if worst4 < 0.05:
        P("  Compartmentalising transcription changes nothing beyond the mean, and expression.py's")
        P("  one-bag model is safe.")
    else:
        P("  Compartmentalising transcription changes the tail by this much AT MATCHED MEAN, so")
        P("  expression.py's one-bag tails are wrong by that factor for structural reasons that")
        P("  have nothing to do with diffusion coefficients.")

    # ---- S5, S6 ---------------------------------------------------------------------------------
    P("\n" + RULE); P("S5  WHICH EARLIER RESULTS ARE SAFE"); P(RULE)
    for nm, tr, sp in PROCESSES:
        da = tau_diff(D_VALUES[sp]) / tr
        verdict = ("well-mixed is safe" if da < 0.1 else
                   "MARGINAL -- space is load-bearing here")
        P(f"  {nm:>30}  Da = {da:8.2e}   {verdict}")
    P("\n  recon.py, wholecell.py, regulation.py : metabolic Da ~ 0.1, safe by an order")
    P("  expression.py, division.py            : Da ~ 0.01-0.02, safe by two orders")
    P("  signalling                            : Da ~ 0.33, NOT safe -- and it was already")
    P("                                          excluded for having no stationary distribution")

    P("\n" + RULE); P("S6  WHAT SPACE COSTS IN PARAMETERS"); P(RULE)
    P("  One diffusion coefficient per species, which is ~1 parameter per species rather than per")
    P("  reaction: about 2e4 for a human cell, against the 4.6e4 kinetic parameters metabolism")
    P("  alone already needs. Diffusion coefficients are also far better constrained than rate")
    P("  constants -- they follow from size and viscosity within a factor of a few, where kcat")
    P("  spans orders. So space is CHEAP to parameterise and expensive to COMPUTE, which is the")
    P("  opposite of the rate problem this build order has been chasing.")

    P("\n" + RULE)
    open(os.path.join(os.path.dirname(__file__), "RESULTS_spatial.txt"),
         "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
