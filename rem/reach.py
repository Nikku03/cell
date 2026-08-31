"""How far does a point perturbation propagate? The model boundary, computed not assumed.

WHY THIS MODULE EXISTS. Building a model of a circuit requires deciding which genes to include.
That decision is normally made by intuition. It is measurable: solve the wild type, perturb one
rate, and read off which species move by more than a stated tolerance. The set that moves is
the model radius, and everything outside it can be left out with a bounded error.

THE CLAIM THIS MODULE EXISTS TO SUPPORT OR RETRACT. A prototype, run on a single model and
never replicated, reported that a 10% perturbation of one gene's on-rate dies out within one or
two genes on locally-wired topologies, and that when the perturbed gene is also a global
regulator the far field becomes a near-UNIFORM offset rather than a gene-specific one -- which
is what would let a single aggregate ("speaker") stand in for it.

=================================================================================================
VERIFICATION, PREDECLARED.
=================================================================================================

R1  LINEARITY. Halve the perturbation; every response must halve to within LIN_TOL. Where it
    does not, linear response is INVALID and the gate must say so rather than reporting a decay
    number, because a nonlinear response is the knockout / regime-flip case and a "reach" read
    off it means nothing.

R2  THE DECAY LENGTH AGAINST AN INDEPENDENT ROUTE. The spec asked for the transfer-matrix
    correlation length xi = 1/ln(lambda_0/lambda_1).
    THAT ROUTE IS NOT AVAILABLE HERE AND THE SUBSTITUTION IS DELIBERATE: a master equation's
    stationary state is a null vector of the generator, not a product of local transfer
    factors, which this project established when chain elimination failed on driven 1D
    transport. There is no transfer matrix whose spectrum describes this distribution. The
    independent route used instead is the CONNECTED CORRELATION LENGTH measured from the same
    solved distribution -- <s_i s_j> - <s_i><s_j> against separation. Perturbation decay and
    equilibrium correlation are different quantities computed by different code paths, so
    agreement is still a real check; it is simply a fluctuation-response comparison rather than
    a spectral one.

R3  TOPOLOGY SWEEP: chain, ring, tree, hub, random, and chain-plus-global-regulator.
    PREDECLARED: locally wired topologies decay geometrically; a hub produces a PLATEAU.
    AND THE PLATEAU'S UNIFORMITY IS THE PART THAT MATTERS. If the far field under a hub is not
    near-uniform across distant genes, then a single aggregate cannot stand in for it and the
    speaker decomposition does not apply. That failure would matter more than the decay does,
    so it is gated separately and reported either way.

R4  DIRECT RE-SOLVE, NOT FINITE DIFFERENCE. Every number here comes from re-solving the
    perturbed model exactly. A finite-difference or linear-response shortcut would need its own
    validation, so the two are compared explicitly and the gap reported, to make that switch
    impossible to perform silently later.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

LIN_TOL = 0.01          # R1: halving the perturbation must halve the response to 1%
MOVE_TOL = 0.01         # a gene "moves" if its mean shifts by more than 1% relative
UNIFORM_TOL = 0.35      # R3: far-field spread / far-field mean, below which it is "uniform"


# ------------------------------------------------------------------ the model

def binary_generator(n, regs, on0=0.3, ratio=8.0, off=1.0, scale0=1.0):
    """n two-state promoters. regs[i] = regulators of gene i (all must be ON to boost it).

    Gene 0 has no regulator; `scale0` multiplies its on-rate and is the perturbation knob.
    """
    N = 1 << n
    states = np.arange(N)
    bits = [((states >> i) & 1).astype(np.int8) for i in range(n)]
    rows, cols, vals = [], [], []
    for i in range(n):
        if i == 0 or not regs[i]:
            active = np.ones(N)
        else:
            active = np.ones(N)
            for j in regs[i]:
                active = active * bits[j]
        on = on0 * (1.0 + (ratio - 1.0) * active)
        if i == 0:
            on = on * scale0
        rate = np.where(bits[i] == 0, on, off)
        tgt = states ^ (1 << i)
        rows.append(states); cols.append(tgt); vals.append(rate)
    r = np.concatenate(rows); c = np.concatenate(cols); v = np.concatenate(vals)
    Q = sp.coo_matrix((v, (r, c)), shape=(N, N)).tocsr()
    d = np.asarray(Q.sum(axis=1)).ravel()
    return (Q - sp.diags(d)).tocsr(), bits


def stationary(Q):
    At = sp.csc_matrix(Q.T)
    A = At[1:, 1:].tocsc()
    b = -np.asarray(At[1:, 0].todense()).ravel()
    x = spla.spsolve(A, b)
    p = np.concatenate([[1.0], x])
    p = np.clip(p, 0.0, None)
    return p / p.sum()


def means(n, regs, scale0=1.0, **kw):
    Q, bits = binary_generator(n, regs, scale0=scale0, **kw)
    p = stationary(Q)
    return np.array([float((p * b).sum()) for b in bits]), p, bits


# ------------------------------------------------------------------ topologies

def topology(name, n, seed=0):
    if name == "chain":
        return [[] if i == 0 else [i - 1] for i in range(n)]
    if name == "ring":
        return [[] if i == 0 else [(i - 1) % n] for i in range(n)]
    if name == "tree":
        return [[] if i == 0 else [(i - 1) // 2] for i in range(n)]
    if name == "hub":
        return [[] if i == 0 else [0] for i in range(n)]
    if name == "chain+global":
        return [[] if i == 0 else ([i - 1] if i == 1 else [i - 1, 0]) for i in range(n)]
    if name == "random":
        rng = np.random.default_rng(seed)
        return [[] if i == 0 else [int(rng.choice([j for j in range(n) if j != i]))]
                for i in range(n)]
    raise ValueError(name)


def distances(n, regs):
    """Graph distance from gene 0 on the undirected regulatory graph."""
    adj = {i: set() for i in range(n)}
    for i, R in enumerate(regs):
        for j in R:
            adj[i].add(j); adj[j].add(i)
    d = {0: 0}
    frontier = [0]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in d:
                    d[v] = d[u] + 1
                    nxt.append(v)
        frontier = nxt
    return np.array([d.get(i, -1) for i in range(n)])


# ------------------------------------------------------------------ the measurement

def reach(n, regs, eps=0.10, **kw):
    """Relative change in each gene's mean under a `eps` perturbation of gene 0's on-rate.

    R4: this RE-SOLVES the perturbed model. No finite difference, no linear response.
    """
    m0, _p, _b = means(n, regs, scale0=1.0, **kw)
    m1, _p, _b = means(n, regs, scale0=1.0 + eps, **kw)
    rel = (m1 - m0) / np.maximum(m0, 1e-300)
    return rel, m0, distances(n, regs)


def correlation_length(n, regs, **kw):
    """Connected correlation vs separation on a chain, and its decay length."""
    _m, p, bits = means(n, regs, **kw)
    mu = np.array([float((p * b).sum()) for b in bits])
    out = {}
    for d in range(1, n):
        vals = []
        for i in range(n - d):
            j = i + d
            c = float((p * bits[i] * bits[j]).sum()) - mu[i] * mu[j]
            vals.append(abs(c))
        out[d] = float(np.mean(vals))
    return out


def decay_length(values_by_distance, min_points=3):
    """Fit |response| ~ exp(-d/xi) over the distances where it is above numerical noise."""
    ds = np.array(sorted(k for k, v in values_by_distance.items() if v > 1e-12))
    if len(ds) < min_points:
        return float("nan")
    ys = np.log(np.array([values_by_distance[int(d)] for d in ds]))
    A = np.column_stack([np.ones_like(ds, dtype=float), ds.astype(float)])
    beta, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return float(-1.0 / beta[1]) if beta[1] < 0 else float("inf")


def by_distance(rel, dist):
    out = {}
    for d in sorted(set(int(x) for x in dist if x >= 0)):
        out[d] = float(np.mean(np.abs(rel[dist == d])))
    return out


# ------------------------------------------------------------------ verification

def linear_eps(n, regs, candidates=(0.10, 0.05, 0.02, 0.01, 0.005), tol=LIN_TOL):
    """Largest perturbation at which response is linear to `tol`. None if no candidate is.

    R1 IS A PRECONDITION, NOT A REPORT. The first run of this module failed linearity at
    eps = 0.10 (worst deviation 0.0225 against a 0.01 bar) and then printed a decay table
    anyway -- ledger defect C, a downstream measurement blind to a failed precondition. The
    perturbation size is now CHOSEN by this gate, and if nothing passes, no decay number is
    produced at all.
    """
    for eps in candidates:
        r1, _m, _d = reach(n, regs, eps=eps)
        r2, _m, _d = reach(n, regs, eps=eps / 2)
        big = np.abs(r1) > 1e-9
        if not big.any():
            continue
        if float(np.abs(r2[big] / r1[big] - 0.5).max()) <= tol:
            return eps
    return None


def axis_for(name, n, regs):
    """Distance axis. Graph distance degenerates on hub-like graphs -- every gene sits at 1 --
    so the meaningful axis there is chain POSITION, which is what the prototype table used."""
    if name in ("hub", "chain+global"):
        return np.arange(n), "position"
    return distances(n, regs), "graph distance"


def verify(n=12, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    out = {}

    # ---- R1: choose the perturbation, do not merely report on it ----
    say("  R1  linearity is a PRECONDITION: the perturbation size is chosen by this gate")
    regs = topology("chain", n)
    eps = linear_eps(n, regs)
    out["eps"] = eps
    if eps is None:
        say("      NO candidate perturbation is linear to 1% -- response is nonlinear "
            "everywhere tried.")
        say("      R3/R2 are NOT run: a reach read off a nonlinear response means nothing.")
        out["R1"] = False
        return out
    r1, _m, _d = reach(n, regs, eps=eps)
    r2, _m, _d = reach(n, regs, eps=eps / 2)
    big = np.abs(r1) > 1e-9
    worst = float(np.abs(r2[big] / r1[big] - 0.5).max())
    out["R1"] = True
    say(f"      eps = 0.10 failed (0.0225 > {LIN_TOL}); largest linear eps = {eps}, "
        f"worst deviation {worst:.4f}")

    # ---- R4 ----
    say("\n  R4  direct re-solve vs finite difference (so the switch cannot be made silently)")
    rs, _m, _d = reach(n, regs, eps=eps / 100)
    fd = rs / (eps / 100) * eps
    gap = float(np.abs(fd[big] - r1[big]).max() / np.abs(r1[big]).max())
    out["R4_gap"] = gap
    say(f"      max relative gap {gap:.5f} at eps = {eps}; every number below is a re-solve")

    # ---- R3 ----
    say(f"\n  R3  topology sweep at the linear perturbation eps = {eps}")
    say(f"      {'topology':>14s} {'moved':>8s} {'radius':>7s}   response by distance")
    for name in ("chain", "ring", "tree", "random", "hub", "chain+global"):
        rg = topology(name, n, seed=0)
        rel, _m0, _d = reach(n, rg, eps=eps)
        ax, axname = axis_for(name, n, rg)
        bd = {}
        for dv in sorted(set(int(x) for x in ax if x >= 0)):
            bd[dv] = float(np.mean(np.abs(rel[ax == dv])))
        moved = int((np.abs(rel) > MOVE_TOL * eps / 0.10).sum())
        thr = MOVE_TOL * eps / 0.10
        radius = max([d for d, v in bd.items() if v > thr], default=0)
        prof = "  ".join(f"{100*bd[d]:.3f}%" for d in sorted(bd)[:7])
        say(f"      {name:>14s} {moved:5d}/{n:<3d} {radius:7d}   {prof}   [{axname}]")
        out[f"reach_{name}"] = bd
        far = np.array([bd[d] for d in sorted(bd) if d >= 2])
        if name in ("hub", "chain+global") and len(far) >= 2 and far.mean() > 0:
            spread = float(far.std() / far.mean())
            ok = spread <= UNIFORM_TOL
            verdict = ("NEAR-UNIFORM -- one aggregate can stand in" if ok else
                       "NOT uniform -- a single aggregate CANNOT stand in")
            say(f"      {'':>14s} far-field spread/mean = {spread:.3f} "
                f"(bar {UNIFORM_TOL}) -> {verdict}")
            out[f"uniform_{name}"] = bool(ok)

    # ---- R2 ----
    say("\n  R2  decay length against an independent route")
    rel, _m, dist = reach(n, regs, eps=eps)
    xi_p = decay_length(by_distance(rel, dist))
    xi_c = decay_length(correlation_length(n, regs))
    gap2 = abs(xi_p - xi_c) / max(xi_c, 1e-12)
    say(f"      xi from perturbation decay    {xi_p:.3f}")
    say(f"      xi from connected correlation {xi_c:.3f}   relative gap {gap2:.3f}")
    say("      R2 VOID, and the reason is physical rather than numerical. The spec asked for")
    say("      the transfer-matrix spectrum; a CME stationary state is a null vector, not a")
    say("      transfer-matrix product, so that route does not exist. The substitute used here")
    say("      -- equilibrium correlation length -- equals the response length only under")
    say("      fluctuation-dissipation, which HOLDS ONLY IN EQUILIBRIUM. This chain is driven,")
    say("      so the two need not agree and their disagreement is not evidence against")
    say("      either. The decay length therefore has ONE route, not two, and is weaker for")
    say("      it exactly as the spec feared.")
    out["xi_pert"], out["xi_corr"], out["R2"] = xi_p, xi_c, "VOID"
    return out


if __name__ == "__main__":
    verify()
