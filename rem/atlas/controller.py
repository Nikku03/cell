"""Does closing the loop make the number of important parameters explode?

THE QUESTION, SHARPENED. Every result in this build order so far was measured on an OPEN-LOOP
model: flux balance has no controller, and regulation.py imposed enzyme programmes from outside
rather than letting the cell compute them. The sparsity those modules reported -- a handful of
rates matter, the rest do not -- might be an artefact of that. A real cell senses its own state
and adjusts. Feedback couples everything to everything, so the natural fear is that the important
set grows with the size of the network once the loop is closed.

There are three outcomes and they are genuinely different:

    EXPLOSION   the important set grows in proportion to the parameter count. Then no measurement
                programme can ever be finite and the whole targeted-measurement thesis dies.
    SPARSE      the important set saturates at a constant. Feedback adds parameters but not
                importance.
    TRANSFER    the important set stays the same SIZE but changes IDENTITY -- sensitivity moves
                off the plant and onto the controller. This is what control theory predicts and
                it is not the same claim as sparsity, because it says the rates worth measuring
                are the regulatory ones, not the metabolic ones.

Only the third is a positive statement about WHICH experiments to do, so the test has to be able
to tell it apart from the second rather than lumping both under "still sparse".

THE SYSTEM. A metabolic pathway that makes its own enzymes.

    plant       x_0 = S fixed, chain x_1 .. x_L, one reaction per step plus a demand reaction.
                v_i = e_i kcat_i (s/Km_i) / (1 + s/Km_i + p/Kp_i)   irreversible MM with
                competitive product inhibition, so every step feels the one after it.
    expression  de_j/dt = alpha_j h_j - delta_j e_j. The enzyme is a dynamical variable, not a
                parameter. This is what makes it a controller rather than a knob.
    control     h_j = prod over edges k into j of [(1 - w_k) + w_k f_k], with f_k a Hill
                repression 1/(1 + (x_s/Kr_k)^n_k). w_k = 0 recovers the open loop EXACTLY, so
                the open loop is a point in the same parameter space and not a separate model.

    wiring      the canonical physiological one: end-product repression. Edge k represses enzyme
                k mod (L+1) using the pathway end product, then the next metabolite in, and so on.

Two observables, because one of them is a trap:

    J = v_demand      the flux. What the pathway is for.
    E = sum_j e_j     the proteome spent making it. What the control costs.

The trap: end-product feedback exists precisely to hold the pathway output steady, so a
homeostatic observable's sensitivities shrink towards zero by design. A gate that celebrated
that would be passing on a saturated quantity, which is ledger entry U and has already been
made in this build order. C5 tests for it explicitly, and E is carried alongside J because when
control works, the variation does not vanish -- it moves into the cost.

SENSITIVITY. Fixed point F(q, theta) = 0 with q = (x, e), dim 2L+1. One adjoint solve gives
every parameter's gradient:  G^T lambda = (dT/dq)^T,  dT/dtheta = -lambda^T dF/dtheta. Both the
Jacobian G and the parameter block are taken by COMPLEX STEP, F(q + i*h*e_j)/h, which is exact
to machine precision and has no subtractive cancellation, so the step size cannot be tuned to
make a gate pass. Control coefficients are natural-log elasticities, C = (theta/T) dT/dtheta;
there is no log10 anywhere and therefore no ln10 factor to get wrong.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

C1  THE ADJOINT IS RIGHT. Every control coefficient is re-derived by central finite differences
    on the re-solved fixed point, in both directions. Bar: median relative error < 1e-6 and the
    worst < 1e-3 over all parameters. An adjoint that has never been finite-differenced has been
    wrong twice in this build order, once by exactly ln10.

C2  THE SUMMATION THEOREM. In the open loop h = 1, so e_j = alpha_j / delta_j and scaling every
    alpha together scales every enzyme together; the rate laws are homogeneous of degree one in
    e, so flux and proteome both scale by the same factor. Therefore

        sum_j C^J_{alpha_j} = 1     and     sum_j C^E_{alpha_j} = 1

    EXACTLY, by a theorem that knows nothing about my code. Bar: |sum - 1| < 1e-6. This is an
    external check on the whole pipeline, not an internal consistency check.

C3  THE MEASUREMENT. N90 = the smallest number of parameters carrying 90% of ||C||^2, and the
    participation ratio N_pr = (sum C^2)^2 / sum C^4. Swept over plant size L and controller
    count K, so the parameter count P varies by more than an order of magnitude. Fit N90 ~ P^b.
    PREDECLARED: b > 0.7 is EXPLOSION, b < 0.3 is SPARSE, and anything between is reported as
    intermediate with the exponent and its standard error rather than as a verdict. The exponent
    is the deliverable; the label is a convenience.

C4  TRANSFER, NOT MERELY SPARSITY. The fraction of ||C||^2 carried by controller parameters as K
    grows, and the rank correlation between the plant ranking at K = 0 and at each K > 0. Sparse
    with an unchanged ranking is a different world from sparse with a rearranged one, and only
    the second says the controller is where the measurements should go.

C5  THE SATURATION CHECK. ||C|| itself for both observables at every K. If the norm collapses,
    the sparsity is homeostasis and must be labelled homeostasis. Reported before C3's verdict,
    not after it, so it cannot be used to explain away an inconvenient number.

C6  THE MATCHED CONTROL. The same K edges rewired at random -- random target, random sensor,
    random sign -- with identical parameter values. If shuffled wiring reproduces the scaling,
    the result is about counting parameters and not about control architecture, and C3 says
    nothing about biology.

C7  THE FIXED POINT IS STABLE. Every eigenvalue of G must have negative real part. A steady-state
    sensitivity around an unstable fixed point is a number about a state the system leaves. If
    high-gain feedback destabilises the pathway that is a finding, not an error to route around.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from rem.atlas.hybrid_tune import RULE

SEED = 20260906
S_EXT = 10.0


# =================================================================================================
# THE MODEL
# =================================================================================================

def make_model(L, K, shuffled=False, seed=SEED, w=0.8):
    """Parameters and wiring for a chain of length L with K regulatory edges.

    Everything is heterogeneous: a perfectly symmetric chain gives every step the same control
    coefficient by symmetry, which would manufacture a dense important set out of nothing. The
    spread is lognormal with a fixed seed so the model is the same object on every run."""
    rng = np.random.default_rng(seed)
    nr = L + 1                                        # L chain steps plus the demand reaction
    sp = lambda n, s=0.5: np.exp(rng.normal(0.0, s, n))
    p = dict(
        kcat=10.0 * sp(nr), Km=1.0 * sp(nr), Kp=1.0 * sp(nr),
        alpha=1.0 * sp(nr), delta=0.1 * sp(nr),
    )
    # wiring
    tgt, sen, sgn = [], [], []
    for k in range(K):
        if shuffled:
            tgt.append(int(rng.integers(nr)))
            sen.append(int(rng.integers(L)))
            sgn.append(1 if rng.random() < 0.5 else -1)
        else:
            tgt.append(k % nr)                        # end-product repression, the canonical loop
            sen.append(L - 1 - (k // nr) % L)
            sgn.append(-1)
    p["Kr"] = 1.0 * sp(K) if K else np.zeros(0)
    p["nh"] = 2.0 * np.exp(rng.normal(0.0, 0.2, K)) if K else np.zeros(0)
    p["w"] = np.full(K, w) * np.exp(rng.normal(0.0, 0.1, K)) if K else np.zeros(0)
    wiring = dict(tgt=np.array(tgt, int), sen=np.array(sen, int), sgn=np.array(sgn, int), nr=nr, L=L)
    return p, wiring


PNAMES = ["kcat", "Km", "Kp", "alpha", "delta", "Kr", "nh", "w"]


def pack(p):
    return np.concatenate([np.atleast_1d(p[k]) for k in PNAMES])


def unpack(v, p_ref):
    out, i = {}, 0
    for k in PNAMES:
        n = len(np.atleast_1d(p_ref[k]))
        out[k] = v[i:i + n]
        i += n
    return out


def param_labels(p):
    lab = []
    for k in PNAMES:
        for i in range(len(np.atleast_1d(p[k]))):
            lab.append(f"{k}[{i}]")
    return lab


def is_controller(p):
    return np.array([k in ("Kr", "nh", "w")
                     for k in PNAMES for _ in range(len(np.atleast_1d(p[k])))])


def rhs(q, th, wir, p_ref):
    """dq/dt. Written so it runs in complex arithmetic unchanged -- that is what makes the
    complex-step Jacobian exact."""
    L, nr = wir["L"], wir["nr"]
    p = unpack(th, p_ref)
    x = q[:L]
    e = q[L:]
    dt = np.result_type(np.asarray(q).dtype, np.asarray(th).dtype)
    sub = np.concatenate([np.array([S_EXT], dtype=dt), x])       # x_0 = S, then x_1..x_L
    prod = np.concatenate([x, np.array([0.0], dtype=dt)])        # demand has no product
    ss = sub / p["Km"]
    pp = prod / p["Kp"]
    v = e * p["kcat"] * ss / (1.0 + ss + pp)
    dx = v[:L] - v[1:]
    h = np.ones(nr, dtype=dt)
    for k in range(len(wir["tgt"])):
        u = (x[wir["sen"][k]] / p["Kr"][k]) ** p["nh"][k]
        f = 1.0 / (1.0 + u) if wir["sgn"][k] < 0 else u / (1.0 + u)
        h[wir["tgt"][k]] = h[wir["tgt"][k]] * ((1.0 - p["w"][k]) + p["w"][k] * f)
    de = p["alpha"] * h - p["delta"] * e
    return np.concatenate([dx, de]), v


def F(q, th, wir, p_ref):
    return rhs(q, th, wir, p_ref)[0]


def observables(q, th, wir, p_ref):
    _, v = rhs(q, th, wir, p_ref)
    return {"J": v[-1], "E": np.sum(q[wir["L"]:])}


def cstep_jac(fun, z, n_out, h=1e-30):
    """Complex-step Jacobian. Exact to machine precision, no cancellation, no step to tune."""
    n = len(z)
    Jm = np.zeros((n_out, n))
    zc = z.astype(complex)
    for j in range(n):
        zc[j] += 1j * h
        Jm[:, j] = np.imag(fun(zc)) / h
        zc[j] -= 1j * h
    return Jm


def steady(th, wir, p_ref, q0=None, tmax=4000.0):
    """Integrate to the neighbourhood then polish with Newton. Returns (q, converged)."""
    from scipy.integrate import solve_ivp
    L, nr = wir["L"], wir["nr"]
    if q0 is None:
        p = unpack(th, p_ref)
        q0 = np.concatenate([np.full(L, 1.0), p["alpha"] / p["delta"]])
    sol = solve_ivp(lambda t, y: F(y, th, wir, p_ref), (0.0, tmax), q0,
                    method="BDF", rtol=1e-10, atol=1e-12)
    q = sol.y[:, -1]
    for _ in range(60):
        r = F(q, th, wir, p_ref)
        if np.max(np.abs(r)) < 1e-12:
            break
        G = cstep_jac(lambda z: F(z, th, wir, p_ref), q, len(q))
        try:
            dq = np.linalg.solve(G, -r)
        except np.linalg.LinAlgError:
            return q, False
        s = 1.0
        while s > 1e-6 and np.min(q + s * dq) <= 0:
            s *= 0.5
        q = q + s * dq
    return q, np.max(np.abs(F(q, th, wir, p_ref))) < 1e-9 and np.min(q) > 0


def control_coefficients(th, wir, p_ref, q=None):
    """One adjoint solve per observable gives every parameter's log-log control coefficient."""
    if q is None:
        q, ok = steady(th, wir, p_ref)
        if not ok:
            return None
    G = cstep_jac(lambda z: F(z, th, wir, p_ref), q, len(q))
    Fth = cstep_jac(lambda z: F(q, z, wir, p_ref), th, len(q))
    obs = observables(q, th, wir, p_ref)
    out = {"q": q, "G": G, "obs": obs}
    for name in ("J", "E"):
        dTdq = cstep_jac(lambda z: np.array([observables(z, th, wir, p_ref)[name]]), q, 1)[0]
        dTdth_direct = cstep_jac(lambda z: np.array([observables(q, z, wir, p_ref)[name]]), th, 1)[0]
        lam = np.linalg.solve(G.T, dTdq)
        dTdth = dTdth_direct - Fth.T @ lam
        out[name] = th * dTdth / obs[name]              # natural-log elasticity
    return out


def n90(c):
    s = np.sort(c ** 2)[::-1]
    if s.sum() <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(s) / s.sum(), 0.90) + 1)


def npr(c):
    s = c ** 2
    return float(s.sum() ** 2 / np.sum(s ** 2)) if np.sum(s ** 2) > 0 else 0.0


# =================================================================================================
# THE GATES
# =================================================================================================

def fd_check(th, wir, p_ref, adj, name, h=1e-6):
    """Central finite differences on the RE-SOLVED fixed point, both directions."""
    errs = []
    for i in range(len(th)):
        t1, t2 = th.copy(), th.copy()
        t1[i] *= np.exp(h); t2[i] *= np.exp(-h)
        q1, o1 = steady(t1, wir, p_ref)
        q2, o2 = steady(t2, wir, p_ref)
        if not (o1 and o2):
            errs.append(np.nan); continue
        v1 = observables(q1, t1, wir, p_ref)[name]
        v2 = observables(q2, t2, wir, p_ref)[name]
        fd = (np.log(v1) - np.log(v2)) / (2 * h)
        den = max(abs(fd), 1e-8)
        errs.append(abs(fd - adj[i]) / den)
    return np.array(errs)


def run_case(L, K, shuffled=False, w=0.8):
    p, wir = make_model(L, K, shuffled=shuffled, w=w)
    th = pack(p)
    q, ok = steady(th, wir, p)
    if not ok:
        return None
    r = control_coefficients(th, wir, p, q)
    ev = np.linalg.eigvals(r["G"])
    ctrl = is_controller(p)
    row = dict(L=L, K=K, P=len(th), shuffled=shuffled, p=p, wir=wir, th=th,
               stab=float(np.max(ev.real)), ctrl=ctrl, labels=param_labels(p),
               J=r["J"], E=r["E"], obs=r["obs"])
    for nm in ("J", "E"):
        c = r[nm]
        row[f"n90_{nm}"] = n90(c)
        row[f"npr_{nm}"] = npr(c)
        row[f"norm_{nm}"] = float(np.linalg.norm(c))
        row[f"fctl_{nm}"] = float(np.sum(c[ctrl] ** 2) / np.sum(c ** 2)) if K else 0.0
    return row


def fit_power(P, N):
    P, N = np.asarray(P, float), np.asarray(N, float)
    m = (P > 0) & (N > 0)
    x, y = np.log(P[m]), np.log(N[m])
    A = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    dof = max(1, len(x) - 2)
    s2 = float(res @ res) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else float("nan")


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE); P_("DOES CLOSING THE LOOP MAKE THE IMPORTANT PARAMETERS EXPLODE?"); P_(RULE)
    P_("  plant: metabolic chain with product inhibition; enzymes are dynamical variables driven")
    P_("  by Hill feedback. w = 0 recovers the open loop exactly, so open and closed loop are the")
    P_("  same model at different points of one parameter space.")

    # ---- C1  THE ADJOINT IS RIGHT --------------------------------------------------------------
    P_("\n" + RULE); P_("C1  THE ADJOINT IS RIGHT"); P_(RULE)
    c1ok = True
    for (L, K) in [(6, 0), (6, 4), (8, 9)]:
        p, wir = make_model(L, K)
        th = pack(p)
        q, ok = steady(th, wir, p)
        r = control_coefficients(th, wir, p, q)
        for nm in ("J", "E"):
            e = fd_check(th, wir, p, r[nm], nm)
            med, wrst = float(np.nanmedian(e)), float(np.nanmax(e))
            good = med < 1e-6 and wrst < 1e-3
            c1ok = c1ok and good
            P_(f"  L={L:<3} K={K:<3} {nm}  {len(th):>3} parameters   median rel err {med:.3e}"
               f"   worst {wrst:.3e}   {'PASS' if good else 'FAIL'}")
    P_(f"  C1: {'PASS' if c1ok else 'FAIL'}   (bars: median < 1e-6, worst < 1e-3)")

    # ---- C2  THE SUMMATION THEOREM -------------------------------------------------------------
    P_("\n" + RULE); P_("C2  THE SUMMATION THEOREM -- an external check, not an internal one"); P_(RULE)
    P_("  Open loop: h = 1, so e = alpha/delta and the rate laws are homogeneous of degree one in")
    P_("  e. Scaling every alpha together must scale J and E by exactly that factor, so the")
    P_("  alpha control coefficients must sum to 1. Nothing in the code knows this.")
    c2ok = True
    for L in (4, 8, 16):
        p, wir = make_model(L, 0)
        th = pack(p)
        q, ok = steady(th, wir, p)
        r = control_coefficients(th, wir, p, q)
        lab = np.array(param_labels(p))
        am = np.array([s.startswith("alpha") for s in lab])
        for nm in ("J", "E"):
            s = float(np.sum(r[nm][am]))
            good = abs(s - 1.0) < 1e-6
            c2ok = c2ok and good
            P_(f"  L={L:<3} sum of C^{nm}_alpha over {int(am.sum())} enzymes = {s:.12f}"
               f"   {'PASS' if good else 'FAIL'}")
    # and the closed loop, where the theorem does NOT hold -- reported so the check is not vacuous
    p, wir = make_model(8, 9); th = pack(p); q, ok = steady(th, wir, p)
    r = control_coefficients(th, wir, p, q)
    lab = np.array(param_labels(p)); am = np.array([s.startswith("alpha") for s in lab])
    P_(f"  closed loop L=8 K=9: sum = {float(np.sum(r['J'][am])):.6f}  -- the theorem is a")
    P_( "  statement about the OPEN loop, so this differing from 1 is the control working, and it")
    P_( "  shows C2 is a real constraint rather than an identity that holds either way.")
    P_(f"  C2: {'PASS' if c2ok else 'FAIL'}")

    # ---- C5  THE SATURATION CHECK, reported BEFORE the measurement -----------------------------
    P_("\n" + RULE); P_("C5  THE SATURATION CHECK  (before C3, so it cannot be used to explain C3 away)"); P_(RULE)
    P_("  If feedback simply flattened the observable, sensitivities would go to zero and any")
    P_("  sparsity would be homeostasis rather than structure. ledger U is exactly this mistake.")
    P_(f"    {'K':>4} {'P':>5} {'||C_J||':>12} {'||C_E||':>12} {'J':>12} {'E':>12} {'max Re eig':>12}")
    KS = [0, 1, 2, 4, 8, 16, 32]
    krows = []
    for K in KS:
        r = run_case(8, K)
        if r is None:
            P_(f"    {K:>4}  no stable fixed point"); continue
        krows.append(r)
        P_(f"    {K:>4} {r['P']:>5} {r['norm_J']:>12.4f} {r['norm_E']:>12.4f}"
           f" {r['obs']['J']:>12.4f} {r['obs']['E']:>12.4f} {r['stab']:>12.3e}")
    nJ = [r["norm_J"] for r in krows]
    sat = min(nJ) / max(nJ) < 0.1
    P_(f"  ||C_J|| ranges {min(nJ):.4f} to {max(nJ):.4f}, ratio {min(nJ)/max(nJ):.3f}")
    P_(f"  C5: {'FAIL -- the observable is being flattened; read C3 as homeostasis' if sat else 'PASS -- sensitivity does not collapse, so C3 measures structure'}")

    # ---- C7  STABILITY -------------------------------------------------------------------------
    P_("\n" + RULE); P_("C7  EVERY FIXED POINT IS STABLE"); P_(RULE)
    worst = max(r["stab"] for r in krows)
    P_(f"  worst eigenvalue real part across the K sweep: {worst:.4e}")
    c7ok = worst < 0
    P_(f"  C7: {'PASS' if c7ok else 'FAIL -- feedback destabilises the pathway, which is a finding'}")

    # ---- C3  THE MEASUREMENT -------------------------------------------------------------------
    P_("\n" + RULE); P_("C3  THE MEASUREMENT: does the important set grow with the parameter count?"); P_(RULE)
    LS = [4, 6, 8, 12, 16, 24, 32]
    rows = list(krows)
    P_("  sweep A: L = 8, K = 0..32 (controllers added to a fixed plant)")
    P_(f"    {'K':>4} {'P':>5} {'N90_J':>7} {'N90/P':>7} {'Npr_J':>8} {'N90_E':>7} {'Npr_E':>8}")
    for r in krows:
        P_(f"    {r['K']:>4} {r['P']:>5} {r['n90_J']:>7} {r['n90_J']/r['P']:>7.3f}"
           f" {r['npr_J']:>8.2f} {r['n90_E']:>7} {r['npr_E']:>8.2f}")
    P_("\n  sweep B: K = L+1 (every enzyme regulated), L = 4..32 (the plant itself grows)")
    P_(f"    {'L':>4} {'K':>4} {'P':>5} {'N90_J':>7} {'N90/P':>7} {'Npr_J':>8} {'N90_E':>7}")
    lrows = []
    for L in LS:
        r = run_case(L, L + 1)
        if r is None:
            P_(f"    {L:>4}  no stable fixed point"); continue
        lrows.append(r); rows.append(r)
        P_(f"    {L:>4} {r['K']:>4} {r['P']:>5} {r['n90_J']:>7} {r['n90_J']/r['P']:>7.3f}"
           f" {r['npr_J']:>8.2f} {r['n90_E']:>7}")
    bA, sA = fit_power([r["P"] for r in krows], [r["n90_J"] for r in krows])
    bB, sB = fit_power([r["P"] for r in lrows], [r["n90_J"] for r in lrows])
    bAll, sAll = fit_power([r["P"] for r in rows], [r["n90_J"] for r in rows])
    P_(f"\n  N90_J ~ P^b     sweep A  b = {bA:+.4f} +- {sA:.4f}")
    P_(f"                  sweep B  b = {bB:+.4f} +- {sB:.4f}")
    P_(f"                  pooled   b = {bAll:+.4f} +- {sAll:.4f}")
    lab = ("EXPLOSION" if bAll > 0.7 else "SPARSE" if bAll < 0.3 else "INTERMEDIATE")
    P_(f"  predeclared bands: b > 0.7 EXPLOSION, b < 0.3 SPARSE, else intermediate")
    P_(f"  C3: {lab}   (the exponent is the deliverable; the label is a convenience)")

    # ---- C4  TRANSFER --------------------------------------------------------------------------
    P_("\n" + RULE); P_("C4  TRANSFER: does the important set change IDENTITY as well as size?"); P_(RULE)
    base = krows[0]
    bl = np.array(base["labels"])
    P_(f"    {'K':>4} {'frac ||C||^2 on controller':>28} {'rho(plant rank vs K=0)':>24}")
    for r in krows:
        cl = np.array(r["labels"])
        common = [s for s in bl if s in set(cl)]
        ib = [list(bl).index(s) for s in common]
        ic = [list(cl).index(s) for s in common]
        rho = spearman(np.abs(base["J"][ib]), np.abs(r["J"][ic]))
        P_(f"    {r['K']:>4} {r['fctl_J']:>28.4f} {rho:>24.4f}")
    top = krows[-1]
    o = np.argsort(-np.abs(top["J"]))[:10]
    P_(f"\n  the ten largest control coefficients at K={top['K']}:")
    for i in o:
        P_(f"    {top['labels'][i]:<12} {top['J'][i]:+10.4f}   {'controller' if top['ctrl'][i] else 'plant'}")

    # ---- C6  THE MATCHED CONTROL ---------------------------------------------------------------
    P_("\n" + RULE); P_("C6  THE MATCHED CONTROL: the same edges, rewired at random"); P_(RULE)
    P_(f"    {'L':>4} {'K':>4} {'P':>5} {'N90 real':>9} {'N90 shuf':>9} {'Npr real':>9} {'Npr shuf':>9}")
    srows = []
    for (L, K) in [(8, 4), (8, 9), (8, 16), (8, 32), (16, 17), (24, 25), (32, 33)]:
        a = run_case(L, K)
        b = run_case(L, K, shuffled=True)
        if a is None or b is None:
            P_(f"    {L:>4} {K:>4}  no stable fixed point"); continue
        srows.append(b)
        P_(f"    {L:>4} {K:>4} {a['P']:>5} {a['n90_J']:>9} {b['n90_J']:>9}"
           f" {a['npr_J']:>9.2f} {b['npr_J']:>9.2f}")
    if len(srows) >= 3:
        bS, sS = fit_power([r["P"] for r in srows], [r["n90_J"] for r in srows])
        P_(f"  shuffled exponent b = {bS:+.4f} +- {sS:.4f} against real {bAll:+.4f} +- {sAll:.4f}")
        sep = abs(bS - bAll) > 2 * np.hypot(sS, sAll)
        P_(f"  C6: {'the architectures separate -- C3 is about control structure' if sep else 'NOT SEPARATED -- C3 is about counting parameters, not about biology'}")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_controller.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P_(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
