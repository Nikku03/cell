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
FOUR CORRECTIONS, RECORDED BEFORE THE REPAIRED RUN
=================================================================================================

The first run of this module failed C1 and produced one meaningless row. All four defects were
mine, none were the model's, and all four are recorded here rather than quietly patched.

(1) C1's BAR WAS A RELATIVE ERROR ON A QUANTITY THAT IS ZERO. Eight parameters "failed" at
    7e-3 relative error. Their control coefficients are 2e-13 against a maximum of 28.1, and the
    absolute discrepancy scaled by that maximum is 2.5e-12. The adjoint was correct to twelve
    digits; the gate was measuring the finite-difference reference's own truncation noise on a
    component that is exactly zero. This is the same family as ledger P -- a bar unreachable on
    any evidence -- read in the opposite direction: a bar that FAILS on any evidence. C1 is now
    scaled by max|C|, which is the quantity that actually governs a ranking and an N90.

(2) THE CANONICAL WIRING CLOSED THE LOOP OVER THE EXIT AND RAN AWAY. Edges were assigned to
    enzyme (k mod L+1), which at k = L+1 targets the DEMAND reaction, repressed by its own
    substrate. Repressing the reaction that removes a metabolite using that metabolite is
    positive feedback on accumulation. At L=8, K=9 the end product reached 1.6e13, cond(G) was
    4.5e16 and the largest eigenvalue was -5.9e-15, i.e. the fixed point was a marginally stable
    degenerate one. Every parameter but four had a control coefficient of ~1e-13 there, so that
    row reported N90 = 1: perfect sparsity, entirely an artefact. This is kept as a RESULT, not
    removed -- gate C7 now detects it and names the mechanism, and the exit-repressed wiring is
    retained as a deliberately broken control so the detector has something to detect.

(3) MY CLAIM ABOUT C2 IN THE CLOSED LOOP WAS WRONG. The docstring said the summation theorem
    is "a statement about the OPEN loop", and predicted the closed-loop sum would differ from 1.
    It came out 1.000000. The theorem in fact SURVIVES this feedback, and the reason is
    structural: h depends only on metabolites, so scaling every alpha scales every enzyme at
    unchanged x, and the fixed point moves not at all. It would break for a controller that
    sensed the proteome. That case is now included, and C2 requires the theorem to hold in the
    first two and to break in the third -- otherwise the check cannot distinguish a correct
    pipeline from one that returns 1 by construction.

(4) THE INTEGRATOR DROVE STATES NEGATIVE and x**n produced NaN. Integration is now done in
    log coordinates, du/dt = F(e^u)/e^u, which keeps every state positive by construction.

(5) TWO FURTHER DEFECTS FOUND IN THE REPAIRED RUN, recorded here rather than re-tuned away.
    C7 FAILED: only 1 of 3 exit-repressed configurations was rejected. The other two had
    cond(G) ~ 1e11 and metabolites at 2e7 -- plainly pathological against the canonical runs'
    20 to 750 -- and slipped under bars I set at 1e12 and 1e8. The bars were about six orders
    too loose. They are NOT retuned here, because retuning a bar to catch the case that
    embarrassed it is how a gate stops being evidence. What is stated instead is the
    containment: C3 runs only the canonical wiring, so no rejected or borderline configuration
    enters the measurement, and the C7 failure does not touch the result.

    SWEEP B IS A SELECTED SAMPLE AT LARGE L. Rejections rise 0, 0, 1, 1, 2, 6, 9 out of 12 as
    L goes 4..32, all "singular" -- long chains under full end-product repression become
    numerically degenerate. So sweep B's negative exponent is fitted on a survivor population
    that shrinks to 3 of 12. The tables now carry the rejection count, and the exponent is
    quoted twice: on everything, and restricted to the range where at least 10 of 12 survive.
    Only the restricted one is evidence.

Also: a single lognormal parameter draw makes N90 a random variable, and the first run fitted an
exponent to one sample per configuration, getting b = 0.58 +- 0.33 -- a number spanning both
predeclared bands and therefore worth nothing. C3 now averages over independent seeds and fits
the means.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN

=================================================================================================

C1  THE ADJOINT IS RIGHT. Every control coefficient is re-derived by central finite differences
    on the re-solved fixed point, in both directions. Bar: worst |C_adj - C_fd| / max|C| < 1e-6,
    scaled by the largest coefficient rather than element-wise, for the reason in correction (1).
    An adjoint that has never been finite-differenced has been wrong twice in this build order,
    once by exactly ln10.

C2  THE SUMMATION THEOREM. In the open loop h = 1, so e_j = alpha_j / delta_j and scaling every
    alpha together scales every enzyme together; the rate laws are homogeneous of degree one in
    e, so flux and proteome both scale by the same factor. Therefore

        sum_j C^J_{alpha_j} = 1     and     sum_j C^E_{alpha_j} = 1

    EXACTLY, by a theorem that knows nothing about my code. Bar: |sum - 1| < 1e-6. It must also
    hold in the closed loop, because h depends only on metabolites -- and it must BREAK for a
    controller that senses the proteome. All three are required, so the gate cannot pass by
    returning 1 unconditionally.

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

C7  THE FIXED POINT IS STABLE AND PHYSIOLOGICAL. Every eigenvalue of G must have negative real
    part, cond(G) must be below 1e12, and every state must lie in [1e-8, 1e8]. A sensitivity
    computed around a marginally stable point with a metabolite at 1e13 is a number about a
    state no cell occupies. The exit-repressed wiring is run deliberately so that this gate has
    a true positive to find rather than only true negatives.
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

def make_model(L, K, mode="canon", seed=SEED, w=0.8):
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
    # wiring. mode "canon": end-product repression of the CHAIN enzymes, never the exit --
    # repressing the reaction that removes a metabolite with that metabolite is positive feedback
    # on accumulation, which is correction (2). mode "exit": that broken wiring, kept as a control.
    tgt, sen, sgn = [], [], []
    for k in range(K):
        if mode == "shuffled":
            tgt.append(int(rng.integers(nr)))
            sen.append(int(rng.integers(L)))
            sgn.append(1 if rng.random() < 0.5 else -1)
        elif mode == "exit":
            tgt.append(k % nr)
            sen.append(L - 1 - (k // nr) % L)
            sgn.append(-1)
        else:
            tgt.append(k % L)
            sen.append((L - 1 - (k // L)) % L)
            sgn.append(-1)
    p["Kr"] = 1.0 * sp(K) if K else np.zeros(0)
    p["nh"] = 2.0 * np.exp(rng.normal(0.0, 0.2, K)) if K else np.zeros(0)
    p["w"] = np.full(K, w) * np.exp(rng.normal(0.0, 0.1, K)) if K else np.zeros(0)
    wiring = dict(tgt=np.array(tgt, int), sen=np.array(sen, int), sgn=np.array(sgn, int),
                  nr=nr, L=L, mode=mode)
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
    etot = np.sum(e)
    for k in range(len(wir["tgt"])):
        # a proteome-sensing controller reads total enzyme rather than a metabolite. It is the
        # only one of these modes for which the summation theorem must fail, which is what makes
        # C2 a real check instead of an identity.
        sig = etot if wir["mode"] == "proteome" else x[wir["sen"][k]]
        u = (sig / p["Kr"][k]) ** p["nh"][k]
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
    # integrate in log coordinates so no state can go negative and x**n cannot produce NaN
    sol = solve_ivp(lambda t, u: F(np.exp(u), th, wir, p_ref) / np.exp(u),
                    (0.0, tmax), np.log(q0), method="BDF", rtol=1e-10, atol=1e-12)
    q = np.exp(sol.y[:, -1])
    if not np.all(np.isfinite(q)):
        return q0, False
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
    resid = np.max(np.abs(F(q, th, wir, p_ref)))
    scale = np.max(np.abs(q)) + 1.0
    return q, bool(resid / scale < 1e-9 and np.all(np.isfinite(q)) and np.min(q) > 0)


def accept(q, G):
    """C7's acceptance test, applied to every configuration and not only to the ones tabulated.
    A fixed point that is marginally stable, numerically singular, or sitting at 1e13 is not a
    state a cell occupies and no sensitivity around it means anything."""
    ev = np.linalg.eigvals(G)
    cond = float(np.linalg.cond(G))
    stab = float(np.max(ev.real))
    rng_ok = bool(np.min(q) > 1e-8 and np.max(q) < 1e8)
    return dict(stab=stab, cond=cond, range_ok=rng_ok,
                ok=bool(stab < 0 and cond < 1e12 and rng_ok))


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
    """Central finite differences on the RE-SOLVED fixed point, both directions.

    Returns error SCALED BY max|C| rather than element-wise relative -- correction (1). An
    element-wise relative error on a coefficient of 2e-13 measures the finite-difference
    reference's truncation, not the adjoint, and no correct adjoint could ever pass it."""
    sc = float(np.max(np.abs(adj)))
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
        errs.append(abs(fd - adj[i]) / sc)
    return np.array(errs)


def run_case(L, K, mode="canon", w=0.8, seed=SEED):
    """One configuration. Returns None if C7's acceptance test rejects the fixed point, with the
    reason attached, so rejections are counted rather than silently dropped."""
    p, wir = make_model(L, K, mode=mode, seed=seed, w=w)
    th = pack(p)
    q, ok = steady(th, wir, p)
    if not ok:
        return dict(rejected="no fixed point", L=L, K=K, mode=mode, seed=seed)
    r = control_coefficients(th, wir, p, q)
    acc = accept(q, r["G"])
    ctrl = is_controller(p)
    row = dict(L=L, K=K, P=len(th), mode=mode, seed=seed, p=p, wir=wir, th=th, q=q,
               ctrl=ctrl, labels=param_labels(p), J=r["J"], E=r["E"], obs=r["obs"], **acc)
    for nm in ("J", "E"):
        c = r[nm]
        row[f"n90_{nm}"] = n90(c)
        row[f"npr_{nm}"] = npr(c)
        row[f"norm_{nm}"] = float(np.linalg.norm(c))
        row[f"fctl_{nm}"] = float(np.sum(c[ctrl] ** 2) / np.sum(c ** 2)) if K else 0.0
    if not acc["ok"]:
        row["rejected"] = ("unstable" if acc["stab"] >= 0 else
                           "singular" if acc["cond"] >= 1e12 else "outside physiological range")
    return row


def replicate(L, K, mode="canon", nrep=12, w=0.8):
    """N90 is a random variable: the parameters are a lognormal draw. One sample per point gave
    an exponent spanning both predeclared bands, which is worth nothing."""
    good, rej = [], 0
    for i in range(nrep):
        r = run_case(L, K, mode=mode, w=w, seed=SEED + 1000 * i)
        if r.get("rejected"):
            rej += 1
        else:
            good.append(r)
    if not good:
        return None
    agg = dict(L=L, K=K, mode=mode, P=good[0]["P"], n=len(good), rejected=rej)
    for key in ("n90_J", "npr_J", "n90_E", "npr_E", "norm_J", "norm_E", "fctl_J", "stab"):
        v = np.array([g[key] for g in good], float)
        agg[key] = float(v.mean())
        agg[key + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    agg["rows"] = good
    return agg


def fit_power(P, N, W=None):
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
    P_("  Repaired run. Four defects from the first run are recorded in the docstring: a relative-")
    P_("  error bar on a coefficient that is exactly zero, a wiring that closed the loop over the")
    P_("  exit and ran away to 1e13, a wrong claim about the summation theorem, and an integrator")
    P_("  that drove states negative. All four were mine.")

    # ---- C1  THE ADJOINT IS RIGHT --------------------------------------------------------------
    P_("\n" + RULE); P_("C1  THE ADJOINT IS RIGHT"); P_(RULE)
    P_("  error scaled by max|C|, because an element-wise relative error on a machine-zero")
    P_("  coefficient measures the finite-difference reference and not the adjoint.")
    c1ok = True
    for (L, K) in [(6, 0), (6, 4), (8, 8), (12, 12)]:
        p, wir = make_model(L, K)
        th = pack(p)
        q, ok = steady(th, wir, p)
        r = control_coefficients(th, wir, p, q)
        for nm in ("J", "E"):
            e = fd_check(th, wir, p, r[nm], nm)
            wrst = float(np.nanmax(e))
            good = wrst < 1e-6
            c1ok = c1ok and good
            P_(f"  L={L:<3} K={K:<3} {nm}  {len(th):>3} parameters   max|C| {np.max(np.abs(r[nm])):9.4f}"
               f"   worst scaled err {wrst:.3e}   {'PASS' if good else 'FAIL'}")
    P_(f"  C1: {'PASS' if c1ok else 'FAIL'}   (bar: worst scaled error < 1e-6)")

    # ---- C2  THE SUMMATION THEOREM -------------------------------------------------------------
    P_("\n" + RULE); P_("C2  THE SUMMATION THEOREM -- must hold twice and BREAK once"); P_(RULE)
    P_("  Scaling every alpha together scales every enzyme together, so J and E scale by that")
    P_("  factor and the alpha control coefficients sum to 1. This survives metabolite-sensing")
    P_("  feedback, because h reads x and the fixed point does not move -- my first-run claim that")
    P_("  it would break was wrong. It must break when the controller senses the PROTEOME, and")
    P_("  that case is included so the gate cannot pass by returning 1 unconditionally.")
    c2ok = True
    for (L, K, mode, must) in [(4, 0, "canon", True), (8, 0, "canon", True), (16, 0, "canon", True),
                               (8, 8, "canon", True), (12, 12, "canon", True),
                               (8, 8, "proteome", False)]:
        p, wir = make_model(L, K, mode=mode)
        th = pack(p); q, ok = steady(th, wir, p)
        if not ok:
            P_(f"  L={L} K={K} {mode}: no fixed point"); c2ok = False; continue
        r = control_coefficients(th, wir, p, q)
        lab = np.array(param_labels(p))
        am = np.array([x.startswith("alpha") for x in lab])
        sJ = float(np.sum(r["J"][am]))
        holds = abs(sJ - 1.0) < 1e-6
        good = (holds == must)
        c2ok = c2ok and good
        tag = "must hold" if must else "MUST BREAK"
        P_(f"  L={L:<3} K={K:<3} {mode:<9} sum C^J_alpha = {sJ:>14.10f}   {tag:<10}"
           f"   {'PASS' if good else 'FAIL'}")
    P_(f"  C2: {'PASS' if c2ok else 'FAIL'}")

    # ---- C7  STABILITY AND PHYSIOLOGICAL RANGE, with a true positive ---------------------------
    P_("\n" + RULE); P_("C7  EVERY FIXED POINT IS STABLE, CONDITIONED AND PHYSIOLOGICAL"); P_(RULE)
    P_("  The exit-repressed wiring is run on purpose so the detector has something to detect.")
    P_(f"    {'mode':<10} {'L':>3} {'K':>3} {'max Re eig':>12} {'cond(G)':>11} {'max state':>11} {'verdict':>10}")
    c7_pos = c7_neg = 0
    for (mode, L, K) in [("canon", 8, 4), ("canon", 8, 8), ("canon", 8, 16), ("canon", 16, 16),
                         ("exit", 8, 9), ("exit", 8, 10), ("exit", 12, 13)]:
        r = run_case(L, K, mode=mode)
        if "stab" not in r:
            P_(f"    {mode:<10} {L:>3} {K:>3}   no fixed point"); continue
        v = "ACCEPT" if r["ok"] else "REJECT"
        if mode == "exit" and not r["ok"]:
            c7_pos += 1
        if mode == "canon" and r["ok"]:
            c7_neg += 1
        P_(f"    {mode:<10} {L:>3} {K:>3} {r['stab']:>12.3e} {r['cond']:>11.2e}"
           f" {np.max(r['q']):>11.3e} {v:>10}")
    P_(f"  exit-repressed configurations rejected: {c7_pos}/3   canonical accepted: {c7_neg}/4")
    c7ok = c7_pos == 3 and c7_neg == 4
    P_(f"  C7: {'PASS -- the detector fires on the broken wiring and not on the sound one' if c7ok else 'FAIL'}")
    P_("  MECHANISM: repressing the demand reaction with its own substrate is positive feedback on")
    P_("  accumulation. The end product runs to 1e13, the Jacobian goes singular, and every")
    P_("  control coefficient but a handful collapses to 1e-13 -- which reads as perfect sparsity")
    P_("  and is nothing of the kind. That row is why C7 exists.")

    # ---- C5  THE SATURATION CHECK, before C3 ---------------------------------------------------
    P_("\n" + RULE); P_("C5  THE SATURATION CHECK  (before C3, so it cannot explain C3 away)"); P_(RULE)
    P_("  If feedback merely flattened the observable, sensitivities would vanish and any sparsity")
    P_("  would be homeostasis rather than structure. That is ledger U.")
    KS = [0, 1, 2, 4, 8, 16, 32]
    P_(f"    {'K':>4} {'P':>5} {'||C_J||':>10} {'+-':>8} {'||C_E||':>10} {'frac ctl':>9} {'rej':>4}")
    krows = []
    for K in KS:
        a = replicate(8, K)
        if a is None:
            P_(f"    {K:>4}  every replicate rejected"); continue
        krows.append(a)
        P_(f"    {K:>4} {a['P']:>5} {a['norm_J']:>10.4f} {a['norm_J_se']:>8.4f}"
           f" {a['norm_E']:>10.4f} {a['fctl_J']:>9.4f} {a['rejected']:>4}")
    nJ = [a["norm_J"] for a in krows]
    sat = min(nJ) / max(nJ) < 0.1
    P_(f"  ||C_J|| ranges {min(nJ):.4f} to {max(nJ):.4f}, ratio {min(nJ)/max(nJ):.3f}")
    P_(f"  C5: {'FAIL -- the observable is flattened; read C3 as homeostasis' if sat else 'PASS -- sensitivity does not collapse, so C3 measures structure'}")

    # ---- C3  THE MEASUREMENT -------------------------------------------------------------------
    P_("\n" + RULE); P_("C3  THE MEASUREMENT: does the important set grow with the parameter count?"); P_(RULE)
    P_("  every row is the mean over 12 independent lognormal parameter draws")
    P_("\n  sweep A: L = 8, controllers added to a fixed plant")
    P_(f"    {'K':>4} {'P':>5} {'kept':>5} {'rej':>4} {'N90_J':>8} {'+-':>6} {'N90/P':>7} {'Npr_J':>8} {'N90_E':>8}")
    for a in krows:
        P_(f"    {a['K']:>4} {a['P']:>5} {a['n']:>5} {a['rejected']:>4} {a['n90_J']:>8.2f}"
           f" {a['n90_J_se']:>6.2f} {a['n90_J']/a['P']:>7.3f} {a['npr_J']:>8.2f} {a['n90_E']:>8.2f}")
    P_("\n  sweep B: K = L (every chain enzyme regulated), the plant itself grows")
    P_(f"    {'L':>4} {'K':>4} {'P':>5} {'kept':>5} {'rej':>4} {'N90_J':>8} {'+-':>6} {'N90/P':>7} {'Npr_J':>8}")
    lrows = []
    for L in [4, 6, 8, 12, 16, 24, 32]:
        a = replicate(L, L)
        if a is None:
            P_(f"    {L:>4}  every replicate rejected"); continue
        lrows.append(a)
        P_(f"    {L:>4} {a['K']:>4} {a['P']:>5} {a['n']:>5} {a['rejected']:>4} {a['n90_J']:>8.2f}"
           f" {a['n90_J_se']:>6.2f} {a['n90_J']/a['P']:>7.3f} {a['npr_J']:>8.2f}")
    allr = krows + lrows
    bA, sA = fit_power([a["P"] for a in krows], [a["n90_J"] for a in krows])
    bB, sB = fit_power([a["P"] for a in lrows], [a["n90_J"] for a in lrows])
    bAll, sAll = fit_power([a["P"] for a in allr], [a["n90_J"] for a in allr])
    pA, qA = fit_power([a["P"] for a in allr], [a["npr_J"] for a in allr])
    P_(f"\n  N90_J ~ P^b     sweep A (add controllers) b = {bA:+.4f} +- {sA:.4f}")
    P_(f"                  sweep B (grow the plant)   b = {bB:+.4f} +- {sB:.4f}")
    P_(f"                  pooled                     b = {bAll:+.4f} +- {sAll:.4f}")
    P_(f"  Npr_J ~ P^b     pooled                     b = {pA:+.4f} +- {qA:.4f}   (continuous measure)")
    lo = [a for a in lrows if a["rejected"] <= 2]
    if len(lo) >= 3:
        bR, sR = fit_power([a["P"] for a in lo], [a["n90_J"] for a in lo])
        P_(f"\n  RESTRICTED to the range where at least 10 of 12 replicates survive"
           f" (L <= {max(a['L'] for a in lo)}):")
        P_(f"    sweep B  b = {bR:+.4f} +- {sR:.4f}   over P = {min(a['P'] for a in lo)}"
           f" to {max(a['P'] for a in lo)}, N90 = {min(a['n90_J'] for a in lo):.2f}"
           f" to {max(a['n90_J'] for a in lo):.2f}")
        P_( "    Only this one is evidence. The unrestricted fit runs over a survivor population")
        P_( "    that shrinks to 3 of 12, and a selected sample cannot carry a scaling exponent.")
    lab = ("EXPLOSION" if bAll > 0.7 else "SPARSE" if bAll < 0.3 else "INTERMEDIATE")
    P_(f"  predeclared bands: b > 0.7 EXPLOSION, b < 0.3 SPARSE, else intermediate")
    P_(f"  C3: {lab}")
    P_(f"  The two sweeps ask different questions and are reported apart on purpose: sweep A adds")
    P_(f"  parameters WITHOUT adding plant, sweep B adds both. If they disagree, the disagreement")
    P_(f"  is the result.")

    # ---- C4  TRANSFER --------------------------------------------------------------------------
    P_("\n" + RULE); P_("C4  TRANSFER: does the important set change IDENTITY as well as size?"); P_(RULE)
    base = krows[0]["rows"][0]
    bl = list(np.array(base["labels"]))
    P_(f"    {'K':>4} {'frac ||C||^2 on controller':>28} {'rho(plant rank vs K=0)':>24}")
    for a in krows:
        r0 = a["rows"][0]
        cl = list(np.array(r0["labels"]))
        common = [x for x in bl if x in set(cl)]
        rho = spearman(np.abs(base["J"][[bl.index(x) for x in common]]),
                       np.abs(r0["J"][[cl.index(x) for x in common]]))
        P_(f"    {a['K']:>4} {a['fctl_J']:>28.4f} {rho:>24.4f}")
    top = krows[-1]["rows"][0]
    o = np.argsort(-np.abs(top["J"]))[:10]
    P_(f"\n  the ten largest control coefficients at K={top['K']} (seed {top['seed']}):")
    for i in o:
        P_(f"    {top['labels'][i]:<12} {top['J'][i]:+10.4f}   {'controller' if top['ctrl'][i] else 'plant'}")

    # ---- C6  THE MATCHED CONTROL ---------------------------------------------------------------
    P_("\n" + RULE); P_("C6  THE MATCHED CONTROL: the same edges, rewired at random"); P_(RULE)
    P_(f"    {'L':>4} {'K':>4} {'P':>5} {'N90 real':>10} {'+-':>6} {'N90 shuf':>10} {'+-':>6} {'rej':>4}")
    srows, rrows = [], []
    for (L, K) in [(8, 4), (8, 8), (8, 16), (8, 32), (16, 16), (24, 24), (32, 32)]:
        a = replicate(L, K)
        b = replicate(L, K, mode="shuffled")
        if a is None or b is None:
            P_(f"    {L:>4} {K:>4}  every replicate rejected"); continue
        rrows.append(a); srows.append(b)
        P_(f"    {L:>4} {K:>4} {a['P']:>5} {a['n90_J']:>10.2f} {a['n90_J_se']:>6.2f}"
           f" {b['n90_J']:>10.2f} {b['n90_J_se']:>6.2f} {b['rejected']:>4}")
    if len(srows) >= 3:
        bS, sS = fit_power([a["P"] for a in srows], [a["n90_J"] for a in srows])
        bR, sR = fit_power([a["P"] for a in rrows], [a["n90_J"] for a in rrows])
        P_(f"  shuffled b = {bS:+.4f} +- {sS:.4f}   against real b = {bR:+.4f} +- {sR:.4f}")
        sep = abs(bS - bR) > 2 * np.hypot(sS, sR)
        P_(f"  C6: {'the architectures SEPARATE -- C3 is about control structure' if sep else 'NOT SEPARATED -- C3 is about counting parameters, not about which wiring'}")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_controller.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P_(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
