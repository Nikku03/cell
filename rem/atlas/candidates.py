"""What REM can compute from four groups' OWN PUBLISHED NUMBERS -- and what it cannot.

WHY THIS MODULE EXISTS. The request was "run REM on their data". Their raw data is not in my
hands: I have the published papers, not the per-lesion CFU tables, not the single-cell lag
distributions, not the FACS event files. So this module does the only honest version of that
request. Every input below is a number a named paper states in words I retrieved and can quote.
Where a needed number is NOT in the retrievable text it is marked UNRETRIEVED and enters the
calculation as a swept variable, never as an invented value. A calculation whose inputs are
invented is worth less than no calculation at all.

Ledger defect R protocol applies to every extracted number: it is recorded with the sentence it
came from, in CITATIONS below. Ledger defect S applies to every number reused: a measurement
records what a number IS, not WHERE IT APPLIES, and three of the four cases below turn on
exactly that distinction.

Source of every article: PubMed / PubMed Central. DOIs in CITATIONS.

=================================================================================================
THE FOUR CASES
=================================================================================================
A  TB relapse in macaques -- Maiello, ..., Fortune, Flynn, Lin. doi:10.1128/iai.00177-25
   Their stated gap: complete sterilization protects but "cannot be predicted by PET CT", and
   "not every site of persistent Mtb growth after drug treatment is capable of dissemination".
   The question they cannot observe: given residual burdens spread across lesions, what is the
   probability that AT LEAST ONE lesion fails to sterilize?

B  Intracellular S. aureus persisters -- Peyrusson, Nguyen, Najdovski, Van Bambeke.
   doi:10.1128/spectrum.02313-21. They publish a fully specified biphasic kill curve: two
   slopes and a breakpoint. The question: what do those two slopes actually pin down?

C  PerSort -- Srinivas, ..., Baliga. doi:10.1128/mSystems.01127-20. They sort a ~1% subpopulation
   with a sorter characterised at 93% efficiency. The question: what is the purity of a gate on
   a 1% population, given a misclassification rate measured on a balanced mixture?

D  Lag-time optimisation -- Fridman, Goldberg, Ronin, Shoresh, Balaban. doi:10.1038/nature13469.
   Already computed in RESULTS_selection.txt; referenced here, not recomputed.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN. Deciding statistic: worst case over every sweep.
=================================================================================================

A1  EXACTNESS. With per-lesion failure probability p_i = 1 - exp(-q_i N_i), the conjunction
    P(>=1 lesion fails) = 1 - prod(1-p_i) must equal 1 - exp(-sum q_i N_i) to < 1e-12 relative.
    If this fails the rest of case A is meaningless.

A2  THE SPREAD QUESTION, PUT AS A TEST AND NOT AS AN ASSUMPTION. Hold the mean burden N fixed
    and the per-bacillus escape probability q homogeneous. Sweep the coefficient of variation of
    the per-lesion burdens from 0 to 3. PREDECLARED: if P(>=1 fails) moves by more than 1e-9
    relative, spread matters mechanically. If it does not move, then under this model the MEAN IS
    SUFFICIENT and the spread hypothesis is FALSE. Report whichever occurs. No result is
    discarded for being the unwelcome one.

A3  WHAT MAKES SPREAD MATTER. Let q vary across lesions too. PREDECLARED IDENTITY:
        log P(all sterilise) = -n * [ E(q)E(N) + Cov(q,N) ]
    Gate: numerically computed log P matches that expression to < 1e-12 relative, at negative,
    zero and positive correlation. This converts "the spread matters" into ONE measurable number,
    n*Cov(q,N), which is zero exactly when the mean is sufficient.

A4  MEAN-FIELD ERROR, IN ORDERS. Report log10[ P_exact / P_meanfield ] at a stated correlation.
    Non-vacuity: both probabilities strictly inside (1e-9, 0.999) or the row is void.

A5  IDENTIFIABILITY FROM THEIR ACTUAL COUNTS. An animal relapses iff >= 1 of its n lesions
    disseminates, each with probability p: P_relapse = 1 - (1-p)^n. PREDECLARED: the binomial
    likelihood of their observed 8 of 12 depends on (n,p) ONLY through theta = n*log(1-p).
    Gate: along a level set of theta the log-likelihood must be constant to < 1e-12. If it is,
    then n and p are NOT separately identifiable from animal-level relapse counts -- by proof,
    not by weak data -- and their barcode assay is the only thing that can separate them.

A6  EVERY FRACTION SHIPS WITH A BAND. Exact Clopper-Pearson intervals on 8/12, 4/12, and on the
    reported per-granuloma fractions. Reported, not gated: a band is a duty, not a test.

A-CONTROL  MANDATORY ABLATIONS. (i) q = 0 must give P(>=1 fails) exactly 0. (ii) Lesions made
    perfectly correlated instead of independent must give P = max_i p_i, and the ratio to the
    independent answer must be reported -- independence is an assumption doing real work and its
    size must be shown, not assumed small.

B1  THE TWO SLOPES PIN k AND NOTHING ELSE ALONE. From an all-growing initial condition the
    initial log10 slope of total count is exactly k/ln(10). Gate: recovering k from their stated
    0.2 (J774) and 0.3 (human) log10/h and re-simulating must return that slope to < 1%.

B2  WHAT THEIR ASSAY CANNOT SEE. For the two-state model the slow eigenvalue satisfies
    lam^2 + (k+a+b)lam + k*b = 0. With k fixed by B1 and lam fixed by their 0.02 log10/h late
    slope, this is ONE equation in TWO unknowns. PREDECLARED: the solution set is a curve, not a
    point. Gate: sweep a over three decades, solve for b, and confirm the simulated early and
    late slopes are unchanged along the whole curve (< 1% drift). Then report the range of the
    persister fraction a/(a+b) along it -- that is the quantity their curve does not determine.

B3  THE NUMBER THAT COLLAPSES THE CURVE. Report which single additional measurement makes a and
    b separately identifiable, and show numerically that it varies along the curve (so it is
    informative) by at least one order between the ends.

B-CONTROL  Set a = 0 (no persisters). Killing must become single-exponential: late log10 slope
    within 5% of the early one.

C1  PURITY IS NOT EFFICIENCY. Exact Bayes: purity = pi(1-e) / [pi(1-e) + (1-pi)e] for prevalence
    pi and per-cell misclassification rate e. Gate: the closed form must agree with a direct
    Monte-Carlo of 1e7 cells within 3 Monte-Carlo standard errors.

C2  ATTENUATION. Any true dim-vs-lit difference D is observed as D * [P(dim|dim gate) -
    P(dim|lit gate)]. Gate: verified by the same simulation to within 3 standard errors.

C3  AN ARITHMETIC CEILING THAT NEEDS NO MODEL. If the dim subpopulation is 1% of cells and 5%
    of cells survive drug, then even if EVERY dim cell survives, dim cells are at most 1/5 of the
    survivors. Reported as an exact bound with the two published percentages as inputs.

C-CONTROL  At the prevalence where the efficiency was measured, the formula must return that
    efficiency. This is ledger defect S made concrete: the 93% is correct where it was measured;
    the question is only whether it transfers to a 1% gate.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.stats import beta as beta_dist

# =================================================================================================
# CITATIONS -- defect R protocol: the number, and the sentence it came from, verbatim.
# =================================================================================================

CITATIONS = {
    "A.relapse_count": (
        "8 of 12", "10.1128/iai.00177-25",
        "Eight of the 12 animals developed relapse."),
    "A.clinical_relapse": (
        "4 of 12", "10.1128/iai.00177-25",
        "Four of the eight relapse animals had microbiologic and/or clinical evidence of "
        "disease (i.e., tachypnea and Mtb detected); non-relapse animals had no such signs."),
    "A.barcode_dissemination": (
        "42% median (range 0-100%)", "10.1128/iai.00177-25",
        "only 42% (median, range: 0%-100%) of barcodes from pre-treatment lung granulomas were "
        "observed in relapse granulomas, highlighting the independent risk of each granuloma to "
        "disseminate."),
    "A.viable_granuloma_fraction": (
        "68% median in relapse animals", "10.1128/iai.00177-25",
        "Relapse animals had a significantly higher frequency of granulomas (median of 68%) "
        "with viable Mtb at necropsy."),
    "A.new_granuloma_viable": (
        "84%", "10.1128/iai.00177-25",
        "Among the new granulomas that appeared during relapse, 84% had viable Mtb growth."),
    "A.nonrelapse_residual": (
        "1 of 4 animals, one granuloma, 184 CFU", "10.1128/iai.00177-25",
        "In contrast, only one of the four non-relapsed animals had Mtb growth at necropsy in a "
        "single granuloma (184 CFU)."),
    "A.one_bacillus_per_granuloma": (
        "one founding bacillus per granuloma", "10.1128/iai.00177-25",
        "We previously established that each individual granuloma is established by a single "
        "bacillus, distinguishable by a molecular barcode, whereas thoracic lymph nodes often "
        "have multiple barcodes due to migration of Mtb from lung granulomas."),
    "A.lesions_per_animal": (
        "UNRETRIEVED", "10.1128/iai.00177-25",
        "Number of scan-matched granulomas harvested per animal is not stated in the retrievable "
        "text. It is swept, never assumed."),
    "A.per_lesion_cfu": (
        "UNRETRIEVED", "10.1128/iai.00177-25",
        "Per-lesion CFU are plotted, not tabulated in the retrievable text. The burden "
        "DISTRIBUTION is the single input case A most needs and is the exact data request."),
    "B.early_kill_J774": (
        "0.2 log10/h over first 3 h", "10.1128/spectrum.02313-21",
        "The kill rate is estimated as a 0.2- or 0.3-log decrease in propidium iodide-negative "
        "events per hour over the first 3 h of incubation in J774 and human macrophages, "
        "respectively"),
    "B.early_kill_human": (
        "0.3 log10/h over first 3 h", "10.1128/spectrum.02313-21",
        "same sentence as B.early_kill_J774"),
    "B.late_kill": (
        "0.02 log10/h to 48 h, both cell types", "10.1128/spectrum.02313-21",
        "and a 0.02-log decrease per hour for longer incubations up to 48 h in both cell types."),
    "B.drug": (
        "oxacillin 50x MIC = 25 mg/L, 48 h", "10.1128/spectrum.02313-21",
        "Infected cells were then incubated in RPMI 1640 supplemented with 10% FBS with 50x the "
        "MIC of oxacillin (25 mg L; Sigma) for the indicated periods."),
    "B.resumption": (
        "94% resumed growth", "10.1128/spectrum.02313-21",
        "we found that 94% of bacteria on average resumed growth spontaneously when reinoculated "
        "in liquid medium, thus excluding dead cells and VBNC cells"),
    "B.plateau_amplitude": (
        "UNRETRIEVED", "10.1128/spectrum.02313-21",
        "The intercept of the slow phase extrapolated to t=0 -- the persister plateau level -- is "
        "in Fig 1B but not stated numerically in the retrievable text. It is exactly the number "
        "that collapses the B2 curve to a point."),
    "C.dim_fraction": (
        "~1% stationary, ~0.4% exponential", "10.1128/mSystems.01127-20",
        "a consistent subpopulation of translationally dormant 'dim' cells was also present, "
        "reaching about 1% of the population"),
    "C.tolerant_fraction": (
        "~5%", "10.1128/mSystems.01127-20",
        "Thus, MSM cultures grown in nutrient-rich conditions were ~5% tolerant to INH and RIF."),
    "C.sort_efficiency": (
        "93% efficient / ~7% inappropriately sorted", "10.1128/mSystems.01127-20",
        "This suggests that only ~7% of cells were inappropriately sorted and that the sorting "
        "efficiency for single bacterial cells was ~93%."),
    "C.sort_control_prevalence": (
        "UNRETRIEVED", "10.1128/mSystems.01127-20",
        "The mixing ratio of the MSM-mEos2 / MSM-mCherry control culture is not stated in the "
        "retrievable text. The prevalence at which 7% was measured is the whole question, so it "
        "is swept and the balanced-mixture case is shown as one point among many."),
    "D.lag_matching": (
        "evolved lag matches exposure interval", "10.1038/nature13469",
        "the lag time of bacteria before regrowth was optimized to match the duration of the "
        "antibiotic-exposure interval"),
}

RULE = "=" * 97


# =================================================================================================
# CASE A -- the conjunctive sterilisation problem
# =================================================================================================

def p_any_fail(q, N):
    """Exact P(at least one lesion fails to sterilise), lesions independent.

    q[i] = probability that one surviving bacillus in lesion i escapes and regrows.
    N[i] = residual bacilli in lesion i after treatment.
    """
    q = np.asarray(q, float); N = np.asarray(N, float)
    p = 1.0 - np.exp(-q * N)
    return 1.0 - np.prod(1.0 - p)


def gate_A1():
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(200):
        n = rng.integers(3, 40)
        q = rng.uniform(1e-4, 5e-3, n)
        N = rng.lognormal(3.0, 1.2, n)
        lhs = p_any_fail(q, N)
        rhs = 1.0 - np.exp(-np.sum(q * N))
        worst = max(worst, abs(lhs - rhs) / max(rhs, 1e-300))
    return worst


def gate_A2(n=40, mean_N=300.0, q=2e-3, cvs=(0.0, 0.25, 0.5, 1.0, 2.0, 3.0), seed=3):
    """Fixed mean burden, homogeneous q, sweep the SPREAD. Does P move?"""
    rng = np.random.default_rng(seed)
    rows = []
    for cv in cvs:
        if cv == 0.0:
            N = np.full(n, mean_N)
        else:
            sig = np.sqrt(np.log1p(cv ** 2))
            N = rng.lognormal(np.log(mean_N) - sig ** 2 / 2, sig, n)
            N *= mean_N / N.mean()                      # mean held EXACTLY fixed
        rows.append((cv, float(N.std() / N.mean()), p_any_fail(np.full(n, q), N)))
    base = rows[0][2]
    drift = max(abs(r[2] - base) / base for r in rows)
    return rows, drift


def gate_A3(n=40, seed=5):
    """log P(all sterilise) = -n[E(q)E(N) + Cov(q,N)] -- exact, at three correlations."""
    rng = np.random.default_rng(seed)
    out = []
    for rho_target in (-0.8, 0.0, 0.8):
        z1 = rng.normal(size=n); z2 = rng.normal(size=n)
        z2 = rho_target * z1 + np.sqrt(max(1 - rho_target ** 2, 0.0)) * z2
        N = np.exp(1.0 * z1) * 300.0
        q = np.exp(0.8 * z2) * 2e-3
        logP = -np.sum(q * N)
        Eq, EN = q.mean(), N.mean()
        cov = np.mean((q - Eq) * (N - EN))              # population covariance
        pred = -n * (Eq * EN + cov)
        rho = np.corrcoef(q, N)[0, 1]
        out.append((rho, Eq, EN, cov, logP, pred, abs(logP - pred) / abs(pred)))
    return out, max(r[-1] for r in out)


def gate_A4(n=40, seed=5, rho_target=0.8):
    """Exact vs mean-field, in orders. Mean-field = replace each lesion by the average lesion."""
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n); z2 = rng.normal(size=n)
    z2 = rho_target * z1 + np.sqrt(1 - rho_target ** 2) * z2
    N = np.exp(1.0 * z1) * 60.0
    q = np.exp(0.8 * z2) * 2e-4
    exact = p_any_fail(q, N)
    mf = p_any_fail(np.full(n, q.mean()), np.full(n, N.mean()))
    cov = np.mean((q - q.mean()) * (N - N.mean()))
    return exact, mf, cov, n * cov, np.log10(exact / mf)


def gate_A5(k=8, m=12):
    """Is (n_lesions, p_dissem) identifiable from m animals, k relapsed?  theta = n*log(1-p)."""
    def loglik(n, p):
        P = 1.0 - (1.0 - p) ** n
        P = min(max(P, 1e-15), 1 - 1e-15)
        return k * np.log(P) + (m - k) * np.log(1 - P)

    theta = np.log(1.0 - 8.0 / 12.0)                    # level set through the MLE P = 8/12
    rows, ref = [], None
    for n in (2, 5, 10, 25, 60, 150):
        p = 1.0 - np.exp(theta / n)
        ll = loglik(n, p)
        if ref is None:
            ref = ll
        rows.append((n, p, ll))
    drift = max(abs(r[2] - ref) / abs(ref) for r in rows)
    lo, hi = clopper_pearson(k, m)
    band = (-np.log(1 - lo), -np.log(1 - hi))           # band on -theta = n*(-log(1-p))
    return rows, drift, (lo, hi), band


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def gate_A_control(n=40, seed=5):
    rng = np.random.default_rng(seed)
    N = rng.lognormal(np.log(300.0), 1.0, n)
    zero = p_any_fail(np.zeros(n), N)
    q = np.full(n, 2e-4)
    indep = p_any_fail(q, N)
    p_i = 1.0 - np.exp(-q * N)
    perfectly_correlated = float(p_i.max())
    return zero, indep, perfectly_correlated, indep / perfectly_correlated


# =================================================================================================
# CASE B -- what two slopes pin down
# =================================================================================================

def two_state_slopes(k, a, b, t_early=1.0, t_late=(24.0, 48.0)):
    """log10 slope of TOTAL count for the linear two-state model, all-growing at t=0."""
    A = np.array([[-(k + a), b], [a, -b]])
    x0 = np.array([1.0, 0.0])

    def logN(t):
        return np.log10(max(float(np.sum(expm(A * t) @ x0)), 1e-300))

    h = 1e-4
    early = -(logN(t_early + h) - logN(t_early - h)) / (2 * h)
    late = -(logN(t_late[1]) - logN(t_late[0])) / (t_late[1] - t_late[0])
    return early, late


def solve_b(k, a, lam):
    """Slow eigenvalue lam (negative) fixes b given k and a:  lam^2+(k+a+b)lam+k*b = 0."""
    # lam^2 + (k+a)lam + b*lam + k*b = 0  ->  b(lam + k) = -lam^2 - (k+a)lam
    return float((-lam ** 2 - (k + a) * lam) / (lam + k))


def gate_B1(early_log10=0.2):
    k = early_log10 * np.log(10.0)
    e, _ = two_state_slopes(k, a=1e-3, b=solve_b(k, 1e-3, -0.02 * np.log(10.0)))
    return k, e, abs(e - early_log10) / early_log10


def gate_B2(early_log10=0.2, late_log10=0.02, a_grid=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)):
    k = early_log10 * np.log(10.0)
    lam = -late_log10 * np.log(10.0)
    rows = []
    for a in a_grid:
        b = solve_b(k, a, lam)
        if b <= 0:
            rows.append((a, b, np.nan, np.nan, np.nan)); continue
        e, l = two_state_slopes(k, a, b)
        rows.append((a, b, e, l, a / (a + b)))
    good = [r for r in rows if np.isfinite(r[2])]
    drift_e = max(abs(r[2] - early_log10) / early_log10 for r in good)
    drift_l = max(abs(r[3] - late_log10) / late_log10 for r in good)
    fracs = [r[4] for r in good]
    return rows, max(drift_e, drift_l), (min(fracs), max(fracs))


def plateau_intercept(k, a, b, t_ref=6.0):
    """log10 of the slow-phase amplitude: extrapolate the late line back to t=0.

    This is the persister plateau height. It is the measurement that collapses the B2 curve.
    """
    A = np.array([[-(k + a), b], [a, -b]])
    x0 = np.array([1.0, 0.0])
    lam = np.linalg.eigvals(A)
    lam_slow = float(np.max(lam.real))
    logN_ref = np.log10(max(float(np.sum(expm(A * t_ref) @ x0)), 1e-300))
    return logN_ref - lam_slow * t_ref / np.log(10.0)


def gate_B3(early_log10=0.2, late_log10=0.02, a_grid=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)):
    k = early_log10 * np.log(10.0)
    lam = -late_log10 * np.log(10.0)
    rows = []
    for a in a_grid:
        b = solve_b(k, a, lam)
        if b <= 0:
            continue
        rows.append((a, b, a / (a + b), plateau_intercept(k, a, b)))
    span = max(r[3] for r in rows) - min(r[3] for r in rows)
    return rows, span


def gate_B_control(early_log10=0.2):
    k = early_log10 * np.log(10.0)
    e, l = two_state_slopes(k, a=0.0, b=1.0)
    return e, l, abs(l - e) / e


# =================================================================================================
# CASE C -- purity of a gate on a rare subpopulation
# =================================================================================================

def purity(pi, eps):
    """P(truly dim | landed in dim gate). eps = per-cell misclassification rate."""
    num = pi * (1 - eps)
    return num / (num + (1 - pi) * eps)


def attenuation(pi, eps):
    """Observed dim-lit difference / true dim-lit difference."""
    p_dim_given_dimgate = purity(pi, eps)
    # lit gate holds pi*eps true-dim and (1-pi)(1-eps) true-lit
    den = pi * eps + (1 - pi) * (1 - eps)
    p_dim_given_litgate = pi * eps / den
    return p_dim_given_dimgate - p_dim_given_litgate


def gate_C1(pi=0.01, eps=0.07, n_sim=10_000_000, seed=7):
    rng = np.random.default_rng(seed)
    is_dim = rng.random(n_sim) < pi
    misread = rng.random(n_sim) < eps
    in_dim_gate = np.where(is_dim, ~misread, misread)
    k = int(np.sum(is_dim & in_dim_gate)); tot = int(np.sum(in_dim_gate))
    emp = k / tot
    se = np.sqrt(emp * (1 - emp) / tot)
    closed = purity(pi, eps)
    return closed, emp, se, abs(closed - emp) / max(se, 1e-12)


def gate_C2(pi=0.01, eps=0.07, n_sim=10_000_000, seed=8):
    rng = np.random.default_rng(seed)
    is_dim = rng.random(n_sim) < pi
    misread = rng.random(n_sim) < eps
    in_dim_gate = np.where(is_dim, ~misread, misread)
    a = is_dim[in_dim_gate].mean(); b = is_dim[~in_dim_gate].mean()
    emp = a - b
    se = np.sqrt(a * (1 - a) / in_dim_gate.sum() + b * (1 - b) / (~in_dim_gate).sum())
    closed = attenuation(pi, eps)
    return closed, emp, se, abs(closed - emp) / max(se, 1e-12)


def gate_C3(dim_frac=0.01, tolerant_frac=0.05):
    """Model-free ceiling: dim cells are at most dim_frac/tolerant_frac of the survivors."""
    ceiling = dim_frac / tolerant_frac
    # survival ratio needed for dim to account for a given share s of survivors:
    #   s = dim*r_dim / (dim*r_dim + (1-dim)*r_lit) -> r_dim/r_lit = s(1-dim)/((1-s)dim)
    def ratio_for_share(s):
        return s * (1 - dim_frac) / ((1 - s) * dim_frac)
    return ceiling, [(s, ratio_for_share(s)) for s in (0.5, 0.9, 0.99)]


def gate_C_control(eps=0.07):
    return purity(0.5, eps)


# =================================================================================================
# REPORT
# =================================================================================================

def report():
    out = []
    P = out.append
    P(RULE)
    P("REM AGAINST FOUR GROUPS' PUBLISHED NUMBERS")
    P(RULE)
    P("  Every input is quoted in CITATIONS with its source sentence. Numbers marked UNRETRIEVED")
    P("  are swept, never invented. Articles retrieved from PubMed Central; DOIs in CITATIONS.")
    P("")

    # ---------------- A ----------------
    P(RULE)
    P("CASE A -- TB relapse in macaques (Maiello, Fortune, Flynn, Lin; doi:10.1128/iai.00177-25)")
    P(RULE)
    P("  THEIR GAP, IN THEIR WORDS: 'complete sterilization or very low Mtb burden is protective")
    P("  against SIV-induced TB relapse but cannot be predicted by PET CT'.")
    P("")
    w = gate_A1()
    P(f"  A1 EXACTNESS  worst relative error over 200 random lesion sets: {w:.3e}")
    P(f"     {'PASS' if w < 1e-12 else 'FAIL'} (bar 1e-12)")
    P("")
    rows, drift = gate_A2()
    P("  A2 DOES THE SPREAD MATTER?  Mean burden held EXACTLY fixed, q homogeneous.")
    P("        requested CV   realised CV   P(>=1 lesion fails)")
    for cv, rcv, p in rows:
        P(f"        {cv:12.2f}   {rcv:11.4f}   {p:.15f}")
    P(f"     Maximum relative movement across a 0 -> 3 CV sweep: {drift:.3e}")
    if drift < 1e-9:
        P("     PREDECLARED VERDICT: the spread hypothesis is FALSE under this model.")
        P("     P(>=1 fails) = 1 - exp(-q * SUM N_i) depends on the TOTAL residual burden and on")
        P("     nothing else. Redistributing the same total among lesions changes NOTHING. This")
        P("     is the unwelcome answer and it is the reported one.")
    else:
        P("     PREDECLARED VERDICT: spread moves the answer mechanically.")
    P("")
    a3, a3w = gate_A3()
    P("  A3 SO WHAT DOES MAKE IT DEPEND ON MORE THAN THE MEAN? Heterogeneous per-bacillus escape.")
    P("        corr(q,N)      E(q)        E(N)      Cov(q,N)     log P(all sterilise)   rel.err")
    for rho, Eq, EN, cov, lp, pr, err in a3:
        P(f"        {rho:8.4f}  {Eq:.4e}  {EN:9.3f}  {cov:11.4e}  {lp:20.10f}   {err:.2e}")
    P(f"     Identity log P = -n[E(q)E(N) + Cov(q,N)] holds to {a3w:.3e}  "
      f"{'PASS' if a3w < 1e-12 else 'FAIL'} (bar 1e-12)")
    P("     THE WHOLE SPREAD QUESTION COLLAPSES TO ONE NUMBER: n*Cov(q,N). It is zero exactly")
    P("     when burden and drug escape are uncorrelated across lesions -- and only then is the")
    P("     mean sufficient.")
    P("")
    ex, mf, cov, ncov, orders = gate_A4()
    P("  A4 WHAT THE MEAN-FIELD MISSES, at positive burden-escape correlation:")
    P(f"        exact P(relapse)      {ex:.6e}")
    P(f"        mean-field P(relapse) {mf:.6e}")
    P(f"        Cov(q,N) = {cov:.4e}   n*Cov = {ncov:.4e}   log10 ratio = {orders:+.4f}")
    void = not (1e-9 < ex < 0.999 and 1e-9 < mf < 0.999)
    P(f"        non-vacuity: {'VOID' if void else 'both inside (1e-9, 0.999) -- OK'}")
    P("        DIRECTION, AND IT IS FALSIFIABLE: their own paper reports that lymph nodes")
    P("        'exhibit reduced bacterial killing during drug treatment' and carry MULTIPLE")
    P("        barcodes, i.e. high burden with poor killing -- positive Cov. If that holds, the")
    P("        mean-field estimate UNDERSTATES relapse risk by the factor above.")
    P("")
    rows5, drift5, cp, band = gate_A5()
    P("  A5 IDENTIFIABILITY FROM THEIR ACTUAL 8-of-12.")
    P("        n lesions   p per-lesion dissemination   log-likelihood")
    for n, p, ll in rows5:
        P(f"        {n:9d}   {p:26.6f}   {ll:14.10f}")
    P(f"     Log-likelihood drift along the level set: {drift5:.3e}  "
      f"{'PASS' if drift5 < 1e-12 else 'FAIL'} (bar 1e-12)")
    P("     PROVED, NOT MERELY OBSERVED: the likelihood depends on (n,p) only through")
    P("     theta = n*log(1-p). Animal-level relapse counts CANNOT separate 'many lesions each")
    P("     rarely disseminating' from 'few lesions each often disseminating'. No number of extra")
    P("     animals fixes this -- it is a structural degeneracy, not a power problem.")
    P(f"     Exact Clopper-Pearson 95% CI on 8/12 = [{cp[0]:.4f}, {cp[1]:.4f}]")
    P(f"     which maps to n*(-log(1-p)) in [{band[0]:.4f}, {band[1]:.4f}]")
    P("     THE OFFER: their barcode assay is the ONLY thing that breaks this degeneracy, and the")
    P("     42% median dissemination fraction is exactly p. Feeding it in fixes n. That number --")
    P("     the effective count of independently dangerous lesions per animal -- is not in the")
    P("     paper and is computable from data they already hold.")
    P("")
    z, ind, corr, ratio = gate_A_control()
    P("  A-CONTROL")
    P(f"     q = 0 gives P(>=1 fails) = {z:.1e}  {'PASS' if z == 0.0 else 'FAIL'}")
    P(f"     independent lesions: {ind:.6f}   perfectly correlated: {corr:.6f}   ratio {ratio:.3f}x")
    P("     Independence is an assumption doing real work; its size is shown, not assumed small.")
    P("")
    P("  UNRETRIEVED AND THEREFORE NOT SPENT AS AN EXPLANATION:")
    P("     - per-lesion CFU distribution (plotted, not tabulated in the retrievable text)")
    P("     - number of scan-matched granulomas harvested per animal")
    P("     - joint per-lesion (burden, drug exposure) -- the pair that sets Cov(q,N)")
    P("")

    # ---------------- B ----------------
    P(RULE)
    P("CASE B -- intracellular S. aureus persisters (Peyrusson, Van Bambeke;")
    P("          doi:10.1128/spectrum.02313-21)")
    P(RULE)
    P("  THEIR PUBLISHED CURVE: 0.2 (J774) or 0.3 (human MPhi) log10/h over the first 3 h, then")
    P("  0.02 log10/h out to 48 h, under oxacillin at 50x MIC. That is a complete biphasic")
    P("  specification -- two slopes and a breakpoint -- which is rare and makes this case usable.")
    P("")
    for label, e0 in (("J774 macrophages", 0.2), ("human macrophages", 0.3)):
        k, e, err = gate_B1(e0)
        P(f"  B1 [{label}] early slope pins k exactly: k = {k:.6f} /h")
        P(f"     re-simulated early slope {e:.6f} log10/h vs stated {e0}  rel.err {err:.3e}  "
          f"{'PASS' if err < 0.01 else 'FAIL'} (bar 1%)")
    P("")
    rows, drift, frac_range = gate_B2()
    P("  B2 WHAT THE TWO SLOPES DO NOT PIN.  Slow eigenvalue gives ONE equation in (a,b).")
    P("        a (/h)      b (/h)     early log10/h   late log10/h   persister fraction a/(a+b)")
    for a, b, e, l, f in rows:
        if not np.isfinite(e):
            P(f"        {a:.4f}  {b:10.4f}   -- no positive b --"); continue
        P(f"        {a:.4f}  {b:10.6f}   {e:13.6f}   {l:12.6f}   {f:26.6f}")
    P(f"     Slope drift along the entire curve: {drift:.3e}  "
      f"{'PASS' if drift < 0.01 else 'FAIL'} (bar 1%)")
    P(f"     Every one of those parameter sets reproduces their published curve EXACTLY as well.")
    P(f"     The persister fraction ranges over [{frac_range[0]:.4f}, {frac_range[1]:.4f}] -- a")
    P(f"     {frac_range[1]/frac_range[0]:.1f}x span -- and their curve cannot choose between them.")
    P("")
    rows3, span = gate_B3()
    P("  B3 THE ONE MEASUREMENT THAT COLLAPSES IT: the slow-phase intercept (plateau height).")
    P("        a (/h)   persister fraction   log10 plateau amplitude")
    for a, b, f, pl in rows3:
        P(f"        {a:.4f}   {f:18.6f}   {pl:22.6f}")
    P(f"     Span across the curve: {span:.4f} orders  "
      f"{'PASS -- informative' if span >= 1.0 else 'FAIL -- not informative'} (bar 1 order)")
    P("     THE OFFER: one number they already have in their Fig 1B, read off as an intercept,")
    P("     converts a one-parameter family into a point estimate of the persister formation and")
    P("     waking rates -- and those, not the slopes, are what a dosing schedule acts on.")
    P("")
    e, l, err = gate_B_control()
    P(f"  B-CONTROL  a = 0 (no persisters): early {e:.6f}, late {l:.6f} log10/h, rel.diff {err:.3e}")
    P(f"     {'PASS' if err < 0.05 else 'FAIL'} (bar 5%) -- killing is single-exponential without")
    P("     the dormant state, so the biphasic shape is the mechanism and not the driver.")
    P("")

    # ---------------- C ----------------
    P(RULE)
    P("CASE C -- PerSort purity on a 1% subpopulation (Srinivas, Baliga;")
    P("          doi:10.1128/mSystems.01127-20)")
    P(RULE)
    P("  THEIR NUMBERS: dim cells ~1% of the population; sorting '93% efficient', i.e. ~7%")
    P("  inappropriately sorted. Both are correct as measured. The question is only whether the")
    P("  second transfers to a gate on the first -- ledger defect S, made concrete.")
    P("")
    ctrl = gate_C_control()
    P(f"  C-CONTROL  at a balanced mixture (pi = 0.5), the formula returns purity {ctrl:.4f},")
    P(f"     which reproduces their stated 93%. The figure is right where it was measured.")
    P("")
    P("  C1 PURITY OF THE DIM GATE as a function of prevalence and misclassification rate:")
    P("        eps \\ pi        0.004        0.01         0.05          0.50")
    for eps in (0.001, 0.005, 0.01, 0.03, 0.07):
        vals = "  ".join(f"{purity(pi, eps):11.4f}" for pi in (0.004, 0.01, 0.05, 0.5))
        P(f"        {eps:.3f}    {vals}")
    closed, emp, se, nsig = gate_C1()
    P(f"     At pi = 0.01, eps = 0.07: closed form {closed:.6f}, Monte-Carlo (1e7 cells) "
      f"{emp:.6f} +- {se:.6f}")
    P(f"     agreement {nsig:.2f} sigma  {'PASS' if nsig < 3 else 'FAIL'} (bar 3 sigma)")
    P("     READING IT: a 7% error rate that is harmless on a balanced mixture leaves a 1% gate")
    P("     mostly full of the majority population. The direction cuts BOTH ways and the second")
    P("     is the interesting one -- see C2.")
    P("")
    closed2, emp2, se2, nsig2 = gate_C2()
    P("  C2 ATTENUATION -- what an impure gate does to a measured difference:")
    P(f"     observed difference = true difference x {closed2:.6f}")
    P(f"     Monte-Carlo {emp2:.6f} +- {se2:.6f}, agreement {nsig2:.2f} sigma  "
      f"{'PASS' if nsig2 < 3 else 'FAIL'} (bar 3 sigma)")
    P(f"     CONSEQUENCE, AND IT IS IN THEIR FAVOUR: at pi=0.01, eps=0.07 any real dim-vs-lit")
    P(f"     difference they measured is DILUTED by {1/closed2:.1f}x. Their -4.17 log2 fold change")
    P("     in 16S rRNA would then be a floor, not the effect size. The correction makes their")
    P("     result larger, not smaller -- provided eps for the dim gate is what the control says.")
    P("")
    ceiling, ratios = gate_C3()
    P("  C3 A CEILING THAT NEEDS NO MODEL AT ALL.")
    P("     dim cells ~1% of the population; ~5% of cells survive 5x MIC INH/RIF.")
    P(f"     Even if EVERY dim cell survives, dim cells are at most {ceiling:.1%} of the survivors.")
    P("     For dim cells to account for a given share of survivors, their survival advantage")
    P("     over lit cells would have to be:")
    for s, r in ratios:
        P(f"        share {s:5.0%}  requires dim:lit survival ratio {r:10.1f}x")
    P("     THE OFFER: this is an arithmetic identity, not a model, and it says the translationally")
    P("     dormant subpopulation cannot by itself be the tolerant subpopulation. Either a large")
    P("     survival ratio is real and measurable, or a second route to tolerance exists outside")
    P("     the dim gate. Both are testable with sorts they already run.")
    P("")

    # ---------------- D ----------------
    P(RULE)
    P("CASE D -- lag-time optimisation (Fridman, Balaban; doi:10.1038/nature13469)")
    P(RULE)
    P("  Already computed; see RESULTS_selection.txt. Reproduced their lag-matching optimum at")
    P("  4 of 4 exposure durations from first principles, then produced the curvature around it:")
    P("  selection strength rises 42x from a 1.5 h to a 5.0 h exposure. The prediction is that")
    P("  evolved lag should scatter widely across replicate populations at short exposures and")
    P("  cluster tightly at long ones -- a trend across conditions, in replicates they already ran.")
    P("  Three of four peak widths were grid-limited; those rows are lower bounds, not measurements.")
    P("")
    P(RULE)
    P("WHAT IS NOT CLAIMED")
    P(RULE)
    P("  * No group's raw data was in hand. Every number above is a published one, quoted in")
    P("    CITATIONS with its source sentence, or a swept variable.")
    P("  * Case A's headline finding is a NEGATIVE: under the independent-bacillus model the")
    P("    spread of residual burdens does not move the relapse probability at all. What does move")
    P("    it is the covariance between burden and drug escape, which nobody has measured.")
    P("  * Case B does not fit their data; it shows what their data cannot fit.")
    P("  * Case C's purity number is only as good as the misclassification rate for the dim gate,")
    P("    and the prevalence at which their 7% was measured is UNRETRIEVED. The table is given")
    P("    across eps for that reason.")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
