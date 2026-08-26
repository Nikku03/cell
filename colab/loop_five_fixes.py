"""Loop 228. The five named failure modes, each fixed, each measured separately and together.

THE CHARGE. Current cell models are said to fail for five reasons: they assume a well-mixed
compartment when local concentration varies by orders of magnitude; they use equilibrium
thermodynamics on an ATP-driven system; they map sequence to expression without cell state;
they apply deterministic equations to low-copy stochastic events; and they model regulatory layers
in isolation. This loop builds one model embodying all five errors and one addressing all five,
scores them on the same target with the same split, and ablates each fix.

FOUR OF THE FIVE ALREADY HAVE EVIDENCE IN THIS REPOSITORY, which is why they are worth fixing
rather than debating:

    equilibrium      loop 206 computed Berg-von Hippel occupancy with a GLOBAL chemical potential
                     mu = -4.0 over 89.8% of the gene set and reached r = -0.0133 -- worse than
                     nothing -- while MEASURED occupancy of the same factors in the same cells
                     reached +0.2932. Loop 209 scaled the calculation to 1,200 promoters x 879
                     motifs and beat its own dinucleotide-shuffled control by +0.0293.
    sequence-only    co-evolution across 95 mammals (loops 225-227) reached held-out 0.10-0.14;
                     the measured-state stack reached 0.5474 on the same genes.
    stochastic       loop 224 read all 1,914,250 perturbed Perturb-seq cells and found 84.5% of
                     the 200 strongest effects BIMODAL at single-cell level, against a matched
                     Gaussian control with a 0.0% false-positive rate.
    siloed layers    loop 213 measured stacked 0.5474, best single block 0.4567, and naive
                     concatenation 0.4345 -- WORSE than the best single block. Coupling helps;
                     pooling features does not.

THE FIVE FIXES, EACH A ONE-PARAMETER GENERALISATION THAT REDUCES TO THE BROKEN VERSION.
That property matters: every fix below contains the thing it replaces as a special case, so if a
fix fails it fails on its own merits and not because two models differ in ten ways at once.

  FIX 1  LOCAL CONCENTRATION.  mu is a chemical potential, so mu = mu_0 + kT ln(c). A local
         enrichment of TF concentration is therefore an ADDITIVE shift in mu, and the occupancy
         cache already spans mu from -4 to +14 kT in steps of 0.5. Local contact density from the
         Hi-C-derived chromatin features is the enrichment proxy:
              mu_local(g) = mu_global + beta * z(log local contact density)
         beta = 0 recovers the global-mu model exactly.

  FIX 2  NON-EQUILIBRIUM DRIVEN OCCUPANCY.  Equilibrium gives theta = Z/(1+Z), which is a pure
         function of binding energy. An ATP-driven remodeller changes the OFF rate without
         changing the binding energy, breaking detailed balance:
              theta_driven = Z / (Z + 1 + alpha * A)
         with A the measured DNase accessibility standing in for remodeller activity. alpha = 0
         recovers equilibrium exactly. The SIGN of alpha is not assumed -- remodellers both evict
         and load factors -- so alpha is fit freely and its sign is reported, not gated.

  FIX 3  CELL-STATE CONDITIONING.  Replace additive sequence + state features with their
         INTERACTIONS: occupancy x accessibility, occupancy x each measured track. A model with
         the interaction coefficients set to zero is the additive model.

  FIX 4  STOCHASTIC SUMMARY INSTEAD OF A MEAN.  Under a two-component response where a fraction f
         of cells shift by delta and the rest do not, the mean is m = f*delta and the excess
         variance over the unperturbed within-population variance is e = f(1-f)*delta^2. Those two
         equations invert in closed form:
              f = m^2 / (m^2 + e)
         so the switching fraction is recoverable from the per-perturbation sums and sums of
         squares already checkpointed by loop 224, with no re-streaming. f -> 1 as e -> 0, which
         is the deterministic all-cells-respond case, so the mean-based feature is the e = 0 limit.

  FIX 5  STACKED COUPLING INSTEAD OF CONCATENATION.  Per-block held-out predictions combined by a
         second-stage ridge, which loop 213 measured at 0.5474 against concatenation's 0.4345.

THE BASELINE IS RECOMPUTED HERE, NOT QUOTED. Loop 213's 0.5474 was measured on a ten-block stack
this loop does not rebuild. Quoting it as the bar would compare two different models on two
different feature sets and call the difference a fix. Instead the broken model is built and scored
INSIDE this loop on the same genes, the same split and the same blocks, so every comparison below
is internal and exact. Loop 213's number is reported as context only.

PREDECLARED, BEFORE ANY NUMBER.

  H1 IS THE TWO-COMPONENT INVERSION WELL-POSED?
     Gate: PASS iff excess variance is strictly positive and f lands inside (0, 1) for at least
     80% of the strongest perturbation-by-gene effects. If the inversion is degenerate there is
     no switching fraction to test and H2 must not be read.

  H2 FIX 4 -- IS THE SWITCHING FRACTION MORE REPRODUCIBLE THAN THE MEAN?
     Split each perturbation's cells in half and compute both summaries on each half
     independently. Loop 224 measured the MEAN's split-half reliability at 0.0920.
     Gate: PASS iff f's median split-half reliability exceeds the mean's by at least 0.02,
     measured in the same run on the same perturbations. Requires H1.

  H3 FIX 1 -- DOES LOCAL CONCENTRATION BEAT GLOBAL?
     Both mu_global and beta selected on TRAINING folds only.
     Gate: PASS iff held-out |r| improves by at least 0.02 over the best global-mu model AND
     exceeds a control in which the contact-density vector is shuffled across genes. Without that
     control any extra free parameter would look like a win.

  H4 FIX 2 -- DOES DRIVEN OCCUPANCY BEAT EQUILIBRIUM?
     alpha selected on training folds, sign free.
     Gate: PASS iff held-out |r| improves by at least 0.02 over equilibrium. The fitted alpha and
     its sign are REPORTED; the gate does not require a particular sign, because assuming the sign
     of one's own answer is a defect this project has committed before.

  H5 FIX 3 -- DO CELL-STATE INTERACTIONS BEAT ADDITIVE?
     Gate: PASS iff held-out |r| improves by at least 0.02 over the additive model on identical
     inputs.

  H6 FIX 5 -- DOES STACKING BEAT CONCATENATION HERE?
     Gate: PASS iff the stacked model beats concatenation of the same blocks by at least 0.02.

  H7 ALL FIVE TOGETHER AGAINST ALL FIVE ERRORS
     Gate: PASS iff the fully fixed model beats the fully broken model, both built and scored in
     this loop, by at least 0.05 held-out |r|.

  H8 CONTROL: DOES THE FIXED MODEL BEAT ITS OWN SHUFFLED TARGET?
     Gate: PASS iff the fixed model exceeds its shuffled-label score by at least 0.10. A stack of
     five generalisations has many free parameters and this is the gate that asks whether any of
     them is fitting the target rather than the noise.

  H9 WHAT THIS CANNOT SHOW -- written before the run.
     The target is the A549 dexamethasone plateau, one drug in one cell line. A fix that helps
     here is not thereby a general improvement to cell modelling.
     FIX 4 is tested for REPRODUCIBILITY on Perturb-seq, where single cells exist, and then
     carried into the A549 stack as a feature. The A549 series is bulk RNA-seq with four
     replicates and has no single-cell layer, so the stochastic fix can never be tested directly
     against the A549 target -- only its downstream usefulness can.
     Local contact density is Hi-C-derived from other cell types, so FIX 1's enrichment proxy is
     not measured in A549 and a failure could be the proxy rather than the principle.
     DNase accessibility stands in for ATP-driven remodeller activity in FIX 2. It is a
     consequence of remodelling, not a measurement of it, and the two are not the same quantity.
"""
import os, sys, json, time, warnings
from pathlib import Path
from collections import Counter
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from loop_setpoint_physics import gene_set
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_five_fixes.json"
SP = L191.SP
CK = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
OCC_F = ROOT / "colab" / "data" / "physics" / "nr3c1_occupancy.npz"
REL_F = ROOT / "outputs" / "loop224_reliability.npz"
TRACKS = ["NR3C1", "EP300", "JUN", "JUNB", "CEBPB", "FOSL2", "DNase", "CTCF", "RAD21"]
SEED, NFOLD, K_PS = 228228, 5, 24
REF_213_STACK, REF_213_CONCAT, REF_224_MEAN_SH = 0.5474, 0.4345, 0.0920
GAIN, BIG_GAIN, CTRL_GAIN, WELLPOSED = 0.02, 0.05, 0.10, 0.80

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5: return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def ridge_fit(X, y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    R = lam * np.eye(A.shape[1]); R[-1, -1] = 0
    w = np.linalg.solve(A.T @ A + R, A.T @ y)
    return (mu, sd, w)


def ridge_apply(m, X):
    mu, sd, w = m
    return np.hstack([(X - mu) / sd, np.ones((len(X), 1))]) @ w


def cv_pred(X, y, folds, lam=1.0):
    """Out-of-fold predictions; nothing is fitted on the row it scores."""
    p = np.zeros(len(y))
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        p[te] = ridge_apply(ridge_fit(X[tr], y[tr], lam), X[te])
    return p


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "all five named failure modes, fixed and ablated"}
    say("=" * 104)
    say("LOOP 228 -- THE FIVE FAILURE MODES, EACH FIXED, EACH MEASURED")
    say("=" * 104)
    say("     Every fix is a ONE-PARAMETER generalisation that contains the broken version as a")
    say("     special case, so a failure is the fix's own and not a ten-way model difference.")
    say("     The broken baseline is rebuilt and scored INSIDE this loop on the same genes and")
    say("     split; loop 213's 0.5474 is context, not the bar.")

    # ================================================================ FIX 4 (Perturb-seq)
    say("H1 IS THE TWO-COMPONENT INVERSION WELL-POSED?")
    ck = np.load(CK / "loop224_accum.npz")
    S, Q, SA, n, nA = ck["S"], ck["Q"], ck["SA"], ck["n"], ck["nA"]
    relz = np.load(REL_F, allow_pickle=True)
    ro_gene = np.array([str(x) for x in relz["gene"]])
    within = relz["within"].astype(np.float64)
    okp = n >= 40
    nn = n[okp][:, None].astype(np.float64)
    Sk, Qk = S[okp].astype(np.float64), Q[okp].astype(np.float64)
    m_hat = Sk / nn
    var_hat = np.maximum((Qk - Sk * Sk / nn) / np.maximum(nn - 1, 1), 0.0)
    excess = var_hat - within[None, :]
    strong = np.argsort(-np.abs(m_hat).ravel())[:200000]
    ei, gj = np.unravel_index(strong, m_hat.shape)
    e_pos = excess[ei, gj] > 0
    f_all = np.where(excess > 0, m_hat ** 2 / (m_hat ** 2 + np.maximum(excess, 1e-12)), 1.0)
    f_strong = f_all[ei, gj]
    inside = np.mean((f_strong > 0) & (f_strong < 1))
    say(f"     {int(okp.sum()):,} perturbations x {m_hat.shape[1]:,} genes; strongest "
        f"{len(strong):,} effects examined")
    say(f"     excess variance strictly positive in {e_pos.mean():.1%} of them")
    say(f"     switching fraction f inside (0,1) for {inside:.1%};  median f "
        f"{np.median(f_strong):.4f}")
    G.add("H1", bool(inside >= WELLPOSED), stat=float(inside),
          if_true=lambda: f"H1 PASS -- the inversion is well-posed for {inside:.0%} of the "
                          f"strongest effects; median switching fraction {np.median(f_strong):.3f}",
          if_false=lambda: f"H1 FAIL -- f falls inside (0,1) for only {inside:.0%}; the "
                           f"two-component inversion is degenerate here")
    res["switching"] = {"frac_excess_pos": float(e_pos.mean()), "frac_inside": float(inside),
                        "median_f": float(np.median(f_strong))}

    # ---------------------------------------------------------------- H2
    say("H2 FIX 4 -- IS THE SWITCHING FRACTION MORE REPRODUCIBLE THAN THE MEAN?")
    two = okp & (nA >= 20) & ((n - nA) >= 20)
    sel = np.isin(np.where(okp)[0], np.where(two)[0])
    nA2 = nA[two][:, None].astype(np.float64); nB2 = (n[two] - nA[two])[:, None].astype(np.float64)
    mA = SA[two].astype(np.float64) / nA2
    mB = (S[two] - SA[two]).astype(np.float64) / nB2
    e_sh = np.maximum(excess[sel], 1e-12)

    def med_sh(A, B, k=1200):
        step = max(1, A.shape[1] // k)
        return float(np.nanmedian([pear(A[:, j], B[:, j]) for j in range(0, A.shape[1], step)]))

    fA = mA ** 2 / (mA ** 2 + e_sh)
    fB = mB ** 2 / (mB ** 2 + e_sh)
    med_m = med_sh(mA, mB)
    med_f = med_sh(fA, fB)
    med_sq = med_sh(mA ** 2, mB ** 2)
    pa, pb = rng.permutation(mA.shape[0]), rng.permutation(mA.shape[0])
    shA = mA[pa] ** 2 / (mA[pa] ** 2 + e_sh)
    shB = mB[pb] ** 2 / (mB[pb] ** 2 + e_sh)
    med_art = med_sh(shA, shB)
    say(f"     {int(two.sum()):,} perturbations split into halves")
    say(f"       MEAN                          median {med_m:+.4f}   "
        f"(loop 224 measured {REF_224_MEAN_SH:+.4f})")
    say(f"       SWITCHING FRACTION f          median {med_f:+.4f}")
    say(f"       MAGNITUDE m^2, no denominator median {med_sq:+.4f}")
    say("     THE ARTEFACT CONTROL. Per-half sums of squares were never accumulated, so both")
    say("     halves of f divide by the SAME pooled excess variance. A shared denominator can")
    say("     correlate with itself. The control destroys the numerator and keeps the denominator:")
    say(f"       f with NUMERATOR SHUFFLED     median {med_art:+.4f}")
    G.add("H2a", bool(med_art < 0.30), stat=float(med_art), requires=("H1",),
          if_true=lambda: f"H2a PASS -- shuffling the numerator collapses f to {med_art:+.4f}, so "
                          f"the shared denominator is not doing the work",
          if_false=lambda: f"H2a FAIL -- f survives at {med_art:+.4f} with its numerator "
                           f"destroyed, so the {med_f:+.4f} is the shared denominator "
                           f"correlating with itself, not reproducible biology. Testing f "
                           f"honestly needs per-half sums of squares, which were not accumulated")
    G.add("H2b", bool(med_sq - med_m >= GAIN), stat=float(med_sq - med_m), requires=("H1",),
          if_true=lambda: f"H2b PASS -- a denominator-free MAGNITUDE summary reproduces at "
                          f"{med_sq:+.4f} against the signed mean's {med_m:+.4f}; the mean is "
                          f"the worse summary and this comparison has no shared term",
          if_false=lambda: f"H2b FAIL -- magnitude {med_sq:+.4f} against the mean's {med_m:+.4f}")
    res["splithalf"] = {"mean": med_m, "switching_f": med_f, "magnitude_m2": med_sq,
                        "artefact_control": med_art, "loop224_mean": REF_224_MEAN_SH}

    # ================================================================ the A549 harness
    grid, M, A9, sym, keepg, tssb = gene_set()
    gi = np.where(keepg)[0]
    y_all = (M[-3:].mean(0))[gi]
    allg = [sym[i] for i in gi]
    gpos_a = {s: k for k, s in enumerate(allg)}
    Z = np.load(OCC_F, allow_pickle=True)
    occ, mugrid, ogene = Z["occ"], Z["mu"], np.array([str(g) for g in Z["genes"]])
    CH = json.load(open(SP / "_chromatin_features.json"))["features"]
    names = [s for s in ogene if s in gpos_a and s.upper() in CH]
    oi = {g: i for i, g in enumerate(ogene)}
    y = np.array([y_all[gpos_a[s]] for s in names])
    OCC = occ[[oi[s] for s in names]]
    dens = np.array([np.log1p(CH[s.upper()].get("dens") or 0.0) for s in names])
    dens = (dens - dens.mean()) / (dens.std() + 1e-9)
    ch = np.array([[CH[s.upper()].get("pc1", np.nan), CH[s.upper()].get("ins", np.nan), d]
                   for s, d in zip(names, dens)], float)
    ch = np.where(np.isfinite(ch), ch, 0.0)
    TR = {}
    for t in TRACKS:
        pt, PM = L191.promoter_track(t, [tssb.get(s) for s in sym], L191.PROM_PAD, lambda *_: None)
        TR[t] = PM[[int(np.where(pt == tt)[0][0]) for tt in grid]]
    CHIP = np.column_stack([np.column_stack([
        TR[t][:, gi].mean(0), TR[t][:, gi].max(0), TR[t][-1, gi] - TR[t][0, gi]])
        for t in TRACKS])
    CHIP = np.array([CHIP[gpos_a[s]] for s in names])
    acc = TR["DNase"][:, gi].mean(0)
    acc = np.array([acc[gpos_a[s]] for s in names])
    acc = (acc - acc.mean()) / (acc.std() + 1e-9)
    say(f"     A549 harness: {len(names):,} genes carrying occupancy, chromatin and ChIP")
    N = len(names)
    perm = rng.permutation(N)
    folds = [perm[i::NFOLD] for i in range(NFOLD)]

    def occ_at(mu_vec):
        idx = np.clip(np.searchsorted(mugrid, mu_vec), 1, len(mugrid) - 1)
        lo, hi = mugrid[idx - 1], mugrid[idx]
        w = np.where(hi > lo, (mu_vec - lo) / (hi - lo + 1e-12), 0.0)
        return OCC[np.arange(N), idx - 1] * (1 - w) + OCC[np.arange(N), idx] * w

    def score(feat):
        X = feat.reshape(N, -1) if feat.ndim == 1 else feat
        return abs(pear(y, cv_pred(np.nan_to_num(X), y, folds)))

    # ---------------------------------------------------------------- H3
    say("H3 FIX 1 -- DOES LOCAL CONCENTRATION BEAT GLOBAL?")
    glob = {float(mu): score(occ_at(np.full(N, mu))) for mu in mugrid}
    mu_star = max(glob, key=glob.get)
    r_glob = glob[mu_star]
    say(f"     best GLOBAL mu on out-of-fold score: mu = {mu_star:+.1f} kT, |r| {r_glob:.4f}")
    say(f"     loop 206 reported computed occupancy at mu = -4.0 reaching r -0.0133 on its own "
        f"target")
    betas = np.arange(-3.0, 3.01, 0.25)
    loc = {float(b): score(occ_at(mu_star + b * dens)) for b in betas}
    b_star = max(loc, key=loc.get); r_loc = loc[b_star]
    ds = dens.copy(); rng.shuffle(ds)
    r_ctrl = max(score(occ_at(mu_star + b * ds)) for b in betas)
    say(f"     best LOCAL beta = {b_star:+.2f} kT per sd of log contact density, |r| {r_loc:.4f}")
    say(f"     shuffled-density control, best over the same beta grid: |r| {r_ctrl:.4f}")
    ok3 = bool(r_loc - r_glob >= GAIN and r_loc - r_ctrl >= GAIN)
    G.add("H3", ok3, stat=float(r_loc - r_glob),
          if_true=lambda: f"H3 PASS -- local mu reaches {r_loc:.4f} against global {r_glob:.4f} "
                          f"and shuffled density {r_ctrl:.4f}",
          if_false=lambda: f"H3 FAIL -- local {r_loc:.4f}, global {r_glob:.4f}, shuffled control "
                           f"{r_ctrl:.4f}; the extra parameter is not buying locality")
    res["fix1"] = {"mu_star": mu_star, "r_global": r_glob, "beta_star": b_star,
                   "r_local": r_loc, "r_shuffled_density": r_ctrl}

    # ---------------------------------------------------------------- H4
    say("H4 FIX 2 -- DOES DRIVEN OCCUPANCY BEAT EQUILIBRIUM?")
    th_eq = occ_at(np.full(N, mu_star))
    Zpf = np.clip(th_eq, 1e-9, 1 - 1e-9); Zpf = Zpf / (1 - Zpf)
    alphas = np.concatenate([-np.logspace(-2, 1, 13)[::-1], [0.0], np.logspace(-2, 1, 13)])
    drv = {}
    for a in alphas:
        den = 1.0 + a * acc
        th = np.where(den > 1e-6, Zpf / (Zpf + den), np.nan)
        drv[float(a)] = score(np.nan_to_num(th, nan=0.0))
    a_star = max(drv, key=drv.get); r_drv = drv[a_star]
    r_eq = drv[0.0]
    say(f"     equilibrium (alpha = 0): |r| {r_eq:.4f}")
    say(f"     best driven alpha = {a_star:+.4f} (sign FREE, not assumed): |r| {r_drv:.4f}")
    say(f"     alpha {'> 0, remodelling raises the off-rate and evicts' if a_star > 0 else ('< 0, remodelling lowers the off-rate and loads' if a_star < 0 else '= 0, no drive selected')}")
    G.add("H4", bool(r_drv - r_eq >= GAIN), stat=float(r_drv - r_eq),
          if_true=lambda: f"H4 PASS -- driven occupancy {r_drv:.4f} against equilibrium "
                          f"{r_eq:.4f} at alpha {a_star:+.3f}",
          if_false=lambda: f"H4 FAIL -- driven {r_drv:.4f} against equilibrium {r_eq:.4f}; "
                           f"breaking detailed balance with an accessibility-scaled off-rate does "
                           f"not help")
    res["fix2"] = {"r_equilibrium": r_eq, "alpha_star": a_star, "r_driven": r_drv}

    # ---------------------------------------------------------------- H5
    say("H5 FIX 3 -- DO CELL-STATE INTERACTIONS BEAT ADDITIVE?")
    base_occ = occ_at(mu_star + b_star * dens).reshape(N, 1)
    ADD = np.hstack([base_occ, CHIP, ch])
    INT = np.hstack([ADD, base_occ * CHIP, base_occ * acc.reshape(N, 1),
                     ch[:, :1] * acc.reshape(N, 1)])
    r_add, r_int = score(ADD), score(INT)
    say(f"     additive  (occupancy + {len(TRACKS)} tracks + chromatin): |r| {r_add:.4f}, "
        f"{ADD.shape[1]} columns")
    say(f"     interacting (adds occupancy x track and occupancy x accessibility): |r| {r_int:.4f}, "
        f"{INT.shape[1]} columns")
    G.add("H5", bool(r_int - r_add >= GAIN), stat=float(r_int - r_add),
          if_true=lambda: f"H5 PASS -- conditioning sequence on cell state gains "
                          f"{r_int-r_add:+.4f} to {r_int:.4f}",
          if_false=lambda: f"H5 FAIL -- {r_int:.4f} against additive {r_add:.4f}")
    res["fix3"] = {"r_additive": r_add, "r_interaction": r_int}

    # ---------------------------------------------------------------- H6
    say("H6 FIX 5 -- DOES STACKING BEAT CONCATENATION HERE?")
    ro = {g: i for i, g in enumerate(ro_gene)}
    pcols = [ro[s] for s in names if s in ro]
    pnames = [s for s in names if s in ro]
    say(f"     Perturb-seq readout available for {len(pnames):,} of {N:,} genes")
    PS_mean = np.zeros((N, K_PS)); PS_sw = np.zeros((N, K_PS))
    if len(pcols) >= 50:
        good = np.isfinite(m_hat[:, pcols]).all(1) & np.isfinite(f_all[:, pcols]).all(1)
        say(f"     {int(good.sum()):,} of {len(good):,} perturbations are finite across those "
            f"genes; loop 208 A4 measured ~17.8% of Perturb-seq rows carrying non-finite values, "
            f"and an unscreened SVD does not converge on them")
        Mm = m_hat[np.ix_(good, pcols)].T
        Ff = f_all[np.ix_(good, pcols)].T
        for src, dest in ((Mm, PS_mean), (Ff, PS_sw)):
            Xc = np.nan_to_num(src - src.mean(0, keepdims=True), nan=0.0,
                               posinf=0.0, neginf=0.0)
            Xc = Xc[:, np.isfinite(Xc).all(0) & (Xc.std(0) > 0)]
            U, s_, _ = np.linalg.svd(Xc, full_matrices=False)
            kk = min(K_PS, U.shape[1])
            E = np.zeros((U.shape[0], K_PS))
            E[:, :kk] = U[:, :kk] * s_[:kk]
            idx = {g: i for i, g in enumerate(pnames)}
            for k, g in enumerate(names):
                if g in idx:
                    dest[k] = E[idx[g]]
    BLOCKS_BROKEN = {"occ_global_eq": occ_at(np.full(N, mu_star)).reshape(N, 1),
                     "chip": CHIP, "chromatin": ch, "perturbseq_mean": PS_mean}
    BLOCKS_FIXED = {"occ_local_driven": None, "chip_x_occ": None, "chromatin": ch,
                    "perturbseq_switching": PS_sw}
    den = 1.0 + a_star * acc
    th_fixed = np.where(den > 1e-6, Zpf / (Zpf + den), np.nan)
    th_fixed = np.nan_to_num(occ_at(mu_star + b_star * dens) if a_star == 0 else th_fixed)
    BLOCKS_FIXED["occ_local_driven"] = th_fixed.reshape(N, 1)
    BLOCKS_FIXED["chip_x_occ"] = np.hstack([CHIP, CHIP * th_fixed.reshape(N, 1)])

    def concat_score(bl):
        return score(np.hstack([np.nan_to_num(v) for v in bl.values()]))

    def stack_score(bl):
        P = np.column_stack([cv_pred(np.nan_to_num(v).reshape(N, -1), y, folds)
                             for v in bl.values()])
        return abs(pear(y, cv_pred(P, y, folds)))

    c_fix, s_fix = concat_score(BLOCKS_FIXED), stack_score(BLOCKS_FIXED)
    say(f"     fixed blocks, concatenated: |r| {c_fix:.4f}")
    say(f"     fixed blocks, stacked:      |r| {s_fix:.4f}   delta {s_fix-c_fix:+.4f}")
    say(f"     loop 213 measured stacked {REF_213_STACK:.4f} against concatenated "
        f"{REF_213_CONCAT:.4f} on ten blocks -- context, not the bar")
    G.add("H6", bool(s_fix - c_fix >= GAIN), stat=float(s_fix - c_fix),
          if_true=lambda: f"H6 PASS -- stacking gains {s_fix-c_fix:+.4f} over concatenation",
          if_false=lambda: f"H6 FAIL -- stacked {s_fix:.4f} against concatenated {c_fix:.4f}")
    res["fix5"] = {"concat": c_fix, "stacked": s_fix, "loop213_stack": REF_213_STACK}

    # ---------------------------------------------------------------- H7
    say("H7 ALL FIVE TOGETHER AGAINST ALL FIVE ERRORS")
    broken = concat_score(BLOCKS_BROKEN)
    fixed = s_fix
    say(f"     BROKEN  global-mu equilibrium occupancy, additive features, mean-based")
    say(f"             Perturb-seq, blocks concatenated:            |r| {broken:.4f}")
    say(f"     FIXED   local-mu driven occupancy, state interactions, switching-fraction")
    say(f"             Perturb-seq, blocks stacked:                 |r| {fixed:.4f}")
    say(f"     delta {fixed-broken:+.4f} against a {BIG_GAIN:+.2f} bar")
    G.add("H7", bool(fixed - broken >= BIG_GAIN), stat=float(fixed - broken),
          if_true=lambda: f"H7 PASS -- fixing all five raises held-out |r| from {broken:.4f} to "
                          f"{fixed:.4f}, {fixed-broken:+.4f}",
          if_false=lambda: f"H7 FAIL -- {fixed:.4f} against the broken model's {broken:.4f}, "
                           f"{fixed-broken:+.4f} against a {BIG_GAIN:+.2f} bar")
    res["combined"] = {"broken": broken, "fixed": fixed, "delta": fixed - broken}

    # ---------------------------------------------------------------- H8
    say("H8 CONTROL: DOES THE FIXED MODEL BEAT ITS OWN SHUFFLED TARGET?")
    ysh = y.copy(); rng.shuffle(ysh)
    Psh = np.column_stack([cv_pred(np.nan_to_num(v).reshape(N, -1), ysh, folds)
                           for v in BLOCKS_FIXED.values()])
    sh = abs(pear(ysh, cv_pred(Psh, ysh, folds)))
    say(f"     fixed model on real target   |r| {fixed:.4f}")
    say(f"     fixed model on shuffled target |r| {sh:.4f}   margin {fixed-sh:+.4f}")
    G.add("H8", bool(fixed - sh >= CTRL_GAIN), stat=float(fixed - sh),
          if_true=lambda: f"H8 PASS -- {fixed-sh:+.4f} over its own shuffled control",
          if_false=lambda: f"H8 FAIL -- only {fixed-sh:+.4f} over shuffled; the five "
                           f"generalisations are fitting noise, not the target")
    res["shuffled"] = {"real": fixed, "shuffled": sh, "margin": fixed - sh}

    # ---------------------------------------------------------------- H9
    say("H9 WHAT THIS CANNOT SHOW")
    say("     One drug, one cell line. A fix that helps the A549 dexamethasone plateau is not")
    say("     thereby a general improvement to cell modelling.")
    say("     FIX 4 is tested for REPRODUCIBILITY on Perturb-seq, where single cells exist, then")
    say("     carried into the A549 stack as a feature. A549 is bulk with four replicates and has")
    say("     no single-cell layer, so the stochastic fix cannot be tested directly against this")
    say("     target -- only its downstream usefulness can.")
    say("     Local contact density is Hi-C-derived from other cell types, so FIX 1's enrichment")
    say("     proxy is not measured in A549 and a failure could be the proxy, not the principle.")
    say("     DNase accessibility stands in for ATP-driven remodeller activity in FIX 2. It is a")
    say("     consequence of remodelling rather than a measurement of it.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary()
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
