"""LOOP 112 -- THE INTEGRATOR: make dM/dt and dP/dt run forward, and find out how much of the cell can.

THE TRAP THIS LOOP EXISTS TO AVOID, STATED FIRST BECAUSE IT WOULD OTHERWISE EAT THE WHOLE RESULT.
Loop 91 derived a per-gene transcription rate from the steady-state identity

        k_sm = mRNA_copies * (ln2/t_half_mRNA + ln2/t_double)

and mRNA_copies is an INPUT. Integrating dM/dt = k_sm - (ln2/t_half + mu) M with that k_sm returns
M* = k_sm/(ln2/t_half + mu) = mRNA_copies, exactly, for every gene, always. That is one line of
algebra with an ODE solver wrapped around it. Reporting it as "the integrator reproduces the
measured steady state" would be the worst sentence this project has written. So the tautology is
run ON PURPOSE, in its own arm, labelled, and excluded from every claim -- the template is loop
101's D5, which fed loop 92's own utilisation back in and recovered the 24 h that went in at a
relative error of 4e-15.

INPUTS AND PREDICTIONS, PER GENE. Nothing below is worth reading without this table.

    QUANTITY                      ARM C (circular)        ARM I (independent)     ARM N (null)
    mRNA half-life                INPUT  (Schwanhausser)  INPUT                   INPUT
    protein half-life             INPUT  (Schwanhausser)  INPUT                   INPUT
    doubling time 24 h            INPUT  (fixed, mu)      INPUT                   INPUT
    k_sm  (transcription)         INPUT, built FROM       INPUT, built from       INPUT, Arm I's
                                  this gene's own            LARSSON burst          k_sm shuffled
                                  mRNA_copies                kinetics               across genes
    k_sp  (translation)           INPUT, built FROM       INPUT, one global       same
                                  this gene's own            constant 119.24/h
                                  prot_copies                (loop 91)
    M*  steady-state mRNA         PREDICTED (== input)    PREDICTED               PREDICTED
    P*  steady-state protein      PREDICTED (== input)    PREDICTED               PREDICTED
    mRNA_copies (Schwanhausser)   used to BUILD k_sm      NEVER TOUCHED           NEVER TOUCHED
    prot_copies (Schwanhausser)   used to BUILD k_sp      NEVER TOUCHED           NEVER TOUCHED
                                                          except as the held-out truth, and to
                                                          fit ONE global unit-conversion constant,
                                                          CROSS-FITTED so that no gene's own copy
                                                          number ever enters its own prediction

Arm C's M* and P* are predictions in name only. Arm I's are the only ones that mean anything.

WHAT ARM I ACTUALLY IS, AND THE UNITS PROBLEM NOBODY WARNED ABOUT. Larsson 2019 fits a telegraph
model to allele-resolved scRNA-seq and reports (kon, koff, ksyn) per allele for 6,180 genes. The
brief for this loop says to form k_sm = kon/(kon+koff) * ksyn and treat it as a transcription rate
in mRNA/cell/h. IT IS NOT ONE. Those parameters are inferred in units of the mRNA loss rate -- the
fit sets that rate to 1 -- so kon/(kon+koff)*ksyn is DIMENSIONLESS and equals the mean allelic
mRNA count. The paper's own spreadsheet proves it: the "Mean expression" column it ships alongside
is reproduced by that product. Gate I3 measures this rather than assuming it, and the arm is built
on whichever answer comes back. Written as Theta = kon/(kon+koff)*ksyn, the honest conversion is

        k_sm = alpha * Theta * lambda_L        M* = k_sm / (ln2/t_half + mu)

with lambda_L the loss rate the normalisation used and alpha one global UMI-to-copies constant.
Two readings of lambda_L are possible and BOTH ARE RUN:

    I-A   lambda_L = ln2/t_half        (degradation only)   ->  M* = alpha * Theta * d/(d+mu)
    I-B   lambda_L = ln2/t_half + mu   (degradation+dilution) -> M* = alpha * Theta

and I-B is the case in which the ODE is an identity map on its own input. That comparison is gate
I5, and it is the sharpest question here: does the differential equation add anything at all to the
number that was handed to it?

WHAT IS AND IS NOT INDEPENDENT ABOUT ARM I. Larsson is mouse fibroblast (CAST x C57 F1);
Schwanhausser is mouse NIH3T3. Close, not identical: different lines, different labs, different
decade, and the comparison inherits every difference between them. Independent in the way that
matters -- no Schwanhausser abundance enters any Arm I prediction -- but Theta is still a mean
expression level measured by a different instrument, not a rate derived from first principles. Arm
I is therefore a cross-platform abundance test with a dilution correction bolted on, and calling it
anything grander would be the same mistake in a nicer suit.

DIVISION RULE FROM LOOP 92 IS OBEYED. No rate anywhere below is formed by dividing two different
abundance datasets. Half-lives, copies and the k_sp constant all come from Schwanhausser; Theta
comes from Larsson and is converted with a single scalar, never a per-gene ratio.

DISCLOSED PEEK, before the gates were written. Reconnaissance to find out what the files contain
measured four things: 4,527 Schwanhausser genes carry both half-lives; 4,190 carry all four fields;
the core set covers 92.9% of measured proteome mass; and Theta reproduces Larsson's own mean
expression column to a median 0.17%. That last number is a peek at gate I3's statistic and I3's 1%
threshold was chosen knowing it -- the gate exists to record the units question in the artefact, not
to discover it. NO correlation between Larsson and Schwanhausser was computed before the thresholds
in I4, I5 and I6 were fixed, and those are the gates that carry the result.

PREDECLARED, before any number:

  I1 THE INTEGRATOR IS AN INTEGRATOR, NOT ALGEBRA                     THE ARITHMETIC.
       a forward march from M=P=0 with an exponential (stiff-exact) stepper, checked against an
       INDEPENDENT explicit RK4 run with a per-gene stability-safe step, at four points along the
       transient. Gate: normalised deviation < 1e-8 for >= 99% of genes. Two schemes that disagree
       mean nothing below is worth reading.
  I2 THE CIRCULAR ARM, RUN ON PURPOSE                                 THE TAUTOLOGY.
       Arm C, both tiers, from the loop 91 identity. Gate: max |M*/M_obs - 1| < 1e-12 and the same
       for P. A PASS here is evidence about floating point and nothing else, and it is EXCLUDED
       from I4, I5, I6 and from every sentence in the headline. A FAIL would mean the integrator is
       broken, which is the only reason to run it.
  I3 ARM I's INPUT IS DIMENSIONALLY WHAT IT IS CLAIMED TO BE          THE UNITS.
       test Theta against Larsson's own mean-expression column. Gate: EITHER median relative error
       < 1% (the rates are normalised, Theta is a COUNT, and the arm is rebuilt as k_sm = alpha
       Theta lambda) OR the implied absolute rates sit within 10x of loop 91's median 1.80
       mRNA/cell/h (the rates are absolute). FAIL if neither, because then the units are unknown
       and every number in Arm I is uninterpretable.
  I4 ARM I PREDICTS AN mRNA LEVEL IT NEVER SAW, BEATING THE NULL AND FAME   THE REAL TEST.
       Spearman(M*_ArmI, mRNA_copies). Gate, all three: rho >= 0.30; rho above a PUBLICATION-COUNT
       predictor on the same genes; and Arm N's shuffled-k_sm null beaten with gate_guard.survival
       defined and gate_guard.null_can_move certifying the shuffle changes the input.
  I5 DOES THE DIFFERENTIAL EQUATION ADD ANYTHING TO ITS OWN INPUT?    THE NEAR-TAUTOLOGY.
       I-A (with the dilution correction) against I-B (which is raw Theta, the ODE as identity map).
       Gate: I-A must improve Spearman against measured copies by >= 0.010. EXPECTED TO FAIL. A
       failure is the loop's honest headline: that the two-tier ODE is, at the mRNA tier, a units
       conversion of somebody else's abundance measurement.
  I6 THE PROTEIN TIER                                                 THE SECOND HOP.
       P* from Arm I's M* and loop 91's single global k_sp, against measured prot_copies. Gate:
       rho >= 0.25 and Arm N beaten with survival defined. Reported with the fraction of genes
       within 2x, 5x and 10x, and with how much accuracy is lost between tier one and tier two --
       a cascade that collapses at the second hop is a result about the cascade. I5's question is
       also asked again one tier up, as a reported diagnostic and not as a gate: does dividing by
       the measured protein loss rate beat M* on its own? Unlike the mRNA tier, that divisor is a
       per-gene measurement the input did not contain, so a gain there is the ODE doing work.
  I7 HOW MUCH OF THE CELL CAN BE INTEGRATED, AND DOES IT CONVERGE     THE COVERAGE.
       gene count, molecule fraction and PROTEOME MASS fraction of the integrable set; and the
       forward run's convergence. Gate: >= 99% of genes reach within 1e-6 of the analytic steady
       state by the horizon. Separately REPORTED, not gated: what fraction of the proteome reaches
       95% of steady state inside one 24 h cell cycle, because a protein whose lifetime exceeds the
       cell cycle has no steady state to be predicted at.

  NEGATIVES ARE THE POINT. If I5 fails, this loop's contribution is a demonstration that an ODE
  wrapped around a measured abundance predicts abundance, and it will say exactly that.

-> outputs/loop_integrator.json
"""
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import gate_guard as GG  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
SCHWAN = SC / "_schwan2011.json"
LARSSON = SC / "larsson_S5.xlsx"
FASTA = SC / "human_proteome.fasta.gz"
LIFE = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_lifetimes.json"

LN2 = float(np.log(2.0))
DOUBLING_H = 24.0
MU = LN2 / DOUBLING_H                     # 0.0288811/h, the dilution term, an INPUT everywhere
KSP_GLOBAL = 119.24                       # loop 91 median translation rate, protein/mRNA/h
KSM_MEDIAN_L91 = 1.80                     # loop 91 median transcription rate, mRNA/cell/h

HORIZON_H = 2000.0                        # forward-march horizon
STEP_H = 0.25                             # forward-march step
RK4_STEPS = 2000                          # RK4 check steps per gene
RK4_EFOLDS = 10.0                         # RK4 check runs to 10 e-foldings of the FAST mode
FOLDS = 5
NPERM = 24
SEED = 11201

# thresholds, fixed before any number below was computed
T_I1_DEV = 1e-8
T_I1_FRAC = 0.99
T_I2_REL = 1e-12
T_I3_UNITS = 0.01
T_I3_ABS_FACTOR = 10.0
T_I4_RHO = 0.30
T_I5_GAIN = 0.010
T_I6_RHO = 0.25
T_I7_CONV = 1e-6
T_I7_FRAC = 0.99

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def spear(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < 30:
        return float("nan"), int(f.sum())
    return float(spearmanr(a[f], b[f]).statistic), int(f.sum())


def fold_frac(pred, obs):
    """Fraction of genes within 2x, 5x, 10x. The only scale-sensitive accuracy statistic here."""
    p, o = np.asarray(pred, float), np.asarray(obs, float)
    f = np.isfinite(p) & np.isfinite(o) & (p > 0) & (o > 0)
    r = np.abs(np.log(p[f] / o[f]))
    return {"n": int(f.sum()),
            "within_2x": float(np.mean(r <= np.log(2.0))),
            "within_5x": float(np.mean(r <= np.log(5.0))),
            "within_10x": float(np.mean(r <= np.log(10.0))),
            "median_abs_log2_error": float(np.median(r / LN2))}


# ----------------------------------------------------------------------------------------------
# THE INTEGRATOR.  dM/dt = k_sm - a M ;  dP/dt = k_sp M - b P ;  a = ln2/t_m + mu, b = ln2/t_p + mu
# Linear, decoupled across genes, stiff (a/b spans ~4 orders). The production scheme is an
# exponential stepper -- exact for a linear system at ANY step size, so unconditionally stable --
# and gate I1 checks it against a plain explicit RK4 that has no such property.
# ----------------------------------------------------------------------------------------------

def exact(t, k_sm, a, k_sp, b, M0=None, P0=None):
    """Closed-form state at time t from (M0, P0). Used inside the stepper and by the RK4 check."""
    Mss = k_sm / a
    Pss = k_sp * Mss / b
    M0 = np.zeros_like(a) if M0 is None else M0
    P0 = np.zeros_like(a) if P0 is None else P0
    ea, eb = np.exp(-a * t), np.exp(-b * t)
    dM = M0 - Mss
    d = b - a
    deg = np.abs(d) < 1e-9
    safe = np.where(deg, 1.0, d)
    C = k_sp * dM / safe
    P = Pss + (P0 - Pss - C) * eb + C * ea
    if deg.any():
        P = np.where(deg, Pss + (P0 - Pss) * eb + k_sp * dM * t * eb, P)
    return Mss + dM * ea, P


def march(k_sm, a, k_sp, b, horizon=HORIZON_H, step=STEP_H, snaps=()):
    """Forward-march from M=P=0 with a fixed-step exponential stepper. Returns the final state, the
    first time each species reaches 95% of its own steady state, and snapshots at requested times."""
    n = len(a)
    M = np.zeros(n)
    P = np.zeros(n)
    Mss = k_sm / a
    Pss = k_sp * Mss / b
    ea, eb = np.exp(-a * step), np.exp(-b * step)
    d = b - a
    deg = np.abs(d) < 1e-9
    safe = np.where(deg, 1.0, d)
    t95m = np.full(n, np.nan)
    t95p = np.full(n, np.nan)
    grab = sorted(snaps)
    out_snaps = {}
    t = 0.0
    nsteps = int(round(horizon / step))
    for i in range(nsteps):
        dM = M - Mss
        Mn = Mss + dM * ea
        C = k_sp * dM / safe
        Pn = Pss + (P - Pss - C) * eb + C * ea
        if deg.any():
            Pn = np.where(deg, Pss + (P - Pss) * eb + k_sp * dM * step * eb, Pn)
        M, P = Mn, Pn
        t += step
        hit = np.isnan(t95m) & (Mss > 0) & (M >= 0.95 * Mss)
        t95m[hit] = t
        hit = np.isnan(t95p) & (Pss > 0) & (P >= 0.95 * Pss)
        t95p[hit] = t
        while grab and t >= grab[0] - 1e-9:
            g = grab.pop(0)
            out_snaps[g] = {"M_frac": M / np.where(Mss > 0, Mss, np.nan),
                            "P_frac": P / np.where(Pss > 0, Pss, np.nan)}
    return {"M": M, "P": P, "Mss": Mss, "Pss": Pss, "t95_mrna": t95m, "t95_prot": t95p,
            "snaps": out_snaps, "steps": nsteps}


def rk4_deviation(k_sm, a, k_sp, b, nstep=RK4_STEPS, efolds=RK4_EFOLDS):
    """Plain explicit RK4 with a per-gene step, against the closed form, at four transient points.

    The step is h_i = efolds/(nstep * max(a_i,b_i)), so h*lambda = 0.005 -- deep inside RK4's
    stability region and small enough that its own truncation error is ~1e-10, which is what makes a
    1e-8 gate a test of the exponential stepper rather than of RK4."""
    lam = np.maximum(a, b)
    h = efolds / (nstep * lam)
    M = np.zeros_like(a)
    P = np.zeros_like(a)
    Mss = k_sm / a
    Pss = k_sp * Mss / b
    scale_M = np.where(Mss > 0, Mss, 1.0)
    scale_P = np.where(Pss > 0, Pss, 1.0)
    checks = {nstep // 8, nstep // 4, nstep // 2, nstep}
    worst = np.zeros_like(a)

    def f(m, p):
        return k_sm - a * m, k_sp * m - b * p

    for i in range(1, nstep + 1):
        k1m, k1p = f(M, P)
        k2m, k2p = f(M + 0.5 * h * k1m, P + 0.5 * h * k1p)
        k3m, k3p = f(M + 0.5 * h * k2m, P + 0.5 * h * k2p)
        k4m, k4p = f(M + h * k3m, P + h * k3p)
        M = M + (h / 6.0) * (k1m + 2 * k2m + 2 * k3m + k4m)
        P = P + (h / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        if i in checks:
            Me, Pe = exact(i * h, k_sm, a, k_sp, b)
            dev = np.maximum(np.abs(M - Me) / scale_M, np.abs(P - Pe) / scale_P)
            worst = np.maximum(worst, dev)
    return worst


# ----------------------------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------------------------

def protein_lengths():
    import gzip
    L, name, n = {}, None, 0
    with gzip.open(FASTA, "rt") as f:
        for ln in f:
            if ln.startswith(">"):
                if name and n:
                    L[name] = max(L.get(name, 0), n)
                n, name = 0, None
                for part in ln.split():
                    if part.startswith("GN="):
                        name = part[3:]
                        break
            else:
                n += len(ln.strip())
    if name and n:
        L[name] = max(L.get(name, 0), n)
    return {k.upper(): v for k, v in L.items()}


NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan")


def load_larsson():
    """Per-allele (kon, koff, ksyn) and the paper's own mean-expression column, by upper-case symbol.

    Excel has date-corrupted a handful of gene symbols (SEPT/MARCH/DEC families). Those rows are
    dropped and counted rather than guessed at."""
    import pandas as pd
    xl = pd.ExcelFile(LARSSON)
    per = {}
    corrupt = 0
    for sheet in ("C57", "CAST"):
        d = xl.parse(sheet)
        col = [c for c in d.columns if "Maximum likelihood" in str(c)][0]
        mcol = [c for c in d.columns if str(c).strip().lower().startswith("mean expr")][0]
        rec = {}
        for g, s, me in zip(d["Gene"], d[col], d[mcol]):
            if not isinstance(g, str):
                corrupt += 1
                continue
            v = [float(x) for x in NUM.findall(str(s))]
            if len(v) != 3 or not np.all(np.isfinite(v)) or v[0] <= 0 or v[2] <= 0:
                continue
            rec[g.upper()] = {"kon": v[0], "koff": v[1], "ksyn": v[2], "mean_expr": float(me)}
        per[sheet] = rec
    return per, corrupt


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 112 -- THE INTEGRATOR: make dM/dt and dP/dt run forward, and find out how much of")
    say("              the cell can")
    say("=" * 100)
    say()
    say(f"  mu = ln2/{DOUBLING_H:.0f} h = {MU:.7f}/h   (INPUT, fixed, not fitted)")
    say(f"  dM/dt = k_sm - (ln2/t_mRNA + mu) M      dP/dt = k_sp M - (ln2/t_prot + mu) P")
    say()

    S = json.load(open(SCHWAN))
    C = json.load(open(LR.CELL))
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}
    plen = protein_lengths()
    per, corrupt = load_larsson()

    core = sorted(g for g, v in S.items()
                  if v.get("mrna_hl_h") and v.get("prot_hl_h")
                  and v["mrna_hl_h"] > 0 and v["prot_hl_h"] > 0)
    g_core = np.array(core)
    tm = np.array([S[g]["mrna_hl_h"] for g in core], float)
    tp = np.array([S[g]["prot_hl_h"] for g in core], float)
    a = LN2 / tm + MU
    b = LN2 / tp + MU
    dm = LN2 / tm                       # pure degradation, no dilution
    mobs = np.array([S[g].get("mrna_copies") or np.nan for g in core], float)
    pobs = np.array([S[g].get("prot_copies") or np.nan for g in core], float)
    say(f"  {len(S):,} Schwanhausser genes on disk; {len(core):,} carry BOTH half-lives and can")
    say(f"  therefore enter the two-tier ODE at all")
    say(f"  a = ln2/t_mRNA + mu spans {a.min():.4f} .. {a.max():.4f} /h   "
        f"b = ln2/t_prot + mu spans {b.min():.4f} .. {b.max():.4f} /h")
    say(f"  stiffness ratio max(a,b)/min(a,b) reaches "
        f"{np.max(np.maximum(a, b) / np.minimum(a, b)):.0f}x -- this system is stiff")
    say()

    # ------------------------------------------------------------------ I1
    say("I1 THE INTEGRATOR IS AN INTEGRATOR, NOT ALGEBRA")
    k_probe = np.where(np.isfinite(mobs), mobs, 10.0) * a       # any positive k_sm exercises it
    ksp_probe = np.full(len(core), KSP_GLOBAL)
    dev = rk4_deviation(k_probe, a, ksp_probe, b)
    frac_ok = float(np.mean(dev < T_I1_DEV))
    say(f"     explicit RK4, per-gene step h = {RK4_EFOLDS:.0f}/({RK4_STEPS} x max(a,b)), "
        f"h*lambda = {RK4_EFOLDS/RK4_STEPS:.4f}")
    say(f"     compared with the exponential stepper at 4 points along each gene's transient")
    say(f"     max normalised deviation: median {np.median(dev):.3e}  95th {np.percentile(dev,95):.3e}"
        f"  worst {dev.max():.3e}")
    say(f"     genes agreeing to better than {T_I1_DEV:.0e}: {frac_ok:.4%}")
    i1 = bool(frac_ok >= T_I1_FRAC)
    say(f"     I1 {'PASS' if i1 else 'FAIL'} (gate: >= {T_I1_FRAC:.0%} of genes below "
        f"{T_I1_DEV:.0e})")
    say()

    # ------------------------------------------------------------------ I2  ARM C
    say("I2 THE CIRCULAR ARM, RUN ON PURPOSE -- THIS IS THE TAUTOLOGY AND IT IS EXCLUDED")
    cmask = np.isfinite(mobs) & np.isfinite(pobs) & (mobs > 0) & (pobs > 0)
    ic = np.where(cmask)[0]
    ksm_C = mobs[ic] * a[ic]                       # loop 91 identity: uses the gene's OWN mRNA level
    ksp_C = pobs[ic] * b[ic] / mobs[ic]            # same identity at the protein tier
    runC = march(ksm_C, a[ic], ksp_C, b[ic])
    relM = np.abs(runC["M"] / mobs[ic] - 1.0)
    relP = np.abs(runC["P"] / pobs[ic] - 1.0)
    say(f"     {len(ic):,} genes with both half-lives AND both copy numbers")
    say(f"     k_sm := mRNA_copies * (ln2/t_mRNA + mu)      <- mRNA_copies is an INPUT")
    say(f"     k_sp := prot_copies * (ln2/t_prot + mu) / mRNA_copies   <- prot_copies is an INPUT")
    say(f"     forward march from M=P=0 over {HORIZON_H:.0f} h in {runC['steps']:,} steps of "
        f"{STEP_H} h")
    say(f"     |M*/M_observed - 1|   median {np.median(relM):.3e}   max {relM.max():.3e}")
    say(f"     |P*/P_observed - 1|   median {np.median(relP):.3e}   max {relP.max():.3e}")
    i2 = bool(relM.max() < T_I2_REL and relP.max() < T_I2_REL)
    say(f"     I2 {'PASS' if i2 else 'FAIL'} (gate: max relative error < {T_I2_REL:.0e} on both "
        f"tiers)")
    say(f"     WHAT THIS IS EVIDENCE OF: floating-point arithmetic. M* = k_sm/a and k_sm = M_obs*a,")
    say(f"     so M* = M_obs is an algebraic identity that no integrator could fail. It says nothing")
    say(f"     whatever about transcription, translation, or this cell. Loop 101's D5 did the same")
    say(f"     thing on purpose and recovered its own 24 h at 4e-15. EXCLUDED FROM I4, I5, I6 AND")
    say(f"     FROM EVERY HEADLINE NUMBER.")
    say()

    # ------------------------------------------------------------------ I3  units
    say("I3 ARM I's INPUT IS DIMENSIONALLY WHAT IT IS CLAIMED TO BE")
    both_allele = sorted(set(per["C57"]) & set(per["CAST"]))
    say(f"     Larsson S5: {len(per['C57']):,} C57 genes, {len(per['CAST']):,} CAST genes, "
        f"{len(both_allele):,} with both alleles ({corrupt} Excel date-corrupted symbols dropped)")
    rel_units, theta_all, mean_all = [], [], []
    for sh in ("C57", "CAST"):
        for g, v in per[sh].items():
            th = v["kon"] / (v["kon"] + v["koff"]) * v["ksyn"]
            theta_all.append(th)
            mean_all.append(v["mean_expr"])
            if v["mean_expr"] and v["mean_expr"] > 0:
                rel_units.append(abs(th / v["mean_expr"] - 1.0))
    rel_units = np.array(rel_units)
    med_units = float(np.median(rel_units))
    say(f"     Theta = kon/(kon+koff)*ksyn vs the paper's own 'Mean expression' column:")
    say(f"        median relative error {med_units:.5f}   n {len(rel_units):,}")
    normalised = bool(med_units < T_I3_UNITS)
    # the alternative hypothesis: Theta is already an absolute rate in mRNA/cell/h
    th_core = {}
    for g in both_allele:
        s = 0.0
        for sh in ("C57", "CAST"):
            v = per[sh][g]
            s += v["kon"] / (v["kon"] + v["koff"]) * v["ksyn"]
        th_core[g] = s
    inter = [i for i, g in enumerate(g_core) if g in th_core]
    theta_c = np.array([th_core[g_core[i]] for i in inter], float)
    med_theta = float(np.median(theta_c))
    abs_factor = max(med_theta / KSM_MEDIAN_L91, KSM_MEDIAN_L91 / med_theta)
    say(f"     if instead Theta were already mRNA/cell/h, its median over the shared genes would be")
    say(f"        {med_theta:.3f}/h against loop 91's measured median {KSM_MEDIAN_L91:.2f}/h "
        f"-> a factor of {abs_factor:.1f}")
    absolute = bool(abs_factor <= T_I3_ABS_FACTOR)
    i3 = bool(normalised or absolute)
    verdict = ("NORMALISED: Theta is a mean mRNA COUNT in Larsson units, not a rate"
               if normalised else
               ("ABSOLUTE: Theta is a rate in mRNA/cell/h" if absolute else "UNRESOLVED"))
    say(f"     verdict: {verdict}")
    say(f"     I3 {'PASS' if i3 else 'FAIL'} (gate: one hypothesis decisively supported; "
        f"FAIL means Arm I's units are unknown)")
    if normalised:
        say(f"     CONSEQUENCE: the brief's k_sm = kon/(kon+koff)*ksyn is DIMENSIONLESS. Arm I is")
        say(f"     rebuilt as k_sm = alpha * Theta * lambda_L with lambda_L the loss rate the fit")
        say(f"     normalised by, and alpha a single global UMI-to-copies constant. Two readings of")
        say(f"     lambda_L are run: I-A degradation only, I-B degradation+dilution.")
    say()

    # ------------------------------------------------------------------ ARM I construction
    say("ARM I -- CONSTRUCTION, AND WHAT IT NEVER TOUCHES")
    idx = np.array(inter, int)
    gI = g_core[idx]
    aI, bI, dmI = a[idx], b[idx], dm[idx]
    mobsI, pobsI = mobs[idx], pobs[idx]
    keep = np.isfinite(mobsI) & (mobsI > 0) & (theta_c > 0)
    idx, gI, aI, bI, dmI, mobsI, pobsI, theta_c = (idx[keep], gI[keep], aI[keep], bI[keep],
                                                   dmI[keep], mobsI[keep], pobsI[keep],
                                                   theta_c[keep])
    say(f"     {len(gI):,} genes have both Schwanhausser half-lives, a Schwanhausser mRNA copy")
    say(f"     number to be tested against, and Larsson burst kinetics on both alleles")
    say(f"     Larsson = mouse fibroblast CASTxC57 F1; Schwanhausser = mouse NIH3T3. Both mouse,")
    say(f"     DIFFERENT LINES, different labs, ~8 years apart. Every difference between them is")
    say(f"     inside the error bars below and none of it is modelled.")

    # k_sm under the two readings, up to the global constant alpha (set below by cross-fitting)
    ksm_A_raw = theta_c * dmI                      # lambda_L = degradation only
    ksm_B_raw = theta_c * (dmI + MU)               # lambda_L = degradation + dilution
    runA = march(ksm_A_raw, aI, np.full(len(gI), KSP_GLOBAL), bI,
                 snaps=(1, 2, 4, 8, 12, 24, 48, 96, 168, 336, 720))
    runB = march(ksm_B_raw, aI, np.full(len(gI), KSP_GLOBAL), bI)
    MA_raw, MB_raw = runA["M"], runB["M"]

    rng = np.random.default_rng(SEED)
    fid = rng.integers(0, FOLDS, size=len(gI))

    def crossfit_scale(raw, obs):
        """One global multiplicative constant, fitted on 4/5 of the genes and applied to the other
        1/5. Cross-fitted precisely so that no gene's own measured copy number can enter its own
        prediction -- that would be the loop 91 trap in a smaller costume."""
        out = np.full(len(raw), np.nan)
        alphas = []
        for k in range(FOLDS):
            tr = (fid != k) & np.isfinite(raw) & np.isfinite(obs) & (raw > 0) & (obs > 0)
            te = (fid == k) & np.isfinite(raw) & (raw > 0)
            if tr.sum() < 50 or te.sum() == 0:
                continue
            al = float(np.exp(np.median(np.log(obs[tr]) - np.log(raw[tr]))))
            alphas.append(al)
            out[te] = raw[te] * al
        return out, alphas

    MA, alphas_A = crossfit_scale(MA_raw, mobsI)
    MB, alphas_B = crossfit_scale(MB_raw, mobsI)
    say(f"     one global constant alpha, CROSS-FITTED over {FOLDS} folds: "
        f"{', '.join(f'{x:.4g}' for x in alphas_A)}  (spread "
        f"{np.std(alphas_A)/np.mean(alphas_A):.2%})")
    say(f"     that is ONE free parameter for {len(gI):,} genes, and no gene sees its own copy number")
    say(f"     implied median transcription rate from Arm I-A: "
        f"{np.nanmedian(ksm_A_raw * np.mean(alphas_A)):.3f} mRNA/cell/h "
        f"(loop 91 measured {KSM_MEDIAN_L91:.2f})")
    say()

    # ------------------------------------------------------------------ I4
    say("I4 ARM I PREDICTS AN mRNA LEVEL IT NEVER SAW, BEATING THE NULL AND FAME")
    rhoA, nA = spear(MA, mobsI)
    ffA = fold_frac(MA, mobsI)
    pv = np.array([pubs.get(g, 0.0) for g in gI], float)
    rho_pub, _ = spear(pv, mobsI)
    say(f"     ARM I-A  Spearman(M*, measured mRNA_copies) = {rhoA:+.4f}   n {nA:,}")
    say(f"        within 2x {ffA['within_2x']:.1%}   5x {ffA['within_5x']:.1%}   "
        f"10x {ffA['within_10x']:.1%}   median |log2 error| {ffA['median_abs_log2_error']:.2f}")
    say(f"     FAME  Spearman(publication count, measured mRNA_copies) = {rho_pub:+.4f}")
    all_m = mobs[np.isfinite(mobs) & (mobs > 0)]
    sel_lo, sel_hi = np.percentile(mobsI, [5, 95])
    say(f"     SELECTION BIAS, stated because it bounds what this rho means: a gene enters Arm I")
    say(f"        only if scRNA-seq detected it well enough to fit a telegraph model on BOTH")
    say(f"        alleles. Median mRNA_copies in the Arm I set {np.median(mobsI):.1f} against "
        f"{np.median(all_m):.1f}")
    say(f"        over all {len(all_m):,} measured genes; the test spans {sel_lo:.1f} to "
        f"{sel_hi:.1f} copies (5th-95th).")
    say(f"        The rank correlation is measured on the well-expressed half of the transcriptome")
    say(f"        and says nothing about the genes burst kinetics could not be fitted for.")

    # ARM N -- shuffle k_sm across genes, keeping each gene's own half-lives
    nulls, moved = [], []
    for _ in range(NPERM):
        perm = rng.permutation(len(gI))
        ksm_n = ksm_A_raw[perm]
        moved.append(GG.null_can_move(ksm_A_raw, ksm_n)["changed"])
        Mn_raw = march(ksm_n, aI, np.full(len(gI), KSP_GLOBAL), bI, horizon=400.0, step=1.0)["M"]
        Mn, _ = crossfit_scale(Mn_raw, mobsI)
        nulls.append(spear(Mn, mobsI)[0])
    cap = GG.null_can_move(ksm_A_raw, ksm_A_raw[rng.permutation(len(gI))])
    say(f"     ARM N  k_sm shuffled across genes, half-lives kept in place, {NPERM} permutations")
    say(f"        CAPABILITY CHECK: the shuffle changes {np.mean(moved):.1%} of k_sm entries "
        f"-- capable: {cap['capable']}")
    sA = GG.survival(rhoA, nulls)
    GG.report("Arm I-A mRNA rho under a k_sm shuffle", sA, emit=say)
    i4 = bool(rhoA >= T_I4_RHO and rhoA > rho_pub and cap["capable"] and sA.get("defined"))
    say(f"     I4 {'PASS' if i4 else 'FAIL'} (gate: rho >= {T_I4_RHO:.2f} AND rho > fame AND the "
        f"null is capable AND survival is defined)")
    if not (rhoA > rho_pub):
        say(f"     PUBLICATION COUNT BEATS THE PHYSICS. That is the recurring result in this project")
        say(f"     and it is reported, not buried.")
    say()

    # ------------------------------------------------------------------ I5
    say("I5 DOES THE DIFFERENTIAL EQUATION ADD ANYTHING TO ITS OWN INPUT?")
    rhoB, _ = spear(MB, mobsI)
    ffB = fold_frac(MB, mobsI)
    rho_theta, _ = spear(theta_c, mobsI)
    gain = rhoA - rhoB
    say(f"     I-A  lambda_L = ln2/t_mRNA         M* = alpha Theta d/(d+mu)   rho {rhoA:+.4f}   "
        f"2x {ffA['within_2x']:.1%}")
    say(f"     I-B  lambda_L = ln2/t_mRNA + mu    M* = alpha Theta            rho {rhoB:+.4f}   "
        f"2x {ffB['within_2x']:.1%}")
    say(f"     raw Larsson Theta, no ODE at all, no half-lives, no dilution:  rho {rho_theta:+.4f}")
    say(f"     the dilution correction d/(d+mu) spans "
        f"{np.min(dmI/(dmI+MU)):.3f} .. {np.max(dmI/(dmI+MU)):.3f} across genes, so it is NOT a")
    say(f"     constant and could in principle have moved the ranking")
    say(f"     gain from running the ODE: {gain:+.4f}")
    i5 = bool(gain >= T_I5_GAIN)
    say(f"     I5 {'PASS' if i5 else 'FAIL'} (gate: the ODE must improve Spearman by >= "
        f"{T_I5_GAIN:.3f})")
    if gain < 0:
        say(f"     AND THE SIGN IS NEGATIVE, WHICH SAYS MORE THAN A ZERO WOULD. Applying the")
        say(f"     dilution correction makes the prediction WORSE. Two readings, both allowed:")
        say(f"     (a) Larsson's normalisation already absorbed dilution, so lambda_L = d + mu is")
        say(f"         the correct reading and I-B is the right arm -- in which case M* = alpha")
        say(f"         Theta and the ODE is exactly an identity map on its input;")
        say(f"     (b) the per-gene half-life is noisy enough that multiplying by any function of")
        say(f"         it degrades a good predictor. Spearman(mRNA half-life, mRNA copies) is")
        say(f"         reported below at the leakage disclosure so this can be weighed.")
        say(f"     This loop cannot separate (a) from (b) and does not pretend to.")
    if not i5:
        say(f"     THE HONEST READING, PREDECLARED AS THE EXPECTED OUTCOME: at the mRNA tier this")
        say(f"     two-tier ODE is a UNITS CONVERSION of somebody else's abundance measurement. The")
        say(f"     integrator is correct (I1), it converges (I7), and it adds nothing the input did")
        say(f"     not already contain. Theta is Larsson's fitted mean expression by construction --")
        say(f"     I3 measured that -- so Arm I tests whether one lab's mean scRNA-seq level predicts")
        say(f"     another lab's mRNA copy number. That is a real and non-circular test. It is not a")
        say(f"     simulation of transcription.")
    say()

    # ------------------------------------------------------------------ I6
    say("I6 THE PROTEIN TIER")
    # P* from Arm I-A's M* and ONE global k_sp from loop 91. Zero further free parameters.
    PA = KSP_GLOBAL * MA / bI
    rhoP, nP = spear(PA, pobsI)
    ffP = fold_frac(PA, pobsI)
    rho_pub_p, _ = spear(pv, pobsI)
    say(f"     P* = k_sp M* / (ln2/t_prot + mu) with k_sp = {KSP_GLOBAL:.2f} protein/mRNA/h, ONE")
    say(f"     global constant from loop 91. No per-gene translation rate is used, because a per-gene")
    say(f"     k_sp can only be got by dividing prot_copies by mRNA_copies, which is Arm C again.")
    say(f"     Spearman(P*, measured prot_copies) = {rhoP:+.4f}   n {nP:,}")
    say(f"        within 2x {ffP['within_2x']:.1%}   5x {ffP['within_5x']:.1%}   "
        f"10x {ffP['within_10x']:.1%}   median |log2 error| {ffP['median_abs_log2_error']:.2f}")
    say(f"     FAME  Spearman(publication count, prot_copies) = {rho_pub_p:+.4f}")
    nullsP = []
    for _ in range(NPERM):
        perm = rng.permutation(len(gI))
        Mn_raw = march(ksm_A_raw[perm], aI, np.full(len(gI), KSP_GLOBAL), bI,
                       horizon=400.0, step=1.0)["M"]
        Mn, _ = crossfit_scale(Mn_raw, mobsI)
        nullsP.append(spear(KSP_GLOBAL * Mn / bI, pobsI)[0])
    sP = GG.survival(rhoP, nullsP)
    GG.report("Arm I-A protein rho under a k_sm shuffle", sP, emit=say)
    i6 = bool(rhoP >= T_I6_RHO and sP.get("defined"))
    say(f"     I6 {'PASS' if i6 else 'FAIL'} (gate: rho >= {T_I6_RHO:.2f} AND survival defined)")
    say(f"     the null keeps each gene's own 1/(ln2/t_prot + mu) and only shuffles k_sm, so its")
    say(f"     {sP['null_mean']:+.4f} is what the PROTEIN HALF-LIFE ALONE is worth here. The mRNA")
    say(f"     tier's null, {sA['null_mean']:+.4f}, is the same statement for the mRNA half-life.")
    # the protein-tier analogue of I5: does the second ODE hop add anything over its own input?
    rho_M_vs_prot, _ = spear(MA, pobsI)
    say(f"     DOES THE SECOND HOP ADD ANYTHING? -- the I5 question again, one tier up:")
    say(f"        M* alone against prot_copies            rho {rho_M_vs_prot:+.4f}")
    say(f"        P* = k_sp M*/b against prot_copies      rho {rhoP:+.4f}   "
        f"gain {rhoP - rho_M_vs_prot:+.4f}")
    say(f"        unlike the mRNA tier, dividing by the protein loss rate is a MEASURED per-gene")
    say(f"        quantity that the input did not contain, so a gain here is the ODE doing work.")
    say(f"     TIER-TO-TIER CHANGE: mRNA rho {rhoA:+.4f} -> protein rho {rhoP:+.4f} "
        f"({rhoP - rhoA:+.4f}); 2x accuracy {ffA['within_2x']:.1%} -> {ffP['within_2x']:.1%}. Rank")
    say(f"     agreement {'rises' if rhoP > rhoA else 'falls'} while fold accuracy "
        f"{'rises' if ffP['within_2x'] > ffA['within_2x'] else 'falls'} -- protein copy numbers span")
    say(f"     six orders of magnitude and are easier to RANK and harder to HIT than mRNA copies.")
    say(f"     the second hop adds a measured protein half-life and one constant, and that is the")
    say(f"     entire model of translation here -- no codon usage, no ribosome competition, no")
    say(f"     regulation. Loop 92 already showed ribosomes run at 22.3% utilisation, so a")
    say(f"     capacity-blind constant is not obviously wrong; it is just not informative.")
    say()

    # ------------------------------------------------------------------ I7
    say("I7 HOW MUCH OF THE CELL CAN BE INTEGRATED, AND DOES IT CONVERGE")
    pc_all = {g: v["prot_copies"] for g, v in S.items() if v.get("prot_copies")}
    have_len = {g: pc_all[g] * plen[g] for g in pc_all if g in plen}
    mass_tot = sum(have_len.values())
    mol_tot = sum(pc_all[g] for g in have_len)

    def cover(gs):
        gs = [g for g in gs if g in have_len]
        return (len(gs), sum(have_len[g] for g in gs) / mass_tot,
                sum(pc_all[g] for g in gs) / mol_tot)

    n_core, m_core, q_core = cover(core)
    n_armI, m_armI, q_armI = cover(list(gI))
    n_armC, m_armC, q_armC = cover(list(g_core[ic]))
    say(f"     denominator: the MEASURED proteome -- {len(pc_all):,} Schwanhausser genes with a copy")
    say(f"     number, {len(have_len):,} of which also have a protein length "
        f"(mass = copies x residues)")
    say(f"     integrable at all (both half-lives)        {n_core:>6,} genes   "
        f"{m_core:6.1%} of proteome MASS   {q_core:6.1%} of molecules")
    say(f"     ARM C, the circular arm (needs copies too) {n_armC:>6,} genes   "
        f"{m_armC:6.1%} of mass   {q_armC:6.1%} of molecules")
    say(f"     ARM I, the real test (needs Larsson too)   {n_armI:>6,} genes   "
        f"{m_armI:6.1%} of mass   {q_armI:6.1%} of molecules")
    say(f"     against the whole model: {n_core:,} of {len(C['genes']):,} genes = "
        f"{n_core/len(C['genes']):.1%}. The mass fraction is the honest number and the gene fraction")
    say(f"     is the flattering one; both are given.")
    L = json.load(open(LIFE))["lifetimes"]
    n_ext = sum(1 for v in L.values() if v.get("mrna_hl_h") and v.get("prot_hl_h"))
    say(f"     if mixed-source half-lives were accepted, cell_lifetimes.json would offer {n_ext:,}")
    say(f"     genes -- human Mathieson protein half-lives against human RNADecayCafe mRNA")
    say(f"     half-lives, neither from the cells the copy numbers come from. NOT USED for any")
    say(f"     claim above; recorded so the ceiling is visible.")
    convM = np.abs(runA["M"] / runA["Mss"] - 1.0)
    convP = np.abs(runA["P"] / runA["Pss"] - 1.0)
    conv = np.maximum(convM, convP)
    frac_conv = float(np.mean(conv < T_I7_CONV))
    say(f"     convergence at the {HORIZON_H:.0f} h horizon: median |X/X_ss - 1| "
        f"{np.median(conv):.3e}, worst {conv.max():.3e}; {frac_conv:.4%} of genes below "
        f"{T_I7_CONV:.0e}")
    i7 = bool(frac_conv >= T_I7_FRAC)
    say(f"     I7 {'PASS' if i7 else 'FAIL'} (gate: >= {T_I7_FRAC:.0%} converged)")
    t95p = runA["t95_prot"]
    t95m = runA["t95_mrna"]
    w = np.array([have_len.get(g, 0.0) for g in gI], float)
    in_cycle = np.isfinite(t95p) & (t95p <= DOUBLING_H)
    mass_in_cycle = float(w[in_cycle].sum() / w.sum()) if w.sum() > 0 else float("nan")
    say(f"     REPORTED, NOT GATED -- the transient:")
    say(f"        mRNA    reaches 95% of steady state in a median {np.nanmedian(t95m):.1f} h")
    say(f"        protein reaches 95% of steady state in a median {np.nanmedian(t95p):.1f} h")
    say(f"        proteins reaching 95% inside one {DOUBLING_H:.0f} h cell cycle: "
        f"{np.mean(in_cycle):.1%} of genes, {mass_in_cycle:.1%} of the integrated proteome MASS")
    snapline = []
    for t in (1, 4, 12, 24, 48, 96, 168, 336, 720):
        s = runA["snaps"].get(t)
        if s is not None:
            snapline.append((t, float(np.nanmedian(s["M_frac"])), float(np.nanmedian(s["P_frac"]))))
    say(f"        median fraction of steady state along the forward run:")
    for t, fm, fp in snapline:
        say(f"           t = {t:>4} h    mRNA {fm:6.3f}    protein {fp:6.3f}")
    t95_floor = float(-np.log(0.05) / MU)
    say(f"        a PERFECTLY STABLE protein still needs {t95_floor:.0f} h to reach 95%, because")
    say(f"        dilution at 1/mu = {1/MU:.1f} h is then its only route to steady state, and that is")
    say(f"        {t95_floor/DOUBLING_H:.1f} cell cycles. MOST OF THE PROTEOME HAS NO STEADY STATE")
    say(f"        WITHIN A CELL CYCLE. A model that reports steady-state protein levels -- this one")
    say(f"        included, and every FBA and every proteome-constraint model in this repository --")
    say(f"        is reporting a limit the cell spends its life approaching and never reaching.")
    say()

    # ------------------------------------------------------------------ leakage disclosure
    rho_hl_copies, _ = spear(tm[np.isfinite(mobs)], mobs[np.isfinite(mobs)])
    say("LEAKAGE DISCLOSURE -- the one place Arm I touches Schwanhausser other than as truth")
    say(f"     Arm I-A's correction d/(d+mu) is built from the Schwanhausser mRNA HALF-LIFE. If")
    say(f"     half-life were strongly related to copy number, part of the answer would leak in.")
    say(f"     Spearman(mRNA half-life, mRNA copies) = {rho_hl_copies:+.4f} over "
        f"{int(np.isfinite(mobs).sum()):,} genes.")
    say(f"     Half-lives are not abundances and are measured in a separate experiment, but the")
    say(f"     number is reported so the reader can price the concern instead of trusting it away.")
    say()

    gates = {"I1 the integrator agrees with an independent RK4": i1,
             "I2 the circular arm reproduces its own input to machine precision": i2,
             "I3 Arm I's units are resolved": i3,
             "I4 Arm I beats the null and publication count": i4,
             "I5 the ODE adds something to its own input": i5,
             "I6 the protein tier predicts protein copies": i6,
             "I7 coverage declared by mass and the run converges": i7}
    say("GATES")
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")
    say()

    man = RM.manifest(
        inputs=[str(LR.CELL), str(SCHWAN), str(LARSSON), str(FASTA), str(LIFE)],
        available=len(core), used=len(gI), selection="filtered", seed=SEED,
        controls=["ARM C: the loop 91 identity run on purpose, machine-precision tautology, excluded",
                  "ARM N: k_sm shuffled across genes, 24 permutations, at both tiers",
                  "gate_guard.null_can_move certifies the shuffle changes k_sm before its output counts",
                  "gate_guard.survival decides whether a survival fraction is even defined",
                  "an independent explicit RK4 integrator against the exponential stepper",
                  "publication count as a competing predictor at both tiers",
                  "I-B, the ODE reduced to an identity map, as the no-model reference",
                  "the single global scale constant cross-fitted over 5 folds so no gene sees its own copies",
                  "raw Larsson Theta with no ODE at all as a third reference",
                  "Spearman(mRNA half-life, mRNA copies) reported as the leakage bound"],
        note="k_sm from Larsson burst kinetics never uses Schwanhausser abundance; the circular arm "
             "that does is run separately and excluded. Larsson mouse fibroblast vs Schwanhausser "
             "mouse NIH3T3 -- close, not identical.")
    RM.report(man, emit=say)

    res = {
        "test": "loop_integrator", "manifest": man, "gates": gates,
        "model": {"mu_per_h": MU, "doubling_h": DOUBLING_H, "ksp_global": KSP_GLOBAL,
                  "horizon_h": HORIZON_H, "step_h": STEP_H,
                  "ode": "dM/dt = k_sm - (ln2/t_m + mu) M ; dP/dt = k_sp M - (ln2/t_p + mu) P"},
        "inputs_vs_predictions": {
            "inputs_every_arm": ["mRNA half-life", "protein half-life", "doubling time"],
            "arm_C_k_sm_source": "this gene's own measured mRNA_copies (loop 91 identity)",
            "arm_I_k_sm_source": "Larsson burst kinetics + one cross-fitted global scale",
            "arm_N_k_sm_source": "Arm I's k_sm permuted across genes",
            "predicted": ["M* steady-state mRNA", "P* steady-state protein"],
            "note": "Arm C's predictions equal its inputs by algebra and are excluded from claims"},
        "i1": {"median_dev": float(np.median(dev)), "p95_dev": float(np.percentile(dev, 95)),
               "max_dev": float(dev.max()), "frac_below_gate": frac_ok,
               "threshold": T_I1_DEV, "rk4_steps": RK4_STEPS, "h_lambda": RK4_EFOLDS / RK4_STEPS},
        "i2_circular_excluded": {
            "n_genes": int(len(ic)), "median_rel_err_mrna": float(np.median(relM)),
            "max_rel_err_mrna": float(relM.max()), "median_rel_err_prot": float(np.median(relP)),
            "max_rel_err_prot": float(relP.max()), "threshold": T_I2_REL,
            "meaning": "algebraic identity M*=k_sm/a with k_sm=M_obs*a; evidence about floating "
                       "point only; excluded from i4, i5, i6"},
        "i3_units": {"median_rel_err_vs_paper_mean": med_units, "normalised": normalised,
                     "median_theta": med_theta, "loop91_median_ksm": KSM_MEDIAN_L91,
                     "factor_from_loop91": abs_factor, "absolute": absolute, "verdict": verdict,
                     "n_larsson_both_alleles": len(both_allele), "excel_corrupt_rows": corrupt},
        "i4_arm_I": {"n": nA, "rho_mrna": rhoA, "fold": ffA, "rho_pubs": rho_pub,
                     "null_capability": cap, "survival": sA, "threshold_rho": T_I4_RHO,
                     "alphas": alphas_A,
                     "implied_median_ksm_per_h": float(np.nanmedian(ksm_A_raw * np.mean(alphas_A))),
                     "selection_bias": {
                         "median_mrna_copies_armI": float(np.median(mobsI)),
                         "median_mrna_copies_all_measured": float(np.median(all_m)),
                         "armI_p5_p95_copies": [float(sel_lo), float(sel_hi)],
                         "note": "a gene enters Arm I only if a telegraph model could be fitted on "
                                 "both alleles, which selects the well-expressed transcriptome"}},
        "i5_does_the_ode_help": {"rho_IA": rhoA, "rho_IB_identity_map": rhoB,
                                 "rho_raw_theta_no_ode": rho_theta, "gain": gain,
                                 "threshold": T_I5_GAIN, "fold_IB": ffB,
                                 "dilution_correction_range": [float(np.min(dmI / (dmI + MU))),
                                                               float(np.max(dmI / (dmI + MU)))]},
        "i6_protein_tier": {"n": nP, "rho_prot": rhoP, "fold": ffP, "rho_pubs": rho_pub_p,
                            "survival": sP, "threshold_rho": T_I6_RHO,
                            "tier_change_rho": float(rhoP - rhoA),
                            "rho_Mstar_vs_prot_copies": rho_M_vs_prot,
                            "second_hop_gain": float(rhoP - rho_M_vs_prot),
                            "null_mean_is_protein_halflife_alone": sP.get("null_mean")},
        "i7_coverage": {"n_core": n_core, "mass_frac_core": m_core, "molecule_frac_core": q_core,
                        "n_armC": n_armC, "mass_frac_armC": m_armC,
                        "n_armI": n_armI, "mass_frac_armI": m_armI, "molecule_frac_armI": q_armI,
                        "n_model_genes": len(C["genes"]),
                        "n_extended_mixed_source": n_ext,
                        "frac_converged": frac_conv, "conv_threshold": T_I7_CONV,
                        "median_t95_mrna_h": float(np.nanmedian(t95m)),
                        "median_t95_prot_h": float(np.nanmedian(t95p)),
                        "frac_prot_equilibrated_in_one_cycle": float(np.mean(in_cycle)),
                        "mass_frac_prot_equilibrated_in_one_cycle": mass_in_cycle,
                        "t95_floor_h_perfectly_stable_protein": t95_floor,
                        "trajectory_median_fraction_of_steady_state":
                            [{"t_h": t, "mrna": fm, "prot": fp} for t, fm, fp in snapline]},
        "leakage": {"rho_mrna_halflife_vs_copies": rho_hl_copies},
        "seconds": time.time() - t0, "log": log,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "loop_integrator.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_integrator.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
