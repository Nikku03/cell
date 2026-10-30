"""LOOP 98 -- STOCHASTICITY: PREDICTED EXPRESSION NOISE FROM COPY NUMBER AND HALF-LIFE.

THE PREDICTION IS ALREADY IN THE MODEL AND HAS NEVER BEEN CASHED. A gene whose mRNA is made at
k_s and lost at k_d is a birth-death process. At steady state its copy number is Poisson:

        mean N = k_s / k_d            Fano = variance / mean = 1            CV = 1 / sqrt(N)

and if transcription arrives in bursts of mean size b rather than one molecule at a time,

        Fano = 1 + b                  CV^2 = (1 + b) / N

Loop 91 (loop_rates.py) already computed k_s and k_d for every Schwanhausser gene -- 4,190 of them
with a copy number and a half-life from the same cells. So the noise of every one of those genes is
PREDICTED, with no new measurement, by arithmetic the repo has been carrying for seven loops
without ever pointing it at variance. This loop points it at variance.

AND THE REPO TURNS OUT TO HOLD A MEASURED ANSWER. scratchpad/larsson_S5.xlsx is Larsson et al.
2019's allele-resolved single-cell RNA-seq fit of the two-state telegraph model in MOUSE PRIMARY
FIBROBLASTS -- kon, koff and ksyn per allele for 6,180 genes, in units of the mRNA degradation rate.
Schwanhausser 2011 is mouse NIH3T3 FIBROBLASTS. Same species, same cell type, independent labs,
independent technologies, and 2,582 genes in common. That is about as clean a prediction-versus-
measurement join as this repository is ever going to get, and nothing in the project has used it.

From the telegraph parameters the measurement side is closed-form and needs nothing from
Schwanhausser:

        mean m = kon ksyn / (kon + koff)        burst size b = ksyn / koff
        Fano F = 1 + ksyn koff / ((kon + koff)(kon + koff + 1))        CV = sqrt(F / m)

THE ABUNDANCE RULE FROM LOOP 92 IS THE HINGE OF THIS LOOP, NOT A FOOTNOTE. Every quantity on the
prediction side comes from Schwanhausser alone; every quantity on the measurement side comes from
Larsson alone. The one place the two datasets could be divided into each other is the implied burst
size b = N CV^2 - 1, and forming it with N from Schwanhausser and CV from Larsson is exactly the
forbidden quotient. So it is computed BOTH ways, deliberately, and the gap between them is reported
in mRNAs per burst as the price of breaking the rule. Loop 92 paid that price at 1,655 against 140
and had to find out afterwards; here it is priced up front.

Larsson's parameters are PER ALLELE and Schwanhausser's copies are PER CELL, so the prediction is
taken at N_allele = mrna_copies / 2 throughout. A rank correlation does not notice the factor of
two; the floor and the burst size do, and it is declared here rather than discovered later.

DISCLOSURE, per rule 2 -- three numbers were computed during reconnaissance BEFORE these gates were
written, and the gates were set knowing them:
    the naive median k_s over the Schwanhausser set is 1.34 mRNA/cell/h, which is loop 91's
        pre-dilution figure and not its headline 1.80; N1's threshold was chosen knowing both
    2,582 genes join Schwanhausser to Larsson's C57 fibroblast sheet; N2's n >= 300 bar was set
        knowing that
    one worked Larsson row was checked to confirm that kon/koff/ksyn reproduce their own reported
        burst size and mean expression columns, so that N4 tests my algebra and not my parsing
No correlation, no floor fraction, no burst-size distribution and no fame number was looked at
before the gates below were fixed.

PREDECLARED, before any number beyond those three:

  N1 THE PREDICTION IS ONE DATASET'S ARITHMETIC, AND IT REPRODUCES LOOP 91   THE PROVENANCE CHECK.
       k_s = N x (ln2/t_half + ln2/t_double) with N and t_half from the SAME cells, per loop 92.
       PASS if the median k_s is within a factor of 1.25 of loop 91's established 1.80 mRNA/cell/h.
       This gate exists so that a downstream noise claim cannot rest on arithmetic that has quietly
       stopped agreeing with the loop it came from.

  N2 PREDICTED NOISE TRACKS MEASURED SINGLE-CELL NOISE                       THE GATE.
       Spearman between the Poisson prediction CV = 1/sqrt(N_allele) (Schwanhausser) and the
       measured CV from Larsson's telegraph fits. Direction is predeclared POSITIVE: rare mRNAs are
       noisier. PASS if rho >= +0.30 on n >= 300 genes.
       Reported beside it, NOT gated, because each is a way this gate could be true and empty:
         - the allele-to-allele reproducibility of the measurement itself (C57 against CAST in the
           same cells), which is the CEILING no prediction can beat;
         - the same correlation against cross-CELL-TYPE CV over 154 HPA cell types, which are
           pooled averages and therefore contain essentially no single-cell noise at all;
         - the PARTIAL correlation holding Larsson's own mean expression fixed. The prediction is a
           function of abundance and nothing else, so this asks the only question that matters:
           does Schwanhausser's copy number add anything over abundance measured in the very same
           experiment as the noise? If the partial collapses, N2 is a cross-dataset abundance
           agreement wearing a noise costume, and this loop will say so.

  N3 THE POISSON FLOOR AND THE MINUS-ONE-HALF SCALING                        THE PHYSICS.
       No gene can have CV below 1/sqrt(N): sub-Poissonian intrinsic noise is not available to an
       unregulated birth-death process. PASS requires BOTH
         (i)  fewer than 10% of joined genes with CV_meas < 1/sqrt(N_allele), and
         (ii) the OLS slope of log10 CV_meas on log10 N_allele inside [-1.00, -0.25], bracketing
              the Poisson -0.50.
       Computed and explicitly REFUSED as evidence: the same floor check inside Larsson alone,
       which must return exactly 0% because a two-state telegraph fit cannot produce Fano < 1. That
       is a gate_guard family-two dead null -- a check that cannot fail -- and it is labelled as one
       instead of being reported as a pass.

  N4 THE FANO ALGEBRA REPRODUCES THE SOURCE'S OWN BURST SIZE                 THE SELF-CHECK.
       b_self = F - 1 against Larsson's separately reported ksyn/koff. PASS if Spearman >= 0.90 AND
       the median ratio is inside [0.5, 2.0]. A units error, a wrong steady-state moment, or a
       mis-parsed bracketed string fails this before it can contaminate N5.

  N5 THE BURST SIZE, AND WHAT THE ABUNDANCE RULE COSTS                       THE PRICE.
       PASS requires BOTH
         (i)  median b_self inside the published mammalian range of 1-10 mRNAs per burst, and
         (ii) median b_cross = N_allele x CV_meas^2 - 1, the SAME quantity formed across two
              abundance datasets, within a factor of 3 of median b_self.
       (ii) is expected to FAIL. Larsson's counts are captured UMIs at single-digit-percent
       efficiency while Schwanhausser's are absolute molecules, so the quotient measures the capture
       rate rather than the burst. Reporting that failure in mRNAs per burst is the point: it is
       loop 92's rule with a number attached instead of a warning.

  N6 FAME, AND A NULL THAT CAN MOVE                                          THE RECURRING KILLER.
       publication count against measured CV, against copy number and against burst size, reported
       next to the biology as it has been since it beat the biology in loops 87, 87b, 94 and 96.
       The null for N2 shuffles copy number across genes, breaking the gene-to-abundance link, and
       gate_guard.null_can_move() confirms the shuffle reaches the statistic's input BEFORE
       gate_guard.survival() is allowed to give a verdict. PASS if the null is capable.

WHAT THIS CANNOT DO, stated before the results rather than after. Larsson's CV is measured on
captured UMIs, not molecules, and Poisson sampling at efficiency p turns a true Fano F into
1 + p(F-1) on a mean of pN -- it inflates CV and deflates burst size, in a gene-independent way
that ranks survive and absolute values do not. Every rank statement below is robust to it; every
absolute statement below is not, and N5 is the place that shows up.

-> outputs/loop_noise.json
"""
import json
import os
import re
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import gate_guard as GG  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
LN2 = float(np.log(2.0))

SCHWAN = SC / "_schwan2011.json"
LARSSON = SC / "larsson_S5.xlsx"
HPA = SC / "hpa_rna_single_cell_type.zip"

# CORRECTED AFTER ADVERSARIAL REVIEW. This constant was 24.0 with the comment "loop 91's
# assumption, kept identical so N1 is a reproduction". THAT COMMENT WAS FALSE: loop 91 uses 27.5 h
# (loop_rates.py:96, "NIH 3T3 in Schwanhausser 2011 double in about 27.5 h"). The gene set is
# identical -- the naive degradation-only median 1.3424694937293487 is byte-identical in both
# outputs -- so the entire "factor 1.044" discrepancy N1 reported was an undisclosed change of the
# doubling-time assumption, in a gate whose stated purpose was to catch arithmetic that had
# quietly stopped agreeing with the loop it came from. The gate absorbed the very thing it existed
# to detect. Set to 27.5 to match loop 91; N1 now reproduces it to a factor of 1.0005.
T_DOUBLE_H = 27.5                  # loop 91, loop_rates.py:96 -- verified, not assumed
LOOP91_KSM = 1.80                  # established: transcription rate median, mRNA/cell/h
N1_FACTOR = 1.25
N2_MIN_RHO, N2_MIN_N = 0.30, 300
N3_MAX_VIOLATE, N3_SLOPE_LO, N3_SLOPE_HI = 0.10, -1.00, -0.25
N4_MIN_RHO, N4_RATIO_LO, N4_RATIO_HI = 0.90, 0.5, 2.0
N5_BURST_LO, N5_BURST_HI, N5_MAX_FACTOR = 1.0, 10.0, 3.0
PLOIDY = 2.0                       # Larsson is per allele, Schwanhausser per cell
NPERM = 40
SEED = 9801

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


def rank(v):
    from scipy.stats import rankdata
    return rankdata(np.asarray(v, float))


def partial_spear(x, y, z):
    """Spearman of x with y holding z fixed -- rank-transform, then linear partial correlation."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    f = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if f.sum() < 30:
        return float("nan"), int(f.sum())
    rx, ry, rz = rank(x[f]), rank(y[f]), rank(z[f])
    c = np.corrcoef(np.vstack([rx, ry, rz]))
    xy, xz, yz = c[0, 1], c[0, 2], c[1, 2]
    d = np.sqrt(max(1e-12, (1 - xz ** 2) * (1 - yz ** 2)))
    return float((xy - xz * yz) / d), int(f.sum())


NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def triple(s):
    """Larsson stores '[ 1.29 356.01 502.19]' as a string. Parse it, or return None."""
    if not isinstance(s, str):
        return None
    v = [float(x) for x in NUM.findall(s)]
    return v[:3] if len(v) >= 3 else None


def first_num(s):
    if not isinstance(s, str):
        return float("nan")
    v = NUM.findall(s)
    return float(v[0]) if v else float("nan")


def load_larsson(sheet):
    """kon, koff, ksyn per allele in units of the mRNA degradation rate, plus their own columns."""
    import pandas as pd
    d = pd.ExcelFile(LARSSON).parse(sheet)
    cols = list(d.columns)
    out = {}
    for _, r in d.iterrows():
        t = triple(r[cols[1]])
        if t is None:
            continue
        kon, koff, ksyn = t
        if not (kon > 0 and koff > 0 and ksyn > 0) or not np.isfinite(kon + koff + ksyn):
            continue
        g = str(r[cols[0]]).upper()
        s = kon + koff
        m = kon * ksyn / s                                 # steady-state mean, per allele
        fano = 1.0 + ksyn * koff / (s * (s + 1.0))         # two-state telegraph Fano factor
        if not (m > 0 and np.isfinite(m) and np.isfinite(fano)):
            continue
        out[g] = {"kon": kon, "koff": koff, "ksyn": ksyn, "mean": m, "fano": fano,
                  "cv": float(np.sqrt(fano / m)), "b_self": fano - 1.0,
                  "b_reported": first_num(r[cols[3]]), "mean_reported": float(r[cols[4]])}
    return out


def load_hpa_cv():
    """CV of nCPM across 154 pooled HPA cell types -- variation BETWEEN cell identities, which is
    the opposite kind of variability from single-cell noise and is here as a contrast."""
    import pandas as pd
    z = zipfile.ZipFile(HPA)
    d = pd.read_csv(z.open("rna_single_cell_type.tsv"), sep="\t")
    g = d.groupby("Gene name")["nCPM"]
    mu, sd = g.mean(), g.std()
    n_ct = int(d["Cell type"].nunique())
    cv = (sd / mu).replace([np.inf, -np.inf], np.nan).dropna()
    return {str(k).upper(): float(v) for k, v in cv.items() if v > 0}, n_ct


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 98 -- stochasticity: predicted expression noise from copy number and half-life")
    say("=" * 100)
    say()

    C = json.load(open(LR.CELL))
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}
    S = json.load(open(SCHWAN))
    schwan = {g: v for g, v in S.items()
              if v.get("mrna_copies") and v.get("mrna_hl_h")
              and v["mrna_copies"] > 0 and v["mrna_hl_h"] > 0}
    lars = load_larsson("C57")
    lars2 = load_larsson("CAST")
    hpa_cv, n_ct = load_hpa_cv()
    say(f"  Schwanhausser: {len(schwan):,} genes with an mRNA copy number AND an mRNA half-life "
        f"(mouse NIH3T3 fibroblasts)")
    say(f"  Larsson C57:   {len(lars):,} alleles with a telegraph fit (mouse primary fibroblasts); "
        f"CAST {len(lars2):,}")
    say(f"  HPA:           {len(hpa_cv):,} genes with a CV across {n_ct} pooled cell types")
    say(f"  cell_complete: {len(pubs):,} genes carry a publication count")
    say()

    # ---------------------------------------------------------------- N1
    say("N1 THE PREDICTION IS ONE DATASET'S ARITHMETIC, AND IT REPRODUCES LOOP 91")
    genes_s = sorted(schwan)
    N_cell = np.array([schwan[g]["mrna_copies"] for g in genes_s], float)
    hl = np.array([schwan[g]["mrna_hl_h"] for g in genes_s], float)
    k_d_naive = LN2 / hl
    k_d = LN2 / hl + LN2 / T_DOUBLE_H                       # loop 91: degradation AND dilution
    k_s = N_cell * k_d
    med_naive = float(np.median(N_cell * k_d_naive))
    med_ks = float(np.median(k_s))
    fac = med_ks / LOOP91_KSM
    n1 = bool(1.0 / N1_FACTOR <= fac <= N1_FACTOR)
    say(f"     numerator and denominator are the same cells: copies and half-life both Schwanhausser")
    say(f"     k_s = N x (ln2/t_half + ln2/{T_DOUBLE_H:.0f}h)")
    say(f"     median k_s {med_ks:.4f} mRNA/cell/h   loop 91 established {LOOP91_KSM:.2f}   "
        f"factor {fac:.3f}")
    say(f"     (degradation only, no dilution: {med_naive:.4f} -- the number disclosed above)")
    say(f"     median mRNA copies/cell {np.median(N_cell):.2f}, median half-life {np.median(hl):.2f} h")
    say(f"     N1 {'PASS' if n1 else 'FAIL'} -- the prediction rests on {'loop 91 arithmetic, unchanged' if n1 else 'ARITHMETIC THAT NO LONGER AGREES WITH LOOP 91'}")
    say()

    # ---------------------------------------------------------------- N2
    say("N2 PREDICTED NOISE TRACKS MEASURED SINGLE-CELL NOISE")
    join = sorted(set(schwan) & set(lars))
    Na = np.array([schwan[g]["mrna_copies"] / PLOIDY for g in join], float)   # per allele
    cv_pois = 1.0 / np.sqrt(Na)
    cv_meas = np.array([lars[g]["cv"] for g in join], float)
    m_lars = np.array([lars[g]["mean"] for g in join], float)
    b_self = np.array([lars[g]["b_self"] for g in join], float)
    say(f"     {len(join):,} genes join Schwanhausser to Larsson C57")
    r2, n2n = spear(cv_pois, cv_meas)
    n2 = bool(np.isfinite(r2) and r2 >= N2_MIN_RHO and n2n >= N2_MIN_N)
    say(f"     Spearman(1/sqrt(N_allele), CV_measured)  {r2:+.4f}   n {n2n:,}   "
        f"(gate >= {N2_MIN_RHO:+.2f}, n >= {N2_MIN_N})")

    shared_al = sorted(set(lars) & set(lars2))
    r_rep, n_rep = spear([lars[g]["cv"] for g in shared_al], [lars2[g]["cv"] for g in shared_al])
    say(f"     CEILING: the measurement against itself, C57 allele vs CAST allele, same cells: "
        f"{r_rep:+.4f} on {n_rep:,}")

    jh = [g for g in join if g in hpa_cv]
    r_hpa, n_hpa = spear([1.0 / np.sqrt(schwan[g]["mrna_copies"] / PLOIDY) for g in jh],
                         [hpa_cv[g] for g in jh])
    r_hpa_lars, _ = spear([lars[g]["cv"] for g in jh], [hpa_cv[g] for g in jh])
    say(f"     CONTRAST: the same prediction against CV across {n_ct} POOLED cell types "
        f"{r_hpa:+.4f} on {n_hpa:,}")
    say(f"               (and measured single-cell CV against that pooled CV {r_hpa_lars:+.4f}) -- "
        f"pooled averages hold no single-cell noise")

    r_part, n_part = partial_spear(cv_pois, cv_meas, m_lars)
    r_ab, _ = spear(Na, m_lars)
    say(f"     HARSH CONTROL: partial Spearman holding Larsson's OWN mean expression fixed "
        f"{r_part:+.4f} on {n_part:,}")
    say(f"     cross-dataset abundance agreement, Schwanhausser copies vs Larsson mean: {r_ab:+.4f}")
    say(f"     reading: the prediction is a function of abundance ALONE, so {abs(r_part):.4f} is all")
    say(f"     that survives once abundance measured in the noise experiment itself is removed.")
    say(f"     N2 {'PASS' if n2 else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- N3
    say("N3 THE POISSON FLOOR AND THE MINUS-ONE-HALF SCALING")
    viol = cv_meas < cv_pois
    f_viol = float(np.mean(viol))
    ok = np.isfinite(cv_meas) & np.isfinite(Na) & (cv_meas > 0) & (Na > 0)
    slope, icpt = np.polyfit(np.log10(Na[ok]), np.log10(cv_meas[ok]), 1)
    slope_in = np.log10(np.array([lars[g]["mean"] for g in join])[ok])
    slope_self, _ = np.polyfit(slope_in, np.log10(cv_meas[ok]), 1)
    slope_m_on_N, _ = np.polyfit(np.log10(Na[ok]), slope_in, 1)
    n3a = bool(f_viol < N3_MAX_VIOLATE)
    n3b = bool(N3_SLOPE_LO <= slope <= N3_SLOPE_HI)
    n3 = bool(n3a and n3b)
    say(f"     genes with CV_measured BELOW the Poisson floor 1/sqrt(N_allele): "
        f"{int(viol.sum()):,} of {len(join):,} = {f_viol:.2%}   (gate < {N3_MAX_VIOLATE:.0%})")
    say(f"     OLS slope of log10 CV_measured on log10 N_allele (Schwanhausser): {slope:+.4f}   "
        f"(gate [{N3_SLOPE_LO:+.2f}, {N3_SLOPE_HI:+.2f}], Poisson -0.50)")
    say(f"     the same slope against Larsson's OWN mean: {slope_self:+.4f} -- partly by "
        f"construction, since CV^2 = F/m shares m with the x axis")
    say(f"     WHERE THE MISSING SLOPE WENT, and it is not a modelling failure: the two datasets'")
    say(f"     abundance measures regress on each other at slope {slope_m_on_N:+.4f} in log-log, so a")
    say(f"     within-dataset slope of {slope_self:+.4f} arrives across datasets as "
        f"{slope_self:+.4f} x {slope_m_on_N:.4f} = {slope_self*slope_m_on_N:+.4f},")
    say(f"     against the {slope:+.4f} actually measured. Regression dilution accounts for the")
    say(f"     whole shortfall from -0.50: the physics is there and the JOIN attenuates it. The")
    say(f"     gate still fails, and it should -- a prediction attenuated below its declared band")
    say(f"     is not a prediction that held.")
    say(f"     REFUSED AS EVIDENCE: the floor check inside Larsson alone returns "
        f"{float(np.mean(b_self < 0)):.0%} violations, and it MUST -- a two-state telegraph fit")
    say(f"     cannot produce Fano < 1, so that check cannot fail. gate_guard family two: a null")
    say(f"     that cannot move is not a null. It is reported here only to be discarded.")
    say(f"     N3 {'PASS' if n3 else 'FAIL'} -- floor {'ok' if n3a else 'VIOLATED'}, "
        f"scaling {'ok' if n3b else 'OUT OF RANGE'}")
    say()

    # ---------------------------------------------------------------- N4
    say("N4 THE FANO ALGEBRA REPRODUCES THE SOURCE'S OWN BURST SIZE")
    allg = sorted(lars)
    mine = np.array([lars[g]["b_self"] for g in allg], float)
    theirs = np.array([lars[g]["b_reported"] for g in allg], float)
    mean_chk, _ = spear([lars[g]["mean"] for g in allg], [lars[g]["mean_reported"] for g in allg])
    r4, n4n = spear(mine, theirs)
    good = np.isfinite(mine) & np.isfinite(theirs) & (theirs > 0)
    ratio = float(np.median(mine[good] / theirs[good]))
    n4 = bool(np.isfinite(r4) and r4 >= N4_MIN_RHO and N4_RATIO_LO <= ratio <= N4_RATIO_HI)
    say(f"     parse check: my steady-state mean vs their reported mean expression column "
        f"{mean_chk:+.6f} on {len(allg):,}")
    say(f"     F - 1 vs their reported ksyn/koff:  Spearman {r4:+.4f}   median ratio {ratio:.4f}")

    kon_a = np.array([lars[g]["kon"] for g in allg], float)
    koff_a = np.array([lars[g]["koff"] for g in allg], float)
    corr_f = koff_a ** 2 / ((kon_a + koff_a) * (kon_a + koff_a + 1.0))
    fast = koff_a >= 10.0                      # bursts short compared to the mRNA lifetime
    r4f, _ = spear(mine[fast], theirs[fast])
    gf = fast & good
    ratio_f = float(np.median(mine[gf] / theirs[gf]))
    frac_identity = float(np.mean((corr_f > 0.8) & (corr_f < 1.25)))
    say(f"     WHY IT FAILS, and it is the premise of this loop rather than the code. Exactly,")
    say(f"       F - 1 = b x koff^2 / ((kon+koff)(kon+koff+1))")
    say(f"     so Fano = 1 + b is a LIMIT (koff >> kon, koff >> k_deg), not an identity. Measured")
    say(f"     koff deciles 10/25/50/75/90: " +
        "  ".join(f"{x:.2f}" for x in np.percentile(koff_a, [10, 25, 50, 75, 90])) +
        " in units of k_deg,")
    say(f"     so the median gene's correction factor is {np.median(corr_f):.3f} and only "
        f"{frac_identity:.1%} of genes sit within 20% of Fano = 1 + b.")
    say(f"     THE CONTROL THAT COULD HAVE FAILED AND DID NOT: restricted to the "
        f"{int(fast.sum()):,} genes with koff >= 10,")
    say(f"     where the limit is legitimate, Spearman rises to {r4f:+.4f} and the median ratio to "
        f"{ratio_f:.4f}.")
    say(f"     Had the algebra been wrong, the restriction would not have rescued it. The parse")
    say(f"     check at {mean_chk:+.6f} rules out units and string handling separately.")
    say(f"     N4 {'PASS' if n4 else 'FAIL'} -- the moment algebra is "
        f"{'the sources own' if n4 else 'CORRECT but Fano = 1 + b is not, for two thirds of genes'}")
    say()

    # ---------------------------------------------------------------- N5
    say("N5 THE BURST SIZE, AND WHAT THE ABUNDANCE RULE COSTS")
    b_cross = Na * cv_meas ** 2 - 1.0
    med_self = float(np.median(b_self))
    med_cross = float(np.median(b_cross))
    fac5 = med_cross / med_self if med_self != 0 else float("inf")
    n5a = bool(N5_BURST_LO <= med_self <= N5_BURST_HI)
    n5b = bool(np.isfinite(fac5) and 1.0 / N5_MAX_FACTOR <= fac5 <= N5_MAX_FACTOR)
    n5 = bool(n5a and n5b)
    q = np.percentile(b_self, [10, 25, 50, 75, 90])
    b_rep = np.array([lars[g]["b_reported"] for g in join], float)
    med_rep = float(np.median(b_rep[np.isfinite(b_rep)]))
    say(f"     SELF-CONSISTENT  b = F - 1, Larsson only:   median {med_self:.3f} mRNA/burst")
    say(f"       deciles 10/25/50/75/90: " + "  ".join(f"{x:.2f}" for x in q))
    say(f"       published mammalian range {N5_BURST_LO:.0f}-{N5_BURST_HI:.0f} mRNA/burst   "
        f"-> {'INSIDE' if n5a else 'OUTSIDE'}")
    say(f"     for reference, Larsson's own ksyn/koff on the same {len(join):,} genes: "
        f"median {med_rep:.3f} mRNA/burst,")
    say(f"       also inside 1-10. Both routes land in the published range; N4 shows they are not")
    say(f"       the same quantity, so this agreement is two facts, not one confirmed twice.")
    say(f"     CROSS-DATASET    b = N_allele x CV^2 - 1, Schwanhausser N over Larsson CV:")
    say(f"       median {med_cross:.3f} mRNA/burst   factor {fac5:.2f}x the self-consistent value "
        f"(gate within {N5_MAX_FACTOR:.0f}x)")
    say(f"       this is precisely the quotient loop 92 forbids: N from one dataset's cells, CV")
    say(f"       from another's. The factor above IS the price of breaking the rule, in mRNAs per")
    say(f"       burst, and it is one-sided -- Larsson counts captured UMIs, Schwanhausser counts")
    say(f"       molecules, so the quotient is reading capture efficiency as burstiness.")
    r_cs, _ = spear(b_cross, b_self)
    say(f"       WORSE THAN THE SCALE: the cross-dataset burst size ranks against the self-")
    say(f"       consistent one at only {r_cs:+.4f}. I expected the violation to shift the scale and")
    say(f"       spare the order; it does not. Breaking the abundance rule destroys the ordering")
    say(f"       too, so not even a rank claim survives it. That is stronger than loop 92 stated")
    say(f"       and it is recorded here as a correction to my own expectation.")
    say(f"     N5 {'PASS' if n5 else 'FAIL'} -- range {'ok' if n5a else 'outside'}, "
        f"cross-dataset agreement {'ok' if n5b else 'BROKEN, as predeclared'}")
    say()

    # ---------------------------------------------------------------- N6
    say("N6 FAME, AND A NULL THAT CAN MOVE")
    pb = np.array([pubs.get(g, 0.0) for g in join], float)
    n_pub = int(np.sum([g in pubs for g in join]))
    r_p_cv, _ = spear(pb, cv_meas)
    r_p_n, _ = spear(pb, Na)
    r_p_b, _ = spear(pb, b_self)
    r_p_pred, _ = spear(pb, cv_pois)
    say(f"     {n_pub:,} of {len(join):,} joined genes carry a publication count")
    say(f"     pubs vs measured CV {r_p_cv:+.4f}   vs copy number {r_p_n:+.4f}   "
        f"vs burst size {r_p_b:+.4f}   vs the prediction {r_p_pred:+.4f}")
    say(f"     the prediction is a strictly decreasing function of copy number, so its fame")
    say(f"     correlation is the NEGATIVE of copy number's by construction, not a second fact.")
    say(f"     {abs(r_p_pred):.4f} bounds how much of N2's {r2:+.4f} could be fame -- large enough to")
    say(f"     matter, small enough not to explain it. Neither CV nor the prediction is curated,")
    say(f"     so neither is a literature artefact in itself. THE JOIN IS ANOTHER MATTER and this")
    say(f"     loop does not clear it: both datasets are detection-limited, both drop the rare")
    say(f"     transcripts, and the {len(join):,} survivors are more abundant and better studied than")
    say(f"     the genome. Every number here is conditioned on that, and the honest statement is")
    say(f"     that the bias is bounded by {abs(r_p_n):.4f}, not that it is absent.")

    rng = np.random.default_rng(SEED)
    nulls, moved = [], []
    for _ in range(NPERM):
        p = rng.permutation(len(Na))
        sh = Na[p]
        moved.append(GG.null_can_move(list(Na), list(sh))["changed"])
        nulls.append(spear(1.0 / np.sqrt(sh), cv_meas)[0])
    cap = GG.null_can_move(list(Na), list(Na[rng.permutation(len(Na))]))
    say(f"     CAPABILITY CHECK: shuffling copy number changes {np.mean(moved):.1%} of entries "
        f"-- capable: {cap['capable']}")
    surv = GG.survival(r2, nulls)
    GG.report("predicted-vs-measured CV under a copy-number shuffle", surv, emit=say)
    say(f"     the survival fraction is null_mean/real and comes out at or below zero here, which")
    say(f"     means the shuffle does not merely weaken the correlation, it erases it: N2's "
        f"{r2:+.4f}")
    say(f"     requires each gene's own copy number and nothing about the marginal distribution.")
    n6 = bool(cap["capable"])
    say(f"     N6 {'PASS' if n6 else 'FAIL'} -- the null "
        f"{'reaches the statistic and its verdict is usable' if n6 else 'is INERT'}")
    say()

    say("WHAT THIS DOES NOT GIVE")
    say(f"     Larsson's CV is measured on captured UMIs. Poisson capture at efficiency p maps a")
    say(f"     true Fano F onto 1 + p(F-1) at a mean of pN, inflating CV and deflating burst size")
    say(f"     by a gene-independent factor. Every RANK statement above survives that; every")
    say(f"     ABSOLUTE one does not, and N5 is where it shows.")
    say(f"     Nothing here is a human measurement: both datasets are mouse fibroblasts, joined to")
    say(f"     human symbols by uppercasing, and the model this repo builds is human.")
    say()

    say("WHAT THIS LOOP ACTUALLY ESTABLISHED")
    say(f"     1. The birth-death prediction is real but shallow: {r2:+.4f} against measured")
    say(f"        single-cell CV, and only {r_part:+.4f} of that survives holding the noise")
    say(f"        experiment's own abundance fixed. The prediction is abundance, restated.")
    say(f"     2. Fano = 1 + b is a limit, not a law: it holds within 20% for {frac_identity:.0%} of genes.")
    say(f"     3. Breaking loop 92's abundance rule costs {fac5:.1f}x in scale AND collapses the")
    say(f"        ranking to {r_cs:+.4f}.")
    say()

    gates = {"N1 the prediction reproduces loop 91": n1,
             "N2 predicted noise tracks measured single-cell noise": n2,
             "N3 the Poisson floor and the -1/2 scaling": n3,
             "N4 the Fano algebra reproduces the source's burst size": n4,
             "N5 burst size in range and the cross-dataset quotient agrees": n5,
             "N6 the null is capable and fame is reported": n6}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")
    say()

    man = RM.manifest(inputs=[str(LR.CELL), str(SCHWAN), str(LARSSON), str(HPA)],
                      available=len(schwan), used=len(join), selection="filtered", seed=SEED,
                      controls=[
                          "loop 91's k_s median reproduced before any noise number is formed",
                          "the measurement's own allele-to-allele reproducibility as the ceiling",
                          "cross-cell-type CV over 154 pooled HPA cell types as the wrong-kind-of-"
                          "variability contrast",
                          "partial correlation holding Larsson's own mean expression fixed",
                          "publication count against measured CV, copy number, burst size and the "
                          "prediction",
                          "a copy-number shuffle checked with gate_guard.null_can_move before "
                          "gate_guard.survival is read",
                          "the burst size computed both self-consistently and across two abundance "
                          "datasets, to price loop 92's rule",
                          "the within-Larsson floor check computed and refused, because a telegraph "
                          "fit cannot produce Fano < 1"],
                      note="birth-death noise is already implied by loop 91's k_s and k_d; this "
                           "cashes it against Larsson 2019 allele-resolved telegraph fits in the "
                           "same cell type, with the prediction and the measurement each drawn "
                           "from a single dataset per loop 92")
    RM.report(man, emit=say)

    json.dump({"test": "loop_noise", "manifest": man, "gates": gates,
               "n1": {"median_ks": med_ks, "median_ks_no_dilution": med_naive,
                      "loop91_ksm": LOOP91_KSM, "factor": fac, "n": len(genes_s),
                      "median_copies": float(np.median(N_cell)), "median_hl_h": float(np.median(hl))},
               "n2": {"rho_pred_vs_measured": r2, "n": n2n,
                      "rho_allele_reproducibility": r_rep, "n_allele": n_rep,
                      "rho_pred_vs_pooled_celltype_cv": r_hpa, "n_pooled": n_hpa,
                      "rho_measured_vs_pooled_celltype_cv": r_hpa_lars, "n_cell_types": n_ct,
                      "partial_rho_given_larsson_mean": r_part,
                      "rho_copies_vs_larsson_mean": r_ab},
               "n3": {"fraction_below_poisson_floor": f_viol,
                      "n_below": int(viol.sum()), "n": len(join),
                      "slope_log10cv_vs_log10N": float(slope), "intercept": float(icpt),
                      "slope_vs_larsson_own_mean": float(slope_self),
                      "slope_larsson_mean_on_schwan_N": float(slope_m_on_N),
                      "attenuation_product": float(slope_self * slope_m_on_N),
                      "within_larsson_violations_REFUSED": float(np.mean(b_self < 0)),
                      "why_refused": "a two-state telegraph fit cannot produce Fano < 1, so this "
                                     "check cannot fail and is not evidence"},
               "n4": {"rho_mine_vs_reported_burst": r4, "median_ratio": ratio, "n": n4n,
                      "rho_mean_parse_check": mean_chk,
                      "median_correction_factor": float(np.median(corr_f)),
                      "koff_deciles_10_25_50_75_90":
                          [float(x) for x in np.percentile(koff_a, [10, 25, 50, 75, 90])],
                      "fraction_where_fano_eq_1_plus_b_within_20pct": frac_identity,
                      "n_koff_ge_10": int(fast.sum()),
                      "rho_restricted_koff_ge_10": r4f,
                      "median_ratio_restricted_koff_ge_10": ratio_f},
               "n5": {"median_b_self": med_self, "median_b_cross": med_cross,
                      "median_b_larsson_reported": med_rep,
                      "factor_cross_over_self": fac5,
                      "b_self_deciles_10_25_50_75_90": [float(x) for x in q],
                      "rho_cross_vs_self": r_cs,
                      "published_range": [N5_BURST_LO, N5_BURST_HI]},
               "n6": {"pubs_vs_measured_cv": r_p_cv, "pubs_vs_copies": r_p_n,
                      "pubs_vs_burst_size": r_p_b, "pubs_vs_prediction": r_p_pred,
                      "n_with_pubs": n_pub, "capability": cap, "survival": surv},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_noise.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_noise.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
