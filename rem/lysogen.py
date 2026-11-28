"""Lambda lysogen stability, gate L0: the parameter-provenance audit, and what it killed.

WHY THIS MODULE EXISTS AND WHY IT STOPS WHERE IT STOPS. The design this implements was to be the
first genuine OUT-OF-SAMPLE prediction in the project: calibrate a lysogeny model on wild-type
lambda, then predict the spontaneous switching rate of operator/promoter MUTANTS that had no part
in the calibration. The design came with an explicit trap attached to it, and with an explicit
instruction about what to do if the trap could not be disarmed:

    If the published parameter set cannot be assembled with clear provenance, stop and report
    that rather than substituting plausible values. The whole design rests on knowing which
    numbers were fitted.

That instruction is the reason this file contains a provenance table and a power calculation and
NOT a simulation. L0 is the gate that decides whether anything downstream is allowed to run. It
returns PARTIAL, and the partial half is fatal to the test as specified. The rest of this
docstring is the evidence for that verdict, written before the code below was executed.

=================================================================================================
THE GATES. L0 RUNS HERE. L1-L5 ARE HELD AND THE REASON IS RECORDED, NOT DEFERRED SILENTLY.
=================================================================================================

L0  PARAMETER PROVENANCE. Every number entering the model is tagged on TWO AXES -- what it is
    (MEASURED, FITTED, DERIVED) and whether this session could read its value (retrieved or not)
    -- each with a retrievable source. The axes are independent and collapsing them inverts the
    answer: the first version of this file had one axis and reported "FITTED: 0" for a parameter
    set whose two most important numbers are both fitted-but-unread. A parameter set in which the
    fitted numbers cannot be SEPARATED from the measured ones fails L0 outright, because the trap
    below is then undetectable rather than absent; a set where the split is known but the values
    are missing is PARTIAL, which is a different and more recoverable failure.

L1  WILD-TYPE CONSISTENCY. Reproduce the lysogen's CI abundance and switching rate. THIS IS NOT A
    TEST and this module says so in the code, not only in prose: L1 is allowed to return
    CONSISTENT or INCONSISTENT and is forbidden to return SUPPORTED.

L2  FALSIFIABILITY OF L1. Before L1 is quoted at all, ask whether L1 COULD have failed: compare
    the predicted wild-type switching rate against the experimental DETECTION FLOOR. A prediction
    that sits below the floor agrees with the measurement no matter what the model says, and the
    agreement is then a fact about the assay, not about the model. (Ledger defect P: a gate that
    cannot return its own pass condition on perfect evidence.)

L3  THE OUT-OF-SAMPLE TEST. Predict switching rates for promoter/operator mutants with NO
    refitting, and compare against measured mutant stabilities.

L3a POWER OF L3, COMPUTED BEFORE L3 RUNS. Count how many mutants in the test set are actually
    DISCRIMINATING -- neither censored at the detection floor nor censored at "too unstable to
    lysogenize at all" -- and compute the best p-value the surviving set could produce if the
    model were perfect. If that best-case p-value exceeds the significance bar, L3 is a bar above
    its own achievable ceiling (ledger defect N, fourth instance) and must not be run and quoted.

L4  DIRECTION. If L3 has power, does the model get the ORDERING of mutant stabilities right,
    which is a weaker and more robust claim than getting the magnitudes right.

L5  HONEST CLOSE. Whatever L3 returns, record it. A refuted model is a result. A model that
    cannot be tested is NOT a result and must not be written up as one.

=================================================================================================
L0 EVIDENCE. WHAT THE LITERATURE SEARCH RETURNED, INCLUDING A CORRECTION TO MY OWN EARLIER REPORT.
=================================================================================================

All bibliographic records below are from PubMed. An earlier pass over this same search concluded
that NONE of the source papers was retrievable and recommended stopping. That conclusion was
WRONG on two of them and is corrected here rather than quietly replaced -- the correction is the
reason the line is alive at all.

  UNRETRIEVED -- the lineage the design named.
    Shea MA, Ackers GK (1985) J Mol Biol 181:211-230, PMID 3157005,
        doi:10.1016/0022-2836(85)90086-5.  Holds the O_R free-energy table. No PMC record.
    Aurell E, Sneppen K (2002) Phys Rev Lett 88:048101, doi:10.1103/PhysRevLett.88.048101.
        Physics journal; no PubMed record at all.
    Aurell E, Brown S, Johanson J, Sneppen K (2002) Phys Rev E 65:051914,
        doi:10.1103/PhysRevE.65.051914.  Same; no PubMed record.

  RECORD FOUND, FULL TEXT NOT SERVED -- indexed in PMC but the text body came back empty.
    Little JW, Shepley DP, Wert DW (1999) EMBO J 18:4299-4307, PMID 10428968, PMC1171506,
        doi:10.1093/emboj/18.15.4299.  The O_R1/O_R3 symmetric-operator mutant phages.
    Little JW, Michalowski CB (2010) J Bacteriol 192:6064-6076, PMID 20870769, PMC2976446,
        doi:10.1128/JB.00726-10.  The intrinsic switching rate and the lambda-prm240 series.
    Santillan M, Mackey MC (2004) Biophys J 86:75-84, PMID 14695251, PMC1303838,
        doi:10.1016/S0006-3495(04)74085-0.
    Michalowski CB, Short MD, Little JW (2004) J Bacteriol 186:7988-7999, PMID 15547271,
        PMC529058, doi:10.1128/JB.186.23.7988-7999.2004.  The P_RM sequence-tolerance alleles.

  FULL TEXT RETRIEVED -- and it replaces the lineage the design named.
    Zong C, So LH, Sepulveda LA, Skinner SO, Golding I (2010) Mol Syst Biol 6:440, PMID 21119634,
        PMC3010116, doi:10.1038/msb.2010.96.

A SECOND PROVENANCE HAZARD, FOUND WHILE READING. The retrieval strips superscripts and
subscripts from the article body. "less than once per 10 cell generations" in the retrieved text
is a number whose EXPONENT WAS SILENTLY DELETED. Every such number is tagged UNRETRIEVED below
rather than reconstructed, because reconstructing an exponent from context is exactly the
substitution of a plausible value the instruction forbids. This is the same failure mode as
ledger defect O -- a missing value that does not announce itself as missing -- arriving through a
tool instead of through a sentinel. Where the same quantity survived intact in the STRUCTURED
metadata (which preserved "10(-8)"), it is tagged MEASURED and the metadata is named as the
source.

=================================================================================================
WHAT THE RETRIEVED PAPER ACTUALLY CHANGES.
=================================================================================================

The design's trap was that in the Shea-Ackers/Arkin lineage the transcription rates R_RM and R_R
were ADJUSTED to reproduce ~200 CI molecules and a ~2e-9 lysis frequency, so wild-type agreement
on those two quantities is circular. Zong et al. build the model differently, and state their own
split in the text:

    FITTED.   k_max, the transcription-initiation rate when the promoter is properly bound, is
              tuned iteratively until the simulated mean gene-activation rate matches the
              smFISH-MEASURED burst frequency. Separately for cI and for cro.
    FITTED.   mu(T), the fraction of active repressor in the cI857 allele, obtained "using a
              comparison of the measured mRNA levels to the predictions of the stochastic
              simulation".
    MEASURED. The burst frequency, burst size, mRNA lifetimes, translation rate, proteins per
              burst, phages per lysis, growth rate.
    FREE.     The lysogen CI abundance. The paper states of its ~300 molecules/cell prediction:
              "Note that no parameters were adjusted to obtain this estimate."

So the circularity moved. In this lineage the fit is anchored on the BURST FREQUENCY, not on the
abundance and not on the switching rate -- which makes the abundance and the switching rate free
predictions, and makes the design's original trap absent here. That is the good news, and it is
why L0 is PARTIAL rather than FAIL.

The bad news is two-fold and it is what stops the build.

  (i) THE MODEL CANNOT BE EVALUATED. The predictor the paper validates is
          P_switch  =  exp( - k_on * mu(T) * tau_cell / ln 2 )
      -- the probability that the fate-determining gene fires NO activity burst for one CI protein
      lifetime, which is tau_cell/ln2 ~ 1.4 generations. k_on is MEASURED. mu(T) is FITTED and its
      table WAS NOT RETRIEVED. tau_cell is stated to have been "measured directly" and its VALUE
      WAS NOT RETRIEVED. Two of the three inputs are missing, so no number can be produced without
      substituting plausible values.

  (ii) AND THE TEST WAS ALREADY DEAD BEFORE THAT MATTERED. This is the part that does not depend
      on any missing number, and it is computed in l3_power() below from counts that WERE
      retrieved. Of the 18 P_RM alleles available as an out-of-sample set, the paper reports that
      12 sit at the wild-type value -- which is itself AT the sensitivity limit of the stability
      assay -- 4 are too unstable to lysogenize at all, and only 2 give an intermediate,
      discriminating rate. A test set with two discriminating points has a best-achievable
      p-value of 0.25 against the no-skill null, so it cannot reject at 0.05 even if the model is
      perfect. L3 as specified is defect N: a bar above its own achievable ceiling. It is not run.

AND THE WILD-TYPE CHECK IS WORSE THAN "NOT A TEST". The design already knew wild type was a
consistency check. L2 asks the sharper question and the answer is quantitative even without
mu(T), because for wild-type CI mu = 1 by definition and only tau_cell is unknown -- and tau_cell
is bounded by ordinary bacterial growth. Bracketing tau_cell over any defensible range and
comparing against the measured detection floor of 1e-8 per generation shows the predicted
wild-type switching rate sitting BELOW the floor by tens of orders of magnitude. The wild-type
agreement is therefore not merely uninformative; it is unfalsifiable by a margin so large that no
conceivable improvement to the assay would make it informative. wild_type_falsifiability() prints
that margin.

WHAT WOULD REVIVE THE LINE, stated concretely so it is actionable rather than a shrug:
  * the mu(T) table and the cell doubling time, which would make the model evaluable;
  * a test set of mutants whose measured switching rates are INTERIOR to the assay's dynamic
    range -- the cI857 temperature series is such a set, spanning ~8 orders of magnitude, but it
    is the series mu(T) was fitted on, so it is calibration, not test;
  * or the lambda-prm240 derivatives of Little & Michalowski 2010, which were explicitly built to
    have intermediate, growth-condition-dependent stabilities. That is the right test set. Its
    numbers are in a paper whose PMC record exists but whose text was not served here.

Nothing below substitutes a value for anything tagged UNRETRIEVED.
"""
from __future__ import annotations

import math

MEASURED = "MEASURED"
FITTED = "FITTED"
DERIVED = "DERIVED"
UNRETRIEVED = "UNRETRIEVED"

ZONG = ("Zong et al. 2010 Mol Syst Biol 6:440", "10.1038/msb.2010.96")
LITTLE10 = ("Little & Michalowski 2010 J Bacteriol 192:6064", "10.1128/JB.00726-10")
MICH04 = ("Michalowski et al. 2004 J Bacteriol 186:7988", "10.1128/JB.186.23.7988-7999.2004")
SHEA85 = ("Shea & Ackers 1985 J Mol Biol 181:211", "10.1016/0022-2836(85)90086-5")


class Param:
    """One number on TWO INDEPENDENT AXES, because collapsing them states the opposite of the truth.

    ORIGIN is what the number is: MEASURED, FITTED or DERIVED. RETRIEVED is whether this session
    could actually read its value. They are orthogonal, and the first version of this class had
    only one axis: it tagged k_max and mu(T) UNRETRIEVED, which made the summary line read
    "FITTED: 0" on a parameter set whose two most important numbers are BOTH fitted. A count that
    says nothing was fitted, on evidence that says two things were, is exactly the overstatement
    this gate exists to catch, so the axis was split rather than the count reworded.

    The value/retrieved invariant is enforced here rather than trusted, because the single most
    expensive error in this project's history was a missing value that carried a number anyway.
    """

    def __init__(self, name, value, unit, origin, source, note="", retrieved=None):
        if origin not in (MEASURED, FITTED, DERIVED):
            raise ValueError(f"{name}: origin must be MEASURED, FITTED or DERIVED")
        if retrieved is None:
            retrieved = value is not None
        if retrieved and value is None:
            raise ValueError(f"{name}: retrieved parameters must carry a value")
        if not retrieved and value is not None:
            raise ValueError(f"{name}: unretrieved parameters must carry value=None")
        self.name, self.value, self.unit = name, value, unit
        self.origin, self.retrieved = origin, bool(retrieved)
        self.source, self.note = source, note

    @property
    def status(self):
        """Display tag. UNRETRIEVED wins for display because it governs what may be computed."""
        return self.origin if self.retrieved else f"{self.origin[:4]}/UNRETR"


PARAMETERS = [
    Param("k_on(cI), burst frequency, MG1655(lambda) 37C", 1.4, "min^-1", MEASURED, ZONG,
          "1.4 +/- 0.2, six independent experiments, smFISH"),
    Param("burst size a(cI), 37C", 4.3, "transcripts/burst", MEASURED, ZONG, "4.3 +/- 0.4"),
    Param("burst size a(cI), 30-40C", 4.1, "transcripts/burst", MEASURED, ZONG, "4.1 +/- 0.5"),
    Param("burst size a(cro)", 1.7, "transcripts/burst", MEASURED, ZONG, "1.7 +/- 0.5"),
    Param("mRNA lifetime, cI", 4.0, "min", MEASURED, ZONG, "4.0 +/- 0.8, qRT-PCR + rifampicin"),
    Param("mRNA lifetime, cro", 2.8, "min", MEASURED, ZONG, "2.8 +/- 0.4"),
    Param("proteins per burst, b(CI)", 20.0, "molecules", DERIVED, ZONG, "b = a * 5, a ~ 4"),
    Param("phages released per lysis", 200.0, "phage/cell", MEASURED, ZONG, "30C and 40C"),
    Param("variance/mean of cI mRNA", 5.3, "-", MEASURED, ZONG, "5.3 +/- 0.4, non-Poissonian"),
    Param("mean CI in lysogen (model output)", 300.0, "molecules/cell", DERIVED, ZONG,
          "FREE prediction: 'no parameters were adjusted to obtain this estimate'"),
    Param("intrinsic switching rate, wild type, RecA-", 1e-8, "per generation", MEASURED,
          LITTLE10, "UPPER BOUND: 'probably less than 10(-8)/generation'; from the structured "
                    "metadata, where the exponent survived"),
    Param("k_max, transcription initiation when bound", None, "s^-1", FITTED, ZONG,
          "Tuned until <k_on>_sim matches the measured k_on. This is the anchor of the whole "
          "calibration, and its value is not given in the retrieved text.", retrieved=False),
    Param("mu(T), active repressor fraction, cI857", None, "-", FITTED, ZONG,
          "Fitted against measured mRNA levels. Table not in the retrieved text.",
          retrieved=False),
    Param("cell doubling time tau", None, "min", MEASURED, ZONG,
          "'Cell growth rate was measured directly'; value not in the retrieved text.",
          retrieved=False),
    Param("O_R free energies dG(CI), dG(Cro), cooperativity", None, "kcal/mol", MEASURED,
          SHEA85, "No PMC record; not retrievable in this environment.", retrieved=False),
    Param("translation rate, cI mRNA", None, "s^-1", MEASURED, ZONG,
          "Text reads 'or 0.02 s' -- the unit exponent was STRIPPED by retrieval. 0.02 s^-1 is "
          "the only dimensionally sensible reading, but reconstructing it is a substitution, so "
          "it is recorded as unretrieved and not used.", retrieved=False),
]

DETECTION_FLOOR = 1e-8       # per generation, Little & Michalowski 2010 (upper bound)
CI_LIFETIME_GENERATIONS = 1.0 / math.log(2.0)   # tau/ln2, ~1.44 generations


def provenance():
    """Counts on BOTH axes. L0 reads this.

    `origin` answers "how many of the model's numbers were fitted" -- which is the question the
    design's trap is about, and which stays answerable even when the fitted values themselves
    could not be read. `retrieved` answers "how many can this session actually use".
    """
    origin = {MEASURED: 0, FITTED: 0, DERIVED: 0}
    retrieved = {True: 0, False: 0}
    fitted_unretrieved = 0
    for p in PARAMETERS:
        origin[p.origin] += 1
        retrieved[p.retrieved] += 1
        if p.origin == FITTED and not p.retrieved:
            fitted_unretrieved += 1
    return {"origin": origin, "retrieved": retrieved[True], "unretrieved": retrieved[False],
            "fitted_but_unretrieved": fitted_unretrieved}


def burst_survival(k_on_per_min, tau_cell_min, mu=1.0):
    """P(switch per CI protein lifetime) = exp(-k_on * mu * tau/ln2).

    The retrieved relation from Zong et al. 2010: switching occurs when the fate-determining gene
    fires no activity burst for one CI protein lifetime, and burst arrivals are Poisson. Every
    argument must be supplied by the caller; this function deliberately holds no defaults for the
    quantities tagged UNRETRIEVED.
    """
    n_events = k_on_per_min * mu * tau_cell_min / math.log(2.0)
    return math.exp(-n_events), n_events


def wild_type_falsifiability(k_on_per_min=1.4, tau_range_min=(20.0, 60.0),
                             floor=DETECTION_FLOOR):
    """L2. How far below the assay floor does the wild-type prediction sit?

    mu = 1 for wild-type CI by definition, so the only unretrieved input is tau_cell, and tau_cell
    is bracketed rather than assumed: 20 min is about as fast as E. coli divides in rich medium
    and 60 min about as slow as the stability assays are run. Returns the orders of magnitude by
    which the prediction undershoots the floor at each end of the bracket. A positive margin means
    the prediction is BELOW the floor, i.e. it agrees with the measurement unconditionally.
    """
    lo_tau, hi_tau = tau_range_min
    p_slow, n_slow = burst_survival(k_on_per_min, lo_tau)     # fewer bursts -> larger P
    p_fast, n_fast = burst_survival(k_on_per_min, hi_tau)
    margin_hi = math.log10(floor) - math.log10(p_slow)        # smallest undershoot
    margin_lo = math.log10(floor) - math.log10(p_fast)        # largest undershoot
    return {"tau_min": lo_tau, "tau_max": hi_tau,
            "P_at_tau_min": p_slow, "P_at_tau_max": p_fast,
            "events_at_tau_min": n_slow, "events_at_tau_max": n_fast,
            "orders_below_floor_min": margin_hi, "orders_below_floor_max": margin_lo,
            "falsifiable": margin_hi < 0.0}


def l3_power(n_at_floor, n_too_unstable, n_discriminating, alpha=0.05):
    """L3a. The best p-value the out-of-sample test could produce IF THE MODEL WERE PERFECT.

    A mutant censored at the detection floor agrees with any prediction below the floor, and a
    mutant that cannot lysogenize agrees with any prediction above the lysogenization threshold.
    Neither carries information about WHERE inside the range the model puts it. Only the interior
    points discriminate.

    Under the no-skill null each discriminating mutant lands on the correct side of the
    floor/unstable dichotomy with probability 1/2, so the smallest attainable one-sided p-value on
    n discriminating points is 2^-n. If that exceeds alpha the test cannot reject at alpha no
    matter what the model does, and running it would be ledger defect N.
    """
    n = int(n_discriminating)
    best_p = 0.5 ** n if n > 0 else 1.0
    total = int(n_at_floor) + int(n_too_unstable) + n
    return {"n_total": total, "n_at_floor": int(n_at_floor),
            "n_too_unstable": int(n_too_unstable), "n_discriminating": n,
            "best_attainable_p": best_p, "alpha": alpha,
            "powered": best_p <= alpha,
            "n_needed_for_alpha": math.ceil(-math.log2(alpha)) if alpha > 0 else None}


def l0_verdict():
    """PASS only if nothing the model needs is UNRETRIEVED and the fitted set is identifiable."""
    counts = provenance()
    missing = [p for p in PARAMETERS if not p.retrieved]
    # The fitted/measured SPLIT is identifiable -- the paper states which numbers were tuned --
    # even though the tuned VALUES were not retrieved. Those are different failures and the gate
    # must not conflate them.
    split_known = True
    evaluable = not any(p.name.startswith(("mu(T)", "cell doubling time")) for p in missing)
    if split_known and evaluable:
        return "PASS", counts, missing
    if split_known:
        return "PARTIAL", counts, missing
    return "FAIL", counts, missing


def verify():
    print("=" * 95)
    print("L0  PARAMETER PROVENANCE")
    print("=" * 95)
    for p in PARAMETERS:
        val = "--" if p.value is None else f"{p.value:g}"
        print(f"  {p.status:<12s} {p.name:<52s} {val:>10s} {p.unit}")
        if p.note:
            print(f"               {p.note}")
    verdict, counts, missing = l0_verdict()
    o = counts["origin"]
    print(f"\n  by ORIGIN     measured {o[MEASURED]}   fitted {o[FITTED]}   derived {o[DERIVED]}")
    print(f"  by RETRIEVAL  retrieved {counts['retrieved']}   unretrieved {counts['unretrieved']}"
          f"   (of which fitted: {counts['fitted_but_unretrieved']})")
    print(f"  L0 VERDICT: {verdict}   ({len(missing)} unretrieved)")

    print("\n" + "=" * 95)
    print("L2  IS THE WILD-TYPE CHECK FALSIFIABLE AT ALL?")
    print("=" * 95)
    f = wild_type_falsifiability()
    print(f"  k_on = 1.4/min (MEASURED), mu = 1 (wild type, by definition),"
          f" tau bracketed {f['tau_min']:.0f}-{f['tau_max']:.0f} min")
    print(f"  burst events per CI lifetime: {f['events_at_tau_min']:.1f} to "
          f"{f['events_at_tau_max']:.1f}")
    print(f"  predicted P(switch/generation): {f['P_at_tau_max']:.2e} to {f['P_at_tau_min']:.2e}")
    print(f"  measured detection floor:       {DETECTION_FLOOR:.0e} (upper bound)")
    print(f"  prediction sits {f['orders_below_floor_min']:.0f} to "
          f"{f['orders_below_floor_max']:.0f} orders of magnitude BELOW the floor")
    tag = "FALSIFIABLE" if f["falsifiable"] else "UNFALSIFIABLE"
    print(f"  L2 VERDICT: {tag} -- wild-type agreement is a fact about the assay's dynamic "
          f"range,\n              not about the model, over the whole defensible tau bracket.")

    print("\n" + "=" * 95)
    print("L3a  POWER OF THE OUT-OF-SAMPLE TEST, COMPUTED BEFORE L3 RUNS")
    print("=" * 95)
    pw = l3_power(n_at_floor=12, n_too_unstable=4, n_discriminating=2)
    print(f"  P_RM allele set (Michalowski et al. 2004, counts as reported by Zong et al. 2010):")
    print(f"    {pw['n_total']:2d} alleles examined")
    print(f"    {pw['n_at_floor']:2d} at the wild-type value, which is AT the assay's "
          f"sensitivity limit  -> censored")
    print(f"    {pw['n_too_unstable']:2d} too unstable to lysogenize at all"
          f"                            -> censored")
    print(f"    {pw['n_discriminating']:2d} with an intermediate, discriminating rate")
    print(f"  best attainable p-value if the model were PERFECT: {pw['best_attainable_p']:.3f}"
          f"   (alpha = {pw['alpha']})")
    print(f"  discriminating mutants needed to reach alpha: {pw['n_needed_for_alpha']}")
    tag = "POWERED" if pw["powered"] else "UNDERPOWERED -- LEDGER DEFECT N"
    print(f"  L3a VERDICT: {tag}")

    print("\n" + "=" * 95)
    print("DISPOSITION")
    print("=" * 95)
    print("  L1  NOT RUN -- two of the three model inputs are UNRETRIEVED (mu(T), tau_cell).")
    print("  L3  NOT RUN -- L3a shows the test cannot reject at 0.05 even on a perfect model.")
    print("  L4  NOT RUN -- depends on L3.")
    print("  L5  This module IS L5: the line is recorded as untested, not as supported.")
    print("\n  The design's own trap is ABSENT in this parameter lineage: Zong et al. anchor the")
    print("  fit on the MEASURED burst frequency, leaving CI abundance and switching rate free.")
    print("  The line dies on evidence availability and test power instead, which is a different")
    print("  and more recoverable failure. The revival path is in the module docstring.")
    return {"L0": verdict, "L2": f, "L3a": pw}


if __name__ == "__main__":
    verify()
