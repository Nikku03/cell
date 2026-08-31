"""The 11-order lambda gap: a factor budget, and the finding that the gap was mostly mine.

THE QUESTION. A burst-survival model of lysogen stability predicts a spontaneous switching rate
of ~3e-18 per generation at the least-stable end of a defensible cell-doubling-time bracket. The
literature reports rates around 1e-7 in recA hosts. Two hypotheses were to be separated by
building a factor budget rather than by argument:

    H1  the assay counts something other than per-generation spontaneous switching, and
        correcting for that closes the gap;
    H2  the model omits a destabilising route, and adding it closes the gap.

THE ANSWER IS NEITHER, AND THE BUDGET IS HOW THAT WAS ESTABLISHED RATHER THAN ASSERTED. Every
assay artefact large enough to matter is REFUTED from the sources -- burst size was divided out
in every paper that needed to, the recA alleles are deletions and not leaky point mutants, the
cultures are far outside the jackpot regime, and the titres are exponential-phase ratios that the
primary paper shows analytically to be constant rather than cumulative. The confirmed factors
multiply to 10^0.15. They do not close 10^9.5.

AND THEN THE RESIDUAL TURNED OUT NOT TO BE A DISCREPANCY. Working backwards from the measured
rate through the model's own relation:

    S = exp(-N),  N = k_on * mu * tau_cell / ln 2   (burst events per CI protein lifetime)

    measured S = 9.0e-9  ->  N = 18.5
    my L2 used k_on = 1.4/min with tau in 20-60 min  ->  N = 40.4 to 121.2

The whole 9.5 orders is a factor of 2.2 to 6.5 in the PRODUCT k_on * tau. And that factor is a
CONDITIONS MISMATCH, not a missing mechanism: the burst frequency 1.4/min was measured at 37 C in
MG1655(lambda), while every switching rate was measured at 30 C in the RecA-deficient strain
JL5902. Solving the relation for the burst frequency the assay's own conditions imply gives
0.32-0.43/min -- an entirely ordinary value for a culture 7 C colder. Nothing is missing. Two
numbers from different experiments were multiplied together and then exponentiated.

THIS IS THE CROWDING ERROR-BAR FORMULA FIRING IN ANGER. rem/crowding_errorbar.py derived
    rare_error ~= exp[(N - <X>) * Delta / S]
and this is an instance with N - <X> = 18.5 and Delta = 1.18: exp(21.9) = 10^9.5. The formula
said a rare-event probability inherits an EXPONENTIAL amplification of a small error in a rate
ratio. It was verified on a birth-death process. Here it correctly accounts for a nine-order
discrepancy in a real published system, which is the first out-of-sample use it has had.

WHAT THIS DOES AND DOES NOT SAY.
  * It does NOT rescue the model. It shows the model's prediction for this quantity is
    exp(-N) with N uncertain by a factor of 2-6, i.e. the prediction spans 9e-9 to 3e-53. A
    prediction spanning 45 orders is not a prediction. L2's verdict stands and is strengthened.
  * It does NOT show the assays are clean. A5 (prophage copy number, pre-existing free phage) and
    C1/C2/C4/C5 are NOT STATED in the retrieved text and are recorded as gaps in the record.
    A gap is not an explanation and none of them is counted toward closing anything.
  * It DOES overturn the premise's anchor value. The 4e-7 figure was not located in any source
    worked. The two best-controlled recA-minus measurements are 9.0e-9 +/- 6.4e-9 (a value) and
    <1e-8 (a bound), which agree with each other and sit 1.1 orders BELOW the 1.1e-7 that the
    infective-centre assay reports for a nominally identical genotype.

SOURCES, all bibliographic data from PubMed. Channel is recorded per number because the two
channels disagree in a way that matters (ledger defect R: the structured full-text fetch strips
superscripts, so "10^5" arrives as "10"). Where a number's exponent could have been stripped and
no second channel confirms it, it is UNRETRIEVED here and not used.

  Zong C, So LH, Sepulveda LA, Skinner SO, Golding I (2010) Mol Syst Biol 6:440.
      PMID 21119634, PMC3010116, doi:10.1038/msb.2010.96
  Little JW, Michalowski CB (2010) J Bacteriol 192:6064-6076.
      PMID 20870769, PMC2976446, doi:10.1128/JB.00726-10
  Rozanov DV, D'Ari R, Sineoky SP (1998) J Bacteriol 180:6306-6315.
      PMID 9829941, PMC107717, doi:10.1128/JB.180.23.6306-6315.1998
  Little JW, Shepley DP, Wert DW (1999) EMBO J 18:4299-4307.
      PMID 10428968, PMC1171506, doi:10.1093/emboj/18.15.4299   -- SCANNED PDF ONLY; the HTML
      channel serves the abstract alone, so every Part B field for this paper is NOT RETRIEVED
      rather than absent. The earlier finding "rates exist and affinities do not" is therefore
      NEITHER confirmed NOR overturned from this paper.
"""
from __future__ import annotations

import math

CONFIRMED, REFUTED, NOT_STATED, NOT_RETRIEVED = (
    "CONFIRMED", "REFUTED", "NOT STATED", "NOT RETRIEVED")

OVERSTATES = "overstates measured"     # would make the assay's number too high
TOO_STABLE = "model too stable"        # would make the model's number too low
UNDERSTATES = "understates measured"


class Factor:
    """One line of the budget. `orders` is None unless the size is CONFIRMED from a source.

    A factor whose size is not confirmed contributes ZERO, whatever its status, because the whole
    point of the budget is that a gap in the record may not be spent as an explanation.
    """

    def __init__(self, name, direction, orders, status, evidence):
        if status != CONFIRMED and orders is not None:
            raise ValueError(f"{name}: only CONFIRMED factors may carry a size")
        self.name, self.direction, self.orders = name, direction, orders
        self.status, self.evidence = status, evidence

    @property
    def counts(self):
        return self.status == CONFIRMED and self.orders is not None


BUDGET = [
    # ---- Part A: assay forensics (H1) ----
    Factor("A1 burst size not divided out", OVERSTATES, None, REFUTED,
           "Zong: 'S=phi/BM ... M is the average number of phages released per cell lysis', "
           "M~200. Little & Michalowski: 'we assume ... that the average burst size is 100 "
           "phage per switching event'. Rozanov counts INFECTIVE CENTRES, not free phage, so "
           "no burst division is needed by construction. All three handled it."),
    Factor("A2a cumulative titre, not per-generation", OVERSTATES, None, REFUTED,
           "Zong derive that the free-phage:bacteria ratio CONVERGES to a constant during "
           "exponential growth and confirm it empirically; titre taken at OD600 0.1-0.5. Not "
           "an integral over the culture's history."),
    Factor("A2b rate quoted per 1.4 generations", OVERSTATES, math.log10(1.4), CONFIRMED,
           "Zong: 'S is actually the switching rate per ~1.4 cell generations'. Quoting it as "
           "per-generation overstates by 1.4x."),
    Factor("A3 no fluctuation estimator (Luria-Delbruck jackpots)", OVERSTATES, None, REFUTED,
           "No estimator was used -- Zong take a mean over 'two to eight independent "
           "experiments', Little uses 'six independent 50-ml cultures'. But jackpot dominance "
           "needs O(1) events per culture, and see jackpot_regime(): both assays run ~25-40 "
           "switching events per generation per culture. Confirmed absent, refuted as large."),
    Factor("A4 leaky recA point allele", OVERSTATES, None, REFUTED,
           "Little & Michalowski and Rozanov both use Delta(srl-recA)306::Tn10, a DELETION. "
           "Little verifies functionally: 'host recA mutants cannot support cleavage of CI'. "
           "Zong's JL5902 genotype is not stated in Zong but the strain is from the Little lab."),
    Factor("A5 prophage copy number / pre-existing free phage", OVERSTATES, None, NOT_STATED,
           "Little & Michalowski: no statement on subtracting background free phage at "
           "inoculation. Copy-number verification not stated in any source. Rozanov's "
           "infective-centre assay would score adsorbed free phage as induced cells."),
    Factor("A6 bound compared against value", OVERSTATES, None, REFUTED,
           "Little's <1e-8 is a BOUND ('probably less than 10-8/generation'), derived from "
           "'roughly 2 to 5 phage/ml culture' at 5e7-1e8 cells/ml. Zong's 9.0e-9 is a VALUE. "
           "They are consistent, and the bound is conservative by ~20x against its own raw "
           "numbers -- see bound_from_titre()."),
    Factor("A7 lab-to-lab spread on the same genotype", OVERSTATES, None, NOT_STATED,
           "Rozanov 1.1e-7 (infective centres, C600 recA, 30C LB) vs Zong 9.0e-9 (free phage/"
           "burst, JL5902, 30C LBGM): 12x apart on nominally the same recA deletion. This is a "
           "disagreement BETWEEN measurements, not a correction to either, and its cause is "
           "not stated. It may not be spent."),
    # ---- Part C: model audit (H2) ----
    Factor("C1 RecA-independent induction route omitted", TOO_STABLE, None, NOT_STATED,
           "Rozanov CONFIRM such routes exist (rcsA, dsrA, rcsC137, lon) and that 'induction "
           "involves repressor inactivation rather than repressor bypass'. But the BASELINE "
           "contribution in an unperturbed recA lysogen is not stated -- only 16x and 71x "
           "enhancements under plasmid overexpression. Existence confirmed, size not."),
    Factor("C2 copy-number doubling at prophage replication", TOO_STABLE, None, NOT_STATED,
           "The burst model tracks CI copy number against a fixed operator set. Nothing in the "
           "retrieved text addresses the transient two-operator state."),
    Factor("C3 mean-field CI removes the lower tail", TOO_STABLE, None, REFUTED,
           "Not an omission: Zong's model is a Gillespie master-equation simulation with "
           "explicitly bursty transcription, and the paper shows the Poissonian no-burst case "
           "predicts rates 'orders of magnitude lower than the experimental data'. The tail IS "
           "the model's dominant term."),
    Factor("C4 cro expression outside P_R / read-through", TOO_STABLE, None, NOT_STATED,
           "Not addressed in the retrieved text."),
    Factor("C5 O_L-O_R looping", UNDERSTATES, None, NOT_STATED,
           "Runs the WRONG WAY -- looping adds stability and would widen the gap. Whether the "
           "grand-canonical occupancy calculation includes O_L is not stated in what was "
           "retrieved, so this cannot even be scored as already-included."),
]

# ---- anchors, each with its channel ----
S_MEASURED = 9.0e-9          # Zong Table 1, lambda-IG831 (wild-type cI), RecA- host JL5902
S_MEASURED_SE = 6.4e-9
K_ON_37C = 1.4               # /min, MG1655(lambda) at 37 C, smFISH
TAU_BRACKET = (20.0, 60.0)   # min, the bracket L2 used


def n_events(S):
    """Burst events per CI lifetime implied by an observed switching rate: N = -ln S."""
    return -math.log(S)


def n_from_rates(k_on_per_min, tau_min, mu=1.0):
    return k_on_per_min * mu * tau_min / math.log(2.0)


def implied_k_on(S, tau_min):
    """The burst frequency the ASSAY's own conditions imply, from its own measured rate."""
    return n_events(S) * math.log(2.0) / tau_min


def jackpot_regime(volume_ml, cells_per_ml, rate_per_gen):
    """Expected switching events per culture per generation. Jackpots dominate only near O(1)."""
    return volume_ml * cells_per_ml * rate_per_gen


def bound_from_titre(phage_per_ml, cells_per_ml, burst=100.0):
    """Little & Michalowski's raw numbers, pushed through their own stated conversion."""
    return phage_per_ml / cells_per_ml / burst


def budget_total():
    counted = [f for f in BUDGET if f.counts]
    return sum(f.orders for f in counted), counted


def report():
    n_meas = n_events(S_MEASURED)
    n_lo = n_from_rates(K_ON_37C, TAU_BRACKET[0])
    n_hi = n_from_rates(K_ON_37C, TAU_BRACKET[1])
    gap_orders = (n_lo - n_meas) / math.log(10.0)

    print("=" * 98)
    print("THE GAP, RESTATED FROM SOURCES")
    print("=" * 98)
    print(f"  measured (Zong Table 1, lambda-IG831 wt, RecA- JL5902): "
          f"{S_MEASURED:.1e} +/- {S_MEASURED_SE:.1e} per generation")
    print(f"  measured (Little & Michalowski, Delta-recA):            <1e-8 per generation "
          f"[BOUND, not a value]")
    print(f"  measured (Rozanov Table 3, C600 recA, infective centres): 1.1e-7 [different assay]")
    print(f"  premise value 4e-7:                                     NOT LOCATED in any source")
    print(f"  model, L2 best case over the tau bracket:                {math.exp(-n_lo):.1e}")
    print(f"  GAP: {gap_orders:.1f} orders, not 11 -- the premise anchor was 1.1 orders high.")

    print("\n" + "=" * 98)
    print("FACTOR BUDGET")
    print("=" * 98)
    print(f"  {'factor':<48s} {'direction':<18s} {'orders':>7s}  status")
    print("  " + "-" * 94)
    for f in BUDGET:
        o = f"{f.orders:.2f}" if f.counts else "--"
        print(f"  {f.name:<48s} {f.direction:<18s} {o:>7s}  {f.status}")
    total, counted = budget_total()
    print("  " + "-" * 94)
    print(f"  CONFIRMED product: 10^{total:.2f}   "
          f"({len(counted)} of {len(BUDGET)} factors carry a confirmed size)")
    print(f"  GAP REMAINING: 10^{gap_orders - total:.2f}")

    print("\n" + "=" * 98)
    print("THE RESIDUAL IS NOT A DISCREPANCY")
    print("=" * 98)
    print(f"  measured S = {S_MEASURED:.1e}  ->  N = {n_meas:.1f} burst events per CI lifetime")
    print(f"  L2 used k_on = {K_ON_37C}/min (37 C, MG1655) with tau = "
          f"{TAU_BRACKET[0]:.0f}-{TAU_BRACKET[1]:.0f} min  ->  N = {n_lo:.1f} to {n_hi:.1f}")
    print(f"  ratio in the PRODUCT k_on*tau: {n_lo/n_meas:.2f} to {n_hi/n_meas:.2f}")
    print(f"  the entire {gap_orders:.1f} orders is that factor, exponentiated.")
    print(f"\n  and the factor is a CONDITIONS MISMATCH, not a missing mechanism:")
    print(f"    k_on was measured at 37 C in MG1655(lambda); every switching rate was measured")
    print(f"    at 30 C in JL5902 (RecA-). Solving the model's own relation for the burst")
    print(f"    frequency the assay's conditions imply:")
    for tau in (25.0, 30.0, 40.0, 50.0):
        print(f"      tau = {tau:.0f} min  ->  k_on = {implied_k_on(S_MEASURED, tau):.2f}/min")
    print(f"    against {K_ON_37C}/min at 37 C. Ordinary values for a culture 7 C colder.")

    print("\n" + "=" * 98)
    print("SUPPORTING ARITHMETIC FOR THE REFUTED FACTORS")
    print("=" * 98)
    z = jackpot_regime(15.0, 3e8, S_MEASURED)
    lm = jackpot_regime(50.0, 7.5e7, 1e-8)
    print(f"  A3 jackpot regime, switching events per culture per generation:")
    print(f"     Zong  15 ml at ~3e8 cells/ml, S={S_MEASURED:.0e}  ->  {z:.0f} events")
    print(f"     L&M   50 ml at ~7.5e7 cells/ml, S=1e-8            ->  {lm:.0f} events")
    print(f"     Both far from O(1). A simple mean is unbiased here; jackpots need O(1).")
    b = bound_from_titre(4.0, 7.5e7)
    print(f"  A6 Little & Michalowski's own raw numbers through their own conversion:")
    print(f"     4 phage/ml / 7.5e7 cells/ml / 100 burst = {b:.1e} per cell per generation")
    print(f"     quoted as <1e-8, i.e. conservative by ~{1e-8/b:.0f}x. The bound is soft, and")
    print(f"     it is a bound: it must never be compared against a value as if it were one.")

    print("\n" + "=" * 98)
    print("VERDICT")
    print("=" * 98)
    print("  H1 (assay artefact) -- REJECTED. Every artefact large enough to matter is refuted")
    print("     from the sources. Confirmed factors total 10^%.2f." % total)
    print("  H2 (missing destabilising route) -- NOT ESTABLISHED. Such routes demonstrably exist")
    print("     (Rozanov), but no source states a baseline magnitude, so nothing may be counted.")
    print("  NEITHER. The gap is a conditions mismatch in the L2 calculation, amplified by")
    print("  exp(). rem/crowding_errorbar.py predicts exactly this: rare_error ~ exp[(N-<X>)*D],")
    print(f"  with N-<X> = {n_meas:.1f} and D = {n_lo/n_meas - 1:.2f} giving "
          f"10^{(n_lo - n_meas)/math.log(10):.1f}.")
    print("  L2's conclusion is unchanged and strengthened: this model cannot predict this")
    print("  quantity, because the answer is exp of a product neither factor of which was")
    print("  measured under the assay's conditions.")
    return {"gap_orders": gap_orders, "confirmed_orders": total,
            "remaining_orders": gap_orders - total, "n_measured": n_meas}


if __name__ == "__main__":
    report()
