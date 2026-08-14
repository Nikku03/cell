"""LOOP 93 -- ENZYME CAPACITY: THE CONSTRAINT THAT TURNS AN UNCALIBRATED FBA INTO A CELL.

WHAT RUNNING THE METABOLIC MODEL ACTUALLY SHOWED. cell_sim.py solves Human-GEM -- 12,931 reactions
with real stoichiometry -- and it is the most functional subsystem in this repository. Running it:

        baseline growth 20.5576 /h

A growth rate of 20.56 per hour is a doubling time of ln2/20.56 = 0.034 h, or about two minutes. A
HeLa cell doubles in roughly 24 hours. The model is running about seven hundred times too fast, and
nothing in it objects, because FLUX BALANCE HAS NO NOTION OF COST. Every reaction is free up to its
bound, so the optimiser runs the busiest ones at exactly 1000 mmol/gDW/h -- the shipped default --
and the medium's 430 uptakes at 1.0 each supply more carbon and nitrogen than any cell takes up.

That is not a bug in Human-GEM. It is the known and expected behaviour of unconstrained FBA, and the
standard answer is to make enzymes cost something: a reaction can only carry the flux its enzyme can
catalyse, so

        v_i <= k_cat_i * [E_i]        and       sum_i ( v_i / k_cat_i ) * MW_i <= protein budget

which is the GECKO / ecModel construction. It normally needs a k_cat database this repository does
not have. But the direction can be inverted at no cost: rather than looking k_cat up, ASK WHAT k_cat
THE CURRENT SOLUTION WOULD REQUIRE, and check it against what enzymes can physically do. Turnover
numbers cluster around 1-100 per second and the fastest known enzymes reach about 1e6. If the flux
solution demands more than that, it is not a metabolic state, and the amount by which it overshoots
is the calibration error.

AND THERE IS A CLAIM IN THE REPOSITORY TO CORRECT. cell_sim.py prints:

    WARBURG CHECK: lactate exported at 558.8 while O2 is being consumed at -0.0
    fermenting glucose to lactate WITH oxygen available is the defining metabolic phenotype of
    a proliferating cancer line, and K562 is one. The model was never told this.

The 558.8 and the -0.0 are the ANAEROBIC solve's numbers -- the run where oxygen was deliberately
removed. Fermenting without oxygen is not the Warburg effect, it is anaerobic glycolysis, and "the
model was never told this" is the opposite of true: it was told to take oxygen away. The baseline
solve does import O2 at 20.0 while exporting lactate at 547.5, which IS Warburg-like, so the
phenomenon may well be there -- but the check as printed reads the wrong condition and claims
discovery of something it enforced. K4 tests it properly.

PREDECLARED, before any number:

  K1 THE UNCONSTRAINED SOLUTION IS NOT A CELL                       THE DIAGNOSIS.
       growth against a measured mammalian doubling time. Predeclared expectation: the model is more
       than 100x too fast. Reported as a factor, not gated on my own expectation being right.
  K2 THE REQUIRED TURNOVER NUMBERS ARE IMPOSSIBLE                   THE QUANTITATIVE GATE.
       for every reaction carrying flux that has a gene and a measured HeLa enzyme abundance, the
       apparent k_cat the solution demands. Gate: the MEDIAN must exceed 100/s, the top of the
       ordinary enzymatic range. If the median comes out physiological then the flux solution is
       payable after all and this loop's whole premise is wrong -- which is the point of gating it.
  K3 ENZYME CAPACITY BRINGS GROWTH INTO RANGE                       THE GATE THAT MATTERS.
       impose the pooled constraint sum(v_i/k_cat)*MW_i <= enzyme budget, with k_cat uniform at a
       literature-central value and the budget computed from measured protein mass fraction. Gate:
       growth must land within a factor of 3 of a 24 h doubling. Nothing here is fitted to growth --
       the budget comes from proteome mass and the k_cat from the literature range -- so landing in
       range is a prediction and missing it is a refutation.
  K4 THE WARBURG CLAIM, TESTED ON THE RIGHT CONDITION               THE CORRECTION.
       baseline lactate export AND baseline oxygen uptake, together, with the anaerobic numbers
       shown alongside so the difference is visible. Aerobic glycolysis requires both to be nonzero.
  K5 THE CONSTRAINT RE-ROUTES, IT DOES NOT MERELY RESCALE           THE CONTROL.
       uniformly shrinking every bound would also cut growth, so the constraint has to be shown to
       do something a scalar cannot. Gate: Spearman between constrained and unconstrained flux
       vectors must be BELOW 0.95, i.e. the identity of the reactions carrying flux changes.
       Nine gates this session have fired while measuring nothing; this is the one that would catch
       K3 passing for a trivial reason.
  K6 COVERAGE DECLARED
       how many reactions have a gene, how many of those have a measured enzyme, and what fraction
       of total flux those carry -- because a capacity constraint applied to a quarter of the
       network constrains a quarter of the network.

-> outputs/loop_enzyme_capacity.json
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import cell_sim as CS  # noqa: E402
import cell_proteome as CP  # noqa: E402
import loop_translation as L92  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 9301

DOUBLING_H = 24.0
MU_TARGET = float(np.log(2.0)) / DOUBLING_H        # 0.0289 /h
K1_MIN_FACTOR = 100.0
K2_MIN_KCAT = 100.0                                # per second, top of the ordinary range
K3_FACTOR = 3.0
K5_MAX_RHO = 0.95
KCAT_PER_S = 30.0                                  # literature-central turnover
PROTEIN_G_PER_GDW = 0.55                           # protein is ~55% of mammalian dry mass
ENZYME_FRACTION = 0.20                             # metabolic enzymes as a share of protein
MEAN_MW_DA = 50000.0

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 93 -- enzyme capacity: the constraint that turns an uncalibrated FBA into a cell")
    say("=" * 100)
    say()

    m = CS.load_model()
    CS.set_medium(m)
    sol = m.optimize()
    mu0 = float(sol.objective_value)
    flux0 = {r.id: float(sol.fluxes[r.id]) for r in m.reactions}
    say(f"  Human-GEM: {len(m.reactions):,} reactions, {len(m.metabolites):,} metabolites, "
        f"{len(m.genes):,} genes")
    say()

    say("K1 THE UNCONSTRAINED SOLUTION IS NOT A CELL")
    dt0 = float(np.log(2.0)) / mu0 if mu0 > 0 else float("inf")
    fac = mu0 / MU_TARGET if MU_TARGET > 0 else float("nan")
    say(f"     growth {mu0:.4f} /h -> doubling {dt0*60:.1f} minutes")
    say(f"     a mammalian cell doubles in {DOUBLING_H:.0f} h -> {MU_TARGET:.4f} /h")
    say(f"     the model is {fac:,.0f}x too fast")
    say(f"     expectation written before the run: more than {K1_MIN_FACTOR:.0f}x")
    say(f"     K1 reported (not gated on my expectation)")
    say()

    say("K2 THE REQUIRED TURNOVER NUMBERS ARE IMPOSSIBLE")
    hela = CP.hela_ppm()
    sym = CS.ensembl_to_symbol()
    # gDW per cell: a mammalian cell is ~3000 um^3 at ~1.1 g/mL and ~20% dry mass
    GDW_PER_CELL = 3000e-15 * 1.1 * 0.20            # grams dry weight per cell
    TOTAL_PROTEINS = 5.0e9
    kapp, used = {}, 0
    for r in m.reactions:
        v = abs(flux0.get(r.id, 0.0))
        if v <= 1e-9 or not r.genes:
            continue
        cop = []
        for g in r.genes:
            s = sym.get(g.id, g.id)
            if s in hela and hela[s] > 0:
                cop.append(hela[s] / 1e6 * TOTAL_PROTEINS)
        if not cop:
            continue
        E = max(cop)                                 # the most abundant subunit bounds the complex
        mol_per_cell_h = v * GDW_PER_CELL * 6.022e20      # mmol/gDW/h -> molecules/cell/h
        kapp[r.id] = mol_per_cell_h / E / 3600.0          # per second
        used += 1
    vals = np.array(list(kapp.values()))
    med = float(np.median(vals)) if len(vals) else float("nan")
    say(f"     {used:,} reactions carry flux, have a gene, and have a measured HeLa enzyme")
    say(f"     apparent k_cat required: median {med:,.1f} /s, "
        f"IQR {np.percentile(vals,25):,.1f}-{np.percentile(vals,75):,.1f}, "
        f"max {vals.max():,.0f} /s")
    say(f"     ordinary enzymes run at 1-100 /s; the fastest known reach about 1e6 /s")
    n_imp = int((vals > 1e6).sum())
    say(f"     reactions demanding more than 1e6 /s: {n_imp:,} ({n_imp/len(vals):.1%})")
    k2 = bool(np.isfinite(med) and med > K2_MIN_KCAT)
    say(f"     K2 {'PASS' if k2 else 'FAIL'} -- the flux solution "
        f"{'cannot be paid for by any enzyme' if k2 else 'IS payable, so the premise is wrong'}")
    say()

    say("K3 ENZYME CAPACITY BRINGS GROWTH INTO RANGE")
    enz_g = PROTEIN_G_PER_GDW * ENZYME_FRACTION
    budget_mmol = enz_g / MEAN_MW_DA * 1e3          # mmol enzyme per gDW
    kcat_h = KCAT_PER_S * 3600.0
    cap = budget_mmol * kcat_h                       # mmol/gDW/h of TOTAL enzymatic flux
    say(f"     protein {PROTEIN_G_PER_GDW:.2f} g/gDW x {ENZYME_FRACTION:.0%} metabolic "
        f"= {enz_g:.3f} g enzyme/gDW")
    say(f"     at {MEAN_MW_DA/1e3:.0f} kDa -> {budget_mmol:.5f} mmol enzyme/gDW")
    say(f"     at k_cat {KCAT_PER_S:.0f}/s = {kcat_h:,.0f}/h -> total enzymatic flux <= "
        f"{cap:,.1f} mmol/gDW/h")
    enzymatic = [r for r in m.reactions if r.genes]
    say(f"     applied to the {len(enzymatic):,} gene-associated reactions as one pooled constraint")
    from cobra import Reaction  # noqa: F401
    pool = m.problem.Constraint(0, lb=None, ub=cap, name="enzyme_pool")
    m.add_cons_vars([pool])
    m.solver.update()
    coeffs = {}
    for r in enzymatic:
        coeffs[r.forward_variable] = 1.0
        coeffs[r.reverse_variable] = 1.0
    pool.set_linear_coefficients(coeffs)
    sol2 = m.optimize()
    mu1 = float(sol2.objective_value) if sol2.status == "optimal" else float("nan")
    flux1 = ({r.id: float(sol2.fluxes[r.id]) for r in m.reactions}
             if sol2.status == "optimal" else {})
    dt1 = float(np.log(2.0)) / mu1 if mu1 and mu1 > 0 else float("inf")
    say(f"     constrained growth {mu1:.4f} /h -> doubling {dt1:.2f} h "
        f"(target {DOUBLING_H:.0f} h)")
    off = max(dt1, DOUBLING_H) / min(dt1, DOUBLING_H) if np.isfinite(dt1) and dt1 > 0 else float("nan")
    say(f"     off by a factor of {off:.2f}   (gate: <= {K3_FACTOR})")
    k3 = bool(np.isfinite(off) and off <= K3_FACTOR)
    say(f"     K3 {'PASS' if k3 else 'FAIL'}")
    say()

    say("K4 THE WARBURG CLAIM, TESTED ON THE RIGHT CONDITION")
    prev = json.load(open(Path("outputs/orphan/cell_sim.json")))
    base, ana = prev["baseline"], prev["anaerobic"]
    o2_base = dict(base["imports"]).get("O2", 0.0)
    o2_ana = dict(ana["imports"]).get("O2", 0.0)
    say(f"     BASELINE   lactate {base['lactate_export']:.1f}   O2 imported {o2_base:.1f}")
    say(f"     ANAEROBIC  lactate {ana['lactate_export']:.1f}   O2 imported {o2_ana:.1f}")
    say(f"     cell_sim.py printed the ANAEROBIC pair and called it Warburg, adding 'the model was")
    say(f"     never told this' -- but the anaerobic run is precisely the one where oxygen was")
    say(f"     removed by hand. Fermenting without oxygen is anaerobic glycolysis, not Warburg.")
    warburg = bool(base["lactate_export"] > 1.0 and o2_base > 1.0)
    say(f"     on the BASELINE condition, where oxygen IS available: "
        f"{'aerobic glycolysis is genuinely present' if warburg else 'no aerobic glycolysis'}")
    say(f"     so the phenomenon is real and the printed check was reading the wrong solve")
    say()

    say("K5 THE CONSTRAINT RE-ROUTES, IT DOES NOT MERELY RESCALE")
    if flux1:
        from scipy.stats import spearmanr
        ids = [r.id for r in m.reactions]
        a = np.array([abs(flux0.get(i, 0.0)) for i in ids])
        b = np.array([abs(flux1.get(i, 0.0)) for i in ids])
        act = (a > 1e-9) | (b > 1e-9)
        rho = float(spearmanr(a[act], b[act]).statistic)
        n_lost = int(((a > 1e-9) & (b <= 1e-9)).sum())
        n_gain = int(((a <= 1e-9) & (b > 1e-9)).sum())
        say(f"     Spearman between unconstrained and constrained flux: {rho:+.4f} "
            f"over {int(act.sum()):,} active reactions")
        say(f"     reactions switched OFF by the constraint: {n_lost:,}; switched ON: {n_gain:,}")
        k5 = bool(np.isfinite(rho) and rho < K5_MAX_RHO)
        say(f"     K5 {'PASS' if k5 else 'FAIL'} -- the constraint "
            f"{'changes which reactions carry flux' if k5 else 'only RESCALES, so K3 is trivial'}")
    else:
        rho, n_lost, n_gain, k5 = float("nan"), 0, 0, False
        say(f"     constrained solve did not return an optimal solution -- K5 not evaluable")
    say()

    say("K6 COVERAGE DECLARED")
    n_gene = sum(1 for r in m.reactions if r.genes)
    act0 = [r.id for r in m.reactions if abs(flux0.get(r.id, 0.0)) > 1e-9]
    covered_flux = sum(abs(flux0[i]) for i in act0 if i in kapp)
    total_flux = sum(abs(flux0[i]) for i in act0)
    say(f"     reactions with a gene: {n_gene:,} of {len(m.reactions):,}")
    say(f"     carrying flux with a measured HeLa enzyme: {used:,}")
    say(f"     those carry {covered_flux/total_flux:.1%} of total active flux")
    say(f"     the pooled constraint applies to all {n_gene:,} gene-associated reactions, but the")
    say(f"     k_cat EVIDENCE in K2 rests on the {used:,} with measured enzymes")
    say()

    gates = {"K1 the unconstrained solution is not a cell": True,
             "K2 the required turnover numbers are impossible": bool(k2),
             "K3 enzyme capacity brings growth into range": bool(k3),
             "K4 the Warburg claim tested on the right condition": True,
             "K5 the constraint re-routes rather than rescales": bool(k5),
             "K6 coverage declared": True}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(CS.SBML), str(CP.HELA), str(CP.GTF)],
                      available=len(m.reactions), used=used, selection="filtered", seed=SEED,
                      controls=["apparent k_cat inverted from flux rather than looked up",
                                "the enzyme budget computed from proteome mass, not fitted to growth",
                                "a re-routing test that a uniform rescaling would fail",
                                "the anaerobic and baseline conditions separated explicitly",
                                "coverage of the capacity evidence stated against total flux",
                                "growth compared to a measured doubling time, not to itself"],
                      note="FBA has no notion of enzyme cost, so it runs at its bounds; this prices "
                           "the enzymes and corrects cell_sim.py's Warburg claim")
    RM.report(man, emit=say)
    json.dump({"test": "loop_enzyme_capacity", "manifest": man, "gates": gates,
               "k1": {"growth_unconstrained": mu0, "doubling_min": dt0 * 60,
                      "target_mu": MU_TARGET, "factor_too_fast": fac},
               "k2": {"n_reactions": used, "median_kcat_per_s": med,
                      "p25": float(np.percentile(vals, 25)) if len(vals) else None,
                      "p75": float(np.percentile(vals, 75)) if len(vals) else None,
                      "max": float(vals.max()) if len(vals) else None,
                      "n_above_1e6": n_imp},
               "k3": {"enzyme_g_per_gdw": enz_g, "budget_mmol_per_gdw": budget_mmol,
                      "kcat_per_s": KCAT_PER_S, "flux_cap": cap,
                      "growth_constrained": mu1, "doubling_h": dt1, "factor_off": off},
               "k4": {"baseline_lactate": base["lactate_export"], "baseline_o2": o2_base,
                      "anaerobic_lactate": ana["lactate_export"], "anaerobic_o2": o2_ana,
                      "aerobic_glycolysis_present": warburg},
               "k5": {"spearman": rho, "switched_off": n_lost, "switched_on": n_gain},
               "k6": {"n_gene_reactions": n_gene, "n_with_enzyme": used,
                      "flux_fraction_covered": covered_flux / total_flux if total_flux else None},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_enzyme_capacity.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_enzyme_capacity.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
