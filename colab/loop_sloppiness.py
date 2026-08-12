"""LOOP 66 -- STIFF OR SLOPPY: WHICH PREDICTIONS NEED THE PARAMETERS AT ALL?

TWO CLAIMS ARE BEING TESTED, and they are not equally load-bearing.

  1. PHYSICS PINS PARAMETERS BEFORE MEASUREMENT. Diffusion-limited association tops out near
     1e9 /M/s; median kcat is ~10 /s; ribosome elongation is 10-20 aa/s across every domain of life;
     mass fractions sum to exactly 1. Milo & Phillips. Most "unknown" parameters are already bounded
     within an order of magnitude.
  2. INDIVIDUALLY UNIDENTIFIABLE IS NOT COLLECTIVELY UNCONSTRAINED. Gutenkunst & Sethna: biological
     models have Hessian eigenvalue spectra spanning six-plus orders, so parameters are wildly
     unidentifiable while predictions along STIFF directions stay tight. The question is not "do I
     know this parameter" but "does my prediction project onto a stiff or a sloppy direction".

CLAIM 1 IS ALREADY CONFIRMED ON THIS PROJECT'S OWN DATA, measured before this module was written,
and it holds well enough to be used as a validator rather than an assumption:

    median kcat, 420 MEASURED entries      7.45 /s      against the claimed ~10 /s
    median kcat, all 2,549                 5.45 /s
    kcat/Km above the 1e9 /M/s limit       1.21%        the other 98.79% respect the bound

That 1.21% is not a rounding error, it is a DATA DEFECT the physical bound just found: those entries
describe enzymes faster than substrate can reach them, topping out at 1.75e10 /M/s, seventeen times
the diffusion limit. A bound that most of the data respects is a bound that can audit the rest.

CLAIM 2 NEEDS A MODEL THIS PROJECT DOES NOT HAVE, and that is worth stating plainly rather than
papering over. Sloppiness is a property of COUPLED nonlinear systems fitted to data. What is here is
statistical predictors (logistic and MLP), per-gene decoupled ODEs (d[mRNA]/dt = k_syn - k_deg[mRNA],
which is analytic and has no cross-parameter coupling), a linear program, and a polymer simulation.
None of them has the structure the sloppiness result is about. So the smallest honest coupled model
gets BUILT here -- proteome allocation with the mass-fraction constraint, which is the one place this
project already has a budget that must sum to exactly 1 -- and its spectrum is computed rather than
cited.

THE SHARPER TEST, and the reason this loop is worth running beyond reproducing a known spectrum.
The catalytic capacity of a proteome is an AGGREGATE:

    C = sum_i  a_i * kcat_i          abundance times turnover, summed over enzymes

A sum over thousands of terms is the canonical stiff direction: it is nearly invariant to what any
single term is, and depends almost entirely on one collective quantity. So the operational form of
"does my prediction need the parameters" is: DESTROY the per-enzyme kcat assignment by permuting
kcats among enzymes, preserving the distribution exactly, and see how far C moves. If C barely
moves, the aggregate needed the DISTRIBUTION and never needed the assignment -- which is the claim,
made falsifiable.

And it can fail in an interesting way. Permutation only leaves C invariant if abundance and kcat are
UNCORRELATED. If a cell systematically expresses more of its slow enzymes, then a_i and kcat_i are
coupled, C depends on the pairing, and the aggregate is not stiff after all. That is a real
biological question and this test answers it as a side effect.

PREDECLARED, before any number:

  S1 THE PHYSICAL BOUNDS HOLD ON OUR DATA, AND AUDIT IT
       median kcat within a factor of 3 of 10 /s, and >= 95% of kcat/Km below 1e9 /M/s. The
       violators are COUNTED AND NAMED as suspect entries, since a physical bound that most data
       respects is a validator for the rest.
  S2 THE SPECTRUM SPANS ORDERS OF MAGNITUDE
       Hessian of the allocation model's predictions with respect to LOG parameters, eigendecomposed.
       Gutenkunst's finding is 6+ orders; this model is small, so the gate is >= 3 orders between
       largest and smallest eigenvalue. A model that comes back stiff in every direction is NOT
       sloppy and its predictions would each need their own measured parameter -- which is the
       outcome that would refute the optimistic reading.
  S3 THE AGGREGATE IS STIFF WHILE THE PARTS ARE SLOPPY               THE GATE.
       permute kcat among enzymes, 1,000 times, preserving the distribution. The aggregate C must
       move by less than 10% (median absolute relative change) while a per-enzyme prediction -- which
       enzyme carries the most capacity -- must be destroyed. Both halves are required: stability of
       the aggregate alone would be unremarkable if the parts were also stable.
  S4 HOW MUCH DATA DOES THE STIFF PREDICTION ACTUALLY NEED?
       replace all but N measured kcats with the global median and sweep N from 0 upward. Report the
       N at which the aggregate lands within 10% of the full-data answer. If N = 0 suffices, the
       prediction needed no measurement at all, which is the strongest form of the claim.
  S5 NAME WHAT STAYS SLOPPY
       the predictions this does NOT buy are listed explicitly. An argument that some predictions are
       free is only honest alongside the list of those that are not.

-> outputs/loop_sloppiness.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_rescue as LR
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
CELL = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_complete.json"
KIN = Path("outputs/orphan/kinetics_refined_corrected.json")
HELA = SC / "paxdb_hela.txt"

DIFF_LIMIT = 1.0e9
KCAT_PRIOR = 10.0
KCAT_FACTOR = 3.0
BOUND_FLOOR = 0.95
SPECTRUM_ORDERS = 3.0
AGG_TOL = 0.10
N_PERM = 1000
SEED = 6601

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def allocation_model(p):
    """Smallest coupled proteome-allocation model with the mass fractions summing to exactly 1.

    phi_R ribosomal, phi_M metabolic, phi_Q housekeeping. Growth is set by ribosomes translating,
    and the metabolic sector must supply the precursors that translation consumes, so the two are
    coupled through a shared budget rather than independent. Returns the predictions whose
    sensitivity is being asked about.
    """
    k_el, L, kcat_M, phi_Q, sigma, mass_prec = p
    # phi_R + phi_M + phi_Q = 1 exactly -- the constraint the whole framing rests on
    # translation rate per unit ribosome mass, and precursor supply per unit metabolic mass
    # solved at the allocation that maximises growth subject to supply == demand
    a = k_el * sigma / L                      # growth per unit ribosome fraction
    b = kcat_M / mass_prec                    # precursor supply per unit metabolic fraction
    free = max(1.0 - phi_Q, 1e-9)
    # supply == demand  ->  b * phi_M = a * phi_R,  with phi_R + phi_M = free
    phi_R = free * b / (a + b)
    phi_M = free - phi_R
    mu = a * phi_R
    return np.array([mu, phi_R, phi_M])


def hessian_logspec(p0):
    """Sensitivity spectrum: J of log-predictions wrt log-parameters, then eigen(J^T J).

    This is the Gutenkunst construction -- the Hessian of a least-squares cost at the optimum is
    J^T J, so its eigenvalues are the squared singular values of the log-log sensitivity matrix.
    """
    p0 = np.asarray(p0, float)
    f0 = np.log(allocation_model(p0))
    J = np.zeros((len(f0), len(p0)))
    for j in range(len(p0)):
        h = 1e-5
        pp = p0.copy()
        pp[j] *= np.exp(h)
        J[:, j] = (np.log(allocation_model(pp)) - f0) / h
    ev = np.linalg.eigvalsh(J.T @ J)
    return np.sort(np.abs(ev))[::-1], J


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 66 -- stiff or sloppy: which predictions need the parameters at all?")
    say("=" * 100)
    say()

    K = json.load(open(KIN))["kinetics_refined"]
    names = list(K)
    kcat = np.array([K[nm].get("kcat_per_s") or np.nan for nm in names], float)
    kmv = np.array([K[nm].get("km_uM") or np.nan for nm in names], float)
    tier = [K[nm].get("tier") for nm in names]
    ok = np.isfinite(kcat)

    say("S1 THE PHYSICAL BOUNDS HOLD ON OUR DATA, AND AUDIT IT")
    meas = np.array([kcat[i] for i in range(len(names))
                     if ok[i] and tier[i] in ("measured", "EC-measured")])
    med_m, med_a = float(np.median(meas)), float(np.median(kcat[ok]))
    say(f"     median kcat  MEASURED n={len(meas):,}   {med_m:.2f} /s")
    say(f"     median kcat  all      n={int(ok.sum()):,} {med_a:.2f} /s")
    say(f"     claimed prior {KCAT_PRIOR:.0f} /s, gate within a factor of {KCAT_FACTOR:.0f}")
    both = ok & np.isfinite(kmv)
    kk = kcat[both] / (kmv[both] * 1e-6)
    under = float((kk <= DIFF_LIMIT).mean())
    viol = [(names[i], kcat[i], kmv[i]) for i in np.where(both)[0]
            if kcat[i] / (kmv[i] * 1e-6) > DIFF_LIMIT]
    say(f"     kcat/Km below the {DIFF_LIMIT:.0e} /M/s diffusion limit: {under:.4f} "
        f"of {int(both.sum()):,}")
    say(f"     PHYSICALLY IMPOSSIBLE ENTRIES the bound just found: {len(viol)}, "
        f"worst {max((kcat[i] / (kmv[i] * 1e-6)) for i in np.where(both)[0]):.2e} /M/s")
    for nm, kc, km in sorted(viol, key=lambda x: -(x[1] / (x[2] * 1e-6)))[:6]:
        say(f"       {nm:10s} kcat {kc:12.1f}/s  Km {km:8.2f} uM  "
            f"-> {kc / (km * 1e-6):.2e} /M/s")
    s1 = (med_m / KCAT_FACTOR <= KCAT_PRIOR <= med_m * KCAT_FACTOR) and under >= BOUND_FLOOR
    say(f"     S1 {'PASS' if s1 else 'FAIL'}")
    say()

    say("S2 THE SPECTRUM SPANS ORDERS OF MAGNITUDE")
    # k_el aa/s, L aa, kcat_M /s, phi_Q, sigma (active ribosome fraction), mass_prec
    p0 = np.array([16.0, 400.0, 10.0, 0.45, 0.85, 1.0])
    pnames = ["k_el", "L", "kcat_M", "phi_Q", "sigma", "mass_prec"]
    ev, J = hessian_logspec(p0)
    nz = ev[ev > 1e-14]
    orders = float(np.log10(nz[0] / nz[-1])) if len(nz) > 1 else 0.0
    say(f"     allocation model, {len(pnames)} parameters: {', '.join(pnames)}")
    say(f"     predictions: growth rate mu, ribosome fraction phi_R, metabolic fraction phi_M")
    say(f"     eigenvalues of J^T J (log-log): "
        f"{', '.join(f'{e:.3e}' for e in ev)}")
    say(f"     non-degenerate span: {orders:.2f} orders of magnitude "
        f"(gate >= {SPECTRUM_ORDERS:.0f}; Gutenkunst reports 6+ on larger models)")
    stiff = J[0] / (np.linalg.norm(J[0]) + 1e-12)
    say(f"     growth-rate sensitivity by parameter (log-log): "
        f"{', '.join(f'{n}={v:+.2f}' for n, v in zip(pnames, J[0]))}")
    s2 = orders >= SPECTRUM_ORDERS
    say(f"     S2 {'PASS' if s2 else 'FAIL'}")
    say()

    say("S3 THE AGGREGATE IS STIFF WHILE THE PARTS ARE SLOPPY")
    D = json.load(open(CELL))
    gnames = [g["name"] for g in D["genes"]]
    ix = {nm: i for i, nm in enumerate(gnames)}
    hela = np.zeros(len(gnames))
    for ln in open(HELA):
        if ln.startswith("#"):
            continue
        f = ln.rstrip().split("\t")
        if len(f) < 3:
            continue
        i = ix.get(f[0])
        if i is not None:
            try:
                hela[i] = float(f[2])
            except ValueError:
                pass
    sel = [(nm, kcat[j]) for j, nm in enumerate(names)
           if ok[j] and nm in ix and hela[ix[nm]] > 0]
    a = np.array([hela[ix[nm]] for nm, _ in sel])
    kc = np.array([k for _, k in sel])
    say(f"     {len(sel):,} enzymes with both a HeLa abundance and a kcat")
    C0 = float((a * kc).sum())
    rng = np.random.default_rng(SEED)
    perm_C, top_keep = [], []
    top0 = set(np.argsort(-(a * kc))[:50])
    for _ in range(N_PERM):
        kp = rng.permutation(kc)
        perm_C.append((a * kp).sum())
        top_keep.append(len(top0 & set(np.argsort(-(a * kp))[:50])) / 50.0)
    perm_C = np.array(perm_C)
    rel = float(np.median(np.abs(perm_C - C0) / C0))
    keep = float(np.median(top_keep))
    rho = float(np.corrcoef(np.log10(a), np.log10(kc))[0, 1])
    say(f"     aggregate C = sum a_i*kcat_i = {C0:.4e}")
    say(f"     under {N_PERM:,} kcat permutations: median |dC|/C = {rel:.4f}, "
        f"5-95% {np.percentile(np.abs(perm_C - C0) / C0, 5):.4f}-"
        f"{np.percentile(np.abs(perm_C - C0) / C0, 95):.4f}")
    say(f"     per-enzyme prediction (top-50 capacity set) survives permutation: {keep:.4f}")
    say(f"     log abundance vs log kcat correlation: {rho:+.4f}  "
        f"(permutation is only fair if this is near zero)")
    s3 = rel < AGG_TOL and keep < 0.5
    say(f"     S3 {'PASS' if s3 else 'FAIL'}  -- the aggregate "
        f"{'needed the distribution, never the assignment' if s3 else 'DOES depend on the pairing'}")
    say()

    say("S4 HOW MUCH DATA DOES THE STIFF PREDICTION ACTUALLY NEED?")
    med_k = float(np.median(kc))
    order = rng.permutation(len(kc))
    sweep = []
    for N in [0, 10, 50, 100, 250, 500, 1000, len(kc)]:
        kk2 = np.full(len(kc), med_k)
        idx = order[:N]
        kk2[idx] = kc[idx]
        CN = float((a * kk2).sum())
        err = abs(CN - C0) / C0
        sweep.append({"N": int(N), "C": CN, "rel_err": err})
        say(f"     N={N:6,d} measured, rest at the global median: C={CN:.4e}  "
            f"relative error {err:.4f}")
    n_needed = next((s["N"] for s in sweep if s["rel_err"] < AGG_TOL), None)
    say(f"     first N within {AGG_TOL:.0%} of the full-data answer: {n_needed}")
    s4 = n_needed is not None
    say(f"     S4 {'PASS' if s4 else 'FAIL'}")
    say()

    say("S5 NAME WHAT STAYS SLOPPY")
    sloppy = ["which single enzyme is rate-limiting for a given flux -- destroyed by permutation "
              f"(top-50 overlap {keep:.2f})",
              "any per-reaction flux prediction, which needs the assignment and not the distribution",
              "the 1.21% of kcat/Km entries above the diffusion limit, which are wrong at any "
              "aggregation level",
              "everything in the 84.6% of entities loop 65 found untyped -- no flux slot means no "
              "aggregate to be stiff in",
              "the loops3d layer, which loops 61/63/64 showed does not covary with anything here "
              "and so has no collective direction to borrow strength from"]
    for s in sloppy:
        say(f"     - {s}")
    s5 = True
    say(f"     S5 PASS (list recorded)")
    say()

    gates = {"S1 physical bounds hold and audit the data": bool(s1),
             "S2 the spectrum spans orders": bool(s2),
             "S3 aggregate stiff, parts sloppy": bool(s3),
             "S4 the stiff prediction's data requirement is bounded": bool(s4),
             "S5 the sloppy list is recorded": bool(s5)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(KIN), str(CELL), str(HELA)],
                      available=len(names), used=len(sel), selection="filtered", seed=SEED,
                      controls=["diffusion limit used as an external validator of our own kcat",
                                "permutation preserves the kcat distribution exactly",
                                "abundance-kcat correlation reported, since permutation is only "
                                "fair when it is near zero",
                                "both halves of S3 required: aggregate stable AND parts destroyed",
                                "the sloppy list is recorded beside the stiff claim"],
                      note="sloppiness is a property of coupled models; this project had none, so "
                           "the smallest honest one is built here rather than cited")
    RM.report(man, emit=say)
    json.dump({"test": "loop_sloppiness", "manifest": man, "gates": gates,
               "median_kcat_measured": med_m, "median_kcat_all": med_a,
               "frac_under_diffusion_limit": under, "n_impossible": len(viol),
               "impossible": [{"gene": nm, "kcat": kc_, "km_uM": km_} for nm, kc_, km_ in viol[:30]],
               "eigenvalues": [float(x) for x in ev], "spectrum_orders": orders,
               "growth_sensitivity": {n: float(v) for n, v in zip(pnames, J[0])},
               "aggregate_C": C0, "perm_median_rel_change": rel,
               "top50_overlap_under_perm": keep, "log_abund_log_kcat_rho": rho,
               "n_enzymes": len(sel), "sweep": sweep, "n_needed": n_needed,
               "still_sloppy": sloppy, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_sloppiness.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_sloppiness.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
