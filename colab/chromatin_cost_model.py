"""WHAT WOULD 4D CHROMATIN ACTUALLY COST -- arithmetic from measured numbers, not an estimate of an estimate.

WHY A FILE AND NOT A PARAGRAPH.  "Can we simulate a chromosome with polymerase, twist, denaturation and
replication" has a numeric answer, and the numbers are already in this repo: one nucleosome solved, its
sparsity measured, its solver timed, its relaxation spectrum computed. Everything below is those numbers
multiplied out. Where a quantity is NOT measured here it is marked ESTIMATE and its basis given, because the
difference between "we measured this" and "this is the literature default" is the difference between a cost
model and a wish.

MEASURED HERE, and every one of these came from a run in this repo:
    13,625 atoms in one nucleosome (1KX5), 263,765 springs, 4,870,395 Hessian nonzeros
    two-level Schwarz: 64 iterations, 1.54 s fine-level, against 21.87 s for a direct solve
    relaxation times spanning tau_min 7.2 to tau_max 48,794 in the same particle
    total arithmetic scales as roughly H^1.2 in sphere size

THE THREE WALLS, and they are hit in this order:
    MEMORY     the atomistic Hessian at chromosome scale. This one is absolute -- it is not a matter of
               waiting longer.
    TIMESTEP   the reason molecular dynamics cannot reach biological time. This is the wall everyone means
               when they say 4D chromatin is impossible.
    EVENTS     what is left after the first two are dodged, and it is the one that is affordable.

THE ARGUMENT THIS FILE EXISTS TO CHECK.  A timestep integrator pays for every 1e-8 s whether anything
happened or not. But chromatin at 37 C is overdamped (Reynolds ~1e-9) and its elastic relaxation is FAST
compared to the biology: a polymerase adds one base every ~30 ms while the structure re-equilibrates orders
of magnitude faster. If that separation holds, the correct scheme is not integration at all -- it is
"advance to the next event, re-solve the static problem, repeat", and the cost becomes the number of EVENTS
rather than the number of timesteps. This file computes both and reports the ratio.

-> outputs/chromatin_cost_model.json
"""
import json
import os
import sys
import time
from pathlib import Path

OUT = Path(os.environ.get("CELL_OUT", "outputs"))

# ---- MEASURED IN THIS REPO ------------------------------------------------------------------------
N_ATOM_NUC = 13_625          # nexus_methyl_propagate.json, 1KX5 with waters stripped
NNZ_NUC = 4_870_395          # chromatin_sphere_scaling.json
T_DIRECT_NUC = 21.868        # s, dense-ish direct solve, one nucleosome
T_FINE_NUC = 1.544           # s, two-level Schwarz fine level
T_MODAL_BUILD = 79.569       # s, eigensolve for the coarse space -- paid ONCE per structure family
TAU_MIN, TAU_MAX = 7.232, 48_793.5     # chromatin_time.json, relaxation-time spread within one particle

# ---- ESTIMATES, each with its basis ---------------------------------------------------------------
BP_GENOME = 3.1e9            # haploid
BP_CHR1 = 249e6
NRL = 200                    # nucleosome repeat length, bp. ESTIMATE: standard human value
BYTES_NZ = 12                # 8-byte value + 4-byte index
FLOPS_PER_CORE = 1e9         # ESTIMATE, deliberately conservative: sustained useful flop/s per
                             # core for sparse matvec work, which is memory-bound rather than flop-bound
CORES = int(os.environ.get("CORES", 4))
DOF_BEAD = 4                 # x, y, z, twist
FLOPS_BEAD_STEP = 200        # ESTIMATE: bonded + bending + twist + neighbour-list excluded volume
DT_BD = 1e-8                 # s. ESTIMATE: typical Brownian-dynamics timestep for ~10 nm chromatin beads
NNZ_BEAD = 256               # ESTIMATE: ~16 interacting neighbours x a 4x4 block. Bonded 2, bending 2,
                             # twist 2, excluded volume ~10 from a neighbour list.
N_COARSE = 200               # ESTIMATE: coarse-space size. The measured run used 194 modes on one
                             # nucleosome and converged in 64 iterations.
ITERS_COLD = 64              # MEASURED, chromatin_twolevel.json, modal coarse space
ITERS_WARM = 5               # ESTIMATE, and the softest number here -- see the note in the output
POL_RATE = 33.0              # bp/s. RNA Pol II elongation, ~2 kb/min
GENE_BP = 10_000             # a mid-sized gene


def fmt(x, unit=""):
    for d, s in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= d:
            return f"{x/d:.1f} {s}{unit}"
    return f"{x:.1f} {unit}"


def secs(s):
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{s/60:.1f} min"
    if s < 86400:
        return f"{s/3600:.1f} h"
    if s < 86400 * 365:
        return f"{s/86400:.1f} days"
    return f"{s/86400/365:.1f} years"


def main():
    log = []

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("WHAT 4D CHROMATIN WOULD COST -- multiplied out from what was actually measured here")
    report("=" * 100)
    nuc_chr1 = BP_CHR1 / NRL
    nuc_gen = BP_GENOME / NRL
    report(f"\n  chr1 {BP_CHR1/1e6:.0f} Mbp -> {fmt(nuc_chr1)} nucleosomes | "
           f"genome {BP_GENOME/1e9:.1f} Gbp -> {fmt(nuc_gen)} nucleosomes  (at {NRL} bp repeat)")

    # ---- WALL 1: MEMORY, and it settles the atomistic route ---------------------------------------
    report("\n  WALL 1 -- MEMORY. The atomistic route, priced from the measured sparsity.")
    nnz_per_atom = NNZ_NUC / N_ATOM_NUC
    report(f"    measured: {NNZ_NUC:,} nonzeros over {N_ATOM_NUC:,} atoms = {nnz_per_atom:.0f} per atom")
    rows = []
    for lab, nuc in (("one nucleosome", 1), ("chr1", nuc_chr1), ("genome", nuc_gen)):
        at = nuc * N_ATOM_NUC
        nz = at * nnz_per_atom
        by = nz * BYTES_NZ
        rows.append((lab, at, nz, by))
        report(f"    {lab:<16}{fmt(at,'atoms'):>16}{fmt(nz,'nonzeros'):>20}{fmt(by,'B'):>14} for the Hessian alone")
    report("    Chromosome-scale atomistics is not slow, it is impossible: the matrix does not fit, and")
    report("    no amount of waiting changes that. This is why the route below is coarse-grained.")

    # ---- WALL 2: TIMESTEP, and it settles naive Brownian dynamics ---------------------------------
    report("\n  WALL 2 -- TIMESTEP. A twistable worm-like chain, one bead per nucleosome, integrated.")
    report(f"    {DOF_BEAD} degrees of freedom per bead (position + twist), {FLOPS_BEAD_STEP} flops per bead-step (ESTIMATE)")
    tx_time = GENE_BP / POL_RATE
    for lab, nuc in (("chr1", nuc_chr1), ("genome", nuc_gen)):
        mem = nuc * DOF_BEAD * 8
        per_step = nuc * FLOPS_BEAD_STEP
        t_step = per_step / (FLOPS_PER_CORE * CORES)
        nsteps = tx_time / DT_BD
        report(f"    {lab:<10} memory {fmt(mem,'B'):>10} | {secs(t_step)}/step | "
               f"transcribing one {GENE_BP//1000} kb gene takes {tx_time:.0f} s biological")
        report(f"    {'':<10} = {fmt(nsteps,'steps')} at dt={DT_BD:.0e} s -> {secs(nsteps*t_step)} of compute")
    report("    Memory is now trivial and the wall has MOVED, not gone. Integration pays for every")
    report(f"    {DT_BD:.0e} s whether anything happened or not, and biology runs on seconds.")

    # ---- WALL 3: EVENTS, and this is the one that is affordable ------------------------------------
    report("\n  WALL 3 -- OR NOT A WALL. Stop integrating. The measured relaxation spectrum says why.")
    report(f"    Chromatin is overdamped (Reynolds ~1e-9) and this repo measured its relaxation times")
    report(f"    spanning tau {TAU_MIN:.1f} to {TAU_MAX:.0f} within a single nucleosome. A polymerase adds")
    report(f"    one base every {1000/POL_RATE:.0f} ms. If elastic re-equilibration is fast against that --")
    report("    and overdamped relaxation is -- then nothing is gained by resolving the interval. The")
    report("    scheme becomes: advance to the next EVENT, re-solve the static problem, repeat.")
    report("    That is exactly the closed-form solver already built here, used as a stepper.")
    report("")
    report("    AND THE PER-EVENT COST IS NOT A FULL RE-SOLVE, which is the whole point of the two-level")
    report("    machinery. Priced at the iteration level instead, because that is what the solver does:")
    report(f"      matvec        2 x nnz, with ~{NNZ_BEAD} nonzeros per bead (bonded, bending, twist, excluded volume)")
    report(f"      coarse        2 x DOF x {N_COARSE} modes -- project, solve a diagonal, prolong")
    report("      local solves  one small dense solve per sphere, ~one matvec in total at the measured H^1.2")
    report(f"    A COLD solve took {ITERS_COLD} two-level iterations here. A polymerase step perturbs a few")
    report("    hundred base pairs and the previous state is an excellent initial guess, so a WARM restart")
    report(f"    should need far fewer -- {ITERS_WARM} is assumed below and is the softest number in this file.")
    report("")
    scen = []
    for lab, nuc, ev in (
            (f"transcribe one {GENE_BP//1000} kb gene, 10 bp/event", nuc_chr1, GENE_BP / 10),
            ("transcribe 1,000 genes concurrently", nuc_chr1, 1000 * GENE_BP / 10),
            ("one replication fork, 1 kb/event, chr1", nuc_chr1, BP_CHR1 / 1000),
            ("full S phase, genome, 1 kb/event", nuc_gen, BP_GENOME / 1000)):
        dof = nuc * DOF_BEAD
        f_it = 2 * nuc * NNZ_BEAD + 2 * dof * N_COARSE + 2 * nuc * NNZ_BEAD
        t_it = f_it / (FLOPS_PER_CORE * CORES)
        scen.append({"scenario": lab, "events": ev, "s_per_iter": t_it,
                     "warm_s": ev * ITERS_WARM * t_it, "cold_s": ev * ITERS_COLD * t_it})
    report(f"    {'scenario':<34}{'events':>9}{'per event':>11}{'warm':>12}{'cold':>12}")
    for s in scen:
        report(f"    {s['scenario']:<34}{fmt(s['events']):>9}"
               f"{secs(s['s_per_iter']*ITERS_WARM):>11}{secs(s['warm_s']):>12}{secs(s['cold_s']):>12}")
    report(f"    on {CORES} cores. These divide by cores almost exactly -- every operation above is a")
    report("    matvec or a dense projection, both of which parallelise without communication headaches.")
    report(f"    On 256 cores the S-phase row falls to {secs(scen[-1]['warm_s']*CORES/256)}.")

    # ---- ADAPTIVE REFINEMENT, which changes what the answer even depends on -------------------------
    # chromatin_timescale measured tau_max = 47.2 ns for a 147 bp particle and showed the quasi-static
    # separation FAILS above ~100 kb, so long-range conformation has to be integrated rather than
    # event-stepped. Integration cost is then set by the timestep, and the timestep is bounded by the
    # FASTEST retained mode -- which coarse-graining softens as b^2. So run coarse everywhere and refine
    # only where resolution is needed. 200 bp is the floor and not an arbitrary one: DNA's persistence
    # length is 147 bp, so at 200 bp a segment is 1.36 Lp and below that a worm-like chain has stopped
    # being a meaningful model of anything.
    report("\n  ADAPTIVE REFINEMENT -- coarse everywhere, fine only where it is needed")
    TAU_NUC, NUC_BP = 4.7219e-8, 147          # MEASURED, chromatin_timescale.json
    TX = GENE_BP / POL_RATE

    def level(bp):
        tf = TAU_NUC * (bp / NUC_BP) ** 2      # fastest retained mode after coarse-graining
        dt = tf / 10                           # stability margin
        return tf, dt, TX / dt

    tf0, dt0, st0 = level(10_000)
    tfF, dtF, stF = level(200)
    report(f"    coarse 10 kb   dt {dt0*1e6:6.2f} us   {st0:.3g} steps")
    report(f"    fine   200 bp  dt {dtF*1e9:6.2f} ns   {stF:.3g} steps   ({200/NUC_BP:.2f} persistence lengths)")
    WIN = 1e4                                  # fine window tracking the polymerase
    report(f"\n    THE DOMAIN IS ALMOST FREE. {WIN/1e3:.0f} kb fine window, multiple time stepping:")
    report(f"    {'domain':<14}{'coarse beads':>14}{'4 cores':>11}{'256 cores':>12}")
    amr = []
    for lab, dom in (("100 kb", 1e5), ("1 Mb", 1e6), ("10 Mb", 1e7), ("chr1 249 Mb", 249e6)):
        bs = st0 * ((dom - WIN) / 10_000) + stF * (WIN / 200)
        c = bs * FLOPS_BEAD_STEP / (FLOPS_PER_CORE * CORES)
        amr.append({"domain": lab, "bead_steps": bs, "seconds_4core": c})
        report(f"    {lab:<14}{(dom-WIN)/1e4:>14.0f}{secs(c):>11}{secs(c*CORES/256):>12}")
    report("    A 2,490-fold increase in domain costs 20% more, because the coarse level is nothing.")
    report(f"\n    IT IS ENTIRELY THE FINE WINDOW, and that is linear in window size:")
    report(f"    {'window':<14}{'fine beads':>12}{'4 cores':>11}{'256 cores':>12}")
    for lab, w in (("2 kb", 2e3), ("10 kb", 1e4), ("50 kb", 5e4), ("100 kb", 1e5)):
        bs = st0 * ((1e6 - w) / 10_000) + stF * (w / 200)
        c = bs * FLOPS_BEAD_STEP / (FLOPS_PER_CORE * CORES)
        report(f"    {lab:<14}{w/200:>12.0f}{secs(c):>11}{secs(c*CORES/256):>12}")
    uni = stF * (1e6 / 200)
    naive = stF * ((1e6 - WIN) / 10_000 + WIN / 200)
    mts = st0 * ((1e6 - WIN) / 10_000) + stF * (WIN / 200)
    report(f"\n    MULTIPLE TIME STEPPING IS NOT OPTIONAL. With one global dt, refining anywhere forces the")
    report(f"    fine timestep EVERYWHERE and most of the saving evaporates:")
    report(f"      uniform 200 bp everywhere          {secs(uni*FLOPS_BEAD_STEP/(FLOPS_PER_CORE*CORES)):>10}"
           f"   {uni/mts:>6.0f}x")
    report(f"      refine the window, one global dt   {secs(naive*FLOPS_BEAD_STEP/(FLOPS_PER_CORE*CORES)):>10}"
           f"   {naive/mts:>6.1f}x")
    report(f"      refine the window + MTS            {secs(mts*FLOPS_BEAD_STEP/(FLOPS_PER_CORE*CORES)):>10}"
           f"   {1.0:>6.1f}x")

    # ---- SEQUENTIAL REFINEMENT, which is cheaper again and buys something different -----------------
    # Run the COARSE level for the full biological time, then refine level by level, each initialised
    # from the one above. The modes that APPEAR when the bead is halved are exactly those with relaxation
    # times between tau(b) and tau(2b) = 4 tau(b), so equilibrating them needs ~5 tau(2b) = 20 tau(b) of
    # simulated time at dt = tau(b)/10 -- about 200 steps, INDEPENDENT of the level. The refinement is
    # therefore nearly free and the whole hierarchy costs what its coarsest level costs.
    report("\n  SEQUENTIAL REFINEMENT -- coarse for the full time, then break the beads down")
    report("    The modes that appear on halving a bead have tau between tau(b) and 4 tau(b), so they")
    report("    equilibrate in ~20 tau(b) at dt = tau(b)/10: about 200 steps per level, at every level.")
    report(f"    {'level':<7}{'bead':>8}{'dt':>11}{'sim time':>12}{'steps':>10}{'beads':>8}{'bead-steps':>12}")
    DOM = 1e6
    seq, tot = [], 0.0
    for i, b in enumerate((10_000, 5000, 2500, 1250, 625, 312, 200)):
        tfk = TAU_NUC * (b / NUC_BP) ** 2
        dtk = tfk / 10
        nb = DOM / b
        simt, stk = (TX, TX / dtk) if i == 0 else (20 * tfk, 200.0)
        bs = stk * nb
        tot += bs
        seq.append({"level": i, "bead_bp": b, "dt_s": dtk, "sim_time_s": simt, "bead_steps": bs})
        report(f"    L{i:<6}{b:>6.0f}bp{dtk*1e6:>9.4f}us{simt:>11.3g}s{stk:>10.0f}{nb:>8.0f}{bs:>12.3g}"
               + ("  FULL biological time" if i == 0 else ""))
    l0 = seq[0]["bead_steps"]
    uni = (TX / ((TAU_NUC * (200 / NUC_BP) ** 2) / 10)) * (DOM / 200)
    c_tot = tot * FLOPS_BEAD_STEP / (FLOPS_PER_CORE * CORES)
    report(f"    TOTAL {tot:.3g} bead-steps = {secs(c_tot)} for 1 Mb on {CORES} cores")
    report(f"    The coarse level is {l0/tot:.1%} of that; all six refinements together add {(tot-l0)/l0:.2%}.")
    report(f"    Against uniform 200 bp for the whole {TX:.0f} s: "
           f"{secs(uni*FLOPS_BEAD_STEP/(FLOPS_PER_CORE*CORES))}, a {uni/tot:.0f}x difference.")

    report("\n    WHAT IT ACTUALLY BUYS, and this is the whole caveat. Sequential refinement gives an")
    report("    EQUILIBRATED FINE SNAPSHOT, not a fine TRAJECTORY. It is legitimate because the fine")
    report(f"    modes are slaved to the coarse ones -- tau(200 bp) = {TAU_NUC*(200/NUC_BP)**2*1e9:.0f} ns against "
           f"tau(10 kb) = {TAU_NUC*(10000/NUC_BP)**2*1e6:.0f} us,")
    report("    a factor of ~2,500 -- so the fine structure is a FUNCTION of the coarse state and can be")
    report("    regenerated on demand rather than carried.")
    report("    It breaks precisely when that stops being true:")
    report("      HYSTERESIS   a plectoneme pinned at a sequence feature, a bubble that stays open. Then")
    report("                   the fine state depends on history, not on the current coarse state.")
    report("      FEEDBACK     a melted segment has C ~ 0 and ssDNA bending stiffness, which changes the")
    report("                   COARSE rod's mechanics. Tier 1B is exactly this case, so denaturation")
    report("                   cannot be a post-hoc refinement -- it has to run inside the loop wherever")
    report("                   melting is possible.")
    report("    So the practical scheme is both: sequential refinement everywhere for conformation and")
    report("    topology, and a persistent fine window only where melting or polymerase activity lives.")

    report("\n    THE ENGINEERING RISK, and it is one thing rather than many. Splitting a 10 kb bead into")
    report("    fifty 200 bp beads and merging them back must conserve LINKING NUMBER EXACTLY. Lk is the")
    report("    quantity the whole model exists to track, it is topological rather than energetic, and a")
    report("    small leak at a moving refinement boundary will not show up as an instability -- it will")
    report("    show up as slowly drifting supercoiling that looks like physics. Lk conservation across")
    report("    every refine and coarsen operation should be an assertion, not a hope.")
    report("    What is NOT a risk: the stiffness mapping. A persistence length is scale-free, so the")
    report("    discrete bending stiffness is just A/b at any level. Atomistic coarse-graining has no such")
    report("    luxury, and this is the one place the rod model is genuinely easier.")

    # ---- what the separation of timescales has to be for this to be legitimate --------------------
    report("\n  THE ASSUMPTION THAT CARRIES ALL OF IT, stated so it can be attacked.")
    report("    Quasi-static stepping is only valid if the structure re-equilibrates BETWEEN events. The")
    report(f"    slowest mode measured here is tau_max = {TAU_MAX:.0f} in the solver's units, and those units")
    report("    are not seconds -- gamma was never calibrated to water. THAT CALIBRATION IS THE ONE")
    report("    MISSING NUMBER, and it decides whether this whole scheme is legitimate or a fiction.")
    report("    It is a small measurement (fit one mode's relaxation against a known diffusion")
    report("    coefficient) and it should be done before any of the costs above are quoted as real.")

    report("\n  READING")
    report("  Atomistic chromosome: ruled out on memory, permanently.")
    report("  Coarse-grained + integration: possible in principle, decades of compute, ruled out in practice.")
    report("  Coarse-grained + EVENT stepping: hours to days on hardware we have, because the cost stops")
    report("  scaling with simulated TIME and starts scaling with the number of things that HAPPEN.")
    report("  Twist and polymerase fit that scheme directly. Replication fits it as a topology edit between")
    report("  solves. Denaturation does NOT -- it is a bond-breaking event the harmonic model cannot")
    report("  represent at any cost, and needs a base-pair free-energy layer instead of more compute.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "chromatin_cost_model",
               "measured": {"n_atom_nucleosome": N_ATOM_NUC, "nnz_nucleosome": NNZ_NUC,
                            "t_direct": T_DIRECT_NUC, "t_fine": T_FINE_NUC,
                            "tau_min": TAU_MIN, "tau_max": TAU_MAX},
               "atomistic": [{"scale": a, "atoms": b, "nnz": c, "bytes": d} for a, b, c, d in rows],
               "event_scenarios": scen,
               "assumptions": {"nnz_per_bead": NNZ_BEAD, "n_coarse": N_COARSE,
                               "iters_cold": ITERS_COLD, "iters_warm": ITERS_WARM,
                               "flops_per_core": FLOPS_PER_CORE, "dt_bd": DT_BD},
               "cores": CORES, "amr": amr, "sequential": seq, "log": log},
              open(OUT / "chromatin_cost_model.json", "w"), indent=2)
    report(f"\n  -> {OUT/'chromatin_cost_model.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
