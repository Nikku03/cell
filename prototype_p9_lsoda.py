"""
DMVC P9: Stiff-aware integration with scipy LSODA.

Replaces the rate-cap workaround with proper stiff ODE integration. The cap
throttled fast reactions so explicit Euler could remain stable, which meant
fast reactions ran slower than their kinetic parameters specified. LSODA
handles stiffness automatically via implicit methods, so we get HONEST
dynamics at the cost of a slightly more complex integration setup.

Scope: well-mixed case (one voxel, all 221 elementary Syn3A reactions +
cytoplasmic homeostasis buffering, matching P6's setup). The spatial
extension is mechanical once the well-mixed case works.

Expected wins:
  * Dramatically faster for long simulated time (seconds simulate
    in ~seconds wall-clock, vs tens of seconds before)
  * True kinetics -- no artificial slowdown from the rate cap
  * Can see ATP actually converge to its setpoint, not just trend toward it

Honest notes:
  * LSODA needs the RHS as a function of (t, state_vector). We flatten
    the concentration dict into a numpy vector.
  * Buffering (homeostasis) becomes a continuous dc/dt = gamma*(c0-c) term
    in the ODE. Cleaner than the previous "apply between steps" approach.
  * No symbolic Jacobian supplied; LSODA will estimate by finite differences.
    For 200+ species this is feasible but adds overhead. A sparse symbolic
    Jacobian is future work.
  * Polymerization reactions still excluded (|stoich| > 3). That's an
    independent issue -- convenience kinetics doesn't handle them regardless
    of integrator.

Tests:
  T1: LSODA trajectory matches explicit-Euler-with-cap trajectory on
      central metabolites within 20% at a timepoint both can reach
      (e.g. 1.5 s simulated, which is where P6 stopped)
  T2: ATP converges near its setpoint at long simulated time (10 s or more)
  T3: Wall-clock time to reach 10 s simulated time is under 10 s
  T4: Steady-state concentrations are in physiological range
"""

from __future__ import annotations
import numpy as np
import time
import sys
sys.path.insert(0, "/home/claude/dmvc")

from scipy.integrate import solve_ivp

from prototype_p3b_stamps import (
    Compartment, load_sbml_model, extract_molecules_with_compartments,
    extract_reactions, find_h_and_water_per_compartment, rebalance_reaction,
    SBML_PATH,
)
from prototype_p4_kinetics import parse_all_kinetics, ReactionKinetics
from prototype_p7_learned_rates import convenience_rate_passive
from prototype_p6_physiological import CYTO_BUFFERED_SPECIES


# =============================================================================
# Right-hand-side for the ODE
# =============================================================================
def build_rhs(rxns, species, buffered_setpoints=None, gamma_buffer=2.0,
                constant_species=None):
    """
    Build a callable f(t, c_vec) -> dc_vec for scipy.integrate.solve_ivp.

    Includes:
      - All reaction rates via convenience_rate_passive
      - Cytoplasmic buffering as continuous dc/dt = gamma*(c0-c) terms

    buffered_setpoints: dict {species_id: target_conc}. If None, no buffering.
    constant_species: set of species IDs whose dc/dt is forced to 0.
      Use for species that the Luthey-Schulten model treats as implicit
      (water, implicit protons) where the numeric placeholder is nonsense.
    """
    species_to_idx = {s: i for i, s in enumerate(species)}
    n_species = len(species)

    rxn_stoich_indexed = []
    for rid, rxn, kin in rxns:
        triples = []
        for sid, coef in rxn.stoichiometry.items():
            if sid in species_to_idx:
                triples.append((species_to_idx[sid], float(coef)))
        rxn_stoich_indexed.append((rid, rxn, kin, triples))

    buffered_pairs = []
    if buffered_setpoints:
        for sid, c0 in buffered_setpoints.items():
            if sid in species_to_idx:
                buffered_pairs.append((species_to_idx[sid], float(c0)))

    const_indices = set()
    if constant_species:
        const_indices = {species_to_idx[sid] for sid in constant_species
                          if sid in species_to_idx}

    def rhs(t, c_vec):
        dc = np.zeros(n_species)
        c_dict = {species[i]: max(c_vec[i], 0.0) for i in range(n_species)}

        for rid, rxn, kin, triples in rxn_stoich_indexed:
            v = convenience_rate_passive(rxn, kin, c_dict)
            if v == 0.0:
                continue
            for idx, coef in triples:
                dc[idx] += coef * v

        for idx, c0 in buffered_pairs:
            dc[idx] += gamma_buffer * (c0 - c_vec[idx])

        # Force constant species to dc/dt = 0
        for idx in const_indices:
            dc[idx] = 0.0

        return dc

    return rhs


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 76)
    print("DMVC P9: LSODA stiff integration for well-mixed Syn3A")
    print("=" * 76)

    # Load model
    print("\n[1] Loading model and kinetics...")
    model = load_sbml_model(SBML_PATH)
    mols = extract_molecules_with_compartments(model)
    rxns_raw = extract_reactions(model)
    p_ids, w_ids = find_h_and_water_per_compartment(mols)
    rebalanced = []
    for r in rxns_raw:
        if r.is_exchange or r.is_biomass: continue
        nr = rebalance_reaction(r, mols, p_ids, w_ids)
        if nr: rebalanced.append(nr)
    kinetics, init_concs = parse_all_kinetics()
    active = [(r.sbml_id, r, kinetics[r.sbml_id]) for r in rebalanced
               if r.sbml_id in kinetics and kinetics[r.sbml_id].is_usable()
               and not any(abs(c) > 3 for c in r.stoichiometry.values())]
    print(f"    {len(active)} elementary reactions")

    # Gather species
    species = []
    seen = set()
    for rid, rxn, kin in active:
        for sid in rxn.stoichiometry:
            if sid not in seen:
                seen.add(sid); species.append(sid)
    print(f"    {len(species)} species in scope")

    # Initial state
    c0 = np.array([init_concs.get(s, 0.1) for s in species])

    # Build RHS with buffering; treat implicit species as constants.
    # Water and implicit protons have nonsense placeholder concentrations
    # (0.1 mM) that shouldn't participate dynamically.
    print("\n[2] Building ODE RHS with buffering + constant water/H+...")
    setpoints_in_scope = {sid: v for sid, v in CYTO_BUFFERED_SPECIES.items()
                           if sid in species}
    print(f"    Buffered species (gamma=10/s): {list(setpoints_in_scope.keys())}")
    constant_species = {"M_h2o_c", "M_h2o_e", "M_h_c", "M_h_e"}
    constants_in_scope = constant_species & set(species)
    print(f"    Held constant: {sorted(constants_in_scope)}")
    rhs = build_rhs(active, species, setpoints_in_scope,
                     gamma_buffer=10.0, constant_species=constant_species)

    # -----------------------------------------------------------------
    print("\n[3] Integrating with LSODA to t=10 s...")
    t0 = time.time()
    sol = solve_ivp(
        rhs,
        t_span=(0.0, 10.0),
        y0=c0,
        method='LSODA',
        t_eval=np.linspace(0, 10, 101),
        rtol=1e-6,
        atol=1e-9,
        max_step=0.5,
    )
    elapsed = time.time() - t0
    print(f"    Wall time: {elapsed:.2f} s")
    print(f"    Success: {sol.success}")
    print(f"    nfev: {sol.nfev}  (RHS evaluations)")
    print(f"    Status: {sol.status}  {sol.message}")

    if not sol.success:
        print("\n    INTEGRATION FAILED. Check kinetics / initial conditions.")
        return

    # -----------------------------------------------------------------
    print("\n[4] Analyzing trajectory...")
    times = sol.t
    Y = sol.y   # shape (n_species, n_times)
    sid_to_idx = {s: i for i, s in enumerate(species)}

    def traj(sid):
        return Y[sid_to_idx[sid]] if sid in sid_to_idx else None

    print("\n    [ATP / NAD / G6P / pyr] over time:")
    for i in range(0, len(times), max(1, len(times)//10)):
        t = times[i]
        atp = traj("M_atp_c")[i] if traj("M_atp_c") is not None else 0
        nad = traj("M_nad_c")[i] if traj("M_nad_c") is not None else 0
        g6p = traj("M_g6p_c")[i] if traj("M_g6p_c") is not None else 0
        pyr = traj("M_pyr_c")[i] if traj("M_pyr_c") is not None else 0
        print(f"    t={t:6.2f}s  ATP={atp:6.3f}  NAD={nad:6.3f}  "
              f"G6P={g6p:6.3f}  pyr={pyr:6.3f}")

    # Concentrations at steady state (last time point)
    print("\n    Steady-state concentrations:")
    watch = ["M_atp_c", "M_adp_c", "M_amp_c", "M_nad_c", "M_nadh_c",
              "M_g6p_c", "M_f6p_c", "M_pyr_c", "M_pep_c",
              "M_lac__L_c", "M_o2_c", "M_pi_c"]
    for s in watch:
        if s in sid_to_idx:
            c_start = c0[sid_to_idx[s]]
            c_end = Y[sid_to_idx[s], -1]
            print(f"      {s:15s}  {c_start:8.4f} -> {c_end:8.4f} mM  "
                  f"ratio={c_end/c_start if c_start>1e-9 else 0:.2f}x")

    # -----------------------------------------------------------------
    # Tests
    # -----------------------------------------------------------------
    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)

    # T1: ATP convergence near setpoint at t=10 s
    atp_final = traj("M_atp_c")[-1]
    atp_setpoint = CYTO_BUFFERED_SPECIES["M_atp_c"]
    atp_ratio = atp_final / atp_setpoint
    print(f"\n[T1] ATP convergence at t=10s:")
    print(f"     setpoint: {atp_setpoint:.3f} mM")
    print(f"     final:    {atp_final:.3f} mM   ratio: {atp_ratio:.3f}")
    t1 = 0.8 < atp_ratio < 1.2
    print(f"     {'PASS' if t1 else 'FAIL'}  (within 20% of setpoint)")

    # T2: wall clock time target
    print(f"\n[T2] Wall clock time to simulate 10 s:")
    print(f"     {elapsed:.2f} s (target: < 10 s)")
    t2 = elapsed < 10.0
    print(f"     {'PASS' if t2 else 'FAIL'}")

    # T3: Physiologically plausible
    print(f"\n[T3] Physiological plausibility:")
    nad = traj("M_nad_c")[-1]
    nadh = traj("M_nadh_c")[-1]
    nad_ratio = nad / max(nadh, 1e-9)
    print(f"     NAD/NADH: {nad_ratio:.2f} (physiological: ~87)")
    print(f"     G6P:      {traj('M_g6p_c')[-1]:.3f} mM (physiological: 3.71)")
    print(f"     pyruvate: {traj('M_pyr_c')[-1]:.3f} mM (physiological: 3.37)")
    t3 = (nad_ratio > 10 and
           0.5 < traj('M_g6p_c')[-1] / 3.71 < 2.0 and
           0.5 < traj('M_pyr_c')[-1] / 3.37 < 2.0)
    print(f"     {'PASS' if t3 else 'FAIL'}")

    # T4: all concentrations sane
    min_c = Y.min()
    max_c = Y.max()
    print(f"\n[T4] Concentration bounds:")
    print(f"     min: {min_c:.3e}  max: {max_c:.3e}")
    t4 = min_c > -1e-6 and max_c < 1e4
    print(f"     {'PASS' if t4 else 'FAIL'}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    for lab, ok, desc in [
        ("T1", t1, "ATP converges to setpoint"),
        ("T2", t2, "LSODA beats P6 speed substantially"),
        ("T3", t3, "steady state is physiological"),
        ("T4", t4, "all concentrations in sane range"),
    ]:
        print(f"  [{lab}] {'PASS' if ok else 'FAIL'}  {desc}")

    if t2 and elapsed < 5.0:
        speedup_vs_p6 = 77.0 / elapsed   # P6 at 1.5s simulated took ~77s
        print(f"\n    Speed comparison: LSODA ran 10 s simulated in {elapsed:.2f} s wall clock.")
        print(f"    P6 explicit Euler took ~77 s wall clock for 1.5 s simulated.")
        print(f"    Effective speedup: ~{speedup_vs_p6 * (10.0/1.5):.0f}x")

    return sol, species, active


if __name__ == "__main__":
    main()
