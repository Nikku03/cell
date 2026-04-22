"""Tests for the AtomUnit MD primitive + vesicle fission demo.

These are fast unit tests plus one ~seconds-scale integration test that
runs a tiny fission scenario. A slower full-scale fission is in
``scripts/run_fission_demo.py``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from cell_sim.atom_engine.atom_soup import SoupSpec, build_soup
from cell_sim.atom_engine.atom_unit import AtomUnit, BondType
from cell_sim.atom_engine.chemistry_demo import ChemistryConfig, run_chemistry
from cell_sim.atom_engine.molecule_builder import (
    LIBRARY,
    build_mixture,
    canonical_formula,
    classify_molecules,
)
from cell_sim.atom_engine.element import Element, default_valence, pair_is_bondable
from cell_sim.atom_engine.fission_demo import FissionConfig, run_fission
from cell_sim.atom_engine.force_field import (
    ForceFieldConfig,
    build_neighbor_list,
    compute_forces,
)
from cell_sim.atom_engine.integrator import (
    IntegratorConfig,
    SimState,
    current_temperature_K,
    run,
    step,
)
from cell_sim.atom_engine.fusion_demo import FusionConfig, run_fusion
from cell_sim.atom_engine.reaction_demo import (
    ReactionConfig,
    audit_stability,
    run_reactions,
)
from cell_sim.atom_engine.vesicle import (
    VesicleSpec,
    build_two_vesicles,
    build_vesicle,
    count_connected_components,
    equatorial_split_metric,
    intermixing_fraction,
    tagged_components,
    vesicle_com_separation,
)


# ---------- AtomUnit + Bond basics ------------------------------------


def test_atom_creation_sets_mass_and_history():
    a = AtomUnit.create(Element.C, position=(0.0, 0.0, 0.0))
    assert a.element is Element.C
    assert math.isclose(a.mass_da, 12.011, rel_tol=1e-3)
    assert a.valence_remaining == default_valence(Element.C) == 4
    assert len(a.history) == 1
    assert a.history[0].kind == "created"


def test_bond_formation_and_valence_accounting():
    c = AtomUnit.create(Element.C, position=(0.0, 0.0, 0.0))
    h = AtomUnit.create(Element.H, position=(0.11, 0.0, 0.0))
    b = c.form_bond(h, kind=BondType.COVALENT_SINGLE, t_ps=0.0,
                    equilibrium_length_nm=0.11,
                    spring_constant_kj_per_nm2=2.0e5)
    assert c.valence_remaining == 3
    assert h.valence_remaining == 0
    assert c.is_bonded_to(h)
    assert h.is_bonded_to(c)
    assert len(c.bonds) == len(h.bonds) == 1
    # Both atoms see the "bond_formed" event.
    assert any(e.kind == "bond_formed" for e in c.history)
    assert any(e.kind == "bond_formed" for e in h.history)
    assert b.order == 1


def test_bond_breaking_logs_event_and_restores_valence():
    c = AtomUnit.create(Element.C, position=(0.0, 0.0, 0.0))
    n = AtomUnit.create(Element.N, position=(0.15, 0.0, 0.0))
    b = c.form_bond(n, kind=BondType.COVALENT_SINGLE, t_ps=0.0,
                    equilibrium_length_nm=0.147,
                    spring_constant_kj_per_nm2=2.0e5)
    c.break_bond(b, t_ps=1.0, reason="test")
    assert c.valence_remaining == 4
    assert n.valence_remaining == 3
    assert b.death_time_ps == 1.0
    assert any(e.kind == "bond_broken" for e in c.history)


def test_cannot_bond_to_self_or_over_valence():
    c = AtomUnit.create(Element.C, position=(0.0, 0.0, 0.0))
    h1 = AtomUnit.create(Element.H, position=(0.11, 0.0, 0.0))
    h2 = AtomUnit.create(Element.H, position=(0.22, 0.0, 0.0))
    # H has valence 1 — once bonded to one C, can't bond another.
    h1.form_bond(c, kind=BondType.COVALENT_SINGLE, t_ps=0.0,
                 equilibrium_length_nm=0.11, spring_constant_kj_per_nm2=1e5)
    assert not h1.can_bond_to(h2, BondType.COVALENT_SINGLE)
    # Self-bond forbidden.
    assert not c.can_bond_to(c, BondType.COVALENT_SINGLE)


def test_pair_is_bondable_respects_allowed_set():
    assert pair_is_bondable(Element.C, Element.H)
    assert pair_is_bondable(Element.H, Element.C)  # symmetric
    assert not pair_is_bondable(Element.Na, Element.K)


# ---------- Force field -----------------------------------------------


def test_lj_force_at_sigma_is_attractive_but_bounded():
    # Two carbons right at their LJ sigma → force magnitude equals 24*eps/r
    # (since sr=1, sr6=1, sr12=1, derivative = 24*eps*(2-1)/r = 24*eps/r).
    a = AtomUnit.create(Element.C, position=(0.0, 0.0, 0.0))
    b = AtomUnit.create(Element.C, position=(0.34, 0.0, 0.0))
    f = compute_forces([a, b], [], t_ps=0.0, cfg=ForceFieldConfig())
    # Net force on each atom should be opposite, equal magnitude.
    assert np.allclose(f[0], -f[1])
    # At r=sigma, d/dr of LJ = +24 eps/r → force magnitude 24*0.457/0.34
    expected = 24.0 * 0.457 / 0.34
    assert math.isclose(np.linalg.norm(f[0]), expected, rel_tol=1e-3)


def test_harmonic_bond_pulls_stretched_atoms_inward():
    a = AtomUnit.create(Element.C, position=(0.0, 0.0, 0.0))
    b = AtomUnit.create(Element.C, position=(0.50, 0.0, 0.0))   # stretched
    bond = a.form_bond(b, kind=BondType.COVALENT_SINGLE, t_ps=0.0,
                       equilibrium_length_nm=0.15,
                       spring_constant_kj_per_nm2=5.0e4)
    f = compute_forces([a, b], [bond], t_ps=0.0, cfg=ForceFieldConfig())
    # b is at +x → force on b should point in -x (toward a)
    assert f[1, 0] < 0.0
    assert f[0, 0] > 0.0
    # Magnitudes equal and opposite
    assert np.allclose(f[0], -f[1])


# ---------- Integrator energy & temperature ---------------------------


def test_integrator_maintains_temperature_roughly():
    rng = np.random.default_rng(0)
    atoms = []
    for i in range(40):
        pos = tuple(rng.uniform(-1.0, 1.0, size=3))
        vel = tuple(rng.normal(0, 0.5, size=3))
        atoms.append(AtomUnit.create(Element.C, position=pos, velocity=vel))
    state = SimState(atoms=atoms, bonds=[])
    ff = ForceFieldConfig(lj_cutoff_nm=1.5)
    ic = IntegratorConfig(dt_ps=0.002, target_temperature_K=300.0,
                          thermostat_tau_ps=0.2)
    run(state, n_steps=500, ff_cfg=ff, int_cfg=ic)
    T = current_temperature_K(state.atoms)
    # Berendsen is not an NVT sampler but it does pull T toward the target.
    assert 150.0 < T < 500.0


def test_bond_breaks_when_overstretched():
    # Put two bonded atoms far apart; the integrator should flag the break.
    a = AtomUnit.create(Element.C, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))
    b = AtomUnit.create(Element.C, position=(0.40, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))
    # Set r well above 1.5 * r0 = 1.5 * 0.15 = 0.225 nm
    bond = a.form_bond(b, kind=BondType.COVALENT_SINGLE, t_ps=0.0,
                       equilibrium_length_nm=0.15,
                       spring_constant_kj_per_nm2=1e3)  # weak so it doesn't explode
    state = SimState(atoms=[a, b], bonds=[bond])
    ff = ForceFieldConfig(lj_cutoff_nm=0.0)   # disable LJ
    ic = IntegratorConfig(dt_ps=0.001, thermostat_tau_ps=1e9,
                          target_temperature_K=0.0, bond_break_fraction=1.5)
    step(state, ff, ic)
    assert bond.death_time_ps is not None
    assert state.events_bonds_broken == 1
    assert a.valence_remaining == 4


# ---------- Vesicle geometry ------------------------------------------


def test_vesicle_has_bilayer_structure():
    spec = VesicleSpec(n_per_leaflet=40, radius_nm=3.0,
                       bilayer_thickness_nm=1.0, seed=1)
    atoms, bonds = build_vesicle(spec)
    assert len(atoms) == 4 * spec.n_per_leaflet   # 2 atoms/lipid × 2 leaflets
    assert len(bonds) == 2 * spec.n_per_leaflet
    # Every atom is COARSE_HEAD or COARSE_TAIL
    heads = [a for a in atoms if a.element is Element.COARSE_HEAD]
    tails = [a for a in atoms if a.element is Element.COARSE_TAIL]
    assert len(heads) == len(tails) == 2 * spec.n_per_leaflet
    # Heads at either R_out or R_in; tails at R_mid.
    head_r = np.array([np.linalg.norm(a.position) for a in heads])
    tail_r = np.array([np.linalg.norm(a.position) for a in tails])
    # Two distinct head-radius clusters
    assert head_r.max() > head_r.min() + 0.5
    # Tails concentrated between the two head shells
    assert tail_r.mean() < head_r.max()
    assert tail_r.mean() > head_r.min()


def test_intact_vesicle_is_one_component():
    # Need enough lipids that neighbor tails fall inside the link cutoff
    # (0.7 nm). For R=3, thickness=1, surface area / n ~= sqrt(78.5/n).
    atoms, bonds = build_vesicle(VesicleSpec(n_per_leaflet=150, seed=2))
    assert count_connected_components(atoms, bonds) == 1


def test_equatorial_metric_is_near_one_for_uniform_vesicle():
    atoms, bonds = build_vesicle(VesicleSpec(n_per_leaflet=120, seed=3))
    m = equatorial_split_metric(atoms, axis=2)
    # Fibonacci sphere is close to uniform → metric should be in (0.7, 1.3).
    assert 0.6 < m < 1.4


# ---------- Fission (slow-ish integration test) -----------------------


def test_two_vesicles_are_initially_two_components():
    spec = VesicleSpec(n_per_leaflet=80, radius_nm=2.0,
                       bilayer_thickness_nm=0.9, seed=7)
    atoms, bonds = build_two_vesicles(spec, z_offset_nm=3.5)
    assert len(atoms) == 2 * 4 * spec.n_per_leaflet
    # Each vesicle tagged by parent_molecule
    tags = {a.parent_molecule for a in atoms}
    assert tags == {"vesicle_upper", "vesicle_lower"}
    # Pre-fusion COM separation should be ~2 * z_offset = 7 nm
    sep = vesicle_com_separation(atoms, axis=2)
    assert 6.0 < sep < 8.0
    # Two tagged components before any MD; intermix is 0.
    assert tagged_components(atoms) == 2
    assert intermixing_fraction(atoms) == 0.0


def test_tagged_components_merges_on_overlap():
    """Two tagged populations within cutoff merge into one tagged component."""
    a = AtomUnit.create(Element.COARSE_HEAD, position=(0.0, 0.0, 0.0),
                        parent_molecule="vesicle_upper")
    b = AtomUnit.create(Element.COARSE_HEAD, position=(0.5, 0.0, 0.0),
                        parent_molecule="vesicle_lower")
    # With cutoff 0.8 nm, a-b separation of 0.5 nm triggers a merge.
    assert tagged_components([a, b], cutoff_nm=0.8) == 1
    assert intermixing_fraction([a, b], cutoff_nm=0.8) == 1.0
    # With a tight 0.3 nm cutoff, they stay separate.
    assert tagged_components([a, b], cutoff_nm=0.3) == 2


@pytest.mark.slow
def test_fission_runs_without_blowing_up():
    """Short fission run — we don't require a full pinch-off here (that's
    in the runnable demo script); we just require the MD stays numerically
    sane."""
    cfg = FissionConfig(
        vesicle=VesicleSpec(n_per_leaflet=50, radius_nm=2.5,
                            bilayer_thickness_nm=0.9, seed=4),
        equilibration_steps=200,
        production_steps=1000,
        report_every=500,
        constriction_ramp_ps=1.0,
        constriction_k_kj_per_nm2=500.0,
    )
    state, result = run_fission(cfg)
    # No NaNs, temperature stays finite
    T = current_temperature_K(state.atoms)
    assert math.isfinite(T)
    assert 0.0 < T < 2000.0
    # Metrics recorded
    assert len(result.neck_fraction) > 0
    # The atom count is preserved
    assert result.n_atoms == 4 * cfg.vesicle.n_per_leaflet


def test_neighbor_list_matches_full_on2():
    """The spatial-hash neighbor list must return the same pair set (after
    distance filtering) as the brute-force upper triangle."""
    rng = np.random.default_rng(0)
    n = 300
    pos = rng.uniform(-2.0, 2.0, size=(n, 3))
    cutoff = 0.9
    iu, ju = build_neighbor_list(pos, cutoff)
    # Brute force
    all_iu, all_ju = np.triu_indices(n, k=1)
    d = pos[all_ju] - pos[all_iu]
    r2 = (d * d).sum(-1)
    mask = r2 < cutoff * cutoff
    expected = {(int(a), int(b)) for a, b in zip(all_iu[mask], all_ju[mask])}
    got = {(int(a), int(b)) for a, b in zip(iu, ju)}
    assert got == expected


def test_neighbor_list_forces_match_full_forces():
    """The LJ forces produced via the neighbor-list path must be numerically
    close to the full O(N^2) path. Use a random soup and no bonds."""
    rng = np.random.default_rng(2)
    atoms = []
    for i in range(120):
        p = tuple(rng.uniform(-1.5, 1.5, size=3))
        atoms.append(AtomUnit.create(Element.C, position=p,
                                     velocity=(0.0, 0.0, 0.0)))
    cfg_full = ForceFieldConfig(lj_cutoff_nm=0.9, use_neighbor_list=False)
    cfg_nl = ForceFieldConfig(lj_cutoff_nm=0.9, use_neighbor_list=True,
                              neighbor_skin_nm=0.0)
    f_full = compute_forces(atoms, [], t_ps=0.0, cfg=cfg_full)
    f_nl = compute_forces(atoms, [], t_ps=0.0, cfg=cfg_nl)
    assert np.allclose(f_full, f_nl, atol=1e-8)


def test_rust_lj_matches_numpy_lj():
    """Regardless of whether cell_sim_rust is loaded, the forces from
    compute_forces must match the pure-NumPy reference (which we force
    by setting ``_HAS_RUST_LJ=False`` on the module at import time in
    this test)."""
    import cell_sim.atom_engine.force_field as ff_mod
    rng = np.random.default_rng(7)
    atoms = []
    for _ in range(200):
        p = tuple(rng.uniform(-2.0, 2.0, size=3))
        atoms.append(AtomUnit.create(Element.C, position=p,
                                     velocity=(0.0, 0.0, 0.0)))
    cfg = ForceFieldConfig(lj_cutoff_nm=0.9, use_neighbor_list=True,
                           neighbor_skin_nm=0.0)
    original = ff_mod._HAS_RUST_LJ
    try:
        ff_mod._HAS_RUST_LJ = False
        f_np = compute_forces(atoms, [], t_ps=0.0, cfg=cfg)
        ff_mod._HAS_RUST_LJ = original
        f_rust = compute_forces(atoms, [], t_ps=0.0, cfg=cfg)
    finally:
        ff_mod._HAS_RUST_LJ = original
    assert np.allclose(f_np, f_rust, atol=1e-7)


def test_molecule_library_templates_have_valid_bonds():
    """Every template in LIBRARY must reference only in-range atom indices
    and use only bondable element pairs."""
    from cell_sim.atom_engine.element import pair_is_bondable
    for name, tmpl in LIBRARY.items():
        n = len(tmpl.atoms)
        assert n >= 2, f"{name}: must have at least 2 atoms"
        for i, j, _kind, _r0, _k in tmpl.bonds:
            assert 0 <= i < n and 0 <= j < n, f"{name}: bond OOB"
            assert pair_is_bondable(tmpl.atoms[i][0], tmpl.atoms[j][0]), \
                f"{name}: non-bondable pair {tmpl.atoms[i][0]}-{tmpl.atoms[j][0]}"


def test_build_mixture_preserves_composition():
    atoms, bonds = build_mixture(
        {"H2O": 5, "CH4": 3, "NH3": 2}, radius_nm=2.0, seed=1,
    )
    formulas = classify_molecules(atoms)
    assert formulas.get("H2O", 0) == 5
    assert formulas.get("CH4", 0) == 3
    assert formulas.get("H3N", 0) == 2   # Hill system: N before H alphabetically? no:
    # Actually Hill ordering puts C, H first, then rest alphabetically.
    # For NH3 (no C, no H at start of sort — H is first by Hill if no C):
    # NH3 → "H3N" via Hill.
    # Double-check: we expect "H3N".
    # Total atom count: 5*3 + 3*5 + 2*4 = 15 + 15 + 8 = 38
    assert len(atoms) == 38


def test_canonical_formula_uses_hill_order():
    # Build a small methanol-like fragment manually: C, H, H, H, O, H
    elements = [0, 1, 1, 1, 4, 1]  # indices into a list of atoms with these elems
    from cell_sim.atom_engine.element import Element
    atoms = [
        AtomUnit.create(Element.C, (0, 0, 0)),
        AtomUnit.create(Element.H, (0, 0, 0)),
        AtomUnit.create(Element.H, (0, 0, 0)),
        AtomUnit.create(Element.H, (0, 0, 0)),
        AtomUnit.create(Element.O, (0, 0, 0)),
        AtomUnit.create(Element.H, (0, 0, 0)),
    ]
    assert canonical_formula(list(range(6)), atoms) == "CH4O"


def test_atom_soup_respects_composition_and_temperature():
    spec = SoupSpec(
        composition={Element.H: 10, Element.C: 4, Element.O: 2},
        radius_nm=1.0, temperature_K=500.0, seed=5,
    )
    atoms = build_soup(spec)
    assert len(atoms) == 16
    counts = {}
    for a in atoms:
        counts[a.element] = counts.get(a.element, 0) + 1
    assert counts == {Element.H: 10, Element.C: 4, Element.O: 2}
    # No pre-existing bonds.
    assert all(len(a.bonds) == 0 for a in atoms)
    # Atoms inside the sphere.
    for a in atoms:
        r2 = sum(x * x for x in a.position)
        assert r2 <= spec.radius_nm ** 2 + 1e-6


def test_audit_stability_catches_no_issues_on_fresh_soup():
    atoms = build_soup(SoupSpec(composition={Element.C: 5, Element.H: 10}, seed=1))
    audit = audit_stability(atoms)
    assert audit["valence_violations"] == 0
    assert audit["duplicate_bonds"] == 0
    assert audit["illegal_pairs"] == 0


@pytest.mark.slow
def test_reactions_preserve_conservation_and_valence():
    """Short reaction run must preserve atom count, respect valence, only
    form bondable pairs, and produce at least some bonds at 2000 K."""
    cfg = ReactionConfig(
        soup=SoupSpec(
            composition={Element.H: 20, Element.C: 6, Element.O: 3, Element.N: 2},
            radius_nm=1.0, temperature_K=2000.0, seed=7,
        ),
        dt_ps=0.001,
        target_temperature_K=2000.0,
        steps=3000,
        report_every=3000,
    )
    state, result = run_reactions(cfg)
    # Atom count preserved.
    assert len(state.atoms) == result.n_atoms == 31
    # Total mass preserved (sanity — no atom destruction).
    total_mass = sum(a.mass_da for a in state.atoms)
    assert math.isclose(total_mass, 20 * 1.008 + 6 * 12.011 + 3 * 15.999 + 2 * 14.007,
                        rel_tol=1e-9)
    # Stability audit passes.
    assert result.valence_violations == 0
    assert result.duplicate_bonds == 0
    assert result.illegal_pairs == 0
    # At 2000 K something reactive should happen.
    assert result.total_bonds_formed > 0
    # Temperature finite and bounded.
    assert math.isfinite(result.final_temperature_K)
    assert 0.0 < result.final_temperature_K < 10000.0


@pytest.mark.slow
def test_reactions_are_cold_enough_to_be_quiet_at_low_T():
    """At 200 K atoms barely approach bond_form_distance, so reaction
    rate should be far lower than at high T."""
    low_cfg = ReactionConfig(
        soup=SoupSpec(
            composition={Element.H: 20, Element.C: 6},
            radius_nm=1.0, temperature_K=200.0, seed=3,
        ),
        dt_ps=0.001,
        target_temperature_K=200.0,
        steps=2000,
        report_every=2000,
    )
    hot_cfg = ReactionConfig(
        soup=SoupSpec(
            composition={Element.H: 20, Element.C: 6},
            radius_nm=1.0, temperature_K=3000.0, seed=3,
        ),
        dt_ps=0.001,
        target_temperature_K=3000.0,
        steps=2000,
        report_every=2000,
    )
    _, low = run_reactions(low_cfg)
    _, hot = run_reactions(hot_cfg)
    # Hot system reacts more (formed + broken events).
    assert (hot.total_bonds_formed + hot.total_bonds_broken) > (
        low.total_bonds_formed + low.total_bonds_broken
    )
    # Both runs still pass the stability audit.
    for r in (low, hot):
        assert r.valence_violations == 0
        assert r.duplicate_bonds == 0
        assert r.illegal_pairs == 0


@pytest.mark.slow
def test_fusion_field_only_runs_without_blowing_up():
    """Short fusion run. We require that the two vesicles either reach
    contact OR stay numerically stable; we do not demand full merge in
    this quick test (that's the runnable demo)."""
    spec = VesicleSpec(n_per_leaflet=50, radius_nm=1.8,
                       bilayer_thickness_nm=0.8, seed=11)
    cfg = FusionConfig(
        vesicle=spec,
        z_offset_nm=2.6,
        equilibration_steps=200,
        production_steps=1500,
        report_every=500,
        attractor_strength_kj_per_nm=5.0,
        attractor_ramp_ps=2.0,
    )
    state, result = run_fusion(cfg)
    T = current_temperature_K(state.atoms)
    assert math.isfinite(T)
    assert 0.0 < T < 2000.0
    assert result.n_atoms == 2 * 4 * spec.n_per_leaflet
    assert result.initial_tagged_components == 2
    # COM separation should decrease monotonically on average
    if len(result.com_separation_nm) >= 2:
        assert result.com_separation_nm[-1] <= result.com_separation_nm[0] + 0.2
