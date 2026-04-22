"""Tests for the AtomUnit MD primitive + vesicle fission demo.

These are fast unit tests plus one ~seconds-scale integration test that
runs a tiny fission scenario. A slower full-scale fission is in
``scripts/run_fission_demo.py``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from cell_sim.atom_engine.atom_unit import AtomUnit, BondType
from cell_sim.atom_engine.element import Element, default_valence, pair_is_bondable
from cell_sim.atom_engine.fission_demo import FissionConfig, run_fission
from cell_sim.atom_engine.force_field import ForceFieldConfig, compute_forces
from cell_sim.atom_engine.integrator import (
    IntegratorConfig,
    SimState,
    current_temperature_K,
    run,
    step,
)
from cell_sim.atom_engine.fusion_demo import FusionConfig, run_fusion
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
