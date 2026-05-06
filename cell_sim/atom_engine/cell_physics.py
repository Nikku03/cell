"""Cell-scale physics engine — minimal, honest, ours.

Treats every molecule (or complex) as one particle. Integrates Langevin
dynamics in 3D inside a sphere with these forces:

    1. Soft excluded volume (cell-scale Lennard-Jones, sigma ~ molecule size)
    2. FENE bonds between consecutive beads of a polymer chain
    3. Harmonic restraints between members of a multi-protein complex
    4. Soft spherical wall keeping particles inside the cell
    5. Reactive proximity events: when two reactants are within
       ``reaction_radius_nm`` for one timestep, the reaction fires with
       probability ``rate * dt`` (Gillespie-style local firing)

What this is
------------
A real physics engine. It moves particles according to actual forces and
fires reactions based on actual proximity. It produces trajectories we
can train ``CellGNN`` against (one of the goals: bootstrap a learned
simulator from physics-based training data).

What this is NOT
----------------
A complete cell simulator. It does **not** do:
- ribosome biogenesis (Earnest, Lai, Chen 2015 — 145 assembly intermediates)
- replication forks / DNA replication kinetics
- membrane growth or division
- lipid composition or curvature dynamics

These are deliberate v1 exclusions, not papered-over features. Adding any
of them requires its own module and its own physics. This file gets us a
working *partial* 4D cell sim — diffusion + reactions + polymer + complexes
inside a fixed sphere — that's enough to generate training data for the
GNN and validate the inference path end-to-end.

Reuse from atom_engine
----------------------
- Langevin update math (Bussi-Parrinello form) — same equations.
- Soft-spherical-wall idea — re-implemented for cell scale.

We do NOT reuse atom_engine's force_field.compute_forces because:
- its LJ parameters are atomic (sigma ~ 0.3 nm); cells need ~5 nm
- its data model (AtomUnit, Bond, Element) is chemistry-coupled
- its dt is 1-2 fs; cells need 1 ns to 1 μs

Reusing those forces here would mean reskinning everything. Cleaner to
write the cell-scale forces directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
@dataclass
class CellPhysicsConfig:
    """Cell-scale particle dynamics is OVERDAMPED — inertia matters for a
    few ps and is dwarfed by friction and noise after that. We use Brownian
    dynamics (Smoldyn-style):
        x_{t+dt} = x_t + D · F · dt / kT  +  sqrt(2 D dt) · N(0, 1)
    Per-species diffusion D (nm²/ns) is the primary parameter; mass is
    informational and not used for integration.
    """
    # Geometry
    cell_radius_nm: float = 200.0           # syn3A-sized
    wall_k_kj_per_nm2: float = 1.0e3        # soft-wall spring strength
    wall_thickness_nm: float = 10.0          # wall takes effect within this margin

    # Time + temperature
    dt_ns: float = 0.01                      # integration step (ns); 10 ps
    kT_kj_per_mol: float = 2.49              # ~300 K

    # Excluded volume (cell-scale soft LJ)
    lj_cutoff_factor: float = 2.5            # cutoff = factor * sigma
    lj_default_sigma_nm: float = 5.0         # protein-sized fallback
    lj_default_epsilon_kj_per_mol: float = 1.0

    # FENE chain bonds (chromosome polymer)
    fene_k_kj_per_nm2: float = 30.0
    fene_r0_nm: float = 1.5                  # equilibrium chain spacing
    fene_r_max_factor: float = 2.0           # max extension = factor * r0
    chain_bend_k_kj_per_rad2: float = 5.0    # 3-body bending stiffness

    # Complex restraints (e.g. ribosome subunits)
    complex_k_kj_per_nm2: float = 50.0
    complex_r0_nm: float = 5.0

    # Reactions
    reaction_radius_nm: float = 1.0          # default proximity for firing
    max_force_kj_per_nm: float = 5.0e3       # safety cap

    # Pair-list rebuild cadence (every N steps)
    pair_rebuild_every: int = 10

    # RNG
    seed: int = 42


# ---------------------------------------------------------------------------
@dataclass
class SpeciesEntry:
    name: str
    sigma_nm: float                          # LJ sigma (effective radius)
    epsilon_kj_per_mol: float                # LJ well depth
    mass_da: float                           # informational
    diffusion_nm2_per_ns: float = 0.1        # PRIMARY dynamics parameter
                                             # ATP ~ 0.3, protein ~ 0.005,
                                             # complex ~ 0.0005 (nm²/ns)


# ---------------------------------------------------------------------------
@dataclass
class Reaction:
    """A_i + A_j -> products. ``rate_per_ns`` is the per-encounter probability
    rate (probability of firing per ns when a qualifying pair is in proximity).
    Products are species ids of the new particles spawned at the reactant
    centroid (reactants are removed). Empty products list = degradation.
    """
    reactant_a: int
    reactant_b: int
    products: tuple[int, ...]
    rate_per_ns: float
    reaction_radius_nm: float = 1.0


# ---------------------------------------------------------------------------
@dataclass
class CellState:
    """Mutable state of the cell. Arrays grow / shrink as reactions fire."""
    pos: np.ndarray                          # (N, 3) nm
    species_id: np.ndarray                   # (N,) int32
    mass: np.ndarray                         # (N,) Da (informational)
    sigma: np.ndarray                        # (N,) nm
    epsilon: np.ndarray                      # (N,) kJ/mol
    diffusion: np.ndarray                    # (N,) nm²/ns — Brownian dynamics primary
    # Polymer chain — list of arrays of indices in chain order. Each chain
    # contributes (len-1) FENE bonds and (len-2) bending terms.
    chains: list[np.ndarray] = field(default_factory=list)
    # Complex restraints — list of (i, j) index pairs.
    complex_pairs: np.ndarray = field(default_factory=lambda:
                                      np.empty((0, 2), dtype=np.int64))
    t_ns: float = 0.0
    step: int = 0
    # Cumulative event counts, keyed by reaction index.
    rxn_counts: dict = field(default_factory=dict)
    # Cached pair list for excluded volume (rebuilt every config.pair_rebuild_every)
    _pair_iu: Optional[np.ndarray] = None
    _pair_ju: Optional[np.ndarray] = None
    _pair_built_step: int = -1

    def n(self) -> int:
        return self.pos.shape[0]


# ---------------------------------------------------------------------------
def initial_state(species_table: list[SpeciesEntry],
                  composition: dict[int, int],
                  config: CellPhysicsConfig,
                  chains: Optional[list[list[int]]] = None,
                  complex_pairs: Optional[list[tuple[int, int]]] = None,
                  ) -> CellState:
    """Build a randomly-placed initial state inside the cell sphere.

    composition: {species_id: count}
    chains: optional list of chains, each a list of length-N giving species
            ids for the N particles forming that chain. Their positions are
            laid out as a random walk seeded near origin.
    complex_pairs: optional list of (i, j) index pairs to restrain.
    """
    rng = np.random.default_rng(config.seed)
    n_species = len(species_table)
    R = config.cell_radius_nm

    # ---- 1. unbonded particles from composition ----
    pos_list, sid_list, mass_list, sig_list, eps_list, diff_list = (
        [], [], [], [], [], [])
    for sid, count in composition.items():
        for _ in range(count):
            # uniform-in-sphere via rejection
            while True:
                v = rng.uniform(-R, R, size=3)
                if v @ v <= R * R:
                    break
            pos_list.append(v)
            sp = species_table[sid]
            sid_list.append(sid)
            mass_list.append(sp.mass_da)
            sig_list.append(sp.sigma_nm)
            eps_list.append(sp.epsilon_kj_per_mol)
            diff_list.append(sp.diffusion_nm2_per_ns)

    # ---- 2. polymer chains ----
    chain_arrays = []
    if chains:
        for chain_species in chains:
            n_c = len(chain_species)
            # Self-avoiding-ish random walk seed
            chain_pos = np.zeros((n_c, 3))
            chain_pos[0] = rng.normal(0, R / 4.0, size=3)
            for k in range(1, n_c):
                step = rng.normal(0, 1, size=3)
                step *= config.fene_r0_nm / max(np.linalg.norm(step), 1e-9)
                chain_pos[k] = chain_pos[k - 1] + step
                # Pull back inside cell if it wandered out
                if np.linalg.norm(chain_pos[k]) > 0.95 * R:
                    chain_pos[k] = chain_pos[k] / np.linalg.norm(chain_pos[k]) * 0.9 * R
            chain_idx = []
            for k, sid in enumerate(chain_species):
                pos_list.append(chain_pos[k])
                sp = species_table[sid]
                sid_list.append(sid)
                mass_list.append(sp.mass_da)
                sig_list.append(sp.sigma_nm)
                eps_list.append(sp.epsilon_kj_per_mol)
                diff_list.append(sp.diffusion_nm2_per_ns)
                chain_idx.append(len(pos_list) - 1)
            chain_arrays.append(np.array(chain_idx, dtype=np.int64))

    pos = np.array(pos_list, dtype=np.float64)
    species_id = np.array(sid_list, dtype=np.int32)
    mass = np.array(mass_list, dtype=np.float64)
    sigma = np.array(sig_list, dtype=np.float64)
    epsilon = np.array(eps_list, dtype=np.float64)
    diffusion = np.array(diff_list, dtype=np.float64)

    cp_arr = (np.array(complex_pairs, dtype=np.int64)
              if complex_pairs else np.empty((0, 2), dtype=np.int64))

    return CellState(pos=pos, species_id=species_id, mass=mass,
                     sigma=sigma, epsilon=epsilon, diffusion=diffusion,
                     chains=chain_arrays, complex_pairs=cp_arr)


# ---------------------------------------------------------------------------
def _build_pair_list(state: CellState, config: CellPhysicsConfig
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Return (iu, ju) index arrays for all i<j pairs within
    config.lj_cutoff_factor * max(sigma_i, sigma_j) — minus bonded pairs
    (1-2 exclusions standard in MD: bonded atoms see only the bond force,
    not LJ).

    Naive O(N²) for N up to a few thousand — fine for v1.
    """
    pos = state.pos
    sig = state.sigma
    n = pos.shape[0]
    if n < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    # Use the largest sigma as a global cutoff — cheap, slightly conservative
    r_cut = config.lj_cutoff_factor * sig.max()
    diff = pos[:, None, :] - pos[None, :, :]              # (N, N, 3)
    dist = np.linalg.norm(diff, axis=-1)                  # (N, N)
    mask = (dist > 0) & (dist < r_cut)
    iu, ju = np.triu_indices(n, k=1)
    keep = mask[iu, ju]
    iu, ju = iu[keep], ju[keep]
    # 1-2 exclusion: drop pairs that share a chain bond or complex restraint.
    if state.chains or state.complex_pairs.shape[0]:
        bonded = set()
        for chain in state.chains:
            for k in range(chain.size - 1):
                i, j = int(chain[k]), int(chain[k + 1])
                bonded.add((min(i, j), max(i, j)))
        for cp in state.complex_pairs:
            i, j = int(cp[0]), int(cp[1])
            bonded.add((min(i, j), max(i, j)))
        if bonded:
            keep_filter = np.fromiter(
                ((int(i), int(j)) not in bonded for i, j in zip(iu, ju)),
                dtype=bool, count=iu.size,
            )
            iu, ju = iu[keep_filter], ju[keep_filter]
    return iu, ju


def _lj_forces(state: CellState, config: CellPhysicsConfig,
                iu: np.ndarray, ju: np.ndarray) -> np.ndarray:
    """Pairwise WCA-like soft excluded volume between non-bonded particles.

    Lorentz-Berthelot: sigma = (sigma_i + sigma_j)/2, eps = sqrt(eps_i*eps_j).
    Forces capped to ``max_force_kj_per_nm`` to keep stuff stable on hot starts.
    """
    forces = np.zeros_like(state.pos)
    if iu.size == 0:
        return forces
    pos_i = state.pos[iu]; pos_j = state.pos[ju]
    sig_i = state.sigma[iu]; sig_j = state.sigma[ju]
    eps_i = state.epsilon[iu]; eps_j = state.epsilon[ju]
    sig = 0.5 * (sig_i + sig_j)
    eps = np.sqrt(eps_i * eps_j)
    r_ij = pos_j - pos_i
    r_mag = np.maximum(np.linalg.norm(r_ij, axis=-1), 0.05 * sig)
    inv_r = 1.0 / r_mag
    sr6 = (sig * inv_r) ** 6
    # F = 24 eps / r * (2 sr12 - sr6) along r̂_ij
    f_mag = 24.0 * eps * inv_r * (2.0 * sr6 * sr6 - sr6)
    # Cap
    f_mag = np.clip(f_mag, -config.max_force_kj_per_nm, config.max_force_kj_per_nm)
    f_vec = (f_mag * inv_r)[:, None] * r_ij
    np.add.at(forces, iu, -f_vec)
    np.add.at(forces, ju, f_vec)
    return forces


def _fene_chain_forces(state: CellState, config: CellPhysicsConfig
                        ) -> np.ndarray:
    """FENE bonds + 3-body bending stiffness along each chain.

    FENE: U(r) = -0.5 K r_max² ln(1 - (r/r_max)²)
          F(r) = -K r / (1 - (r/r_max)²)
    Bending: U(θ) = 0.5 k_θ (θ - π)²    (preferred straight)
    """
    forces = np.zeros_like(state.pos)
    if not state.chains:
        return forces
    K = config.fene_k_kj_per_nm2
    r_max = config.fene_r_max_factor * config.fene_r0_nm
    for chain in state.chains:
        if chain.size < 2:
            continue
        a = chain[:-1]; b = chain[1:]
        r_ij = state.pos[b] - state.pos[a]
        r_mag = np.maximum(np.linalg.norm(r_ij, axis=-1), 1e-3)
        # Avoid singularity: clamp r/r_max < 0.99
        u = np.clip(r_mag / r_max, 0.0, 0.99)
        f_mag = -K * r_mag / (1.0 - u * u)
        f_mag = np.clip(f_mag, -config.max_force_kj_per_nm,
                          config.max_force_kj_per_nm)
        u_ij = r_ij / r_mag[:, None]
        f_vec = f_mag[:, None] * u_ij
        np.add.at(forces, a, f_vec)
        np.add.at(forces, b, -f_vec)
        # Bending — three consecutive beads
        if chain.size >= 3 and config.chain_bend_k_kj_per_rad2 > 0:
            i = chain[:-2]; j = chain[1:-1]; k = chain[2:]
            r_ij = state.pos[j] - state.pos[i]
            r_jk = state.pos[k] - state.pos[j]
            ni = np.maximum(np.linalg.norm(r_ij, axis=-1), 1e-3)[:, None]
            nk = np.maximum(np.linalg.norm(r_jk, axis=-1), 1e-3)[:, None]
            ui = r_ij / ni
            uk = r_jk / nk
            cos_t = np.clip(np.einsum('ij,ij->i', ui, uk), -1.0 + 1e-7, 1.0 - 1e-7)
            theta = np.arccos(cos_t)
            # Straight is theta=0 (vectors aligned)
            dU_dtheta = config.chain_bend_k_kj_per_rad2 * theta
            sin_t = np.sqrt(1 - cos_t * cos_t)
            # Standard angle-force decomposition (simplified)
            f_i = (dU_dtheta / (sin_t + 1e-9))[:, None] * (uk - cos_t[:, None] * ui) / ni
            f_k = (dU_dtheta / (sin_t + 1e-9))[:, None] * (cos_t[:, None] * uk - ui) / nk
            f_j = -(f_i + f_k)
            np.add.at(forces, i, f_i)
            np.add.at(forces, j, f_j)
            np.add.at(forces, k, f_k)
    return forces


def _complex_forces(state: CellState, config: CellPhysicsConfig) -> np.ndarray:
    """Harmonic restraints between complex member pairs."""
    forces = np.zeros_like(state.pos)
    if state.complex_pairs.shape[0] == 0:
        return forces
    a = state.complex_pairs[:, 0]
    b = state.complex_pairs[:, 1]
    r_ij = state.pos[b] - state.pos[a]
    r_mag = np.maximum(np.linalg.norm(r_ij, axis=-1), 1e-3)
    u_ij = r_ij / r_mag[:, None]
    f_mag = config.complex_k_kj_per_nm2 * (r_mag - config.complex_r0_nm)
    f_mag = np.clip(f_mag, -config.max_force_kj_per_nm, config.max_force_kj_per_nm)
    f_vec = f_mag[:, None] * u_ij
    np.add.at(forces, a, f_vec)
    np.add.at(forces, b, -f_vec)
    return forces


def _wall_forces(state: CellState, config: CellPhysicsConfig) -> np.ndarray:
    """Soft inward push when particles approach the cell membrane."""
    r = np.linalg.norm(state.pos, axis=-1)
    R = config.cell_radius_nm
    margin = config.wall_thickness_nm
    excess = r - (R - margin)
    forces = np.zeros_like(state.pos)
    inside = excess > 0
    if inside.any():
        # Force points toward origin
        u_r = state.pos[inside] / np.maximum(r[inside, None], 1e-9)
        f_mag = -config.wall_k_kj_per_nm2 * excess[inside]
        f_mag = np.clip(f_mag, -config.max_force_kj_per_nm, config.max_force_kj_per_nm)
        forces[inside] = (f_mag[:, None] * u_r)
    return forces


# ---------------------------------------------------------------------------
def compute_forces(state: CellState, config: CellPhysicsConfig) -> np.ndarray:
    """Sum of LJ + FENE + complex + wall forces. Per-atom force cap applied
    once at the end (in addition to the per-term caps above)."""
    if state.step % config.pair_rebuild_every == 0 or state._pair_iu is None:
        state._pair_iu, state._pair_ju = _build_pair_list(state, config)
        state._pair_built_step = state.step
    forces = (_lj_forces(state, config, state._pair_iu, state._pair_ju)
              + _fene_chain_forces(state, config)
              + _complex_forces(state, config)
              + _wall_forces(state, config))
    norms = np.linalg.norm(forces, axis=-1)
    too_big = norms > config.max_force_kj_per_nm
    if too_big.any():
        scale = config.max_force_kj_per_nm / norms[too_big]
        forces[too_big] *= scale[:, None]
    return forces


# ---------------------------------------------------------------------------
def brownian_step(state: CellState, config: CellPhysicsConfig,
                   forces: np.ndarray, rng: np.random.Generator) -> None:
    """Overdamped Brownian dynamics (the right physics for cell scale).

        x_{t+dt} = x_t  +  D · F · dt / kT  +  sqrt(2 D dt) · N(0, 1)

    Per-species D in nm²/ns. F in kJ/(mol·nm). kT in kJ/mol. Result is a
    position step in nm. No mass, no velocity — that's the point of the
    overdamped limit (Smoldyn / Langevin-overdamped / Stokes-flow regime).
    """
    dt = config.dt_ns
    kT = config.kT_kj_per_mol
    D = state.diffusion[:, None]                          # (N, 1)
    drift = D * forces * dt / kT
    noise = np.sqrt(2.0 * D * dt) * rng.normal(0, 1, size=state.pos.shape)
    state.pos += drift + noise
    # Hard wall — clamp anything that escaped despite the soft wall
    r = np.linalg.norm(state.pos, axis=-1)
    R = config.cell_radius_nm
    out = r > R
    if out.any():
        scale = (R * 0.999) / np.maximum(r[out], 1e-9)
        state.pos[out] *= scale[:, None]
    state.t_ns += dt
    state.step += 1


# Backwards-compat alias
langevin_step = brownian_step


# ---------------------------------------------------------------------------
def fire_reactions(state: CellState, reactions: list[Reaction],
                    config: CellPhysicsConfig,
                    species_table: list[SpeciesEntry],
                    rng: np.random.Generator) -> int:
    """Scan reactant pairs within reaction_radius and fire stochastically.

    Returns the number of reaction events that fired this step.
    """
    if not reactions or state.n() < 2:
        return 0
    n_events = 0
    to_remove: set[int] = set()
    spawn_pos: list[np.ndarray] = []
    spawn_sids: list[int] = []
    pos = state.pos
    sid = state.species_id
    for r_idx, rxn in enumerate(reactions):
        a_mask = sid == rxn.reactant_a
        b_mask = sid == rxn.reactant_b
        if not (a_mask.any() and b_mask.any()):
            continue
        a_idx = np.where(a_mask)[0]
        b_idx = np.where(b_mask)[0]
        # Pairwise distances (small N times small N — fine)
        diff = pos[a_idx][:, None, :] - pos[b_idx][None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        if rxn.reactant_a == rxn.reactant_b:
            # avoid self-pairs and double-counting by zeroing lower triangle
            d[np.tril_indices_from(d)] = np.inf
        prox = d < rxn.reaction_radius_nm
        if not prox.any():
            continue
        # Each qualifying pair fires with prob = rate * dt
        p_fire = 1.0 - np.exp(-rxn.rate_per_ns * config.dt_ns)
        pairs_ai, pairs_bj = np.where(prox)
        rolls = rng.random(pairs_ai.size)
        fires = rolls < p_fire
        for fi, (ai, bj) in enumerate(zip(pairs_ai[fires], pairs_bj[fires])):
            i = int(a_idx[ai]); j = int(b_idx[bj])
            if i in to_remove or j in to_remove:
                continue
            to_remove.add(i); to_remove.add(j)
            centroid = 0.5 * (pos[i] + pos[j])
            for prod_sid in rxn.products:
                spawn_pos.append(centroid + rng.normal(0, 0.1, size=3))
                spawn_sids.append(prod_sid)
            state.rxn_counts[r_idx] = state.rxn_counts.get(r_idx, 0) + 1
            n_events += 1
    if not n_events:
        return 0
    # Remove reactants, append products
    keep_mask = np.ones(state.n(), dtype=bool)
    keep_mask[list(to_remove)] = False
    state.pos = state.pos[keep_mask]
    state.species_id = state.species_id[keep_mask]
    state.mass = state.mass[keep_mask]
    state.sigma = state.sigma[keep_mask]
    state.epsilon = state.epsilon[keep_mask]
    state.diffusion = state.diffusion[keep_mask]
    # Index remap for chains and complex_pairs
    old_to_new = -np.ones(keep_mask.size, dtype=np.int64)
    old_to_new[keep_mask] = np.arange(keep_mask.sum())
    new_chains = []
    for chain in state.chains:
        remapped = old_to_new[chain]
        # Drop chain entirely if any bead vanished — chains shouldn't get
        # destroyed via reactions in v1, this is a safety net.
        if (remapped < 0).any():
            continue
        new_chains.append(remapped)
    state.chains = new_chains
    if state.complex_pairs.size:
        cp = old_to_new[state.complex_pairs]
        cp_keep = (cp >= 0).all(axis=-1)
        state.complex_pairs = cp[cp_keep]
    # Append products
    if spawn_pos:
        sp_arr = np.array(spawn_pos, dtype=np.float64)
        sp_sid = np.array(spawn_sids, dtype=np.int32)
        sp_mass = np.array([species_table[s].mass_da for s in spawn_sids])
        sp_sig = np.array([species_table[s].sigma_nm for s in spawn_sids])
        sp_eps = np.array([species_table[s].epsilon_kj_per_mol for s in spawn_sids])
        sp_diff = np.array([species_table[s].diffusion_nm2_per_ns
                              for s in spawn_sids])
        state.pos = np.concatenate([state.pos, sp_arr])
        state.species_id = np.concatenate([state.species_id, sp_sid])
        state.mass = np.concatenate([state.mass, sp_mass])
        state.sigma = np.concatenate([state.sigma, sp_sig])
        state.epsilon = np.concatenate([state.epsilon, sp_eps])
        state.diffusion = np.concatenate([state.diffusion, sp_diff])
    # Invalidate pair list (indices changed)
    state._pair_iu = None
    return n_events


# ---------------------------------------------------------------------------
def run(state: CellState, config: CellPhysicsConfig,
        species_table: list[SpeciesEntry],
        reactions: Optional[list[Reaction]] = None,
        n_steps: int = 100,
        save_every: int = 1) -> dict:
    """Run ``n_steps`` Langevin + reaction steps. Returns trajectory dict.

    Trajectory layout (variable-N per frame; padded with NaN):
      positions  (T, N_max, 3)   float32
      species_id (T, N_max)      int16  (-1 = empty slot)
      counts     (T, n_species)  int32
      t_ns       (T,)            float32
      rxn_events (T,)            int32  (per-frame event count)
    """
    rng = np.random.default_rng(config.seed + 1)
    reactions = reactions or []
    n_species_vocab = len(species_table)

    n_frames = (n_steps + save_every - 1) // save_every + 1
    # Generous worst-case slot count
    n_max = int(state.n() * 1.5) + 100
    pos_log = np.full((n_frames, n_max, 3), np.nan, dtype=np.float32)
    sid_log = np.full((n_frames, n_max), -1, dtype=np.int16)
    counts_log = np.zeros((n_frames, n_species_vocab), dtype=np.int32)
    t_log = np.zeros(n_frames, dtype=np.float32)
    rxn_log = np.zeros(n_frames, dtype=np.int32)

    def snapshot(idx: int):
        n_now = state.n()
        slot = min(n_now, n_max)
        pos_log[idx, :slot] = state.pos[:slot].astype(np.float32)
        sid_log[idx, :slot] = state.species_id[:slot].astype(np.int16)
        counts_log[idx] = np.bincount(state.species_id, minlength=n_species_vocab)
        t_log[idx] = state.t_ns

    snapshot(0)
    frame_i = 1
    n_rxn_total = 0
    for s in range(n_steps):
        forces = compute_forces(state, config)
        brownian_step(state, config, forces, rng)
        n_rxn = fire_reactions(state, reactions, config, species_table, rng)
        n_rxn_total += n_rxn
        if (s + 1) % save_every == 0 and frame_i < n_frames:
            snapshot(frame_i)
            rxn_log[frame_i] = n_rxn
            frame_i += 1
    return {
        'positions': pos_log[:frame_i],
        'species_id': sid_log[:frame_i],
        'counts': counts_log[:frame_i],
        't_ns': t_log[:frame_i],
        'rxn_events': rxn_log[:frame_i],
        'species_names': [s.name for s in species_table],
        'n_rxn_total': n_rxn_total,
    }
