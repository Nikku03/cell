"""GPU-accelerated cell_physics — torch backend for the integration loop.

Same semantics as ``cell_physics.run`` but with all hot paths on GPU
(positions, forces, FENE chain, complex restraints, sphere wall, Brownian
step). The pair list is still built on CPU via ``scipy.spatial.cKDTree``
because the C-implemented spatial index is faster than naive GPU pairwise
distance computation for our scale; pair indices are then moved to GPU
for force evaluation.

Reactions stay on CPU for now (less hot, easier to manage births/deaths).
The GPU path is invalidated and rebuilt when reactions fire.

Usage:
    from cell_sim.atom_engine.cell_physics_gpu import run_gpu
    traj = run_gpu(state, config, species, reactions=rxns,
                   n_steps=10000, save_every=100, device='cuda')

Returns the same trajectory dict shape as ``cell_physics.run``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

# We rely on the CPU module for state types, reaction firing, bonded set
# computation, and pair list build (cKDTree).
from .cell_physics import (
    CellPhysicsConfig, SpeciesEntry, Reaction, CellState,
    _build_pair_list, _build_bonded_set, fire_reactions,
)


# ---------------------------------------------------------------------------
def _gpu_forces(pos: torch.Tensor,
                 sigma: torch.Tensor,
                 epsilon: torch.Tensor,
                 iu: torch.Tensor, ju: torch.Tensor,
                 chain_a: Optional[torch.Tensor],
                 chain_b: Optional[torch.Tensor],
                 chain_i: Optional[torch.Tensor],
                 chain_j: Optional[torch.Tensor],
                 chain_k: Optional[torch.Tensor],
                 complex_a: Optional[torch.Tensor],
                 complex_b: Optional[torch.Tensor],
                 cfg: CellPhysicsConfig) -> torch.Tensor:
    """Sum of LJ + FENE bond + chain bending + complex + soft wall, on GPU."""
    n = pos.shape[0]
    forces = torch.zeros_like(pos)
    max_f = cfg.max_force_kj_per_nm

    # ---- LJ excluded volume ----
    if iu.numel():
        pos_i = pos[iu]; pos_j = pos[ju]
        sig = 0.5 * (sigma[iu] + sigma[ju])
        eps = torch.sqrt(epsilon[iu] * epsilon[ju])
        r_ij = pos_j - pos_i
        r_mag = torch.clamp(r_ij.norm(dim=-1), min=0.05 * sig.abs())
        inv_r = 1.0 / r_mag
        sr6 = (sig * inv_r) ** 6
        f_mag = 24.0 * eps * inv_r * (2.0 * sr6 * sr6 - sr6)
        f_mag = torch.clamp(f_mag, -max_f, max_f)
        f_vec = (f_mag * inv_r).unsqueeze(-1) * r_ij
        forces.index_add_(0, iu, -f_vec)
        forces.index_add_(0, ju, f_vec)

    # ---- FENE chain bonds ----
    if chain_a is not None and chain_a.numel():
        K = cfg.fene_k_kj_per_nm2
        r_max = cfg.fene_r_max_factor * cfg.fene_r0_nm
        r_ij = pos[chain_b] - pos[chain_a]
        r_mag = torch.clamp(r_ij.norm(dim=-1), min=1e-3)
        u = torch.clamp(r_mag / r_max, max=0.99)
        f_mag = -K * r_mag / (1.0 - u * u)
        f_mag = torch.clamp(f_mag, -max_f, max_f)
        f_vec = f_mag.unsqueeze(-1) * (r_ij / r_mag.unsqueeze(-1))
        forces.index_add_(0, chain_a, f_vec)
        forces.index_add_(0, chain_b, -f_vec)

    # ---- Chain bending (3-body) ----
    if chain_i is not None and chain_i.numel() and cfg.chain_bend_k_kj_per_rad2 > 0:
        r_ij = pos[chain_j] - pos[chain_i]
        r_jk = pos[chain_k] - pos[chain_j]
        ni = torch.clamp(r_ij.norm(dim=-1), min=1e-3).unsqueeze(-1)
        nk = torch.clamp(r_jk.norm(dim=-1), min=1e-3).unsqueeze(-1)
        ui = r_ij / ni
        uk = r_jk / nk
        cos_t = torch.clamp((ui * uk).sum(dim=-1), -1 + 1e-7, 1 - 1e-7)
        theta = torch.arccos(cos_t)
        sin_t = torch.sqrt(1 - cos_t * cos_t)
        dU_dtheta = cfg.chain_bend_k_kj_per_rad2 * theta
        scale_i = (dU_dtheta / (sin_t + 1e-9)).unsqueeze(-1)
        scale_k = scale_i
        f_i = scale_i * (uk - cos_t.unsqueeze(-1) * ui) / ni
        f_k = scale_k * (cos_t.unsqueeze(-1) * uk - ui) / nk
        f_j = -(f_i + f_k)
        forces.index_add_(0, chain_i, f_i)
        forces.index_add_(0, chain_j, f_j)
        forces.index_add_(0, chain_k, f_k)

    # ---- Complex harmonic restraints ----
    if complex_a is not None and complex_a.numel():
        r_ij = pos[complex_b] - pos[complex_a]
        r_mag = torch.clamp(r_ij.norm(dim=-1), min=1e-3)
        u_ij = r_ij / r_mag.unsqueeze(-1)
        f_mag = cfg.complex_k_kj_per_nm2 * (r_mag - cfg.complex_r0_nm)
        f_mag = torch.clamp(f_mag, -max_f, max_f)
        f_vec = f_mag.unsqueeze(-1) * u_ij
        forces.index_add_(0, complex_a, f_vec)
        forces.index_add_(0, complex_b, -f_vec)

    # ---- Soft sphere wall ----
    r = pos.norm(dim=-1)
    R = cfg.cell_radius_nm
    margin = cfg.wall_thickness_nm
    excess = r - (R - margin)
    inside = excess > 0
    if inside.any():
        u_r = pos[inside] / torch.clamp(r[inside].unsqueeze(-1), min=1e-9)
        f_mag = -cfg.wall_k_kj_per_nm2 * excess[inside]
        f_mag = torch.clamp(f_mag, -max_f, max_f)
        forces[inside] = forces[inside] + f_mag.unsqueeze(-1) * u_r

    # ---- Final per-atom force cap ----
    norms = forces.norm(dim=-1)
    too_big = norms > max_f
    if too_big.any():
        scale = max_f / torch.clamp(norms[too_big], min=1e-9)
        forces[too_big] = forces[too_big] * scale.unsqueeze(-1)
    return forces


# ---------------------------------------------------------------------------
def _gpu_brownian_step(pos: torch.Tensor, diffusion: torch.Tensor,
                        forces: torch.Tensor,
                        cfg: CellPhysicsConfig,
                        gen: torch.Generator) -> torch.Tensor:
    """Overdamped Brownian update + hard wall clamp, GPU."""
    dt = cfg.dt_ns
    kT = cfg.kT_kj_per_mol
    D = diffusion.unsqueeze(-1)
    drift = D * forces * dt / kT
    noise_scale = torch.sqrt(2.0 * D * dt)
    pos = pos + drift + noise_scale * torch.randn(pos.shape, generator=gen,
                                                    device=pos.device,
                                                    dtype=pos.dtype)
    # Hard wall
    r = pos.norm(dim=-1)
    R = cfg.cell_radius_nm
    out = r > R
    if out.any():
        scale = (R * 0.999) / torch.clamp(r[out], min=1e-9)
        pos[out] = pos[out] * scale.unsqueeze(-1)
    return pos


# ---------------------------------------------------------------------------
def _build_chain_indices(state: CellState, device: str
                          ) -> tuple[Optional[torch.Tensor], ...]:
    """Pre-compute chain index tensors for GPU access."""
    if not state.chains:
        return None, None, None, None, None
    chain_a, chain_b = [], []
    chain_i, chain_j, chain_k = [], [], []
    for chain in state.chains:
        a = chain[:-1]; b = chain[1:]
        chain_a.append(a); chain_b.append(b)
        if chain.size >= 3:
            chain_i.append(chain[:-2])
            chain_j.append(chain[1:-1])
            chain_k.append(chain[2:])
    chain_a_t = torch.tensor(np.concatenate(chain_a), dtype=torch.long,
                              device=device) if chain_a else None
    chain_b_t = torch.tensor(np.concatenate(chain_b), dtype=torch.long,
                              device=device) if chain_b else None
    chain_i_t = torch.tensor(np.concatenate(chain_i), dtype=torch.long,
                              device=device) if chain_i else None
    chain_j_t = torch.tensor(np.concatenate(chain_j), dtype=torch.long,
                              device=device) if chain_j else None
    chain_k_t = torch.tensor(np.concatenate(chain_k), dtype=torch.long,
                              device=device) if chain_k else None
    return chain_a_t, chain_b_t, chain_i_t, chain_j_t, chain_k_t


def _build_complex_indices(state: CellState, device: str
                            ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if state.complex_pairs.shape[0] == 0:
        return None, None
    cp = state.complex_pairs
    return (torch.tensor(cp[:, 0], dtype=torch.long, device=device),
            torch.tensor(cp[:, 1], dtype=torch.long, device=device))


# ---------------------------------------------------------------------------
def run_gpu(state: CellState, config: CellPhysicsConfig,
             species_table: list[SpeciesEntry],
             reactions: Optional[list[Reaction]] = None,
             n_steps: int = 100, save_every: int = 1,
             device: str = 'cuda') -> dict:
    """GPU run loop. Same return shape as ``cell_physics.run``.

    Hot path stays on GPU; pair list is rebuilt periodically via cKDTree
    on CPU and pushed back to GPU. Reactions, when they fire, sync state
    back to CPU briefly, fire them, then re-push to GPU and rebuild
    chain/complex tensors.
    """
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    rng_np = np.random.default_rng(config.seed + 1)
    gen = torch.Generator(device=device)
    gen.manual_seed(config.seed + 2)
    reactions = reactions or []
    n_species_vocab = len(species_table)

    # Move state to GPU
    pos_t = torch.tensor(state.pos, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(state.sigma, dtype=torch.float32, device=device)
    epsilon_t = torch.tensor(state.epsilon, dtype=torch.float32, device=device)
    diffusion_t = torch.tensor(state.diffusion, dtype=torch.float32,
                                device=device)

    chain_a, chain_b, chain_i, chain_j, chain_k = _build_chain_indices(
        state, device)
    complex_a, complex_b = _build_complex_indices(state, device)

    # Allocate output buffers (CPU)
    n_frames = (n_steps + save_every - 1) // save_every + 1
    n_max = int(state.n() * 1.5) + 100
    pos_log = np.full((n_frames, n_max, 3), np.nan, dtype=np.float32)
    sid_log = np.full((n_frames, n_max), -1, dtype=np.int16)
    pid_log = np.full((n_frames, n_max), -1, dtype=np.int64)
    counts_log = np.zeros((n_frames, n_species_vocab), dtype=np.int32)
    t_log = np.zeros(n_frames, dtype=np.float32)
    rxn_log = np.zeros(n_frames, dtype=np.int32)

    def snapshot(idx: int):
        n_now = state.n()
        slot = min(n_now, n_max)
        pos_log[idx, :slot] = pos_t[:slot].detach().cpu().numpy()
        sid_log[idx, :slot] = state.species_id[:slot].astype(np.int16)
        pid_log[idx, :slot] = state.persistent_id[:slot].astype(np.int64)
        counts_log[idx] = np.bincount(state.species_id, minlength=n_species_vocab)
        t_log[idx] = state.t_ns

    snapshot(0)
    frame_i = 1
    n_rxn_total = 0

    iu_t, ju_t = None, None

    for s in range(n_steps):
        # ---- Pair list (CPU cKDTree, periodic) ----
        if s % config.pair_rebuild_every == 0 or iu_t is None:
            # Sync positions back briefly to build pair list
            state.pos = pos_t.detach().cpu().numpy().astype(np.float64)
            iu_np, ju_np = _build_pair_list(state, config)
            iu_t = torch.tensor(iu_np, dtype=torch.long, device=device)
            ju_t = torch.tensor(ju_np, dtype=torch.long, device=device)

        # ---- Forces (GPU) ----
        F = _gpu_forces(pos_t, sigma_t, epsilon_t, iu_t, ju_t,
                         chain_a, chain_b, chain_i, chain_j, chain_k,
                         complex_a, complex_b, config)

        # ---- Brownian step (GPU) ----
        pos_t = _gpu_brownian_step(pos_t, diffusion_t, F, config, gen)
        state.t_ns += config.dt_ns
        state.step += 1

        # ---- Reactions (CPU) ----
        n_rxn = 0
        if reactions:
            # Sync positions to state for reaction firing
            state.pos = pos_t.detach().cpu().numpy().astype(np.float64)
            n_rxn = fire_reactions(state, reactions, config, species_table,
                                    rng_np)
            if n_rxn > 0:
                # Rebuild GPU tensors after population change
                pos_t = torch.tensor(state.pos, dtype=torch.float32, device=device)
                sigma_t = torch.tensor(state.sigma, dtype=torch.float32, device=device)
                epsilon_t = torch.tensor(state.epsilon, dtype=torch.float32, device=device)
                diffusion_t = torch.tensor(state.diffusion, dtype=torch.float32,
                                            device=device)
                chain_a, chain_b, chain_i, chain_j, chain_k = _build_chain_indices(
                    state, device)
                complex_a, complex_b = _build_complex_indices(state, device)
                iu_t, ju_t = None, None
        n_rxn_total += n_rxn

        # ---- Snapshot ----
        if (s + 1) % save_every == 0 and frame_i < n_frames:
            # If reactions didn't sync, do it now for snapshot accuracy
            if n_rxn == 0 and reactions:
                state.pos = pos_t.detach().cpu().numpy().astype(np.float64)
            elif not reactions:
                state.pos = pos_t.detach().cpu().numpy().astype(np.float64)
            snapshot(frame_i)
            rxn_log[frame_i] = n_rxn
            frame_i += 1

    return {
        'positions': pos_log[:frame_i],
        'species_id': sid_log[:frame_i],
        'persistent_id': pid_log[:frame_i],
        'counts': counts_log[:frame_i],
        't_ns': t_log[:frame_i],
        'rxn_events': rxn_log[:frame_i],
        'species_names': [s.name for s in species_table],
        'n_rxn_total': n_rxn_total,
    }
