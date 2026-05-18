"""Particle state + time integration.

Velocity-Verlet integrator with periodic-box position wrap. Forces come
from a caller-supplied function `force_fn(state) -> (P, 3) tensor`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ParticleState:
    """Plain dataclass holding the state for P particles.

    Held as torch tensors so the same code runs on CPU and GPU.
    """

    positions: torch.Tensor       # (P, 3)
    velocities: torch.Tensor      # (P, 3)
    charges: torch.Tensor         # (P,)
    masses: torch.Tensor          # (P,)
    box_size: float               # cubic box side length L; positions wrap mod L

    @property
    def n(self) -> int:
        return self.positions.shape[0]

    def kinetic_energy(self) -> torch.Tensor:
        return 0.5 * (self.masses.unsqueeze(-1) * self.velocities ** 2).sum()

    def momentum(self) -> torch.Tensor:
        return (self.masses.unsqueeze(-1) * self.velocities).sum(dim=0)

    def total_charge(self) -> torch.Tensor:
        return self.charges.sum()

    def wrap_positions(self) -> None:
        self.positions = self.positions % self.box_size


# ----------------------- velocity-Verlet integrator -----------------------

def integrate_velocity_verlet(
    state: ParticleState,
    force_fn,
    dt: float,
) -> torch.Tensor:
    """One velocity-Verlet step. Returns the new forces (post-step).

    The standard velocity-Verlet update:
      v <- v + 0.5 * F(x_t) / m * dt
      x <- x + v * dt
      F_new = force_fn(state)
      v <- v + 0.5 * F(x_{t+1}) / m * dt
    """
    # The caller is responsible for having computed the current force on
    # `state` before the first call; we re-use it inside this routine.
    F = force_fn(state)
    state.velocities = state.velocities + 0.5 * F / state.masses.unsqueeze(-1) * dt
    state.positions = state.positions + state.velocities * dt
    state.wrap_positions()
    F_new = force_fn(state)
    state.velocities = state.velocities + 0.5 * F_new / state.masses.unsqueeze(-1) * dt
    return F_new


def integrate_langevin(
    state: ParticleState,
    force_fn,
    dt: float,
    gamma: float,
    kT: float,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Langevin (Brownian) dynamics step.

    Stochastic Verlet:
      F = force_fn(state)  - gamma * m * v  +  sqrt(2 * gamma * m * kT / dt) * xi
      v <- v + F/m * dt
      x <- x + v * dt
    where xi is unit Gaussian per atom per dim.
    """
    F_det = force_fn(state)
    m = state.masses.unsqueeze(-1)
    noise = torch.randn(state.positions.shape, generator=rng, device=state.positions.device,
                         dtype=state.positions.dtype)
    sigma = torch.sqrt(torch.tensor(2.0 * gamma * kT / dt))
    F_total = F_det - gamma * m * state.velocities + sigma * torch.sqrt(m) * noise
    state.velocities = state.velocities + F_total / m * dt
    state.positions = state.positions + state.velocities * dt
    state.wrap_positions()
    return F_total
