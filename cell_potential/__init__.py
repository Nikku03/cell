"""Field-particle hybrid simulator: classical PDE fields + learnable particle units.

A small but real implementation of the particle-in-cell (PIC) architecture for
biophysical simulation. The package is organised so each layer of the stack
can be verified independently:

  grid.py        the field substrate (Poisson, diffusion, gradients)
  particle.py    the particle layer (state, dynamics)
  coupling.py    cloud-in-cell deposition and trilinear interpolation
  forces.py      total-force assembly (field long-range + direct short-range)
  integrator.py  time integration (velocity-Verlet, Langevin)
  verify.py      conservation + invariance + accuracy gates

Three phases, each its own demo with a hard verification contract:

  demo_phase0    one electric field + ~10 particles. Tests:
                 charge conservation, translation invariance, rotation
                 invariance, momentum conservation, energy conservation
                 in Hamiltonian dynamics, long-range Coulomb agreement.

  demo_phase1    add a diffusing concentration field that particles
                 consume from / deposit into. Tests:
                 mass conservation, diffusion of a Gaussian against the
                 analytic solution, particle-source consistency.

  demo_phase2    100 particles with combined short-range pairwise
                 (Lennard-Jones, placeholder for v3 kernel) + long-range
                 field-mediated Coulomb. Tests:
                 P3M agreement vs direct-pair Coulomb baseline, total
                 momentum / charge conservation, energy stability.

The architecture is *infrastructure*. Once the verification suite passes,
the short-range pairwise term can be replaced with a v3-class learned
kernel without touching the rest of the stack.
"""

from .grid import Grid, solve_poisson, diffusion_step_spectral, grid_gradient_spectral
from .particle import ParticleState, integrate_velocity_verlet
from .coupling import deposit_cic, interpolate_scalar, interpolate_vector

__all__ = [
    "Grid",
    "solve_poisson",
    "diffusion_step_spectral",
    "grid_gradient_spectral",
    "ParticleState",
    "integrate_velocity_verlet",
    "deposit_cic",
    "interpolate_scalar",
    "interpolate_vector",
]
