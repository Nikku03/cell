"""M8 upgrade 10/10: Neural ODE flux head (alternative to PINNHead).

The current PINNHead implements a DISCRETE step:
    x_{t+1} = x_t + S · v(x_t)   (in linear space)

The Neural ODE alternative integrates a CONTINUOUS-time rate function:
    dx/dt = f_θ(x, t)
over the interval [t, t+1] using a numerical solver (RK4, dopri5, ...).

Why this could help M7:
  - **Smoother long-horizon trajectories**: the implicit solver tames
    error compounding across many steps.
  - **Adaptive step sizes**: dopri5 takes more steps where dynamics
    are fast (e.g. during sharp transitions) and fewer where slow,
    matching the simulator's own behavior.
  - **Same mass-conservation guarantee** if f_θ is parameterized as
    S · v_θ(x) (S is the stoichiometric matrix).

Why it's risky for M7:
  - Slower to train: backprop through the solver is expensive.
    Adjoint method available but adds noise.
  - Harder to tune the solver (rtol, atol, max_steps).
  - Doesn't break the val_count = 0.0716 stochastic noise floor
    (M8 stochastic head does that).

This is provided as an OPT-IN alternative. The default M7 still uses
PINNHead (which is empirically rock-solid). Switch via:
    cfg.use_node_flux_head = True

Requires torchdiffeq:
    pip install torchdiffeq

The implementation uses the FIXED-STEP RK4 solver for training (no
adjoint, no surprises), and dopri5 for inference (adaptive, more
accurate).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from cell_sim.lgnn.models.pinn_head import signed_log1p, signed_expm1


class NeuralODEFluxHead(nn.Module):
    """Neural ODE-based flux head.

    The forward integrates dx/dt = f(x, t) = S · v_θ(x_t) over [t, t+1].
    Returns x_{t+1} (signed-log1p space) and v_log = log(v(x_t)).

    Args
    ----
    hidden        : encoder hidden dim
    stoich_matrix : (n_species, n_reactions) tensor (same as PINNHead)
    flux_indices  : indices of F_* rows (same as PINNHead)
    rate_clip     : clamp on log-space rate predictions
    solver        : 'rk4' for training, 'dopri5' for inference. Set
                     via switch_to_inference() / switch_to_training().
    step_size     : fixed step size for rk4. 0.1 means 10 sub-steps per
                     unit-time integration. Smaller = more accurate but slower.

    Backwards compatible: same forward signature as PINNHead.
    """

    def __init__(
        self,
        hidden: int,
        stoich_matrix: torch.Tensor,
        flux_indices: torch.Tensor,
        rate_clip: float = 6.0,
        solver: str = 'rk4',
        step_size: float = 0.5,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ):
        super().__init__()
        self.hidden = hidden
        self.rate_clip = rate_clip
        self.solver = solver
        self.step_size = step_size
        self.rtol = rtol
        self.atol = atol

        # Same submodules as PINNHead: per-reaction rate predictor
        n_reactions = stoich_matrix.shape[1]
        self.rate_head = nn.Linear(hidden, 1)              # encodes F_* rows
        # Pre-register stoich + flux idx
        self.register_buffer('stoich', stoich_matrix.float())
        self.register_buffer('flux_indices', flux_indices.long())

        # Optional residual head for non-SBML rows
        self.residual_head = nn.Linear(hidden, 1)

        # Compute SBML-row mask (used by residual head suppression)
        S_abs = stoich_matrix.abs().sum(dim=1)
        self.register_buffer('s_covered_mask', (S_abs > 1e-9).float())

    def _f(self, t: torch.Tensor, x: torch.Tensor, h: torch.Tensor):
        """ODE function: dx/dt = S · v(x) + r(x).

        x: (B, N) in signed-log1p space.
        h: (B, N, hidden) — encoder output (FIXED during integration).
        """
        # In log space, predict log-rates at F_* rows
        v_log_at_flux = self.rate_head(h[:, self.flux_indices, :]).squeeze(-1)
        v_log = torch.clamp(v_log_at_flux, -self.rate_clip, self.rate_clip)
        # Bridge to linear
        v_linear = signed_expm1(v_log)                     # (B, n_rxn)
        # Mass-conserving step: dx_linear = S @ v
        x_linear = signed_expm1(x)
        dx_linear = v_linear @ self.stoich.T               # (B, N)

        # Residual head for non-SBML rows (additive)
        r = self.residual_head(h).squeeze(-1)              # (B, N)
        r = torch.clamp(r, -self.rate_clip, self.rate_clip)
        # Only apply residual where stoich is zero (non-PINN rows)
        residual = r * (1.0 - self.s_covered_mask)         # (B, N)

        # dx/dt in linear space
        dx = dx_linear + residual
        # Return the rate in CURRENT (signed-log1p) space via chain rule:
        # d(signed_log1p(x))/dt = dx/dt / (1 + |x|) * sign(x)
        # but for stability we'll integrate in linear space and convert.
        # Actually: easier to integrate in log space directly.
        # For numerical stability of the ODE solver, we'll work in log
        # space throughout. The chain rule is:
        #   d/dt log(1+|x|) = (1 / (1+|x|)) * sign(x) * dx/dt
        # Equivalently:
        #   dx_log/dt = dx_lin/dt / (1 + |x_lin|)
        denom = 1.0 + x_linear.abs().clamp(min=1e-6)
        return dx / denom

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ):
        """x: (B, N) signed-log1p.  h: (B, N, hidden).
        Returns (x_next, v_log) matching PINNHead's contract.
        """
        try:
            from torchdiffeq import odeint
        except ImportError:
            raise ImportError(
                'torchdiffeq not installed. Install with:\n'
                '  pip install torchdiffeq'
            )

        # Closure to make h a constant during integration
        h_const = h

        def f_t(t, x_t):
            return self._f(t, x_t, h_const)

        t_span = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)
        if self.solver == 'rk4':
            # Fixed-step RK4 — robust for training
            n_sub = max(int(1.0 / self.step_size), 1)
            t_span = torch.linspace(0.0, 1.0, n_sub + 1, device=x.device, dtype=x.dtype)
            x_traj = odeint(f_t, x, t_span, method='rk4',
                            options={'step_size': self.step_size})
        else:
            # Adaptive dopri5 — for inference / verification
            x_traj = odeint(f_t, x, t_span, method='dopri5',
                            rtol=self.rtol, atol=self.atol)
        x_next = x_traj[-1]

        # Re-compute v_log at the initial state for the flux loss
        with torch.no_grad():
            v_log_at_flux = self.rate_head(
                h[:, self.flux_indices, :]).squeeze(-1)
        v_log_at_flux = torch.clamp(
            v_log_at_flux, -self.rate_clip, self.rate_clip)

        return x_next, v_log_at_flux

    def switch_to_inference(self):
        """Switch to adaptive dopri5 solver. Call after training."""
        self.solver = 'dopri5'

    def switch_to_training(self):
        """Switch back to fixed-step rk4 solver. Default."""
        self.solver = 'rk4'
