"""Neural-surrogate force field — drop-in replacement for
``force_field.compute_forces`` that uses a trained AtomGNN.

The goal is not to match physics exactly; it's to demonstrate that the
learned model can drive an MD integrator forward. Accuracy and
long-term stability are both weaker than the true LJ+harmonic force
field, so the main use is ablation / benchmarking, not production.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from .atom_unit import AtomUnit, Bond
from .ml_dataset import N_ELEM_FEATURES, extract_all_edges, extract_node_features


@dataclass
class NeuralForceField:
    """Callable with the same (atoms, bonds, t_ps, cfg, neighbor_pairs, pos)
    signature as ``force_field.compute_forces``. Uses a trained AtomGNN
    force_head to produce per-atom force vectors.

    ``max_force_kj_per_nm`` caps the prediction magnitude — a learned
    model will occasionally output large gradients for configurations
    it hasn't seen, and without a cap those blow up the integrator.
    """

    model: torch.nn.Module
    max_force_kj_per_nm: float = 2.0e4
    device: str = "cpu"
    mode: str = "equivariant"   # "equivariant" or "mlp" (legacy)

    def __call__(
        self,
        atoms: Sequence[AtomUnit],
        bonds: Iterable[Bond],
        t_ps: float,
        cfg,
        neighbor_pairs=None,
        pos: np.ndarray | None = None,
    ) -> np.ndarray:
        if pos is None:
            pos = np.array([a.position for a in atoms], dtype=np.float64)
        nf = extract_node_features(atoms)
        e, ef = extract_all_edges(atoms, list(bonds))
        nf_t = torch.tensor(nf, device=self.device)
        e_t = torch.tensor(e.astype(np.int64), device=self.device)
        ef_t = torch.tensor(ef, device=self.device)
        pos_t = torch.tensor(pos.astype(np.float32), device=self.device)
        self.model.eval()
        with torch.no_grad():
            if self.mode == "equivariant":
                pred_norm = self.model.predict_forces_equivariant(
                    nf_t, e_t, ef_t, pos_t
                )
                # Per-element un-normalisation.
                if hasattr(self.model, "force_elem_mean"):
                    elem_idx = np.argmax(
                        nf[:, :N_ELEM_FEATURES], axis=-1
                    )
                    m = self.model.force_elem_mean.cpu().numpy()[elem_idx]
                    s = self.model.force_elem_std.cpu().numpy()[elem_idx]
                    pred = (pred_norm.cpu().numpy()
                            * s[:, None] + m[:, None])
                else:
                    pred = pred_norm.cpu().numpy()
            else:
                pred_norm = self.model.predict_forces(nf_t, e_t, ef_t)
                if hasattr(self.model, "force_std"):
                    pred = (pred_norm * self.model.force_std
                            + self.model.force_mean).cpu().numpy()
                else:
                    pred = pred_norm.cpu().numpy()
            forces = pred.astype(np.float64)
        # Cap per-atom magnitude.
        norms = np.linalg.norm(forces, axis=1)
        mask = norms > self.max_force_kj_per_nm
        if mask.any():
            forces[mask] *= (self.max_force_kj_per_nm / norms[mask])[:, None]
        return forces


def make_surrogate_step(force_field: NeuralForceField):
    """Return a function that patches ``integrator.compute_forces`` to
    the neural surrogate for the duration of the caller's scope.

    Usage::

        from cell_sim.atom_engine import integrator as integ_mod
        from .neural_force_field import NeuralForceField, make_surrogate_step

        nff = NeuralForceField(model)
        with make_surrogate_step(nff) as _:
            for _ in range(100):
                forces = integ_mod.step(state, ff_cfg, int_cfg, forces)
    """
    from contextlib import contextmanager
    from . import integrator as _integ

    @contextmanager
    def _ctx():
        original = _integ.compute_forces
        _integ.compute_forces = force_field
        try:
            yield force_field
        finally:
            _integ.compute_forces = original

    return _ctx()
