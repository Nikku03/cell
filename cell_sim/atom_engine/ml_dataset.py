"""Next-event dataset extractor.

Runs reactive MD trajectories, snapshots atom state + local graph every
``snapshot_every`` steps, and labels each atom by whether it forms a
NEW bond within a ``horizon`` MD-step window after the snapshot.

Emits NumPy arrays that feed straight into the training loop in
``ml_model.py`` without further preprocessing.

Schema for one snapshot with N atoms:
    node_features: (N, F) float32
    edges:         (E, 2) int32   — undirected live-bond pairs (i, j)
    edge_features: (E, 3) float32 — bond kind one-hot or length
    labels:        (N,)   int8    — 1 if atom forms a new bond within
                                     horizon steps, else 0
    snapshot_id:   scalar int     — for train/val splitting
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .atom_unit import AtomUnit, Bond, BondType
from .element import Element, default_valence
from .force_field import ForceFieldConfig
from .integrator import IntegratorConfig, SimState, step

# Element codes we expose to the model. COARSE_* pseudo-elements are
# excluded; if the soup contains one, it gets a zero one-hot vector.
_ELEMENTS_FOR_ML = [
    Element.H, Element.C, Element.N, Element.O, Element.P, Element.S,
]
_ELEMENT_TO_IDX = {e: i for i, e in enumerate(_ELEMENTS_FOR_ML)}
N_ELEM_FEATURES = len(_ELEMENTS_FOR_ML)

# Feature vector per atom:
#   [one-hot element (6), valence_remaining_norm (1), speed (1),
#    n_live_bonds_norm (1), r_from_origin (1)]  = 10 floats.
N_NODE_FEATURES = N_ELEM_FEATURES + 4


def extract_node_features(atoms: list[AtomUnit]) -> np.ndarray:
    """Return (N, N_NODE_FEATURES) float32 per-atom feature matrix."""
    n = len(atoms)
    out = np.zeros((n, N_NODE_FEATURES), dtype=np.float32)
    for i, a in enumerate(atoms):
        idx = _ELEMENT_TO_IDX.get(a.element)
        if idx is not None:
            out[i, idx] = 1.0
        default_v = default_valence(a.element) or 1
        out[i, N_ELEM_FEATURES] = a.valence_remaining / default_v
        vx, vy, vz = a.velocity
        out[i, N_ELEM_FEATURES + 1] = math.sqrt(vx * vx + vy * vy + vz * vz)
        n_bonds = sum(1 for b in a.bonds if b.death_time_ps is None)
        out[i, N_ELEM_FEATURES + 2] = n_bonds / 4.0
        px, py, pz = a.position
        out[i, N_ELEM_FEATURES + 3] = math.sqrt(px * px + py * py + pz * pz)
    return out


def extract_bond_edges(atoms: list[AtomUnit],
                       bonds: list[Bond]) -> tuple[np.ndarray, np.ndarray]:
    """Return (E, 2) edges (undirected, each pair duplicated i->j and j->i
    for message-passing convenience) and (E, 3) edge features:
    [current_length_nm, is_single, is_double_or_higher].
    """
    id_to_idx = {id(a): i for i, a in enumerate(atoms)}
    e_pairs: list[tuple[int, int]] = []
    e_feat: list[list[float]] = []
    for b in bonds:
        if b.death_time_ps is not None:
            continue
        i = id_to_idx.get(id(b.a))
        j = id_to_idx.get(id(b.b))
        if i is None or j is None:
            continue
        dx = b.b.position[0] - b.a.position[0]
        dy = b.b.position[1] - b.a.position[1]
        dz = b.b.position[2] - b.a.position[2]
        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        is_single = 1.0 if b.kind is BondType.COVALENT_SINGLE else 0.0
        is_multi = 1.0 if b.kind in (BondType.COVALENT_DOUBLE,
                                     BondType.COVALENT_TRIPLE) else 0.0
        feat = [r, is_single, is_multi]
        e_pairs.append((i, j))
        e_feat.append(feat)
        e_pairs.append((j, i))   # reverse direction
        e_feat.append(feat)
    if not e_pairs:
        return (np.zeros((0, 2), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32))
    return (np.asarray(e_pairs, dtype=np.int32),
            np.asarray(e_feat, dtype=np.float32))


@dataclass
class Snapshot:
    node_features: np.ndarray
    edges: np.ndarray
    edge_features: np.ndarray
    labels: np.ndarray            # filled in after horizon simulation
    snapshot_id: int


@dataclass
class TrajectoryCollector:
    """Drives MD + dynamic bonding and emits snapshots with look-ahead labels.

    ``snapshot_every``: steps between snapshots.
    ``horizon``: how many steps after a snapshot we look for a new
        bond_formed event to flag the atom as positive.
    """
    snapshot_every: int = 100
    horizon: int = 100
    snapshots: list[Snapshot] = field(default_factory=list)

    def run(
        self,
        state: SimState,
        ff_cfg: ForceFieldConfig,
        int_cfg: IntegratorConfig,
        n_steps: int,
        trajectory_id: int = 0,
        progress: Optional[Callable[[str], None]] = None,
    ) -> list[Snapshot]:
        """Run ``n_steps``, snapshotting every ``snapshot_every`` and
        labelling with the bond_formed-within-horizon target."""
        atoms = state.atoms
        id_to_idx = {id(a): i for i, a in enumerate(atoms)}
        n = len(atoms)

        # Pending snapshots waiting for their horizon window to close.
        # Each entry: (snapshot, atom_ids_involved_so_far_set, steps_left).
        pending: list[tuple[Snapshot, set[int], int]] = []
        forces = None
        next_snapshot_step = 0

        for k in range(n_steps):
            forces = step(state, ff_cfg, int_cfg, forces)

            # Record new bond-form events.
            new_form_events: list[tuple[int, int]] = []
            # AtomUnit.history records events with atom_ids (the global
            # monotonic id); we need to translate back to per-snapshot
            # indices. Simpler approach: compare bond count before/after.
            # But we don't have 'before' easily. Do this instead: look
            # at state.bonds and flag any bond born at t_ps == current
            # step's t_ps.
            t_now = state.t_ps
            dt = int_cfg.dt_ps
            for b in state.bonds:
                if b.death_time_ps is not None:
                    continue
                # Bond formed THIS step → both endpoints are "positive"
                # for any pending snapshots still within their horizon.
                if abs(b.birth_time_ps - t_now) < 0.5 * dt:
                    i = id_to_idx.get(id(b.a))
                    j = id_to_idx.get(id(b.b))
                    if i is not None:
                        new_form_events.append((i, j if j is not None else -1))
                    if j is not None:
                        new_form_events.append((j, i if i is not None else -1))

            # Update pending snapshots' labels.
            for snap, atoms_already_flagged, _ in pending:
                for (atom_idx, _partner) in new_form_events:
                    if atom_idx not in atoms_already_flagged:
                        snap.labels[atom_idx] = 1
                        atoms_already_flagged.add(atom_idx)

            # Decrement horizons and retire.
            new_pending: list[tuple[Snapshot, set[int], int]] = []
            for snap, seen, steps_left in pending:
                if steps_left - 1 <= 0:
                    self.snapshots.append(snap)
                else:
                    new_pending.append((snap, seen, steps_left - 1))
            pending = new_pending

            # Take a new snapshot?
            if k == next_snapshot_step:
                node_f = extract_node_features(atoms)
                edges, edge_f = extract_bond_edges(atoms, state.bonds)
                labels = np.zeros(n, dtype=np.int8)
                snap = Snapshot(
                    node_features=node_f,
                    edges=edges,
                    edge_features=edge_f,
                    labels=labels,
                    snapshot_id=trajectory_id * 100000 + k,
                )
                pending.append((snap, set(), self.horizon))
                next_snapshot_step = k + self.snapshot_every
                if progress is not None:
                    progress(f"snapshot @ step {k}, t={t_now:.2f} ps, "
                             f"n_pending={len(pending)}")

        # Retire any remaining pending snapshots (their horizon wasn't
        # fully observed, but we still emit them with partial labels).
        for snap, _seen, _left in pending:
            self.snapshots.append(snap)
        return self.snapshots


def save_snapshots(snapshots: list[Snapshot], path) -> None:
    """Save snapshots as a single .npz for easy loading."""
    import pickle
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(snapshots, f)


def load_snapshots(path) -> list[Snapshot]:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
