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
from .force_field import ForceFieldConfig, compute_forces
from .integrator import IntegratorConfig, SimState, step
from .molecule_builder import classify_molecules, canonical_formula

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
    for message-passing convenience) and (E, 4) edge features:
    [current_length_nm, is_single, is_double_or_higher, is_bonded=1].
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
        feat = [r, is_single, is_multi, 1.0]      # last = is_bonded
        e_pairs.append((i, j))
        e_feat.append(feat)
        e_pairs.append((j, i))
        e_feat.append(feat)
    if not e_pairs:
        return (np.zeros((0, 2), dtype=np.int32),
                np.zeros((0, 4), dtype=np.float32))
    return (np.asarray(e_pairs, dtype=np.int32),
            np.asarray(e_feat, dtype=np.float32))


def extract_proximity_edges(atoms: list[AtomUnit],
                            bonds: list[Bond],
                            cutoff_nm: float = 0.35) -> tuple[np.ndarray,
                                                               np.ndarray]:
    """Return (E, 2) edges and (E, 4) features for UNBONDED atom pairs
    within ``cutoff_nm``. Edge feature vector:
    [current_length_nm, is_single=0, is_multi=0, is_bonded=0].

    These proximity edges give the GNN visibility into atoms that could
    PLAUSIBLY form a bond in the near future, not just those already
    bonded — a critical signal for next-event prediction.
    """
    n = len(atoms)
    if n < 2:
        return (np.zeros((0, 2), dtype=np.int32),
                np.zeros((0, 4), dtype=np.float32))
    pos = np.array([a.position for a in atoms], dtype=np.float64)
    bonded_pairs: set[tuple[int, int]] = set()
    id_to_idx = {id(a): i for i, a in enumerate(atoms)}
    for b in bonds:
        if b.death_time_ps is not None:
            continue
        i = id_to_idx.get(id(b.a))
        j = id_to_idx.get(id(b.b))
        if i is None or j is None:
            continue
        bonded_pairs.add((min(i, j), max(i, j)))

    c2 = cutoff_nm * cutoff_nm
    # Use broadcasting for small N; for N > ~500 this would need a
    # neighbor list, but snapshots stay small.
    d = pos[None, :, :] - pos[:, None, :]
    r2 = np.einsum("ijk,ijk->ij", d, d)
    iu, ju = np.triu_indices(n, k=1)
    mask = (r2[iu, ju] < c2) & (r2[iu, ju] > 1e-8)
    iu = iu[mask]
    ju = ju[mask]
    r2_kept = r2[iu, ju]
    # Filter out pairs that are already bonded.
    keep = np.array([(int(a), int(b)) not in bonded_pairs
                     for a, b in zip(iu, ju)], dtype=bool)
    iu = iu[keep]
    ju = ju[keep]
    r2_kept = r2_kept[keep]
    if iu.size == 0:
        return (np.zeros((0, 2), dtype=np.int32),
                np.zeros((0, 4), dtype=np.float32))
    r = np.sqrt(r2_kept).astype(np.float32)
    # Duplicate each pair for both directions.
    iu2 = np.concatenate([iu, ju])
    ju2 = np.concatenate([ju, iu])
    r2a = np.concatenate([r, r])
    edges = np.stack([iu2, ju2], axis=1).astype(np.int32)
    feats = np.zeros((iu2.size, 4), dtype=np.float32)
    feats[:, 0] = r2a
    # is_single=0, is_multi=0, is_bonded=0 → remain zero
    return edges, feats


def extract_all_edges(atoms: list[AtomUnit], bonds: list[Bond],
                      proximity_cutoff_nm: float = 0.35
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate bond edges + proximity edges with a shared feature
    schema."""
    e_b, f_b = extract_bond_edges(atoms, bonds)
    e_p, f_p = extract_proximity_edges(atoms, bonds,
                                       cutoff_nm=proximity_cutoff_nm)
    if e_b.size == 0:
        return e_p, f_p
    if e_p.size == 0:
        return e_b, f_b
    return (np.concatenate([e_b, e_p], axis=0),
            np.concatenate([f_b, f_p], axis=0))


# Set of product molecules we track as classes for reaction-type
# prediction. Any other formula → class 0 ("other"). Class 1 = "no event".
REACTION_CLASSES: list[str] = [
    "other", "none", "H2", "O2", "N2", "H2O", "HO", "NH", "CH",
    "CH2", "CH3", "CH4", "CO", "CO2", "NH3",
]
REACTION_CLASS_TO_IDX = {c: i for i, c in enumerate(REACTION_CLASSES)}
N_REACTION_CLASSES = len(REACTION_CLASSES)


@dataclass
class Snapshot:
    node_features: np.ndarray
    edges: np.ndarray
    edge_features: np.ndarray
    labels: np.ndarray            # binary: forms-bond-in-horizon
    # Multiclass: formula of the molecule this atom becomes part of
    # at the time it (first) forms a bond within the horizon.
    # 1 = "none" (no bond event). 0 = "other" (a product not in the
    # enumerated list).
    reaction_labels: np.ndarray = None
    # (N, 3) per-atom force vector from the reference ``compute_forces``
    # at the time of the snapshot — used as the training target for the
    # neural surrogate force field (Move 3).
    forces_gt: Optional[np.ndarray] = None
    # (N, 3) per-atom xyz positions at the time of the snapshot.
    # Needed by the equivariant force head (requires atom positions to
    # build per-edge unit vectors).
    pos: Optional[np.ndarray] = None
    snapshot_id: int = 0


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
        from .molecule_builder import _connected_components_by_live_bonds

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
            # For reaction_labels we need to know which molecule the
            # atom just joined — take the connected-components formula
            # AFTER this step's events fired.
            if new_form_events:
                comps = _connected_components_by_live_bonds(atoms)
                idx_to_formula: dict[int, str] = {}
                for group in comps:
                    f = canonical_formula(group, atoms)
                    for idx in group:
                        idx_to_formula[idx] = f
            else:
                idx_to_formula = {}

            for snap, atoms_already_flagged, _ in pending:
                for (atom_idx, _partner) in new_form_events:
                    if atom_idx not in atoms_already_flagged:
                        snap.labels[atom_idx] = 1
                        f = idx_to_formula.get(atom_idx, "other")
                        cls_idx = REACTION_CLASS_TO_IDX.get(
                            f, REACTION_CLASS_TO_IDX["other"]
                        )
                        snap.reaction_labels[atom_idx] = cls_idx
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
                edges, edge_f = extract_all_edges(atoms, state.bonds)
                labels = np.zeros(n, dtype=np.int8)
                rlabels = np.full(n, REACTION_CLASS_TO_IDX["none"],
                                  dtype=np.int8)
                # Ground-truth forces for the surrogate FF target.
                forces_gt = compute_forces(atoms, state.bonds,
                                            state.t_ps, ff_cfg)
                pos_now = np.array([a.position for a in atoms],
                                   dtype=np.float32)
                snap = Snapshot(
                    node_features=node_f,
                    edges=edges,
                    edge_features=edge_f,
                    labels=labels,
                    reaction_labels=rlabels,
                    forces_gt=forces_gt.astype(np.float32),
                    pos=pos_now,
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
