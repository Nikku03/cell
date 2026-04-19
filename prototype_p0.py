"""
Dark Manifold Virtual Cell — Prototype P0
==========================================

Scope: smallest possible test of the cell-state representation + atom
conservation architecture. Validates that:

  (1) A vector field Psi(x) with D-dim latents can carry molecular info
      via inner products with a learned molecule-embedding library.
  (2) Atom counts can be made a strict architectural invariant of
      evolution via subspace projection.
  (3) Under a *random* (non-physical) evolution step, after projection,
      atom totals are preserved to machine precision.

What this is NOT:
  - Not the real Lagrangian (just a placeholder random step).
  - Not the adaptive octree (uniform grid).
  - Not real chemistry (synthetic test molecules).
  - No training.

Design:
  * Psi: (Nx, Ny, Nz, D) real-valued field.
  * Molecule library: M molecules, each with a D-dim embedding e_m
    and an atom-count vector a_m in R^K (K atom types: C,H,O,N,P,S).
  * Concentration of molecule m at point x: c_m(x) = < Psi(x), e_m >.
    (Softplus'd to ensure non-negative concentrations when reporting;
     for the linear conservation algebra we use the raw inner product.)
  * Total atom of type k (global): A_k = sum_x sum_m a_{m,k} c_m(x) * dV.

The atom-conservation projector:
  Define the linear map F: Psi -> A in R^K as
      A_k = dV * sum_x sum_m a_{m,k} < Psi(x), e_m >
          = dV * sum_x < Psi(x), v_k >
    where v_k = sum_m a_{m,k} e_m  (a D-vector, the "atom-k direction"
    in latent space).
  So atom conservation = "integral of <Psi(x), v_k> is constant for all k."

  Given an update DPsi, the component that changes A_k is its projection
  along the subspace spanned by {v_k}. We PROJECT OUT that component:
      DPsi_allowed = DPsi - sum_k alpha_k(DPsi) * v_k
  where alpha_k are chosen so that F(DPsi_allowed) = 0. This is a small
  K-dim linear system at each step; K=6 is tiny.

  The projection is applied AFTER computing DPsi. After projection, the
  atom totals are conserved to machine precision.

Test:
  * Seed Psi with a random, atom-consistent initial state.
  * Record initial atom totals A0.
  * Apply N=1000 random evolution steps with projection.
  * Verify A_t == A0 to floating-point precision at every step.
  * Separately verify that WITHOUT projection, atom totals drift.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List


# =============================================================================
# Atom types we track (order matters; indices used throughout)
# =============================================================================
ATOM_TYPES = ["C", "H", "O", "N", "P", "S"]
K = len(ATOM_TYPES)   # = 6


# =============================================================================
# Molecule library
# =============================================================================
@dataclass
class Molecule:
    name: str
    atom_count: np.ndarray   # shape (K,), integer counts of C,H,O,N,P,S
    embedding: np.ndarray    # shape (D,), learned in reality; here random fixed


def make_test_library(D: int, rng: np.random.Generator) -> List[Molecule]:
    """
    A tiny synthetic library with real molecular formulas for a handful of
    small molecules that appear in core metabolism.

    Atom counts from known chemistry (PubChem):
      glucose        C6H12O6
      pyruvate       C3H4O3    (pyruvic acid, neutral form for simplicity)
      ATP            C10H16N5O13P3
      water          H2O
      CO2            CO2
      ammonia        NH3
      oxygen (O2)    O2
      phosphate      H3PO4     (phosphoric acid, neutral form)
    """
    mols_spec = [
        #           C, H,  O,  N, P, S
        ("glucose",  [6, 12,  6,  0, 0, 0]),
        ("pyruvate", [3,  4,  3,  0, 0, 0]),
        ("ATP",      [10, 16, 13, 5, 3, 0]),
        ("water",    [0,  2,  1,  0, 0, 0]),
        ("CO2",      [1,  0,  2,  0, 0, 0]),
        ("NH3",      [0,  3,  0,  1, 0, 0]),
        ("O2",       [0,  0,  2,  0, 0, 0]),
        ("H3PO4",    [0,  3,  4,  0, 1, 0]),
        ("cysteine", [3,  7,  2,  1, 0, 1]),   # amino acid: sulfur-containing
    ]
    mols = []
    for name, atoms in mols_spec:
        emb = rng.standard_normal(D).astype(np.float64)
        emb /= np.linalg.norm(emb)   # unit-norm embedding
        mols.append(Molecule(name=name,
                              atom_count=np.array(atoms, dtype=np.float64),
                              embedding=emb))
    return mols


# =============================================================================
# Atom-direction computation
# =============================================================================
def atom_directions(library: List[Molecule]) -> np.ndarray:
    """
    For each atom type k, build v_k = sum_m a_{m,k} * e_m.
    Returns array of shape (K, D). The subspace span{v_0,...,v_{K-1}}
    is the "atom-changing" subspace of Psi.
    """
    D = library[0].embedding.shape[0]
    V = np.zeros((K, D))
    for m in library:
        for k in range(K):
            V[k] += m.atom_count[k] * m.embedding
    return V


def atom_projection_operator(V: np.ndarray, tol: float = 1e-10):
    """
    Build the D x D projection matrix P_atom onto the span of rows of V
    (each row v_k is the 'atom-k direction' in latent space).

    Handles the case where some atom types are not represented in the library
    (v_k = 0): we only project onto the non-trivial directions. Uses SVD for
    robustness when V has non-trivial null directions.

    Returns (P_atom, P_perp, active_atoms) where:
      P_atom: (D,D) projector onto the atom-changing subspace
      P_perp: (D,D) = I - P_atom, atom-conserving projector
      active_atoms: list of atom-type indices actually present in the library
    """
    # Identify which atom rows are non-trivial
    active_atoms = [k for k in range(V.shape[0]) if np.linalg.norm(V[k]) > tol]
    if not active_atoms:
        raise ValueError("Library contains no molecules with any tracked atoms.")

    V_active = V[active_atoms]                     # (K', D)
    # SVD gives a stable basis even if rows of V_active are linearly dependent
    U_svd, S_sv, _ = np.linalg.svd(V_active.T, full_matrices=False)
    # keep only directions with meaningful singular value
    keep = S_sv > tol * S_sv.max()
    U_basis = U_svd[:, keep]                        # (D, r), orthonormal columns
    P_atom = U_basis @ U_basis.T                    # (D, D) orthogonal projector
    D = V.shape[1]
    P_perp = np.eye(D) - P_atom
    return P_atom, P_perp, active_atoms


# =============================================================================
# Cell state: the field Psi on a uniform 3D grid
# =============================================================================
@dataclass
class CellState:
    Nx: int
    Ny: int
    Nz: int
    L: float         # cell box size (arbitrary units)
    Psi: np.ndarray  # shape (Nx, Ny, Nz, D)

    @property
    def D(self):
        return self.Psi.shape[-1]

    @property
    def dV(self):
        return (self.L / self.Nx) * (self.L / self.Ny) * (self.L / self.Nz)


def seed_state(Nx, Ny, Nz, D, rng, amplitude=1.0) -> CellState:
    """Random initial Psi, shape (Nx,Ny,Nz,D)."""
    Psi = amplitude * rng.standard_normal((Nx, Ny, Nz, D)).astype(np.float64)
    return CellState(Nx=Nx, Ny=Ny, Nz=Nz, L=1.0, Psi=Psi)


# =============================================================================
# Atom totals
# =============================================================================
def atom_totals(state: CellState, V: np.ndarray) -> np.ndarray:
    """
    A_k = dV * sum_x < Psi(x), v_k >   for each k.
    V shape (K, D), Psi shape (Nx,Ny,Nz,D).
    Returns shape (K,).
    """
    # contract Psi with each v_k then sum spatially
    # result[k] = sum_{x,y,z} Psi(x,y,z,:) . v_k   = (tensordot) then sum
    contractions = np.tensordot(state.Psi, V, axes=([-1], [-1]))  # (Nx,Ny,Nz,K)
    totals = contractions.sum(axis=(0, 1, 2)) * state.dV  # (K,)
    return totals


def molecule_concentrations(state: CellState, library: List[Molecule]
                             ) -> np.ndarray:
    """
    Return concentration field for each molecule: shape (M, Nx, Ny, Nz).
    c_m(x) = < Psi(x), e_m >. (Raw linear projection, may be negative;
    real use would apply softplus but we keep linear here for clarity.)
    """
    E = np.stack([m.embedding for m in library], axis=0)  # (M, D)
    c = np.tensordot(state.Psi, E, axes=([-1], [-1]))     # (Nx,Ny,Nz,M)
    return np.moveaxis(c, -1, 0)                          # (M,Nx,Ny,Nz)


# =============================================================================
# Evolution: random step, with and without projection
# =============================================================================
def random_step_unprojected(state: CellState, rng, step_size=0.01):
    """An entirely random update (non-physical). Used as a test input."""
    dPsi = rng.standard_normal(state.Psi.shape) * step_size
    state.Psi += dPsi


def random_step_projected(state: CellState, P_perp: np.ndarray, rng,
                           step_size=0.01):
    """
    Random update with atom-conservation projection:
    dPsi -> (P_perp applied to last axis of dPsi).
    After projection, integral of <Psi, v_k> over space is unchanged.
    """
    dPsi = rng.standard_normal(state.Psi.shape) * step_size
    # project last axis through P_perp: dPsi_proj[...,:] = dPsi[...,:] @ P_perp^T
    # (P_perp is symmetric since it's an orthogonal projector, so P_perp.T = P_perp)
    dPsi_proj = dPsi @ P_perp
    state.Psi += dPsi_proj


# =============================================================================
# Main test
# =============================================================================
def run_test():
    D = 32            # latent dimension
    Nx = Ny = Nz = 8  # tiny cell grid (512 voxels)
    n_steps = 1000
    step_size = 0.05

    print("=" * 70)
    print("DMVC Prototype P0 — Atom conservation test")
    print("=" * 70)
    print(f"Grid: {Nx}x{Ny}x{Nz}, latent D={D}, steps={n_steps}")
    print(f"Atom types tracked: {ATOM_TYPES}")

    rng = np.random.default_rng(0)

    # Build library, compute atom directions and projection operator
    library = make_test_library(D, rng)
    print(f"\nLibrary: {len(library)} molecules")
    for m in library:
        formula = "".join(f"{t}{int(n)}" if n > 0 else ""
                           for t, n in zip(ATOM_TYPES, m.atom_count))
        print(f"  {m.name:10s} {formula}")

    V = atom_directions(library)                     # (K, D)
    print(f"\nAtom-direction matrix V shape: {V.shape}")
    print(f"Rank of V: {np.linalg.matrix_rank(V)} (expected 6)")

    P_atom, P_perp, active_atoms = atom_projection_operator(V)
    print(f"Projection matrices built.")
    print(f"  Active atom types (nonzero in library): "
          f"{[ATOM_TYPES[k] for k in active_atoms]}")
    print(f"  Rank of atom subspace: {int(round(np.trace(P_atom)))} "
          f"(expected {len(active_atoms)})")

    # -----------------------------------------------------------------------
    # Experiment 1: evolution WITHOUT projection  ->  atoms drift
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Experiment 1: random evolution WITHOUT projection")
    print("-" * 70)
    state = seed_state(Nx, Ny, Nz, D, rng)
    A0 = atom_totals(state, V)
    print("Initial atom totals:")
    for k, atom in enumerate(ATOM_TYPES):
        print(f"  {atom}: {A0[k]:+.6f}")

    rng_evolution = np.random.default_rng(1)
    drifts = []
    for step in range(n_steps):
        random_step_unprojected(state, rng_evolution, step_size=step_size)
        A = atom_totals(state, V)
        drifts.append(np.max(np.abs(A - A0)))

    print(f"\nMax drift in any atom total, over {n_steps} steps: "
          f"{max(drifts):.6e}")
    print("Expected: O(step_size * sqrt(n_steps)) ~ "
          f"{step_size * np.sqrt(n_steps) * np.linalg.norm(V):.3f}")
    print(f"Final atom totals:")
    for k, atom in enumerate(ATOM_TYPES):
        print(f"  {atom}: {A[k]:+.6f}   drift = {A[k]-A0[k]:+.6e}")

    # -----------------------------------------------------------------------
    # Experiment 2: evolution WITH projection  ->  atoms conserved exactly
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Experiment 2: random evolution WITH atom-conservation projection")
    print("-" * 70)
    state = seed_state(Nx, Ny, Nz, D, rng)
    A0 = atom_totals(state, V)
    print("Initial atom totals:")
    for k, atom in enumerate(ATOM_TYPES):
        print(f"  {atom}: {A0[k]:+.6f}")

    rng_evolution = np.random.default_rng(1)
    drifts = []
    for step in range(n_steps):
        random_step_projected(state, P_perp, rng_evolution, step_size=step_size)
        A = atom_totals(state, V)
        drifts.append(np.max(np.abs(A - A0)))

    max_drift = max(drifts)
    print(f"\nMax drift in any atom total, over {n_steps} steps: {max_drift:.6e}")
    print(f"(Expected: ~ machine precision ~ 1e-12 or below)")
    print(f"Final atom totals:")
    for k, atom in enumerate(ATOM_TYPES):
        print(f"  {atom}: {A[k]:+.6f}   drift = {A[k]-A0[k]:+.6e}")

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    tol = 1e-8
    if max_drift < tol:
        print(f"PASS: atom conservation holds to < {tol:.0e} under random evolution.")
        print("Architecture foundation is sound.")
    else:
        print(f"FAIL: atom conservation violated (max drift {max_drift:.2e}).")

    # -----------------------------------------------------------------------
    # Bonus: molecule-concentration report
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Bonus: per-molecule mean concentration in the final state")
    print("-" * 70)
    c = molecule_concentrations(state, library)
    for m, cm in zip(library, c):
        print(f"  {m.name:10s}  <c> = {cm.mean():+.5f}   std = {cm.std():.5f}")

    return max_drift, A0, A


if __name__ == "__main__":
    run_test()
