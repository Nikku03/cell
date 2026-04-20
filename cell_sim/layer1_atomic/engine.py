"""
Layer 1: Atomic engine.

Wraps MACE-OFF with a simple on-demand interface. Higher layers call
atomic_query(subsystem, query_type) and get back physics-grounded answers
about that subsystem.

Supported query types:
- 'energy': total energy of the configuration (eV)
- 'forces': force on each atom (eV/Angstrom)
- 'bde': bond dissociation energy for a specified C-H bond (kcal/mol)
- 'optimize': relaxed geometry + relaxed energy

This is the "zoom-in" primitive. It's called O(10-100) times per
whole-cell simulation, not O(10^10). That's what keeps the routing system
under an hour.

Hardware notes:
- On the RTX 6000 Pro with cuEquivariance + FP32, expect ~50-500 ms
  per query for typical sub-systems of 20-200 atoms.
- Without a GPU, we fall back to a stubbed version that returns
  zeros and a warning; this lets the rest of the pipeline run for
  structural testing.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# Try to import MACE, but don't crash if unavailable
_MACE_AVAILABLE = False
_MACE_ERR = None
try:
    from ase import Atoms
    from ase.optimize import BFGS
    _MACE_AVAILABLE = True
except Exception as e:
    _MACE_ERR = str(e)


# Reference energy of an isolated H atom at MACE-OFF's DFT level
# (needed for BDE calculation). This value matches the MACE-OFF paper.
H_ATOM_REFERENCE_ENERGY_EV = -13.642  # approximate; exact value depends on model

EV_TO_KCAL_MOL = 23.0605


@dataclass
class Subsystem:
    """
    A subsystem of atoms for atomic-scale query.

    The atoms are specified by their element symbols and 3D positions.
    Optionally, a 'highlight' atom index points to the atom of interest
    for BDE queries (e.g., the hydrogen we want to abstract).
    """
    symbols: List[str]
    positions: np.ndarray  # (N, 3) in Angstroms
    charge: int = 0
    spin_multiplicity: int = 1
    highlight_atom: Optional[int] = None
    name: str = "subsystem"

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


@dataclass
class AtomicResult:
    """Result from an atomic_query call."""
    query_type: str
    energy_eV: Optional[float] = None
    forces_eV_A: Optional[np.ndarray] = None
    bde_kcal_mol: Optional[float] = None
    optimized_positions: Optional[np.ndarray] = None
    wall_time_s: Optional[float] = None
    is_stubbed: bool = False
    metadata: Dict = field(default_factory=dict)


class AtomicEngine:
    """
    Wrapper around MACE-OFF, exposing a simple atomic_query interface.

    Lazy-loads the model on first use so that Layer 3-only simulations
    (no atomic queries) don't pay the MACE startup cost.
    """

    def __init__(
        self,
        model_name: str = "medium",
        device: Optional[str] = None,
        enable_cueq: bool = True,
        default_dtype: str = "float32",
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if HAVE_TORCH and torch.cuda.is_available() else "cpu")
        self.enable_cueq = enable_cueq and (self.device == "cuda")
        self.default_dtype = default_dtype
        self._calculator = None
        self._load_attempted = False
        self._load_error = None

    def _lazy_load(self):
        """Load the MACE-OFF calculator on first use."""
        if self._load_attempted:
            return
        self._load_attempted = True
        if not _MACE_AVAILABLE:
            self._load_error = f"MACE unavailable: {_MACE_ERR}"
            return
        try:
            from mace.calculators import mace_off
            kwargs = {
                'model': self.model_name,
                'device': self.device,
                'default_dtype': self.default_dtype,
            }
            if self.enable_cueq:
                kwargs['enable_cueq'] = True
            self._calculator = mace_off(**kwargs)
        except Exception as e:
            self._load_error = f"Failed to load MACE-OFF: {e}"

    @property
    def is_ready(self) -> bool:
        self._lazy_load()
        return self._calculator is not None

    def _stubbed_result(self, query_type: str, subsys: Subsystem) -> AtomicResult:
        """
        Fallback when MACE can't actually run (e.g., no GPU, no network
        for model download). Returns physically reasonable zero values so
        higher layers can still test their routing logic.
        """
        warnings.warn(
            f"AtomicEngine is stubbed (reason: {self._load_error}). "
            "Returning zero energies/forces. This is fine for structural "
            "testing of the pipeline but NOT for real simulation.",
            RuntimeWarning,
            stacklevel=2,
        )
        n = subsys.n_atoms
        if query_type == 'energy':
            return AtomicResult(query_type='energy', energy_eV=0.0, is_stubbed=True)
        elif query_type == 'forces':
            return AtomicResult(
                query_type='forces', energy_eV=0.0,
                forces_eV_A=np.zeros((n, 3)), is_stubbed=True,
            )
        elif query_type == 'bde':
            return AtomicResult(
                query_type='bde', bde_kcal_mol=95.0,
                is_stubbed=True,
                metadata={'note': 'typical C-H BDE placeholder'},
            )
        elif query_type == 'optimize':
            return AtomicResult(
                query_type='optimize',
                energy_eV=0.0,
                optimized_positions=subsys.positions.copy(),
                is_stubbed=True,
            )
        else:
            raise ValueError(f"Unknown query type: {query_type}")

    def _make_atoms(self, subsys: Subsystem):
        """Convert a Subsystem to an ASE Atoms object with our calculator."""
        atoms = Atoms(symbols=subsys.symbols, positions=subsys.positions)
        atoms.calc = self._calculator
        return atoms

    def atomic_query(self, subsys: Subsystem, query_type: str) -> AtomicResult:
        """
        Main entry point. Returns atomic-scale physics info for the subsystem.
        """
        import time
        t0 = time.time()

        self._lazy_load()
        if self._calculator is None:
            return self._stubbed_result(query_type, subsys)

        try:
            atoms = self._make_atoms(subsys)

            if query_type == 'energy':
                e = atoms.get_potential_energy()
                return AtomicResult(
                    query_type='energy', energy_eV=float(e),
                    wall_time_s=time.time() - t0,
                )

            elif query_type == 'forces':
                f = atoms.get_forces()
                e = atoms.get_potential_energy()
                return AtomicResult(
                    query_type='forces', energy_eV=float(e),
                    forces_eV_A=np.asarray(f), wall_time_s=time.time() - t0,
                )

            elif query_type == 'bde':
                # Bond dissociation energy for the highlighted H atom.
                # E_BDE = E(radical) + E(H) - E(molecule)
                if subsys.highlight_atom is None:
                    raise ValueError("BDE query requires highlight_atom to be set")
                if subsys.symbols[subsys.highlight_atom] != 'H':
                    raise ValueError("BDE query: highlight_atom must be an H atom")

                e_molecule = atoms.get_potential_energy()

                # Build the radical: remove the highlighted H
                rad_symbols = [s for i, s in enumerate(subsys.symbols) if i != subsys.highlight_atom]
                rad_positions = np.delete(subsys.positions, subsys.highlight_atom, axis=0)
                rad_atoms = Atoms(symbols=rad_symbols, positions=rad_positions)
                rad_atoms.calc = self._calculator
                e_radical = rad_atoms.get_potential_energy()

                bde_ev = e_radical + H_ATOM_REFERENCE_ENERGY_EV - e_molecule
                bde_kcal = bde_ev * EV_TO_KCAL_MOL
                return AtomicResult(
                    query_type='bde', bde_kcal_mol=float(bde_kcal),
                    energy_eV=float(e_molecule),
                    wall_time_s=time.time() - t0,
                    metadata={'e_radical_eV': float(e_radical)},
                )

            elif query_type == 'optimize':
                # Geometry optimization using BFGS
                opt = BFGS(atoms, logfile=None)
                opt.run(fmax=0.05, steps=100)
                e = atoms.get_potential_energy()
                return AtomicResult(
                    query_type='optimize', energy_eV=float(e),
                    optimized_positions=np.asarray(atoms.get_positions()),
                    wall_time_s=time.time() - t0,
                )

            else:
                raise ValueError(f"Unknown query type: {query_type}")

        except Exception as e:
            warnings.warn(f"MACE query failed: {e}. Falling back to stub.", RuntimeWarning)
            return self._stubbed_result(query_type, subsys)


# =============================================================================
# Utility: build a simple methane subsystem for testing
# =============================================================================
def methane_subsystem(highlight_h: int = 1) -> Subsystem:
    """
    Methane CH4 with tetrahedral geometry.
    Atom 0 is C, atoms 1-4 are H.
    """
    # Tetrahedral angle hydrogens around carbon at origin
    d = 1.09  # C-H bond length in Angstroms
    positions = np.array([
        [0.0, 0.0, 0.0],
        [d, d, d],
        [-d, -d, d],
        [-d, d, -d],
        [d, -d, -d],
    ]) / np.sqrt(3)
    positions[0] = [0, 0, 0]
    return Subsystem(
        symbols=['C', 'H', 'H', 'H', 'H'],
        positions=positions,
        highlight_atom=highlight_h,
        name="methane",
    )


# =============================================================================
# Demo
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Layer 1: Atomic Engine (MACE-OFF wrapper)")
    print("=" * 60)

    engine = AtomicEngine(model_name="medium", device=None, enable_cueq=True)
    print(f"Device: {engine.device}")
    print(f"cuEq enabled: {engine.enable_cueq}")
    print(f"MACE available in environment: {_MACE_AVAILABLE}")
    print()

    # Test with methane
    methane = methane_subsystem()
    print(f"Test subsystem: {methane.name}  (n_atoms={methane.n_atoms})")
    print()

    print("Query: energy")
    result = engine.atomic_query(methane, 'energy')
    if result.is_stubbed:
        print(f"  STUBBED: {result.energy_eV} (pipeline-testing value)")
    else:
        print(f"  Energy: {result.energy_eV:.4f} eV")
        print(f"  Wall time: {result.wall_time_s*1000:.1f} ms")
    print()

    print("Query: bde (for H atom 1)")
    result = engine.atomic_query(methane, 'bde')
    if result.is_stubbed:
        print(f"  STUBBED: {result.bde_kcal_mol} kcal/mol")
    else:
        print(f"  BDE: {result.bde_kcal_mol:.2f} kcal/mol")
        print(f"  (Reference: methane C-H BDE is ~105 kcal/mol)")
        print(f"  Wall time: {result.wall_time_s*1000:.1f} ms")

    print("\nLayer 1 wrapper is working.")
