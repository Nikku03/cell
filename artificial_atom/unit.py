"""Richer per-atom unit: the AtomState.

Where `ArtificialAtom` (atom.py) was a thin data carrier with most fields
unused, `AtomState` is what the v2 (multi-mode, stateful) network actually
reads and writes:

  - persistent hidden vector `h` that survives between forward passes
    and is updated by a GRU each call (this is the "memory")
  - the previous `h` exposed as `prev_h` so an analyst can compare
  - chemistry context: hybridisation, formal charge, aromaticity, ring
    membership, hydrogen-bond donor/acceptor flags
  - partial charge (used as the source for the polarisation channel)
  - per-mode energy contributions written by the output heads
  - uncertainty fields populated by the ensemble at inference time

These are still plain Python data; torch tensors are managed by the
network. The unit holds the *interface*, not the math.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Categorical hybridisation labels - matches RDKit's enum order roughly.
HYB_NONE = 0
HYB_SP = 1
HYB_SP2 = 2
HYB_SP3 = 3
HYB_AROMATIC = 4
N_HYB = 5

# Bond-type indices (used as labels on bond edges).
BOND_NONE = 0
BOND_SINGLE = 1
BOND_DOUBLE = 2
BOND_TRIPLE = 3
BOND_AROMATIC = 4
N_BOND = 5

# Interaction-mode indices used by the network.
MODE_BONDED = 0          # covalent-bond channel (uses bond graph)
MODE_NONBOND = 1         # short-range non-bonded channel (steric + vdW)
MODE_ELECTROSTATIC = 2   # long-range, partial-charge-modulated
N_MODES = 3


@dataclass
class AtomState:
    """Everything one atom carries between modules in the v2 pipeline."""

    # ---- identity & geometry (inputs the model reads) ----
    Z: int                                  # atomic number
    position: tuple[float, float, float]
    mass: float | None = None

    # ---- chemistry context (inputs the model reads) ----
    hybridisation: int = HYB_NONE           # see HYB_* constants
    formal_charge: int = 0
    aromatic: bool = False
    in_ring: bool = False
    h_donor: bool = False
    h_acceptor: bool = False
    partial_charge: float = 0.0             # used by the electrostatic channel

    # ---- persistent state (memory) ----
    state: list[float] = field(default_factory=list)        # current h
    prev_state: list[float] = field(default_factory=list)   # h before last call
    step_count: int = 0                     # number of forward passes seen

    # ---- per-mode energy outputs (filled by output heads) ----
    e_bonded: float = 0.0
    e_nonbond: float = 0.0
    e_electrostatic: float = 0.0
    local_energy: float = 0.0               # sum of the three above

    # ---- uncertainty (filled by ensemble at inference) ----
    energy_mean: float = 0.0
    energy_std: float = 0.0


@dataclass
class Bond:
    """A single covalent bond between two atoms."""
    i: int                       # atom index (lower)
    j: int                       # atom index (higher)
    bond_type: int = BOND_SINGLE

    def normalised(self) -> "Bond":
        if self.i > self.j:
            return Bond(i=self.j, j=self.i, bond_type=self.bond_type)
        return self
