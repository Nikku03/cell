"""
Metabolite state utilities — shared infrastructure for Layer 3.

Manages metabolite counts attached to a CellState:
  - mM <-> count conversion via cell volume and Avogadro
  - Initial pool seeding from the CellSpec / SBML species list
  - Infinite-reservoir bookkeeping for buffered species (water, H+,
    extracellular nutrients)

Used by `reversible.py` (Priority 1.5, reversible Michaelis-Menten) and
`gene_expression.py` (Priority 2, central dogma). Both build rules that
call get_species_count / update_species_count to read and mutate this
state during event firing.

HISTORICAL NOTE
---------------
An earlier "Priority 1" simulator lived in this file — it built naive
forward-only stoichiometric rules via `build_coupled_catalysis_rules()`.
That simulator is deprecated: substrate pools drained to zero in ~200 ms
because reactions couldn't run backward and the medium wasn't buffered.
It's been removed in favour of `reversible.py`, which is a superset of
the behavior plus reversibility, Michaelis-Menten saturation, and
medium uptake. The name `coupled.py` is kept for import backward
compatibility; the file is now pure utilities.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from layer2_field.dynamics import CellState
from layer3_reactions.sbml_parser import SBMLModel


# ============================================================================
# Constants
# ============================================================================

AVOGADRO = 6.022e23

# Species we treat as infinite reservoirs (counts stay constant).
# Water, H+, and gases that exchange freely with the medium.
INFINITE_SPECIES = {
    'M_h2o_c', 'M_h_c', 'M_h2o_e', 'M_h_e',
    'M_co2_c', 'M_co2_e',
    'M_o2_c', 'M_o2_e',
}

# Threshold below which we track as integer counts. Above this, species
# are effectively continuous and tracking each discrete change is wasteful.
COUNTABLE_THRESHOLD = 100_000


# ============================================================================
# Unit conversion
# ============================================================================

def mM_to_count(conc_mM: float, volume_L: float) -> int:
    """Convert millimolar concentration + volume to integer molecule count."""
    return int(round(conc_mM * 1e-3 * volume_L * AVOGADRO))


def count_to_mM(count: int, volume_L: float) -> float:
    """Convert integer molecule count + volume to millimolar concentration."""
    if volume_L <= 0:
        return 0.0
    return (count / AVOGADRO) / volume_L * 1000.0


# ============================================================================
# Metabolite state initialization and access
# ============================================================================

def initialize_metabolites(state: CellState, sbml: SBMLModel,
                            cell_volume_um3: float = 0.034) -> Dict[str, int]:
    """
    Seed metabolite counts on the CellState.

    Reads initial concentrations from CellSpec.metabolites (which come
    from initial_concentrations.xlsx) and adds zero-count entries for
    any additional SBML species that will appear only as reaction
    products.

    Cell volume defaults to 0.034 μm^3, corresponding to a 200 nm-radius
    sphere (JCVI-Syn3A's measured size from cryo-ET).

    Returns the metabolite_counts dict for inspection.
    """
    volume_L = cell_volume_um3 * 1e-15  # μm^3 → L

    state.metabolite_counts = {}
    state.metabolite_volume_L = volume_L
    state.metabolite_infinite = set(INFINITE_SPECIES)

    # From CellSpec metabolites (intracellular concentrations from xlsx)
    for met_id, met in state.spec.metabolites.items():
        # Met IDs are BiGG-style ('atp_c'); SBML uses 'M_atp_c'.
        sbml_id = f'M_{met_id}' if not met_id.startswith('M_') else met_id
        count = mM_to_count(met.initial_concentration_mM, volume_L)
        state.metabolite_counts[sbml_id] = count

    # Any SBML species not covered by the xlsx starts at zero
    # (these appear as reaction products during simulation)
    for sid in sbml.species:
        if sid not in state.metabolite_counts:
            state.metabolite_counts[sid] = 0

    return state.metabolite_counts


def get_species_count(state: CellState, species_id: str) -> int:
    """Return species count. Infinite reservoirs return a large constant."""
    if species_id in state.metabolite_infinite:
        return COUNTABLE_THRESHOLD * 10  # effectively infinite for propensity
    return state.metabolite_counts.get(species_id, 0)


def update_species_count(state: CellState, species_id: str, delta: int):
    """Mutate a species count. No-op for infinite reservoirs."""
    if species_id in state.metabolite_infinite:
        return
    if species_id not in state.metabolite_counts:
        state.metabolite_counts[species_id] = 0
    state.metabolite_counts[species_id] += delta
    # Clamp negatives — shouldn't happen given propensity checks, but
    # guards against stochastic rounding at the per-event level
    if state.metabolite_counts[species_id] < 0:
        state.metabolite_counts[species_id] = 0
