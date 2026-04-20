"""
Full kinetic parameter loader for Syn3A.

Unlike the simple `load_kcats()` that only extracted forward k_cat, this
module extracts for each reaction:
  - Forward k_cat (substrate catalytic rate constant)
  - Reverse k_cat (product catalytic rate constant) — 0 if irreversible
  - K_m values for each species (Michaelis-Menten constant)
  - Enzyme complex hint (the gene/complex ID)

This enables proper reversible Michaelis-Menten propensity calculations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np


DATA_ROOT = Path(__file__).parent.parent / 'data' / 'Minimal_Cell_ComplexFormation' / 'input_data'


@dataclass
class ReactionKinetics:
    """Full kinetic parameter set for one reaction."""
    name: str
    subsystem: str = ''
    kcat_forward: float = 0.0      # 1/s, substrate direction
    kcat_reverse: float = 0.0      # 1/s, product direction (0 if irreversible)
    Km: Dict[str, float] = field(default_factory=dict)  # species_id → K_m in mM
    enzyme_hint: str = ''           # e.g., 'P_0445' or 'ATPSynthase'
    is_reversible: bool = False

    def is_forward_only(self) -> bool:
        return self.kcat_reverse == 0.0


def load_all_kinetics() -> Dict[str, ReactionKinetics]:
    """
    Load full kinetic parameters from all metabolic sheets.

    Returns dict keyed by reaction short name (e.g., 'PGI', 'PYK').
    """
    out: Dict[str, ReactionKinetics] = {}
    sheets = ['Central', 'Nucleotide', 'Lipid', 'Cofactor', 'Transport']

    for sheet in sheets:
        df = pd.read_excel(DATA_ROOT / 'kinetic_params.xlsx', sheet_name=sheet)
        current: Optional[ReactionKinetics] = None
        for _, row in df.iterrows():
            name = str(row.get('Reaction Name', '')).strip()
            ptype = str(row.get('Parameter Type', '')).strip()
            species = str(row.get('Related Species', '')).strip()
            val = row.get('Value', '')
            units = str(row.get('Units', '')).strip().lower()

            if name and name != 'nan':
                # Switch to a new reaction if the name changed
                if current is None or current.name != name:
                    # Save any previous
                    if current is not None and current.name:
                        out.setdefault(current.name, current)
                    current = ReactionKinetics(name=name, subsystem=sheet)
            if current is None:
                continue

            # Parse the parameter
            if ptype == 'Eff Enzyme Count':
                current.enzyme_hint = str(val) if val == val else ''  # NaN check
            elif ptype == 'Substrate Catalytic Rate Constant':
                try:
                    current.kcat_forward = float(val)
                except (ValueError, TypeError):
                    pass
            elif ptype == 'Product Catalytic Rate Constant':
                try:
                    current.kcat_reverse = float(val)
                    if current.kcat_reverse > 0:
                        current.is_reversible = True
                except (ValueError, TypeError):
                    pass
            elif ptype == 'Michaelis Menten Constant':
                if species and species != 'nan':
                    try:
                        current.Km[species] = float(val)
                    except (ValueError, TypeError):
                        pass

        if current is not None and current.name:
            out.setdefault(current.name, current)

    return out


# ============================================================================
# Medium loader — extracellular species with fixed concentrations
# ============================================================================
@dataclass
class MediumSpecies:
    species_id: str       # e.g., 'M_glc__D_e'
    name: str
    conc_mM: float


def load_medium() -> Dict[str, MediumSpecies]:
    """Load the simulation medium composition."""
    df = pd.read_excel(DATA_ROOT / 'initial_concentrations.xlsx',
                        sheet_name='Simulation Medium')
    out = {}
    for _, row in df.iterrows():
        met_id = str(row.get('Met ID', '')).strip()
        if not met_id or met_id == 'nan':
            continue
        sid = f'M_{met_id}' if not met_id.startswith('M_') else met_id
        try:
            conc = float(row.get('Conc (mM)', 0))
        except (ValueError, TypeError):
            conc = 0.0
        out[sid] = MediumSpecies(
            species_id=sid,
            name=str(row.get('Metabolite name', '')),
            conc_mM=conc,
        )
    return out


# ============================================================================
# Demo
# ============================================================================
if __name__ == "__main__":
    kinetics = load_all_kinetics()
    print(f"Loaded {len(kinetics)} reaction kinetics entries")

    n_reversible = sum(1 for k in kinetics.values() if k.is_reversible)
    n_forward = sum(1 for k in kinetics.values() if not k.is_reversible)
    print(f"  Reversible:   {n_reversible}")
    print(f"  Irreversible: {n_forward}")

    print("\nSample reactions (first 5):")
    for name, k in list(kinetics.items())[:5]:
        print(f"\n  {name} ({k.subsystem}):")
        print(f"    enzyme: {k.enzyme_hint}")
        print(f"    k_cat fwd: {k.kcat_forward:>10.3f} /s")
        print(f"    k_cat rev: {k.kcat_reverse:>10.3f} /s (reversible: {k.is_reversible})")
        print(f"    K_m values:")
        for sp, km in k.Km.items():
            print(f"      {sp:20s}  K_m = {km:.4f} mM")

    print("\n" + "=" * 60)
    medium = load_medium()
    print(f"\nLoaded {len(medium)} medium species")
    print("\nTop 10 by concentration:")
    for sid, m in sorted(medium.items(), key=lambda x: -x[1].conc_mM)[:10]:
        print(f"  {sid:20s}  {m.conc_mM:>10.3f} mM  ({m.name})")
