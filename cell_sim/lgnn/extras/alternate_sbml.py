"""Tier-3 prep: alternate SBML model (iMB155 variant).

The dataset includes a second SBML model:
  input_data/iMB155_noUnqATP_lipdiomics_wPUNP5_noNBtransport.xml

This isn't a feature - it's an alternative chemistry rule book. The
Syn3A_updated.xml (the default) and this iMB155 variant differ in
which reactions are included and their stoichiometry. Loading this
instead of the default gives the PINNHead a different mass-balance
constraint.

Use cases:
  - Ablation study: train with iMB155, see if metrics differ
  - Sensitivity analysis: which predictions are SBML-dependent

Activation: set cfg.sbml_path_for_S to point at this file instead of
Syn3A_updated.xml. No new loader needed - the existing
load_stoichiometric_matrix_for_pinn handles any SBML XML.

This module just exists to document the path constant and provide a
sanity-check helper for the file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


ALTERNATE_SBML_FILENAME = 'iMB155_noUnqATP_lipdiomics_wPUNP5_noNBtransport.xml'


def find_alternate_sbml_path(input_data_dir: Path | str) -> Optional[Path]:
    """Look for the alternate SBML file in the conventional location.

    Returns the Path if found, else None.
    """
    p = Path(input_data_dir) / ALTERNATE_SBML_FILENAME
    return p if p.exists() else None


def describe_alternate_sbml(xml_path: Path | str) -> dict:
    """Return summary stats about an SBML file without fully loading it.

    Useful pre-flight check before swapping cfg.sbml_path_for_S.
    """
    import xml.etree.ElementTree as ET

    p = Path(xml_path)
    if not p.exists():
        return {'exists': False, 'path': str(p)}

    try:
        tree = ET.parse(p)
        root = tree.getroot()
        ns_m = root.tag.split('}')[0].lstrip('{') if '}' in root.tag else ''
        ns = f'{{{ns_m}}}' if ns_m else ''
        n_species = sum(1 for _ in root.iter(f'{ns}species'))
        n_rxns = sum(1 for _ in root.iter(f'{ns}reaction'))
        return {
            'exists': True,
            'path': str(p),
            'n_species': n_species,
            'n_reactions': n_rxns,
            'size_bytes': p.stat().st_size,
        }
    except Exception as e:
        return {'exists': True, 'path': str(p), 'error': str(e)}
