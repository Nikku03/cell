"""
SBML parser for the Syn3A_updated.xml metabolic model.

Extracts:
  - Species: metabolite IDs (e.g., 'M_atp_c'), names, compartments
  - Reactions: reactants, products with stoichiometries, gene associations
  - Reversibility

This is a minimal SBML-FBC reader — just enough for our needs. We don't
parse flux bounds or objectives.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import xml.etree.ElementTree as ET
import re


SBML_NS = 'http://www.sbml.org/sbml/level3/version1/core'
FBC_NS = 'http://www.sbml.org/sbml/level3/version1/fbc/version2'


@dataclass
class SBMLSpecies:
    species_id: str            # e.g., 'M_atp_c'
    name: str
    compartment: str           # 'c' or 'e'
    formula: str = ''          # chemical formula
    charge: int = 0


@dataclass
class SBMLReaction:
    reaction_id: str                          # e.g., 'R_PGI'
    name: str
    reactants: Dict[str, float] = field(default_factory=dict)  # species_id -> stoichiometry
    products:  Dict[str, float] = field(default_factory=dict)
    reversible: bool = True
    gene_associations: List[str] = field(default_factory=list)  # gene IDs (e.g. 'G_MMSYN1_0445')

    @property
    def short_name(self) -> str:
        """Get just 'PGI' from 'R_PGI'."""
        return self.reaction_id[2:] if self.reaction_id.startswith('R_') else self.reaction_id


@dataclass
class SBMLModel:
    species: Dict[str, SBMLSpecies] = field(default_factory=dict)
    reactions: Dict[str, SBMLReaction] = field(default_factory=dict)
    compartments: List[str] = field(default_factory=list)

    def reactions_by_short_name(self) -> Dict[str, SBMLReaction]:
        return {r.short_name: r for r in self.reactions.values()}


def parse_sbml(path: Path) -> SBMLModel:
    """Parse the Syn3A SBML file."""
    tree = ET.parse(path)
    root = tree.getroot()
    model = SBMLModel()

    ns = {'sbml': SBML_NS, 'fbc': FBC_NS}
    m = root.find('sbml:model', ns)
    if m is None:
        raise RuntimeError(f"No <model> in {path}")

    # Compartments
    for c in m.findall('sbml:listOfCompartments/sbml:compartment', ns):
        cid = c.get('id')
        if cid:
            model.compartments.append(cid)

    # Species
    for s in m.findall('sbml:listOfSpecies/sbml:species', ns):
        sid = s.get('id')
        if not sid:
            continue
        try:
            charge = int(s.get(f'{{{FBC_NS}}}charge', '0'))
        except (ValueError, TypeError):
            charge = 0
        model.species[sid] = SBMLSpecies(
            species_id=sid,
            name=s.get('name', ''),
            compartment=s.get('compartment', ''),
            formula=s.get(f'{{{FBC_NS}}}chemicalFormula', ''),
            charge=charge,
        )

    # Reactions
    for rxn in m.findall('sbml:listOfReactions/sbml:reaction', ns):
        rid = rxn.get('id')
        if not rid:
            continue
        r = SBMLReaction(
            reaction_id=rid,
            name=rxn.get('name', ''),
            reversible=(rxn.get('reversible', 'false').lower() == 'true'),
        )
        for sref in rxn.findall('sbml:listOfReactants/sbml:speciesReference', ns):
            sp = sref.get('species')
            try:
                stoich = float(sref.get('stoichiometry', '1'))
            except (ValueError, TypeError):
                stoich = 1.0
            if sp:
                r.reactants[sp] = stoich
        for sref in rxn.findall('sbml:listOfProducts/sbml:speciesReference', ns):
            sp = sref.get('species')
            try:
                stoich = float(sref.get('stoichiometry', '1'))
            except (ValueError, TypeError):
                stoich = 1.0
            if sp:
                r.products[sp] = stoich
        # Gene associations — walk the fbc:geneProductAssociation tree
        gpa = rxn.find(f'{{{FBC_NS}}}geneProductAssociation')
        if gpa is not None:
            for gpref in gpa.iter(f'{{{FBC_NS}}}geneProductRef'):
                g = gpref.get(f'{{{FBC_NS}}}geneProduct')
                if g:
                    r.gene_associations.append(g)
        model.reactions[rid] = r

    return model


# =============================================================================
# Map SBML gene IDs (G_MMSYN1_0445) → our locus tags (JCVISYN3A_0445)
# =============================================================================
def sbml_gene_to_locus(gene_id: str) -> Optional[str]:
    """Convert SBML gene ID to Syn3A locus tag."""
    # G_MMSYN1_0445 → JCVISYN3A_0445
    m = re.match(r'G_MMSYN1_(\d+)', gene_id)
    if m:
        return f'JCVISYN3A_{m.group(1).zfill(4)}'
    # G_JCVISYN3A_0445 → JCVISYN3A_0445
    m = re.match(r'G_JCVISYN3A_(\d+)', gene_id)
    if m:
        return f'JCVISYN3A_{m.group(1).zfill(4)}'
    return None


if __name__ == "__main__":
    path = Path(__file__).parent.parent / 'data' / 'Minimal_Cell_ComplexFormation' / 'input_data' / 'Syn3A_updated.xml'
    print(f"Parsing {path.name}...")
    model = parse_sbml(path)
    print(f"  Species: {len(model.species)}")
    print(f"  Reactions: {len(model.reactions)}")
    print(f"  Compartments: {model.compartments}")

    print("\nSample species:")
    for sid in list(model.species.keys())[:5]:
        s = model.species[sid]
        print(f"  {sid}  '{s.name}'  [{s.compartment}]  {s.formula}")

    print("\nSample reactions:")
    for rid in list(model.reactions.keys())[:5]:
        r = model.reactions[rid]
        genes = [sbml_gene_to_locus(g) for g in r.gene_associations]
        print(f"  {r.short_name:12s} {dict(r.reactants)}  →  {dict(r.products)}")
        print(f"     genes: {genes}")
        print(f"     reversible: {r.reversible}")
