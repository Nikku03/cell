"""
Layer 0: Genome → Cellular Components

Takes a genome (FASTA or GenBank) and a species identifier, and produces
a CellSpec: a data structure describing what proteins, metabolites, and
reactions the cell contains. This is the "minimum info input" — once we
have this, the higher layers can start simulating.

For real research use, you'd hook this up to UniProt, KEGG, AlphaFold DB,
and similar databases. For a 24-hour prototype, we handle:

1. FASTA/GenBank input (Biopython)
2. Simple ORF finding + protein translation
3. Function class assignment via motif matching (heuristic, not accurate,
   but sufficient for routing decisions)
4. Metabolite loading from a static species table
5. Reaction loading from a static species table

The output CellSpec is what Layers 1-3 consume.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import re


# Minimum ORF length in amino acids to be considered a real gene
MIN_ORF_LENGTH = 30


@dataclass
class Protein:
    """A protein identified from the genome."""
    gene_id: str
    sequence: str  # amino acid sequence
    length: int
    function_class: str  # enzyme / structural / transport / regulatory / unknown
    annotations: Dict[str, str] = field(default_factory=dict)
    structure_pdb_id: Optional[str] = None  # filled in if available
    uniprot_id: Optional[str] = None


@dataclass
class Metabolite:
    """A small molecule tracked by the simulator."""
    met_id: str
    name: str
    smiles: Optional[str] = None
    initial_concentration_mM: float = 0.0
    compartment: str = "cytoplasm"


@dataclass
class Reaction:
    """A biochemical reaction."""
    rxn_id: str
    name: str
    reactants: Dict[str, float]  # met_id -> stoichiometry
    products: Dict[str, float]
    catalyzed_by: Optional[str] = None  # gene_id of enzyme
    k_cat: Optional[float] = None  # if known
    K_m: Optional[Dict[str, float]] = None


@dataclass
class CellSpec:
    """Complete specification of a cell, consumed by all higher layers."""
    species: str
    cell_radius_um: float  # approximate cell size
    proteins: Dict[str, Protein] = field(default_factory=dict)
    metabolites: Dict[str, Metabolite] = field(default_factory=dict)
    reactions: Dict[str, Reaction] = field(default_factory=dict)
    genome_size_bp: int = 0

    def summary(self) -> str:
        lines = [
            f"CellSpec for {self.species}",
            f"  Genome size: {self.genome_size_bp:,} bp",
            f"  Cell radius: {self.cell_radius_um} μm",
            f"  Proteins: {len(self.proteins)}",
            f"  Metabolites: {len(self.metabolites)}",
            f"  Reactions: {len(self.reactions)}",
        ]
        if self.proteins:
            func_counts = {}
            for p in self.proteins.values():
                func_counts[p.function_class] = func_counts.get(p.function_class, 0) + 1
            lines.append("  Function class distribution:")
            for fc, n in sorted(func_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {fc}: {n}")
        return "\n".join(lines)


# =============================================================================
# ORF finding (simple, forward strand only for prototype)
# =============================================================================
GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


def translate(dna: str) -> str:
    """Translate a DNA sequence to protein. Stops at first stop codon."""
    protein = []
    dna = dna.upper()
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i+3]
        aa = GENETIC_CODE.get(codon, 'X')
        if aa == '*':
            break
        protein.append(aa)
    return ''.join(protein)


def find_orfs(sequence: str, min_length: int = MIN_ORF_LENGTH) -> List[Tuple[int, str]]:
    """
    Find open reading frames on the forward strand.

    Returns list of (start_position, protein_sequence) tuples.
    A real implementation would also search the reverse strand and handle
    circular genomes; this is the 24-hour version.
    """
    orfs = []
    seq = sequence.upper()
    for frame in range(3):
        i = frame
        while i < len(seq) - 3:
            if seq[i:i+3] == 'ATG':  # start codon
                protein = translate(seq[i:])
                if len(protein) >= min_length:
                    orfs.append((i, protein))
                    i += (len(protein) + 1) * 3  # skip past this ORF
                else:
                    i += 3
            else:
                i += 3
    return orfs


# =============================================================================
# Function class assignment (heuristic motif matching)
# =============================================================================
# These motifs are intentionally simple placeholders. A real version would use
# Pfam/HMMER or a neural classifier. But even simple motifs give us enough to
# make routing decisions.
FUNCTION_MOTIFS = {
    'enzyme': [
        r'G.GK[ST]',  # P-loop ATPase
        r'HE.H',      # zinc metalloprotease
        r'GDSL',      # lipase
        r'CXXC',      # thioredoxin
        r'[ST]GX{2,4}[GA]',  # kinase
    ],
    'transport': [
        r'GXXXG',     # transmembrane helix
        r'[FY]X{2}[FY]',  # membrane-spanning
    ],
    'regulatory': [
        r'HTH',       # helix-turn-helix (placeholder)
        r'C{4}',      # zinc finger
    ],
    'structural': [
        r'(GX{2}){3,}',  # glycine-rich structural
    ],
}


def classify_protein(sequence: str) -> str:
    """Assign a function class based on motif matching."""
    # Compile once per call is fine for prototype
    for func_class, motifs in FUNCTION_MOTIFS.items():
        for motif in motifs:
            try:
                if re.search(motif, sequence):
                    return func_class
            except re.error:
                continue
    # Fall-through heuristics
    if len(sequence) > 500:
        return 'enzyme'  # long proteins tend to be enzymes
    if sequence.count('K') + sequence.count('R') > 0.25 * len(sequence):
        return 'regulatory'  # many positive residues → DNA binding maybe
    if sequence.count('L') + sequence.count('I') + sequence.count('V') > 0.4 * len(sequence):
        return 'transport'  # hydrophobic → membrane
    return 'unknown'


# =============================================================================
# Species defaults
# =============================================================================
# Minimal metabolite/reaction starter sets per species. A real implementation
# would pull from KEGG/BiGG. For the prototype, we have a few species with
# hand-curated minimal sets.

SPECIES_DEFAULTS = {
    'syn3a': {
        'cell_radius_um': 0.3,
        'metabolites': [
            ('glc', 'glucose', 'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O', 5.0),
            ('g6p', 'glucose-6-phosphate', None, 0.5),
            ('f6p', 'fructose-6-phosphate', None, 0.3),
            ('atp', 'ATP', None, 3.0),
            ('adp', 'ADP', None, 0.5),
            ('pi',  'phosphate', 'OP(=O)(O)O', 10.0),
            ('nad', 'NAD+', None, 1.0),
            ('nadh', 'NADH', None, 0.2),
            ('pyr', 'pyruvate', 'CC(=O)C(=O)O', 0.1),
            ('lac', 'lactate', 'CC(O)C(=O)O', 0.5),
            ('h2o', 'water', 'O', 55000.0),
        ],
        'reactions': [
            ('hex', 'hexokinase', {'glc': 1, 'atp': 1}, {'g6p': 1, 'adp': 1}),
            ('pgi', 'glucose-6-phosphate isomerase', {'g6p': 1}, {'f6p': 1}),
            ('ldh', 'lactate dehydrogenase', {'pyr': 1, 'nadh': 1}, {'lac': 1, 'nad': 1}),
        ],
    },
    'ecoli': {
        'cell_radius_um': 0.5,
        'metabolites': [
            ('glc', 'glucose', 'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O', 10.0),
            ('g6p', 'glucose-6-phosphate', None, 1.0),
            ('atp', 'ATP', None, 9.0),
            ('adp', 'ADP', None, 0.5),
            ('h2o', 'water', 'O', 55000.0),
        ],
        'reactions': [
            ('hex', 'hexokinase', {'glc': 1, 'atp': 1}, {'g6p': 1, 'adp': 1}),
        ],
    },
    'generic': {
        'cell_radius_um': 1.0,
        'metabolites': [
            ('h2o', 'water', 'O', 55000.0),
            ('atp', 'ATP', None, 3.0),
        ],
        'reactions': [],
    },
}


# =============================================================================
# Top-level: parse genome → CellSpec
# =============================================================================
def parse_fasta(path: Path) -> Tuple[str, str]:
    """Minimal FASTA parser that returns (header, sequence)."""
    header = ""
    lines = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if header and lines:
                    break  # only take first sequence for prototype
                header = line[1:]
            else:
                lines.append(line)
    return header, ''.join(lines).upper()


def build_cell_spec(
    genome_path: Optional[Path] = None,
    species: str = 'syn3a',
    max_proteins: int = 500,
) -> CellSpec:
    """
    Build a CellSpec from a genome file + species identifier.

    If genome_path is None, we build a spec from species defaults only
    (useful for testing Layer 3 in isolation).
    """
    defaults = SPECIES_DEFAULTS.get(species, SPECIES_DEFAULTS['generic'])
    spec = CellSpec(
        species=species,
        cell_radius_um=defaults['cell_radius_um'],
    )

    # Load metabolites from defaults
    for met_id, name, smiles, conc in defaults['metabolites']:
        spec.metabolites[met_id] = Metabolite(
            met_id=met_id, name=name, smiles=smiles,
            initial_concentration_mM=conc,
        )

    # Load reactions from defaults
    for rxn_id, name, reactants, products in defaults['reactions']:
        spec.reactions[rxn_id] = Reaction(
            rxn_id=rxn_id, name=name,
            reactants=reactants, products=products,
        )

    # Parse genome if provided
    if genome_path is not None and genome_path.exists():
        header, sequence = parse_fasta(genome_path)
        spec.genome_size_bp = len(sequence)
        orfs = find_orfs(sequence)
        # Keep up to max_proteins to bound memory/time
        for i, (pos, protein_seq) in enumerate(orfs[:max_proteins]):
            gene_id = f"gene_{i:04d}"
            spec.proteins[gene_id] = Protein(
                gene_id=gene_id,
                sequence=protein_seq,
                length=len(protein_seq),
                function_class=classify_protein(protein_seq),
                annotations={'genome_position': str(pos)},
            )

    return spec


# =============================================================================
# Demo / test
# =============================================================================
if __name__ == "__main__":
    # Test without a genome file first
    spec = build_cell_spec(species='syn3a')
    print(spec.summary())
    print()

    # Test with a synthetic mini-genome
    import tempfile, os
    fake_genome = ">test_genome\n" + ("ATG" + "GCTAGCTAGCTACGATCG" * 50 + "TAA") * 10
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(fake_genome)
        tmppath = f.name
    spec2 = build_cell_spec(genome_path=Path(tmppath), species='syn3a')
    print(spec2.summary())
    os.unlink(tmppath)
