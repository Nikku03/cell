"""
Real Syn3A loader.

Reads the Luthey-Schulten lab's data files (GenBank, initial concentrations,
kinetic parameters, complex definitions) and builds a CellSpec +
transition-rule set that represents actual JCVI-Syn3A biology.

Data sources:
  - syn3A.gb: NCBI GenBank accession CP016816, 543 kbp, 496 genes
  - initial_concentrations.xlsx: protein copy numbers from comparative
    proteomics (Breuer 2019, Thornburg 2022)
  - kinetic_params.xlsx: k_cat values from BRENDA and manual curation
  - complex_formation.xlsx: 24 known Syn3A protein complexes

This is the real-biology version of `build_cell_spec('syn3a')`.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from layer0_genome.parser import CellSpec, Protein, Metabolite, Reaction


DATA_ROOT = Path(__file__).parent.parent / 'data' / 'Minimal_Cell_ComplexFormation' / 'input_data'


# ============================================================================
# GenBank parser (subset — enough for our needs)
# ============================================================================
def parse_genbank(path: Path) -> Tuple[Dict[str, dict], int]:
    """
    Parse a GenBank file and return gene annotations.

    Returns:
        genes: dict mapping locus_tag -> {gene, product, function, length, translation}
        genome_length_bp: total genome size
    """
    try:
        from Bio import SeqIO
        rec = next(SeqIO.parse(str(path), 'genbank'))
        genes = {}
        for feat in rec.features:
            if feat.type != 'CDS':
                continue
            quals = feat.qualifiers
            locus = quals.get('locus_tag', [None])[0]
            if locus is None:
                continue
            info = {
                'locus_tag': locus,
                'gene':      quals.get('gene', [''])[0],
                'product':   quals.get('product', [''])[0],
                'function':  quals.get('function', [''])[0] if 'function' in quals else '',
                'ec':        quals.get('EC_number', [''])[0] if 'EC_number' in quals else '',
                'translation': quals.get('translation', [''])[0],
                'length_aa': len(quals.get('translation', [''])[0]),
                'location':  (int(feat.location.start), int(feat.location.end)),
            }
            genes[locus] = info
        return genes, len(rec.seq)
    except Exception as e:
        print(f"ERROR parsing GenBank: {e}")
        return {}, 0


# ============================================================================
# Function classification from GenBank product field
# ============================================================================
def classify_from_product(product: str, ec: str) -> str:
    """Heuristic classification of a protein by its GenBank product field."""
    p = product.lower() if product else ''
    if ec:  # has EC number → enzyme
        return 'enzyme'
    if any(kw in p for kw in [
        'kinase', 'synthase', 'dehydrogenase', 'transferase',
        'isomerase', 'ligase', 'lyase', 'hydrolase', 'oxidase',
        'reductase', 'polymerase', 'ase ', 'ase/'
    ]):
        return 'enzyme'
    if any(kw in p for kw in ['ribosomal', 'ribosome']):
        return 'ribosomal'
    if any(kw in p for kw in ['transport', 'permease', 'pump', 'porter',
                               'channel', 'abc ', 'atpase subunit']):
        return 'transport'
    if any(kw in p for kw in ['trna', 'rrna', 'mrna']):
        return 'rna_processing'
    if any(kw in p for kw in ['transcription', 'sigma', 'regulator',
                               'repressor', 'activator']):
        return 'regulatory'
    if any(kw in p for kw in ['ftsz', 'division', 'cytoskel', 'structural',
                               'capsule', 'cell wall']):
        return 'structural_division'
    if 'hypothetical' in p or 'uncharacterized' in p or not p:
        return 'unknown'
    return 'other'


# ============================================================================
# Load protein initial counts
# ============================================================================
def load_initial_protein_counts() -> Dict[str, int]:
    """Return {locus_tag: initial_count} from the proteomics data."""
    df = pd.read_excel(DATA_ROOT / 'initial_concentrations.xlsx',
                        sheet_name='Comparative Proteomics',
                        skiprows=1)  # first row is header-note
    # Columns include "Locus Tag" and proteomics counts
    out = {}
    if 'Locus Tag' in df.columns:
        locus_col = 'Locus Tag'
    else:
        locus_col = df.columns[0]

    # Try to find the "initial count for simulation" column.
    # Typical column names in this file: 'Syn3A count for sim', 'Syn3A Ptn Count', etc.
    count_col = None
    for c in df.columns:
        cl = str(c).lower()
        if 'syn3a' in cl and ('count' in cl or 'ptn #' in cl):
            count_col = c
            break
    if count_col is None:
        # Fallback: any numeric column
        for c in df.columns:
            if df[c].dtype in (int, float):
                count_col = c
                break

    for _, row in df.iterrows():
        locus = str(row[locus_col])
        if not locus.startswith('JCVISYN3A'):
            continue
        try:
            n = int(round(float(row[count_col])))
            if n > 0:
                out[locus] = n
        except (ValueError, TypeError):
            continue
    return out


# ============================================================================
# Load metabolites
# ============================================================================
def load_metabolites() -> Dict[str, Metabolite]:
    """Load intracellular metabolites with their initial concentrations."""
    df = pd.read_excel(DATA_ROOT / 'initial_concentrations.xlsx',
                        sheet_name='Intracellular Metabolites')
    mets = {}
    for _, row in df.iterrows():
        met_id = str(row.get('Met ID', '')).strip()
        name = str(row.get('Metabolite name', '')).strip()
        conc = row.get('Init Conc (mM)', 0.0)
        if not met_id or met_id == 'nan':
            continue
        try:
            conc = float(conc)
        except (ValueError, TypeError):
            conc = 0.0
        mets[met_id] = Metabolite(
            met_id=met_id, name=name,
            initial_concentration_mM=conc,
            compartment='cytoplasm',
        )
    return mets


# ============================================================================
# Load known complexes
# ============================================================================
@dataclass
class ComplexDef:
    name: str
    gene_locus_tags: List[str]
    pdb_ids: List[str]
    stoichiometry_hint: str = ''


def load_complexes() -> List[ComplexDef]:
    """Load the 24 known Syn3A complex definitions."""
    df = pd.read_excel(DATA_ROOT / 'complex_formation.xlsx',
                        sheet_name='Complexes')
    complexes = []
    for _, row in df.iterrows():
        name = str(row.get('Name', '')).strip()
        genes = str(row.get('Genes Products', ''))
        pdb   = str(row.get('PDB Structures', ''))
        stoich = str(row.get('Stoichiometries', ''))
        init_count = row.get('Init. Count', 0)
        if not name or name == 'nan':
            continue
        # Gene loci look like "0685;0686" — convert to JCVISYN3A_0685 format
        locus_tags = []
        for tok in genes.replace(',', ';').split(';'):
            tok = tok.strip()
            # Accept any digit string; pad to 4 digits
            if tok.isdigit():
                locus_tags.append(f'JCVISYN3A_{tok.zfill(4)}')
        complexes.append(ComplexDef(
            name=name,
            gene_locus_tags=locus_tags,
            pdb_ids=[p.strip() for p in pdb.replace(';', ',').split(',') if p.strip()],
            stoichiometry_hint=str(row.get('How Cal Count', '')),
        ))
    return complexes


# ============================================================================
# Load kinetic parameters (kcat for each metabolic reaction)
# ============================================================================
def load_kcats() -> Dict[str, float]:
    """
    Return {reaction_name: kcat_per_second}.

    Extracts kcat from the Central, Nucleotide, Lipid, Cofactor, Transport
    sheets. Uses the "forward kcat" row (the first rate row after the enzyme
    ID row).
    """
    kcats = {}
    for sheet in ['Central', 'Nucleotide', 'Lipid', 'Cofactor', 'Transport']:
        df = pd.read_excel(DATA_ROOT / 'kinetic_params.xlsx', sheet_name=sheet)
        # Each reaction has multiple parameter rows. Find forward kcat.
        current_rxn = None
        have_kcat = False
        for _, row in df.iterrows():
            rxn = str(row.get('Reaction Name', '')).strip()
            ptype = str(row.get('Parameter Type', '')).strip().lower()
            if rxn and rxn != 'nan' and rxn != current_rxn:
                current_rxn = rxn
                have_kcat = False
            val = row.get('Value', '')
            units = str(row.get('Units', '')).strip().lower()
            if current_rxn and not have_kcat and '1/s' in units:
                try:
                    kcats[current_rxn] = float(val)
                    have_kcat = True
                except (ValueError, TypeError):
                    pass
    return kcats


# ============================================================================
# Main loader — build a real Syn3A CellSpec
# ============================================================================
def build_real_syn3a_cellspec(
    max_genes: Optional[int] = None,
    include_unknown: bool = True,
) -> Tuple[CellSpec, Dict[str, int], List[ComplexDef], Dict[str, float]]:
    """
    Build a fully-loaded Syn3A CellSpec with real proteome, metabolites,
    complexes, and reaction kinetics.

    Returns:
        spec: CellSpec populated with real proteins and metabolites
        protein_counts: {locus_tag: initial_count} from proteomics
        complexes: list of ComplexDef for the 24 known complexes
        kcats: {reaction_name: kcat_per_s} for metabolic reactions
    """
    print("Parsing Syn3A GenBank...")
    genes, genome_bp = parse_genbank(DATA_ROOT / 'syn3A.gb')
    print(f"  Genome: {genome_bp:,} bp, {len(genes)} CDS")

    print("Loading proteomics initial counts...")
    counts = load_initial_protein_counts()
    print(f"  {len(counts)} proteins with initial counts")

    print("Loading metabolites...")
    metabolites = load_metabolites()
    print(f"  {len(metabolites)} metabolites")

    print("Loading known complexes...")
    complexes = load_complexes()
    print(f"  {len(complexes)} known complexes")

    print("Loading kinetic parameters...")
    kcats = load_kcats()
    print(f"  {len(kcats)} reactions with k_cat")

    # Build the CellSpec
    spec = CellSpec(
        species='syn3a_real',
        cell_radius_um=0.2,  # 200 nm radius, from cryo-ET
        genome_size_bp=genome_bp,
    )
    spec.metabolites = metabolites

    # Add proteins
    n_added = 0
    for locus, info in genes.items():
        fclass = classify_from_product(info['product'], info['ec'])
        if not include_unknown and fclass == 'unknown':
            continue
        spec.proteins[locus] = Protein(
            gene_id=locus,
            sequence=info['translation'],
            length=info['length_aa'],
            function_class=fclass,
            annotations={
                'gene_name': info['gene'],
                'product': info['product'],
                'ec': info['ec'],
                'location_start': str(info['location'][0]),
                'location_end':   str(info['location'][1]),
            },
        )
        n_added += 1
        if max_genes and n_added >= max_genes:
            break

    return spec, counts, complexes, kcats


# ============================================================================
# Demo
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Loading REAL Syn3A from Luthey-Schulten data files")
    print("=" * 60)

    spec, counts, complexes, kcats = build_real_syn3a_cellspec()

    print(f"\n{spec.summary()}")

    print(f"\nSample proteins (first 10):")
    for locus, p in list(spec.proteins.items())[:10]:
        print(f"  {locus}  [{p.function_class:18s}]  "
              f"{p.annotations['gene_name']:8s}  {p.annotations['product'][:50]}")

    print(f"\nSample protein counts (first 10):")
    for locus in list(counts.keys())[:10]:
        print(f"  {locus}: {counts[locus]} molecules")

    print(f"\nKnown complexes (first 10):")
    for c in complexes[:10]:
        print(f"  {c.name:12s}  genes: {c.gene_locus_tags}  "
              f"PDB: {c.pdb_ids}")

    print(f"\nSample k_cat values (first 10):")
    for rxn, kcat in list(kcats.items())[:10]:
        print(f"  {rxn:12s}  k_cat = {kcat:8.2f} /s")

    # Function class distribution
    from collections import Counter
    fc = Counter(p.function_class for p in spec.proteins.values())
    print(f"\nFunction class distribution:")
    for cl, n in fc.most_common():
        print(f"  {cl:20s}: {n}")

    # Proteins with counts
    proteins_with_counts = set(spec.proteins.keys()) & set(counts.keys())
    total_molecules = sum(counts[p] for p in proteins_with_counts)
    print(f"\nProteins with initial count data: {len(proteins_with_counts)}")
    print(f"Total protein molecules at t=0: {total_molecules:,}")
