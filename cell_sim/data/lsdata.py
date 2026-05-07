"""
Luthey-Schulten Minimal Cell dataset loader.

Wraps the contents of the upstream Luthey-Schulten data drop
(`Luthey-Schulten-Lab-Minimal_Cell-db048ac/`) so callers don't have to
remember per-asset paths or stream gigabyte-scale tarballs by hand.

Set `LSDATA_ROOT` to point at the dataset root. Default is the Colab Drive
path used during development.

Data layout (after schema-peek, 2026-05):
  - MinCell_counts_and_fluxes.tar.gz  -> 50 replicate CSVs
      counts_and_fluxes.{1..50}.csv
      Each CSV is TRANSPOSED: rows = 8572 species/fluxes (col 0 = name),
      columns = 7201 time points 0.0..7200.0 at 1 s resolution.
      ~268 MB compressed per replicate, ~13 GB uncompressed total.
  - RDME_gCME_ODE/model_data/s1c15/*.tar.gz
      Despite the .tar.gz extension these are gzipped single files, not
      tar archives. Use `open_s1c15(path)` which handles both.
  - Small CSV/TSV/XML/XLSX model assets — listed in REGISTRY, loaded
    directly from disk.
  - DepMap CSVs (CRISPRGeneEffect, OmicsExpression*) — gigabyte-scale
    flat tables; use `iter_depmap_*` for chunked reads.
"""
from __future__ import annotations

import gzip
import io
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import pandas as pd

DEFAULT_ROOT = Path('/content/drive/MyDrive/Luthey-Schulten-Lab-Minimal_Cell-db048ac')
ROOT = Path(os.environ.get('LSDATA_ROOT', DEFAULT_ROOT))

COUNTS_FLUXES_TAR = ROOT / 'MinCell_counts_and_fluxes.tar.gz'
N_REPLICATES = 50
TIMEPOINTS_PER_REPLICATE = 7201  # 0..7200 s, 1 s sampling
SPECIES_PER_REPLICATE = 8572     # rows of the transposed CSV (incl. header)

REGISTRY = {
    'sbml_imb155':            ROOT / 'RDME_gCME_ODE/model_data/iMB155.xml',
    'sbml_imb155_noh2o':      ROOT / 'RDME_gCME_ODE/model_data/iMB155_NoH2O.xml',
    'sbml_imb155_inhib':      ROOT / 'RDME_gCME_ODE/model_data/iMB155_NoH2O_Inhibition.xml',
    'cobra_imb155':           ROOT / 'RDME_gCME_ODE/model_data/iMB155.json',
    'syn3a_genbank':          ROOT / 'RDME_gCME_ODE/model_data/syn3A.gb',
    'syn2_genbank':           ROOT / 'RDME_gCME_ODE/model_data/syn2.gb',
    'global_params':          ROOT / 'RDME_gCME_ODE/model_data/GlobalParameters_Zane-TB-DB.csv',
    'kinetic_nucleotide':     ROOT / 'RDME_gCME_ODE/model_data/Nucleotide_Kinetic_Parameters.tsv',
    'central_aa':             ROOT / 'RDME_gCME_ODE/model_data/Central_AA_Zane_Balanced_direction_fixed_nounqATP.tsv',
    'lipid_balanced':         ROOT / 'RDME_gCME_ODE/model_data/lipid_NoH2O_balanced_model.tsv',
    'transport':              ROOT / 'RDME_gCME_ODE/model_data/transport_NoH2O_Zane-TB-DB.tsv',
    'mrna_counts':            ROOT / 'RDME_gCME_ODE/model_data/mRNA_Counts.csv',
    'proteomics':             ROOT / 'RDME_gCME_ODE/model_data/proteomics.xlsx',
    'reconstruction':         ROOT / 'RDME_gCME_ODE/model_data/reconstruction.xlsx',
    'protein_metabolites':    ROOT / 'RDME_gCME_ODE/model_data/protein_metabolites.csv',
    'membrane_protein_mets':  ROOT / 'RDME_gCME_ODE/model_data/membrane_protein_metabolites.csv',
    'ribo_protein_mets':      ROOT / 'RDME_gCME_ODE/model_data/ribo_protein_metabolites.csv',
    'rnap_proteins':          ROOT / 'RDME_gCME_ODE/model_data/RNApol_proteins.csv',
    'trna_synthase':          ROOT / 'RDME_gCME_ODE/model_data/trna_metabolites_synthase.csv',
    'rrna_1':                 ROOT / 'RDME_gCME_ODE/model_data/rrna_metabolites_1.csv',
    'rrna_2':                 ROOT / 'RDME_gCME_ODE/model_data/rrna_metabolites_2.csv',
    'gpr_manual':             ROOT / 'RDME_gCME_ODE/model_data/manual_GPR_conversion.csv',
    'lipid_rxns_list':        ROOT / 'RDME_gCME_ODE/model_data/lipid_rxns_list.txt',
    'nucleo_rxns_list':       ROOT / 'RDME_gCME_ODE/model_data/nucleo_rxns_list.txt',
    'depmap_crispr':          ROOT / 'CRISPRGeneEffect.csv',
    'depmap_expression':      ROOT / 'OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv',
    'gene_essentiality':      ROOT / 'gene_essentiality_results.csv',
}

S1C15_ARCHIVES = {
    'cg_reps_1_90':    ROOT / 'RDME_gCME_ODE/model_data/s1c15/s1c15_base_CG_reps00001_00090.tar.gz',
    'cg_1000_30reps':  ROOT / 'RDME_gCME_ODE/model_data/s1c15/s1c15_base_1000ribos_30reps.tar.gz',
    'cg_new_ribo':     ROOT / 'RDME_gCME_ODE/model_data/s1c15/new_ribo/s1c15_base_new_CG.tar.gz',
}


# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------
def get(key: str) -> Path:
    if key not in REGISTRY:
        raise KeyError(f'unknown asset {key!r}; known: {sorted(REGISTRY)}')
    p = REGISTRY[key]
    if not p.exists():
        raise FileNotFoundError(f'{key} -> {p} (set LSDATA_ROOT?)')
    return p


def check_registry() -> dict:
    """Return {present: [...], missing: [...]} for sanity-checking the mount."""
    present, missing = [], []
    for k, p in REGISTRY.items():
        (present if p.exists() else missing).append(k)
    return {'present': present, 'missing': missing, 'root': str(ROOT)}


# ---------------------------------------------------------------------------
# counts_and_fluxes — 50 replicate CSVs inside one tar.gz
# ---------------------------------------------------------------------------
def _replicate_member(i: int) -> str:
    if not 1 <= i <= N_REPLICATES:
        raise ValueError(f'replicate {i} outside 1..{N_REPLICATES}')
    return f'counts_and_fluxes.{i}.csv'


def list_replicates() -> List[int]:
    """Return replicate indices actually present in the tar."""
    with tarfile.open(COUNTS_FLUXES_TAR, 'r:gz') as t:
        names = t.getnames()
    out = []
    for n in names:
        if n.startswith('counts_and_fluxes.') and n.endswith('.csv'):
            try:
                out.append(int(n.split('.')[1]))
            except ValueError:
                pass
    return sorted(out)


def replicate_species(i: int = 1) -> List[str]:
    """First column of replicate `i` — names of all tracked species/fluxes."""
    with tarfile.open(COUNTS_FLUXES_TAR, 'r:gz') as t:
        f = t.extractfile(_replicate_member(i))
        if f is None:
            raise RuntimeError(f'extractfile returned None for replicate {i}')
        names: List[str] = []
        f.readline()  # discard header (Time, 0.0, 1.0, ...)
        for raw in f:
            line = raw.decode('utf-8', errors='replace')
            comma = line.find(',')
            names.append(line[:comma] if comma >= 0 else line.rstrip('\n'))
    return names


def load_replicate(
    i: int,
    species: Optional[Sequence[str]] = None,
    time_start: float = 0.0,
    time_end: float = 7200.0,
) -> pd.DataFrame:
    """
    Load replicate `i` as a DataFrame indexed by species name, columns = time.

    Parameters
    ----------
    species : optional list of species names to keep. If given, only those
        rows are parsed (others are streamed past as raw bytes — significant
        speedup when you only need a few hundred of the 8572 rows).
    time_start, time_end : float seconds, inclusive. Subsets the columns.
    """
    member = _replicate_member(i)
    species_set = set(species) if species is not None else None

    with tarfile.open(COUNTS_FLUXES_TAR, 'r:gz') as t:
        f = t.extractfile(member)
        if f is None:
            raise RuntimeError(f'extractfile returned None for replicate {i}')
        # Header: 'Time', '0.0', '1.0', ..., '7200.0'
        header_line = f.readline().decode('utf-8').rstrip('\n')
        cols = header_line.split(',')
        time_cols = [float(c) for c in cols[1:]]
        keep_col_idx = [
            j for j, t_ in enumerate(time_cols)
            if time_start <= t_ <= time_end
        ]
        keep_times = [time_cols[j] for j in keep_col_idx]

        # Build the kept rows
        rows: List[List[float]] = []
        names: List[str] = []
        for raw in f:
            line = raw.decode('utf-8', errors='replace').rstrip('\n')
            first_comma = line.find(',')
            if first_comma < 0:
                continue
            name = line[:first_comma]
            if species_set is not None and name not in species_set:
                continue
            values = line[first_comma + 1:].split(',')
            rows.append([float(values[j]) for j in keep_col_idx])
            names.append(name)

    return pd.DataFrame(rows, index=pd.Index(names, name='species'),
                        columns=pd.Index(keep_times, name='time_s'))


def materialize_replicate(i: int, dst_dir: Path) -> Path:
    """
    Extract replicate `i` to `dst_dir/counts_and_fluxes.{i}.csv` for fast
    repeated reads (Drive read latency dominates otherwise).
    Returns the destination path.
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / _replicate_member(i)
    if dst.exists():
        return dst
    with tarfile.open(COUNTS_FLUXES_TAR, 'r:gz') as t:
        src = t.extractfile(_replicate_member(i))
        if src is None:
            raise RuntimeError(f'extractfile returned None for replicate {i}')
        with open(dst, 'wb') as out:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    return dst


def to_parquet(i: int, dst_dir: Path) -> Path:
    """One-shot conversion: replicate `i` -> parquet (much faster reload)."""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f'counts_and_fluxes.{i}.parquet'
    if dst.exists():
        return dst
    df = load_replicate(i)
    df.to_parquet(dst, compression='zstd')
    return dst


# ---------------------------------------------------------------------------
# DepMap streaming — files are 300-420 MB, too large for naive read_csv
# ---------------------------------------------------------------------------
def iter_depmap(key: str, chunksize: int = 5_000) -> Iterator[pd.DataFrame]:
    """Stream a DepMap CSV in row chunks."""
    return pd.read_csv(get(key), chunksize=chunksize, low_memory=False)


def depmap_columns(key: str) -> List[str]:
    """Header-only read — names of all gene columns."""
    return list(pd.read_csv(get(key), nrows=0).columns)


# ---------------------------------------------------------------------------
# s1c15 archives — mislabeled .tar.gz (actually single gzipped files)
# ---------------------------------------------------------------------------
@dataclass
class S1C15Sniff:
    name: str
    decompressed_size: Optional[int]
    first_bytes: bytes
    fmt_guess: str


def sniff_s1c15(name: str) -> S1C15Sniff:
    """Decompress and identify the file inside an s1c15_*.tar.gz."""
    if name not in S1C15_ARCHIVES:
        raise KeyError(f'unknown s1c15 archive {name!r}; known: {sorted(S1C15_ARCHIVES)}')
    p = S1C15_ARCHIVES[name]
    with gzip.open(p, 'rb') as f:
        head = f.read(256)
    if head.startswith(b'\x89HDF'):
        fmt = 'hdf5'
    elif head[:2] == b'\x80\x02' or head[:2] == b'\x80\x04' or head[:1] == b'\x80':
        fmt = 'pickle'
    elif head.startswith(b'PK'):
        fmt = 'zip'
    elif head.startswith(b'PAR1'):
        fmt = 'parquet'
    elif head.startswith(b'\x93NUMPY'):
        fmt = 'npy'
    elif head[:6].isascii():
        fmt = f'text? (starts: {head[:80]!r})'
    else:
        fmt = f'unknown (magic: {head[:8]!r})'
    return S1C15Sniff(name=name, decompressed_size=None,
                      first_bytes=head[:32], fmt_guess=fmt)


def open_s1c15(name: str) -> io.BufferedReader:
    """Return a binary file-like handle to the decompressed s1c15 file."""
    if name not in S1C15_ARCHIVES:
        raise KeyError(name)
    return gzip.open(S1C15_ARCHIVES[name], 'rb')
