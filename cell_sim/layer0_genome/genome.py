"""Layer 0 Genome API — memory-bank-backed, cited reference data.

Downstream layers read the genome through :class:`Genome`, never directly
from GenBank files or CSV paths. Every value exposed here is cross-checked
against a fact in ``memory_bank/facts/`` at load time, so a silent drift
between the CSV and the fact files causes a loud failure instead of bad
simulation output.

The existing ``cell_sim/layer0_genome/parser.py`` and ``syn3a_real.py``
modules remain untouched; this module is a parallel, thinner API that the
new Layer 1-6 code will use. See ``memory_bank/concepts/dna/DESIGN.md``.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, Iterator

__all__ = ["Gene", "Genome", "GenomeLoadError"]

# Repo-root-relative defaults. ``Genome.load()`` infers the repo root by
# walking up from this file so the loader works regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENE_TABLE_FACT = _REPO_ROOT / "memory_bank/facts/structural/syn3a_gene_table.json"
_GENE_COUNT_FACT = _REPO_ROOT / "memory_bank/facts/structural/syn3a_gene_count.json"
_CHROM_LEN_FACT = _REPO_ROOT / "memory_bank/facts/structural/syn3a_chromosome_length.json"
_ORIC_FACT = _REPO_ROOT / "memory_bank/facts/structural/syn3a_oric_position.json"
_GB_PATH = _REPO_ROOT / "cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb"


class GenomeLoadError(RuntimeError):
    """Raised when Genome.load() can't find the staged data or a cited
    value disagrees with the fact it came from. The message always tells
    the caller *what to do next* — usually re-stage per
    ``memory_bank/data/STAGING.md``."""


@dataclass(frozen=True, slots=True)
class Gene:
    locus_tag: str
    gene_name: str
    feature_type: str
    start_1based: int
    end: int
    strand: str
    length_bp: int
    product: str
    protein_id: str

    @property
    def is_cds(self) -> bool:
        return self.feature_type == "CDS"

    @property
    def is_rna(self) -> bool:
        return self.feature_type in {"tRNA", "rRNA", "ncRNA", "tmRNA"}


@dataclass(frozen=True)
class Genome:
    accession: str
    organism: str
    length_bp: int
    topology: str
    oric_position: int
    genes: tuple[Gene, ...]
    sequence: str | None = None

    # ----- loading -----
    @classmethod
    def load(cls, *, include_sequence: bool = False) -> "Genome":
        facts = _read_required_facts()
        genes = tuple(_read_gene_table(facts["gene_table_csv"]))

        # Cross-check counts against the gene-count fact.
        expected_total = int(facts["gene_count"]["value"]["number"])
        if len(genes) != expected_total:
            raise GenomeLoadError(
                f"Gene table row count ({len(genes)}) disagrees with "
                f"facts/structural/syn3a_gene_count.json ({expected_total}). "
                f"Re-stage per memory_bank/data/STAGING.md or update the fact."
            )
        breakdown = facts["gene_count"]["value"].get("breakdown") or {}
        for ftype, expected in breakdown.items():
            actual = sum(1 for g in genes if g.feature_type == ftype)
            if actual != expected:
                raise GenomeLoadError(
                    f"Feature-type '{ftype}' count {actual} != fact breakdown {expected}."
                )

        length_bp = int(facts["chromosome_length"]["value"]["number"])
        oric = int(facts["oric"]["value"]["number"])
        if not (0 <= oric <= length_bp):
            raise GenomeLoadError(
                f"oriC position {oric} not in [0, {length_bp}]."
            )
        for g in genes:
            if g.start_1based < 1 or g.end > length_bp:
                raise GenomeLoadError(
                    f"Gene {g.locus_tag} coordinates {g.start_1based}..{g.end} "
                    f"fall outside chromosome 1..{length_bp}."
                )

        sequence: str | None = None
        if include_sequence:
            sequence = _read_genbank_sequence()

        return cls(
            accession="CP016816.2",
            organism="JCVI-Syn3A",
            length_bp=length_bp,
            topology="circular",
            oric_position=oric,
            genes=genes,
            sequence=sequence,
        )

    # ----- convenience accessors -----
    def __len__(self) -> int:
        return len(self.genes)

    def __iter__(self) -> Iterator[Gene]:
        return iter(self.genes)

    def __getitem__(self, locus_tag: str) -> Gene:
        try:
            return self._by_tag[locus_tag]
        except KeyError as exc:
            raise KeyError(
                f"locus_tag {locus_tag!r} not in Syn3A genome"
            ) from exc

    @cached_property
    def _by_tag(self) -> dict[str, Gene]:
        return {g.locus_tag: g for g in self.genes}

    def cds_genes(self) -> Iterable[Gene]:
        return (g for g in self.genes if g.is_cds)

    def rna_genes(self) -> Iterable[Gene]:
        return (g for g in self.genes if g.is_rna)

    def knocked_out(self, locus_tags: Iterable[str]) -> "Genome":
        """Return a view of this genome with the named genes removed.

        Useful for Layer 6 knockout sweeps. Does not mutate the original."""
        drop = set(locus_tags)
        missing = drop - set(self._by_tag)
        if missing:
            raise KeyError(
                f"knocked_out() got unknown locus_tags: {sorted(missing)[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )
        kept = tuple(g for g in self.genes if g.locus_tag not in drop)
        return Genome(
            accession=self.accession,
            organism=self.organism,
            length_bp=self.length_bp,
            topology=self.topology,
            oric_position=self.oric_position,
            genes=kept,
            sequence=self.sequence,
        )


# ---------------- file loaders ----------------


def _read_required_facts() -> dict:
    facts: dict[str, dict] = {}
    for key, path in [
        ("gene_table", _GENE_TABLE_FACT),
        ("gene_count", _GENE_COUNT_FACT),
        ("chromosome_length", _CHROM_LEN_FACT),
        ("oric", _ORIC_FACT),
    ]:
        if not path.exists():
            raise GenomeLoadError(
                f"Required fact file missing: {path.relative_to(_REPO_ROOT)}. "
                f"Run Session 3 data-staging step (see memory_bank/data/STAGING.md)."
            )
        with path.open("r", encoding="utf-8") as fh:
            facts[key] = json.load(fh)
    csv_rel = facts["gene_table"]["value"]["data_file"]
    csv_path = _REPO_ROOT / csv_rel
    if not csv_path.exists():
        raise GenomeLoadError(
            f"Gene table CSV missing at {csv_path}. Re-stage per "
            f"memory_bank/data/STAGING.md and regenerate the table."
        )
    facts["gene_table_csv"] = csv_path
    return facts


def _read_gene_table(csv_path: Path) -> Iterable[Gene]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield Gene(
                locus_tag=row["locus_tag"],
                gene_name=row["gene_name"],
                feature_type=row["feature_type"],
                start_1based=int(row["start_1based"]),
                end=int(row["end"]),
                strand=row["strand"],
                length_bp=int(row["length_bp"]),
                product=row["product"],
                protein_id=row["protein_id"],
            )


def _read_genbank_sequence() -> str:
    if not _GB_PATH.exists():
        raise GenomeLoadError(
            f"GenBank file missing at {_GB_PATH}. Re-stage per "
            f"memory_bank/data/STAGING.md."
        )
    try:
        from Bio import SeqIO
    except ImportError as exc:
        raise GenomeLoadError(
            "Genome.load(include_sequence=True) requires biopython. "
            "Install with: pip install biopython"
        ) from exc
    rec = next(SeqIO.parse(str(_GB_PATH), "genbank"))
    return str(rec.seq)
