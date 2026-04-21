"""Layer 0 validation tests.

Each test corresponds to a specific biological or structural claim
recorded as a memory-bank fact. If a test fails, the memory bank and the
code have drifted apart — fix one or the other.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cell_sim.layer0_genome.genome import Genome, GenomeLoadError

REPO_ROOT = Path(__file__).resolve().parents[2]
FACTS = REPO_ROOT / "memory_bank/facts/structural"


def _fact(name: str) -> dict:
    with (FACTS / f"{name}.json").open() as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def genome() -> Genome:
    return Genome.load()


def test_load_smoke(genome: Genome) -> None:
    assert genome.accession == "CP016816.2"
    assert genome.organism == "JCVI-Syn3A"
    assert genome.topology == "circular"


def test_chromosome_length_matches_fact(genome: Genome) -> None:
    fact = _fact("syn3a_chromosome_length")
    assert genome.length_bp == int(fact["value"]["number"])


def test_gene_count_matches_fact(genome: Genome) -> None:
    fact = _fact("syn3a_gene_count")
    assert len(genome) == int(fact["value"]["number"])


def test_oric_matches_fact(genome: Genome) -> None:
    fact = _fact("syn3a_oric_position")
    assert genome.oric_position == int(fact["value"]["number"])


def test_feature_type_breakdown(genome: Genome) -> None:
    fact = _fact("syn3a_gene_count")
    expected = fact["value"]["breakdown"]
    from collections import Counter

    actual = Counter(g.feature_type for g in genome)
    for ftype, n in expected.items():
        assert actual[ftype] == n, f"{ftype}: {actual[ftype]} != {n}"


def test_dnaA_is_first_gene(genome: Genome) -> None:
    """dnaA sits at or near position 1 of the chromosome; oriC fact
    depends on it. If this ever breaks, the oriC inference is wrong."""
    dnaA = genome["JCVISYN3A_0001"]
    assert dnaA.gene_name == "dnaA"
    assert dnaA.start_1based == 1
    assert "replication initiator" in dnaA.product.lower()


def test_coordinates_within_chromosome(genome: Genome) -> None:
    for g in genome:
        assert 1 <= g.start_1based
        assert g.end <= genome.length_bp, (
            f"{g.locus_tag} end={g.end} > chromosome {genome.length_bp}"
        )
        assert g.strand in {"+", "-"}


def test_locus_tag_format(genome: Genome) -> None:
    pat = re.compile(r"^JCVISYN3A_\d{4}$")
    bad = [g.locus_tag for g in genome if not pat.match(g.locus_tag)]
    assert not bad, bad[:5]


def test_knockout_removes_one_gene(genome: Genome) -> None:
    ko = genome.knocked_out(["JCVISYN3A_0001"])
    assert len(ko) == len(genome) - 1
    with pytest.raises(KeyError):
        ko["JCVISYN3A_0001"]
    # Original is unchanged
    assert "JCVISYN3A_0001" in {g.locus_tag for g in genome}


def test_knockout_unknown_tag_raises(genome: Genome) -> None:
    with pytest.raises(KeyError):
        genome.knocked_out(["NOT_A_REAL_TAG"])


def test_sequence_load_when_requested() -> None:
    gb_path = REPO_ROOT / "cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb"
    if not gb_path.exists():
        pytest.skip("GenBank file not staged; see memory_bank/data/STAGING.md")
    g = Genome.load(include_sequence=True)
    assert g.sequence is not None
    assert len(g.sequence) == g.length_bp
    # Spot check: dnaA CDS starts with ATG (the vast majority of bacterial
    # CDS start with ATG; dnaA specifically does in Syn3A).
    dnaA = g["JCVISYN3A_0001"]
    start0 = dnaA.start_1based - 1
    assert g.sequence[start0:start0 + 3].upper() == "ATG"


def test_rpoD_exists(genome: Genome) -> None:
    """The brief names JCVISYN3A_0407 as the major sigma factor (rpoD).
    We keep this assertion to protect against silent changes to the
    gene table that would break downstream Layer 1 transcription work."""
    rpoD = genome["JCVISYN3A_0407"]
    assert rpoD.is_cds
    # Name or product must reference sigma / rpoD somewhere.
    token = (rpoD.gene_name + " " + rpoD.product).lower()
    assert "sigma" in token or "rpod" in token, (rpoD.gene_name, rpoD.product)
