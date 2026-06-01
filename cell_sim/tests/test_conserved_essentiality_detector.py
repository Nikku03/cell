"""Tests for ConservedEssentialityDetector + its ComposedDetector wiring."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cell_sim.layer6_essentiality.harness import FailureMode
from cell_sim.layer6_essentiality.conserved_essentiality_detector import (
    ConservedEssentialityDetector,
    ConservedEssentialityKB,
)


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a minimal cross_species_essentiality.csv at tmp_path."""
    p = tmp_path / "xs.csv"
    cols = [
        "locus_tag",
        "mg_homolog_locus", "mg_homolog_method", "mg_pident", "mg_essential",
        "mp_homolog_locus", "mp_homolog_method", "mp_pident", "mp_essential",
    ]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return p


def test_fires_when_both_refs_call_essential(tmp_path):
    csv_path = _write_csv(tmp_path, [
        {"locus_tag": "JCVISYN3A_0001",
         "mg_homolog_locus": "MG_001", "mg_homolog_method": "gene_name",
         "mg_essential": "1",
         "mp_homolog_locus": "MPN_001", "mp_homolog_method": "gene_name",
         "mp_essential": "1"},
    ])
    det = ConservedEssentialityDetector(kb=ConservedEssentialityKB.load(csv_path))
    mode, t, conf, ev = det.detect_for_gene("JCVISYN3A_0001", None)
    assert mode == FailureMode.HOMOLOG_ESSENTIAL
    assert conf == 0.90        # both-agree confidence
    assert "mg:MG_001" in ev
    assert "mp:MPN_001" in ev


def test_fires_when_only_one_ref_essential(tmp_path):
    csv_path = _write_csv(tmp_path, [
        {"locus_tag": "JCVISYN3A_0050",
         "mg_homolog_locus": "MG_050", "mg_homolog_method": "gene_name",
         "mg_essential": "1",
         "mp_homolog_locus": "", "mp_homolog_method": "",
         "mp_essential": ""},
    ])
    det = ConservedEssentialityDetector(kb=ConservedEssentialityKB.load(csv_path))
    mode, t, conf, ev = det.detect_for_gene("JCVISYN3A_0050", None)
    assert mode == FailureMode.HOMOLOG_ESSENTIAL
    assert conf == 0.78        # single-ref confidence
    assert "mg:MG_050" in ev


def test_abstains_when_homolog_nonessential(tmp_path):
    csv_path = _write_csv(tmp_path, [
        {"locus_tag": "JCVISYN3A_0930",
         "mg_homolog_locus": "MG_930", "mg_homolog_method": "gene_name",
         "mg_essential": "0",
         "mp_homolog_locus": "MPN_930", "mp_homolog_method": "gene_name",
         "mp_essential": "0"},
    ])
    det = ConservedEssentialityDetector(kb=ConservedEssentialityKB.load(csv_path))
    mode, t, conf, ev = det.detect_for_gene("JCVISYN3A_0930", None)
    assert mode == FailureMode.NONE
    # Diagnostic trace should mention both refs as 'non' so we can
    # tell abstention apart from "no homologs at all".
    assert "mg:MG_930=non" in ev
    assert "mp:MPN_930=non" in ev


def test_abstains_when_locus_missing(tmp_path):
    csv_path = _write_csv(tmp_path, [])
    det = ConservedEssentialityDetector(kb=ConservedEssentialityKB.load(csv_path))
    mode, t, conf, ev = det.detect_for_gene("JCVISYN3A_0999", None)
    assert mode == FailureMode.NONE
    assert ev == "hom_none"


def test_min_pident_filters_low_identity_blast(tmp_path):
    csv_path = _write_csv(tmp_path, [
        {"locus_tag": "JCVISYN3A_0100",
         "mg_homolog_locus": "MG_X", "mg_homolog_method": "blast",
         "mg_pident": "22.0",   # below default 30%
         "mg_essential": "1",
         "mp_homolog_locus": "", "mp_homolog_method": "",
         "mp_essential": ""},
    ])
    det = ConservedEssentialityDetector(kb=ConservedEssentialityKB.load(csv_path))
    mode, _, _, _ = det.detect_for_gene("JCVISYN3A_0100", None)
    assert mode == FailureMode.NONE    # filtered

    # Loosen the threshold and it should fire.
    det_loose = ConservedEssentialityDetector(
        kb=ConservedEssentialityKB.load(csv_path), min_pident=20.0,
    )
    mode2, _, _, _ = det_loose.detect_for_gene("JCVISYN3A_0100", None)
    assert mode2 == FailureMode.HOMOLOG_ESSENTIAL


def test_require_both_mode(tmp_path):
    csv_path = _write_csv(tmp_path, [
        {"locus_tag": "JCVISYN3A_0200",
         "mg_homolog_locus": "MG_200", "mg_homolog_method": "gene_name",
         "mg_essential": "1",
         "mp_homolog_locus": "", "mp_homolog_method": "",
         "mp_essential": ""},
    ])
    det_or = ConservedEssentialityDetector(
        kb=ConservedEssentialityKB.load(csv_path), require_both=False,
    )
    assert det_or.detect_for_gene("JCVISYN3A_0200", None)[0] == FailureMode.HOMOLOG_ESSENTIAL

    det_and = ConservedEssentialityDetector(
        kb=ConservedEssentialityKB.load(csv_path), require_both=True,
    )
    assert det_and.detect_for_gene("JCVISYN3A_0200", None)[0] == FailureMode.NONE


def test_composed_detector_or_adds_conserved_layer(tmp_path):
    """Smoke test: ComposedDetector with conserved= populated routes
    correctly. Uses minimal fakes for structural / annotation /
    trajectory sub-detectors."""
    from dataclasses import dataclass

    @dataclass
    class _Stub:
        mode: FailureMode = FailureMode.NONE
        conf: float = 0.0
        ev: str = "stub"
        def detect_for_gene(self, lt, ko=None):
            return self.mode, None, self.conf, self.ev

    csv_path = _write_csv(tmp_path, [
        {"locus_tag": "JCVISYN3A_0500",
         "mg_homolog_locus": "MG_500", "mg_homolog_method": "gene_name",
         "mg_essential": "1",
         "mp_homolog_locus": "", "mp_homolog_method": "", "mp_essential": ""},
    ])
    cons = ConservedEssentialityDetector(
        kb=ConservedEssentialityKB.load(csv_path),
    )

    from cell_sim.layer6_essentiality.composed_detector import ComposedDetector
    cd = ComposedDetector(
        structural=_Stub(),
        annotation=_Stub(),
        conserved=cons,
        trajectory=_Stub(),
    )
    mode, _, conf, ev = cd.detect_for_gene("JCVISYN3A_0500", None)
    assert mode == FailureMode.HOMOLOG_ESSENTIAL
    assert conf == 0.78
    assert ev.startswith("hom_only[")


def test_composed_detector_backward_compatible_when_conserved_none(tmp_path):
    """Without --enable-cross-species, ComposedDetector must behave
    identically to v15 (3-way compose, no hom layer)."""
    from dataclasses import dataclass

    @dataclass
    class _Stub:
        mode: FailureMode = FailureMode.NONE
        conf: float = 0.0
        ev: str = "stub"
        def detect_for_gene(self, lt, ko=None):
            return self.mode, None, self.conf, self.ev

    from cell_sim.layer6_essentiality.composed_detector import ComposedDetector
    cd = ComposedDetector(
        structural=_Stub(),
        annotation=_Stub(),
        trajectory=_Stub(),
        # conserved omitted -> None
    )
    mode, _, _, ev = cd.detect_for_gene("JCVISYN3A_0001", None)
    assert mode == FailureMode.NONE
    # The no-signal trace should include 'hom:hom_off' so the user
    # can see at a glance which layer was disabled.
    assert "hom:hom_off" in ev
