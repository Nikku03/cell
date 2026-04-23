"""Tests for the two prior-knowledge detectors that drove the v10
MCC lift to 0.70 on the balanced-40 panel.

Both detectors are offline and deterministic, so we can unit-test them
against hand-picked Syn3A locus_tags with known labels and make strong
assertions. No simulator, no trajectories, no seeds — pure lookup.
"""
from __future__ import annotations

import pytest

from cell_sim.layer6_essentiality.complex_assembly_detector import (
    ComplexAssemblyDetector,
    ComplexAssemblyKB,
    evaluate_against_breuer as cx_eval,
)
from cell_sim.layer6_essentiality.annotation_class_detector import (
    AnnotationClassDetector,
    AnnotationClassKB,
    evaluate_against_breuer as ann_eval,
)
from cell_sim.layer6_essentiality.composed_detector import ComposedDetector
from cell_sim.layer6_essentiality.harness import FailureMode
from cell_sim.layer6_essentiality.priors_only_predictor import (
    PriorsOnlyPredictor,
)


# ---------- ComplexAssemblyKB ----------

def test_complex_kb_loads_25_complexes_with_121_genes():
    kb = ComplexAssemblyKB.load()
    assert len(kb.all_locus_tags()) == 121, \
        f"expected 121 subunit genes, got {len(kb.all_locus_tags())}"


def test_complex_kb_maps_ribosome_correctly():
    kb = ComplexAssemblyKB.load()
    # Pick one well-known ribosomal subunit. The whole ribosome (58
    # subunits) is a single entry with Init. Count 500.
    # JCVISYN3A_0025 is in the ribosome list.
    members = kb.gene_to_memberships.get("JCVISYN3A_0025", [])
    assert any(m.complex_name == "Ribosome" and m.init_count == 500
               for m in members)


def test_complex_kb_records_pdb_ids_for_known_complexes():
    kb = ComplexAssemblyKB.load()
    # Gyrase has PDB 6RKS; 4Z2C.
    gyrA_members = kb.gene_to_memberships.get("JCVISYN3A_0007", [])
    assert any("6RKS" in m.pdb_structures for m in gyrA_members)


# ---------- ComplexAssemblyDetector ----------

def test_complex_detector_fires_on_ribosomal_subunit():
    d = ComplexAssemblyDetector()
    mode, _, conf, ev = d.detect_for_gene("JCVISYN3A_0025")
    assert mode == FailureMode.TRANSLATION_STALL
    assert conf >= 0.8
    assert "Ribosome" in ev


def test_complex_detector_abstains_on_unknown_gene():
    d = ComplexAssemblyDetector()
    mode, _, conf, ev = d.detect_for_gene("JCVISYN3A_9999")
    assert mode == FailureMode.NONE
    assert conf == 0.0


def test_complex_detector_precision_at_least_95_pct_on_breuer():
    """The known-complex-subunit test is a high-precision prior —
    essentially all known bacterial complexes carry essential
    subunits. If this drops below 0.95 something in the KB loader
    has regressed."""
    r = cx_eval()
    assert r["precision"] >= 0.95, (
        f"precision {r['precision']} < 0.95; tp={r['tp']}, fp={r['fp']}"
    )


# ---------- AnnotationClassKB / Detector ----------

def test_annotation_kb_loads_496_syn3a_genes():
    kb = AnnotationClassKB.load()
    assert len(kb.gene_to_product) == 496


def test_annotation_detector_classifies_trna_ligase():
    d = AnnotationClassDetector()
    # pheRS beta subunit: known aminoacyl-tRNA ligase
    mode, _, conf, ev = d.detect_for_gene("JCVISYN3A_0528")
    assert mode == FailureMode.TRANSLATION_STALL
    assert "aminoacyl_trna_synthetase" in ev


def test_annotation_detector_classifies_flippase():
    d = AnnotationClassDetector()
    mode, _, conf, ev = d.detect_for_gene("JCVISYN3A_0372")
    assert mode == FailureMode.MEMBRANE_INTEGRITY
    assert "flippase" in ev.lower()


def test_annotation_detector_abstains_on_uncharacterized():
    d = AnnotationClassDetector()
    # JCVISYN3A_0352 is "Uncharacterized protein"
    mode, _, conf, ev = d.detect_for_gene("JCVISYN3A_0352")
    assert mode == FailureMode.NONE


def test_annotation_detector_zero_fp_on_breuer():
    """Annotation rules are hand-curated to match only essential
    biological classes. Full-set Breuer precision must be 1.00."""
    r = ann_eval()
    assert r["fp"] == 0, (
        f"annotation detector produced {r['fp']} false positives on "
        f"Breuer (expected 0)"
    )
    assert r["precision"] == 1.0


# ---------- PriorsOnlyPredictor ----------

def test_priors_only_predictor_flags_expected_essentials():
    p = PriorsOnlyPredictor()
    # Ribosomal subunit -> via complex
    r0025 = p.predict("JCVISYN3A_0025")
    assert r0025.essential and r0025.source == "complex"
    # tRNA ligase -> via annotation
    r0528 = p.predict("JCVISYN3A_0528")
    assert r0528.essential and r0528.source == "annotation"
    # Uncharacterized -> neither
    r0352 = p.predict("JCVISYN3A_0352")
    assert not r0352.essential and r0352.source == "none"


def test_priors_only_full_sweep_under_one_second():
    """Regression guard: priors-only scan of all Breuer-labeled genes
    must complete in well under a second. If it regresses beyond 2s,
    something is doing IO per call instead of using the module-level
    singleton KBs."""
    import time
    from cell_sim.layer6_essentiality.priors_only_predictor import (
        PriorsOnlyPredictor, _load_breuer,
    )
    from pathlib import Path
    labels = _load_breuer(
        Path(__file__).resolve().parents[2]
        / "memory_bank" / "data" / "syn3a_essentiality_breuer2019.csv"
    )
    p = PriorsOnlyPredictor()
    t0 = time.perf_counter()
    for locus in labels:
        p.predict(locus)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, (
        f"priors-only sweep took {elapsed:.2f}s (>2s); "
        "check for repeated spreadsheet loads"
    )


# ---------- ComposedDetector (no trajectory sub-detector) ----------

class _DummyTrajectoryDetector:
    """Never fires — used to test that composed detector still works
    when the trajectory sub-detector abstains."""
    def detect_for_gene(self, locus_tag, ko):
        return FailureMode.NONE, None, 0.0, "dummy_abstain"


def test_composed_detector_fires_from_structural_alone():
    """With the trajectory sub-detector abstaining, the composed
    detector should still fire when the structural signal fires —
    the OR composition MUST not require the trajectory to vote
    yes. (Regression guard: an earlier draft AND'd them.)"""
    d = ComposedDetector(
        structural=ComplexAssemblyDetector(),
        annotation=AnnotationClassDetector(),
        trajectory=_DummyTrajectoryDetector(),
    )
    mode, _, conf, ev = d.detect_for_gene("JCVISYN3A_0025", ko=None)
    assert mode == FailureMode.TRANSLATION_STALL
    assert conf > 0
