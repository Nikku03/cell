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


def test_v13_trna_modification_classes_match_expected_loci():
    """v13 adds trna_threonylcarbamoylation (tsaBCDE),
    trna_amidation (gatABC), and trna_thiolation (mnmA + 0240).
    Each must fire on the expected loci with confidence 0.8 and
    produce zero Nonessential FPs."""
    d = AnnotationClassDetector()
    expected = {
        "trna_threonylcarbamoylation": [
            "JCVISYN3A_0079", "JCVISYN3A_0144",
            "JCVISYN3A_0270", "JCVISYN3A_0271",
        ],
        "trna_amidation": [
            "JCVISYN3A_0687", "JCVISYN3A_0688", "JCVISYN3A_0689",
        ],
        "trna_thiolation": ["JCVISYN3A_0387", "JCVISYN3A_0240"],
    }
    for klass, loci in expected.items():
        for lt in loci:
            mode, _, conf, ev = d.detect_for_gene(lt)
            assert mode == FailureMode.TRANSLATION_STALL, (
                f"{lt} ({klass}) returned mode {mode}"
            )
            assert conf == 0.8, f"{lt} conf {conf} != 0.8"
            assert klass in ev, f"{lt} evidence missing class: {ev}"


def test_v13_trna_class_hit_counts():
    """Breuer-wide hit counts for the 3 new classes must match the
    validated set: 4 + 3 + 2 = 9 new TPs, 0 FPs."""
    r = ann_eval()
    hits = r["class_hits"]
    assert hits.get("trna_threonylcarbamoylation") == 4
    assert hits.get("trna_amidation") == 3
    assert hits.get("trna_thiolation") == 2


def test_v14_new_classes_each_zero_fp():
    """All 30 v14 classes must be 0-FP on the full Breuer set.
    Total standalone annotation detector TP must reach 112 with
    precision 1.0."""
    r = ann_eval()
    assert r["fp"] == 0
    assert r["precision"] == 1.0
    assert r["tp"] >= 112, f"expected at least 112 TPs, got {r['tp']}"


def test_v14_specific_class_captures():
    """Spot-check that v14 classes fire on their canonical target
    genes. If one of these fails, a pattern got broken during a
    refactor."""
    d = AnnotationClassDetector()
    targets = {
        "protein_deformylation": "JCVISYN3A_0201",           # def
        "translation_initiation_formyltransferase":
            "JCVISYN3A_0390",                                # fmt
        "methionine_aminopeptidase": "JCVISYN3A_0650",       # map
        "signal_recognition_particle": "JCVISYN3A_0360",     # ffh
        "methionine_adenosyltransferase": "JCVISYN3A_0432",  # metK
        "nad_biosynthesis": "JCVISYN3A_0378",                # nadE
        "dna_ligase": "JCVISYN3A_0690",                      # ligA
        "glycolysis_gapdh": "JCVISYN3A_0451",                # gapdh
        "ribosome_gtpase": "JCVISYN3A_0377",                 # obgE
        "ctp_synthase": "JCVISYN3A_0129",                    # pyrG
        "excinuclease": "JCVISYN3A_0254",                    # uvrC
        "rrna_maturation": "JCVISYN3A_0402",                 # ybeY
        "ribosome_binding_factor": "JCVISYN3A_0289",         # rbfA
        "acp_synthase": "JCVISYN3A_0513",                    # acpS
        "trna_uridine_carboxymethylaminomethyl":
            "JCVISYN3A_0081",                                # mnmE
        "membrane_protein_insertase": "JCVISYN3A_0908",      # yidC
        "iron_sulfur_cluster": "JCVISYN3A_0442",             # iscU
        "cysteine_desulfurase": "JCVISYN3A_0441",            # iscS
        "thioredoxin_reductase": "JCVISYN3A_0819",           # trx
        "adenine_salvage_prt": "JCVISYN3A_0413",             # apt
        "uracil_salvage_prt": "JCVISYN3A_0798",              # upp
        "aaa_protease": "JCVISYN3A_0039",                    # ftsH
        "clp_protease": "JCVISYN3A_0545",                    # clpB
        "ribonuclease_hi": "JCVISYN3A_0283",                 # rnhA
        "ribonuclease_m5": "JCVISYN3A_0003",                 # rnmV
        "formyltetrahydrofolate_cyclo_ligase":
            "JCVISYN3A_0443",                                # yggN
        "ribose_5_phosphate_isomerase": "JCVISYN3A_0800",    # rpiB
        "phosphatidylglycerol_synthase": "JCVISYN3A_0875",   # pgsA
        "glycolipid_synthase": "JCVISYN3A_0113",             # bcsA
    }
    for klass, locus in targets.items():
        mode, _, conf, ev = d.detect_for_gene(locus)
        assert mode != FailureMode.NONE, f"{locus} ({klass}) abstained"
        assert conf == 0.8
        assert klass in ev, (
            f"{locus} matched wrong class; ev={ev}"
        )


def test_v14_dna_polymerase_i_now_caught():
    """v14 broadens dna_replication_core to catch polA (DNA polymerase
    I) on top of the existing 'DNA polymerase III' matches."""
    d = AnnotationClassDetector()
    mode, _, conf, ev = d.detect_for_gene("JCVISYN3A_0611")  # polA
    assert mode == FailureMode.DNA_REPLICATION_BLOCKED
    assert conf == 0.8
    assert "dna_replication_core" in ev


def test_v15_new_classes_zero_fp_and_capture_17_more():
    """v15 adds 12 more classes + widens trna_pseudouridine_synthase
    pattern. Standalone annotation detector TP must reach 125, FP must
    stay 0."""
    r = ann_eval()
    assert r["fp"] == 0
    assert r["precision"] == 1.0
    assert r["tp"] >= 125, f"expected at least 125 TPs, got {r['tp']}"


def test_v15_specific_class_captures():
    """Spot-check that v15 classes + the widened truA/truB pattern fire
    on their canonical target genes."""
    d = AnnotationClassDetector()
    targets = {
        "trna_pseudouridine_synthase": "JCVISYN3A_0640",      # truA
        "transcription_antitermination": "JCVISYN3A_0300",    # nusA
        "ssra_binding": "JCVISYN3A_0776",                     # smpB
        "nucleotide_exchange_factor": "JCVISYN3A_0543",       # grpE
        "ribosome_subunit_maturation": "JCVISYN3A_0366",      # rbgA
        "rna_polymerase_subunit": "JCVISYN3A_0128",           # rpoE
        "pyrophosphohydrolase": "JCVISYN3A_0414",             # relA
        "phosphocarrier_hpr": "JCVISYN3A_0694",               # ptsH
        "dihydrofolate_synthase": "JCVISYN3A_0823",           # folC
        "pts_enzyme_i": "JCVISYN3A_0233",                     # ptsI
        "flavin_reductase": "JCVISYN3A_0302",                 # fre
        "primosomal_protein": "JCVISYN3A_0608",               # dnaI
        "atp_dependent_helicase": "JCVISYN3A_0695",           # pcrA
    }
    for klass, locus in targets.items():
        mode, _, conf, ev = d.detect_for_gene(locus)
        assert mode != FailureMode.NONE, f"{locus} ({klass}) abstained"
        assert conf == 0.8
        assert klass in ev, f"{locus} matched wrong class; ev={ev}"


def test_v15_truB_now_caught():
    """truB has 'pseudouridine(55)' in its product name. The widened
    'trna pseudouridine' pattern (v15) catches it; the original
    'trna pseudouridine synthase' (v14) did not."""
    d = AnnotationClassDetector()
    mode, _, conf, ev = d.detect_for_gene("JCVISYN3A_0290")
    assert mode == FailureMode.TRANSLATION_STALL
    assert "trna_pseudouridine_synthase" in ev


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
