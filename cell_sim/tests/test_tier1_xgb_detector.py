"""Tests for the Tier-1 XGBoost detector and its helpers.

Every test here is deterministic and runs against synthetic or
bundled data. No heavy ML / pretrained-model weights are touched.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cell_sim.layer6_essentiality.tier1_xgb_detector import (
    PriorFeatureSet,
    PriorsUnionDetector,
    Tier1FeatureBundle,
    Tier1XgbDetector,
    build_balanced_panel,
    build_feature_bundle,
    confusion,
    default_registry,
    load_breuer_labels,
    mcc,
)


BREUER = (
    Path(__file__).resolve().parents[2]
    / "memory_bank" / "data" / "syn3a_essentiality_breuer2019.csv"
)


def test_mcc_matches_textbook_formula():
    # Textbook example: TP=5, FP=1, TN=4, FN=2. MCC should be
    # (5*4 - 1*2) / sqrt(6*7*5*6) = 18 / sqrt(1260) ≈ 0.507
    y_t = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_p = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])
    assert abs(mcc(y_t, y_p) - 18 / (1260 ** 0.5)) < 1e-9


def test_mcc_returns_zero_on_degenerate_input():
    y_t = np.array([1, 1, 0, 0])
    y_p = np.array([1, 1, 1, 1])   # all-positive: tn=0, fn=0 -> den=0
    assert mcc(y_t, y_p) == 0.0


def test_confusion_counts():
    y_t = np.array([1, 1, 0, 0, 1])
    y_p = np.array([1, 0, 0, 1, 1])
    c = confusion(y_t, y_p)
    assert c == {"tp": 2, "fp": 1, "tn": 1, "fn": 1}


def test_load_breuer_labels_quasi_positive_vs_strict():
    labels_pos = load_breuer_labels(BREUER, quasi_as_positive=True)
    labels_strict = load_breuer_labels(BREUER, quasi_as_positive=False)
    # Every Quasi ends up in the positive side of labels_pos and is
    # dropped from labels_strict.
    diff = set(labels_pos) - set(labels_strict)
    assert all(labels_pos[t] == 1 for t in diff), \
        "quasi-as-positive should label dropped loci as 1"
    assert len(labels_pos) > len(labels_strict)


def test_build_balanced_panel_is_deterministic_and_balanced():
    labels = load_breuer_labels(BREUER, quasi_as_positive=False)
    p1 = build_balanced_panel(labels, 40, seed=42)
    p2 = build_balanced_panel(labels, 40, seed=42)
    assert p1 == p2, "build_balanced_panel should be deterministic"
    assert len(p1) == 40
    pos = sum(1 for t in p1 if labels[t] == 1)
    neg = sum(1 for t in p1 if labels[t] == 0)
    assert pos == 20 and neg == 20


def test_prior_feature_set_matrix_shape():
    pfs = PriorFeatureSet()
    m = pfs.matrix(["JCVISYN3A_0001", "JCVISYN3A_0025", "JCVISYN3A_9999"])
    assert m.shape == (3, 3)
    # JCVISYN3A_0001 (dnaA): annotation rule fires (replication_initiator).
    assert m[0, 1] == 1.0
    # JCVISYN3A_0025 is a ribosomal subunit in the complex KB.
    assert m[1, 0] == 1.0
    # JCVISYN3A_9999 is unknown: no flag fires.
    assert (m[2] == 0.0).all()


def test_priors_union_detector_matches_v10b_fps_on_full():
    """The PriorsUnionDetector should reproduce the v10b three-prior
    decision. On the full Breuer set (Quasi=positive) with an empty
    trajectory csv (i.e. no trajectory signal available), MCC should
    be >=0.20 and precision >=0.94 — the rule system's floor."""
    labels = load_breuer_labels(BREUER, quasi_as_positive=True)
    pfs = PriorFeatureSet()  # no trajectory_csv -> traj=0 for all
    det = PriorsUnionDetector(prior_set=pfs)
    ordered = sorted(labels.keys())
    y = np.array([labels[t] for t in ordered], dtype=int)
    p = det.predict(ordered)
    m = mcc(y, p)
    c = confusion(y, p)
    # Without the trajectory signal, recall is lower than v10b-full
    # (0.58) but precision stays high.
    precision = c["tp"] / max(1, c["tp"] + c["fp"])
    assert precision >= 0.94, (
        f"priors-union precision fell below 0.94: {precision:.3f}"
    )
    assert m > 0.20, f"priors-union MCC collapsed to {m:.3f}"


def test_priors_union_beats_random():
    """On the full set the priors must exceed zero MCC."""
    labels = load_breuer_labels(BREUER, quasi_as_positive=True)
    pfs = PriorFeatureSet()
    det = PriorsUnionDetector(prior_set=pfs)
    ordered = sorted(labels.keys())
    y = np.array([labels[t] for t in ordered], dtype=int)
    p = det.predict(ordered)
    assert mcc(y, p) > 0.10


def test_default_registry_declares_three_sources(tmp_path):
    reg = default_registry(cache_dir=tmp_path)
    assert set(reg.list_sources()) == {"esm2_650M", "alphafold_db",
                                        "mace_off_kcat"}
    # Nothing cached in tmp_path.
    for n in reg.list_sources():
        assert reg.is_cached(n) is False


def test_feature_bundle_nan_block_when_source_missing(tmp_path):
    """If a declared source isn't cached, build_feature_bundle returns
    an all-NaN block of the right width."""
    reg = default_registry(cache_dir=tmp_path)
    pfs = PriorFeatureSet()
    bundle = build_feature_bundle(
        ["JCVISYN3A_0001", "JCVISYN3A_0002"],
        registry=reg, priors=pfs,
    )
    # ESM-2 block: 2x1280 all NaN.
    assert bundle.esm2.shape == (2, 1280)
    assert np.isnan(bundle.esm2).all()
    # AlphaFold: 2x9.
    assert bundle.alphafold.shape == (2, 9)
    # MACE: 2x7.
    assert bundle.mace.shape == (2, 7)
    # Priors still populated.
    assert bundle.priors.shape == (2, 3)


def test_tier1_xgb_detector_metadata():
    det = Tier1XgbDetector(feature_slice="priors_only")
    assert det.feature_slice == "priors_only"
    # Trees + regularization defaults sane.
    assert det.max_depth == 3
    assert det.reg_lambda >= 1.0


def test_tier1_xgb_detector_feature_slice_dispatch():
    """_slice() must route the right columns per feature_slice."""
    bundle = Tier1FeatureBundle(
        locus_tags=["a", "b"],
        esm2=np.arange(2 * 4, dtype=np.float32).reshape(2, 4),
        alphafold=np.full((2, 9), np.nan, dtype=np.float32),
        mace=np.full((2, 7), np.nan, dtype=np.float32),
        priors=np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32),
    )
    assert Tier1XgbDetector(feature_slice="esm2_only")._slice(bundle).shape == (2, 4)
    assert Tier1XgbDetector(feature_slice="priors_only")._slice(bundle).shape == (2, 3)
    assert Tier1XgbDetector(feature_slice="esm2_plus_priors")._slice(bundle).shape == (2, 7)
    assert Tier1XgbDetector(feature_slice="stacked")._slice(bundle).shape == (2, 23)


def test_tier1_xgb_detector_cv_score_runs_on_priors_only(tmp_path):
    """Smoke test: the CV harness runs end-to-end on priors-only when
    XGBoost is available in the environment. If xgboost isn't
    installed the test is skipped."""
    try:
        import xgboost  # noqa: F401
    except ImportError:
        pytest.skip("xgboost not installed")
    try:
        import sklearn  # noqa: F401
    except ImportError:
        pytest.skip("scikit-learn not installed")
    labels = load_breuer_labels(BREUER, quasi_as_positive=True)
    reg = default_registry(cache_dir=(
        Path(__file__).resolve().parents[1] / "features" / "cache"
    ))
    ordered = sorted(labels.keys())
    # Only run on the subset actually in the cache (if ESM-2 parquet
    # exists we restrict; otherwise use all labels).
    try:
        cached = set(reg.load("esm2_650M").index)
        ordered = [t for t in ordered if t in cached]
    except (FileNotFoundError, ValueError):
        pass
    pfs = PriorFeatureSet()
    bundle = build_feature_bundle(ordered, reg, pfs)
    y = np.array([labels[t] for t in ordered], dtype=int)
    det = Tier1XgbDetector(feature_slice="priors_only", n_estimators=50)
    result = det.cv_score(bundle, y, n_splits=3)
    assert "fold_mccs" in result
    assert len(result["fold_mccs"]) == 3
    assert "aggregated_confusion" in result
    agg = result["aggregated_confusion"]
    assert agg["tp"] + agg["fp"] + agg["tn"] + agg["fn"] == len(ordered)
