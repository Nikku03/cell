"""Sandbox tests for the multi-organism LOOCV trainer.

Covers:
  * synthetic-data generator produces realistic per-organism counts
    and a usable shape for XGBoost
  * LOOCV split keeps held-out organism rows out of the training set
  * confusion-matrix MCC matches sklearn on hand-computed cases
  * decision-band thresholds (>=0.55 INTEGRATE, >=0.45 ENSEMBLE,
    else FALSIFIED) match the rule documented in NEXT_SESSION.md
  * real-data loader raises a clear FileNotFoundError when the
    parquet or labels.csv is missing

Tests do NOT exercise XGBoost training itself (loocv with
n_estimators=400 takes ~60s on the synthetic 10k-row matrix —
too slow for CI). The LOOCV split is verified separately by
constructing a tiny dataset and asserting the fold compositions.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "train_multiorg_xgboost.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "train_multiorg_xgboost", SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ---- synthetic-data generator ------------------------------------


def test_synthetic_data_shape(mod):
    """The generator produces ~10k rows with the per-organism counts
    declared in the README, and the X / y / org arrays line up."""
    X, y, org, feature_cols = mod.make_synthetic_data(n_features=64, seed=42)
    assert len(X) == 4300 + 4200 + 700 + 480 + 455
    assert len(X) == len(y) == len(org)
    assert list(X.columns) == feature_cols
    assert len(feature_cols) == 64
    # Index is the composite key; values must be unique.
    assert X.index.is_unique
    # Per-organism counts.
    counts = org.value_counts()
    assert counts["ecoli"] == 4300
    assert counts["bsub"] == 4200
    assert counts["mpne"] == 700
    assert counts["mgen"] == 480
    assert counts["syn3a"] == 455


def test_synthetic_essentiality_rates(mod):
    """Per-organism essentiality rates roughly match the spec
    (within ±2 percentage points — the spec rounds to integer
    counts so exact equality isn't possible)."""
    X, y, org, _ = mod.make_synthetic_data(n_features=8, seed=42)
    df = pd.DataFrame({"org": org.values, "y": y.values})
    rates = df.groupby("org")["y"].mean()
    assert abs(rates["ecoli"] - 0.07) < 0.005
    assert abs(rates["bsub"] - 0.06) < 0.005
    assert abs(rates["mpne"] - 0.50) < 0.02
    assert abs(rates["mgen"] - 0.80) < 0.005
    assert abs(rates["syn3a"] - 0.84) < 0.005


def test_synthetic_planted_signal_is_actually_planted(mod):
    """The generator plants a +0.4σ shift in features 0-7 for the
    positive class. That signal should be detectable from the
    data alone — i.e. positive rows have higher means in those
    8 features. Catches regressions where the planting drifts."""
    X, y, _, _ = mod.make_synthetic_data(n_features=64, seed=42)
    pos_mean = X[y == 1].iloc[:, :8].values.mean()
    neg_mean = X[y == 0].iloc[:, :8].values.mean()
    assert pos_mean - neg_mean > 0.3, (
        f"planted signal lost: pos_mean={pos_mean:.3f} "
        f"neg_mean={neg_mean:.3f}"
    )
    # Outside the 8-feature signal block, both classes should be ~0.
    pos_rest = X[y == 1].iloc[:, 8:].values.mean()
    neg_rest = X[y == 0].iloc[:, 8:].values.mean()
    assert abs(pos_rest - neg_rest) < 0.05


# ---- MCC math + decision band ------------------------------------


def test_mcc_from_confusion_zeros(mod):
    # All zero confusion -> 0.0 (avoid division-by-zero NaN).
    assert mod._mcc_from_confusion(0, 0, 0, 0) == 0.0


def test_mcc_from_confusion_perfect(mod):
    # 50 TP + 50 TN, no errors -> MCC = 1.0
    assert mod._mcc_from_confusion(50, 0, 50, 0) == pytest.approx(1.0)


def test_mcc_from_confusion_all_wrong(mod):
    # Predicting backwards -> MCC = -1.0
    assert mod._mcc_from_confusion(0, 50, 0, 50) == pytest.approx(-1.0)


def test_mcc_from_confusion_v15_breuer(mod):
    """The v15 confusion (tp=287, fp=3, tn=69, fn=96) on Breuer
    must round-trip to MCC 0.5372 — same number that lives in
    mcc_against_breuer_v15_round2_priors."""
    mcc = mod._mcc_from_confusion(287, 3, 69, 96)
    assert mcc == pytest.approx(0.5372, abs=1e-3)


def test_decision_band_boundaries(mod):
    assert mod._decision_band(0.60) == "INTEGRATE_AS_DETECTOR"
    assert mod._decision_band(0.55) == "INTEGRATE_AS_DETECTOR"
    assert mod._decision_band(0.5499) == "ENSEMBLE_INPUT_ONLY"
    assert mod._decision_band(0.45) == "ENSEMBLE_INPUT_ONLY"
    assert mod._decision_band(0.4499) == "FALSIFIED"
    assert mod._decision_band(0.0) == "FALSIFIED"
    assert mod._decision_band(-0.5) == "FALSIFIED"


# ---- LOOCV split correctness -------------------------------------


def test_loocv_split_holds_out_correct_organism(mod):
    """LOOCV's contract: when held=X, every row with org==X is in
    the test set and every row with org!=X is in the training set.

    We verify this without invoking XGBoost by constructing a
    minimal dataset and patching XGBClassifier with a stub that
    just records what fit/predict were called with."""
    rng = np.random.default_rng(0)
    spec = [("a", 30), ("b", 30), ("c", 40)]
    keys = []
    org_per_row = []
    y_per_row = []
    for o, n in spec:
        for i in range(n):
            keys.append(f"{o}|loc_{i}")
            org_per_row.append(o)
            # 50/50 labels so per-organism class balance is non-degenerate
            y_per_row.append(int(i % 2))
    X = pd.DataFrame(
        rng.standard_normal((100, 4), dtype=np.float32),
        columns=[f"f{i}" for i in range(4)],
        index=pd.Index(keys, name="_key"),
    )
    y = pd.Series(y_per_row, index=X.index)
    org = pd.Series(org_per_row, index=X.index)

    fit_calls: list[tuple[set, set]] = []

    class _StubClf:
        def __init__(self, **kw): pass
        def fit(self, X_in, y_in, verbose=False):
            fit_calls.append((
                set(X_in.index.tolist()),
                set(y_in.index.tolist()),
            ))
        def predict_proba(self, X_in):
            # 50/50 to avoid degenerate confusion matrices
            n = len(X_in)
            return np.column_stack([
                np.full(n, 0.4),
                np.full(n, 0.6),
            ])
        feature_importances_ = np.zeros(4)
    monkey = mod.XGBClassifier
    mod.XGBClassifier = _StubClf
    try:
        res = mod.loocv(X, y, org, organisms=("a", "b", "c"))
    finally:
        mod.XGBClassifier = monkey

    # One entry per organism.
    assert set(res.keys()) == {"a", "b", "c"}
    # Three fits — one per held-out organism.
    assert len(fit_calls) == 3
    # First fold (a held out): training set has b + c rows only.
    held_a_train_keys, _ = fit_calls[0]
    assert all(not k.startswith("a|") for k in held_a_train_keys)
    assert any(k.startswith("b|") for k in held_a_train_keys)
    assert any(k.startswith("c|") for k in held_a_train_keys)
    # Per-organism counts.
    assert res["a"]["n_train"] == 70 and res["a"]["n_test"] == 30
    assert res["b"]["n_train"] == 70 and res["b"]["n_test"] == 30
    assert res["c"]["n_train"] == 60 and res["c"]["n_test"] == 40


def test_loocv_skips_organism_with_no_data(mod):
    """If an organism in the ORGANISMS tuple has zero rows in the
    actual data, loocv() must skip it cleanly rather than crash."""
    # Synthetic data with only ecoli + syn3a present.
    keys = [f"ecoli|loc_{i}" for i in range(20)] + [f"syn3a|loc_{i}" for i in range(20)]
    X = pd.DataFrame(
        np.random.default_rng(0).standard_normal((40, 4), dtype=np.float32),
        columns=[f"f{i}" for i in range(4)],
        index=pd.Index(keys, name="_key"),
    )
    y = pd.Series([i % 2 for i in range(40)], index=X.index)
    org = pd.Series(["ecoli"] * 20 + ["syn3a"] * 20, index=X.index)

    class _Stub:
        def __init__(self, **k): pass
        def fit(self, *a, **k): pass
        def predict_proba(self, X_in):
            n = len(X_in)
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])
        feature_importances_ = np.zeros(4)

    monkey = mod.XGBClassifier
    mod.XGBClassifier = _Stub
    try:
        res = mod.loocv(X, y, org, organisms=("ecoli", "bsub", "syn3a"))
    finally:
        mod.XGBClassifier = monkey

    assert "ecoli" in res
    assert "syn3a" in res
    assert "bsub" not in res  # no rows -> skipped


# ---- real-data loader fail-fast ----------------------------------


def test_load_real_data_missing_labels_raises(mod, tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        mod.load_real_data(
            labels_csv=tmp_path / "missing.csv",
            embed_parquet=tmp_path / "missing.parquet",
        )
    assert "labels" in str(excinfo.value).lower() or \
           "missing" in str(excinfo.value).lower()


def test_load_real_data_missing_embeds_raises(mod, tmp_path):
    # Create a stub labels.csv so the second-file check is what fires.
    labels = tmp_path / "labels.csv"
    labels.write_text("organism,locus_tag,gene_name,essential,source\n")
    with pytest.raises(FileNotFoundError) as excinfo:
        mod.load_real_data(
            labels_csv=labels,
            embed_parquet=tmp_path / "missing.parquet",
        )
    assert "parquet" in str(excinfo.value).lower() or \
           "missing" in str(excinfo.value).lower()


# ---- composite key helper ----------------------------------------


def test_composite_key_format(mod):
    df = pd.DataFrame({
        "organism": ["ecoli", "syn3a"],
        "locus_tag": ["b0001", "JCVISYN3A_0001"],
    })
    keys = mod._composite_key(df)
    assert list(keys) == ["ecoli|b0001", "syn3a|JCVISYN3A_0001"]
