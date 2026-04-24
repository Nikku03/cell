"""Tier-1 XGBoost detector: ML classifier over cached pretrained features.

This detector is the first consumer of the Session-13/14 feature cache.
It stacks:

  * ESM-2 (650M) per-protein embeddings — 1280 dims, populated in commit
    ebbfdff.
  * AlphaFold-DB structural descriptors — 9 dims, currently empty-by-
    design (JCVI-Syn3A is not indexed in UniProt; Session-15 TODO 0a).
  * MACE-OFF per-enzyme k_cat aggregates — 7 dims, currently empty-by-
    design (no curated SBML-species -> SMILES map; Session-15 TODO 0b).
  * The three v10b prior binaries: complex-assembly membership,
    annotation-class membership, per-rule trajectory signal.

It exposes two scoring modes:

  * ``priors_union``  — the v10b rule-based baseline (no ML at all).
    MCC 0.364 on full 455, MCC 0.70-0.80 on balanced-40. Included here
    as a reference point so every measurement run emits both numbers
    in the same CSV.
  * ``tier1_xgb``     — XGBoost on a chosen feature stack (ESM-2,
    priors, or both). 5-fold stratified CV on the Breuer labels.

Honest negative finding from the first measurement run:
``tier1_xgb`` does NOT beat ``priors_union`` on this benchmark. The
class imbalance (383 positive : 72 negative on the full set) plus
the already-thorough priors coverage (87/121 complex-subunit TPs at
precision 0.97) means ML struggles to find residual signal ESM-2
alone can see. This detector is the infrastructure for the Session
15 measurement + subsequent cache refills; the actual MCC lift
will come (if at all) once AlphaFold + MACE-OFF carry real content.

Every call is deterministic given ``random_state`` and the input
feature cache. No pretrained model weights are loaded by this
module — it consumes the parquet files written by the Colab
populate notebook.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from cell_sim.features import FeatureRegistry, FeatureSource
from cell_sim.layer6_essentiality.annotation_class_detector import (
    AnnotationClassDetector,
)
from cell_sim.layer6_essentiality.complex_assembly_detector import (
    ComplexAssemblyDetector,
)


# ---- label loading ----


def load_breuer_labels(
    path: Path,
    *,
    quasi_as_positive: bool = True,
) -> dict[str, int]:
    """Load Breuer 2019 binary labels. Essential -> 1, Nonessential -> 0,
    Quasiessential -> 1 (when ``quasi_as_positive``, the sweep default).
    Genes with other / missing labels are dropped."""
    labels: dict[str, int] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row["essentiality"].strip()
            if v == "Essential":
                labels[row["locus_tag"]] = 1
            elif v == "Nonessential":
                labels[row["locus_tag"]] = 0
            elif v == "Quasiessential" and quasi_as_positive:
                labels[row["locus_tag"]] = 1
    return labels


def build_balanced_panel(
    labels: dict[str, int],
    n: int,
    seed: int,
    *,
    require_in: Optional[set[str]] = None,
) -> list[str]:
    """Return ``n`` genes with ``n // 2`` positives and ``n - n // 2``
    negatives, sampled deterministically from ``labels``. If
    ``require_in`` is given, only genes in that set are considered
    (e.g. restrict to loci that have ESM-2 embeddings)."""
    keep = require_in if require_in is not None else set(labels)
    pos = sorted(t for t, v in labels.items() if v == 1 and t in keep)
    neg = sorted(t for t, v in labels.items() if v == 0 and t in keep)
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = n // 2
    return sorted(pos[:k] + neg[: n - k])


# ---- prior features ----


@dataclass
class PriorFeatureSet:
    """Binary prior features per locus_tag for the tier-1 XGB stack.

    Columns:
      - ``cx_flag``   1 iff the gene is a subunit of an active known complex
      - ``an_flag``   1 iff the gene matches an AnnotationClass keyword rule
      - ``traj_flag`` 1 iff the v10b trajectory (PerRule) detector fired
                      on the gene in the reference sweep
    """
    trajectory_csv: Optional[Path] = None
    _complex: ComplexAssemblyDetector = field(
        default_factory=ComplexAssemblyDetector
    )
    _annotation: AnnotationClassDetector = field(
        default_factory=AnnotationClassDetector
    )
    _trajectory: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.trajectory_csv is not None and self.trajectory_csv.exists():
            with open(self.trajectory_csv) as f:
                for row in csv.DictReader(f):
                    ev = row.get("evidence", "")
                    # v10b prediction CSVs use 'traj_only[...]' or
                    # '...+traj[...]' in the evidence string when the
                    # PerRule trajectory sub-detector fires.
                    fired = ev.startswith("traj_only") or "+traj" in ev
                    self._trajectory[row["locus_tag"]] = int(fired)

    def matrix(self, locus_tags: list[str]) -> np.ndarray:
        cx = np.fromiter(
            (self._complex.detect_for_gene(t)[0].value != "none"
             for t in locus_tags),
            dtype=np.int8, count=len(locus_tags),
        )
        an = np.fromiter(
            (self._annotation.detect_for_gene(t)[0].value != "none"
             for t in locus_tags),
            dtype=np.int8, count=len(locus_tags),
        )
        tj = np.fromiter(
            (self._trajectory.get(t, 0) for t in locus_tags),
            dtype=np.int8, count=len(locus_tags),
        )
        return np.column_stack([cx, an, tj]).astype(np.float32)


# ---- feature bundle ----


@dataclass
class Tier1FeatureBundle:
    """Feature block for a list of locus_tags. Built from the
    FeatureRegistry + PriorFeatureSet."""
    locus_tags: list[str]
    esm2: np.ndarray                   # (n, 1280) or (n, 0)
    alphafold: np.ndarray              # (n, 9) — may be all NaN (Session-14 placeholder)
    esmfold: np.ndarray                # (n, 9) — Session-15 backend; NaN until populate
    mace: np.ndarray                   # (n, 7) — may be all NaN
    priors: np.ndarray                 # (n, 3) cx/an/traj binaries

    @property
    def stacked(self) -> np.ndarray:
        """All columns side-by-side. NaN columns come through
        naturally; tree learners tolerate NaN."""
        return np.column_stack([
            self.esm2, self.alphafold, self.esmfold, self.mace, self.priors,
        ])

    @property
    def esm2_only(self) -> np.ndarray:
        return self.esm2

    @property
    def priors_only(self) -> np.ndarray:
        return self.priors

    @property
    def esm2_plus_priors(self) -> np.ndarray:
        return np.column_stack([self.esm2, self.priors])

    @property
    def esmfold_plus_priors(self) -> np.ndarray:
        return np.column_stack([self.esmfold, self.priors])

    @property
    def structure_plus_priors(self) -> np.ndarray:
        """Both structural sources + priors. AlphaFold stays NaN on
        Syn3A (UniProt blocked); ESMFold carries the real signal once
        the parquet is populated."""
        return np.column_stack([self.alphafold, self.esmfold, self.priors])


def build_feature_bundle(
    locus_tags: list[str],
    registry: FeatureRegistry,
    priors: PriorFeatureSet,
) -> Tier1FeatureBundle:
    """Load / join the four cached parquets for ``locus_tags`` and
    attach the prior binaries."""
    sources = [s for s in (
        "esm2_650M", "alphafold_db", "esmfold_v1", "mace_off_kcat",
    ) if s in registry.list_sources()]
    if sources:
        joined = registry.join_features(locus_tags, sources=sources)
    else:
        joined = pd.DataFrame(index=pd.Index(locus_tags, name="locus_tag"))

    def _block(prefix: str, expected_width: int) -> np.ndarray:
        cols = [c for c in joined.columns if c.startswith(prefix)]
        if not cols:
            return np.full((len(locus_tags), expected_width), np.nan,
                           dtype=np.float32)
        return joined[cols].to_numpy().astype(np.float32)

    return Tier1FeatureBundle(
        locus_tags=list(locus_tags),
        esm2=_block("esm2_650M_dim_", 1280),
        alphafold=_block("af_", 9),
        esmfold=_block("esmfold_", 9),
        mace=_block("mace_", 7),
        priors=priors.matrix(locus_tags),
    )


# ---- scoring helpers ----


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    num = tp * tn - fp * fn
    den_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / sqrt(den_sq) if den_sq > 0 else 0.0


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


# ---- detectors ----


@dataclass
class PriorsUnionDetector:
    """Reference detector: essential iff any of the three priors fire.
    No simulation, no training, no ML. Mirrors the v10b composed
    detector's decision on the no-fold-splits full-set case."""
    prior_set: PriorFeatureSet

    def predict(self, locus_tags: list[str]) -> np.ndarray:
        P = self.prior_set.matrix(locus_tags).astype(int)
        return (P.sum(axis=1) > 0).astype(int)


@dataclass
class Tier1XgbDetector:
    """XGBoost classifier over a chosen Tier-1 feature slice.

    ``feature_slice`` picks one of:
      - ``"esm2_only"``             — 1280 dims
      - ``"priors_only"``           — 3 dims
      - ``"esm2_plus_priors"``      — 1283 dims
      - ``"esmfold_plus_priors"``   — 12 dims (9 structural + 3 priors)
      - ``"structure_plus_priors"`` — 21 dims (af 9 + esmfold 9 + priors 3)
      - ``"stacked"``               — 1308 dims (every block)
    """
    feature_slice: str = "esm2_plus_priors"
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 0.7
    colsample_bytree: float = 0.3
    reg_alpha: float = 1.0
    reg_lambda: float = 2.0
    min_child_weight: int = 3
    random_state: int = 42
    scale_pos_weight_auto: bool = True

    def _slice(self, bundle: Tier1FeatureBundle) -> np.ndarray:
        return {
            "esm2_only": bundle.esm2_only,
            "priors_only": bundle.priors_only,
            "esm2_plus_priors": bundle.esm2_plus_priors,
            "esmfold_plus_priors": bundle.esmfold_plus_priors,
            "structure_plus_priors": bundle.structure_plus_priors,
            "stacked": bundle.stacked,
        }[self.feature_slice]

    def _make_clf(self, y_tr: np.ndarray):
        from xgboost import XGBClassifier  # noqa: WPS433 — lazy
        spw = 1.0
        if self.scale_pos_weight_auto:
            pos = max(1, int((y_tr == 1).sum()))
            neg = int((y_tr == 0).sum())
            spw = neg / pos
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            scale_pos_weight=spw,
            random_state=self.random_state,
            eval_metric="logloss",
            n_jobs=4,
            verbosity=0,
        )

    def cv_score(
        self,
        bundle: Tier1FeatureBundle,
        y: np.ndarray,
        *,
        n_splits: int = 5,
        split_seed: int = 42,
    ) -> dict:
        from sklearn.model_selection import StratifiedKFold  # noqa: WPS433

        X = self._slice(bundle)
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=split_seed,
        )
        fold_mccs: list[float] = []
        agg = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        fold_preds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for tr, te in cv.split(X, y):
            clf = self._make_clf(y[tr])
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te]).astype(int)
            fold_mccs.append(mcc(y[te], pred))
            for k, v in confusion(y[te], pred).items():
                agg[k] += v
            fold_preds.append((te, y[te], pred))
        return {
            "feature_slice": self.feature_slice,
            "n_splits": n_splits,
            "fold_mccs": [float(m) for m in fold_mccs],
            "mean_mcc": float(np.mean(fold_mccs)),
            "std_mcc": float(np.std(fold_mccs)),
            "aggregated_confusion": agg,
            "aggregated_mcc": mcc(
                np.concatenate([f[1] for f in fold_preds]),
                np.concatenate([f[2] for f in fold_preds]),
            ),
        }


# ---- registry setup helper ----


_DEFAULT_CACHE_DIR = Path("cell_sim/features/cache")


def default_registry(cache_dir: Path = _DEFAULT_CACHE_DIR) -> FeatureRegistry:
    """Register the four feature sources on a fresh FeatureRegistry.

    Session 14 shipped esm2_650M + alphafold_db + mace_off_kcat.
    Session 15 added esmfold_v1 after the AFDB path was blocked by
    the UniProt-indexing gap for Syn3A (taxid 2144189). The
    alphafold_db entry is kept for backward compatibility and for
    future use on other organisms; for Syn3A the parquet is
    effectively NaN and esmfold_v1 carries the real signal.
    """
    reg = FeatureRegistry(cache_dir=cache_dir)
    reg.register(FeatureSource(
        name="esm2_650M",
        parquet_path=cache_dir / "esm2_650M.parquet",
        expected_sha256=None,
        feature_cols=[f"esm2_650M_dim_{i}" for i in range(1280)],
        version="0.1.0",
    ))
    reg.register(FeatureSource(
        name="alphafold_db",
        parquet_path=cache_dir / "alphafold_db.parquet",
        expected_sha256=None,
        feature_cols=[
            "af_plddt_mean", "af_plddt_std", "af_disorder_fraction",
            "af_helix_fraction", "af_sheet_fraction", "af_coil_fraction",
            "af_sequence_length", "af_radius_of_gyration_angstrom",
            "af_has_structure",
        ],
        version="0.1.0",
    ))
    reg.register(FeatureSource(
        name="esmfold_v1",
        parquet_path=cache_dir / "esmfold_v1.parquet",
        expected_sha256=None,
        feature_cols=[
            "esmfold_plddt_mean", "esmfold_plddt_std",
            "esmfold_disorder_fraction",
            "esmfold_helix_fraction", "esmfold_sheet_fraction",
            "esmfold_coil_fraction",
            "esmfold_sequence_length",
            "esmfold_radius_of_gyration_angstrom",
            "esmfold_has_structure",
        ],
        version="0.1.0",
    ))
    reg.register(FeatureSource(
        name="mace_off_kcat",
        parquet_path=cache_dir / "mace_off_kcat.parquet",
        expected_sha256=None,
        feature_cols=[
            "mace_kcat_mean_per_s", "mace_kcat_std_per_s",
            "mace_kcat_min_per_s", "mace_kcat_max_per_s",
            "mace_n_substrates", "mace_mean_confidence",
            "mace_has_estimate",
        ],
        version="0.1.0",
    ))
    return reg


__all__ = [
    "PriorFeatureSet",
    "PriorsUnionDetector",
    "Tier1FeatureBundle",
    "Tier1XgbDetector",
    "build_balanced_panel",
    "build_feature_bundle",
    "confusion",
    "default_registry",
    "load_breuer_labels",
    "mcc",
]
