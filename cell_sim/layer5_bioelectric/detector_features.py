"""Bioelectric feature extraction from trajectory snapshots — Phase B1.

A small set of features derived from the per-sample ``VMEM_MV`` value
the bioelectric layer adds to each ``Sample.pools`` dict when
``RealSimulatorConfig.enable_bioelectric`` is True.

Phase B1 ships the extractor only — it is NOT wired into the v15
detector. A future Phase B3 detector variant would consume these
features alongside the existing pool-deviation and event-count
features. Until then, the extractor is exercised only by the
infrastructure tests.

Features
--------

* ``vmem_mean_mv``: arithmetic mean of VMEM_MV across samples (ignoring NaN).
* ``vmem_final_mv``: VMEM_MV at the last sample with a non-NaN value.
* ``vmem_range_mv``: max(VMEM_MV) - min(VMEM_MV) across samples.
* ``vmem_dev_from_wt_mv``: deviation of the trajectory's mean Vmem
  from a supplied WT reference. None if no WT reference is given.
* ``has_data``: True iff at least one sample produced a finite VMEM_MV.

The extractor is deliberately small. A Phase B3 detector that wants
to use these features can call this once per knockout trajectory and
combine with v15's existing detector inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import math


@dataclass
class BioelectricFeatures:
    vmem_mean_mv: float
    vmem_final_mv: float
    vmem_range_mv: float
    vmem_dev_from_wt_mv: Optional[float]
    has_data: bool

    def to_dict(self) -> dict:
        return {
            "vmem_mean_mv": self.vmem_mean_mv,
            "vmem_final_mv": self.vmem_final_mv,
            "vmem_range_mv": self.vmem_range_mv,
            "vmem_dev_from_wt_mv": self.vmem_dev_from_wt_mv,
            "has_data": self.has_data,
        }


def _iter_finite_vmem(samples: Iterable) -> list[float]:
    """Pull VMEM_MV from each sample's pools dict, skipping NaN / missing."""
    out: list[float] = []
    for s in samples:
        pools = getattr(s, "pools", None) or {}
        v = pools.get("VMEM_MV")
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def extract_bioelectric_features(
    trajectory,
    wt_reference_mean_mv: Optional[float] = None,
) -> BioelectricFeatures:
    """Compute bioelectric summary features from a Trajectory's samples.

    ``trajectory`` must expose a ``samples`` iterable where each
    sample has a ``pools`` dict (matches
    ``cell_sim.layer6_essentiality.harness.Trajectory`` /
    ``Sample``).

    If no sample produced a finite VMEM_MV (e.g. ``enable_bioelectric``
    was False, or the simulator volume was zero), the returned
    ``BioelectricFeatures.has_data`` is False and all numeric fields
    are NaN.
    """
    samples = getattr(trajectory, "samples", []) or []
    finite = _iter_finite_vmem(samples)

    if not finite:
        return BioelectricFeatures(
            vmem_mean_mv=float("nan"),
            vmem_final_mv=float("nan"),
            vmem_range_mv=float("nan"),
            vmem_dev_from_wt_mv=None,
            has_data=False,
        )

    mean = sum(finite) / len(finite)
    rng = max(finite) - min(finite)
    final = finite[-1]
    dev = (mean - wt_reference_mean_mv) if wt_reference_mean_mv is not None else None

    return BioelectricFeatures(
        vmem_mean_mv=mean,
        vmem_final_mv=final,
        vmem_range_mv=rng,
        vmem_dev_from_wt_mv=dev,
        has_data=True,
    )


__all__ = ["BioelectricFeatures", "extract_bioelectric_features"]
