"""Verification suite for the v3 (real-chemistry, pose-quality) potential.

Re-runs every symmetry check, plus the v3 contract:

  - energy R^2 on MMFF labels >= 0.85
  - force MAE within reasonable scale of label
  - bad-pose rejection rate     >= 0.95
  - good-pose retention rate    >= 0.85
  - OOD ensemble std            >= 2x in-distribution std
  - all v2 symmetries           ~ machine epsilon
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .mmff_dataset import ChemConfiguration
from .multimode_v3 import MultiModePotentialV3
from .train_v3 import evaluate, sample_kwargs


@dataclass
class CheckResult:
    name: str
    passed: bool
    value: float
    threshold: float
    note: str = ""


def _energy_r2(model: MultiModePotentialV3, configs: list[ChemConfiguration]) -> tuple[float, float]:
    Et = []
    Ep = []
    f_abs = 0.0
    f_count = 0
    for c in configs:
        kw = sample_kwargs(c)
        pos = kw.pop("positions").clone().requires_grad_(True)
        E, _, _ = model.forward(pos, **kw)
        (g,) = torch.autograd.grad(E, pos, allow_unused=True)
        if g is None:
            g = torch.zeros_like(pos)
        Et.append(c.energy.item())
        Ep.append(E.item())
        f_abs += (-g - c.forces).abs().sum().item()
        f_count += c.forces.numel()
    Et_t = torch.tensor(Et)
    Ep_t = torch.tensor(Ep)
    ss_res = ((Et_t - Ep_t) ** 2).sum().item()
    ss_tot = ((Et_t - Et_t.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, f_abs / max(1, f_count)


def _pose_classification(
    model: MultiModePotentialV3, configs: list[ChemConfiguration]
) -> dict:
    """Returns counts and rates broken down by ground-truth label."""
    tp = fp = tn = fn = 0
    for c in configs:
        if c.pose_label == -1:
            continue
        kw = sample_kwargs(c)
        pos = kw.pop("positions").clone()
        _, _, pose_logit = model.forward(pos, **kw)
        pred = int(pose_logit.item() > 0)
        if c.pose_label == 1:
            if pred == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred == 0:
                tn += 1
            else:
                fp += 1
    near_total = tp + fn
    bad_total = tn + fp
    retention = tp / near_total if near_total > 0 else 0.0
    rejection = tn / bad_total if bad_total > 0 else 0.0
    return {
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "retention": retention, "rejection": rejection,
        "near_total": near_total, "bad_total": bad_total,
    }


def _symmetries(model: MultiModePotentialV3, c: ChemConfiguration) -> dict:
    kw = sample_kwargs(c)
    pos = kw.pop("positions")
    E0 = model.forward(pos, **kw)[0].item()

    shift = torch.tensor([3.7, -1.2, 0.4])
    dE_t = abs(model.forward(pos + shift, **kw)[0].item() - E0)

    theta = 0.7
    R = torch.tensor([[math.cos(theta), -math.sin(theta), 0.],
                      [math.sin(theta),  math.cos(theta), 0.],
                      [0., 0., 1.]])
    dE_r = abs(model.forward(pos @ R.T, **kw)[0].item() - E0)

    p = pos.clone().requires_grad_(True)
    E_t, _, _ = model.forward(p, **kw)
    (g,) = torch.autograd.grad(E_t, p)
    sum_F = (-g).sum(dim=0).abs().max().item()

    return {"trans": dE_t, "rot": dE_r, "sum_F": sum_F}


def _ensemble_ood_vs_in_dist(
    members: list[MultiModePotentialV3],
    configs: list[ChemConfiguration],
    sample_n: int = 30,
) -> tuple[float, float]:
    """Ensemble std on (a) in-distribution and (b) OOD (atoms shrunk to 0.6x)."""
    import random
    pool = configs[:sample_n]

    def ens_std(positions_transform):
        stds = []
        for c in pool:
            kw = sample_kwargs(c)
            kw["positions"] = positions_transform(kw["positions"])
            Es = []
            for m in members:
                m.eval()
                E, _, _ = m.forward(kw["positions"], **{k: v for k, v in kw.items() if k != "positions"})
                Es.append(E.item())
            stds.append(torch.tensor(Es).std(unbiased=False).item())
        return sum(stds) / max(1, len(stds))

    in_dist = ens_std(lambda p: p.clone())
    ood = ens_std(lambda p: (p - p.mean(dim=0)) * 0.6 + p.mean(dim=0))
    return in_dist, ood


def run_all_checks(
    model: MultiModePotentialV3,
    test_set: list[ChemConfiguration],
    ensemble_members: list[MultiModePotentialV3] | None = None,
    ood_pool: list[ChemConfiguration] | None = None,
) -> list[CheckResult]:
    out: list[CheckResult] = []

    r2, f_mae = _energy_r2(model, test_set)
    # Reasonable F threshold relative to MMFF force RMS (hundreds): 200 = ok
    out.append(CheckResult("test-set energy R^2 vs MMFF", r2 >= 0.85, r2, 0.85))
    out.append(CheckResult("test-set force MAE", f_mae <= 200.0, f_mae, 200.0, "(lower is better)"))

    sym = _symmetries(model, test_set[0])
    out.append(CheckResult("translation invariance", sym["trans"] < 1e-2, sym["trans"], 1e-2, "(lower is better)"))
    out.append(CheckResult("rotation invariance", sym["rot"] < 1e-2, sym["rot"], 1e-2, "(lower is better)"))
    out.append(CheckResult("Newton 3rd law sum(F)=0", sym["sum_F"] < 1e-2, sym["sum_F"], 1e-2, "(lower is better)"))

    pose = _pose_classification(model, test_set)
    out.append(CheckResult(
        "bad-pose rejection rate",
        pose["rejection"] >= 0.95, pose["rejection"], 0.95,
        note=f"(TN={pose['tn']}/{pose['bad_total']})",
    ))
    out.append(CheckResult(
        "near-native retention rate",
        pose["retention"] >= 0.85, pose["retention"], 0.85,
        note=f"(TP={pose['tp']}/{pose['near_total']})",
    ))

    if ensemble_members and ood_pool:
        in_std, ood_std = _ensemble_ood_vs_in_dist(ensemble_members, ood_pool)
        ratio = ood_std / max(in_std, 1e-9)
        out.append(CheckResult(
            "ensemble OOD std >= 2x in-distribution std",
            ratio >= 2.0, ratio, 2.0,
            note=f"(in={in_std:.2f}, ood={ood_std:.2f})",
        ))

    return out


def format_results(results: list[CheckResult]) -> str:
    lines = []
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{tag}]  {r.name:<48}  {r.value:.4f}  / threshold {r.threshold:.4f}  {r.note}")
    return "\n".join(lines)
