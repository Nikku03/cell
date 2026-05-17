"""Verification suite for the multi-mode stateful potential.

Re-runs every v1 symmetry check on the v2 architecture, then adds new
checks that exercise the new properties: persistent state, chemical
context, bond awareness, mode decomposition, ensemble uncertainty.

Each check returns a CheckResult with a numeric margin and threshold.
The demo prints PASS / FAIL for each.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .dataset import RichConfiguration
from .multimode import MultiModePotential
from .train_v2 import sample_to_kwargs
from .uncertainty import PotentialEnsemble


@dataclass
class CheckResult:
    name: str
    passed: bool
    value: float
    threshold: float
    note: str = ""


def _energy(model: MultiModePotential, kw: dict) -> float:
    kw2 = {k: v for k, v in kw.items() if k not in ("positions", "energy", "forces")}
    pos = kw["positions"].clone()
    return model.forward(pos, **kw2)[0].item()


def _accuracy(model: MultiModePotential, test_set: list[RichConfiguration]) -> tuple[float, float]:
    E_true, E_pred = [], []
    f_abs_sum = 0.0
    f_count = 0
    for c in test_set:
        kw = sample_to_kwargs(c)
        pos = kw.pop("positions").clone().requires_grad_(True)
        E_t = kw.pop("energy")
        F_t = kw.pop("forces")
        E, _ = model.forward(pos, **kw)
        (g,) = torch.autograd.grad(E, pos, allow_unused=True)
        if g is None:
            g = torch.zeros_like(pos)
        F = -g
        E_true.append(E_t.item())
        E_pred.append(E.item())
        f_abs_sum += (F - F_t).abs().sum().item()
        f_count += F_t.numel()
    Et = torch.tensor(E_true)
    Ep = torch.tensor(E_pred)
    ss_res = ((Et - Ep) ** 2).sum().item()
    ss_tot = ((Et - Et.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, f_abs_sum / max(1, f_count)


def _build_canonical_sample(rng_seed: int = 7) -> dict:
    """A small fixed-shape sample for symmetry / property tests."""
    torch.manual_seed(rng_seed)
    N = 5
    pos = torch.tensor([[0.0, 0, 0], [1.55, 0, 0], [3.1, 0.2, 0],
                        [4.2, 1.0, 0], [0.4, 1.7, 0]])
    Z = torch.tensor([6, 6, 8, 6, 6], dtype=torch.long)
    hyb = torch.tensor([3, 3, 2, 2, 3], dtype=torch.long)
    fc = torch.zeros(N, dtype=torch.long)
    arom = torch.zeros(N, dtype=torch.bool)
    ring = torch.zeros(N, dtype=torch.bool)
    hd = torch.zeros(N, dtype=torch.bool)
    ha = torch.tensor([False, False, True, False, False])
    pq = torch.tensor([0.05, -0.05, -0.20, 0.10, 0.10])
    bonds = torch.tensor([[0, 1]], dtype=torch.long)
    btype = torch.tensor([1], dtype=torch.long)
    return {
        "positions": pos, "Z": Z, "hybridisation": hyb,
        "formal_charge": fc, "aromatic": arom, "in_ring": ring,
        "h_donor": hd, "h_acceptor": ha, "partial_charge": pq,
        "bonds": bonds, "bond_type": btype,
    }


def _symmetries(model: MultiModePotential) -> dict:
    base = _build_canonical_sample()
    E0 = _energy(model, base)
    shift = torch.tensor([3.7, -1.2, 0.4])
    base_t = dict(base, positions=base["positions"] + shift)
    dE_t = abs(_energy(model, base_t) - E0)
    theta = 0.7
    R = torch.tensor([[math.cos(theta), -math.sin(theta), 0.],
                      [math.sin(theta),  math.cos(theta), 0.],
                      [0., 0., 1.]])
    base_r = dict(base, positions=base["positions"] @ R.T)
    dE_r = abs(_energy(model, base_r) - E0)

    # Same-element permutation: swap atoms 3 and 4 (both Z=6 in base)
    perm = [0, 1, 2, 4, 3]
    kw_perm = {}
    for k, v in base.items():
        if k == "bonds":
            inv = [perm.index(i) for i in range(len(perm))]
            new_b = []
            for b in v.tolist():
                ni, nj = inv[b[0]], inv[b[1]]
                if ni > nj:
                    ni, nj = nj, ni
                new_b.append([ni, nj])
            kw_perm[k] = torch.tensor(new_b, dtype=torch.long)
        elif k == "bond_type":
            kw_perm[k] = v
        else:
            kw_perm[k] = v[perm] if hasattr(v, "__getitem__") and v.shape[0] == len(perm) else v
    dE_p = abs(_energy(model, kw_perm) - E0)

    # Newton's 3rd law: forces sum to zero.
    kw = {k: v for k, v in base.items() if k != "positions"}
    pos = base["positions"].clone().requires_grad_(True)
    E, _ = model.forward(pos, **kw)
    (g,) = torch.autograd.grad(E, pos, allow_unused=True)
    if g is None:
        g = torch.zeros_like(pos)
    sum_F = (-g).sum(dim=0).abs().max().item()
    return {"trans": dE_t, "rot": dE_r, "perm": dE_p, "sum_F": sum_F}


def _state_persistence(model: MultiModePotential) -> tuple[float, float]:
    """Run forward twice with the same input but different initial_state.

    Returns (|E_with_state - E_no_state|, |final_state - initial_state|).
    The first should be > 0 (state has causal effect). The second
    confirms that the GRU produces an updated state.
    """
    base = _build_canonical_sample()
    kw = {k: v for k, v in base.items() if k != "positions"}
    pos = base["positions"].clone()
    E_none, state_none = model.forward(pos, **kw)
    # Run again with the produced state as initial_state.
    E_with, state_with = model.forward(pos, **kw, initial_state=state_none)
    dE = abs(E_with.item() - E_none.item())
    dS = (state_with - state_none).abs().max().item()
    return dE, dS


def _bond_sensitivity(model: MultiModePotential) -> float:
    base = _build_canonical_sample()
    E_with_bond = _energy(model, base)
    no_bond = dict(base,
                   bonds=torch.empty(0, 2, dtype=torch.long),
                   bond_type=torch.empty(0, dtype=torch.long))
    E_no_bond = _energy(model, no_bond)
    return abs(E_with_bond - E_no_bond)


def _chemistry_sensitivity(model: MultiModePotential) -> float:
    base = _build_canonical_sample()
    E0 = _energy(model, base)
    perturbed = dict(base,
                     hybridisation=torch.tensor([1, 1, 1, 1, 1], dtype=torch.long))
    E_perturbed = _energy(model, perturbed)
    return abs(E_perturbed - E0)


def _mode_decomposition(model: MultiModePotential) -> tuple[float, dict]:
    """Verify total energy = sum of mode contributions (within float noise)."""
    base = _build_canonical_sample()
    kw = {k: v for k, v in base.items() if k != "positions"}
    pos = base["positions"].clone()
    E, _, breakdown = model.forward(pos, **kw, return_breakdown=True)
    # The breakdown's bonded/nonbond/electro are sums of raw per-atom
    # heads BEFORE the global scale/shift. Add the scale*sum + shift*N
    # to reconstruct the total.
    raw_sum = breakdown["bonded"] + breakdown["nonbond"] + breakdown["electro"]
    reconstructed = raw_sum * breakdown["scale"] + breakdown["shift"]
    diff = abs(E.item() - reconstructed.item())
    return diff, {
        "total": E.item(),
        "bonded": breakdown["bonded"].item(),
        "nonbond": breakdown["nonbond"].item(),
        "electro": breakdown["electro"].item(),
    }


def _uncertainty_in_vs_out(
    ensemble: PotentialEnsemble, in_dist_set: list[RichConfiguration]
) -> tuple[float, float]:
    """Compare ensemble std on in-distribution samples vs OOD samples.

    OOD geometry: take the in-distribution sample and shrink all
    pairwise distances by 0.55x (atoms much closer than training range).
    """
    # In-distribution
    in_stds = []
    for c in in_dist_set[:30]:
        sample = sample_to_kwargs(c)
        sample = {k: v for k, v in sample.items() if k not in ("energy", "forces")}
        out = ensemble.predict(sample)
        in_stds.append(out["energy_std"])
    in_mean = sum(in_stds) / len(in_stds)
    # OOD: shrink positions of each sample
    out_stds = []
    for c in in_dist_set[:30]:
        sample = sample_to_kwargs(c)
        sample = {k: v for k, v in sample.items() if k not in ("energy", "forces")}
        sample["positions"] = sample["positions"] * 0.55
        out = ensemble.predict(sample)
        out_stds.append(out["energy_std"])
    out_mean = sum(out_stds) / len(out_stds)
    return in_mean, out_mean


def run_all_checks(
    model: MultiModePotential,
    test_set: list[RichConfiguration],
    ensemble: PotentialEnsemble | None = None,
    train_set_for_ood: list[RichConfiguration] | None = None,
) -> list[CheckResult]:
    out: list[CheckResult] = []

    r2, f_mae = _accuracy(model, test_set)
    out.append(CheckResult("test-set energy R^2", r2 >= 0.85, r2, 0.85))
    out.append(CheckResult("test-set force MAE", f_mae <= 1.0, f_mae, 1.0,
                           note="(lower is better)"))

    sym = _symmetries(model)
    out.append(CheckResult("translation invariance", sym["trans"] < 1e-3, sym["trans"], 1e-3, "(lower is better)"))
    out.append(CheckResult("rotation invariance", sym["rot"] < 1e-3, sym["rot"], 1e-3, "(lower is better)"))
    out.append(CheckResult("permutation invariance", sym["perm"] < 1e-3, sym["perm"], 1e-3, "(lower is better)"))
    out.append(CheckResult("Newton 3rd law sum(F)=0", sym["sum_F"] < 1e-3, sym["sum_F"], 1e-3, "(lower is better)"))

    dE_state, dS = _state_persistence(model)
    out.append(CheckResult("persistent state has causal effect on energy",
                           dE_state > 1e-3, dE_state, 1e-3,
                           note="(higher is better; |E_with_state - E_cold_start|)"))
    out.append(CheckResult("GRU produces a non-trivial state update",
                           dS > 1e-2, dS, 1e-2,
                           note="(higher is better; max-elementwise change of state)"))

    dE_bond = _bond_sensitivity(model)
    out.append(CheckResult("bond label affects predicted energy",
                           dE_bond > 0.05, dE_bond, 0.05,
                           note="(higher is better; |E_with_bond - E_without|)"))

    dE_chem = _chemistry_sensitivity(model)
    out.append(CheckResult("chemistry context affects predicted energy",
                           dE_chem > 0.01, dE_chem, 0.01,
                           note="(higher is better; |E_with_chem - E_alt|)"))

    decomp_err, decomp = _mode_decomposition(model)
    out.append(CheckResult("E_total = sum of per-mode energies",
                           decomp_err < 1e-3, decomp_err, 1e-3,
                           note=f"(lower is better; bonded={decomp['bonded']:.3f}, "
                                f"nonbond={decomp['nonbond']:.3f}, "
                                f"electro={decomp['electro']:.3f})"))

    if ensemble is not None and train_set_for_ood is not None:
        in_std, ood_std = _uncertainty_in_vs_out(ensemble, train_set_for_ood)
        out.append(CheckResult("ensemble uncertainty: OOD > in-distribution",
                               ood_std > in_std * 1.5, ood_std / max(in_std, 1e-9),
                               1.5,
                               note=f"(higher is better; in={in_std:.4f}, ood={ood_std:.4f})"))

    return out


def format_results(results: list[CheckResult]) -> str:
    lines = []
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{tag}]  {r.name:<48}  {r.value:.4f}  / threshold {r.threshold:.4f}  {r.note}")
    return "\n".join(lines)
