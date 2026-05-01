"""SIREN data-scaling smoke test.

Re-runs the original salvage test at higher data densities to
distinguish data-sparsity failure from structural failure.

Original salvage result (April 2026): SIREN failed to beat a plain
MLP on a 3D synthetic metabolite field at 500 training points
(test MSE 0.039 SIREN vs 0.008 MLP). This script tests whether
SIREN catches up at 5k, 50k, 500k training points.

Synthetic field
---------------

A 40^3 grid scalar field. Total grid points = 64000; interior
(strip the outermost shell) = 38^3 = 54872, ~57k as per the
original spec. Field components:

  1. Three Gaussian "ribosome blobs" at random centers, sigma=0.05
     (in [0,1]^3 coords). Smooth low-frequency content.
  2. Radial "membrane gradient" centered on the cube center.
     Smooth large-scale gradient.
  3. High-frequency "crowding" via sin(20*pi*x) * cos(20*pi*y) *
     sin(20*pi*z) * 0.15.

The high-frequency component is the load-bearing test for SIREN —
plain MLPs struggle to represent high spatial frequencies. SIREN
with omega=30 was specifically designed for this.

Models tested
-------------

  * MLP-ReLU         baseline
  * Fourier-MLP      baseline with sin/cos input embedding
  * SIREN(omega=10)  lower-frequency variant
  * SIREN(omega=30)  high-frequency variant (matches original spec)

All models share the same hidden-width 128, depth 3, training
config (Adam lr=1e-3, batch=2048, max 3000 steps with halt criterion
of 5 minutes wall-time per (model, condition) pair).

Output
------

  results/siren_results.csv with columns:
    model, n_train, train_mse, test_mse, test_r2, grad_cosine, wall_s
  results/plots/siren_data_scaling.png

Determinism
-----------

seed=42 fixed for the field, sample selection, and model init.

Honest limitations
------------------

* CPU-only. The 500k-point condition uses a step budget rather than
  a full convergence run; in particular, a very-high-data MLP may not
  reach its true asymptote within 3000 steps.
* This is a smoke test, not a hyperparameter sweep. The widths,
  depths, learning rates, and step budgets are fixed at reasonable
  defaults from the SIREN paper (Sitzmann et al. 2020). Tuning each
  model would change the absolute numbers; the relative comparison
  across data conditions is the point.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
SEED = 42


# ============================================================================
# Synthetic field
# ============================================================================
def synthesize_field_value(coords: torch.Tensor) -> torch.Tensor:
    """Return scalar field f(x,y,z) for coords in [0,1]^3."""
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    # Gaussian ribosome blobs
    g = torch.zeros_like(x)
    rng = np.random.default_rng(SEED)
    centers = rng.uniform(0.2, 0.8, size=(3, 3))
    sigma = 0.05
    for cx, cy, cz in centers:
        dx = x - float(cx)
        dy = y - float(cy)
        dz = z - float(cz)
        g = g + torch.exp(-(dx * dx + dy * dy + dz * dz) / (2 * sigma * sigma))

    # Radial membrane gradient (centered)
    rx, ry, rz = x - 0.5, y - 0.5, z - 0.5
    radial = torch.sqrt(rx * rx + ry * ry + rz * rz)
    membrane = 1.0 - torch.tanh(8.0 * (radial - 0.4))

    # High-frequency crowding (the SIREN test)
    pi = math.pi
    crowd = (
        torch.sin(20.0 * pi * x)
        * torch.cos(20.0 * pi * y)
        * torch.sin(20.0 * pi * z)
        * 0.15
    )

    return 0.45 * g + 0.30 * membrane + crowd


def sample_interior_points(n: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample n points in (1/40, 39/40)^3 (interior of 40^3 grid)."""
    lo = 1.0 / 40.0
    hi = 39.0 / 40.0
    coords_np = rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)
    coords = torch.from_numpy(coords_np)
    values = synthesize_field_value(coords).unsqueeze(-1)
    return coords, values


# ============================================================================
# Models
# ============================================================================
class MLP(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 128, depth: int = 3,
                 activation=nn.ReLU):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(activation())
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FourierMLP(nn.Module):
    """Plain MLP with a sin/cos input embedding at fixed frequencies."""

    def __init__(self, in_dim: int = 3, hidden: int = 128, depth: int = 3,
                 n_freqs: int = 8):
        super().__init__()
        # Geometric frequencies 2^0 ... 2^(n_freqs-1)
        freqs = 2.0 ** torch.arange(n_freqs).float()
        self.register_buffer("freqs", freqs)
        embed_dim = in_dim * n_freqs * 2
        self.mlp = MLP(in_dim=embed_dim, hidden=hidden, depth=depth)

    def forward(self, x):
        # x: (N, 3) -> (N, 3*n_freqs*2)
        x_proj = x.unsqueeze(-1) * self.freqs * math.pi
        emb = torch.cat(
            [torch.sin(x_proj), torch.cos(x_proj)], dim=-1
        ).flatten(start_dim=-2)
        return self.mlp(emb)


class SineLayer(nn.Module):
    """SIREN sine layer (Sitzmann et al. 2020)."""

    def __init__(self, in_features: int, out_features: int,
                 is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 128, depth: int = 3,
                 omega_0: float = 30.0, hidden_omega_0: float | None = None):
        super().__init__()
        if hidden_omega_0 is None:
            hidden_omega_0 = omega_0
        layers: list[nn.Module] = [
            SineLayer(in_dim, hidden, is_first=True, omega_0=omega_0)
        ]
        for _ in range(depth - 1):
            layers.append(
                SineLayer(hidden, hidden, is_first=False,
                          omega_0=hidden_omega_0)
            )
        # Final linear layer (not sine) — SIREN convention
        final = nn.Linear(hidden, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden) / hidden_omega_0
            final.weight.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Train / eval
# ============================================================================
def train_one(
    model: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    *,
    lr: float = 1e-3,
    batch_size: int = 2048,
    max_steps: int = 3000,
    wall_cap_s: float = 300.0,
) -> dict:
    n_train = X_train.shape[0]
    batch_size = min(batch_size, n_train)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    steps_done = 0
    final_loss = float("nan")
    for step in range(max_steps):
        idx = rng.integers(0, n_train, size=batch_size)
        xb = X_train[idx]
        yb = Y_train[idx]
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
        steps_done = step + 1
        if (time.time() - t0) > wall_cap_s:
            break
    wall = time.time() - t0
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train)
        test_pred = model(X_test)
        train_mse = float(((train_pred - Y_train) ** 2).mean().item())
        test_mse = float(((test_pred - Y_test) ** 2).mean().item())
        ss_res = float(((test_pred - Y_test) ** 2).sum().item())
        ss_tot = float(((Y_test - Y_test.mean()) ** 2).sum().item())
        test_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "test_r2": test_r2,
        "final_step_loss": final_loss,
        "steps": steps_done,
        "wall_s": wall,
    }


def gradient_cosine(model: nn.Module, X: torch.Tensor) -> float:
    """Mean cosine similarity between predicted and ground-truth gradients
    over X. Uses autograd on both."""
    model.eval()
    X_test = X.clone().detach().requires_grad_(True)
    pred = model(X_test).sum()
    pred_grad = torch.autograd.grad(pred, X_test, create_graph=False)[0]

    Xg = X.clone().detach().requires_grad_(True)
    truth = synthesize_field_value(Xg).sum()
    truth_grad = torch.autograd.grad(truth, Xg)[0]

    dot = (pred_grad * truth_grad).sum(dim=-1)
    pn = pred_grad.norm(dim=-1).clamp(min=1e-9)
    tn = truth_grad.norm(dim=-1).clamp(min=1e-9)
    cos = (dot / (pn * tn)).mean().item()
    return float(cos)


# ============================================================================
# Sweep
# ============================================================================
def make_models() -> dict[str, callable]:
    return {
        "MLP-ReLU": lambda: MLP(in_dim=3, hidden=128, depth=3),
        "Fourier-MLP": lambda: FourierMLP(in_dim=3, hidden=128, depth=3, n_freqs=8),
        "SIREN-w10": lambda: SIREN(in_dim=3, hidden=128, depth=3, omega_0=10.0),
        "SIREN-w30": lambda: SIREN(in_dim=3, hidden=128, depth=3, omega_0=30.0),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-conditions", default="500,5000,50000,500000")
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--wall-cap-s", type=float, default=300.0,
                        help="halt criterion: max wall time per (model, condition)")
    parser.add_argument("--out-csv", default=str(RESULTS_DIR / "siren_results.csv"))
    parser.add_argument("--out-plot",
                        default=str(PLOTS_DIR / "siren_data_scaling.png"))
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    n_conditions = [int(x) for x in args.n_conditions.split(",")]
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    rng = np.random.default_rng(SEED)
    print(f"[info] sampling shared test set: {args.n_test} points")
    X_test, Y_test = sample_interior_points(args.n_test, rng)

    rows: list[dict] = []
    models = make_models()

    for n_train in n_conditions:
        print(f"\n=== n_train = {n_train} ===")
        # Sample training set freshly per condition (deterministic via shared rng).
        train_rng = np.random.default_rng(SEED + 1)
        X_train, Y_train = sample_interior_points(n_train, train_rng)
        for model_name, build in models.items():
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            model = build()
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  [run] {model_name:14s}  params={n_params:6d}", end="", flush=True)
            t0 = time.time()
            res = train_one(
                model, X_train, Y_train, X_test, Y_test,
                max_steps=args.max_steps, wall_cap_s=args.wall_cap_s,
            )
            grad_cos = gradient_cosine(model, X_test[:512])
            wall = time.time() - t0
            print(
                f"  train_mse={res['train_mse']:.4e}  "
                f"test_mse={res['test_mse']:.4e}  "
                f"test_r2={res['test_r2']:+.3f}  "
                f"grad_cos={grad_cos:+.3f}  "
                f"steps={res['steps']}  wall={wall:.1f}s"
            )
            rows.append({
                "model": model_name,
                "n_train": n_train,
                "n_params": n_params,
                "train_mse": res["train_mse"],
                "test_mse": res["test_mse"],
                "test_r2": res["test_r2"],
                "grad_cosine": grad_cos,
                "steps": res["steps"],
                "wall_s": wall,
            })

    # ----- write CSV -----
    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n[ok] wrote {args.out_csv}")

    # ----- plot -----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_model: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append((r["n_train"], r["test_mse"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, pts in by_model.items():
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=model_name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_train (sampled points)")
    ax.set_ylabel("test MSE")
    ax.set_title("SIREN data-scaling smoke test\n3D synthetic metabolite field, n_test=5000")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=130)
    print(f"[ok] wrote {args.out_plot}")


if __name__ == "__main__":
    main()
