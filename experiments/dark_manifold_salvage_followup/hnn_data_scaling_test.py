"""HNN data-scaling smoke test.

Re-runs the original salvage HNN test at higher data densities to
distinguish data-sparsity failure from structural failure.

Original salvage result (April 2026): HNN passed on the harmonic
oscillator (conservative dynamics) but failed 14× worse than a plain
MLP on the damped oscillator (dissipative dynamics). Hamilton's
equations preserve total energy by construction, so an HNN
*architecturally cannot* represent dissipation. The expected outcome
of this re-test is that HNN's damped failure persists at all data
conditions, confirming a structural — not data-sparsity — failure.

Systems
-------

  Harmonic oscillator (conservative):
      m * x'' + k * x = 0
      H(q, p) = p^2/(2m) + (1/2) * k * q^2

  Damped oscillator (dissipative):
      m * x'' + b * x' + k * x = 0
      No conserved Hamiltonian.

Both with m=1, k=1, b=0.5.

Models
------

  HNN: scalar net H_theta(q, p), with predicted (q_dot, p_dot) =
       (dH/dp, -dH/dq) computed by autograd.
  MLP: predicts (q_dot, p_dot) directly from (q, p).

Both with hidden width 128, depth 3.

Training data
-------------

For each system, generate trajectories from random (q0, p0) initial
conditions integrated by RK4 to t=10s with dt=0.01. Sample uniformly
from the union of all trajectory points.

Conditions: n_train in {100, 1000, 10000, 100000} state-time samples.

Test
----

Held-out trajectories with NEW random initial conditions, rolled
forward by RK4 using the model's predicted (q_dot, p_dot) at each
step. Metrics:

  * trajectory MSE on (q, p) over the entire 10s window
  * energy drift over the 10s window (harmonic only — relevant for
    the conservative case)
  * final position error |q_pred - q_true| at t=10s

Output
------

  results/hnn_results.csv with columns:
    system, model, n_train, traj_mse, energy_drift, final_pos_error,
    n_params, steps, wall_s
  results/plots/hnn_data_scaling.png

Determinism
-----------

seed=42 fixed for trajectory generation, sample selection, and model
init.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
SEED = 42

# System parameters (shared for both)
M = 1.0
K = 1.0
B_DAMPED = 0.5
DT = 0.01
T_END = 10.0


# ============================================================================
# Ground-truth dynamics (RK4 integration of the analytical (q_dot, p_dot))
# ============================================================================
def true_dynamics_harmonic(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return p / M, -K * q


def true_dynamics_damped(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return p / M, -K * q - B_DAMPED * (p / M)


def rk4_step(state: np.ndarray, dt: float, dyn) -> np.ndarray:
    """state: (..., 2) where last dim is (q, p)."""
    q, p = state[..., 0], state[..., 1]
    k1q, k1p = dyn(q, p)
    k2q, k2p = dyn(q + 0.5 * dt * k1q, p + 0.5 * dt * k1p)
    k3q, k3p = dyn(q + 0.5 * dt * k2q, p + 0.5 * dt * k2p)
    k4q, k4p = dyn(q + dt * k3q, p + dt * k3p)
    new_q = q + (dt / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
    new_p = p + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
    return np.stack([new_q, new_p], axis=-1)


def integrate_trajectory(q0: float, p0: float, dyn, t_end: float, dt: float):
    n_steps = int(round(t_end / dt))
    state = np.array([q0, p0], dtype=np.float64)
    states = np.zeros((n_steps + 1, 2), dtype=np.float64)
    states[0] = state
    for i in range(n_steps):
        state = rk4_step(state, dt, dyn)
        states[i + 1] = state
    return states


def gen_training_pool(system: str, n_traj: int, rng: np.random.Generator):
    """Return (states, derivs) flat across all trajectories.
    states[i] = (q_i, p_i); derivs[i] = (q_dot_i, p_dot_i)."""
    dyn = true_dynamics_harmonic if system == "harmonic" else true_dynamics_damped
    all_states = []
    all_derivs = []
    n_per = int(round(T_END / DT))
    for _ in range(n_traj):
        q0 = rng.uniform(-1.0, 1.0)
        p0 = rng.uniform(-1.0, 1.0)
        states = integrate_trajectory(q0, p0, dyn, T_END, DT)
        # Drop the very last state to keep states/derivs aligned by step.
        s = states[:-1]
        qd, pd = dyn(s[:, 0], s[:, 1])
        d = np.stack([qd, pd], axis=-1)
        all_states.append(s)
        all_derivs.append(d)
    states = np.concatenate(all_states, axis=0).astype(np.float32)
    derivs = np.concatenate(all_derivs, axis=0).astype(np.float32)
    return states, derivs


# ============================================================================
# Models
# ============================================================================
class MLPDeriv(nn.Module):
    """Plain MLP: (q, p) -> (q_dot, p_dot) directly."""

    def __init__(self, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d = 2
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.Tanh())
            d = hidden
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, qp):
        return self.net(qp)


class HNN(nn.Module):
    """Hamiltonian Neural Network (Greydanus et al. 2019).

    Learns scalar H(q, p); produces (q_dot, p_dot) =
    (dH/dp, -dH/dq) by autograd.
    """

    def __init__(self, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d = 2
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.Tanh())
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.H = nn.Sequential(*layers)

    def forward(self, qp):
        qp = qp.requires_grad_(True)
        H = self.H(qp).sum()
        grad = torch.autograd.grad(H, qp, create_graph=True)[0]
        dH_dq = grad[..., 0]
        dH_dp = grad[..., 1]
        q_dot = dH_dp
        p_dot = -dH_dq
        return torch.stack([q_dot, p_dot], dim=-1)


# ============================================================================
# Train / eval
# ============================================================================
def train_one(model, X, Y, *, lr=1e-3, batch=512, max_steps=2000,
              wall_cap_s=300.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    batch = min(batch, n)
    t0 = time.time()
    steps_done = 0
    for step in range(max_steps):
        idx = rng.integers(0, n, size=batch)
        xb = X[idx]
        yb = Y[idx]
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps_done = step + 1
        if (time.time() - t0) > wall_cap_s:
            break
    wall = time.time() - t0
    return {"steps": steps_done, "wall_s": wall, "final_loss": float(loss.item())}


def rollout(model, q0: float, p0: float, dt: float, n_steps: int) -> np.ndarray:
    """RK4 rollout using model's predicted (q_dot, p_dot)."""
    state = np.array([q0, p0], dtype=np.float64)
    out = np.zeros((n_steps + 1, 2), dtype=np.float64)
    out[0] = state

    def f(s):
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        with torch.enable_grad():
            d = model(s_t).detach().numpy()[0]
        return d.astype(np.float64)

    for i in range(n_steps):
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = state
    return out


def eval_trajectories(model, system: str, n_test: int = 8) -> dict:
    """Roll out from random ICs, compare to ground truth."""
    rng = np.random.default_rng(SEED + 100)
    dyn = true_dynamics_harmonic if system == "harmonic" else true_dynamics_damped
    n_steps = int(round(T_END / DT))
    traj_se = []
    energy_drifts = []
    final_errs = []
    for _ in range(n_test):
        q0 = rng.uniform(-1.0, 1.0)
        p0 = rng.uniform(-1.0, 1.0)
        truth = integrate_trajectory(q0, p0, dyn, T_END, DT)
        pred = rollout(model, q0, p0, DT, n_steps)
        # Trajectory MSE in (q, p) space
        traj_se.append(((pred - truth) ** 2).mean(axis=1))
        # Energy drift (harmonic case): H = p^2/2m + k q^2 / 2
        e0 = 0.5 * (p0 * p0) / M + 0.5 * K * (q0 * q0)
        if e0 > 1e-12:
            ep = 0.5 * (pred[:, 1] ** 2) / M + 0.5 * K * (pred[:, 0] ** 2)
            energy_drifts.append(np.abs(ep[-1] - e0) / e0)
        # Final position error
        final_errs.append(abs(pred[-1, 0] - truth[-1, 0]))
    return {
        "traj_mse": float(np.mean([s.mean() for s in traj_se])),
        "energy_drift": float(np.mean(energy_drifts)) if energy_drifts else float("nan"),
        "final_pos_error": float(np.mean(final_errs)),
    }


# ============================================================================
# Sweep
# ============================================================================
def make_models() -> dict[str, callable]:
    return {
        "MLP": lambda: MLPDeriv(hidden=128, depth=3),
        "HNN": lambda: HNN(hidden=128, depth=3),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-conditions", default="100,1000,10000,100000")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--wall-cap-s", type=float, default=300.0)
    parser.add_argument("--out-csv", default=str(RESULTS_DIR / "hnn_results.csv"))
    parser.add_argument("--out-plot",
                        default=str(PLOTS_DIR / "hnn_data_scaling.png"))
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    n_conditions = [int(x) for x in args.n_conditions.split(",")]
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    rows: list[dict] = []
    n_per_traj = int(round(T_END / DT))  # 1000 samples per trajectory

    for system in ("harmonic", "damped"):
        print(f"\n=== system: {system} ===")
        # Generate one big training pool — sample n_train points from it
        # for each condition. The pool size scales with the largest
        # condition; for n=100k we need 100 trajectories.
        max_cond = max(n_conditions)
        n_traj = (max_cond + n_per_traj - 1) // n_per_traj
        rng = np.random.default_rng(SEED + (0 if system == "harmonic" else 1))
        states_all, derivs_all = gen_training_pool(system, n_traj, rng)
        print(f"  pool: {len(states_all)} state-deriv pairs from {n_traj} trajectories")

        for n_train in n_conditions:
            n_train_eff = min(n_train, len(states_all))
            sample_rng = np.random.default_rng(SEED + 1000)
            idx = sample_rng.choice(len(states_all), size=n_train_eff, replace=False)
            X = torch.from_numpy(states_all[idx])
            Y = torch.from_numpy(derivs_all[idx])
            for model_name, build in make_models().items():
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                model = build()
                n_params = sum(p.numel() for p in model.parameters())
                print(f"  [run] n_train={n_train_eff:7d}  {model_name:4s}  "
                      f"params={n_params:6d}", end="", flush=True)
                t0 = time.time()
                tr = train_one(
                    model, X, Y,
                    max_steps=args.max_steps, wall_cap_s=args.wall_cap_s,
                )
                ev = eval_trajectories(model, system, n_test=8)
                wall = time.time() - t0
                print(
                    f"  traj_mse={ev['traj_mse']:.4e}  "
                    f"energy_drift={ev['energy_drift']:.3e}  "
                    f"final_pos_err={ev['final_pos_error']:.3e}  "
                    f"steps={tr['steps']}  wall={wall:.1f}s"
                )
                rows.append({
                    "system": system,
                    "model": model_name,
                    "n_train": n_train_eff,
                    "n_params": n_params,
                    "traj_mse": ev["traj_mse"],
                    "energy_drift": ev["energy_drift"],
                    "final_pos_error": ev["final_pos_error"],
                    "steps": tr["steps"],
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

    # ----- plot: 2 subplots (harmonic, damped), HNN vs MLP -----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, system in zip(axes, ("harmonic", "damped")):
        for model_name in ("MLP", "HNN"):
            pts = sorted(
                (r["n_train"], r["traj_mse"]) for r in rows
                if r["system"] == system and r["model"] == model_name
            )
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=model_name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n_train (state-deriv samples)")
        ax.set_ylabel("trajectory MSE")
        ax.set_title(f"HNN vs MLP — {system} oscillator")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=130)
    print(f"[ok] wrote {args.out_plot}")


if __name__ == "__main__":
    main()
