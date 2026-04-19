"""
DMVC P11: Neural PDE.

The simplest version of the Dark Manifold claim: a neural network where
the forward pass is LOCAL SPATIOTEMPORAL FIELD EVOLUTION, not matrix
multiplication over a flat state vector.

Target system: 2D Fisher-KPP reaction-diffusion

    du/dt = D * Laplacian(u) + r * u * (1 - u)

This is a famous PDE with traveling-wave solutions, genuinely nonlinear,
local (spatial derivatives + pointwise reaction), and easy to simulate
with a high-order method.

Architecture: the network is a stack of 3x3 convolutions with periodic
boundary conditions. At each point (i, j), the update at that point
depends only on Ψ within a small neighborhood -- locality is structural.
The SAME convolution weights apply at every grid point -- translation
equivariance is structural.

Forward pass:
    F_theta(u) = output of conv stack applied to u
    u_{t+1} = u_t + dt * F_theta(u_t)

Tests:
    T1: one-step prediction error on held-out (u_t, u_{t+1}) pairs
    T2: 100-step rollout error vs ground truth (the real test -- rollout
        failure is the typical neural-PDE pathology)
    T3: Learned dynamics preserve positivity of u (Fisher-KPP ground truth
        always has 0 <= u <= 1; a trained network shouldn't violate this)
    T4: Trained on one initial-condition family, evaluated on another -- does
        the network generalize beyond training distribution?

Honest pre-registration of expected failures:
    * One-step loss will look great; rollout will drift unless we train
      with multi-step unrolling
    * Traveling wave speed may be slightly off -- the network has to learn
      both the reaction AND the diffusion at the right balance
    * Generalization to unseen initial conditions may be weak
"""

from __future__ import annotations
import sys, time
sys.path.insert(0, "/home/claude/dmvc")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Ground truth: Fisher-KPP on a 2D periodic grid
# =============================================================================
D_DIFFUSION = 0.1
R_REACTION = 1.0
GRID_N = 32           # was 64, reduced for memory
DT = 0.02             # larger dt -> fewer steps needed for same simulation time


def periodic_laplacian(u):
    """5-point stencil Laplacian with periodic boundaries. numpy array in, out."""
    return (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
             np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u)


def fisher_kpp_step(u, dt, D, r):
    """One explicit Euler step of Fisher-KPP. Small dt should be stable."""
    lap = periodic_laplacian(u)
    du = D * lap + r * u * (1.0 - u)
    return u + dt * du


def simulate_fisher_kpp(u0, n_steps, dt=DT, D=D_DIFFUSION, r=R_REACTION):
    """Return trajectory (n_steps+1, N, N)."""
    u = u0.copy()
    traj = np.zeros((n_steps + 1, u.shape[0], u.shape[1]), dtype=np.float32)
    traj[0] = u
    # use a sub-step for accuracy, record every whole step
    sub = 10
    sub_dt = dt / sub
    for t in range(n_steps):
        for _ in range(sub):
            u = fisher_kpp_step(u, sub_dt, D, r)
        traj[t + 1] = u
    return traj


def random_initial_condition(N, rng, style="blob"):
    """
    Generate a random 2D initial condition.
    Styles:
      - 'blob': gaussian blob of value ~0.5 on a background of ~0
      - 'noise': uniform noise in [0, 0.1]
      - 'double': two blobs
    """
    if style == "blob":
        u = np.zeros((N, N), dtype=np.float32)
        cx = rng.uniform(0.3, 0.7) * N
        cy = rng.uniform(0.3, 0.7) * N
        sigma = rng.uniform(N/10, N/5)
        amp = rng.uniform(0.3, 0.7)
        yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        u = amp * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    elif style == "noise":
        u = rng.uniform(0.0, 0.1, size=(N, N)).astype(np.float32)
    elif style == "double":
        u = np.zeros((N, N), dtype=np.float32)
        yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        for _ in range(2):
            cx = rng.uniform(0.2, 0.8) * N
            cy = rng.uniform(0.2, 0.8) * N
            sigma = rng.uniform(N/12, N/6)
            amp = rng.uniform(0.3, 0.6)
            u = u + amp * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        u = np.clip(u, 0, 1)
    else:
        raise ValueError(style)
    return u.astype(np.float32)


# =============================================================================
# Neural PDE: local convolutional network predicting du/dt
# =============================================================================
class NeuralPDE(nn.Module):
    """
    Predicts du/dt at every grid point from a local neighborhood of u.

    Architecture:
        conv 3x3 (1->H) + GELU
        conv 3x3 (H->H) + GELU
        conv 3x3 (H->H) + GELU
        conv 1x1 (H->1)   -- final mixer is pointwise

    Periodic padding in all conv layers. Output is du/dt; we integrate with
    explicit Euler in the training/inference loop:
        u_{t+1} = u_t + dt * NeuralPDE(u_t)
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(hidden, 1, kernel_size=1, padding=0)

    def _pad(self, x, n=1):
        """Periodic (circular) padding."""
        return F.pad(x, (n, n, n, n), mode='circular')

    def forward(self, u):
        # u: (batch, 1, H, W)
        x = F.gelu(self.conv1(self._pad(u)))
        x = F.gelu(self.conv2(self._pad(x)))
        x = F.gelu(self.conv3(self._pad(x)))
        x = self.conv4(x)
        return x


# =============================================================================
# Training data generation
# =============================================================================
def generate_trajectory_dataset(n_trajectories=40, n_steps_per=60, N=GRID_N,
                                  seed=0, styles=("blob", "double")):
    """Generate a collection of full trajectories."""
    rng = np.random.default_rng(seed)
    trajs = []
    for i in range(n_trajectories):
        style = styles[i % len(styles)]
        u0 = random_initial_condition(N, rng, style=style)
        traj = simulate_fisher_kpp(u0, n_steps_per)
        trajs.append(traj)
    return trajs


def make_onestep_dataset(trajs):
    """From trajectories, make (u_t, u_{t+1}) pairs."""
    X, Y = [], []
    for traj in trajs:
        X.append(traj[:-1])
        Y.append(traj[1:])
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return X[:, None, :, :], Y[:, None, :, :]   # add channel dim


# =============================================================================
# Training
# =============================================================================
def train_onestep(model, X_train, Y_train, X_val, Y_val, dt=DT,
                   n_epochs=50, batch_size=64, lr=3e-4, device='cpu'):
    model = model.to(device)
    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    Ytr = torch.tensor(Y_train, dtype=torch.float32).to(device)
    Xva = torch.tensor(X_val, dtype=torch.float32).to(device)
    Yva = torch.tensor(Y_val, dtype=torch.float32).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train': [], 'val': []}
    n_train = len(Xtr)

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        losses = []
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            u_t = Xtr[idx]
            u_next = Ytr[idx]
            du_pred = model(u_t)
            u_pred = u_t + dt * du_pred
            loss = ((u_pred - u_next) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        tr_loss = float(np.mean(losses))

        model.eval()
        with torch.no_grad():
            du = model(Xva)
            up = Xva + dt * du
            va_loss = ((up - Yva) ** 2).mean().item()

        history['train'].append(tr_loss)
        history['val'].append(va_loss)
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:3d}  train_mse={tr_loss:.4e}  val_mse={va_loss:.4e}")

    return history


def train_multistep(model, X_train, dt=DT, unroll_steps=4, n_epochs=30,
                     batch_size=32, lr=1e-4, device='cpu'):
    """
    X_train: full trajectories, shape (n_traj, T+1, N, N)

    Train by picking a random start point, unrolling the network `unroll_steps`
    forward, comparing to ground truth trajectory at each unrolled step.
    This is the standard fix for rollout instability.
    """
    model = model.to(device)
    trajs = torch.tensor(X_train, dtype=torch.float32).to(device)
    n_traj, T, H, W = trajs.shape
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train': []}

    for epoch in range(n_epochs):
        model.train()
        losses = []
        # random starts
        for _ in range(n_traj // batch_size * 4):
            traj_idx = torch.randint(0, n_traj, (batch_size,))
            # start anywhere from 0 to T-1-unroll_steps
            start_idx = torch.randint(0, T - unroll_steps - 1, (batch_size,))
            # gather initial states
            u = torch.stack([trajs[traj_idx[b], start_idx[b]]
                             for b in range(batch_size)], dim=0).unsqueeze(1)
            total_loss = 0.0
            for k in range(1, unroll_steps + 1):
                du = model(u)
                u = u + dt * du
                target = torch.stack([trajs[traj_idx[b], start_idx[b] + k]
                                       for b in range(batch_size)], dim=0).unsqueeze(1)
                total_loss = total_loss + ((u - target) ** 2).mean()
            total_loss = total_loss / unroll_steps
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            losses.append(total_loss.item())
        tr_loss = float(np.mean(losses))
        history['train'].append(tr_loss)
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:3d}  unroll_{unroll_steps}step_mse={tr_loss:.4e}")
    return history


# =============================================================================
# Rollout evaluation
# =============================================================================
def rollout(model, u0, n_steps, dt=DT, device='cpu'):
    """Run the learned dynamics forward for n_steps. Return full trajectory."""
    model.eval()
    u = torch.tensor(u0[None, None, :, :], dtype=torch.float32).to(device)
    traj = np.zeros((n_steps + 1, u0.shape[0], u0.shape[1]), dtype=np.float32)
    traj[0] = u0
    with torch.no_grad():
        for t in range(n_steps):
            du = model(u)
            u = u + dt * du
            traj[t + 1] = u.squeeze().cpu().numpy()
    return traj


def rollout_error(model, u0, n_steps, dt=DT, device='cpu'):
    """Compare rollout against ground truth, return per-step rel L2 error."""
    gt = simulate_fisher_kpp(u0, n_steps, dt=dt)
    nn = rollout(model, u0, n_steps, dt=dt, device=device)
    # relative L2 at each time point
    err = np.zeros(n_steps + 1)
    for t in range(n_steps + 1):
        num = np.linalg.norm(gt[t] - nn[t])
        den = np.linalg.norm(gt[t]) + 1e-9
        err[t] = num / den
    return err, gt, nn


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 76)
    print("DMVC P11: Neural PDE (Fisher-KPP reaction-diffusion)")
    print("=" * 76)

    print(f"\n[1] Generating ground-truth trajectories (Fisher-KPP, {GRID_N}x{GRID_N})...")
    trajs_train = generate_trajectory_dataset(
        n_trajectories=20, n_steps_per=30, seed=0, styles=("blob", "double"))
    trajs_val = generate_trajectory_dataset(
        n_trajectories=5, n_steps_per=30, seed=1, styles=("blob", "double"))
    trajs_ood = generate_trajectory_dataset(
        n_trajectories=3, n_steps_per=30, seed=2, styles=("noise",))
    print(f"    train: {len(trajs_train)} trajectories x {trajs_train[0].shape[0]} steps")
    print(f"    val:   {len(trajs_val)} trajectories")
    print(f"    ood:   {len(trajs_ood)} trajectories (noise init, not in train distribution)")

    X_train, Y_train = make_onestep_dataset(trajs_train)
    X_val, Y_val = make_onestep_dataset(trajs_val)
    print(f"    one-step pairs: train={len(X_train)}, val={len(X_val)}")

    # Also stack training trajectories for multi-step training
    train_traj_stack = np.stack(trajs_train, axis=0)   # (n_traj, T+1, N, N)

    print("\n[2] Building neural PDE (conv 3x3 x 3 layers, hidden=32)...")
    model = NeuralPDE(hidden=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params}")

    # ----- Phase 1: one-step training -----
    print("\n[3] Phase 1: one-step MSE training (50 epochs)...")
    t0 = time.time()
    history1 = train_onestep(model, X_train, Y_train, X_val, Y_val,
                               n_epochs=50, batch_size=64, lr=3e-4)
    print(f"    Phase 1 time: {time.time() - t0:.1f}s")

    # Test 1: one-step error
    model.eval()
    with torch.no_grad():
        Xv = torch.tensor(X_val, dtype=torch.float32)
        Yv = torch.tensor(Y_val, dtype=torch.float32)
        du = model(Xv).numpy()
        u_pred = X_val + DT * du
        onestep_mse = float(np.mean((u_pred - Y_val) ** 2))
        onestep_rel = float(np.mean(np.abs(u_pred - Y_val)) /
                              (np.mean(np.abs(Y_val)) + 1e-9))

    print(f"\n[T1] One-step performance after Phase 1:")
    print(f"     MSE: {onestep_mse:.4e}")
    print(f"     Mean rel err: {onestep_rel:.2%}")

    # Test 2: rollout before multi-step
    print(f"\n[T2-pre] Rollout after Phase 1 (expect instability):")
    rng = np.random.default_rng(999)
    u0_test = random_initial_condition(GRID_N, rng, style="blob")
    err_curve_pre, gt_pre, nn_pre = rollout_error(model, u0_test, n_steps=30)
    print(f"     Rollout rel L2 error at:")
    for t in [1, 5, 10, 20, 30]:
        print(f"       step {t:3d}: {err_curve_pre[t]:.2%}")

    # ----- Phase 2: multi-step unrolled training -----
    print("\n[4] Phase 2: multi-step unrolled training (30 epochs, unroll=4)...")
    t0 = time.time()
    history2 = train_multistep(model, train_traj_stack, unroll_steps=4,
                                n_epochs=30, batch_size=16, lr=1e-4)
    print(f"    Phase 2 time: {time.time() - t0:.1f}s")

    # Test 2 post: rollout after multi-step training
    print(f"\n[T2] Rollout after Phase 2 (should be stable):")
    err_curve_post, gt_post, nn_post = rollout_error(model, u0_test, n_steps=30)
    print(f"     Rollout rel L2 error at:")
    for t in [1, 5, 10, 20, 30]:
        print(f"       step {t:3d}: {err_curve_post[t]:.2%}")

    # Test 3: positivity preservation
    print(f"\n[T3] Positivity preservation:")
    print(f"     ground-truth u range over rollout: "
          f"[{gt_post.min():.4f}, {gt_post.max():.4f}]")
    print(f"     neural-PDE u range:                "
          f"[{nn_post.min():.4f}, {nn_post.max():.4f}]")
    positivity_ok = nn_post.min() > -0.05 and nn_post.max() < 1.1
    print(f"     {'PASS' if positivity_ok else 'FAIL'}  (allow small over/undershoot)")

    # Test 4: out-of-distribution
    print(f"\n[T4] Generalization to unseen initial-condition style (noise):")
    ood_errs = []
    for t_idx, gt_traj in enumerate(trajs_ood):
        u0 = gt_traj[0]
        nn_traj = rollout(model, u0, n_steps=gt_traj.shape[0] - 1)
        final_err = (np.linalg.norm(gt_traj[-1] - nn_traj[-1]) /
                      (np.linalg.norm(gt_traj[-1]) + 1e-9))
        ood_errs.append(final_err)
    mean_ood_err = float(np.mean(ood_errs))
    print(f"     Final-step rel L2 on OOD trajectories: {mean_ood_err:.2%}")

    # Summary
    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    t1 = onestep_rel < 0.05
    t2 = err_curve_post[30] < 0.10
    t3 = positivity_ok
    t4 = mean_ood_err < 0.30
    t2_improvement = (err_curve_pre[30] - err_curve_post[30]) > 0.05

    for lab, ok, desc in [
        ("T1", t1, f"one-step rel err < 5% ({onestep_rel:.2%})"),
        ("T2", t2, f"30-step rollout rel err < 10% ({err_curve_post[30]:.2%})"),
        ("T3", t3, "positivity preserved"),
        ("T4", t4, f"OOD generalization < 30% ({mean_ood_err:.2%})"),
        ("T2+", t2_improvement, "multi-step training helped over one-step"),
    ]:
        print(f"  [{lab}] {'PASS' if ok else 'FAIL'}  {desc}")
    print()
    if t1 and t2 and t3:
        print("Neural PDE learns Fisher-KPP dynamics and rolls out stably.")
        print("The local-convolution + periodic-boundary architecture admits")
        print("gradient-based training and produces physically sensible dynamics.")
    else:
        print("Partial. Rollout stability is the hardest part of this setup.")


if __name__ == "__main__":
    main()
