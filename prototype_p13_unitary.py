"""
DMVC P13: Complex psi with structurally-unitary evolution.

Claim tested: on a quantum-like PDE (Schrödinger equation), a network that
structurally enforces unitary time evolution should beat a vanilla neural-PDE
baseline on long rollouts, primarily because unitarity ==> probability is
conserved exactly rather than drifting under compounding numerical error.

Target system: 2D Schrödinger equation in a static potential

    i d psi/dt = -0.5 * nabla^2 psi + V(x) psi

with psi a complex wavefunction, V(x) a fixed potential. Observables we'll
measure:
    - L^2 norm  |psi|^2 integrated over space  (should be conserved == 1)
    - Rollout accuracy of psi vs ground truth
    - Expected energy <H>  (should be conserved if evolution is unitary)

Ground truth computed with split-step Fourier method (symplectic, unitary to
machine precision).

Architectures:

  Model A -- "vanilla baseline":
      NetPsi: psi -> d psi/dt (complex, 2 real channels)
      Forward Euler: psi_{t+1} = psi_t + dt * d psi / dt
      No constraint. Norm will drift under training error.

  Model B -- "structurally unitary":
      NetH: psi -> H(x) -- a per-site *real* scalar we interpret as a local
            Hermitian diagonal. The network predicts the diagonal of H(x)
            at each point; the off-diagonal (kinetic) part is a fixed
            real-symmetric (hence Hermitian) finite-difference Laplacian.
      The update is:
          psi_{t+1} = exp(-i H_total * dt) psi_t
      where H_total = -0.5 * fixed_Laplacian + diag(H_learned).
      exp(-iHdt) is structurally unitary because H is Hermitian by
      construction.

  Model C -- "baseline with renormalization":
      Same as Model A, but psi is re-normalized to unit L^2 after each step.
      This is a weaker constraint than unitarity but often used in practice.

Tests:
  T1: Unitary model preserves ||psi||^2 to machine precision over rollout
  T2: Baseline model's ||psi||^2 drifts meaningfully (at least 1% in 30 steps)
  T3: Unitary model matches ground-truth psi better on long rollout
      than baseline
  T4: Energy <H_true> is better conserved by unitary model
  T5: Renormalization baseline is between the two (if it's close to
      unitary, norm is the only thing unitary structure buys us;
      if it's close to bare baseline, we buy more than just norm)

Honest pre-registration:
  * T1 should be exact. If it fails we have a bug.
  * T2 should hold because the conv network has no global conservation.
    If it doesn't, the baseline accidentally learned local unitarity.
  * T3 is the real test. Even with renormalization, systematic errors in
    the direction of psi updates will accumulate. Structural unitarity
    constrains that direction.
  * T4 is the hardest. Energy conservation requires the Hamiltonian to
    commute with itself at different times (i.e., time-independent H),
    which our learned H is. Ground truth also has time-independent H.
    So we expect energy conservation for both unitary and renormalized.
    If unitary wins on this too, that's strong evidence for the approach.
"""

from __future__ import annotations
import sys, time
sys.path.insert(0, "/home/claude/dmvc")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Ground truth: 2D Schrödinger via split-step Fourier
# =============================================================================
GRID_N = 32
L_DOMAIN = 8.0          # physical size of the box
DX = L_DOMAIN / GRID_N
DT = 0.05


def make_potential(N, kind="harmonic", strength=1.0):
    """Return V(x) on an N x N grid."""
    x = np.linspace(-L_DOMAIN/2, L_DOMAIN/2, N, endpoint=False, dtype=np.float32)
    y = np.linspace(-L_DOMAIN/2, L_DOMAIN/2, N, endpoint=False, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    if kind == "harmonic":
        V = 0.5 * strength * (xx**2 + yy**2)
    elif kind == "double_well":
        V = strength * ((xx**2 - 1.5)**2 + 0.5 * yy**2) / 10.0
    elif kind == "flat":
        V = np.zeros_like(xx)
    elif kind == "barrier":
        V = np.where(np.abs(xx) < 0.5, strength, 0.0).astype(np.float32)
    else:
        raise ValueError(kind)
    return V.astype(np.float32)


def initial_wavepacket(N, rng, px_range=(-1.5, 1.5)):
    """Gaussian wavepacket with small momentum, centered at random."""
    x = np.linspace(-L_DOMAIN/2, L_DOMAIN/2, N, endpoint=False)
    y = np.linspace(-L_DOMAIN/2, L_DOMAIN/2, N, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    x0 = rng.uniform(-1.5, 1.5)
    y0 = rng.uniform(-1.5, 1.5)
    sigma = rng.uniform(0.6, 1.2)
    px = rng.uniform(*px_range)
    py = rng.uniform(*px_range)
    psi = np.exp(-((xx-x0)**2 + (yy-y0)**2) / (2*sigma**2)) * \
           np.exp(1j * (px*xx + py*yy))
    # normalize
    psi = psi / np.sqrt(np.sum(np.abs(psi)**2))
    return psi.astype(np.complex64)


def split_step_evolve(psi0, V, n_steps, dt=DT):
    """
    Split-step Fourier: exp(-iVdt/2) FFT exp(-iTdt) IFFT exp(-iVdt/2)
    T is the kinetic operator, diagonal in Fourier space.
    Unitary to machine precision.
    """
    N = psi0.shape[0]
    k = np.fft.fftfreq(N, d=DX) * 2 * np.pi
    kx, ky = np.meshgrid(k, k, indexing='ij')
    T_k = 0.5 * (kx**2 + ky**2)              # kinetic eigenvalue
    phase_V = np.exp(-1j * V * dt / 2.0)
    phase_T = np.exp(-1j * T_k * dt)

    psi = psi0.astype(np.complex128).copy()
    traj = np.zeros((n_steps + 1,) + psi0.shape, dtype=np.complex64)
    traj[0] = psi0
    for t in range(n_steps):
        psi = psi * phase_V
        psi_k = np.fft.fft2(psi)
        psi_k = psi_k * phase_T
        psi = np.fft.ifft2(psi_k)
        psi = psi * phase_V
        traj[t + 1] = psi.astype(np.complex64)
    return traj


def compute_energy(psi, V):
    """<H> = <T> + <V> on a periodic grid."""
    N = psi.shape[0]
    k = np.fft.fftfreq(N, d=DX) * 2 * np.pi
    kx, ky = np.meshgrid(k, k, indexing='ij')
    psi_k = np.fft.fft2(psi)
    T_k = 0.5 * (kx**2 + ky**2)
    T_val = np.sum(np.abs(psi_k)**2 * T_k) / (N * N)
    V_val = np.sum(np.abs(psi)**2 * V)
    # normalize by norm to get expectation per probability
    norm = np.sum(np.abs(psi)**2) + 1e-12
    return float((T_val + V_val) / norm)


# =============================================================================
# Shared utilities
# =============================================================================
def circ_pad(x, n=1):
    return F.pad(x, (n, n, n, n), mode='circular')


# A fixed discrete Laplacian kernel, applied as depthwise conv
# This is the "kinetic operator" we'll use in the unitary model. It's a
# real symmetric operator (Hermitian) so exp(-iT dt) applied to it is unitary.
# In practice we use split-step for the kinetic part too: FFT-based for accuracy.
def laplacian_conv(u_ri):
    """5-point Laplacian on (B, 2, H, W) via circular padding. Returns (B, 2, H, W)."""
    padded = circ_pad(u_ri)
    lap = (padded[:, :, 2:, 1:-1] + padded[:, :, :-2, 1:-1] +
            padded[:, :, 1:-1, 2:] + padded[:, :, 1:-1, :-2] -
            4.0 * padded[:, :, 1:-1, 1:-1]) / (DX * DX)
    return lap


# =============================================================================
# Model A: vanilla baseline (non-unitary)
# =============================================================================
class NetVanilla(nn.Module):
    """Predicts d psi / dt from psi (2 channels: real, imag)."""
    def __init__(self, hidden=32):
        super().__init__()
        self.conv1 = nn.Conv2d(2, hidden, 3, padding=0)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.conv_out = nn.Conv2d(hidden, 2, 1, padding=0)

    def forward(self, psi_ri):
        x = F.gelu(self.conv1(circ_pad(psi_ri)))
        x = F.gelu(self.conv2(circ_pad(x)))
        x = F.gelu(self.conv3(circ_pad(x)))
        return self.conv_out(x)


# =============================================================================
# Model B: structurally unitary via learned diagonal Hamiltonian
# =============================================================================
class NetUnitary(nn.Module):
    """
    Predicts a REAL scalar V_eff(x) (the diagonal of a learned local
    Hamiltonian). We then evolve via:
        H = T_kinetic (fixed FFT-based) + V_eff
        psi_{t+1} = exp(-i H dt) psi
    which is structurally unitary because H is Hermitian by construction
    (T is Hermitian, diag(V_eff) with real V_eff is Hermitian, sum is Hermitian).

    Implementation: we use split-step:
        psi' = exp(-i V_eff dt / 2) psi
        psi' = FFT; multiply by exp(-i T(k) dt); IFFT
        psi' = exp(-i V_eff dt / 2) psi'
    The first and third steps are pointwise complex multiplications preserving
    |psi|. The FFT step is unitary. So the whole forward pass is unitary
    regardless of what V_eff the network produces.
    """
    def __init__(self, hidden=32, N=GRID_N):
        super().__init__()
        self.conv1 = nn.Conv2d(2, hidden, 3, padding=0)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.conv_out = nn.Conv2d(hidden, 1, 1, padding=0)   # scalar potential

        # Precompute kinetic phase (diagonal in Fourier space)
        k = np.fft.fftfreq(N, d=DX) * 2 * np.pi
        kx, ky = np.meshgrid(k, k, indexing='ij')
        T_k = 0.5 * (kx**2 + ky**2)
        self.register_buffer("T_k", torch.tensor(T_k, dtype=torch.float32))

    def predict_V(self, psi_ri):
        """Run the conv stack to produce V_eff(x)."""
        x = F.gelu(self.conv1(circ_pad(psi_ri)))
        x = F.gelu(self.conv2(circ_pad(x)))
        x = F.gelu(self.conv3(circ_pad(x)))
        V = self.conv_out(x)     # (B, 1, H, W)
        return V.squeeze(1)     # (B, H, W)

    def forward(self, psi_ri, dt):
        """
        Apply one unitary time step.
        psi_ri: (B, 2, H, W) -- real/imag channels
        dt: scalar float
        returns: (B, 2, H, W) updated psi_ri
        """
        V_eff = self.predict_V(psi_ri)    # (B, H, W), real

        # Build psi as a complex tensor
        re = psi_ri[:, 0]; im = psi_ri[:, 1]   # (B, H, W)
        psi_c = torch.complex(re, im)          # (B, H, W)

        # Step 1: half-step potential
        phase_V = torch.exp(torch.complex(torch.zeros_like(V_eff), -V_eff * dt / 2.0))
        psi_c = psi_c * phase_V

        # Step 2: full-step kinetic in Fourier
        psi_k = torch.fft.fft2(psi_c)
        phase_T = torch.exp(torch.complex(torch.zeros_like(self.T_k),
                                           -self.T_k * dt))
        psi_k = psi_k * phase_T
        psi_c = torch.fft.ifft2(psi_k)

        # Step 3: second half-step potential
        psi_c = psi_c * phase_V

        # Back to real/imag channels
        out = torch.stack([psi_c.real, psi_c.imag], dim=1)
        return out


# =============================================================================
# Model C: vanilla baseline + per-step L2 renormalization
# =============================================================================
def renormalize(psi_ri):
    """Rescale so |psi|^2 sums to 1 per sample."""
    norm_sq = (psi_ri ** 2).sum(dim=(1, 2, 3), keepdim=True)
    return psi_ri / torch.sqrt(norm_sq + 1e-12)


# =============================================================================
# Training
# =============================================================================
def train_vanilla(model, Xt, Xtp1, Xv, Xvp1, dt=DT, n_epochs=40,
                    batch_size=32, lr=3e-4, device='cpu'):
    model = model.to(device)
    Xt_t = torch.tensor(Xt).to(device); Xtp1_t = torch.tensor(Xtp1).to(device)
    Xv_t = torch.tensor(Xv).to(device); Xvp1_t = torch.tensor(Xvp1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    n = len(Xt_t)
    for ep in range(n_epochs):
        perm = torch.randperm(n)
        losses = []
        model.train()
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            u = Xt_t[idx]
            target = Xtp1_t[idx]
            du = model(u)
            pred = u + dt * du
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        tl = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            du = model(Xv_t); pred = Xv_t + dt * du
            vl = ((pred - Xvp1_t) ** 2).mean().item()
        hist.append((tl, vl))
        if ep % 10 == 0 or ep == n_epochs - 1:
            print(f"  epoch {ep:3d}  train={tl:.4e}  val={vl:.4e}")
    return hist


def train_unitary(model, Xt, Xtp1, Xv, Xvp1, dt=DT, n_epochs=40,
                    batch_size=32, lr=3e-4, device='cpu'):
    model = model.to(device)
    Xt_t = torch.tensor(Xt).to(device); Xtp1_t = torch.tensor(Xtp1).to(device)
    Xv_t = torch.tensor(Xv).to(device); Xvp1_t = torch.tensor(Xvp1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    n = len(Xt_t)
    for ep in range(n_epochs):
        perm = torch.randperm(n)
        losses = []
        model.train()
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            u = Xt_t[idx]
            target = Xtp1_t[idx]
            pred = model(u, dt)
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        tl = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            pred = model(Xv_t, dt)
            vl = ((pred - Xvp1_t) ** 2).mean().item()
        hist.append((tl, vl))
        if ep % 10 == 0 or ep == n_epochs - 1:
            print(f"  epoch {ep:3d}  train={tl:.4e}  val={vl:.4e}")
    return hist


# =============================================================================
# Rollouts
# =============================================================================
def rollout_vanilla(model, psi0, n_steps, dt=DT, renormalize_each=False):
    model.eval()
    u = torch.tensor(np.stack([psi0.real, psi0.imag], axis=0)[None],
                      dtype=torch.float32)
    out = np.zeros((n_steps + 1,) + psi0.shape, dtype=np.complex64)
    out[0] = psi0
    with torch.no_grad():
        for t in range(n_steps):
            du = model(u)
            u = u + dt * du
            if renormalize_each:
                u = renormalize(u)
            arr = u.squeeze(0).cpu().numpy()
            out[t + 1] = arr[0] + 1j * arr[1]
    return out


def rollout_unitary(model, psi0, n_steps, dt=DT):
    model.eval()
    u = torch.tensor(np.stack([psi0.real, psi0.imag], axis=0)[None],
                      dtype=torch.float32)
    out = np.zeros((n_steps + 1,) + psi0.shape, dtype=np.complex64)
    out[0] = psi0
    with torch.no_grad():
        for t in range(n_steps):
            u = model(u, dt)
            arr = u.squeeze(0).cpu().numpy()
            out[t + 1] = arr[0] + 1j * arr[1]
    return out


def norm_drift(traj):
    """Relative drift of L2 norm from t=0 to each t."""
    n0 = np.sum(np.abs(traj[0])**2)
    nt = np.array([np.sum(np.abs(traj[t])**2) for t in range(len(traj))])
    return nt / n0 - 1.0


def traj_error(pred, gt):
    """Rel L2 of predicted vs ground-truth trajectory at each time."""
    err = np.zeros(len(pred))
    for t in range(len(pred)):
        num = np.linalg.norm(pred[t] - gt[t])
        den = np.linalg.norm(gt[t]) + 1e-12
        err[t] = num / den
    return err


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 76)
    print("DMVC P13: Structurally unitary evolution vs vanilla baseline")
    print("=" * 76)

    rng = np.random.default_rng(0)

    # ----- Build training data -----
    # Use a single potential (harmonic) for training -- the network must
    # learn to represent it. For OOD we can try a different potential.
    V_train = make_potential(GRID_N, kind="harmonic", strength=0.25)
    V_ood = make_potential(GRID_N, kind="double_well", strength=1.0)

    print("\n[1] Generating ground-truth Schrödinger trajectories...")
    t0 = time.time()
    trajs_train = []
    for i in range(20):
        psi0 = initial_wavepacket(GRID_N, rng)
        traj = split_step_evolve(psi0, V_train, n_steps=30)
        trajs_train.append(traj)
    trajs_val = []
    for i in range(5):
        psi0 = initial_wavepacket(GRID_N, rng)
        traj = split_step_evolve(psi0, V_train, n_steps=30)
        trajs_val.append(traj)
    trajs_ood = []
    for i in range(3):
        psi0 = initial_wavepacket(GRID_N, rng)
        traj = split_step_evolve(psi0, V_ood, n_steps=30)
        trajs_ood.append(traj)
    print(f"    train: {len(trajs_train)}, val: {len(trajs_val)}, "
          f"ood: {len(trajs_ood)}")
    print(f"    simulation time: {time.time()-t0:.1f}s")

    # ground-truth norm conservation sanity check (should be basically 1.0)
    nd = norm_drift(trajs_val[0])
    print(f"    Ground-truth norm drift at step 30: {nd[-1]:+.2e} "
          "(should be ~0)")

    # Build (psi_t, psi_tp1) pairs
    def to_frames(trajs):
        X, Y = [], []
        for traj in trajs:
            for t in range(len(traj) - 1):
                X.append(np.stack([traj[t].real, traj[t].imag], axis=0))
                Y.append(np.stack([traj[t+1].real, traj[t+1].imag], axis=0))
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
    Xt, Xtp1 = to_frames(trajs_train)
    Xv, Xvp1 = to_frames(trajs_val)
    print(f"    training frames: {len(Xt)}")

    # ----- Train Model A (vanilla baseline) -----
    print("\n[2] Training Model A (vanilla baseline)...")
    mA = NetVanilla(hidden=32)
    nA = sum(p.numel() for p in mA.parameters())
    print(f"    Parameters: {nA}")
    train_vanilla(mA, Xt, Xtp1, Xv, Xvp1, n_epochs=40, batch_size=32, lr=3e-4)

    # ----- Train Model B (unitary) -----
    print("\n[3] Training Model B (structurally unitary)...")
    mB = NetUnitary(hidden=32, N=GRID_N)
    nB = sum(p.numel() for p in mB.parameters())
    print(f"    Parameters: {nB}")
    train_unitary(mB, Xt, Xtp1, Xv, Xvp1, n_epochs=40, batch_size=32, lr=3e-4)

    # ----- Rollouts on val (in-distribution) -----
    print("\n[4] Rollout comparison on val trajectories (harmonic potential)...")
    drifts_A, drifts_B, drifts_C = [], [], []
    errs_A, errs_B, errs_C = [], [], []
    energies_true, energies_A, energies_B, energies_C = [], [], [], []
    for traj in trajs_val:
        psi0 = traj[0]
        n_steps = len(traj) - 1
        rA = rollout_vanilla(mA, psi0, n_steps)
        rB = rollout_unitary(mB, psi0, n_steps)
        rC = rollout_vanilla(mA, psi0, n_steps, renormalize_each=True)

        drifts_A.append(norm_drift(rA))
        drifts_B.append(norm_drift(rB))
        drifts_C.append(norm_drift(rC))

        errs_A.append(traj_error(rA, traj))
        errs_B.append(traj_error(rB, traj))
        errs_C.append(traj_error(rC, traj))

        # Energy with the TRUE potential (same during training)
        e_true = [compute_energy(traj[t], V_train) for t in range(n_steps+1)]
        e_A = [compute_energy(rA[t], V_train) for t in range(n_steps+1)]
        e_B = [compute_energy(rB[t], V_train) for t in range(n_steps+1)]
        e_C = [compute_energy(rC[t], V_train) for t in range(n_steps+1)]
        energies_true.append(e_true); energies_A.append(e_A)
        energies_B.append(e_B);       energies_C.append(e_C)

    drifts_A = np.mean(np.array(drifts_A), axis=0)
    drifts_B = np.mean(np.array(drifts_B), axis=0)
    drifts_C = np.mean(np.array(drifts_C), axis=0)
    errs_A = np.mean(np.array(errs_A), axis=0)
    errs_B = np.mean(np.array(errs_B), axis=0)
    errs_C = np.mean(np.array(errs_C), axis=0)

    print(f"\n    Norm drift (relative, averaged over val trajectories):")
    print(f"    {'step':>4s} {'vanilla':>12s} {'unitary':>12s} {'renorm':>12s}")
    for t in [1, 5, 10, 20, 30]:
        print(f"    {t:>4d} {drifts_A[t]:>+12.3e} {drifts_B[t]:>+12.3e} "
              f"{drifts_C[t]:>+12.3e}")

    print(f"\n    Trajectory error vs ground truth:")
    print(f"    {'step':>4s} {'vanilla':>10s} {'unitary':>10s} {'renorm':>10s}")
    for t in [1, 5, 10, 20, 30]:
        print(f"    {t:>4d} {errs_A[t]:>10.2%} {errs_B[t]:>10.2%} "
              f"{errs_C[t]:>10.2%}")

    # Energy drift (averaged)
    energies_true = np.array(energies_true)
    energies_A = np.array(energies_A)
    energies_B = np.array(energies_B)
    energies_C = np.array(energies_C)
    e0 = energies_true[:, 0:1]
    dE_A = np.mean(np.abs(energies_A - e0) / np.abs(e0), axis=0)
    dE_B = np.mean(np.abs(energies_B - e0) / np.abs(e0), axis=0)
    dE_C = np.mean(np.abs(energies_C - e0) / np.abs(e0), axis=0)
    print(f"\n    Energy drift (rel, vs initial true energy):")
    print(f"    {'step':>4s} {'vanilla':>10s} {'unitary':>10s} {'renorm':>10s}")
    for t in [1, 5, 10, 20, 30]:
        print(f"    {t:>4d} {dE_A[t]:>10.2%} {dE_B[t]:>10.2%} "
              f"{dE_C[t]:>10.2%}")

    # ----- Tests -----
    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)

    t1 = abs(drifts_B[-1]) < 1e-4   # unitary preserves norm structurally
    t2 = abs(drifts_A[-1]) > 0.01   # baseline drifts meaningfully
    t3 = errs_B[-1] < errs_A[-1]    # unitary wins trajectory accuracy
    t4 = dE_B[-1] < dE_A[-1]        # unitary conserves energy better
    t5_renorm = errs_C[-1]
    t5_unitary = errs_B[-1]
    # T5: renormalization helps but unitary helps more? Expect C between A and B.
    t5 = (errs_B[-1] < errs_C[-1] < errs_A[-1]) or (errs_C[-1] < errs_A[-1] * 0.9)

    print(f"\n[T1] Unitary preserves L^2 norm (drift at step 30): "
          f"{drifts_B[-1]:+.2e}  {'PASS' if t1 else 'FAIL'}")
    print(f"[T2] Baseline L^2 drifts meaningfully (>1%): "
          f"{drifts_A[-1]:+.2%}  {'PASS' if t2 else 'FAIL'}")
    print(f"[T3] Unitary better rollout accuracy: "
          f"{errs_B[-1]:.2%} vs baseline {errs_A[-1]:.2%}  "
          f"{'PASS' if t3 else 'FAIL'}")
    print(f"[T4] Unitary better energy conservation: "
          f"dE_unit {dE_B[-1]:.2%} vs dE_base {dE_A[-1]:.2%}  "
          f"{'PASS' if t4 else 'FAIL'}")
    print(f"[T5] Renormalization alone partial ({t5_renorm:.2%}) vs "
          f"unitary ({t5_unitary:.2%})  {'PASS' if t5 else 'FAIL'}")

    print()
    print("Interpretation:")
    print(f"  Baseline parameters: {nA};  Unitary parameters: {nB}")
    print("  T1 pure sanity: unitary structure works as designed")
    print("  T2: baseline provably non-unitary -- test is discriminative")
    print("  T3: the real claim: unitarity helps rollout accuracy")
    print("  T4: stronger claim: energy is conserved when evolution is unitary")
    print("  T5: is it just the norm, or does unitarity do more?")


if __name__ == "__main__":
    main()
