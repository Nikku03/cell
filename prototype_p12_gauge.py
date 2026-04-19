"""
DMVC P12: Learned connection field on a gauged complex scalar PDE.

Target system: Abelian-Higgs-lite / gauged complex Ginzburg-Landau

    dpsi/dt = D * (nabla + i*A)^2 psi + r * psi * (1 - |psi|^2)

where:
    psi(x, t)  -- complex scalar field on a 2D periodic grid
    A(x)       -- real vector field (the gauge connection), FIXED in time
    (nabla + i*A)^2 -- covariant Laplacian, defined below

The ground-truth A(x) is hand-designed and held constant through each
trajectory. Training data is trajectories of psi under this dynamics. The
network must learn the dynamics WITHOUT being told what A(x) is.

Architecture test:
    Model A -- "P11 baseline": single conv network predicts dpsi/dt from psi
               alone. No auxiliary field channel.
    Model B -- "P12 gauge": two conv networks.
        NetA: psi -> A_learned (a learned connection field output)
        NetPsi: (psi, A_learned) -> dpsi/dt
        Trained jointly on rollout loss.

Hypothesis: if the target dynamics have genuine gauge structure that cannot
be represented by a purely-psi-local convolution, Model B should beat Model A.
If they tie, the system doesn't actually need the gauge channel and the
architectural claim is weak.

Tests:
    T1: Model A (baseline) rollout error on ground truth with A != 0
    T2: Model B (gauge) rollout error on same ground truth
    T3: Model B beats Model A by a meaningful margin
    T4: Learned A_learned field has nontrivial spatial structure
        (if it collapses to near-zero, the gauge channel was unused)
    T5: Model B trained on one A(x) pattern generalizes to another
        -- the real structural claim.

Honest pre-registration:
    * If A_gt is too weak, both models perform well and T3 fails.
    * If A_gt is too strong, training becomes unstable for both.
    * T5 is the most interesting test. If it passes, the network really
      did learn to infer gauge structure from dynamics.
"""

from __future__ import annotations
import sys, time
sys.path.insert(0, "/home/claude/dmvc")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Ground truth: gauged complex Ginzburg-Landau
# =============================================================================
D_DIFF = 0.08
R_REACT = 1.0
GRID_N = 32
DT = 0.01


def periodic_shift(arr, shift_x, shift_y):
    return np.roll(arr, shift_x, axis=0) if shift_y == 0 else \
            np.roll(np.roll(arr, shift_x, axis=0), shift_y, axis=1)


def covariant_laplacian(psi, Ax, Ay, dx=1.0):
    """
    Discrete covariant Laplacian:
        ((nabla + iA)^2 psi)(x) approximated via the "link variable"
        prescription (standard lattice gauge theory):
            U_x = exp(i * Ax * dx)  between (i,j) and (i+1,j)
            U_y = exp(i * Ay * dx)  between (i,j) and (i,j+1)
        Then covariant Laplacian =
            sum over neighbors [ U_link * psi(neighbor) ] - 4 * psi(x)
    This is gauge-invariant by construction.

    psi: complex, shape (N, N)
    Ax, Ay: real, shape (N, N), values at each site give the link to the
            +x and +y neighbors respectively.
    """
    # U_x(i,j) couples (i,j) -> (i+1,j)
    # When we look at site (i,j), we need:
    #   forward x-neighbor: U_x(i,j) * psi(i+1,j)
    #   backward x-neighbor: conj(U_x(i-1,j)) * psi(i-1,j)  (reverse link = conj)
    # Similarly for y.
    Ux = np.exp(1j * Ax * dx)
    Uy = np.exp(1j * Ay * dx)

    psi_xp = np.roll(psi, -1, axis=0)          # psi(i+1, j)
    psi_xm = np.roll(psi, 1, axis=0)           # psi(i-1, j)
    psi_yp = np.roll(psi, -1, axis=1)
    psi_ym = np.roll(psi, 1, axis=1)

    Ux_m = np.roll(Ux, 1, axis=0)              # U_x at site (i-1,j)
    Uy_m = np.roll(Uy, 1, axis=1)

    lap = (Ux * psi_xp + np.conj(Ux_m) * psi_xm +
            Uy * psi_yp + np.conj(Uy_m) * psi_ym - 4.0 * psi)
    return lap / (dx * dx)


def gauged_gl_step(psi, Ax, Ay, dt, D=D_DIFF, r=R_REACT):
    """One explicit-Euler step of the gauged complex Ginzburg-Landau eqn."""
    lap = covariant_laplacian(psi, Ax, Ay)
    dpsi = D * lap + r * psi * (1.0 - np.abs(psi) ** 2)
    return psi + dt * dpsi


def simulate_gauged_gl(psi0, Ax, Ay, n_steps, dt=DT, sub=5):
    """Record trajectory every `sub`-th internal step."""
    psi = psi0.astype(np.complex128).copy()
    out = np.zeros((n_steps + 1,) + psi0.shape, dtype=np.complex64)
    out[0] = psi0.astype(np.complex64)
    sub_dt = dt / sub
    for t in range(n_steps):
        for _ in range(sub):
            psi = gauged_gl_step(psi, Ax, Ay, sub_dt)
        out[t + 1] = psi.astype(np.complex64)
    return out


def make_gauge_field(N, style="swirl", amplitude=0.5, rng=None):
    """
    Construct a gauge field A(x) = (Ax, Ay) with the specified structure.
    amplitude controls how strong the gauge field is (relative to the
    kinetic term). The "swirl" style creates a divergence-free rotation-like
    field.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    xc = N / 2.0
    yc = N / 2.0
    if style == "swirl":
        # Vortex-like: A_x = -(y - yc)/R, A_y = (x - xc)/R. Scaled by amplitude.
        R = N / 4.0
        rx = (xx - xc) / R
        ry = (yy - yc) / R
        rmag = np.sqrt(rx**2 + ry**2) + 1e-9
        # falloff at the edges so it's smooth
        decay = np.exp(-rmag * 0.5)
        Ax = -ry * decay * amplitude
        Ay = rx * decay * amplitude
    elif style == "uniform":
        Ax = np.full((N, N), amplitude)
        Ay = np.zeros((N, N))
    elif style == "sinusoidal":
        k = 2 * np.pi / N
        Ax = amplitude * np.cos(k * yy)
        Ay = amplitude * np.sin(k * xx)
    elif style == "zero":
        Ax = np.zeros((N, N))
        Ay = np.zeros((N, N))
    else:
        raise ValueError(style)
    return Ax.astype(np.float32), Ay.astype(np.float32)


def random_initial_psi(N, rng):
    """Small complex perturbation around a uniform background."""
    amp_real = rng.uniform(0.3, 0.6)
    amp_imag = rng.uniform(-0.1, 0.1)
    noise_scale = 0.05
    re = amp_real + noise_scale * rng.normal(size=(N, N))
    im = amp_imag + noise_scale * rng.normal(size=(N, N))
    # smooth it out a bit
    from scipy.ndimage import gaussian_filter
    re = gaussian_filter(re, sigma=1.0)
    im = gaussian_filter(im, sigma=1.0)
    return (re + 1j * im).astype(np.complex64)


# =============================================================================
# Networks
# =============================================================================
def circ_pad(x, n=1):
    return F.pad(x, (n, n, n, n), mode='circular')


class NetPsiOnly(nn.Module):
    """
    Model A: P11-style baseline. Predicts dpsi/dt from psi alone.
    psi is represented as 2 channels (real, imag).
    """
    def __init__(self, hidden=32):
        super().__init__()
        in_ch = 2
        self.conv1 = nn.Conv2d(in_ch, hidden, 3, padding=0)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.conv_out = nn.Conv2d(hidden, 2, 1, padding=0)

    def forward(self, psi_ri):
        # psi_ri: (B, 2, H, W) -- channels are (real, imag)
        x = F.gelu(self.conv1(circ_pad(psi_ri)))
        x = F.gelu(self.conv2(circ_pad(x)))
        x = F.gelu(self.conv3(circ_pad(x)))
        return self.conv_out(x)


class NetPsiGauge(nn.Module):
    """
    Model B: psi dynamics network that consumes an additional gauge channel.
    Input: 4 channels (real(psi), imag(psi), A_learned_x, A_learned_y)
    Output: 2 channels (d real(psi)/dt, d imag(psi)/dt)
    """
    def __init__(self, hidden=32):
        super().__init__()
        in_ch = 4
        self.conv1 = nn.Conv2d(in_ch, hidden, 3, padding=0)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.conv_out = nn.Conv2d(hidden, 2, 1, padding=0)

    def forward(self, psi_ri, A_field):
        x_in = torch.cat([psi_ri, A_field], dim=1)
        x = F.gelu(self.conv1(circ_pad(x_in)))
        x = F.gelu(self.conv2(circ_pad(x)))
        x = F.gelu(self.conv3(circ_pad(x)))
        return self.conv_out(x)


class NetA(nn.Module):
    """
    Learns a 2-channel gauge field A(x) from the current psi.
    This is the 'gauge-like connection field' in Dark Manifold terms.
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.conv1 = nn.Conv2d(2, hidden, 3, padding=0)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.conv_out = nn.Conv2d(hidden, 2, 3, padding=0)

    def forward(self, psi_ri):
        x = F.gelu(self.conv1(circ_pad(psi_ri)))
        x = F.gelu(self.conv2(circ_pad(x)))
        return self.conv_out(circ_pad(x))


# =============================================================================
# Training data: (psi_t, psi_{t+1}) pairs from many trajectories
# =============================================================================
def build_dataset(n_trajectories, n_steps_per, A_style_list, amplitude,
                   seed=0):
    """
    Generate trajectories with gauge fields drawn from A_style_list, varying
    amplitude. Returns list of (traj, Ax, Ay) tuples.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_trajectories):
        style = A_style_list[i % len(A_style_list)]
        amp = amplitude * rng.uniform(0.7, 1.3)
        Ax, Ay = make_gauge_field(GRID_N, style=style, amplitude=amp, rng=rng)
        psi0 = random_initial_psi(GRID_N, rng)
        traj = simulate_gauged_gl(psi0, Ax, Ay, n_steps_per)
        out.append((traj, Ax, Ay))
    return out


def trajs_to_riframes(trajs):
    """Convert list-of-(traj, Ax, Ay) into arrays of (B, 2, H, W) real/imag frames.
    Drops Ax/Ay since the models must infer gauge from data. Returns
    (X_t, X_tp1, traj_id_per_sample).
    """
    X_t, X_tp1, tid = [], [], []
    for i, (traj, Ax, Ay) in enumerate(trajs):
        for t in range(len(traj) - 1):
            psi = traj[t]
            psi_next = traj[t + 1]
            X_t.append(np.stack([psi.real, psi.imag], axis=0))
            X_tp1.append(np.stack([psi_next.real, psi_next.imag], axis=0))
            tid.append(i)
    return (np.array(X_t, dtype=np.float32),
             np.array(X_tp1, dtype=np.float32),
             np.array(tid, dtype=np.int64))


# =============================================================================
# Training loops
# =============================================================================
def train_baseline(model, Xt, Xtp1, Xv, Xvp1, dt=DT, n_epochs=40,
                    batch_size=32, lr=3e-4, device='cpu'):
    model = model.to(device)
    Xt = torch.tensor(Xt).to(device); Xtp1 = torch.tensor(Xtp1).to(device)
    Xv = torch.tensor(Xv).to(device); Xvp1 = torch.tensor(Xvp1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    n = len(Xt)
    for ep in range(n_epochs):
        perm = torch.randperm(n)
        losses = []
        model.train()
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            u = Xt[idx]
            target = Xtp1[idx]
            du = model(u)
            pred = u + dt * du
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        tl = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            du = model(Xv)
            pred = Xv + dt * du
            vl = ((pred - Xvp1) ** 2).mean().item()
        hist.append((tl, vl))
        if ep % 10 == 0 or ep == n_epochs - 1:
            print(f"  epoch {ep:3d}  train={tl:.4e}  val={vl:.4e}")
    return hist


def train_gauge(netPsi, netA, Xt, Xtp1, Xv, Xvp1, dt=DT, n_epochs=40,
                  batch_size=32, lr=3e-4, device='cpu'):
    netPsi = netPsi.to(device); netA = netA.to(device)
    Xt = torch.tensor(Xt).to(device); Xtp1 = torch.tensor(Xtp1).to(device)
    Xv = torch.tensor(Xv).to(device); Xvp1 = torch.tensor(Xvp1).to(device)
    params = list(netPsi.parameters()) + list(netA.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    hist = []
    n = len(Xt)
    for ep in range(n_epochs):
        perm = torch.randperm(n)
        losses = []
        netPsi.train(); netA.train()
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            u = Xt[idx]
            target = Xtp1[idx]
            A_learned = netA(u)
            du = netPsi(u, A_learned)
            pred = u + dt * du
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        tl = float(np.mean(losses))
        netPsi.eval(); netA.eval()
        with torch.no_grad():
            A_l = netA(Xv)
            du = netPsi(Xv, A_l)
            pred = Xv + dt * du
            vl = ((pred - Xvp1) ** 2).mean().item()
        hist.append((tl, vl))
        if ep % 10 == 0 or ep == n_epochs - 1:
            print(f"  epoch {ep:3d}  train={tl:.4e}  val={vl:.4e}")
    return hist


# =============================================================================
# Rollout evaluation
# =============================================================================
def rollout_baseline(model, psi0, n_steps, dt=DT, device='cpu'):
    model.eval()
    u = torch.tensor(np.stack([psi0.real, psi0.imag], axis=0)[None],
                      dtype=torch.float32).to(device)
    out = np.zeros((n_steps + 1,) + psi0.shape, dtype=np.complex64)
    out[0] = psi0
    with torch.no_grad():
        for t in range(n_steps):
            du = model(u)
            u = u + dt * du
            arr = u.squeeze().cpu().numpy()
            out[t + 1] = arr[0] + 1j * arr[1]
    return out


def rollout_gauge(netPsi, netA, psi0, n_steps, dt=DT, device='cpu'):
    netPsi.eval(); netA.eval()
    u = torch.tensor(np.stack([psi0.real, psi0.imag], axis=0)[None],
                      dtype=torch.float32).to(device)
    out = np.zeros((n_steps + 1,) + psi0.shape, dtype=np.complex64)
    out[0] = psi0
    A_traj = []
    with torch.no_grad():
        for t in range(n_steps):
            A_l = netA(u)
            du = netPsi(u, A_l)
            u = u + dt * du
            arr = u.squeeze().cpu().numpy()
            out[t + 1] = arr[0] + 1j * arr[1]
            A_traj.append(A_l.squeeze().cpu().numpy())
    return out, A_traj


def rollout_error_complex(pred, gt):
    """Relative L2 of predicted vs ground-truth complex trajectory, per step."""
    err = np.zeros(len(pred))
    for t in range(len(pred)):
        num = np.linalg.norm(pred[t] - gt[t])
        den = np.linalg.norm(gt[t]) + 1e-9
        err[t] = num / den
    return err


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 76)
    print("DMVC P12: Learned connection field on gauged complex Ginzburg-Landau")
    print("=" * 76)

    A_STYLE_TRAIN = ["swirl", "uniform"]
    A_STYLE_OOD = ["sinusoidal"]      # unseen gauge pattern
    AMP = 0.6                           # strong enough that gauge matters

    print("\n[1] Generating training data (nonzero gauge field)...")
    t0 = time.time()
    trajs_train = build_dataset(15, n_steps_per=30, A_style_list=A_STYLE_TRAIN,
                                  amplitude=AMP, seed=0)
    trajs_val = build_dataset(4, n_steps_per=30, A_style_list=A_STYLE_TRAIN,
                                amplitude=AMP, seed=1)
    trajs_ood = build_dataset(3, n_steps_per=30, A_style_list=A_STYLE_OOD,
                                amplitude=AMP, seed=2)
    print(f"    train trajectories: {len(trajs_train)}  "
          f"val: {len(trajs_val)}  ood: {len(trajs_ood)}")
    print(f"    simulation time: {time.time()-t0:.1f}s")

    Xt_tr, Xtp1_tr, _ = trajs_to_riframes(trajs_train)
    Xt_va, Xtp1_va, _ = trajs_to_riframes(trajs_val)
    print(f"    training frames: {len(Xt_tr)}")

    # ----------------------------------------------------------------
    # Model A: baseline (no gauge channel)
    # ----------------------------------------------------------------
    print("\n[2] Training BASELINE (Model A): psi-only convolutional PDE...")
    model_a = NetPsiOnly(hidden=32)
    n_a = sum(p.numel() for p in model_a.parameters())
    print(f"    Parameters: {n_a}")
    hist_a = train_baseline(model_a, Xt_tr, Xtp1_tr, Xt_va, Xtp1_va,
                              n_epochs=40, batch_size=32, lr=3e-4)

    # ----------------------------------------------------------------
    # Model B: gauge-augmented
    # ----------------------------------------------------------------
    print("\n[3] Training GAUGE (Model B): psi + learned A connection field...")
    netPsi = NetPsiGauge(hidden=32)
    netA = NetA(hidden=32)
    n_b = (sum(p.numel() for p in netPsi.parameters()) +
            sum(p.numel() for p in netA.parameters()))
    print(f"    Parameters: {n_b} ({n_b / n_a:.2f}x baseline)")
    hist_b = train_gauge(netPsi, netA, Xt_tr, Xtp1_tr, Xt_va, Xtp1_va,
                           n_epochs=40, batch_size=32, lr=3e-4)

    # ----------------------------------------------------------------
    # Rollouts
    # ----------------------------------------------------------------
    print("\n[4] Rollout evaluation on held-out trajectories (same A style)...")
    errs_a, errs_b = [], []
    for traj, Ax, Ay in trajs_val:
        psi0 = traj[0]
        n_steps = len(traj) - 1
        rb = rollout_baseline(model_a, psi0, n_steps)
        rg, _ = rollout_gauge(netPsi, netA, psi0, n_steps)
        e_a = rollout_error_complex(rb, traj)
        e_b = rollout_error_complex(rg, traj)
        errs_a.append(e_a); errs_b.append(e_b)
    errs_a = np.mean(np.array(errs_a), axis=0)
    errs_b = np.mean(np.array(errs_b), axis=0)
    print(f"    rollout step error (mean over val trajectories):")
    print(f"    {'step':>6s} {'baseline':>12s} {'gauge':>12s}")
    for t in [1, 5, 10, 15, 20, 30]:
        print(f"    {t:>6d} {errs_a[t]:>12.2%} {errs_b[t]:>12.2%}")
    final_a = float(errs_a[-1])
    final_b = float(errs_b[-1])
    rel_improvement = (final_a - final_b) / (final_a + 1e-9)

    # OOD rollout
    print("\n[5] Rollout on OOD gauge pattern (sinusoidal, unseen in training)...")
    errs_a_ood, errs_b_ood = [], []
    for traj, Ax, Ay in trajs_ood:
        psi0 = traj[0]
        n_steps = len(traj) - 1
        rb = rollout_baseline(model_a, psi0, n_steps)
        rg, _ = rollout_gauge(netPsi, netA, psi0, n_steps)
        e_a = rollout_error_complex(rb, traj)
        e_b = rollout_error_complex(rg, traj)
        errs_a_ood.append(e_a); errs_b_ood.append(e_b)
    errs_a_ood = np.mean(np.array(errs_a_ood), axis=0)
    errs_b_ood = np.mean(np.array(errs_b_ood), axis=0)
    print(f"    {'step':>6s} {'baseline':>12s} {'gauge':>12s}")
    for t in [1, 5, 10, 15, 20, 30]:
        print(f"    {t:>6d} {errs_a_ood[t]:>12.2%} {errs_b_ood[t]:>12.2%}")
    ood_a = float(errs_a_ood[-1])
    ood_b = float(errs_b_ood[-1])

    # ----------------------------------------------------------------
    # Learned A inspection
    # ----------------------------------------------------------------
    print("\n[6] Learned A-field structure (spatial statistics):")
    # Run netA at the initial state of a held-out swirl trajectory
    traj, Ax_gt, Ay_gt = trajs_val[0]
    u = torch.tensor(np.stack([traj[0].real, traj[0].imag], axis=0)[None],
                      dtype=torch.float32)
    with torch.no_grad():
        A_l = netA(u).squeeze(0).numpy()
    A_l_x = A_l[0]
    A_l_y = A_l[1]
    def stats(A):
        return {'mean': float(np.mean(A)), 'std': float(np.std(A)),
                'range': (float(A.min()), float(A.max()))}
    print(f"    Ground-truth Ax: {stats(Ax_gt)}")
    print(f"    Learned     Ax: {stats(A_l_x)}")
    print(f"    Ground-truth Ay: {stats(Ay_gt)}")
    print(f"    Learned     Ay: {stats(A_l_y)}")
    # Correlation between learned A and ground-truth A (after centering):
    # Note: the learned A is only determined up to a gauge transformation of
    # the network, so we don't expect a direct match. We report std to check
    # nontriviality.
    learned_std = float((np.std(A_l_x) + np.std(A_l_y)) / 2.0)
    print(f"    Learned A has nontrivial spatial variation: "
          f"{'YES' if learned_std > 0.1 else 'NO (possibly collapsed to constant)'}")

    # ----------------------------------------------------------------
    # Tests
    # ----------------------------------------------------------------
    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)

    t1 = final_a < 1.0                                  # baseline works at all
    t2 = final_b < final_a                              # gauge model wins held-out
    t3 = rel_improvement > 0.10                         # by at least 10%
    t4 = learned_std > 0.1                              # A is nontrivial
    t5 = ood_b < ood_a * 0.9                            # gauge model wins OOD too

    print(f"\n[T1] Baseline finishes rollout: final rel err = {final_a:.2%}  "
          f"{'PASS' if t1 else 'FAIL'}")
    print(f"[T2] Gauge < Baseline (val): {final_b:.2%} vs {final_a:.2%}  "
          f"{'PASS' if t2 else 'FAIL'}")
    print(f"[T3] Relative improvement > 10%: {rel_improvement:.1%}  "
          f"{'PASS' if t3 else 'FAIL'}")
    print(f"[T4] Learned A has structure (std > 0.1): {learned_std:.3f}  "
          f"{'PASS' if t4 else 'FAIL'}")
    print(f"[T5] Gauge < Baseline (OOD):  {ood_b:.2%} vs {ood_a:.2%}  "
          f"{'PASS' if t5 else 'FAIL'}")

    print()
    print("Interpretation:")
    print(f"  Baseline parameters: {n_a};  Gauge parameters: {n_b}")
    print("  If T2/T3 pass but T5 fails -> gauge helps but doesn't generalize")
    print("  If T2/T3/T5 all pass -> architecture learned genuinely transferable")
    print("     gauge-like structure, not just memorized training patterns")
    print("  If T3 fails -> gauge channel is redundant; baseline suffices")


if __name__ == "__main__":
    main()
