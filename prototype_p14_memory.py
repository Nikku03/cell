"""
DMVC P14: Memory bank for parameter-inference in neural PDEs.

The Dark Manifold design attaches a 'memory bank with hyperbolic retrieval'
to the field network. Before implementing it, I need a test where memory
actually helps -- otherwise I'll just add parameters without testing anything.

Task design: parameter inference from observation.

Ground truth: 2D complex Ginzburg-Landau equation
    dpsi/dt = D * nabla^2 psi + r * psi * (1 - |psi|^2)
where r is a scalar PARAMETER that VARIES per trajectory. Each trajectory
is generated with its own r drawn from a range. The network's job: predict
future evolution given current state.

  A Markovian network (sees only current state) cannot know r, so its
  predictions carry fundamental uncertainty -> systematic error.

  A memory-equipped network can observe past trajectory, infer r from
  dynamics, and use that to predict future with much lower error.

Three models, same parameter budget:
  Model A -- Markovian: single state -> next state. Baseline failure mode:
             will produce a mean-over-r prediction, suffering r-specific error.
  Model B -- Windowed: last K states -> next state. Standard RNN-substitute.
             Can infer r implicitly from dynamics in its window.
  Model C -- Retrieval: maintains a bank of recent states; uses attention
             to query the bank when predicting. The 'memory bank' mechanism.

Tests:
  T1: Markovian baseline achieves error consistent with "mean r" prediction
      (demonstrates the task is genuinely memory-hungry)
  T2: Windowed model significantly beats Markovian
  T3: Retrieval model matches or beats windowed at same K
  T4: Retrieval model tolerates longer rollouts where the original windowed
      context is stale (the architectural claim of retrieval-augmentation)
  T5 (honest null check): does the retrieval actually retrieve anything
      non-trivial? If attention weights collapse to uniform, retrieval is
      a no-op and we're measuring extra parameters only.

Honest pre-registration:
  * T1 should show baseline error in a predictable range -- consistent with
    not knowing r. If baseline does fine, the task isn't memory-hungry.
  * T2 should pass clearly. Standard RNN-style architectures do this fine.
  * T3 is the real claim. If retrieval doesn't beat a good windowed baseline,
    the Dark Manifold's memory bank is architecturally redundant on this task.
  * T4 is where retrieval's scalability shows. Windowed with K=4 fails as
    the useful info recedes past step K.
  * T5 is how we avoid another P12: check the mechanism is doing work.
"""

from __future__ import annotations
import sys, time
sys.path.insert(0, "/home/claude/dmvc")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Ground truth: complex Ginzburg-Landau with per-trajectory r
# =============================================================================
D_DIFF = 0.05
GRID_N = 16       # smaller grid; memory is across trajectories so per-frame cost matters
DT = 0.05


def periodic_laplacian_cmplx(psi):
    return (np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
             np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4 * psi)


def gl_step_with_r(psi, dt, D, r):
    lap = periodic_laplacian_cmplx(psi)
    dpsi = D * lap + r * psi * (1.0 - np.abs(psi) ** 2)
    return psi + dt * dpsi


def simulate_gl(psi0, r, n_steps, dt=DT, D=D_DIFF, sub=4):
    """Record trajectory. Sub-steps for accuracy."""
    psi = psi0.astype(np.complex128).copy()
    out = np.zeros((n_steps + 1,) + psi0.shape, dtype=np.complex64)
    out[0] = psi0
    sub_dt = dt / sub
    for t in range(n_steps):
        for _ in range(sub):
            psi = gl_step_with_r(psi, sub_dt, D, r)
        out[t + 1] = psi.astype(np.complex64)
    return out


def random_initial_psi(N, rng):
    """Small smooth perturbation around a fixed background."""
    amp = rng.uniform(0.3, 0.5)
    phase0 = rng.uniform(-0.1, 0.1)
    noise = 0.05 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
    psi = amp * np.exp(1j * phase0) * np.ones((N, N)) + noise
    # smooth
    from scipy.ndimage import gaussian_filter
    re = gaussian_filter(psi.real, sigma=0.7)
    im = gaussian_filter(psi.imag, sigma=0.7)
    return (re + 1j * im).astype(np.complex64)


def build_dataset(n_trajs, n_steps, r_range=(0.3, 1.5), seed=0):
    """Generate trajectories with r drawn per-trajectory."""
    rng = np.random.default_rng(seed)
    trajs, rs = [], []
    for _ in range(n_trajs):
        r = rng.uniform(*r_range)
        psi0 = random_initial_psi(GRID_N, rng)
        traj = simulate_gl(psi0, r, n_steps)
        trajs.append(traj)
        rs.append(r)
    return trajs, np.array(rs)


def to_ri(traj):
    """(T, N, N) complex -> (T, 2, N, N) real/imag."""
    return np.stack([traj.real, traj.imag], axis=1).astype(np.float32)


# =============================================================================
# Utilities
# =============================================================================
def circ_pad(x, n=1):
    return F.pad(x, (n, n, n, n), mode='circular')


# =============================================================================
# Model A: Markovian baseline
# =============================================================================
class NetMarkov(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.c1 = nn.Conv2d(2, hidden, 3, padding=0)
        self.c2 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.c3 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.out = nn.Conv2d(hidden, 2, 1, padding=0)

    def forward(self, psi_ri):
        x = F.gelu(self.c1(circ_pad(psi_ri)))
        x = F.gelu(self.c2(circ_pad(x)))
        x = F.gelu(self.c3(circ_pad(x)))
        return self.out(x)


# =============================================================================
# Model B: Windowed (last K states concatenated as channels)
# =============================================================================
class NetWindowed(nn.Module):
    def __init__(self, K=4, hidden=32):
        super().__init__()
        self.K = K
        self.c1 = nn.Conv2d(2 * K, hidden, 3, padding=0)
        self.c2 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.c3 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.out = nn.Conv2d(hidden, 2, 1, padding=0)

    def forward(self, psi_window):
        # psi_window: (B, K, 2, H, W) -> (B, 2K, H, W)
        B, K, C, H, W = psi_window.shape
        x = psi_window.reshape(B, K * C, H, W)
        x = F.gelu(self.c1(circ_pad(x)))
        x = F.gelu(self.c2(circ_pad(x)))
        x = F.gelu(self.c3(circ_pad(x)))
        return self.out(x)


# =============================================================================
# Model C: Retrieval-augmented
# =============================================================================
class NetRetrieval(nn.Module):
    """
    Maintains a memory bank of encoded (state -> summary vector) entries.
    When predicting the next step at a query state q, we attend over the
    memory bank using cosine similarity and retrieve a weighted combination
    of past summaries. This retrieved context is then fused with local
    convolutional features.

    The 'bank' is the last N_MEM states. N_MEM can exceed the windowed
    model's K without exploding conv-layer input channels -- that's the
    scaling claim retrieval makes.
    """
    def __init__(self, hidden=32, embed=32, n_mem=8):
        super().__init__()
        self.n_mem = n_mem
        self.embed_dim = embed

        # Encoder: state -> global summary vector
        # Small conv + global pool
        self.enc_c1 = nn.Conv2d(2, hidden, 3, padding=0)
        self.enc_c2 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.enc_proj = nn.Linear(hidden, embed)

        # Local prediction conv stack (same as NetMarkov)
        self.c1 = nn.Conv2d(2, hidden, 3, padding=0)
        self.c2 = nn.Conv2d(hidden, hidden, 3, padding=0)

        # Fusion: after c2 we have (B, hidden, H, W); we fuse with retrieved
        # embedding by broadcasting the embedding across space
        self.fuse = nn.Conv2d(hidden + embed, hidden, 1)
        self.c3 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.out = nn.Conv2d(hidden, 2, 1, padding=0)

    def encode(self, psi_ri):
        """State -> summary vector via conv + global average pool."""
        x = F.gelu(self.enc_c1(circ_pad(psi_ri)))
        x = F.gelu(self.enc_c2(circ_pad(x)))
        pooled = x.mean(dim=(2, 3))
        return self.enc_proj(pooled)

    def retrieve(self, query_emb, bank_embs, bank_summaries):
        """
        query_emb: (B, E)
        bank_embs: (B, M, E)     keys
        bank_summaries: (B, M, E)  values (here identical to embeddings but
                                    could differ)
        Returns attention-weighted retrieved vector, plus the weights.
        """
        # Cosine similarity
        q = F.normalize(query_emb, dim=-1).unsqueeze(1)                   # (B, 1, E)
        k = F.normalize(bank_embs, dim=-1)                                 # (B, M, E)
        scores = (q * k).sum(dim=-1) / 0.1                                # (B, M) - temperature 0.1
        weights = F.softmax(scores, dim=-1)
        retrieved = (weights.unsqueeze(-1) * bank_summaries).sum(dim=1)   # (B, E)
        return retrieved, weights

    def forward(self, psi_query, psi_bank):
        """
        psi_query: (B, 2, H, W) -- the current state
        psi_bank:  (B, M, 2, H, W) -- M past states in memory
        """
        B, M, C, H, W = psi_bank.shape
        # Encode everything
        q_emb = self.encode(psi_query)                                     # (B, E)
        bank_flat = psi_bank.reshape(B * M, C, H, W)
        bank_emb_flat = self.encode(bank_flat)                             # (B*M, E)
        bank_emb = bank_emb_flat.reshape(B, M, -1)

        retrieved, _weights = self.retrieve(q_emb, bank_emb, bank_emb)     # (B, E)

        # Predict via local conv, fused with retrieved context
        x = F.gelu(self.c1(circ_pad(psi_query)))
        x = F.gelu(self.c2(circ_pad(x)))
        # Broadcast retrieved across spatial dims
        retrieved_broadcast = retrieved.unsqueeze(-1).unsqueeze(-1)        # (B, E, 1, 1)
        retrieved_broadcast = retrieved_broadcast.expand(-1, -1, H, W)     # (B, E, H, W)
        x = torch.cat([x, retrieved_broadcast], dim=1)
        x = F.gelu(self.fuse(x))
        x = F.gelu(self.c3(circ_pad(x)))
        return self.out(x), _weights


# =============================================================================
# Data packing for each model
# =============================================================================
def pack_markov(trajs):
    """Return (X_t, X_tp1): (N_samples, 2, H, W) each."""
    X, Y = [], []
    for traj in trajs:
        ri = to_ri(traj)
        X.append(ri[:-1])
        Y.append(ri[1:])
    return np.concatenate(X), np.concatenate(Y)


def pack_windowed(trajs, K):
    """Return (X_t, X_tp1): X_t has shape (N, K, 2, H, W)."""
    X, Y = [], []
    for traj in trajs:
        ri = to_ri(traj)
        T = ri.shape[0]
        for t in range(K - 1, T - 1):
            window = ri[t - K + 1: t + 1]                                  # (K, 2, H, W)
            X.append(window)
            Y.append(ri[t + 1])
    return np.stack(X), np.stack(Y)


def pack_retrieval(trajs, M):
    """Similar to windowed but we'll use M slots in the memory bank."""
    X_q, X_bank, Y = [], [], []
    for traj in trajs:
        ri = to_ri(traj)
        T = ri.shape[0]
        for t in range(M, T - 1):
            bank = ri[t - M: t]                                            # (M, 2, H, W)
            query = ri[t]
            target = ri[t + 1]
            X_q.append(query); X_bank.append(bank); Y.append(target)
    return np.stack(X_q), np.stack(X_bank), np.stack(Y)


# =============================================================================
# Training
# =============================================================================
def train_markov(model, X, Y, Xv, Yv, dt=DT, n_epochs=30, bs=32, lr=3e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(X); Y = torch.tensor(Y)
    Xv = torch.tensor(Xv); Yv = torch.tensor(Yv)
    hist = []
    n = len(X)
    for ep in range(n_epochs):
        perm = torch.randperm(n)
        losses = []
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            u = X[idx]; t = Y[idx]
            du = model(u)
            pred = u + dt * du
            loss = ((pred - t) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        tl = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            du = model(Xv); pred = Xv + dt * du
            vl = ((pred - Yv) ** 2).mean().item()
        hist.append((tl, vl))
        if ep % 10 == 0 or ep == n_epochs - 1:
            print(f"  epoch {ep:3d}  train={tl:.4e}  val={vl:.4e}")
    return hist


def train_windowed(model, X, Y, Xv, Yv, dt=DT, n_epochs=30, bs=32, lr=3e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(X); Y = torch.tensor(Y)
    Xv = torch.tensor(Xv); Yv = torch.tensor(Yv)
    hist = []
    n = len(X)
    for ep in range(n_epochs):
        perm = torch.randperm(n)
        losses = []
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            u = X[idx]; t = Y[idx]
            # last state in window is the "current" state for residual
            current = u[:, -1]
            du = model(u)
            pred = current + dt * du
            loss = ((pred - t) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        tl = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            du = model(Xv)
            pred = Xv[:, -1] + dt * du
            vl = ((pred - Yv) ** 2).mean().item()
        hist.append((tl, vl))
        if ep % 10 == 0 or ep == n_epochs - 1:
            print(f"  epoch {ep:3d}  train={tl:.4e}  val={vl:.4e}")
    return hist


def train_retrieval(model, Xq, Xb, Y, Xqv, Xbv, Yv, dt=DT, n_epochs=30,
                       bs=32, lr=3e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xq = torch.tensor(Xq); Xb = torch.tensor(Xb); Y = torch.tensor(Y)
    Xqv = torch.tensor(Xqv); Xbv = torch.tensor(Xbv); Yv = torch.tensor(Yv)
    hist = []
    n = len(Xq)
    for ep in range(n_epochs):
        perm = torch.randperm(n)
        losses = []
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            q = Xq[idx]; b = Xb[idx]; t = Y[idx]
            du, _ = model(q, b)
            pred = q + dt * du
            loss = ((pred - t) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        tl = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            du, _ = model(Xqv, Xbv)
            pred = Xqv + dt * du
            vl = ((pred - Yv) ** 2).mean().item()
        hist.append((tl, vl))
        if ep % 10 == 0 or ep == n_epochs - 1:
            print(f"  epoch {ep:3d}  train={tl:.4e}  val={vl:.4e}")
    return hist


# =============================================================================
# Rollout (for each model) and error measurement
# =============================================================================
def rollout_markov(model, traj_gt, dt=DT):
    """Given ground truth traj, rollout from traj[0] forward and compare."""
    model.eval()
    T = traj_gt.shape[0]
    pred = np.zeros_like(traj_gt)
    pred[0] = traj_gt[0]
    u = torch.tensor(to_ri(traj_gt[0:1])[0:1], dtype=torch.float32)
    with torch.no_grad():
        for t in range(T - 1):
            du = model(u)
            u = u + dt * du
            arr = u.squeeze(0).cpu().numpy()
            pred[t + 1] = arr[0] + 1j * arr[1]
    return pred


def rollout_windowed(model, traj_gt, K, dt=DT):
    """Rollout using ground-truth states for the first K steps, then predict."""
    model.eval()
    T = traj_gt.shape[0]
    pred = traj_gt.copy()  # first K are true
    buf_ri = to_ri(traj_gt[:K])  # (K, 2, H, W)
    with torch.no_grad():
        for t in range(K - 1, T - 1):
            u = torch.tensor(buf_ri[None], dtype=torch.float32)
            du = model(u)
            current = torch.tensor(buf_ri[-1:], dtype=torch.float32)
            nxt = current + dt * du.squeeze(0).unsqueeze(0)
            arr = nxt.squeeze(0).cpu().numpy()
            pred[t + 1] = arr[0] + 1j * arr[1]
            # shift buffer
            buf_ri = np.concatenate([buf_ri[1:], arr[None]], axis=0)
    return pred


def rollout_retrieval(model, traj_gt, M, dt=DT):
    """Same idea: first M ground-truth states fill the bank, then predict."""
    model.eval()
    T = traj_gt.shape[0]
    pred = traj_gt.copy()
    buf_ri = to_ri(traj_gt[:M])  # (M, 2, H, W)
    with torch.no_grad():
        for t in range(M - 1, T - 1):
            query = torch.tensor(buf_ri[-1:], dtype=torch.float32)       # (1,2,H,W)
            bank = torch.tensor(buf_ri[None], dtype=torch.float32)        # (1,M,2,H,W)
            du, _ = model(query, bank)
            pred_nxt = query + dt * du
            arr = pred_nxt.squeeze(0).cpu().numpy()
            pred[t + 1] = arr[0] + 1j * arr[1]
            buf_ri = np.concatenate([buf_ri[1:], arr[None]], axis=0)
    return pred


def traj_err(pred, gt, start=0):
    """Mean rel L2 across time starting from `start`."""
    errs = []
    for t in range(start, len(pred)):
        num = np.linalg.norm(pred[t] - gt[t])
        den = np.linalg.norm(gt[t]) + 1e-12
        errs.append(num / den)
    return np.mean(errs), np.array(errs)


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 76)
    print("DMVC P14: Memory-augmented neural PDE (parameter inference task)")
    print("=" * 76)
    K = 4
    M = 6  # retrieval bank size

    # Data
    print("\n[1] Generating trajectories with per-trajectory reaction rate r...")
    t0 = time.time()
    trajs_tr, rs_tr = build_dataset(25, n_steps=25, r_range=(0.3, 1.5), seed=0)
    trajs_va, rs_va = build_dataset(6, n_steps=25, r_range=(0.3, 1.5), seed=1)
    print(f"    train: {len(trajs_tr)} trajectories (r in [{rs_tr.min():.2f}, {rs_tr.max():.2f}])")
    print(f"    val:   {len(trajs_va)} trajectories (r in [{rs_va.min():.2f}, {rs_va.max():.2f}])")
    print(f"    simulation time: {time.time()-t0:.1f}s")

    X_m, Y_m = pack_markov(trajs_tr)
    Xv_m, Yv_m = pack_markov(trajs_va)
    X_w, Y_w = pack_windowed(trajs_tr, K)
    Xv_w, Yv_w = pack_windowed(trajs_va, K)
    Xq_r, Xb_r, Y_r = pack_retrieval(trajs_tr, M)
    Xqv_r, Xbv_r, Yv_r = pack_retrieval(trajs_va, M)
    print(f"    markov frames: {len(X_m)}  windowed(K={K}): {len(X_w)}  "
          f"retrieval(M={M}): {len(Xq_r)}")

    # ---- Model A: Markovian ----
    print("\n[2] Training Model A (Markovian baseline)...")
    mA = NetMarkov(hidden=32)
    nA = sum(p.numel() for p in mA.parameters())
    print(f"    Parameters: {nA}")
    train_markov(mA, X_m, Y_m, Xv_m, Yv_m, n_epochs=30)

    # ---- Model B: Windowed ----
    print(f"\n[3] Training Model B (Windowed, K={K})...")
    mB = NetWindowed(K=K, hidden=32)
    nB = sum(p.numel() for p in mB.parameters())
    print(f"    Parameters: {nB}")
    train_windowed(mB, X_w, Y_w, Xv_w, Yv_w, n_epochs=30)

    # ---- Model C: Retrieval ----
    print(f"\n[4] Training Model C (Retrieval, M={M})...")
    mC = NetRetrieval(hidden=32, embed=32, n_mem=M)
    nC = sum(p.numel() for p in mC.parameters())
    print(f"    Parameters: {nC}")
    train_retrieval(mC, Xq_r, Xb_r, Y_r, Xqv_r, Xbv_r, Yv_r, n_epochs=30)

    # ---- Rollout evaluation ----
    print("\n[5] Rollout evaluation on val trajectories (each has different r)...")
    errs_A, errs_B, errs_C = [], [], []
    for traj in trajs_va:
        rA = rollout_markov(mA, traj)
        rB = rollout_windowed(mB, traj, K)
        rC = rollout_retrieval(mC, traj, M)
        _, eA_curve = traj_err(rA, traj)
        _, eB_curve = traj_err(rB, traj, start=K - 1)
        _, eC_curve = traj_err(rC, traj, start=M - 1)
        errs_A.append(eA_curve); errs_B.append(eB_curve); errs_C.append(eC_curve)

    # aggregate
    errs_A = np.mean([e[-1] for e in errs_A])     # final-step mean
    errs_B = np.mean([e[-1] for e in errs_B])
    errs_C = np.mean([e[-1] for e in errs_C])
    print(f"\n    Final-step rollout error (mean over val trajectories):")
    print(f"      Markovian:   {errs_A:.2%}  ({nA} params)")
    print(f"      Windowed(K={K}): {errs_B:.2%}  ({nB} params)")
    print(f"      Retrieval(M={M}): {errs_C:.2%}  ({nC} params)")

    # ---- T5: does retrieval actually retrieve anything? ----
    print("\n[6] Inspecting retrieval attention weights on val...")
    mC.eval()
    weights_all = []
    with torch.no_grad():
        Xq = torch.tensor(Xqv_r); Xb = torch.tensor(Xbv_r)
        _, weights = mC(Xq, Xb)
        weights_np = weights.cpu().numpy()    # (N_val_samples, M)
    # Nontriviality: entropy vs uniform
    H_observed = float(-np.mean(np.sum(weights_np * np.log(weights_np + 1e-12),
                                          axis=-1)))
    H_uniform = float(np.log(M))
    H_ratio = H_observed / H_uniform
    print(f"    Mean attention entropy: {H_observed:.3f} (uniform = {H_uniform:.3f})")
    print(f"    Entropy ratio: {H_ratio:.3f}  (1.0 = uniform=unused, <0.95 = selective)")
    # Also peak weight
    max_w = float(np.mean(weights_np.max(axis=-1)))
    print(f"    Mean max-weight: {max_w:.3f}  (1/M = {1.0/M:.3f} uniform, >0.3 = peaky)")

    # ---- Tests ----
    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)
    # T1: Markovian baseline should fail meaningfully (demonstrating task
    # requires memory). Specifically >10% final error.
    t1 = errs_A > 0.10
    # T2: Windowed beats Markovian
    t2 = errs_B < errs_A * 0.85
    # T3: Retrieval matches or beats windowed
    t3 = errs_C <= errs_B * 1.05
    # T5: retrieval is selective
    t5 = H_ratio < 0.95 and max_w > 0.3

    print(f"\n[T1] Markov baseline struggles (rollout err > 10%): "
          f"{errs_A:.2%}  {'PASS' if t1 else 'FAIL'}")
    print(f"[T2] Windowed beats Markov by >15%: "
          f"{errs_B:.2%} < {errs_A:.2%}  {'PASS' if t2 else 'FAIL'}")
    print(f"[T3] Retrieval matches or beats Windowed: "
          f"{errs_C:.2%} vs {errs_B:.2%}  {'PASS' if t3 else 'FAIL'}")
    print(f"[T5] Retrieval attention is selective (entropy ratio < 0.95, max weight > 0.3): "
          f"{H_ratio:.2f}, {max_w:.2f}  {'PASS' if t5 else 'FAIL'}")

    print()
    print("Interpretation:")
    print("  T1 failing = task isn't memory-hungry (baseline does fine anyway)")
    print("  T2 failing = windowed context didn't help (K too small or task easy)")
    print("  T3 failing = retrieval is no better than fixed window (memory bank redundant)")
    print("  T5 failing = retrieval attention collapsed to uniform (mechanism unused)")


if __name__ == "__main__":
    main()
