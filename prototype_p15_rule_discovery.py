"""
DMVC P15: Rule discovery on a multi-regime reaction-diffusion task.

The Dark Manifold design proposes a 'rule discovery / rule bank' layer.
Testing it requires a task where:
    (a) A single-rule baseline cannot reach low error (multi-regime dynamics)
    (b) A rule-discovery variant can separate regions and assign the right rule
The architectural claim is that explicit rule separation strictly beats
an unconstrained conv-PDE baseline on such a task.

Target system: 2D reaction-diffusion with spatially-switching reaction:
    du/dt = D * nabla^2 u + f_k(x)(u)
The domain is tiled into 3 regions, each with a different reaction f_k:
    region 0: logistic        f(u) = u*(1-u)
    region 1: cubic           f(u) = u*(1-u**2)       [same fixed points, different shape]
    region 2: decaying        f(u) = -u*(1-u)         [sign-flipped reaction]
The network is not told which region is which; it must infer.

Architectures (all matched parameter budget ~20k):
    Model A baseline:      single conv PDE network (same as P11)
    Model B mixture-of-K:  K shared rule heads + soft-assignment network
                             Forward: weighted sum over K rule outputs
    Model C hard-cluster:  same as B but argmax the assignment (hard rules)

Tests:
    T1: baseline fails (>3% rollout error) -- confirms task is multi-regime
    T2: mixture model beats baseline meaningfully (by >30% relative)
    T3: rule assignments concentrate on ground-truth region boundaries
        (diagnostic: assignments should correlate with true region mask)
    T4: hard-cluster model recovers distinct rules (each rule head is
        used somewhere; not all samples use the same rule)

Honest pre-registration:
    * T1 should pass. If baseline does fine, task isn't multi-regime enough
      and I have a P14-style redundancy problem.
    * T2 is the load-bearing claim. If mixture wins, rule separation helps.
    * T3 is the 'was the mechanism used' check. Crucial.
    * T4 detects mode collapse: a mixture that puts all weight on one
      rule is a baseline with extra parameters.
"""

from __future__ import annotations
import sys, time
sys.path.insert(0, "/home/claude/dmvc")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Ground truth: multi-regime reaction-diffusion
# =============================================================================
D_DIFFUSION = 0.05
GRID_N = 32
DT = 0.02


def periodic_laplacian(u):
    return (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
             np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u)


def multi_region_reaction(u, region_mask):
    """
    Apply region-specific reaction terms.
    region_mask: int array same shape as u, values in {0, 1, 2}.
    """
    out = np.zeros_like(u)
    # region 0: logistic
    m0 = region_mask == 0
    out[m0] = u[m0] * (1.0 - u[m0])
    # region 1: cubic with same fixed points
    m1 = region_mask == 1
    out[m1] = u[m1] * (1.0 - u[m1]**2)
    # region 2: sign-flipped (decaying toward zero rather than toward 1)
    m2 = region_mask == 2
    out[m2] = -u[m2] * (1.0 - u[m2])
    return out


def multi_regime_step(u, region_mask, dt, D=D_DIFFUSION):
    lap = periodic_laplacian(u)
    react = multi_region_reaction(u, region_mask)
    return u + dt * (D * lap + react)


def make_region_mask(N, rng, style="stripes"):
    """
    Return int array (N, N) with values in {0, 1, 2}.
    """
    mask = np.zeros((N, N), dtype=np.int64)
    if style == "stripes":
        # Three horizontal stripes
        h = N // 3
        mask[:h] = 0
        mask[h:2*h] = 1
        mask[2*h:] = 2
    elif style == "shifted_stripes":
        # Stripes but with a random vertical shift
        h = N // 3
        offset = int(rng.integers(0, N))
        rows = (np.arange(N) + offset) % N
        regions = (rows // h).clip(0, 2)
        mask = np.tile(regions[:, None], (1, N))
    elif style == "quadrants":
        # Four quadrants but we only use three of them
        mask[:N//2, :N//2] = 0
        mask[:N//2, N//2:] = 1
        mask[N//2:, :] = 2
    elif style == "circles":
        # Concentric circles
        xc = N / 2; yc = N / 2
        yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        r = np.sqrt((xx - xc)**2 + (yy - yc)**2)
        mask = np.where(r < N*0.2, 0,
                np.where(r < N*0.35, 1, 2))
    else:
        raise ValueError(style)
    return mask.astype(np.int64)


def random_initial(N, rng):
    """Smooth random initial condition in [0, 1]-ish range."""
    from scipy.ndimage import gaussian_filter
    u = rng.uniform(0.1, 0.7, size=(N, N))
    u = gaussian_filter(u, sigma=1.5)
    return u.astype(np.float32)


def simulate_multi_regime(u0, region_mask, n_steps, dt=DT, sub=4):
    u = u0.copy()
    out = np.zeros((n_steps + 1, N, N), dtype=np.float32)  # placeholder reshape
    out = np.zeros((n_steps + 1,) + u.shape, dtype=np.float32)
    out[0] = u
    sub_dt = dt / sub
    for t in range(n_steps):
        for _ in range(sub):
            u = multi_regime_step(u, region_mask, sub_dt)
        out[t + 1] = u
    return out


N = GRID_N  # alias


# =============================================================================
# Build dataset
# =============================================================================
def build_dataset(n_traj, n_steps, region_styles, seed=0):
    """Each trajectory has its own region mask (drawn from region_styles)
    and random IC."""
    rng = np.random.default_rng(seed)
    trajs, masks = [], []
    for i in range(n_traj):
        style = region_styles[i % len(region_styles)]
        mask = make_region_mask(N, rng, style=style)
        u0 = random_initial(N, rng)
        traj = simulate_multi_regime(u0, mask, n_steps)
        trajs.append(traj); masks.append(mask)
    return trajs, masks


def pack_onestep(trajs):
    X, Y = [], []
    for traj in trajs:
        X.append(traj[:-1, None])       # add channel dim
        Y.append(traj[1:, None])
    return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)


# =============================================================================
# Utils
# =============================================================================
def circ_pad(x, n=1):
    return F.pad(x, (n, n, n, n), mode='circular')


# =============================================================================
# Model A: baseline single-rule neural PDE (like P11)
# =============================================================================
class NetBaseline(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.c1 = nn.Conv2d(1, hidden, 3, padding=0)
        self.c2 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.c3 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.out = nn.Conv2d(hidden, 1, 1, padding=0)

    def forward(self, u):
        x = F.gelu(self.c1(circ_pad(u)))
        x = F.gelu(self.c2(circ_pad(x)))
        x = F.gelu(self.c3(circ_pad(x)))
        return self.out(x)


# =============================================================================
# Model B: mixture-of-K rules with soft assignment
# =============================================================================
class NetMixture(nn.Module):
    """
    Architecture:
        assignment network: u -> (K, H, W) soft weights (softmax over K)
        K independent rule heads, each a small conv stack producing du/dt
        final du/dt = sum_k w_k(x) * rule_k(u)(x)
    The shared-weights assumption across rule heads is intentional: the
    claim is that each rule is structurally a local update operator.
    """
    def __init__(self, K=3, hidden=16):
        super().__init__()
        self.K = K
        # Assignment network
        self.a1 = nn.Conv2d(1, hidden, 3, padding=0)
        self.a2 = nn.Conv2d(hidden, hidden, 3, padding=0)
        self.a_out = nn.Conv2d(hidden, K, 1)

        # K rule heads (small for parameter budget)
        rule_hidden = hidden
        self.rule_heads = nn.ModuleList()
        for _ in range(K):
            self.rule_heads.append(nn.Sequential(
                nn.Conv2d(1, rule_hidden, 3, padding=0),
            ))
        # Shared final projection per rule, 1x1
        self.rule_out = nn.ModuleList()
        for _ in range(K):
            self.rule_out.append(nn.Sequential(
                nn.GELU(),
                nn.Conv2d(rule_hidden, rule_hidden, 3, padding=0),
                nn.GELU(),
                nn.Conv2d(rule_hidden, 1, 1),
            ))

    def _pad(self, x, n=1):
        return F.pad(x, (n, n, n, n), mode='circular')

    def forward(self, u):
        # Assignment weights
        a = F.gelu(self.a1(self._pad(u)))
        a = F.gelu(self.a2(self._pad(a)))
        logits = self.a_out(a)                 # (B, K, H, W)
        weights = F.softmax(logits, dim=1)

        # Apply each rule
        B, _, H, W = u.shape
        rule_outputs = []
        for k in range(self.K):
            # Conv1 of head k
            rh = self.rule_heads[k][0](self._pad(u))
            # Apply 3x3 conv inside rule_out
            ro = self.rule_out[k][0](rh)       # gelu
            ro = self.rule_out[k][1](self._pad(ro))  # conv3x3
            ro = self.rule_out[k][2](ro)       # gelu
            ro = self.rule_out[k][3](ro)       # conv1x1 -> 1 channel
            rule_outputs.append(ro)
        rule_stack = torch.stack(rule_outputs, dim=1).squeeze(2)  # (B, K, H, W)

        # Weighted sum
        combined = (weights * rule_stack).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        return combined, weights


# =============================================================================
# Training
# =============================================================================
def train_baseline(model, X, Y, Xv, Yv, dt=DT, n_epochs=40, bs=32, lr=3e-4):
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


def train_mixture(model, X, Y, Xv, Yv, dt=DT, n_epochs=40, bs=32, lr=3e-4,
                    diversity_weight=0.0):
    """
    diversity_weight: if > 0, add a loss term that encourages the mixture
    weights to use all K rules across the batch (prevents mode collapse).
    """
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
            du, w = model(u)
            pred = u + dt * du
            main_loss = ((pred - t) ** 2).mean()
            # Optional: encourage per-batch usage of all rules
            div_loss = 0.0
            if diversity_weight > 0:
                mean_w = w.mean(dim=(0, 2, 3))   # (K,)
                # penalize if any rule has zero average usage
                div_loss = -torch.log(mean_w + 1e-6).mean()
            loss = main_loss + diversity_weight * div_loss
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(main_loss.item())
        tl = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            du, _ = model(Xv); pred = Xv + dt * du
            vl = ((pred - Yv) ** 2).mean().item()
        hist.append((tl, vl))
        if ep % 10 == 0 or ep == n_epochs - 1:
            print(f"  epoch {ep:3d}  train={tl:.4e}  val={vl:.4e}")
    return hist


# =============================================================================
# Rollout
# =============================================================================
def rollout_baseline(model, u0, n_steps, dt=DT):
    model.eval()
    u = torch.tensor(u0[None, None], dtype=torch.float32)
    out = np.zeros((n_steps + 1,) + u0.shape, dtype=np.float32)
    out[0] = u0
    with torch.no_grad():
        for t in range(n_steps):
            du = model(u); u = u + dt * du
            out[t+1] = u.squeeze().cpu().numpy()
    return out


def rollout_mixture(model, u0, n_steps, dt=DT):
    model.eval()
    u = torch.tensor(u0[None, None], dtype=torch.float32)
    out = np.zeros((n_steps + 1,) + u0.shape, dtype=np.float32)
    out[0] = u0
    weights_final = None
    with torch.no_grad():
        for t in range(n_steps):
            du, w = model(u); u = u + dt * du
            out[t+1] = u.squeeze().cpu().numpy()
            weights_final = w.squeeze(0).cpu().numpy()
    return out, weights_final


def rollout_error(pred, gt):
    errs = np.zeros(len(pred))
    for t in range(len(pred)):
        errs[t] = np.linalg.norm(pred[t] - gt[t]) / (np.linalg.norm(gt[t]) + 1e-9)
    return errs


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 76)
    print("DMVC P15: Rule discovery on multi-regime reaction-diffusion")
    print("=" * 76)

    print("\n[1] Generating multi-regime trajectories...")
    t0 = time.time()
    trajs_tr, masks_tr = build_dataset(
        20, n_steps=20, region_styles=["stripes", "shifted_stripes"], seed=0)
    trajs_va, masks_va = build_dataset(
        5, n_steps=20, region_styles=["stripes", "shifted_stripes"], seed=1)
    trajs_ood, masks_ood = build_dataset(
        3, n_steps=20, region_styles=["quadrants"], seed=2)
    print(f"    train: {len(trajs_tr)}, val: {len(trajs_va)}, "
          f"ood (quadrants): {len(trajs_ood)}")
    print(f"    sim time: {time.time()-t0:.1f}s")

    X_tr, Y_tr = pack_onestep(trajs_tr)
    X_va, Y_va = pack_onestep(trajs_va)
    print(f"    training frames: {len(X_tr)}")

    # ---- Model A baseline ----
    print("\n[2] Training Model A (single-rule baseline)...")
    mA = NetBaseline(hidden=32)
    nA = sum(p.numel() for p in mA.parameters())
    print(f"    Parameters: {nA}")
    train_baseline(mA, X_tr, Y_tr, X_va, Y_va, n_epochs=40)

    # ---- Model B mixture-of-3 ----
    print("\n[3] Training Model B (mixture of K=3 rules, with diversity loss)...")
    mB = NetMixture(K=3, hidden=16)
    nB = sum(p.numel() for p in mB.parameters())
    print(f"    Parameters: {nB}")
    train_mixture(mB, X_tr, Y_tr, X_va, Y_va, n_epochs=40,
                    diversity_weight=0.05)

    # ---- Rollout evaluation ----
    print("\n[4] Rollout evaluation on val trajectories...")
    errs_A, errs_B = [], []
    for traj, mask in zip(trajs_va, masks_va):
        rA = rollout_baseline(mA, traj[0], n_steps=len(traj)-1)
        rB, _w = rollout_mixture(mB, traj[0], n_steps=len(traj)-1)
        eA = rollout_error(rA, traj)
        eB = rollout_error(rB, traj)
        errs_A.append(eA); errs_B.append(eB)
    errs_A = np.mean(np.array(errs_A), axis=0)
    errs_B = np.mean(np.array(errs_B), axis=0)
    print(f"    {'step':>5s} {'baseline':>12s} {'mixture':>12s}")
    for t in [1, 5, 10, 15, 20]:
        print(f"    {t:>5d} {errs_A[t]:>12.2%} {errs_B[t]:>12.2%}")
    final_A = float(errs_A[-1])
    final_B = float(errs_B[-1])
    rel_improvement = (final_A - final_B) / (final_A + 1e-9)

    # ---- T3: does rule assignment correlate with ground truth regions? ----
    print("\n[5] Inspecting rule assignments on val...")
    mB.eval()
    correlations = []
    rule_usage = np.zeros(3)
    peak_weights = []
    with torch.no_grad():
        for traj, mask in zip(trajs_va, masks_va):
            u = torch.tensor(traj[0][None, None], dtype=torch.float32)
            _, w = mB(u)
            w_np = w.squeeze(0).cpu().numpy()       # (K, H, W)
            # argmax rule per pixel
            argmax_rule = w_np.argmax(axis=0)
            # usage fraction
            for k in range(3):
                rule_usage[k] += (argmax_rule == k).sum()
            # peakiness
            peak_weights.append(w_np.max(axis=0).mean())
            # Mutual information via ARI isn't straightforward without imports;
            # instead compute a simpler correlation: for each (learned_rule, true_rule)
            # pair, compute intersection / union
            # (we don't expect learned rule 0 == true rule 0, since
            # the network can permute labels)
            # Use confusion matrix and find best assignment
            M = np.zeros((3, 3))
            for k in range(3):
                for j in range(3):
                    M[k, j] = ((argmax_rule == k) & (mask == j)).sum()
            # greedy best-match
            from itertools import permutations
            best = 0
            total = mask.size
            for perm in permutations(range(3)):
                acc = sum(M[k, perm[k]] for k in range(3)) / total
                if acc > best: best = acc
            correlations.append(best)
    rule_usage = rule_usage / rule_usage.sum()
    mean_peak = float(np.mean(peak_weights))
    mean_agreement = float(np.mean(correlations))
    print(f"    Best-match rule/region agreement: {mean_agreement:.2%}  "
          "(chance = 33%)")
    print(f"    Mean peak weight: {mean_peak:.3f}  (1/3 uniform = 0.333)")
    print(f"    Rule usage fractions: {rule_usage}")

    # ---- Tests ----
    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)
    t1 = final_A > 0.03
    t2 = rel_improvement > 0.30
    t3 = mean_agreement > 0.5 and mean_peak > 0.5
    t4 = rule_usage.min() > 0.05     # all rules used at least a bit

    print(f"\n[T1] Baseline fails meaningfully (>3% rollout err): "
          f"{final_A:.2%}  {'PASS' if t1 else 'FAIL'}")
    print(f"[T2] Mixture beats baseline by >30%: "
          f"{final_B:.2%} vs {final_A:.2%}, "
          f"improvement {rel_improvement:.1%}  {'PASS' if t2 else 'FAIL'}")
    print(f"[T3] Rule assignments correlate with ground truth regions: "
          f"{mean_agreement:.2%}, peak {mean_peak:.3f}  "
          f"{'PASS' if t3 else 'FAIL'}")
    print(f"[T4] All K=3 rules are used (min usage > 5%): "
          f"usage = {rule_usage}  {'PASS' if t4 else 'FAIL'}")

    print()
    print("Interpretation:")
    print(f"  Baseline params: {nA}, Mixture params: {nB}")
    print("  T1 FAIL = baseline solves the task on its own; task not multi-regime enough")
    print("  T2 FAIL = mixture doesn't help; rule separation not load-bearing")
    print("  T3 FAIL = rules not aligned with true regions; learned something else")
    print("  T4 FAIL = mode collapse; effective K < 3")


if __name__ == "__main__":
    main()
