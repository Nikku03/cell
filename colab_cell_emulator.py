"""Colab cell: v3 cell-state emulator - phase-conditioned generator.

v1 failed on normalisation; v2 fixed that but only learned persistence
and its chained rollout diverged (R^2 ~ -3e8). The diagnosis: predicting
tiny 96-second steps and chaining them both teaches nothing and
snowballs.

v3 drops the autoregressive framing entirely. Instead of "predict the
next moment from the last", it learns the cell's whole LIFE STORY as a
direct function of cell-cycle phase:

    state  ~=  f(phase, z)

  - phase : how far through the ~2-hour cycle (0..1)
  - z     : a per-cell latent vector - which stochastic individual this
            is. Each of the 40 training cells gets its own learned z
            (an "auto-decoder"); a new cell is a freshly sampled z.

There is NO chaining, so it CANNOT diverge. To make a 51st trajectory:
sample a z, evaluate f at phase 0,1/T,...,1. To get the state at any
time t: evaluate f(t/T, z). It learns what a wild-type syn3A cell cycle
looks like; it produces the smooth underlying arc (fine second-to-second
stochastic jitter is a later add).

Run on Colab with Drive mounted, GPU runtime.
"""

import glob
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- config ----------------
PARQUET_DIR = ""          # set to the folder if the glob finds nothing
TIME_STRIDE = 6           # decimate 7201 -> ~1201 steps
N_TRAIN_TRAJ = 40         # of 50; the rest are held out
LATENT_DIM = 16           # per-cell latent vector size
N_FREQ = 10               # sinusoidal phase-embedding frequencies
HIDDEN = 512
DEPTH = 4
STEPS = 8000
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5
LAMBDA_Z = 1e-3           # keeps per-cell latents compact so new z's sample cleanly
FIT_STEPS = 500           # steps to fit a latent for a held-out cell
FIT_LR = 0.05
SEED = 0
SAVE_DIR = "/content/drive/MyDrive"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
print(f"[device] {device}")


# ---------------- load + normalise (the v2 pipeline - it works) ----------------
def load_data():
    pat = (f"{PARQUET_DIR}/counts_and_fluxes*.parquet" if PARQUET_DIR
           else "/content/drive/MyDrive/**/counts_and_fluxes*.parquet")
    files = sorted(glob.glob(pat, recursive=True),
                   key=lambda p: int(p.rsplit(".", 2)[-2]))
    assert files, "no parquet files found - set PARQUET_DIR"
    print(f"[data] {len(files)} trajectory files")
    trajs = []
    for f in files:
        arr = pd.read_parquet(f).to_numpy(dtype=np.float32)   # (species, time)
        trajs.append(arr[:, ::TIME_STRIDE].T)                 # -> (time, species)
    return np.stack(trajs, axis=0)


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


data = load_data()
n_traj, T, S_full = data.shape
print(f"[data] stacked {data.shape}  (traj, time, species)")

rng = np.random.RandomState(SEED)
perm = rng.permutation(n_traj)
train_idx, test_idx = perm[:N_TRAIN_TRAJ], perm[N_TRAIN_TRAJ:]

data = signed_log(data)
dtr = data[train_idx]
lo = np.percentile(dtr, 0.5, axis=(0, 1))
hi = np.percentile(dtr, 99.5, axis=(0, 1))
del dtr
span = hi - lo
active = span > 1e-6
print(f"[data] active species: {int(active.sum())} / {S_full}")
data = data[:, :, active]
lo, span = lo[active], span[active]
data = np.clip((data - lo) / span, -0.2, 1.2).astype(np.float32)
S = data.shape[2]

X = torch.from_numpy(data)
train_X = X[train_idx].to(device)                            # (40, T, S)
test_X = X[test_idx].to(device)                              # (10, T, S)
print(f"[data] train {tuple(train_X.shape)}  test {tuple(test_X.shape)}")


# ---------------- model ----------------
class PhaseGenerator(nn.Module):
    """state ~= f(phase, z): a sinusoidal embedding of cell-cycle phase
    and a per-cell latent z, fed through an MLP. No recurrence, no
    chaining - so a generated trajectory cannot diverge."""

    def __init__(self, n_cells, S, latent_dim, n_freq, hidden, depth):
        super().__init__()
        self.n_freq = n_freq
        self.z = nn.Embedding(n_cells, latent_dim)
        nn.init.normal_(self.z.weight, std=0.1)
        layers = [nn.Linear(2 * n_freq + latent_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, S)]
        self.mlp = nn.Sequential(*layers)

    def phase_embed(self, phase):                 # phase: (B,) in [0,1]
        freqs = (2.0 ** torch.arange(self.n_freq, device=phase.device)) * torch.pi
        ang = phase[:, None] * freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, phase, z):                  # phase (B,), z (B, latent)
        return self.mlp(torch.cat([self.phase_embed(phase), z], dim=-1))


model = PhaseGenerator(N_TRAIN_TRAJ, S, LATENT_DIM, N_FREQ, HIDDEN, DEPTH).to(device)
print(f"[model] {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot.clamp(min=1e-12))


# ---------------- train ----------------
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

t0 = time.time()
model.train()
for step in range(STEPS):
    i = torch.randint(0, N_TRAIN_TRAJ, (BATCH,), device=device)
    t = torch.randint(0, T, (BATCH,), device=device)
    phase = t.float() / (T - 1)
    z = model.z(i)
    pred = model(phase, z)
    tgt = train_X[i, t]
    loss = F.mse_loss(pred, tgt) + LAMBDA_Z * (z ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step == 0 or (step + 1) % 1000 == 0:
        print(f"  step {step+1:5d}  loss {float(loss.detach()):.5f}", flush=True)
print(f"[train] {STEPS} steps in {time.time()-t0:.0f}s")


# ---------------- evaluate ----------------
phases = (torch.arange(T, device=device).float() / (T - 1))


@torch.no_grad()
def reconstruct(z_vec):                           # z_vec: (latent,) -> (T, S)
    return model(phases, z_vec[None, :].expand(T, -1))


def fit_latent(traj):
    """Find the latent z that best reproduces a held-out trajectory
    (model frozen). Tests whether the learned f(phase, .) can represent
    a cell it never saw."""
    z = torch.zeros(LATENT_DIM, device=device, requires_grad=True)
    opt_z = torch.optim.Adam([z], lr=FIT_LR)
    for _ in range(FIT_STEPS):
        pred = model(phases, z[None, :].expand(T, -1))
        loss = F.mse_loss(pred, traj)
        opt_z.zero_grad()
        loss.backward()
        opt_z.step()
    return z.detach()


model.eval()
# train reconstruction (sanity: the model should fit what it trained on)
with torch.no_grad():
    tr_r2 = np.mean([r2(reconstruct(model.z.weight[k]), train_X[k])
                     for k in range(N_TRAIN_TRAJ)])

z_mean = model.z.weight.detach().mean(0)
z_std = model.z.weight.detach().std(0)

# held-out: average cell (z = mean) vs per-cell fitted z
with torch.no_grad():
    avg_r2 = np.mean([r2(reconstruct(z_mean), test_X[k])
                      for k in range(test_X.shape[0])])
fit_r2 = np.mean([r2(reconstruct(fit_latent(test_X[k])), test_X[k])
                  for k in range(test_X.shape[0])])

print()
print("=" * 64)
print(f"  train reconstruction   : R^2 {tr_r2:.3f}  (fits the 40 it trained on)")
print(f"  held-out, average cell : R^2 {avg_r2:.3f}  (z=mean; common arc only)")
print(f"  held-out, fitted cell  : R^2 {fit_r2:.3f}  (z fit per cell; generalisation)")
print("=" * 64)


# ---------------- generate a 51st trajectory ----------------
torch.manual_seed(SEED + 7)
z_new = z_mean + z_std * torch.randn(LATENT_DIM, device=device)
with torch.no_grad():
    gen_norm = reconstruct(z_new).cpu().numpy()               # (T, S)
sl = np.clip(gen_norm * span + lo, -15.0, 15.0)               # undo norm (clip = safety)
gen_counts = np.sign(sl) * np.expm1(np.abs(sl))               # -> counts
print(f"[gen] 51st trajectory {gen_counts.shape}  finite={np.isfinite(gen_counts).all()}"
      f"  count range [{gen_counts.min():.0f}, {gen_counts.max():.0f}]")

np.save(f"{SAVE_DIR}/cell_traj_51.npy", gen_counts)
torch.save({"model": model.state_dict(), "lo": lo, "span": span, "active": active,
            "z_mean": z_mean.cpu(), "z_std": z_std.cpu(),
            "config": dict(S=S, latent_dim=LATENT_DIM, n_freq=N_FREQ,
                           hidden=HIDDEN, depth=DEPTH, T=T)},
           f"{SAVE_DIR}/cell_emulator_v3.pt")
print(f"[save] trajectory -> {SAVE_DIR}/cell_traj_51.npy")
print(f"[save] model      -> {SAVE_DIR}/cell_emulator_v3.pt")
