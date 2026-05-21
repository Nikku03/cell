"""Colab cell: v1 cell-state emulator for the 4DWCM minimal-cell data.

Trains a Transformer to predict the JCVI-syn3A cell state forward in
time, on the 50 `counts_and_fluxes` trajectories (each 8572 species x
7201 one-second steps over the ~2-hour cell cycle).

Design choices, and why:
  - DIRECT multi-step: given a CONTEXT-step window the model predicts
    the next HORIZON steps in one shot, not autoregressively. This
    structurally avoids the per-step compounding-error blow-up that
    drove the earlier m7 model's rollout R^2 to ~-1e8.
  - RESIDUAL head: the model predicts the *change* from the last
    observed state. Cell counts move smoothly, so "stay similar" is a
    strong, free prior.
  - SHARED decoder head: HORIZON future states are produced by a
    Transformer decoder with shared output weights - not a giant flat
    Linear - so an 8572-dim output from only 50 trajectories does not
    explode the parameter count.

Run on Colab with Drive mounted (a GPU runtime; standard RAM is fine
at TIME_STRIDE=6). It trains, evaluates honestly on held-out
trajectories (window + rollout R^2), and writes a generated "51st"
trajectory + the model to Drive.
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
TIME_STRIDE = 6           # decimate 7201 -> ~1201 steps (keeps RAM ~2 GB)
CONTEXT = 16              # input window length (steps)
HORIZON = 16              # predict this many steps directly (== CONTEXT)
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 8
N_TRAIN_TRAJ = 40         # of 50; the rest are held out
STEPS = 4000
BATCH = 32
LR = 3e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.1
SEED = 0
SAVE_DIR = "/content/drive/MyDrive"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
print(f"[device] {device}")


# ---------------- load data ----------------
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
    data = np.stack(trajs, axis=0)                            # (traj, time, species)
    print(f"[data] stacked {data.shape}  (traj, time, species)")
    return data


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


data = load_data()
n_traj, T, S = data.shape

# split trajectories, normalise with TRAIN statistics only
rng = np.random.RandomState(SEED)
perm = rng.permutation(n_traj)
train_idx, test_idx = perm[:N_TRAIN_TRAJ], perm[N_TRAIN_TRAJ:]

data = signed_log(data)                                       # overwrite to save RAM
mu = data[train_idx].mean(axis=(0, 1))                        # (S,)
sd = data[train_idx].std(axis=(0, 1)).clip(min=1e-6)          # (S,)
data = (data - mu) / sd

X = torch.from_numpy(data).float()
train_X = X[train_idx].to(device)                             # (40, T, S)
test_X = X[test_idx].to(device)                               # (10, T, S)
print(f"[data] train {tuple(train_X.shape)}  test {tuple(test_X.shape)}")


# ---------------- model ----------------
class CellEmulator(nn.Module):
    """Encoder-decoder Transformer: encode a CONTEXT-step window, then a
    decoder with HORIZON learned query tokens cross-attends to it and
    emits HORIZON future states. Output is a residual from the last
    observed state."""

    def __init__(self, S, d_model, n_layers, n_heads, context, horizon, dropout):
        super().__init__()
        self.context, self.horizon = context, horizon
        self.in_proj = nn.Linear(S, d_model)
        self.ctx_pos = nn.Parameter(torch.randn(context, d_model) * 0.02)
        self.query = nn.Parameter(torch.randn(horizon, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                         dropout=dropout, batch_first=True,
                                         norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        dec = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model,
                                         dropout=dropout, batch_first=True,
                                         norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, n_layers)
        self.out = nn.Linear(d_model, S)

    def forward(self, ctx):                       # ctx: (B, context, S)
        mem = self.encoder(self.in_proj(ctx) + self.ctx_pos)      # (B, C, d)
        q = self.query.unsqueeze(0).expand(ctx.shape[0], -1, -1)  # (B, H, d)
        delta = self.out(self.decoder(q, mem))                    # (B, H, S)
        return ctx[:, -1:] + delta                                # residual


model = CellEmulator(S, D_MODEL, N_LAYERS, N_HEADS, CONTEXT, HORIZON,
                     DROPOUT).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"[model] {n_params/1e6:.1f}M parameters")


# ---------------- windows ----------------
def sample_windows(Xset, batch, gen):
    """Random (context, target) windows from a set of trajectories."""
    nt, Tt, _ = Xset.shape
    ti = torch.randint(0, nt, (batch,), generator=gen).tolist()
    t0 = torch.randint(0, Tt - CONTEXT - HORIZON, (batch,), generator=gen).tolist()
    ctx = torch.stack([Xset[ti[b], t0[b]:t0[b] + CONTEXT] for b in range(batch)])
    tgt = torch.stack([Xset[ti[b], t0[b] + CONTEXT:t0[b] + CONTEXT + HORIZON]
                       for b in range(batch)])
    return ctx, tgt


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot.clamp(min=1e-12))


# ---------------- train ----------------
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
gen = torch.Generator().manual_seed(SEED + 1)

t0 = time.time()
model.train()
for step in range(STEPS):
    ctx, tgt = sample_windows(train_X, BATCH, gen)
    pred = model(ctx)
    loss = F.mse_loss(pred, tgt)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if (step + 1) % 500 == 0:
        print(f"  step {step+1:5d}  train MSE {float(loss):.4f}", flush=True)
train_time = time.time() - t0
print(f"[train] {STEPS} steps in {train_time:.0f}s")


# ---------------- evaluate ----------------
@torch.no_grad()
def eval_windows(Xset, n=400):
    model.eval()
    g = torch.Generator().manual_seed(SEED + 2)
    ctx, tgt = sample_windows(Xset, n, g)
    pred = model(ctx)
    return float(F.mse_loss(pred, tgt)), r2(pred, tgt)


@torch.no_grad()
def rollout(traj):
    """Generate a whole trajectory from its first CONTEXT steps, in
    HORIZON-step blocks. Returns (generated, truth) for the predicted part."""
    model.eval()
    ctx = traj[:CONTEXT].unsqueeze(0)                 # (1, C, S)
    blocks = []
    while CONTEXT + len(blocks) * HORIZON < traj.shape[0]:
        nxt = model(ctx)                              # (1, H, S)
        blocks.append(nxt)
        ctx = nxt                                     # H == C: predicted block is next context
    gen = torch.cat(blocks, dim=1)[0]                 # (n_blocks*H, S)
    n = min(gen.shape[0], traj.shape[0] - CONTEXT)
    return gen[:n], traj[CONTEXT:CONTEXT + n]


w_mse, w_r2 = eval_windows(test_X)
roll_r2 = [r2(*rollout(test_X[i])) for i in range(test_X.shape[0])]
mean_roll = sum(roll_r2) / len(roll_r2)

print()
print("=" * 60)
print(f"  held-out window  : MSE {w_mse:.4f}   R^2 {w_r2:.3f}")
print(f"  held-out rollout : R^2 {mean_roll:.3f}   "
      f"(per-traj min {min(roll_r2):.3f}, max {max(roll_r2):.3f})")
print(f"  reference        : m7 rollout R^2 was ~ -1.5e8")
print("=" * 60)


# ---------------- save model + a generated 51st trajectory ----------------
gen_norm, _ = rollout(test_X[0])
full = torch.cat([test_X[0, :CONTEXT], gen_norm], dim=0).cpu().numpy()
mu_t, sd_t = mu[None, :], sd[None, :]
sl = full * sd_t + mu_t                                   # undo z-score
gen_counts = np.sign(sl) * np.expm1(np.abs(sl))           # undo signed-log -> counts
np.save(f"{SAVE_DIR}/cell_traj_51.npy", gen_counts)
torch.save({"model": model.state_dict(), "mu": mu, "sd": sd,
            "config": dict(S=S, d_model=D_MODEL, n_layers=N_LAYERS,
                           n_heads=N_HEADS, context=CONTEXT, horizon=HORIZON)},
           f"{SAVE_DIR}/cell_emulator_v1.pt")
print(f"[save] generated trajectory -> {SAVE_DIR}/cell_traj_51.npy  {gen_counts.shape}")
print(f"[save] model -> {SAVE_DIR}/cell_emulator_v1.pt")
