"""Colab cell: v4 cell-state emulator - sharper, guard-railed dynamics model.

v3-B's Option-B fixes killed the divergence: rollout R^2 went from
v2's -2.8e8 to +0.478, finite and stable. But it had two warts:
  - 1-step prediction was slightly WORSE than persistence (rollout-only
    training traded single-step sharpness for stability),
  - some species overshot - generated counts ran up to ~3 million
    (nothing real exceeds ~110k) and a few went negative.

v4 fixes both:
  1. COMBINED loss = rollout MSE + an explicit 1-step term, so the model
     is sharp on the next step AND stable over the rollout.
  2. CLAMP guardrail - the state fed back into the rollout is clamped to
     the data's normalised range, so no species can balloon out of
     range; final counts are clamped >= 0.
  3. Longer rollout curriculum (K ramps 1 -> 64) - smaller gap between
     the training horizon and the ~113-step eval rollout.

The reaction-network / conservation upgrade (which would enforce that
molecules can't appear from nothing) is v5 - it needs the SBML files
staged and species-name alignment checked first.

Run on Colab with Drive mounted, GPU runtime (~8 min).
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
TIME_STRIDE = 60          # decimate 7201 -> ~121 steps (~60 s each)
CONTEXT = 8
D_MODEL = 256
N_LAYERS = 3
N_HEADS = 8
DROPOUT = 0.1
N_TRAIN_TRAJ = 40
STEPS = 2000
K_MAX = 64                # rollout-curriculum unroll length ramps 1 -> K_MAX
BATCH = 32
LR = 3e-4
WEIGHT_DECAY = 1e-5
LAMBDA_1STEP = 1.0        # weight on the explicit 1-step accuracy term
CLAMP_LO, CLAMP_HI = -0.2, 1.2   # normalised data range; rollout guardrail
SEED = 0
SAVE_DIR = "/content/drive/MyDrive"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
print(f"[device] {device}")


# ---------------- load + normalise ----------------
def load_data():
    pat = (f"{PARQUET_DIR}/counts_and_fluxes*.parquet" if PARQUET_DIR
           else "/content/drive/MyDrive/**/counts_and_fluxes*.parquet")
    files = sorted(glob.glob(pat, recursive=True),
                   key=lambda p: int(p.rsplit(".", 2)[-2]))
    assert files, "no parquet files found - set PARQUET_DIR"
    print(f"[data] {len(files)} trajectory files")
    trajs = []
    for f in files:
        arr = pd.read_parquet(f).to_numpy(dtype=np.float32)
        trajs.append(arr[:, ::TIME_STRIDE].T)
    return np.stack(trajs, axis=0)


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


data = load_data()
n_traj, T, S_full = data.shape
print(f"[data] stacked {data.shape}  (traj, time, species)  - {T} steps of ~{TIME_STRIDE}s")

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
data = np.clip((data - lo) / span, CLAMP_LO, CLAMP_HI).astype(np.float32)
S = data.shape[2]

X = torch.from_numpy(data)
train_X = X[train_idx].to(device)
test_X = X[test_idx].to(device)
print(f"[data] train {tuple(train_X.shape)}  test {tuple(test_X.shape)}")


# ---------------- model ----------------
class DynamicsModel(nn.Module):
    """Transformer over a CONTEXT-step window -> next state, as a residual."""

    def __init__(self, S, d_model, n_layers, n_heads, context, dropout):
        super().__init__()
        self.in_proj = nn.Linear(S, d_model)
        self.ctx_pos = nn.Parameter(torch.randn(context, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                         dropout=dropout, batch_first=True,
                                         norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.out = nn.Linear(d_model, S)

    def forward(self, ctx):                       # ctx: (B, C, S) -> (B, S)
        h = self.encoder(self.in_proj(ctx) + self.ctx_pos)
        return ctx[:, -1] + self.out(h[:, -1])    # residual


model = DynamicsModel(S, D_MODEL, N_LAYERS, N_HEADS, CONTEXT, DROPOUT).to(device)
print(f"[model] {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot.clamp(min=1e-12))


persist_mse = float(F.mse_loss(test_X[:, :-1], test_X[:, 1:]))
persist_r2 = r2(test_X[:, :-1], test_X[:, 1:])
print(f"[diag] persistence 1-step (test): MSE {persist_mse:.5f}  R^2 {persist_r2:.3f}")


# ---------------- train (combined loss + rollout curriculum) ----------------
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
gen = torch.Generator().manual_seed(SEED + 1)

t_start = time.time()
model.train()
for step in range(STEPS):
    K = 1 + int((K_MAX - 1) * step / STEPS)
    i_t = torch.randint(0, N_TRAIN_TRAJ, (BATCH,), generator=gen)
    t0_t = torch.randint(0, T - CONTEXT - K, (BATCH,), generator=gen)
    i, t0 = i_t.tolist(), t0_t.tolist()
    ctx = torch.stack([train_X[i[b], t0[b]:t0[b] + CONTEXT] for b in range(BATCH)])
    i_dev, t0_dev = i_t.to(device), t0_t.to(device)
    losses = []
    for k in range(K):
        pred = model(ctx)                                       # raw prediction
        true = train_X[i_dev, t0_dev + CONTEXT + k]
        losses.append(F.mse_loss(pred, true))                   # score the raw pred
        ctx = torch.cat([ctx[:, 1:],
                         pred.clamp(CLAMP_LO, CLAMP_HI).unsqueeze(1)], dim=1)
    rollout_loss = torch.stack(losses).mean()
    loss = rollout_loss + LAMBDA_1STEP * losses[0]              # combined
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step == 0 or (step + 1) % 250 == 0:
        print(f"  step {step+1:5d}  K={K:2d}  1-step {float(losses[0].detach()):.5f}"
              f"  rollout {float(rollout_loss.detach()):.5f}", flush=True)
print(f"[train] {STEPS} steps in {time.time()-t_start:.0f}s")


# ---------------- evaluate ----------------
@torch.no_grad()
def one_step(Xset, n=400):
    model.eval()
    g = torch.Generator().manual_seed(SEED + 2)
    nt, Tt, _ = Xset.shape
    i = torch.randint(0, nt, (n,), generator=g).tolist()
    t0 = torch.randint(0, Tt - CONTEXT - 1, (n,), generator=g).tolist()
    ctx = torch.stack([Xset[i[b], t0[b]:t0[b] + CONTEXT] for b in range(n)])
    nxt = torch.stack([Xset[i[b], t0[b] + CONTEXT] for b in range(n)])
    pred = model(ctx)
    return float(F.mse_loss(pred, nxt)), r2(pred, nxt)


@torch.no_grad()
def full_rollout(traj):
    model.eval()
    ctx = traj[:CONTEXT].unsqueeze(0)
    preds = []
    for _ in range(traj.shape[0] - CONTEXT):
        p = model(ctx).clamp(CLAMP_LO, CLAMP_HI)               # guard-railed
        preds.append(p)
        ctx = torch.cat([ctx[:, 1:], p.unsqueeze(1)], dim=1)
    return torch.cat(preds, 0), traj[CONTEXT:]


s_mse, s_r2 = one_step(test_X)
roll_r2 = [r2(*full_rollout(test_X[k])) for k in range(test_X.shape[0])]
mean_roll = sum(roll_r2) / len(roll_r2)

print()
print("=" * 64)
print(f"  persistence 1-step   : MSE {persist_mse:.5f}   R^2 {persist_r2:.3f}")
print(f"  model 1-step (test)  : MSE {s_mse:.5f}   R^2 {s_r2:.3f}"
      f"   {'(beats persistence)' if s_mse < persist_mse else '(still worse)'}")
print(f"  model full rollout   : R^2 {mean_roll:.3f}   "
      f"(per-traj min {min(roll_r2):.3f}, max {max(roll_r2):.3f})  <- headline")
print(f"  (v3-B for reference  : rollout R^2 0.478)")
print("=" * 64)


# ---------------- generate + save ----------------
gen_norm, _ = full_rollout(test_X[0])
full = torch.cat([test_X[0, :CONTEXT], gen_norm], dim=0).cpu().numpy()
sl = full * span + lo
gen_counts = np.maximum(np.sign(sl) * np.expm1(np.abs(sl)), 0.0)   # counts >= 0
print(f"[gen] 51st trajectory {gen_counts.shape}  finite={np.isfinite(gen_counts).all()}"
      f"  count range [{gen_counts.min():.0f}, {gen_counts.max():.0f}]")
np.save(f"{SAVE_DIR}/cell_traj_51.npy", gen_counts)
torch.save({"model": model.state_dict(), "lo": lo, "span": span, "active": active,
            "config": dict(S=S, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                           context=CONTEXT, time_stride=TIME_STRIDE)},
           f"{SAVE_DIR}/cell_emulator_v4.pt")
print(f"[save] trajectory -> {SAVE_DIR}/cell_traj_51.npy")
print(f"[save] model      -> {SAVE_DIR}/cell_emulator_v4.pt")
