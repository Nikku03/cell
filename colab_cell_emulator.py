"""Colab cell: v2 cell-state emulator for the 4DWCM minimal-cell data.

v1 did not learn: training MSE sat flat at ~0.5 while held-out MSE blew
up to ~4e5 - a 6-orders-of-magnitude gap on the same normalised scale.
The cause was the per-species z-score: with 8572 species many are
sparse or near-constant, so dividing their rare values by a tiny std
produced z-scores in the hundreds. Those few garbage species dominated
the loss and drowned the gradient for the species that ARE learnable.

v2 fixes the data pipeline:
  - drop near-constant species (a percentile-range filter - this also
    removes the ultra-sparse species that exploded under z-scoring),
  - robust BOUNDED normalisation: signed-log, then per-species map of
    the [0.5, 99.5] percentile range to [0,1], then clip - so no value
    can explode,
and prints a PERSISTENCE baseline: the MSE of predicting "no change".
The model must beat that; if it cannot, it is not learning.

The model (direct multi-step encoder-decoder Transformer, residual
head) is unchanged - this run isolates the normalisation fix.

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


# ---------------- load ----------------
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
    return np.stack(trajs, axis=0)                            # (traj, time, species)


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


data = load_data()
n_traj, T, S_full = data.shape
print(f"[data] stacked {data.shape}  (traj, time, species)")

rng = np.random.RandomState(SEED)
perm = rng.permutation(n_traj)
train_idx, test_idx = perm[:N_TRAIN_TRAJ], perm[N_TRAIN_TRAJ:]

# ---- robust, bounded normalisation (the v2 fix) ----
data = signed_log(data)
dtr = data[train_idx]
lo = np.percentile(dtr, 0.5, axis=(0, 1))                     # (S_full,)
hi = np.percentile(dtr, 99.5, axis=(0, 1))
del dtr
span = hi - lo
active = span > 1e-6                                          # species that actually vary
print(f"[data] active species: {int(active.sum())} / {S_full}  "
      f"(dropped {S_full - int(active.sum())} near-constant)")

data = data[:, :, active]
lo, span = lo[active], span[active]
data = np.clip((data - lo) / span, -0.2, 1.2).astype(np.float32)
S = data.shape[2]

X = torch.from_numpy(data)
train_X = X[train_idx].to(device)
test_X = X[test_idx].to(device)
print(f"[data] train {tuple(train_X.shape)}  test {tuple(test_X.shape)}")
print(f"[diag] normalised value range: "
      f"[{float(X.min()):.2f}, {float(X.max()):.2f}]  (bounded -> fix applied)")


# ---------------- model ----------------
class CellEmulator(nn.Module):
    """Encode a CONTEXT-step window; a decoder with HORIZON learned query
    tokens cross-attends to it and emits HORIZON future states. The
    output is a residual from the last observed state."""

    def __init__(self, S, d_model, n_layers, n_heads, context, horizon, dropout):
        super().__init__()
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
        mem = self.encoder(self.in_proj(ctx) + self.ctx_pos)
        q = self.query.unsqueeze(0).expand(ctx.shape[0], -1, -1)
        delta = self.out(self.decoder(q, mem))
        return ctx[:, -1:] + delta


model = CellEmulator(S, D_MODEL, N_LAYERS, N_HEADS, CONTEXT, HORIZON,
                     DROPOUT).to(device)
print(f"[model] {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")


# ---------------- windows / metrics ----------------
def sample_windows(Xset, batch, gen):
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


# persistence baseline: predict "no change" - the model must beat this
with torch.no_grad():
    g0 = torch.Generator().manual_seed(SEED + 9)
    c, t = sample_windows(test_X, 400, g0)
    persist_mse = float(F.mse_loss(c[:, -1:].expand_as(t), t))
    persist_r2 = r2(c[:, -1:].expand_as(t), t)
print(f"[diag] PERSISTENCE baseline (test): MSE {persist_mse:.5f}  R^2 {persist_r2:.3f}")
print(f"       -> the trained model must beat MSE {persist_mse:.5f}")


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
    if step == 0 or (step + 1) % 250 == 0:
        print(f"  step {step+1:5d}  train MSE {float(loss.detach()):.5f}", flush=True)
print(f"[train] {STEPS} steps in {time.time()-t0:.0f}s")


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
    model.eval()
    ctx = traj[:CONTEXT].unsqueeze(0)
    blocks = []
    while CONTEXT + len(blocks) * HORIZON < traj.shape[0]:
        nxt = model(ctx)
        blocks.append(nxt)
        ctx = nxt
    gen_t = torch.cat(blocks, dim=1)[0]
    n = min(gen_t.shape[0], traj.shape[0] - CONTEXT)
    return gen_t[:n], traj[CONTEXT:CONTEXT + n]


w_mse, w_r2 = eval_windows(test_X)
roll_r2 = [r2(*rollout(test_X[i])) for i in range(test_X.shape[0])]
mean_roll = sum(roll_r2) / len(roll_r2)

print()
print("=" * 64)
print(f"  persistence baseline : MSE {persist_mse:.5f}   R^2 {persist_r2:.3f}")
print(f"  held-out window      : MSE {w_mse:.5f}   R^2 {w_r2:.3f}"
      f"   {'(beats persistence)' if w_mse < persist_mse else '(WORSE than persistence)'}")
print(f"  held-out rollout     : R^2 {mean_roll:.3f}   "
      f"(per-traj min {min(roll_r2):.3f}, max {max(roll_r2):.3f})")
print("=" * 64)


# ---------------- save model + generated trajectory ----------------
gen_norm, _ = rollout(test_X[0])
full = torch.cat([test_X[0, :CONTEXT], gen_norm], dim=0).cpu().numpy()
sl = full * span + lo                                      # undo bounded norm
gen_counts = np.sign(sl) * np.expm1(np.abs(sl))            # undo signed-log -> counts
np.save(f"{SAVE_DIR}/cell_traj_51.npy", gen_counts)
torch.save({"model": model.state_dict(), "lo": lo, "span": span, "active": active,
            "config": dict(S=S, d_model=D_MODEL, n_layers=N_LAYERS,
                           n_heads=N_HEADS, context=CONTEXT, horizon=HORIZON)},
           f"{SAVE_DIR}/cell_emulator_v2.pt")
print(f"[save] generated trajectory -> {SAVE_DIR}/cell_traj_51.npy  {gen_counts.shape}")
print(f"[save] model -> {SAVE_DIR}/cell_emulator_v2.pt")
