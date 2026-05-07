"""Whole-cell count-dynamics surrogate.

Trains a feed-forward predictor on the Luthey-Schulten Minimal Cell drop:
given a species-count vector x_t at time t (8572-dim), predict the
delta x_{t+1} - x_t one second later. The Drive data is 50 stochastic
replicates × 7201 timepoints, so we get ~360k transitions to fit.

Why not a GNN? The natural graph here is the SBML stoichiometric matrix
(species ↔ reactions). Adding it is a v2; v1 is an MLP baseline so we
have something to compare against. The loss landscape and choice of
target normalisation matter more than architecture for a regression of
this kind, so we lock those down first.

Target normalisation: predict Δ log1p(x). Counts vary from 0
(low-abundance species) to ~60k (chromosome). log1p maps that to roughly
[0, 11], and the delta is well-scaled regardless of absolute count. At
inference time, `predict_step` lifts back to integer counts via expm1
+ rounding (clamped at 0).

Loss: MSE on Δ log1p(x). The conditional mean of x_{t+1} given x_t is
the natural deterministic surrogate for an inherently stochastic
process; capturing variance (e.g. heteroscedastic NLL) is a v3.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def replicate_to_log1p_array(df) -> np.ndarray:
    """Take a (n_species, n_timepoints) DataFrame from
    `lsdata.load_replicate` and return an ndarray of shape
    (n_timepoints, n_species) in float32, log1p-transformed.

    Always copies — the in-place log1p below would otherwise alias back
    into the DataFrame and double-transform on the next call."""
    arr = np.ascontiguousarray(df.to_numpy(dtype=np.float32, copy=True).T)
    np.log1p(arr, out=arr)
    return arr


class CountsTransitionDataset(IterableDataset):
    """Streams (x_t, dx) pairs from one or more replicates.

    The dataset materialises one replicate's array at a time (~250 MB
    float32) and walks it sequentially or shuffled. After the replicate
    is exhausted it loads the next. Designed for Colab where 12-15 GB
    RAM rules out loading all 50 at once.
    """

    def __init__(self, replicate_indices: Sequence[int],
                 lsdata_module,
                 species_filter: Optional[Sequence[str]] = None,
                 shuffle_within_replicate: bool = True,
                 seed: int = 0):
        super().__init__()
        self.replicate_indices = list(replicate_indices)
        self.lsdata = lsdata_module
        self.species_filter = (list(species_filter)
                                if species_filter is not None else None)
        self.shuffle = shuffle_within_replicate
        self.seed = seed

    def __iter__(self) -> Iterator:
        rng = np.random.default_rng(self.seed)
        for rep_i in self.replicate_indices:
            df = self.lsdata.load_replicate(rep_i, species=self.species_filter)
            arr = replicate_to_log1p_array(df)             # (T, S) float32
            n_pairs = arr.shape[0] - 1
            order = np.arange(n_pairs)
            if self.shuffle:
                rng.shuffle(order)
            for j in order:
                x_t = arr[j]
                dx  = arr[j + 1] - arr[j]
                yield (torch.from_numpy(x_t), torch.from_numpy(dx))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CountDynamicsMLP(nn.Module):
    """Plain feed-forward delta predictor with one residual block.

    Input: log1p-counts vector (S,).
    Output: predicted delta log1p-counts (S,).
    """

    def __init__(self, n_species: int, hidden: int = 1024,
                 n_blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        self.n_species = n_species
        self.input_proj = nn.Linear(n_species, hidden)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
            ))
        self.out_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, n_species)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(self.out_norm(h))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    n_species: int = 8572
    hidden: int = 1024
    n_blocks: int = 2
    dropout: float = 0.0
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-5
    train_replicates: tuple = tuple(range(1, 50))   # 1..49 inclusive
    val_replicates: tuple = (50,)
    n_epochs: int = 1
    device: str = 'cpu'
    seed: int = 42
    log_every: int = 200
    species_filter: Optional[List[str]] = None


def _device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def train_count_dynamics(cfg: TrainConfig, lsdata_module,
                          checkpoint_path: Optional[Path] = None) -> dict:
    """Train and return a metrics dict.

    `lsdata_module` is the lsdata.py module (passed in so we can swap a
    fake one in tests). `checkpoint_path`, if set, receives the best-val
    state_dict.
    """
    device = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    model = CountDynamicsMLP(
        n_species=cfg.n_species,
        hidden=cfg.hidden,
        n_blocks=cfg.n_blocks,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    best_val = float('inf')

    for epoch in range(cfg.n_epochs):
        # Train
        ds = CountsTransitionDataset(
            cfg.train_replicates, lsdata_module,
            species_filter=cfg.species_filter,
            shuffle_within_replicate=True,
            seed=cfg.seed + epoch,
        )
        loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0)
        model.train()
        running = 0.0
        n_batches = 0
        t0 = time.time()
        for step, (x, dx) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            dx = dx.to(device, non_blocking=True)
            pred = model(x)
            loss = loss_fn(pred, dx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
            if (step + 1) % cfg.log_every == 0:
                avg = running / n_batches
                wall = time.time() - t0
                print(f'  epoch {epoch} step {step+1:>6d}  '
                      f'train_mse={avg:.4f}  '
                      f'{(step+1)*cfg.batch_size/wall:.0f} samples/s')
        history['train_loss'].append(running / max(n_batches, 1))

        # Val
        val_metrics = evaluate(model, cfg, lsdata_module, device)
        history['val_loss'].append(val_metrics['mse'])
        history['val_r2'].append(val_metrics['r2'])
        print(f'  epoch {epoch}  val_mse={val_metrics["mse"]:.4f}  '
              f'val_r2={val_metrics["r2"]:.4f}')
        if val_metrics['mse'] < best_val:
            best_val = val_metrics['mse']
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'state_dict': model.state_dict(),
                    'cfg': cfg.__dict__,
                    'epoch': epoch,
                    'val_mse': val_metrics['mse'],
                    'val_r2': val_metrics['r2'],
                }, checkpoint_path)

    return {
        'history': history,
        'best_val_mse': best_val,
        'final_model': model,
    }


@torch.no_grad()
def evaluate(model: nn.Module, cfg: TrainConfig, lsdata_module,
             device: torch.device) -> dict:
    """Compute val MSE on Δlog1p and a coefficient-of-determination R²."""
    model.eval()
    ds = CountsTransitionDataset(
        cfg.val_replicates, lsdata_module,
        species_filter=cfg.species_filter,
        shuffle_within_replicate=False,
        seed=cfg.seed,
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0)
    sq_err = 0.0
    sq_total = 0.0
    n = 0
    sum_target = torch.zeros(cfg.n_species, device=device)
    sum_target_sq = torch.zeros(cfg.n_species, device=device)
    sum_resid_sq = torch.zeros(cfg.n_species, device=device)
    count = 0
    for x, dx in loader:
        x = x.to(device, non_blocking=True)
        dx = dx.to(device, non_blocking=True)
        pred = model(x)
        diff = pred - dx
        sq_err += (diff * diff).sum().item()
        sum_target += dx.sum(dim=0)
        sum_target_sq += (dx * dx).sum(dim=0)
        sum_resid_sq += (diff * diff).sum(dim=0)
        count += dx.shape[0]
        n += dx.numel()
    mse = sq_err / max(n, 1)
    # Per-species R² then averaged. Variance computed from second moments.
    mean = sum_target / max(count, 1)
    var = sum_target_sq / max(count, 1) - mean * mean
    var = var.clamp(min=1e-12)
    species_r2 = 1.0 - (sum_resid_sq / max(count, 1)) / var
    r2 = species_r2.clamp(min=-1.0).mean().item()
    return {'mse': mse, 'r2': r2}


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_step(model: nn.Module, x_count: np.ndarray,
                  device: Optional[torch.device] = None) -> np.ndarray:
    """One forward step: integer counts -> next-second integer counts."""
    if device is None:
        device = next(model.parameters()).device
    x_log = np.log1p(x_count.astype(np.float32))
    t = torch.from_numpy(x_log).to(device).unsqueeze(0)
    dx = model(t).squeeze(0).cpu().numpy()
    next_log = x_log + dx
    next_count = np.expm1(next_log)
    return np.clip(np.round(next_count), 0, None).astype(np.int64)
