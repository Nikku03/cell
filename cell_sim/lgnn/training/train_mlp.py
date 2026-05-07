"""Single-step training loop for the MLP baseline.

Loss: MSE on Δsigned-log1p. AdamW. One pass over each train replicate
per epoch, val on held-out replicate(s). Per-species R² computed in
log-space-delta units, averaged over species.

Future model variants (graph, liquid, sparse-pattern) get their own
training modules in this directory rather than feature-flagging this
one. Trains on the dataset modules in cell_sim/lgnn/data/.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cell_sim.lgnn.data.dataset import (
    BufferedShuffleDataset, CountsTransitionDataset,
)
from cell_sim.lgnn.models.mlp_baseline import CountDynamicsMLP


@dataclass
class TrainConfig:
    n_species: int = 8572
    hidden: int = 1024
    n_blocks: int = 2
    dropout: float = 0.0
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-5
    train_replicates: tuple = tuple(range(1, 50))
    val_replicates: tuple = (50,)
    n_epochs: int = 1
    device: str = 'cpu'
    seed: int = 42
    log_every: int = 200
    species_filter: Optional[List[str]] = None
    # Cross-replicate buffered shuffler vs sequential per-replicate.
    # Default True since SGD on correlated batches is the v0 weakness.
    buffered_shuffle: bool = True
    buffer_size: int = 3


def _device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _build_train_dataset(cfg: TrainConfig, lsdata_module, epoch: int):
    if cfg.buffered_shuffle:
        return BufferedShuffleDataset(
            cfg.train_replicates, lsdata_module,
            species_filter=cfg.species_filter,
            buffer_size=cfg.buffer_size,
            seed=cfg.seed + epoch,
        )
    return CountsTransitionDataset(
        cfg.train_replicates, lsdata_module,
        species_filter=cfg.species_filter,
        shuffle_within_replicate=True,
        seed=cfg.seed + epoch,
    )


def train_count_dynamics(cfg: TrainConfig, lsdata_module,
                          checkpoint_path: Optional[Path] = None) -> dict:
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
        ds = _build_train_dataset(cfg, lsdata_module, epoch)
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
    """val MSE on Δlog1p + averaged per-species R²."""
    model.eval()
    ds = CountsTransitionDataset(
        cfg.val_replicates, lsdata_module,
        species_filter=cfg.species_filter,
        shuffle_within_replicate=False,
        seed=cfg.seed,
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0)
    sq_err = 0.0
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
    mean = sum_target / max(count, 1)
    var = sum_target_sq / max(count, 1) - mean * mean
    var = var.clamp(min=1e-12)
    species_r2 = 1.0 - (sum_resid_sq / max(count, 1)) / var
    r2 = species_r2.clamp(min=-1.0).mean().item()
    return {'mse': mse, 'r2': r2}
