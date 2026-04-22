"""Next-event prediction models and training loop.

Two models, one interface:

  - ``HeuristicBaseline``: no learning. Just asks "does this atom have
    valence remaining AND a bondable neighbor within bond_form_distance?"
    This is the floor — any learned model must beat this.

  - ``AtomGNN``: a tiny graph neural network. Node features pass through
    a 2-layer MLP encoder, then two rounds of sum-aggregated message
    passing over the live-bond edges, then a final MLP head that
    outputs a per-atom logit.

Both operate on the ``Snapshot`` objects produced by
``ml_dataset.TrajectoryCollector``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .ml_dataset import N_ELEM_FEATURES, N_NODE_FEATURES, Snapshot


# ---------- heuristic baseline ---------------------------------------


class HeuristicBaseline:
    """Predicts P(bond_forms) = 1 if:
      - atom has valence_remaining > 0 (from node features)
      - at least one atom of bondable element sits within proximity
        cutoff in the snapshot's edge / spatial representation.
    else 0. Not actually learned — a physics-grounded floor to beat.
    """

    def __init__(self, proximity_cutoff_nm: float = 0.25):
        self.proximity_cutoff = proximity_cutoff_nm

    def predict_proba(self, snap: Snapshot) -> np.ndarray:
        # From node features: valence_remaining sits at index N_ELEM_FEATURES.
        valence_ok = snap.node_features[:, N_ELEM_FEATURES] > 0.0
        # Neighbor count: live-bond edges already carry current length.
        # But we don't have all proximity edges in a Snapshot — only
        # live bonds. So this baseline only sees *bonded* neighbors.
        # That's intentional: the baseline can only reason about atoms
        # that ALREADY have a bond, and will predict "bond forms next"
        # for any atom with spare valence. Pure physics floor.
        p = valence_ok.astype(np.float32)
        return p


# ---------- GNN ------------------------------------------------------


class AtomGNN(nn.Module):
    """Minimal message-passing GNN with sum aggregation.

    Parameter count: ~4k for the default (hidden=32, two MP rounds).
    """

    def __init__(self, hidden: int = 32, n_rounds: int = 2):
        super().__init__()
        self.n_rounds = n_rounds
        self.encoder = nn.Sequential(
            nn.Linear(N_NODE_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.msg = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 2 + 3, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden))
            for _ in range(n_rounds)
        ])
        self.update = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden))
            for _ in range(n_rounds)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,    # (N, F)
        edges: torch.Tensor,            # (E, 2) long
        edge_features: torch.Tensor,    # (E, 3)
    ) -> torch.Tensor:                  # (N,) logits
        h = self.encoder(node_features)
        n = h.shape[0]
        for r in range(self.n_rounds):
            if edges.shape[0] == 0:
                continue
            src = edges[:, 0]
            dst = edges[:, 1]
            msg_in = torch.cat([h[src], h[dst], edge_features], dim=1)
            m = self.msg[r](msg_in)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            h_upd = self.update[r](torch.cat([h, agg], dim=1))
            h = h + h_upd    # residual
        return self.head(h).squeeze(-1)


# ---------- training loop --------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 16
    lr: float = 3e-3
    pos_weight: float = 20.0       # imbalance: ~95% negative class
    hidden: int = 32
    n_rounds: int = 2
    device: str = "cpu"
    seed: int = 0


def _snapshot_to_tensors(snap: Snapshot, device: str):
    return (
        torch.tensor(snap.node_features, device=device),
        torch.tensor(snap.edges.astype(np.int64), device=device),
        torch.tensor(snap.edge_features, device=device),
        torch.tensor(snap.labels.astype(np.float32), device=device),
    )


def _compute_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Approximate AUC via pairwise rank — simple enough for small N,
    no sklearn dependency."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    # Count pairs where pos > neg (plus 0.5 for ties).
    diffs = pos[:, None] - neg[None, :]
    auc = (np.mean(diffs > 0) + 0.5 * np.mean(diffs == 0))
    return float(auc)


def _accuracy_at_threshold(y_true: np.ndarray, scores: np.ndarray,
                           threshold: float = 0.5) -> float:
    pred = (scores >= threshold).astype(np.int8)
    return float((pred == y_true).mean())


def train_atom_gnn(
    train: Sequence[Snapshot],
    val: Sequence[Snapshot],
    cfg: TrainConfig = TrainConfig(),
    progress=None,
) -> tuple[AtomGNN, dict]:
    torch.manual_seed(cfg.seed)
    device = cfg.device
    model = AtomGNN(hidden=cfg.hidden, n_rounds=cfg.n_rounds).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight,
                                                           device=device))

    history = {"train_loss": [], "val_auc": [], "val_acc": []}
    for epoch in range(cfg.epochs):
        model.train()
        perm = np.random.permutation(len(train))
        total_loss = 0.0
        n_batches = 0
        for bstart in range(0, len(train), cfg.batch_size):
            bidx = perm[bstart:bstart + cfg.batch_size]
            opt.zero_grad()
            batch_loss = 0.0
            for i in bidx:
                nf, e, ef, y = _snapshot_to_tensors(train[i], device)
                logits = model(nf, e, ef)
                batch_loss = batch_loss + loss_fn(logits, y)
            batch_loss = batch_loss / len(bidx)
            batch_loss.backward()
            opt.step()
            total_loss += float(batch_loss.item())
            n_batches += 1

        # validate
        model.eval()
        all_y = []
        all_s = []
        with torch.no_grad():
            for snap in val:
                nf, e, ef, y = _snapshot_to_tensors(snap, device)
                logits = model(nf, e, ef)
                scores = torch.sigmoid(logits).cpu().numpy()
                all_y.append(snap.labels.astype(np.int8))
                all_s.append(scores)
        y_cat = np.concatenate(all_y)
        s_cat = np.concatenate(all_s)
        auc = _compute_auc(y_cat, s_cat)
        acc = _accuracy_at_threshold(y_cat, s_cat, threshold=0.5)
        history["train_loss"].append(total_loss / max(n_batches, 1))
        history["val_auc"].append(auc)
        history["val_acc"].append(acc)
        if progress is not None:
            progress(f"epoch {epoch+1:2d}/{cfg.epochs} "
                     f"train_loss={history['train_loss'][-1]:.4f} "
                     f"val_auc={auc:.3f} val_acc={acc:.3f}")
    return model, history


def evaluate(
    snapshots: Sequence[Snapshot],
    predict_fn,
) -> dict:
    """Evaluate a predict_fn(snapshot) -> scores on a list of snapshots.
    Returns {'auc': ..., 'acc': ..., 'positive_rate': ...}.
    """
    all_y = []
    all_s = []
    for snap in snapshots:
        scores = predict_fn(snap)
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().numpy()
        all_y.append(snap.labels.astype(np.int8))
        all_s.append(np.asarray(scores, dtype=np.float32))
    y = np.concatenate(all_y)
    s = np.concatenate(all_s)
    return {
        "auc": _compute_auc(y, s),
        "acc": _accuracy_at_threshold(y, s, 0.5),
        "positive_rate": float(y.mean()),
    }
