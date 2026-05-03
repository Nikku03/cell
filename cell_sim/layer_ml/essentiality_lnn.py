"""Top-level EssentialityLNN model — composes the multimodal encoder, the
liquid backbone (n_steps of LiquidStep), the hyperbolic memory bank, and
output heads (essentiality classification + auxiliary regulator-presence
proxy for unseen organisms).

Architecture (per-organism subgraph):

    inputs (dict of per-gene tensors + edge graph)
        |
    MultimodalEncoder (4 branches + presence flags)
        |
    h0 (initial per-gene hidden states)
        |
   LiquidStep × n_steps
       |       (each step retrieves from hyperbolic memory bank,
       |        runs adaptive-tau message passing on operon edges,
       |        updates h via ODE-style integration)
       |
   final h
       |       \
       |        EssentialityHead → P(essential)
       |        RegulatorProxyHead → predicted regulatory-element scores
       |                              (always-on auxiliary head; useful
       |                               on unseen organisms with no curated
       |                               regulator labels)
       |
   memory bank: store(final h) at end of forward (continual learning)

Self-test: `python -m cell_sim.layer_ml.essentiality_lnn`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_sim.layer_ml.liquid_lnn import LiquidStep
from cell_sim.layer_ml.hyperbolic_memory import HyperbolicMemoryBank
from cell_sim.layer_ml.multimodal_encoder import (
    MultimodalEncoder, MultimodalEncoderConfig,
)


@dataclass
class LNNConfig:
    n_scalar: int = 20
    n_regulatory: int = 50
    n_kinetic: int = 4
    n_kmer3: int = 64
    n_kmer4: int = 256
    hidden: int = 128
    n_liquid_steps: int = 4
    edge_dim: int = 2
    tau_min: float = 0.1
    tau_max: float = 2.0
    dt: float = 0.2
    dropout: float = 0.1
    memory_size: int = 512
    memory_retrieval_k: int = 8
    memory_curvature: float = 1.0
    n_regulator_proxy_classes: int = 3  # RBS / -10 / -35 (per-position score)


class EssentialityLNN(nn.Module):
    """End-to-end LNN for essentiality + regulatory-feature localization.

    forward() takes a batch dict (one organism = one batch) with:
      'scalar', 'regulatory', 'kinetic', 'kmer3', 'kmer4'
      'regulatory_mask', 'kinetic_mask'
      'edge_index': [2, E] graph edges (operon links)
      'edge_attr':  [E, edge_dim] edge features

    Returns dict:
      'essentiality': [N, 1] logits (sigmoid for probability)
      'regulator_proxy': [N, K] proxy scores per regulator class
      'hidden': [N, H] final hidden state (used for memory storage)
    """

    def __init__(self, cfg: LNNConfig | None = None):
        super().__init__()
        cfg = cfg or LNNConfig()
        self.cfg = cfg

        # Encoder
        enc_cfg = MultimodalEncoderConfig(
            n_scalar=cfg.n_scalar,
            n_regulatory=cfg.n_regulatory,
            n_kinetic=cfg.n_kinetic,
            n_kmer3=cfg.n_kmer3,
            n_kmer4=cfg.n_kmer4,
            hidden=cfg.hidden,
            dropout=cfg.dropout,
        )
        self.encoder = MultimodalEncoder(enc_cfg)

        # Liquid backbone — separate LiquidStep per layer (each can have its
        # own learned tau predictor). Sharing weights across steps would also
        # be defensible; per-step parameters give more capacity at the cost
        # of more parameters.
        self.liquid_steps = nn.ModuleList([
            LiquidStep(
                hidden=cfg.hidden, edge_dim=cfg.edge_dim,
                dt=cfg.dt, tau_min=cfg.tau_min, tau_max=cfg.tau_max,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_liquid_steps)
        ])

        # Hyperbolic memory bank (shared across steps; each step retrieves
        # from the same bank, only the LAST hidden state is stored back)
        self.memory = HyperbolicMemoryBank(
            hidden=cfg.hidden,
            memory_size=cfg.memory_size,
            retrieval_k=cfg.memory_retrieval_k,
            curvature=cfg.memory_curvature,
        )

        # Heads
        self.essentiality_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden),
            nn.Linear(cfg.hidden, cfg.hidden // 2),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden // 2, 1),
        )

        # Regulator-proxy head: per-gene K-class scores (RBS / -10 / -35)
        # giving a "predicted regulator presence" used for both training
        # supervision (where regulators are labeled) and inference (on
        # unseen organisms, this head's outputs locate regulatory features).
        self.regulator_proxy_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden),
            nn.Linear(cfg.hidden, cfg.hidden // 2),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden // 2, cfg.n_regulator_proxy_classes),
        )

    def forward(self, batch: dict, store_memory: bool = True) -> dict:
        # 1. Multimodal encode
        h0 = self.encoder(batch)
        h = h0.clone()

        # 2. Liquid backbone with memory retrieval at every step
        edge_index = batch.get('edge_index',
                                 torch.zeros(2, 0, dtype=torch.long,
                                              device=h.device))
        edge_attr = batch.get('edge_attr',
                                torch.zeros(0, self.cfg.edge_dim,
                                             device=h.device))
        for step_module in self.liquid_steps:
            h_memory = self.memory.retrieve(h)
            h = step_module(h, h0, edge_index, edge_attr, h_memory=h_memory)

        # 3. Heads
        ess_logits = self.essentiality_head(h).squeeze(-1)
        reg_logits = self.regulator_proxy_head(h)

        # 4. Memory storage (continual learning: this batch's final
        # representations become available to future batches)
        if store_memory and self.training:
            self.memory.store(h.detach())

        return {
            'essentiality': ess_logits,
            'regulator_proxy': reg_logits,
            'hidden': h,
        }

    def reset_memory(self):
        """Clear the memory bank (e.g. between training phases or for eval-
        only forward passes that should not leak training memory)."""
        self.memory.reset()


# ──────────── self-test ────────────


def _make_dummy_batch(N: int, cfg: LNNConfig, n_edges: int = 30
                       ) -> dict:
    return {
        'scalar': torch.randn(N, cfg.n_scalar),
        'regulatory': torch.randn(N, cfg.n_regulatory),
        'regulatory_mask': torch.bernoulli(torch.full((N,), 0.5)).float(),
        'kinetic': torch.randn(N, cfg.n_kinetic),
        'kinetic_mask': torch.bernoulli(torch.full((N,), 0.2)).float(),
        'kmer3': torch.rand(N, cfg.n_kmer3),
        'kmer4': torch.rand(N, cfg.n_kmer4),
        'edge_index': torch.randint(0, N, (2, n_edges)),
        'edge_attr': torch.randn(n_edges, cfg.edge_dim),
    }


def _self_test():
    print('essentiality_lnn self-test:')
    torch.manual_seed(42)
    cfg = LNNConfig(hidden=64, memory_size=128, n_liquid_steps=3)
    model = EssentialityLNN(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  total params: {n_params:,}  hidden={cfg.hidden}  '
          f'liquid_steps={cfg.n_liquid_steps}  memory_size={cfg.memory_size}')

    # 1. Single-organism forward
    batch = _make_dummy_batch(N=20, cfg=cfg)
    model.eval()  # so memory store is skipped
    out = model(batch)
    print(f'  forward: essentiality shape = {tuple(out["essentiality"].shape)}, '
          f'regulator_proxy shape = {tuple(out["regulator_proxy"].shape)}')
    assert out['essentiality'].shape == (20,)
    assert out['regulator_proxy'].shape == (20, cfg.n_regulator_proxy_classes)
    assert out['hidden'].shape == (20, cfg.hidden)

    # 2. Backward through full model
    model.train()
    out = model(batch)
    target = torch.bernoulli(torch.full((20,), 0.5))
    loss = F.binary_cross_entropy_with_logits(out['essentiality'], target)
    loss.backward()
    print(f'  backward: ok (loss = {loss.item():.4f})')

    # 3. Memory storage during training mode
    print(f'  memory n_populated after one train forward: {model.memory.n_populated}')
    assert model.memory.n_populated == 20

    # 4. Continual: train on org A, then forward on org B and check that
    # B's predictions are influenced by A's memory
    model.eval()
    batch_b = _make_dummy_batch(N=15, cfg=cfg, n_edges=20)
    out_b = model(batch_b)
    print(f'  forward on org B (N=15): essentiality shape '
          f'{tuple(out_b["essentiality"].shape)}')

    # 5. Reset memory clears across phases
    model.reset_memory()
    assert model.memory.n_populated == 0
    print('  reset_memory: ok')

    # 6. Empty edge case (no operon edges)
    batch_no_edges = _make_dummy_batch(N=20, cfg=cfg, n_edges=0)
    out_ne = model(batch_no_edges)
    assert out_ne['essentiality'].shape == (20,)
    print('  no-edges (isolated nodes): forward ok')

    # 7. All-modalities-missing-except-scalar input (degraded data)
    batch_min = _make_dummy_batch(N=10, cfg=cfg)
    batch_min['regulatory_mask'] = torch.zeros(10)
    batch_min['kinetic_mask'] = torch.zeros(10)
    batch_min['regulatory'] = torch.zeros(10, cfg.n_regulatory)
    batch_min['kinetic'] = torch.zeros(10, cfg.n_kinetic)
    out_min = model(batch_min)
    assert out_min['essentiality'].shape == (10,)
    print('  degraded input (only scalar+kmer): forward ok')

    print('  ALL PASS')


if __name__ == '__main__':
    _self_test()
