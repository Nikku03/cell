"""MultimodalSparseConceptLNN — three branches share one K-pattern bottleneck.

Branches:
  - STATIC: sequence + scalar gene features (existing EssentialityLNN encoder)
  - TRAJECTORY: per-step v15 KO trajectory (metabolites + rule events)
  - KNOWLEDGE: ortholog prior, PPI degree, conservation indicators

The K=128 pattern dictionary is *shared* across all three branches. Cross-
modal alignment losses force the same pattern to encode the same concept
regardless of which modality the gene is observed through. This is the
"connecting" the architecture is designed for: the model learns a unified
concept space that explains a gene from every angle simultaneously.

Pattern interpretation (after training):
  - For each pattern k, find top-activating genes from each modality
    independently. Cross-reference: pattern k = "translation core" if
    its static-top genes are tRNA synthetases AND its traj-top genes
    show ribosome stall + protein synthesis collapse AND its knowledge-top
    genes have ortholog_prior > 0.85.
  - Patterns that disagree across modalities are flagged as "modality-
    specific" — those are the genuinely informative findings.

See `cell_sim/layer_ml/multimodal_atlas.py` for atlas + connection graph
utilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_sim.layer_ml.essentiality_lnn import EssentialityLNN, LNNConfig


@dataclass
class MultimodalConfig:
    base_cfg: Optional[LNNConfig] = None
    n_patterns: int = 128
    top_k: int = 5
    hidden: int = 256
    # Trajectory branch
    traj_input_dim: int = 960     # 500 metabolites + 460 rule events
    traj_n_steps: int = 21        # t_end=5s, dt=0.25s
    traj_lstm_hidden: int = 128
    # Knowledge branch
    know_input_dim: int = 6       # ortholog_prior, n_orth, ppi_max, ppi_high,
                                    # is_conserved, is_hub
    # Loss weights
    w_main_bce: float = 1.0
    w_per_step_bce: float = 0.3
    w_align_static_traj: float = 0.5
    w_align_static_know: float = 0.3
    w_diversity: float = 0.1
    w_reconstruction: float = 0.1
    # Pattern-selection sharpness
    selection_temperature: float = 5.0


class TrajectoryEncoder(nn.Module):
    """LSTM over per-step state. Returns h_t at every step."""
    def __init__(self, in_dim: int, lstm_hidden: int, out_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, lstm_hidden)
        self.lstm = nn.LSTM(lstm_hidden, lstm_hidden, batch_first=True,
                             num_layers=2, dropout=0.1)
        self.out_proj = nn.Linear(lstm_hidden, out_dim)

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        """traj: [N, T, D] → [N, T, out_dim]"""
        x = F.silu(self.input_proj(traj))
        h_seq, _ = self.lstm(x)
        return self.out_proj(h_seq)


class KnowledgeEncoder(nn.Module):
    """Tiny MLP over knowledge features."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, know: torch.Tensor) -> torch.Tensor:
        return self.net(know)


class TrajectoryReconstructor(nn.Module):
    """Predict next trajectory step from h_t. Self-supervised physics signal."""
    def __init__(self, hidden: int, traj_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, traj_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


class MultimodalSparseConceptLNN(nn.Module):
    """Three-branch model with shared K-pattern bottleneck.

    forward(batch) returns dict with:
      essentiality:       [N] final logits
      per_step_logits:    [N, T] per-step trajectory predictions
      static_concepts:    [N, K] activation per pattern from static branch
      traj_concepts:      [N, K] activation per pattern from traj branch
                                   (mean over time)
      know_concepts:      [N, K] activation per pattern from knowledge branch
      next_step_pred:     [N, T-1, D_traj] reconstruction of trajectory
      diversity_loss:     scalar
      align_static_traj:  scalar
      align_static_know:  scalar
    """

    def __init__(self, cfg: MultimodalConfig | None = None):
        super().__init__()
        cfg = cfg or MultimodalConfig()
        if cfg.base_cfg is None:
            cfg.base_cfg = LNNConfig(hidden=cfg.hidden)
        else:
            cfg.base_cfg.hidden = cfg.hidden
        self.cfg = cfg

        # Static branch: re-use EssentialityLNN's encoder + liquid backbone
        self.static_lnn = EssentialityLNN(cfg.base_cfg)
        H = cfg.hidden

        # Trajectory branch
        self.traj_encoder = TrajectoryEncoder(
            in_dim=cfg.traj_input_dim,
            lstm_hidden=cfg.traj_lstm_hidden,
            out_dim=H,
        )
        # Trajectory autoregressive next-step head (self-supervised physics)
        self.traj_reconstructor = TrajectoryReconstructor(H, cfg.traj_input_dim)

        # Knowledge branch
        self.know_encoder = KnowledgeEncoder(cfg.know_input_dim, H)

        # SHARED pattern dictionary
        self.patterns = nn.Parameter(
            torch.randn(cfg.n_patterns, H) * (0.5 / H ** 0.5))

        # Final essentiality head: takes concatenated 3-modality g_repr
        self.essentiality_head = nn.Sequential(
            nn.LayerNorm(H * 3),
            nn.Linear(H * 3, H),
            nn.SiLU(),
            nn.Dropout(cfg.base_cfg.dropout),
            nn.Linear(H, 1),
        )
        # Per-step head: takes traj h_t alone (predicts essentiality from
        # partial trajectory at every timestep)
        self.per_step_head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H // 2),
            nn.SiLU(),
            nn.Linear(H // 2, 1),
        )

    # ------------------------------------------------------------------
    def _sparse_select(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                          torch.Tensor]:
        """Cosine-sim against patterns, top-k sparse softmax.
        Returns (concept_scores [N, K], g_repr [N, H], pattern_idx [N, top_k]).
        """
        patterns_norm = F.normalize(self.patterns, dim=-1)
        h_norm = F.normalize(h, dim=-1)
        scores = h_norm @ patterns_norm.T                       # [N, K]
        top_vals, top_idx = scores.topk(self.cfg.top_k, dim=-1)
        weights = F.softmax(top_vals * self.cfg.selection_temperature, dim=-1)
        gathered = self.patterns[top_idx]                       # [N, top_k, H]
        g_repr = (weights.unsqueeze(-1) * gathered).sum(dim=1)  # [N, H]
        return scores, g_repr, top_idx

    # ------------------------------------------------------------------
    def forward(self, batch: dict, traj: Optional[torch.Tensor] = None,
                  know: Optional[torch.Tensor] = None,
                  has_traj: Optional[torch.Tensor] = None,
                  store_memory: bool = True) -> dict:
        """
        batch:    standard EssentialityLNN batch (encoder + liquid backbone)
        traj:     [N, T, D_traj] or None (zeros for no traj signal)
        know:     [N, D_know] or None (zeros for no knowledge signal)
        has_traj: [N] boolean — gates whether the trajectory branch contributes
        """
        N = batch['scalar'].shape[0]
        device = batch['scalar'].device

        # ── STATIC branch ──
        h0 = self.static_lnn.encoder(batch)
        h_static = h0.clone()
        edge_index = batch.get('edge_index',
                                 torch.zeros(2, 0, dtype=torch.long, device=device))
        edge_attr = batch.get('edge_attr',
                                torch.zeros(0, self.static_lnn.cfg.edge_dim,
                                             device=device))
        for step in self.static_lnn.liquid_steps:
            if self.static_lnn.memory is not None:
                h_mem = self.static_lnn.memory.retrieve(h_static)
            else:
                h_mem = torch.zeros_like(h_static)
            h_static = step(h_static, h0, edge_index, edge_attr, h_memory=h_mem)
        static_scores, static_g, _ = self._sparse_select(h_static)

        # ── TRAJECTORY branch ──
        if traj is None:
            traj = torch.zeros(N, self.cfg.traj_n_steps,
                                 self.cfg.traj_input_dim, device=device)
        if has_traj is None:
            has_traj = (traj.abs().sum(dim=(1, 2)) > 1e-6).float()
        h_traj_seq = self.traj_encoder(traj)             # [N, T, H]
        h_traj_mean = h_traj_seq.mean(dim=1)             # [N, H]
        # Zero out trajectory contribution for genes that have no traj signal
        h_traj_mean = h_traj_mean * has_traj.unsqueeze(-1)
        traj_scores, traj_g, _ = self._sparse_select(h_traj_mean)
        # Per-step essentiality logits (broadcast head over T)
        per_step_logits = self.per_step_head(h_traj_seq).squeeze(-1)  # [N, T]
        # Autoregressive reconstruction: predict next step from h_t
        next_step_pred = self.traj_reconstructor(h_traj_seq[:, :-1])  # [N, T-1, D]

        # ── KNOWLEDGE branch ──
        if know is None:
            know = torch.zeros(N, self.cfg.know_input_dim, device=device)
        h_know = self.know_encoder(know)
        know_scores, know_g, _ = self._sparse_select(h_know)

        # ── Combined essentiality ──
        combined = torch.cat([static_g, traj_g, know_g], dim=-1)  # [N, 3H]
        ess_logits = self.essentiality_head(combined).squeeze(-1)

        # ── Diversity loss on patterns ──
        K = self.cfg.n_patterns
        eye = torch.eye(K, device=self.patterns.device)
        patterns_norm = F.normalize(self.patterns, dim=-1)
        sim = patterns_norm @ patterns_norm.T - eye
        diversity_loss = (sim ** 2).mean()

        # ── Cross-modal alignment ──
        # Use cosine similarity between concept-activation vectors so the
        # alignment is scale-invariant. Only enforce alignment on genes that
        # have valid signal in both branches.
        align_st_mask = has_traj.bool()
        if align_st_mask.any():
            st = F.normalize(static_scores[align_st_mask], dim=-1)
            tt = F.normalize(traj_scores[align_st_mask], dim=-1)
            align_static_traj = 1.0 - (st * tt).sum(dim=-1).mean()
        else:
            align_static_traj = torch.zeros((), device=device)
        # Knowledge alignment is enforced for all genes (knowledge always exists)
        sk = F.normalize(static_scores, dim=-1)
        kk = F.normalize(know_scores, dim=-1)
        align_static_know = 1.0 - (sk * kk).sum(dim=-1).mean()

        # Memory bank store (optional)
        if (store_memory and self.training
              and self.static_lnn.memory is not None):
            self.static_lnn.memory.store(h_static.detach())

        return {
            'essentiality': ess_logits,
            'per_step_logits': per_step_logits,
            'static_concepts': static_scores,
            'traj_concepts': traj_scores,
            'know_concepts': know_scores,
            'static_g': static_g,
            'traj_g': traj_g,
            'know_g': know_g,
            'next_step_pred': next_step_pred,
            'diversity_loss': diversity_loss,
            'align_static_traj': align_static_traj,
            'align_static_know': align_static_know,
            'has_traj': has_traj,
        }

    # ------------------------------------------------------------------
    def compute_loss(self, out: dict, labels: torch.Tensor,
                       traj_target: Optional[torch.Tensor] = None) -> dict:
        cfg = self.cfg
        bce = F.binary_cross_entropy_with_logits

        # Main BCE
        L_main = bce(out['essentiality'], labels)

        # Per-step BCE — only for genes with traj signal
        has_traj = out['has_traj']
        if has_traj.sum() > 0:
            ps_logits = out['per_step_logits'][has_traj.bool()]
            ps_labels = labels[has_traj.bool()].unsqueeze(-1).expand_as(ps_logits)
            L_step = bce(ps_logits, ps_labels)
        else:
            L_step = torch.zeros((), device=labels.device)

        # Trajectory reconstruction (next-step prediction) — only on genes
        # with traj signal, only on real timesteps
        if traj_target is not None and has_traj.sum() > 0:
            pred = out['next_step_pred'][has_traj.bool()]      # [Nt, T-1, D]
            tgt = traj_target[has_traj.bool(), 1:]              # [Nt, T-1, D]
            L_recon = F.mse_loss(pred, tgt)
        else:
            L_recon = torch.zeros((), device=labels.device)

        L_div = out['diversity_loss']
        L_align_st = out['align_static_traj']
        L_align_sk = out['align_static_know']

        total = (cfg.w_main_bce * L_main
                  + cfg.w_per_step_bce * L_step
                  + cfg.w_align_static_traj * L_align_st
                  + cfg.w_align_static_know * L_align_sk
                  + cfg.w_diversity * L_div
                  + cfg.w_reconstruction * L_recon)
        return {
            'total': total,
            'main_bce': L_main, 'per_step_bce': L_step,
            'reconstruction': L_recon, 'diversity': L_div,
            'align_static_traj': L_align_st, 'align_static_know': L_align_sk,
        }


# ──────────────────────────────────────────────────────────────────────
def _self_test():
    """Synthetic forward+backward smoke test."""
    print('MultimodalSparseConceptLNN self-test:')
    torch.manual_seed(0)
    base = LNNConfig(hidden=64, n_liquid_steps=2, dropout=0.1,
                       use_sequence_encoder=True, seq_max_len=128,
                       seq_channels_per_kernel=8,
                       n_scalar=22, use_memory_bank=False)
    cfg = MultimodalConfig(base_cfg=base, hidden=64, n_patterns=32, top_k=3,
                              traj_input_dim=64, traj_n_steps=10,
                              traj_lstm_hidden=32, know_input_dim=4)
    model = MultimodalSparseConceptLNN(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  params: {n_params:,}')

    N = 8
    batch = {
        'scalar': torch.randn(N, 22),
        'regulatory': torch.randn(N, 50),
        'regulatory_mask': torch.ones(N),
        'kinetic': torch.zeros(N, 4), 'kinetic_mask': torch.zeros(N),
        'kmer3': torch.zeros(N, 64), 'kmer4': torch.zeros(N, 256),
        'seq_tokens': torch.randint(0, 4, (N, 128)),
        'seq_lengths': torch.full((N,), 128, dtype=torch.long),
        'edge_index': torch.zeros(2, 0, dtype=torch.long),
        'edge_attr': torch.zeros(0, 2),
        'labels': torch.randint(0, 2, (N,)).float(),
    }
    traj = torch.randn(N, 10, 64) * 0.1
    know = torch.randn(N, 4)
    has_traj = torch.tensor([1, 1, 0, 1, 0, 1, 1, 0]).float()

    out = model(batch, traj=traj, know=know, has_traj=has_traj)
    print(f'  forward ok:')
    print(f'    essentiality:    {tuple(out["essentiality"].shape)}')
    print(f'    static_concepts: {tuple(out["static_concepts"].shape)}')
    print(f'    traj_concepts:   {tuple(out["traj_concepts"].shape)}')
    print(f'    know_concepts:   {tuple(out["know_concepts"].shape)}')
    print(f'    per_step_logits: {tuple(out["per_step_logits"].shape)}')
    print(f'    diversity_loss:  {float(out["diversity_loss"]):.4f}')
    print(f'    align_st_traj:   {float(out["align_static_traj"]):.4f}')
    print(f'    align_st_know:   {float(out["align_static_know"]):.4f}')

    losses = model.compute_loss(out, batch['labels'].float(), traj_target=traj)
    losses['total'].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters()
                     if p.grad is not None)
    print(f'  backward ok: total_loss={losses["total"].item():.4f}  '
          f'grad_norm={grad_norm:.3f}')

    # Verify all 3 modalities affect different g_reprs
    diff_st = (out['static_g'] - out['traj_g']).abs().mean().item()
    diff_sk = (out['static_g'] - out['know_g']).abs().mean().item()
    assert diff_st > 1e-3 and diff_sk > 1e-3, \
        'modalities collapsed to identical representations'
    print(f'  modalities distinct: |st-tr|={diff_st:.3f}  |st-kn|={diff_sk:.3f}')
    print('  ALL PASS')


if __name__ == '__main__':
    _self_test()
