"""Pairwise protein-protein interaction (PPI) predictor.

Reuses MultimodalEncoder (with optional sequence encoder + multi-head self-
attention) as a siamese gene encoder, then cross-attends between the two
proteins in each pair and produces a single binding logit.

Input format: a "pair batch" is two index tensors `idx_a`, `idx_b` (long,
shape [P]) plus a regular per-organism gene batch dict. Each pair p is
(gene at idx_a[p], gene at idx_b[p]). All P pairs share the underlying
gene batch — we encode every gene once and gather pair representations
afterward, which is much cheaper than encoding the pair list directly.

Output: dict with
  'logits'     : [P]  pre-sigmoid binding score
  'pair_repr'  : [P, hidden] pooled pair representation (for downstream tasks)
  'gene_repr'  : [N, hidden] per-gene embeddings (so callers can cache)

Cross-attention (in the pair head) is symmetric — gene A attends to gene B
and vice versa, the two attended representations are concatenated with the
elementwise product and absolute difference (a standard interaction-aware
combining trick). This makes the model order-insensitive: predict(i, j) ==
predict(j, i) up to numerical noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_sim.layer_ml.multimodal_encoder import (
    MultimodalEncoder, MultimodalEncoderConfig,
)


@dataclass
class PPIPredictorConfig:
    hidden: int = 128
    dropout: float = 0.1
    cross_attn_heads: int = 8
    cross_attn_layers: int = 2
    head_hidden: int = 256

    # Encoder config (will be coerced to use multi-head self-attention by
    # default since high-dim pattern analysis is the whole point).
    encoder_cfg: Optional[MultimodalEncoderConfig] = None


class _CrossAttentionBlock(nn.Module):
    """Two streams (A, B) cross-attend to each other for one round, with
    LayerNorm + FFN residual on each side. After K rounds the streams have
    exchanged information about every position in the other stream."""

    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        # Shared LayerNorm + cross-attention + FFN on both sides → swapping
        # (a, b) gives (a', b') -> (b'_swapped, a'_swapped), so symmetrised
        # pair features (sum, |diff|, prod) downstream are exactly invariant.
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout,
                                            batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        # a, b are [P, 1, H] — one "token" per pair. With shared weights,
        # swapping (a, b) at the input yields (b', a') at the output.
        a_n = self.norm1(a); b_n = self.norm1(b)
        a_attn, _ = self.attn(a_n, b_n, b_n, need_weights=False)
        b_attn, _ = self.attn(b_n, a_n, a_n, need_weights=False)
        a = a + a_attn
        b = b + b_attn
        a = a + self.ffn(self.norm2(a))
        b = b + self.ffn(self.norm2(b))
        return a, b


class PPIPredictor(nn.Module):
    """Siamese encoder + symmetric cross-attention pair head.

    Forward signature
    -----------------
    out = model(gene_batch, idx_a, idx_b)
      gene_batch : the standard MultimodalEncoder batch dict (one organism)
      idx_a, idx_b : LongTensor[P] indexing into the N genes in gene_batch

    Returns dict {'logits': [P], 'pair_repr': [P, H], 'gene_repr': [N, H]}.
    """

    def __init__(self, cfg: PPIPredictorConfig | None = None):
        super().__init__()
        cfg = cfg or PPIPredictorConfig()
        self.cfg = cfg

        enc_cfg = cfg.encoder_cfg or MultimodalEncoderConfig(
            hidden=cfg.hidden, dropout=cfg.dropout,
            use_sequence_encoder=True, seq_max_len=2048,
            use_multi_head_attention=True,
            attention_heads=cfg.cross_attn_heads,
            attention_layers=2,
        )
        # Force hidden dim to match
        enc_cfg.hidden = cfg.hidden
        self.encoder = MultimodalEncoder(enc_cfg)

        self.cross_blocks = nn.ModuleList([
            _CrossAttentionBlock(cfg.hidden, cfg.cross_attn_heads, cfg.dropout)
            for _ in range(cfg.cross_attn_layers)
        ])

        # Pair head: concat (a, b, |a-b|, a*b) -> head_hidden -> logit
        self.pair_head = nn.Sequential(
            nn.Linear(4 * cfg.hidden, cfg.head_hidden),
            nn.LayerNorm(cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.head_hidden // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden // 2, 1),
        )

    def encode_genes(self, gene_batch: dict) -> torch.Tensor:
        """Encode every gene in the batch once (shape [N, H])."""
        return self.encoder(gene_batch)

    def forward(self, gene_batch: dict,
                idx_a: torch.Tensor, idx_b: torch.Tensor,
                gene_repr: Optional[torch.Tensor] = None) -> dict:
        if gene_repr is None:
            gene_repr = self.encode_genes(gene_batch)
        a = gene_repr[idx_a].unsqueeze(1)   # [P, 1, H]
        b = gene_repr[idx_b].unsqueeze(1)   # [P, 1, H]
        for blk in self.cross_blocks:
            a, b = blk(a, b)
        a = a.squeeze(1); b = b.squeeze(1)  # [P, H]

        # Symmetric pairwise feature: order-invariant under swap(a, b).
        # |a-b| is symmetric; a*b is symmetric; (a+b) preserves total signal.
        # We give the head the symmetrised version so the predictor cannot
        # cheat by learning order-of-arguments shortcuts.
        sym_sum = a + b
        sym_diff = (a - b).abs()
        sym_prod = a * b
        # Plus the raw concat of one ordering (still informative — only the
        # relative direction is symmetrised away). Keeping a single half of
        # the concat preserves count = 4*H.
        pair_features = torch.cat([sym_sum, sym_diff, sym_prod, a + b - 2 * (a * b)],
                                    dim=-1)
        logits = self.pair_head(pair_features).squeeze(-1)
        return {
            'logits': logits,
            'pair_repr': pair_features,
            'gene_repr': gene_repr,
        }


# ──────────── self-test ────────────


def _self_test():
    print('ppi_predictor self-test:')
    torch.manual_seed(42)
    N = 50          # 50 genes in a "test organism"
    P = 200         # 200 pairs
    H = 64

    enc_cfg = MultimodalEncoderConfig(
        hidden=H, use_sequence_encoder=True, seq_max_len=128,
        use_multi_head_attention=True, attention_heads=4, attention_layers=1,
    )
    cfg = PPIPredictorConfig(hidden=H, encoder_cfg=enc_cfg,
                              cross_attn_heads=4, cross_attn_layers=2,
                              head_hidden=128)
    model = PPIPredictor(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  params: {n_params:,}  hidden={H}')

    # Build a dummy gene batch
    gene_batch = {
        'scalar': torch.randn(N, enc_cfg.n_scalar),
        'regulatory': torch.randn(N, enc_cfg.n_regulatory),
        'regulatory_mask': torch.ones(N),
        'kinetic': torch.zeros(N, enc_cfg.n_kinetic),
        'kinetic_mask': torch.zeros(N),
        'seq_tokens': torch.randint(0, 4, (N, enc_cfg.seq_max_len)),
        'seq_lengths': torch.full((N,), enc_cfg.seq_max_len, dtype=torch.long),
        'kmer3': torch.zeros(N, enc_cfg.n_kmer3),
        'kmer4': torch.zeros(N, enc_cfg.n_kmer4),
    }
    idx_a = torch.randint(0, N, (P,))
    idx_b = torch.randint(0, N, (P,))

    out = model(gene_batch, idx_a, idx_b)
    print(f'  forward: logits shape={tuple(out["logits"].shape)}, '
          f'mean={out["logits"].mean().item():.3f}, '
          f'std={out["logits"].std().item():.3f}')
    assert out['logits'].shape == (P,)
    assert out['gene_repr'].shape == (N, H)
    assert out['pair_repr'].shape == (P, 4 * H)

    # Backward (in train mode, with dropout active)
    loss = F.binary_cross_entropy_with_logits(
        out['logits'], torch.randint(0, 2, (P,)).float())
    loss.backward()
    print(f'  backward: ok  (loss={loss.item():.3f})')

    # Symmetry check: predict(i, j) == predict(j, i) (eval mode — no dropout)
    model.eval()
    with torch.no_grad():
        out_eval = model(gene_batch, idx_a, idx_b)
        out_swap = model(gene_batch, idx_b, idx_a, gene_repr=out_eval['gene_repr'])
    delta = (out_eval['logits'] - out_swap['logits']).abs().max().item()
    print(f'  symmetry check (swap idx, eval): max |delta| = {delta:.4g}  '
          f'(should be ~0 by construction)')
    assert delta < 1e-4, f'pair head is not order-invariant: {delta}'

    # Cached encoding path: encode once + reuse, vs encode-and-predict
    # (both in eval mode, otherwise dropout makes them differ).
    with torch.no_grad():
        cached = model.encode_genes(gene_batch)
        out_c = model(gene_batch, idx_a, idx_b, gene_repr=cached)
        diff = (out_eval['logits'] - out_c['logits']).abs().max().item()
        print(f'  cached gene_repr path matches fresh eval forward: '
              f'max |delta|={diff:.4g}')
        assert diff < 1e-4

    print('  ALL PASS')


if __name__ == '__main__':
    _self_test()
