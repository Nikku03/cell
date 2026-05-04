"""SparseConceptLNN — essentiality LNN with a sparse pattern bottleneck.

Forces the model to express each gene as a sparse combination of K
learnable "pattern" vectors, then predicts essentiality from the pattern
combination. The K patterns ARE the model's interpretable rule set.

Architecture
============

    encoder + liquid backbone -> hidden h  [N, H]
                                  ↓
    cosine-similarity to K patterns:  scores = h_norm @ patterns_norm.T  [N, K]
                                  ↓
    top-k sparse selection:  keep only top-k patterns per gene
                                  ↓
    pattern combination:  g_repr = softmax(top_k_scores) @ patterns  [N, H]
                                  ↓
    essentiality head:  ess_logits = head(g_repr)  [N]

Why this fixes memorization
---------------------------

The bottleneck forces compression: 14k training genes must all flow
through K=64 patterns. The model PHYSICALLY cannot memorize each gene's
label — it has to find shared patterns across genes. Cross-org transfer
becomes natural because patterns describe biology (DNA replication is
essential everywhere) rather than gene identity.

The K patterns become inspectable: each can be labelled by examining
its top-activating training genes (find_top_genes_per_pattern below).

Diversity loss
--------------

Without explicit pressure, patterns can collapse to identical vectors.
We add a diversity loss:
    L_div = mean of (pattern_i · pattern_j)^2  over i != j
This pushes patterns toward orthogonal so they cover different biology.

Self-test verifies forward/backward + that diversity loss decreases
when patterns are made orthogonal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_sim.layer_ml.essentiality_lnn import EssentialityLNN, LNNConfig


@dataclass
class SparseConceptLNNConfig:
    base_cfg: Optional[LNNConfig] = None
    n_patterns: int = 64
    top_k: int = 3
    diversity_weight: float = 0.1
    selection_temperature: float = 5.0   # sharper softmax over top-k


class SparseConceptLNN(nn.Module):
    """LNN with sparse pattern bottleneck. Same forward signature as
    `EssentialityLNN.forward(batch, store_memory=True) -> dict` plus extra
    keys for pattern diagnostics.

    Returns dict with:
      'essentiality'    : [N] logits (predicted from pattern combination)
      'regulator_proxy' : [N, K_reg] auxiliary head (uses raw hidden)
      'hidden'          : [N, H] raw liquid output (for inspection)
      'g_repr'          : [N, H] pattern-reconstructed representation
      'pattern_weights' : [N, top_k] softmax over selected patterns
      'pattern_idx'     : [N, top_k] indices of selected patterns
      'pattern_scores'  : [N, n_patterns] cosine similarity to all patterns
      'diversity_loss'  : scalar — to be added to the training loss
    """

    def __init__(self, cfg: SparseConceptLNNConfig | None = None):
        super().__init__()
        cfg = cfg or SparseConceptLNNConfig()
        if cfg.base_cfg is None:
            cfg.base_cfg = LNNConfig()
        self.cfg = cfg
        self.base = EssentialityLNN(cfg.base_cfg)

        H = cfg.base_cfg.hidden
        self.n_patterns = cfg.n_patterns
        self.top_k = cfg.top_k
        # Pattern dictionary: K vectors of dim hidden. Init with small noise
        # so cosine similarities start near zero.
        self.patterns = nn.Parameter(torch.randn(cfg.n_patterns, H) * (0.5 / H ** 0.5))

        # Replace base's essentiality head — we'll re-define to take g_repr
        # as input (same shape as h, so the existing head works as-is).
        # Keep the base.essentiality_head architecture; we just rebuild it
        # to ensure no learned weights are inherited from the unused init.
        d = cfg.base_cfg.dropout
        self.base.essentiality_head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H // 2),
            nn.SiLU(),
            nn.Dropout(d),
            nn.Linear(H // 2, 1),
        )

    def forward(self, batch: dict, store_memory: bool = True) -> dict:
        # 1. Encoder + liquid backbone (manually, so we can intercept hidden)
        h0 = self.base.encoder(batch)
        h = h0.clone()
        edge_index = batch.get('edge_index',
                                 torch.zeros(2, 0, dtype=torch.long,
                                              device=h.device))
        edge_attr = batch.get('edge_attr',
                                torch.zeros(0, self.base.cfg.edge_dim,
                                             device=h.device))
        for step_module in self.base.liquid_steps:
            if self.base.memory is not None:
                h_mem = self.base.memory.retrieve(h)
            else:
                h_mem = torch.zeros_like(h)
            h = step_module(h, h0, edge_index, edge_attr, h_memory=h_mem)

        # 2. Sparse pattern bottleneck
        patterns_norm = F.normalize(self.patterns, dim=-1)        # [K, H]
        h_norm = F.normalize(h, dim=-1)                            # [N, H]
        scores = h_norm @ patterns_norm.T                          # [N, K]

        top_k_vals, top_k_idx = scores.topk(self.top_k, dim=-1)    # [N, top_k]
        # Sharper softmax so selection concentrates
        top_k_weights = F.softmax(
            top_k_vals * self.cfg.selection_temperature, dim=-1)   # [N, top_k]
        # Reconstruct gene representation from selected patterns
        gathered = self.patterns[top_k_idx]                        # [N, top_k, H]
        g_repr = (top_k_weights.unsqueeze(-1) * gathered).sum(dim=1)  # [N, H]

        # 3. Heads (essentiality from pattern combination, regulator from raw h)
        ess_logits = self.base.essentiality_head(g_repr).squeeze(-1)
        reg_logits = self.base.regulator_proxy_head(h)

        # 4. Diversity loss: penalize off-diagonal similarities so patterns
        # cover different directions in feature space.
        K = self.n_patterns
        eye = torch.eye(K, device=patterns_norm.device)
        sim = patterns_norm @ patterns_norm.T - eye               # [K, K]
        diversity_loss = (sim ** 2).mean()

        # 5. Memory storage (only if base has a memory bank enabled)
        if store_memory and self.training and self.base.memory is not None:
            self.base.memory.store(h.detach())

        return {
            'essentiality': ess_logits,
            'regulator_proxy': reg_logits,
            'hidden': h,
            'g_repr': g_repr,
            'pattern_weights': top_k_weights,
            'pattern_idx': top_k_idx,
            'pattern_scores': scores,
            'diversity_loss': diversity_loss,
        }

    def reset_memory(self):
        self.base.reset_memory()


# ─────────────── pattern interpretation helpers ───────────────


def find_top_genes_per_pattern(model: SparseConceptLNN,
                                  batches_by_org: dict,
                                  top_n: int = 10,
                                  device: str = 'cpu') -> dict:
    """For each of K patterns, find the top-N most-activating genes across
    all organisms in `batches_by_org`. Returns dict {pattern_idx: [{org,
    locus, name, score, essential}, ...]} sorted by score descending."""
    model.eval()
    K = model.n_patterns
    pool = {k: [] for k in range(K)}
    with torch.no_grad():
        for org, batch in batches_by_org.items():
            out = model(batch, store_memory=False)
            scores = out['pattern_scores'].cpu().numpy()
            loci = batch['locus_tags']
            names = batch['gene_names']
            labels = batch['labels'].cpu().numpy().astype(int)
            for k in range(K):
                col = scores[:, k]
                # Get top genes for this pattern within this organism
                idx = col.argsort()[::-1][:top_n]
                for i in idx:
                    pool[k].append({
                        'organism': org,
                        'locus_tag': loci[i],
                        'gene_name': names[i] if names[i] else '',
                        'score': float(col[i]),
                        'essential': int(labels[i]),
                    })
    # Sort each pattern's list globally and trim to top_n
    for k in range(K):
        pool[k].sort(key=lambda x: -x['score'])
        pool[k] = pool[k][:top_n]
    return pool


def pattern_summary(top_genes: dict) -> dict:
    """Summarize each pattern's biology: dominant gene names, essential
    fraction, organism diversity."""
    out = {}
    for k, genes in top_genes.items():
        if not genes:
            out[k] = {'n': 0, 'essential_frac': 0.0,
                       'top_gene_names': [], 'organisms_covered': 0}
            continue
        n = len(genes)
        n_ess = sum(g['essential'] for g in genes)
        names = [g['gene_name'] for g in genes if g['gene_name']]
        organisms = set(g['organism'] for g in genes)
        out[k] = {
            'n': n,
            'essential_frac': n_ess / n,
            'top_gene_names': names[:5],
            'organisms_covered': len(organisms),
            'mean_score': sum(g['score'] for g in genes) / n,
            'examples': genes[:5],
        }
    return out


# ─────────────── self-test ───────────────


def _self_test():
    print('SparseConceptLNN self-test:')
    torch.manual_seed(42)

    base_cfg = LNNConfig(
        hidden=32, n_liquid_steps=2, dropout=0.1,
        use_sequence_encoder=True, seq_max_len=128,
        seq_channels_per_kernel=8,
        n_scalar=22, use_memory_bank=False,
    )
    cfg = SparseConceptLNNConfig(
        base_cfg=base_cfg, n_patterns=16, top_k=3,
        diversity_weight=0.1, selection_temperature=5.0,
    )
    model = SparseConceptLNN(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  params: {n_params:,}')
    print(f'  patterns: {cfg.n_patterns} x {base_cfg.hidden}  (top-{cfg.top_k} sparse)')

    N = 20
    batch = {
        'scalar': torch.randn(N, 22),
        'regulatory': torch.randn(N, 50),
        'regulatory_mask': torch.ones(N),
        'kinetic': torch.zeros(N, 4),
        'kinetic_mask': torch.zeros(N),
        'kmer3': torch.zeros(N, 64), 'kmer4': torch.zeros(N, 256),
        'seq_tokens': torch.randint(0, 4, (N, 128)),
        'seq_lengths': torch.full((N,), 128, dtype=torch.long),
        'edge_index': torch.zeros(2, 0, dtype=torch.long),
        'edge_attr': torch.zeros(0, 2),
        'labels': torch.randint(0, 2, (N,)).float(),
        'has_label': torch.ones(N),
        'locus_tags': [f'GENE_{i:03d}' for i in range(N)],
        'gene_names': ['']*N,
    }
    out = model(batch)
    print(f'  forward ok:')
    print(f'    essentiality: {tuple(out["essentiality"].shape)}')
    print(f'    g_repr: {tuple(out["g_repr"].shape)}')
    print(f'    pattern_weights: {tuple(out["pattern_weights"].shape)}')
    print(f'    pattern_idx: {tuple(out["pattern_idx"].shape)}')
    print(f'    pattern_scores: {tuple(out["pattern_scores"].shape)}')
    print(f'    diversity_loss: {float(out["diversity_loss"]):.4f}')

    # Verify sparsity: pattern_weights row sums ~ 1
    row_sums = out['pattern_weights'].sum(dim=-1)
    assert (row_sums - 1.0).abs().max() < 1e-5, \
        f'pattern weights not normalized: {row_sums}'
    print(f'  sparsity verified: top-{cfg.top_k} weights sum to 1')

    # Backward
    target = torch.randint(0, 2, (N,)).float()
    bce = F.binary_cross_entropy_with_logits(out['essentiality'], target)
    loss = bce + cfg.diversity_weight * out['diversity_loss']
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters()
                     if p.grad is not None)
    print(f'  backward ok: loss={loss.item():.4f}  total_grad_norm={grad_norm:.3f}')

    # Verify diversity loss responds: setting patterns to orthogonal should
    # drive it to 0.
    with torch.no_grad():
        # Replace patterns with orthogonal vectors (eye-padded)
        K, H = model.patterns.shape
        ortho = torch.eye(K, H) if K <= H else torch.randn(K, H)
        if K > H:
            ortho, _ = torch.linalg.qr(ortho)
            ortho = ortho[:K, :H]
        # Pad/truncate to match shape
        if ortho.shape != model.patterns.shape:
            tmp = torch.zeros_like(model.patterns)
            tmp[:ortho.shape[0], :ortho.shape[1]] = ortho
            ortho = tmp
        model.patterns.data.copy_(ortho)
    out2 = model(batch)
    print(f'  with orthogonal patterns: diversity_loss = '
          f'{float(out2["diversity_loss"]):.6f}  (should be ~0)')
    assert out2['diversity_loss'].item() < 0.01

    # Verify top-k indices are within [0, K)
    assert out['pattern_idx'].min() >= 0
    assert out['pattern_idx'].max() < cfg.n_patterns
    print(f'  pattern indices in valid range [0, {cfg.n_patterns})')

    print('  ALL PASS')


if __name__ == '__main__':
    _self_test()
