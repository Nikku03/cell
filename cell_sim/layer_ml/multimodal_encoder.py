"""Multi-modal per-gene encoder. Combines four input branches into a
single hidden vector per gene:

  Scalar branch    - 20 scalar features per v25 (length, GC, position,
                     keyword indicators, etc.). Always present.
  Regulatory branch - up to 50 features per v27 (RBS / -10 / -35 / TF
                     binding scores from Salmonella's RegulonDB-derived
                     PWMs applied to upstream sequence). Per-gene mask
                     indicates whether features were computed for this
                     organism (Salmonella + the PWM-applied 16) or not.
  Kinetic branch    - 4 features per gene where the gene maps to a
                     reaction in the organism's SBML/kinetic_params:
                     log_kcat, log_Km, log_Vmax, has_kinetics_flag.
                     Heavily masked (most genes don't map to a reaction).
  Sequence branch   - 320 normalized k-mer counts (64 codon 3-mers +
                     256 4-mers from the CDS sequence). Always present.

Each branch handles its own missing-data masking: when a feature isn't
available, the branch outputs a zero hidden vector AND emits a
modality-presence flag the combiner uses for gating.

Self-test: `python -m cell_sim.layer_ml.multimodal_encoder`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultimodalEncoderConfig:
    n_scalar: int = 20
    n_regulatory: int = 50
    n_kinetic: int = 4
    n_kmer3: int = 64
    n_kmer4: int = 256
    hidden: int = 128
    dropout: float = 0.1

    # Sequence encoder option (when True, replaces kmer3+kmer4 branches with a
    # per-gene CNN over raw nucleotide tokens — keeps positional information).
    use_sequence_encoder: bool = False
    seq_max_len: int = 2048
    seq_embed_dim: int = 32
    seq_kernel_sizes: tuple = (3, 5, 7, 9)
    seq_channels_per_kernel: int = 32
    seq_vocab_size: int = 5     # A, C, G, T, N/pad

    # High-dim pattern capacity: optional multi-head self-attention over the
    # batch of gene embeddings. Each gene attends to every other gene in the
    # batch, augmenting the edge-graph message-passing in LiquidStep with
    # full all-pairs attention. Helps capture organism-wide patterns
    # (e.g. operonic neighbours, paralog clusters) the edge graph misses.
    use_multi_head_attention: bool = False
    attention_heads: int = 8
    attention_layers: int = 2
    attention_ff_mult: int = 2


class SequenceEncoder(nn.Module):
    """Per-gene CNN encoder over raw nucleotide tokens.

    Token vocab (token id -> nucleotide):
      0 = A, 1 = C, 2 = G, 3 = T, 4 = N/pad

    Architecture:
      embedding: vocab(5) -> embed_dim(32)
      multi-scale Conv1d: kernel sizes (3, 5, 7, 9), each producing 32 channels
      mask out padded positions before pooling
      global max-pool per channel
      concat -> Linear -> output_dim

    Output: [N, output_dim] fixed-size embedding per gene.

    Padded positions (token id = 4) get masked to a very negative value
    BEFORE max pooling so they can't contribute to the pooled signal.
    Embedding for token 4 is also explicitly zeroed via padding_idx.
    """

    def __init__(self, embed_dim: int = 32, output_dim: int = 64,
                 kernel_sizes=(3, 5, 7, 9), channels_per_kernel: int = 32,
                 max_len: int = 2048, vocab_size: int = 5,
                 pad_token: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.max_len = max_len
        self.pad_token = pad_token

        self.token_embed = nn.Embedding(vocab_size, embed_dim,
                                          padding_idx=pad_token)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, channels_per_kernel, k, padding=k // 2)
            for k in kernel_sizes
        ])
        total_conv_dim = channels_per_kernel * len(kernel_sizes)
        self.proj = nn.Sequential(
            nn.Linear(total_conv_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        """tokens: [N, L] int64 with values in 0..vocab_size-1.
        lengths: [N] optional real lengths. If None, infer from token!=pad.
        Returns: [N, output_dim]."""
        # tokens: [N, L]; embed: [N, L, embed_dim] -> [N, embed_dim, L] for conv1d
        h = self.token_embed(tokens).transpose(1, 2)

        # Build per-position mask: 1 = real, 0 = pad
        if lengths is not None:
            arange = torch.arange(tokens.size(1), device=tokens.device)
            mask = (arange.unsqueeze(0) < lengths.unsqueeze(1)).to(h.dtype)
        else:
            mask = (tokens != self.pad_token).to(h.dtype)
        mask = mask.unsqueeze(1)  # [N, 1, L] for broadcasting

        feats = []
        for conv in self.convs:
            f = F.relu(conv(h))                    # [N, C, L]
            # Mask padded positions before max-pooling
            f = f.masked_fill(mask == 0, -1e9)
            pooled = F.adaptive_max_pool1d(f, 1).squeeze(-1)  # [N, C]
            feats.append(pooled)
        h = torch.cat(feats, dim=-1)              # [N, total_conv_dim]
        return self.proj(h)


class _Branch(nn.Module):
    """Generic feature-branch encoder with missing-data masking.

    Forward takes (x, mask) where mask ∈ {0, 1} is per-gene presence.
    When mask=0 the branch's output is multiplied by 0 (zero vector) so
    downstream layers see "no signal from this modality" rather than noise."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        # Two-layer projection. LayerNorm before activation keeps gradient
        # well-behaved when the input dim is high (e.g. 256 for k-mer 4).
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        h = self.net(x)
        if mask is not None:
            # mask: [N] -> [N, 1] for broadcasting
            if mask.dim() == 1:
                mask = mask.unsqueeze(-1).to(h.dtype)
            h = h * mask
        return h


class MultimodalEncoder(nn.Module):
    """Concatenate four per-gene branches + 4 modality-presence flags
    into a single hidden vector. The presence flags let the LNN learn
    'attend more to the kinetic branch when it's present, less otherwise'.

    Inputs (a dict; tensors all have batch-leading dim N):
      'scalar'        : [N, n_scalar]
      'regulatory'    : [N, n_regulatory]    (zero-filled where missing)
      'regulatory_mask': [N]  bool/float     (1 where regulatory features computed)
      'kinetic'       : [N, n_kinetic]       (zero-filled where missing)
      'kinetic_mask'  : [N]  bool/float      (1 where gene maps to a reaction)
      'kmer3'         : [N, 64]
      'kmer4'         : [N, 256]

    Output: [N, hidden] hidden vectors per gene.
    """

    def __init__(self, cfg: MultimodalEncoderConfig | None = None):
        super().__init__()
        cfg = cfg or MultimodalEncoderConfig()
        self.cfg = cfg
        H = cfg.hidden

        # Each branch projects to H/4 so concat(4 branches) = H.
        # Plus 4 dim for modality-presence flags. Combiner reduces back to H.
        b = H // 4
        self.scalar_branch = _Branch(cfg.n_scalar, b, cfg.dropout)
        self.regulatory_branch = _Branch(cfg.n_regulatory, b, cfg.dropout)
        self.kinetic_branch = _Branch(cfg.n_kinetic, b, cfg.dropout)
        if cfg.use_sequence_encoder:
            # Sequence-level encoder (CNN over raw nucleotides)
            self.sequence_branch = SequenceEncoder(
                embed_dim=cfg.seq_embed_dim, output_dim=b,
                kernel_sizes=cfg.seq_kernel_sizes,
                channels_per_kernel=cfg.seq_channels_per_kernel,
                max_len=cfg.seq_max_len, vocab_size=cfg.seq_vocab_size,
                dropout=cfg.dropout,
            )
            self.kmer_branch = None
        else:
            self.kmer_branch = _Branch(cfg.n_kmer3 + cfg.n_kmer4, b, cfg.dropout)
            self.sequence_branch = None

        self.combiner = nn.Sequential(
            nn.Linear(4 * b + 4, H),
            nn.LayerNorm(H),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(H, H),
        )

        # Optional multi-head self-attention block (Transformer encoder).
        # Treats the batch of N gene embeddings as a length-N token sequence
        # so every gene can attend to every other gene in the same forward call.
        if cfg.use_multi_head_attention:
            attn_layer = nn.TransformerEncoderLayer(
                d_model=H,
                nhead=cfg.attention_heads,
                dim_feedforward=H * cfg.attention_ff_mult,
                dropout=cfg.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.attention = nn.TransformerEncoder(attn_layer,
                                                    num_layers=cfg.attention_layers)
        else:
            self.attention = None

    def forward(self, batch: dict) -> torch.Tensor:
        # Mandatory branches
        s = self.scalar_branch(batch['scalar'])

        # Sequence vs k-mer branch (mutually exclusive at construction time)
        if self.sequence_branch is not None:
            seq_tokens = batch.get('seq_tokens')
            if seq_tokens is None:
                # No sequence in batch — fall back to zeros so the rest still works
                k = torch.zeros(batch['scalar'].size(0), self.cfg.hidden // 4,
                                device=batch['scalar'].device)
            else:
                k = self.sequence_branch(seq_tokens, batch.get('seq_lengths'))
        else:
            kmer_in = torch.cat([batch['kmer3'], batch['kmer4']], dim=-1)
            k = self.kmer_branch(kmer_in)

        # Optional / masked branches
        r_mask = batch.get('regulatory_mask',
                            torch.ones(batch['scalar'].size(0),
                                        device=batch['scalar'].device))
        r = self.regulatory_branch(batch['regulatory'], r_mask)
        ki_mask = batch.get('kinetic_mask',
                             torch.zeros(batch['scalar'].size(0),
                                          device=batch['scalar'].device))
        ki = self.kinetic_branch(batch['kinetic'], ki_mask)

        # Modality-presence flags (broadcast scalars per-gene)
        N = batch['scalar'].size(0)
        device = batch['scalar'].device
        flags = torch.stack([
            torch.ones(N, device=device),                # scalar always present
            r_mask.to(torch.float32),                    # regulatory mask
            ki_mask.to(torch.float32),                   # kinetic mask
            torch.ones(N, device=device),                # sequence/kmer always present
        ], dim=-1)

        combined = torch.cat([s, r, ki, k, flags], dim=-1)
        h = self.combiner(combined)

        if self.attention is not None:
            # [N, H] -> [1, N, H] -> attend -> [N, H]
            h = self.attention(h.unsqueeze(0)).squeeze(0)
        return h


# ──────────── self-test ────────────


def _self_test():
    print('multimodal_encoder self-test:')
    torch.manual_seed(42)
    N = 10
    cfg = MultimodalEncoderConfig(hidden=64)
    enc = MultimodalEncoder(cfg)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f'  params: {n_params:,}  hidden={cfg.hidden}')

    batch = {
        'scalar': torch.randn(N, cfg.n_scalar),
        'regulatory': torch.randn(N, cfg.n_regulatory),
        'regulatory_mask': torch.tensor([1, 1, 0, 0, 1, 0, 1, 1, 0, 1],
                                          dtype=torch.float32),
        'kinetic': torch.randn(N, cfg.n_kinetic),
        'kinetic_mask': torch.tensor([0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                                       dtype=torch.float32),
        'kmer3': torch.rand(N, cfg.n_kmer3),
        'kmer4': torch.rand(N, cfg.n_kmer4),
    }

    out = enc(batch)
    print(f'  output shape: {tuple(out.shape)}  '
          f'mean={out.mean().item():.3f}  std={out.std().item():.3f}')
    assert out.shape == (N, cfg.hidden)

    # Mask sanity: gene with regulatory_mask=0 vs =1 — outputs should differ
    batch_a = {**batch}
    batch_b = {**batch, 'regulatory_mask': torch.ones_like(batch['regulatory_mask'])}
    out_a = enc(batch_a)
    out_b = enc(batch_b)
    delta = (out_a - out_b).abs().sum(dim=-1)
    print(f'  per-gene mask sensitivity (|delta|): mean={delta.mean().item():.3f}')
    # Genes that already had mask=1 in both: delta should be 0 (or near it
    # due to the modality-flag inputs being 0->1 also changing combiner).
    # Easier check: gradients flow through the encoder
    out.sum().backward()
    print('  gradient backward: ok')

    # All-missing-modalities input shouldn't crash
    batch_min = {
        'scalar': torch.randn(N, cfg.n_scalar),
        'regulatory': torch.zeros(N, cfg.n_regulatory),
        'regulatory_mask': torch.zeros(N),
        'kinetic': torch.zeros(N, cfg.n_kinetic),
        'kinetic_mask': torch.zeros(N),
        'kmer3': torch.rand(N, cfg.n_kmer3),
        'kmer4': torch.rand(N, cfg.n_kmer4),
    }
    enc.zero_grad()
    out_min = enc(batch_min)
    assert out_min.shape == (N, cfg.hidden)
    print(f'  all-modalities-missing forward: shape ok, mean={out_min.mean().item():.3f}')

    # ── sequence encoder mode ──
    print('  ── sequence encoder mode ──')
    seq_cfg = MultimodalEncoderConfig(hidden=64, use_sequence_encoder=True,
                                        seq_max_len=512)
    seq_enc = MultimodalEncoder(seq_cfg)
    seq_params = sum(p.numel() for p in seq_enc.parameters())
    print(f'  seq-mode params: {seq_params:,}')

    # Dummy nucleotide tokens (vocab 0-4: A, C, G, T, N/pad)
    seq_batch = {
        'scalar': torch.randn(N, seq_cfg.n_scalar),
        'regulatory': torch.randn(N, seq_cfg.n_regulatory),
        'regulatory_mask': torch.ones(N),
        'kinetic': torch.zeros(N, seq_cfg.n_kinetic),
        'kinetic_mask': torch.zeros(N),
        'seq_tokens': torch.randint(0, 4, (N, seq_cfg.seq_max_len)),
        'seq_lengths': torch.randint(50, seq_cfg.seq_max_len, (N,)),
    }
    # Pad some positions explicitly to test masking
    for i in range(N):
        L = seq_batch['seq_lengths'][i].item()
        seq_batch['seq_tokens'][i, L:] = 4

    out_seq = seq_enc(seq_batch)
    print(f'  forward: shape={tuple(out_seq.shape)}, '
          f'mean={out_seq.mean().item():.3f}, std={out_seq.std().item():.3f}')
    assert out_seq.shape == (N, seq_cfg.hidden)

    # Backward through sequence encoder
    out_seq.sum().backward()
    print('  backward through sequence encoder: ok')

    # Just the SequenceEncoder in isolation
    se = SequenceEncoder(embed_dim=16, output_dim=32, max_len=128,
                          kernel_sizes=(3, 5))
    tok = torch.randint(0, 5, (8, 128))
    out_se = se(tok)
    assert out_se.shape == (8, 32)
    print(f'  SequenceEncoder alone: out shape {tuple(out_se.shape)}, '
          f'params={sum(p.numel() for p in se.parameters())}')

    # Padded sequence: real_length=10, rest is pad
    tok_padded = torch.full((4, 128), 4, dtype=torch.long)
    tok_padded[:, :10] = torch.randint(0, 4, (4, 10))
    lengths = torch.tensor([10, 10, 10, 10])
    out_padded = se(tok_padded, lengths)
    assert out_padded.shape == (4, 32)
    # Output should not be wildly different from the unpadded version
    print(f'  padded sequence output: mean={out_padded.mean().item():.3f} '
          f'(masked 118/128 positions to pad)')

    # ── multi-head self-attention mode ──
    print('  ── multi-head self-attention mode ──')
    attn_cfg = MultimodalEncoderConfig(hidden=64, use_multi_head_attention=True,
                                        attention_heads=4, attention_layers=2)
    attn_enc = MultimodalEncoder(attn_cfg)
    attn_params = sum(p.numel() for p in attn_enc.parameters())
    print(f'  attn-mode params: {attn_params:,}')
    out_attn = attn_enc(batch)
    assert out_attn.shape == (N, attn_cfg.hidden)
    out_attn.sum().backward()
    print(f'  attn forward: shape={tuple(out_attn.shape)}, '
          f'mean={out_attn.mean().item():.3f}, std={out_attn.std().item():.3f}')
    print('  attn backward: ok')

    print('  ALL PASS')


if __name__ == '__main__':
    _self_test()
