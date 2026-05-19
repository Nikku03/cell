"""M8 upgrade 6/10: ESM-2 protein language model features.

Pre-computes ESM-2 embeddings (Meta, 2022) for each Syn3A protein
and stores them as a parquet file keyed by locus_4d. M7's static
feature builder reads this file and adds the embeddings (mean-pooled
per protein, projected to 64 dims) to the static feature tensor for
all P_* rows.

This replaces M7.7's hand-engineered protein representation (8-dim
role one-hot + 13-dim proteomics) with a learned 1280-dim language
model representation, projected to a manageable 64-dim.

ESM-2 model options:
  - esm2_t6_8M_UR50D  : 8M params, 320-dim, fast. Default for testing.
  - esm2_t30_150M_UR50D: 150M params, 640-dim. Better quality.
  - esm2_t33_650M_UR50D: 650M params, 1280-dim. Strongest published.

For Syn3A's 493 proteins, even the 650M model takes ~5 min to embed
on a GPU. One-time cost; saved to parquet for reuse.

Usage:
    # Step 1: extract embeddings (one-time, ~5 min on GPU)
    python -m cell_sim.lgnn.extras.esm2_protein_features \\
        --fasta /path/to/syn3a_proteins.fasta \\
        --output /content/drive/MyDrive/cell_count_dynamics/esm2_embeddings.parquet \\
        --model esm2_t30_150M_UR50D

    # Step 2: train M7 with these features
    cfg = M7TrainConfig(
        ...,
        use_esm2_protein_features=True,
        esm2_embeddings_parquet='/content/drive/.../esm2_embeddings.parquet',
        esm2_proj_dim=64,
    )

Falls back to zero features (no effect) if the parquet is missing,
so this is safe to leave enabled in cfg even before extraction.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch


ESM2_FEATURE_PREFIX = 'esm2'
_LOCUS_RE = re.compile(r'_(\d{4})(?:_|$)')


def build_esm2_protein_features(
    parquet_path: Path | str,
    row_names: Sequence[str],
    *,
    proj_dim: int = 64,
    seed: int = 0,
    verbose: bool = True,
) -> torch.Tensor:
    """Load pre-computed ESM-2 embeddings and project to ``proj_dim``.

    Returns a (len(row_names), proj_dim) FloatTensor. Rows that are not
    proteins (i.e. don't start with P_) get zeros. Rows whose locus is
    not in the parquet also get zeros (and a warning).

    The projection is a fixed random matrix (seeded). Same seed → same
    projection across runs. This is intentional — letting the network
    learn its own projection would be more powerful but adds N×proj_dim
    parameters and the model already has plenty of capacity.
    """
    import pandas as pd

    N = len(row_names)
    out = np.zeros((N, proj_dim), dtype=np.float32)

    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        if verbose:
            print(f'  WARNING: ESM-2 embeddings missing at {parquet_path}; '
                  f'returning zero tensor (no effect on training)')
        return torch.from_numpy(out)

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        if verbose:
            print(f'  WARNING: failed to read {parquet_path}: {e}')
        return torch.from_numpy(out)

    # Expected columns: 'locus_4d' (str), 'embedding' (list/array of floats)
    if 'locus_4d' not in df.columns:
        if verbose:
            print(f'  ESM-2 parquet missing locus_4d column; got {list(df.columns)}')
        return torch.from_numpy(out)
    if 'embedding' not in df.columns:
        if verbose:
            print(f'  ESM-2 parquet missing embedding column; got {list(df.columns)}')
        return torch.from_numpy(out)

    # Build locus → embedding map
    locus_to_emb = {}
    for _, r in df.iterrows():
        locus = str(r['locus_4d']).zfill(4)
        emb = np.asarray(r['embedding'], dtype=np.float32)
        locus_to_emb[locus] = emb

    if not locus_to_emb:
        return torch.from_numpy(out)

    # Get a sample embedding to determine dim
    sample_emb = next(iter(locus_to_emb.values()))
    emb_dim = sample_emb.shape[0]

    # Build random projection matrix (seeded for reproducibility)
    rng = np.random.default_rng(seed)
    proj_mat = rng.standard_normal(
        (emb_dim, proj_dim), dtype=np.float32) / np.sqrt(emb_dim)

    # Fill in features for protein rows
    n_matched = 0
    for i, n in enumerate(row_names):
        if not n.startswith('P_'):
            continue
        m = _LOCUS_RE.search(n)
        if m is None:
            continue
        locus = m.group(1)
        if locus in locus_to_emb:
            out[i] = locus_to_emb[locus] @ proj_mat
            n_matched += 1

    if verbose:
        n_p_rows = sum(1 for n in row_names if n.startswith('P_'))
        print(f'  ESM-2 features: matched {n_matched}/{n_p_rows} P_* rows '
              f'(embed_dim={emb_dim} -> proj_dim={proj_dim})')

    return torch.from_numpy(out)


# ============================================================
# CLI: extract embeddings from a FASTA file
# ============================================================
def extract_esm2_embeddings(
    fasta_path: str,
    output_parquet: str,
    model_name: str = 'esm2_t30_150M_UR50D',
    batch_size: int = 8,
    device: str = 'cuda',
) -> None:
    """Run ESM-2 on each sequence in `fasta_path` and save mean-pooled
    embeddings to `output_parquet`.

    Requires the `fair-esm` package (pip install fair-esm).
    The FASTA file should have headers like ``>JCVISYN3A_0001 ...``
    so the locus_4d (0001) can be parsed.
    """
    try:
        import esm
    except ImportError:
        raise SystemExit(
            'fair-esm not installed; run: pip install fair-esm'
        )
    import pandas as pd

    print(f'Loading ESM-2 model: {model_name}')
    model_loader = getattr(esm.pretrained, model_name, None)
    if model_loader is None:
        raise ValueError(f'unknown ESM-2 model: {model_name}')
    model, alphabet = model_loader()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device).eval()

    # Parse FASTA
    sequences = []
    cur_header = None
    cur_seq_parts = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if cur_header is not None:
                    sequences.append((cur_header, ''.join(cur_seq_parts)))
                cur_header = line[1:].split()[0]   # ID before first space
                cur_seq_parts = []
            else:
                cur_seq_parts.append(line)
        if cur_header is not None:
            sequences.append((cur_header, ''.join(cur_seq_parts)))

    print(f'Loaded {len(sequences)} sequences')

    # Extract embeddings batch-by-batch
    out_records = []
    n_layers = model.num_layers
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        labels, strs, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            result = model(tokens, repr_layers=[n_layers],
                            return_contacts=False)
        token_reprs = result['representations'][n_layers]
        # Mean-pool over sequence length, skip BOS/EOS
        for i, (label, seq) in enumerate(batch):
            L = len(seq)
            emb = token_reprs[i, 1:L + 1].mean(dim=0).float().cpu().numpy()
            # Parse locus from label like 'JCVISYN3A_0001'
            m = re.search(r'(\d{4})', label)
            locus = m.group(1) if m else label
            out_records.append({
                'locus_4d': locus,
                'header': label,
                'embedding': emb.tolist(),
                'seq_len': L,
            })
        print(f'  [{start + len(batch)}/{len(sequences)}] '
              f'last locus: {out_records[-1]["locus_4d"]}')

    df = pd.DataFrame(out_records)
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f'\nwrote {len(df)} embeddings to {output_parquet}')
    print(f'  embedding dim: {len(df.iloc[0]["embedding"])}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fasta', required=True)
    p.add_argument('--output', required=True,
                   help='output parquet path')
    p.add_argument('--model', default='esm2_t30_150M_UR50D',
                   help='ESM-2 model name')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    extract_esm2_embeddings(
        args.fasta, args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == '__main__':
    main()
