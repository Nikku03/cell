"""Score per-gene PPI-derived features for the essentiality LNN (Path A).

For each organism in our 8-org panel, runs the trained PPIPredictor over
all-vs-all gene pairs and aggregates the per-pair scores into 3 per-gene
features:

  ppi_degree_high          # count of partners with predicted_prob >= 0.7
  ppi_degree_mid           # count of partners with predicted_prob >= 0.5
  ppi_max_score            # highest predicted PPI prob involving this gene

Skips the leak-prone "fraction of partners that are essential" feature —
that one would use partner labels and contaminate cross-organism evals.

Output: outputs/ppi_features_per_gene.csv with columns
  organism, locus_tag, ppi_degree_high, ppi_degree_mid, ppi_max_score

Usage
-----
  python scripts/score_ppi_features.py
       [--organisms syn3a,mtuberculosis,...]
       [--checkpoint cell_sim/layer_ml/checkpoints/ppi_predictor.pt]
       [--device cpu|cuda]
       [--pair-batch 4096]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from cell_sim.layer_ml.data_loader import load_organism_batches
from cell_sim.layer_ml.ppi_predictor import (
    PPIPredictor, PPIPredictorConfig,
)
from cell_sim.layer_ml.multimodal_encoder import MultimodalEncoderConfig

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / 'cell_sim/layer_ml/checkpoints/ppi_predictor.pt'
DEFAULT_OUT = ROOT / 'outputs/ppi_features_per_gene.csv'

DEFAULT_ORGS = ['styphimurium', 'ccrescentus', 'saureus', 'mtuberculosis',
                'abaylyi', 'mgenitalium', 'mpne', 'syn3a']

# Thresholds learned roughly from the cross-org PPI runs:
# 0.7 = high confidence (only top-decile pairs); 0.5 = moderate.
HIGH_PROB_T = 0.7
MID_PROB_T = 0.5


def load_predictor(checkpoint_path: Path, device: str) -> PPIPredictor:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt['config']
    enc_dict = cfg_dict['encoder_cfg']
    enc_cfg = MultimodalEncoderConfig(**{
        k: v for k, v in enc_dict.items()
        if k in {f.name for f in MultimodalEncoderConfig.__dataclass_fields__.values()}
    })
    enc_cfg.seq_kernel_sizes = tuple(enc_cfg.seq_kernel_sizes)
    cfg_kwargs = {k: v for k, v in cfg_dict.items() if k != 'encoder_cfg'}
    cfg = PPIPredictorConfig(encoder_cfg=enc_cfg, **{
        k: v for k, v in cfg_kwargs.items()
        if k in {f.name for f in PPIPredictorConfig.__dataclass_fields__.values()}
    })
    model = PPIPredictor(cfg).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, cfg


def score_organism(model: PPIPredictor, gene_batch: dict,
                    pair_batch_size: int, device: str) -> pd.DataFrame:
    """Run all-vs-all pair scoring for one organism. Returns
    DataFrame with per-gene aggregate features."""
    loci = gene_batch['locus_tags']
    N = len(loci)
    if N < 2:
        return pd.DataFrame(columns=['locus_tag', 'ppi_degree_high',
                                       'ppi_degree_mid', 'ppi_max_score'])

    # Move tensors to device
    batch = {}
    for k, v in gene_batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        else:
            batch[k] = v

    # Encode genes once (cache the [N, H] gene_repr to reuse across all pairs)
    with torch.no_grad():
        gene_repr = model.encode_genes(batch)

    # All upper-triangular pairs (i < j); predict(i,j) == predict(j,i) by
    # construction so we save half the work.
    iu, ju = np.triu_indices(N, k=1)
    n_pairs = len(iu)
    print(f'  {N} genes -> {n_pairs:,} pairs; '
          f'encoding done, scoring in batches of {pair_batch_size}...')

    # Per-gene running aggregates
    sum_indicator_high = np.zeros(N, dtype=np.int64)
    sum_indicator_mid = np.zeros(N, dtype=np.int64)
    max_score = np.zeros(N, dtype=np.float32)

    t0 = time.time()
    for start in range(0, n_pairs, pair_batch_size):
        end = min(start + pair_batch_size, n_pairs)
        ia = torch.tensor(iu[start:end], dtype=torch.long, device=device)
        ib = torch.tensor(ju[start:end], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(batch, ia, ib, gene_repr=gene_repr)
        probs = torch.sigmoid(out['logits']).cpu().numpy()

        # Aggregate per-gene (each pair contributes to both endpoints)
        np.add.at(sum_indicator_high, iu[start:end], (probs >= HIGH_PROB_T).astype(int))
        np.add.at(sum_indicator_high, ju[start:end], (probs >= HIGH_PROB_T).astype(int))
        np.add.at(sum_indicator_mid, iu[start:end], (probs >= MID_PROB_T).astype(int))
        np.add.at(sum_indicator_mid, ju[start:end], (probs >= MID_PROB_T).astype(int))
        np.maximum.at(max_score, iu[start:end], probs)
        np.maximum.at(max_score, ju[start:end], probs)

        if (start // pair_batch_size) % 20 == 0 and start > 0:
            elapsed = time.time() - t0
            eta = elapsed / end * (n_pairs - end)
            print(f'    {end}/{n_pairs} pairs  elapsed={elapsed:.1f}s  '
                  f'eta={eta:.1f}s')

    print(f'  done in {time.time()-t0:.1f}s')
    return pd.DataFrame({
        'locus_tag': loci,
        'ppi_degree_high': sum_indicator_high,
        'ppi_degree_mid': sum_indicator_mid,
        'ppi_max_score': max_score,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--organisms', default=','.join(DEFAULT_ORGS),
                     help='comma-separated organism list')
    ap.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--pair-batch', type=int, default=4096,
                     help='pair batch size for the scoring loop')
    args = ap.parse_args()

    organisms = [o.strip() for o in args.organisms.split(',') if o.strip()]
    print(f'[setup] device={args.device}, organisms={organisms}')
    print(f'[1] loading PPI predictor checkpoint {args.checkpoint.name}...')
    model, cfg = load_predictor(args.checkpoint, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  PPIPredictor: {n_params:,} params, hidden={cfg.hidden}')

    print(f'\n[2] loading gene batches for {len(organisms)} organisms...')
    gene_batches = load_organism_batches(organisms, verbose=True)

    print(f'\n[3] scoring per-organism PPI features...')
    all_dfs = []
    for org in organisms:
        if org not in gene_batches:
            print(f'  skip {org}: no gene_batch'); continue
        print(f'\n[{org}]')
        df = score_organism(model, gene_batches[org], args.pair_batch,
                              args.device)
        df.insert(0, 'organism', org)
        all_dfs.append(df)
        print(f'  per-gene summary: '
              f'mean ppi_degree_high={df.ppi_degree_high.mean():.2f}  '
              f'mean ppi_max_score={df.ppi_max_score.mean():.3f}')

    if all_dfs:
        out_df = pd.concat(all_dfs, ignore_index=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f'\nwrote {args.out}: {len(out_df)} rows '
              f'across {out_df.organism.nunique()} organisms')
    else:
        print('\nno per-gene features produced')


if __name__ == '__main__':
    main()
