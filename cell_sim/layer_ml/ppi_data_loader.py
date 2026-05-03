"""PPI pair batch loader.

Wraps the per-organism gene batches produced by `data_loader.load_organism_batches`
and adds pair tables read from `outputs/ppi_pairs.csv` (which is built by
the Colab notebook from STRING physical-only links + IntAct + Negatome).

Expected pair CSV schema:
  organism,locus_a,locus_b,label,source,confidence
where label is 1 (interacts) or 0 (does not interact), and source is one of
{string_physical, intact, negatome, random_negative}.

Produces, per organism, a dict:
  {
    'gene_batch': the standard MultimodalEncoder batch,
    'idx_a':      LongTensor [P]
    'idx_b':      LongTensor [P]
    'labels':     FloatTensor [P]
    'sources':    list[str] — for diagnostic per-source MCC reporting
    'locus_to_idx': dict mapping locus_tag -> gene index inside gene_batch
  }
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import torch

from cell_sim.layer_ml.data_loader import load_organism_batches

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PPI_CSV = ROOT / 'outputs/ppi_pairs.csv'


def _build_locus_to_idx(gene_batch: dict) -> dict:
    """Map locus_tag -> position in the gene_batch tensors."""
    return {lt: i for i, lt in enumerate(gene_batch['locus_tags'])}


def random_negative_pairs(gene_batch: dict, positive_set: set,
                            n_pairs: int, rng: np.random.Generator,
                            same_organism: bool = True) -> list:
    """Sample random pairs that are NOT in the positive set. Returns list of
    (locus_a, locus_b) tuples. Same-organism only."""
    if not same_organism:
        raise NotImplementedError('cross-organism negatives not implemented')
    loci = gene_batch['locus_tags']
    n = len(loci)
    out = []
    attempts = 0
    while len(out) < n_pairs and attempts < n_pairs * 20:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            attempts += 1; continue
        a, b = loci[i], loci[j]
        if a > b: a, b = b, a   # canonical ordering for the positive_set lookup
        if (a, b) in positive_set:
            attempts += 1; continue
        out.append((loci[i], loci[j]))
        attempts += 1
    return out


def build_ppi_batch(organism: str, gene_batch: dict, ppi_df: pd.DataFrame,
                     n_random_negatives_per_positive: float = 1.0,
                     min_total_neg_per_pos: float = 1.0,
                     seed: int = 42) -> dict:
    """Convert a per-organism PPI table into a pair batch ready for
    PPIPredictor.

    Negatives behaviour:
      - All label=0 rows in `ppi_df` for this organism are kept as curated
        negatives (Negatome, paralog hard negatives, etc).
      - Random negatives are added only to reach `min_total_neg_per_pos`
        ratio. If curated negatives already exceed that, no randoms added.
      - `n_random_negatives_per_positive` is now an UPPER bound on the
        random additions (defaults to 1.0 = up to 1:1 random:positive).
    """
    rng = np.random.default_rng(seed)
    locus_to_idx = _build_locus_to_idx(gene_batch)

    sub = ppi_df[ppi_df.organism == organism].copy()
    sub = sub[sub.locus_a.isin(locus_to_idx) & sub.locus_b.isin(locus_to_idx)]
    a_arr = np.minimum(sub.locus_a.values, sub.locus_b.values)
    b_arr = np.maximum(sub.locus_a.values, sub.locus_b.values)
    sub = sub.assign(locus_a=a_arr, locus_b=b_arr).drop_duplicates(
        subset=['locus_a', 'locus_b'])

    pos_mask = sub.label == 1
    pos_pairs = list(zip(sub[pos_mask].locus_a, sub[pos_mask].locus_b))
    neg_pairs = list(zip(sub[~pos_mask].locus_a, sub[~pos_mask].locus_b))
    pos_sources = sub[pos_mask].source.values.tolist()
    neg_sources = sub[~pos_mask].source.values.tolist()

    n_pos = len(pos_pairs)
    n_curated_neg = len(neg_pairs)
    positive_set = set(pos_pairs)

    # Decide how many random negatives to add.
    target_total_neg = int(round(n_pos * min_total_neg_per_pos))
    n_random_neg_target = max(0, target_total_neg - n_curated_neg)
    n_random_neg_target = min(n_random_neg_target,
                                int(round(n_pos * n_random_negatives_per_positive)))

    rand_neg = (random_negative_pairs(gene_batch, positive_set,
                                        n_random_neg_target, rng=rng)
                  if n_random_neg_target > 0 else [])

    all_pairs = pos_pairs + neg_pairs + rand_neg
    all_labels = ([1] * n_pos + [0] * n_curated_neg + [0] * len(rand_neg))
    all_sources = pos_sources + neg_sources + ['random_negative'] * len(rand_neg)

    idx_a = torch.tensor([locus_to_idx[a] for a, _ in all_pairs],
                          dtype=torch.long)
    idx_b = torch.tensor([locus_to_idx[b] for _, b in all_pairs],
                          dtype=torch.long)
    labels = torch.tensor(all_labels, dtype=torch.float32)

    return {
        'organism': organism,
        'gene_batch': gene_batch,
        'idx_a': idx_a,
        'idx_b': idx_b,
        'labels': labels,
        'sources': all_sources,
        'locus_to_idx': locus_to_idx,
        'n_positive': n_pos,
        'n_random_negative': len(rand_neg),
        'n_curated_negative': n_curated_neg,
    }


def load_ppi_batches(organisms: list[str],
                      ppi_csv: Path = DEFAULT_PPI_CSV,
                      n_random_negatives_per_positive: float = 1.0,
                      min_total_neg_per_pos: float = 1.0,
                      verbose: bool = True,
                      seed: int = 42,
                      gene_batches: dict | None = None) -> dict:
    """Load gene batches + PPI pair batches for a list of organisms."""
    if not ppi_csv.exists():
        raise FileNotFoundError(
            f'PPI pair CSV not found at {ppi_csv}. Run the Colab notebook '
            f'`notebooks/ppi_cross_org.ipynb` to fetch it from STRING/IntAct/'
            f'Negatome first.')
    ppi_df = pd.read_csv(ppi_csv)
    if verbose:
        print(f'[ppi_loader] loaded {len(ppi_df)} pairs from {ppi_csv.name}')
        for org, g in ppi_df.groupby('organism'):
            n_pos = int((g.label == 1).sum())
            n_neg = int((g.label == 0).sum())
            print(f'  {org:15s} {len(g):>6d} pairs ({n_pos} pos, {n_neg} neg)')

    if gene_batches is None:
        gene_batches = load_organism_batches(organisms, verbose=verbose)

    out = {}
    for i, org in enumerate(organisms):
        if org not in gene_batches:
            print(f'  skip {org}: not in gene_batches'); continue
        out[org] = build_ppi_batch(
            org, gene_batches[org], ppi_df,
            n_random_negatives_per_positive=n_random_negatives_per_positive,
            min_total_neg_per_pos=min_total_neg_per_pos,
            seed=seed + i,
        )
        if verbose:
            b = out[org]
            n_p = b['n_positive']
            n_cur = b['n_curated_negative']
            n_rand = b['n_random_negative']
            print(f'  ppi_batch[{org}] N_genes={len(b["gene_batch"]["locus_tags"])} '
                  f'P={len(b["labels"])} ({n_p} pos / {n_cur} curated_neg / '
                  f'{n_rand} random_neg)')
    return out


if __name__ == '__main__':
    if not DEFAULT_PPI_CSV.exists():
        print('ppi_pairs.csv not present; create from notebook first. Quitting.')
    else:
        batches = load_ppi_batches(['ccrescentus', 'syn3a'], verbose=True)
        for o, b in batches.items():
            print(o, 'idx_a sample:', b['idx_a'][:5].tolist(),
                  'labels:', b['labels'][:5].tolist())
