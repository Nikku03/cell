"""Cross-organism protein-protein interaction (PPI) prediction.

Trains a PPIPredictor on PPI pairs from N-1 organisms and evaluates on the
held-out one. Designed to test whether the LNN architecture (multimodal
encoder + multi-head self-attention + cross-attention pair head) can predict
PPIs in an unseen organism using only sequence + scalar features.

Data: outputs/ppi_pairs.csv (built by notebooks/ppi_cross_org.ipynb from
STRING physical-only links + IntAct + Negatome).

Usage
-----
  python scripts/train_ppi_cross_org.py [--holdout syn3a] [--epochs 60]
        [--neg-per-pos 1.0]

Output: outputs/ppi_cross_org_results.json

Notes
-----
- Random negatives are sampled per-organism (1:1 by default), excluding
  pairs that appear as positives in the same organism.
- Pairwise BCE + ranking loss: positive pairs ranked above negatives.
- Eval reports MCC, AUC, AP separately for each negative source (random
  vs Negatome, when present) so we can see how brittle the predictor is
  to negative-sample choice.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (matthews_corrcoef, roc_auc_score,
                                average_precision_score, confusion_matrix)

from cell_sim.layer_ml.ppi_predictor import PPIPredictor, PPIPredictorConfig
from cell_sim.layer_ml.multimodal_encoder import MultimodalEncoderConfig
from cell_sim.layer_ml.data_loader import load_organism_batches
from cell_sim.layer_ml.ppi_data_loader import load_ppi_batches

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/ppi_cross_org_results.json'

ALL_ORGS = ['styphimurium', 'ccrescentus', 'saureus', 'mtuberculosis',
            'abaylyi', 'mgenitalium', 'mpne', 'syn3a']

LR = 1e-3
RANKING_MARGIN = 0.5
RANKING_N_PAIRS = 300
LAMBDA_HIGH = 1e-3
LAMBDA_LOW = 1e-5
SEED = 42


def cosine_lambda(ep, total, hi, lo):
    if total <= 1: return hi
    cos = 0.5 * (1 + np.cos(np.pi * ep / max(total - 1, 1)))
    return float(lo + (hi - lo) * cos)


def pairwise_ranking_loss(logits, labels, n_pairs=300, margin=0.5):
    pos = (labels == 1).nonzero(as_tuple=True)[0]
    neg = (labels == 0).nonzero(as_tuple=True)[0]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    n = min(n_pairs, pos.numel(), neg.numel())
    ps = pos[torch.randint(pos.numel(), (n,), device=logits.device)]
    ns = neg[torch.randint(neg.numel(), (n,), device=logits.device)]
    return F.relu(margin - (logits[ps] - logits[ns])).mean()


def evaluate_one(model, ppi_batch, name=''):
    model.eval()
    with torch.no_grad():
        out = model(ppi_batch['gene_batch'], ppi_batch['idx_a'], ppi_batch['idx_b'])
    probs = torch.sigmoid(out['logits']).cpu().numpy()
    y = ppi_batch['labels'].cpu().numpy().astype(int)
    sources = ppi_batch['sources']

    # Best threshold on this org (diagnostic; cross-org we'd report at 0.5)
    best_t, best_mcc = 0.5, -2.0
    for t in sorted(set([0.05, 0.1, 0.5, 0.9, 0.95]
                          + list(np.percentile(probs, np.linspace(2, 98, 49))))):
        p = (probs >= t).astype(int)
        if p.sum() in (0, len(p)): continue
        if len(set(y)) < 2: continue
        m = matthews_corrcoef(y, p)
        if m > best_mcc: best_mcc, best_t = m, t

    pred = (probs >= best_t).astype(int)
    if len(set(pred)) > 1 and len(set(y)) > 1:
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        try: auc = float(roc_auc_score(y, probs))
        except ValueError: auc = float('nan')
        try: ap = float(average_precision_score(y, probs))
        except ValueError: ap = float('nan')
    else:
        tn = fp = fn = tp = 0; auc = float('nan'); ap = float('nan')

    # Per-source breakdown (positive pairs vs each negative source)
    by_source = {}
    pos_mask = y == 1
    pos_probs = probs[pos_mask]
    for src in set(sources):
        if src == 'random_negative' or src == 'negatome':
            src_mask = np.array([s == src for s in sources])
            src_y = y[src_mask]; src_probs = probs[src_mask]
            # Combine with positives for a meaningful AUC
            sub_y = np.concatenate([src_y, np.ones_like(pos_probs, dtype=int)])
            sub_probs = np.concatenate([src_probs, pos_probs])
            if len(set(sub_y)) > 1:
                try: src_auc = float(roc_auc_score(sub_y, sub_probs))
                except ValueError: src_auc = float('nan')
            else:
                src_auc = float('nan')
            by_source[src] = {'n': int(src_mask.sum()), 'auc_vs_pos': src_auc}

    return {
        'organism': name, 'mcc_at_best_t': float(best_mcc),
        'best_t': float(best_t), 'auc': auc, 'ap': ap,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'n_pairs': int(len(y)), 'n_pos': int(pos_mask.sum()),
        'n_neg': int((~pos_mask).sum()),
        'by_source': by_source,
    }


def train_one_fold(model, train_ppi_batches, n_epochs, verbose_every=10):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=LAMBDA_HIGH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    losses = []
    for ep in range(n_epochs):
        wd = cosine_lambda(ep, n_epochs, LAMBDA_HIGH, LAMBDA_LOW)
        for pg in opt.param_groups: pg['weight_decay'] = wd
        model.train()
        ep_losses = []
        for ppi_batch in train_ppi_batches:
            opt.zero_grad()
            out = model(ppi_batch['gene_batch'], ppi_batch['idx_a'],
                         ppi_batch['idx_b'])
            bce = F.binary_cross_entropy_with_logits(
                out['logits'], ppi_batch['labels'])
            rank = pairwise_ranking_loss(out['logits'], ppi_batch['labels'],
                                          n_pairs=RANKING_N_PAIRS,
                                          margin=RANKING_MARGIN)
            loss = bce + 1.0 * rank
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_losses.append(loss.item())
        sched.step()
        losses.append(float(np.mean(ep_losses)))
        if verbose_every and (ep + 1) % verbose_every == 0:
            print(f'    ep {ep+1:3d}: loss={losses[-1]:.4f}  wd={wd:.5f}')
    return losses


def build_model(hidden=128, attn_heads=8, attn_layers=2, cross_layers=2, dropout=0.1):
    enc_cfg = MultimodalEncoderConfig(
        hidden=hidden, dropout=dropout,
        use_sequence_encoder=True, seq_max_len=2048,
        use_multi_head_attention=True,
        attention_heads=attn_heads, attention_layers=attn_layers,
    )
    cfg = PPIPredictorConfig(
        hidden=hidden, dropout=dropout,
        cross_attn_heads=attn_heads, cross_attn_layers=cross_layers,
        head_hidden=hidden * 2, encoder_cfg=enc_cfg,
    )
    return PPIPredictor(cfg), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holdout', default='syn3a',
                     help='organism to hold out for cross-org evaluation')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--neg-per-pos', type=float, default=1.0,
                     help='random negatives sampled per positive pair')
    ap.add_argument('--hidden', type=int, default=128)
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[setup] device={device}, seed={SEED}, holdout={args.holdout}, '
          f'epochs={args.epochs}, neg_per_pos={args.neg_per_pos}')

    print('\n[1] loading gene + PPI data...')
    gene_batches = load_organism_batches(ALL_ORGS, verbose=True)
    ppi_batches = load_ppi_batches(
        ALL_ORGS, n_random_negatives_per_positive=args.neg_per_pos,
        verbose=True, gene_batches=gene_batches,
    )
    # Move tensors to device
    for org in ppi_batches:
        for k, v in ppi_batches[org]['gene_batch'].items():
            if isinstance(v, torch.Tensor):
                ppi_batches[org]['gene_batch'][k] = v.to(device)
        ppi_batches[org]['idx_a'] = ppi_batches[org]['idx_a'].to(device)
        ppi_batches[org]['idx_b'] = ppi_batches[org]['idx_b'].to(device)
        ppi_batches[org]['labels'] = ppi_batches[org]['labels'].to(device)

    print(f'\n[2] building model...')
    model, cfg = build_model(hidden=args.hidden)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  PPIPredictor: {n_params:,} params')

    train_orgs = [o for o in ALL_ORGS
                   if o != args.holdout and o in ppi_batches]
    print(f'  train_orgs = {train_orgs}')
    print(f'  holdout    = {args.holdout}')

    train_ppi_batches = [ppi_batches[o] for o in train_orgs]
    t0 = time.time()
    print(f'\n[3] training {args.epochs} epochs on {len(train_orgs)} orgs...')
    losses = train_one_fold(model, train_ppi_batches, args.epochs)
    train_time = time.time() - t0
    print(f'  trained in {train_time:.1f}s, final loss={losses[-1]:.4f}')

    print(f'\n[4] evaluating...')
    if args.holdout in ppi_batches:
        e_test = evaluate_one(model, ppi_batches[args.holdout],
                                name=args.holdout)
        print(f'  HELDOUT {args.holdout}: MCC={e_test["mcc_at_best_t"]:.4f} '
              f'AUC={e_test["auc"]:.4f} AP={e_test["ap"]:.4f} '
              f'(N={e_test["n_pairs"]}: {e_test["n_pos"]} pos / '
              f'{e_test["n_neg"]} neg)')
        for src, st in e_test['by_source'].items():
            print(f'    AUC vs {src} negatives: {st["auc_vs_pos"]:.4f} '
                  f'(n_neg={st["n"]})')
    else:
        e_test = None
        print(f'  HELDOUT {args.holdout} not in ppi_batches — skipping eval')

    train_evals = {}
    for org in train_orgs:
        ev = evaluate_one(model, ppi_batches[org], name=org)
        train_evals[org] = ev
        print(f'  TRAIN   {org:15s} MCC={ev["mcc_at_best_t"]:.4f} '
              f'AUC={ev["auc"]:.4f} AP={ev["ap"]:.4f}')

    out = {
        'method': 'ppi_predictor_cross_organism',
        'holdout': args.holdout,
        'train_orgs': train_orgs,
        'epochs': args.epochs,
        'neg_per_pos': args.neg_per_pos,
        'hidden': args.hidden,
        'n_params': n_params,
        'train_time_s': train_time,
        'final_loss': losses[-1],
        'holdout_eval': e_test,
        'train_evals': train_evals,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
