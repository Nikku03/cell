"""Train the EssentialityLNN on all 6 organisms WITHOUT Syn3A
(Salmonella + Caulobacter + S.aureus + M.tb + A.baylyi + M.pneumoniae).

Used as the cross-organism ML member of a committee voting ensemble
combined with v15's mechanistic detector + v25's GB. See
scripts/committee_with_simulator.py for the ensemble.

This run skips Phase 2 (continual learning) — single joint Phase 1 over
all 6 training organisms. Syn3A is fully held out.

v32 architecture + loss: pairwise ranking loss + small BCE anchor +
EWC unused since there's no continual phase.

Output:
  outputs/essentiality_lnn_no_syn3a_results.json
  cell_sim/layer_ml/checkpoints/essentiality_lnn_no_syn3a.pt
  outputs/lnn_syn3a_predictions.csv  (per-Syn3A-gene probability)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from cell_sim.layer_ml.essentiality_lnn import EssentialityLNN, LNNConfig
from cell_sim.layer_ml.data_loader import load_organism_batches

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/essentiality_lnn_no_syn3a_results.json'
CHECKPOINT = ROOT / 'cell_sim/layer_ml/checkpoints/essentiality_lnn_no_syn3a.pt'
PREDS_CSV = ROOT / 'outputs/lnn_syn3a_predictions.csv'

TRAIN_ORGS = ['styphimurium', 'ccrescentus', 'saureus', 'mtuberculosis',
              'abaylyi', 'mpne']
TEST_ORGS = ['syn3a']
ALL_ORGS = TRAIN_ORGS + TEST_ORGS

PHASE_1_EPOCHS = 50    # 14k genes/epoch is ~4x slower than v32's 3.5k -> reduce
LR = 1e-3
SEED = 42

LOSS_RANKING_WEIGHT = 1.0
LOSS_BCE_ANCHOR_WEIGHT = 0.1
RANKING_MARGIN = 0.5
RANKING_N_PAIRS = 300


def threshold_search(y_true, probs):
    best_t, best_mcc = 0.5, -2.0
    cand = sorted(set([0.05, 0.1, 0.5, 0.9, 0.95]
                       + list(np.percentile(probs, np.linspace(2, 98, 49)))))
    for t in cand:
        p = (probs >= t).astype(int)
        if p.sum() in (0, len(p)): continue
        if len(set(y_true)) < 2: return t, 0.0
        m = matthews_corrcoef(y_true, p)
        if m > best_mcc: best_mcc, best_t = m, t
    return best_t, best_mcc


def class_weights(y: torch.Tensor) -> tuple[float, float]:
    n = y.numel(); n1 = float(y.sum().item()); n0 = float(n - n1)
    return float(n1 / n), float(n0 / n)


def weighted_bce(logits, target, has_label):
    mask = has_label.to(torch.bool)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    l = logits[mask]; t = target[mask]
    w0, w1 = class_weights(t)
    weights = torch.where(t == 1,
                            torch.tensor(w1, device=l.device),
                            torch.tensor(w0, device=l.device))
    raw = F.binary_cross_entropy_with_logits(l, t, reduction='none')
    return (raw * weights * 2.0).mean()


def pairwise_ranking_loss(logits, target, has_label,
                            n_pairs=300, margin=0.5):
    mask = has_label.to(torch.bool)
    if mask.sum() < 2:
        return torch.tensor(0.0, device=logits.device)
    l = logits[mask]; t = target[mask]
    pos_idx = (t == 1).nonzero(as_tuple=True)[0]
    neg_idx = (t == 0).nonzero(as_tuple=True)[0]
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    n = min(n_pairs, pos_idx.numel(), neg_idx.numel())
    pos_sample = pos_idx[torch.randint(pos_idx.numel(), (n,), device=l.device)]
    neg_sample = neg_idx[torch.randint(neg_idx.numel(), (n,), device=l.device)]
    diff = l[pos_sample] - l[neg_sample]
    return F.relu(margin - diff).mean()


def regulator_proxy_loss(reg_logits, reg_features, reg_mask):
    mask = reg_mask > 0.5
    if mask.sum() == 0:
        return torch.tensor(0.0, device=reg_logits.device)
    if reg_features.size(1) >= 8:
        targets = torch.stack([reg_features[:, 1],
                                 reg_features[:, 4],
                                 reg_features[:, 7]], dim=-1)
    else:
        targets = torch.zeros(reg_features.size(0), 3, device=reg_logits.device)
    targets = targets[mask]
    return F.binary_cross_entropy_with_logits(reg_logits[mask], targets)


def evaluate_one(model, batch, name=''):
    model.eval()
    with torch.no_grad():
        out = model(batch, store_memory=False)
    probs = torch.sigmoid(out['essentiality']).cpu().numpy()
    y = batch['labels'].cpu().numpy().astype(int)
    has = batch['has_label'].cpu().numpy() > 0
    probs = probs[has]; y = y[has]
    best_t, _ = threshold_search(y, probs)
    pred = (probs >= best_t).astype(int)
    if len(set(pred)) > 1 and len(set(y)) > 1:
        mcc = matthews_corrcoef(y, pred)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    else:
        mcc = 0.0; tp = fp = tn = fn = 0
    return {'organism': name, 'mcc': float(mcc),
            'threshold': float(best_t),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'probs': probs.tolist(),
            'locus_tags': [batch['locus_tags'][i] for i in range(len(batch['locus_tags'])) if has[i]],
            'gene_names': [batch['gene_names'][i] for i in range(len(batch['gene_names'])) if has[i]],
            'true_labels': y.tolist()}


def train_epoch(model, opt, batches, reg_proxy_weight=0.1):
    model.train()
    losses = []
    for batch in batches:
        opt.zero_grad()
        out = model(batch, store_memory=True)
        rank_loss = pairwise_ranking_loss(
            out['essentiality'], batch['labels'], batch['has_label'],
            n_pairs=RANKING_N_PAIRS, margin=RANKING_MARGIN)
        bce_anchor = weighted_bce(out['essentiality'],
                                    batch['labels'], batch['has_label'])
        ess_loss = LOSS_RANKING_WEIGHT * rank_loss + LOSS_BCE_ANCHOR_WEIGHT * bce_anchor
        reg_loss = regulator_proxy_loss(out['regulator_proxy'],
                                          batch['regulatory'],
                                          batch['regulatory_mask'])
        loss = ess_loss + reg_proxy_weight * reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[setup] device={device}, seed={SEED}')

    print('\n[1] loading data for all organisms (Syn3A held out)...')
    all_batches = load_organism_batches(ALL_ORGS)
    for org, b in all_batches.items():
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                all_batches[org][k] = v.to(device)

    cfg = LNNConfig(hidden=128, n_liquid_steps=3, memory_size=512,
                     memory_retrieval_k=8, dropout=0.1)
    model = EssentialityLNN(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n[2] LNN: {n_params:,} params  hidden=128  steps=3  memory=512')

    print(f'\n[3] joint training on {len(TRAIN_ORGS)} organisms (no Syn3A) '
          f'for {PHASE_1_EPOCHS} epochs...')
    train_batches = [all_batches[o] for o in TRAIN_ORGS if o in all_batches]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, PHASE_1_EPOCHS)
    t0 = time.time()
    losses = []
    for ep in range(PHASE_1_EPOCHS):
        l = train_epoch(model, opt, train_batches)
        sched.step()
        losses.append(l)
        if (ep + 1) % 10 == 0:
            print(f'  ep {ep+1:3d}: loss={l:.4f}  '
                  f'memory_n={model.memory.n_populated}')
    print(f'  trained in {time.time()-t0:.1f}s')

    print('\n[4] per-organism evaluation (training orgs + Syn3A held out):')
    evals = {}
    for org in ALL_ORGS:
        if org not in all_batches: continue
        e = evaluate_one(model, all_batches[org], name=org)
        evals[org] = e
        kind = 'TRAIN' if org in TRAIN_ORGS else 'TEST '
        print(f'  [{kind}] {org:<15s}  MCC={e["mcc"]:.4f}  '
              f'TP={e["tp"]} FP={e["fp"]} TN={e["tn"]} FN={e["fn"]}  '
              f'(t={e["threshold"]:.2f})')

    # Save Syn3A predictions for the committee experiment
    syn = evals['syn3a']
    df_pred = pd.DataFrame({
        'locus_tag': syn['locus_tags'],
        'gene_name': syn['gene_names'],
        'lnn_prob': syn['probs'],
        'lnn_pred_at_best_t': [int(p >= syn['threshold']) for p in syn['probs']],
        'true_essential': syn['true_labels'],
    })
    df_pred.to_csv(PREDS_CSV, index=False)
    print(f'\n  wrote per-gene Syn3A LNN predictions: {PREDS_CSV}')

    # Save checkpoint
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'config': cfg.__dict__,
        'state_dict': model.state_dict(),
        'memory_buffer': model.memory.memory.cpu(),
        'memory_populated': model.memory.populated.cpu(),
        'train_orgs': TRAIN_ORGS,
        'evals': {o: {k: v for k, v in e.items()
                       if k not in ('probs', 'locus_tags', 'gene_names',
                                      'true_labels')}
                   for o, e in evals.items()},
    }, CHECKPOINT)
    print(f'  wrote checkpoint: {CHECKPOINT}')

    out = {
        'method': 'lnn_v32_arch_trained_on_6_orgs_no_syn3a',
        'training_organisms': TRAIN_ORGS,
        'test_organism': 'syn3a',
        'epochs': PHASE_1_EPOCHS,
        'final_loss': float(losses[-1]),
        'config': cfg.__dict__,
        'n_parameters': n_params,
        'evals_summary': {o: {k: v for k, v in e.items()
                                 if k not in ('probs', 'locus_tags',
                                                'gene_names', 'true_labels')}
                            for o, e in evals.items()},
        'syn3a_test_mcc': evals['syn3a']['mcc'],
        'baselines': {
            'v15_within_organism_syn3a': 0.5372,
            'v25_gb_cross_organism': 0.2270,
            'v29_gb_with_conservation': 0.2438,
            'v32_lnn_phase1_in_domain_syn3a': 0.5347,
            'v32_lnn_cross_org_avg_phase1': 0.275,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f'  wrote {OUT}')


if __name__ == '__main__':
    main()
