"""Leave-one-organism-out cross-organism essentiality with Syn3A and
M. genitalium always in training, optionally using nucleotide-level
sequence features.

For each of the 6 holdout-candidate organisms (Salmonella, Caulobacter,
S.aureus, M.tuberculosis, A.baylyi, M.pneumoniae), train an LNN on the
OTHER 5 holdout candidates PLUS Syn3A and M. genitalium, then test on
the held-out one. Reports MCC per fold and aggregated across folds.

Why Syn3A in training: Syn3A is a curated minimal genome containing
core essential genes that overlap with most bacteria. Including it as
a training organism gives the model "core essentiality" patterns that
should help predict essentials in any organism.

Why M. genitalium in training: Syn3A was derived from M. genitalium G37
by gene-by-gene reduction. Glass et al. 2006 transposon mutagenesis
gives essentiality calls for ~482 M. genitalium genes (~79% essential).
Including the closest natural-genome analogue to Syn3A doubles the
high-essentiality-fraction training signal without contaminating the
held-out tests with simulator-derived Syn3A label noise.

Why leave-one-out on the others (not Syn3A or M. genitalium): the
previous experiments held Syn3A out and showed cross-org transfer to a
minimal-cell organism with 84% essential class is hard. Holding out a
more typical organism (5-15% essential like Caulobacter, M.tb, A.baylyi)
tests the architecture on a more standard cross-org transfer task.

Usage
-----

  python scripts/train_leave_one_org_out.py [--seq] [--epochs N]

  --seq        : use SequenceEncoder (CNN over raw nucleotides, ~17k extra
                  params, replaces k-mer branch). Default: k-mer branch only.
  --epochs N   : training epochs per fold. Default 60.

Output: outputs/leave_one_org_out_results.json
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
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from cell_sim.layer_ml.essentiality_lnn import EssentialityLNN, LNNConfig
from cell_sim.layer_ml.hyperbolic_memory import HyperbolicMemoryBank
from cell_sim.layer_ml.data_loader import load_organism_batches

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/leave_one_org_out_results.json'

ALL_ORGS = ['styphimurium', 'ccrescentus', 'saureus', 'mtuberculosis',
            'abaylyi', 'mgenitalium', 'mpne', 'syn3a']
ALWAYS_IN_TRAINING = ['syn3a', 'mgenitalium']
HOLDOUT_CANDIDATES = [o for o in ALL_ORGS if o not in ALWAYS_IN_TRAINING]

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


def class_weights(y):
    n = y.numel(); n1 = float(y.sum().item()); n0 = float(n - n1)
    return float(n1 / n), float(n0 / n)


def weighted_bce(logits, target, has_label):
    mask = has_label.to(torch.bool)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    l = logits[mask]; t = target[mask]
    w0, w1 = class_weights(t)
    weights = torch.where(t == 1, torch.tensor(w1, device=l.device),
                            torch.tensor(w0, device=l.device))
    raw = F.binary_cross_entropy_with_logits(l, t, reduction='none')
    return (raw * weights * 2.0).mean()


def pairwise_ranking_loss(logits, target, has_label, n_pairs=300, margin=0.5):
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
    return F.relu(margin - (l[pos_sample] - l[neg_sample])).mean()


def regulator_proxy_loss(reg_logits, reg_features, reg_mask):
    mask = reg_mask > 0.5
    if mask.sum() == 0:
        return torch.tensor(0.0, device=reg_logits.device)
    if reg_features.size(1) >= 8:
        targets = torch.stack([reg_features[:, 1], reg_features[:, 4],
                                 reg_features[:, 7]], dim=-1)
    else:
        targets = torch.zeros(reg_features.size(0), 3, device=reg_logits.device)
    return F.binary_cross_entropy_with_logits(reg_logits[mask], targets[mask])


def threshold_search(y, probs):
    best_t, best_mcc = 0.5, -2.0
    cand = sorted(set([0.05, 0.1, 0.5, 0.9, 0.95]
                       + list(np.percentile(probs, np.linspace(2, 98, 49)))))
    for t in cand:
        p = (probs >= t).astype(int)
        if p.sum() in (0, len(p)): continue
        if len(set(y)) < 2: return t, 0.0
        m = matthews_corrcoef(y, p)
        if m > best_mcc: best_mcc, best_t = m, t
    return best_t, best_mcc


def evaluate_one(model, batch, name=''):
    model.eval()
    with torch.no_grad():
        out = model(batch, store_memory=False)
    probs = torch.sigmoid(out['essentiality']).cpu().numpy()
    y = batch['labels'].cpu().numpy().astype(int)
    has = batch['has_label'].cpu().numpy() > 0
    probs = probs[has]; y = y[has]
    best_t, best_mcc = threshold_search(y, probs)
    pred = (probs >= best_t).astype(int)
    if len(set(pred)) > 1 and len(set(y)) > 1:
        mcc = float(matthews_corrcoef(y, pred))
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    else:
        mcc = 0.0; tp = fp = tn = fn = 0
    return {'organism': name, 'mcc': mcc, 'threshold': float(best_t),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'n_genes': int(len(y)),
            'n_essential': int((y == 1).sum()),
            'n_nonessential': int((y == 0).sum())}


def train_one_fold(model, train_batches, n_epochs, lr=LR, verbose=True):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=LAMBDA_HIGH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    losses = []
    for ep in range(n_epochs):
        wd = cosine_lambda(ep, n_epochs, LAMBDA_HIGH, LAMBDA_LOW)
        for pg in opt.param_groups: pg['weight_decay'] = wd
        model.train()
        ep_losses = []
        for batch in train_batches:
            opt.zero_grad()
            out = model(batch, store_memory=True)
            rl = pairwise_ranking_loss(out['essentiality'], batch['labels'],
                                         batch['has_label'],
                                         n_pairs=RANKING_N_PAIRS,
                                         margin=RANKING_MARGIN)
            bce = weighted_bce(out['essentiality'], batch['labels'],
                                batch['has_label'])
            reg = regulator_proxy_loss(out['regulator_proxy'],
                                         batch['regulatory'],
                                         batch['regulatory_mask'])
            loss = 1.0 * rl + 0.1 * bce + 0.1 * reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_losses.append(loss.item())
        sched.step()
        losses.append(float(np.mean(ep_losses)))
        if verbose and (ep + 1) % 10 == 0:
            print(f'    ep {ep+1:3d}: loss={losses[-1]:.4f}  '
                  f'wd={wd:.5f}  mem={model.memory.memory_size}')
    return losses


def build_model(cfg_kwargs, device):
    cfg = LNNConfig(**cfg_kwargs)
    model = EssentialityLNN(cfg).to(device)
    # Replace memory with growable bank
    model.memory = HyperbolicMemoryBank(
        hidden=cfg.hidden, memory_size=cfg.memory_size,
        retrieval_k=cfg.memory_retrieval_k,
        curvature=cfg.memory_curvature,
        growable=True, max_size=8192,
    ).to(device)
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', action='store_true',
                    help='use SequenceEncoder (CNN over raw nucleotides)')
    ap.add_argument('--epochs', type=int, default=60,
                    help='training epochs per fold')
    ap.add_argument('--holdout', default=None,
                    help='single organism to hold out (default: all 6 in turn)')
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[setup] device={device}, seed={SEED}, '
          f'mode={"sequence" if args.seq else "kmer"}, '
          f'epochs/fold={args.epochs}')

    print(f'\n[1] loading data for all {len(ALL_ORGS)} organisms...')
    all_batches = load_organism_batches(ALL_ORGS, verbose=True)
    for org, b in all_batches.items():
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                all_batches[org][k] = v.to(device)

    cfg_kwargs = dict(
        hidden=128, n_liquid_steps=3, memory_size=1024,
        memory_retrieval_k=8, dropout=0.1,
        use_sequence_encoder=args.seq,
        seq_max_len=2048,
    )

    holdout_list = ([args.holdout] if args.holdout else HOLDOUT_CANDIDATES)
    print(f'\n[2] running leave-one-org-out across {len(holdout_list)} folds '
          f'(always in training: {ALWAYS_IN_TRAINING})...')

    fold_results = {}
    for fold_i, held_out in enumerate(holdout_list):
        train_orgs = [o for o in HOLDOUT_CANDIDATES if o != held_out] + ALWAYS_IN_TRAINING
        print(f'\n  [fold {fold_i + 1}/{len(holdout_list)}] held-out = {held_out}, '
              f'train_orgs = {train_orgs}')
        # Fresh model per fold
        torch.manual_seed(SEED + fold_i)
        np.random.seed(SEED + fold_i)
        model, cfg = build_model(cfg_kwargs, device)
        train_batches = [all_batches[o] for o in train_orgs]
        t0 = time.time()
        losses = train_one_fold(model, train_batches, args.epochs)
        train_time = time.time() - t0
        # Evaluate on held-out
        e = evaluate_one(model, all_batches[held_out], name=held_out)
        e['train_time_s'] = train_time
        e['final_loss'] = losses[-1]
        # Also evaluate on training organisms (sanity / overfit check)
        train_evals = {o: evaluate_one(model, all_batches[o], name=o)['mcc']
                        for o in train_orgs}
        e['train_org_mccs'] = train_evals
        fold_results[held_out] = e
        print(f'    test MCC={e["mcc"]:.4f}  TP={e["tp"]} FP={e["fp"]} '
              f'TN={e["tn"]} FN={e["fn"]}  ({train_time:.1f}s)')
        print(f'    train-org MCCs: ' +
              '  '.join(f'{k}={v:.3f}' for k, v in train_evals.items()))

    # Summary
    print('\n[3] summary:')
    print(f'  {"held_out":<15s} {"MCC":>7s} {"TP":>4s} {"FP":>4s} {"TN":>4s} {"FN":>4s}')
    print('  ' + '-' * 50)
    for org, e in fold_results.items():
        print(f'  {org:<15s} {e["mcc"]:>7.4f} '
              f'{e["tp"]:>4d} {e["fp"]:>4d} {e["tn"]:>4d} {e["fn"]:>4d}')
    test_mccs = [e['mcc'] for e in fold_results.values()]
    print(f'  {"mean test":<15s} {np.mean(test_mccs):>7.4f} (std {np.std(test_mccs):.3f})')

    out = {
        'method': ('leave_one_org_out_lnn_with_seq_encoder'
                    if args.seq else 'leave_one_org_out_lnn_kmer'),
        'mode': 'sequence' if args.seq else 'kmer',
        'epochs_per_fold': args.epochs,
        'always_in_training': ALWAYS_IN_TRAINING,
        'holdouts_evaluated': holdout_list,
        'config': cfg_kwargs,
        'fold_results': fold_results,
        'mean_test_mcc': float(np.mean(test_mccs)),
        'std_test_mcc': float(np.std(test_mccs)),
        'baselines': {
            'v15_within_organism_syn3a': 0.5372,
            'v25_gb_cross_org_syn3a': 0.2270,
            'v29_gb_with_conservation_syn3a': 0.2438,
            'v32_lnn_phase1_in_domain_syn3a': 0.5347,
            'v33_committee_v15_alone': 0.5050,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
