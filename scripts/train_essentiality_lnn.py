"""End-to-end training pipeline for the EssentialityLNN.

Three phases:

  Phase 1 — Joint training on Salmonella + Syn3A
    These are the two organisms with rich annotation: Salmonella has
    RegulonDB-derived /regulatory and /protein_bind features; Syn3A
    has the most curated essentiality from Breuer 2019 + Luthey-Schulten
    kinetic_params (placeholder enriched later). Train the LNN with
    class-weighted BCE + auxiliary regulator-proxy loss (where labeled).

  Phase 2 — Continual learning on other organisms
    For each of the 4 remaining-essentiality organisms (Caulobacter,
    S. aureus, M. tb, A. baylyi, plus M. pneumoniae), feed the genome
    through the LNN ONCE per epoch with EWC penalty preserving Phase 1's
    weights. Also: hyperbolic memory bank accumulates representations
    across organisms.

  Phase 3 — Evaluation per organism
    With the final model + accumulated memory, run inference on each
    organism and report MCC + diagnostics. Specifically: Syn3A test MCC
    (compare to v25's 0.227 / v29's 0.244 cross-organism ceiling) AND
    Syn3A within-organism MCC (Syn3A was in training, so this is in-
    distribution).

Output: outputs/essentiality_lnn_results.json
        cell_sim/layer_ml/checkpoints/essentiality_lnn.pt
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make the cell repo root importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from cell_sim.layer_ml.essentiality_lnn import EssentialityLNN, LNNConfig
from cell_sim.layer_ml.continual_learning import EWCRegularizer
from cell_sim.layer_ml.data_loader import load_organism_batches

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/essentiality_lnn_results.json'
CHECKPOINT = ROOT / 'cell_sim/layer_ml/checkpoints/essentiality_lnn.pt'

PHASE_1_ORGS = ['styphimurium', 'syn3a']
PHASE_2_ORGS = ['ccrescentus', 'saureus', 'mtuberculosis', 'abaylyi', 'mpne']
ALL_ORGS = PHASE_1_ORGS + PHASE_2_ORGS

PHASE_1_EPOCHS = 80
PHASE_2_EPOCHS_PER_ORG = 12   # short Phase 2 to limit forgetting under EWC
EWC_LAMBDA = 1500.0           # stronger EWC penalty - first run at 200 forgot
LR = 1e-3
SEED = 42


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
    """Return (w_class_0, w_class_1) inversely proportional to class frequency."""
    n = y.numel()
    n1 = float(y.sum().item())
    n0 = float(n - n1)
    return float(n1 / n), float(n0 / n)


def weighted_bce(logits: torch.Tensor, target: torch.Tensor,
                  has_label: torch.Tensor) -> torch.Tensor:
    """Class-weighted BCE; only applied to genes with has_label=1."""
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


def regulator_proxy_loss(reg_logits: torch.Tensor,
                          reg_features: torch.Tensor,
                          reg_mask: torch.Tensor) -> torch.Tensor:
    """Auxiliary loss: regulator_proxy_head should predict the
    regulatory_class presence (RBS / -10 / -35) from the gene's hidden
    state. Targets come from the v27 PWM-derived `_present` flags in the
    regulatory feature vector. Indices in the regulatory tensor (after
    the build_regulatory_features layout):
      RBS_present     = column index 1
      minus_10_present = column index 4
      minus_35_present = column index 7
    Only applied where reg_mask > 0."""
    mask = reg_mask > 0.5
    if mask.sum() == 0:
        return torch.tensor(0.0, device=reg_logits.device)
    # Pull the three present-flags by index. If the columns aren't in the
    # canonical positions, fall back to zeros (loss becomes ~0).
    if reg_features.size(1) >= 8:
        targets = torch.stack([reg_features[:, 1],
                                 reg_features[:, 4],
                                 reg_features[:, 7]], dim=-1)
    else:
        targets = torch.zeros(reg_features.size(0), 3, device=reg_logits.device)
    targets = targets[mask]
    pred = reg_logits[mask]
    return F.binary_cross_entropy_with_logits(pred, targets)


def evaluate_one(model, batch, name='', t_override=None):
    """Run inference on one organism, return MCC + confusion."""
    model.eval()
    with torch.no_grad():
        out = model(batch, store_memory=False)
    probs = torch.sigmoid(out['essentiality']).cpu().numpy()
    y = batch['labels'].cpu().numpy().astype(int)
    has = batch['has_label'].cpu().numpy() > 0
    probs = probs[has]; y = y[has]
    if t_override is not None:
        best_t = t_override
    else:
        best_t, _ = threshold_search(y, probs)
    pred = (probs >= best_t).astype(int)
    if len(set(pred)) > 1 and len(set(y)) > 1:
        mcc = matthews_corrcoef(y, pred)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    else:
        mcc = 0.0; tp = fp = tn = fn = 0
    return {'organism': name, 'mcc': float(mcc), 'threshold': float(best_t),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'n_genes': int(len(y)),
            'n_essential': int((y == 1).sum()),
            'n_nonessential': int((y == 0).sum()),
            'probs_range': (float(probs.min()), float(probs.max()))}


def train_epoch(model, opt, batches, ewc=None, lambda_ewc=0.0,
                  reg_proxy_weight: float = 0.1):
    """One training epoch over a list of organism batches. Each batch is
    one organism's genes (batch size = N_genes_in_organism)."""
    model.train()
    losses = []
    for batch in batches:
        opt.zero_grad()
        out = model(batch, store_memory=True)
        ess_loss = weighted_bce(out['essentiality'],
                                  batch['labels'], batch['has_label'])
        reg_loss = regulator_proxy_loss(out['regulator_proxy'],
                                          batch['regulatory'],
                                          batch['regulatory_mask'])
        ewc_pen = (ewc.penalty(model, lambda_ewc=lambda_ewc)
                    if ewc is not None and lambda_ewc > 0 else 0.0)
        loss = ess_loss + reg_proxy_weight * reg_loss + ewc_pen
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def make_consolidate_loss_fn(reg_proxy_weight: float = 0.1):
    """Return a loss function for EWC consolidation that uses the same
    structure as training (essentiality + regulator-proxy)."""
    def loss_fn(model, batch):
        out = model(batch, store_memory=False)
        return (weighted_bce(out['essentiality'],
                                batch['labels'], batch['has_label'])
                + reg_proxy_weight * regulator_proxy_loss(
                    out['regulator_proxy'],
                    batch['regulatory'],
                    batch['regulatory_mask']))
    return loss_fn


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[setup] device={device}, seed={SEED}')

    print('\n[1] loading data for all organisms...')
    all_batches = load_organism_batches(ALL_ORGS)
    # Move to device
    for org, b in all_batches.items():
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                all_batches[org][k] = v.to(device)

    print('\n[2] building EssentialityLNN...')
    cfg = LNNConfig(
        hidden=128, n_liquid_steps=3, memory_size=512,
        memory_retrieval_k=8, dropout=0.1,
    )
    model = EssentialityLNN(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  params: {n_params:,}  hidden=128  steps=3  memory=512')

    print('\n[3] PHASE 1: joint training on Salmonella + Syn3A '
          f'({PHASE_1_EPOCHS} epochs)...')
    phase1_batches = [all_batches[o] for o in PHASE_1_ORGS if o in all_batches]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, PHASE_1_EPOCHS)
    t0 = time.time()
    p1_losses = []
    for ep in range(PHASE_1_EPOCHS):
        l = train_epoch(model, opt, phase1_batches, ewc=None, lambda_ewc=0.0)
        sched.step()
        p1_losses.append(l)
        if (ep + 1) % 10 == 0:
            print(f'  ep {ep+1:3d}: loss={l:.4f}  '
                  f'memory_n={model.memory.n_populated}')
    print(f'  Phase 1 trained in {time.time()-t0:.1f}s')

    # Snapshot Phase 1 evaluations before continual learning
    phase1_eval = {}
    for org in ALL_ORGS:
        if org in all_batches:
            phase1_eval[org] = evaluate_one(model, all_batches[org], name=org)
    print('\n  Phase 1 evals:')
    for org, e in phase1_eval.items():
        print(f'    {org:<15s}  MCC={e["mcc"]:.4f}  '
              f'TP={e["tp"]} FP={e["fp"]} TN={e["tn"]} FN={e["fn"]}  '
              f'(t={e["threshold"]:.2f})')

    print(f'\n[4] PHASE 2: continual learning on {len(PHASE_2_ORGS)} other '
          f'organisms with EWC (lambda={EWC_LAMBDA})...')
    ewc = EWCRegularizer()
    consolidate_loss = make_consolidate_loss_fn(reg_proxy_weight=0.1)

    # Consolidate Phase 1 (Salmonella + Syn3A) before continual learning
    print('  consolidating Phase 1 weights...')
    ewc.consolidate(model, phase1_batches, n_samples=len(phase1_batches),
                     loss_fn=consolidate_loss)
    fisher_max = max(f.abs().max().item() for _, f in ewc.tasks[0].values())
    print(f'  Fisher max: {fisher_max:.6f}')

    p2_results = {}
    for new_org in PHASE_2_ORGS:
        if new_org not in all_batches: continue
        new_batch = all_batches[new_org]
        # Train on this single organism for PHASE_2_EPOCHS_PER_ORG epochs
        # while EWC penalty preserves all previous-task weights
        t0 = time.time()
        opt2 = torch.optim.AdamW(model.parameters(), lr=LR / 3.0,
                                   weight_decay=1e-4)
        for ep in range(PHASE_2_EPOCHS_PER_ORG):
            l = train_epoch(model, opt2, [new_batch],
                            ewc=ewc, lambda_ewc=EWC_LAMBDA)
        # Consolidate this organism's weights too (so subsequent organisms
        # also can't forget it)
        ewc.consolidate(model, [new_batch], n_samples=1,
                          loss_fn=consolidate_loss)
        # Evaluate
        e = evaluate_one(model, new_batch, name=new_org)
        # Also re-eval previous orgs to monitor catastrophic forgetting
        prev_evals = {}
        for prev in ALL_ORGS:
            if prev != new_org and prev in all_batches:
                prev_evals[prev] = evaluate_one(model, all_batches[prev],
                                                  name=prev)['mcc']
        p2_results[new_org] = {
            'on_new_org': e,
            'prev_orgs_mcc': prev_evals,
            'final_loss': l,
            'wall_s': time.time() - t0,
        }
        print(f'  added {new_org} ({PHASE_2_EPOCHS_PER_ORG} epochs, '
              f'{p2_results[new_org]["wall_s"]:.1f}s):')
        print(f'    new-org MCC = {e["mcc"]:.4f}')
        print(f'    prev-org MCCs: ' +
              '  '.join(f'{k}={v:.3f}' for k, v in prev_evals.items()))

    print('\n[5] PHASE 3: final evaluation per organism...')
    final_evals = {}
    for org in ALL_ORGS:
        if org in all_batches:
            final_evals[org] = evaluate_one(model, all_batches[org], name=org)
            e = final_evals[org]
            print(f'  {org:<15s}  MCC={e["mcc"]:.4f}  '
                  f'TP={e["tp"]} FP={e["fp"]} TN={e["tn"]} FN={e["fn"]}')

    print('\n[6] saving checkpoint...')
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'config': cfg.__dict__,
        'state_dict': model.state_dict(),
        'memory_buffer': model.memory.memory.cpu(),
        'memory_populated': model.memory.populated.cpu(),
        'phase1_eval': phase1_eval,
        'final_evals': final_evals,
    }, CHECKPOINT)
    print(f'  saved {CHECKPOINT}')

    # Compute key headline numbers
    syn3a_phase1 = phase1_eval.get('syn3a', {}).get('mcc', None)
    syn3a_final = final_evals.get('syn3a', {}).get('mcc', None)
    forgetting = (syn3a_phase1 - syn3a_final
                   if syn3a_phase1 is not None and syn3a_final is not None
                   else None)

    out = {
        'method': 'multimodal_lnn_with_hyperbolic_memory_and_ewc_continual',
        'config': cfg.__dict__,
        'n_parameters': n_params,
        'phase_1': {
            'organisms': PHASE_1_ORGS,
            'epochs': PHASE_1_EPOCHS,
            'final_loss': float(p1_losses[-1]),
            'evals': phase1_eval,
        },
        'phase_2': {
            'organisms': PHASE_2_ORGS,
            'epochs_per_org': PHASE_2_EPOCHS_PER_ORG,
            'ewc_lambda': EWC_LAMBDA,
            'per_org_results': p2_results,
        },
        'final_evals': final_evals,
        'syn3a_phase1_mcc': syn3a_phase1,
        'syn3a_final_mcc': syn3a_final,
        'syn3a_forgetting': forgetting,
        'baselines': {
            'v15_within_organism': 0.5372,
            'v25_gb_cross_organism': 0.2270,
            'v29_gb_with_conservation': 0.2438,
            'v26_lnn_naive_cross_organism': 0.1234,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nwrote {OUT}')

    print('\n[scoreboard]')
    print(f'  v15 (within-organism, Syn3A direct):     0.5372')
    print(f'  v25 (cross-organism GB, original):       0.2270')
    print(f'  v29 (+conservation, fine threshold):     0.2438')
    print(f'  v26 (LNN naive cross-organism):          0.1234')
    print(f'  this LNN Syn3A Phase 1 (in-domain):      {syn3a_phase1}')
    print(f'  this LNN Syn3A Phase 3 (after continual): {syn3a_final}')


if __name__ == '__main__':
    main()
