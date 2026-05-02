"""Diagnostic — does the LNN's 99.3% essential prediction on Syn3A reflect
whole-cell aggregation (organism-level context dominating per-gene features)
or per-gene feature similarity (Syn3A's nonessentials look like essentials)?

Method
------

Reproduce the v26 LNN training. Then forward Syn3A genes through the trained
model but INJECT a foreign organism's h_global into the ContextAwareTau
module instead of computing h_global from Syn3A's own genes. If predictions
shift substantially, whole-cell aggregation dominates. If they stay similar,
per-gene features dominate.

Result (from running this script — see commit message + v26b fact)
------------------------------------------------------------------

Whole-cell context modulates CONFIDENCE (%>0.9) but not BINARY DIRECTION.
With low-essential organism context the %>0.9 drops from 95% to 39-51% but
%essential_predicted stays at 96-99.6%. The bottleneck is per-gene feature
similarity, not whole-cell aggregation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from train_lnn_essentiality import (
    extract_per_organism_data, FEATURES_CSV,
    EdgeAwarePassing, ContextAwareTau, GeneEncoder,
)


class GeneLNN_Probe(nn.Module):
    """Same as GeneLNN but accepts an injected foreign h_global."""
    def __init__(self, n_scalar=20, n_kmer3=64, n_kmer4=256,
                 hidden=128, n_steps=4, dropout=0.1):
        super().__init__()
        self.encoder = GeneEncoder(n_scalar, n_kmer3, n_kmer4, hidden)
        self.tau_pred = ContextAwareTau(hidden)
        self.passing = EdgeAwarePassing(hidden, edge_dim=2, dropout=dropout)
        self.n_steps = n_steps; self.dt = 0.2
        self.n_scalar = n_scalar; self.n_kmer3 = n_kmer3
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, X, edge_index, edge_attr, foreign_h_global=None):
        scalar = X[:, :self.n_scalar]
        kmer3 = X[:, self.n_scalar:self.n_scalar + self.n_kmer3]
        kmer4 = X[:, self.n_scalar + self.n_kmer3:]
        h = self.encoder(scalar, kmer3, kmer4)
        h0 = h.clone()
        for _ in range(self.n_steps):
            if foreign_h_global is None:
                h_global = h.mean(dim=0, keepdim=True).expand_as(h)
            else:
                h_global = foreign_h_global.expand_as(h)
            tau = self.tau_pred(h, h_global)
            msg = self.passing(h, edge_index, edge_attr)
            dh = (msg + h0 - h) / tau.unsqueeze(-1)
            h = h + self.dt * dh
        return self.head(h).squeeze(-1)


def main():
    ess_df = pd.read_csv(FEATURES_CSV)
    print('parsing organisms...')
    ALL_ORGS = ['ccrescentus','saureus','mtuberculosis','styphimurium',
                'abaylyi','mpne','syn3a']
    od = {o: extract_per_organism_data(o, ess_df) for o in ALL_ORGS}

    torch.manual_seed(42); np.random.seed(42)
    device = 'cpu'
    model = GeneLNN_Probe(hidden=128, n_steps=4, dropout=0.1).to(device)

    train_orgs = ['ccrescentus','saureus','mtuberculosis','styphimurium','abaylyi','mpne']
    tensors = {}
    for o in ALL_ORGS:
        tensors[o] = {
            'X': torch.tensor(od[o]['X'], dtype=torch.float32, device=device),
            'y': torch.tensor(od[o]['y'], dtype=torch.float32, device=device),
            'edge_index': torch.tensor(od[o]['edge_index'], dtype=torch.long, device=device),
            'edge_attr': torch.tensor(od[o]['edge_attr'], dtype=torch.float32, device=device),
        }

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 80)
    y_all = np.concatenate([od[o]['y'] for o in train_orgs])
    w0 = float((y_all==1).sum()/len(y_all)); w1 = float((y_all==0).sum()/len(y_all))

    print('training...')
    for ep in range(80):
        model.train()
        order = np.random.permutation(len(train_orgs))
        for oi in order:
            o = train_orgs[oi]; d = tensors[o]
            opt.zero_grad()
            logits = model(d['X'], d['edge_index'], d['edge_attr'])
            target = d['y']
            weights = torch.where(target == 1, torch.tensor(w1), torch.tensor(w0))
            loss = (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * weights * 2.0).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()

    model.eval()

    def compute_h_global(o):
        with torch.no_grad():
            d = tensors[o]
            scalar = d['X'][:, :20]; kmer3 = d['X'][:, 20:84]; kmer4 = d['X'][:, 84:]
            h = model.encoder(scalar, kmer3, kmer4)
            return h.mean(dim=0, keepdim=True)

    h_globals = {o: compute_h_global(o) for o in ALL_ORGS}

    syn = tensors['syn3a']
    syn_y = syn['y'].cpu().numpy()
    print(f'\nSyn3A: {int(syn_y.sum())}/{len(syn_y)} essential ({syn_y.mean()*100:.0f}%)')
    print('\nInjecting foreign whole-cell contexts on Syn3A:')
    print(f'{"injected context":<26} {"%ess_pred":>10} {"%>0.9":>8} {"median":>8} {"min":>8} {"MCC":>8}')
    print('-' * 75)
    for ctx_org in ALL_ORGS:
        with torch.no_grad():
            logits = model(syn['X'], syn['edge_index'], syn['edge_attr'],
                            foreign_h_global=h_globals[ctx_org])
            probs = torch.sigmoid(logits).cpu().numpy()
        pred = (probs >= 0.5).astype(int)
        org_ess_frac = od[ctx_org]['y'].mean()
        label = f'{ctx_org} ({org_ess_frac*100:.0f}% ess)'
        mcc = matthews_corrcoef(syn_y, pred) if len(set(pred))>1 else 0.0
        print(f'{label:<26} {pred.mean()*100:>9.1f}% {(probs>0.9).mean()*100:>7.1f}% '
              f'{np.median(probs):>8.3f} {probs.min():>8.3f} {mcc:>8.4f}')

    # Zero context
    with torch.no_grad():
        logits = model(syn['X'], syn['edge_index'], syn['edge_attr'],
                        foreign_h_global=torch.zeros(1, 128, device=device))
        probs = torch.sigmoid(logits).cpu().numpy()
    pred = (probs >= 0.5).astype(int)
    mcc = matthews_corrcoef(syn_y, pred) if len(set(pred))>1 else 0.0
    print(f'{"zero (neutral)":<26} {pred.mean()*100:>9.1f}% {(probs>0.9).mean()*100:>7.1f}% '
          f'{np.median(probs):>8.3f} {probs.min():>8.3f} {mcc:>8.4f}')

    print('\nSanity: each organism with its OWN context')
    print(f'{"organism":<15} {"truth %ess":>11} {"pred %ess":>11} {"%>0.9":>7} {"MCC":>7}')
    print('-' * 55)
    for o in ALL_ORGS:
        with torch.no_grad():
            d = tensors[o]
            logits = model(d['X'], d['edge_index'], d['edge_attr'])
            probs = torch.sigmoid(logits).cpu().numpy()
        pred = (probs >= 0.5).astype(int)
        y = d['y'].cpu().numpy()
        mcc = matthews_corrcoef(y, pred) if len(set(pred))>1 and len(set(y))>1 else 0.0
        print(f'{o:<15} {y.mean()*100:>10.0f}% {pred.mean()*100:>10.0f}% '
              f'{(probs>0.9).mean()*100:>6.0f}% {mcc:>7.4f}')


if __name__ == '__main__':
    main()
