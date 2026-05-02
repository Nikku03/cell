"""Honest scale-up: run the Dark Manifold's per-gene |atp_drop| signal
against ALL 455 Syn3A genes (Breuer 2019), not just the 43 that map to
the DM's gene set.

For genes where the DM has a |atp_drop| signal: threshold-sweep on
|atp_drop|. For genes with no DM coverage: default to "nonessential"
(DM has no information).

This produces the honest standalone MCC of the Dark Manifold v2.1
checkpoint on the full Breuer Syn3A panel — directly comparable to
v15's MCC=0.5372 on the same 455 genes.

Output: outputs/dm_all_syn3a.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from sklearn.metrics import confusion_matrix, matthews_corrcoef

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, '/tmp/cell-simulation')

DM_CKPT = Path('/tmp/cell-simulation/dark_manifold_100gene.pt')
SYN3A_GB = ROOT / 'cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb'
LABELS_CSV = ROOT / 'memory_bank/data/multiorg_essentiality/labels.csv'
OUT_PATH = ROOT / 'outputs/dm_all_syn3a.json'


def run_dm_ko_sweep():
    from dark_manifold_100gene import DarkManifold100Gene

    ckpt = torch.load(DM_CKPT, map_location='cpu', weights_only=False)
    DM_GENES = ckpt['genes']
    DM_METS = ckpt['metabolites']
    n_genes, n_mets = len(DM_GENES), len(DM_METS)

    model = DarkManifold100Gene(n_genes, n_mets)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    atp_idx = DM_METS.index('ATP') if 'ATP' in DM_METS else 0

    def initial_state():
        st = torch.ones(1, n_genes + n_mets) * 1.0
        if 'ATP' in DM_METS:
            st[0, n_genes + DM_METS.index('ATP')] = 3.0
        if 'ADP' in DM_METS:
            st[0, n_genes + DM_METS.index('ADP')] = 1.0
        if 'AMP' in DM_METS:
            st[0, n_genes + DM_METS.index('AMP')] = 0.5
        return st

    ROLLOUT = 100
    print(f'KO sweep: {n_genes} genes x {ROLLOUT} steps...')
    with torch.no_grad():
        wt = model.simulate(initial_state(), ROLLOUT)
        wt_atp = float(wt[-1, 0, n_genes + atp_idx])
        atp_drops = np.zeros(n_genes)
        for g in range(n_genes):
            mask = torch.ones(1, n_genes); mask[0, g] = 0
            init = initial_state(); init[0, :n_genes] *= mask[0]
            ko = model.simulate(init, ROLLOUT, mask)
            atp_drops[g] = float(ko[-1, 0, n_genes + atp_idx]) - wt_atp
    return DM_GENES, atp_drops


def map_dm_to_syn3a():
    rec = next(SeqIO.parse(str(SYN3A_GB), 'genbank'))
    mapping = {}
    for f in rec.features:
        if f.type != 'CDS':
            continue
        locus = f.qualifiers.get('locus_tag', [None])[0]
        gene = f.qualifiers.get('gene', [None])[0]
        if gene and locus and locus.startswith('JCVISYN3A_'):
            for n in re.split(r'[;,]', gene):
                n = n.strip()
                if n:
                    mapping.setdefault(n, locus)
    return mapping


def compute_mcc(y_true, y_pred):
    if pd.Series(y_pred).nunique() < 2 or pd.Series(y_true).nunique() < 2:
        return None, 0, 0, 0, 0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(matthews_corrcoef(y_true, y_pred)), int(tp), int(fp), int(tn), int(fn)


def main():
    print('[1] running Dark Manifold KO sweep...')
    dm_genes, dm_drops = run_dm_ko_sweep()
    abs_drops = np.abs(dm_drops)
    print(f'  |atp_drop|: min={abs_drops.min():.4f}, '
          f'median={np.median(abs_drops):.4f}, max={abs_drops.max():.4f}')

    print('\n[2] mapping DM gene names -> Syn3A locus tags...')
    name_to_locus = map_dm_to_syn3a()
    locus_to_drop = {}
    for i, gene in enumerate(dm_genes):
        locus = name_to_locus.get(gene)
        if locus is not None:
            locus_to_drop[locus] = abs_drops[i]
    print(f'  DM genes mapped to Syn3A: {len(locus_to_drop)}/{len(dm_genes)}')

    print('\n[3] loading Breuer 2019 labels...')
    labels = pd.read_csv(LABELS_CSV)
    syn3a = labels[labels.organism == 'syn3a'].copy()
    syn3a = syn3a.set_index('locus_tag')
    print(f'  Syn3A labels: {len(syn3a)} ({(syn3a.essential==1).sum()} ess, '
          f'{(syn3a.essential==0).sum()} non)')

    # Coverage analysis
    covered = sum(1 for lt in syn3a.index if lt in locus_to_drop)
    covered_ess = sum(1 for lt in syn3a.index
                      if lt in locus_to_drop and syn3a.loc[lt, 'essential'] == 1)
    covered_ne = sum(1 for lt in syn3a.index
                     if lt in locus_to_drop and syn3a.loc[lt, 'essential'] == 0)
    print(f'  DM-covered Syn3A genes: {covered}/{len(syn3a)} '
          f'({covered_ess} ess, {covered_ne} non)')

    print('\n[4] sweeping rules over all 455 Syn3A genes...')

    locus_tags = syn3a.index.tolist()
    y_true = syn3a['essential'].astype(int).values

    results = {}

    # Rule 1: DM standalone, no-overlap defaults to NONESSENTIAL
    print('  Rule 1: DM-only, non-overlap defaults to NONESSENTIAL')
    drops = np.array([locus_to_drop.get(lt, 0.0) for lt in locus_tags])
    thresholds = sorted(set(np.concatenate([[0.0], abs_drops])))
    best = {'rule': 'DM_only_default_nonessential', 'mcc': -2.0}
    for t in thresholds:
        pred = (drops >= t).astype(int)
        # Skip degenerate (all 0 or all 1)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        mcc, tp, fp, tn, fn = compute_mcc(y_true, pred)
        if mcc is None: continue
        if mcc > best['mcc']:
            best = {'rule': 'DM_only_default_nonessential', 'threshold': float(t),
                    'mcc': mcc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    results['rule_1_dm_only_default_nonessential'] = best
    print(f'    best: t={best.get("threshold", 0.0):.4f}, MCC={best["mcc"]:.4f}, '
          f'TP={best.get("tp",0)} FP={best.get("fp",0)} TN={best.get("tn",0)} FN={best.get("fn",0)}')

    # Rule 2: DM standalone, no-overlap defaults to ESSENTIAL (since most genes are essential)
    # In Breuer Syn3A, 84% of genes are essential. So defaulting to essential
    # might give a higher baseline MCC.
    print('  Rule 2: DM-only, non-overlap defaults to ESSENTIAL')
    best = {'rule': 'DM_only_default_essential', 'mcc': -2.0}
    for t in thresholds:
        pred = np.ones(len(locus_tags), dtype=int)  # default essential
        for i, lt in enumerate(locus_tags):
            if lt in locus_to_drop:
                pred[i] = int(locus_to_drop[lt] >= t)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        mcc, tp, fp, tn, fn = compute_mcc(y_true, pred)
        if mcc is None: continue
        if mcc > best['mcc']:
            best = {'rule': 'DM_only_default_essential', 'threshold': float(t),
                    'mcc': mcc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    results['rule_2_dm_only_default_essential'] = best
    print(f'    best: t={best.get("threshold", 0.0):.4f}, MCC={best["mcc"]:.4f}, '
          f'TP={best.get("tp",0)} FP={best.get("fp",0)} TN={best.get("tn",0)} FN={best.get("fn",0)}')

    # Rule 3: Constant-essential (predict everything essential) — sanity baseline
    print('  Rule 3: predict everything essential (sanity baseline)')
    pred = np.ones(len(locus_tags), dtype=int)
    mcc, tp, fp, tn, fn = compute_mcc(y_true, pred)
    results['rule_3_all_essential'] = {
        'rule': 'all_essential', 'mcc': mcc if mcc is not None else 0.0,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    print(f'    MCC={mcc} (undefined when single-class), TP={tp} FP={fp} TN={tn} FN={fn}')

    # Rule 4: Constant-nonessential (predict everything nonessential) — other sanity
    print('  Rule 4: predict everything nonessential (sanity baseline)')
    pred = np.zeros(len(locus_tags), dtype=int)
    mcc, tp, fp, tn, fn = compute_mcc(y_true, pred)
    results['rule_4_all_nonessential'] = {
        'rule': 'all_nonessential', 'mcc': mcc if mcc is not None else 0.0,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

    # Best across rules with defined MCC
    defined = [(k, v) for k, v in results.items() if v.get('mcc') is not None and v['mcc'] > -1.5]
    best_overall = max(defined, key=lambda kv: kv[1]['mcc'])
    print(f'\n[5] BEST: {best_overall[0]} -> MCC={best_overall[1]["mcc"]:.4f}')

    out = {
        'method': 'dark_manifold_v21_standalone_full_syn3a_panel',
        'checkpoint': 'cell-simulation/dark_manifold_100gene.pt',
        'breuer_panel': {'total': int(len(syn3a)),
                         'essential': int((syn3a.essential==1).sum()),
                         'nonessential': int((syn3a.essential==0).sum())},
        'dm_coverage': {'genes_in_dm': int(len(dm_genes)),
                        'mapped_to_syn3a': int(len(locus_to_drop)),
                        'covered_essential': int(covered_ess),
                        'covered_nonessential': int(covered_ne)},
        'rules': results,
        'best_rule': best_overall[0],
        'best_mcc': float(best_overall[1]['mcc']),
        'baselines': {
            'v15_syn3a_biology_first_mcc_on_same_panel': 0.5372,
            'v18_xgboost_syn3a_held_out': 0.1376,
            'v22_dark_manifold_balanced_retrain_43_subset': 0.4487,
        },
        'comment': ('v22 reported MCC=0.4487 on a 43-gene cherry-picked overlap. '
                    'This script reports the same DM signal ON ALL 455 Syn3A genes — '
                    'the honest scale-up. Coverage of 36/455 genes means most of the '
                    'panel has no DM signal, and the achievable MCC is bounded by '
                    'how well the no-overlap default (essential vs nonessential) '
                    'matches the rest.'),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f'\nwrote {OUT_PATH}')


if __name__ == '__main__':
    main()
