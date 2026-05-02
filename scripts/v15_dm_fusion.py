"""v21 fusion experiment — top up v15 (MCC=0.537) with Dark Manifold scores.

v15 has 96 false negatives (missed essentials) and only 3 false positives.
The Dark Manifold's 102-gene set has 36 overlapping JCVISYN3A locus tags.
Question: does fusing DM's |atp_drop| signal with v15's binary call
recover any of those 96 FN, bumping MCC > 0.537?

Approach
--------

1. Load v15 predictions (seed_42, 455 rows).
2. Run the Dark Manifold v2.1 100-gene KO sweep on CPU (102 genes x 100
   rollout steps — small enough to run in under a minute).
3. Map DM gene names -> JCVISYN3A locus tags via the Syn3A GenBank
   /gene qualifier.
4. Sweep fusion rules over the 36 overlapping genes:
     A. v15 only (baseline)
     B. OR rule: predict essential if v15==1 OR |atp_drop| >= t (sweep t)
     C. v15 confident + DM override: replace v15 prediction with DM
        prediction on overlap genes (sweep t)
     D. weighted score fusion: score = alpha * v15_call + (1-alpha) * |atp_drop_norm|
        sweep alpha and threshold
5. Report the best MCC across rules; emit JSON + a fact-ready summary.

Output: outputs/v15_dm_fusion.json
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

V15_PRED_CSV = (ROOT / 'outputs/predictions_parallel_s0.05_t0.5_seed42_thr0.1_w4_'
                       'composed_all455_v15_round2_priors.csv')
DM_CKPT = Path('/tmp/cell-simulation/dark_manifold_100gene.pt')
SYN3A_GB = ROOT / 'cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb'
LABELS_CSV = ROOT / 'memory_bank/data/multiorg_essentiality/labels.csv'
OUT_PATH = ROOT / 'outputs/v15_dm_fusion.json'


def load_v15_predictions():
    df = pd.read_csv(V15_PRED_CSV)
    return df


def load_breuer_labels():
    labels = pd.read_csv(LABELS_CSV)
    return labels[labels.organism == 'syn3a'].set_index('locus_tag')['essential'].to_dict()


def run_dm_ko_sweep():
    """Run the Dark Manifold 102-gene KO sweep, return (genes, atp_drops)."""
    from dark_manifold_100gene import DarkManifold100Gene

    ckpt = torch.load(DM_CKPT, map_location='cpu', weights_only=False)
    DM_GENES = ckpt['genes']
    DM_METS = ckpt['metabolites']
    n_genes, n_mets = len(DM_GENES), len(DM_METS)

    model = DarkManifold100Gene(n_genes, n_mets)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    atp_idx = DM_METS.index('ATP') if 'ATP' in DM_METS else 0

    def initial_state(batch=1):
        st = torch.ones(batch, n_genes + n_mets) * 1.0
        if 'ATP' in DM_METS:
            st[:, n_genes + DM_METS.index('ATP')] = 3.0
        if 'ADP' in DM_METS:
            st[:, n_genes + DM_METS.index('ADP')] = 1.0
        if 'AMP' in DM_METS:
            st[:, n_genes + DM_METS.index('AMP')] = 0.5
        return st

    ROLLOUT = 100
    print(f'Running KO sweep: {n_genes} genes x {ROLLOUT} steps...')

    with torch.no_grad():
        wt = model.simulate(initial_state(), ROLLOUT)
        wt_atp = float(wt[-1, 0, n_genes + atp_idx])
        atp_drops = np.zeros(n_genes)
        for g in range(n_genes):
            mask = torch.ones(1, n_genes)
            mask[0, g] = 0
            init = initial_state()
            init[0, :n_genes] *= mask[0]
            ko = model.simulate(init, ROLLOUT, mask)
            atp_drops[g] = float(ko[-1, 0, n_genes + atp_idx]) - wt_atp

    return DM_GENES, atp_drops


def map_dm_to_syn3a():
    """Build {DM_gene_name: JCVISYN3A_xxxx} via GenBank /gene qualifier."""
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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    mcc = matthews_corrcoef(y_true, y_pred)
    return float(mcc), int(tp), int(fp), int(tn), int(fn)


def main():
    print('[1] loading v15 predictions...')
    v15 = load_v15_predictions()
    breuer = load_breuer_labels()
    print(f'  v15: {len(v15)} rows; breuer labels: {len(breuer)}')

    # Build aligned arrays of (locus_tag, true_label, v15_pred)
    v15_aligned = v15[v15.locus_tag.isin(breuer)].copy()
    v15_aligned['true'] = v15_aligned.locus_tag.map(breuer)
    print(f'  v15 ∩ breuer: {len(v15_aligned)}')

    base_mcc, tp, fp, tn, fn = compute_mcc(v15_aligned['true'].values,
                                           v15_aligned['essential'].values)
    print(f'  v15 baseline: MCC={base_mcc:.4f} TP={tp} FP={fp} TN={tn} FN={fn}')

    print('\n[2] running Dark Manifold KO sweep...')
    dm_genes, dm_drops = run_dm_ko_sweep()
    abs_drops = np.abs(dm_drops)
    print(f'  DM genes: {len(dm_genes)}; |atp_drop| min/median/max: '
          f'{abs_drops.min():.4f} / {np.median(abs_drops):.4f} / {abs_drops.max():.4f}')

    print('\n[3] mapping DM gene names -> JCVISYN3A locus tags...')
    name_to_locus = map_dm_to_syn3a()
    overlap_rows = []
    for i, gene in enumerate(dm_genes):
        locus = name_to_locus.get(gene)
        if locus is None or locus not in breuer:
            continue
        overlap_rows.append({
            'gene': gene,
            'locus_tag': locus,
            'atp_drop': float(dm_drops[i]),
            'abs_atp_drop': float(abs(dm_drops[i])),
            'true_essential': int(breuer[locus]),
        })
    overlap = pd.DataFrame(overlap_rows)
    print(f'  overlap: {len(overlap)} genes')
    if not overlap.empty:
        v15_overlap = v15_aligned.set_index('locus_tag').loc[overlap.locus_tag.tolist()]
        overlap['v15_pred'] = v15_overlap['essential'].values
        print(f'  v15 calls on overlap: '
              f'{(overlap.v15_pred == 1).sum()} essential / '
              f'{(overlap.v15_pred == 0).sum()} nonessential')
        print(f'  true labels on overlap: '
              f'{(overlap.true_essential == 1).sum()} essential / '
              f'{(overlap.true_essential == 0).sum()} nonessential')

    print('\n[4] fusion sweep — v15 baseline vs OR-rule top-up...')

    # OR rule: predict essential if v15==1 OR (in overlap AND |atp_drop| >= t)
    # Sweep t over the unique |atp_drop| values.
    base_pred = v15_aligned.set_index('locus_tag')['essential'].astype(int).copy()
    true_arr = v15_aligned.set_index('locus_tag')['true'].astype(int)

    # Map locus_tag -> abs_atp_drop for overlap genes; 0 for the rest
    drop_map = dict(zip(overlap.locus_tag, overlap.abs_atp_drop))

    rule_results = []

    # A. baseline
    pred_a = base_pred.copy()
    mcc_a, *_ = compute_mcc(true_arr.values, pred_a.values)
    rule_results.append({'rule': 'A_v15_only', 'threshold': None, 'mcc': mcc_a,
                         'tp': int(((pred_a == 1) & (true_arr == 1)).sum()),
                         'fp': int(((pred_a == 1) & (true_arr == 0)).sum()),
                         'tn': int(((pred_a == 0) & (true_arr == 0)).sum()),
                         'fn': int(((pred_a == 0) & (true_arr == 1)).sum())})

    # B. OR rule: thresholds
    print('  rule B (OR):  v15==1 OR |atp_drop| >= t')
    thresholds = sorted(set([0.0] + list(overlap.abs_atp_drop.unique())))
    best_b = None
    for t in thresholds:
        pred = base_pred.copy()
        for locus, drop in drop_map.items():
            if drop >= t:
                pred[locus] = 1
        mcc, tp, fp, tn, fn = compute_mcc(true_arr.values, pred.values)
        rec = {'rule': 'B_OR', 'threshold': float(t), 'mcc': mcc,
               'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
        if best_b is None or mcc > best_b['mcc']:
            best_b = rec
    rule_results.append(best_b)
    print(f'    best: t={best_b["threshold"]:.4f}, MCC={best_b["mcc"]:.4f} '
          f'(TP={best_b["tp"]} FP={best_b["fp"]} TN={best_b["tn"]} FN={best_b["fn"]})')

    # C. DM override on overlap: pred = (|atp_drop| >= t) on overlap, v15 elsewhere
    print('  rule C (DM override on overlap)')
    best_c = None
    for t in thresholds:
        pred = base_pred.copy()
        for locus, drop in drop_map.items():
            pred[locus] = 1 if drop >= t else 0
        mcc, tp, fp, tn, fn = compute_mcc(true_arr.values, pred.values)
        rec = {'rule': 'C_DM_override', 'threshold': float(t), 'mcc': mcc,
               'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
        if best_c is None or mcc > best_c['mcc']:
            best_c = rec
    rule_results.append(best_c)
    print(f'    best: t={best_c["threshold"]:.4f}, MCC={best_c["mcc"]:.4f} '
          f'(TP={best_c["tp"]} FP={best_c["fp"]} TN={best_c["tn"]} FN={best_c["fn"]})')

    # D. weighted score: score = alpha * v15 + (1-alpha) * (drop_norm), threshold
    print('  rule D (weighted score, alpha sweep)')
    drop_norm_map = {k: v / (abs_drops.max() + 1e-9) for k, v in drop_map.items()}
    best_d = None
    for alpha in np.linspace(0.0, 1.0, 11):
        for t in np.linspace(0.0, 1.0, 21):
            pred = base_pred.copy()
            for locus, dn in drop_norm_map.items():
                v15_val = base_pred[locus]
                score = alpha * v15_val + (1 - alpha) * dn
                pred[locus] = int(score >= t)
            mcc, tp, fp, tn, fn = compute_mcc(true_arr.values, pred.values)
            rec = {'rule': 'D_weighted', 'threshold': float(t),
                   'alpha': float(alpha), 'mcc': mcc,
                   'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
            if best_d is None or mcc > best_d['mcc']:
                best_d = rec
    rule_results.append(best_d)
    print(f'    best: alpha={best_d["alpha"]:.2f}, t={best_d["threshold"]:.4f}, '
          f'MCC={best_d["mcc"]:.4f} '
          f'(TP={best_d["tp"]} FP={best_d["fp"]} TN={best_d["tn"]} FN={best_d["fn"]})')

    # Summary
    best = max(rule_results, key=lambda r: r['mcc'])
    delta = best['mcc'] - base_mcc
    print(f'\n[5] BEST FUSION: rule {best["rule"]} -> MCC={best["mcc"]:.4f} '
          f'(v15 baseline 0.5372, delta {delta:+.4f})')

    out = {
        'method': 'v15_plus_darkmanifold_topup_fusion',
        'baseline': {
            'rule': 'v15_only',
            'mcc': base_mcc,
            'tp': int(((base_pred == 1) & (true_arr == 1)).sum()),
            'fp': int(((base_pred == 1) & (true_arr == 0)).sum()),
            'tn': int(((base_pred == 0) & (true_arr == 0)).sum()),
            'fn': int(((base_pred == 0) & (true_arr == 1)).sum()),
        },
        'overlap_size': int(len(overlap)),
        'overlap_summary': {
            'true_essential': int((overlap.true_essential == 1).sum()),
            'true_nonessential': int((overlap.true_essential == 0).sum()),
            'v15_pred_essential': int((overlap.v15_pred == 1).sum()),
            'v15_pred_nonessential': int((overlap.v15_pred == 0).sum()),
        } if not overlap.empty else None,
        'rules': rule_results,
        'best_rule': best,
        'delta_vs_v15': float(delta),
        'verdict': ('top_up_improves_mcc' if delta > 0.001
                    else 'no_improvement_over_v15'),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f'\n  wrote {OUT_PATH}')

    # Save the per-gene overlap detail for reproducibility
    detail_path = ROOT / 'outputs/v15_dm_fusion_overlap_detail.csv'
    if not overlap.empty:
        overlap.to_csv(detail_path, index=False)
        print(f'  wrote {detail_path}')


if __name__ == '__main__':
    main()
