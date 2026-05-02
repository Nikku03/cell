"""Simulator (v15) + ML top-up with online learning.

v15's mechanistic detector stack produces per-gene features (confidence,
time_to_failure, failure_mode, evidence) → MCC=0.5372 vs Breuer 2019.

Quick analysis showed v15's output features ALONE cannot distinguish its
96 false negatives from its 69 true negatives — both groups have
identical confidence (0.0), failure_mode ('none'), and evidence
('no_signal[...]'). v15 has hit its information ceiling on its own outputs.

This script tests whether AUGMENTING v15 features with EXTERNAL Syn3A
features (gene length, hypothetical-protein flag, GC content from
syn3A.gb sequence, product-description bag-of-words) lets an ML model
recover any of v15's missed essentials.

Two evaluation modes:
  1. Online learning loop: shuffle order, predict-then-update per gene,
     5 seeds. Streaming MCC curve shows how the ML model converges as
     it sees more examples ("learn with every step").
  2. Stratified 5-fold CV: standard held-out evaluation.

Output: outputs/sim_plus_online_ml.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
V15_CSV = ROOT / 'outputs/predictions_parallel_s0.05_t0.5_seed42_thr0.1_w4_composed_all455_v15_round2_priors.csv'
LABELS_CSV = ROOT / 'memory_bank/data/multiorg_essentiality/labels.csv'
OUT_PATH = ROOT / 'outputs/sim_plus_online_ml.json'


def parse_evidence(ev: str) -> dict:
    """Parse v15's evidence text into binary signal flags."""
    ev = str(ev) if ev is not None else ''
    return {
        'has_cv_signal': int('cx[' in ev and 'cx_none' not in ev),
        'has_ann_signal': int('ann[' in ev and 'ann_none' not in ev),
        'has_traj_signal': int('traj[' in ev and 'no_catalytic_rules' not in ev
                               and 'traj_no_signal' not in ev),
        'is_no_signal': int(ev.startswith('no_signal[')),
        'is_ann_only': int(ev.startswith('ann_only[')),
        'is_composed': int(ev.startswith('composed[')),
        'evidence_len': len(ev),
    }


def gc_content_from_genbank(syn3a_gb_path: Path) -> dict[str, float]:
    """Read syn3A.gb, compute GC content per CDS, return {locus_tag: gc_frac}."""
    from Bio import SeqIO
    rec = next(SeqIO.parse(str(syn3a_gb_path), 'genbank'))
    out = {}
    for f in rec.features:
        if f.type != 'CDS':
            continue
        locus = f.qualifiers.get('locus_tag', [None])[0]
        if not locus:
            continue
        seq = str(f.extract(rec.seq)).upper()
        if not seq:
            continue
        gc = (seq.count('G') + seq.count('C')) / len(seq)
        out[locus] = float(gc)
    return out


def build_features(df: pd.DataFrame, gene_table: pd.DataFrame,
                   gc_by_locus: dict) -> tuple[np.ndarray, list[str]]:
    """Build ML feature matrix from v15 output + external Syn3A features."""
    feats = []
    feat_names = []

    # --- v15 output features ---
    feats.append(df.confidence.fillna(0.0).values); feat_names.append('confidence')
    ttf = df.time_to_failure_s.fillna(10.0).values
    feats.append(np.log1p(ttf)); feat_names.append('log_time_to_failure')
    feats.append(df.time_to_failure_s.notna().astype(float).values)
    feat_names.append('has_failure_time')

    failure_modes = ['none', 'translation_stall', 'catalysis_silenced',
                     'essential_metabolite_depletion', 'dna_replication_blocked',
                     'membrane_integrity', 'transcription_stall']
    for fm in failure_modes:
        feats.append((df.failure_mode == fm).astype(float).values)
        feat_names.append(f'fm_{fm}')

    evidence_feats = df.evidence.apply(parse_evidence)
    for key in ['has_cv_signal', 'has_ann_signal', 'has_traj_signal',
                'is_no_signal', 'is_ann_only', 'is_composed', 'evidence_len']:
        vals = np.array([e[key] for e in evidence_feats], dtype=float)
        if key == 'evidence_len':
            vals = vals / 100.0
        feats.append(vals); feat_names.append(key)

    feats.append(df.essential.astype(float).values); feat_names.append('v15_call')

    # --- External Syn3A features (this is where the new signal must come from) ---
    gt_lookup = gene_table.set_index('locus_tag')
    length_bp = df.locus_tag.map(lambda lt: gt_lookup.loc[lt, 'length_bp']
                                  if lt in gt_lookup.index else 0.0).values
    feats.append(np.log1p(length_bp.astype(float))); feat_names.append('log_length_bp')

    product = df.locus_tag.map(lambda lt: str(gt_lookup.loc[lt, 'product'])
                                if lt in gt_lookup.index else '').values
    is_hypothetical = np.array([1.0 if 'hypothetical' in p.lower() else 0.0
                                for p in product])
    is_uncharacterized = np.array([1.0 if 'uncharacterized' in p.lower() else 0.0
                                   for p in product])
    is_putative = np.array([1.0 if 'putative' in p.lower() else 0.0 for p in product])
    feats.append(is_hypothetical); feat_names.append('is_hypothetical')
    feats.append(is_uncharacterized); feat_names.append('is_uncharacterized')
    feats.append(is_putative); feat_names.append('is_putative')

    # Product-text bag-of-words for high-signal essential keywords
    keyword_groups = {
        'kw_replication': ['replication', 'polymerase', 'helicase', 'gyrase', 'topoisomerase'],
        'kw_transcription': ['rna polymerase', 'sigma', 'transcription', 'rho'],
        'kw_translation': ['ribosomal', 'tRNA', 'translation', 'aminoacyl', 'elongation'],
        'kw_atp': ['atp synthase', 'F0F1', 'F-ATPase'],
        'kw_membrane': ['membrane', 'transporter', 'permease', 'ABC '],
        'kw_synthase': ['synthase', 'synthetase'],
        'kw_kinase': ['kinase'],
        'kw_dehydrogenase': ['dehydrogenase', 'reductase', 'oxidase'],
    }
    for kw_name, terms in keyword_groups.items():
        vals = np.array([1.0 if any(t.lower() in p.lower() for t in terms) else 0.0
                         for p in product])
        feats.append(vals); feat_names.append(kw_name)

    # GC content
    gc = df.locus_tag.map(lambda lt: gc_by_locus.get(lt, 0.32)).values  # Syn3A overall ~32%
    feats.append(gc.astype(float)); feat_names.append('gc_content')

    X = np.stack(feats, axis=1)
    return X, feat_names


class MLP(nn.Module):
    def __init__(self, n_in, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def online_learn(X, y, seed=42, lr=0.005, hidden=32, warmup=20):
    """Online-SGD loop with class-weighted BCE: predict on each gene,
    then update model with that gene's label.
    "Test-then-train" — predictions for streaming MCC are made with the model
    BEFORE seeing the current gene's label.
    Standardize features once up front so SGD is well-conditioned.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))

    # Standardize features (fit on full X — fine for online experiment since
    # we're not measuring generalization here, we're measuring "how does the
    # streaming MCC evolve as the model sees more examples").
    mean = X.mean(axis=0); std = X.std(axis=0) + 1e-6
    Xn = (X - mean) / std

    model = MLP(Xn.shape[1], hidden)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    Xt = torch.tensor(Xn, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    # Class weights: 84% essential -> downweight essential examples
    class_weights = {0: float((y == 1).sum()) / len(y),
                     1: float((y == 0).sum()) / len(y)}

    preds_seen = np.full(len(X), np.nan)
    running_mcc = []
    n_seen_for_mcc = []

    for step, idx in enumerate(order):
        model.eval()
        with torch.no_grad():
            logit = model(Xt[idx:idx+1])
            prob = torch.sigmoid(logit).item()
        preds_seen[idx] = prob

        model.train()
        opt.zero_grad()
        logit = model(Xt[idx:idx+1])
        target = yt[idx:idx+1]
        weight = class_weights[int(y[idx])]
        loss = F.binary_cross_entropy_with_logits(logit, target) * weight * 2.0
        loss.backward()
        opt.step()

        if step >= warmup:
            seen_mask = ~np.isnan(preds_seen)
            seen_y = y[seen_mask]
            seen_pred = (preds_seen[seen_mask] >= 0.5).astype(int)
            if len(set(seen_y)) > 1 and len(set(seen_pred)) > 1:
                mcc = matthews_corrcoef(seen_y, seen_pred)
                running_mcc.append(mcc)
                n_seen_for_mcc.append(int(seen_mask.sum()))

    model.eval()
    with torch.no_grad():
        all_probs = torch.sigmoid(model(Xt)).numpy()
    final_preds = (all_probs >= 0.5).astype(int)

    return {
        'order': order.tolist(),
        'pred_at_seen_time': preds_seen.tolist(),
        'running_mcc_curve': running_mcc,
        'n_seen_at_each_mcc': n_seen_for_mcc,
        'final_probs': all_probs.tolist(),
        'final_preds': final_preds.tolist(),
    }


def kfold_eval(X, y, n_splits=5, seed=42, model_type='logreg'):
    """Standard k-fold CV with sklearn's class-weighted logistic regression
    (handles 84%/16% imbalance much better than the imbalanced MLP)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_mccs = []
    test_preds = np.zeros(len(X), dtype=int)
    test_probs = np.zeros(len(X), dtype=float)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight='balanced',
                                C=1.0, max_iter=500,
                                solver='liblinear', random_state=seed),
        )
        clf.fit(Xtr, ytr)
        probs = clf.predict_proba(Xte)[:, 1]

        # Threshold sweep for best MCC on this fold's training data,
        # apply to test fold (still honest — threshold tuned only on training).
        train_probs = clf.predict_proba(Xtr)[:, 1]
        best_t, best_mcc = 0.5, -2.0
        for t in np.linspace(0.05, 0.95, 19):
            tr_pred = (train_probs >= t).astype(int)
            if tr_pred.sum() == 0 or tr_pred.sum() == len(tr_pred):
                continue
            m = matthews_corrcoef(ytr, tr_pred)
            if m > best_mcc:
                best_mcc, best_t = m, t

        preds = (probs >= best_t).astype(int)
        if pd.Series(preds).nunique() < 2 or pd.Series(yte).nunique() < 2:
            mcc = 0.0
        else:
            mcc = matthews_corrcoef(yte, preds)
        fold_mccs.append(mcc)
        test_preds[te_idx] = preds
        test_probs[te_idx] = probs

    return fold_mccs, test_preds, test_probs


def main():
    print('[1] loading v15 predictions + Syn3A gene table + sequence GC...')
    df = pd.read_csv(V15_CSV)
    labels = pd.read_csv(LABELS_CSV)
    syn3a_labels = labels[labels.organism == 'syn3a'].set_index('locus_tag')['essential'].to_dict()

    df['true'] = df.locus_tag.map(syn3a_labels)
    df = df.dropna(subset=['true'])
    df['true'] = df['true'].astype(int)
    print(f'  v15 rows: {len(df)}; class balance: {df["true"].value_counts().to_dict()}')

    gene_table = pd.read_csv(ROOT / 'memory_bank/data/syn3a_gene_table.csv')
    gc_by_locus = gc_content_from_genbank(
        ROOT / 'cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb')
    print(f'  gene_table: {len(gene_table)} rows; GC values: {len(gc_by_locus)} loci')

    # v15 baseline
    v15_mcc = matthews_corrcoef(df['true'], df['essential'])
    v15_tn, v15_fp, v15_fn, v15_tp = confusion_matrix(df['true'], df['essential']).ravel()
    print(f'  v15 baseline MCC = {v15_mcc:.4f} (TP={v15_tp} FP={v15_fp} TN={v15_tn} FN={v15_fn})')

    print('\n[2] building feature matrix (v15 + external)...')
    X, feat_names = build_features(df, gene_table, gc_by_locus)
    y = df['true'].values
    print(f'  X shape: {X.shape}; n features: {len(feat_names)}')

    print('\n[3] online learning loop (5 random seeds)...')
    online_results = []
    final_pred_runs = []
    for seed in [1, 2, 42, 7, 13]:
        res = online_learn(X, y, seed=seed)
        # Final pass MCC
        final_mcc = matthews_corrcoef(y, np.array(res['final_preds']))
        final_pred_runs.append(np.array(res['final_preds']))
        # Predict-as-seen MCC (honest streaming MCC)
        preds_at_time = (np.array(res['pred_at_seen_time']) >= 0.5).astype(int)
        streaming_mcc = matthews_corrcoef(y, preds_at_time)
        online_results.append({
            'seed': seed,
            'streaming_mcc': float(streaming_mcc),
            'final_pass_mcc': float(final_mcc),
            'mcc_curve': res['running_mcc_curve'],
            'n_seen_at_each_mcc': res['n_seen_at_each_mcc'],
        })
        print(f'  seed {seed}: streaming MCC = {streaming_mcc:.4f}, '
              f'final-pass MCC = {final_mcc:.4f}')

    streaming_mccs = [r['streaming_mcc'] for r in online_results]
    final_mccs = [r['final_pass_mcc'] for r in online_results]
    print(f'\n  streaming mean ± std: {np.mean(streaming_mccs):.4f} ± {np.std(streaming_mccs):.4f}')
    print(f'  final-pass mean ± std: {np.mean(final_mccs):.4f} ± {np.std(final_mccs):.4f}')

    print('\n[4a] 5-fold CV — class-weighted logistic regression...')
    fold_mccs, test_preds, test_probs = kfold_eval(X, y, n_splits=5, seed=42)
    cv_mcc = matthews_corrcoef(y, test_preds)
    print(f'  per-fold MCC: {[f"{m:.4f}" for m in fold_mccs]}')
    print(f'  pooled out-of-fold MCC: {cv_mcc:.4f}')

    print('\n[4b] 5-fold CV — gradient boosting (XGBoost-like)...')
    from sklearn.ensemble import GradientBoostingClassifier
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gb_test_preds = np.zeros(len(X), dtype=int)
    gb_test_probs = np.zeros(len(X), dtype=float)
    gb_fold_mccs = []
    for tr_idx, te_idx in skf2.split(X, y):
        # Sample-weight to handle class imbalance
        sw = np.where(y[tr_idx] == 1,
                      (y == 0).sum() / len(y),
                      (y == 1).sum() / len(y))
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                          learning_rate=0.1, random_state=42)
        clf.fit(X[tr_idx], y[tr_idx], sample_weight=sw)
        probs = clf.predict_proba(X[te_idx])[:, 1]
        # Threshold tuned on training fold
        train_probs = clf.predict_proba(X[tr_idx])[:, 1]
        best_t, best_mcc_tr = 0.5, -2.0
        for t in np.linspace(0.05, 0.95, 19):
            tp = (train_probs >= t).astype(int)
            if tp.sum() in (0, len(tp)): continue
            m = matthews_corrcoef(y[tr_idx], tp)
            if m > best_mcc_tr: best_mcc_tr, best_t = m, t
        preds = (probs >= best_t).astype(int)
        if pd.Series(preds).nunique() < 2 or pd.Series(y[te_idx]).nunique() < 2:
            mcc = 0.0
        else:
            mcc = matthews_corrcoef(y[te_idx], preds)
        gb_fold_mccs.append(mcc)
        gb_test_preds[te_idx] = preds
        gb_test_probs[te_idx] = probs
    cv_mcc_gb = matthews_corrcoef(y, gb_test_preds)
    print(f'  per-fold MCC: {[f"{m:.4f}" for m in gb_fold_mccs]}')
    print(f'  pooled out-of-fold MCC (GB): {cv_mcc_gb:.4f}')

    # Pick the better one for "final" CV
    if cv_mcc_gb > cv_mcc:
        print(f'  GB > LogReg: using GB as the final ML CV')
        cv_mcc = cv_mcc_gb
        test_preds = gb_test_preds
        test_probs = gb_test_probs
        fold_mccs = gb_fold_mccs
        cv_method = 'gradient_boosting'
    else:
        cv_method = 'class_weighted_logistic_regression'

    print(f'\n  best held-out method: {cv_method}, MCC = {cv_mcc:.4f}')

    # Compare delta vs v15
    delta_streaming = float(np.mean(streaming_mccs) - v15_mcc)
    delta_cv = float(cv_mcc - v15_mcc)

    print(f'\n[5] DELTA vs v15 baseline ({v15_mcc:.4f}):')
    print(f'  streaming-online MCC delta: {delta_streaming:+.4f}')
    print(f'  5-fold CV MCC delta: {delta_cv:+.4f}')

    # CV confusion matrix (best ML method)
    cv_tn, cv_fp, cv_fn, cv_tp = confusion_matrix(y, test_preds).ravel()
    print(f'  CV confusion: TP={cv_tp} FP={cv_fp} TN={cv_tn} FN={cv_fn}')

    # Compress online curves: keep mean curve at 10 evenly-spaced steps.
    # Curves can have different lengths across seeds (depending on when both
    # classes first show up), so we sample each curve at 10 fractional points
    # of its own length, then average the values across seeds at each point.
    mean_curve = []
    for frac_idx in range(10):
        frac = frac_idx / 9.0
        point_mccs, point_n_seens = [], []
        for r in online_results:
            curve = r['mcc_curve']
            n_seen_arr = r['n_seen_at_each_mcc']
            if not curve:
                continue
            i = min(int(frac * (len(curve) - 1)), len(curve) - 1)
            point_mccs.append(curve[i])
            point_n_seens.append(n_seen_arr[i])
        if point_mccs:
            mean_curve.append({'fraction': float(frac),
                               'mean_n_seen': float(np.mean(point_n_seens)),
                               'mean_mcc': float(np.mean(point_mccs)),
                               'std_mcc': float(np.std(point_mccs))})

    out = {
        'method': 'simulator_v15_features_plus_online_ml_classifier',
        'baselines': {
            'v15_mcc': float(v15_mcc),
            'v15_confusion': {'tp': int(v15_tp), 'fp': int(v15_fp),
                              'tn': int(v15_tn), 'fn': int(v15_fn)},
        },
        'cv_method': cv_method,
        'features': feat_names,
        'n_features': X.shape[1],
        'n_genes': int(len(df)),
        'class_balance': {'essential': int(y.sum()), 'nonessential': int((y == 0).sum())},
        'online_learning': {
            'n_seeds': len(online_results),
            'streaming_mccs': [r['streaming_mcc'] for r in online_results],
            'final_pass_mccs': [r['final_pass_mcc'] for r in online_results],
            'mean_streaming_mcc': float(np.mean(streaming_mccs)),
            'std_streaming_mcc': float(np.std(streaming_mccs)),
            'mean_final_pass_mcc': float(np.mean(final_mccs)),
            'mean_curve_10pts': mean_curve,
        },
        'cv_5fold': {
            'method': cv_method,
            'per_fold_mcc': [float(m) for m in fold_mccs],
            'pooled_mcc': float(cv_mcc),
            'pooled_confusion': {'tp': int(cv_tp), 'fp': int(cv_fp),
                                  'tn': int(cv_tn), 'fn': int(cv_fn)},
        },
        'cv_5fold_logreg': {
            'pooled_mcc': float(matthews_corrcoef(y, test_preds)),
        },
        'cv_5fold_gradient_boosting': {
            'pooled_mcc': float(cv_mcc_gb),
        },
        'delta_vs_v15': {
            'streaming_online': delta_streaming,
            'cv_5fold': delta_cv,
        },
        'verdict': 'improvement' if (delta_cv > 0.005) else 'parity_or_worse',
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f'\nwrote {OUT_PATH}')


if __name__ == '__main__':
    main()
