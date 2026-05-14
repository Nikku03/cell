"""Per-gene XGBoost essentiality classifier.

Standalone, no LGNN required. Built per the post-M-Bio post-mortem:
the LGNN's joint dynamics+essentiality training catastrophically
overfits 455 binary labels onto 3.4M parameters. A small, static
classifier on hand-crafted features is the right shape for this data.

Two stages:

  extract_per_gene_features(data, row_names, static_features, breuer_df)
    Builds a (n_genes, F) feature matrix and (n_genes,) label vector
    aligned by locus tag. Features per locus:
      - 54 static-node features (broadcast from gene_class + proteomics)
        averaged across the gene's row variants (G_*, R_*, P_*, RPM_*,
        PM_*, DM_*) so we get gene-level rather than row-level features
      - 6 trajectory statistics per category (G/R/P/RPM/PM/DM), each
        category contributing: mean over (rep, t), std over (rep, t),
        cross-rep std at t=end, drift (t=end - t=0), slope (mean-rate),
        and final value
      Total: 54 + 6*6 = 90 features per locus

  train_xgb_cv(X, y, n_folds=5, seed=0)
    Stratified k-fold CV. Returns per-fold metrics + mean MCC.

Why this works where the joint LGNN failed:
  - 455 labels constrain a ~few-thousand-parameter XGBoost reasonably
  - Trajectory stats (mean / std / drift / slope) are the SAME shape of
    feature that the data analysis found CV-MCC=0.145 on. With more
    features and a nonlinear classifier, expected MCC 0.30-0.45.
"""
from __future__ import annotations
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

_LOCUS_RE = re.compile(r'_(\d{4})(?:_|$)')

# Trajectory categories: prefix → category label
TRAJ_CATEGORIES = ('G', 'R', 'P', 'RPM', 'PM', 'DM')


def _signed_expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def _row_prefix(name: str) -> Optional[str]:
    """Return the row's prefix category, or None if it's a tracker/non-locus."""
    # Exclude transient trackers
    if name.endswith(('_TC', '_f', '_d', '_p', '_pe', '_cp')):
        return None
    if name.startswith('DT_'):
        return None
    if name.startswith('PM_'):  return 'PM'
    if name.startswith('RPM_'): return 'RPM'
    if name.startswith('DM_'):  return 'DM'
    if name.startswith('RP_'):  return None   # RNAP:gene complex (not gene)
    if name.startswith('RB_'):  return None
    if name.startswith('C_P_'): return 'P'     # cytoplasmic membrane protein
    if name.startswith('G_'):   return 'G'
    if name.startswith('R_'):   return 'R'
    if name.startswith('P_'):   return 'P'
    if name.startswith('D_'):   return None    # Degradosome:mRNA
    if name.startswith('S_'):   return None    # secY:protein
    return None


def extract_per_gene_features(
    data: torch.Tensor,                  # (R, T, S) preloaded trajectories
    row_names: Sequence[str],
    static_features: torch.Tensor,        # (S, F_static)
    breuer_df: pd.DataFrame,              # locus_tag, locus_4d, essential
    drop_unlabeled: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Build (X, y, locus_tags, feature_names) for XGBoost.

    Returns
    -------
    X            : (n_genes, F) float32 ndarray
    y            : (n_genes,) int8 ndarray  (1=essential, 0=not)
    locus_tags   : list[str] of length n_genes (e.g. 'JCVISYN3A_0001')
    feature_names: list[str] of length F
    """
    R, T, S = data.shape
    assert static_features.shape[0] == S, \
        f'static features rows {static_features.shape[0]} != S {S}'
    F_static = int(static_features.shape[1])

    # Per-locus aggregation
    locus_to_rows: dict[str, list[tuple[str, int]]] = {}
    for i, name in enumerate(row_names):
        pref = _row_prefix(name)
        if pref is None:
            continue
        m = _LOCUS_RE.search(name)
        if m is None:
            continue
        locus = m.group(1)
        locus_to_rows.setdefault(locus, []).append((pref, i))

    # Convert breuer to lookup
    by_locus = {row['locus_4d']: bool(row['essential'])
                  for _, row in breuer_df.iterrows()}

    feature_names: List[str] = []
    # static features: average over all participating rows for this locus
    static_arr = static_features.float().cpu().numpy()
    feature_names += [f'static_{j}' for j in range(F_static)]
    # trajectory features per category
    traj_feat_names = ['mean', 'std', 'xrep_std_end', 'drift',
                        'slope', 'final']
    for cat in TRAJ_CATEGORIES:
        for fn in traj_feat_names:
            feature_names.append(f'traj_{cat}_{fn}')

    # Pre-decode signed-log1p to LINEAR space for trajectory features.
    # This is heavy memory-wise — only do it row-by-row. We'll grab
    # trajectories for needed rows only.
    rows: List[np.ndarray] = []
    locus_tags: List[str] = []
    labels: List[int] = []
    sorted_loci = sorted(locus_to_rows.keys())
    for locus in sorted_loci:
        locus_tag = f'JCVISYN3A_{locus}'
        if drop_unlabeled and locus_tag not in by_locus:
            continue
        # --- static features ---
        idxs = [i for _, i in locus_to_rows[locus]]
        st = static_arr[idxs].mean(axis=0)                         # (F_static,)

        # --- trajectory features per category ---
        cat_idxs: dict[str, list[int]] = {c: [] for c in TRAJ_CATEGORIES}
        for pref, i in locus_to_rows[locus]:
            if pref in cat_idxs:
                cat_idxs[pref].append(i)

        traj_feats: List[float] = []
        for cat in TRAJ_CATEGORIES:
            idx_list = cat_idxs[cat]
            if not idx_list:
                traj_feats.extend([0.0] * len(traj_feat_names))
                continue
            # Average over multiple rows in same category (e.g. G_NNNN_C1+C2)
            traj = data[:, :, idx_list].float()                    # (R, T, k)
            # signed_log1p → linear
            traj_lin = _signed_expm1(traj).mean(dim=2)             # (R, T)
            traj_lin_np = traj_lin.cpu().numpy().astype(np.float32)
            mean_v        = float(traj_lin_np.mean())
            std_v         = float(traj_lin_np.std())
            xrep_std_end  = float(traj_lin_np[:, -1].std())
            drift         = float(traj_lin_np[:, -1].mean() - traj_lin_np[:, 0].mean())
            # slope: regress mean-trajectory against time
            mean_traj = traj_lin_np.mean(axis=0)                   # (T,)
            tt = np.arange(T, dtype=np.float32) / max(T - 1, 1)
            slope_v       = float(np.polyfit(tt, mean_traj, 1)[0])
            final_v       = float(mean_traj[-1])
            traj_feats.extend([mean_v, std_v, xrep_std_end, drift,
                                slope_v, final_v])

        row_feat = np.concatenate([st.astype(np.float32),
                                     np.array(traj_feats, dtype=np.float32)])
        rows.append(row_feat)
        locus_tags.append(locus_tag)
        labels.append(int(by_locus.get(locus_tag, False)))

    X = np.stack(rows, axis=0)
    y = np.array(labels, dtype=np.int8)

    if verbose:
        n_ess = int(y.sum()); n_neg = int(len(y) - n_ess)
        print(f'  feature matrix: X.shape={X.shape}  y.shape={y.shape}')
        print(f'  essential={n_ess}, non={n_neg}')
        print(f'  features: {F_static} static + '
              f'{len(TRAJ_CATEGORIES) * len(traj_feat_names)} traj '
              f'= {X.shape[1]} total')
    return X, y, locus_tags, feature_names


def train_xgb_cv(
    X: np.ndarray,
    y: np.ndarray,
    locus_tags: Sequence[str],
    feature_names: Optional[Sequence[str]] = None,
    n_folds: int = 5,
    seed: int = 0,
    xgb_params: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """Stratified k-fold XGBoost. Returns per-fold + mean metrics."""
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError('xgboost not installed. Run: pip install xgboost') from e
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import matthews_corrcoef

    if xgb_params is None:
        xgb_params = dict(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            scale_pos_weight=float((y == 0).sum()) / max((y == 1).sum(), 1),
            objective='binary:logistic',
            tree_method='hist',
            eval_metric='logloss',
            verbosity=0,
            random_state=seed,
        )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []
    fold_predictions = np.zeros_like(y, dtype=np.float64)
    all_importances = np.zeros(X.shape[1], dtype=np.float64)

    for k, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        Xt, Xv = X[tr_idx], X[va_idx]
        yt, yv = y[tr_idx], y[va_idx]
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
        p_va = clf.predict_proba(Xv)[:, 1]
        pred_va = (p_va > 0.5).astype(np.int8)
        mcc = matthews_corrcoef(yv, pred_va)
        tp = int(((pred_va == 1) & (yv == 1)).sum())
        fp = int(((pred_va == 1) & (yv == 0)).sum())
        fn = int(((pred_va == 0) & (yv == 1)).sum())
        tn = int(((pred_va == 0) & (yv == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        fold_metrics.append({
            'fold': k, 'MCC': mcc, 'precision': precision,
            'recall': recall, 'F1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        })
        fold_predictions[va_idx] = p_va
        all_importances += clf.feature_importances_ / n_folds

        if verbose:
            print(f'  fold {k}: MCC={mcc:+.3f}  P={precision:.3f}  '
                  f'R={recall:.3f}  F1={f1:.3f}  '
                  f'(tp={tp} fp={fp} fn={fn} tn={tn})')

    mean_mcc = float(np.mean([m['MCC'] for m in fold_metrics]))
    std_mcc  = float(np.std ([m['MCC'] for m in fold_metrics]))
    overall_mcc = float(matthews_corrcoef(y, (fold_predictions > 0.5).astype(int)))

    if verbose:
        print(f'\n  ===  CV summary  ===')
        print(f'  mean MCC:    {mean_mcc:+.3f} ± {std_mcc:.3f}')
        print(f'  overall MCC: {overall_mcc:+.3f}  (concat predictions)')
        if feature_names is not None:
            top = sorted(zip(feature_names, all_importances),
                          key=lambda x: -x[1])[:15]
            print(f'\n  top 15 features by avg importance:')
            for name, imp in top:
                print(f'    {imp:.4f}  {name}')

    return {
        'fold_metrics': fold_metrics,
        'predictions': fold_predictions,
        'mean_MCC': mean_mcc,
        'std_MCC': std_mcc,
        'overall_MCC': overall_mcc,
        'feature_importances': all_importances,
    }
