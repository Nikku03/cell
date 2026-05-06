"""Pattern atlas + connection graph utilities for MultimodalSparseConceptLNN.

After training, run these to dump:
  - pattern_atlas.csv: each pattern described from STATIC, TRAJ, KNOW views
  - pattern_connections.csv: K×K co-activation matrix per modality + cross-
    modal alignment scores per pattern
  - per_gene_attributions.csv: for every gene, top-3 patterns from each
    modality with their weights — the "explainer"
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
def collect_concept_activations(model, batches: dict, traj_lookup: dict,
                                  know_lookup: dict, device: str = 'cpu'):
    """Run inference on every gene in `batches` and return per-modality
    concept activations + metadata.

    Returns dict with arrays:
      static_concepts [N_total, K]
      traj_concepts   [N_total, K]
      know_concepts   [N_total, K]
      labels          [N_total]
      organisms       [N_total] (string)
      loci            [N_total] (string)
      gene_names      [N_total] (string)
      essentiality_pred [N_total]
      has_traj        [N_total]
    """
    model.eval()
    all_static, all_traj, all_know = [], [], []
    all_labels, all_orgs, all_loci, all_names = [], [], [], []
    all_pred, all_has_traj = [], []

    for org, batch in batches.items():
        N = len(batch['locus_tags'])
        # Move batch tensors to device so the static branch compute matches
        # the model's parameters' device. (Bug fix: previously only traj/
        # know/has_traj were moved; the batch dict — which feeds the static
        # encoder — stayed on CPU and tripped a device mismatch.)
        batch_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                       for k, v in batch.items()}
        # Per-gene trajectory tensor
        T = model.cfg.traj_n_steps
        D = model.cfg.traj_input_dim
        K_d = model.cfg.know_input_dim
        traj = torch.zeros(N, T, D)
        has_traj = torch.zeros(N)
        know = torch.zeros(N, K_d)
        for i, lt in enumerate(batch['locus_tags']):
            t = traj_lookup.get((org, lt))
            if t is not None:
                # Truncate / pad if cached trajectory shape differs from model config
                t_arr = np.asarray(t, dtype=np.float32)
                Tn, Dn = t_arr.shape
                tt = min(Tn, T); dd = min(Dn, D)
                traj[i, :tt, :dd] = torch.as_tensor(t_arr[:tt, :dd])
                has_traj[i] = 1.0
            kf = know_lookup.get((org, lt))
            if kf is not None:
                kf_arr = np.asarray(kf, dtype=np.float32)
                kk = min(kf_arr.shape[0], K_d)
                know[i, :kk] = torch.as_tensor(kf_arr[:kk])
        with torch.no_grad():
            out = model(batch_dev, traj=traj.to(device), know=know.to(device),
                          has_traj=has_traj.to(device), store_memory=False)
            preds = torch.sigmoid(out['essentiality']).cpu().numpy()

        all_static.append(out['static_concepts'].cpu().numpy())
        all_traj.append(out['traj_concepts'].cpu().numpy())
        all_know.append(out['know_concepts'].cpu().numpy())
        all_labels.append(batch['labels'].cpu().numpy().astype(int))
        all_orgs.extend([org] * N)
        all_loci.extend(list(batch['locus_tags']))
        all_names.extend(list(batch.get('gene_names', [''] * N)))
        all_pred.extend(preds.tolist())
        all_has_traj.extend(has_traj.numpy().tolist())

    return {
        'static_concepts': np.concatenate(all_static, axis=0),
        'traj_concepts': np.concatenate(all_traj, axis=0),
        'know_concepts': np.concatenate(all_know, axis=0),
        'labels': np.concatenate(all_labels, axis=0),
        'organisms': np.array(all_orgs),
        'loci': np.array(all_loci),
        'gene_names': np.array(all_names, dtype=object),
        'essentiality_pred': np.array(all_pred),
        'has_traj': np.array(all_has_traj),
    }


# ──────────────────────────────────────────────────────────────────────
def build_pattern_atlas(act: dict, top_n: int = 15) -> pd.DataFrame:
    """For each pattern k in K, find top-N most-activating genes from each
    modality independently and summarize. Returns DataFrame with one row
    per pattern."""
    K = act['static_concepts'].shape[1]
    rows = []
    for k in range(K):
        s_scores = act['static_concepts'][:, k]
        t_scores = act['traj_concepts'][:, k]
        kn_scores = act['know_concepts'][:, k]

        # Top-N indices from each modality
        s_idx = np.argsort(s_scores)[::-1][:top_n]
        t_mask = act['has_traj'] > 0
        if t_mask.sum() > 0:
            t_scores_masked = t_scores.copy()
            t_scores_masked[~t_mask.astype(bool)] = -np.inf
            t_idx = np.argsort(t_scores_masked)[::-1][:top_n]
        else:
            t_idx = np.array([], dtype=int)
        kn_idx = np.argsort(kn_scores)[::-1][:top_n]

        def summarize(idx):
            if len(idx) == 0:
                return {'top_genes': '', 'top_orgs': '',
                        'essential_frac': float('nan'), 'mean_score': 0.0}
            top_loci = list(act['loci'][idx])
            top_orgs = list(act['organisms'][idx])
            top_names = [n for n in act['gene_names'][idx] if n]
            top_label_genes = top_names if top_names else top_loci
            return {
                'top_genes': ' | '.join(top_label_genes[:8]),
                'top_orgs': ','.join(sorted(set(top_orgs))),
                'essential_frac': float(act['labels'][idx].mean()),
                'mean_score': float(s_scores[idx].mean()),
            }

        s = summarize(s_idx)
        t = summarize(t_idx) if len(t_idx) else {'top_genes': '',
                                                    'top_orgs': '',
                                                    'essential_frac': float('nan'),
                                                    'mean_score': 0.0}
        kn = summarize(kn_idx)

        # Cross-modal alignment of pattern k: do the same genes activate it
        # across modalities?
        st_corr = float(np.corrcoef(s_scores, t_scores)[0, 1]) \
            if t_mask.sum() > 1 else float('nan')
        sk_corr = float(np.corrcoef(s_scores, kn_scores)[0, 1])
        tk_corr = float(np.corrcoef(t_scores, kn_scores)[0, 1]) \
            if t_mask.sum() > 1 else float('nan')

        rows.append({
            'pattern_id': k,
            'static_top_genes': s['top_genes'],
            'static_top_orgs': s['top_orgs'],
            'static_essential_frac': s['essential_frac'],
            'traj_top_genes': t.get('top_genes', ''),
            'traj_top_orgs': t.get('top_orgs', ''),
            'traj_essential_frac': t.get('essential_frac', float('nan')),
            'know_top_genes': kn['top_genes'],
            'know_top_orgs': kn['top_orgs'],
            'know_essential_frac': kn['essential_frac'],
            'corr_static_traj': st_corr,
            'corr_static_know': sk_corr,
            'corr_traj_know': tk_corr,
            'mean_alignment': float(np.nanmean([st_corr, sk_corr, tk_corr])),
        })
    df = pd.DataFrame(rows).sort_values('mean_alignment', ascending=False)
    return df


# ──────────────────────────────────────────────────────────────────────
def build_pattern_connections(act: dict, top_k: int = 5,
                                  threshold: float = 0.3) -> pd.DataFrame:
    """K×K pattern co-activation matrix — for each pair (k1, k2), how often
    do they both fire in the top-k of the same gene?

    Returns long-format edge list: (k1, k2, modality, weight, alignment).
    """
    edges = []
    for modality, key in [('static', 'static_concepts'),
                            ('traj', 'traj_concepts'),
                            ('know', 'know_concepts')]:
        scores = act[key]
        if modality == 'traj':
            mask = act['has_traj'] > 0
            scores = scores[mask.astype(bool)]
            if scores.shape[0] == 0:
                continue
        # For each gene, top-k pattern indices
        top_idx = np.argsort(scores, axis=1)[:, -top_k:]
        K = scores.shape[1]
        co = np.zeros((K, K), dtype=np.int64)
        for row in top_idx:
            for i in row:
                for j in row:
                    if i < j:
                        co[i, j] += 1
        # Normalize by total genes in this modality
        n = scores.shape[0]
        for i in range(K):
            for j in range(i + 1, K):
                w = co[i, j] / n
                if w >= threshold:
                    edges.append({
                        'pattern_a': int(i), 'pattern_b': int(j),
                        'modality': modality, 'weight': float(w),
                        'count': int(co[i, j]),
                    })
    return pd.DataFrame(edges).sort_values(
        ['modality', 'weight'], ascending=[True, False])


# ──────────────────────────────────────────────────────────────────────
def per_gene_attributions(act: dict, top_k: int = 3) -> pd.DataFrame:
    """For every gene, list the top-K patterns from each modality with
    their normalized weights. The "explainer" output."""
    rows = []
    for i in range(len(act['loci'])):
        org = act['organisms'][i]; lt = act['loci'][i]
        name = act['gene_names'][i]
        label = int(act['labels'][i])
        pred = float(act['essentiality_pred'][i])

        def topk(scores):
            top = np.argsort(scores)[-top_k:][::-1]
            top_scores = scores[top]
            # Normalize via softmax over selected
            w = np.exp(top_scores - top_scores.max())
            w = w / w.sum()
            return [(int(t), float(w[j])) for j, t in enumerate(top)]

        st = topk(act['static_concepts'][i])
        tr = topk(act['traj_concepts'][i]) if act['has_traj'][i] > 0 else []
        kn = topk(act['know_concepts'][i])

        rows.append({
            'organism': org,
            'locus_tag': lt,
            'gene_name': name,
            'true_essential': label,
            'predicted_prob': pred,
            'static_top_patterns': ','.join(f'p{p}:{w:.2f}' for p, w in st),
            'traj_top_patterns': ','.join(f'p{p}:{w:.2f}' for p, w in tr),
            'know_top_patterns': ','.join(f'p{p}:{w:.2f}' for p, w in kn),
            'has_traj_signal': int(act['has_traj'][i] > 0),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
def write_atlas_files(act: dict, out_dir: Path | str,
                        prefix: str = 'multimodal') -> dict[str, Path]:
    """Generate all 3 atlas CSVs and write them to disk. Returns paths."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    atlas_df = build_pattern_atlas(act)
    p = out_dir / f'{prefix}_pattern_atlas.csv'
    atlas_df.to_csv(p, index=False); files['atlas'] = p

    conn_df = build_pattern_connections(act)
    p = out_dir / f'{prefix}_pattern_connections.csv'
    conn_df.to_csv(p, index=False); files['connections'] = p

    attr_df = per_gene_attributions(act)
    p = out_dir / f'{prefix}_per_gene_attributions.csv'
    attr_df.to_csv(p, index=False); files['attributions'] = p

    return files
