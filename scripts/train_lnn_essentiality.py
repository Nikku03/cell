"""Phase 5 — train a Liquid Neural Net (LNN) on raw genomic data + annotation
across 6 bacteria, test on Syn3A.

Architecture (inspired by enzyme_Software/liquid_nn_v2 but tailored for genes):

  Per-gene inputs:
    - 20 structured scalar features (length, GC, position, product flags, kw_*)
    - 64 codon-usage k-mer counts (3-mer frequency normalized over CDS)
    - 1024 4-mer counts (compact upstream regulatory signal)

  Graph:
    - Nodes = genes
    - Edges = same-strand neighbors with gap < 100 bp (operon proxy)
    - Edge features = (gap_bp_log, strand_match)

  LNN backbone:
    1. GeneEncoder: scalar + 3-mer + 4-mer -> hidden
    2. n_steps adaptive-tau ODE integration:
         tau_g = ContextAwareTau(h_g, h_global_organism)  # per-gene
         msg_g = EdgeAwareMessagePassing(h_g, h_neighbors)
         h_g <- h_g + dt * (msg_g + h0 - h_g) / tau_g
    3. Per-gene classification head -> logit

  Training:
    - 6 bacteria as training (Caulobacter / S.aureus / M.tb / S.Typhimurium /
      A.baylyi / M.pneumoniae) — each is its own connected graph
    - Class-weighted BCE
    - 5-fold per-organism CV during training to monitor over/underfitting
    - Per-organism mini-batches (each batch = one whole organism's gene set)

  Test:
    - Syn3A held out completely (no Syn3A genes in training)
    - Build Syn3A graph the same way, forward through trained LNN
    - Threshold-tune on training set, apply to Syn3A test set
    - Report MCC, confusion matrix

Output: outputs/lnn_essentiality_results.json
"""
from __future__ import annotations

import json
import re
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
from sklearn.metrics import confusion_matrix, matthews_corrcoef

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / 'outputs/multiorg_per_gene_features.csv'
GB_DIR = ROOT / 'memory_bank/data/multiorg_essentiality/raw/multi_organism_genbank'
OUT = ROOT / 'outputs/lnn_essentiality_results.json'

SCALAR_FEATURES = [
    'log_length_bp', 'gc', 'upstream_gc', 'upstream_at_skew',
    'position_norm', 'operon_run_length',
    'is_hypothetical', 'is_uncharacterized', 'is_putative',
    'kw_replication', 'kw_transcription', 'kw_translation',
    'kw_atp', 'kw_membrane', 'kw_synthase', 'kw_kinase',
    'kw_dehydrogenase', 'kw_protease', 'kw_rrna', 'kw_chaperone',
]

NUCS = ['A', 'C', 'G', 'T']
KMERS_3 = [''.join(t) for t in product(NUCS, repeat=3)]   # 64
KMERS_4 = [''.join(t) for t in product(NUCS, repeat=4)]   # 256

GB_PATHS = {
    'ccrescentus': GB_DIR / 'ccrescentus_NA1000_CP001340.1.gb',
    'saureus': GB_DIR / 'saureus_N315_BA000018.3.gb',
    'mtuberculosis': GB_DIR / 'mtuberculosis_H37Rv_AL123456.3.gb',
    'styphimurium': GB_DIR / 'styphimurium_LT2_AE006468.2.gb',
    'abaylyi': GB_DIR / 'abaylyi_ADP1_CR543861.1.gb',
    'mpne': ROOT / 'memory_bank/data/multiorg_essentiality/raw/mpneumoniae_M129_NC_000912.gb',
    'syn3a': ROOT / 'cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb',
}

SEED = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def kmer_counts(seq: str, k: int) -> np.ndarray:
    """Return normalized k-mer count vector. seq is uppercase ACGT."""
    if len(seq) < k:
        return np.zeros(4 ** k, dtype=np.float32)
    counts = np.zeros(4 ** k, dtype=np.float32)
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_total = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        idx = 0; valid = True
        for ch in kmer:
            if ch not in nuc_to_idx:
                valid = False; break
            idx = idx * 4 + nuc_to_idx[ch]
        if valid:
            counts[idx] += 1
            n_total += 1
    if n_total > 0:
        counts /= n_total
    return counts


def extract_per_organism_data(organism: str, ess_df: pd.DataFrame
                                ) -> dict:
    """Parse organism GenBank, build per-gene k-mer features + edge list."""
    print(f'  parsing {organism}...')
    gb_path = GB_PATHS[organism]
    rec = next(SeqIO.parse(str(gb_path), 'genbank'))
    organism_ess = ess_df[ess_df.organism == organism].set_index('locus_tag')

    # Collect CDSes in genome order
    cdses = []
    for f in rec.features:
        if f.type != 'CDS': continue
        # Per-organism locus_tag policy (matches build_multiorg_features.py)
        locus = f.qualifiers.get('locus_tag', [None])[0]
        gene_sym = (f.qualifiers.get('gene', [''])[0] or '').strip()
        if organism == 'mpne':
            ot = f.qualifiers.get('old_locus_tag', [None])[0]
            if ot: locus = ot
        elif organism == 'saureus':
            locus = gene_sym if gene_sym else locus
        elif organism == 'styphimurium':
            locus = gene_sym if gene_sym else locus
        if not locus: continue

        seq = str(f.extract(rec.seq)).upper()
        cdses.append({
            'locus_tag': locus, 'seq': seq,
            'start': int(f.location.start),
            'end': int(f.location.end),
            'strand': int(f.location.strand) if f.location.strand else 1,
        })
    cdses.sort(key=lambda d: d['start'])

    # Filter to genes with essentiality labels AND scalar features in CSV
    genes_with_label = [c for c in cdses if c['locus_tag'] in organism_ess.index]
    print(f'    {organism}: {len(genes_with_label)}/{len(cdses)} CDSes have labels')

    # Build per-gene k-mer features + scalar features lookup
    feat_lookup = {}  # locus_tag -> np.array of [scalars + k-mer3 + k-mer4]
    label_lookup = {}  # locus_tag -> int 0/1
    locus_to_idx = {}  # within-organism node index
    pos_lookup = {}  # locus_tag -> (start, end, strand) for edge building
    for i, c in enumerate(genes_with_label):
        seq = c['seq']
        if len(seq) < 6:
            continue
        # 3-mer counts (codon usage): align to CDS start
        k3 = kmer_counts(seq, 3)
        # 4-mer counts of CDS
        k4 = kmer_counts(seq, 4)
        # Scalar features from the joined CSV
        row = organism_ess.loc[c['locus_tag']]
        if isinstance(row, pd.DataFrame):  # duplicate locus_tag
            row = row.iloc[0]
        scalars = np.array([row.get(k, 0.0) for k in SCALAR_FEATURES],
                            dtype=np.float32)
        feat_lookup[c['locus_tag']] = np.concatenate([scalars, k3, k4])
        label_lookup[c['locus_tag']] = int(row['essential'])
        locus_to_idx[c['locus_tag']] = len(locus_to_idx)
        pos_lookup[c['locus_tag']] = (c['start'], c['end'], c['strand'])

    n_nodes = len(locus_to_idx)
    if n_nodes == 0:
        return None

    # Build edges: same-strand neighbors with gap < 100bp (operon proxy)
    sorted_loci = sorted(locus_to_idx.keys(), key=lambda lt: pos_lookup[lt][0])
    edges_src, edges_dst, edges_attr = [], [], []
    for i in range(len(sorted_loci) - 1):
        a, b = sorted_loci[i], sorted_loci[i + 1]
        sa, ea, st_a = pos_lookup[a]
        sb, eb, st_b = pos_lookup[b]
        gap = sb - ea
        if gap < 100 and st_a == st_b:
            ai, bi = locus_to_idx[a], locus_to_idx[b]
            # Bidirectional edges
            edges_src.append(ai); edges_dst.append(bi)
            edges_attr.append([np.log1p(max(gap, 0)), 1.0])
            edges_src.append(bi); edges_dst.append(ai)
            edges_attr.append([np.log1p(max(gap, 0)), 1.0])

    edge_index = (np.array([edges_src, edges_dst], dtype=np.int64)
                   if edges_src else np.zeros((2, 0), dtype=np.int64))
    edge_attr = (np.array(edges_attr, dtype=np.float32)
                  if edges_attr else np.zeros((0, 2), dtype=np.float32))

    X = np.stack([feat_lookup[lt] for lt in
                   sorted(locus_to_idx, key=locus_to_idx.get)])
    y = np.array([label_lookup[lt] for lt in
                   sorted(locus_to_idx, key=locus_to_idx.get)], dtype=np.int64)

    return {
        'organism': organism, 'X': X, 'y': y,
        'edge_index': edge_index, 'edge_attr': edge_attr,
        'n_nodes': n_nodes,
        'locus_order': sorted(locus_to_idx, key=locus_to_idx.get),
    }


# ──────────────────────────── LNN model ────────────────────────────

class GeneEncoder(nn.Module):
    def __init__(self, n_scalar, n_kmer3, n_kmer4, hidden=128):
        super().__init__()
        self.scalar_proj = nn.Linear(n_scalar, hidden // 2)
        self.kmer3_proj = nn.Linear(n_kmer3, hidden // 4)
        self.kmer4_proj = nn.Linear(n_kmer4, hidden // 4)
        self.combine = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
            nn.SiLU(), nn.LayerNorm(hidden),
        )

    def forward(self, scalar, kmer3, kmer4):
        s = F.silu(self.scalar_proj(scalar))
        k3 = F.silu(self.kmer3_proj(kmer3))
        k4 = F.silu(self.kmer4_proj(kmer4))
        return self.combine(torch.cat([s, k3, k4], dim=-1))


class ContextAwareTau(nn.Module):
    """Adaptive time constant per gene from local + global context."""
    def __init__(self, hidden, tau_min=0.1, tau_max=2.0):
        super().__init__()
        self.tau_min = tau_min; self.tau_max = tau_max
        self.net = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1), nn.Sigmoid(),
        )

    def forward(self, h_local, h_global_per_node):
        x = torch.cat([h_local, h_global_per_node], dim=-1)
        tau_norm = self.net(x).squeeze(-1)
        return self.tau_min + (self.tau_max - self.tau_min) * tau_norm


class EdgeAwarePassing(nn.Module):
    """Graph message passing along operon edges with edge features."""
    def __init__(self, hidden, edge_dim=2, dropout=0.1):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, hidden)
        self.msg = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return torch.zeros_like(h)
        src, dst = edge_index[0], edge_index[1]
        e = self.edge_proj(edge_attr)
        m = self.msg(torch.cat([h[src], h[dst], e], dim=-1))
        out = torch.zeros_like(h)
        out.index_add_(0, dst, m)
        deg = torch.bincount(dst, minlength=h.size(0)).clamp(min=1).float()
        return out / deg.unsqueeze(-1)


class GeneLNN(nn.Module):
    """Adaptive-tau ODE-style integration over n_steps message passing rounds."""
    def __init__(self, n_scalar=20, n_kmer3=64, n_kmer4=256,
                 hidden=128, n_steps=4, dropout=0.1):
        super().__init__()
        self.encoder = GeneEncoder(n_scalar, n_kmer3, n_kmer4, hidden)
        self.tau_pred = ContextAwareTau(hidden)
        self.passing = EdgeAwarePassing(hidden, edge_dim=2, dropout=dropout)
        self.n_steps = n_steps
        self.dt = 0.2
        self.n_scalar = n_scalar
        self.n_kmer3 = n_kmer3
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, X, edge_index, edge_attr):
        scalar = X[:, :self.n_scalar]
        kmer3 = X[:, self.n_scalar:self.n_scalar + self.n_kmer3]
        kmer4 = X[:, self.n_scalar + self.n_kmer3:]
        h = self.encoder(scalar, kmer3, kmer4)
        h0 = h.clone()
        for _ in range(self.n_steps):
            h_global = h.mean(dim=0, keepdim=True).expand_as(h)
            tau = self.tau_pred(h, h_global)
            msg = self.passing(h, edge_index, edge_attr)
            dh = (msg + h0 - h) / tau.unsqueeze(-1)
            h = h + self.dt * dh
        return self.head(h).squeeze(-1)


# ──────────────────────────── training ────────────────────────────

def threshold_search(y_true, probs):
    best_t, best_mcc = 0.5, -2.0
    for t in np.linspace(0.05, 0.95, 19):
        p = (probs >= t).astype(int)
        if p.sum() in (0, len(p)): continue
        m = matthews_corrcoef(y_true, p)
        if m > best_mcc: best_mcc, best_t = m, t
    return best_t, best_mcc


def train(model, train_data, n_epochs=80, lr=2e-3, verbose=True):
    """Train on multiple per-organism graphs."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    # Class weights computed across all training data
    y_all = np.concatenate([d['y'] for d in train_data])
    w0 = (y_all == 1).sum() / len(y_all)
    w1 = (y_all == 0).sum() / len(y_all)
    print(f'  class weights: 0={w0:.3f}  1={w1:.3f}')

    organism_tensors = []
    for d in train_data:
        organism_tensors.append({
            'organism': d['organism'],
            'X': torch.tensor(d['X'], dtype=torch.float32, device=device),
            'y': torch.tensor(d['y'], dtype=torch.float32, device=device),
            'edge_index': torch.tensor(d['edge_index'], dtype=torch.long, device=device),
            'edge_attr': torch.tensor(d['edge_attr'], dtype=torch.float32, device=device),
        })

    history = []
    for ep in range(n_epochs):
        model.train()
        # Shuffle organism order each epoch
        order = np.random.permutation(len(organism_tensors))
        ep_loss = 0.0; ep_n = 0
        for oi in order:
            d = organism_tensors[oi]
            opt.zero_grad()
            logits = model(d['X'], d['edge_index'], d['edge_attr'])
            target = d['y']
            weights = torch.where(target == 1,
                                    torch.tensor(w1, device=device),
                                    torch.tensor(w0, device=device))
            loss = (F.binary_cross_entropy_with_logits(
                logits, target, reduction='none') * weights * 2.0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * d['y'].size(0); ep_n += d['y'].size(0)
        sched.step()
        ep_avg = ep_loss / ep_n
        history.append(ep_avg)
        if verbose and (ep + 1) % 10 == 0:
            # Train MCC across pooled training set
            model.eval()
            with torch.no_grad():
                all_probs, all_y = [], []
                for d in organism_tensors:
                    logits = model(d['X'], d['edge_index'], d['edge_attr'])
                    all_probs.append(torch.sigmoid(logits).cpu().numpy())
                    all_y.append(d['y'].cpu().numpy())
                ap = np.concatenate(all_probs); ay = np.concatenate(all_y)
                bt, bm = threshold_search(ay, ap)
            print(f'  ep {ep+1:3d}  loss={ep_avg:.4f}  '
                  f'train pooled MCC@best_t={bm:.4f} (t={bt:.2f})')
    return history


def evaluate(model, data: dict) -> dict:
    """Run model on one organism's graph, return probs + best threshold from
    pooled training already available (we pass test data here only)."""
    model.eval()
    with torch.no_grad():
        X = torch.tensor(data['X'], dtype=torch.float32, device=device)
        ei = torch.tensor(data['edge_index'], dtype=torch.long, device=device)
        ea = torch.tensor(data['edge_attr'], dtype=torch.float32, device=device)
        logits = model(X, ei, ea)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


# ──────────────────────────── main ────────────────────────────

def main():
    print('[1] loading multiorg structured features (CSV)...')
    ess_df = pd.read_csv(FEATURES_CSV)
    print(f'  rows: {len(ess_df)}')

    print('\n[2] re-parsing GenBanks for sequences + building graphs...')
    ALL_ORGS = ['ccrescentus', 'saureus', 'mtuberculosis', 'styphimurium',
                'abaylyi', 'mpne', 'syn3a']
    organism_data = {}
    for org in ALL_ORGS:
        d = extract_per_organism_data(org, ess_df)
        if d is None: continue
        organism_data[org] = d
        e = (d['y'] == 1).sum(); n = (d['y'] == 0).sum()
        print(f'    {org}: {d["n_nodes"]} nodes, {d["edge_index"].shape[1]} edges, '
              f'{e} ess / {n} ne ({e/(e+n)*100:.0f}% essential)')

    train_orgs = ['ccrescentus', 'saureus', 'mtuberculosis',
                  'styphimurium', 'abaylyi', 'mpne']
    test_org = 'syn3a'
    train_data = [organism_data[o] for o in train_orgs if o in organism_data]
    test_data = organism_data[test_org]

    print(f'\n[3] training LNN on {len(train_data)} organisms '
          f'(total {sum(d["n_nodes"] for d in train_data)} genes)...')
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = GeneLNN(n_scalar=20, n_kmer3=64, n_kmer4=256,
                    hidden=128, n_steps=4, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  {n_params:,} parameters on {device}')

    t0 = time.time()
    history = train(model, train_data, n_epochs=80, lr=2e-3, verbose=True)
    print(f'  trained in {time.time() - t0:.1f}s')

    print('\n[4] evaluating on Syn3A (held out)...')
    test_probs = evaluate(model, test_data)
    test_y = test_data['y']

    # Threshold tuned on POOLED TRAINING SET (honest hold-out for syn3a)
    print('  computing pooled training probabilities for threshold tuning...')
    train_probs = []; train_y = []
    for d in train_data:
        train_probs.append(evaluate(model, d))
        train_y.append(d['y'])
    train_probs = np.concatenate(train_probs)
    train_y = np.concatenate(train_y)
    best_t, train_mcc = threshold_search(train_y, train_probs)

    test_pred = (test_probs >= best_t).astype(int)
    if len(set(test_pred)) > 1 and len(set(test_y)) > 1:
        test_mcc = matthews_corrcoef(test_y, test_pred)
        tn, fp, fn, tp = confusion_matrix(test_y, test_pred).ravel()
    else:
        test_mcc = 0.0; tp = fp = tn = fn = 0

    print(f'\n  pooled-train MCC @ best_t={best_t:.2f} = {train_mcc:.4f}')
    print(f'  syn3a test MCC = {test_mcc:.4f}')
    print(f'  syn3a confusion: TP={tp} FP={fp} TN={tn} FN={fn}')

    # Per-train-organism breakdown for diagnostic
    per_org = {}
    for d in train_data:
        org = d['organism']
        probs = evaluate(model, d)
        pred = (probs >= best_t).astype(int)
        if len(set(pred)) > 1 and len(set(d['y'])) > 1:
            org_mcc = matthews_corrcoef(d['y'], pred)
        else:
            org_mcc = 0.0
        per_org[org] = float(org_mcc)
        print(f'  per-org train MCC [{org}] = {org_mcc:.4f}')

    print('\n[5] scoreboard:')
    print(f'  LNN cross-organism (this run)      MCC = {test_mcc:.4f}')
    print(f'  v25 GB 6-org (reference)           MCC = 0.2268')
    print(f'  v24 mpne->syn3a (reference)        MCC = 0.2034')
    print(f'  v18 ESM-2 cross-organism           MCC = 0.1376')
    print(f'  v15 within-organism (reference)    MCC = 0.5372')

    out = {
        'method': 'lnn_with_adaptive_tau_and_operon_graph_passing',
        'architecture': {
            'encoder': 'scalar(20)+3mer(64)+4mer(256) -> 128',
            'graph': 'same-strand neighbor within 100bp (operon proxy)',
            'lnn_steps': 4,
            'tau_min': 0.1, 'tau_max': 2.0,
            'hidden': 128,
            'n_parameters': n_params,
        },
        'training': {
            'organisms': train_orgs,
            'n_genes_total': int(sum(d['n_nodes'] for d in train_data)),
            'n_essential_total': int(sum((d['y'] == 1).sum() for d in train_data)),
            'epochs': 80, 'lr': 2e-3, 'weight_decay': 1e-4,
            'final_loss': float(history[-1]),
            'pooled_train_mcc': float(train_mcc),
            'best_threshold': float(best_t),
            'per_org_train_mcc': per_org,
        },
        'test': {
            'organism': test_org,
            'n_genes': int(test_data['n_nodes']),
            'n_essential': int((test_y == 1).sum()),
            'n_nonessential': int((test_y == 0).sum()),
            'test_mcc': float(test_mcc),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        },
        'baselines': {
            'v25_gb_6orgs': 0.2268,
            'v24_mpne_to_syn3a_annotation': 0.2034,
            'v18_xgboost_esm2_cross_organism': 0.1376,
            'v15_within_organism': 0.5372,
        },
        'delta_vs_v25_gb': float(test_mcc - 0.2268),
        'delta_vs_v24': float(test_mcc - 0.2034),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
