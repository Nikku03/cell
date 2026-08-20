"""Space A: the ESM/sequence feature blocks of nexus_catalyst_esm, on all 2,231 reactions.

The blocks are the arm's own:  e | s | e*s | |e-s| | freq | enzyme aa-composition | substrate aa-composition.

FAITHFULNESS.  The interaction blocks e*s and |e-s| are elementwise, so they are NOT rotation-equivariant:
computing them in a reduced basis would not be the arm's block. They are therefore built in the arm's own
480-dim z-scored embedding space and only THEN projected, each block onto its own principal axes, with no
whitening -- so the projection is a rotation plus a truncation, and at npca=480 the model the arm trains and
the model trained here differ only by a rotation of the first layer's input, which a dense layer absorbs.
Truncation is the only approximation, it is the same for every configuration, and its price is measured
(N1 reports npca=480 against the arm's published number).
"""
import pickle, sys, time
from pathlib import Path
import numpy as np

SP = Path('/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad')
CACHE = SP / 'nx' / 'bench.pkl'
AA = "ACDEFGHIKLMNPQRSTVWY"
N_DECOY, N_FOLD, DECOY_SEED = 9, 5, 99
BLOCKS = ['esm_enz', 'esm_sub', 'esm_prod', 'esm_absdiff', 'freq', 'enz_seq', 'sub_seq']
ESM_BLOCKS = ('esm_enz', 'esm_sub', 'esm_prod', 'esm_absdiff')


def load():
    return pickle.load(open(CACHE, 'rb'))


def _pca(M, k, seed=0):
    """principal axes of M, no whitening. Returns (components kxd, mean d)."""
    mu = M.mean(0)
    X = M - mu
    if X.shape[0] > 20000:
        rs = np.random.default_rng(seed)
        X = X[rs.choice(X.shape[0], 20000, replace=False)]
    _U, _S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k], mu


def build_space(npca=64, scheme='homology_disjoint', seed=0):
    B = load()
    E, rx, vocab, cats, seq, fam = B['E'], B['rx'], B['vocab'], B['cats'], B['seq'], B['fam']
    genes = sorted(E)
    gi = {g: i for i, g in enumerate(genes)}
    X = np.stack([E[g] for g in genes]).astype(np.float32)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)          # the arm's own normalisation
    d = Z.shape[1]
    k = min(npca, d)

    # cheap sequence block: aa composition + log length
    A = np.zeros((len(genes), len(AA) + 1), np.float32)
    for g, i in gi.items():
        s = seq.get(g, '')
        if s:
            for j, a in enumerate(AA):
                A[i, j] = s.count(a) / len(s)
            A[i, len(AA)] = np.log(len(s))
    A = (A - A.mean(0)) / (A.std(0) + 1e-8)

    cat_of = np.array([r['cat'] for r in rx])
    fold_by = {'reaction_disjoint': np.arange(len(rx)),
               'enzyme_disjoint': np.array([sorted(set(cat_of)).index(c) for c in cat_of]),
               'homology_disjoint': np.array([fam.get(c, -1 - kk) for kk, c in enumerate(cat_of)])}
    from sklearn.model_selection import GroupKFold
    gk = GroupKFold(n_splits=N_FOLD)
    gid = fold_by[scheme]
    held = scheme in ('enzyme_disjoint', 'homology_disjoint')

    # ---- candidate sets, identical for every configuration -------------------
    raw = []
    for f, (tr, te) in enumerate(gk.split(np.arange(len(rx)), groups=gid)):
        pool_tr = np.array(sorted({rx[i]['cat'] for i in tr})) if held else np.array(vocab)
        pool_te = np.array(sorted({rx[i]['cat'] for i in te})) if held else np.array(vocab)
        cnt = {}
        for i in tr:
            cnt[rx[i]['cat']] = cnt.get(rx[i]['cat'], 0) + 1
        rec = {}
        for tag, idx, pool, fs in (('tr', tr, pool_tr, f), ('te', te, pool_te, 100 + f)):
            ci, si = [], []
            for i in idx:
                r = rx[i]
                rs = np.random.default_rng((DECOY_SEED, int(fs), int(r['rx'])))
                bad = set(cats[r['rx']]) | {r['cat']}
                ok = pool[~np.isin(pool, list(bad))]
                if len(ok) < N_DECOY:
                    continue
                dd = list(rs.choice(ok, N_DECOY, replace=False))
                ci.append([gi[c] for c in [r['cat']] + dd]); si.append(gi[r['sub']])
            ci = np.array(ci, np.int64); si = np.array(si, np.int64)
            fq = np.log1p(np.array([[cnt.get(genes[c], 0) for c in row] for row in ci], np.float32))
            rec[tag] = {'cand': ci, 'sub': si, 'freq': fq[:, :, None]}
        raw.append(rec)

    # ---- principal axes, fitted on the ORIGINAL 480-dim blocks ---------------
    rs = np.random.default_rng(seed)
    ci0 = np.concatenate([r['tr']['cand'].ravel() for r in raw])
    si0 = np.concatenate([np.repeat(r['tr']['sub'], r['tr']['cand'].shape[1]) for r in raw])
    take = rs.choice(len(ci0), min(20000, len(ci0)), replace=False)
    Pe = Z[ci0[take]]
    Ps = Z[si0[take]]
    axes = {'esm_enz': _pca(Z, k), 'esm_sub': _pca(Z, k),
            'esm_prod': _pca(Pe * Ps, k), 'esm_absdiff': _pca(np.abs(Pe - Ps), k)}
    evr = {}
    for nm, M in (('esm_enz', Z), ('esm_prod', Pe * Ps), ('esm_absdiff', np.abs(Pe - Ps))):
        C = M - M.mean(0)
        tot = float((C ** 2).sum())
        pr = C @ axes[nm][0].T
        evr[nm] = float((pr ** 2).sum() / tot) if tot > 0 else 1.0
    evr['esm_sub'] = evr['esm_enz']

    # ---- project every block once, per fold ---------------------------------
    def project(nm, M):
        V, mu = axes[nm]
        return ((M - mu) @ V.T).astype(np.float32)

    folds = []
    scale = {}
    for f, rec in enumerate(raw):
        out = {}
        for tag in ('tr', 'te'):
            ci, si = rec[tag]['cand'], rec[tag]['sub']
            n, m = ci.shape
            e = Z[ci.ravel()]
            s = Z[np.repeat(si, m)]
            blk = {'esm_enz': project('esm_enz', e).reshape(n, m, k),
                   'esm_sub': project('esm_sub', s).reshape(n, m, k),
                   'esm_prod': project('esm_prod', e * s).reshape(n, m, k),
                   'esm_absdiff': project('esm_absdiff', np.abs(e - s)).reshape(n, m, k),
                   'freq': rec[tag]['freq'],
                   'enz_seq': A[ci],
                   'sub_seq': A[si][:, None, :].repeat(m, 1)}
            out[tag] = blk
        folds.append(out)
    for b in BLOCKS:                    # one scalar per block: keeps the rotation, fixes conditioning
        v = float(np.sqrt(np.mean(folds[0]['tr'][b] ** 2))) + 1e-8
        scale[b] = v
        for fd in folds:
            for tag in ('tr', 'te'):
                fd[tag][b] = (fd[tag][b] / v).astype(np.float32)
    return {'genes': genes, 'folds': folds, 'rx': rx, 'evr': evr, 'npca': k,
            'scheme': scheme, 'scale': scale, 'dim': d}


def feats(S, fd, tag, use):
    return np.concatenate([S['folds'][fd][tag][b] for b in use], -1).astype(np.float32)


def run_config(S, use, seeds=(0, 1, 2), epochs=25, h=128):
    """The arm's Pair head, fed only the selected blocks. Returns (mean AUC, mean top-1, sd over seeds)."""
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(4)
    if not use:
        return 0.5, 0.1, 0.0
    pre = [(feats(S, f, 'tr', use), feats(S, f, 'te', use)) for f in range(len(S['folds']))]
    aucs, t1s = [], []
    for seed in seeds:
        ys, ps, t1 = [], [], []
        for f in range(len(S['folds'])):
            Xtr, Xte = pre[f]
            torch.manual_seed(seed)
            m = nn.Sequential(nn.Linear(Xtr.shape[-1], h), nn.GELU(), nn.Dropout(0.3),
                              nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1))
            opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
            Tt = torch.from_numpy(Xtr)
            Y = torch.zeros(Xtr.shape[0], Xtr.shape[1]); Y[:, 0] = 1.0
            n = Xtr.shape[0]
            rs = np.random.default_rng(seed)
            for ep in range(epochs):
                m.train()
                perm = rs.permutation(n)
                for kk in range(0, n, 64):
                    b = perm[kk:kk + 64]
                    opt.zero_grad()
                    loss = -(torch.log_softmax(m(Tt[b]).squeeze(-1), 1) * Y[b]).sum(1).mean()
                    loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                sc = m(torch.from_numpy(Xte)).squeeze(-1).numpy()
            ys.append(np.concatenate([[1] + [0] * (sc.shape[1] - 1)] * sc.shape[0]))
            ps.append(sc.ravel())
            t1 += list((sc.argmax(1) == 0).astype(float))
        aucs.append(float(roc_auc_score(np.concatenate(ys), np.concatenate(ps))))
        t1s.append(float(np.mean(t1)))
    return float(np.mean(aucs)), float(np.mean(t1s)), float(np.std(aucs))
