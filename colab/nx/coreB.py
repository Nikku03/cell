"""Space B: the DOCKABLE reactions, where the docking blocks and the sequence blocks coexist.

Eight binary blocks, 2^8 = 256 configurations -- still enumerable, so the profiler is validated here too.

    dock_shape   best, top10, mean, top50cell     the skin-correlation peak statistics
    dock_spread  std, skew, n_z2                  the shape of the rotation-score distribution
    dock_clash   clash                            the ONLY feature needing the core-erosion transform
    size         n_atoms, diam                    the pilot's own artefact control
    log_len      log residue count                the artefact the atom-count matching left open
    aa_comp      20 amino-acid fractions          free, sequence only
    esm_pair     one out-of-homology ESM score    the whole embedding stage, distilled to a column
    freq         log catalysis count              counted OUTSIDE the docked reactions, so no label offset
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core

DOCK_SHAPE = ('best', 'top10', 'mean', 'top50cell')
DOCK_SPREAD = ('std', 'skew', 'n_z2')
BLOCKS_B = ['dock_shape', 'dock_spread', 'dock_clash', 'size',
            'log_len', 'aa_comp', 'esm_pair', 'freq']
ARTEFACT_BLOCKS = ('size', 'log_len')
DOCK_BLOCKS = ('dock_shape', 'dock_spread', 'dock_clash')
ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def usable_reactions(B):
    """docked reactions whose whole shortlist has a sequence AND an embedding -- two histone multi-gene
    entries have neither, and imputing them would put a constant where a feature should be."""
    D, seq, E = B['dock'], B['seq'], B['E']
    return [r for r in sorted(int(k) for k in D)
            if all(g in seq and seq[g] and g in E for g in D[str(r)]['feats'])]


def esm_scores(B, docked_rx, seeds=(0, 1, 2), epochs=25):
    """Out-of-homology ESM pair scores for every docked candidate.

    The Pair model is the arm's own, at the full 480 dims, trained on every reaction whose catalyst lies in a
    DIFFERENT homology cluster from every one of the docked catalysts. So no docked catalyst, and no
    ~50%-identity paralog of one, is ever a training positive."""
    import torch, torch.nn as nn
    torch.set_num_threads(4)
    E, rx, cats, fam, D = B['E'], B['rx'], B['cats'], B['fam'], B['dock']
    test_cats = {D[str(r)]['catalyst'] for r in docked_rx}
    test_fams = {fam.get(c, -999) for c in test_cats}
    genes = sorted(E)
    gi = {g: i for i, g in enumerate(genes)}
    X = np.stack([E[g] for g in genes]).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    dim = X.shape[1]

    drx = set(docked_rx)
    train = [r for r in rx if fam.get(r['cat'], -1) not in test_fams and r['rx'] not in drx]
    vocab_tr = np.array(sorted({r['cat'] for r in train}))
    ci, si = [], []
    for r in train:
        rs = np.random.default_rng((core.DECOY_SEED, 7, int(r['rx'])))
        bad = set(cats[r['rx']]) | {r['cat']}
        ok = vocab_tr[~np.isin(vocab_tr, list(bad))]
        if len(ok) < core.N_DECOY:
            continue
        d = list(rs.choice(ok, core.N_DECOY, replace=False))
        ci.append([gi[c] for c in [r['cat']] + d]); si.append(gi[r['sub']])
    ci, si = np.array(ci), np.array(si)

    def pack(c_idx, s_idx):
        e = X[c_idx]; s = X[s_idx][:, None, :].repeat(e.shape[1], 1)
        return torch.from_numpy(np.concatenate([e, s, e * s, np.abs(e - s)], -1))

    Ttr = pack(ci, si)
    Ytr = torch.zeros(len(ci), ci.shape[1]); Ytr[:, 0] = 1.0
    sub_of = {r['rx']: r['sub'] for r in rx}
    order, tc, ts = [], [], []
    for r in docked_rx:
        gs = sorted(D[str(r)]['feats'])
        order.append((r, gs))
        tc.append([gi[g] for g in gs]); ts.append(gi[sub_of[r]])
    Tte = pack(np.array(tc), np.array(ts))

    out = np.zeros((len(order), len(order[0][1])))
    for seed in seeds:
        torch.manual_seed(seed)
        m = nn.Sequential(nn.Linear(4 * dim, 128), nn.GELU(), nn.Dropout(0.3),
                          nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        rs = np.random.default_rng(seed)
        for ep in range(epochs):
            m.train()
            perm = rs.permutation(len(ci))
            for k in range(0, len(ci), 64):
                b = perm[k:k + 64]
                opt.zero_grad()
                loss = -(torch.log_softmax(m(Ttr[b]).squeeze(-1), 1) * Ytr[b]).sum(1).mean()
                loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            s = m(Tte).squeeze(-1).numpy()
        out += (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)
    return order, out / len(seeds), int(len(ci))


def build_spaceB(B, esm_cache=None, seeds=(0, 1, 2)):
    docked_rx = usable_reactions(B)
    if esm_cache is not None and Path(esm_cache).exists():
        z = np.load(esm_cache, allow_pickle=True)
        order = [(int(r), list(g)) for r, g in zip(z['rxs'], z['cands'])]
        ES, ntr = z['S'], int(z['ntr'])
        assert [o[0] for o in order] == docked_rx, "cached ESM scores are for a different reaction set"
    else:
        order, ES, ntr = esm_scores(B, docked_rx, seeds=seeds)
        if esm_cache is not None:
            np.savez(esm_cache, rxs=np.array([o[0] for o in order]),
                     cands=np.array([o[1] for o in order]), S=ES, ntr=ntr)
    D, seq, cats = B['dock'], B['seq'], B['cats']

    # catalysis counts from every reaction OUTSIDE the docked set: no label-dependent offset, and it is
    # what an orphan reaction would actually have available
    drx = set(docked_rx)
    freq_all = {}
    for i, cs in cats.items():
        if int(i) in drx:
            continue
        for g in cs:
            freq_all[g] = freq_all.get(g, 0) + 1

    Xb, Y, G = {b: [] for b in BLOCKS_B}, [], []
    for gi_, (r, gs) in enumerate(order):
        F = D[str(r)]['feats']
        cat = D[str(r)]['catalyst']
        Xb['dock_shape'].append(np.array([[F[g][k] for k in DOCK_SHAPE] for g in gs], float))
        Xb['dock_spread'].append(np.array([[F[g][k] for k in DOCK_SPREAD] for g in gs], float))
        Xb['dock_clash'].append(np.array([[F[g]['clash']] for g in gs], float))
        Xb['size'].append(np.array([[F[g]['n_atoms'], F[g]['diam']] for g in gs], float))
        Xb['log_len'].append(np.array([[np.log(len(seq[g]))] for g in gs], float))
        Xb['aa_comp'].append(np.array([[seq[g].count(a) / len(seq[g]) for a in core.AA] for g in gs],
                                      float))
        Xb['esm_pair'].append(ES[gi_][:, None].astype(float))
        Xb['freq'].append(np.array([[np.log1p(freq_all.get(g, 0))] for g in gs], float))
        Y.append(np.array([1.0 if g == cat else 0.0 for g in gs]))
        G.append(r)
    for b in BLOCKS_B:                       # z within the shortlist, as the pilot and nn arms do
        Xb[b] = [(m - m.mean(0)) / (m.std(0) + 1e-9) for m in Xb[b]]
    return {'X': Xb, 'Y': Y, 'G': np.array(G), 'ntrain_esm': ntr, 'order': order}


def objectiveB(SB, use, folds=None, yperm=None):
    """listwise ridge on within-shortlist z-scores, 5-fold grouped by reaction, alpha by inner CV.
    Deterministic given the fold assignment."""
    from sklearn.metrics import roc_auc_score
    Y = SB['Y'] if yperm is None else yperm
    n = len(Y)
    if not use:
        return 0.5
    Xs = [np.concatenate([SB['X'][b][i] for b in use], 1) for i in range(n)]
    if folds is None:
        folds = np.arange(n) % 5
    yy, pp = [], []
    for f in range(5):
        tr = [i for i in range(n) if folds[i] != f]
        te = [i for i in range(n) if folds[i] == f]
        Xtr = np.vstack([Xs[i] for i in tr]); ytr = np.concatenate([Y[i] for i in tr])
        ytr = ytr - ytr.mean()
        best, ba = -9.0, ALPHAS[0]
        for a in ALPHAS:
            sc = []
            for k in range(3):
                i1 = [tr[j] for j in range(len(tr)) if j % 3 != k]
                i2 = [tr[j] for j in range(len(tr)) if j % 3 == k]
                A_ = np.vstack([Xs[i] for i in i1]); b_ = np.concatenate([Y[i] for i in i1])
                b_ = b_ - b_.mean()
                w = np.linalg.solve(A_.T @ A_ + a * np.eye(A_.shape[1]), A_.T @ b_)
                v = np.concatenate([Xs[i] @ w for i in i2]); t = np.concatenate([Y[i] for i in i2])
                sc.append(roc_auc_score(t, v) if len(set(t)) > 1 else 0.5)
            if float(np.mean(sc)) > best:
                best, ba = float(np.mean(sc)), a
        w = np.linalg.solve(Xtr.T @ Xtr + ba * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
        for i in te:
            pp.append(Xs[i] @ w); yy.append(Y[i])
    return float(roc_auc_score(np.concatenate(yy), np.concatenate(pp)))
