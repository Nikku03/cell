"""seq_model — Tier-2 of the ML strategy: a multi-task sequence CNN for CRISPR element->gene regulation, and the honest
Go/No-Go against the Tier-1 GBM (0.608 with measured 311-TF ChIP; 0.519 epigenetics-only; 0.393 distance).

Architecture (the strategy's dense-to-sparse idea, concrete):
  element DNA (600bp, one-hot)  --conv motif encoder-->  embedding
     |-- AUX head:  predict which of 311 TFs bind (dense: ~1.2M labels) -> forces the encoder to LEARN the TF motif grammar
     |-- MAIN head: [embedding + 8 tabular epigenetic/distance features] -> regulation (CRISPR hit, sparse: 569 positives)
  loss = main + lambda*aux.  The dense aux task regularizes the sparse regulation head.

Why the bar is subtle: the 0.608 GBM uses MEASURED ChIP for all 311 TFs; this model must PREDICT binding from sequence
(inherently lossy). So the honest win is (a) beat epigenetics-only 0.519 -> it learned TF grammar from sequence; and
(b) approach 0.608 WITHOUT ChIP at inference -> a portable, ChIP-free regulation predictor. Evaluated chromosome-held-out,
AUPRC, with a label-shuffle control. Real ENCODE K562 data; CPU-trained.
"""
import json
import numpy as np
from pathlib import Path
OUT = Path("outputs/orphan/invivo")
NT = {"A": 0, "C": 1, "G": 2, "T": 3}
TAB = ["log_dist", "atac_enh", "h3k27ac_enh", "polr2a_enh", "procap_enh", "promoter_atac", "promoter_polii", "gene_expr"]


def _onehot(seq, L=600):
    a = np.zeros((4, L), dtype=np.float32)
    for i, ch in enumerate(seq[:L]):
        j = NT.get(ch)
        if j is not None:
            a[j, i] = 1.0
    return a


def load_data():
    import pandas as pd
    seqs = json.load(open(OUT / "element_seqs.json"))
    comp = json.load(open(OUT / "compendium_tf.json"))
    et = comp["element_tfs"]; ntf = len(comp["tf_list"])
    df = pd.read_csv("outputs/orphan/crispr_features_compendium.csv")
    df = df[df["element"].isin(seqs)].reset_index(drop=True)
    L = min(len(next(iter(seqs.values()))), 600)
    X = np.stack([_onehot(seqs[e], L) for e in df["element"]])
    # standardize tabular
    T = df[TAB].values.astype(np.float32)
    T = (T - T.mean(0)) / (T.std(0) + 1e-6)
    # aux TF-binding labels per element
    A = np.zeros((len(df), ntf), dtype=np.float32)
    for i, e in enumerate(df["element"].values):
        for ti in et.get(e, []):
            A[i, ti] = 1.0
    y = df["crispr_hit"].values.astype(np.float32)
    chrom = df["chromosome"].values
    return X, T, A, y, chrom, ntf, L


def _model(ntf, ntab, L):
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv1d(4, 128, 19, padding=9), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 128, 7, padding=6, dilation=2), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 128, 7, padding=12, dilation=4), nn.BatchNorm1d(128), nn.ReLU())
            self.aux = nn.Linear(256, ntf)                      # predict TF binding from seq
            self.main = nn.Sequential(nn.Linear(256 + ntab, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

        def embed(self, x):
            h = self.enc(x)
            return __import__("torch").cat([h.max(-1).values, h.mean(-1)], -1)

        def forward(self, x, t):
            e = self.embed(x)
            return self.main(__import__("torch").cat([e, t], -1)).squeeze(-1), self.aux(e)
    return Net()


def _seeded_folds(chrom, seed):
    import random
    uch = sorted(set(chrom)); random.seed(seed); sh = uch[:]; random.shuffle(sh)
    fold = {c: i % 5 for i, c in enumerate(sh)}
    return np.array([fold[c] for c in chrom])


def run(seed=0, lam=1.0, epochs=45, shuffle_labels=False):
    import torch, torch.nn as nn
    from sklearn.metrics import average_precision_score
    torch.manual_seed(seed)
    X, T, A, y, chrom, ntf, L = load_data()
    if shuffle_labels:
        yy = y.copy()
        rng = np.random.default_rng(seed)
        for c in sorted(set(chrom)):
            m = chrom == c; v = yy[m].copy(); rng.shuffle(v); yy[m] = v
        y = yy
    folds = _seeded_folds(chrom, seed)
    oof = np.zeros(len(y))
    for k in range(5):
        tr = folds != k; te = folds == k
        if y[te].sum() < 3:
            continue
        net = _model(ntf, T.shape[1], L)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_w = torch.tensor(max((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), 1.0), dtype=torch.float32)
        bce_main = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        bce_aux = nn.BCEWithLogitsLoss()
        Xtr = torch.tensor(X[tr]); Ttr = torch.tensor(T[tr]); Atr = torch.tensor(A[tr]); ytr = torch.tensor(y[tr])
        idx = np.arange(len(ytr))
        net.train()
        for ep in range(epochs):
            np.random.default_rng(seed * 100 + ep).shuffle(idx)
            for b in range(0, len(idx), 128):
                bi = idx[b:b + 128]
                opt.zero_grad()
                pm, pa = net(Xtr[bi], Ttr[bi])
                loss = bce_main(pm, ytr[bi]) + lam * bce_aux(pa, Atr[bi])
                loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pm, _ = net(torch.tensor(X[te]), torch.tensor(T[te]))
            oof[te] = torch.sigmoid(pm).numpy()
    # per-fold AUPRC then mean (match the GBM protocol)
    aps = []
    for k in range(5):
        te = folds == k
        if y[te].sum() >= 3:
            aps.append(average_precision_score(y[te], oof[te]))
    return float(np.mean(aps))


def main():
    print("=" * 92)
    print("TIER-2 — multi-task sequence CNN for regulation (CPU). Go/No-Go vs the Tier-1 GBM.")
    print("=" * 92)
    scores = [run(seed=s) for s in (0, 1, 2)]
    seq_auprc = float(np.mean(scores))
    shuf = run(seed=0, shuffle_labels=True)
    print(f"\n  sequence CNN (multi-task) AUPRC: {seq_auprc:.3f}  seeds {[round(x,3) for x in scores]}")
    print(f"  label-shuffle control:           {shuf:.3f}  (~base rate; must be low)")
    print("\n  vs Tier-1 baselines (same chromosome-held-out protocol):")
    print("    distance-only ................ 0.393")
    print("    epigenetics-only GBM ......... 0.519   <- BEAT THIS = the model learned TF grammar from SEQUENCE")
    print("    TF-identity GBM (311 ChIP) ... 0.608   <- measured ChIP; approaching it ChIP-free is the big win")
    if seq_auprc >= 0.608:
        verdict = "PASS (strong) -- the sequence model MATCHES/BEATS the measured-ChIP GBM without needing ChIP at inference. Adopt Tier-2."
    elif seq_auprc >= 0.519 + 0.02:
        verdict = ("PASS -- the sequence model beats epigenetics-only, i.e. it LEARNED the TF grammar from sequence and adds "
                   "regulation signal ChIP-free. It trails the measured-ChIP GBM (expected: sequence predicts binding, ChIP measures it).")
    elif seq_auprc >= 0.519 - 0.02:
        verdict = ("MATCH -- the sequence model ties epigenetics-only: it recovers about as much as the activity tracks but does "
                   "not clearly add sequence-learned TF signal on this small label set. ChIP-feature GBM remains the production model.")
    else:
        verdict = ("NO-GO -- the sequence model underperforms epigenetics-only; 569 positives are too few for the CNN to beat the "
                   "GBM. Ship the Tier-1 GBM (0.608); revisit Tier-2 only with more labelled regulation data.")
    print(f"\n  VERDICT: {verdict}")
    out = {"seq_cnn_auprc": round(seq_auprc, 3), "seeds": [round(x, 3) for x in scores],
           "shuffle_control": round(shuf, 3),
           "baselines": {"distance": 0.393, "epigenetics_gbm": 0.519, "tf_identity_gbm": 0.608},
           "verdict": verdict,
           "note": "Tier-2 multi-task sequence CNN (element DNA -> aux 311-TF-binding head [learns motif grammar] + regulation "
                   "head on embedding+8 tabular features). Chromosome-held-out AUPRC vs Tier-1 GBMs. CPU-trained, 3 seeds, "
                   "label-shuffle control. Honest bars: beat epigenetics-only 0.519 (learned TF grammar from sequence); "
                   "approach 0.608 (measured-ChIP GBM) ChIP-free. Real ENCODE K562 CRISPR data."}
    json.dump(out, open("outputs/orphan/seq_model.json", "w"), indent=1)
    print("\n  -> outputs/orphan/seq_model.json")
    return out


if __name__ == "__main__":
    main()
