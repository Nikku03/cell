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
import os
import numpy as np
from pathlib import Path
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan")) / "invivo"
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
                nn.Conv1d(4, 128, 19, padding=9), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
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


def run(seed=0, lam=1.0, epochs=40, shuffle_labels=False, tag="", device=None):
    import torch, torch.nn as nn, time
    from sklearn.metrics import average_precision_score
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
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
        t0 = time.time()
        tr = folds != k; te = folds == k
        if y[te].sum() < 3:
            continue
        net = _model(ntf, T.shape[1], L).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_w = torch.tensor(max((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), 1.0), dtype=torch.float32).to(dev)
        bce_main = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        bce_aux = nn.BCEWithLogitsLoss()
        Xtr = torch.tensor(X[tr]).to(dev); Ttr = torch.tensor(T[tr]).to(dev)
        Atr = torch.tensor(A[tr]).to(dev); ytr = torch.tensor(y[tr]).to(dev)
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
            pm, _ = net(torch.tensor(X[te]).to(dev), torch.tensor(T[te]).to(dev))
            oof[te] = torch.sigmoid(pm).cpu().numpy()
        ap = average_precision_score(y[te], oof[te]) if y[te].sum() >= 3 else float("nan")
        print(f"    [{tag}] fold {k+1}/5  AUPRC {ap:.3f}  ({time.time()-t0:.0f}s, {int(te.sum())} test)", flush=True)
    # per-fold AUPRC then mean (match the GBM protocol)
    aps = []
    for k in range(5):
        te = folds == k
        if y[te].sum() >= 3:
            aps.append(average_precision_score(y[te], oof[te]))
    return float(np.mean(aps))


# ======================================================================================================================
#  THE FULL ARCHITECTURE: pretrain the shared encoder on genome-wide cCREs (311-TF binding), THEN fine-tune the CRISPR head.
#  This is the version designed to solve data starvation — the aux task sees ~100k genome-wide loci, not the 4k CRISPR set.
# ======================================================================================================================
PT = OUT / "pretrain"


def _encoder(dev):
    import torch, torch.nn as nn
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv1d(4, 128, 19, padding=9), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(128, 128, 7, padding=6, dilation=2), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 128, 7, padding=12, dilation=4), nn.BatchNorm1d(128), nn.ReLU())

        def forward(self, x):
            h = self.enc(x)
            return torch.cat([h.max(-1).values, h.mean(-1)], -1)     # 256-d
    return Encoder().to(dev)


def _oh(codes, dev):
    """(B,L) int codes 0-4 -> (B,4,L) float one-hot on device (4 = N -> all zero)."""
    import torch
    c = torch.as_tensor(codes, dtype=torch.long, device=dev)
    oh = torch.zeros((c.shape[0], 5, c.shape[1]), device=dev)
    oh.scatter_(1, c.unsqueeze(1), 1.0)
    return oh[:, :4, :]


def load_pretrain():
    """cCRE codes (N,600) int8 + 311-TF labels (N,311) — the genome-wide auxiliary dataset."""
    import gzip
    meta = json.load(open(PT / "pretrain_meta.json"))
    L = meta["window"]
    seqs = [s for s in gzip.open(PT / "pretrain_seqs.txt.gz", "rt").read().split("\n") if len(s) >= L]
    buf = np.frombuffer("".join(s[:L] for s in seqs).encode(), dtype=np.uint8).reshape(len(seqs), L)
    codes = np.full((len(seqs), L), 4, dtype=np.int8)          # 4 = N
    for ch, v in NT.items():
        codes[buf == ord(ch)] = v
    lab = np.load(PT / "pretrain_labels.npz")
    labels = np.unpackbits(lab["labels"], axis=1)[:, :int(lab["n_tf"])].astype(np.float32)
    return codes, labels, meta["tf_list"]


def pretrain(epochs=12, dev=None, log=print):
    """train the shared encoder + a 311-TF head to predict TF binding from sequence on ~100k genome-wide cCREs."""
    import torch, torch.nn as nn, time
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    codes, labels, tfl = load_pretrain()
    n = len(codes); rng = np.random.default_rng(0); perm = rng.permutation(n)
    val = perm[:5000]; tr = perm[5000:]
    enc = _encoder(dev); aux = nn.Linear(256, labels.shape[1]).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(aux.parameters()), lr=1e-3, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    codes_tr = codes[tr]; codes_val = codes[val]                 # slice once (not per batch)
    Ytr = torch.tensor(labels[tr]).to(dev)
    from sklearn.metrics import roc_auc_score
    for ep in range(epochs):
        enc.train(); aux.train(); t0 = time.time(); idx = rng.permutation(len(tr))
        for b in range(0, len(idx), 256):
            bi = idx[b:b + 256]
            opt.zero_grad()
            emb = enc(_oh(codes_tr[bi], dev))
            loss = bce(aux(emb), Ytr[bi])
            loss.backward(); opt.step()
        # validation aux AUC (macro over a sample of TFs)
        enc.eval(); aux.eval()
        with torch.no_grad():
            vp = []
            for b in range(0, len(val), 512):
                vp.append(torch.sigmoid(aux(enc(_oh(codes_val[b:b + 512], dev)))).cpu().numpy())
            vp = np.concatenate(vp); vy = labels[val]
        aucs = [roc_auc_score(vy[:, t], vp[:, t]) for t in range(vy.shape[1]) if 0 < vy[:, t].sum() < len(vy)]
        log(f"    pretrain epoch {ep+1}/{epochs}  aux TF-binding AUC {np.mean(aucs):.3f}  ({time.time()-t0:.0f}s)")
    return {k: v.cpu() for k, v in enc.state_dict().items()}


def run_finetune(pretrained=None, seed=0, epochs=40, lam=0.3, shuffle_labels=False, tag="", dev=None):
    """fine-tune: CRISPR head on top of the (optionally pretrained) shared encoder, chromosome-held-out AUPRC."""
    import torch, torch.nn as nn, time
    from sklearn.metrics import average_precision_score
    torch.manual_seed(seed)
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    X, T, A, y, chrom, ntf, L = load_data()
    if shuffle_labels:
        yy = y.copy(); rng = np.random.default_rng(seed)
        for c in sorted(set(chrom)):
            m = chrom == c; v = yy[m].copy(); rng.shuffle(v); yy[m] = v
        y = yy
    folds = _seeded_folds(chrom, seed); oof = np.zeros(len(y))
    for k in range(5):
        t0 = time.time(); tr = folds != k; te = folds == k
        if y[te].sum() < 3:
            continue
        enc = _encoder(dev)
        if pretrained is not None:
            enc.load_state_dict({kk: vv.to(dev) for kk, vv in pretrained.items()})
        aux = nn.Linear(256, ntf).to(dev)
        main = nn.Sequential(nn.Linear(256 + T.shape[1], 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)).to(dev)
        opt = torch.optim.Adam(list(enc.parameters()) + list(aux.parameters()) + list(main.parameters()),
                               lr=5e-4, weight_decay=1e-4)
        pw = torch.tensor(max((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), 1.0)).to(dev)
        bm = nn.BCEWithLogitsLoss(pos_weight=pw); ba = nn.BCEWithLogitsLoss()
        Xtr = torch.tensor(X[tr]).to(dev); Ttr = torch.tensor(T[tr]).to(dev)
        Atr = torch.tensor(A[tr]).to(dev); ytr = torch.tensor(y[tr]).to(dev)
        idx = np.arange(len(ytr))
        for ep in range(epochs):
            enc.train(); main.train(); np.random.default_rng(seed * 100 + ep).shuffle(idx)
            for b in range(0, len(idx), 128):
                bi = idx[b:b + 128]; opt.zero_grad()
                emb = enc(Xtr[bi])
                pm = main(torch.cat([emb, Ttr[bi]], -1)).squeeze(-1)
                loss = bm(pm, ytr[bi]) + lam * ba(aux(emb), Atr[bi])
                loss.backward(); opt.step()
        enc.eval(); main.eval()
        with torch.no_grad():
            emb = enc(torch.tensor(X[te]).to(dev))
            oof[te] = torch.sigmoid(main(torch.cat([emb, torch.tensor(T[te]).to(dev)], -1)).squeeze(-1)).cpu().numpy()
        ap = average_precision_score(y[te], oof[te]) if y[te].sum() >= 3 else float("nan")
        print(f"    [{tag}] fold {k+1}/5  AUPRC {ap:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    aps = [average_precision_score(y[folds == k], oof[folds == k]) for k in range(5) if y[folds == k].sum() >= 3]
    return float(np.mean(aps))


def main_full():
    """the full experiment: pretrain on cCREs, then fine-tune WITH vs WITHOUT pretraining. Does pretraining break the gate?"""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 96)
    print(f"REAL TIER-2 — genome-wide aux pretraining (cCREs -> 311-TF) then fine-tune CRISPR head (device: {dev.upper()})")
    print("=" * 96)
    print("\n  [1] PRETRAIN shared encoder on ~100k genome-wide cCREs (predict 311-TF binding from sequence):")
    enc_state = pretrain(dev=dev)
    print("\n  [2] FINE-TUNE — WITHOUT pretraining (from scratch = the reduced Tier-2):")
    scratch = [run_finetune(pretrained=None, seed=s, tag=f"scratch-s{s}", dev=dev) for s in (0, 1)]
    print("\n  [3] FINE-TUNE — WITH genome-wide pretraining (the full architecture):")
    pre = [run_finetune(pretrained=enc_state, seed=s, tag=f"pretrained-s{s}", dev=dev) for s in (0, 1)]
    print("\n  [4] shuffle control (pretrained):")
    shuf = run_finetune(pretrained=enc_state, seed=0, shuffle_labels=True, tag="shuf", dev=dev)
    sc = float(np.mean(scratch)); pr = float(np.mean(pre))
    print("\n  " + "=" * 60)
    print(f"  from-scratch fine-tune (reduced Tier-2) : AUPRC {sc:.3f}  {[round(x,3) for x in scratch]}")
    print(f"  PRETRAINED fine-tune (full Tier-2)      : AUPRC {pr:.3f}  {[round(x,3) for x in pre]}")
    print(f"  shuffle control                         : AUPRC {shuf:.3f}  (~base rate 0.055)")
    print("  baselines: distance 0.393 | epigenetics-GBM 0.519 | TF-identity GBM 0.608")
    if pr >= 0.608:
        verdict = "PASS (strong) -- genome-wide pretraining lets the sequence model MATCH/BEAT the measured-ChIP GBM. Adopt."
    elif pr >= 0.519 + 0.02 and pr >= sc + 0.03:
        verdict = ("PASS -- genome-wide pretraining works: it beats epigenetics-only AND lifts the from-scratch model. The "
                   "encoder learned the TF grammar from millions of loci and transferred it to the sparse CRISPR task, ChIP-free.")
    elif pr >= sc + 0.03:
        verdict = "PARTIAL -- pretraining helps vs from-scratch but still trails the epigenetics/ChIP GBM; a bigger encoder or more cCREs may close it."
    else:
        verdict = "NO-GO -- even genome-wide pretraining does not beat the GBM; the measured ChIP features remain the best. Ship the GBM."
    print(f"\n  VERDICT: {verdict}")
    out = {"pretrained_auprc": round(pr, 3), "scratch_auprc": round(sc, 3), "pretrained_seeds": [round(x, 3) for x in pre],
           "scratch_seeds": [round(x, 3) for x in scratch], "shuffle_control": round(shuf, 3),
           "baselines": {"distance": 0.393, "epigenetics_gbm": 0.519, "tf_identity_gbm": 0.608}, "verdict": verdict,
           "note": "FULL Tier-2: shared encoder PRETRAINED on ~100k genome-wide ENCODE cCREs to predict 311-TF ChIP binding "
                   "from sequence (the dense aux task that solves data starvation), THEN fine-tuned with the sparse CRISPR "
                   "head. Compares WITH vs WITHOUT pretraining to isolate the pretraining benefit. Chromosome-held-out AUPRC, "
                   "label-shuffle control. This is the architecture as designed (unlike the earlier reduced 0.488 version whose "
                   "aux head saw only the 4k CRISPR elements). Real ENCODE K562 data; Colab GPU."}
    json.dump(out, open("outputs/orphan/seq_model_full.json", "w"), indent=1)
    print("\n  -> outputs/orphan/seq_model_full.json")
    return out


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 92)
    print(f"TIER-2 — multi-task sequence CNN for regulation (device: {dev.upper()}). Go/No-Go vs the Tier-1 GBM.")
    print("=" * 92)
    scores = []
    for s in (0, 1, 2):
        print(f"  seed {s}:", flush=True)
        sc = run(seed=s, tag=f"s{s}"); scores.append(sc)
        print(f"  -> seed {s} mean AUPRC {sc:.3f}", flush=True)
    seq_auprc = float(np.mean(scores))
    print("  shuffle control:", flush=True)
    shuf = run(seed=0, shuffle_labels=True, tag="shuf")
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
