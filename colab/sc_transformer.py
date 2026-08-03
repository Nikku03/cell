"""STEP 5: a small TRANSFORMER on the same task, same holdout, same metric as the conditioned ADRN.

WHY A TRANSFORMER HERE AT ALL.  The ridge in sc_adrn.py is linear in its features, and the interaction arm has to
be hand-built as an explicit outer product. A transformer can learn which channels matter for which state without
being told the form of the interaction. That is a real difference in hypothesis class, and it is the only honest
reason to spend the compute -- not because transformers are fashionable.

WHAT IT SEES, identical in content to the ADRN so the comparison is about the model and not the inputs:

    tokens   the perturbation's ACTIVE annotation channels, as a set -- attention over which functional
             memberships co-occur, rather than a fixed 696-dim vector.
    state    s score, g2m score, log depth, state-bin one-hot, line one-hot.

TRAINED PER CELL, EVALUATED PER GROUP.  78,608 perturbed cells are a real training set for a neural model, while
the ~1,400 groups are not. But a single cell's profile is mostly Poisson noise, so scoring is done on the
(perturbation x state bin x line) group means -- exactly the unit sc_adrn.py uses, with the same 40 held-out
perturbations, so the two models' numbers sit in the same table.

ARMS, mirroring the ADRN gates:

    tf_gene        channels only, state inputs zeroed. The transformer's version of pseudobulk.
    tf_gene_state  channels and state. Attention can form the interaction internally.

PREDECLARED: tf_gene_state > tf_gene is the same G1 question the ADRN asks, and the cross-model comparison
(transformer vs ridge at the same arm) prices whether the extra hypothesis class buys anything.

COMPUTE, stated plainly: 4 CPU cores, no GPU. d_model 128, 2 layers, 4 heads is ~1M parameters. This is a small
model by construction and a null result from it does NOT establish that a large one would fail -- it establishes
what is reachable with this data at this scale. That distinction is recorded here rather than argued later.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

OUT, SP = A.OUT, A.SP
NBINS, N_HOLDOUT, MIN_GROUP = 3, 40, 25
D_MODEL, N_LAYER, N_HEAD, EPOCHS, BATCH, LR = 128, 2, 4, 6, 256, 1e-3
MAX_TOK, NPRED = 64, 20


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("STEP 5 -- SMALL TRANSFORMER, same task / holdout / metric as the conditioned ADRN")
    report("=" * 100)

    import torch
    import torch.nn as nn
    torch.manual_seed(A.SEED)
    torch.set_num_threads(4)

    z = np.load(SP / "sc_cohort_cells.npz", allow_pickle=True)
    X = z["X"]
    line = np.array([str(x) for x in z["line"]])
    pert = np.array([str(x) for x in z["pert"]])
    batch = z["batch"]
    ntc = z["is_ntc"]
    s_sc, g2, umi = (z["s_score"].astype(np.float64), z["g2m_score"].astype(np.float64),
                     z["umi"].astype(np.float64))
    perts = sorted(set(pert[~ntc]))
    lines = sorted(set(line))
    nG = X.shape[1]

    zk = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in zk["kos"]]
    universe = sorted(set(kos) | set(perts))
    base, _ = A.build_channels(universe, set(kos), report)
    extra, _, tb = C.extra_channels(universe, set(kos), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}
    n_chan = chan.shape[1]
    tok = {}
    for p in perts:
        act = np.where(chan[urow[p]] > 0)[0][:MAX_TOK]
        tok[p] = (act + 1).astype(np.int64)          # 0 reserved for padding
    report(f"  {len(perts)} perturbations; channel vocabulary {n_chan}; "
           f"mean active channels {np.mean([len(v) for v in tok.values()]):.1f}")

    rng = np.random.default_rng(A.SEED)
    hold = set(np.array(perts)[rng.choice(len(perts), N_HOLDOUT, replace=False)].tolist())
    report(f"  held out {len(hold)} perturbations entirely")

    binid = np.zeros(len(X), np.int64)
    for ln in lines:
        m = line == ln
        q = np.quantile(g2[m], np.linspace(0, 1, NBINS + 1)[1:-1])
        binid[m] = np.digitize(g2[m], q)

    # per-cell targets: cell minus matched (line, batch-set, bin) control mean
    ctrl_mean = {}
    for ln in lines:
        for b in range(NBINS):
            m = (line == ln) & ntc & (binid == b)
            if m.sum() >= 15:
                ctrl_mean[(ln, b)] = X[m].mean(0)
    keep = np.where(~ntc & np.array([(l, b) in ctrl_mean for l, b in zip(line, binid)]))[0]
    report(f"  {len(keep):,} perturbed cells with a matched control group")

    Ytr_cells = np.empty((len(keep), nG), np.float32)
    for i, c in enumerate(keep):
        Ytr_cells[i] = X[c] - ctrl_mean[(line[c], binid[c])]

    st_all = np.stack([(s_sc - s_sc.mean()) / (s_sc.std() + 1e-9),
                       (g2 - g2.mean()) / (g2.std() + 1e-9),
                       (np.log10(umi) - np.log10(umi).mean()) / (np.log10(umi).std() + 1e-9)], 1)
    st_cells = np.concatenate([st_all[keep],
                               np.eye(NBINS, dtype=np.float64)[binid[keep]],
                               np.eye(len(lines), dtype=np.float64)[[lines.index(l) for l in line[keep]]]], 1)
    n_state = st_cells.shape[1]
    is_hold_cell = np.array([pert[c] in hold for c in keep])
    tr_idx = np.where(~is_hold_cell)[0]
    report(f"  training cells {len(tr_idx):,} | state features {n_state}")

    TOKS = np.zeros((len(keep), MAX_TOK), np.int64)
    for i, c in enumerate(keep):
        t = tok[pert[c]]
        TOKS[i, :len(t)] = t

    class Net(nn.Module):
        def __init__(self, use_state):
            super().__init__()
            self.use_state = use_state
            self.emb = nn.Embedding(n_chan + 1, D_MODEL, padding_idx=0)
            self.st = nn.Sequential(nn.Linear(n_state, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
            enc = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_MODEL * 2, batch_first=True, dropout=0.1)
            self.tr = nn.TransformerEncoder(enc, N_LAYER)
            self.head = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, nG))

        def forward(self, t, s):
            h = self.emb(t)
            if self.use_state:
                h = h + self.st(s).unsqueeze(1)
            mask = t == 0
            h = self.tr(h, src_key_padding_mask=mask)
            h = (h * (~mask).unsqueeze(-1)).sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
            return self.head(h)

    # evaluation groups, identical construction to sc_adrn.py
    groups, Yg = [], []
    for ln in lines:
        for p in sorted(hold):
            for b in range(NBINS):
                m = (line == ln) & (pert == p) & ~ntc & (binid == b)
                if m.sum() < MIN_GROUP or (ln, b) not in ctrl_mean:
                    continue
                groups.append((ln, p, b))
                Yg.append(X[m].mean(0) - ctrl_mean[(ln, b)])
    Yg = np.array(Yg, np.float32)
    report(f"  held-out evaluation groups: {len(groups)}")

    gt = np.zeros((len(groups), MAX_TOK), np.int64)
    gs = np.zeros((len(groups), n_state), np.float64)
    for i, (ln, p, b) in enumerate(groups):
        t = tok[p]
        gt[i, :len(t)] = t
        m = (line == ln) & (pert == p) & ~ntc & (binid == b)
        gs[i] = np.concatenate([st_all[m].mean(0), np.eye(NBINS)[b], np.eye(len(lines))[lines.index(ln)]])

    results = {}
    for arm, use_state in (("tf_gene", False), ("tf_gene_state", True)):
        net = Net(use_state)
        npar = sum(p.numel() for p in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
        Tt = torch.from_numpy(TOKS[tr_idx])
        Ts = torch.from_numpy(st_cells[tr_idx]).float()
        Ty = torch.from_numpy(Ytr_cells[tr_idx])
        report(f"\n  {arm}: {npar/1e6:.2f}M params, {len(tr_idx):,} cells, {EPOCHS} epochs")
        for ep in range(EPOCHS):
            net.train()
            perm = torch.randperm(len(tr_idx))
            tot, nb = 0.0, 0
            te = time.time()
            for i in range(0, len(perm), BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad()
                loss = ((net(Tt[idx], Ts[idx]) - Ty[idx]) ** 2).mean()
                loss.backward()
                opt.step()
                tot += float(loss)
                nb += 1
            report(f"    epoch {ep+1}/{EPOCHS}  loss {tot/max(nb,1):.5f}  ({time.time()-te:.0f}s)")
        net.eval()
        with torch.no_grad():
            P = net(torch.from_numpy(gt), torch.from_numpy(gs).float()).numpy()
        r = np.array([float(np.corrcoef(P[i], Yg[i])[0, 1]) if np.std(P[i]) > 0 else 0.0
                      for i in range(len(Yg))])
        pr = []
        for i in range(len(Yg)):
            truth = set(np.argsort(-np.abs(Yg[i]))[:NPRED].tolist())
            pr.append(len(set(np.argsort(-np.abs(P[i]))[:NPRED].tolist()) & truth) / float(NPRED))
        r, pr = np.nan_to_num(r), np.array(pr)
        results[arm] = {"params": int(npar), "r": float(r.mean()), "p20": float(pr.mean()),
                        "r_vec": r.tolist(), "p_vec": pr.tolist()}
        report(f"    {arm}: r {r.mean():.4f}  p@20 {pr.mean():.4f}")

    d_r = np.array(results["tf_gene_state"]["r_vec"]) - np.array(results["tf_gene"]["r_vec"])
    d_p = np.array(results["tf_gene_state"]["p_vec"]) - np.array(results["tf_gene"]["p_vec"])
    br = np.random.default_rng(0)
    out = {}
    for nm, d in (("r", d_r), ("p@20", d_p)):
        bs = d[br.integers(0, len(d), size=(10000, len(d)))].mean(1)
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        out[nm] = {"delta": float(d.mean()), "lo": lo, "hi": hi,
                   "verdict": "RESOLVED +" if lo > 0 else ("RESOLVED -" if hi < 0 else "unresolved")}
        report(f"\n  tf_gene_state - tf_gene ({nm}): {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] {out[nm]['verdict']}")

    report("\n  NOTE: 1M-parameter model on 4 CPU cores. A null here bounds what is reachable at THIS scale;")
    report("  it does not establish that a large model would also fail.")

    json.dump({"test": "sc_transformer", "d_model": D_MODEL, "layers": N_LAYER, "epochs": EPOCHS,
               "n_train_cells": int(len(tr_idx)), "n_groups": len(groups),
               "results": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_vec")}
                           for k, v in results.items()},
               "contrast": out, "log": log}, open(OUT / "sc_transformer.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'sc_transformer.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
