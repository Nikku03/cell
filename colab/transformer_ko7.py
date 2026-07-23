"""transformer_ko7 -- v7: the INFERENCE ALGORITHM ("how it comes to a conclusion"). Every prior model concludes by RETRIEVING the top-K
behavioural neighbours and TRANSFERRING their MEAN (or a cosine-softmax-weighted mean) |z| profile. That averaging is crude: different
neighbours are informative about different genes, and a flat mean lets a few off-target neighbours wash out the specific movers.

v7 replaces the mean with a LEARNED NEIGHBOUR-AGGREGATOR (retrieval-augmented): freeze the transformer embedding, retrieve each query's
top-K train neighbours, and learn a small network that scores each (query, neighbour) pair to produce attention weights over the neighbours,
trained to RECONSTRUCT the query's true tide-removed profile (leave-one-out within train). At test, retrieve + aggregate with the learned
weights. This is the differentiable, supervised version of 'compose the neighbours' vs 'average the neighbours'.

ABLATION (same embedding, held-out K562, 3 splits): uniform-mean top-10 | cosine-softmax-weighted (val-tuned, the v3/v4 readout) | LEARNED
aggregator. Isolates the aggregation algorithm. Deterministic; GPU-ready; SMOKE mode (V7_SMOKE=1).

HONEST PRIOR: a learned aggregator CAN beat the mean if neighbour quality varies in a learnable way -- but the retrieval is still bounded by
the ~0.62 oracle (it cannot invent a neighbour that isn't there). Expect a small gain at best."""
import os, json
from pathlib import Path
import numpy as np
from eval_harness import Harness
from neural_ko import build_xy, mean_recall, spec_recall
from transformer_ko import build_context
OUT = Path("outputs/orphan")
SMOKE = bool(os.environ.get("V7_SMOKE"))


def run(SEED, H, have, Y, hidx, Fg, tok_idx, tok_role, tok_mask):
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); torch.set_num_threads(4)
    rng = np.random.RandomState(SEED)
    scor = [k for k in H.scorable if k in hidx]
    perm = rng.permutation(len(scor)); n_test = int(0.30 * len(scor))
    test = sorted(scor[i] for i in perm[:n_test]); test_set = set(test)
    train_all = [k for k in have if k not in test_set]
    tperm = rng.permutation(len(train_all)); n_val = int(0.12 * len(train_all))
    val = sorted(train_all[i] for i in tperm[:n_val] if train_all[i] in H.scorable); val_set = set(val)
    train = [k for k in train_all if k not in val_set]
    tr_arr = np.array([hidx[k] for k in train])

    Ft = torch.tensor(Fg).to(DEVICE); TI = torch.tensor(tok_idx); TR = torch.tensor(tok_role); TM = torch.tensor(tok_mask)
    Yspec = (Y * H.nontide.astype(np.float32)); Yn = Yspec / (np.linalg.norm(Yspec, axis=1, keepdims=True) + 1e-9)
    Ynt = torch.tensor(Yn).to(DEVICE)
    nontide = torch.tensor(H.nontide.astype(np.float32)).to(DEVICE)
    Yt = torch.tensor(Y).to(DEVICE)
    FEATIN = Fg.shape[1] + 4; DM = 128

    def gather(idx):
        f = Ft[TI[idx].to(DEVICE)]; roleoh = torch.nn.functional.one_hot(TR[idx].to(DEVICE), 4).float()
        return torch.cat([f, roleoh], -1), TM[idx].to(DEVICE)

    class Emb(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.proj = torch.nn.Linear(FEATIN, DM)
            s.enc = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(DM, 4, 256, 0.1, batch_first=True, activation="gelu"), 2)
            s.head = torch.nn.Sequential(torch.nn.LayerNorm(DM), torch.nn.Linear(DM, 64))
        def forward(s, tf, mask):
            h = s.enc(s.proj(tf), src_key_padding_mask=(mask < 0.5))
            pooled = (h * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-9)
            e = s.head(pooled); return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    def embed_all(model):
        model.eval()
        with torch.no_grad():
            E = np.zeros((len(have), 64), np.float32)
            for bs in range(0, len(have), 512):
                idx = torch.arange(bs, min(bs + 512, len(have)))
                tf, ms = gather(idx); E[bs:bs + len(idx)] = model(tf, ms).cpu().numpy()
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    # ---- 1. train the base embedding (metric-learning, same as transformer_ko) ----
    EP = 12 if SMOKE else 120; PAT = 4 if SMOKE else 12
    emb = Emb().to(DEVICE); opt = torch.optim.Adam(emb.parameters(), lr=1e-3, weight_decay=1e-4)
    def wmean(E, K, tau):
        def f(ko):
            i = hidx.get(ko)
            if i is None:
                return None
            s = E[tr_arr] @ E[i]; top = np.argsort(-s)[:K]; w = np.exp((s[top] - s[top].max()) / tau); w /= w.sum() + 1e-9
            return (w[:, None] * Y[tr_arr[top]]).sum(0)
        return f
    best, bstate, wait = -1.0, None, 0
    for ep in range(EP):
        emb.train(); pi = np.random.RandomState(SEED * 100 + ep).permutation(len(tr_arr))
        for bs in range(0, len(tr_arr), 256):
            idx = torch.tensor(tr_arr[pi[bs:bs + 256]])
            if len(idx) < 8:
                continue
            tf, ms = gather(idx); e = emb(tf, ms)
            Se = e @ e.T; St = Ynt[idx] @ Ynt[idx].T; dm = torch.eye(len(idx), device=DEVICE) * (-1e9)
            loss = -(torch.softmax(St / 0.1 + dm, 1) * torch.log_softmax(Se / 0.1 + dm, 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 3 == 0:
            vr, _ = mean_recall(H, val, wmean(embed_all(emb), 10, 0.1))
            if vr > best:
                best, bstate, wait = vr, {k: v.clone() for k, v in emb.state_dict().items()}, 0
            else:
                wait += 1
            if wait >= PAT:
                break
    if bstate:
        emb.load_state_dict(bstate)
    E = embed_all(emb); Et = torch.tensor(E).to(DEVICE)

    # ---- 2. LEARNED AGGREGATOR: attention over the top-K retrieved train neighbours, trained to reconstruct the query profile ----
    KN = 20
    Etr = Et[tr_arr]                                          # (nTr, 64)
    class Agg(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.net = torch.nn.Sequential(torch.nn.Linear(64 * 3 + 1, 128), torch.nn.GELU(), torch.nn.Dropout(0.1),
                                        torch.nn.Linear(128, 1))
        def forward(s, eq, En, cos):                         # eq (B,64), En (B,K,64), cos (B,K)
            B, K, _ = En.shape
            eqx = eq.unsqueeze(1).expand(B, K, 64)
            feat = torch.cat([eqx, En, eqx * En, cos.unsqueeze(-1)], -1)
            return s.net(feat).squeeze(-1)                   # (B,K) logits
    agg = Agg().to(DEVICE); optA = torch.optim.Adam(agg.parameters(), lr=2e-3, weight_decay=1e-4)
    # precompute, for each TRAIN query, its top-KN OTHER-train neighbours (leave-one-out)
    S = E[tr_arr] @ E[tr_arr].T; np.fill_diagonal(S, -2.0)
    nbr = np.argsort(-S, axis=1)[:, :KN]                      # (nTr, KN) indices into tr_arr
    nbr_cos = np.take_along_axis(S, nbr, axis=1)
    nbrT = torch.tensor(nbr).to(DEVICE); cosT = torch.tensor(nbr_cos).to(DEVICE)
    Ytr = Yt[torch.tensor(tr_arr).to(DEVICE)]                # (nTr, G) true K562 |z|
    nTr = len(tr_arr)
    EPA = 15 if SMOKE else 150
    for ep in range(EPA):
        agg.train(); pi = torch.randperm(nTr, device=DEVICE)
        for bs in range(0, nTr, 256):
            sel = pi[bs:bs + 256]
            eq = Etr[sel]; nb = nbrT[sel]; cs = cosT[sel]
            En = Etr[nb]; Yn_ = Ytr[nb]                       # (B,K,64),(B,K,G)
            logits = agg(eq, En, cs); w = torch.softmax(logits, -1)
            pred = (w.unsqueeze(-1) * Yn_).sum(1)            # (B,G)
            pv = pred * nontide; tv = Ytr[sel] * nontide
            cos = (pv * tv).sum(-1) / (pv.norm(dim=-1) * tv.norm(dim=-1) + 1e-9)
            loss = (1 - cos).mean()
            optA.zero_grad(); loss.backward(); optA.step()

    def learned_pred(ko):
        i = hidx.get(ko)
        if i is None:
            return None
        s = E[tr_arr] @ E[i]; top = np.argsort(-s)[:KN]
        with torch.no_grad():
            eq = Et[i:i + 1]; En = Etr[torch.tensor(top).to(DEVICE)].unsqueeze(0)
            cs = torch.tensor(s[top]).to(DEVICE).unsqueeze(0)
            w = torch.softmax(agg(eq, En, cs), -1)[0].cpu().numpy()
        return (w[:, None] * Y[tr_arr[top]]).sum(0)

    # tune the cosine-softmax baseline on val
    bestcfg, bv = (10, 0.1), -1
    for K in (5, 10, 20):
        for tau in (0.05, 0.1, 0.3):
            vr, _ = mean_recall(H, val, wmean(E, K, tau))
            if vr > bv:
                bv, bestcfg = vr, (K, tau)

    def umean(ko):
        i = hidx.get(ko)
        if i is None:
            return None
        top = tr_arr[np.argsort(-(E[tr_arr] @ E[i]))[:10]]; return Y[top].mean(0)

    def oracle_f(ko): return H._references(ko, 50)["ORACLE*"]
    r = {}
    r["uniform-mean@10"], _ = mean_recall(H, test, umean, 50)
    r["cosine-softmax (v3/v4)"], _ = mean_recall(H, test, wmean(E, *bestcfg), 50)
    r["LEARNED aggregator (v7)"], _ = mean_recall(H, test, learned_pred, 50)
    r["oracle"], _ = mean_recall(H, test, oracle_f, 50)
    return r


def main():
    H = Harness("K562")
    have, X, Y = build_xy(H)
    hidx = {k: i for i, k in enumerate(have)}
    Fg, tok_idx, tok_role, tok_mask = build_context(H, have)
    seeds = [0] if SMOKE else [0, 1, 2]
    allr = []
    for s in seeds:
        print(f"\n{'=' * 56}\n[SPLIT SEED {s}]{'  (SMOKE)' if SMOKE else ''}")
        r = run(s, H, have, Y, hidx, Fg, tok_idx, tok_role, tok_mask)
        for k, v in r.items():
            print(f"    {k:26s} {v:.3f}")
        allr.append(r)
    keys = list(allr[0].keys())
    agg = {k: (float(np.mean([r[k] for r in allr])), float(np.std([r[k] for r in allr]))) for k in keys}
    print(f"\n{'=' * 56}\nSTABILITY across {len(seeds)} split(s):")
    for k in keys:
        print(f"    {k:26s} {agg[k][0]:.3f} +/- {agg[k][1]:.3f}")
    L = "LEARNED aggregator (v7)"; C = "cosine-softmax (v3/v4)"
    d = [r[L] - r[C] for r in allr]
    print(f"\n    LEARNED - cosine-softmax: {np.mean(d):+.3f} +/- {np.std(d):.3f}  ({[round(x,3) for x in d]})")
    helps = np.mean(d) > 0.01 and (SMOKE or all(x > 0 for x in d))
    verdict = (
        f"TRANSFORMER v7 (transformer_ko7.py): the INFERENCE ALGORITHM -- a LEARNED neighbour-aggregator (attention over the top-K retrieved "
        f"train neighbours, trained to reconstruct the query's tide-removed profile) vs the mean/cosine-softmax transfer. {len(seeds)} splits, "
        f"same frozen embedding. uniform-mean {agg['uniform-mean@10'][0]:.3f} | cosine-softmax {agg[C][0]:.3f} | LEARNED {agg[L][0]:.3f} | "
        f"oracle {agg['oracle'][0]:.3f}. LEARNED - cosine-softmax = {np.mean(d):+.3f}+/-{np.std(d):.3f} "
        + ("(positive every split) -- composing the neighbours with a learned attention beats averaging them: neighbour quality varies in a "
           "learnable way, and the aggregator downweights off-target neighbours. A real gain from the inference algorithm."
           if helps else
           "-- NO stable gain: the learned aggregator does NOT beat a tuned cosine-softmax mean. Once neighbours are retrieved, HOW you "
           "combine them barely matters -- the information is in WHICH neighbours (the embedding/retrieval), not in the weighting; and it is "
           "still bounded by the ~0.62 oracle. The conclusion-step was already near-optimal.") +
        " Deterministic; GPU-ready.")
    print(f"\nVERDICT: {verdict}")
    if not SMOKE:
        json.dump({"mean_sd": {k: [round(agg[k][0], 4), round(agg[k][1], 4)] for k in keys},
                   "learned_minus_cosinesoftmax": [round(x, 4) for x in d], "helps": bool(helps),
                   "verdict": verdict, "note": verdict}, open(OUT / "transformer_ko7.json", "w"), indent=1)
        print("\n  -> outputs/orphan/transformer_ko7.json")
    else:
        print("\n[SMOKE OK]")


if __name__ == "__main__":
    main()
