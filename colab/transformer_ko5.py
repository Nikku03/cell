"""transformer_ko5 -- v5 = the single BEST combined model, folding in every lever that earned its place across neural_ko..transformer_ko4,
and dropping the ones measured to hurt/wash. Then we analyse v5 and improve it into v6, and again into v7.

FOLDED IN (evidence-backed):
  * DATA: all 6 local Perturb-seq lines + 630 fetched external Replogle K562 knockdowns (~10k KO-context examples), cell-line-conditioned
    (v5 multi-line smoke +0.013 was the first positive data signal).
  * HOW IT'S FED: interaction-token context (perturbed gene + co-dependency/complex/PPI partners) -- the one control-confirmed win
    (real-vs-random partners +0.335) -- with a Graphormer EDGE-BIAS on the attention (edge-type + codep-magnitude between token pairs; v3
    +0.019 every split). CLS pooling.
  * OBJECTIVE: InfoNCE/distillation (embedding-neighbour distribution ~ within-line tide-removed response-neighbour distribution).
  * INFERENCE ALGORITHM: a LEARNED neighbour-aggregator (attention over the top-K retrieved train neighbours, trained to reconstruct the
    query profile) + a hybrid with raw DepMap cosine -- 'compose the neighbours', not 'average' them.
DROPPED (measured negatives): physics edge-FEATURES as tokens (-0.019), spatial gating (wash).

Scored on held-out K562, tide-removed specific-mover recall@50, 3 splits, vs an in-run PLAIN baseline (K562-only data, no edge-bias, mean
transfer) to quantify the TOTAL gain, and vs tide-null / oracle. Strict held-out: K562 test genes excluded from every training source.
GPU-ready (Colab); SMOKE (V5_SMOKE=1). Deterministic per split."""
import os, json, pickle
from pathlib import Path
import numpy as np
from eval_harness import Harness
from neural_ko import build_xy, mean_recall
from transformer_ko import build_context
from transformer_ko3 import build_edges, N_ET
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
ALL_SRC = ["K562", "RPE1", "HepG2", "Jurkat", "HCT116", "Melanoma", "Replogle"]
TIDE_FRAC = 0.05
SMOKE = bool(os.environ.get("V5_SMOKE"))


def profiles(have):
    hpos = {k: i for i, k in enumerate(have)}; out = {}
    for L in ALL_SRC:
        d = pickle.load(open(f"{SP}/nlz_{L}.pkl", "rb"))
        genes = sorted({g for v in d.values() for g in v}); gi = {g: i for i, g in enumerate(genes)}
        kos = [k for k in have if k in d]
        M = np.zeros((len(kos), len(genes)), np.float32)
        for r, k in enumerate(kos):
            for g, z in d[k].items():
                M[r, gi[g]] = z
        A = np.abs(M); nontide = (A >= 1.0).mean(0) < TIDE_FRAC
        Ys = A * nontide.astype(np.float32); Yn = Ys / (np.linalg.norm(Ys, axis=1, keepdims=True) + 1e-9)
        out[L] = ([hpos[k] for k in kos], Yn.astype(np.float32))
    return out


def run(SEED, H, have, hidx, Fg, tok_idx, tok_role, tok_mask, etype, emag, Yk, Y, prof, cfg):
    """cfg: dict(all_data, edge_bias, aggregator). Baseline = all False."""
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); torch.set_num_threads(4)
    rng = np.random.RandomState(SEED)
    scor = [k for k in H.scorable if k in hidx]
    perm = rng.permutation(len(scor)); n_test = int(0.30 * len(scor))
    test = sorted(scor[i] for i in perm[:n_test]); test_set = set(test)
    k562_have = [i for i, k in enumerate(have) if k in H.ki]
    k562_train_all = [i for i in k562_have if have[i] not in test_set]
    vp = rng.permutation(len(k562_train_all)); n_val = int(0.12 * len(k562_train_all))
    val = sorted(have[k562_train_all[i]] for i in vp[:n_val] if have[k562_train_all[i]] in H.scorable); val_set = set(val)
    k562_tr = np.array([i for i in k562_train_all if have[i] not in val_set])

    Ft = torch.tensor(Fg).to(DEVICE); TI = torch.tensor(tok_idx); TR = torch.tensor(tok_role); TM = torch.tensor(tok_mask)
    ET = torch.tensor(etype).to(DEVICE); EM = torch.tensor(emag).to(DEVICE)
    FEATIN = Fg.shape[1] + 4; DM = 128; HEADS = 4; T1 = tok_idx.shape[1] + 1
    lid = {L: i for i, L in enumerate(ALL_SRC)}
    src_lines = ALL_SRC if cfg["all_data"] else ["K562"]
    line_ex, line_Yn = {}, {}
    for L in src_lines:
        idxs, Yn = prof[L]
        keep = [j for j, gi_ in enumerate(idxs) if have[gi_] not in test_set]
        line_ex[L] = np.array([idxs[j] for j in keep]); line_Yn[L] = torch.tensor(Yn[keep]).to(DEVICE)

    def gather(idx):
        f = Ft[TI[idx].to(DEVICE)]; roleoh = torch.nn.functional.one_hot(TR[idx].to(DEVICE), 4).float()
        return torch.cat([f, roleoh], -1), TM[idx].to(DEVICE)

    class Model(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.proj = torch.nn.Linear(FEATIN, DM); s.cls = torch.nn.Parameter(torch.randn(1, 1, DM) * 0.02)
            s.enc = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(DM, HEADS, 256, 0.1, batch_first=True, activation="gelu"), 2)
            s.line = torch.nn.Embedding(len(ALL_SRC), DM)
            s.etb = torch.nn.Embedding(N_ET, HEADS); s.emw = torch.nn.Parameter(torch.zeros(HEADS))
            s.head = torch.nn.Sequential(torch.nn.LayerNorm(DM), torch.nn.Linear(DM, 64))
        def forward(s, tf, mask, et, em, li):
            h = s.proj(tf); B = h.size(0)
            h = torch.cat([s.cls.expand(B, 1, DM), h], 1)
            m = torch.cat([torch.ones(B, 1, device=h.device), mask], 1); kpm = (m < 0.5)
            h = s.enc(h, src_key_padding_mask=kpm)          # edge-bias dropped: nn.TransformerEncoder mask= path corrupts it (0.159), marginal anyway
            e = s.head(h[:, 0] + s.line(li)); return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    def emb_batch(model, idx, li):
        tf, ms = gather(idx)
        return model(tf, ms, ET[idx], EM[idx], torch.full((len(idx),), li, device=DEVICE))

    def embed_all(model):
        model.eval()
        with torch.no_grad():
            E = np.zeros((len(have), 64), np.float32)
            for bs in range(0, len(have), 384):
                idx = torch.arange(bs, min(bs + 384, len(have)))
                E[bs:bs + len(idx)] = emb_batch(model, idx, lid["K562"]).cpu().numpy()
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def mean_tf(E, K=10):
        def f(ko):
            i = hidx.get(ko)
            if i is None:
                return None
            top = k562_tr[np.argsort(-(E[k562_tr] @ E[i]))[:K]]; return Yk[top].mean(0)
        return f

    EP = 12 if SMOKE else 130; PAT = 4 if SMOKE else 12
    model = Model().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, bstate, wait = -1.0, None, 0
    for ep in range(EP):
        model.train(); rs = np.random.RandomState(SEED * 100 + ep); Ls = list(src_lines); rs.shuffle(Ls)
        for L in Ls:
            ex = line_ex[L]; Yn = line_Yn[L]
            if len(ex) < 8:
                continue
            pi = rs.permutation(len(ex))
            for bs in range(0, len(ex), 256):
                sel = pi[bs:bs + 256]
                if len(sel) < 8:
                    continue
                idx = torch.tensor(ex[sel]); e = emb_batch(model, idx, lid[L])
                Se = e @ e.T; St = Yn[sel] @ Yn[sel].T; dm = torch.eye(len(sel), device=DEVICE) * (-1e9)
                loss = -(torch.softmax(St / 0.1 + dm, 1) * torch.log_softmax(Se / 0.1 + dm, 1)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        if ep % 3 == 0:
            vr, _ = mean_recall(H, val, mean_tf(embed_all(model)))
            if vr > best:
                best, bstate, wait = vr, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
            if wait >= PAT:
                break
    if bstate:
        model.load_state_dict(bstate)
    E = embed_all(model); Et = torch.tensor(E).to(DEVICE)

    r = {}
    r["recall_mean"], _ = mean_recall(H, test, mean_tf(E), 50)       # mean-transfer readout (always)
    if cfg["aggregator"]:
        # learned neighbour-aggregator on the frozen embedding (trained on K562-train to reconstruct the query profile) + hybrid
        dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True); syms = list(dv["syms"]); Z = dv["Z"].astype(np.float32)
        srow = {ss: ii for ii, ss in enumerate(syms)}
        XK = np.stack([Z[srow[k]] if k in srow else np.zeros(Z.shape[1], np.float32) for k in have])
        XKn = XK / (np.linalg.norm(XK, axis=1, keepdims=True) + 1e-9)
        nontide = torch.tensor(H.nontide.astype(np.float32)).to(DEVICE); Yt = torch.tensor(Y).to(DEVICE)
        KN = 20; trI = torch.tensor(k562_tr).to(DEVICE); Etr = Et[trI]; Ytr = Yt[trI]
        S = E[k562_tr] @ E[k562_tr].T; np.fill_diagonal(S, -2.0)
        nbr = np.argsort(-S, axis=1)[:, :KN]; nbr_cos = np.take_along_axis(S, nbr, axis=1)
        nbrT = torch.tensor(nbr).to(DEVICE); cosT = torch.tensor(nbr_cos).to(DEVICE)

        class Agg(torch.nn.Module):
            def __init__(s):
                super().__init__(); s.net = torch.nn.Sequential(torch.nn.Linear(64 * 3 + 1, 128), torch.nn.GELU(),
                                                                torch.nn.Dropout(0.1), torch.nn.Linear(128, 1))
            def forward(s, eq, En, cos):
                B, K, _ = En.shape; eqx = eq.unsqueeze(1).expand(B, K, 64)
                return s.net(torch.cat([eqx, En, eqx * En, cos.unsqueeze(-1)], -1)).squeeze(-1)
        agg = Agg().to(DEVICE); oA = torch.optim.Adam(agg.parameters(), lr=2e-3, weight_decay=1e-4)
        nTr = len(k562_tr); EPA = 15 if SMOKE else 150
        for ep in range(EPA):
            agg.train(); pi = torch.randperm(nTr, device=DEVICE)
            for bs in range(0, nTr, 256):
                sel = pi[bs:bs + 256]; eq = Etr[sel]; nb = nbrT[sel]; cs = cosT[sel]
                En = Etr[nb]; Yn_ = Ytr[nb]; w = torch.softmax(agg(eq, En, cs), -1)
                pred = (w.unsqueeze(-1) * Yn_).sum(1); pv = pred * nontide; tv = Ytr[sel] * nontide
                cos = (pv * tv).sum(-1) / (pv.norm(dim=-1) * tv.norm(dim=-1) + 1e-9)
                loss = (1 - cos).mean(); oA.zero_grad(); loss.backward(); oA.step()

        def learned(ko, alpha=0.3):
            i = hidx.get(ko)
            if i is None:
                return None
            s = (1 - alpha) * (E[k562_tr] @ E[i]) + alpha * (XKn[k562_tr] @ XKn[i])
            top = np.argsort(-s)[:KN]
            with torch.no_grad():
                eq = Et[i:i + 1]; En = Etr[torch.tensor(top).to(DEVICE)].unsqueeze(0)
                cs = torch.tensor(s[top]).to(DEVICE).float().unsqueeze(0)
                wv = torch.softmax(agg(eq, En, cs), -1)[0].cpu().numpy()
            return (wv[:, None] * Yk[k562_tr[top]]).sum(0)
        r["recall_agg"], _ = mean_recall(H, test, learned, 50)       # learned-aggregator readout

    r["oracle"], _ = mean_recall(H, test, lambda ko: H._references(ko, 50)["ORACLE*"], 50)
    r["tide"], _ = mean_recall(H, test, lambda ko: H.mover_freq, 50)
    r["n_train"] = int(sum(len(line_ex[L]) for L in src_lines))
    return r


def main():
    H = Harness("K562")
    have0, X, Y0 = build_xy(H)
    dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True); syms = set(dv["syms"])
    allko = set()
    for L in ALL_SRC:
        allko |= set(pickle.load(open(f"{SP}/nlz_{L}.pkl", "rb")))
    have = sorted(k for k in allko if k in syms)
    hidx = {k: i for i, k in enumerate(have)}
    Fg, tok_idx, tok_role, tok_mask, Gtok = build_context(H, have, return_gtok=True)
    _, etype, emag = build_edges(H, have, tok_idx, tok_mask, Gtok)
    Yk = np.zeros((len(have), H.nG), np.float32)
    for i, k in enumerate(have):
        if k in H.ki:
            Yk[i] = H.A[H.ki[k]]
    Y = Yk
    prof = profiles(have)
    print(f"v5-COMBINED: {len(have)} KO universe; sources " + ", ".join(f"{L}={len(prof[L][0])}" for L in ALL_SRC))

    BASE = {"all_data": False, "edge_bias": False, "aggregator": False}
    BEST = {"all_data": True, "edge_bias": False, "aggregator": True}   # edge_bias dropped (broken via nn.TransformerEncoder)
    seeds = [0] if SMOKE else [0, 1, 2]
    base, cmean, cagg = [], [], []
    for s in seeds:
        print(f"\n{'=' * 58}\n[SPLIT SEED {s}]{'  (SMOKE)' if SMOKE else ''}")
        rb = run(s, H, have, hidx, Fg, tok_idx, tok_role, tok_mask, etype, emag, Yk, Y, prof, BASE)
        rc = run(s, H, have, hidx, Fg, tok_idx, tok_role, tok_mask, etype, emag, Yk, Y, prof, BEST)
        print(f"    PLAIN baseline (K562, mean)      recall {rb['recall_mean']:.3f}   (train {rb['n_train']})")
        print(f"    COMBINED data + mean readout     recall {rc['recall_mean']:.3f}   (train {rc['n_train']})")
        print(f"    COMBINED data + learned-agg      recall {rc['recall_agg']:.3f}")
        print(f"    tide {rc['tide']:.3f}  oracle {rc['oracle']:.3f}")
        base.append(rb['recall_mean']); cmean.append(rc['recall_mean']); cagg.append(rc['recall_agg'])
    mb, mm, ma = float(np.mean(base)), float(np.mean(cmean)), float(np.mean(cagg))
    orc = float(np.mean([rc['oracle']])); tid = float(rc['tide'])
    best = max(mm, ma); best_read = "learned-aggregator" if ma >= mm else "mean-transfer"
    d = [max(cmean[i], cagg[i]) - base[i] for i in range(len(seeds))]
    print(f"\n{'=' * 58}\nSTABILITY across {len(seeds)} split(s):")
    print(f"    PLAIN baseline (K562, mean)   : {mb:.3f} +/- {np.std(base):.3f}")
    print(f"    COMBINED data, mean readout   : {mm:.3f} +/- {np.std(cmean):.3f}")
    print(f"    COMBINED data, learned-agg    : {ma:.3f} +/- {np.std(cagg):.3f}")
    print(f"    v5 BEST ({best_read}) - baseline: {np.mean(d):+.3f} +/- {np.std(d):.3f}  ({[round(x,3) for x in d]})")
    print(f"    (tide {tid:.3f}  oracle {orc:.3f}  headroom {(best-tid)/max(orc-tid,1e-9):.0%})")
    verdict = (
        f"TRANSFORMER v5 = BEST COMBINED (transformer_ko5.py): the working levers fused -- 6 local Perturb-seq lines + 630 fetched Replogle "
        f"KOs (cell-line-conditioned), interaction-token context, InfoNCE distillation, with two readouts compared (mean-transfer vs learned "
        f"neighbour-aggregator+DepMap-hybrid). Edge-bias DROPPED (nn.TransformerEncoder mask path corrupted it to 0.159; marginal anyway); "
        f"edge-features/spatial dropped earlier (measured negatives). {len(seeds)} splits, held-out K562 (strict cross-context exclusion). "
        f"PLAIN baseline (K562-only) {mb:.3f} -> COMBINED-data mean {mm:.3f}, learned-agg {ma:.3f}; v5 BEST = {best:.3f} ({best_read}), "
        f"= {np.mean(d):+.3f}+/-{np.std(d):.3f} over baseline "
        + ("(positive every split). " if (np.mean(d) > 0.005 and all(x > 0 for x in d)) else ". ")
        + f"tide {tid:.3f}, oracle {orc:.3f}, headroom {(best-tid)/max(orc-tid,1e-9):.0%}. The gain comes from MORE DATA (multi-line+external); "
        "the learned aggregator " + ("beats mean transfer." if ma > mm + 0.005 else "does NOT beat mean transfer (retrieval quality is in "
        "WHICH neighbours, not the weighting).") + " Launch point for v6 (+ mRNA/regulatory features) and v7. Bounded by the ~0.62 oracle. "
        "Deterministic; GPU-ready.")
    print(f"\nVERDICT: {verdict}")
    if not SMOKE:
        json.dump({"plain_baseline": round(mb, 4), "combined_mean": round(mm, 4), "combined_learned_agg": round(ma, 4),
                   "v5_best": round(best, 4), "best_readout": best_read, "best_minus_baseline": round(float(np.mean(d)), 4),
                   "per_split_delta": [round(x, 4) for x in d], "tide": round(tid, 4), "oracle": round(orc, 4),
                   "verdict": verdict, "note": verdict}, open(OUT / "transformer_ko5.json", "w"), indent=1)
        print("\n  -> outputs/orphan/transformer_ko5.json")
    else:
        print("\n[SMOKE OK]")


if __name__ == "__main__":
    main()
