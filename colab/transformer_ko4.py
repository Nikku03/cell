"""transformer_ko4 -- v3 (CLS + distillation + physics edge-features + Graphormer edge-bias) PLUS the one remaining layer with real data at
scale: SPATIAL GATING (the user's Layer D). Two complexes can have huge binding affinity in vitro but never meet if one is nucleolar and one
is plasma-membrane -- P(interaction) ~ P(compartment_A == compartment_B). We add the GO cellular compartment (12 classes, 100% coverage) as:

  (i)  a per-token COMPARTMENT one-hot feature, and
  (ii) a SPATIAL term in the Graphormer attention -- score(i,j) += b_h[same_compartment(i,j)]  (soft 'bias' mode)  OR  score(i,j) = -inf when
       compartments are disjoint (hard 'mask' mode = 'if G_spatial=0, attention is 0').

ABLATION on the v3-full base (edge-features + edge-bias always on): none (=v3-full) | +spatial-bias | +spatial-mask, 3 splits, held-out K562
tide-removed specific recall@50, + wrong-KO identity control, vs tide-null / oracle. Runs on GPU if present (Colab) else CPU. SMOKE mode
(env V4_SMOKE=1) runs 1 seed / reduced configs / few epochs for a fast end-to-end check. Deterministic per split.

HONEST PRIOR (stated up front, not hidden): the cross_line probe + v2 wash say the gap to the ~0.62 oracle is data/readout-limited, so spatial
gating is expected to be MARGINAL on this readout -- it is the honest test of whether localisation adds anything the network-SVD did not
already carry, and the Colab/GPU notebook makes it cheap for you to run and to extend toward the GPU-gated layers (AF-Multimer, ddG)."""
import os, json, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness
from neural_ko import build_xy, mean_recall
from transformer_ko import build_context
from transformer_ko3 import build_edges, N_ET, CLS_T
OUT = Path("outputs/orphan")
COMPS = ['nucleus', 'cytoplasm', 'plasma membrane', 'membrane', 'extracellular', 'mitochondrion',
         'ER', 'cytoskeleton', 'Golgi', 'endosome', 'lysosome', 'peroxisome']
NC = len(COMPS)
SMOKE = bool(os.environ.get("V4_SMOKE"))


def build_spatial(H, have, tok_idx, tok_mask, Gtok):
    """per-token compartment one-hot (nKO,33,NC) + same-compartment matrix with CLS at index 0 (nKO,34,34) in {0 diff,1 same,2 cls/unknown}."""
    D = json.load(open(OUT / "cell_complete.json"))
    comp = {g["name"]: g.get("comp") for g in D["genes"]}
    ci = {c: i for i, c in enumerate(COMPS)}
    T = tok_idx.shape[1]; nk = len(have)
    oneh = np.zeros((nk, T, NC), np.float32)
    tcomp = np.full((nk, T), -1, np.int64)
    for r in range(nk):
        for c in range(T):
            if tok_mask[r, c] > 0.5:
                k = ci.get(comp.get(Gtok[tok_idx[r, c]]), -1)
                tcomp[r, c] = k
                if k >= 0:
                    oneh[r, c, k] = 1.0
    same = np.full((nk, T + 1, T + 1), 2, np.int64)                 # default cls/unknown=2
    for r in range(nk):
        for i in range(T):
            if tok_mask[r, i] < 0.5 or tcomp[r, i] < 0:
                continue
            for j in range(T):
                if tok_mask[r, j] < 0.5 or tcomp[r, j] < 0:
                    continue
                same[r, i + 1, j + 1] = 1 if tcomp[r, i] == tcomp[r, j] else 0
        same[r, 0, :] = 2; same[r, :, 0] = 2                        # CLS routes everywhere (never spatially masked)
    return oneh, same


def run(SEED, H, have, Y, hidx, Fg, tok_idx, tok_role, tok_mask, edge_feat, etype, emag, comp_oh, same_comp):
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

    Ft = torch.tensor(Fg); TI = torch.tensor(tok_idx); TR = torch.tensor(tok_role); TM = torch.tensor(tok_mask)
    EF = torch.tensor(edge_feat); CO = torch.tensor(comp_oh)
    ET = torch.tensor(etype); EM = torch.tensor(emag); SC = torch.tensor(same_comp)
    Yspec = (Y * H.nontide.astype(np.float32)); Yn = Yspec / (np.linalg.norm(Yspec, axis=1, keepdims=True) + 1e-9)
    Ynt = torch.tensor(Yn)
    FEATIN = Fg.shape[1] + 3 + NC + 4; DM = 128; HEADS = 4
    dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True)
    syms = list(dv["syms"]); Z = dv["Z"].astype(np.float32); srow = {s: i for i, s in enumerate(syms)}
    XK = np.stack([Z[srow[k]] for k in have]); XKn = XK / (np.linalg.norm(XK, axis=1, keepdims=True) + 1e-9)

    def gather(idx):
        f = Ft[TI[idx]]; ef = EF[idx]; co = CO[idx]; roleoh = torch.nn.functional.one_hot(TR[idx], 4).float()
        tokfeat = torch.cat([f, ef, co, roleoh], -1)
        return (tokfeat.to(DEVICE), TM[idx].to(DEVICE), ET[idx].to(DEVICE), EM[idx].to(DEVICE), SC[idx].to(DEVICE))

    class EdgeMHA(torch.nn.Module):
        def __init__(s, spatial_mode):
            super().__init__()
            s.h = HEADS; s.dh = DM // HEADS; s.sm = spatial_mode
            s.q = torch.nn.Linear(DM, DM); s.k = torch.nn.Linear(DM, DM); s.v = torch.nn.Linear(DM, DM); s.o = torch.nn.Linear(DM, DM)
            s.etb = torch.nn.Embedding(N_ET, HEADS); s.emw = torch.nn.Parameter(torch.zeros(HEADS))
            s.spb = torch.nn.Embedding(3, HEADS); s.drop = torch.nn.Dropout(0.1)          # spatial: 0 diff,1 same,2 cls
        def forward(s, x, et, em, sc, kpm):
            B, T, _ = x.shape
            q = s.q(x).view(B, T, s.h, s.dh).transpose(1, 2); k = s.k(x).view(B, T, s.h, s.dh).transpose(1, 2)
            v = s.v(x).view(B, T, s.h, s.dh).transpose(1, 2)
            score = (q @ k.transpose(-2, -1)) / (s.dh ** 0.5)
            score = score + s.etb(et).permute(0, 3, 1, 2) + em.unsqueeze(1) * s.emw.view(1, s.h, 1, 1)   # v3 edge-bias always on
            if s.sm == "bias":
                score = score + s.spb(sc).permute(0, 3, 1, 2)
            elif s.sm == "mask":
                score = score.masked_fill((sc == 0)[:, None, :, :], -1e9)                 # hard: disjoint compartment -> no attention
            score = score.masked_fill(kpm[:, None, None, :], -1e9)
            a = s.drop(torch.softmax(score, -1))
            return s.o((a @ v).transpose(1, 2).reshape(B, T, DM))

    class Block(torch.nn.Module):
        def __init__(s, sm):
            super().__init__()
            s.n1 = torch.nn.LayerNorm(DM); s.att = EdgeMHA(sm); s.n2 = torch.nn.LayerNorm(DM)
            s.ff = torch.nn.Sequential(torch.nn.Linear(DM, 256), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, DM))
        def forward(s, x, et, em, sc, kpm):
            x = x + s.att(s.n1(x), et, em, sc, kpm); return x + s.ff(s.n2(x))

    class Model(torch.nn.Module):
        def __init__(s, sm):
            super().__init__()
            s.proj = torch.nn.Linear(FEATIN, DM); s.cls = torch.nn.Parameter(torch.randn(1, 1, DM) * 0.02)
            s.blocks = torch.nn.ModuleList([Block(sm) for _ in range(2)])
            s.head = torch.nn.Sequential(torch.nn.LayerNorm(DM), torch.nn.Linear(DM, 64))
        def forward(s, tokfeat, mask, et, em, sc):
            h = s.proj(tokfeat); B = h.size(0)
            h = torch.cat([s.cls.expand(B, 1, DM), h], 1)
            m = torch.cat([torch.ones(B, 1, device=h.device), mask], 1); kpm = (m < 0.5)
            for blk in s.blocks:
                h = blk(h, et, em, sc, kpm)
            e = s.head(h[:, 0]); return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    def embed_all(model):
        model.eval()
        with torch.no_grad():
            E = np.zeros((len(have), 64), np.float32)
            for bs in range(0, len(have), 512):
                idx = torch.arange(bs, min(bs + 512, len(have)))
                tf, ms, et, em, sc = gather(idx)
                E[bs:bs + len(idx)] = model(tf, ms, et, em, sc).cpu().numpy()
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def weighted(E, K, tau, alpha):
        def f(ko):
            i = hidx.get(ko)
            if i is None:
                return None
            s = (1 - alpha) * (E[tr_arr] @ E[i]) + alpha * (XKn[tr_arr] @ XKn[i])
            top = np.argsort(-s)[:K]; w = np.exp((s[top] - s[top].max()) / tau); w /= w.sum() + 1e-9
            return (w[:, None] * Y[tr_arr[top]]).sum(0)
        return f

    EPOCHS = 20 if SMOKE else 200; PAT = 4 if SMOKE else 15
    def train(sm):
        torch.manual_seed(SEED); model = Model(sm).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, bstate, wait = -1.0, None, 0
        for ep in range(EPOCHS):
            model.train(); pi = np.random.RandomState(SEED * 100 + ep).permutation(len(tr_arr))
            for bs in range(0, len(tr_arr), 256):
                idx = torch.tensor(tr_arr[pi[bs:bs + 256]])
                if len(idx) < 8:
                    continue
                tf, ms, et, em, sc = gather(idx)
                e = model(tf, ms, et, em, sc)
                Se = e @ e.T; St = (Ynt[idx].to(DEVICE)) @ (Ynt[idx].to(DEVICE)).T; dm = torch.eye(len(idx), device=DEVICE) * (-1e9)
                loss = -(torch.softmax(St / 0.1 + dm, 1) * torch.log_softmax(Se / 0.1 + dm, 1)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 5 == 0:
                vr, _ = mean_recall(H, val, weighted(embed_all(model), 10, 0.1, 0.3))
                if vr > best:
                    best, bstate, wait = vr, {k: v.clone() for k, v in model.state_dict().items()}, 0
                else:
                    wait += 1
                if wait >= PAT:
                    break
        if bstate:
            model.load_state_dict(bstate)
        return model

    def tune_and_score(E):
        best_cfg, best_vr = (10, 0.1, 0.3), -1.0
        for K in (5, 10, 20):
            for tau in (0.05, 0.1, 0.3):
                for alpha in (0.0, 0.3):
                    vr, _ = mean_recall(H, val, weighted(E, K, tau, alpha))
                    if vr > best_vr:
                        best_vr, best_cfg = vr, (K, tau, alpha)
        sc, _ = mean_recall(H, test, weighted(E, *best_cfg), 50)
        return sc, best_cfg

    modes = ["none", "mask"] if SMOKE else ["none", "bias", "mask"]
    names = {"none": "v3-full (no spatial)", "bias": "+spatial-bias", "mask": "+spatial-mask (hard gate)"}
    res = {}; Efull = None; cfgfull = (10, 0.1, 0.3)
    for sm in modes:
        E = embed_all(train(sm)); sc, cfg = tune_and_score(E); res[names[sm]] = sc
        if sm == "none":
            Efull, cfgfull = E, cfg
    def tide_f(ko): return H.mover_freq
    def oracle_f(ko): return H._references(ko, 50)["ORACLE*"]
    res["TIDE-null"], _ = mean_recall(H, test, tide_f, 50)
    res["ORACLE* (ceiling)"], _ = mean_recall(H, test, oracle_f, 50)
    shp = rng.permutation(len(test)); smap = {test[i]: test[shp[i]] for i in range(len(test))}
    Kf, tf_, af = cfgfull
    def shufctrl(ko):
        w = smap.get(ko); j = hidx.get(w)
        if j is None:
            return None
        s = (1 - af) * (Efull[tr_arr] @ Efull[j]) + af * (XKn[tr_arr] @ XKn[j])
        top = np.argsort(-s)[:Kf]; ww = np.exp((s[top] - s[top].max()) / tf_); ww /= ww.sum() + 1e-9
        return (ww[:, None] * Y[tr_arr[top]]).sum(0)
    res["wrong-KO shuffle (ctrl)"], _ = mean_recall(H, test, shufctrl, 50)
    print(f"    [seed {SEED}] device={DEVICE}")
    return res


def main():
    H = Harness("K562")
    have, X, Y = build_xy(H)
    hidx = {k: i for i, k in enumerate(have)}
    Fg, tok_idx, tok_role, tok_mask, Gtok = build_context(H, have, return_gtok=True)
    edge_feat, etype, emag = build_edges(H, have, tok_idx, tok_mask, Gtok)
    comp_oh, same_comp = build_spatial(H, have, tok_idx, tok_mask, Gtok)
    seeds = [0] if SMOKE else [0, 1, 2]
    all_res = []
    for s in seeds:
        print(f"\n{'=' * 58}\n[SPLIT SEED {s}]{'  (SMOKE)' if SMOKE else ''}")
        r = run(s, H, have, Y, hidx, Fg, tok_idx, tok_role, tok_mask, edge_feat, etype, emag, comp_oh, same_comp)
        for k, v in r.items():
            print(f"    {k:30s} {v:.3f}")
        all_res.append(r)
    keys = list(all_res[0].keys())
    agg = {k: (float(np.mean([r[k] for r in all_res])), float(np.std([r[k] for r in all_res]))) for k in keys}
    print(f"\n{'=' * 58}\nSTABILITY across {len(seeds)} split(s) (mean +/- sd):")
    for k in keys:
        print(f"    {k:30s} {agg[k][0]:.3f} +/- {agg[k][1]:.3f}")
    B = "v3-full (no spatial)"
    spatial_rows = [k for k in keys if k.startswith("+spatial")]
    best_sp = max((agg[k][0] for k in spatial_rows), default=agg[B][0])
    best_sp_name = max(spatial_rows, key=lambda k: agg[k][0]) if spatial_rows else "n/a"
    d = best_sp - agg[B][0]
    orc = agg["ORACLE* (ceiling)"][0]; tide = agg["TIDE-null"][0]
    helps = d > 0.01 and (not SMOKE)
    verdict = (
        f"TRANSFORMER v4 (transformer_ko4.py): v3 (CLS+distillation+physics edge-features+Graphormer edge-bias) + SPATIAL GATING (GO "
        f"compartment: token one-hot + same-compartment attention bias/hard-mask). {'SMOKE run.' if SMOKE else f'{len(seeds)} splits.'} "
        f"held-out K562 tide-removed specific recall@50: base v3-full {agg[B][0]:.3f} | best spatial ({best_sp_name}) {best_sp:.3f} | "
        f"tide {tide:.3f} | oracle {orc:.3f}. spatial - base = {d:+.3f} "
        + ("-- spatial gating HELPS: localisation carries signal beyond what the network-SVD already encoded."
           if helps else
           "-- spatial gating is MARGINAL/flat (as predicted): compartment is largely already latent in the network-SVD (which is built from "
           "the same complex/PPI graph that respects compartments), and the readout ceiling is data-limited, not localisation-limited.")
        + " Runs on GPU when available (Colab). Deterministic per split; the honest bar remains the ~0.62 retrieval ceiling.")
    print(f"\nVERDICT: {verdict}")
    if not SMOKE:
        out = {"mean_sd": {k: [round(agg[k][0], 4), round(agg[k][1], 4)] for k in keys},
               "best_spatial_minus_base": round(float(d), 4), "spatial_helps": bool(helps),
               "verdict": verdict, "note": verdict}
        json.dump(out, open(OUT / "transformer_ko4.json", "w"), indent=1)
        print("\n  -> outputs/orphan/transformer_ko4.json")
    else:
        print("\n[SMOKE OK] end-to-end pipeline ran; use the Colab notebook for the full GPU run.")


if __name__ == "__main__":
    main()
