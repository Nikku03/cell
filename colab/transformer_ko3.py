"""transformer_ko3 -- v2 (CLS + InfoNCE-distillation + val-tuned weighted/hybrid transfer) PLUS the two user fixes, cleanly ablated:

  FIX #1  physics into the tokens: each partner token gains EDGE-MAGNITUDE features describing its bond to the knockout --
          [codep-cosine to KO | log interface n_pdb to KO | shared-complex count with KO]. These are the real structural/functional
          magnitudes at scale (n_pdb from 3did, codep from DepMap); the true equivariant ddG tensor field is the GPU wall (only ~55
          structure-backed pairs genome-wide), so we use the highest-fidelity proxies, not a fantasy field.

  FIX #2  Graphormer-style EDGE-CONDITIONED ATTENTION: score(i,j) = (Qi.Kj)/sqrt(d) + b_h[edge_type(i,j)] + w_h * edge_mag(i,j), where
          edge_type in {none, self, same-complex, codep, ppi, cls} over EVERY token pair (incl partner-partner), b_h a learned per-head
          scalar, w_h a learned per-head weight on the continuous codep-cosine magnitude. This hardcodes a topological route so a head can
          snap same-machine tokens together instead of hunting the SVD latent space.

(Fix #3 dual-head deliberately OMITTED: the objective already targets the TIDE-REMOVED response cosine, so the generic-stress gradient is
 already decoupled -- a second head would only add tuning to solve a problem the target definition already solves.)

ABLATION (same split, apples-to-apples, 3 seeds): base | +edge-features | +edge-bias | +full, PLUS a SHUFFLED-edge-type control (full
architecture, permuted edge-type/mag matrices) -- if the bias helps only with REAL topology, the shuffle collapses its gain. Controls:
real-vs-random partners (structure), wrong-KO shuffle (identity). Metric: held-out K562 tide-removed specific-mover recall@50. CPU,
deterministic per split. Honest expectation: edge-bias is the likelier winner; edge-features may be partly redundant with role+network-SVD;
lands ~0.50-0.55, NOT past the ~0.62 retrieval ceiling (that needs transient-kinetics data, unavailable here).
"""
import json, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness
from neural_ko import build_xy, mean_recall
from transformer_ko import build_context
OUT = Path("outputs/orphan")
N_ET = 6                         # none, self, complex, codep, ppi, cls
CLS_T = 5


def build_edges(H, have, tok_idx, tok_mask, Gtok):
    """Per-KO edge-magnitude token features (nKO,33,3) + token-pair edge-type/mag matrices with CLS at index 0 (nKO,34,34)."""
    dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True)
    syms = list(dv["syms"]); Z = dv["Z"].astype(np.float32); srow = {s: i for i, s in enumerate(syms)}
    nr = np.linalg.norm(Z, axis=1, keepdims=True); nr[nr < 1e-9] = 1.0; Zc = Z / nr
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; N = len(names); nidx = {n: i for i, n in enumerate(names)}
    ppi = collections.defaultdict(set)
    for a, b in D["ppi"]:
        if isinstance(a, int) and isinstance(b, int) and a < N and b < N:
            ppi[a].add(b); ppi[b].add(a)
    gcplx = collections.defaultdict(set)
    for ci, mem in enumerate(D["complexes"].values()):
        for x in mem:
            if isinstance(x, int) and x < N:
                gcplx[x].add(ci)
    codep = {int(k): {int(j) for j, _ in v} for k, v in D.get("codep", {}).items()}
    iface = json.load(open(OUT / "iface_gate.json")).get("interface_edges", {})
    def npdb(ai, bi):
        e = iface.get(f"{min(ai, bi)}_{max(ai, bi)}")
        return e["n_pdb"] if e else 0

    T = tok_idx.shape[1]                       # 33 (self + 32)
    nk = len(have)
    edge_feat = np.zeros((nk, T, 3), np.float32)
    etype = np.zeros((nk, T + 1, T + 1), np.int64)
    emag = np.zeros((nk, T + 1, T + 1), np.float32)
    for r in range(nk):
        toks = [Gtok[tok_idx[r, c]] for c in range(T)]
        valid = [c for c in range(T) if tok_mask[r, c] > 0.5]
        ci = [nidx.get(toks[c], -1) for c in range(T)]        # cell_complete idx per token
        zc = np.stack([Zc[srow[toks[c]]] if toks[c] in srow else np.zeros(Zc.shape[1], np.float32) for c in range(T)])
        cos = zc @ zc.T                                       # (T,T) codep-cosine between tokens
        ko = ci[0]
        # per-partner edge features to the KO (token 0)
        for c in valid:
            if c == 0:
                continue
            pc = ci[c]
            sc = len(gcplx.get(ko, set()) & gcplx.get(pc, set())) if ko >= 0 and pc >= 0 else 0
            edge_feat[r, c, 0] = cos[0, c]
            edge_feat[r, c, 1] = np.log1p(npdb(ko, pc)) if ko >= 0 and pc >= 0 else 0.0
            edge_feat[r, c, 2] = sc
        # token-pair edge type + magnitude (offset by +1 for CLS at index 0)
        for i in valid:
            ii = ci[i]
            for j in valid:
                jj = ci[j]
                if i == j:
                    t = 1
                elif ii >= 0 and jj >= 0 and (gcplx.get(ii, set()) & gcplx.get(jj, set())):
                    t = 2
                elif ii >= 0 and jj >= 0 and (jj in codep.get(ii, set()) or ii in codep.get(jj, set())):
                    t = 3
                elif ii >= 0 and jj >= 0 and (jj in ppi.get(ii, set())):
                    t = 4
                else:
                    t = 0
                etype[r, i + 1, j + 1] = t
                emag[r, i + 1, j + 1] = cos[i, j]
        etype[r, 0, :] = CLS_T; etype[r, :, 0] = CLS_T       # CLS routes to/from everything
    # standardise edge_feat columns over valid partner entries
    m = (tok_mask[:, :, None] > 0.5) & (np.arange(T)[None, :, None] >= 1)
    for k in range(3):
        col = edge_feat[:, :, k][m[:, :, 0]]
        mu, sd = float(col.mean()), float(col.std() + 1e-6)
        edge_feat[:, :, k] = (edge_feat[:, :, k] - mu) / sd
    edge_feat *= (np.arange(T)[None, :, None] >= 1)          # zero the self token's edge feats
    print(f"edges: etype counts {dict(zip(*np.unique(etype[etype>0], return_counts=True)))}")
    return edge_feat, etype, emag


def run(SEED, H, have, Y, hidx, Fg, tok_idx, tok_role, tok_mask, edge_feat, etype, emag, etype_sh, emag_sh):
    import torch
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
    EF = torch.tensor(edge_feat)
    ET = torch.tensor(etype); EM = torch.tensor(emag); ETs = torch.tensor(etype_sh); EMs = torch.tensor(emag_sh)
    Yspec = (Y * H.nontide.astype(np.float32)); Yn = Yspec / (np.linalg.norm(Yspec, axis=1, keepdims=True) + 1e-9)
    Ynt = torch.tensor(Yn)
    FEATIN = Fg.shape[1] + 3 + 4; DM = 128; HEADS = 4
    dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True)
    syms = list(dv["syms"]); Z = dv["Z"].astype(np.float32); srow = {s: i for i, s in enumerate(syms)}
    XK = np.stack([Z[srow[k]] for k in have]); XKn = XK / (np.linalg.norm(XK, axis=1, keepdims=True) + 1e-9)

    def gather(idx, use_ef, shuffled=False):
        f = Ft[TI[idx]]
        ef = EF[idx] if use_ef else torch.zeros(len(idx), TI.shape[1], 3)
        roleoh = torch.nn.functional.one_hot(TR[idx], 4).float()
        tokfeat = torch.cat([f, ef, roleoh], -1)
        et = (ETs if shuffled else ET)[idx]; em = (EMs if shuffled else EM)[idx]
        return tokfeat, TM[idx], et, em

    class EdgeMHA(torch.nn.Module):
        def __init__(s, use_bias):
            super().__init__()
            s.h = HEADS; s.dh = DM // HEADS; s.use_bias = use_bias
            s.q = torch.nn.Linear(DM, DM); s.k = torch.nn.Linear(DM, DM); s.v = torch.nn.Linear(DM, DM); s.o = torch.nn.Linear(DM, DM)
            s.etb = torch.nn.Embedding(N_ET, HEADS); s.emw = torch.nn.Parameter(torch.zeros(HEADS)); s.drop = torch.nn.Dropout(0.1)
        def forward(s, x, et, em, kpm):
            B, T, _ = x.shape
            q = s.q(x).view(B, T, s.h, s.dh).transpose(1, 2); k = s.k(x).view(B, T, s.h, s.dh).transpose(1, 2)
            v = s.v(x).view(B, T, s.h, s.dh).transpose(1, 2)
            sc = (q @ k.transpose(-2, -1)) / (s.dh ** 0.5)
            if s.use_bias:
                sc = sc + s.etb(et).permute(0, 3, 1, 2) + em.unsqueeze(1) * s.emw.view(1, s.h, 1, 1)
            sc = sc.masked_fill(kpm[:, None, None, :], -1e9)
            a = s.drop(torch.softmax(sc, -1))
            return s.o((a @ v).transpose(1, 2).reshape(B, T, DM))

    class Block(torch.nn.Module):
        def __init__(s, use_bias):
            super().__init__()
            s.n1 = torch.nn.LayerNorm(DM); s.att = EdgeMHA(use_bias)
            s.n2 = torch.nn.LayerNorm(DM)
            s.ff = torch.nn.Sequential(torch.nn.Linear(DM, 256), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, DM))
        def forward(s, x, et, em, kpm):
            x = x + s.att(s.n1(x), et, em, kpm); return x + s.ff(s.n2(x))

    class Model(torch.nn.Module):
        def __init__(s, use_ef, use_bias):
            super().__init__()
            s.use_ef = use_ef
            s.proj = torch.nn.Linear(FEATIN, DM); s.cls = torch.nn.Parameter(torch.randn(1, 1, DM) * 0.02)
            s.blocks = torch.nn.ModuleList([Block(use_bias) for _ in range(2)])
            s.head = torch.nn.Sequential(torch.nn.LayerNorm(DM), torch.nn.Linear(DM, 64))
        def forward(s, tokfeat, mask, et, em):
            h = s.proj(tokfeat); B = h.size(0)
            h = torch.cat([s.cls.expand(B, 1, DM), h], 1)
            m = torch.cat([torch.ones(B, 1), mask], 1); kpm = (m < 0.5)
            for blk in s.blocks:
                h = blk(h, et, em, kpm)
            e = s.head(h[:, 0]); return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    def embed_all(model, use_ef, shuffled=False):
        model.eval()
        with torch.no_grad():
            tf, ms, et, em = gather(torch.arange(len(have)), use_ef, shuffled)
            E = model(tf, ms, et, em).numpy()
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

    def train(use_ef, use_bias, shuffled=False, te=0.1, tt=0.1):
        torch.manual_seed(SEED)
        model = Model(use_ef, use_bias)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, bstate, wait = -1.0, None, 0
        for ep in range(200):
            model.train(); pi = np.random.RandomState(SEED * 100 + ep).permutation(len(tr_arr))
            for bs in range(0, len(tr_arr), 256):
                idx = torch.tensor(tr_arr[pi[bs:bs + 256]])
                if len(idx) < 8:
                    continue
                tf, ms, et, em = gather(idx, use_ef, shuffled)
                e = model(tf, ms, et, em)
                Se = e @ e.T; St = Ynt[idx] @ Ynt[idx].T; dm = torch.eye(len(idx)) * (-1e9)
                loss = -(torch.softmax(St / tt + dm, 1) * torch.log_softmax(Se / te + dm, 1)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 5 == 0:
                vr, _ = mean_recall(H, val, weighted(embed_all(model, use_ef, shuffled), 10, 0.1, 0.3))
                if vr > best:
                    best, bstate, wait = vr, {k: v.clone() for k, v in model.state_dict().items()}, 0
                else:
                    wait += 1
                if wait >= 15:
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
        K, tau, alpha = best_cfg
        sc, _ = mean_recall(H, test, weighted(E, K, tau, alpha), 50)
        return sc, best_cfg

    # ---- the ablation ----
    configs = [("base", False, False, False), ("+edge-feats", True, False, False),
               ("+edge-bias", False, True, False), ("+full", True, True, False),
               ("+full SHUFFLED-etype (ctrl)", True, True, True)]
    res = {}; cfgs = {}
    E_full = None
    for name, uef, ub, sh in configs:
        m = train(uef, ub, sh)
        E = embed_all(m, uef, sh)
        sc, cfg = tune_and_score(E)
        res[name] = sc; cfgs[name] = cfg
        if name == "+full":
            E_full = E

    # references + controls (identity shuffle on the full model)
    def tide_f(ko): return H.mover_freq
    def oracle_f(ko): return H._references(ko, 50)["ORACLE*"]
    res["TIDE-null"], _ = mean_recall(H, test, tide_f, 50)
    res["ORACLE* (ceiling)"], _ = mean_recall(H, test, oracle_f, 50)
    shp = rng.permutation(len(test)); smap = {test[i]: test[shp[i]] for i in range(len(test))}
    Kf, tauf, alf = cfgs["+full"]
    def full_shuf(ko):
        w = smap.get(ko); j = hidx.get(w)
        if j is None or E_full is None:
            return None
        s = (1 - alf) * (E_full[tr_arr] @ E_full[j]) + alf * (XKn[tr_arr] @ XKn[j])
        top = np.argsort(-s)[:Kf]; ww = np.exp((s[top] - s[top].max()) / tauf); ww /= ww.sum() + 1e-9
        return (ww[:, None] * Y[tr_arr[top]]).sum(0)
    res["full wrong-KO shuffle (ctrl)"], _ = mean_recall(H, test, full_shuf, 50)
    print(f"    [seed {SEED}] full readout cfg K={Kf} tau={tauf} alpha={alf}")
    return res


def main():
    H = Harness("K562")
    have, X, Y = build_xy(H)
    hidx = {k: i for i, k in enumerate(have)}
    Fg, tok_idx, tok_role, tok_mask, Gtok = build_context(H, have, return_gtok=True)
    edge_feat, etype, emag = build_edges(H, have, tok_idx, tok_mask, Gtok)
    # shuffled-edge-type control matrices: permute the (i,j) entries within each KO's real-token block (fixed seed)
    rr = np.random.RandomState(999); T1 = etype.shape[1]
    etype_sh = etype.copy(); emag_sh = emag.copy()
    for r in range(etype.shape[0]):
        idxs = np.arange(1, T1); rr.shuffle(idxs); p = np.concatenate([[0], idxs])
        etype_sh[r] = etype[r][np.ix_(p, p)]; emag_sh[r] = emag[r][np.ix_(p, p)]
        etype_sh[r, 0, :] = CLS_T; etype_sh[r, :, 0] = CLS_T

    seeds = [0, 1, 2]; all_res = []
    for s in seeds:
        print(f"\n{'=' * 60}\n[SPLIT SEED {s}]")
        res = run(s, H, have, Y, hidx, Fg, tok_idx, tok_role, tok_mask, edge_feat, etype, emag, etype_sh, emag_sh)
        for k, v in res.items():
            print(f"    {k:34s} {v:.3f}")
        all_res.append(res)
    keys = list(all_res[0].keys())
    agg = {k: (float(np.mean([r[k] for r in all_res])), float(np.std([r[k] for r in all_res]))) for k in keys}
    print(f"\n{'=' * 60}\nSTABILITY across {len(seeds)} splits (mean +/- sd):")
    for k in keys:
        print(f"    {k:34s} {agg[k][0]:.3f} +/- {agg[k][1]:.3f}")
    B = "base"; Ef = "+edge-feats"; Bi = "+edge-bias"; Fu = "+full"; Sh = "+full SHUFFLED-etype (ctrl)"
    d_full = [r[Fu] - r[B] for r in all_res]; d_ef = [r[Ef] - r[B] for r in all_res]
    d_bi = [r[Bi] - r[B] for r in all_res]; d_shuf = [r[Fu] - r[Sh] for r in all_res]
    print(f"\n    +full  - base                 : {np.mean(d_full):+.3f} +/- {np.std(d_full):.3f}  ({[round(x,3) for x in d_full]})")
    print(f"    +edge-feats - base            : {np.mean(d_ef):+.3f} +/- {np.std(d_ef):.3f}  ({[round(x,3) for x in d_ef]})")
    print(f"    +edge-bias  - base            : {np.mean(d_bi):+.3f} +/- {np.std(d_bi):.3f}  ({[round(x,3) for x in d_bi]})")
    print(f"    +full - SHUFFLED-etype (real!): {np.mean(d_shuf):+.3f} +/- {np.std(d_shuf):.3f}  ({[round(x,3) for x in d_shuf]})")
    fm = agg[Fu][0]; bm = agg[B][0]; tide = agg["TIDE-null"][0]; orc = agg["ORACLE* (ceiling)"][0]
    full_helps = np.mean(d_full) > 0.01 and all(x > 0 for x in d_full)
    bias_helps = np.mean(d_bi) > 0.01
    feats_help = np.mean(d_ef) > 0.01
    topology_real = np.mean(d_shuf) > 0.01           # real edge-types beat shuffled ones
    who = ("edge-BIAS carries the lift (topology was NOT fully latent in the SVD -- explicit routing along real complexes/edges is needed)"
           if bias_helps and np.mean(d_bi) >= np.mean(d_ef) else
           "edge-FEATURES carry the lift" if feats_help and np.mean(d_ef) > np.mean(d_bi) else
           "neither ablation gives a stable lift -- role+network-SVD already encoded this topology")
    headroom = (fm - tide) / max(orc - tide, 1e-9)
    verdict = (
        "TRANSFORMER v3 (transformer_ko3.py): v2 (CLS+InfoNCE-distillation+val-tuned weighted/hybrid transfer) + the user's fixes -- "
        "physics EDGE-FEATURES per partner (codep-cosine, log interface n_pdb, shared-complex count; the real magnitudes at scale, not a "
        "full ddG field) + Graphormer EDGE-CONDITIONED ATTENTION (learned per-head bias on edge-type + continuous codep-magnitude between "
        "every token pair). 3-split held-out K562 (tide-removed specific recall@50). ABLATION (mean+/-sd): base "
        f"{bm:.3f} | +edge-feats {agg[Ef][0]:.3f} | +edge-bias {agg[Bi][0]:.3f} | +full {fm:.3f} | shuffled-etype {agg[Sh][0]:.3f} | "
        f"oracle {orc:.3f}. +full - base = {np.mean(d_full):+.3f}+/-{np.std(d_full):.3f}"
        + (" (positive every split)." if full_helps else " (not a stable gain).")
        + f" ATTRIBUTION: +edge-bias {np.mean(d_bi):+.3f}, +edge-feats {np.mean(d_ef):+.3f} -> {who}. "
        + ("TOPOLOGY CONTROL PASSES: full beats SHUFFLED-edge-type by " + f"{np.mean(d_shuf):+.3f} -- the gain uses REAL interaction "
           "topology, not just 'some bias regularises'." if topology_real else
           f"TOPOLOGY CONTROL: full vs shuffled-etype {np.mean(d_shuf):+.3f} -- the edge-bias does NOT clearly need real topology (treat the "
           "bias gain cautiously; it may be acting as generic regularisation).")
        + f" Head-room used {headroom:.0%} of tide->oracle. HONEST BOUND: still bounded by the ~0.62 retrieval ceiling (the physics helps the "
        "model ROUTE within the reproducible signal, it cannot create signal the 96h buffered data lacks -- breaking 0.62 needs transient "
        "kinetics, unavailable here). CPU, deterministic per split.")
    print(f"\nVERDICT: {verdict}")
    out = {"mean_sd": {k: [round(agg[k][0], 4), round(agg[k][1], 4)] for k in keys},
           "full_minus_base": [round(x, 4) for x in d_full], "edgefeats_minus_base": [round(x, 4) for x in d_ef],
           "edgebias_minus_base": [round(x, 4) for x in d_bi], "full_minus_shuffled_etype": [round(x, 4) for x in d_shuf],
           "full_helps_all_splits": bool(full_helps), "topology_control_passes": bool(topology_real),
           "attribution": who, "headroom_used": round(float(headroom), 3), "verdict": verdict, "note": verdict}
    json.dump(out, open(OUT / "transformer_ko3.json", "w"), indent=1)
    print("\n  -> outputs/orphan/transformer_ko3.json")


if __name__ == "__main__":
    main()
