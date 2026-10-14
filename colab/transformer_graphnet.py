"""transformer_graphnet -- build ON the curated network graph: message-passing over the REAL within-neighbourhood wiring, not a bag of tokens.

The whole arc's one confirmed win (+0.335) was the partner SET (a KO's real curated partners as tokens). But the encoder is a set-transformer:
it attends over the partners as an unordered bag and THROWS AWAY which partner wires to which. The curated graph gives us that wiring for free
(build_edges: typed token-pair edges -- complex / co-dependency / PPI, + a CLS global node). This module tests the one untested lever: does the
WIRING carry signal beyond the SET?

Same metric-encoder, same features, same distillation + retrieval readout as v5/v3. The ONLY thing that changes is attention connectivity:

  base        -- full attention over the token set (the current set-transformer; a KO is a bag of partners)
  graph-real  -- attention HARD-MASKED to the real curated edges (+ self-loops + CLS): message-passing over the actual subgraph
  graph-shuf  -- attention masked to SHUFFLED edges (same density, wiring randomised): the control that says whether the REAL wiring matters

Decision: graph-real - base = does building on the graph's wiring help? graph-real - graph-shuf = is it the REAL wiring or just sparsity?
Honest prior (ceiling_cartography): the reachability door has only ~+0.012 encoder headroom, so expect marginal -- this settles the wiring
question with a clean control, it does not break the 0.62 data ceiling. Held-out K562, 3 seeds, deterministic. GPU-ready; SMOKE (TG_SMOKE=1)."""
import os, json, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness
from neural_ko import build_xy, mean_recall
from transformer_ko import build_context
from transformer_ko3 import build_edges, N_ET, CLS_T

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SMOKE = bool(os.environ.get("TG_SMOKE"))
DM = 128
HEADS = 4


def run(SEED, H, have, Y, hidx, Fg, tok_idx, tok_role, tok_mask, etype, emag, etype_sh, emag_sh):
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
    ET = torch.tensor(etype); EM = torch.tensor(emag); ETs = torch.tensor(etype_sh); EMs = torch.tensor(emag_sh)
    Yspec = (Y * H.nontide.astype(np.float32)); Ynt = torch.tensor(Yspec / (np.linalg.norm(Yspec, axis=1, keepdims=True) + 1e-9))
    FEATIN = Fg.shape[1] + 4
    dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True)
    syms = list(dv["syms"]); Z = dv["Z"].astype(np.float32); srow = {s: i for i, s in enumerate(syms)}
    XK = np.stack([Z[srow[k]] for k in have]); XKn = XK / (np.linalg.norm(XK, axis=1, keepdims=True) + 1e-9)

    def gather(idx, shuffled=False):
        f = Ft[TI[idx]]; roleoh = torch.nn.functional.one_hot(TR[idx], 4).float()
        tokfeat = torch.cat([f, roleoh], -1)
        et = (ETs if shuffled else ET)[idx]; em = (EMs if shuffled else EM)[idx]
        return tokfeat, TM[idx], et, em

    class GraphMHA(torch.nn.Module):
        def __init__(s, mode):
            super().__init__()
            s.h = HEADS; s.dh = DM // HEADS; s.mode = mode          # mode: 'full' | 'mask'
            s.q = torch.nn.Linear(DM, DM); s.k = torch.nn.Linear(DM, DM); s.v = torch.nn.Linear(DM, DM); s.o = torch.nn.Linear(DM, DM)
            s.drop = torch.nn.Dropout(0.1)

        def forward(s, x, et, kpm):
            B, T, _ = x.shape
            q = s.q(x).view(B, T, s.h, s.dh).transpose(1, 2); k = s.k(x).view(B, T, s.h, s.dh).transpose(1, 2)
            v = s.v(x).view(B, T, s.h, s.dh).transpose(1, 2)
            sc = (q @ k.transpose(-2, -1)) / (s.dh ** 0.5)
            if s.mode == "mask":
                sc = sc.masked_fill((et == 0).unsqueeze(1), -1e9)    # restrict to real edges (self=1/complex=2/codep=3/ppi=4/cls=5 are >0)
            sc = sc.masked_fill(kpm[:, None, None, :], -1e9)         # + block padded tokens (CLS key always survives -> no NaN row)
            a = s.drop(torch.softmax(sc, -1))
            return s.o((a @ v).transpose(1, 2).reshape(B, T, DM))

    class Block(torch.nn.Module):
        def __init__(s, mode):
            super().__init__()
            s.n1 = torch.nn.LayerNorm(DM); s.att = GraphMHA(mode)
            s.n2 = torch.nn.LayerNorm(DM)
            s.ff = torch.nn.Sequential(torch.nn.Linear(DM, 256), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, DM))

        def forward(s, x, et, kpm):
            x = x + s.att(s.n1(x), et, kpm); return x + s.ff(s.n2(x))

    class Model(torch.nn.Module):
        def __init__(s, mode):
            super().__init__()
            s.proj = torch.nn.Linear(FEATIN, DM); s.cls = torch.nn.Parameter(torch.randn(1, 1, DM) * 0.02)
            s.blocks = torch.nn.ModuleList([Block(mode) for _ in range(2)])
            s.head = torch.nn.Sequential(torch.nn.LayerNorm(DM), torch.nn.Linear(DM, 64))

        def forward(s, tokfeat, mask, et):
            h = s.proj(tokfeat); B = h.size(0)
            h = torch.cat([s.cls.expand(B, 1, DM), h], 1)           # CLS at index 0 -- matches etype layout
            m = torch.cat([torch.ones(B, 1), mask], 1); kpm = (m < 0.5)
            for blk in s.blocks:
                h = blk(h, et, kpm)
            e = s.head(h[:, 0]); return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    def embed_all(model, shuffled=False):
        model.eval()
        with torch.no_grad():
            tf, ms, et, em = gather(torch.arange(len(have)), shuffled)
            E = model(tf, ms, et).numpy()
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

    EP = 24 if SMOKE else 200; PAT = 3 if SMOKE else 15

    def train(mode, shuffled=False):
        torch.manual_seed(SEED)
        model = Model(mode); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, bstate, wait = -1.0, None, 0
        for ep in range(EP):
            model.train(); pi = np.random.RandomState(SEED * 100 + ep).permutation(len(tr_arr))
            for bs in range(0, len(tr_arr), 256):
                idx = torch.tensor(tr_arr[pi[bs:bs + 256]])
                if len(idx) < 8:
                    continue
                tf, ms, et, em = gather(idx, shuffled)
                e = model(tf, ms, et)
                Se = e @ e.T; St = Ynt[idx] @ Ynt[idx].T; dm = torch.eye(len(idx)) * (-1e9)
                loss = -(torch.softmax(St / 0.1 + dm, 1) * torch.log_softmax(Se / 0.1 + dm, 1)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 5 == 0:
                vr, _ = mean_recall(H, val, weighted(embed_all(model, shuffled), 10, 0.1, 0.3))
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
        grid = [(10, 0.1, 0.3)] if SMOKE else [(K, tau, al) for K in (5, 10, 20) for tau in (0.05, 0.1, 0.3) for al in (0.0, 0.3)]
        best_cfg, best_vr = grid[0], -1.0
        for cfg in grid:
            vr, _ = mean_recall(H, val, weighted(E, *cfg))
            if vr > best_vr:
                best_vr, best_cfg = vr, cfg
        sc, _ = mean_recall(H, test, weighted(E, *best_cfg), 50)
        return sc, best_cfg

    res = {}
    arms = [("base", "full", False), ("graph-real", "mask", False), ("graph-shuf", "mask", True)]
    for name, mode, sh in arms:
        E = embed_all(m := train(mode, sh), sh)
        res[name], _ = tune_and_score(E)

    def tide_f(ko): return H.mover_freq
    def oracle_f(ko): return H._references(ko, 50)["ORACLE*"]
    res["TIDE-null"], _ = mean_recall(H, test, tide_f, 50)
    res["ORACLE*"], _ = mean_recall(H, test, oracle_f, 50)
    return res


def main():
    H = Harness("K562")
    have, X, Y = build_xy(H)
    hidx = {k: i for i, k in enumerate(have)}
    Fg, tok_idx, tok_role, tok_mask, Gtok = build_context(H, have, return_gtok=True)
    edge_feat, etype, emag = build_edges(H, have, tok_idx, tok_mask, Gtok)
    # shuffled-wiring control: permute token positions 1..T within each KO (preserves edge-type density, randomises which pair) -- fixed seed
    rr = np.random.RandomState(999); T1 = etype.shape[1]
    etype_sh = etype.copy(); emag_sh = emag.copy()
    for r in range(etype.shape[0]):
        idxs = np.arange(1, T1); rr.shuffle(idxs); p = np.concatenate([[0], idxs])
        etype_sh[r] = etype[r][np.ix_(p, p)]; emag_sh[r] = emag[r][np.ix_(p, p)]
        etype_sh[r, 0, :] = CLS_T; etype_sh[r, :, 0] = CLS_T
    print(f"graphnet: {len(have)} KOs; real-edge density/KO "
          f"{float((etype[:, 1:, 1:] > 1).sum(1).sum(1).mean()):.1f} typed edges (excl self)")

    seeds = [0] if SMOKE else [0, 1, 2]
    all_res = []
    for s in seeds:
        print(f"\n{'='*60}\n[SPLIT SEED {s}]{'  (SMOKE)' if SMOKE else ''}")
        res = run(s, H, have, Y, hidx, Fg, tok_idx, tok_role, tok_mask, etype, emag, etype_sh, emag_sh)
        for k, v in res.items():
            print(f"    {k:14s} {v:.3f}")
        all_res.append(res)
    keys = list(all_res[0].keys())
    agg = {k: (float(np.mean([r[k] for r in all_res])), float(np.std([r[k] for r in all_res]))) for k in keys}
    d_graph = [r["graph-real"] - r["base"] for r in all_res]
    d_wiring = [r["graph-real"] - r["graph-shuf"] for r in all_res]
    print(f"\n{'='*60}\nSTABILITY across {len(seeds)} split(s) (mean +/- sd):")
    for k in keys:
        print(f"    {k:14s} {agg[k][0]:.3f} +/- {agg[k][1]:.3f}")
    print(f"\n    graph-real - base   (does WIRING help beyond the set?): {np.mean(d_graph):+.3f} +/- {np.std(d_graph):.3f}  "
          f"({[round(x,3) for x in d_graph]})")
    print(f"    graph-real - graph-shuf (is it the REAL wiring?)     : {np.mean(d_wiring):+.3f} +/- {np.std(d_wiring):.3f}  "
          f"({[round(x,3) for x in d_wiring]})")

    graph_helps = np.mean(d_graph) > 0.01 and all(x > 0 for x in d_graph)
    wiring_real = np.mean(d_wiring) > 0.01
    base, gr, tide, orc = agg["base"][0], agg["graph-real"][0], agg["TIDE-null"][0], agg["ORACLE*"][0]
    verdict = (
        f"GRAPH-NATIVE ENCODER (transformer_graphnet.py): build ON the curated network by MESSAGE-PASSING over a KO's real within-"
        f"neighbourhood wiring (typed complex/codep/PPI edges + CLS global node) instead of a bag-of-partners set-transformer. Same features, "
        f"same distillation + retrieval readout; only attention CONNECTIVITY changes. {len(seeds)} splits, held-out K562. "
        f"base(set) {base:.3f} -> graph-real(masked to real edges) {gr:.3f} = {np.mean(d_graph):+.3f}"
        + (" -- the WIRING helps beyond the partner set" if graph_helps else " -- the wiring does NOT help beyond the partner set (a KO is "
           "well-described as a BAG of its curated partners; which-connects-to-which adds nothing the set-transformer didn't already get)")
        + f". Real-vs-shuffled-wiring {np.mean(d_wiring):+.3f} "
        + ("(real topology beats randomised) " if wiring_real else "(indistinguishable from random wiring of the same density) ")
        + f"tide {tide:.3f} < [base {base:.3f}, graph {gr:.3f}] < oracle {orc:.3f}. "
        + ("Consistent with ceiling_cartography (all doors shut, ~+0.012 encoder headroom): building on the graph's wiring is a wash -- the "
           "partner SET already carried the reproducible signal; the ceiling is the DATA, not the graph representation." if not graph_helps
           else "The wiring is a real, if modest, lever on top of the partner set.")
        + " Deterministic.")
    print(f"\nVERDICT: {verdict}")
    if not SMOKE:
        json.dump({k: [round(agg[k][0], 4), round(agg[k][1], 4)] for k in keys}
                  | {"graph_minus_base": round(float(np.mean(d_graph)), 4), "graph_minus_shuffled": round(float(np.mean(d_wiring)), 4),
                     "verdict": verdict, "note": verdict}, open(OUT / "transformer_graphnet.json", "w"), indent=1)
        print("\n  -> outputs/orphan/transformer_graphnet.json")
    else:
        print("\n[SMOKE OK]")


if __name__ == "__main__":
    main()
