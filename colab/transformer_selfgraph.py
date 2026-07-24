"""transformer_selfgraph -- let the transformer MAKE ITS OWN GRAPH. The production model is told each knockout's token neighbourhood from a
CURATED interaction graph (co-dependency + complex + PPI, from cell_complete.json). Here we ask: can the model do as well if it builds its
neighbourhood purely from the DATA, with no curated databases?

Same metric-encoder + retrieval pipeline as transformer_ko; ONLY the token-neighbourhood source varies, and node features are held to
DepMap-only so no curated graph structure (network-SVD, degree) leaks in. Four graph modes, held-out K562, tide-removed specific-mover
recall@50, 3 seeds:

  curated     : the perturbed gene + its 32 prioritised partners from co-dependency + complex + PPI (the hand-built biological graph)
  knn_depmap  : the perturbed gene + its 32 NEAREST genes by DepMap dependency-profile cosine -- a graph the model derives from data alone
  random      : the perturbed gene + 32 random genes (control -- does ANY structured neighbourhood help)
  self_only   : the perturbed gene alone, no partners (ablation -- does the neighbourhood matter at all)

READS: the honest question is whether the curated PPI/complex knowledge beats a pure DepMap-kNN graph. Note co-dependency IS DepMap cosine,
so 'curated' already contains a data-derived core; this isolates the EXTRA value of the curated complex/PPI edges over a plain kNN graph.
GPU-ready; SMOKE (SG_SMOKE=1). Deterministic per split."""
import os, json, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness
from neural_ko import build_xy, mean_recall
OUT = Path("outputs/orphan")
M_PART = 32
SMOKE = bool(os.environ.get("SG_SMOKE"))


def build_graphs(H, have):
    """Return per-mode (tok_idx, tok_role, tok_mask) over a shared token-gene table, with DepMap-only node features Fg."""
    dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True)
    syms = list(dv["syms"]); Z = dv["Z"].astype(np.float32); srow = {s: i for i, s in enumerate(syms)}
    nr = np.linalg.norm(Z, axis=1, keepdims=True); nr[nr < 1e-9] = 1.0; Zc = Z / nr
    D = json.load(open(OUT / "cell_complete.json")); names = [g["name"] for g in D["genes"]]; N = len(names)
    nidx = {n: i for i, n in enumerate(names)}
    ppi = collections.defaultdict(set)
    for a, b in D["ppi"]:
        if isinstance(a, int) and isinstance(b, int) and a < N and b < N:
            ppi[a].add(b); ppi[b].add(a)
    ppideg = {i: len(v) for i, v in ppi.items()}
    gcplx = collections.defaultdict(set); cmem = []
    for ci, mem in enumerate(D["complexes"].values()):
        mm = [x for x in mem if isinstance(x, int) and x < N]; cmem.append(mm)
        for x in mm:
            gcplx[x].add(ci)
    codep = {int(k): [int(j) for j, _ in v] for k, v in D.get("codep", {}).items()}

    def curated_partners(g):
        gi_ = nidx.get(g)
        if gi_ is None:
            return []
        seen, out = {g}, []
        for j in codep.get(gi_, [])[:20]:
            nm = names[j]
            if nm not in seen and nm in srow:
                seen.add(nm); out.append((nm, 1))
        cp = set()
        for ci in gcplx.get(gi_, ()):
            cp.update(cmem[ci])
        for j in sorted(cp, key=lambda x: -ppideg.get(x, 0)):
            nm = names[j]
            if nm not in seen and nm in srow:
                seen.add(nm); out.append((nm, 2))
        for j in sorted(ppi.get(gi_, ()), key=lambda x: -ppideg.get(x, 0)):
            nm = names[j]
            if nm not in seen and nm in srow:
                seen.add(nm); out.append((nm, 3))
        return out[:M_PART]

    # knn_depmap: for each KO gene, the 32 nearest genes by DepMap cosine (over all depmap genes)
    have_rows = np.array([srow[g] for g in have if g in srow])
    have_feat = [g for g in have if g in srow]
    S = Zc[have_rows] @ Zc.T                                   # (nKO_feat, 18443) cosine
    knn = {}
    rng0 = np.random.RandomState(0)
    for i, g in enumerate(have_feat):
        order = np.argsort(-S[i])
        part = [syms[j] for j in order if syms[j] != g][:M_PART]
        knn[g] = part
    # random partners (fixed): 32 random depmap genes per KO
    randp = {}
    for g in have:
        randp[g] = [syms[j] for j in rng0.choice(len(syms), M_PART, replace=False)]

    # shared token-gene table = union of KOs + all partners across modes (all have DepMap)
    tok = set(g for g in have if g in srow)
    for g in have:
        tok.update(nm for nm, _ in curated_partners(g))
        tok.update(knn.get(g, []))
        tok.update(randp.get(g, []))
    Gtok = sorted(tok); t2i = {n: i for i, n in enumerate(Gtok)}
    Fdep = np.stack([Z[srow[n]] for n in Gtok]).astype(np.float32)
    Fg = ((Fdep - Fdep.mean(0)) / (Fdep.std(0) + 1e-6)).astype(np.float32)   # DepMap-only node features

    def pack(part_of):
        nk = len(have)
        ti = np.zeros((nk, 1 + M_PART), np.int64); tr = np.zeros((nk, 1 + M_PART), np.int64); tm = np.zeros((nk, 1 + M_PART), np.float32)
        for r, g in enumerate(have):
            if g not in t2i:
                continue
            ti[r, 0] = t2i[g]; tr[r, 0] = 0; tm[r, 0] = 1.0
            for c, item in enumerate(part_of(g), start=1):
                nm, role = item if isinstance(item, tuple) else (item, 1)
                if nm in t2i:
                    ti[r, c] = t2i[nm]; tr[r, c] = role; tm[r, c] = 1.0
        return ti, tr, tm

    modes = {
        "curated": pack(curated_partners),
        "knn_depmap": pack(lambda g: [(nm, 1) for nm in knn.get(g, [])]),
        "random": pack(lambda g: [(nm, 1) for nm in randp.get(g, [])]),
        "self_only": pack(lambda g: []),
    }
    npart = {m: (modes[m][2].sum(1) - 1).mean() for m in modes}
    print("mean partners/KO:", {m: round(float(v), 1) for m, v in npart.items()})
    return Fg, modes


def run(SEED, H, have, hidx, Fg, ti, tr, tm, Yk, Y):
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); torch.set_num_threads(4)
    rng = np.random.RandomState(SEED)
    scor = [k for k in H.scorable if k in hidx]
    perm = rng.permutation(len(scor)); n_test = int(0.30 * len(scor))
    test = sorted(scor[i] for i in perm[:n_test]); test_set = set(test)
    train_all = [i for i, k in enumerate(have) if k not in test_set and k in H.ki]
    vp = rng.permutation(len(train_all)); n_val = int(0.12 * len(train_all))
    val = sorted(have[train_all[i]] for i in vp[:n_val] if have[train_all[i]] in H.scorable); val_set = set(val)
    tr_arr = np.array([i for i in train_all if have[i] not in val_set])

    Ft = torch.tensor(Fg).to(DEVICE); TI = torch.tensor(ti); TR = torch.tensor(tr); TM = torch.tensor(tm)
    nontide = H.nontide
    Yspec = (Y * nontide.astype(np.float32)); Yn = Yspec / (np.linalg.norm(Yspec, axis=1, keepdims=True) + 1e-9)
    Ynt = torch.tensor(Yn).to(DEVICE)
    FEATIN = Fg.shape[1] + 4; DM = 128

    def gather(idx):
        f = Ft[TI[idx].to(DEVICE)]; ro = torch.nn.functional.one_hot(TR[idx].to(DEVICE), 4).float()
        return torch.cat([f, ro], -1), TM[idx].to(DEVICE)

    class Net(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.proj = torch.nn.Linear(FEATIN, DM)
            s.enc = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(DM, 4, 256, 0.1, batch_first=True, activation="gelu"), 2)
            s.head = torch.nn.Sequential(torch.nn.LayerNorm(DM), torch.nn.Linear(DM, 64))
        def forward(s, tf, mask):
            h = s.enc(s.proj(tf), src_key_padding_mask=(mask < 0.5))
            pooled = (h * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-9)
            e = s.head(pooled); return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    def embed_all():
        model.eval()
        with torch.no_grad():
            E = np.zeros((len(have), 64), np.float32)
            for bs in range(0, len(have), 512):
                idx = torch.arange(bs, min(bs + 512, len(have)))
                tf, ms = gather(idx); E[bs:bs + len(idx)] = model(tf, ms).cpu().numpy()
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def transfer(E):
        def f(ko):
            i = hidx.get(ko)
            if i is None:
                return None
            top = tr_arr[np.argsort(-(E[tr_arr] @ E[i]))[:10]]; return Yk[top].mean(0)
        return f

    EP = 12 if SMOKE else 120; PAT = 4 if SMOKE else 12
    model = Net().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, bstate, wait = -1.0, None, 0
    for ep in range(EP):
        model.train(); pi = np.random.RandomState(SEED * 100 + ep).permutation(len(tr_arr))
        for bs in range(0, len(tr_arr), 256):
            sel = pi[bs:bs + 256]
            if len(sel) < 8:
                continue
            idx = torch.tensor(tr_arr[sel]); tf, ms = gather(idx); e = model(tf, ms)
            Se = e @ e.T; St = Ynt[torch.tensor(tr_arr[sel]).to(DEVICE)] @ Ynt[torch.tensor(tr_arr[sel]).to(DEVICE)].T
            dm = torch.eye(len(sel), device=DEVICE) * (-1e9)
            loss = -(torch.softmax(St / 0.1 + dm, 1) * torch.log_softmax(Se / 0.1 + dm, 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 3 == 0:
            vr, _ = mean_recall(H, val, transfer(embed_all()))
            if vr > best:
                best, bstate, wait = vr, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
            if wait >= PAT:
                break
    if bstate:
        model.load_state_dict(bstate)
    E = embed_all()
    rec, _ = mean_recall(H, test, transfer(E), 50)
    return rec, mean_recall(H, test, lambda ko: H._references(ko, 50)["ORACLE*"], 50)[0]


def main():
    H = Harness("K562")
    have0, X, Y = build_xy(H)                                  # K562 |z| = Y (aligned to have0)
    have = have0
    hidx = {k: i for i, k in enumerate(have)}
    Yk = Y                                                     # K562 magnitude, aligned to have
    Fg, modes = build_graphs(H, have)

    seeds = [0] if SMOKE else [0, 1, 2]
    MODES = ["curated", "knn_depmap", "random", "self_only"]
    res = {m: [] for m in MODES}; orc = []
    for s in seeds:
        print(f"\n{'=' * 56}\n[SPLIT SEED {s}]{'  (SMOKE)' if SMOKE else ''}")
        for m in MODES:
            ti, tr, tm = modes[m]
            rec, orac = run(s, H, have, hidx, Fg, ti, tr, tm, Yk, Y)
            res[m].append(rec)
            if m == "curated":
                orc.append(orac)
            print(f"    {m:12s} recall {rec:.3f}")
        print(f"    oracle {orc[-1]:.3f}")
    print(f"\n{'=' * 56}\nSTABILITY across {len(seeds)} split(s):")
    agg = {m: (float(np.mean(res[m])), float(np.std(res[m]))) for m in MODES}
    for m in MODES:
        print(f"    {m:12s} {agg[m][0]:.3f} +/- {agg[m][1]:.3f}")
    om = float(np.mean(orc))
    d_knn = agg["knn_depmap"][0] - agg["curated"][0]
    verdict = (
        f"SELF-GRAPH (transformer_selfgraph.py): can the transformer build its own graph from data instead of the curated PPI/complex/codep "
        f"one? Same metric-encoder, DepMap-only node features, only the token-NEIGHBOURHOOD source varies. {len(seeds)} splits, held-out "
        f"K562. curated {agg['curated'][0]:.3f} | knn_depmap {agg['knn_depmap'][0]:.3f} | random {agg['random'][0]:.3f} | self_only "
        f"{agg['self_only'][0]:.3f} | oracle {om:.3f}. knn_depmap - curated = {d_knn:+.3f}. "
        + ("The MODEL'S OWN DepMap-kNN graph MATCHES the curated graph -- the curated PPI/complex databases add ~nothing beyond the "
           "dependency data the model already has; it can make its own graph."
           if abs(d_knn) < 0.02 else
           ("The curated graph BEATS the self-made kNN graph -- literature PPI/complex edges carry value beyond DepMap similarity."
            if d_knn < 0 else
            "The self-made DepMap-kNN graph BEATS the curated graph -- the hand-built PPI/complex edges are, if anything, a constraint.")) +
        f" (random {agg['random'][0]:.3f} and self_only {agg['self_only'][0]:.3f} bound how much the neighbourhood matters at all.) "
        "Bounded by the ~0.62 oracle. Deterministic.")
    print(f"\nVERDICT: {verdict}")
    if not SMOKE:
        json.dump({m: {"recall": round(agg[m][0], 4), "sd": round(agg[m][1], 4), "per_seed": [round(x, 4) for x in res[m]]} for m in MODES}
                  | {"oracle": round(om, 4), "knn_minus_curated": round(d_knn, 4), "verdict": verdict, "note": verdict},
                  open(OUT / "transformer_selfgraph.json", "w"), indent=1)
        print("\n  -> outputs/orphan/transformer_selfgraph.json")
    else:
        print("\n[SMOKE OK]")


if __name__ == "__main__":
    main()
