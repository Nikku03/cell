"""transformer_ko6 -- v6 = v5 + the mRNA/REGULATORY layer (the user's insight: Perturb-seq reads mRNA, so same-layer regulatory data should
help where protein-structure could not). Two mechanisms:

  (1) REGULATORY TOKEN FEATURES (all KOs): each gene token gains [is-TF, log regulatory OUT-degree, log IN-degree] from the unified directed
      regulatory graph (TRRUST curated + 60k SIGNOR/CollecTRI causal + K562 ChIP = 4,048 sources / 124k edges). Makes the retrieval embedding
      regulation-aware.
  (2) DIRECT TF->TARGET CHANNEL (fires for the ~23% of KOs that are regulatory sources): a knockout's known targets are predicted movers
      directly, fused with the retrieval prediction (weight beta tuned on val). This is a MECHANISTIC near-field channel, independent of
      retrieval -- and pathway_ko_verify already VERIFIED TRRUST TF->target at 6x enrichment for real KO effects.

Built on v5 (6 lines + 630 Replogle KOs, distillation, mean-transfer readout -- the aggregator was measured to hurt, so dropped). Reports
OVERALL held-out K562 recall AND the TF-KO SUBSET separately (the honest place the direct channel can help). Ablation: v5 (no reg) |
+reg-features | +reg-features+direct-channel. 3 splits. Deterministic; GPU-ready; SMOKE (V6_SMOKE=1).

HONEST PRIOR: only ~23% of KOs are regulatory sources with a median of 2 measured targets, so the DIRECT channel can only move the TF-KO
subset; the reg FEATURES might help the embedding a little more broadly. Still bounded by the ~0.62 oracle."""
import os, json, pickle, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness
from neural_ko import build_xy, mean_recall
from transformer_ko import build_context
from transformer_ko5 import profiles, ALL_SRC, SP
OUT = Path("outputs/orphan")
SMOKE = bool(os.environ.get("V6_SMOKE"))


def reg_graph(names):
    reg = collections.defaultdict(set)
    tr = json.load(open(OUT / "trrust_regulon.json"))["tf_targets"]
    for tf, ts in tr.items():
        for t in ts:
            reg[tf].add(t[0] if isinstance(t, (list, tuple)) else t)
    for e in json.load(open(OUT / "causal_edges.json"))["edges"]:
        a, b = e[0], e[1]
        if isinstance(a, int) and isinstance(b, int) and a < len(names) and b < len(names):
            reg[names[a]].add(names[b])
    for tf, ts in json.load(open(OUT / "chip_reg_edges.json")).get("tf_targets", {}).items():
        for t in ts:
            reg[tf].add(t[0] if isinstance(t, (list, tuple)) else t)
    return reg


def run(SEED, H, have, hidx, Fg, tok_idx, tok_role, tok_mask, Yk, prof, reg, reg_tgt_idx, use_reg):
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
    FEATIN = Fg.shape[1] + 4; DM = 128
    lid = {L: i for i, L in enumerate(ALL_SRC)}
    line_ex, line_Yn = {}, {}
    for L in ALL_SRC:
        idxs, Yn = prof[L]
        keep = [j for j, gi_ in enumerate(idxs) if have[gi_] not in test_set]
        line_ex[L] = np.array([idxs[j] for j in keep]); line_Yn[L] = torch.tensor(Yn[keep]).to(DEVICE)

    def gather(idx):
        f = Ft[TI[idx].to(DEVICE)]; roleoh = torch.nn.functional.one_hot(TR[idx].to(DEVICE), 4).float()
        return torch.cat([f, roleoh], -1), TM[idx].to(DEVICE)

    class Net(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.proj = torch.nn.Linear(FEATIN, DM)
            s.enc = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(DM, 4, 256, 0.1, batch_first=True, activation="gelu"), 2)
            s.line = torch.nn.Embedding(len(ALL_SRC), DM)
            s.head = torch.nn.Sequential(torch.nn.LayerNorm(DM), torch.nn.Linear(DM, 64))
        def forward(s, tf, mask, li):
            h = s.enc(s.proj(tf), src_key_padding_mask=(mask < 0.5))
            pooled = (h * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-9)
            e = s.head(pooled + s.line(li)); return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    def emb_batch(model, idx, li):
        tf, ms = gather(idx); return model(tf, ms, torch.full((len(idx),), li, device=DEVICE))

    def embed_all(model):
        model.eval()
        with torch.no_grad():
            E = np.zeros((len(have), 64), np.float32)
            for bs in range(0, len(have), 384):
                idx = torch.arange(bs, min(bs + 384, len(have)))
                E[bs:bs + len(idx)] = emb_batch(model, idx, lid["K562"]).cpu().numpy()
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    EP = 12 if SMOKE else 120; PAT = 4 if SMOKE else 12
    model = Net().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def retr(E, K=10):
        def f(ko):
            i = hidx.get(ko)
            if i is None:
                return None
            top = k562_tr[np.argsort(-(E[k562_tr] @ E[i]))[:K]]; return Yk[top].mean(0)
        return f
    best, bstate, wait = -1.0, None, 0
    for ep in range(EP):
        model.train(); rs = np.random.RandomState(SEED * 100 + ep); Ls = list(ALL_SRC); rs.shuffle(Ls)
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
            vr, _ = mean_recall(H, val, retr(embed_all(model)))
            if vr > best:
                best, bstate, wait = vr, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
            if wait >= PAT:
                break
    if bstate:
        model.load_state_dict(bstate)
    E = embed_all(model)

    # direct regulatory channel: ko -> its measured K562 targets (z-fused with the retrieval vector, beta tuned on val)
    tide_unit = H.mover_freq
    def reg_vec(ko):
        v = np.zeros(H.nG, np.float32)
        for j in reg_tgt_idx.get(ko, ()):
            v[j] = 1.0
        return v
    def fused(E, beta):
        rf = retr(E)
        def f(ko):
            base = rf(ko)
            if base is None:
                return None
            rv = reg_vec(ko)
            if rv.sum() == 0 or beta == 0:
                return base
            bz = (base - base.mean()) / (base.std() + 1e-9)
            rz = (rv - rv.mean()) / (rv.std() + 1e-9)
            return (1 - beta) * bz + beta * rz
        return f

    def score(pred, kos):
        rs = []
        for ko in kos:
            v = pred(ko)
            if v is None:
                continue
            r = H.ki[ko]; mv = np.where(H.mover[r])[0]; spec = set(int(i) for i in mv if H.nontide[i])
            if len(spec) >= 1:
                rs.append(H._recall(v, spec, H.nontide_idx, 50))
        return float(np.mean(rs)) if rs else float("nan")

    tf_kos = [k for k in test if k in reg_tgt_idx and len(reg_tgt_idx[k]) > 0]     # TF-KO subset (direct channel fires)
    out = {"recall_mean": score(retr(E), test), "n_tf": len(tf_kos)}
    if use_reg:
        # tune beta on val (val TF-KOs)
        val_tf = [k for k in val if k in reg_tgt_idx and len(reg_tgt_idx[k]) > 0]
        best_b, bv = 0.0, -1
        for b in (0.0, 0.1, 0.2, 0.35, 0.5):
            vr = score(fused(E, b), val) if val_tf else -1
            if vr > bv:
                bv, best_b = vr, b
        out["beta"] = best_b
        out["recall_fused"] = score(fused(E, best_b), test)
        out["recall_mean_tfsub"] = score(retr(E), tf_kos) if tf_kos else float("nan")
        out["recall_fused_tfsub"] = score(fused(E, best_b), tf_kos) if tf_kos else float("nan")
    out["oracle"] = score(lambda ko: H._references(ko, 50)["ORACLE*"], test)
    out["tide"] = score(lambda ko: H.mover_freq, test)
    return out


def main():
    H = Harness("K562")
    have0, X, Y0 = build_xy(H)
    dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True); syms = set(dv["syms"])
    allko = set()
    for L in ALL_SRC:
        allko |= set(pickle.load(open(f"{SP}/nlz_{L}.pkl", "rb")))
    have = sorted(k for k in allko if k in syms); hidx = {k: i for i, k in enumerate(have)}
    D = json.load(open(OUT / "cell_complete.json")); names = [g["name"] for g in D["genes"]]
    reg = reg_graph(names)
    Fg, tok_idx, tok_role, tok_mask, Gtok = build_context(H, have, return_gtok=True)
    # (1) regulatory token features appended to Fg (per token gene): [is_tf, log_out, log_in]
    indeg = collections.Counter()
    for s, ts in reg.items():
        for t in ts:
            indeg[t] += 1
    def zc(a):
        a = np.asarray(a, np.float32); return (a - a.mean()) / (a.std() + 1e-6)
    is_tf = zc([1.0 if g in reg else 0.0 for g in Gtok])
    outd = zc([np.log1p(len(reg.get(g, ()))) for g in Gtok])
    ind = zc([np.log1p(indeg.get(g, 0)) for g in Gtok])
    RF = np.stack([is_tf, outd, ind], 1).astype(np.float32)
    Fg_reg = np.concatenate([Fg, RF], 1).astype(np.float32)
    # direct-channel target index: ko -> [K562 gene indices of its regulatory targets]
    reg_tgt_idx = {}
    for s, ts in reg.items():
        idxs = [H.gi[t] for t in ts if t in H.gi]
        if idxs:
            reg_tgt_idx[s] = idxs
    Yk = np.zeros((len(have), H.nG), np.float32)
    for i, k in enumerate(have):
        if k in H.ki:
            Yk[i] = H.A[H.ki[k]]
    prof = profiles(have)
    print(f"v6: {len(have)} KOs; reg graph {len(reg)} sources; {sum(1 for k in have if k in reg_tgt_idx)} KO-universe genes are reg sources")

    seeds = [0] if SMOKE else [0, 1, 2]
    base, feat, fus, base_tf, fus_tf = [], [], [], [], []
    for s in seeds:
        print(f"\n{'=' * 58}\n[SPLIT SEED {s}]{'  (SMOKE)' if SMOKE else ''}")
        r0 = run(s, H, have, hidx, Fg, tok_idx, tok_role, tok_mask, Yk, prof, reg, reg_tgt_idx, use_reg=False)          # v5 (no reg)
        r1 = run(s, H, have, hidx, Fg_reg, tok_idx, tok_role, tok_mask, Yk, prof, reg, reg_tgt_idx, use_reg=True)       # +reg feats +channel
        print(f"    v5 (no reg)              overall {r0['recall_mean']:.3f}")
        print(f"    +reg-features            overall {r1['recall_mean']:.3f}")
        print(f"    +reg-features +channel   overall {r1['recall_fused']:.3f}  (beta {r1['beta']})")
        print(f"    TF-KO subset (n~{r1['n_tf']}): mean {r1['recall_mean_tfsub']:.3f} -> fused {r1['recall_fused_tfsub']:.3f}")
        print(f"    tide {r1['tide']:.3f}  oracle {r1['oracle']:.3f}")
        base.append(r0['recall_mean']); feat.append(r1['recall_mean']); fus.append(r1['recall_fused'])
        base_tf.append(r1['recall_mean_tfsub']); fus_tf.append(r1['recall_fused_tfsub'])
    mb, mf, mfu = np.nanmean(base), np.nanmean(feat), np.nanmean(fus)
    mbt, mft = np.nanmean(base_tf), np.nanmean(fus_tf)
    orc = float(r1['oracle'])
    print(f"\n{'=' * 58}\nSTABILITY across {len(seeds)} split(s):")
    print(f"    v5 (no reg)            : {mb:.3f} +/- {np.nanstd(base):.3f}")
    print(f"    +reg-features          : {mf:.3f} +/- {np.nanstd(feat):.3f}  (feat - v5 {mf-mb:+.3f})")
    print(f"    +reg-features +channel : {mfu:.3f} +/- {np.nanstd(fus):.3f}  (fused - v5 {mfu-mb:+.3f})")
    print(f"    TF-KO subset: mean {mbt:.3f} -> fused {mft:.3f}  (channel on subset {mft-mbt:+.3f})")
    verdict = (
        f"TRANSFORMER v6 (transformer_ko6.py): v5 (multi-line+Replogle, distillation, mean transfer) + the mRNA/REGULATORY layer -- "
        f"regulatory token features (is-TF, in/out-degree from TRRUST+60k SIGNOR/CollecTRI+ChIP) and a DIRECT TF->target channel (fuse a "
        f"knockout's known targets, beta tuned on val). {len(seeds)} splits, held-out K562. OVERALL: v5 {mb:.3f} -> +reg-features {mf:.3f} "
        f"({mf-mb:+.3f}) -> +direct-channel {mfu:.3f} ({mfu-mb:+.3f}). TF-KO SUBSET (~{r1['n_tf']} of ~113 test KOs, the ~23% that are "
        f"regulatory sources): mean {mbt:.3f} -> fused {mft:.3f} ({mft-mbt:+.3f}). "
        + ("The regulatory layer HELPS -- same-layer features/channel add signal the structural features could not."
           if (mfu - mb > 0.005) else
           "The regulatory layer is MARGINAL overall: only ~23% of KOs are sources (median 2 measured targets), so the direct channel "
           "can only move the TF-KO subset, and that is diluted 4:1 by the majority non-source KOs; the reg FEATURES add little the "
           "co-dependency/expression retrieval did not already capture. Right layer, but thin coverage on THIS KO panel.") +
        f" tide {r1['tide']:.3f}, oracle {orc:.3f}. Bounded by the ~0.62 oracle. Deterministic; GPU-ready.")
    print(f"\nVERDICT: {verdict}")
    if not SMOKE:
        json.dump({"v5_no_reg": round(float(mb), 4), "plus_reg_features": round(float(mf), 4),
                   "plus_direct_channel": round(float(mfu), 4), "tf_subset_mean": round(float(mbt), 4),
                   "tf_subset_fused": round(float(mft), 4), "oracle": round(orc, 4), "verdict": verdict, "note": verdict},
                  open(OUT / "transformer_ko6.json", "w"), indent=1)
        print("\n  -> outputs/orphan/transformer_ko6.json")
    else:
        print("\n[SMOKE OK]")


if __name__ == "__main__":
    main()
