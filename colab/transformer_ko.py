"""transformer_ko -- a TRANSFORMER for the knockout far-field that ingests the knockout's INTERACTION CONTEXT, not just the gene.

The user's spec: a transformer, fed (a) the network we built, (b) "the other PPI after knockout and all" (the partners a knockout
touches), and (c) essentiality. A knockout is represented as a SET OF TOKENS and the transformer attends over them:
    token 0   = the perturbed gene itself                       (role SELF)
    tokens... = its post-knockout partners: DepMap CO-DEPENDENCY partners, COMPLEX co-members, and PPI neighbours (roles CODEP/CPLX/PPI)
each token's features = [ DepMap dependency profile (1150-d) | our network-SVD embedding (64-d) | ESSENTIALITY | log-degree | role ].
Self-attention learns WHICH partners matter for the response; masked-mean pooling gives the knockout embedding.

WHY this frame (learned by the last module, neural_ko): a REGRESSION net on this problem collapses to the mean (predicts the tide). The
one deep frame that works is LEARNED RETRIEVAL -- learn an embedding whose cosine matches tide-removed RESPONSE similarity, then transfer a
knockout's nearest TRAIN neighbours' measured profile. So the transformer produces the KO embedding and is trained with that metric-learning
objective (NOT regression). Scored on the same held-out K562 bench (tide-removed specific-mover recall@50), 3 random splits, mean +/- sd.

THE HONEST BAR: does attention over the post-KO interaction neighbourhood + essentiality BEAT the plain MLP on the gene's OWN features
(the previous best, ~0.41)? Controls: MLP-self baseline (same features, no neighbourhood), transformer-self-only at inference (remove the
neighbourhood), and a wrong-KO shuffle (gene-identity control). Deterministic (torch.manual_seed + RandomState per split). CPU.
"""
import json, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness
from neural_ko import build_xy, spec_recall, mean_recall
OUT = Path("outputs/orphan")
M_PART = 32                     # max partners per knockout (self + up to 32)


def build_context(H, have):
    """Build, for every knockout, its token set (self + prioritised partners) and a global gene-feature matrix the tokens index into."""
    dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True)
    syms = list(dv["syms"]); Z = dv["Z"].astype(np.float32); srow = {s: i for i, s in enumerate(syms)}
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; N = len(names)
    essmap = {g["name"]: float(g.get("ess") or 0) for g in D["genes"]}

    # network-SVD view (same as neural_ko: PPI+complex+pathway+regulon graph over H.genes)
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    gi = H.gi; nG = H.nG
    rows, cols = [], []
    for g, nbs in H.nb.items():
        a = gi.get(g)
        if a is None:
            continue
        for b in nbs:
            bb = gi.get(b)
            if bb is not None:
                rows.append(a); cols.append(bb)
    Anet = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(nG, nG))
    Anet = Anet.maximum(Anet.T)
    deg = np.asarray(Anet.sum(1)).ravel() + 1.0
    Dinv = csr_matrix((1.0 / np.sqrt(deg), (range(nG), range(nG))), shape=(nG, nG))
    U, S, _ = svds((Dinv @ Anet @ Dinv).astype(np.float32), k=64)
    Gemb = (U * S).astype(np.float32)
    name2net = {H.genes[i]: Gemb[i] for i in range(nG)}

    # interaction structure (int-indexed into `names`)
    ppi = collections.defaultdict(set)
    for a, b in D["ppi"]:
        if isinstance(a, int) and isinstance(b, int) and a < N and b < N:
            ppi[a].add(b); ppi[b].add(a)
    ppideg = {i: len(v) for i, v in ppi.items()}
    gcplx = collections.defaultdict(set); cmem = []
    for ci, mem in enumerate(D["complexes"].values()):
        mm = [x for x in mem if isinstance(x, int) and x < N]
        cmem.append(mm)
        for x in mm:
            gcplx[x].add(ci)
    codep = {int(k): [int(j) for j, _ in v] for k, v in D.get("codep", {}).items()}
    nidx = {n: i for i, n in enumerate(names)}

    def partners(gname):
        """prioritised partner NAMES: co-dependency (functional/obligate) -> complex co-members -> PPI neighbours."""
        gi_ = nidx.get(gname)
        if gi_ is None:
            return []
        seen, out = {gname}, []
        for j in codep.get(gi_, [])[:20]:                    # co-dep first (the 'affected after KO' functional partners)
            nm = names[j]
            if nm not in seen and nm in srow:
                seen.add(nm); out.append((nm, 1))
        cpart = set()
        for ci in gcplx.get(gi_, ()):
            cpart.update(cmem[ci])
        for j in sorted(cpart, key=lambda x: -ppideg.get(x, 0)):
            nm = names[j]
            if nm not in seen and nm in srow:
                seen.add(nm); out.append((nm, 2))
        for j in sorted(ppi.get(gi_, ()), key=lambda x: -ppideg.get(x, 0)):
            nm = names[j]
            if nm not in seen and nm in srow:
                seen.add(nm); out.append((nm, 3))
        return out[:M_PART]

    # token gene universe = every gene appearing as self or partner (must have a DepMap profile)
    tok_names = set(k for k in have if k in srow)
    ko_partners = {}
    for k in have:
        p = partners(k)
        ko_partners[k] = p
        tok_names.update(nm for nm, _ in p)
    Gtok = sorted(tok_names); t2i = {n: i for i, n in enumerate(Gtok)}
    # base feature matrix over Gtok: depmap(1150) | net(64) | ess(1) | log-degree(1)   (role appended per-token at forward)
    Fdep = np.stack([Z[srow[n]] for n in Gtok]).astype(np.float32)
    Fnet = np.stack([name2net.get(n, np.zeros(64, np.float32)) for n in Gtok]).astype(np.float32)
    Fess = np.array([[essmap.get(n, 0.0)] for n in Gtok], np.float32)
    Fdeg = np.array([[np.log1p(ppideg.get(nidx.get(n, -1), 0))] for n in Gtok], np.float32)
    # standardise each block by its own column stats over Gtok (feature preprocessing, split-independent, no label leak)
    def zc(A):
        return (A - A.mean(0)) / (A.std(0) + 1e-6)
    Fg = np.concatenate([zc(Fdep), zc(Fnet), zc(Fess), zc(Fdeg)], axis=1).astype(np.float32)

    # per-KO token index / role / mask arrays
    nk = len(have)
    tok_idx = np.zeros((nk, 1 + M_PART), np.int64)
    tok_role = np.zeros((nk, 1 + M_PART), np.int64)
    tok_mask = np.zeros((nk, 1 + M_PART), np.float32)
    for r, k in enumerate(have):
        tok_idx[r, 0] = t2i[k]; tok_role[r, 0] = 0; tok_mask[r, 0] = 1.0
        for c, (nm, role) in enumerate(ko_partners[k], start=1):
            tok_idx[r, c] = t2i[nm]; tok_role[r, c] = role; tok_mask[r, c] = 1.0
    npart = tok_mask.sum(1) - 1
    print(f"context: {len(Gtok)} token genes; partners/KO median {int(np.median(npart))} (mean {npart.mean():.1f}, "
          f"{int((npart == 0).sum())} KOs isolated)")
    return Fg, tok_idx, tok_role, tok_mask


def run(SEED, H, have, Y, hidx, Fg, tok_idx, tok_role, tok_mask):
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
    # RANDOM-PARTNER control tokens: SAME counts/roles/mask, but each partner replaced by a RANDOM token gene (fixed seed, split-
    # independent). Decides whether the gain is the REAL interaction structure or just "more tokens / capacity".
    rtok = tok_idx.copy(); rr = np.random.RandomState(12345)
    part_pos = (tok_mask > 0.5) & (np.arange(tok_idx.shape[1])[None, :] >= 1)
    rtok[part_pos] = rr.randint(0, Fg.shape[0], size=int(part_pos.sum()))
    RTI = torch.tensor(rtok)
    Yspec = (Y * H.nontide.astype(np.float32)); Yn = Yspec / (np.linalg.norm(Yspec, axis=1, keepdims=True) + 1e-9)
    Ynt = torch.tensor(Yn)
    FEAT = Fg.shape[1]; DM = 128

    def gather(idx, TIx=TI):                                  # idx: LongTensor of KO rows -> (B, T, FEAT+4 role-onehot)
        f = Ft[TIx[idx]]                                      # (B,T,FEAT)
        roleoh = torch.nn.functional.one_hot(TR[idx], 4).float()  # (B,T,4)
        return torch.cat([f, roleoh], -1), TM[idx]

    class Transformer(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.proj = torch.nn.Linear(FEAT + 4, DM)
            lyr = torch.nn.TransformerEncoderLayer(DM, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True, activation="gelu")
            s.enc = torch.nn.TransformerEncoder(lyr, num_layers=2)
            s.head = torch.nn.Sequential(torch.nn.LayerNorm(DM), torch.nn.Linear(DM, 64))
        def forward(s, x, mask):
            h = s.proj(x)
            h = s.enc(h, src_key_padding_mask=(mask < 0.5))
            pooled = (h * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-9)
            e = s.head(pooled); return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    class MLPSelf(torch.nn.Module):                           # baseline: same features, SELF token only (no neighbourhood)
        def __init__(s):
            super().__init__()
            s.net = torch.nn.Sequential(torch.nn.Linear(FEAT + 4, 256), torch.nn.LayerNorm(256), torch.nn.GELU(),
                                        torch.nn.Dropout(0.2), torch.nn.Linear(256, 64))
        def forward(s, x, mask):
            e = s.net(x[:, 0]); return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    def embed_all(model, self_only=False, TIx=TI):
        model.eval()
        with torch.no_grad():
            xs, ms = gather(torch.arange(len(have)), TIx)
            if self_only:
                ms = ms.clone(); ms[:, 1:] = 0.0
            E = model(xs, ms).numpy()
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def retriever(E):
        def f(ko):
            i = hidx.get(ko)
            if i is None:
                return None
            sim = E[tr_arr] @ E[i]; top = tr_arr[np.argsort(-sim)[:10]]
            return Y[top].mean(0)
        return f

    def train_metric(model, tag, TIx=TI):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, bstate, wait = -1.0, None, 0
        for ep in range(220):
            model.train()
            pi = np.random.RandomState(SEED * 100 + ep).permutation(len(tr_arr))
            for bs in range(0, len(tr_arr), 256):
                idx = torch.tensor(tr_arr[pi[bs:bs + 256]])
                xs, ms = gather(idx, TIx)
                e = model(xs, ms)
                pred = e @ e.T; tgt = Ynt[idx] @ Ynt[idx].T
                loss = ((pred - tgt) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 5 == 0:
                vr, _ = mean_recall(H, val, retriever(embed_all(model, TIx=TIx)))
                if vr > best:
                    best, bstate, wait = vr, {k: v.clone() for k, v in model.state_dict().items()}, 0
                else:
                    wait += 1
                if wait >= 18:
                    break
        if bstate:
            model.load_state_dict(bstate)
        return model, best

    tf, _ = train_metric(Transformer(), "transformer")
    mlp, _ = train_metric(MLPSelf(), "mlp-self")
    tfr, _ = train_metric(Transformer(), "transformer-random-partners", TIx=RTI)   # decisive control
    E_tf = embed_all(tf); E_tf_self = embed_all(tf, self_only=True); E_mlp = embed_all(mlp)
    E_tfr = embed_all(tfr, TIx=RTI)

    # references + controls
    def tide_f(ko): return H.mover_freq
    def oracle_f(ko): return H._references(ko, 50)["ORACLE*"]
    Xn = None
    # classical raw-cosine transfer on DepMap (train-only) for the in-split classical bar
    dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True)
    syms = list(dv["syms"]); Z = dv["Z"].astype(np.float32); srow = {s: i for i, s in enumerate(syms)}
    XK = np.stack([Z[srow[k]] for k in have]); XKn = XK / (np.linalg.norm(XK, axis=1, keepdims=True) + 1e-9)
    def codep_f(ko):
        i = hidx.get(ko)
        if i is None:
            return None
        sim = XKn[tr_arr] @ XKn[i]; top = tr_arr[np.argsort(-sim)[:10]]
        return Y[top].mean(0)
    # wrong-KO shuffle control for the transformer (gene-identity control)
    sh = rng.permutation(len(test)); shuffle_map = {test[i]: test[sh[i]] for i in range(len(test))}
    def tf_shuf_f(ko):
        w = shuffle_map.get(ko); j = hidx.get(w)
        return None if j is None else Y[tr_arr[np.argsort(-(E_tf[tr_arr] @ E_tf[j]))[:10]]].mean(0)

    K = 50; res = {}
    res["TIDE-null"], _ = mean_recall(H, test, tide_f, K)
    res["classical raw-cosine"], _ = mean_recall(H, test, codep_f, K)
    res["MLP-self (feats, no neighbourhood)"], _ = mean_recall(H, test, retriever(E_mlp), K)
    res["TRANSFORMER self-only (infer)"], _ = mean_recall(H, test, retriever(E_tf_self), K)
    res["TRANSFORMER (self+neighbourhood)"], _ = mean_recall(H, test, retriever(E_tf), K)
    res["TRANSFORMER random-partners (ctrl)"], _ = mean_recall(H, test, retriever(E_tfr), K)
    res["TRANSFORMER shuffle (control)"], _ = mean_recall(H, test, tf_shuf_f, K)
    res["ORACLE* (ceiling)"], _ = mean_recall(H, test, oracle_f, K)
    return res, len(train), len(test)


def main():
    H = Harness("K562")
    have, X, Y = build_xy(H)
    hidx = {k: i for i, k in enumerate(have)}
    Fg, tok_idx, tok_role, tok_mask = build_context(H, have)
    seeds = [0, 1, 2]; all_res = []
    for s in seeds:
        print(f"\n{'=' * 66}\n[SPLIT SEED {s}]")
        res, ntr, nte = run(s, H, have, Y, hidx, Fg, tok_idx, tok_role, tok_mask)
        for k, v in res.items():
            print(f"    {k:38s} {v:.3f}")
        all_res.append(res)
    keys = list(all_res[0].keys())
    agg = {k: (float(np.mean([r[k] for r in all_res])), float(np.std([r[k] for r in all_res]))) for k in keys}
    print(f"\n{'=' * 66}\nSTABILITY across {len(seeds)} splits (mean +/- sd, tide-removed specific recall@50):")
    for k in keys:
        print(f"    {k:38s} {agg[k][0]:.3f} +/- {agg[k][1]:.3f}")
    T = "TRANSFORMER (self+neighbourhood)"; Mb = "MLP-self (feats, no neighbourhood)"; Rp = "TRANSFORMER random-partners (ctrl)"
    C = "classical raw-cosine"; So = "TRANSFORMER self-only (infer)"; Tid = "TIDE-null"; Orc = "ORACLE* (ceiling)"
    d_tm = [r[T] - r[Mb] for r in all_res]; d_ts = [r[T] - r[So] for r in all_res]; d_tc = [r[T] - r[C] for r in all_res]
    d_tr = [r[T] - r[Rp] for r in all_res]                # REAL vs RANDOM partners = the decisive control
    print(f"\n    TRANSFORMER - MLP-self (neighbourhood value): {np.mean(d_tm):+.3f} +/- {np.std(d_tm):.3f}  ({[round(x,3) for x in d_tm]})")
    print(f"    TRANSFORMER - RANDOM-partners (structure!)  : {np.mean(d_tr):+.3f} +/- {np.std(d_tr):.3f}  ({[round(x,3) for x in d_tr]})")
    print(f"    TRANSFORMER - self-only  (attention value)  : {np.mean(d_ts):+.3f} +/- {np.std(d_ts):.3f}  ({[round(x,3) for x in d_ts]})")
    print(f"    TRANSFORMER - classical raw-cosine          : {np.mean(d_tc):+.3f} +/- {np.std(d_tc):.3f}  ({[round(x,3) for x in d_tc]})")
    tf_m = agg[T][0]; mlp_m = agg[Mb][0]; rp_m = agg[Rp][0]; tide_m = agg[Tid][0]; orc_m = agg[Orc][0]
    shuf_m = agg["TRANSFORMER shuffle (control)"][0]
    neigh_helps = np.mean(d_tm) > 0.02 and all(x > 0 for x in d_tm)
    structure_real = np.mean(d_tr) > 0.02 and all(x > 0 for x in d_tr)   # real partners beat random partners on every split
    shuffle_ok = shuf_m < tf_m - 0.02
    headroom = (tf_m - tide_m) / max(orc_m - tide_m, 1e-9)
    verdict = (
        "TRANSFORMER for the KO far-field (transformer_ko.py): a knockout is a SET OF TOKENS -- the perturbed gene + its post-KO partners "
        "(DepMap co-dependency, complex co-members, PPI neighbours), each token carrying [DepMap profile | our network-SVD embedding | "
        "ESSENTIALITY | log-degree | role]; a 2-layer self-attention encoder pools them into a KO embedding, trained as a LEARNED-RETRIEVAL "
        "metric (cosine ~ tide-removed response similarity), scored on held-out K562 across 3 splits (specific-mover recall@50). SCORES "
        f"(mean+/-sd): TIDE-null {tide_m:.3f} | classical raw-cosine {agg[C][0]:.3f} | MLP-self (same features, NO neighbourhood) {mlp_m:.3f} "
        f"| TRANSFORMER self-only {agg[So][0]:.3f} | TRANSFORMER (self+neighbourhood) {tf_m:.3f} | ORACLE* {orc_m:.3f}. "
        f"THE HONEST BAR (does attending over the post-KO interaction neighbourhood + essentiality beat the plain MLP on the gene's OWN "
        f"features?): TRANSFORMER - MLP-self = {np.mean(d_tm):+.3f}+/-{np.std(d_tm):.3f} "
        + ("(POSITIVE on every split) -- YES: the neighbourhood tokens (the 'other PPI after knockout') + essentiality carry response signal "
           "the perturbed gene's own DepMap profile does not, and attention extracts it; the transformer earns its place."
           if neigh_helps else
           "-- NO/marginal: the neighbourhood does not beat the plain MLP by a stable margin (representation, not architecture, is the "
           "bottleneck, as in neural_ko).") + " "
        + f"THE DECISIVE CONTROL (real vs RANDOM partners -- same token count/roles, random genes): TRANSFORMER - random-partners = "
        f"{np.mean(d_tr):+.3f}+/-{np.std(d_tr):.3f} "
        + (f"(POSITIVE every split; random-partner transformer only {rp_m:.3f}) -- so the gain is the REAL interaction structure (the actual "
           "co-dependency/complex/PPI partners a knockout touches), NOT extra tokens or capacity. The 'other PPI after knockout' is genuinely "
           "informative about the transcriptional response."
           if structure_real else
           f"-- random partners do about as well ({rp_m:.3f}), so the lift is token-count/capacity/regularisation, NOT the specific "
           "interaction structure; do not credit the biology.") + " "
        + (f"GENE-IDENTITY control PASSES (wrong-KO {shuf_m:.3f} < transformer)." if shuffle_ok else
           f"GENE-IDENTITY control WARNING (wrong-KO {shuf_m:.3f} ~ transformer).") +
        f" Head-room used {headroom:.0%} of tide->oracle; gap to oracle ({orc_m - tf_m:.3f}) stays representation/data-limited. "
        + ("BOTTOM LINE: the transformer over the knockout's interaction context is the best model built -- and the real-vs-random-partner "
           "control confirms the win comes from the actual PPI/co-dependency structure the user asked to include, not architecture alone."
           if (neigh_helps and structure_real and shuffle_ok) else
           "BOTTOM LINE: read the controls above before crediting the headline.") +
        " Single cell type, steady-state mRNA, magnitude target, CPU; deterministic per split.")
    print(f"\nVERDICT: {verdict}")
    out = {"mean_sd": {k: [round(agg[k][0], 4), round(agg[k][1], 4)] for k in keys},
           "transformer_minus_mlpself": [round(x, 4) for x in d_tm], "transformer_minus_selfonly": [round(x, 4) for x in d_ts],
           "transformer_minus_classical": [round(x, 4) for x in d_tc], "transformer_minus_randompartners": [round(x, 4) for x in d_tr],
           "neighbourhood_helps_all_splits": bool(neigh_helps), "structure_beats_random_all_splits": bool(structure_real),
           "shuffle_control_passes": bool(shuffle_ok), "headroom_used": round(float(headroom), 3),
           "verdict": verdict, "note": verdict}
    json.dump(out, open(OUT / "transformer_ko.json", "w"), indent=1)
    print("\n  -> outputs/orphan/transformer_ko.json")


if __name__ == "__main__":
    main()
