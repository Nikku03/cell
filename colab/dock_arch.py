"""ARCHITECTURE A/B: does the ADRN-grade transformer beat the dMaSIF GeoConv on the SAME structural task?

WHY THIS IS THE RIGHT RESPONSE TO A MEASURED FAILURE.  `dock_rescore` and `dock_poserank` both found that an
UNTRAINED network ranked poses as well as a trained one. That is the signature of a model too weak to extract
anything from its input -- and the model in question is small even by 2015 standards:

    dMaSIF GeoConv (the NEXUS/structural line)   edge MLP + learned Gaussian width + self MLP, 2 layers,
                                                 32 dims, 3 Linear layers, ~5k parameters. No attention, no
                                                 LayerNorm, no residual stream. Each point sees only its K=16
                                                 nearest neighbours, ever.
    Cellformer (the ADRN line)                   MultiheadAttention(4 heads), 2 LayerNorms, GELU feed-forward
                                                 at 2x expansion, typed-relation Embedding with a learned
                                                 per-relation attention bias, 16 Linear layers, several heads.

Two separate lineages in one codebase, and the structural side never got the architecture the perturbation side
has. This ports it across and measures the difference with everything else held fixed.

THE TASK, chosen because it is the cleanest place to see an architecture effect.  Interface point-pair
discrimination in the UNBOUND setting -- each chain surfaced from its own atoms, its own APBS field, its own
graph -- which `dock_rescore` measured at AUC 0.714 for GeoConv. Not pose ranking: today's evidence says the
information there may not be in the local contacts at all, so a bigger net on the same features would be
testing the wrong hypothesis. Interface discrimination has a known non-trivial baseline and room above it.

WHAT ATTENTION COULD BUY, stated as a mechanism rather than a hope. GeoConv aggregates over K=16 nearest
neighbours and nothing else, so a surface point can never see the far side of its own patch. Self-attention
over all points in a chain lets a point condition on the whole surface -- which is what "is this a binding
site" plausibly requires, since binding sites are defined by contrast against the rest of the protein.

ARMS, all on identical data, identical splits, identical epoch budget:
    geoconv                the existing architecture. The baseline being challenged.
    geoconv_untrained      random init. This is the control that MATTERED today -- it matched the trained net
                           on pose ranking, and if it matches here too the task is not being learned at all.
    transformer            Cellformer-style: point tokens, MultiheadAttention, LayerNorm, GELU FF, residual.
    transformer_untrained  random init, same architecture.

Parameter counts are printed, because "the bigger model won" is only interesting alongside how much bigger.

FAIRNESS.  Both architectures see the SAME subsampled point set (attention is O(n^2), so both are capped at
N_PTS points and the GeoConv KNN graph is rebuilt on that subsample rather than the full cloud). Same features,
same pairs, same optimiser, same epochs, same seed.

PREDECLARED, before any number:
    transformer beats geoconv AND beats its own untrained control
                        -> architecture WAS the limit on the structural side, and the whole line has been
                           running underpowered. That would also reopen pose ranking, which was abandoned on
                           evidence gathered with the weak net.
    transformer ~ geoconv, both above untrained
                        -> architecture is not the limit; both learn the same thing and the ceiling is the
                           features or the data volume.
    transformer ~ its own untrained control
                        -> capacity does not help because the task is not learnable from these 5 per-point
                           features, which is a statement about the INPUT, not the model.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import dock_fft as DF
import dock_poserank as PRK
import dock_rescore as RS
import flex_physics as fp

OUT = A.OUT
N_TRAIN, N_EVAL = 40, 14
EPOCHS = 25
N_PTS = 512          # attention is O(n^2); BOTH architectures are capped here so the comparison is fair
D_MODEL = 64
DIST_SCALE = 10.0    # Angstrom; raw distances saturate the softmax, see PointFormer
N_HEADS = 4
SEED = 0


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(4)

    report("=" * 100)
    report("ARCHITECTURE A/B -- ADRN-grade transformer vs the dMaSIF GeoConv, same task, same data")
    report("=" * 100)
    report("  Two lineages in one codebase: the ADRN side has MultiheadAttention, LayerNorm, GELU and typed")
    report("  relation embeddings; the structural side has a 2-layer GeoConv with ~5k parameters and a K=16")
    report("  neighbourhood. An UNTRAINED GeoConv matched a trained one on pose ranking today -- the signature")
    report("  of a model too weak to learn. This ports the architecture across and measures the difference.")
    report(f"  Both capped at {N_PTS} points so attention is affordable and the comparison is fair.")
    report("  PREDECLARED: transformer beats geoconv AND its own untrained control => architecture was the")
    report("  limit. transformer ~ geoconv => it is not. transformer ~ untrained => the INPUT is the limit.")

    # ---------------- data: unbound per-chain surfaces, cached ----------------
    dockj = json.load(open(OUT / "dock_fft.json"))
    test_pdbs = set(r["pdb"] for r in dockj["complexes"])
    have = sorted(f[:-4] for f in os.listdir(fp.PDBDIR) if f.endswith(".pdb") and "__" not in f)
    rng = np.random.default_rng(SEED)

    report(f"\n  loading cached unbound surfaces...")
    pool, tb = {}, time.time()
    for pdb in have:
        if len(pool) >= N_TRAIN + N_EVAL:
            break
        p = DF.prepare(pdb)
        if p is None or not p["sane"]:
            continue
        chs, ok = {}, True
        for ch in (p["ca"], p["cb"]):
            tag = RS.chain_pdb(pdb, ch)
            r = PRK.cached_surface(tag) if tag else None
            if r is None:
                ok = False
                break
            chs[ch] = r[0]
        if not ok:
            continue
        A_, B_ = chs[p["ca"]], chs[p["cb"]]
        # subsample BOTH chains to N_PTS, then rebuild everything on that subsample
        ia = rng.choice(len(A_["pts"]), min(N_PTS, len(A_["pts"])), replace=False)
        ib = rng.choice(len(B_["pts"]), min(N_PTS, len(B_["pts"])), replace=False)
        sa = {k: A_[k][ia] for k in ("pts", "nrm", "feat")}
        sb = {k: B_[k][ib] for k in ("pts", "nrm", "feat")}
        pos, neg = RS.cross_pairs(sa["pts"], sb["pts"])
        if len(pos) < 12 or len(neg) < 12:
            continue
        import dmasif as D
        pool[pdb] = {"a": sa, "b": sb, "ga": D.precompute_graph(sa), "gb": D.precompute_graph(sb),
                     "pos": np.array(pos), "neg": np.array(neg)}
    report(f"    {len(pool)} complexes in {time.time()-tb:.0f}s "
           f"({N_PTS} points/chain, KNN graph rebuilt on the subsample)")
    names = sorted(pool)
    ev = [n for n in names if n in test_pdbs][:N_EVAL]
    ev += [n for n in names if n not in test_pdbs and n not in ev][:max(0, N_EVAL - len(ev))]
    tr = [n for n in names if n not in set(ev)][:N_TRAIN]
    if len(tr) < 12 or len(ev) < 5:
        report(f"  not enough: train {len(tr)} eval {len(ev)}")
        return 1
    report(f"    train {len(tr)} | eval {len(ev)} (disjoint)")
    FD = pool[tr[0]]["a"]["feat"].shape[1]

    # ---------------- architecture 1: the existing GeoConv ----------------
    class GeoConv(nn.Module):
        def __init__(s, din, dout):
            super().__init__()
            s.edge = nn.Sequential(nn.Linear(4 + din, dout), nn.ReLU())
            s.logsig = nn.Parameter(torch.tensor(1.0))
            s.self = nn.Sequential(nn.Linear(din + dout, dout), nn.ReLU())

        def forward(s, feat, idx, lc, dist):
            e = s.edge(torch.cat([lc, dist.unsqueeze(-1), feat[idx]], -1))
            w = torch.exp(-(dist ** 2) / (torch.exp(s.logsig) ** 2 + 1e-6)).unsqueeze(-1)
            return s.self(torch.cat([feat, (w * e).sum(1) / (w.sum(1) + 1e-6)], -1))

    class GeoNet(nn.Module):
        def __init__(s, fd):
            super().__init__()
            s.c1, s.c2, s.out = GeoConv(fd, 32), GeoConv(32, 32), nn.Linear(32, 32)

        def forward(s, d, g):
            f = torch.tensor(d["feat"]); idx = torch.tensor(g["idx"])
            lc = torch.tensor(g["lc"]); di = torch.tensor(g["dist"])
            e = s.out(s.c2(s.c1(f, idx, lc, di), idx, lc, di))
            return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    # ---------------- architecture 2: Cellformer-style transformer over surface points ----------------
    class Block(nn.Module):
        """one Cellformer block: pre-norm attention + GELU feed-forward at 2x, both residual"""
        def __init__(s, d, h):
            super().__init__()
            s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            s.attn = nn.MultiheadAttention(d, h, batch_first=True)
            s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        def forward(s, h, bias):
            x = s.ln1(h)
            a, _ = s.attn(x, x, x, attn_mask=bias, need_weights=False)
            h = h + a
            return h + s.ff(s.ln2(h))

    class PointFormer(nn.Module):
        """surface points as tokens. Every point attends to EVERY other point on its own chain, which the
        K=16 GeoConv neighbourhood can never do. Geometry enters as a learned distance bias on attention --
        the structural analogue of Cellformer's typed-relation bias."""
        def __init__(s, fd, d=D_MODEL, h=N_HEADS, nblocks=2):
            super().__init__()
            # NODE INPUTS MUST MATCH GeoConv's EXACTLY or this is not an architecture comparison. The first
            # version also fed the surface NORMAL, which GeoConv does not receive as a node feature (it sees
            # geometry only through local-frame neighbour coordinates). Contacting surfaces face each other, so
            # their normals are anti-parallel: measured -0.465 for contacting pairs vs +0.222 for far pairs.
            # At init the embedding was largely linear in the normal, contacting pairs anti-correlated, and the
            # UNTRAINED transformer scored 0.3897 -- below chance, which both depressed its floor and inflated
            # its apparent gain over that floor. Same 5 node features for both now; geometry reaches each
            # architecture through its own native channel (local frames for GeoConv, distance bias here).
            s.proj = nn.Linear(fd, d)
            # LEARNED RELATIVE-POSITION BIAS, and it must start at ZERO. PyTorch ADDS a float attn_mask to the
            # attention logits before softmax. The first version fed raw Angstrom distances (0-155 A) into an
            # unnormalised MLP, giving bias logits from -47 to +20 against attention logits of O(1): max
            # attention weight 1.000 at initialisation, i.e. every point attending to exactly one other point
            # through a saturated softmax that barely passes gradient. The transformer scored 0.379 UNTRAINED,
            # far below chance, which is what exposed it. Distances are now scaled by DIST_SCALE and the output
            # layer is zero-initialised, so attention begins unbiased and learns the bias -- the standard
            # treatment for relative-position bias.
            s.dist_bias = nn.Sequential(nn.Linear(1, 4 * h), nn.GELU(), nn.Linear(4 * h, h))
            nn.init.zeros_(s.dist_bias[-1].weight)
            nn.init.zeros_(s.dist_bias[-1].bias)
            s.blocks = nn.ModuleList([Block(d, h) for _ in range(nblocks)])
            s.out = nn.Linear(d, 32)
            s.h = h

        def forward(s, d_, g_):
            P = torch.tensor(d_["pts"])
            h = s.proj(torch.tensor(d_["feat"])).unsqueeze(0)
            D2 = (torch.cdist(P, P) / DIST_SCALE).unsqueeze(-1)
            bias = s.dist_bias(D2).permute(2, 0, 1)          # (heads, n, n), exactly 0 at init
            for b in s.blocks:
                h = b(h, bias)
            e = s.out(h.squeeze(0))
            return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    def train_eval(cls, trained, seed=SEED, tag=""):
        torch.manual_seed(seed)
        net = cls(FD)
        npar = sum(p.numel() for p in net.parameters())
        scale = nn.Parameter(torch.tensor(4.0))
        if trained:
            opt = torch.optim.Adam(list(net.parameters()) + [scale], lr=3e-3)
            lf = nn.BCEWithLogitsLoss()
            order = list(tr)
            t1 = time.time()
            for _ in range(EPOCHS):
                np.random.shuffle(order)
                net.train()
                for pdb in order:
                    c = pool[pdb]
                    opt.zero_grad()
                    ea = net(c["a"], c["ga"]); eb = net(c["b"], c["gb"])
                    sp = (ea[c["pos"][:, 0]] * eb[c["pos"][:, 1]]).sum(1) * scale
                    sn = (ea[c["neg"][:, 0]] * eb[c["neg"][:, 1]]).sum(1) * scale
                    loss = lf(torch.cat([sp, sn]),
                              torch.cat([torch.ones(len(sp)), torch.zeros(len(sn))]))
                    loss.backward()
                    opt.step()
            secs = time.time() - t1
        else:
            secs = 0.0
        net.eval()
        ys, ss = [], []
        with torch.no_grad():
            for pdb in ev:
                c = pool[pdb]
                ea = net(c["a"], c["ga"]); eb = net(c["b"], c["gb"])
                sp = (ea[c["pos"][:, 0]] * eb[c["pos"][:, 1]]).sum(1) * scale
                sn = (ea[c["neg"][:, 0]] * eb[c["neg"][:, 1]]).sum(1) * scale
                ss += sp.tolist() + sn.tolist()
                ys += [1] * len(sp) + [0] * len(sn)
        auc = float(roc_auc_score(ys, ss)) if len(set(ys)) > 1 else float("nan")
        return {"auc": auc, "params": npar, "secs": secs}

    report(f"\n  TRAINING (identical data, splits, epochs={EPOCHS}, seed={SEED})")
    res = {}
    for nm, cls, trained in (("geoconv", GeoNet, True), ("geoconv_untrained", GeoNet, False),
                             ("transformer", PointFormer, True), ("transformer_untrained", PointFormer, False)):
        r = train_eval(cls, trained, tag=nm)
        res[nm] = r
        report(f"    {nm:<24} AUC {r['auc']:.4f}   {r['params']:>7,} params   {r['secs']:>5.0f}s")

    report(f"\n  {'arm':<24}{'AUC':>9}{'params':>10}{'vs untrained':>15}")
    for nm in ("geoconv", "transformer"):
        d = res[nm]["auc"] - res[nm + "_untrained"]["auc"]
        report(f"  {nm:<24}{res[nm]['auc']:>9.4f}{res[nm]['params']:>10,}{d:>+15.4f}")
    for nm in ("geoconv_untrained", "transformer_untrained"):
        report(f"  {nm:<24}{res[nm]['auc']:>9.4f}{res[nm]['params']:>10,}{'--':>15}")

    gt, tt = res["geoconv"]["auc"], res["transformer"]["auc"]
    gu, tu = res["geoconv_untrained"]["auc"], res["transformer_untrained"]["auc"]
    ratio = res["transformer"]["params"] / max(res["geoconv"]["params"], 1)

    report("\n  READING")
    if tt > gt + 0.02 and tt > tu + 0.02:
        report(f"  ARCHITECTURE WAS THE LIMIT. The transformer reaches {tt:.4f} against GeoConv's {gt:.4f},")
        report(f"  and beats its own untrained control ({tu:.4f}) by {tt-tu:+.4f}. {ratio:.0f}x the parameters")
        report("  buys real capability on the structural side, which has been running an architecture the ADRN")
        report("  side left behind years ago. Pose ranking was abandoned on evidence from the weak net and")
        report("  deserves re-testing with this one.")
    elif tt > tu + 0.02 and gt > gu + 0.02 and abs(tt - gt) <= 0.02:
        report(f"  ARCHITECTURE IS NOT THE LIMIT. transformer {tt:.4f} vs geoconv {gt:.4f} -- within noise of")
        report(f"  each other, and both clearly above their untrained controls ({tu:.4f}, {gu:.4f}). Both learn")
        report(f"  the same thing despite a {ratio:.0f}x parameter gap. The ceiling is the features or the data")
        report("  volume, not the model.")
    elif tt <= tu + 0.02:
        report(f"  CAPACITY DOES NOT HELP. transformer {tt:.4f} vs its own untrained {tu:.4f} -- {ratio:.0f}x the")
        report("  parameters and full self-attention learn no more than random weights. That is a statement")
        report(f"  about the INPUT: 5 per-point features (APBS, curvature, hydrophobicity, charge, H-bond)")
        report("  may not contain what distinguishes an interface point, however the model is built.")
    else:
        report(f"  MIXED. transformer {tt:.4f} (untrained {tu:.4f}), geoconv {gt:.4f} (untrained {gu:.4f}).")
        report("  Read the per-arm deltas above rather than a single verdict.")

    json.dump({"test": "dock_arch", "n_train": len(tr), "n_eval": len(ev), "n_pts": N_PTS,
               "epochs": EPOCHS, "d_model": D_MODEL, "n_heads": N_HEADS, "arms": res,
               "param_ratio": ratio, "train_pdbs": tr, "eval_pdbs": ev, "log": log},
              open(OUT / "dock_arch.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'dock_arch.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
