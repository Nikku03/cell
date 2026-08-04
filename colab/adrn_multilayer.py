"""MULTIPLEX vs FLAT: does organising the features into LAYERS earn its complexity?

THE PROPOSAL BEING TESTED.  Whole-cell data is argued to need a multi-layered architecture -- epigenomic,
regulatory, signalling, metabolic, spatial -- as a stack of interdependent layers with cross-talk between them,
rather than one flattened canvas. The claim has a sharp, testable consequence: **cross-layer structure must beat
concatenation.** Concatenation IS the flattened canvas. If a model that can attend between layers does no better
than one handed the same numbers in a single vector, the layering is a description of the data rather than a
property of it.

WHY THIS IS NOT A RERUN.  Every layer here has been tested ALONE and lost:

    epigenomic/regulatory   ChIP binding -> which genes move: AUC 0.5068 vs 0.4981 permuted, p = 0.22
    signalling / topology   8 tests and a 36-cell sweep, all killed by degree-matched rewiring
    metabolic               FBA cascade, honest negative
    structural              iface block NULL; ESM-2 fails G1 and G2
    spatial                 no cryo-ET or compartment counts exist for K562 on this disk

The untested claim is that they COMPOSE despite none working alone -- and Norman made that newly plausible today
by showing that measured perturbation effects add (+0.4326 over scrambled, ROBUST 6/6).

THE LAYERS are cut from real annotation provenance, not invented. Every channel already carries a source prefix,
so the partition is a fact about where each number came from:

    spatial      compartment: , go:C:      where the protein is. The nearest thing we have to a spatial axis.
    structure    pfam: , domain:           what it is built from.
    pathway      pathway: , go:P:          which process it belongs to.
    molfunc      go:F:                     what it does chemically.
    complex      complex:                  what it is bound into.
    network      lesion: and the reaction-network structural columns (catalysis, participation, partners).
    genomic      chrom: and the remaining constraint / study-depth / regulatory-architecture channels.
    response     the gene's COLUMN -- how it responds when OTHER genes are perturbed. The only measured layer,
                 and the only one available for a gene nobody has ever perturbed.

ARMS
    ridge_single_<layer>   each layer alone, through the incumbent ridge. Which layers carry anything.
    ridge_all              every layer concatenated. The incumbent, and the FLAT canvas.
    ridge_drop_<layer>     leave-one-layer-out. What each layer adds that the others do not already have.
    tf_flat                a transformer-free MLP on the concatenation.
    tf_multiplex           layers as TOKENS with attention across them. The architecture under test.

PREDECLARED, before any number:
    G1  tf_multiplex > tf_flat AT MATCHED PARAMETER COUNT   -> cross-layer structure earns its complexity.
                                                              Unmatched parameters would make this meaningless,
                                                              so both counts are printed and asserted close.
    G2  ridge_all > best ridge_single                       -> layers add rather than duplicate each other.
    G3  every layer beats its own PERMUTED control          -> any layer failing this is decorative, and is
                                                              reported as decorative rather than quietly carried.

Judged on robustness.Sweeper over basis rank x seed, scored on the two sealed 200-knockout cohorts with
precision@20 -- the same answer key as every other number in this project.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
from robustness import Sweeper

OUT, SP = A.OUT, A.SP
K_VALUES = (30, 60)
SEEDS = (0, 1, 2)
PENALTIES = (1.0, 10.0, 100.0, 1000.0)
COL_SVD, D_MODEL, N_HEADS, N_BLOCKS, EPOCHS = 128, 96, 4, 2, 300


def layer_of(name):
    """Every channel's layer is decided by the prefix it was BUILT with, so the partition is provenance."""
    n = name.lower()
    if n.startswith("compartment:") or n.startswith("go:c:"):
        return "spatial"
    if n.startswith("pfam:") or n.startswith("domain:"):
        return "structure"
    if n.startswith("pathway:") or n.startswith("go:p:"):
        return "pathway"
    if n.startswith("go:f:"):
        return "molfunc"
    if n.startswith("complex:"):
        return "complex"
    if n.startswith("lesion:") or n.startswith("catal") or n.startswith("particip") or n.startswith("partner"):
        return "network"
    return "genomic"


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("MULTIPLEX vs FLAT: does layering the features earn its complexity?")
    report("=" * 100)
    report("  PREDECLARED: G1 multiplex > flat at MATCHED params. G2 all > best single. G3 each layer > its perm.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    gj = {g: j for j, g in enumerate(genes)}
    M = z["M"].astype(np.float32)
    Aabs = np.abs(M)
    tide = (Aabs >= A.TAU).mean(0) >= 0.05

    sealed, answers = set(), {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists():
            answers[tag or "cohort1"] = json.loads(af.read_text())
    train = [k for k in kos if k not in sealed]
    train_rows = np.array([ki[k] for k in train])
    universe = sorted(set(kos) | sealed)
    urow = {g: i for i, g in enumerate(universe)}
    tr_u = np.array([urow[k] for k in train])
    report(f"\n  {len(kos):,} perturbations x {len(genes):,} genes; sealed {len(sealed)}; train {len(train):,}")

    base, base_names = A.build_channels(universe, set(train), report)
    extra, extra_names, tb = C.extra_channels(universe, set(train), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    names = list(base_names) + list(extra_names[:tb])
    assert len(names) == chan.shape[1], f"{len(names)} names for {chan.shape[1]} channels"

    blocks = {}
    for j, nm in enumerate(names):
        blocks.setdefault(layer_of(nm), []).append(j)

    # the RESPONSE layer: the gene's column, built from training rows only (no sealed answer can enter it),
    # diagonal dropped, rotation fitted on training genes and sealed genes projected onto it
    Ctrain = M[train_rows]
    rawcol = np.zeros((len(universe), Ctrain.shape[0]), np.float32)
    for g in universe:
        j = gj.get(g)
        if j is None:
            continue
        v = Ctrain[:, j].copy()
        r = ki.get(g)
        if r is not None:
            v[train_rows == r] = 0.0
        rawcol[urow[g]] = v
    mu = rawcol[tr_u].mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(rawcol[tr_u] - mu, full_matrices=False)
    colf = ((rawcol - mu) @ Vt[:COL_SVD].T).astype(np.float32)

    LAY = {}
    for nm, idx in sorted(blocks.items()):
        B = chan[:, idx].astype(np.float32)
        LAY[nm] = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    LAY["response"] = colf / (np.linalg.norm(colf, axis=1, keepdims=True) + 1e-9)
    order = sorted(LAY)
    report("\n  LAYERS")
    for nm in order:
        report(f"    {nm:<10} {LAY[nm].shape[1]:>4} dims")

    from sklearn.decomposition import NMF
    import torch
    import torch.nn as nn
    torch.set_num_threads(4)

    def sealed_score(pred_by_gene, H):
        out = {}
        for tag, ans in answers.items():
            hits = []
            for k in sorted(ans):
                if k not in pred_by_gene:
                    continue
                s = (pred_by_gene[k] @ H).copy()
                s[tide] = -np.inf
                if k in gj:
                    s[gj[k]] = -np.inf
                top = np.argsort(-s)[:A.NPRED]
                hits.append(len({genes[t] for t in top} & set(ans[k]["movers"])) / float(A.NPRED))
            out[tag] = np.array(hits)
        return out

    def ridge_arm(F, W, H):
        best_v, best_p, rv = -1e18, PENALTIES[0], np.random.default_rng(A.SEED + 2).permutation(len(tr_u))
        nv = max(50, len(rv) // 5)
        for pen in PENALTIES:                      # penalty chosen on a TRAIN-ONLY split, never on the cohorts
            w = A.ridge_fit(F[tr_u[rv[nv:]]], W[rv[nv:]], pen)
            e = -float(np.mean((A.ridge_apply(F[tr_u[rv[:nv]]], w) - W[rv[:nv]]) ** 2))
            if e > best_v:
                best_v, best_p = e, pen
        w = A.ridge_fit(F[tr_u], W, best_p)
        keys = [k for ans in answers.values() for k in ans]
        return sealed_score({k: A.ridge_apply(F[[urow[k]]], w)[0] for k in keys if k in urow}, H)

    class Multiplex(nn.Module):
        """Layers as TOKENS, with attention between them. The architecture under test."""
        def __init__(self, dims, k):
            super().__init__()
            self.proj = nn.ModuleList([nn.Linear(d, D_MODEL) for d in dims])
            enc = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_MODEL * 2, 0.1,
                                             batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(enc, N_BLOCKS)
            self.head = nn.Linear(D_MODEL, k)

        def forward(self, xs):
            t = torch.stack([p(x) for p, x in zip(self.proj, xs)], 1)
            return self.head(self.enc(t).mean(1))

    class Flat(nn.Module):
        """The same numbers in ONE vector. This is the flattened canvas the multiplex claim argues against."""
        def __init__(self, total, k, hidden):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(total, hidden), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(hidden, k))

        def forward(self, xs):
            return self.net(torch.cat(xs, 1))

    def train_torch(model, Xs, Y, seed):
        torch.manual_seed(seed)
        rv = np.random.default_rng(A.SEED + seed).permutation(len(tr_u))
        nv = max(100, len(rv) // 5)
        tr, va = rv[nv:], rv[:nv]
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        Yt = torch.tensor(Y)
        best, bad, state = 1e18, 0, None
        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(len(tr))
            for i in range(0, len(tr), 256):
                b = tr[perm[i:i + 256].numpy()]
                opt.zero_grad()
                loss = ((model([x[b] for x in Xs]) - Yt[b]) ** 2).mean()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = float(((model([x[va] for x in Xs]) - Yt[va]) ** 2).mean())
            if v < best - 1e-6:
                best, bad = v, 0
                state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 30:
                    break
        model.load_state_dict(state)
        return model

    sw1 = Sweeper("multiplex - flat (G1)", axes={"K": list(K_VALUES), "seed": list(SEEDS)})
    sw2 = Sweeper("ridge_all - best single layer (G2)", axes={"K": list(K_VALUES), "seed": list(SEEDS)})
    detail = {}
    for K in K_VALUES:
        X = Aabs[train_rows].copy()
        X[:, tide] = 0.0
        nmf = NMF(n_components=K, init="nndsvda", max_iter=300, random_state=0)
        W = nmf.fit_transform(X).astype(np.float32)
        H = nmf.components_.astype(np.float32)
        del X

        singles, perms = {}, {}
        for nm in order:
            singles[nm] = ridge_arm(LAY[nm], W, H)
            P = LAY[nm][np.random.default_rng(A.SEED + 11 + len(nm)).permutation(len(LAY[nm]))]
            perms[nm] = ridge_arm(P, W, H)
        allF = np.concatenate([LAY[nm] for nm in order], 1)
        r_all = ridge_arm(allF, W, H)
        drops = {nm: ridge_arm(np.concatenate([LAY[m] for m in order if m != nm], 1), W, H) for nm in order}

        def mn(d):
            return float(np.mean([v.mean() for v in d.values()]))
        report(f"\n  K={K}  ridge_all {mn(r_all):.4f}")
        report(f"    {'layer':<10} {'alone':>8} {'permuted':>9} {'drop-it':>9} {'G3':>6}")
        for nm in order:
            g3 = "ok" if mn(singles[nm]) > mn(perms[nm]) else "DECOR"
            report(f"    {nm:<10} {mn(singles[nm]):>8.4f} {mn(perms[nm]):>9.4f} {mn(drops[nm]):>9.4f} {g3:>6}")
        best_single = max(order, key=lambda n: mn(singles[n]))

        Xs = [torch.tensor(LAY[nm][tr_u]) for nm in order]
        dims = [LAY[nm].shape[1] for nm in order]
        seal_u = {k: urow[k] for ans in answers.values() for k in ans if k in urow}
        Xs_seal = [torch.tensor(LAY[nm][list(seal_u.values())]) for nm in order]
        for seed in SEEDS:
            mx = Multiplex(dims, K)
            npar = sum(p.numel() for p in mx.parameters())
            hid = 1
            while True:
                f = Flat(sum(dims), K, hid)
                if sum(p.numel() for p in f.parameters()) >= npar or hid > 4096:
                    break
                hid += 8
            nflat = sum(p.numel() for p in f.parameters())
            assert abs(nflat - npar) / npar < 0.10, \
                f"parameter counts not matched: multiplex {npar:,} vs flat {nflat:,} -- G1 would be meaningless"
            res = {}
            for nm_, mdl in (("tf_multiplex", mx), ("tf_flat", f)):
                mdl = train_torch(mdl, Xs, W, seed)
                mdl.eval()
                with torch.no_grad():
                    P = mdl(Xs_seal).numpy()
                res[nm_] = sealed_score({k: P[i] for i, k in enumerate(seal_u)}, H)
            cfg, tg = {"K": K, "seed": seed}, f"K{K}|s{seed}"
            sw1.add(cfg, tg, np.concatenate([res["tf_multiplex"][t] - res["tf_flat"][t] for t in answers]))
            sw2.add(cfg, tg, np.concatenate([r_all[t] - singles[best_single][t] for t in answers]))
            detail[tg] = {"ridge_all": mn(r_all), "best_single": best_single,
                          "best_single_score": mn(singles[best_single]),
                          "tf_multiplex": mn(res["tf_multiplex"]), "tf_flat": mn(res["tf_flat"]),
                          "params_multiplex": npar, "params_flat": nflat,
                          "singles": {n: mn(singles[n]) for n in order},
                          "perms": {n: mn(perms[n]) for n in order},
                          "drops": {n: mn(drops[n]) for n in order}}
            report(f"    seed{seed}: multiplex {mn(res['tf_multiplex']):.4f} ({npar:,} par) | "
                   f"flat {mn(res['tf_flat']):.4f} ({nflat:,} par) | ridge_all {mn(r_all):.4f}")

    for nm, s in (("G1", sw1), ("G2", sw2)):
        report(s.report())
        if s.inert_axes() or s.duplicate_cells():
            report(f"  GUARD {nm}: inert axes {s.inert_axes()}, duplicate cells {s.duplicate_cells()}")

    decor = sorted({n for d in detail.values() for n in order if d["singles"][n] <= d["perms"][n]})
    g1 = float(np.mean([d["tf_multiplex"] - d["tf_flat"] for d in detail.values()]))
    g2 = float(np.mean([d["ridge_all"] - d["best_single_score"] for d in detail.values()]))
    report("\n  READING")
    report(f"  G1 multiplex - flat      {g1:+.4f}")
    report(f"  G2 all - best single     {g2:+.4f}")
    report(f"  G3 decorative layers     {decor if decor else 'none -- every layer beat its own permutation'}")
    if g1 > 0 and sum(1 for c in sw1._cells.values() if c["lo"] > 0) > len(sw1._cells) / 2:
        report("  Cross-layer attention beats concatenation at matched parameters: the LAYERING is a property of")
        report("  the data, not just a way of describing it, and the multiplex architecture earns its complexity.")
    else:
        report("  Cross-layer attention does NOT beat concatenation at matched parameters. The layers are a")
        report("  useful account of where the numbers CAME FROM, but the model gains nothing from being told")
        report("  which layer a feature belongs to -- flattening loses nothing. The multiplex structure is")
        report("  documentation, not architecture.")

    json.dump({"test": "adrn_multilayer", "K_values": list(K_VALUES), "seeds": list(SEEDS),
               "layers": {n: int(LAY[n].shape[1]) for n in order}, "detail": detail,
               "G1_multiplex_minus_flat": g1, "G2_all_minus_best_single": g2, "G3_decorative": decor,
               "sweeps": {"G1": sw1.verdict(), "G2": sw2.verdict()}, "log": log},
              open(OUT / "adrn_multilayer.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'adrn_multilayer.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
