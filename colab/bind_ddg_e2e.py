"""bind_ddg_e2e — the end-to-end test of the PARTNER-AWARE geodesic surface encoder for ΔΔG-binding MAGNITUDE, at scale
and on GPU. Trains three readouts held-out BY COMPLEX and compares them:
  phys  — physics features (bind_ddg.features) through an MLP           (the baseline; ~0.46)
  surf  — the geodesic surface encoder ALONE, fine-tuned end-to-end     (can structure learn binding magnitude?)
  both  — physics + end-to-end surface                                   (does the structural context ADD to physics?)

The sandbox result (42 complexes) was a decisive NEGATIVE: surf alone ~0.10, both 0.33 < phys 0.42 — the WT surface is
mutation-blind (identical cloud for W304A vs W304G), so the encoder can only see the POSITION, not the substitution.
This module lets Colab confirm it on all ~345 SKEMPI complexes with a GPU. Needs dMaSIF clouds (nexus_train.build).
  run(cache, clouds_dir, epochs, device) -> outputs/orphan/bind_ddg_e2e.json
"""
import os, sys, glob, pickle, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "orphan")
SITE_R = 8.0


def build_samples(cache, clouds_dir):
    """assemble (pdb, site_point_indices, physics_vec, ddg, mut) for every SKEMPI mutation whose complex has a cloud."""
    import flex_physics as fp, bind_ddg
    fp.PDBDIR = cache
    clouds = {}
    for f in glob.glob(os.path.join(clouds_dir, "*.pkl")):
        clouds[os.path.splitext(os.path.basename(f))[0]] = pickle.load(open(f, "rb"))
    samples = []
    for r in fp.load_pdb_muts():
        if r["pdb"] not in clouds:
            continue
        t = fp.table(r["pdb"])
        if t is None:
            continue
        m = (t["ch"] == r["chain"]) & (t["rn"] == r["pos"]) & (t["nm"] == "CA")
        if not m.any():
            continue
        c = t["co"][m][0]; pc = clouds[r["pdb"]]["pc"]
        site = np.where((pc["ch"] == r["chain"]) & (np.linalg.norm(pc["pts"] - c, axis=1) < SITE_R))[0]
        if len(site) < 2:
            continue
        ph = bind_ddg.features(t, r["chain"], r["pos"], r["wt"], r["mut"])
        if ph is None:
            continue
        samples.append((r["pdb"], site, np.array(ph, np.float32), float(r["ddg"]), r["mut"]))
    return samples, clouds


def _make_net(mode, FD, PD, device):
    import torch, torch.nn as nn

    class GeoConv(nn.Module):
        def __init__(s, din, dout):
            super().__init__(); s.edge = nn.Sequential(nn.Linear(4 + din, dout), nn.ReLU())
            s.logsig = nn.Parameter(torch.tensor(1.0)); s.self = nn.Sequential(nn.Linear(din + dout, dout), nn.ReLU())
        def forward(s, feat, idx, lc, dist):
            fn = feat[idx]; e = s.edge(torch.cat([lc, dist.unsqueeze(-1), fn], -1))
            w = torch.exp(-(dist ** 2) / (torch.exp(s.logsig) ** 2 + 1e-6)).unsqueeze(-1)
            return s.self(torch.cat([feat, (w * e).sum(1) / (w.sum(1) + 1e-6)], -1))

    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.mode = mode
            if mode != "phys":
                s.c1 = GeoConv(FD, 32); s.c2 = GeoConv(32, 32)
            hin = (32 if mode != "phys" else 0) + (PD if mode != "surf" else 0)
            s.head = nn.Sequential(nn.Linear(hin, 32), nn.ReLU(), nn.Linear(32, 1))
        def encode(s, cl):
            pc, g = cl["pc"], cl["g"]
            f = torch.tensor(pc["feat"], device=device); idx = torch.tensor(g["idx"], device=device)
            lc = torch.tensor(g["lc"], device=device); dist = torch.tensor(g["dist"], device=device)
            return s.c2(s.c1(f, idx, lc, dist), idx, lc, dist)
        def forward(s, emb, sites, phys):
            import torch
            parts = []
            if s.mode != "phys":
                parts.append(torch.stack([emb[torch.tensor(si, device=device)].mean(0) for si in sites]))
            if s.mode != "surf":
                parts.append(phys)
            return s.head(torch.cat(parts, -1)).squeeze(-1)
    return Net().to(device)


def run(cache, clouds_dir=None, epochs=60, device="cpu"):
    import torch, torch.nn as nn
    from sklearn.model_selection import GroupKFold
    from scipy.stats import pearsonr
    torch.manual_seed(0)
    clouds_dir = clouds_dir or os.path.join(cache, "clouds")
    print(f"assembling samples (cache={cache}) ...", flush=True)
    t0 = time.time(); samples, clouds = build_samples(cache, clouds_dir)
    if not samples:
        print("no samples — build clouds first (nexus_train.build)."); return {"error": "no samples"}
    FD = clouds[samples[0][0]]["pc"]["feat"].shape[1]; PD = len(samples[0][2])
    Y = np.array([s[3] for s in samples]); ymu, ysd = Y.mean(), Y.std()
    grp = np.array([s[0] for s in samples]); muts = np.array([s[4] for s in samples])
    # ROBUST inputs: physics features are heavy-tailed (rotamer-clash / charge-field can be extreme) — a tree ignores
    # that but an MLP diverges on it. Clean NaN/inf here and quantile-normalise per fold below. Also clean surface feats.
    PH = np.nan_to_num(np.stack([s[2] for s in samples]).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    for cl in clouds.values():
        cl["pc"]["feat"] = np.nan_to_num(cl["pc"]["feat"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"{len(samples)} mutations / {len(set(grp))} complexes ({time.time()-t0:.0f}s) [device={device}]", flush=True)

    from sklearn.preprocessing import QuantileTransformer

    def run_mode(mode):
        preds = np.zeros(len(samples)); rng = np.random.default_rng(0)
        for tr, te in GroupKFold(5).split(samples, Y, grp):
            qt = QuantileTransformer(n_quantiles=min(1000, len(tr)), output_distribution="normal", random_state=0)
            PHt = qt.fit(PH[tr]).transform(PH).astype(np.float32)   # robust per-fold feature scaling (no label leak)
            by = {}
            for i in tr:
                by.setdefault(samples[i][0], []).append(i)
            order = list(by.keys())
            net = _make_net(mode, FD, PD, device)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4); lf = nn.MSELoss()
            for ep in range(epochs):
                rng.shuffle(order); net.train()                    # shuffle complex order each epoch (stability)
                for pdb in order:
                    idxs = by[pdb]
                    emb = net.encode(clouds[pdb]) if mode != "phys" else None
                    sites = [samples[i][1] for i in idxs]
                    phys = torch.tensor(PHt[idxs], device=device)
                    yb = torch.tensor([(Y[i] - ymu) / ysd for i in idxs], dtype=torch.float32, device=device)
                    opt.zero_grad(); lf(net(emb, sites, phys), yb).backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()   # grad clip (no divergence)
            net.eval()
            with torch.no_grad():
                byte = {}
                for i in te:
                    byte.setdefault(samples[i][0], []).append(i)
                for pdb, idxs in byte.items():
                    emb = net.encode(clouds[pdb]) if mode != "phys" else None
                    sites = [samples[i][1] for i in idxs]
                    phys = torch.tensor(PHt[idxs], device=device)
                    p = net(emb, sites, phys).cpu().numpy() * ysd + ymu
                    for j, i in enumerate(idxs):
                        preds[i] = p[j]
        ala = muts == "A"
        return {"all": round(float(pearsonr(preds, Y)[0]), 3), "nonala": round(float(pearsonr(preds[~ala], Y[~ala])[0]), 3),
                "alanine": round(float(pearsonr(preds[ala], Y[ala])[0]), 3)}

    res = {}
    for mode in ["phys", "surf", "both"]:
        t = time.time(); res[mode] = run_mode(mode)
        print(f"  {mode:6s}  all-AA {res[mode]['all']}   non-ala {res[mode]['nonala']}   alanine {res[mode]['alanine']}   ({time.time()-t:.0f}s)", flush=True)
    helped = res["both"]["all"] > res["phys"]["all"] + 0.01
    out = {"n_mutations": len(samples), "n_complexes": int(len(set(grp))), "epochs": epochs, "device": device,
           "results": res, "surface_helps_binding": bool(helped),
           "verdict": ("End-to-end partner-aware surface DOES add binding-magnitude signal over physics." if helped else
                       "End-to-end surface does NOT beat physics — the WT surface is mutation-blind (identical cloud per "
                       "position regardless of substitution), so it encodes POSITION not the MUTATION; it belongs on the "
                       "recognition axis (Part 1), not binding magnitude (Part 2). Physics is the ceiling.")}
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, "bind_ddg_e2e.json"), "w"), indent=1, default=float)
    print(f"\n  VERDICT: {out['verdict']}", flush=True)
    return out


if __name__ == "__main__":
    import torch
    cache = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "pdb_cache")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run(cache, epochs=int(sys.argv[2]) if len(sys.argv) > 2 else 60, device=dev)
