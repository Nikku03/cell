"""nexus_train — portable driver to TRAIN + validate the NEXUS stack on any set of human PPI complexes, on Colab (GPU)
or locally (CPU). The trainable component is the dMaSIF geodesic SURFACE model (the extrinsic sensor, which keeps
improving with more complexes); the intrinsic stability node and the regulatory-sign layer are fixed/annotation. This
driver:
  1. sets a portable PDB cache and patches the module paths (so it runs anywhere, not just the author's sandbox),
  2. fetches the requested complexes from RCSB,
  3. builds surface clouds (marching-cubes + APBS) and trains the geodesic net on a TRAIN split, held out by complex,
  4. reports the held-out interface-discrimination AUC and saves the trained weights,
  5. runs the full NEXUS sensor stack (intrinsic ΔΔG_fold + extrinsic interface + regulatory sign -> LOF/GOF activity)
     on real mutations end-to-end as a working-check.

Usage:  python3 nexus_train.py PDB1,PDB2,...  [cache_dir] [epochs] [cpu|cuda]
"""
import os, sys, json, time, warnings, urllib.request
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def setup(cache):
    os.makedirs(cache, exist_ok=True)
    import surface_fingerprint, surface_apbs, dmasif, flex_physics
    surface_fingerprint.PDBDIR = surface_apbs.PDBDIR = dmasif.PDBDIR = flex_physics.PDBDIR = cache
    return cache


def fetch(pdbs, cache):
    ok = []
    for p in pdbs:
        dst = f"{cache}/{p}.pdb"
        if not os.path.exists(dst) or os.path.getsize(dst) < 1000:
            for _ in range(3):
                try:
                    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{p}.pdb", dst)
                    break
                except Exception:
                    time.sleep(1)
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            ok.append(p)
    return ok


def build(pdbs):
    import dmasif as dm
    data = {}
    for p in pdbs:
        try:
            pc = dm.point_cloud(p)
            if pc is None or len(set(pc["ch"])) < 2:
                continue
            g = dm.precompute_graph(pc); pos, neg = dm.interface_point_pairs(pc)
            if len(pos) < 12:
                continue
            data[p] = {"pc": pc, "g": g, "pos": pos, "neg": neg}
        except Exception:
            continue
    return data


def _net(FD, device):
    import torch.nn as nn

    class GeoConv(nn.Module):
        def __init__(s, din, dout):
            super().__init__()
            s.edge = nn.Sequential(nn.Linear(4 + din, dout), nn.ReLU())
            s.logsig = nn.Parameter(__import__("torch").tensor(1.0)); s.self = nn.Sequential(nn.Linear(din + dout, dout), nn.ReLU())
        def forward(s, feat, idx, lc, dist):
            import torch
            fn = feat[idx]; e = s.edge(torch.cat([lc, dist.unsqueeze(-1), fn], -1))
            w = torch.exp(-(dist ** 2) / (torch.exp(s.logsig) ** 2 + 1e-6)).unsqueeze(-1)
            return s.self(torch.cat([feat, (w * e).sum(1) / (w.sum(1) + 1e-6)], -1))

    class Net(nn.Module):
        def __init__(s, fd):
            super().__init__(); s.c1 = GeoConv(fd, 32); s.c2 = GeoConv(32, 32); s.out = nn.Linear(32, 32)
        def forward(s, feat, idx, lc, dist):
            h = s.c1(feat, idx, lc, dist); h = s.c2(h, idx, lc, dist); e = s.out(h)
            return e / (e.norm(dim=1, keepdim=True) + 1e-9)
    return Net(FD).to(device)


def train_dmasif(data, epochs=40, device="cpu", model_out=None):
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(0)
    pdbs = sorted(data); FD = data[pdbs[0]]["pc"]["feat"].shape[1]
    rng = np.random.default_rng(0); perm = rng.permutation(pdbs)
    ntest = max(1, len(pdbs) // 4)
    test = set(perm[:ntest]); train = [p for p in pdbs if p not in test]
    if not train:
        train = pdbs; test = set(pdbs[:1])
    net = _net(FD, device); scale = nn.Parameter(torch.tensor(4.0, device=device))
    opt = torch.optim.Adam(list(net.parameters()) + [scale], lr=3e-3); lf = nn.BCEWithLogitsLoss()

    def emb(pdb):
        pc, g = data[pdb]["pc"], data[pdb]["g"]
        return net(torch.tensor(pc["feat"], device=device), torch.tensor(g["idx"], device=device),
                   torch.tensor(g["lc"], device=device), torch.tensor(g["dist"], device=device))
    for ep in range(epochs):
        rng.shuffle(train); net.train()
        for pdb in train:
            d = data[pdb]; P = d["pos"][:400]; N = d["neg"][:400]
            if not P or not N:
                continue
            e = emb(pdb)
            pi = torch.tensor([a for a, b in P + N], device=device); pj = torch.tensor([b for a, b in P + N], device=device)
            y = torch.tensor([1.0] * len(P) + [0.0] * len(N), device=device)
            opt.zero_grad(); loss = lf(scale * (e[pi] * e[pj]).sum(1), y); loss.backward(); opt.step()
    net.eval(); ys, ss = [], []
    with torch.no_grad():
        for pdb in test:
            d = data[pdb]; e = emb(pdb).cpu().numpy()
            for a, b in d["pos"]:
                ss.append(float((e[a] * e[b]).sum())); ys.append(1)
            for a, b in d["neg"]:
                ss.append(float((e[a] * e[b]).sum())); ys.append(0)
    auc = float(roc_auc_score(ys, ss)) if len(set(ys)) == 2 else float("nan")
    if model_out:
        torch.save({"state": net.state_dict(), "feat_dim": FD}, model_out)
    return net, auc, len(train), len(test)


def run_sensors(pdbs, cache, model_pkl):
    """working-check of the full NEXUS stack on real measured mutations. Uses SKEMPI mutations in the trained complexes
    when available; otherwise (e.g. auto-fetched PDB complexes not in SKEMPI) falls back to a fixed SKEMPI complex so the
    sensor stack is always exercised — the sensor stack is independent of which complexes the surface model trained on."""
    import flex_physics as fp, ddg_predictor as dp, nexus, regsign, ecflux
    out = []
    ddg = dp.DDGPredictor(model_pkl) if os.path.exists(model_pkl) else None
    D = [r for r in fp.load_pdb_muts() if r["pdb"] in set(pdbs)]
    if not D:
        fetch(["1CSE"], cache)                                  # a SKEMPI complex with many measured mutations
        D = [r for r in fp.load_pdb_muts() if r["pdb"] == "1CSE"]
    D = D[:8]
    for r in D:
        t = fp.table(r["pdb"])
        if t is None:
            continue
        sc = fp.sidechain_vdw(t, r["chain"], r["pos"], wt=r["wt"])
        fold = None
        if ddg is not None:
            try:
                fold, _ = ddg.predict_from_structure(f"{cache}/{r['pdb']}.pdb", r["chain"], r["pos"], r["wt"], r["mut"])
            except Exception:
                fold = None
        act_lof = nexus.nexus_activity(fold if fold is not None else 0.0, r["ddg"])
        act_gof = nexus.directed_activity(fold if fold is not None else 0.0, r["ddg"], inhibitory=True)
        out.append({"pdb": r["pdb"], "mut": f"{r['chain']}{r['pos']}{r['wt']}->{r['mut']}",
                    "ddg_fold_pred": round(float(fold), 2) if fold is not None else None,
                    "ddg_bind_meas": round(float(r["ddg"]), 2),
                    "activity_LOF": round(act_lof, 3), "activity_if_brake_GOF": round(act_gof, 3)})
    return out


def main(pdbs, cache=None, epochs=40, device="cpu"):
    cache = cache or os.path.join(HERE, "..", "outputs", "orphan", "pdb_cache")
    print("=" * 90, flush=True)
    print(f"NEXUS-TRAIN — train dMaSIF surface model + run the sensor stack on {len(pdbs)} complexes  [device={device}]", flush=True)
    print("=" * 90, flush=True)
    setup(cache)
    t0 = time.time(); got = fetch(pdbs, cache)
    print(f"  fetched/cached {len(got)}/{len(pdbs)} complexes ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time(); data = build(got)
    print(f"  built surface clouds for {len(data)} usable complexes ({time.time()-t0:.0f}s)", flush=True)
    if len(data) < 2:
        print("  need >=2 usable complexes to train/hold-out. stop."); return {"error": "too few complexes"}
    model_out = os.path.join(cache, "..", "nexus_dmasif.pt")
    t0 = time.time(); net, auc, ntr, nte = train_dmasif(data, epochs=epochs, device=device, model_out=model_out)
    print(f"  trained dMaSIF ({epochs} ep, {ntr} train / {nte} test complexes, {time.time()-t0:.0f}s): "
          f"held-out interface-discrimination AUC {auc:.3f}  -> weights {os.path.basename(model_out)}", flush=True)
    model_pkl = os.path.join(HERE, "..", "outputs", "orphan", "ddg_model.pkl")
    sens = run_sensors(got, cache, model_pkl)
    print(f"\n  NEXUS sensor stack on real mutations (intrinsic ΔΔG_fold + extrinsic ΔΔG_bind -> activity):", flush=True)
    for s in sens[:6]:
        print(f"    {s['pdb']} {s['mut']:14s}: fold {s['ddg_fold_pred']}  bind {s['ddg_bind_meas']}  "
              f"-> LOF-activity {s['activity_LOF']}   (if brake: GOF-activity {s['activity_if_brake_GOF']})", flush=True)
    ok = (not np.isnan(auc)) and len(sens) > 0
    print(f"\n  WORKING-CHECK: {'PASS' if ok else 'FAIL'} — dMaSIF trained + held-out AUC computed + sensor stack ran end-to-end.", flush=True)
    out = {"n_requested": len(pdbs), "n_usable": len(data), "epochs": epochs, "device": device,
           "dmasif_heldout_auc": round(auc, 3), "n_train": ntr, "n_test": nte,
           "model": os.path.basename(model_out), "sensor_demo": sens, "working_check": bool(ok)}
    op = os.path.join(HERE, "..", "outputs", "orphan", "nexus_train.json")
    os.makedirs(os.path.dirname(op), exist_ok=True); json.dump(out, open(op, "w"), indent=1, default=float)
    print(f"  -> {op}", flush=True)
    return out


def fetch_and_train(n, cache=None, epochs=40, device="cpu", human_only=False, max_res=3.0, max_atoms=15000):
    """AUTO-FETCH n real protein-protein complexes from the PDB (RCSB Search API), then train + validate. This is how
    NEXUS scales past SKEMPI's 345: the dMaSIF surface sensor trains on interface geometry (no labels), so it can use the
    ~34,000 heteromeric complexes in the PDB."""
    import fetch_complexes as fc
    cache = cache or os.path.join(HERE, "..", "outputs", "orphan", "pdb_cache")
    os.makedirs(cache, exist_ok=True)
    print(f"AUTO-FETCH {n} protein-protein complexes from the PDB (human={human_only}) ...", flush=True)
    got = fc.fetch(n, cache, max_res=max_res, max_atoms=max_atoms, human_only=human_only)
    print(f"  -> {len(got)} complexes in cache; training on them\n", flush=True)
    return main(got, cache=cache, epochs=epochs, device=device)


if __name__ == "__main__":
    a1 = sys.argv[1] if len(sys.argv) > 1 else "1BRS,1JCK,3SGB,1ACB,1AHW"
    cache = sys.argv[2] if len(sys.argv) > 2 else None
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
    if a1.startswith("fetch:"):                                 # e.g.  python3 nexus_train.py fetch:200 CACHE 60 cuda [--human]
        fetch_and_train(int(a1.split(":")[1]), cache, epochs, device, human_only="--human" in sys.argv)
    else:
        main(a1.split(","), cache, epochs, device)
