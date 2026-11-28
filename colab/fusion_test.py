"""fusion_test — the honest re-run of "complementary layers", but with a STRONG, externally-pretrained dMaSIF surface
net (e.g. the 2,198-complex / 0.947-AUC weights) instead of the weak per-split net the original complementary.py trained
on ~60 complexes. The question this answers:

    Does adding the dMaSIF geodesic SURFACE EMBEDDING at the mutation site, as extra features, improve held-out
    mutation-effect (ΔΔG_bind) prediction over the physics ΔΔG node ALONE — now that the surface net is strong?

Design (leakage-aware):
  - The dMaSIF net is a FIXED, pretrained self-supervised feature extractor. It never saw ΔΔG labels — it was trained on
    interface GEOMETRY only. Using its embedding of a complex is the standard pretrain->probe setup, NOT label leakage.
    (On Colab the 0.947 net was trained on RCSB human complexes, largely DISJOINT from SKEMPI, so it is also
    geometry-out-of-sample for most SKEMPI complexes. `probe_pdbs` lets you force disjointness for a strict check.)
  - The ΔΔG comparison is GroupKFold by COMPLEX: every mutation of a complex is in the same fold, so we always score
    mutations on complexes whose ΔΔG the model was not fit on. node-only vs node+embedding on the SAME folds (controlled).
  - Reported both ways: ΔΔG magnitude (Pearson) AND binding-hotspot classification (ΔΔG>1 kcal/mol, ROC-AUC — the metric
    comparable to NEXUS's ~0.75 binding number).

  run(model_path, cache, maxcplx, folds) -> outputs/orphan/fusion_test.json
"""
import os, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "orphan")
SITE_R = 8.0


def _load_net(model_path, device):
    import torch
    import nexus_train as nt
    ck = torch.load(model_path, map_location=device, weights_only=False)
    net = nt._net(ck["feat_dim"], device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["feat_dim"]


def run(model_path, cache=None, maxcplx=120, folds=5, device="cpu", workers=None, probe_pdbs=None):
    import torch
    from collections import Counter
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from scipy.stats import pearsonr
    import nexus_train as nt, flex_physics as fp, interface_hotspots as ih, surface_fingerprint as sf

    cache = cache or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "orphan", "pdb_cache")
    nt.setup(cache)
    print("=" * 100, flush=True)
    print(f"FUSION TEST — pretrained dMaSIF embedding ⊕ physics ΔΔG node, GroupKFold-by-complex, node-only baseline",
          flush=True)
    print(f"  model: {model_path}", flush=True)
    print("=" * 100, flush=True)

    net, FD = _load_net(model_path, device)

    # SKEMPI alanine mutations (measured ΔΔG_bind); rank complexes by #mutations, cap for build cost
    D = [r for r in fp.load_pdb_muts() if r["mut"] == "A"]
    cnt = Counter(r["pdb"] for r in D)
    cand = [p for p, _ in cnt.most_common(maxcplx)]
    if probe_pdbs is not None:
        cand = [p for p in cand if p in set(probe_pdbs)]
    print(f"  {len(cand)} SKEMPI complexes with alanine mutations to probe (build/cache surfaces) ...", flush=True)

    got = nt.fetch(cand, cache)
    data = nt.build(got, cache, workers=workers)                 # parallel + disk-cached (reuses any clouds already there)
    usable = [p for p in got if p in data]
    print(f"  {len(usable)} complexes usable\n", flush=True)
    if len(usable) < folds + 1:
        print("  too few usable complexes to cross-validate. stop."); return {"error": "too few complexes"}

    def embed(pdb):
        pc, g = data[pdb]["pc"], data[pdb]["g"]
        with torch.no_grad():
            e = net(torch.tensor(pc["feat"], device=device), torch.tensor(g["idx"], device=device),
                    torch.tensor(g["lc"], device=device), torch.tensor(g["dist"], device=device))
        return e.cpu().numpy()
    emb_all = {p: embed(p) for p in usable}
    E = emb_all[usable[0]].shape[1]
    ca = {p: {(r["ch"], r["rnum"]): r["ca"] for r in sf.parse_residues(p)} for p in usable}

    def site_embedding(pdb, chain, pos):
        pc = data[pdb]["pc"]; c = ca[pdb].get((chain, pos))
        if c is None:
            return np.zeros(E, np.float32)
        m = (pc["ch"] == chain) & (np.linalg.norm(pc["pts"] - c, axis=1) < SITE_R)
        if m.sum() < 2:
            return np.zeros(E, np.float32)
        return emb_all[pdb][m].mean(0)

    Xb, Xe, Y, grp = [], [], [], []
    for r in D:
        if r["pdb"] not in data:
            continue
        t = fp.table(r["pdb"])
        if t is None:
            continue
        sc = fp.sidechain_vdw(t, r["chain"], r["pos"], wt=r["wt"])
        if sc is None:
            continue
        nf = np.array(ih._feat(r) + [sc["vdw"], sc["contacts"], sc["elec"], sc["induc"]], float)
        se = site_embedding(r["pdb"], r["chain"], r["pos"])
        Xb.append(nf); Xe.append(np.concatenate([nf, se])); Y.append(float(r["ddg"])); grp.append(r["pdb"])
    Xb, Xe, Y, grp = np.array(Xb), np.array(Xe), np.array(Y), np.array(grp)
    ngrp = len(set(grp))
    k = min(folds, ngrp)
    print(f"  {len(Y)} alanine mutations over {ngrp} complexes; GroupKFold k={k}", flush=True)

    gkf = GroupKFold(n_splits=k)
    yhat_b, yhat_e, ytrue = np.zeros(len(Y)), np.zeros(len(Y)), np.array(Y)
    for tr, te in gkf.split(Xb, Y, grp):
        mb = RandomForestRegressor(300, random_state=0, n_jobs=-1).fit(Xb[tr], Y[tr]); yhat_b[te] = mb.predict(Xb[te])
        me = RandomForestRegressor(300, random_state=0, n_jobs=-1).fit(Xe[tr], Y[tr]); yhat_e[te] = me.predict(Xe[te])
    rb = float(pearsonr(yhat_b, ytrue)[0]); re = float(pearsonr(yhat_e, ytrue)[0])

    # binding-hotspot classification (ΔΔG > 1 kcal/mol), comparable to NEXUS's ~0.75 binding AUC
    yc = (ytrue > 1.0).astype(int)
    auc_b = auc_e = float("nan")
    if len(set(yc)) == 2:
        pb, pe = np.zeros(len(Y)), np.zeros(len(Y))
        for tr, te in gkf.split(Xb, yc, grp):
            if len(set(yc[tr])) < 2:
                pb[te] = yc[tr].mean(); pe[te] = yc[tr].mean(); continue
            cb = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(Xb[tr], yc[tr]); pb[te] = cb.predict_proba(Xb[te])[:, 1]
            ce = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(Xe[tr], yc[tr]); pe[te] = ce.predict_proba(Xe[te])[:, 1]
        auc_b = float(roc_auc_score(yc, pb)); auc_e = float(roc_auc_score(yc, pe))

    print(f"\n  ΔΔG magnitude  (Pearson): node-only {rb:.3f}   node+dMaSIF {re:.3f}   (Δ {re-rb:+.3f})", flush=True)
    print(f"  hotspot ΔΔG>1  (ROC-AUC): node-only {auc_b:.3f}   node+dMaSIF {auc_e:.3f}   (Δ {auc_e-auc_b:+.3f})", flush=True)

    helped = (re - rb) > 0.015 or (auc_e - auc_b) > 0.015
    verdict = (
        f"With the STRONG pretrained dMaSIF net ({os.path.basename(model_path)}) as a fixed surface-embedding feature, "
        f"over {len(Y)} alanine mutations / {ngrp} complexes (GroupKFold k={k}, node-only baseline on the same folds): "
        f"ΔΔG-magnitude Pearson {rb:.2f}->{re:.2f} (Δ{re-rb:+.2f}); hotspot AUC {auc_b:.2f}->{auc_e:.2f} (Δ{auc_e-auc_b:+.2f}). "
        + ("So the strong surface embedding DOES add measurable mutation-effect signal on top of the physics node — the "
           "layers are complementary and worth fusing."
           if helped else
           "So even the strong (0.9-AUC-grade) surface embedding does NOT meaningfully improve mutation-effect prediction "
           "over the physics node — confirming the structural prediction: dMaSIF embeds WILD-TYPE surface geometry, and a "
           "point mutation's effect is a mutant-minus-WT DELTA the WT embedding can't express, however good it is. The "
           "honest architecture stays a PIPELINE (dMaSIF finds/scores the interface & partner; the physics node scores the "
           "specific substitution), not a fusion.")
        + f"  LIMITS: alanine mutations only; hotspot threshold ΔΔG>1; RandomForest probe; the number is what it is.")
    print("\n" + "=" * 100 + f"\nVERDICT: {verdict}\n" + "=" * 100, flush=True)

    out = {"model": os.path.basename(model_path), "n_mutations": int(len(Y)), "n_complexes": int(ngrp), "folds": int(k),
           "node_only_pearson": round(rb, 3), "node_plus_dmasif_pearson": round(re, 3), "pearson_gain": round(re - rb, 3),
           "node_only_hotspot_auc": round(auc_b, 3), "node_plus_dmasif_hotspot_auc": round(auc_e, 3),
           "hotspot_auc_gain": round(auc_e - auc_b, 3), "helped": bool(helped), "verdict": verdict}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/fusion_test.json", "w"), indent=1, default=float)
    print(f"  -> {OUT}/fusion_test.json", flush=True)
    return out


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "nexus_dmasif.pt")
    cache = sys.argv[2] if len(sys.argv) > 2 else None
    maxc = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    run(model, cache, maxc)
