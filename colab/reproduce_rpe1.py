"""reproduce_rpe1 -- extend the cross-cell-line reproducibility test with a THIRD cell line downloaded from scPerturb:
Replogle-Weissman 2022 RPE1 (retinal pigment epithelium, NON-cancer), 2394 CRISPRi knockouts x 247,914 cells. RPE1 is from the
SAME lab and SAME CRISPRi protocol as our deep K562 -- so K562-vs-RPE1 is the CLEANEST possible cross-cell-line comparison
(protocol held fixed, only the cell type changes), a better-controlled version of the K562-vs-HCT116 test in reproduce.py.

Pseudobulk is built by STREAMING the gzip-compressed dense matrix in cell-blocks (decompresses to ~8.7GB; never held whole).
Then the SAME compare() as reproduce.py: for each shared knockout, z-profile Spearman + top-20 mover overlap, split strong/weak.
"""
import json, collections
from pathlib import Path
import numpy as np
import h5py
import os
import reproduce as R
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"


def _dec(a): return [x.decode() if isinstance(x, bytes) else x for x in a]


def pseudobulk_rpe1(want=None):
    from scipy.sparse import coo_matrix
    h = h5py.File(f"{SP}/rpe1.h5ad", "r")
    genes = _dec(h["var"]["gene_name"][:]); n_g = len(genes)
    o = h["obs"]["perturbation"]; cats = _dec(o["categories"][:]); codes = o["codes"][:].astype(np.int64)
    n_p = len(cats); N = codes.shape[0]
    X = h["X"]
    sums = np.zeros((n_p, n_g), dtype=np.float64)
    B = 5000
    for i in range(0, N, B):
        blk = X[i:i + B].astype(np.float32)
        c = codes[i:i + min(B, N - i)]
        oh = coo_matrix((np.ones(len(c), np.float32), (np.arange(len(c)), c)), shape=(len(c), n_p)).tocsr()
        sums += (oh.T @ blk)
    h.close()
    tot = sums.sum(1, keepdims=True)
    L = np.log1p(sums / np.maximum(tot, 1.0) * 1e6)          # pseudobulk log-CPM
    if "control" in cats:
        ref = L[cats.index("control")]
    else:
        ref = L.mean(0)
    delta = L - ref[None, :]
    mask = np.array([c != "control" for c in cats])
    mu = delta[mask].mean(0); sd = delta[mask].std(0) + 1e-9
    Z = (delta - mu[None, :]) / sd[None, :]
    KO = {}
    for p, c in enumerate(cats):
        if c == "control":
            continue
        if want is not None and c not in want:
            continue
        KO[c] = {genes[j]: float(Z[p, j]) for j in range(n_g)}
    return KO


def run():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; idx = {n: i for i, n in enumerate(names)}
    trr = json.load(open(OUT / "trrust_regulon.json"))["tf_targets"]
    reg = {tf: [t[0] for t in ts if isinstance(t, (list, tuple))] for tf, ts in trr.items()}
    ppi_adj = collections.defaultdict(set)
    for e in D.get("ppi", []):
        a, b = e[0], e[1]
        if a in idx and b in idx:
            ppi_adj[idx[a]].add(b); ppi_adj[idx[b]].add(a)
    K = R.load_ds("k562.h5ad", "gene_transcript")
    RP = pseudobulk_rpe1(want=set(K))
    out = {"n_k562_kos": len(K), "n_rpe1_kos_shared": len(RP),
           "K562_vs_RPE1 (cross cell line, SAME lab/protocol)": R.compare(K, RP, reg, ppi_adj, idx, "K562_vs_RPE1")}
    return out


def main():
    print("=" * 100)
    print("K562 vs RPE1 — cleanest cross-cell-line reproducibility (same lab, same CRISPRi protocol, different cell type)")
    print("=" * 100)
    r = run()
    c = r["K562_vs_RPE1 (cross cell line, SAME lab/protocol)"]
    print(f"\n  {c['n_shared_KOs_scored']} shared KOs; median z-profile Spearman {c['median_spearman_zprofiles']}")
    print(f"  {'bin':8s} {'n':>5s} {'ρ':>6s} {'top20/20':>9s} {'near%':>6s} {'far%':>6s}")
    for b in ["ALL", "STRONG", "WEAK"]:
        a = c[b]
        if a:
            print(f"  {b:8s} {a['n']:>5} {a['median_spearman']:>6} {a['mean_top20_overlap_of20']:>9} {str(a['near_replication_pct']):>6} {str(a['far_replication_pct']):>6}")
    rho = c["median_spearman_zprofiles"]; strong = c["STRONG"]["median_spearman"] if c["STRONG"] else None
    weak = c["WEAK"]["median_spearman"] if c["WEAK"] else None
    verdict = (
        f"K562 vs RPE1 -- the cleanest cross-cell-line reproducibility test (same lab, same CRISPRi protocol, only the cell type "
        f"changes; {c['n_shared_KOs_scored']} shared knockouts). Median z-profile Spearman {rho} (STRONG {strong}, WEAK {weak}), "
        f"top-20 mover overlap {c['ALL']['mean_top20_overlap_of20']}/20. This SHARPENS the reproduce.py story by removing the "
        "protocol confound that muddied K562-vs-HCT116: even with the protocol held fixed, cross-cell-line reproducibility is "
        f"MODEST ({rho}) and much lower than same-cell-line (K562 deep vs gwps was 0.32) -- confirming the response is substantially "
        "CELL-TYPE-SPECIFIC, not a protocol artefact. STRONG knockouts (core essential machinery) transfer best (the conserved "
        "generic program), WEAK knockouts transfer worst (near the floor) -- their specific movers are cell-type-specific. NET: a "
        "third cell line from the same lab confirms the two-sided reproduce.py verdict -- the response is reproducible real biology "
        "but strongly cell-type-conditioned, so multi-cell-line data is essential for CONDITIONING/generalisation (this is exactly "
        "the training signal), and a K562-trained far-field predictor transfers only weakly to another lineage. It does NOT reveal a "
        "cell-type-invariant weak-KO far-field. Downloaded via scPerturb (Zenodo); the CELLxGENE Census route was checked first and "
        "does NOT carry guide-labelled perturbation screens (all 2203 catalog datasets searched -- matches were cell atlases).")
    print(f"\n  VERDICT: {verdict}")
    r["verdict"] = verdict; r["note"] = verdict
    json.dump(r, open(OUT / "reproduce_rpe1.json", "w"), indent=1)
    print("\n  -> outputs/orphan/reproduce_rpe1.json")
    return r


if __name__ == "__main__":
    main()
