"""epistasis — does simulate/cure's ADDITIVE assumption hold? Measured on real double perturbations (Norman 2019
CRISPRa genetic-interaction map: ~100 singles + ~130 doubles in K562).

For every double A+B whose singles A and B are also measured:
    additive prediction  = delta_A + delta_B
    real                 = delta_AB   (measured)
    epistasis            = real - additive     (the non-additive residual = genetic interaction)
Reports: how well additive predicts the real double (Pearson r) vs the best single alone, the epistasis magnitude
||real-additive||/||real||, and the pairs with the LARGEST synergy (where additive fails). This is the honest test
of whether 'there is no spoon' combos are trustworthy — and where they need the measured double.

Pseudobulk: mean expression per perturbation (sparse group-matmul), log1p, delta vs control. Top-variance genes.
-> outputs/orphan/epistasis.json
"""
import os, sys, json
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, os.path.dirname(__file__))

OUT = "outputs/orphan"
SCRATCH = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
NORMAN = f"{SCRATCH}/norman.h5ad"


def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def pseudobulk(path=NORMAN):
    import h5py
    f = h5py.File(path, "r")
    shape = tuple(int(x) for x in f["X"].attrs["shape"])
    print(f"  loading sparse X {shape} …", flush=True)
    X = sp.csc_matrix((f["X"]["data"][:], f["X"]["indices"][:], f["X"]["indptr"][:]), shape=shape)
    genes = [x.decode() if isinstance(x, bytes) else x for x in f["var"]["_index"][:]]
    pc = f["obs"]["perturbation"]
    if isinstance(pc, h5py.Group):
        cats = [x.decode() if isinstance(x, bytes) else x for x in pc["categories"][:]]
        codes = pc["codes"][:]
    else:
        cats = sorted(set(x.decode() for x in pc[:])); code_map = {c: i for i, c in enumerate(cats)}
        codes = np.array([code_map[x.decode()] for x in pc[:]])
    f.close()
    labels = [cats[c] if c >= 0 else "?" for c in codes]
    uniq = sorted(set(labels))
    lab2i = {l: i for i, l in enumerate(uniq)}
    rows = np.array([lab2i[l] for l in labels])
    G = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, np.arange(len(rows)))),
                      shape=(len(uniq), shape[0]))
    counts = np.asarray(G.sum(1)).ravel()
    print(f"  {len(uniq)} perturbations; pseudobulking …", flush=True)
    PB = np.asarray((G @ X).todense(), dtype=np.float32) / counts[:, None]   # mean raw counts per perturbation
    PB = np.log1p(PB)                                                        # -> log space (additive fold-changes)
    return PB, uniq, genes, counts


def run(path=NORMAN, n_var_genes=3000):
    PB, uniq, genes, counts = pseudobulk(path)
    idx = {l: i for i, l in enumerate(uniq)}
    ctrl = next((l for l in uniq if l.lower() in ("control", "ctrl", "negctrl", "non-targeting")), None)
    if ctrl is None:
        ctrl = min(uniq, key=lambda l: 0 if "ctrl" in l.lower() or "control" in l.lower() else 1)
    print(f"  control = '{ctrl}' ({int(counts[idx[ctrl]])} cells)", flush=True)
    D = PB - PB[idx[ctrl]]                                                   # delta vs control (log-fold)
    v = D.var(0); keep = np.argsort(-v)[:n_var_genes]
    D = D[:, keep]

    singles = set(l for l in uniq if "_" not in l and l != ctrl)
    doubles = [l for l in uniq if "_" in l and l != ctrl]
    rows = []
    for dbl in doubles:
        parts = dbl.split("_")
        if len(parts) != 2:
            continue
        a, b = parts
        if a in singles and b in singles:
            dab, da, db = D[idx[dbl]], D[idx[a]], D[idx[b]]
            add = da + db
            r_add = _pearson(add, dab)
            r_best_single = max(_pearson(da, dab), _pearson(db, dab))
            epi = float(np.linalg.norm(dab - add) / (np.linalg.norm(dab) + 1e-9))
            rows.append(dict(pair=dbl, r_additive=round(r_add, 3), r_best_single=round(r_best_single, 3),
                             epistasis_frac=round(epi, 3), double_mag=round(float(np.linalg.norm(dab)), 2)))
    rows.sort(key=lambda r: -r["epistasis_frac"])
    n = len(rows)
    r_add = float(np.mean([r["r_additive"] for r in rows]))
    r_sing = float(np.mean([r["r_best_single"] for r in rows]))
    epi = float(np.mean([r["epistasis_frac"] for r in rows]))
    strong_syn = [r for r in rows if r["epistasis_frac"] > 0.5]
    res = {
        "data": "Norman 2019 CRISPRa GI map (K562)", "n_doubles_tested": n,
        "mean_r_additive": round(r_add, 3), "mean_r_best_single": round(r_sing, 3),
        "additive_lift_over_single": round(r_add - r_sing, 3),
        "mean_epistasis_frac": round(epi, 3),
        "frac_pairs_strong_synergy(>0.5)": round(len(strong_syn) / n, 3) if n else None,
        "most_synergistic_pairs": rows[:10],
        "most_additive_pairs": rows[-6:],
        "verdict": None,
    }
    res["verdict"] = (
        f"Additive predicts the real double at r={r_add:.2f} (vs best single r={r_sing:.2f}, so combining the two "
        f"singles genuinely adds +{r_add-r_sing:.2f}). Mean epistasis residual = {epi:.0%} of the double's "
        f"magnitude; {len(strong_syn)}/{n} pairs are strongly non-additive (>50% residual). => simulate/cure's "
        f"additive model is a REAL first-order predictor, but {len(strong_syn)} pairs need the measured double — "
        f"epistasis is not negligible, and it is exactly the synergy combination therapy cares about.")
    json.dump(res, open(f"{OUT}/epistasis.json", "w"), indent=2)
    print("=" * 78)
    print("EPISTASIS — is the additive combo model right? (Norman 2019 real doubles)")
    print("=" * 78)
    print(f"  doubles tested: {n}")
    print(f"  additive vs real double:   r = {r_add:.3f}")
    print(f"  best single vs real:       r = {r_sing:.3f}   (additive adds +{r_add-r_sing:.3f})")
    print(f"  mean epistasis residual:   {epi:.1%} of the double's magnitude")
    print(f"  strongly synergistic pairs (>50% residual): {len(strong_syn)}/{n}")
    print(f"  top synergy: " + ", ".join(r["pair"] for r in rows[:6]))
    print(f"  -> {res['verdict']}")
    print("=" * 78)
    return res


if __name__ == "__main__":
    run()
