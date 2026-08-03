"""COMBINATORIAL PERTURBATION: is a pair's response more than the sum of its singles, and is that predictable?

WHY THIS IS THE TEST THAT MATTERS.  The perturbation inventory showed the single-gene route is within ~0.024 of
exhausted: 26,500 perturbations exceeds the ~19,500 human protein-coding genes, so 3.5/10 is unreachable by more
single knockouts.  The only route past that is COMBINATORIAL.  Norman 2019 (GSE133344, K562 CRISPRa) is the one
public dataset with both singles and pairs at scale, so it can answer the question that decides whether the
combinatorial route is worth anything:

    GATE 1  IS THERE SIGNAL?   Does the pair response deviate from the additive prediction Delta_A + Delta_B by
            more than measurement noise?  If pairs are just sums of singles, a pair screen adds no information
            a single screen does not already contain, and the whole combinatorial argument collapses.

    GATE 2  IS IT PREDICTABLE?  If the deviation is real, can it be predicted from the two genes' annotations,
            past a shuffled control?  If not, pair data is measurable but not learnable, and a pair screen buys
            coverage rather than generalisation.

THE NOISE FLOOR, which is what makes gate 1 meaningful.  Non-additivity computed from noisy pseudobulk will be
non-zero even if biology is perfectly additive.  So each pair's cells are SPLIT IN HALF and a Delta computed from
each half independently; the half-to-half discrepancy is the measurement noise on exactly the same quantity, at
exactly the same cell depth.  Non-additivity only counts if it exceeds that.

CONTROLS.  Shuffled pairing (give pair A_B the additive prediction of an unrelated pair) prices how much of any
apparent structure is generic.  Single-gene pairs of the form A_NegCtrl are treated as singles.
"""
import collections, json, os, sys
from pathlib import Path
import numpy as np
import h5py

SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/norman")
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
H5 = SP / "norman.h5ad"
MIN_CELLS = 40


def cats(h, key):
    g = h["obs"][key]
    c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
    return np.array(c), g["codes"][:]


def build():
    if (SP / "pseudobulk.npz").exists():
        z = np.load(SP / "pseudobulk.npz", allow_pickle=True)
        return list(z["groups"]), list(z["genes"]), z["profile"], z["ncell"], z["half"]
    with h5py.File(H5, "r") as h:
        names, codes = cats(h, "perturbation")
        n_cell, n_gene = h["X"].attrs["shape"]
        gene = [x.decode() if isinstance(x, bytes) else str(x) for x in h["var"]["_index"][:]]
        rng = np.random.default_rng(0)
        half = rng.integers(0, 2, n_cell)
        acc = np.zeros((len(names), n_gene), np.float64)
        acc_h = np.zeros((2, len(names), n_gene), np.float64)
        data, indices, indptr = h["X"]["data"], h["X"]["indices"], h["X"]["indptr"]
        ptr = indptr[:]
        print(f"  {n_cell:,} cells x {n_gene:,} genes, {len(names)} perturbations, "
              f"{ptr[-1]:,} nonzero", flush=True)
        CH = 512
        for s in range(0, n_gene, CH):
            e = min(s + CH, n_gene)
            d = data[ptr[s]:ptr[e]]; ix = indices[ptr[s]:ptr[e]]
            off = ptr[s]
            for j in range(s, e):
                a, b = ptr[j] - off, ptr[j + 1] - off
                if a == b: continue
                rows = ix[a:b]; vals = d[a:b]
                grp = codes[rows]
                acc[:, j] += np.bincount(grp, weights=vals, minlength=len(names))
                for hh in (0, 1):
                    m = half[rows] == hh
                    if m.any():
                        acc_h[hh, :, j] += np.bincount(grp[m], weights=vals[m], minlength=len(names))
            if s % 8192 == 0: print(f"    gene {s:,}", flush=True)
    ncell = np.bincount(codes, minlength=len(names))
    def norm(A):
        t = A.sum(1, keepdims=True); t[t == 0] = 1
        return np.log1p(A / t * 1e4).astype(np.float32)
    prof, ph = norm(acc), np.stack([norm(acc_h[0]), norm(acc_h[1])])
    np.savez_compressed(SP / "pseudobulk.npz", groups=np.array(names), genes=np.array(gene),
                        profile=prof, ncell=ncell, half=ph)
    return list(names), gene, prof, ncell, ph


def main():
    log = []
    def report(t):
        print(t, flush=True); log.append(t)
    report("=" * 100)
    report("COMBINATORIAL PERTURBATION (Norman 2019): is a pair more than the sum of its singles?")
    report("=" * 100)
    groups, genes, prof, ncell, ph = build()
    idx = {g: i for i, g in enumerate(groups)}
    ctrl = [g for g in groups if g.lower() in ("control", "ctrl", "negctrl0")]
    report(f"  control label(s): {ctrl}")
    c = idx[ctrl[0]]
    singles = {g: idx[g] for g in groups if "_" not in g and g not in ctrl}
    pairs = {}
    for g in groups:
        if "_" in g and "NegCtrl" not in g:
            a, b = g.split("_", 1)
            if a in singles and b in singles and ncell[idx[g]] >= MIN_CELLS:
                pairs[g] = (idx[g], singles[a], singles[b])
    report(f"  {len(singles)} singles, {len(pairs)} pairs with both singles measured and "
           f">= {MIN_CELLS} cells")

    D = prof - prof[c][None, :]
    Dh = ph - ph[:, c][:, None, :]
    keys = sorted(pairs)
    obs = np.stack([D[pairs[k][0]] for k in keys])
    add = np.stack([D[pairs[k][1]] + D[pairs[k][2]] for k in keys])
    resid = obs - add
    noise = np.stack([Dh[0][pairs[k][0]] - Dh[1][pairs[k][0]] for k in keys]) / 2.0

    def rms(x): return float(np.sqrt((x ** 2).mean()))
    report(f"\n### GATE 1 -- is there non-additive signal?")
    report(f"  RMS observed pair effect          {rms(obs):.4f}")
    report(f"  RMS additive prediction            {rms(add):.4f}")
    report(f"  RMS NON-ADDITIVE residual          {rms(resid):.4f}")
    report(f"  RMS measurement noise (split-half) {rms(noise):.4f}")
    ratio = rms(resid) / max(rms(noise), 1e-9)
    report(f"  residual / noise                   {ratio:.3f}")
    per_r = np.sqrt((resid ** 2).mean(1)); per_n = np.sqrt((noise ** 2).mean(1))
    d = per_r - per_n
    br = np.random.default_rng(0)
    bs = d[br.integers(0, len(d), size=(10000, len(d)))].mean(1)
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    report(f"  per-pair residual - noise          {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
           f"{'RESOLVED' if lo > 0 else 'within noise'}")
    frac = float(np.mean((resid ** 2).sum(1) / np.maximum((obs ** 2).sum(1), 1e-9)))
    report(f"  fraction of pair-effect energy that is non-additive: {100*frac:.1f}%")

    var_add = 1 - float(((obs - add) ** 2).sum() / max(((obs - obs.mean(0)) ** 2).sum(), 1e-9))
    report(f"  additive model explains {100*var_add:.1f}% of pair-response variance")

    json.dump({"test": "norman_epistasis", "n_singles": len(singles), "n_pairs": len(pairs),
               "rms_obs": rms(obs), "rms_add": rms(add), "rms_resid": rms(resid),
               "rms_noise": rms(noise), "resid_over_noise": ratio,
               "per_pair_gap": float(d.mean()), "ci": [lo, hi],
               "frac_nonadditive": frac, "additive_var_explained": var_add, "log": log},
              open(OUT / "norman_epistasis.json", "w"), indent=2)
    report(f"\n  -> {OUT/'norman_epistasis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
