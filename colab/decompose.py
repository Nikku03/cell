"""BUILD 3 of 5 -- SHARED CORE vs SPECIFIC ARM, and whether the specific arm is real.

WHY THIS IS THE PIVOT OF THE WHOLE DESIGN. DESIGN.md sec 4 and sec 5 both depend on one unmeasured assumption:
that a knockout's response splits into a SHARED STRESS CORE that appears for every damaged cell, plus a SPECIFIC
ARM that belongs to that gene alone. Everything downstream rests on it --

    sec 4  the core and the arm "need different predictors and currently share one"
    sec 5  the interventional regulon must be built on "the SPECIFIC residual after removing the shared core"

but the split has never been performed, and the arm has never been shown to be anything other than noise. If the
residual is noise, build 4 is dead on arrival and this file says so. That is the point of running it before
building 4 rather than after.

WHAT IS ACTUALLY MEASURED

  A  FIT THE CORE ON A TRAINING HALF ONLY. Truncated SVD of the training perturbations gives k gene-space axes;
     every perturbation, train and test, is then projected onto those FIXED axes. Fitting the core on all
     perturbations would let it absorb each perturbation's own specific signal, and the residual would look empty
     for a purely circular reason.

  B  HOW BIG IS EACH PART. Fraction of squared response carried by the core versus the residual, on held-out
     perturbations, as a function of k.

  C  IS COMPONENT 1 A DAMAGE AXIS? If the shared core is what DESIGN.md claims -- one stress response scaled by
     how badly the cell was hurt -- then a perturbation's loading on it should track damage magnitude and gene
     essentiality, and its gene weights should be the globally most-often-moved genes. All three are checked; any
     of them failing means the "shared core" story is wrong even if the maths works.

  D  THE DECISIVE TEST: IS THE RESIDUAL FUNCTIONALLY STRUCTURED? Two knockouts of genes in the SAME PROTEIN
     COMPLEX should leave similar residuals if the residual carries mechanism, and uncorrelated ones if it is
     noise. Measured as cosine similarity between residual profiles, against three controls, because this
     comparison is exactly where a confound would hide:

        magnitude-matched pairs   complex partners are both usually big responders, and any two big responders
                                  look alike. Random pairs are therefore drawn to MATCH the joint distribution of
                                  response magnitude deciles of the real complex pairs, not drawn uniformly.
        permuted membership       complex labels shuffled across genes, keeping complex sizes fixed
        THE SAME TEST ON THE RAW MATRIX   run before the core is removed. If raw coherence is high and residual
                                  coherence is not, then the apparent "functional signal" in this data is the
                                  shared core -- which is precisely DESIGN.md sec 5's stated worry, and would
                                  mean the interventional regulon of build 4 cannot be separated from it.

  E  the residual matrix is written out, because build 4 consumes it.

PRE-REGISTERED CRITERION, SET BEFORE RUNNING. The decomposition is worth keeping only if complex-partner residual
similarity exceeds magnitude-matched pairs with a permutation z >= 3, AND the effect survives removing the core
rather than shrinking to nothing. If it does not, build 4's premise is refuted and that is the finding.
"""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
TAU, TIDE_FRAC, MIN_SPEC = 1.0, 0.05, 5
RANKS = (1, 2, 5, 10, 20, 40, 60)
N_PERM = 200


def load():
    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos, genes, M = list(z["kos"]), list(z["genes"]), np.asarray(z["M"], np.float32)
    bad = int((~np.isfinite(M)).sum())
    assert bad == 0, f"NON-FINITE readout: {bad:,} cells -- rebuild it, do not sanitise it here"
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC
    spec = ((A >= TAU) & ~tide).sum(1)
    print(f"readout: {M.shape[0]:,} perturbations x {M.shape[1]:,} genes  "
          f"(zero specific movers: {int((spec == 0).sum()):,} = {(spec == 0).mean():.1%})")
    return kos, genes, M, spec


def fit_core(Mtr, k, seed=0):
    """k gene-space axes from the training perturbations. Returns V (k x genes), orthonormal rows."""
    from sklearn.utils.extmath import randomized_svd
    _, _, V = randomized_svd(Mtr, n_components=k, random_state=seed)
    return np.ascontiguousarray(V, np.float32)


def split_core(M, V):
    """core = projection onto the training axes; residual = what is left."""
    L = M @ V.T                      # perturbation loadings on the shared axes
    core = L @ V
    return core, M - core, L


def unit_rows(R):
    """Normalise once. Doing this inside the pair loop was materialising R[idx] for hundreds of thousands of
    pairs at 8,246 genes wide -- several GB per call, for a quantity that only needs one pass."""
    n = np.linalg.norm(R, axis=1)
    ok = n > 1e-9
    U = np.zeros_like(R)
    U[ok] = R[ok] / n[ok, None]
    return U, ok


def cos_sim(U, ok, ia, ib, chunk=4096):
    """Cosine similarity for the given index pairs, chunked so peak memory is chunk x n_genes, not n_pairs x
    n_genes."""
    ia, ib = np.asarray(ia, int), np.asarray(ib, int)
    if ia.size == 0:
        return np.array([])
    out = []
    for s in range(0, ia.size, chunk):
        a, b = ia[s:s + chunk], ib[s:s + chunk]
        m = ok[a] & ok[b]
        if m.any():
            out.append(np.einsum("ij,ij->i", U[a[m]], U[b[m]]))
    return np.concatenate(out) if out else np.array([])


def matched_pairs(pa, pb, by_dec, decile, rng, n_per=1):
    """Random pairs drawn to match the joint magnitude-decile composition of the real pairs -- the control that
    stops "complex partners are both loud responders" from masquerading as "complex partners are similar".
    Vectorised per decile: one rng call per decile per repeat instead of one per pair."""
    A, B = [], []
    for _ in range(n_per):
        for side, src in ((A, pa), (B, pb)):
            out = np.empty(src.size, int)
            for d, pool in by_dec.items():
                sel = decile[src] == d
                k = int(sel.sum())
                if k:
                    out[sel] = pool[rng.integers(0, pool.size, k)]
            side.append(out)
    a, b = np.concatenate(A), np.concatenate(B)
    keep = a != b
    return a[keep], b[keep]


def coherence(U, ok, pairs, decile, rng, tag, n_perm=N_PERM):
    """Complex-partner similarity vs magnitude-matched random pairs, with a permutation z-score."""
    if len(pairs) < 30:
        print(f"  [{tag}] only {len(pairs)} usable pairs -- refusing to report")
        return None
    pa = np.array([p[0] for p in pairs], int)
    pb = np.array([p[1] for p in pairs], int)
    by_dec = {d: np.where(decile == d)[0] for d in np.unique(decile)}
    by_dec = {d: v for d, v in by_dec.items() if v.size}
    real = cos_sim(U, ok, pa, pb)
    ma, mb = matched_pairs(pa, pb, by_dec, decile, rng, n_per=3)
    ctrl = cos_sim(U, ok, ma, mb)
    if real.size < 30 or ctrl.size < 30:
        print(f"  [{tag}] too few non-degenerate profiles -- refusing to report")
        return None
    obs = float(real.mean() - ctrl.mean())
    null = np.empty(n_perm)
    for i in range(n_perm):
        a2, b2 = matched_pairs(pa, pb, by_dec, decile, rng, n_per=1)
        s = cos_sim(U, ok, a2, b2)
        null[i] = s.mean() if s.size else np.nan
    null = null[np.isfinite(null)]
    z = float((real.mean() - null.mean()) / (null.std() + 1e-12)) if null.size > 10 else float("nan")
    print(f"  [{tag}] {real.size:,} complex pairs, mean cos {real.mean():+.4f}  |  "
          f"magnitude-matched {ctrl.mean():+.4f}  |  difference {obs:+.4f}  |  permutation z {z:+.2f}")
    return {"n_pairs": int(real.size), "mean_complex": float(real.mean()), "mean_matched": float(ctrl.mean()),
            "difference": obs, "perm_z": z, "n_perm": int(null.size)}


def main():
    kos, genes, M, spec = load()
    ki = {k: i for i, k in enumerate(kos)}
    n = M.shape[0]
    rng = np.random.default_rng(0)
    # md5 of the symbol, matching abstain.py. A first version used the big-endian value of the null-padded
    # 8-byte prefix mod 2, which is the parity of the LAST of those bytes -- and since almost every gene symbol
    # is shorter than 8 characters, that byte was the padding zero. The result was a 5,027 / 93 "half" split
    # that only became visible when a downstream file printed its test-set size.
    tr = np.array([int(hashlib.md5(k.encode()).hexdigest(), 16) % 2 == 0 for k in kos])
    print(f"split: {int(tr.sum()):,} train / {int((~tr).sum()):,} test perturbations")
    assert 0.4 < tr.mean() < 0.6, f"split is not a half ({tr.mean():.3f}) -- refusing to continue"

    # ---- B: how much of the response is shared, measured on held-out perturbations ----
    print("\nB. held-out share of squared response carried by a core fitted on the OTHER half:")
    tot = float((M[~tr] ** 2).sum())
    rank_tab = {}
    for k in RANKS:
        V = fit_core(M[tr], k)
        core, res, _ = split_core(M[~tr], V)
        f = float((core ** 2).sum() / tot)
        rank_tab[k] = f
        print(f"    k={k:>3d}   core {f*100:>5.1f}%   residual {100*(1-f):>5.1f}%")

    K = 20
    V = fit_core(M[tr], K)
    core, R, L = split_core(M, V)
    print(f"\n  using k={K} for everything below "
          f"(core {100*float((core**2).sum()/ (M**2).sum()):.1f}% of total squared response)")

    # ---- C: is component 1 a damage axis? ----
    l1 = np.abs(L[:, 0])
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    depf = {g["name"]: float(g.get("dep_frac") or 0) for g in D["genes"]}
    from scipy.stats import spearmanr
    r_mag = spearmanr(l1, spec).correlation
    have = [i for i, k in enumerate(kos) if k in depf]
    r_dep = spearmanr(l1[have], [depf[kos[i]] for i in have]).correlation
    freq = (np.abs(M) >= TAU).mean(0)
    w1 = np.abs(V[0])
    r_freq = spearmanr(w1, freq).correlation
    top1 = [genes[i] for i in np.argsort(-w1)[:15]]
    print("\nC. is component 1 a damage-magnitude axis?")
    print(f"    |loading| vs number of specific movers      Spearman {r_mag:+.3f}")
    print(f"    |loading| vs DepMap dependency of the gene  Spearman {r_dep:+.3f}  (n={len(have):,})")
    print(f"    |gene weight| vs how often that gene moves  Spearman {r_freq:+.3f}")
    print(f"    heaviest genes on component 1: {', '.join(top1)}")

    # ---- D: is the residual functionally structured? ----
    idx2name = {i: names[i] for i in range(len(names))}
    cplx = collections.defaultdict(set)
    for cname, members in (D.get("complexes") or {}).items():
        for gi in members:
            g = idx2name.get(gi)
            if g in ki:
                cplx[cname].add(ki[g])
    pairs = set()
    for cname, mem in cplx.items():
        m = sorted(mem)
        if not (2 <= len(m) <= 30):          # huge "complexes" are annotation dumps, not machines
            continue
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                pairs.add((m[i], m[j]))
    pairs = sorted(pairs)
    print(f"\nD. complex-partner coherence: {len(cplx):,} complexes touch a perturbed gene, "
          f"giving {len(pairs):,} within-complex perturbation pairs")

    mag = np.linalg.norm(M, axis=1)
    decile = np.digitize(mag, np.quantile(mag, np.linspace(0, 1, 11)[1:-1]))
    UM, okM = unit_rows(M)
    UR, okR = unit_rows(R)
    res = {}
    print("  responsive perturbations only (>= %d specific movers):" % MIN_SPEC)
    act = set(np.where(spec >= MIN_SPEC)[0].tolist())
    apairs = [p for p in pairs if p[0] in act and p[1] in act]
    for tag, U, ok in (("RAW matrix (core still in)", UM, okM), ("RESIDUAL (core removed)", UR, okR)):
        res[tag] = coherence(U, ok, apairs, decile, rng, tag)
    print("  all perturbations:")
    for tag, U, ok in (("RAW matrix, all", UM, okM), ("RESIDUAL, all", UR, okR)):
        res[tag] = coherence(U, ok, pairs, decile, rng, tag)

    # permuted membership: same complex sizes, wrong members
    sizes = [len(m) for m in cplx.values() if 2 <= len(m) <= 30]
    allp = sorted(act) if act else list(range(n))
    perm_pairs = []
    for s in sizes:
        m = rng.choice(allp, size=min(s, len(allp)), replace=False)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                perm_pairs.append((int(m[i]), int(m[j])))
    print("  CONTROL, permuted complex membership (sizes preserved):")
    res["PERMUTED membership (residual)"] = coherence(UR, okR, perm_pairs, decile, rng, "PERMUTED membership")

    ok = res.get("RESIDUAL (core removed)")
    raw = res.get("RAW matrix (core still in)")
    verdict = "FAIL"
    if ok and np.isfinite(ok["perm_z"]) and ok["perm_z"] >= 3 and ok["difference"] > 0:
        verdict = "PASS"
    print("\n" + "=" * 84)
    print("PRE-REGISTERED CRITERION: complex-partner residual similarity must exceed magnitude-matched pairs")
    print("with permutation z >= 3, and must SURVIVE removal of the shared core rather than vanish with it.")
    if ok and raw:
        print(f"  raw      difference {raw['difference']:+.4f}  z {raw['perm_z']:+.2f}")
        print(f"  residual difference {ok['difference']:+.4f}  z {ok['perm_z']:+.2f}")
        print(f"  the specific arm retains {100*ok['difference']/max(raw['difference'],1e-9):.0f}% of the raw "
              f"functional coherence after the shared core is removed")
    print(f"  VERDICT: {verdict}  -- build 4's premise (a separable, mechanistic specific arm) is "
          f"{'supported' if verdict == 'PASS' else 'NOT supported'}")
    print("=" * 84)

    np.savez_compressed(SP / "residual_K562_gwps.npz", kos=np.array(kos), genes=np.array(genes),
                        R=R.astype(np.float32), V=V, L=L.astype(np.float32), train=tr)
    json.dump({"n_perturbations": n, "n_genes": M.shape[1], "k": K, "rank_table": rank_tab,
               "component1": {"vs_n_movers": float(r_mag), "vs_dep_frac": float(r_dep),
                              "vs_move_frequency": float(r_freq), "top_genes": top1},
               "coherence": res, "verdict": verdict},
              open(OUT / "decompose.json", "w"), indent=1)
    print(f"\n  -> {OUT/'decompose.json'}   residual matrix -> {SP/'residual_K562_gwps.npz'}")


if __name__ == "__main__":
    main()
