"""DOES A NETWORK EDGE PREDICT A PROTEIN CHANGE? The network tested against a modality it was not built from.

WHY THIS IS THE TEST WORTH RUNNING. `mrna_protein_gate.py` established that per-gene mRNA-protein coupling is
predictable but only as a per-node CONFIDENCE (biology explains 2.3% of the measurability residual, ridge
floor 0.5%). So protein counts cannot be written onto the network's nodes. What remains -- and it is the more
valuable half -- is using protein as an INDEPENDENT CHECK on edges that were derived from mRNA. An edge that
predicts a transcript change but not a protein change describes something that does not reach the protein
layer, and that is a fact about the network worth knowing.

THE DATA. Papalexi ECCITE-seq: 20,729 THP-1 cells, 24 knocked-out genes resolvable in the network, and four
antibody-derived tags (CD86, PD-L1, PD-L2, CD366) measured in the SAME cells as the transcriptome. The
barcode join is asserted exact, not assumed -- `mrna_vs_protein.py` established 20,729/20,729 and this module
imports that loader rather than re-deriving it.

FOUR PROTEINS IS THIN AND THAT GOES IN THE RESULT, NOT AFTER IT. The universe is 24 KOs x 4 proteins = 96
cells, minus the 2 where the knocked-out gene IS the protein's own gene. Those two are excluded rather than
counted: the network holds no self-loops, so it cannot predict them, and the effect is guaranteed -- scoring
them would be handing the network two free wins it never made a claim about. 94 cells remain, 17 carrying an
edge.

EDGE CLASSES ARE KEPT SEPARATE, and two of them are too sparse to test:
    reg      directed, signed      11 of 94 cells   testable
    ppi      undirected             6 of 94 cells   testable, barely
    sig      directed, signed        2 of 94 cells   REPORTED AS UNTESTABLE
    coexpr   undirected, mRNA-derived 1 of 94 cells  REPORTED AS UNTESTABLE
Pooling them to reach a workable n would be manufacturing power, and the coexpression arm -- the one that
would most sharply test "does an mRNA-derived edge reach protein" -- is exactly the one with a single cell.

THE CONFOUND THAT DECIDES WHETHER THIS IS A TEST AT ALL. A knockout that shifts the whole transcriptome moves
every protein. The network's predicted regulators of immune proteins here are STAT1, IRF1, STAT3, SPI1 --
strong perturbations by construction. So an edge-vs-no-edge difference could be entirely "edges point at
genes whose knockout does a lot". Global perturbation strength is therefore measured per knockout and the
comparison is made at matched strength, not marginally.

THE NULL IS BUILT FROM REAL CONTROL CELLS, AT MATCHED n. There are 9 non-targeting guides and 2,386 NT cells.
For a knockout measured in n cells, the null is drawn by sampling n NT cells and scoring them against the
remaining NT cells, repeatedly. That carries the real per-protein noise and removes the artefact where large
knockout groups look significant because their means are better estimated.

THE P-VALUE IS PERMUTED AT THE KNOCKOUT LEVEL, NOT THE CELL LEVEL. The 94 cells are not 94 independent
observations: each knockout appears four times, and a strong knockout moves all four proteins together.
A Mann-Whitney over 17 vs 77 would ignore that and be anticonservative. The primary test permutes which
knockout carries which edge-pattern, preserving each knockout's row profile and the clustering with it. The
naive test is printed beside it, labelled, as evidence of how much the clustering matters.

DIRECTION IS A SEPARATE AND STRONGER TEST. 10 of the edges carry a sign. Knocking out an activator should
LOWER its target's protein. That prediction needs no matching and no magnitude calibration -- it is an exact
binomial on 10 trials, which is weak but clean.
"""
import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
NET = SP / "cell_complete.json.gz"
H5 = SP / "papalexi.h5ad"

TAG2GENE = {"CD86": "CD86", "PDL1": "CD274", "PDL2": "PDCD1LG2", "CD366": "HAVCR2"}
# MARCH8 is MARCHF8 in the network's gene universe. Without the alias the knockout is silently dropped and
# the universe shrinks from 24 KOs to 23 with no warning -- which is how a 4% data loss goes unnoticed.
ALIAS = {"MARCH8": "MARCHF8"}
MIN_CELLS = 25
NULL_DRAWS = int(os.environ.get("PO_NULL", 400))
N_PERM = int(os.environ.get("PO_PERM", 20000))
MIN_CELLS_CLASS = 5          # an edge class with fewer cells than this is declared untestable, not pooled


def load_network():
    """Edge lookups keyed by (source, target) gene index, one dict per class."""
    if not NET.exists():
        raise SystemExit(f"network absent at {NET}")
    d = json.load(gzip.open(NET, "rt"))
    genes = [g["name"] for g in d["genes"]]
    gi = {g: i for i, g in enumerate(genes)}
    reg, sig = {}, {}
    for s, t, x in d["reg"]:
        reg[(s, t)] = x
    for s, t, x in d["sig"]:
        sig[(s, t)] = x
    ppi = set()
    for a, b in d["ppi"]:
        ppi.add((a, b)); ppi.add((b, a))
    co = set()
    for k, v in d["coexpr"].items():
        k = int(k)
        for x in v:
            co.add((k, int(x[0]))); co.add((int(x[0]), k))
    print(f"  network: {len(genes):,} genes, reg {len(reg):,}, sig {len(sig):,}, "
          f"ppi {len(ppi)//2:,}, coexpr {len(co)//2:,}")
    return gi, reg, sig, ppi, co


def guide_target(g):
    return None if g.startswith("NT") else re.sub(r"[-_]?g?\d+$", "", g)


def group_pseudobulk(labels, ngroups):
    """Per-group summed counts over every gene, in one pass over the CSC matrix.

    Needed for global perturbation strength, which cannot be computed from the four ADT genes alone. Done
    with bincount over a combined (group, gene) index rather than np.add.at, which is ~50x slower on 69M
    nonzeros, and chunked over columns so the combined index never materialises for the whole matrix.
    """
    import h5py
    with h5py.File(H5, "r") as h:
        X = h["X"]
        ncell, ngene = tuple(X.attrs["shape"])
        indptr = X["indptr"][:]
        G = np.zeros((ngroups, ngene), np.float64)
        CH = 512
        for s in range(0, ngene, CH):
            e = min(s + CH, ngene)
            a, b = int(indptr[s]), int(indptr[e])
            if b <= a:
                continue
            idx = X["indices"][a:b]
            val = X["data"][a:b]
            col = np.repeat(np.arange(e - s), np.diff(indptr[s:e + 1]).astype(np.int64))
            lab = labels[idx]
            ok = lab >= 0
            if not ok.any():
                continue
            comb = lab[ok].astype(np.int64) * (e - s) + col[ok]
            acc = np.bincount(comb, weights=val[ok], minlength=ngroups * (e - s))
            G[:, s:e] += acc.reshape(ngroups, e - s)
    return G


def global_strength(G, ncells, ref):
    """How much does this knockout move the whole transcriptome, in an n-aware way?

    An lfc-norm would reward small groups: fewer cells means a noisier lfc means a bigger norm, so the
    covariate would encode cell count and the matching would be matching on the wrong thing. Instead each
    gene gets a Poisson-variance z against the reference, and strength is the count of genes past |z|>3.
    The n-independence of this is not assumed -- it is checked against the 9 NT guides and printed.
    """
    tot = G.sum(1, keepdims=True)
    rate = ref / max(ref.sum(), 1.0)
    exp = tot * rate                                        # expected counts under the NT composition
    z = (G - exp) / np.sqrt(np.maximum(exp, 1.0))
    return (np.abs(z) > 3).sum(1).astype(float)


def main():
    from scipy import stats
    from mrna_vs_protein import load_adt, load_rna
    print("=" * 100)
    print("DOES A NETWORK EDGE PREDICT A PROTEIN CHANGE? -- Papalexi ECCITE-seq, 4 proteins")
    print("=" * 100)

    gi, reg, sig, ppi, co = load_network()
    cells, tags, A = load_adt()
    genes_needed = sorted(set(TAG2GENE.values()))
    bc, gid, cols, tot = load_rna(genes_needed)
    assert len(bc) == len(cells), f"cell count mismatch {len(bc)} vs {len(cells)}"
    agree = sum(1 for a, b in zip(cells, bc) if a == b)
    assert agree == len(bc), "barcode join is not exact -- refusing to interpret anything downstream"
    print(f"  join asserted exact: {agree:,}/{len(bc):,} cells")

    adt_tot = A.sum(0)
    ntmask = np.array([g.startswith("NT") for g in gid])
    ntguides = sorted({g for g in gid if g.startswith("NT")})
    kos = sorted({guide_target(g) for g in set(gid) if guide_target(g)})
    resolved = [k for k in kos if ALIAS.get(k, k) in gi]
    print(f"  {int(ntmask.sum()):,} NT cells across {len(ntguides)} NT guides; "
          f"{len(kos)} knocked-out genes, {len(resolved)} resolvable in the network "
          f"(unresolved {[k for k in kos if k not in resolved]})")

    # ---- global perturbation strength, one pass ----
    lab = np.full(len(gid), -1, np.int64)
    order = resolved + ntguides
    for i, k in enumerate(resolved):
        lab[np.array([guide_target(g) == k for g in gid])] = i
    for j, ng in enumerate(ntguides):
        lab[gid == ng] = len(resolved) + j
    G = group_pseudobulk(lab, len(order))
    nc = np.array([(lab == i).sum() for i in range(len(order))], float)
    ref = G[len(resolved):].sum(0)
    strength = global_strength(G, nc, ref)
    ko_str = {k: strength[i] for i, k in enumerate(resolved)}
    nt_str = strength[len(resolved):]
    nt_n = nc[len(resolved):]
    rn = stats.spearmanr(nt_n, nt_str)
    print(f"\n  GLOBAL PERTURBATION STRENGTH (genes past |z|>3 vs the NT composition)")
    print(f"    NT guides (the floor): {' '.join(f'{x:.0f}' for x in np.sort(nt_str))}")
    print(f"    strength vs cell count across the 9 NT guides: Spearman {rn[0]:+.3f} (p {rn[1]:.2f})"
          f"  <- if this were strong, the covariate would be encoding n, not perturbation")
    for k in sorted(ko_str, key=lambda x: -ko_str[x])[:6]:
        print(f"      {k:10s} {ko_str[k]:6.0f}")
    print(f"      ... median KO {np.median(list(ko_str.values())):.0f}, "
          f"median NT {np.median(nt_str):.0f}")

    # ---- per (KO, protein) effect, z-scored against matched-n NT draws ----
    def norm_lfc(vals, denom, m1, m0):
        n = vals / np.maximum(denom, 1.0)
        a, b = n[m1].mean(), n[m0].mean()
        return np.nan if b <= 0 else float(np.log2((a + 1e-12) / (b + 1e-12)))

    ntidx = np.where(ntmask)[0]
    rng = np.random.default_rng(0)
    nullcache = {}

    def zscore(vals, denom, sel):
        """Effect for this knockout, in sd of a null drawn from real NT cells at the same n."""
        obs = norm_lfc(vals, denom, sel, ntmask)
        n = int(sel.sum())
        key = (id(vals), n)
        if key not in nullcache:
            draws = []
            for _ in range(NULL_DRAWS):
                pick = rng.choice(ntidx, size=min(n, len(ntidx) - 20), replace=False)
                m1 = np.zeros(len(gid), bool); m1[pick] = True
                m0 = ntmask & ~m1
                v = norm_lfc(vals, denom, m1, m0)
                if np.isfinite(v):
                    draws.append(v)
            nullcache[key] = (float(np.mean(draws)), float(np.std(draws) + 1e-12))
        mu, sd = nullcache[key]
        return (obs - mu) / sd, obs

    rows = []
    for k in resolved:
        si = gi[ALIAS.get(k, k)]
        sel = np.array([guide_target(g) == k for g in gid])
        if sel.sum() < MIN_CELLS:
            continue
        for tag, gene in TAG2GENE.items():
            if tag not in tags or gene not in cols or gene not in gi:
                continue
            ti = gi[gene]
            if si == ti:
                continue                     # diagonal: no self-loop to test, effect guaranteed
            pz, plfc = zscore(A[tags.index(tag)], adt_tot, sel)
            mz, mlfc = zscore(cols[gene], tot, sel)
            rsign = reg.get((si, ti)); ssign = sig.get((si, ti))
            rows.append({"ko": k, "tag": tag, "gene": gene, "n_cells": int(sel.sum()),
                         "prot_z": pz, "prot_lfc": plfc, "mrna_z": mz, "mrna_lfc": mlfc,
                         "strength": ko_str[k],
                         "reg": rsign, "sig": ssign,
                         "ppi": int((si, ti) in ppi), "coexpr": int((si, ti) in co)})
    print(f"\n  {len(rows)} testable off-diagonal (knockout, protein) cells")

    CLASSES = {"reg": lambda r: r["reg"] is not None,
               "sig": lambda r: r["sig"] is not None,
               "ppi": lambda r: bool(r["ppi"]),
               "coexpr": lambda r: bool(r["coexpr"]),
               "any": lambda r: (r["reg"] is not None or r["sig"] is not None
                                 or r["ppi"] or r["coexpr"])}
    counts = {c: sum(1 for r in rows if f(r)) for c, f in CLASSES.items()}
    print("  edge coverage per class:")
    for c, n in counts.items():
        note = "" if n >= MIN_CELLS_CLASS else "   <- UNTESTABLE, reported not pooled"
        print(f"    {c:8s} {n:3d}/{len(rows)}{note}")

    R = {"n_cells": len(rows), "n_kos": len(resolved), "counts": counts,
         "nt_strength_vs_n_spearman": float(rn[0]), "rows": rows,
         "null_draws": NULL_DRAWS, "n_perm": N_PERM}

    # ---- POWER, stated before any result ----
    print(f"\n  POWER, BEFORE THE RESULT (n varies by class; {N_PERM:,} knockout-level permutations)")
    for c, n1 in counts.items():
        if n1 < MIN_CELLS_CLASS:
            continue
        n0 = len(rows) - n1
        # smallest p a permutation test can return, and the Cohen's d detectable at 80% power
        d80 = 2.8 * np.sqrt(1 / n1 + 1 / n0)
        print(f"    {c:8s} {n1:3d} vs {n0:3d}   detectable at 80% power: d >= {d80:.2f}"
              f"   ({'large' if d80 > 0.8 else 'moderate'} effects only)")
        R.setdefault("power", {})[c] = {"n_edge": n1, "n_no_edge": n0, "d80": float(d80)}

    # ---- the two arms, with knockout-level permutation ----
    kos_arr = np.array([r["ko"] for r in rows])
    uk = list(dict.fromkeys(kos_arr))

    def perm_test(mask, y, seed=0):
        """Permute which knockout carries which edge-pattern; keep the measurement fixed.

        Row permutation and not cell permutation: each knockout contributes 4 correlated cells, and
        shuffling cells independently would break that structure and understate the p-value.
        """
        pat = {k: mask[kos_arr == k] for k in uk}
        obs = y[mask].mean() - y[~mask].mean() if mask.any() and (~mask).any() else np.nan
        g = np.random.default_rng(seed)
        hits = 0
        for _ in range(N_PERM):
            pk = g.permutation(len(uk))
            m = np.zeros(len(y), bool)
            for a, b in zip(uk, pk):
                sl = kos_arr == a
                src = pat[uk[b]]
                m[sl] = src[:sl.sum()] if len(src) >= sl.sum() else np.resize(src, sl.sum())
            if m.any() and (~m).any():
                if (y[m].mean() - y[~m].mean()) >= obs:
                    hits += 1
        return obs, (hits + 1) / (N_PERM + 1)

    for arm, key in (("PROTEIN -- the independent modality", "prot_z"),
                     ("mRNA -- the modality the network came from", "mrna_z")):
        print(f"\n  {arm}")
        print(f"    {'class':8s} {'|z| edge':>9s} {'|z| no-edge':>12s} {'diff':>8s} "
              f"{'perm p':>9s} {'MWU p':>9s}   {'partial rho | strength':>24s}")
        y = np.abs(np.array([r[key] for r in rows]))
        st = np.array([r["strength"] for r in rows])
        for c, f in CLASSES.items():
            if counts[c] < MIN_CELLS_CLASS:
                continue
            m = np.array([f(r) for r in rows])
            obs, p = perm_test(m, y, seed=hash(c) % 2**31)
            mwu = stats.mannwhitneyu(y[m], y[~m], alternative="greater")[1]
            # partial Spearman: rank-residualise both on global strength, then correlate
            ry = stats.rankdata(y); rs = stats.rankdata(st); rm = stats.rankdata(m.astype(float))
            def resid(v):
                b = np.polyfit(rs, v, 1)
                return v - np.polyval(b, rs)
            pr = stats.spearmanr(resid(rm), resid(ry))
            print(f"    {c:8s} {y[m].mean():9.3f} {y[~m].mean():12.3f} {obs:+8.3f} "
                  f"{p:9.4f} {mwu:9.4f}   {pr[0]:+10.3f} (p {pr[1]:.3f})")
            R.setdefault(key, {})[c] = {"mean_edge": float(y[m].mean()),
                                        "mean_no_edge": float(y[~m].mean()), "diff": float(obs),
                                        "perm_p": float(p), "mwu_p": float(mwu),
                                        "partial_rho": float(pr[0]), "partial_p": float(pr[1])}
        print("    ^ MWU treats 94 cells as independent; they are 24 knockouts x 4 proteins, so the "
              "permuted p is the one to read")

    # ---- the confound, checked directly ----
    ys = np.abs(np.array([r["prot_z"] for r in rows]))
    st = np.array([r["strength"] for r in rows])
    cs = stats.spearmanr(st, ys)
    print(f"\n  IS THE CONFOUND REAL? global strength vs |protein z|: Spearman {cs[0]:+.3f} "
          f"(p {cs[1]:.3g})")
    for c in ("reg", "any"):
        if counts[c] >= MIN_CELLS_CLASS:
            m = np.array([CLASSES[c](r) for r in rows])
            u = stats.mannwhitneyu(st[m], st[~m], alternative="two-sided")
            print(f"    strength of {c:4s}-edge knockouts {np.median(st[m]):7.0f} vs "
                  f"{np.median(st[~m]):7.0f} no-edge  (MWU p {u[1]:.3f})"
                  f"{'   <- edges DO point at stronger perturbations' if u[1] < 0.1 else ''}")
    R["confound"] = {"strength_vs_prot_z_spearman": float(cs[0]), "p": float(cs[1])}

    # ---- direction: the strong, assumption-free test ----
    signed = [r for r in rows if (r["reg"] not in (None, 0)) or (r["sig"] not in (None, 0))]
    print(f"\n  DIRECTION -- knocking out an activator should LOWER its target's protein "
          f"({len(signed)} signed edges)")
    if signed:
        print(f"    {'KO':10s} {'protein':8s} {'sign':>5s} {'prot lfc':>9s} {'prot z':>8s} "
              f"{'mRNA lfc':>9s}   as predicted?")
        okp = okm = 0
        for r in sorted(signed, key=lambda x: x["ko"]):
            s = r["reg"] if r["reg"] not in (None, 0) else r["sig"]
            want_dn = s > 0
            gp = (r["prot_lfc"] < 0) == want_dn
            gm = (r["mrna_lfc"] < 0) == want_dn
            okp += gp; okm += gm
            print(f"    {r['ko']:10s} {r['tag']:8s} {s:+5d} {r['prot_lfc']:+9.4f} "
                  f"{r['prot_z']:+8.2f} {r['mrna_lfc']:+9.4f}   "
                  f"{'protein YES' if gp else 'protein no '} / {'mRNA YES' if gm else 'mRNA no'}")
        n = len(signed)
        bp = stats.binomtest(okp, n, 0.5, alternative="greater")
        bm = stats.binomtest(okm, n, 0.5, alternative="greater")
        print(f"    protein direction correct {okp}/{n}  exact binomial p {bp.pvalue:.4f}")
        print(f"    mRNA    direction correct {okm}/{n}  exact binomial p {bm.pvalue:.4f}")
        R["direction"] = {"n": n, "protein_correct": int(okp), "protein_p": float(bp.pvalue),
                          "mrna_correct": int(okm), "mrna_p": float(bm.pvalue)}

    # ---- verdict ----
    print("\n" + "=" * 100)
    pp = R.get("prot_z", {}).get("reg", {}).get("perm_p", 1.0)
    mp = R.get("mrna_z", {}).get("reg", {}).get("perm_p", 1.0)
    pd_ = R.get("prot_z", {}).get("reg", {}).get("diff", 0.0)
    md = R.get("mrna_z", {}).get("reg", {}).get("diff", 0.0)
    dirp = R.get("direction", {}).get("protein_p", 1.0)
    if pp < 0.05 and dirp < 0.05:
        v = (f"EDGES REACH PROTEIN -- regulatory edges predict protein movement (diff {pd_:+.3f}, "
             f"permuted p {pp:.4f}) and their signs are right (p {dirp:.4f}). On four proteins, so this "
             f"is a consistency check that passed, not a proteome-wide validation.")
    elif pp < 0.05:
        v = (f"MAGNITUDE ONLY -- edges predict THAT the protein moves (diff {pd_:+.3f}, p {pp:.4f}) but "
             f"not reliably which way (direction p {dirp:.4f}). Usable to rank blast radius, not to "
             f"predict the sign of a response.")
    elif mp < 0.05 and pp >= 0.05:
        v = (f"EDGES STOP AT THE TRANSCRIPT -- they predict mRNA movement (diff {md:+.3f}, p {mp:.4f}) but "
             f"not protein movement (diff {pd_:+.3f}, p {pp:.4f}). The honest reading of an mRNA-derived "
             f"network: its edges describe transcript changes that do not demonstrably reach protein. "
             f"With 11 edge cells this is underpowered for anything but a large effect, so it is evidence "
             f"of absence only to the extent the power table above allows.")
    else:
        v = (f"UNDERPOWERED -- neither arm separates (protein p {pp:.4f}, mRNA p {mp:.4f}). With 11 "
             f"regulatory-edge cells out of 94 and only large effects detectable, this is a null result "
             f"about the test's resolution, not about the network. It should not be quoted either way.")
    R["verdict"] = v
    print(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "protein_orthogonal.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'protein_orthogonal.json'}")


if __name__ == "__main__":
    main()
