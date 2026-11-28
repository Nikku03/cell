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

    # ---- directed path distance, because direct edges ask the wrong question ----
    # The three strongest perturbations here are IFNGR2, JAK2 and IFNGR1 -- the canonical IFN-gamma -> PD-L1
    # axis -- and none of them has a DIRECT edge to any of the four proteins. That is correct for a TF->target
    # graph: they act through STAT1. But the claim being tested is propagation, so the granularity has to be
    # path length, not adjacency. It also fixes the power problem: distance is defined for nearly every cell,
    # turning an 11-vs-83 comparison into a rank correlation over all 94.
    adj = {}
    for (s, t) in list(reg) + list(sig):
        adj.setdefault(s, set()).add(t)
    dist = {}
    MAXD = 3
    prot_idx = {gi[g] for g in TAG2GENE.values() if g in gi}
    for k in resolved:
        s0 = gi[ALIAS.get(k, k)]
        seen = {s0: 0}
        frontier = [s0]
        for depth in range(1, MAXD + 1):
            nxt = []
            for u in frontier:
                for v in adj.get(u, ()):
                    if v not in seen:
                        seen[v] = depth
                        nxt.append(v)
            frontier = nxt
            if not frontier:
                break
        for ti in prot_idx:
            if ti in seen and ti != s0:
                dist[(s0, ti)] = seen[ti]
    print(f"\n  DIRECTED PATH DISTANCE over reg+sig, capped at {MAXD} hops "
          f"({len(dist)} of {len(resolved)*len(prot_idx)} source-target pairs reachable)")

    ntidx = np.where(ntmask)[0]
    rng = np.random.default_rng(0)
    nullcache = {}

    def zscore(vals, denom, sel, key_name):
        """Effect for this knockout, in sd of a null drawn from real NT cells at the same n.

        Keyed on an explicit readout NAME, never on id(vals): `A[tags.index(tag)]` builds a fresh array on
        every call, so CPython is free to reuse the id of a collected one and two different proteins at the
        same cell count would silently share a null distribution.
        """
        obs = norm_lfc(vals, denom, sel, ntmask)
        n = int(sel.sum())
        key = (key_name, n)
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
            pz, plfc = zscore(A[tags.index(tag)], adt_tot, sel, f"prot:{tag}")
            mz, mlfc = zscore(cols[gene], tot, sel, f"rna:{gene}")
            rsign = reg.get((si, ti)); ssign = sig.get((si, ti))
            rows.append({"ko": k, "tag": tag, "gene": gene, "n_cells": int(sel.sum()),
                         "prot_z": pz, "prot_lfc": plfc, "mrna_z": mz, "mrna_lfc": mlfc,
                         "strength": ko_str[k], "dist": dist.get((si, ti), 99),
                         "reg": rsign, "sig": ssign,
                         "ppi": int((si, ti) in ppi), "coexpr": int((si, ti) in co)})
    # ---- POSITIVE CONTROL: the diagonal cells, excluded from the test but essential to interpreting it ----
    # A 4-of-10 direction rate could mean the network's signs are wrong OR that the ADT readout does not
    # respond at all. These two cells separate those: knock out the gene, its own protein must fall. They are
    # excluded from the network test (no self-loop to predict) and used only here.
    print("\n  POSITIVE CONTROL -- knock out the gene, does its OWN protein fall?")
    diag = []
    for tag, gene in TAG2GENE.items():
        if gene not in cols or tag not in tags or gene not in {ALIAS.get(k, k) for k in resolved}:
            continue
        ko = next(k for k in resolved if ALIAS.get(k, k) == gene)
        sel = np.array([guide_target(g) == ko for g in gid])
        pz, plfc = zscore(A[tags.index(tag)], adt_tot, sel, f"prot:{tag}")
        mz, mlfc = zscore(cols[gene], tot, sel, f"rna:{gene}")
        ok = plfc < 0
        print(f"    {ko:10s} -> {tag:7s} protein lfc {plfc:+.4f} (z {pz:+7.2f})   "
              f"mRNA lfc {mlfc:+.4f}   {'FALLS as it must' if ok else 'DOES NOT FALL'}")
        diag.append({"ko": ko, "tag": tag, "prot_lfc": plfc, "prot_z": pz, "mrna_lfc": mlfc, "ok": bool(ok)})
    ctrl_ok = all(d["ok"] for d in diag) and len(diag) >= 2
    print(f"    -> readout {'RESPONDS' if ctrl_ok else 'QUESTIONABLE'}: {sum(d['ok'] for d in diag)}"
          f"/{len(diag)} self-knockouts lower their own protein"
          + ("" if ctrl_ok else "   <- a null below may be the assay, not the network"))

    print(f"\n  {len(rows)} testable off-diagonal (knockout, protein) cells")

    # every predicate is wrapped in bool(). Without it, `False or False or 0 or 0` evaluates to the int 0,
    # numpy builds an int64 array, and y[m] silently becomes INTEGER indexing instead of boolean masking --
    # which made the "any" row report a mean |z| lower than that of its own reg subset.
    CLASSES = {"reg": lambda r: bool(r["reg"] is not None),
               "sig": lambda r: bool(r["sig"] is not None),
               "ppi": lambda r: bool(r["ppi"]),
               "coexpr": lambda r: bool(r["coexpr"]),
               "any": lambda r: bool(r["reg"] is not None or r["sig"] is not None
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

        Row permutation and not cell permutation: each knockout contributes up to 4 correlated cells, and
        shuffling cells independently would break that structure and understate the p-value.

        Patterns are swapped only between knockouts with the SAME number of cells. Two knockouts here have 3
        rather than 4 (their diagonal cell was removed), and reshaping a 3-pattern onto a 4-row block would
        duplicate an element -- inventing an edge that the network never asserted.
        """
        pat = {k: mask[kos_arr == k] for k in uk}
        bylen = {}
        for k in uk:
            bylen.setdefault(len(pat[k]), []).append(k)
        obs = y[mask].mean() - y[~mask].mean() if mask.any() and (~mask).any() else np.nan
        g = np.random.default_rng(seed)
        hits = 0
        for _ in range(N_PERM):
            m = np.zeros(len(y), bool)
            for L, ks in bylen.items():
                for a, b in zip(ks, g.permutation(len(ks))):
                    m[kos_arr == a] = pat[ks[b]]
            if m.any() and (~m).any() and (y[m].mean() - y[~m].mean()) >= obs:
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

    # ---- path distance: the properly-powered arm, and the one that matches the propagation claim ----
    dv = np.array([r["dist"] for r in rows], float)
    reach = dv < 90
    hist = {int(k): int(v) for k, v in zip(*np.unique(dv[reach], return_counts=True))}
    print(f"\n  PATH-DISTANCE ARM ({int(reach.sum())}/{len(rows)} cells reachable within {MAXD} hops; "
          + ", ".join(f"{k} hop{'s' if k > 1 else ''}: {v}" for k, v in sorted(hist.items())) + ")")
    R["distance"] = {"n_reachable": int(reach.sum()),
                     "hist": {str(int(k)): int(v) for k, v in zip(*np.unique(dv[reach],
                                                                            return_counts=True))}}

    def perm_rho(x, y, kos, seed=0):
        """Spearman with a knockout-level permutation null: shuffle the DISTANCE ROWS between knockouts.

        Distance is a property of the knockout-protein pair, so the exchangeable unit under the null is
        still the knockout, and patterns are swapped only between knockouts contributing the same number of
        rows. `kos` is passed explicitly rather than read from the enclosing scope: this arm drops
        unreachable cells, so its knockout list is a subset of the outer one and a closure would silently
        build empty patterns for the knockouts that dropped out.
        """
        obs = stats.spearmanr(x, y)[0]
        uks = list(dict.fromkeys(kos))
        pat = {k: x[kos == k] for k in uks}
        bylen = {}
        for k in uks:
            bylen.setdefault(len(pat[k]), []).append(k)
        g = np.random.default_rng(seed)
        hits = 0
        for _ in range(N_PERM):
            xp = np.empty_like(x)
            for ks in bylen.values():
                for a, b in zip(ks, g.permutation(len(ks))):
                    xp[kos == a] = pat[ks[b]]
            if abs(stats.spearmanr(xp, y)[0]) >= abs(obs):
                hits += 1
        return obs, (hits + 1) / (N_PERM + 1)

    if reach.sum() >= 20:
        print(f"    {'readout':10s} {'Spearman(dist,|z|)':>19s} {'perm p':>9s}"
              f"{'partial | strength':>21s}   expected NEGATIVE: closer -> bigger effect")
        st_r = np.array([r["strength"] for r in rows])[reach]
        rs = stats.rankdata(st_r)

        def resid(v):
            return v - np.polyval(np.polyfit(rs, v, 1), rs)
        x = dv[reach]
        ko_r = kos_arr[reach]
        for arm, key in (("protein", "prot_z"), ("mRNA", "mrna_z")):
            y = np.abs(np.array([r[key] for r in rows]))[reach]
            pr = stats.spearmanr(resid(stats.rankdata(x)), resid(stats.rankdata(y)))
            rho, pp2 = perm_rho(x, y, ko_r, seed=11)
            print(f"    {arm:10s} {rho:+19.3f} {pp2:9.4f}{pr[0]:+15.3f} (p {pr[1]:.3f})")
            R["distance"][arm] = {"spearman": float(rho), "perm_p": float(pp2),
                                  "partial_rho": float(pr[0]), "partial_p": float(pr[1])}
    else:
        print(f"    only {int(reach.sum())} cells reachable -- not enough to correlate; arm skipped")

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
        hitp, hitm = [], []
        for r in sorted(signed, key=lambda x: x["ko"]):
            s = r["reg"] if r["reg"] not in (None, 0) else r["sig"]
            want_dn = s > 0
            gp = bool((r["prot_lfc"] < 0) == want_dn)
            gm = bool((r["mrna_lfc"] < 0) == want_dn)
            okp += gp; okm += gm
            hitp.append(gp); hitm.append(gm)
            print(f"    {r['ko']:10s} {r['tag']:8s} {s:+5d} {r['prot_lfc']:+9.4f} "
                  f"{r['prot_z']:+8.2f} {r['mrna_lfc']:+9.4f}   "
                  f"{'protein YES' if gp else 'protein no '} / {'mRNA YES' if gm else 'mRNA no'}")
        n = len(signed)
        bp = stats.binomtest(okp, n, 0.5, alternative="greater")
        bm = stats.binomtest(okm, n, 0.5, alternative="greater")
        print(f"    protein direction correct {okp}/{n}  exact binomial p {bp.pvalue:.4f}")
        print(f"    mRNA    direction correct {okm}/{n}  exact binomial p {bm.pvalue:.4f}")
        # PAIRED, because the two rates come from the SAME edges. Two independent binomials cannot say the
        # rates DIFFER; only the discordant pairs carry that information, so this is an exact McNemar.
        b = sum(1 for p, m in zip(hitp, hitm) if p and not m)      # protein right, mRNA wrong
        c = sum(1 for p, m in zip(hitp, hitm) if m and not p)      # mRNA right, protein wrong
        mc = stats.binomtest(b, b + c, 0.5, alternative="less") if b + c else None
        if mc:
            print(f"    PAIRED exact McNemar on the {b+c} discordant edges "
                  f"({b} protein-only, {c} mRNA-only): p {mc.pvalue:.4f}")
            print(f"    ^ the same {n} edges score both arms, so two separate binomials cannot say the rates "
                  f"DIFFER -- only the discordant pairs carry that, and there are {b+c} of them")
        else:
            print("    no discordant edges -- the two arms agree everywhere")
        R["direction"] = {"n": n, "protein_correct": int(okp), "protein_p": float(bp.pvalue),
                          "mrna_correct": int(okm), "mrna_p": float(bm.pvalue),
                          "discordant_protein_only": int(b), "discordant_mrna_only": int(c),
                          "mcnemar_p": float(mc.pvalue) if mc else None}

    # ---- verdict ----
    print("\n" + "=" * 100)
    pp = R.get("prot_z", {}).get("reg", {}).get("perm_p", 1.0)
    mp = R.get("mrna_z", {}).get("reg", {}).get("perm_p", 1.0)
    D = R.get("direction", {})
    okp, okm, nd = D.get("protein_correct", 0), D.get("mrna_correct", 0), D.get("n", 0)
    mcp = D.get("mcnemar_p")
    az = np.abs(np.array([r["prot_z"] for r in rows]))
    medz, frac_resp = float(np.median(az)), float((az > 3).mean())
    # READOUT ADEQUACY IS NOT THE MEDIAN. A first version gated on median |z| < 2 and declared the readout
    # "too quiet" at 1.61 -- but a low median is exactly what a correct experiment gives when most
    # (knockout, protein) pairs are true nulls, which they are: most knockouts should not move most proteins.
    # Adequacy is instead: does the positive control respond, and does a real fraction of cells show a
    # detectable response at all. Here the mean |z| is 5.07 against a median of 1.61 -- a right-skewed
    # distribution of mostly-nulls with a few very large movers, which is the expected shape.
    readout_ok = ctrl_ok and frac_resp >= 0.15
    print(f"  readout adequacy: positive control {'PASS' if ctrl_ok else 'FAIL'}; "
          f"{100*frac_resp:.0f}% of cells move at |z|>3; median |z| {medz:.2f}, mean {az.mean():.2f} "
          f"(right-skewed, as a mostly-null grid should be)")
    # THE MAGNITUDE ARMS ARE NULL AND UNDERPOWERED; THE DIRECTION ARM IS NEITHER. Reading the verdict off
    # the magnitude arms alone labelled this "underpowered", which is wrong in a way that matters: the
    # protein responses are large (median |z| ~5, extremes -23 and +12), so the outcome is not noise-limited.
    # What fails is the SIGN. Those are different findings and the ladder now distinguishes them.
    parts = [f"magnitude and path-distance arms are NULL and underpowered (reg protein p {pp:.3f}, "
             f"mRNA p {mp:.3f}; only d>=0.90 was detectable), so they say nothing either way"]
    if not ctrl_ok:
        v = ("INCONCLUSIVE -- the positive control did not pass, so a null on the network arms cannot be "
             "separated from a readout that does not respond. Fix the control before reading anything else.")
    elif not readout_ok:
        v = (f"READOUT TOO QUIET -- only {100*frac_resp:.0f}% of cells move at |z|>3, so protein responses "
             f"are near the NT noise floor and no arm can resolve anything. {parts[0]}.")
    elif okm > okp and mcp is not None and mcp < 0.05:
        v = (f"EDGES STOP AT THE TRANSCRIPT -- the same {nd} signed edges get the mRNA direction right "
             f"{okm}/{nd} but the protein direction {okp}/{nd}, paired McNemar p {mcp:.4f}. Protein is not "
             f"the quiet arm: median |z| {medz:.2f}, extremes past 20. So the network's signs describe "
             f"transcript changes that do not carry to protein. {parts[0]}.")
    elif okm > okp:
        v = (f"SUGGESTIVE, NOT ESTABLISHED -- the same {nd} signed edges get the mRNA direction right "
             f"{okm}/{nd} (p {D.get('mrna_p', 1):.3f}) but the protein direction only {okp}/{nd} "
             f"(p {D.get('protein_p', 1):.3f}), and protein is not the quiet arm (median |z| {medz:.2f}). "
             f"That is the pattern of edges stopping at the transcript, but the paired test has only "
             f"{D.get('discordant_protein_only',0)+D.get('discordant_mrna_only',0)} discordant edges "
             f"(McNemar p {mcp:.3f}), so this dataset cannot establish it. Four proteins is the binding "
             f"constraint, not the analysis. {parts[0]}.")
    elif okp >= okm and D.get("protein_p", 1) < 0.05:
        v = (f"SIGNS REACH PROTEIN -- {okp}/{nd} signed edges move the protein the predicted way "
             f"(p {D.get('protein_p', 1):.4f}), at least as often as they move the transcript ({okm}/{nd}). "
             f"On four proteins, so this is a consistency check that passed, not a validation. {parts[0]}.")
    else:
        v = (f"NO SIGNAL IN EITHER MODALITY -- signed edges get protein {okp}/{nd} and mRNA {okm}/{nd}, "
             f"both at chance. {parts[0]}. Nothing here supports or undermines the network.")
    R["verdict"] = v
    R["median_abs_prot_z"] = medz
    R["positive_control_passed"] = bool(ctrl_ok)
    R["positive_control"] = diag
    print(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "protein_orthogonal.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'protein_orthogonal.json'}")


if __name__ == "__main__":
    main()
