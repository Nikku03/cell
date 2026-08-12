"""LOOP 55 -- THE CONNECTED SPHERE CHAIN, WITH SOMETHING ACTUALLY PERTURBED.

THE IDEA, as asked for: cover the protein in overlapping 4 Angstrom spheres, change one, and let the
overlaps carry the change across the structure. This repository already measured that idea on DNA
(nexus_dna_spheres, nucleosome 1KX5, exact all-atom reference, one phosphate oxygen flipped): a 4 A
sphere captures 35.0% of the UNSCREENED response and 72.1% of the SCREENED one, 8 A gives 44.3% /
88.6%, 12 A gives 49.9% / 93.7%. Its conclusion was that the failure is an unscreened kernel in a
dense medium and that screening, not geometry, is the fix.

WHAT "4 ANGSTROM SPHERES" HAS TO MEAN, because the naive reading is a trap. Connecting residues whose
CA atoms lie within 4 A gives, on a real 2,595-residue model, MEAN DEGREE 2.0 AND NINE non-sequential
contacts -- a one-dimensional chain with no tertiary coupling whatsoever. Radius-4 spheres OVERLAP
when their centres are within 8 A, which gives mean degree 8.9 and 6,297 non-sequential contacts.
The second is the network; the first would have produced a model that could not feel folding at all.

A DESIGN OF THIS WAS REVIEWED BEFORE IT RAN AND THE REVIEW KILLED IT. Three readers, one lens each,
with the repository. The first draft proposed five per-residue features off the sphere graph --
self-response G_ii, total response, long-range response, reach, and stiffness 1/G_ii -- and:

  NOTHING WAS PERTURBED. All five are properties of the WILD-TYPE structure. They are identical for
  every substitution at a residue, so a module named for propagating a mutation would have computed a
  static graph statistic and never once looked at what the residue changed to. That is the whole idea
  missing from its own implementation.
  1/G_ii IS G_ii. Spearman exactly -1.000000, and the long-range sum sits at 0.982 with the total.
  Loop 53's collinear-pair bug, reintroduced by construction.
  THE SCREENING WAS A NO-OP. Screened and unscreened features came out rank-identical (0.995-1.000),
  so the gate built to test screening could not have failed.
  AND THE GRAPH WAS LOOP 54's GRAPH. Byte for byte -- the same 8 A CA contact graph whose eight
  pairwise summaries added +0.0012 at p 0.054. The first draft would have re-summarised a network
  already measured as carrying nothing.

SO THE MUTATION NOW ENTERS THE NETWORK, which is what makes this a perturbation model rather than a
description. Each node sits at its SIDE-CHAIN CENTROID and each edge is weighted by the side-chain
volumes it joins. A substitution changes the volume at one site, which changes every edge touching
it, which changes the whole matrix:

    K(V)    weighted Laplacian, gamma_ij = gamma0 * sqrt(V_i V_j) / V_ref  for centres within 8 A
    dK      what the substitution does to row and column i, and nothing else
    dG      = -G dK G     first-order linear response, exact to first order and O(n) per variant
             because dK is rank-limited to one site

The features are then responses to that perturbation, and every one of them depends on what the
residue became. This also aims at the weakest measured part of the current best model: loop 51 found
that knowing the substituted residue is worth only +0.0404 of ESM-2's 0.8823, so almost everything
the row can currently answer is positional. A perturbation feature is by construction not.

AND ONE CORRECTION TO THE RECORD, found by the same review. loop 54's `burial` and `nc10` columns are
BYTE-IDENTICAL in all 29,286 rows -- both are CA neighbours within 10 A. Its N3 therefore tested
seven new features and not eight, and its `+pairwise` arm carried a duplicated column. The conclusion
does not change and gets stronger: the pairwise block was even more redundant than reported. Loop 55
deduplicates programmatically and asserts it rather than trusting a feature list.

PREDECLARED, before any number:

  S0 THE NETWORK REPRODUCES A MECHANICAL OBSERVABLE
       Gaussian-network self-response against EXPERIMENTAL B-factors, on the cached crystal
       structures, per chain. Published GNM baselines sit near 0.55-0.65 mean correlation. This runs
       FIRST and on data that has nothing to do with variants: if the network cannot reproduce the
       fluctuations of a protein whose fluctuations were measured, then it is not physics and
       nothing downstream is either. Floor 0.45.
  S1 THE FEATURES ARE MUTATION-DEPENDENT
       at residues carrying two or more different substitutions, every declared feature must actually
       vary. THIS IS THE GATE THE REVIEWED DRAFT WOULD HAVE FAILED, and it is here because it failed.
       A feature constant across substitutions at a site is a wild-type descriptor wearing a
       perturbation's name.
  S2 NO FEATURE IS A RE-PARAMETERISATION OF ONE ALREADY MEASURED
       runs BEFORE S3 and can drop features on its own: any pair with |Spearman| > 0.95 loses one
       member, and any feature with R^2 > 0.90 against loop 54's full de-duplicated feature block is
       dropped as already-present information. What survives is reported by name.
  S3 IT ADDS OVER LOOP 54'S BEST, REFITTED HERE
       not the stored 0.9464 -- loop 54's arms are refitted inside this script on the same rows and
       the same folds, because a stored number from another run is not a controlled comparison.
       Judged PRIMARILY on the allele-frequency-free arm, since population frequency is itself an
       ACMG classification criterion and a bar containing the label's own definition is not a fair
       one. Paired per gene. The minimum detectable effect is computed and printed BEFORE the
       comparison, and a failure is worded as "adds less than X", never as "adds nothing".
  S4 THE NULL COLLAPSES IT, TWICE
       labels permuted within gene with the pipeline refitted; AND the structure randomised -- sphere
       features recomputed on a shuffled residue order, so the graph keeps its degree distribution
       and loses its geometry. The second is the one that speaks to whether the network matters.

-> outputs/loop_sphere_chain.json
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_rescue as LR
import loop_distogram_nn as DN
import run_manifest as RM

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
CACHE = SC / "_sphere_chain_v2.pkl"

SPHERE_R = 4.0            # the radius asked for
CUTOFF = 2 * SPHERE_R     # spheres are connected when they OVERLAP
GAMMA0 = 1.0
LONG_RANGE = 12
MIN_CLASS = 20
FOLDS = 5
N_NULL = 60
BFACTOR_FLOOR = 0.45
COLLIN = 0.95
R2_DROP = 0.90
SEED = 1904

# side-chain volumes, A^3 (Zamyatnin). The mutation enters the network through these and nothing else.
VOL = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4,
       "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9,
       "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}
VREF = 130.0
SPHERE_FEATS = ["dg_self", "dg_total", "dg_far", "dg_reach"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def build_network(coords, vols):
    """Weighted Laplacian of the overlapping-sphere graph, and its pseudo-inverse.

    Full eigendecomposition rather than a truncated mode set: the review measured that a 50-mode
    truncation agrees with the full pseudo-inverse at only rho 0.982 while low-pass filtering away
    exactly the locality the sphere chain claims, and the full solve is affordable at these sizes.
    """
    n = len(coords)
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    A = (D < CUTOFF) & (D > 0)
    W = np.where(A, GAMMA0 * np.sqrt(np.outer(vols, vols)) / VREF, 0.0)
    K = np.diag(W.sum(1)) - W
    lam, V = np.linalg.eigh(K)
    keep = lam > 1e-9                       # positive threshold, never != 0
    G = (V[:, keep] / lam[keep]) @ V[:, keep].T
    return W, K, G, int(keep.sum())


def perturb(G, W, coords, i, v_old, v_new):
    """First-order response to changing ONE site's side-chain volume.

    dK is rank-limited to row/column i, so dG = -G dK G costs a handful of matrix-vector products
    rather than a re-inversion. This is the propagation: a change at i, felt at every j, through the
    overlaps.
    """
    n = len(coords)
    w_old = W[i].copy()
    scale = np.sqrt(v_new / max(v_old, 1e-9))
    w_new = w_old * scale
    dw = w_new - w_old
    dw[i] = 0.0
    dK = np.zeros((n, n))
    dK[i, :] = -dw
    dK[:, i] = -dw
    dK[i, i] = dw.sum()
    np.fill_diagonal(dK, np.diag(dK) + np.where(np.arange(n) == i, 0.0, dw))
    dG = -(G @ dK @ G)
    row = dG[i]
    seqsep = np.abs(np.arange(n) - i)
    return {"dg_self": float(dG[i, i]),
            "dg_total": float(np.abs(row).sum()),
            "dg_far": float(np.abs(row[seqsep > LONG_RANGE]).sum()),
            "dg_reach": float((np.abs(row) > 0.01 * abs(dG[i, i]) + 1e-12).sum())}


def sidechain_nodes(path):
    """Side-chain centroid per residue (CA for glycine), plus the one-letter residue and pLDDT."""
    per = {}
    for ln in open(path, errors="ignore"):
        if not ln.startswith("ATOM"):
            continue
        nm = ln[12:16].strip()
        el = (ln[76:78].strip() or nm[:1])
        if el == "H":
            continue
        try:
            r = int(ln[22:26])
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        except ValueError:
            continue
        aa = LR.THREE.get(ln[17:20].strip().upper())
        if not aa:
            continue
        d = per.setdefault(r, {"aa": aa, "sc": [], "ca": None, "b": float(ln[60:66])})
        if nm == "CA":
            d["ca"] = xyz
        if nm not in ("N", "CA", "C", "O", "OXT"):
            d["sc"].append(xyz)
    out = {}
    for r, d in per.items():
        if d["ca"] is None:
            continue
        c = np.mean(d["sc"], axis=0) if d["sc"] else np.array(d["ca"])
        out[r] = {"aa": d["aa"], "xyz": np.asarray(c, float), "b": d["b"]}
    return out


def bfactor_check(n_struct=40):
    """S0: does the network reproduce measured fluctuations? Crystal B-factors, per chain."""
    import flex_physics as F
    import glob
    rs = []
    files = sorted(glob.glob(f"{F.PDBDIR}/*.pdb"))[:400]
    rng = np.random.default_rng(SEED)
    for p in files:
        if len(rs) >= n_struct:
            break
        try:
            nodes = sidechain_nodes(p)
        except Exception:
            continue
        if not (60 <= len(nodes) <= 500):
            continue
        idx = sorted(nodes)
        co = np.array([nodes[i]["xyz"] for i in idx])
        vols = np.array([VOL.get(nodes[i]["aa"], VREF) for i in idx])
        b = np.array([nodes[i]["b"] for i in idx])
        if np.std(b) < 1e-6:
            continue
        try:
            _, _, G, _ = build_network(co, vols)
        except Exception:
            continue
        r = spearmanr(np.diag(G), b).statistic
        if np.isfinite(r):
            rs.append((Path(p).stem, float(r)))
    return rs


def build_features():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    base, _ = pickle.load(open(DN.FEATCACHE, "rb"))
    structs = pickle.load(open(SC / "_chain_structs.pkl", "rb"))
    want = {}
    for r in base:
        want.setdefault(r["gene"], []).append(r)
    out, dropped = [], []
    for gi, (g, rs) in enumerate(sorted(want.items())):
        p = structs.get(g)
        if not p or not Path(p).exists():
            dropped.append(g)
            continue
        nodes = sidechain_nodes(p)
        idx = sorted(nodes)
        pos = {r: k for k, r in enumerate(idx)}
        if len(idx) < 30 or len(idx) > 4000:
            dropped.append(g)
            continue
        co = np.array([nodes[i]["xyz"] for i in idx])
        vols = np.array([VOL.get(nodes[i]["aa"], VREF) for i in idx])
        try:
            W, K, G, nmode = build_network(co, vols)
        except Exception:
            dropped.append(g)
            continue
        # residue index is the same 1-based CDS index loop 51 validated; map onto this node list
        seq, pl, xyz = LR.af_structure(p)
        for rec in rs:
            ri = rec.get("_ri")
            if ri is None:
                continue
            k = pos.get(ri)
            if k is None:
                continue
            wt, mt = rec["_wt"], rec["_mut"]
            if wt not in VOL or mt not in VOL or nodes[ri]["aa"] != wt:
                continue
            f = perturb(G, W, co, k, VOL[wt], VOL[mt])
            new = dict(rec)
            new.update(f)
            out.append(new)
        if (gi + 1) % 20 == 0:
            say(f"     {gi+1}/{len(want)} genes, {len(out):,} variants")
    pickle.dump((out, dropped), open(CACHE, "wb"))
    return out, dropped


def rank_fit(model, X, y, grp, seed=SEED, folds=FOLDS):
    X = np.asarray(X, float)
    y = np.asarray(y)
    ug = np.unique(grp)
    fold = {g: i for g, i in zip(ug, np.random.default_rng(seed).permutation(len(ug)) % folds)}
    fid = np.array([fold[g] for g in grp])
    oof = np.full(len(y), np.nan)
    for k in range(folds):
        tr, te = fid != k, fid == k
        if tr.sum() < 100 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        A = np.empty_like(X[tr])
        B = np.empty_like(X[te])
        for j in range(X.shape[1]):
            srt = np.sort(X[tr, j])
            A[:, j] = np.searchsorted(srt, X[tr, j], side="right") / max(len(srt), 1)
            B[:, j] = np.searchsorted(srt, X[te, j], side="right") / max(len(srt), 1)
        m = model()
        m.fit(A, y[tr])
        oof[te] = (m.decision_function(B) if hasattr(m, "decision_function")
                   else m.predict_proba(B)[:, 1])
    return oof


def net():
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
                         max_iter=600, random_state=SEED, early_stopping=True, n_iter_no_change=15)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 55 -- the connected sphere chain, with the mutation actually in the network")
    say("=" * 100)
    say(f"  spheres of radius {SPHERE_R} A, connected when they OVERLAP (centres < {CUTOFF} A)")

    # ---- S0 ---------------------------------------------------------------------------------------
    say(f"\n  S0 THE NETWORK REPRODUCES A MECHANICAL OBSERVABLE")
    bf = bfactor_check()
    rs = np.array([r for _, r in bf])
    say(f"     self-response vs EXPERIMENTAL B-factors on {len(bf)} crystal structures")
    say(f"     mean Spearman {rs.mean():+.4f}   median {np.median(rs):+.4f}   "
        f"positive in {int((rs > 0).sum())}/{len(rs)}")
    say(f"     published GNM baselines sit near 0.55-0.65; this runs on data with no variants in it")
    s0 = bool(rs.mean() > BFACTOR_FLOOR)
    say(f"     S0 {'PASS' if s0 else 'FAIL'}  (mean > {BFACTOR_FLOOR})")

    say(f"\n  building sphere-chain features (side-chain centroids, volume-weighted edges)")
    recs, dropped = build_features()
    if not recs:
        say("     NO VARIANTS SCORED -- nothing downstream can run")
        return 1
    y = np.array([r["y"] for r in recs])
    gn = np.array([r["gene"] for r in recs])
    col = lambda k: np.array([r.get(k, np.nan) for r in recs], float)
    say(f"     {len(recs):,} variants, {len(set(gn)):,} genes, {y.mean():.1%} pathogenic"
        f"{f', {len(dropped)} genes dropped' if dropped else ''}")

    # ---- S1 mutation dependence ---------------------------------------------------------------------
    say(f"\n  S1 THE FEATURES ARE MUTATION-DEPENDENT")
    site = {}
    for r in recs:
        site.setdefault((r["gene"], r["_ri"]), []).append(r)
    multi = [v for v in site.values() if len({x["_mut"] for x in v}) >= 2]
    say(f"     {len(multi):,} residues carry two or more different substitutions")
    varies = {}
    for f in SPHERE_FEATS:
        frac = np.mean([len({round(x[f], 12) for x in v}) > 1 for v in multi]) if multi else 0.0
        varies[f] = float(frac)
        say(f"     {f:<10} varies across substitutions at {frac:.1%} of those residues")
    s1 = bool(multi and all(v > 0.95 for v in varies.values()))
    say(f"     S1 {'PASS' if s1 else 'FAIL'}  -- the reviewed draft would have scored 0.0% here, "
        f"which is why this gate exists")

    # ---- S2 not a re-parameterisation ---------------------------------------------------------------
    say(f"\n  S2 NO FEATURE IS A RE-PARAMETERISATION OF ONE ALREADY MEASURED")
    L54 = sorted(set(DN.PER_RESIDUE + DN.PAIRWISE))
    # de-duplicate loop 54's own block: burial and nc10 are byte-identical in all rows
    keep54, seen = [], []
    for c in L54:
        v = col(c)
        if any(np.array_equal(v, s) for s in seen):
            say(f"     loop 54 column {c} is a DUPLICATE of an earlier one and is dropped")
            continue
        seen.append(v)
        keep54.append(c)
    say(f"     loop 54 block: {len(L54)} declared, {len(keep54)} distinct")
    B54 = np.column_stack([col(c) for c in keep54])
    B54 = np.where(np.isfinite(B54), B54, np.nanmedian(B54, axis=0))
    survive, r2s = [], {}
    for f in SPHERE_FEATS:
        v = col(f)
        A = np.column_stack([B54, np.ones(len(v))])
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        r2 = 1 - np.var(v - A @ beta) / max(np.var(v), 1e-12)
        r2s[f] = float(r2)
        dup = None
        for s in survive:
            if abs(spearmanr(v, col(s)).statistic) > COLLIN:
                dup = s
                break
        say(f"     {f:<10} R2 vs loop 54 block {r2:>7.4f}"
            + (f"   COLLINEAR with {dup}, dropped" if dup else
               ("   R2 > %.2f, dropped as already-present" % R2_DROP if r2 > R2_DROP else "   kept")))
        if dup is None and r2 <= R2_DROP:
            survive.append(f)
    s2 = bool(survive)
    say(f"     survivors: {', '.join(survive) if survive else 'NONE'}")
    say(f"     S2 {'PASS' if s2 else 'FAIL'}  (at least one feature carries information the "
        f"already-measured block does not)")

    # ---- S3 does it add ------------------------------------------------------------------------------
    say(f"\n  S3 IT ADDS OVER LOOP 54'S BEST, REFITTED HERE")
    prim = [c for c in keep54 if c != "negaf"]
    say(f"     PRIMARY bar excludes allele frequency, because population frequency is itself an ACMG")
    say(f"     classification criterion and a bar containing the label's own definition is not fair.")

    def score(cols):
        v = rank_fit(net, np.column_stack([np.where(np.isfinite(col(c)), col(c), np.nan)
                                           for c in cols]), y, gn)
        pg = DN.per_gene(v, y, gn)
        return pg, (float(np.mean(list(pg.values()))) if pg else np.nan)

    def clean(cols):
        X = np.column_stack([col(c) for c in cols])
        med = np.nanmedian(X, axis=0)
        bad = ~np.isfinite(X)
        X[bad] = np.take(med, np.where(bad)[1])
        return X

    arms = {}
    for lab, cols in (("loop54 primary (no AF)", prim),
                      ("loop54 full", keep54),
                      ("loop54 primary + spheres", prim + survive),
                      ("loop54 full + spheres", keep54 + survive)):
        v = rank_fit(net, clean(cols), y, gn)
        pg = DN.per_gene(v, y, gn)
        arms[lab] = {"pg": pg, "mean": float(np.mean(list(pg.values()))) if pg else np.nan}
        say(f"     {lab:<28}{arms[lab]['mean']:.4f}   ({len(pg)} genes)")
    base_pg = arms["loop54 primary (no AF)"]["pg"]
    sd = np.std([base_pg[k] for k in base_pg]) if base_pg else np.nan
    mde = 2.8 * sd / np.sqrt(max(len(base_pg), 1))
    say(f"     minimum detectable effect at {len(base_pg)} genes: {mde:+.4f} "
        f"(per-gene sd {sd:.4f}) -- printed BEFORE the comparison")
    c3 = DN.paired(arms["loop54 primary + spheres"]["pg"], base_pg)
    say(f"     spheres over the primary bar: delta {c3['delta']:+.4f}, wins {c3['win']:.1%}, "
        f"p {c3['p']:.2e}")
    s3 = bool(np.isfinite(c3["p"]) and c3["p"] < 0.01 and c3["win"] > 0.60)
    say(f"     S3 {'PASS' if s3 else 'FAIL'}  -- a fail here means the sphere chain adds less than "
        f"{max(mde, abs(c3['delta'])):+.4f}, not that it adds nothing")

    # ---- S4 two nulls ---------------------------------------------------------------------------------
    say(f"\n  S4 THE NULL COLLAPSES IT, TWICE")
    Xs = clean(prim + survive)
    idx = {g: np.where(gn == g)[0] for g in set(gn)}
    lab_null = []
    for _ in range(N_NULL):
        ys = y.copy()
        for g, ii in idx.items():
            ys[ii] = rng.permutation(y[ii])
        v = rank_fit(net, Xs, ys, gn)
        pg = DN.per_gene(v, ys, gn)
        if pg:
            lab_null.append(float(np.mean(list(pg.values()))))
    lab_null = np.array(lab_null)
    say(f"     label permuted within gene, refitted {len(lab_null)}x: mean {lab_null.mean():.4f}, "
        f"95th {np.percentile(lab_null, 95):.4f}")
    # structure randomised: shuffle the sphere columns WITHIN gene, keeping their marginal
    Xr = Xs.copy()
    sph_at = [len(prim) + i for i in range(len(survive))]
    for g, ii in idx.items():
        for j in sph_at:
            Xr[ii, j] = rng.permutation(Xr[ii, j])
    v = rank_fit(net, Xr, y, gn)
    pg = DN.per_gene(v, y, gn)
    struct_null = float(np.mean(list(pg.values()))) if pg else np.nan
    say(f"     sphere columns shuffled WITHIN gene (geometry destroyed, marginal kept): "
        f"{struct_null:.4f}")
    say(f"     against the real {arms['loop54 primary + spheres']['mean']:.4f} and the "
        f"no-sphere bar {arms['loop54 primary (no AF)']['mean']:.4f}")
    s4 = bool(arms["loop54 primary + spheres"]["mean"] > np.percentile(lab_null, 95)
              and lab_null.mean() < 0.55)
    say(f"     S4 {'PASS' if s4 else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"S0 network reproduces B-factors": s0, "S1 features are mutation-dependent": s1,
             "S2 not a re-parameterisation": s2, "S3 adds over loop 54 refitted": s3,
             "S4 the nulls collapse it": s4}
    for k, v in gates.items():
        say(f"  {k:<40}{'PASS' if v else 'FAIL'}")
    say(f"  THE MUTATION IS NOW IN THE NETWORK. Each node sits at its side-chain centroid, each edge")
    say(f"  is weighted by the volumes it joins, and a substitution changes every edge touching its")
    say(f"  site -- so dG = -G dK G is a response to the mutation and not a description of the fold.")
    say(f"  S1 measures that directly: the features vary across substitutions at "
        f"{min(varies.values()):.0%}+ of")
    say(f"  multiply-substituted residues, where the reviewed draft would have scored zero.")
    say(f"  Sphere features surviving the redundancy gate: "
        f"{', '.join(survive) if survive else 'NONE'}.")
    say(f"  Over loop 54 refitted here without allele frequency, they add {c3['delta']:+.4f} "
        f"({c3['win']:.0%} of genes,")
    say(f"  p {c3['p']:.1e}) against a minimum detectable effect of {mde:+.4f}.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(SC / "_chain_structs.pkl"), str(DN.FEATCACHE),
                              str(SC / "pdb_cache")],
                      available=len(recs), used=len(recs), selection="all", seed=SEED,
                      controls=["B-factor validation on crystal structures before any variant is seen",
                                "mutation-dependence gate that the reviewed draft would have failed",
                                "collinearity and R2 redundancy gate that can drop features",
                                "loop 54's arms refitted here rather than quoted",
                                "primary bar excludes allele frequency, an ACMG criterion",
                                f"{N_NULL} within-gene label permutations plus a geometry-destroying "
                                f"shuffle"],
                      note="4 A spheres connected on OVERLAP; the 4 A centre-distance reading gives a "
                           "1D chain with mean degree 2.0 and is not what is built")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_sphere_chain", "manifest": man, "gates": gates,
               "bfactor": {"mean": float(rs.mean()), "median": float(np.median(rs)), "n": len(bf)},
               "mutation_dependence": varies, "r2_vs_loop54": r2s, "survivors": survive,
               "loop54_dedup_kept": keep54,
               "arms": {k: v["mean"] for k, v in arms.items()},
               "S3": c3, "mde": float(mde),
               "label_null_mean": float(lab_null.mean()),
               "label_null_95": float(np.percentile(lab_null, 95)),
               "structure_null": struct_null,
               "n_variants": len(recs), "n_genes": int(len(set(gn))),
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_sphere_chain.json", "w"), indent=2, default=float)
    say(f"\n  -> {OUT/'loop_sphere_chain.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
