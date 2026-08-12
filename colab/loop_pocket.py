"""LOOP 60 -- DOES THE CONTACT GRAPH ALONE FIND THE FUNCTIONAL RESIDUES?

WHERE THIS COMES FROM, and the correction it makes to loop 55. Loop 55 built exactly the connected
sphere chain that was asked for -- 4 A spheres, connected when they OVERLAP, mutation entering the
spring matrix, first-order response dG = -G dK G -- and asked whether it improved the prediction of
WHETHER a variant is pathogenic. It added -0.0015. That was the sixth representation in a row to land
inside the noise on that question.

The question was wrong, not the representation. Loop 57 flipped from "is this variant bad" to "what
KIND of damage does it do" and returned the first 5/5 in the arc. This applies the same flip to the
chain: not "does the graph predict harm" but "does the graph find the ACTIVE SITE".

WHY THIS QUESTION IS CLEANER THAN EVERYTHING BEFORE IT. Every confound that has killed a loop in this
project is an ATTENTION confound -- ClinVar labels track study, SIDER lists track study, the drug
field's single best feature was literature volume at 0.7913. This task is a WITHIN-PROTEIN RANKING:
which residue of TP53 is its functional site, scored against the other residues of TP53. Every
residue in the comparison shares its protein's fame exactly, so the confound cancels inside the unit
of evaluation. That property is rare here and is the reason to spend a loop on it.

WHAT IS SCORED, measured before this was written. UniProt residue-level annotation over the 141
proteins with cached AlphaFold structures gives 1,909 functional residues in 86 proteins with at
least three. But 957 of those 1,909 -- HALF -- are ZINC FINGER coordination residues, and a zinc
finger is four residues in a tight tetrahedral cluster, which a contact model finds almost by
construction. Two proteins (ZEB2 188, ZBTB20 115) supply a third of them.

  SO THE PRIMARY GATE IS DECLARED ON THE ZINC-FINGER-EXCLUDED SET: 73 proteins, 855 residues drawn
  from Active site (27), Binding site (804), Site (168) and DNA binding (39). The ZF-inclusive number
  is computed and reported as a secondary, never as the headline. This choice is made here, with the
  composition known and no score yet computed, because making it afterwards would be selection.

THE FEATURES ARE COORDINATES AND NOTHING ELSE. No conservation, no ESM, no substitution matrix, no
annotation, and -- deliberately -- NO AMINO-ACID IDENTITY. If the model may see that a residue is a
histidine it can score well by chemistry alone, and "count the histidines in ordered regions" is a
lookup table, not a pocket finder. Identity is held out of the model and scored separately as the
control it is.

    deg8 deg12 deg16      neighbours of the side-chain centroid at three shells
    protrusion            |mean unit vector to neighbours within 12 A| -- 0 in a pocket, high on a
                          convex surface. This is the actual cleft detector.
    pack                  deg8 / deg16, local density against regional density
    farness               mean hop distance on the 8 A graph, over the REACHABLE set only
    reach                 fraction of the protein reachable at all -- AlphaFold's disordered tails
                          leave residues isolated, and calling those averagely central would be a lie
    gnm_msf               G_ii of the standard uniform-gamma Gaussian network, all modes
    gnm_slow              the 3 slowest non-zero modes at i -- hinge regions
    gnm_fast              the 10 fastest modes at i -- the published active-site localisation
    dist_cen              distance from the centroid in units of radius of gyration
    contact_order         mean |i-j| over 8 A contacts
    lr8                   8 A contacts with |i-j| > 12
    seq_rel               i / L, present to expose terminus effects rather than to help

Features are rank-transformed WITHIN each protein, so the model learns "dense for this protein"
rather than an absolute that tracks protein size. Folds are grouped by protein; a protein's residues
are scored only by a model that never saw that protein.

PREDECLARED, before any number:

  A1 THE STRUCTURE INDEXES AGAINST UNIPROT NUMBERING
       AlphaFold DB is built on the UniProt canonical sequence, so the PDB residue number should BE
       the UniProt position -- but loop 51 already found 16 genes shifted by an isoform offset, and
       an annotation read at the wrong residue labels a plausible, wrong site. Per protein the
       residue at the PDB index must equal the UniProt sequence residue for >= 95% of positions;
       proteins below that are NAMED and dropped entirely.
  A2 GEOMETRY RANKS FUNCTIONAL RESIDUES WITHIN THEIR OWN PROTEIN            THE REAL GATE.
       per-protein AUC, macro-averaged over the primary set, against a null that permutes labels
       WITHIN each protein preserving its positive count. Must clear the null's 95th percentile.
  A3 IT IS NOT JUST ORDERED VS DISORDERED
       AlphaFold pLDDT alone, scored identically. Functional residues live in folded domains, so a
       contact count on a predicted model partly measures whether the region is folded at all. The
       geometry block must beat pLDDT alone, or the finding is "active sites are in ordered regions",
       which is already known and is not a pocket finder.
  A4 IT IS NOT JUST WHICH AMINO ACID IT IS
       20-way one-hot identity alone, scored identically, plus the combination. Geometry must add
       over identity on the combined fit. If identity wins outright the honest headline is chemistry,
       and this says so.
  A5 IT IS NOT CARRIED BY THE ZINC FINGERS
       the same fit re-scored on the ZF-inclusive set, reported beside the primary. A large gap means
       the model finds metal-coordination clusters rather than functional sites, and the secondary
       number is then the one that must not be quoted.

  WHAT A PASS WOULD BUY. A geometry-only site finder needs no annotation, no homologue and no
  literature, so it applies to a protein nobody has studied. That is the missing input for the drug
  direction: the ligand identity was stripped from this project's UniProt cache (508 of 541 binding
  sites carry an empty description), so pockets cannot currently be read out of the annotation --
  they would have to be found.

-> outputs/loop_pocket.json
"""
import collections
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_rescue as LR
import loop_sphere_chain as SPH
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC

SPHERE_R = 4.0
CUTOFF = 2 * SPHERE_R          # spheres connect when they overlap -- loop 55's measured construction
A1_FLOOR = 0.95
MIN_POS = 3
MIN_RES = 60
MAX_SPAN = 50                  # a 200-residue "DNA binding" region is a domain, not a residue
FOLDS = 5
N_NULL = 200
SEED = 6003

FUNC = {"Active site", "Binding site", "Site", "DNA binding", "Nucleotide binding"}
ZF = {"Zinc finger"}

GEOM = ["deg8", "deg12", "deg16", "protrusion", "pack", "farness", "reach",
        "gnm_msf", "gnm_slow", "gnm_fast", "dist_cen", "contact_order", "lr8", "seq_rel"]
COLLIN = 0.98             # loop 55's rule: assert the pair is not a re-parameterisation, never trust
                          # a feature list to be distinct
AAS = "ACDEFGHIKLMNPQRSTVWY"

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def labels_for(ft, types):
    """Residue positions carrying one of `types`, skipping spans wide enough to be domains."""
    res = set()
    for f in ft:
        if f["t"] not in types:
            continue
        s = f["s"]
        e = f["e"] or s
        if e - s > MAX_SPAN:
            continue
        res.update(range(s, e + 1))
    return res


def geometry(nodes):
    """Every feature in GEOM, from coordinates alone. No identity, no pLDDT, no annotation."""
    idx = sorted(nodes)
    n = len(idx)
    X = np.array([nodes[r]["xyz"] for r in idx], float)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)

    A8 = D < CUTOFF
    A12 = D < 12.0
    A16 = D < 16.0
    deg8 = A8.sum(1).astype(float)
    deg12 = A12.sum(1).astype(float)
    deg16 = A16.sum(1).astype(float)

    # concavity: neighbours spread evenly around a pocket residue cancel; a convex surface residue
    # has all of its neighbours on one side and the mean unit vector survives.
    prot = np.zeros(n)
    for i in range(n):
        nb = np.where(A12[i])[0]
        if len(nb) == 0:
            continue
        v = X[nb] - X[i]
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        prot[i] = np.linalg.norm(v.mean(0))

    # AlphaFold's disordered tails leave residues isolated in the 8 A graph, so the hop distance is
    # infinite for part of the protein. Filling those with the column mean would call a residue
    # connected to nothing averagely central. Mean hops is taken over the REACHABLE set only, the
    # unreachable fraction becomes its own feature, and a fully isolated residue is given the worst
    # observed distance rather than the middle one.
    g = csr_matrix(A8.astype(np.int8))
    sp = shortest_path(g, method="D", unweighted=True, directed=False)
    fin = np.isfinite(sp)
    np.fill_diagonal(fin, False)
    closeness = np.array([sp[i][fin[i]].mean() if fin[i].any() else np.nan for i in range(n)])
    reach = fin.sum(1).astype(float) / max(n - 1, 1)
    if np.isnan(closeness).any() and np.isfinite(closeness).any():
        closeness = np.where(np.isnan(closeness), np.nanmax(closeness), closeness)
    closeness = np.nan_to_num(closeness, nan=0.0)

    # standard uniform-gamma GNM -- this is a description of the fold, nothing is perturbed here
    K = np.diag(A8.sum(1).astype(float)) - A8.astype(float)
    lam, V = np.linalg.eigh(K)
    nz = np.where(lam > 1e-9)[0]
    msf = ((V[:, nz] ** 2) / lam[nz]).sum(1)
    slow = ((V[:, nz[:3]] ** 2) / lam[nz[:3]]).sum(1)
    fast = (V[:, nz[-10:]] ** 2).sum(1)

    cen = X.mean(0)
    rg = np.sqrt(((X - cen) ** 2).sum(1).mean())
    dist_cen = np.linalg.norm(X - cen, axis=1) / max(rg, 1e-9)

    seqsep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    co = np.array([seqsep[i][A8[i]].mean() if A8[i].any() else 0.0 for i in range(n)])
    lr8 = (A8 & (seqsep > 12)).sum(1).astype(float)

    F = np.column_stack([deg8, deg12, deg16, prot, deg8 / np.maximum(deg16, 1.0), closeness, reach,
                         msf, slow, fast, dist_cen, co, lr8,
                         np.arange(n, dtype=float) / max(n - 1, 1)])
    assert np.isfinite(F).all(), "non-finite geometry feature"
    return idx, F


def within_rank(F):
    """Percentile of each feature among this protein's own residues."""
    R = np.empty_like(F)
    for j in range(F.shape[1]):
        col = F[:, j]
        ok = np.isfinite(col)
        R[:, j] = 0.5
        if ok.sum() > 1:
            srt = np.sort(col[ok])
            R[ok, j] = np.searchsorted(srt, col[ok], side="right") / len(srt)
    return R


def build():
    cache = SC / "_pocket_v1.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    g2a = json.load(open(SC / "_pae_genes.json"))
    structs = pickle.load(open(SC / "_chain_structs.pkl", "rb"))
    rows = []
    dropped = []
    agree_n = agree_d = 0
    for gi, (g, acc) in enumerate(sorted(g2a.items())):
        path = structs.get(g)
        ftp = SC / "uniprot_ft" / f"{acc}.json"
        seqp = SC / "uniprot_seq" / f"{acc}.txt"
        if not path or not os.path.exists(path) or not ftp.exists() or not seqp.exists():
            continue
        seq = seqp.read_text().strip()
        nodes = SPH.sidechain_nodes(path)
        if len(nodes) < MIN_RES:
            continue
        ok = tot = 0
        for r, d in nodes.items():
            if 1 <= r <= len(seq):
                tot += 1
                ok += int(seq[r - 1] == d["aa"])
        frac = ok / max(tot, 1)
        agree_n += ok
        agree_d += max(tot, 1)
        if frac < A1_FLOOR:
            dropped.append({"gene": g, "acc": acc, "agreement": round(frac, 4)})
            continue
        ft = json.load(open(ftp))
        pos = labels_for(ft, FUNC)
        posz = labels_for(ft, FUNC | ZF)
        idx, F = geometry(nodes)
        R = within_rank(F)
        rows.append({"gene": g, "acc": acc, "idx": idx, "R": R,
                     "aa": np.array([nodes[r]["aa"] for r in idx]),
                     "plddt": np.array([nodes[r]["b"] for r in idx], float),
                     "y": np.array([int(r in pos) for r in idx]),
                     "yz": np.array([int(r in posz) for r in idx])})
        say(f"     [{gi + 1:3d}] {g:10s} {acc}  n={len(idx):5d}  func={int(sum(r in pos for r in idx)):4d}"
            f"  +zf={int(sum(r in posz for r in idx)):4d}  seq-agree={frac:.4f}")
    out = {"rows": rows, "dropped": dropped, "agreement": agree_n / max(agree_d, 1)}
    pickle.dump(out, open(cache, "wb"))
    return out


def macro_auc(rows, key, P):
    """Per-protein AUC, macro-averaged. A protein with no contrast contributes nothing."""
    per = {}
    for r, p in zip(rows, P):
        y = r[key]
        if y.sum() < MIN_POS or (len(y) - y.sum()) < 20:
            continue
        per[r["gene"]] = float(roc_auc_score(y, p))
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def fit(rows, X, key):
    """Protein-grouped CV; returns held-out scores in row order."""
    rng = np.random.default_rng(SEED)
    fold = rng.permutation(len(rows)) % FOLDS
    P = [None] * len(rows)
    for k in range(FOLDS):
        tr = [i for i in range(len(rows)) if fold[i] != k]
        te = [i for i in range(len(rows)) if fold[i] == k]
        if not te:
            continue
        Xtr = np.vstack([X[i] for i in tr])
        ytr = np.concatenate([rows[i][key] for i in tr])
        if len(set(ytr)) < 2:
            continue
        m = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
        m.fit(Xtr, ytr)
        for i in te:
            P[i] = m.predict_proba(X[i])[:, 1]
    return P


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 60 -- does the contact graph alone find the functional residues?")
    say("=" * 100)
    say(f"  spheres of radius {SPHERE_R} A, connected when they overlap (centres < {CUTOFF} A)")
    say(f"  primary label set EXCLUDES zinc fingers, declared before any score was computed")
    say()

    B = build()
    rows = B["rows"]
    say()
    say("A1 THE STRUCTURE INDEXES AGAINST UNIPROT NUMBERING")
    say(f"     overall agreement {B['agreement']:.4f}")
    say(f"     proteins below {A1_FLOOR:.0%} and DROPPED entirely: {len(B['dropped'])}")
    for d in B["dropped"]:
        say(f"       {d['gene']:10s} {d['acc']}  {d['agreement']:.4f}")
    a1 = B["agreement"] >= A1_FLOOR
    say(f"     kept {len(rows)} proteins")
    say(f"     A1 {'PASS' if a1 else 'FAIL'}")
    say()

    keep = [r for r in rows if r["y"].sum() >= MIN_POS and (len(r["y"]) - r["y"].sum()) >= 20]
    keepz = [r for r in rows if r["yz"].sum() >= MIN_POS and (len(r["yz"]) - r["yz"].sum()) >= 20]
    npos = int(sum(r["y"].sum() for r in keep))
    nposz = int(sum(r["yz"].sum() for r in keepz))
    nres = int(sum(len(r["y"]) for r in keep))
    say(f"  primary  : {len(keep)} proteins, {npos} functional residues of {nres} ({npos / max(nres,1):.2%})")
    say(f"  secondary: {len(keepz)} proteins, {nposz} residues once zinc fingers are added back")
    say()

    # loop 53 fitted a collinear pair and loop 55's first draft shipped 1/G_ii alongside G_ii. The
    # feature list is not evidence that the features differ, so it is measured and offenders drop.
    pooled = np.vstack([r["R"] for r in keep])
    C = np.corrcoef(pooled, rowvar=False)
    drop = set()
    for i in range(len(GEOM)):
        for j in range(i + 1, len(GEOM)):
            if abs(C[i, j]) > COLLIN and j not in drop:
                drop.add(j)
                say(f"  COLLINEAR  {GEOM[i]} ~ {GEOM[j]}  rho {C[i, j]:+.4f}  -- dropping {GEOM[j]}")
    cols = [i for i in range(len(GEOM)) if i not in drop]
    say(f"  geometry block: {len(cols)} of {len(GEOM)} features kept -- {[GEOM[i] for i in cols]}")
    say()
    for r in rows:                     # keep and keepz hold the same dicts; slice the master list once
        r["R"] = r["R"][:, cols]

    Xg = [r["R"] for r in keep]
    Xp = [r["plddt"].reshape(-1, 1) for r in keep]
    Xa = [np.array([[float(a == c) for c in AAS] for a in r["aa"]]) for r in keep]
    Xga = [np.hstack([g, a]) for g, a in zip(Xg, Xa)]

    Pg = fit(keep, Xg, "y")
    say("A2 GEOMETRY RANKS FUNCTIONAL RESIDUES WITHIN THEIR OWN PROTEIN   -- THE GATE")
    geo, per = macro_auc(keep, "y", Pg)
    rng = np.random.default_rng(SEED + 1)
    nulls = []
    for _ in range(N_NULL):
        vals = []
        for r, p in zip(keep, Pg):
            y = r["y"].copy()
            rng.shuffle(y)
            if y.sum() < MIN_POS:
                continue
            vals.append(roc_auc_score(y, p))
        nulls.append(np.mean(vals))
    nmean, n95 = float(np.mean(nulls)), float(np.percentile(nulls, 95))
    say(f"     macro per-protein AUC   {geo:.4f}")
    say(f"     within-protein null     {nmean:.4f}   95th pct {n95:.4f}   ({N_NULL} permutations)")
    a2 = geo > n95
    say(f"     A2 {'PASS' if a2 else 'FAIL'}")
    say()

    say("A3 IT IS NOT JUST ORDERED VS DISORDERED")
    Pp = fit(keep, Xp, "y")
    pl, _ = macro_auc(keep, "y", Pp)
    say(f"     pLDDT alone             {pl:.4f}")
    say(f"     geometry                {geo:.4f}     delta {geo - pl:+.4f}")
    a3 = geo > pl
    say(f"     A3 {'PASS' if a3 else 'FAIL'}")
    say()

    say("A4 IT IS NOT JUST WHICH AMINO ACID IT IS")
    Pa = fit(keep, Xa, "y")
    aa, _ = macro_auc(keep, "y", Pa)
    Pga = fit(keep, Xga, "y")
    both, _ = macro_auc(keep, "y", Pga)
    say(f"     identity alone          {aa:.4f}")
    say(f"     geometry alone          {geo:.4f}")
    say(f"     both                    {both:.4f}     over identity {both - aa:+.4f}")
    a4 = both > aa
    say(f"     A4 {'PASS' if a4 else 'FAIL'}")
    say()

    say("A5 IT IS NOT CARRIED BY THE ZINC FINGERS")
    Xgz = [r["R"] for r in keepz]
    Pgz = fit(keepz, Xgz, "yz")
    gz, _ = macro_auc(keepz, "yz", Pgz)
    say(f"     zinc-finger-INCLUSIVE   {gz:.4f}   on {len(keepz)} proteins, {nposz} residues")
    say(f"     primary (ZF excluded)   {geo:.4f}   on {len(keep)} proteins, {npos} residues")
    a5 = geo > n95 and (gz - geo) < 0.10
    say(f"     A5 {'PASS' if a5 else 'FAIL'}  (primary must clear its null and the gap stay under 0.10)")
    say()

    best = sorted(per.items(), key=lambda kv: -kv[1])[:8]
    worst = sorted(per.items(), key=lambda kv: kv[1])[:8]
    say("  per-protein, best and worst:")
    for g, v in best:
        say(f"     {g:10s} {v:.4f}")
    say("     ...")
    for g, v in worst:
        say(f"     {g:10s} {v:.4f}")
    say()

    gates = {"A1 structure indexes against UniProt": bool(a1),
             "A2 geometry ranks functional residues": bool(a2),
             "A3 not just order": bool(a3),
             "A4 not just amino-acid identity": bool(a4),
             "A5 not carried by zinc fingers": bool(a5)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(
        inputs=[str(SC / "_chain_structs.pkl"), str(SC / "uniprot_ft"), str(SC / "uniprot_seq")],
        available=len(rows), used=len(keep), selection="filtered",
        seed=SEED,
        controls=["within-protein label permutation preserving positive count",
                  "pLDDT alone as the order control",
                  "amino-acid identity alone as the chemistry control",
                  "zinc fingers excluded from the primary set, declared before scoring",
                  "protein-grouped folds; no protein scored by a model that saw it",
                  "no conservation, no ESM, no annotation and no identity in the geometry block"],
        note="within-protein ranking, so a protein's literature volume is constant inside the "
             "unit of evaluation and cannot separate its own residues")

    RM.report(man, emit=say)          # before the dump, so the manifest's own complaints are saved
    res = {"test": "loop_pocket", "manifest": man, "gates": gates,
           "a1_agreement": B["agreement"], "a1_dropped": B["dropped"],
           "n_proteins": len(keep), "n_residues": nres, "n_functional": npos,
           "geometry_macro": geo, "null_mean": nmean, "null_95": n95,
           "plddt_macro": pl, "identity_macro": aa, "combined_macro": both,
           "zf_inclusive_macro": gz, "n_proteins_zf": len(keepz), "n_functional_zf": nposz,
           "per_protein": per, "seconds": time.time() - t0, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "loop_pocket.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_pocket.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
