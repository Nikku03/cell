"""CELLBRAIN -- the brain, connected to the memory bank, given one task: make as many connections as it can.

The brain (Nikku03/money: its bank, its declare -> commit -> test ledger, its rules from docs/NEURO.md) reads every
dataset of both groups through the cellbrain bank (cellbrain_bank.py: one sha-checked LAYER fact per dataset) and
looks for connections of nine kinds. Every connection is FOUND on one half of the data and must be CONFIRMED on the
other half (or, for gene pairs, by independent kinds of data), against a null that keeps everything but the link.
What is confirmed goes back into the bank as a derived fact citing the layers it came from.

THE RULES, per family (declared before running; the ledger is rem/atlas/cellbrain_ledger.jsonl, its own family):
  carry forward   what the DEV half suggests (an effect size AND a nominal p)
  confirm         the EXAM half, BH q < 0.05 over everything carried forward in the family, same direction
  negative        the whole pipeline on data with the link destroyed (labels shuffled / variants shifted): it must
                  confirm at most max(1, 5% of the real count)
  positive        connections planted in the data: >= 90% must be confirmed
  known answers   textbook links the family must find (count declared per family)
  PASS            all of the above; a family that cannot run its controls gets NO VERDICT
Halves: genes by sha256('cellbrain-c1:' + Ensembl id) (DEV 0 / EXAM 1); DNA by chromosome (odd autosomes DEV, even
EXAM); gene sets by their members' gene halves.

THE FAMILIES
  F1 data <-> data      Which kinds of data say the same thing about genes. 33 gene views (14 knockdown screens, 2
                        LINCS, 8 expression atlases incl. mouse and Geneformer, promoter chromatin, promoter
                        methylation, DepMap, interactions, complexes, pathways, locations, drug targets). For views X, Y:
                        AUROC of Y-similarity, X-neighbour pairs (k = 10) vs random pairs, within a half; null: Y's
                        genes permuted within its level deciles (20 draws, z). Carry DEV AUROC >= 0.55, p < 0.05;
                        confirm EXAM AUROC >= 0.55. Known: complex->K562 knockdown, interactions->complex, GTEx->
                        CELLxGENE, promoter chromatin->ENCODE RNA, DepMap->complex, K562 GW->K562 essential, mouse
                        CELLxGENE->human, methylation->promoter chromatin (>= 6 of 8).
  F2 gene <-> gene      A gene pair linked by independent kinds of data (knockdown, expression, chromatin,
                        essentiality, physical, annotation): each view's 20 nearest genes; support = kinds in which the
                        pair are neighbours. Null: every view's genes permuted within level deciles (3 draws). A pair is
                        CONFIRMED at the smallest support s* whose null-estimated FDR is <= 5%. Held-out truth: pairs
                        found WITHOUT the physical and annotation kinds (support >= 3 of the other 4) share a complex
                        or an interaction >= 10x more often than all pairs. Positive: 300 pairs planted in 3 kinds.
  F3 cell type <-> cell type   Units: CELLxGENE human and mouse cell types, ENCODE biosamples, GTEx tissues, FANTOM5
                        samples, HPA cell lines (RNA), WGBS cell types (promoter methylation, sign flipped), ENCODE
                        promoter chromatin (per biosample). Profiles z-scored per gene within each dataset; similarity
                        = Pearson over genes. Carry each unit's 3 nearest in every dataset on DEV genes; confirm on EXAM
                        genes (z against the unit's similarities to that dataset). Known: K562 ENCODE<->HPA, HepG2
                        ENCODE<->HPA, GTEx liver<->hepatocyte, GTEx pancreas<->acinar cell, mouse<->human hepatocyte,
                        WGBS liver hepatocytes<->hepatocyte, GTEx whole blood<->neutrophil (>= 5 of 7).
  F4 disease <-> cell type   Every Open Targets GWAS trait (one study per disease, >= 30 credible sets, up to 2,000):
                        per DNase-based cCRE annotation, the t statistic of 'open here' for the trait's PIP controlling
                        for element breadth (as trait2cell V1). Carry DEV t >= 3 (top 50 per trait); confirm EXAM t,
                        BH. Negative: 30 traits shifted 5-50 Mb along their chromosomes. Positive: 20 planted traits.
                        Known: the 8 V1 traits, expected category confirmed (>= 6 of 8).
  F5 disease <-> gene <-> tissue (DESCRIPTIVE, no verdict): GWAS and eQTL credible sets sharing causal variants
                        (CLPP = sum of PIP products >= 0.1; GTEx v11 eQTL 50 tissues, IBDverse 72 immune cell types).
  F6 knockdown <-> cell identity   K562 Perturb-seq knockdowns with a phenotype (energy test p < 0.05), genome-wide
                        and essential screens, vs CELLxGENE cell-type specificity: Pearson over measured genes, z by a
                        level-matched gene-permutation null. Carry |z_DEV| >= 4 (top 20 per knockdown); confirm EXAM z,
                        same sign. Known: GATA1, TAL1, KLF1, LMO2, LDB1, ZFPM1, NFE2 knockdown -> away from an
                        erythroid cell type (>= 4 of 7).
  F7 complex / pathway <-> cell type   Mean CELLxGENE specificity of a set's DEV members vs level-matched random
                        sets (z); carry z >= 4 (top 10 per set), confirm on EXAM members. Known: 7 pathway <-> cell
                        type pairs (>= 5).
  F8 drug <-> disease   Through F5 genes: (trait, drug) pairs where a drug target is a trait gene are more often
                        already indicated for that disease than other pairs (Fisher, OR > 1, p < 0.05; the known answer
                        from Nelson 2015 / Minikel 2024). The linked, not-yet-indicated pairs are listed, untested.
  F9 disease <-> disease   F4 cell-type profiles (residual, per half): carry DEV r >= 0.5 (top 20 per trait); confirm
                        EXAM r against 200 annotation permutations. Known: LDL<->total cholesterol, RBC<->haemoglobin,
                        lymphocyte<->white cell count, IBD<->Crohn's (>= 3 of 4).
NOT: correlation, not cause (F6 is the only family made of interventions); one half of a sample is not an independent
cohort; ENCODE and CELLxGENE over-represent some cell types; F5 has no LD and no verdict; the GWAS study per disease is
chosen by size; the reprocessed screens (CC BY-NC-SA) and any connection built on them stay local.

Run: python3 -P rem/atlas/cellbrain.py declare | test [FAMILY ...] | card ENTITY      (--trial: subsets, scratch)
"""

from __future__ import annotations
import collections
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
sys.path.insert(0, HERE)


def _imp(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


cbb = _imp("cellbrain_bank")
t2c = _imp("trait2cell")
e2 = t2c.e2
from brain.bank import Fact  # noqa: E402  (imported by cellbrain_bank from the brain checkout)

VERSION = "C1"
TRIAL = "--trial" in sys.argv
LEDGER = os.path.join(cbb.HOME, "trial_ledger.jsonl") if TRIAL else os.path.join(HERE, "cellbrain_ledger.jsonl")
RUN = os.path.join(cbb.HOME, "trial" if TRIAL else "run")
OUTD = os.path.join(RUN, "out") if TRIAL else R("outputs", "cellbrain")
RES = os.path.join(RUN, "RESULTS.txt") if TRIAL else os.path.join(HERE, "RESULTS_cellbrain.txt")
FAMILIES = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]
GATE = {"q": 0.05, "neg_share": 0.05, "planted": 0.90}
USED = collections.defaultdict(set)          # family -> layer fact refs it read
FAM = [None]


# ------------------------------------------------------------------------------------------------ shared
def L(name):
    f, path = cbb.layer(name)
    if FAM[0]:
        USED[FAM[0]].add(f.ref)
    return f, path


def npz(name):
    return np.load(L(name)[1], allow_pickle=False)


def half(keys, salt="c1"):
    return np.array([int(hashlib.sha256(f"cellbrain-{salt}:{k}".encode()).hexdigest(), 16) % 2 for k in keys], np.int8)


def bh(p):
    p = np.asarray(p, float); m = len(p)
    if not m:
        return p
    o = np.argsort(p); q = np.empty(m); run = 1.0
    for r in range(m - 1, -1, -1):
        run = min(run, p[o[r]] * m / (r + 1)); q[o[r]] = run
    return q


def auroc(pos, neg):
    from scipy.stats import rankdata
    n1, n0 = len(pos), len(neg)
    if not n1 or not n0:
        return float("nan")
    r = rankdata(np.r_[pos, neg])
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def zsf(z):
    from scipy.stats import norm
    return norm.sf(z)


def pca(X, k=50, seed=0):
    from sklearn.utils.extmath import randomized_svd
    X = X - X.mean(0, keepdims=True)
    k = min(k, min(X.shape) - 1)
    U, S, _ = randomized_svd(X.astype(np.float32), k, random_state=seed)
    return (U * S).astype(np.float32)


def unit(X):
    return (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def deciles(level, rng):
    q = np.quantile(level, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, level, side="right") - 1, 0, 9)
    perm = np.arange(len(level))
    for k in range(10):
        ix = np.nonzero(b == k)[0]; perm[ix] = rng.permutation(ix)
    return perm


def gz(X):
    """z-score each row (gene) across columns; rows with no variance dropped by the caller."""
    s = X.std(1, keepdims=True); return (X - X.mean(1, keepdims=True)) / np.where(s > 0, s, 1)


def save(name, obj):
    os.makedirs(RUN, exist_ok=True); json.dump(obj, open(os.path.join(RUN, name), "w"), indent=0, default=float)


# ------------------------------------------------------------------------------------------------ gene views
class View:
    def __init__(self, name, kind, genes, X=None, sets=None, level=None, local=False):
        self.name, self.kind, self.local = name, kind, local
        self.genes = list(genes); self.pos = {g: i for i, g in enumerate(self.genes)}
        self.X, self.level = X, (np.asarray(level, float) if level is not None else None)
        self.M = None
        if sets is not None:                                   # sets: gene -> list of set ids (or neighbours for PPI)
            import scipy.sparse as sp
            ids = sorted({s for v in sets.values() for s in v}); col = {s: j for j, s in enumerate(ids)}
            r, c = [], []
            for g, v in sets.items():
                for s in v:
                    r.append(self.pos[g]); c.append(col[s])
            self.M = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(self.genes), len(ids)))
            if self.level is None:
                self.level = np.asarray(self.M.sum(1)).ravel()

    def sim(self, i, j):
        if self.X is not None:
            return np.einsum("ij,ij->i", self.X[i], self.X[j])
        return (np.asarray(self.M[i].multiply(self.M[j]).sum(1)).ravel() > 0).astype(np.float32)

    def knn(self, idx, k):
        """k nearest of each gene in idx, among idx (indices into self.genes)."""
        idx = np.asarray(idx); out = np.full((len(idx), k), -1, np.int64)
        if self.X is not None:
            U = self.X[idx]
            for s in range(0, len(idx), 2048):
                S = U[s:s + 2048] @ U.T; S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -np.inf
                kk = min(k, len(idx) - 1); out[s:s + S.shape[0], :kk] = idx[np.argpartition(-S, kk, axis=1)[:, :kk]]
            return out
        import scipy.sparse as sp
        Ms = self.M[idx]; w = 1.0 / np.maximum(np.asarray(Ms.sum(0)).ravel(), 1)     # specific sets count more
        S = (Ms @ sp.diags(w) @ Ms.T).tocsr()
        for r in range(S.shape[0]):
            a, b = S.indptr[r], S.indptr[r + 1]; cols, vals = S.indices[a:b], S.data[a:b]
            keep = cols != r; cols, vals = cols[keep], vals[keep]
            if len(cols):
                t = np.argsort(-vals, kind="stable")[:k]; out[r, :len(t)] = idx[cols[t]]
        return out


def _expr(name, kind, genes, X, local=False, pc=50, min_max=None):
    X = np.asarray(X, np.float32); genes = np.asarray(genes)
    ok = np.isfinite(X).all(1) & (X.std(1) > 1e-6)
    if min_max is not None:
        ok &= X.max(1) >= min_max
    X, genes = X[ok], genes[ok]
    return View(name, kind, genes, X=unit(pca(gz(X), pc)), level=X.mean(1), local=local)


def _dedupe(genes, X):
    genes = np.asarray(genes); u, inv = np.unique(genes, return_inverse=True)
    if len(u) == len(genes):
        return genes, X
    S = np.zeros((len(u), X.shape[1]), np.float64); n = np.bincount(inv)
    np.add.at(S, inv, X); return u, (S / n[:, None]).astype(np.float32)


def ko_view(name, local, trial_genes=None):
    z = npz(name); rows, X, st = z["rows"], z["X"].astype(np.float32), z["strength"]
    ok = np.array([str(r).startswith("ENSG") for r in rows]) & np.isfinite(st)
    rows, X, st = rows[ok], X[ok], st[ok]
    keep = st >= np.median(st); rows, X, st = rows[keep], X[keep], st[keep]
    U = unit(X); _, _, Vt = np.linalg.svd(U - U.mean(0), full_matrices=False); P = Vt[:3]
    Xs = unit(U - (U @ P.T) @ P)
    rows, Xs = _dedupe(rows, Xs)
    _, st2 = _dedupe(z["rows"][ok][keep], st[:, None])
    return View(name, "KO", rows, X=unit(Xs), level=st2.ravel(), local=local)


CX_CACHE = {}


def cx_cl():
    """CELLxGENE human per cell type (CL), all tissues pooled by cells: genes x CL mean expression (me x share)."""
    if "h" in CX_CACHE:
        return CX_CACHE["h"]
    z = npz("CX_HUMAN"); pairs = json.load(open(L("CX_HUMAN_PAIRS")[1]))["pairs"]
    cols = [i for i, p in enumerate(pairs) if p["labelled"] and p["cell_type"] != "CL:0000000" and p["cells"] >= 50]
    cls = sorted({pairs[i]["cell_type"] for i in cols}); cpos = {c: j for j, c in enumerate(cls)}
    W = np.zeros(len(pairs)); cl_of = np.full(len(pairs), -1)
    for i in cols:
        W[i] = pairs[i]["cells"]; cl_of[i] = cpos[pairs[i]["cell_type"]]
    row = np.repeat(np.arange(len(z["genes"])), np.diff(z["indptr"])); col = z["col"]
    sel = cl_of[col] >= 0
    v = z["me"][sel] * 0.04 * (z["pc"][sel] / 255.0) * W[col[sel]]
    X = np.zeros((len(z["genes"]), len(cls)), np.float64)
    np.add.at(X, (row[sel], cl_of[col[sel]]), v)
    tot = np.zeros(len(cls)); np.add.at(tot, cl_of[cols], W[cols]); X /= tot[None, :]
    names = {pairs[i]["cell_type"]: pairs[i]["cell_type_name"] for i in cols}
    genes = np.array([str(g).split(".")[0] for g in z["genes"]])
    ok = X.max(1) > 0.01
    CX_CACHE["h"] = (genes[ok], cls, [names[c] for c in cls], X[ok].astype(np.float32))
    return CX_CACHE["h"]


def cx_mouse_cl():
    z = npz("CX_MOUSE"); pairs = json.load(open(L("CX_MOUSE_PAIRS")[1]))["pairs"]
    cols = [i for i, p in enumerate(pairs) if p["labelled"] and p["cell_type"] != "CL:0000000" and p["cells"] >= 50]
    cls = sorted({pairs[i]["cell_type"] for i in cols}); cpos = {c: j for j, c in enumerate(cls)}
    W = np.zeros(len(pairs)); cl_of = np.full(len(pairs), -1)
    for i in cols:
        W[i] = pairs[i]["cells"]; cl_of[i] = cpos[pairs[i]["cell_type"]]
    row = np.repeat(np.arange(len(z["genes"])), np.diff(z["indptr"])); col = z["col"]; sel = cl_of[col] >= 0
    X = np.zeros((len(z["genes"]), len(cls))); np.add.at(X, (row[sel], cl_of[col[sel]]), z["me"][sel] * 0.04 * (z["pc"][sel] / 255.0) * W[col[sel]])
    tot = np.zeros(len(cls)); np.add.at(tot, cl_of[cols], W[cols]); X /= tot[None, :]
    H = cbb.hgnc(); ens = np.array([H["sym"].get(str(s).upper(), "") for s in z["symbols"]])   # mouse symbol, upper-cased
    ok = (ens != "") & (X.max(1) > 0.01)
    g, Xm = _dedupe(ens[ok], X[ok].astype(np.float32))
    names = {pairs[i]["cell_type"]: pairs[i]["cell_type_name"] for i in cols}
    return g, cls, [names[c] for c in cls], Xm


def encode_terms():
    z = npz("RNA_ENCODE_HUMAN"); m = json.load(open(L("RNA_ENCODE_HUMAN_META")[1]))["meta"]
    terms = sorted(set(m["biosample"])); X = z["X"]
    E = np.stack([X[[i for i, b in enumerate(m["biosample"]) if b == t]].astype(np.float32).mean(0) for t in terms], 1)
    return np.array([str(c).split(".")[0] for c in z["cols"]]), terms, E


def gene_views():
    t0 = time.time(); V = {}
    for s, (name, cell, design, src) in cbb.SCREENS.items():
        if s == "F" or name in ("KO_IPSC_FTS", "KO_HESC_V"):
            continue
        V[name] = ko_view(name, src in cbb.LOCAL_ONLY)
    for name in ("LINCS_SH", "LINCS_XPR"):
        z = npz(name); rows = z["rows"]; ok = np.array([str(r).startswith("ENSG") for r in rows])
        X = z["X"][ok].astype(np.float32); g, X = _dedupe(rows[ok], X)
        E = pca(X, 50); U = unit(E); _, _, Vt = np.linalg.svd(U - U.mean(0), full_matrices=False); P = Vt[:3]
        V[name] = View(name, "KO", g, X=unit(U - (U @ P.T) @ P), level=np.linalg.norm(X, axis=1))
    g, cls, _, X = cx_cl(); V["CX_HUMAN"] = _expr("CX_HUMAN", "EXPR", g, X)
    g, cls, _, X = cx_mouse_cl(); V["CX_MOUSE"] = _expr("CX_MOUSE", "EXPR", g, X)
    g, terms, E = encode_terms(); V["RNA_ENCODE"] = _expr("RNA_ENCODE", "EXPR", g, E.astype(np.float32), min_max=math.log1p(1))
    z = npz("RNA_GTEX_V8"); V["RNA_GTEX"] = _expr("RNA_GTEX", "EXPR", [str(c).split(".")[0] for c in z["cols"]], z["X"].T.astype(np.float32), pc=30, min_max=math.log1p(1))
    z = npz("CAGE_FANTOM5_HUMAN"); ens = [cbb.to_ens(s) or "" for s in z["cols"]]; ok = np.array([e != "" for e in ens])
    g, X = _dedupe(np.array(ens)[ok], z["X"].T[ok].astype(np.float32)); V["CAGE_FANTOM5"] = _expr("CAGE_FANTOM5", "EXPR", g, X, min_max=math.log1p(1))
    z = npz("EXPR_HPA_CELLLINES"); V["EXPR_HPA"] = _expr("EXPR_HPA", "EXPR", z["rows"], z["X"].astype(np.float32), min_max=math.log1p(1))
    for name in ("EXPR_ARCHS4", "EMB_GENEFORMER"):
        z = npz(name); ok = np.array([str(r).startswith("ENSG") for r in z["rows"]]); g, X = _dedupe(z["rows"][ok], z["X"][ok].astype(np.float32))
        if name == "EXPR_ARCHS4":
            V[name] = _expr(name, "EXPR", g, X, local=True)
        else:
            V[name] = View(name, "EXPR", g, X=unit(pca(X, 50)), level=np.linalg.norm(X, axis=1))
    z = npz("PROMOTER_OPEN_HUMAN"); O = np.unpackbits(z["bits"], axis=1)[:, :int(z["n_cols"])].astype(np.float32)
    sh = O.mean(1); ok = (sh > 0.02) & (sh < 0.98); g, X = _dedupe(z["rows"][ok], O[ok])
    V["PROMOTER_OPEN"] = View("PROMOTER_OPEN", "CHROM", g, X=unit(pca(X - X.mean(1, keepdims=True), 50)), level=X.mean(1))
    z = npz("METH_PROMOTER"); B = z["beta"].T.astype(np.float32); na = B == 255; B = B / 250.0; B[na] = np.nan
    mu = np.nanmean(B, 1, keepdims=True); B = np.where(np.isnan(B), mu, B); ok = np.isfinite(B).all(1) & (B.std(1) > 0.05)
    V["METH"] = View("METH", "CHROM", z["genes"][ok], X=unit(pca(gz(B[ok]), 50)), level=B[ok].mean(1))
    z = npz("DEP_DEPMAP"); D = z["X"].astype(np.float32); mu = np.nanmean(D, 1, keepdims=True); D = np.where(np.isnan(D), mu, D)
    ok = np.isfinite(D).all(1) & (D.std(1) > 1e-3); g, D = _dedupe(z["rows"][ok], D[ok])
    V["DEP_DEPMAP"] = View("DEP_DEPMAP", "ESS", g, X=unit(pca(gz(D), 50)), level=D.mean(1))
    z = npz("PPI_HPA"); nb = collections.defaultdict(set)
    for a, b in zip(z["a"], z["b"]):
        if a != b:
            nb[str(a)].add(str(b)); nb[str(b)].add(str(a))
    V["PPI"] = ppi_view(nb)
    for k, kind in (("SETS_COMPLEX", "PHYS"), ("SETS_PATHWAY", "ANNOT"), ("SETS_LOCATION", "ANNOT"), ("SETS_DRUGTARGET", "ANNOT")):
        S = json.load(open(L(k)[1])); m = collections.defaultdict(list)
        for sid, d in S.items():
            if len(d["members"]) <= 500:
                for g in d["members"]:
                    m[g].append(sid)
        V[k[5:]] = View(k[5:], kind, list(m), sets=m)                         # sets of up to 500 members
    if TRIAL:
        keep = set(sorted({g for v in V.values() for g in v.genes}, key=lambda g: hashlib.sha256(g.encode()).hexdigest())[:3000])
        for n, v in list(V.items()):
            if n == "PPI":
                V[n] = ppi_view({g: {h for h in s if h in keep} for g, s in v.nb.items() if g in keep}); continue
            ix = [i for i, g in enumerate(v.genes) if g in keep]
            V[n] = View(n, v.kind, [v.genes[i] for i in ix], X=v.X[ix] if v.X is not None else None, level=v.level[ix], local=v.local,
                        sets=None if v.M is None else {v.genes[i]: list(v.M[i].indices) for i in ix if v.M[i].nnz})
    print(f"  {len(V)} gene views built ({time.time() - t0:.0f}s): " + ", ".join(f"{n} {len(v.genes)}" for n, v in V.items()), flush=True)
    return V


def ppi_view(nb):
    v = View("PPI", "PHYS", list(nb), sets={g: [g] for g in nb})          # placeholder incidence, replaced below
    import scipy.sparse as sp
    r, c = [], []
    for g, s in nb.items():
        for h in s:
            if h in v.pos:
                r.append(v.pos[g]); c.append(v.pos[h])
    A = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(v.genes),) * 2)
    v.A = A; v.nb = nb; v.level = np.asarray(A.sum(1)).ravel()
    v.sim = lambda i, j: np.asarray(A[i, j]).ravel().astype(np.float32)
    def knn(idx, k):
        idx = np.asarray(idx); sub = A[idx][:, idx].tocsr(); out = np.full((len(idx), k), -1, np.int64)
        for r_ in range(sub.shape[0]):
            cols = sub.indices[sub.indptr[r_]:sub.indptr[r_ + 1]][:k]; out[r_, :len(cols)] = idx[cols]
        return out
    v.knn = knn
    return v


KINDS = ["KO", "EXPR", "CHROM", "ESS", "PHYS", "ANNOT"]


# ------------------------------------------------------------------------------------------------ F1 data <-> data
F1_KNOWN = [("COMPLEX", "KO_K562_GW"), ("PPI", "COMPLEX"), ("RNA_GTEX", "CX_HUMAN"), ("PROMOTER_OPEN", "RNA_ENCODE"),
            ("DEP_DEPMAP", "COMPLEX"), ("KO_K562_GW", "KO_K562_ESS"), ("CX_MOUSE", "CX_HUMAN"), ("METH", "PROMOTER_OPEN")]


def f1_pairs(V, hv, rng, shuffle=False, plant=None):
    """For each half, ordered view pairs (X, Y): AUROC and null z. Returns {(X, Y): {h: (auc, z, p, n)}}."""
    out = collections.defaultdict(dict); K, NG, NN = 10, 3000, 20
    for h in (0, 1):
        nbr = {}
        for n, v in V.items():
            idx = np.nonzero(hv[n] == h)[0]
            if len(idx) >= 200:
                nbr[n] = (idx, v.knn(idx, K))
        for X, (idx, nb) in nbr.items():
            vx = V[X]
            for Y, vy in V.items():
                if Y == X:
                    continue
                gy = np.array([vy.pos.get(g, -1) for g in vx.genes])          # X index -> Y index
                ok = gy[idx] >= 0
                if ok.sum() < 200:
                    continue
                rows = idx[ok]; nbs = nb[ok]
                if len(rows) > NG:
                    sel = rng.choice(len(rows), NG, replace=False); rows, nbs = rows[sel], nbs[sel]
                a = np.repeat(gy[rows], K); b = gy[np.where(nbs.ravel() >= 0, nbs.ravel(), rows[0])]
                b[nbs.ravel() < 0] = -1; keep = b >= 0; a, b = a[keep], b[keep]
                pool = gy[idx[ok]]; ra = rng.choice(pool, len(a)); rb = rng.choice(pool, len(a))
                yi = np.nonzero(hv[Y] == h)[0]
                if shuffle:                                                   # negative control: Y labels destroyed
                    pm = np.arange(len(vy.genes)); pm[yi] = rng.permutation(yi)
                    a, b, ra, rb = pm[a], pm[b], pm[ra], pm[rb]
                obs = auroc(vy.sim(a, b), vy.sim(ra, rb))
                null = []
                for _ in range(NN):
                    pm = np.arange(len(vy.genes)); pm[yi] = yi[deciles(vy.level[yi], rng)]
                    null.append(auroc(vy.sim(pm[a], pm[b]), vy.sim(pm[ra], pm[rb])))
                sd = np.std(null) + 1e-6; z = (obs - np.mean(null)) / sd
                out[(X, Y)][h] = (obs, float(z), float(zsf(z)), int(len(a)))
        print(f"    F1 half {h}: {sum(1 for k in out if h in out[k])} view pairs", flush=True)
    return out


def f1_decide(res):
    car = [k for k, v in res.items() if 0 in v and 1 in v and v[0][0] >= 0.55 and v[0][2] < 0.05]
    q = bh([res[k][1][2] for k in car]); conf = [k for k, qq in zip(car, q) if qq < GATE["q"] and res[k][1][0] >= 0.55]
    return car, dict(zip(car, q)), conf


def run_f1():
    rng = np.random.default_rng(11); V = gene_views(); hv = {n: half(v.genes) for n, v in V.items()}
    res = f1_pairs(V, hv, rng); car, q, conf = f1_decide(res)
    neg = f1_pairs(V, hv, np.random.default_rng(12), shuffle=True); _, _, nconf = f1_decide(neg)
    rp = np.random.default_rng(13); plant = rp.choice([n for n in V if V[n].X is not None], 10, replace=False)
    PV = {}
    for n in plant:                                     # planted view: X's own structure plus noise
        v = V[n]; X = unit(v.X + rp.normal(0, 1.0 / math.sqrt(v.X.shape[1]), v.X.shape)).astype(np.float32)
        PV[n] = View(n, v.kind, v.genes, X=X, level=v.level); PV["PLANT_" + n] = View("PLANT_" + n, "PLANT", v.genes, X=X, level=v.level)
    pres = {}
    for n in plant:
        sub = {n: V[n], "PLANT_" + n: PV["PLANT_" + n]}; hv2 = {k: half(v.genes) for k, v in sub.items()}
        pres.update(f1_pairs(sub, hv2, rp))
    _, _, pconf = f1_decide(pres); pr = np.mean([(n, "PLANT_" + n) in pconf for n in plant])
    known = [(x, y, (x, y) in conf) for x, y in F1_KNOWN if x in V and y in V]
    rows = [dict(x=k[0], y=k[1], dev_auroc=res[k][0][0], exam_auroc=res[k][1][0], exam_p=res[k][1][2], q=q[k], genes=res[k][1][3],
                 local=V[k[0]].local or V[k[1]].local) for k in conf]
    facts = [_fact(f"LAYER-{_lname(r['x'])}", f"link/F1/{_lname(r['y'])}", r, "F1") for r in rows]
    gate = _gate(len(conf), len(nconf), pr, sum(k[2] for k in known), 6)
    return dict(tested=len(res), carried=len(car), confirmed=len(conf), negative_confirmed=len(nconf), planted_recovered=pr,
                known=[list(k) for k in known], known_needed=6, gate=gate, rows=rows,
                matrix={f"{k[0]}|{k[1]}": [res[k].get(0, (None,))[0], res[k].get(1, (None,))[0]] for k in res}), facts


def _lname(view):
    return {"COMPLEX": "SETS_COMPLEX", "PATHWAY": "SETS_PATHWAY", "LOCATION": "SETS_LOCATION", "DRUGTARGET": "SETS_DRUGTARGET",
            "PPI": "PPI_HPA", "CX_HUMAN": "CX_HUMAN", "CX_MOUSE": "CX_MOUSE", "RNA_ENCODE": "RNA_ENCODE_HUMAN", "RNA_GTEX": "RNA_GTEX_V8",
            "CAGE_FANTOM5": "CAGE_FANTOM5_HUMAN", "EXPR_HPA": "EXPR_HPA_CELLLINES", "PROMOTER_OPEN": "PROMOTER_OPEN_HUMAN",
            "METH": "METH_PROMOTER"}.get(view, view)


# ------------------------------------------------------------------------------------------------ F2 gene <-> gene
def _pairs_of(v, k, perm=None):
    nb = v.knn(np.arange(len(v.genes)), k); a = np.repeat(np.arange(len(v.genes)), k); b = nb.ravel(); ok = b >= 0
    a, b = a[ok], b[ok]
    if perm is not None:
        a, b = perm[a], perm[b]
    return a, b


def f2_support(V, gid, perms=None, kinds=KINDS, plant=None):
    """Unique gene pairs (global ids) and the number of kinds supporting each."""
    N = len(gid); per = []
    for kind in kinds:
        keys = []
        for n, v in V.items():
            if v.kind != kind:
                continue
            a, b = _pairs_of(v, 20, None if perms is None else perms[n])
            ga, gb = gid[n][a], gid[n][b]; lo, hi = np.minimum(ga, gb), np.maximum(ga, gb)
            keys.append(lo.astype(np.int64) * N + hi)
            if plant is not None and kind in plant["kinds"]:
                keys.append(plant["keys"])
        if keys:
            per.append(np.unique(np.concatenate(keys)))
    allk = np.concatenate(per); u, c = np.unique(allk, return_counts=True)
    return u, c


def run_f2():
    rng = np.random.default_rng(21); V = gene_views(); NB = {}
    for n, v in V.items():
        NB[n] = v.knn(np.arange(len(v.genes)), 20)
        v.knn = (lambda nb: (lambda idx, k: nb[idx][:, :k]))(NB[n])                 # cache: every caller asks for all genes
    allg = sorted({g for v in V.values() for g in v.genes}); G = {g: i for i, g in enumerate(allg)}; N = len(allg)
    gid = {n: np.array([G[g] for g in v.genes]) for n, v in V.items()}
    u, c = f2_support(V, gid)
    null = []
    for r in range(3):
        rr = np.random.default_rng(100 + r); perms = {n: deciles(v.level, rr) for n, v in V.items()}
        _, cn = f2_support(V, gid, perms); null.append(np.array([(cn >= s).sum() for s in range(1, 7)]))
    n0 = np.mean(null, 0); n1 = np.array([(c >= s).sum() for s in range(1, 7)]); fdr = n0 / np.maximum(n1, 1)
    s_star = next((s for s in range(2, 7) if fdr[s - 1] <= GATE["q"]), None)
    conf = u[c >= s_star] if s_star else np.array([], np.int64)
    pa = rng.choice(N, 300); pb = rng.choice(N, 300); keys = np.minimum(pa, pb).astype(np.int64) * N + np.maximum(pa, pb)
    up, cp = f2_support(V, gid, plant=dict(kinds=["KO", "EXPR", "CHROM", "ESS"], keys=keys))
    hit = np.isin(keys, up[cp >= (s_star or 99)]).mean()
    u4, c4 = f2_support(V, gid, kinds=["KO", "EXPR", "CHROM", "ESS"]); found = u4[c4 >= 3]
    truth = set()
    for n in ("COMPLEX",):
        v = V[n]; C = (v.M @ v.M.T).tocoo(); m = C.row < C.col
        truth |= set((np.minimum(gid[n][C.row[m]], gid[n][C.col[m]]).astype(np.int64) * N + np.maximum(gid[n][C.row[m]], gid[n][C.col[m]])).tolist())
    A = V["PPI"].A.tocoo(); m = A.row < A.col
    truth |= set((np.minimum(gid["PPI"][A.row[m]], gid["PPI"][A.col[m]]).astype(np.int64) * N + np.maximum(gid["PPI"][A.row[m]], gid["PPI"][A.col[m]])).tolist())
    gk = sorted(set(np.concatenate([gid[n] for n in ("COMPLEX", "PPI")]).tolist()))
    base = len(truth) / (len(gk) * (len(gk) - 1) / 2)
    inside = np.isin(found // N, gk) & np.isin(found % N, gk); prec = np.mean([k in truth for k in found[inside].tolist()]) if inside.any() else 0.0
    enrich = prec / base if base else 0.0
    support_kinds = {}
    for kind in KINDS:
        uk, _ = f2_support(V, gid, kinds=[kind]); support_kinds[kind] = np.isin(conf, uk)
    rows = []
    for i, k in enumerate(conf.tolist()):
        a, b = allg[k // N], allg[k % N]; ks = [kd for kd in KINDS if support_kinds[kd][i]]
        rows.append(dict(a=a, b=b, kinds=ks, n=len(ks), local=True))
    loc = {n for n, v in V.items() if v.local}
    facts = []
    for r in rows:
        for x, y in ((r["a"], r["b"]), (r["b"], r["a"])):
            facts.append(_fact(x, f"link/F2/{y}", {"kinds": r["kinds"]}, "F2"))
    ok_planted = hit >= GATE["planted"]; ok = bool(s_star) and ok_planted and enrich >= 10
    return dict(genes=N, pairs_any=int(len(u)), by_support={str(s): int(n1[s - 1]) for s in range(1, 7)},
                null_by_support={str(s): float(n0[s - 1]) for s in range(1, 7)}, fdr={str(s): float(fdr[s - 1]) for s in range(1, 7)},
                s_star=s_star, confirmed=int(len(conf)), planted_recovered=float(hit), heldout_precision=float(prec), base_rate=float(base),
                enrichment=float(enrich), gate=dict(passed=ok, verdict="PASS" if ok else "FAIL"), rows=rows[:200000],
                local_views=sorted(loc)), facts


# ------------------------------------------------------------------------------------------------ F3 cell type <-> cell type
F3_KNOWN = [("ENC:K562", "HPA:K-562"), ("ENC:HepG2", "HPA:Hep-G2"), ("GTX:Liver", "CXH:hepatocyte"), ("GTX:Pancreas", "CXH:pancreatic acinar cell"),
            ("CXM:hepatocyte", "CXH:hepatocyte"), ("MET:Liver-Hep", "CXH:hepatocyte"), ("GTX:Whole Blood", "CXH:neutrophil")]


def f3_units():
    D = {}
    g, cls, names, X = cx_cl(); D["CXH"] = (g, [f"CXH:{n}" for n in names], X, [f"CT-{c}" for c in cls])
    g, cls, names, X = cx_mouse_cl(); D["CXM"] = (g, [f"CXM:{n}" for n in names], X, [f"CTMOUSE-{c}" for c in cls])
    g, terms, E = encode_terms(); D["ENC"] = (g, [f"ENC:{t}" for t in terms], E, [f"BIO-{t}" for t in terms])
    z = npz("RNA_GTEX_V8"); D["GTX"] = (np.array([str(c).split(".")[0] for c in z["cols"]]), [f"GTX:{r}" for r in z["rows"]], z["X"].T.astype(np.float32), [f"GTEX-{r}" for r in z["rows"]])
    z = npz("CAGE_FANTOM5_HUMAN"); meta = json.load(open(L("CAGE_FANTOM5_HUMAN_META")[1]))["meta"]
    lab = [re.sub(r",? ?(biol_rep\d+|tech_rep\d+|donor ?\d+|rep\d+)", "", d, flags=re.I).strip(" ,") for d in meta["description"]]
    ul = sorted(set(lab)); Xf = z["X"].astype(np.float32); A = np.stack([Xf[[i for i, x in enumerate(lab) if x == u_]].mean(0) for u_ in ul], 1)
    ens = np.array([cbb.to_ens(s) or "" for s in z["cols"]]); ok = ens != ""; gg, A = _dedupe(ens[ok], A[ok])
    D["F5H"] = (gg, [f"F5H:{u_}" for u_ in ul], A, [f"FANTOM5-{u_}" for u_ in ul])
    z = npz("EXPR_HPA_CELLLINES"); D["HPA"] = (z["rows"], [f"HPA:{c}" for c in z["cols"]], z["X"].astype(np.float32), [f"CELLLINE-{c}" for c in z["cols"]])
    z = npz("METH_PROMOTER"); ct = [re.sub(r"^GSM\d+_", "", s).rsplit("-Z", 1)[0] for s in z["samples"]]; uc = sorted(set(ct))
    B = z["beta"].astype(np.float32); B[B == 255] = np.nan; B /= 250.0
    M = np.stack([np.nanmean(B[[i for i, c in enumerate(ct) if c == u_]], 0) for u_ in uc], 1)
    D["MET"] = (z["genes"], [f"MET:{u_}" for u_ in uc], -M, [f"METH-{u_}" for u_ in uc])           # sign flipped: methylation silences
    z = npz("PROMOTER_OPEN_HUMAN"); O = np.unpackbits(z["bits"], axis=1)[:, :int(z["n_cols"])].astype(np.float32)
    bs = z["col_biosample"]; ub = sorted(set(bs.tolist())); P_ = np.stack([O[:, bs == b].mean(1) for b in ub], 1)
    D["PRO"] = (z["rows"], [f"PRO:{b}" for b in ub], P_, [f"BIO-{b}" for b in ub])
    return D


def f3_matrix(D, gsub=None):
    """Gene-z profiles of every unit over the genes shared by >= 5 datasets; returns names, entities, dataset, Z (genes x units)."""
    cnt = collections.Counter(g for g_, *_ in D.values() for g in set(np.asarray(g_).tolist()) if str(g).startswith("ENSG"))
    genes = sorted(g for g, n in cnt.items() if n >= 5)
    if gsub is not None:
        genes = [g for g in genes if g in gsub]
    gp = {g: i for i, g in enumerate(genes)}; cols, names, ents, ds = [], [], [], []
    for k, (g_, nm, X, en) in D.items():
        X = np.asarray(X, np.float32); Z = np.full((len(genes), X.shape[1]), np.nan, np.float32)
        ok = np.isfinite(X).all(1) & (X.std(1) > 1e-6); g_ = np.asarray(g_)
        ix = [(gp[g], i) for i, g in enumerate(g_.tolist()) if ok[i] and g in gp]
        if not ix:
            continue
        a, b = np.array(ix).T; Z[a] = gz(X[b])
        cols.append(Z); names += nm; ents += en; ds += [k] * len(nm)
    return genes, names, ents, np.array(ds), np.concatenate(cols, 1)


def _corr(Z, rows):
    Y = np.nan_to_num(Z[rows]); m = np.isfinite(Z[rows]).astype(np.float32)
    Y = Y - (Y.sum(0) / np.maximum(m.sum(0), 1))[None, :] * m
    Y /= np.linalg.norm(Y, axis=0, keepdims=True) + 1e-9
    return Y.T @ Y


def f3_run(Z, ds, gh, shuffle_ds=None, rng=None):
    Zs = Z.copy()
    if shuffle_ds is not None:
        for d in np.unique(ds):
            c = ds == d; Zs[:, c] = Zs[rng.permutation(len(Zs))][:, c]
    Sd, Se = _corr(Zs, gh == 0), _corr(Zs, gh == 1); n = len(ds); car = []
    for a in range(n):
        for d in np.unique(ds):
            c = np.nonzero(ds == d)[0]; c = c[c != a]
            if len(c) < 3:
                continue
            top = c[np.argsort(-Sd[a, c])[:3]]
            mu, sd = Se[a, c].mean(), Se[a, c].std() + 1e-9
            for b in top:
                car.append((a, int(b), float(Sd[a, b]), float(Se[a, b]), float((Se[a, b] - mu) / sd)))
    p = zsf(np.array([x[4] for x in car])); q = bh(p)
    conf = [(x, qq) for x, qq in zip(car, q) if qq < GATE["q"] and x[3] > 0]
    return car, conf


def run_f3():
    D = f3_units(); genes, names, ents, ds, Z = f3_matrix(D)
    if TRIAL:
        keep = np.isin(ds, ["GTX", "MET", "ENC"]); Z, ds = Z[:, keep], ds[keep]; names = [n for n, k in zip(names, keep) if k]; ents = [e for e, k in zip(ents, keep) if k]
    gh = half(genes); car, conf = f3_run(Z, ds, gh)
    _, nconf = f3_run(Z, ds, gh, shuffle_ds=True, rng=np.random.default_rng(31))
    rp = np.random.default_rng(32); pl = rp.choice(len(names), 20, replace=False); dsets = np.unique(ds); hits = 0
    Zp, dsp = Z.copy(), ds.copy(); targets = []
    for a in pl:
        d = rp.choice([x for x in dsets if x != ds[a]]); col = np.where(np.isfinite(Z[:, a]), Z[:, a], 0) + rp.normal(0, 1.0, len(genes))
        Zp = np.c_[Zp, col]; dsp = np.r_[dsp, d]; targets.append((Zp.shape[1] - 1, a))
    _, pconf = f3_run(Zp, dsp, gh); pc = {(x[0][0], x[0][1]) for x in pconf}
    hits = np.mean([(t, a) in pc for t, a in targets])
    cs = {(names[x[0][0]], names[x[0][1]]) for x in conf}
    known = [(a, b, (a, b) in cs or (b, a) in cs) for a, b in F3_KNOWN]
    rows = [dict(a=names[x[0]], b=names[x[1]], a_entity=ents[x[0]], b_entity=ents[x[1]], dev_r=x[2], exam_r=x[3], exam_z=x[4], q=qq) for x, qq in conf]
    facts = [_fact(r["a_entity"], f"link/F3/{r['b_entity']}", {k: r[k] for k in ("a", "b", "dev_r", "exam_r", "exam_z", "q")}, "F3") for r in rows]
    gate = _gate(len(conf), len(nconf), hits, sum(k[2] for k in known), 5)
    return dict(units=len(names), datasets=dict(collections.Counter(ds.tolist())), genes=len(genes), carried=len(car), confirmed=len(conf),
                negative_confirmed=len(nconf), planted_recovered=float(hits), known=[list(k) for k in known], known_needed=5, gate=gate, rows=rows), facts


# ------------------------------------------------------------------------------------------------ F4 disease <-> cell type
F4_KNOWN = [("EFO_0004611", "liver"), ("EFO_0004587", "lymphoid"), ("EFO_0004305", "erythroid"), ("EFO_0003767", "immune"),
            ("EFO_0000275", "heart"), ("MONDO_0005090", "brain"), ("EFO_0005091", "myeloid"), ("OBA_0003747", "kidney")]


def f4_traits():
    """One GWAS study per disease (the one with most credible sets), >= 30 credible sets, up to 2,000."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    p = os.path.join(t2c.CACHE, "gwas_counts.json")
    if not os.path.exists(p):
        t2c.gwas_studies()
    n = json.load(open(p))
    s = pq.read_table(os.path.join(t2c.OT, "study_0000.parquet"), columns=["studyId", "studyType", "traitFromSource", "nSamples", "diseaseIds"])
    s = s.filter(pc.equal(s["studyType"], "gwas")).to_pylist(); best = {}
    for r in s:
        k = n.get(r["studyId"], 0); tr = r["traitFromSource"] or ""
        if k < 30 or re.search(t2c.EXCLUDE, tr, re.I):
            continue
        key = tuple(sorted(r["diseaseIds"] or [])) or ("TXT:" + tr.lower().strip(),)
        cand = (k, r["nSamples"] or 0, r["studyId"])
        if key not in best or cand > best[key][0]:
            best[key] = (cand, r)
    out = sorted(best.values(), key=lambda x: (-x[0][0], x[0][2]))[:2000]
    if TRIAL:
        want = {d for d, _ in F4_KNOWN}; out = [x for x in out if want & set(x[1]["diseaseIds"] or [])][:8] + out[:32]
    return [dict(studyId=r["studyId"], trait=r["traitFromSource"], diseaseIds=r["diseaseIds"] or [], n=r["nSamples"], cs=c[0]) for c, r in out]


def f4_variants(traits):
    """Per trait, credible-set variants: chrom, pos, pip, credible set id (pip >= 0.001), cached."""
    import pyarrow.parquet as pq
    p = os.path.join(RUN, "f4_variants.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    sid = {t["studyId"]: i for i, t in enumerate(traits)}; T, CH, PS, PP, CS, VID = [], [], [], [], [], []
    for f in sorted(os.listdir(t2c.OT)):
        if not f.startswith("credible_set_"):
            continue
        t = pq.read_table(os.path.join(t2c.OT, f), columns=["studyId", "studyLocusId", "locus"], filters=[("studyId", "in", list(sid))])
        for s_, cs, loc in zip(t["studyId"].to_pylist(), t["studyLocusId"].to_pylist(), t["locus"].to_pylist()):
            for v in loc or []:
                w = float(v["posteriorProbability"] or 0); x = v["variantId"].split("_")
                if w >= 0.001 and len(x) >= 4 and x[1].isdigit():
                    T.append(sid[s_]); CH.append(x[0]); PS.append(int(x[1])); PP.append(w); CS.append(cs); VID.append(v["variantId"])
    out = dict(trait=np.array(T, np.int32), chrom=np.array(["chr" + c for c in CH]), pos=np.array(PS, np.int64), pip=np.array(PP, np.float32),
               cs=np.array(CS), vid=np.array(VID))
    os.makedirs(RUN, exist_ok=True); np.savez_compressed(p, **out)
    return out


def _chrom_half(ch):
    c = np.char.replace(ch.astype(str), "chr", ""); h = np.full(len(c), -1, np.int8)
    num = np.char.isdigit(c); h[num] = (c[num].astype(int) % 2 == 0).astype(np.int8)          # odd autosomes 0 (DEV), even 1 (EXAM)
    return h


def f4_scores(T_cols, reg, ann, br, eh):
    """t statistics per (column, annotation, half). T_cols: elements x columns CSR (PIP sums)."""
    import scipy.sparse as sp
    N = len(br); out = np.zeros((2, T_cols.shape[1], len(ann)), np.float32); G = {}
    for h in (0, 1):
        m = (eh == h).astype(np.float64); Th = sp.diags(m) @ T_cols; Th = Th.tocsr()
        G[h] = dict(N=m.sum(), Sbr=(br * m).sum(), Sbr2=(br * br * m).sum(), St=np.asarray(Th.sum(0)).ravel(),
                    Sbrt=np.asarray(Th.T @ br).ravel(), Stt=np.asarray(Th.multiply(Th).sum(0)).ravel(), T=Th)
    for j, b in enumerate(ann):
        a = t2c.load_open(b)
        for h in (0, 1):
            g = G[h]; ah = a[eh[a] == h]; n = len(ah); sb = br[ah].sum(); stb = np.asarray(g["T"][ah].sum(0)).ravel()
            XtX = np.array([[g["N"], g["Sbr"], n], [g["Sbr"], g["Sbr2"], sb], [n, sb, n]])
            inv = np.linalg.inv(XtX); xty = np.stack([g["St"], g["Sbrt"], stb])          # 3 x cols
            beta = inv @ xty; rss = g["Stt"] - (beta * xty).sum(0); s2 = np.maximum(rss, 1e-300) / (g["N"] - 3)
            out[h, :, j] = beta[2] / np.sqrt(s2 * inv[2, 2])
        if j % 200 == 0:
            print(f"    F4 annotations {j}/{len(ann)}", flush=True)
    return out


def f4_decide(Zs, cols):
    car = []
    for c in cols:
        d = Zs[0, c]; top = np.argsort(-d)[:50]; car += [(c, int(j)) for j in top if d[j] >= 3]
    p = zsf(np.array([Zs[1, c, j] for c, j in car])) if car else np.array([])
    q = bh(p); conf = [(c, j, float(Zs[0, c, j]), float(Zs[1, c, j]), float(qq)) for (c, j), qq in zip(car, q) if qq < GATE["q"] and Zs[1, c, j] > 0]
    return car, conf


def run_f4():
    import scipy.sparse as sp
    t0 = time.time(); traits = f4_traits(); FAM[0] = "F4"; L("GWAS_OPENTARGETS"); L("CCRE_HUMAN")
    tv = f4_variants(traits); reg = e2.Registry(); ann = t2c.dnase_annotations(reg); N = len(reg.chrom)
    el = t2c.elem_of(reg, tv["chrom"], tv["pos"] - 1); ok = el >= 0
    cnt = np.zeros(N, np.int32)
    for b in ann:
        cnt[t2c.load_open(b)] += 1
    br = cnt / len(ann); eh = _chrom_half(reg.chrom)
    cols = [sp.csr_matrix((tv["pip"][ok], (el[ok], tv["trait"][ok])), shape=(N, len(traits)))]
    rng = np.random.default_rng(41); clen = {c: int(reg.end[reg.chrom == c].max()) for c in np.unique(reg.chrom)}
    neg_src = rng.choice(len(traits), 30, replace=False)
    for k, i in enumerate(neg_src):                         # negative: a real trait shifted along its chromosomes
        m = tv["trait"] == i; off = rng.integers(5_000_000, 50_000_000)
        pos = np.array([(p_ + off) % clen.get(c, p_ + off + 1) for c, p_ in zip(tv["chrom"][m], tv["pos"][m])])
        e_ = t2c.elem_of(reg, tv["chrom"][m], pos - 1); o = e_ >= 0
        cols.append(sp.csr_matrix((tv["pip"][m][o], (e_[o], np.zeros(o.sum(), int))), shape=(N, 1)))
    plant = rng.choice(len(ann), 20, replace=False)
    for j in plant:                                         # positive: variants in elements open (specifically) in one annotation
        a = t2c.load_open(ann[j]); a = a[br[a] < 0.10]; pick = rng.choice(a, min(200, len(a)), replace=False); bg = rng.integers(0, N, 1000)
        cols.append(sp.csr_matrix((np.r_[np.full(len(pick), 0.5), np.full(1000, 0.3)], (np.r_[pick, bg], np.zeros(len(pick) + 1000, int))), shape=(N, 1)))
    Tm = sp.hstack(cols).tocsr(); nt = len(traits)
    Zs = f4_scores(Tm, reg, ann, br, eh); np.save(os.path.join(RUN, "f4_scores.npy"), Zs)
    _, conf = f4_decide(Zs, range(nt)); _, nconf = f4_decide(Zs, range(nt, nt + 30))
    _, pconf = f4_decide(Zs, range(nt + 30, nt + 50))
    pr = np.mean([any(c == nt + 30 + k and ann[j_]["biosample"] == ann[plant[k]]["biosample"] for c, j_, *_ in pconf) for k in range(20)])
    sl = t2c.slims(reg.meta); cat = [t2c.categories(b["biosample"], sl) for b in ann]
    known = []
    for did, c_ in F4_KNOWN:
        ti = [i for i, t in enumerate(traits) if did in t["diseaseIds"]]
        known.append((did, c_, any(c in ti and c_ in cat[j] for c, j, *_ in conf)))
    rows = [dict(trait=traits[c]["trait"], study=traits[c]["studyId"], diseaseIds=traits[c]["diseaseIds"], annotation=ann[j]["annotation"],
                 biosample=ann[j]["biosample"], categories=sorted(cat[j]), dev_t=dt, exam_t=et, q=q) for c, j, dt, et, q in conf]
    facts = [_fact(f"TRAIT-{r['study']}", f"link/F4/BIO-{r['biosample']}", {k: r[k] for k in ("annotation", "dev_t", "exam_t", "q")}, "F4") for r in rows]
    facts += [Fact(f"TRAIT-{t['studyId']}", "trait", t, time.strftime("%Y-%m-%d"), cbb.now(), "reported", cbb.source("opentargets_genetics"),
                   inputs=tuple(sorted(USED["F4"]))) for t in traits]
    np.save(os.path.join(RUN, "f4_ann.npy"), np.array([b["annotation"] for b in ann])); save("f4_traits.json", traits)
    gate = _gate(len(conf), len(nconf), pr, sum(k[2] for k in known), 6)
    print(f"    F4 done ({time.time() - t0:.0f}s)", flush=True)
    return dict(traits=nt, annotations=len(ann), confirmed=len(conf), trait_terms=len({(r["study"], r["biosample"]) for r in rows}),
                traits_with_link=len({r["study"] for r in rows}), negative_confirmed=len(nconf), planted_recovered=float(pr),
                known=[list(k) for k in known], known_needed=6, gate=gate, rows=rows), facts


# ------------------------------------------------------------------------------------------------ F5 shared causal variants
def run_f5():
    import pandas as pd
    import tarfile
    traits = json.load(open(os.path.join(RUN, "f4_traits.json"))); tv = f4_variants(traits); FAM[0] = "F5"
    L("QTL_GTEX_V11"); L("SCEQTL_IBDVERSE"); L("GWAS_OPENTARGETS")
    gw = pd.DataFrame(dict(trait=tv["trait"], cs=tv["cs"], v=tv["vid"], p1=tv["pip"])); gw = gw[gw.p1 >= 0.01]
    out = []
    tar = os.path.join(t2c.AG, "genetics", "gtex_v11", "GTEx_v11_eQTL_SuSiE.tar"); gd = os.path.join(t2c.CACHE, "gtex"); os.makedirs(gd, exist_ok=True)
    with tarfile.open(tar) as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".parquet")]
        if TRIAL:
            members = [m for m in members if "Liver" in m.name or "Whole_Blood" in m.name]
        for m in members:
            tissue = m.name.split("/")[-1].split(".v11")[0]; p = os.path.join(gd, tissue + ".parquet")
            if not os.path.exists(p):
                open(p + ".part", "wb").write(tf.extractfile(m).read()); os.replace(p + ".part", p)
            e = pd.read_parquet(p, columns=["phenotype_id", "variant_id", "pip", "cs_id"]); e = e[e.pip >= 0.01]
            e["v"] = e.variant_id.str.replace("_b38", "", regex=False).str.replace("^chr", "", regex=True)
            e["gene"] = e.phenotype_id.str.split(".").str[0]; out.append(_clpp(gw, e, tissue, "GTEx"))
    for f in sorted(os.listdir(os.path.join(t2c.AG, "genetics", "ibdverse"))):
        if TRIAL and len(out) > 4:
            break
        e = pd.read_parquet(os.path.join(t2c.AG, "genetics", "ibdverse", f), columns=["variant", "gene_id", "cs_id", "pip"]); e = e[e.pip >= 0.01]
        e["v"] = e.variant.str.replace("^chr", "", regex=True); e["gene"] = e.gene_id.str.split(".").str[0]; e["phenotype_id"] = e.gene
        out.append(_clpp(gw, e, f.split(".")[0], "IBDverse"))
    df = pd.concat(out) if out else pd.DataFrame(columns=["trait", "gene", "tissue", "source", "clpp"])
    df = df[df.clpp >= 0.1]; best = df.sort_values("clpp", ascending=False).drop_duplicates(["trait", "gene", "tissue"])
    best.to_parquet(os.path.join(RUN, "f5_links.parquet"))
    H = cbb.hgnc(); ldl = [i for i, t in enumerate(traits) if "EFO_0004611" in t["diseaseIds"]]
    sym = best[best.trait.isin(ldl) & best.tissue.str.contains("Liver")].gene.map(lambda g: H["e2s"].get(g, g)).tolist()
    known = any(s in ("SORT1", "CELSR2", "PSRC1") for s in sym)
    rows = [dict(trait=traits[int(r.trait)]["trait"], study=traits[int(r.trait)]["studyId"], gene=r.gene, symbol=H["e2s"].get(r.gene, r.gene),
                 tissue=r.tissue, source=r.source, clpp=float(r.clpp)) for r in best.itertuples()]
    facts = [_fact(f"TRAIT-{r['study']}", f"link/F5/{r['gene']}", {k: r[k] for k in ("tissue", "source", "clpp")}, "F5") for r in rows]
    return dict(links=len(rows), traits_with_gene=len({r["study"] for r in rows}), genes=len({r["gene"] for r in rows}),
                ldl_liver_genes=sorted(set(sym))[:30], known_sort1_locus=bool(known), gate=dict(passed=None, verdict="DESCRIPTIVE"), rows=rows), facts


def _clpp(gw, e, tissue, src):
    m = gw.merge(e[["v", "gene", "cs_id", "pip"]], on="v"); m["w"] = m.p1 * m.pip
    g = m.groupby(["trait", "cs", "gene", "cs_id"], as_index=False).w.sum().rename(columns={"w": "clpp"})
    g = g.groupby(["trait", "gene"], as_index=False).clpp.max(); g["tissue"] = tissue; g["source"] = src
    print(f"    F5 {src} {tissue}: {int((g.clpp >= 0.1).sum())} links", flush=True)
    return g


# ------------------------------------------------------------------------------------------------ F6 knockdown <-> cell identity
F6_KNOWN = ["GATA1", "TAL1", "KLF1", "LMO2", "LDB1", "ZFPM1", "NFE2"]


def run_f6():
    import h5py
    g, cls, names, X = cx_cl(); FAM[0] = "F6"
    Zc = gz(np.log1p(X * 10)); spec = dict(zip(g, range(len(g))))
    res_all, facts, summ = [], [], {}
    for lay, h5 in (("KO_K562_GENESPACE", "GWPS_K562_H5AD"), ("KO_K562ESS_GENESPACE", "ESS_K562_H5AD")):
        z = npz(lay); _, hp = L(h5)
        with h5py.File(hp, "r") as f:
            pv = f["obs"]["energy_test_p_value"][:]
            lev = f["var"]["mean"][:] if "mean" in f["var"] else None
        keep = pv < 0.05; Rm = z["X"][keep].astype(np.float32); kd = z["rows"][keep]; lab = z["labels"][keep]
        cols = np.array([str(c) for c in z["cols"]]); gi = np.array([spec.get(c, -1) for c in cols]); ok = gi >= 0
        Rm, S = Rm[:, ok], Zc[gi[ok]]; lev = (lev[ok] if lev is not None else np.abs(Rm).mean(0)); gh = half(cols[ok])
        if TRIAL:
            Rm, kd, lab = Rm[:300], kd[:300], lab[:300]
        def zs(R_, S_, h, rng, nperm=10):
            m = gh == h; A = gz(R_[:, m]); B = gz(S_[m].T).T; r = A @ B / m.sum()
            nl = []
            for _ in range(nperm):
                p = deciles(lev[m], rng); nl.append(A @ B[p] / m.sum())
            return r / (np.std(nl, 0) + 1e-6)
        def decide(R_, S_, rng):
            zd, ze = zs(R_, S_, 0, rng), zs(R_, S_, 1, rng); car = []
            for i in range(len(R_)):
                o = np.argsort(-np.abs(zd[i]))[:20]; car += [(i, int(j)) for j in o if abs(zd[i, j]) >= 4]
            p = zsf(np.array([np.sign(zd[i, j]) * ze[i, j] for i, j in car])) if car else np.array([])
            q = bh(p)
            return [(i, j, float(zd[i, j]), float(ze[i, j]), float(qq)) for (i, j), qq in zip(car, q) if qq < GATE["q"] and np.sign(zd[i, j]) == np.sign(ze[i, j])]
        rng = np.random.default_rng(61); conf = decide(Rm, S, rng)
        Rn = np.stack([r[rng.permutation(len(r))] for r in Rm]); nconf = decide(Rn, S, rng)
        cj = rng.choice(len(cls), 50); Pm = np.stack([0.3 * gz(S[:, j][None, :])[0] + rng.normal(0, 1, S.shape[0]) for j in cj]).astype(np.float32)
        pconf = decide(Pm, S, rng); pr = np.mean([any(i == k and j == cj[k] for i, j, *_ in pconf) for k in range(50)])
        sym = [cbb.hgnc()["e2s"].get(str(k), str(lab[i]).split("_")[1] if "_" in str(lab[i]) else str(k)) for i, k in enumerate(kd)]
        ery = [j for j, n in enumerate(names) if re.search(r"erythro", n, re.I)]
        known = [(s, any(sym[i] == s and j in ery and zd_ < 0 for i, j, zd_, *_ in conf)) for s in F6_KNOWN]
        rows = [dict(screen=lay, knockdown=sym[i], knockdown_ens=str(kd[i]), cell_type=names[j], cl=cls[j], dev_z=a, exam_z=b, q=q) for i, j, a, b, q in conf]
        res_all += rows
        summ[lay] = dict(knockdowns=len(Rm), cell_types=len(cls), genes=int(ok.sum()), confirmed=len(conf), negative_confirmed=len(nconf),
                         planted_recovered=float(pr), known=[list(k) for k in known])
    for r in res_all:
        facts.append(_fact(r["knockdown_ens"], f"link/F6/CT-{r['cl']}", {k: r[k] for k in ("screen", "cell_type", "dev_z", "exam_z", "q")}, "F6"))
    gw = summ["KO_K562_GENESPACE"]
    both = {(r["knockdown"], r["cl"], np.sign(r["exam_z"])) for r in res_all if r["screen"] == "KO_K562_GENESPACE"} & \
           {(r["knockdown"], r["cl"], np.sign(r["exam_z"])) for r in res_all if r["screen"] == "KO_K562ESS_GENESPACE"}
    gate = _gate(gw["confirmed"], gw["negative_confirmed"], gw["planted_recovered"], sum(k[1] for k in gw["known"]), 4)
    return dict(screens=summ, confirmed=len(res_all), replicated_in_both_screens=len(both), gate=gate, rows=res_all), facts


# ------------------------------------------------------------------------------------------------ F7 gene set <-> cell type
F7_KNOWN = [(r"Erythrocytes take up oxygen", r"erythro"), (r"Generation of second messenger molecules", r"T cell"),
            (r"Surfactant metabolism", r"pneumocyte|alveolar"), (r"^Keratinization$", r"keratinocyte"),
            (r"Striated Muscle Contraction", r"muscle|cardiomyocyte|myocyte"), (r"Bile acid and bile salt metabolism", r"hepatocyte"),
            (r"Neurotransmitter receptors and postsynaptic", r"neuron")]


def run_f7():
    import scipy.sparse as sp
    g, cls, names, X = cx_cl(); FAM[0] = "F7"; Zc = gz(np.log1p(X * 10)); gp = {x: i for i, x in enumerate(g)}; gh = half(g)
    lev = X.mean(1); rng = np.random.default_rng(71)
    sets = {}
    for k in ("SETS_COMPLEX", "SETS_PATHWAY"):
        for sid, d in json.load(open(L(k)[1])).items():
            m = [gp[x] for x in d["members"] if x in gp]
            if len(m) >= 6 and len(m) <= 300:
                sets[(k, sid)] = (d["name"], np.array(m))
    if TRIAL:
        sets = dict(list(sets.items())[:200])
    keys = list(sets)

    def score(mem_lists, h):
        r, c = [], []
        for i, m in enumerate(mem_lists):
            mm = m[gh[m] == h]; r += [i] * len(mm); c += mm.tolist()
        W = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(mem_lists), len(g))); n = np.asarray(W.sum(1)).ravel()
        return (W @ Zc) / np.maximum(n, 1)[:, None], n

    qd = np.quantile(lev, np.linspace(0, 1, 11)); dec = np.clip(np.searchsorted(qd, lev, side="right") - 1, 0, 9)

    def null_z(mem_lists, h, nrep=100):
        """z against random sets with the same number of members per expression-level decile (same half)."""
        obs, n = score(mem_lists, h); pools = [np.nonzero((dec == k) & (gh == h))[0] for k in range(10)]
        si = np.concatenate([np.full(int((gh[m] == h).sum()), i) for i, m in enumerate(mem_lists)])
        md = np.concatenate([dec[m[gh[m] == h]] for m in mem_lists]); S1 = np.zeros_like(obs); S2 = np.zeros_like(obs)
        for _ in range(nrep):
            dr_ = np.empty(len(md), np.int64)
            for k in range(10):
                w = md == k; dr_[w] = rng.choice(pools[k], int(w.sum()))
            W = sp.csr_matrix((np.ones(len(si)), (si, dr_)), shape=(len(mem_lists), len(g)))
            s = (W @ Zc) / np.maximum(n, 1)[:, None]; S1 += s; S2 += s * s
        mu = S1 / nrep; sd = np.sqrt(np.maximum(S2 / nrep - mu * mu, 1e-12))
        return (obs - mu) / sd, n

    def decide(mem_lists):
        zd, nd = null_z(mem_lists, 0); ze, ne = null_z(mem_lists, 1); car = []
        for i in range(len(mem_lists)):
            if nd[i] >= 3 and ne[i] >= 3:
                o = np.argsort(-zd[i])[:10]; car += [(i, int(j)) for j in o if zd[i, j] >= 4]
        p = zsf(np.array([ze[i, j] for i, j in car])) if car else np.array([]); q = bh(p)
        return [(i, j, float(zd[i, j]), float(ze[i, j]), float(qq)) for (i, j), qq in zip(car, q) if qq < GATE["q"] and ze[i, j] > 0]
    mem = [sets[k][1] for k in keys]; conf = decide(mem)
    neg = [rng.choice(len(g), len(m), replace=False) for m in mem]; nconf = decide(neg)
    pc = rng.choice(len(cls), 30, replace=False); pl = []
    for j in pc:
        top = np.argsort(-Zc[:, j])[:200]; pl.append(np.r_[rng.choice(top, 6, replace=False), rng.choice(len(g), 14)])
    pconf = decide(pl); pr = np.mean([any(i == k and j == pc[k] for i, j, *_ in pconf) for k in range(30)])
    known = []
    for pat, cpat in F7_KNOWN:
        ii = [i for i, k in enumerate(keys) if re.search(pat, sets[k][0])]
        known.append((pat, cpat, any(i in ii and re.search(cpat, names[j], re.I) for i, j, *_ in conf)))
    rows = [dict(set=keys[i][1], kind=keys[i][0][5:].lower(), name=sets[keys[i]][0], cell_type=names[j], cl=cls[j], dev_z=a, exam_z=b, q=q) for i, j, a, b, q in conf]
    facts = [_fact(f"{'COMPLEX' if r['kind'] == 'complex' else 'PATHWAY'}-{r['set']}", f"link/F7/CT-{r['cl']}",
                   {k: r[k] for k in ("cell_type", "dev_z", "exam_z", "q")}, "F7") for r in rows]
    gate = _gate(len(conf), len(nconf), pr, sum(k[2] for k in known), 5)
    return dict(sets=len(keys), cell_types=len(cls), confirmed=len(conf), sets_with_link=len({r["set"] for r in rows}), negative_confirmed=len(nconf),
                planted_recovered=float(pr), known=[list(k) for k in known], known_needed=5, gate=gate, rows=rows), facts


# ------------------------------------------------------------------------------------------------ F8 drug <-> disease
def run_f8():
    import pandas as pd
    from scipy.stats import fisher_exact
    FAM[0] = "F8"; traits = json.load(open(os.path.join(RUN, "f4_traits.json"))); best = pd.read_parquet(os.path.join(RUN, "f5_links.parquet"))
    tg = json.load(open(L("SETS_DRUGTARGET")[1])); dr = json.load(open(L("OT_DRUGS")[1]))
    genes = best.groupby("trait").gene.apply(set).to_dict()
    tt = [i for i, t in enumerate(traits) if t["diseaseIds"] and i in genes]
    drugs = [(c, set(d["members"]), {x[0] for x in (dr.get(c) or {}).get("indications", [])}) for c, d in tg.items()]
    a = b = c_ = d_ = 0; cand = []
    for i in tt:
        ids, gs = set(traits[i]["diseaseIds"]), genes[i]
        for ch, tgt, ind in drugs:
            linked, indicated = bool(tgt & gs), bool(ind & ids)
            if linked and indicated: a += 1
            elif linked: b += 1; cand.append((i, ch, sorted(tgt & gs)))
            elif indicated: c_ += 1
            else: d_ += 1
    OR, p = fisher_exact([[a, b], [c_, d_]], alternative="greater") if (a + b) and (c_ + d_) else (float("nan"), 1.0)
    H = cbb.hgnc()
    rows = [dict(trait=traits[i]["trait"], study=traits[i]["studyId"], drug=ch, drug_name=tg[ch]["name"], genes=[H["e2s"].get(x, x) for x in gl])
            for i, ch, gl in cand]
    facts = [_fact(f"DRUG-{r['drug']}", f"link/F8/TRAIT-{r['study']}", {"genes": r["genes"], "status": "genetic link, not indicated (untested)"}, "F8")
             for r in rows]
    ok = bool(OR > 1 and p < 0.05)
    return dict(traits=len(tt), drugs=len(drugs), table=[[a, b], [c_, d_]], odds_ratio=float(OR), p=float(p),
                gate=dict(passed=ok, verdict="PASS" if ok else "FAIL"), candidates=len(rows), rows=rows), facts


# ------------------------------------------------------------------------------------------------ F9 disease <-> disease
F9_KNOWN = [("EFO_0004611", "EFO_0004574"), ("EFO_0004305", "EFO_0004509"), ("EFO_0004587", "EFO_0004308"), ("EFO_0003767", "EFO_0000384")]


def run_f9():
    FAM[0] = "F9"; Zs = np.load(os.path.join(RUN, "f4_scores.npy")); traits = json.load(open(os.path.join(RUN, "f4_traits.json"))); nt = len(traits)
    USED["F9"] |= USED.get("F4", set())

    def prof(Z):
        Z = Z - Z.mean(0, keepdims=True); Z = Z - Z.mean(1, keepdims=True); return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    def decide(Pd, Pe, rng, nperm=200):
        Cd = Pd @ Pd.T; np.fill_diagonal(Cd, -1); car = set()
        for i in range(len(Pd)):
            for j in np.argsort(-Cd[i])[:20]:
                if Cd[i, j] >= 0.5:
                    car.add((min(i, int(j)), max(i, int(j))))
        car = sorted(car)
        if not car:
            return []
        I, J = np.array(car).T; obs = (Pe[I] * Pe[J]).sum(1); ge = np.zeros(len(car))
        for _ in range(nperm):
            pm = rng.permutation(Pe.shape[1]); ge += (Pe[I][:, pm] * Pe[J]).sum(1) >= obs
        p = (1 + ge) / (1 + nperm); q = bh(p)
        return [(i, j, float(Cd[i, j]), float(o), float(qq)) for (i, j), o, qq in zip(car, obs, q) if qq < GATE["q"] and o > 0]
    rng = np.random.default_rng(91); Pd, Pe = prof(Zs[0, :nt]), prof(Zs[1, :nt]); conf = decide(Pd, Pe, rng)
    Pn = np.stack([r[rng.permutation(len(r))] for r in Pe]); nconf = decide(Pd, Pn, rng)
    src = rng.choice(nt, 20, replace=False); Pp_d = np.r_[Pd, Pd[src] + rng.normal(0, 0.02, Pd[src].shape)]
    Pp_e = np.r_[Pe, Pe[src] + rng.normal(0, 0.02, Pe[src].shape)]; Pp_d, Pp_e = prof(Pp_d), prof(Pp_e)
    pc = decide(Pp_d, Pp_e, rng); pr = np.mean([any({i, j} == {int(s), nt + k} for i, j, *_ in pc) for k, s in enumerate(src)])
    cs = {(i, j) for i, j, *_ in conf}; known = []
    for a, b in F9_KNOWN:
        ia = [i for i, t in enumerate(traits) if a in t["diseaseIds"]]; ib = [i for i, t in enumerate(traits) if b in t["diseaseIds"]]
        known.append((a, b, any((min(x, y), max(x, y)) in cs for x in ia for y in ib)))
    rows = [dict(a=traits[i]["trait"], b=traits[j]["trait"], a_study=traits[i]["studyId"], b_study=traits[j]["studyId"], dev_r=d, exam_r=e, q=q) for i, j, d, e, q in conf]
    facts = []
    for r in rows:
        for x, y in ((r["a_study"], r["b_study"]), (r["b_study"], r["a_study"])):
            facts.append(_fact(f"TRAIT-{x}", f"link/F9/TRAIT-{y}", {k: r[k] for k in ("dev_r", "exam_r", "q")}, "F9"))
    gate = _gate(len(conf), len(nconf), pr, sum(k[2] for k in known), 3)
    return dict(traits=nt, confirmed=len(conf), negative_confirmed=len(nconf), planted_recovered=float(pr), known=[list(k) for k in known],
                known_needed=3, gate=gate, rows=rows), facts


# ------------------------------------------------------------------------------------------------ gate, facts, ledger
def _gate(conf, neg, planted, known, need):
    ok = dict(confirmed=conf >= 1, negative=neg <= max(1, GATE["neg_share"] * conf), planted=planted >= GATE["planted"], known=known >= need)
    return dict(ok, passed=all(ok.values()), verdict="PASS" if all(ok.values()) else "FAIL")


def _fact(entity, attr, value, fam):
    t = cbb.now()
    return Fact(entity, attr, value, t[:10], t, "derived", {"name": f"cellbrain:{VERSION}-{fam}", "code_sha": code_sha()},
                inputs=tuple(sorted(USED[fam])))


RUNNERS = {"F1": run_f1, "F2": run_f2, "F3": run_f3, "F4": run_f4, "F5": run_f5, "F6": run_f6, "F7": run_f7, "F8": run_f8, "F9": run_f9}
CLAIMS = {
    "F1": "Kinds of gene data that say the same thing: confirmed view pairs, with the controls and >= 6 of 8 known pairs.",
    "F2": "Gene pairs linked by independent kinds of data: support s* at null FDR <= 5%, planted >= 90%, held-out truth enrichment >= 10x.",
    "F3": "Cell types matched across datasets and modalities: confirmed on EXAM genes, with the controls and >= 5 of 7 known matches.",
    "F4": "Diseases' genetics point to cell types: confirmed on the other chromosomes, with the controls and >= 6 of 8 known answers.",
    "F5": "Diseases and genes sharing causal variants with an eQTL (CLPP >= 0.1): descriptive, no verdict.",
    "F6": "Knockdowns that move K562 toward or away from a cell type's identity: confirmed on EXAM genes, with the controls and >= 4 of 7 erythroid regulators.",
    "F7": "Complexes and pathways specific to cell types: confirmed on EXAM members, with the controls and >= 5 of 7 known pairs.",
    "F8": "Genetic links (F5 genes) predict a drug's existing indication: Fisher OR > 1, p < 0.05.",
    "F9": "Diseases sharing cell-type patterns: confirmed on the other chromosomes, with the controls and >= 3 of 4 known pairs.",
}


def code_sha():
    h = hashlib.sha256()
    for f in ("cellbrain.py", "cellbrain_bank.py", "trait2cell.py", "e2gmap.py"):
        h.update(open(os.path.join(HERE, f), "rb").read())
    rep = os.path.join(cbb.OUTD, "bank_report.json")
    if os.path.exists(rep):
        h.update(json.dumps({k: v["sha256"] for k, v in json.load(open(rep))["layers"].items()}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def _git(*a):
    import subprocess
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True)


def declare():
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        for fam in FAMILIES:
            cid = f"{VERSION}-{fam}"
            if cid in have:
                print(f"{cid}: already declared"); continue
            rec = dict(op="declare", id=cid, at=cbb.now(), claim=CLAIMS[fam], gate=GATE, code_sha=code_sha(), commit=_git("rev-parse", "HEAD").stdout.strip(),
                       brain_commit=cbb.brain_commit(), trial=TRIAL)
            f.write(json.dumps(rec, sort_keys=True) + "\n"); print(f"{cid}: declared")


def test(fams):
    if not TRIAL:
        head = _git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for fam in fams:
            if f"{VERSION}-{fam}" not in dec:
                raise SystemExit(f"{VERSION}-{fam}: not declared in HEAD's ledger; declare, commit, then test")
            if dec[f"{VERSION}-{fam}"]["code_sha"] != code_sha():
                raise SystemExit(f"{VERSION}-{fam}: code or bank changed since it was declared")
    done = {r["id"] for r in _read() if r["op"] == "result"}
    os.makedirs(RUN, exist_ok=True); os.makedirs(OUTD, exist_ok=True)
    bank = cbb.bank() if not TRIAL else None
    for fam in fams:
        cid = f"{VERSION}-{fam}"
        if cid in done and not TRIAL:
            print(f"{cid}: already tested; skipped"); continue
        t0 = time.time(); FAM[0] = fam; print(f"{cid} ...", flush=True)
        res, facts = RUNNERS[fam]()
        res["seconds"] = round(time.time() - t0); rows = res.pop("rows", [])
        json.dump(rows, open(os.path.join(RUN, f"{fam}_connections.json"), "w"), default=float)
        pub = [r for r in rows if not r.get("local")]
        json.dump(pub[:50000], open(os.path.join(OUTD, f"{fam}_connections.json"), "w"), default=float, indent=0)
        n = bank.add_many(facts) if bank is not None else 0
        rec = dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), facts_written=n, inputs=sorted(USED[fam]), **res)
        with open(LEDGER, "a") as f:
            f.write(json.dumps(rec, sort_keys=True, default=float) + "\n")
        print(f"{cid}: {res.get('gate', {}).get('verdict')} -- {json.dumps({k: v for k, v in res.items() if k not in ('matrix',)}, default=float)[:600]}", flush=True)
    report()


def report():
    rs = {r["id"]: r for r in _read() if r["op"] == "result"}
    L_ = ["=" * 97, "CELLBRAIN -- the brain connected to the memory bank, one task: make as many connections as it can", "=" * 97]
    for fam in FAMILIES:
        r = rs.get(f"{VERSION}-{fam}")
        if not r:
            continue
        g = r.get("gate", {})
        L_.append(f"  {fam} {CLAIMS[fam][:80]}")
        L_.append(f"     confirmed {r.get('confirmed', r.get('links', r.get('candidates')))}; negative control {r.get('negative_confirmed', '-')}; "
                  f"planted {r.get('planted_recovered', '-')}; known {sum(k[-1] for k in r.get('known', [])) if r.get('known') else '-'}"
                  f"/{len(r.get('known', [])) or '-'} -> {g.get('verdict')}")
    open(RES, "w").write("\n".join(L_) + "\n"); print("\n".join(L_))


def card(ent):
    b = cbb.bank()
    for f in b.facts(ent):
        v = json.dumps(f.value, default=float)
        print(f"  {f.attr:40s} {v[:140]}{'...' if len(v) > 140 else ''}  [{f.tier}; {f.source.get('name', '')[:40]}]")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--trial"]
    if args[0] == "declare":
        declare()
    elif args[0] == "test":
        test(args[1:] or FAMILIES)
    elif args[0] == "card":
        card(args[1])
    elif args[0] == "report":
        report()
