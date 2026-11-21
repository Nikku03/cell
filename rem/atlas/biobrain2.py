"""BIOBRAIN B2 -- use 5 of the AlphaGenome 2 data: new claims about how the cell works, tested with the brain's neuro
rules (Nikku03/money brain/neuro, docs/NEURO.md; the same rules as biobrain.py B1) on the new memory-bank layers.

THE RULES (as B1): a DEV subject for looking (thresholds set there only); ten fresh subjects nothing looked at; per
subject the measure must reach the threshold AND beat a null that keeps everything but the structure (p < 0.05);
PASS: >= 9 of 10 subjects qualify, >= 90% of those pass, the planted (positive) control passes >= 90% of 10 runs and
the unstructured (negative) control fails >= 90% of 10 runs, BH q < 0.10 over every claim in rem/atlas/bio_ledger.jsonl
(a claim's p is the chance of that many passing subjects at one half each). Declare -> commit -> test.
Threshold rule (fixed before looking): chance + 0.6 x (dev value - chance), rounded to 2 decimals -- the ratio B1
used (alike 0.75 for dev 0.913; rebuild 0.40 for dev 0.645).

THE CLAIMS
  coexpressed    Two subunits of the same protein complex are co-expressed across a tissue's cell types: AUROC of
                 the Pearson correlation of their expression profiles (CELLxGENE mean expression per cell type,
                 me x share expressing), same-complex pairs vs other pairs of complex genes.
                 Subjects: CELLxGENE human tissues; DEV = blood; the 10 fresh = the tissues with the most cell types
                 (labelled, >= 50 cells, not the root 'cell') apart from blood. Genes: Complex Portal members
                 (Ensembl) expressed in >= 5% of the cells of >= 1 cell type there. Null: complex membership
                 permuted among genes of the same mean-expression decile (keeps 'highly expressed genes correlate').
                 Qualifies: >= 100 genes with a complex mate and >= 15 cell types.
  module_alike   Genes that are co-expressed across the body's cell types respond alike to knockdown even when they
                 share no complex. Modules: the screen genes' CELLxGENE profiles across all tissue x cell type pairs
                 (expressed in >= 10% of cells of >= 10 pairs), z-scored, 50 principal components, spherical k-means,
                 K = 100, seed 0 -- built from expression alone, no knockdown data; written to the bank as facts and
                 read back from there. Measure: AUROC of the specific-response cosine (B1's analysis set), same-module
                 pairs vs other pairs, pairs that share a complex removed. Subjects: B1's ten screens; DEV = K562-A
                 TRAIN genes. Null: module labels permuted among genes of the same record-strength decile (B1's null).
  promoter       A promoter that is open in a cell type, though not everywhere, goes with its gene being expressed
                 there more than elsewhere. Per ENCODE biosample with RNA-seq and a DNase-based cCRE annotation (217):
                 promoter open = an element of class PLS..CA-only (activity >= 0.5) within 1 kb of the GENCODE 46
                 TSS; variable promoters = open in 10-60% of the 217; expression specificity = the gene's mean log1p
                 TPM there, z-scored across the 217. Measure: AUROC of specificity, open-here vs closed-here genes
                 (variable promoters, expressed >= 1 TPM somewhere). DEV = K562; the 10 fresh = the biosamples with
                 the most RNA-seq experiments apart from K562. Null: this biosample's RNA paired with another
                 biosample's promoter states (every other biosample, 216 draws): keeps both kinds of structure,
                 removes only the match. Qualifies: >= 100 open and >= 100 closed variable promoters.
CONTROLS on the DEV subject (10 runs each): positive = the mechanism planted in synthetic data (one random programme
per complex / module; expression specificity = signal x open + noise); negative = no mechanism (Gaussian profiles
with the DEV covariance; B1's unstructured responses; the DEV specificities shuffled across genes).
NOT: one complex resource; CELLxGENE aggregates are rolled-up means over many studies (batch and study mix);
modules from one clustering choice (K fixed a priori, not tuned); the 10 promoter subjects include several T-cell
states (not independent); open chromatin is a class, not a signal level.

Run: python3 -P rem/atlas/biobrain2.py look | declare | test      (declare, then commit, then test)
"""

from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
sys.path.insert(0, HERE)


def _imp(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


bb = _imp("biobrain")
e2 = _imp("e2gmap")
AG = e2.AG
CXF = os.path.join(AG, "cellxgene", "cellxgene_human.npz")
CXP = os.path.join(AG, "cellxgene", "cellxgene_human_pairs.json")
LEDGER = bb.LEDGER
FACTS = os.path.join(HERE, "_cache", "biobrain", "modules_facts.jsonl")
OUT = os.path.join(HERE, "RESULTS_biobrain2.txt")
VERSION = "B2"
N_NULL, N_CTRL, QUALIFY, MIN_CT = 200, 10, 100, 15
GATE = bb.GATE
MARGIN = 0.6
DEV_TISSUE, DEV_BIOSAMPLE = "blood", "K562"
K_MODULES = 100
SIGNAL = {"coexpressed": 3.0, "module_alike": 3.0, "promoter": 1.0}   # planted mechanism size vs unit noise
# Look (DEV only, 2026-11-18): coexpressed blood AUROC 0.726 (p 0.005; planted 0.986, unstructured 0.500); module_alike
# K562-A TRAIN 0.518 (p 0.005; 1,123 genes in a module; planted 1.000, unstructured 0.501); promoter K562 0.645 (p 0.005;
# planted 0.933, shuffled 0.507). Thresholds by the rule chance + 0.6 x (dev - chance): 0.64, 0.51, 0.59.
CLAIMS = [
    {"id": "coexpressed", "chance": 0.5, "threshold": 0.64, "dev": 0.726, "expect": "pass",
     "claim": "Two subunits of the same protein complex are co-expressed across a tissue's cell types more than two "
              "genes of different complexes: AUROC of the expression-profile correlation >= threshold, beating the "
              "expression-level-matched membership null."},
    {"id": "module_alike", "chance": 0.5, "threshold": 0.51, "dev": 0.518, "expect": "uncertain (small dev effect)",
     "claim": "Two genes of the same CELLxGENE co-expression module (built from expression alone) that share no "
              "complex respond to knockdown more alike than genes of different modules: AUROC of the specific-response "
              "cosine >= threshold, beating the strength-matched null."},
    {"id": "promoter", "chance": 0.5, "threshold": 0.59, "dev": 0.645, "expect": "pass",
     "claim": "Genes whose (variable) promoter is open in a biosample are expressed there more than elsewhere: AUROC of "
              "the expression specificity, open vs closed promoters, >= threshold, beating the mismatched-biosample null."},
]


# ------------------------------------------------------------------------------------------------ measures
def _ranks(sc):
    rk = np.empty(len(sc)); rk[sc.argsort(kind="stable")] = np.arange(1, len(sc) + 1); return rk


def m_sets(Xs, st, M, rng, null=True):
    """AUROC of the cosine, pairs sharing a set vs other pairs, among genes in >= 1 set (sparse labels; null:
    set membership permuted within level deciles)."""
    import scipy.sparse as sp
    idx = np.nonzero(M.any(1))[0]; n = len(idx)
    iu = np.triu_indices(n, 1); sc = (Xs[idx] @ Xs[idx].T)[iu]
    Rk = np.zeros((n, n)); Rk[iu] = _ranks(sc); npairs = len(sc); del sc
    Ms = sp.csr_matrix(M[idx].astype(np.float32))

    def auc(Mx):
        C = (Mx @ Mx.T).tocoo(); s = C.row < C.col; i, j = C.row[s], C.col[s]; n1 = len(i); n0 = npairs - n1
        return float((Rk[i, j].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), n1
    obs, n1 = auc(Ms); out = dict(stat=obs, genes=int(n), pos_pairs=int(n1))
    if null:
        sub = st[idx]; ge = sum(auc(Ms[bb.deciles(sub, rng)])[0] >= obs for _ in range(N_NULL))
        out["p"] = (1 + ge) / (1 + N_NULL)
    return out


def m_partition(Xs, st, lab, Mc, rng, null=True):
    """AUROC of the cosine, same-label pairs vs other pairs among labelled genes; pairs sharing a complex (Mc)
    removed; null: labels permuted within strength deciles (the complex exclusion stays with the true genes)."""
    idx = np.nonzero(lab >= 0)[0]; n = len(idx)
    iu = np.triu_indices(n, 1); sc = (Xs[idx] @ Xs[idx].T)[iu]
    E = Mc[idx].astype(np.float32); keep = ~((E @ E.T)[iu] > 0)
    i, j = iu[0][keep], iu[1][keep]; rk = _ranks(sc[keep]); del sc, iu
    m = lab[idx]

    def auc(mm):
        y = mm[i] == mm[j]; n1 = int(y.sum()); n0 = len(y) - n1
        return float((rk[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), n1
    obs, n1 = auc(m); out = dict(stat=obs, genes=int(n), pos_pairs=int(n1))
    if null:
        sub = st[idx]; ge = sum(auc(m[bb.deciles(sub, rng)])[0] >= obs for _ in range(N_NULL))
        out["p"] = (1 + ge) / (1 + N_NULL)
    return out


def m_promoter(z, O, t, rng=None, null=True):
    """AUROC of specificity z[t] for open vs closed (O[t]); null: O of every other biosample."""
    obs = bb._auroc(z[t], O[t]); out = dict(stat=obs, genes=int(len(z[t])), open=int(O[t].sum()))
    if null:
        others = [u for u in range(len(O)) if u != t]
        ge = sum(bb._auroc(z[t], O[u]) >= obs for u in others); out["p"] = (1 + ge) / (1 + len(others))
    return out


# ------------------------------------------------------------------------------------------------ data
def complexes_ens():
    import pandas as pd
    cm = pd.read_parquet(R("outputs", "orphan", "science_data", "complex_members.parquet"))
    comp = {}
    for c, e in zip(cm.complex_id, cm.ensembl_gene_id):
        if isinstance(e, str) and e.startswith("ENSG"):
            comp.setdefault(c, set()).add(e)
    return {c: m for c, m in comp.items() if len(m) >= 2}


class CX:
    def __init__(self):
        z = np.load(CXF); self.indptr, self.col, self.me, self.pc = z["indptr"], z["col"], z["me"], z["pc"]
        self.genes, self.symbols = z["genes"], z["symbols"]; self.pairs = json.load(open(CXP))["pairs"]
        self.row = np.repeat(np.arange(len(self.genes), dtype=np.int32), np.diff(self.indptr))

    def eligible(self, tissue=None):
        return [i for i, p in enumerate(self.pairs) if p["labelled"] and p["cell_type"] != "CL:0000000" and p["cells"] >= 50
                and (tissue is None or p["tissue_name"] == tissue)]

    def tissues(self):
        import collections
        return collections.Counter(self.pairs[i]["tissue_name"] for i in self.eligible())

    def dense(self, cols, rows):
        """mean expression (me x share) and share expressing, rows (gene indices) x cols (pair indices)."""
        lc = np.full(len(self.pairs), -1, np.int64); lc[cols] = np.arange(len(cols))
        lr = np.full(len(self.genes), -1, np.int64); lr[rows] = np.arange(len(rows))
        sel = (lc[self.col] >= 0) & (lr[self.row] >= 0)
        X = np.zeros((len(rows), len(cols)), np.float32); P = np.zeros_like(X)
        r_, c_ = lr[self.row[sel]], lc[self.col[sel]]; pc = self.pc[sel] / 255.0
        X[r_, c_] = self.me[sel] * 0.04 * pc; P[r_, c_] = pc
        return X, P


def profiles(X):
    Xc = X - X.mean(1, keepdims=True); return Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)


def tissue_subject(cx, tissue, comp):
    cols = cx.eligible(tissue); members = set().union(*comp.values())
    rows = np.array([i for i, g in enumerate(cx.genes) if g in members])
    X, P = cx.dense(cols, rows); keep = (P.max(1) >= 0.05) & (X.std(1) > 0)
    G = [str(cx.genes[i]) for i in rows[keep]]; Xk = X[keep].astype(np.float64)
    M, _ = bb.incidence(G, comp)
    return dict(G=G, Xs=profiles(Xk), st=Xk.mean(1), raw=Xk, M=M, d=len(cols),
                qualifies=bool(int(M.any(1).sum()) >= QUALIFY and len(cols) >= MIN_CT))


def build_modules(cx, bank):
    """Spherical k-means on the screen genes' CELLxGENE profiles (expression only); cached as bank facts."""
    if os.path.exists(FACTS):
        return {f["members"][k]: int(f["attribute"].split("/")[1]) for f in map(json.loads, open(FACTS)) for k in range(len(f["members"]))}
    known = set(bank.by_gene); rows, syms = [], []
    seen = set()
    for i, s in enumerate(cx.symbols):
        s = str(s)
        if s in known and s not in seen:
            seen.add(s); rows.append(i); syms.append(s)
    cols = cx.eligible(); X, P = cx.dense(cols, np.array(rows))
    keep = ((P >= 0.10).sum(1) >= 10) & (X.std(1) > 0); X = X[keep].astype(np.float64); syms = [s for s, k in zip(syms, keep) if k]
    Z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    U, S, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False); Y = U[:, :50] * S[:50]
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
    rng = np.random.default_rng(0); C = [Y[rng.integers(len(Y))]]
    for _ in range(K_MODULES - 1):                                   # k-means++ (cosine distance)
        d = 1 - np.max(Y @ np.array(C).T, axis=1); d = np.maximum(d, 0) ** 2
        C.append(Y[rng.choice(len(Y), p=d / d.sum())])
    C = np.array(C)
    for _ in range(50):
        a = np.argmax(Y @ C.T, axis=1)
        for k in range(K_MODULES):
            if (a == k).any():
                c = Y[a == k].sum(0); C[k] = c / (np.linalg.norm(c) + 1e-12)
    a = np.argmax(Y @ C.T, axis=1)
    os.makedirs(os.path.dirname(FACTS), exist_ok=True)
    with open(FACTS, "w") as f:
        for k in range(K_MODULES):
            mem = sorted(s for s, x in zip(syms, a) if x == k)
            if mem:
                rec = dict(entity="cellxgene/human/all-tissues", attribute=f"module/{k}", members=mem, k=K_MODULES,
                           source="CELLxGENE WMG aggregates (ag2bank); expression only")
                rec["id"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:16]
                f.write(json.dumps(rec) + "\n")
    return build_modules(cx, bank)


def module_labels(G, mod):
    return np.array([mod.get(g, -1) for g in G])


def promoter_data():
    """z (biosamples x genes, expression specificity), O (open promoters), terms; variable promoters only."""
    reg = e2.Registry()
    rna = np.load(os.path.join(AG, "bulk", "encode_rna_human.npz")); m = json.load(open(os.path.join(AG, "bulk", "encode_rna_human_meta.json")))["meta"]
    cols = rna["cols"].tolist(); gcol = {g: i for i, g in enumerate(cols)}
    terms = sorted(t for t in set(m["biosample"]) & {b["biosample"] for b in reg.meta} if reg.has_dnase(t))
    nexp = {t: m["biosample"].count(t) for t in terms}
    X = rna["X"]; E = np.stack([X[[i for i, b in enumerate(m["biosample"]) if b == t]].astype(np.float32).mean(0) for t in terms])
    tss = [g for g in json.load(open(os.path.join(AG, "raw", "gencode46_tss.json"))) if g[0] in gcol]
    win = []
    for gid, name, gt, ch, pos in tss:
        win.append(reg.overlap(ch, pos - 1000, pos + 1000))
    O = np.zeros((len(terms), len(tss)), bool)
    for k, t in enumerate(terms):
        a = reg.activity(t) >= 0.5
        O[k] = [bool(a[w].any()) if len(w) else False for w in win]
    Eg = E[:, [gcol[g[0]] for g in tss]]
    share = O.mean(0); var = (share >= 0.10) & (share <= 0.60) & (Eg.max(0) >= np.log1p(1.0))
    Ev = Eg[:, var]; z = (Ev - Ev.mean(0)) / (Ev.std(0) + 1e-9)
    return dict(z=z, O=O[:, var], terms=terms, nexp=nexp, genes=[tss[i][0] for i in np.nonzero(var)[0]])


def promoter_subjects(pd_):
    order = sorted((t for t in pd_["terms"] if t != DEV_BIOSAMPLE), key=lambda t: (-pd_["nexp"][t], t))
    return order[:10]


def coexp_subjects(cx):
    c = cx.tissues(); return [t for t, _ in sorted(((t, n) for t, n in c.items() if t != DEV_TISSUE), key=lambda x: (-x[1], x[0]))][:10]


# ------------------------------------------------------------------------------------------------ controls
def ctrl_coexpressed(dev, kind, seed, rng):
    M = dev["M"]; idx = np.nonzero(M.any(1))[0]; M = M[idx]; d = dev["d"]
    r = np.random.default_rng(4000 + seed)
    if kind == "positive":
        Pc = r.normal(size=(M.shape[1], d)); base = M.astype(np.float64) @ Pc
        base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-12
        X = SIGNAL["coexpressed"] * base + r.normal(size=(len(idx), d)) / np.sqrt(d)
    else:
        raw = dev["raw"][idx]; X = r.multivariate_normal(raw.mean(0), np.cov(raw.T), size=len(idx))
    st = r.lognormal(0, 1, size=len(idx))
    o = m_sets(profiles(X), st, M, rng); return o


def ctrl_module(devm, kind, seed, rng, mod, comp):
    Gd = devm["G_all"]
    if kind == "positive":
        lab = module_labels(Gd, mod); M = np.zeros((len(Gd), K_MODULES), bool); ok = lab >= 0; M[np.nonzero(ok)[0], lab[ok]] = True
        Z = bb.planted(Gd, M, seed, signal=SIGNAL["module_alike"])
    else:
        Z = bb.unstructured(devm["Zraw"], len(Gd), seed)
    G, Xs, st, _ = bb.analysis(Gd, Z); Mc, _ = bb.incidence(G, comp)
    return m_partition(Xs, st, module_labels(G, mod), Mc, rng)


def ctrl_promoter(pd_, t, kind, seed, rng):
    r = np.random.default_rng(5000 + seed); z = pd_["z"].copy()
    if kind == "positive":
        o = pd_["O"][t].astype(float); z[t] = SIGNAL["promoter"] * (o - o.mean()) / (o.std() + 1e-9) + r.normal(size=len(o))
    else:
        z[t] = r.permutation(z[t])
    return m_promoter(z, pd_["O"], t, rng)


# ------------------------------------------------------------------------------------------------ look / declare / test
def _passes(c, r):
    return bool(r["stat"] >= c["threshold"] and r["p"] < GATE["null_p_lt"])


def look():
    t0 = time.time(); rng = np.random.default_rng(0)
    cx = CX(); comp_e = complexes_ens()
    dv = tissue_subject(cx, DEV_TISSUE, comp_e); r = m_sets(dv["Xs"], dv["st"], dv["M"], rng)
    print(f"coexpressed DEV {DEV_TISSUE}: {dv['d']} cell types, {r['genes']} complex genes, AUROC {r['stat']:.3f} p {r['p']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    for k in ("positive", "negative"):
        o = ctrl_coexpressed(dv, k, 0, rng); print(f"   control {k}: {o['stat']:.3f} p {o['p']:.3f}", flush=True)
    print("   subjects (counts only):", {t: n for t, n in cx.tissues().items() if t in coexp_subjects(cx)}, flush=True)
    bank, comp, _ = bb.load(); mod = build_modules(cx, bank)
    print(f"modules: {len(set(mod.values()))} modules over {len(mod)} screen genes (expression only)", flush=True)
    genes, Z = bb.subject(bank, "A", "TRAIN"); G, Xs, st, _ = bb.analysis(genes, Z); Mc, _ = bb.incidence(G, comp)
    r = m_partition(Xs, st, module_labels(G, mod), Mc, rng)
    print(f"module_alike DEV K562-A TRAIN: {r['genes']} genes in a module, AUROC {r['stat']:.3f} p {r['p']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    devm = dict(G_all=genes, Zraw=Z)
    for k in ("positive", "negative"):
        o = ctrl_module(devm, k, 0, rng, mod, comp); print(f"   control {k}: {o['stat']:.3f} p {o['p']:.3f}", flush=True)
    pd_ = promoter_data(); t = pd_["terms"].index(DEV_BIOSAMPLE); r = m_promoter(pd_["z"], pd_["O"], t)
    print(f"promoter DEV {DEV_BIOSAMPLE}: {len(pd_['terms'])} biosamples, {r['genes']} variable promoters ({r['open']} open), AUROC {r['stat']:.3f} p {r['p']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    for k in ("positive", "negative"):
        o = ctrl_promoter(pd_, t, k, 0, rng); print(f"   control {k}: {o['stat']:.3f} p {o['p']:.3f}", flush=True)
    print("   subjects:", promoter_subjects(pd_), flush=True)


def code_sha():
    h = hashlib.sha256()
    for f in ("biobrain2.py", "biobrain.py", "e2gmap.py"):
        h.update(open(os.path.join(HERE, f), "rb").read())
    return h.hexdigest()[:16]


def spec(c):
    return dict(c, version=VERSION, gate=GATE, n_null=N_NULL, n_ctrl=N_CTRL, qualify=QUALIFY, margin=MARGIN, signal=SIGNAL,
                k_modules=K_MODULES, code_sha=code_sha())


def declare():
    if any(c["threshold"] is None for c in CLAIMS):
        raise SystemExit("thresholds not set from look")
    have = {r["id"] for r in bb._read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for c in CLAIMS:
            cid = f"{VERSION}-{c['id']}"
            if cid in have:
                print(f"{cid}: already declared"); continue
            rec = dict(op="declare", id=cid, at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), spec=spec(c),
                       commit=bb._git("rev-parse", "HEAD").stdout.strip())
            f.write(json.dumps(rec, sort_keys=True) + "\n"); print(f"{cid}: declared")


def test():
    head = bb._git("show", "HEAD:rem/atlas/bio_ledger.jsonl").stdout
    committed = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
    for c in CLAIMS:
        cid = f"{VERSION}-{c['id']}"
        if cid not in committed:
            raise SystemExit(f"{cid}: its declaration is not in HEAD's ledger; commit it first")
        if committed[cid]["spec"]["code_sha"] != code_sha():
            raise SystemExit(f"{cid}: the code changed since it was declared")
    if any(r["op"] == "result" and r["id"].startswith(VERSION + "-") for r in bb._read()):
        raise SystemExit(f"{VERSION} already tested; a new test is a new VERSION")
    from scipy.stats import binomtest
    t0 = time.time(); rng = np.random.default_rng(7); out = []
    cx = CX(); comp_e = complexes_ens(); bank, comp, _ = bb.load(); mod = build_modules(cx, bank)
    pd_ = promoter_data()
    for c in CLAIMS:
        rows, ctrl = [], {"positive": [], "negative": []}
        if c["id"] == "coexpressed":
            dv = tissue_subject(cx, DEV_TISSUE, comp_e)
            for t in coexp_subjects(cx):
                s = tissue_subject(cx, t, comp_e)
                rows.append(dict(m_sets(s["Xs"], s["st"], s["M"], rng), subject=t, cell_types=s["d"]) if s["qualifies"] else dict(subject=t, qualifies=False))
            for k in ctrl:
                ctrl[k] = [_passes(c, ctrl_coexpressed(dv, k, s_, rng)) for s_ in range(N_CTRL)]
        elif c["id"] == "module_alike":
            for nm, scr, sp in bb.SUBJECTS:
                genes, Z = bb.subject(bank, scr, sp); G, Xs, st, _ = bb.analysis(genes, Z); Mc, _ = bb.incidence(G, comp)
                lab = module_labels(G, mod)
                rows.append(dict(m_partition(Xs, st, lab, Mc, rng), subject=nm) if (lab >= 0).sum() >= QUALIFY else dict(subject=nm, qualifies=False))
            genes, Z = bb.subject(bank, "A", "TRAIN"); devm = dict(G_all=genes, Zraw=Z)
            for k in ctrl:
                ctrl[k] = [_passes(c, ctrl_module(devm, k, s_, rng, mod, comp)) for s_ in range(N_CTRL)]
        else:
            for t in promoter_subjects(pd_):
                i = pd_["terms"].index(t); o = pd_["O"][i]
                ok = o.sum() >= QUALIFY and (~o).sum() >= QUALIFY
                rows.append(dict(m_promoter(pd_["z"], pd_["O"], i), subject=t) if ok else dict(subject=t, qualifies=False))
            td = pd_["terms"].index(DEV_BIOSAMPLE)
            for k in ctrl:
                ctrl[k] = [_passes(c, ctrl_promoter(pd_, td, k, s_, rng)) for s_ in range(N_CTRL)]
        for r in rows:
            if "stat" in r:
                r["qualifies"] = True; r["pass"] = _passes(c, r)
        q = [r for r in rows if r.get("qualifies")]; k = sum(r["pass"] for r in q)
        p = binomtest(k, len(q), 0.5, alternative="greater").pvalue if q else 1.0
        ok_ctrl = np.mean(ctrl["positive"]) >= GATE["ctrl_share"] and np.mean([not x for x in ctrl["negative"]]) >= GATE["ctrl_share"]
        gate = len(q) >= GATE["min_qualify"] and k >= GATE["pass_share"] * len(q) and ok_ctrl
        out.append(dict(op="result", id=f"{VERSION}-{c['id']}", subjects=rows, controls=ctrl, passing=k, qualifying=len(q), p=float(p),
                        controls_ok=bool(ok_ctrl), gate=bool(gate), expect=c["expect"], at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        code_sha=code_sha()))
        print(f"  {c['id']}: {k}/{len(q)} pass; controls +{sum(ctrl['positive'])}/{N_CTRL} -{N_CTRL - sum(ctrl['negative'])}/{N_CTRL} ({time.time() - t0:.0f}s)", flush=True)
    prior = [r for r in bb._read() if r["op"] == "result"]               # BH over every claim in the ledger
    ps = np.array([r["p"] for r in prior] + [o["p"] for o in out]); m = len(ps); order = ps.argsort(); qv = np.empty(m); run = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        run = min(run, ps[i] * m / rank); qv[i] = run
    with open(LEDGER, "a") as f:
        for o, q_ in zip(out, qv[len(prior):]):
            o["q"] = float(q_); o["bh_over"] = m
            o["verdict"] = ("PASS" if o["gate"] and q_ < GATE["bh_q"] else
                            "NO VERDICT" if (o["qualifying"] < GATE["min_qualify"] or not o["controls_ok"]) else "FAIL")
            f.write(json.dumps(o, sort_keys=True) + "\n")
    L = ["=" * 97, "BIOBRAIN B2 -- use 5: new claims on the AlphaGenome 2 data, the brain's neuro rules (declared before they ran)", "=" * 97,
         f"  {'claim':13s} {'threshold':>9s} {'dev':>6s} {'pass/qualify':>13s} {'controls +/-':>13s} {'p':>8s} {'q':>8s}  verdict (expected)"]
    for o, c in zip(out, CLAIMS):
        L.append(f"  {c['id']:13s} {c['threshold']:9.2f} {c['dev']:6.3f} {o['passing']:>6d}/{o['qualifying']:<6d} "
                 f"{sum(o['controls']['positive']):>5d}/{N_CTRL - sum(o['controls']['negative'])}{'':<6s} {o['p']:8.2g} {o['q']:8.2g}  {o['verdict']} ({o['expect']})")
    L.append("  per subject (measure, null p):")
    for o, c in zip(out, CLAIMS):
        L.append(f"    {c['id']:13s} " + "; ".join(f"{r['subject']} {r['stat']:.3f} (p {r['p']:.3f})" if r.get("qualifies") else f"{r['subject']} (does not qualify)"
                                          for r in o["subjects"]))
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    {"look": look, "declare": declare, "test": test}[sys.argv[1]]()
