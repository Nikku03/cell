"""GENEREG -- active genes per cell type, and which gene influences which (activators, repressors, the concepts they run).

Asked by the user: analyse the active genes and the CELLxGENE single-cell data, separate them by cell type, and connect
which gene influences which, so the activators and the other parts and concepts of genes are known.
Data, all through the cellbrain bank (sha-checked layers): CELLxGENE human aggregates (tissue x cell type: mean
expression among expressing cells, share of cells expressing); K562 genome-wide Perturb-seq (11,258 knockdowns x 8,248
genes, 585 non-targeting controls) and the independent K562 essential-gene screen (2,285 knockdowns, 109 controls)
(Replogle 2022, CC BY 4.0); HPA transcription factors (1,485; protein class); Reactome; GENCODE 46 TSS.

PART A, ACTIVE GENES PER CELL TYPE (descriptive). CELLxGENE pairs (labelled, >= 50 cells, not the root 'cell') pooled per
cell type (CL id) by cells: share of cells expressing (pc) and mean expression. Active: pc >= 0.10. Specific: active and
specificity z >= 3 (z of log1p(10 x mean) across cell types). Housekeeping: active in >= 80% of cell types.

CLAIMS (declared before running; rem/atlas/cellbrain_ledger.jsonl, ids G1-*; the brain's rules):
 G1-R1  WHO SWITCHES WHOM, K562 (causal). Effect z of a target = its change / the sd of that gene across the screen's
        non-targeting controls. Regulators: knockdowns with a phenotype (energy test p < 0.05). FOUND in the genome-wide
        screen: |z| >= 4, top 300 per regulator, the regulator itself and genes whose TSS is within 50 kb of its TSS
        excluded (CRISPRi can silence neighbours). CONFIRMED in the essential screen (independent experiment, same
        regulator and target): same sign, BH q < 0.05 over everything testable. Knockdown lowers the target ->
        ACTIVATOR; raises it -> REPRESSOR. Negative: essential-screen target labels shuffled (<= 5% of the real count).
        Positive: 200 planted pairs (z -6 in both screens), >= 90% confirmed. Known activations (>= 4 of 6; found, and
        confirmed where the regulator is in both screens): GATA1->ALAS2, GATA1->HBG2, SREBF2->HMGCS1, SREBF2->LDLR,
        NFE2L1->PSMB1, MYC->NPM1. Descriptive: on-target check (knockdown lowers the regulator's own mRNA).
 G1-R2  DO K562 ACTIVATIONS SHOW ACROSS THE BODY'S CELL TYPES? Correlation across CELLxGENE cell types (log profiles) of
        regulator and target, confirmed activator edges vs random pairs with the same regulators and targets matched
        on expression level: AUROC >= 0.55 and permutation p < 0.05 (200 draws).
 G1-R3  A SWITCH MAP FOR ALL CELL TYPES. Transcription factor -> gene co-expression across cell types, tissues split
        in two halves (sha256): FOUND r >= 0.5 in one half's cell-type profiles (top 100 per TF), CONFIRMED r >= 0.4
        in the other half's. CLAIM: these co-expression links predict causal activation in K562 (knockdown lowers the
        gene, z <= -4, cis excluded) better than other TF-gene pairs testable there: Fisher OR >= 1.5, p < 0.05.
        Negative: TF labels shuffled (OR within 0.67-1.5).
DESCRIPTIVE: concepts per regulator (Reactome enrichment of its confirmed targets, hypergeometric, BH within
regulator); per cell type: active and specific genes, active regulators (TFs active and specific), the confirmed and
co-expression links active there (both genes on). All written to the bank as layers and facts.
NOT: K562 is one cancer cell line; co-expression across cell types is not cause; CELLxGENE single-cell data misses
weakly expressed genes; transcription factors not measured in K562 (most of them) can only be regulators, not targets.

Run: python3 -P rem/atlas/genereg.py declare | test [--trial]
"""

from __future__ import annotations
import collections
import hashlib
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
sys.path.insert(0, HERE)
_s = importlib.util.spec_from_file_location("cellbrain_bank", os.path.join(HERE, "cellbrain_bank.py"))
cbb = importlib.util.module_from_spec(_s); _s.loader.exec_module(cbb)
from brain.bank import Fact  # noqa: E402

TRIAL = "--trial" in sys.argv
LEDGER = os.path.join(cbb.HOME, "trial_ledger.jsonl") if TRIAL else os.path.join(HERE, "cellbrain_ledger.jsonl")
RUN = os.path.join(cbb.HOME, "genereg_trial" if TRIAL else "genereg")
OUTD = os.path.join(RUN, "out") if TRIAL else R("outputs", "genereg")
RES = os.path.join(RUN, "RESULTS.txt") if TRIAL else os.path.join(HERE, "RESULTS_genereg.txt")
CLAIMS = ["G1-R1", "G1-R2", "G1-R3"]
KNOWN = [("GATA1", "ALAS2"), ("GATA1", "HBG2"), ("SREBF2", "HMGCS1"), ("SREBF2", "LDLR"), ("NFE2L1", "PSMB1"), ("MYC", "NPM1")]
USED = set()


def L(name):
    f, p = cbb.layer(name); USED.add(f.ref); return p


def npz(name):
    return np.load(L(name), allow_pickle=False)


def bh(p):
    p = np.asarray(p, float); m = len(p)
    if not m:
        return p
    o = np.argsort(p); q = np.empty(m); run = 1.0
    for r in range(m - 1, -1, -1):
        run = min(run, p[o[r]] * m / (r + 1)); q[o[r]] = run
    return q


def code_sha():
    h = hashlib.sha256()
    for f in ("genereg.py", "cellbrain_bank.py"):
        h.update(open(os.path.join(HERE, f), "rb").read())
    h.update(json.dumps({k: v["sha256"] for k, v in json.load(open(os.path.join(cbb.OUTD, "bank_report.json")))["layers"].items()}, sort_keys=True).encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ part A
def cx_celltypes(tissue_half=None):
    """genes x cell types (CL): share of cells expressing and mean expression, pooled over tissues by cells."""
    z = npz("CX_HUMAN"); pairs = json.load(open(L("CX_HUMAN_PAIRS")))["pairs"]
    th = lambda t: int(hashlib.sha256(("cellbrain-c1:" + t).encode()).hexdigest(), 16) % 2
    cols = [i for i, p in enumerate(pairs) if p["labelled"] and p["cell_type"] != "CL:0000000" and p["cells"] >= 50
            and (tissue_half is None or th(p["tissue_name"]) == tissue_half)]
    cls = sorted({pairs[i]["cell_type"] for i in cols}); cp = {c: j for j, c in enumerate(cls)}
    W = np.zeros(len(pairs)); co = np.full(len(pairs), -1)
    for i in cols:
        W[i] = pairs[i]["cells"]; co[i] = cp[pairs[i]["cell_type"]]
    row = np.repeat(np.arange(len(z["genes"])), np.diff(z["indptr"])); col = z["col"]; sel = co[col] >= 0
    pc = z["pc"][sel] / 255.0; me = z["me"][sel] * 0.04; w = W[col[sel]]; r_, c_ = row[sel], co[col[sel]]
    PC = np.zeros((len(z["genes"]), len(cls))); MX = np.zeros_like(PC)
    np.add.at(PC, (r_, c_), pc * w); np.add.at(MX, (r_, c_), me * pc * w)
    tot = np.zeros(len(cls)); np.add.at(tot, co[cols], W[cols]); PC /= tot; MX /= tot
    names = {pairs[i]["cell_type"]: pairs[i]["cell_type_name"] for i in cols}
    genes = np.array([str(g).split(".")[0] for g in z["genes"]]); keep = PC.max(1) > 0
    return genes[keep], cls, [names[c] for c in cls], PC[keep].astype(np.float32), MX[keep].astype(np.float32)


def part_a():
    g, cls, names, PC, MX = cx_celltypes()
    Lg = np.log1p(10 * MX); spec = (Lg - Lg.mean(1, keepdims=True)) / (Lg.std(1, keepdims=True) + 1e-6)
    act = PC >= 0.10; specific = act & (spec >= 3); hk = act.mean(1) >= 0.80
    os.makedirs(cbb.LAY, exist_ok=True)
    p = cbb.save_layer("ACTIVE_GENES_CELLTYPE", rows=g, cols=np.array(cls), col_names=np.array(names), pc=np.round(PC * 255).astype(np.uint8),
                       mean=MX.astype(np.float16), spec=spec.astype(np.float16))
    return dict(genes=g, cls=cls, names=names, PC=PC, MX=MX, spec=spec, act=act, specific=specific, hk=hk, path=p)


# ------------------------------------------------------------------------------------------------ R1 causal edges
def screen(layer, h5):
    import h5py
    z = npz(layer); lab = z["labels"].astype(str); X = z["X"].astype(np.float32); rows = z["rows"].astype(str); cols = z["cols"].astype(str)
    ctrl = np.array(["non-targeting" in l.lower() for l in lab])
    sd = np.nanstd(X[ctrl], 0); sd = np.where(sd > 0, sd, np.nan); Z = X / sd[None, :]
    with h5py.File(L(h5), "r") as f:
        pv = f["obs"]["energy_test_p_value"][:]
    return dict(Z=Z, rows=rows, cols=cols, ctrl=ctrl, pheno=(pv < 0.05) & ~ctrl)


def tss():
    t = json.load(open(L("TSS_GENCODE46"))); return {g[0]: (g[3], g[4]) for g in t}


def r1_discover(G, T_, regs):
    """GWPS: per regulator, targets with |z| >= 4 (top 300), self and cis (TSS within 50 kb) excluded."""
    Z, cols = G["Z"], G["cols"]; out = []
    for i in regs:
        r = G["rows"][i]; z = Z[i]; ok = np.isfinite(z) & (np.abs(z) >= 4)
        idx = np.nonzero(ok)[0]
        if not len(idx):
            continue
        idx = idx[np.argsort(-np.abs(z[idx]))][:300]; rt = T_.get(r)
        for j in idx:
            c = cols[j]
            if c == r:
                continue
            ct = T_.get(c)
            if rt and ct and rt[0] == ct[0] and abs(rt[1] - ct[1]) <= 50_000:
                continue
            out.append((i, j, float(z[j])))
    return out


def r1_confirm(G, E, found, Zess=None):
    Ze = E["Z"] if Zess is None else Zess
    er = {r: i for i, r in enumerate(E["rows"]) if not E["ctrl"][i]}; ec = {c: j for j, c in enumerate(E["cols"])}
    test, pz = [], []
    for i, j, z in found:
        a, b = er.get(G["rows"][i]), ec.get(G["cols"][j])
        if a is None or b is None or not np.isfinite(Ze[a, b]):
            continue
        test.append((i, j, z, float(Ze[a, b]))); pz.append(-np.sign(z) * Ze[a, b])
    from scipy.stats import norm
    p = norm.cdf(np.array(pz)) if pz else np.array([]); q = bh(p)
    return [(i, j, z, ze, float(qq)) for (i, j, z, ze), qq in zip(test, q) if qq < 0.05 and np.sign(ze) == np.sign(z)], len(test)


def run_r1():
    G = screen("KO_K562_GENESPACE", "GWPS_K562_H5AD"); E = screen("KO_K562ESS_GENESPACE", "ESS_K562_H5AD"); T_ = tss()
    regs = np.nonzero(G["pheno"])[0]
    if TRIAL:
        regs = regs[:300]
    found = r1_discover(G, T_, regs); conf, testable = r1_confirm(G, E, found)
    rng = np.random.default_rng(7); Zs = E["Z"][:, rng.permutation(E["Z"].shape[1])]; nconf, _ = r1_confirm(G, E, found, Zs)
    # positive: 200 planted pairs between regulators in both screens and targets measured in both
    er = {r: i for i, r in enumerate(E["rows"]) if not E["ctrl"][i]}; ec = {c: j for j, c in enumerate(E["cols"])}
    both = [i for i in regs if G["rows"][i] in er]; tcols = [j for j, c in enumerate(G["cols"]) if c in ec]
    Zg2, Ze2 = G["Z"].copy(), E["Z"].copy(); planted = set()
    for _ in range(200):
        i = int(rng.choice(both)); j = int(rng.choice(tcols)); planted.add((i, j))
        Zg2[i, j] = -6.0; Ze2[er[G["rows"][i]], ec[G["cols"][j]]] = -6.0
    G2 = dict(G, Z=Zg2); f2 = r1_discover(G2, T_, sorted({i for i, _ in planted})); c2, _ = r1_confirm(G2, E, f2, Ze2)
    pr = np.mean([(i, j) in {(a, b) for a, b, *_ in c2} for i, j in planted])
    H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e)
    fset = {(G["rows"][i], G["cols"][j]): z for i, j, z in found}; cset = {(G["rows"][i], G["cols"][j]) for i, j, *_ in conf}
    known = []
    for a, b in KNOWN:
        ea, eb = cbb.to_ens(a), cbb.to_ens(b); f_ = fset.get((ea, eb))
        ok = f_ is not None and f_ < 0 and ((ea, eb) in cset or ea not in er)
        known.append((a, b, bool(ok)))
    own = [G["Z"][i, list(G["cols"]).index(G["rows"][i])] for i in regs if G["rows"][i] in set(G["cols"])]
    edges = [dict(regulator=G["rows"][i], regulator_symbol=sym(G["rows"][i]), target=G["cols"][j], target_symbol=sym(G["cols"][j]),
                  z_gw=round(z, 2), z_ess=round(ze, 2), q=q, kind="activates" if z < 0 else "represses") for i, j, z, ze, q in conf]
    gate = dict(confirmed=len(conf) >= 1, negative=len(nconf) <= max(1, 0.05 * len(conf)), planted=pr >= 0.9, known=sum(k[2] for k in known) >= 4)
    gate["verdict"] = "PASS" if all(gate.values()) else "FAIL"
    res = dict(regulators=int(len(regs)), found=len(found), testable=testable, confirmed=len(conf), activations=sum(e["kind"] == "activates" for e in edges),
               repressions=sum(e["kind"] == "represses" for e in edges), negative_confirmed=len(nconf), planted_recovered=float(pr), known=[list(k) for k in known],
               on_target_lowered=float(np.mean(np.array(own) < -2)) if own else None, regulators_with_confirmed=len({e["regulator"] for e in edges}), gate=gate)
    found_rows = [dict(regulator=G["rows"][i], target=G["cols"][j], z_gw=round(z, 2)) for i, j, z in found]
    return res, edges, found_rows


# ------------------------------------------------------------------------------------------------ R2, R3
def logz(MX):
    Lg = np.log1p(10 * MX); return (Lg - Lg.mean(1, keepdims=True)) / (Lg.std(1, keepdims=True) + 1e-6)


def run_r2(A, edges):
    gi = {g: i for i, g in enumerate(A["genes"])}; Zc = logz(A["MX"]); n = Zc.shape[1]
    ed = [(gi[e["regulator"]], gi[e["target"]]) for e in edges if e["kind"] == "activates" and e["regulator"] in gi and e["target"] in gi]
    if not ed:
        return dict(edges=0, gate=dict(verdict="FAIL")), []
    rng = np.random.default_rng(11); lev = A["MX"].mean(1); q = np.quantile(lev, np.linspace(0, 1, 11)); dec = np.clip(np.searchsorted(q, lev, side="right") - 1, 0, 9)
    pools = [np.nonzero(dec == k)[0] for k in range(10)]
    corr = lambda a, b: (Zc[a] * Zc[b]).sum(1) / n

    def draw(k):
        r = np.array([a for a, _ in ed] * k); t = np.array([rng.choice(pools[dec[b]]) for _, b in ed] * 1 for _ in range(k)).ravel() if False else \
            np.concatenate([np.array([rng.choice(pools[dec[b]]) for _, b in ed]) for _ in range(k)])
        return corr(r, t)
    from scipy.stats import rankdata

    def auroc(pos, neg):
        rk = rankdata(np.r_[pos, neg]); n1 = len(pos); return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))
    obs_c = corr(np.array([a for a, _ in ed]), np.array([b for _, b in ed])); ref = draw(5); obs = auroc(obs_c, ref)
    null = [auroc(draw(1), ref) for _ in range(200)]; p = (1 + sum(x >= obs for x in null)) / 201
    ok = obs >= 0.55 and p < 0.05
    return dict(edges=len(ed), auroc=obs, null_mean=float(np.mean(null)), p=p, median_r_edges=float(np.median(obs_c)), median_r_random=float(np.median(ref)),
                gate=dict(verdict="PASS" if ok else "FAIL")), []


def tf_list():
    import pandas as pd
    import zipfile
    with zipfile.ZipFile(L("HPA_PROTEINATLAS")) as z:
        d = pd.read_csv(z.open(z.namelist()[0]), sep="\t", usecols=["Gene", "Ensembl", "Protein class"])
    return set(d[d["Protein class"].fillna("").str.contains("Transcription factors")].Ensembl)


def run_r3(A, found_rows, G_regs_cols):
    tfs = tf_list(); halves = []
    for h in (0, 1):
        g, cls, names, PC, MX = cx_celltypes(h); halves.append((g, cls, PC, MX))
    common = sorted(set(halves[0][0]) & set(halves[1][0]))
    cl_common = sorted(set(halves[0][1]) & set(halves[1][1]))
    prof = []
    for g, cls, PC, MX in halves:
        gi = {x: i for i, x in enumerate(g)}; ci = [cls.index(c) for c in cl_common]
        M = MX[[gi[x] for x in common]][:, ci]; P = PC[[gi[x] for x in common]][:, ci]
        Zh = logz(M); Zh[~np.isfinite(Zh)] = 0; prof.append((Zh, P))
    act = (prof[0][1] >= 0.10).any(1) | (prof[1][1] >= 0.10).any(1)
    genes = np.array(common)[act]; Z0, Z1 = prof[0][0][act], prof[1][0][act]; n = Z0.shape[1]
    ti = [i for i, x in enumerate(genes) if x in tfs]
    if TRIAL:
        ti = ti[:150]

    def links(Za, Zb, tfi, rng=None):
        out = []
        for s in range(0, len(tfi), 200):
            blk = tfi[s:s + 200]; Ra = Za[blk] @ Za.T / n; Rb = Zb[blk] @ Zb.T / n
            for k, i in enumerate(blk):
                ra = Ra[k].copy(); ra[i] = -1; top = np.argsort(-ra)[:100]; top = top[ra[top] >= 0.5]
                out += [(i, int(j), float(ra[j]), float(Rb[k, j])) for j in top if Rb[k, j] >= 0.4]
        return out
    conf = links(Z0, Z1, ti)
    rng = np.random.default_rng(13); nconf = links(Z0, Z1[rng.permutation(len(Z1))], ti)
    # the claim: co-expression links predict causal activation in K562
    gw_regs, gw_cols, Zgw = G_regs_cols
    act_truth = {(r["regulator"], r["target"]) for r in found_rows if r["z_gw"] <= -4}
    tf_test = [i for i in ti if genes[i] in gw_regs]; tcol = [j for j, x in enumerate(genes) if x in gw_cols]
    cset = {(genes[i], genes[j]) for i, j, *_ in conf}

    def table(cs):
        a = b = c = d = 0
        for i in tf_test:
            for j in tcol:
                if i == j:
                    continue
                k = (genes[i], genes[j]); x, y = k in cs, k in act_truth
                if x and y: a += 1
                elif x: b += 1
                elif y: c += 1
                else: d += 1
        return [[a, b], [c, d]]
    from scipy.stats import fisher_exact
    t = table(cset); OR, p = fisher_exact(t, alternative="greater") if t[0][0] + t[0][1] and t[1][0] + t[1][1] else (float("nan"), 1.0)
    perm = dict(zip([genes[i] for i in ti], rng.permutation([genes[i] for i in ti])))
    tn = table({(perm.get(a, a), b) for a, b in cset}); ORn, pn = fisher_exact(tn) if tn[0][0] + tn[0][1] else (float("nan"), 1.0)
    ok = OR >= 1.5 and p < 0.05 and (np.isnan(ORn) or 0.67 <= ORn <= 1.5)
    H = cbb.hgnc()
    edges = [dict(tf=genes[i], tf_symbol=H["e2s"].get(genes[i], genes[i]), gene=genes[j], gene_symbol=H["e2s"].get(genes[j], genes[j]), r_found=round(ra, 3), r_confirmed=round(rb, 3))
             for i, j, ra, rb in conf]
    res = dict(tfs=len(ti), genes=int(len(genes)), cell_types=n, confirmed=len(conf), negative_confirmed=len(nconf), tfs_with_links=len({e["tf"] for e in edges}),
               k562_table=t, odds_ratio=float(OR), p=float(p), negative_table=tn, negative_odds_ratio=float(ORn), gate=dict(verdict="PASS" if ok else "FAIL"))
    return res, edges


# ------------------------------------------------------------------------------------------------ concepts, per cell type
def concepts(edges_by_reg, universe, min_n=10):
    from scipy.stats import hypergeom
    pw = json.load(open(L("SETS_PATHWAY"))); U = set(universe); P = {k: (d["name"], set(d["members"]) & U) for k, d in pw.items()}
    P = {k: v for k, v in P.items() if len(v[1]) >= 5}; N = len(U); out = {}
    for reg, tg in edges_by_reg.items():
        tg = set(tg) & U
        if len(tg) < min_n:
            continue
        rows = []
        for k, (nm, m) in P.items():
            x = len(tg & m)
            if x >= 3:
                rows.append((nm, x, len(m), float(hypergeom.sf(x - 1, N, len(m), len(tg)))))
        if rows:
            q = bh([r[3] for r in rows]); keep = sorted([(r, qq) for r, qq in zip(rows, q) if qq < 0.05], key=lambda x: x[0][3])[:5]
            if keep:
                out[reg] = [dict(pathway=r[0], targets_in=r[1], size=r[2], q=float(qq)) for r, qq in keep]
    return out


# ------------------------------------------------------------------------------------------------ ledger
def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def _git(*a):
    import subprocess
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True)


def declare():
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            if cid in have:
                print(cid, "already declared"); continue
            f.write(json.dumps(dict(op="declare", id=cid, at=cbb.now(), code_sha=code_sha(), commit=_git("rev-parse", "HEAD").stdout.strip(),
                                    claim=__doc__.split(cid)[1].split(" G1-R")[0].split("DESCRIPTIVE")[0].strip()[:900], trial=TRIAL), sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = _git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G1 already tested")
    t0 = time.time(); os.makedirs(RUN, exist_ok=True); os.makedirs(OUTD, exist_ok=True)
    A = part_a(); print(f"part A: {len(A['genes'])} genes x {len(A['cls'])} cell types ({time.time() - t0:.0f}s)", flush=True)
    r1, e1, found = run_r1(); print("G1-R1", json.dumps(r1)[:900], f"({time.time() - t0:.0f}s)", flush=True)
    r2, _ = run_r2(A, e1); print("G1-R2", json.dumps(r2), flush=True)
    G = screen("KO_K562_GENESPACE", "GWPS_K562_H5AD"); regs = set(G["rows"][G["pheno"]])
    r3, e3 = run_r3(A, found, (regs, set(G["cols"]), None)); print("G1-R3", json.dumps(r3), f"({time.time() - t0:.0f}s)", flush=True)
    H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e)
    act_by, rep_by = collections.defaultdict(list), collections.defaultdict(list)
    for e in e1:
        (act_by if e["kind"] == "activates" else rep_by)[e["regulator"]].append(e["target"])
    co_by = collections.defaultdict(list)
    for e in e3:
        co_by[e["tf"]].append(e["gene"])
    con_act = concepts(act_by, G["cols"]); con_rep = concepts(rep_by, G["cols"]); con_co = concepts(co_by, A["genes"], min_n=20)
    # per cell type
    gi = {g: i for i, g in enumerate(A["genes"])}; tfs = tf_list(); per_ct = {}
    for k, cl in enumerate(A["cls"]):
        on = A["act"][:, k]; ong = {A["genes"][i] for i in np.nonzero(on)[0]}
        spec_i = np.nonzero(A["specific"][:, k])[0]; spec_i = spec_i[np.argsort(-A["spec"][spec_i, k])]
        regs_on = [A["genes"][i] for i in spec_i if A["genes"][i] in tfs]
        e1_on = [e for e in e1 if e["regulator"] in ong and e["target"] in ong]; e3_on = [e for e in e3 if e["tf"] in ong and e["gene"] in ong]
        top_reg = collections.Counter(e["tf"] for e in e3_on if e["tf"] in set(regs_on)).most_common(5)
        per_ct[cl] = dict(name=A["names"][k], active_genes=int(on.sum()), specific_genes=int(len(spec_i)),
                          top_specific=[sym(A["genes"][i]) for i in spec_i[:25]], specific_regulators=[sym(x) for x in regs_on[:15]],
                          k562_links_active=len(e1_on), coexpression_links_active=len(e3_on),
                          programmes=[dict(regulator=sym(r), targets=[sym(e["gene"]) for e in e3_on if e["tf"] == r][:15]) for r, _ in top_reg])
    out = dict(part_a=dict(genes=len(A["genes"]), cell_types=len(A["cls"]), housekeeping=int(A["hk"].sum()),
                           median_active=float(np.median(A["act"].sum(0))), median_specific=float(np.median(A["specific"].sum(0)))),
               concepts_activated=len(con_act), concepts_repressed=len(con_rep), concepts_coexpression=len(con_co))
    print(json.dumps(out), flush=True)
    # layers and bank
    cbb.save_layer("GENE_INFLUENCE_K562", regulator=np.array([e["regulator"] for e in e1]), target=np.array([e["target"] for e in e1]),
                   z_gw=np.array([e["z_gw"] for e in e1], np.float32), z_ess=np.array([e["z_ess"] for e in e1], np.float32), q=np.array([e["q"] for e in e1], np.float32))
    cbb.save_layer("GENE_COEXPRESSION_TF", tf=np.array([e["tf"] for e in e3]), gene=np.array([e["gene"] for e in e3]),
                   r_found=np.array([e["r_found"] for e in e3], np.float32), r_confirmed=np.array([e["r_confirmed"] for e in e3], np.float32))
    facts = [] if TRIAL else bank_facts(A, e1, e3, con_act, con_rep, con_co, per_ct)
    n = cbb.bank().add_many(facts) if facts else 0
    for nm, obj in (("influence_k562.json", e1), ("coexpression_tf.json", e3), ("concepts.json", dict(activated=con_act, repressed=con_rep, coexpression=con_co)),
                    ("cell_types.json", per_ct)):
        json.dump(obj, open(os.path.join(OUTD, nm), "w"), default=float)
    for cid, r in (("G1-R1", r1), ("G1-R2", r2), ("G1-R3", r3)):
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(USED), **r), sort_keys=True, default=float) + "\n")
    Ls = ["=" * 97, "GENEREG -- active genes per cell type, and which gene switches which", "=" * 97,
          f"  Part A: {out['part_a']['genes']:,} genes x {out['part_a']['cell_types']} cell types; median active {out['part_a']['median_active']:.0f}, "
          f"specific {out['part_a']['median_specific']:.0f}; housekeeping {out['part_a']['housekeeping']:,}",
          f"  G1-R1 who switches whom (K562): {r1['confirmed']:,} confirmed ({r1['activations']:,} activations, {r1['repressions']:,} repressions) from "
          f"{r1['regulators_with_confirmed']} regulators; shuffled {r1['negative_confirmed']}; planted {r1['planted_recovered']:.0%}; known "
          f"{sum(k[2] for k in r1['known'])}/6 -> {r1['gate']['verdict']}",
          f"  G1-R2 K562 activations across body cell types: AUROC {r2.get('auroc', float('nan')):.3f}, p {r2.get('p', 1):.3g} -> {r2['gate']['verdict']}",
          f"  G1-R3 co-expression switch map: {r3['confirmed']:,} TF->gene links ({r3['tfs_with_links']} TFs); predicts K562 activation OR "
          f"{r3['odds_ratio']:.2f}, p {r3['p']:.3g} (shuffled OR {r3['negative_odds_ratio']:.2f}) -> {r3['gate']['verdict']}",
          f"  Concepts: {len(con_act)} regulators with enriched activated pathways, {len(con_rep)} repressed, {len(con_co)} co-expression TFs; facts written {n:,}"]
    open(RES, "w").write("\n".join(Ls) + "\n"); print("\n".join(Ls))


def bank_facts(A, e1, e3, con_act, con_rep, con_co, per_ct):
    t = cbb.now(); src = {"name": "genereg:G1", "code_sha": code_sha()}; inp = tuple(sorted(USED)); F = []
    H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in e1:
        by[e["regulator"]]["activates" if e["kind"] == "activates" else "represses"].append([sym(e["target"]), e["z_gw"], e["z_ess"]])
        by[e["target"]]["activated_by" if e["kind"] == "activates" else "repressed_by"].append([sym(e["regulator"]), e["z_gw"], e["z_ess"]])
    for e in e3:
        by[e["tf"]]["coexpressed_targets"].append([sym(e["gene"]), e["r_confirmed"]])
    for g, d in by.items():
        for k, v in d.items():
            v = sorted(v, key=lambda x: -abs(x[-1]))[:60]
            F.append(Fact(g, f"influence/{k}", v, t[:10], t, "derived", src, inputs=inp, context={"n": len(d[k])}))
    for name, C in (("activates", con_act), ("represses", con_rep), ("coexpression", con_co)):
        for g, rows in C.items():
            F.append(Fact(g, f"concepts/{name}", rows, t[:10], t, "derived", src, inputs=inp))
    for i in np.nonzero(A["hk"])[0]:
        F.append(Fact(A["genes"][i], "role/housekeeping", {"active_share": round(float(A["act"][i].mean()), 3)}, t[:10], t, "derived", src, inputs=inp))
    sp = A["specific"]
    for i in np.nonzero(sp.any(1))[0]:
        ks = np.nonzero(sp[i])[0]; ks = ks[np.argsort(-A["spec"][i, ks])][:10]
        F.append(Fact(A["genes"][i], "role/marker_of", [[A["names"][k], round(float(A["spec"][i, k]), 1)] for k in ks], t[:10], t, "derived", src, inputs=inp))
    for cl, d in per_ct.items():
        F.append(Fact(f"CT-{cl}", "active/summary", d, t[:10], t, "derived", src, inputs=inp))
    return F


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
