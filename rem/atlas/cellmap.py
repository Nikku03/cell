"""CELLMAP -- every cell mapped down to single proteins: which proteins are there, where they sit, who they work with,
which machines (complexes) are complete, what jobs the cell is doing, and which proteins have a backup.

Asked by the user: "if we have data for what each gene is active in which cell, what each gene does, protein
expression and protein interactions, can't we map all the cell down to single protein?" -- "yes build it".
CELLS (three families):
  BODY    840 human cell types (CELLxGENE aggregates, the genereg G1 layer ACTIVE_GENES_CELLTYPE): present = made in
          >= 10% of the cell type's cells; level = mean expression.
  LINES   the four screened lines K562, RPE1, HepG2, Jurkat (HPA cell-line RNA): present = nTPM >= 1.
  CANCER  1,673 cancer lines (DepMap 24Q4 RNA, EXPR_DEPMAP): present = TPM >= 1 (log2(TPM+1) >= 1).
  Proteins = HGNC protein-coding genes measured in that family.
PER CELL, PER PROTEIN: room(s) (HPA + OpenCell, 11 rooms, as G4); ACTIVE PARTNERS = interaction partners (HPA,
OpenCell, Complex Portal) present in the same cell and sharing a room (the rule G4 tested: PASS); MACHINES = Complex
Portal complexes, fraction of measured parts present, complete if all; JOBS = Reactome pathways (10-300 measured
members) enriched among the cell's proteins (hypergeometric, BH q < 0.05 within the cell); BACKUP = closest paralog
(>= 30% identity, G5) also present; ESTIMATED AMOUNT = RNA level x the gene's protein-per-RNA ratio measured in HEK293T
(OpenCell; Edfors 2016 found such gene-specific ratios carry across tissues) -- an estimate, used only to order
proteins within a cell. Complex names from the Complex Portal human file (EBI).
Also registers the genereg G1 layers (ACTIVE_GENES_CELLTYPE, GENE_INFLUENCE_K562, GENE_COEXPRESSION_TF) as bank layers
(they were saved as files but not banked).

CLAIM (declared before running; rem/atlas/cellbrain_ledger.jsonl, id G7-M1; the brain's rules):
 G7-M1  A COMPLETE MACHINE NEEDS ITS PARTS. In the 1,103 cancer lines with CRISPR gene effects (DepMap) and RNA: for
        each Complex Portal complex with 3-60 parts, >= 3 measured, and each measured part A, among lines where A is
        made: d = mean effect of losing A where every other measured part is made (machine complete) minus where at
        least one is not (machine incomplete); >= 20 lines in each group. More negative effect = more needed.
        CLAIM: median d <= -0.02, one-sided sign test (d < 0) p < 0.05, AND Monte Carlo p < 0.05 against 200 draws in
        which each complex's other parts are swapped for random protein-coding genes of the same RNA-level decile
        (keeps "more genes made" effects of a line). Negative (must not pass): the first random draw. Planted (must
        pass): real d - 0.05.
NOT: RNA stands in for protein (made = RNA present); rooms and interactions come from a few cell lines; no timing, no
modifications, no amounts per room; single-cell data sees only the better-expressed genes (~1,500-4,000 per cell type);
estimated amounts are estimates.

Run: python3 -I rem/atlas/cellmap.py pull | build | declare | test [--trial] | page OUTDIR
"""

from __future__ import annotations
import collections
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
TRIAL = "--trial" in sys.argv


def _load(name):
    argv = sys.argv; sys.argv = [sys.argv[0]] + (["--trial"] if TRIAL else [])
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


pp = _load("generegppi"); pa = _load("generegpara"); xb = _load("xcellbank")
gx = pp.gx; cbb, Fact = gx.cbb, gx.Fact
LEDGER = gx.LEDGER
RAW = os.path.join(HERE, "_cache", "cellmap", "raw")
CP_URL = "https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/9606.tsv"
OUT = R("outputs", "cellmap")
CLAIMS = ["G7-M1"]
SEED, NMC = 20261127, (10 if TRIAL else 200)
ROOMS = list(pp.ROOMS)
cbb.SRC.setdefault("depmap_expr", ("DepMap 24Q4 Public: OmicsExpressionProteinCodingGenesTPMLogp1 and Model.csv (figshare 27993248)",
                                   "https://doi.org/10.25452/figshare.plus.27993248", "CC BY 4.0"))


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "cellmap.py"), "rb").read())
    for m in (pp, pa):
        h.update(m.code_sha().encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ pull
def pull():
    os.makedirs(RAW, exist_ok=True)
    print(xb.fetch(CP_URL, os.path.join(RAW, "complexportal_9606.tsv")))


def complex_names():
    import csv
    out = {}
    with open(os.path.join(RAW, "complexportal_9606.tsv"), newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row and row[0].startswith("CPX-"):
                out[row[0]] = row[1]
    return out


# ------------------------------------------------------------------------------------------------ knowledge shared by all cells
def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); q = np.empty(m); run = 1.0
    for r in range(m - 1, -1, -1):
        run = min(run, p[o[r]] * m / (r + 1)); q[o[r]] = run
    return q


def protein_coding():
    import pandas as pd
    h = pd.read_csv(os.path.join(HERE, "hgnc_complete_set.txt"), sep="\t", dtype=str, usecols=["ensembl_gene_id", "locus_group"])
    return set(h[(h.locus_group == "protein-coding gene") & h.ensembl_gene_id.notna()].ensembl_gene_id)


def knowledge():
    ppi, rooms = pp.knowledge(); best, fam = pa.substitutes()
    comp = {k: v["members"] for k, v in json.load(open(gx.L("SETS_COMPLEX"))).items()}
    pw = {k: (v["name"], v["members"]) for k, v in json.load(open(gx.L("SETS_PATHWAY"))).items()}
    oc = gx.npz("EXPR_OPENCELL_HEK293T"); ptr = {}
    for g, x in zip(oc["rows"].astype(str), oc["X"].astype(np.float64)):
        if np.isfinite(x[0]) and np.isfinite(x[1]) and x[0] > 0 and x[1] > 0:
            ptr[g] = float(np.log2(x[1] / x[0]))
    med = float(np.median(list(ptr.values()))) if ptr else 0.0
    return dict(ppi=ppi, rooms=rooms, best=best, comp=comp, pw=pw, ptr=ptr, ptr_med=med, names=complex_names(), pc=protein_coding())


def families():
    """yield (family, genes, cell ids, cell names, extra info, present bool genes x cells, level, level is log)"""
    z = np.load(os.path.join(cbb.LAY, "active_genes_celltype.npz"), allow_pickle=False)
    g = z["rows"].astype(str); pc = z["pc"].astype(np.float32) / 255.0
    yield "BODY", g, z["cols"].astype(str), z["col_names"].astype(str), None, pc >= 0.10, z["mean"].astype(np.float32), False
    E = gx.npz("EXPR_HPA_CELLLINES"); cols = [str(c) for c in E["cols"]]; X = E["X"].astype(np.float32)
    lines = {"K562": ["K-562"], "RPE1": ["hTERT-RPE1"], "HepG2": ["Hep-G2"], "Jurkat": ["JURKAT", "Jurkat E6.1"]}
    L = np.column_stack([X[:, [cols.index(c) for c in v]].mean(1) for v in lines.values()])
    yield "LINES", E["rows"].astype(str), np.array(list(lines)), np.array(list(lines)), None, L >= np.log1p(1.0), np.expm1(L), True
    D = gx.npz("EXPR_DEPMAP"); ids = D["cols"].astype(str)
    import pandas as pd
    m = pd.read_csv(os.path.join(HERE, "_cache", "paralog", "depmap", "raw", "Model.csv"), dtype=str).set_index("ModelID")
    nm = np.array([str(m.CellLineName.get(i, i)) for i in ids]); lin = [str(m.OncotreeLineage.get(i, "")) for i in ids]
    Xd = D["X"].astype(np.float32)
    yield "CANCER", D["rows"].astype(str), ids, nm, lin, Xd >= 1.0, np.exp2(Xd) - 1, True


def build_family(fam, genes, cids, cnames, extra, P, LV, K):
    from scipy import sparse
    from scipy.stats import hypergeom
    pc = K["pc"]; keep = np.array([gg in pc for gg in genes]); seen = set(); k2 = []
    for i in np.nonzero(keep)[0]:
        if genes[i] not in seen:
            seen.add(genes[i]); k2.append(i)
    k2 = np.array(k2); U = genes[k2]; P = P[k2]; LV = LV[k2]; nU, nC = P.shape; ui = {gg: i for i, gg in enumerate(U)}
    Pf = P.astype(np.float32)
    RM = np.zeros((nU, len(ROOMS)), bool)
    for i, gg in enumerate(U):
        for rr in K["rooms"].get(gg, ()):
            RM[i, ROOMS.index(rr)] = True
    room_counts = (Pf.T @ RM.astype(np.float32)).astype(np.int32)                    # cells x rooms
    # active partnerships: interacting, sharing a room, both present
    ea, eb = [], []
    for p in K["ppi"]:
        x, y = tuple(p)
        if x in ui and y in ui and (RM[ui[x]] & RM[ui[y]]).any():
            ea.append(ui[x]); eb.append(ui[y])
    A = sparse.csr_matrix((np.ones(2 * len(ea), np.float32), (ea + eb, eb + ea)), shape=(nU, nU))
    act = (np.asarray(A.dot(Pf)) * Pf).astype(np.uint16)                               # proteins x cells: active partners
    # machines
    cids_c = [c for c, m in K["comp"].items() if sum(x in ui for x in m) >= 2]
    rr_, cc_ = [], []
    for j, c in enumerate(cids_c):
        for x in K["comp"][c]:
            if x in ui:
                rr_.append(ui[x]); cc_.append(j)
    C = sparse.csr_matrix((np.ones(len(rr_), np.float32), (rr_, cc_)), shape=(nU, len(cids_c)))
    Cc = C.tocsc(); size = np.asarray(C.sum(0)).ravel(); have = np.asarray(C.T.dot(Pf)).T                # cells x complexes
    frac = (have / size[None, :]).astype(np.float32)
    # jobs
    pws = [(k, v[0], [ui[x] for x in v[1] if x in ui]) for k, v in K["pw"].items()]
    pws = [x for x in pws if 10 <= len(x[2]) <= 300]
    rr_, cc_ = [], []
    for j, (_, _, mm) in enumerate(pws):
        rr_ += mm; cc_ += [j] * len(mm)
    W = sparse.csr_matrix((np.ones(len(rr_), np.float32), (rr_, cc_)), shape=(nU, len(pws)))
    hit = np.asarray(W.T.dot(Pf)).T; Ksz = np.asarray(W.sum(0)).ravel(); n = P.sum(0)
    pv = hypergeom.sf(hit - 1, nU, Ksz[None, :], n[:, None])
    # backups
    sub = np.array([ui.get(K["best"][gg][0], -1) if gg in K["best"] else -1 for gg in U]); hs = sub >= 0
    covered = np.zeros((nU, nC), bool); covered[hs] = P[hs] & P[sub[hs]]
    # estimated amount (ordering only)
    lp = np.array([K["ptr"].get(gg, K["ptr_med"]) for gg in U], np.float32)
    est = np.log2(np.maximum(LV, 0) + 1e-3) + lp[:, None]
    H = cbb.hgnc(); sym = np.array([H["e2s"].get(gg, gg) for gg in U])
    summaries = {}
    for c in range(nC):
        pres = np.nonzero(P[:, c])[0]
        if not len(pres):
            continue
        rooms_top = {}
        for r_i, rname in enumerate(ROOMS):
            ids = pres[RM[pres, r_i]]
            if len(ids):
                o = ids[np.argsort(-est[ids, c])][:10]; rooms_top[rname] = dict(n=int(len(ids)), top=[str(sym[i]) for i in o])
        comp_full = np.nonzero(frac[c] >= 1.0)[0]; comp_full = comp_full[np.argsort(-size[comp_full])]
        near = np.nonzero((frac[c] >= 0.75) & (frac[c] < 1.0) & (size >= 3))[0]; near = near[np.argsort(-frac[c, near])]
        q = bh(pv[c]); jobs = [j for j in np.argsort(pv[c])[:8] if q[j] < 0.05]
        hubs = pres[np.argsort(-act[pres, c].astype(np.int32))][:10]
        cov = pres[covered[pres, c]]; cov = cov[np.argsort(-est[cov, c])][:10]

        def missing(j):
            return [str(sym[i]) for i in Cc[:, j].nonzero()[0] if not P[i, c]][:6]
        summaries[str(cids[c])] = dict(
            name=str(cnames[c]), lineage=(extra[c] if extra else None), proteins=int(len(pres)), rooms=rooms_top,
            active_partnerships=int(act[:, c].sum() // 2), machines_complete=int(len(comp_full)), machines_total=int(len(cids_c)),
            complete=[[K["names"].get(cids_c[j], cids_c[j]), int(size[j])] for j in comp_full[:15]],
            nearly_complete=[[K["names"].get(cids_c[j], cids_c[j]), int(size[j]), round(float(frac[c, j]), 2), missing(j)] for j in near[:10]],
            jobs=[[pws[j][1], int(hit[c, j]), int(Ksz[j]), float(q[j])] for j in jobs],
            hubs=[[str(sym[i]), int(act[i, c])] for i in hubs if act[i, c] > 0],
            backups_covered=int(covered[:, c].sum()), backup_examples=[[str(sym[i]), str(sym[sub[i]])] for i in cov])
    np.savez_compressed(os.path.join(cbb.LAY, f"cellmap_{fam.lower()}.npz"), genes=U, cells=np.array(cids, dtype=str), cell_names=np.array(cnames, dtype=str),
                        present=np.packbits(P, axis=1), n_cells=np.array([nC]), active_partners=act, rooms=RM, complexes=np.array(cids_c),
                        complex_frac=frac.astype(np.float16), estimated_log2=est.astype(np.float16))
    stats = dict(cells=int(nC), proteins_universe=int(nU), median_proteins_per_cell=float(np.median(P.sum(0))), interacting_same_room_pairs=len(ea),
                 complexes=len(cids_c), median_complete_machines=float(np.median((frac >= 1.0).sum(1))), pathways=len(pws))
    return summaries, stats


def build():
    t0 = time.time(); K = knowledge(); print(f"knowledge ({time.time() - t0:.0f}s): {len(K['ppi']):,} interactions, {len(K['rooms']):,} genes with rooms, "
                                             f"{len(K['comp']):,} complexes, {len(K['pw']):,} pathways, {len(K['ptr']):,} protein/RNA ratios", flush=True)
    os.makedirs(OUT, exist_ok=True); lf, facts, allstats, page = {}, [], {}, {}
    # register the G1 layers
    for name, src, rows, cols, units, inp in (
            ("ACTIVE_GENES_CELLTYPE", "cellxgene", "genes (Ensembl)", "840+ cell types (CL)", "pc = share of cells expressing x 255; mean; specificity z", "CX_HUMAN"),
            ("GENE_INFLUENCE_K562", "replogle2022", "confirmed regulator -> target links", "z_gw, z_ess, q", "z", "KO_K562_GENESPACE"),
            ("GENE_COEXPRESSION_TF", "cellxgene", "TF -> gene co-expression links", "r_found, r_confirmed", "Pearson r", "CX_HUMAN")):
        f = cbb.layer_fact(name, 1, src, os.path.join(cbb.LAY, name.lower() + ".npz"), rows, cols, units,
                           inputs=(cbb.bank().facts(f"LAYER-{inp}", "layer")[-1].ref,), tier="derived", extra={"made_by": "genereg G1"})
        facts.append(f); lf[name] = f.ref
    f = cbb.layer_fact("COMPLEXPORTAL_HUMAN_TSV", 1, "complexportal", os.path.join(RAW, "complexportal_9606.tsv"), "complexes", "Complex Portal columns", "names, members")
    facts.append(f); lf["COMPLEXPORTAL_HUMAN_TSV"] = f.ref
    t = cbb.now(); src = {"name": "cellmap", "code_sha": code_sha()}
    for fam, genes, cids, cnames, extra, P, LV, _ in families():
        S, st = build_family(fam, genes, cids, cnames, extra, P, LV, K); allstats[fam] = st
        print(f"{fam}: {json.dumps(st)} ({time.time() - t0:.0f}s)", flush=True)
        srck = {"BODY": "cellxgene", "LINES": "hpa", "CANCER": "depmap_expr"}[fam]
        f = cbb.layer_fact(f"CELLMAP_{fam}", 1, srck, os.path.join(cbb.LAY, f"cellmap_{fam.lower()}.npz"), f"{st['proteins_universe']:,} proteins",
                           f"{st['cells']:,} cells", "present (packed bits), active partners, rooms, complex completeness, estimated log2 amount",
                           tier="derived", extra={"made_by": "cellmap"})
        facts.append(f); lf[f"CELLMAP_{fam}"] = f.ref
        for cid, v in S.items():
            facts.append(Fact(f"CELL-{fam}-{cid}", "cellmap/summary", v, t[:10], t, "derived", src, inputs=(lf[f"CELLMAP_{fam}"],),
                              context={"family": fam, "present": {"BODY": ">= 10% of cells", "LINES": "nTPM >= 1", "CANCER": "TPM >= 1"}[fam]}))
        if fam in ("BODY", "LINES"):
            page[fam] = S
        od = OUT if fam != "CANCER" else os.path.join(HERE, "_cache", "cellmap")     # cancer maps stay local (size)
        os.makedirs(od, exist_ok=True)
        with gzip.open(os.path.join(od, f"cellmap_{fam.lower()}.json.gz"), "wt") as fh:
            json.dump(S, fh, default=float)
    n = cbb.bank().add_many(facts)
    rep = dict(built=time.strftime("%Y-%m-%d"), families=allstats, layers=lf, facts_written=n)
    json.dump(rep, open(os.path.join(OUT, "cellmap_report.json"), "w"), indent=1, default=float); print(json.dumps(rep, indent=1, default=float))


# ------------------------------------------------------------------------------------------------ G7-M1
def run_m1(rng):
    K = knowledge()
    D = gx.npz("DEP_DEPMAP"); E = gx.npz("EXPR_DEPMAP")
    dl, el = D["cols"].astype(str), E["cols"].astype(str); lines = sorted(set(dl) & set(el))
    di = {c: i for i, c in enumerate(dl)}; ei = {c: i for i, c in enumerate(el)}
    Xd = D["X"].astype(np.float32)[:, [di[c] for c in lines]]; Xe = E["X"].astype(np.float32)[:, [ei[c] for c in lines]]
    drow = {}
    for i, g in enumerate(D["rows"].astype(str)):
        drow.setdefault(g, i)
    erow = {g: i for i, g in enumerate(E["rows"].astype(str))}
    MADE = Xe >= 1.0
    pcg = [g for g in K["pc"] if g in erow]; pc_idx = np.array([erow[g] for g in pcg])
    dec_all = pa.deciles(np.nanmean(Xe, 1)); pool = {d: pc_idx[dec_all[pc_idx] == d] for d in range(10)}
    comps = []
    for c, m in K["comp"].items():
        if 3 <= len(m) <= 60:
            mm = [x for x in m if x in drow and x in erow]
            if len(mm) >= 3:
                comps.append((c, mm))
    if TRIAL:
        comps = comps[:150]

    def items(parts_made):
        """parts_made: list per complex of (A rows in Xd, made matrix of the parts) -> d per (complex, part)."""
        out = []
        for (c, mm), Mk in zip(comps, parts_made):
            tot = Mk.sum(0)
            for a, x in enumerate(mm):
                aok = MADE[erow[x]]; others_all = (tot - Mk[a]) == (Mk.shape[0] - 1)
                comp_l = aok & others_all; inc_l = aok & ~others_all
                if comp_l.sum() >= 20 and inc_l.sum() >= 20:
                    e = Xd[drow[x]]
                    out.append(float(np.nanmean(e[comp_l]) - np.nanmean(e[inc_l])))
                else:
                    out.append(np.nan)
        return np.array(out)
    real_M = [MADE[[erow[x] for x in mm]] for _, mm in comps]
    d = items(real_M); ok = np.isfinite(d); dv = d[ok]
    real = float(np.median(dv)) if len(dv) else float("nan")
    from scipy.stats import binomtest
    nz = dv[dv != 0]; p_sign = float(binomtest(int((nz < 0).sum()), int(len(nz)), 0.5, alternative="greater").pvalue) if len(nz) else 1.0
    nulls = []
    for b in range(NMC):
        Ms = []
        for c, mm in comps:
            rows = [erow[x] for x in mm]                # random level-matched stand-ins for every part; part A keeps its own row
            rnd = np.array([pool[dec_all[r_]][rng.integers(0, len(pool[dec_all[r_]]))] for r_ in rows])
            Ms.append(MADE[rnd])
        dn = items_random(comps, Ms, MADE, erow, drow, Xd)
        nulls.append(float(np.nanmedian(dn)))
        if b == 0:
            dn0 = dn
    p_mc = float((1 + sum(x <= real for x in nulls)) / (1 + NMC))
    v0 = dn0[np.isfinite(dn0)]; nz0 = v0[v0 != 0]
    neg_med = float(np.median(v0)); neg_p = float(binomtest(int((nz0 < 0).sum()), int(len(nz0)), 0.5, alternative="greater").pvalue) if len(nz0) else 1.0
    neg_mc = float((1 + sum(x <= neg_med for x in nulls[1:])) / max(1, NMC))
    out = dict(lines=len(lines), complexes=len(comps), items=int(ok.sum()), median_d=real, share_negative=float((nz < 0).mean()) if len(nz) else None,
               p_sign=p_sign, null_median=float(np.median(nulls)), null_5=float(np.percentile(nulls, 5)), p_mc=p_mc,
               passed=bool(real <= -0.02 and p_sign < 0.05 and p_mc < 0.05),
               negative=dict(median_d=neg_med, p_sign=neg_p, p_mc=neg_mc, passed=bool(neg_med <= -0.02 and neg_p < 0.05 and neg_mc < 0.05)),
               planted=dict(median_d=real - 0.05, passed=bool(real - 0.05 <= -0.02 and float((1 + sum(x <= real - 0.05 for x in nulls)) / (1 + NMC)) < 0.05)))
    # examples: complexes whose parts are most needed when complete
    H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e); ex = []; k = 0
    for c, mm in comps:
        dd = d[k:k + len(mm)]; k += len(mm)
        if np.isfinite(dd).sum() >= 2:
            ex.append((float(np.nanmedian(dd)), K["names"].get(c, c), [sym(x) for x in mm][:8]))
    ex.sort(key=lambda x: x[0])
    out["most_needed_when_complete"] = [dict(machine=n, median_d=round(v, 3), parts=p) for v, n, p in ex[:15]]
    out["gate"] = dict(real=out["passed"], controls_ok=bool(not out["negative"]["passed"] and out["planted"]["passed"]))
    out["gate"]["verdict"] = "PASS" if out["gate"]["real"] and out["gate"]["controls_ok"] else "FAIL"
    return out


def items_random(comps, Ms, MADE, erow, drow, Xd):
    """null: part A keeps its own made-row and effect; 'machine complete' judged on random level-matched stand-ins for the other parts."""
    out = []
    for (c, mm), Mr in zip(comps, Ms):
        for a, x in enumerate(mm):
            aok = MADE[erow[x]]; others = np.delete(Mr, a, axis=0); others_all = others.all(0)
            comp_l = aok & others_all; inc_l = aok & ~others_all
            if comp_l.sum() >= 20 and inc_l.sum() >= 20:
                e = Xd[drow[x]]; out.append(float(np.nanmean(e[comp_l]) - np.nanmean(e[inc_l])))
            else:
                out.append(np.nan)
    return np.array(out)


def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def declare():
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            if cid in have:
                print(cid, "already declared"); continue
            f.write(json.dumps(dict(op="declare", id=cid, at=cbb.now(), code_sha=code_sha(), commit=gx._git("rev-parse", "HEAD").stdout.strip(),
                                    claim=__doc__.split(" " + cid)[1].split("NOT:")[0].strip()[:1200], trial=TRIAL), sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G7 already tested")
    t0 = time.time(); rng = np.random.default_rng(SEED)
    m1 = run_m1(rng); print(f"G7-M1 done ({time.time() - t0:.0f}s)", flush=True)
    if TRIAL:
        print("trial controls ok:", m1["gate"]["controls_ok"], "; real outcome not shown"); return
    print("G7-M1", json.dumps(m1, default=float)[:2500])
    os.makedirs(OUT, exist_ok=True); json.dump(m1, open(os.path.join(OUT, "machine_test.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "cellmap:G7", "code_sha": code_sha()}
    cbb.bank().add_many([Fact("CLAIM-G7-M1", "machine_test", m1, t[:10], t, "derived", src, inputs=tuple(sorted(gx.USED)))])
    with open(LEDGER, "a") as f:
        f.write(json.dumps(dict(op="result", id="G7-M1", at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **m1), sort_keys=True, default=float) + "\n")
    L_ = ["=" * 97, "CELLMAP -- every cell down to single proteins; G7-M1 complete machines need their parts", "=" * 97,
          f"  G7-M1: {m1['items']:,} machine parts in {m1['complexes']} machines over {m1['lines']:,} cancer lines: median d {m1['median_d']:.3f} "
          f"(share more needed when complete {m1['share_negative']:.2f}, sign p {m1['p_sign']:.2g}); random stand-ins median {m1['null_median']:.3f}, "
          f"p_MC {m1['p_mc']:.2g} -> {m1['gate']['verdict']}"]
    open(os.path.join(HERE, "RESULTS_cellmap.txt"), "a").write("\n".join(L_) + "\n"); print("\n".join(L_))


# ------------------------------------------------------------------------------------------------ page data
def page(outdir):
    """compact data for the visual page: body cell types + the four lines."""
    os.makedirs(outdir, exist_ok=True); cells = []
    for fam in ("BODY", "LINES"):
        S = json.load(gzip.open(os.path.join(OUT, f"cellmap_{fam.lower()}.json.gz"), "rt"))
        for cid, v in S.items():
            cells.append(dict(id=cid, fam=fam, name=v["name"], n=v["proteins"], ap=v["active_partnerships"], mc=v["machines_complete"],
                              rooms={k: [x["n"], x["top"][:8]] for k, x in v["rooms"].items()}, complete=v["complete"][:12],
                              near=v["nearly_complete"][:8], jobs=[[j[0], j[1], j[2]] for j in v["jobs"][:8]], hubs=v["hubs"][:8],
                              bk=v["backups_covered"], bkx=v["backup_examples"][:8]))
    json.dump(dict(cells=cells, rooms=ROOMS), open(os.path.join(outdir, "cells.json"), "w"), separators=(",", ":"))
    print(len(cells), "cells ->", os.path.join(outdir, "cells.json"), os.path.getsize(os.path.join(outdir, "cells.json")), "bytes")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"pull": pull, "build": build, "declare": declare, "test": test, "page": lambda: page(a[1])}[a[0]]()
