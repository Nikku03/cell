"""CELLMAP2 -- the cell maps rebuilt with the fixes that passed their tests (cellfix G8), plus what the remaining
limits allow, labelled by how well each part is supported.

Asked by the user: "first add public available info, then the yes things, then complete the list". Descriptive (no
new claim); every addition carries the test it rests on:
  DEEP family -- 155 HPA single cell types from pseudobulk RNA (made = nCPM >= 1). Rests on G8-D1 PASS (genes found
     only by the deeper RNA have their protein stained 6x more often than genes not made). Where the cell type name
     matches an HPA immunohistochemistry cell type, each made protein is flagged stained / not stained / not tested.
  CANCERPROT family -- 947 cancer lines mapped by MEASURED protein (ProCan: made = protein detected). Fixes "RNA not
     protein" where protein was measured. For cancer lines with RNA only, the per-gene protein/RNA offsets (G8-P1 PASS:
     r 0.46 -> 0.84) give an estimated protein level; they are NOT applied to body tissues (G8-P2 FAIL).
  ON / OFF -- transcription factor activity per DEEP cell type from CollecTRI targets, reported ONLY for TFs whose
     target-based readout was confirmed by knocking them down in K562 (per-TF AUROC >= 0.9; G8-A1 as a whole FAILED,
     0.54, so the readout is trusted per TF, not in general). PROGENy pathway activities are added, marked "not tested
     here".
  UNKNOWN PROTEINS -- jobs (Reactome) and likely partners for protein-coding genes with no pathway or complex, from
     neighbours in DepMap effects, ProCan protein, HPA RNA and K562 knockdowns (G8-U1 PASS, median AUROC 0.77);
     written to the bank as predictions.
  MACHINES -- per complex, how strongly its parts' protein levels move together across cancer lines (G8-M2: 0.23 vs
     0.02 for random genes; but no better than RNA, so marked as co-regulation, not proof of assembly).
  PARTNERS -- BioPlex says which interactions were seen in HEK293T, HCT116 or both (G8-I1: line-specific pairs follow
     which partner is made, direction right, just under the declared bar).
  TIMING -- proteins that vary with the cell cycle (HPA) counted per cell and flagged; no per-cell-type timing exists.

Run: python3 -I rem/atlas/cellmap2.py build | page OUTDIR
"""

from __future__ import annotations
import collections
import gzip
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))


def _load(name):
    argv = sys.argv; sys.argv = [sys.argv[0]]
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


cm = _load("cellmap"); cf = _load("cellfix")
gx = cm.gx; cbb, Fact = gx.cbb, gx.Fact
OUT = R("outputs", "cellmap")


def tf_readout():
    """per-TF AUROC of the target-based readout in the K562 genome-wide screen (same computation as G8-A1)."""
    T = gx.tss(); G = gx.load("K562GW", T); rows = list(G["rows"]); ri = {g: i for i, g in enumerate(rows)}
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.nan_to_num(G["lfc"] / G["se"]).astype(np.float32)
    reg = cf.regulons(set(rows)); tfs, M = cf.reg_matrix(reg, ri)
    kd = [t for t in tfs if t in G["ci"]]; kc = np.array([G["ci"][t] for t in kd]); ti = {t: i for i, t in enumerate(tfs)}
    A = np.asarray(M.dot(Z[:, kc]))
    return {t: float((A[:, j] > A[ti[t], j]).sum() / (A.shape[0] - 1)) for j, t in enumerate(kd)}


def activities(X, genes, auc):
    """TF (validated only) and PROGENy activities per column of a genes x cells log-expression matrix (z across cells)."""
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True); Zc = np.where(sd > 0, (X - mu) / np.where(sd > 0, sd, 1), 0).astype(np.float32)
    gi = {g: i for i, g in enumerate(genes)}
    reg = cf.regulons(set(genes)); keep = {t: v for t, v in reg.items() if auc.get(t, 0) >= 0.9}
    tfs, M = cf.reg_matrix(keep, gi) if keep else ([], None)
    TA = np.asarray(M.dot(Zc)) if keep else np.zeros((0, X.shape[1]))
    pg = gx.npz("PATH_PROGENY"); paths = sorted(set(pg["pathway"].astype(str))); PA = np.zeros((len(paths), X.shape[1]), np.float32)
    for k, pth in enumerate(paths):
        m = pg["pathway"].astype(str) == pth; g = pg["gene"].astype(str)[m]; w = pg["weight"][m]; p = pg["p"][m]
        o = np.argsort(p)[:100]; idx = [(gi[g[i]], w[i]) for i in o if g[i] in gi]
        if idx:
            ii = np.array([a for a, _ in idx]); ww = np.array([b for _, b in idx], np.float32); PA[k] = (ww[:, None] * Zc[ii]).sum(0) / np.abs(ww).sum()
    return tfs, TA, paths, PA


def predicted_roles(K):
    """jobs and likely partners for protein-coding genes with no Reactome pathway and no complex (G8-U1 method)."""
    ev = cf.evidences(); have = collections.Counter()
    for _, (rows, _, _) in ev.items():
        for g in rows:
            have[g] += 1
    pw = {k: (v[0], v[1]) for k, v in K["pw"].items() if 10 <= len(v[1]) <= 300}
    annotated = {g for _, m in pw.values() for g in m} | {g for m in K["comp"].values() for g in m}
    ann = sorted(g for g in annotated if have[g] >= 2); unk = sorted(g for g in K["pc"] if g not in annotated and have[g] >= 2)
    sim, _ = cf.combined_sim(ev, unk, ann); ai = {g: i for i, g in enumerate(ann)}; H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e)
    names, scores = [], []
    for k, (nm, m) in pw.items():
        idx = np.array([ai[g] for g in m if g in ai])
        if len(idx) >= 5:
            names.append(nm); scores.append(cf.pathway_scores(sim, idx))
    S = np.stack(scores, 1) if scores else np.zeros((len(unk), 0)); out = {}
    for i, g in enumerate(unk):
        o = np.argsort(-S[i])[:3]; nb = np.argsort(-np.nan_to_num(sim[i], nan=-9))[:10]
        out[g] = dict(symbol=sym(g), jobs=[[names[j], round(float(S[i, j]), 3)] for j in o],
                      neighbours=[[sym(ann[j]), round(float(np.tanh(sim[i, j])), 3)] for j in nb if np.isfinite(sim[i, j])])
    return out


def build():
    t0 = time.time(); K = cm.knowledge(); facts = []; t = cbb.now(); src = {"name": "cellmap2", "code_sha": cm.code_sha()}
    fx = json.load(open(R("outputs", "cellfix", "fix_tests.json")))
    verdicts = {k: v["gate"]["verdict"] for k, v in fx.items()}
    print("G8 verdicts:", verdicts, flush=True)
    # DEEP family (HPA single cell types)
    S = gx.npz("EXPR_HPA_SCT"); X = S["X"].astype(np.float32); genes = S["rows"].astype(str); cells = S["cols"].astype(str)
    Sm, st = cm.build_family("DEEP", genes, cells, cells, None, X >= 1.0, X, K); print("DEEP", st, f"({time.time() - t0:.0f}s)", flush=True)
    _, _, _, by, matched, norm = cf.hpa_matched(); ccd = json.load(open(gx.L("TIMING_HPA_CCD")))
    auc = tf_readout(); print(f"TF readout: {sum(v >= 0.9 for v in auc.values())} of {len(auc)} TFs confirmed (AUROC >= 0.9) ({time.time() - t0:.0f}s)", flush=True)
    L = np.log2(X + 1); tfs, TA, paths, PA = activities(L, genes, auc)
    H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e); gi = {g: i for i, g in enumerate(genes)}
    for j, c in enumerate(cells):
        v = Sm.get(str(c))
        if v is None:
            continue
        made = [genes[i] for i in np.nonzero(X[:, j] >= 1.0)[0]]
        lv = by.get(norm(c))
        if lv:
            tested = [g for g in made if g in lv]; v["protein_stain"] = dict(tested=len(tested), stained=int(sum(lv[g] >= 1 for g in tested)))
        v["cell_cycle_varying"] = int(sum(1 for g in made if ccd.get(g, {}).get("protein") == "Yes"))
        o = np.argsort(-TA[:, j])[:8] if len(tfs) else []
        v["tf_on"] = [[sym(tfs[i]), round(float(TA[i, j]), 2)] for i in o if TA[i, j] > 0]
        v["pathways"] = [[paths[k], round(float(PA[k, j]), 2)] for k in np.argsort(-PA[:, j])[:4]]
        facts.append(Fact(f"CELL-DEEP-{c}", "cellmap/summary", v, t[:10], t, "derived", src, context={"family": "DEEP", "present": "nCPM >= 1 (HPA pseudobulk)",
                          "tests": {"deep_rna": verdicts.get("G8-D1"), "tf_readout": "per TF, AUROC >= 0.9 only", "progeny": "not tested here"}}))
    with gzip.open(os.path.join(OUT, "cellmap_deep.json.gz"), "wt") as fh:
        json.dump(Sm, fh, default=float)
    # CANCERPROT family (measured protein)
    P = gx.npz("PROT_PROCAN"); Xp = P["X"].astype(np.float32); pg = P["rows"].astype(str); pl = P["cols"].astype(str)
    import pandas as pd
    m = pd.read_csv(os.path.join(HERE, "_cache", "paralog", "depmap", "raw", "Model.csv"), dtype=str).set_index("ModelID")
    nm = np.array([str(m.CellLineName.get(i, i)) for i in pl]); lin = [str(m.OncotreeLineage.get(i, "")) for i in pl]
    Sp, st2 = cm.build_family("CANCERPROT", pg, pl, nm, lin, np.isfinite(Xp), np.exp2(np.nan_to_num(Xp, nan=0.0)), K)
    print("CANCERPROT", st2, f"({time.time() - t0:.0f}s)", flush=True)
    for c, v in Sp.items():
        facts.append(Fact(f"CELL-CANCERPROT-{c}", "cellmap/summary", v, t[:10], t, "derived", src, context={"family": "CANCERPROT", "present": "protein detected (ProCan DIA-MS)"}))
    with gzip.open(os.path.join(HERE, "_cache", "cellmap", "cellmap_cancerprot.json.gz"), "wt") as fh:
        json.dump(Sp, fh, default=float)
    # protein estimate offsets (cancer lines only)
    off = np.load(os.path.join(cbb.LAY, "protein_offset_procan.npz"))
    for g, o in zip(off["genes"].astype(str), off["offset"]):
        if np.isfinite(o):
            facts.append(Fact(g, "protein/per_rna_offset_cancer_lines", round(float(o), 3), t[:10], t, "derived", src,
                              context={"use": "estimated protein (log2) = RNA log2(TPM+1) + offset; tested in cancer lines (G8-P1 PASS), not in tissues (G8-P2 FAIL)"}))
    # unknown proteins
    pr = predicted_roles(K); print(f"predicted roles for {len(pr):,} unannotated proteins ({time.time() - t0:.0f}s)", flush=True)
    for g, v in pr.items():
        facts.append(Fact(g, "predict/job_from_neighbours", v, t[:10], t, "derived", src,
                          context={"kind": "prediction", "method": "G8-U1 (median AUROC 0.77 on held-out genes)"}))
    with gzip.open(os.path.join(OUT, "predicted_roles.json.gz"), "wt") as fh:
        json.dump(pr, fh)
    # machines: protein co-movement per complex; partners by line
    Xr, rr, cr = cf.matrix("EXPR_DEPMAP"); Pm, Rn, g2, _ = cf.align2(*cf.matrix("PROT_PROCAN"), Xr, rr, cr); gi2 = {g: i for i, g in enumerate(g2)}
    mu = np.nanmean(Pm, 1, keepdims=True); sd = np.nanstd(Pm, 1, keepdims=True); ZP = np.nan_to_num((Pm - mu) / np.where(sd > 0, sd, np.nan)); FP = np.isfinite(Pm)
    co = {}
    for c, mm in cf.complexes(2, 60):
        idx = np.array([gi2[x] for x in mm if x in gi2])
        if len(idx) >= 2:
            Ff = FP[idx].astype(np.float32); Zs = ZP[idx] * Ff; num = Zs @ Zs.T; cnt = Ff @ Ff.T; iu = np.triu_indices(len(idx), 1); ok = cnt[iu] >= 30
            if ok.any():
                co[c] = round(float(np.median(num[iu][ok] / cnt[iu][ok])), 3)
    for c, v in co.items():
        facts.append(Fact(f"COMPLEX-{c}", "machine/protein_comovement", v, t[:10], t, "derived", src,
                          context={"name": K["names"].get(c, c), "meaning": "median correlation of parts' protein levels across ~700 cancer lines (random genes ~0.02)"}))
    B = gx.npz("PPI_BIOPLEX"); seen = collections.defaultdict(set)
    for a, b, l in zip(B["a"].astype(str), B["b"].astype(str), B["line"].astype(str)):
        seen[frozenset((a, b))].add(l)
    lines_by_gene = collections.defaultdict(lambda: collections.Counter())
    for p, ls in seen.items():
        for x in p:
            lines_by_gene[x]["both" if len(ls) == 2 else next(iter(ls))] += 1
    for g, cnt in lines_by_gene.items():
        facts.append(Fact(g, "partners/bioplex_by_cell_line", dict(cnt), t[:10], t, "derived", src, context={"note": "local-only source; counts only"}))
    cbb.SRC.setdefault("procan2022", ("Goncalves et al. 2022 Cancer Cell: ProCan-DepMapSanger (figshare 19345397)", "https://doi.org/10.6084/m9.figshare.19345397", "CC BY 4.0"))
    for name, srck, fn, rows_, cols_ in (("CELLMAP_DEEP", "hpa", "cellmap_deep.npz", "proteins", "155 HPA single cell types"),
                                         ("CELLMAP_CANCERPROT", "procan2022", "cellmap_cancerprot.npz", "proteins", "947 cancer lines (protein detected)"),
                                         ("PROTEIN_OFFSET_PROCAN", "procan2022", "protein_offset_procan.npz", "genes", "offset (log2 protein - log2 RNA), DEV lines")):
        facts.append(cbb.layer_fact(name, 1, srck, os.path.join(cbb.LAY, fn), rows_, cols_, "see cellmap2.py", tier="derived", extra={"made_by": "cellmap2 / cellfix"}))
    n = cbb.bank().add_many(facts)
    rep = dict(built=time.strftime("%Y-%m-%d"), deep=st, cancerprot=st2, tf_confirmed=sum(v >= 0.9 for v in auc.values()), tf_tested=len(auc),
               predicted_roles=len(pr), complexes_with_comovement=len(co), facts_written=n, g8_verdicts=verdicts)
    json.dump(rep, open(os.path.join(OUT, "cellmap2_report.json"), "w"), indent=1, default=float); print(json.dumps(rep, indent=1, default=float))


def page(outdir):
    """add the DEEP family to the page data written by cellmap.page."""
    p = os.path.join(outdir, "cells.json"); d = json.load(open(p)); d["cells"] = [c for c in d["cells"] if c["fam"] != "DEEP"]
    S = json.load(gzip.open(os.path.join(OUT, "cellmap_deep.json.gz"), "rt"))
    for cid, v in S.items():
        d["cells"].append(dict(id="HPA-" + cid, fam="DEEP", name=v["name"], n=v["proteins"], ap=v["active_partnerships"], mc=v["machines_complete"],
                               rooms={k: [x["n"], x["top"][:8]] for k, x in v["rooms"].items()}, complete=v["complete"][:12], near=v["nearly_complete"][:8],
                               jobs=[[j[0], j[1], j[2]] for j in v["jobs"][:6]], djobs=[[j[0], j[1], j[2]] for j in v["distinctive_jobs"][:8]],
                               dist=v["distinctive_proteins"][:15], dn=v["distinctive_n"], hubs=v["hubs"][:8], bk=v["backups_covered"], bkx=v["backup_examples"][:8],
                               stain=v.get("protein_stain"), ccv=v.get("cell_cycle_varying"), tf=v.get("tf_on", [])[:8], pth=v.get("pathways", [])))
    json.dump(d, open(p, "w"), separators=(",", ":")); print(len(d["cells"]), "cells;", os.path.getsize(p), "bytes")


if __name__ == "__main__":
    a = sys.argv[1:]
    {"build": build, "page": lambda: page(a[1])}[a[0]]()
