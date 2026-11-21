"""POST-HOC model review (written after cellbank7; nothing here was predeclared; descriptive only).

Asked for by the user: analyse and review the model. Uses only committed artifacts (memory_bank/cell_v6,
outputs/) plus the screen-A file for the measured TEST answers. numpy + h5py only (no torch, no retraining),
so the engine's numbers quoted in the output are the ones already in the ledger.

What it measures
  R1  the ceiling of the representation: Pearson r between each TEST knockdown's measured response and the bank's
      own compressed record of it (the best any prediction made in the 128-component space can score), overall
      and by response strength; and how much of that the first few components give.
  R2  trivial baselines scored exactly like the engine (same TEST genes, same measured answers):
      copy the gene's own record from another cell type; average them; the mean of partners' records; the mean
      of co-essential genes' records; a ridge map from the gene's own other-cell-type records (fitted on TRAIN).
  R3  how concentrated the engine's recall is (stored recalls for 50 and 100 random TEST genes).
  R4  how alike the engine's stated top predictions are across genes (stored prompts).

Run: python3 -P rem/atlas/modelreview_posthoc.py
"""

from __future__ import annotations
import collections
import hashlib
import importlib.util
import json
import os
import re

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
_s = importlib.util.spec_from_file_location("cellbank4", os.path.join(HERE, "cellbank4.py"))
cb4 = importlib.util.module_from_spec(_s); _s.loader.exec_module(cb4)
GWPS = os.path.join(HERE, "_cache", "perturbseq", "K562_gwps_normalized_bulk_01.h5ad")
OUT = os.path.join(HERE, "RESULTS_modelreview.txt")
RULE = "=" * 97
split = lambda k: "TRAIN" if int(hashlib.sha256(k.encode()).hexdigest(), 16) % 2 == 0 else "TEST"


def measured_test():
    """Screen-A TEST answers, built exactly as the notebook builds A_te (NaN on the knocked-down gene)."""
    f = h5py.File(GWPS, "r")
    X = f["X"][()]
    gt = [s.decode() for s in f["obs/gene_transcript"][()]]
    fold, core, ep = f["obs/fold_expr"][()], f["obs/core_control"][()], f["obs/energy_test_p_value"][()]
    cats = [s.decode() for s in f["var/__categories/gene_name"][()]]
    genes = [cats[c] for c in f["var/gene_name"][()]]
    best = {}
    for i, g in enumerate(gt):
        p = g.split("_"); sym = p[1] if len(p) > 1 else g
        if "non-targeting" in g.lower() or core[i]:
            continue
        fv = fold[i] if np.isfinite(fold[i]) else 1.0
        if sym not in best or fv < best[sym][1]:
            best[sym] = (i, fv)
    kos = sorted(best); rows = [best[k][0] for k in kos]
    Xt = X[rows].astype(np.float32); Xt[~np.isfinite(Xt)] = np.nan
    epr = np.array([ep[r] for r in rows])
    gidx = {}
    for j, g in enumerate(genes):
        gidx.setdefault(g, j)
    phen = [i for i, e in enumerate(epr) if np.isfinite(e) and e < 0.05]
    for i in phen:
        if kos[i] in gidx:
            Xt[i, gidx[kos[i]]] = np.nan
    TR = [i for i in phen if split(kos[i]) == "TRAIN"]; TE = [i for i in phen if split(kos[i]) == "TEST"]
    tide = np.nanmean(Xt[TR], 0)
    return [kos[i] for i in TE], Xt[TE] - tide


def pearson_rows(P, A):
    m = np.isfinite(A) & np.isfinite(P)
    n = np.maximum(m.sum(1, keepdims=True), 1)
    Pz, Az = np.where(m, P, 0.0), np.where(m, A, 0.0)
    dp = np.where(m, P - Pz.sum(1, keepdims=True) / n, 0.0); da = np.where(m, A - Az.sum(1, keepdims=True) / n, 0.0)
    return np.nan_to_num((dp * da).sum(1) / np.maximum(np.linalg.norm(dp, axis=1) * np.linalg.norm(da, axis=1), 1e-12))


def main():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    bank = cb4.Bank(R("memory_bank", "cell_v6"))
    B = bank.basis.astype(np.float32)
    te_genes, A_te = measured_test()
    ida = [bank.get(g, "A") for g in te_genes]
    assert None not in ida and all(bank.split[i] == "TEST" for i in ida) and len(te_genes) == 2525
    unit = lambda Z: Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    score = lambda Z, rows: pearson_rows(Z @ B.T, A_te[rows])
    allr = np.arange(len(te_genes))
    ZA = bank.z(ida)

    P_(RULE); P_("POST-HOC MODEL REVIEW (descriptive; nothing here was predeclared)"); P_(RULE)
    # R1 ---------------------------------------------------------------------------------------------------
    ceil = score(ZA, allr)
    norm = np.linalg.norm(np.nan_to_num(A_te), axis=1); q = np.quantile(norm, [0.25, 0.5, 0.75])
    P_("R1 THE CEILING OF THE REPRESENTATION (a perfect prediction inside the 128 components)")
    P_(f"   all {len(ceil):,} TEST knockdowns: median r {np.median(ceil):.3f}   [the engine's best overall figure in the ledger: 0.26]")
    for lab, m in (("weakest quarter", norm <= q[0]), ("2nd quarter", (norm > q[0]) & (norm <= q[1])),
                   ("3rd quarter", (norm > q[1]) & (norm <= q[2])), ("strongest quarter", norm > q[2])):
        P_(f"      {lab:18s} of responses: ceiling {np.median(ceil[m]):.3f}")
    P_("   how much of it the first k components give (a perfect prediction of only those):")
    P_("      " + "   ".join(f"k={k}: {np.median(score(np.pad(ZA[:, :k], ((0, 0), (0, 128 - k))), allr)):.3f}" for k in (1, 2, 3, 5, 10, 20, 50, 128)))
    U = ZA[np.random.default_rng(0).choice(len(ZA), 800, replace=False)] @ B.T
    Uc = unit(U - U.mean(1, keepdims=True)); C = Uc @ Uc.T; off = C[~np.eye(len(C), dtype=bool)]
    P_(f"   two different genes' measured responses: mean r {off.mean():+.3f}, mean |r| {np.abs(off).mean():.3f} (800 random TEST genes)")

    # R2 ---------------------------------------------------------------------------------------------------
    has = {s: np.array([bank.get(g, s) is not None for g in te_genes]) for s in "BCDE"}
    own = {s: np.zeros((len(te_genes), 128), dtype=np.float32) for s in "BCDE"}
    for s in "BCDE":
        ids = [bank.get(g, s) for g in te_genes]
        own[s][has[s]] = unit(bank.z([i for i in ids if i is not None]))
    k562 = {}
    for g, d in bank.by_gene.items():
        if "A" in d and bank.split[d["A"]] == "TRAIN":
            k562[g] = d["A"]
        elif "B" in d:
            k562[g] = d["B"]
    part, coes = np.zeros((len(te_genes), 128), np.float32), np.zeros((len(te_genes), 128), np.float32)
    for t, g in enumerate(te_genes):
        info = bank.genes[g]
        ids = [k562[p] for p in info["partners"] if p in k562 and p != g]
        if ids:
            part[t] = bank.z(ids).mean(0)
        co = [(k562[p], r) for p, r in info["coessential"] if p in k562 and p != g][:5]
        w = np.array([max(r, 0.0) for _, r in co])
        if co and w.sum() > 0:
            coes[t] = (bank.z([i for i, _ in co]) * w[:, None]).sum(0) / w.sum()
    avg = own["C"] + own["D"] + own["E"]

    # ridge from the gene's own other-cell-type records, fitted on TRAIN genes only
    tr_genes = [g for g, d in bank.by_gene.items() if "A" in d and bank.split[d["A"]] == "TRAIN" and any(s in d for s in "CDE")]

    def feats(gs):
        X = np.zeros((len(gs), 3 * 129), np.float32)
        for i, g in enumerate(gs):
            for b, s in enumerate("CDE"):
                j = bank.get(g, s)
                if j is not None:
                    X[i, b * 129:b * 129 + 128] = unit(bank.z([j]))[0]; X[i, b * 129 + 128] = 1.0
        return X
    Xtr, Ytr = feats(tr_genes), unit(bank.z([bank.get(g, "A") for g in tr_genes]))

    def ridge(Xa, Ya, Xb, lam):
        xm, ym = Xa.mean(0), Ya.mean(0)
        return (Xb - xm) @ np.linalg.solve((Xa - xm).T @ (Xa - xm) + lam * np.eye(Xa.shape[1]), (Xa - xm).T @ (Ya - ym)) + ym
    folds = np.array_split(np.random.default_rng(2).permutation(len(tr_genes)), 5); cv = {}
    for lam in (0.1, 1.0, 10.0, 100.0):
        rs = []
        for f in folds:
            k = np.ones(len(tr_genes), bool); k[f] = False
            P = ridge(Xtr[k], Ytr[k], Xtr[f], lam) @ B.T
            rs.append(np.median(pearson_rows(P, Ytr[f] @ B.T)))
        cv[lam] = float(np.mean(rs))
    lam = max(cv, key=cv.get)
    Zr = ridge(Xtr, Ytr, feats(te_genes), lam)
    kc = Xtr[:, 128] > 0                               # the same map from the RPE1 record alone (what E2 was given)
    Zc = ridge(Xtr[kc][:, :129], Ytr[kc], feats(te_genes)[:, :129], lam)

    anyo = has["C"] | has["D"] | has["E"]; all3 = has["C"] & has["D"] & has["E"]
    P_(); P_("R2 TRIVIAL BASELINES, SCORED LIKE THE ENGINE (median r on the same TEST genes and measured answers)")
    P_(f"   ridge map: {len(tr_genes):,} TRAIN genes, lambda {lam:g} chosen by 5-fold CV inside TRAIN")
    rowsets = [("has an RPE1 record", has["C"], "E2 0.350 (cellbank4 post-hoc)"),
               ("has a HepG2 or Jurkat record", has["D"] | has["E"], "E3 0.377, E2 0.355 (cellbank5 G5)"),
               ("has all three (RPE1, HepG2, Jurkat)", all3, "E3 0.430, E2 0.419 (cellbank5)"),
               ("has none (unmeasured elsewhere)", ~anyo, "E3 0.217 (cellbank5)"),
               ("all TEST genes", np.ones(len(te_genes), bool), "E3 0.255-0.259 (three identical reruns)")]
    res = {}
    for lab, m, eng in rowsets:
        rows = np.nonzero(m)[0]
        med = lambda Z, mm: float(np.median(score(Z[mm], np.nonzero(mm)[0]))) if mm.any() else None   # only genes that have it
        fmt = lambda v: "-" if v is None else f"{v:.3f}"
        line = dict(n=int(m.sum()), ceiling=float(np.median(ceil[m])), copy_RPE1=med(own["C"], m & has["C"]),
                    copy_HepG2=med(own["D"], m & has["D"]), copy_Jurkat=med(own["E"], m & has["E"]),
                    average_of_own=med(avg, m & anyo), ridge_from_own=med(Zr, m),
                    partner_mean=med(part, m & (np.abs(part).sum(1) > 0)), coessential_mean=med(coes, m & (np.abs(coes).sum(1) > 0)),
                    replicate_B=med(own["B"], m & has["B"]), ridge_from_RPE1_only=med(Zc, m & has["C"]))
        res[lab] = line
        P_(f"   {lab} (n {line['n']:,}); engine in the ledger: {eng}")
        P_(f"      ceiling {line['ceiling']:.3f} | copy RPE1 {fmt(line['copy_RPE1'])} | copy HepG2 {fmt(line['copy_HepG2'])} | copy Jurkat {fmt(line['copy_Jurkat'])} | "
           f"average of own records {fmt(line['average_of_own'])} | ridge from own records (a constant guess where there are none) {fmt(line['ridge_from_own'])}")
        P_(f"      partners' mean {fmt(line['partner_mean'])} | co-essential mean {fmt(line['coessential_mean'])}"
           f" | the gene's K562-essential replicate, bank record {fmt(line['replicate_B'])}"
           f" | ridge from the RPE1 record alone {fmt(line['ridge_from_RPE1_only'])}")

    # R3 ---------------------------------------------------------------------------------------------------
    P_(); P_("R3 HOW CONCENTRATED THE ENGINE'S RECALL IS (its 3 top recalled experiments per gene, as stored)")
    J = json.load(open(R("outputs", "cellbank4", "claude_jobs.json")))["jobs"]
    rec = collections.Counter(s[1].split(" knockdown")[0] + (" [RPE1]" if "RPE1" in s[1] else " [K562-ess]" if "essential" in s[1] else "")
                              for j in J for s in j["sources"] if "recalled by the engine" in s[1])
    tot = sum(rec.values())
    P_(f"   full-bank engine (cellbank4, {len(J)} random TEST genes): {tot} recalls use {len(rec)} distinct records of ~8,500; "
       f"top 4 = {sum(n for _, n in rec.most_common(4)) / tot:.0%}: " + ", ".join(f"{g} x{n}" for g, n in rec.most_common(6)))
    M = json.load(open(R("outputs", "claudeloop", "menus.json")))["jobs"]
    rec2 = collections.Counter(x for j in M for x in j["memory"][:3])
    tot2 = sum(rec2.values())
    P_(f"   screen-A engine (claudeloop, {len(M)} random TEST genes; first 3 listed memories): {tot2} use {len(rec2)} distinct of 2,527; "
       f"top 4 = {sum(n for _, n in rec2.most_common(4)) / tot2:.0%}: " + ", ".join(f"{g} x{n}" for g, n in rec2.most_common(6)))

    # R4 ---------------------------------------------------------------------------------------------------
    P_(); P_("R4 HOW ALIKE THE ENGINE'S TOP PREDICTIONS ARE ACROSS GENES (stored prompts)")
    s0 = re.findall(r"^S0: [^:]*: raises (.*?)\. Lowers (.*?)\. Most raised", open(R("outputs", "cellbank4", "claude_prompts.txt")).read(), re.M)
    up = collections.Counter(x for u, _ in s0 for x in u.split("; ")); dn = collections.Counter(x for _, d in s0 for x in d.split("; "))
    mito = sum(("mitochondri" in u) and ("MITF" in d) for u, d in s0)
    P_(f"   {len(s0)} genes: {mito} are predicted as 'mitochondrial RNA processing up, lysosome (MITF) genes down'; "
       f"{len({(u.split('; ')[0], d.split('; ')[0]) for u, d in s0})} distinct (top rise, top fall) pairs")
    P_("   most common predicted rises: " + "; ".join(f"{k[:45]} x{n}" for k, n in up.most_common(3)))
    P_("   most common predicted falls: " + "; ".join(f"{k[:45]} x{n}" for k, n in dn.most_common(3)))
    open(OUT, "w").write("\n".join(out) + "\n")
    json.dump(dict(ceiling_median=float(np.median(ceil)), ridge_lambda=lam, ridge_cv=cv, baselines=res,
                   recall_cellbank4=rec.most_common(10), recall_claudeloop=rec2.most_common(10)),
              open(R("outputs", "modelreview.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
