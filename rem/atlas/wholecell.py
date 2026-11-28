"""The whole cell, not a bunch of proteins: can the memory bank predict what happens to the WHOLE
cell when any gene is knocked down -- every process at once, and whether the cell survives?

WHY THE FRAME CHANGES. rawchip / rawchip2 asked "does factor X directly control gene Y" and found
that steady-state knockdowns cannot answer it: the cell rewires, and responses are cell-wide. So
ask the cell-wide question directly. A knockdown is a perturbation of the whole system; its
transcriptome response (8,248 genes) is a snapshot of the whole cell; Reactome's catalogue of human
processes (CC0) turns each snapshot into the activity of every process at once.

DATA. K562 genome-wide CRISPRi Perturb-seq (Replogle 2022, CC BY 4.0; values scaled as in
celldiscover: z-equivalent = value x 6.85, mover |value| >= 3/6.85, non-finite = missing); DepMap 24Q4
K562 gene effect (CC BY 4.0) as the cell's survival; Reactome human pathways (CC0) with 15-300
measured genes as PROCESSES; the bank's facts (cellbank2/celldiscover v3) for gene-level reasoning.
Knockdowns WITH A PHENOTYPE: energy-test p < 0.05. TRAIN / TEST: the SHA-256 parity split used by
cellbank2 and celldiscover. The TIDE: the mean response of all TRAIN phenotype knockdowns -- what
"any knockdown" does to the cell; every model must beat it.

=================================================================================================
GATES, PREDECLARED
=================================================================================================
P0 POSITIVE CONTROL, BLOCKING FOR THE COUPLING MAP. Knocking down cholesterol-biosynthesis genes must
   RAISE the rest of the cholesterol-biosynthesis process (SREBP2 feedback; Reactome R-HSA-191273):
   z > 3. If the map cannot see the best-known whole-cell feedback, it is not reported.

W  PREDICT THE WHOLE CELL for every TEST knockdown with a phenotype; score = Pearson r between the
   predicted and measured response over all measured genes (the knocked-down gene excluded):
     W0 TIDE                the average knockdown response (TRAIN)
     W1 GENE NETWORK        the bank's v3 engine (literature + OmniPath + TRAIN-measured facts):
                            signed, hop-decayed confidence at reached genes, 0 elsewhere
     W2 WHOLE-CELL PROCESS  mean TRAIN response of knockdowns of genes sharing >= 1 process with
                            the target gene (weighted by processes shared); no shared process -> tide
     W2n PROCESS NULL       W2 with every gene's process memberships shuffled (sizes kept)
     W3 PROCESS + NETWORK   equal-weight sum of z-scored W2 and W1
   Paired per-knockdown sign tests, two-sided p < 0.05:
     V1 W2 vs W0   does whole-cell process knowledge beat the generic tide?
     V2 W2 vs W2n  is it the biology of the processes, not the averaging?
     V3 W2 vs W1   process-level vs gene-level reasoning        } on the first 500 TEST knockdowns
     V4 W3 vs W2   does gene-level reasoning add anything?       } by hash (W1 is slow)
F  THE CELL'S BOTTOM LINE (DepMap K562 survival):
     F1 do knockdowns that disturb more of the cell kill it more? Spearman(movers, -gene effect) > 0,
        p < 0.05.
     F2 can process knowledge predict which TEST genes the cell cannot live without (gene effect
        <= -0.5)? Score = weighted mean gene effect of TRAIN genes sharing processes; AUROC must beat
        the 95th percentile of 200 membership-shuffled nulls AND exceed 0.60.
C  THE WHOLE-CELL COUPLING MAP. For process A (>= 8 phenotype knockdowns, all data) and process B
   (gene overlap with A < 20%): effect of knocking down A on B, relative to the tide, as
   z = mean / (sd_B / sqrt(n_A)). A coupling is REPLICATED if |z| >= 5 overall and the same sign with
   |z| >= 2 in each of two random halves of A's knockdowns. Null: the whole procedure with knockdown
   -> process memberships shuffled; estimated FDR = null replicated count / real. Couplings are
   summarised by Reactome top-level system.
X  WHAT THIS IS NOT. One cell line (K562, p53-null leukaemia), CRISPRi, steady state; the transcriptome
   is not the whole cell (no metabolite, protein or flux readout here); Reactome processes overlap.
"""

from __future__ import annotations
import collections
import gzip
import hashlib
import importlib.util
import json
import math
import os
import time
import zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
CA = os.path.join(HERE, "_cache")
RE = os.path.join(CA, "reactome")
OUT = os.path.join(HERE, "RESULTS_wholecell.txt")
ART = R("outputs", "wholecell.json")
BANK = R("memory_bank", "cell_v3")
RULE = "=" * 97
SCALE = 6.85
_s = importlib.util.spec_from_file_location("celldiscover", os.path.join(HERE, "celldiscover.py"))
cd = importlib.util.module_from_spec(_s); _s.loader.exec_module(cd)
cb, dc = cd.cb, cd.dc


def reactome(genes_measured):
    names, parent = {}, collections.defaultdict(set)
    for line in open(os.path.join(RE, "ReactomePathways.txt")):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[2] == "Homo sapiens":
            names[p[0]] = p[1]
    for line in open(os.path.join(RE, "ReactomePathwaysRelation.txt")):
        a, b = line.rstrip("\n").split("\t")[:2]
        if a in names and b in names:
            parent[b].add(a)

    def tops(pid, seen=None):
        seen = seen or set()
        if not parent.get(pid):
            return {pid}
        out = set()
        for q in parent[pid]:
            if q not in seen:
                out |= tops(q, seen | {q})
        return out
    z = zipfile.ZipFile(os.path.join(RE, "ReactomePathways.gmt.zip"))
    mods = {}
    for line in z.read(z.namelist()[0]).decode().splitlines():
        p = line.split("\t")
        pid = p[1]
        if pid not in names:
            continue
        g = {x for x in p[2:] if x in genes_measured}
        if 15 <= len(g) <= 300:
            mods[pid] = g
    top = {pid: sorted(names[t] for t in tops(pid)) for pid in mods}
    return mods, names, top


def pearson_rows(P, A):
    """Row-wise Pearson r between prediction P and actual A, ignoring NaN in A."""
    out = np.full(len(A), np.nan)
    for i in range(len(A)):
        m = np.isfinite(A[i]) & np.isfinite(P[i])
        if m.sum() > 10 and np.std(P[i][m]) > 0 and np.std(A[i][m]) > 0:
            out[i] = np.corrcoef(P[i][m], A[i][m])[0, 1]
    return out


def sign_test(d):
    d = [x for x in d if np.isfinite(x) and x != 0]
    n = len(d); w = sum(x > 0 for x in d)
    k = min(w, n - w)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n) if n else 1.0
    return w, n - w, p


def main():
    from scipy.stats import spearmanr, rankdata
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    rng = np.random.default_rng(20261009)
    X, kos, genes, ep, fold = cd.load_gwps()
    gidx = {g: j for j, g in enumerate(genes)}; kidx = {k: i for i, k in enumerate(kos)}
    zmov = 3.0 / SCALE
    mods, rnames, rtop = reactome(set(genes))
    mid = sorted(mods); M = len(mid)
    Mm = np.zeros((len(genes), M), dtype=np.float32)
    for c, pid in enumerate(mid):
        for g in mods[pid]:
            Mm[gidx[g], c] = 1.0
    gene_mods = collections.defaultdict(set)
    for c, pid in enumerate(mid):
        for g in mods[pid]:
            gene_mods[g].add(c)
    split = lambda k: "TRAIN" if int(hashlib.sha256(k.encode()).hexdigest(), 16) % 2 == 0 else "TEST"
    phen = [k for k, e in zip(kos, ep) if np.isfinite(e) and e < 0.05]
    train = [k for k in phen if split(k) == "TRAIN"]; test = [k for k in phen if split(k) == "TEST"]
    P_(RULE); P_("THE WHOLE CELL: PREDICTING EVERY PROCESS AT ONCE, AND WHETHER THE CELL SURVIVES"); P_(RULE)
    P_(f"  knockdowns {len(kos):,} ({len(phen):,} with a phenotype: TRAIN {len(train):,}, TEST {len(test):,}); "
       f"measured genes {len(genes):,}; Reactome processes (15-300 measured genes) {M:,}")
    Xf = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    fin = np.isfinite(X).astype(np.float32)
    S = (Xf @ Mm) / np.maximum(fin @ Mm, 1)                     # process scores, knockdown x process
    tr_i = np.array([kidx[k] for k in train]); te_i = np.array([kidx[k] for k in test])
    tide = np.nanmean(X[tr_i], 0)
    ph_i = np.array([kidx[k] for k in phen])
    tideS_all = S[ph_i].mean(0); sdS = S[ph_i].std(0)

    # ---------------------------------------------------------------- P0 positive control -------
    P_("\n" + RULE); P_("P0  POSITIVE CONTROL: CHOLESTEROL-SYNTHESIS KNOCKDOWNS RAISE CHOLESTEROL SYNTHESIS (SREBP2 FEEDBACK)"); P_(RULE)
    chol = "R-HSA-191273"
    if chol in mods:
        c = mid.index(chol)
        members = [k for k in phen if k in mods[chol]]
        vals = []
        for k in members:
            gs = [gidx[g] for g in mods[chol] if g != k and np.isfinite(X[kidx[k], gidx[g]])]
            vals.append(np.mean(X[kidx[k], gs]) - tideS_all[c])
        zc = np.mean(vals) / (sdS[c] / math.sqrt(len(vals))) if vals else float("nan")
        p0 = bool(len(vals) >= 3 and zc > 3)
        P_(f"  {len(vals)} cholesterol-synthesis knockdowns: mean shift of the rest of the process {np.mean(vals) * SCALE:+.2f} "
           f"(z-equiv), z {zc:+.1f} -> {'PASS' if p0 else 'FAIL'}")
    else:
        p0 = False
        P_("  cholesterol biosynthesis not among the processes -> FAIL")
    res = {"P0": p0}

    # ---------------------------------------------------------------- W predictions --------------
    P_("\n" + RULE); P_("W  PREDICT THE WHOLE CELL FOR UNSEEN KNOCKDOWNS"); P_(RULE)
    tr_set = set(train)
    mod_tr = collections.defaultdict(list)
    for k in train:
        for c in gene_mods.get(k, ()):
            mod_tr[c].append(kidx[k])
    Pm = np.zeros((M, len(genes)), dtype=np.float32); has = np.zeros(M, bool)
    for c, rows in mod_tr.items():
        Pm[c] = np.nanmean(X[rows], 0); has[c] = True

    def process_pred(k, gm):
        cs = [c for c in gm.get(k, ()) if has_[c]]
        if not cs:
            return tide, False
        return np.mean(Pm_[cs], 0), True
    actual = X[te_i].copy()
    for r, k in enumerate(test):
        if k in gidx:
            actual[r, gidx[k]] = np.nan
    W0 = np.tile(tide, (len(test), 1))
    has_, Pm_ = has, Pm
    W2 = np.zeros_like(W0); cov = 0
    for r, k in enumerate(test):
        W2[r], ok = process_pred(k, gene_mods); cov += ok
    # null: shuffled memberships (sizes kept) -- rebuild module profiles from shuffled membership
    allg = sorted(gene_mods)
    perm = rng.permutation(len(allg)); shuf = {allg[i]: gene_mods[allg[perm[i]]] for i in range(len(allg))}
    mod_tr_n = collections.defaultdict(list)
    for k in train:
        for c in shuf.get(k, ()):
            mod_tr_n[c].append(kidx[k])
    Pm_n = np.zeros_like(Pm); has_n = np.zeros(M, bool)
    for c, rows in mod_tr_n.items():
        Pm_n[c] = np.nanmean(X[rows], 0); has_n[c] = True
    has_, Pm_ = has_n, Pm_n
    W2n = np.zeros_like(W0)
    for r, k in enumerate(test):
        W2n[r], _ = process_pred(k, shuf)
    r0, r2, r2n = pearson_rows(W0, actual), pearson_rows(W2, actual), pearson_rows(W2n, actual)
    P_(f"  TEST knockdowns {len(test):,}; with a process carrying TRAIN knockdowns: {cov:,}")
    for nm, rr in (("W0 tide", r0), ("W2 whole-cell process", r2), ("W2n process null", r2n)):
        P_(f"     {nm:<24} median r {np.nanmedian(rr):.3f}   mean r {np.nanmean(rr):.3f}")
    for key, a, b, nm in (("V1", r2, r0, "W2 process vs W0 tide"), ("V2", r2, r2n, "W2 process vs W2n null")):
        w, l, p = sign_test(a - b)
        v = "BETTER" if p < 0.05 and w > l else "WORSE" if p < 0.05 else "NO DIFFERENCE"
        res[key] = dict(better=w, worse=l, p=p, verdict=v, median_a=float(np.nanmedian(a)), median_b=float(np.nanmedian(b)))
        P_(f"  {key} {nm:<26} better {w}, worse {l}, p {p:.3g} -> {v}")
    # gene-level network on the first 500 TEST knockdowns by hash
    D = json.load(gzip.open(dc.ENCY)); bn = [r["name"] for r in D["genes"]]
    protein = {bn[int(k)]: v for k, v in D["ppm"].items() if k.isdigit()}
    bank_rows = cb.facts_table(D, bn)[0]
    omni_rows = [(s, t, sg, 0 if ty == "transcriptional" else 1) for s, t, sg, ty, _ in cd.load_omnipath()]
    meas_rows = []
    for k in [k for k in kos if split(k) == "TRAIN"]:
        z = X[kidx[k]]
        for j in np.where(np.isfinite(z) & (np.abs(z) >= zmov))[0]:
            if genes[j] != k:
                meas_rows.append((k, genes[j], int(-np.sign(z[j])), 0))
    rows3 = list(bank_rows) + omni_rows + meas_rows
    by3 = collections.defaultdict(list)
    for i, r_ in enumerate(rows3):
        by3[r_[0]].append(i)
    sub = sorted(range(len(test)), key=lambda r: hashlib.sha256(test[r].encode()).hexdigest())[:500]
    W1 = np.zeros((len(sub), len(genes)), dtype=np.float32)
    for q, r in enumerate(sub):
        C = dc.Cell(bn, protein, {}, {}, {}); pr = cb.new_engine(C, rows3, by3, {test[r]: -1})
        for g, (d, hop, cf) in pr.items():
            if g in gidx:
                W1[q, gidx[g]] = d * cf
    zs = lambda A: (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9)
    W3 = zs(W2[sub]) + zs(W1)
    r1, r3 = pearson_rows(W1, actual[sub]), pearson_rows(W3, actual[sub])
    r2s = r2[sub]
    P_(f"  on the first {len(sub)} TEST knockdowns by hash: W1 gene network median r {np.nanmedian(r1):.3f}; "
       f"W2 process {np.nanmedian(r2s):.3f}; W3 process + network {np.nanmedian(r3):.3f}")
    for key, a, b, nm in (("V3", r2s, r1, "W2 process vs W1 gene network"), ("V4", r3, r2s, "W3 process+network vs W2")):
        w, l, p = sign_test(a - b)
        v = "BETTER" if p < 0.05 and w > l else "WORSE" if p < 0.05 else "NO DIFFERENCE"
        res[key] = dict(better=w, worse=l, p=p, verdict=v, median_a=float(np.nanmedian(a)), median_b=float(np.nanmedian(b)))
        P_(f"  {key} {nm:<30} better {w}, worse {l}, p {p:.3g} -> {v}")

    # ---------------------------------------------------------------- F survival ----------------
    P_("\n" + RULE); P_("F  THE CELL'S BOTTOM LINE: DOES IT SURVIVE? (DepMap K562 gene effect)"); P_(RULE)
    dep = cd.load_depmap()
    k562 = dep.loc["ACH-000551"]
    nm_ = {k: int(np.nansum(np.abs(X[kidx[k]]) >= zmov)) for k in kos}
    both = [k for k in kos if k in k562.index and np.isfinite(k562[k])]
    rho, pr_ = spearmanr([nm_[k] for k in both], [-k562[k] for k in both])
    f1 = rho > 0 and pr_ < 0.05
    P_(f"  F1 knockdowns that disturb more of the cell kill it more: Spearman {rho:+.3f} over {len(both):,} genes, p {pr_:.2g} -> {'YES' if f1 else 'NO'}")
    trg = [k for k in both if split(k) == "TRAIN"]; teg = [k for k in both if split(k) == "TEST"]

    def ess_auc(gm):
        msum = collections.defaultdict(list)
        for k in trg:
            for c in gm.get(k, ()):
                msum[c].append(k562[k])
        mm = {c: float(np.mean(v)) for c, v in msum.items()}
        sc, y = [], []
        base = float(np.mean([k562[k] for k in trg]))
        for k in teg:
            cs = [mm[c] for c in gm.get(k, ()) if c in mm]
            sc.append(-(np.mean(cs) if cs else base)); y.append(k562[k] <= -0.5)
        sc, y = np.array(sc), np.array(y)
        rk = rankdata(sc)
        return float((rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (len(y) - y.sum())))
    auc = ess_auc(gene_mods)
    nulls = []
    for _ in range(200):
        pm = rng.permutation(len(allg))
        nulls.append(ess_auc({allg[i]: gene_mods[allg[pm[i]]] for i in range(len(allg))}))
    f2 = auc > np.percentile(nulls, 95) and auc > 0.60
    P_(f"  F2 process knowledge predicts which TEST genes the cell cannot live without: AUROC {auc:.3f} "
       f"(null 95th pct {np.percentile(nulls, 95):.3f}) -> {'PASS' if f2 else 'FAIL'}; essential TEST genes "
       f"{sum(k562[k] <= -0.5 for k in teg):,} of {len(teg):,}")
    res["F1"] = dict(rho=float(rho), p=float(pr_), verdict=f1); res["F2"] = dict(auc=auc, null95=float(np.percentile(nulls, 95)), verdict=f2)

    # ---------------------------------------------------------------- C coupling map ------------
    P_("\n" + RULE); P_("C  THE WHOLE-CELL COUPLING MAP: WHICH PROCESSES PUSH WHICH"); P_(RULE)

    def couplings(gm, seed):
        rr = np.random.default_rng(seed)
        memb = collections.defaultdict(list)
        for k in phen:
            for c in gm.get(k, ()):
                memb[c].append(kidx[k])
        found = []
        for a, rows in memb.items():
            if len(rows) < 8:
                continue
            rows = np.array(rows)
            dlt = S[rows] - tideS_all
            zz = dlt.mean(0) / (sdS / math.sqrt(len(rows)) + 1e-12)
            h = rr.permutation(len(rows)); h1, h2 = rows[h[: len(rows) // 2]], rows[h[len(rows) // 2:]]
            z1 = (S[h1] - tideS_all).mean(0) / (sdS / math.sqrt(len(h1)) + 1e-12)
            z2 = (S[h2] - tideS_all).mean(0) / (sdS / math.sqrt(len(h2)) + 1e-12)
            for b in np.where(np.abs(zz) >= 5)[0]:
                if b == a:
                    continue
                ov = len(mods[mid[a]] & mods[mid[b]]) / max(1, min(len(mods[mid[a]]), len(mods[mid[b]])))
                if ov >= 0.2:
                    continue
                if np.sign(z1[b]) == np.sign(zz[b]) == np.sign(z2[b]) and abs(z1[b]) >= 2 and abs(z2[b]) >= 2:
                    found.append((a, int(b), float(zz[b]), len(rows)))
        return found
    real = couplings(gene_mods, 1)
    pm = rng.permutation(len(allg))
    nullc = couplings({allg[i]: gene_mods[allg[pm[i]]] for i in range(len(allg))}, 1)
    fdr = len(nullc) / max(len(real), 1)
    P_(f"  replicated couplings (|z| >= 5, same sign in both halves, overlap < 20%): {len(real):,}; null {len(nullc):,} -> estimated FDR {fdr:.2f}"
       + ("" if p0 else "   [NOT REPORTED AS A MAP: P0 failed]"))
    res["C"] = dict(real=len(real), null=len(nullc), fdr=fdr)
    if p0:
        sysmap = collections.Counter()
        for a, b, zz, n in real:
            for ta in rtop[mid[a]][:1]:
                for tb in rtop[mid[b]][:1]:
                    if ta != tb:
                        sysmap[(ta, tb, "raises" if zz > 0 else "lowers")] += 1
        P_("  between whole-cell SYSTEMS (Reactome top level), most frequent replicated couplings:")
        for (ta, tb, dirn), n in sysmap.most_common(15):
            P_(f"     {ta[:34]:<34} {dirn:<7} {tb[:34]:<34} ({n} process pairs)")
        P_("  strongest single couplings:")
        for a, b, zz, n in sorted(real, key=lambda x: -abs(x[2]))[:15]:
            P_(f"     knocking down {rnames[mid[a]][:38]:<38} {'raises' if zz > 0 else 'lowers'} {rnames[mid[b]][:38]:<38} z {zz:+.1f} ({n} knockdowns)")
        res["C"]["systems"] = [[ta, tb, d_, n] for (ta, tb, d_), n in sysmap.most_common(40)]
        res["C"]["top"] = [[rnames[mid[a]], rnames[mid[b]], zz, n] for a, b, zz, n in sorted(real, key=lambda x: -abs(x[2]))[:200]]
    os.makedirs(BANK, exist_ok=True)
    json.dump({"whole_cell_couplings": res["C"], "process_definitions": "Reactome human pathways (CC0), 15-300 measured genes",
               "data": "K562 gwps Perturb-seq (CC BY 4.0), DepMap 24Q4 K562 (CC BY 4.0)"},
              open(os.path.join(BANK, "whole_cell_map.json"), "w"), indent=1, default=float)
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump(res, open(ART, "w"), indent=1, default=float)
    P_(f"\n  map written to memory_bank/cell_v3/whole_cell_map.json ; artifact outputs/wholecell.json ; runtime {time.time() - t0:.0f}s")
    P_("  X: one cell line, CRISPRi, steady state; transcriptome only -- no metabolite, protein or flux readout.")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
