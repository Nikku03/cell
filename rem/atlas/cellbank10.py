"""CELLBANK10 -- using the AlphaGenome 2 data (memory_bank/datasets/alphagenome2_holdings.json) in the K562 engine.
USE 1 ("is the gene on?") and USE 2 ("learn from the right cell types"), tested together in one run.

WHY. The model review and cellbank8 found the engine at a plateau (~59% of the ceiling) because most knockdown
records carry little signal, and accuracy rises steeply with record strength. The new bank says, for thousands of cell
types, whether a gene is expressed, whether its promoter / enhancers are active, and whether its promoter is silenced.

USE 1 -- K562 CONTEXT PER GENE (11 numbers; all from rem/atlas/_cache/ag2bank; 0 + a flag where missing):
  K562 expression: ENCODE RNA-seq (mean log1p TPM over the 80 K562 experiments) and FANTOM5 CAGE (K562 samples);
  breadth: share of CELLxGENE human labelled (tissue, cell type) pairs where >= 10% of cells express it, and share of
  GTEx tissues with TPM >= 1; K562 chromatin (ENCODE Registry v4, union of the K562 biosamples): an active element
  within TSS +/- 1 kb (any class; and PLS specifically), log1p number of active pELS/dELS within +/- 100 kb;
  methylation (Loyfer 2023): promoter beta in bone-marrow erythrocyte progenitors (the closest cells to K562 in the
  atlas), and the share of the 253 samples where the promoter is unmethylated (beta < 0.2); missing flags (RNA, meth).
USE 2 -- CONTEXT-AWARE TRANSLATION. cellbank8's translator trusts each record by its strength,
  w = sigmoid(a_s log strength + b_s). Here also by whether the gene is expressed alike in that screen's cell type and
  in K562: w = sigmoid(a_s log strength + b_s + c_s |e_s(g) - e_K562(g)| + d_s missing_s(g)), e = ENCODE mean log1p
  TPM of the screen's cell type (RPE1 none, HepG2, Jurkat 'Jurkat, Clone E6-1', HCT116, HEK293T none, iPSC screens
  WTC11, hESC screens H1). c, d start at 0, so the model is cellbank8's at initialisation.

ARMS (cellbank8's recipe, seeds 0-2 averaged; bank v8; all retrained in this run)
  M2    cellbank8's model, rerun (reproduced exactly in cellbank9).
  M10a  + the 11 context numbers as inputs (USE 1).
  M10b  context-aware translator (USE 2).
  M10   both.

GATES, PREDECLARED (EXAM half of cellbank8: its THIRD read; the first two were cellbank8's and cellbank9's gates;
nothing here was chosen by looking at it; development on DEV only; `exam` scores once and locks)
  U1  USE 1, PRIMARY: M10a vs M2, r, all EXAM genes. BETTER = sign test p < 0.05 and median gain >= +0.01.
  U2  USE 1, confidence: cellbank8's confidence (its 9 features, OLS fitted on DEV to M2's r) vs the same plus the 11
      context numbers, fitted the same way. Spearman with M2's r on EXAM. IMPROVES if the gain >= 0.03 and the 2.5th
      percentile of 2,000 paired bootstrap resamples of EXAM genes is > 0.
  U3  USE 1, 'off means no effect' in other cell types (no model; uses no EXAM data): in the reprocessed screens
      HepG2, Jurkat, HCT116, iPSC-N, hESC-P, Spearman between the target gene's ENCODE expression in that cell type
      and its knockdown record strength (all genes with a record). PASS if rho >= 0.10 with p < 0.05 in >= 4 of 5.
  U4  USE 2: M10b vs M2, r, all EXAM genes. BETTER as U1.
  Descriptive: M10 vs M2; DEV numbers; coverage of each context source.
NOT: one context summary (no per-cell-type learning), ENCODE expression as the cell-type match (no RPE1 / HEK293T
expression), the methylation atlas has no K562 (closest cells used).

Run: python3 -P rem/atlas/cellbank10.py train | dev | exam
"""

from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
sys.path.insert(0, HERE)
_s = importlib.util.spec_from_file_location("cellbank9", os.path.join(HERE, "cellbank9.py"))
cb9 = importlib.util.module_from_spec(_s); _s.loader.exec_module(cb9)
cb8, cb4 = cb9.cb8, cb9.cb4
_s = importlib.util.spec_from_file_location("ag2bank", os.path.join(HERE, "ag2bank.py"))
ag = importlib.util.module_from_spec(_s); _s.loader.exec_module(ag)

AG = ag.BANK
CACHE10 = os.path.join(HERE, "_cache", "cellbank10")
OUTD10 = R("outputs", "cellbank10")
OUT10 = os.path.join(HERE, "RESULTS_cellbank10.txt")
SCREEN_CELL = {"RPE1": None, "HepG2": "HepG2", "Jurkat": "Jurkat, Clone E6-1", "HCT116": "HCT116", "HEK293T": None,
               "iPSC-N": "WTC11", "iPSC-Fsf": "WTC11", "iPSC-Fsnf": "WTC11", "iPSC-Fts": "WTC11", "hESC-P": "H1", "hESC-V": "H1"}
EFFECT_SCREENS = ["HepG2", "Jurkat", "HCT116", "iPSC-N", "hESC-P"]
CTX_NAMES = ["k562_rna", "k562_cage", "breadth_cellxgene", "breadth_gtex", "k562_tss_active", "k562_tss_pls",
             "k562_distal_els", "meth_erythroid", "meth_unmethylated_share", "missing_rna", "missing_meth"]
SEEDS, RULE = cb8.SEEDS, cb8.RULE


# ---------------------------------------------------------------------------------------------------------
# gene ids and the context sources
# ---------------------------------------------------------------------------------------------------------
def sym2ens():
    import csv
    best, alias = {}, {}
    for r in csv.DictReader(open(os.path.join(HERE, "hgnc_complete_set.txt")), delimiter="\t"):
        e = r.get("ensembl_gene_id") or ""
        if not e:
            continue
        best[r["symbol"].upper()] = e
        for k in ("alias_symbol", "prev_symbol"):
            for a in (r.get(k) or "").replace('"', "").split("|"):
                if a:
                    alias.setdefault(a.upper(), set()).add(e)
    for a, es in alias.items():
        if a not in best and len(es) == 1:
            best[a] = next(iter(es))
    return best


def encode_by_biosample():
    import numpy as np
    z = np.load(os.path.join(AG, "bulk", "encode_rna_human.npz")); m = json.load(open(os.path.join(AG, "bulk", "encode_rna_human_meta.json")))["meta"]
    X = z["X"].astype(np.float32); cols = {g: i for i, g in enumerate(z["cols"].tolist())}
    out = {}
    for b in set(m["biosample"]):
        idx = [i for i, x in enumerate(m["biosample"]) if x == b]
        out[b] = X[idx].mean(0)
    return out, cols


def context(G):
    """NPH x 11 K562 context tensor (see the docstring), plus coverage counts."""
    import numpy as np
    torch = G["torch"]; KN = G["KN"]; NPH = G["NPH"]
    s2e = sym2ens(); ens = [s2e.get(g.upper()) for g in KN]
    C = np.zeros((NPH, len(CTX_NAMES)), np.float32); cov = {}
    enc, ecol = encode_by_biosample(); k = enc["K562"]
    v = [k[ecol[e]] if e in ecol else np.nan for e in ens]; C[:, 0] = np.nan_to_num(v); C[:, 9] = np.isnan(v); cov["k562_rna"] = int(np.isfinite(v).sum())
    z = np.load(os.path.join(AG, "bulk", "fantom5_cage_human.npz")); fm = json.load(open(os.path.join(AG, "bulk", "fantom5_cage_human_meta.json")))["meta"]
    rows = [i for i, dsc in enumerate(fm["description"]) if "K562" in dsc]
    cage = z["X"][rows].astype(np.float32).mean(0); fcol = {g.upper(): i for i, g in enumerate(z["cols"].tolist())}
    C[:, 1] = [cage[fcol[g.upper()]] if g.upper() in fcol else 0.0 for g in KN]; cov["k562_cage_samples"] = len(rows)
    x = np.load(os.path.join(AG, "cellxgene", "cellxgene_human.npz")); pj = json.load(open(os.path.join(AG, "cellxgene", "cellxgene_human_pairs.json")))
    lab = np.array([p["labelled"] for p in pj["pairs"]]); nlab = int(lab.sum())
    gi = {g: i for i, g in enumerate(x["genes"].tolist())}; ip, col, pc = x["indptr"], x["col"], x["pc"]
    br = []
    for e in ens:
        i = gi.get(e)
        if i is None:
            br.append(0.0); continue
        c_, p_ = col[ip[i]:ip[i + 1]], pc[ip[i]:ip[i + 1]]
        br.append(float(((p_ >= 26) & lab[c_]).sum()) / nlab)
    C[:, 2] = br
    z = np.load(os.path.join(AG, "bulk", "gtex_v8_tissue_median.npz")); gc = {g: i for i, g in enumerate(z["cols"].tolist())}
    gx = z["X"].astype(np.float32)
    C[:, 3] = [float((gx[:, gc[e]] >= math.log(2)).mean()) if e in gc else 0.0 for e in ens]
    # chromatin in K562: union over the K562 biosamples of the Registry v4
    meta = json.load(open(os.path.join(AG, "ccre", "human", "meta.json"))); reg = np.load(os.path.join(AG, "ccre", "human", "registry.npz"))
    cls = np.full(len(reg["id"]), 255, np.uint8)
    for b in meta["biosamples"]:
        if b["biosample"] == "K562":
            s = np.load(os.path.join(AG, "ccre", "human", "shards", b["annotation"] + ".npz"))
            cls[s["idx"]] = np.minimum(cls[s["idx"]], s["cls"])        # lower code = more active class (PLS first)
    mid = ((reg["start"].astype(np.int64) + reg["end"]) // 2); chrom = reg["chrom"]
    tss = {g[0]: (g[3], g[4]) for g in _tss_cached()}
    bychr = {}
    for c_ in np.unique(chrom):
        ix = np.nonzero(chrom == c_)[0]; o = np.argsort(mid[ix]); bychr[c_] = (mid[ix][o], cls[ix][o])
    for j, e in enumerate(ens):
        t = tss.get(e)
        if t is None or t[0] not in bychr:
            continue
        m_, c_ = bychr[t[0]]
        lo, hi = np.searchsorted(m_, t[1] - 1000), np.searchsorted(m_, t[1] + 1000)
        near = c_[lo:hi]; C[j, 4] = float((near < 255).any()); C[j, 5] = float((near == 0).any())
        lo, hi = np.searchsorted(m_, t[1] - 100000), np.searchsorted(m_, t[1] + 100000)
        C[j, 6] = math.log1p(int(np.isin(c_[lo:hi], [1, 2]).sum()))
    cov["k562_ccre_biosamples"] = sum(b["biosample"] == "K562" for b in meta["biosamples"])
    mz = np.load(os.path.join(AG, "methylation", "loyfer2023_promoter_beta.npz")); mg = {g: i for i, g in enumerate(mz["genes"].tolist())}
    B = mz["beta"]; ery = [i for i, s in enumerate(mz["samples"].tolist()) if "Erythrocyte_progenitors" in s]
    for j, e in enumerate(ens):
        i = mg.get(e)
        if i is None:
            C[j, 10] = 1; continue
        col_ = B[:, i]; ok = col_ < 255
        ev = [col_[k_] for k_ in ery if col_[k_] < 255]
        C[j, 7] = float(np.mean(ev)) / 250 if ev else 0.0; C[j, 10] = float(not ev)
        C[j, 8] = float((col_[ok] < 50).mean()) if ok.any() else 0.0
    cov["meth_erythroid_samples"] = len(ery); cov["genes_with_ensembl"] = int(sum(e is not None for e in ens))
    return torch.tensor(C), cov


def _tss_cached():
    p = os.path.join(AG, "raw", "gencode46_tss.json")
    if not os.path.exists(p):
        json.dump(ag._tss(), open(p, "w"))
    return json.load(open(p))


def delta(G):
    """(S+1) x NPH: |e_screen(g) - e_K562(g)| and a missing flag; index S (the K562 target) is 0."""
    import numpy as np
    torch = G["torch"]; S = len(cb8.SRC); NPH = G["NPH"]
    s2e = sym2ens(); ens = [s2e.get(g.upper()) for g in G["KN"]]
    enc, ecol = encode_by_biosample(); k = enc["K562"]
    D = np.zeros((S + 1, NPH), np.float32); Mi = np.zeros((S + 1, NPH), np.float32)
    for s, name in enumerate(cb8.SRC):
        b = SCREEN_CELL.get(name)
        for j, e in enumerate(ens):
            if b is None or b not in enc or e not in ecol:
                Mi[s, j] = 1.0
            else:
                D[s, j] = abs(float(enc[b][ecol[e]]) - float(k[ecol[e]]))
    return torch.tensor(D), torch.tensor(Mi)


# ---------------------------------------------------------------------------------------------------------
# USE 2: the context-aware translator (cellbank8's, plus c_s |delta| + d_s missing in each record's trust)
# ---------------------------------------------------------------------------------------------------------
def make_translator2(G, S):
    nn, torch = G["nn"], G["torch"]

    class T(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Parameter(torch.randn(S + 1, 128, cb8.T_DIM) * (1 / math.sqrt(128)))
            self.dec = nn.Parameter(torch.randn(S + 1, cb8.T_DIM, 128) * (1 / math.sqrt(cb8.T_DIM)))
            self.ga, self.gb = nn.Parameter(torch.ones(S + 1)), nn.Parameter(torch.zeros(S + 1))
            self.gc, self.gd = nn.Parameter(torch.zeros(S + 1)), nn.Parameter(torch.zeros(S + 1))

        def forward(self, Zg, Mg, tgt, Lg, Dg, Mig):
            w = Mg * torch.sigmoid(self.ga * Lg + self.gb + self.gc * Dg + self.gd * Mig)
            H = torch.einsum("nsd,sdh->nsh", Zg, self.enc) * w[..., None]
            h = H.sum(1) / w.sum(1, keepdim=True).clamp(min=1e-3)
            return torch.einsum("nh,nhd->nd", h, self.dec[tgt])
    return T()


def train_translator2(G, Z, Mk, Lg, D, Mi, genes_ok, seed):
    torch, np = G["torch"], G["np"]
    S = Z.shape[0] - 1
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Zg, Mg, Lgg, Dg, Mig = Z.permute(1, 0, 2), Mk.T.float(), Lg.T, D.T, Mi.T
    ok = [j for j in genes_ok if int(Mk[:, j].sum()) >= 2]
    tr = make_translator2(G, S); opt = torch.optim.Adam(tr.parameters(), lr=cb8.T_LR, weight_decay=cb8.T_WD)
    for step in range(cb8.T_STEPS):
        b = rng.choice(ok, size=min(cb8.T_BATCH, len(ok)), replace=False)
        avail = Mg[b].numpy().astype(bool)
        tgt = np.array([rng.choice(np.nonzero(a)[0]) for a in avail])
        keep = (rng.random(avail.shape) < cb8.T_KEEP) & avail
        keep[np.arange(len(b)), tgt] = False
        for i in np.nonzero(keep.sum(1) == 0)[0]:
            others = [s for s in np.nonzero(avail[i])[0] if s != tgt[i]]; keep[i, rng.choice(others)] = True
        bt = torch.as_tensor(b); x = Zg[bt]; m = torch.tensor(keep, dtype=torch.float32); t = torch.tensor(tgt)
        pred = tr(x, m, t, Lgg[bt], Dg[bt], Mig[bt])
        y = x[torch.arange(len(b)), t]
        loss = 1 - torch.nn.functional.cosine_similarity(pred, y, dim=1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return tr


def translate2(G, tr, Z, Mk, Lg, D, Mi, rows, target):
    torch = G["torch"]
    with torch.no_grad():
        Zg, Mg = Z.permute(1, 0, 2)[rows], Mk.T.float()[rows].clone(); Mg[:, target] = 0.0
        return tr(Zg, Mg, torch.full((len(rows),), target, dtype=torch.long), Lg.T[rows], D.T[rows], Mi.T[rows]), Mg.sum(1)


def translator_inputs2(G, Z, Mk, Lg, D, Mi, t0):
    torch = G["torch"]; KN, rtr, rte, split = G["KN"], G["rtr"], G["rte"], G["split"]
    S = Z.shape[0] - 1
    train_genes = [j for j, g in enumerate(KN) if split(g) == "TRAIN"]
    fold = lambda g: int(hashlib.sha256(("fold:" + g).encode()).hexdigest(), 16) % 5
    Tz = torch.zeros(G["NPH"], 128); Tn = torch.zeros(G["NPH"]); tr_rows = [int(i) for i in rtr]
    for k in range(5):
        tr = train_translator2(G, Z, Mk, Lg, D, Mi, [j for j in train_genes if fold(KN[j]) != k], seed=k)
        rows = [j for j in tr_rows if fold(KN[j]) == k]
        p, nin = translate2(G, tr, Z, Mk, Lg, D, Mi, rows, S); Tz[rows] = p; Tn[rows] = nin
    final = train_translator2(G, Z, Mk, Lg, D, Mi, train_genes, seed=99)
    te_rows = [int(i) for i in rte]
    p, nin = translate2(G, final, Z, Mk, Lg, D, Mi, te_rows, S); Tz[te_rows] = p; Tn[te_rows] = nin
    has_T = Tn > 0
    Tu = torch.where(has_T[:, None], Tz / Tz.norm(dim=1, keepdim=True).clamp(min=1e-12), torch.zeros_like(Tz))
    flags = torch.cat([Mk[:S].T.float(), torch.log1p(Mk[:S].sum(0).float())[:, None], has_T.float()[:, None], Lg[:S].T], 1)
    learned = dict(c=final.gc.detach().tolist(), d=final.gd.detach().tolist())
    print(f"  context-aware translator done: {time.time() - t0:.0f}s; c_s {[round(x, 2) for x in learned['c']]}", flush=True)
    return Tu, flags, learned


# ---------------------------------------------------------------------------------------------------------
# train / dev / exam
# ---------------------------------------------------------------------------------------------------------
def train():
    t0 = time.time()
    G = cb9.nb9(); torch = G["torch"]
    os.makedirs(CACHE10, exist_ok=True); os.makedirs(OUTD10, exist_ok=True)
    bank6 = cb4.Bank(cb8.V6); te_rows = [int(i) for i in G["rte"]]
    CTX, cov = context(G); D, Mi = delta(G)
    print(f"  context ready {tuple(CTX.shape)} {cov}; delta coverage {float((1 - Mi[:len(cb8.SRC)]).mean()):.2f} ({time.time() - t0:.0f}s)", flush=True)
    Z8, Mk8, St8, Lg8 = cb9.tensors(G, cb8.BANK8, cb9.SRC8)
    Tu8, fl8, _ = cb9.translator_inputs(G, Z8, Mk8, Lg8, St8, t0, "v8")
    TuC, flC, learned = translator_inputs2(G, Z8, Mk8, Lg8, D, Mi, t0)
    K, hasK = cb8.k562_records(G, bank6); An8 = cb8.analog(G, Z8, Mk8, K, hasK); Pt, Ct = cb4.inputs(G, bank6, False)
    arms = {"M2": ([Tu8, fl8, An8, Pt, Ct], Tu8), "M10a": ([Tu8, fl8, An8, Pt, Ct, CTX], Tu8),
            "M10b": ([TuC, flC, An8, Pt, Ct], TuC), "M10": ([TuC, flC, An8, Pt, Ct, CTX], TuC)}
    preds, info = {}, {}
    for arm, (blocks, tz) in arms.items():
        X = cb4.scaled(G, blocks); outs = []
        for s in SEEDS:
            pz, best = cb8.train_residual(G, X, tz.to(G["DEV"]), "mlp", s)
            outs.append(pz); info.setdefault(arm, []).append(round(best, 4))
            print(f"  {arm} seed {s}: held-out TRAIN r {best:+.3f} ({X.shape[1]} features, {time.time() - t0:.0f}s)", flush=True)
        preds[arm] = torch.stack(outs).mean(0)
    torch.save(dict(preds=preds, ctx=CTX[te_rows], te_rows=te_rows), os.path.join(CACHE10, "train_out.pt"))
    json.dump(dict(heldout_train_r=info, context_coverage=cov, translator_c_d=learned, seconds=round(time.time() - t0)),
              open(os.path.join(OUTD10, "train_info.json"), "w"), indent=1)
    print(f"train done ({time.time() - t0:.0f}s)")


ARMS = ("M2", "M10a", "M10b", "M10")
LABEL = {"M2": "M2   cellbank8 model, rerun", "M10a": "M10a + K562 context (use 1)", "M10b": "M10b context-aware translation (use 2)", "M10": "M10  both"}


def _common(half):
    import numpy as np
    G = cb9.nb9(); torch = G["torch"]
    o = torch.load(os.path.join(CACHE10, "train_out.pt")); preds = o["preds"]
    score, ceil, _ = cb8.scorer(G)
    genes, dv, ex, _ = cb8.sets(G)
    m = dv if half == "DEV" else ex
    S = {a: score(preds[a]) for a in ARMS}
    return G, o, S, m, dv, ex, ceil, genes


def _confidence(G, o, dv, ex):
    """U2: cellbank8's confidence vs the same + context; OLS on DEV to cellbank8's M2 r; Spearman on EXAM + bootstrap."""
    import numpy as np
    from scipy.stats import spearmanr
    torch = G["torch"]
    o8 = torch.load(os.path.join(cb8.CACHE, "train_out.pt")); score, _, _ = cb8.scorer(G)
    r = score(o8["preds"]["M2"])[0]
    F = cb8.conf_features(G, None, o8); Fc = np.c_[F, o["ctx"].numpy()]
    res = {}
    for lab, X in (("base", F), ("context", Fc)):
        Xd = np.c_[np.ones(dv.sum()), X[dv]]; beta = np.linalg.lstsq(Xd, r[dv], rcond=None)[0]
        res[lab] = np.c_[np.ones(len(X)), X] @ beta
    rb = spearmanr(res["base"][ex], r[ex]).correlation; rc = spearmanr(res["context"][ex], r[ex]).correlation
    rng = np.random.default_rng(10); idx = np.nonzero(ex)[0]; diffs = []
    for _ in range(2000):
        b = rng.choice(idx, len(idx), replace=True)
        diffs.append(spearmanr(res["context"][b], r[b]).correlation - spearmanr(res["base"][b], r[b]).correlation)
    lo = float(np.percentile(diffs, 2.5))
    return dict(base=float(rb), context=float(rc), gain=float(rc - rb), boot_2_5=lo,
                verdict="IMPROVES" if rc - rb >= 0.03 and lo > 0 else "NO IMPROVEMENT")


def _effect_detection():
    """U3: Spearman(target expression in the screen's cell type, record strength), reprocessed screens."""
    import numpy as np
    from scipy.stats import spearmanr
    meta = json.load(open(os.path.join(cb8.BANK8, "records_meta.json")))
    s2e = sym2ens(); enc, ecol = encode_by_biosample(); out = {}
    for name in EFFECT_SCREENS:
        b = SCREEN_CELL[name]; e_, s_ = [], []
        for i, (g, sc) in enumerate(zip(meta["gene"], meta["screen"])):
            if sc == f"S:{name}":
                e = s2e.get(g.upper())
                if e in ecol:
                    e_.append(float(enc[b][ecol[e]])); s_.append(meta["strength"][str(i)])
        sp = spearmanr(e_, s_)
        out[name] = dict(biosample=b, genes=len(e_), rho=float(sp.correlation), p=float(sp.pvalue),
                         **{"pass": bool(sp.correlation >= 0.10 and sp.pvalue < 0.05)})
    k = sum(v["pass"] for v in out.values())
    return dict(screens=out, passing=k, verdict="PASS" if k >= 4 else "FAIL")


def _table(S, m, ceil, P_):
    import numpy as np
    for a in ARMS:
        r, rs = S[a]
        P_(f"     {LABEL[a]:40s} {np.median(r[m]):+.3f} | {np.median(rs[m]):+.3f} | {np.median(r[m]) / np.median(ceil[m]):.0%}")


def dev():
    out = []
    P_ = lambda s="": (print(s, flush=True), out.append(s))
    G, o, S, m, dv, ex, ceil, genes = _common("DEV")
    P_(RULE); P_("CELLBANK10 -- AG2 data in the K562 engine (uses 1 and 2): DEVELOPMENT SET (open)"); P_(RULE)
    P_(f"  DEV genes {m.sum()}; ceiling {float(__import__('numpy').median(ceil[m])):.3f}; arms: median r | specific r | fraction of ceiling")
    _table(S, m, ceil, P_)
    for a in ("M10a", "M10b", "M10"):
        v = cb8.verdict(S[a][0][m], S["M2"][0][m])
        P_(f"     DEV {a:5s} vs M2: better {v['better']} / worse {v['worse']}, median {v['median_diff']:+.4f}, p {v['p']:.2g}")
    open(os.path.join(HERE, "RESULTS_cellbank10_dev.txt"), "w").write("\n".join(out) + "\n")


def exam():
    lock = os.path.join(OUTD10, "exam_lock.json")
    if os.path.exists(lock):
        print("EXAM already scored for cellbank10; refusing."); return
    import numpy as np
    out = []
    P_ = lambda s="": (print(s, flush=True), out.append(s))
    G, o, S, m, dv, ex, ceil, genes = _common("EXAM")
    P_(RULE); P_("CELLBANK10 -- THE EXAM HALF, THIRD READ (scored once for these gates)"); P_(RULE)
    P_(f"  EXAM genes {m.sum()}; ceiling {np.median(ceil[m]):.3f}; arms: median r | specific r | fraction of ceiling")
    _table(S, m, ceil, P_)
    V = {"U1": cb8.verdict(S["M10a"][0][m], S["M2"][0][m]), "U4": cb8.verdict(S["M10b"][0][m], S["M2"][0][m])}
    V["U2"] = _confidence(G, o, dv, ex); V["U3"] = _effect_detection()
    v = V["U1"]; P_(f"  U1 use 1, M10a vs M2, r: better {v['better']} / worse {v['worse']}, median {v['median_diff']:+.4f}, p {v['p']:.2g} -> {v['verdict']}")
    c = V["U2"]; P_(f"  U2 use 1, confidence Spearman: base {c['base']:+.3f}, + context {c['context']:+.3f}, gain {c['gain']:+.3f}, bootstrap 2.5% {c['boot_2_5']:+.3f} -> {c['verdict']}")
    e = V["U3"]; P_(f"  U3 use 1, off -> no effect: {e['passing']}/5 screens -> {e['verdict']}  (" +
                     "; ".join(f"{k} rho {x['rho']:+.2f} n {x['genes']}" for k, x in e["screens"].items()) + ")")
    v = V["U4"]; P_(f"  U4 use 2, M10b vs M2, r: better {v['better']} / worse {v['worse']}, median {v['median_diff']:+.4f}, p {v['p']:.2g} -> {v['verdict']}")
    d = cb8.verdict(S["M10"][0][m], S["M2"][0][m])
    P_(f"  (descriptive) M10 vs M2: better {d['better']} / worse {d['worse']}, median {d['median_diff']:+.4f}, p {d['p']:.2g}")
    open(OUT10, "w").write("\n".join(out) + "\n")
    digest = hashlib.sha256(b"".join(o["preds"][a].numpy().tobytes() for a in ARMS)).hexdigest()
    json.dump(dict(verdicts=V, predictions_sha256=digest, exam_read="third (cellbank8, cellbank9 before)",
                   scored=time.strftime("%Y-%m-%d %H:%M")), open(lock, "w"), indent=1)


if __name__ == "__main__":
    {"train": train, "dev": dev, "exam": exam}[sys.argv[1]]()
