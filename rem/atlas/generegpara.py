"""GENEREGPARA -- substitute proteins: does a sister gene (paralog) cover for a gene, and does it step up when the gene
is switched off? Tested with Monte Carlo, and used to predict which cells need a gene.

Asked by the user: "but what about substitute proteins?" (after the partner-room rule, G4).
SUBSTITUTE of gene A = its closest paralog B (Ensembl Compara 116; the paralog with the highest min(% identity both
ways), >= 30%). Pairs whose TSS are within 100 kb are dropped where CRISPRi is involved (silencing can spread).
Data (cellbrain bank): PARALOGS_ENSEMBL; DepMap 24Q4 CRISPR gene effect (DEP_DEPMAP; more negative = more needed) and
RNA (EXPR_DEPMAP, log2 TPM+1) for the 1,103 cell lines with both; XCELL_* knockdown screens (links as in generegx);
HPA cell-line RNA for the four screened lines (made / level). MATCHED RANDOM GENE for B: a gene that is not a paralog
of A (any identity), drawn from B's decile of mean RNA level (DepMap lines, or that cell line's HPA level).

CLAIMS (declared before running; rem/atlas/cellbrain_ledger.jsonl, ids G5-*):
 G5-S1  COVER ACROSS ~1,100 CANCER CELL LINES. Genes A needed in >= 5% of lines (effect < -0.5) whose substitute is
        measured: Spearman across lines between B's RNA and A's effect (positive = more substitute, A less needed).
        CLAIM: median rho >= 0.05 and Monte Carlo p < 0.05 against 1,000 draws with matched random genes in B's place.
 G5-S2  COVER ACROSS THE FOUR SCREENED CELL TYPES. Knockdowns in all four essential screens whose substitute's RNA
        varies (HPA range of log1p nTPM >= 1): within-knockdown Spearman across K562, RPE1, HepG2, Jurkat between B's
        RNA percentile and A's knockdown reach percentile (share of genes it changes). CLAIM: mean <= -0.10,
        permutation p < 0.05 (10,000; B's cell-type labels permuted within knockdown) and Monte Carlo p < 0.05
        against 1,000 draws with random genes whose RNA also varies in B's place.
 G5-S3  STEPPING UP. Knocking down A (own knockdown measured, <= -0.5) raises B's mRNA. Pooled over the five screens:
        median log2 fold change of B in A's knockdown minus the mean of 1,000 Monte Carlo medians with matched random
        genes (same screen, not within 100 kb of A) >= +0.02 and p_MC < 0.05.
 G5-P1  PREDICT WHICH CELL LINES NEED A GENE. Cell lines split DEV / EXAM by sha256('cellbrain-c1:' + ModelID) parity.
        Per gene A (G5-S1 set): least squares on DEV lines, effect ~ 1 + A's own RNA (baseline) vs + B's RNA (model);
        score = Pearson r of predicted vs measured effect on EXAM lines. CLAIM: BETTER (house rule: sign test over
        genes p < 0.05, median gain >= +0.01) AND Monte Carlo p < 0.05 against 200 draws with matched random genes in
        B's place (median gain each).
Controls: negative = the first random draw must not pass; planted = S1/S3 real values + 0.05 (S2: -0.3) must pass;
P1 planted = effect replaced by the DEV-fitted baseline + B's RNA (scaled) + noise must be BETTER.
NOT: closest paralog only (families can share the load); RNA stands in for protein; DepMap is knockout in cancer
lines grown in a dish; cover is shown by correlation across cells, not by switching the substitute off too.

Run: python3 -I rem/atlas/generegpara.py declare | test [--trial]
"""

from __future__ import annotations
import collections
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
_argv = sys.argv; sys.argv = [sys.argv[0]] + (["--trial"] if TRIAL else [])
_s = importlib.util.spec_from_file_location("generegx", os.path.join(HERE, "generegx.py"))
gx = importlib.util.module_from_spec(_s); _s.loader.exec_module(gx)
sys.argv = _argv
cbb, Fact = gx.cbb, gx.Fact

LEDGER = gx.LEDGER
RUN = os.path.join(cbb.HOME, "generegpara_trial" if TRIAL else "generegpara")
OUTD = os.path.join(RUN, "out") if TRIAL else R("outputs", "genereg")
RES = os.path.join(RUN, "RESULTS.txt") if TRIAL else os.path.join(HERE, "RESULTS_generegpara.txt")
CLAIMS = ["G5-S1", "G5-S2", "G5-S3", "G5-P1"]
ESS, GW, CELL = gx.ESS, gx.GW, gx.CELL
MIN_ID, CIS_PAIR, SEED = 30.0, 100_000, 20261125
N1, N2P, N2, N3, NP = (50, 200, 50, 50, 10) if TRIAL else (1_000, 10_000, 1_000, 1_000, 200)
H_ = lambda e: int(hashlib.sha256(("cellbrain-c1:" + e).encode()).hexdigest(), 16) % 2


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "generegpara.py"), "rb").read()); h.update(gx.code_sha().encode())
    for n in ("PARALOGS_ENSEMBL", "DEP_DEPMAP", "EXPR_DEPMAP"):
        h.update(cbb.bank().facts(f"LAYER-{n}", "layer")[-1].value["sha256"].encode())
    return h.hexdigest()[:16]


def substitutes():
    z = gx.npz("PARALOGS_ENSEMBL"); a, b = z["a"].astype(str), z["b"].astype(str); mi = np.minimum(z["id_a"], z["id_b"])
    best, fam = {}, collections.defaultdict(set)
    for x, y, m in zip(a, b, mi):
        fam[x].add(y); fam[y].add(x)
        for p, q in ((x, y), (y, x)):
            if m >= MIN_ID and (p not in best or m > best[p][1]):
                best[p] = (q, float(m))
    return best, fam


def cis_pair(T, a, b):
    ta, tb = T.get(a), T.get(b)
    return bool(ta and tb and ta[0] == tb[0] and abs(ta[1] - tb[1]) <= CIS_PAIR)


def mc_p(real, nulls, greater=True):
    nulls = np.asarray(nulls, float)
    return float((1 + ((nulls >= real) if greater else (nulls <= real)).sum()) / (1 + len(nulls)))


def zrank(M):
    """rows standardised ranks (NaN -> row mean rank), for Spearman via dot products."""
    from scipy.stats import rankdata
    M = np.where(np.isfinite(M), M, np.nanmean(M, 1, keepdims=True))
    Rk = rankdata(M, axis=1).astype(np.float32); Rk -= Rk.mean(1, keepdims=True); sd = Rk.std(1, keepdims=True)
    return np.where(sd > 0, Rk / np.where(sd > 0, sd, 1), 0)


def deciles(v):
    return np.searchsorted(np.nanquantile(v, np.linspace(0.1, 0.9, 9)), np.nan_to_num(v, nan=-1e9))


def matched_draw(rng, pool_by_dec, dec_needed, exclude_sets, own):
    out = np.empty(len(dec_needed), np.int64)
    for i, (d, ex) in enumerate(zip(dec_needed, exclude_sets)):
        pool = pool_by_dec[d]
        for _ in range(20):
            c = pool[rng.integers(0, len(pool))]
            if c not in ex and c != own[i]:
                break
        out[i] = c
    return out


# ------------------------------------------------------------------------------------------------ DepMap (S1, P1)
def depmap(best, fam, rng):
    D = gx.npz("DEP_DEPMAP"); E = gx.npz("EXPR_DEPMAP")
    dl, el = D["cols"].astype(str), E["cols"].astype(str); lines = sorted(set(dl) & set(el))
    di = {c: i for i, c in enumerate(dl)}; ei = {c: i for i, c in enumerate(el)}
    Xd = D["X"].astype(np.float32)[:, [di[c] for c in lines]]; Xe = E["X"].astype(np.float32)[:, [ei[c] for c in lines]]
    drow = {g: i for i, g in enumerate(D["rows"].astype(str))}; erow = {g: i for i, g in enumerate(E["rows"].astype(str))}
    needed = np.nanmean(Xd < -0.5, 1) >= 0.05
    genes = [g for g, i in drow.items() if needed[i] and g in best and best[g][0] in erow]
    if TRIAL:
        genes = genes[:200]
    A_eff = Xd[[drow[g] for g in genes]]; ib = np.array([erow[best[g][0]] for g in genes])
    Ze = zrank(Xe); Za = zrank(A_eff); n = Za.shape[1]
    rho = (Za * Ze[ib]).sum(1) / n
    lvl = np.nanmean(Xe, 1); dec = deciles(lvl); pool = {d: np.nonzero(dec == d)[0] for d in range(10)}
    exr = [{erow[x] for x in fam.get(g, ()) if x in erow} | ({erow[g]} if g in erow else set()) for g in genes]
    nulls = []
    for b in range(N1):
        ic = matched_draw(rng, pool, dec[ib], exr, ib); nulls.append(float(np.median((Za * Ze[ic]).sum(1) / n)))
        if b == 0:
            neg_rho = (Za * Ze[ic]).sum(1) / n
    real = float(np.median(rho)); p = mc_p(real, nulls)
    neg = float(np.median(neg_rho)); neg_p = mc_p(neg, nulls[1:])
    pl = real + 0.05; pl_p = mc_p(pl, nulls)
    H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e); o = np.argsort(-rho)
    s1 = dict(genes=len(genes), cell_lines=len(lines), median_rho=real, null_median=float(np.median(nulls)), null_95=float(np.percentile(nulls, 95)),
              p_mc=p, share_positive=float((rho > 0).mean()), passed=bool(real >= 0.05 and p < 0.05),
              negative=dict(median_rho=neg, p=neg_p, passed=bool(neg >= 0.05 and neg_p < 0.05)), planted=dict(median_rho=pl, p=pl_p, passed=bool(pl >= 0.05 and pl_p < 0.05)),
              top_cover=[dict(gene=sym(genes[i]), substitute=sym(best[genes[i]][0]), identity=best[genes[i]][1], rho=round(float(rho[i]), 3)) for i in o[:25]])
    s1["gate"] = dict(real=s1["passed"], controls_ok=bool(not s1["negative"]["passed"] and s1["planted"]["passed"]))
    s1["gate"]["verdict"] = "PASS" if s1["gate"]["real"] and s1["gate"]["controls_ok"] else "FAIL"
    # P1: predict on EXAM lines
    dev = np.array([H_(c) == 0 for c in lines]); exm = ~dev
    own = np.array([erow.get(g, -1) for g in genes]); has_own = own >= 0
    Xo = np.where(has_own[:, None], Xe[np.maximum(own, 0)], 0.0)

    def fit_score(Y, preds):
        """per gene OLS on DEV lines with the given predictor rows, Pearson r on EXAM lines."""
        k = len(preds) + 1; Xs = np.stack([np.ones_like(Y)] + preds, 2)            # genes x lines x k
        ok = np.isfinite(Y) & dev[None, :]
        Xt = np.where(ok[..., None], Xs, 0); Yt = np.where(ok, Y, 0)
        A = np.einsum("gli,glj->gij", Xt, Xt) + 1e-6 * np.eye(k)[None]; bvec = np.einsum("gli,gl->gi", Xt, Yt)
        w = np.linalg.solve(A, bvec[..., None])[..., 0]; P = np.einsum("glk,gk->gl", Xs, w)
        m = np.isfinite(Y) & exm[None, :]; Pm = np.where(m, P, np.nan); Ym = np.where(m, Y, np.nan)
        pc = Pm - np.nanmean(Pm, 1, keepdims=True); yc = Ym - np.nanmean(Ym, 1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nansum(pc * yc, 1) / np.sqrt(np.nansum(pc ** 2, 1) * np.nansum(yc ** 2, 1)), w
    r0, w0 = fit_score(A_eff, [Xo]); r1, _ = fit_score(A_eff, [Xo, Xe[ib]])
    real_g = gp_better(r1 - r0); ng = []
    for b in range(NP):
        ic = matched_draw(rng, pool, dec[ib], exr, ib); rr, _ = fit_score(A_eff, [Xo, Xe[ic]]); ng.append(float(np.nanmedian(rr - r0)))
        if b == 0:
            negp = gp_better(rr - r0)
    sB = (Xe[ib] - np.nanmean(Xe[ib], 1, keepdims=True)) / (np.nanstd(Xe[ib], 1, keepdims=True) + 1e-6)
    Yp = (w0[:, :1] + w0[:, 1:2] * Xo) + 0.3 * np.nanstd(A_eff, 1, keepdims=True) * sB + rng.normal(0, 1, A_eff.shape) * 0.2 * np.nanstd(A_eff, 1, keepdims=True)
    pr0, _ = fit_score(Yp, [Xo]); pr1, _ = fit_score(Yp, [Xo, Xe[ib]]); plp = gp_better(pr1 - pr0)
    pm = mc_p(real_g["median"], ng)
    p1 = dict(genes=len(genes), dev_lines=int(dev.sum()), exam_lines=int(exm.sum()), r_own_rna_median=float(np.nanmedian(r0)),
              r_with_substitute_median=float(np.nanmedian(r1)), better=real_g, random_gene_gain_95=float(np.percentile(ng, 95)), p_mc=pm,
              passed=bool(real_g["passed"] and pm < 0.05), negative=negp, planted=plp)
    p1["gate"] = dict(real=p1["passed"], controls_ok=bool(not negp["passed"] and plp["passed"]))
    p1["gate"]["verdict"] = "PASS" if p1["gate"]["real"] and p1["gate"]["controls_ok"] else "FAIL"
    per_gene = {g: dict(substitute=best[g][0], identity=best[g][1], depmap_rho=float(rho[i]), exam_r_own=float(r0[i]), exam_r_with_substitute=float(r1[i]))
                for i, g in enumerate(genes)}
    return s1, p1, per_gene


def gp_better(gain):
    g = gx.binom_greater(np.asarray(gain, float)); g["passed"] = bool(g["p"] < 0.05 and g["median"] >= 0.01); return g


# ------------------------------------------------------------------------------------------------ S2
def run_s2(SS, best, fam, rng):
    K4 = sorted(set.intersection(*(set(SS[s]["cols"]) for s in ESS)))
    reach = np.column_stack([SS[s]["sig"][:, [SS[s]["ci"][k] for k in K4]].sum(0) /
                             np.maximum(1, np.isfinite(SS[s]["lfc"][:, [SS[s]["ci"][k] for k in K4]]).sum(0)) for s in ESS])
    ks = [i for i, k in enumerate(K4) if k in best]; Bg = [best[K4[i]][0] for i in ks]
    EB = gx.hpa_expr(Bg); ok = np.all(np.isfinite(EB), 1) & ((np.nanmax(EB, 1) - np.nanmin(EB, 1)) >= 1.0)
    ks = np.array(ks)[ok]; EB = EB[ok]; Rch = reach[ks]
    Rp = gx.pct(reach)[ks]                                                      # reach percentile among all common knockdowns
    allg = sorted({str(g) for g in gx.npz("EXPR_HPA_CELLLINES")["rows"]}); EA = gx.hpa_expr(allg)
    var_ok = np.all(np.isfinite(EA), 1) & ((np.nanmax(EA, 1) - np.nanmin(EA, 1)) >= 1.0); pool = np.nonzero(var_ok)[0]
    Ap = gx.pct(np.where(np.isfinite(EA), EA, 0))                               # expression percentile among all genes, per line
    gi = {g: i for i, g in enumerate(allg)}; Bp = Ap[[gi[best[K4[i]][0]] for i in ks]]
    rs = gx.rowspear(Bp, Rp); real = float(np.nanmean(rs))
    from scipy.stats import rankdata
    ra = rankdata(Bp, axis=1); rb = rankdata(Rp, axis=1); rb = rb - rb.mean(1, keepdims=True); nb = (rb ** 2).sum(1); le = 0
    for _ in range(N2P):
        x = np.take_along_axis(ra, np.argsort(rng.random(ra.shape), axis=1), 1); x = x - x.mean(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            le += np.nanmean((x * rb).sum(1) / np.sqrt((x ** 2).sum(1) * nb)) <= real
    p_perm = float((1 + le) / (1 + N2P))
    excl = [{gi[x] for x in fam.get(K4[i], ()) if x in gi} | ({gi[K4[i]]} if K4[i] in gi else set()) for i in ks]
    nulls = []
    for b in range(N2):
        ic = np.array([next((c for c in rng.choice(pool, 20) if c not in ex), pool[0]) for ex in excl])
        nv = float(np.nanmean(gx.rowspear(Ap[ic], Rp))); nulls.append(nv)
        if b == 0:
            neg = nv
    p_mc = mc_p(real, nulls, greater=False); neg_p = mc_p(neg, nulls[1:], greater=False)
    pl = float(np.nanmean(gx.rowspear(Bp, np.clip(1 - Bp + rng.normal(0, 0.2, Bp.shape), 0, 1))))
    H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e); o = np.argsort(rs)
    out = dict(knockdowns=int(len(ks)), mean_spearman=real, p_permutation=p_perm, null_mean=float(np.mean(nulls)), null_5=float(np.percentile(nulls, 5)),
               p_mc=p_mc, passed=bool(real <= -0.10 and p_perm < 0.05 and p_mc < 0.05),
               negative=dict(mean=neg, p=neg_p, passed=bool(neg <= -0.10 and neg_p < 0.05)), planted=dict(mean=pl, passed=bool(pl <= -0.10)),
               strongest_cover=[dict(gene=sym(K4[ks[i]]), substitute=sym(best[K4[ks[i]]][0]), spearman=float(rs[i]),
                                     reach={CELL[s]: round(float(Rch[i, j]), 3) for j, s in enumerate(ESS)},
                                     substitute_rna={CELL[s]: round(float(EB[i, j]), 2) for j, s in enumerate(ESS)}) for i in o[:15]])
    out["gate"] = dict(real=out["passed"], controls_ok=bool(not out["negative"]["passed"] and out["planted"]["passed"]))
    out["gate"]["verdict"] = "PASS" if out["gate"]["real"] and out["gate"]["controls_ok"] else "FAIL"
    return out


# ------------------------------------------------------------------------------------------------ S3
def run_s3(SS, best, fam, T, rng):
    real_all, per, groups = [], {}, []
    for s in ESS + [GW]:
        S = SS[s]; rows = list(S["rows"]); col = ESS.index(s) if s in ESS else 0
        lvl = gx.hpa_expr(rows)[:, col]; dec = deciles(lvl); pool = {d: np.nonzero(dec == d)[0] for d in range(10)}
        ia, ib, ex = [], [], []
        for k, j in S["ci"].items():
            if k in best and best[k][0] in S["ri"] and np.isfinite(S["own"][j]) and S["own"][j] <= gx.KD_MIN and not cis_pair(T, k, best[k][0]):
                ia.append(j); ib.append(S["ri"][best[k][0]])
                ex.append({S["ri"][x] for x in fam.get(k, ()) if x in S["ri"]} | ({S["ri"][k]} if k in S["ri"] else set()))
        ia, ib = np.array(ia, np.int64), np.array(ib, np.int64)
        v = S["lfc"][ib, ia]; ok = np.isfinite(v); ia, ib, v = ia[ok], ib[ok], v[ok]; ex = [e for e, o in zip(ex, ok) if o]
        groups.append((S, ia, ib, ex, dec, pool, rows)); real_all.append(v)
        per[CELL[s]] = dict(pairs=int(len(v)), median_lfc=float(np.median(v)) if len(v) else None,
                            share_up_link=float(np.mean(S["sig"][ib, ia] & (v > 0))) if len(v) else None)
    real = float(np.median(np.concatenate(real_all))); nulls = []
    for b in range(N3):
        vals = []
        for S, ia, ib, ex, dec, pool, rows in groups:
            ic = matched_draw(rng, pool, dec[ib], ex, ib)
            bad = np.array([cis_pair(T, S["cols"][a], rows[c]) for a, c in zip(ia, ic)], bool)
            vv = S["lfc"][ic, ia]; vals.append(vv[np.isfinite(vv) & ~bad])
        nulls.append(float(np.median(np.concatenate(vals))))
        if b == 0:
            neg = nulls[0]
    base = float(np.mean(nulls[1:])) if len(nulls) > 1 else float(np.mean(nulls))
    diff = real - float(np.mean(nulls)); p = mc_p(real, nulls)
    out = dict(per_screen=per, pairs=int(sum(len(v) for v in real_all)), median_lfc_substitute=real, null_mean=float(np.mean(nulls)),
               null_95=float(np.percentile(nulls, 95)), diff=diff, p_mc=p, passed=bool(diff >= 0.02 and p < 0.05),
               negative=dict(diff=neg - base, passed=bool(neg - base >= 0.02 and mc_p(neg, nulls[1:]) < 0.05)),
               planted=dict(diff=diff + 0.05, passed=bool(diff + 0.05 >= 0.02 and mc_p(real + 0.05, nulls) < 0.05)))
    out["gate"] = dict(real=out["passed"], controls_ok=bool(not out["negative"]["passed"] and out["planted"]["passed"]))
    out["gate"]["verdict"] = "PASS" if out["gate"]["real"] and out["gate"]["controls_ok"] else "FAIL"
    return out


# ------------------------------------------------------------------------------------------------ ledger
def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def declare():
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            if cid in have:
                print(cid, "already declared"); continue
            f.write(json.dumps(dict(op="declare", id=cid, at=cbb.now(), code_sha=code_sha(), commit=gx._git("rev-parse", "HEAD").stdout.strip(),
                                    claim=__doc__.split(" " + cid)[1].split(" G5-")[0].split("Controls:")[0].strip()[:1200], trial=TRIAL),
                               sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G5 already tested")
    t0 = time.time(); os.makedirs(RUN, exist_ok=True); os.makedirs(OUTD, exist_ok=True); rng = np.random.default_rng(SEED)
    best, fam = substitutes(); print(f"substitutes: {len(best):,} genes with a closest paralog >= {MIN_ID:.0f}%", flush=True)
    out = {}
    out["G5-S1"], out["G5-P1"], per_gene = depmap(best, fam, rng); print("G5-S1, P1 done", f"({time.time() - t0:.0f}s)", flush=True)
    T = gx.tss(); keep = None
    if TRIAL:
        tr = np.random.default_rng(1); ess = sorted({str(e) for e in gx.npz("XCELL_K562ESS")["cols_ens"] if str(e)})
        keep = set(tr.choice(ess, 600, replace=False)) | set(tr.choice(sorted({str(e) for e in gx.npz(f"XCELL_{GW}")["cols_ens"] if str(e)}), 800, replace=False))
    SS = {s: gx.load(s, T, keep) for s in ESS + [GW]}
    out["G5-S2"] = run_s2(SS, best, fam, rng); print("G5-S2 done", f"({time.time() - t0:.0f}s)", flush=True)
    out["G5-S3"] = run_s3(SS, best, fam, T, rng); print("G5-S3 done", f"({time.time() - t0:.0f}s)", flush=True)
    if TRIAL:
        print("trial controls ok:", {c: out[c]["gate"]["controls_ok"] for c in CLAIMS}, "; real outcomes not shown"); return
    for cid in CLAIMS:
        print(cid, json.dumps(out[cid], default=float)[:1500], flush=True)
    json.dump(out, open(os.path.join(OUTD, "substitute_summary.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "generegpara:G5", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED)); H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e)
    facts = [Fact(f"CLAIM-{cid}", "substitute_test", out[cid], t[:10], t, "derived", src, inputs=inp) for cid in CLAIMS]
    for g, v in per_gene.items():
        facts.append(Fact(g, "substitute/closest_paralog", dict(substitute=sym(v["substitute"]), substitute_id=v["substitute"], identity=v["identity"],
                                                                depmap_cover_rho=round(v["depmap_rho"], 3)), t[:10], t, "derived", src, inputs=inp,
                          context={"depmap_cover_rho": "Spearman across ~1,100 cancer lines of the substitute's RNA vs the gene's CRISPR effect; positive = covers"}))
    n = cbb.bank().add_many(facts)
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **out[cid]), sort_keys=True, default=float) + "\n")
    s1, s2, s3, p1 = (out[c] for c in CLAIMS)
    Ls = ["=" * 97, "GENEREGPARA -- substitute proteins (closest paralogs): cover, stepping up, prediction (with Monte Carlo)", "=" * 97,
          f"  G5-S1 cover across {s1['cell_lines']:,} cancer lines: {s1['genes']} genes, median rho {s1['median_rho']:.3f} vs random genes "
          f"{s1['null_median']:.3f} (p_MC {s1['p_mc']:.2g}) -> {s1['gate']['verdict']}",
          f"  G5-S2 cover across the four screened cell types: {s2['knockdowns']} knockdowns, mean Spearman {s2['mean_spearman']:.3f} (perm p "
          f"{s2['p_permutation']:.2g}; random genes {s2['null_mean']:.3f}, p_MC {s2['p_mc']:.2g}) -> {s2['gate']['verdict']}",
          f"  G5-S3 stepping up: {s3['pairs']:,} pairs, median lfc of the substitute {s3['median_lfc_substitute']:.3f} vs random {s3['null_mean']:.3f} "
          f"(p_MC {s3['p_mc']:.2g}) -> {s3['gate']['verdict']}",
          f"  G5-P1 predict which lines need a gene (EXAM lines): r own RNA {p1['r_own_rna_median']:.3f} -> with substitute {p1['r_with_substitute_median']:.3f} "
          f"(gain {p1['better']['median']:.3f}, p_MC {p1['p_mc']:.2g}) -> {p1['gate']['verdict']}",
          f"  Facts written {n:,}"]
    open(RES, "w").write("\n".join(Ls) + "\n"); print("\n".join(Ls))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
