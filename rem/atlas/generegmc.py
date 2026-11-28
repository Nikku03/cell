"""GENEREGMC -- Monte Carlo checks on the four G2 verdicts: could chance have produced them, and do they survive re-drawing?

Asked by the user: "what if we run monte carlo method to rule out the randomness".
NOT BLIND: the G2 results (generegx.py; X1 PASS, X2 FAIL, L1 PASS, L2 FAIL) are already known. The draws, thresholds and
stability rule below are fixed in this file before it runs; the G2 verdicts themselves are never rewritten here.
Two kinds of Monte Carlo per claim, on the same data and rules as G2 (cellbrain bank layers via generegx):
 (a) CHANCE (Monte Carlo null): the claim's own statistic recomputed on scrambled data many times;
     p_MC = (1 + number of null draws >= the real value) / (1 + draws).
 (b) STABILITY (re-sampling): 200 runs; each run takes a fresh seed for every random draw in the test (control noise,
     random-dose genes, the link subsample, mediator shuffles) and a bootstrap re-draw of the units (knockdowns; TFs
     for L2; links follow their unit); the declared G2 gate, controls included, is re-applied in every run.
CLAIMS (declared before running; rem/atlas/cellbrain_ledger.jsonl, ids MC-*):
 MC-X1  X1 null: per knockdown, which of the two screens (second K562 experiment, other cell type) counts as "same
        cell type" is a coin flip (its gap changes sign); 10,000 draws per cell type. STABLE PASS if p_MC < 0.05 for
        RPE1, HepG2 and Jurkat and the full X1 gate passes in >= 95% of the 200 runs.
 MC-X2  X2 null: expression's cell-type labels permuted within each knockdown, 10,000 draws. STABLE FAIL if
        p_MC >= 0.05 and the X2 gate fails in >= 95% of the runs.
 MC-L1  L1 null: each link's own-knockdown dose replaced by a random other gene's (usable in the same cell types),
        200 draws over all links; real = the same statistic with the true dose, its random-dose arm redrawn 200 times
        (median taken). STABLE PASS if p_MC < 0.05 and the L1 gate passes in >= 95% of the runs.
 MC-L2  L2 null: mediators' effects on targets shuffled between mediators, 1,000 draws (MH odds ratio each).
        STABLE FAIL if p_MC >= 0.05 and the L2 gate fails in >= 95% of the runs (each run's 20-shuffle control and
        empirical null drawn from the 1,000).
Anything else: UNSTABLE (the verdict leans on chance draws), reported as such. Also reported: 95% intervals
(2.5-97.5 percentiles over the 200 runs) of each claim's main number.
NOT: Monte Carlo rules out chance in the draws and in which genes happened to be in the screens; it does not rule
out a bias shared by the whole dataset (the lab, the cell lines, DESeq2), and it adds no new biology.

Run: python3 -I rem/atlas/generegmc.py declare | test [--trial]
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
TRIAL = "--trial" in sys.argv
_argv = sys.argv; sys.argv = [sys.argv[0]] + (["--trial"] if TRIAL else [])
_s = importlib.util.spec_from_file_location("generegx", os.path.join(HERE, "generegx.py"))
gx = importlib.util.module_from_spec(_s); _s.loader.exec_module(gx)
sys.argv = _argv
cbb, Fact = gx.cbb, gx.Fact

LEDGER = gx.LEDGER
RUN = os.path.join(cbb.HOME, "generegmc_trial" if TRIAL else "generegmc")
OUTD = os.path.join(RUN, "out") if TRIAL else R("outputs", "genereg")
RES = os.path.join(RUN, "RESULTS.txt") if TRIAL else os.path.join(HERE, "RESULTS_generegmc.txt")
CLAIMS = ["MC-X1", "MC-X2", "MC-L1", "MC-L2"]
G2 = {"MC-X1": "PASS", "MC-X2": "FAIL", "MC-L1": "PASS", "MC-L2": "FAIL"}
RUNS, NULL_X, NULL_L1, NULL_L2, SEED = (3, 50, 5, 5, 20261122) if TRIAL else (200, 10_000, 200, 1_000, 20261122)
ESS, GW, CELL = gx.ESS, gx.GW, gx.CELL


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "generegmc.py"), "rb").read()); h.update(gx.code_sha().encode())
    return h.hexdigest()[:16]


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


def verdict(cid, p_mc, gate_share, expect):
    chance_ok = (p_mc < 0.05) if expect == "PASS" else (p_mc >= 0.05)
    return "STABLE " + expect if chance_ok and gate_share >= 0.95 else "UNSTABLE"


# ------------------------------------------------------------------------------------------------ X1
def mc_x1(SS):
    scr = ESS + [GW]
    T_ = sorted(set.intersection(*(set(SS[s]["rows"]) for s in scr))); K = sorted(set.intersection(*(set(SS[s]["cols"]) for s in scr)))
    A = {s: gx.align(SS[s], T_, K, "lfc") for s in scr}; V = {s: gx.align(SS[s], T_, K, "se") ** 2 for s in scr}
    F = np.ones((len(T_), len(K)), bool)
    for s in scr:
        F &= np.isfinite(A[s]) & np.isfinite(V[s])
    for s in scr:
        A[s][~F] = np.nan; V[s][~F] = np.nan
    rels = {s: np.clip(1 - np.nanmean(V[s], 0) / np.nanvar(A[s], 0), 0, None) for s in scr}
    elig = np.all([rels[s] >= gx.REL_MIN for s in scr], 0)
    A = {s: A[s][:, elig] for s in scr}; V = {s: V[s][:, elig] for s in scr}

    def rho(a, b, va, vb):
        r, ra, rb = gx.colstats(a, b, va, vb)
        with np.errstate(invalid="ignore", divide="ignore"):
            return r / np.sqrt(ra * rb)
    ref = "K562ESS"; rho_ref = rho(A[ref], A[GW], V[ref], V[GW]); n = len(rho_ref)
    gaps = {s: rho_ref - rho(A[ref], A[s], V[ref], V[s]) for s in ESS[1:]}
    rng = np.random.default_rng(SEED); null = {}
    for s, g in gaps.items():
        g = np.where(np.isfinite(g), g, 0); real = float(np.median(g))
        sg = rng.choice([-1.0, 1.0], size=(NULL_X, n))
        nm = np.median(sg * g[None, :], 1)
        null[CELL[s]] = dict(real_median_gap=real, null_median_gap_95=float(np.percentile(nm, 95)), p_mc=float((1 + (nm >= real).sum()) / (1 + NULL_X)))
    passes, med = [], {CELL[s]: [] for s in ESS[1:]}
    for b in range(RUNS):
        r = np.random.default_rng(SEED + 1 + b); idx = r.integers(0, n, n); ok_real = ok_ctrl = True
        for s in ESS[1:]:
            g = gx.binom_greater(gaps[s][idx]); med[CELL[s]].append(g["median"]); ok_real &= g["p"] < 0.05 and g["median"] >= 0.05
            noise = r.standard_normal(A[GW].shape).astype(np.float32) * np.sqrt(V[s]); Vn = V[GW] + V[s]
            gn = gx.binom_greater((rho_ref - rho(A[ref], A[GW] + noise, V[ref], Vn))[idx])
            Xp = A[GW].copy()
            for j in range(Xp.shape[1]):
                okr = np.nonzero(np.isfinite(Xp[:, j]))[0]; h = r.choice(okr, len(okr) // 2, replace=False); Xp[h, j] = Xp[r.permutation(h), j]
            gp = gx.binom_greater((rho_ref - rho(A[ref], Xp + noise, V[ref], Vn))[idx])
            ok_ctrl &= not (gn["p"] < 0.05 and gn["median"] >= 0.05) and (gp["p"] < 0.05 and gp["median"] >= 0.05)
        passes.append(bool(ok_real and ok_ctrl))
    share = float(np.mean([p is True for p in passes]))
    p_all = max(v["p_mc"] for v in null.values())
    return dict(knockdowns=n, chance=null, runs=RUNS, gate_pass_share=share, gate_same_as_G2_share=share,
                median_gap_95ci={k: ci(v) for k, v in med.items()}, verdict=verdict("MC-X1", p_all, share, "PASS"))


# ------------------------------------------------------------------------------------------------ X2
def mc_x2(SS, K4):
    reach = np.column_stack([SS[s]["sig"][:, [SS[s]["ci"][k] for k in K4]].sum(0) /
                             np.maximum(1, np.isfinite(SS[s]["lfc"][:, [SS[s]["ci"][k] for k in K4]]).sum(0)) for s in ESS])
    E = gx.hpa_expr(K4); has = np.all(np.isfinite(E), 1); E, Rch = E[has], reach[has]
    Ep, Rp = gx.pct(E), gx.pct(Rch); var = (E.max(1) - E.min(1)) >= 1.0
    rs = gx.rowspear(Ep[var], Rp[var]); ok = np.isfinite(rs); S = float(rs[ok].mean())
    from scipy.stats import rankdata
    ra = rankdata(Ep[var][ok], axis=1); rb = rankdata(Rp[var][ok], axis=1); rb = rb - rb.mean(1, keepdims=True); nb = (rb ** 2).sum(1)
    rng = np.random.default_rng(SEED); nulls = np.empty(NULL_X)
    for i in range(NULL_X):
        x = np.take_along_axis(ra, np.argsort(rng.random(ra.shape), axis=1), 1); x = x - x.mean(1, keepdims=True)
        nulls[i] = ((x * rb).sum(1) / np.sqrt((x ** 2).sum(1) * nb)).mean()
    p_mc = float((1 + (nulls >= S).sum()) / (1 + NULL_X))
    fails, means = [], []
    for b in range(RUNS):
        r = np.random.default_rng(SEED + 1 + b); idx = r.integers(0, len(E), len(E))
        Eb, Rb = E[idx], Rch[idx]; Epb, Rpb = gx.pct(Eb), gx.pct(Rb); vb = (Eb.max(1) - Eb.min(1)) >= 1.0
        real = gx.x2_stat(Epb, Rpb, vb, r); means.append(real["mean_spearman"])
        sh = r.permutation(len(Eb)); Es = Eb[sh]; neg = gx.x2_stat(gx.pct(Es), Rpb, (Es.max(1) - Es.min(1)) >= 1.0, r)
        plant = gx.x2_stat(Epb, np.clip(Epb + r.normal(0, 0.2, Epb.shape), 0, 1), vb, r)
        fails.append(not (real["passed"] and not neg["passed"] and plant["passed"]))
    share = float(np.mean(fails))
    return dict(knockdowns=int(ok.sum()), chance=dict(real_mean_spearman=S, null_95=float(np.percentile(nulls, 95)), p_mc=p_mc), runs=RUNS,
                gate_fail_share=share, gate_same_as_G2_share=share, mean_spearman_95ci=ci(means), verdict=verdict("MC-X2", p_mc, share, "FAIL"))


# ------------------------------------------------------------------------------------------------ L1
def mc_l1(T4, lfc, sig, own):
    usable = np.isfinite(own) & (own <= gx.KD_MIN); S = sig & usable[:, None, :]
    pos = (S & (lfc > 0)).sum(0); neg_ = (S & (lfc < 0)).sum(0); n = S.sum(0)
    ti, ki = np.nonzero((n >= 2) & ((pos == 0) | (neg_ == 0)))
    M = S[:, ti, ki].T; lE = np.log2(np.abs(lfc[:, ti, ki].T)); lK = np.log2(np.abs(own[:, ki].T)); U = usable.T; NK = U.shape[0]

    def rand_k(r, mask, kk):
        rk = r.integers(0, NK, len(mask))
        for _ in range(50):
            bad = ~np.all(U[rk] | ~mask, 1) | (rk == kk)
            if not bad.any():
                break
            rk[bad] = r.integers(0, NK, int(bad.sum()))
        return rk, np.all(U[rk] | ~mask, 1) & (rk != kk)
    lkof = lambda rk: np.log2(np.abs(own[:, rk].T))
    d_dose = gx.masked_sd(lE - lK, M)
    rng = np.random.default_rng(SEED); real_draws, null_draws = [], []
    for _ in range(NULL_L1):
        r1, g1 = rand_k(rng, M, ki); r0, g0 = rand_k(rng, M, ki); ok = g1 & g0
        d1 = gx.masked_sd(lE - lkof(r1), M)
        real_draws.append(float(np.median((d1 - d_dose)[ok]))); null_draws.append(float(np.median((d1 - gx.masked_sd(lE - lkof(r0), M))[ok])))
    real = float(np.median(real_draws)); p_mc = float((1 + sum(x >= real for x in null_draws)) / (1 + NULL_L1))
    cnt_by_k = np.bincount(ki, minlength=NK); passes, meds = [], []
    for b in range(RUNS):
        r = np.random.default_rng(SEED + 1 + b)
        mult = np.bincount(r.integers(0, NK, NK), minlength=NK)[ki]; idx = np.repeat(np.arange(len(ki)), mult)
        if len(idx) > gx.L1_MAX:
            idx = idx[np.sort(r.choice(len(idx), gx.L1_MAX, replace=False))]
        Mb, lEb, lKb, kb = M[idx], lE[idx], lK[idx], ki[idx]
        r1, g1 = rand_k(r, Mb, kb); r2, g2 = rand_k(r, Mb, kb); ok = g1 & g2
        dd = gx.masked_sd(lEb - lKb, Mb); d1 = gx.masked_sd(lEb - lkof(r1), Mb); d2 = gx.masked_sd(lEb - lkof(r2), Mb)
        beta = np.where(Mb, lEb - lKb, 0).sum(1) / Mb.sum(1); lEp = lKb + beta[:, None] + r.normal(0, 0.3, lKb.shape)
        ps = lambda g: g["p"] < 0.05 and g["median"] >= 0.01
        gr = gx.binom_greater((d1 - dd)[ok]); gn = gx.binom_greater((d2 - d1)[ok])
        gp = gx.binom_greater((gx.masked_sd(lEp - lkof(r1), Mb) - gx.masked_sd(lEp - lKb, Mb))[ok])
        meds.append(gr["median"]); passes.append(bool(ps(gr) and not ps(gn) and ps(gp)))
    share = float(np.mean(passes))
    return dict(links=int(len(ki)), knockdowns_with_links=int((cnt_by_k > 0).sum()),
                chance=dict(real_median_gain=real, null_median_gain_95=float(np.percentile(null_draws, 95)), null_max=float(max(null_draws)), p_mc=p_mc),
                runs=RUNS, gate_pass_share=share, gate_same_as_G2_share=share, median_gain_95ci=ci(meds), verdict=verdict("MC-L1", p_mc, share, "PASS"))


# ------------------------------------------------------------------------------------------------ L2
def mc_l2(G):
    CT = gx.npz("REG_COLLECTRI"); ct = set(zip(CT["tf"].astype(str), CT["target"].astype(str))); ct_tfs = set(CT["tf"].astype(str))
    rows = G["rows"]; rowset = set(rows)
    tfs = [r for r in sorted(gx.tf_list() & set(G["cols"]) & ct_tfs) if any((r, t) in ct for t in rowset)]
    lfc, sig, own, ci_, ri = G["lfc"], G["sig"], G["own"], G["ci"], G["ri"]
    med_cols = np.array([ci_[m] for m in G["cols"] if m in ri and np.isfinite(own[ci_[m]]) and own[ci_[m]] <= gx.KD_MIN])
    med_rows = np.array([ri[G["cols"][j]] for j in med_cols])
    rng = np.random.default_rng(SEED); shs = [rng.permutation(len(med_cols)) for _ in range(NULL_L2)]

    def explain(e, tt, j, perm):
        sel = sig[med_rows, j] & (med_cols != j)
        if not sel.any():
            return np.zeros(len(tt), bool)
        mc, mr = med_cols[sel], med_rows[sel]; mce = med_cols[perm][sel] if perm is not None else mc
        P = np.where(sig[np.ix_(tt, mce)], lfc[np.ix_(tt, mce)] * (lfc[mr, j] / own[mc])[None, :], 0)
        return ((np.sign(P) == np.sign(e)[:, None]) & (np.abs(P) >= 0.5 * np.abs(e)[:, None])).any(1)
    ind, IS, unit, zz, aa, inct = [], [], [], [], [], []
    for u, r in enumerate(tfs):
        j = ci_[r]; tt = np.nonzero(sig[:, j])[0]
        if not len(tt):
            continue
        e = lfc[tt, j]; ind.append(explain(e, tt, j, None)); IS.append(np.stack([explain(e, tt, j, sh) for sh in shs]))
        unit.append(np.full(len(tt), u)); zz.append(np.abs(e / G["se"][tt, j])); aa.append(np.abs(e)); inct.append(np.array([(r, rows[t]) in ct for t in tt]))
    ind, IS, unit, zz, aa, inct = (np.concatenate(ind), np.concatenate(IS, 1), np.concatenate(unit), np.concatenate(zz), np.concatenate(aa), np.concatenate(inct))
    q5 = lambda v: np.searchsorted(np.quantile(v, [0.2, 0.4, 0.6, 0.8]), v)
    st = q5(zz) * 5 + q5(aa); OR, p = gx.cmh(~ind, inct, st)
    sh_or = np.array([gx.cmh(~IS[b], inct, st)[0] for b in range(NULL_L2)])
    p_mc = float((1 + (sh_or >= OR).sum()) / (1 + NULL_L2))
    fails, ors = [], []
    for b in range(RUNS):
        r = np.random.default_rng(SEED + 1 + b); nu = len(tfs)
        mult = np.bincount(r.integers(0, nu, nu), minlength=nu)[unit]; idx = np.repeat(np.arange(len(ind)), mult)
        stb = q5(zz[idx]) * 5 + q5(aa[idx]); ob, pb = gx.cmh(~ind[idx], inct[idx], stb); ors.append(ob)
        pick = r.choice(NULL_L2, min(20, NULL_L2), replace=False); so = [gx.cmh(~IS[k][idx], inct[idx], stb)[0] for k in pick]
        gm = float(np.exp(np.mean(np.log(np.clip(so, 1e-6, 1e6))))); pe = (1 + sum(x >= ob for x in so)) / (1 + len(so))
        real_ok = ob >= 1.5 and pb < 0.05 and pe < 0.05; ctrl_ok = 0.67 <= gm <= 1.5      # planted is fixed by construction (100% in G2)
        fails.append(not (real_ok and ctrl_ok))
    share = float(np.mean(fails))
    return dict(tfs=len(tfs), links=int(len(ind)), chance=dict(real_mh_or=float(OR), cmh_p=float(p), null_or_95=float(np.percentile(sh_or, 95)),
                null_or_median=float(np.median(sh_or)), p_mc=p_mc), runs=RUNS, gate_fail_share=share, gate_same_as_G2_share=share,
                mh_or_95ci=ci(ors), verdict=verdict("MC-L2", p_mc, share, "FAIL"))


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
                                    claim=__doc__.split(" " + cid)[1].split(" MC-")[0].split("Anything else")[0].strip()[:1000], trial=TRIAL,
                                    blind=False), sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("MC already tested")
    t0 = time.time(); os.makedirs(RUN, exist_ok=True); os.makedirs(OUTD, exist_ok=True)
    T = gx.tss(); keep = None
    if TRIAL:
        tr = np.random.default_rng(1)
        ess = sorted({str(e) for e in gx.npz("XCELL_K562ESS")["cols_ens"] if str(e)}); gw = sorted({str(e) for e in gx.npz(f"XCELL_{GW}")["cols_ens"] if str(e)})
        keep = set(tr.choice(ess, 500, replace=False)) | set(tr.choice(sorted(gx.tf_list() & set(gw)), 200, replace=False))
    SS = {s: gx.load(s, T, keep) for s in ESS + [GW]}
    out = {}
    out["MC-X1"] = mc_x1(SS); print("MC-X1 done", f"({time.time() - t0:.0f}s)", flush=True)
    K4 = sorted(set.intersection(*(set(SS[s]["cols"]) for s in ESS)))
    out["MC-X2"] = mc_x2(SS, K4); print("MC-X2 done", f"({time.time() - t0:.0f}s)", flush=True)
    T4, lfc, se, sig, own = gx.stack4(SS, K4); del se
    out["MC-L1"] = mc_l1(T4, lfc, sig, own); print("MC-L1 done", f"({time.time() - t0:.0f}s)", flush=True)
    del lfc, sig
    out["MC-L2"] = mc_l2(SS[GW]); print("MC-L2 done", f"({time.time() - t0:.0f}s)", flush=True)
    if TRIAL:
        print("trial done; outcomes not shown"); return
    for cid in CLAIMS:
        print(cid, json.dumps(out[cid], default=float)[:1500], flush=True)
    json.dump(dict(blind=False, g2_verdicts=G2, **out), open(os.path.join(OUTD, "montecarlo_summary.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "generegmc:G2-MC", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED))
    n = cbb.bank().add_many([Fact(f"CLAIM-G2-{cid[3:]}", "montecarlo", out[cid], t[:10], t, "derived", src, inputs=inp,
                                  context={"g2_verdict": G2[cid], "blind": False}) for cid in CLAIMS])
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), g2_verdict=G2[cid], **out[cid]),
                               sort_keys=True, default=float) + "\n")
    x1, x2, l1, l2 = (out[c] for c in CLAIMS)
    Ls = ["=" * 97, "GENEREGMC -- Monte Carlo checks on the G2 verdicts (not blind: G2 results were known)", "=" * 97,
          f"  MC-X1 (G2 PASS): chance p_MC " + ", ".join(f"{k} {v['p_mc']:.2g}" for k, v in x1["chance"].items()) +
          f"; gate passed in {x1['gate_pass_share']:.0%} of {x1['runs']} re-drawn runs -> {x1['verdict']}",
          f"  MC-X2 (G2 FAIL): mean Spearman {x2['chance']['real_mean_spearman']:.3f} vs chance 95th pct {x2['chance']['null_95']:.3f}, p_MC "
          f"{x2['chance']['p_mc']:.2g}; gate failed in {x2['gate_fail_share']:.0%} of runs -> {x2['verdict']}",
          f"  MC-L1 (G2 PASS): median gain {l1['chance']['real_median_gain']:.3f} vs chance 95th pct {l1['chance']['null_median_gain_95']:.3f}, p_MC "
          f"{l1['chance']['p_mc']:.2g}; gate passed in {l1['gate_pass_share']:.0%} of runs (gain 95% {l1['median_gain_95ci']}) -> {l1['verdict']}",
          f"  MC-L2 (G2 FAIL): OR {l2['chance']['real_mh_or']:.2f} vs shuffled 95th pct {l2['chance']['null_or_95']:.2f}, p_MC {l2['chance']['p_mc']:.2g}; "
          f"gate failed in {l2['gate_fail_share']:.0%} of runs (OR 95% {l2['mh_or_95ci']}) -> {l2['verdict']}",
          f"  Facts written {n}"]
    open(RES, "w").write("\n".join(Ls) + "\n"); print("\n".join(Ls))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
