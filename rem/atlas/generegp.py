"""GENEREGP -- use what is not random to PREDICT: a knockdown's effects in a cell type not measured, and the next step.

Asked by the user: "now we know what is not random but can be predicted; see how we can use this to predict the next
step or cell data". Built on what held up in G2 and its Monte Carlo check: a knockdown's effects carry over between
cell types in part, more between similar cells (K562 ~ Jurkat), and its effects can pass through other genes.
Data: the cellbrain layers XCELL_* (same ~2,400 knockdowns in K562, RPE1, HepG2, Jurkat; K562 genome-wide ~9,860),
log2 fold change per measured gene, links and cis masking exactly as in generegx (Wald BH q < 0.05; 50 kb).
Splits (house rule): knockdowns DEV / EXAM by sha256('cellbrain-c1:' + Ensembl) parity. Models are fit on DEV
knockdowns only and scored ONCE on EXAM. (Trial: fit on half of DEV, scored on the other half; EXAM untouched.)
Scored knockdowns: >= 20 links in the cell type predicted (a clear effect to predict). Score per knockdown = Pearson r
between predicted and measured log2 fold changes over the genes measured. GENERIC = the mean DEV-knockdown profile in
the cell type predicted ("what any knockdown does there"): the baseline every model must beat.
BETTER (house rule): per knockdown gain in r over the comparison, one-sided sign test p < 0.05 AND median gain >= +0.01.

CLAIMS (declared before the EXAM run; rem/atlas/cellbrain_ledger.jsonl, ids G3-*):
 G3-T1  CELL DATA FROM THE OTHER CELL TYPES. For each cell type in turn held out, its profile for a knockdown =
        a x generic + sum of w_c x that knockdown's profile in each of the other three cell types (least squares,
        fit on DEV). CLAIM: BETTER than generic in all four cell types. Negative (must not be BETTER in any): the
        other cell types' profiles taken from a different knockdown (labels deranged), refit. Planted (must be BETTER
        in all): the held-out profile replaced by the mean of the other three plus noise of its own lfcSE.
 G3-T2  CELL DATA FROM K562 ALONE (the genome-wide screen; how to predict knockdowns never screened elsewhere). For
        RPE1, HepG2, Jurkat: profile = a x generic + b x the knockdown's K562 genome-wide profile, fit on DEV.
        CLAIM: BETTER than generic in all three. Same negative and planted controls (single source).
        Only where this passes: predictions for every K562 genome-wide knockdown with >= 20 links that the other
        cell type never screened are written to the bank as predictions (top 30 genes, with the EXAM accuracy).
 G3-C1  THE NEXT STEP (K562 genome-wide). Genes split by sha256('generegp-io:' + Ensembl) parity into INPUT genes
        (also knocked down in the screen, own knockdown <= -0.5) and OUTPUT genes. For knockdown A, its next step on
        the output genes = sum over input genes m that A changes (links) of [lfc(m | A) / own(m)] x lfc(outputs | m),
        i.e. what A does to m, times what m does. Model = a x generic + b x next step, fit on DEV; scored on knockdowns
        with >= 20 output links and >= 5 input links. CLAIM: BETTER than generic AND BETTER than the same model with
        the input genes' profiles shuffled between input genes (which gene is the go-between matters). Negative
        (must not be BETTER than generic): that shuffled model. Planted (must be BETTER than generic): outputs
        replaced by the next-step prediction (scaled) plus noise of their lfcSE.
DESCRIPTIVE: learned weights (which cell types inform which); the K562 two-experiment ceiling; per cell type the
median r of each model; predicted top genes for unscreened knockdowns in RPE1, HepG2, Jurkat (where G3-T2 passes).
NOT: predictions of pattern (correlation), not of exact size; four cell lines; the essential-gene library; a
prediction is not a measurement -- bank facts carry tier "derived" and the word prediction.

Run: python3 -I rem/atlas/generegp.py declare | test [--trial]
"""

from __future__ import annotations
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
_argv = sys.argv; sys.argv = [sys.argv[0]] + (["--trial"] if TRIAL else [])
_s = importlib.util.spec_from_file_location("generegx", os.path.join(HERE, "generegx.py"))
gx = importlib.util.module_from_spec(_s); _s.loader.exec_module(gx)
sys.argv = _argv
cbb, Fact = gx.cbb, gx.Fact

LEDGER = gx.LEDGER
RUN = os.path.join(cbb.HOME, "generegp_trial" if TRIAL else "generegp")
OUTD = os.path.join(RUN, "out") if TRIAL else R("outputs", "genereg")
RES = os.path.join(RUN, "RESULTS.txt") if TRIAL else os.path.join(HERE, "RESULTS_generegp.txt")
CLAIMS = ["G3-T1", "G3-T2", "G3-C1"]
ESS, GW, CELL = gx.ESS, gx.GW, gx.CELL
MIN_LINKS, MIN_INPUTS, TOP, SEED = 20, 5, 30, 20261123
H = lambda tag, e: int(hashlib.sha256((tag + e).encode()).hexdigest(), 16) % 2
SPLIT = lambda e: H("cellbrain-c1:", e)          # 0 DEV, 1 EXAM
SUB = lambda e: H("generegp-dev:", e)            # trial: DEV-A fits, DEV-B scores
IO = lambda e: H("generegp-io:", e)              # cascade: 0 input genes, 1 output genes


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "generegp.py"), "rb").read()); h.update(gx.code_sha().encode())
    return h.hexdigest()[:16]


def roles(K):
    K = np.asarray(K)
    if TRIAL:
        tr = np.array([SPLIT(k) == 0 and SUB(k) == 0 for k in K]); sc = np.array([SPLIT(k) == 0 and SUB(k) == 1 for k in K])
    else:
        tr = np.array([SPLIT(k) == 0 for k in K]); sc = ~tr
    return tr, sc


def colr(P, O):
    m = np.isfinite(O) & np.isfinite(P); P = np.where(m, P, np.nan); O = np.where(m, O, np.nan)
    pm = P - np.nanmean(P, 0); om = O - np.nanmean(O, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(pm * om, 0) / np.sqrt(np.nansum(pm ** 2, 0) * np.nansum(om ** 2, 0))


def fit(F, O):
    """pooled least squares O ~ sum_i w_i F_i over finite O (features: NaN -> 0)."""
    m = np.isfinite(O)
    A = np.column_stack([np.nan_to_num(f)[m] for f in F]); w, *_ = np.linalg.lstsq(A, O[m], rcond=None)
    return w


def pred(F, w):
    return sum(wi * np.nan_to_num(f) for wi, f in zip(w, F))


def better(gain):
    g = gx.binom_greater(np.asarray(gain, float)); g["passed"] = bool(g["p"] < 0.05 and g["median"] >= 0.01); return g


def derange(n, rng):
    p = rng.permutation(n)
    while np.any(p == np.arange(n)):
        bad = np.nonzero(p == np.arange(n))[0]; sw = rng.integers(0, n, len(bad)); p[bad], p[sw] = p[sw], p[bad].copy()
    return p


def transfer(O, SE, srcs, tr, sc, rng):
    """O, SE: genes x knockdowns (target cell type); srcs: list of genes x knockdowns source profiles."""
    gen = np.nanmean(O[:, tr], 1)[:, None] * np.ones((1, O.shape[1]), np.float32)
    F = [gen] + srcs; w = fit([f[:, tr] for f in F], O[:, tr])
    r0 = colr(gen[:, sc], O[:, sc]); r1 = colr(pred([f[:, sc] for f in F], w), O[:, sc])
    real = better(r1 - r0)
    perm = derange(O.shape[1], rng); Fn = [gen] + [s[:, perm] for s in srcs]; wn = fit([f[:, tr] for f in Fn], O[:, tr])
    neg = better(colr(pred([f[:, sc] for f in Fn], wn), O[:, sc]) - r0)
    Op = np.nanmean(np.stack(srcs), 0) + rng.standard_normal(O.shape).astype(np.float32) * np.nan_to_num(SE, nan=0.3)
    Op[~np.isfinite(O)] = np.nan
    genp = np.nanmean(Op[:, tr], 1)[:, None] * np.ones((1, O.shape[1]), np.float32); Fp = [genp] + srcs
    wp = fit([f[:, tr] for f in Fp], Op[:, tr])
    plant = better(colr(pred([f[:, sc] for f in Fp], wp), Op[:, sc]) - colr(genp[:, sc], Op[:, sc]))
    return dict(knockdowns_fit=int(tr.sum()), knockdowns_scored=int(sc.sum()), weights=[float(x) for x in w],
                r_generic_median=float(np.nanmedian(r0)), r_model_median=float(np.nanmedian(r1)), real=real, negative=neg, planted=plant), w, gen


# ------------------------------------------------------------------------------------------------ T1, T2
def run_t1(SS, rng):
    T4 = sorted(set.intersection(*(set(SS[s]["rows"]) for s in ESS))); K4 = sorted(set.intersection(*(set(SS[s]["cols"]) for s in ESS)))
    P = {s: gx.align(SS[s], T4, K4, "lfc") for s in ESS}; out = {}
    for X in ESS:
        links = gx.align(SS[X], T4, K4, "sig").sum(0); ok = links >= MIN_LINKS; tr, sc = roles(K4); tr &= ok; sc &= ok
        srcs = [s for s in ESS if s != X]
        r, w, _ = transfer(P[X], gx.align(SS[X], T4, K4, "se"), [P[s] for s in srcs], tr, sc, rng)
        r["weights"] = dict(zip(["generic"] + [CELL[s] for s in srcs], r["weights"])); out[CELL[X]] = r
    # ceiling: the second K562 experiment as the "prediction" of K562 essential (same scored knockdowns)
    K = [k for k in K4 if k in SS[GW]["ci"]]; Tg = [t for t in T4 if t in SS[GW]["ri"]]
    O = gx.align(SS["K562ESS"], Tg, K, "lfc"); Gp = gx.align(SS[GW], Tg, K, "lfc"); tr, sc = roles(K)
    sc &= gx.align(SS["K562ESS"], Tg, K, "sig").sum(0) >= MIN_LINKS
    ceiling = float(np.nanmedian(colr(Gp[:, sc], O[:, sc])))
    ok_real = all(v["real"]["passed"] for v in out.values())
    ok_ctrl = not any(v["negative"]["passed"] for v in out.values()) and all(v["planted"]["passed"] for v in out.values())
    return dict(per_cell=out, k562_two_experiment_ceiling_r=ceiling, genes=len(T4),
                gate=dict(real=ok_real, controls_ok=ok_ctrl, verdict="PASS" if ok_real and ok_ctrl else "FAIL"))


def run_t2(SS, rng):
    out, models = {}, {}
    for X in ESS[1:]:
        T = sorted(set(SS[X]["rows"]) & set(SS[GW]["rows"])); K = sorted(set(SS[X]["cols"]) & set(SS[GW]["cols"]))
        O = gx.align(SS[X], T, K, "lfc"); Gp = gx.align(SS[GW], T, K, "lfc")
        links = gx.align(SS[X], T, K, "sig").sum(0); tr, sc = roles(K); tr &= links >= MIN_LINKS; sc &= links >= MIN_LINKS
        r, w, gen = transfer(O, gx.align(SS[X], T, K, "se"), [Gp], tr, sc, rng)
        r["weights"] = dict(zip(["generic", "K562 genome-wide"], r["weights"])); out[CELL[X]] = r; models[X] = (T, w, gen[:, 0])
    ok_real = all(v["real"]["passed"] for v in out.values())
    ok_ctrl = not any(v["negative"]["passed"] for v in out.values()) and all(v["planted"]["passed"] for v in out.values())
    return dict(per_cell=out, gate=dict(real=ok_real, controls_ok=ok_ctrl, verdict="PASS" if ok_real and ok_ctrl else "FAIL")), models


def predictions(SS, models, t2):
    """For cell types where T2 passed: predicted profiles of K562 genome-wide knockdowns that cell type never screened."""
    G = SS[GW]; sym = gx.sym_map(); gl = G["sig"].sum(0); res = {}
    for X, (T, w, gen) in models.items():
        if not (t2["per_cell"][CELL[X]]["real"]["passed"] and t2["gate"]["controls_ok"]):
            continue
        ri = np.array([G["ri"][t] for t in T]); unscreened = [j for j, c in enumerate(G["cols"]) if c not in SS[X]["ci"] and gl[j] >= MIN_LINKS]
        acc = t2["per_cell"][CELL[X]]["r_model_median"]; d = {}
        for j in unscreened:
            p = w[0] * gen + w[1] * np.nan_to_num(G["lfc"][ri, j]); o = np.argsort(-np.abs(p))[:TOP]
            d[G["cols"][j]] = dict(symbol=sym(G["cols"][j]), top=[[sym(T[i]), round(float(p[i]), 3)] for i in o])
        res[CELL[X]] = dict(exam_median_r=acc, knockdowns=len(d), predicted=d)
    return res


# ------------------------------------------------------------------------------------------------ C1
def run_c1(SS, rng):
    G = SS[GW]; lfc, sig, own, ri, ci = G["lfc"], G["sig"], G["own"], G["ri"], G["ci"]
    inputs = [m for m in G["cols"] if IO(m) == 0 and m in ri and np.isfinite(own[ci[m]]) and own[ci[m]] <= gx.KD_MIN]
    outr = np.array([i for i, g in enumerate(G["rows"]) if IO(g) == 1])
    in_r = np.array([ri[m] for m in inputs]); in_c = np.array([ci[m] for m in inputs])
    with np.errstate(invalid="ignore", divide="ignore"):
        W = np.where(sig[in_r, :], lfc[in_r, :] / own[in_c][:, None], 0).astype(np.float32)
    W = np.nan_to_num(W)
    LO = np.nan_to_num(lfc[np.ix_(outr, in_c)])                       # outputs x inputs: what each go-between does
    C = LO @ W; O = lfc[outr, :]; SE = G["se"][outr, :]
    olinks = sig[outr, :].sum(0); nin = (W != 0).sum(0)
    tr, sc = roles(G["cols"]); ok = (olinks >= MIN_LINKS) & (nin >= MIN_INPUTS); tr &= ok; sc &= ok
    gen = np.nanmean(O[:, tr], 1)[:, None] * np.ones((1, O.shape[1]), np.float32)
    w = fit([gen[:, tr], C[:, tr]], O[:, tr]); r0 = colr(gen[:, sc], O[:, sc]); r1 = colr(pred([gen[:, sc], C[:, sc]], w), O[:, sc])
    perm = derange(len(inputs), rng); Cs = LO[:, perm] @ W
    ws = fit([gen[:, tr], Cs[:, tr]], O[:, tr]); rs = colr(pred([gen[:, sc], Cs[:, sc]], ws), O[:, sc])
    real_g, real_s, neg = better(r1 - r0), better(r1 - rs), better(rs - r0)
    s = np.nanmedian(np.nanstd(O[:, tr], 0) / np.maximum(np.nanstd(C[:, tr], 0), 1e-9))
    Op = C * s + rng.standard_normal(O.shape).astype(np.float32) * np.nan_to_num(SE, nan=0.3); Op[~np.isfinite(O)] = np.nan
    genp = np.nanmean(Op[:, tr], 1)[:, None] * np.ones((1, O.shape[1]), np.float32); wp = fit([genp[:, tr], C[:, tr]], Op[:, tr])
    plant = better(colr(pred([genp[:, sc], C[:, sc]], wp), Op[:, sc]) - colr(genp[:, sc], Op[:, sc]))
    sym = gx.sym_map(); gain = r1 - r0; scj = np.nonzero(sc)[0]; o = np.argsort(-np.nan_to_num(gain, nan=-9))[:15]
    best = [dict(knockdown=sym(G["cols"][scj[i]]), r_generic=round(float(r0[i]), 3), r_next_step=round(float(r1[i]), 3), inputs_changed=int(nin[scj[i]]))
            for i in o]
    out = dict(input_genes=len(inputs), output_genes=int(len(outr)), knockdowns_fit=int(tr.sum()), knockdowns_scored=int(sc.sum()),
               weights=dict(generic=float(w[0]), next_step=float(w[1])), r_generic_median=float(np.nanmedian(r0)), r_model_median=float(np.nanmedian(r1)),
               r_shuffled_median=float(np.nanmedian(rs)), real_vs_generic=real_g, real_vs_shuffled=real_s, negative=neg, planted=plant, most_helped=best)
    ok_real = real_g["passed"] and real_s["passed"]; ok_ctrl = (not neg["passed"]) and plant["passed"]
    out["gate"] = dict(real=bool(ok_real), controls_ok=bool(ok_ctrl), verdict="PASS" if ok_real and ok_ctrl else "FAIL")
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
                                    claim=__doc__.split(" " + cid)[1].split(" G3-")[0].split("DESCRIPTIVE")[0].strip()[:1200], trial=TRIAL),
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
            raise SystemExit("G3 already tested")
    t0 = time.time(); os.makedirs(RUN, exist_ok=True); os.makedirs(OUTD, exist_ok=True); rng = np.random.default_rng(SEED)
    T = gx.tss(); SS = {s: gx.load(s, T) for s in ESS + [GW]}
    t1 = run_t1(SS, rng); print("G3-T1", json.dumps(t1, default=float)[:2500], f"({time.time() - t0:.0f}s)", flush=True)
    t2, models = run_t2(SS, rng); print("G3-T2", json.dumps(t2, default=float)[:2000], f"({time.time() - t0:.0f}s)", flush=True)
    c1 = run_c1(SS, rng); print("G3-C1", json.dumps(c1, default=float)[:2500], f"({time.time() - t0:.0f}s)", flush=True)
    stage = "DEV (fit on DEV-A, scored on DEV-B)" if TRIAL else "EXAM"
    out = dict(stage=stage, t1=t1, t2=t2, c1=c1)
    json.dump(out, open(os.path.join(OUTD, "predict_summary.json"), "w"), indent=1, default=float)
    if TRIAL:
        print("trial done (DEV only; EXAM untouched)"); return
    P = predictions(SS, models, t2)
    with gzip.open(os.path.join(OUTD, "predict_from_k562gw.json.gz"), "wt") as f:
        json.dump(P, f, default=float)
    t = cbb.now(); src = {"name": "generegp:G3", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED)); facts = []
    for cid, r in (("G3-T1", t1), ("G3-T2", t2), ("G3-C1", c1)):
        facts.append(Fact(f"CLAIM-{cid}", "prediction_test", r, t[:10], t, "derived", src, inputs=inp))
    for cell, d in P.items():
        for g, v in d["predicted"].items():
            facts.append(Fact(g, f"predict/knockdown_effect_{cell}", v["top"], t[:10], t, "derived", src, inputs=inp,
                              context={"kind": "prediction, not a measurement", "from": "K562 genome-wide profile", "exam_median_r": d["exam_median_r"]}))
    n = cbb.bank().add_many(facts)
    for cid, r in (("G3-T1", t1), ("G3-T2", t2), ("G3-C1", c1)):
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **r), sort_keys=True, default=float) + "\n")
    f1 = lambda d: ", ".join(f"{c} {v['r_generic_median']:.2f}->{v['r_model_median']:.2f}" for c, v in d["per_cell"].items())
    Ls = ["=" * 97, "GENEREGP -- predicting a knockdown's effects in an unmeasured cell type, and the next step (EXAM)", "=" * 97,
          f"  G3-T1 from the other three cell types (median r, generic -> model): {f1(t1)}; K562 two-experiment ceiling "
          f"{t1['k562_two_experiment_ceiling_r']:.2f} -> {t1['gate']['verdict']}",
          f"  G3-T2 from K562 genome-wide alone: {f1(t2)} -> {t2['gate']['verdict']}",
          f"  G3-C1 next step (K562): generic {c1['r_generic_median']:.2f}, with next step {c1['r_model_median']:.2f}, shuffled go-betweens "
          f"{c1['r_shuffled_median']:.2f} -> {c1['gate']['verdict']}",
          f"  Predictions written: " + ", ".join(f"{c} {d['knockdowns']:,} knockdowns" for c, d in P.items()) + f"; facts {n:,}"]
    open(RES, "w").write("\n".join(Ls) + "\n"); print("\n".join(Ls))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
