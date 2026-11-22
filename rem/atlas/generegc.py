"""GENEREGC -- one predictor from all three clues (strength, partners, substitutes), tested in two arenas.

Asked by the user: "now combine all three into one predictor and test it" (after G3 strength/transfer, G4 partners in
the same room, G5 substitutes). Splits as before (house rule): fit on DEV, scored once on EXAM.
ARENA A -- WHAT A KNOCKDOWN DOES IN A CELL TYPE (Perturb-seq; K562, RPE1, HepG2, Jurkat; knockdowns DEV/EXAM by
 sha256('cellbrain-c1:' + Ensembl) parity; scored: EXAM knockdowns with >= 20 links in that cell type; score = Pearson r
 of predicted vs measured log2 fold changes). Clues for knockdown A in cell type X:
   STRENGTH  how hard A hits elsewhere x X's generic profile, and A's specific patterns in the other three (G3-T1);
   PARTNERS  mean pattern in X of A's interaction partners made in X, sharing a room, clear in X, not within 100 kb (G4);
   SUBSTITUTE the pattern in X of A's closest paralog (>= 30% identity; if knocked down there, clear, not within
             100 kb), and a cover term: strength x generic x (substitute's RNA percentile in X - its mean in the other
             three) (G5). Missing clues are zero.
 One least-squares model per cell type, fit on DEV knockdowns. Comparisons: GENERIC (X's mean DEV profile) and G3
 (STRENGTH only). Ablations drop one clue.
ARENA B -- WHICH CANCER LINES NEED A GENE (DepMap 24Q4 CRISPR gene effect, 1,103 lines with RNA; lines DEV/EXAM by
 sha256('cellbrain-c1:' + ModelID) parity; genes: needed in >= 5% of lines (effect < -0.5); score per gene = Pearson
 r of predicted vs measured effect over EXAM lines). Per gene, least squares on DEV lines with:
   BASELINE  1 + the gene's own RNA;
   STRENGTH  the line's overall sensitivity (mean effect of all scored genes but this one, in that line);
   PARTNERS  mean effect, in that line, of the gene's partners that share a room with it and are made in that line
             (RNA log2(TPM+1) >= 1); 0 where none;
   SUBSTITUTE the closest paralog's RNA in that line (G5-P1).
POOLING (disclosed before running): the user asked for ONE predictor, so the primary test in arena A pools all four
cell types (each scored knockdown in each cell type is one item); per-cell-type results are reported. RPE1 has missed
the per-cell bar in G3 and G4 (little beyond the shared stress response there), so a per-cell "all four" form was not
chosen as primary; this choice is made knowing that.
BETTER (house rule): sign test over items p < 0.05 AND median gain >= +0.01.
MONTE CARLO (200 draws): partners swapped for random genes in a shared room (same number; made and clear where the arena
asks), the substitute swapped for a random gene (arena A: a random clear knockdown in X outside A's family; arena B: a
gene from the substitute's RNA-level decile outside A's family); refit and rescored each draw.

CLAIMS (declared before running; rem/atlas/cellbrain_ledger.jsonl, ids G6-*):
 G6-A1  ONE PREDICTOR, SCREENS. Pooled over the four cell types, the combined model is BETTER than GENERIC and
        BETTER than G3.
 G6-A2  EACH CLUE ADDS, SCREENS. Pooled: combined BETTER than each ablation (no STRENGTH, no PARTNERS, no
        SUBSTITUTE); and for PARTNERS and SUBSTITUTE, Monte Carlo p < 0.05 (the real clue's median gain over its
        ablation vs the 200 random-clue gains).
 G6-B1  ONE PREDICTOR, CANCER LINES. Combined BETTER than BASELINE over genes, and Monte Carlo p < 0.05 with random
        partners and substitute together.
 G6-B2  EACH CLUE ADDS, CANCER LINES. Combined BETTER than each ablation (no STRENGTH, no PARTNERS, no SUBSTITUTE);
        PARTNERS and SUBSTITUTE Monte Carlo p < 0.05 each.
Controls: negative = the first random draw's combined model must not be BETTER than its ablation (for each random
clue); planted = in arena A the partner clue replaced by the knockdown's own pattern plus noise, in arena B the effect
replaced by the DEV-fitted baseline plus the partner clue (scaled) plus noise, must be BETTER than its ablation.
NOT: predicts pattern (correlation), not exact size; arena B uses other genes measured in the same line (as in a
genome-wide screen where one gene is missing); four cell lines; RNA stands in for protein; no timing.

Run: python3 -I rem/atlas/generegc.py declare | test [--trial]
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


def _load(name):
    argv = sys.argv; sys.argv = [sys.argv[0]] + (["--trial"] if TRIAL else [])
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


pp = _load("generegppi"); pa = _load("generegpara")
gp, gx = pp.gp, pp.gx
cbb, Fact = gx.cbb, gx.Fact
LEDGER = gx.LEDGER
RUN = os.path.join(cbb.HOME, "generegc_trial" if TRIAL else "generegc")
OUTD = os.path.join(RUN, "out") if TRIAL else R("outputs", "genereg")
RES = os.path.join(RUN, "RESULTS.txt") if TRIAL else os.path.join(HERE, "RESULTS_generegc.txt")
CLAIMS = ["G6-A1", "G6-A2", "G6-B1", "G6-B2"]
ESS, CELL = gx.ESS, gx.CELL
MIN_LINKS, SEED = 20, 20261126
NMC = 10 if TRIAL else 200
H_ = lambda e: int(hashlib.sha256(("cellbrain-c1:" + e).encode()).hexdigest(), 16) % 2


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "generegc.py"), "rb").read())
    for m in (pp, pa):
        h.update(m.code_sha().encode())
    return h.hexdigest()[:16]


def better(gain):
    g = gx.binom_greater(np.asarray(gain, float)); g["passed"] = bool(g["p"] < 0.05 and g["median"] >= 0.01); return g


def mc_p(real, nulls):
    nulls = np.asarray(nulls, float); return float((1 + (nulls >= real).sum()) / (1 + len(nulls)))


class Fitter:
    """least squares on masked training entries with cached cross products of the fixed clues."""

    def __init__(self, fixed, O, tr):
        self.m = np.isfinite(O[:, tr]); self.O = np.where(self.m, O[:, tr], 0).astype(np.float32); self.tr = tr
        self.F = [np.where(self.m, np.nan_to_num(f[:, tr]), 0).astype(np.float32) for f in fixed]
        k = len(self.F); self.G = np.array([[float((self.F[i] * self.F[j]).sum()) for j in range(k)] for i in range(k)])
        self.b = np.array([float((f * self.O).sum()) for f in self.F])

    def fit(self, use, extra=()):
        X = [np.where(self.m, np.nan_to_num(e[:, self.tr]), 0).astype(np.float32) for e in extra]
        use = list(use); k = len(use) + len(X); A = np.zeros((k, k)); b = np.zeros(k)
        A[:len(use), :len(use)] = self.G[np.ix_(use, use)]; b[:len(use)] = self.b[use]
        for a, x in enumerate(X):
            i = len(use) + a
            for c, j in enumerate(use):
                A[i, c] = A[c, i] = float((x * self.F[j]).sum())
            for a2, x2 in enumerate(X[:a + 1]):
                A[i, len(use) + a2] = A[len(use) + a2, i] = float((x * x2).sum())
            b[i] = float((x * self.O).sum())
        return np.linalg.solve(A + 1e-9 * np.eye(k), b)


def score(F, w, O, sc):
    return gp.colr(sum(wi * np.nan_to_num(f[:, sc]) for wi, f in zip(w, F)), O[:, sc])


# ------------------------------------------------------------------------------------------------ arena A
def arena_a(rng):
    from scipy import sparse
    T = gx.tss(); keep = None
    if TRIAL:
        tr_ = np.random.default_rng(1); ess = sorted({str(e) for e in gx.npz("XCELL_K562ESS")["cols_ens"] if str(e)})
        keep = set(tr_.choice(ess, 700, replace=False))
    SS = {s: gx.load(s, T, keep) for s in ESS}
    ppi, rooms = pp.knowledge(); best, fam = pa.substitutes()
    T4 = sorted(set.intersection(*(set(SS[s]["rows"]) for s in ESS))); K4 = sorted(set.intersection(*(set(SS[s]["cols"]) for s in ESS)))
    P = {s: gx.align(SS[s], T4, K4, "lfc") for s in ESS}; ki = {k: i for i, k in enumerate(K4)}; nK = len(K4)
    partners = collections.defaultdict(set)
    for p in ppi:
        x, y = tuple(p); partners[x].add(y); partners[y].add(x)
    RM = np.zeros((nK, len(pp.ROOMS)), bool); rn = list(pp.ROOMS)
    for i, k in enumerate(K4):
        for rr in rooms.get(k, ()):
            RM[i, rn.index(rr)] = True
    shared = (RM.astype(np.int32) @ RM.T.astype(np.int32)) > 0
    allg = sorted({str(g) for g in gx.npz("EXPR_HPA_CELLLINES")["rows"]}); EA = gx.hpa_expr(allg); Ap = gx.pct(np.where(np.isfinite(EA), EA, 0))
    gi = {g: i for i, g in enumerate(allg)}
    items = {"generic": [], "g3": [], "combined": [], "no_strength": [], "no_partners": [], "no_substitute": []}
    per, plant, real_part_gain, real_sub_gain = {}, [], [], []
    NP_ = [[] for _ in range(NMC)]; NS_ = [[] for _ in range(NMC)]
    for X in ESS:
        O = P[X]; links = gx.align(SS[X], T4, K4, "sig").sum(0); clear = links >= MIN_LINKS; made = pp.made_in(K4, X)
        tr, sc = gp.roles(K4); tr &= clear; sc &= clear
        gen = np.nanmean(O[:, tr], 1)[:, None] * np.ones((1, nK), np.float32)
        Fb, parts = gp.features(gen, [P[s] for s in ESS if s != X], tr)        # [gen, strength x gen, 3 patterns]
        abar = np.mean([p[0] for p in parts], 0)
        _, EX, _ = gp.split(O, tr); EX = np.nan_to_num(EX)
        on = [[] for _ in range(nK)]
        for i, k in enumerate(K4):
            if made[i]:
                for y in partners.get(k, ()):
                    j = ki.get(y)
                    if j is not None and j != i and clear[j] and made[j] and shared[i, j] and not pp.cis_pair(T, k, y):
                        on[i].append(j)
        sis = np.full(nK, -1)
        for i, k in enumerate(K4):
            if k in best:
                j = ki.get(best[k][0])
                if j is not None and j != i and clear[j] and not pa.cis_pair(T, k, best[k][0]):
                    sis[i] = j
        xi = ESS.index(X); cover = np.zeros(nK, np.float32)
        for i, k in enumerate(K4):
            if k in best and best[k][0] in gi:
                v = Ap[gi[best[k][0]]]; cover[i] = v[xi] - np.mean([v[j] for j in range(4) if j != xi])

        def part_feat(lists):
            rr, cc, vv = [], [], []
            for i, v in enumerate(lists):
                for j in v:
                    rr.append(j); cc.append(i); vv.append(1.0 / len(v))
            M = sparse.csr_matrix((vv, (rr, cc)), shape=(nK, nK)); return np.asarray(M.T.dot(EX.T).T, np.float32)

        def sis_feat(s_):
            F_ = np.zeros_like(EX); ok = s_ >= 0; F_[:, ok] = EX[:, s_[ok]]; return F_
        Fp, Fs = part_feat(on), sis_feat(sis); Fc = gen * (abar * cover)[None, :]
        fixed = Fb + [Fc]                                                        # 0 gen, 1 str, 2-4 patterns, 5 cover
        fit = Fitter(fixed, O, tr); allF = lambda ex: fixed + list(ex)
        ALL, NOSTR, NOSUB = [0, 1, 2, 3, 4, 5], [0, 5], [0, 1, 2, 3, 4]
        r_gen = gp.colr(gen[:, sc], O[:, sc])
        w = fit.fit([0, 1, 2, 3, 4]); r_g3 = score(Fb, w, O, sc)
        w = fit.fit(ALL, [Fp, Fs]); r_all = score(allF([Fp, Fs]), w, O, sc)
        w = fit.fit(NOSTR, [Fp, Fs]); r_nostr = score([fixed[i] for i in NOSTR] + [Fp, Fs], w, O, sc)
        w = fit.fit(ALL, [Fs]); r_nopart = score(allF([Fs]), w, O, sc)
        w = fit.fit(NOSUB, [Fp]); r_nosub = score([fixed[i] for i in NOSUB] + [Fp], w, O, sc)
        for k_, v in (("generic", r_gen), ("g3", r_g3), ("combined", r_all), ("no_strength", r_nostr), ("no_partners", r_nopart), ("no_substitute", r_nosub)):
            items[k_].append(v)
        npart = np.array([len(v) for v in on]); pool = clear & made
        cands = [np.nonzero(shared[i] & pool & (np.arange(nK) != i))[0] if npart[i] else np.array([], int) for i in range(nK)]
        famidx = [{ki[x] for x in fam.get(k, ()) if x in ki} | {i} for i, k in enumerate(K4)]
        clear_idx = np.nonzero(clear)[0]
        for b in range(NMC):
            lists = [list(rng.choice(c, min(len(c), npart[i]), replace=False)) if npart[i] and len(c) else [] for i, c in enumerate(cands)]
            Fpr = part_feat(lists)
            sr = np.array([(next((c for c in rng.choice(clear_idx, 10) if c not in famidx[i]), -1) if sis[i] >= 0 else -1) for i in range(nK)])
            Fsr = sis_feat(sr)
            w = fit.fit(ALL, [Fpr, Fs]); NP_[b].append(score(allF([Fpr, Fs]), w, O, sc) - r_nopart)
            w = fit.fit(ALL, [Fp, Fsr]); NS_[b].append(score(allF([Fp, Fsr]), w, O, sc) - r_nosub)
        Pl = EX + rng.standard_normal(EX.shape).astype(np.float32) * np.nan_to_num(gx.align(SS[X], T4, K4, "se"), nan=0.3)
        w = fit.fit(ALL, [Pl, Fs]); plant.append(score(allF([Pl, Fs]), w, O, sc) - r_nopart)
        real_part_gain.append(r_all - r_nopart); real_sub_gain.append(r_all - r_nosub)
        per[CELL[X]] = dict(scored=int(sc.sum()), with_partners=int((npart[sc] > 0).sum()), with_substitute=int((sis[sc] >= 0).sum()),
                            median_r={k_: float(np.nanmedian(v[-1])) for k_, v in items.items()},
                            vs_generic=better(r_all - r_gen), vs_g3=better(r_all - r_g3))
    cat = lambda v: np.concatenate(v)
    I = {k_: cat(v) for k_, v in items.items()}
    rpg, rsg = cat(real_part_gain), cat(real_sub_gain)
    nullp = [float(np.nanmedian(cat(x))) for x in NP_]; nulls = [float(np.nanmedian(cat(x))) for x in NS_]
    a1 = dict(per_cell=per, items=int(len(I["combined"])), median_r={k_: float(np.nanmedian(v)) for k_, v in I.items()},
              vs_generic=better(I["combined"] - I["generic"]), vs_g3=better(I["combined"] - I["g3"]))
    a1["gate"] = dict(real=bool(a1["vs_generic"]["passed"] and a1["vs_g3"]["passed"]), controls_ok=True)
    a2 = dict(vs_no_strength=better(I["combined"] - I["no_strength"]), vs_no_partners=better(rpg), vs_no_substitute=better(rsg),
              partners_mc=dict(real_median_gain=float(np.nanmedian(rpg)), null_95=float(np.percentile(nullp, 95)), p_mc=mc_p(float(np.nanmedian(rpg)), nullp)),
              substitute_mc=dict(real_median_gain=float(np.nanmedian(rsg)), null_95=float(np.percentile(nulls, 95)), p_mc=mc_p(float(np.nanmedian(rsg)), nulls)),
              negative_partners=better(cat(NP_[0])), negative_substitute=better(cat(NS_[0])), planted_partners=better(cat(plant)))
    a2["gate"] = dict(real=bool(a2["vs_no_strength"]["passed"] and a2["vs_no_partners"]["passed"] and a2["vs_no_substitute"]["passed"]
                                and a2["partners_mc"]["p_mc"] < 0.05 and a2["substitute_mc"]["p_mc"] < 0.05),
                      controls_ok=bool(not a2["negative_partners"]["passed"] and not a2["negative_substitute"]["passed"] and a2["planted_partners"]["passed"]))
    a1["gate"]["controls_ok"] = a2["gate"]["controls_ok"]
    for a in (a1, a2):
        a["gate"]["verdict"] = "PASS" if a["gate"]["real"] and a["gate"]["controls_ok"] else "FAIL"
    return a1, a2


# ------------------------------------------------------------------------------------------------ arena B
def arena_b(rng):
    from scipy import sparse
    ppi, rooms = pp.knowledge(); best, fam = pa.substitutes()
    D = gx.npz("DEP_DEPMAP"); E = gx.npz("EXPR_DEPMAP")
    dl, el = D["cols"].astype(str), E["cols"].astype(str); lines = sorted(set(dl) & set(el))
    di = {c: i for i, c in enumerate(dl)}; ei = {c: i for i, c in enumerate(el)}
    Xd = D["X"].astype(np.float32)[:, [di[c] for c in lines]]; Xe = E["X"].astype(np.float32)[:, [ei[c] for c in lines]]
    dg = D["rows"].astype(str); eg = E["rows"].astype(str); drow = {}
    for i, g in enumerate(dg):
        drow.setdefault(g, i)
    erow = {g: i for i, g in enumerate(eg)}
    needed = np.nanmean(Xd < -0.5, 1) >= 0.05
    genes = [g for g, i in drow.items() if needed[i]]
    if TRIAL:
        genes = genes[:400]
    gi = {g: i for i, g in enumerate(genes)}; nG = len(genes); nL = len(lines)
    Y = Xd[[drow[g] for g in genes]]
    dev = np.array([H_(c) == 0 for c in lines]); exm = ~dev
    own = np.array([erow.get(g, -1) for g in genes]); OWN = np.where((own >= 0)[:, None], Xe[np.maximum(own, 0)], 0)
    tot = np.nansum(Y, 0); cnt = np.isfinite(Y).sum(0)
    STR = (tot[None, :] - np.nan_to_num(Y)) / np.maximum(cnt[None, :] - np.isfinite(Y), 1)
    # partners: share a room, measured in DepMap; counted per line only where made (RNA log2(TPM+1) >= 1)
    allD = {g: i for i, g in enumerate(dg)}
    nb = collections.defaultdict(set)
    for p in ppi:
        x, y = tuple(p); nb[x].add(y); nb[y].add(x)
    plist = [[y for y in nb.get(g, ()) if y in allD and y in erow and (rooms.get(g, set()) & rooms.get(y, set()))] for g in genes]
    univ = sorted({y for v in plist for y in v} | {y for y in allD if y in erow and rooms.get(y)})
    ui = {g: i for i, g in enumerate(univ)}
    UE = np.nan_to_num(Xd[[allD[g] for g in univ]]); UM = (Xe[[erow[g] for g in univ]] >= 1.0).astype(np.float32)

    def part_feat(lists):
        rr, cc = [], []
        for i, v in enumerate(lists):
            for y in v:
                rr.append(i); cc.append(ui[y])
        M = sparse.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(nG, len(univ)))
        num = np.asarray(M.dot(UE * UM)); den = np.asarray(M.dot(UM))
        return np.where(den > 0, num / np.maximum(den, 1), 0).astype(np.float32)
    PART = part_feat(plist)
    sub = np.array([erow.get(best[g][0], -1) if g in best else -1 for g in genes])
    SUB = np.where((sub >= 0)[:, None], Xe[np.maximum(sub, 0)], 0)
    lvl = np.nanmean(Xe, 1); dec = pa.deciles(lvl); pool = {d: np.nonzero(dec == d)[0] for d in range(10)}
    exs = [{erow[x] for x in fam.get(g, ()) if x in erow} | ({erow[g]} if g in erow else set()) for g in genes]

    def fit_score(Yv, preds):
        k = len(preds) + 1; Xs = np.stack([np.ones_like(Yv)] + preds, 2); ok = np.isfinite(Yv) & dev[None, :]
        Xt = np.where(ok[..., None], Xs, 0); Yt = np.where(ok, Yv, 0)
        A = np.einsum("gli,glj->gij", Xt, Xt) + 1e-6 * np.eye(k)[None]; bvec = np.einsum("gli,gl->gi", Xt, Yt)
        w = np.linalg.solve(A, bvec[..., None])[..., 0]; Pr = np.einsum("glk,gk->gl", Xs, w)
        m = np.isfinite(Yv) & exm[None, :]; Pm = np.where(m, Pr, np.nan); Ym = np.where(m, Yv, np.nan)
        pc = Pm - np.nanmean(Pm, 1, keepdims=True); yc = Ym - np.nanmean(Ym, 1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nansum(pc * yc, 1) / np.sqrt(np.nansum(pc ** 2, 1) * np.nansum(yc ** 2, 1)), w
    r_base, wb = fit_score(Y, [OWN]); r_all, _ = fit_score(Y, [OWN, STR, PART, SUB])
    r_nostr, _ = fit_score(Y, [OWN, PART, SUB]); r_nopart, _ = fit_score(Y, [OWN, STR, SUB]); r_nosub, _ = fit_score(Y, [OWN, STR, PART])
    npart = np.array([len(v) for v in plist]); NB, NPt, NSt = [], [], []
    by_room = collections.defaultdict(list)
    for g in univ:
        for rr in rooms.get(g, ()):
            by_room[rr].append(g)
    by_room = {rr: np.array(v) for rr, v in by_room.items()}
    for b in range(NMC):
        lists = []
        for g, n in zip(genes, npart):                  # a random room of the gene, then a random gene in it
            if not n:
                lists.append([]); continue
            rl = [rr for rr in rooms.get(g, ()) if rr in by_room]; ex = nb.get(g, set()) | {g}
            pick = [by_room[rl[i]][rng.integers(0, len(by_room[rl[i]]))] for i in rng.integers(0, len(rl), n * 3)] if rl else []
            lists.append([y for y in dict.fromkeys(pick) if y not in ex][:n])
        PR = part_feat(lists)
        ic = pa.matched_draw(rng, pool, dec[np.maximum(sub, 0)], exs, np.maximum(sub, 0))
        SR = np.where((sub >= 0)[:, None], Xe[ic], 0)
        rp, _ = fit_score(Y, [OWN, STR, PR, SUB]); rs_, _ = fit_score(Y, [OWN, STR, PART, SR]); rb_, _ = fit_score(Y, [OWN, STR, PR, SR])
        NPt.append(float(np.nanmedian(rp - r_nopart))); NSt.append(float(np.nanmedian(rs_ - r_nosub))); NB.append(float(np.nanmedian(rb_ - r_base)))
        if b == 0:
            negp, negs = better(rp - r_nopart), better(rs_ - r_nosub)
    sP = (PART - PART.mean(1, keepdims=True)) / (PART.std(1, keepdims=True) + 1e-6)
    Yp = wb[:, :1] + wb[:, 1:2] * OWN + 0.3 * np.nanstd(Y, 1, keepdims=True) * sP + rng.normal(0, 1, Y.shape) * 0.2 * np.nanstd(Y, 1, keepdims=True)
    pa_, _ = fit_score(Yp, [OWN, STR, PART, SUB]); pn_, _ = fit_score(Yp, [OWN, STR, SUB]); plant = better(pa_ - pn_)
    gall = better(r_all - r_base)
    b1 = dict(genes=nG, dev_lines=int(dev.sum()), exam_lines=int(exm.sum()), with_partners=int((npart > 0).sum()), with_substitute=int((sub >= 0).sum()),
              median_r=dict(baseline=float(np.nanmedian(r_base)), combined=float(np.nanmedian(r_all)), no_strength=float(np.nanmedian(r_nostr)),
                            no_partners=float(np.nanmedian(r_nopart)), no_substitute=float(np.nanmedian(r_nosub))),
              vs_baseline=gall, mc=dict(real_median_gain=gall["median"], null_95=float(np.percentile(NB, 95)), p_mc=mc_p(gall["median"], NB)))
    b2 = dict(vs_no_strength=better(r_all - r_nostr), vs_no_partners=better(r_all - r_nopart), vs_no_substitute=better(r_all - r_nosub),
              partners_mc=dict(real_median_gain=float(np.nanmedian(r_all - r_nopart)), null_95=float(np.percentile(NPt, 95)),
                               p_mc=mc_p(float(np.nanmedian(r_all - r_nopart)), NPt)),
              substitute_mc=dict(real_median_gain=float(np.nanmedian(r_all - r_nosub)), null_95=float(np.percentile(NSt, 95)),
                                 p_mc=mc_p(float(np.nanmedian(r_all - r_nosub)), NSt)),
              negative_partners=negp, negative_substitute=negs, planted_partners=plant)
    ctrl = bool(not negp["passed"] and not negs["passed"] and plant["passed"])
    b1["gate"] = dict(real=bool(gall["passed"] and b1["mc"]["p_mc"] < 0.05), controls_ok=ctrl)
    b2["gate"] = dict(real=bool(b2["vs_no_strength"]["passed"] and b2["vs_no_partners"]["passed"] and b2["vs_no_substitute"]["passed"]
                                and b2["partners_mc"]["p_mc"] < 0.05 and b2["substitute_mc"]["p_mc"] < 0.05), controls_ok=ctrl)
    for a in (b1, b2):
        a["gate"]["verdict"] = "PASS" if a["gate"]["real"] and a["gate"]["controls_ok"] else "FAIL"
    return b1, b2


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
                                    claim=__doc__.split(" " + cid)[1].split(" G6-")[0].split("Controls:")[0].strip()[:1200], trial=TRIAL,
                                    pooling="arena A pooled over the four cell types (disclosed in the source before running)"), sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G6 already tested")
    t0 = time.time(); os.makedirs(RUN, exist_ok=True); os.makedirs(OUTD, exist_ok=True); rng = np.random.default_rng(SEED)
    out = {}
    out["G6-B1"], out["G6-B2"] = arena_b(rng); print("arena B done", f"({time.time() - t0:.0f}s)", flush=True)
    out["G6-A1"], out["G6-A2"] = arena_a(rng); print("arena A done", f"({time.time() - t0:.0f}s)", flush=True)
    if TRIAL:
        print("trial controls ok:", {c: out[c]["gate"]["controls_ok"] for c in CLAIMS}, "; real outcomes not shown"); return
    for cid in CLAIMS:
        print(cid, json.dumps(out[cid], default=float)[:1800], flush=True)
    json.dump(out, open(os.path.join(OUTD, "combined_predictor_summary.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "generegc:G6", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED))
    n = cbb.bank().add_many([Fact(f"CLAIM-{cid}", "combined_predictor_test", out[cid], t[:10], t, "derived", src, inputs=inp) for cid in CLAIMS])
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **out[cid]), sort_keys=True, default=float) + "\n")
    a1, a2, b1, b2 = (out[c] for c in CLAIMS)
    mr = a1["median_r"]; br = b1["median_r"]
    Ls = ["=" * 97, "GENEREGC -- one predictor from strength + partners + substitutes (EXAM)", "=" * 97,
          f"  G6-A1 screens, pooled ({a1['items']:,} knockdown x cell type): median r generic {mr['generic']:.3f}, G3 {mr['g3']:.3f}, combined {mr['combined']:.3f} "
          f"-> {a1['gate']['verdict']}",
          "         per cell type (G3 -> combined): " + ", ".join(f"{c} {v['median_r']['g3']:.3f}->{v['median_r']['combined']:.3f}" for c, v in a1["per_cell"].items()),
          f"  G6-A2 each clue adds (screens): without strength {mr['no_strength']:.3f}, without partners {mr['no_partners']:.3f}, without substitute "
          f"{mr['no_substitute']:.3f}; Monte Carlo p partners {a2['partners_mc']['p_mc']:.2g}, substitute {a2['substitute_mc']['p_mc']:.2g} -> {a2['gate']['verdict']}",
          f"  G6-B1 cancer lines ({b1['genes']:,} genes, EXAM lines): median r own RNA {br['baseline']:.3f} -> combined {br['combined']:.3f} "
          f"(p_MC {b1['mc']['p_mc']:.2g}) -> {b1['gate']['verdict']}",
          f"  G6-B2 each clue adds (lines): without strength {br['no_strength']:.3f}, without partners {br['no_partners']:.3f}, without substitute "
          f"{br['no_substitute']:.3f}; Monte Carlo p partners {b2['partners_mc']['p_mc']:.2g}, substitute {b2['substitute_mc']['p_mc']:.2g} -> {b2['gate']['verdict']}",
          f"  Facts written {n}"]
    open(RES, "w").write("\n".join(Ls) + "\n"); print("\n".join(Ls))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
