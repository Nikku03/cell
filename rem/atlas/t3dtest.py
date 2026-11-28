"""T3DTEST (G17) -- the G16 test repeated in THP-1 macrophages switched on by LPS + interferon-gamma, where the 3D map
(Hi-C) and gene expression were BOTH measured at 8 time points: 0, 30, 60, 90, 120, 240, 360 min and 24 h (layer
THP1_LPIF_3D, built by t3dbank.py; Reed et al. 2022 Cell Reports; GEO GSE201376).

Asked by the user: "yes, run it on THP-1" (after G16 in K562: the 3D map and expression move together, but the genes moved
first and the loops followed).
Methods are G16's -- C1 and L1 use the same code, imported from k3dtest.py -- with 24 h in place of 72 h. Genes:
protein-coding (GENCODE 46, chr1-22 + X), EXPRESSED = mean TPM >= 1 at 0 h or 24 h; e = log2(TPM + 1), mean of the 2
replicates; change = e(24 h) - e(0 h); baseline strata = ten equal-count bins of e(0 h). Halves: DEV = sha256('cellbrain-c1:'
+ Ensembl) parity 0, EXAM = parity 1; the trial used DEV genes and showed only controls; the declared run scores EXAM once.
Gain rule (house rule): one-sided sign test p < 0.05 AND median >= +0.01.

CLAIMS (declared before running; ids G17-*):
 G17-C1 OPEN/CLOSED NEIGHBOURHOOD (repeat of G16-C1 at 24 h). Compartment value E1 per 100-kb bin; SHIFTED bins = the
        10% with the largest |E1(24 h) - E1(0 h)|, STABLE = the half with the smallest; score per expressed gene in a
        shifted bin = sign(E1 change) x (its change - mean change of stable-bin genes in the same half, baseline
        stratum and 0 h E1 quintile). PASS: gain rule on EXAM genes AND median beats 95% of 1,000 circular shifts of the
        E1-change track along each chromosome (p_MC < 0.05).
 G17-L1 LOOPS AT THE GENE START (repeat of G16-L1). Loops from the authors' table (42,690 at 10 kb); changing = the
        authors' time-course adjusted p < 0.1 (938 loops), direction = sign of the loop's log2 fold change at 24 h
        (gained +1, lost -1); static = the rest. (Changed from G16's rule after the DEV trial: the authors' four pattern
        clusters hold only 502 loops, and a planted +0.3 effect was then found in only 58% of 50 DEV control runs; with
        p < 0.1, 98%.) A gene belongs to a loop when its start site is within 10 kb of either anchor; genes with
        changing loops of one direction only get that sign;
        score = sign x (change - mean change of genes in the same half and stratum that sit only at static loops).
        PASS: gain rule on EXAM genes AND median beats 95% of 1,000 shuffles of gained / lost labels (p_MC < 0.05).
 G17-T1 GENES MOVE FIRST (G16-T1 found the reverse of '3D first' in K562; tested here in that direction, with six
        in-between times). Hi-C halves: replicate columns r1-r2 vs r3-r5 of the authors' table (taken to follow the GEO
        sample order, i.e. biological replicate 5 vs 6 and 4); RNA replicates 2 vs 3. Cross-fitted: a loop-gene pair is
        CHOSEN on one Hi-C half + one RNA replicate (|loop change 0 -> 24 h| >= 0.5 and |intron change 0 -> 24 h| >= 0.5,
        log2, same sign) and MEASURED on the other half + other replicate. Pooled over chosen pairs (both folds; turned
        by the sign of the chosen change): share of the 24 h change reached at 30, 60, 90, 120, 240 and 360 min, FY(t)
        for transcription (intron RNA) and FL(t) for the loop; d = mean over the six times of FY(t) - FL(t). PASS:
        d >= +0.01 AND d > 0 in 95% of 2,000 bootstrap resamples of EXAM genes (p_boot < 0.05).
 Controls: C1 and L1 as in G16: 50 planted runs (+0.3 log2 in the predicted direction on a shifted / shuffled track)
        must PASS >= 90%, 50 null runs <= 10% (else VOID). T1: 200 runs simulated with EQUAL timing (common curve = mean
        of FL(t) and FY(t), clipped to 0-1; each pair's own 24 h changes; each assay's noise at each time) must pass
        <= 10%, and 200 with transcription 0.25 ahead and the loop 0.25 behind must pass >= 90% (else VOID).
 REPORTED: C1 at 2 h and 6 h; L1 with each changing loop compared at its own peak time; C1 and L1 with intron signal;
        the T1 curves FL(t), FY(t); T1 with mRNA instead of intron; rank correlations; bins flipping A <-> B;
        per-chromosome compartment checks.
NOT: bulk cells; activation by LPS + IFN-gamma is a fast signalling response, unlike the slow K562 differentiation;
one cell line; loop calls and clusters are the authors'; the replicate-column order is assumed; association, not cause.

Run: python3 -I rem/atlas/t3dtest.py declare | test [--trial]
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


def _load(name):
    argv = sys.argv; sys.argv = [sys.argv[0]]
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


kt = _load("k3dtest")                    # G16's C1 / L1 code, unchanged
gx, cbb, Fact = kt.gx, kt.cbb, kt.Fact
LEDGER = gx.LEDGER
OUT = R("outputs", "thp1_3d")
CLAIMS = ["G17-C1", "G17-L1", "G17-T1"]
SEED = 20261129
TIMES = [0, 30, 60, 90, 120, 240, 360, 1440]
MID = [30, 60, 90, 120, 240, 360]
END = 1440
REPS = (2, 3)
N_MC = 100 if TRIAL else 1000
kt.N_CTRL, kt.N_CTRL_MC = (10, 100) if TRIAL else (50, 200)
N_SIM = 50 if TRIAL else 200
N_BOOT = 500 if TRIAL else 2000
SEL, SHIFT = 0.5, 0.25
H_ = kt.H_
LAYER = "THP1_LPIF_3D"


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "t3dtest.py"), "rb").read()); h.update(open(os.path.join(HERE, "k3dtest.py"), "rb").read())
    h.update(cbb.bank().facts(f"LAYER-{LAYER}", "layer")[-1].value["sha256"].encode())
    return h.hexdigest()[:16]


def load():
    Z = np.load(os.path.join(cbb.LAY, "thp1_lpif_3d.npz")); gx.USED.add(LAYER)
    D = {k: Z[k] for k in Z.files}
    cols = [tuple(map(int, c.split("_"))) for c in D["rna_cols"]]
    e = np.log2(D["tpm"].astype(np.float64) + 1)
    D["e"] = {(t, r): e[:, k] for k, (t, r) in enumerate(cols)}
    D["em"] = {t: np.mean([D["e"][(t, r)] for r in REPS], 0) for t in TIMES}
    tpm_m = {t: np.mean([D["tpm"][:, k] for k, (tt, r) in enumerate(cols) if tt == t], 0) for t in (0, END)}
    D["expressed"] = (tpm_m[0] >= 1) | (tpm_m[END] >= 1)
    D["half"] = np.array([H_(g) for g in D["gene"]])
    D["stratum"] = np.full(len(D["gene"]), -1); ex = D["expressed"]
    q = np.quantile(D["em"][0][ex], np.linspace(0, 1, 11)[1:-1]); D["stratum"][ex] = np.searchsorted(q, D["em"][0][ex], side="right")
    icols = [tuple(map(int, c.split("_"))) for c in D["int_cols"]]; X = D["intron"].astype(np.float64)
    good = np.isfinite(X).all(1) & (X > 0).all(1) & ex
    ref = np.exp(np.log(X[good]).mean(1)); sf = np.median(X[good] / ref[:, None], 0); Xn = X / sf
    p = 0.05 * np.median(Xn[good][:, [k for k, (t, r) in enumerate(icols) if t == 0]])
    D["y"] = {(t, r): np.log2(Xn[:, k] + p) for k, (t, r) in enumerate(icols)}
    with np.errstate(all="ignore"):
        D["y_ok"] = np.isfinite(X).all(1) & ex & (np.nanmean(Xn, 1) > 2 * p)
    bkey = {(c, s): k for k, (c, s) in enumerate(zip(D["bin_chrom"], D["bin_start"]))}
    D["gbin"] = np.array([bkey.get((c, (t // 100000) * 100000), -1) for c, t in zip(D["chrom"], D["tss"])])
    # L1: changing = authors' adjusted p < 0.1, direction = sign of the 24 h log2 fold change (column 0), as declared
    D["loop_lfc_all"] = D["loop_lfc"]; l24 = D["loop_lfc_all"][:, TIMES.index(END)]; chg = (D["loop_padj"] < 0.1) & np.isfinite(l24) & (l24 != 0)
    D["loop_cluster_authors"] = D["loop_cluster"]; D["loop_cluster"] = np.where(chg, np.where(l24 > 0, "gained", "lost"), "static")
    D["loop_lfc"] = np.stack([np.where(chg, np.sign(l24), 0.0), l24], 1)
    return D


# ---------------------------------------------------------------- L1 at each loop's own peak time (reported)
def l1_peak(D, l1, half):
    tp = D["loop_peak_time"]; ti = np.where(np.isfinite(tp), tp - 1, 0).astype(int)        # 1-based index into TIMES
    M = l1.M.tocsr(); m = D["expressed"] & (D["half"] == half); sg = l1.gene_sign(l1.dir)
    test = np.nonzero(m & (sg != 0))[0]; pool = m & l1.on_static & ~l1.on_diff
    ch = {t: D["em"][t] - D["em"][0] for t in TIMES[1:]}
    ctrl = {t: np.array([ch[t][pool & (D["stratum"] == s)].mean() if (pool & (D["stratum"] == s)).sum() >= 5 else np.nan for s in range(10)]) for t in TIMES[1:]}
    s = []
    for g in test:
        L = M.indices[M.indptr[g]:M.indptr[g + 1]]; L = L[l1.diff[L]]
        v = [sg[g] * (ch[TIMES[ti[k]]][g] - ctrl[TIMES[ti[k]]][D["stratum"][g]]) for k in L if ti[k] >= 1]
        if v:
            s.append(np.mean(v))
    s = np.array(s); s = s[np.isfinite(s)]; ok, p, med, n = kt.gain(s)
    return dict(genes=n, sign_p=p, median=med, share_positive=float((s > 0).mean()) if n else None, passed_gain_rule=bool(ok))


# ---------------------------------------------------------------- T1
def t1_data(D):
    libs = [str(x) for x in D["loop_libs"]]; C = np.nan_to_num(D["loop_counts"].astype(np.float64)); tot = C.sum(0)
    halves = {"A": (1, 2), "B": (3, 4, 5)}; L, N = {}, {}
    for h, reps in halves.items():
        for t in TIMES:
            k = [i for i, n in enumerate(libs) if n.startswith(f"m{t:04d}_r") and int(n.split("_r")[1].split("_")[0]) in reps]
            cnt = C[:, k].sum(1); L[(h, t)] = np.log2(cnt / tot[k].sum() * 1e6 + 1); N[(h, t)] = cnt
    return L, N


def t1_choose(D, L, gpair, lpair, Y, half):
    """rows: gene, loop, fold, sign of chosen change, measured loop changes at MID + END, measured intron changes at MID + END."""
    rows = []
    for f, (hs, rs, hm, rm) in enumerate((("A", REPS[0], "B", REPS[1]), ("B", REPS[1], "A", REPS[0]))):
        dLs = L[(hs, END)][lpair] - L[(hs, 0)][lpair]; dYs = Y[(END, rs)][gpair] - Y[(0, rs)][gpair]
        ch = (np.abs(dLs) >= SEL) & (np.abs(dYs) >= SEL) & (np.sign(dLs) == np.sign(dYs)) & (D["half"][gpair] == half)
        k = np.nonzero(ch)[0]
        cols = [gpair[k], lpair[k], np.full(len(k), f), np.sign(dLs[k])]
        cols += [L[(hm, t)][lpair[k]] - L[(hm, 0)][lpair[k]] for t in MID + [END]]
        cols += [Y[(t, rm)][gpair[k]] - Y[(0, rm)][gpair[k]] for t in MID + [END]]
        rows.append(np.stack(cols, 1))
    return np.concatenate(rows)


def t1_stat(rows, rng, n_boot=None):
    n_boot = n_boot or N_BOOT; m = len(MID)
    if len(rows) == 0:
        return dict(genes=0, pairs=0, d=0.0, p_boot=1.0, passed=False)
    g = rows[:, 0].astype(int); u, gi = np.unique(g, return_inverse=True); sg = rows[:, 3]
    sums = np.stack([np.bincount(gi, weights=sg * rows[:, c], minlength=len(u)) for c in range(4, 4 + 2 * (m + 1))], 1)

    def dval(S):
        FL = S[..., :m] / S[..., m:m + 1]; FY = S[..., m + 1:2 * m + 1] / S[..., 2 * m + 1:2 * m + 2]
        return FL, FY, (FY - FL).mean(-1)
    FL, FY, d = dval(sums.sum(0))
    W = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=n_boot)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = dval(W @ sums)[2]
    p = float((1 + np.sum(~(db > 0))) / (n_boot + 1))
    return dict(genes=int(len(u)), pairs=int(len(rows)), FL=dict(zip(map(str, MID), map(float, FL))), FY=dict(zip(map(str, MID), map(float, FY))),
                d=float(d), d_boot_5=float(np.nanquantile(db, 0.05)), p_boot=p, passed=bool(p < 0.05 and d >= 0.01))


def t1_sim(D, L, N, Y, rows, obs, rng, shift):
    """re-measure every chosen pair with simulated noise; equal timing (shift 0) or transcription ahead by `shift`."""
    m = len(MID); T = [0] + MID + [END]
    diff = np.concatenate([(Y[(t, REPS[0])] - Y[(t, REPS[1])])[D["y_ok"]] for t in TIMES])
    lv = np.concatenate([((Y[(t, REPS[0])] + Y[(t, REPS[1])]) / 2)[D["y_ok"]] for t in TIMES])
    qs = np.quantile(lv, np.linspace(0, 1, 11)); sdb = np.array([diff[(lv >= qs[i]) & (lv <= qs[i + 1])].std() / np.sqrt(2) for i in range(10)])
    f = np.clip((np.array(list(obs["FL"].values())) + np.array(list(obs["FY"].values()))) / 2, 0, 1)
    fl, fy = np.concatenate([[0], np.clip(f - shift, 0, 1), [1]]), np.concatenate([[0], np.clip(f + shift, 0, 1), [1]])
    g = rows[:, 0].astype(int); l = rows[:, 1].astype(int); hm = np.where(rows[:, 2] == 0, "B", "A"); rm = np.where(rows[:, 2] == 0, REPS[1], REPS[0])
    dL, dY = rows[:, 4 + m], rows[:, 4 + 2 * m + 1]
    sL = np.stack([1 / (np.log(2) * np.sqrt(np.where(hm == "B", N[("B", t)][l], N[("A", t)][l]) + 1)) for t in T], 1)
    yl = np.stack([np.where(rm == REPS[1], Y[(t, REPS[1])][g], Y[(t, REPS[0])][g]) for t in T], 1)
    sY = sdb[np.clip(np.searchsorted(qs, yl) - 1, 0, 9)]
    Ls = fl[None, :] * dL[:, None] + rng.normal(0, sL); Ys = fy[None, :] * dY[:, None] + rng.normal(0, sY)
    sim = rows.copy(); sim[:, 4:4 + m + 1] = Ls[:, 1:] - Ls[:, :1]; sim[:, 4 + m + 1:4 + 2 * (m + 1)] = Ys[:, 1:] - Ys[:, :1]
    return t1_stat(sim, rng, 500)["passed"]


# ---------------------------------------------------------------- run
def run(rng, half):
    from scipy.stats import spearmanr
    D = load(); out, rep, ctl = {}, {}, {}
    iE = TIMES.index(END); ch = {t: D["em"][t] - D["em"][0] for t in TIMES[1:]}; lfc = ch[END]
    c1 = kt.C1(D, t1=iE); out["G17-C1"] = c1.full(c1.d, lfc, half, rng, N_MC)
    ctl["G17-C1"] = kt.controls(c1, lambda r: c1.roll(r), lfc, half, rng, kt.c1_plant_sign(c1))
    for t in (120, 360):
        ct = kt.C1(D, t1=TIMES.index(t)); rep[f"C1_at_{t}min"] = ct.full(ct.d, ch[t], half, rng, N_MC)
    yl = np.mean([D["y"][(END, r)] - D["y"][(0, r)] for r in REPS], 0)
    Dy = dict(D); Dy["expressed"] = D["expressed"] & D["y_ok"]; c1y = kt.C1(Dy, t1=iE)
    rep["C1_intron_signal"] = c1y.full(c1y.d, np.nan_to_num(yl), half, rng, N_MC)
    E = D["e1"]; v = np.isfinite(E[:, 0]) & np.isfinite(E[:, iE])
    rep["bins"] = dict(valid=int(v.sum()), flip_B_to_A=int(((E[:, 0] < 0) & (E[:, iE] > 0) & v).sum()), flip_A_to_B=int(((E[:, 0] > 0) & (E[:, iE] < 0) & v).sum()),
                       corr_E1_0h_24h=float(np.corrcoef(E[v, 0], E[v, iE])[0, 1]))
    gm = c1.gok & (D["half"] == half); rep["rank_corr_E1_change_vs_expression_change"] = float(spearmanr(c1.d[D["gbin"][gm]], lfc[gm])[0])

    l1 = kt.L1(D, col=0); out["G17-L1"] = l1.full(l1.dir, lfc, half, rng, N_MC)
    ctl["G17-L1"] = kt.controls(l1, lambda r: l1.shuffle(r), lfc, half, rng, lambda d: l1.gene_sign(d))
    rep["L1_at_each_loop_peak_time"] = l1_peak(D, l1, half)
    l1y = kt.L1(Dy, col=0, M=l1.M); rep["L1_intron_signal"] = l1y.full(l1y.dir, np.nan_to_num(yl), half, rng, N_MC)
    lab = D["loop_label"]
    rep["loops"] = dict(changing=int(l1.diff.sum()), gained_24h=int((l1.dir > 0).sum()), lost_24h=int((l1.dir < 0).sum()),
                        **{f"authors_{k}": int((lab == k).sum()) for k in ("up.early", "up.late", "down.early", "down.late")},
                        authors_clusters_among_changing=float(np.isin(lab[l1.diff], ["up.early", "up.late", "down.early", "down.late"]).mean()))
    sg = l1.gene_sign(l1.dir); mm = D["expressed"] & (D["half"] == half) & (sg != 0)
    rep["rank_corr_loop_sign_vs_expression_change"] = float(spearmanr(sg[mm], lfc[mm])[0]) if mm.sum() > 2 else None

    L, N = t1_data(D); M = l1.M.tocoo(); ok = D["y_ok"][M.row]; gp, lp = M.row[ok], M.col[ok]
    rows = t1_choose(D, L, gp, lp, D["y"], half); out["G17-T1"] = t1_stat(rows, rng)
    if len(rows):
        eq = [t1_sim(D, L, N, D["y"], rows, out["G17-T1"], rng, 0.0) for _ in range(N_SIM)]
        pl = [t1_sim(D, L, N, D["y"], rows, out["G17-T1"], rng, SHIFT) for _ in range(N_SIM)]
    else:
        eq, pl = [True], [False]
    ctl["G17-T1"] = dict(equal_timing_pass_rate=float(np.mean(eq)), planted_lead_pass_rate=float(np.mean(pl)), runs=N_SIM, ok=bool(np.mean(eq) <= 0.10 and np.mean(pl) >= 0.9))
    gm2 = D["expressed"][M.row]; rep["T1_mRNA_instead_of_intron"] = t1_stat(t1_choose(D, L, M.row[gm2], M.col[gm2], D["e"], half), rng)
    for cid in CLAIMS:
        out[cid]["controls"] = ctl[cid]
        out[cid]["gate"] = dict(real=out[cid]["passed"], controls_ok=ctl[cid]["ok"],
                                verdict=("PASS" if out[cid]["passed"] else "FAIL") if ctl[cid]["ok"] else "VOID (controls)")
    rep["genes_expressed_in_half"] = int((D["expressed"] & (D["half"] == half)).sum())
    return out, rep


def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def declare():
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            if cid in have:
                print(cid, "already declared"); continue
            f.write(json.dumps(dict(op="declare", id=cid, at=cbb.now(), code_sha=code_sha(), commit=gx._git("rev-parse", "HEAD").stdout.strip(),
                                    claim=__doc__.split(" " + cid)[1].split(" G17-")[0].split(" Controls:")[0].strip()[:1500], trial=TRIAL), sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G17 already tested")
    t0 = time.time(); out, rep = run(np.random.default_rng(SEED), 0 if TRIAL else 1)
    if TRIAL:
        for cid in CLAIMS:
            print(cid, "controls", "ok" if out[cid]["gate"]["controls_ok"] else "NOT ok", json.dumps(out[cid]["controls"]), "units:", out[cid]["genes"])
        print("trial ran on DEV genes; outcomes not shown", f"({time.time() - t0:.0f}s)"); return
    os.makedirs(OUT, exist_ok=True); json.dump(dict(out, reported=rep), open(os.path.join(OUT, "thp1_3d_tests.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "t3dtest:G17", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED))
    cbb.bank().add_many([Fact(f"CLAIM-{cid}", "thp1_3d_test", out[cid], t[:10], t, "derived", src, inputs=inp) for cid in CLAIMS])
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **out[cid]), sort_keys=True, default=float) + "\n")
        print(cid, out[cid]["gate"]["verdict"], flush=True)
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
