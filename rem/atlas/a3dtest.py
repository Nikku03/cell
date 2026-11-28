"""A3DTEST (G18) -- the G16 / G17 test in a third cell type: A549 lung cells treated with the steroid dexamethasone, with
the 3D map (Hi-C, 4 biological replicates) and gene expression (polyA RNA-seq, 4 replicates) at the same 5 time points:
0, 1, 4, 8 and 12 h (layer A549_DEX_3D, built by a3dbank.py from ENCODE, Reddy lab).

Asked by the user: "yes, run it on A549" (after G16 K562 and G17 THP-1: the open/closed neighbourhood moved with the genes
in both, loops did in K562 only, and in both the genes moved first).
Methods are G16 / G17's -- C1 and L1 use the same code, imported from k3dtest.py -- with 12 h as the end point. Genes:
protein-coding (GENCODE 46, chr1-22 + X), EXPRESSED = mean TPM >= 1 at 0 h or 12 h; e = log2(TPM + 1), mean of the 4
replicates; change = e(12 h) - e(0 h); baseline strata = ten equal-count bins of e(0 h). Halves: DEV = sha256('cellbrain-c1:'
+ Ensembl) parity 0, EXAM = parity 1; the trial used DEV genes and showed only controls; the declared run scores EXAM once.
Gain rule (house rule): one-sided sign test p < 0.05 AND median >= +0.01.

CLAIMS (declared before running; ids G18-*):
 G18-C1 OPEN/CLOSED NEIGHBOURHOOD (repeat of G16-C1 / G17-C1 at 12 h). Compartment value E1 per 100-kb bin (4 replicate
        maps summed); SHIFTED bins = the 10% with the largest |E1(12 h) - E1(0 h)|, STABLE = the half with the smallest;
        score per expressed gene in a shifted bin = sign(E1 change) x (its change - mean change of stable-bin genes in
        the same half, baseline stratum and 0 h E1 quintile). PASS: gain rule on EXAM genes AND median beats 95% of 1,000
        circular shifts of the E1-change track along each chromosome (p_MC < 0.05).
 G18-L1 LOOPS AT THE GENE START (repeat of G16-L1 / G17-L1). Loops = ENCODE's HiCCUPS loops of all 5 time points joined
        (a3dbank.py; 60,457); a moderated test of 12 h vs 0 h over the 4 + 4 replicate maps finds only 4 loops at
        q < 0.1 (seen when the layer was built, before any test), so here changing = the 2% of loops with the smallest
        p (the most-changed, as C1 uses the most-shifted bins); direction = sign of the change (gained +1, lost -1);
        static = the rest. A gene belongs to a loop when its start
        site is within 10 kb of either anchor; genes with changing loops of one direction only get that sign; score =
        sign x (change - mean change of genes in the same half and stratum that sit only at static loops). PASS: gain
        rule on EXAM genes AND median beats 95% of 1,000 shuffles of gained / lost labels (p_MC < 0.05).
 Controls: C1 and L1 as in G16 / G17: 50 planted runs (+0.3 log2 in the predicted direction on a shifted / shuffled
        track) must PASS >= 90%, 50 null runs <= 10% (else VOID).
 REPORTED: C1 at 1 h and 4 h; C1 and L1 with intron signal; rank correlations; bins flipping A <-> B; loop classes;
        per-chromosome compartment checks. Timing ('genes move first', G16 / G17-T1) is NOT declared here: the DEV
        trial showed it cannot be measured in A549 -- with the G17 method (halves = replicates 1-2 vs 3-4, pairs chosen
        on |change 0 -> 12 h| >= 0.5, pooled shares at 1, 4, 8 h) a planted 0.25 lead was found in only 36% of runs
        (55 DEV genes), because the loops barely change; its numbers and controls are reported only.
NOT: bulk cells; dexamethasone acts through one receptor (GR) on a few hundred genes, a narrow response, and barely
changes the 3D map (compartments r >= 0.98 to 0 h; 4 significant loop changes), so C1 and L1 here ask whether the
largest of small changes go with the genes; the loop calls are ENCODE's and the change test is ours (no gene outcomes
used); association, not cause.

Run: python3 -I rem/atlas/a3dtest.py declare | test [--trial]
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
OUT = R("outputs", "a549_3d")
CLAIMS = ["G18-C1", "G18-L1"]
SEED = 20261130
TIMES = [0, 60, 240, 480, 720]
MID = [60, 240, 480]
END = 720
REPS = (1, 2, 3, 4)
HALF = {"A": (1, 2), "B": (3, 4)}
N_MC = 100 if TRIAL else 1000
kt.N_CTRL, kt.N_CTRL_MC = (10, 100) if TRIAL else (50, 200)
N_SIM = 50 if TRIAL else 200
N_BOOT = 500 if TRIAL else 2000
SEL, SHIFT = 0.5, 0.25
H_ = kt.H_
LAYER = "A549_DEX_3D"


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "a3dtest.py"), "rb").read()); h.update(open(os.path.join(HERE, "k3dtest.py"), "rb").read())
    h.update(cbb.bank().facts(f"LAYER-{LAYER}", "layer")[-1].value["sha256"].encode())
    return h.hexdigest()[:16]


def load():
    Z = np.load(os.path.join(cbb.LAY, "a549_dex_3d.npz")); gx.USED.add(LAYER)
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
    # L1: changing = the 2% of loops with the smallest p (moderated 12 h vs 0 h test); direction = sign of the 12 h change
    D["loop_lfc_all"] = D["loop_lfc"]; l12 = D["loop_lfc_all"][:, TIMES.index(END)]
    chg = (D["loop_p"] <= np.quantile(D["loop_p"], 0.02)) & np.isfinite(l12) & (l12 != 0)
    D["loop_cluster_q"] = D["loop_cluster"]; D["loop_cluster"] = np.where(chg, np.where(l12 > 0, "gained", "lost"), "static")
    D["loop_lfc"] = np.stack([np.where(chg, np.sign(l12), 0.0), l12], 1)
    # RNA halves (mean of 2 replicates) for T1
    for k, src in (("yh", "y"), ("eh", "e")):
        D[k] = {(t, h): np.mean([D[src][(t, r)] for r in rr], 0) for t in TIMES for h, rr in HALF.items()}
    return D


# ---------------------------------------------------------------- T1
def t1_data(D):
    libs = [str(x) for x in D["loop_libs"]]; C = np.nan_to_num(D["loop_counts"].astype(np.float64)); tot = C.sum(0)
    halves = HALF; L, N = {}, {}
    for h, reps in halves.items():
        for t in TIMES:
            k = [i for i, n in enumerate(libs) if n.startswith(f"m{t:04d}_r") and int(n.split("_r")[1]) in reps]
            cnt = C[:, k].sum(1); L[(h, t)] = np.log2(cnt / tot[k].sum() * 1e6 + 1); N[(h, t)] = cnt
    return L, N


def t1_choose(D, L, gpair, lpair, Y, half):
    """rows: gene, loop, fold, sign of chosen change, measured loop changes at MID + END, measured intron changes at MID + END."""
    rows = []
    for f, (hs, rs, hm, rm) in enumerate((("A", "A", "B", "B"), ("B", "B", "A", "A"))):
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
    diff = np.concatenate([(Y[(t, "A")] - Y[(t, "B")])[D["y_ok"]] for t in TIMES])
    lv = np.concatenate([((Y[(t, "A")] + Y[(t, "B")]) / 2)[D["y_ok"]] for t in TIMES])
    qs = np.quantile(lv, np.linspace(0, 1, 11)); sdb = np.array([diff[(lv >= qs[i]) & (lv <= qs[i + 1])].std() / np.sqrt(2) for i in range(10)])
    f = np.clip((np.array(list(obs["FL"].values())) + np.array(list(obs["FY"].values()))) / 2, 0, 1)
    fl, fy = np.concatenate([[0], np.clip(f - shift, 0, 1), [1]]), np.concatenate([[0], np.clip(f + shift, 0, 1), [1]])
    g = rows[:, 0].astype(int); l = rows[:, 1].astype(int); hm = np.where(rows[:, 2] == 0, "B", "A")
    dL, dY = rows[:, 4 + m], rows[:, 4 + 2 * m + 1]
    sL = np.stack([1 / (np.log(2) * np.sqrt(np.where(hm == "B", N[("B", t)][l], N[("A", t)][l]) + 1)) for t in T], 1)
    yl = np.stack([np.where(hm == "B", Y[(t, "B")][g], Y[(t, "A")][g]) for t in T], 1)
    sY = sdb[np.clip(np.searchsorted(qs, yl) - 1, 0, 9)]
    Ls = fl[None, :] * dL[:, None] + rng.normal(0, sL); Ys = fy[None, :] * dY[:, None] + rng.normal(0, sY)
    sim = rows.copy(); sim[:, 4:4 + m + 1] = Ls[:, 1:] - Ls[:, :1]; sim[:, 4 + m + 1:4 + 2 * (m + 1)] = Ys[:, 1:] - Ys[:, :1]
    return t1_stat(sim, rng, 500)["passed"]


# ---------------------------------------------------------------- run
def run(rng, half):
    from scipy.stats import spearmanr
    D = load(); out, rep, ctl = {}, {}, {}
    iE = TIMES.index(END); ch = {t: D["em"][t] - D["em"][0] for t in TIMES[1:]}; lfc = ch[END]
    c1 = kt.C1(D, t1=iE); out["G18-C1"] = c1.full(c1.d, lfc, half, rng, N_MC)
    ctl["G18-C1"] = kt.controls(c1, lambda r: c1.roll(r), lfc, half, rng, kt.c1_plant_sign(c1))
    for t in (60, 240):
        ct = kt.C1(D, t1=TIMES.index(t)); rep[f"C1_at_{t}min"] = ct.full(ct.d, ch[t], half, rng, N_MC)
    yl = np.mean([D["y"][(END, r)] - D["y"][(0, r)] for r in REPS], 0)
    Dy = dict(D); Dy["expressed"] = D["expressed"] & D["y_ok"]; c1y = kt.C1(Dy, t1=iE)
    rep["C1_intron_signal"] = c1y.full(c1y.d, np.nan_to_num(yl), half, rng, N_MC)
    E = D["e1"]; v = np.isfinite(E[:, 0]) & np.isfinite(E[:, iE])
    rep["bins"] = dict(valid=int(v.sum()), flip_B_to_A=int(((E[:, 0] < 0) & (E[:, iE] > 0) & v).sum()), flip_A_to_B=int(((E[:, 0] > 0) & (E[:, iE] < 0) & v).sum()),
                       corr_E1_0h_12h=float(np.corrcoef(E[v, 0], E[v, iE])[0, 1]))
    gm = c1.gok & (D["half"] == half); rep["rank_corr_E1_change_vs_expression_change"] = float(spearmanr(c1.d[D["gbin"][gm]], lfc[gm])[0])

    l1 = kt.L1(D, col=0); out["G18-L1"] = l1.full(l1.dir, lfc, half, rng, N_MC)
    ctl["G18-L1"] = kt.controls(l1, lambda r: l1.shuffle(r), lfc, half, rng, lambda d: l1.gene_sign(d))
    l1y = kt.L1(Dy, col=0, M=l1.M); rep["L1_intron_signal"] = l1y.full(l1y.dir, np.nan_to_num(yl), half, rng, N_MC)
    rep["loops"] = dict(union=int(len(D["loop_cluster"])), significant_q_below_0_1=int(np.isin(D["loop_cluster_q"], ["gained", "lost"]).sum()), changing=int(l1.diff.sum()), gained_12h=int((l1.dir > 0).sum()), lost_12h=int((l1.dir < 0).sum()),
                        median_counts_per_library=float(np.median(D["loop_counts"])))
    sg = l1.gene_sign(l1.dir); mm = D["expressed"] & (D["half"] == half) & (sg != 0)
    rep["rank_corr_loop_sign_vs_expression_change"] = float(spearmanr(sg[mm], lfc[mm])[0]) if mm.sum() > 2 else None

    L, N = t1_data(D); M = l1.M.tocoo(); ok = D["y_ok"][M.row]; gp, lp = M.row[ok], M.col[ok]
    rows = t1_choose(D, L, gp, lp, D["yh"], half); t1 = t1_stat(rows, rng)
    if len(rows):
        eq = [t1_sim(D, L, N, D["yh"], rows, t1, rng, 0.0) for _ in range(N_SIM)]
        pl = [t1_sim(D, L, N, D["yh"], rows, t1, rng, SHIFT) for _ in range(N_SIM)]
    else:
        eq, pl = [True], [False]
    rep["T1_not_declared"] = dict(t1, controls=dict(equal_timing_pass_rate=float(np.mean(eq)), planted_lead_pass_rate=float(np.mean(pl)), runs=N_SIM))
    gm2 = D["expressed"][M.row]; rep["T1_mRNA_instead_of_intron"] = t1_stat(t1_choose(D, L, M.row[gm2], M.col[gm2], D["eh"], half), rng)
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
                                    claim=__doc__.split(" " + cid)[1].split(" G18-")[0].split(" Controls:")[0].strip()[:1500], trial=TRIAL), sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G18 already tested")
    t0 = time.time(); out, rep = run(np.random.default_rng(SEED), 0 if TRIAL else 1)
    if TRIAL:
        for cid in CLAIMS:
            print(cid, "controls", "ok" if out[cid]["gate"]["controls_ok"] else "NOT ok", json.dumps(out[cid]["controls"]), "units:", out[cid]["genes"])
        print("trial ran on DEV genes; outcomes not shown", f"({time.time() - t0:.0f}s)"); return
    os.makedirs(OUT, exist_ok=True); json.dump(dict(out, reported=rep), open(os.path.join(OUT, "a549_3d_tests.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "a3dtest:G18", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED))
    cbb.bank().add_many([Fact(f"CLAIM-{cid}", "a549_3d_test", out[cid], t[:10], t, "derived", src, inputs=inp) for cid in CLAIMS])
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **out[cid]), sort_keys=True, default=float) + "\n")
        print(cid, out[cid]["gate"]["verdict"], flush=True)
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
