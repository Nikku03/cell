"""DOES TF OCCUPANCY PREDICT THE CHANGE IN EXPRESSION? The fair version of Stage 1.

WHAT THIS IS RETESTING. Stage 1 of the dynamic model has come back NULL twice -- on pseudotime (delta R2
0.0013) and on real clock time (21 lags, permuted time equally good). Both nulls were well controlled: not a
power failure (split-half reliability +0.3991) and not a link-quality failure (validated links scored -0.0034
too). The remaining explanation was that nothing was DRIVING those systems. An unperturbed population sits at
steady state, and a steady state has no transient for occupancy to explain.

The ENCODE A549 dexamethasone series removes that excuse. Dexamethasone activates the glucocorticoid receptor
(NR3C1), which enters the nucleus in minutes and binds thousands of sites; ENCODE measured NR3C1 occupancy and
polyA RNA on the SAME sixteen-point grid (5 min to 12 h) in the same cell line. A real driver, a real clock,
matched readouts. If occupancy dynamics cannot predict expression dynamics here, the null is a statement about
the relationship rather than about the data.

THE TARGET IS THE CHANGE, AND THEN THE CHANGE TWICE OVER. Predicting an expression LEVEL is autoregression --
this project measured R2 0.887 for a level against 0.001 for the corresponding change. So the target is
dlog2(TPM+1) across consecutive timepoints. That alone is still too easy to fake:

  INTERVAL FIXED EFFECT. The intervals are uneven (5 min early, 120 min late) so their dlog scales differ. A
    model can score by learning the interval mean and nothing else. Every arm therefore carries interval
    identity, so occupancy must explain variation WITHIN an interval, ACROSS genes.
  GENE FIXED EFFECT. A gene that rises throughout and happens to carry constant GR occupancy is a static
    association wearing a dynamic costume. The strict target removes each gene's mean change as well, leaving
    only within-gene, within-interval variation -- the only thing that can honestly be called dynamics.

THE CHANGE IS DIFFERENCED WITHIN REPLICATE. The replicate composition of this series is not constant -- t=5
averages replicates {1,3}, t=30 averages {1,2,3,4}, t=300 averages {2,3,4} -- and four of the fifteen
intervals cross a change. Differencing timepoint MEANS across such a boundary subtracts one set of culture
batches from another, leaving a gene-specific batch offset in the target that any gene-level covariate can
partly predict. It is not a subtle effect: it is what produced JUN scoring +0.0752 on the 16-point panel
against +0.0021 on the 11-point panel, since two of the four composition changes lie in the early window only
the 16-point panel includes. Differencing within each replicate and then averaging cancels each batch's own
offset exactly, and every interval keeps at least two common replicates.

FOUR CONTROLS, each destroying a different thing:
  PERMUTED TIME. Permute which occupancy timepoint pairs with which interval. Preserves every occupancy value,
    the gene mapping, the target and every confound; destroys only temporal correspondence.
  SWAPPED GENES. Permute occupancy between genes matched on expression decile. Preserves marginals and the
    confound; destroys gene-specific assignment.
  CTCF AND RAD21 AS A PLACEBO. Architectural proteins should not carry a steroid response. If their occupancy
    predicts expression change as well as GR's, what is being measured is peak density or accessibility rather
    than glucocorticoid signalling. This is the control most able to kill the result, so it runs on exactly
    the same rows as NR3C1 -- see the panel note below.
  REPLICATE RELIABILITY. The two RNA-seq replicates give the reliability of a CHANGE, which is the ceiling on
    any R2 here. A small R2 against a small ceiling means something different from a small R2 against a large
    one, and the earlier nulls were only interpretable because that number existed.

TWO PANELS, AND NEVER A COMPARISON ACROSS THEM. NR3C1, EP300 and JUN were assayed on all sixteen timepoints;
CTCF, RAD21 and the other cofactors start at 30 min. Scoring a target on timepoints it does not have would
feed the model a column that is zero early and nonzero late -- a disguised clock, not occupancy. So each panel
is the intersection of the timepoints its own targets actually have, every arm inside a panel sees identical
rows, and no number from one panel is compared with a number from the other.

READ THE RAW HELD-OUT VALUE. Nested arms are compared with each other (both held out -- an ordinary nested
comparison); the permutation arms establish significance and are never subtracted from anything. This project
has caught itself reading `R2 - shuffled` as an effect size four times.
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
DIR = SP / "grtc"
NET = SP / "cell_complete.json.gz"
SEEDS = (0, 1, 2)
RINGS = ((0, 1_000), (1_000, 10_000), (10_000, 100_000))   # disjoint, so the blocks are not nested
MIN_TPM = 1.0            # a gene must clear this at half its timepoints, else dlog is noise on noise
N_PERM = 6
PLACEBO = ("CTCF", "RAD21")
PANELS = {"16-point (GR + full-grid cofactors)": ("NR3C1", "EP300", "JUN"),
          "11-point (all targets incl. placebo)": ("NR3C1", "EP300", "JUN", "JUNB", "FOSL2", "CEBPB",
                                                   "CTCF", "RAD21")}


# ----------------------------------------------------------------------------- loading

def load_peaks(path):
    """narrowPeak -> (chrom, summit, signalValue). Summit is start+peak when the caller reports one."""
    ch, po, sg = [], [], []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            st, en = int(p[1]), int(p[2])
            off = int(p[9]) if len(p) > 9 and p[9] not in ("", "-1") else -1
            ch.append(p[0])
            po.append(st + off if off >= 0 else (st + en) // 2)
            sg.append(float(p[6]))
    return np.array(ch), np.array(po, dtype=np.int64), np.array(sg, dtype=np.float32)


def occupancy(path, chrom, tss):
    """Summed peak signal in each disjoint TSS ring, per gene. Peaks are sorted per chromosome once and each
    ring costs two binary searches, rather than the O(peaks-per-lookup) scan that made an earlier module in
    this project non-terminating."""
    pc, pp, ps = load_peaks(path)
    out = np.zeros((len(chrom), len(RINGS)), dtype=np.float32)
    for c in np.unique(chrom):
        m = pc == c
        if not m.any():
            continue
        o = np.argsort(pp[m])
        pos, sig = pp[m][o], ps[m][o]
        cum = np.concatenate([[0.0], np.cumsum(sig, dtype=np.float64)])
        gi = np.where(chrom == c)[0]
        t = tss[gi]
        for r, (lo, hi) in enumerate(RINGS):
            a_hi = cum[np.searchsorted(pos, t + hi)] - cum[np.searchsorted(pos, t - hi)]
            a_lo = cum[np.searchsorted(pos, t + lo)] - cum[np.searchsorted(pos, t - lo)]
            out[gi, r] = a_hi - a_lo                 # ring = the hi window minus the lo window
    return out


def build():
    import pandas as pd
    man = json.load(open(DIR / "manifest.json"))
    rna = np.load(DIR / "rna.npz", allow_pickle=True)
    tpm, ensg = rna["tpm"], np.array([str(x) for x in rna["genes"]])
    mins, reps = rna["mins"], rna["reps"]

    im = pd.read_parquet(Path(__file__).resolve().parent.parent / "outputs/orphan/id_map.parquet")
    e2s = {str(a).split(".")[0]: str(b) for a, b in zip(im["ensembl"], im["symbol"])
           if isinstance(a, str)}
    d = json.load(gzip.open(NET, "rt"))
    pos = {g["name"]: (g.get("chrom"), g.get("tss")) for g in d["genes"]}
    del d

    keep, names, chrom, tss = [], [], [], []
    for i, e in enumerate(ensg):
        s = e2s.get(e)
        c, t = pos.get(s, (None, None)) if s else (None, None)
        if not c or t in (None, ""):
            continue
        keep.append(i); names.append(s); chrom.append(c); tss.append(int(float(t)))
    keep = np.array(keep)
    tpm = tpm[:, keep]
    chrom, tss = np.array(chrom), np.array(tss, dtype=np.int64)

    rt = sorted(set(int(x) for x in mins))
    lv = np.log2(1.0 + tpm)
    E = np.zeros((len(rt), tpm.shape[1]), dtype=np.float32)
    Erep = {}
    for j, t in enumerate(rt):
        m = mins == t
        E[j] = lv[m].mean(0)
        for r in sorted(set(int(x) for x in reps[m])):
            mm = m & (reps == r)
            if mm.any():
                Erep.setdefault(int(r), {})[t] = lv[mm].mean(0)

    # PER-TIMEPOINT QUANTILE NORMALISATION, because the peak count is not constant. For one target the
    # number of called peaks swings four-fold across the course (NR3C1 40,657 at 10 min down to 5,391 at
    # 600 min; JUN 15,306 at 30 min against 58,693 at 10 min). Some of that is real -- GR does attenuate --
    # but sequencing depth and ChIP efficiency move too, and raw summed signal cannot tell the two apart. An
    # interval dummy absorbs a uniform shift but not its interaction with how many sites a gene has. Ranking
    # genes WITHIN each timepoint makes every timepoint's distribution identical by construction, so what
    # survives is which genes are occupied relative to their peers -- and any change in that ranking is a
    # change the assay depth cannot manufacture. Both versions are carried and both are reported.
    from scipy.stats import rankdata
    OCC, OCCQ, TT, NPK = {}, {}, {}, {}
    for tgt, byt in man["peaks"].items():
        if not byt:
            continue
        ts = sorted(int(k) for k in byt)
        A = np.stack([occupancy(byt[str(t)]["path"], chrom, tss) for t in ts])
        NPK[tgt] = [int(sum(1 for _ in gzip.open(byt[str(t)]["path"], "rt"))) for t in ts]
        Q = np.zeros_like(A)
        for i in range(A.shape[0]):
            for r in range(A.shape[2]):
                Q[i, :, r] = rankdata(A[i, :, r], method="average") / A.shape[1]
        OCC[tgt], OCCQ[tgt], TT[tgt] = np.log1p(A), Q.astype(np.float32), ts
    return {"names": np.array(names), "chrom": chrom, "tss": tss, "E": E, "times": rt,
            "Erep": Erep, "OCC": OCC, "OCCQ": OCCQ, "TT": TT, "n_peaks": NPK, "manifest": man}


# ----------------------------------------------------------------------------- modelling

def fit_oof_reg(X, y, f):
    import xgboost as xgb
    p = np.zeros(len(y))
    for k in sorted(set(f)):
        tr, te = f != k, f == k
        m = xgb.XGBRegressor(max_depth=4, n_estimators=250, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.6, min_child_weight=20, reg_lambda=2.0,
                             tree_method="hist", n_jobs=8)
        m.fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return p


def r2(y, p):
    ss = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


def folds_chrom(ch, seed):
    u = sorted(set(ch))
    a = np.random.default_rng(seed).permutation(len(u))
    m = {c: a[i] % 5 for i, c in enumerate(u)}
    return np.array([m[c] for c in ch])


def demean(y, keys):
    """Remove the mean of every level of every key, alternating to convergence."""
    z = y.astype(np.float64).copy()
    for _ in range(6):
        for key in keys:
            n = int(key.max()) + 1
            s, c = np.zeros(n), np.zeros(n)
            np.add.at(s, key, z)
            np.add.at(c, key, 1.0)
            z -= (s / np.maximum(c, 1.0))[key]
    return z.astype(np.float32)


def gene_swap(rng, n, gi, mu):
    """Permute gene indices WITHIN expression deciles, so a swapped gene has a comparable expression level --
    the control destroys identity without also destroying the confound identity shares."""
    perm = np.arange(n)
    dec = np.digitize(mu[gi], np.quantile(mu[gi], np.linspace(0.1, 0.9, 9)))
    for d in np.unique(dec):
        idx = gi[dec == d]
        perm[idx] = rng.permutation(idx)
    return perm


# ----------------------------------------------------------------------------- one panel

def run_panel(B, label, targets, gi, report):
    """All arms on identical rows. `targets` must all carry every timepoint in the panel."""
    E, times, chrom = B["E"], B["times"], B["chrom"]
    have = [set(B["TT"][t]) for t in targets if t in B["TT"]]
    tg = [t for t in targets if t in B["TT"]]
    if not have:
        return None
    shared = sorted(set(times).intersection(*have))
    if len(shared) < 4:
        report(f"    {label}: only {len(shared)} shared timepoints -- skipped")
        return None
    ti = {t: times.index(t) for t in shared}
    nint = len(shared) - 1

    # PAIRED-REPLICATE DIFFERENCES, not differences of timepoint means.
    #
    # The replicate composition of this series is NOT constant: t=5 averages replicates {1,3}, t=30 averages
    # {1,2,3,4}, t=300 averages {2,3,4}. Four of the fifteen intervals change composition. Differencing the
    # timepoint MEANS across such a boundary subtracts one set of culture batches from another, so the
    # difference carries a gene-specific batch offset that has nothing to do with time -- and a gene-level
    # covariate like occupancy can predict part of it. That is not a small effect: it is what produced JUN's
    # +0.0752 on the 16-point panel against +0.0021 on the 11-point panel, since two of the four
    # composition changes fall in the early window the 16-point panel adds.
    #
    # Differencing WITHIN each replicate and then averaging cancels each batch's own offset exactly. Every
    # interval here retains at least two common replicates, so nothing is lost.
    Erep = B["Erep"]
    dvec, levvec, nrep = [], [], []
    for k in range(1, len(shared)):
        t0, t1 = shared[k - 1], shared[k]
        rr = [r for r in sorted(Erep) if t0 in Erep[r] and t1 in Erep[r]]
        if not rr:
            raise SystemExit(f"interval {t0}->{t1} shares no replicate between its endpoints")
        dvec.append(np.mean([Erep[r][t1] - Erep[r][t0] for r in rr], 0))
        levvec.append(np.mean([Erep[r][t0] for r in rr], 0))
        nrep.append(len(rr))
    dvec, levvec = np.array(dvec, np.float32), np.array(levvec, np.float32)

    G = np.tile(gi, nint)
    K = np.repeat(np.arange(1, nint + 1), len(gi))
    dE = dvec[K - 1, G]
    prev = np.where(K >= 2, dvec[np.maximum(K - 2, 0), G], 0.0).astype(np.float32)
    lev = levvec[K - 1, G]
    dt = np.array([shared[k] - shared[k - 1] for k in K], np.float32)

    report(f"\n  PANEL {label}")
    report(f"    {len(shared)} timepoints {shared}")
    report(f"    {len(G):,} (gene, interval) rows over {nint} intervals, {len(gi):,} genes")
    report(f"    paired-replicate differences; replicates per interval "
           f"{nrep} (min {min(nrep)}, {sum(1 for k in range(1, len(shared)) if len({r for r in Erep if shared[k] in Erep[r]}) != len({r for r in Erep if shared[k-1] in Erep[r]}))} "
           f"intervals cross a composition change)")

    def occ_block(tgt, perm_time=None, perm_gene=None, static=False, key="OCC"):
        """Occupancy at the interval's start plus its change over the previous interval.

        `static=True` replaces both with the gene's TIME-AVERAGED occupancy, constant across intervals. That
        arm is the decisive one for separating two very different claims -- "GR sites tell you WHICH genes
        move" from "GR occupancy tells you WHEN they move". If a constant per-gene vector scores as well as
        the time-resolved one, only the first claim is supported, and permuted time cannot distinguish them
        because a permutation of a nearly-constant series is nearly the same series.
        """
        A, idx = B[key][tgt], {t: i for i, t in enumerate(B["TT"][tgt])}
        cols = [idx[t] for t in shared]                 # every panel timepoint exists for every target
        gmap = np.arange(A.shape[1]) if perm_gene is None else perm_gene
        gg = gmap[G]
        if static:
            mean = A[cols].mean(0)                      # (genes, rings)
            return np.hstack([mean[gg], np.zeros((len(G), A.shape[2]), np.float32)]).astype(np.float32)
        order = np.arange(len(shared)) if perm_time is None else perm_time
        kk = order[K - 1]                               # which occupancy timepoint this interval sees
        cur = np.array([cols[i] for i in kk])
        pre = np.array([cols[max(i - 1, 0)] for i in kk])
        lvl = A[cur, gg]
        return np.hstack([lvl, lvl - A[pre, gg]]).astype(np.float32)

    IV = np.zeros((len(G), nint), np.float32)
    IV[np.arange(len(G)), K - 1] = 1.0
    HIST = np.column_stack([lev, prev, np.log10(dt)]).astype(np.float32)
    ARMS = {"interval only": IV, "+ gene history": np.hstack([IV, HIST])}
    for t in tg:
        ARMS[f"+ {t}"] = np.hstack([IV, HIST, occ_block(t)])
        ARMS[f"+ {t} [qnorm]"] = np.hstack([IV, HIST, occ_block(t, key="OCCQ")])
        ARMS[f"+ {t} [static]"] = np.hstack([IV, HIST, occ_block(t, static=True)])

    Y = {"raw change": dE, "gene+interval demeaned": demean(dE, [G, K])}
    res = {"label": label, "times": shared, "n_rows": int(len(G)), "n_genes": int(len(gi)),
           "targets": tg, "arms": {}}
    for tname, y in Y.items():
        report(f"\n    held-out R2, chromosome folds x {len(SEEDS)} seeds -- target: {tname}")
        report(f"      {'arm':22s} {'R2':>9s} {'vs history':>12s}")
        base, res["arms"][tname] = None, {}
        for aname, X in ARMS.items():
            sc = [r2(y, fit_oof_reg(X, y, folds_chrom(chrom[G], s))) for s in SEEDS]
            v = float(np.mean(sc))
            if aname == "+ gene history":
                base = v
            d = "" if aname in ("interval only", "+ gene history") else f"{v-base:+12.4f}"
            report(f"      {aname:22s} {v:9.4f} {d:>12s}   (sd {np.std(sc):.4f})")
            res["arms"][tname][aname] = {"r2": v, "sd": float(np.std(sc)), "per_seed": sc}
        res["arms"][tname]["_base"] = base

    # ---- controls, on the strict target ----
    # NOT ONLY FOR NR3C1. Running controls on the prespecified driver alone would leave the LARGEST effect in
    # the table unexamined, which is exactly where a confound would hide. Whichever target gains most gets the
    # same treatment as GR.
    y = Y["gene+interval demeaned"]
    base = res["arms"]["gene+interval demeaned"]["_base"]
    mu = E.mean(0)
    ga = res["arms"]["gene+interval demeaned"]
    best = max([t for t in tg], key=lambda t: ga.get(f"+ {t}", {}).get("r2", -9))
    who = ["NR3C1"] + ([best] if best != "NR3C1" else [])
    res["controls"], res["halves"] = {}, {}
    nR = len(RINGS)
    for tgt in who:
        report(f"\n    CONTROLS on the demeaned target, {tgt} block ({N_PERM} draws each)")
        res["controls"][tgt] = {}
        for cname, mk in (("permuted time",
                           lambda rng, t=tgt: occ_block(t, perm_time=rng.permutation(len(shared)))),
                          ("swapped genes",
                           lambda rng, t=tgt: occ_block(
                               t, perm_gene=gene_swap(rng, B["OCC"][t].shape[1], gi, mu)))):
            vals = []
            for s in range(N_PERM):
                rng = np.random.default_rng(1000 + s)
                X = np.hstack([IV, HIST, mk(rng)])
                vals.append(r2(y, fit_oof_reg(X, y, folds_chrom(chrom[G], s % len(SEEDS)))))
            report(f"      {cname:22s} {np.mean(vals):9.4f} {np.mean(vals)-base:+12.4f}   "
                   f"(max of {N_PERM} draws {max(vals)-base:+.4f})")
            res["controls"][tgt][cname] = {"mean": float(np.mean(vals)), "max": float(max(vals)),
                                           "gain_mean": float(np.mean(vals) - base),
                                           "gain_max": float(max(vals) - base)}
        ob = occ_block(tgt)
        for half, cols in (("level only", list(range(nR))), ("change only", list(range(nR, 2 * nR)))):
            X = np.hstack([IV, HIST, ob[:, cols]])
            v = float(np.mean([r2(y, fit_oof_reg(X, y, folds_chrom(chrom[G], s))) for s in SEEDS]))
            report(f"      {tgt} {half:16s} {v:9.4f} {v-base:+12.4f}")
            res["halves"].setdefault(tgt, {})[half] = {"r2": v, "gain": float(v - base)}
    # ---- where along the course does a block earn its gain? ----
    # Comparing the 16-point and 11-point panels would answer this, and this module forbids exactly that:
    # they differ in rows, intervals and interval lengths all at once, so a difference between them is not
    # attributable. Splitting THIS panel's intervals into an early and a late window keeps genes, features
    # and processing identical and changes only which part of the response is scored.
    if nint >= 8:
        cut = nint // 2
        res["window"] = {}
        report(f"\n    WINDOW SPLIT within this panel (same genes, same features, same processing)")
        report(f"      {'window':16s} {'rows':>8s} {'history':>9s}" +
               "".join(f"{t[:9]:>12s}" for t in who))
        for wname, msk in (("early intervals", K <= cut), ("late intervals", K > cut)):
            ch_w, y_w = chrom[G[msk]], y[msk]
            bw = float(np.mean([r2(y_w, fit_oof_reg(np.hstack([IV, HIST])[msk], y_w,
                                                    folds_chrom(ch_w, s))) for s in SEEDS]))
            row = f"      {wname:16s} {int(msk.sum()):8,d} {bw:9.4f}"
            res["window"][wname] = {"n": int(msk.sum()), "history": bw, "gain": {}}
            for tgt in who:
                X = ARMS[f"+ {tgt}"][msk]
                vw = float(np.mean([r2(y_w, fit_oof_reg(X, y_w, folds_chrom(ch_w, s))) for s in SEEDS]))
                row += f"{vw-bw:+12.4f}"
                res["window"][wname]["gain"][tgt] = float(vw - bw)
            report(row + f"   (intervals {'1' if 'early' in wname else cut+1}-"
                         f"{cut if 'early' in wname else nint})")

    res["best_target"] = best
    res["dE"], res["G"], res["K"] = dE, G, K
    res["gain_nr3c1"] = occ_block("NR3C1")[:, nR:].sum(1)
    return res


def main():
    from scipy import stats
    log = []

    def report(s):
        print(s)
        log.append(s)

    report("=" * 100)
    report("GR OCCUPANCY -> EXPRESSION CHANGE: Stage 1 with a real driver and a real clock")
    report("=" * 100)
    if not (DIR / "manifest.json").exists():
        raise SystemExit(f"absent: {DIR/'manifest.json'} -- run fetch_gr_timecourse.py first")
    B = build()
    report(f"  {len(B['names']):,} genes mapped to a TSS; RNA timepoints {B['times']}")
    for t in sorted(B["TT"]):
        pk = B["n_peaks"][t]
        report(f"    {t:7s} {len(B['TT'][t]):2d} timepoints {B['TT'][t]}")
        report(f"            peaks {min(pk):,}-{max(pk):,} ({max(pk)/max(min(pk),1):.1f}x swing across the "
               f"course -- why the qnorm arm exists)")

    expressed = (B["E"] > np.log2(1 + MIN_TPM)).mean(0) >= 0.5
    gi = np.where(expressed)[0]
    report(f"  expressed (TPM>{MIN_TPM:g} at >=50% of timepoints): {len(gi):,}")

    # RELIABILITY OF A CHANGE, from the two biological replicates -- and of the DEMEANED change separately.
    # Two-way demeaning is the strict target, and it strips real signal along with the fixed effects; if its
    # reliability were near zero the strict arm would be measuring noise and a null there would say nothing.
    # Reporting only the raw number would hide that.
    rel = relD = float("nan")
    if len(B["Erep"]) >= 2:
        ra, rb = sorted(B["Erep"])[:2]
        a, b = B["Erep"][ra], B["Erep"][rb]
        ok = [t for t in B["times"] if t in a and t in b]
        if len(ok) >= 3:
            gg = np.tile(gi, len(ok) - 1)
            kk = np.repeat(np.arange(1, len(ok)), len(gi))
            d1 = np.concatenate([a[ok[k]][gi] - a[ok[k - 1]][gi] for k in range(1, len(ok))])
            d2 = np.concatenate([b[ok[k]][gi] - b[ok[k - 1]][gi] for k in range(1, len(ok))])
            rel = float(np.corrcoef(d1, d2)[0, 1])
            relD = float(np.corrcoef(demean(d1, [gg, kk]), demean(d2, [gg, kk]))[0, 1])
        report(f"  REPLICATE RELIABILITY over {len(ok)} shared timepoints -- the ceiling any R2 is measured "
               f"against")
        report(f"    raw change              r = {rel:.4f}   (R2 ceiling ~ {rel:.3f})")
        report(f"    gene+interval demeaned  r = {relD:.4f}   (R2 ceiling ~ {relD:.3f})")
    else:
        report("  only one RNA replicate series; no reliability ceiling available")

    R = {"n_genes_expressed": int(len(gi)), "reliability_of_change": rel,
         "reliability_of_demeaned_change": relD,
         "rings": [list(r) for r in RINGS], "panels": {}}
    panels = {}
    for label, tg in PANELS.items():
        p = run_panel(B, label, tg, gi, report)
        if p:
            panels[label] = p
            R["panels"][label] = {k: v for k, v in p.items()
                                  if k not in ("dE", "G", "K", "gain_nr3c1")}

    # ---- sign test, on the panel with the finest early resolution ----
    key = next(iter(panels))
    P = panels[key]
    dE, G, K, gain = P["dE"], P["G"], P["K"], P["gain_nr3c1"]
    dEi = demean(dE, [K])                 # interval mean removed; gene-level response deliberately kept
    report(f"\n  SIGN TEST on {key} -- genes GAINING GR occupancy should go UP (direction fixed in advance)")
    pos_g = gain[gain > 0]
    if len(pos_g) > 200:
        thr = float(np.quantile(pos_g, 0.9))
        hi, flat = gain >= thr, gain == 0
        up_hi, up_fl = float((dEi[hi] > 0).mean()), float((dEi[flat] > 0).mean())
        ps = stats.fisher_exact([[int((dEi[hi] > 0).sum()), int((dEi[hi] <= 0).sum())],
                                 [int((dEi[flat] > 0).sum()), int((dEi[flat] <= 0).sum())]])[1]
        report(f"    top-decile GR gain  n {int(hi.sum()):7,d}  frac up {up_hi:.4f}  "
               f"median dlog2 {np.median(dEi[hi]):+.4f}")
        report(f"    no GR change        n {int(flat.sum()):7,d}  frac up {up_fl:.4f}  "
               f"median dlog2 {np.median(dEi[flat]):+.4f}")
        report(f"    Fisher p {ps:.3g}")
        R["sign"] = {"frac_up_gain": up_hi, "frac_up_flat": up_fl, "p": float(ps),
                     "n_gain": int(hi.sum()), "n_flat": int(flat.sum()),
                     "median_gain": float(np.median(dEi[hi])), "median_flat": float(np.median(dEi[flat]))}
    else:
        R["sign"] = None
        report("    too few gaining rows to test")

    # ---- verdict ----
    report("\n" + "=" * 100)
    pk = [k for k in panels if "11-point" in k]
    PP = panels[pk[0]] if pk else panels[key]
    ar = PP["arms"]["gene+interval demeaned"]
    base = ar["_base"]

    def gain(name):
        return ar[name]["r2"] - base if name in ar else float("nan")

    gr = gain("+ NR3C1")
    grs, grq = gain("+ NR3C1 [static]"), gain("+ NR3C1 [qnorm]")
    ctlmax = max(v["gain_max"] for v in PP["controls"]["NR3C1"].values())
    plac = [(p, gain(f"+ {p}")) for p in PLACEBO if f"+ {p}" in ar]
    pbest = max([v for _n, v in plac], default=float("-inf"))
    hv = PP.get("halves", {}).get("NR3C1", {})
    best = PP.get("best_target")
    R["summary"] = {"panel": PP["label"], "history_r2": base, "nr3c1_gain": gr,
                    "nr3c1_static_gain": grs, "nr3c1_qnorm_gain": grq,
                    "best_control_gain": ctlmax, "placebo": dict(plac), "best_target": best,
                    "halves": {k: v["gain"] for k, v in hv.items()}}
    htxt = ("" if not hv else
            f" Split by half, the occupancy LEVEL contributes "
            f"{hv.get('level only', {}).get('gain', float('nan')):+.4f} and the occupancy CHANGE "
            f"{hv.get('change only', {}).get('gain', float('nan')):+.4f}.")
    ptxt = "" if not plac else " Placebo: " + ", ".join(f"{n} {v:+.4f}" for n, v in plac) + "."
    qtxt = (f" Under per-timepoint quantile normalisation, which makes assay depth irrelevant, the gain is "
            f"{grq:+.4f}.")
    btxt = ""
    if best and best != "NR3C1" and f"+ {best}" in ar:
        bc = PP["controls"].get(best, {})
        bmax = max([v["gain_max"] for v in bc.values()], default=float("nan"))
        bh = PP.get("halves", {}).get(best, {})
        btxt = (f" The largest block in the table is {best} ({gain('+ ' + best):+.4f}); its own permutation "
                f"controls reach {bmax:+.4f}, its static arm {gain(f'+ {best} [static]'):+.4f}, and its "
                f"change half {bh.get('change only', {}).get('gain', float('nan')):+.4f} -- reported here "
                f"because leaving the biggest effect uncontrolled is where a confound would hide.")

    # THE DISCRIMINATOR IS STATIC-VS-TIME-RESOLVED, NOT SIGNIFICANCE. A permutation of a nearly constant
    # series is nearly the same series, so permuted time cannot by itself separate "GR sites tell you WHICH
    # genes move" from "GR occupancy tells you WHEN". The static arm can, and it is read first.
    static_explains = np.isfinite(grs) and gr > 0 and grs >= 0.7 * gr
    if gr <= 0.002 or gr <= ctlmax:
        v = (f"NULL FOR A THIRD TIME -- and this one had every excuse removed. On {PP['label']}, with a real "
             f"perturbation, a real clock and matched readouts in one cell line, NR3C1 occupancy adds "
             f"{gr:+.4f} held-out R2 over interval identity and gene history on the gene- and "
             f"interval-demeaned change, against {ctlmax:+.4f} for the best permutation control. Replicate "
             f"reliability of the change is {rel:.3f} raw and {relD:.3f} demeaned, so the change is "
             f"measurable and this is not a ceiling artefact." + qtxt + ptxt + htxt + btxt)
    elif static_explains:
        v = (f"IT TELLS YOU WHICH GENES, NOT WHEN. On {PP['label']} NR3C1 occupancy adds {gr:+.4f} held-out "
             f"R2 over interval identity and gene history -- but a CONSTANT per-gene vector, the gene's "
             f"time-averaged occupancy with no time resolution at all, recovers {grs:+.4f} of that "
             f"({100*grs/max(gr,1e-9):.0f}%). Permuting the time axis costs almost nothing "
             f"({ctlmax:+.4f}) for the same reason, while swapping gene identity destroys it. So GR site "
             f"occupancy identifies which genes have volatile trajectories; it does not say when they move. "
             f"That is a static annotation, not dynamics, and Stage 1 remains unsupported." + qtxt + ptxt
             + htxt + btxt)
    elif pbest >= gr * 0.7:
        v = (f"NOT SPECIFIC TO GR. NR3C1 occupancy adds {gr:+.4f} on {PP['label']}, but an architectural "
             f"placebo adds {pbest:+.4f} -- {100*pbest/max(gr,1e-9):.0f}% of it. What is being measured "
             f"tracks peak density or accessibility rather than glucocorticoid signalling, so it cannot be "
             f"read as the driver explaining the response." + qtxt + ptxt + htxt + btxt)
    else:
        v = (f"OCCUPANCY DYNAMICS PREDICT EXPRESSION DYNAMICS. On {PP['label']}, NR3C1 occupancy adds "
             f"{gr:+.4f} held-out R2 over interval identity and gene history on the gene- and "
             f"interval-demeaned change, against {ctlmax:+.4f} for the best permutation control and "
             f"{grs:+.4f} for a time-averaged constant. Against a replicate-reliability ceiling of "
             f"{relD:.3f} that is {100*gr/max(relD,1e-9):.1f}% of what is measurable." + qtxt + ptxt + htxt
             + btxt)
    R["verdict"] = v
    report(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "gr_dynamics.json", "w"), indent=1, default=float)
    report(f"\n  -> {OUT/'gr_dynamics.json'}")


if __name__ == "__main__":
    main()
