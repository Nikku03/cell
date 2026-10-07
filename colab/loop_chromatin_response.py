"""Loop 245. Chromatin state against the one thing that has carried signal all along.

WHAT THE ARC POINTS AT. Loops 242 and 243 measured the regulatory graph explaining about 0.05 of a
knockdown's transcriptome-wide response, with the arrows decorative at genome scale (Q6: 102%
survived reversing every edge) and the sign real but small (Q2: -3.0 se, R4: reversal removes 53%
locally). Meanwhile the quantity that keeps carrying signal in this project is not WHICH
perturbation was applied but WHICH GENES MOVE AT ALL -- loop 224's reliability, loop 240's
catalogue arm, loop 241's gene-level facts. So the question chromatin should be asked is not
"which gene did this knockdown hit" but:

    does a gene's CHROMATIN STATE predict how much that gene responds to perturbation at all?

Responsiveness is defined per gene as the standard deviation of its change across all 8,917
screened perturbations. It is a property of the gene, not of any perturbation.

THE DATA IS MATCHED TO THE CELL LINE, WHICH MATTERS AND IS NOT AUTOMATIC. Every mark here is K562,
the same line the Perturb-seq was run in, GRCh38, ENCODE replicated or IDR-thresholded peaks:

    H3K4me3   ENCFF403DTU     active promoter
    H3K27ac   ENCFF706DHK     active promoter and enhancer
    H3K4me1   ENCFF486QGD     enhancer, primed or active
    H3K36me3  ENCFF829XLF     transcribed gene body
    H3K27me3  ENCFF578EES     polycomb repression
    H3K9me3   ENCFF963GZJ     constitutive heterochromatin
    ATAC      ENCFF926KTI     accessibility

The twenty replication-timing files already on disk were NOT used and the reason is recorded here
rather than left silent: querying ENCODE for each accession showed none of them is K562 -- they are
erythroid progenitor, CyT49, hepatocyte, myoblast and so on. A cross-cell-type chromatin feature
dressed as a K562 feature would have been the loop 240 defect again, in a new costume.

T3 IS THE GATE THIS LOOP EXISTS FOR. Chromatin marks largely ENCODE expression level: an active
promoter is what a highly expressed gene has. And highly expressed genes are measured better, so
they move more detectably. A model given H3K4me3 that beats a model given nothing has shown almost
nothing; it has to beat a model given the gene's own BASELINE EXPRESSION LEVEL, which the
Perturb-seq file supplies directly. This is loop 241's W3 and loop 244's S3 in a third costume, and
the reason it keeps being needed is that every one of those comparisons came back the wrong way.

PREDECLARED, BEFORE ANY NUMBER.

  T1 IS RESPONSIVENESS A REAL GENE PROPERTY AT ALL?
     The perturbations split in half at random, responsiveness computed independently in each half,
     correlated across genes. If a gene's responsiveness does not reproduce between two halves of
     the same experiment, there is nothing for chromatin to predict and everything below is
     measuring noise.
     Gate: PASS iff the split-half correlation exceeds 0.70. Everything else requires this.

  T2 DOES CHROMATIN PREDICT RESPONSIVENESS?      -- requires T1
     Ridge on the seven marks, held out BY GENE, ten folds.
     Gate: PASS iff the held-out correlation exceeds 0.20.

  T3 DOES CHROMATIN BEAT THE GENE'S OWN EXPRESSION LEVEL?      -- requires T1
     The same ridge against a model given only baseline expression level, and against the two
     combined, paired over held-out genes.
     Gate: PASS iff chromatin-plus-expression exceeds expression-alone by at least 0.05. A FAIL
     means the marks are a proxy for expression level and carry nothing beyond it.

  T4 ARE BIVALENT GENES MORE RESPONSIVE?      -- requires T1
     H3K4me3 and H3K27me3 together is the textbook poised promoter: ready to move in either
     direction. This is a first-principles directional prediction, not something read off this
     data, and it is the one place chromatin makes a claim expression level cannot.
     Gate: PASS iff genes carrying both marks are more responsive than genes carrying H3K4me3
     alone, matched on baseline expression level so the comparison is not just "bivalent genes are
     expressed differently", by at least 2 standard errors.

  T5 CONTROL: CHROMATIN REASSIGNED TO THE WRONG GENES.      -- requires T2, VOID if T2 is under 0.05
     Every gene's mark vector replaced by another gene's, everything else identical.
     Gate: PASS iff the held-out correlation collapses to under 25% of its true value.

  T6 DOES CHROMATIN ADD TO THE REGULATORY GRAPH?      -- requires T1
     Loop 242's graph features and these marks, alone and together, predicting responsiveness.
     Gate: PASS iff the combination beats the better single source by at least 0.02. Gated against
     the BEST SINGLE and not against the graph alone, for the reason loop 243's R6 established:
     beating the weaker of two sources proves nothing about whether the second adds information.

  T7 WHAT THIS CANNOT SHOW -- written before the run.
     Responsiveness is a variance. A gene with low expression has few counts and its variance is
     dominated by sampling noise, so any feature correlated with expression will predict it partly
     for a measurement reason rather than a biological one. T3 bounds this and does not remove it.
     Peak calls are binary summaries of a continuous signal, and a gene just below a peak-calling
     threshold is scored identically to a gene with no mark at all.
     One cell line. Chromatin state is the most cell-type-specific data in this project, so
     nothing here transfers to A549 or RPE1 without being re-measured there.
     The marks and the Perturb-seq come from different laboratories and different K562 stocks.
"""
import os, sys, json, time, gzip, collections, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_chromatin_response.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
K562 = SCR / "perturbseq" / "K562_gwps_normalized_bulk_01.h5ad"
EPI = SCR / "epi"
BG = SCR / "biogrid_hs_edges.tsv.gz"

SEED, NFOLD = 245245, 10
MARKS = ["H3K4me3", "H3K27ac", "H3K4me1", "H3K36me3", "H3K27me3", "H3K9me3", "ATAC"]
NARROW, BROAD = 2000, 20000
T1_BAR, T2_BAR, T3_BAR, T4_SE, T5_MAX, T6_BAR = 0.70, 0.20, 0.05, 2.0, 0.25, 0.02

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5: return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if d.size < 3: return float("nan"), float("nan"), float("nan")
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    mu = float(np.mean(d))
    return mu, se, (mu / se if se > 0 else float("nan"))


def ridge_cv(F, y, folds, lam_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Held-out-by-gene ridge with lambda chosen on an inner split of the training genes."""
    out = np.full(len(y), np.nan)
    rr = np.random.default_rng(0)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        tr = tr[rr.permutation(len(tr))]   # setdiff1d returns SORTED indices and gene order is
        Xtr, ytr = F[tr], y[tr]            # genomic, so an unshuffled inner split is by position
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Z = (Xtr - mu) / sd
        Z = np.concatenate([Z, np.ones((len(Z), 1))], 1)
        n = len(tr); cut = int(0.8 * n)
        best, bl = -9, lam_grid[0]
        for lam in lam_grid:
            A = Z[:cut].T @ Z[:cut] + lam * n * np.eye(Z.shape[1])
            b = np.linalg.solve(A, Z[:cut].T @ ytr[:cut])
            s = pear(Z[cut:] @ b, ytr[cut:])
            if np.isfinite(s) and s > best: best, bl = s, lam
        A = Z.T @ Z + bl * n * np.eye(Z.shape[1])
        b = np.linalg.solve(A, Z.T @ ytr)
        Zt = np.concatenate([(F[te] - mu) / sd, np.ones((len(te), 1))], 1)
        out[te] = Zt @ b
    return out


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "K562 chromatin state predicting per-gene responsiveness to perturbation"}
    say("=" * 104)
    say("LOOP 245 -- CHROMATIN STATE AND WHICH GENES MOVE AT ALL")
    say("=" * 104)
    say("     The graph explains ~0.05 of a knockdown's response (loops 242-243). The quantity")
    say("     that keeps carrying signal is not which perturbation was applied but which genes")
    say("     move at all. That is what chromatin is asked here.")
    say("     Every mark is K562, matched to the Perturb-seq line. The 20 replication-timing")
    say("     files on disk were NOT used: an ENCODE query per accession showed none is K562")
    say("     (erythroid progenitor, CyT49, hepatocyte, myoblast ...). A cross-cell-type feature")
    say("     dressed as a K562 feature would be loop 240's defect in a new costume.")

    # ---------------------------------------------------------------- Perturb-seq
    import h5py
    f = h5py.File(K562, "r")
    cats = f["var"]["__categories"]["gene_name"][:]
    cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
    gname = cats[f["var"]["gene_name"][:]]
    clean_mean = f["var"]["clean_mean"][:] if "clean_mean" in f["var"] else None
    X = f["X"][:]
    f.close()
    gcol = np.isfinite(X).all(0)
    rrow = np.isfinite(X[:, gcol]).all(1)
    MG = gname[gcol]; X = X[np.ix_(rrow, gcol)]
    expr = np.asarray(clean_mean, float)[gcol] if clean_mean is not None else None
    say(f"     Perturb-seq: {X.shape[0]:,} screened perturbations x {X.shape[1]:,} genes")

    resp = X.std(0)
    say(f"     responsiveness = sd of each gene's change across all {X.shape[0]:,} perturbations")

    # ---------------------------------------------------------------- T1
    say("T1 IS RESPONSIVENESS A REAL GENE PROPERTY AT ALL?")
    p = rng.permutation(X.shape[0]); h = len(p) // 2
    r1, r2 = X[p[:h]].std(0), X[p[h:]].std(0)
    sh = pear(r1, r2)
    say(f"     perturbations split in half, responsiveness computed independently in each: "
        f"r = {sh:.4f}")
    G.add("T1", bool(sh >= T1_BAR), stat=float(sh),
          if_true=lambda: f"T1 PASS -- responsiveness reproduces at {sh:.4f} between halves",
          if_false=lambda: f"T1 FAIL -- {sh:.4f} against a {T1_BAR} bar; there is nothing stable "
                           f"for chromatin to predict")
    res["split_half"] = sh

    # ---------------------------------------------------------------- chromatin
    say("     loading K562 peaks and building per-gene mark features ...")
    tss = collections.defaultdict(list)
    with open(EPI / "tss.tsv") as fh:
        for ln in fh:
            p_ = ln.rstrip("\n").split("\t")
            if len(p_) < 4 or not p_[0]: continue
            if p_[1] in ("MT",) or len(p_[1]) > 2: continue
            try: tss[p_[0]].append((f"chr{p_[1]}", int(p_[2])))
            except ValueError: continue
    say(f"     {len(tss):,} gene symbols with a TSS")

    peaks = {}
    for m in MARKS:
        by = collections.defaultdict(list)
        with gzip.open(EPI / f"{m}.bed.gz", "rt") as fh:
            for ln in fh:
                q = ln.rstrip("\n").split("\t")
                if len(q) < 7: continue
                by[q[0]].append((int(q[1]), int(q[2]), float(q[6])))
        peaks[m] = {c: (np.array([x[0] for x in sorted(v)]),
                        np.array([x[1] for x in sorted(v)]),
                        np.array([x[2] for x in sorted(v)])) for c, v in by.items()}
        say(f"       {m:<9} {sum(len(v[0]) for v in peaks[m].values()):,} peaks")

    def mark_at(m, chrom, pos, half):
        d = peaks[m].get(chrom)
        if d is None: return 0.0, 0.0
        s, e, v = d
        lo, hi = pos - half, pos + half
        i = np.searchsorted(e, lo, "right"); j = np.searchsorted(s, hi, "left")
        if j <= i: return 0.0, 0.0
        ov = np.minimum(e[i:j], hi) - np.maximum(s[i:j], lo)
        ov = ov[ov > 0]
        return (float(ov.sum()) / (2 * half)), float(v[i:j].max())

    usable = [g for g in MG if g in tss]
    gi = {g: i for i, g in enumerate(MG)}
    idx = np.array([gi[g] for g in usable])
    FN = []
    for g in usable:
        chrom, pos = tss[g][0]
        row = []
        for m in MARKS:
            cn, sn = mark_at(m, chrom, pos, NARROW)
            cb, sb = mark_at(m, chrom, pos, BROAD)
            row += [cn, np.log1p(sn), cb]
        FN.append(row)
    FN = np.asarray(FN, np.float64)
    y = resp[idx]
    ex = np.log1p(expr[idx]) if expr is not None else None
    say(f"     {len(usable):,} genes have both a response and a TSS; "
        f"{FN.shape[1]} chromatin features each")
    res["n_genes"] = len(usable)

    order = rng.permutation(len(usable))
    folds = [order[i::NFOLD] for i in range(NFOLD)]

    # ---------------------------------------------------------------- T2
    say("T2 DOES CHROMATIN PREDICT RESPONSIVENESS?")
    pc = ridge_cv(FN, y, folds)
    r_chrom = pear(pc, y)
    say(f"     ridge on {FN.shape[1]} chromatin features, held out by gene, {NFOLD} folds: "
        f"r = {r_chrom:.4f}")
    G.add("T2", bool(r_chrom >= T2_BAR), stat=float(r_chrom), requires=("T1",),
          if_true=lambda: f"T2 PASS -- chromatin predicts responsiveness at {r_chrom:.4f}",
          if_false=lambda: f"T2 FAIL -- {r_chrom:.4f} against a {T2_BAR} bar")
    res["T2"] = {"chromatin": r_chrom}

    # ---------------------------------------------------------------- T3
    say("T3 DOES CHROMATIN BEAT THE GENE'S OWN EXPRESSION LEVEL?")
    if ex is None:
        G.add("T3", False, stat=float("nan"), requires=("T1",), void_if=True,
              void_reason="the Perturb-seq file carries no baseline expression column")
        r_expr = r_both = float("nan")
    else:
        pe = ridge_cv(ex[:, None], y, folds)
        pb = ridge_cv(np.concatenate([FN, ex[:, None]], 1), y, folds)
        r_expr, r_both = pear(pe, y), pear(pb, y)
        say(f"     expression level alone:      {r_expr:.4f}")
        say(f"     chromatin alone:             {r_chrom:.4f}")
        say(f"     both together:               {r_both:.4f}")
        pf = np.array([pear(pb[te], y[te]) for te in folds])
        pg = np.array([pear(pe[te], y[te]) for te in folds])
        d3, se3, z3 = paired(pf, pg)
        say(f"     both minus expression alone, paired over folds: {d3:+.4f} +/- {se3:.4f} "
            f"({z3:+.1f} se)")
        G.add("T3", bool(d3 >= T3_BAR), stat=float(d3), requires=("T1",),
              if_true=lambda: f"T3 PASS -- chromatin adds {d3:+.4f} beyond expression level",
              if_false=lambda: f"T3 FAIL -- chromatin adds {d3:+.4f} beyond expression level, "
                               f"against a {T3_BAR} bar; the marks are largely a proxy for it")
        res["T3"] = {"expression": r_expr, "chromatin": r_chrom, "both": r_both, "delta": d3,
                     "se": se3, "z": z3}

    # ---------------------------------------------------------------- T4
    say("T4 ARE BIVALENT GENES MORE RESPONSIVE?")
    k4 = FN[:, MARKS.index("H3K4me3") * 3] > 0
    k27 = FN[:, MARKS.index("H3K27me3") * 3] > 0
    biv, act = k4 & k27, k4 & ~k27
    say(f"     {int(biv.sum()):,} bivalent (H3K4me3 + H3K27me3), {int(act.sum()):,} H3K4me3 only")
    if ex is not None and biv.sum() >= 30 and act.sum() >= 30:
        bins = np.quantile(ex, np.linspace(0, 1, 11))
        db = []
        for i in range(10):
            m = (ex >= bins[i]) & (ex <= bins[i + 1])
            if (biv & m).sum() >= 5 and (act & m).sum() >= 5:
                db.append(y[biv & m].mean() - y[act & m].mean())
        m4, se4, z4 = paired(np.array(db), np.zeros(len(db)))
        say(f"     matched on baseline expression in {len(db)} deciles, bivalent minus "
            f"H3K4me3-only responsiveness: {m4:+.4f} +/- {se4:.4f}  ({z4:+.1f} se)")
        G.add("T4", bool(m4 > 0 and z4 >= T4_SE), stat=float(m4), requires=("T1",),
              if_true=lambda: f"T4 PASS -- poised promoters are {m4:+.4f} more responsive at "
                              f"matched expression ({z4:+.1f} se)",
              if_false=lambda: f"T4 FAIL -- bivalent minus active is {m4:+.4f} ({z4:+.1f} se)")
        res["T4"] = {"delta": m4, "se": se4, "z": z4, "n_bivalent": int(biv.sum()),
                     "n_active": int(act.sum()), "n_deciles": len(db)}
    else:
        G.add("T4", False, stat=float("nan"), requires=("T1",), void_if=True,
              void_reason=f"too few genes in one class (bivalent {int(biv.sum())}, "
                          f"active {int(act.sum())}) or no expression column")

    # ---------------------------------------------------------------- T5
    say("T5 CONTROL: CHROMATIN REASSIGNED TO THE WRONG GENES")
    if r_chrom < 0.05:
        G.add("T5", False, stat=float(r_chrom), requires=("T2",), void_if=True,
              void_reason=f"the real chromatin correlation is {r_chrom:.4f}; nothing to collapse")
    else:
        ps = ridge_cv(FN[rng.permutation(len(usable))], y, folds)
        r_sh = pear(ps, y)
        f5 = r_sh / r_chrom
        say(f"     mark vectors permuted across genes: {r_sh:.4f} against a real {r_chrom:.4f} "
            f"({f5:.0%})")
        G.add("T5", bool(f5 <= T5_MAX), stat=float(f5), requires=("T2",),
              if_true=lambda: f"T5 PASS -- collapses to {f5:.0%} on the wrong genes",
              if_false=lambda: f"T5 FAIL -- {f5:.0%} survives reassignment")
        res["T5"] = {"real": r_chrom, "shuffled": r_sh, "fraction": f5}

    # ---------------------------------------------------------------- T6
    say("T6 DOES CHROMATIN ADD TO THE REGULATORY GRAPH?")
    deg = collections.Counter()
    with gzip.open(BG, "rt") as fh:
        for ln in fh:
            q = ln.rstrip("\n").split("\t")
            if len(q) < 2 or q[0] == q[1]: continue
            deg[q[0]] += 1; deg[q[1]] += 1
    gr = np.array([[np.log1p(deg.get(g, 0))] for g in usable])
    pgh = ridge_cv(gr, y, folds)
    pboth = ridge_cv(np.concatenate([FN, gr], 1), y, folds)
    r_graph, r_cg = pear(pgh, y), pear(pboth, y)
    say(f"     graph degree alone: {r_graph:.4f}   chromatin alone: {r_chrom:.4f}   "
        f"both: {r_cg:.4f}")
    single = "chromatin" if r_chrom >= r_graph else "graph"
    base = np.array([pear((pc if r_chrom >= r_graph else pgh)[te], y[te]) for te in folds])
    comb = np.array([pear(pboth[te], y[te]) for te in folds])
    d6, se6, z6 = paired(comb, base)
    say(f"     best single is {single}; combined minus it: {d6:+.4f} +/- {se6:.4f} ({z6:+.1f} se)")
    G.add("T6", bool(d6 >= T6_BAR), stat=float(d6), requires=("T1",),
          if_true=lambda: f"T6 PASS -- combining adds {d6:+.4f} over {single} alone",
          if_false=lambda: f"T6 FAIL -- combining adds {d6:+.4f} over {single} alone, against a "
                           f"{T6_BAR} bar")
    res["T6"] = {"graph": r_graph, "chromatin": r_chrom, "both": r_cg, "best_single": single,
                 "delta": d6, "se": se6, "z": z6}

    say("T7 WHAT THIS CANNOT SHOW")
    say("     Responsiveness is a variance. A gene with low expression has few counts and its")
    say("     variance is dominated by sampling noise, so any feature correlated with expression")
    say("     predicts it partly for a measurement reason. T3 bounds this; it does not remove it.")
    say("     Peak calls are binary summaries of a continuous signal: a gene just below a calling")
    say("     threshold scores identically to a gene with no mark at all.")
    say("     One cell line. Chromatin is the most cell-type-specific data in this project, so")
    say("     nothing here transfers to A549 or RPE1 without being re-measured.")
    say("     The marks and the Perturb-seq come from different laboratories and K562 stocks.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary(seconds=res["seconds"])
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
