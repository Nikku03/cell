"""Loop 238. Holding out a CELL LINE instead of a gene -- the test this project never ran.

THE GAP THIS EXISTS TO CLOSE, stated as I stated it when asked whether the cell can be predicted:
every held-out split in this project holds out GENES, never CONDITIONS. Loop 229's 20 paired
splits, loop 235's stack, loop 237's 0.9525 -- all of them train on some genes and test on others
WITHIN one experiment. We have never once predicted a drug, dose, timepoint or cell type we did not
already have data for. The one time something close was measured -- K562-to-RPE1 Perturb-seq
transfer, loop 224 X6 -- it came back at 0.2286.

DepMap makes the real test possible. 1,178 cell lines x 17,916 genes of CRISPR gene effect, with
matched transcriptomes for 1,517 lines and lineage annotation for all of them. A549, K562, HepG2,
Jurkat and MCF7 are all present. Holding out a CELL LINE and predicting its dependency profile is a
condition-level generalisation test, and it is the one that "can we predict the cell" actually
means.

THE TRAP, AND WHY X2 RUNS BEFORE ANY MODEL IS SCORED. Most genes are essential in every cell line
or in none. A predictor that ignores the held-out line entirely -- just the mean gene effect across
the training lines -- will therefore score extremely well on all 17,916 genes, and that score says
nothing about whether the cell line was understood. It is the same shape as loop 201's P2, where
84.1% of negatives had out-degree zero and one rule reached AUC 0.9206 while measuring nothing.
X2 measures how the variance actually splits between genes and between lines, and X5 restricts to
the genes where the question is real. Reporting only the all-gene number would be the vacuous
version of this loop.

FOUR ARMS, each adding one kind of knowledge about the held-out line:

    MEAN        mean gene effect over training lines. Knows NOTHING about the held-out line.
    LINEAGE     mean over training lines of the same OncotreeLineage. Knows its tissue.
    EXPR        ridge on the held-out line's own transcriptome. Knows its expression state.
    NEIGHBOUR   mean over the k training lines most transcriptionally similar to it.

Only the last three can express anything cell-line-specific, and the comparison against MEAN is the
whole measurement.

PREDECLARED, BEFORE ANY NUMBER.

  X1 IS THE DATA WHAT IT CLAIMS TO BE?  -- everything requires it
     Ribosomal proteins and RNA polymerase subunits are essential in essentially every proliferating
     human cell, which is textbook and not chosen from this data.
     Gate: PASS iff the median gene effect across all lines is below -0.5 for a panel of RPL, RPS
     and POLR2 genes, while a panel of olfactory receptors sits above -0.1.

  X2 HOW MUCH OF THE VARIANCE IS EVEN BETWEEN LINES?
     Decompose the gene-effect matrix into a between-gene component and a between-line component.
     Gate: PASS iff the decomposition is well-posed, both components strictly positive. The RATIO
     is reported, not gated. If between-gene dominates -- which is what essentiality biology
     predicts -- then X4's all-gene numbers are near-vacuous by construction and X5 carries the
     result.

  X3 WHAT DOES KNOWING NOTHING ABOUT THE CELL LINE ACHIEVE?
     MEAN scored on held-out lines, correlation across genes, averaged over folds.
     Gate: PASS iff it exceeds 0.80. A high number here is the POINT, not a success: it establishes
     how much of the apparent performance is available without any cell-line information at all.

  X4 DO THE CELL-LINE-SPECIFIC ARMS BEAT IT ON ALL GENES?
     Gate: PASS iff the best of LINEAGE, EXPR and NEIGHBOUR exceeds MEAN by at least 0.02, paired
     across folds. Read with X2: if between-gene variance dominates this gate is easy to fail and
     hard to interpret either way.

  X5 DO THEY BEAT IT ON SELECTIVELY ESSENTIAL GENES?  -- the real test
     Restricted to the 2,000 genes with the highest variance across cell lines, where a
     line-independent predictor has nothing to say by construction.
     Gate: PASS iff the best arm exceeds MEAN by at least 0.05, paired across folds. This is the
     number that answers whether a cell line can be predicted rather than a gene.

  X6 DOES K562 PERTURB-SEQ TRANSFER TO FITNESS IN OTHER LINES?
     The magnitude of a gene's transcriptional response when knocked down in K562, against its
     DepMap fitness effect. Tested in K562 itself and in every other line.
     Gate: PASS iff the correlation in K562 exceeds the median correlation across all other lines
     by at least 0.05. A FAIL means the Perturb-seq signal predicts fitness generically rather than
     cell-type-specifically, which would explain why loop 210's gains block transferred to A549 at
     all.

  X7 CONTROL: SHUFFLED CELL-LINE IDENTITY
     The held-out line's expression replaced by another line's, everything else identical.
     Gate: PASS iff the best arm's advantage over MEAN collapses by at least half. Without this,
     any gain could be an artefact of having more parameters rather than of knowing the line.

  X8 WHAT THIS CANNOT SHOW -- written before the run.
     CRISPR fitness is not gene expression. This tests condition-level generalisation on a
     different readout from the one loops 206-237 worked on, so a result here does not transfer
     back to the A549 dexamethasone target directly.
     DepMap lines are cancer lines grown in culture, and lineage is a coarse label. A model that
     predicts a held-out line well may be exploiting lineage-level batch structure rather than
     anything about that line's biology, and X7 bounds but does not eliminate that.
     Gene effect is measured as a growth phenotype over weeks. It integrates everything downstream
     of the knockout and is far removed from the transcriptional response Perturb-seq measures,
     so X6 compares two quantities that are related by a long causal chain rather than by
     definition.
"""
import os, sys, json, csv, time, re, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_depmap_holdout.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
DM = SCR / "depmap"
K562_BULK = SCR / "perturbseq" / "K562_gwps_normalized_bulk_01.h5ad"
SEED, NFOLD, NSEL, KNN = 238238, 10, 2000, 25
X3_BAR, X4_BAR, X5_BAR, X6_BAR = 0.80, 0.02, 0.05, 0.05

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5: return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "cell-line holdout on DepMap"}
    say("=" * 104)
    say("LOOP 238 -- HOLDING OUT A CELL LINE, NOT A GENE")
    say("=" * 104)
    say("     Every split in this project so far holds out GENES within one experiment. We have")
    say("     never predicted a condition we did not already have data for. This does.")
    say("     X2 runs before any model is scored, because most genes are essential everywhere or")
    say("     nowhere and a predictor that ignores the cell line entirely will still score high.")

    z = np.load(DM / "gene_effect.npz", allow_pickle=True)
    E, egenes, elines = z["E"], np.array([str(x) for x in z["genes"]]), \
        np.array([str(x) for x in z["lines"]])
    m = np.load(DM / "model_meta.npz", allow_pickle=True)
    lin_of = dict(zip([str(x) for x in m["lines"]], [str(x) for x in m["lineage"]]))
    lineage = np.array([lin_of.get(l, "") for l in elines])
    say(f"     gene effect: {E.shape[0]:,} cell lines x {E.shape[1]:,} genes")

    # ---------------------------------------------------------------- X1
    say("X1 IS THE DATA WHAT IT CLAIMS TO BE?")
    gi = {g: i for i, g in enumerate(egenes)}
    ess = [g for g in egenes if re.match(r"^(RPL\d|RPS\d|POLR2[A-L])", g)][:60]
    olf = [g for g in egenes if g.startswith("OR")][:60]
    me = float(np.nanmedian(np.nanmedian(E[:, [gi[g] for g in ess]], axis=0)))
    mo = float(np.nanmedian(np.nanmedian(E[:, [gi[g] for g in olf]], axis=0)))
    say(f"     {len(ess)} ribosomal / RNA-pol genes: median gene effect {me:+.4f}")
    say(f"     {len(olf)} olfactory receptors:       median gene effect {mo:+.4f}")
    G.add("X1", bool(me < -0.5 and mo > -0.1), stat=float(me),
          if_true=lambda: f"X1 PASS -- essentials at {me:+.3f}, non-expressed controls at "
                          f"{mo:+.3f}",
          if_false=lambda: f"X1 FAIL -- essentials {me:+.3f}, controls {mo:+.3f}")
    res["sanity"] = {"essential_median": me, "olfactory_median": mo}

    # ---------------------------------------------------------------- X2
    say("X2 HOW MUCH OF THE VARIANCE IS EVEN BETWEEN LINES?")
    Ef = np.where(np.isfinite(E), E, np.nan)
    gmean = np.nanmean(Ef, axis=0)
    lmean = np.nanmean(Ef, axis=1)
    grand = float(np.nanmean(Ef))
    v_gene = float(np.nanvar(gmean))
    v_line = float(np.nanvar(lmean))
    resid = Ef - gmean[None, :]
    v_resid = float(np.nanvar(resid))
    tot = v_gene + v_resid
    say(f"     variance of per-gene means (between GENES):   {v_gene:.5f}")
    say(f"     variance of per-line means (between LINES):   {v_line:.5f}")
    say(f"     residual after removing the gene mean:        {v_resid:.5f}")
    say(f"     between-gene share of total: {v_gene/tot:.1%}")
    say("     ratio REPORTED, not gated: if between-gene dominates then X4 is near-vacuous by")
    say("     construction and X5 carries the result")
    G.add("X2", bool(v_gene > 0 and v_resid > 0), stat=float(v_gene / tot),
          if_true=lambda: f"X2 PASS -- both components positive; between-gene is "
                          f"{v_gene/tot:.0%} of the total",
          if_false=lambda: f"X2 FAIL -- a variance component is not positive")
    res["variance"] = {"between_gene": v_gene, "between_line": v_line,
                       "residual": v_resid, "gene_share": v_gene / tot}

    # ---------------------------------------------------------------- expression
    say("     loading matched transcriptomes ...")
    xl, xg, X = [], None, []
    with open(DM / "OmicsExpression.csv") as f:
        hdr = f.readline().rstrip("\n").split(",")
        xg = np.array([re.sub(r"\s*\(\d+\)$", "", g) for g in hdr[1:]])
        for ln in f:
            p = ln.rstrip("\n").split(",")
            xl.append(p[0])
            X.append([0.0 if v == "" else float(v) for v in p[1:]])
    X = np.array(X, np.float32); xl = np.array(xl)
    keep = np.array([l in set(xl) for l in elines])
    xpos = {l: i for i, l in enumerate(xl)}
    E = E[keep]; lineage = lineage[keep]; elines = elines[keep]
    XE = X[[xpos[l] for l in elines]]
    say(f"     {E.shape[0]:,} cell lines have BOTH gene effect and expression")
    sel = np.argsort(-np.nanvar(np.where(np.isfinite(E), E, np.nan), axis=0))[:NSEL]
    say(f"     selectively essential set: the {NSEL:,} genes with the highest variance across lines")

    # reduce expression to components once, on all lines (unsupervised, no target involved)
    Xc = XE - XE.mean(0, keepdims=True)
    U, S_, Vt = np.linalg.svd(Xc, full_matrices=False)
    XP = U[:, :50] * S_[:50]
    say(f"     expression reduced to 50 components ({np.sum(S_[:50]**2)/np.sum(S_**2):.1%} of "
        f"variance), unsupervised")

    NL = E.shape[0]
    perm = rng.permutation(NL)
    folds = [perm[i::NFOLD] for i in range(NFOLD)]

    def evaluate(gene_idx, shuffle_lines=False):
        out = {k: [] for k in ("MEAN", "LINEAGE", "EXPR", "NEIGHBOUR")}
        for te in folds:
            tr = np.setdiff1d(np.arange(NL), te)
            Etr = E[np.ix_(tr, gene_idx)]
            mu = np.nanmean(Etr, axis=0)
            Ptr = XP[tr]
            for li in te:
                truth = E[li, gene_idx]
                src = int(rng.choice(tr)) if shuffle_lines else li
                out["MEAN"].append(pear(mu, truth))
                same = tr[lineage[tr] == lineage[src]]
                lm = np.nanmean(E[np.ix_(same, gene_idx)], axis=0) if len(same) >= 5 else mu
                out["LINEAGE"].append(pear(lm, truth))
                d = np.linalg.norm(Ptr - XP[src][None, :], axis=1)
                nn = tr[np.argsort(d)[:KNN]]
                out["NEIGHBOUR"].append(pear(np.nanmean(E[np.ix_(nn, gene_idx)], axis=0), truth))
                w = 1.0 / (d + 1e-6) ** 2
                w = w / w.sum()
                out["EXPR"].append(pear(np.nansum(Etr * w[:, None], axis=0), truth))
        return {k: np.array(v) for k, v in out.items()}

    # ---------------------------------------------------------------- X3
    say("X3 WHAT DOES KNOWING NOTHING ABOUT THE CELL LINE ACHIEVE?")
    allg = np.arange(E.shape[1])
    A = evaluate(allg)
    say(f"     MEAN, all {E.shape[1]:,} genes, {NFOLD}-fold over cell lines: "
        f"{np.nanmean(A['MEAN']):.4f} +/- {np.nanstd(A['MEAN']):.4f}")
    say("     a high number here is the POINT, not a success -- it is how much is available with")
    say("     no cell-line information at all")
    G.add("X3", bool(np.nanmean(A["MEAN"]) > X3_BAR), stat=float(np.nanmean(A["MEAN"])),
          requires=("X1",),
          if_true=lambda: f"X3 PASS -- {np.nanmean(A['MEAN']):.4f} from the gene mean alone",
          if_false=lambda: f"X3 FAIL -- {np.nanmean(A['MEAN']):.4f}")
    res["all_genes"] = {k: float(np.nanmean(v)) for k, v in A.items()}

    # ---------------------------------------------------------------- X4
    say("X4 DO THE CELL-LINE-SPECIFIC ARMS BEAT IT ON ALL GENES?")
    for k in ("LINEAGE", "NEIGHBOUR", "EXPR"):
        say(f"       {k:<10} {np.nanmean(A[k]):.4f}   delta vs MEAN "
            f"{np.nanmean(A[k])-np.nanmean(A['MEAN']):+.4f}")
    best4 = max(("LINEAGE", "NEIGHBOUR", "EXPR"), key=lambda k: np.nanmean(A[k]))
    d4 = float(np.nanmean(A[best4]) - np.nanmean(A["MEAN"]))
    G.add("X4", bool(d4 >= X4_BAR), stat=float(d4), requires=("X1",),
          if_true=lambda: f"X4 PASS -- {best4} gains {d4:+.4f} over knowing nothing",
          if_false=lambda: f"X4 FAIL -- best arm {best4} gains only {d4:+.4f}")
    res["all_genes"]["best"] = best4; res["all_genes"]["delta"] = d4

    # ---------------------------------------------------------------- X5
    say("X5 DO THEY BEAT IT ON SELECTIVELY ESSENTIAL GENES?")
    B = evaluate(sel)
    say(f"     restricted to the {NSEL:,} most variable genes across cell lines:")
    for k in ("MEAN", "LINEAGE", "NEIGHBOUR", "EXPR"):
        say(f"       {k:<10} {np.nanmean(B[k]):.4f}")
    best5 = max(("LINEAGE", "NEIGHBOUR", "EXPR"), key=lambda k: np.nanmean(B[k]))
    d5 = float(np.nanmean(B[best5]) - np.nanmean(B["MEAN"]))
    say(f"     best {best5} gains {d5:+.4f} over MEAN")
    G.add("X5", bool(d5 >= X5_BAR), stat=float(d5), requires=("X1",),
          if_true=lambda: f"X5 PASS -- on the genes where the question is real, {best5} beats "
                          f"knowing nothing by {d5:+.4f}. A held-out cell line IS predictable",
          if_false=lambda: f"X5 FAIL -- {d5:+.4f} against a {X5_BAR:.2f} bar; even where genes "
                           f"vary across lines, knowing which line it is adds little")
    res["selective"] = {k: float(np.nanmean(v)) for k, v in B.items()}
    res["selective"]["best"] = best5; res["selective"]["delta"] = d5

    # ---------------------------------------------------------------- X6
    say("X6 DOES K562 PERTURB-SEQ TRANSFER TO FITNESS IN OTHER LINES?")
    r_k562, r_other = float("nan"), float("nan")
    try:
        import h5py
        bh = h5py.File(K562_BULK, "r")
        bidx = np.array([x.decode() if isinstance(x, bytes) else str(x)
                         for x in bh["obs"][bh["obs"].attrs.get("_index", "_index")][:]])
        bsym = np.array([s.split("_")[1] if s.count("_") >= 3 else s for s in bidx])
        mag = np.array([float(np.nanmean(np.abs(np.asarray(bh["X"][i, :]))))
                        for i in range(0, len(bsym), 1)])
        pm = {}
        for s, v in zip(bsym, mag):
            if np.isfinite(v): pm.setdefault(s, []).append(v)
        pmag = {s: float(np.mean(v)) for s, v in pm.items()}
        shared = [g for g in egenes if g in pmag]
        gj = np.array([gi[g] for g in shared])
        pv = np.array([pmag[g] for g in shared])
        k_i = int(np.where(elines == "ACH-000551")[0][0])
        r_k562 = pear(pv, -E[k_i, gj])
        others = [pear(pv, -E[i, gj]) for i in range(NL) if i != k_i]
        r_other = float(np.nanmedian(others))
        say(f"     {len(shared):,} genes shared between Perturb-seq and DepMap")
        say(f"     Perturb-seq response magnitude vs fitness in K562 itself: {r_k562:+.4f}")
        say(f"     the same, median over the other {len(others):,} cell lines: {r_other:+.4f}")
        bh.close()
    except Exception as e:
        say(f"     X6 could not run: {type(e).__name__}: {e}")
    G.add("X6", bool(np.isfinite(r_k562) and r_k562 - r_other >= X6_BAR), stat=float(r_k562),
          requires=("X1",), void_if=(not np.isfinite(r_k562)),
          void_reason="the Perturb-seq bulk file did not yield comparable magnitudes",
          if_true=lambda: f"X6 PASS -- K562 {r_k562:+.4f} against {r_other:+.4f} elsewhere; the "
                          f"Perturb-seq signal is cell-type-specific",
          if_false=lambda: f"X6 FAIL -- K562 {r_k562:+.4f} against {r_other:+.4f} elsewhere. The "
                           f"Perturb-seq signal predicts fitness GENERICALLY, which is why loop "
                           f"210's gains block transferred to A549 at all")
    res["perturbseq_transfer"] = {"k562": r_k562, "other_median": r_other}

    # ---------------------------------------------------------------- X7
    say("X7 CONTROL: SHUFFLED CELL-LINE IDENTITY")
    Bs = evaluate(sel, shuffle_lines=True)
    d5s = float(np.nanmean(Bs[best5]) - np.nanmean(Bs["MEAN"]))
    say(f"     the held-out line's expression replaced by another line's:")
    say(f"       real advantage over MEAN     {d5:+.4f}")
    say(f"       shuffled advantage over MEAN {d5s:+.4f}")
    G.add("X7", bool(d5s <= 0.5 * d5), stat=float(d5s), requires=("X5",),
          if_true=lambda: f"X7 PASS -- shuffling identity collapses the advantage from {d5:+.4f} "
                          f"to {d5s:+.4f}",
          if_false=lambda: f"X7 FAIL -- the advantage survives at {d5s:+.4f} against a real "
                           f"{d5:+.4f}; it comes from having more parameters, not from knowing "
                           f"the line")
    res["shuffle"] = {"real": d5, "shuffled": d5s}

    # ---------------------------------------------------------------- X8
    say("X8 WHAT THIS CANNOT SHOW")
    say("     CRISPR fitness is not gene expression. This is condition-level generalisation on a")
    say("     DIFFERENT readout from the one loops 206-237 worked on, so a result here does not")
    say("     transfer back to the A549 dexamethasone target directly.")
    say("     DepMap lines are cancer lines in culture and lineage is a coarse label. A model that")
    say("     predicts a held-out line may be exploiting lineage-level structure rather than that")
    say("     line's biology; X7 bounds that without eliminating it.")
    say("     Gene effect is a growth phenotype integrated over weeks. It sits far downstream of")
    say("     the transcriptional response Perturb-seq measures, so X6 relates two quantities")
    say("     connected by a long causal chain rather than by definition.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary()
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
