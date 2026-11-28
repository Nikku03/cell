"""Loop 239. The catalogue-and-factory hypothesis, tested as stated.

THE PROPOSAL, IN THE WORDS IT WAS PUT IN:
"Lets take human gene as your catalogue. It contains all genes. Now each cell line or factory has
all the machines but not all are functional. So if we know the info about every machine, every
mechanic or worker. And then we go to other factory and know which machines are working, can't we
then predict the effect of a worker on that factory?"

WHY LOOP 238 DID NOT ALREADY ANSWER THIS. Loop 238 held out a cell line and predicted its whole
dependency profile by PROFILE MATCHING -- averaging the profiles of similar lines. It never asked
the question in the form above. The form above is a per-(worker, factory) model: properties of the
gene, crossed with whether that gene and its backups and its co-workers are actually running in
this particular factory. That is a different model and it has never been fitted here.

THE TARGET, AND WHY IT IS THE RESIDUAL. Loop 238 X2 measured that 85% of the variance in DepMap
gene effect is BETWEEN GENES. A predictor that ignores the cell line entirely scored 0.9250. Any
score computed on the raw gene effect is therefore mostly a restatement of that 85%, and reporting
it would repeat loop 201's P2 -- a number that looks like understanding and measures almost
nothing. So the target here is the RESIDUAL:

    R[line, gene] = gene effect  -  that gene's mean effect over the TRAINING lines

which is exactly and only the part the line-blind predictor cannot express. A residual correlation
of zero is what the loop-238 MEAN baseline scores by construction. Every number in Y2-Y7 is on
this residual. The raw-scale number is reported once, for continuity, and is not the result.

THE ARMS. All are fitted on training lines and scored on held-out lines.

    A0 GENEONLY   intercept + gene-level facts only: training mean effect, sd across lines,
                  BioGRID degree, paralogue count. Knows NOTHING about the held-out factory.
                  A0 is the floor that the other arms must beat -- not zero. Gene-level facts can
                  earn a nonzero residual correlation purely by shrinking a noisy gene mean, and
                  that is not the hypothesis.
    A1 OWN        A0 + is this machine running here (the gene's own expression z) and its
                  interactions with the gene's mean and sd.
    A2 BUFFER     A1 + is there a backup machine running here (mean and max paralogue expression).
    A3 PARTNER    A2 + are its co-workers running here (mean expression of BioGRID partners).
    A5 NEIGHBOUR  the loop-238 profile-matching arm rewritten as a residual predictor: mean of the
                  k transcriptionally nearest training lines, minus the training gene mean. Not
                  fitted. This is the honest competitor.
    A6 COMBINED   A3 plus the A5 prediction as one extra feature.

PREDECLARED, BEFORE ANY NUMBER IS COMPUTED.

  Y1 IS THE PREMISE OF THE WHOLE PROPOSAL EVEN TRUE?
     The proposal rests on one claim: a machine that is not running cannot be broken. If a gene is
     not expressed in a line, knocking it out should do nothing there. This is first-principles,
     not chosen from this data.
     Gate: PASS iff, over the selectively essential set, the mean gene effect in the lines where
     that gene sits in its bottom expression decile is closer to zero than in its top decile, by
     at least 0.05. A FAIL refutes the proposal at the root and everything after it is decoration.

  Y2 DOES KNOWING WHICH MACHINES ARE RUNNING PREDICT THE RESIDUAL AT ALL?
     Best fitted arm minus A0, on the selectively essential set, paired over held-out lines.
     Gate: PASS iff the best arm beats A0 by at least 0.05 AND by at least 3 paired standard
     errors. Everything below requires this.

  Y3 DOES THE MECHANISTIC MODEL BEAT PROFILE MATCHING?      -- requires Y2
     A3 against A5, paired over held-out lines.
     Gate: PASS iff A3 exceeds A5 by at least 0.02. The sign is NOT assumed: if profile matching
     wins, that is the finding and it is reported as the finding.

  Y4 THE BACKUP-MACHINE TEST -- the sharpest form of the proposal.      -- requires Y2
     Two things must both hold, because a gain with the wrong sign is not buffering.
     Gate: PASS iff (a) A2 beats A1 by at least 0.01 paired, AND (b) the fitted coefficient on max
     paralogue expression is POSITIVE, meaning a running backup makes the effect LESS negative.

  Y5 A WORKER WE HAVE NEVER PLACED, IN A FACTORY WE HAVE NEVER SEEN.      -- requires Y2
     The coefficients are fitted on a disjoint half of the genes, then applied to the held-out half
     in the held-out lines. The catalogue entry for a gene (its mean, sd, degree, paralogue count)
     is still allowed -- the proposal grants a complete catalogue -- but the RULE was learned
     without ever seeing this gene.
     Gate: PASS iff the best arm keeps at least half of its Y2 advantage over A0.

  Y6 CONTROL: THE WRONG FACTORY.      -- requires Y2, VOID if the Y2 advantage is under 0.01
     The held-out line's transcriptome swapped for a random training line's, everything else
     identical.
     Gate: PASS iff the advantage over A0 collapses to under 25% of its true value.

  Y7 CONTROL: THE WRONG CATALOGUE.      -- requires Y2, VOID if the Y2 advantage is under 0.01
     The gene-level facts other than the mean effect (sd, degree, paralogue count) permuted across
     genes, so the catalogue no longer describes the machine. Line state left intact. The mean
     effect is not permuted because it defines the target.
     Gate: PASS iff the advantage of A3 over A1 collapses by at least half. If it does not, the
     "know every machine" half of the proposal is inert and only "which machines are running"
     is doing work.

  Y8 WHAT THIS CANNOT SHOW -- written before the run.
     Expression is a staff directory, not an org chart. It says which machines are switched on, not
     what they are wired to. Any failure here is a failure of THIS encoding of functional state,
     not proof that functional state is uninformative.
     DepMap gene effect is a growth phenotype integrated over weeks. It is the factory's output
     after a long time, not the immediate consequence of removing the worker.
     Paralogue identity comes from Ensembl sequence homology. Sequence-similar genes are not
     guaranteed to be functionally redundant, so Y4 tests homology-based buffering specifically.
     Cell lines are cancer lines in culture and share lineage structure; Y6 bounds but does not
     eliminate the possibility that a gain reflects lineage rather than this line.
"""
import os, sys, json, time, gzip, warnings
from pathlib import Path
import numpy as np
from scipy import sparse

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_factory_worker.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
DM = SCR / "depmap"
PARA = SCR / "para" / "paralogs.tsv"
BG = SCR / "biogrid_hs_edges.tsv.gz"
SEED, NFOLD, NSEL, KNN, RIDGE = 239239, 10, 2000, 25, 1.0
Y1_BAR, Y2_BAR, Y2_SE, Y3_BAR, Y4_BAR, Y5_KEEP, Y6_MAX, Y7_DROP = 0.05, 0.05, 3.0, 0.02, 0.01, 0.50, 0.25, 0.50

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


def paired(a, b):
    """Paired mean difference, its standard error and z. Split-to-split variance is common to both
    arms and cancels; this is the loop-229 instrument."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if d.size < 3: return float("nan"), float("nan"), float("nan")
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    mu = float(np.mean(d))
    return mu, se, (mu / se if se > 0 else float("nan"))


# feature layout: index -> (name). Arms are column subsets of one design matrix.
FEATS = ["const", "g_mean", "g_sd", "g_deg", "g_npara",
         "l_own", "own*mean", "own*sd",
         "l_pmean", "l_pmax", "pmean*mean", "pmax*mean", "pmax*sd", "pmax*npara",
         "l_partner", "partner*sd", "own*deg", "a5_nbr"]
FI = {n: i for i, n in enumerate(FEATS)}
ARMS = {
    "A0_GENEONLY": ["const", "g_mean", "g_sd", "g_deg", "g_npara"],
    "A1_OWN":      ["const", "g_mean", "g_sd", "g_deg", "g_npara", "l_own", "own*mean", "own*sd"],
    "A2_BUFFER":   ["const", "g_mean", "g_sd", "g_deg", "g_npara", "l_own", "own*mean", "own*sd",
                    "l_pmean", "l_pmax", "pmean*mean", "pmax*mean", "pmax*sd", "pmax*npara"],
    "A3_PARTNER":  ["const", "g_mean", "g_sd", "g_deg", "g_npara", "l_own", "own*mean", "own*sd",
                    "l_pmean", "l_pmax", "pmean*mean", "pmax*mean", "pmax*sd", "pmax*npara",
                    "l_partner", "partner*sd", "own*deg"],
    "A6_COMBINED": FEATS,
}
FITTED = list(ARMS)


def build_design(Zl, Pm, Px, Qm, gmean, gsd, gdeg, gnpa, nbr):
    """One held-out (or training) line's design matrix, genes x len(FEATS).
    Zl/Pm/Px/Qm are that line's own / paralogue-mean / paralogue-max / partner-mean expression z."""
    n = gmean.size
    X = np.empty((n, len(FEATS)), np.float32)
    X[:, FI["const"]] = 1.0
    X[:, FI["g_mean"]] = gmean
    X[:, FI["g_sd"]] = gsd
    X[:, FI["g_deg"]] = gdeg
    X[:, FI["g_npara"]] = gnpa
    X[:, FI["l_own"]] = Zl
    X[:, FI["own*mean"]] = Zl * gmean
    X[:, FI["own*sd"]] = Zl * gsd
    X[:, FI["l_pmean"]] = Pm
    X[:, FI["l_pmax"]] = Px
    X[:, FI["pmean*mean"]] = Pm * gmean
    X[:, FI["pmax*mean"]] = Px * gmean
    X[:, FI["pmax*sd"]] = Px * gsd
    X[:, FI["pmax*npara"]] = Px * gnpa
    X[:, FI["l_partner"]] = Qm
    X[:, FI["partner*sd"]] = Qm * gsd
    X[:, FI["own*deg"]] = Zl * gdeg
    X[:, FI["a5_nbr"]] = nbr
    return X


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "per-(gene, cell line) functional-state model, held out by cell line"}
    say("=" * 104)
    say("LOOP 239 -- THE CATALOGUE AND THE FACTORY")
    say("=" * 104)
    say("     The target is the RESIDUAL after the training-line gene mean. Loop 238's line-blind")
    say("     MEAN predictor scores exactly 0 on it by construction, and 85% of the raw variance")
    say("     is between genes, so the residual is the only place the hypothesis can be tested.")

    # ---------------------------------------------------------------- data
    z = np.load(DM / "gene_effect.npz", allow_pickle=True)
    E = np.asarray(z["E"], np.float32)
    egenes = np.array([str(x) for x in z["genes"]])
    elines = np.array([str(x) for x in z["lines"]])
    ez = np.load(SCR / "depmap_expr.npz", allow_pickle=True)
    say(f"     gene effect: {E.shape[0]:,} lines x {E.shape[1]:,} genes")

    import re
    xl, X = [], []
    with open(DM / "OmicsExpression.csv") as f:
        hdr = f.readline().rstrip("\n").split(",")
        xg = np.array([re.sub(r"\s*\(\d+\)$", "", g) for g in hdr[1:]])
        for ln in f:
            p = ln.rstrip("\n").split(",")
            xl.append(p[0]); X.append([0.0 if v == "" else float(v) for v in p[1:]])
    X = np.asarray(X, np.float32); xl = np.array(xl)
    xpos = {l: i for i, l in enumerate(xl)}
    keep = np.array([l in xpos for l in elines])
    E, elines = E[keep], elines[keep]
    XE = X[[xpos[l] for l in elines]]
    del X
    say(f"     {E.shape[0]:,} lines have both gene effect and a transcriptome")

    # align expression onto the gene-effect gene axis; genes with no transcriptome measured are
    # given z = 0 everywhere, which is the "no information" value after z-scoring
    gpos = {g: i for i, g in enumerate(xg)}
    cols = np.array([gpos.get(g, -1) for g in egenes])
    have = cols >= 0
    Zexp = np.zeros((E.shape[0], egenes.size), np.float32)
    Zexp[:, have] = XE[:, cols[have]]
    say(f"     {have.sum():,} of {egenes.size:,} gene-effect genes have a matched expression column")
    # z-score each gene's expression across lines. Unsupervised: the dependency target is never
    # touched, so this is done once on all lines rather than per fold.
    mu_, sd_ = Zexp.mean(0), Zexp.std(0)
    Zexp = (Zexp - mu_) / np.where(sd_ > 1e-6, sd_, 1.0)
    Zexp[:, ~have] = 0.0
    Zexp = np.asarray(Zexp, np.float32)

    NG, NL = egenes.size, E.shape[0]
    gi = {g: i for i, g in enumerate(egenes)}

    # ---------------------------------------------------------------- catalogue: paralogues
    pr, pc = [], []
    npair = 0
    with open(PARA) as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 2 or not p[1]: continue
            a, b = gi.get(p[0], -1), gi.get(p[1], -1)
            if a < 0 or b < 0 or a == b: continue
            pr.append(a); pc.append(b); npair += 1
    pr, pc = np.asarray(pr), np.asarray(pc)
    npara = np.bincount(pr, minlength=NG).astype(np.float32)
    say(f"     paralogues: {npair:,} pairs, {int((npara > 0).sum()):,} genes with at least one")

    # ---------------------------------------------------------------- catalogue: partners
    er, ec = [], []
    with open(BG, "rb") as fh:
        for ln in gzip.GzipFile(fileobj=fh):
            p = ln.decode().rstrip("\n").split("\t")
            if len(p) < 2: continue
            a, b = gi.get(p[0], -1), gi.get(p[1], -1)
            if a < 0 or b < 0 or a == b: continue
            er.append(a); ec.append(b); er.append(b); ec.append(a)
    er, ec = np.asarray(er), np.asarray(ec)
    Adj = sparse.csr_matrix((np.ones(er.size, np.float32), (er, ec)), shape=(NG, NG))
    Adj.sum_duplicates(); Adj.data[:] = 1.0
    deg = np.asarray(Adj.sum(1)).ravel().astype(np.float32)
    say(f"     BioGRID: {int(Adj.nnz):,} directed entries, {int((deg > 0).sum()):,} genes with a partner")

    # line x gene neighbourhood expression, computed once
    Wp = sparse.csr_matrix((1.0 / np.maximum(npara[pr], 1), (pr, pc)), shape=(NG, NG), dtype=np.float32)
    Pmean = (Wp @ Zexp.T).T.astype(np.float32)
    Wq = Adj.multiply(1.0 / np.maximum(deg, 1)[:, None]).tocsr().astype(np.float32)
    Qmean = (Wq @ Zexp.T).T.astype(np.float32)
    # max over paralogues, in line chunks (a sparse matmul cannot express a max)
    order = np.argsort(pr, kind="stable")
    pr_s, pc_s = pr[order], pc[order]
    starts = np.searchsorted(pr_s, np.arange(NG))
    uniq = np.unique(pr_s)
    Pmax = np.zeros_like(Pmean)
    CH = 64
    for i0 in range(0, NL, CH):
        blk = Zexp[i0:i0 + CH][:, pc_s]                      # (chunk, npair)
        m = np.maximum.reduceat(blk, starts[uniq], axis=1)
        Pmax[i0:i0 + CH, uniq] = m
    say(f"     neighbourhood expression built: paralogue mean/max and partner mean")

    gdeg = np.log1p(deg).astype(np.float32)
    gnpa = np.log1p(npara).astype(np.float32)

    # selectively essential set -- the genes where the question is real
    Ef = np.where(np.isfinite(E), E, np.nan)
    var_g = np.nanvar(Ef, axis=0)
    SEL = np.argsort(-var_g)[:NSEL]
    ALL = np.arange(NG)
    say(f"     selective set: the {NSEL:,} genes with the highest gene-effect variance across lines")

    # ---------------------------------------------------------------- Y1
    say("Y1 IS THE PREMISE OF THE WHOLE PROPOSAL EVEN TRUE?")
    lo_v, hi_v = [], []
    for g in SEL:
        if not have[g]: continue
        col = Zexp[:, g]
        k = max(int(0.1 * NL), 20)
        o = np.argsort(col)
        a = Ef[o[:k], g]; b = Ef[o[-k:], g]
        if np.isfinite(a).sum() < 10 or np.isfinite(b).sum() < 10: continue
        lo_v.append(np.nanmean(a)); hi_v.append(np.nanmean(b))
    lo_m, hi_m = float(np.mean(lo_v)), float(np.mean(hi_v))
    gap = lo_m - hi_m
    say(f"     {len(lo_v):,} selective genes with expression measured")
    say(f"     mean gene effect where the gene is in its BOTTOM expression decile: {lo_m:+.4f}")
    say(f"     mean gene effect where the gene is in its TOP    expression decile: {hi_m:+.4f}")
    say(f"     gap (bottom minus top, positive means not-running is closer to harmless): {gap:+.4f}")
    G.add("Y1", bool(gap >= Y1_BAR), stat=float(gap),
          if_true=lambda: f"Y1 PASS -- a machine that is not running is {gap:+.4f} closer to "
                          f"harmless when broken; the premise holds",
          if_false=lambda: f"Y1 FAIL -- the gap is {gap:+.4f}, under the {Y1_BAR} bar; expression "
                           f"does not mark which knockouts matter")
    res["premise"] = {"bottom_decile": lo_m, "top_decile": hi_m, "gap": gap, "n": len(lo_v)}

    # ---------------------------------------------------------------- folds
    Xc = XE - XE.mean(0, keepdims=True)
    U, S_, Vt = np.linalg.svd(Xc, full_matrices=False)
    XP = (U[:, :50] * S_[:50]).astype(np.float32)
    say(f"     expression reduced to 50 unsupervised components "
        f"({np.sum(S_[:50] ** 2) / np.sum(S_ ** 2):.1%} of variance) for the A5 neighbour arm")
    perm = rng.permutation(NL)
    folds = [perm[i::NFOLD] for i in range(NFOLD)]
    ghalf = rng.permutation(NG)
    GA, GB = ghalf[:NG // 2], ghalf[NG // 2:]
    inA = np.zeros(NG, bool); inA[GA] = True

    def run(gene_idx, shuffle_line=False, shuffle_cat=False, gene_holdout=False):
        """Returns per-held-out-line residual correlations for every arm."""
        out = {k: [] for k in FITTED + ["A5_NEIGHBOUR"]}
        coefs = []
        gidx = np.asarray(gene_idx)
        for te in folds:
            tr = np.setdiff1d(np.arange(NL), te)
            gmean = np.nanmean(Ef[np.ix_(tr, gidx)], axis=0).astype(np.float32)
            gsd = np.nanstd(Ef[np.ix_(tr, gidx)], axis=0).astype(np.float32)
            gmean = np.nan_to_num(gmean); gsd = np.nan_to_num(gsd)
            cd, cn = gdeg[gidx].copy(), gnpa[gidx].copy()
            if shuffle_cat:
                p = rng.permutation(gidx.size)
                gsd, cd, cn = gsd[p], cd[p], cn[p]
            Ptr = XP[tr]

            def feats(li, src):
                d = np.linalg.norm(Ptr - XP[src][None, :], axis=1)
                d = np.where(tr == src, np.inf, d)   # a training line is its own neighbour: leak
                nn = tr[np.argsort(d)[:KNN]]
                nbr = (np.nanmean(Ef[np.ix_(nn, gidx)], axis=0) - gmean).astype(np.float32)
                nbr = np.nan_to_num(nbr)
                return build_design(Zexp[src][gidx], Pmean[src][gidx], Pmax[src][gidx],
                                   Qmean[src][gidx], gmean, gsd, cd, cn, nbr), nbr

            # ---- fit: accumulate normal equations over training lines
            fitcols = np.arange(gidx.size) if not gene_holdout else np.where(inA[gidx])[0]
            tecols = np.arange(gidx.size) if not gene_holdout else np.where(~inA[gidx])[0]
            if fitcols.size < 50 or tecols.size < 50:
                continue
            P = len(FEATS)
            XtX = np.zeros((P, P)); Xty = np.zeros(P)
            n_acc = 0
            for li in tr:
                Xi, _ = feats(li, li)
                Xi = Xi[fitcols]
                yi = (Ef[li, gidx[fitcols]] - gmean[fitcols])
                ok = np.isfinite(yi) & np.isfinite(Xi).all(1)
                Xi, yi = Xi[ok].astype(np.float64), yi[ok].astype(np.float64)
                XtX += Xi.T @ Xi; Xty += Xi.T @ yi; n_acc += ok.sum()
            beta = {}
            for arm, names in ARMS.items():
                c = [FI[n] for n in names]
                A = XtX[np.ix_(c, c)].copy()
                lam = RIDGE * np.trace(A) / max(len(c), 1)
                A[1:, 1:] += lam * np.eye(len(c) - 1)
                beta[arm] = (np.linalg.solve(A, Xty[c]), c)
            coefs.append({arm: dict(zip(ARMS[arm], beta[arm][0])) for arm in ARMS})

            # ---- score on held-out lines
            for li in te:
                src = int(rng.choice(tr)) if shuffle_line else li
                Xi, nbr = feats(li, src)
                truth = (Ef[li, gidx] - gmean)
                Xi, truth = Xi[tecols], truth[tecols]
                for arm in FITTED:
                    b, c = beta[arm]
                    out[arm].append(pear(Xi[:, c] @ b, truth))
                out["A5_NEIGHBOUR"].append(pear(nbr[tecols], truth))
        return {k: np.asarray(v) for k, v in out.items()}, coefs

    # ---------------------------------------------------------------- Y2
    say("Y2 DOES KNOWING WHICH MACHINES ARE RUNNING PREDICT THE RESIDUAL AT ALL?")
    S, coefs = run(SEL)
    for k in FITTED + ["A5_NEIGHBOUR"]:
        say(f"     {k:<14} residual r = {np.nanmean(S[k]):+.4f}  (sd across held-out lines "
            f"{np.nanstd(S[k]):.4f})")
    best = max(FITTED, key=lambda k: np.nanmean(S[k]) if k != "A0_GENEONLY" else -9)
    d2, se2, z2 = paired(S[best], S["A0_GENEONLY"])
    say(f"     best fitted arm is {best}")
    say(f"     {best} minus A0_GENEONLY, paired over {np.isfinite(S[best]).sum()} held-out lines: "
        f"{d2:+.4f} +/- {se2:.4f}  ({z2:+.1f} se)")
    G.add("Y2", bool(d2 >= Y2_BAR and z2 >= Y2_SE), stat=float(d2),
          if_true=lambda: f"Y2 PASS -- functional state adds {d2:+.4f} over the catalogue alone "
                          f"({z2:+.1f} se)",
          if_false=lambda: f"Y2 FAIL -- functional state adds {d2:+.4f} ({z2:+.1f} se) against a "
                           f"{Y2_BAR} bar at {Y2_SE} se")
    res["Y2"] = {"per_arm": {k: float(np.nanmean(S[k])) for k in S}, "best": best,
                 "delta": d2, "se": se2, "z": z2}

    # raw-scale number, for continuity with loop 238 only
    gm_all = []
    for te in folds:
        tr = np.setdiff1d(np.arange(NL), te)
        mu = np.nanmean(Ef[np.ix_(tr, SEL)], axis=0)
        for li in te: gm_all.append(pear(mu, Ef[li, SEL]))
    say(f"     for continuity only: the loop-238 line-blind MEAN scores "
        f"{np.nanmean(gm_all):+.4f} on the RAW selective-set scale and exactly 0 on the residual")
    res["raw_mean_baseline"] = float(np.nanmean(gm_all))

    # ---------------------------------------------------------------- Y3
    say("Y3 DOES THE MECHANISTIC MODEL BEAT PROFILE MATCHING?")
    d3, se3, z3 = paired(S["A3_PARTNER"], S["A5_NEIGHBOUR"])
    say(f"     A3_PARTNER {np.nanmean(S['A3_PARTNER']):+.4f} vs A5_NEIGHBOUR "
        f"{np.nanmean(S['A5_NEIGHBOUR']):+.4f}")
    say(f"     paired difference {d3:+.4f} +/- {se3:.4f}  ({z3:+.1f} se)")
    G.add("Y3", bool(d3 >= Y3_BAR), stat=float(d3), requires="Y2",
          if_true=lambda: f"Y3 PASS -- the per-machine model beats profile matching by {d3:+.4f}",
          if_false=lambda: f"Y3 FAIL -- the per-machine model is {d3:+.4f} against profile "
                           f"matching; knowing which lines LOOK like this one is worth "
                           f"{'more' if d3 < 0 else 'about the same as'} knowing which machines run")
    res["Y3"] = {"delta": d3, "se": se3, "z": z3}

    # ---------------------------------------------------------------- Y4
    say("Y4 THE BACKUP-MACHINE TEST")
    d4, se4, z4 = paired(S["A2_BUFFER"], S["A1_OWN"])
    cb = float(np.mean([c["A2_BUFFER"]["l_pmax"] for c in coefs]))
    say(f"     A2_BUFFER minus A1_OWN, paired: {d4:+.4f} +/- {se4:.4f}  ({z4:+.1f} se)")
    say(f"     fitted coefficient on max paralogue expression: {cb:+.5f}")
    say(f"     buffering predicts a POSITIVE coefficient -- a running backup makes the knockout")
    say(f"     less harmful, so the residual moves upward")
    G.add("Y4", bool(d4 >= Y4_BAR and cb > 0), stat=float(d4), requires="Y2",
          if_true=lambda: f"Y4 PASS -- backups add {d4:+.4f} and the coefficient is {cb:+.5f}, "
                          f"the buffering direction",
          if_false=lambda: f"Y4 FAIL -- backups add {d4:+.4f} (bar {Y4_BAR}) with coefficient "
                           f"{cb:+.5f} ({'wrong sign' if cb <= 0 else 'right sign'})")
    res["Y4"] = {"delta": d4, "se": se4, "z": z4, "pmax_coef": cb}

    # ---------------------------------------------------------------- Y5
    say("Y5 A WORKER WE HAVE NEVER PLACED, IN A FACTORY WE HAVE NEVER SEEN")
    H, _ = run(SEL, gene_holdout=True)
    d5, se5, z5 = paired(H[best], H["A0_GENEONLY"])
    keepfrac = d5 / d2 if d2 > 1e-9 else float("nan")
    say(f"     coefficients fitted on one half of the genes, scored on the other half in the")
    say(f"     held-out lines: {best} {np.nanmean(H[best]):+.4f}, A0 {np.nanmean(H['A0_GENEONLY']):+.4f}")
    say(f"     advantage {d5:+.4f} +/- {se5:.4f} against {d2:+.4f} single-holdout "
        f"-- {keepfrac:.0%} retained")
    G.add("Y5", bool(np.isfinite(keepfrac) and keepfrac >= Y5_KEEP), stat=float(keepfrac),
          requires="Y2",
          if_true=lambda: f"Y5 PASS -- {keepfrac:.0%} of the advantage survives on genes the rule "
                          f"never saw",
          if_false=lambda: f"Y5 FAIL -- only {keepfrac:.0%} survives; the rule was gene-specific")
    res["Y5"] = {"delta": d5, "se": se5, "retained": keepfrac}

    # ---------------------------------------------------------------- Y6
    say("Y6 CONTROL: THE WRONG FACTORY")
    if d2 < 0.01:
        G.add("Y6", False, stat=float(d2), requires="Y2",
              void_if=True, void_reason=f"the real advantage is {d2:+.4f}; there is nothing for a "
                                        f"control to collapse")
        d6 = float("nan")
    else:
        Sh, _ = run(SEL, shuffle_line=True)
        d6, se6, z6 = paired(Sh[best], Sh["A0_GENEONLY"])
        frac6 = d6 / d2
        say(f"     with a random training line's transcriptome: advantage {d6:+.4f} against a real "
            f"{d2:+.4f}  ({frac6:.0%})")
        G.add("Y6", bool(frac6 <= Y6_MAX), stat=float(frac6), requires="Y2",
              if_true=lambda: f"Y6 PASS -- the advantage collapses to {frac6:.0%} on the wrong "
                              f"factory",
              if_false=lambda: f"Y6 FAIL -- {frac6:.0%} survives with the wrong factory's "
                               f"transcriptome; the gain is not about this line")
        res["Y6"] = {"shuffled": d6, "real": d2, "fraction": frac6}

    # ---------------------------------------------------------------- Y7
    say("Y7 CONTROL: THE WRONG CATALOGUE")
    dcat = np.nanmean(S["A3_PARTNER"]) - np.nanmean(S["A1_OWN"])
    if not (dcat >= 0.01):
        G.add("Y7", False, stat=float(dcat), requires="Y2",
              void_if=True, void_reason=f"A3 beats A1 by only {dcat:+.4f}; the catalogue half of "
                                        f"the proposal contributes nothing to collapse")
    else:
        Sc, _ = run(SEL, shuffle_cat=True)
        dcs = np.nanmean(Sc["A3_PARTNER"]) - np.nanmean(Sc["A1_OWN"])
        frac7 = dcs / dcat
        say(f"     catalogue facts permuted across genes: A3 over A1 falls from {dcat:+.4f} to "
            f"{dcs:+.4f}  ({frac7:.0%})")
        G.add("Y7", bool(frac7 <= 1 - Y7_DROP), stat=float(frac7), requires="Y2",
              if_true=lambda: f"Y7 PASS -- permuting the catalogue removes "
                              f"{1 - frac7:.0%} of what it contributed",
              if_false=lambda: f"Y7 FAIL -- {frac7:.0%} survives a permuted catalogue; the gain is "
                               f"not about knowing the machine")
        res["Y7"] = {"real": float(dcat), "shuffled": float(dcs), "fraction": float(frac7)}

    # ---------------------------------------------------------------- coefficients
    say("     mean fitted coefficients, A3_PARTNER, over folds:")
    for n in ARMS["A3_PARTNER"]:
        say(f"       {n:<12} {np.mean([c['A3_PARTNER'][n] for c in coefs]):+.5f}")
    res["coefficients"] = {n: float(np.mean([c["A3_PARTNER"][n] for c in coefs]))
                           for n in ARMS["A3_PARTNER"]}

    # ---------------------------------------------------------------- Y8
    say("Y8 WHAT THIS CANNOT SHOW")
    say("     Expression is a staff directory, not an org chart: it records which machines are")
    say("     switched on, not what they are wired to. A failure here is a failure of THIS")
    say("     encoding of functional state, not proof that functional state is uninformative.")
    say("     DepMap gene effect is a growth phenotype integrated over weeks -- the factory's")
    say("     output long after the worker left, not the immediate consequence of removing them.")
    say("     Paralogues come from Ensembl sequence homology; sequence-similar genes need not be")
    say("     functionally redundant, so Y4 tests homology-based buffering specifically.")
    say("     Lines share lineage structure. Y6 bounds, without eliminating, the possibility that")
    say("     an advantage reflects the tissue rather than this particular line.")

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
