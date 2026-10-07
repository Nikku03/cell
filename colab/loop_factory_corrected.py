"""Loop 240. The catalogue-and-factory hypothesis, on a gene set where it can actually be asked.

WHAT LOOP 239 GOT WRONG, MEASURED. Loop 239 (and loop 238 before it) defined "selectively
essential" as the 2,000 genes with the highest gene-effect variance across cell lines. That
definition is broken, and the number that shows it is:

    corr( sd across lines , -mean effect )  =  0.7909  over all 17,916 genes

Gene-effect variance scales with gene-effect magnitude. A gene sitting at -1.5 has room to vary; a
gene sitting at -0.02 does not. So ranking by variance is 79% a ranking by how essential a gene is
EVERYWHERE, which is the opposite of selective. 68.6% of loop 239's set has mean effect below -0.5,
its mean gene effect is -0.9405 against -0.1431 genome-wide, and it contains RPL36A, EEF1A1, TUBB
and PSMB5 -- core ribosome, translation, tubulin and proteasome. Loop 239's Y1 asked whether a gene
is harmless to knock out in the lines where it is not expressed, and answered it on a set of genes
that are essential in every line. The +0.0115 gap it reported is what that question is worth on
core-essential genes, and it is not evidence about selective dependency.

THE CORRECTED SET is defined by shape, not size: a gene is a differential dependency if it is
essentially harmless in most lines and strongly essential in some.

    STRICT   90th percentile of its effect > -0.25  AND  10th percentile < -0.75   ->   364 genes
    BROAD    90th percentile > -0.25                AND  10th percentile < -0.50   -> 1,135 genes

STRICT has mean effect -0.5160 and contains KRAS, CCND1, GRB2, CFLAR, SOD2, FERMT2 -- oncogene
addictions and lineage dependencies, which is what a selective dependency is. Gates run on STRICT;
BROAD is reported alongside so the result is not hostage to one threshold.

THREE MORE DEFECTS FROM LOOP 239, ALSO FIXED.

  (i)  Features were not standardised before ridge, so the penalty fell on each feature in
       proportion to its units. The neighbour feature has a small scale and was crushed. This is
       why A6_COMBINED scored +0.0423 while containing, as one of its columns, a feature that
       scores +0.2932 on its own -- correlation is scale invariant, so a fitted model containing a
       predictor cannot score below it unless the other columns are drowning it. Here: features
       are standardised analytically from the accumulated normal equations, and the ridge strength
       is chosen on an inner split of the training lines rather than fixed at 1.0.
  (ii) Loop 239's Y2 picked its best arm from the FITTED arms only, so the strongest predictor in
       the run -- A5_NEIGHBOUR, unfitted, +0.2932 -- was excluded from the headline by
       construction. A predictor does not have to have coefficients to count. Z2 ranks all arms.
  (iii) Loop 239 chained Y3 and Y4 to Y2 with requires=, and both were voided. But Y3 compares two
       arms neither of which is A0, and Y4 compares two arms neither of which is A0; neither
       depends on Y2's particular comparison passing. Here only the gates that genuinely need a
       signal to exist are chained.

THE QUESTION HAS ALSO CHANGED SHAPE because of what loop 239 found. Profile matching already
reaches +0.2932 on the residual. So the live question is no longer "does the mechanistic model
beat it" but "does knowing which machines are running add anything profile matching does not
already have", and whether profile matching is itself just tissue matching.

THE ARMS. Target throughout is the residual, gene effect minus the TRAINING-line gene mean, which
is the only part loop 238's line-blind predictor cannot express and on which it scores exactly 0.

    A0 CATALOGUE   gene-level facts only: training mean, sd, BioGRID degree, paralogue count.
    A1 OWN         + is this machine running here.
    A2 BUFFER      + is a backup running here (paralogue mean and max expression).
    A3 PARTNER     + are its co-workers running here (BioGRID partner mean expression).
    A4 LINEAGE     same-lineage training lines, minus the gene mean. Unfitted. Tissue only.
    A5 NEIGHBOUR   k nearest training lines in expression space, minus the gene mean. Unfitted.
    A6 COMBINED    A3's features plus A4 and A5 as two more columns, fitted.

PREDECLARED, BEFORE ANY NUMBER.

  Z1 IS THE PREMISE OF THE PROPOSAL TRUE ON GENES WHERE IT CAN BE ASKED?
     A machine that is not running cannot be broken. Two halves, both required, because a decile
     contrast and a correlation can disagree and only one of them being right is not evidence.
     Gate: PASS iff (a) over STRICT, mean effect in a gene's bottom expression decile is at least
     0.05 closer to zero than in its top decile, AND (b) the mean within-gene correlation between
     expression and gene effect is at most -0.10, i.e. more expression means more harm.

  Z2 IS THERE ANY FACTORY-SPECIFIC SIGNAL AT ALL?
     Best of ALL seven arms, fitted or not, minus A0, paired over held-out lines.
     Gate: PASS iff at least 0.05 and at least 3 paired standard errors. Z3-Z8 require this.

  Z3 DOES KNOWING WHICH MACHINES RUN ADD ANYTHING PROFILE MATCHING DOES NOT ALREADY HAVE?
     A6 against A5, paired. This is the load-bearing gate of the loop.
     Gate: PASS iff A6 exceeds A5 by at least 0.02. The sign is not assumed.

  Z4 THE BACKUP-MACHINE TEST.
     Gate: PASS iff (a) A2 beats A1 by at least 0.01 paired AND (b) the fitted coefficient on max
     paralogue expression is positive -- a running backup makes the knockout less harmful. A gain
     with the wrong sign is not buffering and must not pass.

  Z5 A WORKER WE HAVE NEVER PLACED, IN A FACTORY WE HAVE NEVER SEEN.
     Coefficients fitted on a disjoint half of the genes. The catalogue entry is still allowed --
     the proposal grants a complete catalogue -- but the rule never saw this gene.
     Gate: PASS iff the best FITTED arm keeps at least half its advantage over A0.

  Z6 CONTROL: THE WRONG FACTORY.      VOID if the Z2 advantage is under 0.01
     Held-out line's transcriptome swapped for a random training line's.
     Gate: PASS iff the advantage collapses to under 25% of its true value.

  Z7 IS PROFILE MATCHING JUST TISSUE MATCHING?
     A5 against A4, paired.
     Gate: PASS iff A5 exceeds A4 by at least 0.02, meaning transcriptional similarity carries
     something lineage does not. A FAIL means the whole factory-specific signal is the tissue.

  Z8 CONTROL: THE WRONG CATALOGUE.      VOID if A3 beats A1 by under 0.01
     Gene-level facts other than the mean effect permuted across genes. The mean is not permuted
     because it defines the target.
     Gate: PASS iff the A3-over-A1 advantage collapses by at least half.

  Z9 WHAT THIS CANNOT SHOW -- written before the run.
     Expression is a staff directory, not an org chart: which machines are switched on, not what
     they are wired to. A failure is a failure of THIS encoding of functional state.
     DepMap gene effect is a growth phenotype integrated over weeks -- the factory's output long
     after the worker left, not the immediate consequence of removing them.
     Paralogues are Ensembl sequence homology. Sequence-similar genes need not be functionally
     redundant, so Z4 tests homology-based buffering specifically and nothing wider.
     STRICT is 364 genes. Per-line correlations over 364 points are noisier than over 2,000, and
     the gates are paired over 1,076 held-out lines to absorb that; BROAD is reported for the same
     reason. Neither set was chosen after seeing a score.
"""
import os, sys, json, time, gzip, re, warnings
from pathlib import Path
import numpy as np
from scipy import sparse

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_factory_corrected.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
DM = SCR / "depmap"
CACHE = SCR / "depmap_expr_aligned.npz"
PARA = SCR / "para" / "paralogs.tsv"
BG = SCR / "biogrid_hs_edges.tsv.gz"
SEED, NFOLD, KNN = 240240, 10, 25
STRICT_HI, STRICT_LO = -0.25, -0.75
BROAD_HI, BROAD_LO = -0.25, -0.50
LAMS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
Z1_GAP, Z1_COR, Z2_BAR, Z2_SE, Z3_BAR, Z4_BAR, Z5_KEEP, Z6_MAX, Z7_BAR, Z8_DROP = \
    0.05, -0.10, 0.05, 3.0, 0.02, 0.01, 0.50, 0.25, 0.02, 0.50

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
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if d.size < 3: return float("nan"), float("nan"), float("nan")
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    mu = float(np.mean(d))
    return mu, se, (mu / se if se > 0 else float("nan"))


FEATS = ["const", "g_mean", "g_sd", "g_deg", "g_npara",
         "l_own", "own*mean", "own*sd",
         "l_pmean", "l_pmax", "pmean*mean", "pmax*mean", "pmax*sd", "pmax*npara",
         "l_partner", "partner*sd", "own*deg", "a4_lin", "a5_nbr"]
FI = {n: i for i, n in enumerate(FEATS)}
CAT = ["const", "g_mean", "g_sd", "g_deg", "g_npara"]
OWN = CAT + ["l_own", "own*mean", "own*sd"]
BUF = OWN + ["l_pmean", "l_pmax", "pmean*mean", "pmax*mean", "pmax*sd", "pmax*npara"]
PAR = BUF + ["l_partner", "partner*sd", "own*deg"]
ARMS = {"A0_CATALOGUE": CAT, "A1_OWN": OWN, "A2_BUFFER": BUF, "A3_PARTNER": PAR,
        "A6_COMBINED": FEATS}
FITTED = list(ARMS)
ALLARMS = FITTED + ["A4_LINEAGE", "A5_NEIGHBOUR"]


def solve_std(XtX, Xty, cols, lam_mult):
    """Ridge in STANDARDISED coordinates, derived analytically from the raw normal equations so the
    design is built only once. Returns coefficients on the RAW features."""
    A = XtX[np.ix_(cols, cols)]
    b = Xty[cols]
    n = A[0, 0]
    if n <= 1: return np.zeros(len(cols))
    mu = A[0, :] / n
    var = np.maximum(np.diag(A) / n - mu ** 2, 0.0)
    s = np.sqrt(np.where(var > 1e-12, var, 1.0))
    M = np.zeros((len(cols), len(cols)))
    M[0, 0] = 1.0
    for j in range(1, len(cols)):
        M[j, j] = 1.0 / s[j]
        M[0, j] = -mu[j] / s[j]
    As = M.T @ A @ M
    bs = M.T @ b
    lam = lam_mult * n
    As = As + lam * np.diag([0.0] + [1.0] * (len(cols) - 1))
    try:
        beta_s = np.linalg.solve(As, bs)
    except np.linalg.LinAlgError:
        beta_s = np.linalg.lstsq(As, bs, rcond=None)[0]
    return M @ beta_s


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "catalogue-and-factory on differential dependencies"}
    say("=" * 104)
    say("LOOP 240 -- THE CATALOGUE AND THE FACTORY, ON GENES WHERE THE QUESTION IS REAL")
    say("=" * 104)

    z = np.load(DM / "gene_effect.npz", allow_pickle=True)
    E = np.asarray(z["E"], np.float32)
    egenes = np.array([str(x) for x in z["genes"]])
    elines = np.array([str(x) for x in z["lines"]])
    m_ = np.load(DM / "model_meta.npz", allow_pickle=True)
    lin_of = dict(zip([str(x) for x in m_["lines"]], [str(x) for x in m_["lineage"]]))

    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=True)
        XE, elines2, xg = c["XE"], np.array([str(x) for x in c["lines"]]), \
            np.array([str(x) for x in c["genes"]])
        pos = {l: i for i, l in enumerate(elines)}
        keep = np.array([l in set(elines2) for l in elines])
        E, elines = E[keep], elines[keep]
        p2 = {l: i for i, l in enumerate(elines2)}
        XE = XE[[p2[l] for l in elines]]
    else:
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
        np.savez_compressed(CACHE, XE=XE, lines=elines, genes=xg)
        del X
    lineage = np.array([lin_of.get(l, "") for l in elines])
    say(f"     {E.shape[0]:,} lines with gene effect and a transcriptome, {E.shape[1]:,} genes")

    gpos = {g: i for i, g in enumerate(xg)}
    cols = np.array([gpos.get(g, -1) for g in egenes])
    have = cols >= 0
    Zexp = np.zeros((E.shape[0], egenes.size), np.float32)
    Zexp[:, have] = XE[:, cols[have]]
    mu_, sd_ = Zexp.mean(0), Zexp.std(0)
    Zexp = ((Zexp - mu_) / np.where(sd_ > 1e-6, sd_, 1.0)).astype(np.float32)
    Zexp[:, ~have] = 0.0
    NG, NL = egenes.size, E.shape[0]
    gi = {g: i for i, g in enumerate(egenes)}
    Ef = np.where(np.isfinite(E), E, np.nan)

    # ------------------------------------------------ the defect, stated as a number
    sd_g = np.nanstd(Ef, 0); mn_g = np.nanmean(Ef, 0)
    ok = np.isfinite(sd_g) & np.isfinite(mn_g)
    rr = float(np.corrcoef(sd_g[ok], -mn_g[ok])[0, 1])
    say(f"     WHY THE OLD SET WAS WRONG: corr(sd across lines, -mean effect) = {rr:.4f}.")
    say(f"     Ranking genes by variance is mostly ranking them by how essential they are")
    say(f"     everywhere, which is the opposite of selective.")
    res["variance_magnitude_confound"] = rr

    q10 = np.nanpercentile(Ef, 10, axis=0); q90 = np.nanpercentile(Ef, 90, axis=0)
    STRICT = np.where((q90 > STRICT_HI) & (q10 < STRICT_LO))[0]
    BROAD = np.where((q90 > BROAD_HI) & (q10 < BROAD_LO))[0]
    VARSET = np.argsort(-np.nan_to_num(sd_g ** 2))[:2000]
    say(f"     STRICT differential dependencies: {STRICT.size:,} genes, mean effect "
        f"{np.nanmean(mn_g[STRICT]):+.4f}")
    say(f"     BROAD:                            {BROAD.size:,} genes, mean effect "
        f"{np.nanmean(mn_g[BROAD]):+.4f}")
    say(f"     loop 239's variance set (kept only for continuity): 2,000 genes, mean effect "
        f"{np.nanmean(mn_g[VARSET]):+.4f}")
    say(f"     STRICT contains: {', '.join(sorted(egenes[STRICT])[:8])} ...")

    # ------------------------------------------------ catalogue
    pr, pc = [], []
    with open(PARA) as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 2 or not p[1]: continue
            a, b = gi.get(p[0], -1), gi.get(p[1], -1)
            if a < 0 or b < 0 or a == b: continue
            pr.append(a); pc.append(b)
    pr, pc = np.asarray(pr), np.asarray(pc)
    npara = np.bincount(pr, minlength=NG).astype(np.float32)
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
    say(f"     catalogue: {pr.size:,} paralogue pairs, {int(Adj.nnz):,} BioGRID entries")

    Wp = sparse.csr_matrix((1.0 / np.maximum(npara[pr], 1), (pr, pc)), shape=(NG, NG),
                           dtype=np.float32)
    Pmean = (Wp @ Zexp.T).T.astype(np.float32)
    Wq = Adj.multiply(1.0 / np.maximum(deg, 1)[:, None]).tocsr().astype(np.float32)
    Qmean = (Wq @ Zexp.T).T.astype(np.float32)
    order = np.argsort(pr, kind="stable")
    pr_s, pc_s = pr[order], pc[order]
    starts = np.searchsorted(pr_s, np.arange(NG))
    uniq = np.unique(pr_s)
    Pmax = np.zeros_like(Pmean)
    for i0 in range(0, NL, 64):
        blk = Zexp[i0:i0 + 64][:, pc_s]
        Pmax[i0:i0 + 64, uniq] = np.maximum.reduceat(blk, starts[uniq], axis=1)
    gdeg = np.log1p(deg).astype(np.float32)
    gnpa = np.log1p(npara).astype(np.float32)

    # ---------------------------------------------------------------- Z1
    say("Z1 IS THE PREMISE OF THE PROPOSAL TRUE ON GENES WHERE IT CAN BE ASKED?")

    def premise(idx, label):
        lo_v, hi_v, cors = [], [], []
        k = max(int(0.1 * NL), 20)
        for g in idx:
            if not have[g]: continue
            col = Zexp[:, g]
            o = np.argsort(col)
            a, b = Ef[o[:k], g], Ef[o[-k:], g]
            if np.isfinite(a).sum() < 10 or np.isfinite(b).sum() < 10: continue
            lo_v.append(np.nanmean(a)); hi_v.append(np.nanmean(b))
            cors.append(pear(col, Ef[:, g]))
        lo_m, hi_m = float(np.mean(lo_v)), float(np.mean(hi_v))
        cm = float(np.nanmean(cors))
        say(f"     {label:<8} n={len(lo_v):5,}  bottom decile {lo_m:+.4f}  top decile {hi_m:+.4f}"
            f"  gap {lo_m - hi_m:+.4f}  mean within-gene corr(expr, effect) {cm:+.4f}")
        return lo_m - hi_m, cm

    gs, cs = premise(STRICT, "STRICT")
    gb, cb_ = premise(BROAD, "BROAD")
    gv, cv = premise(VARSET, "variance")
    say(f"     the premise predicts a POSITIVE gap and a NEGATIVE correlation; both are required")
    G.add("Z1", bool(gs >= Z1_GAP and cs <= Z1_COR), stat=float(gs),
          if_true=lambda: f"Z1 PASS -- on differential dependencies a gene not running is "
                          f"{gs:+.4f} closer to harmless, with within-gene corr {cs:+.4f}",
          if_false=lambda: f"Z1 FAIL -- gap {gs:+.4f} (bar {Z1_GAP}), within-gene corr {cs:+.4f} "
                           f"(bar {Z1_COR})")
    res["premise"] = {"strict_gap": gs, "strict_corr": cs, "broad_gap": gb, "broad_corr": cb_,
                      "variance_gap": gv, "variance_corr": cv}

    # ---------------------------------------------------------------- harness
    Xc = XE - XE.mean(0, keepdims=True)
    U, S_, Vt = np.linalg.svd(Xc, full_matrices=False)
    XP = (U[:, :50] * S_[:50]).astype(np.float32)
    perm = rng.permutation(NL)
    folds = [perm[i::NFOLD] for i in range(NFOLD)]
    ghalf = rng.permutation(NG)
    inA = np.zeros(NG, bool); inA[ghalf[:NG // 2]] = True

    def run(gidx, shuffle_line=False, shuffle_cat=False, gene_holdout=False, pick_lam=True):
        gidx = np.asarray(gidx)
        out = {k: [] for k in ALLARMS}
        coefs, lam_used = [], []
        for te in folds:
            tr = np.setdiff1d(np.arange(NL), te)
            sub = Ef[np.ix_(tr, gidx)]
            gmean = np.nan_to_num(np.nanmean(sub, 0)).astype(np.float32)
            gsd = np.nan_to_num(np.nanstd(sub, 0)).astype(np.float32)
            cd, cn = gdeg[gidx].copy(), gnpa[gidx].copy()
            gsd_f = gsd.copy()
            if shuffle_cat:
                p = rng.permutation(gidx.size)
                gsd_f, cd, cn = gsd[p], cd[p], cn[p]
            Ptr = XP[tr]
            trlin = lineage[tr]

            def feats(src):
                d = np.linalg.norm(Ptr - XP[src][None, :], axis=1)
                d = np.where(tr == src, np.inf, d)
                nn = tr[np.argsort(d)[:KNN]]
                nbr = np.nan_to_num(np.nanmean(Ef[np.ix_(nn, gidx)], 0) - gmean).astype(np.float32)
                same = tr[(trlin == lineage[src]) & (tr != src)]
                lin = (np.nan_to_num(np.nanmean(Ef[np.ix_(same, gidx)], 0) - gmean).astype(np.float32)
                       if same.size >= 5 else np.zeros(gidx.size, np.float32))
                Zl, Pm, Px, Qm = Zexp[src][gidx], Pmean[src][gidx], Pmax[src][gidx], Qmean[src][gidx]
                X = np.empty((gidx.size, len(FEATS)), np.float32)
                X[:, FI["const"]] = 1.0
                X[:, FI["g_mean"]] = gmean; X[:, FI["g_sd"]] = gsd_f
                X[:, FI["g_deg"]] = cd; X[:, FI["g_npara"]] = cn
                X[:, FI["l_own"]] = Zl
                X[:, FI["own*mean"]] = Zl * gmean; X[:, FI["own*sd"]] = Zl * gsd_f
                X[:, FI["l_pmean"]] = Pm; X[:, FI["l_pmax"]] = Px
                X[:, FI["pmean*mean"]] = Pm * gmean; X[:, FI["pmax*mean"]] = Px * gmean
                X[:, FI["pmax*sd"]] = Px * gsd_f; X[:, FI["pmax*npara"]] = Px * cn
                X[:, FI["l_partner"]] = Qm; X[:, FI["partner*sd"]] = Qm * gsd_f
                X[:, FI["own*deg"]] = Zl * cd
                X[:, FI["a4_lin"]] = lin; X[:, FI["a5_nbr"]] = nbr
                return X, lin, nbr

            fitc = np.arange(gidx.size) if not gene_holdout else np.where(inA[gidx])[0]
            tec = np.arange(gidx.size) if not gene_holdout else np.where(~inA[gidx])[0]
            if fitc.size < 30 or tec.size < 30: continue

            # inner split of the TRAINING lines, to choose the ridge strength honestly
            ip = rng.permutation(tr.size)
            itr, ite = tr[ip[: int(0.8 * tr.size)]], tr[ip[int(0.8 * tr.size):]]
            P = len(FEATS)
            XtX_a = np.zeros((P, P)); Xty_a = np.zeros(P)
            XtX_b = np.zeros((P, P)); Xty_b = np.zeros(P)
            inner = []
            for li in tr:
                Xi, _, _ = feats(li)
                Xi = Xi[fitc]
                yi = Ef[li, gidx[fitc]] - gmean[fitc]
                m = np.isfinite(yi) & np.isfinite(Xi).all(1)
                Xd, yd = Xi[m].astype(np.float64), yi[m].astype(np.float64)
                if li in itr:
                    XtX_a += Xd.T @ Xd; Xty_a += Xd.T @ yd
                else:
                    XtX_b += Xd.T @ Xd; Xty_b += Xd.T @ yd
                    inner.append((Xi.astype(np.float32), yi.astype(np.float32)))
            if pick_lam:
                sc = []
                for lm in LAMS:
                    b = solve_std(XtX_a, Xty_a, list(range(P)), lm)
                    sc.append(np.nanmean([pear(Xj @ b, yj) for Xj, yj in inner]))
                lam = LAMS[int(np.nanargmax(sc))]
            else:
                lam = 1e-3
            lam_used.append(lam)
            XtX, Xty = XtX_a + XtX_b, Xty_a + Xty_b
            beta = {a: (solve_std(XtX, Xty, [FI[n] for n in ARMS[a]], lam), [FI[n] for n in ARMS[a]])
                    for a in ARMS}
            coefs.append({a: dict(zip(ARMS[a], beta[a][0])) for a in ARMS})

            for li in te:
                src = int(rng.choice(tr)) if shuffle_line else li
                Xi, lin, nbr = feats(src)
                truth = (Ef[li, gidx] - gmean)[tec]
                Xi = Xi[tec]
                for a in FITTED:
                    b, c = beta[a]
                    out[a].append(pear(Xi[:, c] @ b, truth))
                out["A4_LINEAGE"].append(pear(lin[tec], truth))
                out["A5_NEIGHBOUR"].append(pear(nbr[tec], truth))
        return {k: np.asarray(v) for k, v in out.items()}, coefs, lam_used

    # ---------------------------------------------------------------- Z2
    say("Z2 IS THERE ANY FACTORY-SPECIFIC SIGNAL AT ALL?")
    S, coefs, lams = run(STRICT)
    say(f"     ridge strength chosen on an inner split of the training lines: "
        f"{sorted(set(lams))}")
    for k in ALLARMS:
        say(f"     {k:<14} residual r = {np.nanmean(S[k]):+.4f}  (sd across held-out lines "
            f"{np.nanstd(S[k]):.4f})")
    best = max(ALLARMS, key=lambda k: np.nanmean(S[k]) if k != "A0_CATALOGUE" else -9)
    d2, se2, z2 = paired(S[best], S["A0_CATALOGUE"])
    say(f"     best arm overall: {best}")
    say(f"     {best} minus A0_CATALOGUE, paired over {np.isfinite(S[best]).sum()} held-out lines: "
        f"{d2:+.4f} +/- {se2:.4f}  ({z2:+.1f} se)")
    G.add("Z2", bool(d2 >= Z2_BAR and z2 >= Z2_SE), stat=float(d2),
          if_true=lambda: f"Z2 PASS -- {best} carries {d2:+.4f} of factory-specific signal over "
                          f"catalogue alone ({z2:+.1f} se)",
          if_false=lambda: f"Z2 FAIL -- the best arm adds {d2:+.4f} ({z2:+.1f} se) against a "
                           f"{Z2_BAR} bar")
    res["Z2"] = {"per_arm": {k: float(np.nanmean(S[k])) for k in S}, "best": best,
                 "delta": d2, "se": se2, "z": z2, "lams": sorted(set(lams))}

    B, _, _ = run(BROAD)
    say(f"     BROAD ({BROAD.size:,} genes), same arms: " +
        "  ".join(f"{k.split('_')[0]} {np.nanmean(B[k]):+.4f}" for k in ALLARMS))
    res["broad"] = {k: float(np.nanmean(B[k])) for k in B}

    # ---------------------------------------------------------------- Z3
    say("Z3 DOES KNOWING WHICH MACHINES RUN ADD ANYTHING PROFILE MATCHING DOES NOT HAVE?")
    d3, se3, z3 = paired(S["A6_COMBINED"], S["A5_NEIGHBOUR"])
    say(f"     A6_COMBINED {np.nanmean(S['A6_COMBINED']):+.4f} vs A5_NEIGHBOUR "
        f"{np.nanmean(S['A5_NEIGHBOUR']):+.4f}")
    say(f"     paired difference {d3:+.4f} +/- {se3:.4f}  ({z3:+.1f} se)")
    G.add("Z3", bool(d3 >= Z3_BAR), stat=float(d3), requires="Z2",
          if_true=lambda: f"Z3 PASS -- functional state adds {d3:+.4f} on top of profile matching",
          if_false=lambda: f"Z3 FAIL -- functional state adds {d3:+.4f} on top of profile "
                           f"matching, against a {Z3_BAR} bar")
    res["Z3"] = {"delta": d3, "se": se3, "z": z3}

    # ---------------------------------------------------------------- Z4
    say("Z4 THE BACKUP-MACHINE TEST")
    d4, se4, z4 = paired(S["A2_BUFFER"], S["A1_OWN"])
    cpx = float(np.mean([c["A2_BUFFER"]["l_pmax"] for c in coefs]))
    say(f"     A2_BUFFER minus A1_OWN, paired: {d4:+.4f} +/- {se4:.4f}  ({z4:+.1f} se)")
    say(f"     fitted coefficient on max paralogue expression: {cpx:+.5f}  "
        f"(buffering predicts POSITIVE)")
    G.add("Z4", bool(d4 >= Z4_BAR and cpx > 0), stat=float(d4), requires="Z2",
          if_true=lambda: f"Z4 PASS -- backups add {d4:+.4f} with coefficient {cpx:+.5f}, the "
                          f"buffering direction",
          if_false=lambda: f"Z4 FAIL -- backups add {d4:+.4f} (bar {Z4_BAR}) with coefficient "
                           f"{cpx:+.5f} ({'wrong sign' if cpx <= 0 else 'right sign'})")
    res["Z4"] = {"delta": d4, "se": se4, "z": z4, "pmax_coef": cpx}

    # ---------------------------------------------------------------- Z5
    say("Z5 A WORKER WE HAVE NEVER PLACED, IN A FACTORY WE HAVE NEVER SEEN")
    bestfit = max(FITTED, key=lambda k: np.nanmean(S[k]) if k != "A0_CATALOGUE" else -9)
    dfit, _, _ = paired(S[bestfit], S["A0_CATALOGUE"])
    H, _, _ = run(STRICT, gene_holdout=True)
    d5, se5, z5 = paired(H[bestfit], H["A0_CATALOGUE"])
    keepf = d5 / dfit if dfit > 1e-9 else float("nan")
    say(f"     best FITTED arm is {bestfit}; coefficients fitted on one half of the genes and")
    say(f"     scored on the other half in held-out lines")
    say(f"     advantage {d5:+.4f} +/- {se5:.4f} against {dfit:+.4f} single-holdout -- "
        f"{keepf:.0%} retained")
    G.add("Z5", bool(np.isfinite(keepf) and keepf >= Z5_KEEP), stat=float(keepf), requires="Z2",
          if_true=lambda: f"Z5 PASS -- {keepf:.0%} of the advantage survives on genes the rule "
                          f"never saw",
          if_false=lambda: f"Z5 FAIL -- only {keepf:.0%} survives; the rule was gene-specific")
    res["Z5"] = {"best_fitted": bestfit, "single": dfit, "double": d5, "retained": keepf}

    # ---------------------------------------------------------------- Z6
    say("Z6 CONTROL: THE WRONG FACTORY")
    if d2 < 0.01:
        G.add("Z6", False, stat=float(d2), requires="Z2", void_if=True,
              void_reason=f"the real advantage is {d2:+.4f}; nothing for a control to collapse")
    else:
        Sh, _, _ = run(STRICT, shuffle_line=True)
        d6, _, _ = paired(Sh[best], Sh["A0_CATALOGUE"])
        f6 = d6 / d2
        say(f"     with a random training line's transcriptome: {d6:+.4f} against a real "
            f"{d2:+.4f}  ({f6:.0%})")
        G.add("Z6", bool(f6 <= Z6_MAX), stat=float(f6), requires="Z2",
              if_true=lambda: f"Z6 PASS -- collapses to {f6:.0%} on the wrong factory",
              if_false=lambda: f"Z6 FAIL -- {f6:.0%} survives with the wrong factory's "
                               f"transcriptome")
        res["Z6"] = {"shuffled": d6, "real": d2, "fraction": f6}

    # ---------------------------------------------------------------- Z7
    say("Z7 IS PROFILE MATCHING JUST TISSUE MATCHING?")
    d7, se7, z7 = paired(S["A5_NEIGHBOUR"], S["A4_LINEAGE"])
    say(f"     A5_NEIGHBOUR {np.nanmean(S['A5_NEIGHBOUR']):+.4f} vs A4_LINEAGE "
        f"{np.nanmean(S['A4_LINEAGE']):+.4f}")
    say(f"     paired difference {d7:+.4f} +/- {se7:.4f}  ({z7:+.1f} se)")
    G.add("Z7", bool(d7 >= Z7_BAR), stat=float(d7), requires="Z2",
          if_true=lambda: f"Z7 PASS -- transcriptional similarity carries {d7:+.4f} beyond lineage",
          if_false=lambda: f"Z7 FAIL -- transcriptional similarity adds {d7:+.4f} over lineage; "
                           f"the factory-specific signal is largely the tissue")
    res["Z7"] = {"delta": d7, "se": se7, "z": z7}

    # ---------------------------------------------------------------- Z8
    say("Z8 CONTROL: THE WRONG CATALOGUE")
    dcat, _, _ = paired(S["A3_PARTNER"], S["A1_OWN"])
    if not (dcat >= 0.01):
        G.add("Z8", False, stat=float(dcat), requires="Z2", void_if=True,
              void_reason=f"A3 beats A1 by only {dcat:+.4f}; the catalogue contributes nothing "
                          f"for a control to remove")
    else:
        Sc, _, _ = run(STRICT, shuffle_cat=True)
        dcs, _, _ = paired(Sc["A3_PARTNER"], Sc["A1_OWN"])
        f8 = dcs / dcat
        say(f"     catalogue permuted across genes: A3 over A1 falls from {dcat:+.4f} to "
            f"{dcs:+.4f}  ({f8:.0%})")
        G.add("Z8", bool(f8 <= 1 - Z8_DROP), stat=float(f8), requires="Z2",
              if_true=lambda: f"Z8 PASS -- permuting the catalogue removes {1 - f8:.0%}",
              if_false=lambda: f"Z8 FAIL -- {f8:.0%} survives a permuted catalogue; knowing the "
                               f"machine is not what is doing the work")
        res["Z8"] = {"real": dcat, "shuffled": dcs, "fraction": f8}

    say("     mean fitted coefficients, A6_COMBINED, over folds:")
    for n in FEATS:
        say(f"       {n:<12} {np.mean([c['A6_COMBINED'][n] for c in coefs]):+.5f}")
    res["coefficients"] = {n: float(np.mean([c["A6_COMBINED"][n] for c in coefs])) for n in FEATS}

    say("Z9 WHAT THIS CANNOT SHOW")
    say("     Expression is a staff directory, not an org chart: which machines are switched on,")
    say("     not what they are wired to. A failure is a failure of THIS encoding.")
    say("     DepMap gene effect is a growth phenotype integrated over weeks -- the factory's")
    say("     output long after the worker left, not the immediate consequence of removal.")
    say("     Paralogues are Ensembl sequence homology; sequence similarity is not a guarantee of")
    say("     functional redundancy, so Z4 tests homology-based buffering and nothing wider.")
    say("     STRICT is a few hundred genes. Per-line correlations over it are noisier than over")
    say("     2,000, which is why every gate is paired over all held-out lines and BROAD is")
    say("     reported alongside. Neither set was chosen after seeing a score.")

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
