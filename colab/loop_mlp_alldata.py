"""Loop 241. A neural network on all of it, against the linear model that already works.

WHAT IS BEING PREDICTED. The same target as loops 239-240: the RESIDUAL of DepMap gene effect
after subtracting each gene's mean over the TRAINING cell lines. Loop 238's line-blind predictor
scores exactly 0 on it by construction and 85% of the raw variance is between genes, so the
residual is the only place a claim about predicting a CELL LINE can be tested. Held out BY CELL
LINE throughout: the network never sees the line it is scored on.

ALL THE DATA, meaning every row rather than the 355-gene subset loop 240 gated on:
    1,076 cell lines x 17,916 genes = 19,277,616 (gene, line) pairs
    19,193-gene transcriptomes, reduced to 50 unsupervised components
    30-odd OncotreeLineage labels, one-hot
    213,696 Ensembl paralogue pairs and 1,503,980 BioGRID entries as neighbourhood expression
Scored on all 17,916 genes and, separately, on two subsets where the question is sharper.

THE GATE THIS LOOP EXISTS FOR IS W3, AND IT IS THERE BECAUSE OF LOOP 225. Loop 225's E6 reported
that an MLP beat ridge. Loop 226's F7 reversed it and loop 229's J5 reversed it again; the MLP's
across-split standard deviation was 0.1394 against ridge's 0.0288, so the original "win" was inside
the noise of the instrument that measured it. A neural network is not shown to help by beating a
WEAKER baseline. It is shown to help by beating a LINEAR MODEL GIVEN THE IDENTICAL INPUTS, by more
than its own seed-to-seed spread. Both halves are predeclared here, W3 and W4, and every neural arm
in this loop is paired with a linear twin that receives exactly the same feature vector.

FOUR ARMS, TWO ARCHITECTURES, EACH WITH ITS LINEAR TWIN.

    PAIR      one row per (gene, line). Input 59 dims: 4 gene facts (training mean, sd, BioGRID
              degree, paralogue count), 4 pair facts (own expression z, paralogue mean and max,
              partner mean), 50 line expression components, 1 nearest-neighbour residual estimate,
              plus the lineage one-hot. Generalises to unseen GENES as well as unseen lines.
                  M1_PAIR_MLP      59 -> 256 -> 128 -> 1
                  L1_PAIR_RIDGE    the same inputs, linear
    PROFILE   one row per line. Input: the line's 50 expression components and lineage one-hot.
              Output: all 17,916 residuals at once. This is the neural form of what profile
              matching does non-parametrically, and it cannot be applied to an unseen gene because
              a gene IS an output unit -- so W5 runs on the PAIR arms only, which is stated here
              rather than discovered later.
                  M2_PROFILE_MLP   in -> 256 -> 17,916
                  L2_PROFILE_RIDGE the same inputs, linear, closed form
    A5_NEIGHBOUR   the k-nearest-transcriptome baseline from loop 240, unfitted. +0.3762 on STRICT.
    A0_CATALOGUE   gene facts only, knows nothing about the held-out line. The floor.

PREDECLARED, BEFORE ANY NUMBER.

  W1 IS THIS THE SAME HARNESS LOOP 240 USED?
     If the target, the folds or the scoring differ, comparing to loop 240's numbers is
     meaningless. A5_NEIGHBOUR is recomputed here and must reproduce.
     Gate: PASS iff A5, RERUN UNDER LOOP 240'S OWN CONFIGURATION (10 folds, neighbours from
     every training line, no inner split), lands within 0.03 of loop 240's +0.3762. It is run
     under loop 240's configuration and not this loop's because this loop deliberately differs on
     all three counts, and A5 legitimately scores lower with a smaller neighbour pool -- a gate
     that failed on a change I made on purpose would void the run while measuring nothing. What
     W1 tests is the DATA AND TARGET pipeline, which is what the comparison actually rests on.
     A5 under this loop's own configuration is reported alongside and is the operative baseline.
     Everything else requires this.

  W2 DOES ANY NETWORK BEAT PROFILE MATCHING?
     Best neural arm against A5_NEIGHBOUR, paired over held-out lines, on STRICT.
     Gate: PASS iff the best neural arm exceeds A5 by at least 0.02.

  EVERY ARM TRAINS ON THE SAME LINES, AND THE NETWORKS STOP WHEN THEY STOP IMPROVING.
     15% of the training lines are held back as an inner validation set. The ridges are fitted on
     the remaining 85%; the networks train on the same 85% and keep the weights from their best
     inner-validation epoch. The neighbour arm draws its neighbours from the same 85%. This is
     stated before the run because it changes what a W3 FAIL means: without it, a losing network
     could simply be undertrained, and "the architecture did not help" would be unsupported. The
     epoch each network actually stopped at is reported, so a budget that binds is visible rather
     than assumed away.

  W3 DID THE NETWORK HELP, OR DID THE FEATURES?      -- the loop 225 gate
     Each MLP against its own linear twin on identical inputs, paired over held-out lines.
     Gate: PASS iff at least one MLP exceeds its twin by at least 0.02 on STRICT. A network that
     beats A5 but not its own linear twin has shown that the FEATURES carry the signal, and saying
     otherwise would repeat loop 225 exactly.

  W4 IS THE ADVANTAGE LARGER THAN THE SEED NOISE?      -- requires W3 to have something to test
     Three seeds per neural arm. The across-seed standard deviation of each arm's mean score.
     Gate: PASS iff the W3 advantage exceeds twice the across-seed sd of the winning MLP. VOID if
     W3 found no advantage, because a spread has nothing to be compared against.

  W5 A GENE THE NETWORK NEVER SAW.
     PAIR arms only, weights fitted on a disjoint half of the genes and scored on the other half
     in the held-out lines.
     Gate: PASS iff at least half of M1's advantage over A0 survives.

  W6 CONTROL: THE WRONG FACTORY.      VOID if the best arm's advantage over A0 is under 0.01
     The held-out line's transcriptome and lineage replaced by a random training line's.
     Gate: PASS iff the advantage over A0 collapses to under 25%.

  W7 WHERE IT WINS AND WHERE IT LOSES -- REPORTED, NOT GATED.
     Every arm on all 17,916 genes, on STRICT (355 differential dependencies), and on MINORITY
     (546 genes strongly essential in a real minority of lines: median effect above -0.25, at
     least 10 lines below -0.75, at most 20% of lines below -0.5). MINORITY is included because
     loop 240's percentile-based set provably EXCLUDED the lineage addictions -- SOX10's 10th
     percentile is only -0.47, since melanoma is about 4% of DepMap -- and those are the genes
     where "is this machine even installed here" is the actual mechanism. MINORITY contains
     SOX10, IRF4, PAX8, MITF, GATA3, EGFR, CTNNB1, TP63, SPI1, MYCN, BRAF, NRAS and SMARCA2.
     All three sets are defined before any model is scored and none is chosen after.

  W8 WHAT THIS CANNOT SHOW -- written before the run.
     A held-out cell line still shares lineage, culture conditions and screening batch with the
     training lines. W6 bounds the transcriptome's contribution but not the batch's.
     DepMap gene effect is a growth phenotype integrated over weeks. Nothing here says anything
     about the immediate consequence of a knockout.
     The PROFILE arms cannot be evaluated on unseen genes at all, by construction, so W5 speaks
     only for the PAIR arms.
     Four CPU threads and a fixed epoch budget. A negative W3 is a statement about what these
     architectures reach under this budget, not a proof that no network can beat a linear model
     on this data.
"""
import os, sys, json, time, gzip, re, copy, warnings
from pathlib import Path
import numpy as np
from scipy import sparse

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

import torch
import torch.nn as nn
torch.set_num_threads(4)

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_mlp_alldata.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
DM = SCR / "depmap"
CACHE = SCR / "depmap_expr_aligned.npz"
PARA = SCR / "para" / "paralogs.tsv"
BG = SCR / "biogrid_hs_edges.tsv.gz"

SEED, NFOLD, KNN, NPC = 241241, 5, 25, 50   # 5 folds, not loop 240's 10: see W1's tolerance
SEEDS = [0, 1, 2]
ROWS_PER_EPOCH, PAIR_MAX_EP, PAIR_PATIENCE, BATCH, LR = 1_500_000, 14, 4, 8192, 3e-3
PROFILE_MAX_EP, PROFILE_PATIENCE, PROFILE_LR = 400, 40, 3e-3
VAL_FRAC = 0.15
LOOP240_A5 = 0.3762
W1_TOL, W2_BAR, W3_BAR, W4_MULT, W5_KEEP, W6_MAX = 0.03, 0.02, 0.02, 2.0, 0.50, 0.25

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear_cols(P, T):
    """Pearson down each column of P against T, both (ngene, nline). Returns (nline,)."""
    m = np.isfinite(P) & np.isfinite(T)
    out = np.full(P.shape[1], np.nan)
    for j in range(P.shape[1]):
        a, b = P[m[:, j], j], T[m[:, j], j]
        if a.size < 5: continue
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        if d > 0: out[j] = float((a * b).sum() / d)
    return out


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if d.size < 3: return float("nan"), float("nan"), float("nan")
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    mu = float(np.mean(d))
    return mu, se, (mu / se if se > 0 else float("nan"))


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "MLP on all 19.3M (gene, line) pairs, held out by cell line"}
    say("=" * 104)
    say("LOOP 241 -- A NEURAL NETWORK ON ALL OF IT, AGAINST THE LINEAR MODEL THAT ALREADY WORKS")
    say("=" * 104)
    say("     W3 is the gate this loop exists for: every MLP is paired with a LINEAR TWIN that")
    say("     receives the identical feature vector. Loop 225's E6 reported an MLP win that")
    say("     loops 226 and 229 both reversed, with across-split sd 0.1394 against ridge's 0.0288.")

    # ---------------------------------------------------------------- data
    z = np.load(DM / "gene_effect.npz", allow_pickle=True)
    E = np.asarray(z["E"], np.float32)
    egenes = np.array([str(x) for x in z["genes"]])
    elines = np.array([str(x) for x in z["lines"]])
    mm = np.load(DM / "model_meta.npz", allow_pickle=True)
    lin_of = dict(zip([str(x) for x in mm["lines"]], [str(x) for x in mm["lineage"]]))
    c = np.load(CACHE, allow_pickle=True)
    XEall = c["XE"]; xlines = np.array([str(x) for x in c["lines"]])
    xg = np.array([str(x) for x in c["genes"]])
    p2 = {l: i for i, l in enumerate(xlines)}
    keep = np.array([l in p2 for l in elines])
    E, elines = E[keep], elines[keep]
    XE = XEall[[p2[l] for l in elines]]
    lineage = np.array([lin_of.get(l, "") for l in elines])
    NL, NG = E.shape
    Ef = np.where(np.isfinite(E), E, np.nan)
    say(f"     {NL:,} cell lines x {NG:,} genes = {NL * NG:,} (gene, line) pairs")

    gpos = {g: i for i, g in enumerate(xg)}
    cols = np.array([gpos.get(g, -1) for g in egenes]); have = cols >= 0
    Zexp = np.zeros((NL, NG), np.float32); Zexp[:, have] = XE[:, cols[have]]
    mu_, sd_ = Zexp.mean(0), Zexp.std(0)
    Zexp = ((Zexp - mu_) / np.where(sd_ > 1e-6, sd_, 1.0)).astype(np.float32)
    Zexp[:, ~have] = 0.0

    gi = {g: i for i, g in enumerate(egenes)}
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
    Adj = sparse.csr_matrix((np.ones(len(er), np.float32), (np.asarray(er), np.asarray(ec))),
                            shape=(NG, NG))
    Adj.sum_duplicates(); Adj.data[:] = 1.0
    deg = np.asarray(Adj.sum(1)).ravel().astype(np.float32)
    Wp = sparse.csr_matrix((1.0 / np.maximum(npara[pr], 1), (pr, pc)), shape=(NG, NG),
                           dtype=np.float32)
    Pmean = (Wp @ Zexp.T).T.astype(np.float32)
    Wq = Adj.multiply(1.0 / np.maximum(deg, 1)[:, None]).tocsr().astype(np.float32)
    Qmean = (Wq @ Zexp.T).T.astype(np.float32)
    order = np.argsort(pr, kind="stable"); pr_s, pc_s = pr[order], pc[order]
    starts = np.searchsorted(pr_s, np.arange(NG)); uniq = np.unique(pr_s)
    Pmax = np.zeros_like(Pmean)
    for i0 in range(0, NL, 64):
        Pmax[i0:i0 + 64, uniq] = np.maximum.reduceat(Zexp[i0:i0 + 64][:, pc_s],
                                                     starts[uniq], axis=1)
    gdeg = np.log1p(deg).astype(np.float32); gnpa = np.log1p(npara).astype(np.float32)
    say(f"     catalogue: {pr.size:,} paralogue pairs, {int(Adj.nnz):,} BioGRID entries")

    Xc = XE - XE.mean(0, keepdims=True)
    U, S_, _ = np.linalg.svd(Xc, full_matrices=False)
    XP = (U[:, :NPC] * S_[:NPC]).astype(np.float32)          # loop 240's metric, for kNN
    XPF = ((XP - XP.mean(0)) / (XP.std(0) + 1e-6)).astype(np.float32)   # for the feature vector
    lin_names = sorted(set(lineage))
    LH = np.zeros((NL, len(lin_names)), np.float32)
    for i, l in enumerate(lineage): LH[i, lin_names.index(l)] = 1.0
    LINE = np.concatenate([XPF, LH], 1).astype(np.float32)
    say(f"     line features: {NPC} expression components "
        f"({np.sum(S_[:NPC] ** 2) / np.sum(S_ ** 2):.1%} of variance) + {len(lin_names)} lineages")

    # ---------------------------------------------------------------- gene sets, W7
    q10 = np.nanpercentile(Ef, 10, 0); q90 = np.nanpercentile(Ef, 90, 0)
    med = np.nanmedian(Ef, 0); n75 = np.nansum(Ef < -0.75, 0); n50 = np.nansum(Ef < -0.5, 0)
    SETS = {
        "ALL": np.arange(NG),
        "STRICT": np.where((q90 > -0.25) & (q10 < -0.75))[0],
        "MINORITY": np.where((med > -0.25) & (n75 >= 10) & (n50 <= 0.20 * NL))[0],
    }
    for k, v in SETS.items():
        say(f"     {k:<9} {v.size:6,} genes, mean effect {np.nanmean(np.nanmean(Ef, 0)[v]):+.4f}")
    hits = [g for g in ["SOX10", "IRF4", "PAX8", "MITF", "GATA3", "BRAF", "NRAS", "MYCN"]
            if gi.get(g, -1) in set(SETS["MINORITY"].tolist())]
    say(f"     MINORITY contains {', '.join(hits)}")

    NLF = LINE.shape[1]
    NF = 4 + 4 + NLF + 1
    perm = rng.permutation(NL)
    folds = [perm[i::NFOLD] for i in range(NFOLD)]
    ghalf = rng.permutation(NG); inA = np.zeros(NG, bool); inA[ghalf[:NG // 2]] = True

    FOLD_CACHE = {}

    def rows_for_line(li, src, GF, NBR, g=None):
        """(len(g), NF) design for one line: gene facts, pair facts, line facts, neighbour.
        `g` restricts to a gene subset so inner-validation scoring does not build all 17,916
        rows to read 355 of them."""
        idx = slice(None) if g is None else g
        n = NG if g is None else len(g)
        X = np.empty((n, NF), np.float32)
        X[:, 0:4] = GF[idx]
        X[:, 4] = Zexp[src][idx]; X[:, 5] = Pmean[src][idx]
        X[:, 6] = Pmax[src][idx]; X[:, 7] = Qmean[src][idx]
        X[:, 8:8 + NLF] = LINE[src][None, :]
        X[:, 8 + NLF] = NBR[li][idx]
        return X

    class PairNet(nn.Module):
        def __init__(s, d):
            super().__init__()
            s.f = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                                nn.Linear(128, 1))
        def forward(s, x): return s.f(x).squeeze(-1)

    class ProfileNet(nn.Module):
        def __init__(s, d, o):
            super().__init__()
            s.f = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, o))
        def forward(s, x): return s.f(x)


    def build_fold(fi, tr, te, shuffle_line, gene_holdout):
        """Everything that does not depend on the network seed, built once per fold and reused
        across seeds: the inner split, the gene statistics, the neighbour estimates, the feature
        standardisation and both ridges. The inner split uses a FOLD-indexed generator, not the
        model seed, so the cache is valid and every seed sees the same data."""
        key = (fi, shuffle_line, gene_holdout)
        if key in FOLD_CACHE: return FOLD_CACHE[key]
        rf = np.random.default_rng(SEED + 1000 * fi)
        ip = rf.permutation(tr.size)
        nval = max(20, int(VAL_FRAC * tr.size))
        trv, trf = tr[ip[:nval]], tr[ip[nval:]]

        sub = Ef[np.ix_(trf, np.arange(NG))]
        gmean = np.nan_to_num(np.nanmean(sub, 0)).astype(np.float32)
        gsd = np.nan_to_num(np.nanstd(sub, 0)).astype(np.float32)
        R = (Ef - gmean[None, :]).astype(np.float32)   # residual, NaNs PRESERVED (0.85% of the
        Rz = np.nan_to_num(R)                          # matrix); Rz only where an average needs it
        srcs = np.arange(NL)
        if shuffle_line:
            srcs = srcs.copy()
            for li in te: srcs[li] = int(rf.choice(trf))
        NBR = np.zeros((NL, NG), np.float32)
        for li in range(NL):
            sc = srcs[li]
            d = np.linalg.norm(XP[trf] - XP[sc][None, :], axis=1)
            d = np.where(trf == sc, np.inf, d)
            NBR[li] = Rz[trf[np.argsort(d)[:KNN]]].mean(0)
        GF = np.stack([gmean, gsd, gdeg, gnpa], 1).astype(np.float32)

        fitg = np.where(inA)[0] if gene_holdout else np.arange(NG)
        teg = np.where(~inA)[0] if gene_holdout else np.arange(NG)

        samp = trf[rf.permutation(trf.size)[:80]]
        Xs = np.concatenate([rows_for_line(li, srcs[li], GF, NBR, fitg) for li in samp], 0)
        fmu, fsd = Xs.mean(0), Xs.std(0) + 1e-6
        del Xs

        P = NF + 1
        XtX = np.zeros((P, P)); Xty = np.zeros(P)
        for li in trf:                                 # EVERY fitting line, not a sample: the
            Xi = (rows_for_line(li, srcs[li], GF, NBR, fitg) - fmu) / fsd  # linear twin must not
            yi = R[li, fitg]; ok = np.isfinite(yi)                          # be starved of rows
            Xa = np.concatenate([Xi[ok], np.ones((int(ok.sum()), 1), np.float32)],
                                1).astype(np.float64)
            XtX += Xa.T @ Xa; Xty += Xa.T @ yi[ok].astype(np.float64)
        lam = 1e-3 * np.trace(XtX) / P
        beta = np.linalg.solve(XtX + lam * np.eye(P), Xty)
        c0 = [0, 1, 2, 3, NF]
        beta0 = np.linalg.solve(XtX[np.ix_(c0, c0)] + lam * np.eye(len(c0)), Xty[c0])

        Lf = LINE[srcs[trf]]
        lmu, lsd = Lf.mean(0), Lf.std(0) + 1e-6
        Lf_s = ((Lf - lmu) / lsd).astype(np.float32)
        Wlin = None
        if not gene_holdout:
            Yf = R[np.ix_(trf, fitg)]
            Mf = np.isfinite(Yf).astype(np.float32)
            Yf = np.nan_to_num(Yf)
            A = Lf_s.T @ Lf_s + 1e-2 * np.trace(Lf_s.T @ Lf_s) / NLF * np.eye(NLF)
            Wlin = np.linalg.solve(A, Lf_s.T @ (Yf * Mf))
        FOLD_CACHE[key] = dict(trf=trf, trv=trv, gmean=gmean, R=R, NBR=NBR, GF=GF, srcs=srcs,
                               fitg=fitg, teg=teg, fmu=fmu, fsd=fsd, beta=beta, beta0=beta0,
                               c0=c0, lmu=lmu, lsd=lsd, Lf_s=Lf_s, Wlin=Wlin)
        return FOLD_CACHE[key]

    def val_score(pred_fn, F, gset):
        """Mean residual correlation on the inner validation LINES -- the early-stopping signal.
        Never touches the held-out fold."""
        g = np.intersect1d(F["fitg"], SETS[gset])
        pr_, tr_ = [], []
        for li in F["trv"]:
            pr_.append(pred_fn(li, g)); tr_.append(F["R"][li, g])
        return float(np.nanmean(pear_cols(np.stack(pr_, 1), np.stack(tr_, 1))))

    def run_fold(fi, tr, te, seed, shuffle_line=False, gene_holdout=False):
        F = build_fold(fi, tr, te, shuffle_line, gene_holdout)
        trf, fitg, teg = F["trf"], F["fitg"], F["teg"]
        fmu, fsd, R, NBR, GF, srcs = F["fmu"], F["fsd"], F["R"], F["NBR"], F["GF"], F["srcs"]
        torch.manual_seed(seed); r2 = np.random.default_rng(seed)
        lossf = nn.MSELoss()
        stops = {}

        # ---- M1 PAIR MLP, early stopped on the inner validation lines
        net = PairNet(NF)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        per_epoch_lines = max(1, int(round(ROWS_PER_EPOCH / max(fitg.size, 1))))
        def m1_pred(li, g):
            with torch.no_grad():
                return net(torch.from_numpy(
                    (rows_for_line(li, srcs[li], GF, NBR, g) - fmu) / fsd)).numpy()
        best, best_ep, bad, bw = -9.0, 0, 0, None
        for ep in range(1, PAIR_MAX_EP + 1):
            net.train()
            for li in r2.permutation(trf)[:per_epoch_lines]:
                Xi = (rows_for_line(li, srcs[li], GF, NBR, fitg) - fmu) / fsd
                yi = R[li, fitg]; ok = np.isfinite(yi)
                xt = torch.from_numpy(Xi[ok]); yt = torch.from_numpy(yi[ok])
                idx = torch.randperm(xt.shape[0])
                for b0 in range(0, xt.shape[0], BATCH):
                    k = idx[b0:b0 + BATCH]
                    opt.zero_grad(); lossf(net(xt[k]), yt[k]).backward(); opt.step()
            net.eval()
            v = val_score(m1_pred, F, "STRICT")
            if v > best: best, best_ep, bad, bw = v, ep, 0, copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= PAIR_PATIENCE: break
        if bw is not None: net.load_state_dict(bw)
        net.eval()
        stops["M1_PAIR_MLP"] = (best_ep, ep, best)

        # ---- M2 PROFILE MLP
        pnet = None
        if not gene_holdout:
            Yf = np.nan_to_num(R[np.ix_(trf, fitg)]); Mf = np.isfinite(R[np.ix_(trf, fitg)]).astype(np.float32)
            xt = torch.from_numpy(F["Lf_s"]); yt = torch.from_numpy(Yf); mt = torch.from_numpy(Mf)
            pnet = ProfileNet(NLF, fitg.size)
            popt = torch.optim.Adam(pnet.parameters(), lr=PROFILE_LR)
            def m2_pred(li, g):
                pos = np.searchsorted(fitg, g)
                with torch.no_grad():
                    ls = ((LINE[srcs[li]] - F["lmu"]) / F["lsd"]).astype(np.float32)
                    return pnet(torch.from_numpy(ls[None, :])).numpy()[0][pos]
            best2, best_ep2, bad2, bw2 = -9.0, 0, 0, None
            for ep2 in range(1, PROFILE_MAX_EP + 1):
                popt.zero_grad()
                (((pnet(xt) - yt) ** 2) * mt).sum().div(mt.sum()).backward(); popt.step()
                if ep2 % 5 == 0:
                    v2 = val_score(m2_pred, F, "STRICT")
                    if v2 > best2: best2, best_ep2, bad2, bw2 = v2, ep2, 0, copy.deepcopy(pnet.state_dict())
                    else:
                        bad2 += 5
                        if bad2 >= PROFILE_PATIENCE: break
            if bw2 is not None: pnet.load_state_dict(bw2)
            stops["M2_PROFILE_MLP"] = (best_ep2, ep2, best2)

        # ---- score the held-out lines
        preds = {k: np.zeros((teg.size, te.size), np.float32)
                 for k in ("A0_CATALOGUE", "A5_NEIGHBOUR", "M1_PAIR_MLP", "L1_PAIR_RIDGE",
                           "M2_PROFILE_MLP", "L2_PROFILE_RIDGE")}
        truth = np.zeros((teg.size, te.size), np.float32)
        for j, li in enumerate(te):
            Xi = (rows_for_line(li, srcs[li], GF, NBR, teg) - fmu) / fsd
            Xc1 = np.concatenate([Xi, np.ones((Xi.shape[0], 1), np.float32)], 1)
            truth[:, j] = R[li, teg]                       # NaNs preserved; pear_cols masks them
            preds["A5_NEIGHBOUR"][:, j] = NBR[li, teg]
            preds["A0_CATALOGUE"][:, j] = Xc1[:, F["c0"]] @ F["beta0"]
            preds["L1_PAIR_RIDGE"][:, j] = Xc1 @ F["beta"]
            with torch.no_grad():
                preds["M1_PAIR_MLP"][:, j] = net(torch.from_numpy(Xi)).numpy()
            if gene_holdout:
                preds["L2_PROFILE_RIDGE"][:, j] = np.nan
                preds["M2_PROFILE_MLP"][:, j] = np.nan
            else:
                ls = ((LINE[srcs[li]] - F["lmu"]) / F["lsd"]).astype(np.float32)
                preds["L2_PROFILE_RIDGE"][:, j] = ls @ F["Wlin"]
                with torch.no_grad():
                    preds["M2_PROFILE_MLP"][:, j] = pnet(torch.from_numpy(ls[None, :])).numpy()[0]
        out = {}
        for k in preds:
            out[k] = {}
            for sname in SETS:
                g = np.intersect1d(teg, SETS[sname]); pos = np.searchsorted(teg, g)
                out[k][sname] = pear_cols(preds[k][pos], truth[pos])
        return out, stops

    ARMN = ["A0_CATALOGUE", "A5_NEIGHBOUR", "L1_PAIR_RIDGE", "M1_PAIR_MLP",
            "L2_PROFILE_RIDGE", "M2_PROFILE_MLP"]

    STOPS = []

    def run_all(seed, shuffle_line=False, gene_holdout=False, tag=""):
        acc = {a: {s: [] for s in SETS} for a in ARMN}
        for fi, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(NL), te)
            o, st = run_fold(fi, tr, te, seed, shuffle_line, gene_holdout)
            for a in ARMN:
                for s in SETS: acc[a][s].append(o[a][s])
            STOPS.append({"tag": tag, "seed": seed, "fold": fi, **{k: list(v) for k, v in st.items()}})
            say(f"       {tag}seed {seed} fold {fi + 1}/{NFOLD} done  " +
                "  ".join(f"{k.split('_')[0]} best ep {v[0]}/{v[1]} val {v[2]:+.4f}"
                          for k, v in st.items()) + f"  [{time.time() - t0:.0f}s]")
        return {a: {s: np.concatenate(acc[a][s]) for s in SETS} for a in ARMN}

    say("     training ... 5 folds x 3 seeds for the neural arms")
    runs = {sd: run_all(sd) for sd in SEEDS}
    S = runs[SEEDS[0]]
    res["per_seed"] = {str(sd): {a: {s: float(np.nanmean(runs[sd][a][s])) for s in SETS}
                                 for a in ARMN} for sd in SEEDS}

    # ---------------------------------------------------------------- W1
    say("W1 IS THIS THE SAME HARNESS LOOP 240 USED?")
    say("     W1 reruns A5 under LOOP 240's configuration -- 10 folds, neighbours drawn from every")
    say("     training line, no inner validation split -- because this loop deliberately differs on")
    say("     all three, and a gate that failed on a design change I made on purpose would void the")
    say("     run while measuring nothing. The reproduction tests the DATA AND TARGET pipeline.")
    r10 = np.random.default_rng(240240)
    p10 = r10.permutation(NL)
    rep = []
    st = SETS["STRICT"]
    for te10 in [p10[i::10] for i in range(10)]:
        tr10 = np.setdiff1d(np.arange(NL), te10)
        gm = np.nan_to_num(np.nanmean(Ef[np.ix_(tr10, np.arange(NG))], 0)).astype(np.float32)
        Rr = np.nan_to_num(Ef - gm[None, :]).astype(np.float32)
        for li in te10:
            d = np.linalg.norm(XP[tr10] - XP[li][None, :], axis=1)
            d = np.where(tr10 == li, np.inf, d)
            nbr10 = tr10[np.argsort(d)[:KNN]]     # NOT `nn`: that is the torch.nn import, and
            pr_ = Rr[nbr10].mean(0)[st]           # binding it here unbinds it for every closure
            tr_ = (Ef[li, st] - gm[st])
            m = np.isfinite(pr_) & np.isfinite(tr_)
            a_, b_ = pr_[m] - pr_[m].mean(), tr_[m] - tr_[m].mean()
            dd = np.sqrt((a_ * a_).sum() * (b_ * b_).sum())
            rep.append(float((a_ * b_).sum() / dd) if dd > 0 else np.nan)
    a5rep = float(np.nanmean(rep))
    a5 = float(np.nanmean(S["A5_NEIGHBOUR"]["STRICT"]))
    say(f"     A5 under loop 240's configuration: {a5rep:+.4f} against loop 240's "
        f"{LOOP240_A5:+.4f}  (difference {a5rep - LOOP240_A5:+.4f})")
    say(f"     A5 under THIS loop's configuration ({NFOLD} folds, 15% of training lines held back")
    say(f"     for early stopping, so {int((1 - VAL_FRAC) * (NL - NL / NFOLD)):,} neighbour")
    say(f"     candidates instead of {int(NL - NL / 10):,}): {a5:+.4f}. That is the operative")
    say(f"     baseline every arm below is compared against.")
    G.add("W1", bool(abs(a5rep - LOOP240_A5) <= W1_TOL), stat=float(a5rep),
          if_true=lambda: f"W1 PASS -- the pipeline reproduces loop 240 to "
                          f"{abs(a5rep - LOOP240_A5):.4f}",
          if_false=lambda: f"W1 FAIL -- A5 reproduces at {a5rep:+.4f} against loop 240's "
                           f"{LOOP240_A5:+.4f}; the data or target pipeline has changed and the "
                           f"comparison to loop 240 is not valid")
    res["W1"] = {"reproduction": a5rep, "loop240": LOOP240_A5, "operative": a5}

    # ---------------------------------------------------------------- W7 table first
    say("W7 EVERY ARM ON EVERY GENE SET -- reported, not gated")
    say(f"     {'arm':<18}" + "".join(f"{s:>12}" for s in SETS))
    for a in ARMN:
        say(f"     {a:<18}" + "".join(
            f"{np.mean([np.nanmean(runs[sd][a][s]) for sd in SEEDS]):>+12.4f}" for s in SETS))
    res["W7"] = {a: {s: float(np.mean([np.nanmean(runs[sd][a][s]) for sd in SEEDS]))
                     for s in SETS} for a in ARMN}

    # ---------------------------------------------------------------- W2
    say("W2 DOES ANY NETWORK BEAT PROFILE MATCHING?")
    neur = ["M1_PAIR_MLP", "M2_PROFILE_MLP"]
    bestn = max(neur, key=lambda a: np.nanmean(S[a]["STRICT"]))
    d2, se2, z2 = paired(S[bestn]["STRICT"], S["A5_NEIGHBOUR"]["STRICT"])
    say(f"     best neural arm {bestn} {np.nanmean(S[bestn]['STRICT']):+.4f} vs A5 {a5:+.4f}")
    say(f"     paired {d2:+.4f} +/- {se2:.4f}  ({z2:+.1f} se)")
    G.add("W2", bool(d2 >= W2_BAR), stat=float(d2), requires=("W1",),
          if_true=lambda: f"W2 PASS -- {bestn} beats profile matching by {d2:+.4f}",
          if_false=lambda: f"W2 FAIL -- the best network is {d2:+.4f} against profile matching")
    res["W2"] = {"best_neural": bestn, "delta": d2, "se": se2, "z": z2}

    # ---------------------------------------------------------------- W3
    say("W3 DID THE NETWORK HELP, OR DID THE FEATURES?")
    twins = [("M1_PAIR_MLP", "L1_PAIR_RIDGE"), ("M2_PROFILE_MLP", "L2_PROFILE_RIDGE")]
    tw = {}
    for m, l in twins:
        d, se, zz = paired(S[m]["STRICT"], S[l]["STRICT"])
        tw[m] = (d, se, zz)
        say(f"     {m} {np.nanmean(S[m]['STRICT']):+.4f} vs its linear twin {l} "
            f"{np.nanmean(S[l]['STRICT']):+.4f}   paired {d:+.4f} +/- {se:.4f} ({zz:+.1f} se)")
    bestm = max(tw, key=lambda k: tw[k][0])
    d3 = tw[bestm][0]
    G.add("W3", bool(d3 >= W3_BAR), stat=float(d3), requires=("W1",),
          if_true=lambda: f"W3 PASS -- {bestm} beats its own linear twin by {d3:+.4f}; the "
                          f"architecture is doing work the features do not",
          if_false=lambda: f"W3 FAIL -- the best MLP beats its linear twin by {d3:+.4f} against a "
                           f"{W3_BAR} bar; the signal is in the features, not the network")
    res["W3"] = {a: {"delta": tw[a][0], "se": tw[a][1], "z": tw[a][2]} for a in tw}

    # ---------------------------------------------------------------- W4
    say("W4 IS THE ADVANTAGE LARGER THAN THE SEED NOISE?")
    sds = {a: float(np.std([np.nanmean(runs[sd][a]["STRICT"]) for sd in SEEDS], ddof=1))
           for a in neur}
    for a in neur:
        say(f"     {a} across {len(SEEDS)} seeds: " +
            ", ".join(f"{np.nanmean(runs[sd][a]['STRICT']):+.4f}" for sd in SEEDS) +
            f"   sd {sds[a]:.4f}")
    if not (d3 >= W3_BAR):
        G.add("W4", False, stat=float(d3), requires=("W1",), void_if=True,
              void_reason=f"W3 found no advantage ({d3:+.4f}); a seed spread has nothing to be "
                          f"compared against")
    else:
        G.add("W4", bool(d3 >= W4_MULT * sds[bestm]), stat=float(sds[bestm]), requires=("W1",),
              if_true=lambda: f"W4 PASS -- the {d3:+.4f} advantage is {d3 / max(sds[bestm], 1e-9):.1f}x "
                              f"the across-seed sd of {sds[bestm]:.4f}",
              if_false=lambda: f"W4 FAIL -- a {d3:+.4f} advantage against an across-seed sd of "
                               f"{sds[bestm]:.4f}; this is loop 225 again")
    res["W4"] = {"seed_sd": sds}

    # ---------------------------------------------------------------- W5
    say("W5 A GENE THE NETWORK NEVER SAW")
    H = run_all(SEEDS[0], gene_holdout=True, tag="genehold ")
    dfull, _, _ = paired(S["M1_PAIR_MLP"]["STRICT"], S["A0_CATALOGUE"]["STRICT"])
    dhalf, se5, _ = paired(H["M1_PAIR_MLP"]["STRICT"], H["A0_CATALOGUE"]["STRICT"])
    kf = dhalf / dfull if dfull > 1e-9 else float("nan")
    say(f"     M1 over A0: {dfull:+.4f} with all genes, {dhalf:+.4f} +/- {se5:.4f} when the")
    say(f"     weights never saw the scored genes -- {kf:.0%} retained")
    G.add("W5", bool(np.isfinite(kf) and kf >= W5_KEEP), stat=float(kf), requires=("W1",),
          if_true=lambda: f"W5 PASS -- {kf:.0%} survives on genes the network never saw",
          if_false=lambda: f"W5 FAIL -- only {kf:.0%} survives; the network learned genes, "
                           f"not a rule")
    res["early_stopping"] = STOPS
    res["W5"] = {"full": dfull, "gene_holdout": dhalf, "retained": kf}

    # ---------------------------------------------------------------- W6
    say("W6 CONTROL: THE WRONG FACTORY")
    bestall = max([a for a in ARMN if a != "A0_CATALOGUE"],
                  key=lambda a: np.nanmean(S[a]["STRICT"]))
    dreal, _, _ = paired(S[bestall]["STRICT"], S["A0_CATALOGUE"]["STRICT"])
    if dreal < 0.01:
        G.add("W6", False, stat=float(dreal), requires=("W1",), void_if=True,
              void_reason=f"the best arm's advantage is {dreal:+.4f}; nothing to collapse")
    else:
        Sh = run_all(SEEDS[0], shuffle_line=True, tag="wrongfac ")
        dsh, _, _ = paired(Sh[bestall]["STRICT"], Sh["A0_CATALOGUE"]["STRICT"])
        f6 = dsh / dreal
        say(f"     {bestall} with a random training line's transcriptome and lineage: {dsh:+.4f} "
            f"against a real {dreal:+.4f}  ({f6:.0%})")
        G.add("W6", bool(f6 <= W6_MAX), stat=float(f6), requires=("W1",),
              if_true=lambda: f"W6 PASS -- collapses to {f6:.0%} on the wrong factory",
              if_false=lambda: f"W6 FAIL -- {f6:.0%} survives the wrong factory")
        res["W6"] = {"real": dreal, "shuffled": dsh, "fraction": f6, "arm": bestall}

    # ---------------------------------------------------------------- W8
    say("W8 WHAT THIS CANNOT SHOW")
    say("     A held-out line still shares lineage, culture and screening batch with the training")
    say("     lines. W6 bounds the transcriptome's contribution, not the batch's.")
    say("     Gene effect is a growth phenotype integrated over weeks; nothing here speaks to the")
    say("     immediate consequence of a knockout.")
    say("     The PROFILE arms cannot be applied to an unseen gene at all -- a gene IS an output")
    say("     unit -- so W5 speaks only for the PAIR arms.")
    say("     Four CPU threads and a fixed epoch budget. A negative W3 is a statement about what")
    say("     these architectures reach here, not a proof that no network can beat a linear model.")

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
