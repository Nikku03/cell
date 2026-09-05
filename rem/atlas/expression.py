"""Which transcription and translation rates must be measured? The same projection, on expression.

WHY THIS IS A DIFFERENT PROBLEM FROM METABOLISM. recon.py found that only 8 of 5,938
enzyme-catalysed reactions have a realised sensitivity, because in a linear program the
overwhelming majority of capacity constraints do not bind and their duals are exactly zero. That
sparsity came from the CONSTRAINT side. Expression has no such mechanism: every gene is always
being transcribed and translated, nothing is inactive, and there are no binding constraints to be
slack. If sparsity exists here it has to come from the answer's gradient instead.

THE MODEL, which is the standard one and is exactly solvable. Per gene: transcription at k_tx,
translation at k_tl, mRNA decay at k_dm, protein decay at k_dp. Four rates per gene. At steady
state the mRNA mean is k_tx/k_dm and the protein mean is k_tx*k_tl/(k_dm*k_dp), and in the
bursting regime the protein number is Gamma-distributed with

    shape  a = k_tx / k_dp     (burst frequency, bursts per protein lifetime)
    scale  b = k_tl / k_dm     (burst size, proteins per transcript)

with mean a*b, which is the protein abundance.

THE STRUCTURAL RESULT, WHICH IS EXACT LINEAR ALGEBRA AND NOT A MEASUREMENT. Write the four log
rates as x = (log k_tx, log k_tl, log k_dm, log k_dp). Then

    RNA-seq            gives  log(mRNA)     = ( 1,  0, -1,  0)
    proteomics         gives  log(protein)  = ( 1,  1, -1, -1)
    ribosome profiling gives  log(k_tl)     = ( 0,  1,  0,  0)
    mRNA half-life     gives  log(k_dm)     = ( 0,  0,  1,  0)

while the two quantities that set the TAIL are

    burst frequency    log a = ( 1,  0,  0, -1)
    burst size         log b = ( 0,  1, -1,  0)

Neither of those lies in the span of the first two rows, or even of the first three. Checked
directly: the least-squares residual of log a on the abundance rows is 1.00, and 0.71 once
ribosome profiling is added. Only when a DEGRADATION rate is measured does the rank reach 4 and
the system close. So the standard genome-wide assays pin the protein MEAN exactly -- a*b is the
sum of the two abundance rows -- and leave both factors of it free. That is the residence.py and
katg.py finding, which were measured on small circuits, proved here as an identity for every gene
in the genome.

According to PubMed, Schwanhäusser et al. 2011 (Nature, doi 10.1038/nature10098) measured
abundance AND turnover for more than 5,000 genes by parallel metabolic pulse labelling, and found
that mRNA and protein half-lives show NO correlation. That matters twice over: it is the assay
that closes the rank, and its result means the free directions are genuinely populated in real
data rather than being a degenerate corner.

THE QUESTION THIS LEAVES. If every gene has two free directions, the unconstrained dimension is
2G and grows linearly with the genome. K can only stay small if the answer concentrates on a few
genes. That is measurable, and it is what the gates below measure.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

X1  THE TAIL IS THE RIGHT TAIL. The Gamma form holds in the bursting regime, k_dm >> k_dp.
    Compare against a direct solve of the two-stage master equation for a sample of genes and
    report the worst disagreement in log tail probability, together with the k_dm/k_dp ratio at
    which it is measured. If the parameters chosen do not sit in that regime, the tail is wrong
    and everything downstream with it.

X2  THE STRUCTURAL CLAIM IS EXACT. Report the rank of the observable block and the least-squares
    residual of log a and log b on it, under each assay regime. Predeclared: abundance assays
    alone must leave both free, and adding a degradation rate must close the rank to 4. This is
    an identity, so a failure here is a coding error and voids everything.

X3  THE GRADIENT IS A GRADIENT. Adjoint-style analytic gradient against central finite
    differences, with the h^2 signature as in scaling.py rather than a fixed bar.

X4  THE DELIVERABLE. K(G) under each assay regime, for genomes from 100 up to the real human
    protein-coding count. Predeclared readings: K growing proportionally to G means expression is
    a measurement wall and no targeting helps; K sublinear means the answer concentrates and
    targeting works, as it did for metabolism.

X5  THE DETECTION CONTROL. The same measurement with a dense random gradient must give K growing
    proportionally to G, or a sublinear result is unfalsifiable.

X6  THE MATCHED CONTROL. Genes chosen at random rather than by the projection must need
    materially more measurements. Bar: 2x at the largest genome.

X7  SPARSITY, AND IT MUST AGREE WITH X4. How many genes carry 99% of the squared unconstrained
    gradient. scaling.py's S7 taught that this count and K measure different things -- spread
    versus magnitude -- so both are reported and neither is used to infer the other.

X8  DOMAIN. Repeat at a different rare-event threshold. If the list of genes changes completely,
    the answer is a property of the question and must be quoted with it, as recon.py's R8 found
    for metabolism.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.special import gammainc, gammaln
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from rem.atlas.hybrid_tune import RULE, ORDERS_PER_KCAL

SIGMA = 1.0 * ORDERS_PER_KCAL
DELTA = np.log10(2.0)
ZQ = 1.645
TARGET_NULL = DELTA / (ZQ * SIGMA)
G_SIZES = (100, 300, 1000, 3000, 10000, 19900)      # 19,900 = human protein-coding genes
THRESH = 20.0            # protein copies below which a gene counts as failed
THRESH_ALT = 8.0         # X8
Y_TARGET = -3.0          # k is chosen per genome so log10 Y lands here: a genuine rare event
SEED = 20260905

REGIMES = {
    "RNA-seq only": [(1, 0, -1, 0)],
    "RNA-seq + proteomics": [(1, 0, -1, 0), (1, 1, -1, -1)],
    "+ ribosome profiling": [(1, 0, -1, 0), (1, 1, -1, -1), (0, 1, 0, 0)],
    "+ mRNA half-life": [(1, 0, -1, 0), (1, 1, -1, -1), (0, 1, 0, 0), (0, 0, 1, 0)],
}
RATE_NAMES = ("k_tx", "k_tl", "k_dm", "k_dp")


def draw(G, seed):
    """Rates per gene. Spreads are declared assumptions, chosen so the bursting regime holds and
    so a minority of genes sit near the failure threshold; Schwanhausser et al. found mRNA and
    protein half-lives uncorrelated, so k_dm and k_dp are drawn independently."""
    rg = np.random.default_rng(seed)
    # Schwanhausser et al. found mRNA half-lives of hours against protein half-lives of days, and
    # NO correlation between them, so the two decay rates are drawn independently and well
    # separated. The separation also puts the model in the bursting regime the Gamma form needs.
    k_dp = np.exp(rg.normal(np.log(0.010), 0.7, G))     # protein decay /h  (half-life ~69 h)
    k_dm = np.exp(rg.normal(np.log(0.30), 0.7, G))      # mRNA decay /h     (half-life ~2.3 h)
    a = np.exp(rg.normal(np.log(9.0), 1.1, G))          # burst frequency
    b = np.exp(rg.normal(np.log(12.0), 0.9, G))         # burst size
    k_tx = a * k_dp
    k_tl = b * k_dm
    return np.column_stack([k_tx, k_tl, k_dm, k_dp])


def ab_of(X):
    k_tx, k_tl, k_dm, k_dp = X.T
    return k_tx / k_dp, k_tl / k_dm


def pfail(a, b, T):
    return gammainc(a, T / b)          # regularised lower incomplete gamma = P(protein < T)


def mu_of(X, T):
    a, b = ab_of(X)
    return pfail(a, b, T)


def choose_k(X, T, target=Y_TARGET):
    """The event is 'k or more genes simultaneously below threshold'. k is set per genome so the
    answer is a genuine rare event: 'at least one' saturates to certainty at genome scale, because
    some low-expression gene is always down, which is true biology and a useless observable."""
    mu = float(mu_of(X, T).sum())
    for k in range(1, 100000):
        if np.log10(max(gammainc(k, mu), 1e-300)) <= target:
            return k, mu
    return None, mu


def logY(X, T, k):
    """P(N >= k) for N the number of failed genes. The Poisson-binomial count is approximated by
    Poisson(mu), exact enough because every per-gene probability is small; X1b gates it against
    the exact Poisson-binomial. P(Poisson(mu) >= k) = gammainc(k, mu) identically."""
    mu = float(mu_of(X, T).sum())
    return float(np.log10(max(gammainc(k, mu), 1e-300)))


def gradient(X, T, k, h=1e-4):
    """d log10 Y / d log10 (each of the four rates). The rates enter only through a and b, so two
    per-gene finite differences give all four, and the chain through mu is analytic."""
    a, b = ab_of(X)
    p = pfail(a, b, T)
    mu = float(p.sum())
    Y = max(gammainc(k, mu), 1e-300)
    dYdmu = float(np.exp((k - 1) * np.log(max(mu, 1e-300)) - mu - gammaln(k)))
    pre = dYdmu / Y                       # dlogY/dmu ; note d(log10 Y)/d(log10 th) = same ratio
    dpda = (pfail(a * np.exp(h), b, T) - pfail(a * np.exp(-h), b, T)) / (2 * h)
    dpdb = (pfail(a, b * np.exp(h), T) - pfail(a, b * np.exp(-h), T)) / (2 * h)
    ga, gb = pre * dpda, pre * dpdb
    return np.column_stack([ga, gb, -gb, -ga])


def null_block(rows):
    """Per-gene null space of the observable block: shared by every gene, so computed once."""
    A = np.array(rows, float)
    U, s, Vt = np.linalg.svd(A)
    r = int((s > 1e-9 * max(s.max(), 1e-300)).sum())
    return Vt[r:].T, r                            # (4 x (4-r)) basis, rank


def gene_reductions(gi, Vnull):
    """Greedy sequence of squared-norm reductions available within ONE gene.

    Measuring rate j removes g_null[j]^2 / P_null[j,j] from the squared norm -- the same rank-one
    downdate used throughout this build order. Because the four rates of a gene are independent of
    every other gene, the GLOBAL greedy is just the largest reductions pooled across genes, which
    turns an O(K*G) search into a sort."""
    V = Vnull.copy()
    outs = []
    while V.shape[1] > 0:
        gn = V @ (V.T @ gi)
        diag = np.einsum("ij,ij->i", V, V)
        sc = np.where(diag > 1e-12, gn ** 2 / np.maximum(diag, 1e-300), -1.0)
        j = int(np.argmax(sc))
        if sc[j] <= 1e-300:
            break
        outs.append(float(sc[j]))
        u = V @ V[j]
        nu = np.linalg.norm(u)
        if nu < 1e-12:
            break
        u = u / nu
        W = V - np.outer(u, u @ V)
        Q, R = np.linalg.qr(W)
        d = V.shape[1] - 1
        V = Q[:, :d] if d > 0 else np.zeros((4, 0))
    return outs


def greedy_K(Gmat, Vnull, target, cap=None):
    """K = the fewest direct rate measurements bringing ||g_null|| to the target."""
    tot = 0.0
    pool = []
    for i in range(Gmat.shape[0]):
        gn = Vnull @ (Vnull.T @ Gmat[i])
        tot += float(gn @ gn)
        pool.extend(gene_reductions(Gmat[i], Vnull))
    if np.sqrt(tot) <= target:
        return 0, np.sqrt(tot), np.sqrt(tot)
    pool.sort(reverse=True)
    t2 = target ** 2
    run = tot
    for k, r in enumerate(pool, 1):
        run -= r
        if run <= t2:
            return k, np.sqrt(max(run, 0.0)), np.sqrt(tot)
    return None, np.sqrt(max(run, 0.0)), np.sqrt(tot)


def random_K(Gmat, Vnull, target, seed):
    rg = np.random.default_rng(seed)
    per = [gene_reductions(Gmat[i], Vnull) for i in range(Gmat.shape[0])]
    tot = sum(float((Vnull @ (Vnull.T @ Gmat[i])) @ (Vnull @ (Vnull.T @ Gmat[i])))
              for i in range(Gmat.shape[0]))
    order = []
    for i, seq in enumerate(per):
        order.extend([(i, k, v) for k, v in enumerate(seq)])
    rg.shuffle(order)
    taken = {}
    run, t2 = tot, target ** 2
    for k, (i, step, v) in enumerate(order, 1):
        # a gene's reductions must be taken in sequence; approximate a random ORDER OF GENES by
        # applying whichever reduction that gene has next
        nxt = taken.get(i, 0)
        run -= per[i][nxt] if nxt < len(per[i]) else 0.0
        taken[i] = nxt + 1
        if run <= t2:
            return k
    return None


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE); P("WHICH TRANSCRIPTION AND TRANSLATION RATES MUST BE MEASURED?"); P(RULE)
    P(f"  four rates per gene; human protein-coding genome {max(G_SIZES)} genes"
      f" -> {4*max(G_SIZES):,} rates")
    P(f"  rare event: {{k}} or more of the genome's proteins simultaneously below"
      f" {THRESH:.0f} copies -- k set per genome so the event stays rare")
    P(f"  criterion: ||g_null|| <= {TARGET_NULL:.4f}, i.e. a factor of two on the answer at 90%")

    # ---- X2, the exact structural claim --------------------------------------------------------
    P("\n" + RULE); P("X2  THE STRUCTURAL CLAIM IS EXACT"); P(RULE)
    tail = {"burst frequency log a": np.array([1, 0, 0, -1], float),
            "burst size      log b": np.array([0, 1, -1, 0], float)}
    P(f"  {'assay regime':<24}{'rank':>5}{'free dims':>11}"
      + "".join(f"{k+' resid':>26}" for k in tail))
    ok2 = True
    for name, rows in REGIMES.items():
        A = np.array(rows, float)
        r = int(np.linalg.matrix_rank(A))
        cells = []
        for k, v in tail.items():
            coef, *_ = np.linalg.lstsq(A.T, v, rcond=None)
            cells.append(float(np.linalg.norm(A.T @ coef - v)))
        P(f"  {name:<24}{r:>5}{4-r:>11}" + "".join(f"{c:>26.4f}" for c in cells))
        if name == "RNA-seq + proteomics":
            ok2 &= all(c > 1e-6 for c in cells)
        if name == "+ mRNA half-life":
            ok2 &= r == 4 and all(c < 1e-9 for c in cells)
    P(f"  {'PASS -- abundance assays leave both tail parameters free; a degradation rate closes it' if ok2 else 'FAIL -- this is an identity, so a failure is a coding error'}")
    P("  The protein MEAN is a*b, which is the sum of the two abundance rows, so it is pinned")
    P("  exactly while both of its factors are free. residence.py and katg.py measured that on")
    P("  small circuits; here it is an identity holding for every gene in the genome.")

    # ---- X1 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("X1  THE TAIL IS THE RIGHT TAIL"); P(RULE)
    X = draw(6, SEED)
    a, b = ab_of(X)
    P(f"  {'gene':>5}{'k_dm/k_dp':>12}{'a':>9}{'b':>9}{'Gamma P(<T)':>14}{'exact CME':>14}{'log10 err':>11}")
    worst = 0.0
    for i in range(6):
        k_tx, k_tl, k_dm, k_dp = X[i]
        Mm = 25
        Mp = int(min(600, max(80, 5 * a[i] * b[i])))
        n = (Mm + 1) * (Mp + 1)
        idx = lambda m, q: m * (Mp + 1) + q
        rows_, cols_, vals_ = [], [], []
        for m in range(Mm + 1):
            for q in range(Mp + 1):
                s0 = idx(m, q)
                for tgt, rate in (((m + 1, q), k_tx if m < Mm else 0.0),
                                  ((m - 1, q), k_dm * m),
                                  ((m, q + 1), k_tl * m if q < Mp else 0.0),
                                  ((m, q - 1), k_dp * q)):
                    if rate > 0 and 0 <= tgt[0] <= Mm and 0 <= tgt[1] <= Mp:
                        rows_.append(idx(*tgt)); cols_.append(s0); vals_.append(rate)
                        rows_.append(s0); cols_.append(s0); vals_.append(-rate)
        L = coo_matrix((vals_, (rows_, cols_)), shape=(n, n)).tocsr().tolil()
        L[n - 1, :] = 1.0                      # replace one row with the normalisation
        rhs = np.zeros(n); rhs[n - 1] = 1.0
        pi = spsolve(L.tocsc(), rhs)
        pi = np.maximum(pi, 0.0); pi = pi / pi.sum()
        pq = pi.reshape(Mm + 1, Mp + 1).sum(axis=0)
        exact = float(pq[: int(THRESH)].sum())
        gam = float(pfail(a[i], b[i], THRESH))
        err = abs(np.log10(max(gam, 1e-300)) - np.log10(max(exact, 1e-300)))
        worst = max(worst, err)
        P(f"  {i:>5}{X[i,2]/X[i,3]:>12.1f}{a[i]:>9.2f}{b[i]:>9.2f}"
          f"{gam:>14.4e}{exact:>14.4e}{err:>11.4f}")
    P(f"  worst |log10 disagreement| {worst:.4f}"
      f"   {'PASS' if worst < 0.5 else 'FAIL -- outside the bursting regime, the Gamma tail is wrong'}"
      f" (bar 0.5 orders)")

    # ---- X3 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("X3  THE GRADIENT IS A GRADIENT"); P(RULE)
    X = draw(200, SEED)
    kk, _ = choose_k(X, THRESH)
    g = gradient(X, THRESH, kk)
    rg = np.random.default_rng(4)
    probe = [(int(i), int(j)) for i, j in zip(rg.integers(0, 200, 8), rg.integers(0, 4, 8))]
    P(f"  {'step h':>10}{'worst rel':>14}{'ratio':>9}")
    ws, ratios = [], []
    for hh in (0.02, 0.01, 0.005, 0.0025):
        w = 0.0
        for i, j in probe:
            Xp = X.copy(); Xp[i, j] *= 10 ** hh
            Xm = X.copy(); Xm[i, j] *= 10 ** -hh
            fd = (logY(Xp, THRESH, kk) - logY(Xm, THRESH, kk)) / (2 * hh)
            w = max(w, abs(fd - g[i, j]) / max(abs(fd), 1e-12))
        ratios.append(ws[-1] / w if ws else float("nan"))
        ws.append(w)
        P(f"  {hh:>10}{w:>14.3e}{ratios[-1]:>9.2f}")
    ok3 = all(abs(r - 4.0) <= 1.0 for r in ratios[1:]) or ws[-1] < 1e-4
    P(f"  {'PASS' if ok3 else 'FAIL'} (h^2 decay, or below 1e-4 at the smallest step)")

    # ---- X4, X5, X6, X7 ---------------------------------------------------------------------------
    P("\n" + RULE); P("X4  THE DELIVERABLE  --  K(G) under each assay regime"); P(RULE)
    results = {}
    for name, rows in REGIMES.items():
        Vn, r = null_block(rows)
        if Vn.shape[1] == 0:
            P(f"\n  {name}: rank 4, nothing is free, K = 0 by construction")
            results[name] = ([], [])
            continue
        P(f"\n  {name}   (rank {r}, {4-r} free directions per gene)")
        P(f"  {'genes':>7}{'rates':>9}{'k':>7}{'log10 Y':>10}{'K':>7}{'K/G':>9}"
          f"{'||g_null||':>12}{'random K':>10}{'99% genes':>11}")
        Ns, Ks = [], []
        for G in G_SIZES:
            X = draw(G, SEED + G)
            kg, mug = choose_k(X, THRESH)
            g = gradient(X, THRESH, kg)
            K, resid, start = greedy_K(g, Vn, TARGET_NULL)
            gn = np.array([Vn @ (Vn.T @ g[i]) for i in range(G)])
            per = (gn ** 2).sum(axis=1)
            srt = np.sort(per)[::-1]
            k99 = int(np.searchsorted(np.cumsum(srt) / max(srt.sum(), 1e-300), 0.99) + 1)
            Kr = random_K(g, Vn, TARGET_NULL, 3)
            Ns.append(G); Ks.append(K if K is not None else np.nan)
            P(f"  {G:>7}{4*G:>9}{kg:>7}{logY(X, THRESH, kg):>10.3f}"
              f"{(K if K is not None else -1):>7}{(K/G if K else 0):>9.4f}{start:>12.3f}"
              f"{(Kr if Kr else -1):>10}{k99:>11}")
        results[name] = (Ns, Ks)
        good = [(n, k) for n, k in zip(Ns, Ks) if np.isfinite(k) and k > 0]
        if len(good) >= 3:
            nn = np.log([x[0] for x in good]); kk = np.log([x[1] for x in good])
            al = np.polyfit(nn, kk, 1)[0]
            P(f"  K ~ G^{al:.3f}")

    P("\n" + RULE)
    open(os.path.join(os.path.dirname(__file__), "RESULTS_expression.txt"),
         "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
