"""MEASUREMENT POWER -- the ceiling, derived from the raw cells, not downloaded and not asserted.

WHY THIS AND NOT ANOTHER FEATURE. This project's dataset map lists measurement power as item 16 and says two
things about it that are easy to get wrong. It must be DERIVED from the raw experiment rather than fetched,
and it must model UNCERTAINTY rather than sit in the feature list as though it were biology. The second is
the important one. A column called `n_cells` in a feature table is a covariate a model will happily fit; a
reliability coefficient is a statement about how much of the target is signal at all, and it bounds every
score anyone will ever report on this dataset.

THE DEBT IT PAYS. This project already found that perturbation-effect MAGNITUDE is confounded with detection
power (+0.0881 on a bounded task, larger than any biological block it tested). That was read as "power is a
confound, match on it". It is worse than that. If the measurement is unreliable then apparent magnitude is
partly noise, and matching on cells-per-perturbation does not remove it -- it matches on a covariate while
the target itself stays noisy.

    the ladder stalls at R2 0.0151                  <- read as "the layers add nothing"
    the curated reg layer adds -0.0002 R2           <- read as "the layer is worthless"
    PPI 1.208x, co-dependency 1.070x, reg 1.021x    <- read as "1.02 is not interesting"
    reg_perturb transfers at 67.2% sign agreement   <- read as "67% out of a possible 100%"

Not one of those can be read without knowing what fraction of the measured quantity is real.

THE TEST. Split each perturbation's cells into two halves, compute the log fold change independently in
each, and correlate. Two halves of one experiment share the biology exactly and share nothing else, so their
correlation IS the reliability of a half-sized measurement, and Spearman-Brown converts it to the
reliability of the full one. Reliability is by construction the maximum R2 any predictor can reach against
the observed values: a predictor that recovered the true signal perfectly would still be scored against a
noisy target and would land at exactly R2 = reliability.

THE CONTROL IS SPLIT TOO. lfc = expr(perturbed) - expr(non-targeting). Comparing both halves against the
SAME control estimate puts the control's own sampling noise into both halves as a shared term and inflates
their correlation. So the 11,485 non-targeting cells are split alongside the perturbed ones and half A is
compared only against control half A. The shared-control variant is computed as well and the gap reported,
because a design choice that moves the headline should be visible rather than argued.

TWO SPLITS, BECAUSE ONE OF THEM IS OPTIMISTIC. A random split puts cells from all 56 gemgroups into both
halves, so anything shared across a batch cancels and never counts as noise -- and batch matters here, median
UMI per cell runs from 6,122 to 36,436 across gemgroups. Reliability is therefore measured twice: RANDOM
(balanced within perturbation; optimistic, batch is shared) and BATCH-DISJOINT (the 56 gemgroups
partitioned; stricter, batch counts as noise).

THE SHUFFLED CONTROL IS NOT DECORATION. Reliability is reported for three quantities and one of them is a
trap. Pooled over all (perturbation, gene) pairs, RAW lfc contains per-gene mean effects -- some genes move
under almost any perturbation -- and those are highly reliable and trivially predictable. Correlating half A
against half B with the PERTURBATION LABELS SHUFFLED leaves the gene main effects intact and destroys
everything perturbation-specific, so the shuffled number says exactly how much of each ceiling is the part a
gene-responsiveness baseline already owns. It is run on every draw, not once.

WHAT WOULD FALSIFY THE FRAMING. Reliability near 1 would mean the ceiling is irrelevant and every null this
project measured stands exactly as reported. Reliability near 0 would mean the dataset carries no
perturbation-specific signal at all, which the already-measured RPE1 transfer (67.2% sign agreement against
52.2%) makes impossible -- so a near-zero answer would indict this estimate, not the data. The number has to
be consistent with those transfer results or one of the two is broken.

ALSO DERIVED, because the map asks and because reliability is unreadable without them: cells and total UMIs
per perturbation, per-gene mean expression and dropout, and a standard error on every log fold change. The SE
is EMPIRICAL, from the splits: two independent half-measurements give Var(lfcA - lfcB) = 4 Var(lfc_full)
exactly, control term included, so sigma = rms(lfcA - lfcB)/2 is unbiased with no distributional assumption.
It is cross-checked against a genuine cell-level bootstrap and against the Poisson delta-method SE, whose
ratio to the empirical one is the overdispersion this data carries over pure counting noise.

ONE PIPELINE, ASSERTED. Sum raw UMI per group, CPM, log2(1+x), subtract non-targeting -- identical to
colab/causal_reg.py, and asserted against its cached pseudobulk rather than assumed. One cell line, one
assay, one annotation, raw integer counts asserted. Two pipelines on one axis has burned this project before
and it is not going to happen on the axis that bounds everything else.

RPE1 ONLY, AND THAT IS A HARD LIMIT. Split-half needs CELLS. The K562 arm in this project (gwps.h5ad) is
already pseudobulked to 11,258 x 8,248 with the cells discarded, so no ceiling can be computed for it here.
Every number below is RPE1's. Rescaling a K562 result by an RPE1 ceiling would be exactly the kind of
cross-axis substitution this project keeps getting burned by, so it is not done.
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
RPE1 = SP / "rpe1.h5ad"                       # 247,914 cells x 8,749 genes, raw UMI
REGJ = SP / "entity_registry.json.gz"
PB_CACHE = SP / "rpe1_pseudobulk.npz"         # causal_reg.py's output -- the pipeline pin
LAYER = SP / "cell_measurement_power.json.gz"
MATS = SP / "measurement_power_matrices.npz"

SOURCE = "Replogle et al. 2022 genome-scale Perturb-seq, RPE1 CRISPRi arm (raw UMI counts)"
CTRL_NAME = "non-targeting"
MIN_CELLS = 25          # the analysis set, identical to causal_reg.py's MIN_CELLS
MIN_HALF = 8            # a half below this cannot support a pseudobulk
S_RAND = 20             # random splits   -- >= 20 draws, never read one draw as the result
S_BATCH = 20            # batch-disjoint splits
CHUNK = 20_000
BOOT_B = 200            # bootstrap resamples per checked perturbation
BOOT_PERTS = 50
THRS = (2.0, 3.0, 5.0)  # 5.0 is causal_reg.py's edge threshold, re-tested here within one cell line
SEED = 20260731


# ---------------------------------------------------------------- pipeline (pinned) ----
def expr_from_counts(S):
    """Raw summed UMI -> log2(1 + CPM). Identical to the pseudobulk in causal_reg.py."""
    tot = S.sum(1, keepdims=True)
    cpm = np.divide(S, np.maximum(tot, 1), where=True) * 1e6
    return np.log2(1.0 + cpm)


def robust_z(M):
    """Per-COLUMN robust z, identical to causal_reg.robust_z -- the quantity the edges are defined on."""
    med = np.nanmedian(M, axis=0)
    mad = np.nanmedian(np.abs(M - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-6, np.nan, mad)
    return (M - med) / mad


def rowcorr(A, B):
    """Pearson per row, NaN where a row is constant or too short. Rows = the axis correlated over."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    ok = np.isfinite(A) & np.isfinite(B)
    A = np.where(ok, A, 0.0)
    B = np.where(ok, B, 0.0)
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.where(ok, A - (A.sum(1) / np.maximum(n, 1))[:, None], 0.0)
        B = np.where(ok, B - (B.sum(1) / np.maximum(n, 1))[:, None], 0.0)
        den = np.sqrt((A * A).sum(1) * (B * B).sum(1))
        r = np.where((den > 1e-12) & (n >= 10), (A * B).sum(1) / np.where(den > 0, den, 1.0), np.nan)
    return r


def flatcorr(A, B):
    """Pearson over every finite element of two same-shaped arrays, without a giant transient copy."""
    ok = np.isfinite(A) & np.isfinite(B)
    a = A[ok].astype(np.float64)
    b = B[ok].astype(np.float64)
    if len(a) < 10:
        return float("nan")
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt(a.dot(a) * b.dot(b))
    return float(a.dot(b) / d) if d > 0 else float("nan")


def nanmean0(M):
    """Column means ignoring NaN, quietly.

    An all-NaN column is a quantity that could not be estimated AT ALL -- a gene with no variation across
    perturbations, or a perturbation whose halves never both cleared MIN_HALF. numpy warns on it; the warning
    is suppressed here only because the case is expected and is COUNTED and REPORTED instead. The value stays
    NaN and is written as null downstream, because absence here is UNKNOWN and must never become zero.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(M, axis=0)


def sb(r):
    """Spearman-Brown: reliability of a half-sized measurement -> reliability of the full one.

    Exact rather than approximate here: the full measurement uses twice the perturbed cells AND twice the
    control cells relative to a half, so it is a double-length test of the half in the strict sense.
    """
    return 2.0 * float(r) / (1.0 + float(r))


def sb_item(r):
    """Per-item version. A negative split-half r means no reliable variance was detected, not negative
    reliability, so it is reported as 0 rather than passed through a formula that would explode."""
    if r is None or not np.isfinite(r):
        return None
    return round(sb(r), 4) if r > 0 else 0.0


def selftest(report):
    """Recover a KNOWN reliability from synthetic data before trusting the estimator on real data.

    Built so the answer is known in advance: a per-gene main effect shared by every perturbation, a
    perturbation-specific signal of unit variance, and unit-variance independent noise in each half. A half
    measurement then has reliability Var(signal)/(Var(signal)+Var(noise)) = 0.5 exactly, and Spearman-Brown
    must return 2(0.5)/1.5 = 0.667 for the full-size measurement.

    The shuffled arm is the part that matters. Shuffling perturbation labels leaves the gene main effect
    untouched, so a quantity that still correlates after shuffling is measuring the main effect and not the
    perturbation. RAW lfc must fail that test loudly and the gene-mean-removed and robust-z quantities must
    collapse to zero -- which is exactly why the real ceiling is reported on the latter two.
    """
    rng = np.random.default_rng(0)
    P, G = 300, 400
    main = rng.normal(0, 2, G)
    spec = rng.normal(0, 1, (P, G))
    a = main + spec + rng.normal(0, 1.0, (P, G))
    b = main + spec + rng.normal(0, 1.0, (P, G))
    pm = rng.permutation(P)
    am, bm = np.nanmedian(a, 0), np.nanmedian(b, 0)
    rows = [("raw lfc (main effects left in)", flatcorr(a, b), flatcorr(a, b[pm])),
            ("gene mean removed", flatcorr(a - am, b - bm), flatcorr(a - am, (b - bm)[pm])),
            ("per-gene robust z", flatcorr(robust_z(a), robust_z(b)),
             flatcorr(robust_z(a), robust_z(b)[pm]))]
    report(f"\n  SELF-TEST -- recover a known reliability (true half-r = 0.500, true SB = 0.667)")
    report(f"    {'quantity':34s} {'half r':>9s} {'SB':>8s} {'shuffled':>10s}")
    for nm, r, rs in rows:
        report(f"    {nm:34s} {r:9.4f} {sb(r):8.4f} {rs:10.4f}")
    _nm, r_dev, rs_dev = rows[1]
    _nm2, r_raw, rs_raw = rows[0]
    assert abs(r_dev - 0.5) < 0.03, f"estimator does not recover a known reliability: {r_dev}"
    assert abs(sb(r_dev) - 2 / 3) < 0.04, f"Spearman-Brown wrong: {sb(r_dev)}"
    assert abs(rs_dev) < 0.05, f"shuffled control is not null: {rs_dev}"
    assert rs_raw > 0.5, "shuffled RAW should stay high -- the main-effect trap is not being demonstrated"
    report(f"    recovered {r_dev:.4f} against a true 0.500, SB {sb(r_dev):.4f} against a true 0.667, "
           f"shuffle null {rs_dev:+.4f}")
    report(f"    RAW stays at {rs_raw:.4f} under shuffling: {100*rs_raw/r_raw:.0f}% of its apparent "
           f"reliability is the gene main effect, not the perturbation. That is why the headline")
    report(f"    ceiling below is quoted on robust z, not on raw lfc.")


# ---------------------------------------------------------------- splits ----
def make_splits(codes, batch, P, rng):
    """(label, per-cell boolean 'is in half A'). Both kinds exactly reproducible from SEED."""
    out = []
    n = len(codes)
    for s in range(S_RAND):
        # balanced WITHIN perturbation: order each perturbation's cells at random and alternate. An
        # unbalanced split would give the halves different variances and bias the correlation down.
        r = rng.random(n)
        order = np.lexsort((r, codes))
        rank = np.empty(n, dtype=np.int64)
        srt = codes[order]
        start = np.searchsorted(srt, np.arange(P))
        rank[order] = np.arange(n) - start[srt]
        out.append((f"rand{s}", (rank % 2) == 0))
    ub = np.unique(batch)
    for s in range(S_BATCH):
        perm = rng.permutation(ub)
        out.append((f"batch{s}", np.isin(batch, perm[: len(ub) // 2])))
    return out


# ---------------------------------------------------------------- streaming pass ----
def stream(report, codes, splits, P, G, boot_set, extras):
    """One pass over the 8.7 GB dense matrix. Group sums by sparse one-hot matmul (10x np.add.at).

    Per-chunk group sums are exact in float32 (a chunk sum never approaches 2^24), so accumulating them into
    float64 totals is exact -- which is what lets the result be asserted equal to the cached pseudobulk.
    Called once per split family so that only one family's accumulators are ever resident.
    """
    import h5py
    import scipy.sparse as sp
    S = len(splits)
    TOT = np.zeros((P, G), dtype=np.float64) if extras else None
    HALF = [np.zeros((P, G), dtype=np.float32) for _ in range(S)]
    C = np.zeros(P, dtype=np.int64)
    CA = np.zeros((S, P), dtype=np.int64)
    NZ = np.zeros(G, dtype=np.int64)          # cells with a nonzero count, per gene, over ALL cells
    NZC = np.zeros(G, dtype=np.int64)         # the same restricted to non-targeting cells
    buf = {p: [] for p in (boot_set if extras else [])}
    ctrl = P - 1
    with h5py.File(RPE1, "r") as f:
        X = f["X"]
        n = X.shape[0]
        for i0 in range(0, n, CHUNK):
            i1 = min(i0 + CHUNK, n)
            blk = X[i0:i1]
            if i0 == 0 and extras:
                assert np.all(blk == np.rint(blk)), "X is not raw integer UMI counts"
                assert blk.min() >= 0
            cd = codes[i0:i1]
            m = len(cd)
            if extras:
                oh = sp.csr_matrix((np.ones(m, np.float32), (cd, np.arange(m))), shape=(P, m))
                TOT += (oh @ blk).astype(np.float64)
                NZ += (blk > 0).sum(0)
                cm = cd == ctrl
                if cm.any():
                    NZC += (blk[cm] > 0).sum(0)
                for p in boot_set:
                    sel = np.where(cd == p)[0]
                    if len(sel):
                        buf[p].append(blk[sel].copy())
            C += np.bincount(cd, minlength=P)
            for k, (_lab, aflag) in enumerate(splits):
                idx = np.where(aflag[i0:i1])[0]
                if not len(idx):
                    continue
                # The one-hot selects half A's cells by COLUMN POSITION, so `blk` is never sliced. A
                # fancy-index copy here costs 350 MB per split per chunk and dominated everything else.
                oa = sp.csr_matrix((np.ones(len(idx), np.float32), (cd[idx], idx)), shape=(P, m))
                HALF[k] += oa @ blk
                CA[k] += np.bincount(cd[idx], minlength=P)
            if (i0 // CHUNK) % 4 == 0:
                report(f"      streamed {i1:,}/{n:,} cells")
    buf = {p: (np.concatenate(v) if v else np.zeros((0, G), np.float32)) for p, v in buf.items()}
    return TOT, HALF, C, CA, NZ, NZC, buf


# ---------------------------------------------------------------- main ----
def main():
    from scipy import stats
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("MEASUREMENT POWER -- split-half reliability of Replogle RPE1, and the R2 ceiling it sets")
    report("=" * 100)
    if not RPE1.exists():
        raise SystemExit(f"absent: {RPE1}")

    selftest(report)

    from entity_registry import load as load_reg
    REG = load_reg()

    import h5py
    with h5py.File(RPE1, "r") as f:
        cats = np.array([c.decode() if isinstance(c, bytes) else str(c)
                         for c in f["obs"]["gene"]["categories"][:]])
        codes = f["obs"]["gene"]["codes"][:].astype(np.int64)
        batch = f["obs"]["batch"][:].astype(np.int64)
        ncounts = f["obs"]["ncounts"][:]
        ngenes = f["obs"]["ngenes"][:]
        pmito = f["obs"]["percent_mito"][:]
        gens = np.array([(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
                         for s in f["var"]["ensembl_id"][:]])
        gsym = np.array([s.decode() if isinstance(s, bytes) else str(s)
                         for s in f["var"]["gene_name"][:]])
        # ---- PROVENANCE ASSERTIONS: one line, one assay, one annotation, raw counts ----
        cl = [c.decode() if isinstance(c, bytes) else str(c)
              for c in f["obs"]["cell_line"]["categories"][:]]
        pt = [c.decode() if isinstance(c, bytes) else str(c)
              for c in f["obs"]["perturbation_type"]["categories"][:]]
        assert cl == ["RPE1"] and pt == ["CRISPR"], (cl, pt)
        assert np.array_equal(codes, f["obs"]["perturbation"]["codes"][:].astype(np.int64))
        assert f["var"]["in_matrix"][:].all()
        nC, G = f["X"].shape
    P = len(cats)
    ctrl = int(np.where(cats == CTRL_NAME)[0][0])
    assert ctrl == P - 1, "the control is assumed to be the last category by the accumulators"
    report(f"  {SOURCE}")
    report(f"  {nC:,} cells x {G:,} genes, {P:,} perturbation categories, control = '{CTRL_NAME}'")
    report(f"  one cell line {cl}, one assay {pt}, {len(np.unique(batch))} gemgroups, "
           f"var.in_matrix all True, X asserted raw integer UMI")

    # ---- registry joins, declared rates, raises below them ----
    _g, st_g = REG.join("gene", "ensembl", list(gens), 0.80, "RPE1 measured genes (ensembl)")
    _p, st_p = REG.join("gene", "symbol", [c for c in cats if c != CTRL_NAME], 0.80,
                        "RPE1 perturbations (symbol)")
    report(f"  registry: genes {st_g['joined']:,}/{st_g['of']:,} ({st_g['rate']:.1%}) via ensembl; "
           f"perturbations {st_p['joined']:,}/{st_p['of']:,} ({st_p['rate']:.1%}) via symbol")
    # The cell object keys ppm/abund/coexpr/codep/gene2cplx by the gene's INDEX, not by symbol -- the
    # documented failure this registry exists to prevent. Resolving each measured gene to its positional
    # key here means a downstream user of this layer never has to guess.
    ci = json.load(gzip.open(REGJ, "rt"))["cell_index"]
    eid2idx = {}
    for k, e in ci.items():
        eid2idx.setdefault(e, int(k))
    gene_cell_index = {}
    for e in gens:
        eid = REG.resolve("gene", "ensembl", e)
        if eid is not None and eid in eid2idx:
            gene_cell_index[str(e)] = eid2idx[eid]
    report(f"  cell-object positional keys resolved for {len(gene_cell_index):,}/{G:,} measured genes "
           f"(gene INDEX, not symbol -- the documented silent-join failure)")
    del ci, eid2idx

    rng = np.random.default_rng(SEED)
    allsplits = make_splits(codes, batch, P, rng)
    sp_rand = allsplits[:S_RAND]
    sp_batch = allsplits[S_RAND:]
    report(f"  splits: {S_RAND} random (balanced within perturbation) + {S_BATCH} batch-disjoint "
           f"(gemgroups partitioned) = {len(allsplits)} independent draws, seed {SEED}")

    pre = np.bincount(codes, minlength=P)
    elig = np.where((pre >= MIN_CELLS) & (np.arange(P) != ctrl))[0]
    qs = np.quantile(pre[elig], np.linspace(0, 1, BOOT_PERTS))
    boot_set = sorted({int(elig[np.argmin(np.abs(pre[elig] - q))]) for q in qs})
    report(f"  bootstrap cross-check on {len(boot_set)} perturbations stratified across the "
           f"cells-per-perturbation range ({pre[boot_set].min()}-{pre[boot_set].max()} cells)")

    # ================================================================ pass 1: random splits ----
    report(f"\n  STREAMING PASS 1/2, random splits (X is {nC*G*4/1e9:.1f} GB dense on disk)")
    TOT, HALF_R, C, CA_R, NZ, NZC, buf = stream(report, codes, sp_rand, P, G, boot_set, True)
    assert np.array_equal(C, pre)

    E = expr_from_counts(TOT)
    LFC = (E - E[ctrl]).astype(np.float32)
    keep = (C >= MIN_CELLS) & (np.arange(P) != ctrl)
    A_idx = np.where(keep)[0]
    gmean_cpm = 1e6 * TOT.sum(0) / TOT.sum()
    dropout = 1.0 - NZ / nC
    dropout_c = 1.0 - NZC / max(int(C[ctrl]), 1)

    if PB_CACHE.exists():
        z = np.load(PB_CACHE, allow_pickle=True)
        cp = np.array([str(x) for x in z["perts"]])
        pos = {s: i for i, s in enumerate(cp)}
        rows = [pos[cats[p]] for p in A_idx if cats[p] in pos]
        mine = np.array([p for p in A_idx if cats[p] in pos])
        d = np.abs(z["lfc"][rows] - LFC[mine])
        report(f"\n  PIPELINE PIN: reproduced causal_reg.py's cached RPE1 pseudobulk on "
               f"{len(rows):,}/{len(cp):,} perturbations, max |diff| {d.max():.2e}")
        assert d.max() < 1e-3, "pipeline drift against the cached pseudobulk"
        assert sorted(cp.tolist()) == sorted(cats[A_idx].tolist()), "analysis set differs from the cache"
        del z
    else:
        report("\n  PIPELINE PIN: cache absent -- cannot assert against causal_reg.py")

    # ---------------------------------------------------------- the split-half machinery ----
    report(f"\n  SPLIT-HALF over {S_RAND}+{S_BATCH} draws. The control cells are split alongside the")
    report(f"  perturbed ones, so half A is compared only against control half A: a shared control")
    report(f"  estimate would put its own sampling noise into both halves as a common term.")
    perm_rng = np.random.default_rng(SEED + 1)

    def process(splits, HALF, CA, want_edges):
        """Reduce one split family to its statistics, freeing each accumulator as it is consumed."""
        rp, rg, D2 = [], [], np.zeros((P, G), dtype=np.float64)
        pool = {k: [] for k in ("raw", "dev", "z", "shared", "raw_shuf", "dev_shuf", "z_shuf")}
        rp_shuf, ed = [], []
        colc = {t: np.zeros(G, dtype=np.int64) for t in THRS}
        for k in range(len(splits)):
            HA = HALF[k]
            HB = (TOT - HA).astype(np.float32)
            eA, eB = expr_from_counts(HA), expr_from_counts(HB)
            lA = (eA - eA[ctrl]).astype(np.float32)
            lB = (eB - eB[ctrl]).astype(np.float32)
            lS = (eB - eA[ctrl]).astype(np.float32)      # shared-control variant of half B
            del eA, eB
            nA, nB = CA[k], C - CA[k]
            rows = np.where(keep & (nA >= MIN_HALF) & (nB >= MIN_HALF))[0]
            v = np.full(P, np.nan)
            v[rows] = rowcorr(lA[rows], lB[rows])
            rp.append(v)
            rg.append(rowcorr(lA[rows].T, lB[rows].T))
            # per-perturbation SHUFFLE: half A of p against half B of a different perturbation
            pm = perm_rng.permutation(len(rows))
            vs = np.full(P, np.nan)
            vs[rows] = rowcorr(lA[rows], lB[rows][pm])
            rp_shuf.append(vs)

            a, b, s = lA[rows], lB[rows], lS[rows]
            am, bm, sm = (np.nanmedian(a, 0), np.nanmedian(b, 0), np.nanmedian(s, 0))
            za, zb = robust_z(a), robust_z(b)
            pool["raw"].append(flatcorr(a, b))
            pool["dev"].append(flatcorr(a - am, b - bm))
            pool["z"].append(flatcorr(za, zb))
            pool["shared"].append(flatcorr(a - am, s - sm))
            # SHUFFLED PERTURBATION LABELS: gene main effects survive, everything specific dies
            pool["raw_shuf"].append(flatcorr(a, b[pm]))
            pool["dev_shuf"].append(flatcorr(a - am, (b - bm)[pm]))
            pool["z_shuf"].append(flatcorr(za, zb[pm]))

            D2 += (lA.astype(np.float64) - lB.astype(np.float64)) ** 2
            if want_edges:
                fz = np.isfinite(za) & np.isfinite(zb)
                for t in THRS:
                    sel = fz & (np.abs(za) >= t)
                    if sel.sum() > 50:
                        cols = np.where(sel)[1]
                        colc[t] += np.bincount(cols, minlength=G)
                        ed.append((t, int(sel.sum()),
                                   float((np.sign(za[sel]) == np.sign(zb[sel])).mean()),
                                   float(np.abs(zb[sel]).mean()),
                                   float((np.abs(zb[sel]) >= 2).mean()),
                                   float(np.mean(gmean_cpm[cols]))))
            del lA, lB, lS, a, b, s, za, zb
            HALF[k] = None
            if k % 5 == 0:
                report(f"      split {k+1}/{len(splits)} ({splits[k][0]})")
        return (np.vstack(rp), np.vstack(rg), np.vstack(rp_shuf), pool, D2, ed, colc)

    RP, RG, RPS, POOL, D2, edge_rows, colc = process(sp_rand, HALF_R, CA_R, True)
    del HALF_R
    r_pert, r_gene = nanmean0(RP), nanmean0(RG)

    # ================================================================ pass 2: batch splits ----
    report(f"\n  STREAMING PASS 2/2, batch-disjoint splits")
    _t, HALF_B, _c, CA_B, _z, _zc, _b = stream(report, codes, sp_batch, P, G, [], False)
    RPB, RGB, _RPBS, POOLB, D2B, _e, _c2 = process(sp_batch, HALF_B, CA_B, False)
    del HALF_B
    r_pert_b, r_gene_b = nanmean0(RPB), nanmean0(RGB)

    # ---- coverage of the reliability estimate itself: absence is UNKNOWN, never zero ----
    n_g_nan = int((~np.isfinite(r_gene)).sum())
    n_p_nan = int((~np.isfinite(r_pert[A_idx])).sum())
    n_small = int(((C < MIN_CELLS) & (np.arange(P) != ctrl)).sum())
    report(f"\n  COVERAGE OF THE ESTIMATE  (what has no reliability, and why)")
    report(f"    perturbations: {len(A_idx):,} in the analysis set, {n_small:,} excluded below "
           f"{MIN_CELLS} cells, {n_p_nan:,} in-set but not estimable")
    report(f"    genes: {G - n_g_nan:,}/{G:,} estimable, {n_g_nan:,} with no variation to correlate")
    report(f"    these are written as null, never as 0 -- a gene with no reliability estimate is unknown,")
    report(f"    and treating it as unreliable would be as wrong as treating it as perfect.")

    # ---------------------------------------------------------- the headline ----
    def blk(vals, name):
        v = np.array(vals, dtype=float)
        v = v[np.isfinite(v)]
        return {"name": name, "half_r": float(v.mean()), "sd": float(v.std()),
                "reliability_full_sb": sb(v.mean()), "n_draws": int(len(v))}

    pr, pd_, pz = (blk(POOL["raw"], "pooled raw lfc"),
                   blk(POOL["dev"], "pooled lfc, gene mean removed"),
                   blk(POOL["z"], "pooled per-gene robust z"))
    prs, pds, pzs = (blk(POOL["raw_shuf"], "  same, perturbation labels SHUFFLED"),
                     blk(POOL["dev_shuf"], "  same, perturbation labels SHUFFLED"),
                     blk(POOL["z_shuf"], "  same, perturbation labels SHUFFLED"))
    ps = blk(POOL["shared"], "pooled lfc, SHARED control (variant)")
    report(f"\n  OVERALL CEILING -- pooled over {len(A_idx):,} perturbations x {G:,} genes, "
           f"{S_RAND} draws")
    report(f"    {'quantity':44s} {'half r':>9s} {'sd':>7s} {'reliability (SB)':>17s} "
           f"{'= R2 ceiling':>13s}")
    for b_ in (pr, prs, pd_, pds, pz, pzs, ps):
        report(f"    {b_['name']:44s} {b_['half_r']:9.4f} {b_['sd']:7.4f} "
               f"{b_['reliability_full_sb']:17.4f} {b_['reliability_full_sb']:13.4f}")
    fs = {k: float(np.mean(np.array(POOL[k]) > np.array(POOL[k + "_shuf"])))
          for k in ("raw", "dev", "z")}
    report(f"    fraction of the {S_RAND} draws beating their own shuffled control: "
           f"raw {fs['raw']:.2f}, gene-mean-removed {fs['dev']:.2f}, robust z {fs['z']:.2f}")
    report(f"    -> the RAW number is {100*prs['half_r']/max(pr['half_r'],1e-9):.0f}% reproduced by "
           f"SHUFFLED labels: it is mostly per-gene main effects, which a responsiveness baseline")
    report(f"       already owns. The gene-mean-removed and robust-z rows are the honest ceilings.")
    report(f"    control-sharing inflation: {ps['half_r'] - pd_['half_r']:+.4f} on the half-r "
           f"-- small, because the control carries {C[ctrl]:,} cells")

    rr = blk([np.nanmean(x) for x in RP], "per-perturbation, random split")
    rb = blk([np.nanmean(x) for x in RPB], "per-perturbation, batch-disjoint")
    rs_ = blk([np.nanmean(x) for x in RPS], "per-perturbation, SHUFFLED")
    report(f"\n  PER-PERTURBATION reliability (a perturbation's whole response profile, across genes)")
    report(f"    {'split kind':44s} {'half r':>9s} {'sd':>7s} {'reliability (SB)':>17s}")
    for b_ in (rr, rb, rs_):
        report(f"    {b_['name']:44s} {b_['half_r']:9.4f} {b_['sd']:7.4f} "
               f"{b_['reliability_full_sb']:17.4f}")
    thr95 = float(np.nanpercentile(RPS, 95))
    frac_sig = float(np.nanmean(r_pert[A_idx] > thr95))
    report(f"    over {len(A_idx):,} perturbations: p10 {np.nanpercentile(r_pert[A_idx],10):.3f}  "
           f"median {np.nanmedian(r_pert[A_idx]):.3f}  p90 {np.nanpercentile(r_pert[A_idx],90):.3f}  "
           f"negative in {100*np.nanmean(r_pert[A_idx]<0):.1f}%")
    report(f"    {100*frac_sig:.1f}% of perturbations exceed the 95th percentile of the shuffled "
           f"distribution ({thr95:.4f}); {100*float(np.nanmean(RP > 0)):.1f}% of all "
           f"{RP.shape[0]}x{len(A_idx):,} draw-perturbation cells are positive")

    # ---------------------------------------------------------- SE, and the decomposition ----
    SIG = np.sqrt(D2 / S_RAND) / 2.0                 # unbiased SE of the FULL-data lfc, per (pert, gene)
    SIGB = np.sqrt(D2B / S_BATCH) / 2.0
    N_umi = TOT.sum(1)
    cpm_all = 1e6 * TOT / np.maximum(N_umi[:, None], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        se_p = np.sqrt(cpm_all * 1e6 / np.maximum(N_umi[:, None], 1)) / (np.log(2) * (1.0 + cpm_all))
        se_c = np.sqrt(cpm_all[ctrl] * 1e6 / max(N_umi[ctrl], 1)) / (np.log(2) * (1.0 + cpm_all[ctrl]))
        se_pois = np.sqrt(se_p ** 2 + se_c ** 2)
    del cpm_all, se_p
    def rms(M):
        return np.sqrt(np.mean(np.asarray(M, dtype=np.float64) ** 2, axis=1))

    sig_p, sigb_p, pois_p = rms(SIG[A_idx]), rms(SIGB[A_idx]), rms(se_pois[A_idx])
    obs_rms = rms(LFC[A_idx])
    sig_true = np.sqrt(np.maximum(obs_rms ** 2 - sig_p ** 2, 0.0))
    noise_frac = 1.0 - (sig_true ** 2) / np.maximum(obs_rms ** 2, 1e-12)
    od = sig_p / np.maximum(pois_p, 1e-12)

    cell_umi = np.zeros(P)
    np.add.at(cell_umi, codes, ncounts)
    mgene = np.zeros(P)
    np.add.at(mgene, codes, ngenes)
    mmito = np.zeros(P)
    np.add.at(mmito, codes, pmito)
    mgene, mmito = mgene / np.maximum(C, 1), mmito / np.maximum(C, 1)

    report(f"\n  RELIABILITY vs CELLS PER PERTURBATION -- and why it does NOT rise the way it should")
    report(f"    {'dec':4s} {'n':>5s} {'cells':>7s} {'UMI/pert':>10s} {'ngene':>6s} {'%mito':>6s} "
           f"{'SE(lfc)':>8s} {'signal':>8s} {'obs rms':>8s} {'r half':>7s} {'r batch':>8s}")
    cq = np.digitize(C[A_idx], np.quantile(C[A_idx], np.linspace(.1, .9, 9)))
    dec_rows = []
    for q in range(10):
        m = cq == q
        if not m.sum():
            continue
        gi = A_idx[m]
        row = dict(decile=q, n=int(m.sum()), cells=float(C[gi].mean()), umi=float(cell_umi[gi].mean()),
                   ngenes=float(mgene[gi].mean()), mito=float(mmito[gi].mean()),
                   se=float(sig_p[m].mean()), signal=float(sig_true[m].mean()),
                   obs=float(obs_rms[m].mean()), r=float(np.nanmean(r_pert[gi])),
                   reliability=sb(np.nanmean(r_pert[gi])), r_batch=float(np.nanmean(r_pert_b[gi])),
                   noise_fraction=float(np.median(noise_frac[m])))
        dec_rows.append(row)
        report(f"    {q:<4d} {row['n']:5d} {row['cells']:7.1f} {row['umi']:10.2e} "
               f"{row['ngenes']:6.0f} {row['mito']:6.4f} {row['se']:8.4f} {row['signal']:8.4f} "
               f"{row['obs']:8.4f} {row['r']:7.4f} {row['r_batch']:8.4f}")
    sc_r = stats.spearmanr(C[A_idx], r_pert[A_idx], nan_policy="omit")
    sc_se = stats.spearmanr(C[A_idx], sig_p)
    sc_sig = stats.spearmanr(C[A_idx], sig_true)
    sc_obs = stats.spearmanr(C[A_idx], obs_rms)
    report(f"    Spearman vs cells/pert:  SE {sc_se[0]:+.4f}   true signal {sc_sig[0]:+.4f}   "
           f"observed rms {sc_obs[0]:+.4f}   reliability r {sc_r[0]:+.4f} (p {sc_r[1]:.2g})")
    report(f"    THE NOISE FALLS WITH CELLS EXACTLY AS POWER PREDICTS ({sc_se[0]:+.3f}), BUT SO DOES THE")
    report(f"    TRUE SIGNAL ({sc_sig[0]:+.3f}) -- CRISPRi depletes cells carrying guides against genes")
    report(f"    whose loss hurts, so a perturbation with few cells is one with a LARGE real effect.")
    report(f"    Reliability is signal/(signal+noise), the two move together, and it stays flat.")
    report(f"    -> cells-per-perturbation is NOT a clean power axis in this dataset. Matching on it, as")
    report(f"       this project's earlier +0.0881 confound result did, also matches on effect size.")
    report(f"    residual imbalance across the deciles: ngenes/cell "
           f"{dec_rows[0]['ngenes']:.0f}->{dec_rows[-1]['ngenes']:.0f}, %mito "
           f"{dec_rows[0]['mito']:.4f}->{dec_rows[-1]['mito']:.4f}, UMI/pert "
           f"{dec_rows[0]['umi']:.1e}->{dec_rows[-1]['umi']:.1e}")

    report(f"\n  RELIABILITY vs GENE EXPRESSION (per-gene r = a gene's response correlated ACROSS the "
           f"{len(A_idx):,} perturbations)")
    report(f"    {'dec':4s} {'n':>5s} {'mean CPM':>10s} {'dropout':>8s} {'r half':>7s} "
           f"{'reliability':>12s} {'r batch':>8s} {'overdisp':>9s}")
    gq = np.digitize(gmean_cpm, np.quantile(gmean_cpm, np.linspace(.1, .9, 9)))
    odg = np.sqrt(np.mean((SIG[A_idx] ** 2), 0)) / np.maximum(
        np.sqrt(np.mean((se_pois[A_idx] ** 2), 0)), 1e-12)
    gdec_rows = []
    for q in range(10):
        m = gq == q
        if not m.sum():
            continue
        row = dict(decile=q, n=int(m.sum()), cpm=float(gmean_cpm[m].mean()),
                   dropout=float(dropout[m].mean()), r=float(np.nanmean(r_gene[m])),
                   reliability=sb(np.nanmean(r_gene[m])), r_batch=float(np.nanmean(r_gene_b[m])),
                   overdispersion=float(np.median(odg[m])))
        gdec_rows.append(row)
        report(f"    {q:<4d} {row['n']:5d} {row['cpm']:10.2f} {row['dropout']:8.4f} {row['r']:7.4f} "
               f"{row['reliability']:12.4f} {row['r_batch']:8.4f} {row['overdispersion']:9.2f}")
    sg = stats.spearmanr(gmean_cpm, r_gene, nan_policy="omit")
    report(f"    Spearman(mean CPM, r) = {sg[0]:.4f}  p {sg[1]:.3g}; per-gene r median "
           f"{np.nanmedian(r_gene):.4f}, negative in {100*np.nanmean(r_gene<0):.1f}% of genes")
    report(f"    -> expression IS a clean power axis: a {gdec_rows[-1]['cpm']/gdec_rows[0]['cpm']:.0f}x "
           f"span in CPM buys {gdec_rows[0]['r']:.3f} -> {gdec_rows[-1]['r']:.3f} in reliability.")

    report(f"\n  STANDARD ERROR OF THE LOG FOLD CHANGE (empirical: Var(lfcA-lfcB) = 4 Var(lfc) exactly, "
           f"control term included)")
    report(f"    rms SE per perturbation: median {np.median(sig_p):.4f} "
           f"(p10 {np.percentile(sig_p,10):.4f}, p90 {np.percentile(sig_p,90):.4f})")
    report(f"    from batch-disjoint splits: median {np.median(sigb_p):.4f} "
           f"-> batch inflates the SE only {np.median(sigb_p)/np.median(sig_p):.3f}x, because the control")
    report(f"      is split the SAME way and a gemgroup's depth shift cancels in the difference")
    report(f"    Poisson delta-method SE:    median {np.median(pois_p):.4f}")
    report(f"    OVERDISPERSION over counting noise: median {np.median(od):.2f}x "
           f"(p10 {np.percentile(od,10):.2f}, p90 {np.percentile(od,90):.2f}); by expression decile "
           f"{gdec_rows[0]['overdispersion']:.2f} (low) -> {gdec_rows[-1]['overdispersion']:.2f} (high)")

    # ---------------------------------------------------------- bootstrap cross-check ----
    report(f"\n  BOOTSTRAP CROSS-CHECK ({BOOT_B} cell-level resamples, {len(boot_set)} perturbations)")
    ectrl = E[ctrl]
    brr, bmed = [], []
    rb_rng = np.random.default_rng(SEED + 2)
    for p in boot_set:
        Xp = buf[p]
        n = len(Xp)
        if n < MIN_CELLS:
            continue
        acc = np.empty((BOOT_B, G), dtype=np.float32)
        for b_ in range(BOOT_B):
            # multinomial weights == resampling n cells with replacement, but as one matvec
            w = rb_rng.multinomial(n, np.full(n, 1.0 / n)).astype(np.float32)
            s = (w @ Xp).astype(np.float64)
            acc[b_] = (np.log2(1.0 + 1e6 * s / max(s.sum(), 1)) - ectrl).astype(np.float32)
        bse = acc.std(0, ddof=1)
        ok = np.isfinite(bse) & (bse > 0) & (SIG[p] > 0)
        brr.append(float(np.corrcoef(bse[ok], SIG[p][ok])[0, 1]))
        bmed.append(float(np.median(SIG[p][ok] / bse[ok])))
    del buf
    report(f"    per-gene SE, bootstrap vs split-half: mean Pearson r {np.mean(brr):.4f} "
           f"(min {np.min(brr):.4f}) over {len(brr)} perturbations")
    report(f"    median ratio split-half/bootstrap {np.median(bmed):.4f} "
           f"(p10 {np.percentile(bmed,10):.4f}, p90 {np.percentile(bmed,90):.4f})")
    report(f"    the two are NOT expected to be exactly 1: the split-half SE includes control sampling "
           f"noise (which the bootstrap holds fixed, pushing the ratio up) while log2(1+CPM) is nonlinear")
    report(f"    so the 1/n variance scaling the /2 assumes is only approximate for near-zero genes "
           f"(pushing it down). The split-half SE is used everywhere because it covers all "
           f"{len(A_idx):,} perturbations.")

    # ---------------------------------------------------------- edge replication ----
    report(f"\n  EDGE REPLICATION WITHIN RPE1 -- select on half A, evaluate on the INDEPENDENT half B")
    report(f"    {'|z_A|>=':>8s} {'pairs/draw':>11s} {'sign agrees':>12s} {'sd':>7s} {'mean|z_B|':>10s} "
           f"{'|z_B|>=2':>9s} {'CPM of picks':>13s} {'gene r':>8s}")
    edge_sum = {}
    for t in THRS:
        rws = [r for r in edge_rows if r[0] == t]
        if not rws:
            continue
        cc = colc[t]
        fin = np.isfinite(r_gene) & (cc > 0)
        gr = float((cc[fin] * r_gene[fin]).sum() / max(cc[fin].sum(), 1))
        e = {"n_per_draw": float(np.mean([r[1] for r in rws])),
             "sign_agree": float(np.mean([r[2] for r in rws])),
             "sd": float(np.std([r[2] for r in rws])),
             "mean_abs_z_B": float(np.mean([r[3] for r in rws])),
             "frac_zB_ge2": float(np.mean([r[4] for r in rws])),
             "mean_cpm_of_selected": float(np.mean([r[5] for r in rws])),
             "mean_gene_reliability_of_selected": gr, "n_draws": len(rws)}
        edge_sum[str(t)] = e
        report(f"    {t:8.1f} {e['n_per_draw']:11,.0f} {e['sign_agree']:12.4f} {e['sd']:7.4f} "
               f"{e['mean_abs_z_B']:10.4f} {e['frac_zB_ge2']:9.4f} "
               f"{e['mean_cpm_of_selected']:13.2f} {gr:8.4f}")
    e2, e3, e5 = edge_sum.get("2.0"), edge_sum.get("3.0"), edge_sum.get("5.0")
    best = max(edge_sum.items(), key=lambda kv: kv[1]["sign_agree"])[0] if edge_sum else None
    if e2 and e3 and e5 and e5["sign_agree"] < e3["sign_agree"]:
        report(f"    REPLICATION PEAKS AT |z|>={best} AND FALLS AGAIN, WHICH IS BACKWARDS -- a stricter")
        report(f"    threshold should select stronger, more reproducible effects. It does up to "
               f"|z|>=3 ({100*e3['sign_agree']:.1f}%),")
        report(f"    then reverses to {100*e5['sign_agree']:.1f}% at |z|>=5. The pairs picked at |z|>=5 "
               f"sit on genes whose own")
        report(f"    split-half reliability is {e5['mean_gene_reliability_of_selected']:.3f} against "
               f"{e3['mean_gene_reliability_of_selected']:.3f} at |z|>=3, so the extreme tail of robust z "
               f"is enriched")
        report(f"    for LOW-RELIABILITY genes: dividing by a per-gene MAD lets a gene with little "
               f"variation reach a")
        report(f"    large |z| on little evidence. Mean expression of the picks barely moves "
               f"({e3['mean_cpm_of_selected']:.0f} -> {e5['mean_cpm_of_selected']:.0f} CPM), so")
        report(f"    expression is NOT the mechanism -- per-gene reliability is. The |z|>=5 cut that "
               f"defines this")
        report(f"    project's 537,376-edge layer is buying noise as well as effect size.")

    # ---------------------------------------------------------- outputs ----
    np.savez_compressed(
        MATS, perts=cats[A_idx], genes=gens,
        se_lfc=SIG[A_idx].astype(np.float32), se_lfc_batch=SIGB[A_idx].astype(np.float32),
        ncells=C[A_idx], umi=N_umi[A_idx], r_pert=r_pert[A_idx], r_pert_batch=r_pert_b[A_idx],
        r_gene=r_gene, r_gene_batch=r_gene_b, gene_mean_cpm=gmean_cpm.astype(np.float32),
        dropout=dropout.astype(np.float32))

    per_pert = {}
    for i, p in enumerate(A_idx):
        rv = r_pert[p]
        per_pert[str(cats[p])] = {
            "n_cells": int(C[p]), "umi": float(N_umi[p]), "umi_per_cell": float(N_umi[p] / max(C[p], 1)),
            "mean_ngenes": round(float(mgene[p]), 1), "mean_percent_mito": round(float(mmito[p]), 4),
            "r_split_half": None if not np.isfinite(rv) else round(float(rv), 4),
            "reliability_full": sb_item(rv),
            "r_split_half_batch": None if not np.isfinite(r_pert_b[p]) else round(float(r_pert_b[p]), 4),
            "se_lfc_rms": round(float(sig_p[i]), 5), "obs_rms_lfc": round(float(obs_rms[i]), 5),
            "signal_rms_lfc": round(float(sig_true[i]), 5),
            "noise_fraction": round(float(noise_frac[i]), 4)}
    per_gene = {str(gens[j]): {
        "symbol": str(gsym[j]), "mean_cpm": round(float(gmean_cpm[j]), 4),
        "dropout": round(float(dropout[j]), 4), "dropout_control": round(float(dropout_c[j]), 4),
        "r_split_half": None if not np.isfinite(r_gene[j]) else round(float(r_gene[j]), 4),
        "reliability_full": sb_item(r_gene[j]),
        "r_split_half_batch": None if not np.isfinite(r_gene_b[j]) else round(float(r_gene_b[j]), 4)}
        for j in range(G)}

    LIMITS = [
        "RPE1 ONLY. Split-half needs cells, and this project's K562 arm (gwps.h5ad) is already pseudobulked "
        "to 11,258 x 8,248 with the cells discarded. No ceiling is computed for K562 here, and the RPE1 "
        "ceiling must NOT be used to rescale a K562 score -- cells per perturbation, control size and "
        "library depth all differ",
        "the random split shares gemgroups between halves so batch effects cancel and are not counted as "
        "noise; the batch-disjoint split counts them. A within-batch question takes the random number, a "
        "cross-batch question the batch number. Here they barely differ, because the control is split the "
        "same way and a depth shift cancels in the difference",
        "Spearman-Brown assumes the halves are parallel measurements differing only in length. That holds "
        "for the random split (balanced within perturbation); for the batch split the halves are only "
        "approximately equal in size, so SB is approximate there",
        "reliability is a property of a QUANTITY, not of a dataset. The pooled raw-lfc ceiling is largely "
        "reproduced by shuffled perturbation labels because it contains per-gene main effects; only the "
        "gene-mean-removed and robust-z ceilings bound a model scored above a responsiveness baseline",
        "the empirical SE captures cell sampling and cell-to-cell biological variability. It does NOT "
        "capture guide-level effects (off-target activity, variable knockdown), which are shared by every "
        "cell carrying that guide and therefore cancel in a within-perturbation split -- so every "
        "reliability here is an OVER-estimate with respect to 'what does losing this gene do'",
        "cells-per-perturbation is not a clean power axis: CRISPRi depletes cells carrying guides against "
        "genes whose loss is costly, so low-cell perturbations have systematically LARGER true effects. "
        "Noise and signal both fall as cells rise and reliability stays roughly flat",
        f"perturbations below {MIN_CELLS} cells ({int((C < MIN_CELLS).sum())} of {P}) are outside the "
        f"analysis set; their reliability is not reported",
        "8,749 genes are measured. A gene not measured has no reliability here and absence is unknown "
        "rather than zero; dropout is reported per gene so the missingness is visible, and it is strongly "
        "expression-dependent",
        "the R2 ceiling assumes observed = signal + independent noise. Noise correlated between the halves "
        "(ambient RNA, index hopping, a systematic artefact shared by a perturbation's cells) would be "
        "counted as signal and would push the true ceiling LOWER",
        "the edge-replication rows select on a HALF-sized measurement and evaluate on the other half, so "
        "both sides are noisier than a full-data measurement. Those agreement rates are a LOWER bound on "
        "full-power within-dataset replication",
        "`noise_fraction` and `1 - reliability` are NOT the same number and must not be quoted "
        "interchangeably. noise_fraction is SE^2 / rms(lfc)^2, taken about zero, so a perturbation's mean "
        "shift across genes counts as signal; reliability is a CORRELATION, so each profile is "
        "mean-removed first and only gene-to-gene structure counts. The first answers 'how much of the "
        "measured magnitude is noise', the second 'how much of the measured pattern is reproducible', and "
        "they can differ severalfold for one perturbation"]

    ceiling = {"pooled_raw_lfc": pr, "pooled_raw_lfc_shuffled": prs,
               "pooled_gene_mean_removed": pd_, "pooled_gene_mean_removed_shuffled": pds,
               "pooled_robust_z": pz, "pooled_robust_z_shuffled": pzs,
               "shared_control_variant": ps,
               "per_perturbation_random": rr, "per_perturbation_batch": rb,
               "per_perturbation_shuffled": rs_,
               "frac_draws_beating_shuffle": fs}

    layer = {
        "model": "measurement-power-rpe1-v1", "source": SOURCE, "cell_type": "RPE1",
        "derived_not_downloaded": True,
        "pipeline": "sum raw UMI per group -> CPM -> log2(1+x) -> subtract non-targeting; identical to "
                    "colab/causal_reg.py and asserted against its cached pseudobulk",
        "design": {"n_random_splits": S_RAND, "n_batch_splits": S_BATCH,
                   "control_split_with_perturbations": True, "min_cells": MIN_CELLS,
                   "min_cells_per_half": MIN_HALF, "seed": SEED},
        "ceiling": ceiling,
        "se_method": "empirical split-half: sigma = rms(lfcA - lfcB)/2, unbiased for the full-data lfc "
                     "because Var(lfcA-lfcB) = 4 Var(lfc_full) exactly including the control term; "
                     "cross-checked against a cell-level bootstrap",
        "registry_joins": {"genes": st_g, "perturbations": st_p},
        "gene_cell_index": gene_cell_index,
        "limits": LIMITS, "per_perturbation": per_pert, "per_gene": per_gene}
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh)

    R = {"model": layer["model"], "source": SOURCE, "cell_type": "RPE1",
         "n_cells": int(nC), "n_genes": int(G), "n_perturbations": int(P),
         "n_analysis_perturbations": int(len(A_idx)), "n_control_cells": int(C[ctrl]),
         "median_cells_per_perturbation": float(np.median(C[A_idx])),
         "pipeline": layer["pipeline"], "design": layer["design"], "ceiling": ceiling,
         "per_perturbation_distribution": {
             "p10": float(np.nanpercentile(r_pert[A_idx], 10)),
             "median": float(np.nanmedian(r_pert[A_idx])),
             "p90": float(np.nanpercentile(r_pert[A_idx], 90)),
             "frac_negative": float(np.nanmean(r_pert[A_idx] < 0)),
             "frac_above_shuffled_p95": frac_sig, "shuffled_p95": thr95,
             "frac_draws_positive": float(np.nanmean(RP > 0))},
         "per_gene_summary": {"median_r": float(np.nanmedian(r_gene)),
                              "p90_r": float(np.nanpercentile(r_gene, 90)),
                              "frac_negative": float(np.nanmean(r_gene < 0)),
                              "spearman_cpm_vs_r": float(sg[0]), "p": float(sg[1])},
         "vs_cells_per_perturbation": {
             "deciles": dec_rows,
             "spearman_cells_vs_se": float(sc_se[0]), "spearman_cells_vs_signal": float(sc_sig[0]),
             "spearman_cells_vs_observed": float(sc_obs[0]),
             "spearman_cells_vs_reliability": float(sc_r[0]), "p_reliability": float(sc_r[1]),
             "note": "noise falls with cells as power predicts, but true signal falls too (CRISPRi "
                     "depletes costly perturbations), so reliability stays flat. Cells-per-perturbation "
                     "is not a clean power axis in this dataset."},
         "vs_gene_expression": {"deciles": gdec_rows},
         "coverage": {"n_perturbations_in_analysis_set": int(len(A_idx)),
                      "n_perturbations_excluded_below_min_cells": n_small,
                      "n_perturbations_not_estimable": n_p_nan,
                      "n_genes_estimable": int(G - n_g_nan), "n_genes_not_estimable": n_g_nan,
                      "note": "not-estimable is written as null, never 0"},
         "standard_error": {"method": layer["se_method"],
                            "median_rms_se": float(np.median(sig_p)),
                            "median_rms_se_batch": float(np.median(sigb_p)),
                            "median_poisson_se": float(np.median(pois_p)),
                            "median_overdispersion": float(np.median(od)),
                            "overdispersion_low_expr_decile": gdec_rows[0]["overdispersion"],
                            "overdispersion_high_expr_decile": gdec_rows[-1]["overdispersion"],
                            "bootstrap_check_pearson": float(np.mean(brr)),
                            "bootstrap_check_median_ratio": float(np.median(bmed)),
                            "n_bootstrap_perturbations": len(brr), "n_resamples": BOOT_B},
         "magnitude_confound": {"spearman_cells_vs_observed_rms": float(sc_obs[0]),
                                "spearman_cells_vs_signal_rms": float(sc_sig[0]),
                                "median_noise_fraction": float(np.median(noise_frac)),
                                "noise_fraction_low_cells_decile": dec_rows[0]["noise_fraction"],
                                "noise_fraction_high_cells_decile": dec_rows[-1]["noise_fraction"]},
         "edge_replication_within_rpe1": edge_sum,
         "registry_joins": {"genes": st_g, "perturbations": st_p}, "limits": LIMITS}

    report("\n" + "=" * 100)
    cz, cd_, cr = pz["reliability_full_sb"], pd_["reliability_full_sb"], pr["reliability_full_sb"]
    edge_txt = ""
    if e5 and e3:
        edge_txt = (f" Finally, the |z|>=5 cut that defines this project's 537,376-edge layer replicates "
                    f"its SIGN {100*e5['sign_agree']:.1f}% of the time between two halves of the SAME cell "
                    f"line -- a LOWER bound, since both sides are half-power. That is the honest reference "
                    f"for the 67.2% this project measured for K562 edges evaluated on RPE1: the "
                    f"cross-cell-line penalty is small next to the measurement-noise penalty, because "
                    f"most of the loss is already there when the SAME cells are split in two. And "
                    f"replication PEAKS at |z|>=3 ({100*e3['sign_agree']:.1f}%) and falls to "
                    f"{100*e5['sign_agree']:.1f}% at |z|>=5, on genes whose own reliability drops from "
                    f"{e3['mean_gene_reliability_of_selected']:.3f} to "
                    f"{e5['mean_gene_reliability_of_selected']:.3f} while their expression barely moves "
                    f"-- the extreme tail of a MAD-normalised z is enriched for low-reliability genes, so "
                    f"that threshold buys noise along with effect size.")
    v = (f"THE CEILING ON REPLOGLE RPE1 IS R2 = {cz:.3f}, NOT 1. Splitting each perturbation's cells into "
         f"halves -- with the {C[ctrl]:,} non-targeting cells split alongside them, so no control noise is "
         f"shared -- the halves agree at r = {pz['half_r']:.4f} (sd {pz['sd']:.4f} "
         f"over {S_RAND} draws) on the per-gene robust z this project defines its edges on, which "
         f"Spearman-Brown puts at reliability {cz:.4f} for the full-size measurement. On raw lfc with the "
         f"per-gene mean removed the ceiling is {cd_:.4f}. Leaving the gene main effects in gives "
         f"{cr:.4f}, but SHUFFLING the perturbation labels reproduces "
         f"{100*prs['half_r']/max(pr['half_r'],1e-9):.0f}% of that number, so it is a ceiling on a "
         f"quantity a gene-responsiveness baseline already owns. A model scored on perturbation-SPECIFIC "
         f"RPE1 response therefore cannot exceed R2 ~{cz:.2f} however good it is: the target is "
         f"{100*(1-cz):.0f}% noise. THIS DOES NOT RESCALE THIS PROJECT'S K562 NUMBERS -- the ladder's R2 "
         f"0.0151 and the 1.208x/1.070x/1.021x matched ratios were all scored on K562, whose cells were "
         f"discarded when gwps.h5ad was pseudobulked, so K562's own ceiling is unmeasured and dividing by "
         f"an RPE1 one would be the cross-axis substitution this project keeps getting burned by. What it "
         f"does establish is that a ceiling of this kind exists and is nowhere near 1, and that the same "
         f"computation is cheap wherever cell-level counts survive. Per perturbation the profile "
         f"reliability is {rr['reliability_full_sb']:.3f} ({100*frac_sig:.0f}% of perturbations beat the "
         f"95th percentile of a label-shuffled control), essentially unchanged at "
         f"{rb['reliability_full_sb']:.3f} under batch-disjoint splits because splitting the control the "
         f"same way cancels gemgroup depth. Reliability tracks EXPRESSION cleanly (half-r "
         f"{gdec_rows[0]['r']:.3f} at {gdec_rows[0]['cpm']:.1f} CPM to {gdec_rows[-1]['r']:.3f} at "
         f"{gdec_rows[-1]['cpm']:.0f} CPM, Spearman {sg[0]:.3f}) but NOT cells-per-perturbation "
         f"({sc_r[0]:+.3f}): the SE falls with cells exactly as power predicts ({sc_se[0]:+.3f}) while the "
         f"true signal falls with it just as fast ({sc_sig[0]:+.3f}), because CRISPRi depletes the cells "
         f"of perturbations that hurt. So the magnitude-vs-power confound this project found (+0.0881) is "
         f"real but is NOT fixed by matching on cell count -- that matches on effect size at the same "
         f"time. The median perturbation's observed rms |lfc| is {100*np.median(noise_frac):.0f}% noise "
         f"variance, and a log fold change carries a median rms SE of {np.median(sig_p):.3f}, "
         f"{np.median(od):.1f}x the Poisson delta-method value ({gdec_rows[-1]['overdispersion']:.1f}x for "
         f"the highest-expression decile), reproduced by a cell-level bootstrap at r {np.mean(brr):.3f}."
         + edge_txt)
    R["verdict"] = v
    report(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "measurement_power.json", "w"), indent=1, default=float)
    report(f"\n  -> {LAYER}  ({LAYER.stat().st_size/1e6:.1f} MB)")
    report(f"  -> {MATS}  ({MATS.stat().st_size/1e6:.1f} MB)")
    report(f"  -> {OUT/'measurement_power.json'}")


if __name__ == "__main__":
    main()
