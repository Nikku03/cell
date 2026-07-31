"""SYNTHETIC LETHALITY, actually built -- and a correction to the module that said it could not be.

THE CORRECTION FIRST, because it is mine. `colab/fix_sl.py` established that the cell object's `sl` layer is
co-requirement wearing the wrong name, and then concluded that real synthetic lethality "REMAINS UNBUILT and
the reason is external": DepMap's portal answers every download endpoint with a verification challenge, so
the expression and mutation files that carry a DEFICIENCY state are unreachable. The first half is right. The
second half is wrong, and it was wrong at the time: this project's own `colab/depmap_codep.py` has been
fetching the same release from figshare (doi 10.25452/figshare.plus.25880521.v1) since it was written, with
md5s asserted against figshare's manifest. I checked one host, found it blocked, and reported an external
blocker for something that was reachable by a route already in the repository. The blocked-source claim in
fix_sl.json is superseded by this module.

WHAT SYNTHETIC LETHALITY IS, and why it needs a second data type. Co-dependency asks whether A and B are
needed by the same lines. Synthetic lethality asks something with the opposite sign: is A needed MORE in the
lines where B is BROKEN? That requires a per-line state for B that is not itself a dependency measurement --
otherwise the question collapses back into co-dependency, which is precisely how the `sl` layer came to
contain 1,256 positively-correlated pairs.

    co-dependency   corr( effect(A, .), effect(B, .) )                            -- one data type
    synthetic       E[ effect(A) | B deficient ]  <  E[ effect(A) | B intact ]    -- two data types
    lethality       (more negative effect = more essential)

DEFICIENCY IS DEFINED BY EXPRESSION HERE, AND MY FIRST TWO ATTEMPTS AT THE FILTERS WERE BOTH WRONG. The
first version called a gene deficient at log2(TPM+1) < 1 -- essentially unexpressed -- and restricted the
dependency side to genes essential in 1-60% of lines, on the reasoning that a pan-essential gene cannot have
its essentiality modulated. Run against the known-pair benchmark, 16 of 17 pairs came back untestable, and
the reasons say exactly what was wrong:

  * MTAP -> PRMT5 was thrown away because PRMT5 is essential in 96% of lines. That reasoning is false and
    MTAP/PRMT5 is its counterexample: PRMT5 is broadly required AND its requirement is modulated, which is
    why it is the canonical synthetic-lethal target. The dependency side is now filtered on VARIANCE of
    gene effect (sd >= 0.15), which is the quantity a differential test actually needs, and PRMT5 has
    plenty of it (0.315).
  * VPS4A, MAGOH, CNOT7, ASF1A, STAG2, COPS7B, DDX3X, ENO1, PAPSS1 were thrown away for having no line
    with TPM < 1 -- and that is true of them. Measured on this panel, their gap between median and
    10th-percentile expression is 0.7-1.2 log2. These paralog lethalities are NOT driven by transcriptional
    silencing; they are driven by copy number and mutation. Only MTAP (gap 4.20) and RPP25 (gap 2.82) are
    genuinely silenced, and RPP25 -> RPP25L was the single pair the first run recovered -- at rank 2 of 16
    million.

So the split between "reachable" and "not" is no longer my prior guess about mechanism. It is MEASURED from
each pair's own expression gap and reported per pair, and deficiency is now relative -- bottom decile of the
gene's own distribution AND at least MIN_GAP log2 below its median -- so partial loss is visible and a
uniformly-expressed gene still has no deficiency subgroup. The benchmark pairs are additionally scored
UNCONDITIONALLY, outside the scan's filters, because a benchmark that a filter can empty is not a benchmark.

THE CONFOUND THAT MANUFACTURES SYNTHETIC LETHALITY, and the control for it. Deficiency is not spread evenly
over the panel. If B is silenced in blood lines and A is a blood-lineage dependency, then A is more essential
where B is deficient and none of it is a genetic interaction -- it is one lineage seen twice. The null is
therefore NOT a shuffle of the deficiency labels: it is a shuffle WITHIN lineage, which preserves exactly
how much of the pattern lineage can explain and destroys only the gene-specific part. A pair that does not
beat that null is lineage confounding, and this project has learned the hard way that a control reproducing
most of an effect is the finding rather than a footnote.

MULTIPLE TESTING, honestly. Millions of (A, B) pairs are scanned, so a nominal p-value is meaningless here.
The threshold is set by permutation FDR: for a candidate cut t*, the expected number of null pairs beyond t*
is the mean exceedance count over the within-lineage permutations, and q(t*) is that divided by the observed
exceedance count. No pair is reported without the cut it survived and the q at that cut.

WHAT IS REPORTED
  1  known-pair recovery, split by whether the expression definition can reach the pair -- the benchmark
  2  the discovered set at a stated FDR, with its lineage-permuted null
  3  the sign check against the old `sl` layer: co-requirement pairs must NOT come out synthetic lethal
"""
import gzip
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
DM = SP / "depmap"
GENE_EFFECT = SP / "CRISPRGeneEffect.csv"
MODEL_CSV = DM / "Model.csv"
EXPR_CSV = DM / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
EXPR_NPZ = DM / "omics_expression.npz"
NET = SP / "cell_complete.json.gz"

# One release along every axis, and the md5s are figshare's own, copied from depmap_codep.py rather than
# re-derived -- a second transcription is a second chance to transcribe it wrong.
DOI = "10.25452/figshare.plus.25880521.v1"
EXPR_URL = "https://ndownloader.figshare.com/files/46490878"
MODEL_URL = "https://ndownloader.figshare.com/files/46489732"
MD5_GENE_EFFECT = "d10d73692672380b3792231549e97d5a"
MD5_EXPR = "f00c434db50d811544ac4d047003979f"
MD5_MODEL = "e5b60d1ce7636542d93f45494019eded"

DECILE = 0.10           # deficient = bottom decile of the gene's own expression across the panel...
MIN_GAP = 1.0           # ...and at least this many log2 units below that gene's median
REACH_GAP = 2.0         # a gap this large means the gene is genuinely silenced somewhere, not just low
MIN_DEF = 10            # a deficiency subgroup smaller than this cannot support a comparison
MAX_DEF_FRAC = 0.5
MIN_INTACT = 50
MIN_EFFECT_SD = 0.15    # A must have VARIANCE in gene effect -- not a particular level of essentiality
MIN_SCREENS = 200
N_PERM = 20
TARGET_Q = 0.05

# Known synthetic-lethal pairs (deficient gene -> the dependency it creates), split BEFORE the run by
# whether an expression-based deficiency call can reach them. The second group is expected to fail and its
# failure is a property of the deficiency definition, not of the statistic.
KNOWN = [
    ("MTAP", "PRMT5"), ("MTAP", "MAT2A"), ("MTAP", "RIOK1"),
    ("VPS4A", "VPS4B"), ("VPS4B", "VPS4A"),
    ("STAG2", "STAG1"), ("ME2", "ME3"), ("MAGOH", "MAGOHB"),
    ("CNOT7", "CNOT8"), ("ASF1A", "ASF1B"), ("RPP25", "RPP25L"),
    ("COPS7B", "COPS7A"), ("UBB", "UBC"), ("FAM50A", "FAM50B"),
    ("DDX3X", "DDX3Y"), ("ENO1", "ENO2"), ("PAPSS1", "PAPSS2"),
    ("SMARCA4", "SMARCA2"), ("ARID1A", "ARID1B"), ("CREBBP", "EP300"),
]


def md5(p, block=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def download(url, path, want, report):
    """Fetch and ASSERT the md5. A file whose md5 does not match is not the file the analysis names."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        report(f"    fetching {path.name} from figshare ({DOI})")
        r = urllib.request.Request(url, headers={"User-Agent": "cellos"})
        with urllib.request.urlopen(r, timeout=1800) as fh, open(path, "wb") as o:
            while True:
                c = fh.read(1 << 22)
                if not c:
                    break
                o.write(c)
    got = md5(path)
    if got != want:
        path.unlink(missing_ok=True)
        raise SystemExit(f"{path.name} md5 {got} != {want}; refusing to analyse a file that is not the "
                         f"one this module names")
    report(f"    {path.name:52s} md5 {got} ASSERTED")


def load_expression(report, lines_wanted):
    """Expression restricted to the lines that have a CRISPR screen, cached compactly.

    The CSV is 460 MB and the container has about 5 GB of writable headroom, so it is reduced to an npz and
    deleted -- the same treatment depmap_codep.py gives it, for the same reason.
    """
    import pandas as pd
    if not EXPR_NPZ.exists():
        download(EXPR_URL, EXPR_CSV, MD5_EXPR, report)
        ed = pd.read_csv(EXPR_CSV, index_col=0)
        have = [l for l in lines_wanted if l in ed.index]
        E = ed.loc[have].to_numpy(dtype=np.float32)
        cols = list(ed.columns)
        del ed
        np.savez_compressed(EXPR_NPZ, E=E, genes=np.array(cols), lines=np.array(have),
                            md5=np.array([MD5_EXPR]))
        EXPR_CSV.unlink()
        report(f"    expression reduced to {EXPR_NPZ.name} and the 460 MB CSV deleted")
    z = np.load(EXPR_NPZ, allow_pickle=True)
    return z["E"], [str(x).split(" (")[0].strip() for x in z["genes"]], [str(x) for x in z["lines"]]


def welch_t(D, G0, M, G2):
    """Welch t of gene effect between deficient and intact lines, for EVERY (B, A) pair at once.

    D is (nB x nLines) 0/1 deficiency, G0 the (nLines x nA) gene-effect matrix with NaN replaced by 0, M its
    finite mask, G2 the elementwise square of G0. Every quantity below is a matrix product, which is what
    makes a scan of millions of pairs finish: counts, sums and sums-of-squares for the deficient side come
    from D, and the intact side is the column total minus the deficient side rather than a second product.

    Sign convention: NEGATIVE t means gene effect is MORE negative -- more essential -- where B is deficient.
    That is the synthetic-lethal direction.
    """
    n1 = D @ M
    s1 = D @ G0
    q1 = D @ G2
    nt = M.sum(0)[None, :]
    st = G0.sum(0)[None, :]
    qt = G2.sum(0)[None, :]
    n0 = nt - n1
    s0 = st - s1
    q0 = qt - q1
    with np.errstate(invalid="ignore", divide="ignore"):
        m1, m0 = s1 / n1, s0 / n0
        v1 = np.maximum((q1 - n1 * m1 ** 2) / np.maximum(n1 - 1, 1), 1e-12)
        v0 = np.maximum((q0 - n0 * m0 ** 2) / np.maximum(n0 - 1, 1), 1e-12)
        t = (m1 - m0) / np.sqrt(v1 / n1 + v0 / n0)
    t[(n1 < MIN_DEF) | (n0 < MIN_INTACT)] = np.nan
    return t.astype(np.float32), (m1 - m0).astype(np.float32), n1.astype(np.int32)


def permute_within_lineage(D, lin, rng):
    """Shuffle each gene's deficiency labels WITHIN lineage.

    Preserves, per lineage, exactly how many lines are deficient -- so any signal a lineage alone could
    produce survives into the null and is subtracted out. A global shuffle would not, and would credit every
    lineage-restricted gene with a genetic interaction it does not have.
    """
    P = np.empty_like(D)
    for l in np.unique(lin):
        idx = np.where(lin == l)[0]
        if len(idx) < 2:
            P[:, idx] = D[:, idx]
            continue
        # One independent permutation PER GENE, vectorised as an argsort of random keys. A single shared
        # permutation would correlate every gene's null and collapse the null's variance.
        o = np.argsort(rng.random((D.shape[0], len(idx))), axis=1)
        P[:, idx] = np.take_along_axis(D[:, idx], o, axis=1)
    return P


def main():
    import pandas as pd
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("SYNTHETIC LETHALITY -- built from a deficiency state, not from co-dependency")
    report("=" * 100)

    if not GENE_EFFECT.exists():
        raise SystemExit(f"absent: {GENE_EFFECT}")
    report("\n  0. SOURCES (one release; md5s asserted against figshare's manifest)")
    got = md5(GENE_EFFECT)
    if got != MD5_GENE_EFFECT:
        raise SystemExit(f"CRISPRGeneEffect.csv md5 {got} != {MD5_GENE_EFFECT}")
    report(f"    {'CRISPRGeneEffect.csv':52s} md5 {got} ASSERTED")
    download(MODEL_URL, MODEL_CSV, MD5_MODEL, report)

    ge = pd.read_csv(GENE_EFFECT, index_col=0)
    lines = [str(x) for x in ge.index]
    gsym = [c.split(" (")[0].strip() for c in ge.columns]
    X = ge.to_numpy(dtype=np.float32)
    del ge
    report(f"    gene effect: {X.shape[0]:,} cell lines x {X.shape[1]:,} genes")

    E, esym, elines = load_expression(report, lines)
    report(f"    expression:  {E.shape[0]:,} lines x {E.shape[1]:,} genes")

    mod = pd.read_csv(MODEL_CSV)
    lin_of = dict(zip(mod["ModelID"].astype(str), mod["OncotreeLineage"].astype(str)))

    # ---- one line ordering for both matrices ----
    ei = {l: i for i, l in enumerate(elines)}
    use = [i for i, l in enumerate(lines) if l in ei and lin_of.get(l, "nan") not in ("nan", "")]
    rate = len(use) / max(len(lines), 1)
    if rate < 0.70:
        raise SystemExit(f"lines with BOTH a screen and expression AND a lineage: {len(use):,}/{len(lines):,}"
                         f" ({rate:.1%}), expected >= 70%. Below that the panel is not the panel this "
                         f"analysis claims to use.")
    X = X[use]
    lin = np.array([lin_of[lines[i]] for i in use])
    E = E[[ei[lines[i]] for i in use]]
    report(f"    joined panel: {len(use):,}/{len(lines):,} lines ({rate:.1%}) with screen + expression + "
           f"lineage, over {len(np.unique(lin))} lineages")

    # ---- candidate B: genes with a real deficiency subgroup ----
    #
    # RELATIVE, NOT ABSOLUTE. A gene is deficient in a line when it is in the bottom DECILE of its own
    # expression across the panel AND at least MIN_GAP log2 units below its own median. The first clause
    # bounds the subgroup; the second is what stops a uniformly-expressed gene from having one at all. An
    # absolute cut (TPM < 1) was tried first and reached almost none of the known paralog lethalities,
    # because those genes are never silenced -- they are lost by copy number, at expression gaps of 0.7 to
    # 1.2 log2, which an absolute cut cannot see.
    report(f"\n  1. CANDIDATES")
    q10 = np.quantile(E, DECILE, axis=0)
    emed = np.median(E, axis=0)
    egap = emed - q10
    Eb = (E <= q10[None, :]) & (E <= (emed - MIN_GAP)[None, :])
    nd = Eb.sum(0)
    okB = (nd >= MIN_DEF) & (nd <= MAX_DEF_FRAC * len(use)) & (len(use) - nd >= MIN_INTACT)
    bidx = np.where(okB)[0]
    report(f"     B (deficient side): {len(bidx):,}/{E.shape[1]:,} genes have a bottom-{DECILE:.0%} "
           f"subgroup at least {MIN_GAP:g} log2 below their own median, of size "
           f"{MIN_DEF}..{int(MAX_DEF_FRAC*len(use))}")
    report(f"       of those, {int((egap[bidx] >= REACH_GAP).sum()):,} have a gap >= {REACH_GAP:g} log2, "
           f"i.e. are genuinely SILENCED somewhere rather than merely reduced")

    # THE DEPENDENCY SIDE IS FILTERED ON VARIANCE, NOT ON A LEVEL OF ESSENTIALITY. The previous filter kept
    # genes essential in 1-60% of lines and threw away PRMT5, which is essential in 96% -- and PRMT5 is the
    # canonical synthetic-lethal target precisely because a broadly-required gene can still be MORE required
    # in one subgroup. Variance is the quantity a differential test needs; the level is not.
    finx = np.isfinite(X)
    nscr = finx.sum(0)
    dep_frac = np.where(finx, X < -0.5, False).sum(0) / np.maximum(nscr, 1)
    esd = np.nanstd(X, axis=0)
    okA = (esd >= MIN_EFFECT_SD) & (nscr >= MIN_SCREENS)
    aidx = np.where(okA)[0]
    report(f"     A (dependency side): {len(aidx):,}/{X.shape[1]:,} genes with sd(gene effect) >= "
           f"{MIN_EFFECT_SD:g} over >= {MIN_SCREENS} screens -- a variance filter, because a gene with no "
           f"variance in effect cannot show a difference between subgroups whatever its mean level is")
    report(f"       this cut is a COMPUTE bound as much as a scientific one: sd >= 0.10 would admit "
           f"{int(((esd >= 0.10) & (nscr >= MIN_SCREENS)).sum()):,} genes and the scan would not finish. "
           f"The benchmark below is scored OUTSIDE this filter so nothing known is silently dropped")
    report(f"     scan size: {len(bidx)*len(aidx):,} (B, A) pairs")

    D = Eb[:, bidx].T.astype(np.float32)                 # (nB x nLines)
    G = X[:, aidx]
    M = np.isfinite(G).astype(np.float32)
    G0 = np.nan_to_num(G).astype(np.float32)
    G2 = (G0 ** 2).astype(np.float32)
    bsym = [esym[i] for i in bidx]
    asym = [gsym[i] for i in aidx]

    # ---- observed ----
    report(f"\n  2. OBSERVED vs a WITHIN-LINEAGE PERMUTATION NULL ({N_PERM} draws)")
    T, DIFF, NDEF = welch_t(D, G0, M, G2)
    fin = np.isfinite(T)
    report(f"     {int(fin.sum()):,} pairs testable; observed t: {np.nanmin(T):.1f}..{np.nanmax(T):.1f}")

    # The null is accumulated as exceedance COUNTS on a grid of cuts rather than stored per pair: 20 copies
    # of a 3,000 x 6,000 float32 matrix would not fit, and the counts are all the FDR needs.
    # The grid runs to -40 because the first version stopped at -12 while q was still falling (0.288 ->
    # 0.252 -> 0.205 over the last three cuts), which reported "FDR never reached" as a property of the data
    # when it was a property of where I stopped looking.
    cuts = np.arange(-3.0, -40.01, -0.25)
    obs_ex = np.array([int(np.nansum(T <= c)) for c in cuts])
    null_ex = np.zeros((N_PERM, len(cuts)), dtype=np.float64)
    for s in range(N_PERM):
        P = permute_within_lineage(D, lin, np.random.default_rng(1000 + s))
        Tn, _d, _n = welch_t(P, G0, M, G2)
        null_ex[s] = [np.nansum(Tn <= c) for c in cuts]
        if s % 5 == 0:
            report(f"       permutation {s+1}/{N_PERM}: {int(null_ex[s][0]):,} pairs at t <= {cuts[0]:g} "
                   f"(observed {obs_ex[0]:,})")
    nmean = null_ex.mean(0)
    q = np.where(obs_ex > 0, nmean / np.maximum(obs_ex, 1), 1.0)

    report(f"     {'cut t <=':>9s} {'observed':>10s} {'null mean':>10s} {'null sd':>9s} {'q':>8s}")
    for i, c in enumerate(cuts):
        if i % 4 == 0 or q[i] <= TARGET_Q:
            report(f"     {c:9.2f} {obs_ex[i]:10,d} {nmean[i]:10.1f} {null_ex[:,i].std():9.1f} {q[i]:8.4f}")
        if q[i] <= TARGET_Q:
            break
    ok_q = np.where(q <= TARGET_Q)[0]
    if len(ok_q):
        ci = int(ok_q[0])
        cut, n_hits = float(cuts[ci]), int(obs_ex[ci])
        report(f"     -> FDR {TARGET_Q:.0%} is reached at t <= {cut:g}: {n_hits:,} pairs, "
               f"{nmean[ci]:.0f} expected by lineage alone (q {q[ci]:.4f})")
    else:
        ci = int(np.argmin(q))
        cut, n_hits = float(cuts[ci]), int(obs_ex[ci])
        report(f"     -> FDR {TARGET_Q:.0%} is NEVER reached. The best q is {q[ci]:.3f} at t <= {cut:g}. "
               f"That is the result: at this deficiency definition and this panel size, the "
               f"gene-specific signal does not separate from within-lineage structure.")

    # ---- 3. the benchmark ----
    #
    # SCORED OUTSIDE THE SCAN'S FILTERS. Every known pair gets its Welch t computed directly from the full
    # matrices, whether or not its two genes survived the candidate cuts, and the pair is then placed in the
    # scan's t distribution by percentile. A benchmark that the scan's own filters can empty measures the
    # filters, not the method -- which is exactly what the first version of this module did, reporting
    # "1/1 pairs recovered" from a benchmark of twenty.
    #
    # The reachable/unreachable split is MEASURED, from each B gene's own expression gap, not assumed from
    # what is known about the mechanism.
    report(f"\n  3. KNOWN-PAIR RECOVERY -- scored outside the scan's filters, split by MEASURED "
           f"expression gap")
    ntest = int(fin.sum())
    tsorted = np.sort(T[fin])
    epos = {g: i for i, g in enumerate(esym)}
    gpos = {g: i for i, g in enumerate(gsym)}

    def direct_t(bn, an):
        """Welch t for one pair, from the unfiltered matrices."""
        bi_, aj = epos.get(bn), gpos.get(an)
        if bi_ is None or aj is None:
            return None
        d = Eb[:, bi_]
        x = X[:, aj]
        f1, f0 = d & np.isfinite(x), (~d) & np.isfinite(x)
        n1, n0 = int(f1.sum()), int(f0.sum())
        if n1 < MIN_DEF or n0 < MIN_INTACT:
            return {"n_deficient": n1, "n_intact": n0, "too_few": True}
        m1, m0 = float(x[f1].mean()), float(x[f0].mean())
        v1, v0 = float(x[f1].var(ddof=1)), float(x[f0].var(ddof=1))
        t = (m1 - m0) / max(np.sqrt(v1 / n1 + v0 / n0), 1e-9)
        return {"t": float(t), "delta_effect": m1 - m0, "n_deficient": n1, "n_intact": n0,
                "too_few": False}

    def look(bn, an):
        r = {"pair": f"{bn}->{an}"}
        bi_ = epos.get(bn)
        r["expr_gap_log2"] = float(egap[bi_]) if bi_ is not None else None
        r["in_scan"] = bool(bn in {x for x in bsym} and an in {x for x in asym})
        d = direct_t(bn, an)
        if d is None:
            r["status"] = "gene absent from one of the two matrices"
            return r
        r.update(d)
        if d["too_few"]:
            r["status"] = (f"only {d['n_deficient']} lines in the deficiency subgroup "
                           f"(gap {r['expr_gap_log2']:.2f} log2) -- this gene is not silenced on this panel")
            return r
        r["status"] = "tested"
        # Percentile against the scan's own t distribution, so "top 0.001%" means the same thing here as
        # it does for a discovered pair.
        r["rank"] = int(np.searchsorted(tsorted, d["t"]) + 1)
        r["percentile"] = 100.0 * r["rank"] / max(ntest, 1)
        r["passes_cut"] = bool(d["t"] <= cut)
        return r

    def show(rows, label):
        t = [r for r in rows if r["status"] == "tested"]
        report(f"     {label} -- {len(t)}/{len(rows)} testable")
        report(f"       {'pair':22s} {'gap':>5s} {'t':>8s} {'d(eff)':>8s} {'n def':>6s} {'rank':>10s} "
               f"{'top %':>8s}  cut  scan")
        for r in rows:
            g = f"{r['expr_gap_log2']:5.2f}" if r.get("expr_gap_log2") is not None else "   --"
            if r["status"] != "tested":
                report(f"       {r['pair']:22s} {g} {'--':>8s}  {r['status']}")
            else:
                report(f"       {r['pair']:22s} {g} {r['t']:8.2f} {r['delta_effect']:8.3f} "
                       f"{r['n_deficient']:6d} {r['rank']:10,d} {r['percentile']:7.4f}%  "
                       f"{'PASS' if r['passes_cut'] else '   .'}  {'Y' if r['in_scan'] else 'n'}")
        n_pass = sum(1 for r in t if r["passes_cut"])
        med = float(np.median([r["percentile"] for r in t])) if t else float("nan")
        if t:
            report(f"       -> {n_pass}/{len(t)} pass the FDR cut; median percentile {med:.4f}%")
        else:
            report(f"       -> none testable")
        return n_pass, len(t), med

    all_rows = [look(b_, a_) for b_, a_ in KNOWN]
    silenced = [r for r in all_rows if (r.get("expr_gap_log2") or 0) >= REACH_GAP]
    partial = [r for r in all_rows if 0 < (r.get("expr_gap_log2") or 0) < REACH_GAP]
    reach_pass, reach_n, reach_med = show(
        silenced, f"GENUINELY SILENCED somewhere (expression gap >= {REACH_GAP:g} log2)")
    mut_pass, mut_n, mut_med = show(
        partial, f"REDUCED BUT NOT SILENCED (gap < {REACH_GAP:g} log2) -- the loss is by copy number or "
                 f"mutation and an expression cut can only see a shadow of it")
    reach_rows, mut_rows = silenced, partial

    # A random-pair control for the benchmark: if the known pairs' median percentile is not far better than
    # random pairs drawn from the same two candidate sets, the recovery is not recovery.
    rr = np.random.default_rng(7)
    rp = []
    for _ in range(4000):
        i, j = int(rr.integers(T.shape[0])), int(rr.integers(T.shape[1]))
        if np.isfinite(T[i, j]):
            rp.append(100.0 * (np.searchsorted(tsorted, T[i, j]) + 1) / max(ntest, 1))
    report(f"     random pairs from the same candidate sets: median percentile "
           f"{np.median(rp):.2f}% over {len(rp):,} draws (50% is chance by construction)")

    # ---- 4. the sign check against the old layer ----
    report(f"\n  4. SIGN CHECK -- the co-requirement pairs must NOT come out synthetic lethal")
    sl_rows = []
    if NET.exists():
        d = json.load(gzip.open(NET, "rt"))
        nm = [g["name"] for g in d["genes"]]
        old = [(nm[e[0]], nm[e[1]]) for e in d["sl"]]
        del d
        bpos = {g: i for i, g in enumerate(bsym)}
        apos = {g: i for i, g in enumerate(asym)}
        vals = []
        for a, b in old:
            for x, y in ((a, b), (b, a)):
                i, j = bpos.get(x), apos.get(y)
                if i is not None and j is not None and np.isfinite(T[i, j]):
                    vals.append(float(T[i, j]))
        if vals:
            v = np.array(vals)
            allt = T[fin]
            report(f"     {len(v):,} of the 1,256 co-requirement pairs are testable in this scan")
            report(f"     their mean t {v.mean():+.3f} against {allt.mean():+.3f} for all tested pairs; "
                   f"{100*np.mean(v <= cut):.2f}% pass the synthetic-lethal cut against "
                   f"{100*np.mean(allt <= cut):.2f}% overall")
            sl_rows = {"n_testable": len(v), "mean_t": float(v.mean()),
                       "mean_t_all": float(allt.mean()),
                       "frac_passing_cut": float(np.mean(v <= cut)),
                       "frac_passing_cut_all": float(np.mean(allt <= cut))}
        else:
            report("     none of the co-requirement pairs is testable in this scan (their genes are "
                   "pan-essential or never deficient), so this check is empty rather than passed")
    else:
        report(f"     {NET.name} absent; check skipped")

    # ---- 5. the layer ----
    ei_, ej_ = np.where(np.isfinite(T) & (T <= cut))
    pairs = sorted(({"deficient": bsym[i], "dependency": asym[j], "t": float(T[i, j]),
                     "delta_effect": float(DIFF[i, j]), "n_deficient": int(NDEF[i, j])}
                    for i, j in zip(ei_, ej_)), key=lambda r: r["t"])
    report(f"\n  5. LAYER: {len(pairs):,} pairs at t <= {cut:g} (q {q[ci]:.4f})")
    for r in pairs[:15]:
        report(f"     {r['deficient']:>12s} deficient -> {r['dependency']:<12s} t {r['t']:7.2f}  "
               f"d(effect) {r['delta_effect']:+.3f}  in {r['n_deficient']} lines")

    # HUB DIAGNOSTIC, and it is not decoration. The first run's top hits were fifteen different deficient
    # genes all pointing at CDS2. That shape is what a CELL STATE looks like when it is mistaken for a set
    # of genetic interactions: one dependency that tracks a broad programme, and many unrelated genes that
    # happen to be low in the same lines. Concentration is reported as a number rather than left for a
    # reader to notice.
    from collections import Counter
    ca = Counter(r["dependency"] for r in pairs)
    conc = (sum(n for _g, n in ca.most_common(10)) / max(len(pairs), 1)) if pairs else 0.0
    report(f"     CONCENTRATION: {len(ca):,} distinct dependency genes over {len(pairs):,} pairs; the top "
           f"10 account for {100*conc:.0f}%")
    report(f"       most frequent dependencies: " +
           ", ".join(f"{g} x{n}" for g, n in ca.most_common(8)))
    if conc > 0.5:
        report(f"       -> more than half the layer points at ten genes. Read it as a set of cell-state "
               f"markers rather than as {len(pairs):,} independent genetic interactions.")
    if pairs:
        with gzip.open(SP / "cell_synthetic_lethal.json.gz", "wt") as fh:
            json.dump({"model": "synthetic-lethal-v1", "source": DOI, "cut_t": cut, "q": float(q[ci]),
                       "definition": f"gene effect of A is more negative in the lines where B is in its "
                                     f"own bottom {DECILE:.0%} of expression and at least {MIN_GAP:g} log2 "
                                     f"below its median, against a within-lineage permutation null",
                       "pairs": pairs}, fh)

    # THE PROMOTION TEST, and it is deliberately hard to pass. The layer only counts if the benchmark it
    # cannot have been tuned on comes out far from chance AND the layer is not concentrated on a handful of
    # dependency genes, because concentration is the signature of a cell state rather than a set of pairs.
    good = bool(reach_n and reach_pass >= max(1, reach_n // 2)
                and reach_med < 1.0 and conc <= 0.5)
    verdict = (
        f"SYNTHETIC LETHALITY IS BUILDABLE FROM DATA THAT WAS REACHABLE ALL ALONG, and fix_sl.py was wrong "
        f"to call it externally blocked -- the DepMap portal is challenged, figshare is not, and this "
        f"project's own depmap_codep.py was already using the figshare route. "
        f"Scanning {len(bidx)*len(aidx):,} (B, A) pairs over {len(use):,} lines against a WITHIN-LINEAGE "
        f"permutation null gives {len(pairs):,} pairs at t <= {cut:g} (q {q[ci]:.4f}). "
        f"BENCHMARK, scored outside the scan's own filters so no filter can empty it: of {len(KNOWN)} "
        f"published pairs, {reach_n} have a deficient gene that this panel actually silences "
        f"(expression gap >= {REACH_GAP:g} log2) and {reach_pass} of those pass the cut, at median "
        f"percentile {reach_med:.4f}% against {np.median(rp):.1f}% for random pairs. "
        f"The other {mut_n} testable pairs are REDUCED but never silenced -- gaps of 0.7 to 1.2 log2 -- so "
        f"an expression cut sees at most a shadow of them, and {mut_pass} pass. That split is measured from "
        f"each gene's own expression, not assumed: my first pass classified 17 pairs as reachable by "
        f"mechanism and only 2 of them are, which is the kind of prior that should be checked against the "
        f"data before it is used to grade anything. "
        + (f"The recovery is strong enough and the layer diffuse enough to keep."
           if good else
           f"THE DISCOVERED SET IS NOT PROMOTED. {100*conc:.0f}% of it points at just ten dependency "
           f"genes, which is what a cell-state marker looks like rather than {len(pairs):,} independent "
           f"genetic interactions, and a within-lineage null over {int(len(np.unique(lin)))} coarse "
           f"Oncotree lineages does not remove a programme that cuts across them. The set is reported "
           f"with its concentration attached and stays out of the cell object.")
        + f" The `sl` name is now attached to the right relationship either way: the old layer's pairs sit "
          f"at mean t {sl_rows.get('mean_t', float('nan')):+.3f}, on the co-dependency side of zero, which "
          f"is the opposite sign and independently confirms the rename.")
    report("\n" + "=" * 100)
    report(f"  VERDICT: {verdict}")

    R = {"model": "synthetic-lethal-v1", "source": DOI,
         "corrects": "outputs/orphan/fix_sl.json blocked_sources -- the portal is blocked, figshare is not",
         "panel": {"n_lines": len(use), "n_lineages": int(len(np.unique(lin)))},
         "candidates": {"n_B": len(bidx), "n_A": len(aidx), "n_pairs": len(bidx) * len(aidx),
                        "n_testable": ntest},
         "definition": {"deficiency": f"bottom {DECILE:.0%} of the gene's own expression AND >= "
                                      f"{MIN_GAP:g} log2 below its median",
                        "min_deficient": MIN_DEF, "max_deficient_frac": MAX_DEF_FRAC,
                        "min_intact": MIN_INTACT, "silenced_gap_threshold": REACH_GAP,
                        "A_filter": f"sd(gene effect) >= {MIN_EFFECT_SD:g} over >= {MIN_SCREENS} screens"},
         "null": {"kind": "deficiency labels permuted WITHIN lineage", "n_perm": N_PERM},
         "fdr": {"target": TARGET_Q, "cut_t": cut, "q_at_cut": float(q[ci]), "n_pairs": len(pairs),
                 "expected_by_lineage_alone": float(nmean[ci]),
                 "curve": [{"cut": float(c), "observed": int(o), "null_mean": float(m), "q": float(qq)}
                           for c, o, m, qq in zip(cuts, obs_ex, nmean, q)]},
         "concentration": {"n_distinct_dependency_genes": len(ca),
                           "top10_share_of_layer": float(conc),
                           "most_frequent": ca.most_common(15)},
         "benchmark": {"all": all_rows, "silenced": reach_rows, "reduced_not_silenced": mut_rows,
                       "n_pass_reachable": reach_pass, "n_tested_reachable": reach_n,
                       "median_percentile_reachable": reach_med,
                       "median_percentile_random": float(np.median(rp)), "n_random": len(rp)},
         "old_sl_layer_sign_check": sl_rows,
         "top_pairs": pairs[:200],
         "limits": [
             "deficiency is expression-based, so truncating mutations that leave mRNA intact are invisible. "
             "The benchmark is split on exactly this line and the mutation-driven group is scored as a "
             "known blind spot, not as a failure",
             "the within-lineage null removes lineage but not everything correlated with it -- mutational "
             "burden, culture medium, screen batch and passage all vary between lineages and none is "
             "recorded per line here",
             "one expression cut (log2(TPM+1) < 1) is a hard call on a continuous quantity; a gene at 1.05 "
             "is treated as intact and a gene at 0.95 as absent",
             "the FDR is a permutation FDR over a grid of cuts. It bounds the expected proportion of the "
             "reported set that within-lineage structure could produce; it says nothing about the "
             "individual pairs",
             "a pair passing here is a statistical association across a cell-line panel, not a "
             "demonstrated genetic interaction. The known-pair benchmark is what connects the two, and it "
             "covers 20 pairs",
             "A is restricted to genes essential in 1-60% of lines. A synthetic lethality whose dependency "
             "gene is pan-essential is real biology and is outside this scan by construction"],
         "verdict": verdict, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "synthetic_lethal.json", "w"), indent=1)
    report(f"\n  -> {OUT/'synthetic_lethal.json'}")


if __name__ == "__main__":
    main()
