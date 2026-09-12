"""LOOP 114 -- FORWARD LINK: MAKE THE CHROMATIN MODEL EMIT A TRANSCRIPTION RATE, NOT A CONTACT MAP.

WHAT THIS IS FOR. Loop 95 found that chromatin correlates with transcription rate at +0.1545 raw and
+0.0368 given abundance -- the only layer in this project that beat both fame and abundance. But that
was a CORRELATION COMPUTED AFTERWARDS: the chromatin arc emits contact maps, loop 91 emits rates, and
loop 95 put the two vectors side by side. Nothing in the repository takes a contact map in and hands
back a number in mRNA/cell/h. This loop builds that arrow and then tries to break it.

=====================================================================================================
WHICH QUANTITIES ARE INPUTS AND WHICH ARE PREDICTIONS, PER GENE -- READ THIS FIRST
=====================================================================================================
INPUTS the emitter is allowed to see, per gene:
    the gene's hg19 TSS coordinate                      (_tss_hg19.bed, lifted in loop 63)
    the measured GM12878 Hi-C near-diagonal band        (Rao 2014, KR, 25 kb, streamed)
    GM12878 CTCF peak positions                         (ctcf_gm12878_hg19.bed.gz)
    -- and NOTHING else. Not its expression, not its copy number, not its half-life, not its
    publication count, not its function, not its identity.
INPUTS that are global scalars, declared, not fitted per gene:
    delta_global = ln2 / 9.92 h, the MEDIAN Schwanhausser mRNA half-life, used only to put Larsson's
    dimensionless per-allele rates into 1/h. One number for all genes; it cannot create per-gene rank.
    T_DOUBLE_H = 27.5 h (loop 91) and PLOIDY = 2 (loop 98).
PREDICTION, per gene:
    k_sm_hat in mRNA/cell/h, from one global monotone calibration fitted on a TRAINING half of genes
    and applied unchanged to a HELD-OUT half. Nothing about a held-out gene enters its own prediction.
TARGETS, per gene, never seen by the emitter:
    k_sm_larsson = PLOIDY * kon*ksyn/(kon+koff) * delta_global, from Larsson 2019 allele-resolved
        single-cell burst kinetics. An INDEPENDENT lab, technology and inference. THE REAL TARGET.
    k_sm_identity = mrna_copies * (ln2/t_half + ln2/T_DOUBLE), loop 91's identity. Its own input is
        the gene's mRNA copy number, so it is reported but never used to make the headline claim.

=====================================================================================================
THE TRAP THIS LOOP IS BUILT AROUND, STATED SO IT CANNOT BE FORGOTTEN
=====================================================================================================
k_sm_identity = M_obs * k_d, so integrating dM/dt = k_sm_identity - k_d M returns M* = M_obs exactly,
for every gene, always, to machine precision. That is ALGEBRA, not simulation. Gate X7 runs it ON
PURPOSE, prints the resulting relative error, and labels it as the circular arm that is excluded from
every other claim in this file -- the same construction loop 101's D5 used to recover the 24 h that
went in at rel err 4e-15. X7 then runs the SAME integration with the chromatin-emitted k_sm_hat on
held-out genes, where the gene's own mRNA level entered nothing, and the gap between the two error
distributions is the actual result of that gate.

WHY THE LARSSON TARGET IS NOT THE SAME TRAP. No steady-state synthesis rate is abundance-free -- every
one of them is abundance x turnover, and pretending otherwise would be its own dishonesty. What
matters is whether the target is derived from a quantity the PREDICTOR SEES. It is not: the emitter
sees only TSS coordinates, a contact band and CTCF positions, and the Larsson rate comes from a
different lab, a different technology and a different measurement of a different cell population than
the Schwanhausser copy numbers loop 91 used. The calibration is fitted on disjoint genes.

=====================================================================================================
THE EMISSION
=====================================================================================================
For a gene whose TSS falls in bin b of chromosome c, over the +/-1 Mb window (2 <= |d| <= 40 bins,
the diagonal two bins dropped so the score is not the trivial self-contact plateau):

    S_ctcf  = sum of observed/expected contact between bin b and every mappable CTCF-anchored bin
    D_ctcf  = the count of those bins -- which is exactly what S_ctcf becomes if O/E == 1 everywhere,
              i.e. what a model knowing ONLY genomic distance and anchor positions can emit
    S_all   = the same sum over every mappable bin in the window, anchors or not      (reported)
    S_ctcf / D_ctcf = mean O/E at anchors, the part of the score with the counting removed (reported)

S_ctcf -> log10 k_sm by isotonic regression fitted on the training half only. Isotonic because the
claim is "more promoter contact means more transcription", which is a monotone claim and should be
tested with a monotone function rather than a shape chosen after seeing the data.

AND THE MODEL'S OWN MAP, NOT ONLY THE MEASURED ONE. The same emission is run on the SIMULATED map
from loops 77/79 -- loop-extrusion at the untuned literature parameters, nothing fitted to anything
here -- on chr21 and chr22, the two chromosomes whose Gaussian-network inverse is affordable. That
arm is small (n is reported, and it is small) and is labelled as such, but it is the only version of
this that answers the title literally: the chromatin MODEL emitting a rate.

DISCLOSURE, per rule 2. Before these gates were fixed I ran reconnaissance and looked at exactly
these numbers, all of them counts or timings, none of them an effect size:
    6,180 Larsson S5 rows; 7,297 symbols parse across the C57 and CAST sheets; 5,755 match a
        cell_complete symbol; 5,752 of those carry an hg19 TSS. The n >= 300 bar in X3 was set
        knowing that, and knowing chr21+chr22 alone hold only 212 of them -- which is why the
        measured arm streams the whole genome instead of using the two cached chromosomes.
    4,190 Schwanhausser genes with copies and mRNA half-life, 3,362 with a TSS, 2,444 in both sets.
    median Schwanhausser mRNA half-life 9.92 h -- this became delta_global, an INPUT.
    hicstraw timings: ~15 s index + 3-30 s records per chromosome.
No correlation, no partial correlation, no survival fraction, no calibration and no fold error was
computed before the gate list below was written and its thresholds fixed.

PREDECLARED, before any number beyond those:

  X1 THE BAND IS THE SAME OBJECT LOOP 85 REPAIRED                     THE PREREQUISITE.
       the streamed near-diagonal band and its bincount-derived mappability must reproduce, on
       chr21, the cached full-matrix pipeline loops 33-95 scored on: the SAME mappable-bin count
       (loop 33's 1,377) and Spearman >= 0.99 between band and full-matrix observed values inside
       the window. If the streamed band is not that object, nothing below is comparable to anything
       in this arc and the loop says so.

  X2 THE EMISSION EXISTS, AND ITS COVERAGE IS DECLARED                COVERAGE, REPORTED NOT GATED.
       what fraction of the 16,492 modelled genes receive a k_sm from chromatin alone, and what
       fraction of the Larsson-rated genes do. Printed whatever it is, because an effect size on a
       corner of the genome must be readable as such.

  X3 HELD OUT: THE EMITTED RATE PREDICTS THE INDEPENDENT RATE         THE MEASUREMENT.
       Spearman of k_sm_hat against k_sm_larsson on the held-out half, genes the calibration never
       saw. PASS needs n >= 300, rho > 0, and a label-permutation p < 0.01. The identity target is
       scored alongside and marked as the abundance-derived one.

  X4 IT BEATS FAME                                                    THE GATE THAT KILLED 87, 87b, 94.
       pubs on the same held-out genes, always reported. PASS needs rho(chromatin) > rho(pubs) AND
       the partial correlation given pubs to retain >= 50% of the raw value.

  X5 IT BEATS THE DISTANCE-ONLY NULL                                  THE GATE LOOP 79 SET.
       D_ctcf, the score with O/E set to 1 -- what genomic distance and anchor positions alone can
       say. PASS needs rho(S_ctcf) > rho(D_ctcf) AND partial(S_ctcf | D_ctcf) > 0. If this fails,
       the contact map contributed nothing beyond CTCF density and the map is decoration.

  X6 THE SHUFFLED-POSITION NULL, VERIFIED CAPABLE BEFORE IT COUNTS    THE GUARD, WITH A GUARD.
       TSS bins permuted among scored genes WITHIN a chromosome, recomputed end to end including
       the calibration refit. GG.null_can_move must confirm the null moves the score vector before
       any null output is allowed to count; then GG.survival over NPERM draws must be DEFINED and
       the surviving fraction below 0.50.

  X7 THE CIRCULAR ARM, RUN ON PURPOSE, AND THE HONEST ONE BESIDE IT   THE TAUTOLOGY.
       (a) integrate dM/dt = k_sm_identity - k_d M with loop 91's identity k_sm. M* must equal the
           INPUT mrna_copies to relative error <= 1e-10. That is a PASS of an identity, it is not
           evidence of anything, and it is excluded from X3-X6 and from the headline.
       (b) integrate the same ODE with the chromatin-emitted k_sm_hat on held-out genes and the
           gene's own measured mRNA half-life, and compare M* to the observed copy number the
           emitter never saw. Reported as a fold error, not as a pass.

ADDED AFTER THE FIRST RUN, AND DISCLOSED AS SUCH. Four blocks below were written after seeing the
first run's numbers. NO PREDECLARED THRESHOLD WAS MOVED and no gate changed its verdict; all four
exist to stop this loop over-reading its own output, and three of them make the result WORSE:

  (i)   "WHAT X5's PASS IS WORTH". X5 fired -- +0.0158 beats +0.0065 with a positive partial -- and
        X6 then showed +0.0158 is 0.81 null standard deviations from zero. So X5 compared two
        numbers that are both indistinguishable from nothing. That is exactly gate_guard's FAMILY
        ONE, the fault behind loop 87 C6's "54% survival" and loop 87b B6's "6718%". X5's PASS is
        recorded as PASS because that was the predeclared rule, and it is simultaneously recorded
        as VACUOUS, because it is.
  (ii)  THE SECOND TARGET GETS THE SAME NULL. The score reaches the abundance-derived identity rate
        at a much larger value than the independent one, so the shuffled-position null is run
        against that target too, in the same loop. A number that large deserves the same guard, and
        the contrast between the two targets is the most interesting thing this loop found.
  (iii) IS THE LARSSON TARGET JUST NOISE? If it were, no feature could predict it and the negative
        would be uninformative. pubs is already computed on exactly those held-out genes and is the
        answer.
  (iv)  A CONSTANT-k_sm CONTROL FOR X7(b). M* = k_sm/k_d, and k_d comes from the gene's own measured
        half-life, so a good-looking M* correlation can be entirely the half-life. The same ODE is
        therefore integrated with a CONSTANT k_sm and the two are printed side by side.

-> outputs/loop_chromatin_to_rate.json
"""
import gzip
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import gate_guard as GG  # noqa: E402
import loop_replication as LR  # noqa: E402
import loop_second as L77  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
CELL = LR.CELL
BIN = L77.BIN
CTCF_BED = L77.CTCF
TSS_BED = SC / "_tss_hg19.bed"
SCHWAN = SC / "_schwan2011.json"
LARSSON = SC / "larsson_S5.xlsx"
HIC_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/"
           "GSE63525_GM12878_insitu_primary_30.hic")

HG19 = {"chr1": 249250621, "chr2": 243199373, "chr3": 198022430, "chr4": 191154276,
        "chr5": 180915260, "chr6": 171115067, "chr7": 159138663, "chr8": 146364022,
        "chr9": 141213431, "chr10": 135534747, "chr11": 135006516, "chr12": 133851895,
        "chr13": 115169878, "chr14": 107349540, "chr15": 102531392, "chr16": 90354753,
        "chr17": 81195210, "chr18": 78077248, "chr19": 59128983, "chr20": 63025520,
        "chr21": 48129895, "chr22": 51304566, "chrX": 155270560}
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

WBAND = 44                       # bins cached either side of the diagonal (1.10 Mb)
WSCORE = 40                      # bins used by the score (1.00 Mb)
DMIN = 2                         # drop the two bins either side of the diagonal
MIN_ANCHORS = 5                  # a gene needs this many mappable anchor bins to receive a rate
MAP_MIN = 50                     # loop 33's mappability rule: > 50 non-empty entries in the row

LN2 = float(np.log(2.0))
T_DOUBLE_H = 27.5                # loop 91 / loop 98, verified: Schwanhausser NIH3T3 doubling
MRNA_HL_MEDIAN_H = 9.92          # median Schwanhausser mRNA half-life -- an INPUT, one scalar
DELTA_GLOBAL = LN2 / MRNA_HL_MEDIAN_H
PLOIDY = 2.0                     # Larsson is per allele, Schwanhausser per cell (loop 98)
LOOP91_KSM = 1.80                # established median transcription rate, mRNA/cell/h

# thresholds, fixed before any number below was computed
T_X1_MAPCOUNT = 1377             # loop 33's chr21 mappable-bin count
T_X1_RHO = 0.99
T_X3_N = 300
T_X3_P = 0.01
T_X4_RETAIN = 0.50
T_X6_SURVIVE = 0.50
T_X7_REL = 1e-10

NPERM_LABEL = 200
NPERM_POS = 20
SEED = 11401

# simulated-map arm: loop 77's UNTUNED literature defaults, nothing fitted here
SIM_SEP_KB, SIM_RES_S, SIM_V_KB_S = 150.0, 900.0, 0.75
SIM_DT, SIM_NCFG = 1.0, 50
SIM_CHROMS = [("chr21", "hg19_chr21.fa.gz"), ("chr22", "hg19_chr22.fa.gz")]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


# --------------------------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------------------------
def spear(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < 30 or np.std(a[f]) < 1e-12 or np.std(b[f]) < 1e-12:
        return float("nan"), int(f.sum())
    return float(spearmanr(a[f], b[f]).statistic), int(f.sum())


def partial(x, y, z):
    """Spearman of x and y with the rank of z linearly removed from both. Loop 95's estimator."""
    from scipy.stats import spearmanr, rankdata
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    f = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if f.sum() < 30:
        return float("nan")
    rx, ry, rz = rankdata(x[f]), rankdata(y[f]), rankdata(z[f])
    if rz.std() < 1e-12:
        return float(spearmanr(rx, ry).statistic)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return float("nan")
    return float(spearmanr(ex, ey).statistic)


# --------------------------------------------------------------------------------------------
# the measured map, streamed as a near-diagonal band and cached
# --------------------------------------------------------------------------------------------
def fetch_band(chrom):
    """KR-normalised 25 kb observed, kept only within +/-WBAND bins of the diagonal.

    The genome is 6.1 GB dense and this loop never looks further than 1 Mb from a TSS, so the dense
    matrix is never formed. `cov` is the number of non-empty entries in each FULL row, accumulated
    from every record BEFORE the band restriction, so the mappability rule stays loop 33's rule and
    not a band-local imitation of it -- X1 checks exactly that on chr21 against the cached matrix.
    """
    p = SC / f"_band_{chrom}_25kb.npz"
    if p.exists():
        z = np.load(p)
        return {"B": z["B"], "cov": z["cov"], "n": int(z["n"])}
    import hicstraw
    n = HG19[chrom] // BIN + 1
    hic = hicstraw.HiCFile(HIC_URL)
    mzd = hic.getMatrixZoomData(chrom[3:], chrom[3:], "observed", "KR", "BP", BIN)
    recs = mzd.getRecords(0, HG19[chrom] - 1, 0, HG19[chrom] - 1)
    m = len(recs)
    bi = np.fromiter((r.binX for r in recs), np.int64, m) // BIN
    bj = np.fromiter((r.binY for r in recs), np.int64, m) // BIN
    cv = np.fromiter((r.counts for r in recs), np.float64, m)
    del recs, mzd, hic
    ok = (bi >= 0) & (bi < n) & (bj >= 0) & (bj < n) & np.isfinite(cv) & (cv > 0)
    bi, bj, cv = bi[ok], bj[ok], cv[ok]
    cov = (np.bincount(bi, minlength=n) + np.bincount(bj, minlength=n)
           - np.bincount(bi[bi == bj], minlength=n)).astype(np.int32)
    d = np.abs(bi - bj)
    lo = np.minimum(bi, bj)
    s = d <= WBAND
    B = np.zeros((n, WBAND + 1), np.float32)
    B[lo[s], d[s]] = cv[s]
    np.savez_compressed(p, B=B, cov=cov, n=np.int64(n))
    return {"B": B, "cov": cov, "n": n}


def band_expected(B, mask):
    """Mean non-empty contact at each separation over mappable pairs -- loop 33's `expected`, banded."""
    n = len(B)
    exp = np.full(WBAND + 1, np.nan)
    for d in range(WBAND + 1):
        i = np.arange(0, n - d)
        okp = mask[i] & mask[i + d]
        if okp.sum() < 10:
            continue
        v = B[i[okp], d]
        v = v[v > 0]
        if len(v):
            exp[d] = float(v.mean())
    return exp


def anchor_bins(chrom, n):
    """Bins holding at least one GM12878 CTCF peak summit -- the same peak file the arc has used."""
    a = np.zeros(n, bool)
    npk = 0
    for ln in gzip.open(CTCF_BED, "rt"):
        f = ln.split("\t")
        if f[0] != chrom:
            continue
        st, en = int(f[1]), int(f[2])
        off = (int(f[9]) if len(f) > 9 and f[9].strip().lstrip("-").isdigit() and int(f[9]) >= 0
               else (en - st) // 2)
        b = (st + off) // BIN
        if 0 <= b < n:
            a[b] = True
            npk += 1
    return a, npk


def scores_for_bins(bins, OE, mask, anchor, n):
    """THE FORWARD EMISSION, for a vector of TSS bins on one chromosome.

    Everything here is a function of position, the contact band and the anchor track. No expression,
    no abundance, no half-life, no annotation, no gene identity enters. Returns
    S_ctcf, D_ctcf, S_all, D_all.
    """
    ds = np.arange(DMIN, WSCORE + 1)
    S = np.zeros(len(bins))
    D = np.zeros(len(bins))
    SA = np.zeros(len(bins))
    DA = np.zeros(len(bins))
    for k, b in enumerate(bins):
        rp = b + ds
        vr = rp < n
        lp = b - ds
        vl = lp >= 0
        oe = np.concatenate([OE[b, ds[vr]], OE[lp[vl], ds[vl]]])
        pb = np.concatenate([rp[vr], lp[vl]])
        mm = mask[pb]
        aa = anchor[pb] & mm
        oe = np.where(np.isfinite(oe), oe, 0.0)
        S[k] = float(oe[aa].sum())
        D[k] = float(aa.sum())
        SA[k] = float(oe[mm].sum())
        DA[k] = float(mm.sum())
    return S, D, SA, DA


# --------------------------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------------------------
NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan")


def load_larsson():
    """Per-allele steady-state mean kon*ksyn/(kon+koff) by upper-case symbol, averaged over alleles.

    Larsson 2019 fit the two-state telegraph model to allele-resolved scRNA-seq in mouse primary
    fibroblasts; rates are in units of the mRNA degradation rate, so the steady-state mean per allele
    is kon*ksyn/(kon+koff) and the synthesis rate is that times delta. Excel has date-corrupted a
    handful of symbols (SEPT/MARCH/DEC families); those rows are dropped and counted.
    """
    import pandas as pd
    xl = pd.ExcelFile(LARSSON)
    acc, corrupt = {}, 0
    for sheet in ("C57", "CAST"):
        d = xl.parse(sheet)
        col = [c for c in d.columns if "Maximum likelihood" in str(c)][0]
        for g, s in zip(d["Gene"], d[col]):
            if not isinstance(g, str):
                corrupt += 1
                continue
            v = [float(x) for x in NUM.findall(str(s))]
            if len(v) != 3 or not np.all(np.isfinite(v)) or v[0] <= 0 or v[2] <= 0:
                continue
            mean_allele = v[0] * v[2] / (v[0] + v[1])
            if not np.isfinite(mean_allele) or mean_allele <= 0:
                continue
            acc.setdefault(g.upper(), []).append(mean_allele)
    return {g: float(np.mean(v)) for g, v in acc.items()}, corrupt


def steady_state_by_integration(k_sm, k_d, n_steps=4000, efolds=40.0):
    """RK4 forward march of dM/dt = k_sm - k_d M from M(0) = 0, to `efolds` e-foldings.

    Deliberately an INTEGRATION and not the algebraic k_sm/k_d, because gate X7's whole point is to
    show what an integrator does when it is handed an identity-derived k_sm.
    """
    k_sm = np.asarray(k_sm, float)
    k_d = np.asarray(k_d, float)
    M = np.zeros_like(k_sm)
    h = efolds / k_d / n_steps

    def f(M_):
        return k_sm - k_d * M_
    for _ in range(n_steps):
        a = f(M)
        b = f(M + 0.5 * h * a)
        c = f(M + 0.5 * h * b)
        e = f(M + h * c)
        M = M + (h / 6.0) * (a + 2 * b + 2 * c + e)
    return M


def calibrate(x_tr, y_tr, x_ap):
    """ONE global monotone map from contact score to log10 rate, fitted on the training genes only."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(np.asarray(x_tr, float), np.asarray(y_tr, float))
    return np.asarray(iso.predict(np.asarray(x_ap, float)), float), iso


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):        # before int: bool IS an int in Python, and a gate
        return bool(o)                         # verdict serialised as 1/0 is a gate nobody reads
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 114 -- forward link: make the chromatin model EMIT a transcription rate")
    say("=" * 100)
    say()
    say("  INPUTS to the emitter, per gene : hg19 TSS bin, the measured contact band, CTCF anchors")
    say("  INPUTS as global scalars        : delta_global = ln2/%.2f h = %.5f/h, T_double %.1f h, "
        "ploidy %.0f" % (MRNA_HL_MEDIAN_H, DELTA_GLOBAL, T_DOUBLE_H, PLOIDY))
    say("  PREDICTION, per gene            : k_sm_hat, mRNA/cell/h, from a calibration fitted on a")
    say("                                    DISJOINT training half of genes")
    say("  TARGET (the real claim)         : k_sm_larsson -- independent lab, technology, inference")
    say("  TARGET (circular, X7 only)      : k_sm_identity = mrna_copies x k_d, whose own input is")
    say("                                    the gene's mRNA copy number")
    say()

    C = json.load(open(CELL))
    names = [g["name"] for g in C["genes"]]
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}
    tss = {}
    for ln in open(TSS_BED):
        f = ln.split()
        i = int(f[3][1:])
        if i < len(names):
            tss[names[i]] = (f[0], int(f[1]))
    S9 = json.load(open(SCHWAN))
    ident, mrna_obs, mrna_hl = {}, {}, {}
    for g, v in S9.items():
        if v.get("mrna_copies") and v.get("mrna_hl_h") and v["mrna_hl_h"] > 0:
            ident[g] = v["mrna_copies"] * (LN2 / v["mrna_hl_h"] + LN2 / T_DOUBLE_H)
            mrna_obs[g] = float(v["mrna_copies"])
            mrna_hl[g] = float(v["mrna_hl_h"])
    lars_mean, corrupt = load_larsson()
    lars = {g: PLOIDY * m * DELTA_GLOBAL for g, m in lars_mean.items()}
    n_lars_model = len(set(lars) & set(names))
    say(f"  {len(names):,} modelled genes; {len(tss):,} with an hg19 TSS")
    say(f"  Larsson: {len(lars):,} symbols with a telegraph fit ({corrupt} Excel-corrupted rows "
        f"dropped); {n_lars_model:,} match a modelled gene")
    say(f"  Schwanhausser identity rate available for {len(ident):,} genes")
    m_l, m_i = float(np.median(list(lars.values()))), float(np.median(list(ident.values())))
    say(f"  median k_sm_larsson {m_l:.4f} mRNA/cell/h   median k_sm_identity {m_i:.4f}   "
        f"loop 91 established {LOOP91_KSM:.2f}")
    say(f"  the identity median reproduces loop 91 to {abs(m_i/LOOP91_KSM - 1):.2%}, which is the")
    say(f"  provenance check; the Larsson median sits {m_i/m_l:.1f}x BELOW it, because Larsson's")
    say(f"  UMI counts are a captured fraction of the molecules. That is a SCALE offset shared by")
    say(f"  every gene, so it cannot create or destroy rank -- but it means the absolute units of")
    say(f"  the Larsson-calibrated emission are Larsson's, not copies per cell. X7(b) therefore")
    say(f"  integrates with the IDENTITY-calibrated emission, which is in mRNA/cell/h.")
    say()

    # ------------------------------------------------------------------ X1
    say("X1 THE BAND IS THE SAME OBJECT LOOP 85 REPAIRED")
    b21 = fetch_band("chr21")
    mask21 = b21["cov"] > MAP_MIN
    n21 = b21["n"]
    F = np.load(SC / "hic_chr21_25kb.npy").astype(np.float64)
    F[F == 0] = np.nan
    fmask = np.isfinite(F).sum(1) > MAP_MIN
    nf = min(n21, len(F))
    same_mask = int((mask21[:nf] == fmask[:nf]).sum())
    ii, dd = [], []
    for d in range(DMIN, WSCORE + 1):
        i = np.arange(0, nf - d)
        ii.append(i)
        dd.append(np.full(len(i), d))
    ii, dd = np.concatenate(ii), np.concatenate(dd)
    keep = fmask[ii] & fmask[ii + dd]
    a_band = b21["B"][ii[keep], dd[keep]].astype(float)
    a_full = F[ii[keep], ii[keep] + dd[keep]]
    fin = np.isfinite(a_full) & (a_full > 0)
    rho_bf, n_bf = spear(a_band[fin], a_full[fin])
    maxrel = float(np.max(np.abs(a_band[fin] - a_full[fin]) / np.maximum(a_full[fin], 1e-12)))
    say(f"     streamed band chr21: {n21:,} bins, {int(mask21.sum()):,} mappable "
        f"(loop 33 recorded {T_X1_MAPCOUNT:,})")
    say(f"     cached full matrix : {len(F):,} bins, {int(fmask.sum()):,} mappable; "
        f"{same_mask:,}/{nf:,} bins agree on mappability")
    say(f"     band vs full matrix inside the +/-1 Mb window: rho {rho_bf:+.6f} on {n_bf:,} pairs, "
        f"max relative difference {maxrel:.2e}")
    x1 = bool(int(mask21.sum()) == T_X1_MAPCOUNT and int(fmask.sum()) == T_X1_MAPCOUNT
              and rho_bf >= T_X1_RHO)
    say(f"     X1 {'PASS' if x1 else 'FAIL'}")
    if not x1:
        say("     the streamed band is NOT byte-identical to the object loops 33-95 scored on.")
        say("     Everything below is still internally consistent, but its comparability to this")
        say("     arc's recorded numbers is limited and must be read that way.")
    say()
    del F

    # ------------------------------------------------------------------ X2, the emission
    say("X2 THE EMISSION EXISTS, AND ITS COVERAGE IS DECLARED")
    rows = {}
    per_chrom = []
    CH = {}
    for ch in CHROMS:
        gl = sorted(g for g in tss if tss[g][0] == ch)
        if not gl:
            continue
        t1 = time.time()
        try:
            bd = fetch_band(ch)
        except Exception as e:
            say(f"     {ch:6s} FETCH FAILED {repr(e)[:60]}")
            continue
        n = bd["n"]
        mask = bd["cov"] > MAP_MIN
        anchor, npk = anchor_bins(ch, n)
        exp = band_expected(bd["B"], mask)
        with np.errstate(divide="ignore", invalid="ignore"):
            OE = np.asarray(bd["B"], np.float64) / exp[None, :]
        OE[~np.isfinite(OE)] = 0.0
        CH[ch] = {"OE": OE.astype(np.float32), "mask": mask, "anchor": anchor, "n": n, "exp": exp}
        bins = np.array([tss[g][1] // BIN for g in gl])
        ok = (bins >= 0) & (bins < n)
        gl = [g for g, o in zip(gl, ok) if o]
        bins = bins[ok]
        onmap = mask[bins]
        Sc, Dc, Sa, Da = scores_for_bins(bins, CH[ch]["OE"], mask, anchor, n)
        got = 0
        for g, b, om, s_, d_, sa, da in zip(gl, bins, onmap, Sc, Dc, Sa, Da):
            if not om or d_ < MIN_ANCHORS:
                continue
            rows[g] = {"chrom": ch, "bin": int(b), "S": float(s_), "D": float(d_),
                       "Sall": float(sa), "Dall": float(da),
                       "meanOE": float(s_ / d_), "pubs": pubs.get(g, 0.0)}
            got += 1
        per_chrom.append({"chrom": ch, "bins": n, "mappable": int(mask.sum()), "ctcf_peaks": npk,
                          "anchor_bins": int(anchor.sum()), "genes": len(gl), "emitted": got,
                          "seconds": round(time.time() - t1, 1)})
        say(f"     {ch:6s} {n:5,} bins  {int(mask.sum()):5,} mappable  {npk:6,} CTCF peaks in "
            f"{int(anchor.sum()):4,} bins  {got:5,}/{len(gl):5,} genes emitted  "
            f"[{time.time()-t1:5.1f}s]")
        del bd, OE

    G = sorted(rows)
    lars_g = [g for g in G if g in lars]
    ident_g = [g for g in G if g in ident]
    say(f"     {len(G):,} of {len(names):,} modelled genes receive a k_sm from chromatin alone "
        f"({len(G)/len(names):.1%})")
    say(f"     {len(lars_g):,} of {n_lars_model:,} Larsson-rated modelled genes "
        f"({len(lars_g)/max(n_lars_model,1):.1%}) -- the set X3-X6 are scored on")
    say(f"     {len(ident_g):,} also carry loop 91's identity rate")
    say(f"     X2 PASS (reported)")
    say()

    if len(lars_g) < 60:
        say("  too few emitted Larsson genes to continue; stopping and recording the failure.")
        json.dump(jsonable({"test": "loop_chromatin_to_rate", "aborted": True,
                            "n_larsson_emitted": len(lars_g), "log": log}),
                  open(OUT / "loop_chromatin_to_rate.json", "w"), indent=1)
        return

    # ------------------------------------------------------------------ split + calibration
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(lars_g))
    half = len(lars_g) // 2
    tr = [lars_g[i] for i in perm[:half]]
    te = [lars_g[i] for i in perm[half:]]
    Str = np.array([rows[g]["S"] for g in tr])
    Ytr = np.log10([lars[g] for g in tr])
    Ste = np.array([rows[g]["S"] for g in te])
    Yte = np.array([lars[g] for g in te])
    yhat_log, iso = calibrate(Str, Ytr, Ste)
    khat = 10.0 ** yhat_log

    tr_id = [g for g in tr if g in ident]
    te_id = [g for g in te if g in ident]
    khat_id = 10.0 ** calibrate([rows[g]["S"] for g in tr_id],
                                np.log10([ident[g] for g in tr_id]),
                                [rows[g]["S"] for g in te_id])[0]

    say("   THE CALIBRATION")
    say(f"     random 50/50 gene split, seed {SEED}: {len(tr):,} training, {len(te):,} held out")
    say(f"     isotonic S_ctcf -> log10 k_sm, fitted on the training half ONLY, applied unchanged")
    say(f"     S_ctcf spans {Str.min():.1f} to {Str.max():.1f} on the training half")
    say(f"     emitted k_sm_hat median {np.median(khat):.3f} mRNA/cell/h   target median "
        f"{np.median(Yte):.3f}   loop 91 established {LOOP91_KSM:.2f}")
    say(f"     NOTE: a monotone calibration cannot change ranks, so the held-out rank result below")
    say(f"     is the SCORE's; the calibration is what turns it into a number in mRNA/cell/h, and")
    say(f"     is what X7(b) tests.")
    say()

    # ------------------------------------------------------------------ X3
    say("X3 HELD OUT: THE EMITTED RATE PREDICTS THE INDEPENDENT RATE")
    rho_te, n_te = spear(khat, Yte)
    rng2 = np.random.default_rng(SEED + 1)
    perm_rho = np.array([spear(khat, rng2.permutation(Yte))[0] for _ in range(NPERM_LABEL)])
    p_lab = float((np.sum(np.abs(perm_rho) >= abs(rho_te)) + 1) / (NPERM_LABEL + 1))
    say(f"     k_sm_hat vs k_sm_larsson, HELD-OUT genes    rho {rho_te:+.4f}   n {n_te:,}")
    say(f"     label permutation, {NPERM_LABEL} draws: |rho| null {np.abs(perm_rho).mean():.4f} "
        f"+/- {np.abs(perm_rho).std():.4f}   p {p_lab:.4f}")
    idr, idn = spear([rows[g]["S"] for g in te_id], [ident[g] for g in te_id])
    say(f"     the SAME score against the ABUNDANCE-DERIVED identity rate   rho {idr:+.4f}  "
        f"n {idn:,}")
    say(f"       -- reported for comparison only. k_sm_identity's own input is the gene's mRNA copy")
    say(f"          number, so it does not carry the claim.")
    say(f"     NOT directly comparable to loop 95's +0.1545: that was raw summed contact against the")
    say(f"     identity rate on 3,358 genes with no train/test split. This is an O/E score at CTCF")
    say(f"     anchors against an independent rate on held-out genes. The identity-target number")
    say(f"     above is the closest thing to loop 95's quantity that this loop computes.")
    x3 = bool(n_te >= T_X3_N and np.isfinite(rho_te) and rho_te > 0 and p_lab < T_X3_P)
    say(f"     X3 {'PASS' if x3 else 'FAIL'}")
    say()

    # ------------------------------------------------------------------ X4
    say("X4 IT BEATS FAME")
    Pte = np.array([np.log1p(rows[g]["pubs"]) for g in te])
    rho_pubs, _ = spear(Pte, Yte)
    pr = partial(Ste, Yte, Pte)
    retain = pr / rho_te if (np.isfinite(rho_te) and abs(rho_te) > 1e-9) else float("nan")
    say(f"     pubs vs k_sm_larsson, held out             rho {rho_pubs:+.4f}")
    say(f"     chromatin emission                         rho {rho_te:+.4f}")
    say(f"     chromatin given pubs                       partial {pr:+.4f}")
    say(f"     retained fraction of the raw effect        "
        + (f"{retain:.1%}" if np.isfinite(retain) else "UNDEFINED -- no raw effect to retain"))
    x4 = bool(np.isfinite(rho_te) and np.isfinite(rho_pubs) and rho_te > rho_pubs
              and np.isfinite(retain) and retain >= T_X4_RETAIN)
    say(f"     X4 {'PASS' if x4 else 'FAIL'}")
    say()

    # ------------------------------------------------------------------ X5
    say("X5 IT BEATS THE DISTANCE-ONLY NULL")
    Dte = np.array([rows[g]["D"] for g in te])
    MOEte = np.array([rows[g]["meanOE"] for g in te])
    Allte = np.array([rows[g]["Sall"] for g in te])
    rho_d, _ = spear(Dte, Yte)
    rho_moe, _ = spear(MOEte, Yte)
    rho_all, _ = spear(Allte, Yte)
    p_sd = partial(Ste, Yte, Dte)
    say(f"     S_ctcf  summed O/E at CTCF anchors within 1 Mb          rho {rho_te:+.4f}")
    say(f"     D_ctcf  THE DISTANCE-ONLY NULL -- the same sum with O/E set to 1, i.e. how many")
    say(f"             mappable CTCF-anchored bins lie within 1 Mb     rho {rho_d:+.4f}")
    say(f"     S_ctcf given D_ctcf                                     partial {p_sd:+.4f}")
    say(f"     mean O/E at anchors, the counting removed               rho {rho_moe:+.4f}")
    say(f"     S_all   summed O/E over ALL mappable bins in 1 Mb       rho {rho_all:+.4f}")
    say(f"     (the distance-only null is one deterministic number, not a distribution, so")
    say(f"      GG.survival does not apply to it -- the comparison is the direct one above)")
    x5 = bool(np.isfinite(rho_te) and np.isfinite(rho_d) and rho_te > rho_d
              and np.isfinite(p_sd) and p_sd > 0)
    say(f"     X5 {'PASS' if x5 else 'FAIL'}")
    say()

    # ------------------------------------------------------------------ X6
    say("X6 THE SHUFFLED-POSITION NULL, VERIFIED CAPABLE BEFORE IT COUNTS")
    by_ch = {}
    for g in G:
        by_ch.setdefault(rows[g]["chrom"], []).append(g)
    null_rhos = []
    null_rhos_id = []
    Yid_te = np.array([ident[g] for g in te_id])
    Ytr_id = np.log10([ident[g] for g in tr_id])
    cap = None
    rng3 = np.random.default_rng(SEED + 2)
    for _ in range(NPERM_POS):
        Sn = {}
        for ch, gs in by_ch.items():
            bins = rng3.permutation(np.array([rows[g]["bin"] for g in gs]))
            s_, _, _, _ = scores_for_bins(bins, CH[ch]["OE"], CH[ch]["mask"],
                                          CH[ch]["anchor"], CH[ch]["n"])
            for g, sv_ in zip(gs, s_):
                Sn[g] = float(sv_)
        if cap is None:
            cap = GG.null_can_move([round(rows[g]["S"], 9) for g in G],
                                   [round(Sn[g], 9) for g in G])
            say(f"     GG.null_can_move: {cap['changed']:.1%} of the emitted scores change -- "
                f"{'CAPABLE' if cap['capable'] else 'INERT'}")
            say(f"       {cap['reason']}")
            if not cap["capable"]:
                say("     this null cannot move the statistic, so its output is NOT evidence and is")
                say("     not reported as a survival fraction.")
                break
        yh, _ = calibrate([Sn[g] for g in tr], Ytr, [Sn[g] for g in te])
        r_, _ = spear(10.0 ** yh, Yte)
        null_rhos.append(r_)
        yhi, _ = calibrate([Sn[g] for g in tr_id], Ytr_id, [Sn[g] for g in te_id])
        ri_, _ = spear(10.0 ** yhi, Yid_te)
        null_rhos_id.append(ri_)
    if null_rhos:
        sv = GG.survival(rho_te, null_rhos)
    else:
        sv = {"real": float(rho_te), "n_null": 0, "null_mean": float("nan"),
              "null_sd": float("nan"), "defined": False, "z": float("nan"),
              "fraction": GG.UNDEFINED, "reason": "the null was inert and was not run"}
    GG.report("TSS positions shuffled within chromosome", sv, emit=say)
    x6 = bool(cap and cap["capable"] and sv.get("defined")
              and sv["fraction"] != GG.UNDEFINED and sv["fraction"] < T_X6_SURVIVE)
    say(f"     X6 {'PASS' if x6 else 'FAIL'}")
    say()

    # ---------------------------------------- added after the first run; no threshold moved
    say("   WHAT X5's PASS IS WORTH  [written after the first run, disclosed in the docstring]")
    x5_vacuous = bool(not sv.get("defined"))
    say(f"     X5 fired: {rho_te:+.4f} beats {rho_d:+.4f} with a positive partial, and by the rule")
    say(f"     predeclared above that is a PASS. X6 has just shown that {rho_te:+.4f} sits "
        f"{abs(sv.get('z', float('nan'))):.2f} null")
    say(f"     standard deviations from zero. So X5 compared two numbers that are both")
    say(f"     indistinguishable from nothing -- gate_guard's FAMILY ONE, the fault behind loop 87")
    say(f"     C6's '54% survival' and loop 87b B6's '6718%'. X5 is recorded as PASS because that")
    say(f"     was the rule, and simultaneously as VACUOUS: {'VACUOUS' if x5_vacuous else 'not vacuous'}.")
    say()

    say("   THE SECOND TARGET GETS THE SAME NULL  [added after the first run]")
    sv_id = GG.survival(idr, null_rhos_id) if null_rhos_id else {"defined": False,
                                                                 "fraction": GG.UNDEFINED,
                                                                 "reason": "null not run",
                                                                 "real": idr, "n_null": 0,
                                                                 "null_mean": float("nan"),
                                                                 "null_sd": float("nan"),
                                                                 "z": float("nan")}
    say(f"     the same score against the ABUNDANCE-DERIVED identity rate, {idn:,} held-out genes:")
    GG.report("TSS positions shuffled, identity target", sv_id, emit=say)
    say(f"     so the contact score DOES carry rank about the rate derived from Schwanhausser's own")
    say(f"     copy numbers, and does NOT carry rank about the independently measured one. Loop 95")
    say(f"     only ever tested the first of those two.")
    say()

    say("   IS THE LARSSON TARGET SIMPLY NOISE?  [added after the first run]")
    say(f"     if it were, nothing could predict it and this loop's negative would say nothing.")
    say(f"     pubs, on exactly these held-out genes, reaches rho {rho_pubs:+.4f}. A target that a")
    say(f"     publication count predicts at {rho_pubs:+.3f} is not noise-dominated, so chromatin's")
    say(f"     {rho_te:+.4f} is a failure of the feature and not a failure of the answer key.")
    say()

    # ------------------------------------------------------------------ stricter split
    say("   STRICTER SPLIT: TRAIN AND TEST ON DISJOINT CHROMOSOMES")
    odd = {c for c in CHROMS if c != "chrX" and int(c[3:]) % 2 == 1}
    tr2 = [g for g in lars_g if rows[g]["chrom"] in odd]
    te2 = [g for g in lars_g if rows[g]["chrom"] not in odd]
    if len(tr2) >= 100 and len(te2) >= 100:
        yh2, _ = calibrate([rows[g]["S"] for g in tr2], np.log10([lars[g] for g in tr2]),
                           [rows[g]["S"] for g in te2])
        Y2 = np.array([lars[g] for g in te2])
        rho2, n2 = spear(10.0 ** yh2, Y2)
        rho2d, _ = spear([rows[g]["D"] for g in te2], Y2)
        rho2p, _ = spear([np.log1p(rows[g]["pubs"]) for g in te2], Y2)
        say(f"     trained on odd autosomes ({len(tr2):,} genes), tested on even autosomes + chrX "
            f"({len(te2):,} genes)")
        say(f"     chromatin {rho2:+.4f}   distance-only null {rho2d:+.4f}   pubs {rho2p:+.4f}")
    else:
        rho2 = rho2d = rho2p = float("nan")
        n2 = 0
        say("     too few genes on one side; not run")
    say()

    # ------------------------------------------------------------------ the model's own map
    say("   THE MODEL'S OWN MAP: the same emission run on the SIMULATED contact map")
    say(f"     loop-extrusion at loop 77's UNTUNED literature point (separation {SIM_SEP_KB:.0f} kb,")
    say(f"     residence {SIM_RES_S:.0f} s, speed {SIM_V_KB_S:.2f} kb/s), nothing fitted here")
    sim = {"genes": 0}
    try:
        import loop_map_score as L79
        sim_rows = {}
        for ch, fa in SIM_CHROMS:
            t1 = time.time()
            Cc = L79.build_chrom(ch, fa)
            bf, br = L79.landscape(Cc, Cc["orients"])
            R = L79.run_point(Cc, bf, br, SIM_SEP_KB, SIM_RES_S, SIM_V_KB_S,
                              SIM_DT, SIM_NCFG, SEED)
            Msim, esim = R["M"], R["exp"]
            n = Cc["n"]
            mask = CH[ch]["mask"][:n] if ch in CH else Cc["mask"]
            anchor = CH[ch]["anchor"][:n] if ch in CH else anchor_bins(ch, n)[0]
            Bsim = np.zeros((n, WBAND + 1))
            for d in range(WBAND + 1):
                i = np.arange(0, n - d)
                Bsim[i, d] = Msim[i, i + d]
            esb = np.array([esim[d] if d < len(esim) else np.nan for d in range(WBAND + 1)])
            with np.errstate(divide="ignore", invalid="ignore"):
                OEs = Bsim / esb[None, :]
            OEs[~np.isfinite(OEs)] = 0.0
            gs = [g for g in lars_g if rows[g]["chrom"] == ch]
            bins = np.array([rows[g]["bin"] for g in gs])
            s_, d_, _, _ = scores_for_bins(bins, OEs, mask, anchor, n)
            for g, a_, b_ in zip(gs, s_, d_):
                sim_rows[g] = {"S": float(a_), "D": float(b_)}
            say(f"     {ch}: {n:,} bins simulated, P(s) {R['ps']:+.4f}, "
                f"{len(gs):,} Larsson-rated genes here   [{time.time()-t1:.0f}s]")
            del Msim, Bsim, OEs
        gsim = sorted(sim_rows)
        if len(gsim) >= 60:
            Ys = np.array([lars[g] for g in gsim])
            rs, ns = spear([sim_rows[g]["S"] for g in gsim], Ys)
            rmeas, _ = spear([rows[g]["S"] for g in gsim], Ys)
            rdn, _ = spear([rows[g]["D"] for g in gsim], Ys)
            rp, _ = spear([np.log1p(rows[g]["pubs"]) for g in gsim], Ys)
            yh_s, _ = calibrate(Str, Ytr, [sim_rows[g]["S"] for g in gsim])
            say(f"     on the SAME {len(gsim):,} chr21+chr22 genes, against k_sm_larsson:")
            say(f"       SIMULATED map score   rho {rs:+.4f}   <- the chromatin MODEL emitting a rate")
            say(f"       MEASURED map score    rho {rmeas:+.4f}")
            say(f"       distance-only null    rho {rdn:+.4f}")
            say(f"       pubs                  rho {rp:+.4f}")
            say(f"       n is {len(gsim)}, which is small; this arm is a direction check, not a")
            say(f"       measurement, and is not one of the seven gates.")
            sim = {"genes": len(gsim), "n_scored": ns, "rho_sim": rs, "rho_meas": rmeas,
                   "rho_dist": rdn, "rho_pubs": rp,
                   "median_khat_from_sim": float(np.median(10.0 ** yh_s))}
        else:
            say(f"     only {len(gsim)} genes -- too few to report a correlation")
            sim = {"genes": len(gsim)}
    except Exception as e:
        say(f"     SIMULATED ARM FAILED: {repr(e)[:140]}")
        sim = {"genes": 0, "error": repr(e)[:200]}
    say()

    # ------------------------------------------------------------------ X7
    say("X7 THE CIRCULAR ARM, RUN ON PURPOSE, AND THE HONEST ONE BESIDE IT")
    gid = sorted(ident)
    ksm_id = np.array([ident[g] for g in gid])
    kd_id = np.array([LN2 / mrna_hl[g] + LN2 / T_DOUBLE_H for g in gid])
    Mobs = np.array([mrna_obs[g] for g in gid])
    Mstar = steady_state_by_integration(ksm_id, kd_id)
    rel = np.abs(Mstar - Mobs) / np.maximum(Mobs, 1e-30)
    alg = np.abs(ksm_id / kd_id - Mobs) / np.maximum(Mobs, 1e-30)
    say(f"     (a) CIRCULAR BY CONSTRUCTION. k_sm = mrna_copies x k_d, so M* = k_sm/k_d is the")
    say(f"         INPUT copy number. RK4 forward march to 40 e-foldings, {len(gid):,} genes:")
    say(f"           median relative error {np.median(rel):.3e}   max {np.max(rel):.3e}   "
        f"fraction <= {T_X7_REL:.0e}: {np.mean(rel <= T_X7_REL):.2%}")
    say(f"           the ALGEBRAIC identity |k_sm/k_d - mrna_copies|/mrna_copies: median "
        f"{np.median(alg):.3e}")
    say(f"         THIS IS AN IDENTITY, NOT A SIMULATION. It reproduces the measured steady state")
    say(f"         because the measured steady state is its input. It is EXCLUDED from X3-X6, from")
    say(f"         the simulated arm and from the headline. Loop 101's D5 is the template.")
    x7a = bool(np.median(rel) <= T_X7_REL)
    say(f"         (a) {'demonstrates the tautology as intended' if x7a else 'DID NOT REPRODUCE -- the integrator is broken'}")
    say()
    kmap = {g: float(k) for g, k in zip(te_id, khat_id)}
    ghon = [g for g in te_id if g in mrna_obs]
    kh = np.array([kmap[g] for g in ghon])
    kdh = np.array([LN2 / mrna_hl[g] + LN2 / T_DOUBLE_H for g in ghon])
    Mo = np.array([mrna_obs[g] for g in ghon])
    Mh = steady_state_by_integration(kh, kdh)
    with np.errstate(divide="ignore", invalid="ignore"):
        lf = np.log10(np.maximum(Mh, 1e-30) / np.maximum(Mo, 1e-30))
    lf = lf[np.isfinite(lf)]
    fold = float(10.0 ** np.median(np.abs(lf))) if len(lf) else float("nan")
    rho_ms, nms = spear(Mh, Mo)
    say(f"     (b) THE HONEST ARM. Same ODE, same measured half-lives, but k_sm comes from the")
    say(f"         chromatin emission calibrated on DISJOINT genes. The held-out gene's own mRNA")
    say(f"         copy number enters nothing on the prediction side -- it is only the answer key.")
    say(f"           {len(ghon):,} held-out genes with a chromatin rate and a measured copy number")
    say(f"           median fold error of M* against the observed copy number   {fold:.2f}x")
    say(f"           within 2x {np.mean(np.abs(lf) <= np.log10(2)):.1%}   "
        f"within 10x {np.mean(np.abs(lf) <= 1.0):.1%}")
    say(f"           Spearman M* vs observed copies   {rho_ms:+.4f}   n {nms:,}")
    say(f"         The circular arm's error is {np.median(rel):.1e}. This one's is {fold:.2f}x.")
    say(f"         That gap is what the word 'prediction' costs, and it is the point of X7.")
    say()
    kconst = np.full(len(ghon), float(np.median(kh)))
    Mc = steady_state_by_integration(kconst, kdh)
    with np.errstate(divide="ignore", invalid="ignore"):
        lfc = np.log10(np.maximum(Mc, 1e-30) / np.maximum(Mo, 1e-30))
    lfc = lfc[np.isfinite(lfc)]
    foldc = float(10.0 ** np.median(np.abs(lfc))) if len(lfc) else float("nan")
    rho_mc, _ = spear(Mc, Mo)
    say(f"     (c) THE CONTROL X7(b) NEEDS  [added after the first run; no threshold moved].")
    say(f"         M* = k_sm/k_d and k_d is built from the gene's OWN measured mRNA half-life, so")
    say(f"         a good-looking M* could be entirely the half-life. Same ODE, same k_d, but")
    say(f"         k_sm held CONSTANT at the median emitted rate {float(np.median(kh)):.3f}:")
    say(f"           median fold error {foldc:.2f}x   (chromatin-driven: {fold:.2f}x)")
    say(f"           Spearman M* vs observed copies {rho_mc:+.4f}   (chromatin-driven: "
        f"{rho_ms:+.4f})")
    say(f"         The chromatin emission's entire contribution to M* is its rank agreement with")
    say(f"         k_sm_identity = M_obs x k_d, which is the {idr:+.4f} reported in X3. Everything")
    say(f"         else in {rho_ms:+.4f} is 1/k_d, i.e. the measured half-life, not the contact map.")
    x7 = bool(x7a)
    say(f"     X7 {'PASS' if x7 else 'FAIL'}")
    say()

    gates = {"X1 the streamed band reproduces the corrected full-matrix pipeline": x1,
             "X2 emission coverage declared": True,
             "X3 held-out emission predicts the independent Larsson rate": x3,
             "X4 beats fame": x4,
             ("X5 beats the distance-only null [PASS BUT VACUOUS -- see X6]"
              if x5_vacuous else "X5 beats the distance-only null"): x5,
             "X6 shuffled-position null fires, and was verified capable first": x6,
             "X7 the tautology is demonstrated on purpose and excluded": x7}
    say("  " + "-" * 96)
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")
    say()

    man = RM.manifest(
        inputs=[str(CELL), str(TSS_BED), str(CTCF_BED), str(SCHWAN), str(LARSSON),
                str(SC / "hic_chr21_25kb.npy"), HIC_URL],
        available=n_lars_model, used=len(lars_g), selection="all", seed=SEED,
        controls=["train/test split on genes: the calibration never sees a held-out gene",
                  "chromosome-disjoint split reported as a stricter second split",
                  "distance-only null: the identical score with O/E set to 1",
                  "shuffled-TSS-position null, GG.null_can_move verified before it counts",
                  "GG.survival used so an undefined survival cannot print as a percentage",
                  "fame (pubs) reported on every arm",
                  "the circular identity-k_sm arm run on purpose and excluded from every claim",
                  "the emitter sees no expression, abundance, half-life, pubs or gene identity"],
        note="loop 95 correlated chromatin with rate after the fact; this emits a rate from the "
             "contact map through one global monotone calibration fitted on a disjoint half of "
             "genes, and tests it on the other half against an independently measured rate")
    RM.report(man, emit=say)

    res = {"test": "loop_chromatin_to_rate", "manifest": man, "gates": gates,
           "declaration": {
               "per_gene_inputs_to_emitter": ["hg19 TSS bin", "measured Hi-C band (KR, 25 kb)",
                                              "GM12878 CTCF anchor bins"],
               "global_scalar_inputs": {"delta_global_per_h": DELTA_GLOBAL,
                                        "mrna_hl_median_h": MRNA_HL_MEDIAN_H,
                                        "T_double_h": T_DOUBLE_H, "ploidy": PLOIDY},
               "per_gene_prediction": "k_sm_hat, mRNA/cell/h",
               "targets_never_seen_by_emitter": {
                   "k_sm_larsson": "INDEPENDENT -- Larsson 2019 telegraph fit x delta_global",
                   "k_sm_identity": "CIRCULAR w.r.t. mRNA copy number -- loop 91 identity, X7 only"}},
           "x1": {"band_mappable_chr21": int(mask21.sum()), "full_mappable_chr21": int(fmask.sum()),
                  "expected_mappable": T_X1_MAPCOUNT, "rho_band_vs_full": rho_bf,
                  "n_pairs": n_bf, "max_rel_diff": maxrel, "bins_agreeing_on_mappability": same_mask},
           "coverage": {"modelled_genes": len(names), "with_tss": len(tss), "emitted": len(G),
                        "emitted_fraction": len(G) / len(names),
                        "larsson_modelled": n_lars_model, "larsson_emitted": len(lars_g),
                        "identity_emitted": len(ident_g)},
           "per_chrom": per_chrom,
           "split": {"seed": SEED, "n_train": len(tr), "n_test": len(te)},
           "x3": {"rho_heldout": rho_te, "n": n_te, "p_label_perm": p_lab,
                  "null_abs_rho_mean": float(np.abs(perm_rho).mean()),
                  "rho_vs_identity_target": idr, "n_identity": idn,
                  "median_khat": float(np.median(khat)), "median_target": float(np.median(Yte))},
           "x4": {"rho_pubs": rho_pubs, "partial_given_pubs": pr, "retained": retain},
           "x5": {"rho_contact": rho_te, "rho_distance_only": rho_d,
                  "partial_given_distance_only": p_sd, "rho_mean_oe": rho_moe,
                  "rho_all_bins": rho_all, "vacuous": x5_vacuous,
                  "vacuous_reason": "X6 shows the real value is inside the null's spread, so this "
                                    "gate compared two numbers both indistinguishable from zero "
                                    "-- gate_guard FAMILY ONE"},
           "x6": {"null_can_move": cap, "survival": sv, "n_perm": len(null_rhos),
                  "null_rhos": [float(x) for x in null_rhos],
                  "survival_identity_target": sv_id,
                  "null_rhos_identity": [float(x) for x in null_rhos_id]},
           "target_is_not_noise": {"rho_pubs_on_larsson_heldout": rho_pubs,
                                   "note": "a target a publication count predicts this well is not "
                                           "noise-dominated, so the chromatin negative is a "
                                           "property of the feature"},
           "chrom_disjoint": {"n_test": n2, "rho_chromatin": rho2, "rho_distance_only": rho2d,
                              "rho_pubs": rho2p},
           "simulated_arm": sim,
           "x7": {"circular_arm": {"n": len(gid), "median_rel_err": float(np.median(rel)),
                                   "max_rel_err": float(np.max(rel)),
                                   "median_algebraic_rel_err": float(np.median(alg)),
                                   "fraction_within_tol": float(np.mean(rel <= T_X7_REL)),
                                   "label": "CIRCULAR BY CONSTRUCTION -- excluded from every claim"},
                  "honest_arm": {"n": len(ghon), "median_fold_error": fold,
                                 "within_2x": float(np.mean(np.abs(lf) <= np.log10(2))),
                                 "within_10x": float(np.mean(np.abs(lf) <= 1.0)),
                                 "spearman_Mstar_vs_observed": rho_ms},
                  "constant_ksm_control": {"n": len(ghon), "k_sm_const": float(np.median(kh)),
                                           "median_fold_error": foldc,
                                           "spearman_Mstar_vs_observed": rho_mc,
                                           "note": "same ODE, same measured half-lives, chromatin "
                                                   "removed. What survives here is the half-life, "
                                                   "not the contact map."}},
           "seconds": time.time() - t0, "log": log}
    json.dump(jsonable(res), open(OUT / "loop_chromatin_to_rate.json", "w"), indent=1)
    say(f"  -> {OUT / 'loop_chromatin_to_rate.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
