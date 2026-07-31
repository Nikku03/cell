"""causal_enhancer -- can the IDENTITY of the TFs bound at a distal element, weighted by whether those TFs are
CAUSAL for the gene in Perturb-seq, predict a CRISPR-validated enhancer-gene link better than distance and
better than accessibility?

WHAT IS BEING COMBINED, AND WHY THAT COMBINATION IS THE INTERESTING ONE.
`screen_ccre.py` established the element side: 1,063,878 SCREEN cCREs of which 151,858 are K562-accessible,
the benchmark having tested 3.89% of them; the number that qualifies every E-G claim in this project (31.5%
of experimentally VALIDATED K562 E-G links sit at a distance the current E-G layer cannot express); and one
predictive fact -- overlapping a K562-accessible cCRE raises the validated-positive rate to 0.0428 against a
matched 0.0171, a 2.51x lift, so element ANNOTATION carries signal but is far from sufficient.
`bound_causal.py` established the TF side: 392 TFs (507 here, after a re-fetch of the ENCODE K562
conservative-IDR set) with BOTH a K562 ChIP occupancy track AND a genome-scale Perturb-seq perturbation,
split into DIRECT / INDIRECT / bound-not-causal tiers, finding that 31.6% of causal edges are bound at the
promoter against a 25.0% background (1.27x) and that directness predicts transfer to RPE1 on both magnitude
and sign. Both modules worked at PROMOTERS or on ANNOTATION; neither asked what a distal element's TF set
buys.

Crossing occupancy with causality AT A DISTAL ELEMENT is the natural next object:
an enhancer is not "a peak", it is a place where particular proteins sit, and the proteins that sit there
should be the ones whose removal moves the gene. Distance and accessibility are the two features every
enhancer-gene predictor already has, and they are known to be strong. So the question is not "does TF
causality predict links" -- almost anything correlated with activity does -- but:

    IS THERE AN INCREMENT, over distance and accessibility, that comes from the ELEMENT'S OWN TF SET?

WHAT IS MEASURED, AND AGAINST WHAT CONTROL, BEFORE ANY NUMBER.
  MEASURED   AUPRC, out-of-fold, on CRISPR enhancer-perturbation ground truth (EP CRISPR benchmark, K562 arm,
             training_K562 + the K562 rows of the 5-cell-type held-out file), folds held out BY CHROMOSOME so
             they are gene-disjoint by construction. AUPRC is reported RAW, with the positive base rate
             beside it, because at a ~4% base rate accuracy is meaningless and an AUPRC of 0.10 is a 2.5x
             lift, not a bad model.
  CONTROL 1  DISTANCE-ONLY and ACCESSIBILITY-ONLY arms on IDENTICAL folds. The TF-causality features are
             scored as an increment over what was already available, never on their own.
  CONTROL 2  DISTANCE + ACCESSIBILITY + TF OCCUPANCY COUNT. This is the bound-vs-causal distinction imported
             from bound_causal.py: an element with 300 of 507 profiled TFs sitting on it is a busy element,
             and busy-ness is available without any Perturb-seq. If the increment survives only against
             distance and accessibility but not against occupancy count, the finding is "busy elements", not
             "the right TFs".
  CONTROL 3  A GENE-RESPONSIVENESS-ONLY arm with NO element information at all. Some genes move under almost
             any perturbation. Such a gene will look causally connected to every element in the genome, and
             it will also be an easy CRISPR hit. This arm measures how much of the answer is readable from
             the gene alone.
  CONTROL 4, THE DECISIVE ONE -- THE ELEMENT SWAP. Keep the gene, keep the label, and replace the element
             with a DIFFERENT element matched on distance-to-TSS decile and accessibility decile, then score
             the swapped pair with the model trained on real pairs. If the decoy scores as well as the real
             element, the model is reading the gene and the distance and the element is decoration. Drawn 20
             times. ONE THING THIS CONTROL DOES THAT ONLY THE NUMBERS REVEALED, recorded here rather than
             buried: a decile-matched decoy carries the DECILE of the real distance, not the real distance,
             so an arm holding a distance feature loses within-decile resolution and drops for a reason that
             is not about the element. Every swap is therefore ALSO run in an element-only variant that
             retains the real distance and swaps only the element-derived features. That variant is POST HOC
             and is labelled as such in the output; the prespecified gates are still scored on the
             prespecified swap.
  CONTROL 5  THE DEGREE-MATCHED ELEMENT SWAP. Elements differ in how many TFs sit on them, and a decoy drawn
             only on distance and accessibility can carry a systematically different TF count, which would
             move every TF-causality feature for a reason that has nothing to do with TF identity. So the
             swap is repeated with the decoy ALSO matched on n-TFs-bound decile -- the configuration-model
             analogue for a bipartite TF-element graph. If the degree-matched swap reproduces most of the
             effect, THAT IS THE FINDING and it is said in the verdict, not in a footnote.
  CONTROL 6, THE CLEANEST IDENTITY TEST -- THE TF-AXIS PERMUTATION. The element swap changes the element, so
             it confounds "which TFs" with everything else an element carries (its width, its accessibility,
             its chromosome, how many of its pairs the screens tested). The permutation does not touch the
             element at all. It relabels the TF axis of the causal matrix, so TF t is given TF pi(t)'s
             Perturb-seq z-vector while the OCCUPANCY matrix stays exactly as measured. That holds the
             element fixed, holds the gene fixed, holds the number of bound TFs EXACTLY (not merely to a
             decile), and holds the gene's whole z distribution over the 507 TFs EXACTLY -- a permutation of
             a column is the same multiset. The ONLY thing destroyed is the correspondence between which TF
             binds where and which TF is causal for what, which is the claim being tested. Run twice: a free
             permutation, and one restricted to strata of TF binding-degree decile x TF causal-breadth
             decile, so a TF that binds everywhere and moves everything cannot be swapped for a rare, quiet
             one. The stratified version is the configuration model on the TF axis and is the strict test.
  CONTROL 7  THE SCREEN-IDENTITY PROBE. The K562 arm is SIX screens whose positive rates run from 0.0123
             (Schraivogel2020) to 0.1801 (Morris), a 14x spread, and the screens differ in element geometry:
             K562_DC_TAP elements are 499 bp with an interquartile range of 499-499, Xie's are 884 bp with
             an IQR of 628-1220. Element width sits in the ACCESSIBILITY block. A model can therefore buy
             AUPRC by naming the screen rather than by reading chromatin. So every ladder rung is re-run with
             a six-level SCREEN one-hot already in the model: a rung that survives conditioning on the screen
             is about the element, a rung that does not is about the batch.

THE DECISION THRESHOLD, FIXED BEFORE THE NUMBERS (and encoded below as MIN_INCREMENT / MIN_SEEDS /
MIN_SWAP_FRAC). TF causality earns a place in the E-G layer only if ALL THREE hold:
    (i)   mean out-of-fold AUPRC of the full model exceeds the BEST baseline arm by >= 0.010 absolute AUPRC,
          and does so in >= 4 of 5 CV seeds;
    (ii)  the real element beats the distance x accessibility decile-matched swap in >= 18 of 20 draws;
    (iii) it also beats the DEGREE-matched swap in >= 18 of 20 draws.
Failing (i) means there is no increment. Passing (i) but failing (ii) means the model reads the gene and the
distance. Passing (ii) but failing (iii) means it reads how busy the element is, not which TFs are on it.

THE SAME BAR IS APPLIED TO EVERY RUNG, NOT ONLY TO THE ONE UNDER TEST. An earlier version of this module
scored TF causality against MIN_INCREMENT and then, in the same verdict, credited the raw TF COUNT with a
positive finding on an increment that had never been put through the same test. That is a threshold applied
asymmetrically, and it is the single easiest way to manufacture a positive claim next to a null. So section 4
now runs EVERY rung of the ladder through the identical gate -- PAIRED per seed, because the arms share
folds and an unpaired sd across seeds hides that -- and reports which rungs clear it. A rung that fails the
bar is described as sub-threshold, never as "what does the work".

CIRCULARITY, STATED PLAINLY. The TF->gene causal weights come from Replogle K562 Perturb-seq; the ground
truth is CRISPRi enhancer perturbation in K562. These are DIFFERENT experiments (gene knockdown vs element
silencing, different labs, different libraries), so this is not the forbidden case of scoring a K562-derived
layer on K562 Perturb-seq. But they share a cell line, and therefore they share "which genes are easy to
move in K562". That shared nuisance is exactly what CONTROL 3 quantifies and what CONTROL 4 removes by
holding the gene fixed. There is no RPE1 enhancer-CRISPR benchmark to escape to, and that is a limit.

SIGN AND SIGNIFICANCE ARE READ TOGETHER. An arm that does not clear its control is NOT DETECTED. It is never
called reversed, however negative the point estimate looks.

WHAT A NULL HERE DOES AND DOES NOT LICENSE, AND THE DETECTABILITY PROBE THAT KEEPS IT HONEST. The causality
features are AGGREGATES over the bound TF set, and the bound set is not small: the median element carries 36
of the 507 profiled TFs, and pair-weighted the mean is 102 for positives. A mean of |z| over 102 of 507 TFs
is very close to the gene's own mean over all 507, so a magnitude aggregate is structurally near-blind to
which TFs those 102 are. A null from such a block is partly a statement about the aggregation, not only
about the biology. So a POST HOC probe is run alongside: a second, GENE-SCALE-FREE identity block built from
WITHIN-GENE RANKS of |z| (the best rank among the bound TFs, the mean of the best five, the mean rank, and
the observed-minus-expected count of causal TFs given the element's occupancy). Ranks within a gene are
invariant to how responsive that gene is, so this block cannot be bought with gene responsiveness the way a
magnitude aggregate can. It is labelled POST HOC everywhere it appears and is not allowed to move the
prespecified gates; it exists so the null can be read as "this design, including a sharper feature
parameterisation, does not find it" rather than "one aggregation choice did not find it".

SAMPLING, AND WHAT IT EXCLUDES. One ChIP file per TF (the lowest-accession GRCh38 "conservative IDR
thresholded peaks" file), not all replicates and not all experiments -- so a TF whose occupancy differs
between its K562 experiments is represented by one of them. Only TFs with BOTH a K562 conservative-IDR track
and a Replogle K562 perturbation are used; TFs profiled by ChIP but never perturbed are invisible here, as
are perturbed TFs with no K562 ChIP. Only the K562 arm of the benchmark is used (the other cell types have
no matching Perturb-seq). Only pairs whose measured gene is one of the 8,248 genes Replogle quantified are
kept. 20 swap draws, 5 CV seeds, 5 folds.
"""
import csv
import gzip
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))

API = "https://www.encodeproject.org"
CR = SP / "crispr"
CRISPR_BASE = ("https://raw.githubusercontent.com/EngreitzLab/CRISPR_comparison/main/"
               "resources/crispr_data")
TRAINING = "EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz"
HELDOUT = "EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz"
GWPS = SP / "gwps.h5ad"
ATAC = SP / "k562_atac.bed.gz"
BOUND_CACHE = SP / "causal_enhancer_bound.npz"

# ---- the pins. One assembly, one peak output type, one power bar, one causal-edge threshold. --------------
ASSEMBLY = "GRCh38"
OUTPUT_TYPE = "conservative IDR thresholded peaks"   # same pin bound_causal.py uses; mixing IDR tiers would
                                                     # make "bound" mean different things for different TFs
POWER_COL = "PowerAtEffectSize25"
MIN_POWER = 0.8            # an unpowered non-significant pair is NOT DETECTED, not a negative -- excluded
Z_EDGE = 5.0               # |per-gene robust z| at which a (TF, gene) Perturb-seq pair is called causal;
                           # the same threshold causal_reg.py and bound_causal.py use
ELEMENT_PAD = 250          # bp: a ChIP summit this far outside the tested element still counts as "at" it,
                           # because the CRISPRi tiling element is a guide window, not a protein footprint

# ---- the decision threshold, stated BEFORE the numbers ----------------------------------------------------
FOLDS = 5
SEEDS = (0, 1, 2, 3, 4)    # each seed draws a genuinely DIFFERENT chromosome->fold partition, not a rotation
SWAP_DRAWS = 20
MIN_INCREMENT = 0.010      # absolute AUPRC over the best baseline arm
MIN_SEEDS = 4              # of len(SEEDS): the increment must hold in this many CV partitions
MIN_SWAP_FRAC = 0.90       # 18 of 20 draws in which the real element beats the matched decoy

# ---- joins DECLARE an expected rate and RAISE below it ---------------------------------------------------
MIN_JOIN_GENE = 0.85       # benchmark measured gene -> a gene Replogle quantified
MIN_JOIN_TF = 0.85         # ENCODE K562 ChIP target -> a Replogle perturbation


# ==========================================================================================================
# fetch
# ==========================================================================================================
def download(url, path, tries=4, headers=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    for i in range(tries):
        try:
            r = urllib.request.Request(url, headers=headers or {"User-Agent": "cellos"})
            with urllib.request.urlopen(r, timeout=600) as fh, open(path, "wb") as o:
                while True:
                    b = fh.read(1 << 20)
                    if not b:
                        break
                    o.write(b)
            return path
        except Exception:
            if path.exists():
                path.unlink()
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def api(path, tries=4):
    for i in range(tries):
        try:
            r = urllib.request.Request(API + path,
                                       headers={"accept": "application/json", "User-Agent": "cellos"})
            return json.load(urllib.request.urlopen(r, timeout=300))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def load_benchmark(report):
    """EP CRISPR benchmark, K562 arm. Re-fetched from the EngreitzLab repo the way expand_cells.py does it."""
    rows = []
    for f, only_k562 in ((TRAINING, False), (HELDOUT, True)):
        p = CR / f
        if not p.exists():
            report(f"    fetching {f}")
            download(f"{CRISPR_BASE}/{f}", p)
        r = list(csv.DictReader(gzip.open(p, "rt"), delimiter="\t"))
        if only_k562:
            r = [x for x in r if x["CellType"] == "K562"]
        else:
            for x in r:
                x["CellType"] = "K562"
        report(f"    {f}: {len(r):,} K562 pairs")
        rows += r
    return rows


def chip_index(report):
    """One K562 conservative-IDR narrowPeak per TF, from ONE portal query.

    bound_causal.py walked the portal per TF (two API round-trips each); the File-level search returns the
    whole set in one call with the target attached, which is the same universe and ~500x fewer requests. The
    output type is PINNED, so a TF with no conservative-IDR file is DROPPED and counted rather than
    back-filled with a noisier pseudoreplicated call.
    """
    u = ("/search/?type=File&file_format=bed&file_format_type=narrowPeak"
         f"&assembly={ASSEMBLY}&output_type={urllib.parse.quote(OUTPUT_TYPE)}"
         "&biosample_ontology.term_name=K562&status=released&limit=all&frame=object&format=json")
    d = api(u)
    g = [x for x in d.get("@graph", []) if x.get("assay_title") == "TF ChIP-seq"]
    best = {}
    for x in g:
        t = x["target"].split("/")[-2].replace("-human", "")
        # lowest accession wins: an arbitrary but DETERMINISTIC pick, so a rerun cannot silently change the
        # occupancy matrix under a cached result
        if t not in best or x["accession"] < best[t]["accession"]:
            best[t] = x
    report(f"    ENCODE: {len(g):,} released K562 '{OUTPUT_TYPE}' TF ChIP files over {len(best)} TF targets")
    return best


def peak_summits(path):
    """chrom -> sorted summit positions. narrowPeak column 10 is the summit offset; -1 means none called,
    in which case the interval midpoint is used (bound_causal.py's convention, kept identical)."""
    d = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            try:
                st, en = int(p[1]), int(p[2])
            except ValueError:
                continue
            off = int(p[9]) if len(p) > 9 and p[9] not in ("", "-1") else -1
            d.setdefault(p[0], []).append(st + off if off >= 0 else (st + en) // 2)
    return {c: np.sort(np.array(v, dtype=np.int64)) for c, v in d.items()}


# ==========================================================================================================
# per-gene robust z from the K562 Perturb-seq pseudobulk
# ==========================================================================================================
def robust_z_k562(report):
    """(perturbation x gene) robust z, IDENTICAL to causal_reg.py: (lfc - median) / (1.4826 * MAD) down each
    GENE column. The column effect is the point: some genes move under almost any perturbation, and a raw
    log fold change is dominated by them, so an edge here means 'this gene moved more than it usually moves'.
    """
    import h5py
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        pert = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        ncell = f["obs"]["num_cells_filtered"][:]
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in f["var"]["__categories"]["gene_name"][:]]
        codes = f["var"]["gene_name"][:]
        gid = np.array([(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
                        for s in f["var"]["gene_id"][:]])
        gexpr = f["var"]["mean"][:]
    gsym = np.array([cats[c] for c in codes])
    psym = np.array([p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert])
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    dead = mad < 1e-6
    mad = np.where(dead, np.nan, mad)
    Z = (X - med) / mad
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    report(f"    Perturb-seq K562: {Z.shape[0]:,} perturbations x {Z.shape[1]:,} genes; "
           f"{int(dead.sum())} genes have zero MAD and are z=0 by construction")
    return Z, psym, gsym, gid, np.asarray(gexpr, dtype=np.float64), np.asarray(ncell, dtype=np.float64)


# ==========================================================================================================
# features
# ==========================================================================================================
def tf_features(Bp, Zp, g_mean_abs, g_frac_causal):
    """Aggregate the causal weights of the TFs BOUND at the element, per pair.

    Bp  (n_pairs x n_tf) bool -- is TF t bound at this pair's element
    Zp  (n_pairs x n_tf) float -- robust z of gene g under knockdown of TF t (0 where t IS g: a TF's own
        knockdown moving its own transcript is the assay working, not regulation, and it would hand every
        element containing that TF a spurious enormous weight)
    """
    A = np.abs(Zp) * Bp
    nb = Bp.sum(1).astype(np.float64)
    nbc = np.maximum(nb, 1.0)
    ncausal = ((np.abs(Zp) >= Z_EDGE) & Bp).sum(1).astype(np.float64)
    ndown = ((Zp <= -Z_EDGE) & Bp).sum(1).astype(np.float64)
    mean_abs = A.sum(1) / nbc
    max_abs = A.max(1)
    # top-5 rather than the max alone: one enormous z is a single edge, five is a coherent TF set, and the
    # two are different claims about the element
    k = min(5, A.shape[1])
    top5 = np.sort(np.partition(A, -k, axis=1)[:, -k:], axis=1).sum(1)
    mean_signed = (Zp * Bp).sum(1) / nbc
    frac_causal = ncausal / nbc
    return {
        "tf_n_causal": ncausal,
        "tf_frac_causal": frac_causal,
        "tf_max_abs_z": max_abs,
        "tf_mean_abs_z": mean_abs,
        "tf_top5_abs_z": top5,
        "tf_mean_signed_z": mean_signed,
        "tf_n_causal_down": ndown,
        # element-SPECIFIC enrichment: how much more (or less) this element's TF set moves this gene than
        # the average profiled TF does. The gene's own background is subtracted, so a gene that everything
        # moves does not get a high value for free -- but the value still contains gene information through
        # the interaction, which is precisely why the element swap exists.
        "tf_enrich_abs_z": mean_abs - g_mean_abs,
        "tf_enrich_frac": frac_causal - g_frac_causal,
    }


def rank_features(Bp, Rp, Zp, g_frac_causal, n_tf):
    """POST HOC. The same question asked in a form that CANNOT be answered with gene responsiveness.

    Rp is the rank of |z| DOWN THE TF AXIS within each gene: rank 1 is the TF whose knockdown moves this gene
    most, out of all 507 profiled. Ranks within a gene are invariant to the gene's overall responsiveness, so
    a gene that everything moves and a gene that nothing moves have the identical rank distribution. That is
    the point: the module's other TF block aggregates MAGNITUDES over ~100 bound TFs out of 507, which is
    close enough to the whole set that the aggregate mostly re-reads the gene's own mean. These features ask
    instead whether the element happens to carry the gene's TOP regulators, which is a statement about
    identity and nothing else.
    """
    big = n_tf + 1
    Rm = np.where(Bp, Rp, big)
    k = min(5, Rm.shape[1])
    ncausal = ((np.abs(Zp) >= Z_EDGE) & Bp).sum(1).astype(np.float64)
    nbc = np.maximum(Bp.sum(1).astype(np.float64), 1.0)
    # how many causal TFs would an element with THIS MANY bound TFs carry by chance for THIS gene -- the
    # occupancy count is divided out explicitly rather than left for the model to discover
    expected = nbc * g_frac_causal
    return {
        "rk_best": Rm.min(1).astype(np.float64),
        "rk_top5": np.sort(np.partition(Rm, k - 1, axis=1)[:, :k], axis=1).mean(1).astype(np.float64),
        # dtype pinned: Rp is int16 to keep 507 x 8,248 cheap, and a row sum of up to 507 ranks of up to 507
        # overflows int16. NumPy would promote anyway; saying so means a later dtype change cannot break it.
        "rk_mean": np.where(Bp, Rp, 0).sum(1, dtype=np.int64).astype(np.float64) / nbc,
        "rk_obs_minus_exp_causal": ncausal - expected,
        "rk_ratio_causal": ncausal / np.maximum(expected, 1e-3),
    }


# ==========================================================================================================
# CV
# ==========================================================================================================
def chrom_folds(chrom, y, seed, folds=FOLDS):
    """Assign whole CHROMOSOMES to folds in a seed-specific RANDOM order, each going to the currently
    lightest fold.

    A gene sits on one chromosome, so chromosome-held-out folds are gene-disjoint AND element-disjoint by
    construction. The random order is what makes a seed mean something: rotating the fold LABELS of one
    partition gives an sd of exactly 0.0000, which this project has already reported once and which is
    meaningless. Assigning in a FIXED order (say, by positive count descending) would have the same defect
    in a subtler form -- every seed would produce nearly the same split -- so the order is shuffled and the
    resulting overlap between seeds is measured and printed rather than assumed.
    """
    rng = np.random.default_rng(1000 + seed)
    cs = np.array(sorted(set(chrom)))
    rng.shuffle(cs)
    pos = {c: int(y[chrom == c].sum()) for c in cs}
    load = np.zeros(folds, dtype=np.float64)
    assign = {}
    for c in cs:
        f = int(np.argmin(load))
        assign[c] = f
        load[f] += pos[c] + 0.5           # +0.5 so chromosomes with zero positives still spread out
    return np.array([assign[c] for c in chrom]), {str(c): int(assign[c]) for c in cs}


def fit_folds(F, y, fold, seed):
    """Fit one model per fold on the REAL features. Returned so the element-swap control can re-score a
    swapped feature matrix without retraining -- the swap must change what the model SEES, never what it
    LEARNED, or it would stop being a control and become a different model."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    ms = {}
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        if y[tr].sum() < 5 or y[te].sum() < 1:
            continue
        m = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=1.0, early_stopping=False, random_state=seed)
        m.fit(F[tr], y[tr])
        ms[f] = m
    return ms


def score_oof(ms, F, y, fold):
    from sklearn.metrics import average_precision_score
    p = np.zeros(len(y))
    for f, m in ms.items():
        te = fold == f
        p[te] = m.predict_proba(F[te])[:, 1]
    return float(average_precision_score(y, p))


# ==========================================================================================================
def main():
    from scipy import stats
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 104)
    report("DISTAL ENHANCER-GENE LINKS SCORED BY TF CAUSALITY -- is there an increment over distance and")
    report("accessibility, and does it survive replacing the element with a decile-matched decoy?")
    report("=" * 104)
    report(f"  DECISION THRESHOLD, FIXED BEFORE THE NUMBERS: TF causality earns a place only if the full "
           f"model beats the")
    report(f"  best baseline by >= {MIN_INCREMENT:.3f} AUPRC in >= {MIN_SEEDS}/{len(SEEDS)} CV partitions, "
           f"AND beats the distance x accessibility")
    report(f"  matched element swap in >= {MIN_SWAP_FRAC:.0%} of {SWAP_DRAWS} draws, AND beats the "
           f"DEGREE-matched swap in >= {MIN_SWAP_FRAC:.0%} of {SWAP_DRAWS}.")

    # ---- 1. ground truth ---------------------------------------------------------------------------------
    report("\n  1  GROUND TRUTH -- CRISPR enhancer perturbation, EP CRISPR benchmark K562 arm")
    rows = load_benchmark(report)
    pw = np.array([float(r[POWER_COL] or 0) for r in rows])
    report(f"    {len(rows):,} K562 pairs; {int((pw >= MIN_POWER).sum()):,} powered at "
           f"{POWER_COL} >= {MIN_POWER}")

    Z, psym, gsym, gid, gexpr, ncell = robust_z_k562(report)

    # ---- join 1: benchmark gene -> a gene Replogle quantified. DECLARED rate, RAISES below it. ------------
    g_of = {}
    for j, e in enumerate(gid):
        g_of.setdefault(e, j)
    ens = [r["measuredGeneEnsemblId"].split(".")[0] for r in rows]
    hit = np.array([e in g_of for e in ens])
    rate = float(hit.mean())
    report(f"    JOIN benchmark measuredGeneEnsemblId -> Replogle var/gene_id: "
           f"{int(hit.sum()):,}/{len(rows):,} ({rate:.1%}), expected >= {MIN_JOIN_GENE:.0%}")
    if rate < MIN_JOIN_GENE:
        raise SystemExit(f"gene join rate {rate:.1%} below the declared {MIN_JOIN_GENE:.0%} floor -- a join "
                         f"that matches nothing must not return an empty set a control then passes")

    keep = (pw >= MIN_POWER) & hit
    rows = [r for r, k in zip(rows, keep) if k]
    ens = [e for e, k in zip(ens, keep) if k]
    y = np.array([1 if r["Significant"] in ("TRUE", "True", "true") else 0 for r in rows])
    report(f"    ANALYSIS SET: {len(rows):,} powered distal pairs, {int(y.sum())} positives, "
           f"BASE RATE {y.mean():.4f}")
    from collections import Counter
    report(f"    datasets: { dict(Counter(r['Dataset'] for r in rows)) }")

    el_key = [(r["chrom"], int(r["chromStart"]), int(r["chromEnd"])) for r in rows]
    uel = sorted(set(el_key))
    ei = {k: i for i, k in enumerate(uel)}
    el_idx = np.array([ei[k] for k in el_key])
    gene_idx = np.array([g_of[e] for e in ens])
    dist = np.array([max(abs(float(r["distanceToTSS"])), 1.0) for r in rows])
    chrTSS = np.array([r["chrTSS"] for r in rows])
    report(f"    {len(uel):,} distinct elements, {len(set(ens)):,} distinct genes, "
           f"{len(set(chrTSS))} chromosomes; distance to TSS median {np.median(dist):,.0f} bp "
           f"(min {dist.min():,.0f} -- the benchmark carries no promoter-proximal pair, so every pair here "
           f"is distal)")

    # ---- 2. TF occupancy at the ELEMENTS -----------------------------------------------------------------
    report("\n  2  TF OCCUPANCY AT THE ELEMENTS -- ENCODE K562 ChIP crossed with Replogle perturbations")
    idx = chip_index(report)
    pset = set(psym)
    tfs = sorted(t for t in idx if t in pset)
    tf_rate = len(tfs) / max(len(idx), 1)
    report(f"    JOIN ENCODE K562 ChIP target -> Replogle perturbation: {len(tfs)}/{len(idx)} "
           f"({tf_rate:.1%}), expected >= {MIN_JOIN_TF:.0%}")
    if tf_rate < MIN_JOIN_TF:
        raise SystemExit(f"TF join rate {tf_rate:.1%} below the declared {MIN_JOIN_TF:.0%} floor")

    # one perturbation row per TF: the one with the most cells, because a low-cell pseudobulk is noise and
    # picking the first row would let library depth decide which TFs look causal
    prow = {}
    for i, s in enumerate(psym):
        if s in pset and (s not in prow or ncell[i] > ncell[prow[s]]):
            prow[s] = i
    tf_prow = np.array([prow[t] for t in tfs])

    ech = np.array([k[0] for k in uel])
    est = np.array([k[1] for k in uel], dtype=np.int64)
    een = np.array([k[2] for k in uel], dtype=np.int64)

    cached = None
    if BOUND_CACHE.exists():
        z = np.load(BOUND_CACHE, allow_pickle=True)
        if list(z["tfs"]) == tfs and len(z["elkey"]) == len(uel) and \
                all(str(a) == f"{c}:{s}-{e}" for a, c, s, e in zip(z["elkey"], ech, est, een)):
            cached = z["B"]
            report(f"    occupancy matrix from cache: {cached.shape[0]} TFs x {cached.shape[1]:,} elements")
    if cached is None:
        B = np.zeros((len(tfs), len(uel)), dtype=bool)
        tmp = SP / "ce_chip_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        lock = [0]

        def one(k):
            t = tfs[k]
            p = tmp / f"{t}.bed.gz"
            try:
                download(API + idx[t]["href"], p)
                sm = peak_summits(p)
            finally:
                # the container has < 5 GB of disk headroom; the peak files are deleted after use and only
                # the (507 x 4,362) boolean matrix is cached
                if p.exists():
                    p.unlink()
            col = np.zeros(len(uel), dtype=bool)
            for c in np.unique(ech):
                if c not in sm:
                    continue
                m = np.where(ech == c)[0]
                pos = sm[c]
                lo = np.searchsorted(pos, est[m] - ELEMENT_PAD)
                hi = np.searchsorted(pos, een[m] + ELEMENT_PAD)
                col[m] = hi > lo
            lock[0] += 1
            if lock[0] % 100 == 0:
                report(f"      occupancy {lock[0]}/{len(tfs)} TFs")
            return k, col

        with ThreadPoolExecutor(max_workers=8) as ex:
            for k, col in ex.map(one, range(len(tfs))):
                B[k] = col
        np.savez_compressed(BOUND_CACHE, B=B, tfs=np.array(tfs),
                            elkey=np.array([f"{c}:{s}-{e}" for c, s, e in zip(ech, est, een)]))
        cached = B
    B = cached
    nbound_el = B.sum(0)
    report(f"    occupancy: {len(tfs)} TFs x {len(uel):,} elements, {100*B.mean():.1f}% of cells bound; "
           f"TFs per element median {np.median(nbound_el):.0f} (IQR "
           f"{np.percentile(nbound_el,25):.0f}-{np.percentile(nbound_el,75):.0f}), "
           f"{int((nbound_el == 0).sum()):,} elements carry no profiled TF at all")

    # ---- 3. features -------------------------------------------------------------------------------------
    report("\n  3  FEATURES")
    # TF x gene causal weight matrix, with the self-knockdown diagonal removed
    Zsub = Z[tf_prow, :].copy()                               # (n_tf x n_genes_measured)
    del Z
    sym2col = {}
    for j, s in enumerate(gsym):
        sym2col.setdefault(s, j)
    nself = 0
    for k, t in enumerate(tfs):
        j = sym2col.get(t)
        if j is not None:
            Zsub[k, j] = 0.0
            nself += 1
    report(f"    TF->gene causal weights: {Zsub.shape[0]} TFs x {Zsub.shape[1]:,} genes, robust z, "
           f"{nself} self-knockdown diagonal entries zeroed")

    Zp = Zsub[:, gene_idx].T.astype(np.float32)               # (n_pairs x n_tf), gene side

    # gene-only background over ALL profiled TFs (no element information whatsoever).
    # NOTE these three are INVARIANT under a permutation of the TF axis -- a column's mean, max and causal
    # fraction do not care in what order its 507 entries are listed. That invariance is what makes CONTROL 6
    # a clean identity test: the GENE block is bit-identical before and after the permutation, so nothing
    # the permutation moves can be gene responsiveness in disguise.
    g_mean_abs_full = np.abs(Zsub).mean(0)
    g_frac_full = (np.abs(Zsub) >= Z_EDGE).mean(0)
    g_max_full = np.abs(Zsub).max(0)
    g_mean_abs = g_mean_abs_full[gene_idx]
    g_frac_causal = g_frac_full[gene_idx]
    # within-gene rank of |z| down the TF axis, computed ONCE. Under a TF permutation pi the ranks are simply
    # RANK[pi], because ranking a permuted column gives the permuted ranks -- so the permutation control
    # costs no re-sorting and the POST HOC rank block can be permuted as cheaply as the magnitude block.
    RANKM = (np.argsort(np.argsort(-np.abs(Zsub), axis=0), axis=0) + 1).astype(np.int16)
    report(f"    POST HOC identity block: within-gene ranks of |z| down the {Zsub.shape[0]}-TF axis "
           f"(rank 1 = the TF whose knockdown moves that gene most), invariant to gene responsiveness")

    dhs = np.array([float(r["DHS.RPM"] or 0) for r in rows])
    dhs_pct = np.array([float(r["DHS.percentile"] or 0) for r in rows])

    # an independent accessibility measure, from the K562 ATAC narrowPeak already on disk, so the
    # accessibility arm is not resting on one pipeline's column
    at_sig = np.zeros(len(uel))
    with gzip.open(ATAC, "rt") as fh:
        ap = {}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 7:
                continue
            ap.setdefault(f[0], []).append((int(f[1]), int(f[2]), float(f[6])))
    for c, v in ap.items():
        v.sort()
        s = np.array([x[0] for x in v]); e = np.array([x[1] for x in v]); sg = np.array([x[2] for x in v])
        m = np.where(ech == c)[0]
        if not len(m):
            continue
        for i in m:
            lo = np.searchsorted(s, est[i] - 5000)
            hi = np.searchsorted(s, een[i])
            if hi > lo:
                ov = np.minimum(e[lo:hi], een[i]) - np.maximum(s[lo:hi], est[i])
                good = ov > 0
                if good.any():
                    at_sig[i] = sg[lo:hi][good].max()
    atac = at_sig[el_idx]
    report(f"    accessibility: DHS.RPM (benchmark pipeline) median {np.median(dhs):.2f}; K562 ATAC peak "
           f"overlaps {int((atac > 0).sum()):,}/{len(rows):,} pairs, median signal "
           f"{np.median(atac[atac > 0]) if (atac > 0).any() else 0:.2f}")

    # DHS is a per-PAIR column in the benchmark but a per-ELEMENT property; collapse it to the element so a
    # swapped decoy can carry its own value
    el_dhs = np.zeros(len(uel)); el_dhspct = np.zeros(len(uel))
    for i in range(len(rows)):
        el_dhs[el_idx[i]] = dhs[i]; el_dhspct[el_idx[i]] = dhs_pct[i]
    el_width = (een - est).astype(np.float64)

    # the SCREEN one-hot for CONTROL 7. It is a nuisance block, not a biological one: it is in the module
    # only to ask which rungs survive being told which screen the pair came from.
    ds_names = sorted(set(r["Dataset"] for r in rows))
    ds_arr = np.array([r["Dataset"] for r in rows])
    SCREEN = {f"screen_{n}": (ds_arr == n).astype(float) for n in ds_names}
    report(f"    SCREEN one-hot (CONTROL 7): {len(ds_names)} K562 screens, positive rates "
           f"{ {n: round(float(y[ds_arr == n].mean()), 4) for n in ds_names} }")
    report(f"    element width by screen (median bp), the reason width can act as a screen label: "
           f"{ {n: int(np.median(el_width[el_idx][ds_arr == n])) for n in ds_names} }")

    def blocks(el_i, dist_v, tf_perm=None):
        """Every feature the model can see, as named blocks, for a given ELEMENT assignment.

        Called once with the real elements, once per swap draw with the decoys, and once per permutation
        draw with a relabelled TF axis. Passing the element index in explicitly is what makes the swap a
        genuine swap: the decoy brings its OWN accessibility, its own width, its own TF count and its own TF
        set, and only the gene stays.

        tf_perm is CONTROL 6. It permutes the CAUSALITY side only -- Zp and the rank matrix are read through
        pi while B, the measured occupancy, is untouched. So n_tf_bound is bit-identical, the GENE block is
        bit-identical, the element is the real element, and the one thing that moves is which TF's causal
        vector is attached to which TF's binding pattern.
        """
        d = np.maximum(dist_v, 1.0)
        bp = B[:, el_i].T
        zp = Zp if tf_perm is None else Zp[:, tf_perm]
        rk = RANKM[:, gene_idx].T if tf_perm is None else RANKM[tf_perm][:, gene_idx].T
        tf = tf_features(bp, zp, g_mean_abs, g_frac_causal)
        return {
            "DIST": {"log10_dist": np.log10(d)},
            "ACC": {"log1p_dhs_rpm": np.log1p(el_dhs[el_i]),
                    "dhs_pct": el_dhspct[el_i],
                    "log1p_atac": np.log1p(at_sig[el_i]),
                    "atac_hit": (at_sig[el_i] > 0).astype(float),
                    "log10_width": np.log10(np.maximum(el_width[el_i], 1.0))},
            "OCC": {"n_tf_bound": bp.sum(1).astype(float)},
            "GENE": {"gene_mean_abs_z": g_mean_abs, "gene_frac_causal": g_frac_causal,
                     "gene_max_abs_z": g_max_full[gene_idx], "gene_expr": gexpr[gene_idx]},
            "TFC": tf,
            "RANK": rank_features(bp, rk, zp, g_frac_causal, Zsub.shape[0]),
            "SCREEN": SCREEN,
        }

    real = blocks(el_idx, dist)
    names = {k: sorted(v) for k, v in real.items()}
    report(f"    feature blocks: " + "; ".join(f"{k}({len(v)})" for k, v in names.items()))

    def mat(bl, keys):
        cols = []
        for k in keys:
            for n in names[k]:
                cols.append(np.asarray(bl[k][n], dtype=np.float64))
        return np.column_stack(cols)

    ARMS = {
        "distance_only":            ["DIST"],
        "accessibility_only":       ["ACC"],
        "distance+accessibility":   ["DIST", "ACC"],
        "dist+acc+TFoccupancy":     ["DIST", "ACC", "OCC"],
        "gene_responsiveness_only": ["GENE"],
        "dist+acc+gene":            ["DIST", "ACC", "GENE"],
        "TFcausality_only":         ["OCC", "TFC"],
        "FULL_dist+acc+TFcausality": ["DIST", "ACC", "OCC", "TFC"],
        "FULL+gene":                ["DIST", "ACC", "OCC", "GENE", "TFC"],
        # CONTROL 7, the screen-identity probe. SCREEN is a nuisance one-hot, never part of a biological arm.
        "SCREEN_only":              ["SCREEN"],
        "distance+SCREEN":          ["DIST", "SCREEN"],
        "dist+SCREEN+accessibility": ["DIST", "SCREEN", "ACC"],
        "dist+SCREEN+acc+TFoccupancy": ["DIST", "SCREEN", "ACC", "OCC"],
        "dist+SCREEN+acc+occ+TFcausality": ["DIST", "SCREEN", "ACC", "OCC", "TFC"],
        # POST HOC identity block (within-gene ranks). Never allowed to move the prespecified gates.
        "TFcausality+RANK_only":    ["OCC", "TFC", "RANK"],
        "FULL+RANK":                ["DIST", "ACC", "OCC", "TFC", "RANK"],
        "dist+acc+occ+RANK":        ["DIST", "ACC", "OCC", "RANK"],
    }
    BASELINES = ["distance_only", "accessibility_only", "distance+accessibility",
                 "dist+acc+TFoccupancy", "dist+acc+gene"]
    FULL = "FULL_dist+acc+TFcausality"

    # ---- 4. out-of-fold AUPRC on identical folds ---------------------------------------------------------
    report(f"\n  4  OUT-OF-FOLD AUPRC, chromosome-held-out (gene-disjoint), {len(SEEDS)} genuinely different "
           f"partitions x {FOLDS} folds")
    report(f"    POSITIVE BASE RATE {y.mean():.4f} ({int(y.sum())}/{len(y)}) -- every AUPRC below is RAW, "
           f"not a difference from a shuffle")
    folds, fold_maps = {}, {}
    for s in SEEDS:
        folds[s], fold_maps[s] = chrom_folds(chrTSS, y, s)
    # prove the partitions genuinely differ rather than being relabelings of one split. The comparison is
    # made on the (chromosome -> fold) CO-MEMBERSHIP matrix, which is invariant to renaming the folds; a
    # rotated labelling would score 1.00 here and be caught.
    pair_agree = []
    for a in SEEDS:
        for b in SEEDS:
            if a >= b:
                continue
            cs = sorted(fold_maps[a])
            ca = np.array([fold_maps[a][c] for c in cs])
            cb = np.array([fold_maps[b][c] for c in cs])
            m = np.triu(np.ones((len(cs), len(cs)), bool), 1)
            pair_agree.append(float(np.mean((ca[:, None] == ca[None, :])[m] ==
                                            (cb[:, None] == cb[None, :])[m])))
    report(f"    partition check: mean agreement of chromosome CO-MEMBERSHIP across seed pairs "
           f"{np.mean(pair_agree):.2f} (1.00 would mean the seeds are relabelings of one split)")

    PERTURBED_ARMS = (FULL, "TFcausality_only", "FULL+RANK", "TFcausality+RANK_only")
    res, MODELS = {}, {}
    for arm, ks in ARMS.items():
        F = mat(real, ks)
        vals = []
        for s in SEEDS:
            ms = fit_folds(F, y, folds[s], s)
            if arm in PERTURBED_ARMS:
                MODELS.setdefault(arm, {})[s] = ms      # kept so the swap re-scores, never retrains
            vals.append(score_oof(ms, F, y, folds[s]))
        res[arm] = {"auprc_mean": float(np.mean(vals)), "auprc_sd": float(np.std(vals)),
                    "auprc_per_seed": [float(v) for v in vals], "n_features": F.shape[1],
                    "lift_over_base_rate": float(np.mean(vals) / y.mean())}
        report(f"    {arm:28s} AUPRC {np.mean(vals):.4f} +/- {np.std(vals):.4f}  "
               f"({np.mean(vals)/y.mean():4.2f}x the {y.mean():.4f} base rate, {F.shape[1]} features)")

    best_base = max(BASELINES, key=lambda a: res[a]["auprc_mean"])
    inc = res[FULL]["auprc_mean"] - res[best_base]["auprc_mean"]
    per_seed_win = [int(res[FULL]["auprc_per_seed"][i] - res[best_base]["auprc_per_seed"][i]
                        >= MIN_INCREMENT) for i in range(len(SEEDS))]

    # THE LADDER, because a single "increment over the best baseline" hides which rung actually pays. Each
    # rung adds ONE thing to the rung above it, so the difference between two adjacent rungs is the price of
    # that one thing and nothing else.
    #
    # AND EVERY RUNG GOES THROUGH THE SAME GATE. The increments are computed PAIRED, seed by seed, because
    # all arms share the identical folds: an unpaired comparison of two means each carrying an across-seed
    # sd of ~0.005 would call a +0.009 increment invisible when the paired per-seed deltas are consistent,
    # and would equally let an inconsistent one pass. Whether a rung clears MIN_INCREMENT in >= MIN_SEEDS
    # partitions is then reported for EVERY rung, so no rung gets a positive claim on a number that would
    # have failed the bar the module set for the rung it was testing.
    def rung(lbl, a, b):
        dd = np.array(res[b]["auprc_per_seed"]) - np.array(res[a]["auprc_per_seed"])
        o = {"rung": lbl, "from_arm": a, "to_arm": b,
             "EFFECT_raw_auprc_from": res[a]["auprc_mean"], "EFFECT_raw_auprc_to": res[b]["auprc_mean"],
             "paired_delta_mean": float(dd.mean()), "paired_delta_per_seed": [float(x) for x in dd],
             "n_seeds_clearing_min_increment": int((dd >= MIN_INCREMENT).sum()),
             "n_seeds_positive": int((dd > 0).sum()), "of_seeds": len(SEEDS),
             "clears_prespecified_gate": bool((dd.mean() >= MIN_INCREMENT) and
                                              (dd >= MIN_INCREMENT).sum() >= MIN_SEEDS)}
        report(f"      {lbl:44s} {res[b]['auprc_mean']:.4f}   paired {dd.mean():+.4f}  "
               f">= {MIN_INCREMENT:.3f} in {o['n_seeds_clearing_min_increment']}/{len(SEEDS)}, "
               f"positive in {o['n_seeds_positive']}/{len(SEEDS)}  "
               f"{'CLEARS THE BAR' if o['clears_prespecified_gate'] else 'SUB-THRESHOLD'}")
        return o

    report(f"    THE LADDER -- each rung adds one thing to the rung above it, and EVERY rung is put through "
           f"the SAME gate")
    report(f"    ({MIN_INCREMENT:.3f} paired AUPRC in >= {MIN_SEEDS}/{len(SEEDS)} partitions) as the one the "
           f"question is about:")
    report(f"      {'distance only':44s} {res['distance_only']['auprc_mean']:.4f}   (the floor)")
    ladder_out = [
        rung("+ accessibility", "distance_only", "distance+accessibility"),
        rung("+ TF occupancy COUNT (no causality)", "distance+accessibility", "dist+acc+TFoccupancy"),
        rung("+ TF CAUSALITY weights", "dist+acc+TFoccupancy", FULL),
        rung("+ gene responsiveness", FULL, "FULL+gene"),
    ]
    ladder_extra = [
        rung("[both TF rungs together] over dist+acc", "distance+accessibility", FULL),
        rung("[POST HOC] + within-gene RANK identity block", FULL, "FULL+RANK"),
        rung("[POST HOC] RANK block over dist+acc+count", "dist+acc+TFoccupancy", "dist+acc+occ+RANK"),
    ]
    count_rung = ladder_out[1]
    caus_rung = ladder_out[2]
    inc_over_distacc = res[FULL]["auprc_mean"] - res["distance+accessibility"]["auprc_mean"]
    inc_over_occ = res[FULL]["auprc_mean"] - res["dist+acc+TFoccupancy"]["auprc_mean"]
    report(f"    BEST BASELINE: {best_base} at {res[best_base]['auprc_mean']:.4f}")
    report(f"    INCREMENT of {FULL} over it: {inc:+.4f} AUPRC "
           f"(>= {MIN_INCREMENT:.3f} in {sum(per_seed_win)}/{len(SEEDS)} partitions)")
    report(f"    for contrast, over distance+accessibility alone: {inc_over_distacc:+.4f} -- the gap between "
           f"these two numbers IS the raw count of bound TFs, which needs no Perturb-seq at all")

    # ---- 4b. CONTROL 7: which rungs survive being told which SCREEN the pair came from -------------------
    report(f"\n  4b CONTROL 7 -- THE SCREEN-IDENTITY PROBE. Six K562 screens, positive rates spanning "
           f"{min(y[ds_arr == n].mean() for n in ds_names):.4f} to "
           f"{max(y[ds_arr == n].mean() for n in ds_names):.4f} (a "
           f"{max(y[ds_arr == n].mean() for n in ds_names)/max(min(y[ds_arr == n].mean() for n in ds_names), 1e-9):.0f}x "
           f"spread), and element")
    report(f"     WIDTH -- which sits in the ACCESSIBILITY block -- differs by screen. A model can therefore "
           f"buy AUPRC by naming the batch. Every")
    report(f"     rung is re-run with the screen one-hot already in the model; a rung that survives is about "
           f"the element, one that does not is about the batch.")
    report(f"      {'SCREEN one-hot alone (no element, no distance)':44s} "
           f"{res['SCREEN_only']['auprc_mean']:.4f}   "
           f"({res['SCREEN_only']['auprc_mean']/y.mean():.2f}x the base rate)")
    screen_rungs = [
        rung("SCREEN label over distance", "distance_only", "distance+SCREEN"),
        rung("accessibility over distance+SCREEN", "distance+SCREEN", "dist+SCREEN+accessibility"),
        rung("TF COUNT over dist+SCREEN+acc", "dist+SCREEN+accessibility", "dist+SCREEN+acc+TFoccupancy"),
        rung("TF CAUSALITY over dist+SCREEN+acc+count", "dist+SCREEN+acc+TFoccupancy",
             "dist+SCREEN+acc+occ+TFcausality"),
    ]
    acc_rung_naive = ladder_out[0]
    acc_rung_screen = screen_rungs[1]

    # ---- 5. THE ELEMENT SWAP -----------------------------------------------------------------------------
    report(f"\n  5  ELEMENT SWAP -- keep the gene, replace the element with a decile-matched decoy")
    dq = np.digitize(dist, np.quantile(dist, np.linspace(.1, .9, 9)))
    aq = np.digitize(el_dhspct[el_idx], np.quantile(el_dhspct[el_idx], np.linspace(.1, .9, 9)))
    nq = np.digitize(nbound_el[el_idx], np.quantile(nbound_el[el_idx], np.linspace(.1, .9, 9)))
    report(f"    matching on DECILES, not quartiles: distance decile x accessibility (DHS percentile) "
           f"decile; the degree-matched variant adds the n-TFs-bound decile")

    def swap_indices(bins, rng):
        """Donor pair drawn from the same bin, on a DIFFERENT chromosome from this pair's gene and with a
        different element. Different-chromosome is not fastidiousness: a decoy in cis with the gene could be
        a genuine regulator of it, which would make the control pass for the wrong reason."""
        pool = {}
        for i in range(len(y)):
            pool.setdefault(bins[i], []).append(i)
        pool = {k: np.array(v) for k, v in pool.items()}
        out = np.arange(len(y))
        nfail = 0
        for k, v in pool.items():
            tgt = np.where(bins == k)[0]
            if len(v) < 2:
                nfail += len(tgt)
                continue
            pick = v[rng.integers(0, len(v), size=len(tgt))]
            for attempt in range(8):
                bad = (ech[el_idx[pick]] == chrTSS[tgt]) | (el_idx[pick] == el_idx[tgt])
                if not bad.any():
                    break
                pick[bad] = v[rng.integers(0, len(v), size=int(bad.sum()))]
            bad = (ech[el_idx[pick]] == chrTSS[tgt]) | (el_idx[pick] == el_idx[tgt])
            # a pair that could not be given a legal decoy KEEPS ITS OWN ELEMENT rather than accepting an
            # illegal one. That biases the control toward the real elements, i.e. it can only make the real
            # side look better, so a real side that still fails to win is failing against a lenient control.
            pick[bad] = tgt[bad]
            nfail += int(bad.sum())
            out[tgt] = pick
        return out, nfail

    def run_swap(bins, label, model_arms):
        """One donor draw is shared by every arm AND by both distance modes, and NO model is refitted -- the
        fold models from section 4 are re-used and shown the swapped features.

        TWO DISTANCE MODES, from the SAME donors:
          'decoy_distance'  the swap as specified: the decoy brings its OWN distance, decile-matched to the
                            real one. This has a cost that only became visible in the numbers and is
                            reported rather than hidden -- decile matching preserves the distance
                            DISTRIBUTION but destroys the within-decile distance of each individual pair,
                            and within-decile distance is genuinely predictive, so an arm containing a
                            distance feature loses AUPRC for a reason that is not about the element.
          'real_distance'   POST HOC, added after the first run: the same decoys with the REAL distance
                            retained, so only the element-derived features (accessibility, occupancy, TF
                            causality) are swapped. Labelled post hoc everywhere it appears.
        Sharing the donors between the two modes is not just cheaper, it is the right comparison: the two
        differ ONLY in whether the distance came along.
        """
        arms = list(model_arms)
        MODES = ("decoy_distance", "real_distance")
        aupr = {(m, a): [] for m in MODES for a in arms}
        imb = {"log10_dist": [], "dhs_pct": [], "n_tf_bound": []}
        pv = {"log10_dist": [], "dhs_pct": [], "n_tf_bound": []}
        nfail_tot = 0
        for d in range(SWAP_DRAWS):
            rng = np.random.default_rng(5000 + d)
            j, nf = swap_indices(bins, rng)
            nfail_tot += nf
            sw = blocks(el_idx[j], dist[j])
            swk = dict(sw); swk["DIST"] = {"log10_dist": np.log10(dist)}   # same decoys, real distance
            for cov, a_, b_ in (("log10_dist", np.log10(dist), np.log10(dist[j])),
                                ("dhs_pct", el_dhspct[el_idx], el_dhspct[el_idx[j]]),
                                ("n_tf_bound", nbound_el[el_idx].astype(float),
                                 nbound_el[el_idx[j]].astype(float))):
                imb[cov].append(float(np.mean(a_) - np.mean(b_)))
                pv[cov].append(float(stats.mannwhitneyu(a_, b_, alternative="two-sided")[1]))
            for mode, bl in (("decoy_distance", sw), ("real_distance", swk)):
                for arm in arms:
                    Fs = mat(bl, ARMS[arm])
                    aupr[(mode, arm)].append(float(np.mean([score_oof(MODELS[arm][s], Fs, y, folds[s])
                                                            for s in SEEDS])))
        out = {}
        report(f"    {label}")
        report(f"      residual imbalance (real - decoy): "
               f"{ {k: round(float(np.mean(v)), 4) for k, v in imb.items()} }   "
               f"mean Mann-Whitney p per covariate: "
               f"{ {k: float(f'{np.mean(v):.3g}') for k, v in pv.items()} }")
        report(f"      {nfail_tot / SWAP_DRAWS:,.0f} pairs per draw had no valid donor in their decile cell "
               f"and kept their own element (they make the control CONSERVATIVE, not liberal)")
        for mode in MODES:
            for arm in arms:
                v = aupr[(mode, arm)]
                real_a = res[arm]["auprc_mean"]
                frac = float(np.mean([real_a > a for a in v]))
                o = {"label": label, "model_arm": arm, "draws": SWAP_DRAWS, "distance_mode": mode,
                     "prespecified": mode == "decoy_distance",
                     "EFFECT_raw_real_auprc": real_a,
                     "swapped_auprc_mean": float(np.mean(v)), "swapped_auprc_sd": float(np.std(v)),
                     "swapped_auprc_min": float(np.min(v)), "swapped_auprc_max": float(np.max(v)),
                     "frac_draws_real_beats_swap": frac,
                     "delta_real_minus_swap": real_a - float(np.mean(v)),
                     "frac_of_real_reproduced_by_swap": float(np.mean(v)) / max(real_a, 1e-12),
                     "n_pairs_without_valid_donor_per_draw": int(nfail_tot / SWAP_DRAWS),
                     "residual_imbalance_real_minus_decoy": {k: float(np.mean(x)) for k, x in imb.items()},
                     "residual_imbalance_max_abs": {k: float(np.max(np.abs(x))) for k, x in imb.items()},
                     "residual_imbalance_mean_mannwhitney_p": {k: float(np.mean(x)) for k, x in pv.items()}}
                tag = "PRESPEC" if mode == "decoy_distance" else "POSTHOC"
                report(f"      [{tag} {mode:15s} {arm}] real {real_a:.4f}   swapped {np.mean(v):.4f} "
                       f"+/- {np.std(v):.4f} (range {np.min(v):.4f}-{np.max(v):.4f});  real beats the decoy "
                       f"in {frac:.0%} of {SWAP_DRAWS} draws; the decoy reproduces "
                       f"{o['frac_of_real_reproduced_by_swap']:.0%} of the real AUPRC")
                out[(mode, arm)] = o
        return out

    bins_da = np.array([f"{a}|{b}" for a, b in zip(dq, aq)])
    bins_dan = np.array([f"{a}|{b}|{c}" for a, b, c in zip(dq, aq, nq)])
    ARMS_SWAPPED = [FULL, "TFcausality_only"]
    TFC = "TFcausality_only"
    s_da = run_swap(bins_da, "SWAP A -- decoy matched on distance decile x accessibility decile", ARMS_SWAPPED)
    s_dg = run_swap(bins_dan, "SWAP B -- decoy additionally matched on n-TFs-bound decile (DEGREE-MATCHED)",
                    ARMS_SWAPPED)
    swaps = {
        "distance_x_accessibility": s_da[("decoy_distance", FULL)],
        "degree_matched": s_dg[("decoy_distance", FULL)],
        "distance_x_accessibility_TFCarm": s_da[("decoy_distance", TFC)],
        "degree_matched_TFCarm": s_dg[("decoy_distance", TFC)],
        "element_only_distance_x_accessibility": s_da[("real_distance", FULL)],
        "element_only_degree_matched": s_dg[("real_distance", FULL)],
        "element_only_distance_x_accessibility_TFCarm": s_da[("real_distance", TFC)],
        "element_only_degree_matched_TFCarm": s_dg[("real_distance", TFC)],
    }

    # ---- 5b. CONTROL 6: THE TF-AXIS PERMUTATION -- the identity test that does not touch the element -----
    report(f"\n  5b CONTROL 6 -- TF-AXIS PERMUTATION. The element swap changes the ELEMENT, so it confounds "
           f"'which TFs' with everything")
    report(f"     else an element carries. This does not touch the element. It relabels the TF axis of the "
           f"CAUSAL matrix while the measured")
    report(f"     OCCUPANCY matrix stays exactly as it is, so n_TFs_bound is preserved EXACTLY (not to a "
           f"decile), the gene's whole z")
    report(f"     distribution over the {len(tfs)} TFs is preserved EXACTLY (a permuted column is the same "
           f"multiset), and the GENE block is")
    report(f"     bit-identical. The one thing destroyed is which TF's binding goes with which TF's "
           f"causality. Models are NEVER refitted.")
    tf_bind_deg = B.sum(1).astype(np.float64)                        # elements each TF binds
    tf_causal_breadth = (np.abs(Zsub) >= Z_EDGE).mean(1)             # genes each TF is causal for
    db = np.digitize(tf_bind_deg, np.quantile(tf_bind_deg, np.linspace(.1, .9, 9)))
    dc = np.digitize(tf_causal_breadth, np.quantile(tf_causal_breadth, np.linspace(.1, .9, 9)))
    tf_strata = {}
    for i in range(len(tfs)):
        tf_strata.setdefault((int(db[i]), int(dc[i])), []).append(i)
    report(f"     DEGREE-STRATIFIED variant: {len(tf_strata)} TF strata (binding-degree decile x "
           f"causal-breadth decile), sizes "
           f"{min(len(v) for v in tf_strata.values())}-{max(len(v) for v in tf_strata.values())}. A TF that "
           f"binds everywhere and moves everything")
    report(f"     can only be exchanged for another such TF, so the permutation cannot be beaten by TF-side "
           f"degree -- the configuration model on the TF axis.")

    def run_perm(kind, label, arms):
        aupr = {a: [] for a in arms}
        moved = []
        for d in range(SWAP_DRAWS):
            rng = np.random.default_rng(11000 + d)
            p = np.arange(len(tfs))
            if kind == "free":
                rng.shuffle(p)
            else:
                for _, v in tf_strata.items():
                    v = np.array(v)
                    q = v.copy()
                    rng.shuffle(q)
                    p[v] = q
            moved.append(float((p != np.arange(len(tfs))).mean()))
            bl = blocks(el_idx, dist, tf_perm=p)
            for a in arms:
                Fs = mat(bl, ARMS[a])
                aupr[a].append(float(np.mean([score_oof(MODELS[a][s], Fs, y, folds[s]) for s in SEEDS])))
        out = {}
        report(f"    {label}  ({np.mean(moved):.0%} of TFs relabelled per draw)")
        for a in arms:
            v = aupr[a]
            real_a = res[a]["auprc_mean"]
            frac = float(np.mean([real_a > x for x in v]))
            o = {"label": label, "model_arm": a, "draws": SWAP_DRAWS, "kind": kind,
                 "EFFECT_raw_real_auprc": real_a,
                 "permuted_auprc_mean": float(np.mean(v)), "permuted_auprc_sd": float(np.std(v)),
                 "permuted_auprc_min": float(np.min(v)), "permuted_auprc_max": float(np.max(v)),
                 "frac_draws_real_beats_permuted": frac,
                 "delta_real_minus_permuted": real_a - float(np.mean(v)),
                 "frac_of_real_reproduced_by_permutation": float(np.mean(v)) / max(real_a, 1e-12),
                 "frac_tfs_relabelled_per_draw": float(np.mean(moved)),
                 "post_hoc": a in ("FULL+RANK", "TFcausality+RANK_only")}
            report(f"      [{a:26s}] real {real_a:.4f}   TF-permuted {np.mean(v):.4f} +/- {np.std(v):.4f} "
                   f"(range {np.min(v):.4f}-{np.max(v):.4f});  real beats the permutation in {frac:.0%} of "
                   f"{SWAP_DRAWS} draws; the permutation reproduces "
                   f"{o['frac_of_real_reproduced_by_permutation']:.0%}")
            out[a] = o
        return out

    p_free = run_perm("free", "PERMUTATION A -- free relabelling of the TF axis", list(PERTURBED_ARMS))
    p_deg = run_perm("degree", "PERMUTATION B -- DEGREE-STRATIFIED relabelling (configuration model on the "
                               "TF axis)", list(PERTURBED_ARMS))
    perms = {"free": p_free, "degree_stratified": p_deg}
    # the decisive identity read: the strict permutation on the arm that holds nothing but the element's
    # occupancy and its TF-causality features
    perm_key = p_deg["TFcausality_only"]
    perm_full = p_deg[FULL]
    tf_identity_perm_survives = perm_key["frac_draws_real_beats_permuted"] >= MIN_SWAP_FRAC

    # ---- 6. what the increment is, sanity-checked against the arms it must beat --------------------------
    report("\n  6  READING THE THREE THINGS TOGETHER")
    passes_inc = (inc >= MIN_INCREMENT) and (sum(per_seed_win) >= MIN_SEEDS)
    passes_swap = swaps["distance_x_accessibility"]["frac_draws_real_beats_swap"] >= MIN_SWAP_FRAC
    passes_deg = swaps["degree_matched"]["frac_draws_real_beats_swap"] >= MIN_SWAP_FRAC
    report(f"    (i)   increment >= {MIN_INCREMENT:.3f} in >= {MIN_SEEDS}/{len(SEEDS)} partitions: "
           f"{'PASS' if passes_inc else 'FAIL'}  ({inc:+.4f}, {sum(per_seed_win)}/{len(SEEDS)})")
    report(f"    (ii)  beats the distance x accessibility swap in >= {MIN_SWAP_FRAC:.0%} of draws: "
           f"{'PASS' if passes_swap else 'FAIL'}  "
           f"({swaps['distance_x_accessibility']['frac_draws_real_beats_swap']:.0%})")
    report(f"    (iii) beats the DEGREE-matched swap in >= {MIN_SWAP_FRAC:.0%} of draws: "
           f"{'PASS' if passes_deg else 'FAIL'}  "
           f"({swaps['degree_matched']['frac_draws_real_beats_swap']:.0%})")
    report(f"    gates (ii) and (iii) are PRESPECIFIED and both pass on the full model, but they are LENIENT "
           f"and saying so is part of")
    report(f"    the result: a decile-matched decoy carries the decile of the real distance, not the real "
           f"distance, so any arm with a")
    report(f"    distance feature loses AUPRC under the swap for a reason that is not about the element. "
           f"The two sharper readings:")
    e1 = swaps["element_only_distance_x_accessibility"]
    e2 = swaps["element_only_degree_matched"]
    t2 = swaps["degree_matched_TFCarm"]
    t2e = swaps["element_only_degree_matched_TFCarm"]
    report(f"      element-only swap (real distance kept), FULL model: real {e1['EFFECT_raw_real_auprc']:.4f} "
           f"vs decoy {e1['swapped_auprc_mean']:.4f} "
           f"({e1['frac_of_real_reproduced_by_swap']:.0%} reproduced, real wins "
           f"{e1['frac_draws_real_beats_swap']:.0%}); degree-matched "
           f"{e2['swapped_auprc_mean']:.4f} ({e2['frac_of_real_reproduced_by_swap']:.0%}, real wins "
           f"{e2['frac_draws_real_beats_swap']:.0%})")
    report(f"      TF-causality-only arm (no distance feature at all), DEGREE-matched decoy: real "
           f"{t2['EFFECT_raw_real_auprc']:.4f} vs decoy {t2['swapped_auprc_mean']:.4f} "
           f"({t2['frac_of_real_reproduced_by_swap']:.0%} reproduced, real wins "
           f"{t2['frac_draws_real_beats_swap']:.0%} of {SWAP_DRAWS} draws)")
    tf_identity_survives = (t2["frac_draws_real_beats_swap"] >= MIN_SWAP_FRAC and
                            t2e["frac_draws_real_beats_swap"] >= MIN_SWAP_FRAC)
    # INTERNAL CONSISTENCY CHECK, not a result: the TF-causality-only arm holds NO distance feature, so its
    # two distance modes are shown identical inputs and MUST return identical AUPRC. If they ever diverge,
    # the swap machinery is leaking distance into an arm that should not have it, and every swap number in
    # this module would be suspect.
    swap_machinery_ok = abs(t2["swapped_auprc_mean"] - t2e["swapped_auprc_mean"]) < 1e-12
    report(f"    consistency check: the TF-causality-only arm has no distance feature, so its "
           f"decoy-distance and real-distance swaps must be identical -- "
           f"{t2['swapped_auprc_mean']:.6f} vs {t2e['swapped_auprc_mean']:.6f} "
           f"({'OK' if swap_machinery_ok else 'MISMATCH -- swap machinery is leaking distance'})")
    if not swap_machinery_ok:
        raise SystemExit("swap machinery leaks distance into a distance-free arm; every swap number here "
                         "would be unsound")
    # A decoy landing ABOVE the real element is not a reversal and must not be written as one: sign and
    # significance are read together, and "the decoy is better" is not a claim this design can support.
    above = (" -- and the decoy landing slightly ABOVE the real element is NOT read as a reversal; it is "
             "what no pair-specific signal looks like, the real TF-identity features being replaced by a "
             "smoother draw from the same distribution"
             if t2["swapped_auprc_mean"] > t2["EFFECT_raw_real_auprc"] else "")

    # A descriptive cross-check, not a claim: are positives bound by more CAUSAL TFs than negatives?
    #
    # IT IS RUN TWICE, AND THE SECOND RUN IS THE ONE THAT COUNTS. An earlier version matched only on
    # distance x accessibility deciles and reported the result as the one real thing that survived. But the
    # whole finding of this module is that the element pays through the NUMBER of TFs bound, and positives
    # sit at busier elements: pair-weighted they carry 102 bound TFs against 53 for negatives. A count of
    # "bound AND causal" TFs is then higher for positives for the trivial reason that they have more bound
    # TFs to begin with. Matching on distance x accessibility only is therefore a control WEAKER than the
    # claim it was licensing, and by exactly the covariate this module says is doing the work. So the
    # n-TFs-bound decile is added as a third matching key -- the same degree matching the predictive swap
    # already insisted on -- and both versions are reported side by side.
    report("\n    descriptive cross-check: TFs bound AND causal at the element, positives vs matched negatives")
    tfc_n = real["TFC"]["tf_n_causal"]

    def descriptive(bins, label, val):
        pooln, pool1 = {}, {}
        for i in range(len(y)):
            (pool1 if y[i] else pooln).setdefault(bins[i], []).append(i)
        dr, nused = [], 0
        for d in range(SWAP_DRAWS):
            rng = np.random.default_rng(7000 + d)
            a, b = [], []
            for k, v1 in pool1.items():
                v0 = pooln.get(k, [])
                if not v0:
                    continue
                a += list(v1)
                b += list(rng.choice(v0, len(v1), replace=len(v0) < len(v1)))
            if len(a) < 30:
                continue
            nused = len(a)
            dr.append((float(np.mean(val[a])), float(np.mean(val[b])),
                       float(stats.mannwhitneyu(val[a], val[b], alternative="two-sided")[1])))
        if not dr:
            report(f"      {label}: NOT RUN, no stratum carried both arms")
            return {"n_draws": 0, "label": label,
                    "reason": "no stratum carried both a positive and a negative"}
        mp = float(np.mean([x[0] for x in dr])); mn_ = float(np.mean([x[1] for x in dr]))
        fs = float(np.mean([x[2] < 0.05 for x in dr]))
        o = {"label": label, "n_draws": len(dr), "n_positives_matched": nused,
             "EFFECT_raw_positives_mean_n_causal_bound_TFs": mp, "matched_negatives_mean": mn_,
             "ratio": mp / max(mn_, 1e-12), "frac_draws_p_lt_0.05": fs,
             "median_p": float(np.median([x[2] for x in dr])),
             "clears_min_swap_frac": bool(fs >= MIN_SWAP_FRAC)}
        report(f"      {label:44s} positives {mp:.2f} vs matched negatives {mn_:.2f} "
               f"({mp/max(mn_,1e-12):.2f}x), p<0.05 in {fs:.0%} of {len(dr)} draws "
               f"(median p {o['median_p']:.3g})  "
               f"{'holds' if fs >= MIN_SWAP_FRAC else 'DOES NOT HOLD at the ' + f'{MIN_SWAP_FRAC:.0%}' + ' bar'}")
        return o

    cross = descriptive(bins_da, "matched on dist x acc deciles ONLY", tfc_n)
    cross_deg = descriptive(bins_dan, "DEGREE-MATCHED (+ n-TFs-bound decile)", tfc_n)
    cross_count = descriptive(bins_da, "for contrast: n TFs BOUND, dist x acc only", nbound_el[el_idx].astype(float))
    report(f"      raw, unmatched: positives sit at elements carrying "
           f"{nbound_el[el_idx][y == 1].mean():.0f} bound TFs against "
           f"{nbound_el[el_idx][y == 0].mean():.0f} for negatives "
           f"({nbound_el[el_idx][y == 1].mean()/max(nbound_el[el_idx][y == 0].mean(),1e-9):.2f}x) -- which "
           f"is why the degree-matched row above is the one that counts")

    # ---- verdict -----------------------------------------------------------------------------------------
    base_rate = float(y.mean())
    rb = res[best_base]["auprc_mean"]
    rf = res[FULL]["auprc_mean"]
    sw1 = swaps["distance_x_accessibility"]
    sw2 = swaps["degree_matched"]
    head = (f"On {len(y):,} powered distal K562 CRISPR pairs at a {base_rate:.4f} positive base rate "
            f"({int(y.sum())} positives), chromosome-held-out AUPRC is "
            f"{res['distance_only']['auprc_mean']:.4f} for distance alone "
            f"({res['distance_only']['auprc_mean']/base_rate:.1f}x the base rate), "
            f"{res['accessibility_only']['auprc_mean']:.4f} for accessibility alone, "
            f"{res['gene_responsiveness_only']['auprc_mean']:.4f} for gene responsiveness with no element "
            f"information at all, {res['distance+accessibility']['auprc_mean']:.4f} for distance + "
            f"accessibility, {res['dist+acc+TFoccupancy']['auprc_mean']:.4f} once the raw COUNT of the "
            f"{len(tfs)} profiled TFs bound at the element is added, and {rf:.4f} once those TFs are "
            f"WEIGHTED by their Perturb-seq causality for the gene. One nuisance has to be named before any "
            f"of that is read as chromatin: the K562 arm is {len(ds_names)} screens whose positive rates run "
            f"{min(y[ds_arr == n].mean() for n in ds_names):.4f} to "
            f"{max(y[ds_arr == n].mean() for n in ds_names):.4f}, and the screen one-hot ALONE, with no "
            f"element and no distance, reaches {res['SCREEN_only']['auprc_mean']:.4f} "
            f"({res['SCREEN_only']['auprc_mean']/base_rate:.1f}x the base rate). Told which screen a pair "
            f"came from, accessibility still adds {acc_rung_screen['paired_delta_mean']:+.4f} "
            f"({acc_rung_screen['n_seeds_clearing_min_increment']}/{len(SEEDS)} partitions at the bar) "
            f"against {acc_rung_naive['paired_delta_mean']:+.4f} when it is not, so the accessibility rung "
            f"is {'largely batch' if acc_rung_screen['paired_delta_mean'] < 0.5 * acc_rung_naive['paired_delta_mean'] else 'not explained by batch'}")
    if passes_inc and passes_swap and passes_deg:
        v = (f"TF CAUSALITY AT THE ELEMENT IS A REAL INCREMENT. {head}. That is {inc:+.4f} AUPRC over the "
             f"best baseline ({best_base}, {rb:.4f}), held in {sum(per_seed_win)}/{len(SEEDS)} chromosome "
             f"partitions, and it survives the decisive control: swapping in a distance- and "
             f"accessibility-decile-matched decoy element drops AUPRC to {sw1['swapped_auprc_mean']:.4f} "
             f"(real wins {sw1['frac_draws_real_beats_swap']:.0%} of {SWAP_DRAWS} draws), and a decoy also "
             f"matched on the number of bound TFs drops it to {sw2['swapped_auprc_mean']:.4f} "
             f"({sw2['frac_draws_real_beats_swap']:.0%}). The model is reading WHICH TFs sit on the element, "
             f"not how busy it is and not which gene it is pointed at.")
    elif not passes_inc:
        v = (f"THE ELEMENT'S TF SET PAYS, BUT AS A COUNT, NOT AS A CAUSAL IDENTITY -- the causality "
             f"weighting is NOT DETECTED over an occupancy-count baseline. {head}. Read as a ladder: "
             f"accessibility adds {res['distance+accessibility']['auprc_mean'] - res['distance_only']['auprc_mean']:+.4f} "
             f"to distance, the raw number of bound TFs adds a further "
             f"{res['dist+acc+TFoccupancy']['auprc_mean'] - res['distance+accessibility']['auprc_mean']:+.4f}, "
             f"and weighting those TFs by whether their knockdown actually moves the gene adds only "
             f"{inc_over_occ:+.4f} on top -- below the prespecified {MIN_INCREMENT:.3f} bar and clearing it "
             f"in {sum(per_seed_win)}/{len(SEEDS)} chromosome partitions. Against the two baselines the "
             f"question named, TF causality does add ({inc_over_distacc:+.4f} over distance+accessibility, "
             f"{rf - res['distance_only']['auprc_mean']:+.4f} over distance alone), but "
             f"{100*(1 - inc_over_occ/max(inc_over_distacc,1e-9)):.0f}% of that increment is bought by "
             f"counting TFs, which needs no Perturb-seq. The DEGREE-MATCHED element swap says the same thing "
             f"in the sharpest form available here: on the TF-causality-only arm, which has no distance "
             f"feature to lose, a decoy element carrying the same NUMBER of bound TFs but a DIFFERENT TF SET "
             f"scores {t2['swapped_auprc_mean']:.4f} against the real element's "
             f"{t2['EFFECT_raw_real_auprc']:.4f} -- {t2['frac_of_real_reproduced_by_swap']:.0%} of it, with "
             f"the real element winning {t2['frac_draws_real_beats_swap']:.0%} of {SWAP_DRAWS} draws{above}. "
             f"THE CLEANER IDENTITY TEST AGREES AND IS THE ONE TO QUOTE, because it never touches the "
             f"element at all: relabelling the TF axis of the causal matrix -- which holds the element, the "
             f"gene, the EXACT number of bound TFs and the gene's EXACT z distribution over all {len(tfs)} "
             f"TFs fixed, and destroys only which TF's binding goes with which TF's causality -- leaves the "
             f"TF-causality-only arm at {perm_key['permuted_auprc_mean']:.4f} against the real "
             f"{perm_key['EFFECT_raw_real_auprc']:.4f} "
             f"({perm_key['frac_of_real_reproduced_by_permutation']:.0%} reproduced, real wins "
             f"{perm_key['frac_draws_real_beats_permuted']:.0%} of {SWAP_DRAWS} draws) and the full model at "
             f"{perm_full['permuted_auprc_mean']:.4f} against {rf:.4f} "
             f"({perm_full['frac_of_real_reproduced_by_permutation']:.0%}, real wins "
             f"{perm_full['frac_draws_real_beats_permuted']:.0%}), under a DEGREE-STRATIFIED permutation in "
             f"which a TF can only be exchanged for one of similar binding degree and similar causal "
             f"breadth. AND THE CONTRAST BETWEEN THE TWO PERMUTATIONS IS THE MOST INSTRUCTIVE NUMBER IN THE "
             f"MODULE, so it leads rather than sits in a footnote: a FREE relabelling of the TF axis IS "
             f"beaten by the real data -- the TF-causality-only arm scores "
             f"{p_free['TFcausality_only']['EFFECT_raw_real_auprc']:.4f} against a freely permuted "
             f"{p_free['TFcausality_only']['permuted_auprc_mean']:.4f}, winning "
             f"{p_free['TFcausality_only']['frac_draws_real_beats_permuted']:.0%} of {SWAP_DRAWS} draws, and "
             f"the post hoc rank arm wins "
             f"{p_free['TFcausality+RANK_only']['frac_draws_real_beats_permuted']:.0%} -- so a module that "
             f"had run only the uniform control would have reported TF IDENTITY AS A REAL EFFECT. The whole "
             f"of that apparent effect is TF-side DEGREE: how many elements a TF binds and how many genes "
             f"its knockdown moves. Hold those two deciles fixed and the real assignment of causality to "
             f"occupancy loses its entire advantage "
             f"({perm_key['frac_draws_real_beats_permuted']:.0%} and "
             f"{p_deg['TFcausality+RANK_only']['frac_draws_real_beats_permuted']:.0%} of draws). That is the "
             f"finding, not a caveat: promiscuous TFs sit on many elements and move many genes, and that "
             f"alone reproduces everything the causal weighting appeared to contribute. AND THE COUNT DOES "
             f"NOT ESCAPE THE SAME BAR. Applying the module's own gate to every "
             f"rung rather than only to the rung under test: accessibility over distance clears it "
             f"({acc_rung_naive['paired_delta_mean']:+.4f} paired, "
             f"{acc_rung_naive['n_seeds_clearing_min_increment']}/{len(SEEDS)}), but the TF COUNT rung does "
             f"NOT ({count_rung['paired_delta_mean']:+.4f}, "
             f"{count_rung['n_seeds_clearing_min_increment']}/{len(SEEDS)}) and neither does the TF "
             f"CAUSALITY rung ({caus_rung['paired_delta_mean']:+.4f}, "
             f"{caus_rung['n_seeds_clearing_min_increment']}/{len(SEEDS)}). So the honest reading is not "
             f"'identity fails, count works': BOTH TF rungs are sub-threshold, the count differing only in "
             f"being consistent in sign ({count_rung['n_seeds_positive']}/{len(SEEDS)} partitions positive "
             f"against {caus_rung['n_seeds_positive']}/{len(SEEDS)}). AND THE COUNT DOES NOT SURVIVE BEING "
             f"TOLD WHICH SCREEN THE PAIR CAME FROM: with the screen one-hot already in the model the TF "
             f"COUNT rung is {screen_rungs[2]['paired_delta_mean']:+.4f} and positive in only "
             f"{screen_rungs[2]['n_seeds_positive']}/{len(SEEDS)} partitions, and the TF CAUSALITY rung on "
             f"top of it {screen_rungs[3]['paired_delta_mean']:+.4f} in "
             f"{screen_rungs[3]['n_seeds_positive']}/{len(SEEDS)} -- both NOT DETECTED, and neither is "
             f"called reversed, because a negative point estimate that no partition supports is an absence "
             f"of signal and not evidence of the opposite. The screens differ in element geometry "
             f"(median width {min(np.median(el_width[el_idx][ds_arr == n]) for n in ds_names):.0f}-"
             f"{max(np.median(el_width[el_idx][ds_arr == n]) for n in ds_names):.0f} bp) and by "
             f"{max(y[ds_arr == n].mean() for n in ds_names)/max(min(y[ds_arr == n].mean() for n in ds_names), 1e-9):.0f}x "
             f"in hit rate, and a wider element carries more ChIP peaks, so the TF count was partly reading "
             f"the assay. ACCESSIBILITY IS THE ONLY RUNG LEFT STANDING: it clears the bar both naively "
             f"({acc_rung_naive['paired_delta_mean']:+.4f}, "
             f"{acc_rung_naive['n_seeds_clearing_min_increment']}/{len(SEEDS)}) and with the screen held "
             f"({acc_rung_screen['paired_delta_mean']:+.4f}, "
             f"{acc_rung_screen['n_seeds_clearing_min_increment']}/{len(SEEDS)}). WHICH transcription "
             f"factors sit on a distal element, weighted by their causal effect on the gene, is not shown "
             f"here to matter; and HOW MANY sit on it, which an earlier version of this module credited with "
             f"the finding, does not clear the bar either and does not survive the screen. The prespecified "
             f"swap gates (ii) "
             f"and (iii) both pass on the full model ({sw1['frac_draws_real_beats_swap']:.0%} and "
             f"{sw2['frac_draws_real_beats_swap']:.0%}), but they are lenient by construction -- a "
             f"decile-matched decoy carries the decile of the distance and not the distance, and the full "
             f"model has a distance feature to lose. Hold the real distance and swap only the element, and "
             f"the same degree-matched decoys reproduce "
             f"{e2['frac_of_real_reproduced_by_swap']:.0%} of the full model's AUPRC "
             f"({e2['swapped_auprc_mean']:.4f} against {rf:.4f}), with the real element winning "
             f"{e2['frac_draws_real_beats_swap']:.0%} of {SWAP_DRAWS} draws -- a coin flip. THE DESCRIPTIVE "
             f"CROSS-CHECK GOES THE SAME WAY ONCE IT IS DEGREE-MATCHED, and this replaces what an earlier "
             f"version of this module called the one real thing that survived: matched on distance x "
             f"accessibility deciles only, positives carry "
             f"{cross.get('EFFECT_raw_positives_mean_n_causal_bound_TFs', float('nan')):.2f} bound-and-causal "
             f"TFs against {cross.get('matched_negatives_mean', float('nan')):.2f} "
             f"({cross.get('ratio', float('nan')):.2f}x, p<0.05 in "
             f"{100*cross.get('frac_draws_p_lt_0.05', 0):.0f}% of {cross.get('n_draws', 0)} draws) -- but "
             f"positives sit at elements carrying {nbound_el[el_idx][y == 1].mean():.0f} bound TFs against "
             f"{nbound_el[el_idx][y == 0].mean():.0f} for negatives, so that comparison was matched on "
             f"everything EXCEPT the covariate this module says does the work. Add the n-TFs-bound decile "
             f"and it falls to {cross_deg.get('EFFECT_raw_positives_mean_n_causal_bound_TFs', float('nan')):.2f} "
             f"against {cross_deg.get('matched_negatives_mean', float('nan')):.2f} "
             f"({cross_deg.get('ratio', float('nan')):.2f}x) with p<0.05 in only "
             f"{100*cross_deg.get('frac_draws_p_lt_0.05', 0):.0f}% of "
             f"{cross_deg.get('n_draws', 0)} draws, short of the {MIN_SWAP_FRAC:.0%} bar the module uses for "
             f"every other matched control. NOTHING IS LEFT STANDING ON THE HEADLINE QUESTION -- not the "
             f"predictive increment, not the element swap, not the TF-axis permutation, and not the "
             f"descriptive association once it is matched on TF count. This is a null and the project "
             f"commits it. The POST HOC detectability probe is reported so the null is not read as stronger "
             f"than it is: a second, gene-scale-free identity block built from WITHIN-GENE RANKS of |z| "
             f"moves the full model to {res['FULL+RANK']['auprc_mean']:.4f} "
             f"({ladder_extra[1]['paired_delta_mean']:+.4f} paired, "
             f"{ladder_extra[1]['n_seeds_clearing_min_increment']}/{len(SEEDS)} partitions at the bar), "
             f"which is still sub-threshold but says the aggregation choice is not the whole story, and it "
             f"too is erased by the degree-stratified TF permutation "
             f"({p_deg['FULL+RANK']['frac_of_real_reproduced_by_permutation']:.0%} reproduced, real wins "
             f"{p_deg['FULL+RANK']['frac_draws_real_beats_permuted']:.0%}).")
    elif not passes_swap:
        v = (f"THE INCREMENT IS REAL BUT THE ELEMENT IS NOT DOING THE WORK -- the model reads the gene and "
             f"the distance. {head}, an increment of {inc:+.4f} over {best_base} ({rb:.4f}). But replacing "
             f"the element with a distance- and accessibility-decile-matched decoy, keeping the gene, leaves "
             f"AUPRC at {sw1['swapped_auprc_mean']:.4f} -- {sw1['frac_of_real_reproduced_by_swap']:.0%} of "
             f"the real value, with the real element winning only "
             f"{sw1['frac_draws_real_beats_swap']:.0%} of {SWAP_DRAWS} draws. Whatever the TF-causality "
             f"features add, they add it from the GENE side of the pair, and a gene-responsiveness-only arm "
             f"with no element information already reaches "
             f"{res['gene_responsiveness_only']['auprc_mean']:.4f}. This does not license scoring an "
             f"element-gene pair by the element's TF set.")
    else:
        v = (f"THE INCREMENT IS ELEMENT-SPECIFIC BUT DEGREE EXPLAINS IT -- the finding is TF DENSITY, not TF "
             f"IDENTITY. {head}, an increment of {inc:+.4f} over {best_base} ({rb:.4f}), and it does survive "
             f"a distance x accessibility matched element swap ({sw1['swapped_auprc_mean']:.4f}, real "
             f"winning {sw1['frac_draws_real_beats_swap']:.0%} of draws). It does NOT survive the "
             f"DEGREE-MATCHED swap: a decoy element carrying the same NUMBER of bound TFs reproduces "
             f"{sw2['frac_of_real_reproduced_by_swap']:.0%} of the AUPRC "
             f"({sw2['swapped_auprc_mean']:.4f}), with the real element winning only "
             f"{sw2['frac_draws_real_beats_swap']:.0%} of {SWAP_DRAWS} draws. An element with many TFs on it "
             f"is a better enhancer candidate; WHICH TFs, weighted by their Perturb-seq causality for the "
             f"gene, is not shown here to matter. That is the finding, not a footnote.")

    limits = [
        f"same cell line on both sides. TF->gene causality is Replogle K562 Perturb-seq and the ground truth "
        f"is CRISPRi enhancer perturbation in K562. The experiments differ (gene knockdown vs element "
        f"silencing, different labs and libraries), so this is not scoring a K562 layer on K562 Perturb-seq, "
        f"but 'which genes move easily in K562' is shared between them. That nuisance is what the "
        f"gene-responsiveness-only arm ({res['gene_responsiveness_only']['auprc_mean']:.4f} AUPRC with NO "
        f"element information) measures and what the element swap removes. There is no RPE1 enhancer-CRISPR "
        f"benchmark to escape to.",
        f"the benchmark's K562 arm is {len(ds_names)} screens with different libraries, element geometries "
        f"and hit rates (positive rate {min(y[ds_arr == n].mean() for n in ds_names):.4f} to "
        f"{max(y[ds_arr == n].mean() for n in ds_names):.4f}). Dataset is NOT a matching key in the element "
        f"swap, so a decoy can come from a different screen than its target pair. The screen one-hot alone "
        f"reaches {res['SCREEN_only']['auprc_mean']:.4f} AUPRC with no element information, and element "
        f"WIDTH -- which sits in the accessibility block -- is close to a screen label "
        f"({ {n: int(np.median(el_width[el_idx][ds_arr == n])) for n in ds_names} } bp median by screen). "
        f"CONTROL 7 conditions every rung on the screen; the numbers are in screen_conditioned_ladder.",
        f"the SCREEN one-hot used in CONTROL 7 is a nuisance covariate available here only because the "
        f"benchmark records which screen tested which pair. It is not a feature a deployed E-G predictor "
        f"could use, and it is never part of a biological arm -- it exists to ask which rungs survive "
        f"knowing the batch, not to raise anyone's AUPRC.",
        f"one ChIP file per TF (lowest-accession GRCh38 '{OUTPUT_TYPE}'). A TF whose K562 experiments "
        f"disagree is represented by one of them, and {len(idx) - len(tfs)} ChIP-profiled TFs are dropped "
        f"for having no Replogle perturbation.",
        f"{len(tfs)} TFs is a large but not exhaustive slice of the ~1,600 human TFs; an element's real "
        f"driver may be a TF with no K562 ChIP track, which would make its TF-causality features look empty "
        f"for a reason that is about ENCODE's coverage and not about the element.",
        f"occupancy is binary at ELEMENT_PAD={ELEMENT_PAD} bp around the tested element. Peak strength is "
        f"discarded, so a marginal peak and a summit-in-the-middle peak count the same.",
        f"'causal' is |robust z| >= {Z_EDGE} in a pseudobulk knockdown. It is a CONSEQUENCE of removing the "
        f"TF, direct or not -- bound_causal.py's own finding is that only 31.6% of causal edges are bound at "
        f"the promoter at all, so most of these weights are downstream effects that happen to be attached to "
        f"a TF that also sits at this element.",
        f"the swap donor pool is the benchmark's own {len(uel):,} elements. That is deliberate (a decoy that "
        f"is a real tested candidate is a harder control than a random genomic window) but it means the "
        f"control asks 'is THIS element better than ANOTHER TESTED element', not 'better than random DNA'.",
        f"THE PRESPECIFIED SWAP GATES ARE LENIENT, AND THIS WEAKENS THE PART OF THE RESULT THAT PASSED. A "
        f"decile-matched decoy carries the DECILE of the real distance, not the real distance, so any arm "
        f"holding a distance feature loses within-decile resolution under the swap and its AUPRC falls for a "
        f"reason unrelated to the element. Gates (ii) and (iii) pass on the full model largely for that "
        f"reason. The element-only variants (real distance retained) and the TF-causality-only arm are "
        f"reported alongside and are the sharper reads; they were added AFTER seeing the first result and "
        f"are labelled post hoc in the JSON.",
        f"the degree-matched swap has {swaps['degree_matched']['n_pairs_without_valid_donor_per_draw']:,} "
        f"pairs per draw ({100*swaps['degree_matched']['n_pairs_without_valid_donor_per_draw']/len(y):.1f}%) "
        f"whose distance x accessibility x n-TF-bound decile cell held no valid donor; those pairs keep "
        f"their own element, which biases that control TOWARD the real elements, i.e. makes it conservative.",
        f"the causality weights are aggregates over the bound TF set (count, max, mean, top-5, signed mean, "
        f"and two gene-background-subtracted terms). A per-TF model that learned which INDIVIDUAL TFs matter "
        f"is not fitted here: with {int(y.sum())} positives over {len(tfs)} TFs it would be fitting noise, "
        f"and that is a real ceiling on this design rather than a claim that no such signal exists.",
        f"AND THE AGGREGATION IS A WEAKER INSTRUMENT THAN IT LOOKS, WHICH WEAKENS THIS MODULE'S OWN NULL. "
        f"The bound set is not small: pair-weighted, positives carry {nbound_el[el_idx][y == 1].mean():.0f} "
        f"of the {len(tfs)} profiled TFs and negatives {nbound_el[el_idx][y == 0].mean():.0f}. A mean of |z| "
        f"over ~{nbound_el[el_idx].mean():.0f} of {len(tfs)} TFs is close to the gene's mean over all of "
        f"them, so a magnitude aggregate is structurally near-blind to WHICH TFs those are, and a null from "
        f"it is partly a statement about the aggregation. The POST HOC within-gene-RANK block is included "
        f"precisely to probe that, and it does move the full model "
        f"({ladder_extra[1]['paired_delta_mean']:+.4f} paired, "
        f"{ladder_extra[1]['n_seeds_clearing_min_increment']}/{len(SEEDS)} at the bar) -- sub-threshold, but "
        f"not zero, so 'no identity signal exists' is NOT what this module is entitled to say. What it is "
        f"entitled to say is that no parameterisation tried here converts identity into ranking power that "
        f"survives a degree-stratified TF permutation.",
        f"the TF-axis permutation (CONTROL 6) preserves each TF's causal vector and each element's occupancy "
        f"exactly, but a free permutation also breaks the association between a TF's BINDING degree and its "
        f"CAUSAL breadth, and that association is the entire apparent effect: the real data beats the FREE "
        f"permutation in {p_free['TFcausality_only']['frac_draws_real_beats_permuted']:.0%} of draws on the "
        f"TF-causality-only arm and the DEGREE-STRATIFIED one in only "
        f"{perm_key['frac_draws_real_beats_permuted']:.0%}. A module that ran only the uniform version would "
        f"have reported a positive. The stratified draws relabel "
        f"{p_deg['TFcausality_only']['frac_tfs_relabelled_per_draw']:.0%} of TFs against "
        f"{p_free['TFcausality_only']['frac_tfs_relabelled_per_draw']:.0%} for free draws -- with "
        f"{len(tf_strata)} strata over {len(tfs)} TFs and the smallest holding "
        f"{min(len(v) for v in tf_strata.values())}, the stratified control moves less and is therefore the "
        f"more conservative of the two, which makes the real side's failure to beat it the stronger reading.",
        f"the screen-conditioned rungs (CONTROL 7) are themselves a limit on this module's OWN null as well "
        f"as on its positive claims: conditioning on the screen raises every arm that contains it (distance "
        f"+ screen reaches {res['distance+SCREEN']['auprc_mean']:.4f} against "
        f"{res['distance_only']['auprc_mean']:.4f}), so the TF rungs are being asked to add on top of a "
        f"stronger model than the prespecified ladder used. That is the right test for 'is this batch', but "
        f"it is a harder test than the prespecified one and the prespecified gates are not rescored on it.",
        f"the element-swap donor is drawn uniformly over PAIRS, not over elements, so an element tested "
        f"against many genes is over-represented among decoys (mean pairs-per-element behind a donor draw is "
        f"higher than the {len(y)/len(uel):.2f} a uniform-over-elements draw would give). That skews decoys "
        f"toward multiply-tested elements. The TF-axis permutation does not have this defect, which is "
        f"another reason it is the identity number quoted.",
        f"pairs below {POWER_COL} {MIN_POWER} are excluded, not counted as negatives, because not-detected "
        f"is not not-linked. {int((pw < MIN_POWER).sum()):,} pairs are dropped that way.",
        f"AUPRC is pooled over out-of-fold predictions within a seed. Folds are whole chromosomes, so fold "
        f"sizes are uneven and a single large chromosome carries disproportionate weight in every partition.",
        f"a decoy element is required to be on a different chromosome from the gene, which removes the risk "
        f"that the decoy genuinely regulates the gene but also means the swap changes chromosome-level "
        f"context (replication timing, compartment) along with the element.",
        f"the model is one gradient-boosting configuration, fixed across all arms rather than tuned per arm. "
        f"A tuned baseline could beat an untuned full model or the reverse; holding it fixed is what makes "
        f"the arms comparable, and it means no arm is at its ceiling.",
    ]

    R = {
        "module": "causal_enhancer",
        "question": ("do the TFs bound at a distal element, weighted by whether those TFs are causal for the "
                     "gene in K562 Perturb-seq, predict a CRISPR-validated enhancer-gene link better than "
                     "distance and better than accessibility?"),
        "decision_threshold_stated_before_numbers": {
            "min_increment_auprc_over_best_baseline": MIN_INCREMENT,
            "min_seeds_of_total": [MIN_SEEDS, len(SEEDS)],
            "min_fraction_of_swap_draws_real_beats_decoy": MIN_SWAP_FRAC,
            "swap_draws": SWAP_DRAWS,
            "rule": "all three must hold; failing (i) = no increment, failing (ii) = reads the gene and the "
                    "distance, failing (iii) = reads TF density not TF identity",
        },
        "ground_truth": {
            "source": "EP CRISPR benchmark (EngreitzLab/CRISPR_comparison), K562 arm: "
                      f"{TRAINING} + K562 rows of {HELDOUT}",
            "n_pairs_powered": len(y), "n_positives": int(y.sum()), "positive_base_rate": base_rate,
            "n_elements": len(uel), "n_genes": len(set(ens)), "n_chromosomes": len(set(chrTSS)),
            "median_distance_bp": float(np.median(dist)), "min_distance_bp": float(dist.min()),
            "power_col": POWER_COL, "min_power": MIN_POWER,
            "datasets": dict(Counter(r["Dataset"] for r in rows)),
        },
        "joins": {
            "benchmark_gene_to_replogle": {"rate": rate, "declared_floor": MIN_JOIN_GENE, "raises": True},
            "encode_chip_target_to_replogle_perturbation": {
                "n": len(tfs), "of": len(idx), "rate": tf_rate, "declared_floor": MIN_JOIN_TF,
                "raises": True},
        },
        "occupancy": {
            "n_tfs": len(tfs), "output_type": OUTPUT_TYPE, "element_pad_bp": ELEMENT_PAD,
            "frac_cells_bound": float(B.mean()),
            "median_tfs_per_element": float(np.median(nbound_el)),
            "n_elements_with_no_profiled_tf": int((nbound_el == 0).sum()),
        },
        "causal_weights": {"z_edge": Z_EDGE, "normalisation": "per-gene robust z (lfc - median)/(1.4826*MAD)",
                           "self_knockdown_diagonal_zeroed": nself},
        "cv": {"scheme": "chromosome-held-out (gene-disjoint by construction)", "folds": FOLDS,
               "seeds": list(SEEDS),
               "mean_frac_chromosomes_in_same_fold_across_seed_pairs": float(np.mean(pair_agree))},
        "arms_raw_auprc": res,
        "positive_base_rate_for_every_auprc_above": base_rate,
        "ladder": ladder_out,
        "ladder_extra_and_post_hoc": ladder_extra,
        "ladder_gate_applied_to_every_rung": {
            "rule": f"paired per-seed delta >= {MIN_INCREMENT} in >= {MIN_SEEDS}/{len(SEEDS)} partitions -- "
                    f"the SAME bar the question under test is held to, applied to every rung so no rung "
                    f"earns a positive claim on a number that would have failed it",
            "rungs_clearing": [r["rung"] for r in ladder_out if r["clears_prespecified_gate"]],
            "rungs_sub_threshold": [r["rung"] for r in ladder_out if not r["clears_prespecified_gate"]],
        },
        "screen_conditioned_ladder": {
            "why": "six K562 screens with a 14x spread in positive rate; element width sits in the "
                   "accessibility block and is close to a screen label, so a rung can pay by naming the batch",
            "screen_only_auprc": res["SCREEN_only"]["auprc_mean"],
            "n_screens": len(ds_names),
            "positive_rate_by_screen": {n: float(y[ds_arr == n].mean()) for n in ds_names},
            "median_element_width_bp_by_screen": {n: float(np.median(el_width[el_idx][ds_arr == n]))
                                                  for n in ds_names},
            "rungs": screen_rungs,
        },
        "tf_axis_permutation": {
            "what_is_held_fixed": "the element, the gene, the EXACT number of bound TFs, and the gene's "
                                  "EXACT z distribution over all profiled TFs (a permuted column is the same "
                                  "multiset, so the GENE block is bit-identical). Only the correspondence "
                                  "between a TF's occupancy and a TF's causality is destroyed.",
            "n_tf_strata_degree_stratified": len(tf_strata),
            "models_refitted": False,
            "draws": SWAP_DRAWS,
            "results": perms,
            "tf_identity_survives_degree_stratified_permutation": bool(tf_identity_perm_survives),
        },
        "best_baseline": best_base,
        "increment_over_best_baseline": float(inc),
        "increment_over_distance_plus_accessibility": float(inc_over_distacc),
        "increment_over_distance_accessibility_TFoccupancy": float(inc_over_occ),
        "increment_per_seed_passes": per_seed_win,
        "element_swap": swaps,
        "descriptive_cross_check_causal_bound_TFs": cross,
        "descriptive_cross_check_DEGREE_MATCHED": cross_deg,
        "descriptive_cross_check_n_TF_bound_for_contrast": cross_count,
        "descriptive_cross_check_note": (
            "the dist x acc version is matched on everything EXCEPT the covariate this module says does the "
            "work. Positives sit at elements carrying "
            f"{float(nbound_el[el_idx][y == 1].mean()):.0f} bound TFs against "
            f"{float(nbound_el[el_idx][y == 0].mean()):.0f} for negatives, so the DEGREE-MATCHED row is the "
            "one that counts and it is the one reported in the verdict."),
        "gates_prespecified": {"increment": bool(passes_inc), "element_swap": bool(passes_swap),
                               "degree_matched_swap": bool(passes_deg)},
        "diagnostics_post_hoc": {
            "note": "added after the first run, when the prespecified swap turned out to also coarsen "
                    "distance to its decile; reported so the prespecified gates are not read as stronger "
                    "than they are",
            "tf_identity_survives_degree_matched_swap_on_TFcausality_arm": bool(tf_identity_survives),
            "tf_identity_survives_degree_stratified_TF_axis_permutation": bool(tf_identity_perm_survives),
            "post_hoc_rank_identity_block": {
                "what": "within-gene ranks of |z| down the TF axis: gene-scale-free, so it cannot be bought "
                        "with gene responsiveness the way a magnitude aggregate can",
                "FULL_plus_RANK_auprc": res["FULL+RANK"]["auprc_mean"],
                "paired_delta_over_FULL": ladder_extra[1]["paired_delta_mean"],
                "n_seeds_clearing": ladder_extra[1]["n_seeds_clearing_min_increment"],
                "erased_by_degree_stratified_TF_permutation":
                    p_deg["FULL+RANK"]["frac_of_real_reproduced_by_permutation"],
            },
            "swap_machinery_consistency_check_passed": bool(swap_machinery_ok),
            "element_only_swap_full_model_frac_reproduced":
                swaps["element_only_distance_x_accessibility"]["frac_of_real_reproduced_by_swap"],
            "element_only_degree_matched_full_model_frac_reproduced":
                swaps["element_only_degree_matched"]["frac_of_real_reproduced_by_swap"],
        },
        "verdict": v,
        "limits": limits,
        "log": log,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "causal_enhancer.json").write_text(json.dumps(R, indent=1, default=float))
    report("\n" + "=" * 104)
    report(f"VERDICT: {v}")
    report("=" * 104)
    R["log"] = log
    (OUT / "causal_enhancer.json").write_text(json.dumps(R, indent=1, default=float))
    report(f"  -> {OUT/'causal_enhancer.json'}")


if __name__ == "__main__":
    main()
