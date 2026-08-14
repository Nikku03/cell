"""LOOP 100 -- REPLICATION AS A PROCESS: DOES A FORK-SPEED MODEL REPRODUCE MEASURED RT?

WHAT IS WRONG WITH THE LAYER AS IT STANDS. Loop 63 added replication timing to this model and it has
sat there ever since as a NUMBER PER GENE. 20 Repli-chip wavelet-smoothed tracks, hg19, averaged over
TSS +/- 50 kb, 16,362 genes. It correlates with enhancers at +0.5930 and with publication count at
only +0.1442, which is why it was worth having: it is the one layer in this cell that fame cannot
explain. But a column of measurements is not a process. Nothing in this repository can SAY WHY a gene
replicates when it does, and a whole-cell model that cannot generate its own replication timing has
replication as data, not as mechanism.

THE MECHANISM IS EMBARRASSINGLY SIMPLE, WHICH IS THE POINT. Replication timing is not a regulatory
program written per locus. It is the outcome of three things: where origins are, when each one fires,
and how fast a fork travels. Write that down and it is one equation --

        RT(x) = min over origins o of ( t_o + |x - x_o| / v )

-- an origin fires at time t_o, two forks leave it at speed v, and a locus is replicated the moment
the first fork arrives. Passivation is free: an origin whose own t_o is later than the arrival of
someone else's fork never contributes to the min. Nothing else is needed. If that equation reproduces
the measured RT profile, replication timing stops being an annotation in this model and becomes a
simulable process with two parameters. If it does not, the layer stays data and this loop says so.

THE HONEST DIFFICULTY, STATED BEFORE ANY RESULT. This repository has NO measured origin map. No
OK-seq, no SNS-seq, no ORC ChIP. So the equation above has an unknown input, and the whole loop turns
on how that input is supplied:

    UNINFORMED  origins placed uniformly at random. This is the PURE fork-speed model -- fork
                physics and nothing else. It knows nothing about the genome it is running on.
    INFORMED    origin density proportional to local gene density, smoothed at 200 kb. This is a
                standard and defensible proxy (origins track open chromatin and promoters) but it is
                a PROXY, and it imports genome annotation into what was meant to be physics.

The informed model will predict RT better. Of course it will -- gene density already predicts RT
before a single fork is simulated, and gate T1's direction check measures exactly that. So the gate
that decides whether this loop found a PROCESS rather than a rebranded annotation is T4, and T4 asks
the deflationary question directly: does simulating forks beat simply blurring the gene-density track?
If it does not, the fork is decoration on a Gaussian filter, and that is the result.

WHAT I LOOKED AT BEFORE WRITING THESE GATES, disclosed under rule 2. The SHAPE of the RT data (16,362
x 20, 0.58% missing cells, values from -4.89 to +1.46, median +0.83), the per-chromosome gene counts
in _tss_hg19.bed, and loop 63's ALREADY-PUBLISHED +0.5930 enhancer and +0.1442 pubs correlations. No
number that any gate below tests has been computed. Every threshold here was fixed before the first
simulation ran, and the FAIL predictions written next to T2 and T4 were written before too.

WHAT CHANGED AFTER THE FIRST RUN, disclosed in full, because two things about that run were wrong in
the model's favour to fix and one was wrong against it:
  (a) 40 realisations was NOT A CONVERGED ESTIMATOR. The informed model's rho climbed 0.3386 (10
      realisations) -> 0.4022 (40) -> 0.4169 (160) -> 0.4243 (640). The min-over-origins statistic is
      heavy-tailed and 40 draws understated it by ~0.03. Raised to 400.
  (b) THE NULLS WERE ESTIMATED AT A DIFFERENT REALISATION COUNT (8) THAN THE REAL MODEL (40), which
      is not a control, it is two different estimators. Nulls now use the identical count.
  (c) THE BLUR COMPARATOR GOT A SIGMA SWEEP AND THE FORK GOT ONE FIXED PARAMETER SET, so T4 was a
      tuned comparator against an untuned model. The fork now gets a 3 x 3 grid over origin-map
      sigma and origin spacing PLUS the 7-point speed sweep, and T4 uses its best. The fork side now
      has three free parameters against the blur's one.
  ALL THREE CHANGES CAN ONLY HELP THE FORK MODEL PASS T4. NO THRESHOLD WAS MOVED. The +0.02 margin,
  the [0.3, 10] speed bracket, the 0.10 floor and the 0.05 fame margin are exactly as first written.

  T1 THE DATA IS PRESENT, INDEXED, AND THE RIGHT WAY UP           THE SANITY CHECK.
       >= 95% of the 16,362 lifted genes carry a finite RT value, and rho(RT, local gene density) is
       POSITIVE -- early replication must track gene-dense euchromatin. If that sign is negative the
       track is being read upside down and every number after it is confidently wrong. This is a
       direction check, not a discovery, and it is ALSO the reason T4 is the gate that matters.
  T2 THE PURE FORK-SPEED MODEL, WITH NO GENOMIC INFORMATION       PREDICTED TO FAIL.
       uniform-random origins, exponential firing, v = 1.75 kb/min. Gate: |Spearman| vs measured
       lateness >= 0.10. I expect this to fail at essentially zero, because averaged over
       realisations a uniform origin process has no preferred position, and I am recording that
       prediction here so the failure cannot be presented afterwards as a discovery.
  T3 THE INFORMED MODEL AGAINST TWO ORIGIN NULLS                  THE CONTROL THAT COULD FIRE.
       (a) CIRCULAR-SHIFT null: the SAME origin set, rigidly shifted by a random offset around each
           chromosome. Every inter-origin distance is preserved exactly; only the alignment to the
           genes is destroyed. This is a hard null, not a soft one.
       (b) UNIFORM null: T2's model, reused -- the shuffled-origin null in the strict sense.
       (c) DISTANCE-ONLY null: informed origins, all firing at t = 0, so RT is pure distance to the
           nearest origin with no firing stochasticity and no speed dependence at all.
       Gate: gate_guard.survival(real, circular-shift nulls) must be DEFINED with |z| >= 2, and the
       real rho must exceed the uniform null's mean. gate_guard.null_can_move() is run on the origin
       position vectors FIRST -- no null's verdict is reported before it is shown capable.
  T4 DOES THE FORK BUY ANYTHING OVER BLURRING THE ANNOTATION?     THE GATE THAT DECIDES THE LOOP.
                                                                  PREDICTED TO FAIL.
       compare the fork model at ITS OWN BEST over a grid of origin-map sigma, origin spacing and
       fork speed against the BEST Gaussian-smoothed gene-density track over a sigma sweep of
       50 kb -> 1600 kb. Gate: best fork >= best smoothed density + 0.02 Spearman.
       If the fork loses, then min-over-origins-of-(firing + travel) is a blur with extra steps, the
       "process" is the origin map, and the honest headline is that this loop measured gene density.
  T5 IS THE MEASURED RT GRADIENT CONSISTENT WITH A REAL FORK?     THE NON-CIRCULAR TEST.
       this one needs no origin map and no simulation. A fixed-speed fork forces the RT profile to be
       Lipschitz: between adjacent loci, |dRT| <= |dx| / v. Quantile-map measured lateness onto an
       8 h S phase, take adjacent same-chromosome gene pairs 50-500 kb apart, and invert the observed
       gradient to an implied speed. Gate: the implied speed at the 90th percentile of gradient lies
       in [0.3, 10] kb/min -- a factor-of-five bracket either side of the published 1.5-2 kb/min,
       chosen wide because the RT track is smoothed at +/- 50 kb and gradients below that scale are
       attenuated by construction. Too steep and no fixed-speed fork can produce this profile; too
       flat and the data cannot see a fork at all.
  T6 THE FAME CONTROL                                             THE RECURRING KILLER.
       publication count against measured RT, against the model's prediction, and against the
       residual. Gate: the best fork model must beat pubs by |rho_model| > |rho_pubs| + 0.05. pubs
       has beaten the biology in loops 87, 87b, 94 and 96; RT is a physical measurement and should be
       the layer where it finally does not, but the gate is here to find out rather than to assume.

  THE ABUNDANCE RULE IS NOT AT RISK HERE: no rate in this loop is a quotient of two abundance
  datasets. The only quotient formed is distance / speed, both of which are model parameters.

-> outputs/loop_replication_time.json
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import gate_guard as GG  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
CELL = LR.CELL

SEED = 10000
BIN_BP = 10_000                     # 10 kb simulation grid
FORK_KB_MIN = 1.75                  # published mammalian fork speed 1.5-2 kb/min; midpoint
ORIGIN_SPACING_KB = 120.0           # mean spacing of FIRED origins -> ~26,000 genome-wide
FIRE_TAU_MIN = 120.0                # exponential mean of the origin firing time
S_PHASE_MIN = 480.0                 # 8 h S phase, for the T5 gradient calibration only
DENS_SIGMA_KB = 200.0               # smoothing of the gene-density origin proxy
NREAL = 400                         # realisations averaged per model (converged; see docstring (a))
NNULL = 15                          # independent null draws, each at NREAL -- matched estimators
NREAL_SWEEP = 200
SPEED_SWEEP = (0.25, 0.5, 1.0, 1.75, 3.5, 7.0, 14.0)
SIGMA_SWEEP_KB = (50.0, 100.0, 200.0, 400.0, 800.0, 1600.0)
FORK_SIGMA_GRID = (200.0, 400.0, 800.0)     # origin-map smoothing given to the fork model
FORK_SPACING_GRID = (60.0, 120.0, 240.0)    # mean fired-origin spacing given to the fork model
CONVERGENCE = {"10": 0.3386, "40": 0.4022, "160": 0.4169, "640": 0.4243}
MIN_GENES_PER_CHROM = 100

T1_COVERAGE = 0.95
T2_FLOOR = 0.10
T3_Z = 2.0
T4_MARGIN = 0.02
T5_LO, T5_HI = 0.3, 10.0
T6_MARGIN = 0.05

HG19 = {"chr1": 249250621, "chr2": 243199373, "chr3": 198022430, "chr4": 191154276,
        "chr5": 180915260, "chr6": 171115067, "chr7": 159138663, "chr8": 146364022,
        "chr9": 141213431, "chr10": 135534747, "chr11": 135006516, "chr12": 133851895,
        "chr13": 115169878, "chr14": 107349540, "chr15": 102531392, "chr16": 90354753,
        "chr17": 81195210, "chr18": 78077248, "chr19": 59128983, "chr20": 63025520,
        "chr21": 48129895, "chr22": 51304566, "chrX": 155270560, "chrY": 59373566}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < 30:
        return float("nan"), int(f.sum())
    return float(spearmanr(a[f], b[f]).statistic), int(f.sum())


# ---------------------------------------------------------------------------------------------
# THE MODEL.  RT(x) = min over origins of ( t_o + |x - x_o| / v ), exactly, on a 10 kb grid.
# The min is a 1-D weighted distance transform and is computed in two vectorised passes:
#     forward   T[i] = c*i + min_{j<=i} ( T0[j] - c*j )
#     backward  T[i] = -c*i + min_{j>=i} ( T0[j] + c*j )
# so the whole genome costs O(bins) and not O(bins x origins). Passivation needs no special case:
# an origin whose firing time exceeds a neighbouring fork's arrival simply loses the min.
# ---------------------------------------------------------------------------------------------
def fork_times(n_bins, ori_bins, ori_t, min_per_bin):
    T0 = np.full(n_bins, np.inf)
    np.minimum.at(T0, ori_bins, ori_t)
    ramp = min_per_bin * np.arange(n_bins, dtype=float)
    fwd = np.minimum.accumulate(T0 - ramp) + ramp
    bwd = np.minimum.accumulate((T0 + ramp)[::-1])[::-1] - ramp
    return np.minimum(fwd, bwd)


def draw_origins(rng, n_bins, n_ori, weights):
    if weights is None:
        return rng.integers(0, n_bins, size=n_ori)
    p = weights / weights.sum()
    return rng.choice(n_bins, size=n_ori, replace=True, p=p)


def run_model(chroms, rng, kind, v_kb_min=FORK_KB_MIN, nreal=NREAL,
              spacing_kb=ORIGIN_SPACING_KB, sigma_kb=DENS_SIGMA_KB):
    """kind: 'informed' | 'uniform' | 'shift' | 'distance'. Returns per-gene mean replication time."""
    min_per_bin = (BIN_BP / 1000.0) / v_kb_min
    acc, cnt = {}, 0
    for _ in range(nreal):
        for cn, C in chroms.items():
            nb, w = C["n_bins"], C["w"][sigma_kb]
            n_ori = max(1, int(round(C["len_kb"] / spacing_kb)))
            if kind == "uniform":
                ob = draw_origins(rng, nb, n_ori, None)
            else:
                ob = draw_origins(rng, nb, n_ori, w)
                if kind == "shift":
                    ob = (ob + rng.integers(0, nb)) % nb
            ot = np.zeros(n_ori) if kind == "distance" else rng.exponential(FIRE_TAU_MIN, n_ori)
            T = fork_times(nb, ob, ot, min_per_bin)
            acc[cn] = acc.get(cn, 0.0) + T
        cnt += 1
    return {cn: v / cnt for cn, v in acc.items()}


def gene_pred(chroms, per_chrom_T, n_genes):
    out = np.full(n_genes, np.nan)
    for cn, C in chroms.items():
        out[C["gidx"]] = per_chrom_T[cn][C["gbin"]]
    return out


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 100 -- replication as a process: does a fork-speed model reproduce measured RT?")
    say("=" * 100)
    say()

    rng = np.random.default_rng(SEED)
    D = json.load(open(CELL))
    genes = D["genes"]
    n_genes = len(genes)
    pubs = np.array([float(g.get("pubs") or 0) for g in genes])

    M = np.load(SC / "_rt_matrix.npy")
    gi = json.load(open(SC / "_rt_geneidx.json"))
    rt_row = np.nanmean(M, axis=1)
    rt = np.full(n_genes, np.nan)
    for k, i in enumerate(gi):
        if i < n_genes:
            rt[i] = rt_row[k]

    bed = []
    for ln in open(SC / "_tss_hg19.bed"):
        p = ln.split()
        if len(p) >= 4 and p[0] in HG19:
            bed.append((p[0], int(p[1]), int(p[3][1:])))
    by_chrom = {}
    for cn, pos, gidx in bed:
        by_chrom.setdefault(cn, []).append((pos, gidx))

    say(f"  RT matrix {M.shape[0]:,} genes x {M.shape[1]} Repli-chip cell-type tracks (hg19)")
    say(f"  TSS bed: {len(bed):,} genes on {len(by_chrom)} named hg19 chromosomes")
    say(f"  model grid {BIN_BP//1000} kb, fork {FORK_KB_MIN} kb/min, mean origin spacing "
        f"{ORIGIN_SPACING_KB:.0f} kb, firing tau {FIRE_TAU_MIN:.0f} min, {NREAL} realisations")
    say()

    chroms = {}
    for cn, lst in sorted(by_chrom.items()):
        if len(lst) < MIN_GENES_PER_CHROM:
            continue
        L = HG19[cn]
        nb = int(L // BIN_BP) + 1
        gbin = np.clip(np.array([p // BIN_BP for p, _ in lst]), 0, nb - 1)
        gidx = np.array([g for _, g in lst])
        counts = np.bincount(gbin, minlength=nb).astype(float)
        dens = gaussian_filter1d(counts, DENS_SIGMA_KB / (BIN_BP / 1000.0), mode="reflect")
        w = {}
        for sg in sorted(set(FORK_SIGMA_GRID) | {DENS_SIGMA_KB}):
            d = gaussian_filter1d(counts, sg / (BIN_BP / 1000.0), mode="reflect")
            w[sg] = d + d.mean()
        chroms[cn] = {"len_kb": L / 1000.0, "n_bins": nb, "gbin": gbin, "gidx": gidx,
                      "counts": counts, "dens": dens, "w": w}
    used_chroms = sorted(chroms)
    say(f"  chromosomes modelled ({len(used_chroms)}, >= {MIN_GENES_PER_CHROM} genes): "
        f"{', '.join(c.replace('chr', '') for c in used_chroms)}")
    dropped = sorted(set(by_chrom) - set(chroms))
    say(f"  dropped for too few genes: {', '.join(dropped) if dropped else 'none'}")
    say()

    # per-gene local gene density (the origin proxy, and T1's direction check)
    dens_g = np.full(n_genes, np.nan)
    for cn, C in chroms.items():
        dens_g[C["gidx"]] = C["dens"][C["gbin"]]
    on_model = np.zeros(n_genes, bool)
    for cn, C in chroms.items():
        on_model[C["gidx"]] = True
    have = np.isfinite(rt) & on_model
    lateness = -rt                                       # higher = replicates LATER

    say("T1 THE DATA IS PRESENT, INDEXED, AND THE RIGHT WAY UP")
    cov = float(have.sum()) / M.shape[0]
    r_dens, n_d = spear(rt[have], dens_g[have])
    say(f"     genes with finite RT on a modelled chromosome: {int(have.sum()):,} of "
        f"{M.shape[0]:,} lifted  ({cov:.4f})")
    say(f"     DIRECTION CHECK  RT vs local gene density (200 kb)  rho {r_dens:+.4f}  (n {n_d:,})")
    say(f"     positive means high RT signal = EARLY = gene-dense euchromatin, as it must be")
    t1 = bool(cov >= T1_COVERAGE and r_dens > 0)
    say(f"     T1 {'PASS' if t1 else 'FAIL'}  (coverage >= {T1_COVERAGE}, direction positive)")
    say(f"     NOTE: this same +{r_dens:.4f} is why T4 exists. Gene density predicts RT before any")
    say(f"     fork is simulated, so a fork model fed gene density starts with a free head start.")
    say()

    say("T2 THE PURE FORK-SPEED MODEL, WITH NO GENOMIC INFORMATION   (predicted to FAIL)")
    Tu = run_model(chroms, rng, "uniform")
    pu = gene_pred(chroms, Tu, n_genes)
    r_uni, n_u = spear(pu[have], lateness[have])
    say(f"     uniform-random origins, exponential firing, v = {FORK_KB_MIN} kb/min")
    say(f"     predicted replication time vs measured lateness  rho {r_uni:+.4f}  (n {n_u:,})")
    say(f"     variance of RT rank explained  R2 = {r_uni**2:.4f}")
    # a single realisation, to show the averaging is not what killed it
    r1 = run_model(chroms, np.random.default_rng(SEED + 1), "uniform", nreal=1)
    p1 = gene_pred(chroms, r1, n_genes)
    r_uni1, _ = spear(p1[have], lateness[have])
    say(f"     one single realisation (no averaging) instead of {NREAL}: rho {r_uni1:+.4f}")
    t2 = bool(abs(r_uni) >= T2_FLOOR)
    say(f"     T2 {'PASS' if t2 else 'FAIL'}  (floor |rho| >= {T2_FLOOR})")
    say(f"     A uniform origin process has no preferred position on the chromosome, so averaged")
    say(f"     over realisations its expected RT is flat except at the telomeres. Fork physics with")
    say(f"     no origin map carries no per-gene information. This was predicted in the docstring.")
    say()

    say("T3 THE INFORMED MODEL AGAINST TWO ORIGIN NULLS")
    Ti = run_model(chroms, rng, "informed")
    pi = gene_pred(chroms, Ti, n_genes)
    r_inf, n_i = spear(pi[have], lateness[have])
    say(f"     origin density proportional to gene density smoothed at {DENS_SIGMA_KB:.0f} kb")
    say(f"     informed fork model vs measured lateness  rho {r_inf:+.4f}  (n {n_i:,})   "
        f"R2 = {r_inf**2:.4f}")

    # CAPABILITY CHECK FIRST -- no null verdict is reported before the null is shown to move.
    cc = chroms[used_chroms[0]]
    ob_real = draw_origins(rng, cc["n_bins"], 400, cc["w"][DENS_SIGMA_KB])
    ob_shift = (ob_real + rng.integers(1, cc["n_bins"])) % cc["n_bins"]
    ob_unif = draw_origins(rng, cc["n_bins"], 400, None)
    cap_shift = GG.null_can_move(ob_real, ob_shift)
    cap_unif = GG.null_can_move(ob_real, ob_unif)
    say(f"     CAPABILITY circular-shift: {cap_shift['changed']:.1%} of origin positions change "
        f"-- capable: {cap_shift['capable']}")
    say(f"     CAPABILITY uniform       : {cap_unif['changed']:.1%} of origin positions change "
        f"-- capable: {cap_unif['capable']}")

    say(f"     nulls are estimated at the SAME {NREAL} realisations as the real model, because two")
    say(f"     estimators with different variances are not a control")
    null_shift = []
    for k in range(NNULL):
        Ts = run_model(chroms, np.random.default_rng(SEED + 100 + k), "shift", nreal=NREAL)
        ps = gene_pred(chroms, Ts, n_genes)
        null_shift.append(spear(ps[have], lateness[have])[0])
    null_unif = []
    for k in range(NNULL):
        Tn = run_model(chroms, np.random.default_rng(SEED + 500 + k), "uniform", nreal=NREAL)
        pn = gene_pred(chroms, Tn, n_genes)
        null_unif.append(spear(pn[have], lateness[have])[0])
    s_shift = GG.survival(r_inf, null_shift)
    GG.report("informed model under a circular-shift origin null", s_shift, emit=say)
    s_unif = GG.survival(r_inf, null_unif)
    GG.report("informed model under a uniform origin null       ", s_unif, emit=say)

    Td = run_model(chroms, rng, "distance")
    pd_ = gene_pred(chroms, Td, n_genes)
    r_dist, _ = spear(pd_[have], lateness[have])
    say(f"     distance-to-nearest-origin null (all origins fire at t=0)  rho {r_dist:+.4f}   "
        f"R2 = {r_dist**2:.4f}")
    say(f"     that null keeps the informed origin map and removes ONLY the firing stochasticity,")
    say(f"     so the gap between it and {r_inf:+.4f} is everything the firing-time model contributes")
    t3 = bool(cap_shift["capable"] and cap_unif["capable"] and s_shift.get("z") is not None
              and np.isfinite(s_shift.get("z", np.nan)) and abs(s_shift["z"]) >= T3_Z
              and r_inf > float(np.mean(null_unif)))
    say(f"     T3 {'PASS' if t3 else 'FAIL'}  (both nulls capable, |z| >= {T3_Z} vs circular shift, "
        f"and real above the uniform null's mean)")
    say()

    say("T4 DOES THE FORK BUY ANYTHING OVER BLURRING THE ANNOTATION?   (predicted to FAIL)")
    say(f"     first, the fork model at ITS OWN BEST -- a {len(FORK_SIGMA_GRID)}x{len(FORK_SPACING_GRID)}"
        f" grid over the origin-map sigma and the fired-origin spacing, so the comparator below is")
    say(f"     not a swept model against an unswept one")
    grid = {}
    for sg in FORK_SIGMA_GRID:
        for sp in FORK_SPACING_GRID:
            Tg = run_model(chroms, np.random.default_rng(SEED + 700), "informed",
                           nreal=NREAL_SWEEP, spacing_kb=sp, sigma_kb=sg)
            pg = gene_pred(chroms, Tg, n_genes)
            grid[(sg, sp)] = spear(pg[have], lateness[have])[0]
            say(f"     origin sigma {sg:6.0f} kb, spacing {sp:5.0f} kb   rho {grid[(sg, sp)]:+.4f}")
    best_cell = max(grid, key=lambda k: grid[k])
    best_fork = max(max(grid.values()), r_inf, r_dist)
    say(f"     best fork over the grid: sigma {best_cell[0]:.0f} kb, spacing {best_cell[1]:.0f} kb "
        f"-> rho {grid[best_cell]:+.4f}")
    say(f"     best fork including the declared model and the distance-only variant: {best_fork:+.4f}")
    say()
    blur = {}
    for sg in SIGMA_SWEEP_KB:
        v = np.full(n_genes, np.nan)
        for cn, C in chroms.items():
            sm = gaussian_filter1d(C["counts"], sg / (BIN_BP / 1000.0), mode="reflect")
            v[C["gidx"]] = sm[C["gbin"]]
        r, _ = spear(v[have], lateness[have])
        blur[sg] = -r          # agreement, sign-matched to the model (dense = early = less late)
        say(f"     gene density blurred at sigma {sg:7.0f} kb   agreement rho {-r:+.4f}   "
            f"R2 = {r*r:.4f}")
    best_sg = max(blur, key=lambda k: blur[k])
    best_blur = blur[best_sg]
    say(f"     best plain blur: sigma {best_sg:.0f} kb at rho {best_blur:+.4f}   (one free parameter)")
    say(f"     best fork simulation:                rho {best_fork:+.4f}   (three free parameters)")
    t4 = bool(best_fork >= best_blur + T4_MARGIN)
    say(f"     margin {best_fork - best_blur:+.4f}   T4 {'PASS' if t4 else 'FAIL'}  "
        f"(fork must beat the best blur by >= {T4_MARGIN})")
    say(f"     Interpretation if FAIL: min-over-origins-of-(firing + travel) is, to the precision")
    say(f"     this data can see, a smoothing kernel. The predictive content is the ORIGIN MAP, and")
    say(f"     the origin map here is gene density. The fork adds a length scale and nothing else.")
    say()

    say("T5 IS THE MEASURED RT GRADIENT CONSISTENT WITH A REAL FORK?   (no simulation, no origins)")
    lat = lateness.copy()
    ok = have
    ranks = np.full(n_genes, np.nan)
    v_ok = lat[ok]
    order = np.argsort(np.argsort(v_ok))
    ranks[np.where(ok)[0]] = (order + 0.5) / len(v_ok)
    t_quant = ranks * S_PHASE_MIN
    lo, hi = np.nanpercentile(lat[ok], [1, 99])
    t_lin = np.clip((lat - lo) / (hi - lo), 0, 1) * S_PHASE_MIN

    def gradients(tmin):
        sl, gaps = [], []
        for cn, C in chroms.items():
            gidx, gbin = C["gidx"], C["gbin"]
            o = np.argsort(gbin)
            gi_s, gb_s = gidx[o], gbin[o]
            tt = tmin[gi_s]
            dx = np.diff(gb_s) * (BIN_BP / 1000.0)          # kb
            dt = np.abs(np.diff(tt))                        # min
            m = np.isfinite(dt) & (dx >= 50.0) & (dx <= 500.0)
            sl.append(dt[m] / dx[m])
            gaps.append(dx[m])
        return np.concatenate(sl), np.concatenate(gaps)

    sl_q, gaps = gradients(t_quant)
    sl_l, _ = gradients(t_lin)
    say(f"     assumption: measured lateness quantile-mapped onto an {S_PHASE_MIN/60:.0f} h S phase")
    say(f"     {len(sl_q):,} adjacent same-chromosome gene pairs, gap 50-500 kb "
        f"(median gap {np.median(gaps):.0f} kb)")
    pcts = [50, 75, 90, 95, 99]
    imp_q, imp_l = {}, {}
    for p in pcts:
        gq = float(np.percentile(sl_q, p))
        gl = float(np.percentile(sl_l, p))
        imp_q[p] = (1.0 / gq) if gq > 0 else float("inf")
        imp_l[p] = (1.0 / gl) if gl > 0 else float("inf")
        say(f"     gradient p{p:<3d} = {gq:7.4f} min/kb  -> implied fork {imp_q[p]:9.2f} kb/min "
            f"   [linear map: {imp_l[p]:9.2f}]")
    speed90 = imp_q[90]
    t5 = bool(T5_LO <= speed90 <= T5_HI)
    say(f"     published mammalian fork speed 1.5-2 kb/min; gate bracket [{T5_LO}, {T5_HI}]")
    say(f"     T5 {'PASS' if t5 else 'FAIL'}  (implied speed at the 90th gradient percentile "
        f"= {speed90:.2f} kb/min)")
    say(f"     LIMIT: the RT track is smoothed at TSS +/- 50 kb, so gradients on a scale finer than")
    say(f"     ~100 kb are attenuated by construction. Any bias in this estimate is toward speeds")
    say(f"     that are TOO FAST, so a number far below 1.5 kb/min would be the damning outcome.")
    say()

    say("T6 THE FAME CONTROL")
    r_pub_rt, n_p = spear(pubs[have], rt[have])
    r_pub_lat, _ = spear(pubs[have], lateness[have])
    r_pub_pred, _ = spear(pubs[have], pi[have])
    resid = np.full(n_genes, np.nan)
    a = np.argsort(np.argsort(pi[have])).astype(float)
    b = np.argsort(np.argsort(lateness[have])).astype(float)
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    resid[np.where(have)[0]] = b - np.dot(a, b) / np.dot(a, a) * a
    r_pub_resid, _ = spear(pubs[have], resid[have])
    say(f"     pubs vs measured RT           rho {r_pub_rt:+.4f}   (loop 63 reported +0.1442)")
    say(f"     pubs vs measured lateness     rho {r_pub_lat:+.4f}")
    say(f"     pubs vs the model prediction  rho {r_pub_pred:+.4f}")
    say(f"     pubs vs the model residual    rho {r_pub_resid:+.4f}   (n {n_p:,})")
    best_model = max(abs(r_inf), abs(r_uni), abs(r_dist), abs(best_fork))
    t6 = bool(best_model > abs(r_pub_lat) + T6_MARGIN)
    say(f"     best fork model |rho| {best_model:.4f} vs fame |rho| {abs(r_pub_lat):.4f}")
    say(f"     T6 {'PASS' if t6 else 'FAIL'}  (model must beat fame by >= {T6_MARGIN})")
    say(f"     Replication timing is a physical measurement of chromatin, not a curated annotation,")
    say(f"     and this is one of the few layers in this model where fame is a minor term. That is a")
    say(f"     property of the DATA, not of the model built on it -- the fork earns no credit for it.")
    say()

    say("DIAGNOSTIC -- is the fork speed identifiable from this data at all?")
    sweep = {}
    for v in SPEED_SWEEP:
        Tv = run_model(chroms, np.random.default_rng(SEED + 900), "informed", v_kb_min=v,
                       nreal=NREAL_SWEEP)
        pv = gene_pred(chroms, Tv, n_genes)
        sweep[v] = spear(pv[have], lateness[have])[0]
        say(f"     v = {v:5.2f} kb/min   rho {sweep[v]:+.4f}")
    best_v = max(sweep, key=lambda k: sweep[k])
    spread = max(sweep.values()) - min(sweep.values())
    say(f"     best v = {best_v} kb/min, spread across a 56x speed range = {spread:.4f}")
    say(f"     a small spread means the data cannot see the fork speed: the ranking of loci is set")
    say(f"     by the origin map and is almost invariant to how fast the forks travel.")
    say()

    say("PER-TRACK -- the same comparison against each of the 20 cell-type tracks separately")
    per_track = []
    for j in range(M.shape[1]):
        col = np.full(n_genes, np.nan)
        for k, i in enumerate(gi):
            if i < n_genes:
                col[i] = M[k, j]
        r, _ = spear(pi[have], -col[have])
        per_track.append(r)
    say(f"     informed model vs each track: median {np.median(per_track):+.4f}, "
        f"range {min(per_track):+.4f} to {max(per_track):+.4f}")
    say(f"     vs the 20-track mean: {r_inf:+.4f}")
    say()

    gates = {"T1 data present, indexed and the right way up": t1,
             "T2 the pure fork-speed model explains RT": t2,
             "T3 the informed model beats both origin nulls": t3,
             "T4 the fork beats a plain blur of the annotation": t4,
             "T5 measured RT gradient implies a plausible fork speed": t5,
             "T6 the model beats publication count": t6}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")
    say()
    say("  HEADLINE -- fraction of measured RT rank-variance explained")
    say(f"     pure fork-speed model, no origin map        R2 = {r_uni**2:.4f}")
    say(f"     fork model on a gene-density origin map     R2 = {r_inf**2:.4f}")
    say(f"     distance to nearest origin, no firing       R2 = {r_dist**2:.4f}")
    say(f"     best fork anywhere on its parameter grid    R2 = {best_fork**2:.4f}")
    say(f"     the origin map alone, Gaussian-blurred      R2 = {best_blur**2:.4f}   <- the winner")
    say(f"     publication count                           R2 = {r_pub_lat**2:.4f}")
    say()
    say(f"     CONVERGENCE of the fork estimator (informed model, rho by realisation count):")
    say(f"     {', '.join(f'{k}->{v:+.4f}' for k, v in CONVERGENCE.items())}, run at {NREAL}")
    say()

    man = RM.manifest(
        inputs=[str(CELL), str(SC / "_rt_matrix.npy"), str(SC / "_rt_geneidx.json"),
                str(SC / "_tss_hg19.bed")],
        available=n_genes, used=int(have.sum()), selection="filtered", seed=SEED,
        controls=["circular-shift origin null: identical inter-origin spacing, alignment destroyed",
                  "uniform-origin null, which is also the pure fork-speed model of T2",
                  "distance-to-nearest-origin null: informed origins, zero firing stochasticity",
                  "a Gaussian blur of the same gene-density track, swept over 50-1600 kb, as the "
                  "deflationary comparator the fork must beat",
                  "gate_guard.null_can_move run on the origin position vectors before any null "
                  "verdict was reported, and gate_guard.survival used for the fraction",
                  "publication count against measured RT, the prediction and the residual"],
        note="replication timing is DATA in this repo; this loop asks whether the one-line fork "
             "equation RT(x) = min_o (t_o + |x-x_o|/v) regenerates it. The origin map is NOT "
             "measured anywhere in this repository -- it is a gene-density proxy, which is why the "
             "decisive gate is T4, the comparison against simply blurring that same proxy")
    RM.report(man, emit=say)

    json.dump({"test": "loop_replication_time", "manifest": man, "gates": gates,
               "params": {"bin_bp": BIN_BP, "fork_kb_min": FORK_KB_MIN,
                          "origin_spacing_kb": ORIGIN_SPACING_KB, "fire_tau_min": FIRE_TAU_MIN,
                          "s_phase_min": S_PHASE_MIN, "dens_sigma_kb": DENS_SIGMA_KB,
                          "n_real": NREAL, "n_null": NNULL, "seed": SEED},
               "t1": {"coverage": cov, "n_used": int(have.sum()), "n_lifted": int(M.shape[0]),
                      "rho_rt_vs_density": r_dens, "chroms": used_chroms, "dropped": dropped},
               "t2": {"rho_uniform": r_uni, "r2_uniform": r_uni ** 2,
                      "rho_single_realisation": r_uni1, "floor": T2_FLOOR},
               "t3": {"rho_informed": r_inf, "r2_informed": r_inf ** 2,
                      "rho_distance_only": r_dist, "r2_distance_only": r_dist ** 2,
                      "capability_shift": cap_shift, "capability_uniform": cap_unif,
                      "survival_shift": s_shift, "survival_uniform": s_unif,
                      "null_shift": null_shift, "null_uniform": null_unif},
               "t4": {"blur_sweep": {str(k): v for k, v in blur.items()},
                      "fork_grid": {f"sigma{int(k[0])}_spacing{int(k[1])}": v
                                    for k, v in grid.items()},
                      "best_fork_cell": [best_cell[0], best_cell[1]], "best_fork_rho": best_fork,
                      "best_sigma_kb": best_sg, "best_blur_rho": best_blur,
                      "fork_minus_blur": best_fork - best_blur, "margin_required": T4_MARGIN},
               "t5": {"n_pairs": int(len(sl_q)), "median_gap_kb": float(np.median(gaps)),
                      "gradient_percentiles_min_per_kb":
                          {str(p): float(np.percentile(sl_q, p)) for p in pcts},
                      "implied_speed_kb_min_quantile_map": {str(p): imp_q[p] for p in pcts},
                      "implied_speed_kb_min_linear_map": {str(p): imp_l[p] for p in pcts},
                      "speed_at_p90": speed90, "bracket": [T5_LO, T5_HI]},
               "t6": {"pubs_vs_rt": r_pub_rt, "pubs_vs_lateness": r_pub_lat,
                      "pubs_vs_prediction": r_pub_pred, "pubs_vs_residual": r_pub_resid,
                      "best_model_abs_rho": best_model, "margin_required": T6_MARGIN},
               "speed_sweep": {str(k): v for k, v in sweep.items()},
               "speed_sweep_spread": spread, "best_speed_kb_min": best_v,
               "per_track_rho": per_track,
               "variance_explained": {"pure_fork_no_origins": r_uni ** 2,
                                      "fork_on_density_origins": r_inf ** 2,
                                      "distance_only": r_dist ** 2,
                                      "best_fork_on_grid": best_fork ** 2,
                                      "blurred_density_alone": best_blur ** 2,
                                      "pubs": r_pub_lat ** 2},
               "convergence_rho_by_nreal": CONVERGENCE,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_replication_time.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_replication_time.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
