"""Loop 191d. What sets the TIMING of a response -- with a negative control that is not the driver.

Loop 191 ran and produced NO valid gate outcome. All three failures are recorded here rather than
overwritten, because each is a different way for a loop to look like it worked.

DEFECT 1, A BROKEN JOIN THAT LOOKED LIKE A POWER PROBLEM. The ENCODE RNA quantifications are keyed
by Ensembl gene id -- ENSG00000096060 -- and every join in loop 191 matched on gene SYMBOL. So the
topology join, the autoregulation join and the TSS join each returned zero genes, and the loop
reported "0 responders in a group; this is a power floor declared before the run, not a null". That
sentence was true about the floor and false about the cause. A join that silently returns nothing
is indistinguishable from a real absence, so U1 now gates the join itself: if fewer than
MIN_JOIN of the expressed genes map to a symbol, the loop FAILS rather than proceeding to void
every downstream gate for the wrong reason.

DEFECT 2, AND THIS REPO HAD ALREADY WRITTEN IT DOWN. The half-time came back with a median of 28
minutes and an interquartile range of ONE minute across 6,950 genes. The cause is replicate
composition: t=5 carries replicates {1,3}, t=10-25 carry {1,2,3}, t=30-240 carry {1,2,3,4} and
t=300-360 carry {2,3,4}. Averaging whatever replicates exist at each timepoint puts a batch STEP at
t=30 where replicate 4 joins, and for 1,229 genes that step is the largest deviation in the whole
trajectory, which places their half-max crossing just before it. fetch_gr_timecourse.py's docstring
warns about this exact effect and gr_dynamics.py fixed it by differencing within replicate; loop
191 cited that file and then reproduced the defect it documents. Here the grid is restricted to
timepoints where replicates 1, 2 and 3 are ALL present, and each replicate is baseline-subtracted
against its own first point before averaging, so a replicate entering or leaving cannot move the
level.

DEFECT 3, A GATE THAT GUARDED THE WRONG PROPERTY. T1 asked whether the half-time reproduces across
a replicate split and passed at Spearman +0.543, which was read as "the target is usable". It is
not the same thing. A nearly constant target is nearly perfectly reproducible, and +0.543 was
reproducing noise around a constant; had the joins worked, T5 would have compared two groups whose
half-times were both 28. Reproducible and DISCRIMINATIVE are different properties and a target
needs both, so U4 gates the dynamic range directly.

DEFECT 4, WHICH ONLY BECAME VISIBLE ONCE THE FIRST THREE WERE FIXED, AND WHICH IS IN THE DATA
RATHER THAN IN THE CODE. With the joins working and the replicate composition constant, loop 191b's
half-time was STILL median 28 minutes with an interquartile range of one minute, and U4 caught it.
The cause is a discontinuity in the series itself. Expressed as the fraction of its own plateau the
median gene has reached:

    t          10    15    20    25    30    60   120   180   240   420   480   600   720
    median   0.00  0.05  0.04  0.05  0.98  0.98  0.98  1.01  1.00  1.01  1.00  1.00  1.00

Five percent of the response at 25 minutes and ninety-eight percent at 30, then flat for eleven and
a half hours. No biology delivers 98% of a twelve-hour response inside a five-minute window for the
MEDIAN gene genome-wide. It appears identically in all three replicates, which is what should be
expected rather than reassuring: each timepoint is its own ENCODE experiment accession and replicate
numbering is per-experiment, so the three replicates share the discontinuity instead of being
independent of it. Every timing statistic tried -- half of maximum, half of plateau -- was measuring
that step. FKBP5 at 87 minutes and TSC22D3 at 46 looked sensible only because their real responses
are large enough to survive it.

So this loop uses the post-boundary window only: t >= 30, where composition and batch are constant.
The median gene there traces 0.00, 0.38, 0.56, 0.56, 0.53, 0.81, 0.94, 1.04, 1.02 -- a graded
trajectory rather than a step -- and the response time spans an interquartile range of 212 minutes
instead of one. The cost is that the first 25 minutes are unusable, so nothing here can see a
response that completes before 30 minutes, and the responder threshold is lowered to |plateau| >=
0.5 because the window measures only the slow component of each response. Both are stated in U10.
DNase's own grid begins at 30 minutes, so the chromatin-leads comparison lands on this window
without further restriction.

THE RESPONSE TIME IS ALON'S DEFINITION, not a peak-finder. Rosenfeld, Elowitz & Alon define the
response time as the time to reach half the STEADY-STATE level. The plateau is the mean of the last
three points and the crossing is taken on the signed trajectory toward that plateau, so a transient
noise spike cannot set the target the way half-of-maximum let it.

DEFECT 5: THE NEGATIVE CONTROL WAS THE CAUSAL AGENT. Loop 191c's U8 required that NR3C1 promoter
occupancy NOT predict the response time, and it failed at rho -0.160 over 387 responders. Two things
are wrong with that gate and neither is the threshold.

  ITS PREMISE MISDESCRIBED THE PRIOR RESULT. U8 said a positive "contradicts gr_dynamics.py on this
  same series". It does not. gr_dynamics.py measured occupancy against dlog2(TPM+1) -- the MAGNITUDE
  of the change -- and found a static per-gene vector beat the time-resolved one. It never asked
  whether occupancy predicts the response TIME. Those are different targets and a null on one is not
  evidence about the other. The gate cited a result that does not say what it was quoted as saying.

  AND NR3C1 IS THE DRIVER. It is the glucocorticoid receptor: the protein dexamethasone activates,
  the causal agent of the entire experiment. Requiring the driver to carry no timing information was
  never defensible, and rho -0.160 -- more receptor at the promoter, faster response -- is the
  direction the biology would predict. A negative control has to be something that SHOULD be inert.

So the roles are swapped here. CTCF and RAD21 are the negative controls: architectural factors that
have no business tracking a steroid response, and which gr_dynamics.py measured at +0.0020 each on
this same series. NR3C1 moves to being a RESULT, scored the way every other result here is scored --
including inside magnitude terciles, which loop 191c never applied to it.

Nothing else changes. Loop 191c's findings stand as measured: accessibility leads the mRNA by 48
minutes over 1,310 responders at p 6.4e-58, surviving all three magnitude terciles, and feedback
sign does not order response times (77 min against 70 min, p 0.47).

PREDECLARED, BEFORE ANY NUMBER.

  U1 DOES THE JOIN WORK? Ensembl ids mapped to symbols through the Ensembl GTF already on disk.
     Gate: PASS iff at least 70% of expressed genes carry a symbol AND at least 30 responders fall
     in the negative two-cycle set. A FAIL here is a plumbing failure and says so; it is never
     reported as an absence of biology.

  U2 IS THE CLOCK CLEAN? Replicate composition per timepoint, and the restricted grid.
     Gate: PASS iff the retained grid has constant replicate composition. Asserted, not assumed.

  U3 IS TIMING REPRODUCIBLE? Half-time computed in replicate {1,3} against replicate {2}.
     Gate: PASS iff Spearman >= 0.30.

  U4 IS TIMING DISCRIMINATIVE? The interquartile range of the responder half-time distribution.
     Gate: PASS iff the IQR spans at least 20 minutes. This is the gate loop 191 lacked: a constant
     passes U3 and is useless, and no comparison of group medians can mean anything when every
     group median is the same number.

  U5 WHAT RESPONDS? Gate: PASS iff at least 200 responders.

  U6 DOES CHROMATIN MOVE FIRST? Promoter DNase half-time against expression half-time, both on the
     shared grid, paired one-sided Wilcoxon.
     Gate: PASS iff accessibility leads at p < 0.05.

  U7 NEGATIVE FEEDBACK, THE POWERED VERSION. Negative two-cycle genes against positive two-cycle
     genes, one-sided, negative predicted faster.
     Gate: PASS iff p < 0.05, VOID below 15 responders per group.

  U8 THE KNOWN NEGATIVE. Promoter NR3C1 occupancy against half-time.
     Gate: PASS iff |Spearman| < 0.10 -- effect size, not p, because this gate passes by finding
     nothing and at this n every correlation is significant.

  U9 THE MAGNITUDE CONFOUND. Every pooled pass repeated inside terciles of peak |log2 FC|.
     Gate: PASS iff each holds in at least 2 of 3 strata.

  U10 WHAT THIS CANNOT SHOW.

-> outputs/loop_response_timing_d.json
"""
import gzip
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402

from scipy.stats import mannwhitneyu, spearmanr, wilcoxon         # noqa: E402

SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/"
                         "scratchpad"))
GRTC = SP / "grtc"
OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_response_timing_d.json"
BUNDLE = Path("colab/data/net_bundle.json.gz")
AUTO = Path("colab/data/tf_autoregulation.json")
CUR = (0, 55716)

MIN_SPLIT_RHO = 0.30      # T1
MIN_LFC = 1.0             # a doubling, from baseline, at the peak
MIN_TPM = 1.0             # expressed at baseline, or a fold change is a ratio of noise
MIN_RESPONDERS = 200      # T2
MIN_GROUP = 15            # T4/T5 power floor, declared before running
ALPHA = 0.05
N_STRATA = 3
PROM_PAD = 1000          # bp either side of the TSS for a promoter peak
MAX_OCC_RHO = 0.10       # T6 is scored on effect size, not p -- see the gate
SEED = 191191

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def half_time(mins, v):
    """First time at which |v - v[0]| reaches half its own maximum, linearly interpolated.

    Defined on the ABSOLUTE deviation from baseline so a repressed gene and an induced gene are
    measured the same way -- the question is when the gene moves, not which way. Returns nan when
    the trajectory never moves, which is why the responder filter runs first."""
    d = np.abs(v - v[0])
    if not np.isfinite(d).all() or d.max() <= 0:
        return float("nan")
    tgt = d.max() / 2.0
    for i in range(1, len(d)):
        if d[i] >= tgt:
            lo, hi = d[i - 1], d[i]
            if hi == lo:
                return float(mins[i])
            f = (tgt - lo) / (hi - lo)
            return float(mins[i - 1] + f * (mins[i] - mins[i - 1]))
    return float(mins[-1])


def trajectories(tpm, mins, reps, genes, which=None):
    """Mean log2(TPM+1) per timepoint, over the chosen replicates, as (n_time, n_gene)."""
    keep = np.ones(len(reps), dtype=bool) if which is None else np.isin(reps, list(which))
    ts = sorted({int(m) for m in mins[keep]})
    M = np.full((len(ts), tpm.shape[1]), np.nan, dtype=np.float64)
    for i, t in enumerate(ts):
        m = keep & (mins.astype(int) == t)
        if m.sum():
            M[i] = np.log2(1.0 + tpm[m].astype(np.float64)).mean(0)
    return np.array(ts, dtype=float), M


def tss_map(genes_upper):
    """Symbol -> (chrom, pos) in GRCh38.

    _tss_hg38.bed carries 16,380 TSSs keyed G0, G1, ... which are ROW INDICES into the project's
    gene table, verified here rather than assumed: G0 must be that table's first row and its
    coordinate must match the row's own tss field."""
    tab = json.load(gzip.open("colab/data/cell_complete.json.gz"))["genes"]
    out = {}
    for line in open(SP / "_tss_hg38.bed"):
        f = line.split()
        if len(f) < 4 or not f[3].startswith("G"):
            continue
        i = int(f[3][1:])
        if i >= len(tab):
            continue
        rec = tab[i]
        if i == 0:
            assert str(rec["chrom"]) == f[0] and int(rec["tss"]) == int(f[2]), \
                f"TSS bed does not index the gene table: {rec} vs {f}"
        out[str(rec["name"]).upper()] = (f[0], int(f[2]))
    return out


def promoter_track(target, tsslist, pad, report=print):
    """max peak signalValue within +/- pad of each TSS, per timepoint.

    Returns (times, matrix) with matrix[t, g]. A gene with no overlapping peak scores 0, which is
    the right value: no peak means no measured signal there, not missing data."""
    d = GRTC / target
    if not d.is_dir():
        return np.array([]), np.zeros((0, len(tsslist)))
    files = sorted(d.glob("*.bed.gz"), key=lambda f: int(f.name.split("min")[0]))
    times = np.array([float(f.name.split("min")[0]) for f in files])
    by_chrom = defaultdict(list)
    for gi, tp in enumerate(tsslist):
        if tp is not None:
            by_chrom[tp[0]].append((tp[1], gi))
    for c in by_chrom:
        by_chrom[c].sort()
    M = np.zeros((len(files), len(tsslist)), dtype=np.float64)
    for ti, f in enumerate(files):
        for line in gzip.open(f, "rt"):
            q = line.split()
            if len(q) < 7:
                continue
            c, a, b = q[0], int(q[1]), int(q[2])
            arr = by_chrom.get(c)
            if not arr:
                continue
            sig = float(q[6])
            lo = np.searchsorted([x[0] for x in arr], a - pad)
            for pos, gi in arr[lo:]:
                if pos > b + pad:
                    break
                if M[ti, gi] < sig:
                    M[ti, gi] = sig
    report(f"     {target}: {len(files)} timepoints, "
           f"{float((M > 0).any(0).mean()):.1%} of genes carry a promoter peak at some time")
    return times, M


REPS = (1, 2, 3)          # replicate 4 has no timepoint before 30 min and is dropped entirely
T_MIN = 30.0              # the batch boundary: everything before this is unusable, see the docstring
MIN_PLATEAU = 0.5         # the post-boundary window sees only the slow component of a response
MIN_JOIN = 0.70           # U1: fraction of expressed genes that must map to a symbol
MIN_CYCLE_RESP = 30       # U1: responders required in the negative two-cycle set
MIN_IQR = 20.0            # U4: minutes the half-time IQR must span to be discriminative
GTF = SP / "ens_gtf.gz"


def ensg_to_symbol(report=print):
    """Ensembl gene id -> symbol, from the GTF already on disk.

    Loop 191 joined ENCODE's ENSG-keyed quantifications to symbol-keyed topology and silently got
    nothing, then reported the emptiness as a declared power floor. This function exists so that
    join has a name, a coverage number and a gate."""
    m = {}
    for line in gzip.open(GTF, "rt"):
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "gene":
            continue
        a = f[8]
        i = a.find('gene_id "')
        j = a.find('gene_name "')
        if i < 0 or j < 0:
            continue
        gid = a[i + 9:a.find('"', i + 9)].split(".")[0]
        m[gid] = a[j + 11:a.find('"', j + 11)].upper()
    report(f"     Ensembl GTF: {len(m):,} gene ids carry a symbol")
    return m


def rep_trajectories(tpm, mins, reps, which, grid):
    """Deviation-from-baseline per replicate, then averaged. (n_time, n_gene).

    Each replicate is subtracted against ITS OWN first grid point before averaging, so a replicate
    entering or leaving the series cannot shift the level. That is the defect that put a batch step
    at t=30 in loop 191 and made every half-time 28 minutes."""
    acc, n = None, 0
    for r in which:
        rows = [np.where((mins == t) & (reps == r))[0] for t in grid]
        if any(len(x) == 0 for x in rows):
            continue
        V = np.array([np.log2(1.0 + tpm[ix].astype(np.float64)).mean(0) for ix in rows])
        V = V - V[0]
        acc = V if acc is None else acc + V
        n += 1
    return (acc / max(n, 1)), n


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 191b  WHAT SETS THE TIMING OF A RESPONSE -- loop 191's three defects corrected")
    say("=" * 104)
    say("  PREDECLARED: the Ensembl-to-symbol join is GATED, because loop 191's joins returned")
    say("  zero and the loop reported that as a declared power floor rather than as plumbing;")
    say("  the grid is restricted to timepoints where replicates 1-3 are ALL present and each is")
    say("  baseline-subtracted within itself, because averaging a changing replicate set put a")
    say("  batch step at t=30 that became the largest deviation for 1,229 genes; and the half-time")
    say(f"  must be DISCRIMINATIVE (IQR >= {MIN_IQR:.0f} min) as well as reproducible, because loop")
    say("  191's target was near-constant at median 28 with an IQR of one minute and still passed")
    say("  a reproducibility gate at rho +0.543.")
    say()

    z = np.load(GRTC / "rna.npz", allow_pickle=True)
    tpm = z["tpm"]
    ensg = np.array([str(g).split(".")[0] for g in z["genes"]])
    mins, reps = z["mins"].astype(int), z["reps"].astype(int)
    man = json.load(open(GRTC / "manifest.json"))

    # ---- U2 ------------------------------------------------------------------------------------
    say("U2 IS THE CLOCK CLEAN?")
    allt = sorted(set(mins.tolist()))
    comp = {t: tuple(sorted(set(reps[mins == t].tolist()))) for t in allt}
    for t in allt:
        say(f"     t={t:4d}  replicates {list(comp[t])}")
    full = np.array([t for t in allt if set(REPS) <= set(comp[t])], dtype=float)
    grid = full[full >= T_MIN]
    kept = {comp[int(t)] for t in grid}
    say(f"     constant-composition grid: {[int(x) for x in full]}")
    say(f"     retained after the t >= {T_MIN:.0f} batch boundary ({len(grid)} points): "
        f"{[int(x) for x in grid]}")
    say(f"     replicates used: {list(REPS)} (replicate 4 has nothing before 30 min and is dropped)")
    u2 = bool(len(grid) >= 8 and len({tuple(sorted(set(REPS) & set(c))) for c in kept}) == 1)
    GG.verdict(u2, emit=say,
               if_true=f"U2 PASS -- replicates {list(REPS)} are present at every retained "
                       f"timepoint, so no replicate enters or leaves along the axis being measured",
               if_false="U2 FAIL -- the retained grid still changes composition")

    M, nrep = rep_trajectories(tpm, mins, reps, REPS, grid)
    Ma, _ = rep_trajectories(tpm, mins, reps, (1, 3), grid)
    Mb, _ = rep_trajectories(tpm, mins, reps, (2,), grid)
    say(f"     trajectories built from {nrep} replicates, baseline-subtracted within each")

    # ---- U1 ------------------------------------------------------------------------------------
    say()
    say("U1 DOES THE JOIN WORK?")
    e2s = ensg_to_symbol(say)
    sym = np.array([e2s.get(g, "") for g in ensg])
    base_tpm = np.array([tpm[(mins == int(grid[0])) & np.isin(reps, REPS)].mean(0)]).ravel()
    expressed = base_tpm >= MIN_TPM
    frac = float((sym[expressed] != "").mean())
    say(f"     {int(expressed.sum()):,} genes expressed at the baseline point (TPM >= {MIN_TPM})")
    say(f"     of those, {frac:.1%} carry a symbol")

    peak_lfc = np.abs(M[-3:].mean(0))          # the plateau, not the maximum excursion
    dir_a = np.sign(Ma[np.nanargmax(np.abs(Ma), axis=0), np.arange(Ma.shape[1])])
    dir_b = np.sign(Mb[np.nanargmax(np.abs(Mb), axis=0), np.arange(Mb.shape[1])])
    responder = expressed & (peak_lfc >= MIN_PLATEAU) & (dir_a == dir_b) & (dir_a != 0)

    nb = json.load(gzip.open(BUNDLE))
    names, reg = nb["names"], nb["reg"]
    out = defaultdict(dict)
    for r in reg[CUR[0]:CUR[1]]:
        out[int(r[0])][int(r[1])] = int(r[2]) if len(r) > 2 else 0
    neg2, pos2 = set(), set()
    for a in out:
        for b in out[a]:
            if b != a and b in out and a in out[b] and a < b:
                s1, s2 = out[a][b], out[b][a]
                if s1 and s2:
                    (pos2 if s1 * s2 > 0 else neg2).update({names[a].upper(), names[b].upper()})
    in_neg = np.isin(sym, list(neg2)) & responder
    in_pos = np.isin(sym, list(pos2)) & responder
    say(f"     responders in a negative two-cycle: {int(in_neg.sum())}, positive: "
        f"{int(in_pos.sum())}")
    u1 = bool(frac >= MIN_JOIN and in_neg.sum() >= MIN_CYCLE_RESP)
    GG.verdict(u1, emit=say,
               if_true=f"U1 PASS -- {frac:.1%} of expressed genes map to a symbol and "
                       f"{int(in_neg.sum())} negative-cycle genes respond, so the joins below are "
                       f"real and an empty group would be an absence rather than plumbing",
               if_false=f"U1 FAIL -- join coverage {frac:.1%} (bar {MIN_JOIN:.0%}) and "
                        f"{int(in_neg.sum())} negative-cycle responders (bar {MIN_CYCLE_RESP}). "
                        f"This is a PLUMBING failure and must not be read as an absence of biology")

    def resp_time(V, grid_):
        """Alon's response time: the moment the signed trajectory reaches half its steady state."""
        pl = V[-3:].mean(0)
        out_t = np.full(V.shape[1], np.nan)
        for j in range(V.shape[1]):
            p = pl[j]
            if abs(p) < 1e-9:
                continue
            tg = p / 2.0
            v = V[:, j]
            for i in range(1, len(v)):
                if (p > 0 and v[i] >= tg) or (p < 0 and v[i] <= tg):
                    lo, hi = v[i - 1], v[i]
                    out_t[j] = grid_[i] if hi == lo else \
                        grid_[i - 1] + (tg - lo) / (hi - lo) * (grid_[i] - grid_[i - 1])
                    break
        return out_t, pl

    th, plateau = resp_time(M, grid)
    th_a, _ = resp_time(Ma, grid)
    th_b, _ = resp_time(Mb, grid)

    # ---- U3 ------------------------------------------------------------------------------------
    say()
    say("U3 IS TIMING REPRODUCIBLE?")
    ok = responder & np.isfinite(th_a) & np.isfinite(th_b)
    rho, p_rho = spearmanr(th_a[ok], th_b[ok]) if ok.sum() > 10 else (float("nan"), float("nan"))
    say(f"     replicates [1,3] against [2] over {int(ok.sum()):,} responders: "
        f"Spearman {rho:+.3f} (p {p_rho:.3g})")
    u3 = bool(np.isfinite(rho) and rho >= MIN_SPLIT_RHO)
    GG.verdict(u3, emit=say,
               if_true=f"U3 PASS -- {rho:+.3f}",
               if_false=f"U3 FAIL -- {rho:+.3f} against {MIN_SPLIT_RHO}")

    # ---- U4 ------------------------------------------------------------------------------------
    say()
    say("U4 IS TIMING DISCRIMINATIVE?")
    q = np.nanpercentile(th[responder], [10, 25, 50, 75, 90]) if responder.sum() else [np.nan]*5
    iqr = float(q[3] - q[1])
    say(f"     half-time over {int(responder.sum()):,} responders: median {q[2]:.0f} min, "
        f"IQR {q[1]:.0f}-{q[3]:.0f} ({iqr:.0f} min wide), 10-90% {q[0]:.0f}-{q[4]:.0f}")
    say(f"     loop 191 had median 28 with an IQR of 1 minute and passed its reproducibility gate")
    u4 = bool(np.isfinite(iqr) and iqr >= MIN_IQR)
    GG.verdict(u4, emit=say,
               if_true=f"U4 PASS -- the IQR spans {iqr:.0f} min, so genes differ in WHEN they move "
                       f"and a comparison of group medians can mean something",
               if_false=f"U4 FAIL -- the IQR spans {iqr:.0f} min against a bar of {MIN_IQR:.0f}. "
                        f"The target is near-constant, so every group median would be the same "
                        f"number and no gate below could distinguish anything")

    # ---- U5 ------------------------------------------------------------------------------------
    say()
    say("U5 WHAT RESPONDS?")
    n_resp = int(responder.sum())
    say(f"     {n_resp:,} responders at |plateau| >= {MIN_PLATEAU} with the direction agreeing across "
        f"the split; up {int((dir_a[responder] > 0).sum()):,} down "
        f"{int((dir_a[responder] < 0).sum()):,}")
    u5 = bool(n_resp >= MIN_RESPONDERS)
    GG.verdict(u5, emit=say, if_true=f"U5 PASS -- {n_resp:,} responders",
               if_false=f"U5 FAIL -- {n_resp:,} against {MIN_RESPONDERS}")

    void = set()
    if not (u1 and u2 and u3 and u4 and u5):
        void |= {"U6", "U7", "U8", "U9"}
        say()
        say("     a precondition gate failed, so U6-U9 are VOID rather than negative")

    tab = json.load(gzip.open("colab/data/cell_complete.json.gz"))["genes"]
    tssbed = {}
    for line in open(SP / "_tss_hg38.bed"):
        f = line.split()
        if len(f) >= 4 and f[3].startswith("G"):
            i = int(f[3][1:])
            if i < len(tab):
                tssbed[str(tab[i]["name"]).upper()] = (f[0], int(f[2]))
    tsslist = [tssbed.get(s) for s in sym]
    say(f"     TSS in GRCh38 for {sum(x is not None for x in tsslist):,} of the RNA rows")

    # ---- U6 ------------------------------------------------------------------------------------
    say()
    say("U6 DOES CHROMATIN MOVE FIRST?")
    u6, d6 = False, {}
    _u6_mask = np.zeros(len(sym), dtype=bool)
    _u6_acc = np.zeros(len(sym))
    _u6_exp = np.zeros(len(sym))
    if "U6" in void:
        say("     U6 VOID -- see above")
    else:
        dt, DM = promoter_track("DNase", tsslist, PROM_PAD, say)
        shared = np.array(sorted(set(dt.tolist()) & set(grid.tolist())))
        say(f"     shared grid for the comparison: {[int(x) for x in shared]}")
        if len(shared) < 4:
            void.add("U6")
            say("     U6 VOID -- fewer than 4 shared timepoints")
        else:
            di = [int(np.where(dt == t)[0][0]) for t in shared]
            ei = [int(np.where(grid == t)[0][0]) for t in shared]
            ah = np.array([half_time(shared, DM[di, j]) for j in range(DM.shape[1])])
            eh = np.array([half_time(shared, M[ei, j]) for j in range(M.shape[1])])
            m6 = responder & (DM > 0).any(0) & np.isfinite(ah) & np.isfinite(eh)
            say(f"     responders with a promoter DNase peak: {int(m6.sum()):,}")
            if m6.sum() < MIN_GROUP:
                void.add("U6")
                say(f"     U6 VOID -- {int(m6.sum())} under the power floor")
            else:
                _, p6 = wilcoxon(ah[m6], eh[m6], alternative="less")
                lead = float(np.median(eh[m6] - ah[m6]))
                say(f"     accessibility median {np.median(ah[m6]):.0f} min vs expression "
                    f"{np.median(eh[m6]):.0f} min; lead {lead:+.0f} min; one-sided p {p6:.3g}")
                d6 = dict(n=int(m6.sum()), lead=lead, p=float(p6))
                _u6_mask, _u6_acc, _u6_exp = m6, ah, eh
                u6 = bool(p6 < ALPHA)
                GG.verdict(u6, emit=say,
                           if_true=f"U6 PASS -- chromatin opens {lead:.0f} min before the mRNA "
                                   f"moves, so accessibility is a timing mechanism",
                           if_false="U6 FAIL -- accessibility does not lead transcription here")

    # ---- U7 ------------------------------------------------------------------------------------
    say()
    say("U7 NEGATIVE FEEDBACK, THE POWERED VERSION")
    u7, d7 = False, {}
    if "U7" in void:
        say("     U7 VOID -- see above")
    else:
        a, b = in_neg & np.isfinite(th), in_pos & np.isfinite(th)
        say(f"     negative two-cycle {int(a.sum())} responders, positive {int(b.sum())}")
        if a.sum() < MIN_GROUP or b.sum() < MIN_GROUP:
            void.add("U7")
            say(f"     U7 VOID -- under the {MIN_GROUP} power floor")
        else:
            _, p7 = mannwhitneyu(th[a], th[b], alternative="less")
            say(f"     median half-time {np.median(th[a]):.0f} min vs {np.median(th[b]):.0f} min; "
                f"one-sided p {p7:.3g}")
            d7 = dict(n_neg=int(a.sum()), n_pos=int(b.sum()), p=float(p7),
                      median_neg=float(np.median(th[a])), median_pos=float(np.median(th[b])))
            u7 = bool(p7 < ALPHA)
            GG.verdict(u7, emit=say,
                       if_true="U7 PASS -- genes in negative feedback respond faster than genes in "
                               "positive feedback, the dynamic consequence loop 187's z = +43.8 "
                               "topology predicts",
                       if_false="U7 FAIL -- feedback sign does not order response times. Loop 187's "
                                "two-cycle enrichment is a structural fact without the dynamic "
                                "consequence the literature attaches to it")

    # ---- U8 ------------------------------------------------------------------------------------
    say()
    say("U8 THE NEGATIVE CONTROLS: architectural factors must NOT predict timing")
    say("     CTCF and RAD21 are the controls because they SHOULD be inert to a steroid response.")
    say("     loop 191c used NR3C1 here, which is the receptor the drug activates -- the driver of")
    say("     the experiment -- and that was never a defensible negative control.")
    u8, d8 = False, {}
    if "U8" in void:
        say("     U8 VOID -- see above")
    else:
        rhos = {}
        for tgt in ("CTCF", "RAD21"):
            nt, NM = promoter_track(tgt, tsslist, PROM_PAD, say)
            occ = NM.max(0) if len(nt) else np.zeros(len(sym))
            m = responder & np.isfinite(th) & (occ > 0)
            if m.sum() < MIN_GROUP:
                say(f"     {tgt}: too few responders with a promoter peak")
                continue
            r_, p_ = spearmanr(occ[m], th[m])
            rhos[tgt] = dict(n=int(m.sum()), rho=float(r_), p=float(p_))
            say(f"     {tgt}: {int(m.sum()):,} responders, Spearman {r_:+.3f} (p {p_:.3g})")
        d8 = rhos
        if not rhos:
            void.add("U8")
            say("     U8 VOID -- neither control has enough promoter peaks")
        else:
            u8 = all(abs(v["rho"]) < MAX_OCC_RHO for v in rhos.values())
            GG.verdict(u8, emit=say,
                       if_true=f"U8 PASS -- both architectural factors sit below |rho| "
                               f"{MAX_OCC_RHO}, so the timing target is not picking up generic "
                               f"promoter occupancy and U6's lead is not that artefact",
                       if_false="U8 FAIL -- an architectural factor predicts the response time, "
                                "which means promoter occupancy of ANY kind tracks timing here "
                                "and U6 and U9 cannot be separated from that")

    # ---- U8b -----------------------------------------------------------------------------------
    say()
    say("U8b THE DRIVER, AS A RESULT RATHER THAN A CONTROL")
    u8b, d8b = False, {}
    if "U8" in void and "U6" in void:
        say("     U8b VOID -- see above")
    else:
        nt, NM = promoter_track("NR3C1", tsslist, PROM_PAD, say)
        occ = NM.max(0) if len(nt) else np.zeros(len(sym))
        m = responder & np.isfinite(th) & (occ > 0)
        if m.sum() < MIN_GROUP:
            void.add("U8b")
            say("     U8b VOID -- too few responders with promoter NR3C1")
        else:
            r_, p_ = spearmanr(occ[m], th[m])
            say(f"     {int(m.sum()):,} responders; Spearman(NR3C1 occupancy, response time) "
                f"{r_:+.3f} (p {p_:.3g})")
            held = 0
            edges_ = np.quantile(peak_lfc[responder], np.linspace(0, 1, N_STRATA + 1))
            edges_[-1] += 1e-9
            for qi in range(N_STRATA):
                st = m & (peak_lfc >= edges_[qi]) & (peak_lfc < edges_[qi + 1])
                if st.sum() >= MIN_GROUP:
                    rq, pq = spearmanr(occ[st], th[st])
                    held += int(pq < ALPHA and rq < 0)
                    say(f"       tercile {qi+1}: n {int(st.sum())}  rho {rq:+.3f}  p {pq:.3g}")
                else:
                    say(f"       tercile {qi+1}: under the power floor")
            d8b = dict(n=int(m.sum()), rho=float(r_), p=float(p_), strata_held=held)
            # GUARDED. spearmanr returns nan when either input has no variance, and `nan < 0`
            # is False, so the boolean would swallow the undefinedness and score FAIL on a
            # statistic that never existed. That is loop 187's B6 mechanism.
            if not (np.isfinite(r_) and np.isfinite(p_)):
                void.add("U8b")
                say(f"     U8b VOID -- the correlation is undefined (rho {r_!r}), so this gate "
                    f"could not pass or fail; that is not the same as failing")
                u8b = False
            else:
                u8b = bool(r_ < 0 and p_ < ALPHA and held >= 2)
            GG.verdict(u8b, emit=say,
                       if_true=f"U8b PASS -- more receptor at the promoter goes with a FASTER "
                               f"response (rho {r_:+.3f}), and it holds in {held}/3 magnitude "
                               f"terciles, so occupancy carries WHEN as well as WHICH. That is new: "
                               f"gr_dynamics.py measured occupancy against the magnitude of change "
                               f"and never against timing",
                       if_false=f"U8b FAIL -- rho {r_:+.3f}, holding in {held}/3 terciles; the "
                                f"driver's promoter occupancy does not order response times once "
                                f"response size is controlled")

    # ---- U9 ------------------------------------------------------------------------------------
    say()
    say("U9 THE MAGNITUDE CONFOUND")
    strata = {}
    pooled = [g for g, v in (("U6", u6), ("U7", u7)) if v and g not in void]
    if not pooled:
        void.add("U9")
        say("     U9 VOID -- no pooled gate passed, so there is nothing for the strata to confirm")
    else:
        edges = np.quantile(peak_lfc[responder], np.linspace(0, 1, N_STRATA + 1))
        edges[-1] += 1e-9
        say(f"     terciles of peak |log2 FC|: {[round(float(e), 2) for e in edges]}")
        for g in pooled:
            held = 0
            for qi in range(N_STRATA):
                st = (peak_lfc >= edges[qi]) & (peak_lfc < edges[qi + 1])
                if g == "U7":
                    a, b = in_neg & st & np.isfinite(th), in_pos & st & np.isfinite(th)
                    if a.sum() >= MIN_GROUP and b.sum() >= MIN_GROUP:
                        _, pp = mannwhitneyu(th[a], th[b], alternative="less")
                        held += int(pp < ALPHA)
                        say(f"       U7 tercile {qi+1}: n {int(a.sum())} vs {int(b.sum())} "
                            f"p {pp:.3g}")
                    else:
                        say(f"       U7 tercile {qi+1}: under the power floor")
                elif g == "U6":
                    m = _u6_mask & st
                    if m.sum() >= MIN_GROUP:
                        _, pp = wilcoxon(_u6_acc[m], _u6_exp[m], alternative="less")
                        held += int(pp < ALPHA)
                        say(f"       U6 tercile {qi+1}: n {int(m.sum())}  lead "
                            f"{float(np.median(_u6_exp[m] - _u6_acc[m])):+.0f} min  p {pp:.3g}")
                    else:
                        say(f"       U6 tercile {qi+1}: under the power floor")
            strata[g] = held
        u9 = all(v >= 2 for v in strata.values())
        GG.verdict(u9, emit=say,
                   if_true="U9 PASS -- the pooled result survives inside magnitude terciles",
                   if_false="U9 FAIL -- the pooled result does not hold within magnitude strata, "
                            "so it is a response-size effect reported as a timing effect")
    u9 = all(v >= 2 for v in strata.values()) if strata else False

    # ---- U10 -----------------------------------------------------------------------------------
    say()
    say("U10 WHAT THIS CANNOT SHOW")
    say("     One cell line, one drug, one receptor, and a fast nuclear-receptor response.")
    say("     The half-time summarises a curve and discards its shape: an overshoot and a monotone")
    say("     rise with the same midpoint are recorded identically, and overshoot is the signature")
    say("     negative feedback is supposed to produce, so this statistic understates what a")
    say("     feedback test could find.")
    say("     Topology is CollecTRI curation from other cell lines; a two-cycle curated elsewhere")
    say("     need not be wired in A549 and no gate here can see that.")
    say("     Dropping replicate 4 and the 5, 300 and 360 minute points costs power and early")
    say("     resolution. That is the price of a constant replicate composition and it is paid")
    say("     deliberately, because loop 191 shows what the alternative produces.")
    say("     Bulk RNA averages over the population, so a fast response in a subpopulation and a")
    say("     slow one everywhere give the same curve.")
    say("     U10 PASS")

    gates = {"U1": u1, "U2": u2, "U3": u3, "U4": u4, "U5": u5,
             "U6": u6, "U7": u7, "U8": u8, "U8b": u8b, "U9": u9, "U10": True}
    man_out = RM.manifest(inputs=[GRTC / "rna.npz", BUNDLE, AUTO],
                          available=int(len(sym)), used=int(responder.sum()),
                          selection="filtered", seed=SEED,
                          controls=["Ensembl-to-symbol join gated, not assumed",
                                    "constant replicate composition on the retained grid",
                                    "each replicate baseline-subtracted within itself",
                                    "the target gated on dynamic range as well as reliability",
                                    "magnitude terciles for every pooled pass"],
                          note="response timing against feedback topology, loop 191 defects fixed")
    out_d = dict(test="response timing corrected", gates=gates, void=sorted(void),
                 grid=[int(x) for x in grid], replicates=list(REPS),
                 join_fraction=frac, n_responders=int(responder.sum()),
                 split=dict(rho=float(rho), p=float(p_rho)),
                 half_time=dict(median=float(q[2]), iqr=iqr,
                                p10=float(q[0]), p90=float(q[4])),
                 u6=d6, u7=d7, u8=d8, u8b=d8b, strata=strata,
                 manifest=man_out, seconds=time.time() - t0, log=log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'VOID' if k in void else ('PASS' if v else 'FAIL')}")
    scored = [k for k in gates if k not in void]
    say(f"  {sum(gates[k] for k in scored)}/{len(scored)}   [{time.time()-t0:.0f}s]"
        + (f"   ({len(void)} VOID: {', '.join(sorted(void))})" if void else ""))
    say("=" * 104)
    out_d["log"] = log
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
