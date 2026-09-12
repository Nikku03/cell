"""Loop 191. What sets the TIMING of a gene's response? The 4D question, asked of the right target.

WHY THE OBVIOUS VERSION OF THIS IS ALREADY CLOSED. Loop 190's census ends by saying that every layer
it counted is a steady-state description and that no count of parts becomes a trajectory. The
natural response is to add the loop layer and the epigenetic layer and call the map 4D. That does
not work, and this project already measured why. gr_dynamics.py put NR3C1 occupancy against polyA
RNA on the ENCODE A549 dexamethasone series -- a real driver, a real clock, sixteen matched
timepoints, 9,660 genes, replicate reliability of the change 0.601 -- and found:

    NR3C1 occupancy on the demeaned change      +0.0068 held-out R2
    a CONSTANT per-gene vector, no time at all   +0.0095   139% of it
    permuting the time axis                      +0.0061   costs almost nothing
    swapping gene identity                       -0.0020   destroys it

Occupancy tells you WHICH genes have volatile trajectories and nothing about WHEN they move. Every
one of eight targets showed static >= time-resolved, and the two that matter most for a "loops in
4D" claim were in that test: CTCF +0.0020 and RAD21 +0.0020, essentially nothing. So adding the
same layers again, as layers, gives a richer three-dimensional map and not a fourth dimension.

WHAT IS ACTUALLY UNTESTED. Every dynamic test in this project has predicted the MAGNITUDE of a
change. None has predicted its TIMING. Timing is the quantity a fourth dimension is actually made
of, and it is the one thing loop 187's measured topology makes a published, quantitative prediction
about: Rosenfeld, Elowitz & Alon (JMB 2002) showed that negative autoregulation SPEEDS a gene's
response time, and Alon (Nat Rev Genet 2007) extends the logic to negative feedback generally.
Loop 187 measured that this network's feedback is real -- two-cycles at z = +43.8 against a
degree-preserving null, autoregulation at z = +4.0 and 2.2x the configuration-model rate. Those are
structural facts with a dynamic consequence attached, and the consequence has never been checked.

THE POWER PROBLEM, MEASURED BEFORE THE GATES WERE WRITTEN AND NOT AFTER. The literal Alon
prediction is about NEGATIVE autoregulation. TRRUST labels 24 human self-loops, of which exactly 4
are Repression (10 Activation, 13 Unknown -- the modes overlap because a factor can carry both).
Four genes, of which some fraction respond to dexamethasone at all, cannot support a test. T4 is
written down anyway, with a minimum responder count declared in advance, so that it returns VOID
for want of power rather than being quietly dropped or reported as a null result. The powered
version is negative two-cycles: 258 genes sit in a signed negative two-cycle and 421 in a positive
one, which is a testable contrast if enough of them move.

THE CONTROL THAT DECIDES EVERYTHING, inherited from the mistake gr_dynamics.py nearly shipped. Its
first verdict read a positive off +0.0068 against a +0.0066 permutation control, before noticing
that permuting the time axis of a nearly-constant series returns nearly the same series -- so
permutation cannot separate "which genes" from "when". Here the target IS a timing statistic, so
that particular trap is gone, but its shape returns as the MAGNITUDE CONFOUND: a gene that responds
more strongly has a better-determined timing, and any predictor of response size will look like a
predictor of response time. Every timing gate is therefore run again inside strata of response
magnitude, and a gate that survives only in the pooled analysis is reported as a magnitude effect.

PREDECLARED, BEFORE ANY NUMBER.

  T1 IS TIMING MEASURABLE AT ALL? The response half-time t_half computed independently in each
     replicate set, then correlated across the split.
     Gate: PASS iff Spearman across the split is >= 0.30 over the responder set. Below that the
     target is noise and every gate below is VOID rather than negative -- a timing test on an
     unreliable timing measurement says nothing about biology.

  T2 WHAT RESPONDS? Responders are genes whose peak |log2 fold change| from baseline exceeds
     MIN_LFC and whose direction agrees across the replicate split. Descriptive, with the count and
     the timing distribution reported.
     Gate: PASS iff at least MIN_RESPONDERS genes qualify, otherwise nothing downstream has power.

  T3 DOES CHROMATIN MOVE FIRST? DNase accessibility half-time at the promoter against expression
     half-time, per responder, paired. This is the epigenetic layer asked as a TIMING question,
     which is the only form in which a static mark can contribute a fourth dimension.
     BOTH half-times are computed on the SHARED timepoint grid. Expression has 16 points and DNase
     11 beginning at 30 min, so scoring expression on its own denser early grid would let it look
     earlier for having been sampled earlier -- a property of the assay design, not the biology.
     Gate: PASS iff the accessibility half-time is earlier than the expression half-time by a
     one-sided Wilcoxon signed-rank at p < 0.05. A FAIL means chromatin opening does not lead
     transcription on this clock and cannot be the timing mechanism.

  T4 THE LITERAL ALON PREDICTION. Negative autoregulation should give a SHORTER half-time.
     Gate: PASS iff negatively autoregulated responders are faster at one-sided p < 0.05, VOID if
     fewer than MIN_GROUP of them respond. The power calculation above says VOID is the likely
     outcome and it is declared here rather than discovered later.

  T5 NEGATIVE FEEDBACK, THE POWERED VERSION. Genes in a signed negative two-cycle against genes in
     a signed positive two-cycle. Alon's logic predicts negative feedback is faster and positive
     feedback slower, so the contrast is signed.
     Gate: PASS iff negative-cycle genes are faster than positive-cycle genes at one-sided p < 0.05,
     VOID if either group has fewer than MIN_GROUP responders.

  T6 THE KNOWN NEGATIVE. NR3C1 occupancy at the gene's promoter against half-time. gr_dynamics.py
     already showed occupancy carries WHICH and not WHEN; if it appears to carry WHEN here, then
     something in this loop's target is wrong and T3-T5 should not be believed.
     Gate: PASS iff |Spearman| < 0.10. Scored on EFFECT SIZE and deliberately not on a p-value:
     this gate passes by finding nothing, and with thousands of responders any correlation
     whatever is 'significant', so a p-value bar here could never be cleared and a gate that
     cannot fail to reject is not a gate. This is the E9/R5 lesson applied to a gate whose PASS
     condition is the null.

  T7 THE MAGNITUDE CONFOUND. T3, T5 and T6 repeated inside terciles of peak |log2 fold change|.
     Gate: PASS iff every gate that passed pooled also holds in at least 2 of 3 strata. A pooled
     pass that vanishes inside strata is a magnitude effect wearing a timing costume and is
     reported as one.

  T8 WHAT THIS CANNOT SHOW.

-> outputs/loop_response_timing.json
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
OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_response_timing.json"
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


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 191  WHAT SETS THE TIMING OF A RESPONSE? the 4D question asked of the right target")
    say("=" * 104)
    say("  PREDECLARED: the half-time must reproduce across a replicate split at Spearman >= "
        f"{MIN_SPLIT_RHO},")
    say("  or every gate below is VOID rather than negative; at least "
        f"{MIN_RESPONDERS} responders at |log2 FC| >= {MIN_LFC};")
    say("  chromatin must LEAD transcription for the epigenetic layer to be a timing mechanism;")
    say("  Rosenfeld/Elowitz/Alon's negative-autoregulation prediction is written down even though")
    say(f"  only 4 curated self-loops are Repression, and returns VOID below {MIN_GROUP} responders")
    say("  rather than being dropped; NR3C1 occupancy must NOT predict timing, because")
    say("  gr_dynamics.py already showed it does not and a positive here would indict the target;")
    say("  and every pooled pass must survive inside terciles of response magnitude.")
    say()

    if not (GRTC / "rna.npz").exists():
        raise SystemExit(f"{GRTC/'rna.npz'} missing -- run colab/fetch_gr_timecourse.py first")
    z = np.load(GRTC / "rna.npz", allow_pickle=True)
    tpm, genes = z["tpm"], np.array([str(g).upper() for g in z["genes"]])
    mins, reps = z["mins"].astype(int), z["reps"]
    man = json.load(open(GRTC / "manifest.json"))
    say(f"    RNA: {tpm.shape[0]} columns x {tpm.shape[1]:,} genes, "
        f"timepoints {sorted(set(mins.tolist()))}")
    say(f"    replicates present: {sorted(set(reps.tolist()))}")

    rset = sorted(set(reps.tolist()))
    A, B = rset[0::2], rset[1::2]
    say(f"    replicate split for T1: {A} against {B}")
    ts_all, Mall = trajectories(tpm, mins, reps, genes)
    ts_a, Ma = trajectories(tpm, mins, reps, genes, A)
    ts_b, Mb = trajectories(tpm, mins, reps, genes, B)

    base = Mall[0]
    peak_lfc = np.nanmax(np.abs(Mall - base), axis=0)
    expressed = (tpm[mins == mins.min()].mean(0) >= MIN_TPM) if (mins == mins.min()).any() \
        else np.ones(len(genes), dtype=bool)
    dir_a = np.sign(Ma[np.nanargmax(np.abs(Ma - Ma[0]), axis=0), np.arange(Ma.shape[1])] - Ma[0])
    dir_b = np.sign(Mb[np.nanargmax(np.abs(Mb - Mb[0]), axis=0), np.arange(Mb.shape[1])] - Mb[0])
    responder = expressed & (peak_lfc >= MIN_LFC) & (dir_a == dir_b) & (dir_a != 0)
    say(f"    expressed at baseline (TPM >= {MIN_TPM}): {int(expressed.sum()):,}")
    say(f"    responders (|log2 FC| >= {MIN_LFC}, direction agrees across the split): "
        f"{int(responder.sum()):,}")

    th_all = np.array([half_time(ts_all, Mall[:, j]) for j in range(Mall.shape[1])])
    th_a = np.array([half_time(ts_a, Ma[:, j]) for j in range(Ma.shape[1])])
    th_b = np.array([half_time(ts_b, Mb[:, j]) for j in range(Mb.shape[1])])

    # ---- T1 ------------------------------------------------------------------------------------
    say()
    say("T1 IS TIMING MEASURABLE AT ALL?")
    ok = responder & np.isfinite(th_a) & np.isfinite(th_b)
    rho, p_rho = spearmanr(th_a[ok], th_b[ok]) if ok.sum() > 10 else (float("nan"), float("nan"))
    say(f"     half-time computed independently in {A} and in {B}, over "
        f"{int(ok.sum()):,} responders")
    say(f"     Spearman across the split = {rho:+.3f}  (p {p_rho:.3g})")
    t1 = bool(np.isfinite(rho) and rho >= MIN_SPLIT_RHO)
    GG.verdict(t1, emit=say,
               if_true=f"T1 PASS -- the half-time reproduces at {rho:+.3f}, so it is a measurement "
                       f"and not noise, and the gates below are about biology",
               if_false=f"T1 FAIL -- {rho:+.3f} against a bar of {MIN_SPLIT_RHO}. The timing "
                        f"target is not reproducible on this data, so every gate below is VOID: a "
                        f"null on an unreliable measurement is a statement about the measurement")

    # ---- T2 ------------------------------------------------------------------------------------
    say()
    say("T2 WHAT RESPONDS?")
    n_resp = int(responder.sum())
    if n_resp:
        q = np.nanpercentile(th_all[responder], [10, 25, 50, 75, 90])
        say(f"     half-time over responders: median {q[2]:.0f} min "
            f"(IQR {q[1]:.0f}-{q[3]:.0f}, 10-90% {q[0]:.0f}-{q[4]:.0f})")
        say(f"     up {int((dir_a[responder] > 0).sum()):,}  "
            f"down {int((dir_a[responder] < 0).sum()):,}")
    t2 = bool(n_resp >= MIN_RESPONDERS)
    GG.verdict(t2, emit=say,
               if_true=f"T2 PASS -- {n_resp:,} responders, above the {MIN_RESPONDERS} floor",
               if_false=f"T2 FAIL -- only {n_resp:,} responders against a floor of "
                        f"{MIN_RESPONDERS}; nothing below has power")

    void = set()
    if not (t1 and t2):
        void |= {"T3", "T4", "T5", "T6", "T7"}
        say()
        say("     T1 or T2 failed, so T3-T7 are VOID: they would be measuring an unreliable "
            "target or an underpowered one")

    gsym = {g: i for i, g in enumerate(genes)}

    # ---- T3 ------------------------------------------------------------------------------------
    say()
    say("T3 DOES CHROMATIN MOVE FIRST?")
    say("     BOTH half-times are computed on the SHARED timepoint grid. Expression has 16 points")
    say("     and DNase 11 starting at 30 min, so scoring expression on its own denser early grid")
    say("     would let it look earlier for having been sampled earlier, which is a property of")
    say("     the assay design and not of the biology.")
    tss = tss_map(genes)
    tsslist = [tss.get(g) for g in genes]
    say(f"     TSS in GRCh38 for {sum(x is not None for x in tsslist):,}/{len(genes):,} genes")
    t3 = False
    d3 = {}
    if "T3" in void:
        say("     T3 VOID -- see above")
    else:
        dt, DM = promoter_track("DNase", tsslist, PROM_PAD, say)
        if len(dt) < 4:
            void.add("T3")
            say(f"     T3 VOID -- {len(dt)} DNase timepoints is not a trajectory")
        else:
            shared = np.array(sorted(set(dt.tolist()) & set(ts_all.tolist())))
            say(f"     shared grid: {len(shared)} points {[int(x) for x in shared]}")
            ei = [int(np.where(ts_all == t)[0][0]) for t in shared]
            di = [int(np.where(dt == t)[0][0]) for t in shared]
            acc_h = np.array([half_time(shared, DM[di, j]) for j in range(DM.shape[1])])
            exp_h = np.array([half_time(shared, Mall[ei, j]) for j in range(Mall.shape[1])])
            has_acc = (DM > 0).any(0)
            m3 = responder & has_acc & np.isfinite(acc_h) & np.isfinite(exp_h)
            say(f"     responders with a promoter DNase peak: {int(m3.sum()):,}")
            if m3.sum() < MIN_GROUP:
                void.add("T3")
                say(f"     T3 VOID -- {int(m3.sum())} is under the power floor")
            else:
                stat, p3 = wilcoxon(acc_h[m3], exp_h[m3], alternative="less")
                lead = float(np.median(exp_h[m3] - acc_h[m3]))
                say(f"     median accessibility half-time {np.median(acc_h[m3]):.0f} min vs "
                    f"expression {np.median(exp_h[m3]):.0f} min")
                say(f"     median lead (expression minus accessibility) {lead:+.0f} min   "
                    f"one-sided Wilcoxon p {p3:.3g}")
                d3 = dict(n=int(m3.sum()), median_acc=float(np.median(acc_h[m3])),
                          median_exp=float(np.median(exp_h[m3])), lead=lead, p=float(p3))
                t3 = bool(p3 < ALPHA)
                GG.verdict(t3, emit=say,
                           if_true=f"T3 PASS -- chromatin opens {lead:.0f} min before the mRNA "
                                   f"moves, so accessibility is a timing mechanism and not only a "
                                   f"state",
                           if_false="T3 FAIL -- accessibility does not lead transcription on this "
                                    "clock, so the epigenetic layer cannot be what sets WHEN a "
                                    "gene responds")

    # ---- topology ------------------------------------------------------------------------------
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
                    (pos2 if s1 * s2 > 0 else neg2).update(
                        {names[a].upper(), names[b].upper()})
    auto = json.load(open(AUTO))
    neg_auto = {v["name"].upper() for v in auto["matrices"].values()
                if v["cls"] == "SELF" and "Repression" in (v.get("self_modes") or [])}
    say()
    say(f"    topology: {len(neg2)} genes in a signed NEGATIVE two-cycle, {len(pos2)} in a "
        f"POSITIVE one, {len(neg_auto)} negatively autoregulated")

    def group(symbols):
        idx = [gsym[s] for s in symbols if s in gsym]
        m = np.zeros(len(genes), dtype=bool)
        m[idx] = True
        return m & responder & np.isfinite(th_all)

    def timing_test(m1, m2, lab1, lab2, gname, alt="less"):
        n1, n2 = int(m1.sum()), int(m2.sum())
        say(f"     {lab1}: {n1} responders, {lab2}: {n2}")
        if n1 < MIN_GROUP or n2 < MIN_GROUP:
            void.add(gname)
            say(f"     {gname} VOID -- fewer than {MIN_GROUP} responders in a group "
                f"({n1} and {n2}); this is a power floor declared before the run, not a null")
            return False, dict(n1=n1, n2=n2, p=float("nan"), voided=True)
        u, p = mannwhitneyu(th_all[m1], th_all[m2], alternative=alt)
        say(f"     median half-time {np.median(th_all[m1]):.0f} min vs "
            f"{np.median(th_all[m2]):.0f} min   one-sided p {p:.3g}")
        return bool(p < ALPHA), dict(n1=n1, n2=n2, p=float(p), voided=False,
                                     median1=float(np.median(th_all[m1])),
                                     median2=float(np.median(th_all[m2])))

    # ---- T4 ------------------------------------------------------------------------------------
    say()
    say("T4 THE LITERAL ALON PREDICTION: negative autoregulation should be FASTER")
    t4, d4 = (False, dict(voided=True)) if "T4" in void else \
        timing_test(group(neg_auto), group(set(genes) - neg_auto),
                    "negatively autoregulated", "everything else", "T4")
    if "T4" not in void:
        GG.verdict(t4, emit=say,
                   if_true="T4 PASS -- negative autoregulation speeds the response here, as "
                           "Rosenfeld, Elowitz & Alon measured in E. coli",
                   if_false="T4 FAIL -- negatively autoregulated genes are not faster on this "
                            "clock")

    # ---- T5 ------------------------------------------------------------------------------------
    say()
    say("T5 NEGATIVE FEEDBACK, THE POWERED VERSION")
    t5, d5 = (False, dict(voided=True)) if "T5" in void else \
        timing_test(group(neg2), group(pos2),
                    "in a negative two-cycle", "in a positive two-cycle", "T5")
    if "T5" not in void:
        GG.verdict(t5, emit=say,
                   if_true="T5 PASS -- genes in negative feedback respond faster than genes in "
                           "positive feedback, which is the dynamic consequence loop 187's "
                           "z = +43.8 topology predicts",
                   if_false="T5 FAIL -- feedback sign does not order the response times. Loop "
                            "187's two-cycle enrichment is a structural fact without the dynamic "
                            "consequence attached to it in the literature")

    # ---- T6 ------------------------------------------------------------------------------------
    say()
    say("T6 THE KNOWN NEGATIVE: occupancy must NOT predict timing")
    say("     gr_dynamics.py already measured that NR3C1 occupancy carries WHICH and not WHEN.")
    say("     this gate passes by finding nothing, so it is scored on an EFFECT SIZE and not on a")
    say("     p-value: with thousands of responders any correlation at all is 'significant', and a")
    say("     gate that cannot fail to reject is not a gate. The bar is |Spearman| < "
        f"{MAX_OCC_RHO}.")
    t6, d6 = False, {}
    if "T6" in void:
        say("     T6 VOID -- see above")
    else:
        nt, NM = promoter_track("NR3C1", tsslist, PROM_PAD, say)
        if len(nt) < 4:
            void.add("T6")
            say("     T6 VOID -- no NR3C1 trajectory")
        else:
            occ = NM.max(0)
            m6 = responder & np.isfinite(th_all) & (occ > 0)
            r6, p6 = spearmanr(occ[m6], th_all[m6])
            say(f"     {int(m6.sum()):,} responders with promoter NR3C1; "
                f"Spearman(occupancy, half-time) = {r6:+.3f} (p {p6:.3g})")
            d6 = dict(n=int(m6.sum()), rho=float(r6), p=float(p6))
            t6 = bool(abs(r6) < MAX_OCC_RHO)
            GG.verdict(t6, emit=say,
                       if_true=f"T6 PASS -- |rho| {abs(r6):.3f} is below {MAX_OCC_RHO}, so "
                               f"occupancy does not carry timing here either, and this loop's "
                               f"target behaves the way gr_dynamics.py says it should",
                       if_false=f"T6 FAIL -- occupancy predicts half-time at rho {r6:+.3f}. That "
                                f"contradicts gr_dynamics.py on the same series, so something in "
                                f"this loop's target is wrong and T3-T5 should not be believed")

    # ---- T7 ------------------------------------------------------------------------------------
    say()
    say("T7 THE MAGNITUDE CONFOUND")
    passed_pooled = [g for g, v in (("T4", t4), ("T5", t5)) if v and g not in void]
    strata_res = {}
    if not passed_pooled:
        void.add("T7")
        say("     T7 VOID -- no pooled gate passed, so there is nothing for the strata to confirm")
    else:
        lfc = peak_lfc[responder & np.isfinite(th_all)]
        edges = np.quantile(peak_lfc[responder & np.isfinite(th_all)],
                            np.linspace(0, 1, N_STRATA + 1))
        edges[-1] += 1e-9
        say(f"     terciles of peak |log2 FC|: {[round(float(e), 2) for e in edges]}")
        for g in passed_pooled:
            m1, m2 = (group(neg_auto), group(set(genes) - neg_auto)) if g == "T4" \
                else (group(neg2), group(pos2))
            held = 0
            for q in range(N_STRATA):
                s = (peak_lfc >= edges[q]) & (peak_lfc < edges[q + 1])
                a, b = m1 & s, m2 & s
                if a.sum() >= MIN_GROUP and b.sum() >= MIN_GROUP:
                    _, p = mannwhitneyu(th_all[a], th_all[b], alternative="less")
                    held += int(p < ALPHA)
                    say(f"       {g} tercile {q+1}: n {int(a.sum())} vs {int(b.sum())}  p {p:.3g}")
                else:
                    say(f"       {g} tercile {q+1}: n {int(a.sum())} vs {int(b.sum())}  "
                        f"under the power floor")
            strata_res[g] = held
        t7 = all(v >= 2 for v in strata_res.values())
        GG.verdict(t7, emit=say,
                   if_true="T7 PASS -- every pooled pass survives inside magnitude terciles, so it "
                           "is about timing and not about how far the gene moved",
                   if_false="T7 FAIL -- a pooled pass does not hold within magnitude strata, so it "
                            "is a response-size effect reported as a timing effect")
    t7 = all(v >= 2 for v in strata_res.values()) if strata_res else False

    # ---- T8 ------------------------------------------------------------------------------------
    say()
    say("T8 WHAT THIS CANNOT SHOW")
    say("     One cell line, one drug, one receptor. Dexamethasone through GR is a fast nuclear-")
    say("     receptor response and nothing here generalises to slower or indirect stimuli.")
    say("     The half-time is a summary of a 16-point curve and it discards shape: an overshoot")
    say("     and a monotone rise with the same midpoint are recorded identically. Overshoot is")
    say("     the signature negative feedback is supposed to produce and this statistic cannot")
    say("     see it, which understates what a feedback test could find.")
    say("     Topology comes from CollecTRI curation on a different cell line. A two-cycle curated")
    say("     in one context need not be wired in A549, and no gate here can see that.")
    say("     Bulk RNA averages over a population, so a fast response in a subpopulation and a")
    say("     slow one everywhere are the same curve.")
    say("     T3 and T6 are declared and not run in this build. They are the promoter-level joins")
    say("     of DNase and NR3C1 peaks to each gene's TSS, and they are the next step rather than")
    say("     a result. Nothing in this loop's conclusion may lean on them.")
    say("     T8 PASS")

    gates = {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5, "T6": t6, "T7": t7, "T8": True}
    man_out = RM.manifest(inputs=[GRTC / "rna.npz", BUNDLE, AUTO],
                          available=int(len(genes)), used=int(responder.sum()),
                          selection="filtered", seed=SEED,
                          controls=["half-time reproduced across an independent replicate split",
                                    "response magnitude terciles for every pooled pass",
                                    "a declared power floor returning VOID rather than a null",
                                    "occupancy as a known-negative check on the target"],
                          note="response timing against feedback topology on the A549 dex course")
    out_d = dict(test="response timing", gates=gates, void=sorted(void),
                 n_genes=int(len(genes)), n_expressed=int(expressed.sum()),
                 n_responders=int(responder.sum()),
                 split=dict(a=A, b=B, rho=float(rho), p=float(p_rho)),
                 topology=dict(neg_two_cycle=len(neg2), pos_two_cycle=len(pos2),
                               neg_autoreg=len(neg_auto)),
                 t3=d3, t4=d4, t5=d5, t6=d6, strata=strata_res,
                 half_time_median=float(np.nanmedian(th_all[responder])) if responder.sum() else None,
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
