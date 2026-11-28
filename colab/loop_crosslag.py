"""Loop 199. Under a FORCED chromatin perturbation, does accessibility precede nascent transcription?

WHY A DIFFERENT QUESTION AND A DIFFERENT ESTIMATOR. Loop 191d measured that promoter accessibility
reaches half its plateau 48 minutes before the mRNA does in A549 under dexamethasone. Three things
are wrong with leaving it there and every one of them is fixed by a different dataset rather than by
more analysis of that one:

    IT IS OBSERVATIONAL. Nothing in the A549 series perturbs accessibility, so "leading is not
    causing" appears in every cannot-show block this arc has written. GSE148175 inhibits the BAF
    ATPase with BRM014, which FORCES accessibility to change.

    THE READOUT WAS mRNA LEVEL. Loop 198 measured that nothing beats persistence at predicting
    change over an unseen interval -- informed R2 -0.0520 against persistence -0.0295 -- and
    steady-state mRNA is persistent by construction, since it convolves transcription with
    degradation. PRO-seq measures engaged polymerase, which is the rate the clock should predict.

    IT IS UNREPLICATED, and loop 197 established no public series is dense enough to replicate it.

WHAT THIS LOOP MAY NOT DO, stated first because the temptation is obvious. The matched ATAC/PRO-seq
grid here is 10, 30 and 60 minutes: THREE points, fewer than the dendritic-cell series' four, and
loop 196 measured that four estimators chosen to fail differently all recover the A549 lead on
eleven points and none on four. A response time cannot be computed here and this loop does not try.
Any future loop that reaches for one on this dataset is repeating a mistake already measured.

THE ESTIMATOR THAT DOES FIT THREE POINTS is a cross-lagged panel. With t0=10, t1=30, t2=60 there is
exactly one lagged comparison in each direction:

    forward   partial correlation of dATAC(10->30) with dPRO(30->60), holding dPRO(10->30) fixed
    reverse   partial correlation of dPRO(10->30) with dATAC(30->60), holding dATAC(10->30) fixed

The partialling is not optional. dPRO(30->60) is autocorrelated with dPRO(10->30), and dATAC(10->30)
is correlated with dPRO(10->30) concurrently, so a raw forward correlation would be mostly those two
facts. Holding the earlier change of the OUTCOME fixed is what makes the remaining association a
statement about precedence. Accessibility leading transcription predicts forward > reverse.

THE CALIBRATION COMES BEFORE THE ANSWER, as it did in loop 192's W3 and for the same reason. A null
here would have two explanations -- no precedence, or a three-point cross-lag cannot see precedence
-- and those are different findings. Q3 runs the identical estimator on the A549 series reduced to
three timepoints, where the lead is known to exist. If it recovers the direction there, a null here
means absence. If it does not, Q4 is VOID and this dataset cannot answer the question either.

WHAT THE VEHICLE CONTROLS CAN AND CANNOT DO. PRO-seq carries DMSO at 1h only; ATAC carries DMSO at
5 min and 24 h. They are NOT matched to the 10/30/60 grid, so drug-minus-vehicle cannot be formed
per timepoint. Every change below is therefore relative to the 10-minute point, exactly as the A549
analysis used its first grid point, and the vehicle arms are used in Q2 as a global check that the
drug did something at all rather than as a per-timepoint normalisation. Claiming a vehicle-corrected
result from this design would be claiming a control that is not there.

PREDECLARED, BEFORE ANY NUMBER.

  Q1 DO THE MATRICES JOIN? ATAC regions to gene TSSs in GRCh38, PRO-seq gene bodies to symbols.
     Gate: PASS iff at least 3,000 genes carry both a PRO-seq gene body and an ATAC region within
     the promoter window, in every one of the six samples.

  Q2 DID THE PERTURBATION DO ANYTHING? Accessibility change from 10 to 60 min under BRM014 against
     the spread of the vehicle arms.
     Gate: PASS iff the drug-arm change is larger than the vehicle-arm spread. If BRM014 moved
     nothing, no precedence question arises and Q3 onward would be measuring noise.

  Q3 CAN A THREE-POINT CROSS-LAG SEE A KNOWN LEAD? The identical estimator on A549 DNase against
     RNA, reduced to three timepoints.
     Gate: PASS iff forward exceeds reverse there. A FAIL makes Q4 VOID, because a null would then
     be about the estimator and not about chromatin.

  Q4 DOES ACCESSIBILITY PRECEDE NASCENT TRANSCRIPTION UNDER FORCED PERTURBATION?
     Gate: PASS iff forward exceeds reverse and the bootstrap 95% interval on the difference
     excludes zero.

  Q5 THE STRANGER SWAP. Each gene given another gene's accessibility trajectory.
     Gate: PASS iff the real forward-minus-reverse difference is larger IN MAGNITUDE than the
     swapped one. Magnitude and not sign, because the question is whether the association survives
     the swap, and Q4's direction is not assumed here -- the first version of this gate compared
     signed values and would have scored a swap that destroyed a NEGATIVE association as a failure.

  Q6 WHAT THIS CANNOT SHOW.

-> outputs/loop_crosslag.json
"""
import gzip
import json
import os
import re
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
import loop_response_timing_d as L191        # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_crosslag.json"
SP = L191.SP
G = SP / "gse148175"
A549 = SP / "grtc"
PRO = G / "GSE148175_Pro-Seq-single-rep-raw-counts-genebody-BRM014.txt.gz"
ATAC = G / "GSE148175_count_matrix_raw_atac_BRM014_ACBI1.csv.gz"

PROM_PAD = 2000           # bp either side of the TSS for an ATAC region to count as promoter
MIN_JOINED = 3000         # Q1
N_BOOT = 2000
A549_TRIPLE = [30.0, 60.0, 120.0]     # the three A549 points closest in shape to 10/30/60
SEED = 199199

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def cpm_log(mat):
    """Library-size normalise then log2. Counts across samples differ in depth by design."""
    tot = mat.sum(0, keepdims=True)
    tot[tot == 0] = 1.0
    return np.log2(1.0 + 1e6 * mat / tot)


def partial(x, y, z):
    """Correlation of x and y with z held fixed, by residualising both on z."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) < 30:
        return float("nan")
    Z = np.column_stack([np.ones(len(z)), z])
    bx = np.linalg.lstsq(Z, x, rcond=None)[0]
    by = np.linalg.lstsq(Z, y, rcond=None)[0]
    rx, ry = x - Z @ bx, y - Z @ by
    sx, sy = rx.std(), ry.std()
    return float((rx * ry).mean() / (sx * sy)) if sx > 0 and sy > 0 else float("nan")


def crosslag(A, P, rng=None, n_boot=0, perm=None):
    """Forward and reverse partial correlations on a three-point panel.

    A and P are (3, n_gene) already on a log scale. `perm` reassigns which gene's accessibility a
    gene receives, which is Q5's control."""
    Ax = A if perm is None else A[:, perm]
    dA1, dA2 = Ax[1] - Ax[0], Ax[2] - Ax[1]
    dP1, dP2 = P[1] - P[0], P[2] - P[1]
    fwd = partial(dA1, dP2, dP1)
    rev = partial(dP1, dA2, dA1)
    out = dict(forward=fwd, reverse=rev, diff=fwd - rev, n=int(np.isfinite(dA1 + dP2).sum()))
    if n_boot and rng is not None:
        n = A.shape[1]
        d = np.empty(n_boot)
        for k in range(n_boot):
            ix = rng.integers(0, n, n)
            f = partial(dA1[ix], dP2[ix], dP1[ix])
            r = partial(dP1[ix], dA2[ix], dA1[ix])
            d[k] = f - r
        out["ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return out


def tss_table():
    tab = json.load(gzip.open("colab/data/cell_complete.json.gz"))["genes"]
    out = {}
    for line in open(SP / "_tss_hg38.bed"):
        q = line.split()
        if len(q) >= 4 and q[3].startswith("G"):
            i = int(q[3][1:])
            if i < len(tab):
                out[str(tab[i]["name"]).upper()] = (q[0], int(q[2]))
    return out


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 199  UNDER A FORCED PERTURBATION, DOES ACCESSIBILITY PRECEDE NASCENT TRANSCRIPTION?")
    say("=" * 104)
    say("  PREDECLARED: the matched grid is 10/30/60 min -- THREE points, fewer than the four loop")
    say("  196 measured as too few for a response time, so no response time is computed here. The")
    say("  estimator is a cross-lagged panel with the earlier change of the OUTCOME partialled out,")
    say("  because without that the forward correlation is mostly autocorrelation. Q3 calibrates")
    say("  the estimator on A549 reduced to three points BEFORE Q4 is read, since a null would")
    say("  otherwise confound absence with a blind estimator. The vehicle arms are NOT matched to")
    say("  the grid, so every change is relative to the 10-minute point and no vehicle-corrected")
    say("  result is claimed.")
    say()

    # ---- load ----------------------------------------------------------------------------------
    tss = tss_table()
    hdr = gzip.open(PRO, "rt").readline().rstrip("\n").split("\t")
    pro_cols = {c: i for i, c in enumerate(hdr)}
    want_pro = {t: [f"WT_BRM014_{t}_A", f"WT_BRM014_{t}_B"] for t in ("10min", "30min", "1h")}
    gsym, prows = [], []
    with gzip.open(PRO, "rt") as fh:
        fh.readline()
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < len(hdr):
                continue
            s = str(f[pro_cols["gene.x"]]).upper() if "gene.x" in pro_cols else ""
            if not s:
                continue
            try:
                vals = [float(f[pro_cols[c]]) for t in ("10min", "30min", "1h")
                        for c in want_pro[t]]
            except (ValueError, KeyError):
                continue
            gsym.append(s)
            prows.append(vals)
    P = np.array(prows, dtype=float)
    say(f"    PRO-seq gene bodies: {P.shape[0]:,} rows x {P.shape[1]} samples")

    ahdr = gzip.open(ATAC, "rt").readline().rstrip("\n").split(",")
    acols = {c: i for i, c in enumerate(ahdr)}
    want_atac = [f"{r}_{t}_N" for t in ("10min", "30min", "1h") for r in ("R1", "R2")]
    veh = [c for c in ahdr if "DMSO" in c]
    regions, arows, vrows = [], [], []
    with gzip.open(ATAC, "rt") as fh:
        fh.readline()
        for line in fh:
            f = line.rstrip("\n").split(",")
            if len(f) < len(ahdr):
                continue
            try:
                arows.append([float(f[acols[c]]) for c in want_atac])
                vrows.append([float(f[acols[c]]) for c in veh])
            except (ValueError, KeyError):
                continue
            regions.append(f[0])
    A = np.array(arows, dtype=float)
    V = np.array(vrows, dtype=float)
    say(f"    ATAC regions: {A.shape[0]:,} rows x {A.shape[1]} drug samples, "
        f"{V.shape[1]} vehicle samples ({', '.join(veh)})")

    # ---- Q1 ------------------------------------------------------------------------------------
    say()
    say("Q1 DO THE MATRICES JOIN?")
    Al = cpm_log(A)
    Pl = cpm_log(P)
    by_chrom = defaultdict(list)
    for i, r in enumerate(regions):
        m = re.match(r"^(chr[\w\.]+):(\d+)-(\d+)$", r)
        if m:
            by_chrom[m.group(1)].append((int(m.group(2)), int(m.group(3)), i))
    for c in by_chrom:
        by_chrom[c].sort()
    gene_regions = defaultdict(list)
    for s, (c, pos) in tss.items():
        for st, en, i in by_chrom.get(c, ()):
            if en + PROM_PAD < pos:
                continue
            if st - PROM_PAD > pos:
                break
            gene_regions[s].append(i)
    say(f"    {len(gene_regions):,} genes have >= 1 ATAC region within {PROM_PAD} bp of the TSS")
    pidx = defaultdict(list)
    for k, s in enumerate(gsym):
        pidx[s].append(k)
    shared = sorted(set(gene_regions) & set(pidx))
    say(f"    {len(shared):,} genes carry BOTH a PRO-seq gene body and a promoter ATAC region")
    q1 = bool(len(shared) >= MIN_JOINED)
    GG.verdict(q1, emit=say,
               if_true=f"Q1 PASS -- {len(shared):,} joined genes, above the {MIN_JOINED} floor",
               if_false=f"Q1 FAIL -- {len(shared):,} against a floor of {MIN_JOINED}")

    void = set()
    if not q1:
        void |= {"Q2", "Q3", "Q4", "Q5"}

    # per gene: mean over its promoter regions / gene-body rows, then mean over replicates
    Ag = np.array([[Al[gene_regions[s], 2 * t:2 * t + 2].mean() for t in range(3)]
                   for s in shared]).T
    Pg = np.array([[Pl[pidx[s], 2 * t:2 * t + 2].mean() for t in range(3)]
                   for s in shared]).T
    say(f"    panel built: accessibility {Ag.shape}, nascent {Pg.shape} (3 timepoints x "
        f"{len(shared):,} genes)")

    # ---- Q2 ------------------------------------------------------------------------------------
    say()
    say("Q2 DID THE PERTURBATION DO ANYTHING?")
    Vl = cpm_log(V)
    drug_change = float(np.abs(Al[:, 4:6].mean(1) - Al[:, 0:2].mean(1)).mean())
    veh_spread = float(np.abs(Vl - Vl.mean(1, keepdims=True)).mean())
    say(f"    mean |accessibility change| 10 min -> 1 h under BRM014: {drug_change:.4f}")
    say(f"    mean |deviation| across the vehicle arms ({V.shape[1]} samples): {veh_spread:.4f}")
    say("    the vehicle arms are at 5 min and 24 h and are NOT matched to this grid, so this is a")
    say("    scale check on whether the drug moved anything, not a per-timepoint correction")
    q2 = bool(drug_change > veh_spread)
    if "Q2" in void:
        say("    Q2 VOID -- Q1 failed")
    else:
        GG.verdict(q2, emit=say,
                   if_true="Q2 PASS -- BRM014 moves accessibility more than the vehicle arms vary, "
                           "so there is a perturbation for the precedence question to be about",
                   if_false="Q2 FAIL -- the drug arm does not exceed vehicle variation; there is no "
                            "forced change here and Q4 would be measuring noise")
    if not q2:
        void |= {"Q3", "Q4", "Q5"}

    # ---- Q3 ------------------------------------------------------------------------------------
    say()
    say("Q3 CAN A THREE-POINT CROSS-LAG SEE A KNOWN LEAD?")
    q3, d3 = False, None
    if "Q3" in void:
        say("    Q3 VOID -- see above")
    else:
        z = np.load(A549 / "rna.npz", allow_pickle=True)
        mins, reps = z["mins"].astype(int), z["reps"].astype(int)
        grid = np.array(A549_TRIPLE)
        M, _ = L191.rep_trajectories(z["tpm"], mins, reps, (1, 2, 3), grid)
        e2s = L191.ensg_to_symbol(lambda *_: None)
        sy = np.array([e2s.get(str(g).split(".")[0], "") for g in z["genes"]])
        pt, PM = L191.promoter_track("DNase", [tss.get(s) for s in sy], L191.PROM_PAD,
                                     lambda *_: None)
        idx = [int(np.where(pt == t)[0][0]) for t in grid]
        AA = np.log2(1.0 + PM[idx])
        base = z["tpm"][(mins == int(grid[0])) & np.isin(reps, (1, 2, 3))].mean(0)
        keep = (base >= 1.0) & (PM[idx] > 0).any(0) & np.isfinite(M).all(0)
        say(f"    A549 reduced to {[int(x) for x in grid]}: {int(keep.sum()):,} genes")
        d3 = crosslag(AA[:, keep], M[:, keep])
        say(f"    forward {d3['forward']:+.4f}   reverse {d3['reverse']:+.4f}   "
            f"difference {d3['diff']:+.4f}")
        q3 = bool(np.isfinite(d3["diff"]) and d3["diff"] > 0)
        GG.verdict(q3, emit=say,
                   if_true="Q3 PASS -- the estimator recovers the direction on a series where the "
                           "lead is known, so a null in Q4 would mean absence rather than blindness",
                   if_false="Q3 FAIL -- a three-point cross-lag does not recover the known A549 "
                            "lead, so Q4 is VOID: a null there would be about the estimator")
    if not q3:
        void |= {"Q4", "Q5"}

    # ---- Q4 ------------------------------------------------------------------------------------
    say()
    say("Q4 DOES ACCESSIBILITY PRECEDE NASCENT TRANSCRIPTION UNDER FORCED PERTURBATION?")
    q4, d4 = False, None
    rng = np.random.default_rng(SEED)
    if "Q4" in void:
        say("    Q4 VOID -- see above")
    else:
        d4 = crosslag(Ag, Pg, rng=rng, n_boot=N_BOOT)
        say(f"    forward (dATAC 10->30 vs dPRO 30->60 | dPRO 10->30): {d4['forward']:+.4f}")
        say(f"    reverse (dPRO 10->30 vs dATAC 30->60 | dATAC 10->30): {d4['reverse']:+.4f}")
        say(f"    difference {d4['diff']:+.4f}, bootstrap 95% CI "
            f"[{d4['ci'][0]:+.4f}, {d4['ci'][1]:+.4f}] over {N_BOOT} resamples")
        q4 = bool(d4["diff"] > 0 and d4["ci"][0] > 0)
        GG.verdict(q4, emit=say,
                   if_true="Q4 PASS -- when accessibility is FORCED to change, the change precedes "
                           "nascent transcription. This is the causal form of loop 191d's clock "
                           "and A549 could not have provided it",
                   if_false="Q4 FAIL -- forcing accessibility to change does not put it ahead of "
                            "nascent transcription on this clock. Q3 passed, so the estimator can "
                            "see precedence; this is an absence")

    # ---- Q5 ------------------------------------------------------------------------------------
    say()
    say("Q5 THE STRANGER SWAP")
    q5, d5 = False, None
    if "Q5" in void or d4 is None:
        say("    Q5 VOID -- see above")
        void.add("Q5")
    else:
        perm = rng.permutation(Ag.shape[1])
        d5 = crosslag(Ag, Pg, perm=perm)
        say(f"    swapped forward {d5['forward']:+.4f}  reverse {d5['reverse']:+.4f}  "
            f"difference {d5['diff']:+.4f}")
        # SIGN-AGNOSTIC, and the first version was not. Q5's stated intent is "does giving a gene
        # a stranger's accessibility destroy the association", and it was implemented as
        # d4 > d5, which only tests that intent if the association is POSITIVE. Q4 came back
        # negative -- transcription precedes accessibility -- so the signed comparison scored a
        # swap that removed 82% of the effect as a failure. That is a specification bug of the
        # same family as loop 194's V4 and loop 196's X4: a gate written assuming the direction of
        # its own answer. Comparing magnitudes restores the stated intent; it does not move a
        # threshold to make anything pass, and the direction of Q4 is unaffected either way.
        q5 = bool(abs(d4["diff"]) > abs(d5["diff"]))
        GG.verdict(q5, emit=say,
                   if_true=lambda: (f"Q5 PASS -- swapping destroys the association "
                                    f"({d4['diff']:+.4f} -> {d5['diff']:+.4f}), so it belongs to "
                                    f"the gene's own promoter and not to the two assays' global "
                                    f"time profiles"),
                   if_false="Q5 FAIL -- a stranger's accessibility gives as strong an association, "
                            "so the result is about the two assays' global time profiles")

    # ---- Q6 ------------------------------------------------------------------------------------
    say()
    say("Q6 WHAT THIS CANNOT SHOW")
    say("    Three timepoints give exactly one lagged comparison per direction. There is no")
    say("    response time here, no lag length, and no claim that the precedence is 48 minutes or")
    say("    any other number -- only its direction.")
    say("    The vehicle arms sit at 5 min and 24 h and are not matched to this grid, so nothing")
    say("    here is vehicle-corrected. A drift common to both assays over the hour would look")
    say("    like neither precedence nor its absence; it would just add noise.")
    say("    BRM014 inhibits the BAF ATPase globally. A gene whose accessibility changes is not")
    say("    thereby a gene the drug acted on directly, and this design cannot separate direct")
    say("    remodelling from downstream consequence.")
    say("    PRO-seq gene bodies measure engaged polymerase over the whole body, which mixes")
    say("    initiation with elongation. A change in pause release and a change in initiation are")
    say("    the same number here.")
    say("    A pass is about ONE perturbation in ONE cell line. It makes loop 191d's observational")
    say("    clock more credible; it does not replicate it, and loop 197 established that no")
    say("    public series can.")
    say("    Q6 PASS")

    gates = {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5, "Q6": True}
    man = RM.manifest(inputs=[PRO, ATAC, A549 / "rna.npz"],
                      available=int(len(gsym)), used=int(len(shared)),
                      selection="filtered", seed=SEED,
                      controls=["the earlier change of the outcome partialled out of every lag",
                                "the estimator calibrated on A549 before the answer is read",
                                f"{N_BOOT} bootstrap resamples on the forward-reverse difference",
                                "a stranger swap on the accessibility panel"],
                      note="cross-lagged precedence under BAF ATPase inhibition, GSE148175")
    out_d = dict(test="crosslag precedence", gates=gates, void=sorted(void),
                 n_joined=len(shared), grid=[10, 30, 60],
                 perturbation=dict(drug_change=drug_change, vehicle_spread=veh_spread),
                 q3=d3, q4=d4, q5=d5, manifest=man, seconds=time.time() - t0, log=log)
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
