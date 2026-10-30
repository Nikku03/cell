"""Loop 164. Does ESM-2 650M beat 35M? A SCREEN at n=500, and it says what it cannot see.

WHAT THIS IS AND IS NOT. Measured on this machine, 650M runs at 2.58 s/protein against 35M's 0.405
-- 6.4x -- so all 2,178 proteins would take 1.6 h and 500 take 21.5 min. 500 was chosen knowing
that, and the consequence is arithmetic rather than opinion: the paired standard error on this
benchmark scales as roughly 0.09/sqrt(n), so

    n = 2,178  ->  sem ~0.0019  ->  3 sem is ~0.006
    n =   500  ->  sem ~0.0040  ->  3 sem is ~0.012

The effects this arc has been measuring are +0.0058 for geometry and +0.0123 for all four blocks.
At n=500 the first is INVISIBLE and the second is exactly at the edge. So this loop can detect a
large gain from 650M and cannot detect a small one, and G2 states that in the artefact rather than
leaving a null to be read as evidence of absence.

WHY IT IS WORTH RUNNING ANYWAY. 8M scores 0.7760 and 35M scores 0.7941 -- a 4.4x parameter increase
bought +0.0181. If 650M follows that slope it lands near +0.03, which n=500 CAN see at 7.5 sem. So
the screen has a real hypothesis to reject, and the reason for running it is that rejecting a large
effect cheaply is worth more than confirming a small one expensively.

EVERYTHING ELSE IS HELD FIXED from loop 163d: the same symmetric frequency-matched mini-contests,
the same homology-disjoint folds recomputed on the subset, the same k-NN, the same geometry,
electrostatic and steric blocks. Only the sequence encoder changes.

PREDECLARED, before any number is looked at.

  G1 THE INSTRUMENT SURVIVES THE SUBSET. Popularity on the matched contests over these 500
     proteins, and identical accessions across all five blocks.
     Gate: |AUC - 0.5| <= 0.02. A 500-protein subset has fewer contests and a coarser frequency
     ladder, so the matching that worked at 2,178 has to be re-earned here, not assumed.

  G2 WHAT THIS SCREEN CAN AND CANNOT SEE. Report the measured paired sem on this subset and the
     minimum detectable effect at 3 sem, BEFORE reporting any comparison.
     Gate: passes on being reported. Its purpose is that G3's and G4's numbers are read against a
     stated resolution instead of against zero.

  G3 DOES 650M BEAT 35M, AS A SEQUENCE ARM ALONE? Both re-measured on the same 500 proteins, same
     folds, paired.
     Gate: PASS iff 650M beats 35M by more than 3 sem of the paired difference. A FAIL here means
     only that the screen did not see a gain of that size, and the verdict text must say so.

  G4 DOES IT BEAT 35M INSIDE THE FOUR-BLOCK MERGE? Substitute 650M for 35M in loop 163d's winning
     configuration -- sequence + 0.1*geometry + 0.1*electrostatics + 0.1*sterics -- and compare.
     Gate: more than 3 sem.

  G5 ARE THE TWO ENCODERS EVEN DIFFERENT? Spearman between the 650M and 35M per-case scores, and
     the per-case oracle over the pair.
     Gate: passes on being reported. If they correlate near 1.0 and the oracle is flat, then a
     bigger encoder is reading the same thing and the full 1.6 h run would not change that.

  G6 WHAT THIS CANNOT SHOW. n=500 cannot resolve an effect below about 0.012. Fewer training
     proteins per fold degrades k-NN for EVERY arm, so absolute numbers here sit below the
     2,178-protein run and only the paired differences are comparable. And a screen that fails to
     see an effect has not shown there is none.

-> outputs/loop_esm650_screen.json
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
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                              # noqa: E402
import run_manifest as RM                            # noqa: E402
import loop_replication as LR                        # noqa: E402
from rem.harness import REM                          # noqa: E402
from loop_struct_vs_seq import homology_folds, knn_scores, NFOLD, SEED  # noqa: E402

SEQF = Path("colab/data/ml/esm_enzymes.npz")
STRF = Path("colab/data/ml/struct_enzymes.npz")
ESF = Path("colab/data/ml/elecster_enzymes.npz")
M650 = Path("colab/data/ml/esm650_subset.npz")
OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_esm650_screen.json"
NEG_PER_POS, TOL = 40, 0.02
W = (0.1, 0.1, 0.1)          # loop 163d's fitted weights, FROZEN -- nothing is tuned here

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def contest_auc(sv, mini):
    if not mini:
        return np.nan
    return float(np.mean([float((sv[n] < sv[p]).sum() + 0.5 * (sv[n] == sv[p]).sum()) / len(n)
                          for p, n in mini]))


def main():
    t0 = time.time()
    say("=" * 104)
    say("  ESM-2 650M vs 35M -- a SCREEN at n=500, sized from a measured 2.58 s/protein")
    say("=" * 104)
    say()

    S = np.load(SEQF, allow_pickle=False)
    T = np.load(STRF, allow_pickle=False)
    E = np.load(ESF, allow_pickle=False)
    M = np.load(M650, allow_pickle=False)
    common = sorted(set(map(str, S["accs"])) & set(map(str, T["accs"]))
                    & set(map(str, E["accs"])) & set(map(str, M["accs"])))
    ix = [{a: i for i, a in enumerate(map(str, z["accs"]))} for z in (S, T, E, M)]
    E35 = S["esm35"][[ix[0][a] for a in common]]
    GEO = T["X"][[ix[1][a] for a in common]]
    ELE = E["elec"][[ix[2][a] for a in common]]
    STE = E["steric"][[ix[2][a] for a in common]]
    E650 = M["esm650"][[ix[3][a] for a in common]]
    say(f"     {len(common)} proteins in all five blocks | esm35 {E35.shape[1]}d, "
        f"esm650 {E650.shape[1]}d, geometry {GEO.shape[1]}d, elec {ELE.shape[1]}d, "
        f"steric {STE.shape[1]}d")

    R = REM()
    Z = np.load("colab/data/rem_enzyme.npz", allow_pickle=False)
    sym = list(map(str, Z["symbols"]))
    grx = defaultdict(set)
    for j, g in zip(Z["gpr_rx"], Z["gpr_gene"]):
        grx[sym[int(g)]].add(int(j))
    a2g, seqs, acc, buf = {}, {}, None, []
    with gzip.open(LR.SC / "human_proteome.fasta.gz", "rt", errors="replace") as f:
        for ln in f:
            if ln.startswith(">"):
                if acc and buf:
                    seqs[acc] = "".join(buf)
                m = re.match(r">\w\w\|([^|]+)\|", ln)
                g = re.search(r"GN=(\S+)", ln)
                acc, buf = (m.group(1) if m else None), []
                if acc and g:
                    a2g[acc] = g.group(1)
            else:
                buf.append(ln.strip())
    if acc and buf:
        seqs[acc] = "".join(buf)

    Y = np.zeros((len(common), len(R.noncur)), np.float32)
    for i, a in enumerate(common):
        for j in grx.get(a2g.get(a, ""), ()):
            for m in (R.react_of[j] | R.prod_of[j]) - R.currency:
                Y[i, R.ncmap[int(m)]] = 1.0
    keep = Y.sum(1) > 0
    accs = [a for a, k in zip(common, keep) if k]
    E35, GEO, ELE, STE, E650, Y = (E35[keep], GEO[keep], ELE[keep], STE[keep],
                                   E650[keep], Y[keep])
    pop = Y.mean(0)

    order = np.argsort(pop, kind="stable")
    posn = np.empty(len(pop), int)
    posn[order] = np.arange(len(pop))
    half = NEG_PER_POS // 2
    cand, ndrop = [], 0
    for i in range(len(accs)):
        mini = []
        for p in np.where(Y[i] > 0)[0]:
            b, a_ = [], []
            k = posn[p] - 1
            while k >= 0 and len(b) < half:
                if Y[i, order[k]] == 0:
                    b.append(order[k])
                k -= 1
            k = posn[p] + 1
            while k < len(order) and len(a_) < half:
                if Y[i, order[k]] == 0:
                    a_.append(order[k])
                k += 1
            if len(b) == half and len(a_) == half:
                mini.append((p, np.array(b + a_)))
            else:
                ndrop += 1
        cand.append(mini)
    fold, ncl, _ = homology_folds(seqs, accs)
    say(f"     {sum(len(m) for m in cand):,} mini-contests, {ndrop:,} positives dropped | "
        f"{ncl} homology clusters, folds {[int((fold == f).sum()) for f in range(NFOLD)]}")

    def zs(X):
        return (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    B = {"esm35": zs(E35), "esm650": zs(E650), "geometry": zs(GEO),
         "electrostatics": zs(ELE), "sterics": zs(STE)}
    P = {k: np.zeros_like(Y) for k in B}
    for f in range(NFOLD):
        te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
        for k, X in B.items():
            P[k][te] = knn_scores(X[tr], Y[tr], X[te])
    A = {k: np.array([contest_auc(P[k][i], cand[i]) for i in range(len(accs))]) for k in B}
    A["popularity"] = np.array([contest_auc(pop, cand[i]) for i in range(len(accs))])
    ok = np.isfinite(A["esm35"]) & np.isfinite(A["esm650"])

    def mn(a):
        return float(np.nanmean(a[ok]))

    def pdiff(a, b):
        d = a[ok] - b[ok]
        return float(d.mean()), float(d.std() / np.sqrt(len(d)))

    # ------------------------------------------------------------------ G1
    g1 = bool(abs(mn(A["popularity"]) - 0.5) <= TOL)
    say()
    say(f"G1 popularity on this subset: {mn(A['popularity']):.4f} (gate |AUC-0.5| <= {TOL})")
    GG.verdict(g1, emit=say, if_true=(
        "the frequency matching re-earns itself on 500 proteins."), if_false=(
        "the matching does not hold on the subset -- a coarser frequency ladder over fewer "
        "contests leaves a gradient, and everything below inherits it."))
    say(f"     G1 {'PASS' if g1 else 'FAIL'}")

    # ------------------------------------------------------------------ G2
    _, sem35 = pdiff(A["esm650"], A["esm35"])
    mde = 3 * sem35
    say()
    say("G2 WHAT THIS SCREEN CAN AND CANNOT SEE")
    say(f"     n = {int(ok.sum())} enzymes scored")
    say(f"     paired sem on the 650M-vs-35M difference: {sem35:.4f}")
    say(f"     minimum detectable effect at 3 sem      : {mde:.4f}")
    say(f"     for scale, the effects this arc has measured: geometry +0.0058, four blocks +0.0123")
    say(f"     8M -> 35M was +0.0181 for 4.4x the parameters; 650M is 18.6x the parameters again")
    g2 = True
    say(f"     G2 {'PASS' if g2 else 'FAIL'}")
    say()
    for k in sorted(B, key=lambda x: -mn(A[x])):
        say(f"     {k:<16s} {mn(A[k]):.4f}")
    say(f"     {'popularity':<16s} {mn(A['popularity']):.4f}")

    # ------------------------------------------------------------------ G3
    d3, s3 = pdiff(A["esm650"], A["esm35"])
    g3 = bool(d3 > 3 * s3)
    say()
    say(f"G3 650M minus 35M, as a sequence arm alone: {d3:+.4f} sem {s3:.4f} = {d3/s3:+.1f} sem")
    GG.verdict(g3, emit=say, if_true=(
        f"650M beats 35M by {d3:+.4f}, above this screen's {mde:.4f} resolution. The full "
        f"2,178-protein run is justified."), if_false=(
        f"this screen did not see a gain of {mde:.4f} or more. That is NOT evidence that 650M is no "
        f"better -- the measured difference is {d3:+.4f} and anything below {mde:.4f} is invisible "
        f"at n={int(ok.sum())}. What it rules out is a LARGE gain, and a large gain was the "
        f"hypothesis worth screening cheaply."))
    say(f"     G3 {'PASS' if g3 else 'FAIL'}")

    # ------------------------------------------------------------------ G4
    def nmax(v):
        m = v.max()
        return v / m if m > 0 else v
    merged = {}
    for seqk in ("esm35", "esm650"):
        merged[seqk] = np.array([contest_auc(
            nmax(P[seqk][i]) + W[0] * nmax(P["geometry"][i]) + W[1] * nmax(P["electrostatics"][i])
            + W[2] * nmax(P["sterics"][i]), cand[i]) for i in range(len(accs))])
    d4, s4 = pdiff(merged["esm650"], merged["esm35"])
    g4 = bool(d4 > 3 * s4)
    say()
    say(f"G4 inside loop 163d's four-block merge at frozen weights {W}:")
    say(f"     with 35M  {np.nanmean(merged['esm35'][ok]):.4f}   "
        f"with 650M {np.nanmean(merged['esm650'][ok]):.4f}")
    say(f"     difference {d4:+.4f} sem {s4:.4f} = {d4/s4:+.1f} sem")
    GG.verdict(g4, emit=say, if_true="650M improves the merged predictor too.",
               if_false="the screen did not see a gain inside the merge either, at this resolution.")
    say(f"     G4 {'PASS' if g4 else 'FAIL'}")

    # ------------------------------------------------------------------ G5
    rho = float(stats.spearmanr(A["esm650"][ok], A["esm35"][ok]).statistic)
    orc = np.maximum(A["esm650"], A["esm35"])
    d5, s5 = pdiff(orc, A["esm35"])
    say()
    say("G5 ARE THE TWO ENCODERS EVEN DIFFERENT?")
    say(f"     Spearman between per-case scores: {rho:+.4f}")
    say(f"     per-case oracle over the pair {mn(orc):.4f}, i.e. {d5:+.4f} over 35M "
        f"({d5/s5:+.1f} sem)")
    g5 = True
    GG.verdict(rho < 0.9, emit=say, if_true=(
        f"the encoders disagree case by case at rho {rho:+.4f}, so there is something a larger "
        f"model could be reading differently even if the mean gain is small."), if_false=(
        f"the encoders agree at rho {rho:+.4f} and the oracle over the pair is only {d5:+.4f} above "
        f"35M -- a bigger encoder is reading the same thing on these proteins, and the full run "
        f"would not change that."))
    say(f"     G5 {'PASS' if g5 else 'FAIL'}")

    say()
    say("G6 WHAT THIS CANNOT SHOW")
    say(f"     n={int(ok.sum())} cannot resolve an effect below {mde:.4f}.")
    say("     Fewer training proteins per fold degrades k-NN for EVERY arm, so the absolute numbers")
    say("     here sit below the 2,178-protein run and only the PAIRED differences are comparable.")
    say("     A screen that fails to see an effect has not shown that there is none.")
    g6 = True
    say(f"     G6 {'PASS' if g6 else 'FAIL'}")

    gates = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6}
    man = RM.manifest(inputs=[SEQF, STRF, ESF, M650], available=len(accs), used=int(ok.sum()),
                      selection="random", seed=SEED,
                      controls=["popularity re-gated on the subset rather than assumed from n=2,178",
                                "the minimum detectable effect stated BEFORE any comparison (G2)",
                                "35M re-measured on the identical 500 proteins and folds",
                                "merge weights frozen from loop 163d, nothing tuned here",
                                "the encoders' agreement measured, so a null is diagnosable"],
                      note="does ESM-2 650M beat 35M -- a screen that states its own resolution")
    out = {"test": "esm650 vs esm35 screen at n=500", "gates": gates,
           "arms": {k: mn(A[k]) for k in list(B) + ["popularity"]},
           "n": int(ok.sum()), "paired_sem": sem35, "mde_3sem": mde,
           "g3": [d3, s3], "g4": [d4, s4], "spearman": rho, "oracle_gain": [d5, s5],
           "merged": {k: float(np.nanmean(v[ok])) for k, v in merged.items()},
           "weights": list(W), "manifest": man, "seconds": time.time() - t0, "log": log}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'PASS' if v else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [{time.time()-t0:.0f}s]")
    say("=" * 104)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
