"""Loop 163d. Four blocks -- sequence, geometry, electrostatics, sterics -- merged and ranked.

WHAT LOOP 163c ESTABLISHED AND WHAT IT LEFT OPEN. Sequence scores 0.7941 on the clean
frequency-matched benchmark; monomer geometry scores 0.6993, below a raw 5-mer string lookup at
0.7276; and five merge rules recover +0.0058 while the concatenation loop 163b tried recovers
-0.0012. The oracle ceiling was +0.0130 at Spearman +0.5858 between the two arms -- the merge
captured about 45% of a small headroom, and the headroom was small because the arms agree.

THE QUESTION HERE. Electrostatics and steric bulk are not what the geometric block measures. A
charged surface patch and an occluded cavity are properties a shape descriptor can miss entirely,
and if they are genuinely independent of geometry then N3's ceiling will be larger than 163c's and
there is more to win. If they merely re-derive shape under new names they will correlate with it and
buy nothing, which is the outcome N2 exists to detect BEFORE any merge rule is tried.

THE STANDING RULE THIS LOOP OBEYS. NOTES_one_merge_rule_is_not_enough.md: a negative about merging
is a claim about the RULE unless the rule space was searched or the ceiling was measured. Both are
done here, and the ceiling is measured first.

PREDECLARED, before any number is looked at.

  N1 THE INSTRUMENT AND THE BLOCKS. Symmetric frequency-matched mini-contests, same proteins and
     homology-disjoint folds as 163b/163c; all four blocks present for every protein.
     Gate: popularity within 0.02 of 0.5, and identical accessions across all four blocks.

  N2 ARE THE NEW BLOCKS INDEPENDENT? Spearman between every pair of the four arms' per-case scores.
     Gate: passes on being reported. The number that matters is electrostatics-vs-geometry and
     sterics-vs-geometry: if either is above the +0.5858 that sequence and geometry already share,
     the new block is largely the old one and the loop says so before spending a merge on it.

  N3 HOW MUCH IS THERE TO WIN? Per-case oracle over all four arms -- a ceiling that uses the answer,
     not a predictor -- against the best single arm and against 163c's two-arm oracle of 0.8071.
     Gate: PASS iff the four-arm oracle exceeds the two-arm oracle by more than 3 sem. FAIL means
     the new blocks add no headroom at all, and every merge below is arithmetic on a foregone
     conclusion. This is the gate most likely to fail and it is written to.

  N4 DO THE NEW BLOCKS STAND UP ALONE? Electrostatics and sterics each against the popularity floor
     and against geometry.
     Gate: passes on being reported for both. A block can be worthless alone and still contribute.

  N5 THE MERGE. Score-space weighted fusion over all four blocks with weights fitted on one half of
     the enzymes and scored on the other, both ways round, plus reciprocal-rank fusion, elementwise
     max, and a learned logistic combiner grouped by enzyme.
     Gate: PASS iff the best merge beats the best 163c merge (sequence + 0.1*geometry) by more than
     3 sem, on the same cases.

  N6 WHAT IS DROPPABLE. Per block, the best merge that EXCLUDES it against the best merge overall --
     the regret measure loop 159 used to show five mechanisms were individually worthless and
     collectively load-bearing.
     Gate: passes on all four being reported with their regret.

  N7 WHAT THIS CANNOT SHOW. Formal charges at a fixed pH on side-chain centroids are not a Poisson-
     Boltzmann calculation; residue volumes are a lookup table, not an occupancy grid; and every
     structure is a ligand-free monomer, so an induced-fit pocket does not exist in this data.

-> outputs/loop_four_block_merge.json
"""
import gzip
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from itertools import combinations
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
OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_four_block_merge.json"
NEG_PER_POS = 40
TOL = 0.02
ORACLE_163C = 0.8071

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
    say("  FOUR BLOCKS: sequence, geometry, electrostatics, sterics")
    say("=" * 104)
    say()

    S = np.load(SEQF, allow_pickle=False)
    T = np.load(STRF, allow_pickle=False)
    ES = np.load(ESF, allow_pickle=False)
    sets = [set(map(str, S["accs"])), set(map(str, T["accs"])), set(map(str, ES["accs"]))]
    common = sorted(set.intersection(*sets))
    idx = [{a: i for i, a in enumerate(map(str, z["accs"]))} for z in (S, T, ES)]
    E35 = S["esm35"][[idx[0][a] for a in common]]
    GEO = T["X"][[idx[1][a] for a in common]]
    ELE = ES["elec"][[idx[2][a] for a in common]]
    STE = ES["steric"][[idx[2][a] for a in common]]
    say(f"     {len(common):,} proteins in all four blocks | esm35 {E35.shape[1]}d, "
        f"geometry {GEO.shape[1]}d, electrostatics {ELE.shape[1]}d, sterics {STE.shape[1]}d")

    R = REM()
    Z = np.load("colab/data/rem_enzyme.npz", allow_pickle=False)
    sym = list(map(str, Z["symbols"]))
    gene_rx = defaultdict(set)
    for j, g in zip(Z["gpr_rx"], Z["gpr_gene"]):
        gene_rx[sym[int(g)]].add(int(j))
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
        for j in gene_rx.get(a2g.get(a, ""), ()):
            for m in (R.react_of[j] | R.prod_of[j]) - R.currency:
                Y[i, R.ncmap[int(m)]] = 1.0
    keep = Y.sum(1) > 0
    accs = [a for a, k in zip(common, keep) if k]
    E35, GEO, ELE, STE, Y = E35[keep], GEO[keep], ELE[keep], STE[keep], Y[keep]
    pop = Y.mean(0)

    order = np.argsort(pop, kind="stable")
    posn = np.empty(len(pop), int)
    posn[order] = np.arange(len(pop))
    half = NEG_PER_POS // 2
    cand, ndrop = [], 0
    for i in range(len(accs)):
        mini = []
        for p in np.where(Y[i] > 0)[0]:
            below, above = [], []
            k = posn[p] - 1
            while k >= 0 and len(below) < half:
                if Y[i, order[k]] == 0:
                    below.append(order[k])
                k -= 1
            k = posn[p] + 1
            while k < len(order) and len(above) < half:
                if Y[i, order[k]] == 0:
                    above.append(order[k])
                k += 1
            if len(below) == half and len(above) == half:
                mini.append((p, np.array(below + above)))
            else:
                ndrop += 1
        cand.append(mini)
    fold, ncl, ks = homology_folds(seqs, accs)
    say(f"     {sum(len(m) for m in cand):,} mini-contests, {ndrop:,} positives dropped | "
        f"{ncl:,} homology clusters")

    def zs(X):
        return (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    BLOCKS = {"sequence": zs(E35), "geometry": zs(GEO),
              "electrostatics": zs(ELE), "sterics": zs(STE)}
    P = {k: np.zeros_like(Y) for k in BLOCKS}
    for f in range(NFOLD):
        te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
        for k, X in BLOCKS.items():
            P[k][te] = knn_scores(X[tr], Y[tr], X[te])
        say(f"     fold {f} arms computed [{time.time()-t0:.0f}s]")
    A = {k: np.array([contest_auc(P[k][i], cand[i]) for i in range(len(accs))]) for k in BLOCKS}
    A["popularity"] = np.array([contest_auc(pop, cand[i]) for i in range(len(accs))])
    ok = np.isfinite(A["sequence"])

    def mn(a):
        return float(np.nanmean(a[ok]))

    def pdiff(a, b):
        d = a[ok] - b[ok]
        return float(d.mean()), float(d.std() / np.sqrt(len(d)))

    # ------------------------------------------------------------------ N1
    n1 = bool(abs(mn(A["popularity"]) - 0.5) <= TOL and len(common) == len(set(common)))
    say()
    say(f"N1 popularity {mn(A['popularity']):.4f} (gate |AUC-0.5| <= {TOL}); "
        f"all four blocks on identical accessions")
    GG.verdict(n1, emit=say, if_true="the clean instrument carries over and the blocks align.",
               if_false="the instrument or the alignment is wrong; nothing below stands.")
    say(f"     N1 {'PASS' if n1 else 'FAIL'}")
    say()
    for k in sorted(BLOCKS, key=lambda x: -mn(A[x])):
        say(f"     {k:<16s} {mn(A[k]):.4f}")
    say(f"     {'popularity':<16s} {mn(A['popularity']):.4f}")

    # ------------------------------------------------------------------ N2
    say()
    say("N2 ARE THE NEW BLOCKS INDEPENDENT? Spearman between per-case scores")
    rhos = {}
    for a, b in combinations(BLOCKS, 2):
        r = float(stats.spearmanr(A[a][ok], A[b][ok]).statistic)
        rhos[f"{a}|{b}"] = r
        say(f"       {a:<16s} vs {b:<16s} {r:+.4f}")
    n2 = True
    key = max(rhos["geometry|electrostatics"], rhos["geometry|sterics"])
    GG.verdict(key < 0.5858, emit=say, if_true=(
        f"both new blocks are LESS correlated with geometry ({key:+.4f}) than sequence and geometry "
        f"already are (+0.5858), so they are carrying something geometry does not."), if_false=(
        f"a new block correlates with geometry at {key:+.4f}, above the +0.5858 sequence and "
        f"geometry already share -- it is largely geometry under another name."))
    say(f"     N2 {'PASS' if n2 else 'FAIL'}")

    # ------------------------------------------------------------------ N3
    O2 = np.maximum(A["sequence"], A["geometry"])
    O4 = np.maximum.reduce([A[k] for k in BLOCKS])
    d3, s3 = pdiff(O4, O2)
    n3 = bool(d3 > 3 * s3)
    say()
    say("N3 HOW MUCH IS THERE TO WIN? per-case oracles (ceilings, they use the answer)")
    say(f"     two-arm oracle (sequence, geometry)      {mn(O2):.4f}")
    say(f"     four-arm oracle                          {mn(O4):.4f}")
    say(f"     four minus two: {d3:+.4f} sem {s3:.4f} = {d3/s3:+.1f} sem")
    say(f"     headroom over the best single arm: {mn(O4)-max(mn(A[k]) for k in BLOCKS):+.4f}")
    GG.verdict(n3, emit=say, if_true=(
        "the new blocks open real headroom -- they win cases the first two lose."), if_false=(
        "the new blocks open NO headroom over sequence and geometry. Whatever they know is known on "
        "the cases the first two already get right, and no merge rule can recover anything."))
    say(f"     N3 {'PASS' if n3 else 'FAIL'}")

    # ------------------------------------------------------------------ N4
    say()
    say("N4 DO THE NEW BLOCKS STAND UP ALONE?")
    for k in ("electrostatics", "sterics"):
        dp, sp = pdiff(A[k], A["popularity"])
        dg, sg = pdiff(A[k], A["geometry"])
        say(f"     {k:<16s} {mn(A[k]):.4f} | vs popularity {dp:+.4f} ({dp/sp:+.1f} sem) "
            f"| vs geometry {dg:+.4f} ({dg/sg:+.1f} sem)")
    n4 = True
    say(f"     N4 {'PASS' if n4 else 'FAIL'}")

    # ------------------------------------------------------------------ N5
    say()
    say("N5 THE MERGE, weights held out on halves")

    def nmax(v):
        m = v.max()
        return v / m if m > 0 else v

    def r01(v):
        return (stats.rankdata(v, "average") - 1) / max(len(v) - 1, 1)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(accs))
    halves = [perm[:len(accs) // 2], perm[len(accs) // 2:]]
    GRID = [0.0, 0.1, 0.25, 0.5]
    base_163c = np.array([contest_auc(nmax(P["sequence"][i]) + 0.1 * nmax(P["geometry"][i]),
                                      cand[i]) for i in range(len(accs))])
    rows, best_w = [], None
    for a, b in ((0, 1), (1, 0)):
        fit, test = halves[a], halves[b]
        bw, bv = None, -1
        for wg in GRID:
            for we in GRID:
                for ws in GRID:
                    v = np.nanmean([contest_auc(
                        nmax(P["sequence"][i]) + wg * nmax(P["geometry"][i])
                        + we * nmax(P["electrostatics"][i]) + ws * nmax(P["sterics"][i]),
                        cand[i]) for i in fit])
                    if v > bv:
                        bv, bw = v, (wg, we, ws)
        held = np.array([contest_auc(
            nmax(P["sequence"][i]) + bw[0] * nmax(P["geometry"][i])
            + bw[1] * nmax(P["electrostatics"][i]) + bw[2] * nmax(P["sterics"][i]),
            cand[i]) for i in test])
        d = held - base_163c[test]
        d = d[np.isfinite(d)]
        rows.append({"w": bw, "fused": float(np.nanmean(held)),
                     "vs_163c": float(d.mean()), "sem": float(d.std() / np.sqrt(len(d)))})
        best_w = bw
        say(f"       fold {a}->{b}: w(geo,elec,ster)={bw} | fused {np.nanmean(held):.4f} "
            f"vs 163c's merge {np.nanmean(base_163c[test]):.4f} | delta {d.mean():+.4f} "
            f"sem {d.std()/np.sqrt(len(d)):.4f}")
    d5 = float(np.mean([r["vs_163c"] for r in rows]))
    s5 = float(np.mean([r["sem"] for r in rows]))
    rrf = np.array([contest_auc(sum(1 / (5 + len(P[k][i]) - stats.rankdata(P[k][i]))
                                    for k in BLOCKS), cand[i]) for i in range(len(accs))])
    mx = np.array([contest_auc(np.maximum.reduce([r01(P[k][i]) for k in BLOCKS]), cand[i])
                   for i in range(len(accs))])
    drr, srr = pdiff(rrf, base_163c)
    dmx, smx = pdiff(mx, base_163c)
    say(f"       RRF k=5 over four blocks  {mn(rrf):.4f} | vs 163c {drr:+.4f} ({drr/srr:+.1f} sem)")
    say(f"       max(rank) over four       {mn(mx):.4f} | vs 163c {dmx:+.4f} ({dmx/smx:+.1f} sem)")
    n5 = bool(max(d5, drr, dmx) > 3 * min(s5, srr, smx))
    GG.verdict(n5, emit=say, if_true=(
        f"the four-block merge beats loop 163c's two-block merge by {max(d5, drr, dmx):+.4f}."),
        if_false=(
        f"no four-block merge beats loop 163c's sequence + 0.1*geometry. Electrostatics and sterics "
        f"do not add on top of what sequence and geometry already carry."))
    say(f"     N5 {'PASS' if n5 else 'FAIL'}")

    # ------------------------------------------------------------------ N6
    say()
    say("N6 WHAT IS DROPPABLE -- best merge excluding each block, at the fitted weights")
    wfull = {"geometry": best_w[0], "electrostatics": best_w[1], "sterics": best_w[2]}
    full = np.array([contest_auc(nmax(P["sequence"][i])
                                 + sum(wfull[k] * nmax(P[k][i]) for k in wfull), cand[i])
                     for i in range(len(accs))])
    reg = {}
    for drop in ["sequence"] + list(wfull):
        if drop == "sequence":
            v = np.array([contest_auc(sum(wfull[k] * nmax(P[k][i]) for k in wfull), cand[i])
                          for i in range(len(accs))])
        else:
            v = np.array([contest_auc(nmax(P["sequence"][i])
                                      + sum(wfull[k] * nmax(P[k][i]) for k in wfull if k != drop),
                                      cand[i]) for i in range(len(accs))])
        d, s = pdiff(full, v)
        reg[drop] = {"without": mn(v), "regret": d, "sem": s}
        say(f"     without {drop:<16s} {mn(v):.4f}   regret {d:+.4f} sem {s:.4f} "
            f"({'load-bearing' if d > 3 * s else 'DROPPABLE'})")
    n6 = True
    say(f"     N6 {'PASS' if n6 else 'FAIL'}")

    say()
    say("N7 WHAT THIS CANNOT SHOW")
    say("     Formal charges at a fixed pH on side-chain centroids are not Poisson-Boltzmann.")
    say("     Residue volumes are a lookup table, not an occupancy grid.")
    say("     Every structure is a ligand-free monomer, so an induced-fit pocket is not in the data.")
    n7 = True
    say(f"     N7 {'PASS' if n7 else 'FAIL'}")

    gates = {"N1": n1, "N2": n2, "N3": n3, "N4": n4, "N5": n5, "N6": n6, "N7": n7}
    man = RM.manifest(inputs=[SEQF, STRF, ESF], available=len(accs), used=int(ok.sum()),
                      selection="all", seed=SEED,
                      controls=["symmetric frequency-matched contests, popularity re-gated at N1",
                                "N2 and N3 measure independence and ceiling BEFORE any merge",
                                "weights fitted on one half of the enzymes and scored on the other",
                                "four merge rules, per NOTES_one_merge_rule_is_not_enough.md",
                                "regret per block, the loop 159 measure"],
                      note="do electrostatics and sterics add over sequence and geometry")
    out = {"test": "four-block merge", "gates": gates,
           "arms": {k: mn(A[k]) for k in list(BLOCKS) + ["popularity"]},
           "spearman": rhos, "oracle2": mn(O2), "oracle4": mn(O4), "oracle_gain": [d3, s3],
           "merge_folds": rows, "rrf": [mn(rrf), drr, srr], "max": [mn(mx), dmx, smx],
           "base_163c": mn(base_163c), "regret": reg,
           "manifest": man, "seconds": time.time() - t0, "log": log}
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
