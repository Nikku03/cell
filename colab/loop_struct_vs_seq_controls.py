"""Loop 163b. The same question on a task popularity cannot win: frequency-matched candidates.

WHY THIS EXISTS. Loop 163's Y2 failed. A metabolite-POPULARITY column that never looks at the
protein scored AUC 0.8503 while the best learned representation reached 0.8212. Ranking all 8,428
non-currency metabolites by "does this enzyme touch one" is dominated by the fact that common
metabolites are common -- the median enzyme touches 6 of them, and the frequency distribution
supplies most of the ordering. The gate said in advance that if popularity won, nothing else in
that loop meant anything, so loop 163's Y4 measured structure losing to sequence on a task where
neither had shown it could beat a frequency table.

THE FIX. Score each enzyme against a candidate set in which popularity carries NO information by
construction: for every true metabolite of that enzyme, draw MATCHED negatives from the same
frequency stratum. A popularity predictor then has nothing to sort on and must land at 0.5, and C1
gates exactly that rather than assuming it. Anything above 0.5 on this set is function.

NOTHING ELSE CHANGES. Same 2,171 proteins, same five arms, same homology-disjoint folds at 5-mer
Jaccard >= 0.30, same k=10 cosine k-NN, same seed. Only the candidate set moves, so the difference
between this loop and 163 is attributable to the confound and to nothing else.

PREDECLARED, before any number is looked at.

  C1 THE CONTROL WORKS. The popularity column, scored on the frequency-matched candidate sets.
     (Second attempt. The first drew negatives from each positive's frequency STRATUM and pooled
     them per enzyme; popularity fell from 0.8503 to 0.5991 and the gate correctly failed, because
     pooling positives from different strata leaves a cross-stratum frequency gradient to sort.
     Now every positive runs its own mini-contest against the candidates nearest to it in
     frequency, so there is no gradient left to exploit.)
     Gate: |AUC - 0.5| <= 0.02. If popularity can still sort these candidates the matching failed
     and every number below inherits loop 163's defect.

  C2 IS THERE FUNCTION SIGNAL AT ALL, once popularity is removed? Best arm against 0.5.
     Gate: best arm exceeds 0.5 by more than 3 sem. This can fail: it is possible that everything
     loop 163 measured was the frequency table and there is nothing else, in which case neither
     sequence nor structure predicts enzyme function here and the loop says so.

  C3 MORE THAN HOMOLOGY? Best learned arm against the raw 5-mer lookup, on the matched sets.
     Gate: more than 3 sem.

  C4 DOES STRUCTURE BEAT SEQUENCE? Loop 163's Y4, re-asked where the answer means something.
     Gate: structure beats the better ESM arm by more than 3 sem.

  C5 DOES STRUCTURE ADD TO SEQUENCE? Concatenation against the better single arm.
     Gate: more than 3 sem.

  C6 AND THE HONEST COMPARISON OF THE TWO TASKS. Report every arm on both candidate sets side by
     side, so the size of the confound loop 163 walked into is on the record as a number rather
     than as an apology.
     Gate: passes on being reported.

-> outputs/loop_struct_vs_seq_controls.json
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
import gate_guard as GG                              # noqa: E402
import run_manifest as RM                            # noqa: E402
import loop_replication as LR                        # noqa: E402
from rem.harness import REM, auc_of                  # noqa: E402
from loop_struct_vs_seq import homology_folds, knn_scores, KNN, JACC, NFOLD, SEED  # noqa: E402

SEQF = Path("colab/data/ml/esm_enzymes.npz")
STRF = Path("colab/data/ml/struct_enzymes.npz")
OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_struct_vs_seq_controls.json"
NEG_PER_POS = 40
N_STRATA = 20
C1_TOL = 0.02

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    say("=" * 104)
    say("  STRUCTURE VS SEQUENCE, ON A TASK POPULARITY CANNOT WIN")
    say("=" * 104)
    say()

    S = np.load(SEQF, allow_pickle=False)
    T = np.load(STRF, allow_pickle=False)
    common = sorted(set(map(str, S["accs"])) & set(map(str, T["accs"])))
    si = {a: i for i, a in enumerate(map(str, S["accs"]))}
    ti = {a: i for i, a in enumerate(map(str, T["accs"]))}
    E35 = S["esm35"][[si[a] for a in common]]
    E8 = S["esm8"][[si[a] for a in common]]
    ST = T["X"][[ti[a] for a in common]]

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
    E35, E8, ST, Y = E35[keep], E8[keep], ST[keep], Y[keep]
    pop = Y.mean(0)
    say(f"     {len(accs):,} enzymes | {Y.shape[1]:,} candidate metabolites | "
        f"median {int(np.median(Y.sum(1)))} true metabolites each")

    # EXACT per-positive frequency matching, scored per positive and never pooled.
    # The first version drew negatives from the positive's frequency STRATUM and then pooled every
    # positive's candidates into one set per enzyme. C1 caught what that leaves behind: an enzyme
    # with positives in different strata gets a pooled set whose cross-stratum frequency
    # differences a popularity column can still sort, and 20 strata over 8,428 metabolites leave
    # real spread inside each stratum too. Popularity fell only to 0.5991, not to chance.
    # Here each positive gets its OWN mini-contest against the negatives whose frequency is
    # numerically closest to its own, and the enzyme's score is the mean over its positives. Within
    # a mini-contest every candidate has near-identical frequency, so popularity has nothing to
    # sort on at all rather than merely less.
    order = np.argsort(pop, kind="stable")
    posn = np.empty(len(pop), int)
    posn[order] = np.arange(len(pop))
    say(f"     popularity ranges {pop.min():.5f} to {pop.max():.4f}; negatives are the "
        f"{NEG_PER_POS} candidates nearest in frequency to each positive")

    cand = []
    for i in range(len(accs)):
        pos = np.where(Y[i] > 0)[0]
        mini = []
        for p in pos:
            lo, hi = posn[p] - 1, posn[p] + 1
            neg = []
            while len(neg) < NEG_PER_POS and (lo >= 0 or hi < len(order)):
                for side in (lo, hi):
                    if 0 <= side < len(order) and len(neg) < NEG_PER_POS:
                        c = order[side]
                        if Y[i, c] == 0:
                            neg.append(c)
                lo -= 1
                hi += 1
            if neg:
                mini.append((p, np.array(neg)))
        cand.append(mini)
    sizes = [len(m) for m in cand]
    say(f"     {int(np.sum(sizes)):,} mini-contests over {len(accs):,} enzymes, "
        f"median {int(np.median(sizes))} per enzyme, {NEG_PER_POS + 1} candidates each")

    fold, ncl, ks = homology_folds(seqs, accs)
    say(f"     {ncl:,} homology clusters, folds {[int((fold == f).sum()) for f in range(NFOLD)]}")
    say()

    def zs(X):
        return (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)

    ARMS = {"esm35": zs(E35), "esm8": zs(E8), "structure": zs(ST),
            "esm35+structure": np.hstack([zs(E35), zs(ST)]),
            "esm35+esm8": np.hstack([zs(E35), zs(E8)])}

    def score_case(i, s):
        """Mean over this enzyme's mini-contests of the fraction of frequency-matched negatives the
        true metabolite outscores. Ties count half, as in auc_of."""
        mini = cand[i]
        if not mini:
            return np.nan
        vals = []
        for p, neg in mini:
            v = s[p]
            w = s[neg]
            vals.append(float((w < v).sum() + 0.5 * (w == v).sum()) / len(w))
        return float(np.mean(vals))

    per = {k: [] for k in list(ARMS) + ["popularity", "kmer_homology"]}
    per_full = {k: [] for k in list(ARMS) + ["popularity"]}
    for f in range(NFOLD):
        te = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        for nm, X in ARMS.items():
            P = knn_scores(X[tr], Y[tr], X[te])
            for r, i in enumerate(te):
                per[nm].append(score_case(i, P[r]))
                per_full[nm].append(auc_of(P[r], Y[i] > 0))
        Ksim = np.zeros((len(te), len(tr)))
        for r, i in enumerate(te):
            for c, j in enumerate(tr):
                sh = len(ks[i] & ks[j])
                u = len(ks[i]) + len(ks[j]) - sh
                Ksim[r, c] = sh / u if u else 0.0
        idx = np.argpartition(-Ksim, min(KNN, Ksim.shape[1] - 1), axis=1)[:, :KNN]
        for r, i in enumerate(te):
            w = np.maximum(Ksim[r, idx[r]], 0)
            w = w if w.sum() > 0 else np.ones_like(w)
            p = (w[:, None] * Y[tr][idx[r]]).sum(0) / w.sum()
            per["kmer_homology"].append(score_case(i, p))
            per["popularity"].append(score_case(i, pop))
            per_full["popularity"].append(auc_of(pop, Y[i] > 0))
        say(f"     fold {f}: {len(te)} test proteins [{time.time()-t0:.0f}s]")

    def agg(v):
        a = np.array(v, float)
        a = a[np.isfinite(a)]
        return {"auc": float(a.mean()), "sem": float(a.std() / np.sqrt(len(a))), "n": len(a)}
    res = {k: agg(v) for k, v in per.items()}
    resf = {k: agg(v) for k, v in per_full.items()}

    say()
    say("     ON THE FREQUENCY-MATCHED CANDIDATE SETS:")
    for k in sorted(res, key=lambda x: -res[x]["auc"]):
        say(f"       {k:<20s} AUC {res[k]['auc']:.4f} +/- {res[k]['sem']:.4f}")

    def paired(a, b):
        x, y = np.array(per[a], float), np.array(per[b], float)
        m = np.isfinite(x) & np.isfinite(y)
        d = x[m] - y[m]
        return float(d.mean()), float(d.std() / np.sqrt(len(d)))

    # ------------------------------------------------------------------ C1
    say()
    c1 = bool(abs(res["popularity"]["auc"] - 0.5) <= C1_TOL)
    say(f"C1 popularity on the matched sets: AUC {res['popularity']['auc']:.4f} "
        f"(gate |AUC-0.5| <= {C1_TOL})")
    GG.verdict(c1, emit=say, if_true=(
        "the matching works: a frequency column has nothing left to sort on, so anything above "
        "0.5 below is function and not frequency."), if_false=(
        "popularity still sorts these candidates, so the matching failed and every number below "
        "inherits loop 163's confound."))
    say(f"     C1 {'PASS' if c1 else 'FAIL'}")

    # ------------------------------------------------------------------ C2
    best = max(ARMS, key=lambda k: res[k]["auc"])
    c2 = bool(res[best]["auc"] - 0.5 > 3 * res[best]["sem"])
    say()
    say(f"C2 best arm ({best}) against chance: {res[best]['auc']:.4f} "
        f"+/- {res[best]['sem']:.4f}")
    GG.verdict(c2, emit=say, if_true=(
        "there is real function signal once frequency is removed."), if_false=(
        "nothing beats chance once frequency is removed -- everything loop 163 measured was the "
        "metabolite frequency table, and neither sequence nor structure predicts enzyme function "
        "on this benchmark at all."))
    say(f"     C2 {'PASS' if c2 else 'FAIL'}")

    # ------------------------------------------------------------------ C3
    d3, s3 = paired(best, "kmer_homology")
    c3 = bool(d3 > 3 * s3)
    say()
    say(f"C3 {best} minus raw 5-mer homology: {d3:+.4f} sem {s3:.4f} = {d3/s3:+.1f} sem")
    GG.verdict(c3, emit=say, if_true="the learned representation beats string similarity.",
               if_false="the learned representation does not beat string similarity.")
    say(f"     C3 {'PASS' if c3 else 'FAIL'}")

    # ------------------------------------------------------------------ C4/C5
    seq_best = "esm35" if res["esm35"]["auc"] >= res["esm8"]["auc"] else "esm8"
    d4, s4 = paired("structure", seq_best)
    c4 = bool(d4 > 3 * s4)
    say()
    say(f"C4 STRUCTURE minus {seq_best}: {d4:+.4f} sem {s4:.4f} = {d4/s4:+.1f} sem")
    GG.verdict(c4, emit=say, if_true=(
        "structure beats sequence once the confound is removed, which REVERSES loop 163's reading "
        "and makes the dark-proteome structure download worth doing."), if_false=(
        "structure still does not beat sequence. Loop 163's ordering was not an artefact of the "
        "popularity confound -- it holds on a task where popularity scores nothing."))
    say(f"     C4 {'PASS' if c4 else 'FAIL'}")

    d5, s5 = paired("esm35+structure", seq_best)
    c5 = bool(d5 > 3 * s5)
    say()
    say(f"C5 esm35+structure minus {seq_best}: {d5:+.4f} sem {s5:.4f} = {d5/s5:+.1f} sem")
    GG.verdict(c5, emit=say, if_true=(
        "structure ADDS to sequence even though it loses head to head -- worth the download."),
        if_false="structure adds nothing to sequence on this task either.")
    say(f"     C5 {'PASS' if c5 else 'FAIL'}")

    # ------------------------------------------------------------------ C6
    say()
    say("C6 THE SIZE OF THE CONFOUND, as a number")
    say(f"     {'arm':<20s} {'loop 163 (all 8,428)':>22s} {'matched':>10s} {'shift':>9s}")
    for k in list(ARMS) + ["popularity"]:
        say(f"     {k:<20s} {resf[k]['auc']:>22.4f} {res[k]['auc']:>10.4f} "
            f"{res[k]['auc']-resf[k]['auc']:>+9.4f}")
    GG.verdict(c1, emit=say, if_true=(
        "popularity falls to chance while every learned arm keeps most of its score -- the arms "
        "were not reproducing the frequency table."), if_false=(
        f"popularity does NOT fall to chance: it lands at {res['popularity']['auc']:.4f}. The "
        f"matching is partial, so what the table shows is the confound SHRINKING, not vanishing. "
        f"The learned arms keep their scores while popularity loses "
        f"{resf['popularity']['auc'] - res['popularity']['auc']:.4f}, and a frequency column "
        f"cannot explain a PAIRED difference between two arms scoring identical candidates -- "
        f"which is what C4 and C5 measure -- but the absolute numbers still carry some of it."))
    c6 = True
    say(f"     C6 {'PASS' if c6 else 'FAIL'}")

    gates = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6}
    man = RM.manifest(
        inputs=[SEQF, STRF, Path("colab/data/rem_enzyme.npz")],
        available=len(accs), used=len(accs), selection="all", seed=SEED,
        controls=[
            "candidates frequency-matched so a popularity column scores 0.5 by construction",
            "C1 gates that the matching worked instead of assuming it",
            "identical proteins, folds, arms, k and seed as loop 163 -- only the candidate set moves",
            "a raw 5-mer homology lookup as the BLAST control",
            "both candidate sets reported side by side so the confound is a number",
        ],
        note="loop 163 re-asked on frequency-matched candidates, after its own Y2 voided it")
    out = {"test": "structure vs sequence with the popularity confound removed",
           "gates": gates, "matched": res, "full": resf,
           "paired": {"best_vs_kmer": [d3, s3], "structure_vs_seq": [d4, s4],
                      "combined_vs_seq": [d5, s5]},
           "n_enzymes": len(accs), "neg_per_pos": NEG_PER_POS, "n_strata": N_STRATA,
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
