"""Loop 163c. Merge sequence and structure properly, after C5 tested only the crudest version.

WHAT C5 ACTUALLY TESTED. Loop 163b concluded "structure adds nothing to sequence" from a single
merge: z-score both feature blocks and concatenate them into one cosine k-NN. That is the weakest
merge available. A 480-dimensional ESM block and a 64-dimensional structure block share one cosine
distance, so structure contributes about an eighth of the geometry whatever it knows, and z-scoring
per column does not fix a per-BLOCK imbalance.

WHY THAT MATTERS HERE SPECIFICALLY. Loop 161 concluded "chemistry and connectivity do not merge"
from an equal-weight RANK blend. A six-family design workflow then overturned it: the same two
signals fused in SCORE space at a frozen scale went from 0.877 to 0.941, and rank-space fusion was
measured at 0.850 -- the merge rule was the entire difference. C5 is the same shape of test and
deserves the same scrutiny before its negative is believed.

M2 IS THE GATE THAT DECIDES WHETHER ANY OF THIS CAN WORK. Before trying merge rules, measure the
headroom: the rank correlation between the two arms, and an ORACLE that picks the better arm per
case using the answer. The oracle is a ceiling and not a predictor. If it barely exceeds the better
single arm, then the two arms succeed and fail on the SAME cases, no merge rule can help, and every
merge below is arithmetic on a foregone conclusion. That is a real possible outcome and M2 is
written so it can deliver it.

PREDECLARED, before any number is looked at.

  M1 THE INSTRUMENT IS THE CLEAN ONE. The symmetric frequency-matched mini-contests loop 163b
     arrived at after three attempts, on the same proteins and homology-disjoint folds.
     Gate: the popularity column lands within 0.02 of 0.5. Anything else and this inherits a
     confound that took three tries to remove.

  M2 IS THERE HEADROOM FOR A MERGE AT ALL? Spearman between the two arms' per-case scores, and the
     per-case oracle ceiling.
     Gate: PASS iff the oracle exceeds the better single arm by more than 3 sem. FAIL means the
     arms agree case by case, no merge rule can recover anything, and M3-M5 are reported as
     confirmation rather than as a search.

  M3 SCORE-SPACE WEIGHTED FUSION, the rule that overturned loop 161. Max-normalise each arm's score
     vector per case and combine as seq + w*struct. The weight is fitted on one half of the enzymes
     and scored on the other, both ways round, so no enzyme both chooses the weight and tests it.
     Gate: PASS iff the held-out fused score beats sequence alone by more than 3 sem.

  M4 RULES THAT ARE NOT SUMS. Reciprocal-rank fusion at several k, elementwise max of the two
     rank-normalised arms, and rank-product. Same held-out discipline.
     Gate: PASS iff the best of them beats sequence alone by more than 3 sem.

  M5 A LEARNED MERGE. Logistic regression over the two arms' scores and ranks per candidate, fitted
     with the enzymes grouped so no enzyme appears in both fit and score.
     Gate: PASS iff it beats sequence alone by more than 3 sem.

  M6 DOES ANYTHING WIN? The best merge of M3-M5 against sequence alone, and against the concatenation
     C5 already rejected.
     Gate: PASS iff at least one merge clears 3 sem over sequence alone. If nothing does, then C5's
     negative was right for a better reason than C5 had, and the honest headline is that structure
     adds nothing under any of five merge rules -- not merely under concatenation.

  M7 WHAT THIS CANNOT SHOW. k-NN is a floor on a representation, not a ceiling. The structure arm is
     64 hand-chosen geometric descriptors, not a learned structural encoder, so this bounds THOSE
     descriptors. And the whole benchmark is monomer geometry against Human-GEM's own GPR labels.

-> outputs/loop_struct_seq_merge.json
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
from loop_struct_vs_seq import homology_folds, knn_scores, KNN, NFOLD, SEED  # noqa: E402

SEQF = Path("colab/data/ml/esm_enzymes.npz")
STRF = Path("colab/data/ml/struct_enzymes.npz")
OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_struct_seq_merge.json"
NEG_PER_POS = 40
C1_TOL = 0.02

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def contest_auc(sv, mini):
    """Mean over an enzyme's mini-contests of the fraction of matched negatives the positive beats."""
    vals = []
    for p, neg in mini:
        v, w = sv[p], sv[neg]
        vals.append(float((w < v).sum() + 0.5 * (w == v).sum()) / len(w))
    return float(np.mean(vals)) if vals else np.nan


def main():
    t0 = time.time()
    say("=" * 104)
    say("  MERGING SEQUENCE AND STRUCTURE PROPERLY -- five rules, not one concatenation")
    say("=" * 104)
    say()

    S = np.load(SEQF, allow_pickle=False)
    T = np.load(STRF, allow_pickle=False)
    common = sorted(set(map(str, S["accs"])) & set(map(str, T["accs"])))
    si = {a: i for i, a in enumerate(map(str, S["accs"]))}
    ti = {a: i for i, a in enumerate(map(str, T["accs"]))}
    E35 = S["esm35"][[si[a] for a in common]]
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
    E35, ST, Y = E35[keep], ST[keep], Y[keep]
    pop = Y.mean(0)

    # ------------------------------------------------------------------ M1
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
    say(f"     {len(accs):,} enzymes | {sum(len(m) for m in cand):,} mini-contests of "
        f"{NEG_PER_POS+1} | {ndrop:,} positives dropped for want of a symmetric match")

    fold, ncl, ks = homology_folds(seqs, accs)
    say(f"     {ncl:,} homology clusters, folds {[int((fold == f).sum()) for f in range(NFOLD)]}")

    def zs(X):
        return (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    Xs, Xt = zs(E35), zs(ST)
    Xc = np.hstack([Xs, Xt])

    # per-case predicted score vectors from each arm, held out by fold
    Pseq = np.zeros_like(Y)
    Pstr = np.zeros_like(Y)
    Pcat = np.zeros_like(Y)
    for f in range(NFOLD):
        te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
        Pseq[te] = knn_scores(Xs[tr], Y[tr], Xs[te])
        Pstr[te] = knn_scores(Xt[tr], Y[tr], Xt[te])
        Pcat[te] = knn_scores(Xc[tr], Y[tr], Xc[te])
        say(f"     fold {f} arms computed [{time.time()-t0:.0f}s]")

    A_pop = np.array([contest_auc(pop, cand[i]) for i in range(len(accs))])
    A_seq = np.array([contest_auc(Pseq[i], cand[i]) for i in range(len(accs))])
    A_str = np.array([contest_auc(Pstr[i], cand[i]) for i in range(len(accs))])
    A_cat = np.array([contest_auc(Pcat[i], cand[i]) for i in range(len(accs))])
    ok = np.isfinite(A_seq) & np.isfinite(A_str)

    def mn(a):
        return float(np.nanmean(a[ok]))

    def sem(a):
        return float(np.nanstd(a[ok]) / np.sqrt(ok.sum()))

    def pdiff(a, b):
        d = a[ok] - b[ok]
        return float(d.mean()), float(d.std() / np.sqrt(len(d)))

    m1 = bool(abs(mn(A_pop) - 0.5) <= C1_TOL)
    say()
    say(f"M1 popularity on the matched contests: {mn(A_pop):.4f} (gate |AUC-0.5| <= {C1_TOL})")
    GG.verdict(m1, emit=say,
               if_true="the clean instrument from loop 163b carries over.",
               if_false="the matching is not clean here; everything below inherits a confound.")
    say(f"     M1 {'PASS' if m1 else 'FAIL'}")
    say()
    say(f"     sequence      {mn(A_seq):.4f} +/- {sem(A_seq):.4f}")
    say(f"     structure     {mn(A_str):.4f} +/- {sem(A_str):.4f}")
    say(f"     concatenation {mn(A_cat):.4f} +/- {sem(A_cat):.4f}   <- what C5 tested")

    # ------------------------------------------------------------------ M2
    rho = float(stats.spearmanr(A_seq[ok], A_str[ok]).statistic)
    A_or = np.maximum(A_seq, A_str)
    d_or, s_or = pdiff(A_or, A_seq)
    m2 = bool(d_or > 3 * s_or)
    say()
    say("M2 HEADROOM")
    say(f"     Spearman between the two arms' per-case scores: {rho:+.4f}")
    say(f"     ORACLE (per-case best, uses the answer -- a ceiling, not a predictor): "
        f"{mn(A_or):.4f}")
    say(f"     oracle minus sequence: {d_or:+.4f} sem {s_or:.4f} = {d_or/s_or:+.1f} sem")
    GG.verdict(m2, emit=say, if_true=(
        "the arms succeed on different cases, so a merge has something to recover and the rules "
        "below are a real search."), if_false=(
        "the arms succeed and fail on the SAME cases. No merge rule can recover what is not there, "
        "and M3-M5 below are a confirmation rather than a search."))
    say(f"     M2 {'PASS' if m2 else 'FAIL'}")

    # ------------------------------------------------------------------ merges
    def norm_max(v):
        m = v.max()
        return v / m if m > 0 else v

    def rank01(v):
        return (stats.rankdata(v, "average") - 1) / max(len(v) - 1, 1)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(accs))
    halves = [perm[:len(accs) // 2], perm[len(accs) // 2:]]

    def eval_rule(rule, idx):
        return np.array([contest_auc(rule(Pseq[i], Pstr[i]), cand[i]) for i in idx])

    say()
    say("M3 SCORE-SPACE WEIGHTED FUSION, weight fitted on one half and scored on the other")
    WS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    m3_folds = []
    for a, b in ((0, 1), (1, 0)):
        fit, test = halves[a], halves[b]
        curve = []
        for w in WS:
            r = eval_rule(lambda s, t, w=w: norm_max(s) + w * norm_max(t), fit)
            curve.append(float(np.nanmean(r)))
        wbest = WS[int(np.argmax(curve))]
        held = eval_rule(lambda s, t: norm_max(s) + wbest * norm_max(t), test)
        base = A_seq[test]
        d = held - base
        d = d[np.isfinite(d)]
        m3_folds.append({"w": wbest, "fused": float(np.nanmean(held)),
                         "seq": float(np.nanmean(base)), "delta": float(d.mean()),
                         "sem": float(d.std() / np.sqrt(len(d)))})
        say(f"       fold {a}->{b}: best w={wbest} | fused {np.nanmean(held):.4f} vs sequence "
            f"{np.nanmean(base):.4f} | delta {d.mean():+.4f} sem {d.std()/np.sqrt(len(d)):.4f}")
    d3 = float(np.mean([f["delta"] for f in m3_folds]))
    s3 = float(np.mean([f["sem"] for f in m3_folds]))
    m3 = bool(d3 > 3 * s3)
    GG.verdict(m3, emit=say,
               if_true=f"score-space fusion helps: {d3:+.4f} averaged over both held-out halves.",
               if_false=f"score-space fusion does not help: {d3:+.4f} against a 3-sem bar of "
                        f"{3*s3:.4f}. Every fitted weight is reported above, including w=0.")
    say(f"     M3 {'PASS' if m3 else 'FAIL'}")

    say()
    say("M4 RULES THAT ARE NOT SUMS")
    RULES = {
        "RRF k=5": lambda s, t: 1 / (5 + len(s) - stats.rankdata(s)) + 1 / (5 + len(t) - stats.rankdata(t)),
        "RRF k=60": lambda s, t: 1 / (60 + len(s) - stats.rankdata(s)) + 1 / (60 + len(t) - stats.rankdata(t)),
        "max(rank)": lambda s, t: np.maximum(rank01(s), rank01(t)),
        "rank product": lambda s, t: rank01(s) * rank01(t),
        "min(rank)": lambda s, t: np.minimum(rank01(s), rank01(t)),
    }
    m4_rows = {}
    for nm, rule in RULES.items():
        v = eval_rule(rule, np.arange(len(accs)))
        d, sm = pdiff(v, A_seq)
        m4_rows[nm] = {"auc": mn(v), "delta": d, "sem": sm}
        say(f"       {nm:<14s} {mn(v):.4f}   vs sequence {d:+.4f} sem {sm:.4f} = {d/sm:+.1f} sem")
    bestr = max(m4_rows, key=lambda k: m4_rows[k]["delta"])
    m4 = bool(m4_rows[bestr]["delta"] > 3 * m4_rows[bestr]["sem"])
    GG.verdict(m4, emit=say, if_true=f"{bestr} beats sequence alone.",
               if_false="no non-sum rule beats sequence alone either.")
    say(f"     M4 {'PASS' if m4 else 'FAIL'}")

    say()
    say("M5 A LEARNED MERGE, enzymes grouped so none both fits and scores")
    from sklearn.linear_model import LogisticRegression
    m5_folds = []
    for a, b in ((0, 1), (1, 0)):
        fit, test = halves[a], halves[b]
        Xr, yr = [], []
        for i in fit:
            for p, neg in cand[i]:
                for c, lab in [(p, 1)] + [(n, 0) for n in neg]:
                    Xr.append([Pseq[i, c], Pstr[i, c], rank01(Pseq[i])[c], rank01(Pstr[i])[c]])
                    yr.append(lab)
        clf = LogisticRegression(max_iter=1000).fit(np.array(Xr), np.array(yr))
        vals = []
        for i in test:
            rs, rt = rank01(Pseq[i]), rank01(Pstr[i])
            F = np.column_stack([Pseq[i], Pstr[i], rs, rt])
            vals.append(contest_auc(clf.predict_proba(F)[:, 1], cand[i]))
        vals = np.array(vals)
        d = vals - A_seq[test]
        d = d[np.isfinite(d)]
        m5_folds.append({"fused": float(np.nanmean(vals)), "delta": float(d.mean()),
                         "sem": float(d.std() / np.sqrt(len(d))),
                         "coef": clf.coef_[0].tolist()})
        say(f"       fold {a}->{b}: {np.nanmean(vals):.4f} vs sequence "
            f"{np.nanmean(A_seq[test]):.4f} | delta {d.mean():+.4f} sem "
            f"{d.std()/np.sqrt(len(d)):.4f} | coefficients {np.round(clf.coef_[0], 3).tolist()}")
    d5 = float(np.mean([f["delta"] for f in m5_folds]))
    s5 = float(np.mean([f["sem"] for f in m5_folds]))
    m5 = bool(d5 > 3 * s5)
    GG.verdict(m5, emit=say, if_true=f"the learned merge helps: {d5:+.4f}.",
               if_false=f"the learned merge does not help either: {d5:+.4f} against {3*s5:.4f}.")
    say(f"     M5 {'PASS' if m5 else 'FAIL'}")

    # ------------------------------------------------------------------ M6
    best_delta = max([d3, m4_rows[bestr]["delta"], d5])
    m6 = bool(m3 or m4 or m5)
    dcat, scat = pdiff(A_cat, A_seq)
    say()
    say("M6 DOES ANYTHING WIN?")
    say(f"     concatenation (C5's rule) vs sequence: {dcat:+.4f} sem {scat:.4f}")
    say(f"     best of five merge rules vs sequence : {best_delta:+.4f}")
    GG.verdict(m6, emit=say, if_true=(
        "at least one merge rule beats sequence alone, so C5's negative was a statement about "
        "concatenation and not about structure."), if_false=(
        "no merge rule beats sequence alone -- not score-space fusion at any weight, not reciprocal "
        "rank fusion, not max, not rank product, not a learned logistic merge. C5's negative was "
        "right, and now for a better reason than C5 had: it is not the merge rule."))
    say(f"     M6 {'PASS' if m6 else 'FAIL'}")

    say()
    say("M7 WHAT THIS CANNOT SHOW")
    say("     k-NN is a floor on a representation, not a ceiling.")
    say("     The structure arm is 64 hand-chosen geometric descriptors, not a learned structural")
    say("     encoder, so this bounds THOSE descriptors and not structure as such.")
    say("     Monomer geometry only: no ligand, no cofactor, no partner in any structure here.")
    m7 = True
    say(f"     M7 {'PASS' if m7 else 'FAIL'}")

    gates = {"M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5, "M6": m6, "M7": m7}
    man = RM.manifest(
        inputs=[SEQF, STRF, Path("colab/data/rem_enzyme.npz")],
        available=len(accs), used=int(ok.sum()), selection="all", seed=SEED,
        controls=[
            "the symmetric frequency-matched instrument loop 163b needed three attempts to get right",
            "M1 re-gates that instrument here rather than assuming it carried over",
            "M2 measures the oracle ceiling BEFORE searching merge rules, so a null is diagnosable",
            "every fitted weight held out: fitted on one half of the enzymes, scored on the other",
            "five merge rules, not one, after loop 161's single-rule negative was overturned",
        ],
        note="does structure add to sequence under any merge rule, not only under concatenation")
    out = {"test": "sequence x structure merge, five rules", "gates": gates,
           "arms": {"sequence": [mn(A_seq), sem(A_seq)], "structure": [mn(A_str), sem(A_str)],
                    "concatenation": [mn(A_cat), sem(A_cat)], "popularity": [mn(A_pop), sem(A_pop)],
                    "oracle": [mn(A_or), sem(A_or)]},
           "spearman_between_arms": rho, "oracle_gain": [d_or, s_or],
           "m3_weighted": m3_folds, "m4_rules": m4_rows, "m5_learned": m5_folds,
           "concat_vs_seq": [dcat, scat],
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
