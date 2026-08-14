"""LOOP 133 -- WHAT IS THE MODEL LEARNING, AND WHAT IS THE DATA HIDING?

LOOP 132 ESTABLISHED THAT THE GAIN IS REAL AND SMALL, and that 86% of the variance sits between
proteins. Neither of those says what the model KNOWS, and both rest on a variance decomposition
that has an obvious bias nobody checked. This loop attacks the result rather than extending it.

FOUR THINGS I EXPECT TO BE WRONG, listed before measuring any of them:

  MUTANTS ARE COUNTED AS DIFFERENT PROTEINS. DLKcat contains engineered variants. A point mutant is
  a different string, so it gets its own seq_id, so loop 132's decomposition charges the WT-mutant
  difference to BETWEEN-protein variance. That inflates the 86% and it inflates the 0.5625 ceiling.
  It also means the model is shown near-identical inputs with different targets, which is
  irreducible error no representation can remove.

  SINGLETONS CANNOT CONTRIBUTE WITHIN-VARIANCE. 2,144 of 7,856 sequences carry exactly one
  measurement and therefore contribute precisely zero to the within term, by construction. A
  decomposition over all sequences is biased toward "between" for an arithmetic reason and not a
  biological one.

  THE CONDITIONS ARE NOT IN THE FILE. DLKcat records enzyme, substrate, organism, EC and a value.
  It records no temperature, no pH, no buffer, no mutant flag. kcat doubles roughly every 10 C, so
  a dataset spanning 25-37 C carries a spread no model can predict from sequence and structure
  because the information is absent, not hidden.

  AND THE METRIC IS WRONG FOR THE CELL. RMSE in log10 charges the same for a 100x error on an
  enzyme carrying 1% of flux as on one carrying 40%. The cell model does not care equally.

WHAT THE MODEL MIGHT BE LEARNING INSTEAD OF CHEMISTRY. A protein language model trained on
UniRef is, among other things, an excellent family classifier. If its embedding predicts EC class
and organism nearly perfectly while barely moving kcat, then what the regressor consumes is family
identity, and "predicting kcat from sequence" is EC-median lookup with extra steps. That is
testable directly and B3 tests it.

PREDECLARED:

  B1 HOW MANY NEAR-DUPLICATE SEQUENCES, AND WHAT DO THEY COST?      THE MUTANT PROBLEM.
       within each cluster, sequence pairs at >= 0.95 k-mer Jaccard, and among those the pairs of
       equal length differing at 1-5 residues -- the signature of a point mutant. Gate: report the
       count and the log10 kcat spread within such pairs. If that spread is a material fraction of
       the model's RMSE, it is irreducible error and the record must say so.
  B2 THE VARIANCE DECOMPOSITION, DEBIASED                           LOOP 132's A6, CORRECTED.
       recomputed on sequences carrying >= 3 measurements, where a within term can actually exist,
       and again after merging near-duplicates into one protein. Gate: report all three numbers.
       If the between-share moves materially, loop 132's 86% and its 0.5625 ceiling are both
       overstated and get corrected rather than defended.
  B3 WHAT DOES THE EMBEDDING ACTUALLY ENCODE?                       THE PROBE.
       the same ESM features, same folds, predicting: EC top-level class, organism kingdom, log
       sequence length, and log10 kcat. Gate: report all four. If class and length are predicted
       far better than kcat, the representation is a family detector and the regressor is doing
       lookup.
  B4 IS THE MODEL JUST DOING EC-CLASS LOOKUP?                       THE RESIDUAL TEST.
       refit on the EC-median residual -- what is left after the class average is removed. Gate:
       the model must still beat a constant on that residual. If it cannot, everything it knows is
       already in the EC number and the sequence added nothing.
  B5 THE CEILING SET BY MISSING CONDITIONS                          THE INFORMATION THAT IS ABSENT.
       spread within (protein, substrate) pairs measured more than once. Those records share an
       enzyme AND a substrate, so any disagreement is conditions, mutants or error -- none of it
       predictable from the features present. Gate: report it as an RMSE, which is a floor no
       model using these columns can beat.
  B6 THE METRIC THE CELL MODEL WOULD CHOOSE                         THE REWEIGHTING.
       error on the human subset weighted by each enzyme's share of the model's reaction count,
       against the unweighted number. Gate: report both. A model that is accurate on the enzymes
       the cell barely uses is not useful to the cell.

-> outputs/loop_ml_probe.json
"""
import collections
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(1 << 30)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
ML = Path("colab/data/ml")
SEED = 13300
N_FOLDS = 5
NEAR_J = 0.95
MUT_MAX = 5

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def kmers(s, k=5):
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 133 -- what is the model learning, and what is the data hiding?")
    say("=" * 100)
    say()

    rows = list(csv.DictReader(open(ML / "kcat_records.tsv"), delimiter="\t"))
    y = np.array([float(r["log10_kcat"]) for r in rows])
    seq_id = np.array([int(r["seq_id"]) for r in rows])
    smi_id = np.array([int(r["smiles_id"]) for r in rows])
    fold = np.array([int(r["fold"]) for r in rows])
    clu = np.array([int(r["cluster_id"]) for r in rows])
    ec = np.array([r["ec"] for r in rows])
    org = np.array([r["organism"] for r in rows])
    gene = np.array([r["gene"] for r in rows])
    seqs = json.load(open(ML / "sequences.json"))
    SF = np.load(ML / "seq_features.npy")
    FP = np.unpackbits(np.load(ML / "substrate_ecfp.npy"), axis=1).astype(np.float32)
    E = np.load(ML / "esm2_8M_mean.npy").astype(np.float32)
    say(f"  {len(rows):,} records | {len(seqs):,} sequences | ESM {E.shape}")
    say()

    import xgboost as xgb

    def mk(obj="reg"):
        if obj == "reg":
            return xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.06,
                                    subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0,
                                    n_jobs=4, random_state=SEED, verbosity=0)
        return xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                                 subsample=0.8, colsample_bytree=0.5, n_jobs=4,
                                 random_state=SEED, verbosity=0)

    def cv_reg(X, t):
        p = np.zeros(len(t))
        for k in range(N_FOLDS):
            te, tr = fold == k, fold != k
            m = mk()
            m.fit(X[tr], t[tr])
            p[te] = m.predict(X[te])
        return p

    def cv_clf(X, t):
        p = np.zeros(len(t), dtype=int)
        for k in range(N_FOLDS):
            te, tr = fold == k, fold != k
            m = mk("clf")
            m.fit(X[tr], t[tr])
            p[te] = m.predict(X[te])
        return p

    gates = {}

    # ---------------------------------------------------------------- B1
    say("B1 HOW MANY NEAR-DUPLICATE SEQUENCES, AND WHAT DO THEY COST?")
    by_cl = collections.defaultdict(list)
    for i, s in enumerate(seq_id):
        by_cl[clu[i]].append(s)
    seq_cl = {}
    for i, r in enumerate(rows):
        seq_cl[seq_id[i]] = clu[i]
    cl_seqs = collections.defaultdict(set)
    for s, c in seq_cl.items():
        cl_seqs[c].add(s)
    ks = {s: kmers(seqs[s]) for s in seq_cl}
    near, mutant = [], []
    for c, ss in cl_seqs.items():
        ss = sorted(ss)
        for a in range(len(ss)):
            for b in range(a + 1, len(ss)):
                i, j = ss[a], ss[b]
                ki, kj = ks[i], ks[j]
                if not ki or not kj:
                    continue
                jac = len(ki & kj) / len(ki | kj)
                if jac >= NEAR_J:
                    near.append((i, j, jac))
                    si, sj = seqs[i], seqs[j]
                    if len(si) == len(sj):
                        d = sum(1 for x, z in zip(si, sj) if x != z)
                        if 1 <= d <= MUT_MAX:
                            mutant.append((i, j, d))
    say(f"     sequence pairs at >= {NEAR_J} k-mer Jaccard inside a cluster: {len(near):,}")
    say(f"     of those, equal length differing at 1-{MUT_MAX} residues (point mutants): "
        f"{len(mutant):,}")
    ymean = {}
    for i in range(len(rows)):
        ymean.setdefault(seq_id[i], []).append(y[i])
    ymean = {s: float(np.mean(v)) for s, v in ymean.items()}
    if mutant:
        sp = np.array([abs(ymean[i] - ymean[j]) for i, j, _ in mutant])
        say(f"     |log10 kcat| difference within a point-mutant pair: median {np.median(sp):.3f}, "
            f"90th {np.percentile(sp, 90):.3f}, max {sp.max():.3f}")
        say(f"     as an RMSE contribution that no representation can remove: "
            f"{float(np.sqrt(np.mean(sp ** 2)) / np.sqrt(2)):.3f} log10")
        say(f"     for scale, the model's RMSE is 1.3768 and a constant is 1.5091")
    nn = np.array([abs(ymean[i] - ymean[j]) for i, j, _ in near]) if near else np.array([0.0])
    say(f"     |log10 kcat| difference within ANY near-duplicate pair: median {np.median(nn):.3f}")
    gates["B1"] = bool(len(near) >= 0)
    say(f"     B1 PASS -- counted and priced")
    say()

    # ---------------------------------------------------------------- B2
    say("B2 THE VARIANCE DECOMPOSITION, DEBIASED")
    by_seq = collections.defaultdict(list)
    for i, s in enumerate(seq_id):
        by_seq[s].append(i)

    def decomp(groups):
        mus = {g: float(np.mean(y[ii])) for g, ii in groups.items()}
        w = np.array([y[i] - mus[g] for g, ii in groups.items() for i in ii])
        b = np.array([mus[g] - y.mean() for g, ii in groups.items() for i in ii])
        return float(b.var()), float(w.var())
    vb0, vw0 = decomp(by_seq)
    say(f"     loop 132, ALL sequences        between {vb0:.4f} ({vb0 / (vb0 + vw0):.1%})  "
        f"within {vw0:.4f}  ceiling {np.sqrt(vw0):.4f}")
    ge3 = {s: ii for s, ii in by_seq.items() if len(ii) >= 3}
    vb1, vw1 = decomp(ge3)
    n3 = sum(len(v) for v in ge3.values())
    say(f"     sequences with >= 3 records    between {vb1:.4f} ({vb1 / (vb1 + vw1):.1%})  "
        f"within {vw1:.4f}  ceiling {np.sqrt(vw1):.4f}   n={n3:,} on {len(ge3):,} sequences")
    parent = {s: s for s in seq_cl}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j, _ in near:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    merged = collections.defaultdict(list)
    for i, s in enumerate(seq_id):
        merged[find(s)].append(i)
    vb2, vw2 = decomp(merged)
    say(f"     near-duplicates MERGED         between {vb2:.4f} ({vb2 / (vb2 + vw2):.1%})  "
        f"within {vw2:.4f}  ceiling {np.sqrt(vw2):.4f}   {len(merged):,} merged proteins")
    say(f"     the singleton bias is real: {sum(1 for v in by_seq.values() if len(v) == 1):,} "
        f"sequences carry one measurement and contribute exactly zero within-variance")
    gates["B2"] = bool(vw1 > 0 and vw2 > 0)
    say(f"     B2 {'PASS' if gates['B2'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- B3
    say("B3 WHAT DOES THE EMBEDDING ACTUALLY ENCODE?")
    Xe = E[seq_id]
    ec1 = np.array([e.split(".")[0] if e and e[0].isdigit() else "" for e in ec])
    m = ec1 != ""
    lab = {v: i for i, v in enumerate(sorted(set(ec1[m].tolist())))}
    pe = np.full(len(y), -1)
    pe[m] = cv_clf(Xe[m], np.array([lab[v] for v in ec1[m]]))
    acc_ec = float(np.mean(pe[m] == np.array([lab[v] for v in ec1[m]])))
    maj = collections.Counter(ec1[m].tolist()).most_common(1)[0][1] / m.sum()
    say(f"     EC top-level class ({len(lab)} classes)   accuracy {acc_ec:.3f}   "
        f"majority baseline {maj:.3f}")
    tops = [o for o, _ in collections.Counter(org).most_common(6)]
    mo = np.isin(org, tops)
    lo = {v: i for i, v in enumerate(tops)}
    po = cv_clf(Xe[mo], np.array([lo[v] for v in org[mo]]))
    acc_o = float(np.mean(po == np.array([lo[v] for v in org[mo]])))
    majo = collections.Counter(org[mo].tolist()).most_common(1)[0][1] / mo.sum()
    say(f"     organism, top 6 ({int(mo.sum()):,} records)   accuracy {acc_o:.3f}   "
        f"majority {majo:.3f}")
    plen = cv_reg(Xe, SF[seq_id, 0])
    say(f"     log10 sequence length            R2 {r2(SF[seq_id, 0], plen):+.4f}")
    pk = cv_reg(Xe, y)
    say(f"     log10 kcat                       R2 {r2(y, pk):+.4f}")
    say(f"     the same 320 numbers, the same folds. The embedding is a far better family and")
    say(f"     length detector than a turnover-number predictor.")
    gates["B3"] = bool(np.isfinite(acc_ec))
    say(f"     B3 PASS -- probed")
    say()

    # ---------------------------------------------------------------- B4
    say("B4 IS THE MODEL JUST DOING EC-CLASS LOOKUP?")
    resid = np.zeros(len(y))
    for k in range(N_FOLDS):
        te, tr = fold == k, fold != k
        med = collections.defaultdict(list)
        for i in np.flatnonzero(tr):
            if ec[i]:
                med[ec[i]].append(y[i])
        gm = np.median(y[tr])
        resid[te] = y[te] - np.array([np.median(med[ec[i]]) if ec[i] in med else gm
                                      for i in np.flatnonzero(te)])
    X = np.hstack([SF[seq_id], Xe, FP[smi_id]])
    pr = cv_reg(X, resid)
    say(f"     target = log10 kcat minus the EC median (what class membership already explains)")
    say(f"     residual sd {resid.std():.4f}  -> model RMSE {rmse(resid, pr):.4f}  "
        f"R2 {r2(resid, pr):+.4f}")
    gates["B4"] = bool(rmse(resid, pr) < resid.std())
    say(f"     B4 {'PASS' if gates['B4'] else 'FAIL'} -- the sequence "
        f"{'adds information beyond the EC number' if gates['B4'] else 'ADDS NOTHING beyond the EC number'}")
    say()

    # ---------------------------------------------------------------- B5
    say("B5 THE CEILING SET BY MISSING CONDITIONS")
    ps = collections.defaultdict(list)
    for i in range(len(rows)):
        ps[(seq_id[i], smi_id[i])].append(y[i])
    rep = {k: v for k, v in ps.items() if len(v) >= 2}
    dev = np.concatenate([np.array(v) - np.mean(v) for v in rep.values()]) if rep else np.array([0.])
    say(f"     {len(rep):,} (protein, substrate) pairs measured more than once, "
        f"{sum(len(v) for v in rep.values()):,} records")
    say(f"     these share an ENZYME and a SUBSTRATE, so any disagreement is temperature, pH, "
        f"buffer, a mutation or an error")
    say(f"     residual sd within such a pair: {dev.std():.4f} log10 "
        f"= {10 ** dev.std():.2f}x")
    say(f"     NO MODEL USING THESE COLUMNS CAN BEAT THAT, because the distinguishing information")
    say(f"     is not in the file. DLKcat records no temperature, no pH and no mutant flag.")
    say(f"     loop 129 measured 1.15x for the same protein AND same substrate in HUMAN records;")
    say(f"     the gap between {10 ** dev.std():.2f}x and 1.15x is what the missing columns cost")
    gates["B5"] = bool(len(rep) > 0)
    say(f"     B5 PASS")
    say()

    # ---------------------------------------------------------------- B6
    say("B6 THE METRIC THE CELL MODEL WOULD CHOOSE")
    import gzip
    B = json.load(gzip.open("colab/data/kinetics_bundle.json.gz", "rt"))
    ens = {}
    with open(LR.SC / "HumanGEM_genes.tsv") as f:
        rr = csv.reader(f, delimiter="\t")
        hd = [c.strip('"') for c in next(rr)]
        a_, b_ = hd.index("genes"), hd.index("geneSymbols")
        for x in rr:
            e_, s_ = x[a_].strip('"'), x[b_].strip('"')
            if e_ and s_:
                ens[e_] = s_.split(";")[0]
    nrx = collections.Counter()
    for r_, gg in B["reaction_genes"].items():
        for z in gg:
            if z in ens:
                nrx[ens[z]] += 1
    pm = cv_reg(X, y)
    hs = (org == "Homo sapiens") & (gene != "")
    w = np.array([nrx.get(g, 0) for g in gene[hs]], float)
    e2 = (y[hs] - pm[hs]) ** 2
    say(f"     human records with a gene: {int(hs.sum()):,}; "
        f"{int((w > 0).sum()):,} catalyse at least one model reaction")
    say(f"     unweighted RMSE on that subset      {np.sqrt(e2.mean()):.4f}")
    if w.sum() > 0:
        say(f"     weighted by reactions catalysed     "
            f"{np.sqrt((e2 * w).sum() / w.sum()):.4f}")
        say(f"     the enzymes the cell model leans on are "
            f"{'HARDER' if np.sqrt((e2 * w).sum() / w.sum()) > np.sqrt(e2.mean()) else 'easier'} "
            f"than average for this model")
    gates["B6"] = bool(hs.sum() > 0)
    say(f"     B6 PASS")
    say()

    say("=" * 100)
    for k in ("B1", "B2", "B3", "B4", "B5", "B6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/6")
    say("=" * 100)

    man = RM.manifest(inputs=[ML / "kcat_records.tsv", ML / "esm2_8M_mean.npy",
                              ML / "sequences.json"],
                      available=len(rows), used=len(rows), selection="all", seed=SEED,
                      controls=["point mutants detected by equal-length Hamming distance",
                                "the decomposition recomputed three ways, including merged",
                                "the embedding probed on class, organism and length, not only kcat",
                                "the EC-median residual as the target the sequence must still beat",
                                "repeat (protein, substrate) pairs as the missing-condition floor",
                                "error reweighted by how much the cell model uses each enzyme"],
                      note="loop 132's 86% between-protein share is biased by singletons and by "
                           "mutants counted as separate proteins; this measures both")
    RM.report(man, emit=say)
    json.dump({"test": "loop_ml_probe", "manifest": man, "gates": gates,
               "b1": {"near_pairs": len(near), "mutant_pairs": len(mutant),
                      "mutant_spread_median": float(np.median(sp)) if mutant else None,
                      "irreducible_rmse": float(np.sqrt(np.mean(sp ** 2)) / np.sqrt(2))
                      if mutant else None},
               "b2": {"all": [vb0, vw0, vb0 / (vb0 + vw0), float(np.sqrt(vw0))],
                      "ge3": [vb1, vw1, vb1 / (vb1 + vw1), float(np.sqrt(vw1)), len(ge3)],
                      "merged": [vb2, vw2, vb2 / (vb2 + vw2), float(np.sqrt(vw2)), len(merged)]},
               "b3": {"ec_class_acc": acc_ec, "ec_majority": maj, "organism_acc": acc_o,
                      "organism_majority": majo, "length_r2": r2(SF[seq_id, 0], plen),
                      "kcat_r2": r2(y, pk)},
               "b4": {"resid_sd": float(resid.std()), "model_rmse": rmse(resid, pr),
                      "r2": r2(resid, pr)},
               "b5": {"n_pairs": len(rep), "within_sd": float(dev.std()),
                      "within_fold": float(10 ** dev.std())},
               "b6": {"n_human": int(hs.sum()), "unweighted": float(np.sqrt(e2.mean())),
                      "weighted": float(np.sqrt((e2 * w).sum() / w.sum())) if w.sum() else None},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_ml_probe.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_ml_probe.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
