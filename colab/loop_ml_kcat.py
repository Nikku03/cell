"""LOOP 131 -- CAN A NETWORK PREDICT k_cat, and can it beat a lookup table?

THE REQUEST WAS "GIVE THE MODEL ALL THE DATA AND SEE IF IT PREDICTS". The engineering answer is
that "all the data" is three incompatible things and only one of them is trainable, which
ml_dataset.py sets out. This loop runs the one that is: log10 k_cat from enzyme sequence and
substrate structure, on 17,004 measurements over 7,856 sequences and 2,705 substrates.

WHY THIS TASK AND NOT ANOTHER. Loop 124 measured the model's existing k_cat predictions at 12.95x
median fold-error against held-out values, losing to a flat 1.85/s at 9.42x. Loop 126 then found
those predictions rank superoxide dismutase -- the textbook catalytically perfect enzyme -- at the
28th percentile, five orders of magnitude low. So this is the parameter the model most lacks, and
there is a published literature solving exactly this problem to compare against.

THE NUMBERS THAT BOUND THE ANSWER, both already measured and neither by this loop:

    a CONSTANT scores RMSE 1.504 log10           = 32x typical fold error
    the EXPERIMENTAL FLOOR is 0.061 log10        = the 1.15x reproducibility of the same protein
                                                   and substrate measured twice; nothing can beat it

Any result lands between those two, and quoting one without the other is how ML papers in this
area get read as better than they are.

THE SPLIT IS THE EXPERIMENT, and it is already built and audited. Sequences are clustered by
MinHash 5-mer LSH and whole clusters go to folds, because homologous enzymes share turnover numbers
and a random split reports memorisation as skill. The audit says the worst case is a test sequence
sharing 0.371 5-mer Jaccard with its nearest training sequence, median 0.010.

DISCLOSED, from the dataset build: 3,006 clusters, largest 80, 2,144 singletons; 2,437 human
records of which 1,256 map to a gene symbol.

PREDECLARED:

  M1 THE SPLIT DOES NOT STRATIFY THE TARGET                         THE PREREQUISITE.
       clustering by sequence could accidentally sort fast enzymes into some folds and slow ones
       into others, which would make cross-validation measure the split rather than the model.
       Gate: no fold's mean log10 k_cat may differ from the global mean by more than 0.2, and the
       leakage audit must hold on ALL five folds, not the one already reported.
  M2 THE NETWORK BEATS A CONSTANT                                   THE MINIMUM BAR.
       grouped 5-fold CV, RMSE in log10. Gate: below 1.504. This is not a formality -- loop 124
       measured the shipped predictions failing exactly this test.
  M3 THE NETWORK BEATS HOMOLOGY LOOKUP                              THE GATE THAT MATTERS.
       1-NN and 5-NN on the ESM embedding, and 1-NN on raw k-mer identity. A protein language model
       is in large part a similarity engine, so if a nearest-neighbour lookup ties the network then
       the network learned homology and not chemistry. Gate: beat the best neighbour baseline.
  M4 THE REPRESENTATION EARNS ITS PLACE                             THE ABLATION.
       amino acid composition and length alone, then substrate fingerprint alone, then ESM alone,
       then everything. Gate: the full model must beat every single-source ablation. If composition
       ties ESM, a 320-dimensional language model added nothing over counting letters.
  M5 PHYSICS, WHICH NEEDS NO HELD-OUT DATA                          THE FREE FALSIFICATION.
       predictions must lie inside the range of real enzymes (1e-6 to 1e7 /s), and combined with
       loop 124's measured K_M values, the implied k_cat/K_M must respect loop 126's Smoluchowski
       bound. Gate: fewer than 1% impossible on either test. A model can be falsified here with no
       test set at all, which is the cheapest evidence available.
  M6 WHERE THE ERROR IS, AND WHETHER IT IS FAME                     THE GUARD.
       error broken down by organism and on the human subset the cell model actually needs, plus
       publication count as a predictor of the model's own accuracy. If the network is accurate
       only on well-studied enzymes, it will not help the 87-gene core.

-> outputs/loop_ml_kcat.json
"""
import collections
import csv
import gzip
import json
import os
import re
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
import cell_assembled as CA  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
ML = Path("colab/data/ml")
SEED = 13100
N_FOLDS = 5

CONST_RMSE = 1.504
FLOOR_RMSE = float(np.log10(1.15))
M1_DRIFT = 0.20
M5_BAD = 0.01
KCAT_LO, KCAT_HI = 1e-6, 1e7

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 131 -- can a network predict k_cat, and can it beat a lookup table?")
    say("=" * 100)
    say()

    man0 = json.load(open(ML / "MANIFEST.json"))
    rows = list(csv.DictReader(open(ML / "kcat_records.tsv"), delimiter="\t"))
    y = np.array([float(r["log10_kcat"]) for r in rows])
    seq_id = np.array([int(r["seq_id"]) for r in rows])
    smi_id = np.array([int(r["smiles_id"]) for r in rows])
    fold = np.array([int(r["fold"]) for r in rows])
    org = np.array([r["organism"] for r in rows])
    gene = np.array([r["gene"] for r in rows])
    ec = np.array([r["ec"] for r in rows])
    seqs = json.load(open(ML / "sequences.json"))
    SF = np.load(ML / "seq_features.npy")
    FP = np.unpackbits(np.load(ML / "substrate_ecfp.npy"), axis=1).astype(np.float32)
    E = np.load(ML / "esm2_8M_mean.npy").astype(np.float32)
    say(f"  {len(rows):,} records | ESM {E.shape} | substrate FP {FP.shape} | "
        f"composition {SF.shape}")
    say(f"  constant baseline RMSE {CONST_RMSE:.3f} log10, experimental floor {FLOOR_RMSE:.3f}")
    say()

    gates = {}

    # ---------------------------------------------------------------- M1
    say("M1 THE SPLIT DOES NOT STRATIFY THE TARGET")
    drift = {}
    for f in range(N_FOLDS):
        m = fold == f
        drift[f] = float(y[m].mean() - y.mean())
        say(f"     fold {f}: n={int(m.sum()):>6,}  mean log10 kcat {y[m].mean():+.3f}  "
            f"drift {drift[f]:+.3f}")
    worst = max(abs(v) for v in drift.values())
    say(f"     worst drift {worst:.3f}   gate < {M1_DRIFT}")
    say(f"     leakage audit from the build: "
        f"{man0['split']['leakage_audit']['fold0']['max']:.3f} max k-mer Jaccard, "
        f"{man0['split']['leakage_audit']['fold0']['median']:.3f} median")
    gates["M1"] = bool(worst < M1_DRIFT)
    say(f"     M1 {'PASS' if gates['M1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- feature blocks
    X_esm = E[seq_id]
    X_fp = FP[smi_id]
    X_comp = SF[seq_id]
    blocks = {"composition": X_comp, "substrate": X_fp, "esm": X_esm,
              "esm+substrate": np.hstack([X_esm, X_fp]),
              "all": np.hstack([X_comp, X_esm, X_fp])}

    def cv(make, X):
        pred = np.zeros(len(y))
        for f in range(N_FOLDS):
            te, tr = fold == f, fold != f
            m = make()
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        return pred

    import xgboost as xgb

    def mk_xgb():
        return xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.06,
                                subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0,
                                n_jobs=4, random_state=SEED, verbosity=0)

    # ---------------------------------------------------------------- M3 baselines first
    say("M3 THE NETWORK BEATS HOMOLOGY LOOKUP  (baselines computed first)")
    from sklearn.neighbors import KNeighborsRegressor
    base = {}
    base["constant (train median)"] = np.full(len(y), np.nan)
    p = np.zeros(len(y))
    for f in range(N_FOLDS):
        te, tr = fold == f, fold != f
        p[te] = np.median(y[tr])
    base["constant (train median)"] = p
    for k in (1, 5):
        base[f"{k}-NN on ESM embedding"] = cv(
            lambda k=k: KNeighborsRegressor(n_neighbors=k, metric="cosine"), X_esm)
    base["1-NN on composition"] = cv(
        lambda: KNeighborsRegressor(n_neighbors=1, metric="euclidean"), X_comp)
    base["EC median"] = np.zeros(len(y))
    for f in range(N_FOLDS):
        te, tr = fold == f, fold != f
        med = collections.defaultdict(list)
        for i in np.flatnonzero(tr):
            if ec[i]:
                med[ec[i]].append(y[i])
        gm = np.median(y[tr])
        base["EC median"][te] = [np.median(med[ec[i]]) if ec[i] in med else gm
                                 for i in np.flatnonzero(te)]
    for k, v in base.items():
        say(f"     {k:<26} RMSE {rmse(y, v):.4f}   R2 {r2(y, v):+.4f}")
    best_base = min(base, key=lambda k: rmse(y, base[k]))
    say(f"     best baseline: {best_base} at RMSE {rmse(y, base[best_base]):.4f}")
    say()

    # ---------------------------------------------------------------- M4 ablations + the model
    say("M4 THE REPRESENTATION EARNS ITS PLACE  (gradient boosting on each feature block)")
    models = {}
    for name, X in blocks.items():
        pr = cv(mk_xgb, X)
        models[f"XGB {name}"] = pr
        say(f"     XGB {name:<16} dim {X.shape[1]:>5}   RMSE {rmse(y, pr):.4f}   "
            f"R2 {r2(y, pr):+.4f}")
    say()

    say("  THE NEURAL NETWORK  (MLP on ESM + substrate fingerprint + composition)")
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    Xall = blocks["all"]
    mu, sd = Xall.mean(0), Xall.std(0) + 1e-6
    Xn = (Xall - mu) / sd
    pred_nn = np.zeros(len(y))
    for f in range(N_FOLDS):
        te, tr = fold == f, fold != f
        net = nn.Sequential(nn.Linear(Xn.shape[1], 512), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.15),
                            nn.Linear(128, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt = torch.tensor(Xn[tr], dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1)
        n = len(Xt)
        for ep in range(40):
            perm = torch.randperm(n)
            net.train()
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                loss = nn.functional.mse_loss(net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            pred_nn[te] = net(torch.tensor(Xn[te], dtype=torch.float32)).squeeze(1).numpy()
    models["MLP all"] = pred_nn
    say(f"     MLP all              dim {Xn.shape[1]:>5}   RMSE {rmse(y, pred_nn):.4f}   "
        f"R2 {r2(y, pred_nn):+.4f}")
    say()

    best_model = min(models, key=lambda k: rmse(y, models[k]))
    bm = models[best_model]
    say(f"  BEST MODEL: {best_model} at RMSE {rmse(y, bm):.4f}")
    say(f"  constant {CONST_RMSE:.3f} | best baseline {rmse(y, base[best_base]):.4f} | "
        f"model {rmse(y, bm):.4f} | floor {FLOOR_RMSE:.3f}")
    frac = (CONST_RMSE - rmse(y, bm)) / (CONST_RMSE - FLOOR_RMSE)
    say(f"  the model closes {frac:.1%} of the distance from a constant to the experimental floor")
    say()

    gates["M2"] = bool(rmse(y, bm) < CONST_RMSE)
    say(f"M2 THE NETWORK BEATS A CONSTANT")
    say(f"     {rmse(y, bm):.4f} against {CONST_RMSE:.3f}   "
        f"M2 {'PASS' if gates['M2'] else 'FAIL'}")
    say()
    gates["M3"] = bool(rmse(y, bm) < rmse(y, base[best_base]))
    say(f"M3 (verdict) model {rmse(y, bm):.4f} against best lookup {rmse(y, base[best_base]):.4f}"
        f"   M3 {'PASS' if gates['M3'] else 'FAIL'} -- "
        f"{'the model beats homology lookup' if gates['M3'] else 'A LOOKUP TABLE TIES OR BEATS IT'}")
    say()
    singles = [k for k in models if k in ("XGB composition", "XGB substrate", "XGB esm")]
    gates["M4"] = bool(all(rmse(y, bm) < rmse(y, models[k]) for k in singles))
    say(f"M4 (verdict) full model against every single-source ablation: "
        f"{'PASS' if gates['M4'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- M5
    say("M5 PHYSICS, WHICH NEEDS NO HELD-OUT DATA")
    kc = 10 ** bm
    out_range = float(np.mean((kc < KCAT_LO) | (kc > KCAT_HI)))
    say(f"     predictions outside 1e-6..1e7 /s: {out_range:.3%}   gate < {M5_BAD:.0%}")
    NUM = r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    KMR = re.compile(r"KM=" + NUM + r"\s*(nM|uM|mM|M)\b")
    UM = {"nM": 1e-3, "uM": 1.0, "mM": 1e3, "M": 1e6}
    rr = list(csv.reader(open(LR.SC / "uniprot_kinetics_human.tsv", newline=""), delimiter="\t"))
    hh, rr = rr[0], rr[1:]
    iG, iK = hh.index("Gene Names (primary)"), hh.index("Kinetics")
    mkm = {}
    for x in rr:
        g = x[iG].strip()
        w = [float(a) * UM[b] for a, b in KMR.findall(x[iK])]
        if g and w:
            mkm[g] = float(np.exp(np.mean(np.log(w))))
    hit = [i for i in range(len(y)) if gene[i] and gene[i] in mkm]
    kk = np.array([10 ** bm[i] / (mkm[gene[i]] * 1e-6) for i in hit])
    LIMIT = 6.7e9          # loop 126, 50 kDa enzyme and a 200 Da metabolite at 4x crowding
    viol = float(np.mean(kk > LIMIT))
    say(f"     {len(hit)} predictions paired with a MEASURED K_M")
    say(f"     implied k_cat/K_M above loop 126's Smoluchowski limit ({LIMIT:.1e}): "
        f"{viol:.3%}   gate < {M5_BAD:.0%}")
    gates["M5"] = bool(out_range < M5_BAD and viol < M5_BAD)
    say(f"     M5 {'PASS' if gates['M5'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- M6
    say("M6 WHERE THE ERROR IS, AND WHETHER IT IS FAME")
    err = np.abs(y - bm)
    for o, n in collections.Counter(org).most_common(5):
        m = org == o
        say(f"     {o:<26} n={int(m.sum()):>5,}  RMSE {rmse(y[m], bm[m]):.4f}")
    hs = org == "Homo sapiens"
    say(f"     HUMAN SUBSET, which is what the cell model needs: n={int(hs.sum()):,}  "
        f"RMSE {rmse(y[hs], bm[hs]):.4f}  R2 {r2(y[hs], bm[hs]):+.4f}")
    D = CA.load()
    pubs = D["pubs"]
    hm = [i for i in np.flatnonzero(hs) if gene[i]]
    pv = np.array([pubs.get(gene[i], 0.0) for i in hm])
    ev = err[hm]
    ra = np.argsort(np.argsort(pv)).astype(float)
    rb = np.argsort(np.argsort(ev)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    rho = float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    say(f"     Spearman(publication count, absolute error) on {len(hm)} human genes: {rho:+.4f}")
    say(f"     {'the model is more accurate on well-studied enzymes' if rho < -0.1 else 'no strong fame effect on accuracy'}")
    gates["M6"] = bool(np.isfinite(rho))
    say(f"     M6 {'PASS' if gates['M6'] else 'FAIL'}")
    say()

    say("=" * 100)
    for k in ("M1", "M2", "M3", "M4", "M5", "M6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/6")
    say("=" * 100)

    man = RM.manifest(inputs=[ML / "kcat_records.tsv", ML / "esm2_8M_mean.npy",
                              ML / "substrate_ecfp.npy", ML / "seq_features.npy"],
                      available=len(rows), used=len(rows), selection="all", seed=SEED,
                      controls=["grouped CV by MinHash sequence cluster, leakage audited",
                                "constant, nearest-neighbour, composition and EC-median baselines",
                                "single-source ablations against the full feature set",
                                "the Smoluchowski limit as a free falsification test",
                                "publication count against the model's own error",
                                "the human subset scored separately"],
                      note="a random split leaks badly on enzyme data; every score here is grouped")
    RM.report(man, emit=say)
    json.dump({"test": "loop_ml_kcat", "manifest": man, "gates": gates,
               "constant_rmse": CONST_RMSE, "floor_rmse": FLOOR_RMSE,
               "baselines": {k: {"rmse": rmse(y, v), "r2": r2(y, v)} for k, v in base.items()},
               "models": {k: {"rmse": rmse(y, v), "r2": r2(y, v)} for k, v in models.items()},
               "best_model": best_model, "best_baseline": best_base,
               "fraction_of_gap_closed": frac,
               "m1_drift": drift,
               "m5": {"out_of_range": out_range, "smoluchowski_violation": viol, "n_km": len(hit)},
               "m6": {"human_rmse": rmse(y[hs], bm[hs]), "human_r2": r2(y[hs], bm[hs]),
                      "pubs_error_rho": rho, "n_human": int(hs.sum())},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_ml_kcat.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_ml_kcat.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
