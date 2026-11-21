"""GETTING THE PARTS: can a gene's COLUMN predict its ROW?

THE PROBLEM THIS ATTACKS.  Norman settled half the compositional model today -- given the measured responses to A
and to B, the response to A+B follows by addition at 82-87% of the noise floor. What remains is getting a part at
all for a gene nobody has perturbed. Every input tried so far has failed at exactly that:

    annotation channels   capped by the TWIN CEILING (0.5365 / 0.6212). Two genes with identical channel vectors
                          get identical predictions, forever, however many columns are added.
    ESM-2 sequence        fails G1 and G2.
    network topology      8 tests and a 36-cell sweep, all killed by degree-matched rewiring.
    xline / codep         the two largest gains ever recorded -- but they are MEASUREMENTS, and the transfer
                          stratification showed the whole gain sits on genes already measured elsewhere
                          (+0.0666/+0.0796 measured vs -0.0045/+0.0071 cold).

So the winning features are measurements and the losing features are descriptions, and the cold-start case has no
measurement to offer. Except it does, and we have been throwing it away.

THE OBSERVATION.  The readout matrix is 5,120 perturbations x 8,246 genes, and 4,314 genes appear on BOTH axes.
Every such gene has a ROW (what happens to the cell when you perturb it) and a COLUMN (how it responds when
somebody else is perturbed). We train on rows and predict rows. The column is never used.

But the column exists for genes that were NEVER PERTURBED -- we watched them respond hundreds of times while
other genes were knocked out. And unlike annotation, a column is gene-specific by construction: two annotation
twins respond differently to other perturbations, so they cannot have identical columns. **The column is a
measured feature available in exactly the cold-start case where every other measured feature is absent.**

THE TEST.  Predict the sealed cohorts' responses from:

    chan          annotation channels. The incumbent.
    col           the gene's column, and nothing else.
    chan_col      both.
    col_perm      columns permuted across genes -- every gene keeps a real column, just somebody else's. If this
                  scores like `col` then columns are decorative and the win is the ridge's intercept.
    frequency     the movement-frequency prior. The floor every arm must clear.

LEAK CONTROL, and this is the part that decides whether the result means anything.  A column is a slice DOWN the
matrix, so it contains entries from other perturbations' rows -- and the sealed cohorts' rows are the answers.
Every column is therefore restricted to TRAINING rows only, so no sealed response ever enters a feature. The
diagonal entry M[X, X] is dropped as well: it is the gene's effect on itself, which is part of its own answer.

PREDECLARED, before any number:
    col > frequency, resolved                -> a gene's responsiveness predicts its own effect. Cold start has a
                                                measured input for the first time.
    chan_col > chan, resolved                -> columns add something annotation does not have. This is the
                                                twin ceiling being broken rather than approached.
    col ~ col_perm                           -> columns are decorative and this whole idea is dead.

Swept over basis rank x ridge penalty on robustness.Sweeper, both sealed cohorts reported separately.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
from robustness import Sweeper

OUT, SP = A.OUT, A.SP
K_VALUES = (30, 60, 120)
PENALTIES = (10.0, 100.0, 1000.0)
COL_SVD = 128          # columns are reduced to this many dimensions; raw they are ~4,300-wide and mostly noise


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("GETTING THE PARTS: does a gene's COLUMN (how it responds) predict its ROW (what it does)?")
    report("=" * 100)
    report("  PREDECLARED: col > frequency => cold start has a measured input.")
    report("               chan_col > chan => columns carry what annotation cannot (the twin ceiling).")
    report("               col ~ col_perm  => columns are decorative and the idea is dead.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    gj = {g: j for j, g in enumerate(genes)}
    M = z["M"].astype(np.float32)
    Aabs = np.abs(M)
    tide = (Aabs >= A.TAU).mean(0) >= 0.05

    sealed, answers = set(), {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists():
            answers[tag or "cohort1"] = json.loads(af.read_text())
    train = [k for k in kos if k not in sealed]
    train_rows = np.array([ki[k] for k in train])
    report(f"\n  readout {len(kos):,} x {len(genes):,}; sealed {len(sealed)}; training pool {len(train):,}")

    # ---- COLUMN FEATURES, built ONLY from training rows so no sealed answer can enter a feature ----
    Ctrain = M[train_rows]                                  # (n_train, n_genes)
    def column_of(g):
        j = gj.get(g)
        if j is None:
            return None
        v = Ctrain[:, j].copy()
        r = ki.get(g)
        if r is not None:                                   # drop the gene's own row: that is its answer
            v[train_rows == r] = 0.0
        return v

    universe = sorted(set(kos) | sealed)
    have_col = [g for g in universe if g in gj]
    report(f"  genes with a column: {len(have_col):,} of {len(universe):,} "
           f"({100*len(have_col)/len(universe):.1f}%)")
    for tag, ans in answers.items():
        n = sum(1 for k in ans if k in gj)
        report(f"    sealed {tag}: {n}/{len(ans)} have a column")

    raw = np.zeros((len(universe), Ctrain.shape[0]), np.float32)
    urow = {g: i for i, g in enumerate(universe)}
    for g in have_col:
        raw[urow[g]] = column_of(g)
    # Reduce: a raw column is ~4,300-wide, dominated by noise, and would swamp a ridge with 696 channels.
    # The rotation is fitted on TRAINING genes only and the sealed genes are projected onto it. Fitting it on
    # everything would be transductive -- no answer would leak, since every column already contains only training
    # rows, but the sealed genes would still have helped choose the axes, and that is not a thing to leave
    # arguable in a sealed protocol.
    tr_u = np.array([urow[k] for k in train])
    mu = raw[tr_u].mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(raw[tr_u] - mu, full_matrices=False)
    col = ((raw - mu) @ Vt[:COL_SVD].T).astype(np.float32)
    col /= (np.linalg.norm(col, axis=1, keepdims=True) + 1e-9)
    report(f"  columns reduced {raw.shape[1]:,} -> {COL_SVD} dims, rotation fitted on training genes only "
           f"({100*float((S[:COL_SVD]**2).sum()/(S**2).sum()):.1f}% of training variance)")

    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tb = C.extra_channels(universe, set(train), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    chan = chan / (np.linalg.norm(chan, axis=1, keepdims=True) + 1e-9)
    colp = col[np.random.default_rng(A.SEED + 5).permutation(len(col))]
    report(f"  channels {chan.shape[1]} dims | columns {col.shape[1]} dims")

    from sklearn.decomposition import NMF
    freq = (Aabs[train_rows] >= A.TAU).mean(0)
    freq[tide] = -np.inf

    def score(pred_scores, movers):
        top = np.argsort(-pred_scores)[:A.NPRED]
        return len({genes[t] for t in top} & set(movers)) / float(A.NPRED)

    FEATS = {"chan": chan, "col": col, "chan_col": np.concatenate([chan, col], 1), "col_perm": colp}
    sw = {n: Sweeper(f"{n} - chan", axes={"K": list(K_VALUES), "penalty": list(PENALTIES)})
          for n in ("col", "chan_col", "col_perm")}
    swf = Sweeper("chan_col - frequency", axes={"K": list(K_VALUES), "penalty": list(PENALTIES)})
    detail = {}

    for K in K_VALUES:
        X = Aabs[train_rows].copy()
        X[:, tide] = 0.0
        nmf = NMF(n_components=K, init="nndsvda", max_iter=300, random_state=0)
        W = nmf.fit_transform(X).astype(np.float32)
        H = nmf.components_.astype(np.float32)
        del X
        for pen in PENALTIES:
            per_arm = {}
            for name, F in FEATS.items():
                w = A.ridge_fit(F[tr_u], W, pen)
                per_arm[name] = {}
                for tag, ans in answers.items():
                    hits = []
                    for k in sorted(ans):
                        if k not in urow:
                            continue
                        mix = A.ridge_apply(F[[urow[k]]], w)[0]
                        s = (mix @ H).copy()
                        s[tide] = -np.inf
                        if k in gj:
                            s[gj[k]] = -np.inf
                        hits.append(score(s, ans[k]["movers"]))
                    per_arm[name][tag] = np.array(hits)
            fr = {}
            for tag, ans in answers.items():
                hits = []
                for k in sorted(ans):
                    if k not in urow:
                        continue
                    s = freq.copy()
                    if k in gj:
                        s[gj[k]] = -np.inf
                    hits.append(score(s, ans[k]["movers"]))
                fr[tag] = np.array(hits)

            cfg = {"K": K, "penalty": pen}
            tagl = f"K{K}|p{int(pen)}"
            for nm in ("col", "chan_col", "col_perm"):
                d = np.concatenate([per_arm[nm][t] - per_arm["chan"][t] for t in answers])
                sw[nm].add(cfg, tagl, d)
            swf.add(cfg, tagl, np.concatenate([per_arm["chan_col"][t] - fr[t] for t in answers]))
            detail[tagl] = {"frequency": float(np.mean([fr[t].mean() for t in fr])),
                            **{nm: {t: float(v.mean()) for t, v in per_arm[nm].items()} for nm in FEATS}}
            report(f"  K={K:>3} pen={int(pen):>4}: freq {np.mean([fr[t].mean() for t in fr]):.4f} | " +
                   " | ".join(f"{nm} {np.mean([per_arm[nm][t].mean() for t in answers]):.4f}" for nm in FEATS))

    for nm, s in list(sw.items()) + [("chan_col-freq", swf)]:
        report(s.report())
        if s.inert_axes() or s.duplicate_cells():
            report(f"  GUARD {nm}: inert axes {s.inert_axes()}, duplicate cells {s.duplicate_cells()}")

    def mean_of(nm):
        return float(np.mean([np.mean(list(detail[t][nm].values())) for t in detail]))
    mc, mcol, mcc, mp = (mean_of("chan"), mean_of("col"), mean_of("chan_col"), mean_of("col_perm"))
    mf = float(np.mean([detail[t]["frequency"] for t in detail]))
    report(f"\n  MEANS  frequency {mf:.4f} | chan {mc:.4f} | col {mcol:.4f} | chan_col {mcc:.4f} | "
           f"col_perm {mp:.4f}")

    report("\n  READING")
    vcc = sw["chan_col"].verdict()
    vp = sw["col_perm"].verdict()
    npos = sum(1 for c in sw["chan_col"]._cells.values() if c["lo"] > 0)
    perm_pos = sum(1 for c in sw["col_perm"]._cells.values() if c["lo"] > 0)
    if npos > len(sw["chan_col"]._cells) / 2 and perm_pos == 0:
        report(f"  Columns add {mcc - mc:+.4f} over annotation alone, and permuted columns add "
               f"{mp - mc:+.4f}.")
        report("  A gene's own responsiveness carries information its annotation does not. This is the first")
        report("  measured input available for a gene NOBODY HAS PERTURBED, so it is the first crack in the")
        report("  cold-start wall rather than another retrieval win.")
    elif perm_pos > 0:
        report("  PERMUTED columns score like real ones. The columns are decorative -- whatever is being gained")
        report("  is the ridge's intercept or the extra dimensions, not the gene's biology. Idea dead.")
    else:
        report(f"  Columns add {mcc - mc:+.4f}, not resolved across the grid. Not a result; read the cells.")

    json.dump({"test": "adrn_column_features", "K_values": list(K_VALUES), "penalties": list(PENALTIES),
               "col_svd": COL_SVD, "detail": detail,
               "means": {"frequency": mf, "chan": mc, "col": mcol, "chan_col": mcc, "col_perm": mp},
               "sweeps": {nm: s.verdict() for nm, s in list(sw.items()) + [("chan_col-freq", swf)]},
               "log": log}, open(OUT / "adrn_column_features.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'adrn_column_features.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
