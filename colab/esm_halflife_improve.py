"""THE ENHANCER, ON THE ESM HALF-LIFE TRACK: five turns of test -> analyse -> re-analyse.

improver_loop.py's discipline, applied to a track its ask() cannot read. Its ask() is wired to the
kcat artefact shape (models/baselines/model_rmse) and to the cell layer table; pointing it at
loop_esm_halflife.json would produce checks that run without measuring anything, which is this
repository's oldest recorded failure. So the DISCIPLINE is reproduced here rather than the module:

  MEASURED, NOT PROMISED. A turn proposes a change and then EXECUTES it. Its number replaces its
  forecast before the next turn reads it. No turn is allowed to re-argue a settled item.
  THE STALL. If a turn's inputs are unchanged and nothing new was executed, it emits STALLED and
  names what is blocking it. A stalled turn is a real result.
  THE SAME BAR THROUGHOUT. Every turn is scored on the SAME homology-aware folds, against the SAME
  trivial baselines, under the SAME measured ceiling. A turn that changes the evaluation instead of
  the model is not an improvement and is refused.
  THE DRIFT LEDGER. I propose the turns and I write the checks that judge them. That is a conflict
  and it is recorded: every turn logs what would have falsified it.

THE FIVE TURNS, fixed in advance so the sequence is not steered by its own results:

  T1  BASELINE, already run as loop 156. Mean-pooled ESM-2 8M over the whole sequence.
  T2  TERMINAL WINDOWS. The degron literature says the ends carry the signal: N-degrons at the
      N-terminus, and Koren et al. Cell 2018 (doi 10.1016/j.cell.2018.04.028) showed the eukaryotic
      proteome is shaped by E3s reading C-TERMINAL degrons. Mean pooling over 1022 residues buries
      a 10-residue terminus. Embed the N-terminal and C-terminal 60 residues separately and add
      them as channels. Cheap, and motivated by biology rather than by hyperparameter taste.
  T3  THE ANNOTATION CHANNELS REGA SHIPS. Known degron, UPS component, essentiality. Do curated
      annotations add anything over sequence -- or, given this session's audit, is this another
      static-annotation dead end? The audit predicts it adds little; T3 tests that prediction.
  T4  SCALE. ESM-2 35M instead of 8M, same readout. Loop 133 recorded that a 650M encoder was
      worth less than fixing the pooling. T4 is the direct test of that claim on a new target, and
      it is placed AFTER T2 on purpose so pooling gets its chance first.
  T5  THE PHASE AXIS. Predict Rega's CCD-versus-Stable call directly instead of inferring it from
      a predicted half-life. loop 156's E6 measured a 37.3% recovery through the half-life route;
      T5 asks whether the detour costs anything.

WHAT WOULD FALSIFY THE WHOLE EXERCISE: if no turn beats T1 by more than the fold-to-fold noise of
T1 itself, then the readout was already saturated at turn one and the remaining four turns are
decoration. That noise is measured in T1 and printed as the bar every later turn must clear.

-> outputs/esm_halflife_improve.json
"""
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM        # noqa: E402
import loop_replication as LR    # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
REGA = SC / "destroyer" / "rega_4.xlsx"
EMB8 = Path("colab/data/ml/esm2_8M_halflife.npz")
EMB_TERM = Path("colab/data/ml/esm2_8M_halflife_term.npz")
EMB35 = Path("colab/data/ml/esm2_35M_halflife.npz")

SEED = 15700
KMER, JACCARD, NFOLD = 5, 0.30, 5
TERM = 60
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
AA = "ACDEFGHIKLMNPQRSTVWY"
NBOOT = 200

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def _rank(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[o] = np.arange(len(x), dtype=float)
    i, s = 0, x[o]
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return float("nan")
    ra, rb = _rank(a[m]), _rank(b[m])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    r = _rank(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2.0) / (len(pos) * len(neg)))


def ridge_cv(X, y, folds, alphas=ALPHAS):
    pred = np.full(len(y), np.nan)
    for f in sorted(set(folds)):
        tr, te = folds != f, folds == f
        Xtr, ytr = X[tr], y[tr]
        inner = np.arange(len(ytr)) % 3
        best, ba = -9, alphas[0]
        for a in alphas:
            sc = []
            for k in range(3):
                i_tr, i_te = inner != k, inner == k
                A = Xtr[i_tr]
                mu, sd = A.mean(0), A.std(0) + 1e-8
                An = (A - mu) / sd
                w = np.linalg.solve(An.T @ An + a * np.eye(An.shape[1]),
                                    An.T @ (ytr[i_tr] - ytr[i_tr].mean()))
                sc.append(spear(((Xtr[i_te] - mu) / sd) @ w + ytr[i_tr].mean(), ytr[i_te]))
            m = float(np.nanmean(sc))
            if m > best:
                best, ba = m, a
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        An = (Xtr - mu) / sd
        w = np.linalg.solve(An.T @ An + ba * np.eye(An.shape[1]), An.T @ (ytr - ytr.mean()))
        pred[te] = ((X[te] - mu) / sd) @ w + ytr.mean()
    return pred


def fold_spread(pred, y, folds):
    """Fold-to-fold spread of the score: the bar a later turn must clear to mean anything."""
    v = [spear(pred[folds == f], y[folds == f]) for f in sorted(set(folds))]
    return float(np.nanstd(v)), [float(x) for x in v]


def sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  THE ENHANCER ON THE ESM HALF-LIFE TRACK -- five turns, measured not promised")
    say("=" * 100)
    say()

    import pandas as pd
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    d = pd.read_excel(REGA, sheet_name="Proteome", header=1)
    for c in ("halflife_mean", "halflife_std", "halflife_count"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[np.isfinite(d["halflife_mean"]) & (d["halflife_mean"] > 0)].copy()
    Z = np.load(EMB8, allow_pickle=True)
    accs = [str(a) for a in Z["accs"]]
    pos = {a: i for i, a in enumerate(accs)}
    d = d[d["Accession"].astype(str).isin(pos)].copy()
    o = [pos[str(a)] for a in d["Accession"].astype(str)]
    X8 = Z["X"][o]
    lens = Z["lengths"][o].astype(float)
    y = np.log(d["halflife_mean"].values.astype(float))
    accl = d["Accession"].astype(str).values
    ccd = d["Cell Cycle Dependency"].astype(str).values

    seqs, a_, buf = {}, None, []
    with gzip.open(SC / "human_proteome.fasta.gz", "rt", errors="replace") as f:
        for ln in f:
            if ln.startswith(">"):
                if a_ and buf and a_ in pos:
                    seqs[a_] = "".join(buf)
                m = re.match(r">\w\w\|([^|]+)\|", ln)
                a_, buf = (m.group(1) if m else None), []
            else:
                buf.append(ln.strip())
    if a_ and buf and a_ in pos:
        seqs[a_] = "".join(buf)

    # THE FOLDS AND THE BASELINES ARE FIXED ONCE. Every turn is scored on these and no turn may
    # change them; a turn that improved the score by changing the evaluation would be the exact
    # self-deception this file exists to prevent.
    rows, cols = [], []
    for i, a in enumerate(accl):
        s = seqs.get(a, "")
        for k in {hash(s[j:j + KMER]) % (1 << 20) for j in range(max(0, len(s) - KMER + 1))}:
            rows.append(i)
            cols.append(k)
    M = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(accl), 1 << 20))
    sz = np.asarray(M.sum(1)).ravel()
    I = (M @ M.T).toarray()
    J = I / np.maximum(sz[:, None] + sz[None, :] - I, 1e-9)
    np.fill_diagonal(J, 0.0)
    ncomp, lab = connected_components(csr_matrix(J >= JACCARD), directed=False)
    load = np.zeros(NFOLD)
    fmap = {}
    for c in np.argsort(-np.bincount(lab)):
        f = int(np.argmin(load))
        fmap[c] = f
        load[f] += (lab == c).sum()
    folds = np.array([fmap[c] for c in lab])

    comp = np.zeros((len(accl), len(AA)), np.float32)
    for i, a in enumerate(accl):
        s = seqs.get(a, "")
        if s:
            for j, aa in enumerate(AA):
                comp[i, j] = s.count(aa) / len(s)
    bar_comp = spear(ridge_cv(comp, y, folds), y)

    rep = d[np.isfinite(d["halflife_std"]) & (d["halflife_count"] >= 2)]
    cv = (rep["halflife_std"] / rep["halflife_mean"]).values
    reliab = np.var(np.log(rep["halflife_mean"].values)) / (
        np.var(np.log(rep["halflife_mean"].values)) + np.mean(cv ** 2) / rep["halflife_count"].mean())
    ceiling = math.sqrt(max(0.0, min(1.0, reliab)))

    say(f"  {len(y):,} proteins; {ncomp:,} homology clusters; folds fixed; "
        f"composition bar {bar_comp:+.4f}; measured ceiling {ceiling:.4f}")
    say()

    turns, best, best_name = [], None, None

    def record(n, name, pred, note, cost_s, falsifier, extra=None):
        nonlocal best, best_name
        s = spear(pred, y)
        sd, per = fold_spread(pred, y, folds)
        delta = None if best is None else s - best
        beats = None if best is None else bool(delta > turns[0]["fold_sd"])
        say(f"  T{n} {name}")
        say(f"     Spearman {s:+.4f}   fold sd {sd:.4f}   per-fold {[round(x,3) for x in per]}")
        if best is not None:
            say(f"     vs best so far ({best_name} {best:+.4f}): delta {delta:+.4f}   "
                f"bar = T1 fold sd {turns[0]['fold_sd']:.4f}   "
                f"{'CLEARS' if beats else 'does NOT clear'}")
        say(f"     ceiling check {abs(s):.4f} <= {ceiling:.4f}: "
            f"{'ok' if abs(s) <= ceiling else 'IMPOSSIBLE'}")
        say(f"     what would have falsified this turn: {falsifier}")
        say(f"     cost {cost_s:.1f}s")
        r = {"turn": n, "name": name, "spearman": s, "fold_sd": sd, "per_fold": per,
             "delta_vs_best": delta, "clears_noise_bar": beats, "note": note,
             "falsifier": falsifier, "cost_seconds": cost_s,
             "inside_ceiling": bool(abs(s) <= ceiling)}
        if extra:
            r.update(extra)
        turns.append(r)
        if best is None or s > best:
            best, best_name = s, f"T{n}"
        say()
        return r

    # ------------------------------------------------------------------ T1
    t = time.time()
    p1 = ridge_cv(X8, y, folds)
    record(1, "mean-pooled ESM-2 8M, whole sequence", p1,
           "the loop 156 baseline, recomputed here on the same folds",
           time.time() - t,
           "a score at or below the composition bar of "
           f"{bar_comp:+.4f} would have said the encoder adds nothing")

    # ------------------------------------------------------------------ T2
    say("  T2 EXECUTING: embedding the terminal windows (this writes a new artefact)")
    t = time.time()
    if not EMB_TERM.exists():
        import torch
        import esm
        model, alph = esm.pretrained.esm2_t6_8M_UR50D()
        model.eval()
        bc = alph.get_batch_converter()
        nl = model.num_layers
        Nv, Cv = [], []
        B = 64
        for i in range(0, len(accl), B):
            chunk = accl[i:i + B]
            for store, cut in ((Nv, "N"), (Cv, "C")):
                data = [(a, (seqs.get(a, "X")[:TERM] if cut == "N" else seqs.get(a, "X")[-TERM:]))
                        for a in chunk]
                _, _, tok = bc(data)
                with torch.no_grad():
                    rp = model(tok, repr_layers=[nl])["representations"][nl]
                for k, a in enumerate(chunk):
                    L = min(len(seqs.get(a, "X")), TERM)
                    store.append(rp[k, 1:L + 1].mean(0).numpy())
        np.savez_compressed(EMB_TERM, N=np.array(Nv, np.float32), C=np.array(Cv, np.float32),
                            accs=np.array(accl), term=TERM)
        say(f"     wrote {EMB_TERM} in {time.time() - t:.0f}s")
    ZT = np.load(EMB_TERM, allow_pickle=True)
    tacc = {str(a): i for i, a in enumerate(ZT["accs"])}
    ti = [tacc[a] for a in accl]
    XN, XC = ZT["N"][ti], ZT["C"][ti]
    X2 = np.hstack([X8, XN, XC])
    p2 = ridge_cv(X2, y, folds)
    record(2, "+ N-terminal and C-terminal 60-residue windows", p2,
           "Koren 2018 doi 10.1016/j.cell.2018.04.028: the proteome is shaped by E3s reading "
           "C-terminal degrons; mean pooling over 1022 residues buries a 10-residue terminus",
           time.time() - t,
           "no gain over T1 would have said the ends carry nothing the mean did not already have",
           {"dim": int(X2.shape[1])})

    # ------------------------------------------------------------------ T3
    t = time.time()
    ann = []
    for c, transform in (("Known degron", lambda v: 0.0 if str(v).strip() in ("-", "nan", "") else 1.0),
                         ("UPS components", lambda v: 0.0 if str(v).strip() in ("-", "nan", "") else 1.0),
                         ("Essential Protein", lambda v: 1.0 if str(v).strip().lower() in ("true", "yes") else 0.0)):
        if c in d.columns:
            ann.append(d[c].map(transform).values.astype(np.float32))
    A = np.array(ann, np.float32).T if ann else np.zeros((len(y), 1), np.float32)
    X3 = np.hstack([X2, A])
    p3 = ridge_cv(X3, y, folds)
    record(3, "+ Rega's curated annotation channels (degron, UPS, essential)", p3,
           "this session's audit predicts static annotations add little; T3 tests that prediction "
           "rather than assuming it",
           time.time() - t,
           "a large gain would have overturned the audit's central claim that static annotations "
           "are the wrong instrument",
           {"n_annotation_channels": int(A.shape[1])})

    # ------------------------------------------------------------------ T4
    say("  T4 EXECUTING: ESM-2 35M (this writes a new artefact)")
    t = time.time()
    if not EMB35.exists():
        import torch
        import esm
        model, alph = esm.pretrained.esm2_t12_35M_UR50D()
        model.eval()
        bc = alph.get_batch_converter()
        nl = model.num_layers
        order = sorted(range(len(accl)), key=lambda i: len(seqs.get(accl[i], "")))
        V = [None] * len(accl)
        batch, blen = [], 0
        def flush(b):
            if not b:
                return
            data = [(accl[i], seqs.get(accl[i], "X")[:1022]) for i in b]
            _, _, tok = bc(data)
            with torch.no_grad():
                rp = model(tok, repr_layers=[nl])["representations"][nl]
            for k, i in enumerate(b):
                L = min(len(seqs.get(accl[i], "X")), 1022)
                V[i] = rp[k, 1:L + 1].mean(0).numpy()
        for i in order:
            L = min(len(seqs.get(accl[i], "")), 1022)
            if blen + L > 6000 and batch:
                flush(batch)
                batch, blen = [], 0
            batch.append(i)
            blen += L
        flush(batch)
        np.savez_compressed(EMB35, X=np.array(V, np.float32), accs=np.array(accl))
        say(f"     wrote {EMB35} in {time.time() - t:.0f}s")
    Z35 = np.load(EMB35, allow_pickle=True)
    m35 = {str(a): i for i, a in enumerate(Z35["accs"])}
    X4 = Z35["X"][[m35[a] for a in accl]]
    p4 = ridge_cv(X4, y, folds)
    record(4, "ESM-2 35M mean-pooled, whole sequence (scale instead of readout)", p4,
           "loop 133 recorded a 650M encoder worth less than fixing the pooling; this is the "
           "direct test on a new target, deliberately placed AFTER the pooling turn",
           time.time() - t,
           "a large gain over T1 would have overturned loop 133's finding that scale is the "
           "cheaper axis than readout",
           {"dim": int(X4.shape[1])})

    # ------------------------------------------------------------------ T5
    t = time.time()
    is_ccd, is_st = ccd == "CCD", ccd == "Stable"
    sub = is_ccd | is_st
    ylab = is_ccd[sub].astype(float)
    Xb = X2[sub]
    fb = folds[sub]
    pb = ridge_cv(Xb, ylab, fb)
    a_direct = auc(pb[ylab == 1], pb[ylab == 0])
    a_via = auc(-p2[sub][ylab == 1], -p2[sub][ylab == 0])
    a_true = auc(-y[sub][ylab == 1], -y[sub][ylab == 0])
    say("  T5 the phase axis, predicted directly instead of through a half-life")
    say(f"     {int(is_ccd.sum())} CCD vs {int(is_st.sum())} Stable")
    say(f"     measured half-life          AUC {a_true:.4f}   <- ceiling for any model")
    say(f"     via predicted half-life (T2) AUC {a_via:.4f}")
    say(f"     predicted DIRECTLY           AUC {a_direct:.4f}")
    say(f"     the detour through a rate costs {a_direct - a_via:+.4f} AUC")
    say(f"     what would have falsified this turn: a direct classifier scoring BELOW the "
        f"half-life detour would have said the rate is the better intermediate")
    turns.append({"turn": 5, "name": "phase axis predicted directly", "auc_direct": a_direct,
                  "auc_via_halflife": a_via, "auc_measured_halflife": a_true,
                  "detour_cost": a_direct - a_via, "cost_seconds": time.time() - t,
                  "falsifier": "a direct classifier below the detour"})
    say()

    # ------------------------------------------------------------------ ledger
    say("=" * 100)
    say("  THE LEDGER")
    t1sd = turns[0]["fold_sd"]
    say(f"     the bar every turn had to clear: T1's own fold-to-fold sd, {t1sd:.4f}")
    for r in turns[:4]:
        dv = r.get("delta_vs_best")
        say(f"       T{r['turn']} {r['name'][:52]:<52} {r['spearman']:+.4f}"
            + ("" if dv is None else f"   delta {dv:+.4f}  "
                                     f"{'CLEARS' if r['clears_noise_bar'] else 'noise'}"))
    substantive = [r for r in turns[:4] if r.get("clears_noise_bar")]
    say(f"     {len(substantive)} of 3 improvement turns cleared the noise bar")
    if not substantive:
        say(f"     THE FALSIFIER FIRED. No turn beat T1 by more than T1's own fold noise, so the")
        say(f"     readout was already saturated at turn one and turns 2-4 are decoration. That is")
        say(f"     the honest reading and it is what the file said in advance would falsify it.")
    say(f"     best: {best_name} at {best:+.4f}, against a composition bar of {bar_comp:+.4f} and")
    say(f"     a measured ceiling of {ceiling:.4f}")
    say()
    say("  DRIFT LEDGER -- I proposed these five turns and I wrote the checks that judge them.")
    say("     The sequence was fixed in the docstring BEFORE any turn ran, the folds and baselines")
    say("     were computed once and never touched again, and each turn logged what would have")
    say("     falsified it. That does not remove the conflict; it bounds it.")
    say("=" * 100)

    man = RM.manifest(inputs=[REGA, EMB8, SC / "human_proteome.fasta.gz"],
                      available=len(y), used=len(y), selection="all", seed=SEED,
                      controls=["folds, trivial baselines and ceiling fixed once before turn 1 and "
                                "never recomputed",
                                "every turn scored on the same homology-aware folds",
                                "the bar is T1's own fold-to-fold noise, so a turn must beat the "
                                "measurement's instability to count",
                                "each turn records what would have falsified it"],
                      note="the improver's discipline applied to a track its ask() cannot read. "
                           "Five turns fixed in advance: pooling, annotation, scale, and the phase "
                           "axis, each executed rather than promised.")
    RM.report(man, emit=say)
    json.dump({"test": "the enhancer on the ESM half-life track", "manifest": man,
               "turns": turns, "bar_composition": bar_comp, "ceiling": ceiling,
               "n_clusters": int(ncomp), "best": best, "best_turn": best_name,
               "noise_bar": t1sd, "n_substantive": len(substantive),
               "seconds": time.time() - t0, "log": log},
              open(OUT / "esm_halflife_improve.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'esm_halflife_improve.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
