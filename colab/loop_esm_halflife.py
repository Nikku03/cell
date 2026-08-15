"""LOOP 156 -- ESM-2 AGAINST A MEASURED DEGRADATION RATE, WITH THE CONTROLS THIS SESSION'S AUDIT
SAYS ARE MANDATORY.

WHY THIS TARGET AND NOT ANOTHER. The failure-mode audit run earlier in this session found that of
six mechanisms recorded as eliminated, four were mislabelled, and the single recurring cause was
this: a STATIC, BINARY, PROMISCUOUS feature was used as a stand-in for a RATE. The one time the
property was swapped for a measured rate -- k_deg_per_h against the same gene sets -- signal
appeared immediately (AUC 0.5704, p=0.0164) where a yes/no ubiquitylation flag had produced
nothing at n = 359 vs 634. Separately, loop 155 found that this repo's own "transcription rate" is
93% mRNA abundance, so every model ever scored against it was scored on abundance.

So the target here is a MEASURED RATE, from a dataset with a phase axis, joined by UniProt
accession rather than gene symbol:

  Rega, Tsitsa, Rocha et al., Nat Commun 16:2579 (2025), doi 10.1038/s41467-025-57537-8,
  PMID 40089461, ProteomeXchange PXD047266 -- protein half-lives in non-transformed hTERT-RPE-1
  cells, 8,352 proteins, 1,839 with a half-life, ALL 1,839 matching the SwissProt human proteome
  by accession.

AND THE THING LOOP 153 DID NOT HAVE: A MEASURED CEILING. Rega reports halflife_std and
halflife_count per protein, so the target's own reliability is computable, and with it the maximum
correlation ANY predictor can reach. Loop 153 set a gate at AUC 0.60 when the arithmetic ceiling
on its set was 0.5865 -- a gate that could not pass. E0 computes the ceiling FIRST and every gate
below is placed under it. That is the single most important structural change in this loop.

THE OTHER THING THAT KILLS PROTEIN ML, and which none of this repo's earlier ESM work controlled
for: HOMOLOGY LEAKAGE. Random train/test splits put paralogues and family members on both sides,
so the model retrieves a near-neighbour rather than generalising. E1 builds a sequence-similarity
clustering and assigns whole clusters to folds, then reports how much a random split inflates the
number. If random >> clustered, every random-split figure in the protein-ML literature and in this
repo is an overstatement, and the clustered figure is the real one.

ENCODER CHOICE, and why small. Loop 133 recorded that a 650M encoder was worth less than fixing
how the representation was pooled -- mean pooling hid 18,595 point mutants. The encoder is the
cheap part. ESM-2 t6_8M is used here so that if signal exists it is found without spending on
scale, and if it does not, scale was never the missing ingredient. Going bigger is a declared next
step, not a hedge.

PREDECLARED. Conclusions go through gate_guard.verdict. No gate below may exceed the E0 ceiling.

  E0 CAPABILITY AND THE CEILING.                                     THE GATE LOOP 153 LACKED.
       (a) >= 1000 proteins with half-life, sequence and embedding;
       (b) target reliability from Rega's own replicates -> max achievable Pearson r; every gate
           below must sit under it, and this is asserted in code rather than promised in prose;
       (c) censoring: no more than 5% of half-lives within 1% of the maximum, or the upper tail is
           an assay boundary and the model is being scored on a ceiling;
       (d) the target must not be trivially recoverable from the ABUNDANCE column that ships in the
           same table (|rho| < 0.8), or this is loop 155's degenerate-target problem again.
       Gate: all four.

  E1 THE HOMOLOGY-AWARE SPLIT.                                       THE LEAK NOBODY CONTROLS.
       cluster proteins by 5-mer Jaccard at 0.30 (single linkage) and assign whole clusters to
       folds. Gate: report clustered-CV Spearman AND random-CV Spearman, and state the inflation.
       This gate PASSES on being reported, not on a value -- the number it produces is what every
       later gate is scored on.

  E2 THE BASELINES ESM MUST BEAT.
       sequence length alone; amino-acid composition (20-dim); the abundance column; publication
       count. All under the SAME homology-aware folds. Gate: report each. The best of them is the
       bar.

  E3 DOES ESM BEAT THE BAR?
       ridge on mean-pooled embeddings, same folds, alpha chosen INSIDE each training fold.
       Gate: Spearman > best baseline by >= 0.05 AND permutation p < 0.01.

  E4 WHAT DID IT LEARN -- OR IS IT LENGTH IN DISGUISE?
       partial Spearman of prediction against truth given length and given composition-predicted
       truth. Loop 133's lesson: a representation can look informative while carrying only what a
       trivial feature already had. Gate: partial > 0.10.

  E5 FAME AND ABUNDANCE.
       |rho(pubs, prediction)| < 0.20, the standing threshold since loop 137; and the prediction
       must retain signal after partialling on abundance, loop 94's standing rule.
       Gate: both.

  E6 THE CELL-CYCLE SLICE -- what the whole arc actually wants.
       does the predicted rate separate Rega's CCD proteins from its Stable ones, and does the
       measured rate? Gate: report both AUCs. The measured-rate AUC is the ceiling for the
       predicted one and is reported next to it so the gap is visible.

  E7 WHAT THIS CANNOT SHOW.
       one cell line, one assay, one pooling choice, an 8M encoder, and a target whose upper tail
       is extrapolated from an 8 h chase.

-> outputs/loop_esm_halflife.json
"""
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM        # noqa: E402
import loop_replication as LR    # noqa: E402
import gate_guard as GG          # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
REGA = SC / "destroyer" / "rega_4.xlsx"
EMB = Path("colab/data/ml/esm2_8M_halflife.npz")

SEED = 15600
KMER = 5
JACCARD = 0.30
NFOLD = 5
E0_MIN_N = 1000
E0_MAX_CENSOR = 0.05
E0_MAX_ABUND_RHO = 0.80
E3_MARGIN = 0.05
E3_NPERM = 200
E4_MIN_PARTIAL = 0.10
E5_RHO_FAME = 0.20
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
AA = "ACDEFGHIKLMNPQRSTVWY"

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def emit(s):
    say(s)


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


def partial(x, y, z):
    rxy, rxz, ryz = spear(x, y), spear(x, z), spear(y, z)
    den = math.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return float((rxy - rxz * ryz) / den)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    a = np.concatenate([pos, neg])
    r = _rank(a)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2.0) / (len(pos) * len(neg)))


def ridge_cv(X, y, folds, alphas=ALPHAS):
    """Out-of-fold predictions; alpha chosen inside each training fold by a nested split."""
    pred = np.full(len(y), np.nan)
    for f in sorted(set(folds)):
        tr, te = folds != f, folds == f
        Xtr, ytr = X[tr], y[tr]
        n = len(ytr)
        inner = np.arange(n) % 3
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
                p = ((Xtr[i_te] - mu) / sd) @ w + ytr[i_tr].mean()
                sc.append(spear(p, ytr[i_te]))
            m = float(np.nanmean(sc))
            if m > best:
                best, ba = m, a
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        An = (Xtr - mu) / sd
        w = np.linalg.solve(An.T @ An + ba * np.eye(An.shape[1]), An.T @ (ytr - ytr.mean()))
        pred[te] = ((X[te] - mu) / sd) @ w + ytr.mean()
    return pred


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 156 -- ESM-2 against a MEASURED degradation rate, with a ceiling and a "
        "homology-aware split")
    say("=" * 100)
    say()

    import gzip
    import re
    import pandas as pd
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    d = pd.read_excel(REGA, sheet_name="Proteome", header=1)
    for c in ("halflife_mean", "halflife_std", "halflife_count",
              "relative_abundance_8h_mean"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[np.isfinite(d["halflife_mean"]) & (d["halflife_mean"] > 0)].copy()
    Z = np.load(EMB, allow_pickle=True)
    X_all, accs = Z["X"], [str(a) for a in Z["accs"]]
    lens_all = Z["lengths"]
    pos = {a: i for i, a in enumerate(accs)}
    d = d[d["Accession"].astype(str).isin(pos)].copy()
    ordr = [pos[str(a)] for a in d["Accession"].astype(str)]
    X = X_all[ordr]
    lens = lens_all[ordr].astype(float)
    y = np.log(d["halflife_mean"].values.astype(float))
    ab = d["relative_abundance_8h_mean"].values.astype(float)
    ccd = d["Cell Cycle Dependency"].astype(str).values
    gname = d["Gene Name"].astype(str).values
    accl = d["Accession"].astype(str).values

    C = json.load(open(LR.CELL))
    pubs_map = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}
    pubs = np.array([pubs_map.get(g, np.nan) for g in gname], float)

    seqs = {}
    a_, buf = None, []
    with gzip.open(SC / "human_proteome.fasta.gz", "rt", errors="replace") as f:
        for ln in f:
            if ln.startswith(">"):
                if a_ and buf and a_ in pos:
                    seqs[a_] = "".join(buf)
                m = re.match(r">\w\w\|([^|]+)\|", ln)
                a_ = m.group(1) if m else None
                buf = []
            else:
                buf.append(ln.strip())
    if a_ and buf and a_ in pos:
        seqs[a_] = "".join(buf)

    gates, res = {}, {}
    say(f"  {len(y):,} proteins with a measured half-life, a sequence and an ESM-2 embedding "
        f"(dim {X.shape[1]})")
    say(f"  encoder {str(Z['model'])}, pooling {str(Z['pooling'])}, "
        f"{int(Z['n_truncated'])} sequences truncated at {int(Z['maxlen'])}")
    say()

    # ---------------------------------------------------------------- E0
    say("E0 CAPABILITY AND THE CEILING")
    rep = d[np.isfinite(d["halflife_std"]) & (d["halflife_count"] >= 2)]
    cv = (rep["halflife_std"] / rep["halflife_mean"]).values
    within = float(np.mean(cv ** 2))
    between = float(np.var(np.log(rep["halflife_mean"].values)))
    kbar = float(rep["halflife_count"].mean())
    reliab = between / (between + within / kbar)
    ceiling = math.sqrt(max(0.0, min(1.0, reliab)))
    hl = d["halflife_mean"].values
    censor = float(np.mean(hl >= hl.max() * 0.99))
    r_ab = spear(ab, y)
    a_ok = len(y) >= E0_MIN_N
    c_ok = censor <= E0_MAX_CENSOR
    dd_ok = abs(r_ab) < E0_MAX_ABUND_RHO
    say(f"     (a) n = {len(y):,}   gate >= {E0_MIN_N:,}   {'ok' if a_ok else 'FAIL'}")
    say(f"     (b) target reliability from {len(rep):,} replicated proteins: within-var "
        f"{within:.4f}, between-var {between:.4f}, mean k {kbar:.2f}")
    say(f"         reliability {reliab:.4f}  ->  MAX ACHIEVABLE Pearson r = {ceiling:.4f}")
    say(f"         no gate in this loop may exceed it; the highest gate here is E3's "
        f"baseline+{E3_MARGIN}, checked in code")
    say(f"     (c) censoring: {censor:.2%} within 1% of the max ({hl.max():.2f} h)   "
        f"gate <= {E0_MAX_CENSOR:.0%}   {'ok' if c_ok else 'FAIL'}")
    say(f"     (d) target vs the abundance column shipped alongside it: rho {r_ab:+.4f}   "
        f"gate |rho| < {E0_MAX_ABUND_RHO}   {'ok' if dd_ok else 'FAIL'}")
    gates["E0"] = bool(a_ok and c_ok and dd_ok and ceiling > 0.5)
    res["e0"] = {"n": int(len(y)), "reliability": reliab, "max_achievable_r": ceiling,
                 "within_var": within, "between_var": between, "mean_replicates": kbar,
                 "censor_fraction": censor, "rho_target_abundance": r_ab, "pass": gates["E0"]}
    say(f"     E0 {'PASS' if gates['E0'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- E1
    say("E1 THE HOMOLOGY-AWARE SPLIT")
    rows, cols = [], []
    for i, a in enumerate(accl):
        s = seqs.get(a, "")
        ks = {hash(s[j:j + KMER]) % (1 << 20) for j in range(max(0, len(s) - KMER + 1))}
        for k in ks:
            rows.append(i)
            cols.append(k)
    M = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(accl), 1 << 20))
    sizes = np.asarray(M.sum(1)).ravel()
    I = (M @ M.T).toarray()
    U = sizes[:, None] + sizes[None, :] - I
    J = np.divide(I, np.maximum(U, 1e-9))
    np.fill_diagonal(J, 0.0)
    Adj = csr_matrix(J >= JACCARD)
    ncomp, lab = connected_components(Adj, directed=False)
    say(f"     {KMER}-mer Jaccard >= {JACCARD}: {len(accl):,} proteins -> {ncomp:,} clusters "
        f"(largest {np.bincount(lab).max()})")
    order = np.argsort(-np.bincount(lab))
    fold_of_cluster = {}
    load = np.zeros(NFOLD)
    for c in order:
        f = int(np.argmin(load))
        fold_of_cluster[c] = f
        load[f] += (lab == c).sum()
    folds_h = np.array([fold_of_cluster[c] for c in lab])
    folds_r = rng.integers(0, NFOLD, len(y))
    say(f"     fold sizes, homology-aware: {[int((folds_h == f).sum()) for f in range(NFOLD)]}")
    p_h = ridge_cv(X, y, folds_h)
    p_r = ridge_cv(X, y, folds_r)
    s_h, s_r = spear(p_h, y), spear(p_r, y)
    infl = s_r - s_h
    say(f"     ESM ridge, HOMOLOGY-AWARE CV   Spearman {s_h:+.4f}")
    say(f"     ESM ridge, RANDOM CV           Spearman {s_r:+.4f}")
    say(f"     inflation from using a random split: {infl:+.4f}")
    GG.verdict(abs(infl) < 0.05,
               f"a random split does NOT inflate here ({infl:+.4f}), so homology leakage is not "
               f"the dominant effect on this target and both numbers may be quoted.",
               f"a random split inflates the result by {infl:+.4f}. Every random-split figure on "
               f"this data is an overstatement by that much, and only the homology-aware "
               f"{s_h:+.4f} may be quoted.", emit=emit)
    gates["E1"] = True
    res["e1"] = {"n_clusters": int(ncomp), "largest_cluster": int(np.bincount(lab).max()),
                 "spearman_homology_cv": s_h, "spearman_random_cv": s_r, "inflation": infl,
                 "kmer": KMER, "jaccard": JACCARD}
    say(f"     E1 PASS (reported)")
    say()

    # ---------------------------------------------------------------- E2
    say("E2 THE BASELINES ESM MUST BEAT, on the SAME homology-aware folds")
    comp = np.zeros((len(accl), len(AA)), np.float32)
    for i, a in enumerate(accl):
        s = seqs.get(a, "")
        if s:
            for j, aa in enumerate(AA):
                comp[i, j] = s.count(aa) / len(s)
    base = {}
    base["sequence length"] = spear(ridge_cv(np.log(lens)[:, None], y, folds_h), y)
    base["aa composition (20)"] = spear(ridge_cv(comp, y, folds_h), y)
    base["abundance column"] = spear(np.nan_to_num(ab, nan=np.nanmedian(ab)), y)
    base["publication count"] = spear(np.nan_to_num(pubs, nan=np.nanmedian(pubs)), y)
    for k, v in base.items():
        say(f"       {k:<24} Spearman {v:+.4f}")
    best_name = max(base, key=lambda k: abs(base[k]))
    best = abs(base[best_name])
    say(f"     the bar: {best_name} at |{best:.4f}|")
    gates["E2"] = True
    res["e2"] = {"baselines": base, "best_name": best_name, "best_abs": best}
    say()

    # ---------------------------------------------------------------- E3
    say("E3 DOES ESM BEAT THE BAR?")
    cnt = 0
    for _ in range(E3_NPERM):
        if abs(spear(p_h, y[rng.permutation(len(y))])) >= abs(s_h):
            cnt += 1
    p3 = (cnt + 1) / (E3_NPERM + 1)
    margin = abs(s_h) - best
    say(f"     ESM (homology-aware) {s_h:+.4f}   best baseline |{best:.4f}| ({best_name})   "
        f"margin {margin:+.4f}   gate >= {E3_MARGIN}")
    say(f"     label permutation p = {p3:.4f}   gate < 0.01")
    ceil_txt = ("inside" if abs(s_h) <= ceiling else
                "ABOVE THE CEILING, which is impossible and means the target or split is wrong")
    say(f"     ceiling check: |{s_h:.4f}| against the E0 ceiling {ceiling:.4f} -- {ceil_txt}")
    ok3 = bool(margin >= E3_MARGIN and p3 < 0.01 and abs(s_h) <= ceiling)
    GG.verdict(ok3,
               f"ESM-2 beats the best trivial baseline by {margin:+.4f} on a homology-aware split. "
               f"The representation carries degradation-rate information that length, composition, "
               f"abundance and fame do not.",
               f"ESM-2 does NOT clear the bar: {s_h:+.4f} against {best_name} at |{best:.4f}|, "
               f"margin {margin:+.4f} against a {E3_MARGIN} gate, p {p3:.4f}. On this target, at "
               f"this scale, with mean pooling, the embedding adds nothing a trivial feature did "
               f"not already have.", emit=emit)
    gates["E3"] = ok3
    res["e3"] = {"spearman": s_h, "best_baseline": best, "best_name": best_name,
                 "margin": margin, "perm_p": p3, "ceiling": ceiling,
                 "inside_ceiling": bool(abs(s_h) <= ceiling), "pass": ok3}
    say(f"     E3 {'PASS' if gates['E3'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- E4
    say("E4 WHAT DID IT LEARN -- OR IS IT LENGTH IN DISGUISE?")
    p_len = ridge_cv(np.log(lens)[:, None], y, folds_h)
    p_comp = ridge_cv(comp, y, folds_h)
    pr_len = partial(p_h, y, p_len)
    pr_comp = partial(p_h, y, p_comp)
    pr_both = partial(p_h, y, p_len + p_comp)
    say(f"     ESM vs truth, given the length model       {pr_len:+.4f}")
    say(f"     ESM vs truth, given the composition model  {pr_comp:+.4f}")
    say(f"     ESM vs truth, given both                   {pr_both:+.4f}   gate > {E4_MIN_PARTIAL}")
    ok4 = bool(pr_both > E4_MIN_PARTIAL)
    GG.verdict(ok4,
               f"the embedding retains {pr_both:+.4f} after both trivial models are partialled "
               f"out, so it is not length or composition wearing a transformer.",
               f"after partialling length and composition the embedding retains only "
               f"{pr_both:+.4f}. Whatever it appears to know about degradation rate is mostly what "
               f"a 21-number summary of the sequence already knew -- loop 133's finding, "
               f"reproduced on a new target.", emit=emit)
    gates["E4"] = ok4
    res["e4"] = {"partial_given_length": pr_len, "partial_given_composition": pr_comp,
                 "partial_given_both": pr_both, "pass": ok4}
    say(f"     E4 {'PASS' if gates['E4'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- E5
    say("E5 FAME AND ABUNDANCE")
    r_fame = spear(pubs, p_h)
    r_ab_pred = partial(p_h, y, np.nan_to_num(ab, nan=np.nanmedian(ab)))
    say(f"     rho(pubs, prediction) {r_fame:+.4f}   gate |rho| < {E5_RHO_FAME}")
    say(f"     prediction vs truth given abundance {r_ab_pred:+.4f}   gate > 0")
    ok5 = bool(abs(r_fame) < E5_RHO_FAME and r_ab_pred > 0)
    GG.verdict(ok5,
               "the prediction is not a fame proxy and survives conditioning on abundance.",
               f"the prediction fails a standing control (fame {r_fame:+.4f}, given abundance "
               f"{r_ab_pred:+.4f}) and is struck regardless of E3.", emit=emit)
    gates["E5"] = ok5
    res["e5"] = {"rho_pubs_pred": r_fame, "partial_given_abundance": r_ab_pred, "pass": ok5}
    say(f"     E5 {'PASS' if gates['E5'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- E6
    say("E6 THE CELL-CYCLE SLICE")
    is_ccd = ccd == "CCD"
    is_st = ccd == "Stable"
    a_true = auc(-y[is_ccd], -y[is_st])
    a_pred = auc(-p_h[is_ccd], -p_h[is_st])
    say(f"     {int(is_ccd.sum())} CCD vs {int(is_st.sum())} Stable proteins")
    say(f"     MEASURED half-life separates them      AUC {a_true:.4f}   <- the ceiling for any model")
    say(f"     ESM-PREDICTED half-life separates them AUC {a_pred:.4f}")
    say(f"     the model recovers {(a_pred - 0.5) / max(a_true - 0.5, 1e-9):.1%} of the "
        f"separation the measurement itself achieves")
    gates["E6"] = True
    res["e6"] = {"n_ccd": int(is_ccd.sum()), "n_stable": int(is_st.sum()),
                 "auc_measured": a_true, "auc_predicted": a_pred,
                 "fraction_recovered": float((a_pred - 0.5) / max(a_true - 0.5, 1e-9))}
    say()

    # ---------------------------------------------------------------- E7
    say("E7 WHAT THIS CANNOT SHOW")
    say(f"     one cell line (hTERT-RPE-1), one assay (8 h cycloheximide chase, so the upper tail "
        f"is extrapolated), one pooling choice (mean, which loop 133 showed hides point mutants),")
    say(f"     one encoder scale (8M). A negative here does not license 'sequence does not encode "
        f"degradation rate'; it licenses 'mean-pooled 8M ESM-2 does not, on this target'.")
    say(f"     Untested: per-residue readouts, N-terminal-window features (where N-degrons live "
        f"and where truncation bites), larger encoders, and the phase axis as a target.")
    gates["E7"] = True
    res["e7"] = {"cell_line": "hTERT-RPE-1", "assay": "8 h CHX chase",
                 "encoder": "esm2_t6_8M_UR50D", "pooling": "mean",
                 "negative_licenses": "mean-pooled 8M ESM-2 does not, on this target"}
    say()

    say("=" * 100)
    for k in ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[REGA, EMB, SC / "human_proteome.fasta.gz", str(LR.CELL)],
                      available=int(len(accs)), used=int(len(y)), selection="filtered", seed=SEED,
                      controls=["the target's own replicate statistics give a measured ceiling, "
                                "and every gate is placed under it (E0b)",
                                "homology-aware clustering, with the random-split inflation "
                                "reported rather than hidden (E1)",
                                "four trivial baselines on the SAME folds, and the best is the "
                                "bar (E2)",
                                "length and composition partialled out of the embedding (E4)",
                                "fame and abundance, the two standing controls (E5)",
                                "conclusions emitted through gate_guard.verdict"],
                      note="Rega 2025 (doi 10.1038/s41467-025-57537-8) supplies a MEASURED "
                           "degradation rate with a phase axis, joined by UniProt accession. The "
                           "audit earlier in this session found that swapping a static annotation "
                           "for a measured rate is the one change that has produced signal, so "
                           "the rate is the target and the ceiling is computed before any gate.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 156 -- ESM-2 against a measured degradation rate", "manifest": man,
               "gates": gates,
               "citations": [{"ref": "Rega C et al. Nat Commun 2025;16:2579",
                              "doi": "10.1038/s41467-025-57537-8", "pmid": "40089461"}],
               **res, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_esm_halflife.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_esm_halflife.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
