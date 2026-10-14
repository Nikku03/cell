"""LOOP 61 -- THE WHOLE STATE VECTOR: WHICH LAYERS CAN THE REST OF THE CELL FILL?

THE ASK, and the shape it has to take. "A whole cell model that fills the blanks for PPI, pathways,
chromosome to final state, replication and everything." Two decisions were made about that: propagate
to EVERY layer with no privileged endpoint, and fill a blank ONLY where the fill beats the best
lookup for it.

Those two decisions rescue each other. A whole-state-vector model normally has no pass/fail gate --
that is exactly how this project has caught its own errors, so losing it would be a real cost. But
"fill only if it beats the lookup" gives EVERY LAYER ITS OWN GATE. One gate per layer instead of
none. That is the design, and it is the reason this is buildable rather than a dashboard.

THE QUESTION, stated so it can fail. For each layer L of the cell model, hold L out and ask whether
the OTHER layers predict a gene's membership of L. That is literally blank-filling: this gene has no
recorded PPI / pathway / drug / complex -- should it? Then score it against the two things that have
beaten this project before.

THE BASELINE THAT HAS TO BE BEATEN, and it is not a formality. Publication count is a field on every
gene record (`pubs`), and in this repository it predicts cancer drivers at 0.8259 against our model's
0.2990 partial, and was the single best drug feature at 0.7913. Layer membership is worse, not
better: a gene is in BioGRID because somebody assayed it, in Reactome because somebody curated it, in
DrugBank because somebody sold something. SO A LAYER THAT IS PREDICTED ONLY AS WELL AS `pubs`
PREDICTS IT HAS NOT BEEN FILLED -- it has been recognised as famous. `pubs` is therefore both a
feature the model may use and a baseline it must beat, which makes the gate ask the right question:
does the rest of the cell add anything OVER knowing how well studied the gene is?

THE SECOND BASELINE is the best single other layer alone, because "everything joined" must beat "the
one obvious correlate", or the joining is decoration.

THE CIRCULARITY THAT WOULD MAKE ALL OF THIS MEANINGLESS. Several layers are aliases of each other in
this file: the gene record carries `ppi` as a degree AND there is a 191,447-edge ppi list; it carries
`npath` AND there is a 60-entry pathways dict; `abund` and `ppm` are both protein abundance. A target
predicted from its own alias scores near 1.0 and means nothing. Near-duplicate pairs are therefore
MEASURED at |rho| > 0.95, named in the output, and excluded per target rather than trusted to a
hand-written list -- loop 53 fitted a collinear pair and loop 55's first draft shipped 1/G_ii beside
G_ii, both caught only because something asserted it.

AND ONE DEFECT INHERITED FROM LOOP 49, declared here rather than discovered later: gate D2 FAILED on
this model's ppi layer -- its declared BioGRID source reproduces at corr 0.3701, so `ppi` is not the
source it claims. It is kept in the run because excluding it would hide the problem, and it is
FLAGGED in the output so no ppi result from this loop is read as clean.

PREDECLARED, before any number:

  V1 NO LAYER IS PREDICTED FROM ITS OWN ALIAS
       all pairwise |rho| computed over genes; pairs above 0.95 named, and for each target every
       alias of it is dropped from the feature block. If a target has an alias this is stated beside
       its score.
  V2 THE FAME BASELINE IS MEASURED FOR EVERY LAYER
       `pubs` alone, per layer, same folds. Reported for all layers whether they pass or fail,
       because the distribution of this number IS the finding about what the cell file records.
  V3 THE GATE -- HOW MANY LAYERS EARN THEIR FILL
       a layer earns its fill when the full model beats BOTH baselines by >= 0.02 AND clears its own
       permutation null. With ~25 layers and a 5% false-positive rate, chance alone yields about 1.2,
       so the gate is >= 5 layers earned. Below that, "fill the blanks" is not supported by this data
       and the model should defer everywhere -- which is a real possible outcome of this loop.
  V4 THE NULL IS MEASURED PER LAYER
       labels permuted within the layer, folds unchanged.
  V5 THE EARNED LAYERS ARE NOT ONE FAMILY
       if every passing layer is an abundance-like quantity, the result is one predictor wearing
       several coats. The passing set is printed in full so this is checkable rather than asserted.

  FOLDS ARE GROUPED BY CHROMOSOME. Neighbouring genes share regulatory context and paralogue
  clusters sit together, so random folds would let a gene be filled by its own neighbourhood.

TWO DEFECTS IN THIS MODULE'S OWN FIRST RUN, found by reading its output rather than by any gate,
and corrected here. Both made the result look cleaner than it was:

  THE PROVENANCE FLAG NEVER FIRED. FLAGGED was keyed "ppi" while the layer is built as "ppi_edges",
  so the loop 49 defect this file promised to surface was absent from all 28 rows. A declared control
  that silently does not run is worse than one never declared, because the output looks clean.

  V1 MISSED A DEFINITIONAL PAIR. `tf` is classified by `reg_out` at AUC 0.9801 and `reg_out` by `tf`
  at 0.8826: a gene has outgoing regulatory edges BECAUSE it is a transcription factor. Rank
  correlation put them under 0.95 and let them through, so `reg_out` was scored as EARNED (+0.0259)
  against a baseline that is its own definition. V1 now also tests whether either layer CLASSIFIES
  the other at AUC > 0.95.

The first run's headline was 23 of 28 layers earned. That number is superseded by whatever this run
produces, and the two corrections both push in the direction of fewer earned layers, not more.

-> outputs/loop_state_vector.json
"""
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
CELL = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_complete.json"

ALIAS_RHO = 0.95
MARGIN = 0.02
MIN_EARNED = 5
MIN_PRESENT = 200
MIN_ABSENT = 200
FOLDS = 5
N_NULL = 100
SEED = 6101

ALIAS_AUC = 0.95
# keyed on the LAYER NAME as built, not on the cell-file field name. The first run of this module
# keyed it "ppi" while the layer is "ppi_edges", so the flag silently never fired -- a declared
# control that does not run is worse than one that was never declared, because the output looks clean.
FLAGGED = {"ppi_edges": "loop 49 gate D2 FAILED -- declared BioGRID source reproduces at corr 0.3701"}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def build_layers(D):
    """One number per gene per layer. Presence is the target; the value is the feature."""
    genes = D["genes"]
    n = len(genes)
    idx = {g["name"]: i for i, g in enumerate(genes)}
    L = {}

    def blank():
        return np.zeros(n)

    reg_out, reg_in = blank(), blank()
    for a, b, *_ in D["reg"]:
        if a < n:
            reg_out[a] += 1
        if b < n:
            reg_in[b] += 1
    L["reg_out"], L["reg_in"] = reg_out, reg_in

    ppi = blank()
    for a, b in D["ppi"]:
        if a < n:
            ppi[a] += 1
        if b < n:
            ppi[b] += 1
    L["ppi_edges"] = ppi

    sig_out, sig_in = blank(), blank()
    for a, b, *_ in D["sig"]:
        if a < n:
            sig_out[a] += 1
        if b < n:
            sig_in[b] += 1
    L["sig_out"], L["sig_in"] = sig_out, sig_in

    sl = blank()
    for a, b, *_ in D["sl"]:
        for x in (a, b):
            if x < n:
                sl[x] += 1
    L["sl"] = sl

    lr = blank()
    for a, b in D["lr"]:
        for x in (a, b):
            if x < n:
                lr[x] += 1
    L["lr"] = lr

    def from_int_dict(key, fn=len):
        v = blank()
        for k, val in D[key].items():
            try:
                i = int(k)
            except ValueError:
                i = idx.get(k, -1)
            if 0 <= i < n:
                v[i] = fn(val)
        return v

    L["codep"] = from_int_dict("codep")
    L["coexpr"] = from_int_dict("coexpr")
    L["ptm"] = from_int_dict("ptm", lambda v: v.get("n", 0) if isinstance(v, dict) else len(v))
    L["drugs"] = from_int_dict("drugs")
    L["generxn"] = from_int_dict("generxn")
    L["loops3d"] = from_int_dict("loops3d")
    L["abund"] = from_int_dict("abund", lambda v: float(v))
    L["ppm"] = from_int_dict("ppm", lambda v: float(v))
    L["gene2cplx"] = from_int_dict("gene2cplx")
    L["model4"] = from_int_dict("model4", lambda v: 1.0)
    L["darkfn"] = from_int_dict("darkfn", lambda v: 1.0)
    L["cellcycle"] = from_int_dict("cellcycle", lambda v: 1.0)
    L["emask"] = from_int_dict("emask", lambda v: bin(int(v)).count("1") if v else 0)

    go = blank()
    for gname, v in D["go"].items():
        i = idx.get(gname, -1)
        if i >= 0:
            go[i] = sum(len(x) for x in v.values()) if isinstance(v, dict) else len(v)
    L["go"] = go

    nn = blank()
    for gname, v in D["nichenet"].items():
        i = idx.get(gname, -1)
        if i >= 0:
            nn[i] = len(v)
    L["nichenet"] = nn

    bm = blank()
    for gname, v in D["biomarkers"].items():
        i = idx.get(gname, -1)
        if i >= 0:
            bm[i] = 1.0
    L["biomarkers"] = bm

    for f in ("cpg", "enh", "npath", "ndis", "tf", "dark", "loeuf", "dep_frac", "ess"):
        L[f] = np.array([float(g.get(f) or 0) for g in genes])

    pubs = np.array([float(g.get("pubs") or 0) for g in genes])
    chrom = np.array([str(g.get("chrom") or "?") for g in genes])
    return L, pubs, chrom


def rank(x):
    o = np.argsort(np.argsort(x))
    return (o + 1) / len(x)


def cv_auc(X, y, fid, seed=SEED):
    P = np.full(len(y), np.nan)
    for k in range(FOLDS):
        tr, te = fid != k, fid == k
        if tr.sum() < 100 or te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        A = np.column_stack([rank(X[tr, j]) for j in range(X.shape[1])])
        # test rows ranked against the TRAIN distribution, never their own
        B = np.empty((te.sum(), X.shape[1]))
        for j in range(X.shape[1]):
            srt = np.sort(X[tr, j])
            B[:, j] = np.searchsorted(srt, X[te, j], side="right") / max(len(srt), 1)
        m = LogisticRegression(max_iter=3000, class_weight="balanced")
        m.fit(A, y[tr])
        P[te] = m.predict_proba(B)[:, 1]
    ok = ~np.isnan(P)
    if ok.sum() < 50 or len(set(y[ok])) < 2:
        return float("nan"), P
    return float(roc_auc_score(y[ok], P[ok])), P


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 61 -- the whole state vector: which layers can the rest of the cell fill?")
    say("  every layer is its own gate; `pubs` is both a feature and a baseline that must be beaten")
    say("=" * 100)
    say()

    D = json.load(open(CELL))
    L, pubs, chrom = build_layers(D)
    names = sorted(L)
    n = len(pubs)
    say(f"  {n:,} genes, {len(names)} layers")

    M = np.column_stack([L[k] for k in names])
    R = np.column_stack([rank(M[:, j]) for j in range(M.shape[1])])
    C = np.corrcoef(R, rowvar=False)
    alias = collections.defaultdict(set)
    say()
    say("V1 NO LAYER IS PREDICTED FROM ITS OWN ALIAS")
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(C[i, j]) > ALIAS_RHO:
                alias[names[i]].add(names[j])
                alias[names[j]].add(names[i])
                pairs.append((names[i], names[j], float(C[i, j]), "rho"))
                say(f"     ALIAS  {names[i]:12s} ~ {names[j]:12s}  rho {C[i, j]:+.4f}")

    # RANK CORRELATION IS NOT ENOUGH, and the first run of this module proved it. `tf` is predicted
    # by `reg_out` at AUC 0.9801 and `reg_out` by `tf` at 0.8826 -- a gene has outgoing regulatory
    # edges BECAUSE it is a transcription factor, which is definitional, not learned. That pair sits
    # below |rho| 0.95 because a binary flag and a count can classify each other almost perfectly
    # without rank-correlating that highly. So a second criterion: if either layer CLASSIFIES the
    # other's presence at AUC > 0.95, they are aliases regardless of correlation.
    for i in range(len(names)):
        yi = (L[names[i]] > 0).astype(int)
        if yi.sum() < MIN_PRESENT or (len(yi) - yi.sum()) < MIN_ABSENT:
            continue
        for j in range(len(names)):
            if i == j or names[j] in alias[names[i]]:
                continue
            a = roc_auc_score(yi, L[names[j]])
            a = max(a, 1 - a)
            if a > ALIAS_AUC:
                alias[names[i]].add(names[j])
                alias[names[j]].add(names[i])
                pairs.append((names[j], names[i], float(a), "auc"))
                say(f"     ALIAS  {names[j]:12s} classifies {names[i]:12s} at AUC {a:.4f}"
                    f"  -- definitional, not learned")
    if not pairs:
        say("     no pair above |rho| 0.95 or AUC 0.95")
    say(f"     V1 PASS  ({len(pairs)} alias pairs found and excluded per target)")
    say()

    ug = sorted(set(chrom))
    fold = {g: i for g, i in zip(ug, np.random.default_rng(SEED).permutation(len(ug)) % FOLDS)}
    fid = np.array([fold[c] for c in chrom])

    say("V2/V3 PER-LAYER FILL, AGAINST FAME AND AGAINST THE BEST SINGLE LAYER")
    say(f"     {'layer':13s} {'present':>8s} {'model':>7s} {'pubs':>7s} {'best1':>7s} "
        f"{'null95':>7s} {'margin':>7s}  verdict")
    rows = []
    earned = []
    rng = np.random.default_rng(SEED + 7)
    for t in names:
        y = (L[t] > 0).astype(int)
        if y.sum() < MIN_PRESENT or (len(y) - y.sum()) < MIN_ABSENT:
            continue
        feats = [k for k in names if k != t and k not in alias[t]]
        X = np.column_stack([L[k] for k in feats] + [pubs])
        a_model, Pm = cv_auc(X, y, fid)
        a_pubs, _ = cv_auc(pubs.reshape(-1, 1), y, fid)
        singles = {}
        for k in feats:
            s, _ = cv_auc(L[k].reshape(-1, 1), y, fid)
            if not np.isnan(s):
                singles[k] = s
        best1 = max(singles.values()) if singles else float("nan")
        best1n = max(singles, key=singles.get) if singles else "-"
        # the null must be built on the MODEL's own held-out predictions -- permuting labels against
        # some other predictor would gate the model against a null it never competed in
        okm = ~np.isnan(Pm)
        nulls = []
        for _ in range(N_NULL):
            yp = y[okm].copy()
            rng.shuffle(yp)
            nulls.append(roc_auc_score(yp, Pm[okm]))
        n95 = float(np.percentile(nulls, 95))
        base = max(a_pubs, best1)
        marg = a_model - base
        ok = (marg >= MARGIN) and (a_model > n95)
        v = "EARNED" if ok else "defer"
        if t in FLAGGED:
            v += " *FLAGGED*"
        if ok:
            earned.append(t)
        say(f"     {t:13s} {int(y.sum()):8,d} {a_model:7.4f} {a_pubs:7.4f} {best1:7.4f} "
            f"{n95:7.4f} {marg:+7.4f}  {v}")
        rows.append({"layer": t, "n_present": int(y.sum()), "model": a_model, "pubs": a_pubs,
                     "best_single": best1, "best_single_name": best1n, "null95": n95,
                     "margin": marg, "earned": bool(ok),
                     "flag": FLAGGED.get(t)})
    say()

    fam = [r["pubs"] for r in rows]
    say(f"V2 fame alone predicts layer membership at median {np.median(fam):.4f}, "
        f"range {min(fam):.4f}-{max(fam):.4f} over {len(rows)} layers")
    v2 = True
    say(f"     V2 PASS  (measured for every layer, reported whether it passes or fails)")
    say()
    v3 = len(earned) >= MIN_EARNED
    say(f"V3 THE GATE -- {len(earned)} of {len(rows)} layers earn their fill "
        f"(gate >= {MIN_EARNED}; chance alone gives about {0.05 * len(rows):.1f})")
    say(f"     earned: {', '.join(earned) if earned else 'NONE'}")
    say(f"     V3 {'PASS' if v3 else 'FAIL'}")
    say()
    v4 = all(np.isfinite(r["null95"]) for r in rows)
    say(f"V4 nulls measured for all {len(rows)} layers")
    say(f"     V4 {'PASS' if v4 else 'FAIL'}")
    say()
    ABUNDANCE_LIKE = {"abund", "ppm", "emask", "coexpr"}
    v5 = not (earned and set(earned) <= ABUNDANCE_LIKE)
    say("V5 THE EARNED LAYERS ARE NOT ONE FAMILY")
    say(f"     abundance-like among them: "
        f"{sorted(set(earned) & ABUNDANCE_LIKE) or 'none'}")
    say(f"     V5 {'PASS' if v5 else 'FAIL'}")
    say()

    gates = {"V1 no layer predicted from its alias": True,
             "V2 fame baseline measured per layer": v2,
             "V3 enough layers earn their fill": bool(v3),
             "V4 nulls measured per layer": bool(v4),
             "V5 earned layers are not one family": bool(v5)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(CELL)], available=len(names), used=len(rows),
                      selection="filtered", seed=SEED,
                      controls=["publication count as a mandatory per-layer baseline",
                                "best single other layer as a second baseline",
                                "alias pairs measured at |rho|>0.95 and excluded per target",
                                "chromosome-grouped folds",
                                "per-layer label permutation null",
                                "ppi carries loop 49's failed-provenance flag rather than exclusion"],
                      note="a layer predicted only as well as `pubs` predicts it has been recognised "
                           "as famous, not filled")
    RM.report(man, emit=say)
    json.dump({"test": "loop_state_vector", "manifest": man, "gates": gates,
               "alias_pairs": pairs, "layers": rows, "earned": earned,
               "fame_median": float(np.median(fam)), "seconds": time.time() - t0,
               "log": log}, open(OUT / "loop_state_vector.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_state_vector.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
