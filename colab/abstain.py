"""BUILD 2 of 5 -- MAGNITUDE FIRST, WITH THE RIGHT TO SAY NOTHING.

THE MEASURED PROBLEM. 2,982 of 5,120 perturbations (58.2%) produce ZERO specific movers. Every predictor in this
project emits a ranked gene list unconditionally, so on the majority of inputs it is wrong by construction -- not
wrong in its ordering, wrong in having answered at all. No improvement to the content stage fixes that, because the
content stage is never asked whether there is content to predict.

THE FIX IS AN ORDERING, NOT A MODEL. Ask the cheap question first:

    stage 1   MAGNITUDE   will this perturbation move anything at all?     <- THIS FILE
    stage 2   CLASS       shared stress core, specific arm, or both?       <- build 3
    stage 3   CONTENT     which genes                                      <- exists (response_basis.py)

Stage 1's output is not a gene list. It is a LICENCE for the content stage to speak.

HOW THIS DIFFERS FROM magnitude.py, WHICH ALREADY EXISTS. That file regresses how big a response will be
(held-out Spearman 0.558 on n_movers) on the old top-250-truncated readout. Correlation is the wrong currency for
this decision: a model can rank magnitudes well and still have no threshold at which its "nothing happens" call is
trustworthy. What the pipeline needs is a CALIBRATED PROBABILITY and a defensible cut. Same problem family,
different question, different scoring -- and this one runs on the full-transcriptome gwps rebuild.

WHY THIS IS WORTH DOING, STATED BEFORE MEASURING IT. The base rate is 58.2% silent, so "always abstain" already
has 58.2% precision on the silent call at 100% coverage, and is useless. The bar from DESIGN.md sec 4 is precision
AT USEFUL COVERAGE: among perturbations the model calls silent, >= 90% must really have fewer than 5 specific
movers. Reaching 90% by abstaining on eleven safe cases is not a pass, so the whole precision-coverage curve is
printed and that dodge stays visible.

FEATURES, AND WHAT EACH IS SUSPECTED OF. Every one is knowable before the perturbation is measured.

    cascade breadth        n_starved_rxn from reaction_ablation -- independently validated (z = -6.88)
    ablation verdicts      how many reactions this gene STOPS versus merely buffers
    essentiality           DepMap K562 dependency: a gene the cell does not need is a gene whose loss does nothing
    abundance / expression PaxDb ppm, and K562 RPKM + mRNA half-life from the SLAM-seq decay table
    connectivity           PPI degree, regulon size, co-expression and co-dependency degree, complexes, pathways
    loeuf                  population constraint
    detected in K562       whether the gene appears in the readout's 8,246-gene vocabulary at all
    pubs                   literature count -- INCLUDED DELIBERATELY AS A SUSPECT. If a "how well studied is it"
                           counter carries the model, the model has learned annotation bias rather than biology,
                           and the single-feature table will show it.

CONTROLS, ALL REPORTED NEXT TO THE RESULT (DESIGN.md sec 7).
    always-silent          the trivial abstainer; its precision is the base rate, by definition
    shuffled labels        the same classifier on permuted truth
    single features        each feature alone, so "the model" cannot quietly be one column

SPLIT. Perturbations are divided in two by a hash of the gene symbol -- deterministic, no seed to have chosen --
and it is run BOTH WAYS, so neither half is the one that happened to work.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
TAU, TIDE_FRAC, MIN_SPEC = 1.0, 0.05, 5
TARGET_PRECISION = 0.90          # pre-registered in DESIGN.md sec 4
MIN_COVERAGE = 0.10              # and a pass must abstain on at least this share, or it is a dodge


def readout():
    """The gwps rebuild. Asserts finite -- infinities in this matrix passed the |z|>=1 mover test once already."""
    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos, genes, M = list(z["kos"]), list(z["genes"]), z["M"]
    bad = int((~np.isfinite(M)).sum())
    assert bad == 0, f"NON-FINITE readout: {bad:,} cells -- rebuild it, do not sanitise it here"
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC
    spec = ((A >= TAU) & ~tide).sum(1)
    print(f"readout: {len(kos):,} perturbations x {len(genes):,} genes; tide {int(tide.sum()):,}; "
          f"silent (<{MIN_SPEC} specific movers) {int((spec < MIN_SPEC).sum()):,} "
          f"= {100*(spec < MIN_SPEC).mean():.1f}%")
    gi = {g: i for i, g in enumerate(genes)}
    self_hits = sum(1 for k, r in zip(kos, A) if k in gi and r[gi[k]] >= TAU and not tide[gi[k]])
    print(f"  the knocked-out gene's own transcript is a specific mover in {self_hits:,} perturbations -- at most "
          f"1 of the {MIN_SPEC} needed, so it cannot manufacture an ACTIVE label on its own")
    return kos, genes, spec, set(genes)


def features():
    """gene -> feature dict, from a priori sources only. Every join prints its size and the big ones assert."""
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    N = len(names)
    F = {g["name"]: {"ess": float(g.get("ess") or 0),
                     "dep_frac": float(g.get("dep_frac") or 0),
                     "loeuf": float(g["loeuf"] if g.get("loeuf") is not None else 1.0),
                     "tf": float(g.get("tf") or 0),
                     "ppi_deg": float(g.get("ppi") or 0),
                     "npath": float(g.get("npath") or 0),
                     "pubs": float(g.get("pubs") or 0),
                     "dark": float(g.get("dark") or 0)} for g in D["genes"]}
    print(f"features: {len(F):,} genes from cell_complete")

    ppm = {names[int(k)]: float(v) for k, v in D["ppm"].items()
           if k.isdigit() and int(k) < N and v}
    for g, v in ppm.items():
        if g in F:
            F[g]["log_ppm"] = float(np.log10(v + 1e-3))
    print(f"  PaxDb abundance: {sum('log_ppm' in v for v in F.values()):,} genes")

    # K562 transcript level and mRNA half-life -- the closest thing to "is this gene even on in this cell line"
    nr = 0
    try:
        for r in csv.DictReader(open(SP / "rnadecay" / "AvgKdegs.csv")):
            if r.get("cell_line") != "K562":
                continue
            g = r.get("feature_ID")
            if g not in F:
                continue
            try:
                F[g]["log_rpkm"] = float(np.log10(float(r["avg_RPKM_total"]) + 0.1))
                F[g]["log_halflife"] = float(np.log10(float(r["avg_halflife"]) + 0.01))
                nr += 1
            except (ValueError, KeyError, TypeError):
                pass
    except FileNotFoundError:
        print("  K562 RPKM/half-life table absent -- those two features will be constant")
    print(f"  K562 RPKM + half-life: {nr:,} genes")

    sg, ug = collections.Counter(), collections.Counter()
    for e in D.get("reg", []):
        try:
            s, w = int(e[0]), (int(e[2]) if len(e) > 2 else 0)
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= s < N:
            (sg if w else ug)[names[s]] += 1
    for g in F:
        F[g]["reg_signed_out"] = float(sg.get(g, 0))
        F[g]["reg_unsigned_out"] = float(ug.get(g, 0))
    print(f"  regulatory out-degree: {len(sg):,} genes signed, {len(ug):,} unsigned "
          f"(kept apart: the unsigned layer overlaps real K562 regulation at chance)")

    try:
        trr = json.load(open(OUT / "trrust_regulon.json"))["tf_targets"]
        for g in F:
            F[g]["regulon_size"] = float(len(trr.get(g) or ()))
        print(f"  curated regulon size: {sum(1 for g in F if F[g]['regulon_size'] > 0):,} TFs")
    except (FileNotFoundError, KeyError):
        print("  curated regulon absent")

    # co-expression / co-dependency degree are INDEX-keyed in cell_complete -- the index-space bug that silently
    # zeroed 191,447 PPI edges lived exactly here, so the key type is checked rather than assumed
    for key, col in (("coexpr", "coexpr_deg"), ("codep", "codep_deg")):
        d = D.get(key) or {}
        n = 0
        for k, v in d.items():
            try:
                i = int(k)
            except (TypeError, ValueError):
                continue
            if 0 <= i < N and names[i] in F:
                F[names[i]][col] = float(len(v) if isinstance(v, (list, set, tuple)) else 1)
                n += 1
        print(f"  {key} degree: {n:,} genes")
        assert n > 100, f"{key} JOIN EMPTY -- index space disagrees, refusing to continue"

    ncplx = collections.Counter()
    for k, v in (D.get("gene2cplx") or {}).items():
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        if 0 <= i < N:
            ncplx[names[i]] = len(v) if isinstance(v, (list, set, tuple)) else 1
    for g in F:
        F[g]["n_complex"] = float(ncplx.get(g, 0))
    print(f"  complex membership: {len(ncplx):,} genes")

    ab = json.load(open(OUT / "reaction_ablation.json"))
    stops, buf, red, breadth, nrxn = (collections.Counter(), collections.Counter(), collections.Counter(),
                                      collections.defaultdict(list), collections.Counter())
    for r in ab["rows"]:
        g = r.get("removed")
        if not g or g not in F or not r.get("addressable"):
            continue
        nrxn[g] += 1
        v = r.get("verdict")
        if v == "STOPS":
            stops[g] += 1
            breadth[g].append(float(r.get("n_starved_rxn") or 0))
        elif v == "BUFFERED":
            buf[g] += 1
        elif v == "REDUNDANT":
            red[g] += 1
    for g in F:
        b = breadth.get(g) or [0.0]
        F[g].update({"n_stops": float(stops.get(g, 0)), "n_buffered": float(buf.get(g, 0)),
                     "n_redundant": float(red.get(g, 0)), "n_rxn": float(nrxn.get(g, 0)),
                     "cascade_max": float(max(b)), "cascade_sum": float(sum(b))})
    print(f"  reaction ablation: {len(nrxn):,} genes over {len(ab['rows']):,} ablation rows")
    assert len(nrxn) > 1000, f"ABLATION JOIN TOO SMALL ({len(nrxn)}) -- name spaces disagree"
    return F


COLS = ["ess", "dep_frac", "loeuf", "tf", "ppi_deg", "npath", "pubs", "dark", "log_ppm", "log_rpkm",
        "log_halflife", "reg_signed_out", "reg_unsigned_out", "regulon_size", "coexpr_deg", "codep_deg",
        "n_complex", "n_stops", "n_buffered", "n_redundant", "n_rxn", "cascade_max", "cascade_sum", "detected"]
LOGGED = {"ppi_deg", "npath", "pubs", "reg_signed_out", "reg_unsigned_out", "regulon_size", "coexpr_deg",
          "codep_deg", "n_complex", "n_stops", "n_buffered", "n_redundant", "n_rxn", "cascade_max", "cascade_sum"}


def matrix(kos, F, detected):
    X, keep, used = [], [], []
    for i, k in enumerate(kos):
        f = F.get(k)
        if f is None:
            continue
        X.append([(1.0 if k in detected else 0.0) if c == "detected" else float(f.get(c, 0.0)) for c in COLS])
        keep.append(i)
        used.append(k)
    X = np.asarray(X, float)
    for j, c in enumerate(COLS):                 # counts are heavy-tailed; one ribosomal gene must not set the scale
        if c in LOGGED:
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    assert np.isfinite(X).all(), "NON-FINITE feature matrix"
    return X, np.asarray(keep), used


def fit_predict(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import make_pipeline
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def auc(y, s):
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (1 - y).sum()))


def abstention_curve(p_active, y_silent):
    """p_active = P(this perturbation does something). Abstain on the lowest-scoring fraction; report how pure
    that abstained set is. Coverage is what stops "90% precision on eleven cases" from counting as a pass."""
    o = np.argsort(p_active, kind="mergesort")
    ys = np.asarray(y_silent).astype(int)[o]
    n = np.arange(1, len(ys) + 1)
    return n / len(ys), np.cumsum(ys) / n


def reach(cov, prec, target=TARGET_PRECISION):
    ok = np.where(prec >= target)[0]
    return float(cov[ok[-1]]) if len(ok) else 0.0


def lift_ci(p_active, y_silent, cov_at, n_boot=2000, seed=0):
    """Bootstrap CI on the thing that actually matters: precision of the silent call at a FIXED coverage, MINUS
    the base rate. With 83.6% of perturbations silent, an unqualified "90% precision" is only 6 points above
    answering 'nothing happens' every time, so the lift is reported with its uncertainty, not the raw precision."""
    p_active, y_silent = np.asarray(p_active), np.asarray(y_silent).astype(int)
    rng = np.random.default_rng(seed)
    n, k = len(y_silent), max(int(cov_at * len(y_silent)), 1)
    out = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        o = np.argsort(p_active[i], kind="mergesort")[:k]
        out[b] = y_silent[i][o].mean() - y_silent[i].mean()
    o = np.argsort(p_active, kind="mergesort")[:k]
    return float(y_silent[o].mean() - y_silent.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# Feature subsets. The point of the third is the honest question the single-feature table forces: once the
# MEASURED DepMap dependency of the gene is removed, is there anything left, or is the gate a DepMap lookup?
SUBSETS = {
    "all features": lambda c: True,
    "no DepMap (drop ess, dep_frac)": lambda c: c not in ("ess", "dep_frac"),
    "no DepMap, no expression": lambda c: c not in ("ess", "dep_frac", "log_rpkm", "log_ppm", "log_halflife"),
}


def run_label(X, used, y_active, tag, rng, per_feature=False):
    """Two-way hash split, run both ways. Returns the fold table for one label definition."""
    y_silent = 1 - y_active
    h = np.array([int(hashlib.md5(n.encode()).hexdigest(), 16) % 2 for n in used])
    print(f"\n{'='*84}\nLABEL: {tag}   ({int(y_silent.sum()):,} silent / {len(y_silent):,} = "
          f"{y_silent.mean():.1%} base rate)\n{'='*84}")
    out = {"base_rate": float(y_silent.mean()), "folds": {}}
    for fold in (0, 1):
        tr, te = h != fold, h == fold
        f = {}
        for name, sel in SUBSETS.items():
            cols = [j for j, c in enumerate(COLS) if sel(c)]
            p = fit_predict(X[tr][:, cols], y_active[tr], X[te][:, cols])
            cov, prec = abstention_curve(p, y_silent[te])
            l50, lo, hi = lift_ci(p, y_silent[te], 0.50)
            f[name] = {"auc": auc(y_active[te], p), "coverage_at_p90": reach(cov, prec),
                       "prec_at_25": float(prec[max(int(.25*len(cov))-1, 0)]),
                       "prec_at_50": float(prec[max(int(.50*len(cov))-1, 0)]),
                       "lift_at_50": l50, "lift_ci": [lo, hi], "n_features": len(cols)}
        psh = fit_predict(X[tr], rng.permutation(y_active[tr]), X[te])
        csh, prsh = abstention_curve(psh, y_silent[te])
        lsh, slo, shi = lift_ci(psh, y_silent[te], 0.50)
        f["SHUFFLED labels"] = {"auc": auc(y_active[te], psh), "coverage_at_p90": reach(csh, prsh),
                                "prec_at_25": float(prsh[max(int(.25*len(csh))-1, 0)]),
                                "prec_at_50": float(prsh[max(int(.50*len(csh))-1, 0)]),
                                "lift_at_50": lsh, "lift_ci": [slo, shi], "n_features": len(COLS)}
        base = float(y_silent[te].mean())
        print(f"\nfold {fold}  train {int(tr.sum()):,} -> test {int(te.sum()):,}   "
              f"always-silent precision {base:.3f}")
        print(f"  {'':32s} {'AUC':>7s} {'p@25%':>7s} {'p@50%':>7s} {'lift@50%':>9s} {'95% CI':>18s} "
              f"{'cov@p90':>8s}")
        for name, r in f.items():
            print(f"  {name:32s} {r['auc']:>7.4f} {r['prec_at_25']:>7.3f} {r['prec_at_50']:>7.3f} "
                  f"{r['lift_at_50']:>+9.4f} {'[%+.4f,%+.4f]' % tuple(r['lift_ci']):>18s} "
                  f"{r['coverage_at_p90']*100:>7.1f}%")
        out["folds"][f"fold{fold}"] = f

        if per_feature and fold == 0:
            single = []
            for j, c in enumerate(COLS):
                pj = fit_predict(X[tr][:, [j]], y_active[tr], X[te][:, [j]])
                cj, prj = abstention_curve(pj, y_silent[te])
                single.append((c, auc(y_active[te], pj), reach(cj, prj)))
            single.sort(key=lambda t: -(t[1] if np.isfinite(t[1]) else 0))
            print("\n    single feature alone           AUC   90%-coverage")
            for c, a1, r1 in single:
                print(f"      {c:<26s} {a1:.4f}   {r1*100:>5.1f}%")
            out["single_features"] = [{"feature": c, "auc": a1, "coverage_at_p90": r1}
                                      for c, a1, r1 in single]
    return out


def main():
    kos, genes, spec, detected = readout()
    F = features()
    X, keep, used = matrix(kos, F, detected)
    s = spec[keep]
    print(f"\njoin: {len(used):,} of {len(kos):,} perturbations have features ({100*len(used)/len(kos):.1f}%)")
    print(f"  specific movers: {int((s == 0).sum()):,} produce EXACTLY ZERO ({(s==0).mean():.1%}), "
          f"{int((s < MIN_SPEC).sum()):,} produce fewer than {MIN_SPEC} ({(s<MIN_SPEC).mean():.1%})")
    assert len(used) > 1000, "PERTURBATION/FEATURE JOIN TOO SMALL -- refusing to continue"

    rng = np.random.default_rng(0)
    # BOTH labels are run, because DESIGN.md is internally inconsistent and it matters which one is meant. Its
    # motivating statistic is "58.2% produce ZERO specific movers"; its acceptance test says "< 5 movers". The
    # second has an 83.6% base rate, which puts the 90% bar only ~6 points above answering "nothing" every time.
    # The zero-mover label is the honest version of the same test and is reported as the primary result.
    R = {"pre-registered (<%d movers)" % MIN_SPEC: run_label(X, used, (s >= MIN_SPEC).astype(int),
                                                             "silent = fewer than %d specific movers "
                                                             "(the pre-registered label)" % MIN_SPEC, rng,
                                                             per_feature=True),
         "harder (zero movers)": run_label(X, used, (s > 0).astype(int),
                                           "silent = EXACTLY ZERO specific movers (the harder label, and the one "
                                           "DESIGN.md's 58.2% actually refers to)", rng)}

    print("\n" + "=" * 84)
    print("PRE-REGISTERED TEST (DESIGN.md sec 4): among perturbations called silent, >= "
          f"{TARGET_PRECISION:.0%} must really be silent, at coverage worth having.")
    verdicts = {}
    for lab, r in R.items():
        f0, f1 = r["folds"]["fold0"]["all features"], r["folds"]["fold1"]["all features"]
        s0, s1 = r["folds"]["fold0"]["SHUFFLED labels"], r["folds"]["fold1"]["SHUFFLED labels"]
        n0, n1 = r["folds"]["fold0"]["no DepMap (drop ess, dep_frac)"], \
            r["folds"]["fold1"]["no DepMap (drop ess, dep_frac)"]
        cov = float(np.mean([f0["coverage_at_p90"], f1["coverage_at_p90"]]))
        scov = float(np.mean([s0["coverage_at_p90"], s1["coverage_at_p90"]]))
        lift = float(np.mean([f0["lift_at_50"], f1["lift_at_50"]]))
        pos = f0["lift_ci"][0] > 0 and f1["lift_ci"][0] > 0
        v = "PASS" if (cov >= MIN_COVERAGE and cov > scov and pos) else "FAIL"
        verdicts[lab] = v
        print(f"\n  {lab}:  base rate {r['base_rate']:.3f}")
        print(f"    AUC {np.mean([f0['auc'], f1['auc']]):.4f} (shuffled {np.mean([s0['auc'], s1['auc']]):.4f})"
              f" | coverage at {TARGET_PRECISION:.0%} precision {cov*100:.1f}% (shuffled {scov*100:.1f}%)")
        print(f"    lift over always-silent at 50% coverage {lift:+.4f}, "
              f"both folds' CI exclude zero: {pos}")
        print(f"    with DepMap dependency REMOVED: AUC {np.mean([n0['auc'], n1['auc']]):.4f}, "
              f"lift {np.mean([n0['lift_at_50'], n1['lift_at_50']]):+.4f}")
        print(f"    VERDICT: {v}")
    print("=" * 84)

    json.dump({"n": len(used), "n_zero": int((s == 0).sum()), "n_lt_min": int((s < MIN_SPEC).sum()),
               "target_precision": TARGET_PRECISION, "min_coverage": MIN_COVERAGE,
               "verdicts": verdicts, "labels": R},
              open(OUT / "abstain.json", "w"), indent=1)
    print(f"\n  -> {OUT/'abstain.json'}")


if __name__ == "__main__":
    main()
