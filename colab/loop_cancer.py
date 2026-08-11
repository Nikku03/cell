"""CANCER DRIVERS -- the last row with no pipeline, and two confounds that decide it.

WHERE THIS COMES FROM.  `cancer` is the only one of the six question types left with no module that
consumes its slot and records a controlled result: 839 items encoded, nothing that runs. This gives it a
held-out target.

WHAT IS ADDED.  IntOGen's 2024-06 driver compendium: 633 genes called as drivers across 4,478
cohort-by-gene findings, with a role (loss-of-function or activating) attached. That is a real,
externally curated label this repository has never been scored against.

THE CLAIM UNDER TEST.  A driver gene is one whose alteration changes what a cell does. The cell model
carries several estimates of how much a gene matters -- metabolic cost of losing it, essentiality across
lines, position in a functional neighbourhood, DNA torsion at its promoter. If any of those track
driver status, the model knows something about cancer.

THE TWO CONFOUNDS THAT DECIDE IT, and both are severe enough to produce the whole result on their own.
    GENE LENGTH   Driver calling is a mutation-recurrence test. A long gene accumulates more mutations
                  by chance, so length alone predicts driver status without any biology whatsoever.
    STUDY BIAS    A gene with thousands of papers gets sequenced, annotated and called more often.
                  cell_complete carries a `pubs` count per gene, so unusually this one CAN be measured
                  here rather than merely worried about.
Both are partialled out, and both are reported as predictors in their own right so their strength is
visible instead of hidden inside a comparison.

PREDECLARED, before any number:

    K1 JOIN       at least 200 IntOGen drivers are present among genes this model can score.
    K2 SIGNAL     the best model-derived score separates drivers from non-drivers above the 95th
                  percentile of 200 label shuffles.
    K3 BEATS LENGTH AND CITATIONS   it survives partialling out gene length and publication count.
                  THIS IS THE REAL GATE. fails -> the model is rediscovering that long, well-studied
                  genes get called drivers, which needs no cell model.
    K4 BEATS LOOKUPS   it beats LOEUF and protein abundance, two columns already sitting in
                  cell_complete.

CONTROLS: 200 label shuffles; gene length and publication count partialled out AND reported as
standalone predictors; LOEUF and abundance lookups; and the loss-of-function subset scored separately,
since an activating driver is not something a deletion-based model should be expected to see.

-> outputs/loop_cancer.json
"""
import collections
import csv
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from cell_loop import auc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path(os.environ.get("CELL_SCRATCH",
          "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"
INTOGEN = SC / "intogen.zip"
SEED = 1904
N_SHUFFLE = 200
GATE_JOIN = 200


def partial_auc(score, label, covars, seed=SEED):
    """AUC of the part of `score` that the covariates cannot explain.

    Regress the score on the confounds, keep the residual, score that. If length and citation count
    account for the signal, the residual carries nothing and the AUC falls to chance -- which is the
    outcome this gate exists to be able to report."""
    from sklearn.linear_model import LinearRegression
    X = np.asarray(covars, float)
    v = np.asarray(score, float)
    ok = np.isfinite(v) & np.all(np.isfinite(X), axis=1)
    if ok.sum() < 50:
        return float("nan"), 0
    r = v[ok] - LinearRegression().fit(X[ok], v[ok]).predict(X[ok])
    return auc(r, np.asarray(label)[ok]), int(ok.sum())


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("CANCER DRIVERS -- the last row with no pipeline")
    say("=" * 100)
    say("  Two confounds decide this and both could produce the whole result alone: long genes collect")
    say("  more mutations by chance, and well-studied genes get called more often. Both are partialled")
    say("  out and both are reported as standalone predictors.")

    z = zipfile.ZipFile(INTOGEN)
    nm = [x for x in z.namelist() if x.endswith("Compendium_Cancer_Genes.tsv")][0]
    rows = list(csv.DictReader(io.StringIO(z.read(nm).decode("utf-8", "ignore")), delimiter="\t"))
    drivers = collections.Counter(r["SYMBOL"] for r in rows)
    lof = {r["SYMBOL"] for r in rows if r.get("ROLE") == "LoF"}
    say(f"\n  IntOGen 2024-06: {len(rows):,} cohort-by-gene findings over {len(drivers):,} distinct "
        f"driver genes; {len(lof):,} with a loss-of-function role")

    D = json.load(open(CELL))
    genes = D["genes"]
    ppm = {int(k): float(v) for k, v in D["ppm"].items()}

    cost = {}
    lm = OUT / "loop_medium.json"
    if lm.exists():
        for r in json.load(open(lm)).get("scores", []):
            if r.get("medium") is not None and np.isfinite(r["medium"]):
                cost[r["sym"]] = float(r["medium"])

    tors = {}
    lrc = OUT / "loop_real_chromatin.json"

    rec = []
    for i, g in enumerate(genes):
        s = g["name"]
        ln = (g.get("gene_end") or 0) - (g.get("gene_start") or 0)
        if ln <= 0:
            continue
        rec.append({"sym": s, "y": 1 if s in drivers else 0, "lof": 1 if s in lof else 0,
                    "log_len": float(np.log10(ln)),
                    "log_pubs": float(np.log10((g.get("pubs") or 0) + 1)),
                    "ess": g.get("ess"), "loeuf": g.get("loeuf"),
                    "cost": cost.get(s, np.nan),
                    "ppm": float(np.log10(ppm.get(i, 0.0) + 1e-3))})
    T = pd.DataFrame(rec)
    y = T.y.to_numpy()
    n_drv = int(y.sum())
    say(f"  {len(T):,} genes with coordinates; {n_drv:,} of them IntOGen drivers")
    k1 = bool(n_drv >= GATE_JOIN)
    say(f"  K1 JOIN (>= {GATE_JOIN} drivers)   {'PASS' if k1 else 'FAIL'}")
    if not k1:
        json.dump({"test": "loop_cancer", "K1": False, "n_drivers": n_drv, "log": log},
                  open(OUT / "loop_cancer.json", "w"), indent=2)
        return 1

    say(f"\n  THE CONFOUNDS, measured before anything is claimed:")
    a_len = auc(T.log_len, y)
    a_pub = auc(T.log_pubs, y)
    say(f"    gene length alone        AUC {a_len:.4f}")
    say(f"    publication count alone  AUC {a_pub:.4f}   <- study bias, measurable here because "
        f"cell_complete carries `pubs`")

    cov = np.c_[T.log_len, T.log_pubs]
    cands = {"essentiality (model)": "ess", "metabolic cost (loop 4)": "cost",
             "LOEUF (lookup)": "loeuf", "protein abundance (lookup)": "ppm"}
    res = {}
    for label, col in cands.items():
        v = T[col].astype(float)
        m = v.notna().to_numpy()
        if m.sum() < 50:
            continue
        sgn = -1.0 if col == "loeuf" else 1.0
        raw = auc(sgn * v[m], y[m])
        par, n = partial_auc(sgn * v, y, cov)
        res[label] = {"raw": raw, "partial": par, "n": int(m.sum())}
    say(f"\n  {'predictor':<32}{'raw':>9}{'partial':>10}{'n':>9}")
    for label, v in res.items():
        say(f"  {label:<32}{v['raw']:>9.4f}{v['partial']:>10.4f}{v['n']:>9,}")
    say("    'partial' is the AUC of what remains after length and citations are regressed out.")

    model_keys = [k for k in res if "lookup" not in k]
    best = max(model_keys, key=lambda k: abs(res[k]["partial"] - 0.5)) if model_keys else None
    if best is None:
        say("\n  no model-derived predictor available; nothing to gate.")
        return 1
    col = cands[best]
    v = T[col].astype(float)
    m = v.notna().to_numpy()
    null = np.array([auc(v[m], rng.permutation(y[m])) for _ in range(N_SHUFFLE)])
    lo, hi = np.percentile(null, [2.5, 97.5])
    k2 = bool(res[best]["raw"] < lo or res[best]["raw"] > hi)
    say(f"\n  K2 SIGNAL -- best model predictor '{best}': raw {res[best]['raw']:.4f} against a shuffled "
        f"null of [{lo:.4f}, {hi:.4f}]   K2 {'PASS' if k2 else 'FAIL'}")

    nullp = []
    for _ in range(N_SHUFFLE):
        a, _n = partial_auc(v, rng.permutation(y), cov)
        if np.isfinite(a):
            nullp.append(a)
    nullp = np.array(nullp)
    plo, phi = np.percentile(nullp, [2.5, 97.5])
    k3 = bool(res[best]["partial"] < plo or res[best]["partial"] > phi)
    say(f"\n  K3 BEATS LENGTH AND CITATIONS -- partial {res[best]['partial']:.4f} against "
        f"[{plo:.4f}, {phi:.4f}]   K3 {'PASS' if k3 else 'FAIL'}")
    if not k3:
        say("    With gene length and publication count removed, the model's score no longer separates")
        say("    drivers from non-drivers. What looked like cancer knowledge was the recurrence test's")
        say("    own denominator and the literature's attention.")

    lookups = {k: v for k, v in res.items() if "lookup" in k}
    bl = max(lookups, key=lambda k: abs(lookups[k]["partial"] - 0.5)) if lookups else None
    k4 = bool(bl is None or abs(res[best]["partial"] - 0.5) > abs(lookups[bl]["partial"] - 0.5))
    if bl:
        say(f"\n  K4 BEATS LOOKUPS -- vs {bl}: |{res[best]['partial']:.4f}-0.5| against "
            f"|{lookups[bl]['partial']:.4f}-0.5|   {'PASS' if k4 else 'FAIL'}")

    # the LoF subset -- a deletion-based model has no business predicting activating drivers
    sub = T[(T.y == 0) | (T.lof == 1)]
    ysub = sub.y.to_numpy()
    vs = sub[col].astype(float)
    ms = vs.notna().to_numpy()
    a_lof = auc(vs[ms], ysub[ms]) if ms.sum() > 50 else float("nan")
    say(f"\n  loss-of-function drivers only ({int(sub.y.sum()):,} positives): raw AUC {a_lof:.4f}")
    say("    reported separately because an ACTIVATING driver is a gain of function, and nothing in")
    say("    this model represents one -- it only knows how to remove things.")

    say("\n" + "=" * 100)
    gates = {"K1 join": k1, "K2 signal": k2, "K3 beats length and citations": k3, "K4 beats lookups": k4}
    for k, vv in gates.items():
        say(f"  {k:<36}{'PASS' if vv else 'FAIL'}")
    if k2 and k3:
        say("  THE MODEL KNOWS SOMETHING ABOUT DRIVERS that gene length and citation count do not")
        say("  explain. `cancer` now has a slot, a pipeline and a controlled score that came back")
        say("  positive on the gate that could have killed it.")
    elif k2:
        say("  The signal is real and is explained by length and citations. The cell model added")
        say("  nothing to a mutation-recurrence denominator and the literature's attention.")
    else:
        say("  No signal. `cancer` now has data and a pipeline, and the pipeline says no.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(INTOGEN), str(CELL), str(OUT / "loop_medium.json")],
                      available=len(genes), used=len(T), selection="filtered", seed=SEED,
                      controls=[f"{N_SHUFFLE} label shuffles",
                                "gene length and publication count partialled out",
                                "length and citations as standalone predictors",
                                "LOEUF and abundance lookups",
                                "loss-of-function subset scored separately"],
                      note="IntOGen 2024-06 driver compendium as the held-out label")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_cancer", "manifest": man, "gates": gates,
               "n_drivers": n_drv, "auc_length": a_len, "auc_pubs": a_pub,
               "predictors": res, "best_model": best, "auc_lof_only": a_lof,
               "null": {"lo": float(lo), "hi": float(hi)},
               "null_partial": {"lo": float(plo), "hi": float(phi)}, "log": log},
              open(OUT / "loop_cancer.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_cancer.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
