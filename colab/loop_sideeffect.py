"""SIDE EFFECTS -- the row that had no data at all, and the confound that decides it.

WHERE THIS COMES FROM.  capability_audit ranked `side effect` as the second-worst gap: 48 genes carrying
disease associations, and no compound-to-adverse-event table anywhere in this repository. It was also
the audit's MUST-FAIL calibration precisely because nothing here could possibly answer it. Building this
means the calibration has to move again, and that is the point of building it.

WHAT IS ADDED.  SIDER 4.1 -- side effects read off drug package inserts, 1,430 compounds against MedDRA
preferred terms. Joined by drug name to cell_complete's `drugs` block, 640 compounds carry both a
side-effect list and a set of protein targets in this model. Median 85 side effects, median 4 targets.

THE CLAIM UNDER TEST.  A drug perturbs the proteins it binds. If those proteins are ones the cell
depends on -- metabolically costly to lose, essential across cell lines -- the perturbation should reach
more processes and produce more adverse events. So the model's own estimate of what a gene costs should
predict how many side effects a drug that hits it has.

THE CONFOUND THAT DECIDES THIS, and it is not subtle.  A drug with more targets has more side effects
for arithmetic reasons: more proteins hit, more things broken. Any predictor built by summing over
targets inherits that, and would "work" on a completely uninformative gene score. So the real gate is
the PARTIAL correlation holding target count fixed, and the per-target MEAN is used rather than the sum
wherever a sum would smuggle the count back in.

PREDECLARED, before any number:

    S1 JOIN         at least 200 drugs carry both a SIDER side-effect list and targets in this model.
    S2 SIGNAL       the model's cost estimate correlates with side-effect count above the 95th
                    percentile of 200 drug-label shuffles.
    S3 BEATS TARGET COUNT   the correlation survives partialling out the number of targets.
                    THIS IS THE REAL GATE. fails -> the predictor is counting targets, and the cell
                    model contributed nothing.
    S4 BEATS LOOKUPS  it beats the mean LOEUF and the mean expression of the same targets -- two columns
                    of cell_complete that need no model at all.

A LIMIT THAT CANNOT BE CONTROLLED HERE, stated before the result.  SIDER counts side effects reported on
package inserts, so a widely prescribed, long-marketed drug accumulates more of them than an equally
disruptive new one. That is a study-and-exposure bias on the TARGET variable and nothing in this
container measures prescription volume. It inflates every correlation reported below, including the
controls, and it is a reason to treat a positive here as weaker than its coefficient suggests.

CONTROLS: 200 label shuffles; target count partialled out; mean LOEUF and mean expression of the same
targets; and the raw target count reported as a predictor in its own right so its strength is visible.

-> outputs/loop_sideeffect.json
"""
import collections
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path(os.environ.get("CELL_SCRATCH",
          "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"
SIDER_SE = SC / "sider_se.tsv.gz"
SIDER_NAMES = SC / "sider_names.tsv"
SEED = 1904
N_SHUFFLE = 200
GATE_JOIN = 200


def spearman(a, b):
    return float(pd.Series(np.asarray(a, float)).corr(pd.Series(np.asarray(b, float)),
                                                      method="spearman"))


def partial(a, b, c):
    """Spearman of a and b with c partialled out, on ranks -- the standard partial correlation."""
    ra, rb, rc = (pd.Series(x).rank().to_numpy() for x in (a, b, c))
    rab, rac, rbc = (np.corrcoef(ra, rb)[0, 1], np.corrcoef(ra, rc)[0, 1], np.corrcoef(rb, rc)[0, 1])
    d = np.sqrt(max((1 - rac ** 2) * (1 - rbc ** 2), 1e-12))
    return float((rab - rac * rbc) / d)


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("SIDE EFFECTS -- the row that had no data at all")
    say("=" * 100)
    say("  capability_audit used `side effect` as its MUST-FAIL calibration because nothing here could")
    say("  answer it. This adds SIDER, so that calibration will have to move again -- which is the point.")

    names = {}
    for line in open(SIDER_NAMES):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            names[p[0]] = p[1].strip().lower()
    se = collections.defaultdict(set)
    with gzip.open(SIDER_SE, "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 6 and p[3] == "PT":          # preferred terms only, not lower-level synonyms
                se[p[0]].add(p[5])
    by_name = {names[c]: v for c, v in se.items() if c in names}
    say(f"\n  SIDER 4.1: {len(by_name):,} named compounds with MedDRA preferred-term side effects")

    D = json.load(open(CELL))
    genes = D["genes"]
    tgt = collections.defaultdict(set)
    for gi, lst in D["drugs"].items():
        for e in lst:
            d = (e.get("d") or "").strip().lower()
            if d:
                tgt[d].add(int(gi))
    say(f"  cell_complete: {len(tgt):,} distinct drug names with protein targets")

    cost = {}
    lm = OUT / "loop_medium.json"
    if lm.exists():
        for r in json.load(open(lm)).get("scores", []):
            if r.get("medium") is not None and np.isfinite(r["medium"]):
                cost[r["sym"]] = float(r["medium"])
    say(f"  loop 4's defined-medium growth cost available for {len(cost):,} genes")

    rows = []
    for d, ts in tgt.items():
        if d not in by_name:
            continue
        syms = [genes[i]["name"] for i in ts if i < len(genes)]
        cs = [cost[s] for s in syms if s in cost]
        lo = [genes[i]["loeuf"] for i in ts if i < len(genes)
              and isinstance(genes[i].get("loeuf"), (int, float))]
        pp = [float(D["ppm"].get(str(i), 0.0)) for i in ts]
        es = [genes[i].get("ess") for i in ts if i < len(genes)
              and isinstance(genes[i].get("ess"), (int, float))]
        rows.append({"drug": d, "n_se": len(by_name[d]), "n_targets": len(ts),
                     "mean_cost": float(np.mean(cs)) if cs else np.nan,
                     "n_metabolic": len(cs),
                     "mean_ess": float(np.mean(es)) if es else np.nan,
                     "mean_loeuf": float(np.mean(lo)) if lo else np.nan,
                     "mean_ppm": float(np.log10(np.mean(pp) + 1e-3)) if pp else np.nan})
    T = pd.DataFrame(rows)
    say(f"\n  JOINED: {len(T):,} drugs with both a side-effect list and targets in this model")
    s1 = bool(len(T) >= GATE_JOIN)
    say(f"  S1 JOIN (>= {GATE_JOIN})   {'PASS' if s1 else 'FAIL'}")
    if not s1:
        json.dump({"test": "loop_sideeffect", "S1": False, "n": len(T), "log": log},
                  open(OUT / "loop_sideeffect.json", "w"), indent=2)
        return 1

    y = np.log10(T.n_se.to_numpy() + 1)
    nt = T.n_targets.to_numpy()
    say(f"    side effects per drug: median {int(T.n_se.median())}, range {T.n_se.min()}-{T.n_se.max()}")
    say(f"    targets per drug     : median {int(T.n_targets.median())}, "
        f"range {T.n_targets.min()}-{T.n_targets.max()}")

    # the confound, reported first and as a predictor in its own right
    r_nt = spearman(nt, y)
    say(f"\n  THE CONFOUND, measured before anything is claimed:")
    say(f"    number of targets alone vs side-effect count : {r_nt:+.4f}")

    preds = {}
    for col, label in (("mean_ess", "mean essentiality of targets"),
                       ("mean_cost", "mean growth cost of targets (loop 4 model)"),
                       ("mean_loeuf", "mean LOEUF of targets (lookup)"),
                       ("mean_ppm", "mean abundance of targets (lookup)")):
        m = T[col].notna().to_numpy()
        if m.sum() < 50:
            say(f"    {label:<44} only {int(m.sum())} drugs -- skipped")
            continue
        raw = spearman(T[col][m], y[m])
        par = partial(T[col][m].to_numpy(), y[m], nt[m])
        preds[label] = {"raw": raw, "partial": par, "n": int(m.sum())}

    say(f"\n  {'predictor':<46}{'raw':>9}{'partial':>10}{'n':>7}")
    for label, v in preds.items():
        say(f"  {label:<46}{v['raw']:>+9.4f}{v['partial']:>+10.4f}{v['n']:>7,}")
    say("    'partial' holds the number of targets fixed. That column is the one that matters.")

    # S2 signal, on the model's own score
    best_model = max((k for k in preds if "loop 4" in k or "essentiality" in k),
                     key=lambda k: abs(preds[k]["partial"]), default=None)
    if best_model is None:
        say("\n  no model-derived predictor had enough drugs; nothing to gate.")
        return 1
    col = {"mean essentiality of targets": "mean_ess",
           "mean growth cost of targets (loop 4 model)": "mean_cost"}[best_model]
    m = T[col].notna().to_numpy()
    null = np.array([spearman(T[col][m], rng.permutation(y[m])) for _ in range(N_SHUFFLE)])
    p95 = float(np.percentile(np.abs(null), 95))
    s2 = bool(abs(preds[best_model]["raw"]) > p95)
    say(f"\n  S2 SIGNAL -- best model predictor is '{best_model}'")
    say(f"    raw {preds[best_model]['raw']:+.4f} vs |null| 95th pct {p95:.4f}   "
        f"S2 {'PASS' if s2 else 'FAIL'}")

    nullp = np.array([partial(T[col][m].to_numpy(), rng.permutation(y[m]), nt[m])
                      for _ in range(N_SHUFFLE)])
    p95p = float(np.percentile(np.abs(nullp), 95))
    s3 = bool(abs(preds[best_model]["partial"]) > p95p)
    say(f"\n  S3 BEATS TARGET COUNT -- partial {preds[best_model]['partial']:+.4f} vs |null| 95th pct "
        f"{p95p:.4f}   S3 {'PASS' if s3 else 'FAIL'}")
    if not s3:
        say("    Holding the number of targets fixed, the cell model's estimate of what a gene costs")
        say("    says nothing about how many side effects a drug hitting it has. The apparent signal")
        say("    was target counting, which needs no model at all.")

    lookups = {k: v for k, v in preds.items() if "lookup" in k}
    best_lookup = max(lookups, key=lambda k: abs(lookups[k]["partial"])) if lookups else None
    s4 = bool(best_lookup is None or
              abs(preds[best_model]["partial"]) > abs(lookups[best_lookup]["partial"]))
    say(f"\n  S4 BEATS LOOKUPS -- vs {best_lookup}: "
        f"{abs(preds[best_model]['partial']):.4f} against "
        f"{abs(lookups[best_lookup]['partial']):.4f}   {'PASS' if s4 else 'FAIL'}"
        if best_lookup else "\n  S4 no lookup available")

    say("\n" + "=" * 100)
    gates = {"S1 join": s1, "S2 signal": s2, "S3 beats target count": s3, "S4 beats lookups": s4}
    for k, v in gates.items():
        say(f"  {k:<32}{'PASS' if v else 'FAIL'}")
    if s2 and s3:
        say("  SIDE EFFECT NOW HAS A SLOT, A PIPELINE AND A CONTROLLED SCORE, and the score survives the")
        say("  one confound that could have produced it for free. Remember the exposure bias: SIDER")
        say("  counts what package inserts report, so widely prescribed drugs accumulate more.")
    elif s2:
        say("  The correlation exists and is explained by how many proteins a drug hits. The cell model")
        say("  contributed nothing beyond target counting, and that is the honest reading.")
    else:
        say("  No signal. `side effect` now has data and a pipeline, and the pipeline says no.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(SIDER_SE), str(SIDER_NAMES), str(CELL), str(OUT / "loop_medium.json")],
                      available=len(tgt), used=len(T), selection="filtered", seed=SEED,
                      controls=[f"{N_SHUFFLE} label shuffles", "target count partialled out",
                                "target count as a predictor in its own right",
                                "mean LOEUF lookup", "mean abundance lookup"],
                      note="per-target MEANS used rather than sums, because a sum smuggles the target "
                           "count back into the predictor")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_sideeffect", "manifest": man, "gates": gates,
               "n_drugs": len(T), "r_target_count": r_nt, "predictors": preds,
               "best_model": best_model, "null_p95": p95, "null_partial_p95": p95p,
               "rows": T.to_dict("records"), "log": log},
              open(OUT / "loop_sideeffect.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_sideeffect.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
