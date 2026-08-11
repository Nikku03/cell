"""DRUG EFFECT, ASKED PROPERLY -- because the row currently passes for the wrong reason.

WHERE THIS COMES FROM.  capability_audit flipped `drug effect` to answerable at loop 13, and loop 13
said plainly that it should be read sceptically: no drug pipeline was ever built. The row cleared the
axes only because loop_sideeffect happens to name the `drugs` block while producing a controlled result
about something adjacent. A row that passes for the wrong reason is worse than one that fails, because
it stops anyone looking. This builds the pipeline the row is supposed to have.

THE QUESTION, made specific enough to be wrong.  "What does a compound do to the cell" is not testable
as written. The sharpest version this container can support: WHICH DRUGS ARE CYTOTOXIC? Antineoplastic
chemotherapeutics work by attacking processes every dividing cell needs -- DNA replication, tubulin,
nucleotide synthesis -- which is exactly what "essential" means. A cell model that knows what a gene
costs should therefore separate the cytotoxics from everything else in the pharmacopoeia.

THE LABEL.  ATC classification from SIDER: 1,560 compound-code pairs, of which 84 carry an L01 code
(antineoplastic and immunomodulating agents, cytotoxic branch). That is an external, clinical
classification, assigned for reasons that have nothing to do with this model.

THE CONFOUND, same as last time and it killed the last one.  A drug with more targets looks more
disruptive by arithmetic alone. Target count is partialled out and reported as a standalone predictor.

PREDECLARED, before any number:

    D1 JOIN      at least 30 L01 compounds carry targets in this model.
                 fails -> too few positives to test, and it is reported as untestable rather than run.
    D2 SIGNAL    mean essentiality of a drug's targets separates L01 from the rest above the 95th
                 percentile of 200 label shuffles.
    D3 BEATS TARGET COUNT   it survives partialling out the number of targets.
                 THIS IS THE REAL GATE, and it is the one that failed for side effects.
    D4 BEATS LOOKUPS   it beats mean LOEUF and mean abundance of the same targets.

WHY THIS ONE MIGHT WORK WHERE SIDE EFFECTS DID NOT, stated as a prediction and not as a hedge.  Side
effects happen in a patient, in tissues, after distribution and metabolism -- none of which this model
represents. Cytotoxicity happens in a dividing cell, which is the ONE thing this model does represent.
If the distinction is real, D3 passes here and failed there for a reason rather than by luck.

CONTROLS: 200 label shuffles; target count partialled out and reported alone; mean LOEUF and mean
abundance lookups; and the side-effect result as the direct comparator, since the two differ only in the
label.

-> outputs/loop_drug.json
"""
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from cell_loop import auc  # noqa: E402
from loop_sideeffect import partial as partial_sp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path(os.environ.get("CELL_SCRATCH",
          "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"
ATC = SC / "sider_atc.tsv"
NAMES = SC / "sider_names.tsv"
SEED = 1904
N_SHUFFLE = 200
GATE_JOIN = 30


def partial_auc(score, label, covar):
    """AUC of the residual after regressing the score on the confound."""
    from sklearn.linear_model import LinearRegression
    v, c = np.asarray(score, float), np.asarray(covar, float).reshape(-1, 1)
    ok = np.isfinite(v) & np.isfinite(c[:, 0])
    if ok.sum() < 30:
        return float("nan")
    r = v[ok] - LinearRegression().fit(c[ok], v[ok]).predict(c[ok])
    return auc(r, np.asarray(label)[ok])


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("DRUG EFFECT, ASKED PROPERLY -- which drugs are cytotoxic?")
    say("=" * 100)
    say("  The `drug effect` row passes today only because loop_sideeffect names the drugs block while")
    say("  answering something adjacent. A row that passes for the wrong reason stops anyone looking.")

    names = {}
    for line in open(NAMES):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            names[p[0]] = p[1].strip().lower()
    atc = collections.defaultdict(set)
    for line in open(ATC):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[0] in names:
            atc[names[p[0]]].add(p[1])
    cyto = {d for d, cs in atc.items() if any(c.startswith("L01") for c in cs)}
    say(f"\n  SIDER ATC: {len(atc):,} named compounds classified; {len(cyto):,} carry an L01 "
        f"(antineoplastic) code")

    D = json.load(open(CELL))
    genes = D["genes"]
    ppm = {int(k): float(v) for k, v in D["ppm"].items()}
    tgt = collections.defaultdict(set)
    for gi, lst in D["drugs"].items():
        for e in lst:
            d = (e.get("d") or "").strip().lower()
            if d:
                tgt[d].add(int(gi))

    rows = []
    for d, ts in tgt.items():
        if d not in atc:
            continue
        es = [genes[i].get("ess") for i in ts if i < len(genes)
              and isinstance(genes[i].get("ess"), (int, float))]
        lo = [genes[i].get("loeuf") for i in ts if i < len(genes)
              and isinstance(genes[i].get("loeuf"), (int, float))]
        pp = [ppm.get(i, 0.0) for i in ts]
        rows.append({"drug": d, "y": 1 if d in cyto else 0, "n_targets": len(ts),
                     "mean_ess": float(np.mean(es)) if es else np.nan,
                     "mean_loeuf": float(np.mean(lo)) if lo else np.nan,
                     "mean_ppm": float(np.log10(np.mean(pp) + 1e-3)) if pp else np.nan})
    T = pd.DataFrame(rows)
    y = T.y.to_numpy()
    npos = int(y.sum())
    say(f"  joined to this model's drug-target block: {len(T):,} drugs, {npos:,} of them cytotoxic")
    d1 = bool(npos >= GATE_JOIN)
    say(f"  D1 JOIN (>= {GATE_JOIN} cytotoxics)   {'PASS' if d1 else 'FAIL'}")
    if not d1:
        say("    Too few positives to test. Reported as untestable rather than run on a handful.")
        json.dump({"test": "loop_drug", "D1": False, "n_pos": npos, "log": log},
                  open(OUT / "loop_drug.json", "w"), indent=2)
        return 1

    nt = T.n_targets.to_numpy()
    a_nt = auc(nt, y)
    say(f"\n  THE CONFOUND, measured first: number of targets alone   AUC {a_nt:.4f}")

    preds = {}
    for col, label in (("mean_ess", "mean essentiality of targets (model)"),
                       ("mean_loeuf", "mean LOEUF of targets (lookup)"),
                       ("mean_ppm", "mean abundance of targets (lookup)")):
        v = T[col].astype(float)
        m = v.notna().to_numpy()
        if m.sum() < 30:
            continue
        sgn = -1.0 if col == "mean_loeuf" else 1.0
        preds[label] = {"raw": auc(sgn * v[m], y[m]),
                        "partial": partial_auc(sgn * v, y, nt), "n": int(m.sum())}
    say(f"\n  {'predictor':<40}{'raw':>9}{'partial':>10}{'n':>7}")
    for k, v in preds.items():
        say(f"  {k:<40}{v['raw']:>9.4f}{v['partial']:>10.4f}{v['n']:>7,}")

    best = "mean essentiality of targets (model)"
    v = T["mean_ess"].astype(float)
    m = v.notna().to_numpy()
    null = np.array([auc(v[m], rng.permutation(y[m])) for _ in range(N_SHUFFLE)])
    lo_, hi_ = np.percentile(null, [2.5, 97.5])
    d2 = bool(preds[best]["raw"] < lo_ or preds[best]["raw"] > hi_)
    say(f"\n  D2 SIGNAL -- raw {preds[best]['raw']:.4f} against a shuffled null of "
        f"[{lo_:.4f}, {hi_:.4f}]   D2 {'PASS' if d2 else 'FAIL'}")

    nullp = np.array([partial_auc(v, rng.permutation(y), nt) for _ in range(N_SHUFFLE)])
    nullp = nullp[np.isfinite(nullp)]
    plo, phi = np.percentile(nullp, [2.5, 97.5])
    d3 = bool(preds[best]["partial"] < plo or preds[best]["partial"] > phi)
    say(f"\n  D3 BEATS TARGET COUNT -- partial {preds[best]['partial']:.4f} against [{plo:.4f}, {phi:.4f}]"
        f"   D3 {'PASS' if d3 else 'FAIL'}")

    lookups = {k: v for k, v in preds.items() if "lookup" in k}
    bl = max(lookups, key=lambda k: abs(lookups[k]["partial"] - 0.5)) if lookups else None
    d4 = bool(bl is None or abs(preds[best]["partial"] - 0.5) > abs(lookups[bl]["partial"] - 0.5))
    if bl:
        say(f"\n  D4 BEATS LOOKUPS -- vs {bl}: |{preds[best]['partial']:.4f}-0.5| against "
            f"|{lookups[bl]['partial']:.4f}-0.5|   {'PASS' if d4 else 'FAIL'}")

    # the direct comparator: the same predictor, the same confound, a different label
    se = OUT / "loop_sideeffect.json"
    if se.exists():
        S = json.load(open(se))
        pe = S["predictors"].get("mean essentiality of targets", {})
        say(f"\n  THE DIRECT COMPARATOR -- same predictor, same confound, different label:")
        say(f"    side-effect COUNT  partial {pe.get('partial', float('nan')):+.4f}  (loop 12: FAILED)")
        say(f"    cytotoxic CLASS    partial {preds[best]['partial']:.4f}  "
            f"({'PASSES' if d3 else 'FAILS'})")

    say("\n" + "=" * 100)
    gates = {"D1 join": d1, "D2 signal": d2, "D3 beats target count": d3, "D4 beats lookups": d4}
    for k, vv in gates.items():
        say(f"  {k:<32}{'PASS' if vv else 'FAIL'}")
    if d2 and d3:
        say("  THE MODEL SEPARATES CYTOTOXICS FROM THE REST, and target count does not explain it.")
        say("  `drug effect` now has the pipeline it was passing without. And the contrast with side")
        say("  effects is the interesting part: same predictor, same confound, opposite outcome --")
        say("  because cytotoxicity happens in a dividing cell, which this model represents, and an")
        say("  adverse event happens in a tissue, which it does not.")
    elif d2:
        say("  The signal is target counting again. `drug effect` still has no pipeline that works, and")
        say("  the row should be read as passing on falsifiability alone.")
    else:
        say("  No signal. Recorded.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(ATC), str(NAMES), str(CELL)], available=len(tgt), used=len(T),
                      selection="filtered", seed=SEED,
                      controls=[f"{N_SHUFFLE} label shuffles", "target count partialled out",
                                "target count as a standalone predictor",
                                "LOEUF and abundance lookups",
                                "the side-effect result as a same-predictor comparator"],
                      note="ATC L01 as an external clinical label assigned for reasons unrelated to "
                           "this model")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_drug", "manifest": man, "gates": gates, "n_drugs": len(T),
               "n_cytotoxic": npos, "auc_target_count": a_nt, "predictors": preds,
               "null": {"lo": float(lo_), "hi": float(hi_)},
               "null_partial": {"lo": float(plo), "hi": float(phi)},
               "rows": T.to_dict("records"), "log": log},
              open(OUT / "loop_drug.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_drug.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
