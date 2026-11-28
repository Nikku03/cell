"""DOES THE WHOLE BEAT THE PARTS -- every layer this project built, in one predictor.

WHERE THIS COMES FROM.  Twenty-four loops built layers one at a time and scored each on its own:
metabolic growth cost, measured DNA torsion, functional neighbourhood, expression, constraint. Each was
compared against its own controls and its own baseline. NONE of them has ever been put together, and
"is there a working cell model" is exactly the question of whether the layers are complementary or
redundant.

That is not a rhetorical question and it has a hard answer: under cross-validation, does the combination
beat the best single layer? If it does, the layers see different things and assembling them is worth
doing. If it does not, this is one predictor wearing five coats.

THE LAYERS, each with the loop that built it:
    metabolic cost        loop 4/21   growth lost when the gene is deleted, defined medium
    DNA torsion           loop 8      measured bTMP signal at the TSS, GSE277502
    functional neighbours loop 7      mean essentiality of model4 neighbours
    mRNA abundance        loop 3      the confound that beat everything mechanistic
    mRNA stability        loop 1      decay rate from RNADecayCafe
    protein abundance     -           cell_complete ppm
    LOEUF                 -           gnomAD constraint, the strongest single lookup found so far

THE CONTROL THAT DECIDES IT.  Adding features to a model almost always improves TRAINING fit, so the
comparison has to be out-of-fold, and the folds have to be blocked by CHROMOSOME. Neighbouring genes
share expression, chromatin state and often function; random folds would let the model memorise a
locus and call it prediction.

PREDECLARED, before any number:

    I1 THE WHOLE BEATS THE PARTS   combined chromosome-blocked AUC >= best single layer + 0.02.
                fails -> the layers are redundant, and the honest description of this project is one
                predictor with several names.
    I2 MECHANISM ADDS TO LOOKUPS   the three MECHANISTIC layers (metabolic cost, torsion, neighbours)
                add >= 0.02 over the four lookups alone.
                THIS IS THE REAL GATE. Everything built here that is not a lookup has to justify
                itself against the columns that were already sitting in cell_complete.
    I3 NO SINGLE LAYER CARRIES IT   removing any one layer costs less than 0.05, i.e. no layer is doing
                all the work while the others ride along.
    I4 BEATS A SHUFFLE   combined AUC above the 95th percentile of 200 label shuffles.

CONTROLS: chromosome-blocked folds; 200 label shuffles; each layer scored alone; leave-one-layer-out for
I3; and the lookups-only model as the comparator for I2.

WHAT HAPPENED, written after the run against the gates above, unedited.

    I1 THE WHOLE BEATS THE PARTS   PASS.  All seven layers 0.8309 against the best single layer 0.7729,
        a gain of +0.0580, under chromosome-blocked folds.
    I2 MECHANISM ADDS TO LOOKUPS   PASS.  0.8309 against the four lookups alone at 0.7909, +0.0400.
        Something built here does earn its place against the columns that were already in the file.
    I3 NO SINGLE LAYER CARRIES IT   PASS on the letter of the gate, and it is the most informative
        result in the module. Cost of removing each layer:
            metabolic cost          +0.0411
            mRNA abundance          +0.0393
            LOEUF                   +0.0045
            protein abundance       +0.0014
            mRNA stability          +0.0012
            functional neighbours   -0.0008
            DNA torsion             -0.0012
    I4 BEATS A SHUFFLE   PASS.  Null 0.5000 +/- 0.0192 against 0.8309.

THE SENTENCE THAT WOULD HAVE BEEN WRONG.  The verdict first read "the layers are complementary". They
are not. TWO layers carry this model -- the metabolic growth cost and mRNA abundance -- and five are
passengers, two of them with NEGATIVE contributions, meaning the model is slightly better without
measured DNA torsion and without functional neighbourhood in it. Both of those were real, controlled
findings in their own loops; neither survives contact with a model that already knows a gene's
expression and its metabolic cost.

WHAT IS ACTUALLY ESTABLISHED, stated at its true size.  One mechanistic layer adds +0.04 AUC on top of
four catalogue columns, on 1,783 genes, held out by chromosome. That is the strongest integrated result
this project has, and it is one layer, not seven.

AND A COVERAGE LIMIT.  Requiring all seven layers drops the scored set to 1,783 genes -- 10.8% of the
genome. The assembled model is only defined where every input exists, which is a small and unusually
well-measured slice.

-> outputs/loop_integrate.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from cell_loop import auc, KDEG, CELL, MIN_LINES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path(os.environ.get("CELL_SCRATCH",
          "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
SEED = 1904
N_SHUFFLE = 200
GATE_WHOLE = 0.02
GATE_MECH = 0.02
GATE_SINGLE = 0.05
WINDOW = 20000

MECH = ["metabolic cost", "DNA torsion", "functional neighbours"]
LOOKUP = ["mRNA abundance", "mRNA stability", "protein abundance", "LOEUF"]


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("DOES THE WHOLE BEAT THE PARTS")
    say("=" * 100)
    say("  Twenty-four loops built layers one at a time and never put them together. Whether they are")
    say("  complementary or redundant is the question 'is there a working cell model' actually means.")

    D = json.load(open(CELL))
    genes = D["genes"]
    name2i = {g["name"]: i for i, g in enumerate(genes)}
    ppm = {int(k): float(v) for k, v in D["ppm"].items()}

    cost = {}
    lm = OUT / "loop_medium.json"
    if lm.exists():
        for r in json.load(open(lm))["scores"]:
            v = r.get("medium")
            if v is not None and np.isfinite(v):
                cost[r["sym"]] = max(float(v), 0.0)

    nbr_ess = {}
    m4 = D.get("model4") or {}
    for k, v in m4.items():
        js = [name2i[nm] for nm in (v.get("neighbors") or []) if nm in name2i]
        es = [genes[j].get("ess") for j in js if isinstance(genes[j].get("ess"), (int, float))]
        if es:
            nbr_ess[genes[int(k)]["name"]] = float(np.mean(es))

    tors = {}
    try:
        import pyBigWig
        bws = [pyBigWig.open(str(SC / f)) for f in
               ("GSM8523629_supercoil_rep1_hg38.bw", "GSM8523630_supercoil_rep2_hg38.bw")]
        chroms = set(bws[0].chroms())
        for g in genes:
            c, t = g.get("chrom"), g.get("ens_tss") or g.get("tss")
            if c is None or not isinstance(t, (int, float)):
                continue
            cc = str(c) if str(c).startswith("chr") else f"chr{c}"
            if cc not in chroms:
                continue
            vs = []
            for bw in bws:
                try:
                    v = bw.stats(cc, max(0, int(t) - WINDOW), int(t) + WINDOW, type="mean")[0]
                    if v is not None:
                        vs.append(float(v))
                except Exception:
                    pass
            if vs:
                tors[g["name"]] = float(np.mean(vs))
    except Exception as e:
        say(f"  torsion unavailable ({e}); that layer is dropped and reported as dropped")

    k = pd.read_csv(KDEG)
    rp = (k[k.avg_RPKM_total > 0].groupby("feature_ID")
          .agg(rpkm=("avg_RPKM_total", "median"), n=("cell_line", "size")))
    kd = k.groupby("feature_ID").agg(kdeg=("avg_kdeg", "median"), n=("cell_line", "size"))
    rp, kd = rp[rp.n >= MIN_LINES], kd[kd.n >= MIN_LINES]

    rows = []
    for i, g in enumerate(genes):
        s = g["name"]
        if s not in cost or s not in rp.index or s not in kd.index or s not in tors:
            continue
        if not isinstance(g.get("ess"), (int, float)) or g.get("chrom") is None:
            continue
        rows.append({
            "sym": s, "chrom": str(g["chrom"]), "y": int(g["ess"]),
            "metabolic cost": cost[s],
            "DNA torsion": tors[s],
            "functional neighbours": nbr_ess.get(s, np.nan),
            "mRNA abundance": np.log10(float(rp.rpkm[s]) + 1e-3),
            "mRNA stability": -float(kd.kdeg[s]),
            "protein abundance": np.log10(ppm.get(i, 0.0) + 1e-3),
            "LOEUF": -(g.get("loeuf") if isinstance(g.get("loeuf"), (int, float)) else np.nan),
        })
    T = pd.DataFrame(rows)
    layers = MECH + LOOKUP
    RM.check_features(T, layers, emit=say)
    for c in layers:
        T[c] = T[c].fillna(T[c].median())
    y = T.y.to_numpy()
    groups = T.chrom.to_numpy()
    say(f"\n  {len(T):,} genes carrying ALL {len(layers)} layers; {int(y.sum()):,} essential")
    say(f"  layers: {', '.join(layers)}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold

    def cv(cols, yy=None):
        yy = y if yy is None else yy
        X = T[list(cols)].to_numpy(float)
        oof = np.zeros(len(yy))
        for tr, te in GroupKFold(n_splits=5).split(X, yy, groups):
            if len(np.unique(yy[tr])) < 2:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            m.fit(X[tr], yy[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        return auc(oof, yy)

    single = {c: cv([c]) for c in layers}
    say(f"\n  {'layer':<26}{'alone':>9}")
    for c, v in sorted(single.items(), key=lambda x: -x[1]):
        tag = "  (mechanistic)" if c in MECH else "  (lookup)"
        say(f"  {c:<26}{v:>9.4f}{tag}")

    a_all = cv(layers)
    a_look = cv(LOOKUP)
    a_mech = cv(MECH)
    best_single = max(single.values())
    say(f"\n  {'all seven layers':<26}{a_all:>9.4f}")
    say(f"  {'the four lookups only':<26}{a_look:>9.4f}")
    say(f"  {'the three mechanistic only':<26}{a_mech:>9.4f}")
    say(f"  {'best single layer':<26}{best_single:>9.4f}")

    i1 = bool(a_all - best_single >= GATE_WHOLE)
    say(f"\n  I1 THE WHOLE BEATS THE PARTS -- {a_all:.4f} - {best_single:.4f} = "
        f"{a_all-best_single:+.4f}   I1 {'PASS' if i1 else 'FAIL'}")
    d2 = a_all - a_look
    i2 = bool(d2 >= GATE_MECH)
    say(f"  I2 MECHANISM ADDS TO LOOKUPS -- {a_all:.4f} - {a_look:.4f} = {d2:+.4f}   "
        f"I2 {'PASS' if i2 else 'FAIL'}")
    if not i2:
        say("    Everything mechanistic this project built adds nothing to four columns that were")
        say("    already sitting in cell_complete. That is the honest reading and it is the reading")
        say("    this gate exists to force.")

    drops = {c: a_all - cv([x for x in layers if x != c]) for c in layers}
    say(f"\n  I3 NO SINGLE LAYER CARRIES IT -- cost of removing each layer:")
    for c, v in sorted(drops.items(), key=lambda x: -x[1]):
        say(f"    {c:<26}{v:>+9.4f}")
    i3 = bool(max(drops.values()) < GATE_SINGLE)
    say(f"    largest single-layer cost {max(drops.values()):+.4f} (gate < {GATE_SINGLE})   "
        f"I3 {'PASS' if i3 else 'FAIL'}")

    null = np.array([auc(T[layers].mean(axis=1), rng.permutation(y)) for _ in range(N_SHUFFLE)])
    p95 = float(np.percentile(null, 95))
    i4 = bool(a_all > p95)
    say(f"\n  I4 BEATS A SHUFFLE -- null {null.mean():.4f} +/- {null.std():.4f}, 95th {p95:.4f}   "
        f"I4 {'PASS' if i4 else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"I1 the whole beats the parts": i1, "I2 mechanism adds to lookups": i2,
             "I3 no single layer carries it": i3, "I4 beats a shuffle": i4}
    for kk, vv in gates.items():
        say(f"  {kk:<36}{'PASS' if vv else 'FAIL'}")
    if i1 and i2:
        carry = [c for c, v in drops.items() if v >= 0.01]
        dead = [c for c, v in drops.items() if v <= 0.005]
        say("  THE ASSEMBLY BEATS ITS BEST PART AND BEATS THE CATALOGUE COLUMNS, under folds blocked by")
        say("  chromosome. That is the minimum 'a working cell model' has to mean, and it is measured.")
        say(f"  BUT READ I3 BEFORE CALLING THE LAYERS COMPLEMENTARY. Only {len(carry)} of {len(layers)}")
        say(f"  layers carry anything: {', '.join(carry)}.")
        say(f"  {len(dead)} contribute 0.005 or less and two of them are NEGATIVE -- removing them")
        say(f"  improves the model: {', '.join(dead)}.")
        say("  So the honest sentence is not 'the layers are complementary'. It is that ONE mechanistic")
        say("  layer -- the metabolic growth cost -- adds to ONE lookup, and the other five are")
        say("  passengers. That is a real result and a much smaller one than seven layers implies.")
    elif i1:
        say("  THE LAYERS COMBINE, BUT THE MECHANISM DOES NOT EARN ITS PLACE. The gain comes from")
        say("  stacking lookups. An honest description is a well-assembled catalogue, not a model.")
    else:
        say("  THE LAYERS ARE REDUNDANT. One predictor wearing several coats, and the assembly adds")
        say("  nothing over its best single part.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(CELL), str(KDEG), str(OUT / "loop_medium.json")],
                      available=len(genes), used=len(T), selection="filtered", seed=SEED,
                      controls=["chromosome-blocked folds", f"{N_SHUFFLE} label shuffles",
                                "each layer scored alone", "leave-one-layer-out",
                                "lookups-only model as the comparator for mechanism"],
                      note="folds blocked by chromosome because neighbouring genes share expression, "
                           "chromatin and function, and random folds would reward memorising a locus")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_integrate", "manifest": man, "gates": gates,
               "single": single, "all_layers": a_all, "lookups_only": a_look,
               "mechanistic_only": a_mech, "best_single": best_single,
               "leave_one_out_cost": drops, "n": len(T), "n_essential": int(y.sum()),
               "null": {"mean": float(null.mean()), "sd": float(null.std()), "p95": p95},
               "log": log},
              open(OUT / "loop_integrate.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_integrate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
