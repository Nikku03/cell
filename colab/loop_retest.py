"""LOOP 1, RE-RUN ON A MODEL THAT CANNOT EAT ATP.

WHY THIS EXISTS.  cell_loop closed the growth feedback and then failed the two gates that mattered:
G3 (the feedback earns its place) at -0.0074, and G4 (beats the catalogue) against plain FBA and against
an expression lookup. Those numbers were computed on Human-GEM AS SHIPPED -- 1,660 exchanges open at
-1000, importable ATP, NAD+, LDL, plasminogen, and metabolites of up to 9,642,032 carbons.

loop_medium then showed that medium was holding flux balance back: closing it moved plain FBA from
0.6491 to 0.6850. So loop 1's failures were recorded on a model that has since been shown broken in a
way that directly suppresses the quantity being measured. A negative obtained on a broken instrument is
not a negative about the idea; it is a negative about the instrument, and leaving it standing would be
exactly the kind of banked result this project exists to avoid.

WHAT CHANGES, and nothing else does.  Same code path, same gates, same scored genes, same seed. Only the
model underneath:
    medium      loop 4's 57 uptakes, instead of everything open. The CAP is re-derived here rather
                than reused: loop 4 set it so nutrients ALONE give exactly MU_TARGET, and requiring the
                combined medium+enzyme model to also give MU_TARGET would then force the enzyme layer
                to be inert. The first run of this file did exactly that -- kappa ran to its ceiling at
                999,960, the capacity bounds went effectively infinite, and G3 would have read ~0 for a
                reason with no biology in it. So the constraint is SPLIT by a declared choice: nutrients
                alone support 2x MU_TARGET, and enzymes account for the other factor of two. Both bind
                by construction, and a run where kappa still hits its ceiling is recorded NOT TESTABLE.
    capacity    loop 1's enzyme layer on top, kappa bisected against that headroom
    mu_WT       RE-SOLVED at the final bounds rather than taken from the bisection's last iterate.
                That is the one-line fix loop_slack identified: using the iterate made 1,525 unaffected
                knockouts look like they IMPROVED growth by 7.2e-07, which is what produced loop 4's
                bogus 92% "no slack" figure.

PREDECLARED, and they are loop 1's gates unchanged so the two runs are comparable:

    G1 CLOSURE    mu = F(mu) converges to the same fixed point from mu0 = 0 and 2*mu_WT.
    G2 SIGNAL     loop AUC beats the 95th percentile of 200 label shuffles.
    G3 THE FEEDBACK EARNS ITS PLACE   loop AUC - frozen-mu AUC >= 0.02.
                  On the open medium this was -0.0074. If it is still negative on a model that cannot
                  import its own energy currency, the feedback is genuinely not carrying anything and
                  loop 1's verdict stands on better evidence than it originally had.
    G4 BEATS THE CATALOGUE   beats plain FBA deletion and the best single lookup.
    G5 ADDS TO THE LOOKUP    NEW, and it is the fair version of G4. loop_deficit established that the
                  expression lookup is a general confound that knows nothing about metabolism, so
                  "loses to it" and "carries nothing" are different claims. Under 5-fold CV, does the
                  loop add >= 0.02 AUC on top of the lookup?

CONTROLS: 200 label shuffles; frozen-mu on identical code with only the feedback switched off; plain FBA
with no protein layer; the expression lookup; and loop 1's own recorded numbers as the direct
open-medium comparator, since the two runs differ only in the model.

-> outputs/loop_retest.json
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from cell_loop import auc, gpr_capacity, GEM, GEMGENES, KDEG, CELL, K_DP, MIN_LINES  # noqa: E402
from loop_deficit import cv_auc  # noqa: E402
from loop_medium import PHYS_SET  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 1904
N_SHUFFLE = 200
MU_TARGET = 0.030
MEDIUM_HEADROOM = 2.0        # nutrients alone support this multiple of MU_TARGET; enzymes do the rest
EXTRA = {"gamma-tocopherol", "lipoic acid"}
DAMP = 0.5
TOL = 1e-6
MAXIT = 200
GATE_FEEDBACK = 0.02
GATE_ADDS = 0.02
OPEN = {"loop": 0.6210, "frozen": 0.6284, "plainFBA": 0.6491, "lookup": 0.7766, "delta_g3": -0.0074}


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("LOOP 1, RE-RUN ON A MODEL THAT CANNOT EAT ATP")
    say("=" * 100)
    say("  cell_loop's G3 and G4 failed on Human-GEM as shipped, which loop 4 later showed was")
    say("  suppressing exactly the quantity being measured. Same code, same gates, same genes, same")
    say("  seed -- only the model underneath is fixed.")

    import cobra  # noqa: F401
    from cobra.io import read_sbml_model
    t0 = time.time()
    M = read_sbml_model(str(GEM))
    M.solver = "glpk"
    say(f"\n  Human-GEM loaded ({time.time()-t0:.0f}s)")

    keep = set()
    for r in M.exchanges:
        nm = (list(r.metabolites)[0].name or "").strip().lower()
        if nm in PHYS_SET or nm in EXTRA:
            keep.add(r.id)
    for r in M.exchanges:
        r.lower_bound = 0.0
    # BOTH CONSTRAINTS MUST BIND, and the first version of this file got that wrong in a way that
    # would have produced a meaningless G3. loop 4 calibrated its uptake cap so that NUTRIENTS ALONE
    # give exactly MU_TARGET. Requiring the combined medium+enzyme model to also give MU_TARGET then
    # forces the enzyme layer to be non-binding: the bisection ran kappa to its ceiling (999,960),
    # the capacity bounds became effectively infinite, and the feedback would have had nothing to act
    # on. G3 would have read ~0 for a reason with no biology in it.
    # The fix is a declared split of the constraint between the two layers: open the medium enough that
    # NUTRIENTS ALONE would support MEDIUM_HEADROOM x MU_TARGET, then bisect kappa to bring the combined
    # model down to MU_TARGET. Nutrients allow twice the observed growth; enzymes account for the other
    # factor of two. Both bind by construction, and the split is a stated choice rather than an accident.
    lo_u, hi_u = 1e-6, 1e4
    for _ in range(60):
        mid = np.sqrt(lo_u * hi_u)
        for rid in keep:
            M.reactions.get_by_id(rid).lower_bound = -mid
        v = M.slim_optimize()
        if v is None or not np.isfinite(v) or v < MEDIUM_HEADROOM * MU_TARGET:
            lo_u = mid
        else:
            hi_u = mid
        if hi_u / lo_u < 1.0001:
            break
    CAP = float(np.sqrt(lo_u * hi_u))
    for rid in keep:
        M.reactions.get_by_id(rid).lower_bound = -CAP
    mu_medium_only = float(M.slim_optimize())
    say(f"  medium: {len(keep)} uptakes capped at {CAP:.4g} mmol/gDW/h -> nutrients alone support "
        f"{mu_medium_only:.5f}/h = {MEDIUM_HEADROOM}x target, leaving room for enzymes to bind too")

    gm = pd.read_csv(GEMGENES, sep="\t")
    ens2sym = {a: b for a, b in zip(gm["genes"], gm["geneSymbols"]) if isinstance(b, str)}
    k = pd.read_csv(KDEG)
    rp = (k[k.avg_RPKM_total > 0].groupby("feature_ID")
          .agg(rpkm=("avg_RPKM_total", "median"), n=("cell_line", "size")))
    kd = k.groupby("feature_ID").agg(kdeg=("avg_kdeg", "median"), n=("cell_line", "size"))
    rp, kd = rp[rp.n >= MIN_LINES], kd[kd.n >= MIN_LINES]
    med_r, med_d = float(rp.rpkm.median()), float(kd.kdeg.median())
    KSYN, KDEGG, measured = {}, {}, set()
    for g in M.genes:
        s = ens2sym.get(g.id)
        r = float(rp.rpkm[s]) if s in rp.index else None
        d = float(kd.kdeg[s]) if s in kd.index else None
        if r is not None and d is not None:
            measured.add(g.id)
        r, d = (med_r if r is None else r), (med_d if d is None else d)
        KSYN[g.id], KDEGG[g.id] = r * (d + MU_TARGET), d

    gpr = {r.id: r.gpr for r in M.reactions if r.gene_reaction_rule.strip()}
    RX = [M.reactions.get_by_id(rid) for rid in gpr]
    ORIG = {r.id: r.bounds for r in M.reactions}

    def protein(mu, ko=None):
        return {g: (0.0 if g == ko else KSYN[g] / ((KDEGG[g] + mu) * (K_DP + mu))) for g in KSYN}

    def apply_cap(kappa, E):
        default = float(np.median(list(E.values())))
        for r in RX:
            c = gpr_capacity(gpr[r.id], E, default)
            if c is None:
                continue
            b = kappa * c
            lo, hi = ORIG[r.id]
            r.bounds = (max(lo, -b), min(hi, b))

    def F(mu, kappa, ko=None):
        apply_cap(kappa, protein(mu, ko))
        v = M.slim_optimize()
        return 0.0 if (v is None or not np.isfinite(v)) else float(max(v, 0.0))

    def fixed_point(kappa, ko=None, mu0=MU_TARGET):
        mu = mu0
        for it in range(MAXIT):
            nxt = F(mu, kappa, ko)
            new = (1 - DAMP) * mu + DAMP * nxt
            if abs(new - mu) <= TOL * max(mu, 1e-9):
                return new, it + 1
            mu = new
        return mu, MAXIT

    lo_k, hi_k = 1e-12, 1e12
    for _ in range(80):
        mid = np.sqrt(lo_k * hi_k)
        if fixed_point(mid)[0] < MU_TARGET:
            lo_k = mid
        else:
            hi_k = mid
        if hi_k / lo_k < 1.0001:
            break
    KAPPA = float(np.sqrt(lo_k * hi_k))
    MU_WT, its = fixed_point(KAPPA)
    binding = bool(KAPPA < 1e12 * 0.99)
    say(f"  capacity: kappa {KAPPA:.6g}, wild-type mu {MU_WT:.8f}/h in {its} damped steps")
    say(f"    enzyme layer actually binds: {binding}  (nutrients alone {mu_medium_only:.5f}/h, "
        f"combined {MU_WT:.5f}/h)")
    if not binding:
        say("    kappa ran to its ceiling, so the capacity layer is inert and the feedback has nothing")
        say("    to act on. Recorded as NOT TESTABLE rather than run.")
        json.dump({"test": "loop_retest", "not_testable": True, "kappa": KAPPA, "log": log},
                  open(OUT / "loop_retest.json", "w"), indent=2)
        return 1

    # G1
    a, ia = fixed_point(KAPPA, mu0=0.0)
    b, ib = fixed_point(KAPPA, mu0=2 * MU_WT)
    spread = abs(a - b) / max(MU_WT, 1e-12)
    g1 = bool(spread < 1e-4)
    say(f"\n  G1 CLOSURE -- from 0: {a:.8f}/h ({ia} steps); from 2*mu: {b:.8f}/h ({ib} steps); "
        f"spread {spread:.2e}   G1 {'PASS' if g1 else 'FAIL'}")

    # frozen-mu: E fixed at MU_WT. mu_WT is RE-SOLVED at these bounds, which is loop_slack's fix.
    E0 = protein(MU_WT)
    apply_cap(KAPPA, E0)
    MU_REF = float(M.slim_optimize())
    CAPPED = {r.id: r.bounds for r in RX}
    default0 = float(np.median(list(E0.values())))
    say(f"  reference mu RE-SOLVED at the final bounds: {MU_REF:.8f}/h "
        f"(loop_slack's one-line fix; loop 4 used the bisection iterate and got a bogus 92%)")

    # CACHE EACH PASS. The loop pass runs a full fixed point per knockout and overran the 10-minute
    # ceiling on the first attempt, losing the frozen pass with it. Each pass is now keyed by kappa and
    # written as soon as it completes, so a rerun resumes instead of repeating.
    SC = Path(os.environ.get("CELL_SCRATCH",
              "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
    tag = f"{KAPPA:.6g}_{len(keep)}"

    def cached(name, fn):
        f = SC / f"_retest_{name}_{tag}.json"
        if f.exists():
            say(f"    reusing cached {name} pass from {f.name}")
            return json.load(open(f))
        out = fn()
        json.dump(out, open(f, "w"))
        return out

    gids = [g.id for g in M.genes]
    t1 = time.time()
    def _frozen():
      frozen = {}
      for n, gid in enumerate(gids):
        g = M.genes.get_by_id(gid)
        touched = [r for r in g.reactions if r.id in gpr]
        E2 = dict(E0)
        E2[gid] = 0.0
        for r in touched:
            c = gpr_capacity(gpr[r.id], E2, default0)
            if c is None:
                continue
            bnd = KAPPA * c
            lo0, hi0 = ORIG[r.id]
            r.bounds = (max(lo0, -bnd), min(hi0, bnd))
        v = M.slim_optimize()
        frozen[gid] = 0.0 if (v is None or not np.isfinite(v)) else float(max(v, 0.0))
        for r in touched:
            r.bounds = CAPPED[r.id]
        if n % 1000 == 0:
            say(f"      frozen {n}/{len(gids)} ({time.time()-t1:.0f}s)")
      return frozen
    frozen = cached("frozen", _frozen)
    say(f"    frozen pass {time.time()-t1:.0f}s")

    movers = [g for g in gids if abs(frozen[g] - MU_REF) > 1e-6 * MU_REF]
    say(f"    {len(movers)} of {len(gids)} knockouts move growth by more than 1e-6 relative "
        f"(open medium: 2,848 of 2,848 by a threshold that turned out to be numerical dust)")
    t2 = time.time()

    def _loop():
        out = {}
        for n, gid in enumerate(movers):
            out[gid] = fixed_point(KAPPA, ko=gid, mu0=MU_WT)[0]
            if n % 25 == 0:
                say(f"      loop {n}/{len(movers)} ({time.time()-t2:.0f}s)")
        return out
    loop = dict(frozen)
    loop.update(cached("loop", _loop))
    say(f"    loop pass {time.time()-t2:.0f}s")

    # plain FBA on the same medium, no protein layer
    for r in RX:
        r.bounds = ORIG[r.id]
    mu_plain_wt = float(M.slim_optimize())
    t3 = time.time()

    def _plain():
      plain = {}
      for n, gid in enumerate(gids):
        g = M.genes.get_by_id(gid)
        touched = []
        for r in g.reactions:
            if r.id not in gpr:
                continue
            Eb = {x.id: 1.0 for x in r.genes}
            Eb[gid] = 0.0
            if (gpr_capacity(gpr[r.id], Eb, 1.0) or 0.0) <= 0.0:
                touched.append(r)
                r.bounds = (0.0, 0.0)
        v = M.slim_optimize()
        plain[gid] = 0.0 if (v is None or not np.isfinite(v)) else float(max(v, 0.0))
        for r in touched:
            r.bounds = ORIG[r.id]
      return plain
    plain = cached("plain", _plain)
    say(f"    plainFBA pass {time.time()-t3:.0f}s (no-protein-layer mu {mu_plain_wt:.5f}/h)")

    # ---- score on loop 1's own gene set ---------------------------------------------------------------
    prev = json.load(open(OUT / "cell_loop.json"))
    T0 = pd.DataFrame(prev["scores"])
    sym2gid = {ens2sym[g]: g for g in gids if g in ens2sym}
    rows = []
    for _, r in T0.iterrows():
        gid = sym2gid.get(r["sym"])
        if gid is None:
            continue
        rows.append({"sym": r["sym"], "ess": int(r["ess"]), "rpkm": float(r["rpkm"]),
                     "loop": 1.0 - loop[gid] / MU_REF, "frozen": 1.0 - frozen[gid] / MU_REF,
                     "plain": 1.0 - plain[gid] / max(mu_plain_wt, 1e-12)})
    T = pd.DataFrame(rows)
    y = T.ess.to_numpy()
    lr = np.log10(T.rpkm.to_numpy() + 1e-3)
    say(f"\n  SCORING SET: {len(T)} genes, {int(y.sum())} essential -- loop 1's own set")

    A = {"loop": auc(T["loop"], y), "frozen": auc(T["frozen"], y), "plainFBA": auc(T["plain"], y),
         "mRNA abundance": auc(lr, y)}
    say(f"\n  {'predictor':<20}{'defined medium':>16}{'open medium':>14}{'change':>9}")
    for kk in ("loop", "frozen", "plainFBA"):
        say(f"  {kk:<20}{A[kk]:>16.4f}{OPEN[kk]:>14.4f}{A[kk]-OPEN[kk]:>+9.4f}")
    say(f"  {'mRNA abundance':<20}{A['mRNA abundance']:>16.4f}{OPEN['lookup']:>14.4f}"
        f"{A['mRNA abundance']-OPEN['lookup']:>+9.4f}")

    null = np.array([auc(T["loop"], rng.permutation(y)) for _ in range(N_SHUFFLE)])
    p95 = float(np.percentile(null, 95))
    g2 = bool(A["loop"] > p95)
    say(f"\n  G2 SIGNAL -- null {null.mean():.4f} +/- {null.std():.4f}, 95th {p95:.4f}   "
        f"G2 {'PASS' if g2 else 'FAIL'}")

    d3 = A["loop"] - A["frozen"]
    g3 = bool(d3 >= GATE_FEEDBACK)
    say(f"\n  G3 THE FEEDBACK EARNS ITS PLACE -- {A['loop']:.4f} - {A['frozen']:.4f} = {d3:+.4f}  "
        f"(open medium was {OPEN['delta_g3']:+.4f})   G3 {'PASS' if g3 else 'FAIL'}")

    best_lookup = A["mRNA abundance"]
    g4 = bool(A["loop"] > A["plainFBA"] and A["loop"] > best_lookup)
    say(f"\n  G4 BEATS THE CATALOGUE -- vs plainFBA {A['plainFBA']:.4f}, vs lookup {best_lookup:.4f}   "
        f"G4 {'PASS' if g4 else 'FAIL'}")

    a_look, _ = cv_auc(lr.reshape(-1, 1), y)
    a_both, _ = cv_auc(np.c_[lr, T["loop"], T["frozen"], T["plain"]], y)
    g5 = bool(a_both - a_look >= GATE_ADDS)
    say(f"\n  G5 ADDS TO THE LOOKUP (the fair version) -- 5-fold CV: {a_look:.4f} -> {a_both:.4f} "
        f"({a_both-a_look:+.4f})  (gate +{GATE_ADDS})   G5 {'PASS' if g5 else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"G1 closure": g1, "G2 signal": g2, "G3 feedback earns its place": g3,
             "G4 beats the catalogue": g4, "G5 adds to the lookup": g5}
    for kk, vv in gates.items():
        say(f"  {kk:<34}{'PASS' if vv else 'FAIL'}")
    if g3:
        say("  THE FEEDBACK EARNS ITS PLACE ON A MODEL THAT CANNOT EAT ATP. loop 1's G3 failure was")
        say("  recorded on a broken instrument and is superseded.")
    else:
        say("  THE FEEDBACK STILL DOES NOT EARN ITS PLACE. loop 1's verdict now stands on much better")
        say("  evidence than it originally had: it survives fixing the medium, fixing the reference")
        say("  growth rate, and re-running on the same genes with the same seed.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(GEM), str(GEMGENES), str(KDEG), str(CELL), str(OUT / "cell_loop.json")],
                      available=len(gids), used=len(T), selection="filtered", seed=SEED,
                      controls=[f"{N_SHUFFLE} label shuffles", "frozen-mu (feedback off)",
                                "plain FBA (no protein layer)", "expression lookup",
                                "loop 1's open-medium numbers as the direct comparator"],
                      note="identical code path and gates to cell_loop; only the medium, the capacity "
                           "layer and the re-solved reference mu differ")
    RM.report(man, emit=say)
    json.dump({"test": "loop_retest", "manifest": man, "gates": gates, "auc": A,
               "auc_open_medium": OPEN, "kappa": KAPPA, "mu_wt": MU_WT, "mu_ref": MU_REF,
               "n_movers": len(movers), "delta_g3": d3,
               "cv": {"lookup": a_look, "lookup_plus_loop": a_both},
               "null": {"mean": float(null.mean()), "sd": float(null.std()), "p95": p95},
               "scores": T.to_dict("records"), "log": log},
              open(OUT / "loop_retest.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_retest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
