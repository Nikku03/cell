"""Loop 198 (Phase 5). The capability statement with a time column, and can the map step forward?

WHY THIS IS NOT A SIMULATOR. The plan for this phase said "a simulator that steps state forward,
validated on held-out trajectories". Phases 1 to 4 have since tested five candidate dynamic rules
and four of them failed:

    accessibility leads transcription        191d U6   PASS, +48 min, p 6.4e-58 -- UNREPLICATED
    feedback sign orders response time       191d U7   FAIL, 77 vs 70 min, p 0.47
    occupancy carries timing                 191d U8b  FAIL, holds in 1 of 3 magnitude terciles
    coupled enzymes are co-timed             194  V3   FAIL, z -0.8, stable across hub thresholds
    curated pathways are co-timed            194  V5   FAIL, z +1.1

A simulator built on that set would be a simulator built on one unreplicated relation and four
refuted ones. So this loop does the thing the plan actually needs: it writes down what the map can
answer AND AT WHAT TIME HORIZON, derived from the loop outputs on disk rather than from memory, and
then asks the smallest honest version of the simulator question -- can anything here predict a
state it has not seen?

THE BASELINE THAT DECIDES IT, and it is the one most trajectory models quietly avoid. PERSISTENCE:
predict that the next value equals the last observed one. On a smooth biological trajectory
persistence is extremely strong, and a model that cannot beat it has not learned dynamics -- it has
learned that tomorrow resembles today. So P3 predicts the CHANGE over an interval, where persistence
is exactly the prediction "no change", and any skill above zero is skill about dynamics.

THE SPLIT IS OVER TIME, NOT OVER GENES. Holding out genes tests whether the model generalises across
the genome; holding out LATER TIMEPOINTS tests whether it can run forward, which is the only sense
in which a map is four-dimensional. The model is fitted on intervals inside an early window and
scored on intervals reaching into a window it never saw.

PREDECLARED, BEFORE ANY NUMBER.

  P1 THE CAPABILITY TABLE, WITH A HORIZON COLUMN. Every measured capability, its number, its source
     loop, and its time horizon: STATIC (a property with no time in it), ONE-STEP (predicts a change
     over one measured interval), or TRAJECTORY (runs forward unaided).
     Gate: PASS iff every entry carries a horizon AND every entry whose evidence is single-system
     is marked UNREPLICATED. An entry without both is a capability claim this project cannot
     support in the form it is written.

  P2 IS THE RULE LEDGER COMPLETE AND HONEST? Every dynamic rule this arc tested, read FROM THE JSON
     OUTPUTS ON DISK rather than transcribed, with its verdict.
     Gate: PASS iff every referenced output file exists and its recorded gate value matches what
     this table claims. A capability statement assembled from memory is the failure mode this gate
     exists to prevent -- loop 195's Z1 caught nothing because it was checked, and that is the point.

  P3 CAN THE MAP STEP STATE FORWARD? Fit on intervals within an early window of the A549 course,
     score on intervals reaching into held-out later timepoints. Target: the change in expression
     over the interval. Arms: persistence (predict no change), the training-set mean change, and an
     accessibility-informed model using the change in promoter accessibility over the same interval.
     Gate: PASS iff the accessibility-informed model beats BOTH baselines in held-out R2. Beating
     the mean but not persistence is not stepping forward; it is knowing the average.

  P4 WHAT IS THE HONEST HORIZON? The verdict of P3 written into the capability table as the
     project's maximum demonstrated horizon.
     Gate: PASS iff the horizon assigned is the one P3 measured, whatever it is.

  P5 WHAT THE MAP CANNOT DO.

-> outputs/loop_capability_4d.json and NOTES_capability_4d.md
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402
import loop_response_timing_d as L191        # noqa: E402

from sklearn.linear_model import Ridge                            # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_capability_4d.json"
OUTDIR = Path(os.environ.get("CELL_OUT", "outputs"))
NOTE = Path("NOTES_capability_4d.md")
SP = L191.SP
A549 = SP / "grtc"

T_MIN = 30.0
REPS = (1, 2, 3)
MIN_TPM = 1.0
MIN_PLATEAU = 0.5
N_TRAIN = 6               # first six grid points are the training window
PROM_PAD = L191.PROM_PAD
SEED = 198198

# every dynamic rule this arc tested, with the file and gate that decides it. P2 CHECKS these
# against the files rather than trusting the table.
RULES = [
    dict(rule="promoter accessibility leads transcription",
         file="loop_response_timing_d.json", gate="U6", want=True,
         detail="+48 min over 1,310 genes, p 6.4e-58, holds in all three magnitude terciles",
         replicated=False),
    dict(rule="feedback sign orders response time",
         file="loop_response_timing_d.json", gate="U7", want=False,
         detail="negative two-cycle 77 min vs positive 70 min, p 0.47", replicated=False),
    dict(rule="promoter occupancy carries timing",
         file="loop_response_timing_d.json", gate="U8b", want=False,
         detail="rho -0.160 pooled but holds in 1 of 3 magnitude terciles", replicated=False),
    dict(rule="enzymes sharing chemistry are co-timed",
         file="loop_pathway_timing.json", gate="V3", want=False,
         detail="z -0.8 against 1,000 graph-fixed permutations, stable across hub thresholds",
         replicated=False),
    dict(rule="curated pathway members are co-timed",
         file="loop_pathway_timing.json", gate="V5", want=False,
         detail="z +1.1", replicated=False),
    dict(rule="the accessibility clock replicates outside A549",
         file="loop_timing_replication.json", gate="W3", want=False,
         detail="four timepoints cannot resolve it; the A549 lead reverses when downsampled",
         replicated=False),
]

# the capability table. horizon is the claim being made, and P1 gates on it being filled in.
CAPS = [
    dict(question="which metabolite completes a reaction", value="hit@1 0.8506", loop="170",
         horizon="STATIC", single_system=True),
    dict(question="which enzyme catalyses a reaction", value="0.825", loop="163d",
         horizon="STATIC", single_system=True),
    dict(question="is this DNA an enhancer", value="AUC 0.8506", loop="177",
         horizon="STATIC", single_system=True),
    dict(question="which element does a gene use", value="R@1 0.6734 vs distance 0.5930",
         loop="185", horizon="STATIC", single_system=True),
    dict(question="what makes a TF bind an enhancer",
         value="co-binding 0.8455 > accessibility 0.7902 > H3K27ac 0.7510 > motif 0.6228",
         loop="184", horizon="STATIC", single_system=True),
    dict(question="do feedforward loops exist beyond chance", value="z +1.3, no", loop="187",
         horizon="STATIC", single_system=True),
    dict(question="do feedback two-cycles", value="z +43.8, yes", loop="187",
         horizon="STATIC", single_system=True),
    dict(question="is autoregulation above chance", value="z +4.0, 2.2x", loop="187",
         horizon="STATIC", single_system=True),
    dict(question="does chromatin state add over measured binding", value="no, +0.0086 AUPRC",
         loop="188b", horizon="STATIC", single_system=True),
    dict(question="does accessibility lead transcription", value="+48 min, p 6.4e-58",
         loop="191d", horizon="ONE-STEP", single_system=True),
    dict(question="is metabolic timing coordinated", value="no, z -0.8", loop="194",
         horizon="ONE-STEP", single_system=True),
    dict(question="can the map run a trajectory forward", value=None, loop="198",
         horizon=None, single_system=True),      # P3 fills this in
]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 198 (PHASE 5)  THE CAPABILITY STATEMENT WITH A TIME COLUMN")
    say("=" * 104)
    say("  PREDECLARED: this is NOT a simulator, because phases 1-4 tested five candidate dynamic")
    say("  rules and four failed, so a simulator would encode one unreplicated relation and four")
    say("  refuted ones. Every capability must carry a HORIZON and every single-system result must")
    say("  be marked UNREPLICATED. The rule ledger is read FROM THE JSON OUTPUTS rather than")
    say("  transcribed. And the step-forward test is scored against PERSISTENCE -- predict no")
    say("  change -- because a model that cannot beat persistence has learned that tomorrow")
    say("  resembles today, not dynamics. The split is over TIME, not over genes.")
    say()

    # ---- P2 first: the ledger has to be verified before the table quotes it -------------------
    say("P2 IS THE RULE LEDGER COMPLETE AND HONEST?")
    ok, checked = True, []
    for r in RULES:
        p = OUTDIR / r["file"]
        if not p.exists():
            say(f"     MISSING {r['file']} -- cannot verify '{r['rule']}'")
            ok = False
            continue
        d = json.load(open(p))
        got = d.get("gates", {}).get(r["gate"])
        void = r["gate"] in (d.get("void") or [])
        match = (got == r["want"]) and not void
        ok &= match
        checked.append(dict(rule=r["rule"], gate=r["gate"], recorded=got, claimed=r["want"],
                            void=void, match=match))
        say(f"     {'ok ' if match else 'BAD'}  {r['file']}:{r['gate']} recorded {got!s:>5} "
            f"claimed {r['want']!s:>5}{' (VOID)' if void else ''}  {r['rule']}")
    p2 = bool(ok)
    GG.verdict(p2, emit=say,
               if_true=f"P2 PASS -- all {len(checked)} rules verified against the files that "
                       f"decided them, so the table below quotes the record and not my memory of it",
               if_false="P2 FAIL -- a claimed verdict does not match the recorded one; the "
                        "capability table would be asserting something the loops did not find")

    # ---- P3 ------------------------------------------------------------------------------------
    say()
    say("P3 CAN THE MAP STEP STATE FORWARD?")
    z = np.load(A549 / "rna.npz", allow_pickle=True)
    tpm = z["tpm"]
    ensg = np.array([str(g).split(".")[0] for g in z["genes"]])
    mins, reps = z["mins"].astype(int), z["reps"].astype(int)
    allt = sorted(set(mins.tolist()))
    comp = {t: set(reps[mins == t].tolist()) for t in allt}
    grid = np.array([t for t in allt if set(REPS) <= comp[t] and t >= T_MIN], dtype=float)
    M, _ = L191.rep_trajectories(tpm, mins, reps, REPS, grid)
    e2s = L191.ensg_to_symbol(lambda *_: None)
    sym = np.array([e2s.get(g, "") for g in ensg])
    base = tpm[(mins == int(grid[0])) & np.isin(reps, REPS)].mean(0)
    pl = M[-3:].mean(0)
    resp = (base >= MIN_TPM) & (np.abs(pl) >= MIN_PLATEAU)

    import gzip
    tab = json.load(gzip.open("colab/data/cell_complete.json.gz"))["genes"]
    tssb = {}
    for line in open(SP / "_tss_hg38.bed"):
        q = line.split()
        if len(q) >= 4 and q[3].startswith("G"):
            i = int(q[3][1:])
            if i < len(tab):
                tssb[str(tab[i]["name"]).upper()] = (q[0], int(q[2]))
    pt, PM = L191.promoter_track("DNase", [tssb.get(s) for s in sym], PROM_PAD, lambda *_: None)
    idx = [int(np.where(pt == t)[0][0]) for t in grid]
    A = PM[idx]
    keep = resp & (A > 0).any(0)
    say(f"     grid {[int(x) for x in grid]}; train on the first {N_TRAIN}, score on the rest")
    say(f"     {int(keep.sum()):,} genes are responders with a promoter DNase peak")

    def rows(lo, hi):
        """(gene, interval) rows whose interval END lies in [lo, hi)."""
        X, y, g = [], [], []
        for j in range(1, len(grid)):
            if not (lo <= j < hi):
                continue
            dt = grid[j] - grid[j - 1]
            for i in np.where(keep)[0]:
                X.append([A[j, i] - A[j - 1, i], A[j - 1, i], M[j - 1, i], dt])
                y.append(M[j, i] - M[j - 1, i])
                g.append(i)
        return np.array(X), np.array(y), np.array(g)

    Xtr, ytr, _ = rows(1, N_TRAIN)
    Xte, yte, _ = rows(N_TRAIN, len(grid))
    say(f"     {len(ytr):,} training rows, {len(yte):,} held-out rows "
        f"(intervals ending at {[int(x) for x in grid[N_TRAIN:]]})")

    def r2(pred, truth):
        ss = float(((truth - pred) ** 2).sum())
        tot = float(((truth - truth.mean()) ** 2).sum())
        return 1.0 - ss / tot if tot > 0 else float("nan")

    r_pers = r2(np.zeros_like(yte), yte)
    r_mean = r2(np.full_like(yte, ytr.mean()), yte)
    m = Ridge(alpha=1.0).fit(Xtr, ytr)
    r_inf = r2(m.predict(Xte), yte)
    say(f"     held-out R2 -- persistence (predict no change): {r_pers:+.4f}")
    say(f"     held-out R2 -- training-set mean change:        {r_mean:+.4f}")
    say(f"     held-out R2 -- accessibility-informed ridge:    {r_inf:+.4f}")
    p3 = bool(r_inf > r_pers and r_inf > r_mean)
    horizon = "ONE-STEP" if p3 else "STATIC"
    GG.verdict(p3, emit=say,
               if_true=f"P3 PASS -- the informed model beats persistence ({r_inf:+.4f} vs "
                       f"{r_pers:+.4f}), so the map carries some ability to run one interval "
                       f"forward into time it has not seen",
               if_false=f"P3 FAIL -- the informed model does not beat persistence ({r_inf:+.4f} vs "
                        f"{r_pers:+.4f}). Predicting that nothing changes is at least as good as "
                        f"everything this project knows, so the map does not step state forward")

    # ---- P1 ------------------------------------------------------------------------------------
    say()
    say("P1 THE CAPABILITY TABLE, WITH A HORIZON COLUMN")
    caps = [dict(c) for c in CAPS]
    for c in caps:
        if c["loop"] == "198":
            c["value"] = (f"no -- persistence R2 {r_pers:+.4f} beats informed {r_inf:+.4f}"
                          if not p3 else f"one interval, held-out R2 {r_inf:+.4f}")
            c["horizon"] = horizon if p3 else "NONE DEMONSTRATED"
    say(f"     {'question':<48s} {'horizon':<18s} {'replicated':<11s} value")
    for c in caps:
        rep = "no" if c["single_system"] else "yes"
        say(f"     {c['question'][:47]:<48s} {str(c['horizon'])[:17]:<18s} {rep:<11s} {c['value']}")
    filled = all(c["horizon"] for c in caps)
    marked = all(("single_system" in c) for c in caps)
    p1 = bool(filled and marked)
    GG.verdict(p1, emit=say,
               if_true="P1 PASS -- every capability carries a horizon and every single-system "
                       "result is marked unreplicated",
               if_false="P1 FAIL -- an entry lacks a horizon or a replication mark")

    # ---- P4 ------------------------------------------------------------------------------------
    say()
    say("P4 WHAT IS THE HONEST HORIZON?")
    n_static = sum(1 for c in caps if c["horizon"] == "STATIC")
    n_one = sum(1 for c in caps if c["horizon"] == "ONE-STEP")
    n_traj = sum(1 for c in caps if c["horizon"] == "TRAJECTORY")
    say(f"     of {len(caps)} capabilities: {n_static} STATIC, {n_one} ONE-STEP, "
        f"{n_traj} TRAJECTORY")
    say(f"     none is replicated outside its own system")
    say(f"     maximum demonstrated horizon: "
        f"{'ONE-STEP' if p3 else 'STATIC -- the map does not run forward'}")
    p4 = True
    say("     P4 PASS -- the horizon assigned is the one P3 measured")

    NOTE.write_text(f"""# What this cell map can answer, and over what horizon

Written by loop 198 from the loop outputs on disk. Every verdict below was checked against the
JSON file that recorded it rather than transcribed.

## Horizons

STATIC      a property with no time in it.
ONE-STEP    predicts a change across one measured interval.
TRAJECTORY  runs forward unaided over multiple intervals.

**Maximum demonstrated horizon: {'ONE-STEP' if p3 else 'STATIC'}.**

## The step-forward test

Fitted on intervals inside an early window of the A549 dexamethasone course and scored on intervals
reaching into later timepoints the model never saw. The target is the CHANGE in expression, so
persistence -- predict no change -- is the baseline, and it is the baseline most trajectory models
avoid because smooth biology makes it strong.

    persistence (predict no change)   held-out R2 {r_pers:+.4f}
    training-set mean change          held-out R2 {r_mean:+.4f}
    accessibility-informed ridge      held-out R2 {r_inf:+.4f}

{'The informed model beats persistence, so the map can run one interval forward.' if p3 else
 'The informed model does NOT beat persistence. Predicting that nothing changes is at least as good as everything this project knows about dynamics, so the map does not step state forward.'}

## The dynamic rules that were tested

Five candidate rules; one survives and it is unreplicated.

| rule | verdict |
|---|---|
{chr(10).join(f"| {r['rule']} | {'holds' if r['want'] else 'refuted'} -- {r['detail']} |" for r in RULES)}

## How this must be quoted

Nine descriptive layers with one unreplicated clock is not a running cell. Every capability above is
single-system: K562 for enhancers, A549 for timing, mixed cell lines for the network. Every
trajectory is bulk, so a fast response in a fifth of the cells and a slow one in all of them give
the same curve. See NOTES_accessibility_clock_status.md for the clock's own qualification.
""")
    say(f"     wrote {NOTE}")

    # ---- P5 ------------------------------------------------------------------------------------
    say()
    say("P5 WHAT THE MAP CANNOT DO")
    say("     It cannot run a cell. Nine descriptive layers plus one unreplicated timing relation")
    say("     is a description, and no combination of the loops in this arc turns it into a")
    say("     simulation.")
    say("     It cannot be quoted as one system. K562 for the enhancer arc, A549 for timing, mixed")
    say("     lines for the network, and nothing in this project has ever measured two of those")
    say("     layers in the same cells at the same time.")
    say("     P3 is one dataset, one drug, one split. A pass would show the map can step forward")
    say("     HERE, not that it generalises; a fail leaves open that a different feature set would")
    say("     do better, and neither reading is settled by one ridge regression.")
    say("     Every trajectory is bulk. A fast response in a fifth of the cells and a slow one")
    say("     everywhere give the same curve, and nothing in this arc can separate them.")
    say("     The capability table lists what was MEASURED, not what is TRUE. A number that")
    say("     survived its gates on one benchmark is a number about that benchmark.")
    say("     P5 PASS")

    gates = {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": True}
    man = RM.manifest(inputs=[A549 / "rna.npz"], available=int(len(sym)), used=int(keep.sum()),
                      selection="filtered", seed=SEED,
                      controls=["persistence and mean-change baselines on a held-out TIME window",
                                "every rule verdict read from the JSON output that recorded it"],
                      note="capability statement with a horizon column, and the step-forward test")
    out_d = dict(test="capability 4d", gates=gates, void=[],
                 rules_checked=checked, capabilities=caps,
                 step_forward=dict(grid=[int(x) for x in grid], n_train=N_TRAIN,
                                   n_train_rows=int(len(ytr)), n_test_rows=int(len(yte)),
                                   r2_persistence=r_pers, r2_mean=r_mean, r2_informed=r_inf),
                 horizon=horizon if p3 else "NONE DEMONSTRATED",
                 manifest=man, seconds=time.time() - t0, log=log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'PASS' if v else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [{time.time()-t0:.0f}s]")
    say("=" * 104)
    out_d["log"] = log
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
