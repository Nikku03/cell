"""WHAT MADE model4 -- reconstructed from the artefact itself, because the producer is gone.

WHERE THIS COMES FROM.  cell_complete_build listed 40 of 42 blocks as having NO KNOWN PRODUCER, and
`model4` sits at the top of that backlog because two loops now depend on it. loop 6 measured its
neighbour lists and loop 7 had to retract loop 6's headline after finding they are 8.8% cis. Both files
carry the same caveat: nothing records how those lists were built, so their interpretation is limited by
an unrecorded provenance.

This closes as much of that as can be closed WITHOUT the producer, by reading the artefact's own
numbers. `model4_meta` is five floats that nobody has looked at:

    lift 8.719008189262967   n_train 8792   pc_r2 -0.0503350952547124
    random_at10 0.0011375270162666364   recall_at10 0.009918107370336670

Those are not biology. They are machine-learning evaluation metrics, and they are internally consistent
enough to reconstruct the experiment that produced them.

THE RECONSTRUCTION, and every step is checkable arithmetic rather than inference:
    random_at10 is what a RANDOM ranker scores at k=10, which is 10/(number of candidates). Solving,
        10 / 0.0011375270162666364 = 8791 exactly. So the candidate pool was 8,791 genes -- n_train
        minus the query gene itself.
    lift is recall_at10 / random_at10, and that division reproduces the stored 8.719008189262967 to the
        last digit.
    So model4 is A LEARNED RETRIEVAL MODEL: for each of 8,792 training genes, rank the other 8,791 and
        keep the top few. `neighbors` is that ranking's output, not a measurement of anything.

WHY THIS MATTERS FOR TWO EARLIER LOOPS.  If the neighbour lists are model predictions rather than
measurements, then their quality is bounded by that model's recorded performance -- and the performance
is written right there in the metadata. Loops 6 and 7 measured real, controlled effects on those lists;
this loop measures how much the lists themselves are worth.

PREDECLARED, before any number:

    W1 THE ARITHMETIC RECONSTRUCTS   random_at10 equals 10/(n_train - 1) to floating-point exactness,
                and lift equals recall/random the same way.
                fails -> the metadata is not describing a top-10 retrieval evaluation and the whole
                reconstruction is wrong; say so and stop.
    W2 STORED AS TOP-K   every gene carries the same number of neighbours, which is what a top-k
                retrieval emits and is not what a contact map or an interaction assay emits.
    W3 THE PRODUCER IS ABSENT   no module in this repository emits these field names.
                fails -> the producer was here all along and the backlog is one shorter.
    W4 THE LISTS ARE RELIABLE   recall@10 of at least 5%.
                fails -> the neighbour lists come from a model that finds under one in twenty of its
                own targets, and every result built on them inherits that ceiling. This gate is
                expected to fail and it is the number loops 6 and 7 should be read against.

CONTROLS: the arithmetic is exact rather than approximate; the repository-wide grep is the same
mechanism cell_complete_build used to find producers; random_at10 is itself the built-in random baseline
of whoever trained the thing.

WHAT HAPPENED, written after the run against the gates above, unedited.

    W1 THE ARITHMETIC RECONSTRUCTS  PASS.  10/random_at10 = 8791 and 10/8791 reproduces the stored
        0.0011375270162666364 to the last bit; recall/random reproduces the stored lift
        8.719008189262967 exactly; 8791 = n_train - 1. The reconstruction is not an inference.
    W2 STORED AS TOP-K  PASS.  All 7,700 genes carry exactly 8 neighbours. A contact map or an
        interaction assay emits a VARIABLE count per locus; a fixed count is what a ranking emits.
        Worth recording: evaluated at k=10, stored at k=8.
    W3 THE PRODUCER IS ABSENT  PASS, on the second run. The first reported FAIL with exactly one hit --
        THIS FILE, which names those fields in its own source in order to grep for them. That is the
        same self-credit bug capability_audit hit twice, making it the single most repeatable mistake in
        this project: a checker that matches its own vocabulary. Self is now excluded.
    W4 THE LISTS ARE RELIABLE  FAIL, as predeclared.  recall@10 = 0.99%. The model that produced these
        neighbour lists finds under one in a hundred of its own targets in its top ten. It beats random
        by 8.7x, which is real; it is still 0.99%. And pc_r2 = -0.0503 is NEGATIVE, meaning that
        component predicted worse than a constant would have.

WHAT THIS CHANGES.  model4's backlog entry is no longer blank: it is a learned retrieval model's top-8
output over 8,791 candidates, with its own recorded recall@10 of 0.99% and a negative pc_r2. The producer
is still not in this repository, so the backlog is not one shorter -- but "unknown" has become "known to
be a weak ranker, and here is its scorecard".

AND IT INDEPENDENTLY CONFIRMS LOOP 7 BY A DIFFERENT ROUTE.  Loop 7 concluded model4 is not a fold from
GEOMETRY -- 8.8% cis where Hi-C runs 70-90%. This concludes the same thing from METADATA -- the block is
a ranking, and rankings are predictions. Two independent arguments, same answer, and neither needed the
producer.

HOW LOOPS 6 AND 7 SHOULD NOW BE READ.  Both measured real, controlled effects, and those measurements
stand. What this adds is the ceiling: the lists they measured come from a model that is right about 1%
of the time by its own metric. An effect measured on a weak predictor's output is an effect on a weak
predictor's output.

-> outputs/loop_provenance.json
"""
import collections
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"
SEED = 1904
GATE_RECALL = 0.05


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    say("=" * 100)
    say("WHAT MADE model4 -- reconstructed from the artefact, because the producer is gone")
    say("=" * 100)
    say("  40 of 42 blocks have no known producer. model4 tops that backlog because two loops depend on")
    say("  it. Its metadata is five floats nobody had looked at, and they are enough.")

    D = json.load(open(CELL))
    meta = D.get("model4_meta") or {}
    m4 = D.get("model4") or {}
    say(f"\n  model4_meta as stored:")
    for k, v in sorted(meta.items()):
        say(f"    {k:<14} {v!r}")

    # ---- W1 the arithmetic ---------------------------------------------------------------------------
    n_train = meta.get("n_train")
    r10 = meta.get("random_at10")
    rec = meta.get("recall_at10")
    lift = meta.get("lift")
    cands = None
    if r10:
        cands = int(round(10.0 / r10))
    exact_pool = bool(cands is not None and 10.0 / cands == r10)
    exact_lift = bool(rec and r10 and lift and abs(rec / r10 - lift) < 1e-12)
    say(f"\n  W1 THE ARITHMETIC RECONSTRUCTS")
    say(f"    10 / random_at10          = {10.0 / r10:.6f}  ->  {cands:,} candidates")
    say(f"    10 / {cands} reproduces random_at10 exactly : {exact_pool}")
    say(f"    n_train - 1               = {n_train - 1:,}   {'MATCHES' if cands == n_train - 1 else 'does not match'}")
    say(f"    recall/random = {rec / r10:.15f} vs stored lift {lift!r} : {exact_lift}")
    w1 = bool(exact_pool and exact_lift and cands == n_train - 1)
    say(f"    W1 {'PASS' if w1 else 'FAIL'}")
    if w1:
        say("    So: a learned RETRIEVAL model. For each of 8,792 genes, rank the other 8,791 and keep")
        say("    the top few. `neighbors` is a ranking's output. It is not a measurement of anything.")

    # ---- W2 stored as top-k --------------------------------------------------------------------------
    lens = collections.Counter(len(v.get("neighbors") or []) for v in m4.values())
    k_fixed = len(lens) == 1
    kk = next(iter(lens)) if k_fixed else None
    say(f"\n  W2 STORED AS TOP-K -- neighbour list lengths: {dict(lens)}")
    w2 = bool(k_fixed)
    say(f"    every one of {len(m4):,} genes carries exactly {kk} neighbours   W2 "
        f"{'PASS' if w2 else 'FAIL'}")
    if w2:
        say(f"    A fixed list length is what top-k retrieval emits. A contact map or an interaction")
        say(f"    assay emits a VARIABLE number, because different loci have different numbers of")
        say(f"    partners. Note the mismatch worth recording: evaluated at k=10, stored at k={kk}.")

    # ---- W3 producer absent --------------------------------------------------------------------------
    fields = ["recall_at10", "random_at10", "pc_r2", "model4_meta"]
    hits = []
    for f in fields:
        try:
            r = subprocess.run(["grep", "-rl", "--include=*.py", f, str(ROOT)],
                               capture_output=True, text=True, timeout=120)
            hits += [x for x in r.stdout.split() if x]
        except Exception:
            pass
    # EXCLUDE SELF. The first run reported W3 FAIL with exactly one hit: this file, which names those
    # fields in its own source in order to search for them. That is the same self-credit bug
    # capability_audit hit twice, and it is apparently the single most repeatable mistake in this
    # project -- a checker that matches its own vocabulary.
    hits = sorted(set(hits) - {str(Path(__file__).resolve())})
    w3 = not hits
    say(f"\n  W3 THE PRODUCER IS ABSENT -- python files emitting {fields}: {hits or 'NONE'}")
    say(f"    W3 {'PASS' if w3 else 'FAIL'}")

    # ---- W4 how good is the thing that made the lists? -----------------------------------------------
    say(f"\n  W4 THE LISTS ARE RELIABLE -- recall@10 {rec:.6f} = {rec:.2%}, "
        f"random {r10:.2%}, lift {lift:.2f}x")
    w4 = bool(rec >= GATE_RECALL)
    say(f"    gate >= {GATE_RECALL:.0%}   W4 {'PASS' if w4 else 'FAIL'}")
    if not w4:
        say(f"    The model that produced these neighbour lists finds {rec:.2%} of its own targets in")
        say(f"    its top ten. It beats random by {lift:.1f}x, which is real, and it is still under one")
        say(f"    in a hundred. And pc_r2 is {meta.get('pc_r2'):.4f} -- NEGATIVE, meaning that component")
        say(f"    predicted worse than a constant. Every result built on these lists inherits that")
        say(f"    ceiling, including loops 6 and 7.")

    say("\n" + "=" * 100)
    gates = {"W1 arithmetic reconstructs": w1, "W2 stored as top-k": w2,
             "W3 producer absent": w3, "W4 lists are reliable": w4}
    for k, v in gates.items():
        say(f"  {k:<34}{'PASS' if v else 'FAIL'}")
    if w1 and w2 and not w4:
        say("  PROVENANCE PARTLY RECOVERED, AND IT LOWERS THE VALUE OF TWO EARLIER LOOPS.")
        say("  model4 is a learned retrieval model's top-8 output over 8,791 candidates, with a recorded")
        say("  recall@10 of 0.99% and a negative pc_r2. The producer is still not in this repository, so")
        say("  the backlog is not shorter -- but the entry is no longer blank. Loops 6 and 7 measured")
        say("  real effects on those lists; what this says is what the lists themselves are worth.")
    elif not w1:
        say("  THE RECONSTRUCTION FAILED. The metadata does not describe a top-10 retrieval evaluation")
        say("  and nothing above should be quoted.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(CELL)], available=len(m4), used=len(m4), selection="all", seed=SEED,
                      controls=["exact floating-point reconstruction rather than approximate",
                                "repository-wide grep for the emitting module",
                                "random_at10 as the producer's own built-in random baseline"],
                      note="recovers what can be recovered without the producer, from the artefact")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_provenance", "manifest": man, "gates": gates,
               "meta": meta, "candidates": cands, "k_stored": kk,
               "producer_files": hits, "recall_at10": rec, "lift": lift,
               "conf_distribution": dict(collections.Counter(v.get("conf") for v in m4.values())),
               "log": log},
              open(OUT / "loop_provenance.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
