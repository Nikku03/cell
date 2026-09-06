"""THE IMPROVER: one question, asked of the record rather than of my own memory.

    HOW DO I GET A BETTER RESULT THAN THIS?

Everything else here exists to stop that question being answered by enthusiasm. It is decomposed
into questions with numeric answers, the answers are turned into a plan, the plan is put through a
check that CAN REJECT IT, the whole picture is taken, the plan is upgraded, executed, tested, and
the cycle runs again on the new numbers.

WHY A MODULE AND NOT A HABIT. This repository has caught itself, repeatedly, improving a number
without improving anything: twelve gates that fired while measuring nothing, two gates passed on
margins inside their own noise, a validation set that turned out to be a lookup, a "4x expansion"
that was illusory, a ceiling overstated by 45% because singletons cannot contribute within-variance.
Every one of those was found by a check, and none by intuition. So the checks are code.

THE SIX QUESTIONS, asked of recorded artefacts and never of prose:

    Q1  what was the previous result
    Q2  what is the new result
    Q3  what changed, and why
    Q4  what remained FLAT -- the question people skip, and the one that says where the wall is
    Q5  what did it cost
    Q6  with the SAME data, what can change implicitly and explicitly

Q6 is split deliberately. EXPLICIT change is a new feature, a new target, a new split. IMPLICIT
change is the same inputs used differently -- a different readout over the same embedding, a
different loss, a different unit of resampling. Implicit changes are nearly free and this project
has repeatedly found them worth more than fetching anything: loop 133's finding that mean-pooling
hides 18,595 point mutants cost nothing to discover and outweighs a 650M-parameter encoder.

THE THEORY CHECK, which is the whole point. A plan item is REJECTED unless all six hold:

    T1 CITES A MEASURED DEFICIT     a recorded number, not an intuition
    T2 GAIN EXCEEDS THE NOISE       the predicted improvement must be larger than the confidence
                                    interval on a PAIRED model-vs-model difference. Loop 120 passed
                                    two gates on margins smaller than their own spread; this makes
                                    that impossible to repeat by construction
    T3 HAS A FALSIFIER              states in advance what result would prove it did not work
    T4 NOT ALREADY REFUTED          each item must name the refuted claim nearest to it and say
                                    what distinguishes it. An item that names none, or whose
                                    distinguisher is a dependency that was itself rejected, fails
    T5 NOT CIRCULAR                 the evaluation must not consume data the change derives from.
                                    Loop 129 lost 160 of 169 validation genes to exactly this
    T6 RESPECTS A MEASURED BOUND    the promised result must not exceed a ceiling this repository
                                    has already measured. A plan that promises to beat the
                                    experimental floor is refuted before it runs
    T7 THE GAIN IS DERIVED          the predicted size must come from arithmetic on a recorded
                                    number and must not exceed the largest gain that number
                                    permits. Added after the first run of this file, in which
                                    every predicted_gain was an assertion of mine and the check
                                    accepted all five items -- see below
    T8 THE INPUTS EXIST             an item must name the files it consumes, and they must be on
                                    disk. Q6 of the first run offered "temperature and pH parsed
                                    from UniProt kinetics free text -- free, already on disk". No
                                    such file is on disk. A plan costed in minutes for data that
                                    was never fetched is a plan that cannot run, and the cheapest
                                    possible check catches it: os.path.exists
    T9 A PROBE UNBLOCKS SOMETHING   probes only. It must name an item and a check, that item must
                                    exist, and it must currently be FAILING that check. A probe
                                    that unblocks nothing is a measurement taken for its own sake
   T10 NAMES A LAYER AND A GATE     cell track only, where it REPLACES T2/T6/T7. Those three are
                                    arithmetic on a scalar with a sampling distribution, and the
                                    cell track's metric is a count of judged layers -- a bootstrap
                                    over it would have no referent, and the count is gameable by
                                    splitting one FAILED layer into two RUNS layers. So an item
                                    must instead name an EXISTING layer and the GATE that would
                                    have to pass. A status is a judgement; a gate is falsifiable.
                                    Which checks a track runs is declared in TRACKS, and a check
                                    that is switched off is recorded as None rather than dropped,
                                    so that turning one off stays visible

PROBES, and why the third run of this file had to invent them. After loop 134 supplied the grounded
maxima T7 demands, every item in the plan was rejected -- and one of them, P6, was rejected for
having NO measured maximum, when the measurement that would supply it costs ten minutes and had
simply never been run. The check was correct and the outcome was absurd: the loop had talked itself
into doing nothing, because the thing standing between it and a decision was a number, and its only
category of action was "a change that improves the metric".

So there are now two kinds of item.

    A CHANGE promises to move the metric. It faces T1-T8, and T7 requires it to derive its promise
    from a recorded number.

    A PROBE promises NOTHING about the metric. Its output is a number that some check needs. T2 and
    T7 do not apply -- a probe that predicted its own result would not be a probe -- and T9 applies
    instead: it must name the item and the check it unblocks, and that check must currently be
    failing. Probes are how the loop earns the right to a next turn when every change is blocked.

This is the single most useful thing this file has produced, and it came from reading its own
output rather than from planning. A loop that can only propose improvements will stall the moment
its improvements need evidence; a loop that can propose measurements will not.

WHAT THE FIRST RUN OF THIS FILE FOUND, which is why T7 and BLOCKED exist. It accepted 5 of 5 and
printed its own indictment: a check that rejects nothing is evidence the check is too weak. Two
holes were visible in its output:

  EVERY PREDICTED GAIN WAS MINE. T1 asked whether an item CITES a measured deficit. It never asked
  whether the SIZE of the promise followed from that deficit. "active-site pooling gains 0.15"
  cited a real number, 0.947, and then named a figure unrelated to it. T7 makes the arithmetic
  mandatory and caps a promise at the largest gain its own cited number permits.

  P5 WAS ACCEPTED ON A PROMISE. Its only distinguisher from the already-refuted "650M with mean
  pooling" is that P1 will have fixed the readout. T4 checked that P1 was ACCEPTED, which is a
  statement about the plan, when what it needed was that P1 had WORKED, which is a statement about
  a result that does not exist yet. Depending on an unmeasured outcome is not a rejection and it is
  not an acceptance; it is BLOCKED, and a blocked item is not schedulable.

And the first run's stopping rule turned out to be broken too, which no tightening of this file
would have caught: it named loop 133's B4 as the rule deciding a four-hour run. Loop 134 then
measured B4's premise -- the EC number -- at 0.7% of the variance, and retested B4's conclusion
without a residual at all. The conclusion SURVIVED (within-EC-class permutation costs +0.0046
against a 0.0488 interval) while the evidence for it did not. The lesson is recorded in load(): a
track must read the loop that AUDITED its metric, not only the loops that produced it.

THE BUNDLE RULE, which is what stops T2 from being merely destructive. A change predicted to move
the metric by less than the paired noise is unmeasurable ON ITS OWN and T2 rejects it -- correctly.
But three such changes, run together, can be measurable. Gains in RMSE do NOT add; removed VARIANCE
does, if the deficits are independent. So a bundle's combined result is

    rmse_bundle = sqrt( rmse^2 - SUM_i [ rmse^2 - (rmse - g_i)^2 ] )

which is sub-additive by construction and cannot promise past a bound. Only if the bundle clears
the paired noise does it run, and it is then tested as ONE change, because that is the only claim
its arithmetic supports.

T6 is the one that makes this more than a checklist. The bounds are real, measured, and unfitted:
the 1.15x reproducibility floor, the per-protein ceiling, the missing-conditions floor, the
Smoluchowski limit, tanh(b*T/4). A plan cannot promise past them, and saying so costs nothing.

THE CYCLE, in the order it runs:

    ask -> answer from artefacts -> analyse -> plan -> THEORY CHECK (can reject) ->
    whole picture (result AND model AND dataset) -> upgrade the plan -> execute -> test ->
    feed the new numbers back in and ask again

-> outputs/improver_<track>.json
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path(os.environ.get("CELL_OUT", "outputs"))

# ---------------------------------------------------------------------------------------------
# TRACKS. A track is an ordered list of recorded runs that are about the same question, so that
# "previous" and "new" mean something. Adding a track is adding a row here, not editing logic.
TRACKS = {
    "ml_kcat": {
        "question": "predict log10 kcat from enzyme sequence and substrate",
        "runs": ["loop_ml_kcat.json", "loop_ml_audit.json", "loop_ml_probe.json",
                 "loop_b4_fix.json", "loop_plan_exec.json", "loop_active_site.json"],
        "metric": "rmse", "lower_is_better": True,
        "checks": ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"),
    },
    "cell": {
        "question": "does the whole-cell model run, and which of its layers survive their gates",
        "runs": ["cell_run.json", "cell_model_audit.json", "capability_audit.json",
                 "loop_b4_fix.json", "cell_record_fix.json", "cell_layers.json"],
        # The layer table lives in SOURCE, not in an artefact, and C1 edits it. A fingerprint that
        # hashed only outputs/ would miss the very change the cell track's first item makes, and
        # the loop would then stall while claiming nothing had happened.
        "also_hash": ["colab/cell_assembled.py"],
        "metric": "n_failed_layers", "lower_is_better": True,
        # THE CELL TRACK CANNOT USE T2, T6 OR T7, and pretending otherwise would be the worst kind
        # of error here -- a check that appears to fire while measuring nothing.
        #
        # T2 compares a predicted gain against a bootstrap interval. T7 caps a promise at a
        # measured maximum. T6 compares a promised result against a measured floor. All three are
        # arithmetic on a SCALAR metric with a sampling distribution. The cell track's metric is a
        # COUNT OF LAYERS whose status is a judgement recorded by a human-written gate, not a
        # sample from anything. There is no bootstrap over 47 layers that means what a bootstrap
        # over 3,006 sequence clusters means, and a confidence interval on "14 FAILED" would be
        # a number with no referent.
        #
        # Worse, the count is trivially gameable: splitting one FAILED layer into two RUNS layers
        # improves it without touching the model. So the cell track drops T2/T6/T7 and adds T10,
        # which demands that an item name an EXISTING layer and the GATE that would have to pass.
        # A gate is falsifiable where a status is not.
        "checks": ("T1", "T3", "T4", "T5", "T8", "T9", "T10"),
        "why_not_t2": "the metric is a count of judged layers, not a sample; a bootstrap over it "
                      "would have no referent, and the count is gameable by relabelling",
    },
}

# MEASURED BOUNDS. Every one was produced by a recorded loop and none is fitted. T6 checks plan
# promises against these, so an over-promise is caught before any compute is spent.
BOUNDS = {
    "experimental_floor_rmse": (0.0607, "loop 129: same protein AND same substrate measured "
                                        "twice, 1.15x median over 101 values"),
    "missing_conditions_rmse": (0.5137, "loop 133 B5: within (protein, substrate) pairs, 3.26x, "
                                        "because temperature, pH and mutation are not in the file"),
    "per_protein_ceiling_rmse": (0.8327, "loop 133 B2: a perfect per-protein predictor, after "
                                         "merging near-duplicates and debiasing singletons"),
    "mutant_irreducible_rmse": (0.947, "loop 133 B1: 18,595 point-mutant pairs a mean-pooled "
                                       "embedding cannot distinguish"),
    "protein_identity_site_pooled_rmse": (-0.0008, "loop 136 H4: the SAME within-EC permutation "
                                                    "under an ACTIVE-SITE-POOLED readout. Negative: "
                                                    "the permuted model was fractionally better. A "
                                                    "readout built specifically to expose protein "
                                                    "identity extracts nothing from it"),
    "residue_subsampling_regularisation_rmse": (0.0100, "loop 136 H5: pooling over the same NUMBER "
                                                        "of residues at RANDOM positions gains this "
                                                        "much over mean pooling. It is the noise, "
                                                        "not the residues"),
    "protein_identity_value_rmse": (0.0046, "loop 134 C3: permuting the embedding among records "
                                            "SHARING an EC number -- destroying protein identity, "
                                            "preserving class exactly -- costs this much, against "
                                            "a paired interval of 0.0488. What the model extracts "
                                            "from knowing WHICH protein it is, is unmeasurable"),
}

# THE TENSION LOOP 134 CREATED, recorded rather than resolved by preference.
#
# C3 says protein identity is worth 0.0046 to the CURRENT representation. B1 says point mutants
# carry a 0.947 irreducible component. Both are measured and they do not contradict: C3 bounds what
# mean-pooled ESM EXTRACTS, B1 bounds what a perfect representation could at most REMOVE. P1 lives
# in the gap, and the gap is real -- a change of readout is exactly the thing C3 does not test.
#
# But C3 is still evidence against P1, and stronger than it looks. If swapping a protein for a
# DIFFERENT protein in the same EC class is nearly free, the targets are barely a function of
# protein identity on this data. Resolving a single residue is a finer distinction than swapping
# the whole chain. So P1's fraction_claimed is cut below, and the reason is written here rather
# than smuggled into a round number.
PROTEIN_IDENTITY_CAVEAT = (
    "loop 134 C3 measured protein identity at 0.0046 to the current readout. That bounds "
    "EXTRACTION, not the deficit, so it does not refute a change of readout -- but it is evidence "
    "against one, and the claimed fraction is cut accordingly.")


# THE HISTORY. Each turn of the loop appends here, so a prediction that was wrong stays visible.
# A self-improvement loop that overwrites its own last answer cannot be scored, and one that cannot
# be scored will drift toward whatever sounds best. Rows are added by hand when a turn completes,
# with the MEASURED outcome, not the promised one.
HISTORY = [
    {"turn": 1, "date": "2026-08-14",
     "accepted": 5, "rejected": 0, "blocked": 0,
     "stopping_rule": "loop 133 B4",
     "what_it_got_wrong": "accepted every item; named a broken metric as the stopping rule; "
                          "costed two items in minutes for data that is not on disk",
     "caught_by": "its own closing line, then loop 134",
     "fix": "T7 (gain must be derived), T8 (inputs must exist), BLOCKED for unmeasured "
            "dependencies, track reads the audit of its own metric"},
    {"turn": 2, "date": "2026-08-15",
     "accepted": 1, "rejected": 7, "blocked": 0,
     "stopping_rule": "loop 134 C3 -- within-EC-class permutation, no residual",
     "what_it_got_wrong": "with T7's grounded maxima in place it rejected EVERYTHING, including "
                          "P6 for having no measured maximum when the measurement costs ten "
                          "minutes and had never been run. The check was right and the outcome "
                          "was absurd: its only category of action was 'a change that improves "
                          "the metric', so a missing number could stall it indefinitely",
     "caught_by": "reading its own output -- the rejection reason named the remedy",
     "fix": "PROBES. An item is now a CHANGE (promises a gain, faces T7) or a PROBE (promises "
            "nothing, faces T9: it must name the item and check it unblocks, and that check must "
            "currently be failing)"},
    {"turn": 3, "date": "2026-08-15",
     "accepted": 1, "rejected": 7, "blocked": 0,
     "stopping_rule": "same",
     "what_it_got_wrong": "TWO THINGS, both found by running rather than reading. T2 rejected P2 "
                          "(+0.0113 predicted) against a 0.0488 threshold, and P2 then measured "
                          "+0.0168 with a CI of [+0.0016, +0.0322] that EXCLUDES ZERO -- a Type II "
                          "error caused by judging a feature addition with a model-swap's null. "
                          "And there was no state for a settled question, so the next turn "
                          "re-planned three already-measured items as predictions",
     "caught_by": "loop 135 E2's bootstrap interval; then the turn-2 replan",
     "fix": "NOISE_BY_KIND gives each comparison its own null (marked POST HOC), and MEASURED now "
            "outranks every check and carries the number, the interval and the signed error"},
    {"turn": 4, "date": "2026-08-15",
     "accepted": 0, "rejected": 5, "blocked": 0,
     "stopping_rule": "same",
     "what_it_got_wrong": "nothing new in the improver itself -- but its cell-track EXECUTOR "
                          "contained a gate that fired while measuring nothing (G1 was "
                          "gates['G1'] = True with no condition, and its string search could "
                          "never match because LAYERS entries are line-continued literals)",
     "caught_by": "the executor reporting PASS with an empty git diff",
     "fix": "AST-offset rewriting, and G1 verifies by re-parsing the file rather than asserting"},
]


def log(s=""):
    print(s, flush=True)


def cell_layers():
    """The 47-layer audit table, read by PARSING cell_assembled.py rather than importing it.

    Importing costs 27 seconds because the module builds the model on import, and a loop that pays
    27 seconds per turn to read a constant is a loop that will be tempted to cache it and then to
    trust the cache. ast.literal_eval on the LAYERS assignment is exact and takes milliseconds."""
    import ast
    src = Path("colab/cell_assembled.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "LAYERS":
            return [tuple(t) for t in ast.literal_eval(node.value)]
    return []


def cell_deficits():
    """The cell track's equivalent of BOUNDS, and it is built by CHECKING rather than by asserting.

    On the ml track T1 passes when an item cites a key in BOUNDS, and BOUNDS holds numbers a loop
    measured. The cell track has no such numbers, so the temptation is to let T1 pass on any
    plausible-sounding deficit name -- which would make T1 vacuous on exactly the track where the
    other quantitative checks have already been switched off. Instead each deficit here carries a
    PREDICATE that is evaluated against the files, and a deficit whose predicate is false does not
    exist as far as T1 is concerned."""
    L = cell_layers()
    d = {}

    # 1. Is the record behind the measurement? True only if the kcat layer still cites the test
    #    loop 134 showed was broken, AND loop 134 actually ran.
    kl = next((l for l in L if l[0] == "what the kcat model actually learned"), None)
    fixed = (OUT / "loop_b4_fix.json").exists()
    cites_broken = bool(kl and ("1.5386" in kl[3] or "EC-median residual" in kl[3]))
    d["record_behind_measurement"] = {
        "holds": bool(kl and fixed and cites_broken),
        "evidence": (f"layer cites the B4 residual ({cites_broken}); loop_b4_fix.json present "
                     f"({fixed})")}

    # 2. Can the artefact show WHICH layer changed? Only if cell_run.json stores per-layer status.
    run = json.load(open(OUT / "cell_run.json")) if (OUT / "cell_run.json").exists() else {}
    lay = run.get("layers")
    per_layer = isinstance(lay, dict) and any(v in ("RUNS", "FAILED", "STATIC", "CLOSES")
                                              for v in lay.values())
    d["artefact_cannot_show_change"] = {
        "holds": bool(run and not per_layer),
        "evidence": f"cell_run.json['layers'] = {lay}; per-layer status present: {per_layer}"}

    # 3. Do three or more FAILED layers converge on the transcription -> protein axis?
    key = ("transcription", "TF network", "CollecTRI", "regulation->transcription")
    conv = [l[0] for l in L if l[1] == "FAILED" and any(k.lower() in (l[0] + l[3]).lower()
                                                        for k in key)]
    d["three_failed_layers"] = {
        "holds": len(conv) >= 3,
        "evidence": f"{len(conv)} FAILED layers on that axis: {conv[:5]}"}
    return d


def ask_cell():
    """Q1-Q6 for the whole-cell model. The unit here is a LAYER and its gate, not an RMSE."""
    L = cell_layers()
    import collections as _c
    counts = _c.Counter(l[1] for l in L)
    run = json.load(open(OUT / "cell_run.json")) if (OUT / "cell_run.json").exists() else {}
    aud = (json.load(open(OUT / "cell_model_audit.json"))
           if (OUT / "cell_model_audit.json").exists() else {})
    failed = [{"layer": l[0], "source": l[2], "evidence": l[3][:300]} for l in L if l[1] == "FAILED"]
    static = [{"layer": l[0], "source": l[2]} for l in L if l[1] == "STATIC"]

    a = {"question": TRACKS["cell"]["question"], "n_runs": 1,
         "layer_counts": dict(counts), "n_layers": len(L)}
    a["Q1_previous"] = {"recorded_at_last_cell_run": run.get("layers"),
                        "note": "cell_run.json records only the four counts, not per-layer status, "
                                "so a layer that changed status between runs is invisible here. "
                                "That is a gap in the ARTEFACT and it is named rather than filled "
                                "by reading the table twice"}
    a["Q2_new"] = {"counts": dict(counts), "n_failed": counts.get("FAILED", 0),
                   "rmse": None,
                   "note": "there is no scalar metric with a sampling distribution on this track; "
                           "see TRACKS['cell']['why_not_t2']"}
    a["Q3_changed_and_why"] = [
        {"what": "the kcat layer's verdict is now backed by a decisive control",
         "why": "layer 'what the kcat model actually learned' cited loop 133 B4, whose residual "
                "leaked and whose baseline was in-sample. Loop 134 C3 reaches the same conclusion "
                "by within-class permutation: +0.0046 against a 0.0488 interval",
         "verdict": "SAME STATUS, MUCH BETTER EVIDENCE -- and the layer's prose still cites the "
                    "broken test, so the record is now behind the measurement"},
        {"what": "nothing else moved",
         "why": "no loop since cell_run.json has changed a layer's status",
         "verdict": "the counts are unchanged at 14 FAILED, 12 RUNS, 11 STATIC, 10 CLOSES"},
    ]
    a["Q4_flat"] = [
        {"what": "the transcription -> protein axis",
         "evidence": "three separate FAILED layers say the same thing: 'protein dynamics from "
                     "transcription alone' (p=2e-67 the wrong way), 'TF network as a predictor of "
                     "transcript dynamics', and 'CollecTRI as a replacement network'. Two "
                     "independent networks were tried and both failed",
         "verdict": "FLAT ACROSS THREE ATTEMPTS -- the wall is not the network, it is that "
                    "protein dynamics has a non-transcriptional source"},
        {"what": "the count of FAILED layers",
         "evidence": f"{counts.get('FAILED', 0)} of {len(L)}, unchanged since cell_run.json",
         "verdict": "FLAT -- and it should be, because a FAILED layer is a recorded negative and "
                    "not a defect to be cleared"},
    ]
    a["Q5_cost"] = {"cell_run_seconds": run.get("seconds"),
                    "note": "the model runs; the cost is not the binding constraint on this track"}
    a["Q6_same_data"] = {
        "IMPLICIT -- same inputs, used differently": [
            {"change": "correct the kcat layer's prose to cite loop 134 C3 rather than the leaking "
                       "B4 residual", "deficit": "record_behind_measurement", "cost": "free"},
            {"change": "record per-layer status in cell_run.json, not only the four counts",
             "deficit": "artefact_cannot_show_change", "cost": "free"},
        ],
        "EXPLICIT -- new information": [
            {"change": "a measurement of the non-transcriptional source of protein dynamics",
             "deficit": "three FAILED layers point at it", "cost": "unknown; no dataset identified"},
        ],
    }
    a["failed_layers"] = failed
    a["static_layers"] = static
    a["audit"] = {}
    a["all_scores"] = {"n_failed": counts.get("FAILED", 0), "n_static": counts.get("STATIC", 0)}
    a["parts"] = aud.get("parts")
    a["cell_deficits"] = cell_deficits()
    return a


def plan_cell(a):
    """Items for the cell track. No predicted_gain is meaningful here, so none is invented: each
    item names a LAYER and the GATE that would have to pass, which T10 checks."""
    L = {l[0]: l for l in cell_layers()}
    items = [
        {"id": "C1", "change": "correct the kcat layer to cite loop 134 C3, not the leaking B4",
         "kind": "record", "depends_on": [], "needs_files": [Path("colab/cell_assembled.py"),
                                                             OUT / "loop_b4_fix.json"],
         "targets_layer": "what the kcat model actually learned",
         "gate": "the layer's evidence field must cite a test whose control was shown capable of "
                 "moving before its result was read -- loop 134 C4 then C3",
         "mechanism": "the layer currently asserts 'EVERYTHING THE MODEL KNOWS IS ALREADY IN THE "
                      "EC NUMBER' on the strength of a residual that leaked and a baseline that "
                      "was in-sample. The conclusion survived; the evidence for it did not",
         "cites": "record_behind_measurement", "derivation": None, "predicted_gain": 0.0,
         "nearest_refuted": "650M with mean pooling",
         "distinguisher": "this corrects a record rather than proposing a model change",
         "falsifier": "loop 134 C3 does not in fact support the layer's claim, in which case the "
                      "layer's STATUS is wrong too and not merely its citation",
         "cost_min": 5, "promised_rmse": None},
        {"id": "C2", "change": "record per-layer status in cell_run.json, not only four counts",
         "kind": "record", "depends_on": [], "needs_files": [Path("colab/cell_assembled.py")],
         "targets_layer": None,
         "gate": "a rerun must be able to show WHICH layer changed status, not only that a count "
                 "moved",
         "mechanism": "cell_run.json stores {'RUNS': 11, 'CLOSES': 9, 'FAILED': 12, 'STATIC': 10}. "
                      "Two layers swapping status is invisible to that artefact, so the improver's "
                      "Q3 on this track cannot answer 'what changed' from the record",
         "cites": "artefact_cannot_show_change", "derivation": None, "predicted_gain": 0.0,
         "nearest_refuted": "more kcat records",
         "distinguisher": "changes what is RECORDED, not what is measured",
         "falsifier": "the counts already differ from the current table, which would mean the "
                      "artefact is not merely coarse but stale",
         "cost_min": 5, "promised_rmse": None},
        {"id": "C3", "change": "find the non-transcriptional source of protein dynamics",
         "kind": "explicit", "depends_on": [],
         "needs_files": [Path("colab/data/protein_dynamics_source.tsv.gz")],
         "targets_layer": "protein dynamics from transcription alone",
         "gate": "a candidate mechanism must predict, on held-out genes, which of the 362 proteins "
                 "that oscillate without their transcript will do so",
         "mechanism": "three FAILED layers converge on this and none of them names a replacement. "
                      "Loops 121, 122 and 123 eliminated degrons, translation control and "
                      "relocalisation in turn",
         "cites": "three_failed_layers", "derivation": None, "predicted_gain": 0.0,
         "nearest_refuted": "predict kcat/KM instead",
         "distinguisher": "different subsystem entirely",
         "falsifier": "no dataset distinguishes the 362 from the 38, in which case this is not a "
                      "plan but a wish",
         "cost_min": 240, "promised_rmse": None},
    ]
    for it in items:
        it["layer_exists"] = bool(it["targets_layer"] is None or it["targets_layer"] in L)
    return items, None, "no scalar metric on this track; T2 does not apply"


def load(track):
    t = TRACKS[track]
    runs = []
    for f in t["runs"]:
        p = OUT / f
        if p.exists():
            runs.append((f, json.load(open(p))))
    return t, runs


# ---------------------------------------------------------------------------------------------
def ask(track):
    """Q1-Q6, answered from the artefacts. Every number below is read, never remembered."""
    if track == "cell":
        a = ask_cell()
        a["track"] = track
        return a
    t, runs = load(track)
    a = {"question": t["question"], "n_runs": len(runs)}
    scores = {}
    for f, d in runs:
        if "models" in d:
            scores.update({f"{f}:{k}": v["rmse"] for k, v in d["models"].items()})
        if "baselines" in d:
            scores.update({f"{f}:{k}": v["rmse"] for k, v in d["baselines"].items()})
        for k in ("model_rmse", "constant_rmse", "mlp_tuned_rmse", "ec_rmse"):
            if k in d:
                scores[f"{f}:{k}"] = d[k]
    a["all_scores"] = scores

    prev = next((d for f, d in runs if f == "loop_ml_kcat.json"), {})
    new = next((d for f, d in runs if f == "loop_ml_audit.json"), {})
    probe = next((d for f, d in runs if f == "loop_ml_probe.json"), {})
    # The loop that AUDITED the metric, not one that produced it. The first run of this file made
    # loop 133's B4 its stopping rule without ever reading a check on B4, because no such check
    # existed. It does now, and a track that ignored it would repeat the mistake.
    a["audit"] = fix = next((d for f, d in runs if f == "loop_b4_fix.json"), {})

    a["Q1_previous"] = {"best_model": prev.get("best_model"),
                        "rmse": (prev.get("models") or {}).get(prev.get("best_model"), {}).get("rmse"),
                        "constant": prev.get("constant_rmse"),
                        "note": "loop 131, reported with no error bar"}
    vs = (new.get("a3") or {}).get("vs_xgb") or [None, None, None]
    a["Q2_new"] = {"rmse": new.get("model_rmse"), "mlp_tuned": new.get("mlp_tuned_rmse"),
                   "gain_vs_constant": (new.get("a1") or {}).get("boot_mean"),
                   "ci": [(new.get("a1") or {}).get("boot_lo"), (new.get("a1") or {}).get("boot_hi")],
                   "paired_ci": [vs[1], vs[2]],
                   "note": "loop 132, cluster-bootstrapped. 'ci' is on the gain over a constant; "
                           "'paired_ci' is on a model-vs-model difference on the SAME folds and is "
                           "the correct yardstick for a paired change"}
    a["Q3_changed_and_why"] = [
        {"what": "the gain acquired a confidence interval",
         "why": "loop 131 reported 1.3663 with no spread; a cluster bootstrap gives "
                f"{(new.get('a1') or {}).get('boot_mean')} "
                f"[{(new.get('a1') or {}).get('boot_lo')}, {(new.get('a1') or {}).get('boot_hi')}]",
         "verdict": "the gain is REAL"},
        {"what": "the MLP verdict reversed",
         "why": "loop 131 gave it 40 fixed epochs and no early stopping. With nested CV it reaches "
                f"{new.get('mlp_tuned_rmse')} against gradient boosting's {new.get('model_rmse')}, "
                f"CI on the difference {(new.get('a3') or {}).get('vs_xgb')}",
         "verdict": "'the neural network lost' was a TUNING ARTEFACT"},
        {"what": "the ceiling moved",
         "why": f"loop 132 said 0.5625 over all sequences; debiased it is "
                f"{(probe.get('b2') or {}).get('merged', [None]*4)[3]}",
         "verdict": "loop 132's ceiling was OVERSTATED and is corrected"},
    ]
    c1, c3 = fix.get("c1") or {}, fix.get("c3") or {}
    if c3:
        seq_row = {
            "what": "the sequence beyond the EC number",
            "evidence": f"loop 134 C3, no residual involved: permuting the embedding WITHIN EC "
                        f"class costs {c3.get('cost'):+.4f} against a paired interval of "
                        f"{c3.get('paired_ci')}. C1 measures the EC number at "
                        f"{100 * (c1.get('variance_share') or 0):.1f}% of the variance",
            "verdict": ("NOT FLAT -- the sequence carries information the EC number does not, and "
                        "loop 133's B4 headline is STRUCK"
                        if (c3.get("cost") or 0) > (c3.get("paired_ci") or 1)
                        else "FLAT -- confirmed without a residual")}
    else:
        seq_row = {
            "what": "the sequence beyond the EC number",
            "evidence": f"B4 refit on the EC-median residual: RMSE "
                        f"{(probe.get('b4') or {}).get('model_rmse')} against a residual sd of "
                        f"{(probe.get('b4') or {}).get('resid_sd')}",
            "verdict": "UNAUDITED -- loop 134 has not run and B4's premise is unchecked"}
    a["Q4_flat"] = [
        seq_row,
        {"what": "the substrate channel",
         "evidence": f"shuffling substrates within a protein costs "
                     f"{(new.get('a5') or {}).get('delta', [None])[0]} of a "
                     f"{(new.get('a1') or {}).get('boot_mean')} total gain",
         "verdict": "about 15% of the gain; nearly flat"},
    ]
    a["Q5_cost"] = {
        "esm_8M_embedding_min": 46, "esm_650M_projected_h": 3.9,
        "loop_runtimes_s": {f: d.get("seconds") for f, d in runs},
        "note": "650M was stopped after loop 133 B1 showed a mean-pooled readout cannot see the "
                "mutants, so a larger encoder would have inherited the same blindness",
    }
    a["Q6_same_data"] = {
        "IMPLICIT -- same inputs, used differently": [
            {"change": "pool the embedding over ACTIVE-SITE residues instead of the whole chain",
             "deficit": "mutant_irreducible_rmse", "cost": "one re-embedding, ~46 min at 8M"},
            {"change": "add a mutant flag and substitution count from the sequences already held",
             "deficit": "mutant_irreducible_rmse", "cost": "free, computed from sequences.json"},
            {"change": "normalise the target to 37 C with Q10, rather than feeding temperature",
             "deficit": "missing_conditions_rmse", "cost": "free where temperature is recoverable"},
            {"change": "add EC number as an explicit feature",
             "deficit": "b4_flat", "cost": "free"},
        ],
        "EXPLICIT -- new information": [
            {"change": "UniProt active-site and binding-site annotations",
             "deficit": "mutant_irreducible_rmse", "cost": "one fetch, 216,664 enzymes"},
            {"change": "temperature and pH parsed from UniProt kinetics free text",
             "deficit": "missing_conditions_rmse", "cost": "free, already on disk"},
        ],
    }
    return a


# ---------------------------------------------------------------------------------------------
# T2's NULL DEPENDS ON THE KIND OF CHANGE, and getting this wrong cost a real result.
#
# Turn 1 rejected P2 (mutant flag, predicted +0.0113) on T2, against a threshold of 0.0488. The
# probe then ran anyway -- because the same executor measured it -- and E2 returned a gain of
# +0.0168 with a cluster-bootstrap CI of [+0.0016, +0.0322]. THAT INTERVAL EXCLUDES ZERO. T2 had
# rejected a change that is measurably real.
#
# The error was using one number for every comparison. 0.0488 is loop 132 A3's interval on
# XGBoost-versus-MLP: two different model CLASSES, refit independently, whose disagreement includes
# all the variance of a different inductive bias. Adding a feature block to the SAME model on the
# SAME folds is a far tighter comparison -- E2's own measured interval is 0.0306 wide and E4's
# 0.0296 -- and judging a feature addition by a model-swap's null is a Type II error by
# construction. This repository has spent most of its effort guarding against Type I errors, and
# this is the mirror of that, arrived at by the same route: a threshold used without asking what
# it was the threshold OF.
#
# POST HOC. This change was prompted by seeing E2's interval, exactly as loop 127's selector was
# chosen after seeing which filter worked. The principle -- a null must match its comparison -- is
# not post hoc, but the decision to act on it here was, and saying so is cheaper than defending it.
# The drift ledger in improver_loop.py will flag this edit as a check-source change, which is what
# that ledger is for.
NOISE_POST_HOC = True
NOISE_BY_KIND = {
    "feature_addition": (0.0306, "loop 135 E2: the measured width of a cluster-bootstrap interval "
                                 "on a PAIRED feature-addition difference, same model, same folds"),
    "model_swap": (0.0488, "loop 132 A3: XGBoost versus MLP, two model classes refit independently"),
    "encoder_swap": (0.0488, "same as a model swap -- a different encoder is a different inductive "
                             "bias, not an extra column"),
}


# WHERE EACH ITEM'S OUTCOME IS MEASURED, and the state the first loop run was missing.
#
# Turn 1 executed the probe, and the same script measured P2, P4 and P6 as a side effect. Turn 2
# then re-planned all three as PREDICTIONS and re-rejected them on T2 -- arguing about a forecast
# for something already observed. An item whose outcome exists is neither accepted nor rejected;
# it is MEASURED, and its number replaces its promise. Without this state the loop cannot converge,
# because every turn re-litigates the same settled questions.
OUTCOMES = {
    "ml_kcat": {
        "P2": ("loop_plan_exec.json", ("e2", "gain"), ("e2", "boot"), "mutant flag alone, E2"),
        "P4": ("loop_plan_exec.json", ("e3", "gain"), ("e3", "boot"), "explicit EC alone, E3"),
        "B": ("loop_plan_exec.json", ("e4", "gain"), ("e4", "boot"), "the bundle as one change, E4"),
        "P6": ("loop_plan_exec.json", ("e7", "best_gain"), None, "best EC encoding, E7"),
        "P1": ("loop_active_site.json", ("h3", "gain"), ("h3", "boot"),
               "site vs mean pooling, H3 -- but REFUTED by H5: random positions of the same count "
               "score the same, so the gain is subsampling noise, not catalytic residues"),
        "M2": ("loop_active_site.json", ("h4", "cost"), None,
               "within-EC permutation under the site-pooled readout, H4 -- NEGATIVE, so protein "
               "identity is worth nothing even to a readout built to expose it"),
    },
    "cell": {
        "C1": ("cell_record_fix.json", ("g2",), None, "status unchanged by the correction, G2"),
        "C2": ("cell_record_fix.json", ("g4", "n_layers"), None, "per-layer artefact written, G4"),
    },
}


def _dig(d, path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def fetch_status():
    """WHY a file is missing, which T8 alone cannot say.

    T8 asks os.path.exists and gets a yes or a no. That was enough while the answer was always
    "nobody has fetched it", but fetch_web.py has now probed the sources and the answers differ in
    kind. A file nobody has tried to get is a task. A file whose SOURCE HAS BEEN RETIRED is a
    permanent constraint, and P3 -- Q10 normalisation to 37 C -- is blocked by exactly that:
    SABIO-RK's REST endpoints now redirect to a UI 404. Reporting those two as the same 'file
    absent' would leave the loop waiting forever on something that is never going to arrive, and
    would leave the 0.5137 missing-conditions floor looking like a gap rather than a wall."""
    p = OUT / "fetch_web.json"
    if not p.exists():
        return {}
    d = json.load(open(p))
    st = {}
    if (d.get("g3") or {}).get("api_retired"):
        st["colab/data/kcat_conditions.tsv.gz"] = {
            "kind": "SOURCE_RETIRED",
            "detail": "SABIO-RK REST probed by fetch_web G3: redirects to a UI 404. Temperature "
                      "and pH are not obtainable from the standard source, so loop 133 B5's "
                      "0.5137 floor is a WALL and not a gap"}
    g2 = d.get("g2") or {}
    if g2 and g2.get("fraction", 1) < g2.get("bar", 0):
        st["colab/data/uniprot_sites.tsv.gz"] = {
            "kind": "FETCHED_BUT_INSUFFICIENT",
            "detail": f"downloaded, but only {g2['fraction']:.1%} of our sequences match against a "
                      f"bar of {g2['bar']:.0%} declared before the fetch"}
    st["colab/data/ml/esm2_650M_mean.npy"] = {
        "kind": "COMPUTE_NOT_DOWNLOAD",
        "detail": "an embedding is produced, not fetched; no web source can supply it"}
    return st


def measured_outcomes(track):
    """Read what has actually been observed for this track. Nothing here is remembered."""
    out = {}
    for iid, (fn, vpath, cipath, what) in OUTCOMES.get(track, {}).items():
        p = OUT / fn
        if not p.exists():
            continue
        d = json.load(open(p))
        v = _dig(d, vpath)
        if v is None:
            continue
        ci = _dig(d, cipath) if cipath else None
        out[iid] = {"value": v, "ci": ci, "what": what, "from": fn}
    return out


def item_noise(it, default_noise):
    """The interval an item must clear, chosen by what KIND of comparison it is."""
    k = it.get("noise_kind")
    if k in NOISE_BY_KIND:
        return NOISE_BY_KIND[k][0], NOISE_BY_KIND[k][1]
    return default_noise, "track default"


def combine(cur, gains):
    """Gains in RMSE do not add. Removed VARIANCE adds, if the deficits are independent, so the
    honest combination of several changes is sqrt(rmse^2 - sum of removed variances). This is
    sub-additive by construction, which is the point: a bundle cannot promise past a bound."""
    removed = sum(cur ** 2 - (cur - g) ** 2 for g in gains)
    return (max(cur ** 2 - removed, 0.0)) ** 0.5


def plan(a):
    """Turn the answers into items. An item without a falsifier and a predicted size is not a plan.

    NOISE. The yardstick is NOT the confidence interval on the gain over a constant -- that measures
    how much the model beats a constant, not how well two models can be told apart. Every item here
    is a PAIRED change evaluated on the same folds, so the right spread is the interval loop 132 A3
    measured on a paired model-vs-model difference. It is much tighter, and using the wrong one
    would reject real effects."""
    if a.get("track") == "cell":
        return plan_cell(a)
    cur = a["Q2_new"]["rmse"]
    fix = a.get("audit") or {}

    def cap(component):
        """largest gain available if a change removed that entire variance component. This is the
        number a promise may not exceed, and it is arithmetic, not judgement."""
        return cur - max(cur ** 2 - component ** 2, 0.0) ** 0.5

    # Derivations. Each names the recorded number it comes from, so T7 can check the promise
    # against it. The FRACTION claimed of each maximum is the only judgement left, and it is
    # visible rather than buried inside a round figure.
    mut = BOUNDS["mutant_irreducible_rmse"][0]
    d_pool = {"from": "loop 133 B1 irreducible_rmse, CUT by loop 134 C3", "component": mut,
              "max_gain": cap(mut), "fraction_claimed": 0.08,
              "argument": "the ceiling assumes active-site pooling resolves EVERY point-mutant "
                          "pair perfectly. It will not: pooling over annotated sites still averages "
                          "several residues, and 3,597 merged proteins carry mutations outside any "
                          "annotated site. And " + PROTEIN_IDENTITY_CAVEAT + " The fraction was "
                          "0.40 before loop 134 ran; 0.08 is what survives C3"}
    d_flag = {"from": "loop 133 B1, same component as P1, CUT by loop 134 C3", "component": mut,
              "max_gain": cap(mut), "fraction_claimed": 0.03,
              "argument": "a flag says THAT a record is a variant, never WHICH, so it can only "
                          "recover the mean offset between wild types and mutants -- a small "
                          "fraction of a component whose spread is within-pair. " +
                          PROTEIN_IDENTITY_CAVEAT}
    mc = BOUNDS["missing_conditions_rmse"][0]
    d_q10 = {"from": "loop 133 B5 within-pair sd", "component": mc, "max_gain": cap(mc),
             "fraction_claimed": 0.10,
             "argument": "temperature is one of several missing conditions (pH, buffer, mutation), "
                         "and Q10 can only act on the subset where a temperature is recoverable"}
    c5 = fix.get("c5") or {}
    ec_meas, ec_base = c5.get("sequence + substrate + EC"), c5.get("sequence + substrate")
    d_ec = ({"from": "loop 134 C5, MEASURED against C5's OWN baseline", "component": None,
             "max_gain": max(ec_base - ec_meas, 0.0), "fraction_claimed": 1.0,
             "argument": "C5 ran this feature set, so the gain is the measurement itself. Compared "
                         "against C5's own baseline and not loop 132's, because mixing baselines "
                         "across runs is what produced the 3.2% figure loop 134 C1 had to correct "
                         "to 0.7%"}
            if (ec_meas and ec_base) else
            {"from": "not yet measured", "component": None, "max_gain": None,
             "fraction_claimed": None,
             "argument": "loop 134 C5 has not run, so no maximum exists and T7 must fail this"})
    d_650 = {"from": "loop 133 B3 -- the encoder is a family detector", "component": None,
             "max_gain": None, "fraction_claimed": None,
             "argument": "there is NO recorded number bounding what a larger encoder adds. Saying "
                         "so is the point: T7 fails an item whose size nobody has measured"}
    # P6 exists because loop 134 has a flaw of its own, found while reading its own output.
    d_ecenc = {"from": "loop 134 C5's EC encoding is BROKEN", "component": None,
               "max_gain": (measured_outcomes("ml_kcat").get("P6") or {}).get("value"),
               "fraction_claimed": None,
               "argument": "C5 encoded EC as eci.get(e, -1), an ARBITRARY INTEGER INDEX over "
                           "thousands of classes. A tree splitting on that index groups EC 1.1.1.1 "
                           "with whatever happened to sort beside it, so 'EC only 1.4707' and the "
                           "+0.0041 that P4 inherits are both measured through a broken feature. "
                           "The remedy is an out-of-fold target encoding plus hierarchical levels, "
                           "and until that runs NO maximum exists, so T7 must fail this item. "
                           "That is the correct outcome: the fix is cheap and the measurement is "
                           "the thing that unblocks it"}

    paired = a["Q2_new"].get("paired_ci")
    if paired and all(x is not None for x in paired):
        noise, noise_src = paired[1] - paired[0], "paired model-vs-model CI, loop 132 A3"
    else:
        ci = a["Q2_new"]["ci"]
        noise, noise_src = ((ci[1] - ci[0]) if all(x is not None for x in ci) else 0.10,
                            "fallback: gain-vs-constant CI")
    ML = Path("colab/data/ml")
    items = [
        {"id": "P1", "noise_kind": "encoder_swap", "change": "active-site pooling replaces mean pooling",
         "kind": "implicit+explicit", "depends_on": [],
         "needs_files": [ML / "sequences.json", Path("colab/data/uniprot_sites.tsv.gz")],
         "mechanism": "kcat is set by a few catalytic residues; averaging 320 dims over ~400 "
                      "residues moves the vector ~1% when 5 residues change, so 18,595 point "
                      "mutants differing a median 4.5x in kcat are indistinguishable",
         "cites": "mutant_irreducible_rmse", "derivation": d_pool,
         "predicted_gain": round(d_pool["max_gain"] * d_pool["fraction_claimed"], 4),
         "nearest_refuted": "650M with mean pooling",
         "distinguisher": "this changes the READOUT, which is the thing that refutation named as "
                          "the cause. It is the fix, not a repeat",
         "falsifier": "B4 still fails: the EC-median residual is still unlearnable",
         "cost_min": 60, "promised_rmse": cur - 0.15},
        {"id": "P2", "noise_kind": "feature_addition", "change": "mutant flag and substitution count as features",
         "kind": "implicit", "depends_on": [],
         "needs_files": [ML / "sequences.json", ML / "kcat_records.tsv"],
         "mechanism": "the model cannot resolve a variant, but it can at least "
                      "be told it is looking at one",
         "cites": "mutant_irreducible_rmse", "derivation": d_flag,
         "predicted_gain": round(d_flag["max_gain"] * d_flag["fraction_claimed"], 4),
         "nearest_refuted": "more kcat records",
         "distinguisher": "adds a COLUMN, not rows; the refutation was about n",
         "falsifier": "no change beyond the paired interval", "cost_min": 5,
         "promised_rmse": cur - 0.05},
        {"id": "P3", "noise_kind": "feature_addition", "change": "Q10-normalise the target to 37 C",
         "kind": "implicit", "depends_on": [],
         "needs_files": [Path("colab/data/kcat_conditions.tsv.gz")],
         "mechanism": "kcat(T2) = kcat(T1)*Q10^((T2-T1)/10) removes a known "
                      "variance component using physics instead of asking the "
                      "network to learn Arrhenius",
         "cites": "missing_conditions_rmse", "derivation": d_q10,
         "predicted_gain": round(d_q10["max_gain"] * d_q10["fraction_claimed"], 4),
         "nearest_refuted": "predict kcat/KM instead",
         "distinguisher": "that changed the target to a NOISIER one; this changes the target to "
                          "the same quantity at a common temperature, which cannot add variance",
         "falsifier": "target variance does not fall on the subset with a known temperature",
         "cost_min": 5, "promised_rmse": cur - 0.04},
        {"id": "P4", "noise_kind": "feature_addition", "change": "EC number as an explicit categorical feature",
         "kind": "implicit", "depends_on": [],
         "needs_files": [ML / "kcat_records.tsv"],
         "mechanism": "B4 shows the EC number carries everything the model "
                      "has; give it directly rather than via a proxy",
         "cites": "c3_sequence_beyond_ec", "derivation": d_ec,
         "predicted_gain": round(d_ec["max_gain"], 4) if d_ec["max_gain"] is not None else 0.02,
         "nearest_refuted": "bigger substrate fingerprint",
         "distinguisher": "different channel: that priced the SUBSTRATE side, this the enzyme's "
                          "declared chemistry",
         "falsifier": "no gain, which would mean ESM already encodes EC fully", "cost_min": 2,
         "promised_rmse": cur - 0.02},
        {"id": "P5", "noise_kind": "encoder_swap", "change": "ESM2-650M with the fixed readout",
         "kind": "explicit", "depends_on": ["P1"],
         "needs_files": [ML / "esm2_650M_mean.npy"],
         "depends_on_result": [{"what": "P1 has actually moved the sequence-beyond-EC test",
                                "measured": bool((fix.get("c3") or {}).get("p1_applied"))}],
         "mechanism": "a larger encoder, once the readout no longer discards "
                      "the signal it would add",
         "cites": "b3_family_detector", "derivation": d_650, "predicted_gain": 0.08,
         "nearest_refuted": "650M with mean pooling",
         "distinguisher": "ONLY the fixed readout distinguishes it, so it is refuted unless P1 is "
                          "accepted AND has actually moved B4",
         "falsifier": "gain smaller than the paired interval, as at 8M",
         "cost_min": 240, "promised_rmse": cur - 0.08},
        {"id": "P6", "noise_kind": "feature_addition", "change": "encode EC properly -- out-of-fold target encoding and hierarchy",
         "kind": "implicit", "depends_on": [],
         "needs_files": [ML / "kcat_records.tsv"],
         "mechanism": "loop 134 measured the EC channel through an arbitrary integer index, which "
                      "a tree cannot split on meaningfully. Every EC number in this project's "
                      "recent record -- C1's 0.7%, C5's 1.4707, P4's +0.0041 -- is downstream of "
                      "that encoding",
         "cites": "c3_sequence_beyond_ec", "derivation": d_ecenc,
         "predicted_gain": 0.02,
         "nearest_refuted": "bigger substrate fingerprint",
         "distinguisher": "different channel entirely: that priced the substrate side, this "
                          "repairs an instrument the enzyme side was measured with",
         "falsifier": "a correct encoding scores no better than the integer index, which would "
                      "mean the index was never the problem",
         "cost_min": 10, "promised_rmse": cur - 0.02},
        # --- PROBES. These promise nothing about the metric; they produce a number a check needs.
        {"id": "M1", "change": "measure the EC channel through three encodings on identical folds",
         "kind": "probe", "depends_on": [],
         "unblocks": {"item": "P6", "check": "T7_gain_is_derived"},
         "needs_files": [ML / "kcat_records.tsv", ML / "esm2_8M_mean.npy"],
         "mechanism": "the integer index as loop 134 used it, an out-of-fold target encoding built "
                      "with the nested construction C2 showed B4 had got wrong, and hierarchical "
                      "one-hot on EC levels 1-3. Whichever wins becomes the maximum P6 is allowed "
                      "to promise",
         "cites": "instrument_unverified", "derivation": None, "predicted_gain": 0.0,
         "nearest_refuted": "bigger substrate fingerprint",
         "distinguisher": "this measures an instrument rather than proposing a change",
         "falsifier": "all three encodings score within the paired interval of each other, which "
                      "would mean the integer index was never the problem and P6 should be dropped",
         "cost_min": 10, "promised_rmse": cur},
        {"id": "M2", "change": "measure what protein identity is worth to a NON-mean-pooled readout",
         "kind": "probe", "depends_on": [],
         "unblocks": {"item": "P1", "check": "T2_gain_exceeds_noise"},
         "needs_files": [ML / "sequences.json", Path("colab/data/uniprot_sites.tsv.gz")],
         "mechanism": "loop 134 C3 measured protein identity at 0.0046 to the MEAN-POOLED readout. "
                      "P1 claims a different readout would extract more. Nobody has measured that, "
                      "and C3's design -- permute within EC class -- reruns unchanged against any "
                      "new readout",
         "cites": "protein_identity_value_rmse", "derivation": None, "predicted_gain": 0.0,
         "nearest_refuted": "650M with mean pooling",
         "distinguisher": "measures the readout's ceiling instead of assuming it",
         "falsifier": "within-class permutation costs no more under the new readout than the "
                      "0.0046 it costs under mean pooling, which would retire P1 and P5 together",
         "cost_min": 60, "promised_rmse": cur},
    ]
    return items, noise, noise_src


# ---------------------------------------------------------------------------------------------
def theory_check(items, a, noise):
    """T1-T6. This function exists to REJECT things, and a run where nothing is rejected is
    evidence the check is too weak rather than that the plan is good."""
    refuted = {
        "predict kcat/KM instead": "measured: sd 1.553 against kcat's 1.483, a 10% WORSE target",
        "more kcat records": "loop 133 B5: the binding constraint is missing conditions, not n",
        "bigger substrate fingerprint": "loop 132 A5: substrate is ~15% of the gain",
        "650M with mean pooling": "loop 133 B1: the readout, not the encoder, is what discards it",
        "active-site pooling": "loop 136 H5: it beats mean pooling by +0.0149 [+0.0017, +0.0295], "
                               "and pooling over the same NUMBER of residues at RANDOM positions "
                               "does just as well (site minus random +0.0044 [-0.0131, +0.0206]). "
                               "The gain is residue subsampling acting as regularisation",
        "650M with ANY readout": "loop 136 H4: under site pooling, destroying protein identity "
                                 "costs -0.0008. A larger encoder feeds a readout with nothing to "
                                 "extract, so P5 is refuted by inheritance and not merely blocked",
    }
    # Data an item may NOT derive from, because the evaluation consumes it. This is loop 129's
    # lesson written down: there, a validation bundle was built from the very predictor it scored.
    EVAL_CONSUMES = {"log10_kcat", "dlkcat_predictions", "the fold assignment", "the EC median"}
    track = a.get("track", "ml_kcat")
    ACTIVE = set(TRACKS.get(track, {}).get("checks") or
                 ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"))
    cur = a["Q2_new"].get("rmse")
    floor = max(BOUNDS["experimental_floor_rmse"][0], BOUNDS["missing_conditions_rmse"][0])
    out, verdicts = [], {}
    obs = measured_outcomes(track)
    by_id = {i["id"]: i for i in items}
    for it in items:
        v = {}
        probe = it.get("kind") == "probe"
        if track == "cell":
            cd = a.get("cell_deficits") or {}
            v["T1_cites_measured_deficit"] = bool((cd.get(it["cites"]) or {}).get("holds"))
            it["deficit_evidence"] = (cd.get(it["cites"]) or {}).get("evidence")
        else:
            v["T1_cites_measured_deficit"] = it["cites"] in BOUNDS or it["cites"] in (
                "b4_flat", "b3_family_detector", "c3_sequence_beyond_ec", "instrument_unverified")
        # T2 and T7 are arithmetic on a promised gain. A probe promises no gain, so applying them
        # would reject every probe by construction -- which is precisely the trap that made this
        # category necessary.
        if probe or noise is None:
            v["T2_gain_exceeds_noise"] = True
        else:
            n_i, n_src = item_noise(it, noise)
            it["noise_used"], it["noise_source_used"] = n_i, n_src
            v["T2_gain_exceeds_noise"] = it["predicted_gain"] > n_i
        v["T3_has_falsifier"] = bool(it.get("falsifier"))
        # T4 is a real check: the item must NAME the refuted claim nearest to it and say what
        # distinguishes it. A dependency on another ITEM is a plan-level fact and settled here; a
        # dependency on a RESULT is not, and is handled below as BLOCKED rather than as a pass.
        named = it.get("nearest_refuted")
        v["T4_not_already_refuted"] = bool(
            named in refuted and it.get("distinguisher")
            and all(verdicts.get(d, "").startswith("ACCEPT") for d in it.get("depends_on", [])))
        v["T5_not_circular"] = not (set(it.get("derives_from", [])) & EVAL_CONSUMES)
        v["T6_respects_bounds"] = (True if it.get("promised_rmse") is None
                                   else it["promised_rmse"] > floor)
        # T7: the SIZE of the promise must follow from a recorded number. An item may cite a real
        # deficit and still name a figure that has nothing to do with it -- which is what every
        # item in this file's first run did.
        d = it.get("derivation")
        v["T7_gain_is_derived"] = True if probe else bool(
            d and d.get("max_gain") is not None and it["predicted_gain"] <= d["max_gain"] + 1e-9)
        missing = [f for f in it.get("needs_files", []) if not Path(f).exists()]
        v["T8_inputs_exist"] = not missing
        it["missing_inputs"] = missing
        fs = fetch_status()
        it["blockers"] = [{"file": str(f), **fs.get(str(f), {"kind": "NEVER_FETCHED",
                                                             "detail": "no fetch has been attempted"})}
                          for f in missing]
        if probe:
            u = it.get("unblocks") or {}
            tgt = by_id.get(u.get("item"))
            # the check it claims to unblock must be one the target is ACTUALLY failing right now
            v["T9_probe_unblocks_something"] = bool(
                tgt and u.get("check") and tgt.get("checks", {}).get(u["check"]) is False)
            it["unblock_status"] = (
                f"{u.get('item')} currently "
                f"{'FAILS' if v['T9_probe_unblocks_something'] else 'does not fail'} "
                f"{u.get('check')}")
        # T10, the cell track's replacement for T2/T6/T7. A status is a judgement and cannot be
        # falsified; a GATE can. An item must therefore name an existing layer and the gate that
        # would have to pass, or -- for an item that changes what is recorded rather than what is
        # modelled -- name the gate alone.
        v["T10_names_layer_and_gate"] = bool(it.get("gate") and it.get("layer_exists", True))
        # Only the checks this track declares are allowed to decide anything. Dropping a check is
        # a decision that must be visible, so the inactive ones are recorded as None rather than
        # silently omitted.
        v = {k: (val if k.split("_")[0] in ACTIVE else None) for k, val in v.items()}
        it["checks"] = v
        it["checks_not_applicable"] = [k for k, val in v.items() if val is None]
        it["observed"] = obs.get(it["id"])
        it["rejected_on"] = [k for k, val in v.items() if val is False]
        # A dependency on an unmeasured RESULT is neither pass nor fail. Recording it as either
        # would be a claim about a number nobody has.
        unmet = [r for r in it.get("depends_on_result", []) if not r.get("measured")]
        # A settled question is neither accepted nor rejected. MEASURED outranks every check,
        # because a check is a forecast about a number that now exists.
        if it.get("observed") is not None:
            o = it["observed"]
            it["verdict"] = "MEASURED"
            it["blocked_on"] = []
            num = isinstance(o["value"], (int, float)) and not isinstance(o["value"], bool)
            it["measured_note"] = (
                (f"{o['value']:+.4f} ({o['what']})"
                 if num and it.get("kind") not in ("record",)
                 else f"{o['value']} ({o['what']})")
                + (f" CI [{o['ci'][1]:+.4f}, {o['ci'][2]:+.4f}]"
                   f" -> {'EXCLUDES zero' if o['ci'][1] > 0 else 'includes zero'}"
                   if isinstance(o.get("ci"), (list, tuple)) and len(o["ci"]) == 3 else "")
                + (f"; the plan predicted {it['predicted_gain']:+.4f}, error "
                   f"{o['value'] - it['predicted_gain']:+.4f}"
                   if num and it.get("kind") not in ("probe", "record") else ""))
        elif unmet and not it["rejected_on"]:
            it["verdict"] = "BLOCKED"
            it["blocked_on"] = [r["what"] for r in unmet]
        else:
            it["verdict"] = "ACCEPT" if not it["rejected_on"] else "REJECT"
            it["blocked_on"] = [r["what"] for r in unmet]
        verdicts[it["id"]] = it["verdict"]
        out.append(it)

    # THE BUNDLE. Items rejected ONLY on T2 are not bad ideas -- they are ideas too small to
    # measure alone. Combine them by removed variance and re-test as one change.
    # A MEASURED item is not a candidate for anything. Before this guard, an item whose checks
    # still nominally failed T2 was swept into the bundle and its measured verdict overwritten --
    # the loop proposing to bundle three questions it had already answered.
    small = [i for i in out if i["verdict"] != "MEASURED"
             and i["rejected_on"] == ["T2_gain_exceeds_noise"]]
    bundle = None
    if len(small) > 1:
        # OVERLAP. combine() assumes the deficits are independent. Two items attacking the SAME
        # deficit are not, and adding their removed variances would double-count it -- loop 134's
        # C6 makes this concrete for the mutant flag and active-site pooling, which both cite
        # mutant_irreducible_rmse. Within a deficit, take the largest, never the sum.
        by_deficit = {}
        for i in small:
            by_deficit[i["cites"]] = max(by_deficit.get(i["cites"], 0.0), i["predicted_gain"])
        g = list(by_deficit.values())
        overlapped = len(g) < len(small)
        rb = combine(cur, g)
        bundle = {
            "id": "B", "members": [i["id"] for i in small],
            "change": "the sub-noise items, run and tested as ONE change",
            "why": "each is smaller than the paired interval alone, so alone each is unmeasurable. "
                   "Removed variance adds where RMSE gains do not",
            "naive_sum_gain": sum(i["predicted_gain"] for i in small),
            "combined_gain": cur - rb, "promised_rmse": rb,
            "distinct_deficits": len(g), "overlap_collapsed": overlapped,
            "cost_min": sum(i["cost_min"] for i in small),
            "falsifier": "the bundle's paired difference interval contains zero",
            "checks": {"T2_gain_exceeds_noise": (cur - rb) > noise,
                       "T6_respects_bounds": rb > floor,
                       "T4_members_all_pass_T4": all(i["checks"]["T4_not_already_refuted"]
                                                     for i in small)},
        }
        bundle["verdict"] = "ACCEPT" if all(bundle["checks"].values()) else "REJECT"
        bundle["rejected_on"] = [k for k, val in bundle["checks"].items() if not val]
        if bundle["verdict"] == "ACCEPT":
            for i in small:
                i["verdict"] = "ACCEPT-IN-BUNDLE"
    return out, refuted, bundle


# ---------------------------------------------------------------------------------------------
def whole_picture_cell(a):
    """The cell track's picture. No RMSE, no constant, no bootstrap -- and writing None into those
    fields would be worse than not having them, because a null in a numeric slot reads as a
    measurement that failed rather than as a quantity that does not exist here."""
    cd = a.get("cell_deficits") or {}
    return {
        "result": {"counts": a["Q2_new"]["counts"], "n_failed": a["Q2_new"]["n_failed"],
                   "metric_note": a["Q2_new"]["note"]},
        "what_the_model_needs_it_for": "the layer table IS the model's honesty ledger: a FAILED "
                                       "layer is a recorded negative, and the count is not a "
                                       "defect budget to be driven to zero",
        "dataset": {"layers": a["n_layers"], "parts": a.get("parts")},
        "verified_deficits": {k: v["evidence"] for k, v in cd.items() if v["holds"]},
        "bounds_that_cap_any_plan": {
            "no_scalar_metric": TRACKS["cell"]["why_not_t2"],
            "relabelling": "splitting one FAILED layer into two RUNS layers improves the count "
                           "without touching the model, which is why T10 asks for a gate"},
        "the_uncomfortable_reading":
            "three FAILED layers converge on the same wall -- protein dynamics has a source that "
            "is not transcription, and loops 121, 122 and 123 eliminated degrons, translation "
            "control and relocalisation in turn without naming a replacement. The cell track's "
            "real deficit is not a number, it is that NO DATASET on disk distinguishes the 362 "
            "proteins that oscillate without their transcript from the 38 that do the reverse.",
        "what_the_first_run_of_this_file_got_wrong":
            "it had no cell track at all, and when one was added the first thing it found was that "
            "cell_run.json's stored counts are STALE, not merely coarse",
    }


def whole_picture(a):
    if a.get("track") == "cell":
        return whole_picture_cell(a)
    """The result AND the model AND the dataset, because a plan that improves the metric and not
    the model is a plan to win a benchmark."""
    return {
        "result": {"current_rmse": a["Q2_new"]["rmse"],
                   "constant": a["Q1_previous"]["constant"],
                   "gain_ci": a["Q2_new"]["ci"]},
        "what_the_model_needs_it_for": "the enzyme-capacity constraint v <= kcat*[E]; loop 133 B6 "
                                       "measured the cell's own enzymes as EASIER than average, "
                                       "reaction-weighted 1.1895 against 1.3072",
        "dataset": {"records": 17004, "sequences": 7856, "clusters": 3006,
                    "point_mutant_pairs": 18595,
                    "singletons_contributing_zero_within_variance": 4257},
        "bounds_that_cap_any_plan": {k: v for k, v in BOUNDS.items()},
        "the_uncomfortable_reading": (
            a["Q4_flat"][0]["verdict"] + ". "
            + ("The wall is therefore NOT that the sequence is uninformative -- it is the "
               "0.947 mutant component and the 0.5137 missing-conditions floor, which are "
               "properties of the FILE and not of the representation."
               if "NOT FLAT" in a["Q4_flat"][0]["verdict"] else
               "If P1 does not change that, kcat is not predictable from sequence at this data "
               "scale, and loop 127's measured-or-flagged-constant is already the right design.")),
        "what_the_first_run_of_this_file_got_wrong": (
            "it accepted 5 of 5 items and named a broken number as its stopping rule. The check "
            "now carries T7, a BLOCKED verdict for unmeasured dependencies, and a track that "
            "reads the audit of its own metric."),
    }


def upgrade(items, bundle, picture, noise):
    """Look at the plan again WITH the whole picture, and change it. A cycle that never edits its
    own plan is a cycle that is not thinking."""
    ups = []
    cur = picture["result"].get("current_rmse")
    solo = [i for i in items if i["verdict"] == "ACCEPT"]
    if cur is None:
        # The cell track has no scalar, so the variance arithmetic below has nothing to operate on.
        # Emitting it anyway would print a number with no referent, which is the exact failure this
        # file was written to prevent.
        cheap = sorted(solo, key=lambda i: (i.get("kind") != "probe", i["cost_min"]))
        done = [i for i in items if i["verdict"] == "MEASURED"]
        return [{"change": ("order probes first, then by cost" if cheap
                            else "NOTHING IS SCHEDULABLE"),
                 "why": (f"{len(cheap)} schedulable, {len(done)} already measured. No scalar "
                         f"metric on this track, so no gain arithmetic is emitted"),
                 "new_order": [i["id"] for i in cheap]},
                {"change": "the stopping rule here is a GATE, not a metric",
                 "why": "each item names the gate that would have to pass; a count of layers can "
                        "be improved by relabelling and so cannot be the rule"}]
    units = list(solo) + ([bundle] if bundle and bundle["verdict"] == "ACCEPT" else [])
    cheap = sorted(units, key=lambda i: (i.get("kind") != "probe", i["cost_min"]))
    if not cheap:
        ups.append({"change": "NOTHING IS SCHEDULABLE",
                    "why": "every item failed a check and no probe survived either. This is a "
                           "real state and not an error: it means the next move is to fetch data "
                           "or to relax a promise, and the loop must say so rather than emit an "
                           "order it cannot execute",
                    "new_order": []})
    else:
        ups.append({"change": "order probes first, then by cost",
                    "why": f"the cheapest schedulable unit costs {cheap[0]['cost_min']} min and "
                           f"the dearest {cheap[-1]['cost_min']}. Probes come first regardless of "
                           f"cost, because a probe's output is what the checks need before any "
                           f"change can be judged -- running a change ahead of the probe that "
                           f"would bound it is spending compute to avoid learning something",
                    "new_order": [i["id"] for i in cheap]})
    ups.append({"change": "make B4 the stopping rule, not the RMSE",
                "why": "RMSE can improve while the model still knows nothing the EC number did "
                       "not. B4 -- learnability of the EC-median residual -- is the only gate "
                       "that distinguishes those, so it decides whether P5's 4 hours are spent"})
    naive = sum(i["predicted_gain"] for i in solo) + (bundle["naive_sum_gain"] if bundle else 0.0)
    honest = combine(cur, [i["predicted_gain"] for i in solo]
                     + ([bundle["combined_gain"]] if bundle else []))
    ups.append({"change": "price the whole plan by removed variance, not by adding gains",
                "why": f"the accepted units promise {naive:.2f} if gains are added, which would "
                       f"land at {cur - naive:.4f}. Combined by removed variance the plan promises "
                       f"{honest:.4f}. The floor is "
                       f"{max(BOUNDS['experimental_floor_rmse'][0], BOUNDS['missing_conditions_rmse'][0]):.4f}: "
                       f"the additive reading is the one that would have to be walked back",
                "naive_rmse": cur - naive, "honest_rmse": honest})
    ups.append({"change": "P5 is not schedulable yet",
                "why": "its only distinguisher from the refuted '650M with mean pooling' is P1's "
                       "fixed readout, so its 240 minutes are conditional on P1 having moved B4. "
                       "Listing it as a queued item would be pretending that condition is met"})
    if bundle and bundle["verdict"] == "ACCEPT":
        ups.append({"change": f"test {'+'.join(bundle['members'])} as ONE change, and report no "
                              f"per-item attribution",
                    "why": f"individually each predicts less than the paired interval {noise:.4f}; "
                           f"together {bundle['combined_gain']:.4f}. Any per-item number read off "
                           f"a joint run would be inside its own noise, which is exactly the "
                           f"loop 120 failure"})
    return ups


def main():
    t0 = time.time()
    track = sys.argv[1] if len(sys.argv) > 1 else "ml_kcat"
    log("=" * 100)
    log(f"  THE IMPROVER -- how do I get a better result than this?   [track: {track}]")
    log("=" * 100)
    a = ask(track)
    log(f"\n  question: {a['question']}   ({a['n_runs']} recorded runs)")
    for q in ("Q1_previous", "Q2_new"):
        log(f"\n  {q}: {json.dumps(a[q], default=str)[:240]}")
    log("\n  Q3 WHAT CHANGED AND WHY")
    for c in a["Q3_changed_and_why"]:
        log(f"     - {c['what']}\n         {c['why'][:150]}\n         -> {c['verdict']}")
    log("\n  Q4 WHAT REMAINED FLAT   (the question people skip)")
    for c in a["Q4_flat"]:
        log(f"     - {c['what']}\n         {c['evidence'][:150]}\n         -> {c['verdict']}")
    log(f"\n  Q5 COST: {json.dumps(a['Q5_cost'], default=str)[:200]}")
    log("\n  Q6 WITH THE SAME DATA")
    for kind, lst in a["Q6_same_data"].items():
        log(f"     {kind}")
        for c in lst:
            log(f"       - {c['change']}   [{c['cost']}]")

    items, noise, noise_src = plan(a)
    log(f"\n  PLAN: {len(items)} items; the paired interval on the current result is "
        f"{noise:.4f} ({noise_src}), and T2 requires a predicted gain larger than that")
    items, refuted, bundle = theory_check(items, a, noise)
    log("\n  THEORY CHECK")
    for it in items:
        log(f"     {it['id']} {it['verdict']:<17} {it['change']}")
        if it["rejected_on"]:
            log(f"          rejected on: {', '.join(it['rejected_on'])}")
        if it.get("blocked_on"):
            log(f"          blocked on:  {'; '.join(it['blocked_on'])}")
        if it.get("missing_inputs"):
            log(f"          MISSING INPUT: {', '.join(str(m) for m in it['missing_inputs'])}")
        d = it.get("derivation") or {}
        mg = d.get("max_gain")
        log(f"          predicted {it['predicted_gain']:+.4f} -> {it['promised_rmse']:.4f}; "
            f"cost {it['cost_min']} min")
        log(f"          derived:  {'max ' + format(mg, '.4f') if mg is not None else 'NO MAXIMUM EXISTS'}"
            f"{' x ' + format(d['fraction_claimed'], '.2f') if d.get('fraction_claimed') is not None else ''}"
            f"   ({d.get('from', 'undeclared')})")
        log(f"          falsifier: {it['falsifier'][:80]}")
    if bundle:
        log(f"\n     BUNDLE {bundle['verdict']}   {'+'.join(bundle['members'])}")
        log(f"          adding gains would claim {bundle['naive_sum_gain']:+.4f}; removed variance "
            f"gives {bundle['combined_gain']:+.4f} -> {bundle['promised_rmse']:.4f}")
        log(f"          against a paired interval of {noise:.4f}: "
            f"{'measurable as a bundle' if bundle['verdict'].startswith('ACCEPT') else 'still not measurable'}")
    log(f"\n  ALREADY REFUTED, and therefore not proposed:")
    for k, v in refuted.items():
        log(f"     - {k}: {v}")

    pic = whole_picture(a)
    log("\n  THE WHOLE PICTURE")
    log(f"     result   {pic['result']}")
    log(f"     used for {pic['what_the_model_needs_it_for'][:120]}")
    log(f"     dataset  {pic['dataset']}")
    log(f"     bounds   " + ", ".join(f"{k}={v[0]}" for k, v in pic["bounds_that_cap_any_plan"].items()))
    log(f"     READING  {pic['the_uncomfortable_reading']}")

    ups = upgrade(items, bundle, pic, noise)
    log("\n  UPGRADE THE PLAN")
    for u in ups:
        log(f"     - {u['change']}\n         {u['why'][:230]}")

    order = ups[0]["new_order"]
    log(f"\n  EXECUTE IN THIS ORDER: {' -> '.join(order)}")
    log(f"  STOPPING RULE: B4. If the EC-median residual stays unlearnable after these, the "
        f"expensive ones are not run and the negative result is the answer.")
    n_rej = sum(1 for i in items if i["verdict"] == "REJECT")
    n_blk = sum(1 for i in items if i["verdict"] == "BLOCKED")
    log(f"  THE CHECK REJECTED {n_rej} of {len(items)} items and BLOCKED {n_blk}. A run that "
        f"rejects nothing is evidence the check is too weak, not that the plan is good.")
    if n_rej == 0:
        log(f"  *** THIS RUN REJECTED NOTHING. Treat the check as unproven, not the plan as sound.")
    log("\n  HISTORY OF THIS LOOP   (kept so a wrong prediction stays visible)")
    for h in HISTORY:
        log(f"     turn {h['turn']}  accepted {h['accepted']}, rejected {h['rejected']}, "
            f"blocked {h['blocked']}   stopping rule: {h['stopping_rule']}")
        log(f"        wrong: {h['what_it_got_wrong']}")
        log(f"        caught by: {h['caught_by']}")
        log(f"        fix: {h['fix']}")
    json.dump({"track": track, "answers": a, "plan": items, "bundle": bundle, "refuted": refuted,
               "whole_picture": pic, "upgrades": ups, "execute_order": order, "history": HISTORY,
               "noise": noise, "noise_source": noise_src, "n_rejected": n_rej,
               "seconds": time.time() - t0},
              open(OUT / f"improver_{track}.json", "w"), indent=1, default=str)
    log(f"\n  -> {OUT / f'improver_{track}.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
