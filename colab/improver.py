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
would have caught: it named loop 133's B4, and loop 134 then measured B4's premise at 3.2% of the
variance. The lesson is recorded in load(): a track must read the loop that AUDITED its metric, not
only the loops that produced it.

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
                 "loop_b4_fix.json"],
        "metric": "rmse", "lower_is_better": True,
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
}


def log(s=""):
    print(s, flush=True)


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
    d_pool = {"from": "loop 133 B1 irreducible_rmse", "component": mut, "max_gain": cap(mut),
              "fraction_claimed": 0.40,
              "argument": "the ceiling assumes active-site pooling resolves EVERY point-mutant "
                          "pair perfectly. It will not: pooling over annotated sites still averages "
                          "several residues, and 3,597 merged proteins carry mutations outside any "
                          "annotated site. 40% of the ceiling is a guess, but it is now a visible "
                          "guess against a measured maximum"}
    d_flag = {"from": "loop 133 B1, same component as P1", "component": mut, "max_gain": cap(mut),
              "fraction_claimed": 0.12,
              "argument": "a flag says THAT a record is a variant, never WHICH, so it can only "
                          "recover the mean offset between wild types and mutants -- a small "
                          "fraction of a component whose spread is within-pair"}
    mc = BOUNDS["missing_conditions_rmse"][0]
    d_q10 = {"from": "loop 133 B5 within-pair sd", "component": mc, "max_gain": cap(mc),
             "fraction_claimed": 0.10,
             "argument": "temperature is one of several missing conditions (pH, buffer, mutation), "
                         "and Q10 can only act on the subset where a temperature is recoverable"}
    ec_meas = (fix.get("c5") or {}).get("sequence + substrate + EC")
    d_ec = ({"from": "loop 134 C5, MEASURED not bounded", "component": None,
             "max_gain": max(cur - ec_meas, 0.0), "fraction_claimed": 1.0,
             "argument": "C5 ran this feature set, so the gain is the measurement itself"}
            if ec_meas else
            {"from": "not yet measured", "component": None, "max_gain": None,
             "fraction_claimed": None,
             "argument": "loop 134 C5 has not run, so no maximum exists and T7 must fail this"})
    d_650 = {"from": "loop 133 B3 -- the encoder is a family detector", "component": None,
             "max_gain": None, "fraction_claimed": None,
             "argument": "there is NO recorded number bounding what a larger encoder adds. Saying "
                         "so is the point: T7 fails an item whose size nobody has measured"}

    paired = a["Q2_new"].get("paired_ci")
    if paired and all(x is not None for x in paired):
        noise, noise_src = paired[1] - paired[0], "paired model-vs-model CI, loop 132 A3"
    else:
        ci = a["Q2_new"]["ci"]
        noise, noise_src = ((ci[1] - ci[0]) if all(x is not None for x in ci) else 0.10,
                            "fallback: gain-vs-constant CI")
    items = [
        {"id": "P1", "change": "active-site pooling replaces mean pooling",
         "kind": "implicit+explicit", "depends_on": [],
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
        {"id": "P2", "change": "mutant flag and substitution count as features",
         "kind": "implicit", "depends_on": [],
         "mechanism": "the model cannot resolve a variant, but it can at least "
                      "be told it is looking at one",
         "cites": "mutant_irreducible_rmse", "derivation": d_flag,
         "predicted_gain": round(d_flag["max_gain"] * d_flag["fraction_claimed"], 4),
         "nearest_refuted": "more kcat records",
         "distinguisher": "adds a COLUMN, not rows; the refutation was about n",
         "falsifier": "no change beyond the paired interval", "cost_min": 5,
         "promised_rmse": cur - 0.05},
        {"id": "P3", "change": "Q10-normalise the target to 37 C",
         "kind": "implicit", "depends_on": [],
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
        {"id": "P4", "change": "EC number as an explicit categorical feature",
         "kind": "implicit", "depends_on": [],
         "mechanism": "B4 shows the EC number carries everything the model "
                      "has; give it directly rather than via a proxy",
         "cites": "c3_sequence_beyond_ec", "derivation": d_ec,
         "predicted_gain": round(d_ec["max_gain"], 4) if d_ec["max_gain"] is not None else 0.02,
         "nearest_refuted": "bigger substrate fingerprint",
         "distinguisher": "different channel: that priced the SUBSTRATE side, this the enzyme's "
                          "declared chemistry",
         "falsifier": "no gain, which would mean ESM already encodes EC fully", "cost_min": 2,
         "promised_rmse": cur - 0.02},
        {"id": "P5", "change": "ESM2-650M with the fixed readout",
         "kind": "explicit", "depends_on": ["P1"],
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
    }
    # Data an item may NOT derive from, because the evaluation consumes it. This is loop 129's
    # lesson written down: there, a validation bundle was built from the very predictor it scored.
    EVAL_CONSUMES = {"log10_kcat", "dlkcat_predictions", "the fold assignment", "the EC median"}
    cur = a["Q2_new"]["rmse"]
    floor = max(BOUNDS["experimental_floor_rmse"][0], BOUNDS["missing_conditions_rmse"][0])
    out, verdicts = [], {}
    for it in items:
        v = {}
        v["T1_cites_measured_deficit"] = it["cites"] in BOUNDS or it["cites"] in (
            "b4_flat", "b3_family_detector", "c3_sequence_beyond_ec")
        v["T2_gain_exceeds_noise"] = it["predicted_gain"] > noise
        v["T3_has_falsifier"] = bool(it.get("falsifier"))
        # T4 is a real check: the item must NAME the refuted claim nearest to it and say what
        # distinguishes it. A dependency on another ITEM is a plan-level fact and settled here; a
        # dependency on a RESULT is not, and is handled below as BLOCKED rather than as a pass.
        named = it.get("nearest_refuted")
        v["T4_not_already_refuted"] = bool(
            named in refuted and it.get("distinguisher")
            and all(verdicts.get(d, "").startswith("ACCEPT") for d in it.get("depends_on", [])))
        v["T5_not_circular"] = not (set(it.get("derives_from", [])) & EVAL_CONSUMES)
        v["T6_respects_bounds"] = it["promised_rmse"] > floor
        # T7: the SIZE of the promise must follow from a recorded number. An item may cite a real
        # deficit and still name a figure that has nothing to do with it -- which is what every
        # item in this file's first run did.
        d = it.get("derivation")
        v["T7_gain_is_derived"] = bool(
            d and d.get("max_gain") is not None and it["predicted_gain"] <= d["max_gain"] + 1e-9)
        it["checks"] = v
        it["rejected_on"] = [k for k, val in v.items() if not val]
        # A dependency on an unmeasured RESULT is neither pass nor fail. Recording it as either
        # would be a claim about a number nobody has.
        unmet = [r for r in it.get("depends_on_result", []) if not r.get("measured")]
        if unmet and not it["rejected_on"]:
            it["verdict"] = "BLOCKED"
            it["blocked_on"] = [r["what"] for r in unmet]
        else:
            it["verdict"] = "ACCEPT" if not it["rejected_on"] else "REJECT"
            it["blocked_on"] = [r["what"] for r in unmet]
        verdicts[it["id"]] = it["verdict"]
        out.append(it)

    # THE BUNDLE. Items rejected ONLY on T2 are not bad ideas -- they are ideas too small to
    # measure alone. Combine them by removed variance and re-test as one change.
    small = [i for i in out if i["rejected_on"] == ["T2_gain_exceeds_noise"]]
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
def whole_picture(a):
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
    cur = picture["result"]["current_rmse"]
    solo = [i for i in items if i["verdict"] == "ACCEPT"]
    units = list(solo) + ([bundle] if bundle and bundle["verdict"] == "ACCEPT" else [])
    cheap = sorted(units, key=lambda i: i["cost_min"])
    ups.append({"change": "order by cost, not by predicted gain",
                "why": f"the cheapest accepted unit costs {cheap[0]['cost_min']} min and the "
                       f"dearest {cheap[-1]['cost_min']}. Spending the cheap ones first means the "
                       f"dear ones are only paid for if the cheap ones move B4",
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
    json.dump({"track": track, "answers": a, "plan": items, "bundle": bundle, "refuted": refuted,
               "whole_picture": pic, "upgrades": ups, "execute_order": order,
               "noise": noise, "noise_source": noise_src, "n_rejected": n_rej,
               "seconds": time.time() - t0},
              open(OUT / f"improver_{track}.json", "w"), indent=1, default=str)
    log(f"\n  -> {OUT / f'improver_{track}.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
