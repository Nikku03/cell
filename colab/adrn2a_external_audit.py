"""Audit of the external ADRN-2A benchmark (adrn2.py / adrn2_tasks.py / adrn2_baselines.py).

This targets a *different* codebase from colab/adrn2.py -- the ADRN-2A package
supplied for review, which lives in its own Python package alongside a
``transformer`` module providing ``sinusoidal_positions``.  Point it at that
package with ``--package`` (a directory) and it runs two decisive checks.

Both checks were run before this file was committed; the numbers reported in
UPDATES.md come from executing exactly this script.

CHECK 1 -- what ``return_routing_recall`` can possibly measure
-------------------------------------------------------------
The metric is scored only on stream positions where ``in_return_recall_window``
is true, and ``ADRN2Example.__post_init__`` *forbids* feedback in that window.
``ADRN2.predict`` never mutates ``_active_context``; only ``update`` does, and
``update`` is only called when feedback is available.  At prediction time the
inferred router scores contexts with

    transition_log_prior(c) + feature_likelihood_weight * feature_log_likelihood(c)

and the experiment sets ``feature_likelihood_weight=0.0``, so the second term is
identically zero and the argmax is whichever context is already active whenever
``context_stickiness > 1/n`` (0.55 here).  The reported context is therefore a
constant through the whole scored window, frozen at the last feedback event of
the *previous* phase -- a different context by construction.

Predicted: inferred == 0.0 exactly; no-separation == 1.0 with a single context
(the metric cannot distinguish "always right" from "only one answer exists").

CHECK 2 -- is the GRU workspace load-bearing?
---------------------------------------------
``ADRN2Backbone.represent`` returns

    cat(0.25 * workspace[48], final_digital[136], first_values[16])

where the trailing 152 coordinates are a *fixed, unlearned* function of the
input: the signed pair basis of the final step plus the raw first step.  All
four ADRN-2A rules are linearly separable in those 152 coordinates alone --
XOR(a,b) is exactly the signed pair product -a*b; AND is a pair product;
majority(a,b,c) in +/-1 coding is sign(a+b+c); and c4's delayed operand
x3(t-2) sits in ``first_values``.  So the recurrent workspace should be
removable at no cost.

The digital-zeroed arm is the positive control: if masking works at all, that
arm must collapse.  Without it a null result cannot be distinguished from an
inert ablation.

Usage
-----
    python adrn2a_external_audit.py --package /path/to/adrn2_package [--seeds 3]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

PHASE_LEN = 128
FEEDBACK_PERIOD = 4
RETURN_NO_FEEDBACK = 32
PRETRAIN_EXAMPLES = 1_024
PRETRAIN_EPOCHS = 8
WORKSPACE_BLOCK = 48  # leading coordinates of represent() that come from the GRU


def _load(package: Path):
    """Import the external package by adding its parent to sys.path."""

    package = package.resolve()
    if not (package / "adrn2.py").is_file():
        raise SystemExit(f"{package} does not contain adrn2.py")
    if not (package / "__init__.py").is_file():
        raise SystemExit(
            f"{package} is not a package (no __init__.py); the module uses "
            "relative imports and cannot be loaded as loose files"
        )
    sys.path.insert(0, str(package.parent))
    name = package.name
    mod = __import__(f"{name}.adrn2", fromlist=["*"])
    tasks = __import__(f"{name}.adrn2_tasks", fromlist=["*"])
    experiment = __import__(f"{name}.adrn2_experiment", fromlist=["*"])
    return mod, tasks, experiment


def check_routing(mod, tasks, experiment) -> None:
    import torch

    stream = tasks.make_adrn2_stream(
        examples_per_phase=PHASE_LEN,
        feedback_period=FEEDBACK_PERIOD,
        return_no_feedback=RETURN_NO_FEEDBACK,
        seed=40_000,
    )

    print("== per-context marginal bit rate")
    print("   (the generator draws distractors Bernoulli(0.5) and cycles the relevant")
    print("    truth table, so p(x|c) is uniform on {0,1}^48 for every c and p(c|x)=p(c))")
    by_context: dict[str, list] = {}
    for example in stream:
        by_context.setdefault(example.context, []).append(example.sequence)
    for context in sorted(by_context):
        arr = np.stack(by_context[context])
        print(
            f"   {context}: n={len(arr):4d}  mean={arr.mean():.4f}  "
            f"per-coordinate range=[{arr.mean(0).min():.3f}, {arr.mean(0).max():.3f}]"
        )

    config = mod.ADRN2Config(
        encoder_size=32, workspace_size=48, context_stickiness=0.55,
        feature_likelihood_weight=0.0, poor_feedback_probability=0.60,
        poor_feedback_patience=2, minimum_context_feedback=2, max_contexts=6, seed=0,
    )
    rng = np.random.default_rng(20_000)
    pretrain = rng.integers(
        0, 2, size=(PRETRAIN_EXAMPLES, 3, 16), dtype=np.int8
    ).astype(np.float32)
    template = mod.ADRN2(config, context_mode=mod.ContextMode.NO_SEPARATION)
    template.pretrain_representation(
        pretrain, epochs=PRETRAIN_EPOCHS, batch_size=64
    )

    print("\n== measured routing")
    for name, context_mode in (
        ("no_separation", mod.ContextMode.NO_SEPARATION),
        ("oracle", mod.ContextMode.ORACLE),
        ("inferred", mod.ContextMode.INFERRED),
    ):
        model = mod.ADRN2(replace(config, context_mode=context_mode))
        model.backbone.load_state_dict(template.backbone.state_dict())
        model.reset_online()
        assignments, correct, in_window = [], [], []
        for example in stream:
            supplied = (
                example.context if model.mode is mod.ContextMode.ORACLE else None
            )
            result = model.predict(example.sequence, context=supplied)
            assignments.append(result.context_id)
            correct.append(result.prediction == example.target)
            if example.in_return_recall_window:
                in_window.append((example.context, repr(result.context_id)))
            if example.feedback_available:
                model.update(example.sequence, example.target, context=supplied)
        purity = experiment._context_purity(stream, assignments)
        recall = experiment._return_routing_recall(stream, assignments)
        print(
            f"   {name:14s} accuracy={np.mean(correct):.4f}  purity={purity:.4f}  "
            f"return_routing_recall={recall:.4f}  contexts={len(model.context_ids)}"
        )
        for context in sorted({c for c, _ in in_window}):
            counts = Counter(a for c, a in in_window if c == context)
            print(f"       return window {context}: assignments {dict(counts)}")
    del torch


def check_workspace(mod, tasks, seeds: int) -> None:
    import torch

    torch.set_num_threads(1)
    original = mod.ADRN2Backbone.represent

    def masked(keep_workspace: bool, keep_digital: bool):
        def represent(self, values, padding_mask):
            out = original(self, values, padding_mask).clone()
            if not keep_workspace:
                out[:, :WORKSPACE_BLOCK] = 0.0
            if not keep_digital:
                out[:, WORKSPACE_BLOCK:] = 0.0
            return out

        return represent

    config = mod.ADRN2Config(
        encoder_size=32, workspace_size=48, context_stickiness=0.55,
        feature_likelihood_weight=0.0, poor_feedback_probability=0.60,
        poor_feedback_patience=2, minimum_context_feedback=2, max_contexts=6, seed=0,
    )
    arms = (
        ("full  (workspace+digital)", (True, True)),
        ("digital only (GRU off)  ", (False, True)),
        ("workspace only (GRU)    ", (True, False)),
    )
    rows: list[tuple[str, float, float]] = []
    for seed in range(seeds):
        stream = tasks.make_adrn2_stream(
            examples_per_phase=PHASE_LEN,
            feedback_period=FEEDBACK_PERIOD,
            return_no_feedback=RETURN_NO_FEEDBACK,
            seed=40_000 + seed,
        )
        targets = np.asarray([example.target for example in stream])
        rng = np.random.default_rng(20_000 + seed)
        pretrain = rng.integers(
            0, 2, size=(PRETRAIN_EXAMPLES, 3, 16), dtype=np.int8
        ).astype(np.float32)
        template = mod.ADRN2(
            replace(config, seed=seed), context_mode=mod.ContextMode.NO_SEPARATION
        )
        template.pretrain_representation(
            pretrain, epochs=PRETRAIN_EPOCHS, batch_size=64
        )
        state = template.backbone.state_dict()

        for arm, (keep_workspace, keep_digital) in arms:
            mod.ADRN2Backbone.represent = masked(keep_workspace, keep_digital)
            try:
                # Oracle context removes routing as a confound: every arm is handed
                # the same perfect adapter key, so only the representation differs.
                model = mod.ADRN2(
                    replace(config, seed=seed, context_mode=mod.ContextMode.ORACLE)
                )
                model.backbone.load_state_dict(state)
                model.reset_online()
                predictions = []
                for example in stream:
                    predictions.append(
                        model.predict(
                            example.sequence, context=example.context
                        ).prediction
                    )
                    if example.feedback_available:
                        model.update(
                            example.sequence, example.target, context=example.context
                        )
                metrics = tasks.compute_adrn2_metrics(
                    stream, predictions, immediate_window=16, rolling_window=16,
                    adaptation_threshold=0.8, final_window=16,
                    threshold_metric="balanced_accuracy",
                )
                final = float(
                    np.mean([phase.final_balanced_accuracy for phase in metrics.phases])
                )
                rows.append(
                    (arm, float(np.mean(np.asarray(predictions) == targets)), final)
                )
            finally:
                mod.ADRN2Backbone.represent = original

    print(f"\n== ADRN-2A workspace ablation, ORACLE context, {seeds} seeds")
    print(f"   {'arm':28s} {'accuracy':>18s} {'mean phase-final bal.acc':>26s}")
    for arm, _ in arms:
        subset = [row for row in rows if row[0] == arm]
        accuracy = np.array([row[1] for row in subset])
        final = np.array([row[2] for row in subset])
        spread = (lambda v: v.std(ddof=1) if len(v) > 1 else 0.0)
        print(
            f"   {arm:28s} {accuracy.mean():.4f} +/- {spread(accuracy):.4f}   "
            f"{final.mean():.4f} +/- {spread(final):.4f}"
        )
    print(
        "\n   Read the third row first: it is the positive control.  If masking the\n"
        "   digital block does NOT collapse the arm to chance, the mask is inert and\n"
        "   the second row's null means nothing."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--skip-routing", action="store_true")
    parser.add_argument("--skip-workspace", action="store_true")
    args = parser.parse_args(argv)
    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1")
    mod, tasks, experiment = _load(args.package)
    if not args.skip_routing:
        check_routing(mod, tasks, experiment)
    if not args.skip_workspace:
        check_workspace(mod, tasks, args.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
