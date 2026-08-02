"""Decompose ADRN-2B's Gate B into "exact digital state" and "recurrence".

Gate B is ``workspace_oracle - polynomial_oracle`` on the four temporal operation
families.  The two arms do not differ only in recurrence:

    RecurrentWorkspaceRepresentation.learned_features
        -> cat(digital_temporal_state_tensor(values)[288], neural[32] * scale)
    TransformerWorkspaceRepresentation.learned_features
        -> _fixed_expand(learned[32] * scale, 320)          # no digital block
    polynomial control
        -> expanded_polynomial_tensor(values, 488)          # linear rank 168

So the GRU arm receives 288 coordinates of *exact, unlearned* digital temporal
state that neither other arm gets, and only 32 coordinates from the GRU.  The
288 are not neutral with respect to what Gate B measures:

    running_parity(f)        -> one coordinate of ``parity`` (+/-1 coded answer)
    toggle_reset(t, r)       -> one of the 240 ``toggle_reset`` coordinates,
                                computed with the identical reset-wins-ties
                                order as ``evaluate_rule``
    temporal_order(a, b)     -> sign of first_times[b] - first_times[a]
    last_event(a, b)         -> sign of last_times[a] - last_times[b]

Every Gate B family is therefore a single-coordinate or two-coordinate-linear
readout of the digital block, with no recurrence required.  A Gate B pass is
consistent with "the GRU is necessary" AND with "the answer was tabulated in
advance"; as shipped, the experiment cannot tell those apart.

This script separates them by re-running the whole experiment three times with
``learned_features`` patched, holding everything else -- seeds, tuning, episode
construction, bootstrap -- fixed:

    shipped         cat(digital, neural)        as released
    digital_only    cat(digital, neural * 0)    recurrence removed
    recurrent_only  expand(neural, 320)         digital removed; this is exactly
                                                the Transformer arm's shape

which fills in the 2x2 (P = polynomial_oracle, identical across runs):

                    no digital        + digital
    no recurrence   P                 digital_only
    + GRU           recurrent_only    shipped   <- Gate B numerator

Read the decomposition as:
    digital effect      = digital_only  - P
    recurrence | digital= shipped       - digital_only
    recurrence alone    = recurrent_only- P

If ``shipped - digital_only`` is ~0 while ``digital_only - P`` carries the whole
of Gate B, then Gate B is a feature-engineering result and the recurrent
workspace is not what it measures.

A free machinery check is also reported: the paired delta of ``polynomial_oracle``
against itself must be exactly 0.0 with a degenerate interval.  If it is not, the
pairing is not pairing and no delta below can be believed.

Usage
-----
    python adrn2b_gate_b_decomposition.py --package /path/to/adrn2b_package \\
        --scale full --output out/gate_b_decomposition.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

TEMPORAL_OPERATIONS = {"temporal_order", "last_event", "running_parity", "toggle_reset"}
NON_TEMPORAL_OPERATIONS = {"and", "or", "xor", "xnor", "majority"}


def _load(package: Path):
    package = package.resolve()
    for required in ("adrn2b_experiment.py", "adrn2b_models.py", "__init__.py"):
        if not (package / required).is_file():
            raise SystemExit(f"{package} is missing {required}")
    sys.path.insert(0, str(package.parent))
    name = package.name
    experiment = __import__(f"{name}.adrn2b_experiment", fromlist=["*"])
    models = __import__(f"{name}.adrn2b_models", fromlist=["*"])
    return experiment, models


def _variants(models):
    """Three replacements for RecurrentWorkspaceRepresentation.learned_features.

    Each keeps the returned width at ``learned_size`` (320) so ``represent`` is
    unchanged in shape and the adapter feature width stays at 488.  Each keeps
    the GRU in the autograd graph -- ``neural * 0.0`` rather than a detached
    constant -- so offline training sees defined (zero) gradients instead of
    ``None`` and the optimizer path is identical across arms.
    """

    import torch

    digital_state = models.digital_temporal_state_tensor
    expand = models._fixed_expand

    def neural(self, values):
        encoded = torch.nn.functional.gelu(self.input_projection(values))
        _, hidden = self.workspace(encoded)
        return self.workspace_projection(self.output_norm(hidden[-1]))

    def shipped(self, values):
        return torch.cat(
            (digital_state(values), neural(self, values) * self.config.learned_feature_scale),
            dim=1,
        )

    def digital_only(self, values):
        return torch.cat(
            (digital_state(values), neural(self, values) * 0.0), dim=1
        )

    def recurrent_only(self, values):
        scaled = neural(self, values) * self.config.learned_feature_scale
        return expand(scaled, self.config.learned_size)

    return {
        "shipped": shipped,
        "digital_only": digital_only,
        "recurrent_only": recurrent_only,
    }


def _delta(experiment, left, right, operations, samples, seed):
    return experiment._paired_delta_interval(
        left, right, operations=operations, bootstrap_samples=samples, seed=seed
    )


def _summarize(experiment, result, settings):
    """Pull the Gate B numerator, its per-operation split, and the controls."""

    records = {
        name: [
            experiment.ProbeRecord(**row) if isinstance(row, dict) else row
            for row in rows
        ]
        for name, rows in result["_records"].items()
    }
    samples = settings.bootstrap_samples
    out = {
        "gate_b": _delta(experiment, records["workspace_oracle"],
                         records["polynomial_oracle"], TEMPORAL_OPERATIONS,
                         samples, settings.seed + 81),
        "non_temporal_control": _delta(experiment, records["workspace_oracle"],
                                       records["polynomial_oracle"],
                                       NON_TEMPORAL_OPERATIONS, samples,
                                       settings.seed + 82),
        "per_operation": {
            operation: _delta(experiment, records["workspace_oracle"],
                              records["polynomial_oracle"], {operation},
                              samples, settings.seed + 83 + index)
            for index, operation in enumerate(sorted(TEMPORAL_OPERATIONS))
        },
        "machinery_null_self_delta": _delta(
            experiment, records["polynomial_oracle"], records["polynomial_oracle"],
            TEMPORAL_OPERATIONS, samples, settings.seed + 99,
        ),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--scale", choices=("smoke", "full"), default="full")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arms", default="shipped,digital_only,recurrent_only",
        help="comma-separated subset of the three learned_features variants",
    )
    args = parser.parse_args(argv)

    experiment, models = _load(args.package)
    import torch

    variants = _variants(models)
    requested = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = set(requested) - set(variants)
    if unknown:
        raise SystemExit(f"unknown arms: {sorted(unknown)}")

    settings = (
        experiment.ADRN2BExperimentConfig.smoke()
        if args.scale == "smoke"
        else experiment.ADRN2BExperimentConfig.full()
    )
    original = models.RecurrentWorkspaceRepresentation.learned_features
    payload: dict[str, object] = {
        "scale": args.scale,
        "config_seed": settings.seed,
        "bootstrap_samples": settings.bootstrap_samples,
        "temporal_operations": sorted(TEMPORAL_OPERATIONS),
        "arms": {},
    }

    # The harness returns serialized records; re-run it per arm and keep the raw
    # ProbeRecord lists by intercepting them, so the paired bootstrap below sees
    # the same objects the shipped gate sees.
    captured: dict[str, dict] = {}
    original_gates = experiment._decision_gates

    def capturing_gates(*, summaries, records, config, cue_calibration):
        captured["records"] = records
        captured["summaries"] = summaries
        return original_gates(summaries=summaries, records=records, config=config,
                              cue_calibration=cue_calibration)

    for name in requested:
        print(f"=== arm {name} ===", flush=True)
        models.RecurrentWorkspaceRepresentation.learned_features = variants[name]
        experiment._decision_gates = capturing_gates
        captured.clear()
        started = perf_counter()
        try:
            torch.manual_seed(settings.seed)
            result = experiment.run_adrn2b_experiment(settings, verbose=True)
        finally:
            models.RecurrentWorkspaceRepresentation.learned_features = original
            experiment._decision_gates = original_gates
        elapsed = perf_counter() - started
        result["_records"] = captured["records"]
        arm = _summarize(experiment, result, settings)
        arm["decision_gates"] = result["decision_gates"]
        arm["wall_seconds"] = elapsed
        arm["workspace_oracle_summary"] = {
            key: result["model_summaries"]["workspace_oracle"].get(key)
            for key in ("feedback_free_return_balanced_accuracy",
                        "first_visit_balanced_accuracy_by_labels")
        }
        payload["arms"][name] = arm
        print(f"    gate_b mean_delta={arm['gate_b'].get('mean_delta')} "
              f"n={arm['gate_b'].get('paired_n')} in {elapsed:.1f}s", flush=True)

    arms = payload["arms"]
    if {"shipped", "digital_only"} <= arms.keys():
        shipped = arms["shipped"]["gate_b"]["mean_delta"]
        digital = arms["digital_only"]["gate_b"]["mean_delta"]
        if shipped is not None and digital is not None:
            payload["decomposition"] = {
                "gate_b_as_shipped": shipped,
                "digital_effect_alone": digital,
                "recurrence_given_digital": shipped - digital,
                "digital_share_of_gate_b": (
                    None if abs(shipped) < 1e-12 else digital / shipped
                ),
            }
            if "recurrent_only" in arms:
                payload["decomposition"]["recurrence_alone"] = (
                    arms["recurrent_only"]["gate_b"]["mean_delta"]
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(f"\nwrote {args.output}")
    if "decomposition" in payload:
        print(json.dumps(payload["decomposition"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
