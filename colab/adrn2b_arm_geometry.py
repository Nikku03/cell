"""Feature geometry of the three ADRN-2B arms that Gates B and E compare.

Gates B and E treat ``polynomial``, ``workspace``, and ``transformer`` as arms that
differ in one thing -- what kind of temporal computation produced their features.
All three emit a 488-vector into the same local delta-rule adapter, at the same
tuned learning rate. This script measures whether they are otherwise comparable.

They are not, on two axes beyond information content:

USABLE SUBSPACE.  ``_fixed_expand`` maps the 168 real polynomial coordinates into
488 by appending a fixed cosine projection scaled by ``0.1 / sqrt(168)``.  The map
is injective, so the matrix is technically full rank -- but the appended
directions carry singular values orders of magnitude below the first 168.  The
docstring is explicit that this is deliberate: it leaves "room for an added
unit-scale workspace state to be identifiable by the common local delta rule."
The consequence is that the control's effective subspace stays at 168 while the
workspace arm, whose 320 added coordinates arrive at unit scale, uses far more.
Report effective rank at a 1% singular-value threshold, not ``matrix_rank``,
whose default tolerance is ~1e-11 here and calls all three arms rank 488.

STEP SIZE.  The adapter update is ``outer(feature, error)``, so the logit change
from one labeled example scales with ``||feature||^2 * learning_rate``.  Arms with
larger feature norm therefore take larger effective steps at the same nominal
learning rate -- and the harness tunes ONE ``adapter_learning_rate`` on
``validation_workspace_oracle`` and applies it to every arm in the matrix,
including the polynomial control.  The control runs at a rate selected for a
representation with a different norm.

Neither observation says which arm should win; both say the comparison is not
clean, and the direction of the bias has to be argued rather than assumed.

Usage
-----
    python adrn2b_arm_geometry.py --package /path/to/adrn2b_package \\
        --output out/arm_geometry.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DRAWS = 1_200
THRESHOLD = 0.01


def _load(package: Path):
    package = package.resolve()
    sys.path.insert(0, str(package.parent))
    return __import__(f"{package.name}.adrn2b_models", fromlist=["*"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--sequence-length", type=int, default=16)
    args = parser.parse_args(argv)

    models = _load(args.package)
    import torch

    config = models.ADRN2BRepresentationConfig(
        maximum_sequence_length=args.sequence_length, seed=202_602
    )
    rng = np.random.default_rng(11)
    values = torch.as_tensor(
        rng.integers(0, 2, size=(args.draws, args.sequence_length, 16)).astype(np.float32)
    )
    workspace = models.RecurrentWorkspaceRepresentation(config)
    transformer = models.TransformerWorkspaceRepresentation(config)
    workspace.eval()
    transformer.eval()
    with torch.no_grad():
        arms = {
            "polynomial_control": models.expanded_polynomial_tensor(
                values, config.representation_size
            ),
            "workspace_gru": workspace.represent(values),
            "transformer": transformer.represent(values),
        }
        digital = models.digital_temporal_state_tensor(values)

    reference = float(torch.linalg.vector_norm(arms["polynomial_control"], dim=1).mean())
    rows: dict[str, dict[str, float]] = {}
    print(f"{'arm':22s} {'dim':>4s} {'mean L2':>9s} {'norm vs':>9s} {'eff rank':>9s} "
          f"{'s_1':>9s} {'s_169':>9s} {'s_457':>9s}")
    print(f"{'':22s} {'':>4s} {'':>9s} {'control':>9s} {'(1% s_1)':>9s}")
    for name, tensor in arms.items():
        matrix = tensor.numpy().astype(np.float64)
        singular = np.linalg.svd(matrix, compute_uv=False)
        norm = float(np.linalg.norm(matrix, axis=1).mean())
        effective = int((singular > THRESHOLD * singular[0]).sum())
        rows[name] = {
            "dimension": int(matrix.shape[1]),
            "mean_l2_norm": norm,
            "norm_ratio_vs_control": norm / reference,
            "effective_rank_1pct": effective,
            "exact_matrix_rank": int(np.linalg.matrix_rank(matrix)),
            "s_1": float(singular[0]),
            "s_169": float(singular[168]),
            "s_457": float(singular[456]),
            "step_size_ratio_vs_control": (norm / reference) ** 2,
        }
        r = rows[name]
        print(f"{name:22s} {r['dimension']:4d} {r['mean_l2_norm']:9.3f} "
              f"{r['norm_ratio_vs_control']:9.3f} {r['effective_rank_1pct']:9d} "
              f"{r['s_1']:9.2f} {r['s_169']:9.3f} {r['s_457']:9.4f}")

    digital_norm = float(np.linalg.norm(digital.numpy(), axis=1).mean())
    print(f"\n[digital block alone]  dim={digital.shape[1]}  mean L2={digital_norm:.3f}")
    print(f"\nExact matrix_rank calls every arm "
          f"{rows['polynomial_control']['exact_matrix_rank']}/"
          f"{rows['workspace_gru']['exact_matrix_rank']}/"
          f"{rows['transformer']['exact_matrix_rank']} -- its default tolerance is far below the")
    print("scale of the expanded coordinates. Effective rank at 1% is the number that")
    print("describes what a delta rule with finitely many labels can actually reach.")
    print("\nstep_size_ratio is (norm ratio)^2: the adapter update is outer(feature, error),")
    print("so one labeled example moves the logit by ||feature||^2 * lr. A single")
    print("adapter_learning_rate is tuned on validation_workspace_oracle and applied to all arms.")

    payload = {
        "draws": args.draws,
        "sequence_length": args.sequence_length,
        "singular_value_threshold": THRESHOLD,
        "arms": rows,
        "digital_block": {
            "dimension": int(digital.shape[1]), "mean_l2_norm": digital_norm
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
