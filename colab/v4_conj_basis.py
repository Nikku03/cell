"""Does conjunction growth need an AXIS-ALIGNED basis? Rotate the features and find out.

The question this answers
-------------------------
ADRN-3's conjunction growth worked on DepMap held-out genes (-0.0080 past a
label-shuffled control, 8/8 blocks) and FAILED on the Replogle perturbation-response
task, where it scored 0.0325/0.0320 against the plain bag's 0.0347/0.0336 and lost to
both of its own controls.

The two runs differ in many ways, but one difference is structural rather than
incidental.  On DepMap the features were 60 NAMED gene properties -- compartment,
process, chromosome one-hots, LOEUF, PPI degree, TF flag.  On Replogle the features
were 64 DepMap principal components.  A PC basis is a rotation, and a conjunction that
is sparse in the original basis has no sparse representative after an arbitrary
rotation: the informative product of three named channels becomes a dense polynomial
spread over all 64 coordinates.  The mechanism searches over sparse products of
coordinates, so a rotated basis removes the thing it searches for.

If that is the binding factor, then rotating the DepMap features -- same genes, same
information, same targets, same protocol, ONLY the basis changing -- should destroy the
measured gain.  If the gain survives rotation, the diagnosis is wrong and the Replogle
failure needs a different explanation.

Arms, all on the identical held-out gene blocks
------------------------------------------------
    named          the 60 annotation channels as-is. Reproduces the measured result.
    pca            a full-rank PCA rotation of those same 60 channels. Information
                   preserved exactly (invertible), axis alignment destroyed.
    random_rot     a random orthogonal rotation. Same, without PCA's variance ordering,
                   so any difference between pca and random_rot isolates whether it is
                   the ROTATION or the VARIANCE ORDERING that matters.

Each arm is scored as ridge-with-conjunctions minus plain ridge on the SAME basis, so
the comparison is internal to each basis and cannot be confounded by a basis being
better or worse for the linear part.

PREDECLARED, before running: the named basis keeps its gain; both rotations lose most
or all of it. If instead all three behave alike, then the input basis is not what broke
the Replogle run and I should look at the pooling or the objective mismatch instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v4_abc_conjunctions import (  # noqa: E402
    binarise, build_matrix, expand, grow_conjunctions, ridge_rmse,
)


def rotate(features: np.ndarray, mode: str, seed: int) -> np.ndarray:
    """Return an information-preserving change of basis (or the identity)."""

    if mode == "named":
        return features
    centred = features - features.mean(0, keepdims=True)
    if mode == "pca":
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        return centred @ vt.T                       # full rank: nothing discarded
    if mode == "random_rot":
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(size=(features.shape[1], features.shape[1])))
        return centred @ q
    raise ValueError(mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--max-grown", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=20_000)
    parser.add_argument("--grow-threshold", type=float, default=4.0)
    parser.add_argument("--penalty", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20_260_802)
    args = parser.parse_args(argv)

    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    report("=" * 96)
    report("DOES CONJUNCTION GROWTH NEED AN AXIS-ALIGNED BASIS?")
    report("=" * 96)
    report("  PREDECLARED: named keeps its gain; both rotations lose most or all of it.")
    report("  If all three behave alike, the basis is not what broke the Replogle run.")
    features, target, genes, names = build_matrix(report)

    rng = np.random.default_rng(args.seed)
    blocks = np.array_split(rng.permutation(len(genes)), args.blocks)
    modes = ("named", "pca", "random_rot")
    gains: dict[str, dict[str, list[float]]] = {
        m: {"linear": [], "learned": [], "shuffled": []} for m in modes
    }
    grown_counts: dict[str, list[int]] = {m: [] for m in modes}

    for number, held in enumerate(blocks):
        mask = np.ones(len(genes), bool)
        mask[held] = False
        for mode in modes:
            basis = rotate(features, mode, args.seed + 17 * number)
            train_x, train_y = basis[mask], target[mask]
            test_x, test_y = basis[held], target[held]

            linear = ridge_rmse(train_x, train_y, test_x, test_y, args.penalty)
            gains[mode]["linear"].append(linear)

            signed_train, signed_test = binarise(train_x, test_x)
            block_rng = np.random.default_rng(args.seed + 991 * (number + 1))
            design = np.concatenate([train_x, np.ones((len(train_x), 1))], axis=1)
            weights = np.linalg.solve(
                design.T @ design + args.penalty * np.eye(design.shape[1]),
                design.T @ train_y)
            residual = train_y - design @ weights

            learned = grow_conjunctions(
                signed_train, residual, max_grown=args.max_grown, arities=(2, 3, 5),
                min_obs=64, threshold=args.grow_threshold, rng=block_rng,
                candidates=args.candidates)
            grown_counts[mode].append(len(learned))
            shuffled = grow_conjunctions(
                signed_train, block_rng.permutation(residual),
                max_grown=len(learned), arities=(2, 3, 5), min_obs=64,
                threshold=float("-inf"),
                rng=np.random.default_rng(args.seed + 991 * (number + 1)),
                candidates=args.candidates)

            for key, conj in (("learned", learned), ("shuffled", shuffled)):
                gains[mode][key].append(ridge_rmse(
                    np.concatenate([train_x, expand(signed_train, conj)], axis=1),
                    train_y,
                    np.concatenate([test_x, expand(signed_test, conj)], axis=1),
                    test_y, args.penalty))

    report(f"\n  {args.blocks} held-out gene blocks; every arm scored against plain ridge "
           f"ON ITS OWN BASIS")
    report(f"\n  {'basis':12s} {'ridge':>8s} {'+learned':>9s} {'+shuffled':>10s} "
           f"{'learned-ridge':>14s} {'learned-shuffled':>17s} {'MDE':>8s}  verdict")
    summary: dict[str, object] = {}
    for mode in modes:
        lin = np.array(gains[mode]["linear"])
        lea = np.array(gains[mode]["learned"])
        shu = np.array(gains[mode]["shuffled"])
        gap_l = lea - lin
        gap_s = lea - shu
        sd = float(gap_s.std(ddof=1)) if len(gap_s) > 1 else 0.0
        mde = 3 * sd / np.sqrt(len(gap_s))
        resolved = abs(gap_s.mean()) > mde
        summary[mode] = {
            "ridge": float(lin.mean()), "learned": float(lea.mean()),
            "shuffled": float(shu.mean()),
            "learned_minus_ridge": float(gap_l.mean()),
            "learned_minus_shuffled": float(gap_s.mean()),
            "mde": float(mde), "resolved": bool(resolved),
            "mean_grown": float(np.mean(grown_counts[mode])),
            "blocks_learned_better": int((gap_s < 0).sum()),
        }
        report(f"  {mode:12s} {lin.mean():8.4f} {lea.mean():9.4f} {shu.mean():10.4f} "
               f"{gap_l.mean():+14.4f} {gap_s.mean():+17.4f} {mde:8.4f}  "
               f"{'RESOLVED' if resolved else 'within noise'}")

    named = summary["named"]["learned_minus_shuffled"]
    report(f"\n  RETENTION of the named basis's genuine gain ({named:+.4f}) after rotation:")
    for mode in ("pca", "random_rot"):
        value = summary[mode]["learned_minus_shuffled"]
        report(f"    {mode:12s} {value:+.4f}   "
               f"{100 * value / named if named else float('nan'):6.1f}% retained")
    summary["retention_pca"] = (summary["pca"]["learned_minus_shuffled"] / named
                                if named else None)
    summary["retention_random_rot"] = (summary["random_rot"]["learned_minus_shuffled"] / named
                                       if named else None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"test": "conjunction_growth_basis_dependence",
         "n_genes": len(genes), "n_features": features.shape[1],
         "blocks": args.blocks, "summary": summary, "log": log},
        indent=2, sort_keys=True))
    report(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
