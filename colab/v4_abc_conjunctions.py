"""Does ADRN-3's conjunction growth help on REAL cell data? Test B, held-out genes.

The predeclared prediction, recorded before the run
---------------------------------------------------
It should NOT help, and the reason is stateable in advance.  Criterion 2 measured an
*untrained, zero-parameter* bag-of-partners (0.0341) beating a real Set Transformer
(0.0297) at a matched 108,160-parameter budget on this same data.  That says the
binding problem is learned components fitting noise, not insufficient nonlinearity.
A mechanism whose entire job is to add representational capacity should therefore
make things worse, or at best do nothing.

The reason to run it anyway is that the prediction is falsifiable and the run is
cheap, and because there is one specific reason it might work: gene annotation
contains genuinely discrete features (compartment, process, chromosome, TF flag)
where high-order interactions are biologically plausible in a way they are not for
continuous expression signal.  On ADRN-2B's synthetic parity tasks the same rule went
from a verified floor of 0.4949 to 0.9425, 98.1% of an oracle ceiling, with the
random-conjunction control pinned at the floor.  This asks whether any of that
survives contact with real data.

What Test B is
--------------
`v4_viability_head.py` passed 8/8 gates on held-out CELL LINES, but that is Test A and
Test A never withholds gene identity -- and gene identity is the whole game here:

    GLOBAL-mean         RMSE 0.4207
    MARGINAL-gene alone RMSE 0.1580     a per-gene constant closes 96% of the distance
    MODEL               RMSE 0.1465     everything learned is 0.0114 on top of that

Test B withholds the GENE.  For a held-out gene the per-gene marginal is undefined and
falls back to the global mean, so the baseline is `mu + b_line` and every learned arm
is predicting the residual -- which is, in substance, the gene's own effect deviation
predicted from annotation alone.  This module measures that residual directly at the
gene level, which is the substance of Test B without the (gene, line) sampling
machinery.

Leakage guard, inherited from the A/B/C harness and re-asserted here
--------------------------------------------------------------------
`ess`, `ess_src` and `dep_frac` are DepMap-derived and would leak the label.  They are
excluded, and the exclusion is asserted at matrix-build time rather than assumed.
Co-dependency partners are also forbidden in Test B for the same reason: selecting a
held-out gene's partners requires that gene's own DepMap column.

Arms
----
    baseline_zero        predict the training mean residual. This IS MARGINAL-both's
                         fallback for an unseen gene, and it is the number to beat.
    ridge_linear         ridge on the annotation features. The shipped capability.
    ridge_plus_learned   + conjunctions grown by the ADRN-3 local rule
    ridge_plus_random    + the SAME NUMBER of conjunctions drawn UNIFORMLY from the
                         same pool without seeing a label. Separates "the rule works"
                         from "more features help".
    ridge_plus_shuffled  + the SAME NUMBER of conjunctions chosen by the SAME RULE
                         applied to a PERMUTED residual. This is the stricter control:
                         it holds the candidate pool, the scoring machinery and the
                         top-k selection pressure constant, and destroys only the
                         correspondence between a product and the error. Random draws
                         uniformly; shuffled draws from the extreme tail of a noise
                         distribution, so it prices in the winner's curse of screening
                         20,000 candidates. If learned does not clear shuffled, the
                         rule is exploiting selection-on-noise rather than finding
                         real conjunctions.

Unit of replication is the held-out GENE BLOCK, not the gene and not a seed:
MDE = 3 * sd / sqrt(n_blocks) on the PAIRED per-block gap.

Usage
-----
    python v4_abc_conjunctions.py --blocks 8 --output outputs/orphan/v4_abc_conjunctions.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np

BANNED = {"ess", "ess_src", "dep_frac", "ess_prob"}
NUM_FIELDS = ["loeuf", "tf", "ppi", "ndis", "npath", "tss", "cpg", "enh", "dark", "conf", "pubs"]
CAT_FIELDS = ["comp", "proc", "chrom"]
SP = Path(os.environ.get(
    "CELL_SCRATCH",
    "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))


def build_matrix(report) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Annotation features and the per-gene DepMap residual target."""

    import pandas as pd

    frame = pd.read_csv(SP / "CRISPRGeneEffect.csv", index_col=0)
    frame.columns = [c.split(" (")[0] for c in frame.columns]
    frame = frame.loc[:, ~frame.columns.duplicated()]
    values = frame.to_numpy(np.float32)
    genes = list(frame.columns)
    observed = np.isfinite(values)
    keep = observed.sum(0) >= 100
    values, genes = values[:, keep], [g for g, k in zip(genes, keep) if k]
    report(f"  DepMap: {values.shape[0]:,} lines x {values.shape[1]:,} genes "
           f"(>=100 measured lines)")

    # Target: the per-gene mean after removing the per-LINE marginal, which is what a
    # held-out gene's residual reduces to once its own marginal is unavailable.
    line_mean = np.nanmean(values, axis=1, keepdims=True)
    residual = values - line_mean
    target = np.nanmean(residual, axis=0)
    report(f"  target: per-gene mean residual after the line marginal; "
           f"sd {target.std():.4f}, range [{target.min():.3f}, {target.max():.3f}]")

    records = {g["name"]: g for g in json.load(gzip.open(SP / "cell_complete.json.gz", "rt"))["genes"]}
    numeric = [f for f in NUM_FIELDS if f not in BANNED]
    categorical = [f for f in CAT_FIELDS if f not in BANNED]
    leaked = BANNED & (set(numeric) | set(categorical))
    if leaked:
        raise SystemExit(f"LEAK: DepMap-derived fields {leaked} reached the feature matrix")

    present = [i for i, g in enumerate(genes) if g in records]
    genes = [genes[i] for i in present]
    target = target[present]

    levels = {f: sorted({str(records[g].get(f, "")) for g in genes}) for f in categorical}
    index = {f: {v: i for i, v in enumerate(levels[f])} for f in categorical}
    width = len(numeric) + sum(len(levels[f]) for f in categorical)
    features = np.zeros((len(genes), width), np.float64)
    names: list[str] = list(numeric)
    for f in categorical:
        names.extend(f"{f}={v}" for v in levels[f])
    for row, gene in enumerate(genes):
        record = records[gene]
        for col, field in enumerate(numeric):
            raw = record.get(field, 0.0)
            features[row, col] = float(raw) if isinstance(raw, (int, float)) else 0.0
        offset = len(numeric)
        for f in categorical:
            features[row, offset + index[f][str(record.get(f, ""))]] = 1.0
            offset += len(levels[f])
    block = features[:, :len(numeric)]
    features[:, :len(numeric)] = np.log1p(np.abs(block)) * np.sign(block)
    finite = np.isfinite(target)
    report(f"  features: {features.shape[1]} columns from {len(numeric)} numeric + "
           f"{len(categorical)} categorical; EXCLUDED as DepMap-derived {sorted(BANNED)}")
    report(f"  genes with annotation and a finite target: {int(finite.sum()):,}")
    return features[finite], target[finite], [g for g, f in zip(genes, finite) if f], names


def binarise(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed +/-1 channels, thresholded at the TRAINING median only."""

    threshold = np.median(train, axis=0)
    to_signed = lambda m: np.where(m > threshold[None, :], 1.0, -1.0)
    return to_signed(train), to_signed(other)


def grow_conjunctions(signed: np.ndarray, residual: np.ndarray, *, max_grown: int,
                      arities: tuple[int, ...], min_obs: int, threshold: float,
                      rng: np.random.Generator, candidates: int) -> list[tuple[int, ...]]:
    """The ADRN-3 rule: running mean of (product x error), scored as |mean| * sqrt(n).

    The channel count here is far larger than the synthetic benchmark's 16, so the
    candidate set is sampled once up front rather than enumerated; the random control
    draws from the same pool, so sampling cannot by itself explain a difference.
    """

    n_channels = signed.shape[1]
    pool: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(pool) < candidates:
        arity = int(rng.choice(arities))
        pick = tuple(sorted(rng.choice(n_channels, size=arity, replace=False).tolist()))
        if pick not in seen:
            seen.add(pick)
            pool.append(pick)
    products = np.stack([np.prod(signed[:, list(p)], axis=1) for p in pool])
    if len(residual) < min_obs:
        return []
    mean = (products * residual[None, :]).mean(axis=1)
    score = np.abs(mean) * np.sqrt(len(residual))
    order = np.argsort(-score)
    return [pool[int(i)] for i in order[:max_grown] if score[i] >= threshold]


def expand(signed: np.ndarray, conjunctions: list[tuple[int, ...]]) -> np.ndarray:
    if not conjunctions:
        return np.zeros((len(signed), 0))
    return np.stack([np.prod(signed[:, list(c)], axis=1) for c in conjunctions], axis=1)


def ridge_rmse(train_x, train_y, test_x, test_y, penalty: float) -> float:
    design = np.concatenate([train_x, np.ones((len(train_x), 1))], axis=1)
    gram = design.T @ design + penalty * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ train_y)
    test_design = np.concatenate([test_x, np.ones((len(test_x), 1))], axis=1)
    return float(np.sqrt(np.mean((test_design @ weights - test_y) ** 2)))


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
    report("TEST B on REAL cell data: does ADRN-3 conjunction growth predict a gene "
           "it has never seen?")
    report("=" * 96)
    report("  PREDECLARED: it should not help. Criterion 2 measured an untrained "
           "zero-parameter bag")
    report("  beating a Set Transformer on this data, so the limit is noise-fitting, "
           "not capacity.")
    features, target, genes, names = build_matrix(report)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(genes))
    blocks = np.array_split(order, args.blocks)
    report(f"\n  held-out GENE blocks: {args.blocks} disjoint blocks of "
           f"~{len(blocks[0]):,} genes; unit of replication is the block")

    arms = ("baseline_zero", "ridge_linear", "ridge_plus_learned",
            "ridge_plus_random", "ridge_plus_shuffled")
    per_block: dict[str, list[float]] = {a: [] for a in arms}
    grown_counts: list[int] = []
    overlap_counts: list[int] = []

    for number, held in enumerate(blocks):
        mask = np.ones(len(genes), bool)
        mask[held] = False
        train_x, train_y = features[mask], target[mask]
        test_x, test_y = features[held], target[held]

        centre = train_y.mean()
        per_block["baseline_zero"].append(
            float(np.sqrt(np.mean((test_y - centre) ** 2))))
        per_block["ridge_linear"].append(
            ridge_rmse(train_x, train_y, test_x, test_y, args.penalty))

        signed_train, signed_test = binarise(train_x, test_x)
        block_rng = np.random.default_rng(args.seed + 991 * (number + 1))
        # Grow on the TRAINING residual after the linear fit: the conjunctions are
        # asked to explain what the linear part could not, which is the same job they
        # had on the synthetic benchmark.
        design = np.concatenate([train_x, np.ones((len(train_x), 1))], axis=1)
        gram = design.T @ design + args.penalty * np.eye(design.shape[1])
        weights = np.linalg.solve(gram, design.T @ train_y)
        residual = train_y - design @ weights

        learned = grow_conjunctions(
            signed_train, residual, max_grown=args.max_grown, arities=(2, 3, 5),
            min_obs=64, threshold=args.grow_threshold, rng=block_rng,
            candidates=args.candidates)
        grown_counts.append(len(learned))
        random_pick: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        while len(random_pick) < len(learned):
            arity = int(block_rng.choice((2, 3, 5)))
            pick = tuple(sorted(block_rng.choice(
                signed_train.shape[1], size=arity, replace=False).tolist()))
            if pick not in seen:
                seen.add(pick)
                random_pick.append(pick)

        # SAME rule, SAME pool, SAME count -- on a permuted residual. The threshold is
        # dropped to -inf so the count matches `learned` exactly; the point is to price
        # the top-k selection pressure over 20,000 candidates, not to ask whether noise
        # clears a significance bar.
        shuffled = grow_conjunctions(
            signed_train, block_rng.permutation(residual), max_grown=len(learned),
            arities=(2, 3, 5), min_obs=64, threshold=float("-inf"),
            rng=np.random.default_rng(args.seed + 991 * (number + 1)),
            candidates=args.candidates)
        overlap_counts.append(len(set(learned) & set(shuffled)))

        for arm, conjunctions in (("ridge_plus_learned", learned),
                                  ("ridge_plus_random", random_pick),
                                  ("ridge_plus_shuffled", shuffled)):
            per_block[arm].append(ridge_rmse(
                np.concatenate([train_x, expand(signed_train, conjunctions)], axis=1),
                train_y,
                np.concatenate([test_x, expand(signed_test, conjunctions)], axis=1),
                test_y, args.penalty))

    report(f"\n  conjunctions grown per block: mean {np.mean(grown_counts):.1f}, "
           f"min {min(grown_counts)}, max {max(grown_counts)} (cap {args.max_grown})")
    if max(grown_counts) == 0:
        report("  !! the growth rule never fired: ridge_plus_learned is IDENTICAL to "
               "ridge_linear and is not a result")

    report(f"\n  {'arm':22s} {'RMSE':>9s} {'sd':>8s}   {'vs baseline':>12s} {'MDE':>8s}  verdict")
    base = np.array(per_block["baseline_zero"])
    summary: dict[str, object] = {}
    for arm in arms:
        values = np.array(per_block[arm])
        gap = values - base                      # negative is better
        sd = float(gap.std(ddof=1)) if len(gap) > 1 else 0.0
        mde = 3.0 * sd / np.sqrt(len(gap))
        beats = arm != "baseline_zero" and (-gap.mean()) > mde
        summary[arm] = {
            "rmse_mean": float(values.mean()),
            "rmse_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "paired_gap_vs_baseline": float(gap.mean()),
            "paired_gap_sd": sd, "mde": float(mde),
            "beats_baseline_by_more_than_mde": bool(beats),
            "per_block_rmse": [float(v) for v in values],
        }
        verdict = "" if arm == "baseline_zero" else ("BEATS" if beats else "no")
        report(f"  {arm:22s} {values.mean():9.4f} {values.std(ddof=1):8.4f}   "
               f"{gap.mean():+12.4f} {mde:8.4f}  {verdict}")

    def paired(left: str, right: str) -> dict:
        gap = np.array(per_block[left]) - np.array(per_block[right])
        sd = float(gap.std(ddof=1)) if len(gap) > 1 else 0.0
        mde = 3.0 * sd / np.sqrt(len(gap))
        return {"mean": float(gap.mean()), "sd": sd, "mde": float(mde),
                "resolved": bool(abs(gap.mean()) > mde),
                "left_better_blocks": int((gap < 0).sum()), "n_blocks": len(gap),
                "per_block": [float(v) for v in gap]}

    report(f"\n  {'contrast':38s} {'gap':>9s} {'MDE':>9s}  blocks  verdict")
    contrasts = {
        "learned_minus_linear": ("ridge_plus_learned", "ridge_linear"),
        "random_minus_linear": ("ridge_plus_random", "ridge_linear"),
        "shuffled_minus_linear": ("ridge_plus_shuffled", "ridge_linear"),
        "learned_minus_random": ("ridge_plus_learned", "ridge_plus_random"),
        "learned_minus_shuffled": ("ridge_plus_learned", "ridge_plus_shuffled"),
        "shuffled_minus_random": ("ridge_plus_shuffled", "ridge_plus_random"),
    }
    for name, (left, right) in contrasts.items():
        stats = paired(left, right)
        summary[name] = stats
        report(f"  {name:38s} {stats['mean']:+9.4f} {stats['mde']:9.4f}  "
               f"{stats['left_better_blocks']}/{stats['n_blocks']}   "
               f"{'RESOLVED' if stats['resolved'] else 'within noise'}")
    report(f"\n  learned/shuffled conjunction overlap: mean "
           f"{np.mean(overlap_counts):.1f} of {np.mean(grown_counts):.1f} "
           f"-- near zero is expected; a high overlap would mean the permutation "
           f"failed to break the correspondence")
    summary["mean_learned_shuffled_overlap"] = float(np.mean(overlap_counts))
    total = summary["learned_minus_linear"]["mean"]
    report(f"\n  DECOMPOSITION of the learned arm's {total:+.4f} over plain ridge:")
    for label, key in (("uniform capacity        ", "random_minus_linear"),
                       ("selection-on-noise      ", "shuffled_minus_linear")):
        part = summary[key]["mean"]
        report(f"    {label} {part:+.4f}  ({100*part/total:5.1f}%)")
    residual_gain = summary["learned_minus_shuffled"]["mean"]
    report(f"    genuine conjunction find {residual_gain:+.4f}  "
           f"({100*residual_gain/total:5.1f}%)  "
           f"{'RESOLVED' if summary['learned_minus_shuffled']['resolved'] else 'WITHIN NOISE'}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "test": "B_held_out_gene_real_depmap",
        "predeclared_expectation": "conjunction growth should NOT help; the criterion-2 "
                                   "result implies the limit is noise-fitting, not capacity",
        "n_genes": len(genes), "n_features": features.shape[1],
        "n_blocks": args.blocks, "banned_fields": sorted(BANNED),
        "mean_conjunctions_grown": float(np.mean(grown_counts)),
        "summary": summary, "log": log,
    }, indent=2, sort_keys=True))
    report(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
