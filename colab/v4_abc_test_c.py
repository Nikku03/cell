"""TEST C -- a gene never measured, in a cell line never seen. Plus A and B for scale.

What C adds over B
------------------
Test B (`v4_abc_conjunctions.py`) withheld the GENE and measured the target across
every line.  It answered: can annotation predict a gene's dependency when the gene has
never been measured?  Answer, on real DepMap: yes -- plain ridge beat the fallback by
-0.0656 against an MDE of 0.0109, and grown conjunctions added -0.0080 of genuine
signal past a label-shuffled control.

Test C withholds BOTH.  That matters because B's target is an average over the same
cell-line population the model trained on, so a gene-level predictor could be fitting
population-specific structure.  C measures the target in a DISJOINT set of lines, so a
prediction only survives if the gene-level signal transfers across cell populations.

The three regimes, all sharing one feature matrix, one set of arms, and one blocking
so the numbers are directly comparable:

    A  seen gene, unseen line   predict target measured in HELD-OUT lines, for
                                TRAINING genes. Gene identity is available, so the
                                per-gene training-line mean is a legitimate arm and is
                                expected to dominate -- that is exactly why Test A
                                cannot test what the v4 viability head claimed.
    B  unseen gene, seen line   predict target measured in TRAINING lines, for
                                HELD-OUT genes. No gene marginal exists.
    C  unseen gene, unseen line predict target measured in HELD-OUT lines, for
                                HELD-OUT genes. No gene marginal, and the target comes
                                from a cell population the model never trained on.

PREDECLARED, recorded before the run
-------------------------------------
C should land close to B, not far below it, and the reason is a measurement already in
hand: the cell-line marginal is worth essentially nothing on this data (MARGINAL-line
RMSE 0.4207 against a GLOBAL-mean of 0.4206, and per-fold marginal spreads of gene sd
0.3887 against line sd 0.0076 -- gene identity is ~51x more variable).  If line
identity carries almost no information, then withholding the lines as well should cost
almost nothing on top of withholding the gene.

A large C-vs-B drop would falsify that and would mean the B result was partly
population-specific.  A small drop confirms the gene-level signal is portable, which is
the stronger claim and the one worth having.

Leak guards, same as Test B and re-asserted here: ess, ess_src, dep_frac and ess_prob
are DepMap-derived and excluded at matrix-build time; co-dependency partners are
forbidden because selecting a held-out gene's partners requires its own DepMap column.

Unit of replication is the (gene block, line block) PAIR, n = blocks.
MDE = 3 * sd / sqrt(n) on the paired per-pair gap.

Usage
-----
    python v4_abc_test_c.py --blocks 8 --output outputs/orphan/v4_abc_test_c.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v4_abc_conjunctions import (  # noqa: E402  -- same directory, shared machinery
    BANNED, SP, binarise, expand, grow_conjunctions, ridge_rmse,
)

CAT_FIELDS = ["comp", "proc", "chrom"]
NUM_FIELDS = ["loeuf", "tf", "ppi", "ndis", "npath", "tss", "cpg", "enh", "dark", "conf", "pubs"]


def load(report):
    """Residual matrix (lines x genes) with the per-line marginal removed, + features."""

    import gzip
    import pandas as pd

    frame = pd.read_csv(SP / "CRISPRGeneEffect.csv", index_col=0)
    frame.columns = [c.split(" (")[0] for c in frame.columns]
    frame = frame.loc[:, ~frame.columns.duplicated()]
    values = frame.to_numpy(np.float32)
    genes = list(frame.columns)
    keep = np.isfinite(values).sum(0) >= 100
    values, genes = values[:, keep], [g for g, k in zip(genes, keep) if k]
    residual = values - np.nanmean(values, axis=1, keepdims=True)
    report(f"  DepMap: {residual.shape[0]:,} lines x {residual.shape[1]:,} genes, "
           f"per-line marginal removed")

    records = {g["name"]: g
               for g in json.load(gzip.open(SP / "cell_complete.json.gz", "rt"))["genes"]}
    numeric = [f for f in NUM_FIELDS if f not in BANNED]
    categorical = [f for f in CAT_FIELDS if f not in BANNED]
    if BANNED & (set(numeric) | set(categorical)):
        raise SystemExit("LEAK: a DepMap-derived field reached the feature matrix")
    present = [i for i, g in enumerate(genes) if g in records]
    residual, genes = residual[:, present], [genes[i] for i in present]

    levels = {f: sorted({str(records[g].get(f, "")) for g in genes}) for f in categorical}
    index = {f: {v: i for i, v in enumerate(levels[f])} for f in categorical}
    width = len(numeric) + sum(len(levels[f]) for f in categorical)
    features = np.zeros((len(genes), width), np.float64)
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
    report(f"  features: {features.shape[1]} columns; EXCLUDED {sorted(BANNED)}")
    report(f"  genes carried forward: {len(genes):,}")
    return residual, features, genes


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

    report("=" * 100)
    report("TEST C -- unseen gene in an unseen cell line. A and B reported alongside.")
    report("=" * 100)
    report("  PREDECLARED: C should land CLOSE to B. The line marginal is worth ~nothing")
    report("  on this data (MARGINAL-line 0.4207 vs GLOBAL-mean 0.4206; gene sd 0.3887")
    report("  against line sd 0.0076), so withholding lines too should cost little.")
    residual, features, genes = load(report)

    rng = np.random.default_rng(args.seed)
    gene_blocks = np.array_split(rng.permutation(len(genes)), args.blocks)
    line_blocks = np.array_split(rng.permutation(residual.shape[0]), args.blocks)
    report(f"\n  {args.blocks} gene blocks (~{len(gene_blocks[0]):,} genes) paired with "
           f"{args.blocks} disjoint line blocks (~{len(line_blocks[0])} lines)")
    report("  unit of replication is the (gene block, line block) PAIR")

    arms = ("baseline", "gene_marginal", "ridge_linear",
            "ridge_plus_learned", "ridge_plus_random", "ridge_plus_shuffled")
    results: dict[str, dict[str, list[float]]] = {
        regime: {arm: [] for arm in arms} for regime in ("A", "B", "C")
    }
    grown: list[int] = []

    for number, (held_genes, held_lines) in enumerate(zip(gene_blocks, line_blocks)):
        gene_mask = np.ones(len(genes), bool); gene_mask[held_genes] = False
        line_mask = np.ones(residual.shape[0], bool); line_mask[held_lines] = False

        # Target measured on TRAINING lines vs on HELD-OUT lines. Both are per-gene.
        train_line_target = np.nanmean(residual[line_mask], axis=0)
        held_line_target = np.nanmean(residual[~line_mask], axis=0)
        finite = np.isfinite(train_line_target) & np.isfinite(held_line_target)

        train_genes = gene_mask & finite
        test_genes = (~gene_mask) & finite

        # Regime -> (rows evaluated, target used for evaluation)
        regimes = {
            "A": (train_genes, held_line_target),    # seen gene,   unseen line
            "B": (test_genes, train_line_target),    # unseen gene, seen line
            "C": (test_genes, held_line_target),     # unseen gene, unseen line
        }
        fit_x, fit_y = features[train_genes], train_line_target[train_genes]

        signed_fit, _ = binarise(fit_x, fit_x)
        block_rng = np.random.default_rng(args.seed + 991 * (number + 1))
        design = np.concatenate([fit_x, np.ones((len(fit_x), 1))], axis=1)
        gram = design.T @ design + args.penalty * np.eye(design.shape[1])
        weights = np.linalg.solve(gram, design.T @ fit_y)
        fit_residual = fit_y - design @ weights

        learned = grow_conjunctions(
            signed_fit, fit_residual, max_grown=args.max_grown, arities=(2, 3, 5),
            min_obs=64, threshold=args.grow_threshold, rng=block_rng,
            candidates=args.candidates)
        grown.append(len(learned))
        shuffled = grow_conjunctions(
            signed_fit, block_rng.permutation(fit_residual), max_grown=len(learned),
            arities=(2, 3, 5), min_obs=64, threshold=float("-inf"),
            rng=np.random.default_rng(args.seed + 991 * (number + 1)),
            candidates=args.candidates)
        random_pick: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        while len(random_pick) < len(learned):
            arity = int(block_rng.choice((2, 3, 5)))
            pick = tuple(sorted(block_rng.choice(
                signed_fit.shape[1], size=arity, replace=False).tolist()))
            if pick not in seen:
                seen.add(pick); random_pick.append(pick)

        for regime, (rows, target) in regimes.items():
            eval_x, eval_y = features[rows], target[rows]
            centre = float(fit_y.mean())
            results[regime]["baseline"].append(
                float(np.sqrt(np.mean((eval_y - centre) ** 2))))
            # Gene identity: the per-gene TRAINING-LINE mean. Defined only where the
            # gene was trained on, which is regime A alone -- in B and C it is
            # undefined for every evaluated gene and collapses to the baseline. That
            # collapse is the measurement, not a bug.
            lookup = (train_line_target[rows] if regime == "A"
                      else np.full(int(rows.sum()), centre))
            results[regime]["gene_marginal"].append(
                float(np.sqrt(np.mean((eval_y - lookup) ** 2))))
            results[regime]["ridge_linear"].append(
                ridge_rmse(fit_x, fit_y, eval_x, eval_y, args.penalty))
            _, signed_eval = binarise(fit_x, eval_x)
            for arm, conjunctions in (("ridge_plus_learned", learned),
                                      ("ridge_plus_random", random_pick),
                                      ("ridge_plus_shuffled", shuffled)):
                results[regime][arm].append(ridge_rmse(
                    np.concatenate([fit_x, expand(signed_fit, conjunctions)], axis=1),
                    fit_y,
                    np.concatenate([eval_x, expand(signed_eval, conjunctions)], axis=1),
                    eval_y, args.penalty))

    report(f"\n  conjunctions grown per block: mean {np.mean(grown):.1f} (cap {args.max_grown})")
    summary: dict[str, object] = {}
    for regime, label in (("A", "seen gene, UNSEEN line"),
                          ("B", "UNSEEN gene, seen line"),
                          ("C", "UNSEEN gene, UNSEEN line")):
        report(f"\n  === TEST {regime}: {label} ===")
        report(f"  {'arm':22s} {'RMSE':>9s} {'sd':>8s}   {'vs baseline':>12s} {'MDE':>8s}")
        base = np.array(results[regime]["baseline"])
        block = {}
        for arm in arms:
            values = np.array(results[regime][arm])
            gap = values - base
            sd = float(gap.std(ddof=1)) if len(gap) > 1 else 0.0
            mde = 3.0 * sd / np.sqrt(len(gap))
            block[arm] = {"rmse_mean": float(values.mean()),
                          "rmse_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                          "gap_vs_baseline": float(gap.mean()), "mde": float(mde),
                          "resolved": bool(abs(gap.mean()) > mde),
                          "per_block": [float(v) for v in values]}
            report(f"  {arm:22s} {values.mean():9.4f} {values.std(ddof=1):8.4f}   "
                   f"{gap.mean():+12.4f} {mde:8.4f}"
                   f"{'  resolved' if block[arm]['resolved'] and arm != 'baseline' else ''}")
        summary[regime] = block

    def paired(regime: str, left: str, right: str) -> dict:
        gap = np.array(results[regime][left]) - np.array(results[regime][right])
        sd = float(gap.std(ddof=1)) if len(gap) > 1 else 0.0
        mde = 3.0 * sd / np.sqrt(len(gap))
        return {"mean": float(gap.mean()), "mde": float(mde),
                "resolved": bool(abs(gap.mean()) > mde),
                "left_better_blocks": int((gap < 0).sum()), "n": len(gap)}

    report(f"\n  === does the conjunction gain survive into C? paired per block ===")
    report(f"  {'contrast':34s} {'A':>10s} {'B':>10s} {'C':>10s}")
    for left, right, name in (
        ("ridge_linear", "baseline", "ridge - baseline"),
        ("ridge_plus_learned", "ridge_linear", "learned - ridge"),
        ("ridge_plus_learned", "ridge_plus_shuffled", "learned - shuffled"),
    ):
        cells = []
        for regime in ("A", "B", "C"):
            stats = paired(regime, left, right)
            summary.setdefault("contrasts", {}).setdefault(name, {})[regime] = stats
            cells.append(f"{stats['mean']:+.4f}{'*' if stats['resolved'] else ' '}")
        report(f"  {name:34s} {cells[0]:>10s} {cells[1]:>10s} {cells[2]:>10s}")
    report("  (* = the paired gap exceeds its own MDE)")

    b_ridge = summary["B"]["ridge_linear"]["gap_vs_baseline"]
    c_ridge = summary["C"]["ridge_linear"]["gap_vs_baseline"]
    report(f"\n  THE C-vs-B QUESTION: ridge's gain over baseline is {b_ridge:+.4f} in B "
           f"and {c_ridge:+.4f} in C")
    report(f"    retained into an unseen cell population: {100*c_ridge/b_ridge:.1f}%")
    summary["c_over_b_retention"] = float(c_ridge / b_ridge)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"test": "ABC_gene_and_line_holdout", "n_genes": len(genes),
         "n_blocks": args.blocks, "banned_fields": sorted(BANNED),
         "mean_conjunctions_grown": float(np.mean(grown)),
         "summary": summary, "log": log}, indent=2, sort_keys=True))
    report(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
