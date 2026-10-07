"""The (gene, line) INTERACTION -- how much is there, how much is real, how much is reachable.

Why this is the remaining question
-----------------------------------
Everything measured so far on DepMap predicts a PER-GENE quantity:

    ridge on annotation beats the unseen-gene baseline by -0.0656 (MDE 0.0109), and
    Test C retains 99.9% of that into cell populations never trained on, because the
    mean gene effect reproduces across two disjoint line populations to RMSE 0.0153 --
    4.0% of the signal sd.

That last number is the point.  The per-gene mean is nearly all of the recoverable
signal *for a per-gene target*, and the cell-line marginal is worth nothing over the
global mean (0.4207 against 0.4206).  So if any cross-cell signal exists at all, it
lives in the INTERACTION: the part of Y[line, gene] left after both marginals are
removed.  This module measures it.

Three questions, in the only order that makes them answerable
--------------------------------------------------------------
1. HOW MUCH IS THERE.  Decompose Var(Y) into gene marginal, line marginal and
   interaction.  If the interaction is a rounding error the rest is moot.

2. HOW MUCH IS REAL.  An interaction term is defined as a residual, so it absorbs every
   source of noise in the screen.  Before asking whether anything predicts it, measure
   what fraction of it is REPRODUCIBLE, by splitting the cell lines into two disjoint
   halves, extracting latent factors independently in each, and asking whether the GENE
   LOADINGS agree.  Loadings that reproduce across disjoint cell populations are real
   structure; loadings that do not are screen noise.  This establishes a CEILING, and
   without it a model explaining 3% of the interaction cannot be told apart from a
   model explaining 60% of what is there to explain.

3. HOW MUCH IS REACHABLE FROM ANNOTATION.  The per-gene mean a_g is predictable from
   annotation.  Are the interaction loadings u_g?  If yes, a never-measured gene's
   *line-specific* profile is partly predictable and the cross-cell problem is open.  If
   no, the wall is confirmed from a second direction.

The prediction protocol, and why each half is legitimate
---------------------------------------------------------
For a held-out gene g in a held-out line c, predict I[c,g] = v_c . u_g where

    v_c  the line's factor scores, estimated from TRAINING GENES in line c. Legitimate:
         a new cell line can be profiled with genes that are already characterised, and
         it never touches gene g.
    u_g  the gene's factor loadings, predicted from ANNOTATION by a ridge fitted on
         training genes only. Legitimate: gene g has no DepMap column anywhere in
         training, so nothing about its own measurements is used.

Controls, all of which must be run
-----------------------------------
    zero            predict no interaction at all. The number to beat.
    oracle_loading  u_g taken from the held-out lines themselves. NOT a legitimate
                    predictor -- it is the CEILING, i.e. what perfect loading knowledge
                    would buy given the factor model and the estimated v_c.
    shuffled_loading  predicted loadings permuted across held-out genes. Holds the
                    magnitude and distribution of the prediction fixed and destroys only
                    the gene correspondence.
    annotation      the real arm.

PREDECLARED, recorded before the run.  I expect (1) the interaction to hold a large
share of the variance, (2) only a modest fraction of it to be reproducible, and (3) the
annotation arm to be at or near zero -- because the same annotation predicts a_g well,
and if it also predicted u_g there would have been cross-cell signal visible in the
dense-response work, which measured none across its four contexts.  A positive result
here would be a genuine surprise and would reopen that question.

Leak guards, re-asserted: ess, ess_src, dep_frac, ess_prob are DepMap-derived and
excluded from the feature matrix at build time.

Usage
-----
    python v4_interaction.py --blocks 6 --rank 16 --output outputs/orphan/v4_interaction.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v4_abc_conjunctions import BANNED, SP  # noqa: E402  -- shared guard + data root
from v4_abc_test_c import CAT_FIELDS, NUM_FIELDS, load  # noqa: E402


def decompose(values: np.ndarray, observed: np.ndarray, report) -> dict:
    """Y = mu + a_gene + b_line + interaction, with the variance budget of each."""

    filled = np.where(observed, values, np.nan)
    mu = float(np.nanmean(filled))
    gene = np.nanmean(filled, axis=0) - mu
    line = np.nanmean(filled - gene[None, :], axis=1) - mu
    interaction = filled - mu - gene[None, :] - line[:, None]

    total = float(np.nanvar(filled))
    parts = {
        "gene_marginal": float(np.nanvar(np.broadcast_to(gene[None, :], filled.shape))),
        "line_marginal": float(np.nanvar(np.broadcast_to(line[:, None], filled.shape))),
        "interaction": float(np.nanvar(interaction)),
    }
    report(f"\n  === 1. VARIANCE BUDGET of DepMap gene effect ===")
    report(f"  {'component':22s} {'variance':>10s} {'share':>8s}")
    for name, value in parts.items():
        report(f"  {name:22s} {value:10.5f} {100*value/total:7.1f}%")
    report(f"  {'total':22s} {total:10.5f} {100.0:7.1f}%")
    return {"mu": mu, "gene": gene, "line": line, "interaction": interaction,
            "budget": {k: v / total for k, v in parts.items()}, "total_var": total}


def factors(interaction: np.ndarray, rows: np.ndarray, rank: int):
    """Top-`rank` left/right factors of the interaction restricted to `rows` (lines).

    Missing entries are set to zero AFTER the marginals are removed, so an unmeasured
    (line, gene) contributes nothing rather than a fabricated value.
    """

    block = np.nan_to_num(interaction[rows], nan=0.0)
    gram = block @ block.T                                # lines x lines, cheap
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(-eigenvalues)[:rank]
    scale = np.sqrt(np.maximum(eigenvalues[order], 1e-12))
    scores = eigenvectors[:, order]                       # lines x rank, orthonormal
    loadings = (block.T @ scores)                         # genes x rank
    return scores, loadings, scale


def subspace_alignment(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Canonical correlations between two gene-loading subspaces.

    1.0 means the two disjoint line halves recovered the same gene structure; values at
    the permutation floor mean they recovered independent noise.
    """

    def orthonormal(matrix: np.ndarray) -> np.ndarray:
        q, _ = np.linalg.qr(matrix - matrix.mean(0, keepdims=True))
        return q

    return np.linalg.svd(orthonormal(left).T @ orthonormal(right), compute_uv=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--penalty", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20_260_802)
    args = parser.parse_args(argv)

    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    report("=" * 100)
    report("THE (GENE, LINE) INTERACTION -- how much, how much is real, how much is reachable")
    report("=" * 100)
    report("  PREDECLARED: the interaction should hold a large variance share, only a")
    report("  modest fraction should be reproducible, and the annotation arm should be")
    report("  at or near zero. A positive annotation result would reopen the cross-cell")
    report("  question that the dense-response work closed at n=4 contexts.")

    import pandas as pd
    frame = pd.read_csv(SP / "CRISPRGeneEffect.csv", index_col=0)
    frame.columns = [c.split(" (")[0] for c in frame.columns]
    frame = frame.loc[:, ~frame.columns.duplicated()]
    raw = frame.to_numpy(np.float32)
    genes_all = list(frame.columns)
    keep = np.isfinite(raw).sum(0) >= 100
    raw, genes_all = raw[:, keep], [g for g, k in zip(genes_all, keep) if k]
    observed = np.isfinite(raw)
    report(f"\n  DepMap: {raw.shape[0]:,} lines x {raw.shape[1]:,} genes, "
           f"{100*observed.mean():.1f}% observed")

    parts = decompose(raw, observed, report)
    interaction = parts["interaction"]

    # ---- 2. how much of the interaction reproduces across disjoint line halves ----
    report(f"\n  === 2. REPRODUCIBILITY of the interaction (split-half over LINES) ===")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(interaction.shape[0])
    half = len(order) // 2
    left_rows, right_rows = order[:half], order[half:2 * half]
    _, left_loadings, left_scale = factors(interaction, left_rows, args.rank)
    _, right_loadings, right_scale = factors(interaction, right_rows, args.rank)
    alignment = subspace_alignment(left_loadings, right_loadings)

    # Permutation floor: the same alignment between one real half and a half whose
    # gene columns have been shuffled. Any subspace of this rank has SOME alignment by
    # dimension alone, so the floor is what "no shared structure" actually looks like.
    shuffled = right_loadings[rng.permutation(right_loadings.shape[0])]
    floor = subspace_alignment(left_loadings, shuffled)
    report(f"  canonical correlations between the two halves' gene loadings, rank {args.rank}:")
    report(f"    real     {np.round(alignment, 3).tolist()}")
    report(f"    permuted {np.round(floor, 3).tolist()}")
    above = int((alignment > floor.max()).sum())
    report(f"  factors clearing the permutation floor ({floor.max():.3f}): "
           f"{above} of {args.rank}")
    report(f"  mean canonical correlation: real {alignment.mean():.4f} vs "
           f"permuted {floor.mean():.4f}")

    # ---- 3. can annotation predict the loadings? ----
    report(f"\n  === 3. IS THE REPRODUCIBLE PART REACHABLE FROM ANNOTATION? ===")
    _, features, genes_feat = load(report)
    position = {g: i for i, g in enumerate(genes_all)}
    usable = [i for i, g in enumerate(genes_feat) if g in position]
    features = features[usable]
    columns = np.asarray([position[genes_feat[i]] for i in usable])
    report(f"  genes with both a DepMap column and annotation: {len(columns):,}")

    gene_blocks = np.array_split(rng.permutation(len(columns)), args.blocks)
    line_blocks = np.array_split(rng.permutation(interaction.shape[0]), args.blocks)
    arms = ("zero", "annotation", "shuffled_loading", "oracle_loading")
    per_block: dict[str, list[float]] = {a: [] for a in arms}
    ceiling_fraction: list[float] = []

    for held_genes, held_lines in zip(gene_blocks, line_blocks):
        gene_mask = np.ones(len(columns), bool); gene_mask[held_genes] = False
        line_mask = np.ones(interaction.shape[0], bool); line_mask[held_lines] = False
        train_cols, test_cols = columns[gene_mask], columns[~gene_mask]

        # Factors from TRAINING lines x TRAINING genes only.
        block = np.nan_to_num(interaction[np.ix_(line_mask, train_cols)], nan=0.0)
        gram = block @ block.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        pick = np.argsort(-eigenvalues)[:args.rank]
        train_loadings = block.T @ eigenvectors[:, pick]          # train genes x rank

        # Held-out line factor scores from TRAINING GENES in those lines. Never touches
        # a held-out gene, so a new line may be profiled with characterised genes.
        held_block = np.nan_to_num(interaction[np.ix_(~line_mask, train_cols)], nan=0.0)
        norm = (train_loadings ** 2).sum(0) + 1e-9
        held_scores = (held_block @ train_loadings) / norm[None, :]   # held lines x rank

        # Ridge: annotation -> loadings, fitted on training genes only.
        design = np.concatenate([features[gene_mask],
                                 np.ones((int(gene_mask.sum()), 1))], axis=1)
        ridge = np.linalg.solve(
            design.T @ design + args.penalty * np.eye(design.shape[1]),
            design.T @ train_loadings)
        test_design = np.concatenate([features[~gene_mask],
                                      np.ones((int((~gene_mask).sum()), 1))], axis=1)
        predicted = test_design @ ridge                              # held genes x rank

        truth = np.nan_to_num(interaction[np.ix_(~line_mask, test_cols)], nan=0.0)
        seen = np.isfinite(interaction[np.ix_(~line_mask, test_cols)])
        oracle = (truth.T @ held_scores) / ((held_scores ** 2).sum(0) + 1e-9)[None, :]
        permuted = predicted[np.random.default_rng(
            args.seed + len(per_block["zero"])).permutation(len(predicted))]

        def rmse(loadings: np.ndarray | None) -> float:
            estimate = 0.0 if loadings is None else held_scores @ loadings.T
            error = (truth - estimate)[seen]
            return float(np.sqrt(np.mean(error ** 2)))

        per_block["zero"].append(rmse(None))
        per_block["annotation"].append(rmse(predicted))
        per_block["shuffled_loading"].append(rmse(permuted))
        per_block["oracle_loading"].append(rmse(oracle))
        gain_oracle = per_block["zero"][-1] - per_block["oracle_loading"][-1]
        gain_real = per_block["zero"][-1] - per_block["annotation"][-1]
        ceiling_fraction.append(gain_real / gain_oracle if gain_oracle > 0 else float("nan"))

    report(f"\n  held-out (gene block, line block) pairs: {args.blocks}, rank {args.rank}")
    report(f"  {'arm':20s} {'RMSE':>9s} {'sd':>8s} {'vs zero':>10s} {'MDE':>8s}  verdict")
    base = np.array(per_block["zero"])
    summary: dict[str, object] = {}
    for arm in arms:
        values = np.array(per_block[arm])
        gap = values - base
        sd = float(gap.std(ddof=1)) if len(gap) > 1 else 0.0
        mde = 3.0 * sd / np.sqrt(len(gap))
        summary[arm] = {"rmse_mean": float(values.mean()),
                        "rmse_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "gap_vs_zero": float(gap.mean()), "mde": float(mde),
                        "resolved": bool(abs(gap.mean()) > mde and arm != "zero"),
                        "per_block": [float(v) for v in values]}
        verdict = "" if arm == "zero" else (
            "BEATS zero" if summary[arm]["resolved"] and gap.mean() < 0 else "no")
        report(f"  {arm:20s} {values.mean():9.4f} {values.std(ddof=1):8.4f} "
               f"{gap.mean():+10.4f} {mde:8.4f}  {verdict}")

    fraction = float(np.nanmean(ceiling_fraction))
    report(f"\n  FRACTION OF THE ACHIEVABLE CEILING reached by annotation: {100*fraction:.1f}%")
    report(f"    (oracle loadings buy {base.mean()-np.array(per_block['oracle_loading']).mean():+.4f}; "
           f"annotation buys {base.mean()-np.array(per_block['annotation']).mean():+.4f})")
    summary["ceiling_fraction"] = fraction
    summary["variance_budget"] = parts["budget"]
    summary["split_half_alignment"] = {
        "real": [float(v) for v in alignment],
        "permutation_floor": [float(v) for v in floor],
        "factors_above_floor": above, "rank": args.rank,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"test": "gene_line_interaction", "n_genes": len(columns),
         "n_lines": int(interaction.shape[0]), "rank": args.rank,
         "blocks": args.blocks, "banned_fields": sorted(BANNED),
         "summary": summary, "log": log}, indent=2, sort_keys=True))
    report(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
