"""UBIQUITOUS EXPRESSION as a feature -- the cheapest gain named in the ENCODE-rE2G paper.

WHY. Gschwind et al. report that adding whether a gene is ubiquitously expressed improves the ABC model by
+0.044 AUPRC. That is one column, and it is larger than everything the 4D chromatin arc produced: analytic
Rouse +0.005, best-physics stiffness arm +0.0005, full extrusion + Langevin +0.010 +/- 0.008, measured Hi-C
loops -0.003 +/- 0.003. This project spent an enormous amount of compute on contact and had not spent five
minutes on the gene.

THE BIOLOGY, and why it is a NEGATIVE predictor. Housekeeping genes are driven by strong constitutive
promoters and are comparatively independent of distal enhancers. Knocking out an element near one is
therefore less likely to move it. The feature should push candidate pairs DOWN, which is the opposite of
almost every other feature in the table and part of why it is not redundant with them.

THE SOURCE. `reference/UbiquitouslyExpressedGenes.txt` from broadinstitute/ABC-Enhancer-Gene-Prediction --
the same list the ABC model itself uses, so this reproduces the published feature rather than inventing a
proxy for it. The count is asserted, because a silently changed upstream file would silently change the
result.

THE CONTROL IS AT THE GENE LEVEL, NOT THE PAIR LEVEL. Permuting the flag across the 10,331 PAIRS would
destroy the fact that the label is a property of a gene, and every gene's pairs share an outcome tendency;
that control would be far too weak and the feature would look better than it is. The permutation here is
across the 2,146 DISTINCT GENES, preserving how many genes are flagged and the pair-count distribution
behind each. That asks the honest question: is it THIS set of genes, or merely "some gene-level label"?
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
INV = OUT / "invivo"
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
URL = ("https://raw.githubusercontent.com/broadinstitute/ABC-Enhancer-Gene-Prediction/"
       "main/reference/UbiquitouslyExpressedGenes.txt")
EXPECT_N = 847        # what the list held when this was measured; assert so upstream drift is not silent
SEEDS = (0, 1, 2, 3, 4)


def load_ubiquitous():
    p = SP / "ubiquitous_genes.txt"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, p)
    genes = {x.strip() for x in open(p) if x.strip()}
    assert len(genes) == EXPECT_N, (
        f"ubiquitous list has {len(genes)} genes, expected {EXPECT_N}. The upstream reference changed; "
        "the numbers in UPDATES.md were measured on the 847-gene version.")
    return genes


def main():
    from crispr_gate import _cv_auprc, _seeded_folds
    from competition import competition_features, load as load_comp

    print("=" * 100)
    print("UBIQUITOUS EXPRESSION -- one column, against the +0.044 the rE2G paper reports for it")
    print("=" * 100)
    df, Xid, Cr = load_comp()
    y, ch = df["crispr_hit"].values, df["chromosome"].values
    gene = df["gene"].values
    ubi = load_ubiquitous()
    isu = np.isin(gene, list(ubi)).astype(float)

    print(f"  {len(y):,} pairs, {int(y.sum())} positives, base rate {y.mean():.4f}")
    print(f"  ubiquitous list {len(ubi)} genes; {int(isu.sum()):,} pairs ({isu.mean()*100:.1f}%) target one "
          f"({pd.Series(gene[isu>0]).nunique()} distinct genes matched)")
    m = isu > 0
    print(f"  hit rate  ubiquitous target {y[m].mean():.4f} ({int(y[m].sum())} pos)  |  "
          f"non-ubiquitous {y[~m].mean():.4f} ({int(y[~m].sum())} pos)  -> {y[~m].mean()/y[m].mean():.2f}x lower")

    U = isu[:, None]
    CF = competition_features(df)

    def sc(X, tag):
        v = np.array([_cv_auprc(X, y, _seeded_folds(ch, s)) for s in SEEDS])
        print(f"  {tag:46s} {v.mean():.4f}  [{' '.join(f'{x:.3f}' for x in v)}]")
        return v

    print()
    a = sc(Xid, "epi + TF identity")
    b = sc(np.hstack([Xid, Cr]), "+ CTCF count            [the 0.663 bar]")
    u = sc(np.hstack([Xid, Cr, U]), "+ ubiquitous")
    c = sc(np.hstack([Xid, Cr, CF]), "+ competition")
    f = sc(np.hstack([Xid, Cr, CF, U]), "+ competition + ubiquitous   [full stack]")

    # GENE-LEVEL control: reassign the flag to a random set of genes of the same size.
    genes_u = sorted(set(gene))
    n_flagged = len({g for g in genes_u if g in ubi})
    ctrl = []
    for s in range(5):
        rng = np.random.default_rng(s)
        fake = set(rng.choice(genes_u, n_flagged, replace=False))
        Us = np.isin(gene, list(fake)).astype(float)[:, None]
        ctrl.append(np.array([_cv_auprc(np.hstack([Xid, Cr, Us]), y, _seeded_folds(ch, sd))
                              for sd in SEEDS]) - b)
    ctrl = np.concatenate(ctrl)

    d = u - b
    net = d.mean() - ctrl.mean()
    se = float(np.sqrt(d.var(ddof=1) / len(d) + ctrl.var(ddof=1) / len(ctrl)))
    print(f"\n  ubiquitous ON TOP OF the bar : {d.mean():+.4f}  "
          f"[per-seed {' '.join(f'{x:+.3f}' for x in d)}]")
    print(f"  gene-level shuffled control  : {ctrl.mean():+.4f}  (5 draws x {len(SEEDS)} seeds, "
          f"sd {ctrl.std(ddof=1):.4f})")
    print(f"  NET OF CONTROL               : {net:+.4f} +/- {se:.4f}   z = {net/max(se,1e-9):+.2f}   "
          f"sign consistent: {bool(np.all(d>0))}")
    print(f"\n  full stack over the bar      : {(f-b).mean():+.4f}")
    print(f"  ubiquitous ON TOP OF competition : {(f-c).mean():+.4f}  "
          f"(vs {d.mean():+.4f} on top of the bar alone -> "
          f"{'independent' if abs((f-c).mean()-d.mean())<0.01 else 'partly redundant'})")
    print(f"\n  the paper reports +0.044 for this feature added to ABC; measured here on top of a "
          f"{b.mean():.4f} model")

    R = {"n_ubiquitous_genes": len(ubi), "pairs_ubiquitous": int(isu.sum()),
         "hit_ubiquitous": float(y[m].mean()), "hit_non_ubiquitous": float(y[~m].mean()),
         "identity": float(a.mean()), "ctcf_bar": float(b.mean()), "plus_ubiquitous": float(u.mean()),
         "plus_competition": float(c.mean()), "full_stack": float(f.mean()),
         "delta_ubiq_on_bar": float(d.mean()), "control": float(ctrl.mean()),
         "net_of_control": float(net), "se": se, "z": float(net / max(se, 1e-9)),
         "sign_consistent": bool(np.all(d > 0)),
         "delta_ubiq_on_competition": float((f - c).mean()),
         "delta_full_stack_over_bar": float((f - b).mean()),
         "paper_reported_for_abc": 0.044}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "ubiquitous.json", "w"), indent=1)
    print(f"\n  -> {OUT/'ubiquitous.json'}")


if __name__ == "__main__":
    main()
