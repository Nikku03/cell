"""Essential vs non-essential exploratory analysis.

Question (user): take all essential genes + an equal number of randomly-picked
non-essential genes, analyse each group separately, and compare them in any
way that reveals difference.

This is the foundational EDA the whole project rests on but never did head-on:
WHAT actually distinguishes an essential gene from a non-essential one?

Method:
  1. Load labels (210K genes, 59 organisms) + all feature tables, join on
     (organism, locus_tag).
  2. Take ALL essentials. Randomly sample an EQUAL number of non-essentials
     (balanced, seed-fixed). This removes the 1:3 class imbalance so group
     comparisons are clean.
  3. For every numeric feature, compare the two groups:
       - mean / median / std in each group
       - Cohen's d (standardized effect size; the headline 'how different')
       - Mann-Whitney U p-value (non-parametric, no normality assumption)
       - AUC of that single feature as an essentiality classifier (0.5=useless,
         1.0=perfect separation) -- the most interpretable 'separation' number
  4. Rank features by |effect|. Report the separating structure.
  5. Same analysis WITHIN a couple of single organisms (so cross-organism
     conservation features don't dominate and hide intrinsic gene properties).

No model, no leakage concern -- this is description, not prediction.
Output: outputs/essential_vs_noness_eda.md (+ console).
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data" / "drive_import" / "labels"


def cohens_d(a, b):
    import numpy as np
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return float("nan")
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    if pooled == 0: return float("nan")
    return (a.mean() - b.mean()) / pooled


def auc_and_p(values, labels):
    """Tie-correct single-feature AUC (= Mann-Whitney U / n_pos*n_neg) and a
    normal-approximation two-sided p-value. Pure pandas/numpy (no scipy)."""
    import numpy as np
    import pandas as pd
    import math
    m = ~np.isnan(values)
    v, y = values[m], labels[m]
    pos = y == 1
    n_pos = int(pos.sum()); n_neg = int((~pos).sum())
    if n_pos < 2 or n_neg < 2:
        return float("nan"), float("nan")
    ranks = pd.Series(v).rank(method="average").values  # tie-averaged
    U = ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0
    auc = U / (n_pos * n_neg)
    mean_U = n_pos * n_neg / 2.0
    sd_U = math.sqrt(n_pos * n_neg * (n_pos + n_neg + 1) / 12.0)
    if sd_U == 0:
        return auc, float("nan")
    z = (U - mean_U) / sd_U
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return auc, p


def main() -> int:
    import pandas as pd
    import numpy as np

    rng = np.random.RandomState(0)
    print("Loading labels + features ...")
    lab = pd.read_csv(D / "labels.csv")[["organism", "locus_tag", "essential"]]
    orth = pd.read_csv(D / "orthology_features.csv")
    cod = pd.read_csv(D / "codon_features.csv")
    reg = pd.read_csv(D / "regulator_features.csv")
    fba = pd.read_csv(D / "fba_features.csv")

    df = lab.merge(orth, on=["organism", "locus_tag"], how="left")
    df = df.merge(cod, on=["organism", "locus_tag"], how="left")
    df = df.merge(reg.drop(columns=["og_id"], errors="ignore"),
                  on=["organism", "locus_tag"], how="left")
    df = df.merge(fba, on=["organism", "locus_tag"], how="left")
    print(f"  joined: {len(df):,} genes, {df.organism.nunique()} organisms")

    # ---- balanced sample: ALL essentials + equal random non-essentials ----
    ess = df[df.essential == 1].copy()
    non = df[df.essential == 0].copy()
    n = min(len(ess), len(non))
    non_s = non.iloc[rng.choice(len(non), n, replace=False)]
    bal = pd.concat([ess, non_s], ignore_index=True)
    print(f"  essentials: {len(ess):,}   sampled non-essentials: {len(non_s):,}")
    print(f"  balanced set: {len(bal):,}")

    # ---- feature list (numeric, interpretable) ----
    feats = [
        "family_frac_essential_leakfree", "family_n_organisms",
        "family_size_total", "n_paralogs_in_genome", "is_orphan",
        "cai", "gc", "gc3", "cds_length",
        "intergenic_prev", "intergenic_next",
        "same_strand_prev", "same_strand_next",
        "is_regulator", "is_signaling", "is_transporter", "is_conditional",
        "fba_essential", "fba_n_rxns", "fba_in_model",
    ]
    feats = [f for f in feats if f in bal.columns]

    lines = ["# Essential vs non-essential genes — separation analysis\n",
             f"All {len(ess):,} essential genes vs {len(non_s):,} randomly "
             f"sampled non-essential genes ({df.organism.nunique()} organisms, "
             f"balanced).\n",
             "Cohen's d: standardized mean difference (|d|>0.8 large, >0.5 "
             "medium, >0.2 small). AUC: single-feature essentiality "
             "discrimination (0.5 useless, 1.0 perfect). Positive d = higher "
             "in essential genes.\n",
             "| feature | ess mean | non mean | Cohen's d | AUC | MWU p |",
             "|---------|---------:|---------:|----------:|----:|------:|"]

    y = bal.essential.values.astype(int)
    results = []
    for f in feats:
        v = pd.to_numeric(bal[f], errors="coerce").values.astype(float)
        ea = pd.to_numeric(ess[f], errors="coerce").values.astype(float)
        na = pd.to_numeric(non_s[f], errors="coerce").values.astype(float)
        # drop degenerate features (constant across the whole balanced set)
        if np.nanstd(v) == 0 or np.isnan(np.nanstd(v)):
            continue
        d = cohens_d(ea, na)
        auc, pval = auc_and_p(v, y)
        results.append((f, np.nanmean(ea), np.nanmean(na), d, auc, pval))

    # sort by |AUC-0.5| (separation strength)
    results.sort(key=lambda r: -abs((r[4] if not np.isnan(r[4]) else 0.5) - 0.5))
    for f, em, nm, d, auc, pval in results:
        lines.append(f"| {f} | {em:.3f} | {nm:.3f} | {d:+.3f} | "
                     f"{auc:.3f} | {pval:.1e} |")

    # ---- WITHIN-organism view (strip conservation, see intrinsic props) ----
    lines.append("\n## Within single organisms (conservation features removed)\n")
    lines.append("Cross-organism conservation (family_frac) dominates the "
                 "pooled view. Within ONE organism every gene shares the same "
                 "phylogenetic context, so this isolates INTRINSIC gene "
                 "properties (length, codon usage, GC, genomic neighbourhood).\n")
    intrinsic = [f for f in ["cds_length", "cai", "gc", "gc3",
                             "intergenic_prev", "intergenic_next",
                             "n_paralogs_in_genome", "is_transporter",
                             "is_regulator", "fba_essential"]
                 if f in bal.columns]
    # pick organisms that HAVE codon features (so cai/gc are populated) AND
    # have enough essentials -- these are the strict-essentiality benchmark orgs
    codon_orgs = set(cod.organism.unique())
    ess_counts = df[df.essential == 1].organism.value_counts()
    top_orgs = [o for o in ess_counts.index if o in codon_orgs][:4]
    if not top_orgs:
        top_orgs = ess_counts.head(4).index.tolist()
    lines.append(f"Organisms examined: {top_orgs}\n")
    lines.append("| feature | " + " | ".join(top_orgs) + " |  (AUC) ")
    lines.append("|---------|" + "|".join(["---:"] * len(top_orgs)) + "|")
    for f in intrinsic:
        cells = []
        for org in top_orgs:
            sub = df[df.organism == org]
            v = pd.to_numeric(sub[f], errors="coerce").values.astype(float)
            yy = sub.essential.values.astype(int)
            auc, _ = auc_and_p(v, yy)
            cells.append(f"{auc:.3f}" if not np.isnan(auc) else "  -  ")
        lines.append(f"| {f} | " + " | ".join(cells) + " |")

    # ---- headline summary ----
    lines.append("\n## What separates them (plain reading)\n")
    strong = [(f, auc, d) for f, em, nm, d, auc, pval in results
              if not np.isnan(auc) and abs(auc - 0.5) > 0.1]
    for f, auc, d in strong[:8]:
        direction = "HIGHER" if d > 0 else "LOWER"
        lines.append(f"- **{f}**: AUC {auc:.3f}, {direction} in essential "
                     f"genes (d={d:+.2f})")

    out = REPO / "outputs" / "essential_vs_noness_eda.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
