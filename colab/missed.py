"""WHAT SEPARATES THE 233 MISSED POSITIVES FROM THE 336 FOUND ONES -- and from the negatives they hide among.

THE TRAP THIS AVOIDS. Comparing missed positives to found positives on the model's own features is
circular: the model ranks BY those features, so of course the ones it ranks low differ on them. Every
feature will look "significant" and none of it means anything actionable.

THE TEST THAT IS NOT CIRCULAR. Compare each missed positive to NEGATIVES THE MODEL SCORES THE SAME. If a
missed positive and a negative look identical to the model, but differ on some measurable property, that
property is signal the model is failing to use -- a lead. If they differ on nothing, the information
genuinely is not in the data we have, and the ceiling is real rather than an engineering gap.

    missed positive  vs  score-matched negative     -> UNEXPLOITED SIGNAL if anything differs
    missed positive  vs  found positive             -> descriptive only, circular by construction

Score matching is done in narrow bins of the model's own out-of-fold prediction, so the comparison holds
constant everything the model already knows.

WHAT IS TESTED. Every base epigenetic feature, the 311 individual TF ChIP tracks, the competition block,
ubiquitous expression, measured Hi-C, and two structural questions the feature table cannot express:
whether the missed pair's gene already has another element the model DID find, and whether its element
regulates several genes. Multiple testing is corrected; with 300+ tests, uncorrected p-values are noise.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
INV = OUT / "invivo"
TOPK = 500          # "found" = ranked in the top 500, where precision is still ~0.7
NBINS = 20          # score-matching resolution


def main():
    from scipy import stats
    from ceiling import build, bench_cols, oof
    from contact_gate import load_ctcf, contact_features
    from competition import competition_features
    from ubiquitous import load_ubiquitous
    from hic_gate import read_bedpe, read_domains, hic_features, LOOPS, DOMAINS

    df, X = build()
    y, ch = df["crispr_hit"].values, df["chromosome"].values
    es, pw = bench_cols(df)
    p = np.mean([oof(X, y, ch, s)[0] for s in (0, 1, 2)], 0)
    order = np.argsort(-p)
    rank = np.empty(len(y), int)
    rank[order] = np.arange(len(y))

    found = (y == 1) & (rank < TOPK)
    missed = (y == 1) & (rank >= TOPK)
    neg = y == 0
    print("=" * 100)
    print("WHAT SEPARATES THE MISSED POSITIVES FROM THE ONES WE FIND?")
    print("=" * 100)
    print(f"  found {found.sum()}   missed {missed.sum()}   negatives {neg.sum():,}")

    # ---- assemble every named feature we have ----
    comp = json.load(open(INV / "compendium_tf.json"))
    et, tfl = comp["element_tfs"], comp["tf_list"]
    tss = {}
    with open(INV / "crispr_egpairs.tsv") as fh:
        r = csv.DictReader(fh, delimiter="\t")
        for row in r:
            k = (f"{row['chrom']}:{row['chromStart']}-{row['chromEnd']}", row["measuredGeneSymbol"])
            try:
                tss[k] = (row["chrom"], (int(row["chromStart"]) + int(row["chromEnd"])) // 2,
                          int(row["startTSS"]))
            except ValueError:
                pass
    rows = [tss[(e, g)] for e, g in zip(df["element"], df["gene"])]
    H = hic_features(rows, read_bedpe(LOOPS["intact"]), read_domains(DOMAINS["intact"]))
    Cr = contact_features(rows, load_ctcf())
    CF = competition_features(df)
    tfmat = np.zeros((len(df), len(tfl)), np.int8)
    for a, ek in enumerate(df["element"].values):
        for ti in et.get(ek, []):
            tfmat[a, ti] = 1

    feats = {}
    for c in ["log_dist", "atac_enh", "h3k27ac_enh", "polr2a_enh", "procap_enh", "tf_count",
              "promoter_atac", "promoter_polii", "gene_expr", "abc_score"]:
        feats[c] = df[c].values.astype(float)
    for i, n in enumerate(["ctcf_between", "ctcf_between_max", "ctcf_between_sum",
                           "ctcf_near_elem", "ctcf_near_tss", "contact_insul"][:Cr.shape[1]]):
        feats[f"CTCF:{n}"] = Cr[:, i]
    for i, n in enumerate(["n_candidates", "dist_rank", "activity_rank", "activity_share",
                           "act_x_prox_share", "dist_excess"]):
        feats[f"COMP:{n}"] = CF[:, i]
    for i, n in enumerate(["loop_connects", "loop_obs", "loop_enrich", "anchors_at_elem",
                           "anchors_at_tss", "same_domain", "domain_bounds_between"]):
        feats[f"HIC:{n}"] = H[:, i]
    feats["ubiquitous"] = np.isin(df["gene"].values, list(load_ubiquitous())).astype(float)

    # structural properties the feature table cannot express
    gser, eser = pd.Series(df["gene"].values), pd.Series(df["element"].values)
    gene_pos = pd.Series(y).groupby(gser).transform("sum").values
    feats["STRUCT:other_positives_same_gene"] = gene_pos - y
    feats["STRUCT:found_positive_same_gene"] = (
        pd.Series(found.astype(int)).groupby(gser).transform("sum").values - found.astype(int))
    feats["STRUCT:genes_per_element"] = eser.groupby(eser).transform("size").values
    feats["STRUCT:element_total_positives"] = pd.Series(y).groupby(eser).transform("sum").values - y
    for i, n in enumerate(tfl):
        feats[f"TF:{n}"] = tfmat[:, i].astype(float)

    # ---- THE NON-CIRCULAR TEST: missed positives vs score-matched negatives ----
    edges = np.quantile(p, np.linspace(0, 1, NBINS + 1))
    edges[-1] += 1e-9
    b = np.clip(np.digitize(p, edges) - 1, 0, NBINS - 1)
    wanted = pd.Series(b[missed]).value_counts()
    rng = np.random.default_rng(0)
    sel = []
    for bin_id, cnt in wanted.items():
        pool = np.where(neg & (b == bin_id))[0]
        if len(pool) == 0:
            continue
        sel.append(rng.choice(pool, min(cnt * 10, len(pool)), replace=False))
    matched_neg = np.concatenate(sel)
    print(f"  score-matched negatives drawn: {len(matched_neg):,} "
          f"(median model score  missed {np.median(p[missed]):.4f} vs matched {np.median(p[matched_neg]):.4f})")

    res = []
    for name, v in feats.items():
        a, c = v[missed], v[matched_neg]
        if np.nanstd(np.concatenate([a, c])) == 0:
            continue
        try:
            u, pv = stats.mannwhitneyu(a, c, alternative="two-sided")
        except ValueError:
            continue
        # rank-biserial effect size: 0 = identical, +/-1 = complete separation
        rb = 2 * u / (len(a) * len(c)) - 1
        res.append((pv, rb, name, float(np.nanmean(a)), float(np.nanmean(c))))
    res.sort()
    m = len(res)
    print(f"\n  {m} features tested; Bonferroni threshold p < {0.05/m:.2e}\n")
    print("  MISSED POSITIVES vs SCORE-MATCHED NEGATIVES  (anything here is signal the model is NOT using)")
    print(f"    {'feature':38s} {'p':>10s} {'effect':>8s} {'missed':>10s} {'matched neg':>12s}")
    nsig = 0
    for pv, rb, name, ma, mc in res[:15]:
        star = "  <-- SURVIVES Bonferroni" if pv < 0.05 / m else ""
        nsig += pv < 0.05 / m
        print(f"    {name:38s} {pv:10.2e} {rb:+8.3f} {ma:10.4f} {mc:12.4f}{star}")
    print(f"\n  features surviving Bonferroni: {sum(1 for r in res if r[0] < 0.05/m)} of {m}")

    # ---- descriptive, and labelled as circular ----
    print("\n  MISSED vs FOUND positives (CIRCULAR -- the model ranked on these; shown for description only)")
    res2 = []
    for name, v in feats.items():
        a, c = v[missed], v[found]
        if np.nanstd(np.concatenate([a, c])) == 0:
            continue
        try:
            u, pv = stats.mannwhitneyu(a, c, alternative="two-sided")
        except ValueError:
            continue
        res2.append((pv, 2 * u / (len(a) * len(c)) - 1, name, float(np.nanmean(a)), float(np.nanmean(c))))
    res2.sort()
    for pv, rb, name, ma, mc in res2[:8]:
        print(f"    {name:38s} {pv:10.2e} {rb:+8.3f} {ma:10.4f} {mc:12.4f}")

    R = {"n_found": int(found.sum()), "n_missed": int(missed.sum()),
         "n_matched_neg": int(len(matched_neg)), "n_features": m,
         "bonferroni": 0.05 / m,
         "vs_matched_negatives": [{"feature": n, "p": pv, "effect": rb, "missed": ma, "matched_neg": mc}
                                  for pv, rb, n, ma, mc in res[:25]],
         "n_surviving": int(sum(1 for r in res if r[0] < 0.05 / m)),
         "vs_found_circular": [{"feature": n, "p": pv, "effect": rb} for pv, rb, n, _, _ in res2[:15]]}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "missed.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'missed.json'}")


if __name__ == "__main__":
    main()
