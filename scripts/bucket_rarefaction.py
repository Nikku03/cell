"""Essential-bucket rarefaction analysis.

Naresh's question: take all our 59 bacterial essentiality datasets, sort
each essential gene into a functional bucket, and see whether the number
of distinct buckets PLATEAUS as we add more organisms or KEEPS GROWING.

If it plateaus -> there is a finite "essentialome" function space, and
the bucket method has a well-defined finite target.
If it keeps growing -> essential functions are largely organism-specific
and the bucket method has a moving target across organisms.

GRANULARITY of "bucket":
  We use ortholog group (og_id) as the bucket. Two genes in different
  organisms but sharing an og_id are likely doing the same thing -- this
  is the natural fine-grained functional bucket and the one our existing
  features (family_frac, og_cpd) already operate on. og_id ~= KEGG ortholog
  in granularity.

METHOD:
  1. Join essentiality labels with og_id per gene.
  2. Rarefaction: random orderings of the 59 organisms (50 seeds), for
     each ordering count cumulative distinct OGs of essential genes as we
     add organism k=1..59. Mean + 25-75 percentile band.
  3. Core/accessory split: how many OGs are essential in >=K organisms?
  4. Saturation diagnostics:
       - slope at the end (OGs gained per organism for orgs 50-59)
       - extrapolated asymptote (Chao1-style estimator from singletons)
       - fraction of essential OGs seen in only ONE organism (rarefaction
         tail = open-ended; tail-heavy = no plateau)

Output:
  outputs/essential_bucket_rarefaction.md  -- the curve + diagnostics
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data" / "drive_import" / "labels"


def main() -> int:
    import pandas as pd
    import numpy as np

    print("Loading labels + orthology ...")
    lab = pd.read_csv(D / "labels.csv")[["organism", "locus_tag", "essential"]]
    orth = pd.read_csv(D / "orthology_features.csv",
                       usecols=["organism", "locus_tag", "og_id",
                                "family_n_organisms"])
    ess = lab[lab.essential == 1].merge(orth, on=["organism", "locus_tag"],
                                        how="left")
    ess = ess.dropna(subset=["og_id"])
    print(f"  essentials with og_id: {len(ess):,} over "
          f"{ess.organism.nunique()} organisms, "
          f"{ess.og_id.nunique()} unique og_ids")

    orgs = sorted(ess.organism.unique())
    n_org = len(orgs)
    # essential OGs per organism (set)
    ogs_per_org = {o: set(g.og_id) for o, g in ess.groupby("organism")}

    # === rarefaction over multiple random orderings ===
    print(f"\nRarefaction: 50 seeds, n_orgs={n_org} ...")
    rng = np.random.RandomState(0)
    curves = np.zeros((50, n_org))
    for s in range(50):
        order = list(orgs)
        rng.shuffle(order)
        seen = set()
        for k, o in enumerate(order):
            seen |= ogs_per_org[o]
            curves[s, k] = len(seen)

    mean = curves.mean(0)
    p25 = np.percentile(curves, 25, axis=0)
    p75 = np.percentile(curves, 75, axis=0)

    # === core/accessory: OG essentiality breadth ===
    breadth = (ess.drop_duplicates(["organism", "og_id"])
                  .groupby("og_id").organism.nunique())
    print(f"\nEssentiality breadth distribution (how many orgs an OG is "
          f"essential in):")
    print(f"  N orgs:        1     2     3     5    10    20    30    40")
    cnts = [(breadth >= k).sum() for k in (1, 2, 3, 5, 10, 20, 30, 40)]
    print(f"  OGs >= N:  " + "  ".join(f"{c:>5d}" for c in cnts))
    singleton_frac = float((breadth == 1).mean())
    print(f"  singleton OGs (essential in only 1 org): "
          f"{(breadth==1).sum():,} ({singleton_frac*100:.0f}%)")

    # === slope at the end + Chao1 ===
    tail_slope = mean[-1] - mean[-10]   # OGs gained over last 10 orgs added
    tail_per_org = tail_slope / 10.0
    singletons = (breadth == 1).sum()
    doubletons = (breadth == 2).sum()
    # Chao1 asymptote: observed + singletons^2 / (2 * doubletons)
    chao1 = mean[-1] + (singletons ** 2) / max(2 * doubletons, 1)
    fraction_observed = mean[-1] / chao1

    # === report ===
    out_lines = ["# Essential-bucket rarefaction across 59 bacterial datasets\n"]
    out_lines += [
        f"**Data:** {len(ess):,} essential gene labels with og_id assigned, "
        f"over {n_org} organisms, {int(mean[-1]):,} distinct essential OGs.\n",
        "## Cumulative essential-OG count vs organisms added\n",
        "Median + 25/75-percentile band over 50 random orderings.\n",
        "| n_orgs | mean OGs | p25 | p75 |",
        "|--------|---------:|----:|----:|"]
    for k in [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, n_org]:
        if k <= n_org:
            i = k - 1
            out_lines.append(f"| {k:>3d} | {mean[i]:>8.0f} | "
                             f"{p25[i]:>4.0f} | {p75[i]:>4.0f} |")

    out_lines += [
        "\n## Saturation diagnostics\n",
        f"- **Observed essential OGs at n={n_org}:** {int(mean[-1]):,}",
        f"- **Slope over last 10 orgs added:** "
        f"+{tail_slope:.0f} OGs total, **{tail_per_org:.0f} per org**",
        f"- **Chao1 estimated asymptote:** {chao1:.0f} OGs "
        f"(observed = {fraction_observed*100:.0f}% of estimated total)",
        f"- **Singleton OGs (essential in only 1 org):** "
        f"{singletons:,} ({singleton_frac*100:.0f}% of all essential OGs)\n",
        "## Core vs accessory essentialome\n",
        "| breadth (essential in N orgs) | n OGs | fraction |",
        "|---|---:|---:|"]
    total = int(mean[-1])
    for k in (1, 2, 3, 5, 10, 20, 30, 40):
        c = int((breadth >= k).sum())
        out_lines.append(f"| ≥ {k:>2d} | {c:>5d} | {c/total*100:>5.1f}% |")

    out_lines += [
        "\n## Verdict\n",
        f"- Tail slope **{tail_per_org:.0f} new essential OGs per organism** "
        f"added at the end.",
        f"- Chao1 says we've seen **{fraction_observed*100:.0f}%** of the "
        f"true essential-OG space.",
        f"- **{singleton_frac*100:.0f}% of essential OGs are essential in "
        f"only one organism** -- the open-ended tail.",
    ]
    if tail_per_org < 20 and fraction_observed > 0.85:
        out_lines.append("- **The curve is approaching saturation.** "
                         "The core essentialome is bounded; the bucket "
                         "method has a well-defined finite target.")
    elif tail_per_org < 50:
        out_lines.append("- **Slowing but NOT saturated.** Substantial "
                         "long tail of organism-specific essentials; the "
                         "core is bounded but the accessory is open.")
    else:
        out_lines.append("- **No plateau yet.** Each new organism still "
                         "contributes meaningful new essential OGs; "
                         "essentialome may be much larger than observed.")

    text = "\n".join(out_lines)
    print("\n" + text)
    out = REPO / "outputs" / "essential_bucket_rarefaction.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
