"""Session 26 v2: synth-lethality pair selection with eligibility filter.

The v1 pilot wasted ~64 % of its compute budget on pairs where at
least one gene was already essential by v15 — synth-lethality is
undefined on those by construction. v2 applies the eligibility
filter at SELECTION time so every sampled pair sits in the
synth-lethal denominator.

Selection rules for v2:

    * pre-filter to the 165 v15-NON-essential genes only
    * 4 categories, each populated only from this pool:

        A2 paralogs                  ESM-2 cosine in [0.70, 1.00]
                                     (relaxed from 0.85; we can
                                     afford it now that essentials
                                     are excluded)
        B2 shared substrate          gene_a's reaction and gene_b's
                                     reaction share at least one
                                     reactant metabolite
        C2 shared product            gene_a's reaction and gene_b's
                                     reaction share at least one
                                     product metabolite
        D2 baseline (random)         no shared metabolite + cosine
                                     below 0.5 (hard negative)

Output:
    outputs/synthlet/pilot_v2_pairs.csv

Usage:
    python scripts/synthlet_pilot_v2_pairs.py
    python scripts/synthlet_pilot_v2_pairs.py --n-paralog 100 ...

Hypothesis being tested:
    H1: paralogs of v15-non-essentials are enriched for synth-
        lethality vs random non-essential pairs of similar count,
        because compensatory function is the most common cause of
        single-gene dispensability paired with joint-knockout
        lethality.
    H2: gene pairs that share a metabolite (substrate or product)
        but are not paralogs are also enriched for synth-lethality,
        because the shared metabolite represents a single point of
        functional redundancy.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cell_sim"))

from cell_sim.layer3_reactions.sbml_parser import (  # noqa: E402
    parse_sbml, sbml_gene_to_locus,
)

ESM_PARQUET = REPO_ROOT / "cell_sim/features/cache/esm2_650M.parquet"
SBML_PATH = REPO_ROOT / (
    "cell_sim/data/Minimal_Cell_ComplexFormation/input_data/"
    "Syn3A_updated.xml"
)
V15_PRED = REPO_ROOT / (
    "outputs/predictions_parallel_s0.05_t0.5_seed42_thr0.1_w4"
    "_composed_all455_v15_round2_priors.csv"
)


def load_v15_nonessentials(path: Path) -> set[str]:
    df = pd.read_csv(path)
    df = df[df["essential"] == 0]
    return set(df["locus_tag"].dropna().astype(str))


def load_sbml_gene_metabolites():
    """Build:
        gene_locus -> set of metabolites (reactant + product) across
        all reactions the gene catalyzes
        gene_locus -> set of reactant-only metabolites
        gene_locus -> set of product-only metabolites
    """
    sbml = parse_sbml(SBML_PATH)
    locus_reactants: dict[str, set[str]] = defaultdict(set)
    locus_products: dict[str, set[str]] = defaultdict(set)
    for rxn in sbml.reactions.values():
        for g in rxn.gene_associations:
            locus = sbml_gene_to_locus(g)
            if not locus:
                continue
            for m in rxn.reactants:
                locus_reactants[locus].add(m)
            for m in rxn.products:
                locus_products[locus].add(m)
    return locus_reactants, locus_products


def cosine_matrix_for_pool(esm: pd.DataFrame, pool: set[str]) -> tuple[
    list[str], np.ndarray,
]:
    """Compute the dense cosine-similarity matrix on the pool only."""
    pool_loci = [g for g in esm.index if g in pool]
    M = esm.loc[pool_loci].values.astype(np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    Mn = M / norms
    sim = Mn @ Mn.T
    return pool_loci, sim


def select_paralogs(
    pool_loci: list[str], sim: np.ndarray,
    n_pairs: int, lo: float, hi: float, seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    n = len(pool_loci)
    cands: list[tuple[float, str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if lo <= s <= hi:
                cands.append((s, pool_loci[i], pool_loci[j]))
    cands.sort(reverse=True)
    if len(cands) > n_pairs * 3:
        # mix top + middle band so we don't just get near-duplicates
        top = cands[: n_pairs // 3]
        mid = cands[n_pairs // 3 : n_pairs * 3]
        rng.shuffle(mid)
        cands = top + mid[: n_pairs - len(top)]
    elif len(cands) > n_pairs:
        cands = cands[:n_pairs]
    rows = []
    for s, a, b in cands[:n_pairs]:
        rows.append({
            "locus_a": a, "locus_b": b, "category": "A2_paralog",
            "biological_rationale": (
                f"ESM-2 cosine {s:.3f} (v15-nonessential pool)"
            ),
        })
    return rows


def select_shared_metabolite(
    pool: set[str],
    locus_to_mets: dict[str, set[str]],
    n_pairs: int, label: str,
    rationale_template: str,
    seed: int,
) -> list[dict]:
    """Pairs whose annotated metabolite sets share at least one
    member. Exact mechanism (substrate vs product) is selected by
    which mapping ``locus_to_mets`` carries."""
    rng = random.Random(seed)
    pool_with_mets = [g for g in pool if locus_to_mets.get(g)]
    cands: list[tuple[str, str, str]] = []
    for i, a in enumerate(pool_with_mets):
        ma = locus_to_mets.get(a, set())
        if not ma:
            continue
        for b in pool_with_mets[i + 1 :]:
            mb = locus_to_mets.get(b, set())
            shared = ma & mb
            if shared:
                # take any one shared metabolite for the rationale
                m_example = sorted(shared)[0]
                cands.append((a, b, m_example))
    rng.shuffle(cands)
    rows = []
    for a, b, m in cands[:n_pairs]:
        rows.append({
            "locus_a": a, "locus_b": b, "category": label,
            "biological_rationale": rationale_template.format(metabolite=m),
        })
    return rows


def select_random_baseline(
    pool: set[str],
    locus_reactants: dict[str, set[str]],
    locus_products: dict[str, set[str]],
    pool_loci: list[str], sim: np.ndarray,
    n_pairs: int, seed: int,
    cos_max: float = 0.70,
) -> list[dict]:
    """Pairs with no shared metabolite AND ESM-2 cosine < cos_max.

    A negative-control sample drawn from the same eligible pool. The
    cosine cap defaults to 0.70 — sits below the v2 paralog band and
    well below the typical mean cosine in ESM-2 across non-essential
    Syn3A genes (~0.55-0.65), so we exclude near-paralogs but don't
    over-restrict.
    """
    rng = random.Random(seed)
    pool_list = sorted(pool)
    locus_to_idx = {g: i for i, g in enumerate(pool_loci)}
    rows = []
    tried = 0
    seen: set[tuple[str, str]] = set()
    while len(rows) < n_pairs and tried < n_pairs * 100:
        tried += 1
        a, b = rng.sample(pool_list, 2)
        if a == b:
            continue
        if a > b:
            a, b = b, a
        if (a, b) in seen:
            continue
        seen.add((a, b))
        # No shared metabolite (functional disjointness in SBML sense)
        ma = locus_reactants.get(a, set()) | locus_products.get(a, set())
        mb = locus_reactants.get(b, set()) | locus_products.get(b, set())
        if ma & mb:
            continue
        # Cosine < cos_max (when both genes have ESM-2 entries)
        i = locus_to_idx.get(a); j = locus_to_idx.get(b)
        if i is not None and j is not None:
            if float(sim[i, j]) >= cos_max:
                continue
        rows.append({
            "locus_a": a, "locus_b": b, "category": "D2_random_baseline",
            "biological_rationale": (
                f"negative control: no shared SBML metabolite, "
                f"ESM-2 cosine < {cos_max:.2f}"
            ),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "outputs/synthlet/pilot_v2_pairs.csv",
    )
    # Calibrated cosine bands for the v15-non-essential pool (n=165):
    # mean cos 0.930, std 0.038, 10th pct 0.876, 90th pct 0.969.
    # Paralog signal sits at cos >= 0.97 (top ~10 %); negative control
    # is the bottom decile cos < 0.88 (~10 %).
    ap.add_argument("--n-paralog-tight", type=int, default=24)   # cos >= 0.99
    ap.add_argument("--n-paralog-loose", type=int, default=60)   # cos 0.97-0.99
    ap.add_argument("--n-shared-substrate", type=int, default=60)
    ap.add_argument("--n-shared-product", type=int, default=50)
    ap.add_argument("--n-random", type=int, default=80)
    ap.add_argument("--baseline-cos-max", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("[load] v15 non-essentials + SBML metabolites + ESM-2 ...")
    nonessentials = load_v15_nonessentials(V15_PRED)
    print(f"  v15 non-essentials: {len(nonessentials)}")

    locus_reactants, locus_products = load_sbml_gene_metabolites()
    print(f"  SBML gene-loci with metabolites: "
          f"reactants={len(locus_reactants)}  products={len(locus_products)}")

    esm = pd.read_parquet(ESM_PARQUET)
    pool_loci, sim = cosine_matrix_for_pool(esm, nonessentials)
    print(f"  ESM-2 pool restricted to non-essentials: "
          f"{len(pool_loci)} loci")

    rows: list[dict] = []

    print(f"\n[A2 tight] paralogs cos in [0.99, 1.00] ...")
    a_tight = select_paralogs(
        pool_loci, sim, args.n_paralog_tight, 0.99, 1.00, args.seed,
    )
    for r in a_tight:
        r["category"] = "A2_paralog_tight"
    print(f"  -> {len(a_tight)} pairs")
    rows.extend(a_tight)

    print(f"\n[A2 loose] paralogs cos in [0.97, 0.99) ...")
    a_loose = select_paralogs(
        pool_loci, sim, args.n_paralog_loose, 0.97, 0.99, args.seed + 10,
    )
    for r in a_loose:
        r["category"] = "A2_paralog_loose"
    print(f"  -> {len(a_loose)} pairs")
    rows.extend(a_loose)

    print(f"\n[B2] shared substrate ...")
    b_rows = select_shared_metabolite(
        nonessentials, locus_reactants,
        args.n_shared_substrate, "B2_shared_substrate",
        "shared SBML reactant metabolite ({metabolite}); "
        "v15-nonessential pool",
        args.seed + 1,
    )
    print(f"  -> {len(b_rows)} pairs")
    rows.extend(b_rows)

    print(f"\n[C2] shared product ...")
    c_rows = select_shared_metabolite(
        nonessentials, locus_products,
        args.n_shared_product, "C2_shared_product",
        "shared SBML product metabolite ({metabolite}); "
        "v15-nonessential pool",
        args.seed + 2,
    )
    print(f"  -> {len(c_rows)} pairs")
    rows.extend(c_rows)

    print(f"\n[D2] random baseline (no shared met, cos < "
          f"{args.baseline_cos_max:.2f}) ...")
    d_rows = select_random_baseline(
        nonessentials, locus_reactants, locus_products,
        pool_loci, sim, args.n_random, args.seed + 3,
        cos_max=args.baseline_cos_max,
    )
    print(f"  -> {len(d_rows)} pairs")
    rows.extend(d_rows)

    df = pd.DataFrame(rows).drop_duplicates(subset=["locus_a", "locus_b"])
    print(f"\ntotal unique pairs after dedup: {len(df)}")
    print(df.groupby("category").size().to_string())

    # Sanity: every pair must be in the v15-nonessential set
    leak = (
        ~df["locus_a"].isin(nonessentials)
        | ~df["locus_b"].isin(nonessentials)
    ).sum()
    if leak:
        print(f"[FAIL] {leak} pairs leaked an essential gene; aborting")
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
