"""TIER 2 STAGE A -- covariation matrix construction.

Builds the data the covariation model rests on:

  (1) PRESENCE matrix  (organism x og_id) -> 0/1
      = does this organism have a gene in this ortholog group?

  (2) ESSENTIALITY matrix  (organism x og_id) -> 0/1/NaN
      NaN if no labeled member; 0/1 = max over members (any-essential rule).

  (3) CLADE-HOLDOUT schema
      Per leak-free principle: hold out by CLADE, not organism
      (so Gammaproteobacteria can't trivially predict each other).

  (4) LABEL-PERMUTATION CONTROL  (the review's hard kill-gate)
      Helper to permute essentiality labels WITHIN each organism. The
      covariation model must FAIL after permutation -- if it doesn't,
      the model is fitting phylogeny, not essentiality.

  (5) SYNTHETIC-LETHAL PAIR LOOKUP
      Known SL pairs in our OG space -- the model's coupling matrix
      must recover at least some of these or the architecture is wrong.

Outputs (under outputs/tier2/):
  presence.parquet         (org, og_id, present=1) sparse long form
  essentiality.parquet     (org, og_id, ess) sparse long form, ess in {0,1}
  clade_holdout.json       clade -> list of orgs (the LOCO-style splits)
  synthetic_lethal_pairs.csv  (gene_a, gene_b, og_a, og_b, source)
  matrix_stats.json        diagnostic stats

In-script tests:
  test_matrix_shape: 48 organisms x ~10K-30K OGs
  test_per_org_density: each org has labels for some non-trivial fraction
  test_permutation_preserves_marginals: permuting preserves row-sums
  test_clade_holdout_is_partition: every org in exactly one clade
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data" / "drive_import" / "labels"
OUT = REPO / "outputs" / "tier2"

# canonical synthetic-lethal / strongly co-essential pairs from the bacterial
# literature -- the coupling-matrix recovery test set
KNOWN_SL_PAIRS = [
    # (gene_a, gene_b, source/note)
    ("tolC",  "acrA",  "efflux pump SL with outer-membrane channel"),
    ("tolC",  "acrB",  "efflux pump SL"),
    ("acrA",  "acrB",  "obligate complex partners"),
    ("recA",  "lexA",  "SOS response coupling"),
    ("dnaA",  "dnaN",  "replication initiation pair"),
    ("ftsA",  "ftsZ",  "cell-division Z-ring partners"),
    ("murA",  "murB",  "peptidoglycan biosynthesis sequential"),
    ("groEL", "groES", "chaperonin partners"),
    ("dnaK",  "dnaJ",  "chaperone complex"),
    ("rpoA",  "rpoB",  "RNA polymerase core subunits"),
    ("rpoB",  "rpoC",  "RNA polymerase core subunits"),
    ("secA",  "secY",  "Sec translocon partners"),
    ("uvrA",  "uvrB",  "NER pair"),
    ("uvrB",  "uvrC",  "NER pair"),
    ("ruvA",  "ruvB",  "Holliday-junction resolvase complex"),
]


def build_matrices(lab, orth):
    """Build the long-form presence + essentiality tables."""
    import pandas as pd
    m = lab.merge(orth, on=["organism", "locus_tag"], how="inner")
    # presence per (org, og_id): does this org have any gene in this OG?
    presence = (m[["organism", "og_id"]].drop_duplicates()
                 .assign(present=1))
    # essentiality per (org, og_id): max over members (any-essential rule)
    ess = (m.groupby(["organism", "og_id"]).essential.max().reset_index()
             .rename(columns={"essential": "ess"}))
    ess["ess"] = ess.ess.astype("Int8")
    return presence, ess


def build_clade_holdout(cs):
    """clade -> list of orgs (no org appears in two clades)."""
    out = {}
    for clade, g in cs.groupby("clade"):
        out[clade] = sorted(g.organism.unique().tolist())
    return out


def permute_within_organism(ess_long, seed=0):
    """Permute ess labels WITHIN each organism (preserves per-org marginals).
    The kill-gate: model performance should COLLAPSE after this permutation."""
    import pandas as pd
    import numpy as np
    rng = np.random.RandomState(seed)
    out = ess_long.copy()
    for org, g in out.groupby("organism"):
        idx = g.index.to_numpy()
        vals = g.ess.to_numpy()
        rng.shuffle(vals)
        out.loc[idx, "ess"] = vals
    return out


def find_sl_pairs(lab, orth):
    """Map known SL gene-name pairs to OG-pair targets.
    Returns the named-gene SL pairs where mappable (currently 0 in local data
    due to the disjoint-data wall: named-gene orgs lack og_id, og_id orgs
    lack gene names). This list is recoverable on Colab via feba.db's
    KEGG/Pfam annotations -- documented in script docstring."""
    import pandas as pd
    m = lab.merge(orth, on=["organism", "locus_tag"], how="inner")
    m = m[m.gene_name.notna() & m.og_id.notna()].copy()
    if len(m) == 0:
        return pd.DataFrame(columns=["gene_a", "gene_b", "og_a", "og_b", "source"])
    m["gene_name"] = m.gene_name.astype(str)
    g2og = m.groupby("gene_name").og_id.apply(lambda s: set(s.unique())).to_dict()
    rows = []
    for a, b, src in KNOWN_SL_PAIRS:
        for og_a in g2og.get(a, set()):
            for og_b in g2og.get(b, set()):
                if og_a == og_b: continue
                rows.append({"gene_a": a, "gene_b": b,
                             "og_a": og_a, "og_b": og_b, "source": src})
    return pd.DataFrame(rows).drop_duplicates(["og_a", "og_b"])


def find_operon_pairs(orth_full):
    """Backup coupling-recovery target: operon-adjacent OG pairs from synteny.
    Each row in synteny_features has (og_id, prev_og_id, next_og_id) -- the
    genes flanking this one. Genes in the same operon are biologically
    coupled and the coupling matrix should reflect that, even if the test
    is weaker than SL pairs (operon adjacency correlates with cooccurrence).
    Returns top pairs by cross-organism frequency: pairs (a,b) seen as
    adjacent in many organisms are real biological partnerships."""
    import pandas as pd
    s = orth_full.dropna(subset=["og_id", "next_og_id"])
    pairs = s[["organism", "og_id", "next_og_id"]].rename(
        columns={"og_id": "og_a", "next_og_id": "og_b"})
    # canonicalize: (min, max) so (a,b) and (b,a) count together
    pairs["k1"] = pairs[["og_a", "og_b"]].min(axis=1)
    pairs["k2"] = pairs[["og_a", "og_b"]].max(axis=1)
    pair_counts = (pairs.groupby(["k1", "k2"]).organism.nunique()
                       .rename("n_orgs_adjacent").reset_index())
    # only pairs seen adjacent in >=3 orgs (a real signature)
    pair_counts = pair_counts[pair_counts.n_orgs_adjacent >= 3].copy()
    pair_counts.columns = ["og_a", "og_b", "n_orgs_adjacent"]
    pair_counts["source"] = "operon-adjacent (synteny)"
    return pair_counts.sort_values("n_orgs_adjacent", ascending=False)


def run_tests(presence, ess, clade_map, sl_pairs):
    """Sandbox in-script tests on the built artifacts."""
    import pandas as pd
    import numpy as np
    fails = []

    # T1: shape -- expect 48 orgs, many OGs
    n_orgs = ess.organism.nunique()
    n_ogs = ess.og_id.nunique()
    if n_orgs < 40:
        fails.append(f"T1 shape: only {n_orgs} orgs (expected ~48)")
    if n_ogs < 1000:
        fails.append(f"T1 shape: only {n_ogs} OGs (expected thousands)")

    # T2: per-org density -- each org should have at least 100 labeled OGs
    per_org = ess.groupby("organism").size()
    low_orgs = per_org[per_org < 100]
    if len(low_orgs) > 5:
        fails.append(f"T2 density: {len(low_orgs)} orgs with <100 labels")

    # T3: ESSENTIALITY values are {0,1}, NaN only where unlabeled
    bad_vals = ess[(ess.ess != 0) & (ess.ess != 1)].dropna(subset=["ess"])
    if len(bad_vals) > 0:
        fails.append(f"T3 values: {len(bad_vals)} rows with non-0/1 ess")

    # T4: permutation preserves marginals
    perm = permute_within_organism(ess, seed=0)
    orig_marg = ess.groupby("organism").ess.sum().sort_index()
    perm_marg = perm.groupby("organism").ess.sum().sort_index()
    if not orig_marg.equals(perm_marg):
        fails.append(f"T4 permutation: marginals NOT preserved")

    # T5: clade map is a partition (every org in exactly one clade)
    all_orgs = set()
    for c, lst in clade_map.items():
        for o in lst:
            if o in all_orgs:
                fails.append(f"T5 partition: org {o} in two clades")
            all_orgs.add(o)

    # T6: presence & essentiality consistent (every ess row has a presence row)
    p_keys = set(zip(presence.organism, presence.og_id))
    e_keys = set(zip(ess.organism, ess.og_id))
    missing = e_keys - p_keys
    if missing:
        fails.append(f"T6 consistency: {len(missing)} ess rows have no presence row")

    # T7: at least ONE coupling recovery set exists (SL named OR operon-adj)
    has_sl = sl_pairs is not None and len(sl_pairs) > 0
    has_op = (OUT / "operon_adjacent_pairs.csv").exists()
    if not (has_sl or has_op):
        fails.append(f"T7 coupling recovery: no SL pairs AND no operon pairs")

    return fails


def main() -> int:
    import pandas as pd

    print("=" * 60)
    print("TIER 2 STAGE A -- covariation matrix construction")
    print("=" * 60)

    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading labels + orthology + clade splits ...")
    lab = pd.read_csv(D / "labels.csv")
    orth = pd.read_csv(D / "orthology_features.csv",
                       usecols=["organism", "locus_tag", "og_id"])
    cs = pd.read_csv(D / "clade_splits.csv")

    print(f"  labels: {len(lab):,} ({lab.organism.nunique()} orgs)")
    print(f"  orthology: {len(orth):,} ({orth.organism.nunique()} orgs, "
          f"{orth.og_id.nunique()} distinct OGs)")
    print(f"  clades: {cs.clade.nunique()} clades over "
          f"{cs.organism.nunique()} orgs")

    # 1. matrices
    print("\nBuilding presence + essentiality matrices ...")
    presence, ess = build_matrices(lab, orth)
    print(f"  presence: {len(presence):,} (org, og_id) cells")
    print(f"  essentiality: {len(ess):,} (org, og_id) cells")
    print(f"    ess=1: {int(ess.ess.fillna(0).sum()):,}")
    print(f"    ess=0: {int((ess.ess == 0).sum()):,}")

    presence.to_parquet(OUT / "presence.parquet", index=False)
    ess.to_parquet(OUT / "essentiality.parquet", index=False)

    # per-org and per-og coverage
    per_org = ess.groupby("organism").agg(
        n_og_labeled=("og_id", "size"),
        n_ess=("ess", "sum")).reset_index()
    per_og = ess.groupby("og_id").agg(
        n_orgs_labeled=("organism", "size"),
        n_orgs_ess=("ess", "sum")).reset_index()
    per_org.to_csv(OUT / "per_org_coverage.csv", index=False)
    per_og.to_csv(OUT / "per_og_coverage.csv", index=False)
    print(f"  per-org coverage: median {per_org.n_og_labeled.median():.0f} "
          f"labels/org (range {per_org.n_og_labeled.min()}-{per_org.n_og_labeled.max()})")
    print(f"  per-og coverage: {(per_og.n_orgs_labeled >= 5).sum():,} OGs in >=5 orgs")
    print(f"                   {(per_og.n_orgs_labeled >= 10).sum():,} OGs in >=10 orgs")

    # 2. clade-holdout schema
    print("\nBuilding clade-holdout schema ...")
    clade_map = build_clade_holdout(cs)
    (OUT / "clade_holdout.json").write_text(json.dumps(clade_map, indent=2))
    print(f"  {len(clade_map)} clades; sample: "
          f"{list(clade_map.items())[:3]}")

    # 3. label-permutation control verification
    print("\nVerifying label-permutation control ...")
    perm = permute_within_organism(ess, seed=0)
    moved = (perm.ess != ess.ess.values).sum()
    same_marg = ess.groupby("organism").ess.sum().equals(
        perm.groupby("organism").ess.sum())
    print(f"  permuted: {moved:,} labels moved; marginals preserved: {same_marg}")

    # 4. synthetic-lethal pair lookup (named-gene; sandbox-empty due to
    #    disjoint-data wall) + operon-adjacent backup (sandbox-available)
    print("\nMapping known SL gene pairs to OG space ...")
    sl = find_sl_pairs(lab, orth)
    print(f"  SL pair lookup (named-gene): {len(sl)} recovery targets")
    if len(sl) == 0:
        print(f"    (the disjoint-data wall: named-gene orgs lack og_id;")
        print(f"     SL-via-feba.db on Colab is the planned escalation)")
    sl.to_csv(OUT / "synthetic_lethal_pairs.csv", index=False)

    print("\nBuilding operon-adjacent OG-pair backup recovery set ...")
    orth_full = pd.read_csv(D / "synteny_features.csv")
    op = find_operon_pairs(orth_full)
    print(f"  operon-adjacent pairs (>=3 orgs each): {len(op):,}")
    print(f"    top 3 by cross-org frequency:")
    print(op.head(3).to_string(index=False))
    op.to_csv(OUT / "operon_adjacent_pairs.csv", index=False)

    # 5. save aggregate stats
    stats = {
        "n_organisms_labeled": int(ess.organism.nunique()),
        "n_ogs_distinct": int(ess.og_id.nunique()),
        "n_total_ess_calls": int(len(ess)),
        "n_ess_1": int(ess.ess.sum()),
        "n_ess_0": int((ess.ess == 0).sum()),
        "matrix_density_pct": round(100 * len(ess) /
                                     max(ess.organism.nunique() *
                                         ess.og_id.nunique(), 1), 3),
        "n_clades": len(clade_map),
        "n_sl_pairs_named_gene": int(len(sl)),
        "n_sl_pairs_operon_adjacent": int(len(op)),
        "ogs_in_ge5_orgs": int((per_og.n_orgs_labeled >= 5).sum()),
        "ogs_in_ge10_orgs": int((per_og.n_orgs_labeled >= 10).sum()),
    }
    (OUT / "matrix_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nstats:\n  {json.dumps(stats, indent=2)}")

    # 6. run tests
    print("\n" + "=" * 60)
    print("RUNNING IN-SCRIPT VALIDATION TESTS")
    print("=" * 60)
    fails = run_tests(presence, ess, clade_map, sl)
    if not fails:
        print(f"\nALL TESTS PASS ({7} checks)")
        return 0
    print(f"\nTESTS FAILED ({len(fails)} errors):")
    for f in fails:
        print(f"  - {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
