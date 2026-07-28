"""Build a compact, DURABLE kinetics bundle: reaction -> kcat, with provenance, committed to the repo.

WHY THIS EXISTS. The sandbox rolled back and destroyed dlkcat.tsv, ecHumanGEM_kcats_MARjoined.tsv,
ecHumanGEM_kcat_table.tsv and max_KCAT.txt -- every raw kcat source except HumanGEM.xml. What survived did so
because it had been committed. The lesson is the same one the readout taught an hour earlier: scratchpad files
are not storage. This writes the DERIVED table the models actually consume, which is three orders of magnitude
smaller than the sources it came from, and puts it in git.

WHAT GOES IN, and why each tier is where it is. The tiers are ordered by MEASURED accuracy, leave-one-out, so
no source ever saw its own answer. Two label sets, because a fold-error figure without its subset is not a
figure -- this project already made that mistake once, comparing CatPred's n=950 against a null's n=2,437:

                              915 common labels    full human set
                              (every contender)    (leave-one-out)
    tier 1  human-EC median         2.62x              2.80x      <- best, where human measurements exist
    tier 2  CatPred                 8.38x                --       <- 2,549 genes, real signal over the null
    tier 3  any-organism EC           --                 --       <- coverage fallback
    tier 4  global median          14.23x              9.25x      <- the null; still beats dropping the reaction

    NOT ecHumanGEM (66x, bias +1.71) and NOT EC-max (41x, bias +1.61). Both are kcat_MAX -- upper bounds by
    design, not estimators -- and using them as point estimates inflates every ceiling by ~1.7 log units.

Tiers 1 and 3 come from colab/data/ec_kcat_medians.json.gz, reproduced by build_ec_medians.py after the
rollback destroyed the raw table, and verified identical to the original by a derived count rather than by
its download URL. The bundle records which tiers were available at build time rather than silently producing
a thinner table that looks the same.
"""
import collections
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
DEST = Path("colab/data/kinetics_bundle.json.gz")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def reaction_ec(sbml):
    rid, out = None, collections.defaultdict(set)
    pat = re.compile(r'ec-code/([0-9]+\.[0-9]+\.[0-9]+\.[0-9-]+)')
    for line in open(sbml):
        m = re.search(r'<reaction [^>]*\bid="([^"]+)"', line)
        if m:
            rid = m.group(1)
        if rid:
            for ec in pat.findall(line):
                out[rid].add(ec)
        if "</reaction>" in line:
            rid = None
    return out


def gene_rules(sbml):
    """reaction -> list of gene ids, from the fbc geneProductAssociation block."""
    rid, out = None, collections.defaultdict(set)
    gp = re.compile(r'fbc:geneProduct="([^"]+)"')
    for line in open(sbml):
        m = re.search(r'<reaction [^>]*\bid="([^"]+)"', line)
        if m:
            rid = m.group(1)
        if rid:
            for g in gp.findall(line):
                out[rid].add(g)
        if "</reaction>" in line:
            rid = None
    return out


def main():
    sbml = SP / "HumanGEM.xml"
    assert sbml.exists(), f"HumanGEM.xml missing at {sbml} -- cannot build"
    r2ec = reaction_ec(sbml)
    r2g = gene_rules(sbml)
    print(f"Human-GEM: {len(r2ec):,} reactions with an EC, {len(r2g):,} with a gene rule")

    # --- tier 2: CatPred, per gene, from the repo (survived the rollback because it was committed) ---
    kr = json.load(open(OUT / "kinetics_refined.json"))["kinetics_refined"]
    cat = {g: float(v["kcat_per_s"]) for g, v in kr.items()
           if isinstance(v, dict) and v.get("kcat_per_s") and float(v["kcat_per_s"]) > 0}
    km = {g: float(v["km_uM"]) for g, v in kr.items()
          if isinstance(v, dict) and v.get("km_uM") and float(v["km_uM"]) > 0}
    tiers_of = {g: str(v.get("tier", "")) for g, v in kr.items() if isinstance(v, dict)}
    print(f"CatPred layer: {len(cat):,} genes with kcat, {len(km):,} with Km")
    assert len(cat) > 500, "CATPRED LAYER TOO SMALL -- refusing to build"

    # --- tiers 1 and 3: human / any-organism EC medians ---
    # Read the COMMITTED derived file, not the scratchpad raw. build_ec_medians.py reproduced the raw table
    # and verified its identity (human record count == 2,437, the count kcat_headtohead.py measured on), then
    # wrote the 0.03 MB of statistics actually consumed. Reading the committed file is what makes this build
    # survive a rollback; reading scratchpad is what made tiers 1 and 3 vanish the first time.
    ECMED = Path(__file__).resolve().parent / "data" / "ec_kcat_medians.json.gz"
    ec_hum, ec_all, gmed, have_ec = {}, {}, None, ECMED.exists()
    if have_ec:
        em = json.load(gzip.open(ECMED, "rt"))
        ec_hum = em["ec_human_median_per_s"]
        ec_all = em["ec_all_median_per_s"]
        gmed = float(em["global_median_per_s"])
        assert len(ec_hum) > 200 and len(ec_all) > 1000, (
            f"EC MEDIAN FILE TOO THIN (human {len(ec_hum)}, all {len(ec_all)}) -- refusing to build tiers "
            "1/3 from a truncated table")
        print(f"EC medians (committed): human-EC {len(ec_hum):,}, any-organism EC {len(ec_all):,}, "
              f"global median {gmed:.3g} /s")
    else:
        print(f"{ECMED} MISSING -- tiers 1 and 3 unavailable; run build_ec_medians.py. The bundle records "
              "this rather than hiding it behind a thinner table that looks the same.")
        gmed = float(np.median(list(cat.values())))

    # --- assemble reaction -> kcat by the measured hierarchy ---
    e2s = {}
    try:
        from cell_sim import ensembl_to_symbol
        e2s = ensembl_to_symbol()
    except Exception as e:
        print(f"  (ensembl_to_symbol unavailable: {type(e).__name__}; CatPred tier will be keyed by raw id)")

    table, rtier, tier_count = {}, {}, collections.Counter()
    for rid in set(r2ec) | set(r2g):
        ecs = r2ec.get(rid, set())
        v = [ec_hum[e] for e in ecs if e in ec_hum]
        if v:
            table[rid], t = float(np.median(v)), "1_human_EC"
        else:
            cg = [cat[e2s.get(g, g)] for g in r2g.get(rid, ()) if e2s.get(g, g) in cat]
            if cg:
                table[rid], t = float(np.median(cg)), "2_catpred"
            else:
                v2 = [ec_all[e] for e in ecs if e in ec_all]
                if v2:
                    table[rid], t = float(np.median(v2)), "3_any_organism_EC"
                else:
                    table[rid], t = gmed, "4_global_median"
        rtier[rid] = t
        tier_count[t] += 1
    print(f"\nreaction -> kcat for {len(table):,} reactions: {dict(sorted(tier_count.items()))}")
    assert len(table) > 5000, f"KCAT TABLE TOO SMALL ({len(table)}) -- refusing to write"

    bundle = {
        "reaction_kcat_per_s": table,
        # Kept as a SEPARATE map rather than making each value a (kcat, tier) pair: consumers that only want the
        # number stay simple, and a consumer that ignores provenance cannot silently treat a 14.2x global-median
        # fallback as if it were a 2.62x human measurement.
        "reaction_tier": rtier,
        "gene_kcat_per_s": cat,
        "gene_km_uM": km,
        "gene_tier": tiers_of,
        "reaction_genes": {k: sorted(v) for k, v in r2g.items()},
        "reaction_ec": {k: sorted(v) for k, v in r2ec.items()},
        "provenance": {
            # Tier keys carry NO fold-error number: those depend on the label subset, and a number in a
            # key gets quoted without its subset. The figures live here, each with its labels.
            "hierarchy": ["1 human-EC median", "2 CatPred", "3 any-organism EC", "4 global median"],
            "accuracy_common_subset": {"source": "kcat_headtohead.py, 915 labels covered by every contender",
                                       "1_human_EC": "2.62x", "2_catpred": "8.38x",
                                       "4_global_median": "14.23x"},
            "accuracy_full_human_set": {"source": "build_ec_medians.py leave-one-out, 2,337 records",
                                        "1_human_EC": "2.80x", "4_global_median": "9.25x"},
            "excluded": {"ecHumanGEM": "66.3x, bias +1.71 -- kcat_MAX, an upper bound not an estimator",
                         "EC-max": "40.7x, bias +1.61 -- same reason"},
            "ec_medians_available_at_build": bool(have_ec),
            "global_median_kcat_per_s": gmed,
            "tier_counts": dict(tier_count),
        },
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(DEST, "wt") as f:
        json.dump(bundle, f)
    print(f"\n  -> {DEST}  ({DEST.stat().st_size/1e6:.2f} MB)")

    chk = json.load(gzip.open(DEST, "rt"))
    assert len(chk["reaction_kcat_per_s"]) == len(table), "roundtrip lost reactions"
    assert set(chk["reaction_tier"]) == set(chk["reaction_kcat_per_s"]), "every kcat must carry its provenance"
    print(f"  roundtrip OK: {len(chk['reaction_kcat_per_s']):,} reactions, "
          f"{len(chk['gene_kcat_per_s']):,} genes, {len(chk['reaction_genes']):,} gene rules")


if __name__ == "__main__":
    main()
