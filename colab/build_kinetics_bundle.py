"""Build a compact, DURABLE kinetics bundle: reaction -> kcat, with provenance, committed to the repo.

WHY THIS EXISTS. The sandbox rolled back and destroyed dlkcat.tsv, ecHumanGEM_kcats_MARjoined.tsv,
ecHumanGEM_kcat_table.tsv and max_KCAT.txt -- every raw kcat source except HumanGEM.xml. What survived did so
because it had been committed. The lesson is the same one the readout taught an hour earlier: scratchpad files
are not storage. This writes the DERIVED table the models actually consume, which is three orders of magnitude
smaller than the sources it came from, and puts it in git.

WHAT GOES IN, and why each tier is where it is. The tiers are ordered by MEASURED accuracy from
kcat_headtohead.py -- 915 common labels, leave-one-out, so no source saw its own answer:

    tier 1  human-EC median      2.62x median fold error   <- best, but only where human measurements exist
    tier 2  CatPred              8.38x                     <- 2,549 genes, real signal over the null
    tier 3  any-organism EC      (coverage fallback)
    tier 4  global median       14.23x                     <- the null; better than dropping the reaction

    NOT ecHumanGEM (66x, bias +1.71) and NOT EC-max (41x, bias +1.61). Both are kcat_MAX -- upper bounds by
    design, not estimators -- and using them as point estimates inflates every ceiling by ~1.7 log units.

Tiers 1 and 3 need dlkcat.tsv, which is currently missing. The bundle records which tiers were available at
build time rather than silently producing a thinner table that looks the same.
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

    # --- tiers 1 and 3: human / any-organism EC medians, only if dlkcat survived ---
    ec_hum, ec_all, gmed, have_dlkcat = {}, {}, None, (SP / "dlkcat.tsv").exists()
    if have_dlkcat:
        recs = json.load(open(SP / "dlkcat.tsv"))
        hum, allo = collections.defaultdict(list), collections.defaultdict(list)
        for r in recs:
            ec = (r.get("ECNumber") or "").strip()
            try:
                v = float(r.get("Value"))
            except (TypeError, ValueError):
                continue
            if not ec or v <= 0 or not np.isfinite(v):
                continue
            allo[ec].append(v)
            if r.get("Organism") == "Homo sapiens":
                hum[ec].append(v)
        ec_hum = {e: float(np.median(v)) for e, v in hum.items()}
        ec_all = {e: float(np.median(v)) for e, v in allo.items()}
        gmed = float(np.median([x for v in hum.values() for x in v]))
        print(f"dlkcat: human-EC {len(ec_hum):,}, any-organism EC {len(ec_all):,}, global median {gmed:.3g}")
    else:
        print("dlkcat.tsv MISSING -- tiers 1 and 3 unavailable; bundle records this rather than hiding it")
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
            table[rid], t = float(np.median(v)), "1_human_EC_2.62x"
        else:
            cg = [cat[e2s.get(g, g)] for g in r2g.get(rid, ()) if e2s.get(g, g) in cat]
            if cg:
                table[rid], t = float(np.median(cg)), "2_catpred_8.38x"
            else:
                v2 = [ec_all[e] for e in ecs if e in ec_all]
                if v2:
                    table[rid], t = float(np.median(v2)), "3_any_organism_EC"
                else:
                    table[rid], t = gmed, "4_global_median_14.2x"
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
            "hierarchy": ["1 human-EC median (2.62x)", "2 CatPred (8.38x)", "3 any-organism EC",
                          "4 global median (14.2x)"],
            "accuracy_source": "kcat_headtohead.py, 915 common labels, leave-one-out",
            "excluded": {"ecHumanGEM": "66.3x, bias +1.71 -- kcat_MAX, an upper bound not an estimator",
                         "EC-max": "40.7x, bias +1.61 -- same reason"},
            "dlkcat_available_at_build": bool(have_dlkcat),
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
