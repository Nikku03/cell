"""Summarise the human ortholog/function table into a readable markdown report.

Reads the outputs of ``scripts/human_gene_orthologs.py`` and writes
``outputs/human_orthologs/REPORT.md`` plus the highlight tables that make the
result legible: conservation-depth breakdown, per-clade ortholog coverage, and
the human genes whose function is unknown in human but experimentally known in
an ortholog.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = REPO_ROOT / "outputs" / "human_orthologs"

DEPTH_ORDER = ["eukaryote", "eumetazoan", "bilaterian", "deuterostome", "chordate", "vertebrate",
               "jawed_vertebrate", "bony_vertebrate", "lobe_finned_fish", "tetrapod", "amniote",
               "mammal", "primate_only", "human_only"]
DEPTH_GLOSS = {
    "eukaryote": "ortholog in budding yeast — predates animals (~1.5 Gya)",
    "eumetazoan": "reaches cnidarians (~750 Mya)",
    "bilaterian": "ortholog in fly or worm — predates vertebrates (~700 Mya)",
    "deuterostome": "reaches sea urchin (~600 Mya)",
    "chordate": "ortholog in a non-vertebrate chordate (~550 Mya)",
    "vertebrate": "reaches lamprey/hagfish (~550 Mya)",
    "jawed_vertebrate": "reaches sharks and rays (~450 Mya)",
    "bony_vertebrate": "reaches ray-finned fish (~430 Mya)",
    "lobe_finned_fish": "reaches lungfish/coelacanth (~415 Mya)",
    "tetrapod": "reaches amphibians (~350 Mya)",
    "amniote": "reaches birds/reptiles (~320 Mya)",
    "mammal": "mammals only (~180 Mya)",
    "primate_only": "primates only (~65 Mya)",
    "human_only": "no ortholog found in any of the 899 species",
}


def read_table(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def md_table(header: list[str], rows: list[list], aligns: str = "") -> str:
    aligns = aligns or "l" * len(header)
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(sep[a] for a in aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def truncate(text: str, n: int) -> str:
    text = (text or "").replace("|", "/")
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args(argv)
    d = args.dir

    rows = read_table(d / "human_gene_ortholog_function.tsv.gz")
    stats = json.loads((d / "summary.json").read_text())
    n = len(rows)

    # ---- conservation depth ----
    depth = Counter(r["conservation_depth"] for r in rows)
    depth_rows = [[k, f"{depth.get(k, 0):,}", f"{100 * depth.get(k, 0) / n:.1f}%", DEPTH_GLOSS[k]]
                  for k in DEPTH_ORDER if depth.get(k, 0)]

    # ---- annotation status x conservation ----
    status = Counter(r["annotation_status"] for r in rows)
    cross = defaultdict(Counter)
    for r in rows:
        cross[r["annotation_status"]][r["conservation_depth"]] += 1

    # ---- panel species coverage ----
    panel_cols = [c for c in rows[0] if c.startswith("ortholog_")
                  and c not in ("ortholog_evidence_source", "ortholog_inferred_function",
                                "ortholog_novel_go_terms")]
    panel_rows = []
    for c in panel_cols:
        hit = sum(1 for r in rows if r[c])
        panel_rows.append([c.replace("ortholog_", ""), f"{hit:,}", f"{100 * hit / n:.1f}%"])
    panel_rows.sort(key=lambda x: -int(x[1].replace(",", "")))

    # ---- function transfer: dark genes rescued by orthologs ----
    dark = [r for r in rows if r["annotation_status"] != "characterized"]
    rescued = [r for r in dark if int(r["n_ortholog_novel_go"]) > 0]
    strong = [r for r in rescued if int(r["max_species_support"]) >= 3]
    strong.sort(key=lambda r: (-int(r["max_species_support"]), -int(r["n_ortholog_novel_go"])))

    # Most interesting subset: no human function at all, but deeply conserved
    # and experimentally characterised in an invertebrate/fungal ortholog.
    ancient_dark = [
        r for r in rows
        if r["annotation_status"] == "uncharacterized"
        and int(r["n_ortholog_novel_go"]) > 0
        and (r["ortholog_yeast"] or r["ortholog_fly"] or r["ortholog_worm"])
    ]
    ancient_dark.sort(key=lambda r: (-int(r["max_species_support"]), -int(r["n_ortholog_novel_go"])))

    with open(d / "highlight_ancient_dark_genes.tsv", "wt", encoding="utf-8") as fh:
        cols = ["gene_id", "symbol", "name", "conservation_depth", "ortholog_yeast",
                "ortholog_fly", "ortholog_worm", "ortholog_evidence_source",
                "max_species_support", "n_ortholog_novel_go", "ortholog_inferred_function",
                "ortholog_novel_go_terms"]
        fh.write("\t".join(cols) + "\n")
        for r in ancient_dark:
            fh.write("\t".join(str(r[c]).replace("\t", " ") for c in cols) + "\n")

    # ---- ortholog-count outliers ----
    by_species = sorted(rows, key=lambda r: -int(r["n_ortholog_species"]))
    singletons = [r for r in rows if int(r["n_ortholog_species"]) == 0]

    lines = []
    A = lines.append
    A("# Human genes → orthologs → function\n")
    A(f"Generated by `scripts/human_ortholog_report.py` from "
      f"`human_gene_ortholog_function.tsv.gz`.\n")
    A("## Scope\n")
    A(md_table(
        ["quantity", "value"],
        [["human protein-coding genes", f"{stats['human_protein_coding_genes']:,}"],
         ["genes with ≥1 ortholog", f"{stats['genes_with_any_ortholog']:,} "
          f"({100 * stats['genes_with_any_ortholog'] / n:.1f}%)"],
         ["species searched", f"{stats['species_covered']:,}"],
         ["human↔other ortholog pairs", f"{stats['total_ortholog_pairs']:,}"],
         ["median species per gene", f"{stats['median_ortholog_species_per_gene']:,}"],
         ["genes with a yeast ortholog", f"{stats['conserved_to_yeast']:,}"],
         ["genes with a fly ortholog", f"{stats['conserved_to_fly']:,}"],
         ["genes with a worm ortholog", f"{stats['conserved_to_worm']:,}"],
         ["genes with no ortholog anywhere", f"{stats['no_ortholog_anywhere']:,}"]],
        "lr"))
    A("\n## How deep does each gene go?\n")
    A(md_table(["depth", "genes", "share", "meaning"], depth_rows, "lrrl"))
    A("\n## Ortholog coverage in the reference panel\n")
    A(md_table(["species", "human genes with an ortholog", "share"], panel_rows, "lrr"))
    A("\n## What is already known about the human gene\n")
    A(md_table(
        ["annotation status", "genes", "share", "definition"],
        [["characterized", f"{status['characterized']:,}",
          f"{100 * status['characterized'] / n:.1f}%",
          "UniProt FUNCTION text and ≥3 experimental GO terms"],
         ["sparse", f"{status['sparse']:,}", f"{100 * status['sparse'] / n:.1f}%",
          "some curated function, but thin"],
         ["uncharacterized", f"{status['uncharacterized']:,}",
          f"{100 * status['uncharacterized'] / n:.1f}%",
          "no FUNCTION text and no experimental GO term"]],
        "lrrl"))
    A("\n## Function recovered from orthologs\n")
    A(f"Of the **{len(dark):,}** genes that are not fully characterized in human, "
      f"**{len(rescued):,}** ({100 * len(rescued) / max(len(dark), 1):.1f}%) have at least one "
      f"experimentally-evidenced GO term on an ortholog that the human gene does not carry "
      f"experimentally itself. **{len(strong):,}** of those are supported by ≥3 independent "
      f"species, and **{len(ancient_dark):,}** are genes with *no* human functional annotation "
      f"whose fly/worm/yeast ortholog has been studied experimentally.\n")
    A("### Deeply conserved, unknown in human, characterized in an invertebrate/fungal ortholog\n")
    A(md_table(
        ["human gene", "depth", "yeast", "fly", "worm", "species support", "inferred function"],
        [[f"{r['symbol']}", r["conservation_depth"], truncate(r["ortholog_yeast"], 14),
          truncate(r["ortholog_fly"], 14), truncate(r["ortholog_worm"], 14),
          r["max_species_support"], truncate(r["ortholog_inferred_function"], 110)]
         for r in ancient_dark[:40]],
        "llllrrl"))
    A(f"\nFull list: `highlight_ancient_dark_genes.tsv` ({len(ancient_dark):,} genes). "
      f"All rescued genes: `inferred_function_dark_genes.tsv` ({len(rescued):,} genes).\n")
    A("## Extremes\n")
    A("Most broadly conserved genes (present in the most species):\n")
    A(md_table(["gene", "species", "depth", "name"],
               [[r["symbol"], r["n_ortholog_species"], r["conservation_depth"],
                 truncate(r["name"], 60)] for r in by_species[:15]], "lrll"))
    A(f"\n{len(singletons):,} genes have no ortholog in any of the {stats['species_covered']} "
      f"species searched. Sample:\n")
    A(md_table(["gene", "name", "status"],
               [[r["symbol"], truncate(r["name"], 60), r["annotation_status"]]
                for r in singletons[:15]], "lll"))

    # ---- caveats, computed rather than asserted ----
    mt = sum(1 for r in singletons if r["chromosome"] == "MT")
    src = Counter(r["ortholog_evidence_source"] for r in rescued if r["ortholog_evidence_source"])
    shared = sum(v for v in src.values() if v > 1)
    dup_symbols = Counter(r["symbol"] for r in rows)
    n_dup = sum(v - 1 for v in dup_symbols.values() if v > 1)
    A("\n## Caveats\n")
    A(f"- **\"No ortholog\" does not mean human-specific.** NCBI's ortholog set keeps only "
      f"one-to-one relationships, so large multi-copy families (olfactory receptors, keratin-"
      f"associated proteins, BAGE/ANKRD36 expansions) drop out even though relatives exist. "
      f"{mt} of the {len(singletons):,} zero-ortholog genes are mitochondrial-genome genes, "
      f"which the ortholog dump does not cover at all despite being among the most conserved "
      f"genes we have.")
    A(f"- **{n_dup} rows are redundant mitochondrial records.** NCBI carries three GeneIDs for "
      f"each of the 13 mtDNA protein-coding genes (one HGNC-linked, two from alternate "
      f"mitochondrial references). They are kept as-is rather than silently merged, so "
      f"{len(rows):,} rows correspond to {len(rows) - n_dup:,} distinct genes.")
    A("- **Transferred function is a hypothesis, not evidence.** An ortholog's experimental GO "
      "term is the best available prior for what the human gene does, but orthologs "
      "subfunctionalize and neofunctionalize. Every transferred term here is labelled with its "
      "source species and the number of independent species that support it; treat a 1-species "
      "transfer as a lead and a 3+-species transfer as a reasonably safe bet.")
    A(f"- **Recent human paralogs inflate the rescue count.** {shared:,} of the {len(rescued):,} "
      f"rescued genes share their ortholog evidence with at least one human paralog (the 9 "
      f"TAF11L* genes all point at yeast TAF11, the USP17L* cluster all points at fly scny), so "
      f"the number of *independent* functional hypotheses is {len(src):,}, not {len(rescued):,}.")
    A("- **The annotation tiers are thresholds on curation, not on knowledge.** A gene lands in "
      "`sparse` if it has curated function but fewer than 3 experimental GO terms, which catches "
      "well-studied genes that simply have thin GO records (VANGL2, UNC13A). The `uncharacterized` "
      "tier — no FUNCTION text and no experimental GO term at all — is the one to trust as "
      "genuinely dark.")
    A("- **Two ortholog sources, two methods.** NCBI calls are protein-similarity plus synteny "
      "and vertebrate-only; Alliance calls are DIOPT consensus across up to 9 algorithms and "
      "supply fly/worm/yeast. For large ancient families the Alliance stringent filter can call "
      "a family-level relationship rather than a strict 1:1 one (human HBB matches fly `glob1`), "
      "so deep-conservation claims for family members are weaker than for single-copy genes.")

    (d / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {d / 'REPORT.md'}")
    print(f"dark={len(dark):,} rescued={len(rescued):,} strong={len(strong):,} "
          f"ancient_dark={len(ancient_dark):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
