"""Summarise the deep-homology search for the 235 dark human genes.

Reads the outputs of ``scripts/deep_homology_dark_genes.py`` and writes
``outputs/human_orthologs/DEEP_HOMOLOGY.md``: how far outside the vertebrate
ortholog set each dark gene reaches, which ones land in bacteria or archaea,
and which of those survive the reciprocal-best-hit test.
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

CLADE_GLOSS = {
    "bacteria": "Bacteria",
    "archaea": "Archaea",
    "excavata": "Excavata (Giardia, Trypanosoma)",
    "apicomplexa": "Apicomplexa (Plasmodium)",
    "amoebozoa": "Amoebozoa (Dictyostelium)",
    "plant": "Plants (Arabidopsis)",
    "alga": "Green algae (Chlamydomonas)",
    "fungi_other": "Fission yeast (S. pombe)",
    "choanoflagellate": "Choanoflagellates (Monosiga)",
    "porifera": "Sponges (Amphimedon)",
    "placozoa": "Placozoa (Trichoplax)",
    "cnidaria": "Cnidaria (Nematostella)",
    "echinoderm": "Echinoderms (sea urchin)",
    "cephalochordate": "Cephalochordates (amphioxus)",
    "tunicate": "Tunicates (Ciona)",
    "none": "no hit anywhere in the panel",
}


def md_table(header: list[str], rows: list[list], aligns: str = "") -> str:
    aligns = aligns or "l" * len(header)
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(sep[a] for a in aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def truncate(t: str, n: int) -> str:
    t = (t or "").replace("|", "/")
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def clean_desc(d: str) -> str:
    """UniProt FASTA descriptions carry OS=/OX=/GN= tails we do not need."""
    for tag in (" OS=", " OX=", " GN=", " PE=", " SV="):
        i = d.find(tag)
        if i > 0:
            d = d[:i]
    return d.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args(argv)
    d = args.dir

    genes = list(csv.DictReader(open(d / "dark_gene_deep_homology.tsv"), delimiter="\t"))
    stats = json.loads((d / "deep_homology_summary.json").read_text())
    with gzip.open(d / "dark_gene_deep_homology_hits.tsv.gz", "rt") as fh:
        hits = list(csv.DictReader(fh, delimiter="\t"))
    n = len(genes)

    depth = Counter(g["deepest_domain"] for g in genes)
    prok = [g for g in genes if g["reaches_prokaryote"] == "1"]
    prok_rbh = [g for g in prok if g["prokaryote_reciprocal_best_hit"] == "1"]
    prok.sort(key=lambda g: float(g["best_prokaryote_evalue"] or 1))

    # per-species prokaryote coverage
    prok_hits = [h for h in hits if h["clade"] in ("bacteria", "archaea")]
    sp = defaultdict(set)
    for h in prok_hits:
        sp[h["species"]].add(h["symbol"])

    # Method attribution has to be counted on distinct gene-target pairs: the
    # two passes overlap heavily, so summing hit rows per method double-counts.
    pair_methods, gene_methods = defaultdict(set), defaultdict(set)
    for h in prok_hits:
        pair_methods[(h["symbol"], h["target"])].add(h["method"])
        gene_methods[h["symbol"]].add(h["method"])
    pair_profile_only = sum(1 for v in pair_methods.values() if v == {"profile"})
    pair_both = sum(1 for v in pair_methods.values() if len(v) == 2)
    pair_phmmer_only = sum(1 for v in pair_methods.values() if v == {"phmmer"})
    gene_profile_only = sum(1 for v in gene_methods.values() if v == {"profile"})

    # Reciprocal verdict on each gene's single best prokaryotic hit.
    best: dict[str, tuple[float, dict]] = {}
    for h in prok_hits:
        e = float(h["evalue"])
        if h["symbol"] not in best or e < best[h["symbol"]][0]:
            best[h["symbol"]] = (e, h)
    rev_self = sum(1 for _, h in best.values() if h["reciprocal_best_hit"] == "1")
    rev_other = sum(1 for _, h in best.values() if h["reciprocal_best_hit"] != "1"
                    and h["reciprocal_top_human"] not in ("", "not_tested"))
    rev_none = sum(1 for _, h in best.values() if h["reciprocal_top_human"] in ("", "not_tested"))

    lines = []
    A = lines.append
    A("# Dark genes: how far out do they reach?\n")
    A("The 235 genes in `highlight_ancient_dark_genes.tsv` have no curated human "
      "function but a characterized fly/worm/yeast ortholog. Neither ortholog source used "
      "upstream can see past those three species, so this is a direct sequence search against "
      f"{stats['panel_species']} species that appear in neither: bacteria, archaea, plants, "
      "algae, protists, and the basal metazoans and invertebrate chordates that NCBI's "
      "vertebrate-only set skips.\n")
    A(md_table(["quantity", "value"],
               [["dark genes searched", f"{stats['dark_genes_searched']}"],
                ["panel species", f"{stats['panel_species']}"],
                ["panel proteins searched", f"{stats['panel_proteins']:,}"],
                ["genes with ≥1 homolog in the panel", f"{stats['genes_with_any_hit']}"],
                ["genes with no hit anywhere", f"{stats['genes_with_no_hit']}"],
                ["**genes reaching bacteria or archaea**",
                 f"**{stats['genes_reaching_prokaryotes']}**"],
                ["↳ surviving the reciprocal-best-hit test",
                 f"{stats['genes_reaching_prokaryotes_rbh']}"],
                ["genes reaching bacteria", f"{stats['genes_reaching_bacteria']}"],
                ["genes reaching archaea", f"{stats['genes_reaching_archaea']}"],
                ["total hit rows", f"{stats['total_hit_rows']:,}"]], "lr"))

    A("\n## Deepest domain reached\n")
    A(md_table(["deepest group", "genes", "share"],
               [[CLADE_GLOSS.get(k, k), v, f"{100 * v / n:.1f}%"]
                for k, v in sorted(depth.items(), key=lambda kv: -kv[1])], "lrr"))

    A("\n## Genes that reach bacteria or archaea\n")
    A("`RBH` marks a reciprocal best hit: searching the prokaryotic protein back against the "
      "reviewed human proteome returns this same gene on top. The `reverse hit` column shows "
      "what it returns instead when it is not this gene — and that column is the most "
      "informative one in the table.\n")
    A(f"For {rev_other} of the {len(prok)} genes, the reverse search lands on a *different human "
      f"gene*, and it is almost always the query's better-studied paralog: ABCF2-H2BK1's "
      f"bacterial hit maps back to ABCF1, ATAD3C's maps to VCP, ATP13A5's to ATP2C1, MGAM2's to "
      f"MGAM. The prokaryotic homology is real and often overwhelming (E-values to 1e-206), but "
      f"the human family member that best represents it is not the dark gene. Read that as *the "
      f"family is ancient, and this gene is a young duplicate of it* — not as evidence against "
      f"the homology. Only {rev_self} genes have their best prokaryotic hit come back as a "
      f"reciprocal best hit; {len(prok_rbh)} have at least one RBH hit somewhere in their hit "
      f"list. For {rev_none} genes the reverse search found no human hit at all, which means "
      f"the match exists only at profile level and is the weakest evidence in the table.\n")
    A(md_table(
        ["human gene", "best prokaryotic hit", "species", "E-value", "found by", "RBH",
         "reverse hit"],
        [[g["symbol"], truncate(clean_desc(g["best_prokaryote_hit"]), 44),
          truncate(g["best_prokaryote_species"], 24), g["best_prokaryote_evalue"],
          g["prokaryote_method"], "**yes**" if g["prokaryote_reciprocal_best_hit"] == "1" else "—",
          truncate(best[g["symbol"]][1]["reciprocal_top_human"].replace("not_tested", "—") or "—", 14)]
         for g in prok],
        "lllllll"))

    A("\n## Prokaryotic coverage by species\n")
    A(md_table(["species", "dark genes with a homolog"],
               [[s, len(v)] for s, v in sorted(sp.items(), key=lambda kv: -len(kv[1]))], "lr"))
    A(f"\nMethod attribution, counted on the {len(pair_methods):,} distinct gene–protein pairs "
      f"rather than on hit rows (the two passes overlap, so summing rows double-counts): "
      f"{pair_both:,} pairs were found by both passes, {pair_profile_only:,} by the profile "
      f"search alone, and {pair_phmmer_only} by direct search alone. At gene level, "
      f"{gene_profile_only} of the {len(prok)} genes that reach prokaryotes would have been "
      f"missed entirely without the profile stage — which is the point of building one.\n")

    A("## Caveats\n")
    A("- **Homology is not orthology.** Most prokaryotic hits here are the human protein "
      "landing in an ancient, widely-shared family (P-loop NTPases, ABC transporter ATPase "
      "domains, Rossmann folds). Rows without an RBH should be read as \"this gene contains an "
      "ancient domain\", not \"this gene has a bacterial ortholog\".")
    A("- **But RBH is conservative in exactly the wrong direction here.** These are dark genes, "
      "and a large share of them are recent duplicates or readthrough products of well-studied "
      "human genes. When a bacterial protein's best human match is the parent rather than the "
      "duplicate, RBH fails even though the ancestry is genuine. A failed RBH on a young "
      "paralog is not evidence against ancient origin; it is evidence that the query is not "
      "the family's representative member. The 7 RBH genes are the safest calls, not the "
      "complete set of true ones.")
    A("- **Multidomain proteins can pass RBH on one domain.** DIP2C is 1,500+ residues and its "
      "hit is to an adenylyltransferase; the reciprocal verdict reflects that domain, not the "
      "whole protein. Check `query_coverage` in the hits file before reading a whole-protein "
      "claim into a domain-level match.")
    A("- **A profile search trades specificity for reach.** Profiles built from a gene's "
      "eukaryotic homologs find real remote homology that single-sequence search misses, and "
      "they also drift toward generic domain models when the seed set is large and divergent. "
      "Seed counts are in the `profile_seeds` column of `dark_gene_deep_homology.tsv`.")
    A("- **The panel is a sample, not a census.** 8 bacteria and 4 archaea stand in for two "
      "entire domains of life. A gene with no hit here may still have homologs in lineages "
      "not sampled; absence of evidence is weak evidence at this depth.")
    A("- **Eukaryotic parasites have reduced genomes.** Giardia, Plasmodium and Trypanosoma "
      "have lost many ancestral genes, so a missing hit in those species reflects their "
      "biology as much as the human gene's age.")

    (d / "DEEP_HOMOLOGY.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {d / 'DEEP_HOMOLOGY.md'}")
    print(f"prokaryote-reaching={len(prok)} rbh={len(prok_rbh)} no_hit={stats['genes_with_no_hit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
