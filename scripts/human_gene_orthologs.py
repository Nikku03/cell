"""Build an ortholog + function table for every human protein-coding gene.

For each human gene this answers three questions:

  1. Which other species have an ortholog of it?  (NCBI ``gene_orthologs``
     covers 895 vertebrates; the Alliance/DIOPT combined file adds fly,
     worm, yeast and the two Xenopus species.)
  2. How deep does that conservation go?  A gene found in yeast is a very
     different object from one found only in great apes, so every gene gets
     a ``conservation_depth`` rank derived from the NCBI taxonomy lineage of
     the species its orthologs sit in.
  3. What does it do -- and if the *human* gene has no curated function,
     what do its orthologs say it does?

Point 3 is the reason this script exists. A few thousand human genes have no
UniProt FUNCTION text and no experimentally-evidenced GO term, but their
mouse/zebrafish/fly/worm/yeast orthologs do. Those experimental GO terms are
transferable evidence, so the script emits them per gene together with the
species that supplied them and how many independent species agree.

Everything is a single streaming pass per input file: the two big NCBI dumps
(``All_Data.gene_info`` 1.5 GB, ``gene2go`` 1.3 GB gzipped) are never held in
memory, only the rows whose GeneID we actually need.

Inputs are staged by ``scripts/fetch_ortholog_data.sh`` into
``data_cache/human_orthologs/`` (~3 GB, gitignored).

Usage::

    python scripts/fetch_ortholog_data.sh      # once, ~4 min
    python scripts/human_gene_orthologs.py     # ~10 min, writes outputs/human_orthologs/
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO_ROOT / "data_cache" / "human_orthologs"
DEFAULT_OUT = REPO_ROOT / "outputs" / "human_orthologs"

HUMAN_TAXID = 9606

# ---------------------------------------------------------------------------
# Reference panel: species we report named orthologs for. The NCBI ortholog
# set is vertebrate-only, so fly/worm/yeast/X. laevis come from the Alliance.
# ---------------------------------------------------------------------------
PANEL_NCBI = [
    (9598, "chimpanzee"),
    (9544, "rhesus_macaque"),
    (10090, "mouse"),
    (10116, "rat"),
    (9615, "dog"),
    (9913, "cow"),
    (9823, "pig"),
    (13616, "opossum"),
    (9258, "platypus"),
    (9031, "chicken"),
    (28377, "anole_lizard"),
    (8364, "xenopus_tropicalis"),
    (7955, "zebrafish"),
    (31033, "fugu"),
    (7868, "elephant_shark"),
    (7757, "sea_lamprey"),
]
PANEL_ALLIANCE = [
    (8355, "xenopus_laevis"),
    (7227, "fly"),
    (6239, "worm"),
    (559292, "yeast"),
]
PANEL_ORDER = [name for _, name in PANEL_NCBI] + [name for _, name in PANEL_ALLIANCE]
PANEL_NCBI_TAXIDS = {t for t, _ in PANEL_NCBI}
# Species whose gene_info we read in full so we can map Alliance
# (FlyBase/WormBase/SGD/Xenbase) identifiers onto NCBI GeneIDs.
XREF_TAXIDS = {t for t, _ in PANEL_ALLIANCE}

# Clade buckets. Assignment walks up the NCBI lineage from the species and
# takes the *nearest* matching ancestor, so mammals hit Mammalia long before
# they reach Sarcopterygii. That makes 8287 (Sarcopterygii) safe as the
# lobe-finned bucket: only coelacanth and lungfish reach it.
CLADE_RULES = [
    (9443, "primate"),
    (9989, "rodent"),
    (40674, "other_mammal"),
    (8782, "bird"),
    (8457, "reptile"),
    (8292, "amphibian"),
    (8287, "lobe_finned_fish"),
    (7898, "ray_finned_fish"),
    (7777, "cartilaginous_fish"),
    (1476529, "jawless_fish"),
    (7711, "other_chordate"),
    (7586, "echinoderm"),
    (33317, "protostome"),
    (6073, "cnidarian"),
    (4751, "fungus"),
]

# Conservation depth: rank -> label, assigned from the *deepest* clade the
# gene still has an ortholog in. Ranks follow divergence time from human, so
# lungfish (~415 Mya) sits nearer than ray-finned fish (~430 Mya) despite
# "fish" being the intuitive grouping for both.
DEPTH_BY_CLADE = {
    "primate": (1, "primate_only"),
    "rodent": (2, "mammal"),
    "other_mammal": (2, "mammal"),
    "bird": (3, "amniote"),
    "reptile": (3, "amniote"),
    "amphibian": (4, "tetrapod"),
    "lobe_finned_fish": (5, "lobe_finned_fish"),
    "ray_finned_fish": (6, "bony_vertebrate"),
    "cartilaginous_fish": (7, "jawed_vertebrate"),
    "jawless_fish": (8, "vertebrate"),
    "other_chordate": (9, "chordate"),
    "echinoderm": (10, "deuterostome"),
    "protostome": (11, "bilaterian"),
    "cnidarian": (12, "eumetazoan"),
    "fungus": (13, "eukaryote"),
}

# GO evidence codes that reflect a real experiment, as opposed to a
# computational guess or an annotation already transferred from an ortholog.
# See http://geneontology.org/docs/guide-go-evidence-codes/
EXPERIMENTAL_EVIDENCE = {
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP",   # small-scale experiment
    "HTP", "HDA", "HMP", "HGI", "HEP",          # high-throughput experiment
}

ECO_BRACES = re.compile(r"\s*\{ECO:[^}]*\}")
PUBMED_REF = re.compile(r"\s*\(PubMed:[^)]*\)")
UNCHARACTERIZED_NAME = re.compile(
    r"uncharacterized|putative uncharacterized|hypothetical protein", re.I
)
# Symbols that carry no functional meaning on their own.
PLACEHOLDER_SYMBOL = re.compile(r"^(LOC\d+|C\d+orf\d+|CXorf\d+|CYorf\d+|.*-AS\d*|LINC\d+)$", re.I)


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def opener(path: Path):
    """gzip-aware line iterator."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Stage 1 -- taxonomy
# ---------------------------------------------------------------------------
def load_taxonomy(cache: Path, needed: set[int], t0: float):
    """Return (clade_of_taxid, scientific_name_of_taxid) for `needed` taxids."""
    parent: dict[int, int] = {}
    with open(cache / "nodes.dmp", "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.split("\t|\t", 2)
            parent[int(f[0])] = int(f[1])
    log(f"taxonomy: {len(parent):,} nodes", t0)

    clade: dict[int, str] = {}
    for taxid in needed:
        node, seen = taxid, 0
        label = "other"
        while node and node != 1 and seen < 100:
            for anc, name in CLADE_RULES:
                if node == anc:
                    label = name
                    break
            else:
                node = parent.get(node, 1)
                seen += 1
                continue
            break
        clade[taxid] = label

    names: dict[int, str] = {}
    with open(cache / "names.dmp", "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "scientific name" not in line:
                continue
            f = line.split("\t|\t")
            taxid = int(f[0])
            if taxid in needed:
                names[taxid] = f[1]
    log(f"taxonomy: resolved {len(clade):,} species to clades", t0)
    return clade, names


# ---------------------------------------------------------------------------
# Stage 2 -- human gene universe
# ---------------------------------------------------------------------------
def load_human_genes(cache: Path, t0: float):
    genes: dict[int, dict] = {}
    hgnc_to_gid: dict[str, int] = {}
    with opener(cache / "Homo_sapiens.gene_info.gz") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[9] != "protein-coding":
                continue
            gid = int(f[1])
            hgnc = ensembl = ""
            for xref in f[5].split("|"):
                if xref.startswith("HGNC:"):
                    hgnc = xref.split(":", 1)[1]
                elif xref.startswith("Ensembl:"):
                    ensembl = xref.split(":", 1)[1]
            genes[gid] = {
                "gene_id": gid,
                "symbol": f[2],
                "name": f[8],
                "chromosome": f[6],
                "map_location": f[7],
                "hgnc_id": hgnc,
                "ensembl_id": ensembl,
            }
            if hgnc:
                hgnc_to_gid[hgnc] = gid
    log(f"human: {len(genes):,} protein-coding genes", t0)
    return genes, hgnc_to_gid


# ---------------------------------------------------------------------------
# Stage 3 -- NCBI orthologs (streaming group-by on the human GeneID block)
# ---------------------------------------------------------------------------
def scan_orthologs(cache: Path, human_ids: set[int], t0: float):
    """One pass over gene_orthologs.gz.

    Returns per-human-gene aggregates plus the set of panel ortholog GeneIDs
    whose annotation we need to pull later.
    """
    species_seen: set[int] = set()
    species_gene_counts: dict[int, int] = defaultdict(int)
    per_gene: dict[int, dict] = {}
    panel_ids: set[int] = set()
    rows = 0

    def flush(gid: int, pairs: list[tuple[int, int]]) -> None:
        if gid not in human_ids or not pairs:
            return
        by_species: dict[int, list[int]] = defaultdict(list)
        for other_tax, other_gid in pairs:
            by_species[other_tax].append(other_gid)
        rec = per_gene.setdefault(gid, {"species": set(), "n_ortholog_genes": 0, "panel": {}})
        for taxid in PANEL_NCBI_TAXIDS & by_species.keys():
            rec["panel"].setdefault(taxid, []).extend(by_species[taxid])
            panel_ids.update(by_species[taxid])
        rec["species"].update(by_species)
        rec["n_ortholog_genes"] += sum(len(v) for v in by_species.values())
        for taxid in by_species:
            species_gene_counts[taxid] += 1
        species_seen.update(by_species)

    with opener(cache / "gene_orthologs.gz") as fh:
        current_gid = -1
        buf: list[tuple[int, int]] = []
        for line in fh:
            if line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            tax, gid, other_tax, other_gid = int(f[0]), int(f[1]), int(f[3]), int(f[4])
            if tax != HUMAN_TAXID:
                # The file also stores a handful of pairs with human second.
                if other_tax == HUMAN_TAXID and other_gid in human_ids:
                    rec = per_gene.setdefault(
                        other_gid, {"species": set(), "n_ortholog_genes": 0, "panel": {}}
                    )
                    if tax not in rec["species"]:
                        species_gene_counts[tax] += 1
                    rec["species"].add(tax)
                    rec["n_ortholog_genes"] += 1
                    if tax in PANEL_NCBI_TAXIDS:
                        rec["panel"].setdefault(tax, []).append(gid)
                        panel_ids.add(gid)
                    species_seen.add(tax)
                continue
            rows += 1
            if gid != current_gid:
                flush(current_gid, buf)
                current_gid, buf = gid, []
            buf.append((other_tax, other_gid))
        flush(current_gid, buf)

    log(
        f"orthologs: {rows:,} human pairs -> {len(per_gene):,} genes, "
        f"{len(species_seen)} species, {len(panel_ids):,} panel ortholog genes",
        t0,
    )
    return per_gene, species_seen, panel_ids, dict(species_gene_counts)


# ---------------------------------------------------------------------------
# Stage 4 -- gene_info for ortholog genes (+ xref maps for Alliance IDs)
# ---------------------------------------------------------------------------
def scan_gene_info(cache: Path, panel_ids: set[int], t0: float):
    """Symbols/descriptions for panel orthologs, and xref -> GeneID maps."""
    info: dict[int, tuple[str, str]] = {}
    xref_to_gid: dict[str, int] = {}
    seen = 0
    with opener(cache / "All_Data.gene_info.gz") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            seen += 1
            if seen % 20_000_000 == 0:
                log(f"gene_info: {seen:,} rows scanned", t0)
            tab1 = line.find("\t")
            taxid = int(line[:tab1])
            want_xref = taxid in XREF_TAXIDS
            if not want_xref and taxid not in PANEL_NCBI_TAXIDS:
                continue
            f = line.rstrip("\n").split("\t")
            gid = int(f[1])
            if want_xref:
                for xref in f[5].split("|"):
                    if xref.startswith(("FLYBASE:", "WormBase:", "SGD:", "Xenbase:")):
                        db, acc = xref.split(":", 1)
                        xref_to_gid[acc] = gid
                info[gid] = (f[2], f[8])
            elif gid in panel_ids:
                info[gid] = (f[2], f[8])
    log(f"gene_info: {len(info):,} ortholog records, {len(xref_to_gid):,} xrefs", t0)
    return info, xref_to_gid


# ---------------------------------------------------------------------------
# Stage 5 -- Alliance/DIOPT orthologs (fly, worm, yeast, Xenopus)
# ---------------------------------------------------------------------------
def scan_alliance(cache: Path, hgnc_to_gid: dict[str, int], xref_to_gid: dict[str, int], t0: float):
    """human GeneID -> {taxid: [(symbol, ncbi_gid|None, n_algorithms), ...]}."""
    wanted = {t for t, _ in PANEL_ALLIANCE}
    out: dict[int, dict[int, list[tuple[str, int | None, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    kept = 0
    with opener(cache / "alliance_orthology.tsv.gz") as fh:
        for line in fh:
            if line.startswith(("#", "Gene1ID")):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 11 or not f[2].endswith(":9606"):
                continue
            taxid = int(f[6].rsplit(":", 1)[1])
            if taxid not in wanted:
                continue
            # gene_info stores dbXref "HGNC:HGNC:13265" -> key "HGNC:13265",
            # which is exactly the form the Alliance uses in Gene1ID.
            human_gid = hgnc_to_gid.get(f[0])
            if human_gid is None:
                continue
            acc = f[4].split(":", 1)[1] if ":" in f[4] else f[4]
            out[human_gid][taxid].append((f[5], xref_to_gid.get(acc), int(f[9])))
            kept += 1
    log(f"alliance: {kept:,} human ortholog pairs over {len(out):,} human genes", t0)
    return out


# ---------------------------------------------------------------------------
# Stage 6 -- GO annotation
# ---------------------------------------------------------------------------
def scan_gene2go(cache: Path, needed: set[int], t0: float):
    """GeneID -> {'exp': {GO: (term, category)}, 'all': set(GO)}."""
    go: dict[int, dict] = {}
    seen = 0
    with opener(cache / "gene2go.gz") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            seen += 1
            if seen % 20_000_000 == 0:
                log(f"gene2go: {seen:,} rows scanned", t0)
            f = line.rstrip("\n").split("\t")
            gid = int(f[1])
            if gid not in needed:
                continue
            rec = go.setdefault(gid, {"exp": {}, "all": set()})
            rec["all"].add(f[2])
            if f[3] in EXPERIMENTAL_EVIDENCE and not f[4].startswith("NOT"):
                rec["exp"][f[2]] = (f[5], f[7])
    log(f"gene2go: annotation for {len(go):,} genes of interest", t0)
    return go


# ---------------------------------------------------------------------------
# Stage 7 -- UniProt human function text
# ---------------------------------------------------------------------------
def load_uniprot(cache: Path, t0: float):
    by_gid: dict[int, dict] = {}
    with open(cache / "uniprot_human.tsv", "rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 8 or not f[7]:
                continue
            function = ECO_BRACES.sub("", f[3]).strip()
            if function.startswith("FUNCTION: "):
                function = function[len("FUNCTION: "):]
            try:
                score = float(f[5].split()[0]) if f[5] else 0.0
            except ValueError:
                score = 0.0
            for raw in f[7].split(";"):
                raw = raw.strip()
                if not raw:
                    continue
                gid = int(raw)
                prev = by_gid.get(gid)
                # One gene can map to several accessions; keep the best-annotated.
                if prev is None or len(function) > len(prev["function"]):
                    by_gid[gid] = {
                        "accession": f[0],
                        "protein_name": f[2],
                        "function": function,
                        "annotation_score": score,
                        "protein_existence": f[6],
                    }
    log(f"uniprot: {len(by_gid):,} human genes with a reviewed entry", t0)
    return by_gid


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def summarize_sentence(text: str, limit: int = 300) -> str:
    text = PUBMED_REF.sub("", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    dot = cut.rfind(". ")
    return (cut[: dot + 1] if dot > 80 else cut.rstrip() + "...").strip()


def build(cache: Path, out_dir: Path, t0: float) -> dict:
    human, hgnc_to_gid = load_human_genes(cache, t0)
    human_ids = set(human)

    ortho, species_seen, panel_ids, species_counts = scan_orthologs(cache, human_ids, t0)
    clade, sci_name = load_taxonomy(cache, species_seen | XREF_TAXIDS | PANEL_NCBI_TAXIDS, t0)
    info, xref_to_gid = scan_gene_info(cache, panel_ids, t0)
    alliance = scan_alliance(cache, hgnc_to_gid, xref_to_gid, t0)

    # GO is needed for human genes and for every ortholog we might quote.
    alliance_gids = {
        gid for per in alliance.values() for lst in per.values() for _, gid, _ in lst if gid
    }
    go = scan_gene2go(cache, human_ids | panel_ids | alliance_gids, t0)
    uniprot = load_uniprot(cache, t0)

    taxid_to_panel = {t: n for t, n in PANEL_NCBI} | {t: n for t, n in PANEL_ALLIANCE}
    rows: list[dict] = []
    pair_rows: list[tuple] = []

    for gid, g in sorted(human.items(), key=lambda kv: kv[1]["symbol"]):
        rec = ortho.get(gid)
        species = rec["species"] if rec else set()
        clade_counts: dict[str, int] = defaultdict(int)
        for taxid in species:
            clade_counts[clade.get(taxid, "other")] += 1

        # Named orthologs, per panel species.
        panel_syms: dict[str, list[str]] = {}
        ortholog_gids: dict[str, list[int]] = {}
        if rec:
            for taxid, gids in rec["panel"].items():
                name = taxid_to_panel[taxid]
                syms = [info.get(og, (f"GeneID:{og}", ""))[0] for og in gids]
                panel_syms[name] = syms
                ortholog_gids[name] = gids
        for taxid, lst in alliance.get(gid, {}).items():
            name = taxid_to_panel[taxid]
            panel_syms[name] = [s for s, _, _ in lst]
            ortholog_gids[name] = [og for _, og, _ in lst if og]
            clade_counts[clade.get(taxid, "other")] += 1
            species = species | {taxid}

        depth_rank, depth_label = 0, "human_only"
        for cl in clade_counts:
            r = DEPTH_BY_CLADE.get(cl)
            if r and r[0] > depth_rank:
                depth_rank, depth_label = r

        # ---- what do we already know about the human gene? ----
        up = uniprot.get(gid, {})
        hgo = go.get(gid, {"exp": {}, "all": set()})
        human_function = up.get("function", "")
        n_exp_human = len(hgo["exp"])
        has_function = bool(human_function) or n_exp_human > 0
        placeholder = bool(PLACEHOLDER_SYMBOL.match(g["symbol"])) or bool(
            UNCHARACTERIZED_NAME.search(up.get("protein_name", "") or g["name"])
        )
        if human_function and n_exp_human >= 3:
            status = "characterized"
        elif has_function:
            status = "sparse"
        else:
            status = "uncharacterized"

        # ---- what do the orthologs say? ----
        # Experimental GO terms found in orthologs but not experimentally
        # attached to the human gene: this is the transferable evidence.
        term_species: dict[str, set[str]] = defaultdict(set)
        term_label: dict[str, tuple[str, str]] = {}
        best_src = None
        for sp_name, gids in ortholog_gids.items():
            for og in gids:
                og_go = go.get(og)
                if not og_go or not og_go["exp"]:
                    continue
                for term, (label, cat) in og_go["exp"].items():
                    term_species[term].add(sp_name)
                    term_label[term] = (label, cat)
                cand = (len(og_go["exp"]), sp_name, og)
                if best_src is None or cand[0] > best_src[0]:
                    best_src = cand
        novel = {t: sp for t, sp in term_species.items() if t not in hgo["exp"]}
        ranked = sorted(novel.items(), key=lambda kv: (-len(kv[1]), kv[0]))

        inferred_function = inferred_source = ""
        inferred_terms = ""
        max_support = 0
        if ranked:
            max_support = len(ranked[0][1])
            inferred_terms = "; ".join(
                f"{t}:{term_label[t][0]} [{term_label[t][1]}|{len(sp)}sp]" for t, sp in ranked[:12]
            )
        if best_src:
            _, sp_name, og = best_src
            sym, desc = info.get(og, ("", ""))
            inferred_source = f"{sp_name}:{sym}"
            bp = [term_label[t][0] for t, _ in ranked if term_label[t][1] == "Process"][:4]
            mf = [term_label[t][0] for t, _ in ranked if term_label[t][1] == "Function"][:3]
            parts = []
            if desc:
                parts.append(desc)
            if mf:
                parts.append("molecular function: " + ", ".join(mf))
            if bp:
                parts.append("process: " + ", ".join(bp))
            inferred_function = " | ".join(parts)

        rows.append({
            **{k: g[k] for k in ("gene_id", "symbol", "name", "chromosome", "hgnc_id", "ensembl_id")},
            "uniprot": up.get("accession", ""),
            "n_ortholog_species": len(species),
            "n_ortholog_genes": (rec["n_ortholog_genes"] if rec else 0)
            + sum(len(v) for v in alliance.get(gid, {}).values()),
            "conservation_depth": depth_label,
            "conservation_rank": depth_rank,
            **{f"n_{c}": clade_counts.get(c, 0) for _, c in CLADE_RULES},
            **{f"ortholog_{n}": ",".join(panel_syms.get(n, [])) for n in PANEL_ORDER},
            "annotation_status": status,
            "placeholder_name": int(placeholder),
            "human_go_exp": n_exp_human,
            "human_go_total": len(hgo["all"]),
            "uniprot_annotation_score": up.get("annotation_score", 0.0),
            "protein_existence": up.get("protein_existence", ""),
            "human_function": summarize_sentence(human_function, 600),
            "ortholog_evidence_source": inferred_source,
            "ortholog_inferred_function": summarize_sentence(inferred_function, 400),
            "ortholog_novel_go_terms": inferred_terms,
            "n_ortholog_novel_go": len(novel),
            "max_species_support": max_support,
        })

        for sp_name in PANEL_ORDER:
            for i, sym in enumerate(panel_syms.get(sp_name, [])):
                og = ortholog_gids.get(sp_name, [])
                og_id = og[i] if i < len(og) else ""
                pair_rows.append(
                    (gid, g["symbol"], sp_name, sym, og_id,
                     len(go.get(og_id, {}).get("exp", {})) if og_id else 0)
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv_gz(out_dir / "human_gene_ortholog_function.tsv.gz", rows)
    write_pairs(out_dir / "human_ortholog_pairs_panel.tsv.gz", pair_rows)
    stats = write_reports(out_dir, rows, species_seen, sci_name, clade, species_counts)
    log(f"wrote {len(rows):,} gene rows and {len(pair_rows):,} ortholog pairs", t0)
    return stats


def write_tsv_gz(path: Path, rows: list[dict]) -> None:
    cols = list(rows[0])
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]).replace("\t", " ").replace("\n", " ") for c in cols) + "\n")


def write_pairs(path: Path, pairs: list[tuple]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        fh.write("human_gene_id\thuman_symbol\tspecies\tortholog_symbol\tortholog_gene_id\tortholog_experimental_go\n")
        for p in pairs:
            fh.write("\t".join(str(x) for x in p) + "\n")


def write_reports(out_dir: Path, rows: list[dict], species_seen, sci_name, clade,
                  species_counts: dict[int, int]) -> dict:
    n = len(rows)
    with open(out_dir / "species_coverage.tsv", "wt", encoding="utf-8") as fh:
        fh.write("tax_id\tscientific_name\tclade\thuman_genes_with_ortholog\n")
        for taxid, cnt in sorted(species_counts.items(), key=lambda kv: -kv[1]):
            fh.write(f"{taxid}\t{sci_name.get(taxid, '')}\t{clade.get(taxid, 'other')}\t{cnt}\n")

    by_status = defaultdict(int)
    by_depth = defaultdict(int)
    for r in rows:
        by_status[r["annotation_status"]] += 1
        by_depth[r["conservation_depth"]] += 1

    rescued = [
        r for r in rows
        if r["annotation_status"] != "characterized" and r["n_ortholog_novel_go"] > 0
    ]
    rescued.sort(key=lambda r: (-r["max_species_support"], -r["n_ortholog_novel_go"]))
    dark_total = by_status["uncharacterized"] + by_status["sparse"]

    cols = ["gene_id", "symbol", "name", "conservation_depth", "n_ortholog_species",
            "annotation_status", "human_go_exp", "ortholog_evidence_source",
            "max_species_support", "n_ortholog_novel_go", "ortholog_inferred_function",
            "ortholog_novel_go_terms"]
    with open(out_dir / "inferred_function_dark_genes.tsv", "wt", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rescued:
            fh.write("\t".join(str(r[c]).replace("\t", " ") for c in cols) + "\n")

    stats = {
        "human_protein_coding_genes": n,
        "genes_with_any_ortholog": sum(1 for r in rows if r["n_ortholog_species"] > 0),
        "species_covered": len(species_seen) + len(PANEL_ALLIANCE),
        "total_ortholog_pairs": sum(r["n_ortholog_genes"] for r in rows),
        "annotation_status": dict(by_status),
        "conservation_depth": dict(by_depth),
        "genes_needing_function": dark_total,
        "genes_rescued_by_orthologs": len(rescued),
        "rescue_rate": round(len(rescued) / dark_total, 4) if dark_total else 0.0,
        "rescued_multi_species_support": sum(1 for r in rescued if r["max_species_support"] >= 2),
        "median_ortholog_species_per_gene": sorted(r["n_ortholog_species"] for r in rows)[n // 2],
        "conserved_to_yeast": sum(1 for r in rows if r["ortholog_yeast"]),
        "conserved_to_fly": sum(1 for r in rows if r["ortholog_fly"]),
        "conserved_to_worm": sum(1 for r in rows if r["ortholog_worm"]),
        "no_ortholog_anywhere": sum(1 for r in rows if r["n_ortholog_species"] == 0),
    }
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    t0 = time.time()
    missing = [
        f for f in ("gene_orthologs.gz", "Homo_sapiens.gene_info.gz", "All_Data.gene_info.gz",
                    "gene2go.gz", "alliance_orthology.tsv.gz", "uniprot_human.tsv",
                    "nodes.dmp", "names.dmp")
        if not (args.cache / f).exists()
    ]
    if missing:
        print(f"missing inputs in {args.cache}: {missing}\nrun scripts/fetch_ortholog_data.sh first",
              file=sys.stderr)
        return 1

    stats = build(args.cache, args.out, t0)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
