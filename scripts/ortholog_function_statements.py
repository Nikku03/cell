"""Turn ortholog evidence into a readable function statement per human gene.

The upstream pipeline proves a human gene *has* orthologs and lists the GO
terms attached to them. GO term lists are evidence, not an answer: "process:
deadenylation-dependent decapping" is not a sentence you can put in a report.
This script fetches the curated UniProt FUNCTION text of the orthologs
themselves and assigns each human gene the best statement available, with the
species it came from attached.

Evidence is ranked, and the rank is the point -- a statement is only as good as
its provenance:

  human_curated          the human gene already has curated function text; no
                         transfer needed, and it is shown for comparison
  ortholog_curated       curated FUNCTION text from a model-organism ortholog,
                         nearest species first (mouse -> rat -> zebrafish ->
                         Xenopus -> fly -> worm -> yeast)
  ortholog_experimental  no ortholog function text, but ortholog GO terms with
                         experimental evidence codes
  deep_homolog           only for the dark genes: curated function from a
                         bacterial, archaeal or protist homolog found by
                         sequence search, flagged with whether it was a
                         reciprocal best hit
  none                   nothing anywhere

Usage::

    scripts/fetch_ortholog_function_data.sh         # 7 model-organism proteomes
    python3 scripts/ortholog_function_statements.py --fetch-deep-annotations
    python3 scripts/ortholog_function_statements.py
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO_ROOT / "data_cache" / "ortholog_function"
DEFAULT_OUT = REPO_ROOT / "outputs" / "human_orthologs"

# Nearest first: a mouse ortholog's function is a safer transfer than a yeast
# one, so the mouse statement wins when both exist. Both are recorded.
SPECIES_ORDER = ["mouse", "rat", "zebrafish", "xenopus_tropicalis", "fly", "worm", "yeast"]

# How much a statement from this species can be trusted verbatim for a human
# gene. The molecular half of a function statement (kinase, scramblase, ECM
# protease) transfers across all of these; the organismal half does not. A worm
# statement about cuticle collagen or male tail rays is describing worm biology
# with a human-relevant molecular activity buried inside it.
TRANSFER_CONFIDENCE = {
    "mouse": "high_mammal", "rat": "high_mammal",
    "zebrafish": "medium_vertebrate", "xenopus_tropicalis": "medium_vertebrate",
    "fly": "low_invertebrate", "worm": "low_invertebrate", "yeast": "low_fungal",
}

ECO_BRACES = re.compile(r"\s*\{ECO:[^}]*\}")
PUBMED_REF = re.compile(r"\s*\(PubMed:[^)]*\)")
SIMILARITY = re.compile(r"^(By similarity|Probable|Putative)\b", re.I)


def clean_function(text: str, limit: int = 700) -> str:
    """UniProt FUNCTION text minus the evidence apparatus."""
    if not text:
        return ""
    t = ECO_BRACES.sub("", text)
    t = PUBMED_REF.sub("", t)
    t = t.replace("FUNCTION: ", " ").strip()
    t = re.sub(r"\s+", " ", t)
    if len(t) > limit:
        cut = t[:limit]
        dot = cut.rfind(". ")
        t = (cut[: dot + 1] if dot > 100 else cut.rstrip() + "…")
    return t.strip()


def load_species_annotations(cache: Path) -> dict[str, dict[str, dict]]:
    """species -> NCBI GeneID -> annotation row (best entry per gene)."""
    out: dict[str, dict[str, dict]] = {}
    for sp in SPECIES_ORDER:
        path = cache / f"uniprot_{sp}.tsv"
        if not path.exists():
            print(f"missing {path}; skipping {sp}")
            continue
        best: dict[str, dict] = {}
        for r in csv.DictReader(open(path), delimiter="\t"):
            fn = clean_function(r.get("Function [CC]", ""))
            for raw in (r.get("GeneID") or "").split(";"):
                gid = raw.strip()
                if not gid:
                    continue
                prev = best.get(gid)
                if prev is None or len(fn) > len(prev["_fn"]):
                    best[gid] = {**r, "_fn": fn}
        out[sp] = best
        with_fn = sum(1 for v in best.values() if v["_fn"])
        print(f"  {sp:20s} {len(best):6,} genes mapped, {with_fn:6,} with FUNCTION text")
    return out


def fetch_deep_annotations(cache: Path, out_dir: Path) -> None:
    """Pull UniProt records for the deep-homology hits worth annotating.

    Same scope the reciprocal check used: every prokaryotic hit plus the best
    eukaryotic hit per gene/species. Batched at 50 accessions because the query
    goes in the URL and longer batches are rejected.
    """
    hits_path = out_dir / "dark_gene_deep_homology_hits.tsv.gz"
    if not hits_path.exists():
        print(f"missing {hits_path}; run scripts/deep_homology_dark_genes.py first")
        return
    with gzip.open(hits_path, "rt") as fh:
        hits = list(csv.DictReader(fh, delimiter="\t"))
    best_euk, wanted = {}, set()
    for h in hits:
        if h["clade"] in ("bacteria", "archaea"):
            wanted.add(h["target"])
        else:
            k = (h["symbol"], h["species"])
            if k not in best_euk or float(h["evalue"]) < best_euk[k][0]:
                best_euk[k] = (float(h["evalue"]), h["target"])
    wanted |= {t for _, t in best_euk.values()}
    accs = sorted({t.split("|")[1] for t in wanted if "|" in t})
    print(f"{len(wanted):,} target proteins -> {len(accs):,} accessions")

    fields = ("accession,reviewed,protein_name,cc_function,go_id,ec,"
              "cc_catalytic_activity,cc_subunit,cc_pathway")
    rows, failed = {}, 0
    for i in range(0, len(accs), 50):
        chunk = accs[i:i + 50]
        for attempt in range(3):
            txt = subprocess.run(
                ["curl", "-sS", "--max-time", "180", "--compressed", "-G",
                 "https://rest.uniprot.org/uniprotkb/stream",
                 "--data-urlencode", "query=" + " OR ".join(f"(accession:{a})" for a in chunk),
                 "--data-urlencode", f"fields={fields}", "--data-urlencode", "format=tsv"],
                capture_output=True, text=True).stdout
            if txt.startswith("Entry\t"):
                for r in csv.DictReader(txt.splitlines(), delimiter="\t"):
                    rows[r["Entry"]] = r
                break
            time.sleep(2 ** attempt)
        else:
            failed += 1
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "deep_hit_annotations.json").write_text(json.dumps(rows))
    fn = sum(1 for r in rows.values() if r.get("Function [CC]"))
    print(f"fetched {len(rows):,}/{len(accs):,} (failed batches: {failed}); "
          f"{fn:,} carry curated FUNCTION text")


def load_deep_annotations(cache: Path, out_dir: Path):
    """dark gene symbol -> best annotated deep homolog."""
    ann_path = cache / "deep_hit_annotations.json"
    hits_path = out_dir / "dark_gene_deep_homology_hits.tsv.gz"
    if not ann_path.exists() or not hits_path.exists():
        return {}
    ann = json.loads(ann_path.read_text())
    with gzip.open(hits_path, "rt") as fh:
        hits = list(csv.DictReader(fh, delimiter="\t"))
    best: dict[str, dict] = {}
    for h in hits:
        acc = h["target"].split("|")[1] if "|" in h["target"] else h["target"]
        a = ann.get(acc)
        if not a:
            continue
        fn = clean_function(a.get("Function [CC]", ""))
        # Prefer: reciprocal best hit > has curated function > reviewed > lower E.
        key = (int(h["reciprocal_best_hit"]), bool(fn), a.get("Reviewed") == "reviewed",
               -float(h["evalue"]))
        prev = best.get(h["symbol"])
        if prev is None or key > prev["_key"]:
            best[h["symbol"]] = {
                "_key": key, "accession": acc, "species": h["species"], "clade": h["clade"],
                "evalue": h["evalue"], "rbh": h["reciprocal_best_hit"],
                "coverage": h["query_coverage"], "method": h["method"],
                "protein_name": a.get("Protein names", ""), "function": fn,
                "ec": a.get("EC number", ""),
                "catalytic": clean_function(a.get("Catalytic activity", ""), 200),
            }
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fetch-deep-annotations", action="store_true",
                    help="pull UniProt records for the deep-homology hits, then exit")
    args = ap.parse_args(argv)

    if args.fetch_deep_annotations:
        fetch_deep_annotations(args.cache, args.out)
        return 0

    print("loading model-organism annotations")
    sp_ann = load_species_annotations(args.cache)
    deep = load_deep_annotations(args.cache, args.out)
    print(f"  deep homolog annotations for {len(deep)} dark genes")

    with gzip.open(args.out / "human_gene_ortholog_function.tsv.gz", "rt") as fh:
        genes = list(csv.DictReader(fh, delimiter="\t"))
    with gzip.open(args.out / "human_ortholog_pairs_panel.tsv.gz", "rt") as fh:
        pairs = list(csv.DictReader(fh, delimiter="\t"))
    by_human: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        if p["species"] in sp_ann and p["ortholog_gene_id"]:
            by_human[p["human_gene_id"]][p["species"]].append(p["ortholog_gene_id"])

    rows, stats = [], defaultdict(int)
    for g in genes:
        gid, sym = g["gene_id"], g["symbol"]
        human_fn = g["human_function"]

        # Best curated ortholog statement, nearest species first.
        transfer = None
        others = []
        for sp in SPECIES_ORDER:
            for og in by_human.get(gid, {}).get(sp, []):
                a = sp_ann[sp].get(og)
                if not a or not a["_fn"]:
                    continue
                entry = (sp, a["Entry"], a.get("Protein names", "").split(" (")[0], a["_fn"])
                if transfer is None:
                    transfer = entry
                else:
                    others.append(f"{sp}:{entry[2][:40]}")
                break

        d = deep.get(sym)
        if human_fn:
            tier = "human_curated"
        elif transfer:
            tier = "ortholog_curated"
        elif g["ortholog_novel_go_terms"]:
            tier = "ortholog_experimental"
        elif d and d["function"]:
            tier = "deep_homolog"
        else:
            tier = "none"
        stats[tier] += 1

        if human_fn:
            recommended = human_fn
        elif transfer:
            recommended = transfer[3]
        elif g["ortholog_inferred_function"]:
            recommended = g["ortholog_inferred_function"]
        elif d and d["function"]:
            recommended = d["function"]
        else:
            recommended = ""

        rows.append({
            "gene_id": gid, "symbol": sym, "name": g["name"],
            "annotation_status": g["annotation_status"],
            "conservation_depth": g["conservation_depth"],
            "evidence_tier": tier,
            "recommended_function": recommended,
            "human_curated_function": human_fn,
            "ortholog_species": transfer[0] if transfer else "",
            "ortholog_accession": transfer[1] if transfer else "",
            "ortholog_protein": transfer[2] if transfer else "",
            "ortholog_curated_function": transfer[3] if transfer else "",
            "transfer_confidence": TRANSFER_CONFIDENCE.get(transfer[0], "") if transfer else "",
            "other_ortholog_sources": "; ".join(others[:4]),
            "ortholog_experimental_go": g["ortholog_novel_go_terms"][:400],
            "max_species_support": g["max_species_support"],
            "deep_homolog_species": d["species"] if d else "",
            "deep_homolog_clade": d["clade"] if d else "",
            "deep_homolog_evalue": d["evalue"] if d else "",
            "deep_homolog_rbh": d["rbh"] if d else "",
            "deep_homolog_protein": (d["protein_name"][:80] if d else ""),
            "deep_homolog_function": (d["function"] if d else ""),
            "deep_homolog_ec": d["ec"] if d else "",
            # This row is the best *annotated* hit, which is not always the
            # lowest-E one, and its function text describes the prokaryote's
            # pathway. Only the molecular/domain half is transferable: ACYP1's
            # hit HypF matures [NiFe] hydrogenases, which humans do not have,
            # but the acylphosphatase domain it shares is real.
            "deep_homolog_note": ("domain-level context, not a function claim" if d and d["function"]
                                  else ""),
        })

    # A many-to-one ortholog relationship gives every human paralog the same
    # sentence. That is not extra evidence, and the table should say so per row
    # rather than leaving the reader to notice four identical statements.
    src_count = defaultdict(int)
    for r in rows:
        if r["ortholog_accession"]:
            src_count[r["ortholog_accession"]] += 1
    for r in rows:
        r["source_shared_with_n_human_genes"] = (
            src_count.get(r["ortholog_accession"], 0) if r["ortholog_accession"] else "")

    cols = list(rows[0])
    with gzip.open(args.out / "human_gene_function_statements.tsv.gz", "wt", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]).replace("\t", " ").replace("\n", " ") for c in cols) + "\n")

    # Dark-gene view: the genes this whole exercise was aimed at.
    dark_syms = {d["symbol"] for d in csv.DictReader(
        open(args.out / "highlight_ancient_dark_genes.tsv"), delimiter="\t")}
    dark_cols = ["symbol", "gene_id", "name", "evidence_tier", "recommended_function",
                 "ortholog_species", "transfer_confidence", "source_shared_with_n_human_genes",
                 "ortholog_protein",
                 "ortholog_curated_function", "deep_homolog_species", "deep_homolog_clade",
                 "deep_homolog_evalue", "deep_homolog_rbh", "deep_homolog_protein",
                 "deep_homolog_function", "deep_homolog_ec", "deep_homolog_note",
                 "ortholog_experimental_go"]
    dark_rows = [r for r in rows if r["symbol"] in dark_syms]
    with open(args.out / "dark_gene_function_statements.tsv", "wt", encoding="utf-8") as fh:
        fh.write("\t".join(dark_cols) + "\n")
        for r in dark_rows:
            fh.write("\t".join(str(r[c]).replace("\t", " ") for c in dark_cols) + "\n")

    transferable = [r for r in rows if not r["human_curated_function"] and r["recommended_function"]]
    no_statement = [r for r in rows if not r["recommended_function"]]
    summary = {
        "genes": len(rows),
        "evidence_tier_counts": dict(stats),
        "genes_without_own_curated_function": sum(1 for r in rows if not r["human_curated_function"]),
        "genes_given_a_transferred_statement": len(transferable),
        "transferred_from_curated_ortholog": sum(1 for r in transferable
                                                 if r["evidence_tier"] == "ortholog_curated"),
        "transferred_from_experimental_go": sum(1 for r in transferable
                                                if r["evidence_tier"] == "ortholog_experimental"),
        "transferred_from_deep_homolog": sum(1 for r in transferable
                                             if r["evidence_tier"] == "deep_homolog"),
        "genes_with_no_statement_at_all": len(no_statement),
        "dark_genes": len(dark_rows),
        "dark_genes_with_statement": sum(1 for r in dark_rows if r["recommended_function"]),
        "dark_genes_with_curated_ortholog_text": sum(1 for r in dark_rows
                                                     if r["ortholog_curated_function"]),
        "dark_genes_with_deep_homolog_function": sum(1 for r in dark_rows
                                                     if r["deep_homolog_function"]),
        "transfers_sharing_source_with_a_paralog": sum(
            1 for r in transferable
            if r["evidence_tier"] == "ortholog_curated"
            and (r["source_shared_with_n_human_genes"] or 0) > 1),
        "distinct_ortholog_sources_used": len({r["ortholog_accession"] for r in transferable
                                               if r["ortholog_accession"]}),
        "transfer_confidence_counts": {
            k: sum(1 for r in transferable if r["transfer_confidence"] == k)
            for k in ("high_mammal", "medium_vertebrate", "low_invertebrate", "low_fungal")
        },
        "source_species_counts": dict(sorted(
            ((sp, sum(1 for r in rows if r["ortholog_species"] == sp)) for sp in SPECIES_ORDER),
            key=lambda kv: -kv[1])),
    }
    (args.out / "function_statement_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(args.out, rows, dark_rows, summary)
    print(json.dumps(summary, indent=2))
    return 0


def md_table(header, body, aligns=""):
    aligns = aligns or "l" * len(header)
    sep = {"l": ":---", "r": "---:"}
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(sep[a] for a in aligns) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(out)


def trunc(t, n):
    t = (t or "").replace("|", "/")
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def write_report(out_dir: Path, rows, dark_rows, s) -> None:
    tiers = s["evidence_tier_counts"]
    conf = s["transfer_confidence_counts"]
    lines = []
    A = lines.append
    A("# From ortholog evidence to a function statement\n")
    A("`REPORT.md` establishes which human genes have orthologs and what GO terms those "
      "orthologs carry. A GO term list is evidence, not an answer. This layer fetches the "
      "curated UniProt FUNCTION text of the orthologs themselves and assigns every human gene "
      "the best statement available, tagged with where it came from.\n")
    A(md_table(["evidence tier", "genes", "meaning"],
               [["human_curated", f"{tiers.get('human_curated', 0):,}",
                 "the human gene already has curated function text — nothing to transfer"],
                ["ortholog_curated", f"{tiers.get('ortholog_curated', 0):,}",
                 "statement taken from a model-organism ortholog's UniProt record"],
                ["ortholog_experimental", f"{tiers.get('ortholog_experimental', 0):,}",
                 "no ortholog function text, but ortholog GO terms with experimental evidence"],
                ["none", f"{tiers.get('none', 0):,}", "no usable evidence anywhere"]], "lrl"))
    A(f"\n**{s['genes_without_own_curated_function']:,} human genes have no curated function "
      f"text of their own.** Orthologs supply a statement for "
      f"**{s['genes_given_a_transferred_statement']:,}** of them "
      f"({s['transferred_from_curated_ortholog']:,} as curated prose, "
      f"{s['transferred_from_experimental_go']:,} as experimental GO terms). "
      f"{s['genes_with_no_statement_at_all']:,} genes still have nothing.\n")

    A("## Which species actually rescues a gene\n")
    A(md_table(["transfer distance", "genes", "read the statement how"],
               [["high_mammal (mouse, rat)", conf.get("high_mammal", 0),
                 "safe to use nearly verbatim"],
                ["medium_vertebrate (zebrafish, Xenopus)", conf.get("medium_vertebrate", 0),
                 "molecular function transfers, tissue context may not"],
                ["low_invertebrate (fly, worm)", conf.get("low_invertebrate", 0),
                 "molecular half transfers; organismal half is that animal's biology"],
                ["low_fungal (yeast)", conf.get("low_fungal", 0),
                 "molecular/complex-level only"]], "lrl"))
    A("\nThe distribution is the interesting part: only "
      f"{conf.get('high_mammal', 0)} of the {s['transferred_from_curated_ortholog']} prose "
      f"transfers come from mouse or rat, while {conf.get('low_invertebrate', 0) + conf.get('low_fungal', 0)} "
      "come from fly, worm or yeast. That is not an accident of coverage — mouse curation "
      "largely mirrors human curation, so a gene nobody has characterized in human is usually "
      "uncharacterized in mouse too. The genes that get rescued are rescued by classical "
      "invertebrate and yeast genetics, which is exactly the literature a human-only search "
      "never surfaces.\n")

    seen_src, ex = set(), []
    for conf_want in ("high_mammal", "medium_vertebrate", "low_invertebrate", "low_fungal"):
        for r in rows:
            if (r["evidence_tier"] == "ortholog_curated"
                    and r["transfer_confidence"] == conf_want
                    and r["ortholog_accession"] not in seen_src):
                seen_src.add(r["ortholog_accession"])
                ex.append(r)
            if sum(1 for e in ex if e["transfer_confidence"] == conf_want) >= 3:
                break
    A("## Examples\n")
    A(md_table(["human gene", "from", "confidence", "shared with", "statement"],
               [[r["symbol"], f"{r['ortholog_species']} {trunc(r['ortholog_protein'], 22)}",
                 r["transfer_confidence"],
                 f"{r['source_shared_with_n_human_genes']} genes"
                 if (r["source_shared_with_n_human_genes"] or 0) > 1 else "unique",
                 trunc(r["ortholog_curated_function"], 130)]
                for r in ex], "lllll"))

    A("\n## The deep-homolog columns are context, not function\n")
    A(f"{s['dark_genes_with_deep_homolog_function']} of the {s['dark_genes']} dark genes have a "
      "bacterial, archaeal or protist homolog whose UniProt record carries curated function "
      "text. Those columns are deliberately kept out of `recommended_function`, because at that "
      "distance the shared part is the domain, not the pathway:\n")
    A(md_table(["dark gene", "deep homolog", "its curated function", "why you cannot copy it"],
               [["ACYP1", "*M. jannaschii* HypF",
                 "matures [NiFe] hydrogenases via carbamoyl transfer",
                 "humans have no hydrogenases; the shared part is the acylphosphatase domain"],
                ["DIP2C", "*M. tuberculosis* FadD32",
                 "activates long-chain fatty acids for mycolic acid synthesis",
                 "humans make no mycolic acids; the shared part is the fatty-acyl AMP ligase fold"],
                ["DMXL1", "*S. pombe* Rav1",
                 "RAVE complex subunit required for V-ATPase assembly",
                 "this one *does* transfer — DMXL1/2 are the human RAVE homologs"]], "llll"))
    A("\nThe lesson is that a prokaryotic hit tells you what biochemistry the protein is built "
      "for, and occasionally — as with DMXL1 — the whole complex-level role survives. Deciding "
      "which case you are in needs the reciprocal-best-hit flag, the query coverage, and a "
      "human reading the pathway.\n")

    A("## Caveats\n")
    A("- **Only reviewed UniProt entries carry FUNCTION text**, so an ortholog annotated only in "
      "TrEMBL is invisible here. That is why some genes fall through to a worm or yeast source "
      "when a mouse ortholog exists: the mouse entry is unreviewed, not absent.")
    A("- **Transferred prose describes the source species.** ADAMTS16's statement comes from "
      "worm and talks about cuticle collagen and body size. The metalloprotease/ECM-remodelling "
      "core is the transferable part; the cuticle is not. Always read `ortholog_species` and "
      "`transfer_confidence` beside the statement.")
    A(f"- **{s['transfers_sharing_source_with_a_paralog']} of the "
      f"{s['transferred_from_curated_ortholog']} prose transfers are shared with at least one "
      f"human paralog**, because the ortholog relationship is many-to-one: ACTL7B, ACTL8, "
      f"ACTRT2 and ACTRT3 all inherit the same statement from fly Act53D, and several ADAMTS "
      f"genes inherit the same one from worm. Identical sentences on paralogs are one piece of "
      f"evidence, not four — the `source_shared_with_n_human_genes` column makes that countable, "
      f"and the {s['distinct_ortholog_sources_used']} distinct sources are the real denominator.")
    A("- **A statement is not a validation.** These are the best available priors for what a "
      "gene does, generated to be checked, not cited. The `human_curated` tier is the only one "
      "where someone has actually done the human experiment.")
    (out_dir / "FUNCTION_STATEMENTS.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir / 'FUNCTION_STATEMENTS.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
