"""Search the 235 dark human genes for homologs outside NCBI's ortholog set.

`scripts/human_gene_orthologs.py` leaves 235 human genes that have no curated
human function but a characterized fly/worm/yeast ortholog. NCBI's ortholog
dump is vertebrate-only and the Alliance set stops at fly/worm/yeast, so
nothing in that pipeline can say whether these genes reach *bacteria*, archaea,
plants or protists. That needs actual sequence search, which is what this does.

Two passes, because they answer different questions:

  phmmer     single sequence vs each proteome. Conservative: a hit here is a
             similarity you could have found with BLAST, and at human-to-
             bacteria distance most true homologies are invisible to it.
  profile    for each gene, the eukaryotic phmmer hits are aligned to the
             query and rebuilt into a profile HMM, which is then searched
             against the prokaryotic proteomes only. This is the jackhmmer
             mechanism -- a profile carries the family's conserved positions
             where a single sequence does not -- with the expensive search
             restricted to the 21k prokaryotic sequences instead of all 283k,
             which is the difference between 10 minutes and 4 hours. It is
             also where false positives come from, hence the reciprocal check.

Every hit is then reciprocally searched back against the human proteome. If
the original gene comes back as the top human hit, the pair is a reciprocal
best hit and is reported as an ortholog candidate; otherwise it is reported as
homology only (usually the query hitting a large shared family such as a
P-loop NTPase or a WD40 repeat, where the true counterpart is some other human
gene).

Usage::

    python3 scripts/deep_homology_dark_genes.py --resolve-panel   # once
    scripts/fetch_deep_homology_data.sh                           # ~180 MB
    python3 scripts/deep_homology_dark_genes.py                   # search
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import pyhmmer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO_ROOT / "data_cache" / "deep_homology"
DEFAULT_OUT = REPO_ROOT / "outputs" / "human_orthologs"
DARK_GENES = DEFAULT_OUT / "highlight_ancient_dark_genes.tsv"
MAIN_TABLE = DEFAULT_OUT / "human_gene_ortholog_function.tsv.gz"

# Species outside both ortholog sources used upstream. NCBI covers 895
# vertebrates; the Alliance adds fly, worm, yeast and Xenopus. Nothing here
# overlaps either.
PANEL = [
    (83333, "Escherichia coli K-12", "bacteria"),
    (224308, "Bacillus subtilis 168", "bacteria"),
    (243273, "Mycoplasmoides genitalium G37", "bacteria"),
    (272632, "Mycoplasma mycoides SC PG1", "bacteria"),
    (1111708, "Synechocystis sp. PCC 6803", "bacteria"),
    (83332, "Mycobacterium tuberculosis H37Rv", "bacteria"),
    (300852, "Thermus thermophilus HB8", "bacteria"),
    (272947, "Rickettsia prowazekii", "bacteria"),
    (243232, "Methanocaldococcus jannaschii", "archaea"),
    (273057, "Saccharolobus solfataricus P2", "archaea"),
    (64091, "Halobacterium salinarum NRC-1", "archaea"),
    (69014, "Thermococcus kodakarensis", "archaea"),
    (284812, "Schizosaccharomyces pombe", "fungi_other"),
    (3702, "Arabidopsis thaliana", "plant"),
    (3055, "Chlamydomonas reinhardtii", "alga"),
    (44689, "Dictyostelium discoideum", "amoebozoa"),
    (36329, "Plasmodium falciparum 3D7", "apicomplexa"),
    (185431, "Trypanosoma brucei", "excavata"),
    (184922, "Giardia intestinalis", "excavata"),
    (81824, "Monosiga brevicollis", "choanoflagellate"),
    (45351, "Nematostella vectensis", "cnidaria"),
    (10228, "Trichoplax adhaerens", "placozoa"),
    (400682, "Amphimedon queenslandica", "porifera"),
    (7719, "Ciona intestinalis", "tunicate"),
    (7739, "Branchiostoma floridae", "cephalochordate"),
    (7668, "Strongylocentrotus purpuratus", "echinoderm"),
]
PROKARYOTE_CLADES = {"bacteria", "archaea"}
# Reported depth, deepest first. A prokaryotic hit is the headline result.
DOMAIN_ORDER = ["bacteria", "archaea", "excavata", "apicomplexa", "amoebozoa", "plant", "alga",
                "fungi_other", "choanoflagellate", "porifera", "placozoa", "cnidaria",
                "echinoderm", "cephalochordate", "tunicate"]

HIT_E = 1e-3        # report threshold
STRONG_E = 1e-10    # "strong" annotation in the output


def as_str(x) -> str:
    """pyhmmer >=0.11 returns str names, older versions bytes. Accept both."""
    return x.decode() if isinstance(x, (bytes, bytearray)) else x


def query_name(hits) -> str:
    q = hits.query if hasattr(hits, "query") else None
    return as_str(q.name if q is not None else hits.query_name)


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Panel resolution + query sequences (network stages, run once)
# ---------------------------------------------------------------------------
def curl(url: str, params: list[tuple[str, str]], timeout: int = 120) -> str:
    cmd = ["curl", "-sS", "--max-time", str(timeout), "--compressed", "-G", url]
    for k, v in params:
        cmd += ["--data-urlencode", f"{k}={v}"]
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def resolve_panel(cache: Path) -> None:
    """Write panel.json: one reference proteome per panel species."""
    cache.mkdir(parents=True, exist_ok=True)
    res = []
    for taxid, label, clade in PANEL:
        rows = []
        for q, tag in ((f"(organism_id:{taxid}) AND (reference:true)", "reference"),
                       (f"(organism_id:{taxid})", "fallback_largest")):
            txt = curl("https://rest.uniprot.org/proteomes/stream",
                       [("query", q), ("format", "tsv")]).strip().split("\n")
            rows = [{"upid": f[0], "organism": f[1], "n": int(f[3])}
                    for f in (l.split("\t") for l in txt[1:]) if len(f) >= 4 and f[3].isdigit()]
            if rows:
                break
        if not rows:
            print(f"NOT FOUND: {label}")
            continue
        best = max(rows, key=lambda r: r["n"])
        res.append({**best, "label": label, "clade": clade, "query_taxid": taxid, "selection": tag})
        print(f"{label:34s} {best['upid']:12s} {best['n']:>7,} proteins [{tag}]")
    (cache / "panel.json").write_text(json.dumps(res, indent=1))


def resolve_queries(cache: Path) -> None:
    """Write query_entries.json / query_entries_refseq.json for the dark genes."""
    dark = list(csv.DictReader(open(DARK_GENES), delimiter="\t"))
    gids = [d["gene_id"] for d in dark]
    out: dict[str, tuple] = {}
    for i in range(0, len(gids), 40):
        chunk = gids[i:i + 40]
        txt = curl("https://rest.uniprot.org/uniprotkb/stream", [
            ("query", " OR ".join(f"(xref:geneid-{g})" for g in chunk)),
            ("fields", "accession,reviewed,xref_geneid,length,sequence,xref_pfam,"
                       "xref_interpro,protein_name"),
            ("format", "tsv")], timeout=180)
        for r in csv.DictReader(txt.splitlines(), delimiter="\t"):
            for raw in r["GeneID"].split(";"):
                raw = raw.strip()
                if raw in set(chunk):
                    key = (r["Reviewed"] == "reviewed", int(r["Length"]))
                    if raw not in out or key > out[raw][0]:
                        out[raw] = (key, r)
    (cache / "query_entries.json").write_text(json.dumps({k: v[1] for k, v in out.items()}))

    # Readthrough transcripts and a few others have no UniProt entry; fall back
    # to the RefSeq protein NCBI links to the GeneID.
    eutils = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    rs = {}
    for d in dark:
        if d["gene_id"] in out:
            continue
        j = subprocess.run(["curl", "-sS", "--max-time", "60",
                            f"{eutils}/elink.fcgi?dbfrom=gene&db=protein&id={d['gene_id']}"
                            f"&retmode=json"], capture_output=True, text=True).stdout
        try:
            dbs = json.loads(j)["linksets"][0].get("linksetdbs", [])
            ids = next((l["links"] for l in dbs
                        if l["linkname"] in ("gene_protein_refseq", "gene_protein")), [])
        except Exception:
            ids = []
        if not ids:
            print(f"no protein for {d['symbol']}")
            continue
        fa = subprocess.run(["curl", "-sS", "--max-time", "60",
                             f"{eutils}/efetch.fcgi?db=protein&id={ids[0]}&rettype=fasta"
                             f"&retmode=text"], capture_output=True, text=True).stdout
        lines = fa.strip().split("\n")
        seq = "".join(lines[1:])
        if seq:
            rs[d["gene_id"]] = {"Entry": lines[0].split()[0].lstrip(">"), "Sequence": seq,
                                "Length": str(len(seq)), "Reviewed": "refseq",
                                "Protein names": " ".join(lines[0].split()[1:]),
                                "Pfam": "", "InterPro": ""}
        time.sleep(0.4)
    (cache / "query_entries_refseq.json").write_text(json.dumps(rs))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def load_queries(cache: Path):
    dark = {d["gene_id"]: d for d in csv.DictReader(open(DARK_GENES), delimiter="\t")}
    entries = json.loads((cache / "query_entries.json").read_text())
    entries.update(json.loads((cache / "query_entries_refseq.json").read_text()))
    alphabet = pyhmmer.easel.Alphabet.amino()
    seqs, meta = [], {}
    for gid, e in entries.items():
        if gid not in dark or not e.get("Sequence"):
            continue
        name = f"gene{gid}"
        seqs.append(pyhmmer.easel.TextSequence(
            name=name, description=dark[gid]["symbol"],
            sequence=e["Sequence"]).digitize(alphabet))
        meta[name] = {"gene_id": gid, "symbol": dark[gid]["symbol"],
                      "accession": e["Entry"], "length": int(e["Length"]),
                      "pfam": e.get("Pfam", ""), "interpro": e.get("InterPro", ""),
                      "depth_ncbi": dark[gid]["conservation_depth"],
                      "inferred": dark[gid]["ortholog_inferred_function"]}
    return pyhmmer.easel.DigitalSequenceBlock(alphabet, seqs), meta, alphabet


def load_fasta(path: Path, alphabet):
    with pyhmmer.easel.SequenceFile(str(path), digital=True, alphabet=alphabet) as fh:
        return fh.read_block()


def hit_rows(hits, species: str, clade: str, method: str, meta: dict):
    """Flatten a TopHits into plain dicts."""
    qname = query_name(hits)
    out = []
    for h in hits:
        if h.evalue > HIT_E:
            continue
        dom = h.best_domain
        env = (dom.env_to - dom.env_from + 1) if dom is not None else 0
        out.append({
            "query": qname, "symbol": meta[qname]["symbol"], "gene_id": meta[qname]["gene_id"],
            "species": species, "clade": clade, "method": method,
            "target": as_str(h.name),
            "target_desc": as_str(h.description or "")[:120],
            "evalue": h.evalue, "score": h.score,
            "query_coverage": round(env / max(meta[qname]["length"], 1), 3),
        })
    return out


def search(cache: Path, out_dir: Path, cpus: int, max_seeds: int, t0: float,
           reuse_forward: bool = False) -> dict:
    queries, meta, alphabet = load_queries(cache)
    log(f"queries: {len(queries)} dark-gene proteins", t0)
    panel = json.loads((cache / "panel.json").read_text())

    fwd = cache / "forward_hits.json.gz"
    if reuse_forward and fwd.exists():
        with gzip.open(fwd, "rt", encoding="utf-8") as fh:
            cached = json.load(fh)
        rows, seed_counts = cached["rows"], cached["seed_counts"]
        log(f"reusing cached forward stage: {len(rows):,} hits", t0)
        all_seqs = []
        for p in panel:
            path = cache / "proteomes" / f"{p['label'].replace(' ', '_')}.fasta"
            if path.exists():
                all_seqs.extend(load_fasta(path, alphabet))
        rows = reciprocal_check(rows, cache, all_seqs, meta, alphabet, cpus, t0)
        return write_outputs(out_dir, rows, meta, panel, seed_counts, t0)
    background = pyhmmer.plan7.Background(alphabet)
    builder = pyhmmer.plan7.Builder(alphabet)

    rows: list[dict] = []
    all_seqs, by_name, prok_seqs = [], {}, []
    for p in panel:
        path = cache / "proteomes" / f"{p['label'].replace(' ', '_')}.fasta"
        if not path.exists():
            print(f"missing proteome {path}")
            continue
        targets = load_fasta(path, alphabet)
        for s in targets:
            by_name[as_str(s.name)] = s
            all_seqs.append(s)
            if p["clade"] in PROKARYOTE_CLADES:
                prok_seqs.append(s)
        for hits in pyhmmer.hmmer.phmmer(queries, targets, cpus=cpus, E=HIT_E):
            rows.extend(hit_rows(hits, p["label"], p["clade"], "phmmer", meta))
        log(f"phmmer {p['label']}: {len(targets):,} targets, {len(rows):,} hits so far", t0)

    # ---- profile stage -------------------------------------------------
    # Single-sequence search is close to blind at human-to-bacteria distance.
    # Build a profile per gene from its eukaryotic homologs (the phmmer hits
    # above), then search only the prokaryotic proteomes with it. Same
    # mechanism jackhmmer uses, minus the cost of iterating over the whole
    # panel: the expensive search runs against 21k sequences, not 283k.
    seeds_by_gene: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for r in rows:
        if r["clade"] not in PROKARYOTE_CLADES and r["query_coverage"] >= 0.3:
            seeds_by_gene[r["symbol"]].append((r["evalue"], r["target"]))
    prok_block = pyhmmer.easel.DigitalSequenceBlock(alphabet, prok_seqs)
    log(f"profile stage: {len(prok_block):,} prokaryotic targets", t0)

    hmms, seed_counts = [], {}
    for q in queries:
        qname = as_str(q.name)
        sym = meta[qname]["symbol"]
        seeds = [by_name[t] for _, t in sorted(seeds_by_gene.get(sym, []))[:max_seeds]
                 if t in by_name]
        hmm0, _, _ = builder.build(q, background)
        if seeds:
            msa = pyhmmer.hmmer.hmmalign(hmm0, [q] + seeds)
            dmsa = msa.digitize(alphabet)
            dmsa.name = qname.encode()
            hmm, _, _ = builder.build_msa(dmsa, background)
        else:
            hmm = hmm0
        hmms.append(hmm)
        seed_counts[sym] = len(seeds)
    log(f"built {len(hmms)} profiles (median seeds "
        f"{sorted(seed_counts.values())[len(seed_counts) // 2]})", t0)

    n_before = len(rows)
    prok_species = {as_str(s.name): None for s in prok_seqs}
    species_of = {}
    for p in panel:
        if p["clade"] not in PROKARYOTE_CLADES:
            continue
        path = cache / "proteomes" / f"{p['label'].replace(' ', '_')}.fasta"
        for s in load_fasta(path, alphabet):
            species_of[as_str(s.name)] = (p["label"], p["clade"])
    for hits in pyhmmer.hmmer.hmmsearch(hmms, prok_block, cpus=cpus, E=HIT_E):
        qname = as_str(hits.query.name if hasattr(hits, "query") else hits.query_name)
        for h in hits:
            if h.evalue > HIT_E:
                continue
            sp, clade = species_of.get(as_str(h.name), ("?", "?"))
            dom = h.best_domain
            env = (dom.env_to - dom.env_from + 1) if dom is not None else 0
            rows.append({
                "query": qname, "symbol": meta[qname]["symbol"],
                "gene_id": meta[qname]["gene_id"], "species": sp, "clade": clade,
                "method": "profile", "target": as_str(h.name),
                "target_desc": as_str(h.description or "")[:120],
                "evalue": h.evalue, "score": h.score,
                "query_coverage": round(env / max(meta[qname]["length"], 1), 3),
            })
    log(f"profile search added {len(rows) - n_before:,} prokaryotic hits", t0)

    fwd = cache / "forward_hits.json.gz"
    with gzip.open(fwd, "wt", encoding="utf-8") as fh:
        json.dump({"rows": rows, "seed_counts": seed_counts}, fh)
    log(f"cached forward hits -> {fwd}", t0)

    rows = reciprocal_check(rows, cache, all_seqs, meta, alphabet, cpus, t0)
    return write_outputs(out_dir, rows, meta, panel, seed_counts, t0)


def reciprocal_check(rows, cache: Path, pooled_seqs, meta, alphabet, cpus: int, t0: float):
    """Search each hit back against the human proteome; flag reciprocal best hits."""
    reviewed = cache / "human_reference_reviewed.fasta"
    human = load_fasta(reviewed if reviewed.exists() else cache / "human_reference.fasta",
                       alphabet)
    log(f"reciprocal target: {len(human):,} human proteins", t0)

    by_name = {as_str(s.name): s for s in pooled_seqs}
    # Reciprocal search is quadratic in hit count, so bound it: every
    # prokaryotic hit (the claim that actually needs defending) plus the single
    # best eukaryotic hit per gene/species. Eukaryotic hits beyond the best one
    # are near-identical paralogs whose reciprocal verdict is the same.
    best_euk: dict[tuple[str, str], tuple[float, str]] = {}
    wanted: set[str] = set()
    for r in rows:
        if r["clade"] in PROKARYOTE_CLADES:
            wanted.add(r["target"])
        else:
            key = (r["symbol"], r["species"])
            if key not in best_euk or r["evalue"] < best_euk[key][0]:
                best_euk[key] = (r["evalue"], r["target"])
    wanted |= {t for _, t in best_euk.values()}
    block = pyhmmer.easel.DigitalSequenceBlock(
        alphabet, [by_name[n] for n in sorted(wanted) if n in by_name])
    log(f"reciprocal: {len(block):,} distinct hit sequences", t0)

    top: dict[str, tuple[str, str, float]] = {}
    for hits in pyhmmer.hmmer.phmmer(block, human, cpus=cpus, E=10.0):
        qname = query_name(hits)
        best = next(iter(hits), None)
        if best is not None:
            desc = as_str(best.description or "")
            gn = ""
            for tok in desc.split():
                if tok.startswith("GN="):
                    gn = tok[3:]
            name = as_str(best.name)
            acc = name.split("|")[1] if "|" in name else name
            top[qname] = (acc, gn, best.evalue)

    n_rbh = 0
    for r in rows:
        if r["target"] not in wanted:
            r["reciprocal_top_human"] = "not_tested"
            r["reciprocal_best_hit"] = 0
            continue
        acc, gn, ev = top.get(r["target"], ("", "", float("nan")))
        r["reciprocal_top_human"] = gn or acc
        m = meta[r["query"]]
        r["reciprocal_best_hit"] = int(bool(gn and gn == m["symbol"]) or
                                       bool(acc and acc == m["accession"]))
        n_rbh += r["reciprocal_best_hit"]
    for r in rows:
        r["query"] = meta[r["query"]]["symbol"] if r["query"] in meta else r["query"]
    log(f"reciprocal: {n_rbh:,} of {len(rows):,} hit rows are reciprocal best hits", t0)
    return rows


def write_outputs(out_dir: Path, rows: list[dict], meta: dict, panel, seed_counts: dict, t0: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["symbol", "gene_id", "species", "clade", "method", "target", "target_desc",
            "evalue", "score", "query_coverage", "reciprocal_top_human", "reciprocal_best_hit"]
    rows.sort(key=lambda r: (r["symbol"], r["evalue"]))
    with gzip.open(out_dir / "dark_gene_deep_homology_hits.tsv.gz", "wt", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]).replace("\t", " ") for c in cols) + "\n")

    # Per-gene rollup.
    per_gene: dict[str, dict] = {}
    for name, m in meta.items():
        per_gene[m["symbol"]] = {
            "symbol": m["symbol"], "gene_id": m["gene_id"], "accession": m["accession"],
            "length": m["length"], "pfam": m["pfam"], "interpro": m["interpro"],
            "depth_ncbi": m["depth_ncbi"], "inferred_function": m["inferred"],
            "profile_seeds": seed_counts.get(m["symbol"], 0),
            "clades_hit": set(), "best_prok_e": float("inf"), "best_prok": "",
            "best_prok_species": "", "best_prok_method": "", "prok_rbh": 0,
            "targets": set(), "n_species": set(),
        }
    for r in rows:
        g = per_gene[r["symbol"]]
        g["targets"].add(r["target"])
        g["clades_hit"].add(r["clade"])
        g["n_species"].add(r["species"])
        if r["clade"] in PROKARYOTE_CLADES:
            g["prok_rbh"] = max(g["prok_rbh"], r["reciprocal_best_hit"])
            if r["evalue"] < g["best_prok_e"]:
                g.update(best_prok_e=r["evalue"], best_prok=r["target_desc"] or r["target"],
                         best_prok_species=r["species"], best_prok_method=r["method"])

    summary_cols = ["symbol", "gene_id", "accession", "depth_ncbi", "deepest_domain",
                    "n_hits", "n_species_hit", "clades_hit", "reaches_prokaryote",
                    "best_prokaryote_species", "best_prokaryote_evalue", "best_prokaryote_hit",
                    "prokaryote_method", "prokaryote_reciprocal_best_hit", "profile_seeds",
                    "pfam", "inferred_function"]
    out = []
    for g in per_gene.values():
        clades = [c for c in DOMAIN_ORDER if c in g["clades_hit"]]
        out.append({
            "symbol": g["symbol"], "gene_id": g["gene_id"], "accession": g["accession"],
            "depth_ncbi": g["depth_ncbi"], "deepest_domain": clades[0] if clades else "none",
            "n_hits": len(g["targets"]), "n_species_hit": len(g["n_species"]),
            "clades_hit": ",".join(clades),
            "reaches_prokaryote": int(bool(PROKARYOTE_CLADES & g["clades_hit"])),
            "best_prokaryote_species": g["best_prok_species"],
            "best_prokaryote_evalue": "" if g["best_prok_e"] == float("inf") else f"{g['best_prok_e']:.2e}",
            "best_prokaryote_hit": g["best_prok"], "prokaryote_method": g["best_prok_method"],
            "prokaryote_reciprocal_best_hit": g["prok_rbh"],
            "profile_seeds": g["profile_seeds"], "pfam": g["pfam"],
            "inferred_function": g["inferred_function"],
        })
    out.sort(key=lambda r: (-r["reaches_prokaryote"], -r["prokaryote_reciprocal_best_hit"],
                            r["best_prokaryote_evalue"] or "z"))
    with open(out_dir / "dark_gene_deep_homology.tsv", "wt", encoding="utf-8") as fh:
        fh.write("\t".join(summary_cols) + "\n")
        for r in out:
            fh.write("\t".join(str(r[c]).replace("\t", " ") for c in summary_cols) + "\n")

    prok = [r for r in out if r["reaches_prokaryote"]]
    stats = {
        "dark_genes_searched": len(out),
        "panel_species": len(panel),
        "panel_proteins": sum(p["n"] for p in panel),
        "genes_with_any_hit": sum(1 for r in out if r["n_hits"] > 0),
        "genes_with_no_hit": sum(1 for r in out if r["n_hits"] == 0),
        "genes_reaching_prokaryotes": len(prok),
        "genes_reaching_prokaryotes_rbh": sum(1 for r in prok
                                              if r["prokaryote_reciprocal_best_hit"]),
        "genes_reaching_bacteria": sum(1 for r in out if "bacteria" in r["clades_hit"]),
        "genes_reaching_archaea": sum(1 for r in out if "archaea" in r["clades_hit"]),
        "deepest_domain_counts": {},
        "total_hit_rows": len(rows),
        "hits_by_method": {},
    }
    for r in out:
        stats["deepest_domain_counts"][r["deepest_domain"]] = \
            stats["deepest_domain_counts"].get(r["deepest_domain"], 0) + 1
    for r in rows:
        stats["hits_by_method"][r["method"]] = stats["hits_by_method"].get(r["method"], 0) + 1
    (out_dir / "deep_homology_summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    log(f"wrote {len(out)} gene rows, {len(rows):,} hit rows", t0)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--max-seeds", type=int, default=60,
                    help="cap on eukaryotic homologs seeding each profile")
    ap.add_argument("--reuse-forward", action="store_true",
                    help="skip phmmer/profile and reuse data_cache forward_hits.json.gz")
    ap.add_argument("--resolve-panel", action="store_true", help="write panel.json and exit")
    ap.add_argument("--resolve-queries", action="store_true", help="fetch query sequences and exit")
    args = ap.parse_args(argv)

    t0 = time.time()
    if args.resolve_panel:
        resolve_panel(args.cache)
        return 0
    if args.resolve_queries:
        resolve_queries(args.cache)
        return 0

    stats = search(args.cache, args.out, args.cpus, args.max_seeds, t0, args.reuse_forward)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
