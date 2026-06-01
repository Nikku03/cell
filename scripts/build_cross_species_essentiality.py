"""Build cell_sim/data/cross_species_essentiality.csv.

Maps each syn3A locus to its best homolog in M. genitalium and
M. pneumoniae, and records whether that homolog is essential per the
published saturating Tn-seq calls (Glass 2006 + Lluch-Senar 2015).

Inputs (must exist before this script runs):

  * memory_bank/data/syn3a_gene_table.csv
      syn3A locus_tag -> gene_name, product, protein_id.
  * memory_bank/data/multiorg_essentiality/labels.csv
      per-organism rows of (locus_tag, gene_name, essential, source).
      Populated by notebooks/curate_multiorg_session20.ipynb in Colab.
  * memory_bank/data/multiorg_essentiality/sequences.fasta
      per-organism CDS protein sequences, header
      ``>{organism}|{locus_tag}|{gene_name}``. Same notebook.

Output:

  * cell_sim/data/cross_species_essentiality.csv
      one row per syn3A locus with the following columns:
        locus_tag,
        mg_homolog_locus, mg_homolog_method, mg_pident, mg_essential,
        mp_homolog_locus, mp_homolog_method, mp_pident, mp_essential

Mapping strategy (per syn3A locus):

  1. Gene-name match (cheap, no BLAST): if syn3A locus has a non-empty
     gene_name, find the M.gen / M.pne entry with the same gene_name
     (case-insensitive, stripped of trailing semicolons & punctuation).
     Records ``method='gene_name'``, ``pident=null``.

  2. BLAST best-hit (optional, --use-blast): for syn3A loci unmapped
     by step 1, run blastp against the M.gen / M.pne FASTA. Records
     ``method='blast'``, ``pident=<best hit identity>``. Requires
     ``blastp`` and ``makeblastdb`` on PATH (NCBI BLAST+).

Genes that don't map by either strategy get empty homolog columns —
the detector abstains for them.

Usage::

    # gene-name only (fast, no BLAST dependency)
    python scripts/build_cross_species_essentiality.py

    # with BLAST for unmapped genes (requires blastp on PATH)
    python scripts/build_cross_species_essentiality.py --use-blast \
        --blast-evalue 1e-5
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


SYN3A_GENE_TABLE = REPO_ROOT / "memory_bank" / "data" / "syn3a_gene_table.csv"
MULTIORG_LABELS  = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "labels.csv"
MULTIORG_FASTA   = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "sequences.fasta"
OUTPUT_CSV       = REPO_ROOT / "cell_sim" / "data" / "cross_species_essentiality.csv"


def _norm_gene_name(s: str) -> str:
    """Lowercase, strip whitespace and trailing punctuation, split on ;
    and take the first name."""
    s = (s or "").strip().lower()
    if not s:
        return ""
    s = s.split(";")[0].strip()
    s = s.strip(".,:")
    return s


def _read_syn3a_gene_table() -> list[dict]:
    rows = []
    with open(SYN3A_GENE_TABLE) as f:
        for r in csv.DictReader(f):
            rows.append({
                "locus_tag": r["locus_tag"].strip(),
                "gene_name": r.get("gene_name", "").strip(),
            })
    return rows


def _read_multiorg_labels() -> dict[str, list[dict]]:
    """Return {organism: [{'locus_tag','gene_name','essential'}, ...]}."""
    by_org: dict[str, list[dict]] = {}
    with open(MULTIORG_LABELS) as f:
        for r in csv.DictReader(f):
            org = r["organism"].strip()
            by_org.setdefault(org, []).append({
                "locus_tag": r["locus_tag"].strip(),
                "gene_name": r.get("gene_name", "").strip(),
                "essential": int(r.get("essential", 0)),
            })
    return by_org


def _read_multiorg_fasta() -> dict[tuple[str, str], str]:
    """Return {(organism, locus_tag): protein_sequence}. Header format:
    ``>{organism}|{locus_tag}|{gene_name}``."""
    out: dict[tuple[str, str], str] = {}
    cur_key: Optional[tuple[str, str]] = None
    cur_seq: list[str] = []
    if not MULTIORG_FASTA.exists():
        return out
    with open(MULTIORG_FASTA) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_key is not None:
                    out[cur_key] = "".join(cur_seq)
                parts = line[1:].split("|")
                if len(parts) >= 2:
                    cur_key = (parts[0].strip(), parts[1].strip())
                else:
                    cur_key = None
                cur_seq = []
            else:
                if cur_key is not None:
                    cur_seq.append(line)
        if cur_key is not None:
            out[cur_key] = "".join(cur_seq)
    return out


def _map_by_gene_name(
    syn3a_rows: list[dict], org_rows: list[dict],
) -> dict[str, dict]:
    """Build syn3a_locus -> {'homolog_locus','method','pident','essential'}."""
    org_lookup: dict[str, dict] = {}
    for r in org_rows:
        gn = _norm_gene_name(r["gene_name"])
        if not gn:
            continue
        # If duplicate gene_name in reference (rare), prefer essential=1
        if gn in org_lookup and org_lookup[gn]["essential"] == 1:
            continue
        org_lookup[gn] = r

    mapped: dict[str, dict] = {}
    for s in syn3a_rows:
        gn = _norm_gene_name(s["gene_name"])
        if not gn:
            continue
        hit = org_lookup.get(gn)
        if hit is None:
            continue
        mapped[s["locus_tag"]] = {
            "homolog_locus": hit["locus_tag"],
            "method":        "gene_name",
            "pident":        None,
            "essential":     int(hit["essential"]),
        }
    return mapped


def _run_blastp_best_hits(
    query_seqs: dict[str, str],
    subject_seqs: dict[str, str],
    evalue: float,
) -> dict[str, tuple[str, float]]:
    """Return {query_id: (subject_id, percent_identity)} for the best hit
    per query. ``query_seqs`` and ``subject_seqs`` map id -> protein seq."""
    if shutil.which("blastp") is None or shutil.which("makeblastdb") is None:
        raise RuntimeError(
            "blastp / makeblastdb not found on PATH. Install NCBI BLAST+ "
            "or rerun without --use-blast."
        )
    with tempfile.TemporaryDirectory() as tdir:
        tdir_p = Path(tdir)
        q_fa = tdir_p / "query.faa"
        s_fa = tdir_p / "subject.faa"
        with open(q_fa, "w") as f:
            for qid, seq in query_seqs.items():
                f.write(f">{qid}\n{seq}\n")
        with open(s_fa, "w") as f:
            for sid, seq in subject_seqs.items():
                f.write(f">{sid}\n{seq}\n")
        subprocess.run(
            ["makeblastdb", "-in", str(s_fa), "-dbtype", "prot",
             "-out", str(tdir_p / "subj_db")],
            check=True, capture_output=True,
        )
        out_tab = tdir_p / "hits.tsv"
        subprocess.run(
            ["blastp",
             "-query",   str(q_fa),
             "-db",      str(tdir_p / "subj_db"),
             "-evalue",  str(evalue),
             "-outfmt",  "6 qseqid sseqid pident evalue bitscore",
             "-max_target_seqs", "1",
             "-out", str(out_tab)],
            check=True, capture_output=True,
        )
        best: dict[str, tuple[str, float, float]] = {}
        with open(out_tab) as f:
            for line in f:
                parts = line.rstrip().split("\t")
                if len(parts) < 5:
                    continue
                q, s, pid, ev, bs = parts[0], parts[1], float(parts[2]), \
                                     float(parts[3]), float(parts[4])
                prev = best.get(q)
                if prev is None or bs > prev[2]:
                    best[q] = (s, pid, bs)
        return {q: (s, pid) for q, (s, pid, _) in best.items()}


def _augment_with_blast(
    unmapped_syn3a: list[str],
    syn3a_seqs: dict[str, str],
    org_rows: list[dict],
    org_seqs: dict[str, str],
    evalue: float,
) -> dict[str, dict]:
    """For each unmapped syn3A locus_tag, run blastp vs the org FASTA and
    return the same shape as _map_by_gene_name()."""
    if not org_seqs:
        return {}
    # Build per-locus essentiality lookup for the reference org
    ess_lookup = {r["locus_tag"]: int(r["essential"]) for r in org_rows}
    # Query sequences are only those unmapped + present in syn3a_seqs
    qs = {lt: syn3a_seqs[lt] for lt in unmapped_syn3a if lt in syn3a_seqs}
    if not qs:
        return {}
    hits = _run_blastp_best_hits(qs, org_seqs, evalue)
    mapped: dict[str, dict] = {}
    for qid, (sid, pid) in hits.items():
        # sid was the key in org_seqs i.e. the reference locus_tag
        ess = ess_lookup.get(sid)
        if ess is None:
            continue
        mapped[qid] = {
            "homolog_locus": sid,
            "method":        "blast",
            "pident":        round(pid, 1),
            "essential":     int(ess),
        }
    return mapped


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--use-blast", action="store_true",
                   help="For loci without a gene_name match, run blastp "
                        "vs the M.gen / M.pne sequences. Requires "
                        "blastp + makeblastdb on PATH.")
    p.add_argument("--blast-evalue", type=float, default=1e-5)
    p.add_argument("--out", type=Path, default=OUTPUT_CSV)
    args = p.parse_args()

    print(f"  reading syn3A gene table: {SYN3A_GENE_TABLE}")
    syn3a_rows = _read_syn3a_gene_table()
    print(f"    {len(syn3a_rows)} syn3A loci "
          f"({sum(1 for r in syn3a_rows if r['gene_name'])} with gene_name)")

    print(f"  reading multiorg labels:   {MULTIORG_LABELS}")
    by_org = _read_multiorg_labels()
    n_mgen = len(by_org.get("mgen", []))
    n_mpne = len(by_org.get("mpne", []))
    print(f"    M. genitalium rows: {n_mgen}  "
          f"({sum(1 for r in by_org.get('mgen', []) if r['essential']==1)} essential)")
    print(f"    M. pneumoniae rows: {n_mpne}  "
          f"({sum(1 for r in by_org.get('mpne', []) if r['essential']==1)} essential)")

    if n_mgen == 0 and n_mpne == 0:
        print("  WARNING: no M.gen or M.pne rows in labels.csv. "
              "Run notebooks/curate_multiorg_session20.ipynb in Colab "
              "first to populate the multiorg dataset.")
        # Still write an empty CSV with the right schema.

    seqs_by_loc = _read_multiorg_fasta()
    syn3a_seqs   = {lt: seq for (org, lt), seq in seqs_by_loc.items() if org == "syn3a"}
    mgen_seqs    = {lt: seq for (org, lt), seq in seqs_by_loc.items() if org == "mgen"}
    mpne_seqs    = {lt: seq for (org, lt), seq in seqs_by_loc.items() if org == "mpne"}
    print(f"    syn3a FASTA entries: {len(syn3a_seqs)}")
    print(f"    M.gen FASTA entries: {len(mgen_seqs)}")
    print(f"    M.pne FASTA entries: {len(mpne_seqs)}")

    print(f"\n  gene-name mapping ...")
    mgen_map = _map_by_gene_name(syn3a_rows, by_org.get("mgen", []))
    mpne_map = _map_by_gene_name(syn3a_rows, by_org.get("mpne", []))
    print(f"    syn3A <-> M.gen by gene_name: {len(mgen_map)}")
    print(f"    syn3A <-> M.pne by gene_name: {len(mpne_map)}")

    if args.use_blast:
        print(f"\n  BLAST fallback (e-value <= {args.blast_evalue}) ...")
        unmapped_mgen = [r["locus_tag"] for r in syn3a_rows if r["locus_tag"] not in mgen_map]
        unmapped_mpne = [r["locus_tag"] for r in syn3a_rows if r["locus_tag"] not in mpne_map]
        try:
            mgen_blast = _augment_with_blast(
                unmapped_mgen, syn3a_seqs, by_org.get("mgen", []),
                mgen_seqs, args.blast_evalue,
            )
            print(f"    BLAST added {len(mgen_blast)} M.gen mappings")
            mgen_map.update(mgen_blast)
        except RuntimeError as exc:
            print(f"    skipping M.gen BLAST: {exc}")
        try:
            mpne_blast = _augment_with_blast(
                unmapped_mpne, syn3a_seqs, by_org.get("mpne", []),
                mpne_seqs, args.blast_evalue,
            )
            print(f"    BLAST added {len(mpne_blast)} M.pne mappings")
            mpne_map.update(mpne_blast)
        except RuntimeError as exc:
            print(f"    skipping M.pne BLAST: {exc}")

    print(f"\n  writing: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_with_mgen = 0
    n_with_mpne = 0
    n_either_ess = 0
    n_both_ess = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "locus_tag",
            "mg_homolog_locus", "mg_homolog_method", "mg_pident", "mg_essential",
            "mp_homolog_locus", "mp_homolog_method", "mp_pident", "mp_essential",
        ])
        for r in syn3a_rows:
            lt = r["locus_tag"]
            mg = mgen_map.get(lt)
            mp = mpne_map.get(lt)
            if mg:
                n_with_mgen += 1
            if mp:
                n_with_mpne += 1
            mg_ess = mg["essential"] if mg else None
            mp_ess = mp["essential"] if mp else None
            if mg_ess == 1 or mp_ess == 1:
                n_either_ess += 1
            if mg_ess == 1 and mp_ess == 1:
                n_both_ess += 1
            w.writerow([
                lt,
                mg["homolog_locus"] if mg else "",
                mg["method"]        if mg else "",
                f"{mg['pident']:.1f}" if mg and mg["pident"] is not None else "",
                str(mg_ess)         if mg_ess is not None else "",
                mp["homolog_locus"] if mp else "",
                mp["method"]        if mp else "",
                f"{mp['pident']:.1f}" if mp and mp["pident"] is not None else "",
                str(mp_ess)         if mp_ess is not None else "",
            ])
    print(f"\n  ===  summary  ===")
    print(f"  syn3A loci written:       {len(syn3a_rows)}")
    print(f"    with M.gen homolog:     {n_with_mgen}")
    print(f"    with M.pne homolog:     {n_with_mpne}")
    print(f"    at least one ess call:  {n_either_ess}")
    print(f"    both refs ess agree:    {n_both_ess}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
