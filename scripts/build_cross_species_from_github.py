"""Fetch + build cross_species_essentiality.csv from public GitHub mirrors.

This is the v16 unblocker. The original `scripts/build_cross_species_essentiality.py`
expected pre-curated `memory_bank/data/multiorg_essentiality/{labels.csv,sequences.fasta}`
that the Session 20a fetcher was supposed to populate from DEG — but DEG URLs
went 404 and Session 20a committed only the syn3A rows. This script does the
end-to-end fetch from confirmed-alive GitHub raw URLs (sandbox-reachable).

Sources (all open-access, GitHub-hosted, no auth required):

  M. genitalium G37 (~525 CDS):
    Essentiality: pauloburke/whole-cell-network/mg_build/mg_essentiality_basis.xlsx
                  (Karr 2012 Cell whole-cell-model paper -- column 'Exp' is the
                  Glass 2006 PNAS experimental essentiality call: Y / N)
    GFF (locus_tag -> gene_name):
                  BPHL-Molecular/Amelia/reference/L43967.2.gff3
                  (NCBI L43967.2, M. genitalium G37)

  M. pneumoniae M129 (~688 CDS, only 57 with high-confidence calls):
    Essentiality: CRG-CNAG/killswitch/data/goldsets.csv (gene/class TSV)
                  (Lluch-Senar's group at CRG; high-confidence gold-set:
                  28 essential + 29 non-essential)
    GFF (locus_tag -> gene_name):
                  idoerg/MyDIG/.../NC_000912.gff (M. pneumoniae M129)

Mapping syn3A -> reference:
  Uses gene_name. syn3A's `gene_table.csv` has 368 of 496 rows with non-empty
  gene_name. We map by case-insensitive gene_name string match against the
  reference GFF's gene= qualifier. Coverage caveat: syn3A genes that are
  "Uncharacterized protein" (no gene name) won't map by this method --
  exactly the v15-FN uncharacterized bucket we ESPECIALLY want to rescue.
  Use the BLAST fallback for those (later patch).

Output: `cell_sim/data/cross_species_essentiality.csv` (schema documented
in `cell_sim/layer6_essentiality/conserved_essentiality_detector.py`).

Usage::

    python scripts/build_cross_species_from_github.py

Re-run when the upstream files update (rare).
"""
from __future__ import annotations

import csv
import re
import sys
import urllib.request
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SYN3A_GENE_TABLE = REPO_ROOT / "memory_bank" / "data" / "syn3a_gene_table.csv"
OUTPUT_CSV       = REPO_ROOT / "cell_sim" / "data" / "cross_species_essentiality.csv"
CACHE_DIR        = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "_xs_cache"

URLS = {
    "mg_essentiality_xlsx":
        "https://raw.githubusercontent.com/pauloburke/whole-cell-network/master/"
        "mg_build/mg_essentiality_basis.xlsx",
    "mg_gff3":
        "https://raw.githubusercontent.com/BPHL-Molecular/Amelia/master/"
        "reference/L43967.2.gff3",
    "mg_proteome_fasta":
        "https://raw.githubusercontent.com/berthoazeyeh/KEGG-DATA/main/"
        "organism_sequences/mge/protein_sequence.fasta",
    "mp_goldsets_tsv":
        "https://raw.githubusercontent.com/CRG-CNAG/killswitch/master/"
        "data/goldsets.csv",
    "mp_gff":
        "https://raw.githubusercontent.com/idoerg/MyDIG/master/"
        "mycoplasma_home/static/genomicFiles/new/"
        "Mycoplasma_pneumoniae_M129_uid57709/NC_000912.gff",
    "mp_proteome_fasta":
        "https://raw.githubusercontent.com/InfOmics/pangenes-review/master/"
        "singleton-less/pipeline/pangene_test_2/input/mycoplasma.g-1.fasta",
}

SYN3A_PROTEOME_FASTA = (
    REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "sequences.fasta"
)


def _download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 100:
        return
    print(f"    fetching {url}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "cell-sim/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r, open(dst, "wb") as f:
        f.write(r.read())


def _norm_name(s: str) -> str:
    """Lowercase, strip whitespace, split on ; , and take first token,
    strip trailing punctuation."""
    s = (s or "").strip().lower()
    if not s:
        return ""
    for sep in (";", ","):
        if sep in s:
            s = s.split(sep)[0]
    return s.strip(".,:")


_GFF_GENE_RE = re.compile(r"\b(?:Name|gene)=([^;]+)")
_GFF_LOCUS_RE = re.compile(r"\blocus_tag=([^;]+)")


def _parse_gff_gene_to_locus(gff_path: Path) -> dict[str, str]:
    """Parse a GFF3 file, return {gene_name_lowercase: locus_tag}.

    Picks the first locus_tag we see per gene_name (subsequent rows for
    the same locus are CDS lines that share the same gene+locus_tag).
    """
    out: dict[str, str] = {}
    with open(gff_path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            if parts[2] not in ("gene", "CDS"):
                continue
            attrs = parts[8]
            mg = _GFF_GENE_RE.search(attrs)
            ml = _GFF_LOCUS_RE.search(attrs)
            if not ml:
                continue
            locus = ml.group(1).strip()
            if mg:
                gene = _norm_name(mg.group(1))
                if gene and gene != locus.lower() and gene not in out:
                    out[gene] = locus
    return out


def _parse_mg_essentiality(xlsx_path: Path) -> dict[str, int]:
    """Read mg_essentiality_basis.xlsx Sheet1.

    Columns of interest (0-indexed):
        0: Gene (MG_001 ... MG_525) -- locus_tag
        1: Name -- product/description (used as a secondary key)
        4: Exp -- 'Y' (essential) / 'N' (non-essential) per Glass 2006

    Returns {locus_tag: 1 or 0}.
    """
    import openpyxl
    wb = openpyxl.load_workbook(str(xlsx_path))
    sh = wb["Sheet1"]
    rows = list(sh.iter_rows(values_only=True))
    out: dict[str, int] = {}
    for row in rows[1:]:
        if row is None or row[0] is None:
            continue
        locus = str(row[0]).strip()
        exp = row[4]
        if exp is None:
            continue
        if isinstance(exp, str):
            exp_v = exp.strip().upper()
            if exp_v == "Y":
                out[locus] = 1
            elif exp_v == "N":
                out[locus] = 0
    return out


def _parse_mp_goldsets(tsv_path: Path) -> dict[str, int]:
    """Read CRG-CNAG/killswitch goldsets.csv (TSV).

    Schema: gene\\tclass with class in {E, NE}.
    Returns {locus_tag: 1 or 0}.
    """
    out: dict[str, int] = {}
    with open(tsv_path) as f:
        next(f, None)   # header
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            locus = parts[0].strip()
            cls = parts[1].strip().upper()
            if cls == "E":
                out[locus] = 1
            elif cls in ("NE", "N"):
                out[locus] = 0
    return out


def _parse_fasta_with_locus_header(
    fasta_path: Path,
    locus_re: re.Pattern,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Read a FASTA where the header contains a recognizable locus tag.

    Returns ({locus_tag: sequence}, [(locus_tag, sequence) in file order]).
    """
    out: dict[str, str] = {}
    order: list[tuple[str, str]] = []
    cur_locus: Optional[str] = None
    cur_seq: list[str] = []
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_locus is not None and cur_seq:
                    seq = "".join(cur_seq).replace("*", "")
                    out[cur_locus] = seq
                    order.append((cur_locus, seq))
                m = locus_re.search(line)
                cur_locus = m.group(1) if m else None
                cur_seq = []
            else:
                if cur_locus is not None:
                    cur_seq.append(line)
        if cur_locus is not None and cur_seq:
            seq = "".join(cur_seq).replace("*", "")
            out[cur_locus] = seq
            order.append((cur_locus, seq))
    return out, order


def _read_syn3a_sequences(fasta_path: Path) -> dict[str, str]:
    """Read syn3A FASTA bundled at memory_bank/data/multiorg_essentiality/
    sequences.fasta. Header format: >syn3a|JCVISYN3A_0001|dnaA"""
    out: dict[str, str] = {}
    cur_locus: Optional[str] = None
    cur_seq: list[str] = []
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_locus is not None and cur_seq:
                    out[cur_locus] = "".join(cur_seq).replace("*", "")
                parts = line[1:].split("|")
                if len(parts) >= 2 and parts[0] == "syn3a":
                    cur_locus = parts[1]
                else:
                    cur_locus = None
                cur_seq = []
            else:
                if cur_locus is not None:
                    cur_seq.append(line)
        if cur_locus is not None and cur_seq:
            out[cur_locus] = "".join(cur_seq).replace("*", "")
    return out


def _run_blastp_best_hits(
    query_seqs: dict[str, str],
    subject_seqs: dict[str, str],
    evalue: float = 1e-5,
    threads: int = 4,
) -> dict[str, tuple[str, float]]:
    """Run blastp(query, subject_db) -> {query_id: (best_subject_id, pident)}.
    Requires `blastp` and `makeblastdb` on PATH (NCBI BLAST+)."""
    import shutil, subprocess, tempfile
    if shutil.which("blastp") is None or shutil.which("makeblastdb") is None:
        raise RuntimeError("blastp / makeblastdb not on PATH. "
                            "apt install ncbi-blast+ first.")
    with tempfile.TemporaryDirectory() as tdir:
        tdir_p = Path(tdir)
        q_fa = tdir_p / "query.faa"
        s_fa = tdir_p / "subject.faa"
        with open(q_fa, "w") as f:
            for qid, seq in query_seqs.items():
                if seq:
                    f.write(f">{qid}\n{seq}\n")
        with open(s_fa, "w") as f:
            for sid, seq in subject_seqs.items():
                if seq:
                    f.write(f">{sid}\n{seq}\n")
        subprocess.run(
            ["makeblastdb", "-in", str(s_fa), "-dbtype", "prot",
             "-out", str(tdir_p / "db")],
            check=True, capture_output=True,
        )
        out_tab = tdir_p / "hits.tsv"
        subprocess.run(
            ["blastp",
             "-query", str(q_fa),
             "-db",    str(tdir_p / "db"),
             "-evalue", str(evalue),
             "-outfmt", "6 qseqid sseqid pident evalue bitscore",
             "-max_target_seqs", "1",
             "-num_threads", str(threads),
             "-out", str(out_tab)],
            check=True, capture_output=True,
        )
        best: dict[str, tuple[str, float, float]] = {}
        with open(out_tab) as f:
            for line in f:
                parts = line.rstrip().split("\t")
                if len(parts) < 5:
                    continue
                q, s = parts[0], parts[1]
                pid, ev, bs = float(parts[2]), float(parts[3]), float(parts[4])
                prev = best.get(q)
                if prev is None or bs > prev[2]:
                    best[q] = (s, pid, bs)
        return {q: (s, pid) for q, (s, pid, _) in best.items()}


def _read_syn3a_gene_table() -> list[dict]:
    rows: list[dict] = []
    with open(SYN3A_GENE_TABLE) as f:
        for r in csv.DictReader(f):
            rows.append({
                "locus_tag": r["locus_tag"].strip(),
                "gene_name": r.get("gene_name", "").strip(),
            })
    return rows


def main() -> int:
    print("  cache dir:", CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. download ----
    print("\n  downloading sources...")
    paths = {key: CACHE_DIR / Path(url).name for key, url in URLS.items()}
    for key, url in URLS.items():
        _download(url, paths[key])
        size = paths[key].stat().st_size
        print(f"    {key:25s}  {size:>8d} bytes  ({paths[key]})")

    # ---- 2. parse ----
    print("\n  parsing references...")
    mg_essentiality = _parse_mg_essentiality(paths["mg_essentiality_xlsx"])
    print(f"    M. genitalium essentiality:  {len(mg_essentiality)} loci  "
          f"({sum(mg_essentiality.values())} essential)")
    mg_gene_to_locus = _parse_gff_gene_to_locus(paths["mg_gff3"])
    print(f"    M. genitalium gene -> locus: {len(mg_gene_to_locus)} mappings")

    mp_essentiality = _parse_mp_goldsets(paths["mp_goldsets_tsv"])
    print(f"    M. pneumoniae goldsets:       {len(mp_essentiality)} loci  "
          f"({sum(mp_essentiality.values())} essential)")
    mp_gene_to_locus = _parse_gff_gene_to_locus(paths["mp_gff"])
    print(f"    M. pneumoniae gene -> locus:  {len(mp_gene_to_locus)} mappings")

    # ---- 3a. read reference proteomes ----
    print("\n  reading reference proteomes ...")
    _MG_LOCUS_RE = re.compile(r"\b(MG_\d{3,4})\b")
    _MP_LOCUS_RE = re.compile(r"\b(MPN\d{3,4})\b")
    mg_seqs, _ = _parse_fasta_with_locus_header(
        paths["mg_proteome_fasta"], _MG_LOCUS_RE,
    )
    mp_seqs, _ = _parse_fasta_with_locus_header(
        paths["mp_proteome_fasta"], _MP_LOCUS_RE,
    )
    print(f"    M. genitalium FASTA: {len(mg_seqs)} entries")
    print(f"    M. pneumoniae FASTA: {len(mp_seqs)} entries")

    syn3a_seqs = _read_syn3a_sequences(SYN3A_PROTEOME_FASTA)
    print(f"    syn3A FASTA: {len(syn3a_seqs)} entries")

    # ---- 3b. BLAST syn3A -> each reference proteome (best hit, e-value <= 1e-5) ----
    print("\n  BLASTp syn3A vs M. genitalium ...")
    mg_blast = _run_blastp_best_hits(syn3a_seqs, mg_seqs, evalue=1e-5, threads=4)
    print(f"    {len(mg_blast)} syn3A genes have a BLAST hit (e<=1e-5)")

    print("  BLASTp syn3A vs M. pneumoniae ...")
    mp_blast = _run_blastp_best_hits(syn3a_seqs, mp_seqs, evalue=1e-5, threads=4)
    print(f"    {len(mp_blast)} syn3A genes have a BLAST hit (e<=1e-5)")

    # ---- 4. join with syn3A: gene_name match first, BLAST fallback ----
    print("\n  joining syn3A (gene_name first, BLAST fallback)...")
    syn3a = _read_syn3a_gene_table()
    n_with_name = sum(1 for r in syn3a if r["gene_name"])
    print(f"    syn3A rows: {len(syn3a)}  ({n_with_name} with gene_name)")

    n_mg_match = n_mp_match = 0
    n_mg_call  = n_mp_call  = 0
    n_mg_by_blast = n_mp_by_blast = 0
    n_either_ess = n_both_ess = 0
    rows: list[dict] = []
    for r in syn3a:
        lt = r["locus_tag"]
        gn = _norm_name(r["gene_name"])

        # M. genitalium: try gene_name, then BLAST
        mg_locus, mg_method, mg_pident = "", "", ""
        if gn:
            mg_locus = mg_gene_to_locus.get(gn, "")
            if mg_locus:
                mg_method = "gene_name"
        if not mg_locus and lt in mg_blast:
            blast_id, pid = mg_blast[lt]
            # blast subject id IS the locus tag (we BLAST'd against by-locus FASTA)
            mg_locus = blast_id
            mg_method = "blast"
            mg_pident = f"{pid:.1f}"
            n_mg_by_blast += 1
        mg_ess: Optional[int] = mg_essentiality.get(mg_locus) if mg_locus else None
        if mg_locus: n_mg_match += 1
        if mg_ess is not None: n_mg_call += 1

        # M. pneumoniae: try gene_name, then BLAST
        mp_locus, mp_method, mp_pident = "", "", ""
        if gn:
            mp_locus = mp_gene_to_locus.get(gn, "")
            if mp_locus:
                mp_method = "gene_name"
        if not mp_locus and lt in mp_blast:
            blast_id, pid = mp_blast[lt]
            mp_locus = blast_id
            mp_method = "blast"
            mp_pident = f"{pid:.1f}"
            n_mp_by_blast += 1
        mp_ess: Optional[int] = mp_essentiality.get(mp_locus) if mp_locus else None
        if mp_locus: n_mp_match += 1
        if mp_ess is not None: n_mp_call += 1

        if mg_ess == 1 or mp_ess == 1:
            n_either_ess += 1
        if mg_ess == 1 and mp_ess == 1:
            n_both_ess += 1

        rows.append({
            "locus_tag": lt,
            "mg_homolog_locus":  mg_locus,
            "mg_homolog_method": mg_method,
            "mg_pident":         mg_pident,
            "mg_essential":      str(mg_ess) if mg_ess is not None else "",
            "mp_homolog_locus":  mp_locus,
            "mp_homolog_method": mp_method,
            "mp_pident":         mp_pident,
            "mp_essential":      str(mp_ess) if mp_ess is not None else "",
        })

    # ---- 4. write ----
    print(f"\n  writing: {OUTPUT_CSV}")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "locus_tag",
        "mg_homolog_locus", "mg_homolog_method", "mg_pident", "mg_essential",
        "mp_homolog_locus", "mp_homolog_method", "mp_pident", "mp_essential",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\n  ===  summary  ===")
    print(f"  syn3A loci written:       {len(rows)}")
    print(f"    with M.gen homolog:     {n_mg_match}  "
          f"({n_mg_call} have ess call, {n_mg_by_blast} found by BLAST)")
    print(f"    with M.pne homolog:     {n_mp_match}  "
          f"({n_mp_call} have ess call, {n_mp_by_blast} found by BLAST)")
    print(f"    at least one ess call:  {n_either_ess}")
    print(f"    both refs ess agree:    {n_both_ess}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
