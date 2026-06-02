"""Bacterial essentiality inventory builder.

Fetches confirmed-reachable bacterial essentiality datasets from GitHub
raw URLs (the only outbound path the sandbox allows) and produces a
single normalized labels CSV plus an inventory record.

Sources verified reachable from sandbox (2026-06-02):

  syn3A (Breuer 2019)
    Local: memory_bank/data/syn3a_essentiality_breuer2019.csv

  M. genitalium G37 (Glass 2006 experimental column from Karr 2012 xlsx)
    URL: pauloburke/whole-cell-network/mg_build/mg_essentiality_basis.xlsx
    Sheet1, col 0 = locus, col 4 = 'Exp' Y/N

  M. pneumoniae M129 (CRG-CNAG / Lluch-Senar 2015 gold standard)
    URL: CRG-CNAG/killswitch/data/goldsets.csv
    TSV: gene\\tclass with class in {E, NE}

  S. pneumoniae TIGR4 (Zhu 2019)
    URL: dsurujon/ShinyOmics/data/metadata/T4_metadata.csv
    col 11 = 'Essential' (TRUE/FALSE)

  S. pneumoniae 19F (Zhu 2019)
    URL: dsurujon/ShinyOmics/data/metadata/19F_metadata.csv
    same schema

  M. tuberculosis H37Rv (MTBseq curation, derived from DeJesus et al.)
    URL: ngs-fzb/MTBseq_source/var/cat/MTB_Gene_Categories.txt
    TSV: locus_tag\\tcategory (essential / nonessential / epitope / repetitive)

  Ralstonia eutropha H16 (TraDIS, derived from Jahn lab)
    URL: m-jahn/R-notebook-ralstonia-proteome/data/output/essentiality_escher.csv
    CSV: locus_tag,essentiality (0=non, 1=essential, 2=conditional)

Output:
  memory_bank/data/multiorg_essentiality/labels.csv (organism, locus_tag,
      gene_name, essential, source, source_url)
  memory_bank/data/multiorg_essentiality/inventory.csv (per-source
      summary: organism, n_genes, n_essential, source, url, fetched_at)

Usage:
  python scripts/bacterial_essentiality_inventory.py
"""
from __future__ import annotations

import csv
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality"
CACHE_DIR = OUT_DIR / "_inventory_cache"
LABELS_CSV = OUT_DIR / "labels.csv"
INVENTORY_CSV = OUT_DIR / "inventory.csv"
SYN3A_BREUER = REPO_ROOT / "memory_bank" / "data" / "syn3a_essentiality_breuer2019.csv"

SOURCES = [
    {
        "organism": "syn3a",
        "kind": "local",
        "path": SYN3A_BREUER,
        "url": "memory_bank/data/syn3a_essentiality_breuer2019.csv",
        "format": "syn3a_breuer",
        "citation": "Breuer 2019 eLife",
        "strain": "JCVI-syn3A",
    },
    {
        "organism": "mgen",
        "kind": "remote",
        "url": ("https://raw.githubusercontent.com/pauloburke/whole-cell-network/"
                "master/mg_build/mg_essentiality_basis.xlsx"),
        "format": "karr_mg_xlsx",
        "citation": "Glass 2006 PNAS (via Karr 2012 Cell)",
        "strain": "M. genitalium G37",
    },
    {
        "organism": "mpne",
        "kind": "remote",
        "url": ("https://raw.githubusercontent.com/CRG-CNAG/killswitch/master/"
                "data/goldsets.csv"),
        "format": "killswitch_goldsets",
        "citation": "Lluch-Senar 2015 Mol Syst Biol",
        "strain": "M. pneumoniae M129",
    },
    {
        "organism": "spneT4",
        "kind": "remote",
        "url": ("https://raw.githubusercontent.com/dsurujon/ShinyOmics/master/"
                "data/metadata/T4_metadata.csv"),
        "format": "shinyomics_metadata",
        "citation": "Zhu 2019 (via ShinyOmics)",
        "strain": "S. pneumoniae TIGR4",
    },
    {
        "organism": "spne19F",
        "kind": "remote",
        "url": ("https://raw.githubusercontent.com/dsurujon/ShinyOmics/master/"
                "data/metadata/19F_metadata.csv"),
        "format": "shinyomics_metadata",
        "citation": "Zhu 2019 (via ShinyOmics)",
        "strain": "S. pneumoniae 19F",
    },
    {
        "organism": "mtub",
        "kind": "remote",
        "url": ("https://raw.githubusercontent.com/ngs-fzb/MTBseq_source/master/"
                "var/cat/MTB_Gene_Categories.txt"),
        "format": "mtbseq_categories",
        "citation": "DeJesus 2017 mBio (via MTBseq)",
        "strain": "M. tuberculosis H37Rv",
    },
    {
        "organism": "reut",
        "kind": "remote",
        "url": ("https://raw.githubusercontent.com/m-jahn/"
                "R-notebook-ralstonia-proteome/master/data/output/"
                "essentiality_escher.csv"),
        "format": "jahn_ralstonia",
        "citation": "Jahn lab Ralstonia TraDIS",
        "strain": "R. eutropha H16",
    },
]


def _download(url: str, dst: Path) -> int:
    if dst.exists() and dst.stat().st_size > 100:
        return dst.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "cell-sim/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r, open(dst, "wb") as f:
        f.write(r.read())
    return dst.stat().st_size


def _parse_syn3a_breuer(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            ess = 1 if r["essentiality"] in ("Essential", "Quasiessential") else 0
            rows.append({
                "locus_tag": r["locus_tag"], "gene_name": r.get("gene_name", ""),
                "essential": ess,
            })
    return rows


def _parse_karr_mg_xlsx(path: Path) -> list[dict]:
    import openpyxl
    wb = openpyxl.load_workbook(str(path))
    sh = wb["Sheet1"]
    rows: list[dict] = []
    for row in sh.iter_rows(values_only=True):
        if row is None or row[0] is None or not str(row[0]).startswith("MG_"):
            continue
        exp = row[4]
        if not isinstance(exp, str): continue
        ev = exp.strip().upper()
        if ev not in ("Y", "N"): continue
        rows.append({
            "locus_tag": str(row[0]).strip(),
            "gene_name": str(row[1] or "").split(",")[0].strip()[:30],
            "essential": 1 if ev == "Y" else 0,
        })
    return rows


def _parse_killswitch_goldsets(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        next(f, None)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2: continue
            cls = parts[1].strip().upper()
            if cls not in ("E", "NE", "N"): continue
            rows.append({
                "locus_tag": parts[0].strip(), "gene_name": "",
                "essential": 1 if cls == "E" else 0,
            })
    return rows


def _parse_shinyomics_metadata(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="latin-1") as f:
        for r in csv.DictReader(f):
            ess_str = (r.get("Essential") or "").strip().upper()
            if ess_str not in ("TRUE", "FALSE"): continue
            rows.append({
                "locus_tag": r["Gene"], "gene_name": r.get("Gene.Name", ""),
                "essential": 1 if ess_str == "TRUE" else 0,
            })
    return rows


def _parse_mtbseq_categories(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            if not line or line.startswith("#"): continue
            parts = line.strip().split("\t")
            if len(parts) < 2: continue
            lt = parts[0].strip()
            cat = parts[1].strip().lower()
            if not lt.startswith("Rv"): continue
            if cat == "essential":
                rows.append({"locus_tag": lt, "gene_name": "", "essential": 1})
            elif cat == "nonessential":
                rows.append({"locus_tag": lt, "gene_name": "", "essential": 0})
            # skip epitope / repetitive
    return rows


def _parse_jahn_ralstonia(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                v = int(float(r["essentiality"]))
            except (ValueError, KeyError):
                continue
            # 0 = non, 1 = essential, 2 = conditional (skip conditional for now)
            if v == 0:
                rows.append({"locus_tag": r["locus_tag"], "gene_name": "", "essential": 0})
            elif v == 1:
                rows.append({"locus_tag": r["locus_tag"], "gene_name": "", "essential": 1})
    return rows


PARSERS = {
    "syn3a_breuer":         _parse_syn3a_breuer,
    "karr_mg_xlsx":         _parse_karr_mg_xlsx,
    "killswitch_goldsets":  _parse_killswitch_goldsets,
    "shinyomics_metadata":  _parse_shinyomics_metadata,
    "mtbseq_categories":    _parse_mtbseq_categories,
    "jahn_ralstonia":       _parse_jahn_ralstonia,
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    all_rows: list[dict] = []
    inventory: list[dict] = []

    for src in SOURCES:
        print(f"\n=== {src['organism']} ({src['strain']}) ===")
        try:
            if src["kind"] == "remote":
                fname = Path(src["url"]).name
                local = CACHE_DIR / f"{src['organism']}__{fname}"
                size = _download(src["url"], local)
                print(f"   fetched {size} bytes -> {local.name}")
                path = local
            else:
                path = src["path"]
                size = path.stat().st_size if path.exists() else 0

            parser = PARSERS[src["format"]]
            rows = parser(path)
            n_ess = sum(r["essential"] for r in rows)
            print(f"   parsed {len(rows)} loci ({n_ess} essential)")
            for r in rows:
                all_rows.append({
                    "organism": src["organism"],
                    "locus_tag": r["locus_tag"],
                    "gene_name": r["gene_name"],
                    "essential": r["essential"],
                    "source": src["citation"],
                    "source_url": src["url"],
                })
            inventory.append({
                "organism": src["organism"],
                "strain": src["strain"],
                "n_genes": len(rows),
                "n_essential": n_ess,
                "n_nonessential": len(rows) - n_ess,
                "ess_rate": round(n_ess / max(len(rows), 1), 3),
                "format": src["format"],
                "source": src["citation"],
                "url": src["url"],
                "bytes": size,
                "fetched_at": ts,
            })
        except Exception as exc:
            print(f"   FAIL: {type(exc).__name__}: {exc}")
            inventory.append({
                "organism": src["organism"], "strain": src["strain"],
                "n_genes": 0, "n_essential": 0, "n_nonessential": 0,
                "ess_rate": 0.0, "format": src["format"],
                "source": src["citation"], "url": src["url"],
                "bytes": 0, "fetched_at": ts,
                "error": f"{type(exc).__name__}: {exc}",
            })

    # Write labels.csv
    print(f"\n=== writing {LABELS_CSV} ===")
    with open(LABELS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "organism", "locus_tag", "gene_name", "essential",
            "source", "source_url",
        ])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"   {len(all_rows)} total labeled genes written")

    # Write inventory.csv
    print(f"\n=== writing {INVENTORY_CSV} ===")
    fieldnames = ["organism", "strain", "n_genes", "n_essential",
                   "n_nonessential", "ess_rate", "format", "source",
                   "url", "bytes", "fetched_at", "error"]
    with open(INVENTORY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in inventory:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Summary
    print(f"\n===  inventory summary  ===")
    print(f"  {'organism':<10s}  {'n':>6s}  {'ess':>5s}  {'rate':>5s}  source")
    for r in inventory:
        print(f"  {r['organism']:<10s}  {r['n_genes']:>6d}  "
              f"{r['n_essential']:>5d}  {r['ess_rate']:>5.2f}  {r['source']}")
    total = sum(r["n_genes"] for r in inventory)
    total_ess = sum(r["n_essential"] for r in inventory)
    print(f"  {'TOTAL':<10s}  {total:>6d}  {total_ess:>5d}  "
          f"{total_ess/max(total,1):>5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
