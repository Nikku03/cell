"""Cross-organism codon (H2) + leak-free operon (H3) feature builder.

The E. coli verifier (verify_sequence_hypotheses.py) showed:
  H2 CAI:        Cohen's d = +1.45, AUC = 0.805   (STRONG)
  H3 operon:     Cohen's d = +0.30..2.32          (STRONG, partly leaky)
  H1 promoter:   Cohen's d < 0.1                  (DEAD)

This script scales H2 + the leak-free portion of H3 to all
organisms for which we have a RefSeq genome accession + a clean
locus_tag join. Output:

  codon_features.csv  (organism, locus_tag,
    cai, gc, gc3, cds_length,             # H2 codon
    intergenic_prev, intergenic_next,     # H3 leak-free operon
    same_strand_prev, same_strand_next)

The pipeline per organism:
  1. Download FNA + GFF from NCBI (cached)
  2. Parse, extract CDS sequences in genome order
  3. Build CAI weights from ribosomal-protein reference set
  4. Compute per-gene features
  5. Join to labels via locus_tag -> old_locus_tag -> gene_name
  6. Emit rows for matched genes

Config: CSV with columns
  organism, refseq_accession, assembly_name
plus optional notes/skip_reason for orgs we skip.

Pre-populated with ~14 high-confidence orgs covering ~50K rows.
Extend the config to cover more orgs as you verify accessions.
"""
from __future__ import annotations
import argparse, csv, gzip, math, re, sys, urllib.request, urllib.error
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality"
CACHE_DIR = DATA_DIR / "_seq_cache"
DEFAULT_CONFIG = DATA_DIR / "codon_features_config.csv"
DEFAULT_OUT    = DATA_DIR / "codon_features.csv"
LABELS_CSV     = DATA_DIR / "labels.csv"

# Pre-populated organism -> (accession, assembly_name).
# Picked for high-confidence locus_tag join: each org's BERIL
# locusId scheme matches the assembly's RefSeq locus_tag scheme
# (verified from the diagnostic in this session).
SEED_CONFIG = [
    # (organism_in_labels, accession, assembly_name, locus_pattern_note)
    ("ecoli_BW25113_tradis",   "GCF_000750555.1", "ASM75055v1",   "BW25113_NNNN -- match via gene_name fallback"),
    ("mtub",                   "GCF_000195955.2", "ASM19595v2",   "Rv_NNNN direct match"),
    ("mgen",                   "GCF_000027325.1", "ASM2732v1",    "MG_NNN direct"),
    ("mpne",                   "GCF_000027345.1", "ASM2734v1",    "MPN_NNN direct"),
    ("saur_NCTC8325_biotradis","GCF_000013425.1", "ASM1342v1",    "SAOUHSC_NNNNN direct"),
    ("spneT4",                 "GCF_000006885.1", "ASM688v1",     "SP_NNNN direct"),
    ("syn3a",                  "GCA_000968075.1", "ASM96807v1",   "JCVISYN3A_NNNN GenBank only"),
    ("reut",                   "GCF_000009285.1", "ASM928v1",     "Cupriavidus necator H16"),
    ("reut_full",              "GCF_000009285.1", "ASM928v1",     "same"),
    ("beril_Putida",           "GCF_000007565.2", "ASM756v2",     "PP_NNNN direct"),
    ("beril_BFirm",            "GCF_000019365.1", "ASM1936v1",    "BPHYT_RSNNNNN direct"),
    ("beril_SynE",             "GCF_000012525.1", "ASM1252v1",    "Synpcc7942_NNNN direct"),
    ("beril_Smeli",            "GCF_000006965.1", "ASM696v1",     "SMc_NNNNN may need gene_name"),
    ("beril_Caulo",            "GCF_000006905.1", "ASM690v1",     "CC_NNNN direct"),
    ("beril_RalstoniaGMI1000", "GCF_000009125.1", "ASM912v1",     "RSc/RSp NNNN -- BERIL uses RS_RSNNNNN"),
    ("beril_MR1",              "GCF_000146165.2", "ASM14616v2",   "Shewanella oneidensis MR-1 SO_NNNN -- but BERIL uses numeric, will likely fail"),
    ("beril_DvH",              "GCF_000195755.1", "ASM19575v1",   "D. vulgaris Hildenborough DVU_NNNN -- BERIL uses numeric, will likely fail"),
    ("beril_Btheta",           "GCF_000011065.1", "ASM1106v1",    "B. theta BT_NNNN -- BERIL uses numeric, will likely fail"),
]


# ---------- NCBI helpers ----------
NCBI_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/genomes/all"


def ncbi_genome_urls(accession: str, assembly_name: str) -> tuple[str, str]:
    a = accession
    prefix = a.split("_")[0]                # GCF or GCA
    digits = a.split("_")[1]                 # 000750555.1
    p1, p2, p3 = digits[:3], digits[3:6], digits[6:9]
    base = f"{NCBI_FTP_BASE}/{prefix}/{p1}/{p2}/{p3}/{a}_{assembly_name}"
    return (f"{base}/{a}_{assembly_name}_genomic.fna.gz",
            f"{base}/{a}_{assembly_name}_genomic.gff.gz")


def download_cached(url: str, dst: Path) -> Path | None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 100:
        return dst
    try:
        urllib.request.urlretrieve(url, dst)
        return dst
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"    DOWNLOAD FAIL: {url} -> {e}")
        if dst.exists(): dst.unlink()
        return None


# ---------- parsing ----------
def parse_fasta_gz(path: Path) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    cur = None
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                cur = line[1:].split()[0]; out[cur] = []
            else:
                out[cur].append(line.strip().upper())
    return {k: "".join(v) for k, v in out.items()}


def parse_gff_gz(path: Path) -> list[dict]:
    feats = []
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9: continue
            seqid, src, ftype, start, end, score, strand, phase, attr = parts
            if ftype != "CDS": continue
            a = {}
            for kv in attr.split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1); a[k] = v
            feats.append({
                "seqid":         seqid,
                "start":         int(start),
                "end":           int(end),
                "strand":        strand,
                "locus_tag":     a.get("locus_tag", ""),
                "old_locus_tag": a.get("old_locus_tag", ""),
                "gene":          a.get("gene", ""),
            })
    return feats


REV_COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(seq: str) -> str:
    return seq.translate(REV_COMP)[::-1]


def extract_cds(seq: str, start: int, end: int, strand: str) -> str:
    s = seq[start-1:end]
    return s if strand == "+" else revcomp(s)


# ---------- codon features ----------
GENETIC_CODE = {
  "TTT":"F","TTC":"F","TTA":"L","TTG":"L",
  "CTT":"L","CTC":"L","CTA":"L","CTG":"L",
  "ATT":"I","ATC":"I","ATA":"I","ATG":"M",
  "GTT":"V","GTC":"V","GTA":"V","GTG":"V",
  "TCT":"S","TCC":"S","TCA":"S","TCG":"S",
  "CCT":"P","CCC":"P","CCA":"P","CCG":"P",
  "ACT":"T","ACC":"T","ACA":"T","ACG":"T",
  "GCT":"A","GCC":"A","GCA":"A","GCG":"A",
  "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*",
  "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
  "AAT":"N","AAC":"N","AAA":"K","AAG":"K",
  "GAT":"D","GAC":"D","GAA":"E","GAG":"E",
  "TGT":"C","TGC":"C","TGA":"*","TGG":"W",
  "CGT":"R","CGC":"R","CGA":"R","CGG":"R",
  "AGT":"S","AGC":"S","AGA":"R","AGG":"R",
  "GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}
AA_TO_CODONS: dict[str, list[str]] = defaultdict(list)
for cod, aa in GENETIC_CODE.items(): AA_TO_CODONS[aa].append(cod)


def gc_content(seq: str) -> float:
    s = seq.upper()
    gc = s.count("G") + s.count("C")
    at = s.count("A") + s.count("T")
    n = gc + at
    return gc / n if n else float("nan")


def gc3_content(cds: str) -> float:
    if len(cds) < 3: return float("nan")
    third = [cds[i+2] for i in range(0, len(cds) - 2, 3)]
    if not third: return float("nan")
    gc = sum(1 for b in third if b in "GC")
    n  = sum(1 for b in third if b in "ACGT")
    return gc / n if n else float("nan")


def build_cai_weights(reference_cdss: list[str]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for cds in reference_cdss:
        for i in range(0, len(cds) - 2, 3):
            c = cds[i:i+3]
            if c in GENETIC_CODE: counts[c] += 1
    w: dict[str, float] = {}
    for aa, codons in AA_TO_CODONS.items():
        if aa in ("*", "M", "W"): continue
        max_c = max(counts.get(c, 0) for c in codons)
        if max_c == 0:
            for c in codons: w[c] = 1.0
        else:
            for c in codons:
                w[c] = (counts.get(c, 0) + 1) / (max_c + 1)
    return w


def cai_of(cds: str, weights: dict[str, float]) -> float:
    total, n = 0.0, 0
    for i in range(0, len(cds) - 2, 3):
        c = cds[i:i+3]
        if c in weights:
            total += math.log(weights[c]); n += 1
    return math.exp(total / n) if n else float("nan")


# ---------- per-organism worker ----------
def process_org(org: str, accession: str, assembly: str,
                labels_for_org: "pd.DataFrame", cache_dir: Path) -> list[dict]:
    """Returns a list of feature rows for matched genes, or [] on failure."""
    fna_url, gff_url = ncbi_genome_urls(accession, assembly)
    fna = download_cached(fna_url, cache_dir / f"{accession}_genomic.fna.gz")
    gff = download_cached(gff_url, cache_dir / f"{accession}_genomic.gff.gz")
    if not fna or not gff:
        print(f"    skipping {org}: download failed")
        return []
    seqs = parse_fasta_gz(fna)
    feats = parse_gff_gz(gff)
    if not feats:
        print(f"    skipping {org}: no CDS features in GFF"); return []
    # extract CDS + sort by genome position
    rows = []
    for f in feats:
        if f["seqid"] not in seqs: continue
        cds = extract_cds(seqs[f["seqid"]], f["start"], f["end"], f["strand"])
        if len(cds) < 30 or len(cds) % 3 != 0: continue
        rows.append({**f, "cds": cds})
    rows.sort(key=lambda r: (r["seqid"], r["start"]))

    # CAI reference set: ribosomal-protein gene patterns
    ref = [r["cds"] for r in rows
           if r["gene"] and re.match(r"^rp[slm][A-Z]?$", r["gene"], re.IGNORECASE)]
    if len(ref) < 20:
        ref = [r["cds"] for r in rows
               if r["gene"] and r["gene"][:3].lower() in ("rpl","rps","rpm")]
    weights = build_cai_weights(ref) if len(ref) >= 5 else None
    if weights is None:
        print(f"    {org}: only {len(ref)} ribosomal refs -- using flat CAI");
        weights = {c: 1.0 for c in GENETIC_CODE if GENETIC_CODE[c] not in ("*","M","W")}

    # match labels via locus_tag / old_locus_tag / gene_name (case-insensitive)
    import pandas as pd
    lt_set    = set(labels_for_org.locus_tag.astype(str))
    gene_to_lt: dict[str, str] = {}
    if "gene_name" in labels_for_org.columns:
        for lt, gn in zip(labels_for_org.locus_tag.astype(str),
                            labels_for_org.gene_name):
            if pd.notna(gn) and str(gn).strip():
                gene_to_lt[str(gn).strip().lower()] = lt

    matched = 0
    via = Counter()
    for i, r in enumerate(rows):
        beril_lt = None
        if r["locus_tag"] in lt_set:
            beril_lt = r["locus_tag"]; via["locus_tag"] += 1
        elif r["old_locus_tag"] in lt_set:
            beril_lt = r["old_locus_tag"]; via["old_locus_tag"] += 1
        elif r["gene"] and r["gene"].strip().lower() in gene_to_lt:
            beril_lt = gene_to_lt[r["gene"].strip().lower()]; via["gene_name"] += 1
        r["_beril_lt"] = beril_lt
        if beril_lt is not None: matched += 1

    # operon features (positional)
    for i, r in enumerate(rows):
        prev_r = rows[i-1] if i > 0 and rows[i-1]["seqid"] == r["seqid"] else None
        next_r = rows[i+1] if i+1 < len(rows) and rows[i+1]["seqid"] == r["seqid"] else None
        r["intergenic_prev"]  = (r["start"] - prev_r["end"] - 1) if prev_r else None
        r["intergenic_next"]  = (next_r["start"] - r["end"] - 1) if next_r else None
        r["same_strand_prev"] = int(prev_r["strand"] == r["strand"]) if prev_r else None
        r["same_strand_next"] = int(next_r["strand"] == r["strand"]) if next_r else None

    out = []
    for r in rows:
        if r["_beril_lt"] is None: continue
        out.append({
            "organism":         org,
            "locus_tag":        r["_beril_lt"],
            "cai":              cai_of(r["cds"], weights),
            "gc":               gc_content(r["cds"]),
            "gc3":              gc3_content(r["cds"]),
            "cds_length":       len(r["cds"]),
            "intergenic_prev":  r["intergenic_prev"],
            "intergenic_next":  r["intergenic_next"],
            "same_strand_prev": r["same_strand_prev"],
            "same_strand_next": r["same_strand_next"],
        })
    print(f"    {org}: {matched}/{len(rows)} CDSes matched "
          f"({100*matched/max(len(rows),1):.0f}%, via {dict(via)})")
    return out


# ---------- main ----------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config",     type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--labels-csv", type=Path, default=LABELS_CSV)
    p.add_argument("--cache-dir",  type=Path, default=CACHE_DIR)
    p.add_argument("--out",        type=Path, default=DEFAULT_OUT)
    p.add_argument("--write-seed-config", action="store_true",
                   help="Write the built-in SEED_CONFIG to --config and exit.")
    args = p.parse_args()

    if args.write_seed_config:
        args.config.parent.mkdir(parents=True, exist_ok=True)
        with open(args.config, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["organism","refseq_accession","assembly_name","note"])
            for row in SEED_CONFIG:
                w.writerow(row)
        print(f"wrote seed config to {args.config}")
        return 0

    try:
        import pandas as pd
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1

    # auto-create config from seed if missing
    if not args.config.exists():
        print(f"  config not found at {args.config}, writing seed config")
        args.config.parent.mkdir(parents=True, exist_ok=True)
        with open(args.config, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["organism","refseq_accession","assembly_name","note"])
            for row in SEED_CONFIG:
                w.writerow(row)

    cfg = pd.read_csv(args.config)
    print(f"Config: {len(cfg)} orgs in {args.config}")
    labels = pd.read_csv(args.labels_csv)
    print(f"Labels: {len(labels):,} rows, {labels.organism.nunique()} orgs")

    all_rows: list[dict] = []
    for _, c in cfg.iterrows():
        org = c["organism"]
        lab_for_org = labels[labels.organism == org]
        if len(lab_for_org) == 0:
            print(f"  SKIP {org}: 0 label rows"); continue
        print(f"\n  processing {org}  ({len(lab_for_org)} labelled genes)")
        out = process_org(org, c["refseq_accession"], c["assembly_name"],
                          lab_for_org, args.cache_dir)
        all_rows.extend(out)

    if not all_rows:
        print("ERROR: no rows produced", file=sys.stderr); return 1
    df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df):,} rows to {args.out}")
    print(f"  organism coverage: "
          f"{df.organism.nunique()} of {cfg.organism.nunique()} configured")
    print(f"  feature summary (across all matched rows):")
    for c in ["cai","gc","gc3","cds_length","intergenic_prev",
              "same_strand_prev"]:
        if c in df.columns:
            s = df[c].dropna()
            if len(s):
                print(f"    {c:<20s}  n={len(s):>6d}  mean={s.mean():>9.3f}  "
                      f"std={s.std():>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
