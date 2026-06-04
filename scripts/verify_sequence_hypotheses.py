"""Verify three sequence-derived essentiality hypotheses on a single
organism (default: E. coli K12 MG1655).

Three testable hypotheses, restated precisely:

  H1 (promoter strength): essential genes have stronger / more
      constitutive sigma-70 promoters upstream. Operationally,
      higher motif-match scores for the -10 box (TATAAT) and -35
      box (TTGACA) in the -100..0 region relative to the start
      codon.

  H2 (codon optimization): essential genes show stronger codon-usage
      bias toward the cell's preferred codons. Operationally, higher
      Codon Adaptation Index (CAI) computed against a reference set
      of ribosomal proteins, plus higher GC3 in GC-rich genomes
      (or lower in AT-rich, depending on the organism).

  H3 (operon context): essential genes are more often co-transcribed
      with other essential genes. Operationally, shorter intergenic
      distance to upstream/downstream neighbor + same-strand
      neighbor + higher fraction of operonic neighbors that are
      themselves essential.

Per feature, we compute:
  - mean +/- std for essential vs non-essential
  - Mann-Whitney U p-value (non-parametric, doesn't assume normality)
  - Cohen's d effect size (standardised mean difference)
  - AUC of the feature as a univariate classifier
  - log10-odds of essentiality in the top vs bottom decile

Verdict rules (per hypothesis):
  STRONG    : Cohen's d >= 0.5 AND p < 1e-10 AND AUC >= 0.65
  MODERATE  : Cohen's d >= 0.3 AND p < 1e-5  AND AUC >= 0.58
  WEAK/NONE : otherwise

Run:
  python scripts/verify_sequence_hypotheses.py
"""
from __future__ import annotations
import argparse, csv, gzip, io, json, math, re, sys, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality"
CACHE_DIR = DATA_DIR / "_seq_cache"
LABELS_CSV = DATA_DIR / "labels.csv"

# Defaults target E. coli K12 MG1655 (Keio b-numbers).
# We use MG1655 (GCF_000005845.2) because Keio BW25113 inherits its
# b-number locus tags and shares all annotated CDSes.
DEFAULT_ACCESSION = "GCF_000750555.1"
DEFAULT_ASSEMBLY  = "ASM75055v1"
DEFAULT_ORGANISM_KEY = "ecoli_BW25113_tradis"

NCBI_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/genomes/all"

# Sigma-70 consensus motifs (from Harley & Reynolds 1987 / many texts)
MINUS_10_CONSENSUS = "TATAAT"
MINUS_35_CONSENSUS = "TTGACA"
PROMOTER_SEARCH_WINDOW = 120  # bp upstream of start codon to scan


# ---------- helpers ----------
def ncbi_genome_urls(accession: str, assembly_name: str) -> tuple[str, str]:
    """Return (fna_url, gff_url) for an NCBI Datasets-style accession."""
    a = accession  # e.g. "GCF_000005845.2"
    p1, p2, p3 = a.split("_")[1][:3], a.split("_")[1][3:6], a.split("_")[1][6:9]
    base = f"{NCBI_FTP_BASE}/{a.split('_')[0]}/{p1}/{p2}/{p3}/{a}_{assembly_name}"
    return (f"{base}/{a}_{assembly_name}_genomic.fna.gz",
            f"{base}/{a}_{assembly_name}_genomic.gff.gz")


def download_cached(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 100:
        print(f"  cached: {dst.name}  ({dst.stat().st_size/1e6:.1f} MB)")
        return dst
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, dst)
    print(f"  -> {dst.name}  ({dst.stat().st_size/1e6:.1f} MB)")
    return dst


def parse_fasta_gz(path: Path) -> dict[str, str]:
    """Returns {seqid: sequence} uppercase, no whitespace."""
    out: dict[str, list[str]] = {}
    cur = None
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                cur = line[1:].split()[0]
                out[cur] = []
            else:
                out[cur].append(line.strip().upper())
    return {k: "".join(v) for k, v in out.items()}


def parse_gff_gz(path: Path) -> list[dict]:
    """Return one dict per CDS feature.
    Keys: seqid, start (1-based inclusive), end, strand, locus_tag,
           old_locus_tag, gene, product.
    """
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
                    k, v = kv.split("=", 1)
                    a[k] = v
            feats.append({
                "seqid":          seqid,
                "start":          int(start),
                "end":            int(end),
                "strand":         strand,
                "locus_tag":      a.get("locus_tag", ""),
                "old_locus_tag":  a.get("old_locus_tag", ""),
                "gene":           a.get("gene", ""),
                "product":        a.get("product", ""),
            })
    return feats


REV_COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(seq: str) -> str:
    return seq.translate(REV_COMP)[::-1]


def extract_cds(seq: str, start: int, end: int, strand: str) -> str:
    """GFF coordinates are 1-based inclusive. Python slices are 0-based,
    end-exclusive. So seq[start-1:end] is the forward CDS."""
    s = seq[start-1:end]
    return s if strand == "+" else revcomp(s)


def extract_upstream(seq: str, start: int, end: int, strand: str,
                     window: int) -> str:
    """Window bp immediately 5' of the CDS (i.e. promoter-bearing region)."""
    if strand == "+":
        u_start = max(0, start - 1 - window)
        return seq[u_start:start-1]
    else:
        u_end = min(len(seq), end + window)
        return revcomp(seq[end:u_end])


# ---------- feature computations ----------
def gc_content(seq: str) -> float:
    if not seq: return float("nan")
    s = seq.upper()
    gc = s.count("G") + s.count("C")
    at = s.count("A") + s.count("T")
    n = gc + at
    return gc / n if n else float("nan")


def gc3_content(cds: str) -> float:
    if len(cds) < 3: return float("nan")
    codons = [cds[i:i+3] for i in range(0, len(cds) - len(cds) % 3, 3)]
    third = [c[2] for c in codons if len(c) == 3]
    if not third: return float("nan")
    gc = sum(1 for b in third if b in "GC")
    n  = sum(1 for b in third if b in "ACGT")
    return gc / n if n else float("nan")


# Standard genetic code (codon -> AA, '*' = stop)
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
for cod, aa in GENETIC_CODE.items():
    AA_TO_CODONS[aa].append(cod)


def build_cai_weights(reference_cdss: list[str]) -> dict[str, float]:
    """Sharp & Li 1987 Codon Adaptation Index weights.

    For each codon, w = (count of this codon) / (max count among
    synonyms for the same AA). Skip stop, Met, Trp (single codons).
    Use a Laplace +1 smoother to avoid log(0).
    """
    counts: Counter[str] = Counter()
    for cds in reference_cdss:
        for i in range(0, len(cds) - 2, 3):
            cod = cds[i:i+3]
            if cod in GENETIC_CODE:
                counts[cod] += 1
    w: dict[str, float] = {}
    for aa, codons in AA_TO_CODONS.items():
        if aa in ("*", "M", "W"): continue
        max_c = max(counts.get(c, 0) for c in codons)
        if max_c == 0:
            for c in codons:
                w[c] = 1.0
        else:
            for c in codons:
                w[c] = (counts.get(c, 0) + 1) / (max_c + 1)
    return w


def cai_of(cds: str, weights: dict[str, float]) -> float:
    """Geometric mean of w(codon) over non-Met/Trp/stop codons."""
    total = 0.0
    n = 0
    for i in range(0, len(cds) - 2, 3):
        cod = cds[i:i+3]
        if cod in weights:
            total += math.log(weights[cod])
            n += 1
    if n == 0: return float("nan")
    return math.exp(total / n)


def motif_score(window_seq: str, motif: str) -> float:
    """Best Hamming-similarity match of motif in window_seq.
    Returns score in [0, 1] = (matches / motif length) at best position.
    """
    if not window_seq or len(window_seq) < len(motif):
        return float("nan")
    L = len(motif)
    best = 0
    for i in range(len(window_seq) - L + 1):
        m = sum(1 for a, b in zip(window_seq[i:i+L], motif) if a == b)
        if m > best: best = m
    return best / L


# ---------- statistical tests ----------
def mann_whitney_u(a: list[float], b: list[float]) -> float:
    """Returns two-sided p-value via normal approximation."""
    import numpy as np
    a = np.array([x for x in a if not math.isnan(x)])
    b = np.array([x for x in b if not math.isnan(x)])
    if len(a) < 5 or len(b) < 5: return float("nan")
    combined = np.concatenate([a, b])
    ranks = combined.argsort().argsort() + 1
    ra = ranks[:len(a)].sum()
    n1, n2 = len(a), len(b)
    u1 = ra - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u  = min(u1, u2)
    mu = n1 * n2 / 2
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sigma == 0: return 1.0
    z = (u - mu) / sigma
    # two-sided normal p
    p = math.erfc(abs(z) / math.sqrt(2))
    return p


def cohens_d(a: list[float], b: list[float]) -> float:
    import numpy as np
    a = np.array([x for x in a if not math.isnan(x)])
    b = np.array([x for x in b if not math.isnan(x)])
    if len(a) < 2 or len(b) < 2: return float("nan")
    pooled = math.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
                        / (len(a) + len(b) - 2))
    if pooled == 0: return float("nan")
    return (a.mean() - b.mean()) / pooled


def auc_univariate(scores: list[float], labels: list[int]) -> float:
    import numpy as np
    s = np.array(scores); y = np.array(labels)
    mask = ~np.isnan(s)
    s, y = s[mask], y[mask]
    if y.sum() == 0 or y.sum() == len(y): return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    n_pos = int(y_sorted.sum())
    n_neg = int(len(y_sorted) - n_pos)
    rank_sum = np.where(y_sorted == 1)[0].sum()  # 0-indexed positions
    # AUC = (sum of ranks for positives - n_pos*(n_pos-1)/2) / (n_pos*n_neg)
    # Convert 0-indexed -> 1-indexed ranks
    rank_sum_1 = rank_sum + n_pos
    auc = (rank_sum_1 - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return 1 - auc  # since we sorted desc, flip


def odds_top_vs_bot(scores: list[float], labels: list[int], pct=10) -> float:
    import numpy as np
    s = np.array(scores); y = np.array(labels)
    mask = ~np.isnan(s)
    s, y = s[mask], y[mask]
    if len(s) < 100: return float("nan")
    top_t = np.percentile(s, 100 - pct)
    bot_t = np.percentile(s, pct)
    top = y[s >= top_t]
    bot = y[s <= bot_t]
    p_top = top.mean() if len(top) else 0
    p_bot = bot.mean() if len(bot) else 0
    if p_bot == 0 or p_top == 0: return float("nan")
    return math.log10((p_top / (1 - p_top + 1e-9)) /
                       (p_bot / (1 - p_bot + 1e-9)))


def report(name: str, ess: list[float], noness: list[float],
           scores: list[float], labels: list[int],
           higher_is_essential: bool = True) -> dict:
    p = mann_whitney_u(ess, noness)
    d = cohens_d(ess, noness)
    # if hypothesis predicts essential > non-essential, sign should be positive
    if not higher_is_essential:
        d = -d
        scores = [-s if not math.isnan(s) else s for s in scores]
    auc = auc_univariate(scores, labels)
    odds = odds_top_vs_bot(scores, labels)
    import numpy as np
    e_arr = np.array([x for x in ess if not math.isnan(x)])
    n_arr = np.array([x for x in noness if not math.isnan(x)])
    e_mean = float(e_arr.mean()) if len(e_arr) else float("nan")
    n_mean = float(n_arr.mean()) if len(n_arr) else float("nan")
    # verdict
    if not math.isnan(d) and abs(d) >= 0.5 and p < 1e-10 and auc >= 0.65:
        v = "STRONG"
    elif not math.isnan(d) and abs(d) >= 0.3 and p < 1e-5 and auc >= 0.58:
        v = "MODERATE"
    else:
        v = "WEAK/NONE"
    print(f"  {name:<28s}  "
          f"ess={e_mean:>7.4f}  non={n_mean:>7.4f}  "
          f"d={d:>+5.2f}  p={p:>8.1e}  AUC={auc:>5.3f}  "
          f"logOdds={odds:>+5.2f}  -> {v}")
    return {"feature":name, "ess_mean":e_mean, "non_mean":n_mean,
            "cohen_d":d, "mann_whitney_p":p, "auc":auc,
            "log_odds_top_vs_bot_decile":odds, "verdict":v,
            "n_ess":int(len(e_arr)), "n_non":int(len(n_arr))}


# ---------- main ----------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--accession",    default=DEFAULT_ACCESSION)
    p.add_argument("--assembly",     default=DEFAULT_ASSEMBLY)
    p.add_argument("--organism-key", default=DEFAULT_ORGANISM_KEY,
                   help="organism column value in labels.csv to subset on")
    p.add_argument("--labels-csv",   type=Path, default=LABELS_CSV)
    p.add_argument("--cache-dir",    type=Path, default=CACHE_DIR)
    p.add_argument("--out-csv",      type=Path,
                   default=DATA_DIR / "sequence_features_verify.csv")
    p.add_argument("--out-json",     type=Path,
                   default=DATA_DIR / "sequence_hypothesis_verify.json")
    args = p.parse_args()

    try:
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1

    # ---- 1. download ----
    print(f"Stage 1: download {args.accession}/{args.assembly}")
    fna_url, gff_url = ncbi_genome_urls(args.accession, args.assembly)
    fna_path = download_cached(fna_url, args.cache_dir / f"{args.accession}_genomic.fna.gz")
    gff_path = download_cached(gff_url, args.cache_dir / f"{args.accession}_genomic.gff.gz")

    # ---- 2. parse ----
    print(f"\nStage 2: parse FASTA + GFF")
    seqs = parse_fasta_gz(fna_path)
    print(f"  contigs: {len(seqs)}  total bp: {sum(len(s) for s in seqs.values()):,}")
    feats = parse_gff_gz(gff_path)
    print(f"  CDS features: {len(feats):,}")
    sample_lt = next((f for f in feats if f["old_locus_tag"]), feats[0])
    print(f"  sample feature: locus_tag={sample_lt['locus_tag']}  "
          f"old_locus_tag={sample_lt['old_locus_tag']}  gene={sample_lt['gene']}")

    # ---- 3. extract CDS + upstream per feature ----
    print(f"\nStage 3: extract CDS + upstream sequences")
    rows: list[dict] = []
    skipped_no_seq = skipped_no_seqid = 0
    for f in feats:
        if f["seqid"] not in seqs:
            skipped_no_seqid += 1; continue
        s = seqs[f["seqid"]]
        cds = extract_cds(s, f["start"], f["end"], f["strand"])
        if len(cds) < 30 or len(cds) % 3 != 0:
            skipped_no_seq += 1; continue
        upstream = extract_upstream(s, f["start"], f["end"], f["strand"],
                                     PROMOTER_SEARCH_WINDOW)
        rows.append({
            **f,
            "cds": cds,
            "upstream": upstream,
            "cds_len": len(cds),
        })
    print(f"  usable CDSes: {len(rows):,}  "
          f"(skipped {skipped_no_seq} short/frameshift, "
          f"{skipped_no_seqid} no seqid)")

    # ---- 4. join labels ----
    print(f"\nStage 4: join labels (organism={args.organism_key})")
    labels = pd.read_csv(args.labels_csv)
    org_labels = labels[labels.organism == args.organism_key]
    print(f"  labels rows: {len(org_labels)}  "
          f"(essential: {int(org_labels.essential.sum())})")
    if len(org_labels) == 0:
        print(f"ERROR: no rows for organism={args.organism_key}", file=sys.stderr)
        print(f"  available organisms (first 10): "
              f"{labels.organism.unique()[:10].tolist()}", file=sys.stderr)
        return 1
    # Build lookups: locus_tag -> essential AND gene_name -> essential
    # (the gene_name fallback covers cross-strain matching like
    # MG1655 b-numbers vs BW25113_NNNN, where both strains share
    # the same canonical gene names: thrL, dnaA, rpsA, etc.)
    lt_ess: dict[str, int] = dict(zip(
        org_labels.locus_tag.astype(str),
        org_labels.essential.astype(int)))
    gene_ess: dict[str, int] = {}
    if "gene_name" in org_labels.columns:
        for g, e in zip(org_labels.gene_name, org_labels.essential):
            if pd.notna(g) and str(g).strip():
                gene_ess[str(g).strip().lower()] = int(e)
    print(f"  lookup tables: {len(lt_ess)} by locus_tag, "
          f"{len(gene_ess)} by gene_name")

    matched_lt = matched_old = matched_gene = 0
    for r in rows:
        e = None
        # 1. exact locus_tag
        if r["locus_tag"] and r["locus_tag"] in lt_ess:
            e = lt_ess[r["locus_tag"]]; matched_lt += 1
        # 2. old_locus_tag
        elif r["old_locus_tag"] and r["old_locus_tag"] in lt_ess:
            e = lt_ess[r["old_locus_tag"]]; matched_old += 1
        # 3. fallback: gene name (cross-strain canonical symbol)
        elif r["gene"] and r["gene"].strip().lower() in gene_ess:
            e = gene_ess[r["gene"].strip().lower()]; matched_gene += 1
        r["essential"] = e
    matched = matched_lt + matched_old + matched_gene
    print(f"  matched to label: {matched}/{len(rows)} CDSes "
          f"({100*matched/max(len(rows),1):.1f}%)")
    print(f"    via locus_tag:      {matched_lt}")
    print(f"    via old_locus_tag:  {matched_old}")
    print(f"    via gene_name:      {matched_gene}")
    if matched < 100:
        print(f"WARN: too few matches. RefSeq locus_tag samples: "
              f"{[r['locus_tag'] for r in rows[:3]]} "
              f"  BERIL samples: {list(lt_ess.keys())[:3]}")
        print(f"  RefSeq gene samples: {[r['gene'] for r in rows[:5] if r['gene']]}")
        print(f"  BERIL gene samples:  {list(gene_ess.keys())[:5]}")

    # ---- 5. build CAI reference set (ribosomal protein CDSes) ----
    print(f"\nStage 5: build CAI weights from ribosomal protein reference")
    ref_cdss = [r["cds"] for r in rows
                if r["gene"] and re.match(r"^rp[slm][A-Z]?$",
                                            r["gene"], re.IGNORECASE)]
    if len(ref_cdss) < 20:
        # fall back to anything matching rpl/rps/rpm prefix
        ref_cdss = [r["cds"] for r in rows
                    if r["gene"] and r["gene"][:3].lower() in ("rpl","rps","rpm")]
    print(f"  reference CDSes (ribosomal): {len(ref_cdss)}")
    cai_w = build_cai_weights(ref_cdss)

    # ---- 6. compute features ----
    print(f"\nStage 6: compute features (H1 promoter, H2 codon, H3 operon)")
    # First pass: per-CDS features
    for r in rows:
        cds = r["cds"]
        r["cai"]              = cai_of(cds, cai_w)
        r["gc"]               = gc_content(cds)
        r["gc3"]              = gc3_content(cds)
        r["minus10_score"]    = motif_score(r["upstream"], MINUS_10_CONSENSUS)
        r["minus35_score"]    = motif_score(r["upstream"], MINUS_35_CONSENSUS)
        r["promoter_gc"]      = gc_content(r["upstream"])
        r["promoter_at_run"]  = max((len(m.group(0))
                                       for m in re.finditer(r"A{4,}|T{4,}",
                                                              r["upstream"])),
                                       default=0)
    # H3: operon-context features need sorted-by-position pass
    rows.sort(key=lambda r: (r["seqid"], r["start"]))
    for i, r in enumerate(rows):
        prev_r = rows[i - 1] if i > 0 and rows[i - 1]["seqid"] == r["seqid"] else None
        next_r = rows[i + 1] if i + 1 < len(rows) and rows[i + 1]["seqid"] == r["seqid"] else None
        # intergenic distance: from upstream neighbor's end to this start
        if prev_r:
            r["intergenic_prev"] = max(0, r["start"] - prev_r["end"] - 1)
            r["same_strand_prev_pos"] = int(prev_r["strand"] == r["strand"])
            r["prev_essential"] = prev_r.get("essential")
        else:
            r["intergenic_prev"] = float("nan")
            r["same_strand_prev_pos"] = 0
            r["prev_essential"] = None
        if next_r:
            r["intergenic_next"] = max(0, next_r["start"] - r["end"] - 1)
            r["same_strand_next_pos"] = int(next_r["strand"] == r["strand"])
            r["next_essential"] = next_r.get("essential")
        else:
            r["intergenic_next"] = float("nan")
            r["same_strand_next_pos"] = 0
            r["next_essential"] = None
        # operonic indicator: short intergenic + same strand
        r["operon_prev"] = int(r["intergenic_prev"] is not float("nan")
                                and (r["intergenic_prev"] or 1e9) < 50
                                and r["same_strand_prev_pos"])
        r["operon_next"] = int(r["intergenic_next"] is not float("nan")
                                and (r["intergenic_next"] or 1e9) < 50
                                and r["same_strand_next_pos"])
        # neighbor essentiality (LEAK NOTE: only used for H3 verdict; not
        # safe to use as a model feature without LOO handling)
        if r["prev_essential"] is not None and r["next_essential"] is not None:
            r["neighbor_ess_frac"] = (r["prev_essential"] + r["next_essential"]) / 2
        elif r["prev_essential"] is not None:
            r["neighbor_ess_frac"] = float(r["prev_essential"])
        elif r["next_essential"] is not None:
            r["neighbor_ess_frac"] = float(r["next_essential"])
        else:
            r["neighbor_ess_frac"] = float("nan")

    # ---- 7. statistical tests per hypothesis ----
    print(f"\nStage 7: per-feature essential vs non-essential statistics")
    df_rows = [r for r in rows if r["essential"] is not None]
    print(f"  testable rows: {len(df_rows)} "
          f"({sum(r['essential'] for r in df_rows)} essential)")
    feature_results = []
    for fname, higher_is_ess in [
        # H1: promoter
        ("H1_minus10_score",     True),
        ("H1_minus35_score",     True),
        ("H1_promoter_gc",       False),
        ("H1_promoter_at_run",   True),
        # H2: codon
        ("H2_cai",               True),
        ("H2_gc3",               True),
        ("H2_gc",                True),
        ("H2_cds_len",           True),
        # H3: operon context
        ("H3_intergenic_prev",   False),
        ("H3_intergenic_next",   False),
        ("H3_same_strand_prev",  True),
        ("H3_same_strand_next",  True),
        ("H3_neighbor_ess_frac", True),
    ]:
        key = fname.split("_", 1)[1].lower()
        # map to actual dict key
        key_map = {
            "minus10_score":     "minus10_score",
            "minus35_score":     "minus35_score",
            "promoter_gc":       "promoter_gc",
            "promoter_at_run":   "promoter_at_run",
            "cai":               "cai",
            "gc3":               "gc3",
            "gc":                "gc",
            "cds_len":           "cds_len",
            "intergenic_prev":   "intergenic_prev",
            "intergenic_next":   "intergenic_next",
            "same_strand_prev":  "same_strand_prev_pos",
            "same_strand_next":  "same_strand_next_pos",
            "neighbor_ess_frac": "neighbor_ess_frac",
        }
        k = key_map[key]
        ess = [float(r[k]) for r in df_rows if r["essential"] == 1
               and r[k] is not None and not (isinstance(r[k], float)
                                              and math.isnan(r[k]))]
        non = [float(r[k]) for r in df_rows if r["essential"] == 0
               and r[k] is not None and not (isinstance(r[k], float)
                                              and math.isnan(r[k]))]
        scores = [float(r[k]) if (r[k] is not None and not
                                     (isinstance(r[k], float)
                                       and math.isnan(r[k]))) else float("nan")
                   for r in df_rows]
        labels = [int(r["essential"]) for r in df_rows]
        feature_results.append(
            report(fname, ess, non, scores, labels, higher_is_essential=higher_is_ess))

    # ---- 8. hypothesis verdicts ----
    def best_verdict(results: list[dict]) -> str:
        ranks = {"STRONG": 3, "MODERATE": 2, "WEAK/NONE": 1}
        return max(results, key=lambda r: ranks[r["verdict"]])["verdict"]

    h1 = [r for r in feature_results if r["feature"].startswith("H1_")]
    h2 = [r for r in feature_results if r["feature"].startswith("H2_")]
    h3 = [r for r in feature_results if r["feature"].startswith("H3_")]
    print(f"\n=== HYPOTHESIS VERDICTS ({args.organism_key}) ===")
    print(f"  H1 (promoter strength upstream):  {best_verdict(h1)}")
    print(f"  H2 (codon optimisation):          {best_verdict(h2)}")
    print(f"  H3 (operon context):              {best_verdict(h3)}")

    # ---- 9. save outputs ----
    keep_cols = ["locus_tag","old_locus_tag","gene","seqid","start","end","strand",
                 "cds_len","essential",
                 "minus10_score","minus35_score","promoter_gc","promoter_at_run",
                 "cai","gc3","gc",
                 "intergenic_prev","intergenic_next","same_strand_prev_pos",
                 "same_strand_next_pos","neighbor_ess_frac"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["organism"] + keep_cols)
        w.writeheader()
        for r in rows:
            row = {"organism": args.organism_key}
            for k in keep_cols:
                row[k] = r.get(k)
            w.writerow(row)
    print(f"\nwrote per-gene features: {args.out_csv}")

    summary = {
        "organism":          args.organism_key,
        "accession":         args.accession,
        "n_cds_total":       len(rows),
        "n_labelled":        len(df_rows),
        "n_essential":       int(sum(r["essential"] for r in df_rows)),
        "n_ref_ribosomal":   len(ref_cdss),
        "feature_results":   feature_results,
        "hypothesis_verdicts": {
            "H1_promoter":   best_verdict(h1),
            "H2_codon":      best_verdict(h2),
            "H3_operon":     best_verdict(h3),
        },
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote summary: {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
