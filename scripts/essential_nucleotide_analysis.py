"""Per-gene NUCLEOTIDE-LEVEL analysis: A/C/G/T composition of each essential CDS.

Naresh: "now analyse the nucleotides sequence itself. like A,C,G,T"

For each labeled gene we extract its CDS nucleotide sequence (using GFF
coordinates + genome.fna + reverse-complement if minus strand) and compute
features directly from the bases:

  base composition       A_frac, C_frac, G_frac, T_frac
  GC                     overall (G+C)/len
  GC1, GC2, GC3          GC content at codon position 1, 2, 3 separately.
                         GC3 is the "wobble" position and is the most-tuned
                         by selection on highly-expressed genes.
  GC_skew                (G-C)/(G+C) -- strand-bias signature; leading-strand
                         genes typically have positive skew in bacteria.
  AT_skew                (A-T)/(A+T)
  CpG_obs_exp            observed/expected CpG dinucleotide frequency
                         (CpG depletion = methylation/mutation pressure)
  TpA_obs_exp            TpA dinucleotide (low in mRNA -- ribosome stall)
  length_nt              CDS length in bp
  starts_with_ATG        boolean (most ATG vs GTG/TTG)
  ends_with_stop_TAA     boolean (TAA preferred in low-G+C orgs)
  homopolymer_max        longest run of identical bases (slippage signal)

Then per organism (within-org -- no cross-org GC confound) we compute:
  - mean of each feature in essentials vs non-essentials
  - Cohen's d (signed; positive = higher in essentials)
  - single-feature AUC

ORGS COVERED: ones with .fna + .gff + manifest join_rate >= 0.95:
  beril_HerbieS, beril_Magneto, beril_Dda3937, mtub,
  beril_Methanococcus_S2/JJ (archaea), beril_RalstoniaGMI1000/BSBF1503/PSI07
  -> ~7,000 essential CDSs across diverse GC content (~40-70%).

OUTPUT: outputs/essential_nucleotide_analysis.md + per-gene CSV.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "data" / "drive_import" / "labels" / "labels.csv"
MANIFEST = REPO / "data" / "drive_import" / "labels" / "genome_cache_manifest.csv"
CACHE = REPO / "data" / "drive_import" / "genome_cache"

COMP = str.maketrans("ACGTacgtN", "TGCAtgcaN")


def rev_comp(s: str) -> str:
    return s.translate(COMP)[::-1]


def parse_fna(path: Path) -> dict[str, str]:
    """contig_id -> sequence (uppercase, gaps preserved)"""
    out = {}
    cur_id = None; cur = []
    with open(path) as f:
        for ln in f:
            if ln.startswith(">"):
                if cur_id is not None:
                    out[cur_id] = "".join(cur).upper()
                cur_id = ln[1:].split()[0]
                cur = []
            else:
                cur.append(ln.strip())
    if cur_id is not None:
        out[cur_id] = "".join(cur).upper()
    return out


def parse_gff_cds(path: Path):
    """Yield (locus_tag, contig, start, end, strand) for each protein-coding CDS.
    GFF is 1-based, inclusive on both ends."""
    locus_re = re.compile(r"locus_tag=([^;]+)")
    for line in open(path):
        if line.startswith("#"): continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 9: continue
        if parts[2] != "CDS": continue
        m = locus_re.search(parts[8])
        if not m: continue
        try:
            start = int(parts[3]); end = int(parts[4])
        except ValueError:
            continue
        yield (m.group(1), parts[0], start, end, parts[6])


def features(seq: str) -> dict:
    n = len(seq); A = seq.count("A"); C = seq.count("C")
    G = seq.count("G"); T = seq.count("T")
    valid = A + C + G + T
    if valid == 0: return {}
    Af, Cf, Gf, Tf = A/valid, C/valid, G/valid, T/valid
    gc = (G + C) / valid
    # codon positions 1, 2, 3 -- only over full codons
    n_codon = n // 3
    p1 = seq[0::3][:n_codon]; p2 = seq[1::3][:n_codon]; p3 = seq[2::3][:n_codon]
    def _gc(s):
        a = s.count("A") + s.count("C") + s.count("G") + s.count("T")
        return (s.count("G") + s.count("C")) / a if a else float("nan")
    gc1, gc2, gc3 = _gc(p1), _gc(p2), _gc(p3)
    # skew
    gc_skew = (G - C) / (G + C) if (G + C) else 0.0
    at_skew = (A - T) / (A + T) if (A + T) else 0.0
    # dinucleotide CpG observed/expected
    dn_cpg = sum(1 for i in range(n - 1) if seq[i] == "C" and seq[i+1] == "G")
    dn_tpa = sum(1 for i in range(n - 1) if seq[i] == "T" and seq[i+1] == "A")
    exp_cpg = Cf * Gf * (n - 1)
    exp_tpa = Tf * Af * (n - 1)
    cpg_oe = dn_cpg / exp_cpg if exp_cpg > 0 else float("nan")
    tpa_oe = dn_tpa / exp_tpa if exp_tpa > 0 else float("nan")
    # start codon
    start = seq[:3]
    stop = seq[-3:] if n >= 3 else ""
    # longest homopolymer run
    runs = re.findall(r"A{2,}|C{2,}|G{2,}|T{2,}", seq)
    homo_max = max((len(r) for r in runs), default=0)
    return {
        "length_nt": n, "A_frac": Af, "C_frac": Cf, "G_frac": Gf, "T_frac": Tf,
        "GC": gc, "GC1": gc1, "GC2": gc2, "GC3": gc3,
        "GC_skew": gc_skew, "AT_skew": at_skew,
        "CpG_obs_exp": cpg_oe, "TpA_obs_exp": tpa_oe,
        "starts_with_ATG": int(start == "ATG"),
        "ends_with_TAA": int(stop == "TAA"),
        "ends_with_TAG": int(stop == "TAG"),
        "ends_with_TGA": int(stop == "TGA"),
        "homopolymer_max": homo_max,
    }


def cohens_d(a, b):
    import numpy as np
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return float("nan")
    pooled = (((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
              / (len(a)+len(b)-2)) ** 0.5
    return (a.mean() - b.mean()) / pooled if pooled else float("nan")


def auc(v, y):
    import numpy as np, pandas as pd
    m = ~np.isnan(v)
    v = v[m]; y = y[m]
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    if n_pos < 2 or n_neg < 2: return float("nan")
    r = pd.Series(v).rank(method="average").values
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> int:
    import pandas as pd, numpy as np, os

    print("Loading labels + manifest ...")
    lab = pd.read_csv(LABELS)
    mani = pd.read_csv(MANIFEST)
    ess_per = lab[lab.essential == 1].organism.value_counts()

    candidates = []
    for _, r in mani.iterrows():
        if pd.isna(r.accession): continue
        fna = CACHE / r.accession / "genome.fna"
        gff = CACHE / r.accession / "genomic.gff"
        if fna.exists() and gff.exists() and r.join_rate >= 0.95:
            candidates.append((r.organism, r.accession, int(ess_per.get(r.organism, 0))))
    candidates.sort(key=lambda x: -x[2])
    print(f"  candidates: {len(candidates)}")

    all_rows = []
    for org, acc, n_ess in candidates:
        if n_ess < 100: continue
        fna = CACHE / acc / "genome.fna"
        gff = CACHE / acc / "genomic.gff"
        print(f"\n  {org:<25s} ({acc}) -- extracting CDS ...")
        contigs = parse_fna(fna)
        org_lab = lab[lab.organism == org]
        lt_to_label = dict(zip(org_lab.locus_tag.astype(str),
                               org_lab.essential.astype(int)))
        matched = 0
        for lt, contig, start, end, strand in parse_gff_cds(gff):
            if lt not in lt_to_label: continue
            seq = contigs.get(contig, "")[start-1:end]
            if not seq: continue
            if strand == "-":
                seq = rev_comp(seq)
            # only A/C/G/T-clean CDS to keep features comparable
            if len(seq) - (seq.count("A")+seq.count("C")+
                           seq.count("G")+seq.count("T")) > 5:
                continue
            f = features(seq)
            if not f: continue
            f["organism"] = org; f["locus_tag"] = lt
            f["essential"] = lt_to_label[lt]
            all_rows.append(f); matched += 1
        print(f"    matched {matched} CDSs")

    if not all_rows:
        print("ERROR: no CDSs extracted", file=sys.stderr); return 1

    df = pd.DataFrame(all_rows)
    print(f"\nTotal: {len(df):,} CDSs, "
          f"essential={int(df.essential.sum()):,}, "
          f"non={int((df.essential==0).sum()):,}, "
          f"orgs={df.organism.nunique()}")

    feats = ["length_nt","A_frac","C_frac","G_frac","T_frac",
             "GC","GC1","GC2","GC3","GC_skew","AT_skew",
             "CpG_obs_exp","TpA_obs_exp",
             "starts_with_ATG","ends_with_TAA","ends_with_TAG","ends_with_TGA",
             "homopolymer_max"]

    print(f"\n=== POOLED essential vs non-essential ===")
    pooled = []
    hdr = "d"
    print(f"  {'feature':<22s} {'ess mean':>10s} {'non mean':>10s} "
          f"{hdr:>7s} {'AUC':>6s}")
    for f in feats:
        ea = pd.to_numeric(df[df.essential==1][f], errors="coerce").values.astype(float)
        na = pd.to_numeric(df[df.essential==0][f], errors="coerce").values.astype(float)
        v = pd.to_numeric(df[f], errors="coerce").values.astype(float)
        d = cohens_d(ea, na)
        a = auc(v, df.essential.values.astype(int))
        em = float(np.nanmean(ea)); nm = float(np.nanmean(na))
        print(f"  {f:<22s} {em:>10.4f} {nm:>10.4f} {d:>+7.3f} {a:>6.3f}")
        pooled.append((f, em, nm, d, a))

    print(f"\n=== WITHIN each organism (strips cross-org GC confound) ===")
    orgs = sorted(df.organism.unique())
    print(f"  {'feature':<22s} " + "".join(f"{o[:9]:>10s}" for o in orgs))
    per_org = []
    for f in feats:
        cells = []
        row = [f]
        for org in orgs:
            sub = df[df.organism == org]
            v = pd.to_numeric(sub[f], errors="coerce").values.astype(float)
            y = sub.essential.values.astype(int)
            if (y==1).sum() >= 20 and (y==0).sum() >= 20:
                a = auc(v, y); cells.append(f"{a:>10.3f}"); row.append(a)
            else:
                cells.append(f"{'-':>10s}"); row.append(None)
        print(f"  {f:<22s} " + "".join(cells))
        per_org.append(row)

    out_csv = REPO / "outputs" / "essential_nucleotide_features.csv"
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(df):,} rows)")

    # markdown
    lines = ["# Essential vs non-essential CDS: NUCLEOTIDE-level analysis\n",
             f"**{len(df):,} CDSs** from {df.organism.nunique()} organisms ("
             f"{', '.join(orgs)}). Each per-gene feature is computed directly "
             f"from the A/C/G/T sequence of the CDS (extracted from GFF coords "
             f"+ genome.fna, reverse-complemented where strand=='-').\n",
             "## Pooled comparison (sorted by |AUC-0.5|)\n",
             "| feature | essential mean | non-essential mean | Cohen's d | AUC |",
             "|---|---:|---:|---:|---:|"]
    for f, em, nm, d, a in sorted(pooled, key=lambda r: -abs(r[4]-0.5)):
        lines.append(f"| {f} | {em:.4f} | {nm:.4f} | {d:+.3f} | {a:.3f} |")
    lines.append("\n## Within-organism (cross-org GC confound stripped)\n")
    lines.append("| feature | " + " | ".join(orgs) + " |")
    lines.append("|---|" + "|".join(["---:"]*len(orgs)) + "|")
    for row in per_org:
        cells = []
        for v in row[1:]:
            cells.append(f"{v:.3f}" if v is not None else "-")
        lines.append(f"| {row[0]} | " + " | ".join(cells) + " |")
    (REPO / "outputs" / "essential_nucleotide_analysis.md").write_text("\n".join(lines))
    print(f"wrote outputs/essential_nucleotide_analysis.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
