"""Per-CDS SEQUENCE-PATTERN analysis: which patterns in the A/C/G/T sequence
of essential genes differ from non-essentials?

Not composition (counts of bases / dinucleotides) -- we did that. PATTERN
means actual sequence STRUCTURE:
  - which CODONS (3-mers in frame) are preferred in essentials
  - which 4-/6-mers are enriched in essentials
  - codon-period rhythm strength (translation selection signature)
  - first 30nt and last 30nt regions (translation initiation/termination)

Per-CDS features (computed PER ORGANISM to strip cross-org GC confound):

  CODON-LEVEL
    64-dimensional codon-usage vector per gene (frequency of each codon
    among all in-frame codons). Then per organism: for each of 64 codons,
    AUC discriminating essential vs non-essential. Report top-discriminative
    codons.

  6-MER ENRICHMENT
    Sliding 6-mer counts; per organism find the 20 hexamers most
    differentially abundant in essentials vs non-essentials (log odds ratio
    with prevalence prior). Report what they spell.

  CODON-PERIODICITY STRENGTH
    Discrete Fourier component at period 3 of the base sequence (any base
    -- say A or T). High value = strong codon-period rhythm = strong
    translation selection.

  FIRST/LAST 30nt
    Compare base composition of (1) the first 30nt after the start codon
    (translation-initiation region; SD-region overlap, 5'-mRNA structure)
    and (2) the last 30nt before stop (3' fold, termination context).
    Differences here are subtle but well-documented.

OUTPUT: outputs/essential_nucleotide_patterns.md
"""
from __future__ import annotations
import re, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "data" / "drive_import" / "labels" / "labels.csv"
MANIFEST = REPO / "data" / "drive_import" / "labels" / "genome_cache_manifest.csv"
CACHE = REPO / "data" / "drive_import" / "genome_cache"
COMP = str.maketrans("ACGTacgtN", "TGCAtgcaN")

# the 20 amino acids -> their canonical codons (for grouping later)
AA_BY_CODON = {
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S","AGT":"S","AGC":"S",
    "CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T",
    "GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","TGA":"*",
    "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
    "AAT":"N","AAC":"N","AAA":"K","AAG":"K",
    "GAT":"D","GAC":"D","GAA":"E","GAG":"E",
    "TGT":"C","TGC":"C","TGG":"W",
    "CGT":"R","CGC":"R","CGA":"R","CGG":"R","AGA":"R","AGG":"R",
    "GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}
ALL_CODONS = sorted(AA_BY_CODON.keys())


def rev_comp(s: str) -> str: return s.translate(COMP)[::-1]


def parse_fna(path):
    out, cur_id, cur = {}, None, []
    for ln in open(path):
        if ln.startswith(">"):
            if cur_id is not None: out[cur_id] = "".join(cur).upper()
            cur_id = ln[1:].split()[0]; cur = []
        else: cur.append(ln.strip())
    if cur_id is not None: out[cur_id] = "".join(cur).upper()
    return out


def parse_gff_cds(path):
    locus_re = re.compile(r"locus_tag=([^;]+)")
    for line in open(path):
        if line.startswith("#"): continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 9 or parts[2] != "CDS": continue
        m = locus_re.search(parts[8])
        if not m: continue
        try:
            yield (m.group(1), parts[0], int(parts[3]), int(parts[4]), parts[6])
        except ValueError: continue


def codon_freqs(seq):
    """Return dict codon -> count for in-frame codons, dropping ambiguous."""
    n = len(seq); cnt = Counter()
    for i in range(0, n - 2, 3):
        c = seq[i:i+3]
        if c in AA_BY_CODON: cnt[c] += 1
    return cnt


def period3_strength(seq):
    """DFT magnitude at frequency 1/3 for a binary AT indicator vector,
    normalised by total energy. High -> strong codon-period rhythm."""
    n = len(seq)
    if n < 30: return 0.0
    x = [1.0 if b in "AT" else 0.0 for b in seq]
    # demean
    mu = sum(x) / n
    x = [v - mu for v in x]
    import math
    # only compute the freq=1/3 bin
    cos_s = sin_s = 0.0
    for k in range(n):
        ang = 2 * math.pi * k / 3
        cos_s += x[k] * math.cos(ang)
        sin_s += x[k] * math.sin(ang)
    pwr_3 = (cos_s ** 2 + sin_s ** 2)
    # total power = sum x^2
    tot = sum(v * v for v in x)
    return pwr_3 / max(tot, 1e-9)


def auc(v, y):
    import numpy as np, pandas as pd
    m = ~np.isnan(v); v = v[m]; y = y[m]
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    if n_pos < 5 or n_neg < 5: return float("nan")
    r = pd.Series(v).rank(method="average").values
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> int:
    import pandas as pd, numpy as np

    print("Loading labels + manifest ...")
    lab = pd.read_csv(LABELS)
    mani = pd.read_csv(MANIFEST)
    ess_per = lab[lab.essential == 1].organism.value_counts()

    targets = []
    for _, r in mani.iterrows():
        if pd.isna(r.accession): continue
        fna = CACHE / r.accession / "genome.fna"
        gff = CACHE / r.accession / "genomic.gff"
        if fna.exists() and gff.exists() and r.join_rate >= 0.95 \
                and ess_per.get(r.organism, 0) >= 200:
            targets.append((r.organism, r.accession))
    print(f"  organisms: {len(targets)}")

    per_org_codon_aucs = {}   # org -> dict codon -> AUC
    per_org_kmer = {}         # org -> dict kmer -> log-odds
    per_org_period = {}       # org -> mean period3 (essential vs non)
    per_org_first30 = {}      # org -> (auc dict on first-30 base fractions)
    per_org_n = {}

    for org, acc in targets:
        print(f"\n  {org:<28s} extracting ...")
        contigs = parse_fna(CACHE / acc / "genome.fna")
        org_lab = lab[lab.organism == org][["locus_tag","essential"]]
        lt_to_y = dict(zip(org_lab.locus_tag.astype(str),
                           org_lab.essential.astype(int)))

        codon_freq_rows = []   # list of (label, codon->freq_dict, seq)
        for lt, contig, s, e, strand in parse_gff_cds(CACHE / acc / "genomic.gff"):
            if lt not in lt_to_y: continue
            seq = contigs.get(contig, "")[s-1:e]
            if not seq: continue
            if strand == "-": seq = rev_comp(seq)
            # clean: only A/C/G/T
            if (seq.count("A")+seq.count("C")+seq.count("G")+seq.count("T")) \
                    < len(seq) - 5: continue
            cf = codon_freqs(seq)
            tot = sum(cf.values())
            if tot < 30: continue   # need a reasonable CDS
            freqs = {c: cf.get(c, 0) / tot for c in ALL_CODONS}
            codon_freq_rows.append((lt_to_y[lt], freqs, seq))

        y = np.array([r[0] for r in codon_freq_rows], dtype=int)
        per_org_n[org] = (int((y==1).sum()), int((y==0).sum()))
        if per_org_n[org][0] < 50 or per_org_n[org][1] < 50:
            print(f"    skipped (insufficient)"); continue

        # codon AUCs
        codon_aucs = {}
        for c in ALL_CODONS:
            v = np.array([r[1][c] for r in codon_freq_rows])
            codon_aucs[c] = auc(v, y)
        per_org_codon_aucs[org] = codon_aucs

        # period3 essential vs non
        p3 = np.array([period3_strength(r[2][:300]) for r in codon_freq_rows])
        per_org_period[org] = (float(p3[y==1].mean()), float(p3[y==0].mean()),
                                auc(p3, y))

        # 6-mer enrichment
        # build per-class total counts (sliding window)
        from collections import defaultdict
        cnt_ess = defaultdict(int); cnt_non = defaultdict(int)
        for label, _, seq in codon_freq_rows:
            d = cnt_ess if label == 1 else cnt_non
            for i in range(len(seq) - 5):
                k = seq[i:i+6]
                if "N" in k: continue
                d[k] += 1
        N_ess = sum(cnt_ess.values()); N_non = sum(cnt_non.values())
        # rank by log-odds (essential vs non), prior smoothing
        all_k = set(cnt_ess) | set(cnt_non)
        scores = []
        for k in all_k:
            a = cnt_ess[k] + 1; b = cnt_non[k] + 1
            # log( (a/N_ess) / (b/N_non) )
            scores.append((k, np.log((a/N_ess) / (b/N_non)),
                           cnt_ess[k], cnt_non[k]))
        scores.sort(key=lambda r: -r[1])
        per_org_kmer[org] = scores

        # first 30 nt base composition
        first30 = []
        for label, _, seq in codon_freq_rows:
            if len(seq) < 30: continue
            sub = seq[3:33]  # skip start codon
            if len(sub) != 30: continue
            for b in "ACGT":
                first30.append((label, b, sub.count(b)/30.0))
        first_df = pd.DataFrame(first30, columns=["y","base","frac"])
        f30_aucs = {}
        for b in "ACGT":
            sub = first_df[first_df.base == b]
            f30_aucs[b] = auc(sub.frac.values, sub.y.values.astype(int))
        per_org_first30[org] = f30_aucs

        # progress print
        top_c = sorted(codon_aucs.items(), key=lambda kv: -abs(kv[1] - 0.5))[:5]
        print(f"    n_ess={per_org_n[org][0]} n_non={per_org_n[org][1]}  "
              f"top discriminating codons: " +
              ", ".join(f"{c}({a:.2f})" for c, a in top_c))

    # ======== aggregate report ========
    print(f"\n=== CODON usage: AUCs averaged across organisms ===")
    orgs_done = list(per_org_codon_aucs.keys())
    mean_aucs = {}
    for c in ALL_CODONS:
        vals = [per_org_codon_aucs[o][c] for o in orgs_done
                if not np.isnan(per_org_codon_aucs[o][c])]
        if vals: mean_aucs[c] = (np.mean(vals), np.std(vals), len(vals))

    ranked = sorted(mean_aucs.items(),
                    key=lambda kv: -abs(kv[1][0] - 0.5))[:25]
    print(f"  {'codon':<6s} {'aa':<3s} {'mean_AUC':>9s} {'sd':>6s} {'sign':<8s}")
    for c, (m, sd, n) in ranked:
        direction = "HIGH_ess" if m > 0.5 else "LOW_ess"
        print(f"  {c:<6s} {AA_BY_CODON[c]:<3s} {m:>9.3f} {sd:>6.3f} {direction}")

    print(f"\n=== CODON-PERIOD strength (period-3 rhythm) ===")
    print(f"  {'org':<28s} {'ess mean':>10s} {'non mean':>10s} {'AUC':>6s}")
    for org, (e, n_, a) in per_org_period.items():
        print(f"  {org:<28s} {e:>10.4f} {n_:>10.4f} {a:>6.3f}")

    print(f"\n=== FIRST-30nt base composition (after start codon) ===")
    print(f"  {'org':<28s} " + " ".join(f"A_AUC C_AUC G_AUC T_AUC".split()))
    for org, d in per_org_first30.items():
        cells = " ".join(f"{d[b]:5.2f}" for b in "ACGT")
        print(f"  {org:<28s} {cells}")

    print(f"\n=== TOP DISCRIMINATIVE 6-mers (in 2+ orgs) ===")
    # take top-30 by log-odds per org, intersect orgs
    enriched_ess = Counter()
    for org in orgs_done:
        top = [s[0] for s in per_org_kmer[org][:40]]
        for k in top: enriched_ess[k] += 1
    print(f"  hexamer    n_orgs (of {len(orgs_done)})")
    for k, n in enriched_ess.most_common(20):
        if n >= 2:
            print(f"  {k}    {n}")

    # ----- write markdown -----
    lines = [f"# Sequence-PATTERN analysis: what patterns mark essential CDS\n",
             f"Across {len(orgs_done)} organisms.\n",
             "## Top discriminating codons (mean AUC across organisms)\n",
             "| codon | aa | mean AUC | sd | direction |",
             "|---|---|---:|---:|---|"]
    for c, (m, sd, n) in ranked:
        direction = "HIGHER in essentials" if m > 0.5 else "LOWER in essentials"
        lines.append(f"| {c} | {AA_BY_CODON[c]} | {m:.3f} | {sd:.3f} | {direction} |")
    lines.append("\n## Codon-period-3 rhythm strength\n")
    lines.append("| organism | essential mean | non-essential mean | AUC |")
    lines.append("|---|---:|---:|---:|")
    for org, (e, n_, a) in per_org_period.items():
        lines.append(f"| {org} | {e:.4f} | {n_:.4f} | {a:.3f} |")
    lines.append("\n## Top 6-mers more enriched in essentials, in >=2 organisms\n")
    lines.append("| hexamer | n organisms |")
    lines.append("|---|---:|")
    for k, n in enriched_ess.most_common(20):
        if n >= 2: lines.append(f"| {k} | {n} |")
    out = REPO / "outputs" / "essential_nucleotide_patterns.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
