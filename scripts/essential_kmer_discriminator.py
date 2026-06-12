"""Discriminative k-mer / motif search: subsequences PRESENT in essential
genes but ABSENT (or rare) in non-essential genes.

Naresh: "look for sequences that are common in essential genes but absent in
non essential ... maybe 6 or 10 or any nucleotide length long."

This is discriminative k-mer (motif) discovery by PRESENCE/ABSENCE per gene:
  for each k-mer w and each organism:
    p_ess = fraction of ESSENTIAL genes that CONTAIN w at least once
    p_non = fraction of NON-essential genes that CONTAIN w
    discrimination = p_ess - p_non
  Rank by discrimination, averaged across organisms (require the k-mer to be
  observed in >= MIN_ORGS organisms so we don't chase one-genome quirks).

Directly tests the strong hypothesis "common in essential, ABSENT in
non-essential":
  count k-mers with p_ess >= 0.30 AND p_non <= 0.05 (present in a third of
  essentials, near-absent in non-essentials).

k scanned: 6, 8, 10, 12.

HONEST CAVEAT printed up front: short k-mers (k<=8) occur in almost every
gene -> 'absence' is combinatorially near-impossible; long k-mers (k>=12)
are rare -> 'commonness' is near-impossible. Any real discriminator that
survives is therefore informative. AND: essentials are enriched for certain
FUNCTIONAL CLASSES (ribosomal proteins, etc.), so a discriminative k-mer is
usually a marker of the FUNCTION, not of essentiality per se -- we flag this.

ORGS: the 9 with .fna + .gff + join_rate>=0.95 used in the composition/pattern
analyses.

OUTPUT: outputs/essential_kmer_discriminators.md
"""
from __future__ import annotations
import re, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "data" / "drive_import" / "labels" / "labels.csv"
MANIFEST = REPO / "data" / "drive_import" / "labels" / "genome_cache_manifest.csv"
CACHE = REPO / "data" / "drive_import" / "genome_cache"
COMP = str.maketrans("ACGTacgtN", "TGCAtgcaN")

KS = [6, 8, 10, 12]
MIN_ORGS = 4          # k-mer must discriminate in >= this many organisms
MIN_PREVALENCE = 0.05 # only rank k-mers present in >=5% of either class
ABS_P_ESS = 0.30      # "present in essential" threshold for the absence test
ABS_P_NON = 0.05      # "absent in non-essential" threshold


def rev_comp(s): return s.translate(COMP)[::-1]


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
        p = line.rstrip("\n").split("\t")
        if len(p) < 9 or p[2] != "CDS": continue
        m = locus_re.search(p[8])
        if not m: continue
        try: yield (m.group(1), p[0], int(p[3]), int(p[4]), p[6])
        except ValueError: continue


def kmer_set(seq, k):
    """Distinct canonical k-mers in seq (canonical = min(kmer, revcomp) so
    strand doesn't matter). Skip any with N."""
    s = set()
    for i in range(len(seq) - k + 1):
        w = seq[i:i+k]
        if "N" in w: continue
        rc = rev_comp(w)
        s.add(w if w <= rc else rc)
    return s


def main() -> int:
    import pandas as pd
    import numpy as np

    print("=" * 70)
    print("HONEST CAVEAT before we start:")
    print("  - k<=8 k-mers occur in nearly EVERY gene -> 'absent in")
    print("    non-essential' is combinatorially almost impossible.")
    print("  - k>=12 k-mers are rare -> 'common in essential' is almost")
    print("    impossible. The window where a true present/absent motif can")
    print("    exist is narrow, so anything that survives is meaningful.")
    print("  - Essentials are enriched for FUNCTIONAL CLASSES (ribosomes,")
    print("    polymerases); a discriminative k-mer usually marks the")
    print("    FUNCTION, not essentiality itself. We flag this in the result.")
    print("=" * 70)

    lab = pd.read_csv(LABELS)
    mani = pd.read_csv(MANIFEST)
    ess_per = lab[lab.essential == 1].organism.value_counts()
    targets = []
    for _, r in mani.iterrows():
        if pd.isna(r.accession): continue
        if (CACHE / r.accession / "genome.fna").exists() and \
           (CACHE / r.accession / "genomic.gff").exists() and \
           r.join_rate >= 0.95 and ess_per.get(r.organism, 0) >= 200:
            targets.append((r.organism, r.accession))
    print(f"\norganisms: {len(targets)}")

    # extract CDS per organism once, reuse across k
    print("Extracting CDS sequences ...")
    org_cds = {}   # org -> list[(label, seq)]
    for org, acc in targets:
        contigs = parse_fna(CACHE / acc / "genome.fna")
        lt_to_y = dict(zip(lab[lab.organism==org].locus_tag.astype(str),
                           lab[lab.organism==org].essential.astype(int)))
        rows = []
        for lt, contig, s, e, strand in parse_gff_cds(CACHE / acc / "genomic.gff"):
            if lt not in lt_to_y: continue
            seq = contigs.get(contig, "")[s-1:e]
            if not seq: continue
            if strand == "-": seq = rev_comp(seq)
            if (seq.count("A")+seq.count("C")+seq.count("G")+seq.count("T")) \
                    < len(seq) - 5: continue
            rows.append((lt_to_y[lt], seq))
        org_cds[org] = rows
        ne = sum(1 for y,_ in rows if y==1); nn = len(rows)-ne
        print(f"  {org:<28s} genes={len(rows):>5d}  ess={ne:>5d} non={nn:>5d}")

    report = ["# Discriminative k-mer search: present in essential, "
              "rare/absent in non-essential\n"]

    for k in KS:
        print(f"\n{'='*60}\n=== k = {k} ===")
        # per-org: kmer -> [n_ess_with, n_non_with]; plus class totals
        per_org_disc = {}   # org -> dict kmer -> (p_ess, p_non)
        for org, rows in org_cds.items():
            n_ess = sum(1 for y,_ in rows if y==1)
            n_non = len(rows) - n_ess
            if n_ess < 50 or n_non < 50: continue
            cnt = defaultdict(lambda: [0, 0])
            for y, seq in rows:
                for w in kmer_set(seq, k):
                    cnt[w][0 if y==1 else 1] += 1
            disc = {}
            for w, (ce, cn) in cnt.items():
                p_ess = ce / n_ess; p_non = cn / n_non
                if p_ess >= MIN_PREVALENCE or p_non >= MIN_PREVALENCE:
                    disc[w] = (p_ess, p_non)
            per_org_disc[org] = disc

        # aggregate: mean (p_ess - p_non) across orgs where observed
        agg = defaultdict(list)   # kmer -> list of (p_ess, p_non)
        for org, disc in per_org_disc.items():
            for w, (pe, pn) in disc.items():
                agg[w].append((pe, pn))
        scored = []
        for w, lst in agg.items():
            if len(lst) < MIN_ORGS: continue
            mpe = np.mean([x[0] for x in lst])
            mpn = np.mean([x[1] for x in lst])
            scored.append((w, mpe - mpn, mpe, mpn, len(lst)))

        if not scored:
            print(f"  no k-mers observed in >= {MIN_ORGS} organisms at "
                  f"prevalence >= {MIN_PREVALENCE}")
            report.append(f"\n## k={k}: no qualifying k-mers (too rare/too "
                          f"common across organisms)\n")
            continue

        # top enriched in essentials
        scored.sort(key=lambda r: -r[1])
        print(f"  TOP enriched in ESSENTIAL (mean p_ess - p_non, >= {MIN_ORGS} orgs):")
        print(f"    {'kmer':<14s} {'diff':>6s} {'p_ess':>6s} {'p_non':>6s} {'orgs':>4s}")
        for w, diff, pe, pn, no in scored[:12]:
            print(f"    {w:<14s} {diff:>+6.3f} {pe:>6.3f} {pn:>6.3f} {no:>4d}")

        # the strict "absent" test
        absent_hits = [r for r in scored if r[2] >= ABS_P_ESS and r[3] <= ABS_P_NON]
        print(f"\n  STRICT TEST (present in >={ABS_P_ESS:.0%} of essentials AND "
              f"<={ABS_P_NON:.0%} of non-essentials):")
        if absent_hits:
            for w, diff, pe, pn, no in absent_hits[:20]:
                print(f"    {w:<14s} p_ess={pe:.2f} p_non={pn:.2f}")
        else:
            print(f"    NONE. No k-mer of length {k} is common in essentials "
                  f"yet absent in non-essentials.")
        # max achievable discrimination
        best = scored[0]
        print(f"\n  best single {k}-mer discriminator: {best[0]} "
              f"(p_ess={best[2]:.2f} vs p_non={best[3]:.2f}, diff={best[1]:+.2f})")

        report.append(f"\n## k = {k}\n")
        report.append(f"Top discriminative {k}-mers (present in essential − "
                      f"present in non-essential, mean over ≥{MIN_ORGS} orgs):\n")
        report.append("| k-mer (canonical) | p_ess − p_non | p_ess | p_non | n_orgs |")
        report.append("|---|---:|---:|---:|---:|")
        for w, diff, pe, pn, no in scored[:12]:
            report.append(f"| {w} | {diff:+.3f} | {pe:.3f} | {pn:.3f} | {no} |")
        report.append(f"\n**Strict 'present-in-essential / absent-in-non' "
                      f"test (p_ess≥{ABS_P_ESS:.0%}, p_non≤{ABS_P_NON:.0%}):** "
                      + (f"{len(absent_hits)} k-mers found." if absent_hits
                         else "**none found.**"))

    out = REPO / "outputs" / "essential_kmer_discriminators.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(report))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
