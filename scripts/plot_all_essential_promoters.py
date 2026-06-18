"""Scan EVERY essential gene in GMI1000 for regulatory features.

For each CDS marked essential in our atlas:
  - extract 80 bp upstream + the ORF (strand-aware)
  - find best -35 (TTGACA) and -10 (TATAAT) box hits in the upstream window
  - find best Shine-Dalgarno (AGGAGG) hit in the last 20 bp before the start
  - record start codon, stop codon, ORF length, reading frame
Then summarize across the whole essential set + plot the distributions.

Outputs:
  outputs/orphan/essential_promoters_GMI1000.csv     -- one row per gene
  outputs/orphan/essential_promoters_GMI1000_summary.json
  outputs/orphan/essential_promoter_distributions.png
"""
import re, json
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA  = Path("/home/user/cell/data/drive_import")
OUT   = Path("/home/user/cell/outputs/orphan")
ACC   = "GCF_000009125.1"
UP    = 80
COMP  = str.maketrans("ACGTNacgtn", "TGCANtgcan")
def rc(s): return s.translate(COMP)[::-1]

# ---- atlas -> essential locus_tags ----
atlas = pd.read_parquet(OUT / "atlas_beril_RalstoniaGMI1000.parquet")
essential_lt = set(atlas.loc[atlas.essential == 1, "locus_tag"])
print(f"essential genes in atlas: {len(essential_lt):,}")

# ---- load chromosome contigs ----
contigs = {}; cur = None; buf = []
for line in (DATA / "genome_cache" / ACC / "genome.fna").read_text().splitlines():
    if line.startswith(">"):
        if cur: contigs[cur] = "".join(buf).upper()
        cur = line[1:].split()[0]; buf = []
    else: buf.append(line.strip())
if cur: contigs[cur] = "".join(buf).upper()
print(f"contigs: {[(k, len(v)) for k,v in contigs.items()]}")

# ---- parse GFF CDS lines ----
TBL = {  # standard bacterial code (transl_table=11), simplified
 "TTT":"F","TTC":"F","TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
 "ATT":"I","ATC":"I","ATA":"I","ATG":"M","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
 "TCT":"S","TCC":"S","TCA":"S","TCG":"S","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
 "ACT":"T","ACC":"T","ACA":"T","ACG":"T","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
 "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
 "AAT":"N","AAC":"N","AAA":"K","AAG":"K","GAT":"D","GAC":"D","GAA":"E","GAG":"E",
 "TGT":"C","TGC":"C","TGA":"*","TGG":"W","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
 "AGT":"S","AGC":"S","AGA":"R","AGG":"R","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}
def best_match(text, consensus):
    """Hamming-distance scan; return (offset, mismatches, kmer) of best hit."""
    k = len(consensus); best = (-1, k + 1, "")
    if len(text) < k: return best
    for i in range(len(text) - k + 1):
        win = text[i:i+k]
        mm = sum(a != b for a, b in zip(win, consensus))
        if mm < best[1]:
            best = (i, mm, win)
            if mm == 0: break
    return best

rows = []
for line in (DATA / "genome_cache" / ACC / "genomic.gff").read_text().splitlines():
    if "\tCDS\t" not in line: continue
    c = line.split("\t")
    lt_m = re.search(r"locus_tag=([^;\n]+)", c[8])
    if not lt_m: continue
    lt = lt_m.group(1)
    if lt not in essential_lt: continue
    contig = contigs.get(c[0])
    if not contig: continue
    start, end, strand = int(c[3]), int(c[4]), c[6]
    gene = (re.search(r"gene=([^;\n]+)", c[8]) or [None, ""])[1] if re.search(r"gene=([^;\n]+)", c[8]) else ""
    product = (re.search(r"product=([^;\n]+)", c[8]) or [None, ""])
    product = product.group(1) if hasattr(product, "group") else ""

    if strand == "+":
        upstream = contig[max(0, start - 1 - UP) : start - 1]
        orf      = contig[start - 1 : end]
    else:
        upstream = rc(contig[end : end + UP])
        orf      = rc(contig[start - 1 : end])
    if len(upstream) < UP or len(orf) < 6 or len(orf) % 3 != 0:
        continue

    # promoter scan: -35 box anywhere in upstream, -10 box must be DOWNSTREAM of -35
    p35, m35, h35 = best_match(upstream, "TTGACA")
    sub_for_10 = upstream[p35 + 6:] if p35 >= 0 else upstream
    p10_rel, m10, h10 = best_match(sub_for_10, "TATAAT")
    p10 = (p35 + 6 + p10_rel) if p10_rel >= 0 else -1
    spacing = (p10 - p35 - 6) if (p35 >= 0 and p10 >= 0) else None

    # RBS in last 20 bp before the start codon
    rbs_window = upstream[-20:]
    r_off, rmm, rhit = best_match(rbs_window, "AGGAGG")
    rbs_dist_to_atg = (20 - r_off) if r_off >= 0 else None

    start_codon = orf[:3]
    stop_codon  = orf[-3:]
    nt_len = len(orf)
    aa_len = nt_len // 3 - 1
    n_stops_inframe = sum(1 for i in range(0, nt_len - 3, 3)
                          if TBL.get(orf[i:i+3], "X") == "*")

    rows.append(dict(
        locus_tag=lt, gene=gene, product=product[:60],
        contig=c[0], start=start, end=end, strand=strand,
        nt_len=nt_len, aa_len=aa_len,
        start_codon=start_codon, stop_codon=stop_codon,
        in_frame_stops=n_stops_inframe,
        m35_hit=h35, m35_mismatch=m35,
        m35_dist_to_atg=(UP - p35) if p35 >= 0 else None,
        m10_hit=h10, m10_mismatch=m10,
        m10_dist_to_atg=(UP - p10) if p10 >= 0 else None,
        m35_m10_spacing=spacing,
        rbs_hit=rhit, rbs_mismatch=rmm,
        rbs_dist_to_atg=rbs_dist_to_atg,
    ))

df = pd.DataFrame(rows)
print(f"essential CDS analyzed (strand-aware, full upstream available): {len(df):,}")

# ---- summary ----
start_counts = df.start_codon.value_counts().to_dict()
stop_counts  = df.stop_codon.value_counts().to_dict()
print("\nstart codons:", {k: f"{v} ({100*v/len(df):.1f}%)" for k,v in start_counts.items()})
print("stop codons :", {k: f"{v} ({100*v/len(df):.1f}%)" for k,v in stop_counts.items()})
print(f"\n-10 box: median mismatch {df.m10_mismatch.median():.1f}/6; "
      f"good hits (<=1 mm) = {(df.m10_mismatch <= 1).mean():.0%}")
print(f"-35 box: median mismatch {df.m35_mismatch.median():.1f}/6; "
      f"good hits (<=1 mm) = {(df.m35_mismatch <= 1).mean():.0%}")
print(f"-35/-10 spacing: median {df.m35_m10_spacing.median():.0f} bp "
      f"(canonical 17 ± 2);  in [15,19] = {df.m35_m10_spacing.between(15,19).mean():.0%}")
print(f"RBS: median mismatch {df.rbs_mismatch.median():.1f}/6; "
      f"good (<=2 mm) = {(df.rbs_mismatch <= 2).mean():.0%}; "
      f"median dist to ATG = {df.rbs_dist_to_atg.median():.0f} bp")

df.to_csv(OUT / "essential_promoters_GMI1000.csv", index=False)
summary = {
    "n_essential_analyzed": int(len(df)),
    "start_codons": start_counts,
    "stop_codons":  stop_counts,
    "m10_median_mismatch": float(df.m10_mismatch.median()),
    "m10_strong_pct": float((df.m10_mismatch <= 1).mean()),
    "m35_median_mismatch": float(df.m35_mismatch.median()),
    "m35_strong_pct": float((df.m35_mismatch <= 1).mean()),
    "spacing_canonical_pct": float(df.m35_m10_spacing.between(15,19).mean()),
    "rbs_median_mismatch": float(df.rbs_mismatch.median()),
    "rbs_strong_pct": float((df.rbs_mismatch <= 2).mean()),
    "median_aa_len": int(df.aa_len.median()),
}
(OUT / "essential_promoters_GMI1000_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nwrote essential_promoters_GMI1000.csv + summary.json")

# ---- distribution figure ----
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

ax = axes[0, 0]
labels = list(start_counts.keys())
sizes  = list(start_counts.values())
colors = ["#C0392B", "#E67E22", "#F1C40F", "#888"][:len(labels)]
ax.bar(labels, sizes, color=colors)
for i, v in enumerate(sizes):
    ax.text(i, v, f"{100*v/len(df):.0f}%", ha="center", va="bottom", fontsize=10)
ax.set_title(f"Start codon usage (n={len(df):,} essentials)")
ax.set_ylabel("# genes")

ax = axes[0, 1]
labels = list(stop_counts.keys())
sizes  = list(stop_counts.values())
colors = ["#27AE60", "#3498DB", "#9B59B6"][:len(labels)]
ax.bar(labels, sizes, color=colors)
for i, v in enumerate(sizes):
    ax.text(i, v, f"{100*v/len(df):.0f}%", ha="center", va="bottom", fontsize=10)
ax.set_title("Stop codon usage")
ax.set_ylabel("# genes")

ax = axes[0, 2]
ax.hist(df.aa_len.clip(upper=1500), bins=40, color="#2C3E50", alpha=0.8)
ax.axvline(df.aa_len.median(), color="red", ls="--",
           label=f"median {int(df.aa_len.median())} aa")
ax.set_title("Protein length (essential genes)")
ax.set_xlabel("amino acids (clipped at 1500)")
ax.set_ylabel("# genes")
ax.legend()

ax = axes[1, 0]
ax.hist(df.m10_mismatch, bins=range(8), color="#27AE60", alpha=0.85, align="left",
        rwidth=0.85)
ax.set_title(f"-10 box mismatches vs TATAAT\n(strong ≤1: {(df.m10_mismatch<=1).mean():.0%})")
ax.set_xlabel("mismatches (of 6 bp)")
ax.set_ylabel("# genes")
ax.set_xticks(range(7))

ax = axes[1, 1]
ax.hist(df.m35_mismatch, bins=range(8), color="#8E44AD", alpha=0.85, align="left",
        rwidth=0.85)
ax.set_title(f"-35 box mismatches vs TTGACA\n(strong ≤1: {(df.m35_mismatch<=1).mean():.0%})")
ax.set_xlabel("mismatches (of 6 bp)")
ax.set_ylabel("# genes")
ax.set_xticks(range(7))

ax = axes[1, 2]
sp = df.m35_m10_spacing.dropna().astype(int)
ax.hist(sp.clip(5, 40), bins=range(5, 41), color="#E67E22", alpha=0.85, align="left",
        rwidth=0.85)
ax.axvspan(15, 19, color="red", alpha=0.18, label="canonical 17 ± 2")
ax.set_title(f"-35 to -10 spacing\n(canonical {sp.between(15,19).mean():.0%})")
ax.set_xlabel("bp between -35 box and -10 box")
ax.set_ylabel("# genes")
ax.legend()

fig.suptitle(
    f"GMI1000 essential genes — regulatory architecture across {len(df):,} CDSs",
    fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_png = OUT / "essential_promoter_distributions.png"
fig.savefig(out_png, dpi=140, bbox_inches="tight")
print(f"wrote {out_png}")
