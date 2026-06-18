"""ORF-level pattern analysis across all 1,346 essential genes in GMI1000.

For each essential CDS, computes:
  - GC content (whole ORF, and by codon position 1/2/3 -- the wobble signature)
  - codon usage (full 64-codon table, rates per 1000 codons)
  - amino acid composition
  - second-residue distribution (N-terminal Met-cleavage signature)
  - length distribution

Outputs:
  outputs/orphan/orf_patterns_summary.json
  outputs/orphan/orf_patterns.png         -- 6-panel summary plot
"""
import re, json
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")
ACC  = "GCF_000009125.1"
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
def rc(s): return s.translate(COMP)[::-1]

# bacterial codon table (transl_table=11)
TBL = {
 "TTT":"F","TTC":"F","TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
 "ATT":"I","ATC":"I","ATA":"I","ATG":"M","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
 "TCT":"S","TCC":"S","TCA":"S","TCG":"S","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
 "ACT":"T","ACC":"T","ACA":"T","ACG":"T","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
 "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
 "AAT":"N","AAC":"N","AAA":"K","AAG":"K","GAT":"D","GAC":"D","GAA":"E","GAG":"E",
 "TGT":"C","TGC":"C","TGA":"*","TGG":"W","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
 "AGT":"S","AGC":"S","AGA":"R","AGG":"R","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}

# essential set
atlas = pd.read_parquet(OUT / "atlas_beril_RalstoniaGMI1000.parquet")
essential_lt = set(atlas.loc[atlas.essential == 1, "locus_tag"])
print(f"essential genes: {len(essential_lt):,}")

# chromosome contigs
contigs = {}; cur = None; buf = []
for line in (DATA / "genome_cache" / ACC / "genome.fna").read_text().splitlines():
    if line.startswith(">"):
        if cur: contigs[cur] = "".join(buf).upper()
        cur = line[1:].split()[0]; buf = []
    else: buf.append(line.strip())
if cur: contigs[cur] = "".join(buf).upper()
chrom_len = sum(len(v) for v in contigs.values())
chrom_gc = sum(s.count("G")+s.count("C") for s in contigs.values()) / chrom_len
print(f"genome GC content: {100*chrom_gc:.1f}%")

# per-ORF aggregation
codon_counter   = Counter()
aa_counter      = Counter()
second_aa_ctr   = Counter()   # 2nd residue after Met (Met-cleavage signature)
gc_total, gc_p1, gc_p2, gc_p3 = [], [], [], []
lengths = []
gc_orf_per_gene = []

for line in (DATA / "genome_cache" / ACC / "genomic.gff").read_text().splitlines():
    if "\tCDS\t" not in line: continue
    c = line.split("\t")
    lt_m = re.search(r"locus_tag=([^;\n]+)", c[8])
    if not lt_m or lt_m.group(1) not in essential_lt: continue
    contig = contigs.get(c[0])
    if not contig: continue
    start, end, strand = int(c[3]), int(c[4]), c[6]
    orf = contig[start-1:end]
    if strand == "-": orf = rc(orf)
    if len(orf) < 6 or len(orf) % 3 != 0: continue
    # exclude trailing stop codon for codon/aa counting
    body = orf[:-3]
    lengths.append(len(orf))

    g = body.count("G") + body.count("C")
    gc_total.append(g / len(body))

    p1 = body[0::3]; p2 = body[1::3]; p3 = body[2::3]
    gc_p1.append((p1.count("G") + p1.count("C")) / max(1, len(p1)))
    gc_p2.append((p2.count("G") + p2.count("C")) / max(1, len(p2)))
    gc_p3.append((p3.count("G") + p3.count("C")) / max(1, len(p3)))

    for i in range(0, len(body), 3):
        codon = body[i:i+3]
        if "N" in codon: continue
        codon_counter[codon] += 1
        aa = TBL.get(codon, "X")
        if aa not in ("*", "X"): aa_counter[aa] += 1
    # 2nd-residue: codon at position 3:6
    if len(body) >= 6:
        c2 = body[3:6]
        aa2 = TBL.get(c2, "X")
        if aa2 not in ("*", "X"):
            second_aa_ctr[aa2] += 1

n = len(lengths)
print(f"ORFs analyzed: {n:,}")
print(f"median ORF length: {int(np.median(lengths)):,} nt  "
      f"({int(np.median(lengths)//3 - 1)} aa)")
print(f"\nGC content of essential ORFs:")
print(f"  overall: {100*np.mean(gc_total):.1f}%  (chromosome {100*chrom_gc:.1f}%)")
print(f"  pos 1  : {100*np.mean(gc_p1):.1f}%  (1st codon position)")
print(f"  pos 2  : {100*np.mean(gc_p2):.1f}%  (2nd codon position — most constrained)")
print(f"  pos 3  : {100*np.mean(gc_p3):.1f}%  (3rd wobble position — most free)")

total_codons = sum(codon_counter.values())
print(f"\ntotal codons counted: {total_codons:,}")

# codon usage rates per 1000 codons
codons_sorted = sorted(codon_counter.items(), key=lambda x: -x[1])
print("\ntop 12 codons (per 1000):")
for codon, n_c in codons_sorted[:12]:
    print(f"  {codon} ({TBL[codon]})  {1000*n_c/total_codons:>6.1f}")

# amino-acid composition
total_aa = sum(aa_counter.values())
aa_sorted = sorted(aa_counter.items(), key=lambda x: -x[1])
print("\namino acid composition (top 8):")
for aa, n_aa in aa_sorted[:8]:
    print(f"  {aa}  {100*n_aa/total_aa:>5.1f}%")

# 2nd residue (N-terminal Met-aminopeptidase substrate)
total2 = sum(second_aa_ctr.values())
print(f"\nsecond residue (after start Met) -- N-term Met-cleavage signature:")
print(f"  small (A,C,G,P,S,T,V): "
      f"{100*sum(second_aa_ctr.get(a,0) for a in 'ACGPSTV')/max(1,total2):.1f}%"
      f"   (cleavage-prone)")
print(f"  bulky (F,I,L,M,W,Y,H,Q,K,R,D,E,N): "
      f"{100*sum(second_aa_ctr.get(a,0) for a in 'FILMWYHQKRDEN')/max(1,total2):.1f}%"
      f"   (Met retained)")

summary = {
    "n_orfs": n,
    "median_orf_nt": int(np.median(lengths)),
    "median_orf_aa": int(np.median(lengths)//3 - 1),
    "gc_genome_pct":    round(100*chrom_gc, 2),
    "gc_orf_mean_pct":  round(100*float(np.mean(gc_total)), 2),
    "gc_p1_mean_pct":   round(100*float(np.mean(gc_p1)), 2),
    "gc_p2_mean_pct":   round(100*float(np.mean(gc_p2)), 2),
    "gc_p3_mean_pct":   round(100*float(np.mean(gc_p3)), 2),
    "total_codons": total_codons,
    "top12_codons_per1000": {c: round(1000*n_c/total_codons, 2) for c, n_c in codons_sorted[:12]},
    "amino_acid_pct": {a: round(100*c/total_aa, 2) for a, c in aa_sorted},
    "second_residue_small_pct": round(100*sum(second_aa_ctr.get(a,0)
                                              for a in "ACGPSTV")/max(1,total2), 2),
}
(OUT / "orf_patterns_summary.json").write_text(json.dumps(summary, indent=2))

# ---- 6-panel figure ----
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# (1) length distribution
ax = axes[0, 0]
ax.hist(np.array(lengths)/3 - 1, bins=50, color="#2C3E50", alpha=0.85,
        range=(0, 1500))
ax.axvline(np.median(lengths)/3 - 1, color="red", ls="--",
           label=f"median {int(np.median(lengths)/3 - 1)} aa")
ax.set_title(f"ORF length (n={n:,} essentials)")
ax.set_xlabel("amino acids (clipped at 1500)"); ax.set_ylabel("# genes")
ax.legend()

# (2) GC by codon position
ax = axes[0, 1]
pos_means = [100*np.mean(gc_p1), 100*np.mean(gc_p2), 100*np.mean(gc_p3)]
pos_std   = [100*np.std(gc_p1),   100*np.std(gc_p2),   100*np.std(gc_p3)]
bars = ax.bar(["pos 1", "pos 2", "pos 3 (wobble)"], pos_means,
              yerr=pos_std, color=["#3498DB", "#27AE60", "#E67E22"],
              alpha=0.9, capsize=6)
ax.axhline(100*chrom_gc, color="gray", ls="--",
           label=f"genome {100*chrom_gc:.1f}%")
for i, v in enumerate(pos_means):
    ax.text(i, v+1, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_title("GC content by codon position")
ax.set_ylabel("% GC"); ax.set_ylim(0, 105)
ax.legend()

# (3) overall ORF GC distribution
ax = axes[0, 2]
ax.hist(np.array(gc_total)*100, bins=40, color="#16A085", alpha=0.85)
ax.axvline(100*chrom_gc, color="red", ls="--",
           label=f"genome avg {100*chrom_gc:.1f}%")
ax.set_title("Whole-ORF GC content per gene")
ax.set_xlabel("% GC"); ax.set_ylabel("# genes")
ax.legend()

# (4) top codons (per 1000)
ax = axes[1, 0]
top_n = 20
top_codons = codons_sorted[:top_n]
labels = [f"{c} ({TBL[c]})" for c, _ in top_codons]
vals   = [1000*n_c/total_codons for _, n_c in top_codons]
ys = list(range(top_n))[::-1]
ax.barh(ys, vals, color="#8E44AD", alpha=0.85)
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("codons per 1000")
ax.set_title("Top 20 codons used in essential ORFs")

# (5) amino acid composition
ax = axes[1, 1]
aa_pct = sorted([(a, 100*c/total_aa) for a, c in aa_counter.items()],
                key=lambda x: -x[1])
xs = [a for a, _ in aa_pct]; ys = [v for _, v in aa_pct]
colors = ["#27AE60" if a in "GACSTPV" else "#3498DB" if a in "FILMWY"
          else "#E74C3C" if a in "DE" else "#8E44AD" if a in "KRH"
          else "#888" for a in xs]
ax.bar(xs, ys, color=colors, alpha=0.9)
ax.set_title("Amino-acid composition (essential proteome)")
ax.set_ylabel("% of residues")
ax.set_xlabel("amino acid")

# (6) second residue (after start Met) - Met-cleavage signature
ax = axes[1, 2]
second_pct = sorted([(a, 100*c/max(1,total2)) for a, c in second_aa_ctr.items()],
                    key=lambda x: -x[1])
xs2 = [a for a, _ in second_pct]; ys2 = [v for _, v in second_pct]
colors2 = ["#27AE60" if a in "ACGPSTV" else "#888" for a in xs2]
ax.bar(xs2, ys2, color=colors2, alpha=0.9)
ax.set_title("2nd residue (after Met) — green=Met-cleaved")
ax.set_ylabel("% of essentials")
ax.set_xlabel("amino acid at position 2")
ax.text(0.98, 0.95,
        f"small (cleaved): {summary['second_residue_small_pct']:.0f}%",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold", color="#27AE60")

fig.suptitle(
    f"ORF pattern across {n:,} essential genes in GMI1000  "
    f"(genome GC {100*chrom_gc:.0f}%)",
    fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_png = OUT / "orf_patterns.png"
fig.savefig(out_png, dpi=140, bbox_inches="tight")
print(f"\nwrote {out_png}")
