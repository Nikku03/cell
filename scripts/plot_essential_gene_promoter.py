"""Plot the regulatory features of an essential gene in GMI1000.

Picks a well-known essential CDS, extracts the upstream + ORF nucleotide
sequence, finds: -35 / -10 sigma70 promoter boxes, Shine-Dalgarno (RBS),
start codon, stop codon, reading frame. Prints an annotated ASCII view and
saves a matplotlib figure.
"""
import re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

DATA = Path("/home/user/cell/data/drive_import")
ACC  = "GCF_000009125.1"
GENE = "gyrA"     # universally essential -- clean illustrative example

# ---- 1) parse GFF: find the gyrA CDS coordinates ----
gff = (DATA / "genome_cache" / ACC / "genomic.gff").read_text().splitlines()
cds_line = next(l for l in gff if "\tCDS\t" in l and f"gene={GENE};" in l)
c = cds_line.split("\t")
contig, start, end, strand = c[0], int(c[3]), int(c[4]), c[6]
product = re.search(r"product=([^;]+)", c[8]).group(1)
locus  = re.search(r"locus_tag=([^;]+)", c[8]).group(1)
print(f"{GENE} ({locus}): {contig}:{start}-{end} ({strand}) {product}")

# ---- 2) load chromosome, extract CDS + 80 bp upstream ----
fna_path = DATA / "genome_cache" / ACC / "genome.fna"
seq = []; in_chrom = False
for line in fna_path.read_text().splitlines():
    if line.startswith(">"):
        in_chrom = (contig in line); continue
    if in_chrom: seq.append(line.strip())
chrom = "".join(seq).upper()

UP = 80     # bases before the start codon to look at for promoter/RBS
upstream = chrom[start-1-UP : start-1]              # 0-indexed slice
orf      = chrom[start-1 : end]                     # includes stop codon
# (gyrA is +strand so no reverse-complement needed)

# ---- 3) translate ORF and confirm start/stop ----
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
start_codon = orf[:3]; stop_codon = orf[-3:]
aa = "".join(TBL.get(orf[i:i+3], "X") for i in range(0, len(orf)-3, 3))
print(f"start codon: {start_codon}  ({'canonical ATG' if start_codon=='ATG' else 'alt-start'})")
print(f"stop codon : {stop_codon}  ({'TAA-ochre' if stop_codon=='TAA' else 'TAG-amber' if stop_codon=='TAG' else 'TGA-opal'})")
print(f"ORF length : {len(orf):,} nt = {len(orf)//3 - 1} codons + stop")
print(f"first 8 aa : {aa[:8]}   last 5 aa: {aa[-5:]}")

# ---- 4) scan upstream for promoter elements and RBS ----
def best_match(text, consensus, max_mismatch=2):
    """Hamming search for the best k-mer hit to consensus; return (pos, mismatch, hit)."""
    k = len(consensus); best = (None, k+1, "")
    for i in range(len(text) - k + 1):
        win = text[i:i+k]
        mm = sum(a != b for a, b in zip(win, consensus))
        if mm < best[1]: best = (i, mm, win)
    return best

# -35 box (TTGACA) typically 30-38 bp upstream of TSS, RBS 5-15 upstream of ATG
# We don't know the TSS exactly so search the whole upstream window.
pos35, mm35, hit35 = best_match(upstream, "TTGACA", 2)
pos10, mm10, hit10 = best_match(upstream, "TATAAT", 2)
# RBS: anti-SD complement of 16S 3' end is AGGAGG; allow GGAGG/AGGAG variants
rbs_pos, rbs_mm, rbs_hit = best_match(upstream[-20:], "AGGAGG", 2)
rbs_pos = (len(upstream) - 20) + rbs_pos if rbs_pos is not None else None

print(f"\nupstream {UP} bp:")
print(f"  -35 best hit: '{hit35}' at -{UP-pos35} (mismatch {mm35}/6 vs TTGACA)")
print(f"  -10 best hit: '{hit10}' at -{UP-pos10} (mismatch {mm10}/6 vs TATAAT)")
print(f"  RBS best hit: '{rbs_hit}' at -{UP-rbs_pos} (mismatch {rbs_mm}/6 vs AGGAGG)")

# ---- 5) ASCII annotated track ----
print("\nANNOTATED SEQUENCE (each '.' = 10 bp ruler)")
ruler = "".join((str(((i+1)//10)%10) if (i+1)%10==0 else " ") for i in range(UP+30))
text  = upstream + orf[:30]
mark  = [" "] * len(text)
def stamp(pos, length, ch):
    for i in range(pos, pos+length):
        if 0 <= i < len(mark): mark[i] = ch
stamp(pos35, 6, "5")          # -35 box
stamp(pos10, 6, "1")          # -10 box
if rbs_pos is not None: stamp(rbs_pos, 6, "R")   # RBS
stamp(UP, 3, "S")             # start codon ATG
print("       " + ruler)
print(" 5'... " + text + " ...3'")
print("       " + "".join(mark))
print(f"       legend: 5 = -35 box  1 = -10 box  R = RBS (Shine-Dalgarno)  S = start codon (ATG)")

# ---- 6) save a labeled figure ----
fig, ax = plt.subplots(figsize=(13, 4.0))
TOTAL = UP + 60                                    # window: upstream + 60 nt of ORF
seq_window = upstream + orf[:60]
xs = list(range(-UP, 60))                          # x in nt relative to ATG (which is at 0)

# baseline track
ax.add_patch(Rectangle((-UP, 0.40), TOTAL, 0.04, color="#888"))
# annotate features
feat = [
    (xs[pos35]      , 6, "-35 box (TTGACA)", f"'{hit35}'\n{mm35} mismatch", "#8E44AD"),
    (xs[pos10]      , 6, "-10 box (TATAAT)", f"'{hit10}'\n{mm10} mismatch", "#27AE60"),
    (xs[rbs_pos] if rbs_pos is not None else -10, 6,
        "Shine-Dalgarno / RBS", f"'{rbs_hit}'\n{rbs_mm} mismatch", "#E67E22"),
    (0, 3, "Start codon", f"{start_codon} (Met)", "#C0392B"),
]
for x, w, label, sub, col in feat:
    ax.add_patch(Rectangle((x, 0.30), w, 0.24, color=col, alpha=0.85))
    ax.annotate(label, xy=(x + w/2, 0.56), ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=col)
    ax.annotate(sub, xy=(x + w/2, 0.28), ha="center", va="top",
                fontsize=7, color=col)
# CDS arrow
ax.add_patch(FancyArrowPatch((0, 0.42), (60, 0.42),
    arrowstyle='-|>', mutation_scale=20, lw=8, color='#2C3E50'))
ax.annotate("CDS  (reading frame +1, → 5' to 3')",
            xy=(30, 0.66), ha="center", fontsize=10, fontweight="bold")
ax.annotate(f"M  {aa[1]}  {aa[2]}  {aa[3]}  {aa[4]}  {aa[5]} ... [{len(aa)-1} aa] ... *",
            xy=(30, 0.49), ha="center", fontsize=8, color="white", fontweight="bold")
# stop-codon callout on the right (we only drew the first 60 nt of the ORF)
ax.annotate(f"Stop: {stop_codon}\nat +{len(orf)} ", xy=(60, 0.42),
            xytext=(63, 0.42), ha="left", va="center", fontsize=9, color="#7F0000",
            arrowprops=dict(arrowstyle='->', color="#7F0000"))
ax.axvline(0, ls="--", color="black", lw=0.7)
ax.text(0, 0.05, "ATG = +1\n(start codon)", ha="center", fontsize=8)
ax.text(-UP, 0.05, f"-{UP} bp\n(upstream)", ha="center", fontsize=8)
ax.set_xlim(-UP - 5, 80)
ax.set_ylim(0, 0.8)
ax.set_yticks([])
ax.set_xlabel("position relative to start codon (nt)")
ax.set_title(f"{GENE} ({locus}) — {product}\n"
             f"chromosome NC_003295.1: {start:,}–{end:,} (+ strand) · ESSENTIAL in GMI1000",
             fontsize=11)
ax.spines[['top','right','left']].set_visible(False)
out = Path("/home/user/cell/outputs/orphan/gyrA_promoter_map.png")
fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nfigure saved -> {out}")
