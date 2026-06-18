"""Nucleotide-level motif scan across all 1,346 essential GMI1000 genes.

Four analyses:
  1) Sequence logo around the START codon (-30..+20, anchored at +1=ATG)
  2) Sequence logo around the STOP codon (-15..+30, anchored on stop codon)
  3) Top over-represented 5-mers (relative to genome dinucleotide background)
  4) Rho-independent terminator signature: U-rich (T-rich on the coding
     strand) tail just downstream of stop codons

Logos use bit-information (height = 2 - Shannon entropy), the standard.
"""
import re, json, math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")
ACC  = "GCF_000009125.1"
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
def rc(s): return s.translate(COMP)[::-1]

# --- inputs ---
atlas = pd.read_parquet(OUT / "atlas_beril_RalstoniaGMI1000.parquet")
essential_lt = set(atlas.loc[atlas.essential == 1, "locus_tag"])
contigs = {}; cur = None; buf = []
for line in (DATA / "genome_cache" / ACC / "genome.fna").read_text().splitlines():
    if line.startswith(">"):
        if cur: contigs[cur] = "".join(buf).upper()
        cur = line[1:].split()[0]; buf = []
    else: buf.append(line.strip())
if cur: contigs[cur] = "".join(buf).upper()
all_seq = "".join(contigs.values())
print(f"essentials: {len(essential_lt):,}; genome length: {len(all_seq):,} bp")

# --- gather strand-aware windows around start and stop codons ---
START_UP, START_DN = 30, 20      # window [-30 .. +20] around +1=ATG
STOP_UP,  STOP_DN  = 15, 30      # window around stop codon

start_windows = []   # each is a 50-bp string anchored at ATG=position START_UP
stop_windows  = []   # 45-bp string anchored at stop-codon start = STOP_UP
downstream60  = []   # 60 bp downstream of the stop codon (for terminator scan)
orf_bodies    = []   # for k-mer enrichment (CDS interior, no start/stop)

for line in (DATA / "genome_cache" / ACC / "genomic.gff").read_text().splitlines():
    if "\tCDS\t" not in line: continue
    c = line.split("\t")
    lt_m = re.search(r"locus_tag=([^;\n]+)", c[8])
    if not lt_m or lt_m.group(1) not in essential_lt: continue
    contig = contigs.get(c[0])
    if not contig: continue
    s, e, strand = int(c[3]), int(c[4]), c[6]

    if strand == "+":
        # start codon at index s-1; stop codon at e-3 .. e-1
        sw = contig[s-1-START_UP : s-1+START_DN]
        ew = contig[e-3-STOP_UP : e-3+STOP_DN]
        dn = contig[e : e+60]
        body = contig[s+2 : e-3]                # interior (no start, no stop)
    else:
        # reverse-complement to put start codon at the front
        ws = contig[e-START_DN : e+START_UP]
        sw = rc(ws)
        we = contig[s-1-STOP_DN : s-1+3+STOP_UP]
        ew = rc(we)
        dn = rc(contig[max(0, s-1-60) : s-1])
        body = rc(contig[s+2 : e-3])
    if len(sw) == START_UP + START_DN: start_windows.append(sw)
    if len(ew) == STOP_UP + STOP_DN:   stop_windows.append(ew)
    if len(dn) == 60:                  downstream60.append(dn)
    if len(body) >= 60:                orf_bodies.append(body)

print(f"start windows: {len(start_windows):,}")
print(f"stop windows : {len(stop_windows):,}")
print(f"downstream-60: {len(downstream60):,}")

# --- position-specific nucleotide frequency matrices ---
def pwm(windows):
    L = len(windows[0])
    M = np.zeros((4, L))
    idx = {"A":0,"C":1,"G":2,"T":3}
    for w in windows:
        for i, ch in enumerate(w):
            if ch in idx: M[idx[ch], i] += 1
    M = M / np.maximum(M.sum(axis=0, keepdims=True), 1)
    return M     # 4xL (rows: A,C,G,T)

PWM_start = pwm(start_windows)
PWM_stop  = pwm(stop_windows)
print(f"start ATG fraction at expected positions {START_UP}:{START_UP+3}: "
      f"{PWM_start[0,START_UP]:.2f} A, "
      f"{PWM_start[3,START_UP+1]:.2f} T, "
      f"{PWM_start[2,START_UP+2]:.2f} G")

# --- k-mer enrichment (5-mers) in CDS body vs. expected from base-composition ---
K = 5
body_concat = "".join(orf_bodies)
body_kmers = Counter()
for i in range(len(body_concat) - K + 1):
    km = body_concat[i:i+K]
    if "N" not in km: body_kmers[km] += 1
total_body_kmers = sum(body_kmers.values())
# expected frequency from mononucleotide background of the body
mono = Counter(body_concat); tot_mono = sum(mono.get(b,0) for b in "ACGT")
freq = {b: mono.get(b,0)/tot_mono for b in "ACGT"}
expected = {km: freq[km[0]]*freq[km[1]]*freq[km[2]]*freq[km[3]]*freq[km[4]]
            for km in body_kmers}
enrich = []
for km, cnt in body_kmers.items():
    obs = cnt / total_body_kmers
    exp = expected[km] or 1e-12
    log2_fold = math.log2(obs / exp)
    enrich.append((km, log2_fold, cnt, obs, exp))
enrich.sort(key=lambda x: -x[1])
print("\nTop 10 most over-represented 5-mers in essential ORF bodies:")
for km, lf, cnt, obs, exp in enrich[:10]:
    print(f"  {km}  log2-fold {lf:+.2f}  obs {1000*obs:.2f}/1000  exp {1000*exp:.2f}")

# --- rho-independent terminator signature: T-tract density downstream of stop ---
# Stop codon is on the coding strand -> the mRNA tail is on the same strand
# (since mRNA reads in the same direction). A T-tract on the coding-strand DNA
# corresponds to a U-tract on the mRNA -> the textbook rho-independent signal.
# We measure the fraction of T in 6-nt windows from +1 to +50 after the stop.
t_density = np.zeros(50)
for dn in downstream60:
    s = dn[:50]
    for i in range(50):
        if i < len(s) and s[i] == "T": t_density[i] += 1
t_density = t_density / len(downstream60)

# also: average T fraction in a 6-nt sliding window
t6 = np.array([t_density[i:i+6].mean() if i+6 <= 50 else np.nan
               for i in range(50)])
peak = int(np.nanargmax(t6))
print(f"\nT-tract: peak 6-bp T-density at +{peak+1}..+{peak+6}: {t6[peak]:.2f}"
      f"  (baseline genome T = {(all_seq.count('T')/len(all_seq)):.2f})")

# --- helpers: draw bit-information sequence logos ---
def info_content(M):
    """Shannon entropy -> information bits per position (column).  R = 2 - H."""
    H = -np.nansum(np.where(M > 0, M * np.log2(M), 0), axis=0)
    return 2 - H

LETTER_COLORS = {"A": "#27AE60", "C": "#3498DB", "G": "#E67E22", "T": "#C0392B"}

def draw_logo(ax, M, x0, title):
    """Plot a sequence logo where stacked letter heights = bit contribution."""
    R = info_content(M)
    L = M.shape[1]
    letters = ["A","C","G","T"]
    for x in range(L):
        order = sorted(range(4), key=lambda i: M[i, x])
        y = 0
        for i in order:
            h = M[i, x] * R[x]                  # bits
            if h <= 0.01: continue
            tp = TextPath((0, 0), letters[i], size=1, prop=None)
            verts = tp.vertices
            xmin, ymin = verts.min(axis=0); xmax, ymax = verts.max(axis=0)
            sx = 0.95 / (xmax - xmin); sy = h / (ymax - ymin)
            tform = (Affine2D()
                     .translate(-xmin, -ymin)
                     .scale(sx, sy)
                     .translate(x + x0 + 0.025, y))
            patch = PathPatch(tp.transformed(tform),
                              facecolor=LETTER_COLORS[letters[i]],
                              edgecolor="none")
            ax.add_patch(patch)
            y += h
    ax.set_xlim(x0 - 0.5, x0 + L - 0.5)
    ax.set_ylim(0, 2.05)
    ax.set_ylabel("bits"); ax.set_title(title, fontsize=11)

# --- 4-panel figure ---
fig = plt.figure(figsize=(17, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

# (1) start-codon logo (-30..+20)
ax1 = fig.add_subplot(gs[0, 0])
draw_logo(ax1, PWM_start, -START_UP, "Sequence logo: start codon context (1,346 essentials)")
ax1.axvline(0, color="black", lw=0.7, ls="--")
ax1.text(0, 2.1, "+1 (ATG)", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.text(-13, 2.1, "Shine-Dalgarno", ha="center", va="bottom",
         fontsize=9, color="#888")
ax1.set_xlabel("position relative to start codon (nt)")
ax1.set_xticks(range(-30, 21, 5))

# (2) stop-codon logo (-15..+30)
ax2 = fig.add_subplot(gs[0, 1])
draw_logo(ax2, PWM_stop, -STOP_UP, "Sequence logo: stop codon context (1,346 essentials)")
ax2.axvline(0, color="black", lw=0.7, ls="--")
ax2.text(0, 2.1, "stop (T**)", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_xlabel("position relative to stop codon (nt)")
ax2.set_xticks(range(-15, 31, 5))

# (3) top over-represented 5-mers
ax3 = fig.add_subplot(gs[1, 0])
top = enrich[:20]
labels = [km for km, *_ in top]
vals   = [lf for _, lf, *_ in top]
ys = list(range(len(top)))[::-1]
bars = ax3.barh(ys, vals, color="#8E44AD", alpha=0.85)
ax3.set_yticks(ys); ax3.set_yticklabels(labels, family="monospace", fontsize=9)
ax3.set_xlabel("log2 fold-enrichment over mononucleotide background")
ax3.set_title("Top 20 over-represented 5-mers in essential ORF bodies")
ax3.axvline(0, color="black", lw=0.6)
# annotate each bar with the observed-per-1000 count
for i, (_, lf, cnt, obs, exp) in enumerate(top):
    ax3.text(lf + 0.02, ys[i], f"{1000*obs:.1f}/1000",
             va="center", fontsize=7, color="#444")

# (4) T-tract density downstream of stop (rho-independent terminator signal)
ax4 = fig.add_subplot(gs[1, 1])
xs = np.arange(1, 51)
ax4.bar(xs, t_density*100, color="#C0392B", alpha=0.75, label="T fraction per nt")
ax4.plot(xs[:len(t6)], t6*100, color="black", lw=2,
         label="6-nt sliding T-content")
baseline = 100*(all_seq.count('T')/len(all_seq))
ax4.axhline(baseline, color="gray", ls="--",
            label=f"genome T baseline ({baseline:.0f}%)")
ax4.axvspan(peak+1, peak+6, color="orange", alpha=0.2,
            label=f"peak +{peak+1}..{peak+6}")
ax4.set_xlabel("nt downstream of stop codon")
ax4.set_ylabel("% T (coding strand) = U on mRNA")
ax4.set_title("Rho-independent terminator signature: U-tract just past the stop")
ax4.legend(fontsize=8, loc="upper right")

fig.suptitle(
    f"Nucleotide motifs across 1,346 essential genes in GMI1000",
    fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_png = OUT / "nt_motifs.png"
fig.savefig(out_png, dpi=140, bbox_inches="tight")
print(f"\nwrote {out_png}")

# --- summary JSON ---
summary = {
    "n_essentials": len(start_windows),
    "start_consensus_-30..+20": {
        "ATG_match_at_+1": float(PWM_start[0,START_UP]),
        "T_match_at_+2"  : float(PWM_start[3,START_UP+1]),
        "G_match_at_+3"  : float(PWM_start[2,START_UP+2]),
    },
    "top10_5mers": [
        {"kmer": km, "log2_fold": round(lf,3), "count": int(cnt),
         "obs_per1000": round(1000*obs,2), "exp_per1000": round(1000*exp,2)}
        for km, lf, cnt, obs, exp in enrich[:10]
    ],
    "rho_independent_T_tract": {
        "peak_position": int(peak)+1,
        "peak_T_fraction_6bp": round(float(t6[peak]),3),
        "genome_T_baseline": round(all_seq.count('T')/len(all_seq), 3),
    },
}
(OUT / "nt_motifs_summary.json").write_text(json.dumps(summary, indent=2))
print(f"wrote nt_motifs_summary.json")
