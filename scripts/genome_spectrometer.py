"""Genome spectrometer: a multi-channel trace along the chromosome.

Treat the chromosome as a sweep axis (5' -> 3') and read out every molecular
feature as an intensity channel, stacked like a multi-detector lab instrument.
Sliding 50-kb window; one row per channel; plus a true frequency spectrum
(FFT of the nucleotide signal) revealing the 3-bp coding periodicity peak.

Channels (top to bottom):
  GC%               base composition
  GC skew           (G-C)/(G+C), reveals oriC/terC
  cumulative skew   canonical replication-origin finder
  + strand bias     fraction of CDS on the leading template
  CDS density       genes per 50 kb
  mean gene length  amino acids per CDS
  essential density essential CDS per 50 kb
  conservation      mean family_frac_essential per 50 kb
  paralog load      mean in-genome paralog count per 50 kb
  mobile-element    TE/IS/phage per 50 kb

Plus a frequency panel: power spectrum of the (A+T) signal over a 20-kb central
window, showing the genome-universal 1/3 cycle (coding-frame periodicity).

Outputs:
  outputs/orphan/genome_spectrometer_GMI1000.png
  outputs/orphan/genome_spectrometer_GMI1000_summary.json
"""
import re, json, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")

ORG = "beril_RalstoniaGMI1000"
ACC = "GCF_000009125.1"
WIN = 50_000   # window size, bp
STEP = 10_000  # step
MOB = re.compile(r"transposase|integrase|insertion sequence|IS\d|phage|"
                 r"recombinase|mobile element", re.I)

# labels
ess = {r["locus_tag"]: int(r["essential"])
       for r in csv.DictReader(open(DATA / "labels" / "labels.csv"))
       if r["organism"] == ORG}
orth = {r["locus_tag"]: r
        for r in csv.DictReader(open(DATA / "labels" / "orthology_features.csv"))
        if r["organism"] == ORG}

# sequence
contigs = {}
cur = None; buf = []
for line in (DATA / "genome_cache" / ACC / "genome.fna").read_text().splitlines():
    if line.startswith(">"):
        if cur: contigs[cur] = "".join(buf).upper()
        cur = line[1:].split()[0]; buf = []
    else: buf.append(line.strip())
if cur: contigs[cur] = "".join(buf).upper()
# pick the longest contig (chromosome 1)
ctg = max(contigs, key=lambda c: len(contigs[c]))
seq = contigs[ctg]; L = len(seq)
print(f"chromosome {ctg}: {L:,} bp")

# CDS
cds = []
for line in (DATA / "genome_cache" / ACC / "genomic.gff").read_text().splitlines():
    if "\tCDS\t" not in line: continue
    c = line.split("\t")
    if c[0] != ctg: continue
    m = re.search(r"locus_tag=([^;\n]+)", c[8])
    if not m: continue
    prod = re.search(r"product=([^;\n]+)", c[8])
    cds.append((int(c[3]), int(c[4]), c[6], m.group(1),
                prod.group(1) if prod else ""))
cds.sort()
print(f"CDS on chromosome: {len(cds)}")

# windows
centers = np.arange(WIN//2, L - WIN//2, STEP)
def channel(func):
    return np.array([func(max(0, c - WIN//2), min(L, c + WIN//2)) for c in centers])

# precompute cumulative GC/skew over the sequence
g_arr = np.frombuffer(seq.replace("A","\0").replace("T","\0")
                         .replace("G","\1").replace("C","\0").encode(), dtype=np.int8)
c_arr = np.frombuffer(seq.replace("A","\0").replace("T","\0")
                         .replace("G","\0").replace("C","\1").encode(), dtype=np.int8)
at_arr = 1 - (g_arr + c_arr)
gc_cum = np.cumsum(g_arr + c_arr)
sk_cum = np.cumsum(g_arr - c_arr)

def gc_pct(a, b):
    return (gc_cum[b-1] - (gc_cum[a-1] if a > 0 else 0)) / max(1, b - a)
def gc_skew(a, b):
    g = (gc_cum[b-1] - (gc_cum[a-1] if a > 0 else 0))
    s = (sk_cum[b-1] - (sk_cum[a-1] if a > 0 else 0))
    return s / max(1, g)

# per-window biology channels
cds_mids = np.array([(s + e) // 2 for s, e, *_ in cds])
cds_strand = np.array([1 if st == "+" else -1 for *_, st, _, _ in [(s,e,st,lt,p) for s,e,st,lt,p in cds]])
cds_len_aa = np.array([(e - s + 1) // 3 for s, e, *_ in cds])
cds_lt = [lt for *_, lt, _ in [(s,e,st,lt,p) for s,e,st,lt,p in cds]]
cds_ess = np.array([ess.get(lt, -1) for lt in cds_lt])
cds_cons = np.array([float(orth[lt]["family_frac_essential_leakfree"])
                     if lt in orth and orth[lt]["family_frac_essential_leakfree"] not in ("","nan")
                     else np.nan for lt in cds_lt])
cds_par = np.array([float(orth[lt]["n_paralogs_in_genome"])
                    if lt in orth and orth[lt]["n_paralogs_in_genome"] != ""
                    else 0.0 for lt in cds_lt])
cds_mob = np.array([1 if MOB.search(p or "") else 0
                    for s,e,st,lt,p in cds])

def win_mask(a, b): return (cds_mids >= a) & (cds_mids < b)
def cds_density(a,b): return win_mask(a,b).sum()
def strand_bias(a,b):
    m = win_mask(a,b)
    return float(cds_strand[m].mean()) if m.any() else 0
def mean_len(a,b):
    m = win_mask(a,b)
    return float(cds_len_aa[m].mean()) if m.any() else 0
def ess_density(a,b):
    m = win_mask(a,b) & (cds_ess == 1)
    return int(m.sum())
def cons_mean(a,b):
    m = win_mask(a,b) & ~np.isnan(cds_cons)
    return float(np.nanmean(cds_cons[m])) if m.any() else 0
def par_mean(a,b):
    m = win_mask(a,b)
    return float(cds_par[m].mean()) if m.any() else 0
def mob_density(a,b):
    m = win_mask(a,b) & (cds_mob == 1)
    return int(m.sum())

gc      = channel(gc_pct)
skew    = channel(gc_skew)
cumsk   = np.cumsum(skew - skew.mean())
sbias   = channel(strand_bias)
dens    = channel(cds_density)
ml      = channel(mean_len)
edens   = channel(ess_density)
cons    = channel(cons_mean)
par     = channel(par_mean)
mob     = channel(mob_density)

# oriC / terC from cumulative skew
oriC_idx = int(np.argmin(cumsk)); terC_idx = int(np.argmax(cumsk))
oriC_pos = centers[oriC_idx]; terC_pos = centers[terC_idx]
print(f"oriC ~ {oriC_pos/1e6:.2f} Mb,  terC ~ {terC_pos/1e6:.2f} Mb")

# True frequency spectrum: power of A+T signal in a 20-kb central window
mid = L // 2
seg = at_arr[mid-10000:mid+10000].astype(float)
seg -= seg.mean()
fft = np.fft.rfft(seg); freqs = np.fft.rfftfreq(len(seg), d=1)
power = (np.abs(fft)**2)[1:]; freqs = freqs[1:]   # drop DC

# ---- figure ----
chans = [
    ("GC%",                 gc*100,         "#7F8C8D", "fraction"),
    ("GC skew",             skew,           "#34495E", "(G-C)/(G+C)"),
    ("cumulative skew",     cumsk,          "#9B59B6", "cum sum"),
    ("+ strand bias",       sbias,          "#16A085", "+1 = all leading"),
    ("CDS density",         dens,           "#2980B9", "per 50 kb"),
    ("mean gene length",    ml,             "#1ABC9C", "aa"),
    ("essential density",   edens,          "#C0392B", "per 50 kb"),
    ("conservation",        cons,           "#2C3E50", "family_frac"),
    ("paralog load",        par,            "#E67E22", "mean in-genome"),
    ("mobile-element",      mob,            "#D35400", "per 50 kb"),
]

x_mb = centers / 1e6
fig = plt.figure(figsize=(15, 14))
gs = fig.add_gridspec(len(chans) + 2, 1, hspace=0.05,
                      height_ratios=[0.6]*len(chans) + [0.4, 1.2])

axes = []
for i, (name, y, color, unit) in enumerate(chans):
    ax = fig.add_subplot(gs[i, 0])
    ax.plot(x_mb, y, color=color, lw=1.1)
    ax.fill_between(x_mb, y.min(), y, color=color, alpha=0.18)
    ax.axvline(oriC_pos/1e6, color="green", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(terC_pos/1e6, color="red",   lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel(f"{name}\n[{unit}]", fontsize=8, rotation=0,
                  ha="right", va="center", labelpad=8)
    ax.tick_params(labelsize=7); ax.set_xlim(0, L/1e6)
    if i < len(chans) - 1:
        ax.set_xticklabels([])
    ax.grid(alpha=0.25, axis="x")
    axes.append(ax)
axes[-1].set_xlabel("position along chromosome (Mb)", fontsize=10)
axes[0].text(oriC_pos/1e6, axes[0].get_ylim()[1], " oriC",
             color="green", fontsize=9, va="top")
axes[0].text(terC_pos/1e6, axes[0].get_ylim()[1], " terC",
             color="red", fontsize=9, va="top")

# blank spacer row already in gs; finally the frequency spectrum
ax_fft = fig.add_subplot(gs[-1, 0])
period = 1 / freqs
mask = (period >= 2) & (period <= 200)
ax_fft.plot(period[mask], power[mask], color="#2C3E50", lw=0.8)
# mark the 3-bp codon-frame peak
p3 = power[(period > 2.9) & (period < 3.1)].max() if ((period>2.9)&(period<3.1)).any() else 0
ax_fft.axvline(3, color="#C0392B", ls="--", lw=1.0)
ax_fft.text(3.1, p3*0.95, " 3-bp codon period",
            color="#C0392B", fontsize=9, va="top")
ax_fft.set_xscale("log"); ax_fft.set_yscale("log")
ax_fft.set_xlabel("period (bp)"); ax_fft.set_ylabel("|FFT|² of A+T signal\n(20 kb central window)", fontsize=9)
ax_fft.set_title("Nucleotide-frequency spectrum (true Fourier readout) — the 3-bp peak is the coding signature",
                 fontsize=10)
ax_fft.grid(alpha=0.3, which="both")

fig.suptitle(f"Genome spectrometer — {ORG.replace('beril_','')}  chromosome 1  ({L/1e6:.2f} Mb)\n"
             f"green dashed = oriC ({oriC_pos/1e6:.2f} Mb)  |  red dashed = terC ({terC_pos/1e6:.2f} Mb)",
             fontsize=12, fontweight="bold", y=0.995)
fig.savefig(OUT / "genome_spectrometer_GMI1000.png", dpi=140, bbox_inches="tight")
print(f"\nwrote genome_spectrometer_GMI1000.png")

summary = {
    "organism": ORG, "contig": ctg, "length_bp": int(L),
    "n_cds_on_contig": len(cds),
    "oriC_Mb": round(oriC_pos / 1e6, 3),
    "terC_Mb": round(terC_pos / 1e6, 3),
    "channels": {
        "GC_pct":           {"mean": round(float(gc.mean()*100), 2),
                             "min":  round(float(gc.min()*100), 2),
                             "max":  round(float(gc.max()*100), 2)},
        "essential_per_50kb": {"mean": round(float(edens.mean()), 2),
                               "max":  int(edens.max())},
        "conservation":     {"mean": round(float(cons.mean()), 3)},
        "mobile_per_50kb":  {"mean": round(float(mob.mean()), 2),
                             "max":  int(mob.max())},
    },
    "fft_3bp_period_peak_power": float(p3),
    "windows": int(len(centers)),
}
(OUT / "genome_spectrometer_GMI1000_summary.json").write_text(json.dumps(summary, indent=2))
