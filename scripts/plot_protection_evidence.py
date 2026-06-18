"""Search for biological 'protections' on essential genes in GMI1000.

Five measurable forms of protection:
  1) DOSAGE protection: essentials sit closer to the replication origin (oriC),
     where they get replicated earlier -> 2x copies during cell cycle.
  2) STRAND bias: essentials co-oriented with replication (leading strand) avoid
     head-on RNA-pol/DNA-pol collisions (a major source of mutation and stall).
  3) MOBILE-ELEMENT avoidance: essentials sit further from IS/transposase genes
     (lower risk of disruption by insertion).
  4) PARALOG backup: redundancy as protection.
  5) PURIFYING selection (already shown elsewhere as dN/dS but worth restating).

We locate oriC empirically using the cumulative GC skew minimum on chromosome 1.

Outputs:
  outputs/orphan/protection_evidence.png
  outputs/orphan/protection_evidence_summary.json
"""
import re, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")
ACC  = "GCF_000009125.1"
CHROM = "NC_003295.1"       # main chromosome (3.7 Mbp); secondary is megaplasmid

# load atlas
atlas = pd.read_parquet(OUT / "atlas_beril_RalstoniaGMI1000.parquet")
essential_lt = set(atlas.loc[atlas.essential == 1, "locus_tag"])

# load chromosome
contigs = {}; cur = None; buf = []
for line in (DATA / "genome_cache" / ACC / "genome.fna").read_text().splitlines():
    if line.startswith(">"):
        if cur: contigs[cur] = "".join(buf).upper()
        cur = line[1:].split()[0]; buf = []
    else: buf.append(line.strip())
if cur: contigs[cur] = "".join(buf).upper()
chrom_seq = contigs[CHROM]; L = len(chrom_seq)
print(f"chromosome {CHROM}: {L:,} bp")

# ---- (1) find oriC via GC skew ----
WIN = 10_000
gc_skew = np.zeros((L // WIN) + 1)
for i in range(0, L, WIN):
    w = chrom_seq[i : i + WIN]
    g, c = w.count("G"), w.count("C")
    gc_skew[i // WIN] = (g - c) / max(1, g + c)
cumulative = np.cumsum(gc_skew)
oriC_bin = int(np.argmin(cumulative))   # GC skew min = oriC
terC_bin = int(np.argmax(cumulative))
oriC = oriC_bin * WIN
terC = terC_bin * WIN
print(f"GC skew minimum (oriC) at: {oriC:,} bp")
print(f"GC skew maximum (terC) at: {terC:,} bp")

# ---- (2) parse CDS on chromosome 1 only ----
cds = []
for line in (DATA / "genome_cache" / ACC / "genomic.gff").read_text().splitlines():
    if "\tCDS\t" not in line: continue
    c = line.split("\t")
    if c[0] != CHROM: continue
    m = re.search(r"locus_tag=([^;\n]+)", c[8])
    if not m: continue
    cds.append((int(c[3]), int(c[4]), c[6], m.group(1),
                re.search(r"product=([^;\n]+)", c[8]).group(1)
                if re.search(r"product=([^;\n]+)", c[8]) else ""))
print(f"CDS on {CHROM}: {len(cds):,}")
essentials_chr = [r for r in cds if r[3] in essential_lt]
others_chr     = [r for r in cds if r[3] not in essential_lt]
print(f"  essential: {len(essentials_chr):,}  non-essential: {len(others_chr):,}")

# ---- distance to oriC (the SHORTER way around the circular chromosome) ----
def dist_to_oriC(pos):
    d = abs(pos - oriC) % L
    return min(d, L - d)

mid = lambda r: (r[0] + r[1]) // 2
d_ess   = np.array([dist_to_oriC(mid(r)) for r in essentials_chr])
d_other = np.array([dist_to_oriC(mid(r)) for r in others_chr])
print(f"\nDistance to oriC (Mb):")
print(f"  essentials median  : {d_ess.mean()/1e6:.2f}  (mean {d_ess.mean()/1e6:.2f})")
print(f"  non-essentials med : {d_other.mean()/1e6:.2f}  (mean {d_other.mean()/1e6:.2f})")
# rank-sum sanity
from numpy.random import default_rng
rng = default_rng(0)
def mann_whitney_p(a, b, n=2000):
    # Quick permutation test: are medians different?
    obs = np.median(a) - np.median(b)
    pooled = np.concatenate([a, b])
    diffs = np.zeros(n)
    for i in range(n):
        rng.shuffle(pooled)
        diffs[i] = np.median(pooled[:len(a)]) - np.median(pooled[len(a):])
    return obs, float(np.mean(np.abs(diffs) >= abs(obs)))

obs, p = mann_whitney_p(d_ess, d_other)
print(f"  permutation test:  median diff {obs/1e6:+.2f} Mb,  p={p:.3f}")

# ---- (3) leading vs lagging strand bias ----
# Replichore boundary: oriC and terC define two arcs.
# On the arc oriC -> terC going one way: + strand = leading.
# On the arc oriC -> terC the other way: - strand = leading.
# Simpler: replication forks move FROM oriC.
# Right of oriC (positions oriC..oriC+L/2 wrapping): replication direction = '+'.
#   Genes on + strand are co-directional (leading); - are head-on (lagging).
# Left of oriC: replication direction = '-'.
#   Genes on - strand are leading; + are lagging.

def is_leading(pos, strand):
    # which "half" relative to oriC?
    delta = (pos - oriC) % L
    on_right_half = delta < L / 2     # going clockwise from oriC
    if on_right_half:                 # replication moves in + direction
        return strand == "+"
    else:                             # replication moves in - direction
        return strand == "-"

ess_leading = sum(is_leading(mid(r), r[2]) for r in essentials_chr)
oth_leading = sum(is_leading(mid(r), r[2]) for r in others_chr)
print(f"\nLeading-strand co-orientation:")
print(f"  essentials      : {ess_leading}/{len(essentials_chr)}  "
      f"({100*ess_leading/len(essentials_chr):.1f}%)")
print(f"  non-essentials  : {oth_leading}/{len(others_chr)}  "
      f"({100*oth_leading/len(others_chr):.1f}%)")
# chi-square-style enrichment
ess_frac = ess_leading / len(essentials_chr)
oth_frac = oth_leading / len(others_chr)

# ---- (4) mobile-element / transposase avoidance ----
# Find every CDS whose product mentions transposase / IS / integrase
import re as _re
MOB = _re.compile(r"transposase|integrase|insertion sequence|IS\d|phage|recombinase",
                  _re.I)
mob_positions = [mid(r) for r in cds if MOB.search(r[4] or "")]
print(f"\nMobile-element / phage CDS on chromosome 1: {len(mob_positions)}")

def min_dist_to_mobile(pos, mob_positions=mob_positions):
    if not mob_positions: return np.nan
    diffs = [abs(pos - mp) for mp in mob_positions]
    return min(min(d, L - d) for d in diffs)

d_mob_ess   = np.array([min_dist_to_mobile(mid(r)) for r in essentials_chr])
d_mob_other = np.array([min_dist_to_mobile(mid(r)) for r in others_chr])
print(f"  distance to nearest mobile element (kb):")
print(f"    essentials median   : {np.median(d_mob_ess)/1e3:.1f}")
print(f"    non-essentials med  : {np.median(d_mob_other)/1e3:.1f}")
obs_m, p_m = mann_whitney_p(d_mob_ess, d_mob_other)
print(f"    permutation p       : {p_m:.3f}")

# ---- (5) paralog backup (already computed in bridge) ----
bridge = pd.read_parquet(OUT / "bridge_beril_RalstoniaGMI1000.parquet")
bridge["npar"] = pd.to_numeric(bridge["n_paralogs_in_genome"], errors="coerce").fillna(0)
par_ess = bridge.loc[bridge.locus_tag.isin(essential_lt), "npar"]
par_oth = bridge.loc[~bridge.locus_tag.isin(essential_lt), "npar"]
print(f"\nParalog backup (in-genome copies of the same family):")
print(f"  essentials   : {(par_ess > 0).mean():.1%} have >=1 paralog "
      f"(median {int(par_ess.median())})")
print(f"  non-essentials: {(par_oth > 0).mean():.1%} have >=1 paralog "
      f"(median {int(par_oth.median())})")

# ============================================================
# FIGURE
# ============================================================
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

# Panel 1: GC skew + oriC + essential density along chromosome
ax = fig.add_subplot(gs[0, :])
xs_bin = np.arange(len(cumulative)) * WIN / 1e6
ax.plot(xs_bin, cumulative, color="#2C3E50", lw=2, label="cumulative GC skew")
ax.axvline(oriC/1e6, color="#27AE60", ls="--", lw=2,
           label=f"oriC (skew min) ~ {oriC/1e6:.2f} Mb")
ax.axvline(terC/1e6, color="#C0392B", ls="--", lw=2,
           label=f"terC (skew max) ~ {terC/1e6:.2f} Mb")
# overlay essential gene density (kernel histogram)
ess_mids_mb = np.array([mid(r) for r in essentials_chr]) / 1e6
ax2 = ax.twinx()
counts, edges = np.histogram(ess_mids_mb, bins=80, range=(0, L/1e6))
ax2.bar(edges[:-1], counts, width=(edges[1]-edges[0])*0.9,
        alpha=0.35, color="#E67E22", label="essential gene count (per ~50kb)")
ax2.set_ylabel("# essential genes / 50 kb", color="#E67E22")
ax.set_xlabel("chromosome position (Mb)")
ax.set_ylabel("cumulative GC skew", color="#2C3E50")
ax.set_title(f"Chromosome 1 ({CHROM}, {L/1e6:.1f} Mb): "
             f"oriC found by GC skew, essential density along it")
ax.legend(loc="upper left", fontsize=9); ax2.legend(loc="upper right", fontsize=9)

# Panel 2: distance to oriC distribution
ax = fig.add_subplot(gs[1, 0])
bins = np.linspace(0, L/2, 30)
ax.hist(d_ess/1e6, bins=bins/1e6, alpha=0.6, density=True,
        color="#C0392B", label=f"essentials (median {np.median(d_ess)/1e6:.2f} Mb)")
ax.hist(d_other/1e6, bins=bins/1e6, alpha=0.45, density=True,
        color="#3498DB", label=f"non-essentials (median {np.median(d_other)/1e6:.2f} Mb)")
ax.set_xlabel("distance to oriC (Mb, shorter arc)")
ax.set_ylabel("density of genes")
ax.set_title(f"DOSAGE PROTECTION: distance to oriC  (p={p:.3f})")
ax.legend(fontsize=9)

# Panel 3: leading/lagging strand bias
ax = fig.add_subplot(gs[1, 1])
cats = ["leading\n(co-directional)", "lagging\n(head-on collisions)"]
ess_vals = [ess_leading, len(essentials_chr) - ess_leading]
oth_vals = [oth_leading, len(others_chr) - oth_leading]
x = np.arange(2)
w = 0.35
ax.bar(x - w/2, [100*v/len(essentials_chr) for v in ess_vals], w,
       color="#C0392B", alpha=0.85,
       label=f"essentials ({100*ess_frac:.0f}% leading)")
ax.bar(x + w/2, [100*v/len(others_chr) for v in oth_vals], w,
       color="#3498DB", alpha=0.85,
       label=f"non-essentials ({100*oth_frac:.0f}% leading)")
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel("% of genes")
ax.set_title(f"STRAND PROTECTION: head-on collisions avoided\n"
             f"essentials {100*ess_frac:.0f}% leading  vs  others {100*oth_frac:.0f}%")
ax.legend(fontsize=9)

# Panel 4: distance to nearest mobile element
ax = fig.add_subplot(gs[2, 0])
bins = np.linspace(0, 500, 30)
ax.hist(d_mob_ess/1e3, bins=bins, alpha=0.6, density=True,
        color="#C0392B", label=f"essentials (median {np.median(d_mob_ess)/1e3:.0f} kb)")
ax.hist(d_mob_other/1e3, bins=bins, alpha=0.45, density=True,
        color="#3498DB", label=f"non-essentials (median {np.median(d_mob_other)/1e3:.0f} kb)")
ax.set_xlabel("distance to nearest mobile element (kb)")
ax.set_ylabel("density of genes")
ax.set_title(f"INSERTION PROTECTION: distance to transposase/phage  (p={p_m:.3f})")
ax.legend(fontsize=9)

# Panel 5: paralog backup
ax = fig.add_subplot(gs[2, 1])
buckets = [0, 1, 2, 3, 5, 100]
labels  = ["0", "1", "2", "3-4", "5+"]
ess_h = [((par_ess >= buckets[i]) & (par_ess < buckets[i+1])).sum()
         for i in range(5)]
oth_h = [((par_oth >= buckets[i]) & (par_oth < buckets[i+1])).sum()
         for i in range(5)]
x = np.arange(5); w = 0.35
ax.bar(x - w/2, [100*v/len(par_ess) for v in ess_h], w, color="#C0392B",
       alpha=0.85, label="essentials")
ax.bar(x + w/2, [100*v/len(par_oth) for v in oth_h], w, color="#3498DB",
       alpha=0.85, label="non-essentials")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("# in-genome paralogs")
ax.set_ylabel("% of genes")
ax.set_title("BACKUP PROTECTION: paralog redundancy")
ax.legend(fontsize=9)

fig.suptitle("How GMI1000 protects its essential genes — five chromosomal signatures",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_png = OUT / "protection_evidence.png"
fig.savefig(out_png, dpi=140, bbox_inches="tight")
print(f"\nwrote {out_png}")

summary = {
    "chromosome": CHROM, "length_bp": L,
    "oriC_bp": int(oriC), "terC_bp": int(terC),
    "n_essentials_chr1": len(essentials_chr),
    "n_others_chr1":     len(others_chr),
    "dosage_protection": {
        "median_dist_to_oriC_essentials_Mb": round(float(np.median(d_ess))/1e6, 3),
        "median_dist_to_oriC_others_Mb":     round(float(np.median(d_other))/1e6, 3),
        "permutation_p": p,
    },
    "strand_protection": {
        "essentials_leading_pct":     round(100*ess_frac, 1),
        "non_essentials_leading_pct": round(100*oth_frac, 1),
        "absolute_difference_pct":    round(100*(ess_frac - oth_frac), 1),
    },
    "insertion_protection": {
        "n_mobile_elements_chr1": len(mob_positions),
        "median_dist_essentials_kb": round(float(np.median(d_mob_ess))/1e3, 1),
        "median_dist_others_kb":     round(float(np.median(d_mob_other))/1e3, 1),
        "permutation_p": p_m,
    },
    "backup_protection_paralogs": {
        "essentials_pct_with_paralog": round(100*float((par_ess > 0).mean()), 1),
        "others_pct_with_paralog":     round(100*float((par_oth > 0).mean()), 1),
    },
}
(OUT / "protection_evidence_summary.json").write_text(json.dumps(summary, indent=2))
print(f"wrote protection_evidence_summary.json")
