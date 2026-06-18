"""Refined regulatory scan for GMI1000 essentials:
   (A) operon ENDS  -> rho-independent terminator T-tract should now spike
   (B) operon STARTS -> scan for non-sigma70 promoters (sigma32, sigma54, sigmaE)

OPERON CALL: walk the GFF in chromosome order; a gene is an OPERON END iff the
NEXT CDS on the SAME strand is >50 bp away OR the next CDS is on the opposite
strand OR there's no next CDS. An OPERON START is defined symmetrically using
the previous CDS.

SIGMA CONSENSI (Ralstonia is a betaproteobacterium; E. coli consensi apply):
  sigma32 (heat shock):    -35 'CTTGAA' / -10 'CCCCAT' / spacing 13-15 bp
  sigma54 (nitrogen/RpoN): -24 'TGGCAC' / -12 'TTGCA' / spacing exactly 5 bp
  sigmaE  (envelope stress): -35 'GAACTT' / -10 'TCTGAA' / spacing 16-17 bp

Outputs:
  outputs/orphan/regulatory_followups.png
  outputs/orphan/regulatory_followups_summary.json
"""
import re, json
from pathlib import Path
from collections import Counter, defaultdict
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

# -- inputs --
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
genome_T = all_seq.count("T") / len(all_seq)

# -- parse all CDS, in chromosome order per contig --
all_cds = []
for line in (DATA / "genome_cache" / ACC / "genomic.gff").read_text().splitlines():
    if "\tCDS\t" not in line: continue
    c = line.split("\t")
    m = re.search(r"locus_tag=([^;\n]+)", c[8])
    if not m: continue
    all_cds.append((c[0], int(c[3]), int(c[4]), c[6], m.group(1)))
all_cds.sort()      # (contig, start)
print(f"total CDS: {len(all_cds):,}; essentials: {len(essential_lt):,}")

# -- compute operon-start / operon-end flags --
GAP = 50
is_start, is_end = {}, {}
by_contig = defaultdict(list)
for r in all_cds: by_contig[r[0]].append(r)
for contig, lst in by_contig.items():
    for i, (ct, s, e, strand, lt) in enumerate(lst):
        # OPERON START: previous CDS doesn't form an operon with this one
        if i == 0: is_start[lt] = True
        else:
            ps, pe, pstr = lst[i-1][1], lst[i-1][2], lst[i-1][3]
            same = (strand == pstr)
            gap  = s - pe - 1
            is_start[lt] = (not same) or gap > GAP
        # OPERON END: next CDS doesn't form an operon with this one
        if i == len(lst) - 1: is_end[lt] = True
        else:
            ns, ne, nstr = lst[i+1][1], lst[i+1][2], lst[i+1][3]
            same = (strand == nstr)
            gap  = ns - e - 1
            is_end[lt] = (not same) or gap > GAP

ess_starts = [lt for lt in essential_lt if is_start.get(lt)]
ess_ends   = [lt for lt in essential_lt if is_end.get(lt)]
print(f"essential operon STARTS: {len(ess_starts):,}")
print(f"essential operon ENDS  : {len(ess_ends):,}")

# -- helpers --
TBL_letters = "ACGT"
def best_match(text, consensus):
    k = len(consensus); best = (-1, k+1, "")
    if len(text) < k: return best
    for i in range(len(text)-k+1):
        win = text[i:i+k]
        mm = sum(a != b for a, b in zip(win, consensus))
        if mm < best[1]:
            best = (i, mm, win)
            if mm == 0: break
    return best

# ============================================================
# (A) Terminator scan -- T-tract downstream of stop, ONLY for operon-end ess
# ============================================================
cds_by_lt = {r[4]: r for r in all_cds}
down_all, down_end = [], []
for lt in essential_lt:
    if lt not in cds_by_lt: continue
    ct, s, e, strand, _ = cds_by_lt[lt]
    contig = contigs[ct]
    if strand == "+":
        dn = contig[e : e+60]
    else:
        dn = rc(contig[max(0, s-1-60) : s-1])
    if len(dn) != 60: continue
    down_all.append(dn)
    if is_end.get(lt): down_end.append(dn)
print(f"\ndownstream-60 windows: all essentials {len(down_all):,}, "
      f"operon-end essentials {len(down_end):,}")

def t_profile(seqs):
    arr = np.zeros(50)
    for d in seqs:
        for i in range(50):
            if d[i] == "T": arr[i] += 1
    arr /= len(seqs)
    sliding6 = np.array([arr[i:i+6].mean() if i+6 <= 50 else np.nan
                         for i in range(50)])
    return arr, sliding6

t_all, slid_all = t_profile(down_all)
t_end, slid_end = t_profile(down_end)
peak_end = int(np.nanargmax(slid_end))
peak_all = int(np.nanargmax(slid_all))
print(f"operon-end peak T6:  +{peak_end+1}..+{peak_end+6}: {slid_end[peak_end]:.2f}")
print(f"all-essentials peak: +{peak_all+1}..+{peak_all+6}: {slid_all[peak_all]:.2f}")
print(f"genome baseline T  : {genome_T:.2f}")

# ============================================================
# (B) Alternative sigma factors at operon starts (upstream 80 bp)
# ============================================================
UP = 80
def upstream_window(lt):
    ct, s, e, strand, _ = cds_by_lt[lt]
    contig = contigs[ct]
    if strand == "+":
        if s-1-UP < 0: return None
        return contig[s-1-UP : s-1]
    else:
        if e+UP > len(contig): return None
        return rc(contig[e : e+UP])

# spacing definitions: distance from end of -35/-24 box to start of -10/-12 box
SIGMAS = [
    ("sigma70",  "TTGACA", "TATAAT",  (15, 19)),
    ("sigma32",  "CTTGAA", "CCCCAT",  (12, 16)),
    ("sigma54",  "TGGCAC", "TTGCA",   (5,  5)),     # exact 5 bp
    ("sigmaE",   "GAACTT", "TCTGAA",  (15, 18)),
]
def scan_sigma(window, box1, box2, spacing_range, tol1=2, tol2=2):
    """Return best-match position (relative to start codon)/spacing/mismatches.
    Anchored: scan box2 across whole upstream; for each box2 hit, look for box1
    UPSTREAM by exactly spacing_range[0..1] bp."""
    best = None
    L1, L2 = len(box1), len(box2)
    for i2 in range(len(window) - L2 + 1):
        w2 = window[i2:i2+L2]
        mm2 = sum(a != b for a, b in zip(w2, box2))
        if mm2 > tol2: continue
        # box1 must end exactly (spacing+L1) bp before i2 start
        for sp in range(spacing_range[0], spacing_range[1]+1):
            i1 = i2 - sp - L1
            if i1 < 0 or i1 + L1 > len(window): continue
            w1 = window[i1:i1+L1]
            mm1 = sum(a != b for a, b in zip(w1, box1))
            if mm1 > tol1: continue
            score = mm1 + mm2
            if best is None or score < best[0]:
                best = (score, i1, i2, sp, mm1, mm2, w1, w2)
    return best

results = {name: 0 for name, *_ in SIGMAS}
results_per_gene = {}
for lt in ess_starts:
    w = upstream_window(lt)
    if not w or len(w) < UP: continue
    hits = {}
    for name, b1, b2, sp_rng in SIGMAS:
        h = scan_sigma(w, b1, b2, sp_rng)
        hits[name] = h
        if h is not None: results[name] += 1
    results_per_gene[lt] = hits

n_start_with_window = sum(1 for lt in ess_starts
                         if upstream_window(lt) and len(upstream_window(lt))==UP)
print(f"\nessential operon STARTS scanned: {n_start_with_window:,}")
print("Sigma factor matches (allowing up to 2 mismatches per box, correct spacing):")
for name, *_ in SIGMAS:
    n = results[name]
    print(f"  {name:<8}: {n:>4} ({100*n/max(1,n_start_with_window):>4.1f}%)")

# Of operon-start essentials with NO sigma70 hit -- how many have sigma32/54/E?
n_no_sig70 = sum(1 for lt, hits in results_per_gene.items()
                 if hits.get("sigma70") is None)
n_alt_only = sum(1 for lt, hits in results_per_gene.items()
                 if hits.get("sigma70") is None
                 and any(hits.get(s) is not None for s in
                         ("sigma32","sigma54","sigmaE")))
print(f"\noperon-start essentials lacking sigma70 hit: {n_no_sig70}")
print(f"  of those, {n_alt_only} have an alternative sigma hit "
      f"({100*n_alt_only/max(1,n_no_sig70):.0f}%)")

# ============================================================
# FIGURE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: terminator T-tract (all vs operon-ends)
ax = axes[0]
xs = np.arange(1, 51)
ax.plot(xs[:len(slid_all)], 100*slid_all, "-",  color="#888",  lw=2,
        label=f"all essentials (n={len(down_all):,}), peak {100*slid_all[peak_all]:.0f}%")
ax.plot(xs[:len(slid_end)], 100*slid_end, "-",  color="#C0392B", lw=2.5,
        label=f"operon-end essentials (n={len(down_end):,}), peak {100*slid_end[peak_end]:.0f}%")
ax.axhline(100*genome_T, color="black", ls="--", lw=0.8,
           label=f"genome baseline ({100*genome_T:.0f}%)")
ax.axvspan(peak_end+1, peak_end+6, color="#E67E22", alpha=0.15)
ax.set_xlabel("nt downstream of stop codon")
ax.set_ylabel("% T (6-nt sliding) = U on mRNA")
ax.set_title("Rho-independent terminator: restricted to operon ENDS")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)

# Panel B: sigma factor hit rates at operon starts
ax = axes[1]
names = [n for n, *_ in SIGMAS]
counts = [results[n] for n in names]
total  = n_start_with_window
pcts   = [100*c/total for c in counts]
colors = ["#27AE60", "#E74C3C", "#3498DB", "#8E44AD"]
ax.bar(names, pcts, color=colors, alpha=0.9)
for i, (c, p) in enumerate(zip(counts, pcts)):
    ax.text(i, p+1, f"{c}\n({p:.1f}%)", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
ax.set_ylabel(f"% of {total:,} operon-start essentials with a hit")
ax.set_title("Promoter hits at operon STARTS  (≤2 mm per box, correct spacing)")
ax.set_ylim(0, max(pcts)*1.25)
ax.set_xticklabels([
    "σ70\n(TTGACA / TATAAT, 15–19 bp)",
    "σ32\n(CTTGAA / CCCCAT, 12–16 bp)",
    "σ54\n(TGGCAC / TTGCA, exactly 5 bp)",
    "σE\n(GAACTT / TCTGAA, 15–18 bp)",
], fontsize=8)

fig.suptitle("Refined regulatory scan: GMI1000 essential genes",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_png = OUT / "regulatory_followups.png"
fig.savefig(out_png, dpi=140, bbox_inches="tight")
print(f"\nwrote {out_png}")

summary = {
    "essential_total": len(essential_lt),
    "essential_operon_starts": int(n_start_with_window),
    "essential_operon_ends":   len(down_end),
    "terminator_T_tract": {
        "peak_position_operon_end_genes": peak_end + 1,
        "peak_T_fraction_operon_end_6bp": round(float(slid_end[peak_end]), 3),
        "peak_T_fraction_all_essentials_6bp": round(float(slid_all[peak_all]), 3),
        "genome_T_baseline": round(genome_T, 3),
    },
    "sigma_hits_at_operon_starts": {
        name: {"count": results[name],
               "pct":   round(100*results[name]/total, 2)}
        for name, *_ in SIGMAS
    },
    "no_sigma70_but_alt": {
        "no_sigma70_hit": int(n_no_sig70),
        "alt_sigma_hit":  int(n_alt_only),
        "alt_rescue_pct": round(100*n_alt_only/max(1,n_no_sig70), 2),
    },
}
(OUT / "regulatory_followups_summary.json").write_text(json.dumps(summary, indent=2))
print("wrote regulatory_followups_summary.json")
