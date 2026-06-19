"""Genome spectrometer for NON-ESSENTIAL genes -- mirror of genome_spectrometer.py.

Same sweep axis (Ralstonia GMI1000 chr1, 5'->3') and oriC/terC markers, but
every channel reads DISPENSABILITY instead of protection. The four protection
channels from the essential spectrometer are INVERTED into 'exposure' channels:

  reference physical (unchanged from essential spectrometer):
    GC%               base composition
    GC skew           (G-C)/(G+C), reveals oriC/terC
    cumulative skew   canonical replication-origin finder

  the headline readout:
    non-essential density   measured non-essentials per 50 kb

  inverted-protection / positive-dispensability channels:
    mobile-element density    HGT/accessory hotspots
    paralog load              redundancy buffering
    low-conservation density  fraction with cons<0.3 per 50 kb
    sister-absent fraction    avg sister-strain orthologs missing -- the only
                              channel that is positive within-species
                              dispensability evidence, not a mirror
    head-on long density      lagging-strand long low-cons genes per 50 kb
                              (the configuration the cell avoids for essentials)

  atlas overlay:
    rogue+conditional density   P_rogue / P_conditional quarantined genes
                                per 50 kb (the niche/conditional candidates)

Bottom panel: bar chart of per-channel Pearson correlation with the non-
essential density readout -- which channels actually track dispensability
spatially? This is the analytical mirror of the FFT panel in the essential
spectrometer.

Outputs:
  outputs/orphan/genome_spectrometer_nonessential_GMI1000.png
  outputs/orphan/genome_spectrometer_nonessential_GMI1000_summary.json
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
SISTERS = ["beril_RalstoniaBSBF1503", "beril_RalstoniaPSI07", "beril_RalstoniaUW163"]
WIN = 50_000; STEP = 10_000
MOB = re.compile(r"transposase|integrase|insertion sequence|IS\d|phage|"
                 r"recombinase|mobile element", re.I)

# labels + orthology
ess = {r["locus_tag"]: int(r["essential"])
       for r in csv.DictReader(open(DATA / "labels" / "labels.csv"))
       if r["organism"] == ORG}
orth = {r["locus_tag"]: r
        for r in csv.DictReader(open(DATA / "labels" / "orthology_features.csv"))
        if r["organism"] == ORG}
og_orgs = defaultdict(set)
for r in csv.DictReader(open(DATA / "labels" / "orthology_features.csv")):
    if r["og_id"]:
        og_orgs[r["og_id"]].add(r["organism"])

# atlas tiers (from v3 atlas)
atlas = pd.read_csv(OUT / "atlas_multi" / "beril_RalstoniaGMI1000.csv")
atlas_tier = dict(zip(atlas.locus_tag, atlas.tier))

# sequence
contigs = {}
cur = None; buf = []
for line in (DATA / "genome_cache" / ACC / "genome.fna").read_text().splitlines():
    if line.startswith(">"):
        if cur: contigs[cur] = "".join(buf).upper()
        cur = line[1:].split()[0]; buf = []
    else: buf.append(line.strip())
if cur: contigs[cur] = "".join(buf).upper()
ctg = max(contigs, key=lambda c: len(contigs[c]))
seq = contigs[ctg]; L = len(seq)
print(f"chromosome {ctg}: {L:,} bp")

# parse CDS on the longest contig
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

# oriC/terC via cumulative skew
g_arr = np.frombuffer(seq.replace("A","\0").replace("T","\0")
                         .replace("G","\1").replace("C","\0").encode(), dtype=np.int8)
c_arr = np.frombuffer(seq.replace("A","\0").replace("T","\0")
                         .replace("G","\0").replace("C","\1").encode(), dtype=np.int8)
gc_cum = np.cumsum(g_arr + c_arr); sk_cum = np.cumsum(g_arr - c_arr)
def gc_pct(a,b): return (gc_cum[b-1]-(gc_cum[a-1] if a>0 else 0))/max(1,b-a)
def gc_skew(a,b):
    g=(gc_cum[b-1]-(gc_cum[a-1] if a>0 else 0))
    s=(sk_cum[b-1]-(sk_cum[a-1] if a>0 else 0))
    return s/max(1,g)

centers = np.arange(WIN//2, L - WIN//2, STEP)
def channel(func):
    return np.array([func(max(0,c-WIN//2), min(L,c+WIN//2)) for c in centers])

# per-CDS arrays
cds_mid    = np.array([(s+e)//2 for s,e,*_ in cds])
cds_strand = np.array([1 if st=="+" else -1 for *_,st,_,_ in [(s,e,st,lt,p) for s,e,st,lt,p in cds]])
cds_len    = np.array([(e-s+1)//3 for s,e,*_ in cds])
cds_lt     = [lt for *_,lt,_ in [(s,e,st,lt,p) for s,e,st,lt,p in cds]]
cds_ess    = np.array([ess.get(lt,-1) for lt in cds_lt])
cds_cons   = np.array([float(orth[lt]["family_frac_essential_leakfree"])
                       if lt in orth and orth[lt]["family_frac_essential_leakfree"] not in ("","nan")
                       else np.nan for lt in cds_lt])
cds_par    = np.array([float(orth[lt]["n_paralogs_in_genome"])
                       if lt in orth and orth[lt]["n_paralogs_in_genome"]!=""
                       else 0.0 for lt in cds_lt])
cds_mob    = np.array([1 if MOB.search(p or "") else 0
                       for s,e,st,lt,p in cds])
def sister_count(lt):
    og = orth.get(lt,{}).get("og_id","")
    if not og: return 0
    present = og_orgs.get(og,set())
    return sum(1 for s in SISTERS if s in present)
cds_nsis   = np.array([sister_count(lt) for lt in cds_lt])
cds_tier   = np.array([atlas_tier.get(lt,"") for lt in cds_lt])

# oriC/terC for plotting
def oriC_terC(seq, win=10000):
    sk=[]
    for i in range(0,len(seq),win):
        w=seq[i:i+win]; g,c=w.count("G"),w.count("C")
        sk.append((g-c)/max(1,g+c))
    cum=np.cumsum(sk); return int(np.argmin(cum))*win, int(np.argmax(cum))*win
oriC, terC = oriC_terC(seq)
print(f"oriC ~ {oriC/1e6:.2f} Mb, terC ~ {terC/1e6:.2f} Mb")

def winmask(a,b): return (cds_mid>=a) & (cds_mid<b)
def noness_dens(a,b):
    m = winmask(a,b) & (cds_ess==0)
    return int(m.sum())
def ess_dens(a,b):
    m = winmask(a,b) & (cds_ess==1)
    return int(m.sum())
def mob_dens(a,b):
    m = winmask(a,b) & (cds_mob==1)
    return int(m.sum())
def par_mean(a,b):
    m = winmask(a,b)
    return float(cds_par[m].mean()) if m.any() else 0.0
def lowcons_dens(a,b):
    m = winmask(a,b) & (~np.isnan(cds_cons)) & (cds_cons<0.3)
    return int(m.sum())
def sister_absent_frac(a,b):
    m = winmask(a,b)
    if not m.any(): return 0.0
    # fraction of (3 sisters) missing the gene's OG, averaged over genes in window
    miss = (3 - cds_nsis[m]) / 3.0
    return float(miss.mean())
def head_on_long(a,b):
    # lagging + long (top quartile) + low cons
    long_cut = np.quantile(cds_len, 0.75)
    m = winmask(a,b)
    if not m.any(): return 0
    sub_strand = cds_strand[m]; sub_mid = cds_mid[m]; sub_len = cds_len[m]
    sub_cons = cds_cons[m]
    # leading: ((mid - oriC) % L < L/2) == (strand == "+")
    delta = (sub_mid - oriC) % L
    leading = (sub_strand==1) == (delta < L/2)
    bad = (~leading) & (sub_len >= long_cut) & (np.where(np.isnan(sub_cons),0,sub_cons) < 0.3)
    return int(bad.sum())
def quarantine_dens(a,b):
    m = winmask(a,b) & ((cds_tier == "ROGUE_SUSPECT") | (cds_tier == "CONDITIONAL_SUSPECT"))
    return int(m.sum())

gc      = channel(gc_pct)
skew    = channel(gc_skew)
cumsk   = np.cumsum(skew - skew.mean())
e_den   = channel(ess_dens)
ne_den  = channel(noness_dens)
mob_d   = channel(mob_dens)
par     = channel(par_mean)
lc_den  = channel(lowcons_dens)
sa_frac = channel(sister_absent_frac)
hl_den  = channel(head_on_long)
quar    = channel(quarantine_dens)

# correlations of every predictor channel with the non-essential density readout
def pcorr(x,y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0: return 0.0
    return float(np.corrcoef(x,y)[0,1])
preds = {
    "GC%":               gc,
    "GC skew |abs|":     np.abs(skew),
    "essential density": e_den,
    "mobile-element":    mob_d,
    "paralog load":      par,
    "low-conservation":  lc_den,
    "sister-absent":     sa_frac,
    "head-on long":      hl_den,
    "quarantine\n(rogue+cond)": quar,
}
corrs = {name: pcorr(v, ne_den) for name,v in preds.items()}
print("\ncorrelations with non-essential density (50 kb windows):")
for k,v in sorted(corrs.items(), key=lambda x:-abs(x[1])):
    print(f"  {k:<22} r = {v:+.3f}")

# ---- figure ----
chans = [
    ("GC%",                  gc*100,        "#7F8C8D", "fraction"),
    ("GC skew",              skew,          "#34495E", "(G-C)/(G+C)"),
    ("cumulative skew",      cumsk,         "#9B59B6", "cum"),
    ("non-essential density", ne_den,       "#7F8C8D", "per 50 kb"),
    ("essential density",    e_den,         "#C0392B", "per 50 kb (ref)"),
    ("mobile-element",       mob_d,         "#D35400", "per 50 kb"),
    ("paralog load",         par,           "#E67E22", "mean copies"),
    ("low-conservation",     lc_den,        "#16A085", "cons<0.3 per 50 kb"),
    ("sister-absent",        sa_frac,       "#2980B9", "avg miss frac"),
    ("head-on long",         hl_den,        "#8E44AD", "per 50 kb"),
    ("quarantine\n(atlas)",  quar,          "#F39C12", "per 50 kb"),
]

x_mb = centers / 1e6
fig = plt.figure(figsize=(15, 15))
gs = fig.add_gridspec(len(chans)+2, 1, hspace=0.05,
                      height_ratios=[0.55]*len(chans) + [0.4, 1.3])

axes = []
for i,(name,y,col,unit) in enumerate(chans):
    ax = fig.add_subplot(gs[i,0])
    ax.plot(x_mb, y, color=col, lw=1.1)
    ax.fill_between(x_mb, y.min(), y, color=col, alpha=0.18)
    ax.axvline(oriC/1e6, color="green", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(terC/1e6, color="red",   lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel(f"{name}\n[{unit}]", fontsize=8, rotation=0,
                  ha="right", va="center", labelpad=8)
    ax.tick_params(labelsize=7); ax.set_xlim(0, L/1e6)
    if i < len(chans)-1: ax.set_xticklabels([])
    ax.grid(alpha=0.25, axis="x"); axes.append(ax)
axes[-1].set_xlabel("position along chromosome (Mb)", fontsize=10)
axes[0].text(oriC/1e6, axes[0].get_ylim()[1], " oriC",
             color="green", fontsize=9, va="top")
axes[0].text(terC/1e6, axes[0].get_ylim()[1], " terC",
             color="red", fontsize=9, va="top")

# bottom: correlation panel
ax_corr = fig.add_subplot(gs[-1,0])
items = sorted(corrs.items(), key=lambda x:-abs(x[1]))
names = [k for k,_ in items]; vals = [v for _,v in items]
colors_c = ["#27AE60" if v>=0 else "#C0392B" for v in vals]
yp = np.arange(len(names))[::-1]
ax_corr.barh(yp, vals, color=colors_c, alpha=0.85)
for i,v in enumerate(vals):
    y = yp[i]
    ax_corr.text(v + (0.01 if v>=0 else -0.01), y,
                 f"{v:+.2f}", va="center",
                 ha="left" if v>=0 else "right", fontsize=9)
ax_corr.set_yticks(yp); ax_corr.set_yticklabels(names, fontsize=9)
ax_corr.axvline(0, color="k", lw=0.6)
ax_corr.set_xlim(-1, 1)
ax_corr.set_xlabel("Pearson r with non-essential density (50 kb windows)")
ax_corr.set_title("which channels actually track dispensability spatially?",
                  fontsize=10)
ax_corr.grid(alpha=0.3, axis="x")

fig.suptitle(f"NON-ESSENTIAL spectrometer -- {ORG.replace('beril_','')} chr1 ({L/1e6:.2f} Mb)\n"
             f"every protection channel inverted into a dispensability readout",
             fontsize=12, fontweight="bold", y=0.995)
fig.savefig(OUT / "genome_spectrometer_nonessential_GMI1000.png", dpi=140,
            bbox_inches="tight")
print(f"\nwrote genome_spectrometer_nonessential_GMI1000.png")

(OUT / "genome_spectrometer_nonessential_GMI1000_summary.json").write_text(json.dumps({
    "organism": ORG, "contig": ctg, "length_bp": int(L),
    "n_cds_on_contig": len(cds),
    "oriC_Mb": round(oriC/1e6,3), "terC_Mb": round(terC/1e6,3),
    "noness_total": int((cds_ess==0).sum()),
    "ess_total": int((cds_ess==1).sum()),
    "correlations_with_noness_density": {k: round(v,3) for k,v in corrs.items()},
    "windows": int(len(centers)),
}, indent=2))
