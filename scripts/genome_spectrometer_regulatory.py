"""Genome spectrometer -- REGULATORY / sequence-level channels.

Companion to genome_spectrometer.py. Same sweep axis (Ralstonia GMI1000
chromosome 1, 5'->3'), same oriC/terC markers, but now the detectors read
fine-scale promoter & coding signals, each strand-aware, aggregated per CDS
then smoothed over a sliding 50-kb / 10-kb window:

  start ATG frac        fraction of starts that are ATG  (vs GTG/TTG)
  alt-start frac        fraction GTG+TTG (high-GC alternative starts)
  stop TGA frac         fraction of stops that are TGA   (high-GC bias)
  -35 box hit           frac of genes with a strong TTGACA -35 (<=1 mismatch)
  -10 box hit           frac with a strong TATAAT Pribnow -10 (<=1 mismatch)
  spacer canonical      frac with 15-19 bp between -35 and -10
  RBS / Shine-Dalgarno  frac with a strong GGAGG ribosome-binding site
  motif TGCTG density    top over-represented 5-mer (from nt_motifs), per kb

Consensus values reused from outputs/orphan/*_summary.json so the channels are
consistent with the earlier promoter analysis (ATG 92%, TGA 66%, etc).

Outputs:
  outputs/orphan/genome_spectrometer_regulatory_GMI1000.png
  outputs/orphan/genome_spectrometer_regulatory_GMI1000_summary.json
"""
import re, json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")
ORG = "beril_RalstoniaGMI1000"; ACC = "GCF_000009125.1"
WIN = 50_000; STEP = 10_000
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
def rc(s): return s.translate(COMP)[::-1]

M35 = "TTGACA"; M10 = "TATAAT"; SD = "GGAGG"; MOTIF = "TGCTG"
def mm(a, b):  # mismatches between equal-length strings
    return sum(1 for x, y in zip(a, b) if x != y)
def best_hit(region, consensus):
    """min mismatch of consensus sliding over region; return (mm, offset)."""
    k = len(consensus); best = (k+1, -1)
    for i in range(len(region) - k + 1):
        d = mm(region[i:i+k], consensus)
        if d < best[0]: best = (d, i)
    return best

# sequence -> longest contig
contigs = {}; cur=None; buf=[]
for line in (DATA/"genome_cache"/ACC/"genome.fna").read_text().splitlines():
    if line.startswith(">"):
        if cur: contigs[cur]="".join(buf).upper()
        cur=line[1:].split()[0]; buf=[]
    else: buf.append(line.strip())
if cur: contigs[cur]="".join(buf).upper()
ctg = max(contigs, key=lambda c: len(contigs[c])); seq = contigs[ctg]; L=len(seq)
print(f"chromosome {ctg}: {L:,} bp")

# CDS
cds=[]
for line in (DATA/"genome_cache"/ACC/"genomic.gff").read_text().splitlines():
    if "\tCDS\t" not in line: continue
    c=line.split("\t")
    if c[0]!=ctg: continue
    m=re.search(r"locus_tag=([^;\n]+)",c[8])
    if not m: continue
    cds.append((int(c[3]),int(c[4]),c[6],m.group(1)))
cds.sort()
print(f"CDS: {len(cds)}")

# per-CDS regulatory readout (strand-aware)
mids=[]; atg=[]; altstart=[]; tga=[]; h35=[]; h10=[]; spacer=[]; rbs=[]
for s,e,st,lt in cds:
    mid=(s+e)//2; mids.append(mid)
    if st=="+":
        start_cod=seq[s-1:s+2]; stop_cod=seq[e-3:e]
        up=seq[max(0,s-1-50):s-1]                # 50 bp upstream of start
    else:
        start_cod=rc(seq[e-3:e]); stop_cod=rc(seq[s-1:s+2])
        up=rc(seq[e:min(L,e+50)])
    # up is oriented 5'->3' ending just before the start codon
    atg.append(int(start_cod=="ATG"))
    altstart.append(int(start_cod in ("GTG","TTG")))
    tga.append(int(stop_cod=="TGA"))
    if len(up) >= 40:
        reg10 = up[-18:-4]            # Pribnow region
        reg35 = up[-40:-24]           # -35 region
        regsd = up[-20:-5]            # ribosome-binding region
        d10,o10 = best_hit(reg10, M10)
        d35,o35 = best_hit(reg35, M35)
        dsd,_   = best_hit(regsd, SD)
        h10.append(int(d10<=1)); h35.append(int(d35<=1)); rbs.append(int(dsd<=1))
        # spacer: genomic distance between -35 hit end and -10 hit start
        pos35 = (len(up)-40)+o35+len(M35)
        pos10 = (len(up)-18)+o10
        gap = pos10 - pos35
        spacer.append(int(15 <= gap <= 19 and d10<=2 and d35<=2))
    else:
        h10.append(0); h35.append(0); rbs.append(0); spacer.append(0)
mids=np.array(mids)
arrs={k:np.array(v) for k,v in dict(atg=atg,altstart=altstart,tga=tga,
      h35=h35,h10=h10,spacer=spacer,rbs=rbs).items()}

# motif density on forward strand per kb (count motif + its rc)
mrc = rc(MOTIF)
def motif_count(a,b):
    sub=seq[a:b]; return sub.count(MOTIF)+sub.count(mrc)

# cumulative skew for oriC/terC (same method as main spectrometer)
g=np.frombuffer(seq.replace("A","\0").replace("T","\0").replace("G","\1").replace("C","\0").encode(),dtype=np.int8)
c=np.frombuffer(seq.replace("A","\0").replace("T","\0").replace("G","\0").replace("C","\1").encode(),dtype=np.int8)
skcum=np.cumsum(g-c)

centers=np.arange(WIN//2, L-WIN//2, STEP)
def winmask(a,b): return (mids>=a)&(mids<b)
def frac(key):
    out=[]
    for ct in centers:
        a,b=max(0,ct-WIN//2),min(L,ct+WIN//2); m=winmask(a,b)
        out.append(float(arrs[key][m].mean()) if m.any() else 0.0)
    return np.array(out)
def motif_density():
    return np.array([motif_count(max(0,ct-WIN//2),min(L,ct+WIN//2))/(WIN/1000)
                     for ct in centers])

ch_atg=frac("atg"); ch_alt=frac("altstart"); ch_tga=frac("tga")
ch_35=frac("h35"); ch_10=frac("h10"); ch_sp=frac("spacer"); ch_rbs=frac("rbs")
ch_motif=motif_density()

# oriC/terC
sk_win=np.array([ (skcum[min(L-1,ct+WIN//2-1)]-skcum[max(0,ct-WIN//2-1)]) for ct in centers ])
oriC=centers[int(np.argmin(skcum[centers]))]; terC=centers[int(np.argmax(skcum[centers]))]
print(f"oriC ~ {oriC/1e6:.2f} Mb, terC ~ {terC/1e6:.2f} Mb")
print(f"genome-wide: ATG {arrs['atg'].mean():.2f}  TGA {arrs['tga'].mean():.2f}  "
      f"-35hit {arrs['h35'].mean():.2f}  -10hit {arrs['h10'].mean():.2f}  "
      f"RBS {arrs['rbs'].mean():.2f}  spacer {arrs['spacer'].mean():.2f}")

# ---- figure ----
chans=[
    ("start ATG frac",   ch_atg,  "#27AE60","fraction"),
    ("alt-start GTG/TTG", ch_alt, "#16A085","fraction"),
    ("stop TGA frac",    ch_tga,  "#C0392B","fraction"),
    ("-35 box hit",      ch_35,   "#2980B9","frac <=1 mm"),
    ("-10 box hit",      ch_10,   "#8E44AD","frac <=1 mm"),
    ("spacer 15-19 bp",  ch_sp,   "#D35400","fraction"),
    ("RBS (Shine-Dalg.)",ch_rbs,  "#E67E22","frac <=1 mm"),
    (f"motif {MOTIF}",   ch_motif,"#2C3E50","per kb"),
]
x=centers/1e6
fig=plt.figure(figsize=(15,12))
gs=fig.add_gridspec(len(chans),1,hspace=0.06)
axes=[]
for i,(name,y,col,unit) in enumerate(chans):
    ax=fig.add_subplot(gs[i,0])
    ax.plot(x,y,color=col,lw=1.1); ax.fill_between(x,y.min(),y,color=col,alpha=0.18)
    ax.axvline(oriC/1e6,color="green",ls="--",lw=0.8,alpha=0.7)
    ax.axvline(terC/1e6,color="red",ls="--",lw=0.8,alpha=0.7)
    ax.set_ylabel(f"{name}\n[{unit}]",fontsize=8,rotation=0,ha="right",va="center",labelpad=8)
    ax.tick_params(labelsize=7); ax.set_xlim(0,L/1e6)
    if i<len(chans)-1: ax.set_xticklabels([])
    ax.grid(alpha=0.25,axis="x"); axes.append(ax)
axes[-1].set_xlabel("position along chromosome (Mb)",fontsize=10)
axes[0].text(oriC/1e6,axes[0].get_ylim()[1]," oriC",color="green",fontsize=9,va="top")
axes[0].text(terC/1e6,axes[0].get_ylim()[1]," terC",color="red",fontsize=9,va="top")

# consensus annotation box
txt=(f"consensus scanned (strand-aware):\n"
     f"  start: ATG {arrs['atg'].mean():.0%} | GTG/TTG {arrs['altstart'].mean():.0%}\n"
     f"  stop:  TGA {arrs['tga'].mean():.0%}\n"
     f"  -35 TTGACA hit {arrs['h35'].mean():.0%} | -10 TATAAT hit {arrs['h10'].mean():.0%}\n"
     f"  spacer 15-19bp {arrs['spacer'].mean():.0%} | RBS GGAGG {arrs['rbs'].mean():.0%}")
fig.text(0.012,0.012,txt,fontsize=8,family="monospace",
         bbox=dict(boxstyle="round",fc="#F8F9F9",ec="#ccc"))

fig.suptitle(f"Genome spectrometer (regulatory channels) -- {ORG.replace('beril_','')} chromosome 1 ({L/1e6:.2f} Mb)\n"
             f"promoter / initiator / start-stop / -10 -35 boxes / RBS / motif  --  green=oriC red=terC",
             fontsize=12,fontweight="bold",y=0.995)
fig.savefig(OUT/"genome_spectrometer_regulatory_GMI1000.png",dpi=140,bbox_inches="tight")
print("wrote genome_spectrometer_regulatory_GMI1000.png")

(OUT/"genome_spectrometer_regulatory_GMI1000_summary.json").write_text(json.dumps({
    "organism":ORG,"contig":ctg,"length_bp":int(L),"n_cds":len(cds),
    "oriC_Mb":round(oriC/1e6,3),"terC_Mb":round(terC/1e6,3),
    "genome_wide_fractions":{
        "start_ATG":round(float(arrs['atg'].mean()),3),
        "alt_start_GTG_TTG":round(float(arrs['altstart'].mean()),3),
        "stop_TGA":round(float(arrs['tga'].mean()),3),
        "minus35_hit":round(float(arrs['h35'].mean()),3),
        "minus10_hit":round(float(arrs['h10'].mean()),3),
        "spacer_canonical":round(float(arrs['spacer'].mean()),3),
        "rbs_strong":round(float(arrs['rbs'].mean()),3),
    },
    "motif":MOTIF,"motif_mean_per_kb":round(float(ch_motif.mean()),3),
    "windows":int(len(centers)),
},indent=2))
