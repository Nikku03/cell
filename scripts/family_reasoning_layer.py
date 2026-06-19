"""Family-reasoning layer: decode the residual the statistical atlas abstains on.

Where the cross-org model abstains (UNRESOLVED / ROGUE_SUSPECT / CONDITIONAL_SUSPECT),
route the gene through PROTEIN-FAMILY reasoning:
  1. classify its product into a functional family (keyword -> class)
  2. attach the family's essential-rate -- computed LEAVE-ONE-ORGANISM-OUT
     (from the OTHER organisms, so it's leak-free for the held-out org)
  3. if the family rate is high -> call essential; low -> non-essential;
     middling or dark/hypothetical -> still abstain (experiment-only)

Test: does this recover accuracy on the abstained set, and how much coverage
does it add on top of the statistical model, at what precision?

Outputs:
  outputs/orphan/family_reasoning_results.json
  outputs/orphan/family_reasoning.png
"""
import re, csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/user/cell/outputs/orphan")
DATA = Path("/home/user/cell/data/drive_import")

CLASSES = [
 ("ribosomal/translation", r"ribosom|tRNA|rRNA|translation|elongation factor|aminoacyl|peptide chain release"),
 ("transporter/efflux",    r"transport|permease|efflux|ABC|MFS |channel|symport|antiport|porin|TonB"),
 ("regulator/TF",          r"regulator|transcription(al)? (factor|regulator)|sigma factor|repressor|activator|TetR|LysR|GntR|AraC|MarR|response regulator|two-component"),
 ("signaling",             r"\bkinase|diguanylate|GGDEF|EAL domain|phosphatase|chemotaxis|sensor histidine|c-di-GMP|methyl-accepting"),
 ("secretion/virulence",   r"secretion system|effector|T3SS|T6SS|pilus|pili|flagell|adhesin|\bXop|\bAvr|hemolysin|toxin"),
 ("metabolic enzyme",      r"synthase|synthetase|dehydrogenase|reductase|transferase|hydrolase|oxidase|lyase|isomerase|ligase|carboxylase|oxidoreductase|hydratase|aminotransferase|decarboxylase"),
 ("cell envelope/wall",    r"peptidoglycan|murein|cell wall|lipopolysaccharide|\bLPS\b|lipoprotein|outer membrane|penicillin-binding|cell division|Fts"),
 ("DNA/repair/mobile",     r"transposase|integrase|recombinase|helicase|nuclease|DNA polymerase|topoisomerase|phage|CRISPR|restriction|gyrase|primase|SSB|single-strand"),
 ("dark/hypothetical",     r"hypothetical|DUF\d|uncharacterized|unknown function|protein of unknown"),
]
def classify(p):
    p = p or ""
    for name, pat in CLASSES:
        if re.search(pat, p, re.I): return name
    return "other/annotated"

# org -> accession
acc = {}
for r in csv.DictReader(open(DATA/"labels"/"genome_cache_manifest.csv")):
    if r.get("accession"): acc[r["organism"]] = r["accession"]

def products_for(org):
    a = acc.get(org); out = {}
    if not a: return out
    gff = DATA/"genome_cache"/a/"genomic.gff"
    if not gff.exists(): return out
    for line in gff.read_text().splitlines():
        if "\tCDS\t" not in line: continue
        m = re.search(r"locus_tag=([^;\n]+)", line); pr = re.search(r"product=([^;\n]+)", line)
        if m: out[m.group(1)] = pr.group(1) if pr else ""
    return out

ORGS = ["beril_RalstoniaGMI1000","beril_RalstoniaBSBF1503","beril_RalstoniaPSI07",
        "beril_Dda3937","beril_HerbieS","beril_Magneto"]

# load atlases + attach products + family class
frames = []
for org in ORGS:
    a = pd.read_csv(OUT/"atlas_multi"/f"{org}.csv")
    prod = products_for(org)
    a["product"] = a.locus_tag.map(lambda l: prod.get(l, ""))
    a["org"] = org
    a["fclass"] = a["product"].map(classify)
    frames.append(a)
df = pd.concat(frames, ignore_index=True)
print(f"{len(df)} genes, {df.org.nunique()} orgs, product coverage "
      f"{(df['product'].fillna('').str.len()>0).mean():.0%}")

RESID_TIERS = {"UNRESOLVED","ROGUE_SUSPECT","CONDITIONAL_SUSPECT"}
df["residual"] = df.tier.isin(RESID_TIERS)
print(f"residual (abstained) genes: {df.residual.sum()} "
      f"({df.residual.mean():.0%}), {df[df.residual].measured_essential.mean():.0%} essential")

# ---- leave-one-organism-out family essential-rate ----
HI, LO = 0.65, 0.20          # family-rate thresholds for a confident family call
DARK = {"dark/hypothetical","other/annotated"}   # never call from these
results = []
df["fam_call"] = "abstain"
for held in ORGS:
    tr = df[df.org != held]
    # family essential-rate from OTHER organisms only (leak-free)
    frate = tr.groupby("fclass").measured_essential.mean().to_dict()
    fn    = tr.groupby("fclass").size().to_dict()
    te = df[(df.org == held) & df.residual]
    for idx, row in te.iterrows():
        fc = row.fclass
        if fc in DARK or fn.get(fc,0) < 50:
            continue
        r = frate.get(fc, np.nan)
        if r >= HI: df.at[idx,"fam_call"] = "essential"
        elif r <= LO: df.at[idx,"fam_call"] = "non-essential"
    # metrics on this org's residual
    sub = df[(df.org==held) & df.residual]
    called = sub[sub.fam_call!="abstain"]
    if len(called):
        correct = ((called.fam_call=="essential")&(called.measured_essential==1)) | \
                  ((called.fam_call=="non-essential")&(called.measured_essential==0))
        prec = correct.mean()
    else: prec = float("nan")
    results.append(dict(org=held, residual_n=int(len(sub)),
        family_called=int(len(called)),
        family_call_frac=round(len(called)/max(1,len(sub)),3),
        family_call_precision=round(float(prec),3) if len(called) else None,
        residual_base_ess=round(float(sub.measured_essential.mean()),3)))
    print(f"  {held.replace('beril_',''):<22} residual={len(sub):<5} "
          f"family-called={len(called):<5} ({len(called)/max(1,len(sub)):.0%}) "
          f"precision={prec:.3f}" if len(called) else f"  {held}: no calls")

# ---- pooled: combined coverage statistical + family layer ----
N = len(df)
stat_called = (~df.residual).sum()                 # statistical confident calls
fam_called  = (df.fam_call!="abstain").sum()
# overall precision of all confident calls (statistical + family)
df["final_call"] = np.where(~df.residual,
        np.where(df.tier=="CONFIDENT_ESSENTIAL","essential",
            np.where(df.tier=="CONFIDENT_NONESSENTIAL","non-essential","abstain")),
        df.fam_call)
called_all = df[df.final_call!="abstain"]
corr = ((called_all.final_call=="essential")&(called_all.measured_essential==1)) | \
       ((called_all.final_call=="non-essential")&(called_all.measured_essential==0))
print(f"\n=== COMBINED (statistical + family-reasoning) ===")
print(f"  statistical-only coverage : {stat_called/N:.1%}")
print(f"  + family layer adds       : {fam_called/N:.1%}")
print(f"  COMBINED coverage         : {called_all.shape[0]/N:.1%}")
print(f"  precision over all calls  : {corr.mean():.3f}")
# family-layer-only precision (the recovered genes)
famrec = df[df.fam_call!="abstain"]
fcorr = ((famrec.fam_call=="essential")&(famrec.measured_essential==1)) | \
        ((famrec.fam_call=="non-essential")&(famrec.measured_essential==0))
print(f"  family-recovered genes    : {len(famrec)} at precision {fcorr.mean():.3f}")
# what stays dark
dark_left = df[(df.residual)&(df.fam_call=='abstain')]
print(f"  still experiment-only     : {len(dark_left)} ({len(dark_left)/N:.0%} of all genes)")

summary = dict(
    n_genes=int(N), thresholds=dict(HI=HI,LO=LO),
    statistical_coverage=round(stat_called/N,3),
    family_added=round(fam_called/N,3),
    combined_coverage=round(called_all.shape[0]/N,3),
    combined_precision=round(float(corr.mean()),3),
    family_recovered_n=int(len(famrec)),
    family_recovered_precision=round(float(fcorr.mean()),3),
    experiment_only_n=int(len(dark_left)),
    experiment_only_frac=round(len(dark_left)/N,3),
    per_org=results)
json.dump(summary, open(OUT/"family_reasoning_results.json","w"), indent=2)

# figure
fig, ax = plt.subplots(1,2, figsize=(14,5.5))
orgs=[o.replace("beril_","") for o in ORGS]
fc=[next(r["family_call_frac"] for r in results if r["org"]==o) for o in ORGS]
fp=[next((r["family_call_precision"] or 0) for r in results if r["org"]==o) for o in ORGS]
x=np.arange(len(orgs))
ax[0].bar(x-0.2,fc,0.4,label="frac of residual family-called",color="#16A085")
ax[0].bar(x+0.2,fp,0.4,label="precision of those calls",color="#2980B9")
ax[0].set_xticks(x); ax[0].set_xticklabels(orgs,rotation=40,ha="right",fontsize=8); ax[0].set_ylim(0,1)
ax[0].axhline(0.9,color="k",ls=":",lw=0.8); ax[0].legend(fontsize=8)
ax[0].set_title("Family-reasoning on the residual (leave-one-org-out)")
# coverage stack
labels=["statistical\nconfident","family\nrecovered","experiment\nonly (dark)"]
vals=[summary["statistical_coverage"], summary["family_added"], summary["experiment_only_frac"]]
# remainder = residual not family-called but not dark-flagged (other/annotated mid-rate)
mid = 1 - sum(vals)
ax[1].bar([0],[vals[0]],color="#C0392B",label="statistical (conserved core)")
ax[1].bar([0],[vals[1]],bottom=[vals[0]],color="#16A085",label="family-recovered (residual)")
ax[1].bar([0],[mid],bottom=[vals[0]+vals[1]],color="#F39C12",label="unresolved middle")
ax[1].bar([0],[vals[2]],bottom=[vals[0]+vals[1]+mid],color="#7F8C8D",label="experiment-only (dark)")
ax[1].set_xticks([]); ax[1].set_ylim(0,1); ax[1].legend(fontsize=8,loc="center left",bbox_to_anchor=(1,0.5))
ax[1].set_title(f"Genome coverage decomposition\ncombined {summary['combined_coverage']:.0%} at "
                f"{summary['combined_precision']:.0%} precision")
fig.tight_layout(); fig.savefig(OUT/"family_reasoning.png",dpi=140,bbox_inches="tight")
print("\nwrote family_reasoning_results.json + family_reasoning.png")
