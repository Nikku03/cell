"""DEG cross-validation + honest audit of what DEG actually offers our project.

VERIFIED findings (corrects two earlier overstatements):
  - DEG conditional essentials = 2,506 rows (NOT 8,408; the rest are rich-media
    variants like LB/MHBII/blood-agar, not real conditions).
  - Of our 6 atlas bacteria, only Ralstonia GMI1000 is in DEG. The conditional
    organisms (M. tuberculosis, P. aeruginosa, Francisella, Salmonella,
    Acinetobacter) are NOT in our 48-org orthology -> their conditional labels
    cannot be mapped to our model without ingesting those genomes.

USABLE NOW: GMI1000 is in DEG (independent screen) -> cross-validate our focal
organism's essentiality labels against an independent screen, via a GFF
gene_name->locus_tag bridge (our labels carry no gene_name; DEG carries no
locus_tag).

Outputs:
  outputs/orphan/deg_cross_validation.json
  outputs/orphan/deg_gmi1000_disagreements.csv
"""
import csv, re, json
from pathlib import Path
from collections import Counter

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")

deg = list(csv.DictReader(open(DATA/"deg"/"deg_bacteria_essential_genes.csv")))
COND = re.compile(r'required for|tolerance|model of|infection|sputum|tobramycin|stress|cholesterol|bile|murine|in vivo|host|serum', re.I)
cond = [r for r in deg if COND.search(r['condition'])]
print(f"DEG total essential rows: {len(deg)} | TRUE conditional: {len(cond)}")
print("conditional organisms:", dict(Counter(r['organism'].split(' subsp')[0][:30] for r in cond).most_common(6)))

# ---- GMI1000 cross-screen validation ----
acc="GCF_000009125.1"
g2l={}; l2prod={}
for line in open(DATA/"genome_cache"/acc/"genomic.gff"):
    if "\tCDS\t" not in line and "\tgene\t" not in line: continue
    lt=re.search(r"locus_tag=([^;\n]+)",line); gn=re.search(r"gene=([^;\n]+)",line); pr=re.search(r"product=([^;\n]+)",line)
    if lt and gn: g2l[gn.group(1).lower()]=lt.group(1)
    if lt and pr: l2prod[lt.group(1)]=pr.group(1)

ours={r['locus_tag']:int(r['essential']) for r in csv.DictReader(open(DATA/"labels"/"labels.csv"))
      if r['organism']=='beril_RalstoniaGMI1000'}
deg_g=[r for r in deg if 'GMI1000' in r['organism']]
mapped=[(g2l[r['gene_name'].lower()], r['gene_name']) for r in deg_g
        if r['gene_name'] and r['gene_name']!='-' and r['gene_name'].lower() in g2l]
inboth=[(lt,gn) for lt,gn in mapped if lt in ours]
agree=[(lt,gn) for lt,gn in inboth if ours[lt]==1]
dis=[(lt,gn) for lt,gn in inboth if ours[lt]==0]
print(f"\nGMI1000 cross-screen: {len(deg_g)} DEG essentials -> {len(mapped)} mapped -> {len(inboth)} in our screen")
print(f"  agreement (DEG-ess & our-ess): {len(agree)}/{len(inboth)} = {len(agree)/len(inboth):.0%}")

# the disagreements -- DEG essential, we call non-essential (candidate conditional/screen-specific)
print(f"\n{len(dis)} disagreements (DEG essential, our non-essential):")
dis_rows=[]
for lt,gn in dis:
    p=l2prod.get(lt,"")
    dis_rows.append(dict(locus_tag=lt, gene=gn, product=p))
    print(f"   {gn:<8} {lt:<13} {p[:50]}")
import pandas as pd
pd.DataFrame(dis_rows).to_csv(OUT/"deg_gmi1000_disagreements.csv", index=False)

json.dump(dict(
    deg_total_essential=len(deg),
    deg_true_conditional=len(cond),
    conditional_organisms_not_in_our_orthology=True,
    gmi1000_deg_essentials=len(deg_g),
    gmi1000_mapped=len(mapped),
    gmi1000_in_both=len(inboth),
    gmi1000_agreement=round(len(agree)/len(inboth),3),
    gmi1000_disagreements=len(dis),
    note=("Independent-screen agreement on GMI1000 NAMED core essentials is high "
          "(93%); this is the conserved core, where screens agree. DEG conditional "
          "data is in pathogen organisms disjoint from our orthology -- using it "
          "requires ingesting those genomes.")), open(OUT/"deg_cross_validation.json","w"), indent=2)
print("\nwrote deg_cross_validation.json + disagreements csv")
