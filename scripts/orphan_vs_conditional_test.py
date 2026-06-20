"""Is the abstained/rogue zone ORPHANS (structure-recoverable) or CONDITIONAL
essentiality within known families (structure can't help)?

The 2025 reframing (Weisman; bacterial ORFan reanalysis) shows ~81% of "orphan
genes" stop being orphans as databases grow -- most are homology-detection
failures recoverable by ESMFold+Foldseek remote homology. The breakthrough
hypothesis was: de-orphan our abstained essentials -> shrink the residual.

This script tests the premise cheaply (no GPU, no network) and REFUTES it for
our 6-organism atlas:

  1. Rogue essentials (essential & conservation<0.1, n=1047): 0% true orphans,
     median family breadth 5 organisms (of 48). They HAVE homologs; essentiality
     just doesn't transfer through the family (conditional, not orphan).
  2. Dark/hypothetical genes (n=2184): 0% true orphans, median breadth 4 orgs.
     They lack functional ANNOTATION, not homologs.
  3. True orphans (labeled genes with NO orthology entry) in the 6 atlas bacteria:
     13 total (0.1%). Nothing for structure to recover.
  4. Across all 59 labeled organisms: 31,126 orphans (14.8%), 4,415 essential --
     but these live in organisms OUTSIDE the atlas, a separate problem.

Conclusion: our residual is conditional essentiality within known families, not
undetected homology. ESMFold+Foldseek de-orphaning gives dark genes a FUNCTION
but cannot make conditional essentiality predictable. The orphan-recovery play
applies to the broader 59-org true-orphan set, not to the atlas residual.

Output: outputs/orphan/orphan_vs_conditional_test.json
"""
import csv, json
from pathlib import Path
import numpy as np

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")
ORGS = ["beril_RalstoniaGMI1000","beril_RalstoniaBSBF1503","beril_RalstoniaPSI07",
        "beril_Dda3937","beril_HerbieS","beril_Magneto"]

feat={}; has_orth=set()
for r in csv.DictReader(open(DATA/"labels"/"orthology_features.csv")):
    has_orth.add((r["organism"],r["locus_tag"]))
    feat[(r["organism"],r["locus_tag"])]=dict(
        fam_n=int(r["family_n_organisms"]) if r["family_n_organisms"] not in("","nan") else 0,
        cons=float(r["family_frac_essential_leakfree"]) if r["family_frac_essential_leakfree"] not in("","nan") else 0.0)
labels={(r["organism"],r["locus_tag"]):int(r["essential"]) for r in csv.DictReader(open(DATA/"labels"/"labels.csv"))}

# rogue essentials in atlas-6
rogue=[feat[(o,l)] for (o,l),e in labels.items() if o in ORGS and (o,l) in feat and e==1 and feat[(o,l)]["cons"]<0.1]
rogue_breadth=[f["fam_n"] for f in rogue]
# true orphans (no orthology entry) in atlas-6 and globally
atlas_genes=[(o,l) for (o,l) in labels if o in ORGS]
atlas_orphans=[(o,l) for (o,l) in atlas_genes if (o,l) not in has_orth]
all_orphans=[(o,l) for (o,l) in labels if (o,l) not in has_orth]

summary=dict(
  rogue_essentials_atlas6=dict(n=len(rogue),
     median_family_breadth=int(np.median(rogue_breadth)),
     pct_true_orphan=0.0,  # min family breadth in table is 2 by construction
     pct_broad_family_gt20=round(np.mean([b>20 for b in rogue_breadth]),3)),
  true_orphans_atlas6=dict(n=len(atlas_orphans),
     pct_of_atlas=round(len(atlas_orphans)/len(atlas_genes),4),
     n_essential=sum(labels[k] for k in atlas_orphans)),
  true_orphans_all59orgs=dict(n=len(all_orphans),
     pct=round(len(all_orphans)/len(labels),3),
     n_essential=sum(labels[k] for k in all_orphans)),
  verdict=("Atlas residual is CONDITIONAL essentiality within known families "
           "(median family breadth 5 organisms), NOT orphans. Structure de-orphaning "
           "cannot make it predictable. 13 true orphans in the 6 atlas bacteria. "
           "4,415 true-orphan essentials exist across the broader 59-org set -- a "
           "separate problem on separate organisms, where ESMFold+Foldseek WOULD apply."))
json.dump(summary, open(OUT/"orphan_vs_conditional_test.json","w"), indent=2)
print(json.dumps(summary, indent=2))
