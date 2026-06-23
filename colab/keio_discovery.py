"""Discovery probes on E. coli Keio using the STRICT b-number bridge.

The earlier full-cell analysis was constrained because Keio annotations were
OG-propagated, lumping diverse gene families into one product string. With the
strict bridge (anchor_product + synteny_walk validated by product token),
1,190 Keio genes have real MG1655 gene names + products, and we can ask
gene-level questions honestly.

Probes:
  1 FBA-vs-truth discordance  (now possible -- 494 bridgeable to FBA)
        - FBA essential, Keio non-essential  -> isozyme rescue or wrong-medium
        - FBA non-essential, Keio essential  -> non-metabolic essential FBA can't see
  2 Universal-core-but-Keio-dispensable with NAMED MG1655 genes
        - which conserved genes can E. coli specifically afford to lose?
  3 LysR regulator audit  (the previous "35 essential LysR" was OG artifact;
        - how many true LysR essentials does the bridge confirm?)
  4 Paralog redundancy with proper gene names
        - real paralog groups in E. coli with their essential/non-essential split.
"""
import csv, json, re
from collections import defaultdict, Counter
from pathlib import Path

OUT = Path("outputs/orphan")
ORG = "beril_Keio"

# bridge (strict only)
STRICT = {"anchor_product", "anchor_product_round2", "synteny_walk"}
bridge = {}; b2gene = {}; b2prod = {}
for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv")):
    if r["source"] in STRICT:
        bridge[r["locus_tag"]] = r["b_number"]
        b2gene[r["b_number"]] = r["gene_name"]
        b2prod[r["b_number"]] = r["product"]
print(f"strict bridge: {len(bridge)} Keio -> b-numbers, "
      f"{sum(1 for b in bridge.values() if b2gene.get(b))} with MG1655 gene name")

# truth + OG presence + FBA
truth = {r["locus_tag"]: int(r["essential"])
         for r in csv.DictReader(open("data/drive_import/labels/labels.csv"))
         if r["organism"] == ORG}
og_of = {}; og_present = defaultdict(set)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"):
        og_of[(r["organism"], r["locus_tag"])] = r["og_id"]
        og_present[r["og_id"]].add(r["organism"])
n_orgs = len({o for s in og_present.values() for o in s})
fba = {r["locus_tag"]: int(r["fba_essential"])
       for r in csv.DictReader(open("data/drive_import/labels/fba_features.csv"))
       if r["organism"] == ORG}

# ============== PROBE 1: FBA-vs-truth discordance ==============
print("\n=== PROBE 1: where FBA & Tn-seq truth DISAGREE (real gene-level claims) ===")
fba_pos_truth_neg = []   # FBA says essential, Tn-seq says no -> bypass/isozyme
fba_neg_truth_pos = []   # FBA says non-essential, Tn-seq says yes -> non-metabolic essential
for lt, b in bridge.items():
    if b not in fba or lt not in truth: continue
    if fba[b] == 1 and truth[lt] == 0:
        fba_pos_truth_neg.append((lt, b, b2gene.get(b, "?"), b2prod.get(b, "?")))
    elif fba[b] == 0 and truth[lt] == 1:
        fba_neg_truth_pos.append((lt, b, b2gene.get(b, "?"), b2prod.get(b, "?")))
print(f"\n  A. FBA-essential but Keio-dispensable: {len(fba_pos_truth_neg)} "
      "(candidates for isozyme rescue / wrong medium / paralog backup)")
for lt, b, g, p in fba_pos_truth_neg[:15]:
    print(f"     {b:>6} {g:<8} -- {p[:55]}")

print(f"\n  B. Keio-essential but FBA misses: {len(fba_neg_truth_pos)} "
      "(non-metabolic essentials -- structure, signaling, RNA)")
for lt, b, g, p in fba_neg_truth_pos[:15]:
    print(f"     {b:>6} {g:<8} -- {p[:55]}")

# ============== PROBE 2: Universal-core-but-Keio-dispensable, GENE NAMES ==============
print("\n=== PROBE 2: universal-core genes that E. coli specifically discards ===")
vest = []
for lt, b in bridge.items():
    if lt not in truth or truth[lt] != 0: continue
    og = og_of.get((ORG, lt))
    if og and len(og_present[og]) / n_orgs >= 0.9:
        vest.append((lt, b, b2gene.get(b, "?"), b2prod.get(b, "?"),
                     len(og_present[og])))
vest.sort(key=lambda x: -x[4])
print(f"  {len(vest)} bridged & named (vs 537 before with OG-propagated annotations)")
print("  top 15 by conservation breadth:")
for lt, b, g, p, n in vest[:15]:
    print(f"     {b:>6} {g:<8} present-in {n}/{n_orgs} orgs -- {p[:48]}")

# ============== PROBE 3: LysR regulator audit ==============
print("\n=== PROBE 3: LysR-family regulator audit (was 35 essential w/ OG-propagation) ===")
lysr_bridged = [(lt, b, b2gene.get(b, "?"), b2prod.get(b, "?"))
                for lt, b in bridge.items()
                if "LysR" in b2prod.get(b, "") or "lysr" in b2prod.get(b, "").lower()]
lysr_ess = [(lt, b, g, p) for lt, b, g, p in lysr_bridged if truth.get(lt) == 1]
print(f"  bridged Keio genes with MG1655 product containing 'LysR': {len(lysr_bridged)}")
print(f"  of those, essential in Keio truth: {len(lysr_ess)}")
for lt, b, g, p in lysr_ess[:10]:
    print(f"     {b:>6} {g:<8} -- {p[:55]}")

# ============== PROBE 4: paralog redundancy using REAL MG1655 gene names ==============
print("\n=== PROBE 4: paralog redundancy using MG1655 gene names ===")
prod_to_b = defaultdict(list)
for b, p in b2prod.items():
    if p: prod_to_b[p].append(b)
multi_prod = {p: bs for p, bs in prod_to_b.items() if len(bs) >= 2}
print(f"  MG1655 product strings shared by >=2 b-numbers: {len(multi_prod)}")
keio_by_b = {b: lt for lt, b in bridge.items()}
ess_distrib = []
for p, bs in multi_prod.items():
    ks = [keio_by_b.get(b) for b in bs if keio_by_b.get(b) is not None]
    if len(ks) < 2: continue
    ess_n = sum(truth.get(k, 0) for k in ks)
    ess_distrib.append((p, len(ks), ess_n))
ess_distrib.sort(key=lambda x: -x[1])
print(f"  paralog groups with >=2 BRIDGED Keio members: {len(ess_distrib)}")
all_ess = sum(1 for _, n, e in ess_distrib if e == n)
none_ess = sum(1 for _, n, e in ess_distrib if e == 0)
print(f"  ALL essential (no redundancy): {all_ess}")
print(f"  NONE essential (full redundancy): {none_ess}")
print("\n  largest paralog groups by MG1655 product (bridged):")
for p, n, e in ess_distrib[:12]:
    print(f"     {n:>2} copies, {e} essential: {p[:60]}")

# ---- save ----
out = dict(
    bridge_size=len(bridge),
    fba_essential_keio_dispensable=[dict(lt=lt, b=b, gene=g, product=p)
                                    for lt, b, g, p in fba_pos_truth_neg],
    keio_essential_fba_missed=[dict(lt=lt, b=b, gene=g, product=p)
                               for lt, b, g, p in fba_neg_truth_pos],
    universal_core_dispensable_top30=[dict(lt=lt, b=b, gene=g, product=p, breadth=n)
                                      for lt, b, g, p, n in vest[:30]],
    lysr_essentials_true=[dict(b=b, gene=g, product=p) for _, b, g, p in lysr_ess],
    paralog_groups_top15=[dict(product=p, n=n, essential=e) for p, n, e in ess_distrib[:15]],
)
json.dump(out, open(OUT / "keio_discovery.json", "w"), indent=2)
print(f"\nwrote outputs/orphan/keio_discovery.json")
