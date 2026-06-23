"""Annotate the DEG cells from LOCAL genome_cache GFFs -- no NCBI fetch.

The "protein fetching problem" was a non-problem: the correct genomes are
already in data/drive_import/genome_cache/ with the right locus_tags. The Colab
fetch went to NCBI for mismatched strains; this reads what's on disk.

Per DEG cell, three join routes (best available wins):
  1 direct  : our locus_tag == GFF locus_tag           (mtub Rv####, saur SAOUHSC)
  2 gene    : our gene_name -> GFF gene -> product      (spne/mgen, RefSeq renamed
              SP_0001 -> SP_RS00005, so locus_tag won't match but gene does)
  3 DEG db  : gene_name -> COG family + func + description from DEG itself

Output: appends DEG rows to data/drive_import/labels/gene_annotations.csv
        writes data/drive_import/labels/deg_families.csv
            (organism, locus_tag, product, cog, func_class, source)
So Wheel 2's slot classifier + family matching run on the DEG cells with no
network and no retrain.
"""
import csv, re
from collections import defaultdict
from pathlib import Path

GCF = {
    "mtub":                    "GCF_000195955.2",   # H37Rv,  Rv####
    "saur_NCTC8325_biotradis": "GCF_000013425.1",   # NCTC8325, SAOUHSC_*
    "spneT4":                  "GCF_000006885.1",   # TIGR4, SP_RS* (renamed)
    "mgen":                    "GCF_000027325.1",   # G37,   MG_RS* (renamed)
}
re_lt = re.compile(r"locus_tag=([^;]+)")
re_pr = re.compile(r"product=([^;\n]+)")
re_gn = re.compile(r"gene=([^;]+)")

# our genes + gene_names
genes = defaultdict(list); gname = {}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"] in GCF:
        genes[r["organism"]].append(r["locus_tag"])
        g = (r.get("gene_name") or "").strip()
        if g and g != "-":
            gname[(r["organism"], r["locus_tag"])] = g

# DEG family annotation, keyed by (species-substring, gene_name) -> (cog, func, desc)
DEG_SPECIES = {
    "mtub": "Mycobacterium tuberculosis H37Rv",
    "saur_NCTC8325_biotradis": "Staphylococcus aureus NCTC 8325",
    "spneT4": "Streptococcus pneumoniae",
    "mgen": "Mycoplasma genitalium G37",
}
deg_fam = defaultdict(dict)   # org -> gene_name -> (cog, func, desc)
for r in csv.DictReader(open("data/drive_import/deg/deg_bacteria_essential_genes.csv")):
    for org, sp in DEG_SPECIES.items():
        if r["organism"] == sp and r["gene_name"]:
            deg_fam[org][r["gene_name"].lower()] = (
                r.get("cog", ""), r.get("func_class", ""), r.get("description", ""))

rows = []           # (organism, locus_tag, product, cog, func_class, source)
cov = {}
for org, gcf in GCF.items():
    gff = f"data/drive_import/genome_cache/{gcf}/genomic.gff"
    lt_prod = {}; gene_prod = {}; gene_lt = {}
    for line in open(gff):
        if "\tCDS\t" not in line:
            continue
        m = re_lt.search(line); p = re_pr.search(line); g = re_gn.search(line)
        prod = p.group(1) if p else ""
        if m and prod:
            lt_prod[m.group(1)] = prod
        if g and prod:
            gene_prod[g.group(1).lower()] = prod
        if g and m:
            gene_lt.setdefault(g.group(1).lower(), m.group(1))
    n_direct = n_gene = n_deg = n_none = 0
    for lt in genes[org]:
        gn = gname.get((org, lt), "").lower()
        cog = func = ""
        if gn and gn in deg_fam[org]:
            cog, func, _ = deg_fam[org][gn]
        raw_gn = gname.get((org, lt), "")
        # route 0: some datasets (mgen) put the PRODUCT in the gene_name column
        is_product_text = len(raw_gn) > 6 and (" " in raw_gn or "-" in raw_gn)
        if lt in lt_prod:                                  # route 1: direct
            rows.append((org, lt, lt_prod[lt], cog, func, "gff_direct")); n_direct += 1
        elif gn and gn in gene_prod:                       # route 2: gene_name symbol
            rows.append((org, lt, gene_prod[gn], cog, func, "gff_gene")); n_gene += 1
        elif is_product_text:                              # route 0: gene_name == product
            rows.append((org, lt, raw_gn, cog, func, "genename_product")); n_gene += 1
        elif gn and gn in deg_fam[org]:                    # route 3: DEG db
            rows.append((org, lt, deg_fam[org][gn][2], cog, func, "deg_db")); n_deg += 1
        else:
            rows.append((org, lt, "", cog, func, "none")); n_none += 1
    covered = len(genes[org]) - n_none
    cov[org] = (len(genes[org]), n_direct, n_gene, n_deg, covered)
    print(f"{org:<26} genes={len(genes[org]):>5}  direct={n_direct:>5} "
          f"gene={n_gene:>4} deg={n_deg:>4}  COVERED={covered:>5} "
          f"({100*covered/len(genes[org]):.0f}%)")

# write deg_families.csv (rich: product + COG family)
with open("data/drive_import/labels/deg_families.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["organism", "locus_tag", "product", "cog", "func_class", "source"])
    w.writerows(rows)
print("\nwrote data/drive_import/labels/deg_families.csv")

# UPDATE gene_annotations.csv: fill/replace DEG rows (earlier pass left them
# present but empty), append any DEG gene not already present.
ga = "data/drive_import/labels/gene_annotations.csv"
deg_prod = {(o, lt): prod for (o, lt, prod, cog, func, src) in rows if prod}
all_rows = list(csv.DictReader(open(ga)))
seen = set(); updated = 0
for r in all_rows:
    k = (r["organism"], r["locus_tag"])
    seen.add(k)
    if k in deg_prod and (not r.get("product")):
        r["product"] = deg_prod[k]; r["source"] = "deg_local"; updated += 1
appended = [(o, lt) for (o, lt) in deg_prod if (o, lt) not in seen]
with open(ga, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["organism", "locus_tag", "product", "source"])
    for r in all_rows:
        w.writerow([r["organism"], r["locus_tag"], r.get("product", ""), r.get("source", "")])
    for (o, lt) in appended:
        w.writerow([o, lt, deg_prod[(o, lt)], "deg_local"])
print(f"gene_annotations.csv: updated {updated} empty DEG rows, "
      f"appended {len(appended)} new")

n_cog = sum(1 for r in rows if r[3] and r[3] != "-")
print(f"\nrows with a COG family: {n_cog}/{len(rows)}  "
      f"(COG = precomputed protein-family assignment, no model run needed)")
