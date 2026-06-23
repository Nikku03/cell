"""Build per-gene functional annotations from local genome_cache GFFs.

Network/UniProt is blocked, but 91 genome_cache GFFs carry product= fields
(334k locus_tag->product entries). Strategy:
  1. parse all GFFs -> locus_tag -> product
  2. DIRECT match to our genes by locus_tag (~20%)
  3. OG PROPAGATION: for each og_id, take the majority product among its
     directly-annotated members and assign it to ALL members. Orthologs share
     function, so this lifts coverage of the conserved core dramatically.

Output: data/drive_import/labels/gene_annotations.csv
        (organism, locus_tag, product, source[direct|og|none])
"""
import csv, re, glob
from collections import defaultdict, Counter
from pathlib import Path

OUTCSV = Path("data/drive_import/labels/gene_annotations.csv")

re_lt = re.compile(r"locus_tag=([^;]+)")
re_pr = re.compile(r"product=([^;\n]+)")

print("Parsing genome_cache GFFs ...")
gff_product = {}
for gff in glob.glob("data/drive_import/genome_cache/*/genomic.gff"):
    with open(gff) as f:
        for line in f:
            if "\tCDS\t" not in line:
                continue
            m = re_lt.search(line); p = re_pr.search(line)
            if m and p:
                prod = p.group(1).strip()
                if prod and prod.lower() != "hypothetical protein":
                    gff_product[m.group(1)] = prod
                else:
                    gff_product.setdefault(m.group(1), "hypothetical protein")
print(f"  {len(gff_product):,} locus_tag -> product entries")

# our genes + OG membership
print("Loading genes + OG membership ...")
genes = []
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    genes.append((r["organism"], r["locus_tag"]))
og_by_gene = {}
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"):
        og_by_gene[(r["organism"], r["locus_tag"])] = r["og_id"]

# step 2: direct match
direct = {}
for (org, lt) in genes:
    p = gff_product.get(lt)
    if p:
        direct[(org, lt)] = p
print(f"  direct GFF match: {len(direct):,} / {len(genes):,} ({len(direct)/len(genes)*100:.1f}%)")

# step 3: OG propagation -- majority non-hypothetical product per OG
og_products = defaultdict(list)
for (org, lt), p in direct.items():
    og = og_by_gene.get((org, lt))
    if og and p.lower() != "hypothetical protein":
        og_products[og].append(p)
og_consensus = {}
for og, plist in og_products.items():
    og_consensus[og] = Counter(plist).most_common(1)[0][0]
print(f"  OGs with a consensus product: {len(og_consensus):,}")

# assemble final annotation
n_direct = n_og = n_none = 0
rows = []
for (org, lt) in genes:
    if (org, lt) in direct and direct[(org, lt)].lower() != "hypothetical protein":
        rows.append((org, lt, direct[(org, lt)], "direct")); n_direct += 1
    else:
        og = og_by_gene.get((org, lt))
        if og and og in og_consensus:
            rows.append((org, lt, og_consensus[og], "og")); n_og += 1
        elif (org, lt) in direct:
            rows.append((org, lt, direct[(org, lt)], "direct_hypo")); n_direct += 1
        else:
            rows.append((org, lt, "", "none")); n_none += 1

with open(OUTCSV, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["organism", "locus_tag", "product", "source"])
    w.writerows(rows)

covered = len(genes) - n_none
print(f"\n=== annotation coverage ===")
print(f"  direct (real product) : {n_direct:,}")
print(f"  via OG propagation    : {n_og:,}")
print(f"  no annotation         : {n_none:,}")
print(f"  TOTAL covered         : {covered:,} / {len(genes):,} ({covered/len(genes)*100:.1f}%)")

# coverage among ESSENTIAL genes specifically (what matters for the minimal cell)
ess = {}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    ess[(r["organism"], r["locus_tag"])] = int(r["essential"])
ann = {(o, l): (p, s) for o, l, p, s in rows}
ess_keys = [k for k, v in ess.items() if v == 1]
ess_cov = sum(1 for k in ess_keys if ann.get(k, ("", "none"))[1] != "none")
print(f"  essential-gene coverage: {ess_cov:,} / {len(ess_keys):,} "
      f"({ess_cov/len(ess_keys)*100:.1f}%)")
print(f"\nwrote {OUTCSV}")
