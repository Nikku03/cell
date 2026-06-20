"""Integrate DEG organisms into the label set by AUGMENTING the three driver
CSVs, then (separately) rebuild the cache with build_cache.py.

Inputs (produced by fetch_proteomes.py + assign_ogs.py):
  --work/proteomes/<org>.genes.csv   locus_tag, gene_name, protein_id
  --work/og_assignments.csv          organism, locus_tag, og_id, pident, qcov
  --work/deg_manifest.csv            deg_dataset, org, species   (org = safe name)
  data/drive_import/deg/deg_bacteria_essential_genes.csv  (deg_dataset, gene_name, ...)

Labelling rule (per DEG org):
  a gene that mmseqs-assigned to an existing OG is labelled
    essential=1  if its gene_name is in that dataset's DEG-essential set,
    essential=0  otherwise.
  We report the match rate (DEG-essential gene_names found in the proteome) and
  DROP organisms below --min_match (their negatives would be unreliable).

Output: a new labels dir (--out_dir) = copy of the originals + appended DEG rows
        for labels.csv / orthology_features.csv / clade_splits.csv.
"""
import csv, argparse, shutil, re
from pathlib import Path
from collections import defaultdict

# coarse clade assignment so leave-one-clade-out can use DEG orgs
CLADE_KEYWORDS = [
    ("pseudomonas", "pseudomonas"), ("ralstonia", "ralstonia"),
    ("burkholderia", "burkholderia"), ("dickeya", "dickeya"),
    ("shewanella", "shewanella"), ("escherichia", "escherichia"),
    ("salmonella", "salmonella"), ("vibrio", "vibrio"),
    ("bacillus", "bacillus"), ("staphylococcus", "staphylococcus"),
    ("streptococcus", "streptococcus"), ("mycobacterium", "mycobacterium"),
    ("mycoplasma", "mycoplasma"), ("campylobacter", "campylobacter"),
    ("helicobacter", "helicobacter"), ("acinetobacter", "acinetobacter"),
    ("francisella", "francisella"), ("neisseria", "neisseria"),
    ("haemophilus", "haemophilus"), ("caulobacter", "caulobacter"),
    ("bacteroides", "bacteroides"), ("porphyromonas", "bacteroides"),
    ("sinorhizobium", "rhizobiales"), ("rhodopseudomonas", "rhizobiales"),
    ("brevundimonas", "caulobacter"), ("sphingomonas", "sphingomonas"),
    ("synechococcus", "cyanobacteria"), ("providencia", "enterobacter"),
    ("klebsiella", "enterobacter"),
]


def clade_for(species):
    s = species.lower()
    for kw, cl in CLADE_KEYWORDS:
        if kw in s:
            return cl
    return re.split(r"[ _]", s)[0] or "other"   # genus fallback


def norm_name(g):
    return (g or "").strip().lower()


def main(work, deg_dir, src_dir, out_dir, min_match, min_pident, min_assigned):
    work = Path(work); deg_dir = Path(deg_dir); src_dir = Path(src_dir); out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # manifest: deg_dataset -> (org safe name, species)
    manifest = {}
    for r in csv.DictReader(open(work / "deg_manifest.csv")):
        manifest[r["deg_dataset"]] = (r["org"], r["species"])
    org_to_dataset = {v[0]: k for k, v in manifest.items()}
    org_species = {v[0]: v[1] for v in manifest.values()}

    # DEG essential gene_names per dataset
    deg_ess = defaultdict(set)
    for r in csv.DictReader(open(deg_dir / "deg_bacteria_essential_genes.csv")):
        gn = norm_name(r.get("gene_name"))
        if gn and gn != "-":
            deg_ess[r["deg_dataset"]].add(gn)

    # proteome gene tables: org -> {locus_tag: gene_name}
    gene_name = defaultdict(dict)
    for gpath in (work / "proteomes").glob("*.genes.csv"):
        org = gpath.name[:-len(".genes.csv")]
        for r in csv.DictReader(open(gpath)):
            gene_name[org][r["locus_tag"]] = norm_name(r.get("gene_name"))

    # OG assignments: org -> [(locus_tag, og_id)] (filtered by pident)
    assigned = defaultdict(list)
    for r in csv.DictReader(open(work / "og_assignments.csv")):
        try:
            if float(r.get("pident", 0)) < min_pident:
                continue
        except ValueError:
            continue
        assigned[r["organism"]].append((r["locus_tag"], r["og_id"]))

    # build augmented rows
    add_labels = []; add_orth = []; add_clade = []
    report = []
    for org, pairs in assigned.items():
        ds = org_to_dataset.get(org)
        if ds is None:
            continue
        ess_names = deg_ess.get(ds, set())
        if not ess_names or len(pairs) < min_assigned:
            report.append((org, len(pairs), 0, 0.0, "skip:too_few_or_no_ess"))
            continue
        # paralog counts within this org
        og_count = defaultdict(int)
        for _, og in pairs:
            og_count[og] += 1
        names = gene_name.get(org, {})
        matched_ess = set()
        rows_l = []; rows_o = []
        for lt, og in pairs:
            gn = names.get(lt, "")
            is_ess = 1 if gn and gn in ess_names else 0
            if is_ess:
                matched_ess.add(gn)
            rows_l.append(dict(organism=org, locus_tag=lt, essential=is_ess))
            rows_o.append(dict(organism=org, locus_tag=lt, og_id=og,
                               n_paralogs_in_genome=og_count[og] - 1))
        match_rate = len(matched_ess) / max(1, len(ess_names))
        if match_rate < min_match:
            report.append((org, len(pairs), len(matched_ess), match_rate, "skip:low_match"))
            continue
        add_labels += rows_l; add_orth += rows_o
        add_clade.append(dict(organism=org, clade=clade_for(org_species.get(org, org))))
        ess_rate = sum(r["essential"] for r in rows_l) / len(rows_l)
        report.append((org, len(pairs), len(matched_ess), match_rate,
                       f"KEEP clade={add_clade[-1]['clade']} ess={ess_rate:.2f}"))

    # copy originals + append
    def copy_append(fname, fieldnames, new_rows):
        src = src_dir / fname; dst = out_dir / fname
        shutil.copy(src, dst)
        # determine the source header to preserve column order/extra cols
        with open(src) as f:
            hdr = f.readline().strip().split(",")
        with open(dst, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr, extrasaction="ignore")
            for r in new_rows:
                w.writerow(r)
        return hdr

    copy_append("labels.csv", None, add_labels)
    copy_append("orthology_features.csv", None, add_orth)
    copy_append("clade_splits.csv", None, add_clade)
    # carry cooccurrence_features.csv across unchanged (DEG genes simply have no
    # phyletic feature -> phyl channel = 0 for them); keeps the dir self-sufficient
    if (src_dir / "cooccurrence_features.csv").exists():
        shutil.copy(src_dir / "cooccurrence_features.csv", out_dir / "cooccurrence_features.csv")

    # report
    print(f"\n{'org':<34s} {'assigned':>9s} {'ess_matched':>11s} {'match%':>7s}  status")
    print("-" * 90)
    kept = 0
    for org, na, nm, mr, st in sorted(report, key=lambda x: -x[3]):
        if st.startswith("KEEP"):
            kept += 1
        print(f"{org[:34]:<34s} {na:>9d} {nm:>11d} {mr*100:>6.1f}%  {st}")
    print("-" * 90)
    print(f"kept {kept} DEG organisms; +{len(add_labels):,} labelled genes "
          f"({sum(r['essential'] for r in add_labels):,} essential)")
    print(f"wrote augmented CSVs -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="colab/work")
    ap.add_argument("--deg_dir", default="data/drive_import/deg")
    ap.add_argument("--src_dir", default="data/drive_import/labels")
    ap.add_argument("--out_dir", default="data/drive_import/labels_aug")
    ap.add_argument("--min_match", type=float, default=0.30,
                    help="min fraction of DEG-essential gene_names found in proteome")
    ap.add_argument("--min_pident", type=float, default=30.0,
                    help="min mmseqs %identity to accept an OG assignment")
    ap.add_argument("--min_assigned", type=int, default=500,
                    help="min OG-assigned genes for an org to be usable")
    a = ap.parse_args()
    main(a.work, a.deg_dir, a.src_dir, a.out_dir, a.min_match, a.min_pident, a.min_assigned)
