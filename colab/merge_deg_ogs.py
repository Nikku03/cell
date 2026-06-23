"""Merge DEG orthogroup assignments (from assign_ogs.py / mmseqs on Colab) into
an augmented orthology_features table + a self-contained labels_aug/ directory
that build_cache.py can consume directly.

Input  (produced on Colab):
  colab/work/og_assignments.csv     organism, locus_tag, og_id, pident
Existing:
  data/drive_import/labels/orthology_features.csv   (48 beril orgs, full schema)
  data/drive_import/labels/labels.csv               (all 59 orgs incl. DEG labels)
  data/drive_import/labels/clade_splits.csv         (incl. DEG clades)

Output (build_cache.py --labels_dir data/drive_import/labels_aug):
  data/drive_import/labels_aug/labels.csv
  data/drive_import/labels_aug/clade_splits.csv
  data/drive_import/labels_aug/orthology_features.csv   (union, stats recomputed)

Runs in the sandbox (pure CSV). Safe to run before Colab with --synthetic to
self-test the plumbing on fake DEG assignments.
"""
import csv, argparse, random
from collections import defaultdict, Counter
from pathlib import Path

LAB = Path("data/drive_import/labels")
AUG = Path("data/drive_import/labels_aug")
DEG_ORGS = ["mtub", "syn3a", "mgen", "saur_NCTC8325_biotradis", "spneT4"]
SCHEMA = ["organism", "locus_tag", "og_id", "own_fold", "n_paralogs_in_genome",
          "family_size_total", "family_n_organisms", "is_orphan",
          "family_frac_essential_global", "family_frac_essential_leakfree",
          "family_frac_essential_fold0", "family_frac_essential_fold1",
          "family_frac_essential_fold2", "family_frac_essential_fold3",
          "family_frac_essential_fold4"]


def load_labels():
    ess = {}
    for r in csv.DictReader(open(LAB / "labels.csv")):
        ess[(r["organism"], r["locus_tag"])] = int(r["essential"])
    return ess


def load_clade_fold():
    cl = {}; fold = {}
    for r in csv.DictReader(open(LAB / "clade_splits.csv")):
        cl[r["organism"]] = r["clade"]
        fold[r["organism"]] = int(r.get("fold") or 0)
    return cl, fold


def make_synthetic(ess):
    """fake DEG og_assignments: map ~70% of each DEG org's genes into existing
    OGs (random), to exercise the merge before the real Colab run."""
    existing_ogs = sorted({r["og_id"] for r in
                           csv.DictReader(open(LAB / "orthology_features.csv"))
                           if r.get("og_id")})
    rng = random.Random(0); rows = []
    for org in DEG_ORGS:
        lts = [lt for (o, lt) in ess if o == org]
        for lt in lts:
            if rng.random() < 0.70:
                rows.append(dict(organism=org, locus_tag=lt,
                                 og_id=rng.choice(existing_ogs), pident=85.0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignments", default="colab/work/og_assignments.csv")
    ap.add_argument("--synthetic", action="store_true",
                    help="self-test with fake DEG assignments (no Colab needed)")
    args = ap.parse_args()

    ess = load_labels(); clade, fold = load_clade_fold()
    AUG.mkdir(parents=True, exist_ok=True)

    # 1. existing orthology rows (keep verbatim)
    existing = list(csv.DictReader(open(LAB / "orthology_features.csv")))
    have = {(r["organism"], r["locus_tag"]) for r in existing}

    # 2. DEG assignments
    if args.synthetic:
        deg = make_synthetic(ess)
        print(f"[synthetic] generated {len(deg)} fake DEG OG assignments")
    else:
        deg = [dict(organism=r["organism"], locus_tag=r["locus_tag"],
                    og_id=r.get("og_id", ""), pident=r.get("pident", ""))
               for r in csv.DictReader(open(args.assignments))]
        print(f"loaded {len(deg)} DEG OG assignments from {args.assignments}")

    deg_assigned = {(d["organism"], d["locus_tag"]): d["og_id"]
                    for d in deg if d.get("og_id")}

    # 3. ALL DEG genes from labels (assigned or orphan)
    deg_genes = [(o, lt) for (o, lt) in ess if o in DEG_ORGS]
    print(f"DEG genes in labels: {len(deg_genes)}  "
          f"(assigned an OG: {sum(1 for k in deg_genes if k in deg_assigned)})")

    # 4. recompute family stats over the UNION (existing + DEG)
    og_members = defaultdict(list)       # og -> [(org,lt)]
    for r in existing:
        if r.get("og_id"):
            og_members[r["og_id"]].append((r["organism"], r["locus_tag"]))
    for k, og in deg_assigned.items():
        og_members[og].append(k)
    og_orgs = {og: {o for (o, _) in m} for og, m in og_members.items()}
    og_essrate = {}
    for og, m in og_members.items():
        labelled = [ess[k] for k in m if k in ess]
        og_essrate[og] = (sum(labelled) / len(labelled)) if labelled else 0.0
    # per-org paralog counts
    par = defaultdict(int)
    for og, m in og_members.items():
        c = Counter(o for (o, _) in m)
        for (o, lt) in m:
            par[(o, lt)] = c[o] - 1

    # 5. emit augmented rows: existing (refreshed family stats) + DEG
    def row_for(org, lt, og):
        if og:
            return dict(organism=org, locus_tag=lt, og_id=og,
                        own_fold=fold.get(org, 0),
                        n_paralogs_in_genome=par[(org, lt)],
                        family_size_total=len(og_members[og]),
                        family_n_organisms=len(og_orgs[og]),
                        is_orphan=0,
                        family_frac_essential_global=round(og_essrate[og], 4),
                        family_frac_essential_leakfree=round(og_essrate[og], 4),
                        **{f"family_frac_essential_fold{i}": round(og_essrate[og], 4)
                           for i in range(5)})
        return dict(organism=org, locus_tag=lt, og_id="", own_fold=fold.get(org, 0),
                    n_paralogs_in_genome=0, family_size_total=0,
                    family_n_organisms=0, is_orphan=1,
                    **{c: 0.0 for c in SCHEMA[8:]})

    out_rows = []
    for r in existing:
        out_rows.append(row_for(r["organism"], r["locus_tag"], r.get("og_id", "")))
    for (o, lt) in deg_genes:
        if (o, lt) in have:
            continue
        out_rows.append(row_for(o, lt, deg_assigned.get((o, lt), "")))

    with open(AUG / "orthology_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA); w.writeheader()
        w.writerows(out_rows)
    # labels + clade_splits into the same dir so build_cache is self-contained
    for fn in ("labels.csv", "clade_splits.csv"):
        (AUG / fn).write_text((LAB / fn).read_text())

    n_deg_rows = sum(1 for r in out_rows if r["organism"] in DEG_ORGS)
    n_deg_og = sum(1 for r in out_rows if r["organism"] in DEG_ORGS and r["is_orphan"] == 0)
    print(f"\nwrote {AUG}/orthology_features.csv  ({len(out_rows)} rows)")
    print(f"  DEG rows: {n_deg_rows}  with OG: {n_deg_og}  "
          f"orphan: {n_deg_rows - n_deg_og}")
    print(f"wrote {AUG}/labels.csv + clade_splits.csv")
    print(f"\nnext: python colab/build_cache.py --labels_dir {AUG} "
          f"--out outputs/orphan/af_msa_cache_aug2.npz --include_orphans")


if __name__ == "__main__":
    main()
