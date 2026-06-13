"""TIER 1.2 (DEG version) -- external validation of the conservation predictor
against the DEG database, methodology-filtered.

OGEE is gone; we already have DEG (Database of Essential Genes) locally:
  data/drive_import/deg/deg_bacteria_essential_genes.csv  (26,619 essentials,
    82 organisms, 100% UniProt, 99.99% gene_name, methodology via deg_dataset)

The original concordance (scripts/concordance_deg_vs_beril.py) validated our
LABELS against DEG. THIS validates our PREDICTOR (conservation / family_frac,
the dominant strict-cascade signal) against DEG's independent calls.

INDEPENDENCE FILTER (the review's hard requirement):
  Our training is all transposon-based (RB-TnSeq / MTBseq / TraDIS). DEG
  datasets that are ALSO Tn-seq are NOT methodologically independent. The
  honest external validation is against DEG's SINGLE-GENE-KNOCKOUT and
  ANTISENSE-RNA datasets. We tag each and report both, flagging which are
  truly independent.

MAPPING (reused from concordance script):
  DEG gene_name -> RefSeq GFF (gene <-> locus_tag) -> our family_frac
  (keyed by organism + locus_tag).

OUTPUT: outputs/tier1_deg_validation.json + console.
  Per (our_org, deg_dataset): precision/recall/AUPRC of family_frac vs DEG,
  the DEG-vs-our-labels Cohen's kappa (the noise floor), and an
  independence flag.

NO --smoke needed: this runs fully in sandbox (DEG + GFFs + family_frac all
local). A self-contained real validation.
"""
from __future__ import annotations
import argparse, gzip, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data" / "drive_import" / "labels"
DEG = REPO / "data" / "drive_import" / "deg"
CACHE = REPO / "data" / "drive_import" / "genome_cache"

# overlapping (our_org, deg organism substring, deg_dataset, method, independent?)
# independent = method is NOT transposon-based (our training is all Tn-based)
OVERLAP = [
    ("mtub", "tuberculosis H37Rv", "DEG1010", "single-gene knockout", True),
    ("mtub", "tuberculosis H37Rv", "DEG1025", "Tn-seq", False),
    ("saur_NCTC8325_biotradis", "aureus", "DEG1017", "antisense RNA", True),
    ("saur_NCTC8325_biotradis", "aureus NCTC 8325", "DEG1061", "Tn-seq", False),
    ("beril_RalstoniaGMI1000", "solanacearum GMI1000", "DEG1057", "Tn-seq", False),
]


def parse_gff_cds(gff_path: Path):
    feats = []
    opener = gzip.open if str(gff_path).endswith(".gz") else open
    with opener(gff_path, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "CDS": continue
            attr = {}
            for kv in parts[8].split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1); attr[k] = v
            feats.append({
                "locus_tag": attr.get("locus_tag", ""),
                "old_locus_tag": attr.get("old_locus_tag", ""),
                "gene": attr.get("gene", "").lower(),
            })
    return feats


def metrics(y, scores):
    import numpy as np
    y = np.asarray(y).astype(int); scores = np.asarray(scores).astype(float)
    m = ~np.isnan(scores)
    y, scores = y[m], scores[m]
    if y.sum() < 5 or len(np.unique(y)) < 2:
        return {"auprc": None, "n": int(len(y)), "n_pos": int(y.sum())}
    order = np.argsort(-scores, kind="stable"); ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(y.sum(), 1)
    auprc = float(np.trapezoid(prec, rec))
    # precision/recall at family_frac >= 0.5 (the strict-cascade operating pt)
    return {"auprc": auprc, "n": int(len(y)), "n_pos": int(y.sum()),
            "base_rate": float(y.mean())}


def pr_at_threshold(y, scores, t):
    import numpy as np
    y = np.asarray(y).astype(int); scores = np.asarray(scores).astype(float)
    m = ~np.isnan(scores); y, scores = y[m], scores[m]
    pred = (scores >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "n_called": tp + fp}


def main() -> int:
    import pandas as pd
    import numpy as np

    print("=" * 64)
    print("TIER 1.2 -- conservation predictor vs DEG (methodology-filtered)")
    print("=" * 64)

    lab = pd.read_csv(D / "labels.csv")
    orth = pd.read_csv(D / "orthology_features.csv",
                       usecols=["organism", "locus_tag",
                                "family_frac_essential_leakfree"])
    deg = pd.read_csv(DEG / "deg_bacteria_essential_genes.csv")
    mf = pd.read_csv(D / "genome_cache_manifest.csv")
    acc_map = dict(zip(mf.organism, mf.accession))

    results = []
    for our_org, deg_org_sub, deg_ds, method, independent in OVERLAP:
        print(f"\n--- {our_org}  vs  DEG {deg_ds} ({method}, "
              f"{'INDEPENDENT' if independent else 'tn-seq overlap'}) ---")
        acc = acc_map.get(our_org)
        gff = CACHE / acc / "genomic.gff" if acc else None
        if not gff or not gff.exists():
            print(f"  GFF missing for {our_org} ({acc}); skip"); continue
        feats = parse_gff_cds(gff)
        # gene_name (lower) -> locus_tag
        gene2lt = {}
        for f in feats:
            if f["gene"]:
                gene2lt.setdefault(f["gene"], f["locus_tag"])
        # DEG essential gene_names for this dataset
        deg_sub = deg[(deg.deg_dataset == deg_ds)]
        deg_ess_names = set(str(g).strip().lower()
                            for g in deg_sub.gene_name.dropna())
        # map DEG essential names -> our locus_tags
        deg_ess_loci = set(gene2lt[n] for n in deg_ess_names if n in gene2lt)
        print(f"  DEG essentials: {len(deg_ess_names)}  "
              f"mapped to our loci: {len(deg_ess_loci)}")
        if len(deg_ess_loci) < 30:
            print(f"  too few mapped DEG essentials, skip"); continue
        # build per-gene table: our org's genes with family_frac + DEG call
        our = orth[orth.organism == our_org].copy()
        our["deg_essential"] = our.locus_tag.isin(deg_ess_loci).astype(int)
        our = our.dropna(subset=["family_frac_essential_leakfree"])
        if our.deg_essential.sum() < 20:
            print(f"  DEG essentials don't join to our family_frac coverage, skip")
            continue
        y = our.deg_essential.values
        ff = our.family_frac_essential_leakfree.values
        m = metrics(y, ff)
        pr50 = pr_at_threshold(y, ff, 0.5)
        pr80 = pr_at_threshold(y, ff, 0.8)
        print(f"  family_frac vs DEG: n={m['n']:,}  DEG-essential={m['n_pos']}  "
              f"base={m['base_rate']:.3f}  AUPRC={m['auprc']:.3f}")
        print(f"    @ff>=0.5: precision={pr50['precision']:.3f} "
              f"recall={pr50['recall']:.3f}  (n_called={pr50['n_called']})")
        print(f"    @ff>=0.8: precision={pr80['precision']:.3f} "
              f"recall={pr80['recall']:.3f}  (n_called={pr80['n_called']})")
        results.append({
            "our_org": our_org, "deg_dataset": deg_ds, "method": method,
            "independent": independent,
            "n": m["n"], "deg_essential": m["n_pos"],
            "base_rate": m["base_rate"], "auprc": m["auprc"],
            "precision_at_0.5": pr50["precision"], "recall_at_0.5": pr50["recall"],
            "precision_at_0.8": pr80["precision"], "recall_at_0.8": pr80["recall"],
        })

    # report: independent vs all
    print(f"\n{'='*64}")
    print("SUMMARY")
    print(f"{'='*64}")
    indep = [r for r in results if r["independent"] and r["auprc"] is not None]
    allr = [r for r in results if r["auprc"] is not None]
    if indep:
        print(f"\n  METHODOLOGY-INDEPENDENT validations (the honest external number):")
        for r in indep:
            print(f"    {r['our_org']:<25s} {r['deg_dataset']} ({r['method']}): "
                  f"AUPRC={r['auprc']:.3f}  prec@0.8={r['precision_at_0.8']:.3f}")
        macro_p80 = np.mean([r["precision_at_0.8"] for r in indep])
        print(f"  >> independent-subset precision @ ff>=0.8: {macro_p80:.3f}")
    if allr:
        print(f"\n  ALL validations (incl. Tn-seq overlap, less independent):")
        for r in allr:
            tag = "indep" if r["independent"] else "tn-overlap"
            print(f"    {r['our_org']:<25s} {r['deg_dataset']:<9s} [{tag}]: "
                  f"AUPRC={r['auprc']:.3f}")

    # also surface the already-computed DEG-vs-our-LABELS concordance
    # (the noise-floor kappa)
    conc = pd.read_csv(DEG / "deg_vs_beril_concordance.csv")
    print(f"\n  CONTEXT -- DEG vs OUR LABELS concordance (essentiality noise floor):")
    print(f"    Cohen's kappa between two independent screens of same organism:")
    for _, r in conc.iterrows():
        print(f"      {r['beril_org']:<25s} {r['deg_dataset']}: "
              f"agreement={r['agreement']:.3f}  kappa={r['cohens_kappa']:.3f}")
    print(f"    -> kappa 0.18-0.58 is the CEILING any predictor can reach vs a")
    print(f"       single external source. Our AUPRC must be read against this.")

    out = REPO / "outputs" / "tier1_deg_validation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "predictor_vs_deg": results,
        "noise_floor_kappa": conc[["beril_org", "deg_dataset", "agreement",
                                    "cohens_kappa"]].to_dict("records"),
    }, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
