"""DEG vs BERIL essential-call concordance for same-strain pairs.

For each (BERIL_organism, DEG_dataset, RefSeq assembly) triple where
both labs measured the same strain, ask: do they agree on which genes
are essential?

Bridge: DEG -> RefSeq GFF -> BERIL
  - DEG gives gene_name (e.g. "dnaA")
  - RefSeq GFF gives gene_name <-> locus_tag <-> old_locus_tag
  - BERIL labels.csv uses RefSeq locus_tag (or old_locus_tag) for most
    orgs (verified earlier in this session via the codon-features run)

For each BERIL-labelled gene in the test set:
  call_beril : labels.csv essential (0/1)
  call_deg   : 1 if the gene matches any DEG essential record (via
                 gene_name OR locus_tag), else 0  -- assumes DEG
                 screened the whole genome and listed all essentials.

Confusion: (call_beril, call_deg) in {(0,0),(0,1),(1,0),(1,1)}.
Metrics: agreement %, Cohen's kappa, sensitivity/specificity treating
DEG as the comparison reference (and vice versa -- symmetric).

Skipped:
  - beril_Keio: BERIL uses MicrobesOnline numeric IDs; no clean bridge
    to b-numbers / MG1655 RefSeq locus_tags. (We'd need BERIL's source
    sysName mapping table which isn't in the repo we have.)
  - beril_Putida: no P. putida dataset in DEG (DEG has P. aeruginosa
    only).

Outputs:
  deg_vs_beril_concordance.csv     -- per-pair metrics + sample sizes
  deg_vs_beril_disagreements.csv   -- gene-level rows where the two
                                       calls disagree (for inspection)
  deg_vs_beril_concordance.json    -- full structured report
"""
from __future__ import annotations
import argparse, csv, gzip, json, math, sys, urllib.request, urllib.error
from pathlib import Path

# (beril_organism, deg_dataset, refseq_acc, assembly_name, note)
PAIRS = [
    ("mtub",                    "DEG1010", "GCF_000195955.2", "ASM19595v2",
        "M.tb H37Rv"),
    ("mtub",                    "DEG1025", "GCF_000195955.2", "ASM19595v2",
        "M.tb H37Rv II"),
    ("mtub",                    "DEG1027", "GCF_000195955.2", "ASM19595v2",
        "M.tb H37Rv III"),
    ("saur_NCTC8325_biotradis", "DEG1017", "GCF_000013425.1", "ASM1342v1",
        "S.aureus NCTC8325"),
    ("saur_NCTC8325_biotradis", "DEG1061", "GCF_000013425.1", "ASM1342v1",
        "S.aureus NCTC8325 v2"),
    ("beril_RalstoniaGMI1000",  "DEG1057", "GCF_000009125.1", "ASM912v1",
        "R.solanacearum GMI1000"),
    ("ecoli_BW25113_tradis",    "DEG1018", "GCF_000750555.1", "ASM75055v1",
        "E.coli MG1655 I vs BW25113 (cross-strain)"),
    ("ecoli_BW25113_tradis",    "DEG1019", "GCF_000750555.1", "ASM75055v1",
        "E.coli MG1655 II vs BW25113 (cross-strain)"),
    ("beril_Btheta",            "DEG1023", "GCF_000011065.1", "ASM1106v1",
        "B.theta VPI-5482"),
    ("beril_MR1",               "DEG1029", "GCF_000146165.2", "ASM14616v2",
        "Shewanella MR-1"),
    ("beril_Caulo",             "DEG1020", "GCF_000006905.1", "ASM690v1",
        "Caulobacter CB15"),
    ("beril_SynE",              "DEG1040", "GCF_000012525.1", "ASM1252v1",
        "S.elongatus PCC 7942"),
]

NCBI_BASE = "https://ftp.ncbi.nlm.nih.gov/genomes/all"


def ncbi_gff_url(accession: str, assembly: str) -> str:
    a, asm = accession, assembly
    prefix = a.split("_")[0]; digits = a.split("_")[1]
    p1, p2, p3 = digits[:3], digits[3:6], digits[6:9]
    return (f"{NCBI_BASE}/{prefix}/{p1}/{p2}/{p3}/"
            f"{a}_{asm}/{a}_{asm}_genomic.gff.gz")


def download_cached(url: str, dst: Path) -> Path | None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 100: return dst
    try:
        urllib.request.urlretrieve(url, dst); return dst
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"    DOWNLOAD FAIL: {url} -> {e}")
        if dst.exists(): dst.unlink()
        return None


def parse_gff_cds(gff_path: Path) -> list[dict]:
    feats = []
    with gzip.open(gff_path, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "CDS": continue
            attr = {}
            for kv in parts[8].split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1); attr[k] = v
            feats.append({
                "locus_tag":     attr.get("locus_tag", ""),
                "old_locus_tag": attr.get("old_locus_tag", ""),
                "gene":          attr.get("gene", "").lower(),
            })
    return feats


def cohens_kappa(b00, b01, b10, b11) -> float:
    n = b00 + b01 + b10 + b11
    if n == 0: return float("nan")
    po = (b00 + b11) / n
    p_b1 = (b10 + b11) / n; p_d1 = (b01 + b11) / n
    pe = p_b1*p_d1 + (1-p_b1)*(1-p_d1)
    if abs(1 - pe) < 1e-12: return float("nan")
    return (po - pe) / (1 - pe)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--deg-csv",    type=Path, required=True,
                   help="deg_bacteria_essential_genes.csv from parse_deg.py")
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--cache-dir",  type=Path, required=True,
                   help="dir to cache RefSeq GFFs in (e.g. on Drive)")
    p.add_argument("--out-dir",    type=Path, required=True)
    args = p.parse_args()

    try:
        import pandas as pd
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1

    print(f"Loading inputs ...")
    deg = pd.read_csv(args.deg_csv)
    labels = pd.read_csv(args.labels_csv)
    print(f"  DEG bacterial essential rows: {len(deg):,}")
    print(f"  labels rows:                  {len(labels):,}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    per_pair: list[dict] = []
    disagreements: list[dict] = []

    for beril_org, deg_dataset, acc, asm, note in PAIRS:
        print(f"\n=== {beril_org} <-> {deg_dataset}  ({note}) ===")
        lab = labels[labels.organism == beril_org]
        deg_sub = deg[deg.deg_dataset == deg_dataset]
        print(f"  BERIL labels: {len(lab):,}   DEG essentials: {len(deg_sub):,}")
        if len(lab) == 0 or len(deg_sub) == 0:
            print(f"  -- skip (empty)"); continue

        # Download GFF
        gff_path = args.cache_dir / f"{acc}_genomic.gff.gz"
        if not download_cached(ncbi_gff_url(acc, asm), gff_path):
            print(f"  -- skip (no GFF)"); continue
        feats = parse_gff_cds(gff_path)
        print(f"  GFF CDS features: {len(feats):,}")

        # Build BERIL essential-call map
        beril_essential = dict(zip(
            lab.locus_tag.astype(str).str.strip(),
            lab.essential.astype(int)))
        # Build DEG essential-name set (lowercased gene_names)
        deg_essential_names = set(
            str(g).strip().lower() for g in deg_sub.gene_name.dropna()
            if str(g).strip() and str(g) != "-")
        # Also any RefSeq locus_tag-like identifiers DEG includes
        # (some DEG records have a refseq locus_tag in gene_name; safe to include)
        deg_lt_candidates = deg_essential_names.copy()
        # And any DEG gi numbers (rare bridge)
        print(f"  DEG essential unique gene_names: {len(deg_essential_names):,}")

        # For each RefSeq CDS:
        #   resolve a BERIL locus_tag (lt, old_lt, or via gene name)
        #   resolve a DEG essential call (gene name match)
        b00=b01=b10=b11=0
        unmatched_no_beril = 0
        for f in feats:
            beril_call = None
            for key in (f["locus_tag"], f["old_locus_tag"]):
                if key and key in beril_essential:
                    beril_call = beril_essential[key]; break
            if beril_call is None:
                unmatched_no_beril += 1; continue
            deg_call = 1 if (f["gene"] and f["gene"] in deg_essential_names) else 0
            if   beril_call==0 and deg_call==0: b00 += 1
            elif beril_call==0 and deg_call==1:
                b01 += 1
                disagreements.append({
                    "pair":          f"{beril_org}/{deg_dataset}",
                    "locus_tag":     f["locus_tag"], "gene": f["gene"],
                    "beril_call": 0, "deg_call": 1,
                })
            elif beril_call==1 and deg_call==0:
                b10 += 1
                disagreements.append({
                    "pair":          f"{beril_org}/{deg_dataset}",
                    "locus_tag":     f["locus_tag"], "gene": f["gene"],
                    "beril_call": 1, "deg_call": 0,
                })
            else: b11 += 1

        n_eval = b00 + b01 + b10 + b11
        if n_eval == 0:
            print(f"  -- skip (no genes joinable)"); continue
        agree = (b00 + b11) / n_eval
        kappa = cohens_kappa(b00, b01, b10, b11)
        # treat DEG as reference for sens/spec
        tp_deg = b11; fp_deg = b10; tn_deg = b00; fn_deg = b01
        sens   = tp_deg / max(tp_deg + fn_deg, 1)
        spec   = tn_deg / max(tn_deg + fp_deg, 1)
        # jaccard on essential set
        union = b01 + b10 + b11
        jacc  = b11 / max(union, 1)

        pair_rec = {
            "beril_org": beril_org, "deg_dataset": deg_dataset, "note": note,
            "n_evaluable_genes":       n_eval,
            "n_beril_essential":       b10 + b11,
            "n_deg_essential":         b01 + b11,
            "both_essential":          b11,
            "both_non_essential":      b00,
            "beril_only_essential":    b10,
            "deg_only_essential":      b01,
            "agreement":               round(agree, 4),
            "cohens_kappa":            round(kappa, 4) if kappa==kappa else None,
            "jaccard_essential":       round(jacc, 4),
            "sensitivity_deg_as_ref":  round(sens, 4),
            "specificity_deg_as_ref":  round(spec, 4),
            "unmatched_cds_no_beril":  unmatched_no_beril,
        }
        per_pair.append(pair_rec)
        print(f"  evaluable: {n_eval:,}  (BERIL_ess={b10+b11}, DEG_ess={b01+b11}, "
              f"both={b11})")
        print(f"  agreement = {agree*100:5.1f}%   kappa = "
              f"{kappa:+.3f}   jaccard = {jacc:.3f}")
        print(f"  disagreement: BERIL-only={b10}  DEG-only={b01}")

    if not per_pair:
        print("ERROR: no pairs produced concordance", file=sys.stderr); return 1

    df = pd.DataFrame(per_pair)
    df.to_csv(args.out_dir / "deg_vs_beril_concordance.csv", index=False)
    print(f"\nwrote deg_vs_beril_concordance.csv  ({len(df)} pairs)")

    dis = pd.DataFrame(disagreements)
    dis.to_csv(args.out_dir / "deg_vs_beril_disagreements.csv", index=False)
    print(f"wrote deg_vs_beril_disagreements.csv  ({len(dis):,} rows)")

    # ---- aggregate summary ----
    print(f"\n=== HEADLINE per-pair ===")
    print(f"{'pair':<48s} {'n':>7s} {'agree':>7s} {'kappa':>7s} {'B-only':>7s} {'D-only':>7s}")
    for r in per_pair:
        pair = f"{r['beril_org']}/{r['deg_dataset']}"
        k = r["cohens_kappa"]
        print(f"  {pair:<46s} {r['n_evaluable_genes']:>7d} "
              f"{r['agreement']*100:>6.1f}% "
              f"{k if k is not None else 0:>+7.3f} "
              f"{r['beril_only_essential']:>7d} {r['deg_only_essential']:>7d}")

    # Macro-average kappa weighted by n_evaluable
    weights = [r["n_evaluable_genes"] for r in per_pair]
    kappas  = [r["cohens_kappa"] for r in per_pair if r["cohens_kappa"] is not None]
    if kappas:
        wk = sum(k*w for k,w,r in zip([p["cohens_kappa"] for p in per_pair],
                                          weights, per_pair)
                  if k is not None) / sum(w for k,w,r in zip(
                      [p["cohens_kappa"] for p in per_pair], weights, per_pair)
                      if k is not None)
    else:
        wk = None
    print(f"\n  weighted mean kappa (across pairs): "
          f"{wk:+.3f}" if wk is not None else "  weighted kappa: undefined")

    out = {
        "pairs": per_pair,
        "weighted_mean_kappa": wk,
        "n_pairs": len(per_pair),
        "n_disagreements_total": len(dis),
    }
    (args.out_dir / "deg_vs_beril_concordance.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nwrote deg_vs_beril_concordance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
