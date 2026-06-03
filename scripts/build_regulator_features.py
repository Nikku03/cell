"""Per-gene regulator / signaling / transporter flags.

Derived from BERIL's essential_families.tsv rep_desc column.

Why: error analysis showed the dominant FP class is conditional-
essentiality genes -- transcriptional regulators, two-component
systems, transporters, conditional biosynthesis. Top wrong OGs:
ynfL, ycfQ, cpxA, gltI, artQ, ntrX, rpoE, narP, phoB, gntR, tsr,
acrA. Right now the model has no feature saying "this is a conditional
regulator." This script tags each gene with three boolean flags so
XGBoost can learn the discount.

  is_regulator    : transcription factor / sigma factor / regulator family
  is_signaling    : two-component, histidine kinase, response regulator,
                      sensor kinase, chemotaxis
  is_transporter  : ABC, MFS, permease, transporter, channel, efflux,
                      porin, exporter, importer
  is_conditional  : OR of the three

NOTE this is the inverse of a fan-out-essential-master-regulator
signal. This feature is meant to PULL DOWN the prediction for the
common dispensable-conditional-regulator class, which is the dominant
FP mode. A separate fan-out feature would pull UP rare essential
master regulators -- different sign, different gene set.

Inputs:
  BERIL/projects/essential_genome/data/essential_families.tsv  (OG -> rep_desc)
  BERIL/projects/essential_genome/data/all_ortholog_groups.csv  ((org, locus) -> OG)

Output:
  memory_bank/data/multiorg_essentiality/regulator_features.csv
"""
from __future__ import annotations
import argparse, csv, re, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BERIL = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "_inventory_cache" / "BERIL"
OUT_CSV   = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "regulator_features.csv"


REGULATOR_PATTERNS = [
    r"transcriptional regulator", r"transcription factor",
    r"sigma factor", r"sigma-factor", r"\bsigma[- ]?[0-9]+",
    r"\blysr\b", r"\bgntr\b", r"\btetr\b", r"\barac\b", r"\bmarr\b",
    r"\bmerr\b", r"\bluxr\b", r"\bdeor\b", r"\bpadr\b", r"\biclr\b",
    r"\brok\b", r"\brpir\b", r"\bfnr\b", r"\barsr\b",
    r"\brepressor\b", r"\bactivator\b", r"\bregulator\b",
    r"helix-?turn-?helix", r"\bhth\b",
]
SIGNALING_PATTERNS = [
    r"two[- ]?component", r"histidine kinase", r"response regulator",
    r"sensor kinase", r"sensor histidine", r"signal transduc",
    r"chemotaxis", r"methyl-?accepting chemotaxis",
    r"\bphosphorelay\b",
]
TRANSPORTER_PATTERNS = [
    r"\babc[- ]?transporter", r"\babc[- ]?type", r"\babc[- ]?family",
    r"\bpermease\b", r"\btransporter\b", r"\bporin\b", r"\befflux\b",
    r"\bchannel\b", r"\bexporter\b", r"\bimporter\b",
    r"\bmfs\b", r"major facilitator", r"\buptake\b",
    r"symporter", r"antiporter", r"\btransport protein\b",
    r"membrane transport", r"\btransport system\b",
]


REGS  = [re.compile(p, re.IGNORECASE) for p in REGULATOR_PATTERNS]
SIGS  = [re.compile(p, re.IGNORECASE) for p in SIGNALING_PATTERNS]
TRANS = [re.compile(p, re.IGNORECASE) for p in TRANSPORTER_PATTERNS]


def classify(desc: str | None) -> tuple[int, int, int]:
    if not isinstance(desc, str) or not desc.strip():
        return 0, 0, 0
    s = desc.lower()
    return (int(any(p.search(s) for p in REGS)),
            int(any(p.search(s) for p in SIGS)),
            int(any(p.search(s) for p in TRANS)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--beril-dir", type=Path, default=DEFAULT_BERIL)
    p.add_argument("--out",       type=Path, default=OUT_CSV)
    args = p.parse_args()

    og_csv  = args.beril_dir / "projects" / "essential_genome" / "data" / "all_ortholog_groups.csv"
    fam_tsv = args.beril_dir / "projects" / "essential_genome" / "data" / "essential_families.tsv"
    for f in (og_csv, fam_tsv):
        if not f.exists():
            print(f"ERROR: missing {f}", file=sys.stderr); return 1

    # ---- OG -> flags ----
    og_flags: dict[str, tuple[int, int, int]] = {}
    n_total = n_reg = n_sig = n_trans = n_any = 0
    sample = []
    with open(fam_tsv) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            og   = r.get("OG_id")
            desc = (r.get("rep_desc") or "")
            if not og: continue
            n_total += 1
            flags = classify(desc)
            og_flags[og] = flags
            n_reg   += flags[0]
            n_sig   += flags[1]
            n_trans += flags[2]
            n_any   += int(any(flags))
            if len(sample) < 8 and any(flags):
                sample.append((og, r.get("rep_gene", "?"), desc[:60], flags))

    print(f"  scanned {n_total} OGs")
    print(f"  is_regulator   : {n_reg:>6d}  ({100*n_reg/max(n_total,1):>5.1f}%)")
    print(f"  is_signaling   : {n_sig:>6d}  ({100*n_sig/max(n_total,1):>5.1f}%)")
    print(f"  is_transporter : {n_trans:>6d}  ({100*n_trans/max(n_total,1):>5.1f}%)")
    print(f"  is_conditional : {n_any:>6d}  ({100*n_any/max(n_total,1):>5.1f}%)")
    print(f"  sample tagged OGs:")
    for og, g, d, f in sample:
        print(f"    {og}  {g:<10s}  reg={f[0]} sig={f[1]} trans={f[2]}  '{d}'")

    # ---- propagate to (org, locus) ----
    rows_out = []
    n_unmatched_og = 0
    with open(og_csv) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            og  = r["OG_id"]; org = r["orgId"]; loc = r["locusId"]
            flags = og_flags.get(og)
            if flags is None:
                n_unmatched_og += 1
                flags = (0, 0, 0)
            rows_out.append({
                "organism":       f"beril_{org}",
                "locus_tag":      loc,
                "og_id":          og,
                "is_regulator":   flags[0],
                "is_signaling":   flags[1],
                "is_transporter": flags[2],
                "is_conditional": int(any(flags)),
            })
    print(f"\n  propagated to {len(rows_out)} (org, locus) rows "
          f"({n_unmatched_og} no OG metadata)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["organism","locus_tag","og_id","is_regulator","is_signaling",
              "is_transporter","is_conditional"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_out: w.writerow(r)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
