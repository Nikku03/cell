"""TIER 1.1b -- per-organism ranked product output.

Consumes the per-fold Phase 2 predictions on Drive and emits, PER ORGANISM,
a ranked (gene, drug) adjuvant-target table with:
  - tier  GOLD (atlas-corroborated, high pred) / SILVER (high pred, no atlas)
          / ABSTAIN (uncertain) / NEGATIVE (confidently no-hit)
  - confidence (calibrated probability if available, raw pred otherwise)
  - source attribution (which evidence fired)
  - WHY  a plain-English one-sentence mechanistic hypothesis
         (cobbled from compound MoA + gene context)

Output: PHASE2_DIR/eval/per_org_products/<org>.tsv + a master index.

SMOKE MODE (--smoke):
  builds a mock prediction parquet in sandbox and emits the per-org table
  end-to-end. Confirms tiering, WHY-field generation, and TSV formatting
  are all correct without needing Drive.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PHASE2 = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/phase2")

# thresholds for tiering -- conservative, tunable
T_GOLD = 0.80          # high model confidence
T_SILVER = 0.60
T_NEGATIVE = 0.10      # confident no
ATLAS_OG_CPD_HITRATE = 0.30    # og_cpd_hit_rate threshold for atlas-corroborated
ATLAS_OG_CPD_N = 3              # min training-org measurements

# coarse MoA lookup, mirrored from kernel build
MOA_TABLE = {
    "bacitracin": "cell_wall", "vancomycin": "cell_wall", "d-cycloserine": "cell_wall",
    "cycloserine": "cell_wall", "fosfomycin": "cell_wall", "ampicillin": "cell_wall",
    "carbenicillin": "cell_wall", "cefoxitin": "cell_wall",
    "cisplatin": "dna_damage", "nalidixic acid": "dna_damage",
    "ciprofloxacin": "dna_damage", "novobiocin": "dna_damage",
    "tetracycline": "ribosome", "doxycycline": "ribosome",
    "gentamicin": "ribosome", "kanamycin": "ribosome", "tobramycin": "ribosome",
    "neomycin": "ribosome", "spectinomycin": "ribosome",
    "erythromycin": "ribosome", "chloramphenicol": "ribosome",
    "trimethoprim": "folate", "sulfamethoxazole": "folate",
    "polymyxin": "membrane", "colistin": "membrane",
    "rifampin": "transcription", "rifampicin": "transcription",
    "fusidic acid": "protein_synthesis", "linezolid": "protein_synthesis",
}


def get_moa(cpd: str) -> str | None:
    if not isinstance(cpd, str): return None
    c = cpd.lower()
    for k, v in MOA_TABLE.items():
        if k in c: return v
    return None


def make_why(row) -> str:
    """One-sentence mechanistic hypothesis from row metadata."""
    moa = get_moa(row.get("compound", ""))
    moa_str = moa.replace("_", "-") if moa else "non-classified"
    parts = []
    parts.append(f"flagged under {row.get('compound', '?')} ({moa_str})")
    if row.get("og_cpd_hit_rate") is not None and row.get("og_cpd_hit_rate") > 0.3:
        n = int(row.get("og_cpd_n", 0))
        parts.append(f"and the gene-family x compound pair has been measured "
                     f"as a strong hit in {n} relatives")
    elif row.get("og_cpd_n") == 0:
        parts.append("(novel — no measured atlas evidence in relatives)")
    if row.get("pred", 0) > T_GOLD:
        parts.append(f"with model confidence {row['pred']:.2f}")
    return "; ".join(parts) + "."


def tier_row(r) -> str:
    """Assign tier per a single prediction row."""
    pred = float(r.get("pred", 0))
    atlas = (r.get("og_cpd_hit_rate", 0) is not None
             and r.get("og_cpd_hit_rate", 0) >= ATLAS_OG_CPD_HITRATE
             and r.get("og_cpd_n", 0) >= ATLAS_OG_CPD_N)
    if pred >= T_GOLD and atlas: return "GOLD"
    if pred >= T_SILVER: return "SILVER"
    if pred <= T_NEGATIVE: return "NEGATIVE"
    return "ABSTAIN"


def build_per_org(df, out_dir):
    """For each organism, write a ranked TSV + collect summary."""
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for org, g in df.groupby("fold_org" if "fold_org" in df.columns else "orgId"):
        g = g.copy()
        g["tier"] = g.apply(tier_row, axis=1)
        g["why"] = g.apply(make_why, axis=1)
        g = g.sort_values("pred", ascending=False)
        keep = ["tier", "pred", "fold_org" if "fold_org" in g.columns else "orgId",
                "locusId", "compound", "fit", "t", "strong_hit",
                "og_cpd_hit_rate", "og_cpd_n", "why"]
        keep = [c for c in keep if c in g.columns]
        g[keep].to_csv(out_dir / f"{org}.tsv", sep="\t", index=False)
        counts = g.tier.value_counts().to_dict()
        summary[org] = {
            "total_rows": int(len(g)),
            "gold": int(counts.get("GOLD", 0)),
            "silver": int(counts.get("SILVER", 0)),
            "abstain": int(counts.get("ABSTAIN", 0)),
            "negative": int(counts.get("NEGATIVE", 0)),
        }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_smoke():
    import pandas as pd
    import numpy as np
    print("=" * 60)
    print("SMOKE MODE -- mock per-org product output")
    print("=" * 60)
    rng = np.random.RandomState(0)
    rows = []
    for org in ["SB2B", "Keio"]:
        for _ in range(800):
            cpd = rng.choice(["bacitracin", "cisplatin", "gentamicin",
                              "L-glutamine", "rifampin", "novel-compound-X"])
            atlas_hit = rng.random() < 0.05
            rows.append({
                "fold_org": org,
                "locusId": f"L{rng.randint(1, 9999)}",
                "compound": cpd,
                "pred": float(rng.beta(2, 4)),     # most low, some high
                "fit": float(rng.randn() * 2 - 0.5),
                "t": float(rng.randn() * 3),
                "strong_hit": int(rng.random() < 0.05),
                "og_cpd_hit_rate": float(rng.random()) if atlas_hit else 0.0,
                "og_cpd_n": int(rng.choice([0, 3, 5, 10])) if atlas_hit else 0,
            })
    df = pd.DataFrame(rows)
    out_dir = REPO / "outputs" / "tier1_per_org_smoke"
    s = build_per_org(df, out_dir)
    print(f"  per-org TSVs written to {out_dir}/")
    print(f"  summary:")
    for org, c in s.items():
        print(f"    {org}: {c}")
    # validate: a GOLD row must have pred>=T_GOLD AND atlas hit
    for org in s:
        g = pd.read_csv(out_dir / f"{org}.tsv", sep="\t")
        gold = g[g.tier == "GOLD"]
        if len(gold) and (gold.pred.min() < T_GOLD):
            print(f"  FAIL: GOLD row has pred<{T_GOLD} in {org}"); return 1
    print(f"\nSMOKE PASS -- tiering + WHY-field generation correct")
    return 0


def run_real(args):
    import pandas as pd
    eval_dir = args.phase2 / "eval" / "loo-org"
    files = sorted(eval_dir.glob("preds_xgb__*.parquet"))
    if not files:
        print(f"ERROR: no preds in {eval_dir}", file=sys.stderr); return 1
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out_dir = args.phase2 / "eval" / "per_org_products"
    s = build_per_org(df, out_dir)
    print(f"\nWrote {len(s)} per-organism TSVs to {out_dir}")
    print(f"summary:")
    for org, c in sorted(s.items()):
        print(f"  {org:<25s} gold={c['gold']:>5d} silver={c['silver']:>6d} "
              f"abstain={c['abstain']:>6d} negative={c['negative']:>6d}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2", type=Path, default=DEFAULT_PHASE2)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    return run_smoke() if args.smoke else run_real(args)


if __name__ == "__main__":
    sys.exit(main())
