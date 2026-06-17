"""PATH-ORPHAN, STEP 3 -- DEG1057 cross-grade (independent screen).

The atlas predictions were trained on the BERIL/FB RB-TnSeq labels. DEG1057 is
an INDEPENDENT essentiality screen of R. solanacearum GMI1000 (465 genes,
distinct methodology). Grading rogue-zone predictions against DEG gives an
honest precision number that cannot be circular -- our model didn't see DEG.

LINKING: DEG keys genes by RefSeq protein_id (WP_*) embedded in the `gi` column
(e.g. "GeneID:1221466; Protein_ID: WP_011002530.1"). Our bridge already has
protein_id per locus_tag -> direct join.

KILL-GATE: rogue-zone predictions should beat their own base rate in DEG.

--smoke : validate WP_ extraction + precision/recall logic on synthetic data
--real  : grade atlas predictions on outputs/orphan/atlas_<org>.parquet
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "drive_import"
OUT = REPO / "outputs" / "orphan"
ROGUE_CONS = 0.10
WP_RE = re.compile(r'WP_\d+\.\d+')


def extract_wp(s):
    """Pull RefSeq WP_ protein_id out of DEG's 'gi' column."""
    if not s:
        return None
    m = WP_RE.search(str(s))
    return m.group(0) if m else None


def precision_recall(predicted_set, truth_set, universe):
    """precision/recall of a predicted essential set vs DEG truth, evaluated on
    the genes that appear in both atlas universe AND have a DEG record."""
    pred = predicted_set & universe
    tru = truth_set & universe
    if not pred:
        return {"n_predicted": 0, "n_truth": len(tru), "tp": 0,
                "precision": float("nan"), "recall": 0.0}
    tp = pred & tru
    return {"n_predicted": len(pred), "n_truth": len(tru), "tp": len(tp),
            "precision": len(tp) / len(pred),
            "recall": len(tp) / max(1, len(tru))}


def load_deg_essential(deg_dataset="DEG1057"):
    """Return set of WP_ protein_ids essential per DEG."""
    import csv
    out = set()
    with open(DATA / "deg" / "deg_bacteria_essential_genes.csv") as f:
        for r in csv.DictReader(f):
            if r["deg_dataset"] == deg_dataset:
                wp = extract_wp(r.get("gi", ""))
                if wp:
                    out.add(wp)
    return out


def run_real(args):
    import pandas as pd
    atlas = OUT / f"atlas_{args.org}.parquet"
    bridge = OUT / f"bridge_{args.org}.parquet"
    if not atlas.exists() or not bridge.exists():
        print("ERROR: run orphan_atlas.py --real first"); return 2
    a = pd.read_parquet(atlas)
    b = pd.read_parquet(bridge)[["locus_tag", "protein_id"]]
    df = a.merge(b, on="locus_tag", how="left").rename(
        columns={"protein_id": "wp_id"})
    deg = load_deg_essential(args.deg_dataset)
    print("=" * 70)
    print(f"DEG cross-grade -- {args.org} vs {args.deg_dataset}")
    print("=" * 70)
    print(f"  DEG essential genes (WP_): {len(deg):,}")
    # universe: genes in our atlas AND have a WP_ that could appear in DEG
    universe_wp = set(df["wp_id"].dropna())
    deg_in_universe = deg & universe_wp
    print(f"  atlas-universe WP overlap with DEG: {len(deg_in_universe):,} "
          f"({100*len(deg_in_universe)/max(1,len(deg)):.0f}%)")

    # 1) overall: our model's predicted essentials (essential==1) vs DEG truth
    pred_ess_wp = set(df[df.essential == 1]["wp_id"].dropna())
    overall = precision_recall(pred_ess_wp, deg, universe_wp)
    print(f"\n  OVERALL  predicted_ess={overall['n_predicted']:>5}  "
          f"DEG-truth={overall['n_truth']:>3}  TP={overall['tp']:>3}  "
          f"P={overall['precision']:.3f}  R={overall['recall']:.3f}")

    # 2) ROGUE-ZONE: atlas's predicted essentials in the rogue zone (cons<0.1)
    rogue_pred = set(df[(df.essential == 1) & (df.conservation < ROGUE_CONS)]["wp_id"].dropna())
    rogue_universe = set(df[df.conservation < ROGUE_CONS]["wp_id"].dropna())
    rogue = precision_recall(rogue_pred, deg, rogue_universe)
    print(f"  ROGUE    predicted_ess={rogue['n_predicted']:>5}  "
          f"DEG-truth={rogue['n_truth']:>3}  TP={rogue['tp']:>3}  "
          f"P={rogue['precision']:.3f}  R={rogue['recall']:.3f}  "
          f"(independent-screen confirmation of rogue calls)")

    # 3) the live-ghost hit-list: do DEG essentials enrich among our top picks?
    ghost = df[df.live_ghost == 1] if "live_ghost" in df.columns else df.iloc[:0]
    ghost_wp = set(ghost["wp_id"].dropna())
    ghost_in_deg = ghost_wp & deg
    print(f"  LIVE-GHOST hit-list ({len(ghost_wp):,}): "
          f"{len(ghost_in_deg):,} confirmed by DEG  "
          f"({100*len(ghost_in_deg)/max(1,len(ghost_wp)):.0f}%; base rate "
          f"{100*len(deg_in_universe)/max(1,len(universe_wp)):.0f}%)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"deg_grade_{args.org}.json").write_text(json.dumps({
        "org": args.org, "deg_dataset": args.deg_dataset,
        "deg_essential": len(deg),
        "atlas_universe_wp": len(universe_wp),
        "deg_in_universe": len(deg_in_universe),
        "overall": overall, "rogue": rogue,
        "live_ghost_in_deg": len(ghost_in_deg),
        "live_ghost_total": len(ghost_wp),
    }, indent=2))
    print(f"\n  wrote deg_grade_{args.org}.json")
    return 0


def run_smoke():
    print("=" * 70)
    print("DEG cross-grade SMOKE -- WP_ extraction + precision/recall")
    print("=" * 70)
    assert extract_wp("GeneID:1221466; Protein_ID: WP_011002530.1") == "WP_011002530.1"
    assert extract_wp("GeneID:1; Protein_ID: NP_000001.1") is None
    assert extract_wp("") is None
    print("  WP_ extraction: OK")
    # precision/recall: 5 predicted, 3 truth, 2 overlap, universe 10
    universe = {f"WP_{i}" for i in range(10)}
    truth = {"WP_1", "WP_2", "WP_3"}
    pred = {"WP_1", "WP_2", "WP_4", "WP_5", "WP_9"}
    pr = precision_recall(pred, truth, universe)
    print(f"  pred={len(pred)} truth={len(truth)} tp={pr['tp']} "
          f"P={pr['precision']:.2f} R={pr['recall']:.2f}")
    assert pr["tp"] == 2 and abs(pr["precision"] - 0.4) < 1e-9
    assert abs(pr["recall"] - 2/3) < 1e-9
    print("\n=== DEG SMOKE PASS ===")
    print("  WP_ extraction + precision/recall logic validated.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--org", default="beril_RalstoniaGMI1000")
    ap.add_argument("--deg_dataset", default="DEG1057")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
