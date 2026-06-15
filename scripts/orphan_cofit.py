"""PATH-ORPHAN, STEP 5 (Phase A) -- cofitness coupling feature [LEAK-FREE].

A rogue essential that co-varies in fitness with conserved-essential machinery
across hundreds of conditions is probably essential for the same reason. The
Fitness Browser `Cofit` table gives precomputed gene-gene cofitness.

LEAK RULE (HANDOFF, non-negotiable): never use a same-organism partner's
essentiality LABEL -- that leaks the test label within the organism. Instead we
aggregate each gene's top-N cofit partners' LEAK-FREE CONSERVATION (family_frac,
computed excluding this organism). That captures "this gene couples to
essential-type machinery" without touching any in-organism label.

INPUT (real): feba.db `Cofit` (orgId,locusId,hitId,cofit,rank);
              orthology_features.csv (partner locus_tag -> family_frac leakfree)
OUTPUT: outputs/orphan/cofit_<org>.parquet
        locus_tag, cofit_cons (median partner conservation), cofit_cons_max,
        cofit_n (partners used)

--smoke : validate top-N selection + leak-free conservation aggregation
--real  : read feba.db (Colab); FB orgId for the organism via --fb_org
"""
from __future__ import annotations
import argparse, sys, csv, sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "drive_import"
OUT = REPO / "outputs" / "orphan"
TOP_N = 10


def aggregate(partners, cons_map, top_n=TOP_N):
    """partners: list of (partner_locus, cofit, rank) for a focal gene.
    cons_map: partner_locus -> leak-free conservation. Returns dict."""
    import numpy as np
    ps = sorted(partners, key=lambda x: -x[1])[:top_n]      # top by cofit
    vals = [cons_map[p] for p, _, _ in ps if p in cons_map and cons_map[p] == cons_map[p]]
    if not vals:
        return {"cofit_cons": float("nan"), "cofit_cons_max": float("nan"),
                "cofit_n": 0}
    return {"cofit_cons": float(np.median(vals)),
            "cofit_cons_max": float(np.max(vals)), "cofit_n": len(vals)}


def _cons_map(org):
    """partner locus_tag -> leak-free conservation, for the focal organism."""
    cm = {}
    with open(DATA / "labels" / "orthology_features.csv") as f:
        for r in csv.DictReader(f):
            if r["organism"] == org:
                try:
                    cm[r["locus_tag"]] = float(r["family_frac_essential_leakfree"])
                except (TypeError, ValueError):
                    pass
    return cm


def run_real(args):
    import pandas as pd, numpy as np
    bridge = OUT / f"bridge_{args.org}.parquet"
    if not bridge.exists():
        print("ERROR: run orphan_bridge.py --real first"); return 2
    if not Path(args.feba).exists():
        print(f"ERROR: feba.db not found at {args.feba} (Colab step)"); return 2
    fb_org = args.fb_org
    con = sqlite3.connect(args.feba)
    cof = pd.read_sql(
        "SELECT locusId, hitId, cofit, rank FROM Cofit WHERE orgId=?",
        con, params=(fb_org,))
    con.close()
    print(f"  Cofit edges for {fb_org}: {len(cof):,}")
    cons = _cons_map(args.org)
    # join: FB locusId == our locus_tag (validate)
    df = pd.read_parquet(bridge)
    overlap = len(set(cof.locusId) & set(df.locus_tag))
    print(f"  locusId<->locus_tag overlap: {overlap}/{df.locus_tag.nunique()} "
          f"({100*overlap/df.locus_tag.nunique():.0f}%)")
    by_gene = {}
    for r in cof.itertuples():
        by_gene.setdefault(r.locusId, []).append((r.hitId, r.cofit, r.rank))
    rows = []
    for lt in df.locus_tag:
        agg = aggregate(by_gene.get(lt, []), cons, args.top_n)
        agg["locus_tag"] = lt; rows.append(agg)
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / f"cofit_{args.org}.parquet", index=False)
    print(f"  genes with >=1 cofit partner: {(out.cofit_n>0).sum():,}")
    print(f"  wrote cofit_{args.org}.parquet")
    return 0


def run_smoke():
    import numpy as np
    print("=" * 64)
    print("STEP 5 cofit SMOKE -- leak-free conservation aggregation")
    print("=" * 64)
    cons = {"A": 0.9, "B": 0.8, "C": 0.1, "D": float("nan"), "E": 0.7}
    # focal gene cofits (partner, cofit, rank); top-3 by cofit = A,E,B
    partners = [("A", 0.95, 1), ("E", 0.90, 2), ("B", 0.85, 3),
                ("C", 0.10, 4), ("Z", 0.99, 0)]  # Z highest cofit, no conservation
    agg = aggregate(partners, cons, top_n=3)
    print(f"  top-3 by cofit = Z,A,E; Z dropped (no cons) -> cofit_cons="
          f"{agg['cofit_cons']:.3f} max={agg['cofit_cons_max']:.3f} "
          f"n={agg['cofit_n']}")
    # top-3 by cofit are Z(0.99),A(0.95),E(0.90); Z has no conservation so the
    # usable partners are A=0.9,E=0.7 -> median 0.8, n=2, max 0.9
    assert abs(agg["cofit_cons"] - 0.8) < 1e-9, "median of usable top-N cons"
    assert agg["cofit_n"] == 2 and abs(agg["cofit_cons_max"] - 0.9) < 1e-9
    # gene that only couples to a low-conservation partner
    agg2 = aggregate([("C", 0.99, 1)], cons, top_n=3)
    print(f"  couples only to C(cons0.1) -> cofit_cons={agg2['cofit_cons']:.3f}")
    assert abs(agg2["cofit_cons"] - 0.1) < 1e-9
    # gene with no usable partners
    agg3 = aggregate([("D", 0.9, 1), ("Z", 0.9, 2)], cons, top_n=3)
    assert agg3["cofit_n"] == 0 and np.isnan(agg3["cofit_cons"])
    print(f"  no usable partner -> n=0, cons=NaN  OK")
    print("\n=== STEP 5 SMOKE PASS ===")
    print("  top-N cofit selection + LEAK-FREE partner-conservation median")
    print("  (no in-org labels used) validated. Real reads feba.db on Colab.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--org", default="beril_RalstoniaGMI1000")
    ap.add_argument("--fb_org", default="Ralstonia", help="FB orgId")
    ap.add_argument("--feba", default="/content/feba.db")
    ap.add_argument("--top_n", type=int, default=TOP_N)
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
