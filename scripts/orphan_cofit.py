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


def _manifest_acc(org):
    with open(DATA / "labels" / "genome_cache_manifest.csv") as f:
        for r in csv.DictReader(f):
            if r["organism"] == org and r["accession"]:
                return r["accession"]
    return None


def _locus_bridge(acc):
    """Map every old_locus_tag (e.g. RSc0001) AND the RefSeq locus_tag itself
    -> our RefSeq locus_tag (RS_RS#####), from the GFF. Handles %2C-separated
    multi old tags. Lets us join FB (which uses old tags) to our labels."""
    import re, urllib.parse
    gff = DATA / "genome_cache" / acc / "genomic.gff"
    bridge = {}
    if not gff.exists():
        return bridge
    with open(gff) as f:
        for line in f:
            c = line.split("\t")
            if len(c) < 9 or c[2] not in ("gene", "CDS"):
                continue          # old_locus_tag lives on the 'gene' feature
            lt = re.search(r'locus_tag=([^;\n]+)', c[8])
            if not lt:
                continue
            lt = lt.group(1)
            bridge[lt] = lt
            old = re.search(r'old_locus_tag=([^;\n]+)', c[8])
            if old:
                for t in urllib.parse.unquote(old.group(1)).split(","):
                    if t:
                        bridge[t.strip()] = lt
    return bridge


def _resolve_org(con, want_regex):
    """Find the FB orgId with the most Cofit rows whose Organism name matches."""
    import pandas as pd, re
    try:
        org = pd.read_sql("SELECT * FROM Organism", con)
    except Exception:
        org = None
    cof_counts = pd.read_sql(
        "SELECT orgId, COUNT(*) n FROM Cofit GROUP BY orgId", con)
    cand = cof_counts.copy()
    if org is not None:
        namecol = next((c for c in ("genus", "species", "Organism", "name")
                        if c in org.columns), None)
        if "orgId" in org.columns and namecol:
            org["_nm"] = org[namecol].astype(str)
            ids = org[org["_nm"].str.contains(want_regex, case=False, regex=True,
                                              na=False)]["orgId"].tolist()
            if ids:
                cand = cof_counts[cof_counts.orgId.isin(ids)]
    cand = cand[cand.n > 0].sort_values("n", ascending=False)
    return cand


def run_real(args):
    import pandas as pd, numpy as np
    bridge_pq = OUT / f"bridge_{args.org}.parquet"
    if not bridge_pq.exists():
        print("ERROR: run orphan_bridge.py --real first"); return 2
    if not Path(args.feba).exists():
        print(f"ERROR: feba.db not found at {args.feba} (Colab step)"); return 2
    con = sqlite3.connect(args.feba)

    # 1) resolve FB orgId (auto: most Cofit rows matching the name regex)
    cand = _resolve_org(con, args.fb_org_regex)
    if args.fb_org:
        forced = cand[cand.orgId == args.fb_org]
        cand = forced if len(forced) else cand
    if len(cand) == 0:
        print(f"  no Cofit orgId matches /{args.fb_org_regex}/; available top:")
        print(pd.read_sql("SELECT orgId,COUNT(*) n FROM Cofit GROUP BY orgId "
                          "ORDER BY n DESC LIMIT 10", con).to_string(index=False))
        con.close(); return 1
    fb_org = cand.iloc[0]["orgId"]
    print(f"  FB orgId resolved: {fb_org!r}  (Cofit rows {int(cand.iloc[0]['n']):,})")

    # 2) old-tag bridge: FB tag -> our RS_RS#####
    acc = _manifest_acc(args.org)
    lb = _locus_bridge(acc) if acc else {}
    # FB Gene table: locusId/sysName for this org -> our locus_tag via bridge
    gene = pd.read_sql("SELECT locusId, sysName FROM Gene WHERE orgId=?",
                       con, params=(fb_org,))
    fb2rs = {}
    for r in gene.itertuples():
        for key in (r.sysName, r.locusId):
            if key in lb:
                fb2rs[r.locusId] = lb[key]
                break
    print(f"  FB gene -> our locus_tag mapped: {len(fb2rs):,}/{len(gene):,} "
          f"({100*len(fb2rs)/max(1,len(gene)):.0f}%)")

    # 3) Cofit, remapped to our locus_tag on both ends
    cof = pd.read_sql(
        "SELECT locusId, hitId, cofit, rank FROM Cofit WHERE orgId=?",
        con, params=(fb_org,))
    con.close()
    cons = _cons_map(args.org)
    df = pd.read_parquet(bridge_pq)
    by_gene = {}
    for r in cof.itertuples():
        g = fb2rs.get(r.locusId); h = fb2rs.get(r.hitId)
        if g is None or h is None:
            continue
        by_gene.setdefault(g, []).append((h, r.cofit, r.rank))
    rows = []
    for lt in df.locus_tag:
        agg = aggregate(by_gene.get(lt, []), cons, args.top_n)
        agg["locus_tag"] = lt; rows.append(agg)
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / f"cofit_{args.org}.parquet", index=False)
    print(f"  genes with >=1 usable cofit partner: {int((out.cofit_n>0).sum()):,}"
          f" / {len(out):,}")
    if (out.cofit_n > 0).sum() == 0:
        print("  *** still 0 -- inspect Gene.sysName/locusId format vs GFF "
              "old_locus_tag; the namespace bridge needs adjustment. ***")
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
    # old-tag bridge: %2C-separated old_locus_tag -> RefSeq locus_tag
    import tempfile
    global DATA
    d = Path(tempfile.mkdtemp()); (d / "genome_cache" / "G").mkdir(parents=True)
    (d / "genome_cache" / "G" / "genomic.gff").write_text(
        "c\tx\tCDS\t1\t9\t.\t+\t0\tlocus_tag=RS_RS25220;old_locus_tag=RS06178%2CRSc0001\n"
        "c\tx\tCDS\t20\t29\t.\t+\t0\tlocus_tag=RS_RS00005;old_locus_tag=RSc0002\n")
    saved = DATA; DATA = d
    try:
        lb = _locus_bridge("G")
    finally:
        DATA = saved
    print(f"  locus bridge: RSc0001->{lb.get('RSc0001')}  RSc0002->{lb.get('RSc0002')}")
    assert lb["RSc0001"] == "RS_RS25220" and lb["RS06178"] == "RS_RS25220"
    assert lb["RSc0002"] == "RS_RS00005" and lb["RS_RS25220"] == "RS_RS25220"
    print("  FB old-tag (RSc####) -> RefSeq (RS_RS#####) bridge: OK")
    print("\n=== STEP 5 SMOKE PASS ===")
    print("  top-N cofit selection + LEAK-FREE partner-conservation median")
    print("  (no in-org labels used) validated. Real reads feba.db on Colab.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--org", default="beril_RalstoniaGMI1000")
    ap.add_argument("--fb_org", default="", help="force FB orgId (else auto)")
    ap.add_argument("--fb_org_regex", default="alstonia|olanacearum",
                    help="Organism-name regex to auto-resolve the FB orgId")
    ap.add_argument("--feba", default="/content/feba.db")
    ap.add_argument("--top_n", type=int, default=TOP_N)
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
