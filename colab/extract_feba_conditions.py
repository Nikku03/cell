"""Stage 1 of conditional essentiality: extract per-(gene, condition) labels
from the Fitness Browser feba.db.

Downloads feba.db (~2.3 GB compressed, ~7 GB uncompressed; ~10-min on Colab)
via the project's existing `fetch_feba`. Then:

  1. Read Experiment table -> (orgId, expName, media, aerobic, condition_1).
     Cluster key = "media | aerobic | condition_1" -- the standard feba cluster.
  2. Map feba orgIds to our beril_* organism names (heuristic + manual map).
     Print the match table; verify gene-id overlap for each mapped org.
  3. For each mapped org, stream its GeneFitness rows; call a gene "essential
     in cluster C" iff `frac of C's experiments with (fit < -2 AND |t| >= 4) >= 0.5`.
  4. Emit a long-format table:
       organism, locus_tag, condition, n_experiments, frac_essential, essential
     where (organism, locus_tag) repeats once per condition cluster.

Outputs (under data/drive_import/labels_cond/):
  feba_conditions.csv        the per-(gene, condition) labels
  feba_org_map.json          how feba orgIds map to our org names
  condition_vocabulary.json  cluster name -> count
  gene_match_report.json     per-org locusId<->locus_tag overlap (sanity check)

Run on Colab:
  python colab/extract_feba_conditions.py --feba_db /tmp/feba.db
"""
import argparse, csv, json, sqlite3, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

ESS_FIT_T = -2.0
ESS_T_T   = 4.0


# Manual map seeded with known beril<->feba pairs from the project context.
# (heuristic match fills in the rest where possible.)
MANUAL_MAP = {
    "Keio":     "beril_Keio",
    "MR1":      "beril_MR1",
    "WCS417":   "beril_WCS417",
    "Smelio":   "beril_Smeli",
    "Btheta":   "beril_Btheta",
    "Caulo":    "beril_Caulo",
    "DvH":      "beril_DvH",
    "SynE":     "beril_SynE",
    "Putida":   "beril_Putida",
    "PA14":     "beril_PA14",
    "Koxytoca": "beril_Koxy",
    "azobra":   "beril_azobra",
}


def map_feba_to_our_orgs(feba_orgs, our_orgs):
    out = {}
    for fb in feba_orgs:
        if fb in MANUAL_MAP and MANUAL_MAP[fb] in our_orgs:
            out[fb] = MANUAL_MAP[fb]
            continue
        # heuristic: lowercase substring match after stripping prefix/underscores
        fl = fb.lower().replace("_", "")
        best = None
        for ours in our_orgs:
            ol = ours.lower().replace("_", "").replace("beril", "")
            if fl == ol or fl in ol or ol in fl:
                best = ours; break
        if best:
            out[fb] = best
    return out


def verify_gene_overlap(con, feba_org, our_org, our_genes, fit_table):
    """Print how many feba locusIds match our locus_tags for this org."""
    import pandas as pd
    feba_genes = pd.read_sql(
        f"SELECT DISTINCT locusId FROM {fit_table} WHERE orgId = ?",
        con, params=(feba_org,))["locusId"].astype(str).tolist()
    ours_set = our_genes.get(our_org, set())
    matched = sum(1 for g in feba_genes if g in ours_set)
    return dict(feba_n=len(feba_genes), our_n=len(ours_set),
                matched=matched,
                match_pct=round(100 * matched / max(1, len(feba_genes)), 1))


def pick_table(con, candidates, what):
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    have = {r[0] for r in cur.fetchall()}
    for c in candidates:
        if c in have:
            return c
    sys.exit(f"{what}: none of {candidates} found in feba.db (have: {sorted(have)[:10]}...)")


def main(db_path, labels_dir, out_dir):
    import pandas as pd
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(db_path)

    if not db_path.exists():
        print(f"downloading feba.db -> {db_path} ...")
        from kappa_floor import fetch_feba
        fetch_feba(db_path)
    print(f"feba.db: {db_path.stat().st_size / 1e9:.2f} GB")

    con = sqlite3.connect(str(db_path))
    exp_table = pick_table(con, ["Experiment", "Experiments", "Exp"], "experiment")
    fit_table = pick_table(con, ["GeneFitness", "Fitness", "GeneFit"], "fitness")

    exp = pd.read_sql(
        f"SELECT orgId, expName, media, aerobic, condition_1 FROM {exp_table}", con)
    for c in ("media", "aerobic", "condition_1"):
        exp[c] = exp[c].fillna("-").astype(str).str.strip()
    exp["cluster"] = exp.media + " | " + exp.aerobic + " | " + exp.condition_1
    print(f"\nfeba experiments: {len(exp):,} | orgs: {exp.orgId.nunique()} | "
          f"unique clusters: {exp.cluster.nunique():,}")

    # ---- our orgs + per-org gene sets ----
    our_genes = {}
    for r in csv.DictReader(open(Path(labels_dir) / "labels.csv")):
        our_genes.setdefault(r["organism"], set()).add(r["locus_tag"])
    our_orgs = set(our_genes)
    feba_orgs = sorted(exp.orgId.unique())
    org_map = map_feba_to_our_orgs(feba_orgs, our_orgs)

    print(f"\nmapped {len(org_map)} feba orgs -> our orgs:")
    overlap_report = {}
    for fb, ours in sorted(org_map.items()):
        v = verify_gene_overlap(con, fb, ours, our_genes, fit_table)
        overlap_report[fb] = dict(our_org=ours, **v)
        print(f"  {fb:<14s} -> {ours:<25s}  feba_genes={v['feba_n']:<5}  "
              f"matched={v['matched']:<5} ({v['match_pct']:>5.1f}%)")
    json.dump(org_map, open(out_dir / "feba_org_map.json", "w"), indent=2)
    json.dump(overlap_report, open(out_dir / "gene_match_report.json", "w"), indent=2)

    keep_orgs = [fb for fb, v in overlap_report.items() if v["match_pct"] >= 30.0]
    print(f"\nkeeping {len(keep_orgs)} orgs with >= 30% gene-id match")

    # ---- per-org: cluster -> per-gene essential calls ----
    all_rows = []
    cluster_counts = {}
    for fb_org in keep_orgs:
        our_org = org_map[fb_org]
        org_exp = exp[exp.orgId == fb_org]
        usable = {c for c, g in org_exp.groupby("cluster")
                  if g.expName.nunique() >= 2}
        if not usable:
            print(f"  [skip] {fb_org}: no usable clusters (need >=2 experiments)")
            continue
        sub = pd.read_sql(
            f"SELECT locusId, expName, fit, t FROM {fit_table} WHERE orgId = ?",
            con, params=(fb_org,))
        if sub.empty:
            continue
        sub["is_ess"] = ((sub.fit < ESS_FIT_T) & (sub.t.abs() >= ESS_T_T)).astype(int)
        sub = sub.merge(org_exp[["expName", "cluster"]], on="expName", how="left")
        sub = sub[sub.cluster.isin(usable)]
        agg = (sub.groupby(["locusId", "cluster"])
                  .agg(n_exp=("expName", "nunique"),
                       frac_ess=("is_ess", "mean"))
                  .reset_index())
        agg["organism"] = our_org
        agg["essential"] = (agg.frac_ess >= 0.5).astype(int)
        agg = agg.rename(columns={"locusId": "locus_tag", "cluster": "condition",
                                  "n_exp": "n_experiments",
                                  "frac_ess": "frac_essential"})
        all_rows.append(agg[["organism", "locus_tag", "condition",
                             "n_experiments", "frac_essential", "essential"]])
        ess = agg.essential.sum()
        print(f"  {fb_org:<14s} -> {our_org:<25s}  "
              f"{len(agg):>6} rows  ({ess:>5} essential calls; {len(usable)} clusters)")
        for c in usable:
            cluster_counts[c] = cluster_counts.get(c, 0) + 1

    if not all_rows:
        sys.exit("no rows produced -- did any org pass the gene-id match threshold?")
    df = pd.concat(all_rows, ignore_index=True)
    df.frac_essential = df.frac_essential.round(3)
    out_csv = out_dir / "feba_conditions.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n=== {len(df):,} (org, gene, condition) rows over {df.organism.nunique()} orgs ===")
    print(f"  essential calls: {df.essential.sum():,} ({df.essential.mean()*100:.1f}%)")
    print(f"  unique conditions: {df.condition.nunique():,}")
    json.dump({k: v for k, v in sorted(cluster_counts.items(), key=lambda x: -x[1])[:200]},
              open(out_dir / "condition_vocabulary.json", "w"), indent=2)
    print(f"\nwrote {out_csv}")
    print(f"wrote {out_dir / 'feba_org_map.json'}")
    print(f"wrote {out_dir / 'gene_match_report.json'}")
    print(f"wrote {out_dir / 'condition_vocabulary.json'}")
    print("\nnext: review feba_org_map.json + gene_match_report.json -- "
          "if any expected org is missing, add to MANUAL_MAP and rerun.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feba_db", default="/tmp/feba.db",
                    help="path to feba.db (will be downloaded if absent)")
    ap.add_argument("--labels_dir", default="data/drive_import/labels",
                    help="dir containing labels.csv with our organism names")
    ap.add_argument("--out_dir", default="data/drive_import/labels_cond",
                    help="output dir for the conditional labels")
    a = ap.parse_args()
    main(a.feba_db, a.labels_dir, a.out_dir)
