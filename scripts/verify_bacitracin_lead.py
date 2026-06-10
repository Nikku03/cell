"""Gate the envZ/ompR/pspB x bacitracin LEAD against the FULL fitness data.

The conditional atlas is pre-filtered at fit < -3, so it cannot answer the
specificity question that decides whether this is an adjuvant target or just a
fragile mutant. This script queries the unfiltered Fitness Browser data
(feba.db, on Drive) and runs three tests:

  TEST 1  per-gene specificity
          For each candidate locus, pull fitness across ALL experiments.
          Is the gene ~neutral in non-bacitracin conditions and negative only
          under bacitracin?  (rank of the bacitracin experiments within the
          gene's own fitness distribution; how negative is the gene's MEDIAN
          across everything else)

  TEST 2  per-condition background
          Under each bacitracin experiment, how many genes go fit < -3?
          If bacitracin is broadly toxic in this strain (many genes down),
          envZ/ompR is not special. If few genes respond, the hit is
          condition-specific too.  Report the percentile of the candidate
          gene within that experiment.

  TEST 3  species resolution
          Resolve org nickname -> genus/species from the Organism table so
          cross-clade claims can be stated correctly.

Run on Colab (has Drive mounted). Defaults assume feba.db at the standard
path; override with --db.
"""
from __future__ import annotations
import argparse, sqlite3, sys
from pathlib import Path

DEFAULT_DB = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/feba.db")

# (gene label, org nickname as it appears in the atlas, locusId from atlas)
CANDIDATES = [
    ("envZ", "beril_PV4",  "5211222"),
    ("envZ", "beril_SB2B", "6939419"),
    ("ompR", "beril_PV4",  "5211221"),
    ("ompR", "beril_SB2B", "6939418"),
    ("pspB", "beril_MR1",  "200970"),
    ("pspB", "beril_SB2B", "6937067"),
]
BACITRACIN_KEY = "bacitracin"


def org_to_fb(nick: str) -> str:
    """atlas nickname (beril_PV4) -> Fitness Browser orgId (PV4)."""
    return nick.replace("beril_", "")


def introspect(con):
    cur = con.cursor()
    tabs = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"  tables: {tabs}")
    for t in ("GeneFitness", "Experiment", "Organism", "Gene"):
        if t in tabs:
            cols = [r[1] for r in cur.execute(f"PRAGMA table_info({t})").fetchall()]
            print(f"    {t}: {cols}")
    return tabs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args()
    if not args.db.exists():
        print(f"ERROR: feba.db not found at {args.db}", file=sys.stderr)
        print("Pass --db /path/to/feba.db", file=sys.stderr)
        return 1

    import pandas as pd
    con = sqlite3.connect(str(args.db))

    print("=== schema introspection ===")
    tabs = introspect(con)

    # Column names in the Fitness Browser schema (standard). Adjust here if
    # introspection above shows different names.
    GF, EXP, ORG = "GeneFitness", "Experiment", "Organism"

    print("\n=== TEST 3: species resolution ===")
    try:
        orgs = pd.read_sql(f"SELECT * FROM {ORG}", con)
        for nick in sorted({c[1] for c in CANDIDATES}):
            fb = org_to_fb(nick)
            row = orgs[orgs.apply(lambda r: fb in r.astype(str).values, axis=1)]
            print(f"  {nick} ({fb}):")
            if len(row):
                print("    " + row.iloc[0].to_dict().__repr__())
            else:
                print("    not matched in Organism table")
    except Exception as e:
        print(f"  (skipped: {e})")

    for gene, nick, locus in CANDIDATES:
        fb = org_to_fb(nick)
        print(f"\n========== {gene}  {nick} ({fb})  locus={locus} ==========")

        # full fitness profile for this gene across ALL experiments
        try:
            gf = pd.read_sql(
                f"SELECT g.expName, g.fit, g.t, e.expDesc, e.expGroup, "
                f"       e.condition_1, e.media "
                f"FROM {GF} g JOIN {EXP} e "
                f"  ON g.orgId=e.orgId AND g.expName=e.expName "
                f"WHERE g.orgId=? AND g.locusId=?",
                con, params=(fb, locus))
        except Exception as e:
            print(f"  query failed ({e}); trying minimal columns")
            gf = pd.read_sql(
                f"SELECT expName, fit, t FROM {GF} "
                f"WHERE orgId=? AND locusId=?", con, params=(fb, locus))
            gf["expDesc"] = gf["condition_1"] = ""

        if len(gf) == 0:
            print("  NO fitness rows — check orgId/locus mapping")
            continue

        n = len(gf)
        med_all = gf.fit.median()
        is_baci = gf.expDesc.str.lower().str.contains(BACITRACIN_KEY, na=False) | \
                  gf.get("condition_1", pd.Series([""]*n)).astype(str)\
                    .str.lower().str.contains(BACITRACIN_KEY, na=False)
        baci = gf[is_baci]
        rest = gf[~is_baci]

        print(f"  experiments: {n}   median fit (all): {med_all:+.2f}")
        print(f"  --- TEST 1 specificity ---")
        print(f"  non-bacitracin median fit: {rest.fit.median():+.2f}  "
              f"(n={len(rest)})   "
              f"frac of non-baci with fit<-2: {(rest.fit < -2).mean():.2f}")
        print(f"  bacitracin experiments (n={len(baci)}):")
        for _, r in baci.sort_values("fit").iterrows():
            # rank of this experiment within the gene's own distribution
            rank = (gf.fit < r.fit).sum()
            pct = 100 * rank / n
            print(f"    fit={r.fit:+.2f} t={r.t:+.2f}  "
                  f"[{rank}/{n} = {pct:.0f}th pctile of gene's own range]  "
                  f"{str(r.expDesc)[:50]}")
        # verdict heuristic
        if len(rest) and len(baci):
            sep = rest.fit.median() - baci.fit.median()
            print(f"  SPECIFICITY GAP (non-baci median - baci median) = {sep:+.2f}")
            if rest.fit.median() > -1.5 and baci.fit.min() < -3:
                print(f"    -> looks SPECIFIC (neutral elsewhere, negative on baci)")
            elif rest.fit.median() < -2:
                print(f"    -> NOT specific: gene is broadly negative (fragile mutant)")
            else:
                print(f"    -> ambiguous; inspect distribution")

        # TEST 2: per-condition background for each bacitracin experiment
        print(f"  --- TEST 2 condition background ---")
        for _, r in baci.iterrows():
            try:
                allg = pd.read_sql(
                    f"SELECT fit FROM {GF} WHERE orgId=? AND expName=?",
                    con, params=(fb, r.expName))
                n_down = int((allg.fit < -3).sum())
                n_tot = len(allg)
                worse = int((allg.fit < r.fit).sum())
                print(f"    exp {r.expName}: {n_down}/{n_tot} genes fit<-3 "
                      f"({100*n_down/max(n_tot,1):.1f}%); this gene is "
                      f"#{worse+1} most-negative")
            except Exception as e:
                print(f"    exp {r.expName}: background query failed ({e})")

    con.close()
    print("\n=== DONE ===")
    print("Decision rule:")
    print("  PASS (validated surprise) if: non-baci median fit > ~ -1.5 AND")
    print("    bacitracin in the gene's bottom decile AND bacitracin experiments")
    print("    do NOT down >~20% of all genes (i.e. not broadly toxic).")
    print("  FAIL -> downgrade to 'fragile mutant' or 'broadly toxic condition'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
