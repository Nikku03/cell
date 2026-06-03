"""FBA-based essentiality + network-topology features per gene.

Fix 3 of the error-mode plan:
  - The dominant FN class (artQ, cysA, rbsA, bioH, caiC -- transporters
    and conditional biosynthesis) is essential by NETWORK position in
    a specific organism, not by family conservation.
  - The large-genome FP skew (Pseudomonas, Ralstonia, Dickeya) is
    hidden isozyme redundancy: family says essential, but a paralog
    exists that provides the same flux.
  - Both are direct FBA signals.

Per gene, computes:
  - fba_essential       : 1 if single-gene knockout reduces biomass
                            < threshold * WT, else 0
  - fba_n_rxns          : number of reactions catalyzed (degree)
  - fba_in_model        : 1 if gene is in the GEM (so XGBoost can
                            distinguish "predicted non-essential" from
                            "no model"; otherwise fba_essential == 0
                            for both)

Caveat on coverage:
  Only ~10-15 of our 47 BERIL organisms have curated GEMs in BiGG.
  For the rest, all three features are NaN -- XGBoost handles missing
  natively, but the feature only meaningfully helps on the covered
  organisms. To extend coverage, point this script at additional SBML
  files (e.g. CarveMe-generated reconstructions for uncovered orgs).

Inputs:
  --bigg-dir : directory of *.xml or *.xml.gz SBML files (or *.json)
  --map-csv  : two-col CSV mapping bigg_model_filename -> beril_organism

Output:
  memory_bank/data/multiorg_essentiality/fba_features.csv
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality"
OUT_CSV   = DATA_DIR / "fba_features.csv"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bigg-dir",  type=Path, required=True,
                   help="Directory holding downloaded BiGG SBML models")
    p.add_argument("--map-csv",   type=Path, required=True,
                   help="CSV: bigg_model_file,beril_organism (e.g. "
                        "iML1515.xml,beril_Keio)")
    p.add_argument("--out",       type=Path, default=OUT_CSV)
    p.add_argument("--essential-frac",  type=float, default=0.01,
                   help="KO biomass < this fraction of WT -> essential (default 0.01)")
    args = p.parse_args()

    try:
        import cobra
    except ImportError:
        print("ERROR: pip install cobra", file=sys.stderr); return 1

    # ---- map: bigg model file -> beril organism ----
    mapping: list[tuple[str, str]] = []
    with open(args.map_csv) as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row or row[0].startswith("#") or row[0].lower().startswith("bigg"):
                continue
            if len(row) < 2: continue
            mapping.append((row[0].strip(), row[1].strip()))
    print(f"Mapping: {len(mapping)} model -> organism entries")
    for m, o in mapping: print(f"  {m:<25s} -> {o}")

    rows_out: list[dict] = []
    for model_fn, beril_org in mapping:
        path = args.bigg_dir / model_fn
        if not path.exists():
            print(f"  SKIP missing {path}")
            continue

        print(f"\nLoading {path} ...")
        t0 = time.time()
        if path.suffix == ".json":
            model = cobra.io.load_json_model(path)
        elif path.suffixes[-2:] == [".xml", ".gz"]:
            model = cobra.io.read_sbml_model(str(path))
        else:
            model = cobra.io.read_sbml_model(str(path))
        print(f"  {len(model.reactions)} reactions  {len(model.metabolites)} metabolites  "
              f"{len(model.genes)} genes  (load {time.time()-t0:.1f}s)")

        # WT growth
        try:
            wt = model.optimize().objective_value
        except Exception as e:
            print(f"  WT optimisation failed: {e}"); continue
        if wt is None or wt < 1e-6:
            print(f"  WT biomass is zero ({wt}); skipping"); continue
        print(f"  WT biomass: {wt:.4f}  -> essentiality threshold {args.essential_frac * wt:.4f}")

        # Single-gene knockout via cobra utility
        t0 = time.time()
        df_ko = cobra.flux_analysis.single_gene_deletion(model)
        print(f"  KO done in {time.time()-t0:.1f}s ({len(df_ko)} genes)")

        # df_ko: index is frozenset({gene_id}), columns ['ids','growth','status']
        # Newer cobra: 'ids' is the frozenset; 'growth' is biomass.
        if "growth" in df_ko.columns:
            growth_col = "growth"
        elif "Growth" in df_ko.columns:
            growth_col = "Growth"
        else:
            print(f"  unexpected columns: {df_ko.columns.tolist()}"); continue
        ess_thresh = args.essential_frac * wt

        # Build gene -> n_reactions map
        gene_n_rxns = {g.id: len(g.reactions) for g in model.genes}

        n_essential = 0
        for ids, growth in zip(df_ko["ids"], df_ko[growth_col]):
            gene_id = list(ids)[0] if isinstance(ids, (frozenset, set, tuple, list)) else ids
            if growth is None or growth != growth:  # NaN check
                ess = 0
            else:
                ess = 1 if growth < ess_thresh else 0
            n_essential += ess
            rows_out.append({
                "organism":      beril_org,
                "locus_tag":     gene_id,
                "fba_essential": ess,
                "fba_n_rxns":    gene_n_rxns.get(gene_id, 0),
                "fba_in_model":  1,
            })
        print(f"  -> {n_essential}/{len(df_ko)} predicted essential by FBA "
              f"({100*n_essential/max(len(df_ko),1):.1f}%)")

    print(f"\nTotal rows: {len(rows_out)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["organism","locus_tag","fba_essential","fba_n_rxns","fba_in_model"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_out: w.writerow(r)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
