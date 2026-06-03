"""Download a curated set of bacterial BiGG genome-scale metabolic models.

Models chosen for coverage against our BERIL organism set + biological
diversity (gammaproteobacteria, firmicutes, actinobacteria,
cyanobacteria). Each is the most recent / most genes / most curated
version available for that species in BiGG.

Run:
  python scripts/download_bigg_models.py \\
      --out-dir memory_bank/data/multiorg_essentiality/_bigg_cache
"""
from __future__ import annotations
import argparse, sys, time, urllib.request, urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "_bigg_cache"

# (bigg_model_id, species, default beril organism guess)
# Verify the beril org against build_clade_splits.py BERIL_ORG_TO_CLADE
# keys and adjust the mapping CSV before running build_fba_features.
MODELS = [
    ("iML1515",   "Escherichia coli K12 MG1655",        "beril_Keio"),
    ("iJO1366",   "Escherichia coli K12 MG1655 (older)", "beril_Keio"),
    ("iYO844",    "Bacillus subtilis 168",               "beril_BS168"),
    ("iSB619",    "Staphylococcus aureus N315",          "beril_SaureusN315"),
    ("iJN1463",   "Pseudomonas putida KT2440",           "beril_Pput"),
    ("iJN746",    "Pseudomonas putida (older)",          "beril_Pput"),
    ("iPae1146",  "Pseudomonas aeruginosa PA14",         "beril_PA14"),
    ("iLB1027_lipid", "Klebsiella pneumoniae",           "beril_KlebPneumo"),
    ("STM_v1_0",  "Salmonella Typhimurium",              "beril_STyphi"),
    ("iEK1008",   "Mycobacterium tuberculosis H37Rv",    "beril_Mtb"),
    ("iSyn669",   "Synechocystis sp PCC 6803",           "beril_Synecho"),
    ("iCN718",    "Acinetobacter baumannii",             "beril_Abaum"),
    ("iAF692",    "Methanosarcina barkeri",              "ARCHAEON_SKIP"),
    ("iJN678",    "Synechocystis (older)",               "beril_Synecho"),
]

BIGG_BASE = "http://bigg.ucsd.edu/static/models"


def download(model_id: str, out_dir: Path) -> bool:
    url = f"{BIGG_BASE}/{model_id}.xml.gz"
    dst = out_dir / f"{model_id}.xml.gz"
    if dst.exists() and dst.stat().st_size > 1000:
        print(f"  cached: {dst.name}  ({dst.stat().st_size/1e6:.1f} MB)")
        return True
    try:
        t0 = time.time()
        urllib.request.urlretrieve(url, dst)
        print(f"  {dst.name}  ({dst.stat().st_size/1e6:.1f} MB, "
              f"{time.time()-t0:.1f}s)")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # Some models are only available as .xml (no .gz)
            try:
                url2 = f"{BIGG_BASE}/{model_id}.xml"
                dst2 = out_dir / f"{model_id}.xml"
                urllib.request.urlretrieve(url2, dst2)
                print(f"  {dst2.name}  ({dst2.stat().st_size/1e6:.1f} MB)")
                return True
            except Exception as e2:
                print(f"  FAILED {model_id}: {e2}"); return False
        print(f"  FAILED {model_id}: {e}"); return False
    except Exception as e:
        print(f"  FAILED {model_id}: {e}"); return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--map-csv", type=Path,
                   default=DEFAULT_OUT / "bigg_to_beril_map.csv",
                   help="Where to write the bigg_model -> beril_organism map")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {args.out_dir}")
    ok = 0
    for mid, sp, _ in MODELS:
        print(f"\n[{sp}]")
        if download(mid, args.out_dir): ok += 1

    # write mapping CSV (excluding archaeon, and using .xml.gz suffix that the
    # download path produced; user can adjust if a model came down as plain .xml)
    print(f"\nWriting mapping to {args.map_csv}")
    args.map_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.map_csv, "w") as f:
        f.write("bigg_model_file,beril_organism\n")
        for mid, sp, org in MODELS:
            if org == "ARCHAEON_SKIP": continue
            # Pick whichever extension we have
            gz  = args.out_dir / f"{mid}.xml.gz"
            xml = args.out_dir / f"{mid}.xml"
            jsn = args.out_dir / f"{mid}.json"
            for cand in (gz, xml, jsn):
                if cand.exists():
                    f.write(f"{cand.name},{org}\n")
                    break

    print(f"\nDone: {ok}/{len(MODELS)} downloaded.")
    print(f"REVIEW {args.map_csv} -- the beril_organism column is a best-guess. "
           "Check against scripts/build_clade_splits.py BERIL_ORG_TO_CLADE "
           "and edit before running build_fba_features.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
