"""PATH-ORPHAN, structure channel RUN-half (Colab) -- AFDB pull + Foldseek.

orphan_foldseek.py already GRADES a foldseek .m8 (named-vs-dark, evalue/TM/pLDDT
filters). The missing half, designed today, is producing that .m8: map our
RefSeq protein_ids -> UniProt, pull AlphaFold structures (no prediction needed),
Foldseek them vs AFDB-cluster reps, and extract per-query pLDDT.

This script holds the testable plumbing (URL building, pLDDT parsing, idmapping
request shaping); the heavy steps (UniProt REST, wget AFDB, foldseek easy-search)
run on Colab and are wired in run_real with explicit commands.

Pipeline (run_real, Colab):
  uniprot_request_<org>.txt (RefSeq ids, from step 0)
    -> UniProt idmapping RefSeq_Protein->UniProtKB
    -> AFDB pull  AF-<acc>-F1-model_v4.pdb  into queries/
    -> foldseek easy-search queries/ afdb_clust result.m8
    -> plddt.tsv (mean pLDDT per query) + target_anno.tsv
  then: orphan_foldseek.py --real grades it.

--smoke : validate AFDB URL, pLDDT parsing, idmapping request shaping
--real  : run the pull+search on Colab
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "orphan"
AFDB_TMPL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb"


def afdb_url(uniprot_acc):
    return AFDB_TMPL.format(acc=uniprot_acc)


def mean_plddt(pdb_text):
    """AlphaFold PDBs store per-residue pLDDT in the B-factor column of CA atoms.
    Return the mean over CA atoms (the model-confidence used to filter hits)."""
    vals = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                vals.append(float(line[60:66]))
            except ValueError:
                pass
    return sum(vals) / len(vals) if vals else float("nan")


def idmapping_payload(refseq_ids):
    """Shape the UniProt idmapping POST body: RefSeq_Protein -> UniProtKB."""
    return {"from": "RefSeq_Protein", "to": "UniProtKB",
            "ids": ",".join(refseq_ids)}


def run_real(args):
    req = OUT / f"uniprot_request_{args.org}.txt"
    if not req.exists():
        print("ERROR: run orphan_bridge.py --real first (need uniprot_request)"); return 2
    ids = [x.strip() for x in req.read_text().splitlines() if x.strip()]
    print("=" * 70)
    print(f"STRUCTURE channel run -- {args.org}: {len(ids):,} RefSeq ids")
    print("=" * 70)
    print("  Colab steps (run these cells):")
    print("  1) UniProt idmapping RefSeq_Protein->UniProtKB:")
    print(f"     POST https://rest.uniprot.org/idmapping/run  {idmapping_payload(ids[:2])} ...")
    print("  2) pull AFDB structures into queries/:")
    print(f"     for acc in <uniprot>: wget {afdb_url('<acc>')}")
    print("  3) foldseek databases Alphafold/UniProt50 afdb_clust tmp   # ~25GB once")
    print("     foldseek easy-search queries/ afdb_clust outputs/orphan/foldseek_result.m8 tmp \\")
    print("       --format-output 'query,target,fident,evalue,bits,alntmscore'")
    print("  4) write plddt.tsv (mean_plddt per query) + target_anno.tsv (target->product)")
    print("  5) python scripts/orphan_foldseek.py --real  # grades the .m8")
    print("\n  (this driver shapes the requests; the network/foldseek steps are Colab)")
    return 0


def run_smoke():
    print("=" * 70)
    print("STRUCTURE-RUN SMOKE -- URL, pLDDT parse, idmapping shaping")
    print("=" * 70)
    assert afdb_url("P0AD86") == \
        "https://alphafold.ebi.ac.uk/files/AF-P0AD86-F1-model_v4.pdb"
    print("  AFDB URL: OK")
    # synthetic AlphaFold PDB lines: two CA atoms, pLDDT 90 and 70 -> mean 80.
    # build with exact PDB columns so tempFactor lands in cols 61-66.
    def atom(serial, name, resseq, b):
        return (f"ATOM  {serial:>5} {name:<4} MET A{resseq:>4}    "
                f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{b:6.2f}          C")
    pdb = "\n".join([atom(1, "N", 1, 88.0), atom(2, "CA", 1, 90.0),
                     atom(3, "CA", 2, 70.0)])
    m = mean_plddt(pdb)
    print(f"  mean pLDDT parsed = {m:.1f} (expect 80.0)")
    assert abs(m - 80.0) < 1e-6, "pLDDT B-factor parse"
    p = idmapping_payload(["WP_1", "WP_2"])
    assert p["from"] == "RefSeq_Protein" and p["to"] == "UniProtKB" \
        and p["ids"] == "WP_1,WP_2"
    print("  idmapping payload: OK")
    print("\n=== STRUCTURE-RUN SMOKE PASS ===")
    print("  URL/pLDDT/idmapping plumbing validated. Pull+search run on Colab,")
    print("  then orphan_foldseek.py grades the .m8.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--org", default="beril_RalstoniaGMI1000")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
