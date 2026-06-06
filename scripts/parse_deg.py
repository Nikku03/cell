"""Parse DEG (Database of Essential Genes) bacterial bulk files into a
clean essential-gene table, and report overlap with our 59 organisms.

DEG ships two relevant files (semicolon-delimited, double-quoted):

  deg_annotation_p.csv  -- one row per ESSENTIAL gene (prokaryotes):
    deg_dataset_id; deg_gene_id; gene_name; GI; COG; function_class;
    description; organism; refseq; condition; (unused); GO_terms; uniprot

  deg_bacteria.csv      -- one row per dataset/organism (metadata):
    organism; reference...; PMID; refseq; n_essential; n_total; ...;
    method; condition; deg_dataset_id; date
    (variable field count due to multiple free-text references --
     parsed best-effort by token sniffing)

Note: DEG lists ONLY essential genes. To build a (locus_tag,
essential 0/1) label table for an organism you also need its full
gene complement (from the RefSeq GFF) and mark essential = in DEG.
That genome-join step is a separate script; this one just produces
the clean DEG essential-gene inventory + overlap report.

Outputs:
  deg_bacteria_essential_genes.csv  -- cleaned gene-level table
  deg_dataset_metadata.csv          -- cleaned per-organism metadata
  deg_overlap_report.json           -- overlap with our labels.csv orgs
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality"

GENE_COLS = ["deg_dataset", "deg_gene_id", "gene_name", "gi", "cog",
             "func_class", "description", "organism", "refseq",
             "condition", "_unused", "go_terms", "uniprot"]

# biological species for each of our organisms (for fuzzy DEG matching)
OUR_TO_SPECIES = {
    "syn3a": "mycoplasma mycoides", "mgen": "mycoplasma genitalium",
    "mpne": "mycoplasma pneumoniae", "mtub": "mycobacterium tuberculosis",
    "saur_NCTC8325_biotradis": "staphylococcus aureus",
    "spneT4": "streptococcus pneumoniae", "spne19F": "streptococcus pneumoniae",
    "ecoli_BW25113_tradis": "escherichia coli", "beril_Keio": "escherichia coli",
    "beril_Putida": "pseudomonas putida", "beril_BFirm": "burkholderia",
    "beril_SynE": "synechococcus elongatus", "beril_Btheta": "bacteroides thetaiotaomicron",
    "beril_DvH": "desulfovibrio vulgaris", "beril_MR1": "shewanella oneidensis",
    "beril_Caulo": "caulobacter", "beril_RalstoniaGMI1000": "ralstonia solanacearum",
    "beril_Koxy": "klebsiella", "reut": "cupriavidus", "reut_full": "cupriavidus",
    "beril_Smeli": "sinorhizobium meliloti", "beril_azobra": "azospirillum",
    "aeromonas": "aeromonas",
}


def norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\b(str\.?|strain|substr\.?|subsp\.?|sp\.?)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--deg-dir", type=Path,
                   default=Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/deg"),
                   help="dir holding deg_annotation_p.csv + deg_bacteria.csv")
    p.add_argument("--labels-csv", type=Path, default=DATA_DIR / "labels.csv")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="where to write cleaned outputs (default: --deg-dir)")
    args = p.parse_args()
    out_dir = args.out_dir or args.deg_dir

    try:
        import pandas as pd
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1

    anno = args.deg_dir / "deg_annotation_p.csv"
    meta = args.deg_dir / "deg_bacteria.csv"
    if not anno.exists():
        print(f"ERROR: {anno} not found. Download deg_annotation_p.csv.zip and "
              f"unzip into {args.deg_dir}", file=sys.stderr)
        return 1

    # ---- gene-level table ----
    print(f"Parsing {anno.name} ...")
    df = pd.read_csv(anno, sep=";", quotechar='"', header=None,
                      names=GENE_COLS, usecols=range(len(GENE_COLS)),
                      engine="python", on_bad_lines="skip")
    # strip whitespace/quotes
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.strip('"').str.strip()
    print(f"  total prokaryote essential-gene rows: {len(df):,}")

    # bacteria = DEG10xx datasets (archaea = DEG20xx)
    df["kingdom"] = df["deg_dataset"].str.extract(r"(DEG\d0)")
    bact = df[df["deg_dataset"].str.startswith("DEG10")].copy()
    print(f"  bacterial rows (DEG10xx): {len(bact):,}  "
          f"({bact.deg_dataset.nunique()} datasets, "
          f"{bact.organism.nunique()} organism strings)")

    keep = ["deg_dataset","deg_gene_id","gene_name","cog","func_class",
            "description","organism","refseq","condition","uniprot","gi"]
    bact_out = bact[keep]
    out_genes = out_dir / "deg_bacteria_essential_genes.csv"
    bact_out.to_csv(out_genes, index=False)
    print(f"  wrote {out_genes}  ({len(bact_out):,} rows)")

    # per-organism inventory
    print(f"\n=== DEG bacterial inventory (essential genes per dataset) ===")
    inv = (bact.groupby(["deg_dataset","organism"])
              .size().reset_index(name="n_essential")
              .sort_values("n_essential", ascending=False))
    for _, r in inv.head(30).iterrows():
        print(f"  {r.deg_dataset:<9s} {r.n_essential:>5d}  {r.organism}")
    print(f"  ... {len(inv)} datasets total")

    # ---- dataset metadata (best-effort token parse) ----
    meta_rows = []
    if meta.exists():
        print(f"\nParsing {meta.name} (best-effort) ...")
        for line in meta.read_text(errors="replace").splitlines():
            if not line.strip(): continue
            fields = [f.strip().strip('"') for f in line.split(";")]
            organism = fields[0]
            deg_id = next((f for f in fields if re.fullmatch(r"DEG10\d{2}", f)), None)
            refseq = next((f for f in fields if f.startswith("NC_")), None)
            # numeric fields: n_essential then n_total appear early
            nums = [int(f) for f in fields if re.fullmatch(r"\d{2,5}", f)]
            n_ess = nums[0] if len(nums) >= 1 else None
            n_tot = nums[1] if len(nums) >= 2 else None
            method = next((f for f in fields if f in
                            ("Single-gene knockout","Tn-seq","Antisense RNA",
                             "Genetic footprinting","Transposon mutagenesis",
                             "Insertion sequencing")), None)
            if deg_id:
                meta_rows.append({"deg_dataset": deg_id, "organism": organism,
                                   "refseq": refseq, "n_essential_reported": n_ess,
                                   "n_total_reported": n_tot, "method": method})
        meta_df = pd.DataFrame(meta_rows)
        out_meta = out_dir / "deg_dataset_metadata.csv"
        meta_df.to_csv(out_meta, index=False)
        print(f"  parsed {len(meta_df)} dataset metadata rows -> {out_meta}")

    # ---- overlap with our organisms ----
    print(f"\n=== overlap with our organisms ===")
    labels = pd.read_csv(args.labels_csv)
    our_orgs = sorted(labels.organism.unique())
    deg_orgs = sorted(bact.organism.dropna().unique())
    deg_norm = [(norm(o), o) for o in deg_orgs]

    overlap = []
    for our, sp in OUR_TO_SPECIES.items():
        nsp = norm(sp)
        hits = [orig for n_, orig in deg_norm if nsp in n_ or n_ in nsp]
        if hits:
            overlap.append({"ours": our, "species": sp, "deg_matches": hits})
            print(f"  {our:<28s} ({sp}) -> {hits}")
    matched_species = {norm(o["species"]) for o in overlap}
    novel = [o for n_, o in deg_norm
             if not any(ms in n_ or n_ in ms for ms in matched_species)]
    print(f"\n  DEG organisms NOT mapped to ours ({len(novel)} -- "
          f"training-augmentation candidates):")
    for o in novel:
        n_ess = int((bact.organism == o).sum())
        print(f"    {n_ess:>5d}  {o}")

    report = {
        "deg_bacterial_datasets":  int(bact.deg_dataset.nunique()),
        "deg_bacterial_organisms": len(deg_orgs),
        "deg_essential_genes":     int(len(bact)),
        "overlap_with_ours":       overlap,
        "n_overlap":               len(overlap),
        "novel_deg_organisms":     novel,
        "n_novel":                 len(novel),
    }
    out_report = out_dir / "deg_overlap_report.json"
    out_report.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_report}")
    print(f"\nSUMMARY: {len(bact):,} bacterial essential genes across "
          f"{bact.deg_dataset.nunique()} datasets; "
          f"{len(overlap)} overlap with ours, {len(novel)} novel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
