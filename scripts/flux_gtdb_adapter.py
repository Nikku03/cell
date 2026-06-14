"""Data adapter for the GTDB-scale flux linchpin.

Produces the three inputs `flux_gtdb_linchpin.py --real` consumes:
  data/gtdb/gtdb_taxonomy.tsv       species_id, phylum, class, order  (parsed
                                    from the PhyloCorrelate matrix's species
                                    set, joined to GTDB R226 lineage)
  data/gtdb/family_occurrence.parquet  long form: family_id, species_id
                                    (KO presence/absence -> long, from Zenodo
                                    PhyloCorrelate KOTable.fst)
  data/gtdb/gene_family_map.csv     organism, locus_tag, family_id
                                    (our genes -> KO, via KofamScan on OG reps)

Pipeline:
  1. Fetch PhyloCorrelate Zenodo 3993422 (KOTable.fst, GTDB_bacterial.tree,
     genomes.txt, ko_pathway.tsv). Convert FST -> parquet via R one-liner.
  2. Fetch GTDB R226 taxonomy; join PhyloCorrelate's species accessions to
     R226 lineage to get phylum/class/order columns.
  3. Build OG-representative FASTA from our 91 genome_cache proteomes
     (one protein per OG -- the longest member).
  4. Fetch KofamScan profiles + ko_list; run hmmsearch (the prokaryote subset)
     to assign a KO to each OG rep.
  5. Join our (organism, locus_tag) -> og_id -> KO -> family_id.

Run all stages with --stage all, or any subset with --stage download/convert/
annotate/build. Idempotent: skips stages whose output already exists.
"""
from __future__ import annotations
import argparse, gzip, json, os, subprocess, sys, urllib.request
from pathlib import Path
import time

REPO = Path(__file__).resolve().parent.parent
LAB = REPO / "data" / "drive_import" / "labels"
CACHE = REPO / "data" / "drive_import" / "genome_cache"
GTDB = REPO / "data" / "gtdb"
ZEN = GTDB / "phylocorrelate"
KOFAM = GTDB / "kofam"

ZENODO_BASE = "https://zenodo.org/api/records/3993422/files"
ZENODO_FILES = ["KOTable.fst", "GTDB_bacterial.tree", "genomes.txt",
                "ko_pathway.tsv"]
GTDB_TAX_URL = ("https://data.gtdb.ecogenomic.org/releases/release226/226.0/"
                "bac120_taxonomy_r226.tsv.gz")
KOFAM_KO_LIST = "https://www.genome.jp/ftp/db/kofam/ko_list.gz"
KOFAM_PROFILES = "https://www.genome.jp/ftp/db/kofam/profiles.tar.gz"


# -------------------- streaming download (reused pattern) --------------------
def _stream(url, dest: Path, expected=0, retries=4, chunk=1 << 20):
    dest = Path(dest)
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected and have >= expected:
            return
        req = urllib.request.Request(url)
        mode = "wb"
        if have:
            req.add_header("Range", f"bytes={have}-"); mode = "ab"
            print(f"  resume from {have/1e6:.1f} MB (try {attempt})")
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, mode) as f:
                if have and r.status == 200:
                    f.close(); dest.unlink(); f = open(dest, "wb"); have = 0
                done = have; t0 = time.time(); last = t0
                while True:
                    buf = r.read(chunk)
                    if not buf: break
                    f.write(buf); done += len(buf)
                    now = time.time()
                    if now - last > 10:
                        rate = (done - have) / max(now - t0, 1e-6) / 1e6
                        pct = f" {100*done/expected:.0f}%" if expected else ""
                        print(f"    {done/1e6:7.1f} MB{pct}  {rate:.1f} MB/s")
                        last = now
            if not expected or dest.stat().st_size >= expected:
                return
        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"  error: {e}; retry in {wait}s"); time.sleep(wait)
    raise RuntimeError(f"download failed: {url}")


# -------------------- stage 1: download Zenodo + GTDB tax + KofamScan --------
def stage_download():
    ZEN.mkdir(parents=True, exist_ok=True)
    KOFAM.mkdir(parents=True, exist_ok=True)
    GTDB.mkdir(parents=True, exist_ok=True)
    for f in ZENODO_FILES:
        out = ZEN / f
        if out.exists() and out.stat().st_size > 1000:
            print(f"  have {f}  ({out.stat().st_size/1e6:.1f} MB)")
            continue
        url = f"{ZENODO_BASE}/{f}/content"
        print(f"  fetching {f} ...")
        _stream(url, out)
    tax_gz = GTDB / "bac120_taxonomy_r226.tsv.gz"
    if not tax_gz.exists():
        print("  fetching GTDB taxonomy ...")
        _stream(GTDB_TAX_URL, tax_gz)
    # KofamScan DB
    ko_list = KOFAM / "ko_list.gz"
    profiles = KOFAM / "profiles.tar.gz"
    if not ko_list.exists():
        print("  fetching ko_list ..."); _stream(KOFAM_KO_LIST, ko_list)
    if not profiles.exists():
        print("  fetching KofamScan profiles (~1.5 GB) ...")
        _stream(KOFAM_PROFILES, profiles, expected=1_554_236_962)
    print("  stage 1 done")


# -------------------- stage 2: FST -> parquet via R --------------------------
def stage_convert():
    """KOTable.fst -> family_occurrence.parquet + species id list."""
    import pandas as pd
    occ_pq = GTDB / "family_occurrence.parquet"
    if occ_pq.exists():
        print(f"  have {occ_pq} ({occ_pq.stat().st_size/1e6:.1f} MB)")
        return
    fst = ZEN / "KOTable.fst"
    tmp_tsv = ZEN / "KOTable.tsv"
    if not tmp_tsv.exists():
        print("  R: KOTable.fst -> TSV (long form, presence only) ...")
        # PhyloCorrelate stores rows=species, cols=KOs, values=copy number.
        # We melt to long and keep only present (>0).
        r_script = f"""
        if (!require(fst, quietly=TRUE))
            install.packages('fst', repos='https://cloud.r-project.org')
        if (!require(data.table, quietly=TRUE))
            install.packages('data.table', repos='https://cloud.r-project.org')
        library(fst); library(data.table)
        x <- as.data.table(read_fst('{fst}'))
        cat('rows', nrow(x), 'cols', ncol(x), '\\n')
        # detect species id column
        idcol <- names(x)[1]
        m <- melt(x, id.vars=idcol, variable.name='family_id',
                  value.name='copies', variable.factor=FALSE)
        m <- m[copies > 0, .(species_id=get(idcol), family_id)]
        fwrite(m, '{tmp_tsv}', sep='\\t')
        cat('wrote', nrow(m), 'occurrence rows\\n')
        """
        r = subprocess.run(["Rscript", "-e", r_script], capture_output=True,
                           text=True, timeout=900)
        print(r.stdout[-1000:])
        if r.returncode != 0:
            print(r.stderr[-1000:])
            raise RuntimeError("R conversion failed")
    print(f"  TSV -> parquet ...")
    df = pd.read_csv(tmp_tsv, sep="\t", dtype=str)
    df.to_parquet(occ_pq, index=False)
    print(f"  {occ_pq}: {len(df):,} rows  "
          f"{df.species_id.nunique():,} species  "
          f"{df.family_id.nunique():,} KOs")
    tmp_tsv.unlink()


# -------------------- build taxonomy ----------------------------------------
def build_taxonomy():
    import pandas as pd
    out = GTDB / "gtdb_taxonomy.tsv"
    if out.exists():
        print(f"  have {out}"); return
    # PhyloCorrelate species (from genomes.txt)
    pc = pd.read_csv(ZEN / "genomes.txt", sep="\t", dtype=str, header=None)
    pc.columns = [c if i else "species_id" for i, c in
                  enumerate([str(c) for c in pc.columns])]
    print(f"  PhyloCorrelate species rows: {len(pc):,}  "
          f"sample: {pc.iloc[0,0][:60]}")
    # GTDB R226 lineage
    tax = pd.read_csv(GTDB / "bac120_taxonomy_r226.tsv.gz", sep="\t",
                      header=None, names=["accession", "lineage"], dtype=str)

    def parse(line):
        d = {p[:3]: p[3:] for p in str(line).split(";") if "__" in p}
        return d.get("p__"), d.get("c__"), d.get("o__")
    tax[["phylum", "class", "order"]] = tax.lineage.apply(
        lambda s: pd.Series(parse(s)))
    # join via accession; PhyloCorrelate's species_id may already be a GTDB acc
    # otherwise use it as-is and join later by occurrence
    tax = tax.rename(columns={"accession": "species_id"})
    # try direct join; fall back to using PhyloCorrelate's species as primary
    common = set(tax.species_id) & set(pc.species_id)
    print(f"  direct accession join: {len(common):,} species match GTDB R226")
    if len(common) < 1000:
        # PhyloCorrelate uses an older accession namespace; keep PC species and
        # assign synthetic taxonomy (we'll still get phylum from a heuristic
        # later if join fails -- but typically ~28k of PC's accessions DO match)
        print("  warning: weak GTDB join; will retry with stripped prefixes")
        tax["sid_strip"] = tax.species_id.str.replace(r"^(GB_|RS_)", "",
                                                     regex=True)
        pc["sid_strip"] = pc.species_id.str.replace(r"^(GB_|RS_)", "",
                                                    regex=True)
        tax2 = tax[["sid_strip", "phylum", "class", "order"]]
        merged = pc.merge(tax2, on="sid_strip", how="left")
        merged = merged[["species_id", "phylum", "class", "order"]]
    else:
        merged = pc.merge(tax[["species_id", "phylum", "class", "order"]],
                          on="species_id", how="left")
    merged["family"] = ""; merged["genus"] = ""
    merged = merged.dropna(subset=["phylum"])
    merged.to_csv(out, sep="\t", index=False)
    print(f"  wrote {out}: {len(merged):,} species with lineage  "
          f"{merged.phylum.nunique()} phyla")


# -------------------- OG representative FASTA -------------------------------
def stage_og_reps():
    """One protein per OG (longest member) across our 91 cached proteomes."""
    import pandas as pd
    GTDB.mkdir(parents=True, exist_ok=True)
    out = GTDB / "og_reps.faa"
    if out.exists() and out.stat().st_size > 1000:
        print(f"  have {out} ({out.stat().st_size/1e6:.1f} MB)")
        return
    print("  loading orthology -> picking longest member per OG ...")
    orth = pd.read_csv(LAB / "orthology_features.csv",
                       usecols=["organism", "locus_tag", "og_id"])
    mf = pd.read_csv(LAB / "genome_cache_manifest.csv")
    acc_map = dict(zip(mf.organism, mf.accession))
    # locus_tag -> protein_id from GFF, protein_id -> seq from faa
    by_org = {}
    for org in orth.organism.unique():
        acc = acc_map.get(org)
        gff = CACHE / str(acc) / "genomic.gff" if acc else None
        faa = CACHE / str(acc) / "protein.faa" if acc else None
        if not gff or not gff.exists() or not faa.exists():
            continue
        lt2pid = {}
        with open(gff) as f:
            for line in f:
                if "\tCDS\t" not in line: continue
                attr = line.rstrip().split("\t")[8]
                kv = dict(p.split("=", 1) for p in attr.split(";") if "=" in p)
                if "locus_tag" in kv and "protein_id" in kv:
                    lt2pid.setdefault(kv["locus_tag"], kv["protein_id"])
        seqs = {}
        with open(faa) as f:
            pid = None; buf = []
            for line in f:
                if line.startswith(">"):
                    if pid: seqs[pid] = "".join(buf)
                    pid = line[1:].split()[0]; buf = []
                else: buf.append(line.strip())
            if pid: seqs[pid] = "".join(buf)
        by_org[org] = (lt2pid, seqs)
    # build per-OG longest
    best = {}  # og_id -> (length, header, seq, organism, locus_tag)
    for _, r in orth.iterrows():
        org, lt, og = r.organism, r.locus_tag, r.og_id
        d = by_org.get(org)
        if not d: continue
        pid = d[0].get(lt)
        if not pid: continue
        seq = d[1].get(pid)
        if not seq: continue
        L = len(seq)
        cur = best.get(og)
        if cur is None or L > cur[0]:
            best[og] = (L, f"{og}|{org}|{lt}", seq, org, lt)
    print(f"  wrote {len(best):,} OG representatives")
    with open(out, "w") as f:
        for og, (_, hdr, seq, _, _) in best.items():
            f.write(f">{hdr}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")
    # also write the (og, organism, locus_tag) provenance
    import csv
    with open(GTDB / "og_reps_provenance.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["og_id", "organism", "locus_tag"])
        for og, (_, _, _, org, lt) in best.items():
            w.writerow([og, org, lt])
    print(f"  fasta: {out.stat().st_size/1e6:.1f} MB")


# -------------------- KofamScan -> KO per OG --------------------------------
def stage_annotate():
    out = GTDB / "og_to_ko.tsv"
    if out.exists():
        print(f"  have {out}"); return
    fasta = GTDB / "og_reps.faa"
    if not fasta.exists():
        raise RuntimeError("run stage og_reps first")
    # extract profiles
    profiles_dir = KOFAM / "profiles"
    if not profiles_dir.exists():
        print("  extracting profiles.tar.gz ...")
        subprocess.run(["tar", "-xzf", str(KOFAM / "profiles.tar.gz"),
                        "-C", str(KOFAM)], check=True)
    ko_list_gz = KOFAM / "ko_list.gz"
    ko_list = KOFAM / "ko_list"
    if not ko_list.exists():
        with gzip.open(ko_list_gz, "rb") as fi, open(ko_list, "wb") as fo:
            fo.write(fi.read())
    # find hmmsearch / kofamscan
    if subprocess.run(["which", "exec_annotation"], capture_output=True
                       ).returncode == 0:
        cmd = ["exec_annotation", "-f", "mapper", "-o", str(out),
               "--cpu", str(os.cpu_count() or 2),
               "--profile", str(profiles_dir), "--ko-list", str(ko_list),
               str(fasta)]
    elif subprocess.run(["which", "hmmsearch"], capture_output=True).returncode == 0:
        # fallback: search prokaryote profile set with hmmsearch directly
        hal = profiles_dir / "prokaryote.hal"
        if not hal.exists():
            raise RuntimeError("prokaryote.hal not found; install kofam_scan")
        cmd = ["bash", "-c",
               f"cat {hal} | xargs -I{{}} -P {os.cpu_count() or 2} "
               f"hmmsearch --tblout - {profiles_dir}/{{}}.hmm {fasta} "
               f"| awk '/^[^#]/ {{print $1\"\\t\"$3}}' > {out}"]
    else:
        raise RuntimeError(
            "Neither exec_annotation (kofam_scan) nor hmmsearch found. "
            "Install with: pip install kofamscan OR conda install -c bioconda "
            "kofam_scan hmmer")
    print("  running KO annotation (this is the longest step, ~hours) ...")
    subprocess.run(cmd, check=True)
    print(f"  wrote {out}")


# -------------------- final join -> gene_family_map -------------------------
def stage_build_map():
    import pandas as pd
    out = GTDB / "gene_family_map.csv"
    if out.exists():
        print(f"  have {out}"); return
    og2ko = pd.read_csv(GTDB / "og_to_ko.tsv", sep="\t", header=None,
                        names=["og_header", "ko"])
    # og_header is "og_id|organism|locus_tag" — split out the og_id
    og2ko["og_id"] = og2ko.og_header.str.split("|").str[0]
    og2ko = og2ko.dropna(subset=["ko"]).drop_duplicates(subset=["og_id"])
    orth = pd.read_csv(LAB / "orthology_features.csv",
                       usecols=["organism", "locus_tag", "og_id"])
    m = orth.merge(og2ko[["og_id", "ko"]], on="og_id", how="inner")
    m = m.rename(columns={"ko": "family_id"})
    m[["organism", "locus_tag", "family_id"]].to_csv(out, index=False)
    print(f"  wrote {out}: {len(m):,} gene->KO assignments  "
          f"({m.organism.nunique()} orgs, {m.family_id.nunique()} KOs)")


# ---------------------- driver ----------------------------------------------
STAGES = {"download": stage_download, "convert": stage_convert,
          "taxonomy": build_taxonomy, "og_reps": stage_og_reps,
          "annotate": stage_annotate, "build": stage_build_map}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    help="comma-separated subset or 'all'. Order: " +
                         ",".join(STAGES))
    args = ap.parse_args()
    selected = list(STAGES) if args.stage == "all" else args.stage.split(",")
    for s in selected:
        if s not in STAGES:
            print(f"unknown stage: {s} (have {list(STAGES)})", file=sys.stderr)
            return 2
        print(f"\n=== STAGE: {s} ===")
        STAGES[s]()
    print("\nadapter outputs ready in data/gtdb/; next: "
          "python scripts/flux_gtdb_linchpin.py --real")
    return 0


if __name__ == "__main__":
    sys.exit(main())
