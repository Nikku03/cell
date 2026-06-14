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
# Add PFAMTable.fst (the Pfam family route — faster than KEGG/genome.jp):
ZENODO_FILES = ["KOTable.fst", "PFAMTable.fst", "GTDB_bacterial.tree",
                "genomes.txt", "ko_pathway.tsv"]
GTDB_TAX_URL = ("https://data.gtdb.ecogenomic.org/releases/release226/226.0/"
                "bac120_taxonomy_r226.tsv.gz")
KOFAM_KO_LIST = "https://www.genome.jp/ftp/db/kofam/ko_list.gz"
KOFAM_PROFILES = "https://www.genome.jp/ftp/db/kofam/profiles.tar.gz"
EXPECTED_PROFILES_SIZE = 1_554_236_962  # verified bytes from genome.jp HEAD
# Pfam route (default — genome.jp KEGG mirror is heavily throttled at ~0.4 MB/s,
# EBI Pfam ships at >10 MB/s and Pfam covers more than KO for our purpose).
PFAM_HMM_URL = ("https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/"
                "Pfam-A.hmm.gz")
PFAM_HMM_DAT_URL = ("https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/"
                    "Pfam-A.hmm.dat.gz")
# choose route at module level; cell sets FLUX_FAMILY=pfam (default) or ko
FAMILY = os.environ.get("FLUX_FAMILY", "pfam").lower()
assert FAMILY in ("pfam", "ko"), f"FLUX_FAMILY must be pfam or ko, got {FAMILY}"
# Bump when conversion logic changes (forces parquet regen):
CONVERT_LOGIC_VERSION = 4


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
    if tax_gz.exists() and tax_gz.stat().st_size > 1000:
        print(f"  have GTDB taxonomy  ({tax_gz.stat().st_size/1e6:.1f} MB)")
    else:
        print("  fetching GTDB taxonomy ...")
        _stream(GTDB_TAX_URL, tax_gz)
    # KofamScan DB
    ko_list = KOFAM / "ko_list.gz"
    profiles = KOFAM / "profiles.tar.gz"
    if ko_list.exists() and ko_list.stat().st_size > 1000:
        print(f"  have ko_list  ({ko_list.stat().st_size/1e6:.1f} MB)")
    else:
        print("  fetching ko_list ..."); _stream(KOFAM_KO_LIST, ko_list)
    if FAMILY == "ko":
        if profiles.exists() and profiles.stat().st_size == EXPECTED_PROFILES_SIZE:
            print(f"  have KofamScan profiles  "
                  f"({profiles.stat().st_size/1e9:.2f} GB, full size)")
        else:
            if profiles.exists():
                actual = profiles.stat().st_size
                print(f"  profiles.tar.gz INCOMPLETE: {actual/1e6:.0f} MB / "
                      f"{EXPECTED_PROFILES_SIZE/1e6:.0f} MB; re-downloading")
                profiles.unlink()
            print("  fetching KofamScan profiles (1.5 GB, slow ~genome.jp) ...")
            _stream(KOFAM_PROFILES, profiles, expected=EXPECTED_PROFILES_SIZE)
            prof_dir = KOFAM / "profiles"
            if prof_dir.exists():
                import shutil; shutil.rmtree(prof_dir)
    else:
        # Pfam route: download Pfam-A.hmm from EBI (fast, ~1.5 GB)
        pfam = KOFAM / "Pfam-A.hmm.gz"
        pfam_dat = KOFAM / "Pfam-A.hmm.dat.gz"
        if not pfam.exists() or pfam.stat().st_size < 100_000_000:
            print("  fetching Pfam-A.hmm.gz from EBI (~1.5 GB, fast) ...")
            _stream(PFAM_HMM_URL, pfam)
        else:
            print(f"  have Pfam-A.hmm.gz  ({pfam.stat().st_size/1e9:.2f} GB)")
        if not pfam_dat.exists() or pfam_dat.stat().st_size < 100_000:
            print("  fetching Pfam-A.hmm.dat.gz from EBI ...")
            _stream(PFAM_HMM_DAT_URL, pfam_dat)
        else:
            print(f"  have Pfam-A.hmm.dat.gz")
    print("  stage 1 done")


# -------------------- stage 2: FST -> parquet via R --------------------------
def stage_convert():
    """KOTable.fst -> family_occurrence.parquet + species id list."""
    import pandas as pd
    occ_pq = GTDB / "family_occurrence.parquet"
    ver_stamp = GTDB / ".convert_version"
    have_ver = (int(ver_stamp.read_text())
                if ver_stamp.exists() and ver_stamp.read_text().strip().isdigit()
                else 0)
    if occ_pq.exists() and have_ver >= CONVERT_LOGIC_VERSION:
        # quick integrity check: species_id should have many unique values
        sample = pd.read_parquet(occ_pq, columns=["species_id"]).head(50_000)
        nu = sample.species_id.nunique()
        if nu > 100:
            print(f"  have {occ_pq} ({occ_pq.stat().st_size/1e6:.1f} MB, "
                  f"{nu:,} distinct species in 50k-row sample)")
            return
        print(f"  {occ_pq}: species_id has only {nu} unique values in sample "
              f"-> corrupt; regenerating")
        occ_pq.unlink()
    elif occ_pq.exists():
        print(f"  {occ_pq} predates convert v{CONVERT_LOGIC_VERSION}; "
              f"regenerating")
        occ_pq.unlink()
    fst_name = "PFAMTable.fst" if FAMILY == "pfam" else "KOTable.fst"
    fst = ZEN / fst_name
    tmp_tsv = ZEN / fst_name.replace(".fst", ".tsv")
    print(f"  family route: {FAMILY.upper()}  source: {fst_name}")
    if tmp_tsv.exists():
        # don't trust a TSV from a buggy prior run
        tmp_tsv.unlink()
    print("  R: FST -> TSV (long form, presence only) ...")
    # PhyloCorrelate's matrices store rows=species, cols=families, values
    # 0/1 (KOTable, integer) OR TRUE/FALSE (PFAMTable, logical).
    # Species IDs live in R row names. The `fst` package's read_fst() does
    # NOT restore row names; we use `read.fst()` from the package which
    # returns a data.frame with row names preserved, then promote them.
    r_script = f"""
    if (!require(fst, quietly=TRUE))
        install.packages('fst', repos='https://cloud.r-project.org')
    if (!require(data.table, quietly=TRUE))
        install.packages('data.table', repos='https://cloud.r-project.org')
    library(fst); library(data.table)
    # read.fst() restores row names; read_fst() (underscore) does not.
    x <- read.fst('{fst}')
    cat('dim:', nrow(x), ncol(x), '\\n')
    cat('col1 name:', names(x)[1], '\\n')
    col1_head <- as.character(x[[1]][1:min(5, nrow(x))])
    cat('col1 head:', paste(col1_head, collapse=','), '\\n')
    rn <- rownames(x)
    cat('rowname head:', paste(head(rn, 3), collapse=','), '\\n')

    has_species_col1 <- any(grepl('^(GB_|RS_|GCA_|GCF_)', col1_head))
    has_species_rn   <- !is.null(rn) && length(rn) == nrow(x) &&
        any(grepl('^(GB_|RS_|GCA_|GCF_)', head(rn, 5)))

    if (has_species_col1) {{
        idcol <- names(x)[1]
        cat('species_id source: col1 (', idcol, ')\\n')
    }} else if (has_species_rn) {{
        x$species_id <- rn
        idcol <- 'species_id'
        cat('species_id source: row names\\n')
    }} else {{
        # last resort: assume row index is the species, take from genomes.txt
        gen <- readLines('{ZEN / "genomes.txt"}')
        if (length(gen) == nrow(x)) {{
            x$species_id <- gen
            idcol <- 'species_id'
            cat('species_id source: genomes.txt (positional)\\n')
        }} else {{
            stop(paste('cannot locate species_id; col1 head:',
                       paste(col1_head, collapse=','),
                       'rn head:', paste(head(rn, 3), collapse=','),
                       'genomes.txt lines:', length(gen),
                       'matrix rows:', nrow(x)))
        }}
    }}

    xdt <- as.data.table(x)
    m <- melt(xdt, id.vars=idcol, variable.name='family_id',
              value.name='copies', variable.factor=FALSE)
    # values may be integer (0/1), numeric, or logical (TRUE/FALSE).
    m <- m[as.logical(copies) | copies > 0,
           .(species_id=get(idcol), family_id)]
    fwrite(m, '{tmp_tsv}', sep='\\t')
    cat('wrote', nrow(m), 'occurrence rows; distinct species:',
        length(unique(m$species_id)),
        'distinct families:', length(unique(m$family_id)), '\\n')
    """
    r = subprocess.run(["Rscript", "-e", r_script], capture_output=True,
                       text=True, timeout=900)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print("R STDERR:", r.stderr[-1500:])
        raise RuntimeError("R conversion failed")
    print(f"  TSV -> parquet ...")
    df = pd.read_csv(tmp_tsv, sep="\t", dtype=str)
    df.to_parquet(occ_pq, index=False)
    n_sp = df.species_id.nunique()
    print(f"  {occ_pq}: {len(df):,} rows  {n_sp:,} species  "
          f"{df.family_id.nunique():,} KOs")
    if n_sp < 1000:
        raise RuntimeError(
            f"converted parquet has only {n_sp} species (expected ~27k). "
            f"R FST conversion is reading the wrong column.")
    tmp_tsv.unlink()
    ver_stamp.write_text(str(CONVERT_LOGIC_VERSION))


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
    if out.exists() and out.stat().st_size > 1000:
        # quick sanity: should be tab-separated OG_header <TAB> K-id lines
        ok = False
        with open(out) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) == 2 and p[1].startswith("K") and p[0].startswith("OG"):
                    ok = True; break
        if ok:
            print(f"  have {out}  "
                  f"({sum(1 for _ in open(out)):,} OG->KO lines)")
            return
        print(f"  {out} exists but looks malformed; regenerating")
        out.unlink()
    fasta = GTDB / "og_reps.faa"
    if not fasta.exists():
        raise RuntimeError("run stage og_reps first")
    # ----- Pfam route: hmmsearch our OG reps against Pfam-A.hmm ----------
    if FAMILY == "pfam":
        pfam_gz = KOFAM / "Pfam-A.hmm.gz"
        pfam_hmm = KOFAM / "Pfam-A.hmm"
        if not pfam_hmm.exists() or pfam_hmm.stat().st_size < 1_000_000_000:
            print("  gunzip Pfam-A.hmm.gz ...")
            with gzip.open(pfam_gz, "rb") as fi, open(pfam_hmm, "wb") as fo:
                while True:
                    b = fi.read(1 << 20)
                    if not b: break
                    fo.write(b)
            print(f"  Pfam-A.hmm: {pfam_hmm.stat().st_size/1e9:.2f} GB")
        tbl = KOFAM / "pfam_hmmsearch.tbl"
        if not tbl.exists() or tbl.stat().st_size < 1000:
            ncpu = os.cpu_count() or 2
            print(f"  hmmsearch Pfam-A vs {fasta.name}  ({ncpu} cpu, ~30-60 min) ...")
            subprocess.run(["hmmsearch", "--cpu", str(ncpu),
                            "--tblout", str(tbl), "-o", "/dev/null",
                            str(pfam_hmm), str(fasta)], check=True)
            print(f"  tblout: {tbl.stat().st_size/1e6:.1f} MB")
        # Parse tblout: target = our OG header, query = Pfam name
        # Pfam-A.hmm queries are named like "ABC_transporter" but the
        # accession (PF00005.27) is in column 4. Use the accession.
        print("  parsing tblout -> best Pfam per OG ...")
        best = {}  # og_header -> (evalue, pfam_acc)
        with open(tbl) as f:
            for line in f:
                if line.startswith("#"): continue
                p = line.split()
                if len(p) < 5: continue
                og_hdr = p[0]
                pfam_acc = p[3]  # accession column for queries
                if not pfam_acc.startswith("PF"):
                    # fall back to query name if accession is "-"
                    pfam_acc = p[2]
                try: evalue = float(p[4])
                except ValueError: continue
                cur = best.get(og_hdr)
                if cur is None or evalue < cur[0]:
                    best[og_hdr] = (evalue, pfam_acc)
        # Pfam accessions in PhyloCorrelate's PFAMTable are versionless
        # ("PF00005" not "PF00005.27") -- strip the trailing ".N".
        with open(out, "w") as fo:
            for og_hdr, (_, acc) in best.items():
                fo.write(f"{og_hdr}\t{acc.split('.')[0]}\n")
        nseq = sum(1 for L in open(fasta) if L.startswith(">"))
        print(f"  wrote {out}: {len(best):,} of {nseq:,} OG reps got a Pfam")
        return
    # ----- KO route (original) -----------------------------------------
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
        print("  running exec_annotation (~hours) ...")
        subprocess.run(cmd, check=True)
        print(f"  wrote {out}")
        return
    if subprocess.run(["which", "hmmsearch"], capture_output=True).returncode != 0:
        raise RuntimeError(
            "Neither exec_annotation nor hmmsearch found. Install with: "
            "apt install hmmer  OR  conda install -c bioconda kofam_scan hmmer")
    # hmmsearch fallback: concatenate the prokaryote KO HMMs into one library
    # and run a single multi-threaded hmmsearch over it. Far faster and far
    # more robust than thousands of xargs invocations.
    hal = profiles_dir / "prokaryote.hal"
    if not hal.exists():
        raise RuntimeError("prokaryote.hal not found; check KofamScan extract")
    big = KOFAM / "prokaryote.hmm"
    if not big.exists() or big.stat().st_size < 50_000_000:
        print("  concatenating prokaryote KO HMMs into one library ...")
        n_in, n_miss = 0, 0
        with open(hal) as fi, open(big, "w") as fo:
            for line in fi:
                ko = line.strip().rstrip(".hmm")
                if not ko: continue
                p = profiles_dir / f"{ko}.hmm"
                if p.exists():
                    fo.write(p.read_text()); n_in += 1
                else:
                    n_miss += 1
        print(f"  {big.name}: {big.stat().st_size/1e6:.0f} MB  "
              f"({n_in:,} KOs in, {n_miss} missing)")
    tbl = KOFAM / "hmmsearch.tbl"
    if not tbl.exists() or tbl.stat().st_size < 1000:
        print(f"  hmmsearch (KO HMMs vs {fasta.name}, "
              f"{os.cpu_count() or 2} threads, ~1-2 h) ...")
        cmd = ["hmmsearch", "--cpu", str(os.cpu_count() or 2),
               "--tblout", str(tbl), "-o", "/dev/null",
               str(big), str(fasta)]
        subprocess.run(cmd, check=True)
        print(f"  tblout: {tbl.stat().st_size/1e6:.1f} MB")
    # Parse tblout, keep best KO per OG by E-value.
    # tblout columns: target_name accession query_name accession E-value score ...
    print("  parsing tblout -> best KO per OG ...")
    best = {}
    with open(tbl) as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.split()
            if len(p) < 5: continue
            og_hdr, ko = p[0], p[2]
            try:
                evalue = float(p[4])
            except ValueError:
                continue
            cur = best.get(og_hdr)
            if cur is None or evalue < cur[0]:
                best[og_hdr] = (evalue, ko)
    with open(out, "w") as fo:
        for og_hdr, (_, ko) in best.items():
            fo.write(f"{og_hdr}\t{ko}\n")
    print(f"  wrote {out}: {len(best):,} OG->KO assignments "
          f"(of {sum(1 for _ in open(fasta) if _.startswith('>')):,} OG reps)")


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
