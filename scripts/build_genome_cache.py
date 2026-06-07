"""Build a complete genome + annotation cache for all organisms we have
essentiality data for (our 50 bridgeable BERIL/lab strains + the DEG
bacterial datasets).

Design choices:
  * Download via the NCBI Datasets API download-by-accession endpoint
    (needs only the GCF/GCA accession -- no assembly-name guessing,
    which is what 404'd the FTP approach). One request yields genome
    FASTA + GFF + protein FASTA.
  * Resolve accessions in priority order:
        1. explicit override CSV (organism,accession)
        2. SEED table below (confident, hand-verified accessions)
        3. already-cached GFF on disk (filename -> accession)
        4. NCBI species search (Datasets API taxon endpoint), picking
           the reference/representative assembly
  * VERIFY every download by computing the locus_tag join-rate against
    labels.csv. A wrong assembly -> near-zero join -> flagged, so we
    never silently cache the wrong genome.
  * Resumable: skips accessions already extracted.

Numeric-ID organisms (beril_Keio, DvH, MR1, Btheta, ANA3, Dino, Miya,
PV4, SB2B) are downloaded for completeness but flagged join_blocked
because their MicrobesOnline numeric locus IDs can't bridge to RefSeq
without BERIL's source sysName table.

Outputs (under --cache-dir, default on Drive):
  <accession>/  extracted genome.fna, genomic.gff, protein.faa
  genome_cache_manifest.csv   per-organism: accession, n_cds, join_rate, status
"""
from __future__ import annotations
import argparse, csv, gzip, io, json, sys, time, zipfile
import urllib.request, urllib.error, urllib.parse
from pathlib import Path

DATASETS_API = "https://api.ncbi.nlm.nih.gov/datasets/v2"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ---- confident, hand-verified accessions (organism -> GCF/GCA) ----
SEED_ACC = {
    # already cached on Drive (accession known from filename)
    "mtub":                    "GCF_000195955.2",
    "saur_NCTC8325_biotradis": "GCF_000013425.1",
    "beril_Caulo":             "GCF_000006905.1",   # C. crescentus NA1000 (CCNA_)
    "beril_RalstoniaGMI1000":  "GCF_000009125.1",
    "beril_SynE":              "GCF_000012525.1",   # Synechococcus elongatus PCC 7942
    "ecoli_BW25113_tradis":    "GCF_000750555.1",
    # from codon_features_config (verified join earlier)
    "beril_Putida":            "GCF_000007565.2",   # P. putida KT2440 (PP_)
    "beril_BFirm":             "GCF_000019365.1",   # Burkholderia phytofirmans PsJN (BPHYT_)
    "mgen":                    "GCF_000027325.1",
    "mpne":                    "GCF_000027345.1",
    "spneT4":                  "GCF_000006885.1",   # S. pneumoniae TIGR4 (SP_)
    "syn3a":                   "GCA_000968075.1",
    # identifiable from locus-tag prefix / species
    "beril_Smeli":             "GCF_000006965.1",   # Sinorhizobium meliloti 1021 (SMc)
    "beril_SyringaeB728a":     "GCF_000012245.1",   # P. syringae pv. syringae B728a (Psyr_)
    "beril_SyringaeB728a_mexBdelta": "GCF_000012245.1",
    "reut":                    "GCF_000009285.1",   # Cupriavidus necator H16 (H16_)
    "reut_full":               "GCF_000009285.1",
}

# species names for NCBI taxon-search fallback (organism -> species string)
SPECIES = {
    "aeromonas":               "Aeromonas",
    "beril_Burk376":           "Burkholderia sp. 376MFSha3.1",
    "beril_Cola":              "Echinicola vietnamensis",
    "beril_Cup4G11":           "Cupriavidus basilensis 4G11",
    "beril_Dda3937":           "Dickeya dadantii 3937",
    "beril_Ddia6719":          "Dickeya dianthicola",
    "beril_DdiaME23":          "Dickeya dianthicola ME23",
    "beril_Dyella79":          "Dyella japonica UNC79MFTsu3.2",
    "beril_HerbieS":           "Herbaspirillum seropedicae SmR1",
    "beril_Kang":              "Kangiella aquimarina",
    "beril_Korea":             "Pseudomonas fluorescens",
    "beril_Koxy":              "Klebsiella michiganensis",
    "beril_Magneto":           "Magnetospirillum magneticum AMB-1",
    "beril_Methanococcus_JJ":  "Methanococcus maripaludis",
    "beril_Methanococcus_S2":  "Methanococcus maripaludis S2",
    "beril_PS":                "Dechlorosoma suillum PS",
    "beril_Pedo557":           "Pedobacter sp. GW460-11-11-14-LB5",
    "beril_Phaeo":             "Phaeobacter inhibens BS107",
    "beril_Ponti":             "Pontibacter actiniarum",
    "beril_RalstoniaBSBF1503": "Ralstonia solanacearum",
    "beril_RalstoniaPSI07":    "Ralstonia solanacearum PSI07",
    "beril_RalstoniaUW163":    "Ralstonia solanacearum UW163",
    "beril_acidovorax_3H11":   "Acidovorax sp. GW101-3H11",
    "beril_azobra":            "Azospirillum brasilense",
    "beril_psRCH2":            "Pseudomonas stutzeri RCH2",
    "beril_pseudo13_GW456_L13":"Pseudomonas fluorescens GW456-L13",
    "beril_pseudo1_N1B4":      "Pseudomonas fluorescens N1B4",
    "beril_pseudo3_N2E3":      "Pseudomonas fluorescens N2E3",
    "beril_pseudo5_N2C3_1":    "Pseudomonas fluorescens N2C3",
    "beril_pseudo6_N2E2":      "Pseudomonas fluorescens N2E2",
    "spne19F":                 "Streptococcus pneumoniae Taiwan19F-14",
}

NUMERIC_BLOCKED = {"beril_Keio","beril_DvH","beril_MR1","beril_Btheta",
                    "beril_ANA3","beril_Dino","beril_Miya","beril_PV4","beril_SB2B"}


def http_get(url: str, timeout=120) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent":"genome-cache/1.0",
                                                 "Accept":"application/zip, application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            if attempt == 2:
                print(f"      GET failed ({e}) for {url[:90]}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def _eutils_json(url: str):
    time.sleep(0.4)  # NCBI rate limit (3/s without key)
    data = http_get(url, timeout=40)
    if not data: return None
    try:
        return json.loads(data)
    except Exception:
        return None


def eutils_to_assembly(term: str, db: str = "nuccore") -> str | None:
    """Resolve an exact identifier (locus_tag or NC accession) to the
    GCF/GCA assembly that owns it, via esearch -> elink -> esummary.
    Prefers the RefSeq (GCF) accession; falls back to GCA."""
    # 1. esearch for the term
    s = _eutils_json(f"{EUTILS}/esearch.fcgi?db={db}"
                     f"&term={urllib.parse.quote(term)}&retmax=1&retmode=json")
    if not s: return None
    ids = s.get("esearchresult", {}).get("idlist", [])
    if not ids: return None
    uid = ids[0]
    # 2. elink nuccore -> assembly
    e = _eutils_json(f"{EUTILS}/elink.fcgi?dbfrom={db}&db=assembly"
                     f"&id={uid}&retmode=json")
    if not e: return None
    asm_uid = None
    for ls in e.get("linksets", []):
        for db_block in ls.get("linksetdbs", []):
            if db_block.get("dbto") == "assembly" and db_block.get("links"):
                asm_uid = db_block["links"][0]; break
        if asm_uid: break
    if not asm_uid: return None
    # 3. esummary assembly -> accession (prefer refseq)
    su = _eutils_json(f"{EUTILS}/esummary.fcgi?db=assembly"
                      f"&id={asm_uid}&retmode=json")
    if not su: return None
    rec = su.get("result", {}).get(asm_uid, {})
    syn = rec.get("synonym", {})
    return syn.get("refseq") or rec.get("assemblyaccession") or syn.get("genbank")


def clean_species(s: str) -> str:
    """Reduce a strain-suffixed organism string to genus+species for taxon
    search fallback ('Staphylococcus aureus N315' -> 'Staphylococcus aureus')."""
    toks = s.replace("/", " ").split()
    return " ".join(toks[:2]) if len(toks) >= 2 else s


def resolve_by_species(species: str) -> str | None:
    """Query NCBI Datasets taxon endpoint, return a reference/representative
    GCF accession for the species (best-effort)."""
    url = (f"{DATASETS_API}/genome/taxon/{urllib.parse.quote(species)}/dataset_report"
           f"?filters.assembly_source=refseq&page_size=20")
    data = http_get(url, timeout=60)
    if not data: return None
    try:
        rep = json.loads(data)
    except Exception:
        return None
    reports = rep.get("reports", [])
    if not reports: return None
    # prefer reference genome, then representative, then first
    def rank(r):
        lvl = r.get("assembly_info", {}).get("refseq_category", "")
        order = {"reference genome":0, "representative genome":1}.get(lvl, 2)
        # also prefer complete genomes
        status = r.get("assembly_info", {}).get("assembly_level", "")
        order2 = 0 if status == "Complete Genome" else 1
        return (order, order2)
    reports.sort(key=rank)
    return reports[0].get("accession")


def download_genome(accession: str, dest_dir: Path) -> dict | None:
    """Download genome+gff+protein via Datasets API into dest_dir/<accession>/.
    Returns dict of extracted file paths, or None on failure."""
    out = dest_dir / accession
    fna = out / "genome.fna"; gff = out / "genomic.gff"; faa = out / "protein.faa"
    if fna.exists() and gff.exists():
        return {"fna": fna, "gff": gff, "faa": faa if faa.exists() else None,
                "cached": True}
    out.mkdir(parents=True, exist_ok=True)
    url = (f"{DATASETS_API}/genome/accession/{accession}/download"
           f"?include_annotation_type=GENOME_FASTA"
           f"&include_annotation_type=GENOME_GFF"
           f"&include_annotation_type=PROT_FASTA")
    blob = http_get(url, timeout=300)
    if not blob:
        return None
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        print(f"      not a zip (likely API error): {blob[:120]!r}")
        return None
    # locate files inside ncbi_dataset/data/<acc>/
    got = {"fna": None, "gff": None, "faa": None, "cached": False}
    for name in zf.namelist():
        low = name.lower()
        if low.endswith(".fna") and "cds" not in low and got["fna"] is None:
            fna.write_bytes(zf.read(name)); got["fna"] = fna
        elif low.endswith(".gff"):
            gff.write_bytes(zf.read(name)); got["gff"] = gff
        elif low.endswith(".faa"):
            faa.write_bytes(zf.read(name)); got["faa"] = faa
    return got if got["gff"] else None


def gff_locus_tags(gff_path: Path) -> set[str]:
    tags = set()
    op = gzip.open if str(gff_path).endswith(".gz") else open
    with op(gff_path, "rt") as f:
        for line in f:
            if line.startswith("#") or "\tCDS\t" not in line: continue
            attr = line.rstrip("\n").split("\t")[-1]
            for kv in attr.split(";"):
                if kv.startswith("locus_tag=") or kv.startswith("old_locus_tag="):
                    tags.add(kv.split("=",1)[1])
    return tags


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labels-csv",  type=Path, required=True)
    p.add_argument("--deg-csv",     type=Path, default=None,
                   help="deg_bacteria_essential_genes.csv (adds DEG datasets)")
    p.add_argument("--cache-dir",   type=Path, required=True)
    p.add_argument("--override-csv",type=Path, default=None,
                   help="optional organism,accession overrides")
    p.add_argument("--only",        nargs="*", default=None,
                   help="restrict to these organism names (for staged runs)")
    p.add_argument("--skip-species-search", action="store_true",
                   help="don't hit NCBI taxon search for unresolved orgs")
    p.add_argument("--manifest",    type=Path, default=None)
    args = p.parse_args()

    try:
        import pandas as pd
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.cache_dir / "genome_cache_manifest.csv")

    labels = pd.read_csv(args.labels_csv)
    our_orgs = sorted(labels.organism.unique())

    overrides = {}
    if args.override_csv and args.override_csv.exists():
        for row in csv.DictReader(open(args.override_csv)):
            overrides[row["organism"]] = row["accession"]

    # build the work list: [name, accession_or_None, kind, exact_key, species]
    #   exact_key = a sample locus_tag (ours) or NC accession (DEG) for
    #               eutils resolution; species = fallback taxon string.
    work = []
    for org in our_orgs:
        acc = overrides.get(org)   # SEED is a fallback after eutils, set in loop
        sample_lt = None
        if org not in NUMERIC_BLOCKED:
            lts = labels[labels.organism == org].locus_tag.astype(str)
            sample_lt = lts.iloc[0] if len(lts) else None
        work.append([org, acc, "ours", sample_lt, SPECIES.get(org)])

    # DEG datasets: resolve via the NC accession in the data (exact),
    # falling back to genus+species of the organism string.
    if args.deg_csv and args.deg_csv.exists():
        deg = pd.read_csv(args.deg_csv)
        for ds, g in deg.groupby("deg_dataset"):
            org_str = str(g.organism.dropna().iloc[0]) if g.organism.notna().any() else ds
            nc = None
            if "refseq" in g.columns and g.refseq.notna().any():
                nc = str(g.refseq.dropna().iloc[0]).split()[0]  # first NC_ if multi
            name = f"DEG:{ds}:{org_str[:40]}"
            work.append([name, overrides.get(name), "deg", nc,
                          clean_species(org_str)])

    if args.only:
        work = [w for w in work if w[0] in set(args.only)]

    print(f"Genome cache build: {len(work)} targets "
          f"({sum(1 for w in work if w[2]=='ours')} ours, "
          f"{sum(1 for w in work if w[2]=='deg')} DEG)")

    manifest = []
    for i, (org, acc, kind, exact_key, species) in enumerate(work, 1):
        t0 = time.time()
        blocked = org in NUMERIC_BLOCKED
        # resolution priority: override/seed (acc) -> exact eutils key ->
        # species taxon search.
        if not acc and exact_key:
            acc = eutils_to_assembly(exact_key, db="nuccore")
            if acc:
                print(f"  [{i}/{len(work)}] {org}: resolved {exact_key} -> {acc}")
        if not acc:  # SEED fallback when eutils fails
            acc = SEED_ACC.get(org)
            if acc:
                print(f"  [{i}/{len(work)}] {org}: seed accession -> {acc}")
        if not acc and species and not args.skip_species_search:
            acc = resolve_by_species(species)
            if acc:
                print(f"  [{i}/{len(work)}] {org}: resolved species '{species}' -> {acc}")
        if not acc:
            print(f"  [{i}/{len(work)}] {org}: NO ACCESSION (unresolved)")
            manifest.append({"organism":org,"kind":kind,"accession":None,
                              "n_cds":0,"join_rate":None,
                              "status":"no_accession"})
            continue

        got = download_genome(acc, args.cache_dir)
        if not got:
            print(f"  [{i}/{len(work)}] {org}: DOWNLOAD FAILED ({acc})")
            manifest.append({"organism":org,"kind":kind,"accession":acc,
                              "n_cds":0,"join_rate":None,"status":"download_failed"})
            continue

        gff_tags = gff_locus_tags(got["gff"])
        # join rate vs our labels (only meaningful for 'ours')
        join_rate = None; n_lab = 0
        if kind == "ours":
            lab_tags = set(labels[labels.organism==org].locus_tag.astype(str))
            n_lab = len(lab_tags)
            if n_lab:
                join_rate = len(lab_tags & gff_tags) / n_lab
        status = ("join_blocked_numeric" if blocked else
                  "ok" if (join_rate is None or join_rate >= 0.5) else
                  "low_join_wrong_assembly")
        cached_flag = " (cached)" if got.get("cached") else ""
        print(f"  [{i}/{len(work)}] {org}: {acc}  CDS_tags={len(gff_tags)}  "
              f"join={join_rate if join_rate is not None else 'n/a'}"
              f"{'' if join_rate is None else f' ({join_rate*100:.0f}%)'}  "
              f"faa={'Y' if got.get('faa') else 'N'}  {status}{cached_flag}  "
              f"{time.time()-t0:.0f}s")
        manifest.append({"organism":org,"kind":kind,"accession":acc,
                          "n_cds":len(gff_tags),
                          "join_rate":round(join_rate,4) if join_rate is not None else None,
                          "has_protein_faa": bool(got.get("faa")),
                          "status":status})

    mf = pd.DataFrame(manifest)
    mf.to_csv(manifest_path, index=False)
    print(f"\nwrote {manifest_path}")
    # summary
    print(f"\n=== SUMMARY ===")
    print(mf.status.value_counts().to_string())
    ok_join = mf[(mf.kind=="ours") & (mf.join_rate.notna())]
    if len(ok_join):
        good = (ok_join.join_rate >= 0.5).sum()
        print(f"\n  ours with join >= 50%: {good}/{len(ok_join)}")
        print(f"  median join_rate (ours): {ok_join.join_rate.median():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
