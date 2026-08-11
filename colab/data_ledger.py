"""THE DATA LEDGER -- every external input, where it came from, and how to get it back.

THE PROBLEM THIS EXISTS FOR, stated from what actually happened.  Four inputs that nexus_txn needs were
absent from this container: AvgKdegs.csv, k562_atac.bed.gz, _abcfeat.json, lambert_tf_symbols.json. The
module could be read but not run, and its recorded numbers could not be checked. Finding all four again
took a Zenodo search, three ENCODE API queries, a GEO sample-level crawl and a rebuild of the ABC
aggregation from a different predictor. That knowledge currently lives in commit messages. If this
container is reclaimed, the next session repeats every step of it.

Separately: cell_complete.json is 38 MB, read by 227 modules, and EXPLICITLY GITIGNORED. When I patched
it this morning there was no version-controlled original to compare against, and I could only verify the
edit was additive because an unrelated .gz from an earlier date happened to be sitting next to it.

Both are the same failure. The repo depends on files it does not describe.

WHAT A LEDGER ENTRY RECORDS
    raw       a URL or accession that can be fetched again, with the content hash it must have
    derived   the RAW inputs it is built from and the TRANSFORM FUNCTION, recorded as code in this file
              rather than as prose, so it can be re-executed rather than re-implemented
Both carry a sha256 and a byte count. A file whose hash has drifted is not the file the recorded results
were produced from, and the ledger says so instead of letting the run proceed.

THE CHECK THAT MAKES THIS MORE THAN A LIST.  A derived entry must REBUILD TO THE SAME HASH from its raw
inputs. If _abcfeat.json rebuilt from re2g_k562.bed.gz does not reproduce 20f7aa4ae6040e16, then the
transform is not deterministic, or it is not the transform that produced the file, and either way the
provenance claim is false. This is checkable and it can fail, which is the whole difference between a
ledger and a README.

PREDECLARED, before any number:
    every declared file present with a matching hash
        -> the ledger describes this container accurately.
    a hash mismatch
        -> the file has drifted from what the recorded results used. Report it; do NOT silently accept.
    a derived file rebuilds to a different hash
        -> the recorded transform is not the one that produced it. That is a defect in the ledger and it
           is reported as such rather than by adjusting the expected hash to match.
    a raw source 404s
        -> the input is no longer retrievable and the results depending on it are no longer reproducible
           by anyone. Worth knowing loudly.

-> outputs/data_ledger.json
"""
import csv
import gzip
import hashlib
import json
import os
import statistics
import subprocess
import sys
import collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
SC = ROOT / "outputs" / "orphan" / "supercoil"
SCRATCH = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")


# ---- transforms, recorded as code because prose cannot be re-executed --------------------------------
def build_abcfeat(src, dest):
    """rE2G thresholded element-gene links -> the five per-gene fields nexus_txn's L3 reads.

    The original _abcfeat.json came from ABC and has no producer in this repo. This rebuilds the same
    SHAPE from ENCODE-rE2G, a different predictor: 87,490 K562 links over 14,465 genes against the
    original 12,438. L3 measured 0.190 against a recorded 0.1878, so the substitution is sound, but it is
    a substitution and the ledger records which.
    """
    rows = collections.defaultdict(list)
    with gzip.open(src, "rt") as f:
        h = f.readline().lstrip("#").rstrip().split("\t")
        ix = {c: i for i, c in enumerate(h)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(h):
                continue
            try:
                rows[p[ix["TargetGene"]]].append((float(p[ix["ABC.Score.Feature"]]),
                                                  float(p[ix["distanceToTSS.Feature"]]),
                                                  p[ix["isSelfPromoter"]].upper() == "TRUE"))
            except ValueError:
                continue
    abc = {}
    for g, v in rows.items():
        s = [x[0] for x in v]
        d = [x[1] for x in v]
        abc[g] = {"n": len(v), "ssum": float(sum(s)), "smax": float(max(s)),
                  "dmed": float(statistics.median(d)), "ndistal": int(sum(1 for x in v if not x[2]))}
    json.dump(abc, open(dest, "w"))
    return len(abc)


def build_lambert(src, dest):
    """Lambert et al. 2018 v1.01 -> the flat symbol list nexus_txn's L4 filters regulators against."""
    rows = list(csv.DictReader(open(src)))
    tfs = sorted({r["HGNC symbol"] for r in rows
                  if r.get("Is TF?", "").strip().lower() == "yes" and r.get("HGNC symbol")})
    json.dump(tfs, open(dest, "w"))
    return len(tfs)


LEDGER = [
    {"name": "AvgKdegs.csv", "path": SP / "rnadecay" / "AvgKdegs.csv", "kind": "raw",
     "sha256": "b8869c749b2b4a24", "bytes": 21962834,
     "url": "https://zenodo.org/records/16884513/files/AvgKdegs_genes_v1.1.csv?download=1",
     "source": "RNADecayCafe v1.1, DOI 10.5281/zenodo.16884513",
     "why": "target for nexus_txn: k_syn = avg_RPKM_total * avg_kdeg, 8,867 usable K562 genes"},
    {"name": "k562_atac.bed.gz", "path": SP / "k562_atac.bed.gz", "kind": "raw",
     "sha256": "2f50c1a82609dccb", "bytes": 1268446,
     "url": "https://www.encodeproject.org/files/ENCFF926KTI/@@download/ENCFF926KTI.bed.gz",
     "source": "ENCODE ENCFF926KTI, K562 ATAC-seq IDR thresholded peaks, GRCh38, 55,575 peaks",
     "why": "nexus_txn L2 promoter accessibility. NOT the original file, which is lost"},
    {"name": "re2g_k562.bed.gz", "path": SP / "re2g_k562.bed.gz", "kind": "raw",
     "sha256": "a853c2745e1b4cc5", "bytes": 3744436,
     "url": "https://www.encodeproject.org/files/ENCFF976OKL/@@download/ENCFF976OKL.bed.gz",
     "source": "ENCODE-rE2G ENCSR627ANP / ENCFF976OKL, K562 thresholded element-gene links, 87,490",
     "why": "raw source for _abcfeat.json"},
    {"name": "lambert.csv", "path": SP / "lambert.csv", "kind": "raw",
     "sha256": "5438940ddef71b6f", "bytes": 2107299,
     "url": "http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv",
     "source": "Lambert et al. 2018, Human TFs database v1.01",
     "why": "raw source for lambert_tf_symbols.json"},
    {"name": "ctcf_gw.bed.gz", "path": SP / "ctcf_gw.bed.gz", "kind": "raw",
     "sha256": "39d4e2566f1d008c", "bytes": 776906,
     "url": "https://www.encodeproject.org/files/ENCFF362OPG/@@download/ENCFF362OPG.bed.gz",
     "source": "ENCODE ENCFF362OPG, K562 CTCF IDR thresholded, ENCSR000AKO, GRCh38, 44,104 peaks",
     "why": "the 0.663 CRISPR bar. NOT the original file; rebuilt bar reads 0.6737 against 0.6632"},
    {"name": "GSM8523629_supercoil_rep1_hg38.bw", "path": SC / "GSM8523629_supercoil_rep1_hg38.bw",
     "kind": "raw", "sha256": "5a22b743af5798b4", "bytes": 1437675,
     "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8523nnn/GSM8523629/suppl/"
            "GSM8523629_supercoil_rep1_hg38.bw",
     "source": "GEO GSE277502, bTMP-seq hTERT-RPE1, hg38, 20 kb bins",
     "why": "torsion attempt 1"},
    {"name": "GSM8523630_supercoil_rep2_hg38.bw", "path": SC / "GSM8523630_supercoil_rep2_hg38.bw",
     "kind": "raw", "sha256": "b74136ed78bfb469", "bytes": 1437304,
     "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8523nnn/GSM8523630/suppl/"
            "GSM8523630_supercoil_rep2_hg38.bw",
     "source": "GEO GSE277502 replicate 2", "why": "replicate-concordance control, r = 0.796"},
    {"name": "GSM8523628_RPE1_bTMP_genomic_control_hg38.bw",
     "path": SC / "GSM8523628_RPE1_bTMP_genomic_control_hg38.bw", "kind": "raw",
     "sha256": "c4cfe8f1ec137a6e", "bytes": 1143230,
     "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8523nnn/GSM8523628/suppl/"
            "GSM8523628_RPE1_bTMP_genomic_control_hg38.bw",
     "source": "GEO GSE277502 naked genomic DNA control",
     "why": "the matched control GSE285699 does not ship"},
    {"name": "GSM8706920_1_RPKM.bw", "path": SC / "GSM8706920_1_RPKM.bw", "kind": "raw",
     "sha256": "4e7bf678aa47b232", "bytes": 107688596,
     "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8706nnn/GSM8706920/suppl/GSM8706920_1_RPKM.bw",
     "source": "GEO GSE285699, TMP-seq SH-SY5Y WT rep1, hg38, 70-90 bp run-length encoded",
     "why": "torsion attempt 2; carries the twin-domain z = +23.21"},
    {"name": "GSM8706924_5_RPKM.bw", "path": SC / "GSM8706924_5_RPKM.bw", "kind": "raw",
     "sha256": "da0c81bb13975374", "bytes": 110110646,
     "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8706nnn/GSM8706924/suppl/GSM8706924_5_RPKM.bw",
     "source": "GEO GSE285699 WT rep2", "why": "replicate for attempt 2"},
    {"name": "_abcfeat.json", "path": SP / "_abcfeat.json", "kind": "derived",
     "sha256": "20f7aa4ae6040e16", "bytes": 1195158,
     "from": ["re2g_k562.bed.gz"], "transform": "build_abcfeat",
     "source": "aggregated from ENCODE-rE2G by build_abcfeat() in this file",
     "why": "nexus_txn L3. The ORIGINAL came from ABC and has no producer anywhere in this repo"},
    {"name": "lambert_tf_symbols.json", "path": SP / "lambert_tf_symbols.json", "kind": "derived",
     "sha256": "3e58c688dec13289", "bytes": 15116,
     "from": ["lambert.csv"], "transform": "build_lambert",
     "source": "filtered from Lambert v1.01 by build_lambert() in this file",
     "why": "nexus_txn L4, 1,639 TFs"},
    # ---- the metabolic model the closed loop runs on. Both were re-downloaded from the URLs below and
    # hashed against the copies already in the container: bit-identical, so the provenance is verified
    # rather than remembered.
    {"name": "HumanGEM.xml", "path": SP / "HumanGEM.xml", "kind": "raw",
     "sha256": "cc5a4383c6116b0c", "bytes": 43115559,
     "url": "https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/main/model/Human-GEM.xml",
     "source": "Human-GEM (SysBioChalmers), SBML, 12,931 reactions / 8,461 metabolites / 2,848 genes",
     "why": "the stoichiometry cell_loop closes the growth feedback through; cell_complete's own "
            "metabolism is 3,355 reaction STRINGS with no stoichiometric matrix behind them"},
    {"name": "clinvar.vcf.gz", "path": SC / "clinvar.vcf.gz", "kind": "raw",
     "sha256": "a99917628ff65443", "bytes": 193118277,
     "url": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz",
     "source": "NCBI ClinVar, GRCh38 VCF, 4,461,717 records",
     "why": "the variant-addressable slot loop_variant built: 4,443 genes with pathogenic missense "
            "and 14,987 with benign missense. NOTE this file is REGENERATED WEEKLY by NCBI, so the "
            "hash will drift and a mismatch here means a newer release, not corruption"},
    {"name": "ctcf_gm12878_hg19.bed.gz", "path": SC / "ctcf_gm12878_hg19.bed.gz", "kind": "raw",
     "sha256": "8795b92127be542e", "bytes": 655789,
     "url": "https://www.encodeproject.org/files/ENCFF833FTF/@@download/ENCFF833FTF.bed.gz",
     "source": "ENCODE ENCSR000AKB / ENCFF833FTF, GM12878 CTCF optimal IDR peaks, hg19, 40,790 peaks",
     "why": "the barrier landscape the whole extrusion model runs on. CELL-TYPE MATCHED to the Hi-C "
            "rather than borrowed from K562, and hg19 to match the map -- the project's other CTCF "
            "file is K562/GRCh38 and would have been a silent double mismatch"},
    {"name": "hg19_chr21.fa.gz", "path": SC / "hg19_chr21.fa.gz", "kind": "raw",
     "sha256": "1953ac4ea94b6588", "bytes": 11549785,
     "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/chr21.fa.gz",
     "source": "UCSC hg19 chr21",
     "why": "CTCF motif ORIENTATION, scanned with JASPAR MA0139.1. Orientation is the entire mechanism "
            "-- a forward site blocks a leftward leg and a reverse site a rightward one -- and it is "
            "not in the peak file, so it has to come from sequence. Also supplies GC and CpG"},
    {"name": "hg19_chr22.fa.gz", "path": SC / "hg19_chr22.fa.gz", "kind": "raw",
     "sha256": "4a5a0049b23232b1", "bytes": 11327826,
     "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/chr22.fa.gz",
     "source": "UCSC hg19 chr22",
     "why": "the HELD-OUT chromosome. loop_surrogate's whole result -- physics 0.2852 against a "
            "trained CNN's 0.0893 -- depends on chr22 never having been fitted"},
    {"name": "ctcf_pfm.json", "path": SC / "ctcf_pfm.json", "kind": "raw",
     "sha256": "987a2fd393dc17e8", "bytes": 509,
     "url": "https://jaspar.elixir.no/api/v1/matrix/MA0139.1/?format=json",
     "source": "JASPAR MA0139.1, CTCF, 19 bp position frequency matrix",
     "why": "the motif whose direction decides which way a barrier faces"},
    {"name": "HumanGEM_genes.tsv", "path": SC / "HumanGEM_genes.tsv", "kind": "raw",
     "sha256": "2a6058a157b3b9f3", "bytes": 1132077,
     "url": "https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/main/model/genes.tsv",
     "source": "Human-GEM gene table, 2,848 Ensembl IDs with gene symbols",
     "why": "REQUIRED, and its absence is silent. The SBML ships EMPTY gene names, so any module that "
            "reads cobra's g.name gets an Ensembl ID and every symbol join returns nothing. That cost "
            "loop_buffering two full runs reporting its isozyme features at exactly 0.5000 -- the "
            "signature of a column that was never computed, not of a feature that does not work"},
    {"name": "GSE63525_GM12878_insitu_primary_30.hic", "path": SC / "__remote_only__", "kind": "raw",
     "sha256": "REMOTE", "bytes": 0,
     "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/"
            "GSE63525_GM12878_insitu_primary_30.hic",
     "source": "Rao 2014 GM12878 in situ primary, the deepest published human Hi-C, hg19",
     "why": "the target the whole 4D arc is scored against. NEVER DOWNLOADED -- hicstraw range-requests "
            "one chromosome, so a ~20 GB file costs ~20 s and ~700k records. There is no local hash "
            "because there is no local copy, and that is recorded rather than hidden. The replicate "
            "file (…_insitu_replicate_30.hic) is read the same way and gives the 0.9285 ceiling"},
    {"name": "ens_gtf.gz", "path": SC / "ens_gtf.gz", "kind": "raw",
     "sha256": "8c87436bab973c87", "bytes": 55529204,
     "url": "https://ftp.ensembl.org/pub/release-112/gtf/homo_sapiens/"
            "Homo_sapiens.GRCh38.112.gtf.gz",
     "source": "Ensembl 112 GRCh38 annotation; only the CDS lines are read",
     "why": "the exon structure that makes loop_chain's genomic-coordinate-to-codon walk possible. "
            "Without it a point mutation cannot be carried to an amino acid at all, and the middle "
            "link of the chain would have to be a lookup instead of a computation"},
    {"name": "ens_cds.fa.gz", "path": SC / "ens_cds.fa.gz", "kind": "raw",
     "sha256": "1b7b6fa74c427b40", "bytes": 23111086,
     "url": "https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/cds/"
            "Homo_sapiens.GRCh38.cds.all.fa.gz",
     "source": "Ensembl 112 coding sequences, one per transcript",
     "why": "the codon itself. Paired with the GTF this gives the reference base at every variant, "
            "which is what let loop_chain check its own translation: 100.00% reference-base agreement "
            "over 966,223 variants, before ClinVar's consequence field was ever consulted"},
    {"name": "skempi_v2.csv", "path": SC / "skempi_v2.csv", "kind": "raw",
     "sha256": "76f8e60d09eaa4f0", "bytes": 1602208,
     "url": "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv",
     "source": "SKEMPI 2.0, measured binding free-energy changes for mutations in protein complexes",
     "why": "the only MEASURED interface data here, and the task nexus was actually validated on. "
            "loop_nexus_fair needs it as a positive control: without it there is no way to tell a "
            "sensor that does not work from a harness that does not work"},
    {"name": "sider_se.tsv.gz", "path": SC / "sider_se.tsv.gz", "kind": "raw",
     "sha256": "119b2f5319a9398d", "bytes": 2381171,
     "url": "http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz",
     "source": "SIDER 4.1, drug side effects from package inserts, STITCH compound ids x MedDRA terms",
     "why": "the ONLY compound-to-adverse-event data anywhere in this project; `side effect` had 48 "
            "gene-disease rows and nothing else"},
    {"name": "sider_names.tsv", "path": SC / "sider_names.tsv", "kind": "raw",
     "sha256": "6427a3e3202c71a8", "bytes": 34759,
     "url": "http://sideeffects.embl.de/media/download/drug_names.tsv",
     "source": "SIDER 4.1 compound id to drug name",
     "why": "the join from SIDER's STITCH ids to cell_complete's drug NAMES"},
    {"name": "intogen.zip", "path": SC / "intogen.zip", "kind": "raw",
     "sha256": "854def4465bcd7f9", "bytes": 964758,
     "url": "https://www.intogen.org/download?file=IntOGen-Drivers-20240920.zip",
     "source": "IntOGen 2024-06-18 driver compendium, Compendium_Cancer_Genes.tsv",
     "why": "cancer driver labels; the `cancer` row had 839 items and no held-out target"},
    {"name": "HumanGEM_genes.tsv", "path": SP / "HumanGEM_genes.tsv", "kind": "raw",
     "sha256": "2a6058a157b3b9f3", "bytes": 1132077,
     "url": "https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/main/model/genes.tsv",
     "source": "Human-GEM gene annotation table, ENSG -> symbol / Entrez / UniProt",
     "why": "the join between Human-GEM's ENSG ids and everything else here, which is keyed on symbol"},
    # The three below are the sources that gave cell_complete.json a producer. Every other entry in
    # this ledger backs a RESULT; these back the FILE 227 modules read, which had no provenance at all.
    {"name": "gnomad_loeuf.txt.bgz", "path": SP / "gnomad_loeuf.txt.bgz", "kind": "raw",
     "sha256": "153031d34b6794e8", "bytes": 4609488,
     "url": "https://storage.googleapis.com/gcp-public-data--gnomad/release/2.1.1/constraint/"
            "gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz",
     "source": "gnomAD v2.1.1 per-gene loss-of-function constraint",
     "why": "the source of cell_complete's loeuf column, and the one field that reproduces EXACTLY -- "
            "1.0000 exact over 15,038 genes. That exactness is what pins the version: v2.1.1 and not "
            "v4, which would have shown the same drift the coordinates do"},
    {"name": "cpgIslandExt.txt.gz", "path": SP / "cpgIslandExt.txt.gz", "kind": "raw",
     "sha256": "2339f8bad0ec9993", "bytes": 717984,
     "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz",
     "source": "UCSC hg38 CpG islands, Gardiner-Garden and Frommer calls",
     "why": "the source of the cpg column. It is the right track -- agreement rises from 0.72 to 0.92 "
            "when the query point moves to the file's own tss -- but no declared rule reaches the 0.90 "
            "gate, so the field is recorded as unverified rather than claimed"},
    {"name": "goa_human.gaf.gz", "path": SP / "goa_human.gaf.gz", "kind": "raw",
     "sha256": "db472faff1785878", "bytes": 15041996,
     "url": "https://current.geneontology.org/annotations/goa_human.gaf.gz",
     "source": "GO annotations for human, current release",
     "why": "fetched for cell_complete's proc and comp columns and NOT used for them. Those are each "
            "one of exactly 12 buckets and the GO-term-to-bucket map is recorded nowhere, so the "
            "ontology is retrievable and the coarsening is not. Kept because that gap is the finding"},
]
TRANSFORMS = {"build_abcfeat": build_abcfeat, "build_lambert": build_lambert}


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def fetch(entry):
    """Restore one input. Raw entries download; derived entries rebuild from their raw inputs."""
    dest = Path(entry["path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if entry["kind"] == "raw":
        r = subprocess.run(["curl", "-sSL", "--max-time", "1800", "-o", str(dest), entry["url"]],
                           capture_output=True, text=True)
        return r.returncode == 0
    src = next(e for e in LEDGER if e["name"] == entry["from"][0])
    if not Path(src["path"]).exists() and not fetch(src):
        return False
    TRANSFORMS[entry["transform"]](Path(src["path"]), dest)
    return True


REMOTE_ONLY = "REMOTE"


def resolve(e):
    """Find the file by name across the places raw data actually lands.

    THE DEFECT THIS FIXES.  `SC` points at outputs/orphan/supercoil, which holds the bigWigs and
    nothing else. Seven entries -- clinvar, both SIDER files, intogen, and the three added for
    loop_chain -- were recorded against that path while living in the session scratchpad, so this
    ledger reported them MISSING while they sat on disk being read by other modules. A ledger that
    cannot tell 'absent' from 'recorded at the wrong path' is not auditing anything, and its headline
    count was wrong in the direction that flatters nobody but hides a real hole. Where a file is found
    is now reported alongside whether it is found."""
    for base in (Path(e["path"]).parent, SCRATCH):
        p = base / Path(e["path"]).name
        if p.exists():
            return p
    return Path(e["path"])


def status():
    rows = []
    for e in LEDGER:
        if e["sha256"] == REMOTE_ONLY:
            # A source that is never fully downloaded cannot be hashed, and pretending otherwise would
            # be worse than saying so. hicstraw range-requests one chromosome out of ~20 GB.
            rows.append({**{k: e[k] for k in ("name", "kind", "source", "why")},
                         "state": "REMOTE (range-read, never downloaded)", "have": None,
                         "want": REMOTE_ONLY})
            continue
        p = resolve(e)
        if not p.exists():
            rows.append({**{k: e[k] for k in ("name", "kind", "source", "why")},
                         "state": "MISSING", "have": None, "want": e["sha256"]})
            continue
        h = sha16(p)
        rows.append({**{k: e[k] for k in ("name", "kind", "source", "why")},
                     "state": "OK" if h == e["sha256"] else "HASH MISMATCH",
                     "have": h, "want": e["sha256"], "bytes": p.stat().st_size,
                     "found_at": str(p.parent)})
    return rows


def main():
    log = []

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("THE DATA LEDGER -- every external input, where it came from, and how to get it back")
    report("=" * 100)
    report("  Four inputs nexus_txn needs were absent from this container, so a recorded result could be")
    report("  read but not checked. Recovering them took a Zenodo search, three ENCODE queries, a GEO")
    report("  crawl and a rebuild from a different predictor. That knowledge lived only in commit")
    report("  messages. This file makes it executable.")

    rows = status()
    report(f"\n  {'input':<44}{'kind':<9}{'state':<15}{'hash':>18}")
    for r in rows:
        report(f"  {r['name']:<44}{r['kind']:<9}{r['state']:<15}{(r['have'] or '-'):>18}")
    ok = sum(1 for r in rows if r["state"] == "OK")
    report(f"\n  {ok} of {len(rows)} present and hash-matching")

    # ---- THE CHECK: derived files must rebuild to the same hash --------------------------------------
    report("\n  REBUILD CHECK -- a derived file must reproduce its hash from its raw inputs, or the")
    report("  recorded transform is not the one that made it")
    rebuilds = []
    for e in LEDGER:
        if e["kind"] != "derived":
            continue
        src = next(x for x in LEDGER if x["name"] == e["from"][0])
        if not resolve(src).exists():
            report(f"    {e['name']:<34}source {src['name']} missing -- cannot check")
            rebuilds.append({"name": e["name"], "result": "SOURCE MISSING"})
            continue
        tmp = Path(str(e["path"]) + ".rebuild")
        n = TRANSFORMS[e["transform"]](Path(src["path"]), tmp)
        h = sha16(tmp)
        same = h == e["sha256"]
        tmp.unlink()
        report(f"    {e['name']:<34}{e['transform']}({src['name']}) -> {n:,} entries, hash {h}"
               f"   {'MATCHES' if same else 'DIFFERS from ' + e['sha256']}")
        rebuilds.append({"name": e["name"], "result": "MATCHES" if same else "DIFFERS",
                         "rebuilt_hash": h, "expected": e["sha256"], "n": n})

    det = all(r["result"] == "MATCHES" for r in rebuilds if r["result"] != "SOURCE MISSING")
    report("")
    if det:
        report("    Every derived file rebuilds bit-identically. The transforms in this file ARE the ones")
        report("    that produced them, so the provenance claim is checkable rather than asserted.")
    else:
        report("    A DERIVED FILE DID NOT REBUILD. The recorded transform is not the one that produced")
        report("    it. Reported as a defect in the ledger; the expected hash is NOT adjusted to match.")

    miss = [r for r in rows if r["state"] != "OK"]
    report(f"\n  TO RESTORE ANYTHING MISSING:  python3 colab/data_ledger.py --fetch <name|all>")
    if miss:
        report(f"  currently restorable: {', '.join(r['name'] for r in miss)}")

    report("\n  STILL NOT IN THIS LEDGER, and still the biggest hole, but smaller than it was:")
    report("  cell_complete.json is 38 MB, read by 227 modules, gitignored, and has no external source")
    report("  to fetch from. It is a DERIVED artefact whose original producer is lost. It now has a")
    report("  PARTIAL one -- colab/build_cell_complete.py rebuilds 8 of its 27 per-gene fields from the")
    report("  three sources added above plus the Ensembl GTF and Lambert, and CHECKS each against the")
    report("  surviving copy: 7 agree, cpg does not, and 17% of all module-reads are now backed by a")
    report("  named source rather than by the file's continued existence. The other 83% is classified,")
    report("  which is not the same as recovered: 30% has a URL and no download, 20% has real data with")
    report("  the rule or the source unrecorded, 6% is labels this project invented and cannot rebuild.")
    report("  A ledger cannot close that; only the missing downloads and the lost rules can.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "data_ledger", "entries": rows, "rebuilds": rebuilds,
               "all_present": ok == len(rows), "transforms_deterministic": bool(det), "log": log},
              open(OUT / "data_ledger.json", "w"), indent=2)
    report(f"\n  -> {OUT/'data_ledger.json'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--fetch":
        want = sys.argv[2]
        for e in LEDGER:
            if want in ("all", e["name"]):
                print(f"fetching {e['name']} ...", flush=True)
                print("  ok" if fetch(e) else "  FAILED", flush=True)
        raise SystemExit(0)
    raise SystemExit(main())
