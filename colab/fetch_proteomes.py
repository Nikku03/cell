"""Fetch proteomes (CDS protein translations) from NCBI for DEG organisms and/or
GTDB/RefSeq genomes, via E-utilities efetch (rettype=fasta_cds_aa). Stdlib only.

DEG mode:
  Reads data/drive_import/deg/deg_bacteria.csv (';'-delimited) for accession +
  DEG dataset id, picks the NOVEL datasets (those not already in our beril set,
  per deg_overlap_report.json), fetches each genome's CDS proteins, and writes:
    work/proteomes/<org>.faa          >locus_tag\\nSEQ
    work/proteomes/<org>.genes.csv    locus_tag, gene_name, protein_id
    work/deg_manifest.csv             deg_dataset, org, species

GTDB mode:
  Reads an accession list (one nucleotide accession per line) and writes
  work/proteomes_gtdb/<acc>.faa for the presence expansion (no labels needed).

Set NCBI_API_KEY in the environment to raise the rate limit (10/s vs 3/s).

Usage:
  python colab/fetch_proteomes.py --mode deg
  python colab/fetch_proteomes.py --mode deg --all          # incl. overlap datasets
  python colab/fetch_proteomes.py --mode gtdb --acc_list accs.txt
"""
import csv, json, re, os, sys, time, argparse, urllib.request, urllib.parse
from pathlib import Path

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
API_KEY = os.environ.get("NCBI_API_KEY", "")
MIN_INTERVAL = 0.11 if API_KEY else 0.34     # be polite
_last = [0.0]


def _throttle():
    dt = time.time() - _last[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last[0] = time.time()


def efetch_cds_aa(accession, retries=4):
    params = {"db": "nuccore", "id": accession, "rettype": "fasta_cds_aa", "retmode": "text"}
    if API_KEY:
        params["api_key"] = API_KEY
    url = EFETCH + "?" + urllib.parse.urlencode(params)
    for i in range(retries):
        try:
            _throttle()
            with urllib.request.urlopen(url, timeout=120) as resp:
                return resp.read().decode("utf-8", "replace")
        except Exception as e:
            wait = 2 ** i
            print(f"      efetch {accession} failed ({e}); retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    return ""


def parse_cds_aa(text):
    """Yield (locus_tag, gene_name, protein_id, sequence) from fasta_cds_aa."""
    rec_id = None; gene = ""; prot = ""; seq = []
    def tag(defline, key):
        m = re.search(r"\[" + key + r"=([^\]]+)\]", defline)
        return m.group(1) if m else ""
    for line in text.splitlines():
        if line.startswith(">"):
            if rec_id is not None and seq:
                yield rec_id, gene, prot, "".join(seq)
            defline = line[1:]
            lt = tag(defline, "locus_tag"); gene = tag(defline, "gene")
            prot = tag(defline, "protein_id")
            rec_id = lt or prot or defline.split()[0]
            seq = []
        else:
            seq.append(line.strip())
    if rec_id is not None and seq:
        yield rec_id, gene, prot, "".join(seq)


def safe(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:40]


def fetch_deg(deg_dir, work, include_all):
    deg_dir = Path(deg_dir); work = Path(work)
    (work / "proteomes").mkdir(parents=True, exist_ok=True)
    # which species are NOVEL (not already covered by a beril org)
    rep = json.load(open(deg_dir / "deg_overlap_report.json"))
    novel_names = set(n.lower() for n in rep.get("novel", []))
    # deg_bacteria.csv: ';'-delimited; name@0, refseq@3, deg_id@12
    rows = list(csv.reader(open(deg_dir / "deg_bacteria.csv"), delimiter=";"))
    manifest = []
    for row in rows:
        if len(row) < 13:
            continue
        name = row[0].strip(); accs = row[3].strip(); ds = row[12].strip()
        if not ds.startswith("DEG"):
            continue
        is_novel = name.lower() in novel_names
        if not include_all and not is_novel:
            continue
        org = "deg_" + safe(name)
        faa = work / "proteomes" / f"{org}.faa"
        gcsv = work / "proteomes" / f"{org}.genes.csv"
        manifest.append(dict(deg_dataset=ds, org=org, species=name))
        if faa.exists() and gcsv.exists():
            print(f"  [skip] {org} (exists)")
            continue
        print(f"  fetching {org}  ({ds}; {accs})")
        genes = []
        with open(faa, "w") as fo:
            for acc in [a for a in accs.split(",") if a.strip() and a != "-"]:
                txt = efetch_cds_aa(acc.strip())
                n = 0
                for lt, gene, prot, sq in parse_cds_aa(txt):
                    if not sq:
                        continue
                    fo.write(f">{lt}\n{sq}\n")
                    genes.append(dict(locus_tag=lt, gene_name=gene, protein_id=prot))
                    n += 1
                print(f"      {acc}: {n} CDS proteins")
        with open(gcsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["locus_tag", "gene_name", "protein_id"])
            w.writeheader(); w.writerows(genes)
    with open(work / "deg_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["deg_dataset", "org", "species"])
        w.writeheader(); w.writerows(manifest)
    print(f"\n  manifest: {len(manifest)} DEG datasets -> work/deg_manifest.csv")


def fetch_gtdb(acc_list, work):
    work = Path(work); out = work / "proteomes_gtdb"; out.mkdir(parents=True, exist_ok=True)
    accs = [l.strip() for l in open(acc_list) if l.strip() and not l.startswith("#")]
    print(f"  {len(accs)} accessions to fetch")
    for acc in accs:
        faa = out / f"{safe(acc)}.faa"
        if faa.exists():
            print(f"  [skip] {acc}"); continue
        txt = efetch_cds_aa(acc)
        n = 0
        with open(faa, "w") as fo:
            for lt, gene, prot, sq in parse_cds_aa(txt):
                if sq:
                    fo.write(f">{lt}\n{sq}\n"); n += 1
        print(f"  {acc}: {n} proteins")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["deg", "gtdb"], required=True)
    ap.add_argument("--deg_dir", default="data/drive_import/deg")
    ap.add_argument("--work", default="colab/work")
    ap.add_argument("--all", action="store_true", help="DEG: include overlap datasets too")
    ap.add_argument("--acc_list", default=None, help="GTDB: file of nucleotide accessions")
    a = ap.parse_args()
    if a.mode == "deg":
        fetch_deg(a.deg_dir, a.work, a.all)
    else:
        if not a.acc_list:
            sys.exit("--acc_list required for gtdb mode")
        fetch_gtdb(a.acc_list, a.work)
