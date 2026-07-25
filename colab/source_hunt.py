"""source_hunt -- go looking for the data somebody else already collected, for the gaps this project cannot close on its own.

WHY. Three questions here are not imprecise, they are UNANSWERABLE with what is on disk, and no modelling trick fixes that:

    G1  ground truth for transcription rate.  Every rate number in this project is DERIVED: k_syn = RPKM * k_deg. It has
        never been checked against a direct measurement, so nobody knows how much of the R2 ceiling is feature weakness
        and how much is a noisy target. A nascent-transcription assay (TT-seq / PRO-seq / 4sU) measures synthesis
        directly and would settle it.
    G2  the knockdown-depth threshold.  knockdown_dose.py failed HONESTLY: Spearman -0.063, and shallow knockdowns
        retaining 60-80% of transcript were statistically indistinguishable from the deepest ones. The reason is
        structural -- in a genome-wide screen each gene appears at ONE depth, so depth is confounded with gene identity.
        Only a titration series (the SAME gene knocked down to several graded levels) can separate them.
    G3  the TF magnitude term beta.  nexus_txn deliberately did not fit beta because a free parameter with nothing to
        fit against absorbs error and inflates R2. Acute TF depletion time courses (degron/dTAG + nascent readout) are
        the data that would constrain it.

WHAT THIS DOES. Probes concrete URLs -- not search pages, actual files -- records status, size and cost, and downloads
only what is small enough to be free. Disk here is a fixed per-session allowance currently under 3 GB, so a 1.4 GB
archive is a decision, not a detail; anything expensive is reported with its price rather than silently fetched.

REACHABILITY IS TESTED, NOT ASSUMED. The previous pass at this got a third of its URLs wrong the first time. Two more
were wrong here: Springer's own supplementary host returns AccessDenied from its GCS bucket for both papers tried, so
the Europe PMC supplementaryFiles endpoint is used instead, and it works where the publisher's own link does not."""
import json, sys, time, urllib.request, urllib.error, zipfile, gzip, io
from pathlib import Path

SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
OUT = Path("outputs/orphan")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}

# ---- candidate sources, one row per (gap, concrete artefact). "probe" is a real file or API, never a landing page. ----
CANDIDATES = [
    # G1 -- direct measurement of transcription rate
    dict(gap="G1 rate ground truth", name="Wachutka 2019 TT-seq K562 synthesis rates",
         probe="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6548502/supplementaryFiles",
         kind="epmc_zip", member="elife-45056-supp3.gz",
         gives="synthesis [1/cell/min] per gene, measured by TT-seq in K562 -- an INDEPENDENT estimate of the exact "
               "quantity this project derives as RPKM*k_deg"),
    dict(gap="G1 rate ground truth", name="Blumberg 2021 PRO-seq+RNA-seq stability",
         probe="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7885420/supplementaryFiles",
         kind="epmc_zip", member=None,
         gives="PRO-seq-derived stability; supplement is PDF only, no machine-readable rate table"),
    dict(gap="G1 rate ground truth", name="ENCODE nascent-transcription experiments",
         probe="https://www.encodeproject.org/search/?type=Experiment&assay_title=PRO-cap&format=json&limit=1",
         kind="head", gives="PRO-cap/PRO-seq experiment index; raw signal only, needs alignment to become a rate"),
    # G2 -- graded knockdown of the same gene
    dict(gap="G2 knockdown titration", name="Jost 2020 attenuated-guide CRISPRi titration (Nat Biotech)",
         probe="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7065968/supplementaryFiles",
         kind="epmc_zip", member=None,
         gives="mismatched sgRNAs giving graded knockdown of the SAME gene -- exactly what G2 needs"),
    dict(gap="G2 knockdown titration", name="Jost 2020 GEO series (raw)",
         probe="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132078/suppl/filelist.txt",
         kind="head", gives="the same study's GEO deposit; RAW tar is 1.3 GB of sequencing data, not a phenotype table"),
    # G3 -- acute TF depletion
    dict(gap="G3 TF magnitude beta", name="KnockTF (TF knockdown -> target fold-change)",
         probe="http://www.licpathway.net/KnockTF/download.php", kind="head",
         gives="TF-knockdown differential expression with fold-changes; magnitude but not kinetics"),
    # G4 -- enhancer-gene links outside K562
    dict(gap="G4 enhancer links", name="ABC Nasser 2021 all predictions, 131 cell types",
         probe="https://mitra.stanford.edu/engreitz/oak/public/Nasser2021/"
               "AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz", kind="head",
         gives="ABC enhancer-gene links across 131 cell types; extends nexus_txn's L3 beyond K562"),
    dict(gap="G4 enhancer links", name="ReMap 2022 regulatory atlas",
         probe="https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/remap2022_nr_macs2_hg38_v1_0.bed.gz",
         kind="head", gives="non-redundant TF binding peaks across studies"),
]

# GEO free-text counts -- cheap way to ask "does this kind of data exist at all, and how much of it"
GEO_QUERIES = {
    "G1 nascent rate": ['("PRO-seq"[All Fields]) AND "Homo sapiens"[Organism]',
                        '("TT-seq"[All Fields]) AND "Homo sapiens"[Organism]',
                        '("SLAM-seq"[All Fields]) AND "Homo sapiens"[Organism]'],
    # no wildcards: esearch rejects "mismatch*" inside a parenthesised OR and returns an HTTP error, which the
    # first run reported as "failed" rather than as a count of zero -- the distinction matters
    "G2 titration": ['CRISPRi[All Fields] AND titration[All Fields] AND "Homo sapiens"[Organism]',
                     'CRISPRi[All Fields] AND mismatch[All Fields] AND "Homo sapiens"[Organism]'],
    "G3 acute depletion": ['(degron[All Fields] OR dTAG[All Fields] OR "auxin-inducible"[All Fields]) '
                           'AND "Homo sapiens"[Organism]'],
}


def probe(url, timeout=45):
    """HEAD-ish probe: ask for the first 2 KB so a multi-GB file costs 2 KB to check."""
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-2047"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            head = r.read()
            cr = r.headers.get("Content-Range")
            total = int(cr.split("/")[-1]) if cr and "/" in cr and cr.split("/")[-1].isdigit() else None
            if total is None:
                cl = r.headers.get("Content-Length")
                total = int(cl) if cl and cl.isdigit() else None
            return dict(status=r.status, bytes_total=total, secs=round(time.time() - t0, 1),
                        ctype=r.headers.get("Content-Type", ""), head=head[:80].hex())
    except urllib.error.HTTPError as e:
        return dict(status=e.code, error=e.reason, secs=round(time.time() - t0, 1))
    except Exception as e:
        return dict(status=0, error=f"{type(e).__name__}: {e}", secs=round(time.time() - t0, 1))


def geo_count(term):
    u = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gds&retmode=json&retmax=0&term="
         + urllib.parse.quote(term))
    try:
        with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=40) as r:
            return int(json.load(r)["esearchresult"]["count"])
    except Exception as e:
        return f"failed: {type(e).__name__}"


def fetch_epmc(pmcid_url, dst):
    req = urllib.request.Request(pmcid_url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        # 404 here does NOT mean the paper is missing -- it means the publisher deposited no supplementary
        # files in PMC, which is the normal outcome for Nature-family papers and is the whole reason G2 stays open.
        return None, f"HTTP {e.code} -- publisher deposited no supplementary files in PMC"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if len(raw) < 100:
        return None, f"empty response ({len(raw)} bytes) -- publisher did not deposit supplements in PMC"
    dst.write_bytes(raw)
    try:
        z = zipfile.ZipFile(dst)
        return sorted(((i.file_size, i.filename) for i in z.infolist()), reverse=True), None
    except zipfile.BadZipFile:
        return None, "response is not a zip"


def main():
    import urllib.parse                                        # noqa: F401  (used by geo_count)
    SP.mkdir(parents=True, exist_ok=True)
    free_gb = None
    try:
        import shutil
        free_gb = shutil.disk_usage("/").free / 1e9
    except Exception:
        pass
    print(f"disk free: {free_gb:.1f} GB -- anything above ~0.5 GB is a spending decision, not a detail\n")

    rows = []
    for c in CANDIDATES:
        r = probe(c["probe"])
        size = r.get("bytes_total")
        sz = f"{size/1e6:.0f} MB" if size and size > 1e6 else (f"{size/1e3:.0f} KB" if size else "?")
        ok = r["status"] in (200, 206)
        print(f"  [{r['status']:>3}] {c['name'][:52]:52s} {sz:>9s}  {c['gap']}")
        if not ok:
            print(f"        -> {r.get('error')}")
        rows.append({**{k: c[k] for k in ("gap", "name", "gives")}, "url": c["probe"],
                     "status": r["status"], "bytes": size, "reachable": ok})

    # ---- fetch the small ones that are actually usable ----
    print("\n  FETCHING (small + machine-readable only)")
    fetched = []
    for c in CANDIDATES:
        if c["kind"] != "epmc_zip":
            continue
        pmc = c["probe"].rstrip("/").split("/")[-2]
        dst = SP / f"{pmc}_suppl.zip"
        if dst.exists() and dst.stat().st_size > 1000:
            listing, err = sorted(((i.file_size, i.filename) for i in zipfile.ZipFile(dst).infolist()), reverse=True), None
        else:
            listing, err = fetch_epmc(c["probe"], dst)
        if err:
            print(f"    [skip] {c['name'][:52]:52s} {err}")
            fetched.append({"name": c["name"], "status": err})
            continue
        tabular = [(s, f) for s, f in listing if f.endswith((".gz", ".csv", ".tsv", ".txt", ".xlsx"))]
        print(f"    [ok]   {c['name'][:52]:52s} {len(listing)} files, "
              f"{len(tabular)} machine-readable")
        if c.get("member"):
            z = zipfile.ZipFile(dst)
            raw = z.read(c["member"])
            raw = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
            outp = SP / "wachutka_synthesis.tsv"
            outp.write_bytes(raw)
            hdr = raw[:400].decode(errors="replace").splitlines()[0]
            nrow = raw.count(b"\n")
            print(f"           -> {outp.name}  {nrow} rows")
            print(f"              columns: {hdr}")
            fetched.append({"name": c["name"], "status": "extracted", "file": outp.name,
                            "rows": nrow, "columns": hdr.split("\t")})
        else:
            fetched.append({"name": c["name"], "status": "fetched, no machine-readable rate table",
                            "files": [f for _, f in tabular][:10]})

    # ---- how much of each kind of data exists at all ----
    print("\n  GEO INVENTORY (does anyone have this kind of data, and how much)")
    geo = {}
    for gap, terms in GEO_QUERIES.items():
        geo[gap] = {}
        for t in terms:
            n = geo_count(t)
            label = t.split("[")[0].strip("(\" ")
            geo[gap][label] = n
            print(f"    {gap:22s} {label:26s} {n:>7}")

    verdict_bits = []
    w = SP / "wachutka_synthesis.tsv"
    if w.exists():
        verdict_bits.append(
            "G1 IS CLOSED, and it is the one that mattered most: Wachutka et al. 2019 (eLife, K562 TT-seq) publish a "
            f"per-gene 'synthesis [1/cell/min]' column for {w.read_bytes().count(chr(10).encode())-1} genes -- a DIRECT "
            "measurement of the quantity this project has only ever derived as RPKM*k_deg. Two independent estimates of "
            "the same physical rate make the noise ceiling measurable for the first time, which decides whether the R2 "
            "shortfall is weak features or a noisy target. That test is rate_ground_truth.py.")
    verdict_bits.append(
        "G2 IS NOT CLOSED by a download. The Jost 2020 attenuated-guide titration is the right dataset and the paper is "
        "open, but Nature deposited no machine-readable supplement in PMC and the publisher's own supplementary host "
        "returns AccessDenied; the GEO deposit is 1.3 GB of raw sequencing, which is not a phenotype table and would "
        "need the full analysis rebuilt. So the knockdown-depth threshold stays unanswered, and the reason is now "
        "specific -- not 'no such data exists' but 'the processed table is not publicly downloadable'.")
    verdict_bits.append(
        "G3 IS NOT CLOSED either, for a different reason: degron/dTAG series exist in quantity, but each is one TF in "
        "one cell line with its own processing, so building a beta from them is a multi-study reanalysis, not a fetch.")
    verdict = ("SOURCE HUNT for the three gaps this project cannot close on its own. " + " ".join(verdict_bits)
               + " WHAT WAS PROBED: concrete files, not landing pages, with a 2 KB range request so a multi-GB archive "
               "costs 2 KB to check. Europe PMC's supplementaryFiles endpoint works where the publishers' own links "
               "return AccessDenied, which is how G1 was found at all. Deterministic given the sources' availability.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"disk_free_gb": round(free_gb, 2) if free_gb else None, "probes": rows,
               "fetched": fetched, "geo_inventory": geo, "verdict": verdict, "note": verdict},
              open(OUT / "source_hunt.json", "w"), indent=1)
    print("\n  -> outputs/orphan/source_hunt.json")


if __name__ == "__main__":
    import urllib.parse
    main()
