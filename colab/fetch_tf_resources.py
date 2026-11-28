"""fetch_tf_resources -- pull the small, reachable pieces of the NEXUS data plan that actually unblock work.

REACHABILITY WAS TESTED, NOT ASSUMED, and the first test was wrong about a third of the time -- four of twelve URLs
returned 404 because the URL was wrong, not because the resource was blocked. AlphaFold needs model_v6 (v4 is retired),
Lambert lives at humantfs.ccbr.utoronto.ca, ChIP-Atlas needs the full file path, and KnockTF answers at licpathway.net
while its v2 host returns 502. Anything reported unreachable below was retested with a corrected URL first.

WHAT THIS FETCHES -- only the small files whose absence blocks a concrete next step:
    Lambert TF list     the definitive ~1,600 human TFs, so knockouts can be filtered to real TFs (plan step 1)
    JASPAR PWMs         position weight matrices, the practical stand-in for structural groove docking
    HOCOMOCO PWMs       second motif source, for agreement checking
    ChIP-Atlas metadata  the experiment index, to locate K562 TF ChIP without downloading peaks

WHAT IT DELIBERATELY DOES NOT FETCH: the multi-GB ATAC and peak archives. Disk here is a fixed per-session allowance
and was recently at 4.1 GB free; pulling a genome-wide peak atlas would exhaust it and break the session. Those are
listed with their URLs and sizes so the decision to spend the disk is explicit rather than accidental."""
import os
import json, sys, urllib.request, gzip, io
from pathlib import Path

SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))

SMALL = [
    ("lambert_tfs.csv", "http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv",
     "Lambert et al. definitive human TF list"),
    ("jaspar_core_vert.meme", "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt",
     "JASPAR 2024 CORE vertebrate PWMs"),
    ("hocomoco11_human.meme", "https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_HUMAN_mono_meme_format.meme",
     "HOCOMOCO v11 human mononucleotide PWMs"),
    ("chipatlas_experiments.tab", "https://chip-atlas.dbcls.jp/data/metadata/experimentList.tab",
     "ChIP-Atlas experiment index (all assays, all species)"),
]
# listed, NOT fetched -- each would consume a large share of the remaining disk allowance
LARGE = [
    ("CATLAS 222-cell-type cCREs", "http://catlas.org/catlas_downloads/humantissues/cCRE_by_cell_type.tar.gz",
     "~2-4 GB", "404 on the paths tried; needs the current CATLAS download URL confirmed"),
    ("ChIP-Atlas all peaks hg38 q<0.05", "https://chip-atlas.dbcls.jp/data/hg38/allPeaks_light/allPeaks_light.hg38.05.bed.gz",
     "~5 GB", "reachable (200) but would exhaust the disk allowance"),
    ("ENCODE 4sU/RNA half-life", "https://www.encodeproject.org/search/?type=Experiment&format=json",
     "varies", "portal search reachable (200); assay_title=4sU-seq is not a valid facet, needs the correct facet name"),
]


def get(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw


def main():
    SP.mkdir(parents=True, exist_ok=True)
    got = []
    for name, url, desc in SMALL:
        dst = SP / name
        if dst.exists() and dst.stat().st_size > 1000:
            print(f"  [cached] {name} ({dst.stat().st_size/1024:.0f} KB)")
            got.append({"file": name, "bytes": dst.stat().st_size, "desc": desc, "status": "cached"})
            continue
        try:
            raw = get(url)
            dst.write_bytes(raw)
            print(f"  [ok]     {name} ({len(raw)/1024:.0f} KB)  {desc}")
            got.append({"file": name, "bytes": len(raw), "desc": desc, "status": "fetched", "url": url})
        except Exception as e:
            print(f"  [FAIL]   {name}: {type(e).__name__} {e}")
            got.append({"file": name, "desc": desc, "status": f"failed: {type(e).__name__}", "url": url})

    # --- summarise the TF list, which is the piece that unblocks the most ---
    summary = {}
    lam = SP / "lambert_tfs.csv"
    if lam.exists():
        import csv
        rows = list(csv.DictReader(io.StringIO(lam.read_text(errors="replace"))))
        key = next((k for k in (rows[0] if rows else {}) if "Is TF" in k or "IsTF" in k), None)
        tfs = [r for r in rows if key and str(r.get(key, "")).strip().lower().startswith("yes")]
        namecol = next((k for k in (rows[0] if rows else {}) if k.strip() in ("HGNC symbol", "Gene Information")), None)
        summary["lambert"] = {"rows": len(rows), "is_tf_yes": len(tfs), "tf_column": key, "name_column": namecol}
        print(f"\n  Lambert: {len(rows)} rows, {len(tfs)} flagged as true TFs (column '{key}')")
        if namecol:
            syms = sorted({(r.get(namecol) or "").strip() for r in tfs} - {""})
            (SP / "lambert_tf_symbols.json").write_text(json.dumps(syms))
            print(f"  wrote {len(syms)} TF symbols -> lambert_tf_symbols.json")
            summary["lambert"]["n_symbols"] = len(syms)

    for name in ("jaspar_core_vert.meme", "hocomoco11_human.meme"):
        p = SP / name
        if p.exists():
            n = p.read_text(errors="replace").count("MOTIF ")
            summary[name] = {"motifs": n}
            print(f"  {name}: {n} motifs")
    ca = SP / "chipatlas_experiments.tab"
    if ca.exists():
        hs = k562 = 0
        for line in ca.read_text(errors="replace").splitlines():
            f = line.split("\t")
            if len(f) > 5 and f[1] == "hg38":
                hs += 1
                if "K-562" in line:
                    k562 += 1
        summary["chipatlas"] = {"hg38_experiments": hs, "k562_experiments": k562}
        print(f"  ChIP-Atlas: {hs} hg38 experiments, {k562} in K-562")

    print("\n  NOT FETCHED (large; the disk allowance here is finite and was recently at ~4 GB free):")
    for nm, url, size, note in LARGE:
        print(f"    {nm:34s} {size:>8s}  {note}")
        print(f"      {url}")

    json.dump({"fetched": got, "summary": summary,
               "not_fetched": [{"name": n, "url": u, "size": s, "note": t} for n, u, s, t in LARGE]},
              open(OUT / "tf_resources.json", "w"), indent=1)
    print("\n  -> outputs/orphan/tf_resources.json")


if __name__ == "__main__":
    main()
