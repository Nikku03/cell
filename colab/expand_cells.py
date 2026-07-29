"""EXPAND BEYOND K562: resolve and fetch the 1D + 3D stack for every cell type with CRISPR ground truth.

WHY THIS IS THE ONE EXPERIMENT LEFT. Every negative in the contact ledger -- analytic Rouse +0.005, KMC
extrusion + Langevin -0.0027, measured Hi-C -0.0031, Borzoi +0.0054 -- was measured on a model carrying 311
measured K562 TF ChIP tracks. `data_scarce.py` then showed the marginal value of contact GROWS as those
tracks are stripped: CTCF-count goes +0.0523 -> +0.0993 and measured Hi-C goes -0.0053 -> +0.0254. So the
ledger is conditional on a data richness almost no cell type has, and the condition is load-bearing.

That test stripped features from K562 as a PROXY for a data-poor cell type. This fetches the real thing.

THE GROUND TRUTH, and why it is the binding constraint rather than Hi-C or ATAC. Those are downloadable for
dozens of cell types; CRISPR-measured element-gene pairs are not. The Engreitz lab's held-out split is the
only cross-cell-type E-G benchmark that exists:

    WTC11    1,921 pairs    35 positives      base rate 0.0182
    K562     1,918 pairs   171 positives      base rate 0.0892   <- excluded, it is our training cell type
    HCT116     396 pairs    40 positives      base rate 0.1010
    Jurkat      75 pairs     7 positives      base rate 0.0933
    GM12878     68 pairs    21 positives      base rate 0.3088

2,460 non-K562 pairs carrying 103 positives. Small, and the only honest out-of-sample test available.

PEAKS, NOT SIGNAL TRACKS, AND THE SAME FOR K562. The obvious approach -- bigWig signal at each element --
needs hundreds of MB per assay per cell type and would exhaust the disk. narrowPeak files are 0.5-20 MB. But
the reason to prefer them is not size: if the held-out cell types get peak-derived features while K562
training keeps its bigWig-derived signal, the two feature spaces are not the same space and the model
cannot transfer. Any cross-cell-type claim would then be measuring a feature mismatch rather than biology.
So K562 is rebuilt from peaks too, and train and test share one definition.

WHAT IS FETCHED PER CELL TYPE
    DNase-seq / ATAC-seq   accessibility
    H3K27ac ChIP-seq       enhancer activity
    CTCF ChIP-seq          the feature that carries contact, and the one that grows in the data-poor regime
    POLR2A ChIP-seq        engaged polymerase
    Hi-C loops (bedpe)     measured contact, where it exists
"""
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
DEST = SP / "celltypes"
API = "https://www.encodeproject.org"
CRISPR_BASE = ("https://raw.githubusercontent.com/EngreitzLab/CRISPR_comparison/main/"
               "resources/crispr_data")
HELDOUT = "EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz"
TRAINING = "EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz"

# ENCODE biosample term names, which are not the names people use. "H1-hESC" returns 0 experiments and
# "H1" returns 204; "Jurkat clone E6-1" returns 0 and "Jurkat" returns 2. Getting these wrong looks
# identical to a cell type having no data, so each was checked against the portal rather than assumed.
#
#   K562     the training cell type: 311 TF ChIP tracks, PRO-cap, CRISPRi ground truth
#   GM12878  the data-scarce transfer test, and ultra-deep Hi-C
#   HCT116   held-out CRISPR pairs (396 / 40 positives)
#   WTC11    the largest held-out CRISPR block (1,921 / 35), plus Micro-C
#   H1       developmental; no CRISPR pairs, so it can extend the atlas but cannot validate
#   A549     dexamethasone time-course -- for condition-dependence, not for E-G validation
#   MCF-7    estrogen time-course -- likewise
#
# Jurkat is dropped: 2 released experiments total, and its 75 CRISPR pairs could not support a
# conclusion regardless. Recorded here rather than silently omitted.
CELLS = ["K562", "GM12878", "HCT116", "WTC11", "H1", "A549", "MCF-7"]

ASSAYS = [
    ("accessibility", "assay_title=DNase-seq"),
    ("accessibility_atac", "assay_title=ATAC-seq"),
    ("h3k27ac", "target.label=H3K27ac"),
    ("ctcf", "target.label=CTCF"),
    ("polr2a", "target.label=POLR2A"),
]


def api(path, tries=4):
    """Retry with backoff. The first pass over five cell types lost WTC11's CTCF to a URLError and all of
    Jurkat to HTTPErrors -- transient rate-limiting that is indistinguishable, in the output, from a cell
    type genuinely lacking the assay. A missing assay must be a fact about the portal, not about how hard
    we happened to hit it."""
    import time
    for k in range(tries):
        try:
            req = urllib.request.Request(API + path, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as fh:
                return json.load(fh)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def best_peak_file(cell, query):
    """Released, GRCh38, replicated narrowPeak for this cell type and assay.

    Preference order matters and is not cosmetic: a 'conservative IDR thresholded peaks' file is the
    replicated consensus, while 'peaks' from one replicate is noisier. Taking whatever came back first
    would silently mix quality tiers across cell types and confound any cross-cell-type comparison with
    processing differences.
    """
    q = (f"/search/?type=Experiment&biosample_ontology.term_name={urllib.parse.quote(cell)}"
         f"&{query}&status=released&format=json&limit=25")
    try:
        d = api(q)
    except Exception as e:
        return None, f"query failed: {type(e).__name__}"
    best = None
    rank = {"conservative IDR thresholded peaks": 0, "pseudoreplicated IDR thresholded peaks": 1,
            "replicated peaks": 2, "IDR thresholded peaks": 3, "peaks": 4}
    for g in d.get("@graph", []):
        try:
            exp = api(g["@id"] + "?format=json")
        except Exception:
            continue
        for f in exp.get("files", []):
            if (f.get("file_format") == "bed" and f.get("file_format_type") == "narrowPeak"
                    and f.get("assembly") == "GRCh38" and f.get("status") == "released"):
                r = rank.get(f.get("output_type", ""), 9)
                if best is None or r < best[0]:
                    best = (r, f["accession"], f.get("href"), f.get("output_type"),
                            (f.get("file_size") or 0))
        if best and best[0] == 0:
            break
    if best is None:
        return None, "no GRCh38 narrowPeak found"
    return {"accession": best[1], "href": best[2], "output_type": best[3], "size": best[4]}, None


def hic_loops(cell):
    """bedpe loop calls, queried on the FILE endpoint rather than by walking experiments.

    The first version paged through the first 15 Hi-C experiments and looked inside each. GM12878 has 85
    Hi-C experiments and 19 released GRCh38 loop files, and the walk reached none of them -- it reported
    ABSENT for a cell type with some of the deepest Hi-C in existence. Asking the file index directly is
    one request instead of fifteen and cannot miss files that sit past the page boundary.
    """
    try:
        d = api(f"/search/?type=File&file_format=bedpe&output_type=loops&assembly=GRCh38"
                f"&status=released&biosample_ontology.term_name={urllib.parse.quote(cell)}"
                f"&format=json&limit=50&field=accession&field=href&field=file_size")
    except Exception as e:
        return None, f"query failed: {type(e).__name__}"
    cand = [f for f in d.get("@graph", []) if f.get("href")]
    if not cand:
        return None, "no GRCh38 bedpe loops found"
    f = max(cand, key=lambda x: x.get("file_size") or 0)      # deepest = most loops called
    return {"accession": f["accession"], "href": f["href"],
            "output_type": "loops", "size": f.get("file_size") or 0}, None


def fetch(url, path):
    if path.exists() and path.stat().st_size > 0:
        return path.stat().st_size, "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-sSL", "-o", str(path), url], check=False)
    return (path.stat().st_size if path.exists() else 0), "fetched"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("EXPANDING BEYOND K562 -- resolving the 1D + 3D stack per cell type")
    print("=" * 100)

    # ground truth first: without labels the tracks are useless
    (SP / "crispr").mkdir(parents=True, exist_ok=True)
    for f in (HELDOUT, TRAINING):
        n, how = fetch(f"{CRISPR_BASE}/{f}", SP / "crispr" / f)
        print(f"  {how:8s} {f}  {n/1e3:.0f} KB")

    manifest, total = {}, 0
    for cell in CELLS:
        print(f"\n  --- {cell} ---")
        manifest[cell] = {}
        for name, q in ASSAYS:
            info, err = best_peak_file(cell, q)
            if info is None:
                print(f"    {name:20s} ABSENT ({err})")
                manifest[cell][name] = None
                continue
            total += info["size"]
            print(f"    {name:20s} {info['accession']}  {info['output_type'][:34]:34s} "
                  f"{info['size']/1e6:6.1f} MB")
            manifest[cell][name] = info
        info, err = hic_loops(cell)
        if info is None:
            print(f"    {'hic_loops':20s} ABSENT ({err})")
            manifest[cell]["hic_loops"] = None
        else:
            total += info["size"]
            print(f"    {'hic_loops':20s} {info['accession']}  {'loops':34s} {info['size']/1e6:6.1f} MB")
            manifest[cell]["hic_loops"] = info

    print(f"\n  total to download: {total/1e6:.0f} MB")
    if os.environ.get("EXPAND_FETCH", "1") == "1":
        print("\n  downloading (set EXPAND_FETCH=0 to resolve only):")
        for cell, assays in manifest.items():
            for name, info in assays.items():
                if not info:
                    continue
                ext = ".bedpe.gz" if name == "hic_loops" else ".bed.gz"
                n, how = fetch(API + info["href"], DEST / cell / f"{name}{ext}")
                info["path"] = str(DEST / cell / f"{name}{ext}")
                print(f"    {how:8s} {cell:8s} {name:20s} {n/1e6:6.1f} MB")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(OUT / "celltype_manifest.json", "w"), indent=1)
    have = {c: sum(1 for v in a.values() if v) for c, a in manifest.items()}
    print(f"\n  assays resolved per cell type: {have}")
    print(f"  -> {OUT/'celltype_manifest.json'}")


if __name__ == "__main__":
    main()
