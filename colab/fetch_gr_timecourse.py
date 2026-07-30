"""FETCH THE ENCODE A549 DEXAMETHASONE TIME COURSE -- the one dataset where the dynamic question is fair.

WHY THIS SERIES AND NOT ANOTHER. Stage 1 of the dynamic model has come back NULL twice: once on pseudotime
(delta R2 0.0013) and once on real clock time (21 lags, permuted time equally good). Neither was a power
failure -- split-half reliability was +0.3991 -- and neither was a bad-links failure, since validated links
scored -0.0034 too. The remaining explanation was that nothing was DRIVING the system: in an unperturbed
population the regulatory layer is at steady state, so there is no causal transient for occupancy to explain.

ENCODE's A549 dexamethasone series is the fix. 100 nM dexamethasone activates the glucocorticoid receptor
(NR3C1), which translocates to the nucleus within minutes and binds thousands of sites. It is a real driver,
with a real clock, and the readouts are on the SAME time grid in the SAME cell line:

    NR3C1 ChIP-seq   5 10 15 20 25 30 60 120 180 240 300 360 420 480 600 720 min   (16 points)
    polyA RNA-seq    5 10 15 20 25 30 60 120 180 240 300 360 420 480 600 720 min   (16 points)
    DNase-seq                    30 60 120 180 240 300 360 420 480 600 720 min     (11 points)
    EP300, JUN       the full 16-point grid
    JUNB FOSL2 CEBPB BCL3 HES2 CTCF SMC3 RAD21   the 11-point grid

That is a causal perturbation with matched occupancy and expression at sixteen times. If occupancy dynamics
cannot predict expression dynamics HERE, the null is about the relationship rather than about the data.

PEAKS, NOT BIGWIGS, AND THE REASON IS NOT ONLY DISK. The fold-change bigWigs are 1.4 GB each; sixteen
timepoints of one target would exhaust the volume. But peak-level fold-enrichment is also the right
measurement for this particular biology: before dex there is essentially no GR on chromatin and within 30
minutes there are thousands of sites, so the signal IS peak appearance. The cost is recorded honestly -- a
site below the peak threshold reads as exactly zero rather than as a small number -- and the fraction of the
union site set that is zero at each timepoint is reported so the size of that cost is visible.

READ TSV COLUMNS BY NAME. A previous fetch in this project mixed featureCounts and RSEM outputs and read
RSEM's `effective_length` (93.00) as a genomic Start coordinate. Every column here is looked up in the header.
"""
import gzip
import io
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
API = "https://www.encodeproject.org"
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
DIR = SP / "grtc"
ROOT = Path(__file__).resolve().parent.parent
CELL = "A549"
DRUG = "dexamethasone"
# ONE LAB ACROSS THE WHOLE TIME AXIS. Two productions contributed to this series: Tim Reddy's lab ran the
# full 16-point GGR course for RNA, NR3C1 and every cofactor, while Richard Myers' lab contributed legacy
# ENCODE2 experiments AT 60 MINUTES ONLY -- five RNA and four NR3C1. Averaging those into the 60-minute point
# would place a lab/library batch effect at exactly one timepoint, which shows up as a spurious jump into and
# out of t=60 and corrupts two of the fifteen intervals. Since the target here IS the interval-to-interval
# change, that is not a small contamination. Same principle as pinning one pipeline version: a processing
# difference must never lie along the axis being measured.
LAB = "Tim Reddy, Duke"
# the targets worth the disk: GR itself, its two best-covered co-factors on the full grid, and the
# architectural pair, which is the control -- CTCF/RAD21 occupancy should NOT track a steroid response
TARGETS = ("NR3C1", "EP300", "JUN", "JUNB", "FOSL2", "CEBPB", "CTCF", "RAD21")
UNITS = {"minute": 1.0, "hour": 60.0, "day": 1440.0}


def api(path, tries=4):
    last = None
    for i in range(tries):
        try:
            r = urllib.request.Request(API + path,
                                       headers={"accept": "application/json", "User-Agent": "cellos"})
            return json.load(urllib.request.urlopen(r, timeout=180))
        except Exception as ex:                       # network only; a 4xx is re-raised on the last try
            last = ex
            if i == tries - 1:
                raise
            import time
            time.sleep(2 ** (i + 1))
    raise last


def download(url, path, tries=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    for i in range(tries):
        try:
            r = urllib.request.Request(url, headers={"User-Agent": "cellos"})
            with urllib.request.urlopen(r, timeout=600) as fh, open(path, "wb") as out:
                while True:
                    b = fh.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
            return path
        except Exception:
            if path.exists():
                path.unlink()
            if i == tries - 1:
                raise
            import time
            time.sleep(2 ** (i + 1))


def minutes(exp):
    """Treatment duration in minutes, from whichever replicate carries it."""
    out = set()
    for r in exp.get("replicates", []):
        for t in r.get("library", {}).get("biosample", {}).get("treatments", []):
            if t.get("duration") is not None:
                out.add(float(t["duration"]) * UNITS.get(t.get("duration_units", "minute"), 1.0))
    return sorted(out)


def index():
    """Every released A549 + dexamethasone experiment, with its treatment duration."""
    cached = DIR / "index.json"
    if cached.exists():
        return json.load(open(cached))
    q = ("/search/?type=Experiment"
         f"&biosample_ontology.term_name={urllib.parse.quote(CELL)}"
         f"&replicates.library.biosample.treatments.treatment_term_name={urllib.parse.quote(DRUG)}"
         "&status=released&limit=500"
         "&field=accession&field=assay_title&field=target.label&field=replicates&field=lab.title")
    d = api(q)
    rows, dropped = [], 0
    for g in d["@graph"]:
        m = minutes(g)
        if not m:
            continue
        lab = (g.get("lab") or {}).get("title")
        if lab != LAB:
            dropped += 1
            continue
        rows.append({"acc": g["accession"], "assay": g.get("assay_title"), "lab": lab,
                     "target": (g.get("target") or {}).get("label"), "min": m[0]})
    print(f"  provenance filter: kept {len(rows)} experiments from '{LAB}', dropped {dropped} from other labs")
    DIR.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(cached, "w"), indent=1)
    return rows


_FILES = {}


def exp_files(acc):
    """Every released GRCh38 file of an experiment, in ONE request.

    The obvious `/experiments/ACC/?frame=object` then one GET per href costs 13-60 round trips per
    experiment; across ~90 experiments that was the whole runtime. The File search endpoint returns the
    embedded objects in a single call.
    """
    if acc not in _FILES:
        d = api(f"/search/?type=File&dataset=/experiments/{acc}/&status=released&limit=300&frame=object")
        _FILES[acc] = [f for f in d.get("@graph", []) if f.get("assembly") == "GRCh38"]
    return _FILES[acc]


def pick_file(acc, want):
    """One file per experiment. `want` is a list of (file_format, file_format_type, output_type) in
    PREFERENCE ORDER -- the first match wins, so every timepoint of a target gets the same output type and
    the series stays comparable. A timepoint whose preferred type is missing is dropped rather than
    substituted, because mixing output types across a time course would put a processing difference on the
    time axis, and that would be indistinguishable from biology.

    Several experiments carry more than one file of the same output type (re-analyses under different
    pipeline versions), so the tie is broken deterministically: ENCODE's preferred_default first, then
    accession order. An arbitrary tie-break would silently mix pipeline versions across timepoints.
    """
    files = exp_files(acc)
    for fmt, ftype, otype in want:
        cands = [f for f in files if f.get("file_format") == fmt
                 and (ftype is None or f.get("file_format_type") == ftype)
                 and f.get("output_type") == otype]
        if cands:
            cands.sort(key=lambda f: (not f.get("preferred_default"), f.get("accession")))
            return cands[0]
    return None


PEAK_WANT = [("bed", "narrowPeak", "conservative IDR thresholded peaks"),
             ("bed", "narrowPeak", "pseudoreplicated IDR thresholded peaks"),
             ("bed", "narrowPeak", "IDR thresholded peaks"),
             ("bed", "narrowPeak", "replicated peaks"),
             ("bed", "narrowPeak", "peaks")]
RNA_WANT = [("tsv", None, "gene quantifications")]


def fetch_peaks(rows, assay, target, tag):
    sel = sorted([r for r in rows if r["assay"] == assay and r["target"] == target],
                 key=lambda r: r["min"])
    out = {}
    for r in sel:
        f = pick_file(r["acc"], PEAK_WANT)
        if f is None:
            print(f"    {tag} t={r['min']:.0f} -- no GRCh38 narrowPeak, dropped")
            continue
        p = DIR / tag / f"{int(r['min'])}min_{r['acc']}.bed.gz"
        download(API + f["href"], p)
        # keep ONE experiment per timepoint: replicated timepoints would otherwise weight that point
        # several times in every fit
        key = int(r["min"])
        if key not in out:
            out[key] = {"path": str(p), "acc": r["acc"], "output_type": f["output_type"],
                        "size": p.stat().st_size}
    # ENFORCE what the preference list only intends. pick_file walks PEAK_WANT in order, so a timepoint
    # missing the preferred output type silently receives the next one down -- conservative IDR peaks for
    # most of the course and pseudoreplicated peaks for one point, whose signalValue is on a different scale.
    # That would place a processing difference on the time axis, which is precisely what the docstring says
    # must never happen; stating the rule is not the same as checking it.
    kinds = {v["output_type"] for v in out.values()}
    if len(kinds) > 1:
        raise SystemExit(f"{tag}: mixed peak output types across the time course {sorted(kinds)} -- the "
                         f"series is not comparable and is not written")
    return out


def read_gene_tsv(path):
    """gene_id -> TPM, by column NAME. RSEM writes gene_id/TPM; some ENCODE runs write different orders."""
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        head = fh.readline().rstrip("\n").split("\t")
        try:
            gi, ti = head.index("gene_id"), head.index("TPM")
        except ValueError:
            raise SystemExit(f"{path}: no gene_id/TPM columns, header was {head[:8]}")
        out = {}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) <= max(gi, ti):
                continue
            g = p[gi].split(".")[0]
            if not g.startswith("ENSG"):
                continue                              # RSEM rows also carry spike-ins and tRNAs
            out[g] = out.get(g, 0.0) + float(p[ti])   # PAR_Y duplicates share a stripped id
    return out


RNA_ANNOTATION = "V29"          # the RSEM re-processing; see fetch_rna


def fetch_rna(rows):
    """One expression vector per (timepoint, biological replicate). Replicates are kept separate because
    they are the only estimate of how reliable a CHANGE in expression is, which bounds any R2 below.

    EVERY EXPERIMENT IN THIS SERIES CARRIES TWO INCOMPATIBLE QUANTIFICATIONS under the same
    `gene quantifications` output_type: the ENCODE RSEM re-processing against GENCODE V29, which has a TPM
    column, and the original Reddy-lab featureCounts run against V22, which is raw counts with a
    `# Program:featureCounts ...` comment line where the header should be. Taking whichever came first would
    mix normalisations and annotations along the time axis -- and this project has already been bitten by
    exactly that, reading RSEM's `effective_length` of 93.00 as a genomic start coordinate. So the RSEM
    annotation is pinned explicitly and the choice is asserted uniform across the series rather than assumed.
    """
    sel = sorted([r for r in rows if r["assay"] == "polyA plus RNA-seq"], key=lambda r: r["min"])
    cols, gset, anns = [], None, set()
    for r in sel:
        cands = [f for f in exp_files(r["acc"])
                 if f.get("file_format") == "tsv" and f.get("output_type") == "gene quantifications"
                 and f.get("genome_annotation") == RNA_ANNOTATION]
        if not cands:
            print(f"    RNA t={r['min']:.0f} {r['acc']} -- no {RNA_ANNOTATION} gene quantifications, dropped")
            continue
        cands.sort(key=lambda f: f.get("accession"))
        for f in cands:
            anns.add(f.get("genome_annotation"))
            tmp = DIR / "rna_tmp" / f"{f['accession']}.tsv"
            download(API + f["href"], tmp)
            tpm = read_gene_tsv(tmp)
            tmp.unlink()                              # 10 MB each; keep the vector, drop the file
            reps = f.get("biological_replicates") or [0]
            cols.append({"min": int(r["min"]), "exp": r["acc"], "file": f["accession"],
                         "rep": int(reps[0]), "tpm": tpm})
            gset = set(tpm) if gset is None else (gset & set(tpm))
        print(f"    RNA t={int(r['min']):>4d} min  {r['acc']}  {len(cands)} replicate file(s)  "
              f"reps {sorted({c['rep'] for c in cols if c['exp']==r['acc']})}")
    if len(anns) > 1:
        raise SystemExit(f"mixed genome annotations across the RNA time course: {sorted(anns)}")
    return cols, sorted(gset or [])


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("FETCH -- ENCODE A549 dexamethasone time course (occupancy + expression on one clock)")
    print("=" * 100)
    rows = index()
    print(f"  {len(rows)} released A549 + dexamethasone experiments carry a treatment duration")

    man = {"cell": CELL, "drug": DRUG, "peaks": {}, "assembly": "GRCh38"}
    for t in TARGETS:
        got = fetch_peaks(rows, "TF ChIP-seq", t, t)
        man["peaks"][t] = got
        mb = sum(v["size"] for v in got.values()) / 1e6
        print(f"  {t:7s} {len(got):2d} timepoints  {sorted(got)}  ({mb:.1f} MB)")
    dn = fetch_peaks(rows, "DNase-seq", None, "DNase")
    man["peaks"]["DNase"] = dn
    print(f"  {'DNase':7s} {len(dn):2d} timepoints  {sorted(dn)}")

    print("\n  RNA-seq gene quantifications (TPM kept, TSVs deleted)")
    cols, genes = fetch_rna(rows)
    if not cols:
        raise SystemExit("no RNA-seq gene quantifications retrieved")
    gidx = {g: i for i, g in enumerate(genes)}
    M = np.zeros((len(cols), len(genes)), dtype=np.float32)
    for j, c in enumerate(cols):
        for g, v in c["tpm"].items():
            i = gidx.get(g)
            if i is not None:
                M[j, i] = v
    np.savez_compressed(DIR / "rna.npz", tpm=M, genes=np.array(genes),
                        mins=np.array([c["min"] for c in cols]),
                        reps=np.array([c["rep"] for c in cols]),
                        exps=np.array([c["exp"] for c in cols]),
                        files=np.array([c["file"] for c in cols]))
    man["rna"] = {"n_columns": len(cols), "n_genes": len(genes),
                  "timepoints": sorted({c["min"] for c in cols}),
                  "columns": [{k: c[k] for k in ("min", "exp", "file", "rep")} for c in cols]}
    json.dump(man, open(DIR / "manifest.json", "w"), indent=1)

    print(f"\n  {len(cols)} RNA columns x {len(genes):,} genes -> {DIR/'rna.npz'}")
    tp = sorted({c['min'] for c in cols})
    print(f"  RNA timepoints: {tp}")
    common = sorted(set(tp) & set(int(k) for k in man["peaks"]["NR3C1"]))
    print(f"  timepoints with BOTH NR3C1 occupancy and expression: {len(common)}  {common}")
    print(f"  -> {DIR/'manifest.json'}")


if __name__ == "__main__":
    main()
