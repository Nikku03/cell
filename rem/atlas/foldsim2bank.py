"""FOLDSIM2BANK -- data for making the folding simulator better and testing it harder (G22, foldsim2test.py).

Asked by the user: "Well do all 5" -- (1) measure the best possible score, (2) add the missing stop signs and anchors,
(3) test on 'break one part' experiments, (4) make it truly 3D and check against microscope distances, (5) more cells,
including single cells.
Fixed regions: the same 88 windows of 3 Mb at 25 kb as foldsimbank.py (odd chromosomes DEV, even EXAM).
Sources (each file read as data):
  (1) K562 repeat maps -- Bond et al. 2023 (GEO GSE214123; local only): per-replicate .hic of 0 h; half A = biological
      replicates 1 and 2 (6 files), half B = replicates 3 and 4 (4 files); region maps summed per half.
  (3) HCT116 'break one part' maps -- ENCODE intact Hi-C (Aiden lab; free use), degron lines: CTCF untreated ENCFF029ISF /
      6 h auxin ENCFF661HPC; RNA polymerase II (POLR2A) untreated ENCFF037MOQ / 6 h auxin ENCFF518GOJ.
  (4, 5) microscope traces of single chromosomes -- Bintu et al. 2018 Science 362:eaau1783 (tables from
      github.com/BogdanBintu/ChromatinImaging, Data/; no licence stated -> local only): 3D positions (nm) of consecutive
      30 kb segments: K562 chr21:28-30 Mb; HCT116 RAD21-degron chr21:34.6-37.1 Mb untreated and after 6 h auxin
      (coordinates taken as hg38). Kept here: per data set the median distance between every two segments and, per
      position, the share of single chromosomes with a domain border there (foldsim.border_calls).
  (5) more cell types -- ENCODE (free use): contact maps GM12878 ENCFF457YAT, IMR-90 ENCFF281ILS, HepG2 ENCFF306VTV;
      CTCF ChIP-seq peaks GM12878 ENCFF356LIU, IMR-90 ENCFF307XFM, HepG2 ENCFF612ZUY, scanned for the CTCF motif as before;
      H3K27ac peaks of HCT116, GM12878, IMR-90, HepG2 (the first ENCODE experiment by accession with a default peak file).
Layers: FOLDSIM2_LOCAL (npz; K562 half maps + trace summaries; local only), FOLDSIM2_ENCODE (npz; maps and sites).

Run: python3 -I rem/atlas/foldsim2bank.py pull | build | bank      (committed before it ran)
"""

from __future__ import annotations
import gzip
import importlib.util
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
argv = sys.argv; sys.argv = [sys.argv[0]]


def _mod(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


fb = _mod("foldsimbank"); fs = _mod("foldsim")
sys.argv = argv
lb, kb, xb, hr, cbb = fb.lb, fb.kb, fb.xb, fb.hr, fb.cbb

RAW = os.path.join(HERE, "_cache", "foldsim2")
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/"
ENC_MAPS = dict(hct116_ctcf_untreated="ENCFF029ISF", hct116_ctcf_auxin="ENCFF661HPC", hct116_pol2_untreated="ENCFF037MOQ", hct116_pol2_auxin="ENCFF518GOJ",
                gm12878="ENCFF457YAT", imr90="ENCFF281ILS", hepg2="ENCFF306VTV")
CTCF = dict(gm12878="ENCFF356LIU", imr90="ENCFF307XFM", hepg2="ENCFF612ZUY")
AC_CELLS = {"hct116": "HCT116", "gm12878": "GM12878", "imr90": "IMR-90", "hepg2": "HepG2"}
TRACES = dict(k562_28=("K562_chr21-28-30Mb.csv", 28_000_000), hct116_34_untreated=("HCT116_chr21-34-37Mb_untreated.csv", 34_600_000),
              hct116_34_auxin=("HCT116_chr21-34-37Mb_6h auxin.csv", 34_600_000))
AUTO, BIN, NB = fb.AUTO, fb.BIN, fb.NB
cbb.SRC["bintu2018"] = ("Bintu et al. 2018 Science: chromatin tracing of single chromosomes (github.com/BogdanBintu/ChromatinImaging)", "https://doi.org/10.1126/science.aau1783",
                        "no licence stated (local only)")
cbb.SRC["encode_foldsim2"] = ("ENCODE: HCT116 CTCF / POLR2A degron intact Hi-C; GM12878, IMR-90, HepG2 Hi-C, CTCF and H3K27ac peaks", "https://www.encodeproject.org", "ENCODE: free use")
cbb.LOCAL_ONLY.add("bintu2018")


def region_maps(url, pre, out, tag):
    if os.path.exists(out):
        return np.load(out)["maps"]
    h = hr.HicRemote(url); chroms = {c: h.chroms[pre + c][1] for c in AUTO}; Rg = fb.regions(chroms); M = np.zeros((len(Rg), NB, NB), np.float32)
    for k, (c, b0) in enumerate(Rg):
        M[k] = h.region(pre + c, BIN, b0, b0 + NB)
    np.savez_compressed(out + ".part.npz", maps=M); os.replace(out + ".part.npz", out); print("maps", tag, int(M.sum()), flush=True)
    return M


def scan_peaks(path, out):
    if os.path.exists(out):
        return
    P = [l.split("\t") for l in gzip.open(path, "rt")]; P = [f for f in P if f[0].replace("chr", "") in AUTO]
    ch = np.array([f[0].replace("chr", "") for f in P]); st = np.array([int(f[1]) for f in P]); en = np.array([int(f[2]) for f in P]); sig = np.array([float(f[6]) for f in P])
    W = lb.pwm(); tb = lb.TwoBit(lb.TWOBIT); score = np.full(len(P), -99.0); strand = np.array(["+"] * len(P)); pos = np.zeros(len(P), np.int64)
    for c in AUTO:
        arr, size = tb.chrom("chr" + c)
        for i in np.nonzero(ch == c)[0]:
            s, e = max(0, int(st[i]) - 20), min(size, int(en[i]) + 20); sc, sd, off = lb.scan(lb.TwoBit.window(arr, s, e), W); score[i], strand[i], pos[i] = sc, sd, s + off
    np.savez_compressed(out, chrom=ch, start=st, stop=en, signal=sig, score=score, strand=strand, pos=pos, max=float(W.max(0).sum()), min=float(W.min(0).sum()))
    print("motif scan", os.path.basename(out), len(P), flush=True)


def ac_file(cell):
    """first ENCODE H3K27ac experiment (by accession) of the plain cell line with a default GRCh38 peak file."""
    u = ("https://www.encodeproject.org/search/?type=Experiment&assay_title=Histone+ChIP-seq&target.label=H3K27ac&status=released&limit=all&frame=embedded&format=json"
         "&biosample_ontology.term_name=" + cell)
    g = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"Accept": "application/json", "User-Agent": "cellbrain/1.0"}), timeout=300).read())["@graph"]
    for e in sorted(g, key=lambda e: e["accession"]):
        if e.get("biosample_summary", "").strip() != "Homo sapiens " + cell:
            continue
        fl = sorted((f for f in e["files"] if f.get("status") == "released" and f.get("assembly") == "GRCh38" and f.get("file_format") == "bed" and f.get("preferred_default")
                     and "peaks" in f.get("output_type", "")), key=lambda f: (f["output_type"] != "replicated peaks", f["accession"]))
        if fl:
            return e["accession"], fl[0]["accession"]
    raise SystemExit("no H3K27ac peak file for " + cell)


def pull():
    rep = {}; M = os.path.join(RAW, "maps", "raw"); S = os.path.join(RAW, "sites", "raw"); os.makedirs(M, exist_ok=True); os.makedirs(S, exist_ok=True)
    get = lambda url, path: "kept" if os.path.exists(path) else xb.fetch(url, path)
    for k, acc in CTCF.items():
        p = os.path.join(S, f"ctcf_{k}_{acc}.bed.gz"); rep[acc] = get(lb.ENC.format(acc, "bed.gz"), p); scan_peaks(p, os.path.join(S, f"ctcf_{k}_motifs.npz"))
    for k, cell in AC_CELLS.items():
        mp = os.path.join(S, f"h3k27ac_{k}.json")
        if not os.path.exists(mp):
            exp, acc = ac_file(cell); json.dump(dict(experiment=exp, file=acc), open(mp, "w"))
        acc = json.load(open(mp))["file"]; rep[f"h3k27ac_{k}"] = acc; get(lb.ENC.format(acc, "bed.gz"), os.path.join(S, f"h3k27ac_{k}_{acc}.bed.gz"))
    for k, acc in ENC_MAPS.items():
        region_maps(lb.ENC.format(acc, "hic"), "chr", os.path.join(M, k + ".npz"), k)
    fl = os.path.join(RAW, "maps", "filelist_GSE214123.txt"); get(GEO + "series/GSE214nnn/GSE214123/suppl/filelist.txt", fl)
    halves = {"A": [], "B": []}
    for line in open(fl).read().splitlines():
        m = re.match(r"File\t((GSM\d+)_MEGA_K562_WT_PMA_0_([12])_(\d)_inter\.hic)", line)
        if m:
            halves["A"].append((m.group(1), GEO + f"samples/{m.group(2)[:-3]}nnn/{m.group(2)}/suppl/{m.group(1)}"))
    for b in (3, 4):
        for t in (1, 2):
            f = f"GSE214123_MEGA_K562_WT_PMA_0_{b}_{t}_inter.hic"; halves["B"].append((f, GEO + "series/GSE214nnn/GSE214123/suppl/" + f))
    rep["k562_halves"] = {h: [f for f, _ in v] for h, v in halves.items()}
    for h, v in halves.items():
        out = os.path.join(M, f"k562_half{h}.npz")
        if not os.path.exists(out):
            tot = sum(region_maps(url, "", os.path.join(M, "part_" + f + ".npz"), f) for f, url in v)
            np.savez_compressed(out, maps=tot.astype(np.float32))
            for f, _ in v:
                os.remove(os.path.join(M, "part_" + f + ".npz"))
    json.dump(rep, open(os.path.join(RAW, "pull.json"), "w"), indent=1, default=str)


def traces(name):
    f, start = TRACES[name]; rows = np.loadtxt(os.path.join(RAW, "tracing", "raw", f), delimiter=",", skiprows=2)
    ci = rows[:, 0].astype(int); si = rows[:, 1].astype(int); n, m = ci.max(), si.max(); X = np.full((n, m, 3), np.nan); X[ci - 1, si - 1] = rows[:, [3, 4, 2]]
    return X, start


def build():
    M = os.path.join(RAW, "maps", "raw"); S = os.path.join(RAW, "sites", "raw"); loc, enc = {}, {}
    for h in ("A", "B"):
        loc[f"maps_k562_half{h}"] = np.load(os.path.join(M, f"k562_half{h}.npz"))["maps"]
    info = dict(traces={})
    for name in TRACES:
        X, start = traces(name); n = X.shape[1]; D = np.full((n, n), np.nan); cnt = np.zeros((n, n))
        for i in range(n):
            d = np.sqrt(((X[:, i:i + 1] - X) ** 2).sum(2)); D[i] = np.nanmedian(d, 0); cnt[i] = np.isfinite(d).sum(0)
        bp, used = fs.border_calls(X)
        loc[f"trace_{name}_median_nm"] = D; loc[f"trace_{name}_pairs"] = cnt; loc[f"trace_{name}_border_share"] = bp; loc[f"trace_{name}_start"] = start
        info["traces"][name] = dict(chromosomes=int(X.shape[0]), segments=int(n), chromosomes_with_calls=int(used), median_neighbour_nm=float(np.nanmedian(np.diag(D, 1))))
        print("traces", name, info["traces"][name], flush=True)
    for k in ENC_MAPS:
        enc[f"maps_{k}"] = np.load(os.path.join(M, k + ".npz"))["maps"]
    for k in CTCF:
        Z = np.load(os.path.join(S, f"ctcf_{k}_motifs.npz"))
        for f in Z.files:
            enc[f"ctcf_{k}_{f}"] = Z[f]
    for k in AC_CELLS:
        acc = json.load(open(os.path.join(S, f"h3k27ac_{k}.json")))["file"]; P = [l.split("\t") for l in gzip.open(os.path.join(S, f"h3k27ac_{k}_{acc}.bed.gz"), "rt")]
        P = [f for f in P if f[0].replace("chr", "") in AUTO]
        enc[f"ac_{k}_chrom"] = np.array([f[0].replace("chr", "") for f in P]); enc[f"ac_{k}_pos"] = np.array([(int(f[1]) + int(f[2])) // 2 for f in P]); enc[f"ac_{k}_signal"] = np.array([float(f[6]) for f in P])
    np.savez_compressed(os.path.join(cbb.LAY, "foldsim2_local.npz"), **loc); np.savez_compressed(os.path.join(cbb.LAY, "foldsim2_encode.npz"), **enc)
    info.update(maps={k: float(np.median(enc[f"maps_{k}"].sum((1, 2)))) for k in ENC_MAPS}, k562_half_counts=[float(np.median(loc[f"maps_k562_half{h}"].sum((1, 2)))) for h in "AB"],
                ctcf={k: int(len(enc[f"ctcf_{k}_chrom"])) for k in CTCF}, h3k27ac={k: int(len(enc[f"ac_{k}_chrom"])) for k in AC_CELLS})
    json.dump(info, open(os.path.join(RAW, "build_info.json"), "w"), indent=1); print(json.dumps(info)[:900])


def bank():
    info = json.load(open(os.path.join(RAW, "build_info.json")))
    raws = [cbb.layer_fact("FOLDSIM2_RAW_MAPS", 1, "encode_foldsim2", os.path.join(RAW, "maps", "raw"), "downloaded map regions", "-", "raw counts"),
            cbb.layer_fact("FOLDSIM2_RAW_SITES", 1, "encode_foldsim2", os.path.join(RAW, "sites", "raw"), "downloaded files", "-", "as published"),
            cbb.layer_fact("FOLDSIM2_RAW_TRACING", 1, "bintu2018", os.path.join(RAW, "tracing", "raw"), "downloaded tables", "-", "nm")]
    a = cbb.layer_fact("FOLDSIM2_LOCAL", 1, "bintu2018", os.path.join(cbb.LAY, "foldsim2_local.npz"), "K562 half maps (88 regions); 3 trace data sets", "median distances; border share per position",
                       "raw counts; nm; fraction", inputs=(raws[0].ref, raws[2].ref), tier="derived")
    b = cbb.layer_fact("FOLDSIM2_ENCODE", 1, "encode_foldsim2", os.path.join(cbb.LAY, "foldsim2_encode.npz"), "7 map sets x 88 regions; CTCF and H3K27ac sites of 3-4 cell types", "-",
                       "raw counts; log2-odds; signal", inputs=(raws[0].ref, raws[1].ref), tier="derived")
    n = cbb.bank().add_many(raws + [a, b]); rep = dict(added=time.strftime("%Y-%m-%d"), facts_written=n, layers={f.entity: f.ref for f in raws + [a, b]}, build=info)
    json.dump(rep, open(R("outputs", "cellbrain", "foldsim2_report.json"), "w"), indent=1, default=int); print(json.dumps(rep, default=int)[:300])


if __name__ == "__main__":
    {"pull": pull, "build": build, "bank": bank}[sys.argv[1]]()
