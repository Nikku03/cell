"""LOOPCODEBANK -- data to test the RULES that fold a chromosome (G19 static rules, G20 dynamics; loopcodetest.py,
loopdyntest.py): where loops form, what makes them, what sets the open / closed zones, and what changes them.

Asked by the user: "crack the loop mechanism which is guiding the dynamic chromosome folding ... what guides them, how?
... download any data you need, also check the memory bank".
The working model being tested (from the field: loop extrusion + compartment affinity), piece by piece on data:
  a ring-shaped motor (cohesin) reels DNA in from both sides until each side meets a CTCF protein whose DNA site points
  INTO the loop; chromatin of the same activity state sticks together (open with open, closed with closed).
Sources (each file read as data):
  K562 sites and their change over time -- Bond et al. 2023 (GEO GSE213908 / GSE213295; local only): per-site count
    tables of CUT&RUN CTCF, RAD21 (cohesin), H3K27ac, JUN and ATAC at 0, 6 h, 72 h of PMA (2 replicates; RAD21 0 and
    72 h); hg38. Loops, compartments and RNA come from layer K562_PMA_3D (k3dbank.py).
  CTCF site direction -- JASPAR MA0139.1 (CTCF; CC BY 4.0) scanned over each CTCF site in the hg38 sequence (UCSC
    hg38.2bit, read remotely, nothing stored but the hits): log2-odds score, best hit on either strand, relative score.
  Chromatin marks in K562 -- ENCODE (free use) peak files, Bernstein lab: H3K27ac ENCFF038DDS, H3K4me1 ENCFF135ZLM,
    H3K4me3 ENCFF909PMV, H3K36me3 ENCFF053DAC, H3K27me3 ENCFF049HUP, H3K9me3 ENCFF963GZJ (GRCh38).
  Removing the motor -- ENCODE (free use; Aiden lab; Rao et al. 2017 Cell re-processed on GRCh38): in situ Hi-C of
    HCT116 cells whose cohesin subunit RAD21 carries a degron: untreated ENCFF750AOC, auxin 6 h (cohesin destroyed)
    ENCFF301BWY, and 20 / 40 / 60 / 180 min after auxin is washed out (cohesin returns) ENCFF775PBO, ENCFF944AII,
    ENCFF545AXP, ENCFF085TWV; loops called in untreated cells ENCFF402GDT. Read over HTTPS (hicremote.py): per loop
    the 3 x 3 pixels at 10 kb around its centre and around two distance-matched control positions (both ends shifted
    100 kb left, and 100 kb right); 100 kb maps of untreated and auxin 6 h for compartments.
Layers: LOOPCODE_K562 (npz; local only), LOOPCODE_HCT116 (npz).

Run: python3 -I rem/atlas/loopcodebank.py pull | build | bank      (committed before it ran)
"""

from __future__ import annotations
import collections
import gzip
import importlib.util
import json
import os
import struct
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
argv = sys.argv; sys.argv = [sys.argv[0]]
_s = importlib.util.spec_from_file_location("k3dbank", os.path.join(HERE, "k3dbank.py"))
kb = importlib.util.module_from_spec(_s); _s.loader.exec_module(kb)
sys.argv = argv
xb, hr, cbb = kb.xb, kb.hr, kb.cbb

RAW = os.path.join(HERE, "_cache", "loopcode")
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE213nnn/{}/suppl/{}"
TABLES = {"ctcf": ("GSE213908", "GSE213908_CTCFSummary.txt.gz"), "rad21": ("GSE213908", "GSE213908_Rad21Summary.txt.gz"),
          "h3k27ac": ("GSE213908", "GSE213908_H3K27Summary.txt.gz"), "jun": ("GSE213908", "GSE213908_JunSummary.txt.gz"),
          "atac": ("GSE213295", "GSE213295_atacSummary.txt.gz")}
JASPAR = "https://jaspar.elixir.no/api/v1/matrix/MA0139.1.pfm"
TWOBIT = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit"
ENC = "https://www.encodeproject.org/files/{0}/@@download/{0}.{1}"
MARKS = {"H3K27ac": "ENCFF038DDS", "H3K4me1": "ENCFF135ZLM", "H3K4me3": "ENCFF909PMV", "H3K36me3": "ENCFF053DAC", "H3K27me3": "ENCFF049HUP", "H3K9me3": "ENCFF963GZJ"}
HCT = {"untreated": "ENCFF750AOC", "auxin_6h": "ENCFF301BWY", "washout_20": "ENCFF775PBO", "washout_40": "ENCFF944AII", "washout_60": "ENCFF545AXP",
       "washout_180": "ENCFF085TWV"}
HCT_LOOPS = "ENCFF402GDT"
AUTO = [str(c) for c in range(1, 23)]
BIN, LOOP_BIN, SHIFT = 100_000, 10_000, 10
cbb.SRC["jaspar_ctcf"] = ("JASPAR MA0139.1 CTCF matrix", "https://jaspar.elixir.no/matrix/MA0139.1/", "CC BY 4.0")
cbb.SRC["encode_k562_marks"] = ("ENCODE K562 histone ChIP-seq peaks (Bernstein lab)", "https://www.encodeproject.org", "ENCODE: free use")
cbb.SRC["encode_hct116_rad21"] = ("ENCODE HCT116 RAD21-degron in situ Hi-C (Aiden lab; Rao et al. 2017 Cell)", "https://www.encodeproject.org", "ENCODE: free use")


def _get(url, start, n, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + n - 1}", "User-Agent": "cellbrain/1.0"})
            with urllib.request.urlopen(req, timeout=900) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * 2 ** k)


class TwoBit:
    """UCSC .2bit read over HTTPS: one chromosome's packed DNA at a time (2 bits per base: T, C, A, G)."""
    def __init__(self, url):
        self.url = url; n = 1 << 16
        while True:
            b = _get(url, 0, n)
            try:
                sig, ver, cnt, _ = struct.unpack_from("<IIII", b, 0)
                if sig != 0x1A412743:
                    raise ValueError("not a little-endian .2bit file")
                p = 16; self.idx = {}
                for _ in range(cnt):
                    ln = b[p]; name = b[p + 1:p + 1 + ln].decode(); off = struct.unpack_from("<I", b, p + 1 + ln)[0]; self.idx[name] = off; p += 5 + ln
                break
            except (struct.error, IndexError):
                n *= 4

    def chrom(self, name):
        off = self.idx[name]; size, nn = struct.unpack("<II", _get(self.url, off, 8))
        mc = struct.unpack("<I", _get(self.url, off + 8 + 8 * nn, 4))[0]
        start = off + 8 + 8 * nn + 4 + 8 * mc + 4; nb = (size + 3) // 4; parts = []
        for a in range(0, nb, 1 << 25):
            parts.append(_get(self.url, start + a, min(1 << 25, nb - a)))
        return np.frombuffer(b"".join(parts), np.uint8), size

    @staticmethod
    def window(arr, s, e):
        """bases s..e as indices into A, C, G, T."""
        b = arr[s // 4:(e + 3) // 4]; codes = ((b[:, None] >> np.array([6, 4, 2, 0], np.uint8)) & 3).ravel()[s % 4:s % 4 + (e - s)]
        return np.array([3, 1, 0, 2], np.int8)[codes]


def pwm():
    rows = [l.split() for l in open(os.path.join(RAW, "motif", "raw", "MA0139.1.pfm")).read().splitlines() if l and not l.startswith(">")]
    C = np.array(rows, float); n = C.sum(0); f = (C + np.sqrt(n) * 0.25) / (n + np.sqrt(n)); return np.log2(f / 0.25)        # rows A, C, G, T


def scan(seq, W):
    """best log-odds hit of W on either strand of seq (ACGT indices): (score, strand, offset)."""
    L = W.shape[1]; n = len(seq) - L + 1
    if n <= 0:
        return -99.0, "+", 0
    Wr = W[::-1, ::-1]; sp = np.zeros(n); sm = np.zeros(n)
    for k in range(L):
        sp += W[seq[k:k + n], k]; sm += Wr[seq[k:k + n], k]
    ip, im = int(sp.argmax()), int(sm.argmax())
    return (float(sp[ip]), "+", ip) if sp[ip] >= sm[im] else (float(sm[im]), "-", im)


def table(key):
    L = [l.rstrip("\n").split("\t") for l in gzip.open(os.path.join(RAW, "k562", "raw", TABLES[key][1]), "rt")]
    h = L[0]; rows = [r[1:] if len(r) == len(h) + 1 else r for r in L[1:]]; c = {k: i for i, k in enumerate(h)}
    cc = [k for k in h if k.startswith("MEGA_")]
    tm = [int(k.split("_S_")[1].split("_")[0]) for k in cc]; rp = [int(k.split("_S_")[1].split("_")[1].split(".")[0]) for k in cc]
    keep = [r for r in rows if r[c["chr"]].replace("chr", "") in AUTO + ["X"]]
    return dict(chrom=np.array([r[c["chr"]].replace("chr", "") for r in keep]), start=np.array([int(r[c["start"]]) for r in keep]),
                stop=np.array([int(r[c["stop"]]) for r in keep]), counts=np.array([[kb.fnum(r[c[k]]) for k in cc] for r in keep], np.float32),
                cols=np.array([f"{t}_{r}" for t, r in zip(tm, rp)]))


def hct_loops():
    out = []
    for l in gzip.open(os.path.join(RAW, "hct116", "raw", HCT_LOOPS + ".bedpe.gz"), "rt"):
        if l.startswith("#"):
            continue
        f = l.split("\t"); ch = f[0].replace("chr", "")
        if f[0] == f[3] and ch in AUTO:
            a, b = sorted((int(f[21]), int(f[22]))); out.append((ch, a, b))
    return sorted(out, key=lambda r: (AUTO.index(r[0]), r[1], r[2]))


def pull():
    rep = {}; get = lambda url, path: "kept" if os.path.exists(path) else xb.fetch(url, path)
    for k, (gse, f) in TABLES.items():
        rep[f] = get(GEO.format(gse, f), os.path.join(RAW, "k562", "raw", f))
    rep["jaspar"] = get(JASPAR, os.path.join(RAW, "motif", "raw", "MA0139.1.pfm"))
    for m, acc in MARKS.items():
        rep[acc] = get(ENC.format(acc, "bed.gz"), os.path.join(RAW, "marks", "raw", f"{m}_{acc}.bed.gz"))
    rep[HCT_LOOPS] = get(ENC.format(HCT_LOOPS, "bedpe.gz"), os.path.join(RAW, "hct116", "raw", HCT_LOOPS + ".bedpe.gz"))
    out = os.path.join(RAW, "motif", "raw", "ctcf_site_motifs.npz")                # CTCF site direction from the sequence
    if not os.path.exists(out):
        T = table("ctcf"); W = pwm(); tb = TwoBit(TWOBIT); n = len(T["chrom"])
        score = np.full(n, -99.0); strand = np.array(["+"] * n); pos = np.zeros(n, np.int64)
        for c in AUTO + ["X"]:
            arr, size = tb.chrom("chr" + c)
            for i in np.nonzero(T["chrom"] == c)[0]:
                s, e = max(0, int(T["start"][i]) - 20), min(size, int(T["stop"][i]) + 20)
                sc, st, off = scan(TwoBit.window(arr, s, e), W); score[i], strand[i], pos[i] = sc, st, s + off
            print("motif scan", c, flush=True)
        np.savez_compressed(out, score=score, strand=strand, pos=pos, max=float(W.max(0).sum()), min=float(W.min(0).sum()))
    L = hct_loops(); rep["hct116_loops"] = len(L)
    for cond, acc in HCT.items():
        out = os.path.join(RAW, "hct116", "raw", f"{cond}.npz")
        if os.path.exists(out):
            continue
        h = hr.HicRemote(ENC.format(acc, "hic")); arr = {}; obs = np.zeros(len(L)); cl = np.zeros(len(L)); crr = np.zeros(len(L))
        for c in AUTO:
            ks = [k for k, r in enumerate(L) if r[0] == c]; P = [(L[k][1] // LOOP_BIN, L[k][2] // LOOP_BIN) for k in ks]
            v = h.pixels("chr" + c, LOOP_BIN, P + [(i - SHIFT, j - SHIFT) for i, j in P] + [(i + SHIFT, j + SHIFT) for i, j in P], pad=1)
            obs[ks], cl[ks], crr[ks] = v[:len(P)], v[len(P):2 * len(P)], v[2 * len(P):]
            if cond in ("untreated", "auxin_6h"):
                i, j, w = h.intra("chr" + c, BIN); arr[f"{c}_i"], arr[f"{c}_j"], arr[f"{c}_v"] = i, j, w.astype(np.float32)
        np.savez_compressed(out + ".part.npz", obs=obs, ctrl_left=cl, ctrl_right=crr, **arr); os.replace(out + ".part.npz", out)
        print("hct116", cond, acc, "loop pixels", int(obs.sum()), int(cl.sum()), int(crr.sum()), flush=True)
    json.dump(rep, open(os.path.join(RAW, "pull.json"), "w"), indent=1, default=str)


def e1_tracks(maps, tss_pc):
    """compartment E1 per 100 kb bin for a list of named maps; the first is turned by gene density, the rest by the first."""
    E, chrom_of, start_of, info = [], [], [], {}
    for c in AUTO:
        n = int(max(z[f"{c}_j"].max() for _, z in maps)) + 1; dens = np.zeros(n)
        for p_ in tss_pc.get(c, []):
            if p_ // BIN < n:
                dens[p_ // BIN] += 1
        vec = []
        for k, (name, z) in enumerate(maps):
            V = kb.eig3(z[f"{c}_i"], z[f"{c}_j"], z[f"{c}_v"].astype(np.float64), n); ref = dens if k == 0 else vec[0]
            rs = [np.corrcoef(V[:, q][~np.isnan(V[:, q]) & ~np.isnan(ref)], ref[~np.isnan(V[:, q]) & ~np.isnan(ref)])[0, 1] for q in range(3)]
            q = int(np.argmax(np.abs(rs))); vec.append(kb.zs(V[:, q] * np.sign(rs[q]))); info.setdefault(c, {})[name] = dict(pc=q + 1, r=round(float(abs(rs[q])), 3))
        E.append(np.stack(vec, 1)); chrom_of += [c] * n; start_of += list(range(0, n * BIN, BIN))
    return np.concatenate(E), np.array(chrom_of), np.array(start_of), info


def build():
    Z = np.load(os.path.join(cbb.LAY, "k562_pma_3d.npz")); M = np.load(os.path.join(RAW, "motif", "raw", "ctcf_site_motifs.npz"))
    T = {k: table(k) for k in TABLES}
    bkey = {(c, s): k for k, (c, s) in enumerate(zip(Z["bin_chrom"], Z["bin_start"]))}; nb = len(Z["bin_chrom"])
    cover = {}
    for m, acc in MARKS.items():                                              # share of each 100 kb bin covered by peaks of the mark
        cv = np.zeros(nb)
        for l in gzip.open(os.path.join(RAW, "marks", "raw", f"{m}_{acc}.bed.gz"), "rt"):
            f = l.split("\t"); c = f[0].replace("chr", ""); s, e = int(f[1]), int(f[2])
            for b0 in range(s // BIN * BIN, e, BIN):
                k = bkey.get((c, b0))
                if k is not None:
                    cv[k] += min(e, b0 + BIN) - max(s, b0)
        cover[m] = np.minimum(cv / BIN, 1.0)
    out = dict(motif_score=M["score"], motif_strand=M["strand"], motif_pos=M["pos"], motif_max=M["max"], motif_min=M["min"],
               bin_chrom=Z["bin_chrom"], bin_start=Z["bin_start"], mark_names=np.array(list(MARKS)), mark_cover=np.stack([cover[m] for m in MARKS], 1))
    for k, t in T.items():
        for f in ("chrom", "start", "stop", "counts", "cols"):
            out[f"{k}_{f}"] = t[f]
    np.savez_compressed(os.path.join(cbb.LAY, "loopcode_k562.npz"), **out)
    tss = [g for g in json.load(open(os.path.join(kb.AG, "raw", "gencode46_tss.json"))) if g[2] == "protein_coding"]
    tss_pc = collections.defaultdict(list)
    for g in tss:
        tss_pc[g[3].replace("chr", "")].append(g[4] - 1)
    H = {c: np.load(os.path.join(RAW, "hct116", "raw", f"{c}.npz")) for c in HCT}; L = hct_loops()
    E1, bchr, bstart, cinfo = e1_tracks([("untreated", H["untreated"]), ("auxin_6h", H["auxin_6h"])], tss_pc)
    np.savez_compressed(os.path.join(cbb.LAY, "loopcode_hct116.npz"), loop_chrom=np.array([r[0] for r in L]), loop_a=np.array([r[1] for r in L]),
                        loop_b=np.array([r[2] for r in L]), conditions=np.array(list(HCT)), obs=np.stack([H[c]["obs"] for c in HCT], 1),
                        ctrl=np.stack([(H[c]["ctrl_left"] + H[c]["ctrl_right"]) / 2 for c in HCT], 1), bin_chrom=bchr, bin_start=bstart,
                        e1=E1.astype(np.float32), e1_cols=np.array(["untreated", "auxin_6h"]))
    rel = (M["score"] - M["min"]) / (M["max"] - M["min"])
    info = dict(k562={k: int(len(t["chrom"])) for k, t in T.items()}, ctcf_sites_with_motif_rel_0_8=int((rel >= 0.8).sum()),
                motif_plus_share=float((M["strand"][rel >= 0.8] == "+").mean()), marks={m: round(float(cover[m].mean()), 4) for m in MARKS},
                hct116=dict(loops=len(L), compartments=cinfo, loop_pixel_counts={c: int(H[c]["obs"].sum()) for c in HCT}))
    json.dump(info, open(os.path.join(RAW, "build_info.json"), "w"), indent=1); print(json.dumps({k: v for k, v in info.items() if k != "hct116"}), info["hct116"]["loops"], info["hct116"]["loop_pixel_counts"])


def bank():
    info = json.load(open(os.path.join(RAW, "build_info.json")))
    raws = [cbb.layer_fact(f"LOOPCODE_RAW_{k.upper()}", 1, src, os.path.join(RAW, k, "raw"), "downloaded files", "-", "as published")
            for k, src in (("k562", "bond2023"), ("motif", "jaspar_ctcf"), ("marks", "encode_k562_marks"), ("hct116", "encode_hct116_rad21"))]
    a = cbb.layer_fact("LOOPCODE_K562", 1, "bond2023", os.path.join(cbb.LAY, "loopcode_k562.npz"), f"{info['k562']['ctcf']} CTCF sites with direction; cohesin, H3K27ac, JUN, ATAC sites; 100-kb bins",
                       "counts at 0, 6, 72 h (2 reps); motif score + strand; share of bin covered by 6 histone marks", "raw counts; log2-odds; fraction",
                       inputs=tuple(f.ref for f in raws[:3]) + ("LAYER-K562_PMA_3D",), tier="derived")
    b = cbb.layer_fact("LOOPCODE_HCT116", 1, "encode_hct116_rad21", os.path.join(cbb.LAY, "loopcode_hct116.npz"), f"{info['hct116']['loops']} loops; 100-kb bins",
                       "loop and control pixel counts in 6 conditions (cohesin present / destroyed / returning); compartment E1 (untreated, auxin 6 h)",
                       "raw counts; E1 z per chromosome", inputs=(raws[3].ref,), tier="derived")
    n = cbb.bank().add_many(raws + [a, b])
    rep = dict(added=time.strftime("%Y-%m-%d"), facts_written=n, layers={f.entity: f.ref for f in raws + [a, b]}, build=info)
    json.dump(rep, open(R("outputs", "cellbrain", "loopcode_report.json"), "w"), indent=1, default=int); print(json.dumps(rep, default=int)[:500])


if __name__ == "__main__":
    {"pull": pull, "build": build, "bank": bank}[sys.argv[1]]()
