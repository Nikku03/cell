"""FOLDSIMBANK -- measured contact maps of fixed test regions, and the HCT116 stop-sign map, for checking the folding
simulator (foldsim.py) against real cells (G21, foldsimtest.py).

Asked by the user: "Do whatever is needed. Give me final product" (after G19 / G20: build the simulator from the rules and
test whether it reproduces the measured maps).
Regions (fixed here, before anything is simulated): on every autosome four windows of 3 Mb that start at 20%, 35%, 65% and
80% of the chromosome's length (rounded to 25 kb); 88 regions; odd chromosomes DEV, even EXAM (house rule).
Measured maps, raw counts at 25 kb (120 x 120 bins per region), read over HTTPS (hicremote.region):
  K562 at 0 h: Bond et al. 2023 merged map GSE214123_MEGA_K562_WT_0_inter.hic (GEO; local only);
  HCT116 RAD21-degron cells untreated ENCFF750AOC and after 6 h auxin (no cohesin) ENCFF301BWY (ENCODE; free use).
HCT116 stop signs: ENCODE CTCF ChIP-seq peaks ENCFF463FGL (Bernstein lab, experiment ENCSR240PRQ; free use), each scanned
  for the CTCF motif (JASPAR MA0139.1) in the hg38 sequence exactly as the K562 sites were (loopcodebank.py).
Layers: FOLDSIM_K562_MAPS (npz; local only), FOLDSIM_HCT116 (npz).

Run: python3 -I rem/atlas/foldsimbank.py pull | build | bank      (committed before it ran)
"""

from __future__ import annotations
import gzip
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
argv = sys.argv; sys.argv = [sys.argv[0]]
_s = importlib.util.spec_from_file_location("loopcodebank", os.path.join(HERE, "loopcodebank.py"))
lb = importlib.util.module_from_spec(_s); _s.loader.exec_module(lb)
sys.argv = argv
kb, xb, hr, cbb = lb.kb, lb.xb, lb.hr, lb.cbb

RAW = os.path.join(HERE, "_cache", "foldsim")
K562_HIC = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE214nnn/GSE214123/suppl/GSE214123_MEGA_K562_WT_0_inter.hic"
HCT = {"untreated": "ENCFF750AOC", "auxin_6h": "ENCFF301BWY"}
HCT_CTCF = "ENCFF463FGL"
AUTO, BIN, NB, FRACS = lb.AUTO, 25_000, 120, (0.20, 0.35, 0.65, 0.80)
cbb.SRC["encode_hct116_ctcf"] = ("ENCODE HCT116 CTCF ChIP-seq peaks ENCFF463FGL (Bernstein lab)", "https://www.encodeproject.org", "ENCODE: free use")


def regions(chroms):
    """(chrom, first 25 kb bin) of the fixed windows; chroms = {name: length}."""
    return [(c, int(round(f * chroms[c] / BIN))) for c in AUTO for f in FRACS]


def pull():
    rep = {}; os.makedirs(os.path.join(RAW, "maps", "raw"), exist_ok=True)
    p = os.path.join(RAW, "hct116", "raw", HCT_CTCF + ".bed.gz")
    rep[HCT_CTCF] = "kept" if os.path.exists(p) else xb.fetch(lb.ENC.format(HCT_CTCF, "bed.gz"), p)
    out = os.path.join(RAW, "hct116", "raw", "ctcf_site_motifs.npz")
    if not os.path.exists(out):
        P = [l.split("\t") for l in gzip.open(p, "rt")]; P = [f for f in P if f[0].replace("chr", "") in AUTO]
        ch = np.array([f[0].replace("chr", "") for f in P]); st = np.array([int(f[1]) for f in P]); en = np.array([int(f[2]) for f in P]); sig = np.array([float(f[6]) for f in P])
        W = lb.pwm(); tb = lb.TwoBit(lb.TWOBIT); score = np.full(len(P), -99.0); strand = np.array(["+"] * len(P)); pos = np.zeros(len(P), np.int64)
        for c in AUTO:
            arr, size = tb.chrom("chr" + c)
            for i in np.nonzero(ch == c)[0]:
                s, e = max(0, int(st[i]) - 20), min(size, int(en[i]) + 20); sc, sd, off = lb.scan(lb.TwoBit.window(arr, s, e), W); score[i], strand[i], pos[i] = sc, sd, s + off
            print("motif scan", c, flush=True)
        np.savez_compressed(out, chrom=ch, start=st, stop=en, signal=sig, score=score, strand=strand, pos=pos, max=float(W.max(0).sum()), min=float(W.min(0).sum()))
    for name, url, pre in [("k562_0h", K562_HIC, "")] + [(f"hct116_{k}", lb.ENC.format(a, "hic"), "chr") for k, a in HCT.items()]:
        out = os.path.join(RAW, "maps", "raw", name + ".npz")
        if os.path.exists(out):
            continue
        h = hr.HicRemote(url); chroms = {c: h.chroms[pre + c][1] for c in AUTO}; Rg = regions(chroms); M = np.zeros((len(Rg), NB, NB), np.float32)
        for k, (c, b0) in enumerate(Rg):
            M[k] = h.region(pre + c, BIN, b0, b0 + NB)
            if k % 8 == 0:
                print(name, k, c, b0, flush=True)
        np.savez_compressed(out + ".part.npz", maps=M, chrom=np.array([c for c, _ in Rg]), bin0=np.array([b for _, b in Rg]), chrom_names=np.array(AUTO),
                            chrom_len=np.array([chroms[c] for c in AUTO])); os.replace(out + ".part.npz", out); rep[name] = len(Rg)
    json.dump(rep, open(os.path.join(RAW, "pull.json"), "w"), indent=1, default=str)


def build():
    K = np.load(os.path.join(RAW, "maps", "raw", "k562_0h.npz")); U = np.load(os.path.join(RAW, "maps", "raw", "hct116_untreated.npz")); A = np.load(os.path.join(RAW, "maps", "raw", "hct116_auxin_6h.npz"))
    assert list(K["chrom"]) == list(U["chrom"]) and (K["bin0"] == U["bin0"]).all() and (U["bin0"] == A["bin0"]).all()
    np.savez_compressed(os.path.join(cbb.LAY, "foldsim_k562_maps.npz"), maps=K["maps"], chrom=K["chrom"], bin0=K["bin0"], chrom_names=K["chrom_names"], chrom_len=K["chrom_len"])
    S = np.load(os.path.join(RAW, "hct116", "raw", "ctcf_site_motifs.npz"))
    np.savez_compressed(os.path.join(cbb.LAY, "foldsim_hct116.npz"), maps_untreated=U["maps"], maps_auxin=A["maps"], chrom=U["chrom"], bin0=U["bin0"],
                        chrom_names=U["chrom_names"], chrom_len=U["chrom_len"], **{f"ctcf_{k}": S[k] for k in S.files})
    rel = (S["score"] - S["min"]) / (S["max"] - S["min"])
    info = dict(regions=int(len(K["chrom"])), k562_counts_per_region_median=float(np.median(K["maps"].sum((1, 2)))),
                hct116_counts_per_region_median=dict(untreated=float(np.median(U["maps"].sum((1, 2)))), auxin_6h=float(np.median(A["maps"].sum((1, 2))))),
                hct116_ctcf_peaks=int(len(S["chrom"])), hct116_ctcf_with_motif=int((rel >= 0.8).sum()), motif_plus_share=float((S["strand"][rel >= 0.8] == "+").mean()))
    json.dump(info, open(os.path.join(RAW, "build_info.json"), "w"), indent=1); print(json.dumps(info))


def bank():
    info = json.load(open(os.path.join(RAW, "build_info.json")))
    raws = [cbb.layer_fact("FOLDSIM_RAW_MAPS", 1, "bond2023", os.path.join(RAW, "maps", "raw"), "downloaded map regions", "-", "raw counts"),
            cbb.layer_fact("FOLDSIM_RAW_HCT116", 1, "encode_hct116_ctcf", os.path.join(RAW, "hct116", "raw"), "downloaded files", "-", "as published")]
    a = cbb.layer_fact("FOLDSIM_K562_MAPS", 1, "bond2023", os.path.join(cbb.LAY, "foldsim_k562_maps.npz"), f"{info['regions']} regions of 3 Mb", "120 x 120 bins of 25 kb, K562 0 h",
                       "raw contact counts", inputs=(raws[0].ref,), tier="derived")
    b = cbb.layer_fact("FOLDSIM_HCT116", 1, "encode_hct116_rad21", os.path.join(cbb.LAY, "foldsim_hct116.npz"), f"{info['regions']} regions of 3 Mb; {info['hct116_ctcf_peaks']} CTCF sites",
                       "120 x 120 bins of 25 kb with and without cohesin; CTCF sites with motif direction", "raw contact counts; log2-odds", inputs=tuple(f.ref for f in raws), tier="derived")
    n = cbb.bank().add_many(raws + [a, b]); rep = dict(added=time.strftime("%Y-%m-%d"), facts_written=n, layers={f.entity: f.ref for f in raws + [a, b]}, build=info)
    json.dump(rep, open(R("outputs", "cellbrain", "foldsim_report.json"), "w"), indent=1, default=int); print(json.dumps(rep, default=int)[:300])


if __name__ == "__main__":
    {"pull": pull, "build": build, "bank": bank}[sys.argv[1]]()
