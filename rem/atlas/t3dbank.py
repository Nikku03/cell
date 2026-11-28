"""T3DBANK -- THP-1 macrophages switched on by LPS + interferon-gamma, with the 3D chromosome map AND gene expression at
the SAME 8 time points, added to the memory bank (cellbrain) for G17 (t3dtest.py), the repeat of G16 (K562) with many
in-between times.

Asked by the user: "yes, run it on THP-1".
Source (Reed et al. 2022 Cell Reports 41:111567, doi 10.1016/j.celrep.2022.111567; GEO SuperSeries GSE201376; no licence
stated -> local only), times 0, 30, 60, 90, 120, 240, 360, 1440 min:
  Hi-C (GSE201353): merged in situ Hi-C maps per time point ('megaMap', .hic v8, hg38, 20-23 GB each), read over HTTPS
    with byte ranges (hicremote.py; only 100 kb intra-chromosomal counts kept); the authors' loop table
    (GSE201353_LIMA_HiC_diffLoops.tsv.gz: 42,690 loops at 10 kb, clusters static / up.early / up.late / down.early /
    down.late, log2 fold change 0 h vs each time, raw counts at each time in 5 replicate columns r1-r5).
  RNA-seq (GSE201354): Salmon transcript quantities and coverage bigWigs per replicate (2: labelled 2 and 3).
  All THP-1 files are on the older genome build GRCh37 / hg19 (chr1 = 249,250,621 bp in the .hic headers, bigWigs and
  loop table), so genes are placed with GENCODE release 46 mapped to GRCh37 ('lift37' comprehensive GTF; same gene and
  transcript ids as release 46; TSS = gene start on + strand, gene end on - strand).
Built exactly as k3dbank.py (same functions): gene TPM, clean-intron signal, compartment E1 per 100 kb bin per time
point (at 0 h the eigenvector that follows gene density, later the one that follows 0 h), loops as published (cluster
up.* -> 'gained', down.* -> 'lost'; the original label kept).
Layer: THP1_LPIF_3D (npz). Raw downloads stay in _cache/thp1lpif (local only).

Run: python3 -I rem/atlas/t3dbank.py pull | build | bank      (committed before it ran)
"""

from __future__ import annotations
import collections
import gzip
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
argv = sys.argv; sys.argv = [sys.argv[0]]
_s = importlib.util.spec_from_file_location("k3dbank", os.path.join(HERE, "k3dbank.py"))
kb = importlib.util.module_from_spec(_s); _s.loader.exec_module(kb)
sys.argv = argv
xb, hr, cbb = kb.xb, kb.hr, kb.cbb

RAW = os.path.join(HERE, "_cache", "thp1lpif")
FTP = "https://ftp.ncbi.nlm.nih.gov/geo/"
TIMES = [0, 30, 60, 90, 120, 240, 360, 1440]
HIC_URL = FTP + "series/GSE201nnn/GSE201353/suppl/GSE201353_LIMA_THP1_WT_LPIF_{:04d}_S_0.0.0_megaMap_inter.hic"
LOOPS = FTP + "series/GSE201nnn/GSE201353/suppl/GSE201353_LIMA_HiC_diffLoops.tsv.gz"
RNA_LIST = FTP + "series/GSE201nnn/GSE201354/suppl/filelist.txt"
RNA_URL = FTP + "samples/{nnn}/{gsm}/suppl/{f}"
GENCODE37 = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/GRCh37_mapping/gencode.v46lift37.annotation.gtf.gz"
BIN, CHROMS = kb.BIN, kb.CHROMS
cbb.SRC["reed2022"] = ("Reed et al. 2022 Cell Reports: THP-1 LPS + IFN-gamma time course, Hi-C + RNA-seq (GEO GSE201376)",
                       "https://doi.org/10.1016/j.celrep.2022.111567", "no licence stated (local only)")
cbb.LOCAL_ONLY.add("reed2022")
cbb.SRC["gencode46_lift37"] = ("GENCODE release 46 mapped to GRCh37 (lift37 comprehensive GTF)", "https://www.gencodegenes.org/human/release_46lift37.html", "free use")


def rna_files():
    fl = os.path.join(RAW, "rna", "filelist.txt")
    if not os.path.exists(fl):
        xb.fetch(RNA_LIST, fl)
    out = []
    for line in open(fl).read().splitlines():
        f = line.split("\t")
        m = re.match(r"(GSM\d+)_LIMA_RNA_THP1_WT_LPIF_(\d{4})_S_(\d)\.1\.1(_quant\.sf\.gz|\.bw)$", f[1]) if len(f) > 1 and f[0] == "File" else None
        if m:
            out.append(dict(gsm=m.group(1), t=int(m.group(2)), rep=int(m.group(3)), f=f[1]))
    return out


def pull():
    rep = {}
    get = lambda url, path: "kept" if os.path.exists(path) else xb.fetch(url, path)
    rep["gencode37"] = get(GENCODE37, os.path.join(RAW, "gencode", "raw", "gencode.v46lift37.annotation.gtf.gz"))
    for d in rna_files():
        rep[d["f"]] = get(RNA_URL.format(nnn=d["gsm"][:-3] + "nnn", gsm=d["gsm"], f=d["f"]), os.path.join(RAW, "rna", "raw", d["f"])); print(d["f"], rep[d["f"]], flush=True)
    rep["loops"] = get(LOOPS, os.path.join(RAW, "hic", "raw", "GSE201353_LIMA_HiC_diffLoops.tsv.gz"))
    for t in TIMES:
        out = os.path.join(RAW, "hic", "raw", f"intra100k_{t}.npz")
        if os.path.exists(out):
            rep[t] = "kept"; continue
        h = hr.HicRemote(HIC_URL.format(t)); arr = {}
        for c in CHROMS:
            i, j, v = h.intra(c, BIN); arr[f"{c}_i"], arr[f"{c}_j"], arr[f"{c}_v"] = i, j, v.astype(np.float32)
        np.savez_compressed(out + ".part.npz", **arr); os.replace(out + ".part.npz", out)
        rep[t] = dict(version=h.version, url=HIC_URL.format(t), contacts=float(sum(arr[f"{c}_v"].sum() for c in CHROMS))); print("hic", t, rep[t], flush=True)
    json.dump(rep, open(os.path.join(RAW, "pull.json"), "w"), indent=1, default=str)


def gtf37():
    """GRCh37 positions: transcript -> gene, gene spans (with type), strand + name, exons (all genes)."""
    tx, span, info, ex = {}, {}, {}, collections.defaultdict(list)
    for line in gzip.open(os.path.join(RAW, "gencode", "raw", "gencode.v46lift37.annotation.gtf.gz"), "rt"):
        if line[0] == "#":
            continue
        f = line.split("\t", 8)
        if f[2] not in ("gene", "transcript", "exon") or f[0] == "chrY":
            continue
        g = re.search(r'gene_id "([^".]+)', f[8]).group(1); ch = f[0].replace("chr", "")
        if f[2] == "gene":
            span[g] = (ch, int(f[3]) - 1, int(f[4]), re.search(r'gene_type "([^"]+)', f[8]).group(1))
            info[g] = (f[6], re.search(r'gene_name "([^"]+)', f[8]).group(1))
        elif f[2] == "transcript":
            tx[re.search(r'transcript_id "([^".]+)', f[8]).group(1)] = g
        else:
            ex[ch].append((int(f[3]) - 1, int(f[4])))
    return tx, span, info, ex


def compartments(tss_pc):
    """as k3dbank.compartments, for 8 time points (0 h vector chosen by gene density, later ones by the 0 h vector)."""
    H = {t: np.load(os.path.join(RAW, "hic", "raw", f"intra100k_{t}.npz")) for t in TIMES}
    E1, info, chrom_of, start_of = [], {}, [], []
    for c in CHROMS:
        n = int(max(H[t][f"{c}_j"].max() for t in TIMES)) + 1
        dens = np.zeros(n)
        for p in tss_pc.get(c, []):
            if p // BIN < n:
                dens[p // BIN] += 1
        vec, rec = {}, {}
        for t in TIMES:
            V = kb.eig3(H[t][f"{c}_i"], H[t][f"{c}_j"], H[t][f"{c}_v"].astype(np.float64), n)
            ref = dens if t == 0 else vec[0]
            rs = [np.corrcoef(V[:, k][~np.isnan(V[:, k]) & ~np.isnan(ref)], ref[~np.isnan(V[:, k]) & ~np.isnan(ref)])[0, 1] for k in range(3)]
            k = int(np.argmax(np.abs(rs))); vec[t] = kb.zs(V[:, k] * np.sign(rs[k])); rec[t] = dict(pc=k + 1, r=float(abs(rs[k])))
        info[c] = dict(bins=n, density_r_0h=rec[0]["r"], pc_0h=rec[0]["pc"], **{f"r_vs_0h_{t}": rec[t]["r"] for t in TIMES[1:]}, **{f"pc_{t}": rec[t]["pc"] for t in TIMES[1:]})
        E1.append(np.stack([vec[t] for t in TIMES], 1)); chrom_of += [c] * n; start_of += list(range(0, n * BIN, BIN))
        print("compartments", c, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info[c].items()}), flush=True)
    return np.concatenate(E1), np.array(chrom_of), np.array(start_of), info


def build():
    import pyBigWig
    tx, span, ginfo, ex = gtf37()
    # rows as in gencode46_tss.json: (gene, symbol, type, chrom, 1-based TSS)
    tss = sorted([(g, ginfo[g][1], ty, "chr" + c, (s + 1) if ginfo[g][0] == "+" else e) for g, (c, s, e, ty) in span.items() if ty == "protein_coding" and c in CHROMS],
                 key=lambda r: (CHROMS.index(r[3][3:]), r[4]))
    genes = [g[0] for g in tss]; gi = {g: k for k, g in enumerate(genes)}
    files = sorted(rna_files(), key=lambda d: (d["rep"], d["t"]))
    quant = [d for d in files if d["f"].endswith("quant.sf.gz")]; bws = [d for d in files if d["f"].endswith(".bw")]
    mid = os.path.join(RAW, "build_intermediate.npz")
    if os.path.exists(mid) and list(np.load(mid)["gene"]) == genes:
        C = np.load(mid); TPM, INT, E1, bchr, bstart, mapped = C["tpm"], C["intron"], C["e1"], C["bin_chrom"], C["bin_start"], list(C["mapped"])
        cinfo = json.load(open(mid + ".json")); intr = kb.clean_introns(genes, span, ex)
    else:
        TPM = np.zeros((len(genes), len(quant)), np.float32); mapped = []
        for k, d in enumerate(quant):
            tot = got = 0.0
            for line in gzip.open(os.path.join(RAW, "rna", "raw", d["f"]), "rt"):
                f = line.split("\t")
                if f[0] == "Name":
                    continue
                t = float(f[3]); tot += t; g = tx.get(f[0].split(".")[0].split("|")[0])
                if g in gi:
                    TPM[gi[g], k] += t; got += t
            mapped.append(got / tot); print("tpm", d["f"], round(got / tot, 3), flush=True)
        intr = kb.clean_introns(genes, span, ex); INT = np.full((len(genes), len(bws)), np.nan, np.float32)
        for k, d in enumerate(bws):
            bw = pyBigWig.open(os.path.join(RAW, "rna", "raw", d["f"])); ch = bw.chroms()
            for g, pcs in intr.items():
                c = "chr" + span[g][0]
                if c not in ch:
                    continue
                s = l = 0.0
                for a, b in pcs:
                    v = bw.stats(c, a, min(b, ch[c]), type="mean", exact=True)[0]; s += (v or 0.0) * (b - a); l += b - a
                INT[gi[g], k] = s / l
            bw.close(); print("introns", d["f"], flush=True)
        E1, bchr, bstart, cinfo = compartments({c: [g[4] - 1 for g in tss if g[3].replace("chr", "") == c] for c in CHROMS})
        np.savez_compressed(mid, gene=np.array(genes), tpm=TPM, intron=INT, e1=E1, bin_chrom=bchr, bin_start=bstart, mapped=np.array(mapped))
        json.dump(cinfo, open(mid + ".json", "w"))
    L = [line.rstrip("\n").split("\t") for line in gzip.open(os.path.join(RAW, "hic", "raw", "GSE201353_LIMA_HiC_diffLoops.tsv.gz"), "rt")]
    head = L[0]; L = L[1:]; col = {h: k for k, h in enumerate(head)}
    raw = [h for h in head if h.endswith("_RAW")]
    lab = np.array([r[col["cluster"]] for r in L])
    cl = np.where(np.char.startswith(lab, "up"), "gained", np.where(np.char.startswith(lab, "down"), "lost", "static"))
    lfc_cols = ["m0v30_LFC", "m0v60_LFC", "m0v90_LFC", "m0v120_LFC", "m0v240_LFC", "m0v360_LFC", "m0v1440_LFC"]
    np.savez_compressed(os.path.join(cbb.LAY, "thp1_lpif_3d.npz"),
                        gene=np.array(genes), symbol=np.array([g[1] for g in tss]), chrom=np.array([g[3].replace("chr", "") for g in tss]),
                        tss=np.array([g[4] - 1 for g in tss]), span=np.array([[span[g][1], span[g][2]] for g in genes]),
                        rna_cols=np.array([f"{d['t']}_{d['rep']}" for d in quant]), tpm=TPM, int_cols=np.array([f"{d['t']}_{d['rep']}" for d in bws]), intron=INT,
                        intron_bp=np.array([sum(b - a for a, b in intr.get(g, [])) for g in genes]),
                        bin_chrom=bchr, bin_start=bstart, e1=E1.astype(np.float32), e1_cols=np.array([str(t) for t in TIMES]),
                        loop_anchor=np.array([[r[col["chromosome1"]].replace("chr", ""), r[col["x1"]], r[col["x2"]], r[col["chromosome2"]].replace("chr", ""),
                                               r[col["y1"]], r[col["y2"]]] for r in L]),
                        loop_lfc=np.array([[0.0] + [kb.fnum(r[col[c]]) for c in lfc_cols] for r in L]), loop_lfc_cols=np.array([str(t) for t in TIMES]),
                        loop_padj=np.array([kb.fnum(r[col["adjusted_pval"]]) for r in L]), loop_peak_time=np.array([kb.fnum(r[col["max_abs_LFC_TP"]]) for r in L]),
                        loop_cluster=cl, loop_label=lab, loop_libs=np.array(raw),
                        loop_counts=np.array([[kb.fnum(r[col[h]]) for h in raw] for r in L], np.float32))
    info = dict(genes=len(genes), tpm_share_mapped_to_protein_coding=[round(x, 3) for x in mapped], genes_with_clean_introns=len(intr),
                bins=int(len(bchr)), bins_valid_all_times=int((~np.isnan(E1).any(1)).sum()), compartments=cinfo, loops=len(L),
                loop_labels=dict(collections.Counter(lab.tolist())))
    json.dump(info, open(os.path.join(RAW, "build_info.json"), "w"), indent=1); print(json.dumps({k: v for k, v in info.items() if k != "compartments"}))


def bank():
    info = json.load(open(os.path.join(RAW, "build_info.json")))
    raws = [cbb.layer_fact(f"THP1LPIF_RAW_{k.upper()}", 1, src, os.path.join(RAW, k, "raw"), "downloaded files", "-", "as published")
            for k, src in (("rna", "reed2022"), ("hic", "reed2022"), ("gencode", "gencode46_lift37"))]
    lay = cbb.layer_fact("THP1_LPIF_3D", 1, "reed2022", os.path.join(cbb.LAY, "thp1_lpif_3d.npz"),
                         f"{info['genes']} protein-coding genes; {info['bins']} 100-kb bins; {info['loops']} loops",
                         "RNA TPM + intron signal (8 times x 2 reps); compartment E1 (8 times); loop counts (8 times x 5 replicate columns)",
                         "TPM; mean coverage; E1 z per chromosome; raw counts", inputs=tuple(f.ref for f in raws), tier="derived")
    n = cbb.bank().add_many(raws + [lay])
    rep = dict(added=time.strftime("%Y-%m-%d"), facts_written=n, layers={f.entity: f.ref for f in raws + [lay]}, build=info)
    json.dump(rep, open(R("outputs", "cellbrain", "thp1lpif_report.json"), "w"), indent=1, default=int); print(json.dumps(rep, default=int)[:800])


if __name__ == "__main__":
    {"pull": pull, "build": build, "bank": bank}[sys.argv[1]]()
