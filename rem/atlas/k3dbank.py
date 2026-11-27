"""K3DBANK -- K562 time course with the 3D chromosome map AND gene expression at matching time points, added to the memory
bank (cellbrain) for G16 (k3dtest.py): does a change in the chromosome's 3D shape go with a change in the genes there?

Asked by the user: "build G16 with K562 first" (after: "look at the chromosome and its 3d map at that time, and compare it
with that time's gene expression so we can link how change in chromosome links to gene expression").
Source (Bond et al. 2023 Genome Research 33:1258, doi 10.1101/gr.277397.122; GEO SuperSeries GSE213909; no licence
stated -> local only): K562 cells pushed toward megakaryocyte with PMA.
  Hi-C (GSE214123): merged in situ Hi-C maps (.hic, Juicer v8, hg38) at 0, 6 h and 72 h, read over HTTPS with byte ranges
    (hicremote.py; nothing stored but the 100 kb intra-chromosomal counts); the authors' loop table
    (GSE214123_hicSummary.txt.gz: 33,914 loops at 10 kb, raw counts in 30 libraries = 3 time points x 4 biological
    replicates with technical repeats, DESeq2 log2 fold change at 6 h and 72 h, adjusted p, gained / lost / static).
  RNA-seq (GSE213386): Salmon transcript quantities and coverage bigWigs per replicate (2) at 0, 30 min, 90 min, 3 h, 6 h,
    24 h, 48 h, 72 h.
  GENCODE release 46 comprehensive annotation (EMBL-EBI; free use): transcript -> gene, exons, gene spans (introns).
Built here (all per time point):
  expression: gene TPM = sum of its transcripts' TPM (protein-coding genes, chr1-22 + X);
  transcription now (intron signal): mean bigWig coverage over the gene's 'clean' introns (gene span minus every exon
    of any gene and minus other genes' spans, >= 1 kb in total); intron RNA is short-lived, so it tracks transcription
    with less delay than mature mRNA;
  compartments (open 'A' vs closed 'B' neighbourhood) per 100 kb bin: raw counts -> bins with < 20% of the median
    coverage dropped -> iterative correction -> observed / expected by distance (first 2 diagonals set to 1) ->
    Pearson correlation -> top 3 eigenvectors; at 0 h the one that best follows gene density (protein-coding TSS per
    bin) is kept and turned so A (gene-rich) is positive; at 6 h and 72 h the one that best follows the 0 h vector;
    each scaled to mean 0, SD 1 per chromosome ('E1');
  loops: the authors' table as published.
Layer: K562_PMA_3D (npz). Raw downloads stay in _cache/k562pma (local only).

Run: python3 -I rem/atlas/k3dbank.py pull | build | bank      (committed before it ran)
"""

from __future__ import annotations
import collections
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
_s = importlib.util.spec_from_file_location("xcellbank", os.path.join(HERE, "xcellbank.py"))
xb = importlib.util.module_from_spec(_s); _s.loader.exec_module(xb)
_h = importlib.util.spec_from_file_location("hicremote", os.path.join(HERE, "hicremote.py"))
hr = importlib.util.module_from_spec(_h); _h.loader.exec_module(hr)
cbb, Fact = xb.cbb, xb.Fact

RAW = os.path.join(HERE, "_cache", "k562pma")
AG = os.path.join(HERE, "_cache", "ag2bank")
FTP = "https://ftp.ncbi.nlm.nih.gov/geo/"
HIC = {0: "GSE214123_MEGA_K562_WT_0_inter.hic", 360: "GSE214123_MEGA_K562_WT_360_inter.hic", 4320: "GSE214123_MEGA_K562_WT_4320_inter.hic"}
HIC_URL = FTP + "series/GSE214nnn/GSE214123/suppl/{}"
SUMMARY = HIC_URL.format("GSE214123_hicSummary.txt.gz")
RNA_LIST = FTP + "series/GSE213nnn/GSE213386/suppl/filelist.txt"
RNA_URL = FTP + "samples/{nnn}/{gsm}/suppl/{f}"
GENCODE = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.annotation.gtf.gz"
RNA_T = [0, 30, 90, 180, 360, 1440, 2880, 4320]
HIC_T = [0, 360, 4320]
BIN = 100_000
CHROMS = [str(c) for c in range(1, 23)] + ["X"]
cbb.SRC["bond2023"] = ("Bond et al. 2023 Genome Research: K562 PMA time course, Hi-C + RNA-seq (GEO GSE213909)", "https://doi.org/10.1101/gr.277397.122",
                       "no licence stated (local only)")
cbb.SRC["gencode46_full"] = ("GENCODE release 46 comprehensive annotation (GTF)", "https://www.gencodegenes.org/human/release_46.html", "free use")
cbb.LOCAL_ONLY.add("bond2023")


def rna_files():
    fl = os.path.join(RAW, "rna", "filelist.txt")
    if not os.path.exists(fl):
        xb.fetch(RNA_LIST, fl)
    txt = open(fl).read(); out = []
    for line in txt.splitlines():
        f = line.split("\t")
        if len(f) > 1 and f[0] == "File" and re.match(r"GSM\d+_MEGA_RNA_K562_WT_PMA_\d{4}_S_\d\.1\.1\.(quant\.sf\.gz|bw)$", f[1]):
            m = re.match(r"(GSM\d+)_MEGA_RNA_K562_WT_PMA_(\d{4})_S_(\d)", f[1])
            out.append(dict(gsm=m.group(1), t=int(m.group(2)), rep=int(m.group(3)), f=f[1]))
    return out


def pull():
    rep = {}
    for d in rna_files():
        url = RNA_URL.format(nnn=d["gsm"][:-3] + "nnn", gsm=d["gsm"], f=d["f"])
        rep[d["f"]] = xb.fetch(url, os.path.join(RAW, "rna", "raw", d["f"])); print(d["f"], rep[d["f"]], flush=True)
    rep["hicSummary"] = xb.fetch(SUMMARY, os.path.join(RAW, "hic", "raw", "GSE214123_hicSummary.txt.gz"))
    rep["gencode"] = xb.fetch(GENCODE, os.path.join(RAW, "gencode", "raw", "gencode.v46.annotation.gtf.gz"))
    for t, f in HIC.items():                      # 100 kb intra-chromosomal raw counts, read remotely
        out = os.path.join(RAW, "hic", "raw", f"intra100k_{t}.npz")
        if os.path.exists(out):
            rep[f] = "kept"; continue
        h = hr.HicRemote(HIC_URL.format(f)); arr = {}
        for c in CHROMS:
            i, j, v = h.intra(c, BIN); arr[f"{c}_i"], arr[f"{c}_j"], arr[f"{c}_v"] = i, j, v.astype(np.float32)
            print(t, c, len(v), int(v.sum()), flush=True)
        np.savez_compressed(out + ".part.npz", **arr); os.replace(out + ".part.npz", out)
        rep[f] = dict(version=h.version, url=HIC_URL.format(f), chroms=len(CHROMS), contacts=float(sum(arr[f"{c}_v"].sum() for c in CHROMS)))
    json.dump(rep, open(os.path.join(RAW, "pull.json"), "w"), indent=1, default=str)


# ---------------------------------------------------------------- annotation
def gtf():
    """transcript -> gene, gene spans, exon intervals (all genes), from GENCODE 46 comprehensive."""
    tx, span, ex = {}, {}, collections.defaultdict(list)
    for line in gzip.open(os.path.join(RAW, "gencode", "raw", "gencode.v46.annotation.gtf.gz"), "rt"):
        if line[0] == "#":
            continue
        f = line.split("\t", 8)
        if f[2] not in ("gene", "transcript", "exon"):
            continue
        if f[0] == "chrY":                         # PAR copies would overwrite the chrX records
            continue
        g = re.search(r'gene_id "([^".]+)', f[8]).group(1); ch = f[0].replace("chr", "")
        if f[2] == "gene":
            span[g] = (ch, int(f[3]) - 1, int(f[4]), re.search(r'gene_type "([^"]+)', f[8]).group(1))
        elif f[2] == "transcript":
            tx[re.search(r'transcript_id "([^".]+)', f[8]).group(1)] = g
        else:
            ex[ch].append((int(f[3]) - 1, int(f[4])))
    return tx, span, ex


def merge(iv):
    iv = sorted(iv); out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def subtract(a, b, cuts):
    """[a,b) minus sorted merged intervals cuts -> list of pieces."""
    out, cur = [], a
    lo = np.searchsorted(cuts[:, 1], a, side="right") if len(cuts) else 0
    for k in range(lo, len(cuts)):
        s, e = cuts[k]
        if s >= b:
            break
        if s > cur:
            out.append((cur, min(s, b)))
        cur = max(cur, e)
        if cur >= b:
            break
    if cur < b:
        out.append((cur, b))
    return out


def clean_introns(genes, span, ex):
    """per gene: its span minus all exons (any gene, any strand) minus other genes' spans; pieces >= 200 bp."""
    exm = {c: np.array(merge(v), np.int64).reshape(-1, 2) for c, v in ex.items()}
    none = np.zeros((0, 2), np.int64)
    by = collections.defaultdict(list)
    for g, (c, s, e, ty) in span.items():
        if ty in ("protein_coding", "lncRNA"):
            by[c].append((s, e, g))
    for c in by:
        by[c].sort()
    out = {}
    for g in genes:
        c, s, e, _ = span[g]
        others = [(a, b) for a, b, h in by[c] if h != g and a < e and b > s]
        X = exm.get(c, none); lo, hi = np.searchsorted(X[:, 1], s, side="right"), np.searchsorted(X[:, 0], e, side="left")
        cuts = np.array(merge([tuple(x) for x in X[lo:hi]] + others), np.int64).reshape(-1, 2)
        pcs = [(a, b) for a, b in subtract(s, e, cuts) if b - a >= 200]
        if sum(b - a for a, b in pcs) >= 1000:
            out[g] = pcs
    return out


# ---------------------------------------------------------------- compartments
def ice(M, iters=300, tol=1e-6):
    W = M.copy()
    for _ in range(iters):
        s = W.sum(1); m = s[s > 0].mean(); s = s / m; s[s == 0] = 1
        W /= s[:, None]; W /= s[None, :]
        if np.abs(s - 1).max() < tol:
            break
    return W


def eig3(i, j, v, n):
    M = np.zeros((n, n)); np.add.at(M, (i, j), v); M = M + M.T - np.diag(np.diag(M))
    cov = M.sum(1); ok = cov >= 0.2 * np.median(cov[cov > 0])
    idx = np.nonzero(ok)[0]; Wf = np.full((n, n), np.nan); Wf[np.ix_(ok, ok)] = ice(M[np.ix_(ok, ok)])
    OE = np.full((n, n), np.nan); r = np.arange(n)
    for k in range(n):                             # observed / expected along each diagonal (genomic distance k bins)
        dg = Wf[r[:n - k], r[:n - k] + k]; mu = np.nanmean(dg) if np.isfinite(dg).any() else np.nan
        val = dg / mu if mu and mu > 0 else np.zeros_like(dg)
        OE[r[:n - k], r[:n - k] + k] = val; OE[r[:n - k] + k, r[:n - k]] = val
    OE = OE[np.ix_(ok, ok)]; d = np.abs(idx[:, None] - idx[None, :]); OE[d < 2] = 1.0; OE = np.nan_to_num(OE)
    C = np.corrcoef(OE); C = np.nan_to_num(C)
    w, V = np.linalg.eigh(C); V = V[:, np.argsort(w)[::-1][:3]]
    out = np.full((n, 3), np.nan); out[idx] = V
    return out


def zs(x):
    m = ~np.isnan(x); y = np.full_like(x, np.nan); y[m] = (x[m] - x[m].mean()) / x[m].std(); return y


def compartments(tss_pc):
    H = {t: np.load(os.path.join(RAW, "hic", "raw", f"intra100k_{t}.npz")) for t in HIC_T}
    E1, info, chrom_of, start_of = [], {}, [], []
    for c in CHROMS:
        n = int(max(H[t][f"{c}_j"].max() for t in HIC_T)) + 1
        dens = np.zeros(n)
        for p in tss_pc.get(c, []):
            if p // BIN < n:
                dens[p // BIN] += 1
        vec, rec = {}, {}
        for t in HIC_T:
            V = eig3(H[t][f"{c}_i"], H[t][f"{c}_j"], H[t][f"{c}_v"].astype(np.float64), n)
            ref = dens if t == 0 else vec[0]
            rs = [np.corrcoef(V[:, k][~np.isnan(V[:, k]) & ~np.isnan(ref)], ref[~np.isnan(V[:, k]) & ~np.isnan(ref)])[0, 1] for k in range(3)]
            k = int(np.argmax(np.abs(rs))); vec[t] = zs(V[:, k] * np.sign(rs[k])); rec[t] = dict(pc=k + 1, r=float(abs(rs[k])))
        info[c] = dict(bins=n, density_r_0h=rec[0]["r"], pc_0h=rec[0]["pc"], **{f"r_vs_0h_{t}": rec[t]["r"] for t in HIC_T[1:]},
                       **{f"pc_{t}": rec[t]["pc"] for t in HIC_T[1:]})
        E1.append(np.stack([vec[t] for t in HIC_T], 1)); chrom_of += [c] * n; start_of += list(range(0, n * BIN, BIN))
        print("compartments", c, info[c], flush=True)
    return np.concatenate(E1), np.array(chrom_of), np.array(start_of), info


# ---------------------------------------------------------------- build
def fnum(x):
    return float("nan") if x in ("NA", "", "NaN") else float(x)


def build():
    import pyBigWig
    tx, span, ex = gtf()
    mid = os.path.join(RAW, "build_intermediate.npz")      # TPM, intron signal and E1 are slow: kept if already made
    tss = [g for g in json.load(open(os.path.join(AG, "raw", "gencode46_tss.json"))) if g[2] == "protein_coding" and g[3].replace("chr", "") in CHROMS]
    tss = [g for g in tss if g[0] in span]
    genes = [g[0] for g in tss]
    gi = {g: k for k, g in enumerate(genes)}
    files = sorted(rna_files(), key=lambda d: (d["rep"], d["t"]))
    quant = [d for d in files if d["f"].endswith("quant.sf.gz")]; bws = [d for d in files if d["f"].endswith(".bw")]
    TPM = np.zeros((len(genes), len(quant)), np.float32); mapped = []
    cached = np.load(mid, allow_pickle=False) if os.path.exists(mid) else None
    if cached is not None and list(cached["gene"]) == genes:
        TPM, INT, E1, bchr, bstart = cached["tpm"], cached["intron"], cached["e1"], cached["bin_chrom"], cached["bin_start"]
        mapped = list(cached["mapped"]); intr = clean_introns(genes, span, ex); cinfo = json.load(open(mid + ".json"))
    for k, d in enumerate(quant if cached is None else []):
        tot = got = 0.0
        for line in gzip.open(os.path.join(RAW, "rna", "raw", d["f"]), "rt"):
            f = line.split("\t")
            if f[0] == "Name":
                continue
            t = float(f[3]); tot += t; g = tx.get(f[0].split(".")[0])
            if g in gi:
                TPM[gi[g], k] += t; got += t
        mapped.append(got / tot); print("tpm", d["f"], round(got / tot, 3), flush=True)
    if cached is None:
        intr = clean_introns(genes, span, ex)
        INT = np.full((len(genes), len(bws)), np.nan, np.float32)
    for k, d in enumerate(bws if cached is None else []):
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
    if cached is None:
        E1, bchr, bstart, cinfo = compartments({c: [g[4] - 1 for g in tss if g[3].replace("chr", "") == c] for c in CHROMS})
        np.savez_compressed(mid, gene=np.array(genes), tpm=TPM, intron=INT, e1=E1, bin_chrom=bchr, bin_start=bstart, mapped=np.array(mapped))
        json.dump(cinfo, open(mid + ".json", "w"))
    L = [line.rstrip("\n").split("\t") for line in gzip.open(os.path.join(RAW, "hic", "raw", "GSE214123_hicSummary.txt.gz"), "rt")]
    head = L[0]; L = L[1:]; libs = [h for h in head if h.startswith("MEGA_K562")]
    col = {h: k for k, h in enumerate(head)}
    np.savez_compressed(os.path.join(cbb.LAY, "k562_pma_3d.npz"),
                        gene=np.array(genes), symbol=np.array([g[1] for g in tss]), chrom=np.array([g[3].replace("chr", "") for g in tss]),
                        tss=np.array([g[4] - 1 for g in tss]), span=np.array([[span[g][1], span[g][2]] for g in genes]),
                        rna_cols=np.array([f"{d['t']}_{d['rep']}" for d in quant]), tpm=TPM, int_cols=np.array([f"{d['t']}_{d['rep']}" for d in bws]), intron=INT,
                        intron_bp=np.array([sum(b - a for a, b in intr.get(g, [])) for g in genes]),
                        bin_chrom=bchr, bin_start=bstart, e1=E1.astype(np.float32), e1_cols=np.array([str(t) for t in HIC_T]),
                        loop_anchor=np.array([[L_[col["seqnames1"]].replace("chr", ""), L_[col["start1"]], L_[col["end1"]], L_[col["seqnames2"]].replace("chr", ""),
                                               L_[col["start2"]], L_[col["end2"]]] for L_ in L]),
                        loop_lfc=np.array([[fnum(L_[col["log2FoldChange_360"]]), fnum(L_[col["log2FoldChange_4320"]]), fnum(L_[col["padj_360"]]),
                                            fnum(L_[col["padj_4320"]])] for L_ in L]),
                        loop_cluster=np.array([L_[col["cluster"]] for L_ in L]), loop_libs=np.array(libs),
                        loop_counts=np.array([[fnum(L_[col[h]]) for h in libs] for L_ in L], np.float32))
    info = dict(genes=len(genes), tpm_share_mapped_to_protein_coding=[round(x, 3) for x in mapped], genes_with_clean_introns=len(intr),
                bins=int(len(bchr)), bins_valid_all_times=int((~np.isnan(E1).any(1)).sum()), compartments=cinfo, loops=len(L),
                loop_clusters=dict(collections.Counter(L_[col["cluster"]] for L_ in L)))
    json.dump(info, open(os.path.join(RAW, "build_info.json"), "w"), indent=1); print(json.dumps({k: v for k, v in info.items() if k != "compartments"}))


def bank():
    info = json.load(open(os.path.join(RAW, "build_info.json")))
    raws = [cbb.layer_fact(f"K562PMA_RAW_{k.upper()}", 1, src, os.path.join(RAW, k, "raw"), "downloaded files", "-", "as published")
            for k, src in (("rna", "bond2023"), ("hic", "bond2023"), ("gencode", "gencode46_full"))]
    lay = cbb.layer_fact("K562_PMA_3D", 1, "bond2023", os.path.join(cbb.LAY, "k562_pma_3d.npz"),
                         f"{info['genes']} protein-coding genes; {info['bins']} 100-kb bins; {info['loops']} loops",
                         "RNA TPM + intron signal (8 times x 2 reps); compartment E1 (0, 6, 72 h); loop counts (30 libraries)",
                         "TPM; mean coverage; E1 z per chromosome; raw counts", inputs=tuple(f.ref for f in raws), tier="derived")
    n = cbb.bank().add_many(raws + [lay])
    rep = dict(added=time.strftime("%Y-%m-%d"), facts_written=n, layers={f.entity: f.ref for f in raws + [lay]}, build=info)
    json.dump(rep, open(R("outputs", "cellbrain", "k562pma_report.json"), "w"), indent=1, default=int); print(json.dumps(rep, default=int)[:800])


if __name__ == "__main__":
    {"pull": pull, "build": build, "bank": bank}[sys.argv[1]]()
