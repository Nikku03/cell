"""A3DBANK -- A549 lung cells treated with the steroid dexamethasone, with the 3D chromosome map AND gene expression at the
same 5 time points (0, 1, 4, 8, 12 h), added to the memory bank (cellbrain) for G18 (a3dtest.py), the third cell type
after G16 (K562) and G17 (THP-1).

Asked by the user: "yes, run it on A549".
Source: ENCODE (Reddy lab, Duke; D'Ippolito et al. 2018 Cell Systems doi 10.1016/j.cels.2018.06.007 and McDowell et al.
2018 Genome Research; ENCODE data: free use), GRCh38:
  in situ Hi-C: ENCSR662QKG (0 h), ENCSR244BBG (1 h), ENCSR393LOP (4 h), ENCSR757IVO (8 h), ENCSR499RVD (12 h); one .hic
    per biological replicate (4 per time point; Juicer v8), read over HTTPS with byte ranges (hicremote.py); ENCODE's
    HiCCUPS loop list per time point (tsv).
  polyA RNA-seq: ENCSR632DQP (0 h), ENCSR656FIH (1 h), ENCSR326PTW (4 h), ENCSR546PPG (8 h), ENCSR154TDP (12 h); per
    biological replicate (4): gene quantifications (RSEM, GENCODE V29) and plus / minus strand unique-read signal
    (bigWig; each downloaded, read and deleted).
  GENCODE release 46 comprehensive annotation (shared with k3dbank.py).
Built as k3dbank.py (same functions) except where ENCODE's data differ:
  expression: gene TPM straight from ENCODE's gene quantifications (Ensembl id, version dropped);
  intron signal: mean of (plus + minus) coverage over the gene's clean introns (strands summed, as the unstranded K562 and
    THP-1 tracks);
  compartments: the 4 replicate maps summed per time point, then exactly as k3dbank (E1 per 100 kb bin);
  loops: ENCODE's HiCCUPS loops from all 5 time points joined (loops whose centroids are within 20 kb on both sides are
    one loop, the one with most observed contacts kept); per loop and per replicate map: raw counts summed over the
    3 x 3 pixels at 10 kb around the centroid; per library CPM over all loops, l = log2(CPM + 1); change 0 h -> each time
    = mean l difference; moderated Welch test, 12 h vs 0 h, 4 vs 4 replicates (each loop's variance shrunk toward the
    variance of loops with similar mean, 4 prior degrees of freedom); Benjamini-Hochberg q. Called here because ENCODE
    provides no differential loops; the gene outcomes are not used.
Layer: A549_DEX_3D (npz). Raw downloads stay in _cache/a549dex.

Run: python3 -I rem/atlas/a3dbank.py pull | build | bank      (committed before it ran)
"""

from __future__ import annotations
import collections
import gzip
import importlib.util
import json
import os
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

RAW = os.path.join(HERE, "_cache", "a549dex")
ENC = "https://www.encodeproject.org"
TIMES = [0, 60, 240, 480, 720]
HIC_EXP = {0: "ENCSR662QKG", 60: "ENCSR244BBG", 240: "ENCSR393LOP", 480: "ENCSR757IVO", 720: "ENCSR499RVD"}
RNA_EXP = {0: "ENCSR632DQP", 60: "ENCSR656FIH", 240: "ENCSR326PTW", 480: "ENCSR546PPG", 720: "ENCSR154TDP"}
REPS = (1, 2, 3, 4)
BIN, CHROMS = kb.BIN, kb.CHROMS
LOOP_BIN = 10_000
cbb.SRC["encode_a549_dex"] = ("ENCODE A549 dexamethasone time course (Reddy lab): in situ Hi-C + polyA RNA-seq", "https://www.encodeproject.org", "ENCODE: free use")


def meta(acc):
    p = os.path.join(RAW, "meta", acc + ".json")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        b = urllib.request.urlopen(urllib.request.Request(f"{ENC}/experiments/{acc}/?format=json&frame=embedded", headers={"User-Agent": "cellbrain/1.0", "Accept": "application/json"}), timeout=300).read()
        open(p, "wb").write(b)
    return json.load(open(p))


def files():
    """the exact files used: {kind: [(time, rep, accession, url)]}."""
    out = collections.defaultdict(list)
    for t, acc in HIC_EXP.items():
        for f in meta(acc)["files"]:
            if f.get("status") != "released" or f.get("assembly") != "GRCh38":
                continue
            if f["output_type"] == "mapping quality thresholded contact matrix" and f["file_format"] == "hic":
                out["hic"].append((t, f["biological_replicates"][0], f["accession"], ENC + f["href"]))
            elif f["output_type"] == "loops":
                out["loops"].append((t, 0, f["accession"], ENC + f["href"]))
    kinds = {"gene quantifications": "genes", "plus strand signal of unique reads": "plus", "minus strand signal of unique reads": "minus"}
    for t, acc in RNA_EXP.items():
        for f in meta(acc)["files"]:
            if f.get("status") == "released" and f.get("assembly") == "GRCh38" and f.get("genome_annotation") == "V29" and f["output_type"] in kinds:
                out[kinds[f["output_type"]]].append((t, f["biological_replicates"][0], f["accession"], ENC + f["href"]))
    return {k: sorted(v) for k, v in out.items()}


def union_loops():
    rows = []
    for t, _, acc, _ in files()["loops"]:
        L = open(os.path.join(RAW, "hic", "raw", acc + ".tsv")).read().splitlines()
        h = L[0].split("\t"); c = {k: i for i, k in enumerate(h)}
        for line in L[1:]:
            f = line.split("\t")
            if f[0] != f[3] or f[0].replace("chr", "") not in CHROMS:
                continue
            a, b = float(f[c["centroid1"]]), float(f[c["centroid2"]])
            rows.append((f[0].replace("chr", ""), min(a, b), max(a, b), float(f[c["observed"]]), t))
    rows.sort(key=lambda r: -r[3]); kept = collections.defaultdict(list); out = []
    for ch, a, b, obs, t in rows:                          # greedy join: centroids within 20 kb on both sides
        key = (ch, int(a // 100000))
        if any(abs(a - x) <= 20000 and abs(b - y) <= 20000 for k2 in (key, (ch, key[1] - 1), (ch, key[1] + 1)) for x, y in kept.get(k2, [])):
            continue
        kept[key].append((a, b)); out.append((ch, a, b, obs, t))
    return sorted(out, key=lambda r: (CHROMS.index(r[0]), r[1], r[2]))


def pull():
    F = files(); rep = {k: len(v) for k, v in F.items()}
    get = lambda url, path: "kept" if os.path.exists(path) else xb.fetch(url, path)
    for t, r, acc, url in F["loops"]:
        get(url, os.path.join(RAW, "hic", "raw", acc + ".tsv"))
    for t, r, acc, url in F["genes"]:
        get(url, os.path.join(RAW, "rna", "raw", acc + ".tsv"))
    U = union_loops(); rep["union_loops"] = len(U)
    pairs = collections.defaultdict(list)
    for k, (ch, a, b, obs, t) in enumerate(U):
        pairs[ch].append((k, int(a // LOOP_BIN), int(b // LOOP_BIN)))
    for t, r, acc, url in F["hic"]:
        out = os.path.join(RAW, "hic", "raw", f"rep_{t}_{r}.npz")
        if os.path.exists(out):
            continue
        h = hr.HicRemote(url); arr = {}; cnt = np.zeros(len(U))
        for c in CHROMS:
            i, j, v = h.intra("chr" + c, BIN); arr[f"{c}_i"], arr[f"{c}_j"], arr[f"{c}_v"] = i, j, v.astype(np.float32)
            if pairs[c]:
                ks, ii, jj = zip(*pairs[c]); cnt[list(ks)] = h.pixels("chr" + c, LOOP_BIN, list(zip(ii, jj)), pad=1)
        np.savez_compressed(out + ".part.npz", loops=cnt, **arr); os.replace(out + ".part.npz", out)
        print("hic", t, r, acc, int(sum(arr[f"{c}_v"].sum() for c in CHROMS)), "loop counts", int(cnt.sum()), flush=True)
    json.dump(rep, open(os.path.join(RAW, "pull.json"), "w"), indent=1)


def moderated_test(A, B):
    """Welch-type test with each row's variance shrunk toward the variance trend (20 bins of mean), prior df 4."""
    from scipy.stats import t as tdist
    ma, mb = A.mean(1), B.mean(1); va, vb = A.var(1, ddof=1), B.var(1, ddof=1); m = (ma + mb) / 2
    q = np.quantile(m, np.linspace(0, 1, 21)); k = np.clip(np.searchsorted(q, m) - 1, 0, 19)
    pa = np.array([np.median(va[k == i]) for i in range(20)])[k]; pb = np.array([np.median(vb[k == i]) for i in range(20)])[k]
    d0 = 4.0; na, nb = A.shape[1], B.shape[1]
    sa = (d0 * pa + (na - 1) * va) / (d0 + na - 1); sb = (d0 * pb + (nb - 1) * vb) / (d0 + nb - 1)
    se = np.sqrt(sa / na + sb / nb); tt = (mb - ma) / se; df = d0 + na + nb - 2
    return mb - ma, 2 * tdist.sf(np.abs(tt), df)


def build():
    import pyBigWig
    F = files(); tx, span, ex = kb.gtf()
    tss = [g for g in json.load(open(os.path.join(kb.AG, "raw", "gencode46_tss.json"))) if g[2] == "protein_coding" and g[3].replace("chr", "") in CHROMS]
    tss = [g for g in tss if g[0] in span]; genes = [g[0] for g in tss]; gi = {g: k for k, g in enumerate(genes)}
    cols = [(t, r) for t in TIMES for r in REPS]; ci = {c: k for k, c in enumerate(cols)}
    mid = os.path.join(RAW, "build_intermediate.npz")
    if os.path.exists(mid) and list(np.load(mid)["gene"]) == genes:
        C = np.load(mid); TPM, INT, E1, bchr, bstart = C["tpm"], C["intron"], C["e1"], C["bin_chrom"], C["bin_start"]
        cinfo = json.load(open(mid + ".json")); intr = kb.clean_introns(genes, span, ex)
    else:
        TPM = np.zeros((len(genes), len(cols)), np.float32)
        for t, r, acc, url in F["genes"]:
            L = open(os.path.join(RAW, "rna", "raw", acc + ".tsv")).read().splitlines(); h = L[0].split("\t"); it = h.index("TPM")
            for line in L[1:]:
                f = line.split("\t"); g = f[0].split(".")[0]
                if g in gi:
                    TPM[gi[g], ci[(t, r)]] = float(f[it])
            print("tpm", t, r, acc, flush=True)
        intr = kb.clean_introns(genes, span, ex); INT = np.zeros((len(genes), len(cols)), np.float32)
        for kind in ("plus", "minus"):
            for t, r, acc, url in F[kind]:
                p = os.path.join(RAW, "rna", "tmp", acc + ".bigWig"); xb.fetch(url, p)
                bw = pyBigWig.open(p); ch = bw.chroms()
                for g, pcs in intr.items():
                    c = "chr" + span[g][0]
                    if c not in ch:
                        continue
                    s = l = 0.0
                    for a, b in pcs:
                        v = bw.stats(c, a, min(b, ch[c]), type="mean", exact=True)[0]; s += (v or 0.0) * (b - a); l += b - a
                    INT[gi[g], ci[(t, r)]] += s / l
                bw.close(); os.remove(p); print("introns", kind, t, r, acc, flush=True)
        for g in genes:
            if g not in intr:
                INT[gi[g]] = np.nan
        tss_pc = {c: [g[4] - 1 for g in tss if g[3].replace("chr", "") == c] for c in CHROMS}
        E1, info, chrom_of, start_of = [], {}, [], []
        H = {t: [np.load(os.path.join(RAW, "hic", "raw", f"rep_{t}_{r}.npz")) for r in REPS] for t in TIMES}
        for c in CHROMS:
            n = int(max(z[f"{c}_j"].max() for t in TIMES for z in H[t])) + 1
            dens = np.zeros(n)
            for p_ in tss_pc.get(c, []):
                if p_ // BIN < n:
                    dens[p_ // BIN] += 1
            vec, rec = {}, {}
            for t in TIMES:
                i = np.concatenate([z[f"{c}_i"] for z in H[t]]); j = np.concatenate([z[f"{c}_j"] for z in H[t]]); v = np.concatenate([z[f"{c}_v"] for z in H[t]]).astype(np.float64)
                V = kb.eig3(i, j, v, n); ref = dens if t == 0 else vec[0]
                rs = [np.corrcoef(V[:, k][~np.isnan(V[:, k]) & ~np.isnan(ref)], ref[~np.isnan(V[:, k]) & ~np.isnan(ref)])[0, 1] for k in range(3)]
                k = int(np.argmax(np.abs(rs))); vec[t] = kb.zs(V[:, k] * np.sign(rs[k])); rec[t] = dict(pc=k + 1, r=float(abs(rs[k])))
            info[c] = dict(bins=n, density_r_0h=rec[0]["r"], pc_0h=rec[0]["pc"], **{f"r_vs_0h_{t}": rec[t]["r"] for t in TIMES[1:]}, **{f"pc_{t}": rec[t]["pc"] for t in TIMES[1:]})
            E1.append(np.stack([vec[t] for t in TIMES], 1)); chrom_of += [c] * n; start_of += list(range(0, n * BIN, BIN))
            print("compartments", c, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info[c].items()}), flush=True)
        E1, bchr, bstart, cinfo = np.concatenate(E1), np.array(chrom_of), np.array(start_of), info
        np.savez_compressed(mid, gene=np.array(genes), tpm=TPM, intron=INT, e1=E1, bin_chrom=bchr, bin_start=bstart)
        json.dump(cinfo, open(mid + ".json", "w"))
    U = union_loops()
    LC = np.stack([np.load(os.path.join(RAW, "hic", "raw", f"rep_{t}_{r}.npz"))["loops"] for t, r in cols], 1)
    Lg = np.log2(LC / LC.sum(0) * 1e6 + 1)
    lfc = np.stack([np.zeros(len(U))] + [Lg[:, [ci[(t, r)] for r in REPS]].mean(1) - Lg[:, [ci[(0, r)] for r in REPS]].mean(1) for t in TIMES[1:]], 1)
    d, p = moderated_test(Lg[:, [ci[(0, r)] for r in REPS]], Lg[:, [ci[(720, r)] for r in REPS]])
    o = np.argsort(p); qv = np.empty(len(p)); qq = p[o] * len(p) / np.arange(1, len(p) + 1); qv[o] = np.minimum(np.minimum.accumulate(qq[::-1])[::-1], 1)
    cl = np.where(qv < 0.1, np.where(d > 0, "gained", "lost"), "static")
    W = LOOP_BIN
    np.savez_compressed(os.path.join(cbb.LAY, "a549_dex_3d.npz"),
                        gene=np.array(genes), symbol=np.array([g[1] for g in tss]), chrom=np.array([g[3].replace("chr", "") for g in tss]),
                        tss=np.array([g[4] - 1 for g in tss]), span=np.array([[span[g][1], span[g][2]] for g in genes]),
                        rna_cols=np.array([f"{t}_{r}" for t, r in cols]), tpm=TPM, int_cols=np.array([f"{t}_{r}" for t, r in cols]), intron=INT,
                        intron_bp=np.array([sum(b - a for a, b in intr.get(g, [])) for g in genes]),
                        bin_chrom=bchr, bin_start=bstart, e1=E1.astype(np.float32), e1_cols=np.array([str(t) for t in TIMES]),
                        loop_anchor=np.array([[ch, int(a // W) * W, int(a // W) * W + W, ch, int(b // W) * W, int(b // W) * W + W] for ch, a, b, obs, t in U]).astype(str),
                        loop_lfc=lfc, loop_lfc_cols=np.array([str(t) for t in TIMES]), loop_padj=qv, loop_p=p, loop_cluster=cl,
                        loop_first_called=np.array([t for *_, t in U]), loop_libs=np.array([f"m{t:04d}_r{r}" for t, r in cols]), loop_counts=LC.astype(np.float32))
    info = dict(genes=len(genes), genes_with_clean_introns=len(intr), bins=int(len(bchr)), bins_valid_all_times=int((~np.isnan(E1).any(1)).sum()),
                compartments=cinfo, loops=len(U), loop_classes=dict(collections.Counter(cl.tolist())), loop_counts_median=float(np.median(LC.sum(1))))
    json.dump(info, open(os.path.join(RAW, "build_info.json"), "w"), indent=1); print(json.dumps({k: v for k, v in info.items() if k != "compartments"}))


def bank():
    info = json.load(open(os.path.join(RAW, "build_info.json")))
    raws = [cbb.layer_fact(f"A549DEX_RAW_{k.upper()}", 1, "encode_a549_dex", os.path.join(RAW, k, "raw"), "downloaded files", "-", "as published") for k in ("rna", "hic")]
    lay = cbb.layer_fact("A549_DEX_3D", 1, "encode_a549_dex", os.path.join(cbb.LAY, "a549_dex_3d.npz"),
                         f"{info['genes']} protein-coding genes; {info['bins']} 100-kb bins; {info['loops']} loops",
                         "RNA TPM + intron signal (5 times x 4 reps); compartment E1 (5 times); loop counts (5 times x 4 reps)",
                         "TPM; mean coverage; E1 z per chromosome; raw counts", inputs=tuple(f.ref for f in raws) + ("LAYER-K562PMA_RAW_GENCODE",), tier="derived")
    n = cbb.bank().add_many(raws + [lay])
    rep = dict(added=time.strftime("%Y-%m-%d"), facts_written=n, layers={f.entity: f.ref for f in raws + [lay]}, build=info)
    json.dump(rep, open(R("outputs", "cellbrain", "a549dex_report.json"), "w"), indent=1, default=int); print(json.dumps(rep, default=int)[:800])


if __name__ == "__main__":
    {"pull": pull, "build": build, "bank": bank}[sys.argv[1]]()
