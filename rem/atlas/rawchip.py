"""Would RAWER binding data let the bank name direct targets? One small part: 12 factors.

FROM celldiscover.py. "Knocked down AND bound AND in no database" failed its null: a factor's
knockdown movers overlapped OTHER factors' binding lists as often as its own (FDR 1.23). The
binding data were 2015 Enrichr GENE SETS -- ~1,200 genes per factor, no positions, no strengths.

THE RAWER DATA. ENCODE's own peak files: every binding site's genomic position (summit) and
strength (signalValue), GRCh38, released, from the ENCODE portal (ENCODE data: no use
restrictions). A gene counts as BOUND only if a peak summit lies within 1,000 bp of one of its
transcription start sites (UCSC hg38 refGene), and its binding strength is the strongest such
peak. (Truly raw reads are hundreds of GB; peak calls are the lowest practical level here.)

THE SMALL PART. The 12 factors, among those knocked down in K562 genome-wide Perturb-seq AND in
the 2015 ENCODE K562 sets, with the MOST movers in their own knockdown (so there is signal to
find). Per factor, one ENCODE K562 TF ChIP-seq experiment: the first released experiment (by
accession) that has GRCh38 'conservative IDR thresholded peaks' (else 'IDR thresholded peaks').
Movers: |value| >= 3 / 6.85 (celldiscover's measured scale), finite.

=================================================================================================
GATES, PREDECLARED -- the same questions asked of BOTH binding sources on the SAME 12 factors
=================================================================================================
T1 BEATS THE NULL? total triangulated pairs (factor's movers bound by that factor) vs 1,000
   random derangements of factor <-> binding set among the 12. One-sided permutation p < 0.05 ->
   SHARP ENOUGH; else NOT SHARP ENOUGH.
T2 ENRICHMENT PER FACTOR: odds ratio of moving, bound vs unbound measured genes (Fisher two-sided);
   count factors with OR > 1 and p < 0.05.
T3 DOSE-RESPONSE (peaks only): among bound genes, Spearman between peak strength and |value|,
   per factor; sign test across factors (p < 0.05 -> STRONGER BINDING, BIGGER EFFECT).
T4 SIGN CONTROL: agreement of triangulated pairs with OmniPath's sign; a verdict only if >= 20.
DISCOVERY only if T1 passes for the peaks: triangulated pairs in neither OmniPath nor the bank,
with estimated FDR = null mean / observed. If T1 fails, nothing is listed as a candidate.
"""

from __future__ import annotations
import collections
import gzip
import importlib.util
import json
import math
import os
import time
import urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
CA = os.path.join(HERE, "_cache")
EDIR = os.path.join(CA, "encode", "peaks")
OUT = os.path.join(HERE, "RESULTS_rawchip.txt")
ART = R("outputs", "rawchip.json")
RULE = "=" * 97
ENC = "https://www.encodeproject.org"
WIN, NTF, NPERM = 1000, 12, 1000
_s = importlib.util.spec_from_file_location("celldiscover", os.path.join(HERE, "celldiscover.py"))
cd = importlib.util.module_from_spec(_s); _s.loader.exec_module(cd)


def get_json(url):
    """ENCODE answers an EMPTY search with HTTP 404; that means 'no results', not failure."""
    import urllib.error
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=120))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"@graph": []}
        raise


def fetch_peaks(tf):
    """First released K562 TF ChIP-seq experiment (by accession) with GRCh38 conservative IDR peaks."""
    os.makedirs(EDIR, exist_ok=True)
    exps = get_json(f"{ENC}/search/?type=Experiment&assay_title=TF+ChIP-seq&biosample_ontology.term_name=K562"
                    f"&status=released&target.label={tf}&format=json&limit=all")["@graph"]
    for e in sorted(exps, key=lambda x: x["accession"]):
        for ot in ("conservative IDR thresholded peaks", "IDR thresholded peaks"):
            fs = get_json(f"{ENC}/search/?type=File&dataset=/experiments/{e['accession']}/&output_type={ot.replace(' ', '+')}"
                          f"&assembly=GRCh38&file_format=bed&status=released&format=json&limit=all")["@graph"]
            if fs:
                f = sorted(fs, key=lambda x: x["accession"])[0]
                path = os.path.join(EDIR, f"{tf}_{f['accession']}.bed.gz")
                if not os.path.exists(path):
                    urllib.request.urlretrieve(ENC + f["href"], path)
                return dict(tf=tf, experiment=e["accession"], file=f["accession"], output_type=ot, path=path)
    return None


def tss_table():
    path = os.path.join(CA, "encode", "refGene_hg38.txt.gz")
    if not os.path.exists(path):
        urllib.request.urlretrieve("https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/refGene.txt.gz", path)
    by_chr = collections.defaultdict(list)
    for line in gzip.open(path, "rt"):
        p = line.split("\t")
        chrom, strand, txs, txe, name = p[2], p[3], int(p[4]), int(p[5]), p[12]
        by_chr[chrom].append((txs if strand == "+" else txe, name))
    return {c: (np.array([x[0] for x in v]), [x[1] for x in v]) for c, v in by_chr.items()}


def bound_genes(path, tss):
    """gene -> strongest signalValue of a peak summit within WIN of one of its TSSs."""
    out = {}
    for line in gzip.open(path, "rt"):
        p = line.rstrip("\n").split("\t")
        chrom, st, sig, off = p[0], int(p[1]), float(p[6]), int(p[9])
        summit = st + (off if off >= 0 else (int(p[2]) - st) // 2)
        if chrom not in tss:
            continue
        pos, names = tss[chrom]
        lo, hi = np.searchsorted(pos, summit - WIN), np.searchsorted(pos, summit + WIN)
        for i in range(lo, hi):
            g = names[i]
            out[g] = max(out.get(g, 0.0), sig)
    return out


def main():
    from scipy.stats import fisher_exact, spearmanr
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    rng = np.random.default_rng(20261007)
    X, kos, genes, ep, fold = cd.load_gwps()
    gidx = {g: j for j, g in enumerate(genes)}; kidx = {k: i for i, k in enumerate(kos)}
    zmov = 3.0 / 6.85
    sets15 = cd.load_encode(); omni = cd.load_omnipath()
    omni_tx = {(s, t): sg for s, t, sg, ty, _ in omni if ty == "transcriptional"}
    omni_all = {(s, t) for s, t, _, _, _ in omni}
    D = json.load(gzip.open(cd.dc.ENCY)); bn = [r["name"] for r in D["genes"]]
    bank_pairs = {(s, t) for s, t, _, _ in cd.cb.facts_table(D, bn)[0]}
    cand = [tf for tf in sets15 if tf in kidx]
    nmov = {tf: int(np.nansum(np.abs(X[kidx[tf]]) >= zmov)) for tf in cand}
    tfs = sorted(cand, key=lambda t: (-nmov[t], t))[:NTF]
    P_(RULE); P_("RAWER BINDING DATA ON ONE SMALL PART: 12 FACTORS, PEAKS vs 2015 GENE SETS"); P_(RULE)
    P_("  factors (most movers in their own K562 knockdown): " + ", ".join(f"{t} ({nmov[t]})" for t in tfs))
    tss = tss_table()
    meta, peaks = [], {}
    for tf in tfs:
        m = fetch_peaks(tf)
        if m is None:
            P_(f"    {tf}: no GRCh38 IDR peak file found on ENCODE -- dropped"); continue
        peaks[tf] = bound_genes(m["path"], tss)
        meta.append(m)
        P_(f"    {tf:<8} {m['experiment']} {m['file']} ({m['output_type']}): promoter-bound genes {len(peaks[tf]):,} "
           f"(2015 gene set {len(sets15[tf]):,})")
    tfs = [t for t in tfs if t in peaks]
    meas = [g for g in genes]
    res = {}
    for src_name, B in (("PEAKS (raw positions + strength)", {t: set(peaks[t]) for t in tfs}),
                        ("2015 GENE SETS", {t: set(sets15[t]) for t in tfs})):
        P_("\n" + RULE); P_(src_name); P_(RULE)
        mv = {t: {g for g in genes if g != t and np.isfinite(X[kidx[t], gidx[g]]) and abs(X[kidx[t], gidx[g]]) >= zmov} for t in tfs}
        bset = {t: B[t] & set(meas) for t in tfs}
        real = sum(len(mv[t] & bset[t]) for t in tfs)
        null = []
        for _ in range(NPERM):
            perm = list(tfs)
            while True:
                rng.shuffle(perm)
                if all(a != b for a, b in zip(tfs, perm)):
                    break
            null.append(sum(len(mv[t] & (bset[o] - {t})) for t, o in zip(tfs, perm)))
        p1 = (1 + sum(n >= real for n in null)) / (NPERM + 1)
        v1 = "SHARP ENOUGH" if p1 < 0.05 else "NOT SHARP ENOUGH"
        P_(f"  T1 triangulated pairs {real:,} vs null {np.mean(null):,.1f} +/- {np.std(null):.1f}; permutation p {p1:.4f} -> {v1}")
        ors, sig = [], 0
        for t in tfs:
            a = len(mv[t] & bset[t]); b = len(bset[t] - mv[t] - {t}); c = len(mv[t] - bset[t]); d = len(meas) - a - b - c - 1
            orr, p = fisher_exact([[a, b], [c, d]])
            ors.append(orr); sig += int(orr > 1 and p < 0.05)
            P_(f"     {t:<8} movers {len(mv[t]):>5}  bound {len(bset[t]):>6}  both {a:>4}  odds ratio {orr:>6.2f}  p {p:.2g}")
        P_(f"  T2 factors with significant enrichment (OR > 1, p < 0.05): {sig}/{len(tfs)}; median OR {np.median(ors):.2f}")
        tri = [(t, g, float(X[kidx[t], gidx[g]])) for t in tfs for g in mv[t] & bset[t]]
        agr = [(-np.sign(z) == omni_tx[(t, g)]) for t, g, z in tri if (t, g) in omni_tx]
        P_(f"  T4 sign control: agreement with OmniPath {sum(agr)}/{len(agr)}" + ("" if len(agr) >= 20 else " (fewer than 20: no verdict)"))
        r = dict(real=real, null_mean=float(np.mean(null)), p=p1, verdict=v1, sig_factors=sig, median_or=float(np.median(ors)),
                 sign_agree=int(sum(agr)), sign_n=len(agr))
        if src_name.startswith("PEAKS"):
            rhos = []
            for t in tfs:
                gs = [g for g in peaks[t] if g in gidx and g != t and np.isfinite(X[kidx[t], gidx[g]])]
                if len(gs) >= 20:
                    rho = spearmanr([peaks[t][g] for g in gs], [abs(X[kidx[t], gidx[g]]) for g in gs]).correlation
                    rhos.append(rho)
            w = sum(x > 0 for x in rhos); n_ = len(rhos)
            p3 = min(1.0, 2 * sum(math.comb(n_, i) for i in range(0, min(w, n_ - w) + 1)) / 2 ** n_) if n_ else 1.0
            v3 = "STRONGER BINDING, BIGGER EFFECT" if p3 < 0.05 and w > n_ - w else "NO DOSE-RESPONSE"
            P_(f"  T3 dose-response: Spearman(peak strength, |effect|) positive in {w}/{n_} factors (median {np.median(rhos):+.3f}), p {p3:.3g} -> {v3}")
            r.update(dose_pos=w, dose_n=n_, dose_median=float(np.median(rhos)), dose_p=p3, dose_verdict=v3)
            if p1 < 0.05:
                novel = sorted([(t, g, z) for t, g, z in tri if (t, g) not in omni_all and (t, g) not in bank_pairs], key=lambda x: -abs(x[2]))
                fdr = float(np.mean(null)) / max(real, 1)
                P_(f"  DISCOVERY: {len(novel)} candidate direct links in neither OmniPath nor the bank; estimated FDR {fdr:.2f}")
                for t, g, z in novel[:25]:
                    P_(f"     {t:>8} {'activates' if z < 0 else 'represses'} {g:<10} z-equiv {z * 6.85:+.1f}, peak strength {peaks[t][g]:.0f}")
                r.update(novel=len(novel), fdr=fdr, novel_list=[(t, g, round(z * 6.85, 2), round(peaks[t][g], 1)) for t, g, z in novel])
            else:
                P_("  DISCOVERY: none listed -- T1 did not pass for the peaks.")
        res[src_name.split()[0]] = r
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"factors": tfs, "files": meta, "results": res}, open(ART, "w"), indent=1, default=float)
    P_(f"\n  artifact: outputs/rawchip.json   runtime {time.time() - t0:.0f}s")
    P_("  NOT: 12 factors, one cell line; promoter peaks within 1 kb miss distal enhancers; binding is not regulation.")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    import sys
    if "--fetch-only" in sys.argv:
        X, kos, genes, ep, fold = cd.load_gwps(); kidx = {k: i for i, k in enumerate(kos)}
        s15 = cd.load_encode(); cand = [t for t in s15 if t in kidx]
        nm = {t: int(np.nansum(np.abs(X[kidx[t]]) >= 3 / 6.85)) for t in cand}
        tfs = sorted(cand, key=lambda t: (-nm[t], t))[:NTF]
        tss = tss_table(); print("TSS chromosomes", len(tss))
        for tf in tfs:
            m = fetch_peaks(tf)
            print(tf, "->", None if m is None else (m["experiment"], m["file"], m["output_type"], os.path.getsize(m["path"])))
    else:
        main()
