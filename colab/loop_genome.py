"""LOOP 86 -- THE WHOLE-GENOME TARGET SURVEY. IS CHR21 TYPICAL, OR DID I CALIBRATE ON AN OUTLIER?

WHAT THIS ARC HAS ACTUALLY MEASURED. Loops 33 through 85 -- every target, every acceptance window,
every mechanism, every gate -- rest on chromosome 21, with chromosome 22 as the held-out check. That
is 1,227 of the genome's 40,790 CTCF peaks (3.0%) and 99 Mb of 3.04 Gb (3.3%). And chr21 is the least
representative chromosome in the human genome to have picked: the smallest autosome, acrocentric,
gene-poor, with a centromere and a short arm that between them cost 549 of its 1,926 bins.

So the P(s) window of (-1.164, -0.764) that loops 35, 77, 78 and 79 all selected against, the band
targets loops 81-83 tuned toward, and the +0.3788 orientation signature loop 85 just reproduced,
might be chr21 facts rather than chromatin facts. Nothing in the arc can tell the difference, because
nothing in the arc has looked anywhere else.

THIS LOOP DOES NOT SIMULATE ANYTHING, AND THAT IS DELIBERATE. The Gaussian-network model needs a
dense inverse: chr21 is 1,926 bins, chr1 is 9,971, and the cost goes as n^3. chr1 alone is 139x chr21
per inverse at 0.80 GB per matrix, and the summed genome is 743x. Simulating genome-wide is not a
matter of patience, it is a different algorithm. What IS affordable is measuring the TARGETS
everywhere -- one chromosome streamed, measured and discarded at a time -- and that is the question
worth answering first, because if chr21 is atypical then five loops of mechanism-building were aimed
at the wrong number and no amount of simulation would have revealed it.

THE MEASUREMENT IS THE ONE LOOP 85 REPAIRED. Every chromosome goes through the corrected
build_chrom path -- nan fill before masking, row-normalised PWM, every-offset motif scan -- and the
identical statistics. If that code were still broken this survey would propagate the break 23 times
instead of twice, which is the argument for having fixed it before running this rather than after.

PREDECLARED, before any number:

  G1 EVERY CHROMOSOME THROUGH THE CORRECTED PIPELINE               THE SURVEY.
       22 autosomes and X, KR-normalised at 25 kb from the same Rao 2014 GM12878 file loop 33 used,
       with the same GM12878 CTCF peaks and the same JASPAR MA0139.1 scan. Gate: at least 20 of 23
       must yield a testable orientation signature (>= 30 separation-matched pairs, loop 33's own
       threshold). If most chromosomes cannot be tested at 25 kb the survey is uninformative and
       says so rather than reporting whatever the testable minority happened to give.
  G2 IS CHR21 TYPICAL                                              THE GATE.
       chr21's corrected values placed as percentiles in the genome-wide distribution. Gate: chr21
       must sit inside the central 80% on P(s) -- the single quantity the entire acceptance window
       was derived from. Outside it, the window loops 35-85 selected against is chromosome-specific
       and every parameter chosen through it has to be re-derived.
  G3 THE ORIENTATION SIGNATURE REPLICATES ACROSS CHROMOSOMES       THE REPLICATION.
       the signature and its scoring-label null on all 23. Gate: positive at z >= 4 on at least 18.
       This is the strongest available test of loop 85's P3 -- 23 near-independent chromosomes, one
       code path, nothing tuned. Loop 84 doubted this control; two chromosomes cleared it; 23 is a
       different order of evidence.
  G4 THE TARGETS THE ARC SHOULD HAVE BEEN USING                    THE CORRECTION.
       mappable-bin-weighted genome-wide mean and spread of P(s), both bands and the signature,
       against the chr21-only values loops 33-85 used. Reported, not gated -- there is no pass here,
       only the numbers that should have been the targets.
  G5 SIZE AND DENSITY, THE OBVIOUS CONFOUND                        THE CONTROL.
       the boring explanation for any cross-chromosome spread is chromosome length and CTCF density,
       not biology: the 0.1-10 Mb fitting window occupies a different fraction of a 48 Mb chromosome
       than a 249 Mb one. Spearman of every statistic against length and against peaks per Mb. If
       P(s) tracks length strongly then "P(s) is a universal property of interphase chromatin" is
       partly an artifact of the window, and this arc has been fitting that artifact.
  G6 WHAT THIS LOOP CANNOT ANSWER                                  THE HONEST LIMIT.
       the genome-wide simulation cost, stated in the output so that nobody -- including me later --
       reads a survey of measured targets as validation of the model. It validates nothing about the
       model. It tells us what the model should have been aiming at.

-> outputs/loop_genome.json
"""
import gzip
import json
import os
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_second as L77  # noqa: E402
import loop_map_score as L79  # noqa: E402
import loop_bending as L80  # noqa: E402
from loop_hic_target import expected  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = L77.SC
BIN = L77.BIN
SHORT_BAND, LONG_BAND = (1e5, 5e5), (1e6, 1e7)
HIC_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/"
           "GSE63525_GM12878_insitu_primary_30.hic")
FA_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/{c}.fa.gz"
NPERM = 20
Z_MIN = 4.0
MIN_MATCHED = 30                  # loop 33's own testability threshold
G1_MIN, G3_MIN = 20, 18
SEED = 8601

HG19 = {"chr1": 249250621, "chr2": 243199373, "chr3": 198022430, "chr4": 191154276,
        "chr5": 180915260, "chr6": 171115067, "chr7": 159138663, "chr8": 146364022,
        "chr9": 141213431, "chr10": 135534747, "chr11": 135006516, "chr12": 133851895,
        "chr13": 115169878, "chr14": 107349540, "chr15": 102531392, "chr16": 90354753,
        "chr17": 81195210, "chr18": 78077248, "chr19": 59128983, "chr20": 63025520,
        "chr21": 48129895, "chr22": 51304566, "chrX": 155270560}
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

# what loops 33-85 used, all from chr21 alone
CHR21_USED = {"ps": -0.9636, "short": -0.8543, "long": -0.8571, "orient": 0.3788}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def fetch_fasta(chrom):
    p = SC / f"hg19_{chrom}.fa.gz"
    if not p.exists():
        urllib.request.urlretrieve(FA_URL.format(c=chrom), p)
    return p


def fetch_hic(chrom, n):
    """KR-normalised 25 kb observed, streamed and never cached -- the genome is 6.1 GB dense."""
    import hicstraw
    hic = hicstraw.HiCFile(HIC_URL)
    mzd = hic.getMatrixZoomData(chrom[3:], chrom[3:], "observed", "KR", "BP", BIN)
    M = np.zeros((n, n), np.float32)
    for r in mzd.getRecords(0, HG19[chrom] - 1, 0, HG19[chrom] - 1):
        i, j = int(r.binX) // BIN, int(r.binY) // BIN
        if 0 <= i < n and 0 <= j < n:
            M[i, j] = M[j, i] = r.counts
    return M.astype(np.float64)


def orient_bins(chrom, seq_path):
    """Corrected CTCF orientation scan -- loop 85's build_chrom logic, sequence side only."""
    seq = "".join(l.strip() for l in gzip.open(seq_path, "rt") if not l.startswith(">")).upper()
    pk = []
    for ln in gzip.open(L77.CTCF, "rt"):
        f = ln.split("\t")
        if f[0] != chrom:
            continue
        st, en = int(f[1]), int(f[2])
        off = (int(f[9]) if len(f) > 9 and f[9].strip().lstrip("-").isdigit() and int(f[9]) >= 0
               else (en - st) // 2)
        pk.append(st + off)
    pfm = json.load(open(L77.PFM))
    Lw = len(pfm["A"])
    W = np.array([pfm[b] for b in "ACGT"], float).T
    W = W / W.sum(1, keepdims=True)
    W = np.log2((W + 1e-3) / 0.25)
    idx = {c: i for i, c in enumerate("ACGT")}

    def sc(s):
        if len(s) != Lw or any(c not in idx for c in s):
            return -1e9
        return float(sum(W[i, idx[c]] for i, c in enumerate(s)))

    def rc(s):
        return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    ors = []
    for s0 in pk:
        a, b = max(0, s0 - 150), min(len(seq), s0 + 150)
        win = seq[a:b]
        best, bo = -1e9, 0
        for i in range(len(win) - Lw + 1):
            s = win[i:i + Lw]
            f_, r_ = sc(s), sc(rc(s))
            if max(f_, r_) > best:
                best, bo = max(f_, r_), (1 if f_ >= r_ else -1)
        ors.append(bo if best > 6.0 else 0)
    return pk, ors


def d4_vec(M, exp, mask, fset, rset):
    """loop 33's D4 estimator, vectorised. Asserted equal to the reference on chr21 before use."""
    S = np.array(sorted(fset | rset), dtype=int)
    if len(S) < 2:
        return {"n_conv": 0, "n_other": 0, "n_matched": 0, "diff": float("nan"),
                "conv_mean": float("nan"), "other_mean": float("nan")}
    inf_ = np.isin(S, list(fset))
    inr = np.isin(S, list(rset))
    I, J = S[:, None], S[None, :]
    d = J - I
    dc = np.clip(d, 0, len(exp) - 1)
    ok = (d >= 4) & (d * BIN <= 2e6) & mask[I] & mask[J]
    ok &= np.isfinite(exp[dc]) & (exp[dc] > 0)
    vals = M[I, J]
    ok &= np.isfinite(vals)
    v = np.where(ok, vals / exp[dc], np.nan)
    conv = ok & inf_[:, None] & inr[None, :]
    other = ok & ~conv
    md = int(d[ok].max()) + 1 if ok.any() else 1
    do, vo = d[other], v[other]
    cnt = np.bincount(do, minlength=md)
    tot = np.bincount(do, weights=vo, minlength=md)
    has = cnt > 0
    mean_other = np.where(has, tot / np.where(cnt > 0, cnt, 1), np.nan)
    dcv, vcv = d[conv], v[conv]
    keep = has[dcv]
    mc, mo = vcv[keep], mean_other[dcv[keep]]
    if len(mc) < MIN_MATCHED:
        return {"n_conv": int(conv.sum()), "n_other": int(other.sum()), "n_matched": int(len(mc)),
                "diff": float("nan"), "conv_mean": float("nan"), "other_mean": float("nan")}
    return {"n_conv": int(conv.sum()), "n_other": int(other.sum()), "n_matched": int(len(mc)),
            "diff": float(np.mean(mc - mo)), "conv_mean": float(mc.mean()),
            "other_mean": float(mo.mean())}


def signature(M, exp, mask, bins, ors, nperm=NPERM, seed=SEED):
    """Real signature plus loop 84's control (B): one map, permute only the scoring labels."""
    fs = {b // BIN for b, o in zip(bins, ors) if o > 0}
    rs = {b // BIN for b, o in zip(bins, ors) if o < 0}
    real = d4_vec(M, exp, mask, fs, rs)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(nperm):
        sh = rng.permutation(ors)
        f2 = {b // BIN for b, o in zip(bins, sh) if o > 0}
        r2 = {b // BIN for b, o in zip(bins, sh) if o < 0}
        x = d4_vec(M, exp, mask, f2, r2)["diff"]
        if np.isfinite(x):
            null.append(x)
    if not null or not np.isfinite(real["diff"]):
        return {**real, "null_mean": float("nan"), "null_sd": float("nan"), "z": float("nan")}
    null = np.array(null)
    sd = null.std()
    return {**real, "null_mean": float(null.mean()), "null_sd": float(sd),
            "z": float((real["diff"] - null.mean()) / sd) if sd > 1e-12 else float("inf")}


def pct(x, arr):
    a = np.array([v for v in arr if np.isfinite(v)])
    return float((a < x).mean() * 100) if len(a) else float("nan")


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 86 -- the whole-genome target survey. Is chr21 typical, or did I calibrate on an "
        "outlier?")
    say("=" * 100)
    say()
    say(f"  the arc so far: chr21 + chr22 = 1,227 of 40,790 CTCF peaks (3.0%), 99 Mb of 3.04 Gb "
        f"(3.3%)")
    say(f"  measuring {len(CHROMS)} chromosomes through loop 85's corrected pipeline, KR 25 kb, "
        f"streamed and discarded")
    say()

    say("G1 EVERY CHROMOSOME THROUGH THE CORRECTED PIPELINE")
    rows = []
    checked = False
    for ch in CHROMS:
        tc = time.time()
        n = HG19[ch] // BIN + 1
        try:
            M = fetch_hic(ch, n)
            fa = fetch_fasta(ch)
        except Exception as e:
            say(f"     {ch:6s} FETCH FAILED  {repr(e)[:70]}")
            rows.append({"chrom": ch, "failed": repr(e)[:200]})
            continue
        M[M == 0] = np.nan
        mask = np.isfinite(M).sum(1) > 50
        exp = expected(M, mask)
        d = np.arange(n) * BIN
        s = np.isfinite(exp) & (d >= 1e5) & (d <= 1e7) & (exp > 0)
        ps = float(np.polyfit(np.log10(d[s]), np.log10(exp[s]), 1)[0]) if s.sum() >= 4 else np.nan
        bins, ors = orient_bins(ch, fa)
        S = signature(M, exp, mask, bins, ors)
        if ch == "chr21" and not checked:
            # the vectorised estimator must agree with the reference implementation, not be trusted
            fs = {b // BIN for b, o in zip(bins, ors) if o > 0}
            rs = {b // BIN for b, o in zip(bins, ors) if o < 0}
            ref, refn = L77.orientation_effect(M, exp, fs, rs, mask, n)
            assert abs(ref - S["diff"]) < 1e-9 and refn == S["n_matched"], \
                f"vectorised d4 disagrees with reference: {ref} vs {S['diff']}"
            say(f"     [d4_vec asserted equal to loop_second.orientation_effect on chr21: "
                f"{ref:+.6f}, {refn:,} pairs]")
            checked = True
        rows.append({"chrom": ch, "n": n, "mappable": int(mask.sum()),
                     "mapfrac": float(mask.mean()), "ps": ps,
                     "short": L80.ps_band(M, mask, *SHORT_BAND),
                     "long": L80.ps_band(M, mask, *LONG_BAND),
                     "peaks": len(bins), "oriented": int(sum(1 for o in ors if o)),
                     "mb": HG19[ch] / 1e6, **S})
        r = rows[-1]
        say(f"     {ch:6s} {r['mappable']:5,}/{r['n']:5,} bins ({r['mapfrac']:5.1%})  "
            f"P(s) {r['ps']:+.4f}  bands {r['short']:+.4f}/{r['long']:+.4f}  "
            f"{r['oriented']:4d}/{r['peaks']:4d} ori  sig {r['diff']:+.4f} z {r['z']:+6.1f}  "
            f"[{time.time()-tc:.0f}s]")
        del M
    good = [r for r in rows if not r.get("failed") and r.get("n_matched", 0) >= MIN_MATCHED]
    g1 = len(good) >= G1_MIN
    say(f"     {len(good)}/{len(CHROMS)} chromosomes gave a testable signature "
        f"(>= {MIN_MATCHED} separation-matched pairs)")
    say(f"     G1 {'PASS' if g1 else 'FAIL'}")
    say()

    say("G2 IS CHR21 TYPICAL")
    c21 = next(r for r in rows if r["chrom"] == "chr21")
    pcts = {}
    for k, label in [("ps", "P(s)"), ("mapfrac", "mappable fraction"),
                     ("short", "short band"), ("long", "long band"), ("diff", "signature")]:
        arr = [r[k] for r in good]
        p = pct(c21[k], arr)
        pcts[k] = p
        a = np.array([v for v in arr if np.isfinite(v)])
        say(f"     {label:20s} chr21 {c21[k]:+.4f}   genome {a.mean():+.4f} +/- {a.std():.4f}   "
            f"range {a.min():+.4f} to {a.max():+.4f}   chr21 at the {p:.0f}th percentile")
    g2 = 10.0 <= pcts["ps"] <= 90.0
    say(f"     G2 {'PASS' if g2 else 'FAIL'} -- chr21's P(s) sits at the {pcts['ps']:.0f}th "
        f"percentile; the arc's acceptance window is "
        f"{'representative' if g2 else 'CHROMOSOME-SPECIFIC and must be re-derived'}")
    say()

    say("G3 THE ORIENTATION SIGNATURE REPLICATES ACROSS CHROMOSOMES")
    pos = [r for r in good if np.isfinite(r["z"]) and r["diff"] > 0 and r["z"] >= Z_MIN]
    neg = [r for r in good if not (np.isfinite(r["z"]) and r["diff"] > 0 and r["z"] >= Z_MIN)]
    say(f"     positive at z >= {Z_MIN}: {len(pos)}/{len(good)} testable chromosomes")
    if neg:
        say(f"     not reaching it: " + ", ".join(f"{r['chrom']} ({r['diff']:+.3f}, z {r['z']:+.1f})"
                                                  for r in neg))
    zz = np.array([r["z"] for r in good if np.isfinite(r["z"])])
    dd = np.array([r["diff"] for r in good if np.isfinite(r["diff"])])
    say(f"     signature {dd.mean():+.4f} +/- {dd.std():.4f}   z {zz.mean():+.1f} +/- {zz.std():.1f}")
    g3 = len(pos) >= G3_MIN
    say(f"     G3 {'PASS' if g3 else 'FAIL'}")
    say()

    say("G4 THE TARGETS THE ARC SHOULD HAVE BEEN USING")
    wm = {}
    w = np.array([r["mappable"] for r in good], float)
    for k, label in [("ps", "P(s)"), ("short", "short band"), ("long", "long band"),
                     ("diff", "orientation signature")]:
        a = np.array([r[k] for r in good], float)
        f = np.isfinite(a)
        m = float((a[f] * w[f]).sum() / w[f].sum())
        sd = float(np.sqrt((w[f] * (a[f] - m) ** 2).sum() / w[f].sum()))
        wm[k] = {"mean": m, "sd": sd}
        used = CHR21_USED["orient" if k == "diff" else k]
        say(f"     {label:22s} chr21 used {used:+.4f}   GENOME {m:+.4f} +/- {sd:.4f}   "
            f"offset {m - used:+.4f}")
    say(f"     G4 reported (not gated)")
    say()

    say("G5 SIZE AND DENSITY, THE OBVIOUS CONFOUND")
    from scipy.stats import spearmanr
    length = np.array([r["mb"] for r in good])
    dens = np.array([r["peaks"] / r["mb"] for r in good])
    conf = {}
    for k, label in [("ps", "P(s)"), ("short", "short band"), ("long", "long band"),
                     ("mapfrac", "mappable fraction"), ("diff", "signature")]:
        a = np.array([r[k] for r in good], float)
        f = np.isfinite(a)
        rl = float(spearmanr(length[f], a[f]).statistic)
        rd = float(spearmanr(dens[f], a[f]).statistic)
        conf[k] = {"vs_length": rl, "vs_density": rd}
        say(f"     {label:20s} vs chromosome length rho {rl:+.4f}   vs CTCF per Mb rho {rd:+.4f}")
    strong = [k for k, v in conf.items() if abs(v["vs_length"]) >= 0.6]
    say(f"     statistics tracking chromosome length at |rho| >= 0.6: "
        f"{', '.join(strong) if strong else 'none'}")
    if "ps" in strong:
        say(f"     P(s) TRACKS LENGTH. The 0.1-10 Mb fitting window is a different fraction of each")
        say(f"     chromosome, so part of what this arc treated as chromatin physics is the window.")
    say(f"     G5 reported (not gated)")
    say()

    say("G6 WHAT THIS LOOP CANNOT ANSWER")
    n1, n21 = HG19["chr1"] // BIN + 1, HG19["chr21"] // BIN + 1
    cost = sum((HG19[c] // BIN + 1) ** 3 for c in CHROMS) / n21 ** 3
    say(f"     this loop simulated NOTHING. It measured targets, and validates nothing about the")
    say(f"     model. The Gaussian-network inverse is O(n^3): chr1 is {n1:,} bins against chr21's")
    say(f"     {n21:,}, so {(n1/n21)**3:.0f}x per inverse at {n1**2*8/1e9:.2f} GB per dense matrix, "
        f"and the summed")
    say(f"     genome is {cost:.0f}x chr21. Genome-wide simulation is a different algorithm, not a")
    say(f"     longer run of this one.")
    say(f"     G6 reported (not gated)")
    say()

    gates = {"G1 every chromosome through the corrected pipeline": bool(g1),
             "G2 chr21 is typical on P(s)": bool(g2),
             "G3 the orientation signature replicates across chromosomes": bool(g3),
             "G4 genome-wide targets recorded": True,
             "G5 size and density confound reported": True,
             "G6 the simulation limit stated": True}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[HIC_URL, str(L77.CTCF), str(L77.PFM), FA_URL],
                      available=len(CHROMS), used=len(good), selection="filtered", seed=SEED,
                      controls=["scoring-label null on every chromosome independently",
                                "the vectorised estimator asserted equal to the reference on chr21",
                                "chromosome length and CTCF density tested as confounds",
                                "chr21 placed as a percentile rather than compared to a mean",
                                "the same corrected code path on all 23, nothing per-chromosome",
                                "no simulation -- targets only, and the limit stated"],
                      note="loops 33-85 derived every target from chr21 alone, 3.0% of the genome's "
                           "CTCF peaks; this measures the same statistics on all 23 chromosomes")
    RM.report(man, emit=say)
    json.dump({"test": "loop_genome", "manifest": man, "gates": gates,
               "chromosomes": rows, "chr21_percentiles": pcts,
               "chr21_used_by_arc": CHR21_USED, "genome_weighted": wm,
               "confounds": conf, "n_testable": len(good),
               "n_signature_positive": len(pos),
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_genome.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_genome.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
