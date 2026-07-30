"""REAL CLOCK TIME: DOES ACCESSIBILITY LEAD TRANSCRIPTION AT ANY LAG? A549 + dexamethasone.

WHY THIS EXISTS. `dynamic_stage1.py` found accessibility does not forecast expression change on a pseudotime
trajectory, and `dynamic_diagnose.py` showed that was neither a power failure (split-half reliability +0.3991)
nor an artefact of its bad link definition (CRISPR-validated links gave the same answer). But a pseudotime null
cannot separate "accessibility does not lead" from "pseudotime cannot resolve a lead": cells at bin t do not
become the cells at bin t+1, so there is no temporal precedence to detect even in principle.

Real elapsed time can separate those. ENCODE's A549 + 100 nM dexamethasone series has DNase-seq and polyA
RNA-seq at ELEVEN shared timepoints (0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12 h), with RNA down to 5 min.

THE DESIGN WAS SPECIFIED BEFORE RESULTS (recorded in UPDATES.md) so the analysis cannot drift:

 1 A LAG CURVE, NOT ONE TRANSITION. dR2(tau) for every available lag. This is the core fix: the pseudotime
   test asked a single t -> t+1 question, so it could only ever have found a lead at one arbitrary interval.
   A genuine lead-lag must appear as a SPECIFIC TEMPORAL WINDOW. A flat curve near zero is a null; a uniform
   tiny gain at every lag is also a null, and is what a nuisance correlation looks like.

 2 FORECAST ADVANTAGE = dR2_lagged(tau) - dR2_contemporaneous. If accessibility explains CURRENT expression
   but not FUTURE change, it is a state marker and not a forecasting variable. The pseudotime design could
   not see this distinction at all.

 3 PERMUTED-TIME CONTROL. Shuffle the timepoint labels: every distribution is preserved and only the ordering
   is destroyed. This is the control that a lag curve specifically needs.

 4 LINKS PRESPECIFIED BY BIOLOGY, NOT BY VARIANCE. "Linked" means the enhancer carries a glucocorticoid
   receptor (NR3C1) ChIP peak -- GR is the direct effector of dexamethasone, so these are the enhancers
   expected to act. Stage 1 used the most VARIABLE nearby peak, which made its linked-vs-random control
   vacuous. Matched control: a non-GR-bound peak at a similar distance from the same TSS.

 5 GENOME-WIDE AND A PRESPECIFIED RESPONSIVE SET, both reported. Most genes do not respond to dexamethasone,
   so a genome-wide test dilutes a real effect. The subset is defined by GR binding within the window plus
   measurable baseline expression -- biology and detectability, never the outcome. Selecting genes by how much
   they moved would be circular, and is not done.

HELD OUT BY CHROMOSOME, since there is one cell line and genes are the repeated unit; a random row split would
put the same gene on both sides at different timepoints.

ONE REPLICATE PER TIMEPOINT in most of this series, so the time course is a single trajectory and no
replicate-level variance can be estimated. That is a real limit on what a null here can claim, and it is
printed with the result rather than left out.
"""
import glob
import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
D = SP / "a549"
WINDOW = int(os.environ.get("LC_WINDOW", 250_000))
PROM_EXCL = 5000                 # a "distal" enhancer must be at least this far from the TSS
MIN_TPM = float(os.environ.get("LC_MINTPM", 1.0))


def load_rna():
    """TPM per timepoint plus gene coordinates. Handles BOTH formats present in this series.

    ENCODE mixes quantification pipelines here: 15 of the 16 files are featureCounts
    (Geneid/Chr/Start/End/Strand/Length/count) and one is RSEM
    (gene_id/transcript_id(s)/length/effective_length/expected_count/TPM/...). Assuming one format made the
    parser read an RSEM effective_length as a genomic Start and abort. Coordinates come only from the
    featureCounts files, since RSEM carries none.
    """
    out, coords = {}, {}
    for fn in sorted(glob.glob(str(D / "rna_t*.tsv"))):
        t = float(re.search(r"rna_t([\d.]+)\.tsv", fn).group(1))
        hdr, rows = None, []
        with open(fn) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if hdr is None:
                    hdr = f
                    continue
                rows.append(f)
        if not hdr:
            continue
        if hdr[0] == "Geneid":                      # featureCounts
            gid, cnt, ln = [], [], []
            for f in rows:
                try:
                    c, L = float(f[6]), max(float(f[5]), 1.0)
                except (IndexError, ValueError):
                    continue
                g = f[0].split(".")[0]
                gid.append(g); cnt.append(c); ln.append(L)
                if g not in coords:
                    try:
                        ch = f[1].split(";")[0]
                        st = min(int(x) for x in f[2].split(";"))
                        en = max(int(x) for x in f[3].split(";"))
                        strand = f[4].split(";")[0]
                        coords[g] = (ch, en if strand == "-" else st)
                    except (IndexError, ValueError):
                        pass
            rate = np.array(cnt) / np.array(ln)
            out[t] = dict(zip(gid, 1e6 * rate / max(rate.sum(), 1e-9)))
        elif "TPM" in hdr:                          # RSEM
            j = hdr.index("TPM")
            d = {}
            for f in rows:
                try:
                    d[f[0].split(".")[0]] = float(f[j])
                except (IndexError, ValueError):
                    continue
            out[t] = d
        else:
            print(f"    skipping {Path(fn).name}: unrecognised header {hdr[:4]}")
    return out, coords


def load_peaks(pattern):
    """Per-chromosome sorted (starts, ends, signal) ARRAYS.

    Built once as arrays, not returned as tuple lists. The first version returned lists and every sig_at call
    rebuilt a numpy array of all starts on the chromosome -- O(peaks) per lookup across ~330k lookups, with
    1.3M GR peaks loaded, which made the run effectively non-terminating.
    """
    by = {}
    for fn in sorted(glob.glob(str(D / pattern))):
        with gzip.open(fn, "rt") as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) < 7:
                    continue
                try:
                    by.setdefault(f[0], []).append((int(f[1]), int(f[2]), float(f[6])))
                except ValueError:
                    continue
    out = {}
    for c, v in by.items():
        v.sort()
        out[c] = (np.array([x[0] for x in v]), np.array([x[1] for x in v]),
                  np.array([x[2] for x in v]))
    return out


def load_dnase():
    out = {}
    for fn in sorted(glob.glob(str(D / "dnase_t*.bed.gz"))):
        t = float(re.search(r"dnase_t([\d.]+)\.bed", fn).group(1))
        out[t] = load_peaks(Path(fn).name)
    return out


def sig_at(pk, ch, lo, hi):
    """Summed signal of peaks overlapping [lo, hi], vectorised over a bounded slice."""
    if ch not in pk:
        return 0.0
    st, en, sg = pk[ch]
    i = int(np.searchsorted(st, hi))
    if i == 0:
        return 0.0
    j = max(0, i - 400)
    m = en[j:i] >= lo
    return float(sg[j:i][m].sum()) if m.any() else 0.0


def overlaps(pk, ch, lo, hi):
    """Is there any peak overlapping [lo, hi]."""
    if ch not in pk:
        return False
    st, en, _ = pk[ch]
    i = int(np.searchsorted(st, hi))
    if i == 0:
        return False
    j = max(0, i - 400)
    return bool((en[j:i] >= lo).any())


def main():
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    print("=" * 100)
    print("REAL-TIME LAG CURVE -- A549 + dexamethasone, does accessibility lead transcription?")
    print("=" * 100)
    if not D.exists():
        raise SystemExit(f"A549 data absent at {D}")
    rna, coords = load_rna()
    dn = load_dnase()
    shared = sorted(set(rna) & set(dn))
    print(f"  RNA timepoints {sorted(rna)}")
    print(f"  DNase timepoints {sorted(dn)}")
    print(f"  SHARED: {shared}  ({len(shared)} points)")
    if len(shared) < 5:
        raise SystemExit("fewer than 5 shared timepoints; a lag curve is not estimable")

    gr = load_peaks("gr_*.bed.gz")
    ngr = sum(len(v[0]) for v in gr.values())
    print(f"  GR (NR3C1) ChIP peaks: {ngr:,} over {len(gr)} contigs")

    # candidate enhancers per gene: GR-bound (prespecified LINKED) and a distance-matched non-GR peak
    t0 = shared[0]
    base_pk = dn[t0]
    genes, link, rnd = [], [], []
    for g, (ch, tss) in coords.items():
        if g not in rna[t0] or rna[t0][g] < MIN_TPM:
            continue
        if ch not in base_pk:
            continue
        bst, ben, bsg = base_pk[ch]
        mid = (bst + ben) // 2
        d_all = np.abs(mid - tss)
        sel = np.where((d_all <= WINDOW) & (d_all > PROM_EXCL))[0]
        if len(sel) < 2:
            continue
        near = [(int(bst[k]), int(ben[k]), float(bsg[k])) for k in sel]
        bound = [(a, b) for a, b, v in near if overlaps(gr, ch, a, b)]
        if not bound:
            continue
        la, lb = bound[0]
        d = abs((la + lb) // 2 - tss)
        pool = [(a, b) for a, b, v in near
                if not overlaps(gr, ch, a, b) and 0.5 * d <= abs((a + b) // 2 - tss) <= 2 * d]
        if not pool:
            continue
        genes.append((g, ch, tss))
        link.append((la, lb))
        rnd.append(pool[len(pool) // 2])
    print(f"  {len(genes):,} genes with a GR-bound distal peak, a distance-matched non-GR peak, "
          f"and baseline TPM >= {MIN_TPM}")
    if len(genes) < 100:
        raise SystemExit(f"only {len(genes)} usable genes; the prespecified set is too small to test")

    # per-timepoint matrices
    T = len(shared)
    G = len(genes)
    Rm = np.full((G, T), np.nan)
    Al = np.zeros((G, T))
    Ar = np.zeros((G, T))
    for ti, t in enumerate(shared):
        pk = dn[t]
        for gi, (g, ch, tss) in enumerate(genes):
            Rm[gi, ti] = rna[t].get(g, np.nan)
            a, b = link[gi]
            Al[gi, ti] = sig_at(pk, ch, a, b)
            a, b = rnd[gi]
            Ar[gi, ti] = sig_at(pk, ch, a, b)
    LR = np.log1p(Rm)
    LL, LRnd = np.log1p(Al), np.log1p(Ar)
    chrom = np.array([c for _, c, _ in genes])
    ok = np.isfinite(LR).all(1)
    print(f"  {int(ok.sum()):,}/{G:,} genes with complete RNA across all {T} timepoints")
    LR, LL, LRnd, chrom = LR[ok], LL[ok], LRnd[ok], chrom[ok]

    def loco_r2(X, y, ch):
        sc = []
        for c in sorted(set(ch)):
            te = ch == c
            if te.sum() < 20 or (~te).sum() < 100:
                continue
            mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-9
            m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit((X[~te] - mu) / sd, y[~te])
            sc.append(r2_score(y[te], m.predict((X[te] - mu) / sd)))
        return float(np.mean(sc)) if sc else np.nan

    # contemporaneous: does ATAC(t) explain RNA(t) at all? (the state-marker arm)
    Xc, yc, chc = [], [], []
    for ti in range(T):
        Xc.append(np.column_stack([LL[:, ti], LRnd[:, ti]]))
        yc.append(LR[:, ti]); chc.append(chrom)
    Xc, yc, chc = np.vstack(Xc), np.concatenate(yc), np.concatenate(chc)
    r2_none = loco_r2(np.zeros((len(yc), 1)), yc, chc)
    r2_cont = loco_r2(Xc[:, :1], yc, chc)
    d_cont = r2_cont - r2_none
    print(f"\n  CONTEMPORANEOUS: ATAC(t) -> RNA(t)   R2 {r2_cont:+.4f} vs intercept {r2_none:+.4f}"
          f"   dR2 {d_cont:+.4f}")
    print(f"    ^ if this is large while the lagged gains below are ~0, accessibility is a STATE MARKER")

    lags = sorted({round(shared[j] - shared[i], 2)
                   for i in range(T) for j in range(i + 1, T)})
    print(f"\n  LAG CURVE over tau = {lags}")
    print(f"    {'tau(h)':>7s} {'n':>7s} {'AR':>8s} {'+link':>8s} {'+rand':>8s} "
          f"{'link-rand':>10s} {'permuted':>9s} {'fcst adv':>9s}")
    rng = np.random.default_rng(0)
    curve = {}
    for tau in lags:
        rows_x, rows_y, rows_c, rows_xp = [], [], [], []
        for i in range(T):
            j = next((k for k in range(T) if abs((shared[k] - shared[i]) - tau) < 1e-6), None)
            if j is None:
                continue
            rows_x.append(np.column_stack([LR[:, i], LL[:, i], LRnd[:, i]]))
            rows_y.append(LR[:, j] - LR[:, i])
            rows_c.append(chrom)
            # permuted time: ATAC from a RANDOM timepoint, so the ordering is destroyed and nothing else is
            p = int(rng.integers(T))
            rows_xp.append(np.column_stack([LR[:, i], LL[:, p], LRnd[:, p]]))
        if not rows_x:
            continue
        X = np.vstack(rows_x); y = np.concatenate(rows_y); cc = np.concatenate(rows_c)
        Xp = np.vstack(rows_xp)
        ar = loco_r2(X[:, :1], y, cc)
        lk = loco_r2(X[:, :2], y, cc)
        rd = loco_r2(X[:, [0, 2]], y, cc)
        pm = loco_r2(Xp[:, :2], y, cc)
        adv = (lk - ar) - d_cont
        print(f"    {tau:7.2f} {len(y):7d} {ar:8.4f} {lk-ar:+8.4f} {rd-ar:+8.4f} "
              f"{lk-rd:+10.4f} {pm-ar:+9.4f} {adv:+9.4f}")
        curve[str(tau)] = {"n": int(len(y)), "ar": ar, "link_gain": lk - ar, "rand_gain": rd - ar,
                           "link_minus_rand": lk - rd, "permuted_gain": pm - ar, "forecast_adv": adv}

    # direction, at the lag with the largest link gain
    best_tau = max(curve, key=lambda k: curve[k]["link_gain"]) if curve else None
    R = {"shared_timepoints": shared, "n_genes": int(len(LR)), "gr_peaks": int(ngr),
         "contemporaneous_gain": d_cont, "curve": curve, "best_tau": best_tau,
         "window": WINDOW, "min_tpm": MIN_TPM,
         "caveat": "one replicate per timepoint in most of this series; no replicate variance estimable"}
    if best_tau:
        tau = float(best_tau)
        fx, fy, rx, ry, cc2 = [], [], [], [], []
        for i in range(T):
            j = next((k for k in range(T) if abs((shared[k] - shared[i]) - tau) < 1e-6), None)
            if j is None:
                continue
            fx.append(np.column_stack([LR[:, i], LL[:, i]])); fy.append(LR[:, j] - LR[:, i])
            rx.append(np.column_stack([LL[:, i], LR[:, i]])); ry.append(LL[:, j] - LL[:, i])
            cc2.append(chrom)
        cc2 = np.concatenate(cc2)
        f_gain = loco_r2(np.vstack(fx), np.concatenate(fy), cc2) - \
            loco_r2(np.vstack(fx)[:, :1], np.concatenate(fy), cc2)
        r_gain = loco_r2(np.vstack(rx), np.concatenate(ry), cc2) - \
            loco_r2(np.vstack(rx)[:, :1], np.concatenate(ry), cc2)
        print(f"\n  DIRECTION at tau={tau}h")
        print(f"    ATAC(t) adds to dRNA : {f_gain:+.4f}")
        print(f"    RNA(t)  adds to dATAC: {r_gain:+.4f}")
        print(f"    asymmetry            : {f_gain-r_gain:+.4f}")
        R["direction"] = {"tau": tau, "forward": f_gain, "reverse": r_gain,
                          "asymmetry": f_gain - r_gain}

    # ---- verdict ----
    print("\n" + "=" * 100)
    if not curve:
        v = "INCONCLUSIVE -- no lag had usable timepoint pairs."
    else:
        gains = np.array([c["link_gain"] for c in curve.values()])
        lmr = np.array([c["link_minus_rand"] for c in curve.values()])
        perm = np.array([c["permuted_gain"] for c in curve.values()])
        best = curve[best_tau]
        peaked = best["link_gain"] > 0.01 and best["link_gain"] > 3 * max(np.median(gains), 1e-6)
        specific = best["link_minus_rand"] > 0.005
        beats_perm = best["link_gain"] - best["permuted_gain"] > 0.005
        asym = R.get("direction", {}).get("asymmetry", 0.0)
        if peaked and specific and beats_perm and asym > 0.005:
            v = (f"ACCESSIBILITY LEADS TRANSCRIPTION, at tau={best_tau} h. Gain {best['link_gain']:+.4f} over "
                 f"the autoregressive floor, {best['link_minus_rand']:+.4f} over a distance-matched non-GR "
                 f"peak, {best['link_gain']-best['permuted_gain']:+.4f} over permuted time, asymmetry "
                 f"{asym:+.4f}, and the curve is PEAKED rather than flat. Pseudotime could not resolve this; "
                 f"real elapsed time can.")
        elif specific and beats_perm:
            v = (f"WEAK, SPECIFIC SIGNAL at tau={best_tau} h: {best['link_minus_rand']:+.4f} over a matched "
                 f"non-GR peak and {best['link_gain']-best['permuted_gain']:+.4f} over permuted time, but the "
                 f"lag curve is not clearly peaked (best {best['link_gain']:+.4f} vs median "
                 f"{np.median(gains):+.4f}), so a specific temporal window is not established.")
        elif d_cont > 0.02 and max(gains) < 0.01:
            v = (f"ACCESSIBILITY IS A STATE MARKER, NOT A FORECASTER. It explains CURRENT expression "
                 f"({d_cont:+.4f} dR2 contemporaneous) but adds at most {max(gains):+.4f} to future change at "
                 f"ANY lag, with forecast advantage {best['forecast_adv']:+.4f}. This is the distinction the "
                 f"pseudotime design could not draw, and it is the same conclusion by a sharper route: "
                 f"accessibility marks where regulation is possible, not when it happens.")
        else:
            v = (f"NULL ACROSS ALL LAGS. Best gain {best['link_gain']:+.4f} at tau={best_tau} h, "
                 f"link-minus-random {best['link_minus_rand']:+.4f}, permuted-time gain "
                 f"{best['permuted_gain']:+.4f}, contemporaneous {d_cont:+.4f}. Real elapsed time, GR-bound "
                 f"prespecified links and a full lag curve reproduce the pseudotime null, which removes "
                 f"temporal resolution and link quality as explanations for it.")
    R["verdict"] = v
    print(f"  VERDICT: {v}")
    print(f"\n  CAVEAT: {R['caveat']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "lagcurve.json", "w"), indent=1, default=float)
    print(f"  -> {OUT/'lagcurve.json'}")


if __name__ == "__main__":
    main()
