"""MEASURED Hi-C against the 0.663 bar -- the thing the whole chromatin arc was trying to predict.

WHAT THIS SETTLES. Contact between an element and a promoter was attacked four ways in this project:
counting CTCF peaks between them (0.6632), an analytic Rouse-with-loops model (+0.005 on top of that), a
bending-stiffness sweep down to 200 bp/bead (+0.0005 at its most realistic), and full KMC extrusion plus
batched Langevin dynamics. Every one of those is an attempt to PREDICT a quantity that has been MEASURED
in K562 -- an ENCODE Tier 1 line -- and released publicly.

So: use the measurement.

WHY LOOP AND DOMAIN CALLS RATHER THAN THE CONTACT MATRIX. The .hic matrices for these experiments are
2-36 GB. The loop, contact-domain and stripe calls derived from them are 0.2-3 MB and are the features
the simulation was actually reaching for:

    a called loop connecting the two   the extrusion simulation's entire output, measured instead
    same contact domain (TAD)          what "insulation" means, measured instead
    domain boundaries between them     the measured analogue of ctcf_between, which carried +0.0514
    loop-anchor density at each end    how loop-engaged each locus is

INTACT Hi-C, NOT IN SITU. ENCSR479XDG (intact) yields 47,594 loops; ENCSR545YBD (in situ) yields 8,802 from
the same cell line. Both are loaded and compared, because "measured" is not one number -- assay depth
changes what is callable, and a feature that only works on the deep set is a different claim from one that
works on both.

THE CONTROL. Loop and domain positions are shuffled within their own chromosome, preserving count per
chromosome and the observed/expected distribution. That is the same control the CTCF-count feature had to
pass, and it asks the same question: is this feature marking a specific pair of loci, or merely marking
gene-dense, loop-rich territory?
"""
import bisect
import collections
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
INV = OUT / "invivo"
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
HIC = SP / "hic"
SEEDS = (0, 1, 2, 3, 4)
TOL = 25_000        # anchor match tolerance; loops are called at 1-10 kb so this admits a few bins

# ENCODE K562, GRCh38. Same assembly as the CRISPR benchmark, so no liftover.
LOOPS = {"intact": "ENCFF598CLH", "insitu": "ENCFF134HIZ"}
DOMAINS = {"intact": "ENCFF967NTN", "insitu": None}


def read_bedpe(acc):
    """chrom -> arrays of (anchor1 mid, anchor2 mid, observed, enrichment). Intra-chromosomal only."""
    d = collections.defaultdict(list)
    if acc is None:
        return d
    p = HIC / f"{acc}.bedpe.gz"
    if not p.exists():
        return d
    hdr = None
    with gzip.open(p, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                if hdr is None:
                    hdr = line.lstrip("#").rstrip("\n").split("\t")
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 6 or f[0] != f[3]:
                continue
            try:
                x = (int(f[1]) + int(f[2])) // 2
                y = (int(f[4]) + int(f[5])) // 2
            except ValueError:
                continue
            obs, enr = 1.0, 1.0
            if hdr and "observed" in hdr:
                i = hdr.index("observed")
                j = hdr.index("expectedDonut") if "expectedDonut" in hdr else None
                try:
                    obs = float(f[i])
                    if j is not None and float(f[j]) > 0:
                        enr = obs / float(f[j])
                except (ValueError, IndexError):
                    pass
            d[f[0]].append((min(x, y), max(x, y), obs, enr))
    return {k: np.array(v, float) for k, v in d.items() if v}


def read_domains(acc):
    """chrom -> sorted (start, end) of contact domains."""
    d = collections.defaultdict(list)
    if acc is None:
        return d
    p = HIC / f"{acc}.bedpe.gz"
    if not p.exists():
        return d
    with gzip.open(p, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 6 or f[0] != f[3]:
                continue
            try:
                d[f[0]].append((int(f[1]), int(f[5])))
            except ValueError:
                continue
    return {k: np.array(sorted(v), float) for k, v in d.items() if v}


def shuffle_within_chrom(arr, seed):
    """Relocate each interval, keeping its SPAN and its chromosome. Preserves intervals-per-chromosome,
    the span distribution and any score columns -- only the genomic position moves. A feature that
    survives this is marking specific loci, not loop-rich neighbourhoods.

    Handles both the 4-column loop arrays (lo, hi, observed, enrichment) and the 2-column domain arrays
    (start, end); an earlier version assumed 4 and raised IndexError on the domains.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for c, a in arr.items():
        span = a[:, 1] - a[:, 0]
        new_lo = rng.uniform(a[:, 0].min(), a[:, 0].max(), len(a))
        cols = [new_lo, new_lo + span] + [a[:, k] for k in range(2, a.shape[1])]
        out[c] = np.column_stack(cols)
    return out


def hic_features(rows, loops, domains):
    """Per pair: loop connection, loop strength, domain co-membership, boundary crossings, anchor density."""
    F = np.zeros((len(rows), 7))
    by_chrom = collections.defaultdict(list)
    for i, (c, e, t) in enumerate(rows):
        by_chrom[c].append(i)
    for c, idx in by_chrom.items():
        L = loops.get(c)
        D = domains.get(c)
        if L is not None and len(L):
            x1, x2, obs, enr = L[:, 0], L[:, 1], L[:, 2], L[:, 3]
        for i in idx:
            _, e, t = rows[i]
            lo, hi = (e, t) if e <= t else (t, e)
            if L is not None and len(L):
                # a loop CONNECTS the pair if one anchor is near each end (either orientation)
                m = ((np.abs(x1 - lo) < TOL) & (np.abs(x2 - hi) < TOL)) | \
                    ((np.abs(x1 - hi) < TOL) & (np.abs(x2 - lo) < TOL))
                if m.any():
                    F[i, 0] = 1.0
                    F[i, 1] = np.log1p(obs[m].max())
                    F[i, 2] = np.log1p(enr[m].max())
                # how loop-engaged is each end at all
                F[i, 3] = ((np.abs(x1 - lo) < TOL) | (np.abs(x2 - lo) < TOL)).sum()
                F[i, 4] = ((np.abs(x1 - hi) < TOL) | (np.abs(x2 - hi) < TOL)).sum()
            if D is not None and len(D):
                s, en = D[:, 0], D[:, 1]
                F[i, 5] = float(((s <= lo) & (en >= hi)).any())          # same contact domain
                # boundaries strictly between -- the measured analogue of ctcf_between
                F[i, 6] = (((s > lo) & (s < hi)) | ((en > lo) & (en < hi))).sum()
    return F


FEATNAMES = ["loop_connects", "loop_obs", "loop_enrich", "anchors_at_elem", "anchors_at_tss",
             "same_domain", "domain_bounds_between"]


def load_pairs():
    from contact_gate import load_ctcf, contact_features
    df = pd.read_csv(OUT / "crispr_features_compendium.csv")
    comp = json.load(open(INV / "compendium_tf.json"))
    et, tfl = comp["element_tfs"], comp["tf_list"]
    tss = {}
    with open(INV / "crispr_egpairs.tsv") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ix = {k: hdr.index(k) for k in ("chrom", "chromStart", "chromEnd", "startTSS", "measuredGeneSymbol")}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(hdr):
                continue
            try:
                tss[(f"{p[ix['chrom']]}:{p[ix['chromStart']]}-{p[ix['chromEnd']]}",
                     p[ix["measuredGeneSymbol"]])] = (
                    p[ix["chrom"]], (int(p[ix["chromStart"]]) + int(p[ix["chromEnd"]])) // 2,
                    int(p[ix["startTSS"]]))
            except ValueError:
                continue
    rows = [tss[(e, g)] for e, g in zip(df["element"], df["gene"])]
    assert len(rows) == len(df)
    Cr = contact_features(rows, load_ctcf())
    tfmat = np.zeros((len(df), len(tfl)), np.int8)
    for a, ek in enumerate(df["element"].values):
        for ti in et.get(ek, []):
            tfmat[a, ti] = 1
    base = ["log_dist", "atac_enh", "h3k27ac_enh", "polr2a_enh", "procap_enh",
            "promoter_atac", "promoter_polii", "gene_expr"]
    return df, rows, np.hstack([df[base].values, tfmat]), Cr


def main():
    from crispr_gate import _cv_auprc, _seeded_folds
    print("=" * 100)
    print("MEASURED Hi-C vs the 0.663 CTCF-COUNT bar")
    print("=" * 100)
    df, rows, Xid, Cr = load_pairs()
    y, ch = df["crispr_hit"].values, df["chromosome"].values
    print(f"  {len(y):,} pairs, {int(y.sum())} positives, base rate {y.mean():.4f}")

    R = {"base_rate": float(y.mean()), "n_pairs": int(len(y))}

    def sc(X, tag):
        v = np.array([_cv_auprc(X, y, _seeded_folds(ch, s)) for s in SEEDS])
        print(f"  {tag:44s} {v.mean():.4f}  [{' '.join(f'{x:.3f}' for x in v)}]")
        return v

    a = sc(Xid, "epi + TF identity")
    b = sc(np.hstack([Xid, Cr]), "+ CTCF count       [the 0.663 bar]")
    R["identity"], R["ctcf_bar"] = float(a.mean()), float(b.mean())

    for tag in ("intact", "insitu"):
        loops = read_bedpe(LOOPS[tag])
        doms = read_domains(DOMAINS[tag])
        nl = sum(len(v) for v in loops.values())
        nd = sum(len(v) for v in doms.values())
        print(f"\n  --- {tag} Hi-C: {nl:,} intra-chromosomal loops, {nd:,} contact domains ---")
        if not nl:
            print("      files absent, skipped")
            continue
        H = hic_features(rows, loops, doms)
        cov = {n: float((H[:, i] != 0).mean()) for i, n in enumerate(FEATNAMES)}
        print(f"      coverage: " + " ".join(f"{n}={cov[n]:.3f}" for n in FEATNAMES[:3]))
        print(f"      pairs with a connecting loop: {int(H[:,0].sum()):,} "
              f"({H[:,0].mean()*100:.1f}%) | hit rate among them "
              f"{y[H[:,0]>0].mean():.4f} vs {y[H[:,0]==0].mean():.4f} without")

        h = sc(np.hstack([Xid, H]), f"+ Hi-C {tag} only  (no CTCF count)")
        hb = sc(np.hstack([Xid, Cr, H]), f"+ Hi-C {tag} + CTCF count")
        # control: shuffled loop/domain positions, several draws
        ctrl = []
        for s in range(3):
            Hs = hic_features(rows, shuffle_within_chrom(loops, s),
                              shuffle_within_chrom(doms, s + 100) if nd else doms)
            ctrl.append(np.array([_cv_auprc(np.hstack([Xid, Cr, Hs]), y, _seeded_folds(ch, sd))
                                  for sd in SEEDS]) - b)
        ctrl = np.concatenate(ctrl)
        d = hb - b
        net = d.mean() - ctrl.mean()
        se = float(np.sqrt(d.var(ddof=1) / len(d) + ctrl.var(ddof=1) / len(ctrl)))
        print(f"      Hi-C ON TOP OF the bar : {d.mean():+.4f}  "
              f"[per-seed {' '.join(f'{x:+.3f}' for x in d)}]")
        print(f"      shuffled control       : {ctrl.mean():+.4f}  (3 draws x {len(SEEDS)} seeds, "
              f"sd {ctrl.std(ddof=1):.4f})")
        print(f"      NET OF CONTROL         : {net:+.4f} +/- {se:.4f}   z = {net/max(se,1e-9):+.2f}")
        R[tag] = {"n_loops": nl, "n_domains": nd, "hic_only": float(h.mean()),
                  "hic_plus_ctcf": float(hb.mean()), "delta_on_top": float(d.mean()),
                  "control": float(ctrl.mean()), "net": float(net), "se": se,
                  "z": float(net / max(se, 1e-9)), "coverage": cov,
                  "sign_consistent": bool(np.all(d > 0))}

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "hic_gate.json", "w"), indent=1)
    print(f"\n  -> {OUT/'hic_gate.json'}")


if __name__ == "__main__":
    main()
