"""WHY DID STAGE 1 FAIL? Two candidate explanations, one of which is a defect in Stage 1 itself.

STAGE 1 FOUND: accessibility adds +0.0005 R2 to a forecast of expression CHANGE, a distance-matched random
peak adds the same (-0.0000 difference), and the forward/reverse asymmetry is +0.0002. Read at face value that
says accessibility does not lead expression. Before that is believed, two things have to be ruled out, because
the static version of the same question WORKS: activity features add +0.0390 pooled AUPRC to enhancer-gene
prediction and the model beats the distance floor at R@1 0.697 vs 0.651. Accessibility demonstrably carries
enhancer-gene information. So either the dynamic question is genuinely different, or Stage 1 was built wrong.

DEFECT 1 -- STAGE 1'S "LINKED ENHANCER" WAS NOT A LINK. It was chosen as the most VARIABLE distal peak within
250 kb of the TSS. That is an arbitrary peak selected by variance, not a validated regulatory link. So the
headline control -- linked versus distance-matched random -- was comparing one arbitrary peak against another
arbitrary peak, and was guaranteed to return zero. It did (-0.0000), and that number says nothing about real
enhancer-gene pairs. This module uses the CRISPR-VALIDATED positives from the benchmark instead: 820 pairs
where perturbing the element measurably changed the gene. K562 is erythroleukaemic and the multiome is bone
marrow, so the lineage is related but not identical, and that is stated rather than glossed.

DEFECT 2 -- THE NOISE FLOOR WAS NEVER MEASURED. The delta target has sd 0.083 in log space over pseudobulk
means. If splitting the SAME pseudotime bin into two random halves produces a delta of comparable size, then
almost all of the between-bin delta is sampling noise, the achievable R2 is near zero for ANY feature, and
+0.0005 is a power failure rather than a biological result. The split-half delta is exactly measurable:

    noise_sd    = sd of [ log RNA(half A of bin) - log RNA(half B of bin) ]
    signal_sd   = sd of [ log RNA(bin t+1) - log RNA(bin t) ]
    reliability = 1 - noise_var / signal_var        <- the ceiling on any R2 for this target

If reliability is near zero the experiment had no power and Stage 1's verdict must be withdrawn, not
softened. If it is high, the null stands and the failure is biological.

THESE ARE OPPOSITE CONCLUSIONS FROM THE SAME NUMBERS, which is why the diagnostic comes before any
interpretation.
"""
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
H5 = SP / "multiome" / "bmmc.h5ad"
NBIN = int(os.environ.get("DD_NBIN", 10))
MIN_CELLS_BIN = int(os.environ.get("DD_MINCELLS", 120))    # needs to survive being halved
PEAK_RE = re.compile(r"^(.+)-(\d+)-(\d+)$")


def cat(o, k):
    import h5py
    g = o[k]
    if isinstance(g, h5py.Group):
        cs = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return np.array([cs[c] if 0 <= c < len(cs) else "?" for c in g["codes"][:]])
    a = g[:]
    if a.dtype.kind in "SO":
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])
    return a


def pseudobulk(split_halves=False):
    """Per (donor, bin) mean profiles. With split_halves, each bin yields TWO independent half-profiles
    from disjoint random cell subsets, which is what makes the noise floor measurable."""
    import h5py
    with h5py.File(H5, "r") as h:
        v = h["var"]
        ik = v.attrs.get("_index", "_index")
        ik = ik.decode() if isinstance(ik, bytes) else ik
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in v[ik][:]]
        is_peak = np.array([bool(PEAK_RE.match(n)) for n in names])
        o = h["obs"]
        donor = cat(o, "DonorID").astype(str)
        pt = cat(o, "ATAC_pseudotime_order").astype(float)
        X = h["X"]
        shape = tuple(X.attrs["shape"])
        indptr = X["indptr"][:]
        data, indices = X["data"], X["indices"]

        rng = np.random.default_rng(0)
        lab = np.full(shape[0], -1, np.int64)
        keys = []
        ok = np.isfinite(pt)
        for dn in sorted(set(donor)):
            m = ok & (donor == dn)
            if m.sum() < NBIN * MIN_CELLS_BIN:
                continue
            q = np.quantile(pt[m], np.linspace(0, 1, NBIN + 1))
            b = np.clip(np.digitize(pt[m], q[1:-1]), 0, NBIN - 1)
            idx = np.where(m)[0]
            for bb in range(NBIN):
                sel = idx[b == bb]
                if len(sel) < MIN_CELLS_BIN:
                    continue
                if split_halves:
                    p = rng.permutation(len(sel))
                    half = len(sel) // 2
                    for hh, part in enumerate((sel[p[:half]], sel[p[half:2 * half]])):
                        lab[part] = len(keys)
                        keys.append((dn, bb, hh))
                else:
                    lab[sel] = len(keys)
                    keys.append((dn, bb, 0))
        ng = len(keys)
        P = np.zeros((ng, shape[1]))
        cnt = np.zeros(ng)
        CH = 4000
        for s in range(0, shape[0], CH):
            e = min(s + CH, shape[0])
            a, b2 = int(indptr[s]), int(indptr[e])
            if b2 <= a:
                continue
            rows = np.repeat(np.arange(s, e), np.diff(indptr[s:e + 1]).astype(np.int64))
            gl = lab[rows]
            keep = gl >= 0
            if keep.any():
                np.add.at(P, (gl[keep], indices[a:b2][keep]), data[a:b2][keep])
            u, c = np.unique(lab[s:e], return_counts=True)
            for g_, c_ in zip(u, c):
                if g_ >= 0:
                    cnt[g_] += c_
        P = P / np.maximum(cnt[:, None], 1)
        return names, is_peak, P, keys


def crispr_links():
    """CRISPR-validated positive element-gene pairs from the benchmark: perturbing the element measurably
    changed the gene. This is what "linked" has to mean; Stage 1's most-variable-peak heuristic was not."""
    from ranking_gate import load
    rows = load()
    out = []
    for r in rows:
        if r.get("Significant") not in ("TRUE", "True", "true"):
            continue
        try:
            out.append((r["measuredGeneSymbol"], r["chrom"],
                        (int(r["chromStart"]) + int(r["chromEnd"])) // 2,
                        int(float(r["startTSS"])), r["CellType"]))
        except (ValueError, KeyError, TypeError):
            continue
    return out


def main():
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    print("=" * 100)
    print("WHY DID STAGE 1 FAIL? -- noise floor, and links that are actually links")
    print("=" * 100)
    if not H5.exists():
        raise SystemExit(f"multiome absent at {H5}")

    # ---- DEFECT 2: the noise floor, measured ----
    names, is_peak, Ph, keysh = pseudobulk(split_halves=True)
    gidx = np.where(~is_peak)[0]
    Lh = np.log1p(np.maximum(Ph, 0))
    kd = {k: i for i, k in enumerate(keysh)}
    noise, signal = [], []
    for (dn, bb, hh) in keysh:
        if hh != 0:
            continue
        a, b = kd.get((dn, bb, 0)), kd.get((dn, bb, 1))
        if a is None or b is None:
            continue
        noise.append(Lh[a, gidx] - Lh[b, gidx])            # same bin, disjoint cells -> pure noise
        nxt = kd.get((dn, bb + 1, 0))
        if nxt is not None:
            signal.append(Lh[nxt, gidx] - Lh[a, gidx])      # adjacent bins -> signal + noise
    noise = np.concatenate(noise) if noise else np.array([])
    signal = np.concatenate(signal) if signal else np.array([])
    nv, sv = float(np.var(noise)), float(np.var(signal))
    # a between-bin delta of two half-profiles carries two half-bins of noise; the same-bin delta measures
    # exactly that, so reliability = 1 - noise_var/signal_var is the ceiling on R2 for this target
    rel = 1.0 - nv / sv if sv > 0 else np.nan
    print(f"\n  NOISE FLOOR (split-half within the same pseudotime bin, {len(gidx):,} genes)")
    print(f"    same-bin delta      sd {np.sqrt(nv):.4f}   (pure sampling noise)")
    print(f"    adjacent-bin delta  sd {np.sqrt(sv):.4f}   (signal + noise)")
    print(f"    RELIABILITY = 1 - noise_var/signal_var = {rel:+.4f}")
    print(f"      ^ this is the CEILING on R2 for the delta target. Stage 1 measured +0.0005.")
    R = {"noise_sd": float(np.sqrt(nv)), "signal_sd": float(np.sqrt(sv)), "reliability": rel,
         "stage1_gain": 0.0005, "n_genes": int(len(gidx))}
    if rel < 0.1:
        print(f"    -> Stage 1 had essentially NO POWER: at most {100*max(rel,0):.1f}% of the delta is real "
              f"signal, so no feature could have scored well and its verdict must be WITHDRAWN, not softened.")
    else:
        print(f"    -> the delta is {100*rel:.0f}% real signal, so Stage 1 had power and its null stands.")

    # ---- DEFECT 1: links that are actually links ----
    names2, is_peak2, P, keys = pseudobulk(split_halves=False)
    L = np.log1p(np.maximum(P, 0))
    gi = {n: i for i, n in enumerate(names2) if not is_peak2[i]}
    peaks = []
    for i, n in enumerate(names2):
        if is_peak2[i]:
            m = PEAK_RE.match(n)
            peaks.append((m.group(1), (int(m.group(2)) + int(m.group(3))) // 2, i))
    bych = {}
    for c, mid, i in peaks:
        bych.setdefault(c, []).append((mid, i))
    for c in bych:
        bych[c].sort()

    links = crispr_links()
    print(f"\n  CRISPR-VALIDATED LINKS: {len(links)} positive pairs from the benchmark")
    rng = np.random.default_rng(1)
    recs = []
    matched, no_gene, no_peak = 0, 0, 0
    for g, ch, emid, tss, cell in links:
        ig = gi.get(g)
        if ig is None:
            no_gene += 1
            continue
        cands = bych.get(ch) or bych.get(ch.replace("chr", "")) or bych.get("chr" + ch)
        if not cands:
            no_peak += 1
            continue
        mids = np.array([m for m, _ in cands])
        j = int(np.argmin(np.abs(mids - emid)))
        if abs(mids[j] - emid) > 5000:                 # the validated element must actually be a peak here
            no_peak += 1
            continue
        link_i = cands[j][1]
        d = abs(emid - tss)
        pool = [cands[k][1] for k in range(len(cands))
                if cands[k][1] != link_i and 0.5 * d <= abs(mids[k] - tss) <= 2 * d]
        if not pool:
            continue
        rand_i = pool[int(rng.integers(len(pool)))]
        matched += 1
        for t in range(NBIN - 1):
            for dn in sorted({k[0] for k in keys}):
                a = next((i for i, k in enumerate(keys) if k == (dn, t, 0)), None)
                b = next((i for i, k in enumerate(keys) if k == (dn, t + 1, 0)), None)
                if a is None or b is None:
                    continue
                recs.append((dn, L[a, ig], L[a, link_i], L[a, rand_i], L[b, ig] - L[a, ig]))
    print(f"    matched to a multiome peak and gene: {matched}/{len(links)}  "
          f"(gene absent {no_gene}, element not a peak here {no_peak})")
    R["links_matched"] = matched
    if matched < 30 or not recs:
        R["verdict"] = (f"only {matched} CRISPR-validated links could be matched into the multiome; too few "
                        f"to retest the linked-vs-random control. Stage 1's control remains unusable rather "
                        f"than informative -- it compared two arbitrary peaks -- and no replacement is "
                        f"available from this dataset.")
        print(f"\n  {R['verdict']}")
    else:
        Dn = np.array([r[0] for r in recs])
        M = np.array([[r[1], r[2], r[3]] for r in recs])
        yv = np.array([r[4] for r in recs])
        print(f"    {len(recs):,} (validated link, t->t+1, donor) records")

        def loco(cols):
            sc = []
            for dn in sorted(set(Dn)):
                te = Dn == dn
                if te.sum() < 20 or (~te).sum() < 60:
                    continue
                A = M[:, cols]
                mu, sd = A[~te].mean(0), A[~te].std(0) + 1e-9
                m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit((A[~te] - mu) / sd, yv[~te])
                sc.append(r2_score(yv[te], m.predict((A[te] - mu) / sd)))
            return float(np.mean(sc)) if sc else np.nan
        ar = loco([0])
        lk = loco([0, 1])
        rd = loco([0, 2])
        print(f"\n    predicting delta log RNA on VALIDATED links, held out by donor")
        print(f"      AR floor (rna_t)                {ar:+.4f}")
        print(f"      + validated enhancer ATAC_t     {lk:+.4f}   ({lk-ar:+.4f})")
        print(f"      + distance-matched random peak  {rd:+.4f}   ({rd-ar:+.4f})")
        print(f"      VALIDATED minus RANDOM          {lk-rd:+.4f}")
        R["validated"] = {"ar": ar, "linked": lk, "random": rd, "linked_minus_random": lk - rd}
        if rel < 0.1:
            R["verdict"] = (
                f"STAGE 1'S VERDICT IS WITHDRAWN, on the power argument. The split-half reliability of the "
                f"delta target is {rel:+.4f} -- at most {100*max(rel,0):.0f}% of the between-bin change is "
                f"real signal, so no feature could have scored meaningfully and the +0.0005 was a power "
                f"failure, not evidence about biology. Separately, Stage 1's linked-vs-random control was "
                f"invalid: it used the most-variable nearby peak rather than a link. Retested on "
                f"{matched} CRISPR-validated pairs, validated-minus-random is {lk-rd:+.4f}, which given the "
                f"reliability above is equally uninformative. The honest position is that this dataset "
                f"cannot answer the question, and a real stimulus time course is required.")
        elif lk - rd > 0.005:
            R["verdict"] = (
                f"STAGE 1 FAILED BECAUSE ITS LINKS WERE NOT LINKS. On {matched} CRISPR-validated pairs the "
                f"validated enhancer beats a distance-matched random peak by {lk-rd:+.4f}, where Stage 1's "
                f"most-variable-peak heuristic gave -0.0000. Reliability of the delta target is {rel:+.4f}, "
                f"so the measurement had power. Accessibility does carry gene-specific dynamic information; "
                f"Stage 1's control was the defect.")
        else:
            R["verdict"] = (
                f"THE NULL SURVIVES A PROPER CONTROL. Reliability {rel:+.4f} says the delta target has "
                f"power, and on {matched} CRISPR-validated pairs the validated enhancer beats a "
                f"distance-matched random peak by only {lk-rd:+.4f}. So Stage 1's conclusion holds for a "
                f"better reason than it was originally reached: even real, perturbation-validated enhancers "
                f"do not forecast their gene's expression change on this trajectory.")
        print(f"\n  VERDICT: {R['verdict']}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "dynamic_diagnose.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'dynamic_diagnose.json'}")


if __name__ == "__main__":
    main()
