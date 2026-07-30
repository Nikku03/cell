"""STAGE 1: DOES ACCESSIBILITY AT t PREDICT EXPRESSION AT t+1, BEYOND EXPRESSION AT t?

THE CLAIM BEING TESTED is the one that makes a model "dynamic" rather than another static association:
enhancer accessibility rises, and the gene follows. Stated as a prediction:

    ATAC_e(t), ATAC_promoter(t), RNA_g(t)  ->  RNA_g(t+1)

THREE CONTROLS, WITHOUT WHICH THIS IS UNINTERPRETABLE. Along any trajectory everything is smooth, so a model
of future expression will look excellent while learning nothing about regulation:

 1 THE AUTOREGRESSIVE FLOOR. RNA_g(t) predicts RNA_g(t+1) nearly perfectly. Every number here is a gain over
   that floor, never over zero. Reporting R2 against zero would make a useless model look strong.
 2 LINKED vs DISTANCE-MATCHED RANDOM ENHANCER. If the linked enhancer's accessibility helps no more than a
   random peak the same distance away, the "gain" is regional co-accessibility, not a regulatory link. This
   is the same discipline as the degree-matched controls used elsewhere in this project.
 3 DIRECTION. If ATAC(t) -> RNA(t+1) is no better than RNA(t) -> ATAC(t+1), no temporal ordering has been
   established and "accessibility precedes expression" is unsupported. A symmetric result means both are
   driven by a third thing, which is the null this test exists to exclude.

PSEUDOTIME IS NOT TIME, AND THE ORDERING IS CHOSEN TO AVOID CIRCULARITY. These are 69,249 cells from a single
snapshot of haematopoietic differentiation, so t is a position along a trajectory, not a clock. Worse, the
dataset ships GEX_pseudotime_order, which is DERIVED FROM EXPRESSION -- ordering cells by it and then
predicting expression at the next position is partly definitional. So ATAC_pseudotime_order is used for the
ordering, which is derived from the other modality. The GEX ordering is run as a labelled comparison, because
the size of the difference between them is itself informative.

THE DATA. NeurIPS 2021 BMMC multiome (GSE194122): 69,249 cells, 13,431 genes and 116,490 ATAC peaks measured
in THE SAME cells, 22 cell types, 10 donors, 4 sites. Same-cell RNA and ATAC is what makes the pairing real
rather than a cross-dataset join.

HELD OUT BY DONOR. Splitting cells at random would put the same donor on both sides and the model could learn
donor-specific accessibility. Donors are the exchangeable unit.
"""
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
NBIN = int(os.environ.get("DS_NBIN", 20))          # pseudotime bins per donor
WINDOW = int(os.environ.get("DS_WINDOW", 250_000))  # bp around the TSS for candidate peaks
MIN_CELLS_BIN = 50
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


def load():
    """Pseudobulk ATAC and RNA per (donor, pseudotime bin), plus the feature index split.

    Pseudobulked rather than per-cell: single-cell ATAC is far too sparse for a per-cell peak-level
    regression, and the question is about trajectory-level dynamics anyway. Summed over a CSR row block at a
    time so the 69,249 x 129,921 matrix is never held dense.
    """
    import h5py
    with h5py.File(H5, "r") as h:
        v = h["var"]
        ik = v.attrs.get("_index", "_index")
        ik = ik.decode() if isinstance(ik, bytes) else ik
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in v[ik][:]]
        # split by NAME pattern, not by the feature_types codes: the codes are 0/1 with no stored mapping,
        # so trusting them would be a guess. A peak is "chrom-start-end"; a gene symbol is not.
        is_peak = np.array([bool(PEAK_RE.match(n)) for n in names])
        print(f"  {len(names):,} features -> {int((~is_peak).sum()):,} genes, {int(is_peak.sum()):,} peaks")
        o = h["obs"]
        donor = cat(o, "DonorID").astype(str)
        pt_atac = cat(o, "ATAC_pseudotime_order").astype(float)
        pt_gex = cat(o, "GEX_pseudotime_order").astype(float)
        ctype = cat(o, "cell_type").astype(str)
        X = h["X"]
        shape = tuple(X.attrs["shape"])
        indptr = X["indptr"][:]
        data, indices = X["data"], X["indices"]

        out = {}
        for tag, pt in (("atac_order", pt_atac), ("gex_order", pt_gex)):
            ok = np.isfinite(pt)
            lab = np.full(shape[0], -1, np.int64)
            keys = []
            for dn in sorted(set(donor)):
                m = ok & (donor == dn)
                if m.sum() < NBIN * MIN_CELLS_BIN:
                    continue
                q = np.quantile(pt[m], np.linspace(0, 1, NBIN + 1))
                b = np.clip(np.digitize(pt[m], q[1:-1]), 0, NBIN - 1)
                idx = np.where(m)[0]
                for bb in range(NBIN):
                    sel = idx[b == bb]
                    if len(sel) >= MIN_CELLS_BIN:
                        lab[sel] = len(keys)
                        keys.append((dn, bb))
            ng = len(keys)
            P = np.zeros((ng, shape[1]))
            cnt = np.zeros(ng)
            CH = 4000
            for s in range(0, shape[0], CH):
                e = min(s + CH, shape[0])
                a, b2 = int(indptr[s]), int(indptr[e])
                if b2 <= a:
                    continue
                dd = data[a:b2]
                ii = indices[a:b2]
                rows = np.repeat(np.arange(s, e), np.diff(indptr[s:e + 1]).astype(np.int64))
                gl = lab[rows]
                keep = gl >= 0
                if not keep.any():
                    continue
                np.add.at(P, (gl[keep], ii[keep]), dd[keep])
                for g_ in np.unique(lab[s:e]):
                    if g_ >= 0:
                        cnt[g_] += (lab[s:e] == g_).sum()
            P = P / np.maximum(cnt[:, None], 1)          # mean per cell, so bin size does not leak in
            out[tag] = {"P": P, "keys": keys}
            print(f"  {tag}: {ng} (donor, bin) groups over {len(set(k[0] for k in keys))} donors")
        return names, is_peak, out, ctype


def tss_map(names, is_peak):
    """Gene TSS from the peak coordinate system is not available, so genes are located by the midpoint of
    the peaks that carry their own name -- not possible here. Instead a GENCODE-free fallback is used: the
    gene's position is taken from the project's own network annotation, which stores chrom and tss per gene."""
    import gzip
    net = SP / "cell_complete.json.gz"
    if not net.exists():
        raise SystemExit(f"{net} absent; needed for gene TSS coordinates")
    d = json.load(gzip.open(net, "rt"))
    pos = {}
    for g in d["genes"]:
        try:
            pos[g["name"]] = (g["chrom"], int(g["tss"]))
        except (KeyError, ValueError, TypeError):
            continue
    del d
    return pos


def main():
    from scipy import stats
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    print("=" * 100)
    print("STAGE 1 -- does accessibility at t predict expression at t+1, beyond expression at t?")
    print("=" * 100)
    if not H5.exists():
        raise SystemExit(f"multiome absent at {H5}")
    names, is_peak, pb, ctype = load()
    pos = tss_map(names, is_peak)

    gi = {n: i for i, n in enumerate(names) if not is_peak[i]}
    peaks = []
    for i, n in enumerate(names):
        if not is_peak[i]:
            continue
        m = PEAK_RE.match(n)
        peaks.append((m.group(1), (int(m.group(2)) + int(m.group(3))) // 2, i))
    bych = {}
    for c, mid, i in peaks:
        bych.setdefault(c, []).append((mid, i))
    for c in bych:
        bych[c].sort()
    print(f"  peaks on {len(bych)} contigs; gene coordinates known for "
          f"{sum(1 for g in gi if g in pos):,}/{len(gi):,} measured genes")

    R = {"nbin": NBIN, "window": WINDOW, "results": {}}
    rng = np.random.default_rng(0)

    for order_tag in ("atac_order", "gex_order"):
        P, keys = pb[order_tag]["P"], pb[order_tag]["keys"]
        donors = np.array([k[0] for k in keys])
        bins = np.array([k[1] for k in keys])
        print("\n" + "-" * 100)
        note = ("ordering from ATAC, independent of the target modality" if order_tag == "atac_order"
                else "ordering from GEX -- DERIVED FROM THE TARGET, so partly definitional")
        print(f"  ORDER: {order_tag}   ({note})")

        # build (gene, t) -> (t+1) records
        rows_g, Xf, Yf, Dn = [], [], [], []
        genes_used = 0
        for g, ig in gi.items():
            if g not in pos:
                continue
            ch, tss = pos[g]
            cands = bych.get(ch) or bych.get(ch.replace("chr", "")) or bych.get("chr" + ch)
            if not cands:
                continue
            mids = np.array([m for m, _ in cands])
            lo, hi = np.searchsorted(mids, tss - WINDOW), np.searchsorted(mids, tss + WINDOW)
            near = [cands[j][1] for j in range(lo, hi)]
            if len(near) < 3:
                continue
            # promoter peak = nearest to TSS; linked enhancer = most variable distal peak in window
            dists = np.abs(mids[lo:hi] - tss)
            prom = near[int(np.argmin(dists))]
            distal = [(near[j], dists[j]) for j in range(len(near)) if dists[j] > 5000]
            if not distal:
                continue
            var = [P[:, p].std() for p, _ in distal]
            link, link_d = distal[int(np.argmax(var))]
            # distance-matched random peak: another distal peak at a similar distance, different location
            pool = [(p, dd) for p, dd in distal if p != link and 0.5 * link_d <= dd <= 2 * link_d]
            if not pool:
                continue
            randp = pool[int(rng.integers(len(pool)))][0]
            genes_used += 1
            for t in range(NBIN - 1):
                m0 = (bins == t)
                m1 = (bins == t + 1)
                for dn in set(donors[m0]) & set(donors[m1]):
                    i0 = np.where(m0 & (donors == dn))[0][0]
                    i1 = np.where(m1 & (donors == dn))[0][0]
                    rows_g.append(g)
                    Dn.append(dn)
                    Xf.append([P[i0, ig], P[i0, prom], P[i0, link], P[i0, randp],
                               P[i1, ig], P[i1, link]])
                    Yf.append([P[i1, ig], P[i1, link]])
        Xf, Yf = np.array(Xf), np.array(Yf)
        Dn = np.array(Dn)
        print(f"  {genes_used:,} genes usable, {len(Xf):,} (gene, t->t+1, donor) records")
        if len(Xf) < 500:
            print("    too few records; order skipped")
            continue

        # log1p everything: pseudobulk means are heavily skewed
        L = np.log1p(np.maximum(Xf, 0))
        Ly = np.log1p(np.maximum(Yf, 0))
        fn = ["rna_t", "atac_prom_t", "atac_link_t", "atac_rand_t", "rna_t1", "atac_link_t1"]

        def loco_r2(cols, target):
            """Held out by DONOR: random cell splits would put a donor on both sides."""
            sc = []
            for dn in sorted(set(Dn)):
                te = Dn == dn
                if te.sum() < 30 or (~te).sum() < 100:
                    continue
                A, b = L[:, cols], target
                mu, sd = A[~te].mean(0), A[~te].std(0) + 1e-9
                m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit((A[~te] - mu) / sd, b[~te])
                sc.append(r2_score(b[te], m.predict((A[te] - mu) / sd)))
            return float(np.mean(sc)) if sc else np.nan

        y_next_rna = Ly[:, 0]
        ARMS = {
            "AR floor: rna_t": [0],
            "+ promoter ATAC_t": [0, 1],
            "+ LINKED enhancer ATAC_t": [0, 1, 2],
            "CTRL + RANDOM dist-matched peak": [0, 1, 3],
            "+ linked AND random": [0, 1, 2, 3],
        }
        print(f"\n    predicting RNA_g(t+1), held out by donor")
        print(f"      {'arm':34s} {'R2':>8s} {'gain over AR':>13s}")
        base = None
        rr = {}
        for tag, cols in ARMS.items():
            r2 = loco_r2(cols, y_next_rna)
            if base is None:
                base = r2
            print(f"      {tag:34s} {r2:8.4f} {r2-base:+13.4f}")
            rr[tag] = {"r2": r2, "gain_over_ar": r2 - base}

        # DIRECTION: same machinery, swapped roles
        r_fwd = loco_r2([0, 1, 2], y_next_rna) - loco_r2([0], y_next_rna)
        y_next_atac = Ly[:, 1]
        r_rev = loco_r2([2, 0], y_next_atac) - loco_r2([2], y_next_atac)
        print(f"\n    DIRECTION TEST")
        print(f"      ATAC_link(t) adds to RNA(t+1) beyond RNA(t) : {r_fwd:+.4f}")
        print(f"      RNA(t) adds to ATAC_link(t+1) beyond ATAC(t): {r_rev:+.4f}")
        print(f"      asymmetry (forward - reverse)               : {r_fwd-r_rev:+.4f}"
              f"   <- positive means accessibility leads")
        rr["direction"] = {"forward_gain": r_fwd, "reverse_gain": r_rev,
                           "asymmetry": r_fwd - r_rev}
        linked = rr["+ LINKED enhancer ATAC_t"]["gain_over_ar"]
        rand = rr["CTRL + RANDOM dist-matched peak"]["gain_over_ar"]
        rr["linked_minus_random"] = linked - rand
        print(f"\n    LINKED minus DISTANCE-MATCHED RANDOM: {linked-rand:+.4f}"
              f"   <- the number that separates a regulatory link from regional co-accessibility")
        R["results"][order_tag] = rr

    # ---- verdict, read off the ATAC ordering ----
    print("\n" + "=" * 100)
    a = R["results"].get("atac_order")
    if not a:
        v = "INCONCLUSIVE -- the ATAC ordering produced too few records to fit."
    else:
        lmr = a["linked_minus_random"]
        asym = a["direction"]["asymmetry"]
        arg = a["+ LINKED enhancer ATAC_t"]["gain_over_ar"]
        if lmr > 0.005 and asym > 0.005:
            v = (f"DYNAMIC SIGNAL PRESENT. Linked-enhancer accessibility at t adds {arg:+.4f} R2 to future "
                 f"expression beyond expression at t, and beats a distance-matched random peak by "
                 f"{lmr:+.4f}, with a forward-minus-reverse asymmetry of {asym:+.4f}. Accessibility leads "
                 f"expression on this trajectory.")
        elif arg > 0.005 and lmr <= 0.005:
            v = (f"REGIONAL, NOT REGULATORY. Enhancer accessibility adds {arg:+.4f} over the autoregressive "
                 f"floor, but a DISTANCE-MATCHED RANDOM peak adds essentially the same "
                 f"(difference {lmr:+.4f}). The gain is regional co-accessibility, not a specific "
                 f"enhancer-gene link, and would have looked like a result without that control.")
        elif abs(asym) <= 0.005:
            v = (f"NO TEMPORAL ORDERING. Forward gain {a['direction']['forward_gain']:+.4f} vs reverse "
                 f"{a['direction']['reverse_gain']:+.4f}, asymmetry {asym:+.4f}. Accessibility predicts "
                 f"future expression no better than expression predicts future accessibility, so the two "
                 f"move together and nothing here supports accessibility leading.")
        else:
            v = (f"AUTOREGRESSION DOMINATES. Enhancer accessibility adds {arg:+.4f} over the "
                 f"autoregressive floor; linked-minus-random {lmr:+.4f}; asymmetry {asym:+.4f}. "
                 f"Expression at t is doing the work.")
    R["verdict"] = v
    print(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "dynamic_stage1.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'dynamic_stage1.json'}")


if __name__ == "__main__":
    main()
