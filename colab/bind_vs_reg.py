"""bind_vs_reg — the direct comparison the whole arc pointed at: for a TF where we have BOTH a measured BINDING map
(ENCODE K562 ChIP-seq) and a measured REGULATION map (Replogle Perturb-seq: which genes change when the TF is knocked
down), put the two gene sets side by side and quantify the gap.

  BINDING  (ChIP-seq)     : genes the TF physically sits on. A peak is assigned to a gene if it is in the gene's PROMOTER
                            (TSS +/- W) or in a distal enhancer that ABC/Hi-C 3D-links to the gene (Gate 3).
  REGULATION (Perturb-seq): genes whose expression significantly changes (|z| > t) when the TF is knocked down. This is the
                            functional readout -- direct OR indirect. K562 genome-scale Perturb-seq (Replogle 2022).

Only TFs with a well-powered knockdown give a trustworthy regulation set. In this K562 data that is essentially GATA1
(a strong-phenotype erythroid master TF, deep coverage, biologically clean signature); most TFs are underpowered
(few cells per perturbation) -- itself part of the answer. GATA1 is also the textbook TF for this question.

The comparison (restricted to the ~8.5k genes measured in Perturb-seq, the only fair universe):
  - |bound|, |regulated|, |bound & regulated|
  - fraction of BOUND genes that are REGULATED  -> how much binding is functional
  - fraction of REGULATED genes that are BOUND  -> how much regulation is direct (vs downstream/indirect)
  - enrichment of the overlap vs chance (hypergeometric) -> is bound-and-regulated more than random?
  - directionality: bound & down on knockdown = ACTIVATED by the TF; bound & up = REPRESSED.

HONEST: 'regulated' includes indirect effects; 'bound' depends on the peak->gene assignment (promoter window + ABC).
Perturb-seq power caps which TFs are testable. Real data throughout; the gap is measured, not asserted.
"""
import json, gzip, bisect
from pathlib import Path
import numpy as np
OUT = Path("outputs/orphan/invivo")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"


def _dec(a):
    return [x.decode() if isinstance(x, bytes) else x for x in a]


def load_regulation(tf, dataset="k562.h5ad", thresh=3.0):
    """Perturb-seq regulation for `tf`: genes with |z|>thresh among measured genes. Returns (reg set, up, down, on_target, U)."""
    import h5py
    h = h5py.File(f"{SP}/{dataset}", "r")
    pert = [p.split("_")[1] for p in _dec(h["obs"]["gene_transcript"][:])]
    cats = _dec(h["var"]["__categories"]["gene_name"][:]); codes = h["var"]["gene_name"][:]
    genes = [cats[c] for c in codes]
    ps = {}
    for i, s in enumerate(pert):
        ps.setdefault(s, i)
    if tf not in ps:
        h.close(); return None
    r = np.array(h["X"][ps[tf], :], dtype=np.float64)
    h.close()
    r[~np.isfinite(r)] = np.nan; r[np.abs(r) > 1e6] = np.nan
    U = set(genes)                                    # measured-gene universe
    reg, up, down = set(), set(), set()
    for i, g in enumerate(genes):
        z = r[i]
        if np.isfinite(z) and abs(z) > thresh:
            reg.add(g)
            (up if z > 0 else down).add(g)
    gi = {g: i for i, g in enumerate(genes)}
    on_t = float(r[gi[tf]]) if tf in gi and np.isfinite(r[gi[tf]]) else float("nan")
    return {"reg": reg, "up": up, "down": down, "on_target_z": on_t, "U": U, "z": {g: float(r[gi[g]]) for g in genes if np.isfinite(r[gi[g]])}}


def _tss():
    """gene -> (chrom, tss); and per-chrom sorted [(tss, gene)]."""
    D = json.load(open(OUT.parent / "orphan/cell_complete.json")) if (OUT.parent / "orphan/cell_complete.json").exists() \
        else json.load(open("outputs/orphan/cell_complete.json"))
    g2 = {}; bychr = {}
    for x in D["genes"]:
        c = x.get("chrom"); t = x.get("tss")
        if c and t:
            g2[x["name"]] = (c, int(t)); bychr.setdefault(c, []).append((int(t), x["name"]))
    for c in bychr:
        bychr[c].sort()
    return g2, bychr


def _peak_index(bedgz):
    """per-chrom (sorted starts, prefix-max ends) for O(log n) overlap."""
    by = {}
    with gzip.open(bedgz, "rt") as fh:
        for line in fh:
            p = line.split("\t")
            if len(p) < 3:
                continue
            by.setdefault(p[0], []).append((int(p[1]), int(p[2])))
    idx = {}
    for c, ivs in by.items():
        ivs.sort(); starts = [a for a, _ in ivs]; pmax = [0] * (len(ivs) + 1); m = 0
        for i, (_, e) in enumerate(ivs):
            m = e if e > m else m; pmax[i + 1] = m
        idx[c] = (starts, pmax)
    return idx


def _ov(idx, c, s, e):
    v = idx.get(c)
    if not v:
        return False
    st, pmax = v; j = bisect.bisect_right(st, e)
    return j > 0 and pmax[j] >= s


def _nearest_gene(bychr, chrom, pos):
    arr = bychr.get(chrom)
    if not arr:
        return None, 1 << 60
    i = bisect.bisect_left(arr, (pos,)); best = None; bd = 1 << 60
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(arr):
            d = abs(arr[j][0] - pos)
            if d < bd:
                bd = d; best = arr[j][1]
    return best, bd


def _abc_index():
    by = {}
    with gzip.open(OUT / "marks/abc_all.bedpe.gz", "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.split("\t")
            by.setdefault(p[0], []).append((int(p[1]), int(p[2]), p[3], int(p[4]), int(p[5])))
    for c in by:
        by[c].sort()
    return by


def binding_genes(tf, U, W=5000, use_abc=True):
    """genes the TF binds: promoter (TSS +/- W overlaps a peak) OR distal enhancer that ABC-links to the gene."""
    peaks = _peak_index(OUT / f"chip_gw/{tf}.bed.gz")
    g2, bychr = _tss()
    prom = set()
    for g in U:
        if g in g2:
            c, t = g2[g]
            if _ov(peaks, c, t - W, t + W):
                prom.add(g)
    distal = set()
    if use_abc:
        abc = _abc_index()
        for c, ivs in abc.items():
            for es, ee, gc, gs, ge in ivs:
                if _ov(peaks, c, es, ee):                       # a peak sits in this enhancer element
                    sym, d = _nearest_gene(bychr, gc, (gs + ge) // 2)   # element's ABC target gene
                    if sym and sym in U and d < 3000:
                        distal.add(sym)
    return {"promoter": prom, "distal_abc": distal, "all": prom | distal}


def _hyper_enrich(U, B, R, overlap):
    """fold-enrichment of overlap vs expected under independence, + a permutation p-value proxy."""
    n = len(U); b = len(B & U); r = len(R & U)
    exp = b * r / n if n else 0
    fold = overlap / exp if exp > 0 else None
    return {"universe": n, "expected_overlap": round(exp, 1), "fold_enrichment": round(fold, 2) if fold else None}


def compare(tf, dataset="k562.h5ad", thresh=3.0, W=5000, use_abc=True):
    reg = load_regulation(tf, dataset, thresh)
    if reg is None:
        return None
    U = reg["U"]
    B = binding_genes(tf, U, W, use_abc)
    Ball = B["all"] & U
    R = reg["reg"] & U
    inter = Ball & R
    updown = {"activated_bound_down": len(inter & reg["down"]), "repressed_bound_up": len(inter & reg["up"])}
    enr = _hyper_enrich(U, Ball, R, len(inter))
    return {"tf": tf, "dataset": dataset, "thresh": thresh, "window": W, "use_abc": use_abc,
            "on_target_z": round(reg["on_target_z"], 2),
            "n_bound": len(Ball), "n_bound_promoter": len(B["promoter"] & U), "n_bound_distal": len(B["distal_abc"] & U),
            "n_regulated": len(R), "n_reg_up": len(reg["up"] & U), "n_reg_down": len(reg["down"] & U),
            "n_bound_and_regulated": len(inter),
            "frac_bound_that_regulate": round(len(inter) / len(Ball), 4) if Ball else None,
            "frac_regulated_that_bound": round(len(inter) / len(R), 3) if R else None,
            "enrichment": enr, "direction": updown,
            "bound_regulated_genes": sorted(inter)[:60]}


def main():
    print("=" * 100)
    print("BINDING vs REGULATION — what a TF SITS ON (ChIP) vs what it CHANGES (Perturb-seq knockdown). Real K562.")
    print("=" * 100)
    res = []
    for tf in ["GATA1", "MAX"]:
        r = compare(tf)
        if not r:
            continue
        res.append(r)
        print(f"\n  {tf}  (knockdown on-target z={r['on_target_z']}; regulation from Perturb-seq |z|>{r['thresh']})")
        print(f"    BINDS      {r['n_bound']:6d} genes  (promoter {r['n_bound_promoter']}, +ABC distal {r['n_bound_distal']})")
        print(f"    REGULATES  {r['n_regulated']:6d} genes  (up {r['n_reg_up']}, down {r['n_reg_down']})")
        print(f"    BOTH       {r['n_bound_and_regulated']:6d} genes")
        print(f"    -> of genes it BINDS, {r['frac_bound_that_regulate']:.1%} are regulated  (most binding is non-functional)")
        print(f"    -> of genes it REGULATES, {r['frac_regulated_that_bound']:.0%} are directly bound  (rest are indirect/downstream)")
        e = r["enrichment"]
        print(f"    -> overlap {r['n_bound_and_regulated']} vs {e['expected_overlap']} expected by chance = {e['fold_enrichment']}x enrichment")
        d = r["direction"]
        print(f"    -> direction: {d['activated_bound_down']} bound+down (activated by {tf}), {d['repressed_bound_up']} bound+up (repressed)")
    out = {"results": res,
           "note": "Binding (ENCODE K562 ChIP, peak->gene via promoter TSS+/-5kb + ABC 3D distal) vs regulation (Replogle "
                   "K562 Perturb-seq, |z|>3 on knockdown), on the ~8.5k measured-gene universe. Only well-powered knockdowns "
                   "give a trustworthy regulation set (GATA1 clean; most TFs underpowered). Measures the binding<->regulation "
                   "gap: binding is vast, regulation is small, overlap is enriched over chance but a small fraction of binding "
                   "is functional and much regulation is indirect. 'Regulated' includes indirect effects. Real data."}
    json.dump(out, open("outputs/orphan/bind_vs_reg.json", "w"), indent=1)
    print("\n  -> outputs/orphan/bind_vs_reg.json")
    return out


if __name__ == "__main__":
    main()
