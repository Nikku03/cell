"""WHY ARE REPRESSIVE ELEMENTS DISTAL? Four candidate mechanisms, tested with distance held fixed.

WHAT IS ESTABLISHED. `polarity_stress.py` showed repressive elements sit at a median 332 kb from their TSS
against 32 kb for activating (MWU p 7.7e-38), that distance alone supplies 82% of the polarity
discrimination, and that a real but modest residual survives distance/power/effect matching (AUROC 0.6497,
permutation p 0.032 on 276 balanced pairs). The largest fact about repressive elements is WHERE they are. This
module asks what mechanism puts them there.

TWO ARTIFACT CHECKS FIRST, both of which changed how the evidence must be read.

 1 `direct_rate_negative` correlates with log10 distance at Spearman **-0.9996**. It is distance relabelled.
   Its apparent difference between the classes (p 7.13e-38) is numerically the SAME p-value as distance's,
   because it is the same fact. It is therefore excluded from every arm here rather than counted as
   independent support, which is what a naive read of the column list would have done.
   `indirect_rate_negative` is different: it correlates with distance at only +0.137 while still separating
   the classes (median 0.00254 vs 0.00035, p 2.4e-14), so it IS independent information and is tested.

 2 The repressive fraction ranges from 8% (Schraivogel2020) to 61% (K562_DC_TAP) across the ten contributing
   screens, so a between-screen batch effect was a live explanation for the whole finding. It is not the
   explanation: the distal-repressive relationship replicates WITHIN screen in every well-powered one --
   Gasperini2019 rho +0.410 (p 9e-18, n=402), Nasser2021 +0.486 (p 4e-10), Morris +0.376, Xie +0.441. Only
   HCT116 is null (+0.006, n=40), and there every activating pair shares one distance. Reported per screen
   rather than pooled.

FOUR MECHANISMS, each with its direction stated in advance so a wrong-signed result cannot be reinterpreted
as support:

 M1 INSULATOR DISRUPTION. Silencing a CTCF boundary lets an enhancer reach a gene it was insulated from, so
    the gene goes UP. Predicts CTCF signal HIGHER in repressive elements. The benchmark's own
    `elementChromatinCategory` is suggestive: "CTCF element" is 70.6% repressive against 13.4% for
    "High H3K27ac" -- but n=17, so it needs testing at distance-matched scale.

 M2 POLYCOMB SILENCER. The element is a genuine silencer. Predicts H3K27me3 HIGHER in repressive.

 M3 NOT AN ENHANCER AT ALL. "No H3K27ac" elements are 84.6% repressive (n=13). Predicts H3K27ac, DHS and
    H3K4me1 all LOWER in repressive -- i.e. the class is largely elements that are not enhancers, which would
    make "repressive" a statement about what the element is not.

 M4 INDIRECT EFFECT THROUGH ANOTHER GENE. A distal element may be another gene's promoter or body; silencing
    it lowers THAT gene, and if that gene represses the measured one, the measured gene rises. This is the
    most mundane explanation for a distal positive effect and the one most likely to be mistaken for a
    silencer. Predicts repressive elements are CLOSER to a non-target gene's TSS, and have a higher
    `indirect_rate_negative`.

EVERY MECHANISM IS TESTED WITH DISTANCE HELD FIXED, on the matched subsample, because distance already
explains most of the polarity and any mechanism that merely tracks distance would otherwise look causal.
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
SEEDS = (0, 1, 2)
CATS = ["High H3K27ac", "H3K27ac", "CTCF element", "No H3K27ac", "H3K27me3 element"]


def other_gene_distance(srows):
    """Distance from each element to the nearest TSS that is NOT the measured gene.

    This is the M4 test. If a distal "repressive" element is really another gene's promoter, silencing it
    lowers that gene, and a repressor among those genes raises the measured one. Coordinates come from the
    project's own annotation, which stores chrom and tss per gene.
    """
    net = SP / "cell_complete.json.gz"
    if not net.exists():
        raise SystemExit(f"{net} absent; needed for other-gene TSS coordinates")
    d = json.load(gzip.open(net, "rt"))
    by = {}
    for g in d["genes"]:
        try:
            by.setdefault(g["chrom"], []).append((int(g["tss"]), g["name"]))
        except (KeyError, ValueError, TypeError):
            continue
    del d
    for c in by:
        by[c].sort()
    arr = {c: (np.array([x[0] for x in v]), [x[1] for x in v]) for c, v in by.items()}
    out, names = [], []
    for r in srows:
        ch = r["chrom"]
        mid = (int(r["chromStart"]) + int(r["chromEnd"])) // 2
        tgt = r["measuredGeneSymbol"]
        if ch not in arr:
            out.append(np.nan); names.append(None); continue
        pos, nm = arr[ch]
        i = int(np.searchsorted(pos, mid))
        best, bn = np.inf, None
        for j in range(max(0, i - 6), min(len(pos), i + 6)):
            if nm[j] == tgt:
                continue
            dd = abs(int(pos[j]) - mid)
            if dd < best:
                best, bn = dd, nm[j]
        out.append(best if np.isfinite(best) else np.nan)
        names.append(bn)
    return np.array(out, float), names


def main():
    from scipy import stats
    from sklearn.metrics import average_precision_score, roc_auc_score
    from ranking_gate import load, build, fit_oof, fnum, ACT, CTCFC
    from polarity_stress import folds_by
    print("=" * 100)
    print("DISTAL-SPECIFIC MECHANISM -- four hypotheses, distance held fixed")
    print("=" * 100)
    rows = load()
    d = build(rows)
    sig = d["y"] == 1
    srows = [r for r, k in zip(rows, sig) if k]
    eff = np.array([fnum(r, "EffectSize") for r in rows])[sig]
    rep = (eff > 0).astype(int)
    ae = np.abs(eff)
    pw = np.array([fnum(r, "PowerAtEffectSize25") for r in rows])[sig]
    dist = d["dist"][sig]
    lg = np.log10(dist)
    ch = d["ch"][sig]
    ct = d["ct"][sig]
    ds = np.array([r.get("Dataset") or "NA" for r in srows])
    cat = np.array([r.get("elementChromatinCategory") or "NA" for r in srows])
    irate = np.array([fnum(r, "indirect_rate_negative") for r in srows])
    drate = np.array([fnum(r, "direct_rate_negative") for r in srows])
    print(f"  {len(rep)} causal pairs, {int(rep.sum())} repressive ({rep.mean():.3f})")

    R = {"n": int(len(rep)), "n_repressive": int(rep.sum())}

    # ---- artifact check 1: is direct_rate_negative just distance? ----
    rho_d = float(stats.spearmanr(lg, drate)[0])
    rho_i = float(stats.spearmanr(lg, irate)[0])
    print(f"\n  [0] ARTIFACT CHECKS")
    print(f"    direct_rate_negative   vs log10 distance: Spearman {rho_d:+.4f}"
          f"{'  <- IS distance; excluded from all arms' if abs(rho_d) > 0.95 else ''}")
    print(f"    indirect_rate_negative vs log10 distance: Spearman {rho_i:+.4f}"
          f"{'  <- independent; testable' if abs(rho_i) < 0.5 else ''}")
    R["direct_rate_vs_distance"] = rho_d
    R["indirect_rate_vs_distance"] = rho_i

    # ---- artifact check 2: within-screen replication ----
    print(f"    within-screen distal-repressive replication:")
    R["within_screen"] = {}
    for s in sorted(set(ds), key=lambda x: -(ds == x).sum()):
        m = ds == s
        if m.sum() < 25 or rep[m].sum() < 5 or (rep[m] == 0).sum() < 5:
            continue
        rr = stats.spearmanr(lg[m], rep[m])
        flag = "" if rr[1] < 0.05 else "   <- null here"
        print(f"      {s[:17]:18s} n={int(m.sum()):4d} rep={int(rep[m].sum()):3d}  "
              f"rho {rr[0]:+.3f} (p {rr[1]:.3g}){flag}")
        R["within_screen"][s] = {"n": int(m.sum()), "spearman": float(rr[0]), "p": float(rr[1])}
    nsig = sum(1 for v in R["within_screen"].values() if v["p"] < 0.05 and v["spearman"] > 0)
    print(f"      -> replicates in {nsig}/{len(R['within_screen'])} screens, so not a batch effect")

    # ---- chromatin category ----
    print(f"\n  [1] elementChromatinCategory (the benchmark's own call)")
    print(f"    {'category':20s} {'n':>5s} {'repressive':>11s} {'med dist':>10s}")
    R["category"] = {}
    for c in CATS + ["NA"]:
        m = cat == c
        if not m.any():
            continue
        print(f"    {c:20s} {int(m.sum()):5d} {rep[m].mean():11.3f} {np.median(dist[m]):10.0f}")
        R["category"][c] = {"n": int(m.sum()), "repressive_frac": float(rep[m].mean()),
                            "median_dist": float(np.median(dist[m]))}

    # ---- other-gene distance (M4) ----
    og, ogn = other_gene_distance(srows)
    ok = np.isfinite(og)
    print(f"\n  [2] DISTANCE TO THE NEAREST NON-TARGET TSS (M4), computed for {int(ok.sum())}/{len(og)}")

    # ---- matched subsample: distance, power, |effect| ----
    rng = np.random.default_rng(0)
    key3 = np.column_stack([np.digitize(pw, np.quantile(pw, [0.5])),
                            np.digitize(ae, np.quantile(ae, [0.25, 0.5, 0.75])),
                            np.digitize(lg, np.quantile(lg, [0.25, 0.5, 0.75]))])
    ks = [tuple(k) for k in key3]
    pr, pa = {}, {}
    for i, k in enumerate(ks):
        (pr if rep[i] else pa).setdefault(k, []).append(i)
    keep = []
    for k, ri in pr.items():
        ai = pa.get(k, [])
        n = min(len(ri), len(ai))
        if n:
            keep += list(ri[:n]) + list(rng.choice(ai, n, replace=False))
    keep = np.array(sorted(keep))
    print(f"\n  [3] MECHANISM TESTS ON THE DISTANCE-MATCHED SET ({len(keep)} pairs, "
          f"{int(rep[keep].sum())} repressive)")
    print(f"    residual log10-distance imbalance MWU p "
          f"{stats.mannwhitneyu(lg[keep][rep[keep]==1], lg[keep][rep[keep]==0])[1]:.3g}")

    A = d["A"][sig]
    C = d["C"][sig]
    idx = {n: i for i, n in enumerate(ACT)}
    cidx = {n: i for i, n in enumerate(CTCFC)}
    TESTS = [
        ("M1 insulator", "CTCF.RPM", C[:, cidx["CTCF.RPM"]], "higher"),
        ("M1 insulator", "CTCF_peak_overlap", C[:, cidx["CTCF_peak_overlap"]], "higher"),
        ("M2 polycomb", "H3K27me3.RPM", A[:, idx["H3K27me3.RPM"]], "higher"),
        ("M3 not-an-enhancer", "H3K27ac.RPM", A[:, idx["H3K27ac.RPM"]], "lower"),
        ("M3 not-an-enhancer", "DHS.RPM", A[:, idx["DHS.RPM"]], "lower"),
        ("M3 not-an-enhancer", "H3K4me1.RPM", A[:, idx["H3K4me1.RPM"]], "lower"),
        ("M4 indirect", "indirect_rate_negative", irate, "higher"),
        ("M4 indirect", "dist to other TSS", np.where(ok, og, np.nan), "lower"),
    ]
    print(f"    {'mechanism':20s} {'feature':24s} {'predicted':>10s} {'repressive':>11s} "
          f"{'activating':>11s} {'MWU p':>10s}  verdict")
    R["mechanisms"] = {}
    for mech, nm, v, pred in TESTS:
        vv = v[keep]
        m1, m0 = vv[rep[keep] == 1], vv[rep[keep] == 0]
        g1, g0 = m1[np.isfinite(m1)], m0[np.isfinite(m0)]
        if len(g1) < 10 or len(g0) < 10:
            continue
        p = stats.mannwhitneyu(g1, g0, alternative="two-sided")[1]
        obs = "higher" if np.median(g1) > np.median(g0) else "lower"
        verd = ("as predicted" if obs == pred and p < 0.05 else
                "OPPOSITE" if obs != pred and p < 0.05 else "no difference")
        print(f"    {mech:20s} {nm:24s} {pred:>10s} {np.median(g1):11.4g} {np.median(g0):11.4g} "
              f"{p:10.3g}  {verd}")
        R["mechanisms"][f"{mech}|{nm}"] = {"predicted": pred, "observed": obs,
                                           "rep_median": float(np.median(g1)),
                                           "act_median": float(np.median(g0)), "p": float(p),
                                           "verdict": verd}

    # ---- which block carries the residual discrimination? ----
    print(f"\n  [4] WHICH BLOCK CARRIES THE RESIDUAL, on the matched set (base rate "
          f"{rep[keep].mean():.3f})")
    onehot = np.column_stack([(cat == c).astype(float) for c in CATS])
    BLOCKS = {"CTCF only (M1)": C[:, [cidx[n] for n in CTCFC]],
              "H3K27me3 only (M2)": A[:, [idx["H3K27me3.RPM"], idx["H3K27me3_peak_overlap"]]],
              "activity only (M3)": A[:, [idx["DHS.RPM"], idx["H3K27ac.RPM"], idx["H3K4me1.RPM"]]],
              "indirect only (M4)": np.column_stack([irate, np.where(ok, og, np.nanmedian(og))]),
              "chromatin category": onehot,
              "all four blocks": np.column_stack([
                  C[:, [cidx[n] for n in CTCFC]],
                  A[:, [idx["H3K27me3.RPM"], idx["DHS.RPM"], idx["H3K27ac.RPM"], idx["H3K4me1.RPM"]]],
                  irate, np.where(ok, og, np.nanmedian(og)), onehot])}
    print(f"    {'block':24s} {'AUPRC':>8s} {'AUROC':>8s} {'permuted AUROC':>15s}")
    R["blocks"] = {}
    for tag, X in BLOCKS.items():
        Xm = np.nan_to_num(X[keep])
        ap, au = [], []
        for s in SEEDS:
            p = fit_oof(Xm, rep[keep], folds_by(ch[keep], s))
            ap.append(average_precision_score(rep[keep], p))
            au.append(roc_auc_score(rep[keep], p))
        pmv = []
        for i in range(8):
            tp = rep[keep][np.random.default_rng(700 + i).permutation(len(keep))]
            pmv.append(roc_auc_score(tp, fit_oof(Xm, tp, folds_by(ch[keep], 0))))
        print(f"    {tag:24s} {np.mean(ap):8.4f} {np.mean(au):8.4f} {np.mean(pmv):15.4f}")
        R["blocks"][tag] = {"auprc": float(np.mean(ap)), "auroc": float(np.mean(au)),
                            "permuted_auroc": float(np.mean(pmv))}

    # ---- verdict ----
    print("\n" + "=" * 100)
    supported = [k for k, v in R["mechanisms"].items() if v["verdict"] == "as predicted"]
    opposite = [k for k, v in R["mechanisms"].items() if v["verdict"] == "OPPOSITE"]
    best = max(R["blocks"], key=lambda k: R["blocks"][k]["auroc"] - R["blocks"][k]["permuted_auroc"])
    bb = R["blocks"][best]
    parts = [f"distal-repressive replicates in {nsig}/{len(R['within_screen'])} screens, so it is not a "
             f"batch effect; and direct_rate_negative was excluded because it is distance relabelled "
             f"(rho {rho_d:+.4f})"]
    if supported:
        parts.append("mechanisms supported with distance held fixed: " + ", ".join(supported))
    if opposite:
        parts.append("mechanisms CONTRADICTED (wrong sign, significant): " + ", ".join(opposite))
    if not supported and not opposite:
        parts.append("no mechanism separates the classes once distance is matched")
    parts.append(f"best single block is {best} (AUROC {bb['auroc']:.4f} vs permuted "
                 f"{bb['permuted_auroc']:.4f})")
    R["verdict"] = ". ".join(parts) + "."
    print(f"  VERDICT: {R['verdict']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "distal_mechanism.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'distal_mechanism.json'}")


if __name__ == "__main__":
    main()
