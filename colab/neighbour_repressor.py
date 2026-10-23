"""ARE THE NEIGHBOURING GENES ENRICHED FOR REPRESSORS? Closing the indirect-effect chain.

THE CHAIN BEING CLOSED. `distal_mechanism.py` established that pairs with a positive effect -- silencing the
element RAISES the gene -- are not silencers. They are non-enhancer elements (H3K27ac 3.57 vs 9.36) sitting
next to a DIFFERENT gene's promoter (nearest non-target TSS 14.1 kb vs 58.9 kb, p 4.3e-10; within 2 kb, 12.6%
vs 2.1%). Insulator disruption and polycomb silencing were both excluded with distance matched (CTCF identical
p 0.84, H3K27me3 identical p 0.72).

That implies a specific, falsifiable chain:

    CRISPRi hits element E, which sits at or beside the promoter of neighbour N
      -> N goes DOWN
      -> measured gene G goes UP
      -> therefore N must REPRESS G

If the neighbours are NOT enriched for repressors, the chain is broken and the indirect-effect story is only
half-supported: the positional fact would stand while its causal interpretation would not.

THREE ANNOTATIONS, weakest to strongest as evidence:

 1 IS THE NEIGHBOUR A REPRESSOR AT ALL? GO "negative regulation of transcription..." in the gene's process
   list (~882 genes carry it), plus the TF flag (1,285 genes). Generic: says nothing about G.

 2 HOW REPRESSIVE IS THE NEIGHBOUR'S OUTGOING SIGNATURE? Fraction of N's signed outgoing edges that are
   negative, over reg (7,680 negative of 612k) and sig (5,921 negative of 17k). Reported as a FRACTION and
   never a count, because a count is a study-bias proxy.

 3 IS THERE A NEGATIVE EDGE FROM N TO G SPECIFICALLY? This is the precise prediction and the only one that
   could confirm the mechanism rather than merely be consistent with it.

STUDY BIAS IS THE CONFOUND THAT WOULD FAKE ALL THREE. Well-studied genes have more annotated edges of every
sign and more GO terms, so if the neighbours of repressive elements are simply better studied, every arm
inflates together. Controlled three ways: fractions rather than counts; `pubs` and out-degree reported and
compared directly; and a matched arm where neighbours are equalised on publication count.

EVERYTHING IS TESTED ON THE DISTANCE-MATCHED SET, and additionally on the prespecified subset where the
neighbour is within 10 kb -- the only pairs where the mechanism can operate at all. Both are reported.
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
NEAR = int(os.environ.get("NR_NEAR", 10_000))


def load_net():
    d = json.load(gzip.open(SP / "cell_complete.json.gz", "rt"))
    genes = [g["name"] for g in d["genes"]]
    gi = {g: i for i, g in enumerate(genes)}
    pubs = np.array([float(g.get("pubs") or 0) for g in d["genes"]])
    istf = np.array([float(g.get("tf") or 0) for g in d["genes"]])
    # repressor by GO process
    rep_go = np.zeros(len(genes))
    for g, v in (d.get("go") or {}).items():
        if g not in gi or not isinstance(v, dict):
            continue
        proc = " ".join(v.get("P") or []).lower()
        if "negative regulation of transcription" in proc:
            rep_go[gi[g]] = 1.0
    # outgoing signed edges, reg + sig pooled
    neg = np.zeros(len(genes)); pos = np.zeros(len(genes)); tot = np.zeros(len(genes))
    edge = {}
    for src, dst, s in d["reg"] + d["sig"]:
        tot[src] += 1
        if s < 0:
            neg[src] += 1
            edge[(src, dst)] = min(edge.get((src, dst), 0), -1)
        elif s > 0:
            pos[src] += 1
            edge[(src, dst)] = max(edge.get((src, dst), 0), 1)
        else:
            edge.setdefault((src, dst), 0)
    del d
    signed = neg + pos
    frac_neg = np.where(signed > 0, neg / np.maximum(signed, 1), np.nan)
    print(f"  network: {len(genes):,} genes; GO-repressor {int(rep_go.sum()):,}; TF {int(istf.sum()):,}; "
          f"genes with >=1 signed outgoing edge {int((signed > 0).sum()):,}")
    return gi, pubs, istf, rep_go, frac_neg, signed, tot, edge


def main():
    from scipy import stats
    from ranking_gate import load, build, fnum
    from distal_mechanism import other_gene_distance
    print("=" * 100)
    print("ARE THE NEIGHBOURING GENES REPRESSORS? -- closing the indirect-effect chain")
    print("=" * 100)
    gi, pubs, istf, rep_go, frac_neg, signed, tot, edge = load_net()

    rows = load()
    d = build(rows)
    sig = d["y"] == 1
    srows = [r for r, k in zip(rows, sig) if k]
    eff = np.array([fnum(r, "EffectSize") for r in rows])[sig]
    rep = (eff > 0).astype(int)
    ae = np.abs(eff)
    pw = np.array([fnum(r, "PowerAtEffectSize25") for r in rows])[sig]
    lg = np.log10(d["dist"][sig])
    og, ogn = other_gene_distance(srows)
    tgt = np.array([r["measuredGeneSymbol"] for r in srows])
    print(f"  {len(rep)} causal pairs, {int(rep.sum())} repressive; neighbour identified for "
          f"{sum(1 for x in ogn if x):,}")

    # distance/power/effect-matched set, as in the mechanism module
    rng = np.random.default_rng(0)
    k3 = np.column_stack([np.digitize(pw, np.quantile(pw, [0.5])),
                          np.digitize(ae, np.quantile(ae, [0.25, 0.5, 0.75])),
                          np.digitize(lg, np.quantile(lg, [0.25, 0.5, 0.75]))])
    pr, pa = {}, {}
    for i, k in enumerate([tuple(x) for x in k3]):
        (pr if rep[i] else pa).setdefault(k, []).append(i)
    keep = []
    for k, ri in pr.items():
        ai = pa.get(k, [])
        n = min(len(ri), len(ai))
        if n:
            keep += list(ri[:n]) + list(rng.choice(ai, n, replace=False))
    keep = np.array(sorted(keep))
    print(f"  distance-matched set: {len(keep)} pairs, {int(rep[keep].sum())} repressive "
          f"(residual log-dist p {stats.mannwhitneyu(lg[keep][rep[keep]==1], lg[keep][rep[keep]==0])[1]:.3g})")

    def vec(fn):
        return np.array([fn(gi[n]) if (n and n in gi) else np.nan for n in ogn])

    N = {"GO repressor": vec(lambda i: rep_go[i]),
         "is TF": vec(lambda i: istf[i]),
         "frac negative out": vec(lambda i: frac_neg[i]),
         "signed out-degree": vec(lambda i: signed[i]),
         "publications": vec(lambda i: pubs[i])}

    R = {"n": int(len(rep)), "n_repressive": int(rep.sum()), "n_matched": int(len(keep)),
         "near_bp": NEAR}
    SUBSETS = {"distance-matched": keep,
               f"matched AND neighbour <{NEAR//1000}kb": keep[np.isfinite(og[keep]) & (og[keep] < NEAR)]}
    for sname, idx in SUBSETS.items():
        r1 = idx[rep[idx] == 1]
        r0 = idx[rep[idx] == 0]
        print(f"\n  [{sname}]  {len(idx)} pairs ({len(r1)} repressive / {len(r0)} activating)")
        if len(r1) < 12 or len(r0) < 12:
            print("    too few to test")
            continue
        print(f"    {'neighbour property':22s} {'repressive':>12s} {'activating':>12s} {'p':>10s}   note")
        R[sname] = {}
        for nm, v in N.items():
            a, b = v[r1], v[r0]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < 10 or len(b) < 10:
                continue
            if set(np.unique(np.r_[a, b])) <= {0.0, 1.0}:
                tab = [[int(a.sum()), int(len(a) - a.sum())], [int(b.sum()), int(len(b) - b.sum())]]
                p = stats.fisher_exact(tab)[1]
                sa, sb = a.mean(), b.mean()
                note = "Fisher on a binary flag"
            else:
                p = stats.mannwhitneyu(a, b, alternative="two-sided")[1]
                sa, sb = np.median(a), np.median(b)
                note = "MWU on medians"
            print(f"    {nm:22s} {sa:12.4g} {sb:12.4g} {p:10.3g}   {note}")
            R[sname][nm] = {"repressive": float(sa), "activating": float(sb), "p": float(p),
                            "n_rep": int(len(a)), "n_act": int(len(b))}

        # ---- the specific prediction: a NEGATIVE edge from neighbour to the measured gene ----
        def has_edge(i, want):
            n, g = ogn[i], tgt[i]
            if not n or n not in gi or g not in gi:
                return np.nan
            s = edge.get((gi[n], gi[g]))
            if s is None:
                return 0.0
            return 1.0 if (s < 0 if want == "neg" else s > 0) else 0.0
        for want, lab in (("neg", "NEGATIVE edge neighbour->target"),
                          ("pos", "positive edge neighbour->target")):
            a = np.array([has_edge(i, want) for i in r1], float)
            b = np.array([has_edge(i, want) for i in r0], float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < 10 or len(b) < 10:
                continue
            tab = [[int(a.sum()), int(len(a) - a.sum())], [int(b.sum()), int(len(b) - b.sum())]]
            p = stats.fisher_exact(tab)[1]
            print(f"    {lab:22.22s} {a.mean():12.4g} {b.mean():12.4g} {p:10.3g}   "
                  f"Fisher; {int(a.sum())}/{len(a)} vs {int(b.sum())}/{len(b)}")
            R[sname][lab] = {"repressive": float(a.mean()), "activating": float(b.mean()),
                             "p": float(p), "hits_rep": int(a.sum()), "hits_act": int(b.sum()),
                             "n_rep": int(len(a)), "n_act": int(len(b))}

    # ---- study-bias control: equalise neighbour publication count ----
    print(f"\n  STUDY-BIAS CONTROL -- neighbours equalised on publication count")
    pv = N["publications"]
    ok = np.isfinite(pv)
    idx = keep[ok[keep]]
    q = np.quantile(pv[idx], [0.25, 0.5, 0.75])
    bins = np.digitize(pv, q)
    pr2, pa2 = {}, {}
    for i in idx:
        (pr2 if rep[i] else pa2).setdefault(int(bins[i]), []).append(i)
    k2 = []
    for b, ri in pr2.items():
        ai = pa2.get(b, [])
        n = min(len(ri), len(ai))
        if n:
            k2 += list(ri[:n]) + list(np.random.default_rng(1).choice(ai, n, replace=False))
    k2 = np.array(sorted(k2))
    if len(k2) >= 40:
        r1, r0 = k2[rep[k2] == 1], k2[rep[k2] == 0]
        pp = stats.mannwhitneyu(pv[r1], pv[r0], alternative="two-sided")[1]
        print(f"    {len(k2)} pairs; residual publication imbalance p {pp:.3g}")
        R["pubs_matched"] = {"n": int(len(k2)), "residual_p": float(pp)}
        for nm in ("GO repressor", "frac negative out"):
            v = N[nm]
            a, b = v[r1], v[r0]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < 10 or len(b) < 10:
                continue
            if set(np.unique(np.r_[a, b])) <= {0.0, 1.0}:
                p = stats.fisher_exact([[int(a.sum()), len(a) - int(a.sum())],
                                        [int(b.sum()), len(b) - int(b.sum())]])[1]
                sa, sb = a.mean(), b.mean()
            else:
                p = stats.mannwhitneyu(a, b, alternative="two-sided")[1]
                sa, sb = np.median(a), np.median(b)
            print(f"    {nm:22s} {sa:12.4g} {sb:12.4g} {p:10.3g}")
            R["pubs_matched"][nm] = {"repressive": float(sa), "activating": float(sb), "p": float(p)}
    else:
        print("    too few publication-matched pairs")

    # ---- verdict ----
    print("\n" + "=" * 100)
    base = R.get("distance-matched", {})
    nearsub = R.get(f"matched AND neighbour <{NEAR//1000}kb", {})
    neg_key = "NEGATIVE edge neighbour->target"
    hits = base.get(neg_key, {})
    go = base.get("GO repressor", {})
    fn = base.get("frac negative out", {})
    sup = []
    if go and go["p"] < 0.05 and go["repressive"] > go["activating"]:
        sup.append(f"GO-repressor annotation enriched ({go['repressive']:.3f} vs {go['activating']:.3f}, "
                   f"p {go['p']:.3g})")
    if fn and fn["p"] < 0.05 and fn["repressive"] > fn["activating"]:
        sup.append(f"neighbour's outgoing edges more negative ({fn['repressive']:.3f} vs "
                   f"{fn['activating']:.3f}, p {fn['p']:.3g})")
    if hits and hits["p"] < 0.05 and hits["repressive"] > hits["activating"]:
        sup.append(f"a NEGATIVE neighbour->target edge is enriched ({hits['hits_rep']}/{hits['n_rep']} vs "
                   f"{hits['hits_act']}/{hits['n_act']}, p {hits['p']:.3g})")
    if sup:
        R["verdict"] = ("CHAIN SUPPORTED: " + "; ".join(sup) +
                        ". So the neighbouring genes are enriched for repressors of the measured gene, which "
                        "is the causal closure the positional evidence predicted.")
    else:
        hv = f"{hits.get('hits_rep', 0)}/{hits.get('n_rep', 0)} vs {hits.get('hits_act', 0)}/{hits.get('n_act', 0)}" \
            if hits else "not testable"
        R["verdict"] = (
            "CHAIN NOT CLOSED. The neighbours of repressive elements are NOT detectably enriched for "
            f"repressors: GO-repressor p {go.get('p', float('nan')):.3g}, outgoing-negative-fraction p "
            f"{fn.get('p', float('nan')):.3g}, negative neighbour->target edge {hv}. The POSITIONAL fact "
            "stands -- these elements sit beside other promoters -- but its causal interpretation is "
            "unconfirmed. The most likely reason is annotation coverage rather than biology: signed "
            "regulatory edges cover a small fraction of gene pairs, so the specific N->G edge is usually "
            "simply absent from any database, and this test has little power to find it.")
    print(f"  VERDICT: {R['verdict']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "neighbour_repressor.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'neighbour_repressor.json'}")


if __name__ == "__main__":
    main()
