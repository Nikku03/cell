"""direction_general -- does "down is a coherent machine, up is a diffuse programme" hold across knockouts, or was that just GATA1?

TWO PROBLEMS WITH THE FINDING IT TESTS. direction_split reported that GATA1's 88 down-regulated genes carry 1,130 internal PPI edges
against 9.8 expected (115x) while its 533 up-regulated genes carry 369 against 363 (1.02x, chance). That is a striking asymmetry and it has
two weaknesses that have to be removed before anything is built on it.

  n = 1.  It is one knockout. This project has already been burned once today by accepting a passing signal without asking how far it
          generalises: the cross-cell-line gate passed at r=0.537 and AUC 0.903, was treated as licence to pool, and the pooled model then
          made ranking significantly worse. So the asymmetry is measured here across EVERY knockout with enough movers in both directions.

  THE NULL WAS TOO WEAK.  The 115x was computed against gene sets drawn UNIFORMLY from the panel. But the down group is mostly ribosomal
          proteins, and ribosomal proteins are PPI hubs -- they have high degree, so ANY set of them will look densely wired against a
          uniform null. Much of that 115x could be hubness rather than coherence. Here the null is DEGREE-MATCHED: each permuted set draws
          genes from the same log2-degree bins as the observed set, so the only thing left to explain is whether these particular
          high-degree genes are wired TO EACH OTHER more than equally-connected genes would be.

WHAT WOULD KILL THE FINDING: if the down/up asymmetry disappears under the degree-matched null, or if it is present only in GATA1 and a
handful of others rather than being a general property. Either result is worth more than the original number, because a two-headed model is
only worth building if the two heads are really different objects across the board."""
import os
import json, collections, sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
T = 4.2
TIDE_FRAC = 0.05
MINSRC = 20
MINSIDE = 10        # a source needs this many movers in BOTH directions to give a paired comparison
NPERM = 400


def main():
    from scipy.stats import wilcoxon
    df = pd.read_parquet(SP / "repl_k562_zscores.parquet")
    genes = [str(g) for g in df.columns]
    kos = [str(k) for k in df.index]
    Z = df.values.astype(np.float32); A = np.abs(Z)
    tide = (A >= T).mean(0) >= TIDE_FRAC
    keep = np.where(~tide)[0]
    panel = [genes[i] for i in keep]
    pidx = {g: i for i, g in enumerate(panel)}

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; N = len(names)
    ppi = collections.defaultdict(set)
    for e in (D.get("ppi") or []):
        try:
            a, b = names[e[0]], names[e[1]]
        except (TypeError, IndexError):
            continue
        ppi[a].add(b); ppi[b].add(a)
    # adjacency restricted to the panel, as index sets, so edge counting is fast
    adj = {g: {pidx[b] for b in ppi.get(g, ()) if b in pidx} for g in panel}
    deg = np.array([len(adj[g]) for g in panel])
    # DEGREE BINS: the whole point of this script. permuted sets are drawn from the same log2-degree bins.
    dbin = np.minimum((np.log2(deg + 1)).astype(int), 8)
    bins = collections.defaultdict(list)
    for i, b in enumerate(dbin):
        bins[b].append(i)
    bins = {b: np.array(v) for b, v in bins.items()}
    rng = np.random.RandomState(0)

    def edges_within(idxs):
        s = set(idxs)
        return sum(len(adj[panel[i]] & s) for i in idxs) // 2

    def deg_matched_null(idxs, nperm):
        want = collections.Counter(dbin[i] for i in idxs)
        out = np.empty(nperm)
        for k in range(nperm):
            pick = []
            for b, c in want.items():
                pool = bins[b]
                pick.append(rng.choice(pool, min(c, len(pool)), replace=False))
            out[k] = edges_within(np.concatenate(pick))
        return out

    per = (A >= T).sum(1)
    rows = []
    srcs = [kos[i] for i in range(len(kos)) if per[i] >= MINSRC]
    print(f"testing the down/up asymmetry across {len(srcs)} knockouts, degree-matched null, {NPERM} permutations each")
    for s in srcs:
        r = kos.index(s)
        z = Z[r][keep]
        dn = [i for i in range(len(panel)) if z[i] <= -T and panel[i] != s]
        upi = [i for i in range(len(panel)) if z[i] >= T and panel[i] != s]
        if len(dn) < MINSIDE or len(upi) < MINSIDE:
            continue
        rec = {"ko": s, "n_down": len(dn), "n_up": len(upi)}
        for lab, idxs in (("down", dn), ("up", upi)):
            obs = edges_within(idxs)
            nul = deg_matched_null(idxs, NPERM)
            m = float(nul.mean())
            rec[f"{lab}_edges"] = obs
            rec[f"{lab}_exp"] = round(m, 1)
            rec[f"{lab}_fold"] = round(obs / m, 3) if m > 0 else (float("inf") if obs else 1.0)
            rec[f"{lab}_p"] = float((nul >= obs).mean())
        rows.append(rec)
        if len(rows) % 40 == 0:
            print(f"  {len(rows)} sources done", flush=True)

    dfold = np.array([r["down_fold"] for r in rows if np.isfinite(r["down_fold"])])
    ufold = np.array([r["up_fold"] for r in rows if np.isfinite(r["up_fold"])])
    paired = [(r["down_fold"], r["up_fold"]) for r in rows
              if np.isfinite(r["down_fold"]) and np.isfinite(r["up_fold"])]
    a = np.array([x for x, _ in paired]); b = np.array([y for _, y in paired])
    p = float(wilcoxon(a, b).pvalue) if len(a) > 5 and np.any(a != b) else 1.0
    frac = float(np.mean(a > b)) if len(a) else 0.0
    dsig = float(np.mean([r["down_p"] < 0.05 for r in rows]))
    usig = float(np.mean([r["up_p"] < 0.05 for r in rows]))
    g = next((r for r in rows if r["ko"] == "GATA1"), None)

    print(f"\n  {len(rows)} knockouts had >={MINSIDE} movers in BOTH directions")
    print(f"  DOWN internal-PPI fold over a DEGREE-MATCHED null: median {np.median(dfold):.2f}, mean {dfold.mean():.2f}")
    print(f"  UP   internal-PPI fold over a DEGREE-MATCHED null: median {np.median(ufold):.2f}, mean {ufold.mean():.2f}")
    print(f"  down > up in {frac:.1%} of knockouts, paired p={p:.3g}")
    print(f"  significantly wired (p<0.05): down {dsig:.1%} of knockouts, up {usig:.1%}")
    if g:
        print(f"\n  GATA1 re-tested under the degree-matched null:")
        print(f"    down {g['n_down']:4d} genes: {g['down_edges']} edges vs {g['down_exp']} expected = {g['down_fold']}x (p={g['down_p']:.3g})")
        print(f"    up   {g['n_up']:4d} genes: {g['up_edges']} edges vs {g['up_exp']} expected = {g['up_fold']}x (p={g['up_p']:.3g})")
        print(f"    (the uniform-null figure was 115.3x for down, 1.02x for up)")

    top = sorted(rows, key=lambda r: -r["down_fold"])[:10]
    print(f"\n  most coherent DOWN groups")
    print(f"    {'knockout':12s} {'n_dn':>5s} {'edges':>6s} {'exp':>7s} {'fold':>7s} {'n_up':>5s} {'up fold':>8s}")
    for r in top:
        print(f"    {r['ko']:12s} {r['n_down']:>5d} {r['down_edges']:>6d} {r['down_exp']:>7.1f} "
              f"{r['down_fold']:>7.2f} {r['n_up']:>5d} {r['up_fold']:>8.2f}")

    holds = bool(frac > 0.6 and p < 0.05 and np.median(dfold) > np.median(ufold))
    verdict = (
        f"DIRECTION ASYMMETRY, GENERALISED ({len(rows)} knockouts with at least {MINSIDE} movers in both directions): tests whether "
        "'down-regulated genes form a coherent physical module while up-regulated genes do not' is a general property of this data or was "
        "specific to GATA1, and simultaneously replaces the too-weak uniform null with a DEGREE-MATCHED one. The original 115x for GATA1's "
        "down group was computed against uniformly drawn gene sets, but that group is largely ribosomal proteins and ribosomal proteins are "
        "PPI hubs, so much of the enrichment could have been hubness rather than coherence. Each permuted set now draws from the same "
        "log2-degree bins as the observed set. "
        f"RESULT: internal-PPI enrichment over the degree-matched null has median {np.median(dfold):.2f}x for down groups against "
        f"{np.median(ufold):.2f}x for up groups; down exceeds up in {frac:.1%} of knockouts (paired p={p:.3g}); "
        f"{dsig:.0%} of down groups are individually significant against {usig:.0%} of up groups. "
        + (f"GATA1 under the stricter null: down {g['down_fold']}x (p={g['down_p']:.3g}), up {g['up_fold']}x (p={g['up_p']:.3g}), against "
           f"115.3x and 1.02x under the uniform null. " if g else "")
        + ("THE ASYMMETRY GENERALISES and survives degree matching, so it is a property of knockout responses in this data rather than a "
           "quirk of GATA1 or an artifact of ribosomal hubness. Down-regulated sets are coherent physical modules and up-regulated sets are "
           "not, which means a model predicting one blended response is averaging two objects that need different features -- complex and "
           "PPI structure for the down half, regulatory structure for the up half. " if holds else
           "THE ASYMMETRY DOES NOT GENERALISE in the form claimed. It should not be used as the basis for a two-headed model without a "
           "narrower statement of where it does hold. ")
        + "Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"n_knockouts": len(rows), "down_fold_median": float(np.median(dfold)),
               "up_fold_median": float(np.median(ufold)), "frac_down_gt_up": frac, "paired_p": p,
               "frac_down_significant": dsig, "frac_up_significant": usig, "gata1": g,
               "generalises": holds, "per_knockout": rows, "verdict": verdict, "note": verdict},
              open(OUT / "direction_general.json", "w"), indent=1)
    print("\n  -> outputs/orphan/direction_general.json")


if __name__ == "__main__":
    main()
