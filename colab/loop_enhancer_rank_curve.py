"""POST HOC. Loop 173 reported recall at 1. This reports the whole curve -- R@1, 2, 3, 5, 10, 20 --
and the one number without which none of it can be read.

WHY THE CURVE NEEDS A DENOMINATOR PRINTED BESIDE IT. Within-gene recall at k asks whether a real
enhancer is in the top k of the elements the screen tested against that gene. If a gene had only
six elements tested, R@10 is 1.0 for every model including a coin, and reporting it as an accuracy
would be reporting the library design. So this module prints the distribution of candidates per
evaluable gene FIRST, then the curve, then the same curve for a random pick on the identical genes
and folds. The gap between a model's R@k and the random R@k at the same k is the only part of the
number that belongs to the model.

Nothing here is gated. Loop 173's gates were fixed before its numbers and are not being revisited;
this is the shape of one metric it already reported, and it is labelled POST HOC because it was
written after those numbers were seen.

Arms carried over from loop 173, on identical folds and seeds:
    distance        the bar
    dist+comp       element width, GC, CpG -- the only rung that cleared its gate
    FULL            the whole sequence chain
    FULL_shuffled   the same chain on dinucleotide-shuffled elements, which scored higher

-> outputs/loop_enhancer_rank_curve.json
"""
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from enh import scan as SC                   # noqa: E402
import loop_enhancer_grammar as L            # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_rank_curve.json"
KS = [1, 2, 3, 5, 10, 20]
log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def curve(scores, y, g_idx, jitter):
    """R@k for every k in KS, plus the count of evaluable genes and, per k, how many of them have
    at least k candidates -- because a gene with fewer than k candidates is a free hit."""
    by = defaultdict(list)
    for i in range(len(y)):
        by[int(g_idx[i])].append(i)
    hit = {k: 0 for k in KS}
    reach = {k: 0 for k in KS}
    n = 0
    for g, ix in by.items():
        if len(ix) < 2:
            continue
        yy = y[ix]
        if yy.sum() == 0:
            continue
        n += 1
        o = np.argsort(-(scores[ix] + jitter[ix]))
        r = int(np.argmax(yy[o] == 1))
        for k in KS:
            if len(ix) <= k:
                reach[k] += 1
            hit[k] += int(r < k)
    return {k: hit[k] / max(n, 1) for k in KS}, {k: reach[k] / max(n, 1) for k in KS}, n


def main():
    t0 = time.time()
    say("=" * 104)
    say("POST HOC  LOOP 173's RANK CURVE: R@1 through R@20, with the candidate counts that set it")
    say("=" * 104)
    P = SC.load(say)
    y = P["y"].astype(int)
    g_idx = P["g_idx"]
    chrom = np.array([str(c) for c in P["chrom"]])
    jitter = np.random.default_rng(L.TIE_SEED).uniform(0, 1e-9, size=len(y))

    by = defaultdict(list)
    for i in range(len(y)):
        by[int(g_idx[i])].append(i)
    sizes = [len(v) for k, v in by.items() if len(v) >= 2 and y[v].sum() > 0]
    sizes = np.array(sizes)
    say()
    say(f"  {len(sizes)} evaluable genes (at least two tested elements and at least one validated)")
    say(f"  candidates per gene: median {np.median(sizes):.0f}, mean {sizes.mean():.1f}, "
        f"range {sizes.min()}-{sizes.max()}")
    for k in KS:
        say(f"    genes with <= {k:2d} candidates: {int((sizes <= k).sum()):3d}/{len(sizes)} "
            f"({(sizes <= k).mean():.1%})  <- free hits at R@{k}")

    F, _, _ = L.build_features(P, "el", report=lambda *_: None)
    Fs, _, _ = L.build_features(P, "sh", report=lambda *_: None)
    for fr in (F, Fs):
        for c in fr:
            fr[c] = np.nan_to_num(fr[c], nan=0.0, posinf=0.0, neginf=0.0)

    arms = {"distance": (F, L.ARMS["distance"]),
            "dist+comp": (F, L.ARMS["dist+comp"]),
            "FULL": (F, L.ARMS["FULL"]),
            "FULL_shuffled": (Fs, L.ARMS["FULL"])}
    out = {}
    say()
    say("  " + "arm".ljust(16) + "".join(f"   R@{k:<5}" for k in KS))
    for name, (fr, blocks) in arms.items():
        X, _ = L.matrix(fr, blocks)
        acc = {k: [] for k in KS}
        for s in L.SEEDS:
            fold = L.folds_for(chrom, s)
            sc = L.oof_scores(X, y, fold, s)
            c, reach, n = curve(sc, y, g_idx, jitter)
            for k in KS:
                acc[k].append(c[k])
        out[name] = {k: float(np.mean(v)) for k, v in acc.items()}
        out[name + "_sem"] = {k: float(np.std(v, ddof=1) / np.sqrt(len(v))) for k, v in acc.items()}
        say("  " + name.ljust(16) + "".join(f"  {out[name][k]:.4f}" for k in KS))

    rk = {k: [] for k in KS}
    for s in L.SEEDS:
        rr = np.random.default_rng(1000 + s).uniform(size=len(y))
        c, reach, n = curve(rr, y, g_idx, jitter)
        for k in KS:
            rk[k].append(c[k])
    out["random"] = {k: float(np.mean(v)) for k, v in rk.items()}
    say("  " + "random-pick".ljust(16) + "".join(f"  {out['random'][k]:.4f}" for k in KS))
    say("  " + "free-hit floor".ljust(16) + "".join(f"  {reach[k]:.4f}" for k in KS))

    say()
    say("  THE PART THAT BELONGS TO THE MODEL: R@k minus the random-pick R@k at the same k")
    say("  " + "arm".ljust(16) + "".join(f"   R@{k:<5}" for k in KS))
    for name in arms:
        say("  " + name.ljust(16)
            + "".join(f"  {out[name][k]-out['random'][k]:+.4f}" for k in KS))
    out["random_reach"] = {k: float(reach[k]) for k in KS}
    out["n_evaluable"] = int(n)
    out["candidates_per_gene"] = dict(median=float(np.median(sizes)), mean=float(sizes.mean()),
                                      min=int(sizes.min()), max=int(sizes.max()))
    out["log"] = log
    out["post_hoc"] = True
    out["seconds"] = time.time() - t0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({str(k): v for k, v in out.items()}, open(OUT, "w"), indent=1, default=str)
    say()
    say(f"  -> {OUT}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
