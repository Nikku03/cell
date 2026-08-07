"""WHICH NEW CHANNELS ARE EVEN VIABLE -- the ten-minute check that comes before any of them is built.

WHY THIS FILE EXISTS.  In-context examples took obscure-enzyme top-1 from 0.463 to 0.688. The proposed next
channels are TF co-regulation, protein domains (Pfam), genomic position, and EC class. Each is a week of
work. But every one of them depends on an ANNOTATION, and annotation is exactly what obscure genes do not
have -- that is close to the definition of obscure.

So there is a failure mode that would waste all of it: a channel that looks strong in aggregate because it
works on the famous enzymes, while contributing nothing on the 9,186 gaps, which are obscure by
construction. It would show a real average lift and zero lift where it matters, and the average is the
number people quote.

THE CHECK IS COVERAGE, STRATIFIED BY OBSCURITY, and it costs minutes rather than days:

    for each literature bin, what fraction of the true catalysts carry the annotation at all?

A channel whose annotation is missing on the obscure bin cannot help there no matter how good the model on
top of it is. That is not a prediction about the model, it is arithmetic about the input.

AND THE SECOND HALF, WHICH IS THE ONE THAT ACTUALLY BITES.  Coverage being merely LOWER on obscure genes is
survivable. Coverage being CORRELATED with literature count is not, because then the annotation is partly a
proxy for fame, and a model using it will look like it learned biology when it learned popularity. Both are
measured here: the coverage curve across bins, and the correlation between having the annotation and the
raw paper count.

WHAT IS CHECKED
    pfam        outputs/orphan/prot_pfam.json, 8,609 proteins -> Pfam accessions
    tf_in       outputs/orphan/tf_core.json, 9,906 TF->target edges: does the gene have a known regulator
    position    chrom + tss from cell_complete.json -- present for essentially everything, so this doubles
                as the POSITIVE CONTROL for the check itself. An annotation that is universal must show flat
                coverage across bins; if `position` slopes, the method is broken, not the channel.
    enh         enhancer count, a positional/regulatory annotation that is NOT universal
    ppi         interaction degree, included because it is the textbook example of an annotation that is
                almost pure fame -- it should show the steepest slope of anything here and thereby calibrate
                what "contaminated by popularity" looks like on this axis

PREDECLARED, before any number:
    coverage on the obscure bin > 80% and flat across bins
        -> the channel is viable and the annotation is not a fame proxy. Build it.
    coverage on the obscure bin > 80% but rising steeply with fame
        -> usable, but every arm needs literature count as a covariate and the honest test is the
           obscure bin alone, never the average.
    coverage on the obscure bin < 50%
        -> the channel is dead for this task regardless of model quality. Do not build it. Report the
           number and move on.

-> outputs/orphan/cell_orphan_channel_gate.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_orphan_ties_nn as T  # noqa: E402  -- the SAME benchmark builder every other arm uses

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
BINS = ((0, 20), (21, 50), (51, 150), (151, 500), (501, 10 ** 9))


def binof(v):
    for b, (lo, hi) in enumerate(BINS):
        if lo <= v <= hi:
            return b
    return len(BINS) - 1


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("ARE THE PROPOSED CHANNELS EVEN VIABLE? -- annotation coverage, stratified by obscurity")
    report("=" * 100)
    report("  Every proposed channel rests on an annotation, and annotation is what obscure genes lack.")
    report("  A channel can show a real average lift while contributing nothing on the 9,186 gaps, which")
    report("  are obscure by construction. This costs minutes and decides which channels are worth days.")

    D = json.load(open(OUT / "cell_complete.json"))
    genes = D["genes"]
    idx = {g["name"]: i for i, g in enumerate(genes)}
    pubs = {g["name"]: int(g.get("pubs") or 0) for g in genes}

    pfam = {int(k): set(v) for k, v in json.load(open(OUT / "prot_pfam.json")).items()}
    tf = json.load(open(OUT / "tf_core.json"))
    tf_targets, tf_sources = set(), set()
    for e in tf["edges"]:
        tf_sources.add(int(e[0]))
        tf_targets.add(int(e[1]))
    report(f"\n  prot_pfam: {len(pfam)} genes carry domains | tf_core: {len(tf['edges'])} edges, "
           f"{len(tf_targets)} distinct targets, {len(tf_sources)} distinct regulators")

    B = T.build()
    cats = B["cats"]
    P = B["P"]

    # the population that matters: the catalysts of enzyme-shaped reactions, which is what a gap-filler
    # would have to name. Deduplicated to GENES, since coverage is a property of the gene not the reaction.
    pool = set()
    for i in cats:
        if P[i]:
            pool.update(cats[i])
    pool = sorted(g for g in pool if g in idx)
    report(f"  {len(pool)} distinct annotated catalyst genes in the benchmark's enzyme-shaped population")

    def has(g, what):
        i = idx[g]
        if what == "pfam":
            return i in pfam and len(pfam[i]) > 0
        if what == "tf_in":
            return i in tf_targets
        if what == "position":
            return bool(genes[i].get("chrom")) and bool(genes[i].get("tss"))
        if what == "enh":
            return int(genes[i].get("enh") or 0) > 0
        if what == "ppi":
            return int(genes[i].get("ppi") or 0) > 0
        return False

    CHANNELS = ["pfam", "tf_in", "position", "enh", "ppi"]
    bidx = np.array([binof(pubs.get(g, 0)) for g in pool])
    lit = np.array([pubs.get(g, 0) for g in pool], float)

    report("\n  COVERAGE BY LITERATURE BIN (fraction of catalyst genes carrying the annotation)")
    labels = [f"{lo}-{hi if hi < 10**9 else '+'}" for lo, hi in BINS]
    report("    " + f"{'channel':<12}" + "".join(f"{l:>12}" for l in labels) + f"{'all':>9}{'slope':>9}")
    res = {}
    for ch in CHANNELS:
        v = np.array([has(g, ch) for g in pool], float)
        row = []
        for b in range(len(BINS)):
            m = bidx == b
            row.append(float(v[m].mean()) if m.sum() else float("nan"))
        # slope of coverage against log10 literature count: the fame-proxy measure. Not a correlation with
        # the raw count, which one enormous gene would dominate.
        x = np.log10(np.maximum(lit, 1.0))
        sl = float(np.polyfit(x, v, 1)[0]) if np.std(x) > 0 else float("nan")
        res[ch] = {"by_bin": row, "overall": float(v.mean()), "slope_vs_log_lit": sl,
                   "n_by_bin": [int((bidx == b).sum()) for b in range(len(BINS))]}
        report(f"    {ch:<12}" + "".join(f"{c:>12.3f}" for c in row) + f"{v.mean():>9.3f}{sl:>9.3f}")
    report("    " + f"{'n genes':<12}" + "".join(f"{int((bidx==b).sum()):>12}" for b in range(len(BINS))) +
           f"{len(pool):>9}")

    report("\n  READING -- the gate is the obscure column (0-20 papers), not the average.")
    ob = 0
    for ch in CHANNELS:
        c = res[ch]["by_bin"][ob]
        sl = res[ch]["slope_vs_log_lit"]
        if ch == "position":
            # the positive control: chrom/tss is universal, so flat coverage here is what a WORKING check
            # looks like. If this one slopes, the measurement is broken and no other row is readable.
            verdict = "POSITIVE CONTROL" + (" -- flat, so the check itself is sound" if abs(sl) < 0.02
                                            else " -- SLOPING, so this measurement is BROKEN")
        elif c < 0.50:
            verdict = "DEAD for this task: the annotation is absent where the gaps are"
        elif sl > 0.08:
            verdict = f"usable but fame-loaded (slope {sl:+.3f}); test on the obscure bin alone"
        else:
            verdict = "VIABLE: present on obscure genes and not a fame proxy"
        report(f"    {ch:<12} obscure coverage {c:.3f}  slope {sl:+.3f}   {verdict}")

    report("\n  Note on `ppi`: it is included as a CALIBRATOR, not a proposal. Interaction degree is the")
    report("  textbook fame-driven annotation, so its slope is what a contaminated channel looks like on")
    report("  this axis, and the others should be read against it rather than against zero.")

    OUT.mkdir(parents=True, exist_ok=True)
    out = {"test": "cell_orphan_channel_gate", "n_catalyst_genes": len(pool),
           "bins": labels, "channels": res, "log": log}
    json.dump(out, open(OUT / "cell_orphan_channel_gate.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_channel_gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
