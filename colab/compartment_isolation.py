"""DOES A SEPARATE GENOME MAKE A TIGHTER MODULE? WHY MITOCHONDRIAL TRANSLATION TOPPED BOTH TESTS.

THE OBSERVATION THAT PROMPTED THIS. Of the 43 pathways the dark-gene placements targeted, "Mitochondrial
translation initiation" scored highest on BOTH independent tests -- location consistency +0.3669 and DepMap
co-essentiality +0.3054 -- while "Generic Transcription Pathway", which absorbed 47% of all placements, scored
+0.0255 and +0.0012. Two tests that share no evidence agreed on the same extreme.

THE PROPOSED EXPLANATION, WHICH IS TESTABLE. The mitochondrion carries its own genome, and a separate genome
requires a separate, dedicated apparatus to express it: its own ribosome, its own initiation and elongation
factors, its own tRNA synthetases. That apparatus is NOT shared with cytosolic translation, so it has no
redundancy -- lose any part and mitochondrial protein synthesis fails as a unit. A module that is spatially
isolated AND functionally non-redundant should therefore show unusually tight co-essentiality, which is exactly
what both tests are sensitive to.

Nuclear transcription is the opposite on every axis: it is spatially diffuse, enormous, and heavily redundant,
with many transcription factors individually dispensable.

A FACT THAT SHARPENS RATHER THAN WEAKENS THE HYPOTHESIS. Zero of the 13 mtDNA-encoded genes appear in DepMap, and
that is not an omission -- CRISPR/Cas9 does not edit mitochondrial DNA. So every mitochondrial gene measured here
is NUCLEAR-encoded machinery imported into the organelle. The tight co-essentiality is therefore a property of the
nuclear-encoded apparatus that serves the separate genome, not of the separate genome's own genes.

WHAT IS MEASURED, over all pathways rather than the handful that dark genes happened to target:

    tightness   mean pairwise DepMap co-essentiality among a pathway's members, versus SIZE-MATCHED random gene
                sets, because a large set regresses to zero correlation and would look loose for free
    purity      the fraction of a pathway's reaction steps that occur in its single most common compartment
    the test    does purity predict tightness across all pathways, and where do the mitochondrial ones rank?

If the explanation is right, spatial purity should predict tightness across the board, and mitochondrial modules
should sit near the top rather than being a one-off.
"""
import collections
import csv
import json
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
GMT = SP / "ReactomePathways.gmt"
DEPMAP = SP / "CRISPRGeneEffect.csv"
MIN_LINES, MIN_SET, MAX_SET, NRAND = 200, 8, 400, 30
MAX_PAIRS = 600         # cap pairs sampled per set; a 400-gene set has 80k pairs and adds nothing over a sample


def load_depmap():
    with open(DEPMAP) as f:
        r = csv.reader(f)
        head = next(r)
        genes = [h.split(" (")[0] for h in head[1:]]
        rows = [[np.nan if x == "" else float(x) for x in row[1:]] for row in r]
    M = np.array(rows, np.float32)
    keep = (~np.isnan(M)).sum(0) >= MIN_LINES
    idx = {g: i for i, g in enumerate(genes) if keep[i]}
    Z = np.full(M.shape, np.nan, np.float32)
    for g, i in idx.items():
        v = M[:, i]
        m = ~np.isnan(v)
        s = v[m].std()
        if s > 0:
            Z[m, i] = (v[m] - v[m].mean()) / s
    return Z, idx


def tightness(Z, idx, members, rng):
    """Mean pairwise co-essentiality within a gene set, over a bounded random sample of pairs."""
    ids = [idx[g] for g in members if g in idx]
    if len(ids) < MIN_SET:
        return float("nan"), 0
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    if len(pairs) > MAX_PAIRS:
        pairs = [pairs[i] for i in rng.choice(len(pairs), MAX_PAIRS, replace=False)]
    cs = []
    for a, b in pairs:
        x, y = Z[:, a], Z[:, b]
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() >= MIN_LINES:
            cs.append(float(np.dot(x[ok], y[ok]) / ok.sum()))
    return (float(np.mean(cs)) if cs else float("nan")), len(ids)


def main():
    paths = {}
    for line in GMT.read_text().splitlines():
        p = line.split("\t")
        if len(p) > 3:
            paths[p[0]] = set(p[2:])
    comp = json.loads((OUT / "compartments.json").read_text())["steps"]
    net = json.loads((OUT / "reaction_network.json").read_text())
    gene_steps = collections.defaultdict(set)
    for s in net["steps"]:
        gs = set(s["catalysts"])
        for p in s["in"] + s["out"]:
            gs |= set(p["genes"])
        for g in gs:
            gene_steps[g].add(s["id"])

    Z, idx = load_depmap()
    mt = [g for g in idx if g.startswith("MT-")]
    print(f"DepMap {len(idx):,} usable genes; mtDNA-encoded present: {len(mt)} "
          f"(CRISPR does not edit mtDNA, so every mitochondrial gene here is nuclear-encoded)")

    rng = np.random.default_rng(0)
    allg = sorted(idx)
    rows = []
    for name, genes in paths.items():
        n_in = len([g for g in genes if g in idx])
        if not (MIN_SET <= n_in <= MAX_SET):
            continue
        t, n = tightness(Z, idx, genes, rng)
        if t != t:
            continue
        # SIZE-MATCHED random control: a big set regresses toward zero correlation and would look loose for free
        ctrl = []
        for _ in range(min(NRAND, 12)):
            samp = [allg[i] for i in rng.choice(len(allg), n, replace=False)]
            c, _ = tightness(Z, idx, samp, rng)
            if c == c:
                ctrl.append(c)
        if not ctrl:
            continue
        cs = collections.Counter()
        for g in genes:
            for sid in gene_steps.get(g, ()):
                for c in comp.get(sid, {}).get("compartments", []):
                    cs[c] += 1
        purity = (cs.most_common(1)[0][1] / sum(cs.values())) if cs else float("nan")
        rows.append({"pathway": name, "n": n, "tight": t, "ctrl": float(np.mean(ctrl)),
                     "delta": t - float(np.mean(ctrl)), "purity": purity,
                     "modal": cs.most_common(1)[0][0] if cs else None})
    print(f"  pathways scored: {len(rows):,}")

    d = np.array([r["delta"] for r in rows])
    pur = np.array([r["purity"] if r["purity"] == r["purity"] else np.nan for r in rows])
    ok = ~np.isnan(pur)
    rc = float(np.corrcoef(pur[ok], d[ok])[0, 1])
    print(f"\n  DOES SPATIAL PURITY PREDICT CO-ESSENTIALITY TIGHTNESS?")
    print(f"    corr(purity, tightness above size-matched control) = {rc:+.4f}  over {int(ok.sum()):,} pathways")

    print(f"\n  TIGHTEST MODULES (tightness minus size-matched control)")
    for r in sorted(rows, key=lambda r: -r["delta"])[:12]:
        print(f"    {r['pathway'][:46]:<48} n={r['n']:<4} delta {r['delta']:+.4f}  "
              f"purity {r['purity']:.2f} ({r['modal']})")

    print(f"\n  LOOSEST MODULES")
    for r in sorted(rows, key=lambda r: r["delta"])[:6]:
        print(f"    {r['pathway'][:46]:<48} n={r['n']:<4} delta {r['delta']:+.4f}  "
              f"purity {r['purity']:.2f} ({r['modal']})")

    bymod = collections.defaultdict(list)
    for r in rows:
        if r["modal"]:
            bymod[r["modal"]].append(r["delta"])
    print(f"\n  BY MODAL COMPARTMENT -- is the mitochondrion special, or is every isolated compartment tight?")
    for c, ds in sorted(bymod.items(), key=lambda kv: -np.mean(kv[1])):
        if len(ds) >= 5:
            print(f"    {c:<24} n={len(ds):<4} mean delta {np.mean(ds):+.4f}")

    mito = [r for r in rows if "itochond" in r["pathway"]]
    if mito:
        rank = {r["pathway"]: i for i, r in enumerate(sorted(rows, key=lambda r: -r["delta"]), 1)}
        print(f"\n  MITOCHONDRIAL PATHWAYS specifically ({len(mito)} of {len(rows):,}):")
        for r in sorted(mito, key=lambda r: -r["delta"])[:8]:
            print(f"    #{rank[r['pathway']]:<5} {r['pathway'][:44]:<46} delta {r['delta']:+.4f}  n={r['n']}")
        med = float(np.median([rank[r["pathway"]] for r in mito]))
        print(f"    median rank {med:.0f} of {len(rows):,}  "
              f"(top {med/len(rows):.0%} -- chance would be 50%)")

    json.dump({"n_pathways": len(rows), "corr_purity_tightness": rc,
               "mtdna_genes_in_depmap": len(mt),
               "by_modal_compartment": {k: float(np.mean(v)) for k, v in bymod.items() if len(v) >= 5},
               "rows": rows}, open(OUT / "compartment_isolation.json", "w"))
    print(f"\n  -> {OUT/'compartment_isolation.json'}")


if __name__ == "__main__":
    main()
