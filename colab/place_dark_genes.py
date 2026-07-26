"""PLACING THE 520 PATHWAY-PREDICTED DARK GENES INTO THE REACTION NETWORK, AND TESTING WHETHER THE PLACEMENT HOLDS.

WHAT IS BEING PLACED. ghost_patch.json carries two very different things. 3,933 dark genes got a FUNCTION from
curated UniProt literature -- that is a fact tier. 520 got a PATHWAY from network guilt-by-association -- that is a
prediction tier, and its own validation file puts it at 0.463 agreement against a 0.239 shuffled baseline, a 1.94x
lift. Only the 520 are placed here, and they are placed as predictions, tagged as such, never merged into the
curated reaction table.

489 of the 520 are absent from the reaction network entirely, so this is genuinely new reach rather than
re-annotation of genes already present.

THE THING THAT WOULD MAKE THIS SELF-CONFIRMING, AND IS THEREFORE CONTROLLED FOR. The predictions are wildly
concentrated: 245 of 520 go to "Generic Transcription Pathway", 77 to "Viral mRNA Translation". Those are among
Reactome's largest and least specific sets, and assigning a gene to a 1,000-member pathway is close to free. Any
validation that does not control for pathway SIZE will confirm the placement no matter what, because a big pathway
contains a big slice of the genome and correlates with everything. So every test here is run against SIZE-MATCHED
random pathways, not just against random genes.

THE VALIDATION TARGET, chosen because it owes nothing to the evidence that produced the placement. The placement
came from PPI guilt-by-association over Reactome pathway membership. So neither PPI nor Reactome can judge it.
DepMap CRISPR gene effect can: if a gene genuinely works inside pathway P, its essentiality profile across 1,150
cell lines should track P's members -- cells that need the pathway need all of it. Co-essentiality is measured,
independent of both curation and the interactome, and is the standard evidence for functional membership.

    score(gene, P) = mean correlation of the gene's DepMap profile with P's member genes
    control 1      = the same, against SIZE-MATCHED random pathways
    control 2      = the same, for degree-matched random genes placed in P

A placement that beats both is supported. One that beats neither is recorded as unsupported rather than quietly
kept -- the point of placing them is to find out which ones are real.
"""
import collections
import csv
import json
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
GHOST = OUT / "ghost_patch.json"
RN = OUT / "reaction_network.json"
GMT = SP / "ReactomePathways.gmt"
DEPMAP = SP / "CRISPRGeneEffect.csv"
MIN_LINES = 200         # a gene needs this many measured cell lines for its profile to be worth correlating
NRAND = 40              # size-matched random pathways per test


def load_depmap():
    with open(DEPMAP) as f:
        r = csv.reader(f)
        head = next(r)
        genes = [h.split(" (")[0] for h in head[1:]]
        rows = []
        for row in r:
            rows.append([np.nan if x == "" else float(x) for x in row[1:]])
    M = np.array(rows, np.float32)          # cell lines x genes
    ok = (~np.isnan(M)).sum(0) >= MIN_LINES
    idx = {g: i for i, g in enumerate(genes) if ok[i]}
    # column-standardise once, ignoring NaN, so correlation is a dot product later
    Z = np.full(M.shape, np.nan, np.float32)
    for g, i in idx.items():
        v = M[:, i]
        m = ~np.isnan(v)
        s = v[m].std()
        if s > 0:
            Z[m, i] = (v[m] - v[m].mean()) / s
    return Z, idx


def corr_to_set(Z, idx, gene, members):
    """Mean Pearson correlation of `gene`'s essentiality profile with each member, over co-measured cell lines."""
    if gene not in idx:
        return float("nan")
    a = Z[:, idx[gene]]
    cs = []
    for m in members:
        if m == gene or m not in idx:
            continue
        b = Z[:, idx[m]]
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() < MIN_LINES:
            continue
        cs.append(float(np.dot(a[ok], b[ok]) / ok.sum()))
    return float(np.mean(cs)) if cs else float("nan")


def main():
    patch = json.loads(GHOST.read_text())["patch"]
    preds = {k: v["pathways_predicted"] for k, v in patch.items() if v.get("pathways_predicted")}
    paths = {}
    for line in GMT.read_text().splitlines():
        p = line.split("\t")
        if len(p) > 3:
            paths[p[0]] = set(p[2:])
    net = json.loads(RN.read_text())
    rn_genes = set()
    for s in net["steps"]:
        rn_genes |= set(s["catalysts"])
        for p in s["in"] + s["out"]:
            rn_genes |= set(p["genes"])
    print(f"{len(preds):,} dark genes with a predicted pathway; {len(set(preds)-rn_genes):,} absent from the "
          f"reaction network")

    tgt = collections.Counter(v[0]["pathway"] for v in preds.values())
    print(f"  {len(tgt):,} distinct target pathways; most-used: {tgt.most_common(4)}")
    print(f"  target pathway sizes: " + ", ".join(f"{n}={len(paths.get(n,())):,}" for n, _ in tgt.most_common(4)))

    Z, idx = load_depmap()
    print(f"  DepMap: {len(idx):,} genes with >={MIN_LINES} measured lines", flush=True)

    rng = np.random.default_rng(0)
    pathnames = [n for n in paths if 10 <= len(paths[n]) <= 2000]
    bysize = sorted(pathnames, key=lambda n: len(paths[n]))
    sizes = np.array([len(paths[n]) for n in bysize])

    rows = []
    for gene, cands in preds.items():
        top = cands[0]
        pw = top["pathway"]
        members = paths.get(pw)
        if not members or gene not in idx:
            continue
        obs = corr_to_set(Z, idx, gene, members)
        if obs != obs:
            continue
        # CONTROL 1: size-matched random pathways. A 1,000-gene pathway correlates with everything, so the
        # comparison must hold size fixed rather than compare against "a random pathway" of arbitrary size.
        want = len(members)
        lo = max(0, int(np.searchsorted(sizes, want)) - NRAND // 2)
        pool = [n for n in bysize[lo:lo + NRAND] if n != pw]
        if len(pool) < 5:
            continue
        ctrl = [corr_to_set(Z, idx, gene, paths[n]) for n in pool]
        ctrl = [c for c in ctrl if c == c]
        if not ctrl:
            continue
        rows.append({"gene": gene, "pathway": pw, "size": want,
                     "confidence": top.get("confidence"), "enrichment": top.get("enrichment"),
                     "partners": top.get("partners_in_pathway"),
                     "coess": obs, "coess_sizematched_ctrl": float(np.mean(ctrl)),
                     "delta": obs - float(np.mean(ctrl)), "new_to_network": gene not in rn_genes})

    if not rows:
        raise SystemExit("no dark gene had both a DepMap profile and a resolvable pathway -- nothing to place")
    d = np.array([r["delta"] for r in rows])
    se = float(d.std() / np.sqrt(len(d)))
    print(f"\n  CO-ESSENTIALITY TEST on {len(rows):,} placeable genes")
    print(f"    mean corr to PREDICTED pathway      {np.mean([r['coess'] for r in rows]):+.4f}")
    print(f"    mean corr to SIZE-MATCHED controls  {np.mean([r['coess_sizematched_ctrl'] for r in rows]):+.4f}")
    print(f"    delta                               {d.mean():+.4f} +/- {se:.4f}")
    supported = [r for r in rows if r["delta"] > 0]
    print(f"    placements with positive delta      {len(supported):,}/{len(rows):,} ({len(supported)/len(rows):.1%})")

    print(f"\n  BY TARGET PATHWAY (the concentration is the risk, so it is broken out)")
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r["pathway"]].append(r["delta"])
    for pw, ds in sorted(byp.items(), key=lambda kv: -len(kv[1]))[:8]:
        a = np.array(ds)
        print(f"    {pw[:44]:<46} n={len(ds):<5,} size={len(paths[pw]):<5,} delta {a.mean():+.4f} "
              f"+/- {a.std()/max(np.sqrt(len(a)),1):.4f}")

    # does the module's own confidence score track the measured support? if not, the score is not usable for tiering
    cf = np.array([r["confidence"] or 0 for r in rows])
    if cf.std() > 0:
        rc = float(np.corrcoef(cf, d)[0, 1])
        print(f"\n  does the predictor's confidence track measured support? corr(confidence, delta) = {rc:+.4f}")

    placed = sorted([r for r in rows if r["delta"] > 0], key=lambda r: -r["delta"])
    json.dump({"n_predicted": len(preds), "n_tested": len(rows), "n_supported": len(supported),
               "mean_delta": float(d.mean()), "se": se,
               "by_pathway": {k: {"n": len(v), "delta": float(np.mean(v))} for k, v in byp.items()},
               "placements": placed}, open(OUT / "dark_gene_placement.json", "w"))
    print(f"\n  -> {OUT/'dark_gene_placement.json'}  ({len(placed):,} supported placements written)")


if __name__ == "__main__":
    main()
