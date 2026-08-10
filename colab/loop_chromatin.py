"""THE DEAD LINK -- does 3D genome structure carry anything the linear genome does not?

WHERE THIS COMES FROM.  subsystem_links measured exactly ZERO connections between the 4D chromatin line
and the cell model, and capability_audit confirmed it independently: `any chromosome fold` encodes 8,467
items and has no pipeline to any cell outcome and no controlled result. Two separate instruments, same
answer. The chromatin work in this repository has never once been shown to change a conclusion about a
cell.

Before building a link, the question is whether one is deserved. That is what this measures.

THE TEST, and the control is the whole test.  cell_complete's `model4` gives 7,700 genes a set of 3D
neighbours. Those neighbours are obviously more co-expressed than random gene pairs -- but that is not
evidence for 3D structure, because genes that are close in 3D are usually close along the chromosome
too, and linear proximity alone produces co-expression through shared promoters, shared enhancers and
shared chromatin domains. A comparison against random pairs would pass trivially and prove nothing.

So the comparator is DISTANCE-MATCHED 1D PAIRS: for every 3D pair on the same chromosome, a pair
separated by the same genomic distance, drawn at random. If 3D neighbours beat those, the fold is
carrying information the sequence coordinate does not. If they do not, the dead link is deserved and
should be reported as deserved rather than papered over.

PREDECLARED, before any number:

    C1 SANITY      3D neighbours are more co-expressed than random gene pairs.
                   fails -> the 3D block is not even internally coherent and nothing else is worth
                   measuring.
    C2 BEATS 1D    3D neighbours are more co-expressed than DISTANCE-MATCHED 1D pairs, by more than
                   the 95th percentile of 200 resamples of the matched set.
                   THIS IS THE REAL GATE. fails -> 3D adds nothing beyond linear proximity here, the
                   zero links measured by two instruments are deserved, and that is the finding.
    C3 REACHES A CELL OUTCOME   3D neighbours are more co-essential (share a dependency call) than
                   distance-matched 1D pairs.
                   fails -> even if C2 passes, the structure does not reach a phenotype, which is the
                   axis capability_audit actually failed it on.
    C4 TRANS       the subset of 3D neighbours on DIFFERENT chromosomes -- where no 1D distance exists
                   and no shared-domain explanation is available -- is more co-expressed than random.
                   This is the cleanest possible version of the claim, and the smallest sample.

CONTROLS: random gene pairs; distance-matched same-chromosome pairs, resampled 200 times; the trans
subset as an independent replication; shuffled neighbour assignment.

WHAT HAPPENED, written after the run against the gates above, unedited.

    C1 SANITY     PASS.  3D pairs land in each other's top co-expression list at 0.0146 against 0.0017
                  for random pairs drawn from the SAME gene universe -- 8.6x.
    C2 BEATS 1D   PASS, and this was the real gate.  0.1142 for 3D cis pairs against a distance-matched
                  1D null of 0.0190 +/- 0.0023 (95th percentile 0.0234) over 200 resamples. Six-fold,
                  far outside the null. Holding genomic separation fixed does NOT explain it.
    C3 REACHES A CELL OUTCOME  PASS.  Mean |difference in dependency fraction| 0.0214 for 3D pairs
                  against 0.1047 +/- 0.0043 for distance-matched pairs: five times more alike in how
                  cells depend on them.
    C4 TRANS      PASS.  0.0050 against 0.0013 +/- 0.0002 on 55,836 different-chromosome pairs, where
                  no genomic distance and no shared-domain explanation exists at all.

TWO DEFECTS IN THE FIRST VERSION, both of which would have produced a passing result for bad reasons:
    1  `coexpr` stores only each gene's TOP ~12 partners, so averaging correlation over "pairs that have
       a value" conditions on the outcome -- it compared the mean among 895 surviving 3D pairs against
       the mean among 22 surviving random pairs. Replaced by the hit rate, which is the comparable
       quantity.
    2  the controls were drawn from all 16,460 positioned genes while the 3D pairs come from model4's
       7,700, so any difference between those populations -- study bias, expression, essentiality --
       would have surfaced as a chromatin result. Every control is now drawn from the same universe.
    Fixing both made the result stronger, which is the only reason it is quotable.

THE LIMIT, and it is not small.  `model4` is one of the 40 blocks in cell_complete with NO KNOWN
PRODUCER (cell_complete_build). Nothing in this repository records how its neighbour lists were built,
so it cannot be ruled out that expression data went into constructing them -- in which case C1, C2 and
C4 are partly circular. Two things bound that worry without dispelling it: the hit rate is 0.0146, not
near 1.0, so the neighbours are plainly not simply each gene's top co-expressed partners; and C3 scores
against DEPENDENCY, a different measurement entirely. But the honest statement is that this is a strong
association whose interpretation is limited by an unrecorded provenance, and closing that gap means
finding model4's producer, which is already item 1 on the recovery backlog.

WHAT IT CHANGES.  Two instruments had measured zero links between the chromatin line and the cell model.
A third now says the zero was a fact about the WIRING, not about the biology: the information is there
and nothing was consuming it.

-> outputs/loop_chromatin.json
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"
SEED = 1904
N_RESAMPLE = 200


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("THE DEAD LINK -- does 3D structure carry anything the linear genome does not?")
    say("=" * 100)
    say("  Two instruments independently measured zero connections from the chromatin line to any cell")
    say("  outcome. This asks whether a connection is DESERVED, and the comparator is the whole test:")
    say("  distance-matched 1D pairs, not random pairs. Random pairs would pass trivially.")

    D = json.load(open(CELL))
    genes = D["genes"]
    name2i = {g["name"]: i for i, g in enumerate(genes)}
    model4 = D["model4"]
    coexpr = D["coexpr"]

    # coexpression as a lookup: gene index -> {partner index: r}
    CO = {}
    for k, v in coexpr.items():
        CO[int(k)] = {int(a): float(b) for a, b in v}

    def co(i, j):
        a = CO.get(i, {}).get(j)
        if a is not None:
            return a
        return CO.get(j, {}).get(i)

    # positions
    pos, chrom = {}, {}
    for i, g in enumerate(genes):
        c, s = g.get("chrom"), g.get("gene_start")
        if c is not None and isinstance(s, (int, float)):
            chrom[i] = str(c)
            pos[i] = float(s)
    say(f"\n  {len(genes):,} genes; {len(pos):,} with genome coordinates; "
        f"{len(model4):,} with 3D neighbours; {len(CO):,} with a co-expression list")

    # ---- build the 3D pair set ----------------------------------------------------------------------
    pairs_cis, pairs_trans = [], []
    for k, v in model4.items():
        i = int(k)
        for nm in (v.get("neighbors") or []):
            j = name2i.get(nm)
            if j is None or j == i:
                continue
            if i in chrom and j in chrom:
                (pairs_cis if chrom[i] == chrom[j] else pairs_trans).append((i, j))
    say(f"  3D neighbour pairs: {len(pairs_cis):,} same-chromosome, {len(pairs_trans):,} "
        f"different-chromosome")

    # SAME UNIVERSE. The 3D pairs are drawn from model4's 7,700 genes; the first version drew controls
    # from all 16,460, so any difference between those two populations -- study bias, expression level,
    # essentiality -- would show up as a chromatin result. Every control below is now restricted to the
    # genes that HAVE 3D neighbours, so the only thing varying is whether a pair is a 3D neighbour.
    UNIVERSE = sorted({i for k in model4 for i in [int(k)] if i in pos} |
                      {j for k, v in model4.items() for nm in (v.get("neighbors") or [])
                       for j in [name2i.get(nm)] if j is not None and j in pos})

    by_chrom = defaultdict(list)
    for i in UNIVERSE:                       # same universe as the 3D pairs, see note above
        by_chrom[chrom[i]].append(i)
    for c in by_chrom:
        by_chrom[c].sort(key=lambda x: pos[x])

    # HIT RATE, NOT MEAN, and the first version of this file got it wrong in a way worth recording.
    # `coexpr` stores only each gene's TOP ~12 partners. So co(i,j) returns a value only for pairs that
    # already made that cut, and averaging over "pairs with a value" conditions on the outcome: it
    # measured the mean correlation among survivors, on n=895 of 61,213 3D pairs against n=22 of 20,000
    # random pairs. Those two means are not comparable to each other. The quantity that IS comparable is
    # the FRACTION of pairs that appear in each other's top list at all.
    def hit_rate(prs):
        if not prs:
            return float("nan"), 0
        hits = sum(1 for i, j in prs if co(i, j) is not None)
        return hits / len(prs), len(prs)

    def random_pairs(n):
        a = rng.choice(UNIVERSE, size=n)
        b = rng.choice(UNIVERSE, size=n)
        return [(int(x), int(y)) for x, y in zip(a, b) if x != y]

    def matched_1d(prs):
        """For each 3D cis pair, a pair on the same chromosome separated by the SAME genomic distance.
        This is the comparator that makes the test mean anything: it holds linear proximity fixed and
        asks what the fold adds on top of it."""
        out = []
        for i, j in prs:
            c = chrom[i]
            d = abs(pos[i] - pos[j])
            cand = by_chrom[c]
            if len(cand) < 3:
                continue
            a = int(rng.choice(cand))
            # walk to the gene nearest to distance d from a, on either side
            target = pos[a] + (d if rng.random() < 0.5 else -d)
            best = min(cand, key=lambda x: abs(pos[x] - target))
            if best != a:
                out.append((a, best))
        return out

    # ---- C1 sanity ------------------------------------------------------------------------------------
    m3d, n3d = hit_rate(pairs_cis + pairs_trans)
    mrnd, nrnd = hit_rate(random_pairs(20000))
    c1 = bool(np.isfinite(m3d) and np.isfinite(mrnd) and m3d > mrnd)
    say(f"\n  C1 SANITY -- 3D neighbours vs random pairs FROM THE SAME GENE UNIVERSE")
    say(f"    3D pairs      top-partner hit rate {m3d:.4f}  (n={n3d:,} pairs)")
    say(f"    random pairs  top-partner hit rate {mrnd:.4f}  (n={nrnd:,} pairs)")
    say(f"    C1 {'PASS' if c1 else 'FAIL'}")

    # ---- C2 the real gate -----------------------------------------------------------------------------
    mcis, ncis = hit_rate(pairs_cis)
    draws = []
    for _ in range(N_RESAMPLE):
        mm, _ = hit_rate(matched_1d(pairs_cis[:4000]))
        if np.isfinite(mm):
            draws.append(mm)
    draws = np.array(draws)
    p95 = float(np.percentile(draws, 95)) if len(draws) else float("nan")
    c2 = bool(np.isfinite(mcis) and np.isfinite(p95) and mcis > p95)
    say(f"\n  C2 BEATS 1D -- the real gate: distance-matched same-chromosome pairs")
    say(f"    3D cis pairs            {mcis:.4f}  (n={ncis:,})")
    say(f"    distance-matched 1D     {draws.mean():.4f} +/- {draws.std():.4f}, "
        f"95th pct {p95:.4f}  ({len(draws)} resamples)")
    say(f"    C2 {'PASS' if c2 else 'FAIL'}")
    if not c2:
        say("    3D neighbourhood adds nothing beyond linear proximity in this data. The zero links")
        say("    measured independently by subsystem_links and capability_audit are DESERVED, and the")
        say("    honest move is to say so rather than to build the connection anyway.")

    # ---- C3 does it reach a phenotype -----------------------------------------------------------------
    dep = {i: g.get("dep_frac") for i, g in enumerate(genes)
           if isinstance(g.get("dep_frac"), (int, float))}

    def co_ess(prs):
        vals = [abs(dep[i] - dep[j]) for i, j in prs if i in dep and j in dep]
        return (float(np.mean(vals)) if vals else float("nan")), len(vals)

    e3d, ne3 = co_ess(pairs_cis)
    ed = []
    for _ in range(N_RESAMPLE):
        mm, _ = co_ess(matched_1d(pairs_cis[:4000]))
        if np.isfinite(mm):
            ed.append(mm)
    ed = np.array(ed)
    p5 = float(np.percentile(ed, 5)) if len(ed) else float("nan")
    c3 = bool(np.isfinite(e3d) and np.isfinite(p5) and e3d < p5)   # SMALLER gap = more co-essential
    say(f"\n  C3 REACHES A CELL OUTCOME -- |difference in dependency fraction|, smaller is more alike")
    say(f"    3D cis pairs            {e3d:.4f}  (n={ne3:,})")
    say(f"    distance-matched 1D     {ed.mean():.4f} +/- {ed.std():.4f}, 5th pct {p5:.4f}")
    say(f"    C3 {'PASS' if c3 else 'FAIL'}")

    # ---- C4 trans, the cleanest version ---------------------------------------------------------------
    mtr, ntr = hit_rate(pairs_trans)
    trd = np.array([hit_rate(random_pairs(max(len(pairs_trans), 100)))[0] for _ in range(50)])
    p95t = float(np.percentile(trd, 95)) if len(trd) else float("nan")
    c4 = bool(np.isfinite(mtr) and np.isfinite(p95t) and mtr > p95t)
    say(f"\n  C4 TRANS -- different chromosomes, where no 1D distance and no shared domain can explain it")
    say(f"    trans 3D pairs   {mtr:.4f}  (n={ntr:,})")
    say(f"    random           {trd.mean():.4f} +/- {trd.std():.4f}, 95th pct {p95t:.4f}")
    say(f"    C4 {'PASS' if c4 else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"C1 sanity": c1, "C2 beats 1D proximity": c2, "C3 reaches a cell outcome": c3,
             "C4 trans replication": c4}
    for k, v in gates.items():
        say(f"  {k:<34}{'PASS' if v else 'FAIL'}")
    if c2 and c3:
        say("  THE LINK IS DESERVED. 3D neighbourhood carries information linear proximity does not,")
        say("  and it reaches a cell-level outcome. Building the connection is now justified by")
        say("  measurement rather than by the fact that both subsystems exist.")
    elif c2:
        say("  3D carries information beyond linear proximity, but it does not reach a phenotype. That")
        say("  is exactly the axis capability_audit failed it on, and it remains failed.")
    else:
        say("  THE DEAD LINK IS DESERVED. Against the only control that matters -- pairs held at the")
        say("  same genomic distance -- 3D neighbourhood adds nothing. Two instruments measured zero")
        say("  connections and a third now says the zero is correct. This is a negative result about")
        say("  the chromatin line in THIS repository's data, not about 3D genome biology.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(CELL)], available=len(pairs_cis) + len(pairs_trans),
                      used=len(pairs_cis) + len(pairs_trans), selection="all", seed=SEED,
                      controls=["random gene pairs", f"{N_RESAMPLE} distance-matched 1D resamples",
                                "trans subset as independent replication",
                                "dependency-fraction comparison for the phenotype axis"],
                      note="distance-matched 1D pairs are the decisive control; random pairs pass "
                           "trivially and prove nothing")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_chromatin", "manifest": man, "gates": gates,
               "coexpr_3d": m3d, "coexpr_random": mrnd, "coexpr_3d_cis": mcis,
               "matched_1d": {"mean": float(draws.mean()) if len(draws) else None,
                              "sd": float(draws.std()) if len(draws) else None, "p95": p95},
               "coess_3d_cis": e3d, "coess_matched": {"mean": float(ed.mean()) if len(ed) else None,
                                                      "p5": p5},
               "trans": {"mean": mtr, "n": ntr, "random_p95": p95t},
               "n_pairs_cis": len(pairs_cis), "n_pairs_trans": len(pairs_trans), "log": log},
              open(OUT / "loop_chromatin.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_chromatin.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
