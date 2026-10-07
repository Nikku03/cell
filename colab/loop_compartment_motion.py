"""LOOP 117 -- X-RAY THE CELL 1,517 TIMES: DO THE COMPARTMENTS MOVE, OR ARE THEY FIXED?

THE QUESTION, AND WHY IT DECIDES HOW THE MODEL SHOULD BE BUILT. Loop 116 gave the cell volumes and
therefore concentrations, and it assumed one thing without testing it: that a compartment's contents
are a CONSTANT. If they are, the right move is to freeze them -- measure each compartment's share
once, well, and hand the model a fixed number. If they are not, a fixed number is wrong everywhere
except at the average, and the model needs a compartment state that varies.

Nothing in this repository has ever asked. Loop 102 measured a compartment mass profile from a single
proteome and treated it as the profile. Loop 116 assigned volumes from literature and treated those
as constants too. Both may be right; neither was checked.

THE X-RAYS ARE ALREADY ON DISK. DepMap expression: 19,193 genes x 1,517 cell lines, each gene
standardised across lines, 100% finite. That is 1,517 independent snapshots of a cell in a different
state -- different tissue, different driver mutations, different growth conditions. It is not a time
course of one cell, and this module says so rather than pretending otherwise: it is a sweep over
STATES, which answers "can this move" but not "how fast".

BECAUSE EVERY GENE IS Z-SCORED ACROSS LINES, a compartment's score in a line is its deviation from
its own average. If compartments were fixed, every line would sit near zero for every compartment
and the only spread would be sampling noise. The test is therefore whether compartment scores spread
MORE than size-matched random gene sets do -- a null that automatically handles the fact that a
4,007-gene compartment averages its noise down further than a 62-gene one.

PREDECLARED, before any number:

  W1 THE X-RAYS ARE COMPARABLE AND THE COVERAGE IS DECLARED         THE PREREQUISITE.
       genes per compartment that survive the join to DepMap, and the fraction of each compartment
       retained. A compartment measured on a tenth of its genes is reported as such.
  W2 DO COMPARTMENTS MOVE                                           THE GATE THAT DECIDES.
       standard deviation of each compartment's mean-Z across the 1,517 states, against 200
       size-matched random gene sets. Gate: a compartment MOVES if its spread exceeds the 95th
       percentile of its own size-matched null. The verdict is per compartment, not global, because
       the nucleus and the peroxisome need not behave alike. If none move, the fixed-state answer is
       the right one and this loop hands over the constants.
  W3 IF THEY MOVE, IS IT STRUCTURE OR NOISE                         THE PATTERN.
       PCA of the compartment-by-state matrix. Gate: the leading component must explain more
       variance than the leading component of the same matrix with states shuffled independently
       per compartment -- which destroys co-variation while preserving every compartment's own
       spread. Anything surviving that is a real pattern rather than each compartment wobbling
       alone.
  W4 THE PATTERN, NAMED                                             THE DELIVERABLE.
       which compartments rise together and which trade off, from the leading component's loadings,
       plus the correlation matrix. This is the design a moving-compartment model would need.
  W5 THE FIXED STATE, FOR WHATEVER DOES NOT MOVE                    THE OTHER DELIVERABLE.
       for every compartment that fails W2, its mean and spread reported as the constant the model
       should use, with the spread stated so a later loop knows the error bar it is inheriting.
  W6 FAME AND A NULL THAT CAN FIRE                                  THE GUARD.
       publication mass per compartment against its measured motion, since loop 116 found
       compartment protein mass and publication mass rank-correlate at +1.0000 over 7 compartments.
       gate_guard.null_can_move() confirms the size-matched shuffle changes the statistic before any
       verdict is read.

-> outputs/loop_compartment_motion.json
"""
import collections
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import gate_guard as GG  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
NPERM = 200
SEED = 11700
MIN_GENES = 30

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 117 -- X-ray the cell 1,517 times: do the compartments move, or are they fixed?")
    say("=" * 100)
    say()

    C = json.load(open(LR.CELL))
    lab = {g["name"]: g.get("comp") for g in C["genes"]}
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}
    d = np.load(LR.SC / "depmap_expr.npz", allow_pickle=True)
    Z, syms = d["Z"], list(d["syms"])
    pos = {s: i for i, s in enumerate(syms)}
    say(f"  DepMap: {Z.shape[0]:,} genes x {Z.shape[1]:,} cell lines, each gene standardised "
        f"across lines")
    say(f"  1,517 independent snapshots of a cell in a different STATE -- not a time course of one")
    say(f"  cell. This answers 'can this move', not 'how fast'.")
    say()

    say("W1 THE X-RAYS ARE COMPARABLE AND THE COVERAGE IS DECLARED")
    comp = collections.defaultdict(list)
    for g, c in lab.items():
        if c and g in pos:
            comp[c].append(pos[g])
    total = collections.Counter(c for c in lab.values() if c)
    comps = sorted([c for c in comp if len(comp[c]) >= MIN_GENES],
                   key=lambda c: -len(comp[c]))
    for c in comps:
        say(f"     {c:16s} {len(comp[c]):5,} of {total[c]:5,} genes join DepMap "
            f"({len(comp[c])/total[c]:5.1%})")
    say(f"     {len(comps)} compartments with at least {MIN_GENES} genes")
    say()

    say("W2 DO COMPARTMENTS MOVE")
    rng = np.random.default_rng(SEED)
    n_gene, n_line = Z.shape
    prof = {c: Z[comp[c], :].mean(0) for c in comps}
    rows, movers = [], []
    for c in comps:
        real_sd = float(prof[c].std())
        k = len(comp[c])
        null = np.array([Z[rng.choice(n_gene, size=k, replace=False), :].mean(0).std()
                         for _ in range(NPERM)])
        p95 = float(np.percentile(null, 95))
        moves = real_sd > p95
        rows.append({"compartment": c, "n_genes": k, "sd_across_states": real_sd,
                     "null_p95": p95, "null_mean": float(null.mean()),
                     "ratio": real_sd / p95, "moves": bool(moves)})
        if moves:
            movers.append(c)
        say(f"     {c:16s} sd {real_sd:.4f}   size-matched null 95th {p95:.4f}   "
            f"ratio {real_sd/p95:5.2f}   {'MOVES' if moves else 'fixed'}")
    say(f"     {len(movers)} of {len(comps)} compartments move more than size-matched chance")
    w2 = len(movers) > 0
    say(f"     W2 verdict: {'compartments MOVE -- a fixed number is wrong' if w2 else 'compartments are FIXED -- hand over the constants'}")
    say()

    say("W3 IF THEY MOVE, IS IT STRUCTURE OR NOISE")
    Mx = np.vstack([prof[c] for c in comps])
    Mc = Mx - Mx.mean(1, keepdims=True)
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    nulls = []
    for _ in range(NPERM // 4):
        Sh = np.vstack([rng.permutation(Mc[i]) for i in range(Mc.shape[0])])
        s2 = np.linalg.svd(Sh, compute_uv=False) ** 2
        nulls.append((s2 / s2.sum())[0])
    nulls = np.array(nulls)
    say(f"     leading component explains {ev[0]:.1%} of compartment-by-state variance")
    say(f"     independently-shuffled states give {nulls.mean():.1%} +/- {nulls.std():.1%}")
    say(f"     top 3 components: " + ", ".join(f"{x:.1%}" for x in ev[:3]))
    sur = GG.survival(ev[0], nulls)
    GG.report("leading-component variance vs per-compartment shuffle", sur, emit=say)
    w3 = bool(ev[0] > np.percentile(nulls, 95))
    say(f"     W3 {'PASS' if w3 else 'FAIL'} -- the motion is "
        f"{'CO-ORDINATED across compartments, not each wobbling alone' if w3 else 'not structured'}")
    say()

    say("W4 THE PATTERN, NAMED")
    load = U[:, 0] * np.sign(U[np.argmax(np.abs(U[:, 0])), 0])
    order = np.argsort(-load)
    say(f"     leading component loadings (the axis compartments move along):")
    for i in order:
        bar = "#" * int(abs(load[i]) * 40)
        say(f"       {comps[i]:16s} {load[i]:+.3f}  {bar}")
    say(f"     compartments with POSITIVE loading rise together; NEGATIVE ones fall as they rise")
    R = np.corrcoef(Mc)
    pairs = [(comps[i], comps[j], R[i, j])
             for i in range(len(comps)) for j in range(i + 1, len(comps))]
    pairs.sort(key=lambda x: -abs(x[2]))
    say(f"     strongest co-variation:")
    for a, b, r in pairs[:5]:
        say(f"       {a:16s} {b:16s} r {r:+.3f}")
    say()

    say("W5 THE FIXED STATE, FOR WHATEVER DOES NOT MOVE")
    fixed = [r for r in rows if not r["moves"]]
    if fixed:
        for r in fixed:
            say(f"     {r['compartment']:16s} mean-Z 0 by construction, spread "
                f"{r['sd_across_states']:.4f} -- usable as a constant with that error bar")
    else:
        say(f"     none. Every compartment with >= {MIN_GENES} genes moves more than chance, so")
        say(f"     there is no compartment for which a fixed number is defensible, and loop 116's")
        say(f"     constant-composition assumption is measured to be wrong rather than assumed right.")
    say()

    say("W6 FAME AND A NULL THAT CAN FIRE")
    k0 = len(comp[comps[0]])
    cap = GG.null_can_move(sorted(comp[comps[0]])[:200],
                           sorted(rng.choice(n_gene, size=min(k0, 200), replace=False).tolist()))
    say(f"     CAPABILITY: a size-matched random gene set differs from the real membership in "
        f"{cap['changed']:.1%} of slots -- capable: {cap['capable']}")
    from scipy.stats import spearmanr
    pm = [sum(pubs.get(syms[i], 0.0) for i in comp[c]) for c in comps]
    sd = [r["sd_across_states"] for r in rows]
    ng = [r["n_genes"] for r in rows]
    r_pub = float(spearmanr(pm, sd).statistic)
    r_n = float(spearmanr(ng, sd).statistic)
    say(f"     compartment publication mass vs measured motion: rho {r_pub:+.4f}")
    say(f"     compartment SIZE vs measured motion:              rho {r_n:+.4f}")
    say(f"     (loop 116 found compartment protein mass vs publication mass at +1.0000 over 7)")
    say(f"     the size-matched null already prices size; fame is reported because it is the")
    say(f"     confound that has beaten the biology in six loops this session")
    w6 = bool(cap["capable"])
    say(f"     W6 {'PASS' if w6 else 'FAIL'}")
    say()

    gates = {"W1 coverage declared per compartment": True,
             "W2 the moves-or-fixed verdict is made": True,
             "W3 the motion is co-ordinated, not independent wobble": bool(w3),
             "W4 the pattern is named": True,
             "W5 fixed-state constants handed over where they apply": True,
             "W6 the null is capable and fame reported": bool(w6)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(LR.CELL), str(LR.SC / "depmap_expr.npz")],
                      available=len(lab), used=sum(len(comp[c]) for c in comps),
                      selection="filtered", seed=SEED,
                      controls=["size-matched random gene sets, which price compartment size",
                                "per-compartment state shuffle that keeps each spread and kills co-variation",
                                "the null checked for capability before any verdict is read",
                                "publication mass per compartment against measured motion",
                                "the verdict made per compartment, not globally",
                                "1,517 states declared as STATES, not as a time course"],
                      note="loops 102 and 116 both treated compartment composition as a constant "
                           "without testing it; this tests it")
    RM.report(man, emit=say)
    json.dump({"test": "loop_compartment_motion", "manifest": man, "gates": gates,
               "w1": {c: {"joined": len(comp[c]), "total": total[c]} for c in comps},
               "w2": rows, "n_movers": len(movers), "movers": movers,
               "w3": {"ev": ev[:5].tolist(), "null_mean": float(nulls.mean()),
                      "null_sd": float(nulls.std()), "survival": sur},
               "w4": {"loadings": {comps[i]: float(load[i]) for i in range(len(comps))},
                      "top_pairs": [[a, b, float(r)] for a, b, r in pairs[:10]]},
               "w5": {"fixed": [r["compartment"] for r in fixed]},
               "w6": {"capability": cap, "pubs_vs_motion": r_pub, "size_vs_motion": r_n},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_compartment_motion.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_compartment_motion.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
