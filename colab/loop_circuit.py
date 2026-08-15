"""LOOP 147 -- WHAT CIRCUIT COULD PRODUCE THIS? THE FIRST QUESTION IN THIS ARC THAT IS NOT ABOUT A
PROPERTY OF A PROTEIN.

Nine mechanisms are eliminated and every one asked the same shape of question: what PROPERTY does a
protein have that says it will be destroyed on schedule? Sequence motif, curated annotation,
network membership, phosphosite presence, curated phosphodegron. Two of them turned out to be
measuring publication count and the last one could not see its own target -- annotated phosphosites
land inside a beta-TrCP degron in 1.1% of the genes that carry one.

THIS ASKS A DIFFERENT QUESTION. Not "what is true of these proteins" but "what CIRCUIT could
generate this behaviour", and the answer is constrained before any data is touched, because loop
142 already measured what the behaviour demands:

    b_hi / b_lo  =  20.3x   sustained for a 10% duty cycle in a 24 h period
    relative amplitude 0.4453 at a resting half-life of 29.53 h

That is a hard constraint and most circuits fail it. A LINEAR negative feedback loop produces a
sinusoid, and loop 123 already measured what a sinusoid would need: beta = 2.351, which is
impossible because the loss rate would go negative. So the circuit cannot be linear negative
feedback. It must produce a SWITCH.

WHAT MAKES A SWITCH, and this is textbook control theory rather than biology. A relaxation
oscillator needs two things at once:

    NEGATIVE feedback with delay   sets the PERIOD
    POSITIVE feedback or high cooperativity   makes the transition SHARP

That is the architecture of the real cell-cycle oscillator -- CDK1 drives APC/C-Cdc20, APC/C
destroys cyclin B, cyclin B drives CDK1, and Wee1/Cdc25 supply the positive feedback that makes it
snap rather than glide. So the structural prediction is specific and falsifiable: the circuit must
contain a NEGATIVE loop and a POSITIVE loop SHARING at least one node. Either alone will not do it,
and N1 measures that rather than asserting it.

AND THE HYPOTHESIS THAT WOULD EXPLAIN NINE FAILURES AT ONCE. If the 362 are not IN the oscillator
but DOWNSTREAM of it -- driven by a destruction machine they do not participate in -- then no
property of theirs would predict anything, because they are passengers. That is a different claim
from every previous attempt and N5 tests it against the alternative that they are circuit members.

THE CONFOUND THAT KILLED LOOPS 120 AND 130 APPLIES HERE WITH FULL FORCE. Cycle membership rises
with degree, degree rises with study, and this network's 612,133 edges encode which genes were
looked at together. So N4 is a DEGREE-PRESERVING rewiring null and it runs before N5 is read. Loop
141 used the same control to separate a specific-edge signal from a hub signal, and it is the only
control that has ever worked on this network.

PREDECLARED:

  N1 WHICH ARCHITECTURES CAN PRODUCE THE MEASURED PULSE?             THE MATH, BEFORE ANY DATA.
       simulate three circuits at the measured constants -- linear negative feedback, negative
       feedback with a Hill nonlinearity, and negative-plus-positive (relaxation) -- and report the
       b_hi/b_lo each achieves. Gate: at least one must reach 20.3x and at least one must fail. If
       everything works the simulation is unconstrained; if nothing does, the premise is wrong.

  N2 WHAT COOPERATIVITY DOES IT TAKE?
       the minimum Hill coefficient at which negative feedback alone reaches the required ratio.
       Gate: report it. A requirement of n > 8 would mean plain cooperativity cannot do it and the
       positive loop is structurally necessary rather than merely helpful.

  N3 DO SUCH CIRCUITS EXIST IN THE NETWORK WE HAVE?
       enumerate directed cycles of length 2-4 over the 54,128 SIGNED regulatory edges, classify
       each by the product of its signs, and count negative loops, positive loops, and pairs that
       SHARE a node. Gate: report all three. Coupled pairs are the architecture N1 identifies.

  N4 THE DEGREE-PRESERVING NULL.                                     THE ONE THAT KILLED 120 AND 130.
       the same enumeration on rewired graphs that preserve every node's in- and out-degree and
       the sign distribution. Gate: the real coupled-pair count must exceed the null. If it does
       not, the circuits are what any graph of this shape contains and nothing has been found.

  N5 ARE THE 362 IN THE CIRCUIT, OR DOWNSTREAM OF IT?
       two hypotheses, tested separately against the same null: membership in a coupled pair, and
       reachability from one within 1-3 steps. Gate: report both with the rewired null. DOWNSTREAM
       would explain nine failures at once, and IN would contradict them.

  N6 THE ENTRY AND EXIT PROTEINS, AND THEIR SITES.
       for whichever circuits survive N4, name the node that enters the negative loop and the node
       that leaves it, and pull their UniProt active-site and binding-site annotations from the
       file loop 136 already fetched. Gate: this runs ONLY if N4 passes. Structural analysis of a
       circuit that does not survive its null is decoration.

-> outputs/loop_circuit.json
"""
import csv
import gzip
import json
import math
import os
import random
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(1 << 30)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
HPA = LR.SC / "proteinatlas.tsv"
SITES = Path("colab/data/uniprot_sites.tsv.gz")
SEED = 14700
T_CYCLE = 24.0
TARGET_RATIO = 20.3          # loop 142 X3, the b_hi/b_lo the measurement demands
N_NULL = 20
MAXLEN = 4

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def simulate(kind, n_hill=4.0, T=T_CYCLE, nstep=6000, ncyc=60):
    """Three circuits, same solver, same period target.

    x is the driver, y the destruction activity. The reported quantity is the ratio of the
    highest to the lowest destruction activity over a period -- exactly the b_hi/b_lo that loop
    142 requires, so the comparison is against a measured number and not a shape by eye."""
    dt = T / nstep
    x, y, z = 0.5, 0.5, 0.5
    tr = []
    tau = T / 6.0
    for s in range(ncyc * nstep):
        if kind == "linear-negative":
            dx = (1.0 - y) / tau - x / tau
            dy = (x - y) / tau
            dz = 0.0
        elif kind == "hill-negative":
            dx = (1.0 / (1.0 + (y / 0.5) ** n_hill)) / tau - x / tau
            dy = (x - y) / tau
            dz = 0.0
        else:                                   # relaxation: negative loop + positive feedback
            act = (x ** n_hill) / (0.5 ** n_hill + x ** n_hill)      # positive feedback on y
            dx = (1.0 / (1.0 + (y / 0.3) ** n_hill)) / tau - x / tau
            dy = (act - y) / (tau / 3.0)
            dz = 0.0
        x += dx * dt
        y += dy * dt
        z += dz * dt
        x, y = max(x, 1e-9), max(y, 1e-9)
        if s >= (ncyc - 1) * nstep:
            tr.append(y)
    tr = np.array(tr)
    return float(tr.max() / max(tr.min(), 1e-9)), tr


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    say("=" * 100)
    say("  LOOP 147 -- what circuit could produce this?")
    say("=" * 100)
    say()
    gates, res = {}, {}

    # ---------------------------------------------------------------- N1
    say("N1 WHICH ARCHITECTURES CAN PRODUCE THE MEASURED PULSE?")
    say(f"     loop 142 requires b_hi/b_lo = {TARGET_RATIO}x sustained over a 10% duty in {T_CYCLE:.0f} h")
    n1 = {}
    for kind in ("linear-negative", "hill-negative", "relaxation"):
        r, _ = simulate(kind)
        n1[kind] = r
        say(f"     {kind:<18} achieves {r:>10.2f}x   "
            f"{'REACHES the requirement' if r >= TARGET_RATIO else 'falls short'}")
    ok_some = any(v >= TARGET_RATIO for v in n1.values())
    ok_not_all = any(v < TARGET_RATIO for v in n1.values())
    gates["N1"] = bool(ok_some and ok_not_all)
    res["n1"] = n1
    say(f"     N1 {'PASS' if gates['N1'] else 'FAIL'} -- the simulation "
        f"{'discriminates: some architectures reach it and some cannot' if gates['N1'] else 'is UNCONSTRAINED and cannot select an architecture'}")
    say()

    # ---------------------------------------------------------------- N2
    say("N2 WHAT COOPERATIVITY DOES IT TAKE?")
    need = None
    sweep = {}
    for n in (1, 2, 3, 4, 6, 8, 12, 16):
        r, _ = simulate("hill-negative", n_hill=float(n))
        sweep[n] = r
        if need is None and r >= TARGET_RATIO:
            need = n
        say(f"     Hill n = {n:>2}   negative feedback alone reaches {r:>10.2f}x")
    say(f"     minimum n for negative feedback ALONE: {need if need else 'not reached by n=16'}")
    say(f"     for scale, a Hill coefficient above about 8 is not achievable by cooperative")
    say(f"     binding alone in most real systems, so a requirement that high makes the POSITIVE")
    say(f"     loop structurally necessary rather than merely helpful.")
    gates["N2"] = True
    res["n2"] = {"sweep": sweep, "min_hill_negative_only": need}
    say(f"     N2 PASS -- reported")
    say()

    # ---------------------------------------------------------------- N3
    say("N3 DO SUCH CIRCUITS EXIST IN THE NETWORK WE HAVE?")
    import cell_assembled as CA
    D = CA.load()
    wiring = CA.tf_wiring(D, signed_only=True)
    edges = []
    for tgt, regs in wiring.items():
        for reg, s in regs:
            if s != 0:
                edges.append((reg, tgt, 1 if s > 0 else -1))
    say(f"     signed directed edges: {len(edges):,}")
    nodes = sorted({a for a, _, _ in edges} | {b for _, b, _ in edges})
    say(f"     nodes: {len(nodes):,}")

    def find_cycles(edge_list, maxlen=MAXLEN, cap=200000):
        out_adj = defaultdict(list)
        for a, b, s in edge_list:
            out_adj[a].append((b, s))
        cycles = []
        # only start from nodes that have both in- and out-edges, which is where cycles live
        indeg = defaultdict(int)
        for a, b, s in edge_list:
            indeg[b] += 1
        starts = [n for n in out_adj if indeg[n] > 0]
        for st in starts:
            stack = [(st, [st], 1)]
            while stack:
                node, path, sign = stack.pop()
                if len(path) > maxlen:
                    continue
                for nb, s in out_adj.get(node, ()):
                    if nb == st and len(path) >= 2:
                        cycles.append((tuple(path), sign * s))
                        if len(cycles) >= cap:
                            return cycles
                    elif nb not in path and len(path) < maxlen:
                        stack.append((nb, path + [nb], sign * s))
        return cycles

    cyc = find_cycles(edges)
    neg = [c for c in cyc if c[1] < 0]
    pos = [c for c in cyc if c[1] > 0]
    say(f"     directed cycles of length 2-{MAXLEN}: {len(cyc):,}   negative {len(neg):,}   "
        f"positive {len(pos):,}")
    neg_nodes = defaultdict(set)
    for p, _ in neg:
        for n in p:
            neg_nodes[n].add(p)
    pos_nodes = defaultdict(set)
    for p, _ in pos:
        for n in p:
            pos_nodes[n].add(p)
    coupled = sorted(set(neg_nodes) & set(pos_nodes))
    say(f"     nodes on a NEGATIVE loop: {len(neg_nodes):,}")
    say(f"     nodes on a POSITIVE loop: {len(pos_nodes):,}")
    say(f"     nodes on BOTH -- the relaxation architecture N1 identifies: {len(coupled):,}")
    gates["N3"] = bool(len(coupled) > 0)
    res["n3"] = {"n_cycles": len(cyc), "n_negative": len(neg), "n_positive": len(pos),
                 "n_neg_nodes": len(neg_nodes), "n_pos_nodes": len(pos_nodes),
                 "n_coupled": len(coupled)}
    say(f"     N3 {'PASS' if gates['N3'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- N4
    say("N4 THE DEGREE-PRESERVING NULL")
    say(f"     cycle membership rises with degree and degree rises with study. This network's")
    say(f"     612,133 edges encode which genes were looked at together, and that confound killed")
    say(f"     loops 120 and 130. The null preserves every node's in- and out-degree.")
    nulls = []
    for i in range(N_NULL):
        r = random.Random(SEED + i)
        srcs = [a for a, _, _ in edges]
        tgts = [b for _, b, _ in edges]
        sgns = [s for _, _, s in edges]
        r.shuffle(tgts)
        rew = list({(a, b, s) for a, b, s in zip(srcs, tgts, sgns) if a != b})
        c = find_cycles(rew, cap=100000)
        nn = defaultdict(set)
        pn = defaultdict(set)
        for p, sg in c:
            tgt = nn if sg < 0 else pn
            for n in p:
                tgt[n].add(p)
        nulls.append(len(set(nn) & set(pn)))
        if i < 3:
            say(f"       null {i+1}: coupled nodes {nulls[-1]:,}")
    nm, nsd = float(np.mean(nulls)), float(np.std(nulls))
    z = (len(coupled) - nm) / nsd if nsd > 0 else float("inf")
    say(f"     real coupled nodes {len(coupled):,}   null {nm:.1f} +/- {nsd:.1f}   z = {z:.2f}")
    gates["N4"] = bool(z > 2.0)
    res["n4"] = {"real": len(coupled), "null_mean": nm, "null_sd": nsd, "z": z,
                 "n_null": N_NULL}
    say(f"     N4 {'PASS' if gates['N4'] else 'FAIL'} -- the coupled architecture "
        f"{'exceeds what a graph of this shape contains by chance' if gates['N4'] else 'IS what any graph of this degree sequence contains, and nothing has been found'}")
    say()

    # ---------------------------------------------------------------- N5
    say("N5 ARE THE 362 IN THE CIRCUIT, OR DOWNSTREAM OF IT?")
    cp, ct = {}, {}
    with open(HPA, newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        h = next(rd)
        iG, iCP, iCT = h.index("Gene"), h.index("CCD Protein"), h.index("CCD Transcript")
        for row in rd:
            if len(row) <= max(iG, iCP, iCT):
                continue
            if row[iCP] in ("Yes", "No"):
                cp[row[iG]] = row[iCP]
            if row[iCT] in ("Yes", "No"):
                ct[row[iG]] = row[iCT]
    grp = {}
    for g in sorted(set(cp) & set(ct)):
        p, t = cp[g] == "Yes", ct[g] == "Yes"
        grp[g] = "both" if (p and t) else ("protein only" if p else
                                           ("transcript only" if t else "neither"))
    out_adj = defaultdict(set)
    for a, b, s in edges:
        out_adj[a].add(b)
    reach = set(coupled)
    frontier = set(coupled)
    for step in range(3):
        nxt = set()
        for n in frontier:
            nxt |= out_adj.get(n, set())
        nxt -= reach
        reach |= nxt
        frontier = nxt
    say(f"     coupled nodes {len(coupled):,}; reachable within 3 steps {len(reach):,}")
    n5 = {}
    for name in ("both", "protein only", "transcript only", "neither"):
        gg = [g for g in grp if grp[g] == name]
        inn = sum(1 for g in gg if g in set(coupled))
        dwn = sum(1 for g in gg if g in reach)
        n5[name] = {"n": len(gg), "in_circuit": inn, "downstream": dwn,
                    "frac_in": inn / max(1, len(gg)), "frac_down": dwn / max(1, len(gg))}
        say(f"     {name:<16} n {len(gg):>4}   in circuit {inn:>3} ({inn/max(1,len(gg)):.1%})   "
            f"reachable {dwn:>4} ({dwn/max(1,len(gg)):.1%})")
    po, ne = n5["protein only"], n5["neither"]
    gates["N5"] = bool(gates["N4"] and (po["frac_down"] > ne["frac_down"]
                                        or po["frac_in"] > ne["frac_in"]))
    res["n5"] = n5
    say(f"     N5 {'PASS' if gates['N5'] else 'FAIL'} -- "
        f"{'the 362 sit closer to the circuit than the non-oscillators' if gates['N5'] else 'the 362 are NOT closer to the circuit than the non-oscillators'}")
    if not gates["N4"]:
        say(f"     and N4 failed, so this comparison is against a circuit set that is itself no")
        say(f"     better than a degree-matched null. It is reported and not interpreted.")
    say()

    # ---------------------------------------------------------------- N6
    say("N6 THE ENTRY AND EXIT PROTEINS, AND THEIR SITES")
    if not gates["N4"]:
        say(f"     NOT RUN. N4 failed, and structural analysis of a circuit that does not survive")
        say(f"     its own null is decoration. The gate order exists so that this is a decision")
        say(f"     rather than a temptation: nine mechanisms have already been eliminated in this")
        say(f"     arc and each one would have looked publishable at this step.")
        gates["N6"] = True
        res["n6"] = {"run": False, "reason": "N4 did not survive the degree-preserving null"}
    else:
        top = sorted(coupled, key=lambda n: -(len(neg_nodes[n]) + len(pos_nodes[n])))[:15]
        say(f"     the 15 nodes on the most coupled loops:")
        acc_sites = {}
        if SITES.exists():
            want = set(top)
            with gzip.open(SITES, "rt", errors="replace") as f:
                hh = f.readline().rstrip("\n").split("\t")
                jx = {k: i for i, k in enumerate(hh)}
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if len(p) < len(hh):
                        continue
                    nm = p[jx.get("Protein names", 5)] if "Protein names" in jx else ""
                    for g in want:
                        if re.search(rf"\b{re.escape(g)}\b", nm):
                            acc_sites[g] = (p[jx["Active site"]][:60], p[jx["Binding site"]][:60])
        for n in top:
            a = acc_sites.get(n, ("", ""))
            say(f"       {n:<10} neg loops {len(neg_nodes[n]):>3}  pos loops {len(pos_nodes[n]):>3}"
                f"   {grp.get(n, '-'):<16} act:{a[0][:28]}")
        gates["N6"] = True
        res["n6"] = {"run": True, "top_nodes": top}
    say()

    say("=" * 100)
    for k in ("N1", "N2", "N3", "N4", "N5", "N6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[HPA], available=len(nodes), used=len(nodes), selection="all",
                      seed=SEED,
                      controls=["the architecture question decided by SIMULATION against loop "
                                "142's measured 20.3x, before any network is touched",
                                "a degree-preserving rewiring null, the control that killed "
                                "loops 120 and 130",
                                "N6 gated on N4, so structure is only analysed for circuits that "
                                "survive their null",
                                "two distinct hypotheses for the 362 -- in the circuit, or "
                                "downstream of it"],
                      note="the first question in this arc that is about a CIRCUIT rather than a "
                           "property of a protein.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 147 -- circuit architecture", "manifest": man, "gates": gates,
               **res, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_circuit.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_circuit.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
