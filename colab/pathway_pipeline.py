"""THE PATHWAY PROLOGUE: what has to happen before a pathway's first step can fire.

THE ASK, AND WHY IT IS A DIFFERENT QUESTION FROM THE LAST THREE MODULES. Start at a pathway. Its first reaction
consumes some protein. That protein did not appear from nowhere -- it was translated, modified into its active form,
carried to the compartment where the pathway runs, and assembled into the complex that actually does the work. So
the chain is:

    synthesis  ->  preparation (modification)  ->  transport  ->  assembly  ->  the pathway's own step

catalyst_arrows.py and assembly_transport_arrows.py both asked "which way does this ONE PPI edge point", and both ran
into the same wall: SIGNOR records regulatory causality, and most of these steps are not regulatory events. Transport
scored 0.382 against SIGNOR -- BELOW chance -- not because the transport annotation is wrong but because when SIGNOR
has an opinion about a transporter it is usually recording that some kinase phosphorylates it, which is a different
fact about a different relation.

THIS MODULE THEREFORE STOPS SCORING AGAINST SIGNOR. The prologue is a claim about ORDER and NECESSITY, not about
regulatory sign, so it gets a validation target that actually matches it and that is independent of Reactome:

    K562 PERTURB-SEQ. If protein X genuinely has to be prepared/transported/assembled before pathway P can run, then
    knocking X out should disturb P's member genes more than knocking out a comparable gene that has nothing to do
    with P. That is measured in the cell line we care about, it owes nothing to curation, and it is the same data the
    rest of this project is scored on.

THE CONFOUND THAT WOULD MAKE THIS TRIVIALLY TRUE, AND THE CONTROL FOR IT. Prologue genes are chaperones, kinases,
importins and ribosomal proteins -- exactly the high-degree, broadly-essential genes whose knockout perturbs almost
everything. Comparing them to a uniformly random gene would "prove" the hypothesis while measuring nothing but
essentiality. So the control is DEGREE-MATCHED and drawn from the same screen, the readout excludes tide genes (those
that move under almost any knockout), and the prologue gene is required NOT to be a member of the pathway it is
supposed to be preparing -- otherwise the test is circular by construction.

Reference points that stay fixed: chance 0.500 | degree rule 0.5664 | global precedence 0.6547 | curated 0.783.
"""
import collections
import json
import os
import pickle
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
CACHE = SP / "catalyst_arrows_cache.json"
PECOMP = SP / "pe_compartment.json"
ORDER = SP / "reactome_order.json"
GMT = SP / "ReactomePathways.gmt"

MAXPROD = 50        # an entity produced by more than this many reactions is currency (ATP/ADP/H2O/Pi)
DEPTH = 4           # how far back the prologue is traced
TAU, TIDE_FRAC = 1.0, 0.05
MIN_PATH, MAX_PATH = 5, 200     # pathway gene-set size band worth testing
NPERM = 2000


# ----------------------------------------------------------------------------------------------------------------
def expand(comp_map, gene_map):
    """PE -> (gene symbols, leaf PE ids). Leaves are needed because compartment lives on the leaf, not the complex."""
    gmemo, lmemo = {}, {}

    def rec(pe, stack=()):
        if pe in gmemo:
            return gmemo[pe], lmemo[pe]
        if pe in stack:
            return set(), set()
        g = set(gene_map.get(pe, ()))
        leaves = {pe} if g else set()
        for c in comp_map.get(pe, ()):
            cg, cl = rec(c, stack + (pe,))
            g |= cg
            leaves |= cl
        gmemo[pe], lmemo[pe] = g, leaves
        return g, leaves

    for pe in set(gene_map) | set(comp_map):
        rec(pe)
    return gmemo, lmemo


def classify(rx, pe2sym, pe2leaf, pecomp):
    """Label every reaction with the step types it performs. Types are FLAGS, not a partition -- one reaction can
    both transport and assemble -- so they are counted as flags and a primary label is assigned by fixed precedence.
    Reactome's own `category` field is carried alongside as an independent cross-check rather than as the answer."""
    kinds = {}
    for r, v in rx.items():
        gin = {g for p in v["in"] for g in pe2sym.get(p, ())}
        gout = {g for p in v["out"] for g in pe2sym.get(p, ())}
        cin, cout = collections.defaultdict(set), collections.defaultdict(set)
        for side, acc in (("in", cin), ("out", cout)):
            for p in v[side]:
                for lf in pe2leaf.get(p, ()):
                    c = pecomp.get(lf)
                    if c:
                        for g in gene_of(lf, pe2sym):
                            acc[g].add(c)
        moved = {g for g in cin if g in cout and cout[g] - cin[g]}
        grew = any(a < b for a in (frozenset(pe2sym.get(p, ())) for p in v["in"]) if a
                   for b in (frozenset(pe2sym.get(p, ())) for p in v["out"]) if len(b) > 1)
        shrank = any(b < a for a in (frozenset(pe2sym.get(p, ())) for p in v["in"]) if len(a) > 1
                     for b in (frozenset(pe2sym.get(p, ())) for p in v["out"]) if b)
        lin = {lf for p in v["in"] for lf in pe2leaf.get(p, ())}
        lout = {lf for p in v["out"] for lf in pe2leaf.get(p, ())}
        flags = set()
        if moved:
            flags.add("TRANSPORT")
        if grew:
            flags.add("ASSEMBLY")
        if shrank:
            flags.add("DISSOCIATION")
        if gout - gin:
            flags.add("SYNTHESIS" if not gin else "RECRUIT")
        if gin and gin == gout and (lin != lout) and not moved:
            flags.add("MODIFICATION")
        for lab in ("TRANSPORT", "ASSEMBLY", "MODIFICATION", "DISSOCIATION", "SYNTHESIS", "RECRUIT"):
            if lab in flags:
                primary = lab
                break
        else:
            primary = "OTHER"
        kinds[r] = {"flags": sorted(flags), "primary": primary, "cat": v.get("cat"),
                    "gin": sorted(gin), "gout": sorted(gout), "moved": sorted(moved)}
    return kinds


def gene_of(leaf, pe2sym):
    return pe2sym.get(leaf, ())


# ----------------------------------------------------------------------------------------------------------------
def main():
    if not CACHE.exists():
        raise SystemExit(f"{CACHE} missing -- run colab/catalyst_arrows.py first (it performs the Reactome fetch)")
    F = json.loads(CACHE.read_text())
    rx, gene, comp = F["rx"], F["gene"], F["comp"]
    pe2sym, pe2leaf = expand(comp, gene)
    pecomp = json.loads(PECOMP.read_text())["comp"]
    preceding = json.loads(ORDER.read_text())["preceding"]

    kinds = classify(rx, pe2sym, pe2leaf, pecomp)
    prim = collections.Counter(k["primary"] for k in kinds.values())
    flagc = collections.Counter(f for k in kinds.values() for f in k["flags"])
    print(f"{len(rx):,} reactions classified")
    print(f"  primary step type : {dict(prim.most_common())}")
    print(f"  flags (overlapping): {dict(flagc.most_common())}")
    # Reactome's own vocabulary, as an independent cross-check on the classifier
    cross = collections.Counter((k["primary"], k["cat"]) for k in kinds.values())
    print(f"  primary x Reactome category (top 8): {cross.most_common(8)}")

    # ---- reaction DAG: entity flow, currency-capped ----
    producers = collections.defaultdict(set)
    for r, v in rx.items():
        for p in v["out"]:
            producers[p].add(r)
    npro = {p: len(s) for p, s in producers.items()}
    dropped = sum(1 for p, n in npro.items() if n > MAXPROD)
    producers = {p: s for p, s in producers.items() if len(s) <= MAXPROD}
    print(f"\n  entity flow: {dropped:,} entities dropped as currency (produced by >{MAXPROD} reactions); "
          f"{len(producers):,} usable")

    # ---- pathways ----
    paths = {}
    for line in GMT.read_text().splitlines():
        p = line.split("\t")
        if len(p) > 3:
            paths[p[1]] = {"name": p[0], "genes": set(p[2:])}
    rxgenes = {r: set(k["gin"]) | set(k["gout"]) for r, k in kinds.items()}
    # a reaction belongs to a pathway when ALL of its protein participants are annotated to that pathway. This is
    # strict on purpose: a loose overlap rule would drag currency-adjacent reactions into every pathway.
    p_rx = collections.defaultdict(set)
    by_gene = collections.defaultdict(set)
    for r, gs in rxgenes.items():
        for g in gs:
            by_gene[g].add(r)
    for pid, d in paths.items():
        cand = set()
        for g in d["genes"]:
            cand |= by_gene.get(g, set())
        for r in cand:
            if rxgenes[r] and rxgenes[r] <= d["genes"]:
                p_rx[pid].add(r)
    have = {p: rs for p, rs in p_rx.items() if rs}
    print(f"  pathways {len(paths):,}; with >=1 fully-contained reaction {len(have):,} "
          f"({len(have)/max(len(paths),1):.1%}); median reactions/pathway "
          f"{int(np.median([len(v) for v in have.values()])) if have else 0}")

    # ---- the prologue: backward trace from each pathway's entry inputs ----
    prologue = {}
    for pid, rs in have.items():
        internal_out = {p for r in rs for p in rx[r]["out"]}
        entry = {p for r in rs for p in rx[r]["in"] if p not in internal_out}
        seen, frontier, chain = set(rs), set(entry), []
        for d in range(1, DEPTH + 1):
            nxt = set()
            for p in frontier:
                for r in producers.get(p, ()):
                    if r in seen:
                        continue
                    seen.add(r)
                    chain.append({"depth": d, "rx": r, "type": kinds[r]["primary"],
                                  "genes": kinds[r]["gin"]})
                    nxt |= set(rx[r]["in"])
            frontier = nxt
            if not frontier:
                break
        if chain:
            prologue[pid] = chain
    depths = [c["depth"] for ch in prologue.values() for c in ch]
    ptypes = collections.Counter(c["type"] for ch in prologue.values() for c in ch)
    print(f"\n  pathways with a traced prologue: {len(prologue):,}")
    print(f"    prologue steps: {len(depths):,}; depth distribution {dict(collections.Counter(depths))}")
    print(f"    step types in prologues: {dict(ptypes.most_common())}")

    # ---- ORDER SANITY: does the trace agree with Reactome's own precedingEvent where both have an opinion? ----
    agree = disagree = 0
    for pid, ch in prologue.items():
        inpath = have[pid]
        for c in ch:
            for succ in inpath:
                if c["rx"] in preceding.get(succ, ()):
                    agree += 1
                elif succ in preceding.get(c["rx"], ()):
                    disagree += 1
    tot = agree + disagree
    print(f"    vs precedingEvent: {agree:,} agree, {disagree:,} contradict"
          + (f" ({agree/tot:.1%} agree)" if tot else " (no overlap)"))

    # ---- THE INDEPENDENT TEST: K562 Perturb-seq ----
    KO = pickle.load(open(SP / "nlz_K562.pkl", "rb"))
    kos = sorted(KO)
    ki = {k: i for i, k in enumerate(kos)}
    genes = sorted({g for v in KO.values() for g in v})
    gi = {g: i for i, g in enumerate(genes)}
    M = np.zeros((len(kos), len(genes)), np.float32)
    for k, v in KO.items():
        for g, z in v.items():
            M[ki[k], gi[g]] = z
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC
    print(f"\n  Perturb-seq: {len(kos):,} knockouts x {len(genes):,} genes; "
          f"{int(tide.sum()):,} tide genes excluded from the readout")

    # degree is the confound; match on it. Degree here is PPI degree from the network being modelled.
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    N = len(names)
    deg = collections.Counter()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            deg[names[a]] += 1
            deg[names[b]] += 1
    kdeg = np.array([deg.get(k, 0) for k in kos], float)
    order = np.argsort(kdeg)
    rank = np.empty(len(kos), int)
    rank[order] = np.arange(len(kos))

    rows = []
    for pid, ch in prologue.items():
        tgt = [gi[g] for g in paths[pid]["genes"] if g in gi and not tide[gi[g]]]
        if not (MIN_PATH <= len(tgt) <= MAX_PATH):
            continue
        pgenes = paths[pid]["genes"]
        cand = {g for c in ch for g in c["genes"] if g in ki and g not in pgenes}
        for g in cand:
            i = ki[g]
            obs = float(A[i, tgt].mean())
            # degree-matched control: the 40 knockouts nearest in PPI-degree rank, excluding pathway members
            lo, hi = max(0, rank[i] - 20), min(len(kos), rank[i] + 20)
            pool = [j for j in order[lo:hi] if kos[j] not in pgenes and j != i]
            if len(pool) < 10:
                continue
            ctrl = float(A[np.array(pool)[:, None], np.array(tgt)[None, :]].mean())
            rows.append((pid, g, obs, ctrl, obs - ctrl, len(tgt)))

    if rows:
        d = np.array([r[4] for r in rows])
        obs_m = float(np.mean([r[2] for r in rows]))
        ctl_m = float(np.mean([r[3] for r in rows]))
        rng = np.random.default_rng(0)
        null = np.array([np.mean(rng.permutation(d) * rng.choice([-1, 1], len(d))) for _ in range(200)])
        se = float(np.std(d) / np.sqrt(len(d)))
        print(f"\n  PROLOGUE GENE KNOCKOUT vs DEGREE-MATCHED CONTROL, on pathway member genes")
        print(f"    testable (pathway, prologue gene) pairs : {len(rows):,}")
        print(f"    mean |z| on pathway genes, prologue KO  : {obs_m:.4f}")
        print(f"    mean |z| on pathway genes, matched ctrl : {ctl_m:.4f}")
        print(f"    difference                              : {np.mean(d):+.4f} +/- {se:.4f}")
        print(f"    sign-flip null spread                   : +/- {float(np.std(null)):.4f}")
        by_type = collections.defaultdict(list)
        g2type = {g: c["type"] for ch in prologue.values() for c in ch for g in c["genes"]}
        for r in rows:
            by_type[g2type.get(r[1], "?")].append(r[4])
        print(f"    by prologue step type:")
        for t, vs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
            print(f"      {t:<14} n={len(vs):<6,} delta {np.mean(vs):+.4f}")
    else:
        print("\n  PROLOGUE TEST: no testable (pathway, prologue gene) pairs -- unmeasurable, not negative")

    json.dump({"n_reactions": len(rx), "primary": dict(prim), "flags": dict(flagc),
               "n_pathways": len(paths), "n_pathways_with_rx": len(have),
               "n_pathways_with_prologue": len(prologue),
               "prologue_step_types": dict(ptypes),
               "precedes_agree": agree, "precedes_disagree": disagree,
               "perturb_pairs": len(rows),
               "perturb_obs": obs_m if rows else None, "perturb_ctrl": ctl_m if rows else None,
               "perturb_delta": float(np.mean(d)) if rows else None,
               "prologue": {p: ch[:50] for p, ch in list(prologue.items())[:500]}},
              open(OUT / "pathway_pipeline.json", "w"))
    print(f"\n  -> {OUT/'pathway_pipeline.json'}")


if __name__ == "__main__":
    main()
