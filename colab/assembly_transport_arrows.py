"""ARROWS FROM ASSEMBLY ORDER, TRANSPORT, AND MODIFICATION-THEN-TRANSPORT CHAINS.

WHY THESE THREE AND NOT MORE BINDING. catalyst_arrows.py established the hard limit of reading direction off a single
reaction: a binding step is A + B -> AB, both partners are inputs, and nothing inside that step says which came
first. Only 95 of the 36,415 PPI edges whose endpoints share a reaction split across input and output. The three
sources here were proposed precisely because each has an asymmetry that a binding step lacks:

  ASSEMBLY   complexes are built stepwise -- A + B -> AB, then AB + C -> ABC. Inside the second reaction AB and C are
             both inputs, so the reaction gives no arrow. But ACROSS the two reactions, A and B were present before
             C. The asymmetry is in the sequence of assembly steps, not inside any one of them.
  TRANSPORT  a cargo protein is carried between compartments. Input is X in compartment 1, output is the SAME X in
             compartment 2. A transporter acts on its cargo and the cargo does not transport the transporter, so
             this relation is asymmetric by construction rather than by annotation.
  MOD->TRANS a protein is modified in one reaction and that modification is what licenses its transport in a later
             one, so the modifier acts before the transporter on the same cargo.

WHAT THIS MODULE REFUSES TO ASSUME. "Assembled earlier" is not the same relation as "causally upstream". SIGNOR
records regulatory causality -- kinase phosphorylates substrate -- and a subunit being present before another subunit
is a statement about time, not influence. Scoring assembly order against SIGNOR therefore tests a specific and
falsifiable hypothesis (that assembly order predicts signalling direction) rather than measuring whether the assembly
order itself is right. Both are reported, separately, and the assembly rule additionally gets an internal-consistency
target that IS the right one for it: Reactome's complex NESTING vs Reactome's own assembly REACTIONS.

CONTROLS, because every one of these rules has a way of being right for the wrong reason:
  matched degree   a walk's direction on an undirected graph is provably the degree ratio, so every rule is scored
                   against the degree rule computed ON ITS OWN EDGE SET, never against a global number.
  hub share        transport especially risks becoming "hub -> everything": KPNB1, XPO1, RAN and the nuclear pore are
                   enormous PPI hubs. The fraction of arrows leaving the top hubs is reported, and accuracy is
                   re-measured with them removed.
  size share       assembly risks firing mostly on the ribosome, proteasome and spliceosome, which would re-encode
                   degree. Arrow counts are reported per complex size band.
  independence     Reactome precedingEvent already backs the 0.6547 global rule. Any rule derived from precedingEvent
                   is NOT independent evidence and is labelled as such so it is never counted as corroboration.

Reference points, fixed: chance 0.500 | degree rule 0.5664 | global precedence 0.6547 | curated-vs-curated 0.783.
"""
import collections
import json
import re
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
CACHE = SP / "catalyst_arrows_cache.json"           # written by catalyst_arrows.py: rx, ca, gene, comp
PECOMP = SP / "pe_compartment.json"                 # leaf PE -> compartment, parsed from NCBI2Reactome_PE_Reactions
MAXPROD = 50    # an entity produced by more than this many reactions is treated as currency (ATP, ADP, H2O, Pi)


# ----------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------
def load_compartments():
    """PE stId -> compartment. Built from the local Reactome PE/reaction dump, so it costs no requests.

    The compartment lives only inside the display-name string ("PTK6 [cytosol]"), which is why this is parsed rather
    than read from a field. Coverage is every leaf entity that participates in a human reaction -- and crucially the
    dump lists leaf entities even when they participate via a complex, which is exactly what transport detection
    needs, since the cargo is usually buried inside a cargo:carrier complex.
    """
    if PECOMP.exists():
        return json.loads(PECOMP.read_text())["comp"]
    comp = {}
    with open(SP / "NCBI2Reactome_PE_Reactions.txt") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 8 or p[7] != "Homo sapiens":
                continue
            m = re.match(r"^(.*) \[([^\]]+)\]$", p[2])
            if m:
                comp[p[1]] = m.group(2)
    PECOMP.write_text(json.dumps({"comp": comp}))
    return comp


def expand(comp_map, gene_map):
    """PE stId -> set of gene symbols, and PE stId -> set of LEAF PE stIds (needed for compartments)."""
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


def load_ppi():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    N = len(names)
    ppi = set()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            ppi.add((names[min(a, b)], names[max(a, b)]))
    edge_of = collections.defaultdict(set)
    for a, b in ppi:
        edge_of[a].add(b)
        edge_of[b].add(a)
    S = set()
    for e in D.get("sig", []) or []:
        try:
            s_, t_ = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= s_ < N and 0 <= t_ < N and s_ != t_:
            S.add((names[s_], names[t_]))
    gt = {frozenset(p): p for p in S if (p[1], p[0]) not in S and p[1] in edge_of.get(p[0], ())}
    return ppi, edge_of, gt


# ----------------------------------------------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------------------------------------------
def score(cnt, label, gt, deg, n_ppi, blocks=None, drop=frozenset(), nboot=2000, seed=0):
    """Resolve a Counter of ordered votes per edge into arrows, then score against gt with a matched degree control.

    An edge is only oriented when every vote agrees. Edges with votes both ways are counted as ambiguous and
    discarded rather than resolved by majority -- a majority vote here would silently convert a genuine two-way
    relation (which the curated sources show is common for TF feedback pairs) into a confident single arrow.

    THE BINOMIAL INTERVAL IS THE WRONG INTERVAL HERE, AND NOT BY A LITTLE. These rules emit arrows in BLOCKS: one
    assembly reaction whose output complex gains k subunits emits |sub-complex| x k arrows that all point the same
    way and are all right or all wrong together. A single ribosomal step emits thousands. Treating those as
    independent Bernoulli draws overstates the effective sample size by roughly the mean block size, so a naive
    +/- sqrt(p(1-p)/n) can be off by an order of magnitude. The fix is to resample BLOCKS (reactions), not edges.
    Both intervals are printed so the size of the correction is visible rather than hidden.
    """
    orient = {k: v.most_common(1)[0][0] for k, v in cnt.items() if len(v) == 1}
    if drop:
        orient = {k: d for k, d in orient.items() if d[0] not in drop}
    amb = len(cnt) - len(orient)
    test = [(k, orient[k]) for k in orient if k in gt]
    n = len(test)
    acc = sum(1 for k, d in test if gt[k] == d) / n if n else float("nan")
    se = float(np.sqrt(acc * (1 - acc) / n)) if n else float("nan")
    dacc = (sum(1 for k, _ in test if deg.get(gt[k][0], 0) < deg.get(gt[k][1], 0)) / n) if n else float("nan")

    lo = hi = float("nan")
    nblk = 0
    if n and blocks:
        by_blk = collections.defaultdict(list)
        for k, d in test:
            for b in blocks.get(k, ("?",)):
                by_blk[b].append(1 if gt[k] == d else 0)
        keys = list(by_blk)
        nblk = len(keys)
        if nblk >= 2:
            rng = np.random.default_rng(seed)
            draws = np.empty(nboot)
            for i in range(nboot):
                pick = rng.integers(0, nblk, nblk)
                hits = [h for j in pick for h in by_blk[keys[j]]]
                draws[i] = np.mean(hits) if hits else np.nan
            lo, hi = (float(x) for x in np.nanpercentile(draws, [2.5, 97.5]))

    print(f"\n  {label}")
    print(f"    oriented   {len(orient):>7,}  ({len(orient)/n_ppi:.2%} of interactome)   ambiguous {amb:,}")
    if n >= 30:
        print(f"    vs SIGNOR  n={n:<6,} acc {acc:.4f}   matched-degree {dacc:.4f}   delta {acc-dacc:+.4f}")
        print(f"      naive binomial +/- {se:.4f}"
              + (f"   |   CLUSTER bootstrap over {nblk:,} reactions: [{lo:.4f}, {hi:.4f}]"
                 if nblk >= 2 else "   |   no block map, cluster CI unavailable"))
    elif n:
        print(f"    vs SIGNOR  n={n:<6,} acc {acc:.4f}   UNMEASURABLE (n<30), reported for completeness only")
    else:
        print(f"    vs SIGNOR  no overlap at all -- this rule orients edges SIGNOR has no opinion on")
    return {"orient": orient, "amb": amb, "n": n, "acc": acc, "se": se, "deg": dacc,
            "n_blocks": nblk, "boot_lo": lo, "boot_hi": hi,
            "n_oriented": len(orient), "frac": len(orient) / n_ppi}


def main():
    if not CACHE.exists():
        raise SystemExit(f"{CACHE} missing -- run colab/catalyst_arrows.py first (it does the Reactome fetch)")
    F = json.loads(CACHE.read_text())
    rx, ca, gene, comp = F["rx"], F["ca"], F["gene"], F["comp"]
    pe2sym, pe2leaf = expand(comp, gene)
    pecomp = load_compartments()
    ppi, edge_of, gt = load_ppi()
    deg = {g: len(v) for g, v in edge_of.items()}
    print(f"{len(rx):,} reactions | {len(pe2sym):,} entities mapped | PPI {len(ppi):,} | SIGNOR-orientable {len(gt):,}")

    def catalysts(r):
        """Gene symbols of the acting catalyst: activeUnit when Reactome names one, else the whole catalyst entity.

        Preferring activeUnit is load-bearing, not cosmetic. Reactome's catalyst physicalEntity is usually the entire
        ENZYME:SUBSTRATE complex, so expanding it emits enzyme->substrate and substrate->enzyme together and the edge
        cancels to ambiguous. activeUnit names the subunit that actually acts.
        """
        s, s_au = set(), set()
        for c in rx[r]["ca"]:
            d = ca.get(str(c))
            if not d:
                continue
            if d["au"]:
                u = set().union(*(pe2sym.get(x, set()) for x in d["au"])) if d["au"] else set()
                s |= u
                s_au |= u
            elif d["pe"]:
                s |= pe2sym.get(d["pe"], set())
        return s, s_au

    blocks = collections.defaultdict(set)   # edge -> reactions that voted on it, for the cluster bootstrap

    def add(cnt, a, b, blk=None):
        if a != b and b in edge_of.get(a, ()):
            cnt[frozenset((a, b))][(a, b)] += 1
            if blk is not None:
                blocks[frozenset((a, b))].add(blk)
            return True
        return False

    # ------------------------------------------------------------------------------------------------------------
    # RULE 1 -- ASSEMBLY ORDER FROM REACTIONS.  input complex I subset of output complex O  =>  members(I) precede
    # the newly added members(O)-members(I).  Derived from reaction structure, NOT from precedingEvent, so it is
    # independent of the source behind the existing 0.6547 rule.
    # ------------------------------------------------------------------------------------------------------------
    asm = collections.defaultdict(collections.Counter)
    asm_small = collections.defaultdict(collections.Counter)
    size_band = collections.Counter()
    for r, v in rx.items():
        outs = [(p, pe2sym.get(p, set())) for p in v["out"]]
        ins = [(p, pe2sym.get(p, set())) for p in v["in"]]
        for _, og in outs:
            if len(og) < 2:
                continue
            for _, ig in ins:
                if len(ig) < 1 or not ig < og:      # strict subset: something genuinely joined
                    continue
                added = og - ig
                if not added or len(added) > 6:     # a step that adds many subunits at once is not a resolved order
                    continue
                band = "2-4" if len(og) <= 4 else ("5-9" if len(og) <= 9 else ("10-19" if len(og) <= 19 else "20+"))
                for a in ig:
                    for b in added:
                        if add(asm, a, b, r):
                            size_band[band] += 1
                            if len(og) <= 9:
                                add(asm_small, a, b, r)

    # ------------------------------------------------------------------------------------------------------------
    # RULE 1b -- ASSEMBLY ORDER FROM NESTING.  Complex X contains sub-complex Y => members(Y) precede members(X)-Y.
    # This is the cheap proxy. It is scored against rule 1 as its own validation target, because whether nesting
    # encodes assembly order or is merely a curatorial grouping is exactly the question, and SIGNOR cannot answer it.
    # ------------------------------------------------------------------------------------------------------------
    nest = collections.defaultdict(collections.Counter)
    for x, kids in comp.items():
        xg = pe2sym.get(x, set())
        if len(xg) < 2:
            continue
        for k in kids:
            kg = pe2sym.get(k, set())
            if len(kg) >= 1 and kg < xg and len(xg - kg) <= 6:
                for a in kg:
                    for b in xg - kg:
                        add(nest, a, b, x)

    # ------------------------------------------------------------------------------------------------------------
    # RULE 2 -- TRANSPORT.  A gene whose leaf entity appears on the input side in one compartment and on the output
    # side in another has been translocated. The reaction's catalyst is the transporter; transporter -> cargo.
    # ------------------------------------------------------------------------------------------------------------
    def moved_genes(r):
        """Genes present on both sides of reaction r but reaching a compartment they were not already in.

        Compartment comes from the LEAF entity, not the enclosing complex, which is what makes this work: the cargo
        of a nuclear import is buried inside a cargo:importin complex, and only the leaf carries "[cytosol]" on the
        input side and "[nucleoplasm]" on the output side.
        """
        gin, gout = collections.defaultdict(set), collections.defaultdict(set)
        for side, acc in (("in", gin), ("out", gout)):
            for p in rx[r][side]:
                for lf in pe2leaf.get(p, ()):
                    c = pecomp.get(lf)
                    if c:
                        for g in gene.get(lf, ()):
                            acc[g].add(c)
        return {g for g in gin if g in gout and gout[g] - gin[g]}

    tr = collections.defaultdict(collections.Counter)
    cargo_seen, n_transport_rx = collections.Counter(), 0
    src_count = collections.Counter()
    transport_rx = set()
    for r in rx:
        moved = moved_genes(r)
        if not moved:
            continue
        transport_rx.add(r)
        n_transport_rx += 1
        cat, _ = catalysts(r)
        for g in moved:
            cargo_seen[g] += 1
        for a in cat:
            for b in moved:
                if add(tr, a, b, r):
                    src_count[a] += 1

    # ------------------------------------------------------------------------------------------------------------
    # RULE 3 -- MODIFICATION THEN TRANSPORT.  Reaction R1 (catalyst K) outputs entity E; reaction R2 (catalyst T)
    # consumes E and translocates it. K acted before T on the same cargo, so K -> T.
    # Entity flow is used rather than precedingEvent, deliberately: precedingEvent already backs the 0.6547 rule and
    # reusing it here would make this a re-slicing of an existing source rather than new evidence.
    # ------------------------------------------------------------------------------------------------------------
    # THE CURRENCY-MOLECULE TRAP, WHICH THIS RULE WALKS STRAIGHT INTO IF UNGUARDED. Linking R1 to R2 through a
    # shared entity is the classic way to build a dense, confident-looking, meaningless network: ATP is an output of
    # thousands of reactions and an input to most transport reactions, so joining on it would declare that every
    # kinase acts before every transporter. The guard is data-driven rather than a hand-written blacklist -- count
    # how many distinct reactions produce each entity and drop the promiscuous ones -- and the cap is swept so its
    # effect is visible rather than assumed.
    producers = collections.defaultdict(set)
    for r, v in rx.items():
        for p in v["out"]:
            producers[p].add(r)
    prod_n = {p: len(s) for p, s in producers.items()}
    hot = sorted(prod_n.values(), reverse=True)[:5]
    print(f"\n  entity promiscuity: most-produced entities appear as output of {hot} reactions; "
          f"cap MAXPROD={MAXPROD}")
    producers = {p: s for p, s in producers.items() if len(s) <= MAXPROD}
    mt = collections.defaultdict(collections.Counter)
    n_chain = 0
    for r2 in transport_rx:
        t_syms, _ = catalysts(r2)
        if not t_syms:
            continue
        for p in rx[r2]["in"]:
            for r1 in producers.get(p, ()):
                if r1 == r2:
                    continue
                k_syms, _ = catalysts(r1)
                for k in k_syms:
                    for t in t_syms:
                        if add(mt, k, t, r2):
                            n_chain += 1

    # ------------------------------------------------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------------------------------------------------
    n_ppi = len(ppi)
    R = {}
    R["ASSEMBLY-rxn"] = score(asm, "ASSEMBLY ORDER (from assembly reactions)", gt, deg, n_ppi, blocks)
    R["ASSEMBLY-small"] = score(asm_small, "ASSEMBLY ORDER (complexes of <=9 subunits only)", gt, deg, n_ppi, blocks)
    R["ASSEMBLY-nest"] = score(nest, "ASSEMBLY ORDER (from complex nesting)", gt, deg, n_ppi, blocks)
    R["TRANSPORT"] = score(tr, "TRANSPORT (transporter -> cargo)", gt, deg, n_ppi, blocks)
    hubs = frozenset(g for g, _ in src_count.most_common(10))
    R["TRANSPORT-nohub"] = score(tr, f"TRANSPORT excluding top-10 source hubs {sorted(hubs)}", gt, deg, n_ppi, blocks,
                                 drop=hubs)
    R["MOD-THEN-TRANSPORT"] = score(mt, "MODIFIER -> TRANSPORTER (same cargo, consecutive)", gt, deg, n_ppi, blocks)

    # ---- confound reporting ----
    print(f"\n  CONFOUND CHECKS")
    print(f"    assembly arrows by complex size: {dict(size_band.most_common())}")
    tot = sum(src_count.values())
    top = sum(k for _, k in src_count.most_common(10))
    print(f"    transport: {n_transport_rx:,} translocation reactions, {len(cargo_seen):,} distinct cargo genes")
    print(f"    transport: top-10 sources emit {top:,}/{tot:,} arrows"
          + (f" ({top/tot:.1%}) -- {[g for g,_ in src_count.most_common(10)]}" if tot else ""))
    print(f"    mod->transport: {n_chain:,} chain votes over {len(transport_rx):,} translocation reactions")

    # ---- does nesting encode assembly order? the right question for rule 1b ----
    a_or, n_or = R["ASSEMBLY-rxn"]["orient"], R["ASSEMBLY-nest"]["orient"]
    shared = set(a_or) & set(n_or)
    nest_agree = sum(1 for k in shared if a_or[k] == n_or[k]) / len(shared) if shared else float("nan")
    print(f"\n    nesting vs reaction-derived assembly order: {len(shared):,} shared edges, "
          + (f"agreeing {nest_agree:.1%}" if shared else "no overlap"))

    # ---- overlap with what is already in the directed network ----
    DN = json.load(open(OUT / "directed_network.json"))
    known = {frozenset((a, b)): (a, b) for a, b, p, t in DN["edges"] if p != "undirected"}
    for name, r in R.items():
        new = len(set(r["orient"]) - set(known))
        print(f"    {name:<20} adds {new:>7,} edges not already directed")
        r["new_vs_existing"] = new

    best = max((r for r in R.values() if r["n"] >= 30), key=lambda r: r["acc"] - 2 * r["se"], default=None)
    best_name = next((k for k, v in R.items() if v is best), None)
    payload = {"n_ppi": n_ppi, "n_gt": len(gt),
               "results": {k: {kk: vv for kk, vv in v.items() if kk != "orient"} for k, v in R.items()},
               "size_band": dict(size_band), "transport_reactions": n_transport_rx,
               "transport_top_sources": src_count.most_common(10),
               "nesting_vs_reaction_agreement": nest_agree, "best": best_name,
               "edges": {k: [[d[0], d[1]] for d in v["orient"].values()] for k, v in R.items()}}
    json.dump(payload, open(OUT / "assembly_transport_arrows.json", "w"))
    print(f"\n  -> {OUT/'assembly_transport_arrows.json'}")
    return R


if __name__ == "__main__":
    main()
