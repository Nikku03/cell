"""THE RIGHT QUESTION IS NOT "WHICH SIDE IS EACH PROTEIN ON", IT IS "WHICH PROTEIN CHANGED".

THIS OVERTURNS THE CENTRAL NEGATIVE RESULT OF THE LAST FOUR MODULES, so the reasoning is worth stating exactly.

catalyst_arrows.py measured that of the 36,415 PPI edges whose endpoints share a Reactome reaction, only 95 split
across input and output, while 1,593 are BOTH inputs. The conclusion drawn was: a binding step is A + B -> AB, both
partners are inputs, and the step carries no internal direction. That part is true. The part that was WRONG was
treating the enormous "appears on both sides" bucket as ambiguity to be discarded. That bucket is where all the
signal lives, and here is why.

Reactome mints a NEW physical-entity id for every modified form of a protein. In a phosphorylation the substrate is
present on both sides -- as `AKT1` on the input side and `p-T308,S473-AKT1` on the output side -- while the kinase
appears as the SAME entity on both sides, because acting on something does not change you. So within one reaction:

    transformed(R) = genes whose set of leaf entities DIFFERS between input and output   <- the acted-upon
    unchanged(R)   = genes whose set of leaf entities is IDENTICAL on both sides         <- the actor

and the arrow is unchanged -> transformed. At gene level a protein being "both input and output" is the norm
(the earlier count of 1,593 was measured at top-level entity granularity, before complex expansion); at LEAF entity
level a gene is never literally both, because the modified form has its own stId. The old rule and this one are the
same fact seen from opposite sides.

WHY THIS BEATS THE CATALYST ROUTE IT REPLACES. It recovers kinase -> substrate WITHOUT needing the catalystActivity
annotation, which requires a second resolution fetch and is present on only a minority of reactions. It therefore
reaches several times as many edges.

THE STATED MECHANISM IS ITSELF A FALSIFIABLE PREDICTION, AND THIS MODULE TESTS IT RATHER THAN ASSERTING IT. The claim
is that the unchanged partner IS the enzyme. If that is true, then wherever Reactome independently annotates a
catalyst for the same reaction, that catalyst should land on the UNCHANGED side, never the transformed side. The
measured agreement rate is printed below. A low rate would not change the accuracy figures, but it would mean the
rule works for some reason other than the one given here -- which is worth knowing, so the number is reported
whether or not it is flattering.

MEASURED, WITH THE CONTROLS THIS PROJECT REQUIRES -- the degree rule computed ON THE SAME EDGES (because a walk's
direction on an undirected graph is provably the degree ratio), and a cluster bootstrap over REACTIONS (because one
reaction emits many arrows that are right or wrong together, so a binomial interval over edges is far too narrow):

    ALL          3,822 edges   n=657   0.8326 +/- 0.0146   degree 0.5373   boot [0.8094, 0.8659]
    chemical     3,128 edges   n=618   0.8495 +/- 0.0144   degree 0.5340   boot [0.8205, 0.8790]
    translocation  900 edges   n= 86   0.6860 +/- 0.0500   degree 0.5465   boot [0.6249, 0.8033]

A CORRECTION THIS FORCES ON EARLIER WRITE-UPS. Several modules quote 0.783 as a "curated-vs-curated ceiling", from
the rate at which two curated direction sources agree with each other. This rule exceeds it. That number was never a
ceiling on accuracy -- it was the agreement rate between two SPECIFIC sources over the edges they both cover, and
disagreement there is dominated by TF pairs where a kinase acts on a TF while that TF drives the kinase's gene, so
both arrows are true. Calling it a ceiling was an overstatement and is withdrawn. The honest reference points are:
chance 0.500, degree rule 0.5664 (0.5373 on this rule's own edges), global precedence 0.6547.

WHY THE TRANSLOCATION SPLIT IS REPORTED SEPARATELY RATHER THAN FOLDED IN. A compartment change also makes the leaf
entity differ, so translocation satisfies the rule too -- but it is a different relation (a carrier moving a cargo,
not an enzyme modifying a substrate) and it scores materially lower. Mixing them would let a 0.85 headline be quoted
over edges that are really 0.69. This is the same discipline applied in pathway_inputs.py, where merging a weaker
source into a stronger one diluted the stronger one.
"""
import collections
import json
import re
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
TREE = SP / "entity_tree.json"          # PE -> {sc, cls, nm, kids, up}: the recursive Reactome entity tree
UP2G = SP / "reactome/up2gene.tsv"      # UniProt accession -> gene symbol
RXIO = SP / "reaction_io.json"          # reaction -> top-level input/output physical entities
NBOOT = 2000


def load_tree():
    T = json.loads(TREE.read_text())
    up2g = {}
    for line in UP2G.read_text().splitlines():
        p = line.split("\t")
        if len(p) >= 2 and p[0] and p[1]:
            up2g[p[0]] = p[1]
    memo = {}

    def leaves(pe, stack=()):
        """Leaf entities under pe that carry a UniProt-mapped gene.

        Currency molecules need no exclusion list here: ATP, ADP, H2O and phosphate are SimpleEntity records with no
        referenceEntity, so they carry no UniProt accession and drop out for free. That is what makes this rule
        immune to the join-on-ATP trap that wrecks naive entity-flow networks.
        """
        if pe in memo:
            return memo[pe]
        if pe in stack:                                  # cycle guard; do not assume the tree is a DAG
            return set()
        d = T.get(pe)
        if d is None:
            return set()
        out = {pe} if (d.get("up") and up2g.get(d["up"])) else set()
        for k in d.get("kids") or []:
            out |= leaves(k, stack + (pe,))
        memo[pe] = out
        return out

    return T, up2g, leaves


def load_graph():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    N = len(names)
    edge_of = collections.defaultdict(set)
    npp = 0
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            edge_of[names[a]].add(names[b])
            edge_of[names[b]].add(names[a])
            npp += 1
    S = set()
    for e in D.get("sig", []) or []:
        try:
            s_, t_ = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= s_ < N and 0 <= t_ < N and s_ != t_:
            S.add((names[s_], names[t_]))
    # ground truth excludes pairs SIGNOR records BOTH ways: those are real two-way relations (a kinase acting on a
    # TF that drives its gene), not annotation noise, and scoring a single arrow against them is meaningless.
    gt = {frozenset(p): p for p in S if (p[1], p[0]) not in S and p[1] in edge_of.get(p[0], ())}
    return edge_of, {g: len(v) for g, v in edge_of.items()}, gt, npp


def score(votes, blocks, label, gt, deg, n_ppi):
    orient = {k: c.most_common(1)[0][0] for k, c in votes.items() if len(c) == 1}
    amb = len(votes) - len(orient)
    test = [(k, orient[k]) for k in orient if k in gt]
    n = len(test)
    acc = sum(1 for k, d in test if gt[k] == d) / n if n else float("nan")
    se = float(np.sqrt(acc * (1 - acc) / n)) if n else float("nan")
    dacc = (sum(1 for k, _ in test if deg.get(gt[k][0], 0) < deg.get(gt[k][1], 0)) / n) if n else float("nan")
    lo = hi = float("nan")
    by = collections.defaultdict(list)
    for k, d in test:
        for b in blocks.get(k, ("?",)):
            by[b].append(1 if gt[k] == d else 0)
    keys = list(by)
    if len(keys) > 1:
        rng = np.random.default_rng(0)
        draws = np.array([np.mean([h for j in rng.integers(0, len(keys), len(keys)) for h in by[keys[j]]])
                          for _ in range(NBOOT)])
        lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    print(f"\n  {label}")
    print(f"    oriented {len(orient):>7,} ({len(orient)/n_ppi:.2%} of interactome)   ambiguous {amb:,}")
    if n:
        print(f"    n={n:<6,} accuracy {acc:.4f} +/- {se:.4f}    MATCHED degree {dacc:.4f}   delta {acc-dacc:+.4f}")
        print(f"    cluster bootstrap over {len(keys):,} reactions: [{lo:.4f}, {hi:.4f}]")
    return {"n_oriented": len(orient), "ambiguous": amb, "n": n, "acc": acc, "se": se, "deg": dacc,
            "boot_lo": lo, "boot_hi": hi, "n_blocks": len(keys), "orient": orient}


def main():
    T, up2g, leaves = load_tree()
    edge_of, deg, gt, n_ppi = load_graph()
    io = json.loads(RXIO.read_text())["io"]
    print(f"entity tree {len(T):,} | uniprot->gene {len(up2g):,} | reactions {len(io):,}")
    print(f"PPI edges {n_ppi:,} | unambiguous SIGNOR-on-PPI ground truth {len(gt):,}")

    def gene_of(leaf):
        return up2g.get((T.get(leaf) or {}).get("up"))

    def stem(leaf):
        """Display name with the [compartment] suffix removed. The regex is greedy and END-ANCHORED on purpose:
        entity names legitimately contain colons, parentheses and internal brackets
        ("DFF:associated with the importin-alpha:importin-beta complex [cytosol]"), and a lazy or unanchored pattern
        would mangle them. Verified against all 31,108 distinct display names in the local dump: 0 failures."""
        nm = (T.get(leaf) or {}).get("nm") or ""
        m = re.match(r"^(.*) \[([^\]]+)\]$", nm)
        return m.group(1) if m else nm

    kinds = ("ALL", "CHEM", "TRANSLOC")
    V = {k: collections.defaultdict(collections.Counter) for k in kinds}
    B = {k: collections.defaultdict(set) for k in kinds}
    n_fire = n_none = 0
    for r, v in io.items():
        pin, pout = collections.defaultdict(set), collections.defaultdict(set)
        for side, acc in (("in", pin), ("out", pout)):
            for p in v.get(side, []):
                for lf in leaves(p):
                    g = gene_of(lf)
                    if g:
                        acc[g].add(lf)
        transformed = {g for g in pin if g in pout and pin[g] != pout[g]}
        if not transformed:
            # A reaction with nothing transformed is a pure binding, dissociation or assembly event. It genuinely
            # has no internal direction, and forcing one is exactly what made the earlier input->output rule score
            # BELOW chance. Emitting nothing here is the whole point.
            n_none += 1
            continue
        unchanged = {g for g in pin if g in pout and pin[g] == pout[g]}
        n_fire += 1
        for b in transformed:
            kind = "CHEM" if {stem(l) for l in pin[b]} != {stem(l) for l in pout[b]} else "TRANSLOC"
            for a in unchanged:
                if a == b or b not in edge_of.get(a, ()):
                    continue
                for tag in ("ALL", kind):
                    V[tag][frozenset((a, b))][(a, b)] += 1
                    B[tag][frozenset((a, b))].add(r)

    print(f"\nreactions with something transformed: {n_fire:,}; with nothing transformed (emit no arrow): {n_none:,}")
    R = {k: score(V[k], B[k], f"UNCHANGED -> TRANSFORMED  [{k}]", gt, deg, n_ppi) for k in kinds}
    print(f"\n    reference points: chance 0.500 | degree rule 0.5664 | global precedence 0.6547")

    # how much of this is genuinely new relative to what the directed network already has?
    DN = json.load(open(OUT / "directed_network.json"))
    known = {frozenset((a, b)): (a, b) for a, b, p, t in DN["edges"] if p != "undirected"}
    for k in kinds:
        o = R[k]["orient"]
        both = set(o) & set(known)
        ag = sum(1 for e in both if o[e] == known[e]) / len(both) if both else float("nan")
        R[k]["new_edges"] = len(set(o) - set(known))
        R[k]["overlap"] = len(both)
        R[k]["overlap_agreement"] = ag
        print(f"    {k:<9} adds {R[k]['new_edges']:>7,} edges not already directed; "
              f"overlaps {len(both):,}" + (f", agreeing {ag:.1%}" if both else ""))

    # ---- DOES THIS RULE ACTUALLY READ CATALYSIS, OR ONLY CORRELATE WITH IT? ----
    # The mechanistic claim is that the unchanged partner in a modification reaction IS the enzyme. If so, then
    # wherever Reactome independently annotates a catalyst, the catalyst should be the unchanged partner. That is a
    # falsifiable prediction and it is checked here rather than asserted. Low agreement would mean the rule works for
    # some other reason and the stated mechanism is wrong -- which would matter even though accuracy is unaffected.
    CAT = SP / "catalyst_arrows_cache.json"
    cat_agree = None
    if CAT.exists():
        F = json.loads(CAT.read_text())
        rx, ca = F["rx"], F["ca"]
        same = diff = norule = 0
        for r, v in rx.items():
            if not v.get("ca") or r not in io:
                continue
            cs = set()
            for c in v["ca"]:
                d = ca.get(str(c))
                if not d:
                    continue
                src = d["au"] or ([d["pe"]] if d["pe"] else [])
                for x in src:
                    cs |= {g for g in (gene_of(l) for l in leaves(x)) if g}
            if not cs:
                continue
            pin, pout = collections.defaultdict(set), collections.defaultdict(set)
            for side, acc in (("in", pin), ("out", pout)):
                for pe in io[r].get(side, []):
                    for lf in leaves(pe):
                        g = gene_of(lf)
                        if g:
                            acc[g].add(lf)
            tr = {g for g in pin if g in pout and pin[g] != pout[g]}
            un = {g for g in pin if g in pout and pin[g] == pout[g]}
            if not tr:
                norule += 1
                continue
            # the annotated catalyst should sit on the UNCHANGED side, not the transformed side
            same += len(cs & un) > 0
            diff += (len(cs & un) == 0) and (len(cs & tr) > 0)
        tot = same + diff
        cat_agree = same / tot if tot else float("nan")
        print(f"\n  MECHANISM CHECK -- is the unchanged partner really the enzyme?")
        print(f"    reactions with an annotated catalyst AND a transformed set: {tot:,}")
        print(f"    catalyst is on the UNCHANGED side (rule's claim): {same:,}  ({cat_agree:.1%})")
        print(f"    catalyst is on the TRANSFORMED side (contradicts): {diff:,}")
        print(f"    reactions with a catalyst but nothing transformed (rule stays silent): {norule:,}")

    a = R["ALL"]
    verdict = (
        f"THE QUESTION WAS WRONG, NOT THE DATA. Four modules concluded that a Reactome reaction cannot orient a PPI "
        f"edge, because only 95 of 36,415 co-participating edges split across input and output while 1,593 sit on "
        f"the same side. The error was treating the huge 'appears on both sides' bucket as ambiguity. Reactome mints "
        f"a NEW entity id for every modified form, so in a phosphorylation the SUBSTRATE appears on both sides as "
        f"two different entities while the KINASE appears as the same entity twice. Comparing leaf-entity sets per "
        f"gene therefore separates the acted-upon (transformed) from the actor (unchanged), and the arrow is "
        f"unchanged -> transformed. "
        f"MEASURED: {a['n_oriented']:,} PPI edges oriented ({a['n_oriented']/n_ppi:.2%} of the interactome), "
        f"accuracy {a['acc']:.4f} +/- {a['se']:.4f} on n={a['n']:,}, against a MATCHED degree baseline of "
        f"{a['deg']:.4f} on the same edges -- a gain of {a['acc']-a['deg']:+.4f}, which is far too large to be the "
        f"degree rule in disguise. The cluster bootstrap over {a['n_blocks']:,} reactions, which is the correct "
        f"interval because one reaction emits many arrows that are right or wrong together, gives "
        f"[{a['boot_lo']:.4f}, {a['boot_hi']:.4f}]. "
        f"Splitting by what changed: chemical modification {R['CHEM']['acc']:.4f} (n={R['CHEM']['n']:,}) versus "
        f"translocation {R['TRANSLOC']['acc']:.4f} (n={R['TRANSLOC']['n']:,}) -- reported separately, because "
        f"folding them together would let the higher number be quoted over edges that do not earn it. "
        f"A CORRECTION THIS FORCES: earlier modules called 0.783 a 'curated-vs-curated ceiling'. This exceeds it. "
        f"That figure was the agreement rate between two specific curated sources over the edges they share, where "
        f"disagreement is dominated by TF pairs whose two arrows are both true. It was never a bound on accuracy, "
        f"and describing it as a ceiling is withdrawn. "
        f"WHAT IT STILL CANNOT SAY: Reactome describes canonical human biology, not K562; Reactome and SIGNOR "
        f"curators read overlapping literature, so this is not fully independent validation; and coverage remains "
        f"the binding constraint -- only {len(gt):,} PPI edges ({len(gt)/n_ppi:.2%}) have any unambiguous curated "
        f"direction at all, so every accuracy here describes that slice, not the network.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"n_ppi": n_ppi, "n_gt": len(gt), "reactions_firing": n_fire, "reactions_silent": n_none,
               "results": {k: {kk: vv for kk, vv in R[k].items() if kk != "orient"} for k in kinds},
               "catalyst_side_agreement": cat_agree, "verdict": verdict,
               "edges": {k: [[d[0], d[1]] for d in R[k]["orient"].values()] for k in kinds}},
              open(OUT / "transformed_partner_arrows.json", "w"))
    print(f"\n  -> {OUT/'transformed_partner_arrows.json'}")


if __name__ == "__main__":
    main()
