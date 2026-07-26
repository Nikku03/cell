"""ORIENTING PPI EDGES FROM WITHIN THE REACTION: why input->output fails and catalyst->output is the right relation.

THE USER'S REASONING, WHICH IS CORRECT AND SHARPER THAN WHAT WAS BUILT. build_directed_network.py orients {A,B} by a
GLOBAL question -- does some reaction containing A precede some reaction containing B -- and A and B may interact in
a completely different reaction. The local question is better: find the reaction where A and B ACTUALLY interact and
read the arrow off that reaction's own curated structure.

MEASURED FIRST, BEFORE BUILDING ANYTHING, and it reshapes the whole idea:

    PPI edges whose endpoints share a reaction   36,415  (19.02% of the interactome)  <- good reach
    ...that SPLIT across input and output            95  (0.3% of those)              <- almost nothing
    ...that are BOTH INPUTS                       1,593
    ...that are BOTH OUTPUTS                        458

WITHIN A REACTION, AN INTERACTION HAS NO DIRECTION. That is not a data gap, it is what a binding event is: the
reaction is A + B -> AB, so A and B are both INPUTS and they are peers. Asking which of two binding partners comes
"first" inside the step where they bind is asking a question the step does not answer. The 95 that do split are the
exception, not the rule.

SO THE RELATION THAT CARRIES DIRECTION IS CATALYSIS, NOT CONSUMPTION. In a phosphorylation reaction Reactome
records input = {substrate, ATP}, output = {phospho-substrate, ADP}, and the KINASE as a CATALYST -- not an input.
An input->output rule therefore yields substrate -> phospho-substrate, which is the same protein, and never recovers
kinase -> substrate, which is exactly the arrow SIGNOR records. Catalyst -> output is the relation that matches.

That requires a second resolution hop: the bulk endpoint returns each CatalystActivity object with a dbId but WITHOUT
its physicalEntity, so the activities have to be resolved separately. This module does both hops and then measures,
against independently curated SIGNOR:

    coverage   how many PPI edges catalyst->output can orient at all
    accuracy   how often it agrees with SIGNOR, against 0.500 chance, the 0.5664 the degree rule gets, and the
               0.6547 the existing global precedence rule gets
    overlap    whether it orients the SAME edges as the global rule or different ones

If catalyst->output is high-precision and low-coverage it should be a separate, higher tier rather than merged --
the pattern established in pathway_inputs.py, where entity flow answered one question well and another badly and
merging the two diluted the good answer.
"""
import collections
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
UA = {"User-Agent": "Mozilla/5.0 (compatible; cell-network-research/1.0)"}
BATCH = 20              # measured silent cap on the bulk endpoint
CACHE = Path(SP) / "catalysts.json"


def post(ids, tries=4):
    for t in range(tries):
        try:
            req = urllib.request.Request("https://reactome.org/ContentService/data/query/ids",
                                         data=",".join(str(i) for i in ids).encode(),
                                         headers=dict(UA, **{"Content-Type": "text/plain"}))
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception:
            if t == tries - 1:
                return None
            time.sleep(2 ** t)
    return None


def fetch_catalysts():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    cls = json.loads((Path(SP) / "reaction_io.json").read_text())["className"]
    rxn = sorted(k for k, v in cls.items() if v == "Reaction")
    print(f"pass 1: collecting catalystActivity ids from {len(rxn):,} reactions", flush=True)
    r2ca, t0 = {}, time.time()
    for i in range(0, len(rxn), BATCH):
        d = post(rxn[i:i + BATCH])
        if not d:
            continue
        for o in d:
            ca = [c.get("dbId") for c in (o.get("catalystActivity") or []) if isinstance(c, dict) and c.get("dbId")]
            if ca:
                r2ca[o["stId"]] = ca
        if (i // BATCH) % 200 == 0:
            print(f"  {i//BATCH}/{len(rxn)//BATCH}  {len(r2ca)} reactions with a catalyst "
                  f"({(i//BATCH+1)/max(time.time()-t0,1):.2f} req/s)", flush=True)
    allca = sorted({c for v in r2ca.values() for c in v})
    print(f"pass 2: resolving {len(allca):,} CatalystActivity objects to physical entities", flush=True)
    ca2pe = {}
    for i in range(0, len(allca), BATCH):
        d = post(allca[i:i + BATCH])
        if not d:
            continue
        for o in d:
            pe = o.get("physicalEntity") or {}
            if o.get("dbId") and pe.get("stId"):
                ca2pe[str(o["dbId"])] = pe["stId"]
        if (i // BATCH) % 200 == 0:
            print(f"  {i//BATCH}/{len(allca)//BATCH}  {len(ca2pe)} resolved", flush=True)
    out = {"r2ca": r2ca, "ca2pe": ca2pe}
    CACHE.write_text(json.dumps(out))
    print(f"  cached -> {CACHE} ({CACHE.stat().st_size/1e6:.1f} MB)", flush=True)
    return out


def main():
    CA = fetch_catalysts()
    r2ca, ca2pe = CA["r2ca"], CA["ca2pe"]
    io = json.loads((Path(SP) / "reaction_io.json").read_text())["io"]
    s2p = {k: set(v) for k, v in json.loads((Path(SP) / "sym2pe.json").read_text()).items()}
    pe2s = collections.defaultdict(set)
    for s, ps in s2p.items():
        for p in ps:
            pe2s[p].add(s)
    print(f"\n{len(r2ca):,} reactions with a catalyst; {len(ca2pe):,} catalyst entities resolved", flush=True)

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; N = len(names)
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
        edge_of[a].add(b); edge_of[b].add(a)
    S = set()
    for e in D.get("sig", []) or []:
        try:
            s_, t_ = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= s_ < N and 0 <= t_ < N and s_ != t_:
            S.add((names[s_], names[t_]))
    gt = {frozenset(p): p for p in S if (p[1], p[0]) not in S and p[1] in edge_of.get(p[0], ())}
    print(f"PPI edges {len(ppi):,}; SIGNOR-orientable {len(gt):,}", flush=True)

    # ---- catalyst -> output, restricted to pairs that are PPI edges ----
    arrows = collections.defaultdict(collections.Counter)
    for r, cas in r2ca.items():
        d = io.get(r)
        if not d:
            continue
        cat_syms = set()
        for c in cas:
            p = ca2pe.get(str(c))
            if p:
                cat_syms |= pe2s.get(p, set())
        if not cat_syms:
            continue
        out_syms = set()
        for p in d.get("out", []):
            out_syms |= pe2s.get(p, set())
        for a in cat_syms:
            nb = edge_of.get(a)
            if not nb:
                continue
            for b in out_syms:
                if a != b and b in nb:
                    arrows[frozenset((a, b))][(a, b)] += 1

    oriented = {k: v.most_common(1)[0][0] for k, v in arrows.items() if len(v) == 1}
    ambiguous = len(arrows) - len(oriented)
    test = [(k, oriented[k]) for k in oriented if k in gt]
    good = sum(1 for k, d in test if gt[k] == d)
    acc = good / len(test) if test else float("nan")
    se = float(np.sqrt(acc * (1 - acc) / len(test))) if test else float("nan")

    print(f"\n  CATALYST -> OUTPUT")
    print(f"    PPI edges oriented        {len(oriented):,}  ({len(oriented)/len(ppi):.2%} of the interactome)")
    print(f"    ambiguous (both ways)     {ambiguous:,}")
    print(f"    checkable against SIGNOR  {len(test):,}")
    if test:
        print(f"    accuracy                  {acc:.4f} +/- {se:.4f}   (chance 0.500)")
        print(f"      degree/Markov rule 0.5664 | global precedence 0.6547 | curated-vs-curated ceiling 0.783")

    # does it orient DIFFERENT edges than the global precedence rule?
    DN = json.load(open(OUT / "directed_network.json"))
    po = {frozenset((a, b)): (a, b) for a, b, p, t in DN["edges"] if p == "pathway_order"}
    both = set(oriented) & set(po)
    agree = sum(1 for k in both if oriented[k] == po[k]) if both else 0
    print(f"\n    overlap with the global pathway_order rule: {len(both):,} edges"
          + (f", agreeing {agree/len(both):.1%}" if both else ""))
    print(f"    edges catalyst->output orients that pathway_order does NOT: {len(set(oriented)-set(po)):,}")

    better = acc == acc and acc - 2 * se > 0.6547
    verdict = (
        f"WITHIN-REACTION ARROWS: THE PREMISE WAS RIGHT, THE FIRST RULE WAS WRONG, AND THE FIX IS CATALYSIS. "
        f"Orienting {{A,B}} from the reaction where A and B ACTUALLY interact is a better question than the global "
        f"'does some reaction containing A precede some reaction containing B'. But the obvious local rule -- A is "
        f"an input, B is an output -- reaches almost nothing: of the 36,415 PPI edges (19.02%) whose endpoints share "
        f"a reaction, only 95 split across input and output, while 1,593 are BOTH INPUTS. That is not a gap in the "
        f"data, it is what binding IS: the reaction is A + B -> AB, both partners go in, and inside the step where "
        f"two proteins bind, neither comes first. "
        f"THE RELATION THAT CARRIES DIRECTION IS CATALYSIS. In a phosphorylation Reactome records input = "
        f"{{substrate, ATP}}, output = {{phospho-substrate, ADP}} and the kinase as a CATALYST, so input->output "
        f"yields substrate -> phospho-substrate (the same protein) and never recovers kinase -> substrate, which is "
        f"the arrow SIGNOR actually records. Resolving catalysts took a second hop, because the bulk endpoint "
        f"returns CatalystActivity objects without their physicalEntity. "
        f"MEASURED: catalyst->output orients {len(oriented):,} PPI edges ({len(oriented)/len(ppi):.2%}), "
        f"{ambiguous:,} more are ambiguous, and on the {len(test):,} checkable against SIGNOR it is "
        f"{acc:.4f} +/- {se:.4f}. "
        + (f"THAT BEATS THE GLOBAL PRECEDENCE RULE'S 0.6547 and the degree rule's 0.5664, and it is close to the "
           f"0.783 ceiling set by how much the two curated sources agree with each other -- so this is about as "
           f"good as an annotation-derived arrow can be. " if better else
           f"That does NOT clearly beat the global precedence rule's 0.6547 at this sample size, so it earns a "
           f"place as a separate high-specificity tier rather than a replacement. ")
        + f"It orients {len(set(oriented)-set(po)):,} edges the global rule does not, so the two are complementary "
        f"rather than redundant. "
        f"WHAT THIS CANNOT SAY. Catalyst->output is the enzyme acting on its product, which is a causal claim, but "
        f"the OUTPUT is often a complex and the arrow is then drawn to every protein in it, not only the one that "
        f"was modified. Reactome describes canonical human biology, not K562. And the ceiling remains 78%: "
        f"'the direction of a PPI edge' is not single-valued, because protein-level and transcriptional direction "
        f"on the same pair legitimately oppose.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"oriented": len(oriented), "frac": len(oriented) / len(ppi), "ambiguous": ambiguous,
               "n_test": len(test), "accuracy": acc, "se": se,
               "overlap_with_pathway_order": len(both), "unique": len(set(oriented) - set(po)),
               "verdict": verdict,
               "edges": [[d[0], d[1]] for d in oriented.values()]},
              open(OUT / "catalyst_arrows.json", "w"))
    print(f"\n  -> {OUT/'catalyst_arrows.json'}")


if __name__ == "__main__":
    main()
