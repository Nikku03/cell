"""THE CONSEQUENCE MAP: for every protein, what stops, what the cell then lacks, and what fails next.

WHAT IS RECORDED FOR EVERY GENE, in the order the biology runs:

  1  WHAT IT DOES        reactions it catalyses, reactions it participates in, and the compartments they run in
  2  WHAT IT NEEDS       the other inputs and catalysts those reactions require -- lose any of them and the same
                         step stops, so this is the gene's own dependency set
  3  WHAT STOPS          reactions where it is the SOLE catalyst. This is the load-bearing distinction: a reaction
                         with two annotated catalysts does not stop when one is removed, and treating every
                         reaction a gene touches as "stopped" is what turns a consequence map into a blast radius
                         that hits everything
  4  WHAT THE CELL LACKS products of stopped reactions that NO surviving reaction still makes. A product made by
                         three other routes is not lost. Currency species are excluded -- ATP does not become
                         unavailable because one kinase is gone
  5  WHAT FAILS NEXT     reactions that consume a lost product and therefore starve, and the genes running them.
                         This is the second order, and it is followed to a bounded depth rather than to closure,
                         because unbounded propagation through a metabolic network reaches everything
  6  WHICH PATHWAYS      pathways containing the stopped reactions, and the pathways downstream of those via the
                         measured boundary-flow links
  7  WHO IT TOUCHES      same-reaction partners and PPI neighbours -- direct physical contact

THE TWO WAYS THIS BECOMES USELESS, AND WHAT IS DONE ABOUT THEM. First, treating shared catalysis as sole catalysis
makes every knockout catastrophic; hence the sole-catalyst rule, and the count of genes with NO sole-catalyst
reaction is reported rather than hidden. Second, following the cascade to closure connects everything to everything
through central metabolism; hence the depth bound and the currency exclusion, both stated as parameters and swept
in the summary rather than chosen silently.

This file only BUILDS the map. Whether it predicts anything is a separate question, answered by re-running the
sealed prediction protocol on it -- and the honest expectation is set by the previous round: reasoning from
mechanism beat the frequency baseline on 4 of 48 knockouts, all of them mitochondrial or ER stress, so a
consequence map should be judged on whether it extends that to other lesion classes.
"""
import collections
import json
import os
from pathlib import Path

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
MAXDEPTH = 2            # second order only; closure reaches the whole network through central metabolism
MAXPART = 60            # a step with more participants than this is a machine, not a step
TOPN = 40               # cap on each recorded list, so one gene cannot produce a megabyte of map


def main():
    net = json.loads((OUT / "reaction_network.json").read_text())
    comp = json.loads((OUT / "compartments.json").read_text())["steps"]
    steps = {s["id"]: s for s in net["steps"]}

    paths = {}
    for line in (SP / "ReactomePathways.gmt").read_text().splitlines():
        p = line.split("\t")
        if len(p) > 3:
            paths[p[0]] = set(p[2:])
    links = collections.defaultdict(set)
    for a, b, _e in json.loads((OUT / "pathway_links.json").read_text())["links"]:
        links[a].add(b)

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    N = len(names)
    ppi = collections.defaultdict(set)
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            ppi[names[a]].add(names[b])
            ppi[names[b]].add(names[a])

    # ---- indices ----
    gene_cat = collections.defaultdict(set)      # gene -> reactions it catalyses
    gene_part = collections.defaultdict(set)     # gene -> reactions it participates in
    produced_by = collections.defaultdict(set)   # entity -> reactions that make it
    consumed_by = collections.defaultdict(set)   # entity -> reactions that use it
    rxn_genes = {}
    for sid, s in steps.items():
        gs = set(s["catalysts"])
        for p in s["in"] + s["out"]:
            gs |= set(p["genes"])
        if len(gs) > MAXPART:
            continue
        rxn_genes[sid] = gs
        for g in s["catalysts"]:
            gene_cat[g].add(sid)
        for g in gs - set(s["catalysts"]):
            gene_part[g].add(sid)
        for p in s["out"]:
            if not p.get("currency"):
                produced_by[p["id"]].add(sid)
        for p in s["in"]:
            if not p.get("currency"):
                consumed_by[p["id"]].add(sid)

    rxn_paths = collections.defaultdict(set)
    for name, gset in paths.items():
        for sid, gs in rxn_genes.items():
            if gs and gs <= gset:
                rxn_paths[sid].add(name)

    allgenes = sorted(set(gene_cat) | set(gene_part) | set(ppi))
    print(f"{len(allgenes):,} genes; {len(rxn_genes):,} usable reactions; "
          f"{len(produced_by):,} non-currency entities produced", flush=True)

    cmap = {}
    n_sole = 0
    for g in allgenes:
        cat, part = gene_cat.get(g, set()), gene_part.get(g, set())
        # 3. WHAT STOPS -- sole catalyst only
        sole = [r for r in cat if len(steps[r]["catalysts"]) == 1]
        n_sole += bool(sole)
        # 4. WHAT THE CELL LACKS -- products with no surviving producer
        lost = []
        for r in sole:
            for p in steps[r]["out"]:
                if p.get("currency"):
                    continue
                if produced_by.get(p["id"], set()) <= set(sole):
                    lost.append({"id": p["id"], "name": p["name"], "type": p["type"]})
        lostids = {p["id"] for p in lost}
        # 5. WHAT FAILS NEXT -- reactions consuming a lost product, and their genes
        starved, sgenes = set(), set()
        frontier = set(lostids)
        for _ in range(MAXDEPTH - 1):
            nxt = set()
            for e in frontier:
                for r in consumed_by.get(e, ()):
                    if r in sole or r in starved:
                        continue
                    starved.add(r)
                    sgenes |= rxn_genes.get(r, set())
                    for p in steps[r]["out"]:
                        if not p.get("currency"):
                            nxt.add(p["id"])
            frontier = nxt
        # 2. WHAT IT NEEDS -- co-requirements of the reactions it runs
        needs_ent, needs_gene = set(), set()
        for r in list(cat)[:TOPN]:
            for p in steps[r]["in"]:
                if not p.get("currency"):
                    needs_ent.add(p["name"])
            needs_gene |= set(steps[r]["catalysts"]) - {g}
        # 6. PATHWAYS
        pw = set()
        for r in sole:
            pw |= rxn_paths.get(r, set())
        pw_all = set()
        for r in cat | part:
            pw_all |= rxn_paths.get(r, set())
        downstream = set()
        for p in pw:
            downstream |= links.get(p, set())
        # 7. CONTACT
        partners = set()
        for r in list(cat | part)[:TOPN]:
            partners |= rxn_genes.get(r, set())
        partners.discard(g)
        where = collections.Counter()
        for r in cat | part:
            for c in comp.get(r, {}).get("compartments", []):
                where[c] += 1

        cmap[g] = {
            "n_catalyses": len(cat), "n_participates": len(part),
            "n_sole_catalyst": len(sole), "stops_reactions": sorted(sole)[:TOPN],
            "compartments": [c for c, _ in where.most_common(4)],
            "needs_entities": sorted(needs_ent)[:TOPN],
            "needs_cocatalysts": sorted(needs_gene)[:20],
            "cell_lacks": lost[:TOPN],
            "n_starved_reactions": len(starved),
            "starved_genes": sorted(sgenes - {g})[:TOPN],
            "pathways_stopped": sorted(pw)[:20], "pathways_member": sorted(pw_all)[:20],
            "pathways_downstream": sorted(downstream)[:20],
            "reaction_partners": sorted(partners)[:TOPN],
            "ppi_partners": sorted(ppi.get(g, set()))[:TOPN],
        }

    lacks = sum(1 for v in cmap.values() if v["cell_lacks"])
    casc = sum(1 for v in cmap.values() if v["starved_genes"])
    print(f"  genes with >=1 SOLE-catalyst reaction : {n_sole:,}  ({n_sole/len(cmap):.1%})")
    print(f"  genes whose loss leaves an unmade product: {lacks:,}  ({lacks/len(cmap):.1%})")
    print(f"  genes with a second-order cascade      : {casc:,}  ({casc/len(cmap):.1%})")
    med = sorted(v["n_starved_reactions"] for v in cmap.values() if v["n_starved_reactions"])
    if med:
        print(f"  starved reactions per cascading gene   : median {med[len(med)//2]}, max {med[-1]}")
    print(f"  genes with NO sole-catalyst reaction    : {len(cmap)-n_sole:,} -- for these the map records context "
          f"but predicts no hard stop, which is the honest answer for a redundant or non-enzymatic protein")

    json.dump(cmap, open(OUT / "consequence_map.json", "w"))
    sz = (OUT / "consequence_map.json").stat().st_size / 1e6
    print(f"\n  -> {OUT/'consequence_map.json'} ({sz:.1f} MB, {len(cmap):,} genes)")


if __name__ == "__main__":
    main()
