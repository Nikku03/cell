"""PREDICTIONS FROM THE CONSEQUENCE MAP ALONE -- no hand reasoning, no signature panels.

The previous round predicted from hand-written signature panels (ISR, RiBi, spliceosome, ...) and beat the
frequency baseline on 4 of 48 knockouts, all mitochondrial or ER. This round asks a different question: does the
mechanical consequence map -- what stops, what the cell lacks, what starves next -- predict better than that, on
exactly the same 50 knockouts and the same sealed answers?

THE PREDICTION IS THE CASCADE, IN MECHANISTIC ORDER:

    1  starved_genes        genes whose reactions lose an input because the knockout stopped the only route to it.
                            This is the strongest claim the map makes: the cell cannot run these steps.
    2  reaction_partners    proteins in the same step -- direct physical contact, the machine that fails together
    3  pathways_stopped     members of pathways containing a stopped reaction
    4  pathways_downstream  members of pathways fed by those, via measured boundary flow

WHAT THIS TESTS THAT THE PANELS DID NOT. The panels encoded a stress RESPONSE -- what the cell does about a lesion.
The map encodes the LESION itself -- what the cell can no longer do. Perturb-seq measures mRNA, and there is no
guarantee that a starved reaction changes the transcript level of the gene running it. If the map scores worse than
the panels, that asymmetry is the reason, and it is worth stating in advance: a metabolic cascade is a statement
about flux, not about transcription.

THE HARD LIMIT, KNOWN BEFORE SCORING. Only 10.2% of genes have any sole-catalyst reaction, and only 4.2% have a
second-order cascade. For everything else the map has no hard stop to offer and falls back to contact and pathway
membership, which is much weaker. So a large fraction of the 50 will be predicted from context alone, and that
should be reported per knockout rather than averaged away.
"""
import json
import os
from pathlib import Path

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))


def main():
    cmap = json.loads((OUT / "consequence_map.json").read_text())
    doss = json.loads((OUT / "ko_pred_dossiers.json").read_text())
    paths = {}
    SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
    for line in (SP / "ReactomePathways.gmt").read_text().splitlines():
        p = line.split("\t")
        if len(p) > 3:
            paths[p[0]] = list(p[2:])

    preds, tiers = {}, {}
    for k in doss:
        m = cmap.get(k)
        if not m:
            preds[k] = {"reasoning": "gene absent from the consequence map", "predicted_genes": []}
            continue
        genes, seen = [], {k}

        def add(xs):
            for x in xs:
                if x not in seen and "; " not in x:
                    seen.add(x)
                    genes.append(x)

        add(m["starved_genes"])
        n1 = len(genes)
        add(m["reaction_partners"])
        n2 = len(genes)
        for p in m["pathways_stopped"][:3]:
            add(paths.get(p, [])[:15])
        n3 = len(genes)
        for p in m["pathways_downstream"][:3]:
            add(paths.get(p, [])[:10])
        if len(genes) < 20:
            add(m["ppi_partners"])
        tier = ("hard stop + cascade" if m["starved_genes"] else
                "stops a reaction, no cascade" if m["n_sole_catalyst"] else
                "no hard stop -- context only")
        tiers[k] = tier
        preds[k] = {
            "reasoning": f"{tier}: stops {m['n_sole_catalyst']} sole-catalyst reactions, "
                         f"cell lacks {len(m['cell_lacks'])} products, {m['n_starved_reactions']} reactions starve",
            "tier": tier,
            "n_from_cascade": n1, "n_from_partners": n2 - n1, "n_from_pathways": n3 - n2,
            "predicted_genes": genes[:60],
        }
    json.dump(preds, open(OUT / "ko_predictions_map.json", "w"), indent=1)
    import collections
    print(f"{len(preds)} predictions from the consequence map")
    for t, n in collections.Counter(tiers.values()).most_common():
        print(f"   {t:<32} {n}")
    sz = sorted(len(p["predicted_genes"]) for p in preds.values())
    print(f"  prediction size: median {sz[len(sz)//2]}, {sum(1 for s in sz if s==0)} empty")
    print(f"  -> {OUT/'ko_predictions_map.json'}")


if __name__ == "__main__":
    main()
