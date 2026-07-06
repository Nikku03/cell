"""VALIDATE the Phase-1 emask — does the populated cell-type expression recover cell-type LINEAGE identity?

Phase 1 populated `emask` (per-gene bitmask over 179 Tabula-Sapiens cell types). Test whether it encodes
real cell biology: are known lineage MASTER TFs expressed in cell types of their correct lineage, and
enriched there vs a random baseline? Ground truth = textbook developmental biology (never trained on).

Also reports the context-pruning FEASIBILITY flag: what fraction of reprogramming recipe factors are present
in emask — because if key masters are missing (shallow census), hard-pruning the network to emask-expressed
genes would drop functional drivers and must NOT be done.

Reads outputs/orphan/celltype_mask.json (ctnames + emask, from the Phase-1 export) — or cell_complete.json on Colab.
-> celltype_identity_validation.json
"""
import json, random
from pathlib import Path
from statistics import median

OUT = Path("outputs/orphan")

LINEAGE = {
    "cardiac":     ["cardi", "myocyte", "myoblast", "endocardial"],
    "hepatic":     ["hepato", "cholangio", "hepatic stellate", "hepatic pit"],
    "myeloid":     ["monocyte", "macrophage", "dendritic", "granulocyt", "neutrophil", "myeloid",
                    "kupffer", "promyelo", "promono", "phagocyte", "mast cell", "basophil"],
    "Bcell":       ["b cell", "pro-b", "pre-b", "b-1 b", "b-2 b", "plasma cell", "plasmablast",
                    "precursor b", "naive b", "memory b", "mature b", "immature b"],
    "Tcell":       ["t cell", "cd4", "cd8", "thymus", "regulatory t", "nk t", "helper t",
                    "alpha-beta t", "gamma-delta t", "invariant t", "intraepithelial t"],
    "erythroid":   ["erythro", "proerythrob"],
    "endothelial": ["endothelial", "vein endo", "capillary", "vascular"],
}
MASTERS = {
    "cardiac":     ["GATA4", "TBX5", "NKX2-5", "MEF2C", "HAND2", "MYOCD", "TBX20"],
    "hepatic":     ["HNF4A", "FOXA2", "HNF1A", "FOXA1", "ONECUT1"],
    "myeloid":     ["SPI1", "CEBPA", "CEBPB", "IRF8", "KLF4"],
    "Bcell":       ["PAX5", "EBF1", "TCF3", "POU2AF1", "POU2F2", "IRF4"],
    "Tcell":       ["GATA3", "TCF7", "LEF1", "TBX21", "FOXP3", "BCL11B"],
    "erythroid":   ["GATA1", "KLF1", "TAL1", "NFE2"],
    "endothelial": ["ERG", "FLI1", "SOX17", "SOX18"],
}
RECIPES = ["POU5F1", "SOX2", "KLF4", "MYC", "GATA4", "MEF2C", "TBX5", "CEBPA",
           "SPI1", "PAX5", "EBF1", "HNF4A", "FOXA1", "FOXA2"]


def load():
    p = OUT / "celltype_mask.json"
    if p.exists():
        d = json.load(open(p)); return d["ctnames"], d["emask"], set(d["tf_genes"])
    cc = OUT / "cell_complete.json"
    if cc.exists():
        D = json.load(open(cc)); G = D["genes"]
        em = {G[int(k)]["name"]: int(v) for k, v in D["emask"].items()}
        return D["ctnames"], em, set(G[i]["name"] for i, g in enumerate(G) if g.get("tf"))
    return None, None, None


def validate():
    ctnames, em, tfgenes = load()
    if ctnames is None:
        print("no celltype_mask.json / cell_complete.json"); return None
    ct = [c.lower() for c in ctnames]
    lin_types = {L: [i for i, c in enumerate(ct) if any(p in c for p in pats)] for L, pats in LINEAGE.items()}

    def types_of(g):
        m = em.get(g)
        return None if m is None else set(i for i in range(len(ct)) if (m >> i) & 1)

    rows = []; enrich = []
    for L, ms in MASTERS.items():
        Lt = set(lin_types[L]); base = len(Lt) / len(ct)
        for g in ms:
            te = types_of(g)
            if te is None:
                rows.append(dict(tf=g, lineage=L, present=False)); continue
            inL = len(te & Lt); frac = inL / len(te) if te else 0
            e = (frac / base) if base else 0; enrich.append(e)
            rows.append(dict(tf=g, lineage=L, present=True, hit=inL > 0,
                             in_lineage=inL, n_types=len(te), enrichment=round(e, 1)))
    present = [r for r in rows if r["present"]]
    hits = [r for r in present if r["hit"]]
    # random baseline: random TF in a random lineage
    rng = random.Random(0); tf_em = [g for g in em if g in tfgenes]; rhit = 0; N = 2000
    Ls = list(lin_types)
    for _ in range(N):
        g = rng.choice(tf_em); L = rng.choice(Ls); Lt = set(lin_types[L])
        te = types_of(g)
        if te and te & Lt:
            rhit += 1
    recipe_present = [r for r in RECIPES if r in em]
    summary = dict(
        n_masters=len(rows), n_present=len(present), coverage=round(len(present) / len(rows), 3),
        recall=round(len(hits) / len(present), 3) if present else None,
        median_enrichment=round(median(enrich), 1) if enrich else None,
        random_baseline=round(rhit / N, 3),
        recall_over_random=round((len(hits) / len(present)) / (rhit / N), 1) if present and rhit else None,
        recipe_coverage=round(len(recipe_present) / len(RECIPES), 3),
        recipe_missing=[r for r in RECIPES if r not in em],
        context_pruning_safe=(len(recipe_present) / len(RECIPES) >= 0.9),
    )
    return dict(summary=summary, per_master=rows)


def main():
    res = validate()
    if not res:
        return
    json.dump(res, open(OUT / "celltype_identity_validation.json", "w"), indent=2)
    s = res["summary"]
    print("=" * 74)
    print("PHASE 1 — cell-type-lineage identity recovery from the populated emask")
    print("=" * 74)
    print(f"  masters present in emask: {s['n_present']}/{s['n_masters']} (coverage {s['coverage']})")
    print(f"  lineage recall (master expressed in its lineage): {s['recall']} "
          f"({s['recall_over_random']}x over random {s['random_baseline']})")
    print(f"  median enrichment over lineage base-rate: {s['median_enrichment']}x")
    print(f"  context-pruning feasibility: recipe coverage {s['recipe_coverage']} "
          f"-> hard-prune {'SAFE' if s['context_pruning_safe'] else 'UNSAFE (would drop '+', '.join(s['recipe_missing'])+')'}")


if __name__ == "__main__":
    main()
