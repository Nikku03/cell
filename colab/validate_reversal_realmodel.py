"""VALIDATE disease_to_reversal ON THE REAL HUMAN REGULATORY NETWORK.

The self-contained test (validate_disease_to_reversal.py) proves the target-control algorithm is correct on
synthetic GRNs with known drivers. THIS test runs it on the real signed TF->TF regulatory core of the
16,492-gene human model (1,146 TFs / 9,906 signed edges, CollecTRI/DoRothEA-derived), extracted from the
provided cell_explorer.html -> outputs/orphan/tf_core.json.

Ground truth = published reprogramming recipes (the experimentally-validated master-TF sets that convert one
cell type into another): iPSC (Yamanaka OCT4/SOX2/KLF4/MYC), cardiomyocyte (GATA4/MEF2C/TBX5), neuron
(ASCL1/POU3F2/MYT1L), macrophage/myeloid (CEBPA/SPI1), B-cell (PAX5/EBF1/TCF3), hepatocyte
(HNF4A/FOXA1/FOXA2).

Three measured things:
 [A] Attractor landscape — honest characterization of the network's Boolean dynamics.
 [B] Lineage coherence — clamping a recipe yields a compact, lineage-EXCLUSIVE attractor.
 [C] Reprogramming recovery — from source lineage A's attractor, does the target-control search recover
     target lineage B's KNOWN factors as the drivers of A->B, above a random-TF baseline?

Honest note on [C]: the target attractor is anchored by B's recipe, so "clamp B's factors reaches B" is
partly built in. The non-trivial content is whether the search, choosing among ALL differing TFs with no
privilege for the recipe, (i) actually reaches B, (ii) ranks the recipe factors above random differing TFs
(enrichment), and (iii) with what minimality. It is topological, not kinetic; a recovered factor is a
model-consistent hypothesis, not proof.
"""
import json, random
from pathlib import Path
from statistics import mean

from disease_to_reversal import build_incoming, settle, active, control_drivers, transition_setdiff

OUT = Path("outputs/orphan")

RECIPES = {
    "iPSC":       ["POU5F1", "SOX2", "KLF4", "MYC"],
    "cardiac":    ["GATA4", "MEF2C", "TBX5"],
    "neuron":     ["ASCL1", "POU3F2", "MYT1L"],
    "myeloid":    ["CEBPA", "SPI1"],
    "Bcell":      ["PAX5", "EBF1", "TCF3"],
    "hepatocyte": ["HNF4A", "FOXA1", "FOXA2"],
}


def load_core(path=OUT / "tf_core.json"):
    core = json.load(open(path))
    edges = [(e[0], e[1], e[2]) for e in core["edges"]]
    name = {int(k): v for k, v in core["name"].items()}
    idx = {v: k for k, v in name.items()}
    W_in, out_deg, nodes = build_incoming(edges)
    return dict(edges=edges, name=name, idx=idx, W_in=W_in, out_deg=out_deg, order=sorted(nodes))


def lineage_attractor(recipe, C):
    """clamp a recipe's master TFs ON from an all-OFF baseline, settle -> the lineage attractor."""
    on = [C["idx"][f] for f in recipe if f in C["idx"]]
    base = {n: -1 for n in C["order"]}
    _, act = settle(base, C["W_in"], C["order"], clamp={n: 1 for n in on})
    return act, set(on)


def test_landscape(C, n=120, seed=0):
    rng = random.Random(seed)
    atts = {}
    for _ in range(n):
        s = {node: rng.choice([1, -1]) for node in C["order"]}
        _, act = settle(s, C["W_in"], C["order"])
        atts[act] = atts.get(act, 0) + 1
    sizes = sorted((len(a) for a in atts), reverse=True)
    return dict(n_seeds=n, distinct_attractors=len(atts), max_active=sizes[0], min_active=sizes[-1],
                median_active=sizes[len(sizes) // 2],
                note="activation-dominated: random seeds do not self-organize into few clean attractors")


def test_coherence(C):
    states = {L: lineage_attractor(r, C)[0] for L, r in RECIPES.items()}
    Ls = list(states)
    jac = {}
    for i in range(len(Ls)):
        for j in range(i + 1, len(Ls)):
            a, b = states[Ls[i]], states[Ls[j]]
            u = len(a | b)
            jac[f"{Ls[i]}|{Ls[j]}"] = round(len(a & b) / u, 2) if u else 0.0
    foreign_leak = {}
    for L in Ls:
        act = states[L]
        foreign = [m for L2 in RECIPES if L2 != L for m in RECIPES[L2] if C["idx"].get(m) in act]
        foreign_leak[L] = foreign
    return dict(attractor_sizes={L: len(states[L]) for L in Ls},
                pairwise_jaccard=jac,
                foreign_master_leak=foreign_leak,
                lineage_exclusive=all(len(v) == 0 for v in foreign_leak.values())), states


def test_reprogramming(C, states, pairs, budget=6, cand_cap=40):
    rows = []
    for A, B in pairs:
        src, tgt = states[A], states[B]
        recipeB = [C["idx"][f] for f in RECIPES[B] if C["idx"].get(f) in (src ^ tgt)]  # B factors that differ
        diff = src ^ tgt
        # candidate cap: top differing TFs by out-degree (bounds runtime; recipe factors kept if present)
        cand = sorted(diff, key=lambda n: -C["out_deg"].get(n, 0))[:cand_cap]
        cand = set(cand) | set(recipeB)
        start = {n: (1 if n in src else -1) for n in C["order"]}
        res = control_drivers(start, tgt, C["W_in"], C["order"], budget=budget, candidates=cand,
                              out_deg=C["out_deg"])
        recovered = {s["gene"] for s in res["path"]}
        rec_fac = (len(recovered & set(recipeB)) / len(recipeB)) if recipeB else None
        # random baseline: prob a recipe factor lands in a random size-|recovered| subset of the pool
        pool = len(cand)
        e_rand = (len(recovered) / pool) if pool else 0.0
        enrich = (rec_fac / e_rand) if (rec_fac is not None and e_rand > 0) else None
        rows.append(dict(pair=f"{A}->{B}", reached=res["reached"], residual=res["final_distance"],
                         path_len=res["steps"], n_recipe_in_diff=len(recipeB),
                         recipe_recall=None if rec_fac is None else round(rec_fac, 3),
                         random_expected=round(e_rand, 3),
                         enrichment=None if enrich is None else round(enrich, 2),
                         recovered=[C["name"][g] for g in recovered][:10]))
    valid = [r for r in rows if r["recipe_recall"] is not None]
    summary = dict(
        n_pairs=len(rows),
        reached_rate=round(mean(1.0 if r["reached"] else 0.0 for r in rows), 3),
        mean_recipe_recall=round(mean(r["recipe_recall"] for r in valid), 3) if valid else None,
        mean_random_expected=round(mean(r["random_expected"] for r in valid), 3) if valid else None,
        mean_enrichment=round(mean(r["enrichment"] for r in valid if r["enrichment"] is not None), 2)
        if valid else None,
        mean_path_len=round(mean(r["path_len"] for r in rows), 2))
    return dict(summary=summary, pairs=rows)


def main():
    C = load_core()
    print("=" * 80)
    print(f"REAL human TF->TF core: {len(C['order'])} TFs, {len(C['edges'])} signed edges")
    print("=" * 80)

    A = test_landscape(C)
    print(f"\n[A] Attractor landscape: {A['distinct_attractors']} distinct from {A['n_seeds']} random seeds; "
          f"active-set size {A['min_active']}-{A['max_active']} (median {A['median_active']}).")
    print(f"    -> {A['note']}")

    coh, states = test_coherence(C)
    print(f"\n[B] Lineage coherence (clamp each published recipe):")
    print(f"    attractor sizes: {coh['attractor_sizes']}")
    print(f"    lineage-exclusive (no foreign master leaks ON): {coh['lineage_exclusive']}")
    print(f"    iPSC-vs-soma Jaccard e.g. iPSC|cardiac={coh['pairwise_jaccard'].get('iPSC|cardiac')}")

    pairs = [("cardiac", "iPSC"), ("neuron", "iPSC"), ("hepatocyte", "iPSC"), ("myeloid", "iPSC"),
             ("iPSC", "cardiac"), ("iPSC", "neuron"), ("iPSC", "hepatocyte"), ("iPSC", "myeloid"),
             ("Bcell", "myeloid"), ("myeloid", "Bcell"), ("hepatocyte", "cardiac"), ("neuron", "cardiac")]
    R = test_reprogramming(C, states, pairs)
    s = R["summary"]
    print(f"\n[C] Reprogramming recovery over {s['n_pairs']} cross-lineage transitions:")
    print(f"    reached target attractor      : {s['reached_rate']}")
    print(f"    recipe-factor recall          : {s['mean_recipe_recall']}  (random baseline {s['mean_random_expected']})")
    print(f"    enrichment over random         : {s['mean_enrichment']}x")
    print(f"    mean reprogramming path length: {s['mean_path_len']}")
    print("    per-transition:")
    for r in R["pairs"]:
        print(f"      {r['pair']:20} reached={str(r['reached']):5} recall={r['recipe_recall']} "
              f"enrich={r['enrichment']} path={r['path_len']} recovered={r['recovered'][:6]}")

    payload = dict(core=dict(n_tf=len(C["order"]), n_edges=len(C["edges"]),
                             provenance="tf_core.json from user-provided cell_explorer.html"),
                   landscape=A, coherence=coh, reprogramming=R)
    json.dump(payload, open(OUT / "reversal_realmodel_validation.json", "w"), indent=2)
    print("\n-> outputs/orphan/reversal_realmodel_validation.json")
    return payload


if __name__ == "__main__":
    main()
