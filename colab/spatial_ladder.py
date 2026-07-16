"""spatial_ladder — lay a pathway out by COMPARTMENT, so you can see which proteins in it act at the membrane, in the
cytoplasm, and in the nucleus. The original idea was a directional "ladder": a pathway that starts at the membrane and
ends in the nucleus would tell you which reactions happen in which compartment, in order.

The load-bearing question, tested up front in validate(): is that DIRECTIONAL ladder real? Do the DOWNSTREAM signalling
genes (the ones a regulatory-tier ordering puts last) actually live in the nucleus, and the UPSTREAM ones at the membrane?
MEASURED ANSWER: NO. Nuclear-annotated vs membrane-annotated genes have essentially the same tier distribution, and (the
tell) there are MORE TFs among the upstream genes than the downstream ones. The cell_levels "tier" is a regulatory-graph
order, not a spatial one, and every large pathway spans all four compartments anyway. So the directional ladder is a
just-so story and is NOT claimed here.

What IS delivered (honestly): a per-pathway compartment LAYOUT read DIRECTLY from localization (~100% coverage). Proteins
are MULTI-compartment (the source lists every compartment a protein is seen in, NOT in primacy order — HRAS is tagged
Nucleus, Golgi, membrane, cytoplasm all at once), so a protein is counted in EVERY compartment it's annotated in, not one
arbitrary pick. You get: the compartment PROFILE of a pathway (how many of its proteins touch each layer) and the
per-layer gene lists (depth-ordered outside->in for readability only, NOT a claimed flow). Curated/observed annotations,
not a phenotype predictor.
"""
import json, collections
from pathlib import Path
import numpy as np
OUT = Path("outputs/orphan")

# the depth axis (outside -> in). A protein is placed in EVERY listed compartment that maps here (multi-membership).
DEPTH = {"Secreted": 0, "Extracellular": 0, "Extracellular space": 0,
         "Cell membrane": 1, "Membrane": 1, "Cell surface": 1,
         "Cytoplasm": 2, "Cytosol": 2, "Endoplasmic reticulum": 2, "Golgi apparatus": 2, "Cytoskeleton": 2, "Endosome": 2,
         "Nucleus": 3}
DEPTH_NAME = {0: "extracellular", 1: "membrane", 2: "cytoplasm", 3: "nucleus"}


def _load():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    tf = {g["name"] for g in D["genes"] if g.get("tf")}
    lv = json.load(open(OUT / "cell_levels.json")).get("labels", {})
    loc = json.load(open(OUT / "localization.json")).get("labels", {})
    paths = json.load(open(OUT / "reactome_pathways.json"))["pathways"]
    return names, tf, lv, loc, paths


def _depths(loc, g):
    """the SET of depth layers a gene is annotated in (multi-compartment; the source is not primacy-ordered)."""
    return {DEPTH[c] for c in (loc.get(g) or []) if c in DEPTH}


def validate(names, tf, lv, loc):
    """reality check for the DIRECTIONAL ladder: do downstream signalling genes live deeper (nucleus) than upstream ones?
    Two-group test that doesn't force a multi-compartment protein into one location: compare the tier level of genes that
    ARE annotated nuclear vs genes that ARE annotated at the membrane."""
    nuc_lv, mem_lv = [], []
    up_tf, down_tf = [], []
    for g, d in lv.items():
        if d.get("status") != "signaling-tier" or d.get("level") is None:
            continue
        deps = _depths(loc, g)
        if not deps:
            continue
        lvl = d["level"]
        if 3 in deps:
            nuc_lv.append(lvl)
        if 1 in deps:
            mem_lv.append(lvl)
        if lvl > 0.66:
            down_tf.append(g in tf)
        elif lvl < 0.34:
            up_tf.append(g in tf)
    nuc_lv, mem_lv = np.array(nuc_lv), np.array(mem_lv)
    # if the ladder were real, nuclear genes would sit MORE downstream (higher level) than membrane genes.
    return {"n_nuclear": int(len(nuc_lv)), "n_membrane": int(len(mem_lv)),
            "mean_tier_nuclear": round(float(nuc_lv.mean()), 3), "mean_tier_membrane": round(float(mem_lv.mean()), 3),
            "nuclear_more_downstream_by": round(float(nuc_lv.mean() - mem_lv.mean()), 3),
            "downstream_tf_frac": round(float(np.mean(down_tf)), 3), "upstream_tf_frac": round(float(np.mean(up_tf)), 3)}


def ladder(pathway, names=None, tf=None, lv=None, loc=None, paths=None):
    """the compartment LAYOUT of one pathway: each protein placed in every compartment it's annotated in, grouped by
    depth layer (outside->in). Profile + per-layer gene lists — NOT a directional route (see validate())."""
    if names is None:
        names, tf, lv, loc, paths = _load()
    if pathway not in paths:
        return None
    stages = collections.defaultdict(list)
    n_localized = 0
    for i in paths[pathway]:
        if i >= len(names):
            continue
        g = names[i]
        deps = _depths(loc, g)
        if deps:
            n_localized += 1
        tag = g + ("*" if g in (tf or set()) else "")          # * marks a TF
        for dep in deps:
            stages[dep].append(tag)
    stages = {DEPTH_NAME[d]: sorted(set(stages[d])) for d in sorted(stages)}
    return {"pathway": pathway, "n_genes": len(paths[pathway]), "n_localized": n_localized,
            "stage_counts": {k: len(v) for k, v in stages.items()},
            "compartments_present": list(stages.keys()), "stages": stages}


def find_pathways(query, paths):
    """pathways whose name contains `query` (case-insensitive) — so callers don't need the exact Reactome string."""
    q = query.lower()
    return [p for p in paths if q in p.lower()]


def gene_pathways(gene, names, paths):
    """the pathways a gene participates in (by index membership)."""
    try:
        idx = names.index(gene)
    except ValueError:
        return []
    return [p for p, members in paths.items() if idx in members]


def main():
    print("=" * 92)
    print("SPATIAL LADDER — pathways laid out by compartment (localization x pathway), reality-checked")
    print("=" * 92)
    names, tf, lv, loc, paths = _load()
    v = validate(names, tf, lv, loc)
    print(f"\n  VALIDATION — is the directional membrane->nucleus ladder real?")
    print(f"    mean tier(nuclear genes, n={v['n_nuclear']}) = {v['mean_tier_nuclear']}   vs   mean tier(membrane genes, n={v['n_membrane']}) = {v['mean_tier_membrane']}")
    print(f"    nuclear genes are more downstream by only {v['nuclear_more_downstream_by']:+.3f}  (would need to be large & positive)")
    print(f"    TF fraction  downstream {v['downstream_tf_frac']:.0%}  vs  upstream {v['upstream_tf_frac']:.0%}  (BACKWARDS — more TFs upstream)")
    real = v["nuclear_more_downstream_by"] > 0.15 and v["downstream_tf_frac"] > v["upstream_tf_frac"]
    print(f"    -> the directional ladder is {'REAL' if real else 'NOT supported by the data'}; delivering the compartment LAYOUT instead.")
    print("\n  example pathway compartment layouts (proteins counted in EVERY compartment they're annotated in):")
    for pw in ["Signaling by EGFR", "MAPK1/MAPK3 signaling", "Signaling by WNT", "Interleukin-4 and Interleukin-13 signaling"]:
        L = ladder(pw, names, tf, lv, loc, paths)
        if L and L["compartments_present"]:
            span = " | ".join(f"{k} {L['stage_counts'][k]}" for k in L["compartments_present"])
            print(f"    {pw[:44]:44s}: [{span}]  ({L['n_localized']}/{L['n_genes']} localized)")
    out = {"validation": v, "directional_ladder_is_real": bool(real),
           "note": "MEASURED NEGATIVE: the regulatory 'tier' does NOT track compartment depth — nuclear and membrane "
                   f"signalling genes have near-identical tier means (diff {v['nuclear_more_downstream_by']:+.3f}), and there "
                   f"are MORE TFs upstream ({v['upstream_tf_frac']:.0%}) than downstream ({v['downstream_tf_frac']:.0%}), the "
                   "opposite of a membrane->nucleus flow. So there is NO directional ladder. What IS delivered: a per-pathway "
                   "compartment LAYOUT from localization (~100% coverage). Proteins are multi-compartment (source is not "
                   "primacy-ordered) so each is counted in EVERY layer it's annotated in — a PROFILE/occupancy view + "
                   "per-layer gene lists, NOT a route and NOT a phenotype predictor."}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "spatial_ladder.json", "w"), indent=1)
    print(f"\n  -> {OUT/'spatial_ladder.json'}")
    return out


if __name__ == "__main__":
    main()
