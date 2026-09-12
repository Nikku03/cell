"""layers — the cell graph made MANAGEABLE: the ~889k-edge blob reorganized into a small set of named LAYERS, each tagged with
its semantics, its READOUT (what that layer actually predicts), and its measured validation. The organizing principle is the
session's central, measured finding: the cell is not one network -- it is several typed networks, each of which predicts its OWN
readout, and collapsing them loses that. So layers are grouped by READOUT TIER, not lumped together.

  TIER                                  layers                              predicts (readout)              validated
  ---------------------------------------------------------------------------------------------------------------------------
  PHYSICAL     (who TOUCHES whom)        ppi, complex, ligand_receptor       binding/complex assembly        complementary yet
                                                                                                             coherent (6% overlap,
                                                                                                             46-214x above chance)
  REGULATORY   (who CONTROLS transcript) regulatory, signaling, causal       transcription (Perturb-seq)     propagate L1 9.2x;
                                                                                                             sign 73%
  REACTION     (who TRANSFORMS -> product) reaction_metabolic                metabolic FLUX                  ecFlux mutation->flux
  DEPENDENCY   (who NEEDS whom to live)  codependency, synthetic_lethal,     fitness / essentiality          centrality+paralog
                                         coexpression                                                        AUC 0.86
  SPATIAL      (3D contact)              chromatin_loops                     enhancer->gene 3D link          invivo ABC (nearest
                                                                                                             gene wrong 39%)
  MEMBERSHIP   (process grouping)        reactome_pathway                    shared-process grouping         condition top-1 67%

This is a REGISTRY/MANIFEST + queryable manager on top of the existing wiring (network.py provides the 9 physical/functional
layer adjacencies; this adds the reaction/pathway/spatial layers and the readout tagging). It makes the graph manageable: one
place that says what every layer IS, what it PREDICTS, and how it was VALIDATED -- and lets you pull any single layer or tier
instead of the tangled union. HONEST: the reaction layer's proper readout is flux (validated for metabolism via ecFlux); a
Reactome 15k-reaction co-membership superset did NOT beat abstract PPI for the TRANSCRIPTIONAL readout once size-controlled
(Wilcoxon p=0.32) -- a readout mismatch, recorded here so the layering is used against the right target.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")

# the registry: layer -> its tier, direction, what an edge MEANS, what the layer PREDICTS, and how it was validated.
REGISTRY = {
    # PHYSICAL
    "ppi":               ("physical", False, "two proteins physically associate (STRING)", "binding partner / complex", "physical!=functional (complementary), 46-214x coherent"),
    "complex":           ("physical", False, "co-membership of one protein complex", "the machine a protein is a part of", "2,039 complexes; disassembly on subunit loss"),
    "ligand_receptor":   ("physical", True,  "a secreted ligand meets a receptor", "cell-cell signaling entry", "948 pairs (curated)"),
    # REGULATORY
    "regulatory":        ("regulatory", True, "TF -> target, signed (activates/represses)", "TRANSCRIPTION of the target", "propagate L1 regulon 9.2x chance; sign 73%"),
    "signaling":         ("regulatory", True, "signaling cascade edge, signed", "signaling state of the target", "17,432 directed"),
    "causal":            ("regulatory", True, "SIGNOR/CollecTRI signed causal X->Y", "Y functionally depends on X", "curated directed"),
    # REACTION
    "reaction_metabolic":("reaction", True, "an enzyme transforms substrate -> product", "metabolic FLUX", "ecFlux mutation->flux VALIDATED (metabolism)"),
    # DEPENDENCY
    "codependency":      ("dependency", False, "DepMap co-essentiality (need each other to live)", "shared fitness dependency", "essentiality centrality+paralog AUC 0.86"),
    "synthetic_lethal":  ("dependency", False, "lose BOTH and the cell dies", "fitness (redundancy)", "curated + inferred"),
    "coexpression":      ("dependency", False, "co-regulated expression", "co-regulation module", "tracks physical modules"),
    # SPATIAL
    "chromatin_loops":   ("spatial", False, "3D chromatin contact (Hi-C loop)", "enhancer->gene 3D link", "invivo ABC; nearest-gene wrong 39%"),
    # MEMBERSHIP
    "reactome_pathway":  ("membership", False, "co-membership of a Reactome pathway", "shared-process grouping (NOT propagation)", "condition inference top-1 67%"),
}
TIER_READOUT = {
    "physical":   "binding / complex assembly (structural)",
    "regulatory": "transcriptional response (Perturb-seq)",
    "reaction":   "metabolic flux",
    "dependency": "fitness / essentiality",
    "spatial":    "enhancer->gene 3D contact",
    "membership": "shared-process grouping",
}
TIER_ORDER = ["physical", "regulatory", "reaction", "dependency", "spatial", "membership"]


def _counts():
    """edge counts per layer: 9 from network.py's wiring, plus reaction/spatial/membership from the image."""
    import network
    net = network.CellNetwork()
    cnt = {}
    for L in network.INTERCONNECTED + network.INTERDEPENDENT:
        adj = net.out.get(L, {})
        if L in network.DIRECTED:
            cnt[L] = sum(len(v) for v in adj.values())
        else:
            cnt[L] = sum(len(v) for v in adj.values()) // 2
    D = json.load(open(os.path.join(OUT, "cell_complete.json")))
    # reaction (metabolic, with real substrate->product) from generxn
    cnt["reaction_metabolic"] = len(D.get("generxn") or [])
    # spatial 3D loops
    cnt["chromatin_loops"] = len(D.get("loops3d") or [])
    # reactome pathway co-membership -> count total memberships (not clique edges, which would be huge)
    R = json.load(open(os.path.join(OUT, "reactome_pathways.json")))["pathways"]
    cnt["reactome_pathway"] = sum(len(v) for v in R.values())
    return cnt, net


def manifest():
    """the layered scorecard: every layer, its tier/readout/semantics/validation and edge count."""
    cnt, net = _counts()
    tiers = {}
    for L, (tier, directed, semantics, readout, valid) in REGISTRY.items():
        tiers.setdefault(tier, []).append({
            "layer": L, "directed": directed, "edges": cnt.get(L, 0),
            "edge_means": semantics, "predicts": readout, "validated": valid})
    out = {"n_genes": net.n, "n_tiers": len(TIER_ORDER), "n_layers": len(REGISTRY),
           "total_edges": sum(cnt.values()),
           "tiers": [{"tier": t, "readout": TIER_READOUT[t], "layers": tiers.get(t, [])} for t in TIER_ORDER],
           "principle": "grouped by READOUT: each typed network predicts its OWN readout; do not collapse them. "
                        "physical->binding, regulatory->transcription, reaction->flux, dependency->fitness."}
    return out, net


def of(gene):
    """a gene as it sits across the layers, grouped by readout tier (the manageable per-node view)."""
    _, net = manifest()
    i = net._i(gene)
    if i is None:
        return {"status": "unknown", "gene": gene}
    import network
    view = {t: {} for t in TIER_ORDER}
    for L in network.INTERCONNECTED + network.INTERDEPENDENT:
        tier = REGISTRY[L][0]
        n = len(net.out.get(L, {}).get(i, {}))
        if L in network.DIRECTED:
            n_in = len(net.inn.get(L, {}).get(i, {}))
            if n or n_in:
                view[tier][L] = {"out": n, "in": n_in}
        elif n:
            view[tier][L] = n
    return {"status": "ok", "gene": net._nm(i),
            "by_readout_tier": {t: {"readout": TIER_READOUT[t], "layers": view[t]} for t in TIER_ORDER if view[t]}}


def main():
    print("=" * 104)
    print("LAYERS — the cell graph as managed, readout-tagged layers (each typed network predicts its OWN readout)")
    print("=" * 104)
    M, net = manifest()
    print(f"\n  {net.n:,} genes | {M['n_layers']} layers in {M['n_tiers']} readout tiers | ~{M['total_edges']:,} total edges\n")
    for t in M["tiers"]:
        print(f"  ── {t['tier'].upper():11s} predicts: {t['readout']}")
        for L in t["layers"]:
            d = "→" if L["directed"] else "—"
            print(f"       {L['layer']:20s} {d} {L['edges']:>8,} edges   [{L['validated']}]")
    print("\n  a gene across tiers (GATA1):")
    g = of("GATA1")
    for t, d in g.get("by_readout_tier", {}).items():
        print(f"       {t:11s} {d['layers']}")
    print("\n  [manageable: pull any single layer or readout tier instead of the tangled union. HONEST -- the reaction layer's")
    print("   readout is FLUX (validated for metabolism via ecFlux); a Reactome co-membership superset did NOT beat abstract PPI")
    print("   for TRANSCRIPTION once size-controlled (p=0.32) -- a readout mismatch, so use each layer against its right target.]")
    out = {k: v for k, v in M.items()}
    out["note"] = ("The cell graph reorganized into managed, readout-tagged layers (layers.py, 'layers' syscall). Organizing "
                   "principle = the measured finding that the cell is several typed networks each predicting its OWN readout: "
                   "PHYSICAL (ppi/complex/ligand_receptor) -> binding/complex; REGULATORY (regulatory/signaling/causal) -> "
                   "transcription (propagate L1 9.2x); REACTION (metabolic) -> flux (ecFlux validated); DEPENDENCY (codep/SL/"
                   "coexpr) -> fitness (AUC 0.86); SPATIAL (loops) -> 3D enhancer link; MEMBERSHIP (reactome) -> process grouping. "
                   "Layers are COMPLEMENTARY (low overlap) yet COHERENT (above chance). A registry/manifest + per-tier query on "
                   "top of network.py. HONEST: a Reactome 15k-reaction co-membership superset did NOT beat abstract PPI for the "
                   "transcriptional readout once size-controlled (Wilcoxon p=0.32) -- a readout mismatch, recorded so each layer "
                   "is applied to its proper target.")
    json.dump(out, open(os.path.join(OUT, "layers.json"), "w"), indent=1)
    print("\n  -> outputs/orphan/layers.json")
    return out


if __name__ == "__main__":
    main()
