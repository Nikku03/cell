"""ko_cell_map -- place a knockout's measured movers into an empty cell, with each protein's trafficking route and molecular state.

Takes the dossier produced by ko_dossier.py and prepares what a spatial view needs:

  WHERE IT ENDS UP        the assigned subcellular compartment, which is the annotation and is MEASURED-ADJACENT (curated localisation).
  WHERE IT STARTED        every protein-coding gene starts at a chromosomal locus in the nucleus. Reported as chrom:TSS.
  HOW IT GOT THERE        the canonical route to that destination, with the MOLECULAR STATE at each hop -- hnRNA, mRNA, or protein.
  WHAT MOVES THINGS       transporter proteins are flagged separately, since they are the machinery of the transport step rather than
                          cargo of it.

HONESTY ABOUT THE ROUTE. The destination compartment is curated annotation. The ROUTE is NOT measured -- it is the canonical trafficking
path implied by that destination under standard cell biology (SRP/co-translational entry to ER for the secretory branch, TOM/TIM for
mitochondria, NLS re-import for nuclear proteins, and so on). It is inference from destination, not observation, and is labelled that way
in the output. What IS measured is: the gene moved, by how much, and where its product is annotated to live.

Every route shares the same first two hops because every protein-coding gene does:
    chromatin locus --transcription--> hnRNA --splice/cap/polyA--> mRNA --nuclear pore export--> mRNA (cytosol)
and then diverges by destination. The state label on each hop is the point: a gene's product is hnRNA, then mRNA, then protein, and the
compartment it is "in" means something different at each stage."""
import os
import json, collections, sys
from pathlib import Path

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))

# canonical route to each destination. (location, state, event)
BASE = [("nucleus", "hnRNA", "transcribed at locus"),
        ("nucleus", "mRNA", "capped, spliced, polyadenylated"),
        ("nuclear pore", "mRNA", "exported to cytosol")]
ROUTES = {
    "cytoplasm":       BASE + [("cytosol", "protein", "translated on free ribosome")],
    "cytoskeleton":    BASE + [("cytosol", "protein", "translated on free ribosome"),
                               ("cytoskeleton", "protein", "assembles onto filament")],
    "nucleus":         BASE + [("cytosol", "protein", "translated on free ribosome"),
                               ("nuclear pore", "protein", "NLS-mediated re-import")],
    "mitochondrion":   BASE + [("cytosol", "protein", "translated on free ribosome"),
                               ("mitochondrion", "protein (unfolded)", "TOM/TIM import, presequence cleaved")],
    "peroxisome":      BASE + [("cytosol", "protein", "translated on free ribosome"),
                               ("peroxisome", "protein (folded)", "PTS1/PTS2 import")],
    "ER":              BASE + [("ER membrane", "protein (nascent)", "SRP-targeted, co-translational translocation")],
    "Golgi":           BASE + [("ER membrane", "protein (nascent)", "SRP-targeted, co-translational translocation"),
                               ("COPII vesicle", "protein", "ER exit site budding"),
                               ("Golgi", "protein", "cisternal maturation, glycosylation")],
    "lysosome":        BASE + [("ER membrane", "protein (nascent)", "SRP-targeted, co-translational translocation"),
                               ("Golgi", "protein", "M6P tag added"),
                               ("lysosome", "protein", "M6P-receptor vesicle delivery")],
    "endosome":        BASE + [("ER membrane", "protein (nascent)", "SRP-targeted, co-translational translocation"),
                               ("Golgi", "protein", "sorted at trans-Golgi network"),
                               ("endosome", "protein", "vesicular delivery")],
    "plasma membrane": BASE + [("ER membrane", "protein (nascent)", "SRP-targeted, co-translational translocation"),
                               ("Golgi", "protein", "glycosylated, sorted"),
                               ("secretory vesicle", "protein", "TGN budding"),
                               ("plasma membrane", "protein", "vesicle fusion, inserted in bilayer")],
    "membrane":        BASE + [("ER membrane", "protein (nascent)", "SRP-targeted, co-translational translocation"),
                               ("Golgi", "protein", "sorted"),
                               ("membrane", "protein", "vesicle fusion")],
    "extracellular":   BASE + [("ER lumen", "protein (nascent)", "SRP-targeted, signal peptide cleaved"),
                               ("Golgi", "protein", "glycosylated"),
                               ("secretory vesicle", "protein", "TGN budding"),
                               ("extracellular", "protein", "exocytosis, released")],
}
TRANSPORT_GO = ("transporter activity", "transmembrane transport", "channel activity", "symporter",
                "antiporter", "ATPase-coupled", "carrier activity", "permease", "ion transport",
                "protein transport", "nucleocytoplasmic", "vesicle-mediated")


def main(ko="GATA1", topn=140):
    dz = json.load(open(OUT / f"ko_dossier_{ko}.json"))
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; N = len(names)
    ppi = collections.defaultdict(set)
    for e in (D.get("ppi") or []):
        try:
            a, b = names[e[0]], names[e[1]]
        except (TypeError, IndexError):
            continue
        ppi[a].add(b); ppi[b].add(a)

    def is_transporter(m):
        j = m["journey"]
        if (j["end_state"]["process"] or "") in ("transport/uptake", "trafficking/secretion"):
            return True
        blob = " ".join(j["end_state"]["GO_function"] + j["end_state"]["GO_process"]).lower()
        return any(t.lower() in blob for t in TRANSPORT_GO)

    moved = dz["moved"]
    for m in moved:
        m["transporter"] = is_transporter(m)
    # keep the strongest movers, but never drop a transporter or a cascade TF
    cascade_tfs = {c["tf"] for c in dz["interconnection"]["tf_cascade_within_movers"]}
    ranked = sorted(moved, key=lambda m: -abs(m["z"]))
    keep, seen = [], set()
    for m in ranked:
        if len(keep) < topn or m["transporter"] or m["gene"] in cascade_tfs:
            keep.append(m); seen.add(m["gene"])
    nodes = []
    for m in keep:
        j = m["journey"]
        comp = j["transport"]["compartment"] or "cytoplasm"
        route = ROUTES.get(comp, ROUTES["cytoplasm"])
        nodes.append({
            "gene": m["gene"], "z": m["z"], "dir": m["direction"],
            "direct": m["direct_curated_target"], "transporter": m["transporter"],
            "compartment": comp,
            "origin": f"{j['gene']['chrom']}:{j['gene']['tss']}" if j["gene"]["chrom"] else "unmapped",
            "abundance_ppm": j["translation"]["abundance_ppm"],
            "process": j["end_state"]["process"], "pathway": j["end_state"]["reactome_primary"],
            "go_function": j["end_state"]["GO_function"][:2],
            "n_ppi": j["interaction"]["n_ppi"], "complexes": j["interaction"]["complexes"][:2],
            "ptm": (j["modification"] or {}).get("c") if j["modification"] else None,
            "route": [{"loc": a, "state": b, "event": c} for a, b, c in route],
            "n_regulators": j["transcription"]["n_regulators"],
        })
    kept = {n["gene"] for n in nodes}
    edges = sorted({tuple(sorted((a, b))) for a in kept for b in ppi.get(a, ()) if b in kept and a != b})
    casc = [{"tf": c["tf"], "n": c["n_downstream_also_moved"],
             "targets": [t for t in c["examples"] if t in kept][:6]}
            for c in dz["interconnection"]["tf_cascade_within_movers"][:8]]

    comp_all = collections.Counter(
        (m["journey"]["transport"]["compartment"] or "unannotated") for m in moved)
    trans_by_comp = collections.Counter(
        (m["journey"]["transport"]["compartment"] or "unannotated") for m in moved if m["transporter"])

    out = {
        "knockout": ko, "cell_line": dz["cell_line"],
        "self": {"gene": ko, "compartment": dz["self"]["identity"]["comp"],
                 "origin": f"{dz['self']['identity']['chrom']}:{dz['self']['identity']['tss']}",
                 "route": [{"loc": a, "state": b, "event": c} for a, b, c in ROUTES["nucleus"]]},
        "totals": {"moved": dz["effect"]["n_moved_specific"], "up": dz["effect"]["n_up"],
                   "down": dz["effect"]["n_down"], "shown": len(nodes),
                   "transporters_total": sum(1 for m in moved if m["transporter"]),
                   "direct_targets": dz["effect"]["n_direct_curated_targets_among_moved"],
                   "curated_targets": dz["effect"]["curated_targets_total"]},
        "by_compartment": comp_all.most_common(),
        "transporters_by_compartment": trans_by_comp.most_common(),
        "pathways": dz["interconnection"]["pathways_shared_by_movers"][:12],
        "processes": dz["interconnection"]["processes"],
        "complexes": dz["interconnection"]["complexes_with_2plus_movers"][:8],
        "ppi_edges_total": dz["interconnection"]["ppi_edges_among_movers"],
        "coverage": dz["annotation_coverage_over_movers"],
        "nodes": nodes, "edges": [list(e) for e in edges], "cascade": casc,
        "route_legend": {"note": "Destination compartment is curated annotation. The ROUTE is inferred from that "
                                 "destination by canonical trafficking rules, not measured.",
                         "states": ["hnRNA", "mRNA", "protein", "protein (nascent)", "protein (unfolded)",
                                    "protein (folded)"]},
    }
    json.dump(out, open(OUT / f"ko_cell_map_{ko}.json", "w"), indent=1)
    print(f"{ko}: {out['totals']['moved']} movers, {len(nodes)} placed, {len(edges)} PPI edges among them")
    print(f"  transporters among movers: {out['totals']['transporters_total']}")
    print(f"  by compartment: {out['by_compartment']}")
    print(f"  transporters by compartment: {out['transporters_by_compartment']}")
    print(f"  -> outputs/orphan/ko_cell_map_{ko}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "GATA1")
