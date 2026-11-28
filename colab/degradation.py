"""degradation — U1: the missing EDGE TYPE. The signed network carries signalling and transcription but NOT
protein degradation, so an E3-ubiquitin-ligase loss can't propagate to its stabilised substrate. That is why
VHL loss never lit up HIF2A (the belzutifan axis): VHL→HIF is degradation, not signalling.

An E3 → substrate degradation edge is, for perturbation purposes, a REPRESSION of the substrate's abundance:
sign −1. So clamping the E3 to LOF (−1) propagates (−1)·(−1) = +1 to the substrate → the substrate goes UP.
Same machinery as a repressive regulatory edge; it was simply absent.

Curated seed of well-established E3 (or degradation-complex) → substrate pairs from the ubiquitin-proteasome
literature (UbiBrowser / Reactome degradation). High-confidence, disease-relevant; extensible to the full
UbiBrowser set in Colab. Written as signed edges folded into the signed propagation network.
-> outputs/orphan/degradation.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

OUT = "outputs/orphan"

# E3 / degradation-complex  ->  [substrates it targets for destruction]  (edge sign in propagation = -1)
E3_SUBSTRATE = {
    "VHL":   ["HIF1A", "EPAS1", "HIF3A"],                       # renal: VHL loss -> HIF stabilised (belzutifan axis)
    "MDM2":  ["TP53"],                                          # MDM2 loss -> p53 up
    "MDM4":  ["TP53"],
    "FBXW7": ["MYC", "NOTCH1", "CCNE1", "JUN", "MCL1", "KLF5"], # FBXW7 loss -> oncoprotein accumulation
    "SKP2":  ["CDKN1B", "CDKN1A", "RASSF1"],                    # SKP2 loss -> p27/p21 up (growth brake restored)
    "KEAP1": ["NFE2L2"],                                        # KEAP1 loss -> NRF2 up (lung / oxidative)
    "CBL":   ["EGFR", "MET", "KIT", "FLT3", "PDGFRA"],          # CBL loss -> RTK stabilised (leukaemia)
    "BTRC":  ["CTNNB1", "NFKBIA", "CDC25A", "SNAI1"],           # bTrCP loss -> beta-catenin / NFkB signalling up
    "FBXW11":["CTNNB1", "NFKBIA"],
    "SPOP":  ["AR", "SRC3", "ERG", "BRD4"],                     # SPOP loss -> AR / BRD4 up (prostate)
    "NEDD4": ["PTEN", "RASGRP1"],
    "ITCH":  ["JUN", "JUNB", "NOTCH1"],
    "SIAH1": ["SNAI1", "SNAI2"],
    "DTL":   ["CDT1", "CDKN1A", "SET8"],                        # CRL4-CDT2
    "CDC20": ["CCNB1", "SECURIN", "CCNA2"],
    "FZR1":  ["CCNB1", "CCNA2", "CDC20", "SKP2"],               # APC/C-Cdh1
    "RNF43": ["FZD5", "FZD1", "LRP6"],                          # RNF43 loss -> WNT receptors up (MSI CRC)
    "TRIM33":["SMAD4"],
    "HUWE1": ["MCL1", "MYC", "CDC6"],
    "PARK2": ["CCNE1", "CDCA5"],                                # PRKN
    "STUB1": ["ERBB2", "HIF1A", "AKT1"],                        # CHIP
    "MARCHF7": ["RB1"],
    "UBE3A": ["TP53"],
}


def build(C):
    """resolve the curated pairs to model indices; write degradation.json. Returns edge list [(e3_idx, sub_idx)]."""
    idx = C.idx
    edges = []; meta = {}
    for e3, subs in E3_SUBSTRATE.items():
        if e3 not in idx:
            continue
        present = [s for s in subs if s in idx]
        for s in present:
            edges.append([idx[e3], idx[s]])
        if present:
            meta[e3] = present
    out = {"edges": edges, "by_e3": meta, "n_e3": len(meta), "n_edges": len(edges),
           "note": "E3->substrate degradation; propagation sign = -1 (E3 loss stabilises substrate). Seed set; "
                   "extend from UbiBrowser 2.0 in Colab."}
    json.dump(out, open(f"{OUT}/degradation.json", "w"))
    return out


def signed_edges(C):
    """[(e3_idx, substrate_idx, -1)] for folding into the signed propagation network. Loads/builds as needed."""
    p = f"{OUT}/degradation.json"
    d = json.load(open(p)) if os.path.exists(p) else build(C)
    return [(a, b, -1) for a, b in d["edges"]]


if __name__ == "__main__":
    from complete_cell import CompleteCell
    C = CompleteCell()
    d = build(C)
    print(f"degradation layer: {d['n_e3']} E3s, {d['n_edges']} E3->substrate edges")
    # sanity: VHL -> HIF present?
    print("  VHL targets in model:", d["by_e3"].get("VHL"))
    print("  KEAP1 targets:", d["by_e3"].get("KEAP1"), "| CBL:", d["by_e3"].get("CBL"))
