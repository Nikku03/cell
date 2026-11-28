"""curated_edges — high-confidence SIGNED regulatory edges that are established biology but absent from the
network (usually because a target gene was itself missing until P0 added it). STRING gives undirected PPI; these
are the DIRECTED, SIGNED functional relationships needed for perturbation propagation.

Flagship: the sickle-cell / β-thalassaemia switch — BCL11A (and ZBTB7A/LRF) REPRESS the fetal-globin genes
HBG1/HBG2. Removing BCL11A de-represses fetal haemoglobin (the exa-cel / Casgevy mechanism). HBG1/2 were only
added to the model in P0, so this edge could not exist before. Sign −1 = repression (target rises when the
regulator is lost).

Seed set; extend from SIGNOR / CollecTRI for other newly-added disease genes.
-> folded into cancer_cell_map._signed_out
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

# regulator -> [(target, sign)]   sign: +1 activates, -1 represses
CURATED_REG = {
    "BCL11A": [("HBG1", -1), ("HBG2", -1), ("HBB", +1)],        # fetal-globin silencing (sickle-cell switch)
    "ZBTB7A": [("HBG1", -1), ("HBG2", -1)],                     # LRF — parallel fetal-globin repressor
    "KLF1":   [("BCL11A", +1), ("HBB", +1), ("HBG1", -1)],      # KLF1 drives BCL11A -> represses fetal globin
    "GATA1":  [("HBB", +1), ("HBA1", +1), ("ALAS2", +1)],       # master erythroid activator
    "MYB":    [("HBG1", -1), ("HBG2", -1)],                     # MYB raises BCL11A -> fetal-globin repression
    "NFE2L2": [("HMOX1", +1), ("NQO1", +1), ("GCLC", +1)],      # NRF2 antioxidant programme (KEAP1 loss axis)
    "HIF1A":  [("VEGFA", +1), ("SLC2A1", +1), ("LDHA", +1)],    # hypoxia programme (VHL-loss axis, downstream)
    "EPAS1":  [("EPO", +1), ("VEGFA", +1), ("PDGFB", +1)],      # HIF2A programme (renal)
}


def signed_edges(C):
    """[(reg_idx, target_idx, sign)] for the curated regulatory edges whose BOTH endpoints are in the model."""
    idx = C.idx
    out = []
    for reg, targets in CURATED_REG.items():
        ri = idx.get(reg)
        if ri is None:
            continue
        for tgt, sign in targets:
            ti = idx.get(tgt)
            if ti is not None and ti != ri:
                out.append((ri, ti, sign))
    return out


if __name__ == "__main__":
    from complete_cell import CompleteCell
    C = CompleteCell()
    e = signed_edges(C)
    print(f"curated signed regulatory edges resolvable in model: {len(e)}")
    # sickle-cell test: BCL11A knockout should DE-REPRESS HBG1/HBG2
    import cancer_cell_map as ccm
    out = ccm._signed_out(C)
    pert = ccm.propagate_signed(out, {C.idx["BCL11A"]: -1})
    for g in ("HBG1", "HBG2", "HBB"):
        gi = C.idx.get(g)
        print(f"   BCL11A-KO -> {g}: {round(pert.get(gi, 0.0), 4) if gi is not None else '(missing)'}"
              + ("   (fetal Hb UP = sickle correction)" if g.startswith("HBG") else ""))
