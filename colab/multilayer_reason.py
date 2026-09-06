"""MULTI-LAYER REASONING: lesion -> functional loss -> sensor -> transcription factor -> measurable mRNA.

WHY THE PREVIOUS TWO ATTEMPTS FAILED, WHICH DETERMINES THIS DESIGN.

  hand panels        guessed a stress programme per knockout. Beat the frequency baseline on 4/48, and all four
                     were mitochondrial or ER -- the cases where a lesion happens to route into a transcription
                     factor. Everything else scored zero.
  consequence map    propagated flux: what stops, what the cell lacks, what starves. Beat it on 1/48. A starved
                     reaction is a statement about FLUX; Perturb-seq measures mRNA. The two never met.

The missing piece is the coupling between them. A cell does not transcribe anything because a reaction stopped --
it transcribes because a SENSOR detected a consequence and activated a TRANSCRIPTION FACTOR. So the chain has to
run all the way through:

    LESION -> FUNCTIONAL LOSS -> what the cell now lacks -> SENSOR that detects it -> TF -> mRNA programme

LAYER 1  DIRECT LOSS      the protein's own reactions stop (sole catalyst), and its own TF regulon is released
LAYER 2  MISLOCALISATION  a transport/pore/import lesion does not destroy its cargo, it STRANDS it. The cargo is
                          present but functionally absent from the compartment where it works, so the cargo's
                          reactions there stop even though the cargo gene is untouched. This is the layer a
                          metabolite cascade cannot express.
LAYER 3  NO MATURATION    an ER/Golgi lesion leaves clients unglycosylated and unfolded: present, not functional
LAYER 4  COFACTOR/ION     an ion or cofactor transporter lesion starves every enzyme that requires that ion, in a
                          compartment those enzymes may be nowhere near
LAYER 5  PATHWAY COUPLING a stopped pathway stops feeding the pathways downstream of it, by measured boundary flow
LAYER 6  SENSOR -> TF     the accumulated deficit is matched to the sensors that detect it, and each sensor emits
                          the transcriptional programme it is known to drive
LAYER 7  READOUT          only the mRNA of layer 6 (plus direct TF targets from layer 1) is predicted, because
                          only mRNA is measured

THE SENSOR TABLE IS DOMAIN KNOWLEDGE AND IS LABELLED AS SUCH. It is not derived from the network -- it is the
part a human biologist supplies. Each entry names the deficit, the sensing mechanism, and the programme. It is
also where the cell line matters: K562 is TP53-NULL, so the p53 sensor is present in biology and DELETED here, and
any prediction routed through it would be wrong for this line specifically.
"""
import collections
import json
import os
import re
from pathlib import Path

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))

# ---------------------------------------------------------------------------------------------------------------
# LAYER 6: the sensor table. deficit -> how the cell detects it -> the TF -> the mRNA programme it drives.
# Domain knowledge, deliberately explicit so each link can be argued with.
# ---------------------------------------------------------------------------------------------------------------
SENSORS = {
    "mito_stress": dict(
        detects="unimported/unprocessed matrix precursors, loss of membrane potential",
        via="OMA1 cleaves DELE1 -> HRI kinase -> eIF2a-P -> ATF4",
        tf="ATF4/ATF5/DDIT3",
        programme=["ATF4", "ATF5", "DDIT3", "TRIB3", "SESN2", "MTHFD2", "MTHFD2L", "SHMT2", "CEBPB",
                   "SLC7A11", "SLC3A2", "EIF4EBP1", "GDF15", "CTH", "XBP1", "VEGFA", "NUPR1"]),
    "er_stress": dict(
        detects="unfolded/mistranslocated protein in the ER lumen",
        via="PERK/IRE1/ATF6 -> XBP1s, ATF4, ATF6f",
        tf="XBP1/ATF4/ATF6",
        programme=["HSPA5", "XBP1", "DDIT3", "HERPUD1", "SEL1L", "EDEM1", "DNAJB9", "DNAJB11", "PDIA6",
                   "PDIA3", "CANX", "CALR", "SSR1", "SEC61B", "ATF4", "MANF", "CRELD2"]),
    "translation_stress": dict(
        detects="stalled/colliding ribosomes, shortage of ternary complex, uncharged tRNA",
        via="GCN2 / ZNF598 -> eIF2a-P -> ATF4 (same output as mito, different trigger)",
        tf="ATF4",
        programme=["ATF4", "ATF5", "DDIT3", "TRIB3", "SESN2", "MTHFD2", "SHMT2", "SLC3A2", "SLC7A11",
                   "EIF4EBP1", "CEBPB"]),
    "proteotoxic": dict(
        detects="unfolded protein in the cytosol",
        via="HSF1 trimerises and enters the nucleus",
        tf="HSF1",
        programme=["HSPA1A", "HSPA1B", "HSPA8", "DNAJB1", "DNAJA1", "HSPH1", "BAG3", "AHSA1", "CHORDC1",
                   "HSPB1", "HSPD1", "HSPE1"]),
    "proteasome_loss": dict(
        detects="reduced proteasome capacity",
        via="NFE2L1 escapes ERAD degradation -> bounce-back response",
        tf="NFE2L1",
        programme=["PSMA1", "PSMA3", "PSMA4", "PSMA5", "PSMA7", "PSMB1", "PSMB3", "PSMB4", "PSMB5", "PSMB6",
                   "PSMC1", "PSMC2", "PSMD1", "PSMD2", "PSMD11", "POMP", "NFE2L1"]),
    "oxidative": dict(
        detects="electrophiles and ROS oxidising KEAP1 cysteines",
        via="NFE2L2 escapes KEAP1",
        tf="NFE2L2",
        programme=["NQO1", "HMOX1", "TXNRD1", "GCLM", "GCLC", "SLC7A11", "SRXN1", "TXN", "PRDX1", "GSR",
                   "G6PD", "FTL"]),
    "sterol_low": dict(
        detects="low ER cholesterol",
        via="SCAP/INSIG release -> SREBF2 cleaved",
        tf="SREBF2",
        programme=["HMGCR", "HMGCS1", "LDLR", "SQLE", "FDFT1", "IDI1", "MVD", "INSIG1", "SREBF2", "DHCR7"]),
    "iron_low": dict(
        detects="low labile iron / failed Fe-S cluster assembly",
        via="IRP1/IRP2 bind IREs -- stabilise TFRC mRNA, block ferritin translation",
        tf="IRP1/2 (post-transcriptional)",
        programme=["TFRC", "SLC11A2", "STEAP3", "ACO1", "IREB2", "FTL", "FTH1", "ALAS2", "SLC25A37"]),
    "export_block": dict(
        detects="mRNA retained in the nucleus, cytoplasmic mRNA falls",
        via="reduced translational load feeds the eIF2a axis",
        tf="ATF4 (indirect)",
        programme=["ATF4", "DDIT3", "SESN2", "EIF4EBP1", "SLC3A2", "TRIB3"]),
}
# K562 is TP53-null: the p53/DNA-damage sensor exists in biology and is ABSENT in this line, so it is defined and
# then deliberately not used. Recording it makes the omission a decision rather than an oversight.
ABSENT_IN_K562 = {"p53_dna_damage": ["CDKN1A", "MDM2", "GADD45A", "BBC3", "TP53I3", "SESN1", "RRM2B"]}

# ---------------------------------------------------------------------------------------------------------------
# LAYER 1-5: which lesion classes a gene belongs to, decided from its own annotation.
# ---------------------------------------------------------------------------------------------------------------
LESION_RULES = [
    ("mito_stress", ("Mitochondrial protein import", "Mitochondrial translation", "mitochondrial",
                     "Mitochondrial biogenesis", "Respiratory electron transport", "Citric acid cycle",
                     "Mitochondrial calcium", "Processing of SMDT1", "Mitochondrial RNA degradation")),
    ("er_stress", ("Unfolded Protein Response", "ER-Phagosome", "Asparagine N-linked glycosylation",
                   "Translocation", "Retrograde transport", "COPI", "COPII", "ER to Golgi", "Golgi")),
    ("translation_stress", ("rRNA processing", "rRNA modification", "Major pathway of rRNA",
                            "Translation", "Eukaryotic Translation", "tRNA processing", "tRNA aminoacylation",
                            "Ribosome", "Nonsense Mediated Decay")),
    ("proteasome_loss", ("Proteasome", "Antigen processing: Ubiquitination", "Cross-presentation")),
    ("proteotoxic", ("HSF1", "chaperon", "Protein folding", "Cellular response to heat stress")),
    ("oxidative", ("Detoxification of Reactive Oxygen", "glutathione", "Peroxisomal")),
    ("sterol_low", ("cholesterol", "Sterol", "SREBP")),
    ("iron_low", ("iron", "Heme", "Porphyrin", "Iron uptake")),
    ("export_block", ("Nuclear Pore", "Transport of Mature", "mRNA export", "Nuclear import",
                      "SUMOylation of SUMOylation proteins", "Nuclear Envelope")),
]


def classify(gene, pathways, compartments, function_text):
    hits = set()
    hay = " ; ".join(pathways) + " ; " + (function_text or "")
    for lesion, keys in LESION_RULES:
        for k in keys:
            if k.lower() in hay.lower():
                hits.add(lesion)
                break
    # compartment is an independent signal from pathway text and can rescue a gene with thin annotation
    if "mitochondrion" in compartments:
        hits.add("mito_stress")
    if "endoplasmic reticulum" in compartments or "golgi" in compartments:
        hits.add("er_stress")
    return hits


def main(tag=""):
    doss = json.loads((OUT / f"ko_pred_dossiers{tag}.json").read_text())
    cmap = json.loads((OUT / "consequence_map.json").read_text())
    paths = {}
    for line in (SP / "ReactomePathways.gmt").read_text().splitlines():
        p = line.split("\t")
        if len(p) > 3:
            paths[p[0]] = list(p[2:])

    preds, stats = {}, collections.Counter()
    for k, v in doss.items():
        m = cmap.get(k, {})
        pw = (v.get("pathways") or []) + (m.get("pathways_member") or [])
        comps = v.get("compartments") or m.get("compartments") or []
        lesions = classify(k, pw, comps, v.get("function"))
        stats.update(lesions or {"<no lesion class>"})

        genes, seen = [], {k}

        def add(xs):
            for x in xs:
                if x and x not in seen and "; " not in x:
                    seen.add(x)
                    genes.append(x)

        # LAYER 1: a TF's own signed regulon is the most direct transcriptional consequence there is
        add([g for g in (v["tf_targets_down"] + v["tf_targets_up"])])
        n_reg = len(genes)
        # LAYER 6: every sensor this lesion trips, in the order the rules fired
        for L in sorted(lesions):
            add(SENSORS[L]["programme"])
        n_sensor = len(genes) - n_reg
        # LAYER 2-5 fallback: only if the lesion classes gave nothing, fall back to physical context
        if not lesions:
            add(m.get("starved_genes", [])[:15])
            add(v.get("reaction_partners", [])[:15])
        preds[k] = {
            "lesions": sorted(lesions),
            "sensors": [f"{L}: {SENSORS[L]['detects']} -> {SENSORS[L]['via']}" for L in sorted(lesions)],
            "reasoning": (f"lesion classes {sorted(lesions) or 'none'}; "
                          f"{n_reg} direct regulon targets + {n_sensor} sensor-programme genes"),
            "n_regulon": n_reg, "n_sensor": n_sensor,
            "predicted_genes": genes[:60],
        }
    json.dump(preds, open(OUT / f"ko_predictions_multilayer{tag}.json", "w"), indent=1)
    print(f"{len(preds)} multi-layer predictions")
    print("  lesion classes assigned:")
    for c, n in stats.most_common():
        print(f"    {c:<22} {n}")
    sz = sorted(len(p["predicted_genes"]) for p in preds.values())
    print(f"  prediction size median {sz[len(sz)//2]}, empty {sum(1 for s in sz if s == 0)}")
    print(f"  p53 sensor defined but NOT used: K562 is TP53-null ({len(ABSENT_IN_K562['p53_dna_damage'])} genes "
          f"withheld)")
    print(f"  -> {OUT/('ko_predictions_multilayer'+tag+'.json')}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "")
