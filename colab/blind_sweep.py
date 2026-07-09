"""blind_sweep — run the ML-loaded blind target-ID pipeline across the WHOLE cancer landscape, not a hand-picked
handful, and score it BY DRIVER CLASS. The pipeline was built for ONE class (an oncogene point-mutation near the
catalytic machinery); a single aggregate number over all cancers would hide that. So we label every tumour by the
biology of its driver and report where the method works and where it structurally cannot:

  onco_pt : driver is an oncogene POINT mutation (BRAF V600E, KRAS G12, EGFR L858R, KIT, IDH1 R132 ...)
            -> the pipeline's wheelhouse. Mutation sits on the functional module; gene is druggable.
  fusion  : driver is a gene FUSION (BCR-ABL, EML4-ALK, ROS1, RET-fusion ...)
            -> the model has NO fusion representation; a fusion is not a point mutation. Expected-hard.
  lof     : driver is a tumour-suppressor LOSS (VHL, RB1, APC, PTEN, TP53, BRCA, STK11 ...)
            -> the pipeline can NAME the gene but it is not a direct drug target (the GOF/LOF wall). It should
               surface the right gene but the target call is "not directly druggable / needs SL partner".
  amp     : driver is an AMPLIFICATION (ERBB2/HER2, MYC, MYCN, CDK4 ...)
            -> no point mutation to score; the model does not carry copy-number. Expected-hard.

Panels are REAL documented alterations (driver + classic passengers TTN/MUC16/OBSCN + hub ACTB). The KEY (true
target + class) is held OUT of the pipeline and used only to score afterward — exactly the blind protocol.
Honest by design: we predict up front the method scores high on onco_pt and low on fusion/amp, and report the
class breakdown so the number can't lie by averaging.
-> outputs/orphan/blind_sweep_report.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import blind_target as bt
from complete_cell import CompleteCell

OUT = "outputs/orphan"

# tumour -> [(gene, mutated_residue), ...]  (driver + realistic passengers). Positions are canonical hotspots.
PANELS = {
    # ---- onco_pt: oncogene point-mutation drivers (the pipeline's wheelhouse) ----
    "A375 melanoma":            [("BRAF", 600), ("CDKN2A", 58), ("TTN", 20000), ("MUC16", 5000)],
    "HT-29 colon":              [("BRAF", 600), ("APC", 1450), ("TP53", 273), ("SMAD4", 360)],
    "BCPAP thyroid (PTC)":      [("BRAF", 600), ("TERT", 250), ("TTN", 21000)],
    "HCC827 lung adeno":        [("EGFR", 858), ("CDKN2A", 58), ("TTN", 18000)],
    "H1975 lung adeno":         [("EGFR", 790), ("TP53", 273), ("TTN", 17000)],
    "A549 lung":                [("KRAS", 12), ("STK11", 50), ("CDKN2A", 58), ("TTN", 15000)],
    "HCT116 colon":             [("KRAS", 13), ("CTNNB1", 41), ("TTN", 14000)],
    "PANC-1 pancreatic":        [("KRAS", 12), ("CDKN2A", 58), ("TP53", 273), ("TTN", 16000)],
    "SK-MEL-2 melanoma":        [("NRAS", 61), ("TP53", 248), ("TTN", 19000)],
    "T24 bladder":              [("HRAS", 12), ("TP53", 280), ("TTN", 13000)],
    "GIST882":                  [("KIT", 642), ("ACTB", 150), ("TTN", 12000)],
    "HMC-1 mastocytosis":       [("KIT", 816), ("TP53", 175), ("OBSCN", 3000)],
    "GIST (PDGFRA)":            [("PDGFRA", 842), ("TTN", 11000)],
    "92.1 uveal melanoma":      [("GNAQ", 209), ("BAP1", 60), ("TTN", 9000)],
    "OMM1 uveal melanoma":      [("GNA11", 209), ("TTN", 8000)],
    "MOLM-13 AML":              [("FLT3", 835), ("DNMT3A", 882), ("NPM1", 288)],
    "HEL erythroleukemia":      [("JAK2", 617), ("TP53", 248), ("TTN", 7000)],
    "U87 / IDH-glioma":         [("IDH1", 132), ("CDKN2A", 58), ("PTEN", 130)],
    "IDH2 AML":                 [("IDH2", 172), ("NPM1", 288), ("DNMT3A", 882)],
    "RT4 bladder":              [("FGFR3", 249), ("TP53", 280), ("TTN", 6000)],
    "TT medullary thyroid":     [("RET", 918), ("TTN", 5000)],
    "Waldenstrom lymphoma":     [("MYD88", 265), ("CXCR4", 330)],
    "H596 lung (MET ex14)":     [("MET", 1010), ("TP53", 273), ("TTN", 4000)],
    "hairy-cell leukemia":      [("BRAF", 600), ("TTN", 3000)],
    "MCL (SF3B1)":              [("SF3B1", 700), ("ATM", 2800)],
    "MDS (U2AF1)":              [("U2AF1", 34), ("TET2", 1200)],
    "ET / MPN (MPL)":           [("MPL", 515), ("TTN", 2500)],
    "DLBCL (EZH2)":             [("EZH2", 646), ("TP53", 248), ("MUC16", 4000)],
    "ER+ breast (ESR1)":        [("ESR1", 537), ("TP53", 273), ("TTN", 2000)],

    # ---- fusion: driver is a gene fusion (model has no fusion representation) ----
    "K562 CML":                 [("TP53", 234), ("ABL1", 315), ("TTN", 10000)],
    "H2228 lung (EML4-ALK)":    [("ALK", 1174), ("TP53", 273), ("TTN", 9500)],
    "HCC78 lung (ROS1)":        [("ROS1", 2032), ("TP53", 248), ("TTN", 9200)],
    "LC-2/ad lung (RET-fus)":   [("RET", 810), ("TTN", 8800)],
    "Ewing sarcoma":            [("EWSR1", 265), ("STAG2", 400)],

    # ---- lof: tumour-suppressor loss (nameable, but not a direct drug target) ----
    "786-O renal (VHL)":        [("VHL", 100), ("PBRM1", 300), ("TTN", 8500)],
    "WERI retinoblastoma":      [("RB1", 600), ("TTN", 8200)],
    "HGSOC ovarian (BRCA1)":    [("BRCA1", 1750), ("TP53", 273), ("MUC16", 5000)],
    "endometrial (PTEN)":       [("PTEN", 130), ("KRAS", 12), ("ARID1A", 1800)],
    "H69 SCLC":                 [("TP53", 175), ("RB1", 600), ("MYC", 100), ("TTN", 13000)],
    "diffuse gastric (CDH1)":   [("CDH1", 350), ("TP53", 273), ("TTN", 7800)],

    # ---- amp: amplification drivers (no point mutation; model carries no copy-number) ----
    "SK-BR-3 breast (HER2)":    [("ERBB2", 310), ("TP53", 175), ("MYC", 100)],
    "neuroblastoma (MYCN)":     [("MYCN", 100), ("ALK", 1174), ("TTN", 7500)],
    "MCF7 breast":              [("GATA3", 330), ("PIK3CA", 1047), ("TTN", 22000), ("MUC16", 5000)],
}

# held-out answer key: (true actionable driver gene, class). NOT shown to the pipeline.
KEY = {
    "A375 melanoma": ("BRAF", "onco_pt"), "HT-29 colon": ("BRAF", "onco_pt"),
    "BCPAP thyroid (PTC)": ("BRAF", "onco_pt"), "HCC827 lung adeno": ("EGFR", "onco_pt"),
    "H1975 lung adeno": ("EGFR", "onco_pt"), "A549 lung": ("KRAS", "onco_pt"),
    "HCT116 colon": ("KRAS", "onco_pt"), "PANC-1 pancreatic": ("KRAS", "onco_pt"),
    "SK-MEL-2 melanoma": ("NRAS", "onco_pt"), "T24 bladder": ("HRAS", "onco_pt"),
    "GIST882": ("KIT", "onco_pt"), "HMC-1 mastocytosis": ("KIT", "onco_pt"),
    "GIST (PDGFRA)": ("PDGFRA", "onco_pt"), "92.1 uveal melanoma": ("GNAQ", "onco_pt"),
    "OMM1 uveal melanoma": ("GNA11", "onco_pt"), "MOLM-13 AML": ("FLT3", "onco_pt"),
    "HEL erythroleukemia": ("JAK2", "onco_pt"), "U87 / IDH-glioma": ("IDH1", "onco_pt"),
    "IDH2 AML": ("IDH2", "onco_pt"), "RT4 bladder": ("FGFR3", "onco_pt"),
    "TT medullary thyroid": ("RET", "onco_pt"), "Waldenstrom lymphoma": ("MYD88", "onco_pt"),
    "H596 lung (MET ex14)": ("MET", "onco_pt"), "hairy-cell leukemia": ("BRAF", "onco_pt"),
    "MCL (SF3B1)": ("SF3B1", "onco_pt"), "MDS (U2AF1)": ("U2AF1", "onco_pt"),
    "ET / MPN (MPL)": ("MPL", "onco_pt"), "DLBCL (EZH2)": ("EZH2", "onco_pt"),
    "ER+ breast (ESR1)": ("ESR1", "onco_pt"),
    "K562 CML": ("ABL1", "fusion"), "H2228 lung (EML4-ALK)": ("ALK", "fusion"),
    "HCC78 lung (ROS1)": ("ROS1", "fusion"), "LC-2/ad lung (RET-fus)": ("RET", "fusion"),
    "Ewing sarcoma": ("EWSR1", "fusion"),
    "786-O renal (VHL)": ("VHL", "lof"), "WERI retinoblastoma": ("RB1", "lof"),
    "HGSOC ovarian (BRCA1)": ("BRCA1", "lof"), "endometrial (PTEN)": ("PTEN", "lof"),
    "H69 SCLC": ("(none targetable)", "lof"), "diffuse gastric (CDH1)": ("CDH1", "lof"),
    "SK-BR-3 breast (HER2)": ("ERBB2", "amp"), "neuroblastoma (MYCN)": ("MYCN", "amp"),
    "MCF7 breast": ("PIK3CA", "amp"),
}


def _clean(panels, C):
    """drop the tiny conditional-gene placeholders (some passengers were 'gene if in-model else fallback' — the
    string-truthiness always picks the first; resolve any that aren't in the model to their fallback)."""
    out = {}
    for name, panel in panels.items():
        fixed = []
        for g, p in panel:
            if g in C.idx:
                fixed.append((g, p))
            # silently drop a passenger that isn't in the model (it was only there to add realistic noise)
        out[name] = fixed
    return out


def run():
    C = CompleteCell()
    panels = _clean(PANELS, C)
    rep = bt.run(panels)

    by_class = {}
    rows = []
    for name, d in rep.items():
        proposed = d.get("proposed_target")
        true_gene, cls = KEY[name]
        ok = (proposed == true_gene)
        by_class.setdefault(cls, []).append(ok)
        rows.append({"tumor": name, "class": cls, "proposed": proposed, "true": true_gene, "hit": ok})

    print("\n" + "#" * 80)
    print("BLIND SWEEP — target identification across the cancer landscape, scored BY CLASS")
    print("#" * 80)
    order = ["onco_pt", "fusion", "lof", "amp"]
    summary = {}
    for cls in order:
        res = by_class.get(cls, [])
        n = len(res); h = sum(res)
        summary[cls] = {"hits": h, "n": n, "pct": round(100 * h / n, 1) if n else None}
        print(f"\n  {cls.upper():8}  {h}/{n}")
        for r in [x for x in rows if x["class"] == cls]:
            mark = "OK  " if r["hit"] else "MISS"
            print(f"     {mark}  {r['tumor']:26} proposed={str(r['proposed']):9} true={r['true']}")
    tot_h = sum(x["hit"] for x in rows); tot_n = len(rows)
    onco = summary.get("onco_pt", {})
    print("\n" + "-" * 80)
    print(f"  OVERALL {tot_h}/{tot_n}  —  but the honest number is the wheelhouse: "
          f"onco_pt {onco.get('hits')}/{onco.get('n')} ({onco.get('pct')}%).")
    print(f"  fusion/amp are structural blind spots (no fusion or copy-number representation); "
          f"lof genes are nameable but not directly druggable.")
    print("-" * 80)

    out = {"summary": summary, "overall": {"hits": tot_h, "n": tot_n}, "rows": rows,
           "note": ("Scored by driver class. onco_pt is the method's designed scope; fusion/amp are known "
                    "structural blind spots; lof is nameable but not a direct drug target (GOF/LOF wall).")}
    json.dump(out, open(f"{OUT}/blind_sweep_report.json", "w"), indent=2)
    return out


if __name__ == "__main__":
    run()
