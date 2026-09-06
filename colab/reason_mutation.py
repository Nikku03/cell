"""reason_mutation — run the cell's reasoning WITH a specific mutation. This fuses the two validated reasoners:

  A) VARIANT layer  (reasoned_variant.py): does THIS mutation break the protein? AlphaMissense = the reliable
     function-aware call; ΔΔG + functional-site distance = the interpretable WHY; a sickle-cell-style
     charge→hydrophobic + fold-intact pattern OVERRIDES a benign call into "uncertain" (the GOF blind spot).
  B) CELL layer     (cellos.reason: AUC 0.80, calibrated): does the cell DEPEND on that protein? — chains
     independent essentiality lines (Perturb-seq shock, FBA viability, LOEUF, complex membership, network hub)
     to a calibrated P(essential).

The fusion is the reasoning the user asked for: a mutation only matters to the cell if it (1) damages the protein
AND (2) the cell can't buffer that protein's loss. The join NAMES the regime, because essentiality (cell-fitness)
and pathogenicity (disease) are DIFFERENT axes — they agree for loss-of-function in load-bearing genes (classic
inborn errors), and DIVERGE for tumor suppressors (LOF is pathogenic but not cell-lethal) and gain-of-function.
Reasoning that hid that divergence would be wrong; this states it.

  reason_mutation(gene, uniprot, pos, wt, mut) -> printed trace + dict.  Honest confidence, honest blind spots.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))


class MutationReasoner:
    def __init__(self, kernel=None):
        from reasoned_variant import ReasonedVariant
        self.rv = ReasonedVariant()
        if kernel is not None:
            self.k = kernel
        else:
            from cellos import CellKernel
            self.k = CellKernel()

    # ---- pull the structured essentiality read out of the gene-level reasoning string ----
    def _gene_layer(self, gene):
        s = self.k.reason(gene)
        concl = "UNCERTAIN"
        for tag in ("NON-ESSENTIAL", "ESSENTIAL"):
            if tag in s:
                concl = tag
                break
        pe = None
        if "P≈" in s:
            try:
                pe = float(s.split("P≈")[1].split("%")[0]) / 100.0
            except (ValueError, IndexError):
                pe = None
        nyes = None
        if "independent lines say essential" in s:
            try:
                nyes = s.split("──")[1].split(" of ")[0].strip()
                nyes = f"{nyes} of {s.split(' of ')[1].split(' ')[0]}"
            except (ValueError, IndexError):
                nyes = None
        return {"conclusion": concl, "p_essential": pe, "lines": nyes, "text": s}

    # ---- the fusion ----
    def reason(self, gene, uniprot, pos, wt, mut, verbose=True):
        var = self.rv.predict(gene, uniprot, pos, wt, mut)     # A: does the mutation break the protein?
        gl = self._gene_layer(gene)                             # B: does the cell depend on it?

        call = var["call"]
        protein_hit = call == "likely damaging"
        protein_ok = call == "likely tolerated"
        blind = bool(var.get("ml_blind_spot")) or "uncertain" in call or call == "ambiguous"
        essential = gl["conclusion"] == "ESSENTIAL"
        buffered = gl["conclusion"] == "NON-ESSENTIAL"

        # regime + cell-level conclusion (the reasoning join)
        if blind:
            regime = "GAIN-OF-FUNCTION / BLIND SPOT"
            cell = ("DO NOT TRUST a benign call — the per-residue predictors miss this mechanism (aggregation/"
                    "gain-of-function). Needs a functional assay. The cell-dependency layer can't rescue this.")
            conf = "LOW (honest uncertainty)"
        elif protein_ok:
            regime = "TOLERATED"
            cell = "no protein-level damage predicted → no cell consequence expected (whatever the gene's importance)."
            conf = "MEDIUM-HIGH" if (var.get("confidence") or 0) > 0.5 else "MEDIUM"
        elif protein_hit and essential:
            regime = "LOSS-OF-FUNCTION in a LOAD-BEARING gene  (both layers agree)"
            cell = ("the variant disables a protein the cell CANNOT buffer → expect a strong phenotype. This is the "
                    "regime where the two independent reasonings reinforce each other — highest confidence.")
            conf = "HIGH"
        elif protein_hit and buffered:
            regime = "FUNCTION-ALTERING in a NON-fitness gene  (layers diverge — READ THE REGIME)"
            cell = ("the protein is damaged, but the cell-FITNESS lens says the cell survives its loss. That does NOT "
                    "mean benign: tumor-suppressors (LOF pathogenic, not cell-lethal), dominant-negative/GOF, and "
                    "tissue-specific genes are pathogenic on a DIFFERENT axis than cell-fitness essentiality. Trust "
                    "the variant call for pathogenicity; the essentiality layer is the wrong ruler here.")
            conf = "MEDIUM (variant-driven; essentiality axis N/A)"
        else:  # protein_hit and UNCERTAIN gene layer
            regime = "PROTEIN HIT, cell-dependency unresolved"
            cell = "variant damages the protein; the cell-dependency lines split — treat the cell impact as uncertain."
            conf = "MEDIUM-LOW"

        out = {
            "mutation": var["variant"], "regime": regime,
            "variant_layer": {"call": call, "score": var.get("score"), "source": var.get("source"),
                              "mechanisms": var["reasoning"]["mechanisms"], "why": var["reasoning"]["why"],
                              "ml_blind_spot": var.get("ml_blind_spot")},
            "cell_layer": {"essentiality": gl["conclusion"], "p_essential": gl["p_essential"], "lines": gl["lines"]},
            "cell_level_conclusion": cell, "joint_confidence": conf,
            "caveats": ["AlphaMissense is strong but misses gain-of-function (sickle-cell class) — the override flags it",
                        "cell essentiality is a cell-fitness lens (DepMap/Perturb-seq, cell-line/context specific), "
                        "NOT disease pathogenicity — they diverge for suppressors/GOF/tissue-specific genes",
                        "ΔΔG is the noisy mechanism rung (r~0.4) — trust it for ranking strong effects, not per-call"],
        }
        if verbose:
            self._print(out, gl)
        return out

    @staticmethod
    def _print(o, gl):
        v = o["variant_layer"]; c = o["cell_layer"]
        print("=" * 92)
        print(f"REASONING WITH A MUTATION:  {o['mutation']}")
        print("=" * 92)
        print("  A) VARIANT LAYER — does the mutation break the protein?")
        print(f"       call: {v['call'].upper()}   (AlphaMissense {v['score']}, {v['source']})")
        print(f"       mechanism: {', '.join(v['mechanisms'])}")
        print(f"       why: {v['why']}")
        print("  B) CELL LAYER — does the cell depend on this protein?   (cellos.reason, AUC 0.80, calibrated)")
        print(f"       essentiality: {c['essentiality']}"
              + (f"   calibrated P≈{c['p_essential']:.0%}" if c['p_essential'] is not None else "")
              + (f"   ({c['lines']} lines)" if c['lines'] else ""))
        print(f"  ── REGIME: {o['regime']}")
        print(f"     CELL-LEVEL CONCLUSION: {o['cell_level_conclusion']}")
        print(f"     joint confidence: {o['joint_confidence']}")
        print("=" * 92)


PANEL = [
    # gene, uniprot, pos, wt, mut, known clinical truth (for the honest check)
    ("MLH1", "P40692", 618, "K", "T", "PATHOGENIC — Lynch syndrome (DNA mismatch-repair, load-bearing)"),
    ("TP53", "P04637", 175, "R", "H", "PATHOGENIC — Li-Fraumeni hotspot (dominant-negative/GOF tumor suppressor)"),
    ("TP53", "P04637", 72, "P", "R", "BENIGN — P72R common polymorphism (same gene, contrast)"),
    ("HBB", "P68871", 7, "E", "V", "PATHOGENIC via GAIN-OF-FUNCTION — sickle cell (the blind-spot case)"),
]


def demo():
    mr = MutationReasoner()
    results = []
    for gene, up, pos, wt, mut, truth in PANEL:
        o = mr.reason(gene, up, pos, wt, mut)
        o["clinical_truth"] = truth
        print(f"     CLINICAL TRUTH: {truth}\n")
        results.append(o)
    json.dump(results, open("outputs/orphan/reason_mutation_demo.json", "w"), indent=1)
    return results


if __name__ == "__main__":
    demo()
