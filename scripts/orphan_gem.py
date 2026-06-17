"""PATH-ORPHAN, mechanistic rung (Colab) -- GEM + environment + in-silico KO.

The one approach that EXTRAPOLATES to unseen conditions (mechanism generalizes;
data interpolates). Auto-build a genome-scale metabolic model from the genome
(gapseq -- ~80% essentiality accuracy per the benchmark), set the medium from
Fitness Browser condition metadata, run single-gene-deletion FBA, and read
conditional essentiality for the metabolic genes.

KILL-GATE (designed today): does the in-silico KO recover the ~55 METABOLIC rogue
essentials that sequence/ESM scored 0 on -- AND correctly extrapolate to a
held-out FB condition? If gap-filling just fakes it, it fails the metabolic genes
too.

This script holds the testable logic (FB-condition -> in-silico medium mapping;
the gene-deletion essentiality readout via gene-protein-reaction rules). The
heavy steps (gapseq build, cobrapy FBA) run on Colab.

--smoke : validate media mapping + GPR single-deletion essentiality on a toy model
--real  : gapseq build + media + FBA single-gene-deletion screen (Colab)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "orphan"

# minimal map: FB condition fields -> exchange reactions to OPEN (uptake allowed)
CARBON_EX = {
    "glucose": "EX_glc__D", "D-glucose": "EX_glc__D", "putrescine": "EX_ptrc",
    "acetate": "EX_ac", "succinate": "EX_succ", "pyruvate": "EX_pyr",
    "lactate": "EX_lac__L", "glycerol": "EX_glyc", "citrate": "EX_cit",
}
MINIMAL_BASE = ["EX_nh4", "EX_pi", "EX_so4", "EX_o2", "EX_h2o", "EX_h", "EX_k",
                "EX_mg2", "EX_fe2"]


def condition_to_media(condition_1, media="minimal"):
    """FB condition -> set of OPEN exchange reactions (the in-silico medium).
    Minimal base + the carbon/nitrogen source named in condition_1."""
    open_ex = set(MINIMAL_BASE)
    key = (condition_1 or "").strip()
    if key in CARBON_EX:
        open_ex.add(CARBON_EX[key])
    elif "rich" in (media or "").lower() or "LB" in (media or ""):
        open_ex |= {f"EX_aa_{i}" for i in range(20)}     # rich: all AAs provided
        open_ex.add("EX_glc__D")
    return open_ex


def gene_essential_toy(reactions, gpr, biomass_inputs, open_ex, gene):
    """Toy in-silico knockout: a gene is essential under this medium iff deleting
    it (removing reactions whose GPR requires it) makes any biomass precursor
    unreachable from the open exchanges. Pure-Python reachability stand-in for
    FBA single-gene-deletion (real path uses cobrapy).
      reactions: {rxn: (set inputs, set outputs)}
      gpr: {rxn: set(genes that are each individually required)}  (AND logic)
      biomass_inputs: set of metabolites required for growth
      open_ex: set of exchange rxns supplying metabolites
    """
    def reachable(active_rxns):
        have = set()
        for r in active_rxns:
            if r in open_ex:
                have |= reactions.get(r, (set(), set()))[1]
        changed = True
        while changed:
            changed = False
            for r in active_rxns:
                ins, outs = reactions.get(r, (set(), set()))
                if ins <= have and not (outs <= have):
                    have |= outs; changed = True
        return have
    # knock out: drop reactions that require this gene
    ko_rxns = {r for r, gs in gpr.items() if gene in gs}
    active_wt = set(reactions)
    active_ko = active_wt - ko_rxns
    wt_ok = biomass_inputs <= reachable(active_wt)
    ko_ok = biomass_inputs <= reachable(active_ko)
    return wt_ok and not ko_ok          # essential = WT grows, KO doesn't


def run_real(args):
    bridge = OUT / f"bridge_{args.org}.parquet"
    if not bridge.exists():
        print("ERROR: run orphan_bridge.py --real first"); return 2
    print("=" * 70)
    print(f"GEM mechanistic rung -- {args.org}")
    print("=" * 70)
    print("  Colab steps:")
    print("  1) pip install gapseq cobra; gapseq doall genome.faa  -> model.xml")
    print("  2) for each FB condition: condition_to_media(condition_1) -> medium")
    print("  3) cobra single_gene_deletion(model) per medium -> essential genes")
    print("  4) KILL-GATE: recall of the 55 metabolic rogue essentials vs the")
    print("     sequence-feature 0; AND leave-one-condition-out extrapolation")
    print("  (gapseq + cobra FBA run on Colab; this driver maps media + reads KO)")
    return 0


def run_smoke():
    print("=" * 70)
    print("GEM SMOKE -- media mapping + in-silico single-gene-deletion readout")
    print("=" * 70)
    m = condition_to_media("putrescine", media="minimal")
    print(f"  putrescine medium opens: EX_ptrc present = {'EX_ptrc' in m}")
    assert "EX_ptrc" in m and "EX_nh4" in m
    assert "EX_ptrc" not in condition_to_media("glucose")
    print("  FB condition -> in-silico medium mapping: OK")
    # toy network: C(carbon) --r1[geneA]--> I --r2[geneB]--> BIO
    # plus a bypass r3[geneC]: C --> BIO directly (redundant route)
    reactions = {
        "EX_C": (set(), {"C"}), "r1": ({"C"}, {"I"}),
        "r2": ({"I"}, {"BIO"}), "r3": ({"C"}, {"BIO"}),
    }
    gpr = {"r1": {"geneA"}, "r2": {"geneB"}, "r3": {"geneC"}}
    bio = {"BIO"}; open_ex = {"EX_C"}
    # geneA alone NOT essential (bypass r3 via geneC); knock BOTH routes to test
    essA = gene_essential_toy(reactions, gpr, bio, open_ex, "geneA")
    print(f"  geneA essential (has bypass r3)? {essA} (expect False)")
    assert essA is False, "redundant route -> not essential"
    # remove bypass: now geneA essential
    reactions2 = {k: v for k, v in reactions.items() if k != "r3"}
    gpr2 = {k: v for k, v in gpr.items() if k != "r3"}
    essA2 = gene_essential_toy(reactions2, gpr2, bio, open_ex, "geneA")
    print(f"  geneA essential (no bypass)? {essA2} (expect True)")
    assert essA2 is True, "sole route -> essential"
    # geneC (only the bypass) essential when it's the sole route to BIO from C?
    # with both routes, geneC not essential (r1+r2 path remains)
    essC = gene_essential_toy(reactions, gpr, bio, open_ex, "geneC")
    assert essC is False
    print("  toy single-gene-deletion essentiality (redundancy-aware): OK")
    print("\n=== GEM SMOKE PASS ===")
    print("  media mapping + in-silico KO readout validated. gapseq+cobra on Colab.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--org", default="beril_RalstoniaGMI1000")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
