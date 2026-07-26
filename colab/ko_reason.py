"""REASONED KNOCKOUT PREDICTIONS: what the cell lacks, and what it does about it.

This is the prediction step of the sealed protocol in ko_predict.py. The answers file is NOT read here.

THE REASONING, WHICH IS THE POINT. For each knockout: what does the protein do -> what can the cell no longer do ->
what compensatory or collapsing transcriptional program follows. Perturb-seq reads mRNA, so a prediction only counts
if the mechanism ends in a transcriptional consequence. Three facts from the data shape every call below:

  K562 IS TP53-NULL. So the textbook nucleolar-stress and DNA-damage answer -- p53 stabilisation, CDKN1A, MDM2 --
  cannot fire. Any prediction resting on p53 would be wrong for this cell line specifically.

  THE 43 TIDE GENES ARE EXACTLY THE RIBOSOMAL PROTEINS AND THE mtDNA-ENCODED OXPHOS SUBUNITS, and they are excluded
  from the truth. So "ribosome down" and "OXPHOS down" -- the two most obvious answers for most of these knockouts --
  are unscoreable by construction. Predicting them would be predicting the tide.

  NO HISTONE GENES ARE MEASURED. The classic replication-arrest signature (replication-dependent histone mRNA
  collapse) therefore cannot be used, which was checked before committing to it rather than after.

What is left that IS measurable and IS mechanistic: the integrated stress response (ATF4 axis, p53-independent, so
it survives in K562), the ribosome-biogenesis regulon itself, the spliceosome, the S-phase/E2F programme, nuclear
transport, mitochondrial import and protease machinery, and the ER unfolded-protein response.

TWO KNOCKOUTS ARE DELIBERATE NEGATIVE CONTROLS. OR4K1 is an olfactory receptor and PPY is a pancreatic hormone
acting on GPCRs; neither has a job in an erythroleukaemia line. If the reasoning is worth anything it should fail on
these, and predicting a confident signature for them would show the method generates stories regardless of biology.
They are predicted as "no coherent programme" -- a broad low-information guess -- and are expected to score at the
frequency baseline.
"""
import json
from pathlib import Path

OUT = Path("outputs/orphan")

# ---------------------------------------------------------------------------------------------------------------
# Signature panels. Each is a transcriptional programme with a named trigger, not a list of "related genes".
# ---------------------------------------------------------------------------------------------------------------
ISR = ["ATF4", "ATF5", "DDIT3", "TRIB3", "SESN2", "MTHFD2", "MTHFD2L", "SHMT2", "CEBPB", "SLC7A11",
       "SLC3A2", "SLC7A5", "CTH", "EIF4EBP1", "GDF15", "XBP1"]
RIBI = ["NOP56", "NOP58", "NCL", "NPM1", "FBL", "BMS1", "GNL3", "GNL2", "DDX21", "LTV1", "RRP9", "NOC4L",
        "DDX10", "DDX18", "PWP2", "RCL1", "PNO1", "RRS1", "BOP1", "PES1", "WDR12", "UTP14A", "DHX37",
        "TBL3", "WDR36", "NOP14", "NOP2", "NOP16", "RRP12", "NIFK", "PAK1IP1", "RRP1", "RRP15"]
SPLICE = ["SNRPD1", "SNRPB", "SNRPA1", "SF3B1", "SF3A1", "PRPF8", "PRPF19", "PRPF31", "PRPF3", "PRPF4",
          "EFTUD2", "SNRNP200", "LSM2", "SRSF1", "SRSF2", "SRSF3", "U2AF1", "U2AF2", "DDX23", "DDX39B",
          "TXNL4A", "RBM22", "PHF5A", "SNRNP70", "CDC5L", "DHX15", "AQR", "BUD31"]
REPL = ["MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "CDC45", "GINS1", "GINS2", "GINS3", "GINS4",
        "CLSPN", "CHEK1", "FEN1", "LIG1", "PCNA", "POLA1", "POLE", "PRIM1", "RFC2", "RFC3", "RFC4", "RFC5",
        "RPA1", "RPA2", "RPA3", "TIMELESS", "RRM2", "TYMS", "E2F1"]
PORE = ["NUP107", "NUP133", "NUP153", "NUP155", "NUP160", "NUP188", "NUP205", "NUP210", "NUP214", "NUP35",
        "NUP37", "NUP43", "NUP54", "NUP62", "NUP85", "NUP88", "NUP93", "NUP98", "SEH1L", "SEC13",
        "RANBP2", "RAN", "KPNB1", "KPNA2", "NXF1", "ALYREF", "THOC1", "XPO1"]
MITO = ["TIMM17A", "TIMM23", "TIMM44", "TIMM50", "TIMM13", "TIMM10", "TOMM20", "TOMM22", "TOMM40", "HSPA9",
        "PMPCA", "PMPCB", "LONP1", "CLPP", "AFG3L2", "DNAJC19", "GRPEL1", "PHB1", "PHB2", "ROMO1", "SPG7"]
ER = ["HSPA5", "XBP1", "DDIT3", "HERPUD1", "SEL1L", "EDEM1", "DNAJB9", "DNAJB11", "SEC61A1", "SEC61B",
      "SEC62", "SEC63", "SRPRA", "SSR1", "ATF4", "CANX", "CALR", "PDIA3", "PDIA6"]
CHAP = ["HSPA8", "DNAJA1", "DNAJB1", "HSPH1", "BAG3", "AHSA1", "CHORDC1", "HSPB1", "HSPD1", "HSPE1"]
EXOSOME = ["EXOSC1", "EXOSC2", "EXOSC3", "EXOSC4", "EXOSC5", "EXOSC6", "EXOSC7", "EXOSC8", "EXOSC9",
           "EXOSC10", "MTREX", "ZCCHC8", "RBM7", "DIS3"]
POL2 = ["POLR2A", "POLR2B", "POLR2C", "POLR2E", "POLR2F", "POLR2G", "POLR2H", "POLR2I", "POLR2J", "POLR2K",
        "POLR2L", "GTF2B", "GTF2F1", "GTF2H1", "CDK7", "CCNH", "MNAT1", "RPAP1", "RPAP2", "RPAP3", "GPN1",
        "GPN2", "GPN3", "URI1", "RUVBL1", "RUVBL2"]
CHROM = ["WDR5", "ASH2L", "DPY30", "RBBP5", "KMT2A", "KMT2D", "KANSL1", "KAT8", "HCFC1", "SETD1A", "MEN1"]

# ---------------------------------------------------------------------------------------------------------------
# Per-knockout reasoning. `why` is the mechanism; `panels` is what that mechanism predicts transcriptionally.
# ---------------------------------------------------------------------------------------------------------------
CALLS = {
  # --- ribosome biogenesis / nucleolus: nucleolar stress, but p53-null so the answer is the RiBi regulon + ISR ---
  "PNO1":    ("SSU processome / 40S maturation. Cell cannot finish small-subunit assembly; immature pre-40S "
              "accumulates, translation capacity falls -> ISR, and the RiBi regulon collapses with it.", [RIBI, ISR]),
  "PWP2":    ("UTP-B subcomplex of the SSU processome; same failure point as PNO1, one step earlier.", [RIBI, ISR]),
  "RCL1":    ("RNA 3'-terminal phosphate cyclase-like, required for 18S rRNA cleavage at A2. Same programme.",
              [RIBI, ISR]),
  "RRP12":   ("Nucleolar export factor for pre-ribosomal particles.", [RIBI, ISR]),
  "NIFK":    ("MKI67-interacting nucleolar phosphoprotein, rRNA maturation.", [RIBI, ISR]),
  "PAK1IP1": ("Named for PAK1 but it is a nucleolar ribosome-biogenesis factor; the function text is misleading "
              "and the compartment plus partners say nucleolus.", [RIBI, ISR]),
  "POLR1A":  ("RNA Pol I catalytic subunit -- no 47S pre-rRNA at all. The most upstream RiBi lesion here.",
              [RIBI, ISR]),
  "POLR1B":  ("RNA Pol I second-largest subunit, same lesion as POLR1A.", [RIBI, ISR]),
  "POP1":    ("RNase P / RNase MRP shared subunit: tRNA 5' processing AND rRNA processing both fail.",
              [RIBI, ISR]),
  "OGFOD1":  ("Prolyl hydroxylase of RPS23 in the decoding centre. Loss degrades translation fidelity and "
              "triggers a stress response rather than a biogenesis collapse.", [ISR, RIBI]),

  # --- spliceosome: intron retention -> broad transcript loss, and the spliceosome regulon itself ---
  "PRPF3":   ("U4/U6 tri-snRNP. Splicing fails globally; short-half-life transcripts drop first.", [SPLICE, ISR]),
  "PRPF4":   ("U4/U6 tri-snRNP partner of PRPF3.", [SPLICE, ISR]),
  "PRPF19":  ("PRP19/CDC5L complex, catalytic activation of the spliceosome; also DNA repair.",
              [SPLICE, REPL]),
  "PRPF40A": ("Bridges U1 to the branch point; 3'-end processing too.", [SPLICE]),
  "SNRNP70": ("U1 snRNP 70K -- 5' splice site recognition, the first committed step.", [SPLICE, ISR]),
  "RBM22":   ("Delivers the catalytic metal to the spliceosome core.", [SPLICE]),
  "PHF5A":   ("SF3B component, branch-point recognition.", [SPLICE]),
  "NKAP":    ("Splicing factor, spliceosome B complex.", [SPLICE]),
  "RBM8A":   ("Exon junction complex core (with EIF4A3, MAGOH). Loss disables EJC-dependent NMD and export, so "
              "NMD substrates rise and the EJC/export machinery is the readable programme.",
              [SPLICE, PORE]),
  "PPP1R10": ("PNUTS-PP1: PP1 targeting for RNA Pol II CTD dephosphorylation and 3'-end processing.",
              [SPLICE, POL2]),

  # --- replication / genome maintenance: S-phase arrest -> E2F programme down. Histones unmeasured. ---
  "POLA1":   ("Pol alpha-primase catalytic subunit: no primers, no replication initiation. Cells arrest in S "
              "and the E2F/S-phase programme collapses. p53-null, so no CDKN1A arm.", [REPL]),
  "POLE":    ("Leading-strand polymerase; same S-phase collapse.", [REPL]),
  "PCNA":    ("Sliding clamp for every replicative polymerase and most repair. Broadest replication lesion.",
              [REPL, ISR]),
  "MMS22L":  ("MMS22L-TONSL, homologous recombination at collapsed forks. Fork instability -> checkpoint.",
              [REPL]),
  "POT1":    ("Shelterin: represses ATR at telomeres. Loss gives telomere-derived DNA damage signalling.",
              [REPL]),
  "RAD21":   ("Cohesin kleisin. Sister chromatid cohesion fails and, separately, cohesin shapes enhancer-promoter "
              "contacts, so transcription is broadly perturbed alongside the cell-cycle arm.", [REPL, CHROM]),
  "PIAS4":   ("SUMO E3 ligase acting on the DNA damage response (TP53BP1, RNF8/RNF168 axis).", [REPL]),

  # --- nuclear pore / transport: mRNA export block ---
  "NUP54":   ("Central FG channel of the NPC. mRNA export and nucleocytoplasmic transport fail.", [PORE, ISR]),
  "NUP62":   ("Central FG channel with NUP54/NUP58.", [PORE, ISR]),
  "NUP88":   ("Cytoplasmic-face NUP with NUP214; export side of the pore.", [PORE, ISR]),
  "NUP98":   ("FG nucleoporin, mRNA export and also a transcriptional regulator.", [PORE, ISR]),

  # --- mitochondria: import/protease failure -> OMA1-DELE1-HRI -> ISR. p53-independent, fires in K562. ---
  "PAM16":   ("Regulates the TIM23 import motor. Matrix protein import stalls -> mitochondrial stress -> ISR. "
              "The mtDNA-encoded OXPHOS genes that would also move are tide and unscoreable.", [ISR, MITO]),
  "PMPCB":   ("Mitochondrial processing peptidase beta: presequences are not cleaved, unprocessed precursors "
              "accumulate in the matrix -> the same ISR.", [ISR, MITO]),
  "PHB2":    ("Prohibitin complex, inner-membrane scaffold and m-AAA protease partner.", [MITO, ISR]),
  "ROMO1":   ("Inner-membrane, ROS modulator, contributes to the MICOS/import machinery.", [MITO, ISR]),
  "PNPT1":   ("PNPase: mitochondrial RNA degradation and RNA import. Loss gives mitochondrial dsRNA "
              "accumulation, which is sensed -- so an interferon arm is plausible on top of the ISR.",
              [MITO, ISR]),
  "PPRC1":   ("PRC / PGC-1-related coactivator: drives the mitochondrial biogenesis programme with NRF1 and "
              "GABPA. Loss should lower the nuclear-encoded mitochondrial regulon.", [MITO, ISR]),

  # --- RNA Pol II assembly and chromatin ---
  "GPN3":    ("GPN-loop GTPase required for RNA Pol II assembly and nuclear import. Pol II never reaches the "
              "nucleus, so Pol II subunits and its assembly chaperones are the readable programme.", [POL2]),
  "RPAP1":   ("RNA Pol II-associated protein 1, interface between Pol II and the R2TP/prefoldin chaperone.",
              [POL2]),
  "RUVBL1":  ("AAA+ ATPase of R2TP: assembles Pol II, snoRNPs, PIKKs and the TIP60/INO80 remodellers. Hits "
              "Pol II assembly and ribosome biogenesis at once.", [POL2, RIBI]),
  "RBBP5":   ("Core of every SET1/MLL H3K4 methyltransferase complex, with WDR5/ASH2L/DPY30.", [CHROM, POL2]),
  "OGT":     ("O-GlcNAc transferase; also cleaves HCF1. Modifies Pol II CTD and thousands of nuclear proteins, "
              "so the effect is broad rather than pathway-specific.", [CHROM, ISR]),

  # --- nuclear RNA surveillance / export ---
  "NRDE2":   ("Binds MTREX and restrains nuclear exosome targeting. Loss deregulates exosome-mediated decay.",
              [EXOSOME, SPLICE]),

  # --- ER ---
  "SEC62":   ("Post-translational translocation into the ER. Small secretory precursors mistarget -> UPR.",
              [ER, ISR]),

  # --- TFs with a measured signed regulon: predict the regulon itself, direction included ---
  "NRF1":    ("Nuclear respiratory factor 1: the master activator of nuclear-encoded mitochondrial genes. Its "
              "53 measured activated targets should FALL when it is removed.", []),
  "SFPQ":    ("Splicing factor proline/glutamine-rich; also a transcriptional regulator with a measured regulon.",
              [SPLICE]),

  # --- deliberate negative controls ---
  "OR4K1":   ("Olfactory receptor. Not expressed and has no role in an erythroleukaemia line. If the method is "
              "honest it should have nothing specific to say, and a confident prediction here would be evidence "
              "the method invents stories.", []),
  "PPY":     ("Pancreatic polypeptide, a secreted GPCR ligand for NPY receptors. No job in K562. Same status as "
              "OR4K1: predicted to have no coherent programme.", []),

  # --- remaining, weaker calls ---
  "RBM14":   ("Nuclear coactivator / paraspeckle protein, RUNX2 partner. Transcription-adjacent, no sharp "
              "programme; predicted broadly.", [POL2, SPLICE]),
  "SLC7A6OS":("Uncharacterised; the only handle is that SLC7A6 is an amino-acid transporter, so an amino-acid "
              "stress arm is a guess rather than a mechanism. Low confidence, stated as such.", [ISR]),
}


def main():
    doss = json.loads((OUT / "ko_pred_dossiers.json").read_text())
    preds = {}
    for k, v in doss.items():
        why, panels = CALLS.get(k, ("no specific call", [ISR]))
        genes = []
        for p in panels:
            for g in p:
                if g != k and g not in genes:
                    genes.append(g)
        # A TF's own signed regulon is a direct, gene-level prediction and outranks any generic panel.
        regulon = [g for g in (v["tf_targets_down"] + v["tf_targets_up"]) if g != k]
        genes = regulon + [g for g in genes if g not in regulon]
        # Reaction partners are the machine the protein sits in; include a few for complex members.
        for g in v["reaction_partners"][:8]:
            if g not in genes and g != k and "; " not in g:
                genes.append(g)
        preds[k] = {"reasoning": why, "n_regulon": len(regulon), "predicted_genes": genes[:60]}
    json.dump(preds, open(OUT / "ko_predictions.json", "w"), indent=1)
    print(f"{len(preds)} predictions written, median size "
          f"{sorted(len(p['predicted_genes']) for p in preds.values())[len(preds)//2]}")
    print(f"  -> {OUT/'ko_predictions.json'}")


if __name__ == "__main__":
    main()
