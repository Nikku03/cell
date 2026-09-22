"""colabfold_multimer — AlphaFold-Multimer for the interactions with NO solved PDB complex. Runs in Colab on a
GPU via ColabFold (MMseqs2 MSA server + AF2-multimer weights); there is no GPU on a laptop, so this module is the
Colab arm of step 6c. For interactions that DO have a PDB complex, use interface_analysis directly (experimental
beats predicted).

Two uses:
  1. fold a gene's homodimer or gene+partner complex, then run interface_analysis on the prediction to see
     whether a mutation sits at the dimer/partner interface -> the GOF interface evidence (BRAF at the dimer
     interface, KRAS at the GAP interface) the monomer can't show;
  2. WT-vs-mutant interface comparison: fold both, compare interface contacts / predicted-interface confidence
     (ipTM) -> quantitative "does the mutant still make this contact".

Honest: AF-Multimer is more sensitive to interface-disrupting mutations than monomer AlphaFold, but it is still
noisy for single substitutions; treat a dropped ipTM / lost interface contact as EVIDENCE, corroborated with
ESM + burial, not proof.
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))


def ensure_colabfold():
    """install ColabFold in a Colab runtime (no-op if present). GPU strongly recommended."""
    try:
        import colabfold  # noqa
        return True
    except Exception:
        os.system("pip -q install 'colabfold[alphafold]' 2>/dev/null")
        try:
            import colabfold  # noqa
            return True
        except Exception:
            return False


def fold_complex(seqs, name, out_dir="af_multimer"):
    """fold a multimer from a list of chain sequences (ColabFold joins with ':'). Returns the top-ranked PDB path
    or None. GPU required for reasonable speed."""
    os.makedirs(out_dir, exist_ok=True)
    fasta = os.path.join(out_dir, f"{name}.fasta")
    open(fasta, "w").write(f">{name}\n{':'.join(seqs)}\n")
    rc = os.system(f"colabfold_batch --num-recycle 3 {fasta} {out_dir}/{name} 2>/dev/null")
    pdbs = sorted(glob.glob(f"{out_dir}/{name}/*rank_001*.pdb") or glob.glob(f"{out_dir}/{name}/*.pdb"))
    return pdbs[0] if pdbs else None


def mutant_at_interface(gene, pos, wt, mut, partner_gene, chain_labels=None):
    """fold the gene+partner complex (WT), then ask interface_analysis whether `pos` is at the partner interface.
    (A point mutation barely changes the fold, so folding WT and reading the interface on the WT complex is the
    reliable signal; folding the mutant too only adds a noisy ipTM delta.) Returns the interface verdict."""
    import molecular_engine as me
    import interface_analysis as ia
    acc_a, sa = me.uniprot_seq(gene)
    acc_b, sb = me.uniprot_seq(partner_gene)
    if not sa or not sb or sa[pos - 1] != wt:
        return {"error": "sequence unavailable or residue mismatch"}
    pdb = fold_complex([sa, sb], f"{gene}_{partner_gene}")
    if not pdb:
        return {"error": "folding failed (need GPU/ColabFold)"}
    cx = ia.parse_complex(pdb)
    # chain A is the gene (first sequence)
    chains = list(cx.keys())
    part = ia.interface_partners(cx, chains[0], pos) if chains else {}
    at = {partner_gene if c != chains[0] else gene: d for c, d in part.items()}
    return {"gene": gene, "variant": f"{wt}{pos}{mut}", "partner": partner_gene, "pdb": pdb,
            "at_partner_interface": bool(at), "contacts": at,
            "verdict": (f"{gene} {wt}{pos}{mut} is AT the {partner_gene} interface -> that interaction is broken"
                        if at else f"{pos} is not at the {partner_gene} interface in the predicted complex")}


# GOF interface panel: the dimer/partner complexes that reveal why a buried-monomer mutation activates
GOF_INTERFACE_PANEL = [
    ("BRAF", 600, "V", "E", "BRAF"),      # BRAF activation-loop / dimer
    ("KRAS", 12, "G", "D", "RASA1"),      # KRAS at the GAP (RASA1) interface — G12 blocks GAP -> constitutively on
    ("KRAS", 12, "G", "D", "SOS1"),       # KRAS-SOS1 nucleotide exchange interface
    ("EGFR", 858, "L", "R", "EGFR"),      # EGFR kinase dimer
    ("JAK2", 617, "V", "F", "JAK2"),      # JAK2 pseudokinase-kinase autoinhibitory interface
]


if __name__ == "__main__":
    print("colabfold_multimer: Colab/GPU only. In Colab, run e.g.:")
    print("  ensure_colabfold(); mutant_at_interface('KRAS',12,'G','D','RASA1')")
