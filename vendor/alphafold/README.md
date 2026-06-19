# Vendored AlphaFold source (Apache-2.0)

From github.com/google-deepmind/alphafold (AF2). Used as the reference for
porting the Evoformer cross-talk ops into scripts/af_evoformer.py, molded to
the genes-as-residues mapping (genome=protein, gene=residue, organism=MSA row).

- modules.py        : Evoformer modules. OuterProductMean (Alg.10) is ported
                      verbatim (exact einsums) into scripts/af_evoformer.py.
- common_modules.py : Linear/LayerNorm helpers (reference only).
- LICENSE           : Apache License 2.0.

We cannot RUN AF here (needs JAX + GPU + genome-scale tensors). The vendored
code is the reference; the NumPy port is the runnable, gradient-checked
adaptation. To scale up, run the real JAX modules on GPU with the same
genes-as-residues feature construction.
