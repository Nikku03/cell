"""REM applied to protein-protein docking.

Four sub-problems, each mapped to a REM strength:
  1. rigid pose search        -> FFT correlation      (rem.fftcorr)
  2. side-chain repacking     -> variable elimination (rem.factorgraph, mode="min")
  3. flexible refinement      -> clustered/chain elimination, sample only where treewidth blows
  4. binding free energy      -> sum-product elimination, -kT ln Z with entropy included

THE CENTRAL LESSON THIS SUITE MUST KEEP SEPARABLE: REM makes the SEARCH exact. Accuracy is
then bounded by the SCORING FUNCTION. Every report splits those two error sources, and the
bound-to-bound ablation exists precisely to measure search error with scoring error removed.
"""
