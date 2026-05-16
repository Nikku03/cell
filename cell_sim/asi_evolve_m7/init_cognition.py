"""Seed the ASI-Evolve cognition store with M7-specific knowledge.

Each entry gets embedded with sentence-transformers and stored in
the FAISS index. The Researcher retrieves top-K relevant entries
when proposing the next mutation.

Run once before launching evolve:
    python init_cognition.py > cognition_seed.json
or import COGNITION_SEED directly in main.py.
"""
from __future__ import annotations

import json
import sys


COGNITION_SEED = [
    # ---- Hard biological facts ----
    {
        'title': 'Syn3A genes count',
        'content': (
            'Syn3A has 493 protein-coding genes. The simulator tracks '
            '8572 species rows total (gene copies, mRNA, protein, '
            'protein-metabolite complexes, plus 175 reaction flux trackers '
            'F_*). PINN constrains 169 of the 175 reactions stoichiometrically.'
        ),
    },
    {
        'title': 'Aminoacyl-tRNA synthetases dominate top KOs',
        'content': (
            'The top 13 predicted-essential genes from M7.7 knockout sweep '
            'are all aminoacyl-tRNA synthetases (genes encoding the enzymes '
            'that load amino acids onto tRNAs). These genes are biologically '
            'plausible essentials - charging tRNAs is required for all protein '
            'synthesis. Mutations that improve prediction here likely improve '
            'overall AUROC.'
        ),
    },
    {
        'title': 'Operon at 0509 + 0510',
        'content': (
            'Genes 0509 and 0510 are both tRNAs, adjacent on the genome, '
            'with cross-replicate correlation 0.997. Strong candidate operon. '
            'If model treats them independently, it loses biological coupling.'
        ),
    },
    # ---- ML lessons learned ----
    {
        'title': 'Stochastic noise floor at 0.0716',
        'content': (
            'val_count plateaus at exactly 0.0716 across M7.3 (3 features), '
            'M7.6 (8 features broken kinetic), M7.7 (8 features fixed). This '
            'is irreducible stochastic variance - deterministic models cannot '
            'go lower. Chasing this metric wastes training. Focus on '
            'val_rollout_200_mse which has slack (0.0826 currently).'
        ),
    },
    {
        'title': 'Kinetic + medium features are decorative',
        'content': (
            'Input-ablation tests on M7.7 show kinetic_priors and medium '
            'features are decorative: zeroing their input columns produces '
            'ZERO change in predicted fluxes. The model never learned to use '
            'them. Conclusion: features that need to vary with time are '
            'unhelpful as static inputs. Could try time-varying medium '
            'features (recompute each rollout step) but risky.'
        ),
    },
    {
        'title': 'Learning rate stability range',
        'content': (
            'Learning rates above ~5e-4 destabilize the M7 encoder during '
            'fine-tune (loss explodes). Safe range: 3e-5 to 3e-4. AdamW '
            'with weight_decay 1e-4 is the working baseline.'
        ),
    },
    {
        'title': 'Bias_strength sweet spot',
        'content': (
            'Bias_strength controls fraction of training samples drawn from '
            'weakness-targeted regions. Sweet spot 0.6-0.75. Above 0.85 '
            'the model overfits to weak regions and loses uniform-distribution '
            'performance. Below 0.4 the targeted training has no effect.'
        ),
    },
    {
        'title': 'Flux MSE already very low',
        'content': (
            'val_mse_flux is ~0.0010 (already 70× lower than count). The PINN '
            'head constrains flux predictions. Targeting flux rows more '
            'aggressively rarely helps; targeting count and cumulative rows '
            'has more headroom.'
        ),
    },
    {
        'title': 'Simulator-init artifact',
        'content': (
            'Timesteps t=0..9 contain huge non-physical residuals (negative '
            'event counts on F_* rows, M_lac__L_c jumps +86,059). These are '
            'the simulator finishing its boot protocol. Training on these '
            'pairs teaches artifacts. t_skip_initial=10 already enforces this; '
            'consider raising to 20-30 for safety.'
        ),
    },
    # ---- Architecture hints ----
    {
        'title': 'Multi-step prediction head (M7.8)',
        'content': (
            'M7.8 adds auxiliary heads predicting x_{t+2}..x_{t+H} from the '
            'same encoder forward pass. Gives H× more gradient signal per '
            'forward. aux_loss_weight=0.5 is the working balance; above 1.0 '
            'overwhelms the primary head; below 0.1 gives no contribution.'
        ),
    },
    {
        'title': 'PINN constraint mechanics',
        'content': (
            'For SBML-mapped rows, M7 predicts Δx via Δx = S · v where S is '
            'the stoichiometric matrix and v is the GNN-predicted rate vector. '
            'This guarantees mass conservation by construction. Mutations '
            'that change the rate clip or PINN residual weight need to be '
            'careful - signed_log1p / signed_expm1 bridges keep gradient '
            'well-conditioned, breaking the bridge causes exp() overflow.'
        ),
    },
    # ---- Suggested directions ----
    {
        'title': 'Per-pathway loss weighting hypothesis',
        'content': (
            'Different pathways have different importance. Central metabolism '
            '(PGI, PFK, FBA, etc.) drives growth. Transcription/translation '
            'drives gene expression. Weighting these pathways differently in '
            'the loss could help. The kinetic_params_new.xlsx has subsystem '
            'labels (Central, Nucleotide, Lipid, Amino Acid, Cofactor, '
            'Transport) - could group rows by subsystem and weight.'
        ),
    },
    {
        'title': 'Curriculum mutation hypothesis',
        'content': (
            'Currently fine-tune is single-stage. A 2-stage curriculum '
            '(short rollouts first, then long) might warm up the model '
            'better. But each stage adds wall-clock cost; budget is 10 '
            'minutes per candidate.'
        ),
    },
]


def write_seed(path: str = 'cognition_seed.json') -> None:
    with open(path, 'w') as f:
        json.dump(COGNITION_SEED, f, indent=2)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'cognition_seed.json'
    write_seed(out)
    print(f'wrote {len(COGNITION_SEED)} cognition seeds to {out}',
          file=sys.stderr)
