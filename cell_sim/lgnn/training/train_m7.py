"""M7 training - deterministic cell-state dynamics surrogate.

Goal: same architecture quality as M3/M6, training wallclock cut 10-20x.
Stochastic head deferred to M8.

Architectural recipe (see cell_sim/lgnn/models/gnn_v7_hybrid.py):
  - M3 encoder (CellGNNv3-style): 3x CfC-attention layers, hidden=64,
    2-channel input (count + cross-replicate sigma), RATE_LAW /
    MASS_BALANCE edge split.
  - PINNHead with use_residual=True: SBML rows update via the hardwired
    Δx = S * expm1(v_log) bridge; non-SBML rows (RP/RPM/PM/DM/...) get
    a learned residual delta head. Mass conservation exact on the SBML
    block, dynamics learned everywhere else.

Loss (per category weights from M3, flux-primary from M-PINN):
  L = w_count * MSE_count(x_next_log)
    + w_flux  * MSE_flux (v_log vs F_* observations)
    + w_cum   * MSE_cum  (PM/RPM/DM cumulative counters)
    + lambda_attn * attention_entropy_warmup
    + lambda_mass_balance * MSE_anti_drift_on_SBML  (default off; see below)

The optional anti-drift term penalises ||x_t_log - x_0_log||^2 averaged
over SBML-covered species. PINNHead already enforces dx = S*v exactly,
so this is purely a stability prior against long-rollout drift; default
0.0, dial up only if 7200-step rollout shows SBML metabolites wandering.

Speed levers (all togglable, defaults to a 'fast' preset):
  (a) samples_per_epoch        : subsample (rep,t) pairs instead of full
                                  enumeration. 20_000 by default.
  (b) batch_size               : 256 by default (M3/M6 used 64).
  (c) use_compile              : torch.compile(mode='reduce-overhead').
                                  Default False — M6 saw an allocator
                                  leak across k_curriculum changes.
  (d) use_bf16                 : default True. Encoder + head run native bf16.
  (e) truncated_bptt_window    : at long k_curriculum, only backprop
                                  through the last N rollout steps.
                                  Default 4. Set None to disable.
  (f) p_ss_max, p_ss_warmup    : scheduled sampling from M6 (Bengio 2015).
  (g) use_checkpoint           : default False (with bs=256 + tbptt this
                                  fits comfortably on 48 GB). Flip on if
                                  you cap memory.

Each lever can be toggled independently so you can A/B test what's
buying speed vs hurting R^2. The trainer logs samples/sec per epoch
and the per-category val MSEs so the wallclock/quality tradeoff is
visible.

Primary metric (not measured per-epoch; use full_rollout_evaluation):
  7200-step autoregressive rollout R^2 on val rep50. Per-category R^2
  and the mass-balance residual time series are also reported.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from cell_sim.lgnn.data.species_graph import SpeciesGraph
from cell_sim.lgnn.data.stoichiometric_matrix import (
    load_stoichiometric_matrix_for_pinn,
)
from cell_sim.lgnn.models.gnn_v7_hybrid import CellGNNv7Hybrid, count_parameters
from cell_sim.lgnn.training.train_m1_axis2_fast import (
    _gather_windows, _index_iterator, _torch_dtype, _warmup_ramp,
    preload_to_gpu,
)
from cell_sim.lgnn.training.train_m3 import (
    M3TrainConfig, _device, _k_for_epoch, categorise_row_indices,
    compute_variance_channel,
)


@dataclass
class M7TrainConfig(M3TrainConfig):
    """M7 config - extends M3 with PINN bits + speed levers.

    Defaults are the 'fast' preset; pass overrides at the call site to
    opt back into M3/M6-style full-coverage training when you need
    apples-to-apples comparisons.
    """
    # ---- PINN-specific ----
    # Path to the stoichiometric-matrix source. Strongly prefer the SBML
    # XML (Syn3A_updated.xml); the .lm-based loader was unreliable for
    # column mapping. Required, no default.
    sbml_path_for_S: str = ''
    rate_clip: float = 6.0
    use_residual: bool = True

    # Flux-primary loss weights (Critique 11): direct v_log <-> F_*
    # supervision drives learning; count + cumulative are secondary.
    weight_flux:       float = 1.0
    weight_cumulative: float = 0.5
    weight_count:      float = 0.05

    # ---- Speed levers ----
    # (a) Subsample (rep, t) pairs per epoch. 49 * 7185 = 352_065 pairs
    # in the full corpus; 20_000 is ~5.7% per epoch, ~22% across 4
    # epochs. Set None for full enumeration (matches M3 throughput).
    samples_per_epoch: Optional[int] = 20_000
    # (b) Larger batch. With bs=256 + bf16 + 96 GB headroom this is fine.
    batch_size: int = 256
    # (c) torch.compile. Default OFF — M6 saw allocator leak across
    # k_curriculum changes. Flip ON only if you stick with a single k.
    use_compile: bool = False
    compile_mode: str = 'reduce-overhead'
    # (d) bf16 (inherits M3 default).
    # (e) Truncated BPTT. At k_cur > tbptt_window, only the last
    # `truncated_bptt_window` steps backprop; earlier steps roll the
    # state forward under torch.no_grad(). Saves activation memory +
    # backward time at long k. Set None to disable.
    truncated_bptt_window: Optional[int] = 4
    # (f) Scheduled sampling (carried over from M6).
    scheduled_sampling: bool = True
    p_ss_max: float = 0.5
    p_ss_warmup_steps: int = 2000
    # (g) Activation checkpointing. Recomputes layer forwards during backward
    # instead of storing activations. ~4x memory savings for ~33% wall cost.
    # Default ON: at bs=256 with the M3 encoder + PINN head + sigma channel,
    # the k=4 rollout step's edge-message tensors blow past 94 GB without it
    # (verified empirically — same OOM mode M6 hit before flipping this).
    # tbptt_window doesn't help here because it only activates when
    # k_cur > tbptt_window, so the k=4 epoch is full-BPTT.
    use_checkpoint: bool = True

    # ---- k-curriculum: same shape as M6/M-PINN ----
    k_curriculum: tuple = (1, 4, 8, 16)
    rollout_gamma: float = 1.0          # equal-weight rollout (M6)
    max_k: int = 16
    n_epochs: int = 4

    # ---- Soft anti-drift mass-balance term (default OFF) ----
    # When > 0, adds lambda * mean( (x_t_log - x_0_log)^2 ) over SBML-
    # covered species. PINNHead already enforces dx = S*v exactly, so
    # the soft term mainly buys stability against long-rollout drift.
    lambda_mass_balance: float = 0.0

    # ---- Logging / budget ----
    log_every: int = 25
    wall_clock_budget_s: float = 1 * 3600.0      # 1 h default

    # The variance channel is on by default (matches M3); flip to 1 to
    # ablate or to reuse a checkpoint trained without it.
    n_input_channels: int = 2

    # ---- M7.1 - row-type static features ----
    # Pass 8 one-hot row-type indicators (is_gene / is_mrna / is_protein
    # / is_membrane_protein / is_translation_complex / is_degradation /
    # is_mrna_decay / is_cofactor_or_flux) as additional input channels.
    # Categorical only - the model can learn "this is a protein" vs "this
    # is an mRNA" but cannot use them to identify individual rows (8 flags
    # encode 8 types, not 8572 rows). Avoids the M5 shortcut-learning
    # failure mode while giving the residual head the row-type
    # discrimination it needs to predict different deltas for proteins
    # vs mRNAs vs gene rows. Default ON.
    use_role_features: bool = True

    # ---- M7.3 - spatial + proteomics features ----
    # When enabled, additionally inject:
    #   spatial   : 12 per-locus columns from the lm-derived parquet
    #               (distance to membrane, fractions in cytoplasm/DNA/etc.)
    #   proteomics: 13 per-locus columns from initial_concentrations.xlsx
    #               (5 function one-hot + 6 localization one-hot + 2 scalar)
    # M5 broke training with 41 features that included row-uniquely-identifying
    # gene_class columns (protein length, EC numbers etc.). Spatial + role +
    # proteomics is 33 cols, all categorical-ish or coarse-continuous, well
    # below the shortcut threshold.
    use_spatial_features:    bool = False
    use_proteomics_features: bool = False
    spatial_parquet:        Optional[str] = None
    proteomics_xlsx:        Optional[str] = None
    gene_class_csv:         str = 'memory_bank/data/syn3a_gene_class_features.csv'

    # ---- M7.4 - Tier-2/3 optional extras (each independent + disable-able) ----
    # Each addition can be toggled separately. If a data file is missing,
    # the loader emits a warning and returns a zero tensor, so the
    # training keeps running with that feature group disabled.
    use_kinetic_priors:        bool = False
    use_complex_constraints:   bool = False
    use_medium_features:       bool = False
    kinetic_params_xlsx:       Optional[str] = None
    complex_formation_xlsx:    Optional[str] = None
    medium_xlsx:               Optional[str] = None

    # ---- M7.5 prep - additional optional extras (prepped but default OFF) ----
    # Each loader is implemented in cell_sim/lgnn/extras/ and tested in
    # cell_sim/tests/test_extras_modules.py. Activation is one config
    # flag flip; the loader handles missing files with a WARNING.
    use_regulatory_features:    bool = False
    use_ribosome_subunit:       bool = False
    use_thermodynamics:         bool = False
    protein_metabolites_xlsx:   Optional[str] = None
    large_subunit_xlsx:         Optional[str] = None
    gibbs_csv_path:             Optional[str] = None

    # ---- M8 upgrade 6/10: ESM-2 protein language model features ----
    # Pre-computed embeddings keyed by locus_4d, mean-pooled per protein,
    # projected to esm2_proj_dim via a seeded random projection. Adds
    # protein representation power without retraining ESM-2 itself.
    use_esm2_protein_features:  bool = False
    esm2_embeddings_parquet:    Optional[str] = None
    esm2_proj_dim:              int = 64

    # ---- M7.6 - exclude simulator-initialization timesteps from training ----
    # The narrate_window analysis revealed that t=0 -> t=1 in the trajectory
    # data is dominated by simulator initialization artifacts (massive
    # unexplained residuals on metabolites, negative event counts on flux
    # rows). Training on these (rep, t<100) pairs corrupts the model's
    # dynamics with init-only noise. Setting t_skip_initial > 0 excludes
    # the first N seconds from both training-pair sampling and per-epoch
    # validation. Default 0 preserves prior behaviour.
    t_skip_initial: int = 0

    # ---- M7.2 - per-stage curriculum knobs for long-horizon training ----
    # If set, these override the scalar samples_per_epoch and
    # truncated_bptt_window for each curriculum stage independently. Lets
    # you spend lots of compute at low k (cheap, broad coverage) and pay
    # for high-k rollouts only at the very end with reduced sample budget.
    # Example:
    #   k_curriculum             = (1,     4,    32,   128,  512)
    #   samples_per_epoch_stage  = (20000, 20000, 10000, 5000, 1000)
    #   tbptt_window_per_stage   = (None,  4,    2,    1,    1)
    # tbptt=1 in particular cuts long-k cost dramatically: only the last
    # rollout step backprops (and gets all the cumulative-error gradient
    # signal); the earlier steps run under torch.no_grad() at ~2x the
    # forward speed and zero activation memory. Compute at k=512 with
    # tbptt=1 is closer to ~256x a k=1 step than the ~512x you'd expect
    # from full BPTT.
    # If None (default), falls back to the scalar samples_per_epoch /
    # truncated_bptt_window applied uniformly.
    samples_per_epoch_per_stage: Optional[tuple] = None
    tbptt_window_per_stage:      Optional[tuple] = None

    # ---- M7.8 - random-anchor warmup + multi-step head ----
    # New training strategy that decouples *rollout horizon* from
    # *gradient horizon* by running a no-grad warmup phase to expose
    # the model to its own predictions at random anchor timesteps,
    # then a single forward pass that predicts H future timesteps in
    # parallel via auxiliary heads.
    #
    # Set train_strategy='random_anchor' to activate; default 'standard'
    # preserves the M7.1-7.7 behaviour.
    #
    # random_anchor_warmup_max_per_stage: max warmup length to sample from
    #   per curriculum stage. e.g. (10, 100, 1000, 7000) lets the model
    #   see anchors from t=10..[10,110,1010,7010]. Picked uniformly per
    #   batch.
    # multi_step_horizon: how many future timesteps the auxiliary heads
    #   should predict (in addition to the standard 1-step prediction).
    #   H=4 gives 4 parallel predictions for the cost of ~1 forward pass
    #   plus 1 backward — DeepSeek-V3 MTP-style.
    # aux_loss_weight: weighting of the auxiliary multi-step loss
    #   relative to the primary one-step loss.
    train_strategy:                       str = 'standard'
    random_anchor_warmup_max_per_stage:   Optional[tuple] = None
    multi_step_horizon:                   int = 1
    aux_loss_weight:                      float = 0.5

    # Chunked-anchor amortization: split each long warmup into K
    # equally-spaced anchors, training at each. n_anchor_chunks=10 means
    # one 7000-step warmup yields 10 training signals instead of 1 -
    # an effective 10x speedup. Setting to 1 disables (vanilla random
    # anchor behaviour). The implementation caps K at warmup_steps so
    # short warmups don't produce duplicate anchors.
    n_anchor_chunks_per_warmup:           int = 1

    # Optional path to a previously-trained M7 checkpoint to warm-start
    # the encoder weights from. Newly added params (aux_heads etc.) keep
    # their random init via strict=False loading. Saves ~stage-0 worth
    # of training when going M7.7 -> M7.8.
    init_from_checkpoint:                 str = ''

    # ---- M8 stochastic head: predict (μ, σ), train with NLL ----
    # When True, the model includes a per-species log_σ head and the
    # primary loss becomes Gaussian NLL instead of MSE. This is the
    # single biggest accuracy unlock - it lets the model express
    # "this transition is irreducibly noisy" (e.g. count noise on rare
    # species) instead of being penalised for not predicting variance.
    # The deterministic val_count = 0.0716 noise floor that every M7.*
    # variant hits is broken by this head.
    #
    # nll_loss_weight: weight on the NLL loss term relative to MSE.
    #   0.0 -> pure MSE (deterministic M7 behaviour)
    #   1.0 -> NLL dominates
    #   0.5 -> mixed; useful for warm-starting from a deterministic ckpt
    use_stochastic_head:                  bool = False
    nll_loss_weight:                      float = 1.0
    mse_warmup_steps:                     int = 0      # train MSE-only for first N steps, then mix in NLL

    # ---- Selected optimiser ----
    # 'adamw'    : default, M7.*-compatible
    # 'ademamix' : Apple, 2024. EMA + slow EMA combination, faster convergence
    # 'lion'     : Google, 2023. Sign-based update, often beats AdamW
    optimizer_name:                       str = 'adamw'

    # ---- SWA + EMA ----
    # When True, maintains an exponential moving average of model
    # weights (decay=ema_decay) and ALSO a Stochastic Weight Average
    # over the final swa_n_epochs of training. EMA weights are used
    # at every validation; SWA weights at the final eval and saved
    # alongside the regular best checkpoint.
    use_ema:                              bool = False
    ema_decay:                            float = 0.999
    use_swa:                              bool = False
    swa_start_epoch:                      int = 2     # begin SWA at this epoch
    swa_lr:                               float = 1e-4

    # ---- Inference-time toggles passed to model construction ----
    # When True, model is wrapped with torch.compile(mode='reduce-overhead')
    # AFTER training so the saved checkpoint runs ~2x faster at inference.
    # NOT applied during training (compile + checkpointing don't play well).
    compile_at_eval:                      bool = False

    # ---- M8 upgrades 7/9/10 (opt-in architectural swaps) ----
    # encoder_backbone:
    #   'cfc'    -> default M7 CfC-Attention layers, uses graph edges
    #   'mamba2' -> Mamba-2 SSM stack (linear-time, ignores edges)
    #               requires `pip install mamba-ssm causal-conv1d`
    encoder_backbone:                     str = 'cfc'
    # use_moe_residual: replace PINNHead's single-Linear residual head with
    # an 11-expert MoE statically routed by row-name prefix. Each row goes
    # to exactly one expert; no learned router. ~450 extra params.
    use_moe_residual:                     bool = False
    moe_n_experts:                        int = 11
    # use_node_flux_head: swap PINNHead for NeuralODEFluxHead which
    # integrates dx/dt = S·v_θ(x) over [t, t+1] via RK4 (train) / dopri5
    # (infer). Requires `pip install torchdiffeq`. Slower to train but
    # may give smoother long-horizon rollouts.
    use_node_flux_head:                   bool = False
    node_solver:                          str = 'rk4'
    node_step_size:                       float = 0.5


# ----------------------------------------------------------------------
# Subsampled (rep, t) iterator (speed lever a)
# ----------------------------------------------------------------------

def _build_role_features(row_names) -> torch.Tensor:
    """Build (N, 8) one-hot row-type indicators.

    Categorical species-type label per row, derived from row name
    prefix. Used as static input channels in M7.1 so the residual
    head can differentiate proteins from mRNAs from gene rows from
    complexes without M5's shortcut-learning failure mode.

    Returns
    -------
    torch.FloatTensor of shape (len(row_names), 8) where column j is
    1.0 if the row matches the j-th category in
    cell_sim.lgnn.data.static_node_features.ROLE_COLS, else 0.0.
    """
    from cell_sim.lgnn.data.static_node_features import (
        _role_indicators, ROLE_COLS,
    )
    N = len(row_names)
    features = torch.zeros(N, len(ROLE_COLS), dtype=torch.float32)
    for i, name in enumerate(row_names):
        roles = _role_indicators(name)
        for j, col in enumerate(ROLE_COLS):
            features[i, j] = float(roles[col])
    return features


def _build_combined_static_features(
    row_names,
    *,
    use_role: bool = True,
    use_spatial: bool = False,
    use_proteomics: bool = False,
    use_kinetic_priors: bool = False,
    use_complex_constraints: bool = False,
    use_medium_features: bool = False,
    use_regulatory_features: bool = False,
    use_ribosome_subunit: bool = False,
    use_thermodynamics: bool = False,
    use_esm2_protein_features: bool = False,
    spatial_parquet: Optional[str] = None,
    proteomics_xlsx: Optional[str] = None,
    gene_class_csv: Optional[str] = None,
    kinetic_params_xlsx: Optional[str] = None,
    complex_formation_xlsx: Optional[str] = None,
    medium_xlsx: Optional[str] = None,
    protein_metabolites_xlsx: Optional[str] = None,
    large_subunit_xlsx: Optional[str] = None,
    gibbs_csv_path: Optional[str] = None,
    esm2_embeddings_parquet: Optional[str] = None,
    esm2_proj_dim: int = 64,
    verbose: bool = True,
) -> Optional[torch.Tensor]:
    """Compose role / spatial / proteomics features into one (N, F) tensor.

    Builds only the columns the user enables. Skips the 19 gene_class
    columns by default - those triggered M5's shortcut-learning failure
    because protein_length_aa, has_gene_name etc. encoded row identity.
    Role (8) + spatial (12) + proteomics (13) = 33 cols, comfortably
    below the M5 threshold.

    The spatial parquet and proteomics xlsx are only consulted if the
    corresponding `use_*` flag is True AND the path is provided. If a
    requested file is missing, raises a clear error.

    Returns None when nothing is enabled (model gets dynamic-only input).
    """
    parts: list[torch.Tensor] = []
    col_names: list[str] = []

    if use_role:
        from cell_sim.lgnn.data.static_node_features import ROLE_COLS
        role = _build_role_features(row_names)
        parts.append(role)
        col_names.extend(ROLE_COLS)
        if verbose:
            print(f'  + role features: {role.shape[1]} cols')

    if use_spatial or use_proteomics:
        # The static_node_features builder produces ALL 54 columns; we
        # slice afterwards. It requires gene_class_csv to exist even if
        # we discard those columns - point at the in-repo default.
        from cell_sim.lgnn.data.static_node_features import (
            build_static_node_features,
            ALL_STATIC_COLS, SPATIAL_COLS, PROTEOMICS_COLS,
        )
        from pathlib import Path

        gc_csv = gene_class_csv or 'memory_bank/data/syn3a_gene_class_features.csv'
        if not Path(gc_csv).exists():
            raise FileNotFoundError(
                f'gene_class_csv {gc_csv} not found; cannot build full '
                f'static features. Either generate it via '
                f'scripts/build_syn3a_gene_class_features.py or disable '
                f'use_spatial_features / use_proteomics_features.'
            )
        if use_spatial and (spatial_parquet is None
                            or not Path(spatial_parquet).exists()):
            raise FileNotFoundError(
                f'use_spatial_features=True but spatial_parquet '
                f'{spatial_parquet} not found. Run step B (extractor) first.'
            )
        if use_proteomics and (proteomics_xlsx is None
                               or not Path(proteomics_xlsx).exists()):
            raise FileNotFoundError(
                f'use_proteomics_features=True but proteomics_xlsx '
                f'{proteomics_xlsx} not found.'
            )

        full = build_static_node_features(
            row_names=row_names,
            gene_class_csv=gc_csv,
            spatial_parquet=spatial_parquet if use_spatial else None,
            proteomics_xlsx=proteomics_xlsx  if use_proteomics else None,
            initial_state_signed_log1p=None,
            verbose=verbose,
        )
        # full is (N, len(ALL_STATIC_COLS)); slice out the cols we want.
        col_to_idx = {c: i for i, c in enumerate(ALL_STATIC_COLS)}
        if use_spatial:
            sp_idx = [col_to_idx[c] for c in SPATIAL_COLS]
            sp = full[:, sp_idx].float()
            parts.append(sp)
            col_names.extend(SPATIAL_COLS)
            if verbose:
                print(f'  + spatial features: {sp.shape[1]} cols')
        if use_proteomics:
            pr_idx = [col_to_idx[c] for c in PROTEOMICS_COLS]
            pr = full[:, pr_idx].float()
            parts.append(pr)
            col_names.extend(PROTEOMICS_COLS)
            if verbose:
                print(f'  + proteomics features: {pr.shape[1]} cols')

    # ---- Tier-2/3 extras ----
    if use_kinetic_priors:
        from cell_sim.lgnn.extras.kinetic_priors import build_kinetic_prior_features
        kp = build_kinetic_prior_features(
            kinetic_params_xlsx or '', row_names, verbose=verbose,
        )
        if kp is not None:
            parts.append(kp)
            col_names.extend(['log_kcat', 'log_km'])

    if use_complex_constraints:
        from cell_sim.lgnn.extras.complex_constraints import (
            build_complex_membership_features,
        )
        cx = build_complex_membership_features(
            complex_formation_xlsx or '', row_names, verbose=verbose,
        )
        if cx is not None:
            parts.append(cx)
            col_names.extend([f'in_complex_{j}' for j in range(cx.shape[1])])

    if use_medium_features:
        from cell_sim.lgnn.extras.medium_features import build_medium_features
        mf = build_medium_features(
            medium_xlsx or '', row_names, verbose=verbose,
        )
        if mf is not None:
            parts.append(mf)
            col_names.extend([f'medium_{j}' for j in range(mf.shape[1])])

    # ---- M7.5 prep extras (default OFF) ----
    if use_regulatory_features:
        from cell_sim.lgnn.extras.regulatory_features import (
            build_regulatory_features, REGULATORY_FEATURE_COLS,
        )
        reg = build_regulatory_features(
            protein_metabolites_xlsx or '', row_names, verbose=verbose,
        )
        if reg is not None:
            parts.append(reg)
            col_names.extend(REGULATORY_FEATURE_COLS)

    if use_ribosome_subunit:
        from cell_sim.lgnn.extras.ribosome_subunit import (
            build_ribosome_subunit_features, RIBOSOME_SUBUNIT_COLS,
        )
        rs = build_ribosome_subunit_features(
            large_subunit_xlsx or '', row_names, verbose=verbose,
        )
        if rs is not None:
            parts.append(rs)
            col_names.extend(RIBOSOME_SUBUNIT_COLS)

    if use_thermodynamics:
        from cell_sim.lgnn.extras.thermodynamics import (
            build_thermodynamics_features, THERMO_FEATURE_COLS,
        )
        th = build_thermodynamics_features(
            gibbs_csv_path or '', row_names, verbose=verbose,
        )
        if th is not None:
            parts.append(th)
            col_names.extend(THERMO_FEATURE_COLS)

    if use_esm2_protein_features:
        from cell_sim.lgnn.extras.esm2_protein_features import (
            build_esm2_protein_features, ESM2_FEATURE_PREFIX,
        )
        em = build_esm2_protein_features(
            esm2_embeddings_parquet or '', row_names,
            proj_dim=esm2_proj_dim, verbose=verbose,
        )
        if em is not None:
            parts.append(em)
            col_names.extend([f'{ESM2_FEATURE_PREFIX}_{i}'
                              for i in range(em.shape[1])])

    if not parts:
        return None
    combined = torch.cat(parts, dim=1)
    if verbose:
        print(f'  combined static features: shape {tuple(combined.shape)} '
              f'({len(col_names)} named cols)')
    return combined


def _subsample_pair_iterator(
    R: int, n_valid: int, batch_size: int,
    generator: torch.Generator, device: torch.device,
    n_samples: Optional[int],
    t_skip_initial: int = 0,
):
    """Yield (rep_idx, t_idx) batches.

    If `n_samples` is None or >= R*(n_valid - t_skip_initial), falls
    through to the M3 full-enumeration permutation over the
    [t_skip_initial, n_valid) range. Otherwise samples without
    replacement when n_samples <= R*(n_valid - t_skip_initial).

    t_skip_initial excludes early timesteps that may contain simulator
    initialization artifacts (see narrate_window analysis at t=0).
    """
    n_eff = max(n_valid - t_skip_initial, 1)
    total = R * n_eff
    if n_samples is None or n_samples >= total:
        # Full enumeration over (rep, t_offset) where t_offset >= t_skip_initial
        perm = torch.randperm(total, generator=generator, device=device)
        rep_idx_all = perm // n_eff
        t_idx_all = (perm % n_eff) + t_skip_initial
        for s in range(0, total, batch_size):
            e = min(s + batch_size, total)
            yield rep_idx_all[s:e], t_idx_all[s:e]
        return
    perm = torch.randperm(total, generator=generator, device=device)[:n_samples]
    rep_idx_all = perm // n_eff
    t_idx_all = (perm % n_eff) + t_skip_initial
    for s in range(0, n_samples, batch_size):
        e = min(s + batch_size, n_samples)
        yield rep_idx_all[s:e], t_idx_all[s:e]


# ----------------------------------------------------------------------
# M7.8: random-anchor + multi-step training step
# ----------------------------------------------------------------------
def _random_anchor_train_step(
    model,
    train_data,
    sigma,
    rep_idx,
    t_idx,
    *,
    warmup_steps: int,
    multi_step_horizon: int,
    p_ss: float,
    flux_indices,
    count_mask,
    cum_mask,
    cfg,
    model_dtype,
):
    """One training step under random-anchor warmup + multi-step prediction.

    1. Gather window [t, t + warmup + H + 1) from train_data.
    2. With NO_GRAD, roll model forward for `warmup_steps` steps starting
       from x_w[:, 0]. Use scheduled sampling (mix model output and
       ground truth at probability p_ss) to expose the model to its own
       drifted predictions at every warmup step.
    3. The model's prediction at the anchor (= x_w[:, warmup_steps]) is
       used as the input to ONE GRAD forward pass with
       return_multi_step=True. The pinn_head produces x_{t+1} and v_log;
       the aux_heads produce x_{t+2..t+H}.
    4. Loss = primary 1-step loss + aux_weight * sum of H-1 multi-step
       MSE losses. One backward pass back-propagates through ONE forward.

    This decouples rollout horizon (warmup_steps, can be 0..7000) from
    gradient horizon (one forward pass), giving per-sample compute that
    is O(warmup_steps) for forward + O(1) for backward.
    """
    H = multi_step_horizon
    window_len = warmup_steps + H + 1
    x_w = _gather_windows(train_data, rep_idx, t_idx, window_len
                          ).to(model_dtype)

    # ---- Phase 1: NO_GRAD warmup with scheduled sampling ----
    # Disable activation checkpointing during warmup - it's overhead
    # for no_grad rollouts (no backward to recompute for).
    _saved_ckpt = getattr(model, 'use_checkpoint', False)
    if hasattr(model, 'use_checkpoint'):
        model.use_checkpoint = False
    x_pred = x_w[:, 0, :]
    if warmup_steps > 0:
        with torch.no_grad():
            for s in range(warmup_steps):
                v_cur = sigma[t_idx + s].to(model_dtype)
                if s > 0 and p_ss < 1.0:
                    # mix model output and ground truth p_ss% of the time
                    use_tf = (torch.rand(x_pred.shape[0], 1,
                                         device=x_pred.device) > p_ss
                              ).to(x_pred.dtype)
                    x_input = use_tf * x_w[:, s, :] + (1 - use_tf) * x_pred
                else:
                    x_input = x_pred
                x_next, _ = model(x_input, x_var=v_cur)
                x_pred = x_next

    # Re-enable checkpointing for the grad forward (we DO need it for
    # backward to be memory-safe)
    if hasattr(model, 'use_checkpoint'):
        model.use_checkpoint = _saved_ckpt

    # ---- Phase 2: ONE GRAD forward at the anchor, multi-step prediction ----
    v_anchor = sigma[t_idx + warmup_steps].to(model_dtype)
    if H > 1:
        x_next_log, v_log, aux_preds = model(
            x_pred, x_var=v_anchor, return_multi_step=True,
        )
    else:
        x_next_log, v_log = model(x_pred, x_var=v_anchor)
        aux_preds = None

    # Targets are the actual training data at anchor+1..anchor+H
    primary_target = x_w[:, warmup_steps + 1, :]
    L_primary, breakdown = _m7_step_loss(
        x_next_log, v_log, primary_target, flux_indices,
        count_mask, cum_mask,
        cfg.weight_count, cfg.weight_flux, cfg.weight_cumulative,
    )

    L_aux_total = torch.zeros((), device=x_pred.device, dtype=model_dtype)
    if aux_preds is not None:
        for h_idx in range(H - 1):
            aux_target = x_w[:, warmup_steps + 2 + h_idx, :]
            L_aux_total = L_aux_total + (
                aux_preds[:, h_idx] - aux_target
            ).pow(2).mean()
        L_aux_total = L_aux_total / max(H - 1, 1)

    return L_primary, L_aux_total, breakdown


# ----------------------------------------------------------------------
# M7.8: Chunked-anchor amortization
# ----------------------------------------------------------------------
def _chunked_anchor_train_step(
    model,
    train_data,
    sigma,
    rep_idx,
    t_idx,
    *,
    warmup_steps: int,
    n_anchor_chunks: int,
    multi_step_horizon: int,
    p_ss: float,
    flux_indices,
    count_mask,
    cum_mask,
    cfg,
    model_dtype,
    opt,
    aux_weight: float,
):
    """Amortize ONE long warmup over `n_anchor_chunks` training anchors.

    Standard random-anchor wastes the warmup rollout: 7000 steps of
    no-grad forward produce a SINGLE anchor's state for ONE training
    signal. Chunked amortization treats the warmup as a SEQUENCE of
    anchor states - we stop at K equally-spaced points along the
    warmup and do a small grad forward+backward at each. One 7000-step
    warmup produces K training signals instead of 1, an effective K×
    speedup. K=10 means 10 anchors per warmup; total samples = K *
    (n_warmup_batches).

    Each anchor uses the multi_step_horizon=H aux head trick: one grad
    forward, one backward, H gradient signals (1 primary + H-1 aux).

    Returns the average per-anchor loss across all K anchors. Each
    anchor's gradient is applied immediately (opt.step()), so the
    function consumes n_anchor_chunks worth of training samples per
    call.
    """
    H = multi_step_horizon
    # We need x_w long enough to: roll past the LAST anchor + supply
    # H targets after it. Last anchor is at t_idx + warmup_steps.
    window_len = warmup_steps + H + 1
    x_w = _gather_windows(train_data, rep_idx, t_idx, window_len
                          ).to(model_dtype)

    # Anchor step positions: evenly spaced unique anchors, capped at
    # warmup_steps so we never produce duplicates. If warmup is short
    # and K is large (e.g. warmup=6 with K=40), we'd get 40 anchors all
    # at position 6 with the original formula. The .max(1, K_eff) cap
    # forces 1..warmup_steps unique anchors instead.
    if n_anchor_chunks <= 1 or warmup_steps == 0:
        anchor_positions = [warmup_steps]
    else:
        K_eff = min(n_anchor_chunks, max(1, warmup_steps))
        if K_eff <= 1:
            anchor_positions = [warmup_steps]
        else:
            anchor_positions = sorted({
                min((i + 1) * max(1, warmup_steps // K_eff),
                    warmup_steps)
                for i in range(K_eff)
            })

    # Disable checkpointing during warmup; restore for grad step
    _saved_ckpt = getattr(model, 'use_checkpoint', False)

    x_pred = x_w[:, 0, :]
    cur_pos = 0
    total_primary = torch.zeros((), device=x_pred.device, dtype=torch.float32)
    total_aux = torch.zeros((), device=x_pred.device, dtype=torch.float32)
    cumulative_breakdown = None
    n_anchors_done = 0

    for anchor_pos in anchor_positions:
        # Phase A: advance NO_GRAD from cur_pos to anchor_pos
        if anchor_pos > cur_pos:
            if hasattr(model, 'use_checkpoint'):
                model.use_checkpoint = False
            with torch.no_grad():
                for s in range(cur_pos, anchor_pos):
                    v_cur = sigma[t_idx + s].to(model_dtype)
                    if s > 0 and p_ss < 1.0:
                        use_tf = (torch.rand(
                            x_pred.shape[0], 1, device=x_pred.device
                        ) > p_ss).to(x_pred.dtype)
                        x_input = (use_tf * x_w[:, s, :]
                                   + (1 - use_tf) * x_pred)
                    else:
                        x_input = x_pred
                    x_next, _ = model(x_input, x_var=v_cur)
                    x_pred = x_next
            cur_pos = anchor_pos
            if hasattr(model, 'use_checkpoint'):
                model.use_checkpoint = _saved_ckpt

        # Phase B: ONE grad forward at this anchor + multi-step
        v_anchor = sigma[t_idx + anchor_pos].to(model_dtype)
        if H > 1:
            x_next_log, v_log, aux_preds = model(
                x_pred.detach(), x_var=v_anchor, return_multi_step=True,
            )
        else:
            x_next_log, v_log = model(x_pred.detach(), x_var=v_anchor)
            aux_preds = None

        primary_target = x_w[:, anchor_pos + 1, :]
        L_primary, breakdown = _m7_step_loss(
            x_next_log, v_log, primary_target, flux_indices,
            count_mask, cum_mask,
            cfg.weight_count, cfg.weight_flux, cfg.weight_cumulative,
        )
        L_aux = torch.zeros((), device=x_pred.device, dtype=model_dtype)
        if aux_preds is not None:
            for h_idx in range(H - 1):
                tgt = x_w[:, anchor_pos + 2 + h_idx, :]
                L_aux = L_aux + (aux_preds[:, h_idx] - tgt).pow(2).mean()
            L_aux = L_aux / max(H - 1, 1)
        L = L_primary + aux_weight * L_aux

        opt.zero_grad(set_to_none=True)
        L.backward()
        opt.step()

        total_primary = total_primary + L_primary.detach().float()
        total_aux     = total_aux + L_aux.detach().float()
        if cumulative_breakdown is None:
            cumulative_breakdown = {k: v.detach().float()
                                    for k, v in breakdown.items()}
        else:
            for k in cumulative_breakdown:
                cumulative_breakdown[k] = cumulative_breakdown[k] \
                    + breakdown[k].detach().float()
        n_anchors_done += 1

        # Update x_pred for next chunk (using fresh prediction from same forward)
        # We need to step forward to give the next chunk a starting state.
        # Use the multi-step or single-step prediction.
        x_pred = x_next_log.detach()
        cur_pos = anchor_pos + 1

    # Average
    avg_primary = total_primary / max(n_anchors_done, 1)
    avg_aux = total_aux / max(n_anchors_done, 1)
    avg_breakdown = {k: v / max(n_anchors_done, 1)
                     for k, v in (cumulative_breakdown or {}).items()}
    return avg_primary, avg_aux, avg_breakdown, n_anchors_done


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------

def _m7_step_loss(
    x_next_pred: torch.Tensor,         # (B, S) predicted state, signed-log1p
    v_log_pred:  torch.Tensor,         # (B, R) predicted log-rates
    x_next_target: torch.Tensor,       # (B, S) observed state
    flux_indices: torch.Tensor,        # (R,) flux row indices
    count_mask:  torch.Tensor,         # (S,) bool
    cum_mask:    torch.Tensor,         # (S,) bool
    w_count: float, w_flux: float, w_cum: float,
    log_sigma: Optional[torch.Tensor] = None,   # (B, S) per-species log_sigma
    nll_weight: float = 0.0,
):
    """Per-step M7 multi-task loss.

    Flux supervision is DIRECT: v_log compared element-wise to the
    F_* rows of the observed next state. Count + cumulative supervision
    use the PINN-derived next state on the appropriate row masks.

    M8: if `log_sigma` is provided AND `nll_weight > 0`, an additional
    Gaussian NLL term is added (proportional to nll_weight). This is
    the noise-floor-breaking objective:
        L_nll = 0.5*((x_target - x_pred)/sigma)^2 + log(sigma)
    averaged over count_mask rows (where the stochastic noise floor
    sits) - flux rows already have low MSE and don't benefit from σ.
    """
    diff_sq_state = (x_next_pred - x_next_target).pow(2)        # (B, S)

    if int(count_mask.sum()) > 0:
        mse_count = diff_sq_state[:, count_mask].mean()
    else:
        mse_count = torch.zeros((), device=x_next_pred.device,
                                 dtype=x_next_pred.dtype)
    if int(cum_mask.sum()) > 0:
        mse_cum = diff_sq_state[:, cum_mask].mean()
    else:
        mse_cum = torch.zeros((), device=x_next_pred.device,
                                 dtype=x_next_pred.dtype)

    flux_target_log = x_next_target.index_select(1, flux_indices)   # (B, R)
    mse_flux = (v_log_pred - flux_target_log).pow(2).mean()

    total = w_count * mse_count + w_flux * mse_flux + w_cum * mse_cum

    # M8 stochastic NLL term — only on count rows where the noise floor sits
    nll_count = torch.zeros((), device=x_next_pred.device,
                             dtype=x_next_pred.dtype)
    if log_sigma is not None and nll_weight > 0.0:
        # σ clamped at 1e-3 to prevent division blow-ups during early
        # training when log_σ output might be uncalibrated.
        sigma = log_sigma.exp().clamp(min=1e-3)              # (B, S)
        per_elem = 0.5 * diff_sq_state / sigma.pow(2) + log_sigma
        if int(count_mask.sum()) > 0:
            nll_count = per_elem[:, count_mask].mean()
            total = total + nll_weight * nll_count

    return total, {
        'mse_count': mse_count.detach(),
        'mse_flux':  mse_flux .detach(),
        'mse_cum':   mse_cum  .detach(),
        'nll_count': nll_count.detach(),
    }


# ----------------------------------------------------------------------
# Preload helper (shared cache file format with M6 / M-PINN)
# ----------------------------------------------------------------------

def _maybe_load_or_preload(
    replicate_indices, lsdata_module, device, dtype,
    species_filter, cache_path,
):
    if cache_path is not None and Path(cache_path).exists():
        t0 = time.time()
        print(f'  loading cached preload from {cache_path}...')
        data = torch.load(cache_path, map_location=device, weights_only=False)
        if data.dim() == 3 and data.shape[0] == len(replicate_indices):
            print(f'  loaded {tuple(data.shape)} in {time.time()-t0:.1f}s')
            return data.to(dtype) if data.dtype != dtype else data
        print(f'  cache shape mismatch ({tuple(data.shape)}); rebuilding')
    t0 = time.time()
    data = preload_to_gpu(replicate_indices, lsdata_module, device,
                          dtype=dtype, species_filter=species_filter)
    print(f'  preloaded fresh in {time.time()-t0:.1f}s')
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        torch.save(data.cpu(), cache_path)
        print(f'  cached to {cache_path} in {time.time()-t1:.1f}s')
    return data


# ----------------------------------------------------------------------
# Train loop
# ----------------------------------------------------------------------

def train_m7(
    cfg: M7TrainConfig,
    lsdata_module,
    graph: SpeciesGraph,
    row_names: Sequence[str],
    checkpoint_path: Optional[Path] = None,
    train_cache_path: Optional[Path] = None,
    val_cache_path: Optional[Path] = None,
) -> dict:
    device = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    pre_dtype = _torch_dtype(cfg.preload_dtype)

    if not cfg.sbml_path_for_S:
        raise ValueError(
            'cfg.sbml_path_for_S must point at the SBML model '
            '(typically Syn3A_updated.xml). The .lm-based loader is '
            'unreliable for the PINN column mapping.'
        )

    # --- Stoichiometric matrix ---
    print(f'Loading stoichiometric matrix from {cfg.sbml_path_for_S}...')
    S_pinn, flux_indices, reaction_ids, mapped_mask = (
        load_stoichiometric_matrix_for_pinn(cfg.sbml_path_for_S, row_names)
    )
    print(f'  S_pinn={tuple(S_pinn.shape)}  flux_indices={flux_indices.shape[0]}')
    if flux_indices.numel() == 0:
        raise RuntimeError('S_pinn has zero columns; F_<rxn> naming does '
                           'not match SBML reaction ids. Aborting.')
    flux_indices = flux_indices.to(device)

    # --- Preload corpus ---
    print(f'preloading {len(cfg.train_replicates)} train + '
          f'{len(cfg.val_replicates)} val replicates to {device}...')
    train_data = _maybe_load_or_preload(
        cfg.train_replicates, lsdata_module, device, pre_dtype,
        cfg.species_filter, train_cache_path,
    )
    val_data = _maybe_load_or_preload(
        cfg.val_replicates, lsdata_module, device, pre_dtype,
        cfg.species_filter, val_cache_path,
    )
    R, T, S = train_data.shape
    n_valid = T - cfg.max_k - 1
    print(f'  train{tuple(train_data.shape)}  val{tuple(val_data.shape)}')

    # --- Row categorisation + variance channel ---
    count_mask, flux_mask, cum_mask = categorise_row_indices(row_names)
    count_mask = count_mask.to(device); flux_mask = flux_mask.to(device)
    cum_mask   = cum_mask  .to(device)
    print(f'  rows: count={int(count_mask.sum())}  '
          f'flux={int(flux_mask.sum())}  cumulative={int(cum_mask.sum())}')

    print('  precomputing per-species cross-replicate sigma...')
    sigma = compute_variance_channel(train_data)              # (T, S)

    # --- Static features (M7.1 role; M7.3 spatial+proteomics; M7.4 extras) ---
    print(f'  building static features  role={cfg.use_role_features}  '
          f'spatial={cfg.use_spatial_features}  '
          f'proteomics={cfg.use_proteomics_features}  '
          f'kinetic={cfg.use_kinetic_priors}  '
          f'complex={cfg.use_complex_constraints}  '
          f'medium={cfg.use_medium_features}')
    static_features = _build_combined_static_features(
        row_names=row_names,
        use_role=cfg.use_role_features,
        use_spatial=cfg.use_spatial_features,
        use_proteomics=cfg.use_proteomics_features,
        use_kinetic_priors=cfg.use_kinetic_priors,
        use_complex_constraints=cfg.use_complex_constraints,
        use_medium_features=cfg.use_medium_features,
        use_regulatory_features=cfg.use_regulatory_features,
        use_ribosome_subunit=cfg.use_ribosome_subunit,
        use_thermodynamics=cfg.use_thermodynamics,
        use_esm2_protein_features=cfg.use_esm2_protein_features,
        spatial_parquet=cfg.spatial_parquet,
        proteomics_xlsx=cfg.proteomics_xlsx,
        gene_class_csv=cfg.gene_class_csv,
        kinetic_params_xlsx=cfg.kinetic_params_xlsx,
        complex_formation_xlsx=cfg.complex_formation_xlsx,
        medium_xlsx=cfg.medium_xlsx,
        protein_metabolites_xlsx=cfg.protein_metabolites_xlsx,
        large_subunit_xlsx=cfg.large_subunit_xlsx,
        gibbs_csv_path=cfg.gibbs_csv_path,
        esm2_embeddings_parquet=cfg.esm2_embeddings_parquet,
        esm2_proj_dim=cfg.esm2_proj_dim,
        verbose=True,
    )

    # --- Model ---
    model = CellGNNv7Hybrid(
        graph=graph,
        stoich_matrix=S_pinn,
        flux_indices=flux_indices.cpu(),
        hidden=cfg.hidden, n_layers=cfg.n_layers,
        n_input_channels=cfg.n_input_channels,
        static_features=static_features,
        use_checkpoint=cfg.use_checkpoint,
        edge_chunk_size=cfg.edge_chunk_size,
        cfc_tau_min=cfg.cfc_tau_min,
        rate_clip=cfg.rate_clip,
        use_residual=cfg.use_residual,
        multi_step_horizon=cfg.multi_step_horizon,
        use_stochastic_head=cfg.use_stochastic_head,
        encoder_backbone=cfg.encoder_backbone,
        use_moe_residual=cfg.use_moe_residual,
        moe_n_experts=cfg.moe_n_experts,
        use_node_flux_head=cfg.use_node_flux_head,
        node_solver=cfg.node_solver,
        node_step_size=cfg.node_step_size,
    ).to(device)
    if cfg.use_bf16 and device.type == 'cuda':
        model = model.to(torch.bfloat16)
    model_dtype = next(model.parameters()).dtype

    # Warm-start from a pretrained checkpoint (e.g. M7.7 -> M7.8).
    # Loads with strict=False so that newly added params (aux_heads,
    # aux_delta_scale) keep their fresh random init while everything
    # the previous checkpoint already trained (encoder, pinn_head) gets
    # the head start. Skips silently if path is empty/missing.
    init_path = getattr(cfg, 'init_from_checkpoint', '')
    if init_path:
        try:
            ckpt = torch.load(init_path, map_location=device, weights_only=False)
            init_state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            missing, unexpected = model.load_state_dict(init_state, strict=False)
            n_loaded = sum(1 for k in init_state if k not in unexpected)
            print(f'  warm-start: loaded {n_loaded} tensors from {init_path}')
            if missing:
                print(f'    skipped (newly added in M7.8): {missing[:6]}'
                      f'{"..." if len(missing) > 6 else ""}')
        except Exception as e:
            print(f'  WARNING: failed to warm-start from {init_path}: {e}')

    n_params = count_parameters(model)
    print(f'M7: {n_params:,} params  (hidden={cfg.hidden}, '
          f'n_layers={cfg.n_layers}, channels={cfg.n_input_channels}, '
          f'use_residual={cfg.use_residual}, dtype={model_dtype})')
    print(f'  M8 stack  stochastic={cfg.use_stochastic_head}  '
          f'optimizer={cfg.optimizer_name}  ema={cfg.use_ema}  swa={cfg.use_swa}  '
          f'esm2={cfg.use_esm2_protein_features}')
    print(f'  arch swaps  encoder={cfg.encoder_backbone}  '
          f'moe_residual={cfg.use_moe_residual}  '
          f'node_flux={cfg.use_node_flux_head}')
    print(f'  k_curriculum={cfg.k_curriculum}  rollout_gamma={cfg.rollout_gamma}  '
          f'scheduled_sampling={cfg.scheduled_sampling}  '
          f'tbptt={cfg.truncated_bptt_window}')
    print(f'  loss weights  count={cfg.weight_count}  flux={cfg.weight_flux}  '
          f'cum={cfg.weight_cumulative}  lambda_mb={cfg.lambda_mass_balance}')
    print(f'  speed levers  bs={cfg.batch_size}  '
          f'samples/epoch={cfg.samples_per_epoch}  '
          f'compile={cfg.use_compile}  checkpoint={cfg.use_checkpoint}')

    if cfg.use_compile and device.type == 'cuda':
        try:
            model = torch.compile(model, mode=cfg.compile_mode)
            print(f'  torch.compile enabled (mode={cfg.compile_mode})')
        except Exception as e:
            print(f'  WARNING: torch.compile failed ({e}); continuing uncompiled')

    from cell_sim.lgnn.training.optimizers import build_optimizer
    opt = build_optimizer(
        model.parameters(),
        name=cfg.optimizer_name,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    print(f'  optimizer: {cfg.optimizer_name}')

    # SBML mask for the optional anti-drift term, cached on device.
    # PINNHead's s_covered_mask is float; coerce to bool here.
    sbml_mask = (model.pinn_head.s_covered_mask > 0).to(device)
    n_sbml = int(sbml_mask.sum())

    history = {
        'train_total': [], 'train_mse_count': [], 'train_mse_flux': [],
        'train_mse_cum': [], 'train_rollout': [], 'train_attn_entropy': [],
        'train_mass_balance': [], 'train_p_ss': [],
        'val_singlestep_mse': [], 'val_mse_count': [], 'val_mse_flux': [],
        'val_mse_cum': [], 'val_rollout_mse_avg': [],
        'val_mse_per_step': [], 'k_per_epoch': [], 'samples_per_sec': [],
    }
    best_val = float('inf')
    # ---- M8 upgrade 3/5: EMA + SWA setup ----
    # EMA: a running exponential moving average of the model weights,
    # updated every optimiser step. At eval, swap the EMA weights in
    # (model.parameters() <- ema_model.parameters()), evaluate, swap
    # back. Typically +2-5% on val with zero training-time cost.
    #
    # SWA: average of model weights over the FINAL swa_n_epochs of
    # training. Final eval uses these averaged weights. Standard
    # paper trick that adds ~1-3% on val.
    ema_model = None
    swa_model = None
    if cfg.use_ema or cfg.use_swa:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        if cfg.use_ema:
            ema_model = AveragedModel(
                model,
                multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay),
            )
            print(f'  EMA enabled (decay={cfg.ema_decay})')
        if cfg.use_swa:
            swa_model = AveragedModel(model)
            print(f'  SWA enabled (start_epoch={cfg.swa_start_epoch})')

    train_t0 = time.time()
    global_step = 0

    for epoch in range(cfg.n_epochs):
        k_cur = _k_for_epoch(cfg.k_curriculum, epoch)
        history['k_per_epoch'].append(k_cur)
        eff_lambda_rollout = (0.0 if (k_cur == 1 and cfg.skip_rollout_at_k1)
                              else cfg.lambda_rollout)

        # Per-stage overrides: pick the stage-specific samples_per_epoch
        # and tbptt window if the tuples are set, else fall back to the
        # scalar fields.
        if cfg.samples_per_epoch_per_stage is not None:
            stage_samples = int(cfg.samples_per_epoch_per_stage[
                min(epoch, len(cfg.samples_per_epoch_per_stage) - 1)
            ])
        else:
            stage_samples = cfg.samples_per_epoch
        if cfg.tbptt_window_per_stage is not None:
            stage_tbptt = cfg.tbptt_window_per_stage[
                min(epoch, len(cfg.tbptt_window_per_stage) - 1)
            ]
        else:
            stage_tbptt = cfg.truncated_bptt_window

        # Truncated-BPTT cutoff index: steps with index >= grad_start
        # are computed with grad; earlier steps roll under no_grad.
        if stage_tbptt is not None and k_cur > stage_tbptt:
            grad_start = k_cur - stage_tbptt
        else:
            grad_start = 0      # all steps with grad

        gen = torch.Generator(device=device).manual_seed(cfg.seed + epoch)
        model.train()
        run = {k: torch.zeros((), device=device, dtype=torch.float32)
               for k in ('total', 'mse_count', 'mse_flux', 'mse_cum',
                         'rollout', 'attn_entropy', 'mass_balance')}
        n_batches = 0
        t_epoch = time.time()
        n_seen = 0
        p_ss = 0.0

        pair_iter = _subsample_pair_iterator(
            R, n_valid, cfg.batch_size, gen, device, stage_samples,
            t_skip_initial=cfg.t_skip_initial,
        )
        # M7.8 random-anchor stage: figure out warmup range AND adjust
        # iter count when chunked anchors is on (else we'd run 195
        # iterations × K anchors-each, giving stage_samples * K total
        # samples instead of stage_samples).
        if cfg.train_strategy == 'random_anchor':
            if cfg.random_anchor_warmup_max_per_stage:
                stage_max_warmup = int(cfg.random_anchor_warmup_max_per_stage[
                    min(epoch, len(cfg.random_anchor_warmup_max_per_stage) - 1)
                ])
            else:
                stage_max_warmup = 100
            effective_n_valid = max(
                1, T - cfg.multi_step_horizon - stage_max_warmup - 1
            )
            # Compute expected anchors per iter so we hit stage_samples
            # in (stage_samples / (batch*avgK)) iters instead of all K=100
            # times more.
            if cfg.n_anchor_chunks_per_warmup > 1:
                avg_K = min(cfg.n_anchor_chunks_per_warmup,
                            max(1, stage_max_warmup // 2))
                target_samples_per_iter = cfg.batch_size * max(1, avg_K)
                iter_samples = max(
                    cfg.batch_size,
                    int(stage_samples / max(avg_K, 1)),
                )
            else:
                avg_K = 1
                iter_samples = stage_samples
            pair_iter = _subsample_pair_iterator(
                R, effective_n_valid, cfg.batch_size, gen, device,
                iter_samples, t_skip_initial=cfg.t_skip_initial,
            )
            n_iters_est = max(1, iter_samples // cfg.batch_size)
            print(f'  ep{epoch} M7.8 random_anchor: '
                  f'samples={stage_samples}  max_warmup={stage_max_warmup}  '
                  f'avg_K={avg_K}  iters={n_iters_est}  '
                  f'multi_step_H={cfg.multi_step_horizon}  '
                  f'aux_w={cfg.aux_loss_weight}')
        elif (cfg.samples_per_epoch_per_stage is not None
                or cfg.tbptt_window_per_stage is not None):
            print(f'  ep{epoch} stage: k={k_cur}  samples={stage_samples}  '
                  f'tbptt={stage_tbptt}  grad_start_step={grad_start}')
        for rep_idx, t_idx in pair_iter:
            # ---- M7.8 random-anchor + multi-step branch ----
            if cfg.train_strategy == 'random_anchor':
                if cfg.scheduled_sampling:
                    p_ss = min(global_step / max(cfg.p_ss_warmup_steps, 1),
                               1.0) * cfg.p_ss_max
                else:
                    p_ss = 0.0
                warmup_steps = int(torch.randint(
                    0, stage_max_warmup + 1, (1,),
                    generator=gen, device=device,
                ).item())
                if cfg.n_anchor_chunks_per_warmup > 1:
                    # Chunked amortization: K anchors per warmup
                    L_primary, L_aux, breakdown, n_anchors = (
                        _chunked_anchor_train_step(
                            model, train_data, sigma, rep_idx, t_idx,
                            warmup_steps=warmup_steps,
                            n_anchor_chunks=cfg.n_anchor_chunks_per_warmup,
                            multi_step_horizon=cfg.multi_step_horizon,
                            p_ss=p_ss,
                            flux_indices=flux_indices,
                            count_mask=count_mask, cum_mask=cum_mask,
                            cfg=cfg, model_dtype=model_dtype,
                            opt=opt, aux_weight=cfg.aux_loss_weight,
                        )
                    )
                    L = L_primary + cfg.aux_loss_weight * L_aux
                    global_step += n_anchors
                    n_seen += int(rep_idx.shape[0]) * n_anchors
                else:
                    L_primary, L_aux, breakdown = _random_anchor_train_step(
                        model, train_data, sigma, rep_idx, t_idx,
                        warmup_steps=warmup_steps,
                        multi_step_horizon=cfg.multi_step_horizon,
                        p_ss=p_ss,
                        flux_indices=flux_indices,
                        count_mask=count_mask, cum_mask=cum_mask,
                        cfg=cfg, model_dtype=model_dtype,
                    )
                    L = L_primary + cfg.aux_loss_weight * L_aux
                    opt.zero_grad(set_to_none=True)
                    L.backward()
                    opt.step()
                    if ema_model is not None:
                        ema_model.update_parameters(model)
                    global_step += 1
                    n_seen += int(rep_idx.shape[0])
                run['total']     += L.detach().float()
                run['mse_count'] += breakdown['mse_count'].float()
                run['mse_flux']  += breakdown['mse_flux'].float()
                run['mse_cum']   += breakdown['mse_cum'].float()
                run['rollout']   += L_aux.detach().float()
                n_batches += 1
                # In chunked mode, total iters may be small (~2-8) so log
                # every iter; otherwise honour cfg.log_every.
                effective_log_every = (1 if cfg.n_anchor_chunks_per_warmup > 1
                                       else cfg.log_every)
                if n_batches % effective_log_every == 0:
                    wall = time.time() - t_epoch
                    vals = {k: float(v.item()) / n_batches
                            for k, v in run.items()}
                    print(f'  ep{epoch} ra step{n_batches:>5d}'
                          f'  total={vals["total"]:.4f}'
                          f'  count={vals["mse_count"]:.4f}'
                          f'  flux={vals["mse_flux"]:.4f}'
                          f'  aux={vals["rollout"]:.4f}'
                          f'  w={warmup_steps:>4d}  p_ss={p_ss:.2f}'
                          f'  {n_seen/wall:.0f} s/s')
                if (time.time() - train_t0) > cfg.wall_clock_budget_s:
                    print(f'  WARNING: wall-clock budget exceeded')
                    break
                continue
            # ---- Standard (M7.1-7.7) branch unchanged below ----
            x_w = _gather_windows(train_data, rep_idx, t_idx, k_cur + 1
                                  ).to(model_dtype)
            x = x_w[:, 0, :]                            # (B, S)

            # Current scheduled-sampling probability
            if cfg.scheduled_sampling and k_cur > 1:
                p_ss = min(global_step / max(cfg.p_ss_warmup_steps, 1),
                           1.0) * cfg.p_ss_max
            else:
                p_ss = 0.0

            x_pred = x
            attn_entropy = None
            L_rollout = torch.zeros((), device=device, dtype=model_dtype)
            L_full_ss = None
            mse_breakdown_step0 = None
            L_mass_balance = torch.zeros((), device=device, dtype=model_dtype)
            gamma_sum = 0.0
            for s in range(k_cur):
                v_cur = sigma[t_idx + s].to(model_dtype)

                # Scheduled sampling at s > 0: with prob (1 - p_ss) inject
                # the ground-truth state instead of x_pred.
                if s > 0 and p_ss < 1.0:
                    use_tf = (torch.rand(x_pred.shape[0], 1, device=device)
                              > p_ss).to(x_pred.dtype)
                    x_input = use_tf * x_w[:, s, :] + (1 - use_tf) * x_pred
                else:
                    x_input = x_pred

                # Truncated BPTT: forward under no_grad until grad_start,
                # then enable grad for the tail tbptt_window steps. Step 0
                # is always in the grad block (we need L_full_ss).
                step_has_grad = (s == 0) or (s >= grad_start)
                ctx = torch.enable_grad() if step_has_grad else torch.no_grad()
                want_log_sigma = cfg.use_stochastic_head and step_has_grad
                with ctx:
                    if s == 0:
                        out = model(
                            x_input, x_var=v_cur,
                            return_attention_entropy=True,
                            return_log_sigma=want_log_sigma,
                        )
                        if want_log_sigma:
                            x_next, v_log, ent, log_sigma = out
                        else:
                            x_next, v_log, ent = out
                            log_sigma = None
                        attn_entropy = ent
                    else:
                        out = model(x_input, x_var=v_cur,
                                     return_log_sigma=want_log_sigma)
                        if want_log_sigma:
                            x_next, v_log, log_sigma = out
                        else:
                            x_next, v_log = out
                            log_sigma = None

                if step_has_grad:
                    target = x_w[:, s + 1, :]
                    # M8: ramp NLL weight after mse_warmup_steps
                    if cfg.use_stochastic_head:
                        eff_nll = (cfg.nll_loss_weight
                                   if global_step >= cfg.mse_warmup_steps else 0.0)
                    else:
                        eff_nll = 0.0
                    step_loss, breakdown = _m7_step_loss(
                        x_next, v_log, target, flux_indices,
                        count_mask, cum_mask,
                        cfg.weight_count, cfg.weight_flux, cfg.weight_cumulative,
                        log_sigma=log_sigma, nll_weight=eff_nll,
                    )
                    if s == 0:
                        L_full_ss = step_loss
                        mse_breakdown_step0 = breakdown
                    w = cfg.rollout_gamma ** s
                    L_rollout = L_rollout + w * step_loss
                    gamma_sum += w

                    # Anti-drift mass-balance term (only at the final step,
                    # measures cumulative drift x_t - x_0 over SBML rows).
                    if cfg.lambda_mass_balance > 0.0 and n_sbml > 0 and s == k_cur - 1:
                        drift = (x_next - x_w[:, 0, :])             # (B, S)
                        L_mass_balance = drift[:, sbml_mask].pow(2).mean()

                x_pred = x_next

            L_rollout = L_rollout / max(gamma_sum, 1e-12)

            cur_lambda_attn = _warmup_ramp(
                global_step, cfg.lambda_attn,
                cfg.lambda_attn_warmup_steps, cfg.lambda_attn_ramp_steps,
            )

            if eff_lambda_rollout == 0.0:
                supervised = L_full_ss
            else:
                supervised = eff_lambda_rollout * L_rollout
            L = supervised + cur_lambda_attn * attn_entropy
            if cfg.lambda_mass_balance > 0.0:
                L = L + cfg.lambda_mass_balance * L_mass_balance

            opt.zero_grad(set_to_none=True)
            L.backward()
            opt.step()
            if ema_model is not None:
                ema_model.update_parameters(model)
            global_step += 1

            run['total']        += L.detach().float()
            run['mse_count']    += mse_breakdown_step0['mse_count'].float()
            run['mse_flux']     += mse_breakdown_step0['mse_flux'].float()
            run['mse_cum']      += mse_breakdown_step0['mse_cum'].float()
            run['rollout']      += L_rollout.detach().float()
            run['attn_entropy'] += attn_entropy.detach().float()
            run['mass_balance'] += L_mass_balance.detach().float()
            n_batches += 1
            n_seen += int(rep_idx.shape[0])

            if n_batches % cfg.log_every == 0:
                wall = time.time() - t_epoch
                vals = {k: float(v.item()) / n_batches for k, v in run.items()}
                print(f'  ep{epoch} k={k_cur} step{n_batches:>5d}'
                      f'  total={vals["total"]:.4f}'
                      f'  count={vals["mse_count"]:.4f}'
                      f'  flux={vals["mse_flux"]:.4f}'
                      f'  cum={vals["mse_cum"]:.4f}'
                      f'  rollout={vals["rollout"]:.4f}'
                      f'  mb={vals["mass_balance"]:.4f}'
                      f'  p_ss={p_ss:.2f}'
                      f'  {n_seen/wall:.0f} s/s')

            if (time.time() - train_t0) > cfg.wall_clock_budget_s:
                print(f'  WARNING: wall-clock budget '
                      f'{cfg.wall_clock_budget_s/3600:.2f}h exceeded')
                break

        epoch_wall = time.time() - t_epoch
        history['samples_per_sec'].append(n_seen / max(epoch_wall, 1e-6))
        history['train_p_ss'].append(p_ss)
        for key in ('total', 'mse_count', 'mse_flux', 'mse_cum',
                    'rollout', 'attn_entropy', 'mass_balance'):
            history[f'train_{key}'].append(
                float(run[key].item()) / max(n_batches, 1)
            )

        # SWA update at end of epoch (after swa_start_epoch)
        if swa_model is not None and epoch >= cfg.swa_start_epoch:
            swa_model.update_parameters(model)

        # --- Validation (cheap; uses cfg.max_k k-step rollout on val) ---
        # Use EMA weights if enabled — they typically generalise better.
        eval_model = ema_model.module if ema_model is not None else model
        eval_model.eval()
        val_metrics = _evaluate_m7(
            eval_model, val_data, sigma, flux_indices, count_mask, cum_mask,
            cfg, model_dtype, device,
        )
        for k, v in val_metrics.items():
            key = f'val_{k}'
            history.setdefault(key, []).append(v)
        val_ss = val_metrics['singlestep_mse']
        print(f'ep{epoch} k={k_cur} val singlestep={val_ss:.4f}  '
              f'count={val_metrics["mse_count"]:.4f}  '
              f'flux={val_metrics["mse_flux"]:.4f}  '
              f'cum={val_metrics["mse_cum"]:.4f}  '
              f'rollout_avg={val_metrics["rollout_mse_avg"]:.4f}  '
              f'wall={epoch_wall:.1f}s  '
              f'{history["samples_per_sec"][-1]:.0f} s/s')

        if val_ss < best_val:
            best_val = val_ss
            if checkpoint_path is not None:
                payload = {
                    'state_dict': (model._orig_mod.state_dict()
                                   if hasattr(model, '_orig_mod')
                                   else model.state_dict()),
                    'cfg': cfg.__dict__,
                    'epoch': epoch,
                    'val_singlestep_mse': val_ss,
                    'val_rollout_mse_avg': val_metrics['rollout_mse_avg'],
                    'val_mse_per_step': val_metrics['mse_per_step'],
                    'k_at_save': k_cur,
                    'p_ss_at_save': p_ss,
                    'reaction_ids': reaction_ids,
                    'flux_indices': flux_indices.cpu(),
                    'architecture': 'M7_hybrid',
                }
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, checkpoint_path)
                print(f'  -> saved best (val_ss={val_ss:.4f}) to {checkpoint_path}')

    return {
        'history': history,
        'best_val_singlestep_mse': best_val,
        'wall_clock_total_s': time.time() - train_t0,
        'final_model': model,
        'reaction_ids': reaction_ids,
        'flux_indices': flux_indices.cpu(),
    }


# ----------------------------------------------------------------------
# Validation (cheap k-step rollout; full 7200-step eval lives in lgnn/eval)
# ----------------------------------------------------------------------

@torch.no_grad()
def _evaluate_m7(
    model, val_data, sigma, flux_indices,
    count_mask, cum_mask, cfg, model_dtype, device,
):
    R_v, T_v, S = val_data.shape
    n_valid = T_v - cfg.max_k - 1
    bs = cfg.batch_size
    mse_per_step = [0.0] * cfg.max_k
    cnt = 0
    total_full_count = 0.0
    total_full_flux  = 0.0
    total_full_cum   = 0.0
    n_full = 0

    # Skip early timesteps in validation too (cfg.t_skip_initial), so
    # val numbers reflect the model's behaviour on real cell dynamics
    # rather than being dragged down by simulator-init artifacts at t=0.
    t_start_lo = getattr(cfg, 't_skip_initial', 0)
    for t_start in range(t_start_lo, n_valid, bs):
        t_idx = torch.arange(t_start, min(t_start + bs, n_valid), device=device)
        rep_idx = torch.zeros_like(t_idx)
        x_w = _gather_windows(val_data, rep_idx, t_idx, cfg.max_k + 1
                              ).to(model_dtype)
        x_pred = x_w[:, 0, :]
        for s in range(cfg.max_k):
            v_s = sigma[t_idx + s].to(model_dtype)
            x_next, v_log = model(x_pred, x_var=v_s)
            tgt = x_w[:, s + 1, :]
            mse_per_step[s] += float((x_next - tgt).pow(2).mean().item())
            if s == 0:
                diff_sq = (x_next - tgt).pow(2)
                if int(count_mask.sum()) > 0:
                    total_full_count += float(diff_sq[:, count_mask].mean().item())
                if int(cum_mask.sum()) > 0:
                    total_full_cum   += float(diff_sq[:, cum_mask].mean().item())
                total_full_flux += float(
                    (v_log - tgt.index_select(1, flux_indices)).pow(2).mean().item()
                )
                n_full += 1
            x_pred = x_next
        cnt += 1

    mse_per_step = [m / max(cnt, 1) for m in mse_per_step]
    rollout_avg = sum(mse_per_step) / max(len(mse_per_step), 1)
    return {
        'singlestep_mse':    mse_per_step[0],
        'mse_count':         total_full_count / max(n_full, 1),
        'mse_flux':          total_full_flux  / max(n_full, 1),
        'mse_cum':           total_full_cum   / max(n_full, 1),
        'rollout_mse_avg':   rollout_avg,
        'mse_per_step':      mse_per_step,
    }
