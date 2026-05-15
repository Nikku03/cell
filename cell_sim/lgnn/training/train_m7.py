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
    spatial_parquet: Optional[str] = None,
    proteomics_xlsx: Optional[str] = None,
    gene_class_csv: Optional[str] = None,
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

    if not parts:
        return None
    combined = torch.cat(parts, dim=1)
    if verbose:
        print(f'  combined static features: shape {tuple(combined.shape)}')
    return combined


def _subsample_pair_iterator(
    R: int, n_valid: int, batch_size: int,
    generator: torch.Generator, device: torch.device,
    n_samples: Optional[int],
):
    """Yield (rep_idx, t_idx) batches.

    If `n_samples` is None or >= R*n_valid, falls through to the M3
    full-enumeration permutation. Otherwise samples without replacement
    when n_samples <= R*n_valid (one random subset per epoch).
    """
    total = R * n_valid
    if n_samples is None or n_samples >= total:
        yield from _index_iterator(R, n_valid, batch_size, generator, device)
        return
    perm = torch.randperm(total, generator=generator, device=device)[:n_samples]
    rep_idx_all = perm // n_valid
    t_idx_all   = perm %  n_valid
    for s in range(0, n_samples, batch_size):
        e = min(s + batch_size, n_samples)
        yield rep_idx_all[s:e], t_idx_all[s:e]


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
):
    """Per-step M7 multi-task loss.

    Flux supervision is DIRECT: v_log compared element-wise to the
    F_* rows of the observed next state. Count + cumulative supervision
    use the PINN-derived next state on the appropriate row masks.
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
    return total, {
        'mse_count': mse_count.detach(),
        'mse_flux':  mse_flux .detach(),
        'mse_cum':   mse_cum  .detach(),
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

    # --- Static features (M7.1: role; M7.3: + spatial + proteomics) ---
    print(f'  building static features  role={cfg.use_role_features}  '
          f'spatial={cfg.use_spatial_features}  '
          f'proteomics={cfg.use_proteomics_features}')
    static_features = _build_combined_static_features(
        row_names=row_names,
        use_role=cfg.use_role_features,
        use_spatial=cfg.use_spatial_features,
        use_proteomics=cfg.use_proteomics_features,
        spatial_parquet=cfg.spatial_parquet,
        proteomics_xlsx=cfg.proteomics_xlsx,
        gene_class_csv=cfg.gene_class_csv,
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
    ).to(device)
    if cfg.use_bf16 and device.type == 'cuda':
        model = model.to(torch.bfloat16)
    model_dtype = next(model.parameters()).dtype
    n_params = count_parameters(model)
    print(f'M7: {n_params:,} params  (hidden={cfg.hidden}, '
          f'n_layers={cfg.n_layers}, channels={cfg.n_input_channels}, '
          f'use_residual={cfg.use_residual}, dtype={model_dtype})')
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

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

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
        )
        if (cfg.samples_per_epoch_per_stage is not None
                or cfg.tbptt_window_per_stage is not None):
            print(f'  ep{epoch} stage: k={k_cur}  samples={stage_samples}  '
                  f'tbptt={stage_tbptt}  grad_start_step={grad_start}')
        for rep_idx, t_idx in pair_iter:
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
                with ctx:
                    if s == 0:
                        x_next, v_log, ent = model(
                            x_input, x_var=v_cur,
                            return_attention_entropy=True,
                        )
                        attn_entropy = ent
                    else:
                        x_next, v_log = model(x_input, x_var=v_cur)

                if step_has_grad:
                    target = x_w[:, s + 1, :]
                    step_loss, breakdown = _m7_step_loss(
                        x_next, v_log, target, flux_indices,
                        count_mask, cum_mask,
                        cfg.weight_count, cfg.weight_flux, cfg.weight_cumulative,
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

        # --- Validation (cheap; uses cfg.max_k k-step rollout on val) ---
        model.eval()
        val_metrics = _evaluate_m7(
            model, val_data, sigma, flux_indices, count_mask, cum_mask,
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

    for t_start in range(0, n_valid, bs):
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
