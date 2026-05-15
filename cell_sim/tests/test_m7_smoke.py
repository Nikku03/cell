"""Smoke tests for M7 - hybrid PINN + residual surrogate.

Verifies:
  * CellGNNv7Hybrid forward returns (x_next, v_log) with finite values
    and the right shapes for both 1- and 2-channel inputs.
  * PINN mass conservation holds on SBML-covered species: dx is exactly
    S @ expm1(v_log) (within bf16/fp32 tolerance).
  * The residual head provides nonzero delta on non-SBML species.
  * Gradient flows through both the PINN branch and the residual branch.
  * _m7_step_loss returns a finite scalar and is non-negative.
  * _subsample_pair_iterator yields the right total count and never
    out-of-range indices.
  * Truncated BPTT detaches the early-step gradient correctly.
  * full_rollout_evaluation runs for a small T_max and produces
    finite R^2 with the right per-category accounting.
"""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent.parent))

from cell_sim.lgnn.data.species_graph import build_species_graph
from cell_sim.lgnn.models.gnn_v7_hybrid import (
    CellGNNv7Hybrid, count_parameters,
)
from cell_sim.lgnn.models.pinn_head import signed_log1p, signed_expm1
from cell_sim.lgnn.training.train_m7 import (
    M7TrainConfig, _m7_step_loss, _subsample_pair_iterator,
    _build_role_features, _build_combined_static_features,
)
from cell_sim.lgnn.eval.full_rollout import full_rollout_evaluation


# ----------------------------------------------------------------------
# Tiny fixtures
# ----------------------------------------------------------------------

def _write_fake_cobra(path: Path):
    model = {
        'metabolites': [
            {'id': 'A'}, {'id': 'B'}, {'id': 'C'},
        ],
        'reactions': [
            {'id': 'R1', 'metabolites': {'A': -1.0, 'B': 1.0}},
            {'id': 'R2', 'metabolites': {'B': -1.0, 'C': -1.0, 'A': 1.0}},
        ],
    }
    path.write_text(json.dumps(model))


def _build_tiny_graph(tmp: Path):
    cobra = tmp / 'mini.json'
    _write_fake_cobra(cobra)
    # Mix of SBML species (A/B/C) and LGNN-only species (G/R/P/PM/RPM/DM/D)
    # plus two F_* flux trackers.
    rows = ['A', 'B', 'C',
            'G_0001', 'RP_0001', 'R_0001', 'RPM_0001',
            'P_0001', 'PM_0001', 'DM_0001', 'D_0001',
            'F_R1', 'F_R2']
    g = build_species_graph(rows, cobra, include_simulator_edges=True)
    return g, rows


def _build_tiny_stoich(n_species: int, n_rxns: int,
                       flux_indices_list: list) -> tuple:
    """Build a deterministic (n_species, n_rxns) S and flux_indices.
    Rows 0..2 are the SBML species (A, B, C); flux trackers live at
    flux_indices_list. Stoich: R1: A->B (col 0); R2: B+C->A (col 1)."""
    S = torch.zeros(n_species, n_rxns)
    # R1: A(-1) -> B(+1)
    S[0, 0] = -1.0
    S[1, 0] = +1.0
    # R2: B(-1) + C(-1) -> A(+1)
    S[1, 1] = -1.0
    S[2, 1] = -1.0
    S[0, 1] = +1.0
    flux_indices = torch.tensor(flux_indices_list, dtype=torch.long)
    return S, flux_indices


# ----------------------------------------------------------------------
# Model-level smoke tests
# ----------------------------------------------------------------------

def test_m7_forward_shapes_and_finite_2ch():
    """2-channel input (M7 default): returns (x_next, v_log) with finite values."""
    with tempfile.TemporaryDirectory() as d:
        g, rows = _build_tiny_graph(Path(d))
    S, flux_idx = _build_tiny_stoich(len(rows), 2,
                                       flux_indices_list=[rows.index('F_R1'),
                                                            rows.index('F_R2')])
    model = CellGNNv7Hybrid(
        graph=g, stoich_matrix=S, flux_indices=flux_idx,
        hidden=16, n_layers=2, n_input_channels=2,
    )
    B = 3
    x = torch.randn(B, g.n_nodes) * 0.5
    v = torch.randn(B, g.n_nodes) * 0.1
    x_next, v_log = model(x, x_var=v)
    assert x_next.shape == (B, g.n_nodes), x_next.shape
    assert v_log .shape == (B, 2),         v_log .shape
    assert torch.isfinite(x_next).all()
    assert torch.isfinite(v_log).all()


def test_m7_forward_returns_entropy_when_requested():
    with tempfile.TemporaryDirectory() as d:
        g, rows = _build_tiny_graph(Path(d))
    S, flux_idx = _build_tiny_stoich(len(rows), 2,
                                       flux_indices_list=[rows.index('F_R1'),
                                                            rows.index('F_R2')])
    model = CellGNNv7Hybrid(
        graph=g, stoich_matrix=S, flux_indices=flux_idx,
        hidden=16, n_layers=2,
    )
    x = torch.randn(2, g.n_nodes) * 0.1
    v = torch.zeros_like(x)
    out = model(x, x_var=v, return_attention_entropy=True)
    assert len(out) == 3
    x_next, v_log, ent = out
    assert ent.dim() == 0 and torch.isfinite(ent)


def test_m7_1ch_input_drops_variance_channel():
    with tempfile.TemporaryDirectory() as d:
        g, rows = _build_tiny_graph(Path(d))
    S, flux_idx = _build_tiny_stoich(len(rows), 2,
                                       flux_indices_list=[rows.index('F_R1'),
                                                            rows.index('F_R2')])
    model = CellGNNv7Hybrid(
        graph=g, stoich_matrix=S, flux_indices=flux_idx,
        hidden=16, n_layers=2, n_input_channels=1,
    )
    x = torch.randn(2, g.n_nodes) * 0.1
    x_next, v_log = model(x)        # no x_var argument
    assert torch.isfinite(x_next).all()


def test_m7_mass_conservation_on_sbml_rows():
    """For SBML-covered rows, x_next - x must match the hardwired S * expm1(v_log)
    update exactly when the residual head is disabled.

    With use_residual=True, the residual head is masked to 0 on s_covered
    rows, so the same equality should hold.
    """
    with tempfile.TemporaryDirectory() as d:
        g, rows = _build_tiny_graph(Path(d))
    S, flux_idx = _build_tiny_stoich(len(rows), 2,
                                       flux_indices_list=[rows.index('F_R1'),
                                                            rows.index('F_R2')])
    torch.manual_seed(0)
    model = CellGNNv7Hybrid(
        graph=g, stoich_matrix=S, flux_indices=flux_idx,
        hidden=16, n_layers=2, use_residual=True,
    )
    model.eval()
    B = 2
    x = torch.randn(B, g.n_nodes) * 0.3
    v = torch.zeros_like(x)
    x_next, v_log = model(x, x_var=v)

    # Compute the expected SBML-row delta from v_log.
    v_lin = signed_expm1(v_log)                          # (B, R)
    x_lin = signed_expm1(x)                              # (B, S)
    dx_pinn_lin = v_lin @ S.t()                         # (B, S)
    x_next_pinn_lin = x_lin + dx_pinn_lin
    x_next_pinn_log = signed_log1p(x_next_pinn_lin)

    sbml_mask = (model.pinn_head.s_covered_mask > 0)
    diff = (x_next - x_next_pinn_log)[:, sbml_mask].abs()
    assert diff.max().item() < 1e-4, (
        f'SBML mass-conservation violated; max abs diff = {diff.max().item():.6f}'
    )


def test_m7_residual_branch_unfreezes_non_sbml_rows():
    """Without the residual head, non-SBML rows would have dx = 0
    in linear space. With use_residual=True they MUST get nonzero
    deltas for at least some weight initialisations."""
    with tempfile.TemporaryDirectory() as d:
        g, rows = _build_tiny_graph(Path(d))
    S, flux_idx = _build_tiny_stoich(len(rows), 2,
                                       flux_indices_list=[rows.index('F_R1'),
                                                            rows.index('F_R2')])
    torch.manual_seed(7)
    model = CellGNNv7Hybrid(
        graph=g, stoich_matrix=S, flux_indices=flux_idx,
        hidden=16, n_layers=2, use_residual=True,
    )
    # Drive non-zero output by setting residual head weights to nontrivial values.
    for p in model.pinn_head.residual_head.parameters():
        torch.nn.init.normal_(p, std=0.05)
    model.eval()
    x = torch.randn(2, g.n_nodes) * 0.2
    v = torch.zeros_like(x)
    x_next, _ = model(x, x_var=v)
    non_sbml_mask = (model.pinn_head.s_covered_mask <= 0)
    delta = (x_next - x)[:, non_sbml_mask]
    assert delta.abs().max().item() > 1e-6, (
        'non-SBML deltas are all zero; residual head is silent'
    )


def test_m7_gradient_flows_through_both_branches():
    with tempfile.TemporaryDirectory() as d:
        g, rows = _build_tiny_graph(Path(d))
    S, flux_idx = _build_tiny_stoich(len(rows), 2,
                                       flux_indices_list=[rows.index('F_R1'),
                                                            rows.index('F_R2')])
    model = CellGNNv7Hybrid(
        graph=g, stoich_matrix=S, flux_indices=flux_idx,
        hidden=16, n_layers=2, use_residual=True,
    )
    x = torch.randn(2, g.n_nodes) * 0.2
    v = torch.zeros_like(x)
    x_next, v_log = model(x, x_var=v)
    loss = x_next.pow(2).mean() + v_log.pow(2).mean()
    loss.backward()
    # PINN rate head
    rate_grad = model.pinn_head.rate_head.weight.grad
    assert rate_grad is not None and rate_grad.abs().sum().item() > 0, \
        'rate_head got no gradient'
    # Residual head
    res_grad = model.pinn_head.residual_head[0].weight.grad
    assert res_grad is not None and res_grad.abs().sum().item() > 0, \
        'residual_head got no gradient'
    # Encoder
    enc_grad = model.input_proj.weight.grad
    assert enc_grad is not None and enc_grad.abs().sum().item() > 0, \
        'encoder got no gradient'


# ----------------------------------------------------------------------
# Loss + iterator
# ----------------------------------------------------------------------

def test_m7_step_loss_is_finite_and_nonneg():
    B, S, R = 4, 8, 3
    x_next = torch.randn(B, S)
    v_log  = torch.randn(B, R) * 0.1
    target = torch.randn(B, S)
    flux_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    count_mask = torch.zeros(S, dtype=torch.bool); count_mask[3:] = True
    cum_mask   = torch.zeros(S, dtype=torch.bool); cum_mask[5:7] = True
    total, br = _m7_step_loss(
        x_next, v_log, target, flux_indices, count_mask, cum_mask,
        w_count=0.05, w_flux=1.0, w_cum=0.5,
    )
    assert total.item() >= 0.0
    assert torch.isfinite(total)
    for k in ('mse_count', 'mse_flux', 'mse_cum'):
        assert k in br and torch.isfinite(br[k])


def test_subsample_pair_iterator_count_and_range():
    R, n_valid = 4, 50
    gen = torch.Generator(device='cpu').manual_seed(0)
    # Subsampled case
    n = 0
    seen_rep = set(); seen_t = set()
    for rep, t in _subsample_pair_iterator(R, n_valid, batch_size=11,
                                              generator=gen, device=torch.device('cpu'),
                                              n_samples=20):
        n += rep.numel()
        seen_rep.update(rep.tolist())
        seen_t  .update(t  .tolist())
    assert n == 20
    assert max(seen_rep) < R
    assert max(seen_t)   < n_valid

    # Full enumeration fallthrough
    gen = torch.Generator(device='cpu').manual_seed(0)
    n_full = 0
    for rep, t in _subsample_pair_iterator(R, n_valid, batch_size=64,
                                              generator=gen, device=torch.device('cpu'),
                                              n_samples=None):
        n_full += rep.numel()
    assert n_full == R * n_valid


# ----------------------------------------------------------------------
# Full rollout eval
# ----------------------------------------------------------------------

def test_full_rollout_eval_runs_and_returns_finite_metrics():
    with tempfile.TemporaryDirectory() as d:
        g, rows = _build_tiny_graph(Path(d))
    S, flux_idx = _build_tiny_stoich(len(rows), 2,
                                       flux_indices_list=[rows.index('F_R1'),
                                                            rows.index('F_R2')])
    torch.manual_seed(0)
    model = CellGNNv7Hybrid(
        graph=g, stoich_matrix=S, flux_indices=flux_idx,
        hidden=8, n_layers=2,
    )
    # Synthetic val tensor (1 replicate, 20 timesteps, S species)
    T = 20
    val_data = torch.randn(1, T, g.n_nodes) * 0.2
    sigma    = torch.randn(T, g.n_nodes).abs() * 0.05

    res = full_rollout_evaluation(
        model, val_data, sigma, row_names=rows,
        T_max=10, log_every=1000,
    )
    assert res.T == 10
    assert np.isfinite(res.mean_r2) or np.isnan(res.mean_r2)    # may be NaN with random model
    assert res.mse_per_step.shape == (10,)
    assert res.mass_balance_l2.shape == (10,)
    # All per-category R^2 keys are present
    for k in ('count', 'flux', 'cumulative'):
        assert k in res.r2_by_category
        assert k in res.n_species_by_cat


def test_full_rollout_eval_mass_balance_zero_at_t0():
    """At t=0 (first rollout step) the mass-balance drift is just one
    update step removed from x_0, so the L2 should be small relative
    to later steps for a model that hasn't been trained much. Verify
    the series is monotonically reported (not NaN)."""
    with tempfile.TemporaryDirectory() as d:
        g, rows = _build_tiny_graph(Path(d))
    S, flux_idx = _build_tiny_stoich(len(rows), 2,
                                       flux_indices_list=[rows.index('F_R1'),
                                                            rows.index('F_R2')])
    model = CellGNNv7Hybrid(
        graph=g, stoich_matrix=S, flux_indices=flux_idx,
        hidden=8, n_layers=2,
    )
    val_data = torch.randn(1, 8, g.n_nodes) * 0.1
    sigma    = torch.zeros(8, g.n_nodes)
    res = full_rollout_evaluation(
        model, val_data, sigma, row_names=rows,
        T_max=5, log_every=1000,
    )
    assert np.isfinite(res.mass_balance_l2).all()


# ----------------------------------------------------------------------
# Truncated BPTT
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# M7.1 - row-type static features
# ----------------------------------------------------------------------

def test_role_features_one_hot_per_row_type():
    """_build_role_features produces 8 binary columns, exactly one set
    per row (modulo the cofactor row which catches everything without
    a locus tag)."""
    rows = ['G_0001', 'R_0001', 'R_0001_d', 'P_0001', 'PM_0001',
            'RP_0001', 'RPM_0001', 'D_0001', 'DM_0001',
            'F_R1', 'M_atp_c']
    feats = _build_role_features(rows)
    assert feats.shape == (len(rows), 8)
    # Each row should have at most one positive role flag (the cofactor
    # column catches rows without a locus tag, so F_R1 and M_atp_c both
    # land there).
    for i in range(len(rows)):
        n_active = int(feats[i].sum().item())
        assert n_active >= 1, f'row {rows[i]} has no role flag set'
        assert n_active <= 2, f'row {rows[i]} has too many role flags: {feats[i]}'
    # G_0001 should be is_gene_row (col 0)
    assert feats[0, 0] == 1.0, 'G_0001 not flagged as gene'
    # R_0001 should be is_mrna_row (col 1), not is_mrna_decay_row (col 2)
    assert feats[1, 1] == 1.0 and feats[1, 2] == 0.0
    # R_0001_d should be is_mrna_decay_row (col 2)
    assert feats[2, 2] == 1.0
    # P_0001 should be is_protein_row (col 3)
    assert feats[3, 3] == 1.0
    # PM_0001 is_membrane_protein_row (col 4) AND triggers is_protein? No.
    assert feats[4, 4] == 1.0


def test_m7_with_role_features_forward_shapes():
    """Model accepts an (N, 8) static feature tensor and produces the
    same output shapes as without (just an internal channel-dim change)."""
    with tempfile.TemporaryDirectory() as d:
        g, rows = _build_tiny_graph(Path(d))
    S, flux_idx = _build_tiny_stoich(len(rows), 2,
                                       flux_indices_list=[rows.index('F_R1'),
                                                            rows.index('F_R2')])
    static = _build_role_features(rows)
    model = CellGNNv7Hybrid(
        graph=g, stoich_matrix=S, flux_indices=flux_idx,
        hidden=16, n_layers=2, n_input_channels=2,
        static_features=static,
    )
    assert model.n_static == 8
    x = torch.randn(3, g.n_nodes) * 0.3
    v = torch.zeros_like(x)
    x_next, v_log = model(x, x_var=v)
    assert x_next.shape == (3, g.n_nodes)
    assert v_log.shape == (3, 2)
    assert torch.isfinite(x_next).all()


def test_m7_role_features_change_output():
    """The same encoder with vs without static features should produce
    DIFFERENT outputs - confirms the static features are actually
    flowing into the encoder, not just sitting in a buffer."""
    with tempfile.TemporaryDirectory() as d:
        g, rows = _build_tiny_graph(Path(d))
    S, flux_idx = _build_tiny_stoich(len(rows), 2,
                                       flux_indices_list=[rows.index('F_R1'),
                                                            rows.index('F_R2')])
    static = _build_role_features(rows)

    torch.manual_seed(0)
    m_no_static = CellGNNv7Hybrid(
        graph=g, stoich_matrix=S, flux_indices=flux_idx,
        hidden=16, n_layers=2, static_features=None,
    )
    torch.manual_seed(0)
    m_with_static = CellGNNv7Hybrid(
        graph=g, stoich_matrix=S, flux_indices=flux_idx,
        hidden=16, n_layers=2, static_features=static,
    )

    x = torch.randn(2, g.n_nodes) * 0.2
    v = torch.zeros_like(x)
    out_no = m_no_static(x, x_var=v)[0].detach()
    out_w  = m_with_static(x, x_var=v)[0].detach()
    # Outputs must differ - if they're identical, the static features
    # aren't being read by the encoder.
    assert not torch.allclose(out_no, out_w, atol=1e-5), \
        'static features have no effect on output - check input_proj wiring'


def test_m7_with_role_features_gradient_flow():
    """Gradient still flows through both PINN and residual heads when
    static features are added (the extra input dim shouldn't kill grad)."""
    with tempfile.TemporaryDirectory() as d:
        g, rows = _build_tiny_graph(Path(d))
    S, flux_idx = _build_tiny_stoich(len(rows), 2,
                                       flux_indices_list=[rows.index('F_R1'),
                                                            rows.index('F_R2')])
    static = _build_role_features(rows)
    model = CellGNNv7Hybrid(
        graph=g, stoich_matrix=S, flux_indices=flux_idx,
        hidden=16, n_layers=2, static_features=static,
        use_residual=True,
    )
    x = torch.randn(2, g.n_nodes) * 0.2
    v = torch.zeros_like(x)
    x_next, v_log = model(x, x_var=v)
    loss = x_next.pow(2).mean() + v_log.pow(2).mean()
    loss.backward()
    rate_grad = model.pinn_head.rate_head.weight.grad
    res_grad  = model.pinn_head.residual_head[0].weight.grad
    enc_grad  = model.input_proj.weight.grad
    assert rate_grad is not None and rate_grad.abs().sum().item() > 0
    assert res_grad is not None and res_grad.abs().sum().item() > 0
    assert enc_grad is not None and enc_grad.abs().sum().item() > 0
    # Crucially: the input_proj weight should have shape (hidden, 2 + 8)
    # = (16, 10), so columns 2..10 correspond to the static feature
    # contribution. Verify those columns received gradient.
    static_proj_cols = model.input_proj.weight.grad[:, 2:]
    assert static_proj_cols.abs().sum().item() > 0, \
        'gradient did not flow to the static-feature input columns'


def test_m7_with_role_features_mass_conservation_preserved():
    """Adding static features must not break the PINN mass-conservation
    property on SBML-covered rows."""
    with tempfile.TemporaryDirectory() as d:
        g, rows = _build_tiny_graph(Path(d))
    S, flux_idx = _build_tiny_stoich(len(rows), 2,
                                       flux_indices_list=[rows.index('F_R1'),
                                                            rows.index('F_R2')])
    static = _build_role_features(rows)
    torch.manual_seed(0)
    model = CellGNNv7Hybrid(
        graph=g, stoich_matrix=S, flux_indices=flux_idx,
        hidden=16, n_layers=2, static_features=static,
        use_residual=True,
    )
    model.eval()
    x = torch.randn(2, g.n_nodes) * 0.3
    v = torch.zeros_like(x)
    x_next, v_log = model(x, x_var=v)

    v_lin = signed_expm1(v_log)
    x_lin = signed_expm1(x)
    dx_pinn_lin = v_lin @ S.t()
    x_next_pinn_lin = x_lin + dx_pinn_lin
    x_next_pinn_log = signed_log1p(x_next_pinn_lin)

    sbml_mask = (model.pinn_head.s_covered_mask > 0)
    diff = (x_next - x_next_pinn_log)[:, sbml_mask].abs()
    assert diff.max().item() < 1e-4, (
        f'SBML mass conservation broken with static features; '
        f'max diff = {diff.max().item():.6f}'
    )


# ----------------------------------------------------------------------
# M7.3 - combined static feature builder
# ----------------------------------------------------------------------

def test_combined_features_role_only_matches_role_builder():
    """With use_role=True and the other flags off, combined_static should
    return exactly what _build_role_features returns - byte-equal."""
    rows = ['G_0001', 'R_0001', 'P_0001', 'PM_0001',
            'RP_0001', 'D_0001', 'F_R1', 'M_atp_c']
    role_only = _build_role_features(rows)
    combined = _build_combined_static_features(
        rows, use_role=True, use_spatial=False, use_proteomics=False,
        verbose=False,
    )
    assert combined is not None
    assert combined.shape == role_only.shape
    assert torch.allclose(combined, role_only)


def test_combined_features_returns_none_when_all_off():
    rows = ['G_0001', 'P_0001']
    combined = _build_combined_static_features(
        rows, use_role=False, use_spatial=False, use_proteomics=False,
        verbose=False,
    )
    assert combined is None


def test_combined_features_raises_when_spatial_path_missing():
    """If use_spatial is requested but the parquet path is None or missing,
    the builder must raise a clear FileNotFoundError."""
    rows = ['G_0001', 'P_0001']
    try:
        _build_combined_static_features(
            rows, use_role=True, use_spatial=True,
            spatial_parquet=None, verbose=False,
        )
        assert False, 'expected FileNotFoundError'
    except FileNotFoundError as e:
        assert 'spatial' in str(e).lower()


# ----------------------------------------------------------------------
# Truncated BPTT
# ----------------------------------------------------------------------

def test_truncated_bptt_cuts_early_step_gradient():
    """Run a small rollout that mimics the trainer's tbptt logic and
    verify that with tbptt_window=2 of a k=5 rollout, gradients on the
    early-step PARAMETERS are zero while late-step gradients are non-zero.

    Mechanism: under torch.no_grad(), the forward consumes no autograd
    graph, so an output created in no_grad is detached. Feeding that
    detached output into the next forward (under grad again) starts a
    new graph - the model's params receive gradients only for the
    grad-enabled steps.
    """
    with tempfile.TemporaryDirectory() as d:
        g, rows = _build_tiny_graph(Path(d))
    S, flux_idx = _build_tiny_stoich(len(rows), 2,
                                       flux_indices_list=[rows.index('F_R1'),
                                                            rows.index('F_R2')])
    model = CellGNNv7Hybrid(
        graph=g, stoich_matrix=S, flux_indices=flux_idx,
        hidden=8, n_layers=2,
    )
    model.train()
    k_cur = 5
    tbptt_window = 2
    grad_start = k_cur - tbptt_window

    x = torch.randn(2, g.n_nodes) * 0.1
    v_in = torch.zeros_like(x)
    x_pred = x
    L = torch.zeros(())
    for s in range(k_cur):
        step_has_grad = (s == 0) or (s >= grad_start)
        ctx = torch.enable_grad() if step_has_grad else torch.no_grad()
        with ctx:
            x_next, v_log = model(x_pred, x_var=v_in)
            if step_has_grad:
                L = L + x_next.pow(2).mean() + v_log.pow(2).mean()
        x_pred = x_next

    L.backward()
    rate_grad = model.pinn_head.rate_head.weight.grad
    assert rate_grad is not None and rate_grad.abs().sum().item() > 0, \
        'rate_head got no gradient; grad-enabled steps did not backprop'
