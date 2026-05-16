"""Pre-acceptance safety checks for M7-Evolve v2 candidates.

Before a mutated candidate gets its full benchmark, these snapshot
tests run quickly (~30 sec total) and reject obviously broken
mutations. This stops the loop from spending 10 min benchmarking a
candidate whose model can't even forward.

Exit code 0 = pass, non-zero = fail with reason on stderr.
"""
from __future__ import annotations

import os
import sys
import traceback

import torch

sys.path.insert(0, '/content/cell')


def _fail(msg: str) -> None:
    print(f'SNAPSHOT_FAIL: {msg}', file=sys.stderr)
    sys.exit(1)


def main(checkpoint_path: str) -> None:
    from cell_sim.lgnn.data.species_graph import load_species_graph
    from cell_sim.lgnn.self_improve.iterate import load_m7_with_state

    DRIVE_DIR = os.environ.get('M7_DRIVE_DIR',
                                '/content/drive/MyDrive/cell_count_dynamics')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test 1: model loads
    try:
        sg = load_species_graph(f'{DRIVE_DIR}/species_graph_v7.pt')
        row_names = list(sg.row_names)
        model, cfg, fidx = load_m7_with_state(
            checkpoint_path, sg, row_names, DEVICE,
        )
    except Exception as e:
        _fail(f'model load: {type(e).__name__}: {e}\n{traceback.format_exc()}')

    # Test 2: forward pass produces correct shape, no NaN
    try:
        x = torch.randn(2, len(row_names), device=DEVICE,
                         dtype=torch.float32) * 0.1
        v = torch.randn(2, len(row_names), device=DEVICE,
                         dtype=torch.float32).abs() * 0.05
        with torch.no_grad():
            out = model(x, x_var=v)
        x_next = out[0] if isinstance(out, tuple) else out
        if x_next.shape != x.shape:
            _fail(f'forward shape mismatch: got {tuple(x_next.shape)} '
                  f'expected {tuple(x.shape)}')
        if torch.isnan(x_next).any():
            _fail('forward produced NaN values')
        if torch.isinf(x_next).any():
            _fail('forward produced inf values')
    except Exception as e:
        _fail(f'forward: {type(e).__name__}: {e}\n{traceback.format_exc()}')

    # Test 3: short rollout doesn't diverge
    try:
        val_data = torch.load(f'{DRIVE_DIR}/preload_val.pt',
                               map_location=DEVICE, weights_only=False)
        sigma = torch.load(f'{DRIVE_DIR}/sigma_train_reps_1to49.pt',
                            map_location=DEVICE, weights_only=False)
        x = val_data[0, 100].clone().to(torch.float32)
        for s in range(10):
            v_in = sigma[100 + s].unsqueeze(0).to(torch.float32)
            with torch.no_grad():
                out = model(x.unsqueeze(0), x_var=v_in)
            x = (out[0] if isinstance(out, tuple) else out).squeeze(0)
            if torch.isnan(x).any() or x.abs().max() > 1000:
                _fail(f'rollout diverged at step {s} '
                      f'(max abs value: {x.abs().max().item():.2f})')
    except Exception as e:
        _fail(f'rollout: {type(e).__name__}: {e}\n{traceback.format_exc()}')

    # Test 4: knockout produces nonzero impact (single gene)
    try:
        from cell_sim.lgnn.analysis.knockout_sweep import _full_locus_row_indices
        groups = _full_locus_row_indices(row_names)
        if not groups:
            _fail('no knockout groups found in row_names')
        locus = sorted(groups.keys())[0]
        x_base = val_data[0, 100].clone().to(torch.float32)
        x_ko = x_base.clone()
        for r in groups[locus]:
            x_ko[r] = 0.0
        with torch.no_grad():
            out_base = model(x_base.unsqueeze(0),
                              x_var=sigma[100].unsqueeze(0).to(torch.float32))
            out_ko = model(x_ko.unsqueeze(0),
                            x_var=sigma[100].unsqueeze(0).to(torch.float32))
        x_b = (out_base[0] if isinstance(out_base, tuple)
               else out_base).squeeze(0)
        x_k = (out_ko[0] if isinstance(out_ko, tuple)
               else out_ko).squeeze(0)
        delta = (x_b - x_k).abs().mean().item()
        if delta < 1e-6:
            _fail(f'knockout produced no measurable impact (Δ={delta:.2e})')
    except Exception as e:
        _fail(f'knockout: {type(e).__name__}: {e}\n{traceback.format_exc()}')

    print('SNAPSHOT_PASS: all 4 safety checks passed')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: snapshot_tests.py <checkpoint_path>', file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
