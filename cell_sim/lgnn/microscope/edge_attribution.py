"""Edge attribution microscope — dump per-edge attention from
M1+axis2 (or M2) across many timesteps and aggregate.

The axis-2 attention layer produces a softmax-weighted distribution
α over each destination node's incoming non-self edges. We capture
α at every timestep on a sample slice of the trajectory, then
aggregate per edge to get:

  - mean attention (how heavily is this edge used on average?)
  - std attention (does its usage vary with cell state?)
  - top-attended source per destination (the "load-bearing connection"
    interpretation from the axis-2 thesis)

Output: a parquet that the pattern-atlas / knockout-sweep workflow
can join against for "which connections does the model think matter."

The model must be an M1+axis2 (CellGNNv1Axis2) or M2 (CellGNNv2)
instance — anything with `_AttentionGNNLayer` instances exposing
attn_mlp, kind_embedding, and the self-loop masking convention.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def capture_attention(
    model: torch.nn.Module,
    X_states: torch.Tensor,
    layer_idx: int = 0,
    sample_every: int = 10,
) -> dict:
    """Run the attention layer at `layer_idx` on `X_states` and capture
    per-edge α at every `sample_every`-th timestep.

    Args
    ----
    model : trained M1+axis2 or M2 model, eval() mode, on device.
    X_states : (T, S) signed-log1p states. On the same device as model.
    layer_idx : which message-passing layer to inspect (0 = first).
    sample_every : skip this many timesteps between captures. T=7200
        with sample_every=10 → 720 captures.

    Returns
    -------
    dict with:
        T_sampled     -- number of timesteps captured
        edge_index    -- (2, E) the model's sorted edge index
        edge_kind     -- (E,) edge-kind labels
        alpha         -- (T_sampled, E) per-edge α over time
        timestep_idx  -- (T_sampled,) which raw timesteps were captured
    """
    from cell_sim.lgnn.models.gnn_v1_axis2 import _segment_softmax
    from cell_sim.lgnn.data.species_graph import EdgeKind

    device = X_states.device
    layer = model.layers[layer_idx]
    edge_index = model.edge_index                                   # (2, E)
    edge_attr = model.edge_attr
    edge_kind = model.edge_kind
    N = model.n_nodes
    E = int(edge_index.shape[1])

    T = int(X_states.shape[0])
    sampled_t = np.arange(0, T, sample_every, dtype=np.int64)
    T_s = len(sampled_t)

    # Pre-compute the self-loop mask once.
    is_self_loop = (edge_kind == int(EdgeKind.SELF_LOOP))           # (E,)
    src_all = edge_index[0]
    dst_all = edge_index[1]

    alpha_stack = torch.empty(T_s, E, device=device, dtype=torch.float32)

    for i, t in enumerate(sampled_t):
        x = X_states[t:t + 1]                                       # (1, S)
        h = model.input_proj(x.unsqueeze(-1))                       # (1, N, H)
        h_src = h.index_select(1, src_all)
        h_dst = h.index_select(1, dst_all)
        attr_b = edge_attr.unsqueeze(0).expand(1, -1, -1)
        kind_emb = layer.kind_embedding(edge_kind)
        kind_emb_b = kind_emb.unsqueeze(0).expand(1, -1, -1)
        inp = torch.cat([h_src, h_dst, attr_b, kind_emb_b], dim=-1)
        logits = layer.attn_mlp(inp).squeeze(-1)                    # (1, E)
        logits = logits.masked_fill(is_self_loop.unsqueeze(0),
                                      float('-inf'))
        alpha = _segment_softmax(logits, dst_all, N)                # (1, E)
        alpha_stack[i] = alpha.squeeze(0).float()

    return {
        'T_sampled':    T_s,
        'edge_index':   edge_index.cpu().numpy(),
        'edge_kind':    edge_kind.cpu().numpy(),
        'alpha':        alpha_stack.cpu().numpy(),
        'timestep_idx': sampled_t,
    }


def aggregate_attention(
    capture: dict,
    row_names: Sequence[str],
) -> pd.DataFrame:
    """Aggregate per-edge α to mean/std/quartiles. Returns a DataFrame
    suitable for parquet export.

    Columns:
        edge_idx, src_idx, dst_idx, src_name, dst_name, edge_kind,
        mean_alpha, std_alpha, q25_alpha, q50_alpha, q75_alpha,
        max_alpha, frac_above_0_5
    """
    alpha = capture['alpha']                                         # (T_s, E)
    edge_index = capture['edge_index']
    edge_kind = capture['edge_kind']
    E = alpha.shape[1]

    mean_a = alpha.mean(axis=0)
    std_a  = alpha.std(axis=0)
    q25    = np.percentile(alpha, 25, axis=0)
    q50    = np.percentile(alpha, 50, axis=0)
    q75    = np.percentile(alpha, 75, axis=0)
    max_a  = alpha.max(axis=0)
    frac_dom = (alpha > 0.5).mean(axis=0)

    src_idx = edge_index[0]
    dst_idx = edge_index[1]
    return pd.DataFrame({
        'edge_idx':       np.arange(E, dtype=np.int64),
        'src_idx':        src_idx,
        'dst_idx':        dst_idx,
        'src_name':       [row_names[i] for i in src_idx],
        'dst_name':       [row_names[i] for i in dst_idx],
        'edge_kind':      edge_kind,
        'mean_alpha':     mean_a,
        'std_alpha':      std_a,
        'q25_alpha':      q25,
        'q50_alpha':      q50,
        'q75_alpha':      q75,
        'max_alpha':      max_a,
        'frac_above_0.5': frac_dom,
    })


def top_attended_per_node(
    aggregated: pd.DataFrame,
    top_k: int = 3,
    metric: str = 'mean_alpha',
) -> pd.DataFrame:
    """For each destination node, return the top-K most-attended
    incoming edges by `metric`.

    Returns DataFrame: dst_idx, dst_name, rank, src_idx, src_name,
                       edge_kind, <metric>.
    """
    # Stable sort by metric within each dst group, take top-K
    sorted_df = aggregated.sort_values(
        ['dst_idx', metric], ascending=[True, False],
    )
    sorted_df['rank'] = sorted_df.groupby('dst_idx').cumcount() + 1
    top = sorted_df[sorted_df['rank'] <= top_k]
    return top[['dst_idx', 'dst_name', 'rank',
                 'src_idx', 'src_name', 'edge_kind', metric]
              ].reset_index(drop=True)
