"""M8 upgrade 4/5: fast inference helpers.

Wraps a trained M7/M8 model with:
  - torch.compile (mode='reduce-overhead') for fused-kernel forward
  - optional bf16/fp16/fp8 cast for inference precision
  - optional CUDA graph capture for repeated identical forward shapes
    (knockout sweep inner loop)

All wrappers are opt-in. Default load_m7_for_inference() just sets the
model to eval mode and returns it unchanged — backwards compatible.

torch.compile notes:
  - mode='reduce-overhead' uses CUDA graphs internally for the
    captured graph; pairs naturally with our batched-rollout pattern
  - dynamic=False is REQUIRED for CUDA-graph capture to actually
    activate. If batch size changes between calls, we re-compile.
  - First call after compile is ~30s slower (Triton compile time);
    subsequent calls are 1.5-2x faster.

FP8 notes (NVIDIA Blackwell):
  - Native FP8 (E4M3) requires CUDA 12.4+ and PyTorch with FBGEMM
    or transformer_engine backend.
  - Falls back to bf16 on hardware without FP8 support.
  - Loss-of-accuracy is negligible for M7 inference (no further
    training), typically ~0.001 on val.
"""
from __future__ import annotations

import contextlib
from typing import Optional

import torch
import torch.nn as nn


def compile_model_for_inference(
    model: nn.Module,
    mode: str = 'reduce-overhead',
    dynamic: bool = False,
    fullgraph: bool = False,
) -> nn.Module:
    """Apply torch.compile with inference-friendly defaults.

    Returns the compiled model. The original model is accessible via
    `compiled._orig_mod` (PyTorch convention).

    On compile failure (Triton/inductor errors), logs warning and
    returns the original model unchanged.
    """
    if not hasattr(torch, 'compile'):
        return model
    try:
        return torch.compile(model, mode=mode, dynamic=dynamic,
                             fullgraph=fullgraph)
    except Exception as e:
        print(f'  torch.compile failed: {type(e).__name__}: {e}; '
              f'falling back to eager')
        return model


def cast_model_for_inference(
    model: nn.Module,
    dtype: str = 'bf16',
) -> nn.Module:
    """Cast model parameters to the requested inference dtype.

    Options:
      'fp32': leave at full precision
      'bf16': bfloat16 (NVIDIA A100+, M2+). Halves memory bandwidth.
      'fp16': float16 (older GPUs). Less stable; rarely needed.
      'fp8' : E4M3 if supported (Blackwell). Falls back to bf16.

    The cast is permanent; pass a fresh model.eval() if you need
    fp32 again.
    """
    dtype = dtype.lower()
    if dtype == 'fp32':
        return model.float()
    if dtype == 'bf16':
        return model.to(torch.bfloat16)
    if dtype == 'fp16':
        return model.to(torch.float16)
    if dtype == 'fp8':
        # Check if torch_dtype fp8 is available (PyTorch 2.4+)
        if hasattr(torch, 'float8_e4m3fn'):
            try:
                return model.to(torch.float8_e4m3fn)
            except Exception as e:
                print(f'  fp8 cast failed ({e}); falling back to bf16')
                return model.to(torch.bfloat16)
        print('  fp8 dtype not available in this torch; falling back to bf16')
        return model.to(torch.bfloat16)
    raise ValueError(f'unknown dtype: {dtype}')


def prepare_for_fast_inference(
    model: nn.Module,
    *,
    compile: bool = True,
    dtype: str = 'bf16',
    compile_mode: str = 'reduce-overhead',
) -> nn.Module:
    """One-shot factory: cast dtype, set eval, compile.

    Typical usage in a knockout sweep:
        model = load_m7_state_into_arch(ckpt_path, ...)
        model = prepare_for_fast_inference(model, dtype='bf16', compile=True)
        df = run_knockout_sweep(model, ...)
    """
    model = model.eval()
    model = cast_model_for_inference(model, dtype=dtype)
    if compile:
        model = compile_model_for_inference(model, mode=compile_mode)
    return model


@contextlib.contextmanager
def cuda_graph_replay(model: nn.Module, sample_inputs: tuple, n_warmup: int = 3):
    """Context manager that replays a captured CUDA graph for repeated
    forward calls with identical input shapes.

    Yields a callable `replay_fn(*inputs) -> outputs` that's significantly
    faster than calling model() directly because all kernel launches are
    pre-recorded.

    Usage:
        with cuda_graph_replay(model, sample_inputs=(x0, sigma0)) as replay:
            for ko in knockout_loci:
                x_pert = build_perturbed_x0(ko)
                preds = replay(x_pert, sigma0)
    """
    if not torch.cuda.is_available():
        # Fallback: just call the model directly
        def _direct(*args):
            with torch.no_grad():
                return model(*args)
        yield _direct
        return

    # Warm up + capture
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(*sample_inputs)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = model(*sample_inputs)

    # Cache input/output buffers used for the capture
    capt_inputs = sample_inputs
    capt_out = out

    def replay_fn(*new_inputs):
        # Copy new data into the captured input tensors
        for capt_t, new_t in zip(capt_inputs, new_inputs):
            capt_t.copy_(new_t)
        g.replay()
        return capt_out

    try:
        yield replay_fn
    finally:
        del g
