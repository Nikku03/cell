"""Optimizer alternatives to AdamW for M8 training.

  - AdEMAMix (Pagliardini, Ablin, Grangier — Apple, 2024)
      arXiv:2409.03137. Combines a fast EMA (Adam-like) with a slow EMA
      (decay ≈ 0.9999). The slow EMA keeps old gradients alive longer,
      giving 10-30% faster convergence on most language and vision
      benchmarks. Works well as a drop-in AdamW replacement.

  - Lion (Chen et al. — Google, 2023)
      arXiv:2302.06675. Sign-based update with momentum. Half the
      memory of AdamW (no second-moment estimate). Often slightly
      worse early but comparable final accuracy with a smaller LR.

Both are pure-Python implementations following the canonical
algorithms. No external dependencies beyond torch.
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
from torch.optim.optimizer import Optimizer


class AdEMAMix(Optimizer):
    """AdEMAMix optimizer.

    Standard AdamW state plus a SECOND EMA buffer (m2) with much
    slower decay (β3 ≈ 0.9999). The update direction is a mix:
        update ∝ (m1 + α * m2) / sqrt(v) + weight_decay * θ
    where m1 is the usual fast EMA and m2 is the slow one. The slow
    EMA preserves direction information from earlier gradients,
    which empirically helps convergence speed.

    Args
    ----
    lr           : learning rate (1e-4 to 3e-4 typical for M7)
    betas        : (β1, β2) for fast EMA (m1, v); standard AdamW values
    beta3        : slow EMA decay; 0.9999 is the paper default
    alpha        : weight of slow EMA in update; 5.0 paper default,
                    but 2-3 works well at smaller scale
    eps          : numerical stability constant for sqrt(v)
    weight_decay : decoupled weight decay (AdamW-style)

    Memory cost: 50% more than AdamW (extra m2 buffer per param).
    Compute cost: ~10% per step (one extra EMA update).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        beta3: float = 0.9999,
        alpha: float = 2.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if not 0.0 < lr:
            raise ValueError(f'invalid lr: {lr}')
        defaults = dict(lr=lr, betas=betas, beta3=beta3, alpha=alpha,
                        eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            beta3 = group['beta3']
            alpha = group['alpha']
            lr = group['lr']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg']      = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq']   = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_slow'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m1 = state['exp_avg']
                v  = state['exp_avg_sq']
                m2 = state['exp_avg_slow']

                state['step'] += 1
                step = state['step']

                # Fast EMA (Adam m1)
                m1.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # Variance (Adam v)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                # Slow EMA (m2) — NOT bias-corrected per paper
                m2.mul_(beta3).add_(grad, alpha=1.0 - beta3)

                # Bias correction for m1, v (m2 has no bias correction per paper)
                bias1 = 1.0 - beta1 ** step
                bias2 = 1.0 - beta2 ** step
                m1_hat = m1 / bias1
                v_hat  = v / bias2

                # Combined update direction
                denom = v_hat.sqrt().add_(eps)
                direction = (m1_hat + alpha * m2) / denom

                # Decoupled weight decay (AdamW style)
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(direction, alpha=-lr)

        return loss


class Lion(Optimizer):
    """Lion optimizer (Chen et al., Google 2023).

    Sign-based update with momentum:
        c_t = β1 * m_{t-1} + (1 - β1) * g
        update = sign(c_t)
        m_t = β2 * m_{t-1} + (1 - β2) * g

    Half the memory of AdamW. Recommended LR ≈ 1/10 of AdamW's LR.

    Args
    ----
    lr           : learning rate. For M7, try 1e-5 to 3e-5 (10x smaller
                    than AdamW's 1e-4).
    betas        : (β1, β2). Lion paper recommends (0.95, 0.98).
    weight_decay : decoupled weight decay. Lion benefits from larger
                    weight_decay than AdamW (paper used 0.01-0.1).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-5,
        betas: Tuple[float, float] = (0.95, 0.98),
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                m = state['exp_avg']
                # Combined update (paper eq 1)
                c = m.mul(beta1).add_(grad, alpha=1.0 - beta1)
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(torch.sign(c), alpha=-lr)
                # Update momentum
                m.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


def build_optimizer(
    params,
    name: str,
    lr: float,
    weight_decay: float = 0.0,
) -> Optimizer:
    """Factory matching M7TrainConfig.optimizer_name.

    'adamw'    : default torch.optim.AdamW
    'ademamix' : AdEMAMix (Apple 2024)
    'lion'     : Lion (Google 2023). Note: use lr ≈ AdamW lr / 10
    """
    name = name.lower()
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == 'ademamix':
        return AdEMAMix(params, lr=lr, weight_decay=weight_decay)
    if name == 'lion':
        return Lion(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(
        f'unknown optimizer_name: {name!r}. '
        f"Use 'adamw', 'ademamix', or 'lion'."
    )
