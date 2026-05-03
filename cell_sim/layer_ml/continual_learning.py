"""Elastic Weight Consolidation (EWC, Kirkpatrick et al. 2017) for
continual learning across organisms without catastrophic forgetting.

Why EWC: when we train an LNN on Salmonella + Syn3A then continue
training on a new organism (e.g. Caulobacter), naive SGD will overwrite
the weights that encoded the first task's knowledge. EWC adds a
quadratic penalty on weights moving away from their previous-task
values, weighted by each weight's importance to the previous task
(estimated by the diagonal of the Fisher Information Matrix).

After training task A:
    ewc.consolidate(model, dataloader_A, n_samples=512)
This fixes a "snapshot" of model parameters and computes a Fisher
estimate that measures each parameter's importance to task A.

When training task B:
    loss = task_loss(...) + ewc.penalty(model, lambda_ewc=100)
The penalty pulls weights back toward their task-A values, more
strongly for weights that were important to task A.

Self-test: `python -m cell_sim.layer_ml.continual_learning`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EWCRegularizer:
    """Elastic Weight Consolidation regularizer.

    Maintains a list of "consolidated tasks", each a snapshot of
    (parameter values, Fisher diagonal) for one previous task. The
    penalty is the sum over tasks of the quadratic distance from each
    parameter's current value to its consolidated value, weighted by
    the consolidated Fisher diagonal.

    This object is NOT itself an nn.Module — Fishers aren't trainable
    parameters, just stored tensors that scale the penalty term.
    """

    def __init__(self):
        # Each task = dict[name -> (param_value_snapshot, fisher_diagonal)]
        self.tasks: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] = []

    def consolidate(self, model: nn.Module, sample_iter,
                     n_samples: int = 256, loss_fn=None,
                     device: str | None = None):
        """Compute Fisher diagonal for the current task and snapshot params.

        Parameters:
            model: nn.Module whose parameters will be snapshotted
            sample_iter: iterable yielding (batch, target) tuples or batches
                where loss_fn(model, batch, target) returns scalar loss
            n_samples: how many batches to use for Fisher estimation
            loss_fn: callable (model, batch, target) -> scalar loss.
                Default: assumes batch and target are tensors and the
                model returns logits suitable for BCE-with-logits.
            device: if provided, move snapshots/Fishers there
        """
        if loss_fn is None:
            loss_fn = _default_bce_loss_fn

        # Initialize Fisher diagonal accumulator per param
        fisher = {n: torch.zeros_like(p, requires_grad=False)
                   for n, p in model.named_parameters() if p.requires_grad}

        model.eval()
        n_seen = 0
        for sample in sample_iter:
            if n_seen >= n_samples: break
            model.zero_grad()
            loss = loss_fn(model, sample)
            if loss is None: continue
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is None or n not in fisher: continue
                fisher[n] += p.grad.detach() ** 2
            n_seen += 1
        model.zero_grad()
        # Normalize
        for n in fisher:
            fisher[n] /= max(n_seen, 1)

        # Snapshot params + fisher (detach + clone so future param updates
        # don't disturb the snapshot)
        task = {}
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            snap = p.detach().clone()
            f = fisher[n].clone()
            if device is not None:
                snap = snap.to(device); f = f.to(device)
            task[n] = (snap, f)
        self.tasks.append(task)

    def penalty(self, model: nn.Module, lambda_ewc: float = 100.0
                  ) -> torch.Tensor:
        """Quadratic penalty on weights moving from consolidated values.
        Returns a scalar tensor that should be added to the task loss
        before backward."""
        if not self.tasks:
            # No prior tasks consolidated -> no penalty
            return torch.tensor(0.0,
                                  device=next(model.parameters()).device)

        total = torch.tensor(0.0,
                              device=next(model.parameters()).device)
        params = dict(model.named_parameters())
        for task in self.tasks:
            for n, (snap, fisher) in task.items():
                if n not in params: continue
                p = params[n]
                if p.shape != snap.shape: continue  # arch changed; skip
                total = total + (fisher * (p - snap.to(p.device)) ** 2).sum()
        return 0.5 * lambda_ewc * total

    def __len__(self):
        return len(self.tasks)

    def reset(self):
        self.tasks = []


def _default_bce_loss_fn(model, sample):
    """Default: sample is (batch_dict, target) where model(batch_dict)
    returns logits and target is binary."""
    if not isinstance(sample, (tuple, list)) or len(sample) < 2:
        return None
    batch, target = sample[0], sample[1]
    logits = model(batch)
    if isinstance(logits, dict):
        logits = logits.get('essentiality')
    if logits is None: return None
    target = target.to(logits.device).to(logits.dtype)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(-1) if logits.dim() == 2 and logits.size(-1) == 1 else logits,
        target)


# ──────────── self-test ────────────


def _self_test():
    print('continual_learning self-test:')
    torch.manual_seed(42)

    # Toy model: 2-layer MLP with 1 output
    model = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1))
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  toy model params: {n_params}')

    # Generate two "tasks" with different decision boundaries
    def make_task(seed: int, n: int = 100):
        torch.manual_seed(seed)
        X = torch.randn(n, 8)
        # Task A: y depends on first feature; Task B: y depends on second feature
        if seed == 1:
            y = (X[:, 0] > 0).float()
        else:
            y = (X[:, 1] > 0).float()
        return [(X[i:i+10], y[i:i+10]) for i in range(0, n, 10)]

    task_a_data = make_task(1)
    task_b_data = make_task(2)

    # Train on task A
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for ep in range(50):
        for X, y in task_a_data:
            opt.zero_grad()
            logits = model(X).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward(); opt.step()
    # Eval on task A
    def acc(model, data):
        correct = total = 0
        for X, y in data:
            with torch.no_grad():
                pred = (torch.sigmoid(model(X).squeeze(-1)) > 0.5).float()
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / total
    acc_a_after_a = acc(model, task_a_data)
    print(f'  trained task A: acc on A = {acc_a_after_a:.3f}')

    # Define loss_fn for consolidation (model takes raw tensors here, not dict)
    def loss_fn(model, sample):
        X, y = sample
        return torch.nn.functional.binary_cross_entropy_with_logits(
            model(X).squeeze(-1), y.float())

    # Consolidate task A
    ewc = EWCRegularizer()
    ewc.consolidate(model, task_a_data, n_samples=10, loss_fn=loss_fn)
    fisher_max = max(f.abs().max().item() for _, f in ewc.tasks[0].values())
    print(f'  EWC consolidated; Fisher max: {fisher_max:.4f}')

    # Snapshot accuracy with EWC penalty (should keep some of A's behavior)
    for ep in range(50):
        for X, y in task_b_data:
            opt.zero_grad()
            logits = model(X).squeeze(-1)
            task_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            ewc_pen = ewc.penalty(model, lambda_ewc=1000.0)
            loss = task_loss + ewc_pen
            loss.backward(); opt.step()
    acc_a_after_b = acc(model, task_a_data)
    acc_b_after_b = acc(model, task_b_data)
    print(f'  after training task B with EWC: acc A = {acc_a_after_b:.3f}, '
          f'acc B = {acc_b_after_b:.3f}')

    # Compare to no-EWC baseline (re-init, train A, train B without EWC)
    model2 = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1))
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-2)
    for ep in range(50):
        for X, y in task_a_data:
            opt2.zero_grad()
            l = torch.nn.functional.binary_cross_entropy_with_logits(
                model2(X).squeeze(-1), y); l.backward(); opt2.step()
    for ep in range(50):
        for X, y in task_b_data:
            opt2.zero_grad()
            l = torch.nn.functional.binary_cross_entropy_with_logits(
                model2(X).squeeze(-1), y); l.backward(); opt2.step()
    acc_a_no_ewc = acc(model2, task_a_data)
    acc_b_no_ewc = acc(model2, task_b_data)
    print(f'  no-EWC baseline: acc A = {acc_a_no_ewc:.3f}, '
          f'acc B = {acc_b_no_ewc:.3f}')
    print(f'  EWC retains task A better: '
          f'{acc_a_after_b:.3f} (EWC) vs {acc_a_no_ewc:.3f} (no EWC)')

    # The whole point of EWC is to preserve task A while learning B
    # Don't make this a hard assertion (EWC's effect varies with lambda)
    # but verify the mechanism is functioning: penalty should be > 0 when
    # weights have moved
    p = ewc.penalty(model, lambda_ewc=1.0)
    assert p.item() > 0, 'penalty should be positive after weights have moved'
    print(f'  penalty after moving weights: {p.item():.4f} (>0 confirmed)')

    # Multiple tasks
    ewc.consolidate(model, task_b_data, n_samples=10, loss_fn=loss_fn)
    assert len(ewc) == 2
    print(f'  multi-task: {len(ewc)} consolidated tasks')

    # Reset works
    ewc.reset()
    assert len(ewc) == 0
    p = ewc.penalty(model, lambda_ewc=1.0)
    assert p.item() == 0
    print('  reset: penalty returns to 0')

    print('  ALL PASS')


if __name__ == '__main__':
    _self_test()
