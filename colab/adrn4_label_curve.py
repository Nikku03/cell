"""Acquisition curve out to 512 labels: does the ordering hold past the measured range?

Why this exists
---------------
ADRN-2B's checkpoints stop at 128 labels, so A(256) was never measured, and the
container rollback destroyed both the uploaded ADRN-2B package and the scratchpad copy,
so the original harness cannot be re-run.  What CAN be done is rebuild the essential
comparison from the formulas recorded in this repo's own committed modules and push the
curve further.

THIS IS A REIMPLEMENTATION, NOT THE ORIGINAL HARNESS.  Absolute values will not match
ADRN-2B.  What it can settle is the question actually being asked: at 128 labels the
per-doubling increments were still ACCELERATING (polynomial_oracle went +0.0162,
+0.0281, +0.0408, +0.0538), which means a log-linear extrapolation to A(256) is fitted
to the wrong shape.  This measures where the curve actually goes.

The representations, reproduced from the formulas documented in
colab/adrn2b_digital_tabulation.py and colab/adrn2b_gate_b_decomposition.py:

    polynomial   3-step signed suffix (48) + current-step signed pair products (120)
                 = 168 coordinates, fixed and unlearned. This is the control that won
                 every routing mode in ADRN-2B.
    digital      + per-channel running parity (16), first-occurrence times (16),
                 last-occurrence times (16), ordered toggle/reset states (240) = 288
                 more, also fixed and unlearned. This is the arm that PASSED Gate B
                 once the GRU was switched off.
    workspace    polynomial + digital + a 32-unit GRU trained offline on training rules
    transformer  polynomial + a 32-unit attention encoder trained the same way

Protocol: each held-out rule gets its own adapter (oracle context banking -- the
mechanism ADRN-2B showed actually works), trained by an online delta rule on k labels,
then scored on a probe set that is IDENTICAL across every arm and every k.

Unit of replication is the SEED, each of which draws its own disjoint train/test rule
split and its own probes. MDE = 3*sd/sqrt(n_seeds) on the paired per-seed gap.

PREDECLARED, before running: the ordering should hold -- polynomial-based arms ahead of
the transformer, and the GRU adding nothing over the fixed digital basis -- and the
curve should keep rising through 256 rather than saturating, because the Bayesian
oracle-grammar arm reached A(32) = 1.0000 on this task family, so there is a great deal
of headroom above the ~0.63 reached at 128.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

N_CHANNELS = 16
STEPS = 16
CHECKPOINTS = (8, 16, 32, 64, 128, 256, 512)
OPERATIONS = ("and", "or", "xor", "xnor", "majority", "parity",
              "temporal_order", "last_event", "running_parity", "toggle_reset")


# --------------------------------------------------------------------------- #
# rules                                                                        #
# --------------------------------------------------------------------------- #

def make_rule(rng: np.random.Generator) -> tuple:
    op = str(rng.choice(OPERATIONS))
    arity = {"and": 2, "or": 2, "xor": 2, "xnor": 2, "majority": 3, "parity": 3,
             "temporal_order": 2, "last_event": 2, "running_parity": 1,
             "toggle_reset": 2}[op]
    channels = tuple(sorted(rng.choice(N_CHANNELS, size=arity, replace=False).tolist()))
    # Point operations read a bounded time offset; the 3-step suffix makes offsets 0..2
    # reachable by the polynomial basis, which is what ADRN-2B's basis exposed too.
    offset = int(rng.integers(0, 3)) if op in ("and", "or", "xor", "xnor",
                                               "majority", "parity") else 0
    return (op, channels, offset)


def evaluate(rule: tuple, x: np.ndarray) -> np.ndarray:
    """Vectorised over a batch of (batch, STEPS, N_CHANNELS) binary sequences."""

    op, channels, offset = rule
    step = STEPS - 1 - offset
    if op in ("and", "or", "xor", "xnor", "majority", "parity"):
        bits = x[:, step, list(channels)].astype(np.int64)
        if op == "and":
            return bits.min(axis=1)
        if op == "or":
            return bits.max(axis=1)
        if op == "xor":
            return bits.sum(axis=1) % 2
        if op == "xnor":
            return 1 - bits.sum(axis=1) % 2
        if op == "majority":
            return (bits.sum(axis=1) * 2 > bits.shape[1]).astype(np.int64)
        return bits.sum(axis=1) % 2                       # parity over 3 channels
    if op == "running_parity":
        return x[:, :, channels[0]].sum(axis=1).astype(np.int64) % 2
    present = x > 0.5
    if op in ("temporal_order", "last_event"):
        a, b = channels
        idx = np.arange(STEPS)[None, :]
        big = STEPS + 1
        first_a = np.where(present[:, :, a], idx, big).min(axis=1)
        first_b = np.where(present[:, :, b], idx, big).min(axis=1)
        last_a = np.where(present[:, :, a], idx, -1).max(axis=1)
        last_b = np.where(present[:, :, b], idx, -1).max(axis=1)
        if op == "temporal_order":
            return ((first_a < first_b) & (first_a < big)).astype(np.int64)
        return ((last_a > last_b) & (last_a >= 0)).astype(np.int64)
    toggle, reset = channels                              # toggle_reset
    state = np.zeros(len(x), np.int64)
    for t in range(STEPS):
        state = np.where(x[:, t, toggle] > 0.5, 1 - state, state)
        state = np.where(x[:, t, reset] > 0.5, 0, state)
    return state


# --------------------------------------------------------------------------- #
# representations                                                              #
# --------------------------------------------------------------------------- #

def polynomial_basis(x: torch.Tensor) -> torch.Tensor:
    """3-step signed suffix (48) + current-step signed pair products (120) = 168."""

    signed = 2.0 * x - 1.0
    rows, cols = torch.triu_indices(N_CHANNELS, N_CHANNELS, offset=1)
    pairs = signed[:, -1, rows] * signed[:, -1, cols]
    return torch.cat((signed[:, -3:].flatten(1), pairs), dim=1)


def digital_state(x: torch.Tensor) -> torch.Tensor:
    """parity(16) + first(16) + last(16) + ordered toggle/reset(240) = 288."""

    parity = torch.prod(1.0 - 2.0 * x, dim=1)
    present = x > 0.5
    has = torch.any(present, dim=1)
    first = torch.argmax(present.to(torch.int64), dim=1)
    last = x.shape[1] - 1 - torch.argmax(
        torch.flip(present, dims=(1,)).to(torch.int64), dim=1)
    denom = max(x.shape[1] - 1, 1)
    scale = lambda v: torch.where(has, 2.0 * v.to(x.dtype) / denom - 1.0,
                                  torch.full_like(v, -1.0, dtype=x.dtype))
    state = torch.zeros((x.shape[0], N_CHANNELS, N_CHANNELS), dtype=x.dtype)
    for t in range(x.shape[1]):
        tog, res = x[:, t, :, None], x[:, t, None, :]
        state = state + tog - 2.0 * state * tog
        state = state * (1.0 - res)                       # reset wins ties
    off = ~torch.eye(N_CHANNELS, dtype=torch.bool)
    return torch.cat((parity, scale(first), scale(last),
                      2.0 * state[:, off] - 1.0), dim=1)


class GRUEncoder(nn.Module):
    def __init__(self, size: int = 32) -> None:
        super().__init__()
        self.project = nn.Linear(N_CHANNELS, size)
        self.gru = nn.GRU(size, 64, batch_first=True)
        self.norm = nn.LayerNorm(64)
        self.out = nn.Linear(64, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(torch.nn.functional.gelu(self.project(x)))
        return self.out(self.norm(hidden[-1]))


class AttentionEncoder(nn.Module):
    def __init__(self, size: int = 32) -> None:
        super().__init__()
        self.project = nn.Linear(N_CHANNELS, size)
        self.token = nn.Parameter(torch.zeros(1, 1, size))
        nn.init.normal_(self.token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=size, nhead=4, dim_feedforward=128, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(size)
        position = torch.arange(STEPS + 1, dtype=torch.float32)[:, None]
        index = torch.arange(size, dtype=torch.float32)[None, :]
        angle = position / torch.pow(10000.0, 2 * torch.div(index, 2, rounding_mode="floor") / size)
        pe = torch.zeros(STEPS + 1, size)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(angle[:, 0::2]), torch.cos(angle[:, 1::2])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = torch.cat((self.token.expand(len(x), -1, -1), self.project(x)), dim=1)
        return self.norm(self.encoder(embedded + self.pe[None, : embedded.shape[1]])[:, 0])


def pretrain(encoder: nn.Module, rules: list, rng: np.random.Generator, *,
             examples: int, epochs: int, report) -> None:
    """Multitask offline training on TRAINING rules only, then frozen."""

    x = rng.integers(0, 2, size=(examples, STEPS, N_CHANNELS)).astype(np.float32)
    targets = np.stack([evaluate(r, x) for r in rules], axis=1)
    xt = torch.as_tensor(x)
    yt = torch.as_tensor(targets, dtype=torch.float32)
    head = nn.Linear(32, len(rules))
    optimiser = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    encoder.train()
    for epoch in range(epochs):
        order = torch.randperm(len(xt))
        total = 0.0
        for start in range(0, len(xt), 128):
            batch = order[start:start + 128]
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(head(encoder(xt[batch])), yt[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimiser.step()
            total += float(loss.detach()) * len(batch)
        if epoch == epochs - 1:
            report(f"      final multitask loss {total/len(xt):.4f}")
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)


# --------------------------------------------------------------------------- #
# online adapter: one per context (oracle banking), local delta rule           #
# --------------------------------------------------------------------------- #

def acquisition(features_train: np.ndarray, labels: np.ndarray,
                features_probe: np.ndarray, probe_labels: np.ndarray,
                *, rate: float, checkpoints, decay: float = 0.5) -> dict[int, float]:
    """Online delta rule with a 1/sqrt(t) step, evaluated at each checkpoint.

    The first draft used a fixed step and produced a NON-MONOTONIC curve -- A(128)
    0.5591, A(256) 0.4983, A(512) 0.5593 -- which is the signature of an unstable
    stochastic update rather than an acquisition curve. A decaying step is the standard
    fix and makes the single-pass estimate converge instead of oscillating; the decay
    exponent is fixed at 0.5 for every arm so it cannot favour one.
    """

    weights = np.zeros(features_train.shape[1] + 1)
    curve: dict[int, float] = {}
    scale = np.sqrt((features_train ** 2).sum(1)).mean() + 1e-9
    probe_design = np.concatenate(
        [features_probe / scale, np.ones((len(features_probe), 1))], axis=1)
    for index in range(max(checkpoints)):
        row = np.concatenate([features_train[index] / scale, [1.0]])
        target = 2.0 * labels[index] - 1.0
        step_size = rate / (1.0 + index) ** decay
        weights += step_size * (target - np.tanh(row @ weights)) * row
        step = index + 1
        if step in checkpoints:
            prediction = (probe_design @ weights > 0).astype(np.int64)
            scores = []
            for value in (0, 1):
                mask = probe_labels == value
                scores.append(float((prediction[mask] == value).mean()) if mask.any() else np.nan)
            curve[step] = float(np.nanmean(scores))
    return curve


def tune_rate(encoderless_rules: list, rng: np.random.Generator, *, probe: int,
              candidates, report) -> float:
    """Pick the adapter step on VALIDATION rules, using the fixed polynomial basis.

    ADRN-2B tuned its adapter learning rate on a validation split and froze it; a
    hardcoded rate is the reason the first draft sat at exactly 0.5000 for A(8) on every
    arm -- the bias term dominated and the predictor was constant. The rate is chosen
    once, on rules disjoint from both training and test, and shared by every arm so it
    cannot advantage one.
    """

    best, best_score = candidates[0], -1.0
    for rate in candidates:
        scores = []
        for rule in encoderless_rules:
            x = rng.integers(0, 2, size=(max(CHECKPOINTS), STEPS, N_CHANNELS)).astype(np.float32)
            xp = rng.integers(0, 2, size=(probe, STEPS, N_CHANNELS)).astype(np.float32)
            y, yp = evaluate(rule, x), evaluate(rule, xp)
            if len(np.unique(yp)) < 2:
                continue
            with torch.no_grad():
                ft = polynomial_basis(torch.as_tensor(x)).numpy()
                fp = polynomial_basis(torch.as_tensor(xp)).numpy()
            curve = acquisition(ft, y, fp, yp, rate=rate, checkpoints=CHECKPOINTS)
            scores.append(np.mean(list(curve.values())))
        mean = float(np.mean(scores)) if scores else 0.0
        report(f"      rate {rate:5.3f} -> mean curve {mean:.4f}")
        if mean > best_score:
            best, best_score = rate, mean
    report(f"      selected adapter rate {best} on validation rules")
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train-rules", type=int, default=64)
    parser.add_argument("--test-rules", type=int, default=24)
    parser.add_argument("--validation-rules", type=int, default=12)
    parser.add_argument("--probe", type=int, default=4096)
    parser.add_argument("--pretrain-examples", type=int, default=6144)
    parser.add_argument("--pretrain-epochs", type=int, default=12)
    parser.add_argument("--rate", type=float, default=0.05)
    args = parser.parse_args(argv)

    torch.set_num_threads(2)
    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    report("=" * 96)
    report("ACQUISITION CURVE TO 512 LABELS -- reimplementation, not the ADRN-2B harness")
    report("=" * 96)
    report("  PREDECLARED: ordering holds (polynomial-based ahead of transformer, GRU")
    report("  adding nothing over the fixed digital basis) and the curve keeps rising")
    report("  through 256 -- the Bayesian oracle reached 1.0000 on this family, so there")
    report("  is headroom well above the ~0.63 reached at 128.")

    arms = ("polynomial", "digital(+poly)", "workspace(GRU+digital+poly)",
            "transformer(+poly)")
    collected: dict[str, list[dict[int, float]]] = {a: [] for a in arms}

    for seed in range(args.seeds):
        rng = np.random.default_rng(4_000 + seed)
        seen: set[tuple] = set()
        rules: list[tuple] = []
        total = args.train_rules + args.validation_rules + args.test_rules
        while len(rules) < total:
            rule = make_rule(rng)
            if rule not in seen:
                seen.add(rule)
                rules.append(rule)
        train_rules = rules[:args.train_rules]
        validation_rules = rules[args.train_rules:args.train_rules + args.validation_rules]
        test_rules = rules[args.train_rules + args.validation_rules:]
        assert not (set(train_rules) & set(test_rules))
        assert not (set(validation_rules) & set(test_rules))
        assert not (set(train_rules) & set(validation_rules))
        report(f"\n  seed {seed}: {len(train_rules)} train / {len(validation_rules)} "
               f"validation / {len(test_rules)} held-out rules, all disjoint")
        rate = tune_rate(validation_rules, rng, probe=args.probe,
                         candidates=(0.4, 1.0, 2.5, 6.0, 15.0, 40.0), report=report)

        gru, attention = GRUEncoder(), AttentionEncoder()
        report("      pretraining GRU"); pretrain(gru, train_rules, rng,
                                                  examples=args.pretrain_examples,
                                                  epochs=args.pretrain_epochs, report=report)
        report("      pretraining attention"); pretrain(attention, train_rules, rng,
                                                        examples=args.pretrain_examples,
                                                        epochs=args.pretrain_epochs, report=report)

        per_arm: dict[str, list[dict[int, float]]] = {a: [] for a in arms}
        for rule in test_rules:
            x_train = rng.integers(0, 2, size=(max(CHECKPOINTS), STEPS, N_CHANNELS)).astype(np.float32)
            x_probe = rng.integers(0, 2, size=(args.probe, STEPS, N_CHANNELS)).astype(np.float32)
            y_train, y_probe = evaluate(rule, x_train), evaluate(rule, x_probe)
            if len(np.unique(y_probe)) < 2:
                continue
            tt, tp = torch.as_tensor(x_train), torch.as_tensor(x_probe)
            with torch.no_grad():
                poly_t, poly_p = polynomial_basis(tt).numpy(), polynomial_basis(tp).numpy()
                dig_t, dig_p = digital_state(tt).numpy(), digital_state(tp).numpy()
                gru_t, gru_p = gru(tt).numpy(), gru(tp).numpy()
                att_t, att_p = attention(tt).numpy(), attention(tp).numpy()
            views = {
                "polynomial": (poly_t, poly_p),
                "digital(+poly)": (np.hstack([poly_t, dig_t]), np.hstack([poly_p, dig_p])),
                "workspace(GRU+digital+poly)": (np.hstack([poly_t, dig_t, gru_t]),
                                                np.hstack([poly_p, dig_p, gru_p])),
                "transformer(+poly)": (np.hstack([poly_t, att_t]), np.hstack([poly_p, att_p])),
            }
            for arm, (ftr, fpr) in views.items():
                per_arm[arm].append(acquisition(ftr, y_train, fpr, y_probe,
                                                rate=rate, checkpoints=CHECKPOINTS))
        for arm in arms:
            collected[arm].append({k: float(np.mean([c[k] for c in per_arm[arm]]))
                                   for k in CHECKPOINTS})

    report(f"\n  === held-out-rule acquisition, mean over {args.seeds} seeds "
           f"(chance 0.5000) ===")
    header = " ".join(f"A({k}){'':<2s}" for k in CHECKPOINTS)
    report(f"  {'arm':30s} {header}")
    summary: dict[str, object] = {}
    for arm in arms:
        curve = {k: float(np.mean([c[k] for c in collected[arm]])) for k in CHECKPOINTS}
        sd = {k: float(np.std([c[k] for c in collected[arm]], ddof=1)) for k in CHECKPOINTS}
        summary[arm] = {"mean": curve, "sd": sd,
                        "per_seed": [{str(k): c[k] for k in CHECKPOINTS} for c in collected[arm]]}
        report(f"  {arm:30s} " + " ".join(f"{curve[k]:.4f}" for k in CHECKPOINTS))

    report(f"\n  per-doubling increments (is the curve still accelerating at 128?)")
    for arm in arms:
        curve = summary[arm]["mean"]
        steps = [curve[b] - curve[a] for a, b in zip(CHECKPOINTS, CHECKPOINTS[1:])]
        report(f"  {arm:30s} " + " ".join(f"{s:+.4f}" for s in steps))

    report(f"\n  paired per-seed contrasts at A(256), MDE = 3*sd/sqrt({args.seeds})")
    for left, right in (("digital(+poly)", "polynomial"),
                        ("workspace(GRU+digital+poly)", "digital(+poly)"),
                        ("polynomial", "transformer(+poly)")):
        gap = np.array([a[256] - b[256] for a, b in
                        zip(collected[left], collected[right])])
        sd = float(gap.std(ddof=1)) if len(gap) > 1 else 0.0
        mde = 3 * sd / np.sqrt(len(gap))
        summary[f"{left} - {right} @256"] = {
            "mean": float(gap.mean()), "mde": float(mde),
            "resolved": bool(abs(gap.mean()) > mde)}
        report(f"  {left[:26]:28s} - {right[:22]:24s} {gap.mean():+.4f}  MDE {mde:.4f}  "
               f"{'RESOLVED' if abs(gap.mean()) > mde else 'within noise'}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"note": "reimplementation of the ADRN-2B comparison, not the original harness; "
                 "absolute values are not comparable, ordering and curve shape are",
         "checkpoints": list(CHECKPOINTS), "seeds": args.seeds,
         "train_rules": args.train_rules, "test_rules": args.test_rules,
         "summary": summary, "log": log}, indent=2, sort_keys=True))
    report(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
