"""ADRN-3 direction 3: online conjunction construction on a provable floor.

The capability gap this attacks
-------------------------------
The ADRN-2B gate audit established that the deployed adapter is a linear threshold
on a fixed 488-d basis (`stable_logits` is a hard zero vector), and that basis
contains no >=3-way product of arbitrary channels.  The `parity3` and `parity5`
rules are therefore exactly orthogonal to all 456 fixed coordinates: an
out-of-sample linear fit on 20,000 labels reaches 0.495-0.501 balanced accuracy
against 1.000 for `xor`.  A GRU trained exclusively on parity tasks still ceilings
at 0.505.  Nothing in ADRN can construct a new nonlinear feature.

That makes parity3/parity5 an unusually good test bed: **the floor is exactly
known**.  Any arm that gets meaningfully above 0.50 has done something real, and
no ceiling artifact can manufacture it.

The mechanism
-------------
Clustered plasticity, rather than the gradient-trained dendrites of ADRN-1 (which
ablation showed were worth +0.0557 and +0.2315 to REMOVE).  The rule is local:

  * maintain an eligibility score for each candidate pair/triple of coordinates,
    accumulated as the product of their signed activations times the current
    prediction error -- a coincidence-with-error detector, not a gradient;
  * when a candidate's score crosses a threshold, GROW it into the feature set as
    an explicit product term with its own adapter weight;
  * decay every grown feature's contribution score; PRUNE the ones that stop
    paying, freeing capacity.

Growth is over products of the raw signed channels, so a grown feature can express
a conjunction the fixed basis cannot.

Controls, all of which must be run for the result to mean anything
-------------------------------------------------------------------
  fixed          the shipped arm: 488-d basis + delta rule. Known floor ~0.50.
  random_grow    the SAME NUMBER of conjunctions, chosen uniformly at random
                 instead of by the local rule. This is the control that decides
                 whether the learning rule matters or merely the extra capacity.
                 If random matches learned, the rule is worthless.
  oracle_grow    the true conjunction handed over. The ceiling: what the adapter
                 could do if feature construction were solved perfectly.
  learned_grow   the mechanism under test.

Reported on `xor` as well as `parity3`/`parity5`: xor is already in the fixed basis
as a pair product, so every arm should be near 1.000 there. An arm that fails xor
is broken, and an arm whose parity gain is not accompanied by xor competence is
suspect.

Usage
-----
    python adrn3_conjunctions.py --package /path/to/adrn2b_package --output out/conj.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

LABEL_CHECKPOINTS = (8, 32, 128, 512)
N_CHANNELS = 16


def _load(package: Path):
    package = package.resolve()
    if not (package / "adrn2b_models.py").is_file():
        raise SystemExit(f"{package} does not contain adrn2b_models.py")
    sys.path.insert(0, str(package.parent))
    models = __import__(f"{package.name}.adrn2b_models", fromlist=["*"])
    tasks = __import__(f"{package.name}.adrn2b_tasks", fromlist=["*"])
    return models, tasks


# --------------------------------------------------------------------------- #
# task: parity over k channels of the current step, plus xor as a sanity rule   #
# --------------------------------------------------------------------------- #

def make_task(kind: str, rng: np.random.Generator, *, steps: int, n: int):
    """Return (sequences, labels, the true channel set).

    `parity3` / `parity5` are XOR over 3 or 5 current-step channels -- exactly the
    families the audit showed are orthogonal to the fixed basis.  `xor` is the
    2-channel case, which the fixed basis already contains as a pair product.
    """

    arity = {"xor": 2, "parity3": 3, "parity5": 5}[kind]
    channels = tuple(sorted(rng.choice(N_CHANNELS, size=arity, replace=False).tolist()))
    sequences = rng.integers(0, 2, size=(n, steps, N_CHANNELS)).astype(np.float32)
    labels = sequences[:, -1, list(channels)].sum(axis=1).astype(np.int64) % 2
    return sequences, labels, channels


# --------------------------------------------------------------------------- #
# the online learner                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class Grown:
    channels: tuple[int, ...]
    weight: float = 0.0
    contribution: float = 0.0
    age: int = 0


@dataclass
class ConjunctionLearner:
    """Fixed basis + a growing set of explicit conjunctions, all learned online.

    The base weights and every grown feature's weight use the same delta rule, so
    a grown feature has no optimiser advantage -- only a representational one.
    """

    basis_size: int
    mode: str                       # fixed | learned_grow | random_grow | oracle_grow
    max_grown: int = 24
    max_arity: int = 5
    learning_rate: float = 0.10
    grow_threshold: float = 4.0      # standard errors, see _maybe_grow
    min_observations: int = 8
    eligibility_decay: float = 0.97
    prune_after: int = 64
    prune_floor: float = 0.02
    seed: int = 0
    oracle_channels: tuple[int, ...] | None = None

    weights: np.ndarray = field(init=False)
    bias: float = field(init=False, default=0.0)
    grown: list[Grown] = field(init=False, default_factory=list)
    eligibility: dict[tuple[int, ...], float] = field(init=False, default_factory=dict)
    n_updates: int = field(init=False, default=0)
    n_grown_total: int = field(init=False, default=0)
    n_pruned: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.weights = np.zeros(self.basis_size, dtype=np.float64)
        self._rng = np.random.default_rng(self.seed + 4242)
        if self.mode == "learned_grow":
            keys: list[tuple[int, ...]] = []
            for arity in range(3, self.max_arity + 1, 2):   # 3 and 5: what the basis lacks
                keys.extend(combinations(range(N_CHANNELS), arity))
            # Pad each key to max_arity by repeating its first channel: s*s == 1 for
            # a +/-1 value, so the padded product equals the true product and one
            # rectangular gather covers both arities.
            self._candidate_keys = keys
            padded = [key + (key[0],) * (self.max_arity - len(key)) for key in keys]
            self._candidate_index = np.asarray(padded, dtype=np.int64)
            self._candidate_mean = np.zeros(len(keys), dtype=np.float64)
            self._candidate_count = 0
        if self.mode == "oracle_grow":
            if not self.oracle_channels:
                raise ValueError("oracle_grow needs the true channel set")
            self.grown = [Grown(tuple(self.oracle_channels))]
            self.n_grown_total = 1
        elif self.mode == "random_grow":
            self._seed_random_conjunctions()
        elif self.mode not in ("fixed", "learned_grow"):
            raise ValueError(f"unknown mode {self.mode}")

    def _seed_random_conjunctions(self) -> None:
        """Same budget as the learned arm, chosen without looking at any label."""

        seen: set[tuple[int, ...]] = set()
        while len(seen) < self.max_grown:
            arity = int(self._rng.integers(2, self.max_arity + 1))
            pick = tuple(sorted(self._rng.choice(N_CHANNELS, size=arity, replace=False).tolist()))
            seen.add(pick)
        self.grown = [Grown(channels) for channels in sorted(seen)]
        self.n_grown_total = len(self.grown)

    @staticmethod
    def _product(signed_current: np.ndarray, channels: tuple[int, ...]) -> float:
        return float(np.prod(signed_current[list(channels)]))

    def _grown_vector(self, signed_current: np.ndarray) -> np.ndarray:
        if not self.grown:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(
            [self._product(signed_current, item.channels) for item in self.grown],
            dtype=np.float64,
        )

    def score(self, basis: np.ndarray, signed_current: np.ndarray) -> float:
        total = float(basis @ self.weights) + self.bias
        for item, value in zip(self.grown, self._grown_vector(signed_current), strict=True):
            total += item.weight * value
        return total

    def predict(self, basis: np.ndarray, signed_current: np.ndarray) -> int:
        return int(self.score(basis, signed_current) > 0.0)

    def update(self, basis: np.ndarray, signed_current: np.ndarray, label: int) -> None:
        target = 2.0 * label - 1.0
        prediction = np.tanh(self.score(basis, signed_current))
        error = target - prediction
        self.weights += self.learning_rate * error * basis
        self.bias += self.learning_rate * error
        for item, value in zip(self.grown, self._grown_vector(signed_current), strict=True):
            before = item.weight
            item.weight += self.learning_rate * error * value
            item.age += 1
            item.contribution = 0.98 * item.contribution + 0.02 * abs(item.weight * value)
            del before
        self.n_updates += 1

        if self.mode == "learned_grow":
            self._accumulate_eligibility(signed_current, error)
            self._maybe_grow()
            self._maybe_prune()

    def _accumulate_eligibility(self, signed_current: np.ndarray, error: float) -> None:
        """Coincidence-with-error, local and gradient-free.

        Every candidate channel set keeps a running mean of

            (product of its signed activations) x (prediction error)

        A set whose product is systematically aligned with what the linear part is
        getting wrong drifts away from zero; an irrelevant set averages to zero.

        The first version of this decayed a sparse sampled dictionary, which made
        the mechanism arithmetically unreachable: a triple was touched 4.3% of
        steps against a 0.97-per-step decay, so eligibility never came within two
        orders of magnitude of the growth threshold and `learned_grow` returned
        byte-identical predictions to `fixed`.  Candidates are now enumerated
        exhaustively (560 triples + 4368 quintuples, one vectorised product each)
        and scored by a t-like statistic, so growth depends on evidence rather
        than on how often the sampler happened to visit a set.
        """

        products = np.prod(signed_current[self._candidate_index], axis=1)
        values = products * error
        self._candidate_count += 1
        self._candidate_mean += (values - self._candidate_mean) / self._candidate_count

    def _maybe_grow(self) -> None:
        if len(self.grown) >= self.max_grown:
            return
        if self._candidate_count < self.min_observations:
            return
        # |mean| * sqrt(n): how many standard errors the alignment sits from zero
        # for a bounded +/-1 product. Scale-free, so one threshold works at every
        # label budget.
        score = np.abs(self._candidate_mean) * np.sqrt(self._candidate_count)
        active = {item.channels for item in self.grown}
        order = np.argsort(-score)
        for index in order[:8]:
            key = self._candidate_keys[int(index)]
            if key in active:
                continue
            if float(score[index]) < self.grow_threshold:
                return
            self.grown.append(Grown(key))
            self.n_grown_total += 1
            self._candidate_mean[index] = 0.0
            return

    def _maybe_prune(self) -> None:
        if self.mode != "learned_grow":
            return
        keep = []
        for item in self.grown:
            if item.age >= self.prune_after and item.contribution < self.prune_floor:
                self.n_pruned += 1
                continue
            keep.append(item)
        self.grown = keep


# --------------------------------------------------------------------------- #
# evaluation                                                                   #
# --------------------------------------------------------------------------- #

def balanced_accuracy(predictions: np.ndarray, truth: np.ndarray) -> float:
    scores = []
    for value in (0, 1):
        mask = truth == value
        if not mask.any():
            return float("nan")
        scores.append(float((predictions[mask] == value).mean()))
    return float(np.mean(scores))


def run_arm(models, kind: str, mode: str, *, seed: int, steps: int,
            train_n: int, test_n: int, max_grown: int,
            checkpoints: tuple[int, ...] = LABEL_CHECKPOINTS) -> dict:
    import torch

    rng = np.random.default_rng(seed)
    train_x, train_y, channels = make_task(kind, rng, steps=steps, n=train_n)
    test_x, test_y, _ = make_task(kind, np.random.default_rng(seed + 500_000),
                                  steps=steps, n=test_n)
    # Reuse the true channel set for the held-out draw so the rule is the same rule.
    test_y = test_x[:, -1, list(channels)].sum(axis=1).astype(np.int64) % 2

    with torch.no_grad():
        train_basis = models.expanded_polynomial_tensor(
            torch.as_tensor(train_x), 488).numpy().astype(np.float64)
        test_basis = models.expanded_polynomial_tensor(
            torch.as_tensor(test_x), 488).numpy().astype(np.float64)
    train_signed = 2.0 * train_x[:, -1, :].astype(np.float64) - 1.0
    test_signed = 2.0 * test_x[:, -1, :].astype(np.float64) - 1.0

    learner = ConjunctionLearner(
        basis_size=488, mode=mode, max_grown=max_grown, seed=seed,
        oracle_channels=channels if mode == "oracle_grow" else None,
    )
    curve: dict[str, float] = {}
    for index in range(train_n):
        learner.update(train_basis[index], train_signed[index], int(train_y[index]))
        if (index + 1) in checkpoints:
            predictions = np.asarray([
                learner.predict(test_basis[j], test_signed[j]) for j in range(test_n)
            ])
            curve[str(index + 1)] = balanced_accuracy(predictions, test_y)
    found = any(item.channels == channels for item in learner.grown)
    return {
        "curve": curve,
        "true_channels": list(channels),
        "n_grown_total": learner.n_grown_total,
        "n_grown_final": len(learner.grown),
        "n_pruned": learner.n_pruned,
        "found_true_conjunction": bool(found),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--train", type=int, default=512)
    parser.add_argument("--test", type=int, default=2048)
    parser.add_argument("--max-grown", type=int, default=24)
    args = parser.parse_args(argv)

    models, _ = _load(args.package)
    modes = ("fixed", "random_grow", "learned_grow", "oracle_grow")
    checkpoints = tuple(k for k in LABEL_CHECKPOINTS if k <= args.train)
    if not checkpoints:
        raise SystemExit(f"--train {args.train} is below the first checkpoint "
                         f"{LABEL_CHECKPOINTS[0]}")
    payload: dict[str, object] = {
        "seeds": args.seeds, "train_labels": args.train, "test_examples": args.test,
        "checkpoints": list(checkpoints), "max_grown": args.max_grown,
        "chance": 0.5, "results": {},
    }

    for kind in ("xor", "parity3", "parity5"):
        print(f"\n=== {kind} ===  (chance 0.5000, {args.seeds} seeds, "
              f"balanced accuracy on {args.test} held-out draws)")
        print(f"{'arm':16s} " + " ".join(f"A({k}){'':<7s}" for k in checkpoints)
              + "  found true conj.")
        payload["results"][kind] = {}
        for mode in modes:
            runs = [run_arm(models, kind, mode, seed=1000 + s, steps=args.steps,
                            train_n=args.train, test_n=args.test,
                            max_grown=args.max_grown, checkpoints=checkpoints)
                    for s in range(args.seeds)]
            summary = {}
            for checkpoint in checkpoints:
                values = np.array([r["curve"][str(checkpoint)] for r in runs])
                summary[str(checkpoint)] = {
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
            found = float(np.mean([r["found_true_conjunction"] for r in runs]))
            summary["found_true_conjunction_rate"] = found
            summary["mean_grown_final"] = float(np.mean([r["n_grown_final"] for r in runs]))
            summary["mean_pruned"] = float(np.mean([r["n_pruned"] for r in runs]))
            payload["results"][kind][mode] = summary
            cells = " ".join(
                f"{summary[str(k)]['mean']:.4f}+/-{summary[str(k)]['sd']:.3f}"
                for k in checkpoints
            )
            print(f"{mode:16s} {cells}   {found:.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwrote {args.output}")
    print("\nRead 'fixed' on parity3/parity5 first: the audit puts it at 0.495-0.501, "
          "so it is a\nverified floor. Then read random_grow -- if it matches "
          "learned_grow, the local rule is\nworth nothing and only the extra capacity "
          "mattered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
