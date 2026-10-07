"""ADRN-3 direction 1 and 2: isolated context-routing benchmark.

Why isolate routing at all
--------------------------
The largest measured number in the ADRN line is the gap between being handed the
context key and inferring it: ADRN-2A scored 0.8681 retention with an oracle key
against 0.5069 inferring it, and ADRN-2B's learned router recovers only 0.3824 of
the same gap.  Everything downstream of routing works once the index is supplied.
So routing is where the value is, and it is worth measuring on its own before
paying 23 minutes per end-to-end run.

This module measures ONLY "which context am I in", using ADRN-2B's own cue
generator (`make_context_signature_bank`), so the difficulty is the shipped
difficulty and not something re-tuned to flatter a new mechanism.

Direction 1 -- evidence accumulation
------------------------------------
The shipped router decides from ONE noisy 12-D draw per probe.  Its transition
prior is conditioned on a *point estimate* of the active slot (`_transition_log_prior`
in adrn2b_router.py reads `self._active_slot`), and `predict` is pure, so nothing
carries across probes even though a return window contains 256 probes that all
share a context.  The audit measured the cue at ~80% decodable per draw, which
caps single-shot decoding near 0.80 -- and Gate A asks for 0.80.

The principled fix is one change: carry a *distribution* over contexts and
propagate it through the same stickiness matrix, i.e. an HMM forward filter

    P_t(c)  proportional to  p(cue_t | c) * sum_c' T(c | c') P_{t-1}(c')

This is the multi-hypothesis generalisation of the drift-diffusion accumulation
that cortex is best characterised as doing, and it is a strict superset of the
shipped rule (a one-hot P_{t-1} recovers the shipped transition prior exactly).
It touches routing state only -- no adapter, no weight, nothing that could let a
probe train the model.

Direction 2 -- dentate-gyrus style pattern separation
-----------------------------------------------------
Expansive random projection to a wide layer, k-winner-take-all sparsification,
then nearest-prototype in the sparse code, optionally with a CA3-style completion
sweep.  Roughly the DG->CA3 motif: decorrelate similar inputs by expanding into a
sparse high-dimensional code, then complete from a partial cue.

PREDECLARED EXPECTATION, recorded before running: direction 2 should NOT beat a
correctly specified Gaussian decoder.  The cue noise here is isotropic Gaussian by
construction (`ContextSignatureBank.sample` adds `rng.normal(0, noise_std)`), so
the likelihood decoder is Bayes-optimal and any lossy recoding can only lose
information.  The informative case is the LOW-SAMPLE regime, where the router must
estimate its own prototypes and variances from a handful of labelled visits -- there
a robust code could beat a badly estimated Gaussian.  Both regimes are measured.
Direction 2 failing in the known-prototype regime is the expected result and is
reported as a result, not as a bug.

Controls
--------
Every accumulating arm is paired with a SHUFFLED-CUE control that accumulates cues
drawn from a uniformly random context instead of the true one.  Accumulation
mechanically reduces variance, so an arm that improves under shuffled cues is
improving for a reason unrelated to context evidence.  `chance` and `oracle` bound
the scale.

Usage
-----
    python adrn3_routing.py --package /path/to/adrn2b_package --output out/routing.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PHASE_PATTERN = ("A", "B", "C", "A", "D", "B")  # matches adrn2b_tasks.PHASE_PATTERN
POSITION_CHECKPOINTS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _load(package: Path):
    package = package.resolve()
    if not (package / "adrn2b_tasks.py").is_file():
        raise SystemExit(f"{package} does not contain adrn2b_tasks.py")
    sys.path.insert(0, str(package.parent))
    return __import__(f"{package.name}.adrn2b_tasks", fromlist=["*"])


@dataclass(frozen=True)
class Stream:
    """A cue stream plus the context truth the evaluator keeps to itself."""

    cues: np.ndarray            # (steps, 12)
    truth: np.ndarray           # (steps,) index into prototypes
    phase_of: np.ndarray        # (steps,) which phase each step belongs to
    position: np.ndarray        # (steps,) 0-based position within its phase
    supervised: np.ndarray      # (steps,) bool: is the context revealed here
    prototypes: np.ndarray      # (n_contexts, 12) evaluator-only truth
    noise_std: float


def make_stream(tasks, *, difficulty: str, probes_per_phase: int,
                supervised_prefix: int, seed: int) -> Stream:
    """Build a phase-structured cue stream.

    `supervised_prefix` steps at the start of each phase reveal the context, as
    labelled feedback does in the real harness.  Everything after is unsupervised,
    and the final phases repeat earlier contexts, so the tail of each repeat phase
    is the feedback-free return window Gate A is read from.
    """

    bank = tasks.make_context_signature_bank(
        n_contexts=4, difficulty=difficulty, seed=seed
    )
    rng = np.random.default_rng(seed + 7717)
    slots = sorted(set(PHASE_PATTERN))
    slot_to_index = {slot: index for index, slot in enumerate(slots)}

    cues, truth, phase_of, position, supervised = [], [], [], [], []
    for phase_index, slot in enumerate(PHASE_PATTERN):
        index = slot_to_index[slot]
        context_id = bank.context_ids[index]
        for step in range(probes_per_phase):
            cues.append(np.asarray(bank.sample(context_id, rng), dtype=np.float64))
            truth.append(index)
            phase_of.append(phase_index)
            position.append(step)
            supervised.append(step < supervised_prefix)
    return Stream(
        cues=np.stack(cues), truth=np.asarray(truth), phase_of=np.asarray(phase_of),
        position=np.asarray(position), supervised=np.asarray(supervised),
        prototypes=np.asarray(bank.prototypes, dtype=np.float64),
        noise_std=float(bank.noise_std),
    )


# --------------------------------------------------------------------------- #
# prototype estimators                                                         #
# --------------------------------------------------------------------------- #

class PrototypeStore:
    """Running per-context mean and variance, updated only on supervised steps.

    `known=True` installs the evaluator's true prototypes, which is the ceiling on
    what any decoder could know about the cue distribution.  `known=False` is the
    realistic regime: the router sees a context's prototype only through the few
    supervised steps it has had.
    """

    def __init__(self, n_contexts: int, dim: int, *, floor: float,
                 known_prototypes: np.ndarray | None = None,
                 known_variance: float | None = None) -> None:
        self.known = known_prototypes is not None
        self.floor = float(floor)
        if self.known:
            self.mean = np.array(known_prototypes, dtype=np.float64)
            self.var = np.full((n_contexts, dim), float(known_variance), dtype=np.float64)
            self.count = np.full(n_contexts, 1_000_000, dtype=np.int64)
        else:
            self.mean = np.zeros((n_contexts, dim), dtype=np.float64)
            self.m2 = np.zeros((n_contexts, dim), dtype=np.float64)
            self.count = np.zeros(n_contexts, dtype=np.int64)
            self.var = np.full((n_contexts, dim), self.floor, dtype=np.float64)

    def observe(self, index: int, cue: np.ndarray) -> None:
        if self.known:
            return
        self.count[index] += 1
        delta = cue - self.mean[index]
        self.mean[index] += delta / self.count[index]
        self.m2[index] += delta * (cue - self.mean[index])
        if self.count[index] > 1:
            self.var[index] = np.maximum(
                self.m2[index] / (self.count[index] - 1), self.floor
            )

    @property
    def seen(self) -> np.ndarray:
        return self.count > 0

    def log_likelihood(self, cue: np.ndarray) -> np.ndarray:
        """Per-context Gaussian log-likelihood, summed over the 12 dimensions."""

        difference = cue[None, :] - self.mean
        terms = difference**2 / self.var + np.log(2.0 * np.pi * self.var)
        return -0.5 * terms.sum(axis=1)


# --------------------------------------------------------------------------- #
# direction 2: DG-style expansive sparse recoding                              #
# --------------------------------------------------------------------------- #

class PatternSeparator:
    """Expansive random projection + k-winner-take-all, with optional completion.

    The projection is fixed and random (granule cells receive a fixed, sparse,
    effectively random sample of entorhinal input); kWTA stands in for the strong
    feedback inhibition that keeps DG activity at a few percent.  Completion is a
    single CA3-style sweep: re-express the sparse code as a weighted blend of the
    stored sparse prototypes and re-sparsify.
    """

    def __init__(self, dim: int, *, expansion: int, sparsity: float,
                 complete: bool, seed: int, completion_gain: float = 3.0) -> None:
        rng = np.random.default_rng(seed + 31337)
        self.width = int(dim * expansion)
        self.k = max(1, int(round(self.width * sparsity)))
        self.projection = rng.normal(size=(dim, self.width)) / np.sqrt(dim)
        self.complete = bool(complete)
        self.completion_gain = float(completion_gain)
        self.completion_inert = 0
        self.completion_active = 0

    def encode(self, cue: np.ndarray) -> np.ndarray:
        activation = cue @ self.projection
        sparse = np.zeros(self.width, dtype=np.float64)
        winners = np.argpartition(-activation, self.k - 1)[: self.k]
        sparse[winners] = 1.0
        return sparse

    def encode_prototypes(self, means: np.ndarray) -> np.ndarray:
        return np.stack([self.encode(row) for row in means])

    def scores(self, cue: np.ndarray, means: np.ndarray) -> np.ndarray:
        """Overlap between the sparse cue code and each sparse prototype code."""

        code = self.encode(cue)
        prototype_codes = self.encode_prototypes(means)
        overlap = prototype_codes @ code
        if self.complete:
            # One CA3-style completion sweep: blend toward the stored patterns and
            # re-sparsify.  `completion_gain` must exceed 1 or the sweep is a no-op:
            # the cue's own units carry value 1.0 while the blend contributes at
            # most 1.0, so with unit gain the top-k always re-selects the original
            # winners and `completed == code` identically.  That silent identity is
            # exactly the kind of switched-off mechanism this project keeps finding,
            # so the gain is explicit and the liveness check below verifies it bites.
            weights = np.exp(overlap - overlap.max())
            weights /= weights.sum()
            blended = self.completion_gain * (weights @ prototype_codes) + code
            winners = np.argpartition(-blended, self.k - 1)[: self.k]
            completed = np.zeros(self.width, dtype=np.float64)
            completed[winners] = 1.0
            if np.array_equal(completed, code):
                self.completion_inert += 1
            else:
                self.completion_active += 1
            overlap = prototype_codes @ completed
        return overlap / self.k


# --------------------------------------------------------------------------- #
# routers                                                                      #
# --------------------------------------------------------------------------- #

def run_router(stream: Stream, *, mode: str, stickiness: float, known: bool,
               separator: PatternSeparator | None, shuffle_cues: bool,
               variance_floor: float, seed: int) -> dict:
    """Route every step and score against the withheld truth.

    mode 'single'      -- one draw per step, argmax of likelihood + point-estimate prior
                          (this is the shipped ADRN-2B rule)
    mode 'accumulate'  -- HMM forward filter: carry a distribution and propagate it
    """

    n_contexts, dim = stream.prototypes.shape
    store = PrototypeStore(
        n_contexts, dim, floor=variance_floor,
        known_prototypes=stream.prototypes if known else None,
        known_variance=stream.noise_std**2 if known else None,
    )
    rng = np.random.default_rng(seed + 909)
    transition = np.full((n_contexts, n_contexts), (1.0 - stickiness) / (n_contexts - 1))
    np.fill_diagonal(transition, stickiness)
    log_transition = np.log(transition)

    posterior = np.full(n_contexts, 1.0 / n_contexts)
    active = 0
    predictions = np.zeros(len(stream.cues), dtype=np.int64)

    for step, cue in enumerate(stream.cues):
        evidence_cue = cue
        if shuffle_cues:
            # CONTROL: accumulate a cue from a uniformly random context. Any arm
            # that still improves is not improving because of context evidence.
            fake = int(rng.integers(0, n_contexts))
            evidence_cue = stream.prototypes[fake] + rng.normal(
                0.0, stream.noise_std, dim
            )

        seen = store.seen if not store.known else np.ones(n_contexts, dtype=bool)
        if not seen.any():
            predictions[step] = 0
        else:
            if separator is None:
                log_like = store.log_likelihood(evidence_cue)
            else:
                # Overlap is a similarity, not a density; put it on a comparable
                # scale by treating it as an unnormalised log score.
                log_like = separator.scores(evidence_cue, store.mean) * dim
            log_like = np.where(seen, log_like, -np.inf)

            if mode == "single":
                scores = log_like + log_transition[active]
                posterior = _softmax(scores)
            elif mode == "accumulate":
                predicted = _logsumexp_matmul(np.log(posterior + 1e-300), log_transition)
                scores = log_like + predicted
                posterior = _softmax(scores)
            else:
                raise ValueError("mode must be 'single' or 'accumulate'")
            predictions[step] = int(np.argmax(posterior))
            active = predictions[step]

        if stream.supervised[step]:
            true_index = int(stream.truth[step])
            store.observe(true_index, cue)
            # A supervised step pins the context; both arms get this equally.
            active = true_index
            posterior = np.full(n_contexts, 1e-6)
            posterior[true_index] = 1.0
            posterior /= posterior.sum()

    result = _score(stream, predictions)
    result["_predictions"] = predictions
    if separator is not None and separator.complete:
        total = separator.completion_active + separator.completion_inert
        result["completion_active_fraction"] = (
            separator.completion_active / total if total else 0.0
        )
    return result


def _softmax(scores: np.ndarray) -> np.ndarray:
    finite = np.where(np.isfinite(scores), scores, -np.inf)
    if not np.isfinite(finite).any():
        return np.full(len(scores), 1.0 / len(scores))
    shifted = finite - finite.max()
    weights = np.exp(shifted)
    total = weights.sum()
    return weights / total if total > 0 else np.full(len(scores), 1.0 / len(scores))


def _logsumexp_matmul(log_vector: np.ndarray, log_matrix: np.ndarray) -> np.ndarray:
    """log( exp(log_vector) @ exp(log_matrix) ), stably."""

    combined = log_vector[:, None] + log_matrix
    peak = combined.max(axis=0)
    return peak + np.log(np.exp(combined - peak[None, :]).sum(axis=0))


def _score(stream: Stream, predictions: np.ndarray) -> dict:
    correct = predictions == stream.truth
    unsupervised = ~stream.supervised
    # The return windows: unsupervised steps of a phase whose context appeared before.
    first_seen: dict[int, int] = {}
    is_return = np.zeros(len(stream.cues), dtype=bool)
    for phase in np.unique(stream.phase_of):
        rows = stream.phase_of == phase
        context = int(stream.truth[rows][0])
        if context in first_seen:
            is_return |= rows
        else:
            first_seen[context] = int(phase)
    by_position = {}
    for checkpoint in POSITION_CHECKPOINTS:
        rows = unsupervised & (stream.position < checkpoint)
        if rows.any():
            by_position[str(checkpoint)] = float(correct[rows].mean())
    return {
        "overall_unsupervised": float(correct[unsupervised].mean()),
        "return_windows": float(correct[unsupervised & is_return].mean()),
        "first_visits": float(correct[unsupervised & ~is_return].mean()),
        "cumulative_by_position": by_position,
        "n_unsupervised": int(unsupervised.sum()),
    }


ARMS = {
    "1_single_draw         (shipped rule)": dict(mode="single", sep=False, comp=False, shuf=False),
    "2_accumulate          (direction 1)": dict(mode="accumulate", sep=False, comp=False, shuf=False),
    "3_separate_single     (direction 2)": dict(mode="single", sep=True, comp=False, shuf=False),
    "4_separate_accumulate (1 + 2)": dict(mode="accumulate", sep=True, comp=False, shuf=False),
    "5_separate_complete   (2 + CA3 sweep)": dict(mode="accumulate", sep=True, comp=True, shuf=False),
    "6_CTRL_shuffled_acc   (must not gain)": dict(mode="accumulate", sep=False, comp=False, shuf=True),
}


def _evaluate(tasks, *, difficulty, known, stickiness, args) -> tuple[dict, list[str]]:
    collected: dict[str, list[dict]] = {name: [] for name in ARMS}
    warnings: list[str] = []
    identical = 0
    for seed in range(args.seeds):
        stream = make_stream(
            tasks, difficulty=difficulty, probes_per_phase=args.probes_per_phase,
            supervised_prefix=args.supervised_prefix, seed=1000 + seed,
        )
        for name, spec in ARMS.items():
            separator = None
            if spec["sep"]:
                separator = PatternSeparator(
                    stream.prototypes.shape[1], expansion=args.expansion,
                    sparsity=args.sparsity, complete=spec["comp"], seed=seed,
                )
            collected[name].append(run_router(
                stream, mode=spec["mode"], stickiness=stickiness, known=known,
                separator=separator, shuffle_cues=spec["shuf"],
                variance_floor=args.variance_floor, seed=seed,
            ))
        # MECHANISM LIVENESS. An accumulating router that emits byte-identical
        # predictions to the single-draw router has not been tested -- it has been
        # switched off. This project has twice reported ablations that turned out
        # to be inert, so the check is explicit rather than assumed.
        single = collected["1_single_draw         (shipped rule)"][-1]["_predictions"]
        accumulate = collected["2_accumulate          (direction 1)"][-1]["_predictions"]
        if np.array_equal(single, accumulate):
            identical += 1
    if identical:
        warnings.append(
            f"accumulate is byte-identical to single_draw on {identical}/{args.seeds} seeds "
            f"at stickiness={stickiness}: the forward filter carries no history, so "
            f"direction 1 is INERT here and its numbers are not a result"
        )
    completion = [
        run.get("completion_active_fraction")
        for run in collected["5_separate_complete   (2 + CA3 sweep)"]
    ]
    completion = [value for value in completion if value is not None]
    if completion and float(np.mean(completion)) < 0.01:
        warnings.append(
            f"CA3 completion sweep changed the sparse code on only "
            f"{float(np.mean(completion)):.4f} of steps -- effectively inert"
        )

    summary: dict[str, object] = {}
    for name, runs in collected.items():
        def agg(key):
            values = np.array([run[key] for run in runs])
            return {"mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
        summary[name] = {
            "overall_unsupervised": agg("overall_unsupervised"),
            "return_windows": agg("return_windows"),
            "first_visits": agg("first_visits"),
            "cumulative_by_position": {
                key: float(np.mean([run["cumulative_by_position"][key] for run in runs]))
                for key in runs[0]["cumulative_by_position"]
            },
        }
    if completion:
        summary["_completion_active_fraction"] = float(np.mean(completion))
    return summary, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--probes-per-phase", type=int, default=256)
    parser.add_argument("--supervised-prefix", type=int, default=8)
    parser.add_argument("--variance-floor", type=float, default=0.01)
    parser.add_argument("--expansion", type=int, default=50)
    parser.add_argument("--sparsity", type=float, default=0.02)
    parser.add_argument(
        "--stickiness", default="",
        help="comma-separated; default sweeps the shipped 0.25, two intermediates, "
             "and the dwell-matched 1 - 1/probes_per_phase",
    )
    args = parser.parse_args(argv)

    tasks = _load(args.package)
    dwell_matched = round(1.0 - 1.0 / args.probes_per_phase, 6)
    if args.stickiness.strip():
        sweep = [float(v) for v in args.stickiness.split(",") if v.strip()]
    else:
        sweep = [0.25, 0.85, 0.98, dwell_matched]

    payload: dict[str, object] = {
        "probes_per_phase": args.probes_per_phase,
        "supervised_prefix": args.supervised_prefix,
        "seeds": args.seeds,
        "phase_pattern": list(PHASE_PATTERN),
        "chance": 0.25,
        "shipped_stickiness": 0.25,
        "dwell_matched_stickiness": dwell_matched,
        "stickiness_sweep": sweep,
        "note_on_shipped_setting": (
            "ADRN-2B ships context_stickiness=0.25 with 4 real contexts. At 4 contexts "
            "0.25 == 1/4 exactly, so the transition matrix is UNIFORM and a forward "
            "filter multiplies by a constant every step: no history survives and "
            "evidence accumulation is arithmetically impossible at that setting."
        ),
        "regimes": {}, "warnings": [],
    }

    for stickiness in sweep:
        for difficulty in ("easy", "medium", "hard"):
            for known in (True, False):
                label = "known-prototypes" if known else "learned-prototypes"
                regime = f"stick={stickiness}/{difficulty}/{label}"
                summary, warnings = _evaluate(
                    tasks, difficulty=difficulty, known=known,
                    stickiness=stickiness, args=args,
                )
                payload["regimes"][regime] = summary
                payload["warnings"].extend(f"{regime}: {w}" for w in warnings)
                print(f"\n=== {regime} ===  (chance 0.2500, {args.seeds} seeds)")
                print(f"{'arm':38s} {'overall':>17s} {'return windows':>17s}")
                for name, values in summary.items():
                    if name.startswith("_"):
                        continue
                    o, r = values["overall_unsupervised"], values["return_windows"]
                    print(f"{name:38s} {o['mean']:.4f} +/-{o['sd']:.4f}  "
                          f"{r['mean']:.4f} +/-{r['sd']:.4f}")
                for warning in warnings:
                    print(f"    !! {warning}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwrote {args.output}")
    if payload["warnings"]:
        print(f"{len(payload['warnings'])} liveness warnings recorded in the output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
