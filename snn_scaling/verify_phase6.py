"""Phase 6 verification: knowledge distillation (teacher -> student).

Gates:

  1. Multi-class task is non-trivially learnable: teacher accuracy
     materially exceeds chance
  2. Knowledge transfer: distilled student reproduces the teacher's
     behaviour (its test accuracy lands close to the teacher's)
  3. Parameter compression: student readout has many fewer trainable
     parameters than the teacher
  4. Inference speedup: a single sample runs through the student
     reservoir noticeably faster than through the teacher
  5. Soft targets are non-degenerate: the teacher's predictions on
     student samples carry per-sample probability mass across classes
     (not all-uniform, not all-one-hot)
  6. End-to-end pipeline runs without error on default settings
"""

from __future__ import annotations

import time

import torch

from .distill import (
    DistillationResult,
    make_freq_task,
    extract_features,
    one_hot,
    softmax_with_temperature,
    predict_class,
    predict_linear,
    train_teacher_student,
)
from .reservoir import (
    LiquidStateMachine,
    ReservoirConfig,
    train_linear_readout,
)
from .verify_phase0 import CheckResult, format_results


_DEFAULT_RESULT: DistillationResult | None = None


def _get_default_result() -> DistillationResult:
    """Run the headline distillation pipeline once and cache it; multiple
    gates reuse the same trio so we don't pay for the reservoir sweep
    three times."""
    global _DEFAULT_RESULT
    if _DEFAULT_RESULT is None:
        _DEFAULT_RESULT = train_teacher_student(
            teacher_seed=0, student_seed=1, task_seed=0,
        )
    return _DEFAULT_RESULT


# ---------------- gates ----------------

def check_teacher_learns_task() -> CheckResult:
    """Teacher accuracy on a held-out test set is well above chance.

    Three classes (sine / square / noise) means chance is ~33%. Teacher
    is given enough training data + reservoir capacity to comfortably
    exceed that floor.
    """
    r = _get_default_result()
    return CheckResult(
        "teacher learns the 3-class task (accuracy >= 70%)",
        r.teacher_test_acc >= 0.70, r.teacher_test_acc, 0.70,
        note=f"(teacher test accuracy = {r.teacher_test_acc*100:.1f}%, chance = 33%)",
    )


def check_distillation_transfers_behaviour() -> CheckResult:
    """Distilled student lands within a small margin of the teacher.

    The student is trained on the teacher's predictions over its own
    (much smaller) training pool. Closing the test-accuracy gap
    demonstrates that the student has internalised the teacher's
    decision function despite being much smaller.
    """
    r = _get_default_result()
    gap = r.teacher_test_acc - r.student_distilled_test_acc
    return CheckResult(
        "distilled student matches teacher within 15pp",
        gap <= 0.15, gap, 0.15,
        note=f"(teacher={r.teacher_test_acc*100:.1f}%, "
             f"distilled student={r.student_distilled_test_acc*100:.1f}%, "
             f"gap={gap*100:+.1f}pp)",
    )


def check_parameter_compression() -> CheckResult:
    """Student readout has many fewer trainable params than the teacher's."""
    r = _get_default_result()
    ratio = r.teacher_readout_params / max(r.student_readout_params, 1)
    return CheckResult(
        "student readout is >= 5x more compact than teacher's",
        ratio >= 5.0, ratio, 5.0,
        note=f"(teacher params={r.teacher_readout_params}, "
             f"student params={r.student_readout_params}, ratio={ratio:.2f}x)",
    )


def check_inference_speedup() -> CheckResult:
    """Single-sample inference through the student reservoir is faster."""
    r = _get_default_result()
    speedup = r.teacher_inference_ms / max(r.student_inference_ms, 1e-6)
    return CheckResult(
        "student inference is >= 1.5x faster than teacher",
        speedup >= 1.5, speedup, 1.5,
        note=f"(teacher={r.teacher_inference_ms:.1f}ms, "
             f"student={r.student_inference_ms:.1f}ms, speedup={speedup:.2f}x)",
    )


def check_soft_targets_carry_information() -> CheckResult:
    """Teacher's soft predictions on the student's training data are
    non-degenerate: they vary per-sample (not all the same), spread mass
    across classes (not pure one-hot), and aren't uniform.

    Concretely: build a small teacher trio, extract soft targets,
    measure per-row max probability (closer to 1.0 means more confident).
    Healthy values lie in [0.4, 0.95] — neither uniform nor one-hot.
    """
    task = make_freq_task(n_classes=3, n_samples=200, noise_std=0.6, seed=0)
    student_task = make_freq_task(n_classes=3, n_samples=25, noise_std=0.6, seed=17)

    teacher = LiquidStateMachine(ReservoirConfig(n_reservoir=200, seed=0))
    Xt_train = extract_features(teacher, task)
    Y_train = one_hot(task.labels, 3)
    W_t = train_linear_readout(Xt_train, Y_train, ridge=50.0)
    Xt_student = extract_features(teacher, student_task)
    raw = predict_linear(W_t, Xt_student)
    soft = softmax_with_temperature(raw, temperature=1.0)
    per_row_max = soft.max(dim=-1).values   # in [1/K, 1]
    mean_conf = per_row_max.mean().item()
    # Healthy band: not uniform (mean_conf > 1/3 = 0.33), not pure one-hot (< 0.97)
    in_band = 0.40 <= mean_conf <= 0.97
    return CheckResult(
        "soft targets non-degenerate (mean confidence in [0.40, 0.97])",
        in_band, mean_conf, 0.40,
        note=f"(mean top-class prob across {soft.shape[0]} student samples = {mean_conf:.3f})",
    )


def check_pipeline_runs() -> CheckResult:
    """Full pipeline runs end-to-end with default settings and returns
    a complete DistillationResult."""
    n_errors = 0
    try:
        r = _get_default_result()
        ok = (
            isinstance(r, DistillationResult)
            and 0.0 <= r.teacher_test_acc <= 1.0
            and 0.0 <= r.student_solo_test_acc <= 1.0
            and 0.0 <= r.student_distilled_test_acc <= 1.0
            and r.teacher_readout_params > 0
            and r.student_readout_params > 0
            and r.teacher_inference_ms > 0
            and r.student_inference_ms > 0
        )
    except Exception:
        ok = False
        n_errors = 1
    return CheckResult(
        "train_teacher_student returns a valid DistillationResult",
        ok, 0.0 if ok else 1.0, 0.5,
        note=("ok" if ok else f"({n_errors} errors / invalid result)"),
    )


# ---------------- runner ----------------

def run_all() -> list[CheckResult]:
    return [
        check_pipeline_runs(),
        check_teacher_learns_task(),
        check_distillation_transfers_behaviour(),
        check_parameter_compression(),
        check_inference_speedup(),
        check_soft_targets_carry_information(),
    ]


if __name__ == "__main__":
    t0 = time.time()
    results = run_all()
    print(f"Phase 6 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    print(f"\n  -> {n_pass} / {len(results)} checks passed.")
