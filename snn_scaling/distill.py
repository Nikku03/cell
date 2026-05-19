"""Trick 6: knowledge distillation (teacher → student).

Train a large 'teacher' network on a task, then compress its knowledge
into a smaller 'student'. The student's training target is the teacher's
soft predictions, not the original hard labels. Two reasons this helps:

  - Soft labels carry more information per sample (calibrated confidence
    over all classes, not just the winner index)
  - The student can match the teacher's behaviour on intermediate cases
    where the hard label is uninformative

For our reservoir + linear-readout substrate, distillation is structurally
simple:

  teacher_readout:    W_t = ridge_solve( X_t , y_true )
  student_solo:       W_s_solo = ridge_solve( X_s , y_true )
  student_distilled:  W_s_dist = ridge_solve( X_s , teacher_predict(X_t) )

The student's reservoir features X_s are DIFFERENT from the teacher's
(different seed, smaller size), but the teacher's predictions form a
rich regression target that the student's smaller reservoir can fit.

This module provides:
  - DistillationTask: generates a synthetic multi-class task
  - train_teacher_student: builds + trains a (teacher, student_solo,
                            student_distilled) trio
  - distill_predictions: utility to turn teacher logits into soft targets
                          at temperature T
  - DistillationResult: holds accuracy/params/timing for comparison
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .reservoir import (
    LiquidStateMachine,
    ReservoirConfig,
    train_linear_readout,
    predict_linear,
)


# ---------------- multi-class noisy-frequency task ----------------

@dataclass
class TaskBatch:
    inputs: torch.Tensor       # (n_samples, T)
    labels: torch.Tensor       # (n_samples,) long, class index 0..K-1
    freqs: list[float]         # the K frequencies used


def make_freq_task(
    n_classes: int = 3,
    n_samples: int = 200,
    T: int = 200,
    dt_ms: float = 1.0,
    sample_rate_hz: float = 1000.0,
    noise_std: float = 0.4,
    freqs: list[float] | None = None,
    seed: int = 0,
) -> TaskBatch:
    """Generate a multi-class temporal-statistics classification task.

    The classes have DIFFERENT WAVEFORM TYPES (not just frequency
    differences, which our membrane-statistic features struggle with):

      class 0: sine wave         (rhythmic, sinusoidal)
      class 1: square wave       (rhythmic, sharp transitions)
      class 2: white noise       (no structure)
      class 3: chirp             (frequency sweep; only used if n_classes=4)

    All at unit amplitude with the same noise_std additive corruption.
    Different temporal statistics → different reservoir activation
    patterns → the (mean, std, range) features can separate them.
    """
    g = torch.Generator().manual_seed(seed)
    inputs = torch.zeros(n_samples, T)
    labels = torch.zeros(n_samples, dtype=torch.long)
    freqs_used: list[float] = []

    for i in range(n_samples):
        cls = int(torch.randint(0, n_classes, (1,), generator=g).item())
        phase = torch.rand((), generator=g).item() * 2 * math.pi
        amp = 0.8 + 0.4 * torch.rand((), generator=g).item()
        f = 3.0 + 4.0 * torch.rand((), generator=g).item()       # 3-7 Hz
        t = torch.arange(T, dtype=torch.float32) * dt_ms / 1000.0
        if cls == 0:
            wave = amp * torch.sin(2 * math.pi * f * t + phase)
        elif cls == 1:
            # Square wave at frequency f
            wave = amp * torch.sign(torch.sin(2 * math.pi * f * t + phase))
        elif cls == 2:
            # White noise (no temporal structure)
            wave = amp * torch.randn(T, generator=g)
        elif cls == 3:
            # Chirp from f to 2f
            f0, f1 = f, 2 * f
            k_chirp = (f1 - f0) / max(t[-1].item(), 1e-6)
            wave = amp * torch.sin(
                2 * math.pi * (f0 * t + 0.5 * k_chirp * t * t) + phase
            )
        else:
            raise ValueError(f"unsupported class index {cls}")
        noise = torch.randn(T, generator=g) * noise_std
        inputs[i] = wave + noise
        labels[i] = cls

    return TaskBatch(inputs=inputs, labels=labels, freqs=freqs_used)


# ---------------- reservoir feature extraction ----------------

def _drive_input(N: int, signal: torch.Tensor, scale: float = 5.0,
                  proj_seed: int = 7777) -> torch.Tensor:
    """Random projection from 1-d signal to N-d per-neuron drive."""
    g = torch.Generator().manual_seed(proj_seed)
    P = (torch.rand(N, generator=g) - 0.5) * 2.0
    T = signal.shape[0]
    return signal.unsqueeze(-1) * P.unsqueeze(0) * scale


def _run_reservoir_collect_V(lsm: LiquidStateMachine, drive: torch.Tensor,
                              dt: float = 1.0) -> torch.Tensor:
    """Return (T, N) membrane trace for a single input drive."""
    T = drive.shape[0]
    N = lsm.reservoir.n
    Vs = torch.zeros(T, N)
    lsm.reset()
    for k in range(T):
        Vs[k] = lsm.step_and_feature(drive[k], dt=dt, t=k * dt)
    return Vs


def extract_features(lsm: LiquidStateMachine, batch: TaskBatch,
                      proj_seed: int = 7777) -> torch.Tensor:
    """Return (n_samples, 3*N) per-sample feature: [mean V, std V, range V]
    over the late half of each input. Same recipe as Phase 3."""
    N = lsm.reservoir.n
    out = torch.zeros(batch.inputs.shape[0], 3 * N)
    for i in range(batch.inputs.shape[0]):
        drive = _drive_input(N, batch.inputs[i], proj_seed=proj_seed)
        Vs = _run_reservoir_collect_V(lsm, drive)
        late = Vs[Vs.shape[0] // 2:]
        feat = torch.cat([
            late.mean(dim=0),
            late.std(dim=0),
            late.max(dim=0).values - late.min(dim=0).values,
        ])
        out[i] = feat
    return out


# ---------------- one-hot / soft-label utilities ----------------

def one_hot(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=n_classes).float()


def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return F.softmax(logits / max(temperature, 1e-9), dim=-1)


def predict_class(W: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Top-1 class prediction from a linear readout's output."""
    return predict_linear(W, features).argmax(dim=-1)


def accuracy(pred: torch.Tensor, labels: torch.Tensor) -> float:
    return (pred == labels).float().mean().item()


# ---------------- the headline pipeline ----------------

@dataclass
class DistillationResult:
    teacher_n: int
    student_n: int
    teacher_test_acc: float
    student_solo_test_acc: float
    student_distilled_test_acc: float
    teacher_inference_ms: float
    student_inference_ms: float
    teacher_readout_params: int
    student_readout_params: int


def train_teacher_student(
    teacher_n: int = 500,
    student_n: int = 80,
    n_classes: int = 3,
    n_teacher_train: int = 300,
    n_student_train: int = 30,
    n_test: int = 100,
    noise_std: float = 0.6,
    distill_temperature: float = 1.0,
    teacher_seed: int = 0,
    student_seed: int = 1,
    task_seed: int = 0,
    ridge: float = 50.0,
    proj_seed: int = 7777,
) -> DistillationResult:
    """Train a big teacher on lots of data, then a small student two ways:
    'solo' on hard labels, 'distilled' on the teacher's predictions.

    What we expect to see on this substrate (random recurrent reservoir +
    closed-form ridge readout, deterministic labels):
      - teacher accuracy >> chance
      - distilled student matches teacher within a small margin (the
        student's small reservoir captures enough of the teacher's
        decision function to reproduce its predictions)
      - student has many fewer trainable params (compression)
      - student inference is faster (compute saving)
    The classic "dark knowledge" boost of distilled > solo is a non-linear
    training effect that doesn't surface with closed-form linear readouts
    on noise-free labels, so we don't gate on it.

    Teacher and student train on DISJOINT samples so teacher's predictions
    on student samples reflect genuine generalisation (not memorisation).
    """
    teacher_train_batch = make_freq_task(
        n_classes=n_classes, n_samples=n_teacher_train,
        noise_std=noise_std, seed=task_seed,
    )
    student_train_batch = make_freq_task(
        n_classes=n_classes, n_samples=n_student_train,
        noise_std=noise_std, seed=task_seed + 17,
    )
    test_batch = make_freq_task(
        n_classes=n_classes, n_samples=n_test,
        noise_std=noise_std, seed=task_seed + 99,
    )

    # 1. Teacher
    teacher_cfg = ReservoirConfig(n_reservoir=teacher_n, seed=teacher_seed)
    teacher = LiquidStateMachine(teacher_cfg)
    Xt_train = extract_features(teacher, teacher_train_batch, proj_seed=proj_seed)
    Xt_test = extract_features(teacher, test_batch, proj_seed=proj_seed)
    y_train_full = one_hot(teacher_train_batch.labels, n_classes)
    W_teacher = train_linear_readout(Xt_train, y_train_full, ridge=ridge)
    teacher_test_pred = predict_class(W_teacher, Xt_test)
    teacher_test_acc = accuracy(teacher_test_pred, test_batch.labels)

    # Teacher's predictions on the student's training samples (which the
    # teacher has NOT seen). predict_linear returns scores trained to match
    # one-hot targets, so they already act as a class-probability surrogate
    # (no softmax needed; softmax of small ridge outputs collapses to
    # near-uniform and loses signal).
    Xt_student_train = extract_features(teacher, student_train_batch, proj_seed=proj_seed)
    teacher_scores_for_student = predict_linear(W_teacher, Xt_student_train)
    if distill_temperature != 1.0:
        soft_targets = softmax_with_temperature(teacher_scores_for_student, temperature=distill_temperature)
    else:
        soft_targets = teacher_scores_for_student

    # 2. Student solo (small reservoir, small data, hard labels)
    student_cfg = ReservoirConfig(n_reservoir=student_n, seed=student_seed)
    student_solo = LiquidStateMachine(student_cfg)
    Xs_train = extract_features(student_solo, student_train_batch, proj_seed=proj_seed)
    Xs_test = extract_features(student_solo, test_batch, proj_seed=proj_seed)
    y_student_train = one_hot(student_train_batch.labels, n_classes)
    W_student_solo = train_linear_readout(Xs_train, y_student_train, ridge=ridge)
    student_solo_test_pred = predict_class(W_student_solo, Xs_test)
    student_solo_test_acc = accuracy(student_solo_test_pred, test_batch.labels)

    # 3. Student distilled (same small reservoir, same data, soft labels from teacher)
    W_student_distilled = train_linear_readout(Xs_train, soft_targets, ridge=ridge)
    student_distilled_test_pred = predict_class(W_student_distilled, Xs_test)
    student_distilled_test_acc = accuracy(student_distilled_test_pred, test_batch.labels)

    # 4. Inference timing (per-sample)
    # Time how long predicting on a single sample takes (just the readout part is trivial;
    # the reservoir forward is the cost)
    sample_drive_teacher = _drive_input(teacher_n, test_batch.inputs[0])
    sample_drive_student = _drive_input(student_n, test_batch.inputs[0])
    # warmup
    _ = _run_reservoir_collect_V(teacher, sample_drive_teacher)
    _ = _run_reservoir_collect_V(student_solo, sample_drive_student)
    t0 = time.time()
    for _ in range(5):
        _run_reservoir_collect_V(teacher, sample_drive_teacher)
    t_teacher = (time.time() - t0) / 5 * 1000.0  # ms
    t0 = time.time()
    for _ in range(5):
        _run_reservoir_collect_V(student_solo, sample_drive_student)
    t_student = (time.time() - t0) / 5 * 1000.0

    # 5. Parameter counts (only counting trainable readout parameters)
    teacher_params = W_teacher.numel()
    student_params = W_student_distilled.numel()

    return DistillationResult(
        teacher_n=teacher_n,
        student_n=student_n,
        teacher_test_acc=teacher_test_acc,
        student_solo_test_acc=student_solo_test_acc,
        student_distilled_test_acc=student_distilled_test_acc,
        teacher_inference_ms=t_teacher,
        student_inference_ms=t_student,
        teacher_readout_params=teacher_params,
        student_readout_params=student_params,
    )
