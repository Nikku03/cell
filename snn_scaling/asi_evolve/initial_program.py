"""ASI-Evolve candidate program for snn_scaling reversal benchmark.

Runs a multi-seed reversal-learning experiment (5 stim classes x 3 actions,
4 phases x PHASE_LENGTH trials per phase, mapping reshuffles per phase),
with three agents on the SAME reservoir features:

  - Naive Q-learning (linear Q + TD)
  - MemRec (single recency-weighted hippocampus)
  - CLS    (hippocampus + neocortex consolidation)

ASI-Evolve mutates the MUTABLE SECTION between the BEGIN/END markers.
The harness footer below DOES NOT change - it runs the benchmark, prints
metrics, and emits the ===EVAL_METRICS=== block the evaluator parses.

Composite score (lower = better, see input.md):
  composite = -mean_memrec_cum_reward + 0.1 * wall_time_minutes
            + penalty if MemRec mean < integrated_threshold

================================================================
== STRUCTURE: THREE sections                                   ==
==   1. HARNESS HEADER  - imports, paths. DO NOT MODIFY.       ==
==   2. MUTABLE SECTION - the Researcher's only target.        ==
==   3. HARNESS FOOTER  - pipeline + metrics. DO NOT MODIFY.   ==
================================================================
"""
# ================================================================
# === HARNESS HEADER - DO NOT MODIFY ===
# ================================================================
import json
import sys
import time
import statistics

sys.path.insert(0, '/content/cell')

import torch
import torch.nn.functional as F

from snn_scaling.reservoir import SpikingReservoir, ReservoirConfig
from snn_scaling.rl_tasks import (
    make_5class_task,
    extract_centered_features,
    make_reversal_phase_maps,
)
from snn_scaling.rl_agents import (
    NaiveQAgent,
    TimeWeightedMemoryAgent,
    CLSAgent,
)
# ================================================================
# === END HARNESS HEADER ===
# ================================================================


# ================================================================
# === BEGIN MUTABLE SECTION ===
# Researcher: this is the ONLY block you should change.
# ================================================================

# ---- Reservoir (SpikingReservoir / ReservoirConfig) ----
N_RESERVOIR = 2000
P_RECURRENT = 0.10
G_EXC = 0.04
G_INH = 0.30
TAU_SYN = 5.0
RES_NOISE_STD = 0.2
EXC_FRACTION = 0.8

# ---- Task / feature extraction ----
T_TIMESTEPS = 200            # samples per trial input
TASK_NOISE_STD = 0.4         # additive Gaussian noise on inputs
N_PER_CLASS_POOL = 30        # 5 * N_PER_CLASS_POOL distinct stimuli in pool

# ---- MemRec hyperparameters ----
MEMREC_TAU = 50.0            # recency time constant
MEMREC_TOP_K = 10
MEMREC_EPS = 0.10

# ---- CLS hyperparameters ----
CLS_HIPPO_TAU = 20.0         # hippo recency (faster than MemRec because cortex stabilises)
CLS_HIPPO_TOP_K = 10
CLS_HIPPO_CAPACITY = 200
CLS_CORTEX_THRESHOLD = 0.6   # cosine threshold for joining existing cluster
CLS_CORTEX_LR = 0.05         # EMA learning rate for cortex Q-tables
CLS_ALPHA = 0.7              # blend: alpha * hippo + (1-alpha) * cortex
CLS_EPS = 0.10

# ---- Naive baseline ----
NAIVE_LR = 0.05
NAIVE_EPS = 0.10

# Notes (do NOT include these comment lines in SEARCH blocks):
# - N_RESERVOIR: 500-3000 safe; bigger costs O(N^2) edges in reservoir build
# - MEMREC_TAU: 20-100 safe; lower = faster forgetting; 50 is the prior best
# - CLS_HIPPO_TAU: typically < MEMREC_TAU because cortex provides stability
# - CLS_ALPHA: 0.6-0.9 safe; <0.5 lets stale cortex Q dominate
# - CLS_CORTEX_THRESHOLD: 0.4-0.8 safe; too low = single cluster, too high = no consolidation
# - TASK_NOISE_STD: 0.2-0.6 safe; >0.6 makes the task feature-impossible
# - All epsilons in [0.05, 0.20]

# ================================================================
# === END MUTABLE SECTION ===
# ================================================================


# ================================================================
# === HARNESS FOOTER - DO NOT MODIFY ===
# ================================================================

N_SEEDS = 3                    # multi-seed average
PHASE_LENGTH = 150
N_PHASES = 4
N_ACTIONS = 3
N_CLASSES = 5
RECOVERY_WINDOW = 30           # rolling window for trials-to-criterion
RECOVERY_THRESHOLD = 0.70


def _run_one_seed(seed: int, device: str) -> dict:
    n_trials = PHASE_LENGTH * N_PHASES
    pool = make_5class_task(
        n_per_class=N_PER_CLASS_POOL, T=T_TIMESTEPS,
        noise_std=TASK_NOISE_STD, seed=seed,
    )
    pool_labels = pool.labels.to(device=device)

    res_cfg = ReservoirConfig(
        n_reservoir=N_RESERVOIR, p_recurrent=P_RECURRENT,
        g_exc=G_EXC, g_inh=G_INH, exc_fraction=EXC_FRACTION,
        tau_syn=TAU_SYN, noise_std=RES_NOISE_STD,
        seed=seed,
    )
    reservoir = SpikingReservoir(res_cfg, device=device)
    X = extract_centered_features(reservoir, pool, device=device,
                                     center=True, l2_normalize=True)

    phase_maps_cpu = make_reversal_phase_maps(
        n_phases=N_PHASES, n_classes=N_CLASSES, n_actions=N_ACTIONS,
        seed=seed + 90000,
    )
    phase_maps = [m.to(device=device) for m in phase_maps_cpu]

    feat_dim = X.shape[1]
    naive = NaiveQAgent(feat_dim, N_ACTIONS, lr=NAIVE_LR, eps=NAIVE_EPS,
                          device=device, seed=seed + 100)
    memrec = TimeWeightedMemoryAgent(
        feat_dim, N_ACTIONS,
        capacity=n_trials, top_k=MEMREC_TOP_K,
        eps=MEMREC_EPS, tau=MEMREC_TAU,
        device=device, seed=seed + 100,
    )
    cls = CLSAgent(
        feat_dim, N_ACTIONS,
        hippo_capacity=CLS_HIPPO_CAPACITY,
        hippo_tau=CLS_HIPPO_TAU, hippo_top_k=CLS_HIPPO_TOP_K,
        cortex_clusters=50, cortex_threshold=CLS_CORTEX_THRESHOLD,
        cortex_lr=CLS_CORTEX_LR, alpha=CLS_ALPHA, eps=CLS_EPS,
        device=device, seed=seed + 100,
    )

    stim_rng = torch.Generator(device="cpu").manual_seed(seed + 200)
    n_samples = X.shape[0]
    n_cor, m_cor, c_cor = [], [], []
    n_rew, m_rew, c_rew = 0.0, 0.0, 0.0

    for trial in range(n_trials):
        phase = trial // PHASE_LENGTH
        cmap = phase_maps[phase]
        idx = int(torch.randint(0, n_samples, (1,), generator=stim_rng).item())
        x = X[idx]
        true_cls = int(pool_labels[idx].item())
        correct_a = int(cmap[true_cls].item())

        for agent, cor_list, rew_acc, key in (
            (naive, n_cor, n_rew, "naive"),
            (memrec, m_cor, m_rew, "memrec"),
            (cls, c_cor, c_rew, "cls"),
        ):
            a = agent.act(x)
            r = 1.0 if a == correct_a else -1.0
            agent.update(x, a, r)
            cor_list.append(int(a == correct_a))
            if key == "naive":
                n_rew += r
            elif key == "memrec":
                m_rew += r
            else:
                c_rew += r

    return {
        "seed": seed,
        "naive_cum": n_rew,
        "memrec_cum": m_rew,
        "cls_cum": c_rew,
        "naive_acc_last100": sum(n_cor[-100:]) / 100.0,
        "memrec_acc_last100": sum(m_cor[-100:]) / 100.0,
        "cls_acc_last100": sum(c_cor[-100:]) / 100.0,
        "n_trials": n_trials,
    }


def main():
    torch.manual_seed(0)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[candidate] device={device}")
    if device == "cuda":
        print(f"[candidate] gpu={torch.cuda.get_device_name(0)}")
    print(f"[candidate] N_RESERVOIR={N_RESERVOIR}, T={T_TIMESTEPS}, "
          f"MEMREC_TAU={MEMREC_TAU}, CLS_HIPPO_TAU={CLS_HIPPO_TAU}, "
          f"CLS_ALPHA={CLS_ALPHA}")

    per_seed = []
    for s in range(N_SEEDS):
        st = time.time()
        r = _run_one_seed(s, device=device)
        per_seed.append(r)
        print(f"  seed {s}: naive={r['naive_cum']:+.0f} "
              f"memrec={r['memrec_cum']:+.0f} cls={r['cls_cum']:+.0f}   "
              f"(last-100 acc: n={r['naive_acc_last100']*100:.1f}% "
              f"m={r['memrec_acc_last100']*100:.1f}% "
              f"c={r['cls_acc_last100']*100:.1f}%) "
              f"[{time.time()-st:.1f}s]")

    n_cum_m = statistics.mean(r["naive_cum"] for r in per_seed)
    m_cum_m = statistics.mean(r["memrec_cum"] for r in per_seed)
    c_cum_m = statistics.mean(r["cls_cum"] for r in per_seed)
    n_cum_s = statistics.stdev(r["naive_cum"] for r in per_seed) if N_SEEDS >= 2 else 0.0
    m_cum_s = statistics.stdev(r["memrec_cum"] for r in per_seed) if N_SEEDS >= 2 else 0.0
    c_cum_s = statistics.stdev(r["cls_cum"] for r in per_seed) if N_SEEDS >= 2 else 0.0

    wall_time = time.time() - t0
    gap_memrec_naive = m_cum_m - n_cum_m
    gap_cls_naive = c_cum_m - n_cum_m
    best_integrated_cum = max(m_cum_m, c_cum_m)

    print(f"\n[candidate] MEAN ± STD across {N_SEEDS} seeds:")
    print(f"  naive  cum = {n_cum_m:+.1f} ± {n_cum_s:.1f}")
    print(f"  memrec cum = {m_cum_m:+.1f} ± {m_cum_s:.1f}")
    print(f"  cls    cum = {c_cum_m:+.1f} ± {c_cum_s:.1f}")
    print(f"  gap memrec-naive = {gap_memrec_naive:+.1f}")
    print(f"  gap cls-naive    = {gap_cls_naive:+.1f}")
    print(f"  wall time        = {wall_time:.1f}s")

    # Composite score (lower = better):
    #   primary term: -best_integrated_cum (reward the BEST integrated agent)
    #   time penalty: 0.1 * wall_time_minutes (gentle pressure)
    #   sanity penalty: +200 if naive cum > best integrated cum (the architecture
    #     should not actively underperform a baseline)
    sanity_penalty = 200.0 if n_cum_m > best_integrated_cum else 0.0
    composite = -best_integrated_cum + 0.1 * (wall_time / 60.0) + sanity_penalty

    metrics = {
        "naive_cum_mean":   float(n_cum_m),
        "naive_cum_std":    float(n_cum_s),
        "memrec_cum_mean":  float(m_cum_m),
        "memrec_cum_std":   float(m_cum_s),
        "cls_cum_mean":     float(c_cum_m),
        "cls_cum_std":      float(c_cum_s),
        "gap_memrec_naive": float(gap_memrec_naive),
        "gap_cls_naive":    float(gap_cls_naive),
        "best_integrated_cum": float(best_integrated_cum),
        "best_integrated_agent": "cls" if c_cum_m > m_cum_m else "memrec",
        "n_seeds": int(N_SEEDS),
        "phase_length": int(PHASE_LENGTH),
        "n_phases": int(N_PHASES),
        "wall_time_sec": float(wall_time),
        "composite": float(composite),
        "sanity_penalty": float(sanity_penalty),
        "n_reservoir": int(N_RESERVOIR),
        "t_timesteps": int(T_TIMESTEPS),
    }

    print("\n===EVAL_METRICS===")
    print(json.dumps(metrics))
    print("===END_EVAL_METRICS===")


if __name__ == "__main__":
    main()
# ================================================================
# === END HARNESS FOOTER ===
# ================================================================
