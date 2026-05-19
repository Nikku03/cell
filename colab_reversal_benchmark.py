"""Self-contained Colab cell: reversal-learning benchmark (non-stationary RL).

Same 5-class temporal-pattern substrate as the Pavlovian benchmark, but the
class -> correct-action mapping reshuffles every PHASE_LENGTH trials. The
agent must DETECT each phase shift and RELEARN the new mapping. There is
no signal that the phase has changed - only the reward stream goes
negative for previously-correct actions.

Three agents on the SAME reservoir features:

  1. Naive               : linear Q(s,a) + TD updates (eps-greedy)
  2. Memory no-recency   : the Pavlovian winner - k-NN over ALL past
                            (state, action, signed-reward) entries.
                            Expected to fail here because old-phase
                            entries pollute new-phase retrieval.
  3. Memory + recency    : same k-NN but each retrieved entry is
                            additionally weighted by exp(-age/tau).
                            Old entries fade exponentially.

Headline metric:
  - per-phase trials-to-recovery (first rolling-30 acc >= 70% within a phase)
  - mean drop at each phase boundary (last 30 of prev - first 30 of new)
  - cumulative reward over 600 trials

5 seeds, multi-seed mean +- std. Expected A100 runtime: ~3-4 min.
"""

# ============================================================
# Imports
# ============================================================
import math
import time
import statistics
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}", flush=True)
if DEVICE == "cuda":
    print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)


# ============================================================
# Phase 0: vectorised LIF + sparse synapses
# ============================================================

@dataclass
class NeuronParams:
    tau_m: float = 20.0
    V_rest: float = -70.0
    V_thresh: float = -50.0
    V_reset: float = -75.0
    R_m: float = 10.0
    refractory: float = 2.0
    noise_std: float = 0.0


class LIFPopulation(nn.Module):
    def __init__(self, n, params=None, device="cpu", dtype=torch.float32):
        super().__init__()
        params = params or NeuronParams()
        self.n = n; self.device = torch.device(device); self.dtype = dtype
        self.register_buffer("V", torch.full((n,), params.V_rest, dtype=dtype, device=self.device))
        self.register_buffer("refr", torch.zeros(n, dtype=dtype, device=self.device))
        self.register_buffer("spiked", torch.zeros(n, dtype=torch.bool, device=self.device))
        for name in ("tau_m", "V_rest", "V_thresh", "V_reset", "R_m", "refractory", "noise_std"):
            self.register_buffer(f"_{name}", torch.full((n,), float(getattr(params, name)), dtype=dtype, device=self.device))

    @torch.no_grad()
    def reset_state(self):
        self.V.fill_(0.0); self.V += self._V_rest
        self.refr.zero_(); self.spiked.zero_()

    @torch.no_grad()
    def step(self, I_syn, g_syn, dt, t):
        was_refr = self.refr > 0
        self.refr.sub_(dt).clamp_(min=0.0)
        one_plus = 1.0 + self._R_m * g_syn
        V_ss = (self._V_rest + self._R_m * (I_syn + g_syn * self.V)) / one_plus
        tau_eff = self._tau_m / one_plus
        decay = torch.exp(-dt / tau_eff)
        V_new = V_ss + (self.V - V_ss) * decay
        if torch.any(self._noise_std > 0):
            V_new = V_new + torch.randn_like(V_new) * self._noise_std * math.sqrt(dt)
        V_new = torch.where(was_refr, self._V_reset, V_new)
        spike = (V_new >= self._V_thresh) & (~was_refr)
        V_new = torch.where(spike, self._V_reset, V_new)
        self.refr = torch.where(spike, self._refractory, self.refr)
        self.V = V_new; self.spiked = spike
        return spike


SYN_EXC, SYN_INH = 0, 1
_E_SYN = {SYN_EXC: 0.0, SYN_INH: -80.0}


class SparseSynapses(nn.Module):
    def __init__(self, n_pre, n_post, edge_pre, edge_post, kind, g_max, tau_syn=5.0,
                 device="cpu", dtype=torch.float32):
        super().__init__()
        device = torch.device(device); E = edge_pre.numel()
        self.n_pre, self.n_post, self.E = n_pre, n_post, E
        self.device, self.dtype = device, dtype
        self.register_buffer("edge_pre", edge_pre.to(device=device, dtype=torch.long))
        self.register_buffer("edge_post", edge_post.to(device=device, dtype=torch.long))
        self.register_buffer("kind", kind.to(device=device, dtype=torch.long))
        self.register_buffer("g_max", g_max.to(device=device, dtype=dtype))
        tau_t = (tau_syn.to(device=device, dtype=dtype) if isinstance(tau_syn, torch.Tensor)
                 else torch.full((E,), float(tau_syn), device=device, dtype=dtype))
        self.register_buffer("tau_syn", tau_t)
        E_syn = torch.where(
            self.kind == SYN_EXC,
            torch.full((E,), _E_SYN[SYN_EXC], device=device, dtype=dtype),
            torch.full((E,), _E_SYN[SYN_INH], device=device, dtype=dtype),
        )
        self.register_buffer("E_syn", E_syn)
        self.register_buffer("g", torch.zeros(E, device=device, dtype=dtype))

    @torch.no_grad()
    def reset_state(self): self.g.zero_()

    @torch.no_grad()
    def decay(self, dt):
        if self.E == 0: return
        self.g.mul_(torch.exp(-dt / self.tau_syn))

    @torch.no_grad()
    def deliver_spikes(self, pre_spiked):
        if self.E == 0: return
        fire_e = pre_spiked[self.edge_pre]
        self.g.add_(self.g_max * fire_e.to(self.dtype))

    @torch.no_grad()
    def post_currents(self, V_post):
        if self.E == 0:
            zeros = torch.zeros(self.n_post, device=self.device, dtype=self.dtype)
            return zeros, zeros.clone()
        V_at_post = V_post.index_select(0, self.edge_post)
        I_per_edge = self.g * (self.E_syn - V_at_post)
        I_per_post = torch.zeros(self.n_post, device=self.device, dtype=self.dtype)
        g_per_post = torch.zeros(self.n_post, device=self.device, dtype=self.dtype)
        I_per_post.index_add_(0, self.edge_post, I_per_edge)
        g_per_post.index_add_(0, self.edge_post, self.g)
        return I_per_post, g_per_post


# ============================================================
# Phase 3: spiking reservoir
# ============================================================

@dataclass
class ReservoirConfig:
    n_reservoir: int = 500
    p_recurrent: float = 0.10
    g_exc: float = 0.04
    g_inh: float = 0.30
    exc_fraction: float = 0.8
    tau_syn: float = 5.0
    noise_std: float = 0.2
    seed: int = 0


class SpikingReservoir(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        N = cfg.n_reservoir; device = torch.device(device)
        pop = LIFPopulation(N, params=NeuronParams(noise_std=cfg.noise_std), device=device)
        n_e = int(round(cfg.exc_fraction * N))
        counts = torch.distributions.Binomial(
            total_count=N - 1, probs=torch.tensor(cfg.p_recurrent, dtype=torch.float64),
        ).sample(sample_shape=(N,)).long()
        all_pre, all_post, all_kind, all_g = [], [], [], []
        for i in range(N):
            c = int(counts[i].item())
            if c == 0: continue
            mask = torch.ones(N, dtype=torch.bool); mask[i] = False
            avail = torch.arange(N)[mask]
            chosen = avail[torch.randperm(N - 1, generator=g)[:c]]
            all_pre.append(torch.full((c,), i, dtype=torch.long))
            all_post.append(chosen)
            kind = SYN_EXC if i < n_e else SYN_INH
            gmax = cfg.g_exc if kind == SYN_EXC else cfg.g_inh
            all_kind.append(torch.full((c,), kind, dtype=torch.long))
            all_g.append(torch.full((c,), gmax))
        if all_pre:
            pre = torch.cat(all_pre); post = torch.cat(all_post)
            kind = torch.cat(all_kind); gmax = torch.cat(all_g)
        else:
            pre = torch.empty(0, dtype=torch.long); post = torch.empty(0, dtype=torch.long)
            kind = torch.empty(0, dtype=torch.long); gmax = torch.empty(0)
        tau = torch.full((pre.numel(),), cfg.tau_syn)
        self.population = pop
        self.synapses = SparseSynapses(
            n_pre=N, n_post=N, edge_pre=pre, edge_post=post,
            kind=kind, g_max=gmax, tau_syn=tau, device=device,
        )
        self.n = N

    def reset(self):
        self.population.reset_state(); self.synapses.reset_state()

    @torch.no_grad()
    def step(self, input_current, dt, t):
        V = self.population.V
        I_syn, g_syn = self.synapses.post_currents(V)
        if input_current is not None: I_syn = I_syn + input_current
        spike = self.population.step(I_syn, g_syn, dt=dt, t=t)
        self.synapses.deliver_spikes(spike)
        self.synapses.decay(dt)
        return spike


# ============================================================
# Phase 5: episodic memory (with on-demand recency weighting)
# ============================================================

@dataclass
class MemoryConfig:
    capacity: int = 1000
    key_dim: int = 64
    value_dim: int = 64
    top_k: int = 1


class EpisodicMemory(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        self.cfg = cfg; device = torch.device(device)
        self.register_buffer("keys", torch.zeros(cfg.capacity, cfg.key_dim, device=device))
        self.register_buffer("values", torch.zeros(cfg.capacity, cfg.value_dim, device=device))
        self.register_buffer("ages", torch.full((cfg.capacity,), -1.0, device=device))
        self.n_stored = 0; self.t_global = 0.0

    @torch.no_grad()
    def write(self, key, value):
        slot = self.n_stored if self.n_stored < self.cfg.capacity else int(self.ages.argmin().item())
        if self.n_stored < self.cfg.capacity: self.n_stored += 1
        self.keys[slot] = key; self.values[slot] = value; self.ages[slot] = self.t_global

    @torch.no_grad()
    def read(self, query, top_k=None, recency_tau=None):
        """Cosine k-NN retrieval. If recency_tau is given, retrieved entries
        are additionally weighted by exp(-(t_now - age) / tau) BEFORE softmax."""
        if top_k is None: top_k = self.cfg.top_k
        n = self.n_stored
        keys = self.keys[:n]; vals = self.values[:n]; ages = self.ages[:n]
        q_n = F.normalize(query.unsqueeze(0), dim=-1)
        k_n = F.normalize(keys, dim=-1)
        sims = (k_n @ q_n.t()).squeeze(-1)
        if recency_tau is not None and recency_tau > 0:
            recency = torch.exp(-(self.t_global - ages) / recency_tau)
            sims = sims * recency
        top_sims, top_idx = torch.topk(sims, min(top_k, n))
        weights = F.softmax(top_sims, dim=0)
        mixed = (weights.unsqueeze(-1) * vals[top_idx]).sum(dim=0)
        return {"mixed_value": mixed, "weights": weights, "slot_ids": top_idx}

    def step_time(self, dt=1.0): self.t_global += dt


# ============================================================
# Task: 5-class temporal patterns (same as Pavlovian)
# ============================================================

@dataclass
class TaskBatch:
    inputs: torch.Tensor
    labels: torch.Tensor


def make_5class_task(n_per_class=30, T=200, noise_std=0.4, seed=0):
    n_classes = 5
    n_samples = n_classes * n_per_class
    g = torch.Generator().manual_seed(seed)
    inputs = torch.zeros(n_samples, T)
    labels = torch.zeros(n_samples, dtype=torch.long)
    t = torch.arange(T, dtype=torch.float32) * 0.001
    idx = 0
    for cls in range(n_classes):
        for _ in range(n_per_class):
            phase = torch.rand((), generator=g).item() * 2 * math.pi
            f = 3.0 + 4.0 * torch.rand((), generator=g).item()
            if cls == 0:
                wave = torch.sin(2 * math.pi * f * t + phase)
            elif cls == 1:
                wave = torch.sign(torch.sin(2 * math.pi * f * t + phase))
            elif cls == 2:
                wave = (2 / math.pi) * torch.arcsin(torch.sin(2 * math.pi * f * t + phase))
            elif cls == 3:
                wave = (2 * t / t[-1] - 1)
            else:
                env = torch.zeros(T); env[T//2 - T*15//100 : T//2 + T*15//100] = 1.0
                wave = torch.sin(2 * math.pi * f * t + phase) * env
            noise = torch.randn(T, generator=g) * noise_std
            inputs[idx] = wave + noise; labels[idx] = cls; idx += 1
    return TaskBatch(inputs=inputs, labels=labels)


def _drive_input(N, signal, device, scale=5.0, proj_seed=7777):
    g = torch.Generator().manual_seed(proj_seed)
    P = ((torch.rand(N, generator=g) - 0.5) * 2.0).to(device=device)
    return signal.to(device=device).unsqueeze(-1) * P.unsqueeze(0) * scale


def extract_features(reservoir, batch, device):
    N = reservoir.n
    out = torch.zeros(batch.inputs.shape[0], 3 * N, device=device)
    for i in range(batch.inputs.shape[0]):
        drive = _drive_input(N, batch.inputs[i], device)
        T = drive.shape[0]
        Vs = torch.zeros(T, N, device=device)
        reservoir.reset()
        for k in range(T):
            reservoir.step(drive[k], dt=1.0, t=k * 1.0)
            Vs[k] = reservoir.population.V.clone()
        late = Vs[T // 2:]
        out[i] = torch.cat([
            late.mean(dim=0), late.std(dim=0),
            late.max(dim=0).values - late.min(dim=0).values,
        ])
    return out


# ============================================================
# Agents
# ============================================================

class NaiveQAgent:
    """Linear Q + TD updates."""
    def __init__(self, feat_dim, n_actions, lr=0.05, eps=0.1, device="cpu", seed=0):
        self.W = torch.zeros(feat_dim, n_actions, device=device)
        self.lr = lr; self.eps = eps; self.n_actions = n_actions
        self.device = device
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def act(self, x):
        if torch.rand((), generator=self.rng).item() < self.eps:
            return int(torch.randint(0, self.n_actions, (1,), generator=self.rng).item())
        return int((x @ self.W).argmax().item())

    def update(self, x, a, r):
        td = r - (self.W[:, a] * x).sum()
        self.W[:, a] = self.W[:, a] + self.lr * td * x


class MemoryAgent:
    """Episodic k-NN over signed-reward entries. No recency weighting -
    the Pavlovian winner. Expected to STRUGGLE on reversal because old
    entries with the previous mapping persist and dominate retrieval."""
    def __init__(self, feat_dim, n_actions, capacity=1000, top_k=10, eps=0.1, device="cpu", seed=0):
        self.memory = EpisodicMemory(
            MemoryConfig(capacity=capacity, key_dim=feat_dim, value_dim=n_actions, top_k=top_k),
            device=device)
        self.n_actions = n_actions; self.top_k = top_k; self.eps = eps
        self.device = device
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def act(self, x):
        if self.memory.n_stored == 0 or torch.rand((), generator=self.rng).item() < self.eps:
            return int(torch.randint(0, self.n_actions, (1,), generator=self.rng).item())
        out = self.memory.read(x, top_k=min(self.top_k, self.memory.n_stored), recency_tau=None)
        return int(out["mixed_value"].argmax().item())

    def update(self, x, a, r):
        v = torch.zeros(self.n_actions, device=self.device); v[a] = float(r)
        self.memory.write(x, v); self.memory.step_time(1.0)


class TimeWeightedMemoryAgent:
    """Same as MemoryAgent but retrieval weights = cosine * exp(-age/tau).
    Old entries fade exponentially, so non-stationarity is naturally
    handled by the read function rather than by explicit forgetting."""
    def __init__(self, feat_dim, n_actions, capacity=1000, top_k=10, eps=0.1,
                 tau=50.0, device="cpu", seed=0):
        self.memory = EpisodicMemory(
            MemoryConfig(capacity=capacity, key_dim=feat_dim, value_dim=n_actions, top_k=top_k),
            device=device)
        self.n_actions = n_actions; self.top_k = top_k; self.eps = eps
        self.tau = tau; self.device = device
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def act(self, x):
        if self.memory.n_stored == 0 or torch.rand((), generator=self.rng).item() < self.eps:
            return int(torch.randint(0, self.n_actions, (1,), generator=self.rng).item())
        out = self.memory.read(x, top_k=min(self.top_k, self.memory.n_stored), recency_tau=self.tau)
        return int(out["mixed_value"].argmax().item())

    def update(self, x, a, r):
        v = torch.zeros(self.n_actions, device=self.device); v[a] = float(r)
        self.memory.write(x, v); self.memory.step_time(1.0)


# ============================================================
# Reversal experiment runner
# ============================================================

def _phase_stats(correct, phase_boundaries, recovery_window=30, recovery_threshold=0.70):
    """Per-phase analysis: mean accuracy, trials-to-recovery within phase,
    drop magnitude at the phase boundary."""
    phases = []
    starts = [0] + list(phase_boundaries[:-1])
    for p_idx, (p_start, p_end) in enumerate(zip(starts, phase_boundaries)):
        phase_correct = correct[p_start:p_end]
        mean_acc = sum(phase_correct) / max(1, len(phase_correct))
        ttr = None
        for tt in range(recovery_window, len(phase_correct)):
            if sum(phase_correct[tt - recovery_window: tt]) / recovery_window >= recovery_threshold:
                ttr = tt
                break
        if p_idx > 0:
            prev_acc = sum(correct[p_start - recovery_window: p_start]) / recovery_window
            new_acc = sum(correct[p_start: p_start + recovery_window]) / recovery_window
            drop = prev_acc - new_acc
        else:
            drop = None
        phases.append({"mean_acc": mean_acc, "ttr": ttr, "drop": drop})
    return phases


def run_reversal(
    seed: int = 0,
    phase_length: int = 150,
    n_phases: int = 4,
    n_reservoir: int = 2000,
    T: int = 200,
    n_per_class_pool: int = 30,
    noise_std: float = 0.4,
    n_actions: int = 3,
    lr: float = 0.05,
    eps: float = 0.10,
    top_k: int = 10,
    tau: float = 50.0,
    device: str = "cpu",
):
    n_trials = phase_length * n_phases
    phase_boundaries = [phase_length * (p + 1) for p in range(n_phases)]
    print(f"\n[seed {seed}] {n_trials} trials, N_res={n_reservoir}, T={T}, "
          f"{n_phases} phases x {phase_length} trials, tau={tau}", flush=True)

    pool = make_5class_task(n_per_class=n_per_class_pool, T=T, noise_std=noise_std, seed=seed)
    pool_labels = pool.labels.to(device=device)
    t0 = time.time()
    reservoir = SpikingReservoir(ReservoirConfig(n_reservoir=n_reservoir, seed=seed), device=device)
    X = extract_features(reservoir, pool, device=device)
    X = X - X.mean(dim=0, keepdim=True)
    X = F.normalize(X, dim=-1)
    feat_time = time.time() - t0
    print(f"      features: {feat_time:.1f}s, dim={X.shape[1]}", flush=True)

    # Generate a sequence of phase-specific class -> action mappings
    rng_phase = torch.Generator(device="cpu").manual_seed(seed + 90000)
    phase_maps = []
    last_map = None
    for p in range(n_phases):
        while True:
            m = torch.randint(0, n_actions, (5,), generator=rng_phase)
            # Force phase change: don't repeat the previous mapping verbatim
            if last_map is None or not torch.equal(m, last_map):
                break
        phase_maps.append(m.to(device=device))
        last_map = m

    feat_dim = X.shape[1]
    naive = NaiveQAgent(feat_dim, n_actions, lr=lr, eps=eps, device=device, seed=seed + 100)
    mem   = MemoryAgent(feat_dim, n_actions, capacity=n_trials, top_k=top_k,
                          eps=eps, device=device, seed=seed + 100)
    twm   = TimeWeightedMemoryAgent(feat_dim, n_actions, capacity=n_trials, top_k=top_k,
                                      eps=eps, tau=tau, device=device, seed=seed + 100)

    stim_rng = torch.Generator(device="cpu").manual_seed(seed + 200)
    n_samples = X.shape[0]
    n_corr = []; m_corr = []; t_corr = []
    n_rew = []; m_rew = []; t_rew = []
    t0 = time.time()
    for trial in range(n_trials):
        phase = trial // phase_length
        correct_map = phase_maps[phase]
        idx = int(torch.randint(0, n_samples, (1,), generator=stim_rng).item())
        x = X[idx]; true_cls = int(pool_labels[idx].item())
        correct_a = int(correct_map[true_cls].item())

        for agent, cor, rew in ((naive, n_corr, n_rew), (mem, m_corr, m_rew), (twm, t_corr, t_rew)):
            a = agent.act(x); r = 1.0 if a == correct_a else -1.0
            agent.update(x, a, r)
            cor.append(int(a == correct_a)); rew.append(r)
    trial_time = time.time() - t0
    print(f"      trial loop: {trial_time:.1f}s ({1000*trial_time/n_trials:.2f}ms/trial)", flush=True)

    n_stats = _phase_stats(n_corr, phase_boundaries)
    m_stats = _phase_stats(m_corr, phase_boundaries)
    t_stats = _phase_stats(t_corr, phase_boundaries)

    print(f"      Naive    cum={sum(n_rew):+5.0f}  per-phase mean_acc: "
          + " ".join(f"{s['mean_acc']*100:5.1f}%" for s in n_stats))
    print(f"      MemNoRec cum={sum(m_rew):+5.0f}  per-phase mean_acc: "
          + " ".join(f"{s['mean_acc']*100:5.1f}%" for s in m_stats))
    print(f"      MemRec   cum={sum(t_rew):+5.0f}  per-phase mean_acc: "
          + " ".join(f"{s['mean_acc']*100:5.1f}%" for s in t_stats), flush=True)

    return {
        "seed": seed,
        "phase_boundaries": phase_boundaries,
        "naive_cum": sum(n_rew),
        "mem_cum": sum(m_rew),
        "twm_cum": sum(t_rew),
        "naive_stats": n_stats,
        "mem_stats": m_stats,
        "twm_stats": t_stats,
        "naive_correct": n_corr,
        "mem_correct": m_corr,
        "twm_correct": t_corr,
    }


# ============================================================
# Multi-seed driver
# ============================================================

N_SEEDS = 5
PHASE_LENGTH = 150
N_PHASES = 4

results = []
for s in range(N_SEEDS):
    print(f"\n{'#'*60}\n# seed {s}\n{'#'*60}", flush=True)
    results.append(run_reversal(
        seed=s,
        phase_length=PHASE_LENGTH,
        n_phases=N_PHASES,
        n_reservoir=2000,
        T=200,
        n_per_class_pool=30,
        noise_std=0.4,
        n_actions=3,
        lr=0.05,
        eps=0.10,
        top_k=10,
        tau=50.0,
        device=DEVICE,
    ))


def _ms(vals):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    return m, s


n_trials_total = PHASE_LENGTH * N_PHASES
print("\n" + "=" * 70)
print(f"  REVERSAL LEARNING  ({N_SEEDS} seeds, {N_PHASES} phases x {PHASE_LENGTH} trials)")
print(f"  5 stim classes x 3 actions; mapping reshuffles each phase")
print("  " + "-" * 66)
for name, key in [("Naive Q-learning", "naive"),
                   ("Memory (no recency)", "mem"),
                   ("Memory + recency tau=50", "twm")]:
    cum_m, cum_s = _ms([r[f"{key}_cum"] for r in results])
    print(f"  {name:<24s}  cum_reward: {cum_m:+7.1f}  +/- {cum_s:5.1f}  "
          f"(max = +{n_trials_total})")

# Per-phase recovery analysis
print("  " + "-" * 66)
print(f"  Per-phase trials-to-recovery (rolling-30 acc >= 70%; '-' = never)")
print(f"  {'phase':>6} | {'naive':>14} | {'mem (no rec)':>14} | {'mem+recency':>14}")
for p_idx in range(N_PHASES):
    naive_ttrs = [r["naive_stats"][p_idx]["ttr"] for r in results]
    mem_ttrs   = [r["mem_stats"][p_idx]["ttr"]   for r in results]
    twm_ttrs   = [r["twm_stats"][p_idx]["ttr"]   for r in results]
    def fmt(ttrs):
        ttrs_f = [t if t is not None else PHASE_LENGTH for t in ttrs]
        m = statistics.mean(ttrs_f); s = statistics.stdev(ttrs_f) if len(ttrs_f) >= 2 else 0
        nn = sum(1 for t in ttrs if t is None)
        return f"{m:>5.0f} +/- {s:>4.0f}{(f' ({nn} never)' if nn else '       ')}"
    print(f"  {p_idx:>6d} | {fmt(naive_ttrs):>14s} | {fmt(mem_ttrs):>14s} | {fmt(twm_ttrs):>14s}")

# Per-phase drop analysis (only meaningful for phase > 0)
print("  " + "-" * 66)
print(f"  Per-phase DROP at phase boundary (last 30 of prev - first 30 of new)")
print(f"  {'shift':>6} | {'naive':>14} | {'mem (no rec)':>14} | {'mem+recency':>14}")
for p_idx in range(1, N_PHASES):
    nd = [r["naive_stats"][p_idx]["drop"] for r in results]
    md = [r["mem_stats"][p_idx]["drop"]   for r in results]
    td = [r["twm_stats"][p_idx]["drop"]   for r in results]
    def fmt2(ds):
        m, s = _ms(ds)
        return f"{m*100:>+5.1f}% +/- {s*100:>4.1f}"
    print(f"  {p_idx:>6d} | {fmt2(nd):>14s} | {fmt2(md):>14s} | {fmt2(td):>14s}")
print("=" * 70)

# Per-seed summary table
print("\nPer-seed cumulative reward:")
print(f"  {'seed':>4} | {'naive':>7} {'mem':>7} {'mem+rec':>7}")
for r in results:
    print(f"  {r['seed']:>4d} | {r['naive_cum']:>+7.0f} {r['mem_cum']:>+7.0f} {r['twm_cum']:>+7.0f}")
