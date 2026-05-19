"""Self-contained Colab cell: Pavlovian reward/shock conditioning.

Task:
  - 5 stimulus classes (temporal waveform patterns)
  - 3 possible actions per trial
  - Random class -> correct-action mapping fixed at experiment start
  - Each trial: see stimulus -> pick action -> get +1 (correct) or -1 (wrong)
  - Cumulative reward over 500 trials is the headline metric

Two agents, same reservoir features:
  - NAIVE       : linear Q(s,a) = W_a^T phi(s), TD updates, epsilon-greedy
  - INTEGRATED  : episodic memory of (feature -> action) for rewarded trials,
                  cosine k-NN retrieval, epsilon-greedy

Multi-seed (5 seeds, 500 trials each). Expected runtime on A100: ~5 min
(feature extraction dominates; trial loop is cheap).
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
# Phase 5: episodic memory
# ============================================================

@dataclass
class MemoryConfig:
    capacity: int = 1000
    key_dim: int = 64
    value_dim: int = 64
    top_k: int = 1
    similarity: str = "cosine"


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
    def read(self, query, top_k=None):
        if top_k is None: top_k = self.cfg.top_k
        n = self.n_stored
        keys = self.keys[:n]
        q_n = F.normalize(query.unsqueeze(0), dim=-1)
        k_n = F.normalize(keys, dim=-1)
        sims = (k_n @ q_n.t()).squeeze(-1)
        top_sims, top_idx = torch.topk(sims, min(top_k, n))
        weights = F.softmax(top_sims, dim=0)
        values_topk = self.values[top_idx]
        mixed = (weights.unsqueeze(-1) * values_topk).sum(dim=0)
        return {"mixed_value": mixed, "weights": weights, "slot_ids": top_idx}

    def step_time(self, dt=1.0): self.t_global += dt


# ============================================================
# 5-class task (subset of the 20-class one: 5 shapes at amp=1.0)
# ============================================================

@dataclass
class TaskBatch:
    inputs: torch.Tensor
    labels: torch.Tensor


def make_5class_task(n_per_class=30, T=200, noise_std=0.4, seed=0):
    """5 waveform shapes at amplitude 1.0: sine, square, triangle, ramp, burst."""
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
            amp = 1.0
            if cls == 0:
                wave = amp * torch.sin(2 * math.pi * f * t + phase)
            elif cls == 1:
                wave = amp * torch.sign(torch.sin(2 * math.pi * f * t + phase))
            elif cls == 2:
                wave = amp * (2 / math.pi) * torch.arcsin(torch.sin(2 * math.pi * f * t + phase))
            elif cls == 3:
                wave = amp * (2 * t / t[-1] - 1)
            else:
                env = torch.zeros(T); env[T//2 - T*15//100 : T//2 + T*15//100] = 1.0
                wave = amp * torch.sin(2 * math.pi * f * t + phase) * env
            noise = torch.randn(T, generator=g) * noise_std
            inputs[idx] = wave + noise
            labels[idx] = cls
            idx += 1
    return TaskBatch(inputs=inputs, labels=labels)


# ============================================================
# Reservoir feature extraction
# ============================================================

def _drive_input(N, signal, device, scale=5.0, proj_seed=7777):
    g = torch.Generator().manual_seed(proj_seed)
    P = ((torch.rand(N, generator=g) - 0.5) * 2.0).to(device=device)
    return signal.to(device=device).unsqueeze(-1) * P.unsqueeze(0) * scale


def extract_features(reservoir, batch, device):
    """(n_samples, 3*N) [mean, std, range] of late-half membrane V."""
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
    """Linear Q(s, a) = W_a^T phi(s), TD updates W_a += alpha * (r - Q(s,a)) * phi(s).

    eps-greedy action selection. This is the standard online linear-RL baseline.
    """

    def __init__(self, feat_dim, n_actions, lr=0.05, eps=0.1, device="cpu", seed=0):
        self.W = torch.zeros(feat_dim, n_actions, device=device)
        self.lr = lr; self.eps = eps; self.n_actions = n_actions
        self.device = device
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def act(self, x):
        if torch.rand((), generator=self.rng).item() < self.eps:
            return int(torch.randint(0, self.n_actions, (1,), generator=self.rng).item())
        Q = x @ self.W  # (n_actions,)
        return int(Q.argmax().item())

    def update(self, x, a, r):
        q_sa = (self.W[:, a] * x).sum()
        td = r - q_sa
        self.W[:, a] = self.W[:, a] + self.lr * td * x


class IntegratedMemoryAgent:
    """Episodic memory of (feature -> per-action signed reward).

    For each trial we store a value vector of length n_actions, with the
    sign of the reward placed in the chosen action's slot (rewarded trials
    get +1 at action a, shocked trials get -1 at action a; other slots stay
    zero). Retrieval is cosine k-NN: pull the top-k most-similar past
    trials, sum their value vectors weighted by similarity (via softmax),
    and pick the argmax. The negative imprint matters - without it, k-NN
    retrieves wrong-class entries from confusable features and votes the
    'whatever was rewarded' action, which is exactly what tanked the
    initial 'rewarded-only' variant.
    """

    def __init__(self, feat_dim, n_actions, capacity=1000, top_k=5, eps=0.1, device="cpu", seed=0):
        self.memory = EpisodicMemory(MemoryConfig(
            capacity=capacity, key_dim=feat_dim, value_dim=n_actions,
            similarity="cosine", top_k=top_k,
        ), device=device)
        self.n_actions = n_actions; self.top_k = top_k; self.eps = eps
        self.device = device
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def act(self, x):
        if (self.memory.n_stored == 0
                or torch.rand((), generator=self.rng).item() < self.eps):
            return int(torch.randint(0, self.n_actions, (1,), generator=self.rng).item())
        out = self.memory.read(x, top_k=min(self.top_k, self.memory.n_stored))
        return int(out["mixed_value"].argmax().item())

    def update(self, x, a, r):
        # Store EVERY trial; the chosen action gets the signed reward (+1/-1),
        # other action slots stay zero (no info about untried actions for that key)
        v = torch.zeros(self.n_actions, device=self.device)
        v[a] = float(r)
        self.memory.write(x, v)
        self.memory.step_time(1.0)


# ============================================================
# Experiment runner
# ============================================================

@dataclass
class RLResult:
    seed: int
    n_trials: int
    naive_cum_reward: float
    integ_cum_reward: float
    naive_final_acc: float            # rolling-50 acc at end of run
    integ_final_acc: float
    naive_tcc: int | None             # trials-to-criterion (rolling-50 acc >= 0.80), or None
    integ_tcc: int | None
    naive_rewards: list[float]
    integ_rewards: list[float]
    naive_correct: list[int]
    integ_correct: list[int]


def run_rl_experiment(
    seed: int = 0,
    n_trials: int = 500,
    criterion_window: int = 50,
    criterion_acc: float = 0.80,
    n_reservoir: int = 2000,
    T: int = 200,
    n_per_class_pool: int = 30,
    noise_std: float = 0.4,
    n_actions: int = 3,
    lr: float = 0.05,
    eps: float = 0.10,
    top_k: int = 5,
    device: str = "cpu",
) -> RLResult:
    print(f"\n[seed {seed}] {n_trials} trials, N_res={n_reservoir}, T={T}, "
          f"pool={5*n_per_class_pool} stimuli, eps={eps}", flush=True)

    # Pool of stimuli: 5 classes x n_per_class_pool samples
    pool = make_5class_task(n_per_class=n_per_class_pool, T=T,
                             noise_std=noise_std, seed=seed)
    pool_labels = pool.labels.to(device=device)

    # Reservoir + feature extraction
    t0 = time.time()
    reservoir = SpikingReservoir(ReservoirConfig(n_reservoir=n_reservoir, seed=seed), device=device)
    X = extract_features(reservoir, pool, device=device)
    feat_mean = X.mean(dim=0, keepdim=True)
    X = X - feat_mean                   # centering (from previous diagnostics)
    X = F.normalize(X, dim=-1)          # L2-normalise so cosine == dot for memory
    feat_time = time.time() - t0
    print(f"      features: {feat_time:.1f}s, dim={X.shape[1]}, "
          f"||mean||={feat_mean.norm().item():.1f}", flush=True)

    # Random class -> correct-action mapping (the "task")
    rng = torch.Generator(device="cpu").manual_seed(seed + 50000)
    correct_action = torch.randint(0, n_actions, (5,), generator=rng).to(device=device)
    n_classes = 5

    # Agents
    feat_dim = X.shape[1]
    naive = NaiveQAgent(feat_dim, n_actions, lr=lr, eps=eps,
                          device=device, seed=seed + 100)
    integ = IntegratedMemoryAgent(feat_dim, n_actions, capacity=n_trials, top_k=top_k,
                                    eps=eps, device=device, seed=seed + 100)

    # Stimulus sampling RNG (shared so both agents see SAME stimulus sequence)
    stim_rng = torch.Generator(device="cpu").manual_seed(seed + 200)
    n_samples = X.shape[0]

    naive_rewards, integ_rewards = [], []
    naive_correct, integ_correct = [], []

    t0 = time.time()
    for trial in range(n_trials):
        idx = int(torch.randint(0, n_samples, (1,), generator=stim_rng).item())
        x = X[idx]
        true_cls = int(pool_labels[idx].item())
        correct_a = int(correct_action[true_cls].item())

        a_n = naive.act(x); r_n = 1.0 if a_n == correct_a else -1.0
        naive.update(x, a_n, r_n)
        naive_rewards.append(r_n); naive_correct.append(int(a_n == correct_a))

        a_i = integ.act(x); r_i = 1.0 if a_i == correct_a else -1.0
        integ.update(x, a_i, r_i)
        integ_rewards.append(r_i); integ_correct.append(int(a_i == correct_a))
    trial_time = time.time() - t0

    def tcc(correct):
        for tt in range(criterion_window, len(correct)):
            if sum(correct[tt - criterion_window: tt]) / criterion_window >= criterion_acc:
                return tt
        return None

    n_tcc = tcc(naive_correct); i_tcc = tcc(integ_correct)
    n_final = sum(naive_correct[-criterion_window:]) / criterion_window
    i_final = sum(integ_correct[-criterion_window:]) / criterion_window
    n_cum = sum(naive_rewards); i_cum = sum(integ_rewards)

    print(f"      trial loop: {trial_time:.1f}s "
          f"({1000*trial_time/n_trials:.1f}ms/trial)", flush=True)
    print(f"      NAIVE      cum_reward={n_cum:+.0f}  final_acc={n_final*100:5.1f}%  "
          f"TCC={n_tcc if n_tcc is not None else 'never'}", flush=True)
    print(f"      INTEGRATED cum_reward={i_cum:+.0f}  final_acc={i_final*100:5.1f}%  "
          f"TCC={i_tcc if i_tcc is not None else 'never'}", flush=True)

    return RLResult(
        seed=seed, n_trials=n_trials,
        naive_cum_reward=n_cum, integ_cum_reward=i_cum,
        naive_final_acc=n_final, integ_final_acc=i_final,
        naive_tcc=n_tcc, integ_tcc=i_tcc,
        naive_rewards=naive_rewards, integ_rewards=integ_rewards,
        naive_correct=naive_correct, integ_correct=integ_correct,
    )


# ============================================================
# Multi-seed driver
# ============================================================

N_SEEDS = 5
N_TRIALS = 500

results = []
for s in range(N_SEEDS):
    print(f"\n{'#' * 60}\n# seed {s}\n{'#' * 60}", flush=True)
    results.append(run_rl_experiment(
        seed=s,
        n_trials=N_TRIALS,
        n_reservoir=2000,
        T=200,
        n_per_class_pool=30,
        noise_std=0.4,
        n_actions=3,
        lr=0.05,
        eps=0.10,
        top_k=5,
        device=DEVICE,
    ))


def _ms(values):
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) >= 2 else 0.0
    return m, s


n_cum_m, n_cum_s = _ms([r.naive_cum_reward for r in results])
i_cum_m, i_cum_s = _ms([r.integ_cum_reward for r in results])
n_fin_m, n_fin_s = _ms([r.naive_final_acc for r in results])
i_fin_m, i_fin_s = _ms([r.integ_final_acc for r in results])

n_tccs = [r.naive_tcc if r.naive_tcc is not None else N_TRIALS for r in results]
i_tccs = [r.integ_tcc if r.integ_tcc is not None else N_TRIALS for r in results]
n_tcc_m, n_tcc_s = _ms(n_tccs)
i_tcc_m, i_tcc_s = _ms(i_tccs)

print("\n" + "=" * 64)
print(f"  PAVLOVIAN CONDITIONING  ({N_SEEDS} seeds, {N_TRIALS} trials each)")
print(f"  5 stim classes x 3 actions; random class->action map per seed")
print("  " + "-" * 60)
print(f"  Naive Q-learning")
print(f"    cumulative reward    : {n_cum_m:+7.1f}  +/- {n_cum_s:.1f}    (max = +{N_TRIALS})")
print(f"    rolling-50 final acc : {n_fin_m*100:6.2f}%  +/- {n_fin_s*100:.2f}")
print(f"    trials-to-criterion  : {n_tcc_m:6.1f}   +/- {n_tcc_s:.1f}")
print("  " + "-" * 60)
print(f"  Integrated memory")
print(f"    cumulative reward    : {i_cum_m:+7.1f}  +/- {i_cum_s:.1f}    (max = +{N_TRIALS})")
print(f"    rolling-50 final acc : {i_fin_m*100:6.2f}%  +/- {i_fin_s*100:.2f}")
print(f"    trials-to-criterion  : {i_tcc_m:6.1f}   +/- {i_tcc_s:.1f}")
print("  " + "-" * 60)
print(f"  GAP   cumulative reward: {i_cum_m - n_cum_m:+.1f}")
print(f"  GAP   final accuracy   : {(i_fin_m - n_fin_m)*100:+.2f}pp")
print(f"  GAP   TCC (fewer=faster): {i_tcc_m - n_tcc_m:+.1f} trials")
print("=" * 64)

print("\nPer-seed breakdown:")
print(f"  {'seed':>4} | {'n_cum':>7} {'i_cum':>7} | "
      f"{'n_fin':>6} {'i_fin':>6} | {'n_tcc':>6} {'i_tcc':>6}")
for r in results:
    n_tcc_str = str(r.naive_tcc) if r.naive_tcc is not None else "never"
    i_tcc_str = str(r.integ_tcc) if r.integ_tcc is not None else "never"
    print(f"  {r.seed:>4d} | {r.naive_cum_reward:>+7.0f} {r.integ_cum_reward:>+7.0f} | "
          f"{r.naive_final_acc*100:>5.1f}% {r.integ_final_acc*100:>5.1f}% | "
          f"{n_tcc_str:>6} {i_tcc_str:>6}")
