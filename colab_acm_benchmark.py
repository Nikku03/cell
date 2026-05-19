"""Self-contained Colab cell: Actor-Critic-Memory on the reversal task.

Same reversal-learning setup as colab_reversal_benchmark.py. Three agents:

  Naive Q-learning              : linear Q + TD updates (baseline)
  MemRec (single bank, tau=50)  : the prior reversal winner
  ACM   (Actor-Critic-Memory)   : same hippo as MemRec PLUS a critic
                                   that adds:
                                    - counterfactual Q (all-time mean
                                      per action, no recency)
                                    - per-action recent reward VARIANCE
                                    - adaptive exploration: epsilon
                                      auto-rises when variance is high
                                      (i.e. after a phase shift, when
                                      old- and new-phase rewards both
                                      sit in recent memory)
                                    - blended decision:
                                       q = alpha*q_A + (1-alpha)*q_B
                                           - beta * sigma_B

The Actor's hippocampus uses tau=50 (SAME as MemRec) so the only
difference between MemRec and ACM is the Critic's contribution and the
adaptive epsilon. That isolates the new mechanism.

Predicted advantage of ACM over MemRec:
  - faster phase-1+ recovery (variance spikes -> more exploration -> agent
    breaks old policy quickly)
  - lower variance across seeds (critic adds stability)
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
# Phase 0: LIF + sparse synapses
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
# Phase 5: episodic memory (with recency-weighted read)
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
# 5-class task + feature extraction
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
            if cls == 0: wave = torch.sin(2*math.pi*f*t + phase)
            elif cls == 1: wave = torch.sign(torch.sin(2*math.pi*f*t + phase))
            elif cls == 2: wave = (2/math.pi)*torch.arcsin(torch.sin(2*math.pi*f*t + phase))
            elif cls == 3: wave = (2*t/t[-1] - 1)
            else:
                env = torch.zeros(T); env[T//2 - T*15//100 : T//2 + T*15//100] = 1.0
                wave = torch.sin(2*math.pi*f*t + phase) * env
            inputs[idx] = wave + torch.randn(T, generator=g) * noise_std
            labels[idx] = cls; idx += 1
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


class TimeWeightedMemoryAgent:
    """Single hippo bank, recency tau=50."""
    def __init__(self, feat_dim, n_actions, capacity=1000, top_k=10, eps=0.1,
                 tau=50.0, device="cpu", seed=0):
        self.memory = EpisodicMemory(
            MemoryConfig(capacity=capacity, key_dim=feat_dim, value_dim=n_actions, top_k=top_k),
            device=device)
        self.n_actions = n_actions; self.top_k = top_k; self.eps = eps; self.tau = tau
        self.device = device
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def act(self, x):
        if self.memory.n_stored == 0 or torch.rand((), generator=self.rng).item() < self.eps:
            return int(torch.randint(0, self.n_actions, (1,), generator=self.rng).item())
        out = self.memory.read(x, top_k=min(self.top_k, self.memory.n_stored), recency_tau=self.tau)
        return int(out["mixed_value"].argmax().item())

    def update(self, x, a, r):
        v = torch.zeros(self.n_actions, device=self.device); v[a] = float(r)
        self.memory.write(x, v); self.memory.step_time(1.0)


class ActorCriticMemoryAgent:
    """Actor (recency-weighted hippo) + Critic (medium-recency per-action Q +
    recent per-action variance) + adaptive exploration.

    Critic Q uses a LONGER recency (critic_tau) than the actor - it provides
    a medium-term view that the actor's faster recency might overshoot.
    Pure 'all-time' critic was tried in v1 of the smoke; it carries too
    much stale phase-0 info to be useful past phase 1.

    sigma_B defaults to 0 (NOT 1) for actions without recent same-action
    evidence, so adaptive eps doesn't get inflated by untried actions.
    """

    def __init__(self, feat_dim, n_actions, capacity=1000,
                 actor_tau=50.0, actor_top_k=10,
                 critic_tau=150.0, critic_top_k=30, recent_window=30,
                 alpha=0.7, beta=0.2, gamma=0.3,
                 eps_base=0.05, eps_max=0.4,
                 device="cpu", seed=0):
        self.memory = EpisodicMemory(
            MemoryConfig(capacity=capacity, key_dim=feat_dim, value_dim=n_actions, top_k=actor_top_k),
            device=device)
        self.n_actions = n_actions
        self.actor_tau = actor_tau
        self.actor_top_k = actor_top_k
        self.critic_tau = critic_tau
        self.critic_top_k = critic_top_k
        self.recent_window = recent_window
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps_base = eps_base
        self.eps_max = eps_max
        self.device = device
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

        # Bookkeeping for diagnostics
        self.last_eps = eps_base
        self.last_disagree = 0
        self.disagree_total = 0
        self.decisions = 0

    @torch.no_grad()
    def _critic_evaluate(self, x):
        """Return (q_B, sigma_B) of shape (n_actions,).

        q_B[a]:   medium-recency mean reward when action a was taken from a
                  state similar to x. critic_tau weights so old entries fade
                  slower than the actor's but still fade.
        sigma_B[a]: variance of rewards for action a within recent_window
                    trials. Set to 0 (not 1) if fewer than 2 such trials, so
                    adaptive epsilon does not get inflated by untried actions.
        """
        n = self.memory.n_stored
        q_B = torch.zeros(self.n_actions, device=self.device)
        sigma_B = torch.zeros(self.n_actions, device=self.device)
        if n == 0:
            return q_B, sigma_B

        keys = self.memory.keys[:n]
        vals = self.memory.values[:n]
        ages = self.memory.ages[:n]
        t_now = self.memory.t_global

        # Cosine similarity to current state
        x_n = F.normalize(x.unsqueeze(0), dim=-1)
        k_n = F.normalize(keys, dim=-1)
        sims = (k_n @ x_n.t()).squeeze(-1)

        # Critic recency weight (slower than the actor's)
        recency_weight = torch.exp(-(t_now - ages) / self.critic_tau)

        for a in range(self.n_actions):
            mask_a = vals[:, a] != 0
            if not mask_a.any():
                continue
            sims_a = sims[mask_a]
            vals_a = vals[mask_a, a]
            ages_a = ages[mask_a]
            recency_a = recency_weight[mask_a]

            # q_B: weighted mean, similarity * critic_recency
            combined = sims_a * recency_a
            k = min(self.critic_top_k, combined.numel())
            top_combined, top_idx = torch.topk(combined, k)
            weights = F.softmax(top_combined, dim=0)
            q_B[a] = (weights * vals_a[top_idx]).sum()

            # sigma_B: only if we have >=2 recent same-action trials
            recent_mask = (t_now - ages_a) < self.recent_window
            if int(recent_mask.sum().item()) >= 2:
                sigma_B[a] = vals_a[recent_mask].var(unbiased=False)
            # else keep 0 (no signal -> no exploration push)
        return q_B, sigma_B

    @torch.no_grad()
    def act(self, x):
        n = self.memory.n_stored
        if n == 0:
            self.last_eps = self.eps_max
            return int(torch.randint(0, self.n_actions, (1,), generator=self.rng).item())

        # Actor: recency-weighted hippo retrieval (same as MemRec)
        actor_out = self.memory.read(x, top_k=min(self.actor_top_k, n), recency_tau=self.actor_tau)
        q_A = actor_out["mixed_value"]
        a_A = int(q_A.argmax().item())

        # Critic
        q_B, sigma_B = self._critic_evaluate(x)
        a_B = int(q_B.argmax().item())

        # Adaptive exploration: epsilon scales with mean uncertainty
        mean_sigma = sigma_B.mean().item()
        eps_eff = min(self.eps_base + self.gamma * mean_sigma, self.eps_max)
        self.last_eps = eps_eff

        if a_A != a_B:
            self.disagree_total += 1
            self.last_disagree = 1
        else:
            self.last_disagree = 0
        self.decisions += 1

        if torch.rand((), generator=self.rng).item() < eps_eff:
            return int(torch.randint(0, self.n_actions, (1,), generator=self.rng).item())

        # Blended decision
        q_final = self.alpha * q_A + (1 - self.alpha) * q_B - self.beta * sigma_B
        return int(q_final.argmax().item())

    @torch.no_grad()
    def update(self, x, a, r):
        v = torch.zeros(self.n_actions, device=self.device); v[a] = float(r)
        self.memory.write(x, v); self.memory.step_time(1.0)


# ============================================================
# Reversal experiment runner
# ============================================================

def _phase_stats(correct, phase_boundaries, recovery_window=30, recovery_threshold=0.70):
    phases = []
    starts = [0] + list(phase_boundaries[:-1])
    for p_idx, (p_start, p_end) in enumerate(zip(starts, phase_boundaries)):
        phase_correct = correct[p_start:p_end]
        mean_acc = sum(phase_correct) / max(1, len(phase_correct))
        ttr = None
        for tt in range(recovery_window, len(phase_correct)):
            if sum(phase_correct[tt - recovery_window: tt]) / recovery_window >= recovery_threshold:
                ttr = tt; break
        if p_idx > 0:
            prev_acc = sum(correct[p_start - recovery_window: p_start]) / recovery_window
            new_acc = sum(correct[p_start: p_start + recovery_window]) / recovery_window
            drop = prev_acc - new_acc
        else:
            drop = None
        phases.append({"mean_acc": mean_acc, "ttr": ttr, "drop": drop})
    return phases


def run_reversal_acm(
    seed: int = 0, phase_length: int = 150, n_phases: int = 4,
    n_reservoir: int = 2000, T: int = 200, n_per_class_pool: int = 30,
    noise_std: float = 0.4, n_actions: int = 3,
    lr: float = 0.05, eps: float = 0.10,
    memrec_tau: float = 50.0,
    acm_actor_tau: float = 50.0, acm_critic_tau: float = 150.0,
    acm_alpha: float = 0.7, acm_beta: float = 0.2, acm_gamma: float = 0.3,
    acm_eps_base: float = 0.05, acm_eps_max: float = 0.4,
    acm_recent_window: int = 30,
    device: str = "cpu",
):
    n_trials = phase_length * n_phases
    phase_boundaries = [phase_length * (p + 1) for p in range(n_phases)]
    print(f"\n[seed {seed}] {n_trials} trials, N_res={n_reservoir}, "
          f"{n_phases}p x {phase_length}t, ACM alpha={acm_alpha} beta={acm_beta} "
          f"gamma={acm_gamma}", flush=True)

    pool = make_5class_task(n_per_class=n_per_class_pool, T=T, noise_std=noise_std, seed=seed)
    pool_labels = pool.labels.to(device=device)
    t0 = time.time()
    reservoir = SpikingReservoir(ReservoirConfig(n_reservoir=n_reservoir, seed=seed), device=device)
    X = extract_features(reservoir, pool, device=device)
    X = X - X.mean(dim=0, keepdim=True)
    X = F.normalize(X, dim=-1)
    feat_time = time.time() - t0
    print(f"      features: {feat_time:.1f}s, dim={X.shape[1]}", flush=True)

    rng_phase = torch.Generator(device="cpu").manual_seed(seed + 90000)
    phase_maps = []; last_map = None
    for p in range(n_phases):
        while True:
            m = torch.randint(0, n_actions, (5,), generator=rng_phase)
            if last_map is None or not torch.equal(m, last_map): break
        phase_maps.append(m.to(device=device)); last_map = m

    feat_dim = X.shape[1]
    naive  = NaiveQAgent(feat_dim, n_actions, lr=lr, eps=eps, device=device, seed=seed + 100)
    memrec = TimeWeightedMemoryAgent(feat_dim, n_actions, capacity=n_trials, top_k=10,
                                       eps=eps, tau=memrec_tau, device=device, seed=seed + 100)
    acm    = ActorCriticMemoryAgent(feat_dim, n_actions, capacity=n_trials,
                                      actor_tau=acm_actor_tau, actor_top_k=10,
                                      critic_tau=acm_critic_tau,
                                      critic_top_k=30, recent_window=acm_recent_window,
                                      alpha=acm_alpha, beta=acm_beta, gamma=acm_gamma,
                                      eps_base=acm_eps_base, eps_max=acm_eps_max,
                                      device=device, seed=seed + 100)

    stim_rng = torch.Generator(device="cpu").manual_seed(seed + 200)
    n_samples = X.shape[0]
    n_cor, m_cor, a_cor = [], [], []
    n_rew, m_rew, a_rew = [], [], []
    # Track ACM diagnostics per trial
    acm_eps_trace = []
    t0 = time.time()
    for trial in range(n_trials):
        phase = trial // phase_length
        correct_map = phase_maps[phase]
        idx = int(torch.randint(0, n_samples, (1,), generator=stim_rng).item())
        x = X[idx]; true_cls = int(pool_labels[idx].item())
        correct_a = int(correct_map[true_cls].item())

        for agent, cor, rew in ((naive, n_cor, n_rew), (memrec, m_cor, m_rew), (acm, a_cor, a_rew)):
            a = agent.act(x); r = 1.0 if a == correct_a else -1.0
            agent.update(x, a, r)
            cor.append(int(a == correct_a)); rew.append(r)
        acm_eps_trace.append(acm.last_eps)
    trial_time = time.time() - t0
    print(f"      trial loop: {trial_time:.1f}s ({1000*trial_time/n_trials:.2f}ms/trial); "
          f"ACM disagreements: {acm.disagree_total}/{acm.decisions} "
          f"({100*acm.disagree_total/max(1,acm.decisions):.1f}%)", flush=True)

    n_stats = _phase_stats(n_cor, phase_boundaries)
    m_stats = _phase_stats(m_cor, phase_boundaries)
    a_stats = _phase_stats(a_cor, phase_boundaries)

    print(f"      Naive    cum={sum(n_rew):+5.0f}  per-phase: "
          + " ".join(f"{s['mean_acc']*100:5.1f}%" for s in n_stats))
    print(f"      MemRec   cum={sum(m_rew):+5.0f}  per-phase: "
          + " ".join(f"{s['mean_acc']*100:5.1f}%" for s in m_stats))
    print(f"      ACM      cum={sum(a_rew):+5.0f}  per-phase: "
          + " ".join(f"{s['mean_acc']*100:5.1f}%" for s in a_stats), flush=True)

    # Mean eps per phase (does adaptive eps actually spike at shifts?)
    eps_per_phase = []
    for p_idx in range(n_phases):
        p_start = p_idx * phase_length; p_end = (p_idx + 1) * phase_length
        eps_per_phase.append(sum(acm_eps_trace[p_start:p_end]) / max(1, p_end - p_start))
    eps_first_30 = []
    for p_idx in range(n_phases):
        p_start = p_idx * phase_length
        eps_first_30.append(sum(acm_eps_trace[p_start: p_start + 30]) / 30)
    print(f"      ACM mean eps per phase: " + " ".join(f"{e:.3f}" for e in eps_per_phase))
    print(f"      ACM eps first-30 of each phase: " + " ".join(f"{e:.3f}" for e in eps_first_30),
          flush=True)

    return {
        "seed": seed,
        "phase_boundaries": phase_boundaries,
        "naive_cum": sum(n_rew), "memrec_cum": sum(m_rew), "acm_cum": sum(a_rew),
        "naive_stats": n_stats, "memrec_stats": m_stats, "acm_stats": a_stats,
        "acm_disagree_pct": 100 * acm.disagree_total / max(1, acm.decisions),
        "acm_eps_per_phase": eps_per_phase,
        "acm_eps_first_30": eps_first_30,
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
    results.append(run_reversal_acm(
        seed=s, phase_length=PHASE_LENGTH, n_phases=N_PHASES,
        n_reservoir=2000, T=200, n_per_class_pool=30,
        noise_std=0.4, n_actions=3, lr=0.05, eps=0.10,
        memrec_tau=50.0,
        acm_actor_tau=50.0, acm_critic_tau=150.0,
        acm_alpha=0.7, acm_beta=0.2, acm_gamma=0.3,
        acm_eps_base=0.05, acm_eps_max=0.4,
        acm_recent_window=30,
        device=DEVICE,
    ))


def _ms(vals):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    return m, s


n_trials_total = PHASE_LENGTH * N_PHASES
print("\n" + "=" * 72)
print(f"  REVERSAL LEARNING WITH ACM  ({N_SEEDS} seeds, {N_PHASES}p x {PHASE_LENGTH}t)")
print("  " + "-" * 68)
for name, key in [("Naive Q-learning",            "naive"),
                   ("MemRec (single hippo, tau=50)", "memrec"),
                   ("ACM (actor + critic + adaptive eps)", "acm")]:
    cum_m, cum_s = _ms([r[f"{key}_cum"] for r in results])
    print(f"  {name:<40s}  cum: {cum_m:+7.1f}  +/- {cum_s:5.1f}  (max +{n_trials_total})")

dm, ds = _ms([r["acm_disagree_pct"] for r in results])
print(f"  ACM actor/critic disagreement rate              {dm:5.1f}% +/- {ds:.1f}")

print("  " + "-" * 68)
print(f"  Per-phase TTR (rolling-30 acc >= 70%; '-' = never)")
print(f"  {'phase':>6} | {'naive':>20} | {'memrec':>20} | {'ACM':>20}")
for p_idx in range(N_PHASES):
    n_ttrs = [r["naive_stats"][p_idx]["ttr"] for r in results]
    m_ttrs = [r["memrec_stats"][p_idx]["ttr"] for r in results]
    a_ttrs = [r["acm_stats"][p_idx]["ttr"] for r in results]
    def fmt(ttrs):
        ttrs_f = [t if t is not None else PHASE_LENGTH for t in ttrs]
        m = statistics.mean(ttrs_f); s = statistics.stdev(ttrs_f) if len(ttrs_f) >= 2 else 0
        nn = sum(1 for t in ttrs if t is None)
        return f"{m:>3.0f}+/-{s:>3.0f} ({nn} never)"
    print(f"  {p_idx:>6d} | {fmt(n_ttrs):>20s} | {fmt(m_ttrs):>20s} | {fmt(a_ttrs):>20s}")

print("  " + "-" * 68)
print(f"  Per-phase mean accuracy")
print(f"  {'phase':>6} | {'naive':>20} | {'memrec':>20} | {'ACM':>20}")
for p_idx in range(N_PHASES):
    n_a = [r["naive_stats"][p_idx]["mean_acc"] for r in results]
    m_a = [r["memrec_stats"][p_idx]["mean_acc"] for r in results]
    a_a = [r["acm_stats"][p_idx]["mean_acc"] for r in results]
    def fmt2(accs):
        m, s = _ms(accs)
        return f"{m*100:>5.1f}% +/- {s*100:>4.1f}    "
    print(f"  {p_idx:>6d} | {fmt2(n_a):>20s} | {fmt2(m_a):>20s} | {fmt2(a_a):>20s}")

print("  " + "-" * 68)
print(f"  ACM eps in first 30 trials of each phase (does it spike at shifts?)")
print(f"  {'phase':>6} | {'mean_eps_in_30':>14}")
for p_idx in range(N_PHASES):
    ep = [r["acm_eps_first_30"][p_idx] for r in results]
    em, es = _ms(ep)
    print(f"  {p_idx:>6d} | {em:>.3f} +/- {es:.3f}")
print("=" * 72)

print("\nPer-seed cumulative reward:")
print(f"  {'seed':>4} | {'naive':>7} {'memrec':>7} {'ACM':>7}")
for r in results:
    print(f"  {r['seed']:>4d} | {r['naive_cum']:>+7.0f} {r['memrec_cum']:>+7.0f} {r['acm_cum']:>+7.0f}")
