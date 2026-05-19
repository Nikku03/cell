"""Mixed-regime reversal benchmark: 3 stationary classes + 2 reversal classes.

5 stimulus classes, but only 2 of them reshuffle their correct-action at
phase boundaries. The other 3 keep the SAME correct-action across all
4 phases. This is the regime where CLS theory predicts the cortex should
finally pay off: it can consolidate the stationary mappings into durable
storage while the hippocampus handles the reshuffling 2.

Three agents on the same reservoir features:
  - Naive Q-learning
  - MemRec (single hippo, tau=50) - current best on pure reversal
  - CLS (hippo tau=20 + neocortex Q-table, alpha=0.7)

Per-class-type metrics report accuracy on stationary classes vs reversal
classes SEPARATELY so we can see if CLS is winning the stationary subset
(its prediction) while MemRec is winning the reversal subset.
"""

# ============================================================
import math, time, statistics
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
# Phase 5: hippocampus + neocortex
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


class CortexMemory(nn.Module):
    def __init__(self, key_dim, n_actions, max_clusters=50,
                 similarity_threshold=0.6, lr=0.05, centroid_lr=0.05, device="cpu"):
        super().__init__()
        device = torch.device(device)
        self.max_clusters = max_clusters
        self.threshold = similarity_threshold
        self.lr = lr; self.centroid_lr = centroid_lr
        self.n_actions = n_actions; self.device = device
        self.register_buffer("keys", torch.zeros(max_clusters, key_dim, device=device))
        self.register_buffer("q_tables", torch.zeros(max_clusters, n_actions, device=device))
        self.register_buffer("counts", torch.zeros(max_clusters, device=device))
        self.n_clusters = 0

    def _nearest(self, x):
        if self.n_clusters == 0: return -1, -1.0
        x_n = F.normalize(x.unsqueeze(0), dim=-1)
        k_n = F.normalize(self.keys[:self.n_clusters], dim=-1)
        sims = (k_n @ x_n.t()).squeeze(-1)
        return int(sims.argmax().item()), float(sims.max().item())

    @torch.no_grad()
    def update(self, x, a, r):
        idx, sim = self._nearest(x)
        if idx == -1 or (sim < self.threshold and self.n_clusters < self.max_clusters):
            slot = self.n_clusters
            self.keys[slot] = x; self.q_tables[slot] = 0.0
            self.q_tables[slot, a] = float(r); self.counts[slot] = 1
            self.n_clusters += 1
            return slot
        self.keys[idx] = (1 - self.centroid_lr) * self.keys[idx] + self.centroid_lr * x
        self.q_tables[idx, a] = (1 - self.lr) * self.q_tables[idx, a] + self.lr * float(r)
        self.counts[idx] += 1
        return idx

    @torch.no_grad()
    def read(self, x):
        if self.n_clusters == 0:
            return torch.zeros(self.n_actions, device=self.device)
        idx, _ = self._nearest(x)
        return self.q_tables[idx].clone()


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
# Mixed regime: stationary + reversal class -> action mappings
# ============================================================

def make_mixed_phase_maps(n_phases, n_classes=5, n_actions=3, n_stationary=3, seed=0):
    """Build phase maps where n_stationary classes have FIXED correct_action
    across all phases and (n_classes - n_stationary) classes have a fresh
    correct_action each phase (different from the previous phase).
    Returns (list of phase_maps [each (n_classes,) tensor], is_stationary (n_classes,) bool)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    stationary_idx = torch.randperm(n_classes, generator=g)[:n_stationary]
    is_stationary = torch.zeros(n_classes, dtype=torch.bool)
    is_stationary[stationary_idx] = True
    fixed_actions = torch.randint(0, n_actions, (n_classes,), generator=g)
    phase_maps = []
    for p in range(n_phases):
        m = fixed_actions.clone()
        for cls in range(n_classes):
            if not is_stationary[cls]:
                while True:
                    a = int(torch.randint(0, n_actions, (1,), generator=g).item())
                    if p == 0 or a != int(phase_maps[p - 1][cls].item()):
                        m[cls] = a; break
        phase_maps.append(m)
    return phase_maps, is_stationary


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


class CLSAgent:
    def __init__(self, feat_dim, n_actions, hippo_capacity=200, hippo_tau=20.0,
                 hippo_top_k=10, cortex_clusters=50, cortex_threshold=0.6,
                 cortex_lr=0.05, alpha=0.7, eps=0.1, device="cpu", seed=0):
        self.hippo = EpisodicMemory(MemoryConfig(
            capacity=hippo_capacity, key_dim=feat_dim,
            value_dim=n_actions, top_k=hippo_top_k,
        ), device=device)
        self.cortex = CortexMemory(
            key_dim=feat_dim, n_actions=n_actions, max_clusters=cortex_clusters,
            similarity_threshold=cortex_threshold, lr=cortex_lr,
            centroid_lr=0.05, device=device,
        )
        self.n_actions = n_actions; self.hippo_top_k = hippo_top_k
        self.hippo_tau = hippo_tau; self.alpha = alpha; self.eps = eps
        self.device = device
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def act(self, x):
        if torch.rand((), generator=self.rng).item() < self.eps:
            return int(torch.randint(0, self.n_actions, (1,), generator=self.rng).item())
        if self.hippo.n_stored == 0 and self.cortex.n_clusters == 0:
            return int(torch.randint(0, self.n_actions, (1,), generator=self.rng).item())
        q_hippo = (self.hippo.read(x, top_k=min(self.hippo_top_k, self.hippo.n_stored),
                                     recency_tau=self.hippo_tau)["mixed_value"]
                   if self.hippo.n_stored > 0
                   else torch.zeros(self.n_actions, device=self.device))
        q_cortex = self.cortex.read(x)
        q_blend = self.alpha * q_hippo + (1 - self.alpha) * q_cortex
        return int(q_blend.argmax().item())

    def update(self, x, a, r):
        v = torch.zeros(self.n_actions, device=self.device); v[a] = float(r)
        self.hippo.write(x, v); self.hippo.step_time(1.0)
        self.cortex.update(x, a, r)


# ============================================================
# Runner
# ============================================================

def run_mixed_regime(
    seed=0, phase_length=150, n_phases=4, n_stationary=3,
    n_reservoir=2000, T=200, n_per_class_pool=30,
    noise_std=0.4, n_actions=3,
    lr=0.05, eps=0.10, memrec_tau=50.0,
    cls_hippo_tau=20.0, cls_alpha=0.7,
    cls_cortex_threshold=0.6, cls_cortex_lr=0.05,
    device="cpu",
):
    n_trials = phase_length * n_phases
    n_classes = 5
    n_reversal = n_classes - n_stationary
    print(f"\n[seed {seed}] {n_trials} trials, N_res={n_reservoir}, "
          f"{n_phases}p x {phase_length}t, "
          f"{n_stationary} stationary + {n_reversal} reversal classes", flush=True)

    pool = make_5class_task(n_per_class=n_per_class_pool, T=T, noise_std=noise_std, seed=seed)
    pool_labels = pool.labels.to(device=device)
    t0 = time.time()
    reservoir = SpikingReservoir(ReservoirConfig(n_reservoir=n_reservoir, seed=seed), device=device)
    X = extract_features(reservoir, pool, device=device)
    X = X - X.mean(dim=0, keepdim=True)
    X = F.normalize(X, dim=-1)
    feat_time = time.time() - t0
    print(f"      features: {feat_time:.1f}s, dim={X.shape[1]}", flush=True)

    phase_maps, is_stationary = make_mixed_phase_maps(
        n_phases=n_phases, n_classes=n_classes, n_actions=n_actions,
        n_stationary=n_stationary, seed=seed + 90000,
    )
    phase_maps_dev = [m.to(device=device) for m in phase_maps]
    is_stationary_dev = is_stationary.to(device=device)
    print(f"      stationary mask: {is_stationary.tolist()}; "
          f"fixed actions: {phase_maps[0][is_stationary].tolist()}", flush=True)
    print(f"      phase 0 map: {phase_maps[0].tolist()}")
    print(f"      phase 1 map: {phase_maps[1].tolist()}")
    print(f"      phase 2 map: {phase_maps[2].tolist()}")
    print(f"      phase 3 map: {phase_maps[3].tolist()}")

    feat_dim = X.shape[1]
    naive  = NaiveQAgent(feat_dim, n_actions, lr=lr, eps=eps, device=device, seed=seed + 100)
    memrec = TimeWeightedMemoryAgent(feat_dim, n_actions, capacity=n_trials, top_k=10,
                                       eps=eps, tau=memrec_tau, device=device, seed=seed + 100)
    cls    = CLSAgent(feat_dim, n_actions, hippo_capacity=200,
                        hippo_tau=cls_hippo_tau, hippo_top_k=10,
                        cortex_clusters=50, cortex_threshold=cls_cortex_threshold,
                        cortex_lr=cls_cortex_lr, alpha=cls_alpha,
                        eps=eps, device=device, seed=seed + 100)

    stim_rng = torch.Generator(device="cpu").manual_seed(seed + 200)
    n_samples = X.shape[0]
    # Track per-class-type accuracy: (agent, class-type)
    counts = {("naive", "stat"): 0, ("naive", "rev"): 0,
              ("memrec", "stat"): 0, ("memrec", "rev"): 0,
              ("cls", "stat"): 0, ("cls", "rev"): 0}
    correct = {("naive", "stat"): 0, ("naive", "rev"): 0,
                ("memrec", "stat"): 0, ("memrec", "rev"): 0,
                ("cls", "stat"): 0, ("cls", "rev"): 0}
    cum_reward = {"naive": 0.0, "memrec": 0.0, "cls": 0.0}

    # Also track per-(agent, phase) and per-(agent, phase, class-type) accuracy
    phase_type_correct = {(name, p, ct): 0
                           for name in ("naive", "memrec", "cls")
                           for p in range(n_phases) for ct in ("stat", "rev")}
    phase_type_total = {(name, p, ct): 0
                         for name in ("naive", "memrec", "cls")
                         for p in range(n_phases) for ct in ("stat", "rev")}

    t0 = time.time()
    for trial in range(n_trials):
        phase = trial // phase_length
        cmap = phase_maps_dev[phase]
        idx = int(torch.randint(0, n_samples, (1,), generator=stim_rng).item())
        x = X[idx]; true_cls = int(pool_labels[idx].item())
        correct_a = int(cmap[true_cls].item())
        ct = "stat" if bool(is_stationary[true_cls]) else "rev"

        for (name, agent) in [("naive", naive), ("memrec", memrec), ("cls", cls)]:
            a = agent.act(x); r = 1.0 if a == correct_a else -1.0
            agent.update(x, a, r)
            cum_reward[name] += r
            counts[(name, ct)] += 1
            correct[(name, ct)] += int(a == correct_a)
            phase_type_total[(name, phase, ct)] += 1
            phase_type_correct[(name, phase, ct)] += int(a == correct_a)
    trial_time = time.time() - t0
    print(f"      trial loop: {trial_time:.1f}s ({1000*trial_time/n_trials:.2f}ms/trial); "
          f"cortex_clusters={cls.cortex.n_clusters}", flush=True)

    # Per-class-type accuracies (over full run)
    res_summary = {}
    for name in ("naive", "memrec", "cls"):
        for ct in ("stat", "rev"):
            tot = max(1, counts[(name, ct)])
            res_summary[(name, ct, "acc")] = correct[(name, ct)] / tot

    print(f"      Naive   stat={res_summary[('naive','stat','acc')]*100:5.1f}%  "
          f"rev={res_summary[('naive','rev','acc')]*100:5.1f}%  cum={cum_reward['naive']:+.0f}")
    print(f"      MemRec  stat={res_summary[('memrec','stat','acc')]*100:5.1f}%  "
          f"rev={res_summary[('memrec','rev','acc')]*100:5.1f}%  cum={cum_reward['memrec']:+.0f}")
    print(f"      CLS     stat={res_summary[('cls','stat','acc')]*100:5.1f}%  "
          f"rev={res_summary[('cls','rev','acc')]*100:5.1f}%  cum={cum_reward['cls']:+.0f}",
          flush=True)

    return {
        "seed": seed,
        "is_stationary": is_stationary.tolist(),
        "cum_reward": dict(cum_reward),
        "summary": dict(res_summary),
        "phase_type_correct": dict(phase_type_correct),
        "phase_type_total": dict(phase_type_total),
        "cortex_clusters": cls.cortex.n_clusters,
    }


# ============================================================
# Multi-seed driver
# ============================================================

N_SEEDS = 5
PHASE_LENGTH = 150
N_PHASES = 4
N_STATIONARY = 3

results = []
for s in range(N_SEEDS):
    print(f"\n{'#'*60}\n# seed {s}\n{'#'*60}", flush=True)
    results.append(run_mixed_regime(
        seed=s, phase_length=PHASE_LENGTH, n_phases=N_PHASES,
        n_stationary=N_STATIONARY,
        n_reservoir=2000, T=200, n_per_class_pool=30,
        noise_std=0.4, n_actions=3, lr=0.05, eps=0.10,
        memrec_tau=50.0, cls_hippo_tau=20.0, cls_alpha=0.7,
        cls_cortex_threshold=0.6, cls_cortex_lr=0.05,
        device=DEVICE,
    ))


def _ms(vals):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    return m, s


n_trials_total = PHASE_LENGTH * N_PHASES
print("\n" + "=" * 76)
print(f"  MIXED REGIME  ({N_SEEDS} seeds, {N_PHASES}p x {PHASE_LENGTH}t, "
      f"{N_STATIONARY} stationary + {5 - N_STATIONARY} reversal classes)")
print("  " + "-" * 72)
print(f"  {'agent':<10s} | {'cum reward':>14} | {'stationary acc':>14} | {'reversal acc':>14}")

for name in ("naive", "memrec", "cls"):
    cum_m, cum_s = _ms([r["cum_reward"][name] for r in results])
    sa_m, sa_s = _ms([r["summary"][(name, "stat", "acc")] for r in results])
    ra_m, ra_s = _ms([r["summary"][(name, "rev",  "acc")] for r in results])
    pretty = {"naive": "Naive", "memrec": "MemRec", "cls": "CLS"}[name]
    print(f"  {pretty:<10s} | {cum_m:>+7.1f} +/- {cum_s:>4.1f} "
          f"| {sa_m*100:>5.1f}% +/- {sa_s*100:>4.1f} "
          f"| {ra_m*100:>5.1f}% +/- {ra_s*100:>4.1f}")

print("  " + "-" * 72)
print("  Per-phase accuracy on STATIONARY classes (does CLS pull ahead over phases?)")
print(f"  {'phase':>6} | {'naive':>14} | {'memrec':>14} | {'cls':>14}")
for p in range(N_PHASES):
    rows = []
    for name in ("naive", "memrec", "cls"):
        cs = [r["phase_type_correct"][(name, p, "stat")] / max(1, r["phase_type_total"][(name, p, "stat")])
              for r in results]
        m, s = _ms(cs)
        rows.append(f"{m*100:>5.1f}% +/- {s*100:>4.1f}")
    print(f"  {p:>6d} | {rows[0]:>14s} | {rows[1]:>14s} | {rows[2]:>14s}")

print("  " + "-" * 72)
print("  Per-phase accuracy on REVERSAL classes (recovery test)")
print(f"  {'phase':>6} | {'naive':>14} | {'memrec':>14} | {'cls':>14}")
for p in range(N_PHASES):
    rows = []
    for name in ("naive", "memrec", "cls"):
        cs = [r["phase_type_correct"][(name, p, "rev")] / max(1, r["phase_type_total"][(name, p, "rev")])
              for r in results]
        m, s = _ms(cs)
        rows.append(f"{m*100:>5.1f}% +/- {s*100:>4.1f}")
    print(f"  {p:>6d} | {rows[0]:>14s} | {rows[1]:>14s} | {rows[2]:>14s}")

print("=" * 76)
print("\nPer-seed cumulative reward:")
print(f"  {'seed':>4} | {'naive':>7} {'memrec':>7} {'cls':>7}")
for r in results:
    print(f"  {r['seed']:>4d} | {r['cum_reward']['naive']:>+7.0f} "
          f"{r['cum_reward']['memrec']:>+7.0f} {r['cum_reward']['cls']:>+7.0f}")
