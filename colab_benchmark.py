"""Self-contained Colab cell: snn_scaling composite benchmark at N=2000, T=200.

Copy this entire file into a single Colab cell. Runs on CPU; auto-uses GPU
if available. Expected runtime:
  - 8-core CPU: ~3-5 minutes
  - T4 GPU:     ~1-2 minutes
"""

# ============================================================
# Imports
# ============================================================
import math
import time
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
        self.n = n
        self.device = torch.device(device)
        self.dtype = dtype
        self.register_buffer("V", torch.full((n,), params.V_rest, dtype=dtype, device=self.device))
        self.register_buffer("refr", torch.zeros(n, dtype=dtype, device=self.device))
        self.register_buffer("spiked", torch.zeros(n, dtype=torch.bool, device=self.device))
        for name in ("tau_m", "V_rest", "V_thresh", "V_reset", "R_m", "refractory", "noise_std"):
            self.register_buffer(f"_{name}", torch.full((n,), float(getattr(params, name)), dtype=dtype, device=self.device))

    def set_heterogeneous(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, f"_{k}", v.to(device=self.device, dtype=self.dtype))

    @torch.no_grad()
    def reset_state(self):
        self.V.fill_(0.0)
        self.V += self._V_rest
        self.refr.zero_()
        self.spiked.zero_()

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
            noise = torch.randn_like(V_new) * self._noise_std * math.sqrt(dt)
            V_new = V_new + noise
        V_new = torch.where(was_refr, self._V_reset, V_new)
        spike = (V_new >= self._V_thresh) & (~was_refr)
        V_new = torch.where(spike, self._V_reset, V_new)
        self.refr = torch.where(spike, self._refractory, self.refr)
        self.V = V_new
        self.spiked = spike
        return spike


SYN_EXC, SYN_INH = 0, 1
_E_SYN = {SYN_EXC: 0.0, SYN_INH: -80.0}


class SparseSynapses(nn.Module):
    def __init__(self, n_pre, n_post, edge_pre, edge_post, kind, g_max, tau_syn=5.0,
                 device="cpu", dtype=torch.float32):
        super().__init__()
        device = torch.device(device)
        E = edge_pre.numel()
        self.n_pre, self.n_post, self.E = n_pre, n_post, E
        self.device, self.dtype = device, dtype
        self.register_buffer("edge_pre", edge_pre.to(device=device, dtype=torch.long))
        self.register_buffer("edge_post", edge_post.to(device=device, dtype=torch.long))
        self.register_buffer("kind", kind.to(device=device, dtype=torch.long))
        self.register_buffer("g_max", g_max.to(device=device, dtype=dtype))
        if isinstance(tau_syn, torch.Tensor):
            tau_t = tau_syn.to(device=device, dtype=dtype)
        else:
            tau_t = torch.full((E,), float(tau_syn), device=device, dtype=dtype)
        self.register_buffer("tau_syn", tau_t)
        E_syn = torch.where(
            self.kind == SYN_EXC,
            torch.full((E,), _E_SYN[SYN_EXC], device=device, dtype=dtype),
            torch.full((E,), _E_SYN[SYN_INH], device=device, dtype=dtype),
        )
        self.register_buffer("E_syn", E_syn)
        self.register_buffer("g", torch.zeros(E, device=device, dtype=dtype))

    @torch.no_grad()
    def reset_state(self):
        self.g.zero_()

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
        N = cfg.n_reservoir
        device = torch.device(device)
        pop = LIFPopulation(N, params=NeuronParams(noise_std=cfg.noise_std), device=device)
        n_e = int(round(cfg.exc_fraction * N))
        counts = torch.distributions.Binomial(
            total_count=N - 1, probs=torch.tensor(cfg.p_recurrent, dtype=torch.float64),
        ).sample(sample_shape=(N,)).long()
        all_pre, all_post, all_kind, all_g = [], [], [], []
        for i in range(N):
            c = int(counts[i].item())
            if c == 0: continue
            mask = torch.ones(N, dtype=torch.bool)
            mask[i] = False
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
        self.population.reset_state()
        self.synapses.reset_state()

    @torch.no_grad()
    def step(self, input_current, dt, t):
        V = self.population.V
        I_syn, g_syn = self.synapses.post_currents(V)
        if input_current is not None:
            I_syn = I_syn + input_current
        spike = self.population.step(I_syn, g_syn, dt=dt, t=t)
        self.synapses.deliver_spikes(spike)
        self.synapses.decay(dt)
        return spike


def train_linear_readout(features, targets, ridge=1e-2):
    if targets.dim() == 1:
        targets = targets.unsqueeze(-1)
    X = features
    y = targets.to(dtype=X.dtype)
    F_dim = X.shape[1]
    XtX = X.t() @ X
    reg = ridge * torch.eye(F_dim, dtype=X.dtype, device=X.device)
    XtY = X.t() @ y
    W = torch.linalg.solve(XtX + reg, XtY)
    return W


def predict_linear(W, features):
    return features @ W


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
        self.cfg = cfg
        device = torch.device(device)
        self.register_buffer("keys", torch.zeros(cfg.capacity, cfg.key_dim, device=device))
        self.register_buffer("values", torch.zeros(cfg.capacity, cfg.value_dim, device=device))
        self.register_buffer("ages", torch.full((cfg.capacity,), -1.0, device=device))
        self.n_stored = 0
        self.t_global = 0.0

    @torch.no_grad()
    def write(self, key, value):
        if self.n_stored < self.cfg.capacity:
            slot = self.n_stored
            self.n_stored += 1
        else:
            slot = int(self.ages.argmin().item())
        self.keys[slot] = key
        self.values[slot] = value
        self.ages[slot] = self.t_global
        return slot

    @torch.no_grad()
    def read(self, query, top_k=None):
        if top_k is None: top_k = self.cfg.top_k
        n = self.n_stored
        keys = self.keys[:n]
        if self.cfg.similarity == "cosine":
            q_n = F.normalize(query.unsqueeze(0), dim=-1)
            k_n = F.normalize(keys, dim=-1)
            sims = (k_n @ q_n.t()).squeeze(-1)
        else:
            sims = keys @ query
        top_sims, top_idx = torch.topk(sims, min(top_k, n))
        weights = F.softmax(top_sims, dim=0)
        values_topk = self.values[top_idx]
        mixed = (weights.unsqueeze(-1) * values_topk).sum(dim=0)
        return {"mixed_value": mixed, "slot_ids": top_idx, "weights": weights}

    def step_time(self, dt=1.0):
        self.t_global += dt


# ============================================================
# 20-class temporal-pattern task
# ============================================================

@dataclass
class TaskBatch:
    inputs: torch.Tensor
    labels: torch.Tensor


def make_pattern_task(n_classes=20, n_per_class=1, T=200, dt_ms=1.0, noise_std=0.4, seed=0):
    """20 classes = 5 shapes x 4 amplitudes. Each sample has random phase
    and base frequency 3-7 Hz, plus additive Gaussian noise."""
    if n_classes != 20:
        raise ValueError("fixed to 20 classes")
    n_samples = n_classes * n_per_class
    g = torch.Generator().manual_seed(seed)
    inputs = torch.zeros(n_samples, T)
    labels = torch.zeros(n_samples, dtype=torch.long)
    t = torch.arange(T, dtype=torch.float32) * dt_ms / 1000.0
    amps = [0.5, 0.7, 0.9, 1.1]
    idx = 0
    for cls in range(n_classes):
        shape_idx = cls % 5
        amp_idx = cls // 5
        amp = amps[amp_idx]
        for _ in range(n_per_class):
            phase = torch.rand((), generator=g).item() * 2 * math.pi
            f = 3.0 + 4.0 * torch.rand((), generator=g).item()
            if shape_idx == 0:
                wave = amp * torch.sin(2 * math.pi * f * t + phase)
            elif shape_idx == 1:
                wave = amp * torch.sign(torch.sin(2 * math.pi * f * t + phase))
            elif shape_idx == 2:
                wave = amp * (2 / math.pi) * torch.arcsin(torch.sin(2 * math.pi * f * t + phase))
            elif shape_idx == 3:
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
    """Return (n_samples, 3*N) [mean, std, range] of late-half membrane V."""
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
        feat = torch.cat([
            late.mean(dim=0),
            late.std(dim=0),
            late.max(dim=0).values - late.min(dim=0).values,
        ])
        out[i] = feat
    return out


# ============================================================
# Phase 7: dendritic random projection
# ============================================================

class DendriticFeatureProjection(nn.Module):
    def __init__(self, n_in, n_out, n_dendrites=4, sparsity=0.04,
                 gain=4.0, threshold=0.5, seed=0, device="cpu"):
        super().__init__()
        self.K = n_dendrites; self.gain = gain; self.threshold = threshold
        g = torch.Generator().manual_seed(seed)
        mask = (torch.rand(n_dendrites, n_out, n_in, generator=g) < sparsity).float()
        weights = torch.randn(n_dendrites, n_out, n_in, generator=g) * mask
        active = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        norm = weights.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / norm * torch.sqrt(active)
        self.register_buffer("weights", weights.to(device=device))

    @torch.no_grad()
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        d_input = torch.einsum("kon,bn->bko", self.weights, x)
        d_out = torch.sigmoid((d_input - self.threshold) * self.gain)
        return d_out.sum(dim=1)


def kwta_top_k(x, k):
    if x.dim() == 1: x = x.unsqueeze(0)
    k = min(k, x.shape[-1])
    topk_vals, topk_idx = torch.topk(x, k, dim=-1)
    out = torch.zeros_like(x)
    out.scatter_(-1, topk_idx, topk_vals)
    return out


def one_hot(labels, n_classes):
    return F.one_hot(labels, num_classes=n_classes).float()


# ============================================================
# The benchmark
# ============================================================

@dataclass
class BenchmarkResult:
    naive_acc: float
    integrated_acc: float
    gap_pp: float
    n_reservoir: int
    T: int
    n_train: int
    n_test: int
    feature_extraction_time_s: float
    naive_decision_time_s: float
    integrated_decision_time_s: float


def run_benchmark(n_reservoir=2000, T=200,
                  n_per_class_train=6, n_per_class_test=6,
                  noise_std=0.4, seed=0,
                  n_dendrite_out=800, n_dendrites=4, k_active=100,
                  ridge=1.0, device="cpu"):
    """Run NAIVE (ridge readout) vs INTEGRATED (centroid memory) at the
    requested shot count. The integrated system computes a PER-CLASS
    CENTROID across all training samples for that class, then stores
    one denoised prototype per class. This is the architectural
    advantage of the memory-based approach when shots > 1: noisy
    individual samples get averaged into a clean class signature."""
    n_train = 20 * n_per_class_train
    n_test = 20 * n_per_class_test
    print(f"\n[benchmark] N_reservoir={n_reservoir}, T={T}, "
          f"train={n_train} ({n_per_class_train}/class), "
          f"test={n_test} ({n_per_class_test}/class), device={device}", flush=True)

    print("[1/4] Building 20-class task...", flush=True)
    train = make_pattern_task(n_per_class=n_per_class_train, T=T, noise_std=noise_std, seed=seed)
    test = make_pattern_task(n_per_class=n_per_class_test, T=T, noise_std=noise_std, seed=seed + 99)

    print(f"[2/4] Building reservoir (N={n_reservoir})...", flush=True)
    t0 = time.time()
    reservoir = SpikingReservoir(ReservoirConfig(n_reservoir=n_reservoir, seed=seed), device=device)
    print(f"      reservoir built in {time.time()-t0:.1f}s "
          f"({reservoir.synapses.E} recurrent edges)", flush=True)

    print(f"[3/4] Extracting features for {train.inputs.shape[0]} + {test.inputs.shape[0]} samples...", flush=True)
    t0 = time.time()
    X_train = extract_features(reservoir, train, device)
    X_test = extract_features(reservoir, test, device)
    feat_time = time.time() - t0
    print(f"      done in {feat_time:.1f}s; feature dim = {X_train.shape[1]}", flush=True)

    train_labels = train.labels.to(device=device)
    test_labels = test.labels.to(device=device)
    n_classes = 20

    print("[4/4] NAIVE (reservoir + ridge) vs INTEGRATED (dendrites + kWTA + centroid memory)...", flush=True)

    # Naive: ridge regression on all training samples
    t0 = time.time()
    Y = one_hot(train_labels, n_classes)
    W = train_linear_readout(X_train, Y, ridge=ridge)
    naive_pred = predict_linear(W, X_test).argmax(dim=-1)
    naive_t = time.time() - t0
    naive_acc = (naive_pred == test_labels).float().mean().item()
    print(f"      NAIVE      accuracy = {naive_acc*100:.2f}%   ({naive_t*1000:.1f}ms decision)", flush=True)

    # Integrated: dendritic projection -> kWTA -> per-class CENTROID -> memory -> cosine top-1
    t0 = time.time()
    dendritic = DendriticFeatureProjection(
        n_in=X_train.shape[1], n_out=n_dendrite_out,
        n_dendrites=n_dendrites, seed=seed + 100, device=device,
    )
    Z_train = kwta_top_k(dendritic.forward(X_train), k_active)
    Z_test = kwta_top_k(dendritic.forward(X_test), k_active)

    # Build one centroid per class from the training samples (this is what
    # makes multi-shot meaningfully different from 1-shot for the memory)
    centroids = torch.zeros(n_classes, Z_train.shape[1], device=device)
    for c in range(n_classes):
        mask = (train_labels == c)
        if mask.any():
            centroids[c] = Z_train[mask].mean(dim=0)

    memory = EpisodicMemory(MemoryConfig(
        capacity=n_classes, key_dim=n_dendrite_out,
        value_dim=n_classes, similarity="cosine", top_k=1,
    ), device=device)
    for c in range(n_classes):
        memory.write(centroids[c], F.one_hot(torch.tensor(c, device=device), n_classes).float())
        memory.step_time(1.0)

    integrated_pred = torch.zeros(Z_test.shape[0], dtype=torch.long, device=device)
    for i, z in enumerate(Z_test):
        out = memory.read(z, top_k=1)
        integrated_pred[i] = int(out["mixed_value"].argmax().item())
    integrated_t = time.time() - t0
    integrated_acc = (integrated_pred == test_labels).float().mean().item()
    print(f"      INTEGRATED accuracy = {integrated_acc*100:.2f}%   ({integrated_t*1000:.1f}ms decision)", flush=True)

    # Diagnostic: how separable are the centroids?
    if Z_train.shape[0] >= n_classes:
        cn = F.normalize(centroids, dim=-1)
        sim_matrix = cn @ cn.t()
        off_diag = sim_matrix - torch.eye(n_classes, device=device)
        max_cross = off_diag.max().item()
        mean_cross = (off_diag.sum() / (n_classes * (n_classes - 1))).item()
        print(f"      [diag] centroid pairwise cosine: mean={mean_cross:.3f}, max={max_cross:.3f}", flush=True)

    gap = (integrated_acc - naive_acc) * 100
    print(f"\n[result] GAP: {gap:+.1f}pp  (integrated - naive at same neuron budget)", flush=True)

    return BenchmarkResult(
        naive_acc=naive_acc,
        integrated_acc=integrated_acc,
        gap_pp=gap,
        n_reservoir=n_reservoir,
        T=T,
        n_train=train.inputs.shape[0],
        n_test=test.inputs.shape[0],
        feature_extraction_time_s=feat_time,
        naive_decision_time_s=naive_t,
        integrated_decision_time_s=integrated_t,
    )


# ============================================================
# RUN
# ============================================================

# 110 train + 110 test ≈ 5-6 samples per class (20 classes), interpreted as
# 6/class for symmetry. With multiple shots, the integrated system can
# compute per-class CENTROIDS (denoised prototypes), which is its real win
# over single-sample storage.
result = run_benchmark(
    n_reservoir=2000,
    T=200,
    n_per_class_train=6,    # 120 train samples
    n_per_class_test=6,     # 120 test samples
    noise_std=0.4,
    seed=0,
    n_dendrite_out=800,
    n_dendrites=4,
    k_active=100,
    ridge=1.0,
    device=DEVICE,
)

print("\n" + "=" * 60)
print(f"  N reservoir: {result.n_reservoir}")
print(f"  T timesteps: {result.T}")
print(f"  Train: {result.n_train} (6/class), Test: {result.n_test} (6/class)")
print(f"  Feature extraction: {result.feature_extraction_time_s:.1f}s")
print(f"  Naive      : {result.naive_acc*100:6.2f}%   (chance = 5%)")
print(f"  Integrated : {result.integrated_acc*100:6.2f}%")
print(f"  GAP        : {result.gap_pp:+.2f}pp")
print("=" * 60)
