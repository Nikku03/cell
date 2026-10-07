"""ADRN-1 -- Adaptive Dendritic Reasoning Network, the minimal executable core (spec sections 17-18).

WHAT THIS IS. A replacement for the point-neuron / transformer primitive. Each unit is a stateful
dendritic neuron: several nonlinear branches, a soma, adaptation, calcium, and synapses that carry six
state variables rather than one scalar. Computation unfolds over T internal steps rather than one forward
pass. Learning combines a local eligibility trace with a global modulatory vector.

WHAT IS IN, per spec section 17:
    reduced multi-branch dendritic neurons      NMDA-like branch nonlinearity, per-branch transfer psi
    recurrent spiking dynamics                  surrogate-gradient spikes, dynamic threshold
    eligibility traces                          pre x post x dendritic-event, decaying
    vector neuromodulation                      5-channel M(t), per-synapse receptor vectors q
    fast and slow synaptic weights              W_f decays; W_s consolidates behind a calcium tag
    firing-rate homeostasis                     threshold tracks a target rate
    sparse structural pruning                   utility U drives structural availability A
    simple episodic memory                      top-K retrieval injected as apical current
    a prediction head                           self-supervised next-state prediction

WHAT IS DELIBERATELY OUT, also per section 17: ion-channel kinetics, glia, gene expression, literal
metabolism, neurogenesis, all-pairs structural growth, language.

THE TWO-LOOP DESIGN (section 13, mode D). The inner loop is local and biological: eligibility, modulation,
fast/slow weights, homeostasis, pruning. It runs with NO gradient. The outer loop trains only the slow
developmental parameters -- branch time constants, receptor sensitivities, target rates, plasticity rates,
encoder/decoder projections -- by ordinary backprop through the unrolled dynamics with a surrogate spike
gradient. `local_only=True` disables the outer loop entirely so the two can be separated in ablations,
which spec section 22 requires as baselines 8 and 9.

HONEST NOTE ON WHAT THIS CAN AND CANNOT SHOW. Every mechanism here is about TIME: eligibility exists to
bridge delayed credit, fast/slow weights exist to separate timescales, adaptive halting exists to spend
more steps on harder cases. On a static i.i.d. tabular task all of that machinery is inert by construction,
and a null there would say nothing about the primitive. That is why `adrn_control_suite.py` exists and must
be run FIRST: without a positive control on the temporal tasks the spec itself specifies (section 21), a
null on any other task cannot be distinguished from a broken implementation.

INTEGRATION. Continuous-time ODEs are integrated by explicit Euler with dt folded into the rate constants,
so every `tau` below is "steps to decay by 1/e" and every state update is a first-order recurrence. This is
the standard reduction and it is stated rather than hidden: none of the numbers this produces are claims
about real membrane physics.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid derivative backward (Zenke-Ganguli style surrogate).

    The spike is a hard 0/1 in the forward pass -- the dynamics must be genuinely event-driven or the
    energy proxy is a lie -- while the backward pass uses a smooth surrogate so the OUTER loop can train
    developmental parameters through the unrolled dynamics. The inner local rules never touch this.
    """

    @staticmethod
    def forward(ctx, v, beta):
        ctx.save_for_backward(v)
        ctx.beta = beta
        return (v >= 0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad):
        (v,) = ctx.saved_tensors
        return grad / (ctx.beta * v.abs() + 1.0) ** 2, None


def spike(v, beta=10.0):
    return SurrogateSpike.apply(v, beta)


class ADRNLayer(nn.Module):
    """One population of ADRN neurons with dendritic branches and full synaptic state.

    Shapes, with `Bt` the batch:
        W_f, W_s, A, U   (N, B, I)     per-synapse fast / slow / structural / utility
        q                (N, 5)        receptor vector onto the 5-channel modulatory signal
        V_d              (Bt, N, B)    dendritic branch voltage
        V_s, a, r_bar    (Bt, N)       soma voltage, adaptation, mean rate
        C                (Bt, N, B)    calcium
        e                (Bt, N, B, I) eligibility -- the largest tensor here, and why B and I stay small
    """

    N_MOD = 5  # reward, novelty, attention, uncertainty, error

    def __init__(self, n_in, n_neurons, n_branch=4, *, tau_d=4.0, tau_s=8.0, tau_a=20.0, tau_e=15.0,
                 tau_c=12.0, tau_r=100.0, theta_nmda=0.55, g_nmda=1.2, kappa_nmda=6.0,
                 target_rate=0.08, eta_fast=0.03, eta_slow=0.004, lam_fast=0.02, lam_slow=1e-4,
                 eta_theta=0.01, theta_tag=0.55, prune_every=25, theta_prune=0.02,
                 learn_dynamics=True, dendrites=True, fast_slow=True, structural=True):
        super().__init__()
        self.N, self.B, self.I = n_neurons, (n_branch if dendrites else 1), n_in
        self.dendrites, self.fast_slow, self.structural = dendrites, fast_slow, structural
        self.prune_every, self.theta_prune = prune_every, theta_prune
        self.theta_tag = theta_tag

        # ---- DEVELOPMENTAL PARAMETERS (outer loop trains these; spec section 13 mode D) ----
        def dev(x, shape):
            t = torch.full(shape, float(x))
            return nn.Parameter(t, requires_grad=learn_dynamics)
        # stored as inverse-softplus so the trained value stays positive without a clamp
        self.log_tau_d = dev(math.log(math.expm1(tau_d)), (self.B,))
        self.log_tau_s = dev(math.log(math.expm1(tau_s)), (1,))
        self.log_tau_a = dev(math.log(math.expm1(tau_a)), (1,))
        self.log_tau_c = dev(math.log(math.expm1(tau_c)), (1,))
        self.psi_gain = dev(1.0, (self.B,))          # per-branch transfer slope (sub/supra-linear)
        self.psi_bias = dev(0.0, (self.B,))
        self.g_branch = dev(2.0, (self.N, self.B))
        self.g_nmda = dev(g_nmda, (self.B,))
        self.theta_nmda = dev(theta_nmda, (self.B,))
        self.kappa_nmda = float(kappa_nmda)
        self.q = nn.Parameter(torch.randn(self.N, self.N_MOD) * 0.3, requires_grad=learn_dynamics)
        self.eta_fast, self.eta_slow = eta_fast, eta_slow
        self.lam_fast, self.lam_slow = lam_fast, lam_slow
        self.eta_theta, self.target_rate = eta_theta, target_rate
        self.tau_e, self.tau_r = tau_e, tau_r
        self.rho_a = 1.0

        # ---- SLOW STRUCTURAL / SYNAPTIC STATE (inner loop, no gradient) ----
        w0 = torch.randn(self.N, self.B, self.I) / math.sqrt(self.I)
        self.register_buffer("W_s", w0)
        self.register_buffer("W_f", torch.zeros_like(w0))
        self.register_buffer("A", torch.ones_like(w0))
        self.register_buffer("U", torch.zeros_like(w0))
        # theta starts LOW. At theta=1.0 the soma (which settles around 0.05 with these time
        # constants) never reaches it, the network is silent, and eligibility/plasticity/
        # halting all become no-ops while still "running". Homeostasis then regulates it.
        self.register_buffer("theta", torch.full((n_neurons,), 0.25))
        self.register_buffer("prune_count", torch.zeros(self.N, self.B, self.I))
        self.register_buffer("step_count", torch.zeros(()))
        # W_in is the one ordinary trainable projection: it is the "developmental" wiring the outer loop
        # is allowed to shape. The plastic part is W_f/W_s and is updated locally only.
        self.W_in = nn.Parameter(w0.clone() * 0.5, requires_grad=learn_dynamics)

    # ---------------------------------------------------------------- state
    def init_state(self, batch, device):
        z = lambda *s: torch.zeros(*s, device=device)
        return {"V_d": z(batch, self.N, self.B), "V_s": z(batch, self.N),
                "a": z(batch, self.N), "C": z(batch, self.N, self.B),
                "e": z(batch, self.N, self.B, self.I), "r_bar": z(batch, self.N),
                "x_pre": z(batch, self.I), "x_post": z(batch, self.N),
                "n_spk": z(()), "n_syn": z(()), "n_dend": z(())}

    def w_eff(self):
        """A * (W_s + W_f) + W_in. Structural availability gates the plastic part."""
        plastic = self.W_s + self.W_f if self.fast_slow else self.W_s
        return self.A * plastic + self.W_in

    # ---------------------------------------------------------------- one step
    def forward(self, s_in, st, m_vec=None, apical=None):
        """One internal timestep.

        s_in    (Bt, I)      presynaptic activity (spike trace or continuous drive)
        st                   state dict from init_state
        m_vec   (Bt, 5)      modulatory vector M(t); None disables plasticity this step
        apical  (Bt, N)      episodic-memory / context current injected at the soma
        """
        Bt = s_in.shape[0]
        tau_d = F.softplus(self.log_tau_d).clamp(1.0, 200.0)
        tau_s = F.softplus(self.log_tau_s).clamp(1.0, 200.0)
        tau_a = F.softplus(self.log_tau_a).clamp(1.0, 500.0)
        tau_c = F.softplus(self.log_tau_c).clamp(1.0, 500.0)

        # --- dendritic compartments: synaptic drive + NMDA-like local nonlinearity ---
        drive = torch.einsum("bi,nji->bnj", s_in, self.w_eff())          # (Bt, N, B)
        nmda_on = torch.sigmoid(self.kappa_nmda * (st["V_d"] - self.theta_nmda))
        i_nmda = self.g_nmda * nmda_on * (1.0 - st["V_d"])               # E_N normalised to 1
        V_d = st["V_d"] + (-(st["V_d"]) + drive + i_nmda) / tau_d

        # --- branch transfer psi_b, then the soma integrates BRANCH outputs, not raw synapses ---
        D = torch.tanh(self.psi_gain * V_d + self.psi_bias)
        soma_in = (self.g_branch * D).sum(-1)
        if apical is not None:
            soma_in = soma_in + apical
        V_s = st["V_s"] + (-(st["V_s"]) + soma_in - st["a"]) / tau_s

        # --- spike with dynamic threshold, then reset ---
        S = spike(V_s - self.theta)
        V_s = V_s * (1.0 - S.detach())
        a = st["a"] + (-(st["a"]) + self.rho_a * S) / tau_a

        # --- calcium: pre, post, and branch-local dendritic events ---
        dend_evt = (nmda_on > 0.5).to(V_d.dtype)
        C = st["C"] + (-(st["C"]) + 0.3 * s_in.sum(-1)[:, None, None]
                       + 0.5 * S[..., None] + 0.7 * dend_evt) / tau_c

        # --- eligibility: pre x post, anti-Hebbian term, dendritic-event term ---
        x_pre = st["x_pre"] * (1 - 1 / self.tau_e) + s_in
        x_post = st["x_post"] * (1 - 1 / self.tau_e) + S
        pre_post = torch.einsum("bi,bn->bni", x_pre, S)[:, :, None, :].expand(-1, -1, self.B, -1)
        anti = torch.einsum("bi,bn->bni", s_in, x_post)[:, :, None, :].expand(-1, -1, self.B, -1)
        dterm = torch.einsum("bi,bnj->bnji", s_in, dend_evt)
        e = st["e"] + (-(st["e"]) + pre_post - 0.4 * anti + 0.6 * dterm) / self.tau_e

        r_bar = st["r_bar"] + (-(st["r_bar"]) + S) / self.tau_r

        new = {"V_d": V_d, "V_s": V_s, "a": a, "C": C, "e": e, "r_bar": r_bar,
               "x_pre": x_pre, "x_post": x_post,
               "n_spk": st["n_spk"] + S.detach().sum(),
               "n_syn": st["n_syn"] + (s_in.detach() > 0).sum() * self.A.detach().gt(0.5).float().mean(),
               "n_dend": st["n_dend"] + dend_evt.detach().sum()}

        if m_vec is not None:
            self._plastic_update(new, m_vec)
        return S, new

    # ---------------------------------------------------------------- local plasticity (NO gradient)
    @torch.no_grad()
    def _plastic_update(self, st, m_vec):
        """Three-factor rule: eligibility x per-synapse modulation. Section 5, 7, 8."""
        m_i = (m_vec @ self.q.t())                                   # (Bt, N) local modulation
        de = (st["e"] * m_i[:, :, None, None]).mean(0)                # average over batch
        self.W_f.mul_(1.0 - self.lam_fast).add_(self.eta_fast * de)

        if self.fast_slow:
            # consolidation is gated by a calcium tag AND a positive modulatory signal, so a fast change
            # only becomes permanent when the synapse was active, calcium was high, and something
            # meaningful happened. Otherwise fast memory simply decays.
            tag = torch.sigmoid(st["C"].mean(0) - self.theta_tag) * m_i.mean(0).clamp(min=0)[:, None]
            self.W_s.mul_(1.0 - self.lam_slow).add_(self.eta_slow * tag[..., None] * self.W_f)

        # firing-rate homeostasis: threshold chases the target rate
        rate = st["r_bar"].mean(0)
        # asymmetric: a SILENT neuron is pulled down hard (10x) because silence is absorbing --
        # no spikes means no eligibility means no way back without this.
        err = rate - self.target_rate
        gain = torch.where(rate < 1e-4, torch.full_like(err, 10.0), torch.ones_like(err))
        self.theta.add_(self.eta_theta * gain * err)
        self.theta.clamp_(0.02, 5.0)

        # synaptic normalisation -- a total incoming budget, preserving relative differences
        tot = (self.A * (self.W_s + self.W_f)).abs().sum(-1, keepdim=True) + 1e-6
        scale = (math.sqrt(self.I) / tot).clamp(max=1.0)
        self.W_s.mul_(scale)
        self.W_f.mul_(scale)

        if self.structural:
            self.U.mul_(0.99).add_(0.01 * (st["e"].mean(0) * m_i.mean(0)[:, None, None]).abs())
            self.step_count += 1
            if self.prune_every and int(self.step_count.item()) % self.prune_every == 0:
                weak = self.U < self.theta_prune
                self.prune_count = torch.where(weak, self.prune_count + 1,
                                               torch.zeros_like(self.prune_count))
                self.A = torch.where(self.prune_count >= 3, torch.zeros_like(self.A), self.A)

    def sparsity(self):
        return float(self.A.gt(0.5).float().mean())


class EpisodicMemory:
    """Top-K retrieval over a small store. Not attention over every past token (spec section 12)."""

    def __init__(self, dim, capacity=256, k=4, temp=0.5):
        self.dim, self.cap, self.k, self.temp = dim, capacity, k, temp
        self.K, self.V, self.R = [], [], []

    def write(self, key, val, sig=1.0):
        self.K.append(key.detach())
        self.V.append(val.detach())
        self.R.append(float(sig))
        if len(self.K) > self.cap:                       # evict the least significant, not the oldest
            j = int(min(range(len(self.R)), key=lambda i: self.R[i]))
            self.K.pop(j), self.V.pop(j), self.R.pop(j)

    def read(self, q):
        if not self.K:
            return torch.zeros(q.shape[0], self.dim, device=q.device)
        Km = torch.stack(self.K).to(q.device)
        Vm = torch.stack(self.V).to(q.device)
        sim = F.normalize(q, dim=-1) @ F.normalize(Km, dim=-1).t()
        k = min(self.k, Km.shape[0])
        w, idx = sim.topk(k, dim=-1)
        w = torch.softmax(w / self.temp, dim=-1)
        return (w.unsqueeze(-1) * Vm[idx]).sum(1)

    def clear(self):
        self.K, self.V, self.R = [], [], []


class ADRN(nn.Module):
    """Encoder -> ADRN workspace -> {output, prediction} with adaptive halting and an energy proxy."""

    def __init__(self, n_in, n_out, n_neurons=64, n_branch=4, T=8, *, dendrites=True, fast_slow=True,
                 structural=True, local_only=False, use_memory=True, halt=True, mem_k=4,
                 theta_halt=0.15, k_halt=2, min_steps=4, **kw):
        super().__init__()
        self.T, self.halt, self.theta_halt, self.k_halt = T, halt, theta_halt, k_halt
        # a silent network has a constant state, so unguarded halting would stop at step 2 and
        # hide the silence. Never halt before min_steps.
        self.min_steps = min_steps
        self.use_memory, self.local_only = use_memory, local_only
        self.enc = nn.Linear(n_in, n_neurons)
        self.layer = ADRNLayer(n_neurons, n_neurons, n_branch, dendrites=dendrites,
                               fast_slow=fast_slow, structural=structural,
                               learn_dynamics=not local_only, **kw)
        self.out = nn.Linear(n_neurons, n_out)
        self.pred = nn.Linear(n_neurons, n_in)              # self-supervised prediction head
        self.mem_proj = nn.Linear(n_neurons, n_neurons)
        self.mem = EpisodicMemory(n_neurons, k=mem_k)
        if local_only:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x, m_vec=None, return_trace=False):
        """x (Bt, n_in) or (Bt, T, n_in). Returns logits, and a dict of episode diagnostics."""
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.T, -1)
        Bt, T = x.shape[0], x.shape[1]
        st = self.layer.init_state(Bt, x.device)
        acc = torch.zeros(Bt, self.layer.N, device=x.device)
        prev, halted, steps = None, torch.zeros(Bt, dtype=torch.bool, device=x.device), 0
        quiet = torch.zeros(Bt, device=x.device)
        preds, trace = [], []

        for t in range(T):
            drive = torch.relu(self.enc(x[:, t]))
            apical = self.mem.read(self.mem_proj(acc)) if self.use_memory else None
            S, st = self.layer(drive, st, m_vec=m_vec, apical=apical)
            acc = acc + S
            preds.append(self.pred(acc))
            if return_trace:
                trace.append(S.detach().mean().item())
            steps = t + 1
            if self.halt and prev is not None and t + 1 >= self.min_steps:
                # section 11: halt when the state stops changing for k consecutive steps
                delta = (acc - prev).abs().mean(-1)
                quiet = torch.where(delta < self.theta_halt, quiet + 1, torch.zeros_like(quiet))
                halted = halted | (quiet >= self.k_halt)
                if bool(halted.all()):
                    break
            prev = acc.clone()

        logits = self.out(acc / max(steps, 1))
        energy = (0.4 * st["n_spk"] + 0.2 * st["n_syn"] + 0.4 * st["n_dend"]) / max(Bt, 1)
        info = {"steps": steps, "energy": float(energy), "spikes": float(st["n_spk"]) / max(Bt, 1),
                "sparsity": self.layer.sparsity(), "pred": preds[-1],
                "rate": float(acc.detach().mean()) / max(steps, 1)}
        if return_trace:
            info["trace"] = trace
        return logits, info

    def write_episode(self, x, sig=1.0):
        with torch.no_grad():
            _, info = self.forward(x)
        return info

    def reset_memory(self):
        self.mem.clear()


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
