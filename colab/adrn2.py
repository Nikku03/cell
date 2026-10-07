"""ADRN-2A -- digital encoder + recurrent workspace + context manager + fast/consolidated adapters.

Spec sections 2, 4, 5, 6.2, 6.3, 7, 8. STAGE 2A ONLY: no dendrites, no episodic memory, no world model,
no planner. Those belong to later stages and deliberately do not appear in this file.

--------------------------------------------------------------------------------------------------
WHAT IS BEING CLAIMED, AND THE GATE AND MDE THAT DECIDE IT -- DECLARED HERE, BEFORE ANY NUMBER EXISTS
--------------------------------------------------------------------------------------------------
THE CLAIM. On a piecewise-stationary stream whose latent contexts RECUR, a system that (a) infers a
latent context, (b) keeps a per-context fast adapter updated by a local delta rule with no backprop
through the encoder, and (c) consolidates that fast adapter into a slower per-context store, will reach
a higher held-out online accuracy than the same system without a context manager, and than conventional
online SGD on the same frozen features.

PRIMARY METRIC, and it is the RAW HELD-OUT VALUE. `online_acc` = prequential accuracy over the whole
online stream: at every step the model predicts a NEVER-SEEN example, is scored, and only then receives
the label. No example is ever scored twice and none is scored after being trained on. A control arm
establishes SIGNIFICANCE only; the effect size reported everywhere is the raw prequential accuracy of
the arm itself, never a difference net of a control.

SECONDARY PREDECLARED METRIC. `retention_acc` = accuracy on FRESH examples from each context, evaluated
after the stream ends with all learning switched off and the context inferred without labels. This is
the retrieval/consolidation measure.

UNIT OF REPLICATION. The SEED, where one seed is an independent initialisation AND an independent
environment draw (fresh class prototypes, fresh permutations, fresh stream order, fresh examples).
Seeds inside a single environment draw would measure optimiser noise, not the claim. n = ADRN2_SEEDS.

MDE. 3 * sd / sqrt(n) computed on the PAIRED per-seed gap between the two arms being compared, since
every arm sees the identical stream within a seed. An arm is credited only if its gap exceeds its MDE.
A gap that is not positive FAILS without needing an MDE. A POSITIVE gap whose paired sample sd is
exactly zero has an UNESTIMABLE MDE and is reported UNSUPPORTED, never as a pass: n seeds landing on
the same difference is n coincidences on a count-valued metric, not evidence of zero population
variance, and this file has already shipped a PASS off exactly that (G3 consolidation, gap +0.1000
against MDE 0.0000, n=2, the only substantive PASS in its condition).

GATES (all predeclared, evaluated per feedback condition):
  G1  full beats the BEST control arm on online_acc     (controls: frozen, sgd_head, sgd_full,
  G1b full beats the BEST control arm on retention_acc   global_fast). BOTH metrics are gated and
      both are reported, so the verdict cannot be assembled by picking whichever one happened to win.
  G2  the context manager contributes      full - global_fast          > MDE  on online_acc
  G3  consolidation contributes            full - no_consolidation     > MDE  on online_acc
  G3b consolidation contributes            full - no_consolidation     > MDE  on retention_acc
  G4  the plasticity controller contributes full - no_controller       > MDE  on online_acc
  G5  the recurrent workspace contributes  full - no_workspace         > MDE  on online_acc
  G6  halting is live and not costly       mean realised steps < cap AND full - no_halting > -MDE
  G7  eligibility earns its keep when feedback is delayed:
        delayed condition   full - no_eligibility > MDE
        immediate condition full and no_eligibility must be BIT-IDENTICAL (eligibility is gated OFF,
        so it must cost exactly nothing). This inverts ADRN-1's defect: byte-identical arms are here a
        PREDECLARED REQUIREMENT in one condition and a FAILURE in the other, and both are asserted.

CEILING AND FLOOR, ADJUDICATED PER METRIC -- on online_acc AND on retention_acc, because G1b and G3b
are read on the latter and an earlier version checked only the former, leaving half the gates with no
ceiling/floor check at all on a metric that sat near chance for every arm. If the best arm on a metric
reaches >= 0.99 the metric cannot separate the arms that reach it and it is declared UNINFORMATIVE for
those arms -- but an arm sitting more than one MDE BELOW a ceiling the others reach is highly
informative and is reported as such. If every arm is within max(MDE, 0.02) of chance (1/n_classes) the
metric is too hard, it is declared UNINFORMATIVE, and every gate read on it is moved out of the
pass/fail tally into `gates_unsupported` rather than being reported as a result in either direction.

--------------------------------------------------------------------------------------------------
THE FOUR DEFECTS ADRN-1 SHIPPED, AND THE MACHINERY IN THIS FILE THAT MAKES EACH ONE IMPOSSIBLE
--------------------------------------------------------------------------------------------------
1 THE NETWORK WAS SILENT (theta=1.0 vs a soma at 0.05 -> 0.00 spikes/example, every downstream
  mechanism a no-op while scores were still returned).
2 THE BASELINE WAS HANDICAPPED (the point-neuron control was silent for the same reason, so fixing only
  the proposed model would have manufactured a win).
3 THE ABLATION WAS VACUOUS (prune_every defaulted to 0, so structural=True/False changed nothing).
4 THE LEARNING SYSTEM NEVER RAN (net(x) was called with no modulatory vector, so eligibility,
  neuromodulation, fast/slow weights, consolidation, homeostasis and pruning were ALL inert; the tell,
  visible twice before it was read correctly, was three different ablations returning BYTE-IDENTICAL
  accuracy, energy and spike counts).

Against 1 and 4: `assert_liveness` runs on EVERY arm-run BEFORE its score is allowed into the results
table, and raises `MechanismInert` if a claimed mechanism is numerically switched off -- fast adapters
that do not change between updates, a context posterior that is collapsed onto one context or is flat
uniform, a consolidated adapter identical to (or indistinguishable from) the fast adapter, a plasticity
controller emitting a constant, a workspace whose state does not move across internal steps, an
eligibility trace that is never touched when feedback is delayed (or that IS touched when it is not),
a context manager that never allocates a second context, an encoder that silently moved when it was
supposed to be frozen.

Against 2: every control arm is built to the same parameter budget by bisection with realised counts
printed, the SGD controls get an EXACT input buffer for delayed feedback while the local arms get only
a decaying trace, and every arm with a free scalar step size (both SGD controls, the no-controller
ablation) has that step size calibrated on a SEPARATE validation environment draw over the same grid.
A control is never allowed to lose because nobody tuned it. `assert_digital_competence` additionally
refuses to score ANY arm whose offline encoder failed to train, or whose workspace state does not make
the task linearly solvable -- the ADRN-1 "silent network" failure in its supervised form. It earned its
place immediately: the context embedding table had been sized by the number of ADAPTER slots, so every
context-blind control saw six mutually contradictory permutations during offline pretraining with no
way to tell them apart and reached 0.36 held-out where the context arms reached 0.87. That is ADRN-1's
defect 2 exactly, it was invisible in the results table, and it would have manufactured the entire
result.

Against 3: `flag_effect_probe` toggles each component flag on one pretrained model, replays the SAME
probe stream, and raises unless the output logits measurably differ. And `assert_arms_distinct` refuses
to report if any two differently-configured arms produced identical metrics across all seeds, with the
single predeclared exception named in G7.

"It should work" is not evidence anywhere in this file.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("ADRN2_THREADS", "1")))

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SMOKE = os.environ.get("ADRN2_SMOKE", "0") == "1"


# ==================================================================================================
# CONFIG
# ==================================================================================================
class Cfg:
    """Every knob, from env vars, with a SMOKE mode that shrinks the run and labels the output."""

    def __init__(self):
        g = lambda k, d, f=int: f(os.environ.get(k, d))
        self.seeds = g("ADRN2_SEEDS", 6)
        self.rounds = g("ADRN2_ROUNDS", 4)          # how many times the stream revisits every context
        self.block = g("ADRN2_BLOCK", 60)           # steps a context is held before it switches
        self.budget = g("ADRN2_BUDGET", 30000)
        self.delay = g("ADRN2_DELAY", 4)            # feedback delay in the delayed condition
        self.t_win = g("ADRN2_TWIN", 8)             # input window length
        self.n_classes = g("ADRN2_CLASSES", 4)
        self.d_in = g("ADRN2_DIN", 8)
        self.k_slots = g("ADRN2_KSLOTS", 6)         # context slots available to the manager
        self.pre_n = g("ADRN2_PRE_N", 3072)
        self.pre_epochs = g("ADRN2_PRE_EPOCHS", 14)
        self.meta_iters = g("ADRN2_META", 60)
        self.eval_n = g("ADRN2_EVALN", 256)         # fresh examples per context for retention_acc
        self.p_flip = g("ADRN2_PFLIP", 0.12, float)  # label noise -> analytic ceiling below 1.0
        self.noise = g("ADRN2_NOISE", 0.40, float)
        self.max_steps = g("ADRN2_MAXSTEPS", 4)     # workspace step cap
        self.theta_h = g("ADRN2_THETAH", 0.5, float)
        # CONTEXT CREATION. A slot is allocated when NO stored context explains the data as well as
        # the active slot's own best-ever smoothed loss, by theta_new nats, for `persist` consecutive
        # steps. The reference has to be relative: an absolute threshold cannot separate "the context
        # changed" from "this slot has not finished adapting", and the context-free loss of the
        # stable decoder cannot serve either -- measured at 3.0-3.7 nats online, because a decoder
        # trained on other permutations is confidently WRONG rather than merely uninformed, so
        # nothing is ever "worse than having no adapter at all".
        self.theta_new = g("ADRN2_THETANEW", 0.40, float)   # nats of degradation vs the slot's best
        self.min_obs_create = g("ADRN2_MINOBS", 25)   # the ACTIVE slot must have had its chance first
        self.create_cooldown = g("ADRN2_COOLDOWN", 22)
        # A newly created slot starts with a ZERO adapter, which means its likelihood is the stable
        # decoder's -- and that decoder is confidently WRONG on an unseen permutation (3.0-3.7 nats).
        # So a fresh slot loses the posterior on its first step and can never adapt. It gets a
        # protected trial window instead, which is what "allocate a new hypothesis" has to mean.
        self.protect_steps = g("ADRN2_PROTECT", 20)
        self.persist = g("ADRN2_PERSIST", 3)   # consecutive failing steps required to declare a change
        self.warm = g("ADRN2_WARM", 20)   # labelled examples used to IDENTIFY the context at retention
        self.pre_restarts = g("ADRN2_RESTARTS", 2)
        self.probe_blocks = g("ADRN2_PBLOCKS", 3)   # blocks of the validation draw used for lr calibration
        self.lam_e = g("ADRN2_LAME", 0.8, float)    # eligibility decay, used ONLY when delay > 0
        self.beta_lik = g("ADRN2_BETA", 1.5, float)
        self.gamma_ws = g("ADRN2_GAMMA", 2.0, float)
        self.stick = g("ADRN2_STICK", 0.02, float)  # transition prior: off-diagonal mass
        self.probe_len = g("ADRN2_PROBE", 15)       # steps after a switch counted as "revisit"
        self.ceiling = g("ADRN2_CEIL", 0.99, float)
        if SMOKE:
            self.seeds, self.rounds, self.block = 2, 2, 40
            # pre_n/pre_epochs are NOT cut to the bone in SMOKE. At 768x6 the encoder came out at
            # 0.30 held-out (chance is 0.25) and every arm sat on the floor -- a smoke that proves
            # nothing about the mechanisms because the digital half was never trained. The
            # `assert_digital_competence` check below now refuses to run in that state anyway.
            self.pre_n, self.pre_epochs, self.meta_iters = 2048, 10, 15
            self.eval_n, self.probe_blocks = 64, 2
        self.chance = 1.0 / self.n_classes

    def asdict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class MechanismInert(RuntimeError):
    """Raised when a mechanism that a reported number depends on is numerically switched off."""


# ==================================================================================================
# ENVIRONMENT -- a piecewise-stationary stream whose latent contexts RECUR
# ==================================================================================================
class ContextualStream:
    """Latent class s is encoded in a marked pulse inside a noisy window; the CONTEXT permutes labels.

    The window carries one MARKED pulse (channel -1 set to 1) holding prototype[s], and one UNMARKED
    distractor pulse holding prototype[s'] for a random s'. So the encoder has a real job: find the
    marked slot. The context applies a permutation pi_c to the label, y = pi_c(s).

    The ONLINE contexts are the n_classes CYCLIC shifts. Visited equally, they make the marginal
    p(y|s) exactly uniform, so a context-blind model's Bayes rate is exactly chance = 1/n_classes.
    That gives the floor check something unambiguous to test against. The OFFLINE (pretraining)
    contexts are drawn from the NON-cyclic permutations, so the online contexts are genuinely unseen
    and the stable decoder cannot already know them.

    Label noise p_flip caps the achievable accuracy at (1-p_flip) + p_flip/n_classes, deliberately
    below 1.0 so the task is not at ceiling by construction. That value is printed as the analytic
    ceiling and the empirical ceiling check is run on top of it regardless.
    """

    def __init__(self, rng, cfg: Cfg):
        C, d = cfg.n_classes, cfg.d_in
        self.cfg = cfg
        self.d_sig = d - 2                      # last two channels: constant bias, pulse marker
        p = rng.standard_normal((C, self.d_sig)).astype(np.float32)
        self.proto = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
        allp = list(itertools.permutations(range(C)))
        cyc = [tuple((i + k) % C for i in range(C)) for k in range(C)]
        noncyc = [q for q in allp if q not in cyc]
        idx = rng.choice(len(noncyc), size=cfg.k_slots, replace=False)
        self.pre_perms = [np.array(noncyc[i], np.int64) for i in idx]
        self.on_perms = [np.array(q, np.int64) for q in cyc]
        self.k_true = len(self.on_perms)
        self.analytic_ceiling = (1 - cfg.p_flip) + cfg.p_flip / C

    def _draw(self, n, perm, rng):
        cfg, C, T, d = self.cfg, self.cfg.n_classes, self.cfg.t_win, self.cfg.d_in
        s = rng.integers(0, C, n)
        X = (cfg.noise * rng.standard_normal((n, T, d))).astype(np.float32)
        X[:, :, -2] = 1.0
        X[:, :, -1] = 0.0
        sig_t = rng.integers(0, T, n)
        dis_t = (sig_t + 1 + rng.integers(0, T - 1, n)) % T
        s2 = rng.integers(0, C, n)
        rows = np.arange(n)
        X[rows, sig_t, :self.d_sig] += 1.5 * self.proto[s]
        X[rows, sig_t, -1] = 1.0
        X[rows, dis_t, :self.d_sig] += 1.2 * self.proto[s2]
        y = perm[s].copy()
        flip = rng.random(n) < cfg.p_flip
        y[flip] = rng.integers(0, C, int(flip.sum()))
        return torch.from_numpy(X), torch.from_numpy(y)

    def pretrain_batch(self, n, rng):
        """Offline batch: the context slot is KNOWN and supplied to the workspace, as it would be."""
        slot = rng.integers(0, len(self.pre_perms), n)
        X = torch.empty(n, self.cfg.t_win, self.cfg.d_in)
        y = torch.empty(n, dtype=torch.long)
        for c in np.unique(slot):
            m = slot == c
            xb, yb = self._draw(int(m.sum()), self.pre_perms[c], rng)
            X[torch.from_numpy(m)] = xb
            y[torch.from_numpy(m)] = yb
        return X, y, torch.from_numpy(slot)

    def online(self, ctx, n, rng):
        return self._draw(n, self.on_perms[ctx], rng)


def make_stream(rng, cfg: Cfg, k_true: int):
    """rounds x (a fresh random order of every context), so every context recurs `rounds` times."""
    blocks = []
    prev = -1
    for _ in range(cfg.rounds):
        order = list(rng.permutation(k_true))
        if order[0] == prev and len(order) > 1:
            order[0], order[1] = order[1], order[0]
        blocks += [(int(c), cfg.block) for c in order]
        prev = order[-1]
    return blocks


# ==================================================================================================
# THE MODEL
# ==================================================================================================
D_MOD, D_CTX = 5, 6


class Flags:
    NAMES = ("workspace", "halting", "context", "fast", "consolidate", "controller", "eligibility")

    def __init__(self, **kw):
        for n in self.NAMES:
            setattr(self, n, bool(kw.get(n, True)))
        self.oracle_context = bool(kw.get("oracle_context", False))
        self.sgd = kw.get("sgd", "none")            # 'none' | 'head' | 'full'

    def asdict(self):
        d = {n: getattr(self, n) for n in self.NAMES}
        d["oracle_context"] = self.oracle_context
        d["sgd"] = self.sgd
        return d


class ADRN2A(nn.Module):
    """Digital encoder -> recurrent workspace with adaptive halting -> stable decoder + adapters.

    ALLOCATION follows the build-time flags (an arm without a context manager does not pay for six
    adapter banks), so the parameter budget is honest. BEHAVIOUR follows the runtime flags of the same
    name, which are mutable, so `flag_effect_probe` can toggle a component on one fully-allocated
    pretrained model and prove the toggle changes the output.
    """

    def __init__(self, cfg: Cfg, flags: Flags, d_h: int):
        super().__init__()
        self.cfg, self.flags = cfg, flags
        d_w = max(8, d_h // 2)
        self.d_h, self.d_w, self.C = d_h, d_w, cfg.n_classes
        self.k = cfg.k_slots if flags.context else 1

        # ---- DIGITAL ENCODER: an ordinary GRU over the input window, trained offline by backprop.
        # Deliberately conventional: ADRN-2's thesis is that the digital machinery stays where it is
        # strongest and the adaptive machinery sits on top of it.
        self.enc = nn.GRU(cfg.d_in, d_h, batch_first=True)

        # ---- RECURRENT WORKSPACE: w_{t+1} = F(w_t, h, m_t, c_t), gated update, + halting head.
        self.w0 = nn.Linear(d_h, d_w)
        din = d_w + d_h + D_MOD + D_CTX
        self.fz = nn.Linear(din, d_w)
        self.fn = nn.Linear(din, d_w)
        self.halt = nn.Linear(d_w, 1)
        # The embedding table is ALWAYS the full k_slots, even for arms whose context manager is
        # switched off. Sizing it by self.k handed every context-blind arm -- frozen, both SGD
        # controls, global_fast -- a single embedding row, so OFFLINE pretraining showed their
        # workspace six mutually contradictory permutations with no way to tell them apart and their
        # encoders never converged: 0.36 offline held-out against 0.87 for the context arms, on the
        # SAME data with the SAME budget. That is ADRN-1's defect 2 exactly -- a control that loses
        # because it was crippled rather than because the mechanism under test is better -- and it
        # would have manufactured the entire result.
        self.ctx_emb = nn.Embedding(cfg.k_slots, D_CTX)

        # ---- STABLE DECODER W_s, trained offline, shared across contexts. Operates on w_aug =
        # [w, 1], so the per-context adapters inherit a bias column for free.
        self.dec = nn.Linear(d_w + 1, self.C, bias=False)

        # ---- PLASTICITY CONTROLLER (spec sec 8): (eta_t, lambda_t, g_t) from
        # [w summary, delta_t, u_t, n_t, c_t summary].
        self.ctl_in = 8
        self.controller = nn.Sequential(nn.Linear(self.ctl_in, 16), nn.Tanh(), nn.Linear(16, 3)) \
            if flags.controller else None

        # ---- ADAPTERS. Buffers, not parameters: they are written by the LOCAL rule only. They are
        # nonetheless counted into the parameter budget, because a bank of per-context adapters is
        # capacity and an arm that carries one must pay for it.
        z = torch.zeros(self.k, self.C, d_w + 1)
        self.register_buffer("W_f", z.clone() if flags.fast else torch.zeros(0))
        self.register_buffer("W_m", z.clone() if flags.consolidate else torch.zeros(0))
        self.register_buffer("keys", torch.zeros(self.k, d_w))

        self.reset_online()

    # ------------------------------------------------------------------ budget accounting
    def param_counts(self):
        trainable = sum(p.numel() for p in self.parameters())
        adaptive = int(self.W_f.numel() + self.W_m.numel() + self.keys.numel())
        return trainable, adaptive, trainable + adaptive

    # ------------------------------------------------------------------ online state
    def reset_online(self):
        k = self.k
        self.p = torch.zeros(k)
        self.p[0] = 1.0
        self.used = torch.zeros(k, dtype=torch.bool)
        self.used[0] = True
        self.loss_ema = torch.full((k,), math.log(self.C))
        # A SECOND, FASTER loss estimate exists only for change detection. With a single 0.9-decay
        # EMA the manager needs ~8 steps to notice a context switch while the fast adapter needs ~8
        # steps to overwrite the old context's slot, and the two race: creation fired on some switches
        # and not others, so one slot ended up holding three different contexts and retention died.
        # Detection has to be strictly faster than adaptation or the memory is destroyed before it is
        # ever filed.
        self.loss_fast = torch.full((k,), math.log(self.C))
        # BEST-SEEN loss per slot: the reference a context change is measured AGAINST. An absolute
        # threshold cannot serve, because how low a slot can drive its loss depends on how wrong the
        # stable decoder happens to be for that particular permutation -- slots that plateaued at 1.4
        # were never marked usable and so could never trigger the creation of the NEXT context, and
        # one slot silently absorbed three different contexts. Stays +inf until a slot has enough
        # observations to have a baseline at all, which is what blocks creation from a slot that has
        # not established one.
        self.best_seen = torch.full((k,), float("inf"))
        self.base_loss_ema = math.log(self.C)   # what the STABLE DECODER alone achieves, smoothed
        self.util = torch.zeros(k)
        self.n_obs = torch.zeros(k)
        self.acc_ema, self.err_ema, self.novelty = self.cfg.chance, math.log(self.C), 0.5
        self.last_create = -10 ** 9
        self.protect_slot, self.protect_until = -1, -10 ** 9
        self.change_run = 0
        if self.W_f.numel():
            self.W_f.zero_()
        if self.W_m.numel():
            self.W_m.zero_()
        self.keys.zero_()

    # ------------------------------------------------------------------ forward pieces
    def encode(self, x):
        _, h = self.enc(x)
        return h[-1]

    def mod_vector(self, delayed: bool, b=1):
        m = torch.tensor([self.err_ema / math.log(self.C), self.acc_ema, self.novelty,
                          _ent_t(self.p) / math.log(max(2, self.k)), 1.0 if delayed else 0.0])
        return m.expand(b, D_MOD).contiguous()

    def ctx_input(self, p, b=1):
        # A context-blind arm online has no context estimate, so it receives the MARGINAL embedding
        # (the mean over slots) rather than an arbitrary single row.
        if not (self.flags.context and self.k > 1):
            return self.ctx_emb.weight.mean(0, keepdim=True).expand(b, D_CTX)
        return (p @ self.ctx_emb.weight).unsqueeze(0).expand(b, D_CTX)

    def workspace(self, h, m, c_in, hard_halt=True, record=None):
        """Returns (w_final, n_steps). Hard halting when p_halt > theta_h or the budget is reached."""
        w = torch.tanh(self.w0(h))
        if not self.flags.workspace:
            if record is not None:
                record.append(1)
            return w, 1
        traj = [w]
        n = 1
        for t in range(self.cfg.max_steps):
            u = torch.cat([w, h, m, c_in], -1)
            z = torch.sigmoid(self.fz(u))
            w = (1 - z) * w + z * torch.tanh(self.fn(u))
            traj.append(w)
            n = t + 1
            if hard_halt and self.flags.halting:
                if float(torch.sigmoid(self.halt(w)).mean()) > self.cfg.theta_h:
                    break
        if record is not None:
            record.append(n)
            record.append(float(torch.stack(traj).std(0).mean()) if len(traj) > 1 else 0.0)
        return w, n

    def workspace_act(self, h, m, c_in):
        """ACT-style differentiable halting used OFFLINE only: expected output + ponder cost.

        Trains the halting head so that the online hard rule is a trained decision and not a coin
        flip. Without this the halting head is random and 'adaptive halting' is a decoration.
        """
        w = torch.tanh(self.w0(h))
        still, out, expect = torch.ones(h.shape[0], 1), 0.0, 0.0
        for t in range(self.cfg.max_steps):
            u = torch.cat([w, h, m, c_in], -1)
            z = torch.sigmoid(self.fz(u))
            w = (1 - z) * w + z * torch.tanh(self.fn(u))
            p = torch.sigmoid(self.halt(w))
            wt = still * p if t < self.cfg.max_steps - 1 else still
            out = out + wt * self.dec(_aug(w))
            expect = expect + wt * (t + 1)
            still = still * (1 - p)
        return out, expect.mean()

    # ------------------------------------------------------------------ decoding with adapters
    def _adapter(self):
        A = 0.0
        if self.W_f.numel():
            A = A + self.W_f
        if self.W_m.numel():
            A = A + self.W_m
        return A if torch.is_tensor(A) else None

    def logits_all(self, w_aug):
        """(k, C) -- the logits EVERY context slot would produce. Drives likelihood and creation."""
        base = self.dec(w_aug)
        A = self._adapter()
        if A is None:
            return base.expand(self.k, self.C)
        return base.unsqueeze(0) + torch.einsum("kcd,d->kc", A, w_aug)

    def logits_mix(self, w_aug, p):
        base = self.dec(w_aug)
        A = self._adapter()
        if A is None:
            return base
        return base + torch.einsum("k,kcd,d->c", p, A, w_aug)

    # ------------------------------------------------------------------ CONTEXT MANAGER
    def transition_prior(self):
        """Sticky transition prior over the used slots (spec sec 6.2)."""
        k = self.k
        e = self.cfg.stick
        nu = max(1, int(self.used.sum()))
        pr = (1 - e) * self.p + e / max(1, nu - 1) * (self.used.float() * (1 - self.p))
        pr = pr * self.used.float()
        return pr / (pr.sum() + 1e-12)

    def protected(self, step):
        return self.protect_slot >= 0 and step < self.protect_until

    def one_hot_slot(self, j):
        p = torch.zeros(self.k)
        p[j] = 1.0
        return p

    def posterior_pre(self, prior, w, step=10 ** 9):
        """Prior x WORKSPACE term. Available BEFORE the label, so predictions use only this."""
        if not (self.flags.context and self.k > 1) or self.flags.oracle_context:
            return prior
        if self.protected(step):
            return self.one_hot_slot(self.protect_slot)
        kk = F.normalize(self.keys, dim=-1)
        sc = kk @ F.normalize(w, dim=-1)
        sc = sc * self.used.float()
        p = prior * torch.exp(self.cfg.gamma_ws * (sc - sc.max()))
        p = p * self.used.float()
        return p / (p.sum() + 1e-12)

    def posterior_post(self, pre, w_aug, y, step=10 ** 9):
        """... x LIKELIHOOD, once the (possibly delayed) label has arrived."""
        L = F.cross_entropy(self.logits_all(w_aug), y.expand(self.k), reduction="none")
        base_L = float(F.cross_entropy(self.dec(w_aug).unsqueeze(0), y))
        self._track(L, base_L)
        if not (self.flags.context and self.k > 1) or self.flags.oracle_context:
            return pre, L
        if self.protected(step):
            return self.one_hot_slot(self.protect_slot), L
        p = pre * torch.exp(-self.cfg.beta_lik * (L - L.min()))
        p = p * self.used.float()
        p = p / (p.sum() + 1e-12)
        return p, L

    def _track(self, L, base_L):
        self.loss_ema = torch.where(self.used, 0.9 * self.loss_ema + 0.1 * L, self.loss_ema)
        self.loss_fast = torch.where(self.used, 0.75 * self.loss_fast + 0.25 * L, self.loss_fast)
        ready = self.used & (self.n_obs >= 8)
        self.best_seen = torch.where(ready, torch.minimum(self.best_seen, self.loss_ema),
                                     self.best_seen)
        self.base_loss_ema = 0.9 * self.base_loss_ema + 0.1 * base_L
        # UTILITY of a slot = how many nats it saves over having NO adapter at all. Defining it as
        # "accuracy above chance" instead created a deadlock that silently disabled consolidation on
        # every context the stable decoder happened to be badly wrong about: a slot could not
        # consolidate until it was accurate, and it could not become accurate because the leaky fast
        # store alone cannot reach the magnitude needed to overturn a confident wrong prior -- only
        # the non-leaky consolidated store can. Measured: W_m stayed at 0.00 for the whole stream on
        # such slots while their accuracy sat at 0.10, below chance.
        self.util = torch.where(self.used, 0.97 * self.util + 0.03 * (base_L - L), self.util)

    def maybe_create(self, step, w, tel):
        """CONTEXT CREATION (spec sec 6.2): allocate a slot when no stored context explains the data
        as well as the active slot's own best-ever loss, by theta_new nats, for `persist` steps."""
        if not (self.flags.context and self.k > 1) or self.flags.oracle_context:
            return False
        if bool(self.used.all()) or step - self.last_create < self.cfg.create_cooldown:
            return False
        # PERSISTENCE, not a single bad step. One hard example spikes any fast statistic; only a
        # sustained failure of every stored context is evidence that the context itself changed.
        # Without this the detector fired mid-block and allocated a slot to the context it was
        # already in, which then absorbed the next two contexts as well.
        ref = float(self.best_seen[int(self.p.argmax())])
        fail = (math.isfinite(ref)
                and float(self.loss_fast[self.used].min()) > ref + self.cfg.theta_new)
        self.change_run = self.change_run + 1 if fail else 0
        if self.change_run < self.cfg.persist:
            return False
        # The guard has to be on the ACTIVE slot, not on the best-observed slot. Guarding on
        # max(n_obs) let a freshly created (and therefore still hopeless) slot trip the creation rule
        # again on the very next eligible step: the manager burned all six slots inside two blocks,
        # threw away each adapter as soon as it started working, and drove accuracy BELOW chance.
        if float(self.n_obs[int(self.p.argmax())]) < self.cfg.min_obs_create:
            return False
        j = int((~self.used).nonzero()[0, 0])
        self.used[j] = True
        self.keys[j] = F.normalize(w, dim=-1)
        self.loss_ema[j] = self.loss_fast[j] = float(self.loss_fast[self.used].min())
        self.best_seen[j] = float("inf")     # no baseline yet, so it cannot trigger the next creation
        self.n_obs[j] = 0.0
        self.util[j] = 0.0
        if self.W_f.numel():
            self.W_f[j].zero_()
        if self.W_m.numel():
            self.W_m[j].zero_()
        self.p = self.one_hot_slot(j)
        self.last_create = step
        self.change_run = 0
        self.protect_slot, self.protect_until = j, step + self.cfg.protect_steps
        tel["created_at"].append(step)
        return True

    # ------------------------------------------------------------------ PLASTICITY CONTROLLER
    def rates(self, w, correct, ent, fixed_eta):
        """(eta_t, lambda_t, g_t). Learned when the controller is on; a calibrated constant when not."""
        if self.controller is None:
            return fixed_eta, 0.98, 1.0
        feat = torch.tensor([float(w.mean()), float(w.std()), self.err_ema / math.log(self.C),
                             self.acc_ema, float(self.util[self.p.argmax()]), self.novelty,
                             ent / math.log(max(2, self.k)),
                             float(self.n_obs[self.p.argmax()] / 100.0)])
        o = self.controller(feat)
        eta = 0.02 + 0.60 * torch.sigmoid(o[0])
        lam = 0.93 + 0.069 * torch.sigmoid(o[1])
        g = 0.2 + 1.6 * torch.sigmoid(o[2])
        return float(eta), float(lam), float(g)

    # ------------------------------------------------------------------ FAST + CONSOLIDATED ADAPTERS
    @torch.no_grad()
    def local_update(self, credit, y, p_post, eta, lam, g, tel):
        """W_f^(c) <- lambda_f W_f^(c) + eta_f (y - yhat) credit^T, then consolidation into W_m^(c).

        `credit` is the vector the error is paired with: the CURRENT w_aug when feedback is immediate,
        the ELIGIBILITY TRACE when it is delayed. The step is normalised-LMS (divide by ||credit||^2+1)
        so eta is a scale-free step size rather than a number that has to be retuned per width.

        lambda_f is applied to EVERY slot on EVERY step, not only the active one. That is what makes
        the fast store genuinely fast: a context left alone for one block loses lambda^block of its
        fast adapter, which is precisely the job the consolidated store exists to do. If lambda were
        applied only to the active slot, W_f would never forget and consolidation would be decoration.
        """
        if not self.flags.fast or not self.W_f.numel():
            return
        before = self.W_f.norm().item()
        prev = self.W_f.clone()
        self.W_f.mul_(lam)
        yhat = torch.softmax(self.logits_mix(credit, p_post), -1)
        err = F.one_hot(y, self.C).float().squeeze(0) - yhat
        scale = eta / (float(credit.dot(credit)) + 0.1)
        self.W_f.add_(scale * torch.einsum("k,c,d->kcd", p_post, err, credit))
        d = (self.W_f - prev).norm().item()
        tel["dWf_rel"].append(d / (before + 1e-8) if before > 1e-8 else (1.0 if d > 0 else 0.0))
        tel["dWf_abs"].append(d)

        # CONSOLIDATION: high confidence AND repeated utility, per spec sec 6.3.
        if self.flags.consolidate and self.W_m.numel():
            c = int(p_post.argmax())
            conf = float(p_post.max())
            if conf > 0.55 and self.n_obs[c] >= 15 and float(self.util[c]) > 0.05:
                eta_m = min(0.5, 0.10 * g)
                self.W_m[c].mul_(1 - eta_m).add_(eta_m * self.W_f[c])
                tel["n_consol"] += 1

    @torch.no_grad()
    def bookkeeping(self, p_post, w, correct):
        c = int(p_post.argmax())
        self.n_obs[c] += 1
        if self.flags.context and self.k > 1 and not self.flags.oracle_context:
            self.keys[c] = F.normalize(0.95 * self.keys[c] + 0.05 * w, dim=-1)
        self.p = p_post
        self.novelty = float(np.clip(float(self.loss_ema[self.used].min()) / math.log(self.C), 0, 2))


def _aug(w):
    return torch.cat([w, torch.ones(*w.shape[:-1], 1)], -1)


def _ent_t(p):
    q = torch.clamp(p, 1e-12, 1.0)
    return float(-(q * q.log()).sum())


def _ent(p):
    q = np.clip(np.asarray(p, np.float64), 1e-12, 1.0)
    q = q / q.sum()
    return float(-(q * np.log(q)).sum())


# ==================================================================================================
# OFFLINE TRAINING -- the digital half, by ordinary backprop. No adapters are involved.
# ==================================================================================================
def pretrain(model: ADRN2A, env: ContextualStream, cfg: Cfg, rng):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    X, y, slot = env.pretrain_batch(cfg.pre_n, rng)
    n = len(X)
    acc_ema, err_ema = cfg.chance, math.log(cfg.n_classes)
    for _ in range(cfg.pre_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            b = perm[i:i + 64]
            h = model.encode(X[b])
            m = torch.tensor([err_ema / math.log(cfg.n_classes), acc_ema, 0.5, 0.0, 0.0])
            m = m.expand(len(b), D_MOD)
            c_in = model.ctx_emb(slot[b])
            out, ponder = model.workspace_act(h, m, c_in)
            loss = F.cross_entropy(out, y[b]) + 0.01 * ponder
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                err_ema = 0.9 * err_ema + 0.1 * float(loss)
                acc_ema = 0.9 * acc_ema + 0.1 * float((out.argmax(-1) == y[b]).float().mean())
    return {"pretrain_acc": acc_ema, "pretrain_loss": err_ema}


def meta_train_controller(model: ADRN2A, env: ContextualStream, cfg: Cfg, rng):
    """Train the plasticity controller by unrolling the LOCAL rule and minimising post-update loss.

    Only (eta, lambda, g) carry gradient: w and h are detached, so nothing here backprops into the
    encoder. Without this the controller is a random net and 'learned plasticity' would be a lie.
    """
    if model.controller is None:
        return {"meta": "no controller"}
    opt = torch.optim.Adam(model.controller.parameters(), lr=5e-3)
    hist = []
    for it in range(cfg.meta_iters):
        perm = env.pre_perms[int(rng.integers(0, len(env.pre_perms)))]
        X, y = env._draw(10, perm, rng)
        W = torch.zeros(model.C, model.d_w + 1)
        total = 0.0
        for t in range(len(X)):
            with torch.no_grad():
                h = model.encode(X[t:t + 1])
                m = model.mod_vector(False)
                c_in = model.ctx_input(model.p)
                w, _ = model.workspace(h, m, c_in, hard_halt=False)
                wa = _aug(w)[0].detach()
            logit = model.dec(wa) + W @ wa
            loss = F.cross_entropy(logit.unsqueeze(0), y[t:t + 1])
            total = total + loss
            feat = torch.tensor([float(wa[:-1].mean()), float(wa[:-1].std()),
                                 float(loss.detach()) / math.log(model.C), 0.5, 0.0, 0.5, 0.0,
                                 t / 100.0])
            o = model.controller(feat)
            eta = 0.02 + 0.60 * torch.sigmoid(o[0])
            lam = 0.93 + 0.069 * torch.sigmoid(o[1])
            err = F.one_hot(y[t], model.C).float() - torch.softmax(logit, -1)
            W = lam * W + (eta / (float(wa.dot(wa)) + 0.1)) * torch.outer(err, wa)
        opt.zero_grad()
        (total / len(X)).backward()
        nn.utils.clip_grad_norm_(model.controller.parameters(), 1.0)
        opt.step()
        hist.append(float(total.detach()) / len(X))
    return {"meta_loss_first": hist[0], "meta_loss_last": hist[-1]}


@torch.no_grad()
def assert_digital_competence(model: ADRN2A, env: ContextualStream, cfg: Cfg, rng, tag):
    """Two prerequisites, both raised rather than reported, both direct analogues of ADRN-1 defect 1.

    (a) THE DIGITAL HALF MUST BE TRAINED. If the encoder and workspace are at chance on the offline
        task WITH the context supplied, then every adapter downstream is fitting noise and every
        number this file could print would be a number about nothing. ADRN-1 shipped a network that
        was silent and still returned scores; an untrained encoder is the same failure wearing
        different clothes, and at SMOKE settings it genuinely happened (0.30 vs chance 0.25).
    (b) THE TASK MUST BE SOLVABLE FROM THE WORKSPACE STATE. A ridge probe fitted per ONLINE context
        on w must beat chance. If it cannot, no local rule could either, and a null would say
        nothing about the local rule -- which is exactly the unreadable null this project keeps
        shipping.
    """
    X, y, slot = env.pretrain_batch(384, rng)
    h = model.encode(X)
    out, _ = model.workspace_act(h, model.mod_vector(False, len(X)), model.ctx_emb(slot))
    off = float((out.argmax(-1) == y).float().mean())
    probes = []
    for c in range(env.k_true):
        xs, ys = env.online(c, 128, rng)
        hh = model.encode(xs)
        ws = torch.stack([_aug(model.workspace(hh[i:i + 1], model.mod_vector(False),
                                               model.ctx_input(model.p), hard_halt=True)[0])[0]
                          for i in range(len(xs))])
        T = F.one_hot(ys, cfg.n_classes).float()
        sol = torch.linalg.lstsq(ws.T @ ws + 1e-2 * torch.eye(ws.shape[1]), ws.T @ T).solution
        probes.append(float(((ws @ sol).argmax(-1) == ys).float().mean()))
    probe = float(np.mean(probes))
    bad = []
    if off < cfg.chance + 0.15:
        bad.append(f"the DIGITAL HALF is untrained: offline held-out accuracy with the context "
                   f"supplied is {off:.4f} against chance {cfg.chance:.3f} and an analytic ceiling of "
                   f"{env.analytic_ceiling:.3f}. Every adapter below would be fitting noise.")
    if probe < cfg.chance + 0.15:
        bad.append(f"the TASK is not solvable from the workspace state: a ridge probe fitted per "
                   f"online context reaches only {probe:.4f} against chance {cfg.chance:.3f}. A null "
                   f"from any local rule would be uninterpretable.")
    if bad:
        raise MechanismInert(f"[{tag}] PREREQUISITE FAILED before any score was reported:\n  - "
                             + "\n  - ".join(bad))
    return {"offline_heldout_acc": off, "workspace_probe_acc": probe, "probe_per_context": probes,
            "analytic_ceiling": env.analytic_ceiling}


# ==================================================================================================
# ONLINE STREAM -- the only place any adaptation happens after pretraining
# ==================================================================================================
def new_tel():
    return {"correct": [], "post": [], "steps": [], "wstd": [], "eta": [], "lam": [], "g": [],
            "dWf_rel": [], "dWf_abs": [], "n_consol": 0, "created_at": [], "n_elig": 0,
            "mod": [], "ctx_true": [], "argmax": [], "block_start": [], "revisit_mask": [],
            # slots ALLOCATED at each step. The uniformity bound on the posterior is log of THIS,
            # not log of the slot count at the end of the stream: a posterior that is exactly
            # uniform at every step still sits far below the final log k while the manager is
            # still allocating, and comparing against the final value let that pass silently.
            "used_at": []}


def run_stream(model: ADRN2A, env: ContextualStream, blocks, cfg: Cfg, rng, delay: int,
               fixed_eta=0.15, lr=1e-2):
    """One pass of the online stream. Batch size is ONE example per step, so every local rule is exact.

    Prequential protocol: predict -> score -> (delay) -> learn. Nothing is scored after being learned
    from, so `online_acc` is a held-out value at every single step.
    """
    tel = new_tel()
    model.reset_online()
    flags = model.flags
    elig_on = flags.eligibility and delay > 0
    elig = torch.zeros(model.d_w + 1)
    fifo = deque()
    seen, step = set(), 0

    if flags.sgd != "none":
        ps = list(model.dec.parameters()) if flags.sgd == "head" else \
            list(model.dec.parameters()) + list(model.enc.parameters()) + \
            list(model.w0.parameters()) + list(model.fz.parameters()) + list(model.fn.parameters())
        for p in model.parameters():
            p.requires_grad_(False)
        for p in ps:
            p.requires_grad_(True)
        opt = torch.optim.Adam(ps, lr=lr)

    for bi, (ctx, ns) in enumerate(blocks):
        revisit = ctx in seen
        seen.add(ctx)
        tel["block_start"].append((step, ctx, revisit))
        for _ in range(ns):
            x, y = env.online(ctx, 1, rng)
            with torch.no_grad():
                h = model.encode(x)
                prior = model.transition_prior()
                if flags.oracle_context and model.k > 1:
                    prior = torch.zeros(model.k)
                    prior[ctx % model.k] = 1.0
                    model.used[ctx % model.k] = True
                c_in = model.ctx_input(prior)
                m = model.mod_vector(delay > 0)
                rec = []
                w, nst = model.workspace(h, m, c_in, hard_halt=True, record=rec)
                wa = _aug(w)[0]
                pre = model.posterior_pre(prior, w[0], step)
                if flags.oracle_context and model.k > 1:
                    pre = prior
                logits = model.logits_mix(wa, pre)
                pred = int(logits.argmax())
            ok = int(pred == int(y))
            tel["correct"].append(ok)
            tel["post"].append(pre.numpy().copy())
            tel["used_at"].append(int(model.used.sum()))
            tel["argmax"].append(int(pre.argmax()))
            tel["ctx_true"].append(ctx)
            tel["revisit_mask"].append(int(revisit))
            tel["steps"].append(rec[0])
            if len(rec) > 1:
                tel["wstd"].append(rec[1])
            tel["mod"].append(m[0].numpy().copy())
            model.acc_ema = 0.98 * model.acc_ema + 0.02 * ok
            model.err_ema = 0.98 * model.err_ema + 0.02 * float(
                F.cross_entropy(logits.unsqueeze(0), y))

            # ELIGIBILITY, and ONLY when feedback is genuinely delayed (spec sec 7.2). When the label
            # arrives on the same step the trace is not maintained at all and costs nothing; that is
            # asserted, not assumed.
            if elig_on:
                elig = cfg.lam_e * elig + (1 - cfg.lam_e) * wa
                tel["n_elig"] += 1

            fifo.append((x, y, wa))
            if len(fifo) > delay:
                x_d, y_d, wa_d = fifo.popleft()
                credit = elig if elig_on else wa
                if flags.sgd != "none":
                    # The conventional control gets the EXACT stored example and a true gradient step.
                    # That is deliberately stronger than the local arms, which only ever see a trace.
                    h2 = model.encode(x_d)
                    m2 = model.mod_vector(delay > 0)
                    w2, _ = model.workspace(h2, m2, model.ctx_input(model.p), hard_halt=True)
                    loss = F.cross_entropy(model.dec(_aug(w2)), y_d)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(ps, 1.0)
                    opt.step()
                    tel["eta"].append(lr)
                    tel["lam"].append(0.0)
                    tel["g"].append(0.0)
                else:
                    with torch.no_grad():
                        post, _ = model.posterior_post(pre, credit, y_d, step)
                        if flags.oracle_context and model.k > 1:
                            post = pre
                        eta, lam, g = model.rates(w[0], ok, _ent_t(post), fixed_eta)
                        tel["eta"].append(eta)
                        tel["lam"].append(lam)
                        tel["g"].append(g)
                        model.local_update(credit, y_d, post, eta, lam, g, tel)
                        model.bookkeeping(post, w[0], ok)
                        model.maybe_create(step, w[0], tel)
            step += 1
    return tel


@torch.no_grad()
def retention_eval(model: ADRN2A, env: ContextualStream, cfg: Cfg, rng, delay):
    """RETENTION: IDENTIFY the context from a few labelled examples, then apply the STORED adapter.

    Protocol per context, after the stream has ended and ALL learning is switched off:
      1. reset the posterior to uniform over used slots;
      2. show `cfg.warm` labelled examples that move ONLY the posterior -- no adapter, no key and no
         utility is touched, so nothing is re-learned;
      3. score `cfg.eval_n` FRESH examples with the posterior and every adapter frozen.
    Only step 3 is scored, and those examples are never used for anything else, so the number is a
    clean held-out value.

    The warm-up has to be LABELLED because in this environment the contexts differ ONLY in the label
    mapping: the input distribution is identical across contexts, so the workspace state carries no
    information about which context is active and a label-free retrieval test would be at chance for
    every arm no matter how well its adapters had been stored. That would have measured nothing.
    """
    accs, warm_slots = [], []
    for c in range(env.k_true):
        model.p = model.used.float() / max(1.0, float(model.used.sum()))
        model.protect_slot, model.protect_until = -1, -10 ** 9

        def step_once(update_posterior, y=None):
            x, yy = env.online(c, 1, rng)
            h = model.encode(x)
            prior = model.transition_prior()
            if model.flags.oracle_context and model.k > 1:
                prior = model.one_hot_slot(c % model.k)
            w, _ = model.workspace(h, model.mod_vector(delay > 0), model.ctx_input(prior),
                                   hard_halt=True)
            pre = prior if model.flags.oracle_context else model.posterior_pre(prior, w[0])
            wa = _aug(w)[0]
            ok = int(int(model.logits_mix(wa, pre).argmax()) == int(yy))
            if update_posterior:
                keep = (model.loss_ema.clone(), model.loss_fast.clone(), model.best_seen.clone(),
                        model.util.clone(), model.base_loss_ema)
                post, _ = model.posterior_post(pre, wa, yy)
                (model.loss_ema, model.loss_fast, model.best_seen, model.util,
                 model.base_loss_ema) = keep       # identification only: no state is learned
                model.p = post
            else:
                model.p = pre
            return ok

        for _ in range(cfg.warm):
            step_once(True)
        warm_slots.append(int(model.p.argmax()))
        accs.append(sum(step_once(False) for _ in range(cfg.eval_n)) / cfg.eval_n)
    return float(np.mean(accs)), {"per_context": accs, "identified_slot": warm_slots}


# ==================================================================================================
# MECHANISM LIVENESS -- runs BEFORE any score is allowed into the results table
# ==================================================================================================
def assert_liveness(arm, model: ADRN2A, tel, cfg: Cfg, delay, enc_sig_before, enc_sig_after):
    """Raise unless every mechanism this arm CLAIMS is numerically alive. No claim, no check; but
    every claim is checked, with a number, and the numbers are returned for the record."""
    f, bad, rep = model.flags, [], {}

    # -- 1. fast adapters must actually change between updates -----------------------------------
    if f.fast:
        rel = np.asarray(tel["dWf_rel"], np.float64)
        rep["fast_updates"] = int(rel.size)
        rep["fast_dW_rel_median"] = float(np.median(rel)) if rel.size else 0.0
        rep["fast_norm_final"] = float(model.W_f.norm())
        if rel.size == 0:
            bad.append("fast adapter: ZERO updates were applied over the whole stream")
        elif rep["fast_dW_rel_median"] < 1e-6:
            bad.append(f"fast adapter: median relative change per update is "
                       f"{rep['fast_dW_rel_median']:.3g} -- the adapter is not moving")
        if rep["fast_norm_final"] <= 1e-9:
            bad.append("fast adapter: final ||W_f|| is zero -- nothing was ever written")
    else:
        rep["fast_norm_final"] = float(model.W_f.norm()) if model.W_f.numel() else 0.0
        if rep["fast_norm_final"] > 1e-9:
            bad.append("fast adapter is DISABLED for this arm but W_f is non-zero")

    # -- 2. context posterior must be non-degenerate ----------------------------------------------
    if f.context and model.k > 1 and not f.oracle_context:
        P = np.asarray(tel["post"], np.float64)
        U = np.asarray(tel["used_at"], np.int64)
        marg = P.mean(0)
        ku = max(2, int(model.used.sum()))
        ent_t = np.array([_ent(p) for p in P])
        h_marg, h_cond = _ent(marg), float(ent_t.mean())
        # HEADROOM below the entropy AVAILABLE at that step, over the steps where more than one slot
        # existed to be uncertain between. Comparing against log of the FINAL slot count let a
        # posterior that was forced to be exactly uniform pass, because while the manager is still
        # allocating the achievable entropy is far below the final log k. Verified by mutation.
        multi = U >= 2
        headroom = (float(np.mean(np.log(np.maximum(2, U[multi])) - ent_t[multi]))
                    if multi.any() else float("nan"))
        n_argmax = int(len(set(P.argmax(1).tolist())))
        rep.update(posterior_marginal_entropy=h_marg, posterior_conditional_entropy=h_cond,
                   posterior_information=h_marg - h_cond, contexts_created=len(tel["created_at"]),
                   contexts_used=int(model.used.sum()), log_k_used=math.log(ku),
                   posterior_entropy_headroom_per_step=headroom,
                   posterior_multi_slot_steps=int(multi.sum()),
                   posterior_distinct_argmax_slots=n_argmax,
                   created_at=list(tel["created_at"]),
                   slot_utility=[round(float(v), 4) for v in model.util],
                   slot_best_loss=[round(float(v), 4) for v in model.best_seen],
                   base_loss_ema=round(float(model.base_loss_ema), 4))
        if h_marg < 0.25:
            bad.append(f"context posterior COLLAPSED: entropy of the time-averaged posterior is "
                       f"{h_marg:.4f} nats -- effectively one context for the whole stream")
        if multi.any() and headroom <= 0.15:
            bad.append(f"context posterior is UNIFORM: over the {int(multi.sum())} steps at which "
                       f"more than one slot existed, its mean entropy sits only {headroom:.4f} nats "
                       f"below the maximum entropy AVAILABLE at that step -- it is not deciding "
                       f"anything")
        if h_marg - h_cond < 0.05:
            bad.append(f"context posterior carries no information: marginal - conditional entropy = "
                       f"{h_marg - h_cond:.4f} nats")
        if int(model.used.sum()) >= 2 and n_argmax < 2:
            bad.append(f"context posterior never SELECTS a second context: over {len(P)} steps its "
                       f"argmax took {n_argmax} distinct value(s) while {int(model.used.sum())} slots "
                       f"were allocated -- one adapter wearing a context manager's name")
        if len(tel["created_at"]) < 1:
            bad.append("context CREATION never fired: the manager ran the whole stream on one slot")

    # -- 3. consolidated adapters must exist and must DIFFER from the fast adapters ---------------
    if f.consolidate:
        nm, nf = float(model.W_m.norm()), float(model.W_f.norm())
        d = float((model.W_m - model.W_f).norm()) / (nm + nf + 1e-12) if model.W_f.numel() else 1.0
        rep.update(n_consolidations=tel["n_consol"], W_m_norm=nm, W_m_vs_W_f_rel=d)
        if tel["n_consol"] == 0:
            bad.append("consolidation NEVER fired: W_m is a permanently empty store")
        if nm <= 1e-9:
            bad.append("consolidated adapter norm is zero")
        if d < 1e-3:
            bad.append(f"consolidated and fast adapters are indistinguishable (rel diff {d:.3g}) -- "
                       f"consolidation is a copy, not a second timescale")
    elif model.W_m.numel() and float(model.W_m.norm()) > 1e-9:
        bad.append("consolidation is DISABLED for this arm but W_m is non-zero")

    # -- 4. halting must not simply run to the cap -------------------------------------------------
    st = np.asarray(tel["steps"], np.float64)
    rep.update(mean_steps=float(st.mean()), max_steps_cap=cfg.max_steps,
               steps_at_cap_frac=float((st >= cfg.max_steps).mean()), steps_sd=float(st.std()))
    if f.halting and f.workspace:
        if rep["steps_at_cap_frac"] >= 1.0 - 1e-12:
            bad.append(f"adaptive halting is INERT: every one of {st.size} steps ran to the cap "
                       f"({cfg.max_steps}); the halting head is never used")
    if not f.halting and f.workspace and rep["steps_at_cap_frac"] < 1.0 - 1e-12:
        bad.append("halting is DISABLED for this arm but the workspace did not always run to the cap")

    # -- 5. plasticity controller must not be emitting a constant ---------------------------------
    if f.controller and f.sgd == "none":
        e, l = np.asarray(tel["eta"]), np.asarray(tel["lam"])
        rep.update(eta_mean=float(e.mean()), eta_sd=float(e.std()), lam_mean=float(l.mean()),
                   lam_sd=float(l.std()),
                   # A non-zero sd is not the same as a mechanism doing work. The coefficient of
                   # variation is recorded so that "learned plasticity" cannot be claimed off a
                   # network whose output is flat to a fraction of a percent; a 1e-8 threshold does
                   # not distinguish that case from a live controller.
                   eta_cv=float(e.std() / abs(e.mean())) if e.size and abs(e.mean()) > 0 else 0.0,
                   lam_cv=float(l.std() / abs(l.mean())) if l.size and abs(l.mean()) > 0 else 0.0)
        if e.size == 0:
            bad.append("plasticity controller was never queried")
        elif e.std() < 1e-8 or l.std() < 1e-8:
            bad.append(f"plasticity controller is CONSTANT (sd eta {e.std():.3g}, sd lambda "
                       f"{l.std():.3g}) -- it is a hard-coded number wearing a network")

    # -- 6. eligibility must be gated on the feedback delay, in BOTH directions -------------------
    rep["elig_updates"] = tel["n_elig"]
    if f.eligibility and delay > 0 and f.sgd == "none" and tel["n_elig"] == 0:
        bad.append("eligibility is enabled and feedback IS delayed, but the trace was never updated")
    if delay == 0 and tel["n_elig"] != 0:
        bad.append("eligibility ran with IMMEDIATE feedback -- spec sec 7.2 forbids paying that cost")

    # -- 7. the workspace state must actually move across internal steps --------------------------
    if f.workspace:
        ws = np.asarray(tel["wstd"], np.float64)
        rep["workspace_step_sd"] = float(ws.mean()) if ws.size else 0.0
        if ws.size and rep["workspace_step_sd"] < 1e-8:
            bad.append("recurrent workspace is INERT: w does not change across internal steps")

    # -- 8. the modulatory vector must not be constant --------------------------------------------
    mm = np.asarray(tel["mod"], np.float64)
    rep["mod_sd"] = float(mm.std(0).max())
    if rep["mod_sd"] < 1e-8:
        bad.append("the modulatory vector m_t is constant for the whole stream")

    # -- 9. the encoder must be frozen for local arms and must have MOVED for sgd_full ------------
    rep["encoder_moved"] = abs(enc_sig_after - enc_sig_before)
    if f.sgd != "full" and rep["encoder_moved"] > 1e-9:
        bad.append(f"the encoder MOVED during online learning ({rep['encoder_moved']:.3g}); this arm "
                   f"claims no backprop through the encoder")
    if f.sgd == "full" and rep["encoder_moved"] <= 1e-9:
        bad.append("sgd_full claims to fine-tune the encoder online but the encoder did not move")

    if bad:
        raise MechanismInert(f"[{arm}] MECHANISM LIVENESS FAILED before any score was reported:\n  - "
                             + "\n  - ".join(bad))
    return rep


def flag_effect_probe(cfg: Cfg, env, model: ADRN2A, blocks, seed, delay):
    """Toggle each component on ONE pretrained model and prove the toggle changes the OUTPUT.

    ADRN-1's third defect was an ablation flag that changed nothing (prune_every defaulted to 0) and
    its fourth was three ablations returning byte-identical numbers. This makes both impossible: a
    flag that does not measurably move the logits raises here, long before any table is printed.
    """
    out = {}
    for name in ("workspace", "halting", "context", "fast", "consolidate", "controller",
                 "eligibility"):
        if name == "eligibility" and delay == 0:
            continue
        res = {}
        for val in (True, False):
            m = copy.deepcopy(model)
            setattr(m.flags, name, val)
            if name == "controller" and val is False:
                m.controller = None
            tel = run_stream(m, env, blocks, cfg, np.random.default_rng(90000 + seed), delay)
            res[val] = (np.asarray(tel["correct"], np.float64), m)
        a, b = res[True][0], res[False][0]
        n_diff = int((a != b).sum())
        # a per-step logit comparison on a fixed probe batch, which is stricter than the accuracy trace
        xs, ys = env.online(0, 32, np.random.default_rng(77))
        lg = []
        for mm in (res[True][1], res[False][1]):
            h = mm.encode(xs)
            L = []
            for i in range(len(xs)):
                w, _ = mm.workspace(h[i:i + 1], mm.mod_vector(delay > 0),
                                    mm.ctx_input(mm.p), hard_halt=True)
                L.append(mm.logits_mix(_aug(w)[0], mm.p))
            lg.append(torch.stack(L))
        dl = float((lg[0] - lg[1]).abs().max())
        out[name] = {"steps_differing": n_diff, "max_logit_delta": dl,
                     "acc_on": float(a.mean()), "acc_off": float(b.mean())}
        if n_diff == 0 and dl < 1e-6:
            raise MechanismInert(
                f"FLAG '{name}' IS VACUOUS: switching it on and off produced an identical prediction "
                f"trace ({n_diff} steps differ) AND identical logits (max delta {dl:.3g}). An ablation "
                f"of a mechanism that is not running is not an ablation.")
    return out


def assert_arms_distinct(rows, expect_identical, cond):
    """No two differently-configured arms may return identical per-seed metrics -- except where the
    spec says they MUST (eligibility under immediate feedback, which is gated off and so must cost
    exactly nothing). Both directions are enforced."""
    names = sorted(rows)
    bad = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            same = (np.allclose(rows[a]["online_acc"], rows[b]["online_acc"], atol=0, rtol=0) and
                    np.allclose(rows[a]["retention_acc"], rows[b]["retention_acc"], atol=0, rtol=0))
            must = tuple(sorted((a, b))) in expect_identical
            if same and not must:
                bad.append(f"'{a}' and '{b}' returned BYTE-IDENTICAL per-seed metrics in the {cond} "
                           f"condition -- this is ADRN-1's defect-4 signature")
            if must and not same:
                bad.append(f"'{a}' and '{b}' MUST be identical in the {cond} condition (eligibility is "
                           f"gated off when feedback is immediate, so it must cost nothing) but they "
                           f"differ: {rows[a]['online_acc']} vs {rows[b]['online_acc']}")
    if bad:
        raise MechanismInert("ARM DISTINCTNESS FAILED:\n  - " + "\n  - ".join(bad))


# ==================================================================================================
# ARMS
# ==================================================================================================
ARMS = {
    # ---- controls -----------------------------------------------------------------------------
    "frozen": Flags(workspace=True, halting=True, context=False, fast=False, consolidate=False,
                    controller=False, eligibility=False, sgd="none"),
    "sgd_head": Flags(context=False, fast=False, consolidate=False, controller=False,
                      eligibility=False, sgd="head"),
    "sgd_full": Flags(context=False, fast=False, consolidate=False, controller=False,
                      eligibility=False, sgd="full"),
    "global_fast": Flags(context=False),
    # ---- ablations of ADRN-2A -------------------------------------------------------------------
    "no_consolidation": Flags(consolidate=False),
    "no_controller": Flags(controller=False),
    "no_workspace": Flags(workspace=False, halting=False),
    "no_halting": Flags(halting=False),
    "no_eligibility": Flags(eligibility=False),
    # ---- the proposal, and its oracle upper reference --------------------------------------------
    "adrn2a_full": Flags(),
    "oracle_context": Flags(oracle_context=True),
}
CONTROLS = ["frozen", "sgd_head", "sgd_full", "global_fast"]
CALIBRATE = {"sgd_head": [3e-3, 1e-2, 3e-2], "sgd_full": [3e-4, 1e-3, 3e-3],
             "no_controller": [0.05, 0.15, 0.40]}


def width_for(cfg, flags, budget, lo=8, hi=160):
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        n = ADRN2A(cfg, flags, mid).param_counts()[2]
        if n <= budget:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def enc_sig(model):
    return float(sum(p.detach().abs().sum() for p in model.enc.parameters()))


# ==================================================================================================
# MAIN
# ==================================================================================================
def main():
    t0 = time.time()
    cfg = Cfg()
    log = []

    def rep(s=""):
        print(s, flush=True)
        log.append(s)

    rep("=" * 108)
    rep("ADRN-2A -- context manager + fast/consolidated adapters on a recurring-context stream")
    rep("=" * 108)
    if SMOKE:
        rep("  *** SMOKE RUN -- plumbing and liveness check, NOT a result. ***")
    rep(f"  GATE, declared before any number exists: an arm is credited only if its PAIRED per-seed")
    rep(f"  gap over the comparison arm exceeds MDE = 3*sd/sqrt(n), n = {cfg.seeds} SEEDS, where one")
    rep(f"  seed is an independent initialisation AND an independent environment draw. The effect size")
    rep(f"  reported everywhere is the RAW held-out prequential accuracy, never net of a control.")
    rep(f"  Ceiling {cfg.ceiling:.2f}; floor (chance) {cfg.chance:.3f}; parameter budget "
        f"{cfg.budget:,} matched by bisection per arm, realised counts printed.")

    widths, pcounts = {}, {}
    for a, f in ARMS.items():
        widths[a] = width_for(cfg, f, cfg.budget)
        tr, ad, tot = ADRN2A(cfg, f, widths[a]).param_counts()
        pcounts[a] = {"width_d_h": widths[a], "trainable": tr, "adaptive": ad, "total": tot}
    rep("\n  PARAMETER BUDGET (bisected per architecture; adapters counted as capacity)")
    for a in ARMS:
        p = pcounts[a]
        rep(f"    {a:18s} d_h {p['width_d_h']:>4}   trainable {p['trainable']:>7,}"
            f"   adaptive {p['adaptive']:>6,}   total {p['total']:>7,}")

    conditions = [("immediate", 0), ("delayed", cfg.delay)]
    results = {c: {a: {"online_acc": [], "retention_acc": [], "revisit_acc": [], "mean_steps": [],
                       "steps_sd": [], "n_contexts": [], "calibrated": []} for a in ARMS}
               for c, _ in conditions}
    liveness = {}
    probe_done = False
    analytic_ceiling = None

    for seed in range(cfg.seeds):
        env_rng = np.random.default_rng(20000 + 977 * seed)
        env = ContextualStream(env_rng, cfg)
        analytic_ceiling = env.analytic_ceiling
        blocks = make_stream(env_rng, cfg, env.k_true)
        rep(f"\n  ── seed {seed}  ({len(blocks)} blocks x {cfg.block} steps = "
            f"{sum(b[1] for b in blocks)} online examples, {env.k_true} recurring contexts) ──")

        # Pretraining depends only on (width, controller-present) -- every other flag changes only
        # buffers and runtime behaviour -- so arms sharing an architecture share ONE pretrained
        # digital half. Identical weights, not merely equivalent ones, which also removes pretraining
        # variance as a confound between those arms.
        pre_cache, by_key = {}, {}
        for arm, flags in ARMS.items():
            key = (widths[arm], flags.controller)
            if key not in by_key:
                attempt, proto, pr, mt, dc = 0, None, None, None, None
                while True:
                    torch.manual_seed(4000 + seed + 100000 * attempt)
                    proto = ADRN2A(cfg, flags, widths[arm])
                    pr = pretrain(proto, env, cfg, np.random.default_rng(31000 + seed + 7 * attempt))
                    mt = meta_train_controller(proto, env, cfg, np.random.default_rng(41000 + seed))
                    try:
                        # PREREQUISITE, RAISED NOT REPORTED: the digital half must be trained and the
                        # workspace must make the task linearly solvable, or no score below means
                        # anything. Retried on a fresh init a bounded number of times because a GRU
                        # occasionally lands in a bad basin; the criterion is the OFFLINE task with
                        # the context supplied, which no online mechanism under test can influence,
                        # so the retry cannot favour any arm.
                        dc = assert_digital_competence(
                            proto, env, cfg, np.random.default_rng(51000 + seed), f"{arm}/seed{seed}")
                        break
                    except MechanismInert:
                        attempt += 1
                        if attempt > cfg.pre_restarts:
                            raise
                dc["pretrain_restarts"] = attempt
                by_key[key] = ({k: v for k, v in proto.state_dict().items()
                                if not k.startswith(("W_f", "W_m", "keys"))}, pr, mt, dc)
            sd, pr, mt, dc = by_key[key]
            torch.manual_seed(4000 + seed)
            model = ADRN2A(cfg, flags, widths[arm])
            model.load_state_dict(sd, strict=False)
            liveness.setdefault("digital_competence", {}).setdefault(arm, []).append(dc)
            pre_cache[arm] = (model, pr, mt)
        rep(f"    digital half: offline held-out (context known) "
            f"{np.mean([liveness['digital_competence'][a][-1]['offline_heldout_acc'] for a in ARMS]):.4f}"
            f", per-context ridge probe on w "
            f"{np.mean([liveness['digital_competence'][a][-1]['workspace_probe_acc'] for a in ARMS]):.4f}"
            f", analytic ceiling {env.analytic_ceiling:.3f}, chance {cfg.chance:.3f}, "
            f"restarts {sum(liveness['digital_competence'][a][-1]['pretrain_restarts'] for a in ARMS)}")

        for cond, delay in conditions:
            for arm, flags in ARMS.items():
                base, pr, mt = pre_cache[arm]
                # -- calibrate any free scalar step size on a SEPARATE validation environment draw.
                # A control that loses because nobody tuned it is defect 2 and is not allowed here.
                lr, feta, cal = 1e-2, 0.15, None
                if arm in CALIBRATE:
                    val_rng = np.random.default_rng(55000 + seed)
                    val_env = ContextualStream(val_rng, cfg)
                    val_blocks = make_stream(val_rng, cfg, val_env.k_true)[:cfg.probe_blocks]
                    best = (-1, None)
                    for v in CALIBRATE[arm]:
                        m = copy.deepcopy(base)
                        t = run_stream(m, val_env, val_blocks, cfg,
                                       np.random.default_rng(66000 + seed), delay,
                                       fixed_eta=v, lr=v)
                        s = float(np.mean(t["correct"]))
                        if s > best[0]:
                            best = (s, v)
                    cal = best[1]
                    lr = feta = cal

                model = copy.deepcopy(base)
                sig0 = enc_sig(model)
                tel = run_stream(model, env, blocks, cfg, np.random.default_rng(70000 + seed),
                                 delay, fixed_eta=feta, lr=lr)
                sig1 = enc_sig(model)

                # ---- LIVENESS RUNS HERE, BEFORE THE SCORE IS ALLOWED INTO THE TABLE ----
                lv = assert_liveness(f"{arm}/{cond}/seed{seed}", model, tel, cfg, delay, sig0, sig1)
                liveness.setdefault(cond, {}).setdefault(arm, lv)

                ret, retinfo = retention_eval(model, env, cfg, np.random.default_rng(80000 + seed),
                                              delay)
                lv["retention_detail"] = retinfo
                corr = np.asarray(tel["correct"], np.float64)
                rv = [i for s0, _, r in tel["block_start"] if r
                      for i in range(s0, min(s0 + cfg.probe_len, len(corr)))]
                results[cond][arm]["online_acc"].append(float(corr.mean()))
                results[cond][arm]["retention_acc"].append(ret)
                results[cond][arm]["revisit_acc"].append(float(corr[rv].mean()) if rv else float("nan"))
                results[cond][arm]["mean_steps"].append(float(np.mean(tel["steps"])))
                results[cond][arm]["steps_sd"].append(float(np.std(tel["steps"])))
                results[cond][arm]["n_contexts"].append(int(model.used.sum()))
                results[cond][arm]["calibrated"].append(cal)

            if not probe_done and seed == 0:
                rep(f"\n    FLAG-EFFECT PROBE ({cond}): toggling each component on one pretrained "
                    f"model and requiring the output to move")
                pm = copy.deepcopy(pre_cache["adrn2a_full"][0])
                # The probe replays the FULL stream, not a truncation. On a two-block truncation the
                # consolidate flag legitimately changed nothing under delayed feedback -- the
                # consolidation gate had not fired yet -- and the probe raised on a mechanism that is
                # in fact live over the real workload. A liveness probe has to exercise the same
                # workload the scored arms do or it tests the probe's own length.
                fx = flag_effect_probe(cfg, env, pm, blocks, seed, delay)
                for n, v in fx.items():
                    rep(f"      {n:14s} steps differing {v['steps_differing']:>5}   max|dlogit| "
                        f"{v['max_logit_delta']:9.4f}   acc on/off {v['acc_on']:.3f}/{v['acc_off']:.3f}")
                liveness.setdefault("flag_effect", {})[cond] = fx
                if cond == "delayed":
                    probe_done = True

    # ============================================================== scoring, gates, ceiling/floor
    def stats(v):
        v = np.asarray(v, np.float64)
        return float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0

    def paired(cond, a, b, metric="online_acc"):
        x = np.asarray(results[cond][a][metric], np.float64)
        y = np.asarray(results[cond][b][metric], np.float64)
        d = x - y
        sd = float(np.nanstd(d, ddof=1)) if d.size > 1 else 0.0
        return float(np.nanmean(d)), 3 * sd / math.sqrt(max(1, d.size))

    def credited(gap, mde, n):
        """'PASS' | 'FAIL' | 'UNSUPPORTED' for a gap that must exceed its MDE.

        A paired per-seed sample sd of exactly zero does not mean the population sd is zero; it means
        n seeds landed on the same difference, which is easy when the metric is a count over a fixed
        stream. It yields MDE = 0.0000, and then any positive gap clears it. This file has already
        shipped that: 'immediate | G3 consolidation contributes' passed on gap +0.1000 against MDE
        0.0000 with n=2, and it was the only substantive PASS in its condition. A non-positive gap
        still FAILS without needing an MDE; a positive gap with an unestimable MDE is UNSUPPORTED.
        """
        if not (gap > 0):
            return "FAIL"
        if n < 2 or not np.isfinite(mde) or mde <= 0.0:
            return "UNSUPPORTED"
        return "PASS" if gap > mde else "FAIL"

    def floor_ceiling(cond, metric, chance, mde_ref):
        """Ceiling AND floor adjudication for ONE metric, with the DECLARED max(MDE, 0.02) tolerance.

        This used to be computed for `online_acc` only, while G1b and G3b are read on `retention_acc`
        -- so half the gates were adjudicated against no ceiling/floor check at all, on a metric that
        in this file's own smoke sat within 0.15 of chance for every arm. The declared rule is per
        condition and says nothing about being restricted to one metric, so it is applied to both.
        """
        vals = {a: stats(results[cond][a][metric])[0] for a in ARMS}
        best_arm = max(vals, key=vals.get)
        tol = max(mde_ref, 0.02) if np.isfinite(mde_ref) else 0.02
        all_floor = all(abs(v - chance) <= tol for v in vals.values())
        hitters = [a for a, v in vals.items() if v >= cfg.ceiling]
        below = []
        if hitters:
            for a in ARMS:
                if a in hitters:
                    continue
                g, m = paired(cond, best_arm, a, metric)
                if g > m > 0:
                    below.append((a, vals[a]))
        return {"metric": metric, "per_arm": vals, "best_arm": best_arm,
                "best": vals[best_arm], "chance": chance, "floor_tolerance": tol,
                "at_ceiling": bool(vals[best_arm] >= cfg.ceiling), "ceiling_hitters": hitters,
                "informative_below_ceiling": below, "all_at_floor": bool(all_floor),
                "uninformative": bool(all_floor or (vals[best_arm] >= cfg.ceiling
                                                    and len(hitters) > 1 and not below))}

    gates, gapinfo, cf, unsupported_gates = {}, {}, {}, {}
    for cond, delay in conditions:
        rep("\n" + "=" * 108)
        rep(f"  CONDITION: {cond} feedback (delay = {delay} steps)")
        rep("  " + "-" * 106)
        rep(f"    {'arm':18s} {'online_acc':>18s} {'retention':>12s} {'revisit':>10s} "
            f"{'steps':>13s} {'ctx':>5s} {'params':>9s}")
        flat = []
        for arm in ARMS:
            m, s = stats(results[cond][arm]["online_acc"])
            rm, _ = stats(results[cond][arm]["retention_acc"])
            vm, _ = stats(results[cond][arm]["revisit_acc"])
            st, _ = stats(results[cond][arm]["mean_steps"])
            ss, _ = stats(results[cond][arm]["steps_sd"])
            nc, _ = stats(results[cond][arm]["n_contexts"])
            if ARMS[arm].halting and ARMS[arm].workspace and ss < 1e-9:
                flat.append(arm)
            rep(f"    {arm:18s} {m:>9.4f} ± {s:<6.4f} {rm:>12.4f} {vm:>10.4f} "
                f"{st:>6.2f}±{ss:<6.2f} {nc:>5.1f} {pcounts[arm]['total']:>9,}")
        if flat:
            rep(f"      NOTE: halting is live (never at the {cfg.max_steps}-step cap) but emits a "
                f"CONSTANT step count for {', '.join(flat)}. It is not costing anything and it is not "
                f"adapting either; treat 'adaptive halting' for those arms as unsupported.")

        # CEILING AND FLOOR, PER METRIC. Previously this was computed for `online_acc` alone while
        # G1b and G3b are read on `retention_acc` -- so half the gates were adjudicated against no
        # ceiling/floor check at all. The declared rule is also max(MDE, 0.02) of chance, not a bare
        # 0.02, and the MDE term had been dropped.
        g1m = paired(cond, "adrn2a_full", max(CONTROLS,
                     key=lambda a: stats(results[cond][a]["online_acc"])[0]))[1]
        g1bm = paired(cond, "adrn2a_full", max(CONTROLS,
                      key=lambda a: stats(results[cond][a]["retention_acc"])[0]), "retention_acc")[1]
        cf_on = floor_ceiling(cond, "online_acc", cfg.chance, g1m)
        cf_ret = floor_ceiling(cond, "retention_acc", cfg.chance, g1bm)
        cf[cond] = {"online_acc": cf_on, "retention_acc": cf_ret,
                    "analytic_ceiling": analytic_ceiling,
                    # kept for continuity with earlier records
                    "best_arm": cf_on["best_arm"], "best_online_acc": cf_on["best"],
                    "chance": cfg.chance, "at_ceiling": cf_on["at_ceiling"],
                    "ceiling_hitters": cf_on["ceiling_hitters"],
                    "informative_below_ceiling": cf_on["informative_below_ceiling"],
                    "all_at_floor": cf_on["all_at_floor"],
                    "uninformative": cf_on["uninformative"]}
        for tag, d in (("online_acc", cf_on), ("retention_acc", cf_ret)):
            if d["all_at_floor"]:
                rep(f"\n    *** FLOOR on {tag}: every arm is within {d['floor_tolerance']:.4f} of "
                    f"chance ({cfg.chance:.3f}). This metric cannot separate the arms at this scale; "
                    f"every gate read on it is UNINFORMATIVE and is not a result. ***")
            if d["at_ceiling"]:
                rep(f"\n    *** CEILING on {tag}: '{d['best_arm']}' reaches {d['best']:.4f} >= "
                    f"{cfg.ceiling}. No separation AMONG the {len(d['ceiling_hitters'])} arm(s) at "
                    f"ceiling. Arms falling BELOW it by more than their MDE remain informative: "
                    f"{', '.join(f'{a} {v:.4f}' for a, v in d['informative_below_ceiling']) or 'none'} ***")

        def record_gate(key, status):
            """PASS/FAIL enter the tally; UNSUPPORTED is recorded separately and never counted."""
            if status == "UNSUPPORTED":
                unsupported_gates[key] = "UNSUPPORTED"
            else:
                gates[key] = (status == "PASS")

        def metric_readable(met):
            return not (cf_on if met == "online_acc" else cf_ret)["all_at_floor"]

        # -------- predeclared gates --------
        bc = max(CONTROLS, key=lambda a: stats(results[cond][a]["online_acc"])[0])
        g, m = paired(cond, "adrn2a_full", bc)
        s = credited(g, m, cfg.seeds) if metric_readable("online_acc") else "UNSUPPORTED"
        record_gate(f"{cond} | G1 full beats best control ({bc}) on online_acc", s)
        gapinfo[f"{cond}|G1"] = {"vs": bc, "gap": g, "mde": m, "status": s,
                                 "full_raw": stats(results[cond]["adrn2a_full"]["online_acc"])[0],
                                 "other_raw": stats(results[cond][bc]["online_acc"])[0]}
        bcr = max(CONTROLS, key=lambda a: stats(results[cond][a]["retention_acc"])[0])
        gr, mr = paired(cond, "adrn2a_full", bcr, "retention_acc")
        sr = credited(gr, mr, cfg.seeds) if metric_readable("retention_acc") else "UNSUPPORTED"
        record_gate(f"{cond} | G1b full beats best control ({bcr}) on retention_acc", sr)
        gapinfo[f"{cond}|G1b"] = {"vs": bcr, "gap": gr, "mde": mr, "status": sr,
                                  "full_raw": stats(results[cond]["adrn2a_full"]["retention_acc"])[0],
                                  "other_raw": stats(results[cond][bcr]["retention_acc"])[0]}
        for tag, other, met in (("G2 context manager", "global_fast", "online_acc"),
                                ("G3 consolidation", "no_consolidation", "online_acc"),
                                ("G3b consolidation (retention)", "no_consolidation",
                                 "retention_acc"),
                                ("G4 plasticity controller", "no_controller", "online_acc"),
                                ("G5 recurrent workspace", "no_workspace", "online_acc")):
            g, m = paired(cond, "adrn2a_full", other, met)
            s = credited(g, m, cfg.seeds) if metric_readable(met) else "UNSUPPORTED"
            record_gate(f"{cond} | {tag} contributes", s)
            gapinfo[f"{cond}|{tag}"] = {"vs": other, "metric": met, "gap": g, "mde": m, "status": s,
                                        "full_raw": stats(results[cond]["adrn2a_full"][met])[0],
                                        "other_raw": stats(results[cond][other][met])[0]}
        st = stats(results[cond]["adrn2a_full"]["mean_steps"])[0]
        g, m = paired(cond, "adrn2a_full", "no_halting")
        # G6 is a NON-INFERIORITY gate (gap > -MDE), so a zero MDE makes it strict rather than
        # permissive and needs no UNSUPPORTED branch; only the floor check applies.
        record_gate(f"{cond} | G6 halting is live and not costly",
                    ("PASS" if (st < cfg.max_steps - 1e-9 and g > -m) else "FAIL")
                    if metric_readable("online_acc") else "UNSUPPORTED")
        gapinfo[f"{cond}|G6"] = {"mean_steps": st, "cap": cfg.max_steps, "gap": g, "mde": m}
        g, m = paired(cond, "adrn2a_full", "no_eligibility")
        if delay > 0:
            s = credited(g, m, cfg.seeds) if metric_readable("online_acc") else "UNSUPPORTED"
            record_gate(f"{cond} | G7 eligibility earns its keep under delay", s)
        else:
            ident = np.allclose(results[cond]["adrn2a_full"]["online_acc"],
                                results[cond]["no_eligibility"]["online_acc"], atol=0, rtol=0)
            gates[f"{cond} | G7 eligibility costs exactly nothing when gated off"] = bool(ident)
        gapinfo[f"{cond}|G7"] = {"gap": g, "mde": m, "delay": delay}
        og, om = paired(cond, "oracle_context", "adrn2a_full")
        gapinfo[f"{cond}|oracle_headroom"] = {"gap": og, "mde": om,
                                              "oracle_raw": stats(
                                                  results[cond]["oracle_context"]["online_acc"])[0]}

        expect_ident = {tuple(sorted(("adrn2a_full", "no_eligibility")))} if delay == 0 else set()
        assert_arms_distinct(results[cond], expect_ident, cond)

    rep("\n  PREDECLARED GATES")
    for k, v in gates.items():
        rep(f"    {'PASS' if v else 'FAIL'}  {k}")
    npass = sum(gates.values())
    if unsupported_gates:
        rep("\n  GATES THAT ARE NOT READABLE, AND ARE THEREFORE NEITHER PASSED NOR FAILED")
        rep("    (a metric on which every arm sits within max(MDE, 0.02) of chance cannot separate")
        rep("     anything, and a positive gap whose paired sample sd is exactly zero has no")
        rep("     estimable MDE. Both are excluded from the tally rather than counted either way.)")
        for k in unsupported_gates:
            rep(f"    UNSUPPORTED  {k}")

    # ============================================================== verdict
    def line(cond):
        d = gapinfo[f"{cond}|G1"]
        c = cf[cond]
        s = (f"In the {cond}-feedback condition ADRN-2A reaches a raw held-out prequential accuracy of "
             f"{d['full_raw']:.4f}, against {d['other_raw']:.4f} for the best control "
             f"({d['vs']}), a paired per-seed gap of {d['gap']:+.4f} against an MDE of {d['mde']:.4f}; "
             f"the oracle-context upper reference sits at "
             f"{gapinfo[f'{cond}|oracle_headroom']['oracle_raw']:.4f} and chance is {cfg.chance:.3f}. "
             f"Removing the context manager costs "
             f"{gapinfo[f'{cond}|G2 context manager']['gap']:+.4f} (MDE "
             f"{gapinfo[f'{cond}|G2 context manager']['mde']:.4f}), removing consolidation costs "
             f"{gapinfo[f'{cond}|G3 consolidation']['gap']:+.4f} on the stream and "
             f"{gapinfo[f'{cond}|G3b consolidation (retention)']['gap']:+.4f} on held-out retention, "
             f"removing the plasticity controller costs "
             f"{gapinfo[f'{cond}|G4 plasticity controller']['gap']:+.4f}, and removing the recurrent "
             f"workspace costs {gapinfo[f'{cond}|G5 recurrent workspace']['gap']:+.4f}. Realised "
             f"workspace steps averaged {gapinfo[f'{cond}|G6']['mean_steps']:.2f} against a cap of "
             f"{cfg.max_steps}.")
        for met in ("online_acc", "retention_acc"):
            k = c[met]
            if not k["uninformative"]:
                continue
            s += (f" THE {met} METRIC IS DECLARED UNINFORMATIVE IN THIS CONDITION, and every gate "
                  f"read on it is excluded from the tally rather than reported as a result: " +
                  (f"every arm sits within {k['floor_tolerance']:.4f} of chance "
                   f"({cfg.chance:.3f}) -- best arm {k['best_arm']} at {k['best']:.4f} -- so the "
                   f"metric separates nothing."
                   if k["all_at_floor"] else
                   f"{len(k['ceiling_hitters'])} arms are at or above the {cfg.ceiling} ceiling, so "
                   f"no separation among them can be read; arms that fall below that ceiling by more "
                   f"than their MDE ("
                   f"{', '.join(f'{a} at {v:.4f}' for a, v in k['informative_below_ceiling']) or 'none'}"
                   f") remain informative."))
        return s

    verdict = (
        ("SMOKE RUN -- plumbing and liveness check, NOT a result. " if SMOKE else "") +
        f"ADRN-2A: {npass}/{len(gates)} predeclared gates pass. " + " ".join(line(c) for c, _ in
                                                                            conditions) +
        f" Every arm was built to a matched budget of ~{cfg.budget:,} parameters by bisection with the "
        f"realised counts printed, and the per-context adapter banks are counted as capacity, so an "
        f"ablation that drops a mechanism is compensated with a wider encoder rather than quietly "
        f"handicapped; the two SGD controls additionally receive an exact stored input for delayed "
        f"feedback while the local arms receive only a decaying eligibility trace, and every free "
        f"scalar step size (both SGD controls, the no-controller ablation) was calibrated on a "
        f"separate validation environment draw over the same grid, so no control loses for being "
        f"untuned. The unit of replication is the SEED ({cfg.seeds} independent initialisations AND "
        f"independent environment draws -- fresh prototypes, fresh permutations, fresh stream order), "
        f"never a step or a batch, and every MDE is 3*sd/sqrt(n) on the PAIRED per-seed gap. Before "
        f"any number above was allowed into the table, each arm-run had to pass a mechanism liveness "
        f"assertion proving numerically that its fast adapters moved between updates, that its context "
        f"posterior was neither collapsed onto one context nor flat uniform, that its consolidated "
        f"adapter existed and differed from the fast adapter, that its plasticity controller emitted a "
        f"varying rate, that its workspace state moved across internal steps and did not simply run to "
        f"the halting cap, that eligibility was maintained when and only when feedback was delayed, and "
        f"that the encoder did not move for any arm claiming no backprop through it; separately, every "
        f"component flag was toggled on one pretrained model and required to move the output logits, "
        f"and no two differently-configured arms were permitted to return identical per-seed metrics "
        f"except the one pair the specification requires to be identical.")
    rep("\n" + "=" * 108)
    rep("  VERDICT: " + verdict)

    limits = [
        "FALSIFIER: the environment is synthetic. The latent contexts are label PERMUTATIONS over a "
        "fixed encoder representation, which is the friendliest possible form of context change for a "
        "linear per-context adapter. If contexts instead changed the input statistics or required a "
        "different feature, the fast adapter would have nothing linear to fix and the result should "
        "not be expected to transfer. Nothing here is evidence about biology data.",
        "FALSIFIER: consolidation is only useful because lambda_f is applied to every slot on every "
        "step, so a fast adapter left alone for one block decays by lambda^block. If fast weights were "
        "instead decayed only when their context is active, W_f would never forget and G3 should "
        "collapse to zero. That single line is the load-bearing modelling choice behind the "
        "consolidation claim and it is stated, not hidden.",
        "FALSIFIER: the online contexts are the cyclic permutations, chosen so that a context-blind "
        "model's Bayes rate is exactly chance. If the contexts partially agreed, the frozen and "
        "global-fast controls would rise and the context manager's margin would shrink accordingly.",
        "FALSIFIER: ablation arms are given a WIDER encoder because they do not pay for the machinery "
        "they lack. That biases every ablation gap against the full model, so a passing G2-G5 is "
        "conservative and a failing one may partly reflect the extra capacity.",
        "the offline pretraining supplies the TRUE context slot to the workspace, while online the "
        "workspace receives a posterior-weighted mixture. That is a train/test mismatch in the "
        "digital half and it is not corrected for.",
        "the eligibility comparison is not a like-for-like credit-assignment comparison: the SGD "
        "controls buffer the exact delayed example and take a true gradient step, whereas the local "
        "arms pair the delayed error with a decaying trace. This favours the controls.",
        "one example per step means each seed's prequential accuracy carries a binomial standard "
        "error of about 0.02 at this stream length; the MDE is computed on the PAIRED gap, which "
        "removes most of that, but a single seed's raw value should not be over-read.",
        "compute is matched only in parameters, not in wall-clock: the workspace runs up to "
        f"{cfg.max_steps} internal steps per example and the context manager evaluates every slot's "
        "head on every step.",
        "context creation uses a fixed loss threshold rather than a proper model-selection criterion; "
        "too low and it fragments contexts, too high and it merges them, and the realised number of "
        "allocated contexts is reported per arm so that failure mode is visible.",
        "STAGE 2A ONLY: no dendrites, no episodic memory, no world model, no planner. Any claim about "
        "those is out of scope for this file and is not supported by anything in it.",
    ]

    R = {"model": "adrn2a-v1", "smoke": bool(SMOKE), "config": cfg.asdict(),
         "unit_of_replication": "seed (independent initialisation AND independent environment draw)",
         "primary_metric": "online_acc = prequential held-out accuracy over the online stream (raw)",
         "secondary_metric": "retention_acc = accuracy on fresh examples per context, learning off",
         "mde_rule": "3*sd/sqrt(n) on the PAIRED per-seed gap; a positive gap whose paired sample sd "
                     "is exactly zero has an UNESTIMABLE MDE and is reported UNSUPPORTED, never as a "
                     "pass. Ceiling/floor is adjudicated per METRIC (online_acc AND retention_acc) "
                     "with the declared max(MDE, 0.02) tolerance, and gates read on a floored metric "
                     "are UNSUPPORTED rather than counted.",
         "arms": {a: {"flags": ARMS[a].asdict(), "params": pcounts[a],
                      "results": {c: {k: (v if k == "calibrated" else
                                          {"mean": stats(v)[0], "sd": stats(v)[1], "per_seed": v})
                                      for k, v in results[c][a].items()}
                                  for c, _ in conditions}} for a in ARMS},
         "mde": {k: {"gap": v.get("gap"), "mde": v.get("mde")} for k, v in gapinfo.items()},
         "gaps": gapinfo, "gates": gates, "gates_unsupported": unsupported_gates,
         "ceiling_floor": cf,
         "liveness": _jsonable(liveness), "verdict": verdict, "limits": limits,
         "runtime_sec": time.time() - t0, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "adrn2.json"
    json.dump(R, open(p, "w"), indent=1)
    rep(f"\n  -> {p}    ({time.time() - t0:.1f}s)")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return o


if __name__ == "__main__":
    main()
