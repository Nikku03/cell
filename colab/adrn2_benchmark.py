"""ADRN-2 SPEC SECTION 18 BASELINE LADDER -- eight arms on the latent-rule environment.

Environment: `adrn2_env.Adrn2Env` (section 18, c1->c2->c3->c1->c4->c2, returns at phases 4 and 6).
Model:       `adrn2.ADRN2A` (digital encoder + recurrent workspace + context manager + fast/consolidated
             adapters). This file adds NO mechanism of its own to the model; it only drives it.

=========================================================================================================
THE GATES AND THE MDE, DECLARED HERE BEFORE ANY NUMBER IN THIS FILE EXISTS
=========================================================================================================
UNIT OF REPLICATION: the SEED. One seed = one independent initialisation AND one independent environment
draw (independent input stream, independent pretraining draw, independent validation draw for step-size
calibration). Seeds inside one draw would measure optimiser noise, not the claim. n = ADRN2_SEEDS.

MDE = 3 * sd / sqrt(n) computed on the PAIRED per-seed gap between the two arms being compared, because
every arm sees the byte-identical stream within a seed. Never the pooled sd.

EFFECT SIZE IS THE RAW HELD-OUT VALUE. Every accuracy printed is the raw held-out accuracy of that arm.
The reference rows (oracle, context-blind Bayes, stale-rule) and the naive-probe control establish
SIGNIFICANCE and INTERPRETATION only. Nothing is ever reported net of a control.

PRIMARY METRIC -- RETENTION. Accuracy inside the feedback-WITHHELD window that opens a RETURN phase
(phase 4 on c1, phase 6 on c2). Chance is 0.5.

PREDECLARED GATES (primary cadence F = ADRN2_PRIMARY_F unless stated):
  G1  arm 8 (ADRN-2A inferred context) beats the BEST of arms 1-5 on RETENTION by more than the MDE.
  G2  arm 8 beats arm 6 (ADRN-2A without context separation) on RETENTION by more than the MDE.
      <- this is the gate that makes context separation earn its place; arm 6 is the same code with the
         context manager switched off, so the comparison isolates it and nothing else.
  G3  arm 8's RETENTION is above chance (0.5) by more than a one-sample MDE at ALL FOUR feedback
      cadences F in {1, 4, 16, 64}.
  G4  arm 7 (ORACLE context) minus arm 8 (INFERRED context) is REPORTED whatever its sign or size.
      This gate passes iff the number is computed and printed; it can never be failed by an unfavourable
      value, because a diagnostic you are allowed to suppress is not a diagnostic.
  G5  CEILING AND FLOOR: the task discriminates. It FAILS if every arm sits within max(MDE, 0.02) of
      chance (too hard to separate anything), and it FAILS if two or more arms sit at or above
      ADRN2_CEIL with no arm falling informatively below them. A ceiling kills separation only AMONG the
      arms that reach it: an arm falling more than its MDE BELOW a ceiling the others reach is highly
      informative and is reported as such rather than being swallowed by the word "uninformative".

THE HEADLINE DIAGNOSTIC, NOT A FOOTNOTE. arm 7 - arm 8 is the ORACLE-MINUS-INFERRED gap. If it is large,
CONTEXT DETECTION is the bottleneck and the context manager is what needs work. If it is near zero and
both beat arm 6, context separation is doing real work. It is printed on its own line, in every cadence.

=========================================================================================================
WHAT IS INFORMATION-THEORETICALLY AVAILABLE INSIDE THE RETENTION WINDOW -- STATED BEFORE THE NUMBERS
=========================================================================================================
This determines what any arm CAN score and therefore how the numbers must be read. The environment
guarantees (its gate G9, a structural invariant of the generator: one global BIT_P, no context argument)
that the input distribution is IDENTICAL in every context. Feedback is withheld for the whole window. So
inside the window there is no evidence -- none -- about WHICH context is active. Retrieval by
identification is impossible there for any non-oracle system of any size. Three references bound what is
achievable and all three are computed on the realised windows and printed alongside the arms:

    stale-rule            keep applying the immediately preceding phase's rule. The floor: what a system
                          with no change detection and no context memory gets.  (measured ~0.50)
    context-blind Bayes   detect that a change happened and average over the rules of the contexts SEEN
                          SO FAR. The CEILING for any non-oracle arm in this window. (measured ~0.67)
    oracle                the context handed over. 1.0 by construction.

Consequently arm 8 cannot exceed context-blind Bayes, and if it merely reaches it the honest reading is
"it HEDGED, it did not RETRIEVE". That reading is written into the verdict rather than left for a reader
to discover, and G1/G2 remain meaningful because arms 1-5 and arm 6 have no per-context store to hedge
over at all.

CHANGE DETECTION IS LEGITIMATELY OBSERVABLE and is given to EVERY arm identically. `feedback_available`
is part of the environment's 6-tuple contract, and because withholding happens only inside probe windows,
a cadence slot that delivers no label is a visible signal that a probe (hence a phase boundary) has
begun. `ProbeDetector` derives that flag from the feedback channel alone -- it never reads `y` -- and
every arm receives it and uses it in the strongest form available to its own architecture:
    arms 1,2,3   fall back to their frozen pretrained CONTEXT-BLIND head
    arm 4        falls back to a reservoir over all history instead of the recency FIFO
    arm 5        falls back to its Polyak-averaged weights (the context-blind average)
    arm 6        falls back to the stable decoder alone (no adapter)
    arms 7,8     replace the sticky transition prior with a uniform prior over ALLOCATED context slots
Giving this only to the proposal would be ADRN-1's defect 2 (a handicapped baseline manufacturing a win),
so it is given to all eight and its liveness is asserted.

=========================================================================================================
THE FOUR DEFECTS ADRN-1 SHIPPED AND WHAT IN THIS FILE MAKES EACH IMPOSSIBLE
=========================================================================================================
1 THE NETWORK WAS SILENT (theta=1.0 against a soma at 0.05 -> 0.00 spikes/example; every downstream
  mechanism a no-op while scores were still returned).
2 THE BASELINE WAS HANDICAPPED (the point-neuron control was silent for the same reason, so repairing
  only the proposal would have manufactured the win).
3 THE ABLATION WAS VACUOUS (prune_every defaulted to 0, so structural=True/False changed nothing).
4 THE LEARNING SYSTEM NEVER RAN (net(x) called with no modulatory vector, so eligibility,
  neuromodulation, fast/slow weights, consolidation, homeostasis and pruning were ALL inert; the tell,
  visible twice before it was read correctly, was three different ablations returning BYTE-IDENTICAL
  accuracy, energy and spike counts).

Against 1 and 4 -- `assert_liveness` runs on EVERY arm-run BEFORE its score is allowed into the table and
raises `MechanismInert` unless, numerically:
    * the fast adapters CHANGED between updates (median relative ||dW_f|| > 0 and final ||W_f|| > 0);
    * the context posterior is NON-DEGENERATE (time-averaged entropy above a collapse floor, mean
      per-step entropy below log k_used, and marginal-minus-conditional information above zero);
    * the CONSOLIDATED adapter exists, fired, and DIFFERS from the fast adapter by more than a copy;
    * the workspace state moves across internal steps and halting does not simply run to the cap;
    * every baseline arm actually updated (n_updates > 0) and its adaptive state actually moved;
    * the encoder did NOT move for every arm claiming a frozen encoder, and DID move for arm 1;
    * the probe detector fired and the change-detection fallback changed at least one prediction.
Against 3 -- `flag_effect_probe` toggles context / fast / consolidate / workspace on ONE pretrained model,
replays the SAME full stream, and raises unless the prediction trace or the logits measurably move.
Against 4's signature -- `assert_arms_distinct` refuses to report if any two differently-configured arms
returned byte-identical per-seed retention AND overall accuracy.
Against 2 -- every arm is built to the same bisected parameter budget with realised counts printed; the
digital half is pretrained by an IDENTICAL context-blind recipe for every neural arm and must pass
`assert_digital_competence` (a per-context linear probe) or the run raises rather than reporting a null
that could equally mean "the encoder never trained"; and every free scalar step size in arms 1-5 is
calibrated per seed and per cadence on a SEPARATE validation environment draw over the same grid, while
the ADRN-2A arms get NO calibrated scalar at all (their step size comes from the meta-trained plasticity
controller). A control is never allowed to lose because nobody tuned it.

PREDECLARED RULE FOR A MECHANISM THAT CANNOT ENGAGE AT A GIVEN CADENCE. The context manager needs a
minimum number of FEEDBACK EVENTS to allocate a slot, and consolidation needs a minimum number of
observations on a slot. At F=64 the stream delivers 64x fewer labels than at F=1, so a mechanism may be
genuinely starved rather than broken. Declared before any number: an inert mechanism RAISES if it is
inert at the PRIMARY cadence, and RAISES if it is inert at every cadence; at a non-primary cadence it is
recorded as `mechanism_inert_at_F` and that cadence's arm-8 gate readings are declared UNSUPPORTED and
excluded from the verdict's claims rather than being reported as a result.

"It should work" is not evidence anywhere in this file.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

torch.set_num_threads(int(os.environ.get("ADRN2_THREADS", "1")))

from adrn2_env import (BIT_P, CONTEXTS, LAG, N_BITS, PHASE_SCHEDULE, SUPPORTED_FEEDBACK_EVERY,
                       Adrn2Env, LivenessError, apply_rule)
from adrn2_env import liveness_check as env_liveness_check
import adrn2 as A2
from adrn2 import ADRN2A, D_CTX, D_MOD, Flags, MechanismInert, _aug

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SMOKE = os.environ.get("ADRN2_SMOKE", "0") == "1"


# =========================================================================================== config
class BCfg:
    """Every knob from an env var, with a SMOKE mode that shrinks the run and labels the output."""

    def __init__(self):
        g = lambda k, d, f=int: f(os.environ.get(k, d))
        self.seeds = g("ADRN2_SEEDS", 6)
        self.phase_len = g("ADRN2_PHASE_LEN", 1024)
        self.probe_len = g("ADRN2_W", 64)
        self.relearn_feedbacks = g("ADRN2_RELEARN_FEEDBACKS", 4)
        self.budget = g("ADRN2_BUDGET", 30000)
        self.hist = g("ADRN2_HIST", 3)          # input window; >= LAG+1 so c4's t-2 term is reachable
        self.k_slots = g("ADRN2_KSLOTS", 6)     # context slots available to the manager (4 true contexts)
        self.max_steps = g("ADRN2_MAXSTEPS", 3)
        self.pre_n = g("ADRN2_PRE_N", 16384)
        self.pre_epochs = g("ADRN2_PRE_EPOCHS", 10)
        self.pre_bs = g("ADRN2_PRE_BS", 64)
        self.pre_restarts = g("ADRN2_RESTARTS", 2)
        self.meta_iters = g("ADRN2_META", 40)
        self.val_phases = g("ADRN2_VALPHASES", 2)   # phases of the validation draw used for calibration
        self.competence_min = g("ADRN2_COMPMIN", 0.85, float)
        self.ceiling = g("ADRN2_CEIL", 0.99, float)
        self.nn_k = g("ADRN2_NNK", 5)
        self.primary_F = g("ADRN2_PRIMARY_F", 4)
        self.cadences = tuple(int(v) for v in
                              os.environ.get("ADRN2_CADENCES", "1,4,16,64").split(","))
        if SMOKE:
            self.seeds = 2
            self.phase_len, self.probe_len, self.relearn_feedbacks = 320, 32, 3
            self.pre_n, self.pre_epochs, self.meta_iters = 6144, 8, 12
        self.chance = 0.5
        for f in self.cadences:
            if f not in SUPPORTED_FEEDBACK_EVERY:
                raise ValueError(f"cadence {f} is not one of {SUPPORTED_FEEDBACK_EVERY}")
        if self.hist < LAG + 1:
            raise ValueError(f"hist={self.hist} cannot reach the t-{LAG} term c4 depends on")

    def asdict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


# ======================================================================== offline data (i.i.d. windows)
def make_windows(n, rng, hist):
    """i.i.d. input windows drawn from the SAME global BIT_P the stream uses, with per-context labels.

    Rows are independent across time exactly as they are in the stream, so a window drawn here has the
    same joint law as a window taken from the stream. Returned labels: one per context (the rule that
    context would apply to this window) plus `y_mix`, the label under a uniformly random context.
    """
    X = (rng.random((n, hist, N_BITS)) < BIT_P[None, None, :]).astype(np.int8)
    cur, lag = X[:, -1], X[:, -1 - LAG]
    ys = np.stack([apply_rule(c, cur, lag) for c in CONTEXTS])          # (4, n)
    pick = rng.integers(0, len(CONTEXTS), n)
    y_mix = ys[pick, np.arange(n)]
    return (torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(ys.astype(np.int64)),
            torch.from_numpy(y_mix.astype(np.int64)))


def blind_bayes_accuracy_over(contexts, x_cur, x_lag, y):
    """Accuracy of the Bayes rule that knows only that the context is uniform over `contexts`.

    Ties (exactly half the rules firing) are broken to 0; at a tie the accuracy is 0.5 either way.
    """
    r = np.stack([apply_rule(c, x_cur, x_lag) for c in contexts]).mean(0)
    return float(((r > 0.5).astype(np.int64) == y).mean())


# ================================================================================= encoders + heads
class TfmEncoder(nn.Module):
    """Small transformer over the `hist`-step input window; the last token is the representation."""

    def __init__(self, d, hist, heads=2):
        super().__init__()
        self.inp = nn.Linear(N_BITS, d)
        self.pos = nn.Parameter(torch.zeros(hist, d))
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, batch_first=True,
                                           dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, 1)
        self.d = d

    def forward(self, seq):
        return self.tr(self.inp(seq) + self.pos)[:, -1]


class GruEncoder(nn.Module):
    def __init__(self, d, hist):
        super().__init__()
        self.rnn = nn.GRU(N_BITS, d, batch_first=True)
        self.d = d

    def forward(self, seq):
        return self.rnn(seq)[0][:, -1]


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def bisect_width(build, budget, lo=4, hi=256, step=1):
    """Bisect a hidden width so an architecture lands on the shared parameter budget. Realised counts
    are printed by the caller; a baseline must never lose for being smaller than the proposal."""
    best = lo
    while lo <= hi:
        mid = ((lo + hi) // 2 // step) * step
        mid = max(step, mid)
        try:
            n = build(mid)
        except Exception:
            hi = mid - step
            continue
        if n <= budget:
            best, lo = mid, mid + step
        else:
            hi = mid - step
    return best


# ================================================================================ offline pretraining
def pretrain_plain(enc, hist, cfg: BCfg, rng, seed):
    """Train `enc` so that (a) a linear head reproduces the context-blind mixture and (b) EVERY rule is
    linearly decodable from the representation.

    The four per-context heads are a training signal only and are DISCARDED; what survives online is the
    encoder and the context-blind head. The recipe is byte-identical for the transformer, the GRU and
    ADRN-2A, so no arm can lose because its digital half was trained differently -- ADRN-2's own record
    shows how easily that happens (its context embedding table had been sized by the number of adapter
    slots, which crippled every context-blind control at 0.36 against 0.87 and would have manufactured
    the entire result).
    """
    torch.manual_seed(seed)
    blind = nn.Linear(enc.d, 2)
    heads = nn.ModuleList([nn.Linear(enc.d, 2) for _ in CONTEXTS])
    params = list(enc.parameters()) + list(blind.parameters()) + list(heads.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    X, Y, Ym = make_windows(cfg.pre_n, rng, hist)
    for _ in range(cfg.pre_epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), cfg.pre_bs):
            b = perm[i:i + cfg.pre_bs]
            h = enc(X[b])
            loss = F.cross_entropy(blind(h), Ym[b])
            for j in range(len(CONTEXTS)):
                loss = loss + F.cross_entropy(heads[j](h), Y[j][b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    return blind, heads


def adrn_expected_feature(model: ADRN2A, seq):
    """ACT-style expected workspace feature and ponder cost, used OFFLINE only.

    Runs the workspace for the full step budget and weights each step by the halting head's probability,
    so the halting decision the online hard rule makes is a TRAINED decision rather than a coin flip. An
    untrained halting head would make "adaptive halting" a decoration, which is precisely the class of
    claim this project has been burned by.
    """
    b = seq.shape[0]
    h = model.encode(seq)
    m = model.mod_vector(False, b)
    c_in = model.ctx_emb.weight.mean(0, keepdim=True).expand(b, D_CTX)
    w = torch.tanh(model.w0(h))
    still = torch.ones(b, 1)
    feat, expect = 0.0, 0.0
    for t in range(model.cfg.max_steps):
        u = torch.cat([w, h, m, c_in], -1)
        z = torch.sigmoid(model.fz(u))
        w = (1 - z) * w + z * torch.tanh(model.fn(u))
        p = torch.sigmoid(model.halt(w))
        wt = still * p if t < model.cfg.max_steps - 1 else still
        feat = feat + wt * _aug(w)
        expect = expect + wt * (t + 1)
        still = still * (1 - p)
    return feat, expect.mean()


def pretrain_adrn(model: ADRN2A, hist, cfg: BCfg, rng, seed):
    """The same recipe as `pretrain_plain`, on the ADRN-2A workspace state w_aug -- which is exactly the
    vector its per-context adapters act on, so the adapters have something linear to fix."""
    torch.manual_seed(seed)
    heads = nn.ModuleList([nn.Linear(model.d_w + 1, 2) for _ in CONTEXTS])
    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    X, Y, Ym = make_windows(cfg.pre_n, rng, hist)
    for _ in range(cfg.pre_epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), cfg.pre_bs):
            b = perm[i:i + cfg.pre_bs]
            feat, ponder = adrn_expected_feature(model, X[b])
            loss = F.cross_entropy(model.dec(feat), Ym[b]) + 0.01 * ponder
            for j in range(len(CONTEXTS)):
                loss = loss + F.cross_entropy(heads[j](feat), Y[j][b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    return heads


def meta_train_controller(model: ADRN2A, cfg: BCfg, rng, seed):
    """Train the plasticity controller by unrolling the LOCAL delta rule and minimising post-update loss.

    Only (eta, lambda) carry gradient; the workspace state is detached, so nothing here backprops into
    the encoder. Without this the controller is a random network and "learned plasticity" would be a
    claim about noise. This is the ONLY free step size the ADRN-2A arms have, and unlike the controls'
    calibrated scalars it is never tuned on validation accuracy.
    """
    if model.controller is None:
        return {"meta": "no controller"}
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.controller.parameters(), lr=5e-3)
    hist_loss = []
    for _ in range(cfg.meta_iters):
        c = int(rng.integers(0, len(CONTEXTS)))
        X, Y, _ = make_windows(10, rng, cfg.hist)
        y = Y[c]
        W = torch.zeros(model.C, model.d_w + 1)
        total = 0.0
        for t in range(len(X)):
            with torch.no_grad():
                h = model.encode(X[t:t + 1])
                w, _ = model.workspace(h, model.mod_vector(False), model.ctx_input(model.p),
                                       hard_halt=False)
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
        hist_loss.append(float(total.detach()) / len(X))
    return {"meta_loss_first": hist_loss[0], "meta_loss_last": hist_loss[-1]}


@torch.no_grad()
def assert_digital_competence(tag, feat_fn, heads, blind, hist, cfg: BCfg, rng):
    """RAISE unless the digital half is trained. Direct analogue of ADRN-1's defect 1.

    (a) EVERY rule must be linearly decodable from the representation. If it is not, the online adapters
        and heads have nothing to select between and any null they produce would be a statement about a
        broken encoder rather than about context separation -- the unreadable null this project keeps
        shipping.
    (b) The context-blind head must land near the analytic context-blind Bayes rate. A blind head far
        below it means the shared readout never trained either.
    """
    X, Y, Ym = make_windows(2048, rng, hist)
    f = feat_fn(X)
    per = {}
    for j, c in enumerate(CONTEXTS):
        per[c] = float((heads[j](f).argmax(-1) == Y[j]).float().mean())
    blind_acc = float((blind(f).argmax(-1) == Ym).float().mean())
    cur, lag = X[:, -1].numpy().astype(np.int8), X[:, -1 - LAG].numpy().astype(np.int8)
    blind_bayes = blind_bayes_accuracy_over(CONTEXTS, cur, lag, Ym.numpy())
    bad = []
    for c, v in per.items():
        if v < cfg.competence_min:
            bad.append(f"rule {c} is NOT linearly decodable from the pretrained representation "
                       f"({v:.4f} < {cfg.competence_min}); every adapter and head downstream would be "
                       f"selecting between features that do not exist")
    if blind_acc < blind_bayes - 0.08:
        bad.append(f"the context-blind head reaches {blind_acc:.4f} against an analytic context-blind "
                   f"Bayes rate of {blind_bayes:.4f}; the shared readout never trained")
    if bad:
        raise MechanismInert(f"[{tag}] DIGITAL COMPETENCE FAILED before any score was reported:\n  - "
                             + "\n  - ".join(bad))
    return {"per_context_probe": per, "blind_head_acc": blind_acc,
            "blind_bayes_analytic": blind_bayes}


# ================================================================================== probe detection
class ProbeDetector:
    """Infers "a feedback-withheld window has begun" from the FEEDBACK CHANNEL ALONE.

    It never reads `y`. It observes only whether a label arrived at each cadence slot, which is part of
    the environment's 6-tuple contract. Because withholding happens only inside probe windows, a cadence
    slot that delivers nothing is evidence a phase boundary has just passed. The detection LAG is up to F
    steps, so at F=64 with W=64 the flag can arrive after the window has ended -- that is a real property
    of the protocol at that cadence and it is measured (`probe_flag_recall`) rather than assumed away.
    """

    def __init__(self, F):
        self.F = F
        self.flag = False
        self.n_flag = 0

    def observe(self, t, got_feedback):
        if (t + 1) % self.F == 0:
            self.flag = not got_feedback
            if self.flag:
                self.n_flag += 1


# =========================================================================================== arms
class BaseArm:
    """Contract: predict(ob) -> 0/1, learn(ob, y) called ONLY on feedback steps.

    `ob` carries the input window, the derived probe flag and (for the oracle arm alone) the true
    context. `learn` never sees a label on a withheld step because the harness never calls it there.
    """

    uses_oracle = False
    frozen_encoder = True

    def __init__(self, name):
        self.name = name
        self.n_updates = 0

    def predict(self, ob):
        raise NotImplementedError

    def learn(self, ob, y):
        raise NotImplementedError

    def state_sig(self):
        """Scalar fingerprint of the ADAPTIVE state. Liveness requires it to move over a stream."""
        return 0.0

    def enc_sig(self):
        """Scalar fingerprint of the ENCODER. Liveness requires it to be frozen unless the arm says
        otherwise."""
        return 0.0

    def tele(self):
        return {}


class TorchHeadArm(BaseArm):
    """Arms 1, 2, 3. A pretrained encoder plus an online linear head; `full` also fine-tunes the encoder.

    The online head is INITIALISED FROM the pretrained context-blind head rather than from noise, so the
    arm starts at the best context-blind predictor instead of at chance. On a detected change it falls
    back to that frozen blind head, which is the strongest context-blind policy available to it.
    """

    def __init__(self, name, enc, blind, lr, full):
        super().__init__(name)
        self.enc, self.blind = enc, copy.deepcopy(blind)
        self.head = copy.deepcopy(blind)
        self.full = full
        self.frozen_encoder = not full
        for p in self.blind.parameters():
            p.requires_grad_(False)
        for p in self.enc.parameters():
            p.requires_grad_(full)
        ps = list(self.head.parameters()) + (list(self.enc.parameters()) if full else [])
        self.ps = ps
        self.opt = torch.optim.Adam(ps, lr=lr)
        self.n_fallback = 0

    def predict(self, ob):
        with torch.no_grad():
            h = self.enc(ob["seq"])
            if ob["probe"]:
                self.n_fallback += 1
                return int(self.blind(h).argmax(-1))
            return int(self.head(h).argmax(-1))

    def learn(self, ob, y):
        if self.full:
            h = self.enc(ob["seq"])
        else:
            with torch.no_grad():
                h = self.enc(ob["seq"])
        loss = F.cross_entropy(self.head(h), torch.tensor([y]))
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.ps, 1.0)
        self.opt.step()
        self.n_updates += 1

    def state_sig(self):
        return float(sum(p.detach().abs().sum() for p in self.head.parameters()))

    def enc_sig(self):
        return float(sum(p.detach().abs().sum() for p in self.enc.parameters()))

    def tele(self):
        return {"n_fallback": self.n_fallback}


class LookupNNArm(BaseArm):
    """Arm 4. Nearest-neighbour / lookup over the raw input window, budget-matched in memory cells.

    Two stores, both fed by every feedback event: a RECENCY FIFO (tracks the current context) and a
    RESERVOIR over all history (context-blind). The recency store is used normally and the reservoir on
    a detected change, which is this arm's form of the same fallback every other arm gets.
    """

    def __init__(self, name, mem, k, dim):
        super().__init__(name)
        self.mem, self.k, self.dim = mem, k, dim
        self.F_x = np.zeros((mem, dim), np.float32)
        self.F_y = np.zeros(mem, np.int64)
        self.f_n = self.f_ptr = 0
        self.R_x = np.zeros((mem, dim), np.float32)
        self.R_y = np.zeros(mem, np.int64)
        self.r_n = 0
        self.seen = 0
        self.rng = np.random.default_rng(12345)
        self.n_fallback = 0

    def _vote(self, X, Y, n, q):
        if n == 0:
            return 0
        d = np.abs(X[:n] - q[None, :]).sum(1)
        k = min(self.k, n)
        idx = np.argpartition(d, k - 1)[:k]
        return int(Y[idx].mean() > 0.5)

    def predict(self, ob):
        q = ob["flat"]
        if ob["probe"]:
            self.n_fallback += 1
            return self._vote(self.R_x, self.R_y, self.r_n, q)
        return self._vote(self.F_x, self.F_y, self.f_n, q)

    def learn(self, ob, y):
        q = ob["flat"]
        self.F_x[self.f_ptr] = q
        self.F_y[self.f_ptr] = y
        self.f_ptr = (self.f_ptr + 1) % self.mem
        self.f_n = min(self.f_n + 1, self.mem)
        if self.r_n < self.mem:
            self.R_x[self.r_n] = q
            self.R_y[self.r_n] = y
            self.r_n += 1
        else:
            j = int(self.rng.integers(0, self.seen + 1))
            if j < self.mem:
                self.R_x[j] = q
                self.R_y[j] = y
        self.seen += 1
        self.n_updates += 1

    def state_sig(self):
        return float(self.F_x[:self.f_n].sum() + self.F_y[:self.f_n].sum() + self.f_n)

    def tele(self):
        return {"n_fallback": self.n_fallback, "fifo_filled": int(self.f_n),
                "reservoir_filled": int(self.r_n)}


class LogRegArm(BaseArm):
    """Arm 5. Online logistic regression on the raw binary window plus a bias.

    On a detected change it falls back to its Polyak-averaged weights, which are the arm's own
    context-blind average over everything it has seen. NOTE, stated rather than hidden: a linear model on
    raw bits cannot represent XOR, so its ceiling on c1 is 0.5 by construction. That is a property of the
    baseline the specification names, not a handicap this harness imposes -- arms 2 and 3 are the
    capacity-matched linear-head controls on a LEARNED representation and they are the ones that make G1
    a real gate.
    """

    def __init__(self, name, dim, lr):
        super().__init__(name)
        self.w = np.zeros(dim + 1, np.float64)
        self.wbar = np.zeros(dim + 1, np.float64)
        self.lr = lr
        self.n_fallback = 0

    def _z(self, w, q):
        return float(w[:-1] @ q + w[-1])

    def predict(self, ob):
        q = ob["flat"].astype(np.float64)
        if ob["probe"]:
            self.n_fallback += 1
            return int(self._z(self.wbar, q) > 0)
        return int(self._z(self.w, q) > 0)

    def learn(self, ob, y):
        q = ob["flat"].astype(np.float64)
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, self._z(self.w, q)))))
        g = (y - p)
        self.w[:-1] += self.lr * g * q
        self.w[-1] += self.lr * g
        self.n_updates += 1
        self.wbar += (self.w - self.wbar) / self.n_updates
        self.n_fb_used = self.n_updates

    def state_sig(self):
        return float(np.abs(self.w).sum())

    def tele(self):
        return {"n_fallback": self.n_fallback, "w_norm": float(np.linalg.norm(self.w))}


class AdrnArm(BaseArm):
    """Arms 6, 7, 8. Drives `adrn2.ADRN2A` -- no mechanism is reimplemented here.

    arm 6  flags.context = False        one global adapter slot; on a detected change it falls back to
                                        the stable decoder alone, which is its context-blind estimate.
    arm 7  flags.oracle_context = True  the true context is handed over as a one-hot prior.
    arm 8  flags default                the posterior is inferred; on a detected change the sticky
                                        transition prior is replaced by a uniform prior over ALLOCATED
                                        slots, i.e. the system hedges over the contexts it has stored.

    Eligibility is switched OFF for all three and no claim is made about it: in this environment feedback
    is SPARSE, not delayed -- the label at a feedback step refers to THAT step -- so pairing the error
    with a decaying trace of earlier steps would credit the wrong example. Its liveness is asserted in
    the negative (the trace must never be updated), which is the same discipline ADRN-2A applies to its
    immediate-feedback condition.
    """

    uses_oracle = False

    def __init__(self, name, model: ADRN2A, oracle: bool):
        super().__init__(name)
        self.model = model
        self.oracle = oracle
        self.uses_oracle = oracle
        self.model.reset_online()
        self.tel = A2.new_tel()
        self.step = 0
        self.n_fallback = 0
        self.post_log = []
        self._cache = None

    def predict(self, ob):
        m = self.model
        with torch.no_grad():
            h = m.encode(ob["seq"])
            hedge = bool(ob["probe"]) and m.k > 1
            if self.oracle and m.k > 1:
                j = int(ob["ctx"]) % m.k
                m.used[j] = True
                prior = m.one_hot_slot(j)
            elif hedge:
                prior = m.used.float() / max(1.0, float(m.used.sum()))
            else:
                prior = m.transition_prior()
            c_in = m.ctx_input(prior)
            mod = m.mod_vector(False)
            rec = []
            w, _ = m.workspace(h, mod, c_in, hard_halt=True, record=rec)
            wa = _aug(w)[0]
            if self.oracle or hedge:
                pre = prior            # a hedge must not be undone by the workspace-key term, which
                                       # carries no context information here (env gate G9)
            else:
                pre = m.posterior_pre(prior, w[0], self.step)
            if bool(ob["probe"]) and m.k == 1:
                # arm 6's fallback: the stable decoder alone, with no adapter contribution.
                self.n_fallback += 1
                logits = m.dec(wa)
            else:
                if hedge:
                    self.n_fallback += 1
                logits = m.logits_mix(wa, pre)
            pred = int(logits.argmax())
            self.tel["steps"].append(rec[0])
            if len(rec) > 1:
                self.tel["wstd"].append(rec[1])
            self.tel["mod"].append(mod[0].numpy().copy())
            self.post_log.append(pre.numpy().copy())
            m.err_ema = 0.98 * m.err_ema + 0.02 * float(
                F.cross_entropy(logits.unsqueeze(0), torch.tensor([pred])))
        self._cache = (w, wa, pre)
        self.step = ob["t"]
        return pred

    def learn(self, ob, y):
        m = self.model
        w, wa, pre = self._cache
        yt = torch.tensor([y])
        with torch.no_grad():
            ok = int(int(m.logits_mix(wa, pre).argmax()) == y)
            m.acc_ema = 0.98 * m.acc_ema + 0.02 * ok
            post, _ = m.posterior_post(pre, wa, yt, self.step)
            if self.oracle and m.k > 1:
                post = pre
            eta, lam, g = m.rates(w[0], ok, A2._ent_t(post), 0.15)
            self.tel["eta"].append(eta)
            self.tel["lam"].append(lam)
            self.tel["g"].append(g)
            m.local_update(wa, yt, post, eta, lam, g, self.tel)
            m.bookkeeping(post, w[0], ok)
            if not self.oracle:
                m.maybe_create(self.step, w[0], self.tel)
        self.n_updates += 1

    def state_sig(self):
        return float(self.model.W_f.abs().sum() + self.model.W_m.abs().sum())

    def enc_sig(self):
        return float(sum(p.detach().abs().sum() for p in self.model.enc.parameters()))

    def tele(self):
        m = self.model
        return {"n_fallback": self.n_fallback, "contexts_used": int(m.used.sum()),
                "contexts_created": len(self.tel["created_at"]),
                "created_at": list(self.tel["created_at"]),
                "n_consolidations": int(self.tel["n_consol"]),
                "W_f_norm": float(m.W_f.norm()) if m.W_f.numel() else 0.0,
                "W_m_norm": float(m.W_m.norm()) if m.W_m.numel() else 0.0}


# =========================================================================================== driver
def run_arm(arm: BaseArm, env: Adrn2Env, Wseq, seq_t, F):
    """One pass of the stream. predict -> score -> (only if the label is delivered) learn.

    Every step is scored on a NEVER-TRAINED-ON example, so every accuracy is a raw held-out value. The
    label is taken from `y_feedback`, which the environment hard-nulls inside withheld windows; `y` is
    read for SCORING only and never reaches the arm.
    """
    n = env.total_len
    det = ProbeDetector(F)
    correct = np.zeros(n, np.int8)
    probe_hat = np.zeros(n, np.int8)
    t_pred = t_learn = 0.0
    for st in env.stream():
        t = st.t
        if not st.feedback_available and st.y_feedback is not None:
            raise LivenessError(f"FEEDBACK LEAK at t={t}: the environment offered a label on a withheld "
                                f"step; the harness would have trained on the measurement window")
        ob = {"seq": seq_t[t:t + 1], "flat": Wseq[t], "t": t, "probe": det.flag,
              "ctx": (CONTEXTS.index(st.context_id) if arm.uses_oracle else None)}
        probe_hat[t] = int(det.flag)
        c0 = time.perf_counter()
        p = arm.predict(ob)
        t_pred += time.perf_counter() - c0
        correct[t] = int(p == int(st.y))
        if st.feedback_available:
            c0 = time.perf_counter()
            arm.learn(ob, int(st.y_feedback))
            t_learn += time.perf_counter() - c0
        det.observe(t, st.feedback_available)
    return correct, probe_hat, t_pred, t_learn


def segment_metrics(env: Adrn2Env, correct):
    """RETENTION / RELEARNING / STEADY / naive control / FORGETTING, reported separately and never
    averaged together. Retention is THE metric; relearning is a different quantity entirely and
    conflating them would let a fast relearner pass as a good retainer."""
    seg = env.segment.astype(str)
    out = {}
    for k, s in (("retention", "retention_probe"), ("naive", "naive_probe"),
                 ("relearn", "relearn"), ("steady", "steady")):
        m = seg == s
        out[k] = float(correct[m].mean()) if m.any() else float("nan")
    out["overall"] = float(correct.mean())
    out["per_context"] = {c: float(correct[env.context_id == c].mean()) for c in CONTEXTS}

    # FORGETTING: the drop on a context between its first and its second visit. Measured twice, because
    # the two questions are different: at the RETURN (before any new feedback) and after RELEARNING.
    steady_by_phase, ret_by_phase = {}, {}
    for p in range(1, env.n_phases + 1):
        mp = (env.phase_id == p)
        steady_by_phase[p] = float(correct[mp & (seg == "steady")].mean())
        mr = mp & (seg == "retention_probe")
        if mr.any():
            ret_by_phase[p] = float(correct[mr].mean())
    fg_probe, fg_steady, detail = [], [], {}
    for p in range(1, env.n_phases + 1):
        if not env.is_return[p - 1]:
            continue
        c = env.schedule[p - 1]
        first = next(i + 1 for i, cc in enumerate(env.schedule) if cc == c)
        fp = steady_by_phase[first] - ret_by_phase[p]
        fs = steady_by_phase[first] - steady_by_phase[p]
        fg_probe.append(fp)
        fg_steady.append(fs)
        detail[f"{c}:phase{first}->phase{p}"] = {"first_steady": steady_by_phase[first],
                                                 "return_retention": ret_by_phase[p],
                                                 "return_steady": steady_by_phase[p],
                                                 "forgetting_at_return": fp,
                                                 "forgetting_after_relearn": fs}
    out["forgetting"] = float(np.mean(fg_probe)) if fg_probe else float("nan")
    out["forgetting_after_relearn"] = float(np.mean(fg_steady)) if fg_steady else float("nan")
    out["forgetting_detail"] = detail
    return out


def window_references(env: Adrn2Env):
    """Non-arm REFERENCES on the retention windows. Analysis only; no arm is ever reported net of them.

    oracle              the context handed over -- 1.0 by construction.
    blind_bayes_seen    detect the change, then average over the rules of the contexts SEEN SO FAR. This
                        is the CEILING for any non-oracle arm inside a withheld window, because the
                        environment guarantees the input distribution is identical across contexts, so
                        no evidence about WHICH context is active exists there.
    stale_rule          keep applying the previous phase's rule. The FLOOR: no change detection, no
                        context memory.
    """
    out = {"per_window": {}}
    b, s = [], []
    for w in env.retention_windows():
        sl = slice(w.start, w.stop)
        xc, xl, y = env.x[sl], env.x_lag[sl], env.y[sl]
        seen = list(dict.fromkeys(list(env.schedule[:w.phase_id])))
        bb = blind_bayes_accuracy_over(seen, xc, xl, y)
        prev = env.schedule[w.phase_id - 2]
        stale = float((apply_rule(prev, xc, xl) == y).mean())
        out["per_window"][f"phase{w.phase_id}:{w.context_id}"] = {
            "contexts_seen": seen, "blind_bayes_seen": bb, "stale_rule": stale,
            "stale_context": prev, "n": int(w.stop - w.start)}
        b.append(bb)
        s.append(stale)
    out["blind_bayes_seen"] = float(np.mean(b))
    out["stale_rule"] = float(np.mean(s))
    out["oracle"] = 1.0
    nb = []
    for w in env.naive_windows():
        sl = slice(w.start, w.stop)
        seen = list(dict.fromkeys(list(env.schedule[:w.phase_id])))
        nb.append(blind_bayes_accuracy_over(seen, env.x[sl], env.x_lag[sl], env.y[sl]))
    out["naive_blind_bayes_seen"] = float(np.mean(nb))
    return out


# ====================================================================== MECHANISM LIVENESS ASSERTIONS
def assert_liveness(tag, arm: BaseArm, correct, probe_hat, sigs, F, cfg: BCfg, primary):
    """RAISE unless every mechanism this arm CLAIMS is numerically alive. No claim, no check -- but every
    claim is checked, with a number, and the numbers are returned for the record.

    Returns (report, starved). `starved` names mechanisms that could not engage because the cadence
    delivered too few labels; per the rule declared in the module docstring that raises at the primary
    cadence and is otherwise recorded and the condition's gates declared UNSUPPORTED.
    """
    bad, starved, rep = [], [], {}
    sig0_state, sig1_state, sig0_enc, sig1_enc = sigs
    rep["n_updates"] = int(arm.n_updates)
    rep["updates_per_example"] = arm.n_updates / max(1, len(correct))
    rep["state_moved"] = abs(sig1_state - sig0_state)
    rep["encoder_moved"] = abs(sig1_enc - sig0_enc)
    rep["probe_flag_steps"] = int(probe_hat.sum())

    # -- every arm must actually have learned something online -----------------------------------
    if arm.n_updates == 0:
        bad.append("the arm received ZERO online updates: its 'online learning' never ran")
    if rep["state_moved"] <= 0.0:
        bad.append(f"the arm's adaptive state did not move over the whole stream "
                   f"(|d state| = {rep['state_moved']:.3g}); it is a frozen model wearing an online "
                   f"learner's name")

    # -- the encoder must be frozen where the arm claims it is, and must move where it claims to move --
    if arm.frozen_encoder and rep["encoder_moved"] > 1e-9:
        bad.append(f"the encoder MOVED ({rep['encoder_moved']:.3g}) for an arm that claims a frozen "
                   f"encoder")
    if not arm.frozen_encoder and rep["encoder_moved"] <= 1e-9:
        bad.append("this arm claims FULL-model online updates but its encoder did not move")

    # -- the change-detection fallback must be live and must change predictions -------------------
    t = arm.tele()
    rep.update({k: v for k, v in t.items()})
    if probe_hat.sum() == 0:
        bad.append(f"the probe detector never fired at F={F}: the change-detection fallback that every "
                   f"arm is given was switched off, so no arm exercised it")
    if t.get("n_fallback", 0) == 0 and probe_hat.sum() > 0:
        bad.append("the probe detector fired but this arm never took its fallback path")

    if isinstance(arm, AdrnArm):
        m, tel = arm.model, arm.tel
        f = m.flags

        # -- 1. FAST ADAPTERS MUST ACTUALLY CHANGE BETWEEN UPDATES --------------------------------
        if f.fast:
            rel = np.asarray(tel["dWf_rel"], np.float64)
            rep["fast_updates"] = int(rel.size)
            rep["fast_dW_rel_median"] = float(np.median(rel)) if rel.size else 0.0
            rep["fast_norm_final"] = float(m.W_f.norm())
            if rel.size == 0:
                bad.append("fast adapter: ZERO updates were applied over the whole stream")
            elif rep["fast_dW_rel_median"] < 1e-6:
                bad.append(f"fast adapter: median relative change per update is "
                           f"{rep['fast_dW_rel_median']:.3g} -- the adapter is not moving")
            if rep["fast_norm_final"] <= 1e-9:
                bad.append("fast adapter: final ||W_f|| is zero -- nothing was ever written")

        # -- 2. THE CONTEXT POSTERIOR MUST BE NON-DEGENERATE --------------------------------------
        if f.context and m.k > 1:
            P = np.asarray(arm.post_log, np.float64)
            ku = max(2, int(m.used.sum()))
            h_marg = A2._ent(P.mean(0))
            h_cond = float(np.mean([A2._ent(p) for p in P]))
            rep.update(posterior_marginal_entropy=h_marg, posterior_conditional_entropy=h_cond,
                       posterior_information=h_marg - h_cond, log_k_used=math.log(ku))
            if h_marg < 0.25:
                bad.append(f"context posterior COLLAPSED: entropy of the time-averaged posterior is "
                           f"{h_marg:.4f} nats -- effectively one context for the whole stream")
            if h_cond > math.log(ku) - 0.15:
                bad.append(f"context posterior is UNIFORM: mean per-step entropy {h_cond:.4f} against "
                           f"log(k_used) = {math.log(ku):.4f} -- it is not deciding anything")
            if h_marg - h_cond < 0.05:
                bad.append(f"context posterior carries no information: marginal minus conditional "
                           f"entropy = {h_marg - h_cond:.4f} nats")
            if not f.oracle_context and len(tel["created_at"]) < 1:
                starved.append(f"context CREATION never fired (the manager ran the whole stream on one "
                               f"slot) with only {arm.n_updates} feedback events at F={F}")

        # -- 3. CONSOLIDATED ADAPTERS MUST EXIST AND DIFFER FROM THE FAST ADAPTERS ----------------
        if f.consolidate:
            nm, nf = float(m.W_m.norm()), float(m.W_f.norm())
            d = float((m.W_m - m.W_f).norm()) / (nm + nf + 1e-12) if m.W_f.numel() else 1.0
            rep.update(W_m_norm=nm, W_m_vs_W_f_rel=d, n_consolidations=int(tel["n_consol"]))
            if tel["n_consol"] == 0:
                starved.append(f"CONSOLIDATION never fired at F={F}: W_m is a permanently empty store "
                               f"({arm.n_updates} feedback events over the stream)")
            elif nm <= 1e-9:
                bad.append("consolidation fired but the consolidated adapter norm is zero")
            elif d < 1e-3:
                bad.append(f"consolidated and fast adapters are indistinguishable (rel diff {d:.3g}) -- "
                           f"consolidation is a copy, not a second timescale")

        # -- 4. THE WORKSPACE MUST MOVE, AND HALTING MUST NOT SIMPLY RUN TO THE CAP ---------------
        st = np.asarray(tel["steps"], np.float64)
        rep.update(mean_steps=float(st.mean()), steps_sd=float(st.std()),
                   steps_at_cap_frac=float((st >= m.cfg.max_steps).mean()))
        if f.workspace:
            ws = np.asarray(tel["wstd"], np.float64)
            rep["workspace_step_sd"] = float(ws.mean()) if ws.size else 0.0
            if ws.size and rep["workspace_step_sd"] < 1e-8:
                bad.append("recurrent workspace is INERT: w does not change across internal steps")
            if f.halting and rep["steps_at_cap_frac"] >= 1.0 - 1e-12:
                bad.append(f"adaptive halting is INERT: every one of {st.size} steps ran to the cap "
                           f"({m.cfg.max_steps}); the halting head is never used")

        # -- 5. THE PLASTICITY CONTROLLER MUST NOT BE EMITTING A CONSTANT ------------------------
        if f.controller:
            e, l = np.asarray(tel["eta"]), np.asarray(tel["lam"])
            rep.update(eta_mean=float(e.mean()) if e.size else 0.0,
                       eta_sd=float(e.std()) if e.size else 0.0,
                       lam_sd=float(l.std()) if l.size else 0.0)
            if e.size == 0:
                bad.append("plasticity controller was never queried")
            elif e.std() < 1e-8 or l.std() < 1e-8:
                bad.append(f"plasticity controller is CONSTANT (sd eta {e.std():.3g}, sd lambda "
                           f"{l.std():.3g}) -- a hard-coded number wearing a network")

        # -- 6. THE MODULATORY VECTOR MUST NOT BE CONSTANT ---------------------------------------
        mm = np.asarray(tel["mod"], np.float64)
        rep["mod_sd"] = float(mm.std(0).max())
        if rep["mod_sd"] < 1e-8:
            bad.append("the modulatory vector m_t is constant for the whole stream")

        # -- 7. ELIGIBILITY IS CLAIMED OFF AND MUST THEREFORE COST EXACTLY NOTHING ---------------
        rep["elig_updates"] = int(tel["n_elig"])
        if tel["n_elig"] != 0:
            bad.append("eligibility ran although this environment has sparse, not delayed, feedback; "
                       "the arm is paying for a mechanism it does not claim")

    if bad:
        raise MechanismInert(f"[{tag}] MECHANISM LIVENESS FAILED before any score was reported:\n  - "
                             + "\n  - ".join(bad))
    if starved and primary:
        raise MechanismInert(
            f"[{tag}] MECHANISM INERT AT THE PRIMARY CADENCE F={F}, so no score from it may be "
            f"reported:\n  - " + "\n  - ".join(starved))
    return rep, starved


def flag_effect_probe(proto: ADRN2A, env, Wseq, seq_t, F, cfg: BCfg):
    """Toggle each component on ONE pretrained model, replay the SAME full stream, and require the
    OUTPUT to move. ADRN-1's third defect was a flag wired to nothing and its fourth was three ablations
    returning byte-identical numbers; a flag that does not move the prediction trace raises here, long
    before any table is printed. The probe replays the FULL stream, never a truncation, because a
    truncated replay can miss a gate (consolidation) that is live over the real workload."""
    base = AdrnArm("probe/on", copy.deepcopy(proto), oracle=False)
    c_on, _, _, _ = run_arm(base, env, Wseq, seq_t, F)
    out = {}
    for name in ("context", "fast", "consolidate", "workspace"):
        m = copy.deepcopy(proto)
        setattr(m.flags, name, False)
        if name == "workspace":
            m.flags.halting = False
        arm = AdrnArm(f"probe/{name}=off", m, oracle=False)
        c_off, _, _, _ = run_arm(arm, env, Wseq, seq_t, F)
        ndiff = int((c_on != c_off).sum())
        out[name] = {"steps_differing": ndiff, "acc_on": float(c_on.mean()),
                     "acc_off": float(c_off.mean())}
        if ndiff == 0:
            raise MechanismInert(
                f"FLAG '{name}' IS VACUOUS: switching it off produced a byte-identical prediction trace "
                f"over {len(c_on)} steps. An ablation of a mechanism that is not running is not an "
                f"ablation -- this is ADRN-1's defect 3 and defect 4 signature.")
    return out


def assert_arms_distinct(per_arm, cond):
    """No two differently-configured arms may return identical per-seed metrics. There is no predeclared
    exception in this benchmark: every arm differs in architecture or in a live mechanism."""
    names = sorted(per_arm)
    bad = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if (np.allclose(per_arm[a]["retention"], per_arm[b]["retention"], atol=0, rtol=0) and
                    np.allclose(per_arm[a]["overall"], per_arm[b]["overall"], atol=0, rtol=0)):
                bad.append(f"'{a}' and '{b}' returned BYTE-IDENTICAL per-seed retention AND overall "
                           f"accuracy in condition {cond} -- ADRN-1's defect-4 signature")
    if bad:
        raise MechanismInert("ARM DISTINCTNESS FAILED:\n  - " + "\n  - ".join(bad))


# =========================================================================================== arms table
ARM_ORDER = ["1_transformer_full", "2_transformer_frozen_head", "3_gru_head", "4_nn_lookup",
             "5_online_logreg", "6_adrn2a_no_context", "7_adrn2a_oracle_context",
             "8_adrn2a_inferred_context"]
LADDER = {
    "1_transformer_full": "Transformer, online FULL-model updates",
    "2_transformer_frozen_head": "Transformer FROZEN, online linear head",
    "3_gru_head": "GRU, online head",
    "4_nn_lookup": "nearest-neighbour / lookup",
    "5_online_logreg": "online logistic regression",
    "6_adrn2a_no_context": "ADRN-2A WITHOUT context separation",
    "7_adrn2a_oracle_context": "ADRN-2A with ORACLE context",
    "8_adrn2a_inferred_context": "ADRN-2A with INFERRED context",
}
CONTROLS = ARM_ORDER[:5]
GRID = {"1_transformer_full": [1e-4, 3e-4, 1e-3],
        "2_transformer_frozen_head": [3e-3, 1e-2, 3e-2],
        "3_gru_head": [3e-3, 1e-2, 3e-2],
        "4_nn_lookup": [1, 5, 15],
        "5_online_logreg": [1e-2, 3e-2, 1e-1]}


def adrn_cfg(cfg: BCfg, F):
    """ADRN-2A's own Cfg, with the creation/consolidation timing scaled to the FEEDBACK BUDGET.

    ADRN-2A's context manager counts OBSERVATIONS, and an observation is a feedback event. At F=64 the
    stream delivers 64x fewer of them than at F=1, so a threshold expressed in raw steps would make the
    manager unusable at one cadence and trigger-happy at another for reasons that have nothing to do
    with the mechanism. The thresholds below are a declared function of the labels a phase delivers, not
    a tuned constant, and the realised counts are printed.
    """
    c = A2.Cfg()
    c.d_in, c.t_win, c.n_classes = N_BITS, cfg.hist, 2
    c.chance = 0.5
    c.k_slots = cfg.k_slots
    c.max_steps = cfg.max_steps
    n_fb_phase = max(1, (cfg.phase_len - cfg.probe_len) // F)
    c.min_obs_create = max(3, n_fb_phase // 3)
    c.create_cooldown = max(3, n_fb_phase // 3)
    c.persist = 2
    c.protect_steps = max(8, 2 * F)
    c.theta_new = 0.40
    c._n_fb_phase = n_fb_phase
    return c


def build_models(cfg: BCfg, seed, rep):
    """Pretrain the digital half once per seed and hand identical weights to every arm that shares an
    architecture, so pretraining variance is not a confound between them."""
    rng = np.random.default_rng(310000 + 977 * seed)
    cfg2 = adrn_cfg(cfg, cfg.primary_F)

    d_tfm = bisect_width(lambda d: count_params(TfmEncoder(d, cfg.hist)) + 2 * (d + 1),
                         cfg.budget, 4, 256, step=2)
    d_gru = bisect_width(lambda d: count_params(GruEncoder(d, cfg.hist)) + 2 * (d + 1),
                         cfg.budget, 4, 256, step=1)
    d_h = {}
    flags = {"6_adrn2a_no_context": Flags(context=False, eligibility=False),
             "7_adrn2a_oracle_context": Flags(oracle_context=True, eligibility=False),
             "8_adrn2a_inferred_context": Flags(eligibility=False)}
    for a, f in flags.items():
        d_h[a] = A2.width_for(cfg2, f, cfg.budget, 8, 160)

    counts, comp = {}, {}
    built = {}
    for attempt in range(cfg.pre_restarts + 1):
        try:
            torch.manual_seed(9000 + seed + 100000 * attempt)
            enc_t = TfmEncoder(d_tfm, cfg.hist)
            bt, ht = pretrain_plain(enc_t, cfg.hist, cfg, rng, 9000 + seed + 100000 * attempt)
            comp["transformer"] = assert_digital_competence(
                f"transformer/seed{seed}", enc_t, ht, bt, cfg.hist, cfg,
                np.random.default_rng(510000 + seed))
            enc_g = GruEncoder(d_gru, cfg.hist)
            bg, hg = pretrain_plain(enc_g, cfg.hist, cfg, rng, 9100 + seed + 100000 * attempt)
            comp["gru"] = assert_digital_competence(
                f"gru/seed{seed}", enc_g, hg, bg, cfg.hist, cfg,
                np.random.default_rng(510100 + seed))
            adrn_by_width = {}
            for a, f in flags.items():
                key = d_h[a]
                if key not in adrn_by_width:
                    proto = ADRN2A(cfg2, f, key)
                    hh = pretrain_adrn(proto, cfg.hist, cfg, rng, 9200 + seed + 100000 * attempt)
                    meta = meta_train_controller(proto, cfg, rng, 9300 + seed)
                    comp[f"adrn_d{key}"] = assert_digital_competence(
                        f"adrn_d{key}/seed{seed}",
                        lambda X: adrn_expected_feature(proto, X)[0], hh, proto.dec, cfg.hist, cfg,
                        np.random.default_rng(510200 + seed))
                    comp[f"adrn_d{key}"].update(meta)
                    adrn_by_width[key] = {k: v for k, v in proto.state_dict().items()
                                          if not k.startswith(("W_f", "W_m", "keys"))}
                built[a] = (adrn_by_width[key], key, f)
            break
        except MechanismInert as e:
            if attempt >= cfg.pre_restarts:
                raise
            rep(f"      pretraining restart {attempt + 1}/{cfg.pre_restarts} after: "
                f"{str(e).splitlines()[0]}")

    counts["1_transformer_full"] = count_params(enc_t) + 2 * count_params(bt)
    counts["2_transformer_frozen_head"] = counts["1_transformer_full"]
    counts["3_gru_head"] = count_params(enc_g) + 2 * count_params(bg)
    dim = cfg.hist * N_BITS
    mem = max(8, cfg.budget // dim)
    counts["4_nn_lookup"] = 2 * mem * dim
    counts["5_online_logreg"] = 2 * (dim + 1)
    widths = {"1_transformer_full": d_tfm, "2_transformer_frozen_head": d_tfm, "3_gru_head": d_gru,
              "4_nn_lookup": mem, "5_online_logreg": dim + 1}
    for a in flags:
        sd, key, f = built[a]
        counts[a] = ADRN2A(cfg2, f, key).param_counts()[2]
        widths[a] = key
    return {"enc_t": enc_t, "blind_t": bt, "enc_g": enc_g, "blind_g": bg, "adrn": built,
            "counts": counts, "widths": widths, "mem": mem, "dim": dim, "comp": comp,
            "d_tfm": d_tfm, "d_gru": d_gru}


def make_arm(name, M, cfg: BCfg, cfg2, hyper):
    if name == "1_transformer_full":
        return TorchHeadArm(name, copy.deepcopy(M["enc_t"]), M["blind_t"], hyper, full=True)
    if name == "2_transformer_frozen_head":
        return TorchHeadArm(name, copy.deepcopy(M["enc_t"]), M["blind_t"], hyper, full=False)
    if name == "3_gru_head":
        return TorchHeadArm(name, copy.deepcopy(M["enc_g"]), M["blind_g"], hyper, full=False)
    if name == "4_nn_lookup":
        return LookupNNArm(name, M["mem"], int(hyper), M["dim"])
    if name == "5_online_logreg":
        return LogRegArm(name, M["dim"], hyper)
    sd, key, f = M["adrn"][name]
    model = ADRN2A(cfg2, f, key)
    model.load_state_dict(sd, strict=False)
    return AdrnArm(name, model, oracle=f.oracle_context)


def windows_for(env: Adrn2Env, hist):
    X = env.x.astype(np.float32)
    n = len(X)
    W = np.zeros((n, hist, N_BITS), np.float32)
    for j in range(hist):
        lag = hist - 1 - j
        if lag == 0:
            W[:, j] = X
        else:
            W[lag:, j] = X[:-lag]
    return W, torch.from_numpy(W), W.reshape(n, -1)


# =========================================================================================== main
def main():
    t0 = time.time()
    cfg = BCfg()
    log = []

    def rep(s=""):
        print(s, flush=True)
        log.append(s)

    W = 112
    rep("=" * W)
    rep("ADRN-2 SECTION 18 BASELINE LADDER -- eight arms on the latent-rule environment")
    rep("=" * W)
    if SMOKE:
        rep("  *** SMOKE RUN -- plumbing and liveness check, NOT a result. ***")
    rep(f"  GATES, DECLARED BEFORE ANY NUMBER EXISTS. Primary metric is RETENTION: raw held-out accuracy")
    rep(f"  inside the feedback-WITHHELD window that opens a RETURN phase (phase 4 on c1, phase 6 on c2).")
    rep(f"    G1  arm 8 beats the BEST of arms 1-5 on RETENTION by more than the MDE")
    rep(f"    G2  arm 8 beats arm 6 (no context separation) on RETENTION by more than the MDE")
    rep(f"    G3  arm 8's RETENTION is above chance {cfg.chance:.2f} at ALL of F = "
        f"{', '.join(str(f) for f in cfg.cadences)}")
    rep(f"    G4  arm 7 (ORACLE context) minus arm 8 (INFERRED) is REPORTED whatever it is")
    rep(f"    G5  ceiling/floor: the task discriminates (not all at chance, not all at ceiling)")
    rep(f"  UNIT OF REPLICATION: the SEED -- independent init AND independent environment draw. "
        f"n = {cfg.seeds}.")
    rep(f"  MDE = 3*sd/sqrt(n) on the PAIRED per-seed gap. Effect size is always the RAW held-out value;")
    rep(f"  controls and reference rows establish significance and interpretation and are never")
    rep(f"  subtracted from a reported number.")
    rep(f"  Parameter budget {cfg.budget:,}, bisected per architecture, realised counts printed below.")
    rep(f"  Config: phase_len={cfg.phase_len} W={cfg.probe_len} hist={cfg.hist} "
        f"k_slots={cfg.k_slots} max_steps={cfg.max_steps} primary F={cfg.primary_F}")

    # ---------------------------------------------------- the instrument must pass its own check first
    rep("\n  ENVIRONMENT INSTRUMENT CHECK (adrn2_env.liveness_check; raises rather than warns)")
    envcfg = {"seeds_start": 0, "phase_len": cfg.phase_len, "probe_len": cfg.probe_len,
              "feedback_every": cfg.primary_F, "relearn_feedbacks": cfg.relearn_feedbacks,
              "n_fit": 4000 if SMOKE else 20000, "n_eval": 4000 if SMOKE else 20000,
              "fit_cap": 2000 if SMOKE else 4000, "budget_cells": 8}
    live_env = env_liveness_check(envcfg)
    rep(f"    OK  c4 needs history: memoryless fit {live_env['L3_c4_memoryless_fit']:.4f} "
        f"(closed form {live_env['L3_c4_analytic_memoryless']:.4f}) vs history fit "
        f"{live_env['L3_c4_history_fit']:.4f}")
    rep(f"    OK  fitter positive control: budgeted lookup recovers c1 XOR at "
        f"{live_env['L8_fitter_c1_heldout']:.4f} held out")
    rep(f"    OK  feedback never leaks: {live_env['L9_steps_checked']} steps checked, y_feedback nulled "
        f"on every withheld step")

    # ---------------------------------------------------------------------------------- experiment
    res = {F: {a: {} for a in ARM_ORDER} for F in cfg.cadences}
    for F in cfg.cadences:
        for a in ARM_ORDER:
            for k in ("retention", "naive", "relearn", "steady", "overall", "forgetting",
                      "forgetting_after_relearn", "sec_per_example", "updates_per_example",
                      "calibrated"):
                res[F][a][k] = []
    liveness, refs, starved_at, pcounts, cal_log = {}, {}, {}, {}, {}
    flagfx = None
    fb_counts = {}

    for seed in range(cfg.seeds):
        rep(f"\n  ── seed {seed} " + "─" * 90)
        M = build_models(cfg, seed, rep)
        pcounts = M["counts"]
        c = M["comp"]
        rep(f"    digital half: per-context linear probe  " +
            "  ".join(f"{fam}=" + "/".join(f"{v:.3f}" for v in c[fam]["per_context_probe"].values())
                       for fam in c) )
        rep(f"    context-blind head " +
            "  ".join(f"{fam} {c[fam]['blind_head_acc']:.4f}" for fam in c) +
            f"   (analytic context-blind Bayes {list(c.values())[0]['blind_bayes_analytic']:.4f})")

        for F in cfg.cadences:
            cfg2 = adrn_cfg(cfg, F)
            env = Adrn2Env(seed, phase_len=cfg.phase_len, probe_len=cfg.probe_len,
                           feedback_every=F, relearn_feedbacks=cfg.relearn_feedbacks)
            Wseq_np, seq_t, flat = windows_for(env, cfg.hist)
            obs_flat = flat
            fb_counts[F] = {"total_feedback": int(env.feedback_available.sum()),
                            "per_phase": cfg2._n_fb_phase, "total_steps": env.total_len}
            refs.setdefault(F, []).append(window_references(env))

            # -- step-size calibration on a SEPARATE validation environment draw, same grid ---------
            vsched = tuple(PHASE_SCHEDULE[:cfg.val_phases])
            venv = Adrn2Env(900000 + seed, phase_len=cfg.phase_len, probe_len=cfg.probe_len,
                            feedback_every=F, relearn_feedbacks=cfg.relearn_feedbacks,
                            schedule=vsched)
            vW, vseq, vflat = windows_for(venv, cfg.hist)
            hyper = {}
            for a in ARM_ORDER:
                if a not in GRID:
                    hyper[a] = None
                    continue
                best = (-1.0, GRID[a][0])
                for v in GRID[a]:
                    arm = make_arm(a, M, cfg, cfg2, v)
                    cc, _, _, _ = run_arm(arm, venv, vflat, vseq, F)
                    s = float(cc.mean())
                    if s > best[0]:
                        best = (s, v)
                hyper[a] = best[1]
            cal_log[f"F{F}/seed{seed}"] = hyper

            for a in ARM_ORDER:
                arm = make_arm(a, M, cfg, cfg2, hyper[a])
                s0, e0 = arm.state_sig(), arm.enc_sig()
                correct, probe_hat, tp, tl = run_arm(arm, env, obs_flat, seq_t, F)
                s1, e1 = arm.state_sig(), arm.enc_sig()

                # ---- LIVENESS RUNS HERE, BEFORE THE SCORE IS ALLOWED INTO THE TABLE ----
                lv, starved = assert_liveness(f"{a}/F{F}/seed{seed}", arm, correct, probe_hat,
                                              (s0, s1, e0, e1), F, cfg, primary=(F == cfg.primary_F))
                liveness.setdefault(f"F{F}", {})[a] = lv
                if starved:
                    starved_at.setdefault(F, {}).setdefault(a, []).extend(starved)

                mt = segment_metrics(env, correct)
                for k in ("retention", "naive", "relearn", "steady", "overall", "forgetting",
                          "forgetting_after_relearn"):
                    res[F][a][k].append(mt[k])
                res[F][a]["sec_per_example"].append((tp + tl) / env.total_len)
                res[F][a]["updates_per_example"].append(arm.n_updates / env.total_len)
                res[F][a]["calibrated"].append(hyper[a])
                liveness[f"F{F}"][a]["segment_detail"] = mt["forgetting_detail"]
                liveness[f"F{F}"][a]["per_context"] = mt["per_context"]

            if flagfx is None and F == cfg.primary_F and seed == 0:
                rep(f"\n    FLAG-EFFECT PROBE (F={F}, seed 0, FULL stream): toggling each component on "
                    f"one pretrained model and requiring the prediction trace to move")
                sd, key, f8 = M["adrn"]["8_adrn2a_inferred_context"]
                proto = ADRN2A(cfg2, f8, key)
                proto.load_state_dict(sd, strict=False)
                flagfx = flag_effect_probe(proto, env, obs_flat, seq_t, F, cfg)
                for n_, v in flagfx.items():
                    rep(f"      {n_:12s} steps differing {v['steps_differing']:>6}   "
                        f"acc on/off {v['acc_on']:.4f}/{v['acc_off']:.4f}")

    # ---------------------------------------------------------------------------------- statistics
    def stat(v):
        v = np.asarray(v, np.float64)
        return float(np.nanmean(v)), (float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0)

    def paired(F, a, b, metric="retention"):
        x = np.asarray(res[F][a][metric], np.float64)
        y = np.asarray(res[F][b][metric], np.float64)
        d = x - y
        sd = float(np.nanstd(d, ddof=1)) if d.size > 1 else 0.0
        return float(np.nanmean(d)), 3 * sd / math.sqrt(max(1, d.size)), [float(v) for v in d]

    def one_sample(F, a, const, metric="retention"):
        x = np.asarray(res[F][a][metric], np.float64) - const
        sd = float(np.nanstd(x, ddof=1)) if x.size > 1 else 0.0
        return float(np.nanmean(x)), 3 * sd / math.sqrt(max(1, x.size))

    refmean = {F: {k: float(np.mean([r[k] for r in refs[F]]))
                   for k in ("blind_bayes_seen", "stale_rule", "oracle", "naive_blind_bayes_seen")}
               for F in cfg.cadences}

    rep("\n" + "=" * W)
    rep("  PARAMETER BUDGET (bisected per architecture; adapter banks counted as capacity)")
    for a in ARM_ORDER:
        note = ""
        if a == "5_online_logreg":
            note = "  <- BELOW budget by construction: a linear model on raw bits has no width to bisect"
        if a == "4_nn_lookup":
            note = f"  <- {M['mem']} exemplars x {M['dim']} bits x 2 stores"
        rep(f"    {a:28s} width {M['widths'][a]:>5}   params {pcounts[a]:>8,}{note}")

    gates, gapinfo, cfinfo = {}, {}, {}
    for F in cfg.cadences:
        prim = (F == cfg.primary_F)
        rep("\n" + "=" * W)
        rep(f"  CADENCE F = {F}  (feedback every {F} steps -> "
            f"{fb_counts[F]['total_feedback']} labels over {fb_counts[F]['total_steps']} steps, "
            f"{fb_counts[F]['per_phase']} per phase)" + ("   *** PRIMARY ***" if prim else ""))
        rep("  " + "-" * (W - 2))
        rep(f"    {'arm':28s} {'RETENTION':>17s} {'naive':>8s} {'RELEARN':>8s} {'STEADY':>8s} "
            f"{'overall':>8s} {'FORGET':>8s} {'upd/ex':>7s} {'ms/ex':>7s}")
        for a in ARM_ORDER:
            m, s = stat(res[F][a]["retention"])
            row = " ".join(f"{stat(res[F][a][k])[0]:8.4f}" for k in
                           ("naive", "relearn", "steady", "overall", "forgetting"))
            rep(f"    {a:28s} {m:>8.4f} ± {s:<6.4f} {row} "
                f"{stat(res[F][a]['updates_per_example'])[0]:7.3f} "
                f"{1000 * stat(res[F][a]['sec_per_example'])[0]:7.3f}")
        r = refmean[F]
        rep(f"    {'-- reference: stale rule':28s} {r['stale_rule']:>8.4f}          "
            f"(FLOOR: keep applying the previous phase's rule)")
        rep(f"    {'-- reference: blind Bayes':28s} {r['blind_bayes_seen']:>8.4f}          "
            f"(CEILING for any non-oracle arm inside a withheld window)")
        rep(f"    {'-- reference: oracle':28s} {r['oracle']:>8.4f}          "
            f"(context handed over; 1.0 by construction)")
        rep(f"    {'-- chance':28s} {cfg.chance:>8.4f}")

        # ------------------------------------------------ THE HEADLINE DIAGNOSTIC
        og, om, oper = paired(F, "7_adrn2a_oracle_context", "8_adrn2a_inferred_context")
        o7 = stat(res[F]["7_adrn2a_oracle_context"]["retention"])[0]
        o8 = stat(res[F]["8_adrn2a_inferred_context"]["retention"])[0]
        rep("")
        rep(f"    >>> ORACLE MINUS INFERRED (the section-18 diagnostic, reported as a headline):")
        rep(f"        arm 7 oracle context   RETENTION {o7:.4f}")
        rep(f"        arm 8 inferred context RETENTION {o8:.4f}")
        rep(f"        gap {og:+.4f}  MDE {om:.4f}  ->  " +
            ("CONTEXT DETECTION IS THE BOTTLENECK: handing the context over is worth more than the "
             "MDE, so the context manager, not the adapters, is what limits retention."
             if og > om else
             "oracle ~= inferred: handing the context over buys less than the MDE, so context "
             "DETECTION is not the limiting factor at this cadence."))
        gapinfo[f"F{F}|G4_oracle_minus_inferred"] = {"gap": og, "mde": om, "per_seed": oper,
                                                     "oracle_raw": o7, "inferred_raw": o8}

        # ------------------------------------------------ predeclared gates
        bc = max(CONTROLS, key=lambda a: stat(res[F][a]["retention"])[0])
        g1, m1, p1 = paired(F, "8_adrn2a_inferred_context", bc)
        gates[f"F{F} | G1 arm 8 beats best of arms 1-5 ({bc}) on RETENTION"] = bool(g1 > m1)
        gapinfo[f"F{F}|G1"] = {"vs": bc, "gap": g1, "mde": m1, "per_seed": p1,
                               "arm8_raw": o8, "other_raw": stat(res[F][bc]["retention"])[0]}
        g2, m2, p2 = paired(F, "8_adrn2a_inferred_context", "6_adrn2a_no_context")
        gates[f"F{F} | G2 arm 8 beats arm 6 (no context separation) on RETENTION"] = bool(g2 > m2)
        gapinfo[f"F{F}|G2"] = {"vs": "6_adrn2a_no_context", "gap": g2, "mde": m2, "per_seed": p2,
                               "arm8_raw": o8,
                               "other_raw": stat(res[F]["6_adrn2a_no_context"]["retention"])[0]}
        g3, m3 = one_sample(F, "8_adrn2a_inferred_context", cfg.chance)
        gapinfo[f"F{F}|G3"] = {"above_chance": g3, "mde": m3, "arm8_raw": o8, "chance": cfg.chance}

        # ------------------------------------------------ ceiling / floor
        vals = {a: stat(res[F][a]["retention"])[0] for a in ARM_ORDER}
        best_arm = max(vals, key=vals.get)
        at_ceiling = [a for a, v in vals.items() if v >= cfg.ceiling]
        below = []
        if at_ceiling:
            for a in ARM_ORDER:
                if a in at_ceiling:
                    continue
                g, m, _ = paired(F, best_arm, a)
                if g > m:
                    below.append((a, vals[a]))
        floor_tol = max(m1, 0.02)
        all_floor = all(abs(v - cfg.chance) <= floor_tol for v in vals.values())
        discriminates = (not all_floor) and (len(at_ceiling) < 2 or len(below) > 0)
        gates[f"F{F} | G5 ceiling/floor: the task discriminates"] = bool(discriminates)
        cfinfo[f"F{F}"] = {"per_arm_retention": vals, "best_arm": best_arm,
                           "ceiling_hitters": at_ceiling, "ceiling": cfg.ceiling,
                           "informative_below_ceiling": below, "all_at_floor": bool(all_floor),
                           "floor_tolerance": floor_tol, "chance": cfg.chance,
                           "discriminates": bool(discriminates),
                           "blind_bayes_ceiling_for_non_oracle": r["blind_bayes_seen"]}
        if all_floor:
            rep(f"\n    *** FLOOR: every arm sits within {floor_tol:.4f} of chance "
                f"({cfg.chance:.3f}). This cadence cannot separate anything and is UNINFORMATIVE. ***")
        if at_ceiling:
            rep(f"\n    *** CEILING: {len(at_ceiling)} arm(s) at or above {cfg.ceiling:.2f} "
                f"({', '.join(at_ceiling)}). No separation can be read AMONG them -- but arms falling "
                f"more than their MDE BELOW that ceiling remain highly informative: "
                f"{', '.join(f'{a} {v:.4f}' for a, v in below) or 'none'} ***")

        if F in starved_at:
            rep(f"\n    *** MECHANISM STARVED AT F={F} (predeclared rule; this cadence's arm-8 gate "
                f"readings are UNSUPPORTED, not a result): ***")
            for a, msgs in starved_at[F].items():
                for msg in msgs:
                    rep(f"        {a}: {msg}")

        assert_arms_distinct(res[F], f"F={F}")

    # G3 spans cadences
    g3ok, g3d = True, {}
    for F in cfg.cadences:
        d = gapinfo[f"F{F}|G3"]
        ok = bool(d["above_chance"] > d["mde"]) and (F not in starved_at)
        g3d[str(F)] = {"retention": d["arm8_raw"], "above_chance": d["above_chance"],
                       "mde": d["mde"], "passes": ok,
                       "supported": bool(F not in starved_at)}
        g3ok = g3ok and ok
    gates[f"G3 arm 8 RETENTION above chance at ALL of F={list(cfg.cadences)}"] = bool(g3ok)
    gapinfo["G3_by_cadence"] = g3d
    gates["G4 oracle minus inferred is REPORTED at every cadence"] = bool(
        all(np.isfinite(gapinfo[f"F{F}|G4_oracle_minus_inferred"]["gap"]) for F in cfg.cadences))

    rep("\n" + "=" * W)
    rep("  COST (measured, not asserted -- ADRN-1's efficiency claim was never measured)")
    rep(f"    {'arm':28s} " + " ".join(f"{'F=' + str(F):>18s}" for F in cfg.cadences))
    for a in ARM_ORDER:
        cells = " ".join(f"{1000 * stat(res[F][a]['sec_per_example'])[0]:8.3f}ms/"
                         f"{stat(res[F][a]['updates_per_example'])[0]:5.3f}u"
                         for F in cfg.cadences)
        rep(f"    {a:28s} {cells}")

    rep("\n  PREDECLARED GATES")
    for k, v in gates.items():
        rep(f"    {'PASS' if v else 'FAIL'}  {k}")
    npass = int(sum(gates.values()))

    # ---------------------------------------------------------------------------------- verdict
    P = cfg.primary_F
    rP = refmean[P]
    g1d, g2d, g4d = gapinfo[f"F{P}|G1"], gapinfo[f"F{P}|G2"], gapinfo[f"F{P}|G4_oracle_minus_inferred"]
    a8 = stat(res[P]["8_adrn2a_inferred_context"]["retention"])[0]
    a8_rel = stat(res[P]["8_adrn2a_inferred_context"]["relearn"])[0]
    a8_st = stat(res[P]["8_adrn2a_inferred_context"]["steady"])[0]
    a8_fg = stat(res[P]["8_adrn2a_inferred_context"]["forgetting"])[0]
    a8_ms = 1000 * stat(res[P]["8_adrn2a_inferred_context"]["sec_per_example"])[0]
    a8_up = stat(res[P]["8_adrn2a_inferred_context"]["updates_per_example"])[0]
    hedge_note = ("at or below the context-blind Bayes ceiling for a withheld window, so it is best read "
                  "as HEDGING over stored contexts rather than as RETRIEVING a particular one"
                  if a8 <= rP["blind_bayes_seen"] + 0.01 else
                  "above the context-blind Bayes ceiling computed for this window, which should be "
                  "impossible and must be investigated before the number is believed")
    verdict = (
        ("SMOKE RUN -- plumbing and liveness check, NOT a result; rerun with ADRN2_SMOKE unset. "
         if SMOKE else "") +
        f"ADRN-2 SECTION 18 BASELINE LADDER: {npass}/{len(gates)} predeclared gates pass over "
        f"n={cfg.seeds} seeds, where one seed is an independent initialisation AND an independent "
        f"environment draw, and every MDE is 3*sd/sqrt(n) on the PAIRED per-seed gap. At the primary "
        f"cadence F={P}, ADRN-2A with an INFERRED context (arm 8) reaches a raw held-out RETENTION of "
        f"{a8:.4f} inside the feedback-withheld windows that open the two return phases, against "
        f"{g1d['other_raw']:.4f} for the best of arms 1-5 ({g1d['vs']}) -- a paired per-seed gap of "
        f"{g1d['gap']:+.4f} against an MDE of {g1d['mde']:.4f} -- and against "
        f"{g2d['other_raw']:.4f} for the same system with context separation switched off (arm 6), a "
        f"gap of {g2d['gap']:+.4f} against an MDE of {g2d['mde']:.4f}. THE SECTION-18 DIAGNOSTIC, "
        f"reported as a headline rather than a footnote: handing the context over (arm 7) gives "
        f"{g4d['oracle_raw']:.4f} against arm 8's {g4d['inferred_raw']:.4f}, a gap of "
        f"{g4d['gap']:+.4f} against an MDE of {g4d['mde']:.4f}, so "
        + ("CONTEXT DETECTION IS THE BOTTLENECK -- the adapters hold what they are told to hold and it "
           "is the manager's identification of which context is active that limits retention."
           if g4d["gap"] > g4d["mde"] else
           "context detection is NOT the limiting factor at this cadence: the oracle buys less than "
           "the MDE, so what limits retention lies in the adapters rather than in identification.") +
        f" That gap has an information-theoretic explanation which is stated here rather than left to "
        f"be discovered: the environment guarantees by construction that the input distribution is "
        f"identical in every context and feedback is withheld for the entire window, so inside it there "
        f"is NO evidence about which context is active and identification is impossible for any "
        f"non-oracle system of any size. The three references computed on the realised windows bound "
        f"what is achievable -- the stale-rule floor (keep applying the previous phase's rule) at "
        f"{rP['stale_rule']:.4f}, the context-blind Bayes ceiling for a non-oracle arm (detect the "
        f"change, average over the contexts seen so far) at {rP['blind_bayes_seen']:.4f}, and the "
        f"oracle at 1.0000 -- and arm 8's {a8:.4f} sits {hedge_note}. The four metrics are reported "
        f"separately and never averaged together: at F={P} arm 8 shows RETENTION {a8:.4f}, RELEARNING "
        f"{a8_rel:.4f} once feedback resumes, STEADY {a8_st:.4f} late within a phase, and FORGETTING "
        f"{a8_fg:+.4f} measured as the drop from a context's first-visit steady accuracy to its "
        f"retention on the return; COST is measured rather than asserted at {a8_ms:.3f} ms and "
        f"{a8_up:.3f} updates per example. Every arm was built to a matched budget of ~{cfg.budget:,} "
        f"parameters by bisection with realised counts printed; the digital half of every neural arm was "
        f"pretrained by a byte-identical context-blind recipe and had to pass a per-context linear-probe "
        f"competence assertion before any online number was taken; every free scalar step size in arms "
        f"1-5 was calibrated per seed and per cadence on a SEPARATE validation environment draw over the "
        f"same grid while the ADRN-2A arms received no calibrated scalar at all; and the change-detection "
        f"signal derived from the feedback channel was given to all eight arms in the strongest form each "
        f"architecture admits, so no baseline loses for being untuned or for being denied information the "
        f"proposal was given. Before any number above was allowed into the table, every arm-run had to "
        f"pass a mechanism liveness assertion proving numerically that its fast adapters changed between "
        f"updates, that its context posterior was neither collapsed onto one context nor flat uniform, "
        f"that its consolidated adapter existed and differed from the fast adapter by more than a copy, "
        f"that its workspace moved across internal steps and did not simply run to the halting cap, that "
        f"its plasticity controller emitted a varying rate, that its encoder was frozen where it claimed "
        f"to be frozen and moved where it claimed to fine-tune, and that it actually performed online "
        f"updates at all; separately, four component flags were toggled on one pretrained model and "
        f"replayed over the SAME full stream with the prediction trace required to move, and no two "
        f"differently-configured arms were permitted to return byte-identical per-seed metrics.")
    rep("\n" + "=" * W)
    rep("  VERDICT: " + verdict)

    limits = [
        "FALSIFIER, and the most important one: inside a feedback-withheld window the environment "
        "guarantees the input distribution is identical across contexts, so no non-oracle arm can "
        "identify which context returned. If a future environment leaked ANY context information into x "
        "-- different input statistics, a marker channel, a temporal signature -- the retention metric "
        "would start measuring inference from inputs rather than retrieval from memory, every "
        "non-oracle arm would rise, and the oracle-minus-inferred gap reported here would shrink for "
        "reasons that have nothing to do with the context manager.",
        "FALSIFIER: the change-detection signal is derived from the FEEDBACK MASK -- a cadence slot that "
        "delivers no label reveals that a probe window has begun. If withholding were randomised so that "
        "gaps in feedback carried no information about phase boundaries, every arm's fallback would stop "
        "firing and retention should collapse to the stale-rule floor for all of them. That single "
        "protocol property is load-bearing for every retention number in this file and it is stated, not "
        "hidden.",
        "FALSIFIER: the digital half is pretrained on windows from all four rules, so every rule is "
        "linearly decodable before the stream starts. This benchmark therefore tests SELECTION and "
        "RETRIEVAL, not representation learning. If the encoder had to discover the rules online, arms "
        "1-3 would be far weaker and the ADRN-2A arms would have nothing linear for their adapters to "
        "fix, and neither direction of that change is measured here.",
        "CONFOUND, by design and not correctable within the specification: feedback cadence and total "
        "supervision are the same knob. F=64 delivers 64x fewer labels than F=1 over an identical "
        "stream, so a retention drop at high F cannot be attributed to delay rather than to starvation. "
        "The realised label counts per cadence are printed so the confound is visible in the table.",
        "the retention metric rests on exactly TWO windows per seed (phase 4 on c1, phase 6 on c2), of "
        f"W={cfg.probe_len} examples each, so a single seed's retention carries a binomial standard "
        f"error near {math.sqrt(0.25 / (2 * cfg.probe_len)):.3f}. The MDE is computed on the PAIRED "
        "per-seed gap, which removes the shared stream noise, but no single seed's raw retention should "
        "be over-read.",
        "the environment never returns to c4, the one TEMPORAL context, so retention is measured on c1 "
        "and c2 only -- both memoryless rules. Nothing here supports a claim about retaining a rule that "
        "requires history.",
        "arm 5 (online logistic regression) operates on raw bits and cannot represent XOR, so its "
        "ceiling on c1 is 0.5 by construction. That is a property of the baseline the specification "
        "names, not a handicap imposed here; arms 2 and 3 are the capacity-matched linear-head controls "
        "on a learned representation and they are what makes G1 a real gate. If G1 passed only because "
        "arm 5 was weak it would be worthless, which is why the best of arms 1-5 is named in the table.",
        "arm 4 is matched on MEMORY CELLS rather than on trainable parameters, because a lookup arm has "
        "no trainable parameters to match. The two budgets are not the same currency and the comparison "
        "should be read as capacity-comparable rather than capacity-identical.",
        "ablation arm 6 is allocated a WIDER encoder because it does not pay for a bank of per-context "
        "adapters. That biases G2 against arm 8, so a passing G2 is conservative and a failing G2 may "
        "partly reflect arm 6's extra capacity.",
        "the ADRN-2A context manager's creation and consolidation thresholds are expressed as a declared "
        "function of the labels a phase delivers rather than as raw step counts, because a raw threshold "
        "would make the manager unusable at F=64 and trigger-happy at F=1 for reasons unrelated to the "
        "mechanism. That scaling is a modelling choice; a different scaling would change how many "
        "contexts get allocated, and the realised counts are reported per arm.",
        "compute is matched only in parameters, not in wall-clock. The ADRN-2A arms run up to "
        f"{cfg.max_steps} workspace steps per example and evaluate every context slot's head on every "
        "step, so their measured ms/example is structurally higher; the cost table reports the realised "
        "numbers rather than asserting efficiency.",
        "eligibility is switched OFF for every ADRN-2A arm and no claim is made about it, because "
        "feedback here is SPARSE rather than DELAYED -- the label at a feedback step refers to that step. "
        "A genuinely delayed-credit variant of this environment would be needed to test that mechanism.",
        "STAGE 2A ONLY: no dendrites, no episodic memory, no world model, no planner. Nothing in this "
        "file supports a claim about those, and nothing in it is evidence about biology data.",
    ]

    R = {"model": "adrn2-benchmark-section18-ladder-v1",
         "smoke": bool(SMOKE),
         "config": cfg.asdict(),
         "unit_of_replication": "seed (independent initialisation AND independent environment draw)",
         "primary_metric": "retention = raw held-out accuracy inside the feedback-withheld window that "
                           "opens a return phase",
         "secondary_metrics": ["relearning", "steady", "forgetting", "cost"],
         "mde_rule": "3*sd/sqrt(n) on the PAIRED per-seed gap",
         "ladder": LADDER,
         "arms": {a: {"description": LADDER[a], "params": pcounts[a], "width": M["widths"][a],
                      "results": {f"F{F}": {k: (v if k == "calibrated" else
                                                {"mean": stat(v)[0], "sd": stat(v)[1],
                                                 "per_seed": [float(x) for x in v]})
                                            for k, v in res[F][a].items()}
                                  for F in cfg.cadences}} for a in ARM_ORDER},
         "references": {f"F{F}": refmean[F] for F in cfg.cadences},
         "reference_detail_seed0": {f"F{F}": refs[F][0] for F in cfg.cadences},
         "feedback_budget": {f"F{F}": fb_counts[F] for F in cfg.cadences},
         "mde": {k: {"gap": v.get("gap"), "mde": v.get("mde")} for k, v in gapinfo.items()
                 if isinstance(v, dict) and "gap" in v},
         "gaps": gapinfo,
         "gates": gates,
         "ceiling_floor": cfinfo,
         "mechanism_starved": {f"F{F}": v for F, v in starved_at.items()},
         "flag_effect_probe": flagfx,
         "calibrated_step_sizes": cal_log,
         "liveness": _jsonable(liveness),
         "environment_instrument_check": _jsonable(live_env),
         "verdict": verdict,
         "limits": limits,
         "runtime_sec": time.time() - t0,
         "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "adrn2_benchmark.json"
    with open(p, "w") as fh:
        json.dump(_jsonable(R), fh, indent=1, default=float)
    rep(f"\n  -> {p}    ({time.time() - t0:.1f}s)")
    return 0


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


if __name__ == "__main__":
    sys.exit(main())
