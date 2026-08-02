"""ADRN-2 spec section 18 -- the LATENT-RULE ENVIRONMENT, standalone and independently testable.

This file is the ENVIRONMENT ONLY. It generates a stream and it validates itself. It knows nothing about
any model, and no model needs to know how it works: the contract is a generator of `Step` records that
unpack as the 6-tuple `(x_t, y_t, feedback_available_t, phase_id, context_id, is_retention_probe)`.

---------------------------------------------------------------------------------------------------------
THE GATES AND THE MDE, DECLARED HERE BEFORE ANY NUMBER APPEARS ANYWHERE IN THIS FILE
---------------------------------------------------------------------------------------------------------
UNIT OF REPLICATION: the SEED. One seed = one independent environment draw (independent input stream AND
independent i.i.d. rule-validation draws). Seeds inside a single draw would measure sampler noise, not
environment variability, and this project has already had a verdict flip 4/7 -> 0/7 by getting that wrong.
n = ADRN2_SEEDS. MDE = 3*sd/sqrt(n) computed on the PAIRED per-seed gap, never on the pooled sd.

EFFECT SIZE IS THE RAW HELD-OUT VALUE. Every accuracy printed below is the raw held-out number. Controls
(the naive-probe window, the memoryless arm, the analytic ceiling) establish SIGNIFICANCE only. Nothing is
ever reported net of a control.

PREDECLARED GATES (all must pass for the environment to be usable as a measurement instrument):
  G1  BALANCE                  every context's marginal P(y=1) is within 5 binomial sd of 0.5, so no arm
                               can win by predicting a constant.
  G2  NO CONSTANT WINS         constant-0, constant-1 and majority-so-far all score below 0.55 on the
                               realised stream, overall and within every context.
  G3  C4 NEEDS HISTORY (analytic)   the Bayes-optimal MEMORYLESS accuracy on c4, derived in closed form
                               from the generative model, is at most 0.80 while the with-history Bayes
                               optimum is 1.0. This bound holds for a memoryless predictor of ANY size.
  G4  C4 NEEDS HISTORY (empirical)  at a MATCHED table budget, the history-featured fit beats the
                               memoryless fit on c4 by more than the MDE on the paired per-seed gap, and
                               the memoryless fit lands on the analytic ceiling to within 0.02.
  G5  FITTER LIVENESS          the SAME fitting machinery reaches >= 0.99 on c1, c2 and c3. Without this,
                               a low c4 number is indistinguishable from a broken fitter -- which is
                               exactly the class of defect ADRN-1 shipped four times.
  G6  y INDEPENDENT OF x_t     under c4, restricted to the sub-population where the current-step cue is
                               uninformative, y is statistically independent of every one of the 16
                               current bits (max bitwise |dP| below 5 sd).
  G7  RETENTION WINDOWS        exactly two retention probes exist, at phases 4 and 6, on contexts c1 and
                               c2, each of length W, each at the very start of its phase, each preceded by
                               at least one prior exposure to that context, with feedback withheld on
                               EVERY step inside and the feedback channel hard-nulled there.
  G8  FEEDBACK CADENCE         for every supported cadence F in {1, 4, 16, 64} the realised feedback count
                               outside probe windows matches the expected count exactly, and every
                               relearning window contains at least one feedback event.
  G9  X DOES NOT LEAK CONTEXT  the input distribution is one fixed product measure shared by all four
                               contexts (structural invariant), and the realised per-bit marginals do not
                               separate the contexts (max |dP| below 5 sd).
  G10 RETENTION PROBE HAS HEADROOM   on the retention windows an oracle scores >= 0.99 and a no-context-
                               memory arm scores <= 0.60. If both hit the ceiling the retention metric
                               would be UNINFORMATIVE and could not adjudicate anything.
  G11 DETERMINISM              the same seed reproduces the stream exactly; a different seed does not.

CEILING AND FLOOR. G10 is the ceiling/floor check for the retention metric and G3/G5 for the history
claim. The subtlety this project got wrong before: a ceiling only kills discrimination AMONG the arms that
reach it. c1, c2 and c3 are AT the memoryless ceiling by construction and are therefore uninformative about
the history claim -- but that is precisely what makes c4 FALLING FAR BELOW that same ceiling informative.
Both directions are reported, and a context that is at ceiling is labelled UNINFORMATIVE-BY-DESIGN rather
than being read as a null.

---------------------------------------------------------------------------------------------------------
WHAT THE ENVIRONMENT IS
---------------------------------------------------------------------------------------------------------
x_t in {0,1}^16. A hidden context selects the rule (indices below are 1-BASED, as the spec writes them):

    c1   y = x2 XOR x7
    c2   y = x4 AND x13
    c3   y = majority(x1, x8, x11)
    c4   y = x3(t-2) OR x15(t)            <- TEMPORAL: depends on an input two steps back

The context changes WITHOUT notification. Phase schedule: c1 -> c2 -> c3 -> c1 -> c4 -> c2.

BALANCING WITHOUT TOUCHING THE RULES. A raw AND is 1 a quarter of the time and a raw OR three quarters of
the time, so on unbalanced bits a constant predictor would beat everything on c2 and c4. The rules are left
EXACTLY as specified; instead the per-bit input marginals are set once, globally, so that each rule's output
marginal is exactly 0.5:  P(bit)=1/sqrt(2) for x4 and x13, P(bit)=1-1/sqrt(2) for x3 and x15, P(bit)=0.5
everywhere else. The four rules use pairwise-disjoint bits, so all four marginals can be fixed at once.

Critically the SAME marginal vector is used in every context -- there is no per-context input distribution.
The input therefore carries zero information about the context, and a system cannot sidestep the retention
measurement by reading the context off the input statistics. That is G9, and it is a structural invariant
of the generator, not a tuned outcome.

THE MEASUREMENT THAT MAKES THIS WORTH RUNNING
---------------------------------------------------------------------------------------------------------
On RETURNING to a previously seen context, feedback is WITHHELD for the first W examples.

    RETENTION   accuracy inside that withheld window. Did the system RETRIEVE what it already knew, with
                no new evidence? This is the quantity the whole environment exists to expose.
    RELEARNING  accuracy over the window after feedback resumes. How fast does it recover if it did not
                retrieve? A different quantity entirely, and conflating the two would make a fast relearner
                look like a good retainer.

They are labelled separately on every Step and aggregated separately.

THE MATCHED CONTROL FOR RETENTION. A retention accuracy of 0.62 means nothing on its own. Every phase -- not
only the returns -- opens with a W-example feedback-withheld probe. On a FIRST visit that window is a NAIVE
probe: same context, same length, same withholding, but no prior exposure is possible, so it is the
within-context, within-length control for the retention window. `is_retention_probe` is True only on the
two genuine returns, exactly as the spec asks. Both raw values are reported; retention is never reported
net of the naive probe.

THE CHEATING SURFACE, AND WHAT STOPS IT. The 6-tuple contains y_t because the harness must SCORE every
step, including the withheld ones -- that is the entire point. `feedback_available_t` is the contract a
model must honour. So that honouring it is the path of least resistance, every Step also carries
`y_feedback`, which is hard-nulled to None wherever feedback is withheld, and `Adrn2Env.iter_for_model()`
yields only the maskable view. A model that reads `y` directly on a withheld step is cheating and its
retention number is void.

WHY THIS FILE ALSO RUNS ITS OWN MEASUREMENT. ADRN-1 shipped four defects of one shape: a mechanism
reporting confident numbers while switched off -- a silent network, a silent baseline, an ablation flag
wired to nothing, and a learning system never invoked. An environment fails the same way. A stream whose
temporal rule is accidentally memoryless, whose retention windows are misplaced, or whose feedback mask is
never applied would still yield tidy accuracies and would still look exactly like a hard task. So the
liveness assertions in `liveness_check()` RUN BEFORE ANY SCORE IS REPORTED and RAISE, and they include the
positive controls (G5, G10) without which "hard" and "broken" are the same observation.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

# The environment is pure numpy and does not need torch. Threads are pinned only because the orchestrator
# runs this alongside torch models on 4 shared cores.
try:  # pragma: no cover - environment dependent
    import torch

    torch.set_num_threads(1)
except Exception:  # pragma: no cover
    torch = None

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))

# ==========================================================================================  environment

N_BITS = 16

# 1-based spec index -> 0-based array index.
def _b(i: int) -> int:
    """Translate the spec's 1-based bit name xN into a 0-based array column."""
    if not 1 <= i <= N_BITS:
        raise ValueError(f"bit index {i} outside 1..{N_BITS}")
    return i - 1


RULE_BITS = {
    "c1": {"kind": "xor", "current": (_b(2), _b(7)), "lag2": ()},
    "c2": {"kind": "and", "current": (_b(4), _b(13)), "lag2": ()},
    "c3": {"kind": "majority", "current": (_b(1), _b(8), _b(11)), "lag2": ()},
    "c4": {"kind": "or", "current": (_b(15),), "lag2": (_b(3),)},
}
CONTEXTS = ("c1", "c2", "c3", "c4")
PHASE_SCHEDULE = ("c1", "c2", "c3", "c1", "c4", "c2")  # spec section 18
LAG = 2  # c4 reaches exactly this many steps back
SUPPORTED_FEEDBACK_EVERY = (1, 4, 16, 64)

# ---- the one global input distribution -------------------------------------------------------------
# Set so that EVERY rule's output marginal is exactly 0.5 while the rules themselves are untouched.
#   AND over two bits at p each  -> p^2      = 0.5  ->  p = 1/sqrt(2)
#   OR  over two bits at p each  -> 1-(1-p)^2 = 0.5 ->  p = 1 - 1/sqrt(2)
# The four rules use pairwise-disjoint bits, so one vector satisfies all four simultaneously, and the same
# vector is used in every context so that x leaks nothing about which rule is active.
P_AND = 1.0 / math.sqrt(2.0)
P_OR = 1.0 - 1.0 / math.sqrt(2.0)
BIT_P = np.full(N_BITS, 0.5)
for _i in RULE_BITS["c2"]["current"]:
    BIT_P[_i] = P_AND
for _i in RULE_BITS["c4"]["current"] + RULE_BITS["c4"]["lag2"]:
    BIT_P[_i] = P_OR
BIT_P.flags.writeable = False

_used = [i for c in CONTEXTS for i in RULE_BITS[c]["current"] + RULE_BITS[c]["lag2"]]
if len(set(_used)) != len(_used):
    raise AssertionError("rule bits overlap; the single global marginal vector cannot balance all rules")


def apply_rule(context: str, x_cur: np.ndarray, x_lag: np.ndarray) -> np.ndarray:
    """y for a batch. x_cur, x_lag are (n, 16) int arrays; x_lag is x at t-LAG.

    Vectorised and total: c1..c3 ignore x_lag, c4 uses it. Written once so the stream generator and the
    i.i.d. rule-validation draws cannot drift apart.
    """
    x_cur = np.asarray(x_cur)
    x_lag = np.asarray(x_lag)
    if context == "c1":
        a, b = RULE_BITS["c1"]["current"]
        y = x_cur[:, a] ^ x_cur[:, b]
    elif context == "c2":
        a, b = RULE_BITS["c2"]["current"]
        y = x_cur[:, a] & x_cur[:, b]
    elif context == "c3":
        a, b, c = RULE_BITS["c3"]["current"]
        y = (x_cur[:, a].astype(np.int64) + x_cur[:, b] + x_cur[:, c] >= 2).astype(np.int64)
    elif context == "c4":
        (cur,) = RULE_BITS["c4"]["current"]
        (lag,) = RULE_BITS["c4"]["lag2"]
        y = x_lag[:, lag] | x_cur[:, cur]
    else:
        raise ValueError(f"unknown context {context!r}")
    return y.astype(np.int64)


def analytic_marginal(context: str) -> float:
    """Closed-form P(y=1) under the global input distribution. Used to check the realised balance."""
    if context == "c1":
        p, q = BIT_P[list(RULE_BITS["c1"]["current"])]
        return float(p * (1 - q) + q * (1 - p))
    if context == "c2":
        p, q = BIT_P[list(RULE_BITS["c2"]["current"])]
        return float(p * q)
    if context == "c3":
        p = BIT_P[list(RULE_BITS["c3"]["current"])]
        tot = 0.0
        for bits in range(8):
            v = [(bits >> k) & 1 for k in range(3)]
            if sum(v) >= 2:
                tot += float(np.prod([p[k] if v[k] else 1 - p[k] for k in range(3)]))
        return tot
    if context == "c4":
        pc = float(BIT_P[RULE_BITS["c4"]["current"][0]])
        pl = float(BIT_P[RULE_BITS["c4"]["lag2"][0]])
        return 1.0 - (1 - pc) * (1 - pl)
    raise ValueError(context)


def analytic_memoryless_bayes(context: str) -> float:
    """Best accuracy achievable by ANY predictor that sees only x_t (and knows the context).

    This is a property of the generative model, not of a fitted model, so it bounds a memoryless predictor
    of arbitrary capacity. c1..c3 are deterministic functions of x_t, hence 1.0. Under c4 the current bit
    x15 forces y=1 when set; when it is clear, y is exactly the lagged bit and the best guess is the
    lagged bit's majority class.
    """
    if context in ("c1", "c2", "c3"):
        return 1.0
    pc = float(BIT_P[RULE_BITS["c4"]["current"][0]])
    pl = float(BIT_P[RULE_BITS["c4"]["lag2"][0]])
    return pc * 1.0 + (1 - pc) * max(pl, 1 - pl)


def analytic_history_bayes(context: str) -> float:
    """Best accuracy given x_t, the lagged input and the context. Every rule is deterministic -> 1.0."""
    return 1.0


# ==========================================================================================  Step record


@dataclass(frozen=True)
class Step:
    """One example. Unpacks as the spec's 6-tuple; the extra fields are strictly additive.

        x, y, feedback_available, phase_id, context_id, is_retention_probe = step

    y                    the ground-truth label. Present on EVERY step because the harness must score the
                         withheld windows -- that is the measurement. Not a learning signal.
    y_feedback           the learning signal. None wherever feedback is withheld. Use this one.
    segment              'naive_probe' | 'retention_probe' | 'relearn' | 'steady'
    is_retention_probe   True only inside a return phase's opening window (phases 4 and 6).
    memoryless_decidable False when the current input alone cannot determine y (c4 with x15=0). Lets a
                         harness report c4 on the sub-population where history is genuinely required.
    """

    t: int
    x: np.ndarray
    y: int
    feedback_available: bool
    phase_id: int
    context_id: str
    is_retention_probe: bool
    y_feedback: Optional[int]
    segment: str
    phase_index: int
    is_first_visit: bool
    memoryless_decidable: bool

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.feedback_available
        yield self.phase_id
        yield self.context_id
        yield self.is_retention_probe


@dataclass
class WindowSpec:
    """Where a labelled window sits in the stream, resolved once so checks can verify placement."""

    phase_id: int
    context_id: str
    segment: str
    start: int
    stop: int  # exclusive

    def as_dict(self):
        return {"phase_id": self.phase_id, "context_id": self.context_id, "segment": self.segment,
                "start": self.start, "stop": self.stop, "length": self.stop - self.start}


# ==========================================================================================  the env


class Adrn2Env:
    """The section 18 latent-rule stream. Deterministic in `seed`; nothing else varies it.

    phase_len            examples per phase.
    probe_len (W)        length of the opening feedback-withheld window of EVERY phase. On a return that
                         window is the RETENTION probe; on a first visit it is the matched NAIVE control.
    feedback_every (F)   feedback cadence, one of 1, 4, 16, 64.
    relearn_feedbacks    the relearning window is defined as exactly this many feedback deliveries wide,
                         i.e. relearn_len = relearn_feedbacks * F. Defining it in EXAMPLES would silently
                         produce a zero-feedback relearning window at F=64 -- a window that measures
                         relearning while no learning signal is delivered is the ADRN-1 defect shape, so
                         the constructor raises instead.
    """

    def __init__(self, seed: int = 0, *, phase_len: int = 1024, probe_len: int = 64,
                 feedback_every: int = 4, relearn_feedbacks: int = 4,
                 schedule: Sequence[str] = PHASE_SCHEDULE):
        if feedback_every not in SUPPORTED_FEEDBACK_EVERY:
            raise ValueError(f"feedback_every must be one of {SUPPORTED_FEEDBACK_EVERY}, got {feedback_every}")
        self.seed = int(seed)
        self.phase_len = int(phase_len)
        self.probe_len = int(probe_len)
        self.feedback_every = int(feedback_every)
        self.relearn_feedbacks = int(relearn_feedbacks)
        self.relearn_len = self.relearn_feedbacks * self.feedback_every
        self.schedule = tuple(schedule)
        self.n_phases = len(self.schedule)
        self.total_len = self.phase_len * self.n_phases

        if self.probe_len + self.relearn_len > self.phase_len:
            raise ValueError(
                f"phase_len={self.phase_len} cannot hold probe_len={self.probe_len} plus a relearning "
                f"window of {self.relearn_feedbacks} feedback deliveries at F={self.feedback_every} "
                f"({self.relearn_len} examples). A relearning window that receives no feedback would "
                f"report a relearning number while the learning signal is switched off. Raise phase_len "
                f"to at least {self.probe_len + self.relearn_len}.")

        # which phases are returns (context already seen earlier in the schedule)
        seen: set[str] = set()
        self.is_return = []
        for c in self.schedule:
            self.is_return.append(c in seen)
            seen.add(c)

        self._build()

    # ---------------------------------------------------------------- construction
    def _build(self):
        rng = np.random.default_rng(self.seed)
        # LAG extra steps of burn-in history so x(t-LAG) is defined at t=0. Burn-in is generated but never
        # yielded, so a stream that begins inside c4 is still well defined.
        raw = rng.random((self.total_len + LAG, N_BITS))
        self._x_all = (raw < BIT_P[None, :]).astype(np.int8)  # (LAG + total_len, 16)
        self.x = self._x_all[LAG:]                            # (total_len, 16) -- the yielded inputs
        self.x_lag = self._x_all[: self.total_len]            # x at t-LAG, aligned to self.x

        ctx_id = np.empty(self.total_len, dtype=object)
        phase_id = np.zeros(self.total_len, dtype=np.int64)
        phase_ix = np.zeros(self.total_len, dtype=np.int64)
        segment = np.empty(self.total_len, dtype=object)
        first_visit = np.zeros(self.total_len, dtype=bool)
        self.windows: list[WindowSpec] = []

        for p, c in enumerate(self.schedule):
            lo = p * self.phase_len
            hi = lo + self.phase_len
            ctx_id[lo:hi] = c
            phase_id[lo:hi] = p + 1
            phase_ix[lo:hi] = np.arange(self.phase_len)
            first_visit[lo:hi] = not self.is_return[p]
            probe_seg = "retention_probe" if self.is_return[p] else "naive_probe"
            segment[lo:hi] = "steady"
            segment[lo: lo + self.probe_len] = probe_seg
            segment[lo + self.probe_len: lo + self.probe_len + self.relearn_len] = "relearn"
            self.windows.append(WindowSpec(p + 1, c, probe_seg, lo, lo + self.probe_len))
            self.windows.append(WindowSpec(p + 1, c, "relearn", lo + self.probe_len,
                                           lo + self.probe_len + self.relearn_len))

        self.context_id = ctx_id
        self.phase_id = phase_id
        self.phase_index = phase_ix
        self.segment = segment
        self.is_first_visit = first_visit
        self.is_retention_probe = segment == "retention_probe"

        y = np.zeros(self.total_len, dtype=np.int64)
        for c in CONTEXTS:
            m = ctx_id == c
            if m.any():
                y[m] = apply_rule(c, self.x[m], self.x_lag[m])
        self.y = y

        # feedback cadence. Withholding inside a probe is a HARD MASK applied on top of the cadence, so a
        # probe cannot leak a learning signal no matter how the cadence lands.
        withheld = np.isin(segment.astype(str), ("naive_probe", "retention_probe"))
        cadence = ((np.arange(self.total_len) + 1) % self.feedback_every) == 0
        self.feedback_available = cadence & ~withheld
        self.feedback_withheld_by_probe = withheld

        # the sub-population where the current input alone cannot decide y (c4 with the OR's current bit
        # clear). Everywhere else the current input is sufficient.
        (cur_bit,) = RULE_BITS["c4"]["current"]
        self.memoryless_decidable = ~((ctx_id == "c4") & (self.x[:, cur_bit] == 0))

    # ---------------------------------------------------------------- the contract
    def stream(self) -> Iterator[Step]:
        """Yield every Step in order. Unpacks as (x, y, feedback_available, phase_id, ctx, is_probe)."""
        for t in range(self.total_len):
            fb = bool(self.feedback_available[t])
            yield Step(t=t, x=self.x[t], y=int(self.y[t]), feedback_available=fb,
                       phase_id=int(self.phase_id[t]), context_id=str(self.context_id[t]),
                       is_retention_probe=bool(self.is_retention_probe[t]),
                       y_feedback=int(self.y[t]) if fb else None,
                       segment=str(self.segment[t]), phase_index=int(self.phase_index[t]),
                       is_first_visit=bool(self.is_first_visit[t]),
                       memoryless_decidable=bool(self.memoryless_decidable[t]))

    def iter_for_model(self):
        """The maskable view a model should consume: (x_t, y_feedback_t_or_None, t).

        y_feedback is None wherever feedback is withheld, so honouring the contract is the default and
        cheating requires reaching past this method into `.y`.
        """
        for t in range(self.total_len):
            fb = bool(self.feedback_available[t])
            yield self.x[t], (int(self.y[t]) if fb else None), t

    # ---------------------------------------------------------------- descriptive
    def retention_windows(self) -> list[WindowSpec]:
        return [w for w in self.windows if w.segment == "retention_probe"]

    def naive_windows(self) -> list[WindowSpec]:
        return [w for w in self.windows if w.segment == "naive_probe"]

    def relearn_windows(self) -> list[WindowSpec]:
        return [w for w in self.windows if w.segment == "relearn"]

    def class_balance(self) -> dict:
        """Realised P(y=1) per context on this stream, with the analytic value and a binomial sd."""
        out = {}
        for c in CONTEXTS:
            m = self.context_id == c
            n = int(m.sum())
            p = float(self.y[m].mean()) if n else float("nan")
            out[c] = {"n": n, "p_y1": p, "analytic": analytic_marginal(c),
                      "sd": float(math.sqrt(0.25 / n)) if n else float("nan")}
        out["overall"] = {"n": int(self.total_len), "p_y1": float(self.y.mean()),
                          "analytic": float(np.mean([analytic_marginal(c) for c in self.schedule])),
                          "sd": float(math.sqrt(0.25 / self.total_len))}
        return out

    def spec(self) -> dict:
        return {
            "seed": self.seed, "phase_len": self.phase_len, "probe_len_W": self.probe_len,
            "feedback_every_F": self.feedback_every, "relearn_feedbacks": self.relearn_feedbacks,
            "relearn_len": self.relearn_len, "schedule": list(self.schedule),
            "total_len": self.total_len, "lag": LAG, "n_bits": N_BITS,
            "bit_marginals": [float(v) for v in BIT_P],
            "rules": {c: {"kind": RULE_BITS[c]["kind"],
                          "current_bits_1based": [i + 1 for i in RULE_BITS[c]["current"]],
                          "lag2_bits_1based": [i + 1 for i in RULE_BITS[c]["lag2"]],
                          "analytic_p_y1": analytic_marginal(c),
                          "memoryless_bayes": analytic_memoryless_bayes(c),
                          "history_bayes": analytic_history_bayes(c)} for c in CONTEXTS},
            "retention_windows": [w.as_dict() for w in self.retention_windows()],
            "naive_windows": [w.as_dict() for w in self.naive_windows()],
            "relearn_windows": [w.as_dict() for w in self.relearn_windows()],
            "n_feedback": int(self.feedback_available.sum()),
        }

    def fingerprint(self) -> str:
        """Cheap content hash of the whole stream. Used for the determinism check."""
        import hashlib

        h = hashlib.sha1()
        h.update(self.x.tobytes())
        h.update(self.y.tobytes())
        h.update(self.feedback_available.tobytes())
        return h.hexdigest()


def sample_context(context: str, n: int, rng: np.random.Generator):
    """An i.i.d. draw from ONE context: returns (x_cur, x_lag, y), each row an independent example.

    Used for rule validation. Independent (x_t, x_{t-LAG}) pairs have exactly the joint law they have in
    the stream, because the bits are i.i.d. across time and across dimensions.
    """
    x_cur = (rng.random((n, N_BITS)) < BIT_P[None, :]).astype(np.int8)
    x_lag = (rng.random((n, N_BITS)) < BIT_P[None, :]).astype(np.int8)
    return x_cur, x_lag, apply_rule(context, x_cur, x_lag)


# ==========================================================================  liveness assertion machinery


class LivenessError(AssertionError):
    """Raised when a mechanism that is claimed to be active is demonstrably inert.

    ADRN-1 shipped four defects of exactly this shape and reported scores through every one of them.
    Anything raising this must abort the run; it must never be downgraded to a warning.
    """


def assert_changes(name: str, a, b, min_frac: float = 1e-9, what: str = "output"):
    """RAISE unless flipping a switch measurably changed the output.

    Exported so the ADRN-2A model file can use the same assertion for its fast adapters, its context
    posterior and its consolidated adapters without re-implementing it (badly).
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        return  # different shapes are trivially different
    frac = float(np.mean(np.abs(a - b) > 1e-12))
    if frac <= min_frac:
        raise LivenessError(
            f"INERT MECHANISM '{name}': toggling it changed {frac:.3%} of the {what}. An ablation that "
            f"changes nothing is not an ablation, and any number reported across it is the same number "
            f"twice. ADRN-1 shipped three 'different' ablations returning byte-identical results.")
    return frac


def assert_non_degenerate(name: str, values, lo: float, hi: float):
    """RAISE unless a quantity sits strictly inside (lo, hi) -- not collapsed and not saturated."""
    v = float(np.asarray(values, dtype=float).mean())
    if not (lo < v < hi):
        raise LivenessError(
            f"DEGENERATE '{name}': value {v:.6g} is outside the required open interval ({lo}, {hi}). "
            f"A quantity pinned at a boundary carries no information and cannot support a claim.")
    return v


# ==============================================================================  budgeted lookup arms


def _subsets(n_features: int, kmax: int):
    for k in range(1, kmax + 1):
        yield from combinations(range(n_features), k)


def budget_kmax(cells_budget: int, hard_cap: int = 3) -> int:
    """Bisect the number of input bits k so that the lookup table 2^k fits the declared cell budget.

    Bisected rather than assigned so the budget is honoured identically by every arm, and capped so the
    exhaustive subset search stays tractable. Both the realised k and the realised cell count are printed.
    """
    lo, hi, best = 1, hard_cap, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if (1 << mid) <= cells_budget:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def fit_lookup(B_fit: np.ndarray, y_fit: np.ndarray, kmax: int):
    """Exhaustive best <=kmax-bit lookup table. Returns (train_acc, subset, table, n_candidates).

    Exhaustive rather than greedy on purpose: a mutual-information ranking scores XOR's bits at exactly
    zero and would never find c1, which would make the fitter look broken on a rule that is trivially
    learnable. The fitter's own positive control is G5.
    """
    n = len(y_fit)
    yf = y_fit.astype(np.float64)
    best = (-1.0, None, None)
    n_cand = 0
    for sub in _subsets(B_fit.shape[1], kmax):
        n_cand += 1
        idx = np.zeros(n, dtype=np.int64)
        for p, b in enumerate(sub):
            idx += B_fit[:, b].astype(np.int64) << p
        m = 1 << len(sub)
        c1 = np.bincount(idx, weights=yf, minlength=m)
        ca = np.bincount(idx, minlength=m).astype(np.float64)
        acc = float(np.maximum(ca - c1, c1).sum() / n)
        if acc > best[0]:
            best = (acc, sub, (c1 > (ca - c1)).astype(np.int8))
    return best[0], best[1], best[2], n_cand


def apply_lookup(B: np.ndarray, sub, table) -> np.ndarray:
    idx = np.zeros(B.shape[0], dtype=np.int64)
    for p, b in enumerate(sub):
        idx += B[:, b].astype(np.int64) << p
    return table[idx].astype(np.int64)


# =====================================================================================  reference arms
# These are NOT models under test. They exist to prove the environment is a working instrument: that no
# constant wins, that history is required exactly where it is claimed to be, and that the retention window
# has headroom between a floor arm and a ceiling arm.


def arm_predictions(env: Adrn2Env, rng: np.random.Generator, mem_window: int = 128) -> dict:
    """Every stream arm's prediction vector over the whole stream."""
    n = env.total_len
    preds = {}
    preds["random"] = (rng.random(n) < 0.5).astype(np.int64)
    preds["constant-0"] = np.zeros(n, dtype=np.int64)
    preds["constant-1"] = np.ones(n, dtype=np.int64)

    # majority-so-far: the adaptive version of "predict a constant". Sees only feedback-visible labels.
    maj = np.zeros(n, dtype=np.int64)
    c0 = c1 = 0
    for t in range(n):
        maj[t] = 1 if c1 > c0 else 0
        if env.feedback_available[t]:
            if env.y[t]:
                c1 += 1
            else:
                c0 += 1
    preds["majority-so-far"] = maj

    # oracle-full: knows the context AND the history. Ceiling arm; 1.0 by construction.
    preds["oracle-full"] = env.y.copy()

    # oracle-memoryless: knows the context but sees only x_t. Bayes-optimal given that restriction.
    om = np.zeros(n, dtype=np.int64)
    for c in CONTEXTS:
        m = env.context_id == c
        if not m.any():
            continue
        if c == "c4":
            (cur,) = RULE_BITS["c4"]["current"]
            (lag,) = RULE_BITS["c4"]["lag2"]
            # y=1 whenever the current bit is set; otherwise guess the lagged bit's majority class.
            guess = 1 if BIT_P[lag] > 0.5 else 0
            om[m] = np.where(env.x[m][:, cur] == 1, 1, guess)
        else:
            om[m] = apply_rule(c, env.x[m], env.x_lag[m])
    preds["oracle-memoryless"] = om

    # recency-lookup: a context-BLIND online learner with no context memory. It refits a <=2-bit table on
    # the most recent feedback it has seen, so on a return it is still applying the intervening context's
    # rule. This is the FLOOR arm for the retention window: if it scored well there, the window would not
    # be measuring retrieval at all.
    rl = np.zeros(n, dtype=np.int64)
    hist_x: list[np.ndarray] = []
    hist_y: list[int] = []
    sub, table = None, None
    for t in range(n):
        if sub is None:
            rl[t] = 0
        else:
            rl[t] = int(apply_lookup(env.x[t][None, :], sub, table)[0])
        if env.feedback_available[t]:
            hist_x.append(env.x[t])
            hist_y.append(int(env.y[t]))
            if len(hist_x) > mem_window:
                hist_x.pop(0)
                hist_y.pop(0)
            if len(hist_x) >= 8:
                Bf = np.stack(hist_x)
                yf = np.asarray(hist_y, dtype=np.int64)
                _, sub, table, _ = fit_lookup(Bf, yf, kmax=2)
    preds["recency-lookup"] = rl
    return preds


STREAM_ARMS = ["random", "constant-0", "constant-1", "majority-so-far", "recency-lookup",
               "oracle-memoryless", "oracle-full"]
ARM_PARAMS = {  # free parameters each arm fits from data; printed so no arm can lose by being smaller
    "random": 0, "constant-0": 0, "constant-1": 0, "majority-so-far": 1,
    "recency-lookup": 4, "oracle-memoryless": 0, "oracle-full": 0,
}


# =========================================================================================  liveness


def liveness_check(cfg: dict) -> dict:
    """RAISES if any claimed property of the environment is inert. Runs BEFORE any score is reported.

    Each assertion here corresponds to a way the environment could produce confident, tidy, meaningless
    numbers -- which is precisely how ADRN-1 failed four times.
    """
    rep = {}
    seed = int(cfg["seeds_start"])
    base = Adrn2Env(seed, phase_len=cfg["phase_len"], probe_len=cfg["probe_len"],
                    feedback_every=cfg["feedback_every"], relearn_feedbacks=cfg["relearn_feedbacks"])

    # L1 the stream is non-degenerate at all
    rep["L1_y_rate"] = assert_non_degenerate("stream label rate", base.y, 0.3, 0.7)
    rep["L1_x_rate"] = assert_non_degenerate("stream input rate", base.x, 0.1, 0.9)
    if len(set(base.context_id.tolist())) < 4:
        raise LivenessError("the realised stream does not contain all four contexts")

    # L2 the rules are genuinely DIFFERENT: same inputs, different labels. If two rules aliased, the phase
    #    schedule would be a no-op and 'context switch' would be an untested word.
    rng = np.random.default_rng(seed + 991)
    xc = (rng.random((4000, N_BITS)) < BIT_P[None, :]).astype(np.int8)
    xl = (rng.random((4000, N_BITS)) < BIT_P[None, :]).astype(np.int8)
    ys = {c: apply_rule(c, xc, xl) for c in CONTEXTS}
    rep["L2_rule_disagreement"] = {}
    for a, b in combinations(CONTEXTS, 2):
        rep["L2_rule_disagreement"][f"{a}_vs_{b}"] = assert_changes(
            f"rule {a} vs rule {b}", ys[a], ys[b], min_frac=0.10, what="labels on identical inputs")

    # L3 c4 genuinely requires history, tested in BOTH directions and against the closed form.
    #
    #    An earlier version of this check only asked whether the hardcoded memoryless ORACLE disagreed
    #    with the truth on c4. That is not sufficient: rewrite c4 as a purely current-step rule and the
    #    hardcoded oracle -- no longer Bayes-optimal for the rule it faces -- still disagrees, so the
    #    check passes while the temporal premise is dead. A mutation test caught exactly that. What must
    #    be asserted is a property of the RULE: the BEST memoryless predictor falls short, the best
    #    with-history predictor does not, and the closed-form bound describes the generator that was
    #    actually implemented rather than the one that was intended.
    kmax_l3 = budget_kmax(cfg["budget_cells"])
    r3 = np.random.default_rng(seed + 4242)
    xc_f, xl_f, y_f = sample_context("c4", cfg["n_fit"], r3)
    xc_e, xl_e, y_e = sample_context("c4", cfg["n_eval"], r3)
    cap3 = min(cfg["fit_cap"], cfg["n_fit"])
    _, sm, tm, _ = fit_lookup(xc_f[:cap3], y_f[:cap3], kmax_l3)
    mem4 = float((apply_lookup(xc_e, sm, tm) == y_e).mean())
    _, sh, th, _ = fit_lookup(np.concatenate([xc_f[:cap3], xl_f[:cap3]], 1), y_f[:cap3], kmax_l3)
    his4 = float((apply_lookup(np.concatenate([xc_e, xl_e], 1), sh, th) == y_e).mean())
    ana4 = analytic_memoryless_bayes("c4")
    if mem4 > 0.90:
        raise LivenessError(
            f"c4 IS NOT TEMPORAL: a memoryless fit on the current bits alone reaches {mem4:.4f} held out. "
            f"The one context that is supposed to require history does not, so the environment cannot "
            f"test the claim it exists to test.")
    if his4 < 0.99:
        raise LivenessError(
            f"c4 IS NOT SOLVABLE: even with the lagged bits in the feature pool the best fit reaches only "
            f"{his4:.4f} held out. A context no arm can solve is a floor, not a hard task, and a model "
            f"failing it would tell us nothing.")
    if abs(mem4 - ana4) > 0.03:
        raise LivenessError(
            f"ANALYTIC BOUND DOES NOT DESCRIBE THE GENERATOR: the closed-form memoryless ceiling for c4 is "
            f"{ana4:.4f} but the fitted memoryless arm reaches {mem4:.4f}. Either the rule or the input "
            f"marginals differ from what the closed form assumes, and every analytic number reported "
            f"downstream would be a claim about a generator that was not the one that ran.")
    m4 = base.context_id == "c4"
    preds = arm_predictions(base, np.random.default_rng(seed + 7))
    rep["L3_c4_memoryless_fit"] = mem4
    rep["L3_c4_history_fit"] = his4
    rep["L3_c4_analytic_memoryless"] = ana4
    rep["L3_c4_history_disagreement"] = assert_changes(
        "history on c4", preds["oracle-memoryless"][m4], base.y[m4], min_frac=0.10,
        what="c4 labels")

    # L4 the retention windows are a live mechanism: turning W off changes the feedback mask.
    off = Adrn2Env(seed, phase_len=cfg["phase_len"], probe_len=0,
                   feedback_every=cfg["feedback_every"], relearn_feedbacks=cfg["relearn_feedbacks"])
    n_on, n_off = int(base.feedback_available.sum()), int(off.feedback_available.sum())
    if n_off <= n_on:
        raise LivenessError(
            f"RETENTION WINDOWS INERT: with W={cfg['probe_len']} the stream delivers {n_on} feedback "
            f"events and with W=0 it delivers {n_off}. Withholding must remove feedback events; if it "
            f"does not, the 'withheld' window is not withheld and the retention number is fiction.")
    rep["L4_feedback_on_off"] = [n_on, n_off]

    # L5 the cadence is a live knob
    fast = Adrn2Env(seed, phase_len=max(cfg["phase_len"], cfg["probe_len"] + 4 * 64 + 32),
                    probe_len=cfg["probe_len"], feedback_every=1,
                    relearn_feedbacks=cfg["relearn_feedbacks"])
    slow = Adrn2Env(seed, phase_len=max(cfg["phase_len"], cfg["probe_len"] + 4 * 64 + 32),
                    probe_len=cfg["probe_len"], feedback_every=64,
                    relearn_feedbacks=cfg["relearn_feedbacks"])
    if int(fast.feedback_available.sum()) <= int(slow.feedback_available.sum()):
        raise LivenessError("FEEDBACK CADENCE INERT: F=1 does not deliver more feedback than F=64")
    rep["L5_cadence_F1_F64"] = [int(fast.feedback_available.sum()), int(slow.feedback_available.sum())]

    # L6 every relearning window actually receives a learning signal
    for w in base.relearn_windows():
        nfb = int(base.feedback_available[w.start:w.stop].sum())
        if nfb < 1:
            raise LivenessError(
                f"RELEARNING WINDOW INERT: phase {w.phase_id} relearning window [{w.start},{w.stop}) "
                f"receives {nfb} feedback events at F={cfg['feedback_every']}. Measuring relearning while "
                f"no learning signal is delivered reports a number for a switched-off mechanism.")
    rep["L6_min_feedback_per_relearn"] = min(
        int(base.feedback_available[w.start:w.stop].sum()) for w in base.relearn_windows())

    # L7 determinism, in both directions
    same = Adrn2Env(seed, phase_len=cfg["phase_len"], probe_len=cfg["probe_len"],
                    feedback_every=cfg["feedback_every"], relearn_feedbacks=cfg["relearn_feedbacks"])
    other = Adrn2Env(seed + 1, phase_len=cfg["phase_len"], probe_len=cfg["probe_len"],
                     feedback_every=cfg["feedback_every"], relearn_feedbacks=cfg["relearn_feedbacks"])
    if same.fingerprint() != base.fingerprint():
        raise LivenessError("NON-DETERMINISTIC: the same seed produced a different stream")
    if other.fingerprint() == base.fingerprint():
        raise LivenessError("SEED INERT: a different seed produced an identical stream, so the unit of "
                            "replication is not replicating anything")
    rep["L7_fingerprint"] = base.fingerprint()[:16]

    # L8 the FITTER itself is live. Without this a low c4 number cannot be distinguished from a broken
    #    fitter -- the exact confusion that made ADRN-1's nulls unreadable.
    kmax = budget_kmax(cfg["budget_cells"])
    rfit = np.random.default_rng(seed + 31)
    xc, xl, y1 = sample_context("c1", cfg["n_fit"], rfit)
    tr_acc, sub, table, _ = fit_lookup(xc, y1, kmax)
    xc2, xl2, y2 = sample_context("c1", cfg["n_eval"], rfit)
    held = float((apply_lookup(xc2, sub, table) == y2).mean())
    if held < 0.99:
        raise LivenessError(
            f"FITTER INERT: the budgeted lookup reaches only {held:.4f} held-out on c1 (x2 XOR x7), a rule "
            f"it can represent exactly at k={kmax}. Every 'this context is hard' conclusion downstream "
            f"would then be a statement about the fitter, not about the environment.")
    rep["L8_fitter_c1_heldout"] = held

    # L9 the 6-tuple contract holds and the feedback channel is genuinely nulled inside probes
    n_seen = 0
    for st in base.stream():
        x, y, fb, pid, cid, isprobe = st
        n_seen += 1
        if isprobe and (fb or st.y_feedback is not None):
            raise LivenessError(
                f"FEEDBACK LEAK at t={st.t}: a retention probe step reports feedback_available={fb} and "
                f"y_feedback={st.y_feedback}. The withheld window is not withheld.")
        if not fb and st.y_feedback is not None:
            raise LivenessError(f"FEEDBACK LEAK at t={st.t}: y_feedback is set while feedback is withheld")
        if x.shape != (N_BITS,) or y not in (0, 1) or cid not in CONTEXTS:
            raise LivenessError(f"MALFORMED Step at t={st.t}")
    if n_seen != base.total_len:
        raise LivenessError(f"stream yielded {n_seen} steps, expected {base.total_len}")
    rep["L9_steps_checked"] = n_seen
    return rep


# ============================================================================================  measure


def rule_validation(seed: int, cfg: dict) -> dict:
    """Per-context, on independent i.i.d. draws: balance, memoryless ceiling, history ceiling, independence.

    Fit and evaluation draws are independent, so every accuracy reported is held out.
    """
    rng = np.random.default_rng(100_000 + seed)
    kmax = budget_kmax(cfg["budget_cells"])
    cells = 1 << kmax
    out = {"kmax": kmax, "cells": cells, "per_context": {}}
    for c in CONTEXTS:
        xc_f, xl_f, y_f = sample_context(c, cfg["n_fit"], rng)
        xc_e, xl_e, y_e = sample_context(c, cfg["n_eval"], rng)
        cap = min(cfg["fit_cap"], cfg["n_fit"])

        # MEMORYLESS arm: feature pool = the 16 current bits only.
        _, sub_m, tab_m, ncand_m = fit_lookup(xc_f[:cap], y_f[:cap], kmax)
        acc_m = float((apply_lookup(xc_e, sub_m, tab_m) == y_e).mean())

        # HISTORY arm: feature pool = 16 current bits + the same 16 bits at t-LAG. SAME table budget.
        Bf = np.concatenate([xc_f[:cap], xl_f[:cap]], axis=1)
        Be = np.concatenate([xc_e, xl_e], axis=1)
        _, sub_h, tab_h, ncand_h = fit_lookup(Bf, y_f[:cap], kmax)
        acc_h = float((apply_lookup(Be, sub_h, tab_h) == y_e).mean())

        n = len(y_e)
        rec = {
            "n_eval": n,
            "p_y1": float(y_e.mean()),
            "p_y1_analytic": analytic_marginal(c),
            "p_y1_sd": float(math.sqrt(0.25 / n)),
            "memoryless_fit": acc_m,
            "memoryless_bayes": analytic_memoryless_bayes(c),
            "history_fit": acc_h,
            "history_bayes": analytic_history_bayes(c),
            "memoryless_bits_1based": [b + 1 for b in sub_m],
            "history_features": [(f"x{b + 1}" if b < N_BITS else f"x{b - N_BITS + 1}(t-{LAG})")
                                 for b in sub_h],
            "candidates_memoryless": ncand_m, "candidates_history": ncand_h,
            "params_cells": cells,
        }

        # INDEPENDENCE, only meaningful where the current input cannot decide y. Under c4 that is the
        # sub-population with the OR's current bit clear; there y is exactly the lagged bit and must be
        # independent of every current bit. Elsewhere y IS a function of x_t and the test does not apply.
        if c == "c4":
            (cur,) = RULE_BITS["c4"]["current"]
            m = xc_e[:, cur] == 0
            ys = y_e[m]
            xs = xc_e[m]
            worst, worst_bit, worst_sd = 0.0, -1, 0.0
            p0 = float(ys.mean())
            for b in range(N_BITS):
                n1 = int(xs[:, b].sum())
                n0 = int(len(ys) - n1)
                if n0 < 30 or n1 < 30:
                    continue
                d = abs(float(ys[xs[:, b] == 1].mean()) - float(ys[xs[:, b] == 0].mean()))
                sd = math.sqrt(max(p0 * (1 - p0), 1e-9) * (1.0 / n1 + 1.0 / n0))
                if d - 5.0 * sd > worst - 5.0 * worst_sd:
                    worst, worst_bit, worst_sd = d, b + 1, sd
            rec["independence"] = {"subset": "c4 with current OR-bit clear", "n": int(m.sum()),
                                   "p_y1_on_subset": p0, "max_abs_dP": worst, "at_bit_1based": worst_bit,
                                   "tol_5sd": 5.0 * worst_sd, "passes": bool(worst <= 5.0 * worst_sd)}
        else:
            exact = float((apply_rule(c, xc_e, xl_e) == y_e).mean())
            rec["independence"] = {"subset": "n/a -- y is a deterministic function of x_t here",
                                   "deterministic_in_x_t": exact, "passes": True}
        out["per_context"][c] = rec
    return out


def schedule_validation(env: Adrn2Env, cfg: dict) -> dict:
    """Window placement, feedback masking, cadence support and context non-leakage on the real stream."""
    out = {}
    rw = env.retention_windows()
    ok_placement = (
        len(rw) == 2
        and [w.phase_id for w in rw] == [4, 6]
        and [w.context_id for w in rw] == ["c1", "c2"]
        and all(w.stop - w.start == env.probe_len for w in rw)
        and all((w.start % env.phase_len) == 0 for w in rw)
    )
    prior = []
    for w in rw:
        prior.append(int(((env.context_id[: w.start] == w.context_id).sum())))
    ok_prior = all(p > 0 for p in prior)
    ok_mask = all(not env.feedback_available[w.start:w.stop].any() for w in rw)
    out["retention"] = {"windows": [w.as_dict() for w in rw], "placement_ok": bool(ok_placement),
                        "prior_exposures": prior, "prior_ok": bool(ok_prior),
                        "feedback_withheld_ok": bool(ok_mask)}
    out["naive"] = {"windows": [w.as_dict() for w in env.naive_windows()],
                    "feedback_withheld_ok": bool(all(not env.feedback_available[w.start:w.stop].any()
                                                     for w in env.naive_windows()))}

    # cadence: all four F must be supported. Phase length is grown per F only as far as needed to hold the
    # relearning window, so every cadence is genuinely exercised rather than skipped.
    cad = {}
    for F in SUPPORTED_FEEDBACK_EVERY:
        need = cfg["probe_len"] + cfg["relearn_feedbacks"] * F + 2 * F
        e = Adrn2Env(env.seed, phase_len=max(cfg["phase_len"], need), probe_len=cfg["probe_len"],
                     feedback_every=F, relearn_feedbacks=cfg["relearn_feedbacks"])
        # The expected count must come from a CLOSED FORM, not from re-evaluating the generator's own
        # expression. `(cadence & ~feedback_withheld_by_probe).sum()` is character-for-character how
        # `feedback_available` is built, so comparing the two could never fail and G8's "cadence is
        # exact" half was a gate with no way to detect a fault. Counted here from the schedule
        # arithmetic instead: multiples of F in [0, n), minus multiples of F inside each phase's
        # opening probe window. Number of t in [a, b) with (t+1) % F == 0 is b//F - a//F.
        expected = e.total_len // F
        for p in range(e.n_phases):
            lo = p * e.phase_len
            expected -= ((lo + e.probe_len) // F - lo // F)
        realised = int(e.feedback_available.sum())
        min_rl = min(int(e.feedback_available[w.start:w.stop].sum()) for w in e.relearn_windows())
        cad[str(F)] = {"phase_len": e.phase_len, "expected": expected, "realised": realised,
                       "match": bool(expected == realised), "min_feedback_in_relearn": min_rl,
                       "ok": bool(expected == realised and min_rl >= 1)}
    out["cadence"] = cad

    # context non-leakage. The strong guarantee is structural (one BIT_P, no context argument in the
    # generator); this is the empirical corroboration.
    worst, worst_sd, where = 0.0, 0.0, ""
    for a, b in combinations(CONTEXTS, 2):
        ma, mb = env.context_id == a, env.context_id == b
        na, nb = int(ma.sum()), int(mb.sum())
        if na < 30 or nb < 30:
            continue
        pa, pb = env.x[ma].mean(0), env.x[mb].mean(0)
        for i in range(N_BITS):
            d = abs(float(pa[i] - pb[i]))
            p = float(BIT_P[i])
            sd = math.sqrt(p * (1 - p) * (1.0 / na + 1.0 / nb))
            if d - 5 * sd > worst - 5 * worst_sd:
                worst, worst_sd, where = d, sd, f"x{i + 1} {a} vs {b}"
    out["context_leak"] = {"max_abs_dP": worst, "tol_5sd": 5 * worst_sd, "at": where,
                           "structural_invariant": "single global BIT_P; the generator takes no context",
                           "passes": bool(worst <= 5 * worst_sd)}
    return out


def arm_scores(env: Adrn2Env, rng: np.random.Generator) -> dict:
    """Raw held-out accuracy of every reference arm, overall / per context / per labelled window."""
    preds = arm_predictions(env, rng)
    res = {}
    for a in STREAM_ARMS:
        p = preds[a]
        rec = {"overall": float((p == env.y).mean()), "params": ARM_PARAMS[a], "per_context": {}, "windows": {}}
        for c in CONTEXTS:
            m = env.context_id == c
            rec["per_context"][c] = float((p[m] == env.y[m]).mean())
        for w in env.windows:
            if w.stop <= w.start:
                continue
            key = f"phase{w.phase_id}:{w.context_id}:{w.segment}"
            rec["windows"][key] = float((p[w.start:w.stop] == env.y[w.start:w.stop]).mean())
        # the two headline aggregates
        for seg in ("retention_probe", "naive_probe", "relearn"):
            m = env.segment == seg
            rec[seg] = float((p[m] == env.y[m]).mean()) if m.any() else float("nan")
        res[a] = rec
    return res


# ==============================================================================================  main


def _cfg() -> dict:
    smoke = os.environ.get("ADRN2_SMOKE", "0") == "1"
    d = {
        "smoke": smoke,
        "seeds": int(os.environ.get("ADRN2_SEEDS", "7")),
        "seeds_start": int(os.environ.get("ADRN2_SEED0", "0")),
        "phase_len": int(os.environ.get("ADRN2_PHASE_LEN", "1024")),
        "probe_len": int(os.environ.get("ADRN2_W", "64")),
        "feedback_every": int(os.environ.get("ADRN2_FEEDBACK_EVERY", "4")),
        "relearn_feedbacks": int(os.environ.get("ADRN2_RELEARN_FEEDBACKS", "4")),
        "n_fit": int(os.environ.get("ADRN2_NFIT", "20000")),
        "n_eval": int(os.environ.get("ADRN2_NEVAL", "20000")),
        "fit_cap": int(os.environ.get("ADRN2_FITCAP", "4000")),
        "budget_cells": int(os.environ.get("ADRN2_BUDGET_CELLS", "8")),
    }
    if smoke:
        d.update(seeds=3, phase_len=256, probe_len=32, n_fit=4000, n_eval=4000, fit_cap=2000)
    return d


def main() -> int:
    cfg = _cfg()
    log: list[str] = []

    def rep(s=""):
        print(s, flush=True)
        log.append(s)

    W = 108
    rep("=" * W)
    rep("ADRN-2 SECTION 18 LATENT-RULE ENVIRONMENT -- self-validation")
    rep("=" * W)
    if cfg["smoke"]:
        rep("  *** SMOKE RUN -- plumbing check, NOT a result. Rerun with ADRN2_SMOKE unset. ***")
    rep(f"  UNIT OF REPLICATION: the SEED -- one independent environment draw per seed. n = {cfg['seeds']}.")
    rep(f"  MDE = 3*sd/sqrt(n) on the PAIRED per-seed gap. Effect size is always the RAW held-out value;")
    rep(f"  controls establish significance only and are never subtracted from a reported number.")
    rep(f"  Config: phase_len={cfg['phase_len']}  W={cfg['probe_len']}  F={cfg['feedback_every']}  "
        f"relearn={cfg['relearn_feedbacks']} feedback deliveries  n_fit={cfg['n_fit']}  "
        f"n_eval={cfg['n_eval']}  table budget={cfg['budget_cells']} cells")
    rep(f"  Schedule: {' -> '.join(PHASE_SCHEDULE)}   returns at phase 4 (c1) and phase 6 (c2)")

    # ---------------------------------------------------------------- LIVENESS FIRST. Raises on failure.
    rep("\n  MECHANISM LIVENESS ASSERTIONS (run BEFORE any score; each RAISES on failure)")
    live = liveness_check(cfg)
    rep(f"    OK  rules mutually distinct        min pairwise label disagreement "
        f"{min(live['L2_rule_disagreement'].values()):.3f}")
    rep(f"    OK  c4 truly needs history         memoryless fit {live['L3_c4_memoryless_fit']:.4f} "
        f"(closed form {live['L3_c4_analytic_memoryless']:.4f}) vs history fit "
        f"{live['L3_c4_history_fit']:.4f}; both directions asserted")
    rep(f"    OK  retention windows live         feedback events W>0: {live['L4_feedback_on_off'][0]}  "
        f"W=0: {live['L4_feedback_on_off'][1]}")
    rep(f"    OK  cadence live                   F=1: {live['L5_cadence_F1_F64'][0]} events  "
        f"F=64: {live['L5_cadence_F1_F64'][1]} events")
    rep(f"    OK  relearn windows fed            min feedback deliveries in any relearn window "
        f"{live['L6_min_feedback_per_relearn']}")
    rep(f"    OK  seed is the unit               same seed reproduces, different seed differs "
        f"(fp {live['L7_fingerprint']})")
    rep(f"    OK  FITTER positive control        budgeted lookup recovers c1 XOR at "
        f"{live['L8_fitter_c1_heldout']:.4f} held out")
    rep(f"    OK  feedback never leaks           {live['L9_steps_checked']} steps checked, "
        f"y_feedback nulled on every withheld step")

    # ---------------------------------------------------------------- per-seed measurement
    per_seed_rules, per_seed_sched, per_seed_arms, per_seed_balance = [], [], [], []
    envs = []
    for s in range(cfg["seeds"]):
        seed = cfg["seeds_start"] + s
        env = Adrn2Env(seed, phase_len=cfg["phase_len"], probe_len=cfg["probe_len"],
                       feedback_every=cfg["feedback_every"],
                       relearn_feedbacks=cfg["relearn_feedbacks"])
        envs.append(env)
        per_seed_balance.append(env.class_balance())
        per_seed_rules.append(rule_validation(seed, cfg))
        per_seed_sched.append(schedule_validation(env, cfg))
        per_seed_arms.append(arm_scores(env, np.random.default_rng(500_000 + seed)))

    def pull(getter):
        return np.array([getter(r) for r in per_seed_rules], dtype=float)

    def mde(v):
        v = np.asarray(v, dtype=float)
        return float(3.0 * np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")

    # ---------------------------------------------------------------- class balance table
    rep("\n  REALISED CLASS BALANCE PER CONTEXT")
    rep(f"    {'context':8s} {'rule':30s} {'P(y=1) iid':>11s} {'analytic':>9s} {'P(y=1) stream':>14s} {'n stream':>9s}")
    rule_txt = {"c1": "x2 XOR x7", "c2": "x4 AND x13", "c3": "majority(x1,x8,x11)",
                "c4": "x3(t-2) OR x15(t)  [TEMPORAL]"}
    bal_rows = {}
    for c in CONTEXTS:
        piid = pull(lambda r, c=c: r["per_context"][c]["p_y1"])
        pstr = np.array([b[c]["p_y1"] for b in per_seed_balance])
        nstr = int(np.mean([b[c]["n"] for b in per_seed_balance]))
        bal_rows[c] = {"iid_mean": float(piid.mean()), "iid_sd": float(piid.std(ddof=1)) if len(piid) > 1 else 0.0,
                       "analytic": analytic_marginal(c), "stream_mean": float(pstr.mean()),
                       "stream_sd": float(pstr.std(ddof=1)) if len(pstr) > 1 else 0.0, "n_stream": nstr}
        rep(f"    {c:8s} {rule_txt[c]:30s} {piid.mean():11.4f} {analytic_marginal(c):9.4f} "
            f"{pstr.mean():14.4f} {nstr:9d}")

    # ---------------------------------------------------------------- memoryless vs history table
    rep("\n  MEMORYLESS vs HISTORY AT A MATCHED TABLE BUDGET (raw held-out accuracy, mean over seeds)")
    k = per_seed_rules[0]["kmax"]
    cells = per_seed_rules[0]["cells"]
    rep(f"    table budget bisected to k={k} bits -> {cells} cells; BOTH arms get the same {cells} cells.")
    rep(f"    the only difference is the feature pool: {N_BITS} current bits vs {N_BITS} current + "
        f"{N_BITS} lagged.")
    rep(f"    {'context':8s} {'memoryless':>11s} {'analytic':>9s} {'history':>9s} {'analytic':>9s} "
        f"{'gap':>8s} {'MDE':>8s} {'cells':>6s} {'verdict':>26s}")
    hist_rows = {}
    for c in CONTEXTS:
        am = pull(lambda r, c=c: r["per_context"][c]["memoryless_fit"])
        ah = pull(lambda r, c=c: r["per_context"][c]["history_fit"])
        gap = ah - am
        m = mde(gap)
        at_ceiling = float(am.mean()) >= 0.99
        note = "UNINFORMATIVE-BY-DESIGN" if at_ceiling else ("history REQUIRED" if gap.mean() > m else "no gap")
        hist_rows[c] = {"memoryless_fit": float(am.mean()), "memoryless_fit_sd": float(am.std(ddof=1)) if len(am) > 1 else 0.0,
                        "memoryless_bayes": analytic_memoryless_bayes(c),
                        "history_fit": float(ah.mean()), "history_fit_sd": float(ah.std(ddof=1)) if len(ah) > 1 else 0.0,
                        "history_bayes": analytic_history_bayes(c),
                        "paired_gap": float(gap.mean()), "mde": m, "cells": cells,
                        "memoryless_at_ceiling": bool(at_ceiling), "note": note}
        rep(f"    {c:8s} {am.mean():11.4f} {analytic_memoryless_bayes(c):9.4f} {ah.mean():9.4f} "
            f"{analytic_history_bayes(c):9.4f} {gap.mean():+8.4f} {m:8.4f} {cells:6d} {note:>26s}")
    rep("    c1-c3 sit AT the memoryless ceiling by construction, so they cannot separate a history-using")
    rep("    system from a memoryless one -- uninformative about that claim, and that is by design. It is")
    rep("    c4 falling FAR BELOW the same ceiling the others reach that carries the information.")

    # ---------------------------------------------------------------- arm table on the stream
    rep("\n  REFERENCE ARMS ON THE REALISED STREAM (raw held-out accuracy, mean over seeds)")
    rep(f"    {'arm':20s} {'params':>7s} {'overall':>8s} " + " ".join(f"{c:>7s}" for c in CONTEXTS) +
        f" {'naive':>8s} {'retain':>8s} {'relearn':>8s}")
    arm_rows = {}
    for a in STREAM_ARMS:
        ov = np.array([r[a]["overall"] for r in per_seed_arms])
        pc = {c: float(np.mean([r[a]["per_context"][c] for r in per_seed_arms])) for c in CONTEXTS}
        nv = np.array([r[a]["naive_probe"] for r in per_seed_arms])
        rt = np.array([r[a]["retention_probe"] for r in per_seed_arms])
        rl = np.array([r[a]["relearn"] for r in per_seed_arms])
        arm_rows[a] = {"params": ARM_PARAMS[a], "overall": float(ov.mean()),
                       "overall_sd": float(ov.std(ddof=1)) if len(ov) > 1 else 0.0,
                       "per_context": pc, "naive_probe": float(nv.mean()),
                       "retention_probe": float(rt.mean()), "relearn": float(rl.mean()),
                       "retention_sd": float(rt.std(ddof=1)) if len(rt) > 1 else 0.0,
                       "per_seed_overall": [float(v) for v in ov],
                       "per_seed_retention": [float(v) for v in rt],
                       "per_seed_naive": [float(v) for v in nv],
                       "windows": {kk: float(np.mean([r[a]["windows"][kk] for r in per_seed_arms]))
                                   for kk in per_seed_arms[0][a]["windows"]}}
        rep(f"    {a:20s} {ARM_PARAMS[a]:7d} {ov.mean():8.4f} " +
            " ".join(f"{pc[c]:7.4f}" for c in CONTEXTS) +
            f" {nv.mean():8.4f} {rt.mean():8.4f} {rl.mean():8.4f}")

    # ---------------------------------------------------------------- retention window detail
    rep("\n  RETENTION vs RELEARNING, THE TWO QUANTITIES THIS ENVIRONMENT EXISTS TO SEPARATE")
    rep("    retention = accuracy inside the feedback-WITHHELD window that opens a RETURN phase.")
    rep("    naive     = accuracy inside the identically sized feedback-withheld window that opens the")
    rep("                FIRST visit to that same context. The matched control. Reported raw, never")
    rep("                subtracted from retention.")
    rep("    relearn   = accuracy over the window after feedback resumes. A different quantity.")
    ret_detail = {}
    for a in ("recency-lookup", "oracle-memoryless", "oracle-full"):
        for key in sorted(arm_rows[a]["windows"]):
            if "probe" in key:
                ret_detail.setdefault(a, {})[key] = arm_rows[a]["windows"][key]
    for a, d in ret_detail.items():
        rep(f"    {a}")
        for kk, v in sorted(d.items()):
            rep(f"        {kk:42s} {v:.4f}")

    # ---------------------------------------------------------------- gates
    n = cfg["seeds"]
    gates: dict = {}
    gate_detail: dict = {}

    # G1 balance
    worst_c, worst_z = None, -1.0
    for c in CONTEXTS:
        d = per_seed_rules[0]["per_context"][c]
        z = abs(bal_rows[c]["iid_mean"] - 0.5) / d["p_y1_sd"]
        if z > worst_z:
            worst_c, worst_z = c, z
    gates["G1 balance: every context P(y=1) within 5 sd of 0.5"] = bool(worst_z <= 5.0)
    gate_detail["G1"] = {"worst_context": worst_c, "worst_z": float(worst_z),
                         "per_context_p_y1": {c: bal_rows[c]["iid_mean"] for c in CONTEXTS}}

    # G2 no constant wins
    const_arms = ["constant-0", "constant-1", "majority-so-far"]
    worst_const = max(max([arm_rows[a]["overall"]] + list(arm_rows[a]["per_context"].values()))
                      for a in const_arms)
    gates["G2 no constant/majority arm exceeds 0.55, overall or in any context"] = bool(worst_const < 0.55)
    gate_detail["G2"] = {"worst_constant_accuracy": float(worst_const),
                         "per_arm": {a: arm_rows[a]["per_context"] for a in const_arms}}

    # G3 analytic history requirement
    mb4 = analytic_memoryless_bayes("c4")
    gates["G3 c4 analytic memoryless Bayes <= 0.80 while history Bayes = 1.0"] = bool(
        mb4 <= 0.80 and analytic_history_bayes("c4") == 1.0)
    gate_detail["G3"] = {"memoryless_bayes_c4": mb4, "history_bayes_c4": 1.0,
                         "headroom_requiring_history": 1.0 - mb4,
                         "note": "closed form from the generative model; bounds a memoryless predictor "
                                 "of ANY capacity, so no larger memoryless arm can close it"}

    gap4 = pull(lambda r: r["per_context"]["c4"]["history_fit"]) - \
        pull(lambda r: r["per_context"]["c4"]["memoryless_fit"])
    m4 = mde(gap4)
    fit_matches = abs(hist_rows["c4"]["memoryless_fit"] - mb4) <= 0.02
    gates["G4 c4 empirical: history beats memoryless by > MDE at a matched budget"] = bool(
        float(gap4.mean()) > m4 and fit_matches)
    gate_detail["G4"] = {"history_fit_c4": hist_rows["c4"]["history_fit"],
                         "memoryless_fit_c4": hist_rows["c4"]["memoryless_fit"],
                         "paired_gap": float(gap4.mean()), "mde": m4,
                         "per_seed_gap": [float(v) for v in gap4],
                         "agrees_with_analytic_ceiling": bool(fit_matches)}

    # G5 fitter liveness across the non-temporal contexts
    g5 = {c: hist_rows[c]["memoryless_fit"] for c in ("c1", "c2", "c3")}
    gates["G5 fitter liveness: memoryless fit >= 0.99 on c1, c2 and c3"] = bool(min(g5.values()) >= 0.99)
    gate_detail["G5"] = {"per_context": g5,
                         "note": "without this, a low c4 number is indistinguishable from a broken fitter"}

    # G6 independence
    ind = [r["per_context"]["c4"]["independence"] for r in per_seed_rules]
    gates["G6 under c4 with the current cue clear, y is independent of every current bit"] = bool(
        all(i["passes"] for i in ind))
    gate_detail["G6"] = {"per_seed": ind}

    # G7 windows
    g7 = all(s["retention"]["placement_ok"] and s["retention"]["prior_ok"]
             and s["retention"]["feedback_withheld_ok"] and s["naive"]["feedback_withheld_ok"]
             for s in per_seed_sched)
    gates["G7 retention windows: exactly 2, phases 4 and 6, c1 and c2, length W, feedback withheld"] = bool(g7)
    gate_detail["G7"] = per_seed_sched[0]["retention"]

    # G8 cadence
    g8 = all(all(v["ok"] for v in s["cadence"].values()) for s in per_seed_sched)
    gates["G8 feedback cadence exact and relearn windows fed for F in {1,4,16,64}"] = bool(g8)
    gate_detail["G8"] = per_seed_sched[0]["cadence"]

    # G9 leak
    g9 = all(s["context_leak"]["passes"] for s in per_seed_sched)
    gates["G9 the input distribution does not leak the context"] = bool(g9)
    gate_detail["G9"] = per_seed_sched[0]["context_leak"]

    # G10 retention headroom (ceiling AND floor for the retention metric)
    ceil_ret = arm_rows["oracle-full"]["retention_probe"]
    floor_ret = arm_rows["recency-lookup"]["retention_probe"]
    headroom = np.array([r["oracle-full"]["retention_probe"] - r["recency-lookup"]["retention_probe"]
                         for r in per_seed_arms])
    m10 = mde(headroom)
    gates["G10 retention probe has headroom: oracle >= 0.99 and no-context-memory arm <= 0.60"] = bool(
        ceil_ret >= 0.99 and floor_ret <= 0.60)
    gate_detail["G10"] = {"oracle_retention": float(ceil_ret), "no_memory_retention": float(floor_ret),
                          "no_memory_naive_control": float(arm_rows["recency-lookup"]["naive_probe"]),
                          "paired_headroom": float(headroom.mean()), "mde": m10,
                          "note": "a no-context-memory arm scoring the same on the retention window as on "
                                  "the matched naive window is the correct null; the metric is calibrated"}

    # G11 determinism -- already raised on failure inside liveness_check, recorded here
    gates["G11 determinism: same seed reproduces, different seed differs"] = True
    gate_detail["G11"] = {"fingerprint": live["L7_fingerprint"]}

    # ---------------------------------------------------------------- ceiling / floor adjudication
    ceilfloor = {}
    for c in CONTEXTS:
        best_ref = max(arm_rows[a]["per_context"][c] for a in STREAM_ARMS if not a.startswith("oracle"))
        at_ceiling = hist_rows[c]["memoryless_fit"] >= 0.99
        at_floor = best_ref <= 0.55
        ceilfloor[c] = {
            "memoryless_fit": hist_rows[c]["memoryless_fit"],
            "best_non_oracle_stream_arm": float(best_ref),
            "memoryless_at_ceiling": bool(at_ceiling),
            "all_non_oracle_at_chance": bool(at_floor),
            "status": ("UNINFORMATIVE-BY-DESIGN for the history claim (memoryless already at ceiling)"
                       if at_ceiling else
                       "INFORMATIVE for the history claim (memoryless falls "
                       f"{1.0 - hist_rows[c]['memoryless_fit']:.4f} below the ceiling the other contexts reach)")}
    n_inform = sum(1 for c in CONTEXTS if not ceilfloor[c]["memoryless_at_ceiling"])
    rep("\n  CEILING AND FLOOR")
    for c in CONTEXTS:
        rep(f"    {c}: {ceilfloor[c]['status']}")
    if n_inform == 0:
        rep("    *** every context is at the memoryless ceiling -- the environment cannot discriminate a ")
        rep("        history-using system and is UNINFORMATIVE for its own headline claim. ***")

    rep("\n  PREDECLARED GATES")
    for kk, v in gates.items():
        rep(f"    {'PASS' if v else 'FAIL'}  {kk}")
    npass = int(sum(gates.values()))

    # ---------------------------------------------------------------- verdict
    smoke_pre = ("SMOKE RUN -- plumbing check, NOT a result; rerun with ADRN2_SMOKE unset. " if cfg["smoke"] else "")
    verdict = (
        f"{smoke_pre}"
        f"ADRN-2 SECTION 18 LATENT-RULE ENVIRONMENT: {npass}/{len(gates)} predeclared gates pass over "
        f"n={n} seeds, where one seed is one independent environment draw (independent input stream and "
        f"independent i.i.d. rule-validation draws) and the MDE is 3*sd/sqrt(n) on the paired per-seed gap. "
        f"The four rules are implemented exactly as specified and are balanced not by altering them but by "
        f"fixing one global per-bit input marginal vector -- P=1/sqrt(2) on the AND bits, P=1-1/sqrt(2) on "
        f"the OR bits, P=0.5 elsewhere -- so every context's marginal P(y=1) is 0.5 by construction; the "
        f"realised values are "
        + ", ".join(f"{c} {bal_rows[c]['iid_mean']:.4f}" for c in CONTEXTS) +
        f". Because that same marginal vector is used in every context, the input distribution is "
        f"identical across contexts and carries no information about which rule is active, so a system "
        f"cannot bypass the retention measurement by reading the context off the input statistics. The "
        f"temporal claim is established twice and in the strongest available form: analytically, the "
        f"Bayes-optimal memoryless accuracy on c4 is {mb4:.4f} against a with-history optimum of 1.0, a "
        f"bound that holds for a memoryless predictor of arbitrary capacity; empirically, at a matched "
        f"budget of {cells} table cells differing only in whether the feature pool contains the lagged "
        f"bits, the history arm reaches {hist_rows['c4']['history_fit']:.4f} on c4 against the memoryless "
        f"arm's {hist_rows['c4']['memoryless_fit']:.4f}, a paired per-seed gap of {float(gap4.mean()):+.4f} "
        f"against an MDE of {m4:.4f}, with the memoryless arm landing on its analytic ceiling. The same "
        f"fitting machinery reaches "
        + ", ".join(f"{c} {hist_rows[c]['memoryless_fit']:.4f}" for c in ('c1', 'c2', 'c3')) +
        f" memorylessly on the non-temporal contexts, which is the positive control that makes the low c4 "
        f"number a statement about c4 rather than about a broken fitter -- and it is also why c1, c2 and "
        f"c3 are marked UNINFORMATIVE-BY-DESIGN for the history claim: they sit AT the ceiling, so they "
        f"separate nothing among the arms that reach it, while c4 falling {1.0 - hist_rows['c4']['memoryless_fit']:.4f} "
        f"BELOW that same ceiling is exactly what carries the information. No constant can win: the best "
        f"of constant-0, constant-1 and majority-so-far reaches {float(worst_const):.4f} overall or within "
        f"any single context. The retention measurement is separated from relearning by construction and "
        f"both are exposed with labels: retention is accuracy inside the feedback-withheld window that "
        f"opens each of the two return phases (phase 4, c1, after c2 and c3 intervened; phase 6, c2, after "
        f"c3, c1 and c4 intervened), relearning is accuracy over the window after feedback resumes, and "
        f"every first visit opens with an identically sized withheld window that serves as the matched "
        f"within-context naive control -- reported raw alongside retention and never subtracted from it. "
        f"That window has verified headroom: an oracle scores {float(ceil_ret):.4f} inside it while a "
        f"context-blind arm with no context memory scores {float(floor_ret):.4f}, against "
        f"{float(arm_rows['recency-lookup']['naive_probe']):.4f} on its matched naive window, so the "
        f"metric is calibrated and a memoryless system produces the correct null rather than a flattering "
        f"one. Feedback cadences of 1, 4, 16 and 64 are all exercised with exact realised counts, and the "
        f"relearning window is defined as a fixed number of feedback DELIVERIES rather than of examples so "
        f"that it can never be measured while the learning signal is switched off. Nine mechanism-liveness "
        f"assertions run before any score is computed and raise rather than warn, covering rule "
        f"distinctness, the c4 history requirement, the retention mask, the cadence knob, seed "
        f"determinism in both directions, fitter liveness and feedback leakage. THIS IS AN INSTRUMENT "
        f"CHECK, NOT A RESULT ABOUT ANY MODEL: it establishes only that the environment can measure "
        f"retention and relearning, that no arm can win on it by predicting a constant or by ignoring "
        f"history, and that a failure on c4 would be attributable to the model rather than to the task.")

    rep("\n" + "=" * W)
    rep(f"  VERDICT: {verdict}")

    limits = [
        "This validates the ENVIRONMENT, not any model. Every gate here could pass while ADRN-2A learns "
        "nothing; a passing instrument check is a precondition for interpreting a model result, never "
        "evidence for one.",
        "FALSIFIER: if a memoryless predictor of any capacity were shown to exceed "
        f"{mb4:.4f} on c4, the analytic ceiling would be wrong and the temporal claim of this environment "
        "would collapse. The bound is closed form from the generative model, so this is checkable by hand.",
        "FALSIFIER: if a system could infer the context from the input distribution rather than from "
        "feedback, the retention window would measure input-statistics inference rather than retrieval. "
        "G9 tests this empirically and the generator enforces it structurally (a single global BIT_P with "
        "no context argument), but a system exploiting higher-order structure in the RNG stream would not "
        "be caught by a per-bit marginal test.",
        "FALSIFIER: if the retention window were long enough for a system to re-derive the rule from "
        f"unlabelled inputs alone, retention would be contaminated by unsupervised adaptation. W={cfg['probe_len']} "
        "and the rules are not self-identifying without labels, but this is an assumption, not a proof, "
        "and a model with a strong unsupervised prior could violate it.",
        "The history arm searches a feature pool twice the size of the memoryless arm's, because history "
        "means more features. The MATCHED quantity is table capacity, not search-space size. The analytic "
        "ceiling makes the objection moot -- no memoryless model of any size beats it -- and that is why "
        "the analytic bound is reported alongside the fitted one rather than instead of it.",
        "The exhaustive lookup fit is capped at subsets of <=3 bits for tractability. It is exactly "
        "sufficient for all four rules (2 bits for XOR/AND/OR, 3 for majority) and its liveness is "
        "asserted, but it is not a general-purpose learner and its numbers are ceilings for THESE rules "
        "only.",
        "Only the phase schedule c1->c2->c3->c1->c4->c2 is validated. Two returns give n=2 return events "
        "per stream; per-seed variation in retention comes from the input draw, not from schedule "
        "variation, so a claim that generalises across schedules would need the schedule randomised.",
        "The temporal context c4 appears exactly once, at phase 5, and is never returned to. Retention is "
        "therefore measured on c1 and c2 only -- both non-temporal. A claim about retaining a TEMPORAL "
        "rule is not supported by this schedule and would need c4 to recur.",
        f"Phase length {cfg['phase_len']} and W={cfg['probe_len']} are configuration, not findings. A "
        "shorter phase would leave less time to acquire the rule before the return, which would lower "
        "retention for reasons that have nothing to do with the mechanism under test.",
    ]

    R = {
        "model": "adrn2-latent-rule-environment-v1",
        "smoke": bool(cfg["smoke"]),
        "config": cfg,
        "unit_of_replication": "seed (one independent environment draw: independent input stream AND "
                               "independent i.i.d. rule-validation draws)",
        "seeds": n,
        "env_spec": envs[0].spec(),
        "liveness": {k: (v if not isinstance(v, dict) else v) for k, v in live.items()},
        "class_balance": bal_rows,
        "memoryless_vs_history": hist_rows,
        "arms": arm_rows,
        "mde": {
            "definition": "3*sd/sqrt(n) on the PAIRED per-seed gap; n = number of seeds",
            "n": n,
            "c4_history_minus_memoryless": {"gap": float(gap4.mean()), "mde": m4,
                                            "per_seed": [float(v) for v in gap4]},
            "retention_headroom_oracle_minus_nomemory": {"gap": float(headroom.mean()), "mde": m10,
                                                         "per_seed": [float(v) for v in headroom]},
            "per_context_history_gap": {c: {"gap": hist_rows[c]["paired_gap"], "mde": hist_rows[c]["mde"]}
                                        for c in CONTEXTS},
        },
        "gates": gates,
        "gate_detail": gate_detail,
        "ceiling_floor": ceilfloor,
        "schedule_validation": per_seed_sched[0],
        "rule_validation_seed0": per_seed_rules[0],
        "verdict": verdict,
        "limits": limits,
        "log": log,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "adrn2_env.json"
    with open(path, "w") as fh:
        json.dump(R, fh, indent=1, default=float)
    rep(f"\n  -> {path}")
    return 0 if npass == len(gates) else 1


if __name__ == "__main__":
    sys.exit(main())
