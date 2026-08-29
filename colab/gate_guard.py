"""GUARDS FOR THE TWO WAYS A GATE CAN FIRE WHILE MEASURING NOTHING.

TWELVE TIMES IN ONE SESSION. Every one of these passed or failed on a number that could not have
carried information, and the record says so in each case:

    loop 76 G5    a constant-ACTIVATING sham reached 79% of the real effect, because the null
                  shuffled signs inside an arm that had only one sign
    loop 77 V2    compared the 1 s map against the 1 s map and got 1.0000 by construction
    loop 81 C3    "long moved more than short" -- true, and meaningless, when both had collapsed
    loop 82 D3    returned exactly one bin, 25.0 kb, for every kappa including zero
    loops 77-83   the re-simulation orientation shuffle, which rebuilds the landscape and rescores
                  with the permuted labels, so those labels are that run's TRUE labels
    loop 86 G5    my finite-size prediction, refuted -- reported, not vacuous, but the same family
    loop 87 C3    compared against a distance-only null whose observed/expected is 1.0 everywhere,
                  giving zero variance and a nan Spearman; the gate evaluated `x > nan`
    loop 87 C6    "54% survival" from -0.0025 against -0.0046, two numbers that are both zero
    loop 87b B6   "6718% survival" from -0.0073 against -0.0001, same fault, next module
    loop 92 S1    262% at every proteome size, because the constant cancels -- caught, not missed
    loop 93 K2    gated on a median when the error lived in a handful of high-flux reactions
    loop 94 N4    a "degree-preserving rewiring" that permutes the target column, preserving every
                  in-degree EXACTLY, so an in-degree statistic is invariant under it by
                  construction: +0.1438 +/- 0.0000 against a real +0.1438

They fall into exactly two families, and each has a mechanical guard.

FAMILY ONE: THE RATIO WITH NO DENOMINATOR. "What fraction of the effect survives the null" is
undefined when there is no effect. Loops 87 C6 and 87b B6 both divided one indistinguishable-from-
zero number by another and reported 54% and 6718%. The guard is not a bigger epsilon -- it is to
ask first whether the real value is distinguishable from the null AT ALL, using the null's own
spread, and to return UNDEFINED rather than a number when it is not. A survival fraction is only
meaningful once there is something to survive.

FAMILY TWO: THE NULL THAT CANNOT MOVE. A control is evidence only if it could have changed the
statistic. Loop 94's rewiring could not, arithmetically. Loop 77's V2 could not, by construction.
The guard is to compute the statistic's input under the null and confirm it actually differs from
the real input before the null's output is allowed to count.

Neither guard needs judgement. Both are cheap. Neither existed.

AND THEN THIS MODULE COMMITTED FAMILY TWO ITSELF, which is recorded here rather than quietly fixed.
`null_can_move` compared the fraction of changed entries against a fixed 0.5. Under a permutation
that fraction is bounded by 1 - sum(p_k^2), which for any BINARY label vector is 2p(1-p) <= 0.5 --
so the check declared every binary permutation null INERT regardless of how much spread that null
actually had. Loop 119 caught it: an 11%-prevalence label vector can change at most 19.6% of its
entries, the guard said INERT, and the condemned null was giving real 88 against 50.1 +/- 5.1,
z = +7.4. A gate that cannot pass is the same fault as a null that cannot move, and this module
had one in it for fourteen loops. The bar is now a fraction of what the input can actually reach.
"""
import numpy as np

UNDEFINED = "UNDEFINED"


def survival(real, nulls, z_min=2.0):
    """What fraction of `real` survives under `nulls` -- or UNDEFINED if there is no effect.

    Returns a dict, never a bare float, so a caller cannot accidentally format UNDEFINED as a
    percentage. `defined` is False when |real - mean(nulls)| is smaller than z_min null standard
    deviations: at that point the real value and the null are the same number and their ratio is
    a property of the noise, not of the model.
    """
    n = np.asarray([x for x in np.atleast_1d(nulls) if np.isfinite(x)], float)
    out = {"real": float(real), "n_null": int(len(n)),
           "null_mean": float(n.mean()) if len(n) else float("nan"),
           "null_sd": float(n.std()) if len(n) else float("nan")}
    if not np.isfinite(real) or len(n) < 2 or out["null_sd"] <= 0:
        out.update({"defined": False, "reason": "no finite real value or no null spread",
                    "z": float("nan"), "fraction": UNDEFINED})
        return out
    z = (real - out["null_mean"]) / out["null_sd"]
    out["z"] = float(z)
    if abs(z) < z_min:
        out.update({"defined": False, "fraction": UNDEFINED,
                    "reason": f"|z| {abs(z):.2f} < {z_min}: the real value is not distinguishable "
                              f"from the null, so a survival fraction has no denominator"})
        return out
    frac = float(out["null_mean"] / real)
    # THE THIRD FAILURE MODE, FOUND BY THIS MODULE'S OWN V2 GATE. Loop 94's N4 has z = -2.5, so the
    # real value IS distinguishable from the null -- but in the WRONG DIRECTION: the null (+0.1148)
    # is larger than the real value (+0.0983), and "117% survival" is not survival, it is the null
    # outperforming the thing it was meant to destroy. A fraction at or above 1 means there is no
    # effect in the claimed direction, and reporting it as a percentage invites exactly the reading
    # loop 94 gave it. Kept separate from UNDEFINED because the two say different things: one is
    # "no signal", the other is "signal, pointing the other way".
    if frac >= 1.0:
        out.update({"defined": False, "fraction": UNDEFINED, "raw_fraction": frac,
                    "reason": f"the null ({out['null_mean']:+.4f}) equals or exceeds the real value "
                              f"({real:+.4f}): there is no effect in the claimed direction, so "
                              f"'survival' has no meaning"})
        return out
    out.update({"defined": True, "fraction": frac,
                "reason": "effect is distinguishable from the null and exceeds it"})
    return out


def achievable_change(x):
    """The largest fraction of entries a PERMUTATION of x can be expected to change.

    A permutation cannot change an entry that lands on an equal value, so with class frequencies
    p_k the expected change fraction is exactly 1 - sum(p_k^2). For an all-distinct vector this is
    1 - 1/n, effectively 1. For a BINARY vector it is 2p(1-p), which is at most 0.5 and reaches it
    only at a perfect 50/50 split.
    """
    v, c = np.unique(np.asarray(x, dtype=object).ravel(), return_counts=True)
    p = c / c.sum()
    return float(1.0 - (p ** 2).sum())


def null_can_move(real_input, null_input, min_changed=None, min_frac=0.5):
    """Did the null actually change the statistic's INPUT? Loop 94's rewiring did not.

    Called with the feature vector (or label list) as the statistic sees it, before and after the
    null is applied. Returns the fraction of entries that changed and whether that clears the bar.

    THE BAR IS RELATIVE, AND IT WAS NOT, AND THAT WAS THIS MODULE COMMITTING ITS OWN FAMILY-TWO
    ERROR. The first version compared `changed` against a fixed 0.5. But `changed` under a
    permutation is bounded above by 1 - sum(p_k^2), which for ANY binary label vector is 2p(1-p)
    <= 0.5, with equality only at an exact 50/50 split. So the check declared EVERY binary
    permutation null INERT, for every dataset, no matter how much spread the null actually had --
    a gate arithmetically incapable of passing, which is precisely the fault this module exists to
    catch, in mirror image. Loop 119 hit it: an 11%-prevalence label vector maxes out at 19.6%
    changed, the check said INERT, and the null it condemned was producing 88 real against
    50.1 +/- 5.1, z = +7.4. The null was fine. The guard was broken.

    So the comparison is now against what a permutation of THIS input can reach. `min_frac` of the
    achievable change is required, and the achievable change must itself be non-trivial -- a vector
    that is 99.9% one value cannot be meaningfully permuted at all, and that is worth saying rather
    than passing. For continuous or all-distinct inputs achievable is ~1 and the bar is ~0.5, i.e.
    unchanged from the original behaviour on every caller that was not binary.

    Loop 94's case is unaffected and still caught: its rewiring left the in-degree vector EXACTLY
    invariant, changed = 0.0, which is 0% of anything achievable.

    Pass `min_changed` to pin an absolute threshold and restore the old semantics explicitly.
    """
    a = np.asarray(real_input, dtype=object)
    b = np.asarray(null_input, dtype=object)
    if a.shape != b.shape:
        return {"capable": True, "changed": 1.0, "achievable": 1.0,
                "reason": "shapes differ, so the null rebuilt the input"}
    changed = float(np.mean(a != b))
    if min_changed is not None:
        return {"capable": bool(changed >= min_changed), "changed": changed,
                "achievable": achievable_change(a), "threshold": float(min_changed),
                "reason": ("the null changes the statistic's input"
                           if changed >= min_changed else
                           f"only {changed:.1%} of entries change against an absolute bar of "
                           f"{min_changed:.1%} -- this null is INERT and its output is not evidence")}
    ach = achievable_change(a)
    bar = min_frac * ach
    if ach < 0.02:
        return {"capable": False, "changed": changed, "achievable": ach, "threshold": bar,
                "reason": f"the input is almost constant (a permutation could change at most "
                          f"{ach:.1%} of it), so no permutation null is informative about it"}
    return {"capable": bool(changed >= bar), "changed": changed, "achievable": ach,
            "threshold": bar,
            "reason": ("the null changes the statistic's input"
                       if changed >= bar else
                       f"only {changed:.1%} of entries change against {ach:.1%} achievable "
                       f"({changed / ach:.0%} of the reachable move) -- this null is INERT and "
                       f"its output is not evidence")}


def report(name, s, emit=print):
    """Print a survival result so that UNDEFINED can never be read as a percentage."""
    if s.get("defined"):
        emit(f"     {name}: real {s['real']:+.4f}  null {s['null_mean']:+.4f} "
             f"+/- {s['null_sd']:.4f}  z {s['z']:+.1f}  survives {s['fraction']:.0%}")
    else:
        emit(f"     {name}: real {s['real']:+.4f}  null {s['null_mean']:+.4f} "
             f"+/- {s['null_sd']:.4f}  survival UNDEFINED -- {s['reason']}")
    return s


def verdict(gate, if_true, if_false, emit=print, indent="     "):
    """Emit a conclusion that CANNOT contradict its own gate.

    Added after the same defect appeared three times in the loop 148-150 arc: a say()
    line asserting a conclusion was written as a literal and printed unconditionally,
    so it stayed on screen after the gate beneath it failed. Loop 149's M5 printed
    "processivity is a real requirement of this mechanism" while its own sweep showed
    the amplification flat; loop 150's R3 printed "it does not depend on q" while its
    own gate was FAILING on 12 of 21 settings. Both were caught by re-reading output
    rather than by anything structural, which is luck.

    Narration that states a result must take the gate as an argument. Passing a single
    string for both branches is allowed and is the honest way to say something that
    holds either way.

        RETROFIT, and it protects every existing caller without touching one of them. The messages
    were interpolated directly into the emit() call, so BOTH f-strings were built by Python before
    verdict() was even entered -- which is how loop 196's X4 died on d4[None] and loop 197's Y4 on
    an empty qual[0], each after its gate had already decided FAIL correctly. Routing through
    _render does two things: a message may now be a CALLABLE, which is not evaluated unless its
    branch is chosen, and a message that raises while rendering is reported rather than fatal. By
    the time we are here the verdict is decided and the only remaining job is to say it, so
    narration must not be able to kill the run.

    A caller whose success message references a success-only value should pass a lambda. One that
    passes an eager f-string is still exposed to the interpolation happening at ITS call site --
    Python builds it before the call and nothing here can intercept that -- which is why
    lint_gates.py flags those separately.
    """
    emit(f"{indent}{_render(if_true if gate else if_false, 'PASS' if gate else 'FAIL')}")
    return bool(gate)


# =============================================================================================
# THE GATE LEDGER, added after four defects in one arc that were each patched in one loop and
# then written again in the next. Every one of them was in the MACHINERY around a gate rather
# than in the science it was testing, and every one put a false sentence into a log.
#
#   A  NO VOID. Loop 187's B6 required a self-loop count above chance in a network with zero
#      self-loops, so z was 0/0. PASS/FAIL cannot express "the test did not apply", and forcing
#      the nan into FAIL printed "loop 175's framing stands" -- a claim the run had not earned.
#      Every loop since has hand-rolled a `void` set, and loop 196's X4 hand-rolled it wrongly:
#      it printed "X4 VOID" and then fell through to a verdict that printed "X4 FAIL", putting
#      both in the same summary.
#
#   B  EAGER NARRATION. verdict() takes if_true and if_false as strings, and Python builds BOTH
#      before the call. Loop 196's X4 crashed on d4[winner] with winner None; loop 197's Y4
#      crashed on qual[0] with qual empty, ONE LOOP after the first was fixed. In both cases the
#      gate had already decided FAIL correctly and died while saying so. Narration must never be
#      able to kill a decided verdict.
#
#   C  DOWNSTREAM GATES THAT DO NOT KNOW. Loop 194's V4 and V6 confirm a positive V3 did not
#      find. V4 failed with "the result depends on the hub threshold" when the negative was
#      identical at every threshold; V6 failed with "the coherence goes" when there was no
#      coherence to lose. Both false about their own numbers, because neither could see that its
#      precondition had failed.
#
#   D  A GATE THAT ASSUMES ITS OWN SIGN. Loop 199's Q5 asked "does swapping destroy the
#      association" and implemented it as real > swapped, which tests that only if the
#      association is positive. Q4 came back negative, so a swap that removed 82% of the effect
#      scored as a failure.
#
#   E  A PRECONDITION THAT IS A STRING. `requires` defaults to a tuple and is consumed by
#      iterating it, so requires="Z2" iterates the CHARACTERS 'Z' and '2'. Neither is a
#      registered gate, so `missing` is always non-empty and EVERY gate declaring a single
#      string precondition is VOID no matter what that precondition did. Loop 240 hit it in the
#      loudest possible way: Z2 PASSED at +0.3910 (+67.7 se) and all six gates that depended on
#      it printed "VOID -- Z, 2 did not pass". The comma in that message is the bug printing
#      itself, and it had already printed the same comma in loop 239 without being read.
#      Six measured comparisons were computed, logged, and then discarded by a type error.
#      Two fixes, because the first alone would leave the second failure silent: a string
#      requirement is now wrapped rather than iterated, AND a requirement naming a gate that was
#      never registered is a DECLARATION ERROR with its own message, not an ordinary
#      precondition failure -- otherwise a typo voids a gate while looking like a result.
#
#   G  A MESSAGE THAT ASSUMES THE SIGN OF ITS OWN STATISTIC. Defect D in the narration rather
#      than the logic. Loop 256's H4 asked whether the line tower does anything and printed
#      "zeroing the line embedding costs -0.0543". The verb says the ablation hurt; the sign
#      says it helped. Both in one sentence, and the sentence closed with "the cell line is
#      still decoration" -- which UNDERSTATES the measurement, because the line tower was not
#      inert at 0.3001 against an ablated 0.3544, it was actively harmful. A FAIL branch is
#      exactly where the statistic is least constrained in sign, so a directional verb there is
#      a coin flip. Sign-neutral phrasing ("is worth {d:+.4f}") states the same fact without
#      betting on it, and where direction matters the branch must read the sign, not assume it.
#
#   H  AN ABLATION ARM THAT IS NOT MATCHED ON TRAINABILITY. Loop 257's I3 was declared the
#      load-bearing gate: K=8 context-gated against a K=1 "context-blind" operator. It printed
#      "gating adds -0.0152; the correction is a better GENE model, not a context model". The
#      K=1 arm had never trained. Its validation MSE sat at 0.254322 against 0.254313 at
#      INITIALISATION, and its correction norm was 0.0218 against a target of 16.42 -- a factor
#      of 750 short of the thing it was meant to predict -- while the K=8 arm cut validation MSE
#      by 19% over 30 epochs. Deleting seven of eight experts also deletes ~8x of the effective
#      output scale and early gradient signal, so with PATIENCE=7 the small arm stopped at epoch
#      3 before it had started. The gate measured TRAINED against BARELY-TRAINED, not GATED
#      against BLIND, and then named a cause ("a better GENE model") that its own numbers cannot
#      support: an arm producing no correction at all is not a better model of anything.
#      A matched ablation holds parameter count and optimisation dynamics fixed and removes ONLY
#      the information under test -- here, feed the hypernetwork a CONSTANT instead of the line
#      features, so the gates still exist, still train, and simply cannot vary with context.
#      The general rule, now enforced by reporting: an ablation arm must show that it LEARNED
#      before its score is allowed to mean anything. Every arm now prints its validation MSE at
#      initialisation and at its best, and a gate whose control did not move is VOID, not FAIL.
#
#   I  A CONTROL THAT MOVES MORE THAN ONE THING. The wrong-line control in loops 256-259 bound
#      one variable `c` to the substitute line and then consumed it TWICE: once for the model's
#      context input, which was intended, and once for the additive baseline's line-mean term,
#      which was not. So the arm changed both what the model was told AND what the standing
#      answer was, while printing "hypernetwork fed another line's properties" -- true of half
#      of what it did. A degraded baseline leaves more correlation headroom for a fixed-alpha
#      correction, so the measured margin ratio inflates systematically, by roughly 15-20% in a
#      toy calibrated to this project's own numbers, and that ratio is then compared against a
#      hard 25% bar. The bias happened to be conservative here -- it makes a positive look more
#      like memorisation -- but a control biased in a knowable direction is not a control.
#      colab/lincs_harness.py had it right all along: evaluate(shuffle_line=True) substitutes
#      the line only inside features(), while residuals(D, hold) keeps the true held-out line.
#      The four loop files departed from the harness's own convention without saying so.
#      RULE: a control names exactly one thing it destroys, and everything else in that arm must
#      be byte-identical to the arm it is compared against. When one variable feeds two consumers,
#      split it and let only the consumer under test see the substitute.
#
#   J  AN ORACLE FITTED AND SCORED ON THE SAME ROWS. Loop 262 run 1 gated N1 on a 978x978
#      operator fitted on the held-out line's rows and then scored on those same rows, and
#      reported the off-diagonal as worth +0.1138 beyond the diagonal. Refitting on half of
#      each line's GENES and scoring on the other half -- lambda chosen inside the fitting
#      half -- put it at +0.0435. Half the headline was the fit reading back its own
#      training data. The diagonal, with 978 parameters instead of 956,484, lost almost
#      nothing on the same test (+0.0247 -> +0.0239), which is exactly why an oracle's
#      trustworthiness scales with how few parameters it has and must never be assumed.
#      A "cheating" arm is legitimate as a CEILING only when it is cheating in one specific
#      way -- seeing the held-out answers -- and not also in the ordinary way of being
#      scored on its own fitting rows. Those are different sins and only the first is
#      informative. Every oracle now needs a held-out split inside it.
#      The same run also mis-stated the fit's difficulty: it divided 956,484 parameters by
#      ~3,700 ROWS to claim 250:1 underdetermined, when each COLUMN of W has 978 parameters
#      against 3,700 observations of that landmark -- 3.8:1 OVERdetermined. Parameters are
#      counted against OBSERVATIONS, and a row of a multi-output regression is 978 of them.
#
#   K  A RANKING TEST WHOSE ITEMS DIFFER IN SIZE. REM's Z7 asked whether ensemble
#      (-RT ln Z) rescoring reorders anything relative to single-pose (min E) scoring, and
#      implemented it by ranking five repacking problems of 3, 4, 5, 6 and 7 residues.
#      Adding a residue adds a large negative unary term, so BOTH orderings are just the
#      residue count sorted descending. The test reported "identical ordering" and could
#      not have reported anything else: the answer was fixed by the construction, not
#      measured. It is defect H's shape moved from ablation to ranking -- the arms were not
#      matched on the thing that dominates the statistic.
#      A ranking test compares ALTERNATIVES TO ONE ANOTHER: same variables, same rotamers,
#      same residue count, differing only in the pose being scored. Then a reordering is
#      information. RULE: before reporting that two scores agree or disagree on a ranking,
#      check that the items being ranked are exchangeable under the null. If one item is
#      bigger than another in a way that swamps the effect, the comparison is decorative.
#
#   L  AN ABSOLUTE BAR ON AN ASYMPTOTIC LIMIT. REM's Z4 predeclared "as T -> infinity,
#      ln Z -> ln(n_configs); GATE |ln Z - ln N| < 1e-3 at T = 1e6 K" and measured 3.87e-2,
#      a factor of 39 over the bar. Nothing was wrong with the code. The residual of that
#      limit is exactly -<E>/RT, and <E> for this instance is 77.5 kcal/mol, so at 1e6 K the
#      residual CANNOT be smaller than 0.039 -- the bar was unreachable the moment it was
#      written, and it was written before <E> was known. Measured across five decades the
#      residual tracked -<E>/RT to four significant figures and the log-log slope came out
#      -0.9897 against a theoretical -1.
#      This is loop 267's T4 again in a different domain: an absolute bar fixed before the
#      statistic's scale was known. The verdict STANDS -- Z4 was predeclared and it failed,
#      and the failure is recorded rather than the bar moved. The repair is a SEPARATELY
#      declared gate that tests what the limit actually asserts. "x -> L as T -> infinity"
#      is a statement about a RATE, so the gate is on the rate: the residual must fall like
#      1/T across decades. That is strictly stronger than the one-point bar it replaces --
#      a code path that converged to the wrong constant would pass a loose point bar and
#      fail the slope.
#      RULE: a limit is a rate. Gate it across a sweep, not at one point; and any absolute
#      tolerance on a physical quantity must be derived from that quantity's own scale,
#      measured, before it is written down.
#
# A commit message is not a mechanism. This is.
# =============================================================================================

PASS, FAIL, VOID = "PASS", "FAIL", "VOID"


def _render(msg, fallback):
    """Render a gate message, and NEVER let rendering kill a decided verdict.

    Messages may be callables, which is the fix for defect B: a lambda is not evaluated unless
    its branch is chosen, so a success message may safely reference a success-only value. Plain
    strings still work and are still built eagerly by Python at the call site -- the try/except
    is what stops that from being fatal, since by the time we are here the gate has ALREADY been
    decided and the only thing left to do is say so."""
    try:
        return msg() if callable(msg) else str(msg)
    except Exception as exc:                                          # noqa: BLE001
        return f"{fallback} [message could not be rendered: {type(exc).__name__}: {exc}]"


def is_defined(x):
    """True unless x is a non-finite number. Defect A's trigger, as a function."""
    try:
        return bool(np.isfinite(x))
    except (TypeError, ValueError):
        return x is not None


class Gates:
    """A ledger where VOID is a first-class outcome and preconditions are declared, not assumed.

    Usage mirrors verdict() but the gate name and its dependencies are part of the call:

        G = Gates(emit=say)
        G.add("Q3", diff > 0, if_true=..., if_false=...)
        G.add("Q4", ok, requires=("Q3",), if_true=lambda: f"...{best['x']}...", if_false=...)
        G.summary()

    A gate whose `requires` are not all PASS is VOID and says which precondition failed, so a
    confirmatory gate can never report a finding about a result that does not exist. A gate whose
    verdict value is a non-finite number is VOID rather than FAIL. VOID gates are excluded from
    the score rather than counted against it, because a test that did not apply is not a test
    that was failed."""

    def __init__(self, emit=print, indent="     "):
        self.emit, self.indent = emit, indent
        self.status, self.why = {}, {}

    def add(self, name, ok=None, *, stat=None, if_true="", if_false="", if_void=None,
            requires=(), void_if=False, void_reason=""):
        """`stat` is the STATISTIC the gate compared, and passing it is what catches defect A.

        A comparison against nan silently yields False -- `float("nan") > 3.0` is False, not nan --
        so by the time the verdict is a boolean the undefinedness is gone and the gate scores FAIL.
        That is exactly how loop 187's B6 turned a 0/0 z-score into "loop 175's framing stands".
        Checking `ok` alone cannot see it; the raw statistic can. Pass stat=z whenever the verdict
        is a threshold comparison on a number that could be undefined."""
        if isinstance(requires, str):
            requires = (requires,)                       # defect E: a string iterates as chars
        requires = tuple(requires)
        unknown = [r for r in requires if r not in self.status]
        missing = [r for r in requires if r in self.status and self.status[r] != PASS]
        if unknown:
            st, txt = VOID, (f"{name} VOID -- DECLARATION ERROR: precondition(s) "
                             f"{', '.join(repr(u) for u in unknown)} were never registered as "
                             f"gates. This is a bug in the loop, not a result about the data; "
                             f"registered so far: {', '.join(self.status) or '(none)'}")
        elif missing:
            st, txt = VOID, (f"{name} VOID -- {', '.join(missing)} did not pass, so there is "
                             f"nothing here to test")
        elif (void_if() if callable(void_if) else bool(void_if)):
            st, txt = VOID, f"{name} VOID -- {void_reason or 'the test did not apply'}"
        elif stat is not None and not is_defined(stat):
            st, txt = VOID, (f"{name} VOID -- the statistic is undefined ({stat!r}), so this gate "
                             f"could not pass or fail; that is not the same as failing")
        elif not is_defined(ok):
            st, txt = VOID, (f"{name} VOID -- the verdict is undefined ({ok!r}), so this gate "
                             f"could not pass or fail; that is not the same as failing")
        else:
            st = PASS if ok else FAIL
            txt = _render(if_true if ok else if_false, f"{name} {st}")
        if st == VOID and if_void is not None:
            txt = _render(if_void, txt)
        self.status[name], self.why[name] = st, txt
        self.emit(f"{self.indent}{txt}")
        return st == PASS

    def voided(self, name):
        return self.status.get(name) == VOID

    def as_dict(self):
        """gates/void in the shape every loop in this project already writes to JSON."""
        return ({k: (v == PASS) for k, v in self.status.items()},
                sorted(k for k, v in self.status.items() if v == VOID))

    def summary(self, seconds=None):
        self.emit("=" * 104)
        for k, v in self.status.items():
            self.emit(f"  {k}  {v}")
        scored = [k for k, v in self.status.items() if v != VOID]
        n_void = len(self.status) - len(scored)
        line = (f"  {sum(1 for k in scored if self.status[k] == PASS)}/{len(scored)}"
                + (f"   [{seconds:.0f}s]" if seconds is not None else ""))
        if n_void:
            line += (f"   ({n_void} VOID: "
                     f"{', '.join(sorted(k for k, v in self.status.items() if v == VOID))})")
        self.emit(line)
        self.emit("=" * 104)
        return self.status


def weakened_by(real, control, min_ratio=1.0):
    """Did a control weaken the association, WITHOUT assuming which way the association points?

    Defect D, as a function. Loop 199's Q5 asked exactly this and wrote `real > control`, which
    silently assumes a positive association; the answer came back negative and a control that
    removed 82% of the effect was scored as a failure. Magnitude is what "weakened" means."""
    a, b = abs(float(real)), abs(float(control))
    return dict(real=float(real), control=float(control), ratio=(b / a) if a > 0 else float("nan"),
                weakened=bool(a > b * min_ratio))


def finite(*arrays, report=None, label="comparison"):
    """Drop positions where ANY input is non-finite, and COUNT what went.

    Defect A's quieter cousin, and the one that cost a real result. Loop 188's G2 tested three
    signed predictions with np.median and scipy; 90 of 4,482 elements had no measured CpG, so
    5mC's median was nan, its p-value was nan, and the loop printed REFUTED for a test that never
    ran. Loop 188b re-tested it properly: medians 4.00 against 9.56, p 3.55e-16, HOLDS -- one of
    the cleanest directional results in that arc, hidden by two percent missingness.

    np.median, np.mean, np.std, np.corrcoef and the scipy tests all propagate a single nan and
    none of them warns. A coverage gate does not protect them: 98% defined is fine for a model
    that imputes and fatal for a median, which has no threshold at all."""
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    keep = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        keep &= np.isfinite(a)
    dropped = int((~keep).sum())
    if report is not None and dropped:
        report(f"     {label}: dropped {dropped} non-finite of {len(keep)}")
    return tuple(a[keep] for a in arrs) + (dropped,)
