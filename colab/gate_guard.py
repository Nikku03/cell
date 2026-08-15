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
    """
    emit(f"{indent}{if_true if gate else if_false}")
    return bool(gate)
