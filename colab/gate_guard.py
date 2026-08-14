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
    out.update({"defined": True, "fraction": float(out["null_mean"] / real),
                "reason": "effect is distinguishable from the null"})
    return out


def null_can_move(real_input, null_input, min_changed=0.5):
    """Did the null actually change the statistic's INPUT? Loop 94's rewiring did not.

    Called with the feature vector (or label list) as the statistic sees it, before and after the
    null is applied. Returns the fraction of entries that changed and whether that clears
    min_changed. A null below the threshold is inert and its output is not evidence about anything.
    """
    a = np.asarray(real_input, dtype=object)
    b = np.asarray(null_input, dtype=object)
    if a.shape != b.shape:
        return {"capable": True, "changed": 1.0, "reason": "shapes differ, so the null rebuilt the input"}
    changed = float(np.mean(a != b))
    return {"capable": bool(changed >= min_changed), "changed": changed,
            "reason": ("the null changes the statistic's input"
                       if changed >= min_changed else
                       f"only {changed:.1%} of entries change -- this null is INERT and its "
                       f"output is not evidence")}


def report(name, s, emit=print):
    """Print a survival result so that UNDEFINED can never be read as a percentage."""
    if s.get("defined"):
        emit(f"     {name}: real {s['real']:+.4f}  null {s['null_mean']:+.4f} "
             f"+/- {s['null_sd']:.4f}  z {s['z']:+.1f}  survives {s['fraction']:.0%}")
    else:
        emit(f"     {name}: real {s['real']:+.4f}  null {s['null_mean']:+.4f} "
             f"+/- {s['null_sd']:.4f}  survival UNDEFINED -- {s['reason']}")
    return s
