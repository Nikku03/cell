"""THE DOCTRINE -- the rules this toolchain runs on, each one with a check that can fail.

WHY THIS FILE EXISTS.  Every rule below was paid for. None is a preference, an aesthetic, or something
read in a paper: each one is the generalisation of a specific defect that produced a specific wrong number
in this repo, and the comment on each says which. A doctrine assembled that way is worth writing down. A
doctrine assembled from taste is not.

THE PROBLEM WITH WRITING DOWN A PHILOSOPHY.  Stated principles are free. Anyone can write "report negative
results prominently" and then not do it, and nothing anywhere will object. So every rule here that CAN be
checked mechanically is checked, against the toolchain as it actually stands, and a rule the toolchain
violates is reported as a violation rather than quietly reworded until it passes.

THE PROBLEM WITH A SELF-AUDIT.  A suite that always passes is indistinguishable from a suite that tests
nothing, and this one is written by the same hand as the thing it audits -- so of course it passes. That
is not evidence. Every check is therefore paired with a MUTATION: a deliberate reintroduction of the
original defect, applied to the live code, which the check must catch. A check that still passes on the
broken code is not a check, and is reported as VACUOUS, which is a worse finding than a violation.

    rule                                                   checked      mutation
    1  speed is structure, not arithmetic                   no           --
    2  every test must be able to fail                      yes          --
    3  a threshold must be anchored to something invariant  yes          absolute cut
    4  "could not tell" is not "none"                       yes          force calibration
    5  graded, not binary                                   yes          BAND = 1
    6  two independent computations, or you will not notice yes          desynchronise the clamp
    7  a probe must confirm it perturbed something          yes          remove the window guard
    8  the negative is the deliverable                      yes          --
    9  predeclare                                           no           --

PREDECLARED, before any number:
    a rule is violated
        -> that is the finding. Report it, do not reword the rule.
    every check passes AND every mutation is caught
        -> the doctrine is enforced, which is the only claim this file is entitled to make.
    a check passes on its own mutation
        -> that check is decorative. Reported as VACUOUS and worth more than a green tick, because a
           decorative check is how a suite comes to certify a broken tool.

-> outputs/physics_doctrine.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import physics_reduce as PR  # noqa: E402
import physics_reduce_unknown as UNK  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 91117
N_PT = int(os.environ.get("DOCTRINE_N", 700))


# ---- the corpus the rules are checked against -------------------------------------------------------
def corpus(rng):
    """Three operators on one geometry, chosen so that between them they exercise every branch: a normal
    one, one whose spectrum is rescaled far from unity, and one where rigid motion is NOT free."""
    co = UNK.cloud(rng, n=N_PT, box=6.0)
    K, _ = UNK.network(co, rng=rng)
    ground = (K + 5.0 * sp.identity(K.shape[0], format="csr")).tocsr()
    return co, {"plain": K, "rescaled": (K * 1e12).tocsr(), "grounded": ground}


def build(co, K):
    S = PR.System(co, K, PR.rigid_basis(co))
    S.setup()
    return S


# ---- the rules --------------------------------------------------------------------------------------
def rule_2_can_fail(co, Ks, rng):
    """EVERY TEST MUST BE ABLE TO FAIL.

    Paid for by: the first reduction run scored 5/5 on the nucleosome, which proved nothing, because a
    probe that has never returned anything but HOLDS is indistinguishable from `return "HOLDS"`. The
    four-system run exists because of this rule.

    Checked against the RECORDED runs rather than fresh ones, so it audits results that are already
    committed rather than results generated to satisfy it."""
    seen = {}
    for fn in ("physics_reduce_unknown.json", "physics_reduce_rod.json"):
        p = OUT / fn
        if not p.exists():
            continue
        d = json.load(open(p))
        blocks = d.get("systems") or d.get("results") or []
        if "reductions" in d:
            blocks = list(blocks) + [d]
        for b in blocks:
            for r in b.get("reductions", []):
                seen.setdefault(r["name"], set()).add(r["verdict"])
    if not seen:
        return None, "no recorded runs found to audit -- rule unchecked, not passed"
    stuck = {k: sorted(v)[0] for k, v in seen.items() if len(v) == 1 and sorted(v)[0] == "HOLDS"}
    ok = not stuck
    detail = "; ".join(f"{k}: {'/'.join(sorted(v))}" for k, v in sorted(seen.items()))
    return ok, (f"{len(seen)} probes over the recorded corpus, each seen with: {detail}" if ok else
                f"ALWAYS HOLDS and therefore unfalsified here: {sorted(stuck)}")


def rule_3_invariant(co, Ks, rng):
    """A THRESHOLD MUST BE ANCHORED TO SOMETHING INVARIANT.

    Paid for by: the null-space cut, twice. An absolute 1e-9 found ZERO null modes on a spectrum reaching
    1e17; a cut relative to ||K||inf found FOURTEEN once a stretch modulus widened the spectrum 219x. The
    same physics, rescaled, must give the same answer -- if it does not, the number is about the units."""
    a = build(co, Ks["plain"])
    b = build(co, Ks["rescaled"])
    ok = a.n_null == b.n_null
    return ok, (f"n_null {a.n_null} unchanged when the operator is multiplied by 1e12 "
                f"(gap {a.gap:.2e} -> {b.gap:.2e})" if ok else
                f"n_null CHANGED under pure rescaling: {a.n_null} -> {b.n_null}")


def rule_4_declines(co, Ks, rng):
    """"COULD NOT TELL" IS NOT "NONE".

    Paid for by: a grounded operator, where rigid motion costs real energy so there is no direction known
    to be zero to calibrate against. The cut opened past the operator's own scale, the best available
    ratio was 1.05, and six null directions were reported on a system that has none. Reporting 0 would
    have been just as wrong in the other direction -- 0 asserts an answer, and there was none to assert."""
    S = build(co, Ks["grounded"])
    v = PR.probe_nullspace(np.array([1.0, 2.0, 3.0]), S.n_null, S)
    ok = (S.calibrated is False) and v["verdict"] == "NOT TESTABLE"
    return ok, (f"grounded operator: noise {S.noise:.1e} relative, calibration declined, verdict "
                f"{v['verdict']}" if ok else
                f"grounded operator answered anyway: calibrated={S.calibrated}, verdict {v['verdict']}")


def rule_5_graded(co, Ks, rng):
    """GRADED, NOT BINARY.

    Paid for by: timescale separation spanning six orders of magnitude across four systems, reported as
    "1 of 8 verdicts moved" -- a property of the REPORTING, blamed on the systems. And by a linearity
    error of 2e-8 against a 1e-8 threshold rendered as a categorical FAILS."""
    near = PR.verdict("t", 1.5e-8, 1e-8, "below", 1, "")["verdict"]
    far_bad = PR.verdict("t", 1e-2, 1e-8, "below", 1, "")["verdict"]
    far_good = PR.verdict("t", 1e-14, 1e-8, "below", 1, "")["verdict"]
    ok = near == "MARGINAL" and far_bad == "FAILS" and far_good == "HOLDS"
    return ok, (f"1.5x from the threshold -> {near}; 1e6x over -> {far_bad}; 1e-6x under -> {far_good}"
                if ok else f"grading collapsed: {near} / {far_bad} / {far_good}")


def rule_6_two_computations(co, Ks, rng):
    """TWO INDEPENDENT COMPUTATIONS OF THE SAME NUMBER, OR YOU WILL NOT NOTICE.

    Paid for by: two of the five defects in the null-space cut. setup said 17 null directions and the
    probe said 23, printed four lines apart in the same output, and nothing compared them. The fix was not
    to delete one -- it was to keep both and make the disagreement visible."""
    S = build(co, Ks["plain"])
    from scipy.sparse.linalg import eigsh
    vals = np.sort(np.abs(eigsh(Ks["plain"].tocsc(), k=64, sigma=-1e-4, which="LM")[0]))
    n_zero = int((vals <= S.cut).sum())
    v = PR.probe_nullspace(vals, n_zero, S)
    g2 = v["detail"].get("gap", float("nan"))
    ok = (n_zero == S.n_null) and (0.1 < g2 / max(S.gap, 1e-300) < 10.0)
    return ok, (f"setup {S.n_null} @ gap {S.gap:.2e}; independent k=64 solve {n_zero} @ gap {g2:.2e}"
                if ok else
                f"THE TWO DISAGREE: setup {S.n_null} @ {S.gap:.2e}, independent {n_zero} @ {g2:.2e}")


def rule_7_perturbed(co, Ks, rng):
    """A PROBE MUST CONFIRM IT PERTURBED SOMETHING.

    Paid for by: the finite-amplitude probe, twice in opposite directions. At 1e-8 bond lengths everything
    is linear, so it reported HOLDS on a system that is not -- a test that could only pass. Corrected too
    far, the descent ran to 6,600 bond lengths and it reported FAILS for the wrong reason. Both are
    non-measurements; the window has to be two-sided and the probe must decline outside it."""
    class Tiny(PR.System):
        """A harmonic system whose relax() deliberately moves almost nothing."""
        def __init__(self, co, K):
            super().__init__(co, K, PR.rigid_basis(co))
            self.energy = lambda z: 0.0

        def relax(self, f, steps=1, lr=None):
            return 1e-12 * self.solve(f)

    S = Tiny(co, Ks["plain"])
    S.setup()
    v = PR.probe_finite_linearity(S, rng, [])
    ok = v is not None and v["verdict"] == "NOT TESTABLE"
    return ok, (f"a perturbation of {v['detail']['rel_amp']:.1e} bond lengths -> {v['verdict']}"
                if ok else f"graded a non-perturbation: {v['verdict'] if v else 'probe returned None'}")


def rule_8_negative(co, Ks, rng):
    """THE NEGATIVE IS THE DELIVERABLE.

    Paid for by: the whole point of the tool. Knowing a reduction is INVALID is what stops someone
    building a scheme on it, so a FAILS record has to carry every field a HOLDS record does -- the
    quantity, the margin, the note, the supporting detail. A bare "no" is not actionable."""
    fields = {"quantity", "threshold", "margin", "note", "detail", "speedup"}
    h = set(PR.verdict("t", 1e-14, 1e-8, "below", 1, "n", {"x": 1}))
    f = set(PR.verdict("t", 1e-2, 1e-8, "below", 1, "n", {"x": 1}))
    ok = fields <= h and fields <= f and h == f
    return ok, ("a FAILS record carries the same fields as a HOLDS record" if ok else
                f"FAILS records are impoverished: missing {sorted(fields - f)}")


RULES = [
    (1, "speed is structure, not arithmetic", None,
     "Every large factor measured in this repo came from noticing the system HAD A PROPERTY and then not "
     "doing the work that property made unnecessary -- not from faster arithmetic, and not from a network "
     "that predicts forces. This is a claim about where to look. It cannot be unit-tested, and saying so "
     "is more honest than inventing a check that would pass either way."),
    (2, "every test must be able to fail", rule_2_can_fail, None),
    (3, "a threshold must be anchored to something invariant", rule_3_invariant, None),
    (4, '"could not tell" is not "none"', rule_4_declines, None),
    (5, "graded, not binary", rule_5_graded, None),
    (6, "two independent computations, or you will not notice", rule_6_two_computations, None),
    (7, "a probe must confirm it perturbed something", rule_7_perturbed, None),
    (8, "the negative is the deliverable", rule_8_negative, None),
    (9, "predeclare", None,
     "The gates go in the docstring before the run, so that a surprising number cannot be reinterpreted "
     "after the fact into a result. Nothing mechanical can verify this -- the check would be whether I "
     "wrote the gate before or after seeing the number, and only the commit history knows that. Claimed, "
     "not proven, and marked as such."),
]


# ---- mutations: a check that cannot fail is not a check ----------------------------------------------
class Mutation:
    """Reintroduce a defect on the live code, confirm the rule catches it, put it back."""

    def __init__(self, name, apply, undo):
        self.name, self._apply, self._undo = name, apply, undo

    def __enter__(self):
        self._apply()
        return self

    def __exit__(self, *a):
        self._undo()
        return False


def mutations():
    saved = {}

    def absolute_cut():
        """Rule 3's original defect: a cut that is a constant instead of a property of the spectrum."""
        saved["setup"] = PR.System.setup

        def broken(self, k0=16):
            saved["setup"](self, k0)
            from scipy.sparse.linalg import eigsh
            vals = np.sort(np.abs(eigsh(self.K.tocsc(), k=min(32, self.n3 - 2),
                                        sigma=-1e-4, which="LM")[0]))
            self.cut = 1e-9
            self.n_null = int((vals <= self.cut).sum())
            return self.t_setup
        PR.System.setup = broken

    def force_calibration():
        """Rule 4's original defect: assume rigid motion is free and answer regardless."""
        saved["setup4"] = PR.System.setup

        def broken(self, k0=16):
            t = saved["setup4"](self, k0)
            self.calibrated = True
            return t
        PR.System.setup = broken

    def binary_band():
        """Rule 5's original defect: a hard threshold with no margin."""
        saved["band"] = PR.BAND
        PR.BAND = 1.0

    def desync_clamp():
        """Rule 6's original defect: the two computations use different conventions, so they report
        different numbers for one quantity and the difference reads as noise rather than a bug."""
        saved["eps"] = np.finfo(float).eps
        saved["nullspace"] = PR.probe_nullspace

        def broken(vals, n_zero, S):
            r = saved["nullspace"](vals, n_zero, S)
            r["detail"]["gap"] = r["detail"].get("gap", 1.0) * 1e6
            return r
        PR.probe_nullspace = broken

    def no_window():
        """Rule 7's original defect: grade the deviation without checking the perturbation was real."""
        saved["fin"] = PR.probe_finite_linearity

        def broken(S, rng, log):
            v = saved["fin"](S, rng, log)
            if v is not None and v["verdict"] == "NOT TESTABLE":
                v["verdict"] = "HOLDS"          # what the probe did before the guard existed
            return v
        PR.probe_finite_linearity = broken

    return [
        Mutation("absolute cut instead of the gap", absolute_cut,
                 lambda: setattr(PR.System, "setup", saved["setup"])),
        Mutation("assume the calibration is valid", force_calibration,
                 lambda: setattr(PR.System, "setup", saved["setup4"])),
        Mutation("binary verdicts (BAND = 1)", binary_band,
                 lambda: setattr(PR, "BAND", saved["band"])),
        Mutation("desynchronise the two gap computations", desync_clamp,
                 lambda: setattr(PR, "probe_nullspace", saved["nullspace"])),
        Mutation("grade without the amplitude window", no_window,
                 lambda: setattr(PR, "probe_finite_linearity", saved["fin"])),
    ]


MUTATION_TARGET = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7}     # which rule each mutation must trip


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    report("=" * 100)
    report("THE DOCTRINE -- the rules, and a check for each that can fail")
    report("=" * 100)
    report("  Every rule below is the generalisation of a specific defect that produced a specific wrong")
    report("  number in this repo. Stated principles are free, so each one that can be checked is checked")
    report("  against the toolchain as it stands -- and each check is then run against a deliberate")
    report("  reintroduction of the original defect, because a check that cannot fail is not a check.")

    co, Ks = corpus(rng)
    report(f"\n  corpus: {N_PT} points, three operators (plain, rescaled 1e12, grounded)")

    results = []
    report(f"\n  {'#':<3}{'rule':<52}{'verdict':<14}")
    for num, name, fn, prose in RULES:
        if fn is None:
            report(f"  {num:<3}{name:<52}{'NOT CHECKABLE':<14}")
            for line in _wrap(prose, 94):
                report(f"      {line}")
            results.append({"rule": num, "name": name, "checked": False, "note": prose})
            continue
        try:
            ok, note = fn(co, Ks, rng)
        except Exception as ex:
            ok, note = False, f"check itself raised {type(ex).__name__}: {ex}"
        v = "UNCHECKED" if ok is None else ("HOLDS" if ok else "VIOLATED")
        report(f"  {num:<3}{name:<52}{v:<14}")
        report(f"      {note}")
        results.append({"rule": num, "name": name, "checked": True, "verdict": v, "note": note})

    report("\n  MUTATION TEST -- reintroduce each original defect and confirm its rule catches it")
    report(f"  {'mutation':<44}{'rule':<6}{'caught':<10}")
    muts = []
    for i, m in enumerate(mutations()):
        target = MUTATION_TARGET[i]
        fn = dict((n, f) for n, _, f, _ in [(r[0], r[1], r[2], r[3]) for r in RULES])[target]
        with m:
            try:
                ok, note = fn(co, Ks, rng)
            except Exception as ex:
                ok, note = False, f"raised {type(ex).__name__}"
        caught = ok is False
        report(f"  {m.name:<44}{target:<6}{'yes' if caught else 'NO -- VACUOUS':<10}")
        if not caught:
            report(f"      the check passed on the broken code, so it is decorative: {note}")
        muts.append({"mutation": m.name, "rule": target, "caught": bool(caught), "note": note})

    checked = [r for r in results if r.get("checked")]
    violated = [r for r in checked if r["verdict"] == "VIOLATED"]
    vacuous = [m for m in muts if not m["caught"]]
    report("\n  READING")
    report(f"    {len(checked)} of {len(RULES)} rules are mechanically checkable; "
           f"{len(RULES)-len(checked)} are claims about method and are marked as unproven.")
    if violated:
        report(f"    {len(violated)} RULE(S) VIOLATED by the current toolchain: "
               f"{[r['rule'] for r in violated]}. That is the finding.")
    else:
        report("    No rule is violated by the current toolchain.")
    if vacuous:
        report(f"    {len(vacuous)} CHECK(S) ARE VACUOUS -- they passed on code with the original defect")
        report(f"    put back: {[m['rule'] for m in vacuous]}. A decorative check is worse than a missing")
        report("    one, because it is how a suite comes to certify a broken tool.")
    else:
        report("    Every check caught its own mutation, so none of them is decorative.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "physics_doctrine", "n_pt": N_PT, "rules": results, "mutations": muts,
               "violated": [r["rule"] for r in violated], "vacuous": [m["rule"] for m in vacuous],
               "log": log}, open(OUT / "physics_doctrine.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'physics_doctrine.json'}")
    return 0


def _wrap(s, w):
    out, line = [], ""
    for word in s.split():
        if len(line) + len(word) + 1 > w:
            out.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
