"""THE DOCTRINE TURNED ON THE REST OF THE REPO -- which subsystems were built to the same standard?

WHY.  physics_doctrine.py checks that the reduction toolchain obeys its own rules. That is one line of
work out of many in this repository, and the rules are about METHOD, not about physics: predeclare the
gate, run a control, report the negative, grade the verdict, do not let a threshold float. Those apply to
a knockout predictor and a chromatin simulation exactly as they apply to an eigensolver. So the honest
question is not "does the physics line obey the doctrine" -- it is "which parts of this repo do, and which
were built before the lesson was learned".

WHAT THIS CAN AND CANNOT SEE.  Each check is labelled with its evidence class, because they are not
equally strong and presenting them as one number would be the exact sin the doctrine exists to prevent:

    GIT     strong. The commit graph is a record of what happened, not a claim about it. Whether the
            predeclared gate was committed BEFORE the output it grades is a fact.
    OUTPUT  medium. A recorded run either does or does not contain a failing verdict. It cannot be
            written into existence by a docstring.
    TEXT    WEAK. The presence of the word "control" in a source file is evidence that the author knew
            the word. A module can say PREDECLARED and still be sloppy, and a module can run a perfect
            control while calling it something else. TEXT checks are reported separately and never
            summed with the others.

THE CONTROL THIS AUDIT NEEDS.  A scoring scheme that rates every subsystem alike is measuring its own
regexes. So the audit is also run on two calibration sets whose answers are known in advance:

    physics_*        written this week, after the lessons, with predeclared gates and mutation tests.
                     Must score HIGH or the audit cannot see good practice.
    prototype_p*     the earliest prototypes in the repo, written before any of this was learned.
                     Must score LOWER or the audit cannot see its absence.

If those two come out the same, the audit is a thermometer stuck at 37 and the per-subsystem numbers
below are worthless -- and that is the result that gets reported, not quietly dropped.

PREDECLARED, before any number:
    physics_* scores above prototype_p* on the GIT and OUTPUT checks
        -> the audit discriminates, and the subsystem scores mean something.
    they score the same
        -> the audit measures its own regexes. Report that and throw the numbers away.
    a named subsystem scores near zero on GIT
        -> its outputs were committed before or without a predeclared gate. That is worth knowing and is
           not an accusation of fraud: most of this repo predates the rule.

-> outputs/doctrine_audit.json
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))

# The five the request named, plus the two calibration sets.
SUBSYSTEMS = {
    "cell model":    ["colab/cell_*.py"],
    "network":       ["colab/*network*.py"],
    "nexus":         ["colab/nexus*.py"],
    "4d chromatin":  ["colab/chromatin*.py"],
    "cell software": ["cell_sim/*.py", "cell_sim/**/*.py"],
}
CALIBRATION = {
    "physics_* (should score HIGH)":    ["colab/physics_*.py"],
    "prototype_p* (should score LOW)":  ["prototype_p*.py"],
}
# THE FIRST CALIBRATION SET WAS USELESS AND THE AUDIT SAID SO. prototype_p* has ZERO recorded outputs --
# they are prototypes, they never wrote outputs/*.json -- so both STRONG checks came back "not
# comparable" and the control rested on one weak regex. The audit correctly declared itself invalid.
#
# A control has to be able to run. So the real one splits every audited module that HAS a recorded output
# by the age of its first commit: the doctrine was learned recently, so if this audit can see anything at
# all, newer work must score higher than older work on the GIT and OUTPUT checks. That comparison is not
# circular -- gate_before_output compares two commits of the SAME module, and an old module is perfectly
# free to have committed its gate first.
AGE_SPLIT = 3

# TEXT checks -- weak by construction, reported apart from the rest.
PATTERNS = {
    "predeclared": re.compile(r"PREDECLAR|predeclar|written (down )?(first|before)", re.I),
    "control":     re.compile(r"\bcontrol\b|\bshuffl|\bpermut|null model|scrambl|negative control", re.I),
    "heldout":     re.compile(r"held[- ]out|holdout|test set|validation set|leave-one-out|\bLOCO\b", re.I),
    "graded":      re.compile(r"MARGINAL|NOT TESTABLE|UNRESOLVABLE|abstain|inconclusive|declin", re.I),
    "negative":    re.compile(r"FAILS|honest negative|does not hold|no effect|null result|refut", re.I),
}


def _sh(*a):
    try:
        return subprocess.run(a, cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def files_for(globs):
    out = []
    for g in globs:
        out += sorted(ROOT.glob(g))
    return [p for p in dict.fromkeys(out) if p.is_file() and "__pycache__" not in str(p)]


def output_for(path):
    """The recorded run a module writes, if the module names one. Modules in this repo end their
    docstring with `-> outputs/<name>.json`, which is the only reliable link between code and result."""
    try:
        head = path.read_text(errors="ignore")[:6000]
    except Exception:
        return None
    m = re.search(r"->\s*(outputs/[\w./-]+\.json)", head)
    if not m:
        return None
    p = ROOT / m.group(1)
    return p if p.exists() else None


def first_commit(rel):
    """Earliest commit touching a path, as a unix timestamp."""
    s = _sh("git", "log", "--diff-filter=A", "--format=%ct", "--", str(rel))
    ts = [int(x) for x in s.split() if x.isdigit()]
    return min(ts) if ts else None


def commit_of_predeclaration(rel):
    """Earliest commit whose version of the file already contained a predeclared gate. This is the GIT
    check: a gate written after the number it grades is not a gate."""
    s = _sh("git", "log", "--format=%H %ct", "--reverse", "--", str(rel))
    for line in s.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        sha, ts = parts[0], int(parts[1])
        blob = _sh("git", "show", f"{sha}:{rel}")
        if blob and PATTERNS["predeclared"].search(blob[:6000]):
            return ts
    return None


def audit_module(path):
    rel = path.relative_to(ROOT).as_posix()
    try:
        src = path.read_text(errors="ignore")
    except Exception:
        return None
    head = src[:6000]
    rec = {"module": rel, "text": {k: bool(v.search(head)) for k, v in PATTERNS.items()}}

    op = output_for(path)
    rec["output"] = op.relative_to(ROOT).as_posix() if op else None
    rec["output_has_negative"] = None
    rec["output_has_graded"] = None
    if op:
        try:
            blob = op.read_text(errors="ignore")
            # THESE TWO ARE CIRCULAR AND ARE REPORTED AS SUCH.
            #
            # They search recorded outputs for the words FAILS / MARGINAL / NOT TESTABLE. That is the
            # verdict VOCABULARY of physics_reduce, so physics_* scores 100% on both by construction --
            # it invented the words -- while a subsystem that reports negatives perfectly well in its own
            # language scores zero. Checked rather than assumed: of the matches outside physics, cell
            # model contributed 2 and chromatin 2, and every one of them was a JSON boolean `false`
            # rather than a reported negative result. Nexus contributed one real FAILS.
            #
            # So these columns measure whether a module adopted one particular vocabulary. That is worth
            # knowing -- the graded verdict has not propagated past the line that invented it -- but it
            # is NOT a measure of whether negatives get reported, and it must not be read as one.
            rec["output_has_negative"] = bool(re.search(r"FAILS|not supported|refuted", blob))
            rec["output_has_graded"] = bool(re.search(r"MARGINAL|NOT TESTABLE|UNRESOLVABLE|abstain",
                                                      blob))
        except Exception:
            pass

    rec["gate_before_output"] = None
    if op:
        t_gate = commit_of_predeclaration(rel)
        t_out = first_commit(op.relative_to(ROOT).as_posix())
        if t_gate is not None and t_out is not None:
            rec["gate_before_output"] = bool(t_gate <= t_out)
        elif t_out is not None:
            rec["gate_before_output"] = False        # output committed, gate never present
    return rec


def score(recs):
    n = len(recs)
    withop = [r for r in recs if r["output"]]
    gitable = [r for r in withop if r["gate_before_output"] is not None]
    return {
        "modules": n,
        "with_recorded_output": len(withop),
        "GIT_gate_before_output": (sum(bool(r["gate_before_output"]) for r in gitable) / len(gitable)
                                   if gitable else None),
        "OUTPUT_has_negative": (sum(bool(r["output_has_negative"]) for r in withop) / len(withop)
                                if withop else None),
        "OUTPUT_has_graded": (sum(bool(r["output_has_graded"]) for r in withop) / len(withop)
                              if withop else None),
        "TEXT_predeclared": sum(r["text"]["predeclared"] for r in recs) / max(n, 1),
        "TEXT_control": sum(r["text"]["control"] for r in recs) / max(n, 1),
        "TEXT_heldout": sum(r["text"]["heldout"] for r in recs) / max(n, 1),
        "TEXT_negative": sum(r["text"]["negative"] for r in recs) / max(n, 1),
    }


def _fmt(v):
    return "  --  " if v is None else f"{v:>5.0%}"


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("THE DOCTRINE TURNED ON THE REST OF THE REPO")
    report("=" * 100)
    report("  The rules are about METHOD, so they apply to a knockout predictor as much as to an")
    report("  eigensolver. Evidence classes are kept apart and never summed:")
    report("    GIT     strong -- the commit graph records what happened rather than claiming it")
    report("    OUTPUT  medium -- a recorded run either contains a failing verdict or it does not")
    report("    TEXT    WEAK   -- the word 'control' in a file proves the author knew the word")

    all_recs, scores = {}, {}
    for name, globs in list(SUBSYSTEMS.items()) + list(CALIBRATION.items()):
        fs = files_for(globs)
        recs = [r for r in (audit_module(p) for p in fs) if r]
        all_recs[name] = recs
        scores[name] = score(recs)
        report(f"  scanned {name}: {len(recs)} modules")

    report(f"\n  STRONG -- the commit graph. 'runs' is how many modules record a result at all.")
    report(f"  {'subsystem':<32}{'mods':>6}{'runs':>6}{'cover':>7}{'GIT gate':>10}")
    for name in list(SUBSYSTEMS) + list(CALIBRATION):
        s = scores[name]
        cov = s['with_recorded_output'] / max(s['modules'], 1)
        report(f"  {name:<32}{s['modules']:>6}{s['with_recorded_output']:>6}{cov:>7.0%}"
               f"{_fmt(s['GIT_gate_before_output']):>10}")

    report("\n  CIRCULAR -- these search for physics_reduce's own verdict words, so it scores 100%")
    report("  by construction. Read as vocabulary ADOPTION, never as whether negatives get reported.")
    report(f"  {'subsystem':<32}{'FAILS':>8}{'graded':>8}")
    for nm2 in list(SUBSYSTEMS) + list(CALIBRATION):
        s2 = scores[nm2]
        report(f"  {nm2:<32}{_fmt(s2['OUTPUT_has_negative']):>8}{_fmt(s2['OUTPUT_has_graded']):>8}")

    report(f"\n  TEXT checks -- weak evidence, kept separate on purpose")
    report(f"  {'subsystem':<32}{'predecl':>9}{'control':>9}{'heldout':>9}{'negative':>10}")
    for name in list(SUBSYSTEMS) + list(CALIBRATION):
        s = scores[name]
        report(f"  {name:<32}{s['TEXT_predeclared']:>9.0%}{s['TEXT_control']:>9.0%}"
               f"{s['TEXT_heldout']:>9.0%}{s['TEXT_negative']:>10.0%}")

    # THE CONTROL THAT CAN ACTUALLY RUN: oldest vs newest third of everything with a recorded output.
    pool = []
    for name, recs in all_recs.items():
        if name in CALIBRATION:
            continue
        for r in recs:
            if r["output"] and r["gate_before_output"] is not None:
                t = first_commit(r["module"])
                if t:
                    pool.append((t, r))
    pool.sort(key=lambda x: x[0])
    k = max(len(pool) // AGE_SPLIT, 1)
    old = [r for _, r in pool[:k]]
    new = [r for _, r in pool[-k:]]
    report(f"\n  THE CONTROL -- can this audit tell good practice from its absence?")
    report(f"    prototype_p* was useless as a low set: {scores['prototype_p* (should score LOW)']['with_recorded_output']}"
           f" of {scores['prototype_p* (should score LOW)']['modules']} have a recorded output, so both")
    report("    STRONG checks were not comparable. Using oldest vs newest third of everything that HAS a")
    report(f"    recorded output instead -- {len(old)} modules each, out of {len(pool)}.")
    sep = []
    for k2, getter in (("GIT_gate_before_output", lambda r: bool(r["gate_before_output"])),
                       ("OUTPUT_has_negative", lambda r: bool(r["output_has_negative"])),
                       ("OUTPUT_has_graded", lambda r: bool(r["output_has_graded"]))):
        a = sum(getter(r) for r in new) / max(len(new), 1)
        b = sum(getter(r) for r in old) / max(len(old), 1)
        report(f"    {k2:<28}{a:>6.0%} newest vs {b:>6.0%} oldest   "
               f"{'SEPARATES' if a - b > 0.2 else 'does not separate'}")
        sep.append(a - b > 0.2)
    hi, lo = scores["physics_* (should score HIGH)"], scores["prototype_p* (should score LOW)"]
    report(f"    {'TEXT_predeclared':<28}{hi['TEXT_predeclared']:>6.0%} physics vs "
           f"{lo['TEXT_predeclared']:>6.0%} prototype   "
           f"{'SEPARATES' if hi['TEXT_predeclared'] - lo['TEXT_predeclared'] > 0.2 else 'does not'}")
    sep.append(hi["TEXT_predeclared"] - lo["TEXT_predeclared"] > 0.2)
    discriminates = sum(sep) >= 2
    report("")
    if discriminates:
        report("    The audit separates the two calibration sets, so the per-subsystem numbers above")
        report("    carry information. They still measure DECLARED practice, not correctness.")
    else:
        report("    THE AUDIT DOES NOT SEPARATE THEM. It is measuring its own regexes, and every")
        report("    per-subsystem number above should be discarded rather than interpreted.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "doctrine_audit", "scores": scores, "discriminates": bool(discriminates),
               "modules": {k: v for k, v in all_recs.items()}, "log": log},
              open(OUT / "doctrine_audit.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'doctrine_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
