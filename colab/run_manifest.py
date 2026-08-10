"""THE RUN MANIFEST -- move the data rules from prose to artefact.

WHY THIS EXISTS.  Two rounds of auditing this repo produced the same failure twice. The first audit's
"reports negatives" and "graded verdicts" columns turned out to be searching for physics_reduce's own
verdict words, so the line that invented the vocabulary scored 100% and everyone else scored zero. The
second audit's PROVENANCE rule scored HIGHER on modules that read no data at all -- 69% against 63% --
because it was matching the word SEED in any file that sets one. Both checks read prose. A docstring is a
claim about a run; only the recorded output is evidence of it.

So the fix is not a better regex. It is to make every experiment emit the facts, and check those.

WHAT A MANIFEST RECORDS
    inputs      every dataset actually read, with a content hash, so a rerun that quietly consumed a
                different file is visible instead of silent.
    coverage    rows available against rows used, and BY WHAT RULE they were chosen.
    seed        the seed, or an explicit statement that the run is deterministic without one.
    controls    which controls ran. Named, not implied.

THE FIELD THAT EXISTS BECAUSE OF A SPECIFIC BUG.  `selection` is a controlled vocabulary, not free text,
and the reason is the orphan fill. Its worklist was built as real[:1400] -- a PREFIX of a source-ordered
list -- and reported as a sample. It was 100% Reactome, the headline accuracy was wrong, and nothing in
the code or the output said which of the 10,249 candidates had been taken or how. A manifest recording
`available 10249, used 1400, selection "prefix"` makes that visible on the face of the artefact, because
a prefix of an ordered list is not a sample and the word says so.

    all         everything available was used
    random      drawn with the recorded seed
    stratified  drawn to match a declared distribution
    prefix      the first N in whatever order the source produced   <- ORDER-DEPENDENT, warned
    filtered    everything passing a stated predicate
    manual      chosen by hand                                      <- ORDER-DEPENDENT, warned

A manifest is not a claim of correctness. It is the minimum set of facts that lets someone else check the
claim, and it costs one function call.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

SELECTION = ("all", "random", "stratified", "prefix", "filtered", "manual")
ORDER_DEPENDENT = ("prefix", "manual")
BIG = 64 << 20          # hash the whole file below this; sample it above, and say which


def digest(path):
    """Content hash of an input. Large files are sampled at both ends rather than skipped, and the
    method is recorded so a hash is never silently comparing unlike things."""
    p = Path(path)
    if not p.exists():
        return {"path": str(path), "missing": True}
    size = p.stat().st_size
    h = hashlib.sha256()
    if size <= BIG:
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        how = "sha256:full"
    else:
        with open(p, "rb") as fh:
            h.update(fh.read(8 << 20))
            fh.seek(-(8 << 20), os.SEEK_END)
            h.update(fh.read())
        how = "sha256:ends8M"
    return {"path": str(path), "bytes": int(size), "hash": h.hexdigest()[:16], "method": how}


def manifest(inputs=(), available=None, used=None, selection="all", seed=None, controls=(), note=""):
    """Build the record. Returns the dict AND its own warnings -- a manifest that cannot criticise the run
    it describes would be decoration."""
    warn = []
    if selection not in SELECTION:
        warn.append(f"selection '{selection}' is not one of {SELECTION}; coverage cannot be interpreted")
    if available is None or used is None:
        warn.append("coverage NOT DECLARED -- rows available and rows used are both required, because "
                    "a score on an undeclared subset is not a score on the dataset")
    else:
        if used < available and selection in ORDER_DEPENDENT:
            warn.append(f"ORDER-DEPENDENT SAMPLE: {used} of {available} by '{selection}'. This is a "
                        f"property of the source order, not a sample of the population, unless that "
                        f"order is independently known to be random. The orphan fill was 100% one "
                        f"source for exactly this reason")
        if used > available:
            warn.append(f"used ({used}) exceeds available ({available})")
    if seed is None:
        warn.append("no seed recorded -- a rerun is not guaranteed to be the same experiment")
    if not controls:
        warn.append("no control declared -- a number with nothing to compare against is not a result")
    ins = [digest(p) for p in inputs]
    for i in ins:
        if i.get("missing"):
            warn.append(f"declared input does not exist: {i['path']}")
    return {"manifest_version": 1, "inputs": ins,
            "coverage": {"available": available, "used": used, "selection": selection,
                         "fraction": (used / available) if (available and used is not None) else None},
            "seed": seed, "controls": list(controls), "note": note, "warnings": warn}


REQUIRED = ("manifest_version", "inputs", "coverage", "seed", "controls")


def check(obj):
    """Is there a usable manifest in this recorded output? Returns (present, complete, warnings)."""
    m = obj.get("manifest") if isinstance(obj, dict) else None
    if not isinstance(m, dict):
        return False, False, []
    complete = all(k in m for k in REQUIRED) and m.get("coverage", {}).get("available") is not None
    return True, bool(complete), list(m.get("warnings", []))


def report(m, emit=print):
    """Print the manifest's own complaints. Silence here means the run declared everything it should."""
    c = m["coverage"]
    emit(f"    manifest: {len(m['inputs'])} inputs, coverage "
         f"{c['used']}/{c['available']}" + (f" ({c['fraction']:.1%})" if c["fraction"] else "") +
         f" by '{c['selection']}', seed {m['seed']}, controls {len(m['controls'])}")
    for w in m["warnings"]:
        emit(f"      WARN {w}")
    return m


def save(obj, path, available=None, used=None, selection="all", seed=None, controls=(), note="",
         inputs=None, indent=2):
    """Write a result WITH its manifest attached, taking the inputs from the live trace.

    THE ADOPTION PROBLEM THIS SOLVES. Manifest adoption sat at 0 of 153 modules, and asking 442 modules to
    add a bookkeeping call was never going to change that -- voluntary adoption of anything across a repo
    this size does not happen. But almost every module here ends the same way:

        json.dump(result, open(OUT / "x.json", "w"), indent=2)      ->      save(result, OUT / "x.json")

    That substitution is mechanical, scriptable, and carries no argument. Under `with trace():` the inputs
    are filled in from what the process actually opened, so the module declares nothing and gets a
    complete, hash-pinned input list -- including the paths its own source could never express.

    Coverage still has to be passed by hand. That is deliberate: only the author knows how many rows were
    available and by what rule they were chosen, and inferring it would produce exactly the confident
    wrong number the manifest exists to prevent."""
    if inputs is None:
        try:
            import run_trace
            t = run_trace.active()
            inputs = t.input_paths() if t is not None else []
        except Exception:
            inputs = []
    m = manifest(inputs=inputs, available=available, used=used, selection=selection, seed=seed,
                 controls=controls, note=note)
    body = dict(obj) if isinstance(obj, dict) else {"result": obj}
    body["manifest"] = m
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(body, fh, indent=indent)
    return m


if __name__ == "__main__":
    # A demonstration that the warning fires on the exact shape of the bug it exists for.
    print("the orphan fill's worklist, as it actually was:")
    report(manifest(available=10249, used=1400, selection="prefix", seed=None))
    print("\nthe same worklist, built correctly:")
    report(manifest(available=10249, used=1400, selection="random", seed=17, controls=["shuffled-label"]))
    sys.exit(0)
