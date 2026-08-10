"""OBSERVE THE RUN INSTEAD OF PARSING THE SOURCE.

THE WEAKNESS THIS RETIRES.  Static analysis could verify the inputs of 42 of 357 modules. The other 315
were invisible for three reasons it can never fix: 93 build the path in a variable (pd.read_parquet(UP)),
some interpolate it (f"ko_dossier_{ko}.json"), and 104 name a file that is not where the tool looked.
Every one of those modules is opening a real file at runtime. The information is not missing -- it is
simply not present in the text, and no amount of AST work will put it there.

    static:   what the source SAYS it might read
    traced:   what the process ACTUALLY opened

That distinction has cost this repo seven rounds of false findings. string_ppi was ranked a top
stale-schema candidate for reading its own STRING table correctly, because the path was a variable.
ko_cell_map likewise, because the path was an f-string. Both vanish the moment you watch the run.

HOW.  builtins.open, gzip.open, json.load, numpy.load and the pandas readers are wrapped for the duration
of a context manager, and every path opened is recorded with its mode. Dicts returned by json.load are
handed back as a dict SUBCLASS that logs which keys are actually fetched -- so field access becomes a
measurement rather than an AST approximation, and a key read through any amount of indirection is caught.
Nothing is monkeypatched permanently and nothing needs editing in the 442 modules.

THE COST, STATED PLAINLY.  The module has to run. That is a real restriction -- an expensive pipeline is
expensive to audit -- but 153 modules already have a recorded output on disk, which means they were run at
least once, so they are runnable by construction.

PREDECLARED, before any number:
    the tracer resolves inputs that the static extractor cannot
        -> the blindness was an artefact of reading text, and the ceiling on rules 1 and 4 lifts.
    the tracer finds exactly what the static extractor found and no more
        -> the variable paths and f-strings were not actually opened at run time, the static picture was
           complete, and this file is unnecessary.
    the tracer misses a literal path the static extractor found
        -> a code path that did not execute. Traced input lists are a LOWER bound on one run, never the
           full set, and anything built on them has to say so.

-> outputs/run_trace_selftest.json
"""
import builtins
import gzip
import io
import json
import os
import sys
import time
from pathlib import Path

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
_ACTIVE = None


class TracedDict(dict):
    """A dict that remembers which keys were fetched. Still a dict, so isinstance passes and json.dump
    serialises it unchanged -- the wrapper must be invisible to the code being observed."""

    __slots__ = ("_tr", "_src")

    def __new__(cls, data, tr, src):
        o = super().__new__(cls, data)
        return o

    def __init__(self, data, tr, src):
        super().__init__(data)
        self._tr, self._src = tr, src

    def _wrap(self, v):
        return TracedDict(v, self._tr, self._src) if isinstance(v, dict) else v

    def __getitem__(self, k):
        if isinstance(k, str):
            self._tr.key(self._src, k)
        return self._wrap(super().__getitem__(k))

    def get(self, k, default=None):
        if isinstance(k, str):
            self._tr.key(self._src, k)
        return self._wrap(super().get(k, default))


class Trace:
    def __init__(self, keys=True):
        self.reads, self.writes, self.keys_by_file = {}, {}, {}
        self.track_keys = keys
        self.t0 = time.time()

    def _rec(self, d, path, how):
        try:
            p = str(Path(path).resolve())
        except Exception:
            p = str(path)
        d.setdefault(p, set()).add(how)

    def read(self, path, how):
        self._rec(self.reads, path, how)

    def write(self, path, how):
        self._rec(self.writes, path, how)

    def key(self, src, k):
        self.keys_by_file.setdefault(src, set()).add(k)

    def summary(self):
        return {
            "reads": {k: sorted(v) for k, v in sorted(self.reads.items())},
            "writes": {k: sorted(v) for k, v in sorted(self.writes.items())},
            "keys": {k: sorted(v) for k, v in sorted(self.keys_by_file.items())},
            "seconds": round(time.time() - self.t0, 2),
        }

    def input_paths(self):
        """Files opened for reading and not written by this same run -- the manifest's `inputs`."""
        return [p for p in self.reads if p not in self.writes]


class trace:
    """Context manager. Everything is restored on exit, including after an exception."""

    def __init__(self, keys=True):
        self.tr = Trace(keys)
        self._saved = {}

    def __enter__(self):
        global _ACTIVE
        t = self.tr
        self._saved["open"] = builtins.open
        self._saved["gzip_open"] = gzip.open
        self._saved["json_load"] = json.load

        def _mode_of(a, kw):
            m = kw.get("mode")
            if m is None and len(a) > 1 and isinstance(a[1], str):
                m = a[1]
            return m or "r"

        def op(*a, **kw):
            m = _mode_of(a, kw)
            if a:
                (t.write if any(c in m for c in "wax") else t.read)(a[0], "open")
            return self._saved["open"](*a, **kw)

        def gop(*a, **kw):
            m = _mode_of(a, kw)
            if a:
                (t.write if any(c in m for c in "wax") else t.read)(a[0], "gzip")
            return self._saved["gzip_open"](*a, **kw)

        def jload(fp, *a, **kw):
            name = getattr(fp, "name", None)
            obj = self._saved["json_load"](fp, *a, **kw)
            if name:
                t.read(name, "json.load")
                if t.track_keys and isinstance(obj, dict):
                    return TracedDict(obj, t, str(Path(name).resolve()))
            return obj

        builtins.open, gzip.open, json.load = op, gop, jload

        try:
            import numpy as np
            self._saved["np_load"] = np.load

            def nload(f, *a, **kw):
                if isinstance(f, (str, os.PathLike)):
                    t.read(f, "np.load")
                return self._saved["np_load"](f, *a, **kw)
            np.load = nload
        except Exception:
            pass

        if "pandas" in sys.modules:
            self._patch_pandas(t)
        global _ACTIVE
        _ACTIVE = t
        return t

    def _patch_pandas(self, t):
        import pandas as pd
        for fn in ("read_parquet", "read_csv", "read_table", "read_json", "read_excel", "read_pickle"):
            if not hasattr(pd, fn):
                continue
            self._saved[f"pd_{fn}"] = getattr(pd, fn)

            def mk(orig, label):
                def f(path, *a, **kw):
                    if isinstance(path, (str, os.PathLike)):
                        t.read(path, label)
                    return orig(path, *a, **kw)
                return f
            setattr(pd, fn, mk(self._saved[f"pd_{fn}"], f"pd.{fn}"))

    def __exit__(self, *exc):
        global _ACTIVE
        builtins.open = self._saved["open"]
        gzip.open = self._saved["gzip_open"]
        json.load = self._saved["json_load"]
        if "np_load" in self._saved:
            import numpy as np
            np.load = self._saved["np_load"]
        if "pandas" in sys.modules:
            import pandas as pd
            for k, v in self._saved.items():
                if k.startswith("pd_"):
                    setattr(pd, k[3:], v)
        _ACTIVE = None
        return False


def active():
    return _ACTIVE


def _selftest(report):
    """THE CONTROL. A module whose inputs are deliberately unresolvable from the text: one literal path,
    one f-string, one variable, one gzip. The static extractor should find only the literal; the tracer
    must find all four. If they agree, this file is unnecessary and should be deleted."""
    import ast
    import tempfile
    import shutil
    d = Path(tempfile.mkdtemp())
    for n in ("plain.json", "interp_7.json", "hidden.json"):
        json.dump({"alpha": 1, "beta": 2}, open(d / n, "w"))
    with gzip.open(d / "zipped.json.gz", "wt") as fh:
        json.dump({"gamma": 3}, fh)
    src = f'''
import json, gzip
from pathlib import Path
D = Path({str(d)!r})
VAR = D / "hidden.json"
def main(i=7):
    a = json.load(open(D / "plain.json"))["alpha"]
    b = json.load(open(D / f"interp_{{i}}.json"))["beta"]
    c = json.load(open(VAR))
    with gzip.open(D / "zipped.json.gz", "rt") as fh:
        g = json.load(fh)["gamma"]
    return a, b, c, g
'''
    mod = d / "synthetic_mod.py"
    mod.write_text(src)

    DATAFILE = ("json", "gz")
    lits = {c.value for c in ast.walk(ast.parse(src))
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
            and c.value.split(".")[-1] in DATAFILE}
    sys.path.insert(0, str(d))
    import importlib
    m = importlib.import_module("synthetic_mod")
    with trace() as t:
        m.main()
    traced = {Path(p).name for p in t.input_paths()}
    sys.path.remove(str(d))

    report(f"    {'':4}{'static (source text)':<34}{sorted(lits)}")
    report(f"    {'':4}{'traced (what ran)':<34}{sorted(traced)}")
    want = {"plain.json", "interp_7.json", "hidden.json", "zipped.json.gz"}
    ok = want <= traced
    report(f"    {'':4}{'tracer found all four':<34}{ok}")
    report(f"    {'':4}{'static missed':<34}{sorted(want - set(lits) - {'zipped.json.gz'} | ({'zipped.json.gz'} if 'zipped.json.gz' not in lits else set()))}")
    keys = {Path(k).name: v for k, v in t.summary()["keys"].items()}
    report(f"    {'':4}{'keys observed':<34}{keys}")
    shutil.rmtree(d, ignore_errors=True)
    return ok, sorted(lits), sorted(traced), keys


def main():
    log = []

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("OBSERVE THE RUN INSTEAD OF PARSING THE SOURCE")
    report("=" * 100)
    report("  Static analysis could verify 42 of 357 modules. The other 315 build their paths in")
    report("  variables, interpolate them, or name files the scanner could not locate. Every one of them")
    report("  opens a real file at runtime; the information is not missing, it is just not in the text.")
    report("\n  SELF-TEST -- a module written to be unresolvable from its source")
    ok, lits, traced, keys = _selftest(report)
    report("")
    if ok:
        report("    The tracer resolves paths the source cannot express. The ceiling on rules 1 and 4")
        report("    was an artefact of reading text, and lifts for any module that can be executed.")
    else:
        report("    THE TRACER DID NOT RESOLVE THEM. This file does not do what it claims; do not build")
        report("    on it.")
    report("\n  THE LIMIT, STATED: a traced input list is what ONE run touched. A branch that did not")
    report("  execute contributes nothing, so traces are a LOWER bound and never a complete input set.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "run_trace_selftest", "passed": bool(ok), "static": lits, "traced": traced,
               "keys": keys, "log": log},
              open(OUT / "run_trace_selftest.json", "w"), indent=2)
    report(f"\n  -> {OUT/'run_trace_selftest.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
