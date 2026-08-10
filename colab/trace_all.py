"""RUN THE TRACER OVER EVERY MODULE THAT RECORDS A RESULT.

THE CEILING THIS TESTS.  Static analysis could fully verify the inputs of 42 of 357 modules. The other 315
were invisible because they build paths in variables, interpolate them, or name files the scanner could
not locate. run_trace showed on one module that watching the process resolves all of it. The question this
answers is whether that holds at scale, or whether it merely moves the ceiling from "cannot parse" to
"cannot run".

    153 modules declare a recorded output, which means each was executed at least once at some point.
    That is the population. Anything that will not run today is reported as such rather than dropped.

METHOD.  Each module runs in its own subprocess under a timeout, so a hang, a crash or a sys.exit cannot
take the harness with it. The child imports the module under `with trace():`, calls main() if there is
one, and prints the trace as JSON. Failures are recorded with their exception -- a module that dies after
opening three files still tells us those three files, and a partial trace is a real measurement.

SAFETY.  These modules WRITE. They are run with the environment they expect, because most read and write
the same directory and redirecting the output would simply stop them finding their inputs. The working
tree is committed clean beforehand and outputs/ is restored from git afterwards, so the only thing kept
from this exercise is the trace itself.

PREDECLARED, before any number:
    traced modules with fully resolved inputs substantially exceeds the static 42
        -> the ceiling was an artefact of reading text, and rules 1 and 4 can be checked on most of the
           repo rather than a tenth of it.
    most modules fail to run
        -> the ceiling moves rather than lifts, and the honest description of this approach is that it
           audits the runnable subset. The failure count is the headline in that case, not a footnote.
    traced inputs are a subset of static ones everywhere
        -> the variable paths were never exercised and static analysis was already complete.

-> outputs/trace_all.json
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SCRATCH = Path(os.environ.get("TRACE_SCRATCH", "/tmp/trace_all"))
TIMEOUT = int(os.environ.get("TRACE_TIMEOUT", 75))
JOBS = int(os.environ.get("TRACE_JOBS", 4))
MARK = "@@TRACE@@"

DRIVER = r'''
import sys, json, os, importlib.util
sys.path.insert(0, "colab")
import run_trace
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("m_under_trace", path)
m = importlib.util.module_from_spec(spec)
status = "ok"
with run_trace.trace() as t:
    try:
        spec.loader.exec_module(m)
        if hasattr(m, "main"):
            m.main()
    except SystemExit:
        pass
    except BaseException as e:
        status = f"{type(e).__name__}: {str(e)[:140]}"
s = t.summary()
sys.stdout.write("@@TRACE@@" + json.dumps({
    "status": status,
    "inputs": t.input_paths(),
    "writes": sorted(s["writes"]),
    "keys": {k: v for k, v in s["keys"].items()},
}) + "\n")
'''


def run_one(mod):
    drv = SCRATCH / "driver.py"
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, str(drv), mod], cwd=ROOT, timeout=TIMEOUT,
                           capture_output=True, text=True,
                           env={**os.environ, "MPLBACKEND": "Agg", "TRACE_CHILD": "1"})
        out = r.stdout
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        return {"module": mod, "status": f"TIMEOUT after {TIMEOUT}s", "inputs": [], "writes": [],
                "keys": {}, "seconds": round(time.time() - t0, 1)}
    except Exception as ex:
        return {"module": mod, "status": f"launch failed: {type(ex).__name__}", "inputs": [],
                "writes": [], "keys": {}, "seconds": round(time.time() - t0, 1)}
    rec = {"module": mod, "status": "no trace emitted", "inputs": [], "writes": [], "keys": {}}
    for line in out.splitlines():
        if line.startswith(MARK):
            try:
                rec.update(json.loads(line[len(MARK):]))
            except Exception:
                pass
            rec["module"] = mod
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "driver.py").write_text(DRIVER)

    inv = json.load(open(OUT / "data_doctrine.json"))["inventory"]
    mods = [r for r in inv if r.get("declared_output")]
    static = {r["module"]: r for r in inv}

    report("=" * 100)
    report("RUN THE TRACER OVER EVERY MODULE THAT RECORDS A RESULT")
    report("=" * 100)
    report(f"  {len(mods)} modules declare a recorded output, so each ran at least once at some point.")
    report(f"  Each runs in its own subprocess, {TIMEOUT}s timeout, {JOBS} at a time. A module that dies")
    report("  after opening three files still tells us those three files; partial traces are kept.")

    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        recs = list(ex.map(lambda r: run_one(r["module"]), mods))

    ok = [r for r in recs if r["status"] == "ok"]
    to = [r for r in recs if r["status"].startswith("TIMEOUT")]
    err = [r for r in recs if r["status"] not in ("ok",) and not r["status"].startswith("TIMEOUT")]
    traced_any = [r for r in recs if r["inputs"]]
    report(f"\n  {'outcome':<34}{'modules':>9}")
    report(f"  {'ran to completion':<34}{len(ok):>9}")
    report(f"  {'raised':<34}{len(err):>9}")
    report(f"  {'timed out':<34}{len(to):>9}")
    report(f"  {'produced ANY resolved input':<34}{len(traced_any):>9}")

    # what the trace saw that the source could not express
    gained, lost = 0, 0
    detail = []
    for r in recs:
        s = static.get(r["module"], {})
        st = {Path(x).name for x in s.get("reads", [])}
        tr = {Path(x).name for x in r["inputs"]}
        new, missed = tr - st, st - tr
        gained += len(new)
        lost += len(missed)
        if new:
            detail.append({"module": r["module"], "new": sorted(new)[:6], "n_new": len(new)})
    report(f"\n  {gained} input files were resolved by tracing that the source never named")
    report(f"  {lost} files named in source were NOT opened -- branches that did not run, or the")
    report("  scanner reading a write as a read. Traces are a LOWER bound on one execution.")
    report(f"\n  BIGGEST GAINS -- files opened that no static reader could have found")
    report(f"    {'module':<40}{'new':>5}  examples")
    for r in sorted(detail, key=lambda x: -x["n_new"])[:12]:
        report(f"    {Path(r['module']).name:<40}{r['n_new']:>5}  {', '.join(r['new'][:3])[:46]}")

    nkeys = sum(len(v) for r in recs for v in r["keys"].values())
    withkeys = [r for r in recs if r["keys"]]
    report(f"\n  {nkeys} field accesses observed across {len(withkeys)} modules -- measured at runtime,")
    report("  not approximated from the AST, so a key reached through any indirection is caught.")

    report(f"\n  WHY MODULES DID NOT RUN -- the ceiling this approach actually has")
    kinds = {}
    for r in err + to:
        k = r["status"].split(":")[0][:40]
        kinds[k] = kinds.get(k, 0) + 1
    for k, n in sorted(kinds.items(), key=lambda x: -x[1])[:10]:
        report(f"    {k:<48}{n:>4}")

    res = {"test": "trace_all", "n_modules": len(mods), "ok": len(ok), "raised": len(err),
           "timeout": len(to), "traced_any": len(traced_any), "gained": gained, "lost": lost,
           "records": recs, "log": log}
    dest = SCRATCH / "trace_all.json"
    json.dump(res, open(dest, "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {dest}  (copied into outputs/ after the restore)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
