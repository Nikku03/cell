"""THE DATA MAP OF THREE SUBSYSTEMS, BEFORE AND AFTER TRACING.

BEFORE is what the source text says a module reads: string literals that look like filenames, found by
walking the AST. AFTER is what the process actually opened, recorded by wrapping open/json.load/np.load and
the pandas readers for the duration of the run.

The two differ because a path can be built rather than written. pd.read_parquet(UP), f"ko_dossier_{ko}.json"
and a hash-prefixed cache name are all invisible to the first and exact in the second. The gap between the
columns is the part of this repo's data flow that no amount of static analysis could have recovered.

WHAT EACH COLUMN CAN AND CANNOT SUPPORT
    before   complete for the modules that hardcode their paths, silently partial for the rest, and there
             is no way to tell which from the text alone.
    after    exact for the run that happened, and a LOWER bound on every other run. A branch that did not
             execute contributes nothing, so a module that reads a file only under a flag will show fewer
             inputs here than it can actually touch.
    neither is a superset of the other, and the map prints both rather than merging them.

-> outputs/subsystem_map.json
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
DATA_EXT = {"json", "csv", "npz", "tsv", "parquet", "npy", "txt", "pkl", "gz", "h5", "h5ad", "fa",
            "fasta", "pdb", "xml"}
SUBSYS = {
    "cell model":   lambda m: m.startswith("colab/cell_"),
    "nexus":        lambda m: m.startswith("colab/nexus"),
    "4d chromatin": lambda m: m.startswith("colab/chromatin"),
}


def is_data(p):
    return "/site-packages/" not in p and "/home/user/cell/" in p \
        and Path(p).suffix.lstrip(".").lower() in DATA_EXT


def main():
    log = []

    def report(x):
        print(x, flush=True)
        log.append(x)

    inv = {r["module"]: r for r in json.load(open(OUT / "data_doctrine.json"))["inventory"]}
    tr = {r["module"]: r for r in json.load(open(OUT / "trace_all.json"))["records"]}

    report("=" * 100)
    report("DATA MAP -- BEFORE (what the source names) vs AFTER (what the run opened)")
    report("=" * 100)
    report("  BEFORE walks the AST for filename literals. AFTER wraps open/json.load/np.load/pandas and")
    report("  records every path the process actually touched. A built path -- pd.read_parquet(UP),")
    report("  f\"ko_dossier_{ko}.json\", a hash-prefixed cache -- is invisible to one and exact in the other.")

    out = {}
    for name, pred in SUBSYS.items():
        mods = sorted(m for m in inv if pred(m))
        rows = []
        for m in mods:
            before = {Path(x).name for x in inv[m].get("reads", [])}
            t = tr.get(m)
            after = {Path(p).name for p in t["inputs"] if is_data(p)} if t else set()
            keys = sum(len(v) for v in t["keys"].values()) if t else 0
            rows.append({"module": m, "before": sorted(before), "after": sorted(after),
                         "n_before": len(before), "n_after": len(after),
                         "status": (t or {}).get("status", "not traced"), "keys": keys})
        traced = [r for r in rows if r["status"] != "not traced"]
        ran = [r for r in traced if r["status"] == "ok"]
        b_all = set().union(*[set(r["before"]) for r in rows]) if rows else set()
        a_all = set().union(*[set(r["after"]) for r in rows]) if rows else set()
        gained = a_all - b_all
        lost = b_all - a_all

        report(f"\n  {'='*94}")
        report(f"  {name.upper()}   {len(mods)} modules, {len(traced)} traced, {len(ran)} ran clean")
        report(f"  {'='*94}")
        report(f"    {'module':<38}{'before':>8}{'after':>7}{'keys':>8}  status")
        for r in rows:
            if not (r["n_before"] or r["n_after"] or r["status"] not in ("not traced",)):
                continue
            st = r["status"] if r["status"] in ("ok", "not traced") else r["status"].split(":")[0][:22]
            report(f"    {Path(r['module']).name:<38}{r['n_before']:>8}{r['n_after']:>7}"
                   f"{r['keys']:>8}  {st}")
        report(f"    {'-'*90}")
        report(f"    {'DISTINCT DATASETS':<38}{len(b_all):>8}{len(a_all):>7}")
        report(f"    files the source never named ....... {len(gained)}")
        if gained:
            report(f"      {', '.join(sorted(gained)[:8])[:86]}")
        report(f"    named but not opened in this run ... {len(lost)}")
        if lost:
            report(f"      {', '.join(sorted(lost)[:8])[:86]}")
        out[name] = {"modules": len(mods), "traced": len(traced), "ran_clean": len(ran),
                     "before_datasets": sorted(b_all), "after_datasets": sorted(a_all),
                     "gained": sorted(gained), "lost": sorted(lost), "rows": rows}

    report(f"\n  {'='*94}")
    report(f"  SUMMARY")
    report(f"  {'='*94}")
    report(f"    {'subsystem':<18}{'mods':>6}{'traced':>8}{'clean':>7}{'before':>9}{'after':>8}{'new':>6}")
    for name, d in out.items():
        report(f"    {name:<18}{d['modules']:>6}{d['traced']:>8}{d['ran_clean']:>7}"
               f"{len(d['before_datasets']):>9}{len(d['after_datasets']):>8}{len(d['gained']):>6}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "subsystem_map", "subsystems": out, "log": log},
              open(OUT / "subsystem_map.json", "w"), indent=2)
    report(f"\n  -> {OUT/'subsystem_map.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
