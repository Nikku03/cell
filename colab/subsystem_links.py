"""WHAT ALREADY CONNECTS CELL MODEL, NEXUS AND 4D CHROMATIN -- and what could.

THE QUESTION.  Three subsystems are to be joined into one cell model. Before designing that, it is worth
knowing what is ALREADY joined, because a link that exists in code is evidence that the interface works,
and a link that does not exist may be missing for a reason rather than by oversight.

TWO KINDS OF LINK, MEASURED SEPARATELY
    code    module A imports module B. A hard dependency: B's functions run inside A.
    data    module A writes a file that module B reads. A soft dependency, and the one that actually
            carries scientific content between subsystems.

Both are read off the repository rather than asserted. The import graph comes from the AST. The data graph
comes from writes and reads -- static where the path is a literal, and from the runtime traces where it is
not, since 527 repo files were only resolvable that way.

WHAT A MISSING EDGE MEANS, AND WHAT IT DOES NOT.  An absent link is not automatically an opportunity. It
can mean the two subsystems operate at scales that do not meet -- and that is a real finding for a project
whose whole difficulty is spanning scales from a base pair to a cell. So every absent edge is reported
with the scale gap that would have to be crossed, not as a to-do item.

PREDECLARED, before any number:
    all three subsystems are already connected
        -> integration is a matter of tightening interfaces that exist, and the work is engineering.
    exactly one pair is connected
        -> that pair is the spine; the third subsystem is the integration problem and should be planned
           around rather than assumed.
    none are connected
        -> the three lines were developed independently and the "complete cell model" is a new build,
           not an assembly. That is worth knowing before committing to it.

-> outputs/subsystem_links.json
"""
import ast
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))

SUB = [
    ("cell model",   lambda m: m.startswith("colab/cell_") or m in ("colab/cellos.py",)),
    ("nexus",        lambda m: m.startswith("colab/nexus")),
    ("4d chromatin", lambda m: m.startswith("colab/chromatin") or m.startswith("colab/physics_")),
]


def classify(mod):
    for name, pred in SUB:
        if pred(mod):
            return name
    return None


def local_imports(path):
    """Modules in this repo that `path` imports. Only sibling modules count -- a shared numpy import is
    not a link between subsystems."""
    try:
        tree = ast.parse(Path(ROOT / path).read_text(errors="ignore"))
    except Exception:
        return set()
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            names.add(n.module.split(".")[0])
        elif isinstance(n, ast.Import):
            for a in n.names:
                names.add(a.name.split(".")[0])
    return names


def main():
    log = []

    def report(x):
        print(x, flush=True)
        log.append(x)

    inv = {r["module"]: r for r in json.load(open(OUT / "data_doctrine.json"))["inventory"]}
    tr = {r["module"]: r for r in json.load(open(OUT / "trace_all.json"))["records"]}
    mods = {m: classify(m) for m in inv}
    members = defaultdict(list)
    for m, s in mods.items():
        if s:
            members[s].append(m)

    report("=" * 100)
    report("WHAT ALREADY CONNECTS CELL MODEL, NEXUS AND 4D CHROMATIN")
    report("=" * 100)
    for s, ms in members.items():
        report(f"  {s:<16}{len(ms)} modules")

    stem = {Path(m).stem: m for m in inv}

    # ---- CODE LINKS ----------------------------------------------------------------------------------
    code = defaultdict(list)
    for m, s in mods.items():
        if not s:
            continue
        for name in local_imports(m):
            t = stem.get(name)
            if not t:
                continue
            ts = mods.get(t)
            if ts and ts != s:
                code[(s, ts)].append((m, t))
    report(f"\n  CODE LINKS -- module A imports module B, so B's functions run inside A")
    if not code:
        report("    NONE. No module in any of the three imports a module from another.")
    for (a, b), pairs in sorted(code.items()):
        report(f"    {a}  ->  {b}   ({len(pairs)})")
        for m, t in sorted(set(pairs))[:6]:
            report(f"      {Path(m).name:<34} imports  {Path(t).name}")

    # ---- DATA LINKS ----------------------------------------------------------------------------------
    def writes_of(m):
        w = {Path(x).name for x in inv[m].get("writes", [])}
        if m in tr:
            w |= {Path(x).name for x in tr[m].get("writes", [])}
        return w

    def reads_of(m):
        r = {Path(x).name for x in inv[m].get("reads", [])}
        if m in tr:
            r |= {Path(x).name for x in tr[m].get("inputs", [])}
        return r

    prod = defaultdict(set)
    for m, s in mods.items():
        if s:
            for f in writes_of(m):
                prod[f].add(s)
    data = defaultdict(set)
    detail = defaultdict(list)
    for m, s in mods.items():
        if not s:
            continue
        for f in reads_of(m):
            for src in prod.get(f, ()):
                if src != s:
                    data[(src, s)].add(f)
                    detail[(src, s)].append((f, m))
    report(f"\n  DATA LINKS -- A writes a file B reads (static paths + runtime traces)")
    if not data:
        report("    NONE.")
    for (a, b), files in sorted(data.items()):
        report(f"    {a}  ->  {b}   ({len(files)} files)")
        for f in sorted(files)[:6]:
            who = sorted({Path(x[1]).name for x in detail[(a, b)] if x[0] == f})[:3]
            report(f"      {f:<38} read by {', '.join(who)[:44]}")

    # ---- THE MATRIX ----------------------------------------------------------------------------------
    names = [s for s, _ in SUB]
    report(f"\n  CONNECTION MATRIX   (c = code link, d = data link, . = none)")
    hdr = "from / to"
    report(f"    {hdr:<16}" + "".join(f"{n[:12]:>15}" for n in names))
    edges = 0
    for a in names:
        row = f"    {a:<16}"
        for b in names:
            if a == b:
                row += f"{'--':>15}"
                continue
            mark = ("c" if (a, b) in code else "") + ("d" if (a, b) in data else "")
            edges += bool(mark)
            row += f"{(mark or '.'):>15}"
        report(row)

    pairs = [(a, b) for a in names for b in names if a != b]
    connected = {frozenset((a, b)) for a, b in pairs if (a, b) in code or (a, b) in data}
    report(f"\n  {len(connected)} of 3 subsystem PAIRS are connected in at least one direction")
    for p in sorted(map(tuple, map(sorted, connected))):
        report(f"    {p[0]}  <->  {p[1]}")
    missing = [tuple(sorted(p)) for p in
               ({frozenset((a, b)) for a, b in pairs} - connected)]
    for p in sorted(missing):
        report(f"    NOT CONNECTED:  {p[0]}  <->  {p[1]}")

    res = {"test": "subsystem_links",
           "members": {k: sorted(v) for k, v in members.items()},
           "code_links": {f"{a} -> {b}": sorted(set(v)) for (a, b), v in code.items()},
           "data_links": {f"{a} -> {b}": sorted(v) for (a, b), v in data.items()},
           "connected_pairs": [list(p) for p in sorted(map(tuple, map(sorted, connected)))],
           "missing_pairs": [list(p) for p in sorted(missing)], "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "subsystem_links.json", "w"), indent=2)
    report(f"\n  -> {OUT/'subsystem_links.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
