"""RULE 4 MADE REAL -- is the field the module reads actually IN the file it opens?

THE GAP THIS CLOSES.  Six of the seven data rules now rest on something. Rule 4, "fit for the question",
did not: it searched module docstrings for words like predict and classify and scored 93%, which is a
measure of how people write, not of whether the data supports the question. Saying so was honest; leaving
it there was not.

WHAT CAN ACTUALLY BE CHECKED WITHOUT TOUCHING 442 MODULES.  A module names its fields in the source --
d["kcat_tier"], v.get("incell_rate_per_s") -- and the dataset it loads has a key vocabulary that can be
read off the file. So the question becomes mechanical:

    of the fields a module reaches for, how many exist in the data it opens?

That needs no retrofit, no manifest, and no cooperation from the module. It catches a stale schema after a
field is renamed, a module reading a file it does not use, and a question asked of data that cannot answer
it. It does NOT catch a field that exists but means something different, and nothing static will.

THE CONTROL THAT DECIDES WHETHER THIS IS WORTH ANYTHING.  Field names are English. "name", "id", "score",
"gene" appear in every schema in this repository, so a module's keys will partly match ANY dataset,
including one it has never read. If the true module-to-dataset pairing matches no better than a random
pairing, this check is measuring vocabulary overlap and must be thrown out exactly like the provenance
rule was. So every number here is reported against a PERMUTED pairing computed the same way.

PREDECLARED, before any number:
    true pairing matches substantially better than permuted
        -> the check sees schema, and the per-module numbers mean something.
    true and permuted match equally
        -> it measures English. Report that, discard rule 4's numbers, and say the rule remains open --
           which is a better outcome than a 93% that meant nothing.
    modules that open a file and touch NONE of its keys
        -> a dead data dependency: either the file is loaded and unused, or it is consumed whole as an
           array. Both are worth listing; the first is a defect.
    fields reached for that exist in NO file the module opens
        -> stale-schema candidates. Expected to be noisy, because a module also subscripts dicts it
           built itself, and that noise is why the permuted control carries the argument rather than the
           raw count.

-> outputs/data_fitness.json
"""
import ast
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 5150
SAMPLE = 60           # records sampled from a table to learn its field names
MAXBYTES = 200 << 20

_VOCAB = {}


def locate(name):
    for base in ("outputs", "data", "."):
        b = ROOT / base
        if not b.exists():
            continue
        hits = sorted(b.rglob(name))
        if hits:
            return hits[0]
    return None


def vocab(path):
    """The field names a file actually offers, and WHETHER THE FILE COULD BE READ AT ALL.

    Returning an empty set for a file that failed to open is the defect that produced this check's first
    finding list. pyarrow was not installed, every .parquet came back with no fields, and 72 modules were
    reported as having 'dead data dependencies' -- almost all of them parquet readers doing nothing wrong.
    An unreadable file and a file with no fields are not the same fact and must not return the same value.
    """
    key = str(path)
    if key in _VOCAB:
        return _VOCAB[key]
    ks, ok = set(), True
    try:
        if path.stat().st_size > MAXBYTES:
            _VOCAB[key] = (ks, False)
            return _VOCAB[key]
        sfx = path.suffix.lower()
        if sfx == ".json":
            obj = json.load(open(path))
            if isinstance(obj, dict):
                ks |= set(map(str, obj.keys()))
                for v in list(obj.values())[:SAMPLE]:
                    if isinstance(v, dict):
                        ks |= set(map(str, v.keys()))
            elif isinstance(obj, list):
                for v in obj[:SAMPLE]:
                    if isinstance(v, dict):
                        ks |= set(map(str, v.keys()))
        elif sfx in (".npz", ".npy"):
            z = np.load(path, allow_pickle=True)
            names = list(z.files) if hasattr(z, "files") else []
            ks |= set(map(str, names))
            for n in names[:6]:
                try:
                    a = z[n]
                    if a.dtype == object and a.size:
                        first = a.ravel()[0]
                        if isinstance(first, dict):
                            ks |= set(map(str, first.keys()))
                    if a.dtype.names:
                        ks |= set(map(str, a.dtype.names))
                except Exception:
                    pass
        elif sfx in (".csv", ".tsv", ".txt"):
            head = open(path, errors="ignore").readline().strip()
            sep = "\t" if (sfx == ".tsv" or head.count("\t") > head.count(",")) else ","
            ks |= {c.strip().strip('"') for c in head.split(sep) if c.strip()}
        elif sfx == ".parquet":
            import pyarrow.parquet as pq
            ks |= set(map(str, pq.read_schema(path).names))
        else:
            ok = False                      # .pkl, .gz and friends are not introspected here
    except Exception:
        ok = False
    _VOCAB[key] = (ks, ok)
    return _VOCAB[key]


def _is_environ(node):
    """os.environ["CELL_OUT"] is a lookup in the process environment, not a field in a dataset. Counting
    it inflated the denominator and put ten modules at a spurious 0% field match, every one of them
    'unmatched' on CELL_OUT and CELL_SCRATCH."""
    v = node.value
    if isinstance(v, ast.Attribute) and v.attr == "environ":
        return True
    return isinstance(v, ast.Name) and v.id == "environ"


def accessed(tree):
    """Every field name the module reaches for: d["x"] and d.get("x")."""
    ks = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) \
                and isinstance(n.slice.value, str):
            if not _is_environ(n):
                ks.add(n.slice.value)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" \
                and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            if not (isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "environ"):
                ks.add(n.args[0].value)
    # environment-variable names by convention, reached for through a wrapper this check cannot see
    return {k for k in ks if k and len(k) > 1 and not k.startswith(("/", "."))
            and not (k.isupper() and k.startswith("CELL_"))}


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    report("=" * 100)
    report("RULE 4 MADE REAL -- is the field the module reads actually IN the file it opens?")
    report("=" * 100)
    report("  The old rule searched docstrings for the word 'predict' and scored 93%, which measures how")
    report("  people write. This reads the fields out of the source and the schemas out of the files.")
    report("  Field names are English, so every number is reported against a PERMUTED pairing.")

    inv = json.load(open(OUT / "data_doctrine.json"))["inventory"]
    rows, all_ds = [], {}
    for r in inv:
        if not r["reads"]:
            continue
        try:
            tree = ast.parse((ROOT / r["module"]).read_text(errors="ignore"))
        except Exception:
            continue
        acc = accessed(tree)
        if not acc:
            continue
        ins = []
        for d in r["reads"]:
            p = locate(Path(d).name)
            if p:
                ins.append(p)
                all_ds.setdefault(str(p), p)
        if not ins:
            continue
        voc, per, unread = set(), {}, []
        for p in ins:
            v, ok = vocab(p)
            if not ok:
                unread.append(p.name)      # cannot be judged; NOT the same as having no fields
                continue
            per[p.name] = len(acc & v)
            voc |= v
        if not per:
            continue                        # every input unreadable: this module cannot be scored
        rows.append({"module": r["module"], "n_keys": len(acc), "hit": len(acc & voc),
                     "frac": len(acc & voc) / len(acc), "inputs": [p.name for p in ins],
                     "per_input": per, "unreadable": unread, "unmatched": sorted(acc - voc)[:12],
                     "dead_inputs": [n for n, c in per.items() if c == 0]})
    report(f"\n  {len(rows)} modules have both named fields and a locatable input")
    report(f"  {len(all_ds)} distinct datasets introspected")

    # ---- THE CONTROL ---------------------------------------------------------------------------------
    pool = list(all_ds.values())
    perm = []
    for r in rows:
        try:
            tree = ast.parse((ROOT / r["module"]).read_text(errors="ignore"))
        except Exception:
            continue
        acc = accessed(tree)
        pick = rng.choice(len(pool), size=min(len(r["inputs"]), len(pool)), replace=False)
        voc = set()
        for i in pick:
            voc |= vocab(pool[int(i)])[0]
        perm.append(len(acc & voc) / max(len(acc), 1))
    true_m = float(np.mean([r["frac"] for r in rows]))
    perm_m = float(np.mean(perm)) if perm else 0.0
    report(f"\n  {'pairing':<26}{'mean field match':>18}")
    report(f"  {'TRUE (what it reads)':<26}{true_m:>18.1%}")
    report(f"  {'PERMUTED (random file)':<26}{perm_m:>18.1%}")
    lift = true_m / max(perm_m, 1e-9)
    sees = (true_m - perm_m) > 0.10
    report(f"  {'lift':<26}{lift:>17.2f}x")
    report("")
    if sees:
        report("    The true pairing matches substantially better than a random one, so this is reading")
        report("    schema rather than English. Rule 4's numbers below carry information.")
    else:
        report("    TRUE AND PERMUTED MATCH ALIKE. This measures vocabulary overlap, not schema. Rule 4's")
        report("    numbers are discarded and the rule stays OPEN -- which is a better outcome than the")
        report("    93% it used to report, because that number meant nothing and this one says so.")

    # ---- the findings --------------------------------------------------------------------------------
    dead = [r for r in rows if r["dead_inputs"]]
    report(f"\n  DEAD DATA DEPENDENCIES -- a file is opened and none of its fields are touched")
    report(f"    {len(dead)} of {len(rows)} modules ({len(dead)/max(len(rows),1):.0%})")
    report("    Consuming a file whole -- an array, a matrix -- looks identical to this check, so these")
    report("    are candidates. A module that loads a keyed table and reads no key from it is a defect.")
    nun = sum(len(r.get("unreadable", [])) for r in rows)
    report(f"    {nun} input slots were UNREADABLE and are excluded rather than counted as dead -- the")
    report("    first run counted them, and reported 72 dead dependencies that were mostly parquet.")
    for r in sorted(dead, key=lambda x: -len(x["dead_inputs"]))[:10]:
        report(f"      {Path(r['module']).name:<36}{', '.join(r['dead_inputs'])[:52]}")

    worst = sorted(rows, key=lambda x: x["frac"])[:10]
    report(f"\n  LOWEST FIELD MATCH -- reaches for fields its inputs do not offer")
    report(f"    {'module':<38}{'match':>7}{'keys':>6}  unmatched examples")
    for r in worst:
        report(f"    {Path(r['module']).name:<38}{r['frac']:>7.0%}{r['n_keys']:>6}  "
               f"{', '.join(r['unmatched'][:4])[:44]}")

    man = RM.manifest(inputs=[], available=len(inv), used=len(rows), selection="filtered", seed=SEED,
                      controls=["permuted module-dataset pairing"],
                      note="modules with both named fields and a locatable input")
    report("\n  THIS TOOL'S OWN MANIFEST")
    RM.report(man, emit=report)

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "data_fitness", "manifest": man, "true_match": true_m, "permuted_match": perm_m,
               "lift": lift, "sees_schema": bool(sees), "n_modules": len(rows),
               "n_datasets": len(all_ds), "rows": rows, "log": log},
              open(OUT / "data_fitness.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'data_fitness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
