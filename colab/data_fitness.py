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


LOADERS = {"load", "read_csv", "read_parquet", "read_json", "read_table", "loadtxt", "read_pickle"}


def data_names(tree):
    """Variables that hold LOADED data, and the ones derived from them by iteration.

    Without this the field match is dominated by dicts the module built itself. train_tier1_xgboost was
    reported at 0% on keys like fold_mcc_mean and failure_mode -- its own RESULT fields, which of course
    appear in no input. Scoping the question to names that actually came from a load is the difference
    between a stale-schema detector and a list of everybody's output vocabulary.

    Two passes rather than a fixpoint: d -> v in d.items() -> w in v.items() is as deep as this repo goes,
    and an unbounded walk would start following names that are no longer the data.
    """
    names = set()
    for _ in range(2):
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                tgt = [t.id for t in n.targets if isinstance(t, ast.Name)]
                v = n.value
                if isinstance(v, ast.Call):
                    f = v.func
                    nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                    if nm in LOADERS:
                        names.update(tgt)
                elif isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name) \
                        and v.value.id in names:
                    names.update(tgt)
            elif isinstance(n, ast.For):
                it = n.iter
                base = None
                if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) \
                        and it.func.attr in ("items", "values"):
                    base = it.func.value
                elif isinstance(it, ast.Name):
                    base = it
                if isinstance(base, ast.Name) and base.id in names:
                    t = n.target
                    if isinstance(t, ast.Name):
                        names.add(t.id)
                    elif isinstance(t, ast.Tuple):
                        names.update(e.id for e in t.elts if isinstance(e, ast.Name))
    return names


def accessed_data(tree):
    """Field names reached for ON LOADED DATA specifically -- the scoped version of accessed()."""
    names = data_names(tree)
    ks = set()
    for n in ast.walk(tree):
        base = None
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) \
                and isinstance(n.slice.value, str):
            base, k = n.value, n.slice.value
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" \
                and n.args and isinstance(n.args[0], ast.Constant) \
                and isinstance(n.args[0].value, str):
            base, k = n.func.value, n.args[0].value
        if base is None:
            continue
        while isinstance(base, (ast.Subscript, ast.Attribute)):
            base = base.value
        if isinstance(base, ast.Name) and base.id in names and len(k) > 1:
            ks.add(k)
    return ks


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
        scoped = accessed_data(tree)          # keys reached for ON LOADED DATA only
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
                     "frac": len(acc & voc) / len(acc),
                     "n_scoped": len(scoped), "scoped_hit": len(scoped & voc),
                     "scoped_frac": (len(scoped & voc) / len(scoped)) if scoped else None,
                     "scoped_missing": sorted(scoped - voc)[:10], "inputs": [p.name for p in ins],
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

    sc = [r for r in rows if r["scoped_frac"] is not None]
    sc_mean = float(np.mean([r["scoped_frac"] for r in sc])) if sc else 0.0
    report(f"\n  SCOPED TO LOADED DATA -- keys reached for on a variable that came from a load")
    report(f"    {len(sc)} of {len(rows)} modules have any such key")
    report(f"    {'all keys in the file':<34}{true_m:>8.1%}")
    report(f"    {'keys on loaded data only':<34}{sc_mean:>8.1%}")
    report("    The unscoped number is dragged down by dicts the module BUILT -- result fields like")
    report("    fold_mcc_mean that appear in no input because they are outputs. Scoping is what turns")
    report("    this from everyone's output vocabulary into a stale-schema question.")

    worst = sorted([r for r in sc if r["n_scoped"] >= 4], key=lambda x: x["scoped_frac"])[:10]
    report(f"\n  STALE-SCHEMA CANDIDATES -- loaded-data fields its inputs do not offer")
    report(f"    {'module':<36}{'match':>7}{'keys':>6}  missing from every input")
    for r in worst:
        report(f"    {Path(r['module']).name:<36}{r['scoped_frac']:>7.0%}{r['n_scoped']:>6}  "
               f"{', '.join(r['scoped_missing'][:4])[:46]}")

    man = RM.manifest(inputs=[], available=len(inv), used=len(rows), selection="filtered", seed=SEED,
                      controls=["permuted module-dataset pairing"],
                      note="modules with both named fields and a locatable input")
    report("\n  THIS TOOL'S OWN MANIFEST")
    RM.report(man, emit=report)

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "data_fitness", "manifest": man, "true_match": true_m, "permuted_match": perm_m,
               "scoped_match": sc_mean,
               "lift": lift, "sees_schema": bool(sees), "n_modules": len(rows),
               "n_datasets": len(all_ds), "rows": rows, "log": log},
              open(OUT / "data_fitness.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'data_fitness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
