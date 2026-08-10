"""THE DATA DOCTRINE -- seven rules about the dataset, applied to every module in the repo.

WHY A SECOND VOCABULARY.  physics_reduce has seven structural properties, each with a decisive test and a
price. This is the same idea pointed at the other half of the problem. A reduction that holds on the wrong
dataset buys nothing, and a model measured on a set it was fitted to is not measured at all. So: seven
rules about DATA, each with a check, run over every module in the repository.

    1  NAMED SOURCE          the module names every dataset it reads. What cannot be named cannot be
                             checked, and a module that loads a path assembled at runtime is opaque to
                             every rule below it.
    2  DEDICATED SET         the set a module SCORES on is not one that other modules FIT on. This is the
                             only rule here computed from the code graph rather than from prose: who reads
                             what is a fact about the source, not a claim in a docstring.
    3  HELD OUT              the split exists and is declared, rather than the score being reported on
                             everything the module could reach.
    4  FIT FOR THE QUESTION  the module says what it is predicting. A file that reads three tables and
                             never names its target cannot be checked for whether the target is in them.
    5  CONTROL POPULATION    something to compare against -- shuffled labels, a matched set, a baseline.
                             A number with no comparison is not a result, and this repo has paid for that
                             lesson repeatedly.
    6  COVERAGE DECLARED     how much of the data was actually used. Silent subsetting -- a top-N, a
                             dropna, a first-1400 prefix -- turns a sample into a convenience sample, and
                             the fill worklist in this repo was 100% one source for exactly that reason.
    7  PROVENANCE PINNED     a seed, a version, a hash, an accession. Without one a rerun is a different
                             experiment wearing the same name.

WHAT IS STRONG AND WHAT IS NOT.  Rule 2 is structural: it comes from parsing every load call in the repo
and building the module-to-dataset graph. Rules 1 and 4 are structural in part. Rules 3, 5, 6 and 7 are
TEXT checks and are weak in exactly the way the previous audit's were -- they find modules that use the
vocabulary, and will miss a module that does the right thing in its own words. They are reported apart and
never summed with rule 2.

THE CONTROL.  The previous audit's first control was useless and it said so. This one uses two:
    age split      newest third against oldest third of modules that read data. If the analyser sees
                   anything, later work should score higher.
    no-data set    modules that read NO dataset at all. They must score near zero on the data rules --
                   if a module with no data scores like one with data, the checks are matching prose
                   rather than practice, and the whole table goes in the bin.

PREDECLARED, before any number:
    the dedication graph shows most datasets read by exactly one module
        -> sets are dedicated, and rule 2 is largely satisfied across the repo.
    a few datasets are read by very many modules
        -> those are the shared surfaces where a fit-then-score mistake can hide, and they are the list
           worth having. Being shared is not itself an error -- a reference table SHOULD be shared.
    the no-data control scores like the data modules
        -> the text checks are matching prose. Report it and discard them.

-> outputs/data_doctrine.json
"""
import ast
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))

# THE FIT-AND-SCORE PAIR. A module that both fits something and scores something, on a dataset many other
# modules also read, is where a leakage mistake can hide. These are CANDIDATES for hand-checking, not
# findings -- fitting and scoring in one file is completely correct when the split is honest.
FITS = re.compile(r"\.fit\(|train|regress|LogisticRegression|RandomForest|GradientBoost|coef_|lstsq"
                  r"|curve_fit|minimize\(", re.I)
SCORES = re.compile(r"accuracy|roc_auc|\bAUC\b|precision|recall|\br2\b|spearman|pearson|score\(", re.I)

READERS = {"load", "read_csv", "read_parquet", "read_text", "read_json", "loadtxt", "read_pickle",
           "read_table", "read_excel", "load_npz", "read_bytes"}
WRITERS = {"dump", "dumps", "savez", "savez_compressed", "save", "to_csv", "write_text", "savefig",
           "to_parquet", "write_bytes"}
DATAFILE = re.compile(r"^[\w./-]+\.(json|csv|npz|tsv|parquet|txt|fa|fasta|xml|pkl|h5|gz|mat|npy)$")

TEXT_RULES = {
    "heldout":    re.compile(r"held[- ]out|holdout|train/test|test set|validation set|leave-one-out"
                             r"|\bLOCO\b|split", re.I),
    "control":    re.compile(r"\bcontrol\b|\bshuffl|\bpermut|null model|scrambl|baseline|matched", re.I),
    "coverage":   re.compile(r"coverage|dropna|how many .{0,20}(used|kept)|n\s*=\s*\d|subsample"
                             r"|top[- ]?\d+|first \d+", re.I),
    "provenance": re.compile(r"SEED|seed=|version|accession|\bhash\b|checksum|downloaded|release|\bDOI\b"
                             r"|GEO|ENCODE", re.I),
    "target":     re.compile(r"predict|classif|estimat|infer|score|accuracy|AUC|correlat|recover", re.I),
}


def _sh(*a):
    try:
        return subprocess.run(a, cwd=ROOT, capture_output=True, text=True, timeout=90).stdout.strip()
    except Exception:
        return ""


def _strings_under(node):
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and DATAFILE.match(n.value):
            out.append(n.value)
    return out


def _fname(call):
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def inventory(path):
    """What the module TAKES, what it DOES, and what it PRODUCES -- read off the source, not the prose."""
    rel = path.relative_to(ROOT).as_posix()
    try:
        src = path.read_text(errors="ignore")
        tree = ast.parse(src)
    except Exception:
        return None
    doc = (ast.get_docstring(tree) or "").strip()
    reads, writes = set(), set()
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        name = _fname(n)
        strs = _strings_under(n)
        if not strs:
            continue
        mode_w = any(isinstance(a, ast.Constant) and isinstance(a.value, str)
                     and a.value.startswith(("w", "a")) for a in n.args)
        if name in WRITERS or (name == "open" and mode_w):
            writes.update(strs)
        elif name in READERS or name == "open":
            reads.update(strs)
    # a module's own declared output is a write even when it is assembled elsewhere
    m = re.search(r"->\s*(outputs/[\w./-]+\.json)", src[:6000])
    if m:
        writes.add(Path(m.group(1)).name)
    reads -= writes
    head = src[:8000]
    # OUTPUT-CLASS: does the recorded run carry a manifest? This is the whole point of the rewrite --
    # prose lost its credibility twice, so the question becomes what the artefact contains.
    man_present = man_complete = None
    if m:
        p = ROOT / m.group(1)
        if p.exists():
            try:
                man_present, man_complete, _ = RM.check(json.load(open(p)))
            except Exception:
                man_present = man_complete = False
    return {
        "module": rel,
        "function": (doc.splitlines() or [""])[0][:130],
        "reads": sorted(reads),
        "writes": sorted(writes),
        "loc": src.count("\n") + 1,
        "text": {k: bool(v.search(head)) for k, v in TEXT_RULES.items()},
        "names_source": bool(reads),
        "has_docstring": bool(doc),
        "declared_output": m.group(1) if m else None,
        "manifest_present": man_present,
        "manifest_complete": man_complete,
        "fits_and_scores": bool(FITS.search(src) and SCORES.search(src)),
    }


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("THE DATA DOCTRINE -- seven rules about the dataset, over every module in the repo")
    report("=" * 100)
    report("  A reduction that holds on the wrong dataset buys nothing, and a model measured on the set")
    report("  it was fitted to is not measured. Rule 2 is computed from the code graph -- who reads what")
    report("  is a fact about the source. Rules 3, 5, 6, 7 are TEXT and are weak in the usual way.")

    files = [p for p in sorted(ROOT.rglob("*.py"))
             if "__pycache__" not in str(p) and ".venv" not in str(p) and "/target/" not in str(p)]
    inv = [r for r in (inventory(p) for p in files) if r]
    report(f"\n  {len(inv)} modules parsed out of {len(files)} python files")

    with_data = [r for r in inv if r["reads"]]
    no_data = [r for r in inv if not r["reads"]]
    report(f"  {len(with_data)} read at least one named dataset; {len(no_data)} read none")

    # ---- RULE 2, the structural one: who reads what --------------------------------------------------
    users = defaultdict(set)
    for r in with_data:
        for d in r["reads"]:
            users[Path(d).name].add(r["module"])
    share = Counter({d: len(m) for d, m in users.items()})
    dedicated = [d for d, n in share.items() if n == 1]
    report(f"\n  RULE 2 -- DEDICATION, from the code graph")
    report(f"    {len(users)} distinct datasets read across the repo")
    report(f"    {len(dedicated)} are read by exactly ONE module ({len(dedicated)/max(len(users),1):.0%}) "
           f"-- dedicated by construction")
    report(f"\n    the shared surfaces, where a fit-then-score mistake can hide:")
    report(f"    {'dataset':<44}{'modules':>9}")
    for d, n in share.most_common(12):
        if n < 2:
            break
        report(f"    {d:<44}{n:>9}")
    report("    A shared dataset is not an error -- a reference table SHOULD be shared. It is the list")
    report("    to check by hand, because it is where the same rows can be fitted and scored.")

    # ---- the rest, per rule --------------------------------------------------------------------------
    def frac(rows, key):
        return sum(r["text"][key] for r in rows) / max(len(rows), 1) if rows else 0.0

    report(f"\n  THE SEVEN RULES over the {len(with_data)} modules that read data")
    rows7 = [
        ("1 NAMED SOURCE", sum(r["names_source"] for r in with_data) / max(len(with_data), 1),
         "structural"),
        ("2 DEDICATED SET", len(dedicated) / max(len(users), 1), "structural"),
        ("3 HELD OUT", frac(with_data, "heldout"), "TEXT"),
        ("4 FIT FOR THE QUESTION", frac(with_data, "target"), "TEXT"),
        ("5 CONTROL POPULATION", frac(with_data, "control"), "TEXT"),
        ("6 COVERAGE DECLARED", frac(with_data, "coverage"), "TEXT"),
        ("7 PROVENANCE PINNED", frac(with_data, "provenance"), "TEXT"),
    ]
    report(f"    {'rule':<26}{'share':>8}{'evidence':>12}")
    for nm, v, cls in rows7:
        report(f"    {nm:<26}{v:>8.0%}{cls:>12}")

    # ---- OUTPUT-class: rules 6 and 7 rebuilt on the artefact ----------------------------------------
    withrun = [r for r in inv if r["manifest_present"] is not None]
    mp = sum(bool(r["manifest_present"]) for r in withrun)
    mc = sum(bool(r["manifest_complete"]) for r in withrun)
    report(f"\n  RULES 6 AND 7 REBUILT AS OUTPUT-CLASS -- does the recorded run carry a manifest?")
    report(f"    {len(withrun)} modules declare a recorded output that exists on disk")
    report(f"    {mp} carry a manifest ({mp/max(len(withrun),1):.0%}); "
           f"{mc} declare coverage in it ({mc/max(len(withrun),1):.0%})")
    report("    This replaces the TEXT provenance rule, which was withdrawn for scoring HIGHER on")
    report("    modules that read no data (69% against 63%) -- it was matching the word SEED.")
    report("    Adoption starts near zero by construction. The number to watch is whether it rises.")

    # ---- the shared-surface leakage candidates -------------------------------------------------------
    report(f"\n  LEAKAGE CANDIDATES on the most-shared datasets -- modules that BOTH fit and score")
    report("  while reading a surface many other modules also read. Candidates for hand-check, NOT")
    report("  findings: fitting and scoring in one file is correct whenever the split is honest.")
    cands = []
    for d, n in share.most_common(6):
        if n < 5:
            break
        rs = [r for r in with_data if d in [Path(x).name for x in r["reads"]] and r["fits_and_scores"]]
        report(f"    {d:<40}{len(rs):>4} of {n:>4} readers fit AND score")
        cands += [{"dataset": d, "module": r["module"]} for r in rs]
    report(f"    {len(cands)} module-dataset pairs to check by hand, out of "
           f"{sum(n for _, n in share.most_common(6))} reader slots on those surfaces")

    # ---- CONTROL 1: modules with no data must score near zero on the text rules ----------------------
    report(f"\n  CONTROL A -- modules that read NO dataset ({len(no_data)}). If these score like the data")
    report("  modules, the text checks are matching prose rather than practice.")
    report(f"    {'rule':<26}{'with data':>11}{'no data':>10}{'':>4}")
    sepA = []
    for key, nm in (("heldout", "3 HELD OUT"), ("control", "5 CONTROL POPULATION"),
                    ("coverage", "6 COVERAGE DECLARED"), ("provenance", "7 PROVENANCE PINNED")):
        a, b = frac(with_data, key), frac(no_data, key)
        ok = a - b > 0.10
        report(f"    {nm:<26}{a:>11.0%}{b:>10.0%}   {'separates' if ok else 'DOES NOT'}")
        sepA.append(ok)

    # ---- CONTROL 2: age split -----------------------------------------------------------------------
    report(f"\n  CONTROL B -- newest against oldest third of the data-reading modules")
    dated = []
    for r in with_data:
        s = _sh("git", "log", "--diff-filter=A", "--format=%ct", "--", r["module"])
        ts = [int(x) for x in s.split() if x.isdigit()]
        if ts:
            dated.append((min(ts), r))
    dated.sort(key=lambda x: x[0])
    k = max(len(dated) // 3, 1)
    old, new = [r for _, r in dated[:k]], [r for _, r in dated[-k:]]
    report(f"    {len(dated)} dated; {k} in each third")
    report(f"    {'rule':<26}{'newest':>9}{'oldest':>9}")
    sepB = []
    for key, nm in (("heldout", "3 HELD OUT"), ("control", "5 CONTROL POPULATION"),
                    ("coverage", "6 COVERAGE DECLARED"), ("provenance", "7 PROVENANCE PINNED")):
        a, b = frac(new, key), frac(old, key)
        ok = a - b > 0.10
        report(f"    {nm:<26}{a:>9.0%}{b:>9.0%}   {'separates' if ok else 'flat'}")
        sepB.append(ok)

    valid = sum(sepA) >= 2
    report("")
    if valid:
        report("    The no-data control separates, so the text rules are tracking something real about")
        report("    the modules rather than matching prose everywhere. They remain weak evidence.")
    else:
        report("    THE NO-DATA CONTROL DOES NOT SEPARATE. Modules that read no dataset score like the")
        report("    ones that do, so rules 3-7 are matching prose and every text number above should be")
        report("    discarded. Rule 2 stands regardless -- it is computed from the code graph.")

    # ---- the inventory the request asked for ---------------------------------------------------------
    report(f"\n  THE HEAVIEST DATA CONSUMERS -- module, what it does, how many datasets")
    report(f"    {'module':<40}{'in':>4}{'out':>5}  function")
    for r in sorted(with_data, key=lambda x: -len(x["reads"]))[:15]:
        report(f"    {Path(r['module']).name:<40}{len(r['reads']):>4}{len(r['writes']):>5}  "
               f"{r['function'][:60]}")
    nodoc = [r for r in with_data if not r["has_docstring"]]
    report(f"\n    {len(nodoc)} of {len(with_data)} data-reading modules have NO docstring, so rule 4")
    report("    cannot be checked on them at all -- they do not say what they are for.")

    # THE TOOL DECLARES ITS OWN COVERAGE. It reads 900-odd files and grades everyone else on whether they
    # say how much of their data they used; it would be absurd for it not to say how much of the repo it
    # scanned, and under what rule.
    man = RM.manifest(inputs=[], available=len(files), used=len(inv), selection="filtered", seed=None,
                      controls=["no-data modules", "age split"],
                      note="every *.py outside __pycache__/.venv/target; 'used' is those that parsed")
    report("\n  THIS TOOL'S OWN MANIFEST")
    RM.report(man, emit=report)

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "data_doctrine", "manifest": man, "leak_candidates": cands,
               "n_modules": len(inv), "n_with_data": len(with_data),
               "rules": {nm: v for nm, v, _ in rows7}, "text_valid": bool(valid),
               "dataset_users": {d: sorted(m) for d, m in users.items()},
               "share": dict(share), "inventory": inv, "log": log},
              open(OUT / "data_doctrine.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'data_doctrine.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
