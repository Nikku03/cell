"""THE PHILOSOPHER TURNED ON THE GOAL ITSELF -- "answer anything in the cell".

WHY THIS EXISTS.  The stated goal is a model that can answer any mutation, any protein change, any
chromosome fold, any drug effect, any side effect, cancer, everything. As stated it cannot fail, and a
goal that cannot fail cannot be worked towards -- every week would look like progress. The philosopher's
whole job here is to convert it into something that CAN come back negative, and then to come back
negative wherever it should.

THE THREE AXES.  A question is answered only if all three hold, and they are deliberately independent:

    ENCODE      is there a slot in the model whose keys ARE that perturbation? Not a related table -- the
                thing itself. A drug-target table encodes drugs; a gene-level abundance column does not
                encode a point mutation, because you cannot address a mutation with it.
    PIPELINE    is there a module that ADDRESSES that slot by name AND writes a recorded result? Not
                a path -- the same module, doing both. See the note below on why reachability was
                thrown out.
    CHECK       is there a recorded run that scores that path on held-out data WITH a named control? A
                number that cannot be wrong is not an answer, it is an opinion with a decimal point.

EMITS vs ANSWERS.  Almost everything here EMITS. Give the repo a gene and it will return predictions
from several modules. The distinction this audit exists to hold is that emitting requires only ENCODE,
while answering requires all three, and the gap between those two counts is the honest measure of how
far the goal is. It is reported as two numbers, not one.

THE CALIBRATION, predeclared, and the audit is void if either fails.  An instrument that says yes to
everything is not an instrument. Two known answers are checked first:

    chromosome fold -> cell outcome  MUST come back NO PATH.
        subsystem_links measured exactly zero links between the 4D chromatin line and the cell model.
        If this audit finds a path, the audit is wrong, not the repo.
    metabolic growth -> cell outcome MUST come back a PATH with a control.
        cell_loop closed that cycle today against four declared controls. If this audit cannot see it,
        the audit is blind and its negatives mean nothing.

A yes on the first or a no on the second voids the run. That is the mutation test: reintroduce a known
truth and check the check notices.

WHY REACHABILITY WAS THROWN OUT, and it is the reason this file exists in its second form.  The middle
axis was first written as a graph walk: start at the modules that name the block, follow real read/write
edges, see if a cell-level outcome is reachable. It voided on the calibration -- chromosome fold came
back with a path, when subsystem_links had measured exactly zero chromatin-to-cell-model links. The
cause is structural and worth stating: 284 modules read cell_complete.json. A single 38 MB monolith that
almost everything opens makes the graph nearly complete, so ANY block can reach ANY outcome through it.
Reachability through a shared hub is not evidence of a biological path, and the calibration is what said
so. The replacement demands the connection be carried by ONE module -- the same code must both address
the perturbation and produce a recorded result -- which no amount of hub-sharing can manufacture.

WHAT THIS DOES NOT DO.  It does not score quality. A question type can pass all three axes on a weak
model with a poor AUC. Passing means the question is ASKABLE, CONNECTED and FALSIFIABLE -- the minimum
for the word "answer" to mean anything. Quality is the next audit, not this one.

THE ENTRY MAPPING IS A JUDGEMENT and is written out in full below with its counts, so that anyone who
disagrees can see exactly which block was credited to which question and argue with it. That is the
point of writing it down rather than burying it in a regex.

WHAT HAPPENED, written after the run, unedited.

    calibration    chromosome fold -> no pipeline, no check   OK (it must fail, and it did)
                   metabolic growth -> pipeline, check        OK (it must pass, and it did)

    any point mutation    ENCODE yes(13)   PIPELINE no    CHECK no
    any protein change    ENCODE yes       PIPELINE yes   CHECK yes    <- the only one
    any chromosome fold   ENCODE yes       PIPELINE no    CHECK no
    drug effect           ENCODE yes       PIPELINE no    CHECK no
    side effect           ENCODE yes(48)   PIPELINE no    CHECK no
    cancer                ENCODE yes       PIPELINE no    CHECK no

    OF THE 6 QUESTION TYPES THE GOAL NAMES: 6 CAN BE ASKED, 1 CAN BE ANSWERED.

And the one that passes does so at the weakest admissible strength, which should be said plainly rather
than banked: `any protein change` is credited because cell_loop reads the abundance blocks and a gene
knockout is a protein change to zero. That is a real controlled pipeline, but it answers one perturbation
shape (delete a protein) and not the general question (change any protein's abundance or activity by any
amount and predict the consequence).

THE RANKING OF WHAT IS MISSING, by how far each falls short rather than by appetite:
    any point mutation   worst on every axis. 13 of 16,492 genes carry residue-level data -- 0.08%.
                         `biomarkers` looks relevant and is not: it is keyed on GENE, so it cannot be
                         handed a variant. There is no variant-addressable slot in this model at all.
    side effect          48 genes, and no compound-to-adverse-event table exists anywhere here.
    cancer               839 items encoded, nothing that runs and records.
    drug effect          4,275 drug-target entries encoded, and not one controlled recorded result.
    chromosome fold      8,467 items, the known dead link -- 4D chromatin has never once connected to
                         a cell-level outcome, which subsystem_links measured independently as zero.

-> outputs/capability_audit.json
"""
import collections
import json
import re
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"
SEED = 1904

# The six question types are the user's own words, kept verbatim so the audit is answering the goal that
# was actually set rather than one I found convenient to score well on.
QUESTIONS = [
    ("any point mutation", "a single base change anywhere in the genome"),
    ("any protein change", "abundance, modification or activity of any protein"),
    ("any chromosome fold", "a change in 3D genome organisation"),
    ("drug effect", "what a compound does to the cell"),
    ("side effect", "what it does that was not intended, elsewhere"),
    ("cancer", "driver alteration to transformed phenotype"),
    ("metabolic growth", "CALIBRATION -- must pass; closed by cell_loop today"),
]

# Which slots in the model are ADDRESSABLE BY that perturbation. A block earns a credit only if you can
# hand it the perturbation as a key and get something back.
ENTRY = {
    "any point mutation": {"blocks": ["struct"], "note": "residue-level: 13 genes carry per-residue "
                           "pathogenic/common counts. `biomarkers` holds mutation-conditioned "
                           "associations for 599 genes but is keyed on GENE, not on variant"},
    "any protein change": {"blocks": ["ppm", "abund", "ptm", "complexes"], "note": "gene-keyed"},
    "any chromosome fold": {"blocks": ["loops3d", "model4"], "note": "loop anchors and a 4D model"},
    "drug effect": {"blocks": ["drugs"], "note": "gene-keyed drug/target/action"},
    "side effect": {"blocks": ["otdis"], "note": "48 genes with disease associations; "
                    "no compound-to-adverse-event table exists here at all"},
    "cancer": {"blocks": ["biomarkers", "celltypes", "ctnames"], "note": "biomarker associations and "
               "cell-type identities; no tumour genotype-to-phenotype block"},
    # CALIBRATION. Note carefully which slot is credited: the metabolism the model actually computes on
    # is Human-GEM, a ledgered external input. cell_complete's own `generxn` is 3,355 reaction STRINGS
    # with no stoichiometry, and cell_loop never opens it. Crediting generxn here would have been the
    # convenient mapping and it would have been false.
    "metabolic growth": {"blocks": ["generxn", "reactions"], "files": ["HumanGEM.xml"],
                         "note": "the solved metabolism is Human-GEM (2,848 genes, 12,931 reactions), "
                                 "ledgered and external; cell_complete's generxn is strings"},
}
for _q in ENTRY:
    ENTRY[_q].setdefault("files", [])

# Files that carry a CELL-LEVEL OUTCOME -- the thing a question has to reach to have been answered.
PHENOTYPE_FILES = {"cell_loop.json", "nexus_txn.json", "cell_model_audit.json",
                   "integration_assessment.json"}
PHENOTYPE_WORDS = ("growth", "essential", "viab", "depend", "fitness", "proliferat")


FILE_RE = re.compile(r"""["']([A-Za-z0-9_.\-]+\.(?:json|csv|tsv|parquet|npz|npy|gz|xml|bed|bw|pkl))["']""")
ARROW_RE = re.compile(r"->\s*(?:outputs/)?([A-Za-z0-9_.\-/]+\.json)")


def scan_module(path, text):
    """Reads and writes for a module the inventory has never seen.

    The inventory in data_doctrine.json is a SNAPSHOT and every module written after it is invisible to
    any graph built from it -- including cell_loop, which is the calibration this audit depends on. A
    stale node list silently turns 'no path' into 'no data', and those must not look alike. So modules
    missing from the inventory are scanned here: the docstring's `-> outputs/x.json` line is the write,
    every other quoted data filename is a read."""
    writes = set(Path(m).name for m in ARROW_RE.findall(text))
    reads = set(FILE_RE.findall(text)) - writes
    return {"module": path, "reads": sorted(reads), "writes": sorted(writes),
            "function": (text.split('"""')[1].strip().splitlines() or [""])[0] if '"""' in text else "",
            "SCANNED": True}


def load_graph():
    """The module coupling graph, rebuilt exactly as cell_model_audit does: edges are real reads and
    writes, static inventory plus the runtime trace over 153 executed modules -- plus a static scan of
    any module the inventory predates."""
    inv = {r["module"]: r for r in json.load(open(OUT / "data_doctrine.json"))["inventory"]}
    tr = {r["module"]: r for r in json.load(open(OUT / "trace_all.json"))["records"]}
    fresh = 0
    for p in sorted((ROOT / "colab").glob("*.py")):
        rel = str(p.relative_to(ROOT))
        if rel not in inv:
            inv[rel] = scan_module(rel, p.read_text(errors="ignore"))
            fresh += 1
    load_graph.fresh = fresh
    W, R = collections.defaultdict(set), collections.defaultdict(set)
    for m, r in inv.items():
        w = {Path(x).name for x in r.get("writes", [])}
        rd = {Path(x).name for x in r.get("reads", [])}
        if m in tr:
            w |= {Path(x).name for x in tr[m].get("writes", [])}
            rd |= {Path(x).name for x in tr[m].get("inputs", []) if "/home/user/cell/" in x}
        W[m], R[m] = w, rd
    prod = collections.defaultdict(set)
    for m, w in W.items():
        for f in w:
            prod[f].add(m)
    edges = collections.defaultdict(set)
    for m, rd in R.items():
        for f in rd:
            for p in prod.get(f, ()):
                if p != m:
                    edges[p].add(m)
    return inv, W, R, edges


def reachable(starts, edges):
    seen, work = set(starts), list(starts)
    while work:
        v = work.pop()
        for w in edges.get(v, ()):
            if w not in seen:
                seen.add(w)
                work.append(w)
    return seen


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    say("=" * 100)
    say('THE PHILOSOPHER TURNED ON THE GOAL -- "answer anything in the cell"')
    say("=" * 100)
    say("  The goal as stated cannot fail, so it cannot be worked towards. Three independent axes make")
    say("")

    D = json.load(open(CELL))
    inv, W, R, edges = load_graph()
    say(f"\n  graph: {len(inv)} modules, {sum(len(v) for v in edges.values())} directed edges from real "
        f"reads and writes; {getattr(load_graph, 'fresh', 0)} of those modules postdate the inventory "
        f"snapshot and were scanned from source so they are not invisible")

    # THE RESOLUTION PROBLEM, and the fix.  The coupling graph is FILE-level, so all 42 blocks of
    # cell_complete.json collapse into one node and every question type would inherit the same paths --
    # the audit would say yes to everything, which is exactly the failure mode it exists to avoid. So
    # propagation is resolved to the BLOCK by reading module source for the block name as a literal:
    # a module uses `loops3d` if the string "loops3d" appears quoted in its code. That is measurable,
    # it can be wrong in both directions, and it is far sharper than sharing one node.
    # KEY EVERYTHING BY THE SAME NAME. The first version of this keyed source by file stem while the
    # graph keys modules by repo-relative path, so every propagation walk started from names that were
    # not nodes and returned nothing -- the audit reported NO for all seven questions and the
    # metabolic-growth calibration caught it. Same identifier or the walk is silent.
    SRC = {}
    for p in sorted((ROOT / "colab").glob("*.py")):
        try:
            SRC[str(p.relative_to(ROOT))] = p.read_text(errors="ignore")
        except Exception:
            pass
    cc_readers = sorted(m for m, rd in R.items() if "cell_complete.json" in rd)
    say(f"  {len(cc_readers)} modules read cell_complete.json; block-level use resolved from source of "
        f"{len(SRC)} modules")

    # INSTRUMENTS ARE NOT PIPELINES, and leaving them in made every question pass.
    # First run of this criterion returned YES on all seven, including the must-fail calibration. The
    # reason was self-credit: capability_audit names every block in this very file's ENTRY table, and it
    # writes a manifest-carrying result, so each question was connected to an outcome BY THE AUDIT
    # ITSELF. cell_model_audit did the same. This is the third time in this repo that a check has scored
    # its own vocabulary -- doctrine_audit searched for physics_reduce's verdict words, the PROVENANCE
    # rule matched the word SEED -- so the rule is mechanical rather than a hand-kept list: a module
    # whose INPUTS are this repository's own audit records is an instrument, not biology.
    META = {"data_doctrine.json", "trace_all.json", "doctrine_audit.json", "data_fitness.json",
            "cell_model_audit.json", "capability_audit.json", "subsystem_map.json"}

    def is_instrument(m):
        if Path(m).stem == Path(__file__).stem:
            return True
        rd = {Path(x).name for x in inv.get(m, {}).get("reads", [])}
        return bool(rd & META)

    instruments = {m for m in SRC if is_instrument(m)}

    def users_of(blocks):
        """Modules that address one of these blocks by name, EXCLUDING instruments."""
        out = set()
        for m, src in SRC.items():
            if m in instruments:
                continue
            if any((f'"{b}"' in src or f"'{b}'" in src) for b in blocks):
                out.add(m)
        return out

    # which modules land on a cell-level outcome
    pheno_mods = set()
    for m, w in W.items():
        for f in w:
            if f in PHENOTYPE_FILES:
                pheno_mods.add(m)
    # The inventory's one-line summary of what a module is FOR is stored under `function`. The first
    # version read `headline`/`claims`/`result`, none of which exist in that record, so this set had one
    # member and nothing could reach an outcome. Checked against the schema now rather than guessed.
    for m, r in inv.items():
        blob = str(r.get("function") or "").lower()
        if any(t in blob for t in PHENOTYPE_WORDS) and r.get("writes"):
            pheno_mods.add(m)
    say(f"  {len(pheno_mods)} modules write something that is a cell-level outcome")

    # the CHECK axis: recorded runs carrying a manifest with at least one named control
    checked = {}
    for p in sorted(OUT.glob("*.json")):
        try:
            o = json.load(open(p))
        except Exception:
            continue
        present, complete, _ = RM.check(o)
        if present and o["manifest"].get("controls"):
            checked[p.name] = len(o["manifest"]["controls"])
    say(f"  {len(checked)} recorded results carry a manifest with at least one named control: "
        f"{', '.join(sorted(checked))}")
    say(f"  {len(instruments)} modules are INSTRUMENTS (they read this repo's own audit records) and "
        f"are excluded from every axis, because an audit crediting itself is how the last three checks "
        f"here went wrong")

    rows = []
    for q, desc in QUESTIONS:
        e = ENTRY[q]
        counts = {b: (len(D[b]) if b in D and hasattr(D[b], "__len__") else 0) for b in e["blocks"]}
        n = sum(counts.values())
        encode = n > 0

        # PIPELINE: one module must BOTH address the slot by name AND write a recorded result. No graph
        # walk, so no credit for sharing a hub file with something that works.
        starts = users_of(e["blocks"] + e["files"])
        produced = {}
        for m in starts:
            for f in inv.get(m, {}).get("writes", []):
                if (OUT / Path(f).name).exists():
                    produced.setdefault(Path(f).name, m)
        pipeline = bool(produced)

        # CHECK: one of those recorded results must carry a manifest with a named control.
        # (The first version compared Path(f).stem against path-keyed module names and never matched --
        # cell_loop.json was sitting in `checked` and the calibration still read False.)
        ck = sorted(f for f in produced if f in checked)
        check = bool(ck)
        hits = sorted(set(starts) & pheno_mods)
        namers = starts

        rows.append({"question": q, "what": desc, "blocks": counts, "n_items": n,
                     "encode": encode, "pipeline": pipeline, "check": check,
                     "answers": bool(encode and pipeline and check),
                     "emits": bool(encode), "namers": sorted(namers)[:8],
                     "produces": sorted(produced)[:8],
                     "phenotype_hits": hits[:6], "controlled": ck[:6], "note": e["note"]})

    say(f"\n  {'question':<22}{'ENCODE':>9}{'PIPELINE':>10}{'CHECK':>8}   {'items':>8}  answers?")
    for r in rows:
        say(f"  {r['question']:<22}{'yes' if r['encode'] else 'NO':>9}"
            f"{'yes' if r['pipeline'] else 'NO':>10}{'yes' if r['check'] else 'NO':>8}"
            f"   {r['n_items']:>8,}  {'YES' if r['answers'] else 'no'}")
    say("")
    for r in rows:
        say(f"    {r['question']:<22} slots {r['blocks']}")
        say(f"    {'':<22} modules that address it: {', '.join(Path(x).stem for x in r['namers']) or 'NONE'}")
        say(f"    {'':<22} recorded results: {', '.join(r['produces']) or 'NONE'}"
            f"   controlled: {', '.join(r['controlled']) or 'NONE'}")

    # ---- the calibration, which can void the run ------------------------------------------------------
    fold = next(r for r in rows if r["question"] == "any chromosome fold")
    grow = next(r for r in rows if r["question"] == "metabolic growth")
    say("\n  CALIBRATION -- two known answers, checked before any of the above is believed")
    say(f"    chromosome fold must NOT reach a cell outcome (subsystem_links measured zero links)")
    say(f"      -> pipeline = {fold['pipeline']}, check = {fold['check']}   "
        f"{'OK' if not fold['check'] else 'AUDIT VOID'}")
    say(f"    metabolic growth must reach one, with a control (cell_loop closed it today)")
    say(f"      -> pipeline = {grow['pipeline']}, check = {grow['check']}   "
        f"{'OK' if (grow['pipeline'] and grow['check']) else 'AUDIT VOID'}")
    void = bool(fold["check"] or not (grow["pipeline"] and grow["check"]))

    real = [r for r in rows if not r["question"].startswith("metabolic")]
    emits = sum(r["emits"] for r in real)
    answers = sum(r["answers"] for r in real)
    say("\n" + "=" * 100)
    if void:
        say("  AUDIT VOID. The calibration failed, so every verdict above is unreliable and none of it")
        say("  should be quoted. The instrument is wrong before the repo is.")
        say("  Check which module was credited with the connection and what it actually writes. The")
        say("  criterion is deliberately strict -- one module addressing the slot AND recording a")
        say("  controlled result -- so a failure here is usually a naming collision, not a real path.")
    else:
        say(f"  OF THE 6 QUESTION TYPES THE GOAL NAMES: {emits} can be ASKED, {answers} can be ANSWERED.")
        say(f"  The gap between those two numbers is the distance left, and it is the honest headline.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(CELL), str(OUT / "data_doctrine.json"), str(OUT / "trace_all.json")],
                      available=len(QUESTIONS), used=len(rows), selection="all", seed=SEED,
                      controls=["chromosome-fold must-fail calibration",
                                "metabolic-growth must-pass calibration"],
                      note="entry mapping is a declared judgement, written out per question with counts")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "capability_audit", "manifest": man, "void": void, "rows": rows,
               "n_emits": emits, "n_answers": answers, "n_questions": len(real),
               "controlled_results": checked, "n_modules": len(inv),
               "n_phenotype_modules": len(pheno_mods), "log": log},
              open(OUT / "capability_audit.json", "w"), indent=2)
    say(f"\n  -> {OUT/'capability_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
