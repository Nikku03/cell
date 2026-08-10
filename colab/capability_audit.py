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
    PROPAGATE   from that slot, is there a directed path through what modules ACTUALLY read and write to
                something that is a cell-level outcome? Measured on the same file-level coupling graph
                cell_model_audit built -- real edges from real reads and writes, static plus traced.
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

WHAT THIS DOES NOT DO.  It does not score quality. A question type can pass all three axes on a weak
model with a poor AUC. Passing means the question is ASKABLE, CONNECTED and FALSIFIABLE -- the minimum
for the word "answer" to mean anything. Quality is the next audit, not this one.

THE ENTRY MAPPING IS A JUDGEMENT and is written out in full below with its counts, so that anyone who
disagrees can see exactly which block was credited to which question and argue with it. That is the
point of writing it down rather than burying it in a regex.

-> outputs/capability_audit.json
"""
import collections
import json
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
    "metabolic growth": {"blocks": ["generxn", "reactions"], "note": "gene-reaction associations"},
}

# Files that carry a CELL-LEVEL OUTCOME -- the thing a question has to reach to have been answered.
PHENOTYPE_FILES = {"cell_loop.json", "nexus_txn.json", "cell_model_audit.json",
                   "integration_assessment.json"}
PHENOTYPE_WORDS = ("growth", "essential", "viab", "depend", "fitness", "proliferat")


def load_graph():
    """The module coupling graph, rebuilt exactly as cell_model_audit does: edges are real reads and
    writes, static inventory plus the runtime trace over 153 executed modules."""
    inv = {r["module"]: r for r in json.load(open(OUT / "data_doctrine.json"))["inventory"]}
    tr = {r["module"]: r for r in json.load(open(OUT / "trace_all.json"))["records"]}
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
    say("  it fail somewhere: ENCODE (is the perturbation addressable), PROPAGATE (does it reach a cell")
    say("  outcome through edges that actually exist), CHECK (is there a held-out score with a control).")

    D = json.load(open(CELL))
    inv, W, R, edges = load_graph()
    say(f"\n  graph: {len(inv)} modules, {sum(len(v) for v in edges.values())} directed edges from real "
        f"reads and writes")

    # THE RESOLUTION PROBLEM, and the fix.  The coupling graph is FILE-level, so all 42 blocks of
    # cell_complete.json collapse into one node and every question type would inherit the same paths --
    # the audit would say yes to everything, which is exactly the failure mode it exists to avoid. So
    # propagation is resolved to the BLOCK by reading module source for the block name as a literal:
    # a module uses `loops3d` if the string "loops3d" appears quoted in its code. That is measurable,
    # it can be wrong in both directions, and it is far sharper than sharing one node.
    SRC = {}
    for p in sorted((ROOT / "colab").glob("*.py")):
        try:
            SRC[p.stem] = p.read_text(errors="ignore")
        except Exception:
            pass
    cc_readers = sorted(m for m, rd in R.items() if "cell_complete.json" in rd)
    say(f"  {len(cc_readers)} modules read cell_complete.json; block-level use resolved from source of "
        f"{len(SRC)} modules")

    def users_of(blocks):
        """Modules that address one of these blocks by name."""
        out = set()
        for m, s in SRC.items():
            if any((f'"{b}"' in s or f"'{b}'" in s) for b in blocks):
                out.add(m)
        return out

    # which modules land on a cell-level outcome
    pheno_mods = set()
    for m, w in W.items():
        for f in w:
            if f in PHENOTYPE_FILES:
                pheno_mods.add(m)
    for m, r in inv.items():
        blob = " ".join(str(r.get(k, "")) for k in ("headline", "claims", "result")).lower()
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

    rows = []
    for q, desc in QUESTIONS:
        e = ENTRY[q]
        counts = {b: (len(D[b]) if b in D and hasattr(D[b], "__len__") else 0) for b in e["blocks"]}
        n = sum(counts.values())
        encode = n > 0

        # PROPAGATE: start ONLY at modules that address this block by name, then follow real edges.
        starts = users_of(e["blocks"])
        reach = (reachable(starts, edges) | starts) if starts else set()
        hits = sorted(reach & pheno_mods)
        propagate = bool(hits)

        # CHECK: a controlled, recorded result written by a module on that path
        ck = sorted(f for f in checked if Path(f).stem in reach)
        check = bool(ck)
        namers = starts

        rows.append({"question": q, "what": desc, "blocks": counts, "n_items": n,
                     "encode": encode, "propagate": propagate, "check": check,
                     "answers": bool(encode and propagate and check),
                     "emits": bool(encode), "namers": len(namers),
                     "phenotype_hits": hits[:6], "controlled": ck[:6], "note": e["note"]})

    say(f"\n  {'question':<22}{'ENCODE':>9}{'PROPAG':>9}{'CHECK':>8}   {'items':>8}  answers?")
    for r in rows:
        say(f"  {r['question']:<22}{'yes' if r['encode'] else 'NO':>9}"
            f"{'yes' if r['propagate'] else 'NO':>9}{'yes' if r['check'] else 'NO':>8}"
            f"   {r['n_items']:>8,}  {'YES' if r['answers'] else 'no'}")

    # ---- the calibration, which can void the run ------------------------------------------------------
    fold = next(r for r in rows if r["question"] == "any chromosome fold")
    grow = next(r for r in rows if r["question"] == "metabolic growth")
    say("\n  CALIBRATION -- two known answers, checked before any of the above is believed")
    say(f"    chromosome fold must NOT reach a cell outcome (subsystem_links measured zero links)")
    say(f"      -> propagate = {fold['propagate']}   {'OK' if not fold['propagate'] else 'AUDIT VOID'}")
    say(f"    metabolic growth must reach one, with a control (cell_loop closed it today)")
    say(f"      -> propagate = {grow['propagate']}, check = {grow['check']}   "
        f"{'OK' if (grow['propagate'] and grow['check']) else 'AUDIT VOID'}")
    void = bool(fold["propagate"] or not (grow["propagate"] and grow["check"]))

    real = [r for r in rows if not r["question"].startswith("metabolic")]
    emits = sum(r["emits"] for r in real)
    answers = sum(r["answers"] for r in real)
    say("\n" + "=" * 100)
    if void:
        say("  AUDIT VOID. The calibration failed, so every verdict above is unreliable and none of it")
        say("  should be quoted. The instrument is wrong before the repo is.")
        say("  Look at the block-name resolution first: if the fold blocks are named by a module that")
        say("  happens to sit upstream of a phenotype writer for unrelated reasons, the path is an")
        say("  artefact of shared files rather than shared biology, and the resolution needs to go")
        say("  finer than the block name before any row here is quoted.")
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
