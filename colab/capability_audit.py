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

    a SYNTHETIC negative control  MUST fail on every axis.
        Its slot names a block and a file that do not exist and never will. Nothing can close it, no
        amount of progress can invalidate it, and it still tests the only thing a negative control has
        to: that this instrument is capable of saying no.

        THIS HAD TO BECOME SYNTHETIC, and the reason is worth recording. The must-fail slot held
        `chromosome fold` until loop 8 built that link, then `side effect` until loop 12 built that one.
        Both times the audit correctly voided itself; both times the fix was to move the calibration.
        The instrument's negative control was being eaten by the work it was meant to police, which is
        backwards. A calibration that has to move whenever the project succeeds is not a calibration.

        (This slot used to hold `chromosome fold`, and the swap is a record of progress rather than a
        convenience. That calibration asserted a measured fact -- subsystem_links found exactly zero
        chromatin-to-cell links -- and on 2026-08-10 loop_real_chromatin made it false by building one:
        bTMP torsion against transcription, four gates, controlled against a naked-DNA track. The audit
        then voided ITSELF, which is the correct behaviour: it was told a fact that had stopped being
        true. A must-fail calibration has to be a case that is still genuinely negative, so it moved to
        one, and chromosome fold moved to the must-pass side below.)
    metabolic growth -> cell outcome MUST come back a PATH with a control.
        cell_loop closed that cycle against five declared controls. If this audit cannot see it, the
        audit is blind and its negatives mean nothing.
    chromosome fold -> cell outcome  MUST NOW come back a PATH with a control.
        loop_real_chromatin established one on measured bTMP torsion. Same reasoning as above: an
        instrument that cannot see a link that was built and controlled is not measuring anything.

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

THE FOURTH COLUMN, added at loop 13 because the third had started to mislead.  ENCODE/PIPELINE/CHECK
ask whether a question can be GOT WRONG. After twelve loops five of six rows cleared them -- and three
of those five did so on a pipeline whose own decisive gate FAILED. `side effect` passes while its
partial correlation sits inside the null; `any point mutation` passes while its dose-response bought
-0.0016 AUC; `drug effect` passes only because loop_sideeffect happens to name the drugs block. A count
that rises when a controlled negative is recorded is measuring falsifiability, which is real and is not
the same as knowing the answer, and quoting it alone would be exactly the overstatement this file exists
to prevent.

So VERDICT reads the credited result's own gate table and reports how many of its gates passed. A row is
ANSWERED WELL only if every gate in the pipeline that answers it came back positive. Two numbers are
reported from here on and the second is the honest one.

WHAT THIS DOES NOT DO.  It does not score quality beyond that gate tally. A question type can pass all three axes on a weak
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

RE-RUN, loop 30: 6 ASKABLE, 6 CAN BE GOT WRONG, 3 ANSWERED WELL -- and the count went DOWN.
    A SUPERSEDED map was added and `cancer` is its first entry, so the row keeps its gates and loses
    its vote. loop_cancer is still 4/4 on the gates it declared; loop_survivable then re-asked its
    central claim with matched pairs instead of partialling and found essentiality at 0.4949, inside
    the null. The 0.2990 anti-correlation was over-control, so nothing certifies that row any more.
    This is the failure mode the fourth column could not catch on its own: gate booleans are written
    before a retraction exists and can never see one, so without this map a question keeps reading
    ANSWERED WELL on the strength of a number nothing stands behind. Un-believing a result now costs
    an explicit entry naming the loop that did the withdrawing, which is the right price for it.
    Also new here: loop 21 finished, and `metabolic growth` still does not clear its gates -- the
    feedback's deficit shrank from -0.0074 to -0.0013 on the fixed model without changing sign.

RE-RUN, loop 26: 6 ASKABLE, 6 CAN BE GOT WRONG, 4 ANSWERED WELL.
    `any protein change` joins the list, via loop_quantity -- and this is the first row credited to a
    pipeline that predicts a NUMBER rather than separating classes. Given a gene's mRNA abundance and
    decay rate, it predicts protein abundance at held-out R-squared 0.4332, and the loop's own
    turnover-and-dilution term is what takes it there from 0.3783.
    The other three are unchanged: chromosome fold (loop_real_chromatin), drug effect (loop_drug),
    cancer (loop_cancer).
    Two still fail and each for a recorded reason. `side effect`: the correlation was target counting.
    `metabolic growth`: loop 1's feedback did not earn its place -- being retested in loop 21 on a model
    that cannot import ATP, since the original failure was measured on one that could.

RE-RUN, loop 19: 6 ASKABLE, 6 CAN BE GOT WRONG, 3 ANSWERED WELL.
    All six question types now have a slot, a pipeline and a controlled score. Three have a pipeline
    whose every gate came back positive:
        any chromosome fold  <- loop_real_chromatin  (measured bTMP torsion vs transcription, 4/4)
        drug effect          <- loop_drug            (cytotoxic class from target essentiality, 4/4)
        cancer               <- loop_cancer          (IntOGen drivers, 4/4)
    Three do not, and each for a recorded reason: `any point mutation` (the dose-response bought
    -0.0016 AUC), `side effect` (the correlation was target counting), `metabolic growth` (the feedback
    did not earn its place and lost to a lookup).

    AND ONE CAVEAT THE VERDICT COLUMN CANNOT SEE.  It reads gate BOOLEANS, not their meaning.
    loop_cancer is 4/4 -- and its finding is an ANTI-correlation: after removing gene length and
    citation count, essentiality predicts NON-drivers at 0.7010, and the loss-of-function subset the
    model could mechanistically explain sits at 0.5073, which is chance. Four passing gates encode "the
    model can say what is not a driver". A boolean cannot carry that, and a reader who stops at the
    column will get it backwards.

RE-RUN, loop 13, with the VERDICT column: 6 ASKABLE, 5 CAN BE GOT WRONG, 1 ANSWERED WELL.
    Three things moved at once and only one of them is good news.
    `side effect` gained a slot for the first time -- SIDER 4.1, added by loop 12 -- so it clears the
        first three axes. Its own decisive gate failed: the correlation was target counting.
    `drug effect` also flipped, and it should be read sceptically: no drug pipeline was built. It clears
        the axes only because loop_sideeffect happens to name the drugs block while producing a
        controlled result about something adjacent.
    The VERDICT column then says what the first three cannot. Of the five rows that clear them, exactly
        ONE has a pipeline whose every gate came back positive: `any chromosome fold` at 4/4, on
        measured bTMP torsion. `any point mutation` sits at 2/4, `metabolic growth` at 2/4, `any protein
        change` at 9/12.
    The negative control also became synthetic here, after being invalidated twice by ordinary progress.

RE-RUN, loop 11: 6 CAN BE ASKED, 3 CAN BE ANSWERED.
    `any point mutation` moved to yes/yes/yes once this file's ENTRY row was corrected to name the slot
    loop 5 actually built -- the ClinVar VCF -- instead of only `struct`, whose 13 genes no module opens.
    SAY THE REST OF IT PLAINLY: loop_variant's own gates V3 and V4 FAILED. The dose-response bought
    -0.0016 AUC over a plain knockout and lost to LOEUF. So this row passes because the question is now
    ASKABLE, CONNECTED and FALSIFIABLE, which is exactly what these three axes measure and exactly what
    they do not measure is whether the answer is any good. Two of the three passing rows are in that
    position. A count that rises because a controlled NEGATIVE was recorded is still progress -- the
    question can now be got wrong, which it could not be this morning -- but it is not the same thing as
    knowing the answer.

RE-RUN 2026-08-10, after eight loops of work: 6 CAN BE ASKED, 2 CAN BE ANSWERED.
    `any chromosome fold` moved from NO/NO to yes/yes -- loop_real_chromatin built the pipeline on
    measured bTMP torsion against transcription rate, four gates, controlled against a naked-DNA track.
    Its item count also FELL, from 8,467 to 767, because model4 was removed from that row: loop 7 showed
    it is a functional neighbourhood and not a conformation, so 7,700 of those items were never
    chromatin. A smaller, honest number replacing a larger, wrong one is the direction this should move.
    The other four -- point mutation, drug effect, side effect, cancer -- are unchanged.

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
    ("functional neighbourhood", "NOT one of the six -- reported so model4's items are not lost"),
    ("NEGATIVE CONTROL", "CALIBRATION -- a perturbation this model has no slot for, and never will"),
    ("drug effect", "what a compound does to the cell"),
    ("side effect", "what it does that was not intended, elsewhere"),
    ("cancer", "driver alteration to transformed phenotype"),
    ("metabolic growth", "CALIBRATION -- must pass; closed by cell_loop today"),
]

# Which slots in the model are ADDRESSABLE BY that perturbation. A block earns a credit only if you can
# hand it the perturbation as a key and get something back.
ENTRY = {
    # CORRECTED after loop 5. This row used to credit only `struct` -- 13 genes of per-residue counts --
    # and it still reads NO on pipeline and check because loop_variant does not open that block. The
    # slot loop 5 actually built is ClinVar: 4.46M records, missense variants with clinical assertions,
    # joined to genes that have stoichiometry behind them. Leaving the row pointing at `struct` would
    # have kept the audit blind to work that was done.
    "any point mutation": {"blocks": ["struct"], "files": ["clinvar.vcf.gz"],
                           "note": "13 genes of residue-level counts, PLUS the ClinVar VCF that "
                                   "loop_variant consumes. `biomarkers` looks relevant and is not -- "
                                   "it is keyed on GENE, so a variant cannot be handed to it"},
    "any protein change": {"blocks": ["ppm", "abund", "ptm", "complexes"], "note": "gene-keyed"},
    # CORRECTED after loop 7. `model4` was credited here and it is NOT a fold: 8.8% of its neighbour
    # pairs are same-chromosome where a Hi-C map runs 70-90%, and 51.7% share its own functional label
    # ("transport/uptake", Reactome pathway names). Crediting it inflated this row's item count from 767
    # to 8,467 and would have let a functional-annotation result answer a chromatin question. The real
    # chromatin here is loops3d plus the ledgered bTMP-seq tracks.
    "any chromosome fold": {"blocks": ["loops3d"],
                            "files": ["GSM8523629_supercoil_rep1_hg38.bw",
                                      "GSM8523630_supercoil_rep2_hg38.bw"],
                            "note": "767 loop anchors, plus measured bTMP torsion from GSE277502; "
                                    "model4 was removed from this row because it is a functional "
                                    "neighbourhood, not a conformation"},
    "functional neighbourhood": {"blocks": ["model4"], "files": [],
                                 "note": "NOT a fold -- 8.8% cis, half its pairs sharing a functional "
                                         "label. Kept as its own row so the 7,700 items are still "
                                         "counted, under a heading that describes them"},
    "drug effect": {"blocks": ["drugs"], "note": "gene-keyed drug/target/action"},
    "side effect": {"blocks": ["otdis"], "files": ["sider_se.tsv.gz"],
                    "note": "48 genes with disease associations, PLUS SIDER 4.1 -- 1,430 compounds "
                            "with MedDRA side effects, added by loop 12. Before that there was no "
                            "compound-to-adverse-event data here at all"},
    # CORRECTED after loop 14, same reason as the point-mutation row: this listed only cell_complete
    # blocks, and the held-out label loop_cancer actually scores against is IntOGen's driver compendium.
    "cancer": {"blocks": ["biomarkers", "celltypes", "ctnames"], "files": ["intogen.zip"],
               "note": "biomarker associations and cell-type identities, PLUS IntOGen's 2024-06 driver "
                       "compendium (633 genes) added by loop 14 as the held-out label"},
    # THE PERMANENT NEGATIVE CONTROL, and the reason it had to become synthetic.
    # The must-fail calibration has now been invalidated TWICE by ordinary progress: it was `chromosome
    # fold` until loop 8 built that link, then `side effect` until loop 12 built that one. Each time the
    # audit correctly voided itself, and each time the fix was to move the calibration -- which means the
    # instrument's negative control was being consumed by the work it was meant to police. That is
    # backwards. A calibration must be permanent or it is not a calibration.
    # So the negative control is now SYNTHETIC: a question whose slot names a block that does not exist
    # and never will. Nothing can close it, no amount of progress can invalidate it, and it still tests
    # exactly what a negative control must -- that this instrument is capable of saying no.
    "NEGATIVE CONTROL": {"blocks": ["__no_such_block__"], "files": ["__no_such_file__"],
                         "note": "deliberately unsatisfiable; if this ever passes, the audit is broken"},
    # CALIBRATION. Note carefully which slot is credited: the metabolism the model actually computes on
    # is Human-GEM, a ledgered external input. cell_complete's own `generxn` is 3,355 reaction STRINGS
    # with no stoichiometry, and cell_loop never opens it. Crediting generxn here would have been the
    # convenient mapping and it would have been false.
    "metabolic growth": {"blocks": ["generxn", "reactions"], "files": ["HumanGEM.xml"],
                         "note": "the solved metabolism is Human-GEM (2,848 genes, 12,931 reactions), "
                                 "ledgered and external; cell_complete's generxn is strings"},
}
# WHICH MODULE COUNTS AS A QUESTION'S OWN PIPELINE, and why this list has to exist.
# Block-name matching credits any module that mentions a block anywhere, including as a CONTROL. That
# over-credited three times: `drug effect` via loop_sideeffect, and -- the one that made this
# unavoidable -- `any protein change` marked ANSWERED WELL via loop_cancer.json, a cancer-driver module
# credited because it uses protein abundance as a lookup. A cancer module is not an answer to "any
# protein change".
# So each question declares which module names count as ITS pipeline. This is a judgement, it is written
# out here to be argued with exactly like ENTRY is, and anything else that touches the slot is reported
# as ADJACENT rather than counted.
ACCEPTS = {
    "any point mutation": ("variant", "pervariant", "chain", "nexus_fair", "rescue", "physics_bind",
                           "fitted_bind", "distogram_nn"),
    "any protein change": ("cell_loop", "deficit", "medium", "quantity", "retest", "amplify"),
    "any chromosome fold": ("real_chromatin", "supercoil", "hic_target", "polymer",
                            "extrusion", "stiffness", "fourth", "insulation",
                            "compartment", "affinity", "surrogate", "loading",
                            "langevin"),
    "drug effect": ("drug",),
    "side effect": ("sideeffect", "shared_effect"),
    "cancer": ("cancer", "survivable"),
    "metabolic growth": ("cell_loop", "medium", "slack", "retest", "tail", "integrate",
                         "recall", "buffering"),
    "functional neighbourhood": ("fold_link", "chromatin"),
    "NEGATIVE CONTROL": (),
}

# A module here passed its own gates and had its central claim withdrawn by a later, better-controlled
# loop. It keeps its row and loses its vote. Adding to this map is the only way a result in this
# repository gets un-believed, and it requires naming the loop that did the withdrawing.
SUPERSEDED = {
    "loop_cancer.json": "loop_survivable.json -- matched pairs instead of partialling put essentiality "
                        "at 0.4949 inside the null; the 0.2990 anti-correlation was over-control",
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

        # VERDICT: read the gate table of whatever controlled result was credited to this row.
        gp = gf = 0
        detail = {}
        for f in ck:
            try:
                o = json.load(open(OUT / f))
            except Exception:
                continue
            g = o.get("gates")
            if isinstance(g, dict) and g:
                pas = sum(1 for v in g.values() if v is True)
                fal = sum(1 for v in g.values() if v is False)
                gp += pas
                gf += fal
                detail[f] = {"passed": pas, "failed": fal}
        # ANSWERED WELL means SOME pipeline answers it well, not that every module touching the slot
        # does. Aggregating across all of them marked `drug effect` poorly because loop_sideeffect
        # failed a gate, even though loop_drug is 4/4 -- penalising a question for an adjacent module's
        # honest negative. The right criterion is existential: is there a controlled pipeline all of
        # whose gates came back positive?
        acc = ACCEPTS.get(q, ())
        own = {f: v for f, v in detail.items() if any(a in f for a in acc)}
        adjacent = sorted(set(detail) - set(own))
        # A MODULE CANNOT CERTIFY A CLAIM A LATER LOOP TOOK BACK. loop_cancer is 4/4 on its own gates
        # and loop_survivable then showed its central finding -- essentiality predicting NON-drivers at
        # 0.2990 -- was manufactured by over-control. Gate booleans are written before the retraction
        # exists and can never see it, so a question would keep reading ANSWERED WELL on the strength
        # of a number nothing still stands behind. Superseded modules stay in the table and stay out
        # of the verdict.
        retracted = {f: SUPERSEDED[f] for f in own if f in SUPERSEDED}
        live = {f: v for f, v in own.items() if f not in SUPERSEDED}
        answered_well = any(v["failed"] == 0 and v["passed"] > 0 for v in live.values())
        best_pipeline = next((f for f, v in live.items()
                              if v["failed"] == 0 and v["passed"] > 0), None)

        rows.append({"question": q, "what": desc, "blocks": counts, "n_items": n,
                     "gates_passed": gp, "gates_failed": gf, "gate_detail": detail,
                     "answered_well": answered_well, "best_pipeline": best_pipeline,
                     "adjacent_credits": adjacent, "retracted": retracted,
                     "encode": encode, "pipeline": pipeline, "check": check,
                     "answers": bool(encode and pipeline and check),
                     "emits": bool(encode), "namers": sorted(namers)[:8],
                     "produces": sorted(produced)[:8],
                     "phenotype_hits": hits[:6], "controlled": ck[:6], "note": e["note"]})

    say(f"\n  {'question':<22}{'ENCODE':>9}{'PIPE':>7}{'CHECK':>7}{'items':>9}"
        f"{'askable':>10}{'gates':>9}{'well?':>7}")
    for r in rows:
        g = f"{r['gates_passed']}/{r['gates_passed']+r['gates_failed']}" if r["check"] else "-"
        say(f"  {r['question']:<22}{'yes' if r['encode'] else 'NO':>9}"
            f"{'yes' if r['pipeline'] else 'NO':>7}{'yes' if r['check'] else 'NO':>7}"
            f"{r['n_items']:>9,}{'YES' if r['answers'] else 'no':>10}{g:>9}"
            f"{'YES' if r['answered_well'] else 'no':>7}")
    say("")
    for r in rows:
        say(f"    {r['question']:<22} slots {r['blocks']}")
        say(f"    {'':<22} modules that address it: {', '.join(Path(x).stem for x in r['namers']) or 'NONE'}")
        if r.get("best_pipeline"):
            say(f"    {'':<22} answered well by: {r['best_pipeline']}")
        if r.get("adjacent_credits"):
            say(f"    {'':<22} ADJACENT (touch the slot, do not answer it): "
                f"{', '.join(r['adjacent_credits'])}")
        for f, why in (r.get("retracted") or {}).items():
            say(f"    {'':<22} SUPERSEDED, gates kept and vote removed: {f}")
            say(f"    {'':<22}   {why}")
        say(f"    {'':<22} recorded results: {', '.join(r['produces']) or 'NONE'}"
            f"   controlled: {', '.join(r['controlled']) or 'NONE'}")

    # ---- the calibration, which can void the run ------------------------------------------------------
    fold = next(r for r in rows if r["question"] == "any chromosome fold")
    grow = next(r for r in rows if r["question"] == "metabolic growth")
    neg = next(r for r in rows if r["question"] == "NEGATIVE CONTROL")
    say("\n  CALIBRATION -- two known answers, checked before any of the above is believed")
    say(f"    the synthetic NEGATIVE CONTROL must fail on every axis -- nothing can ever satisfy it")
    say(f"      -> encode = {neg['encode']}, pipeline = {neg['pipeline']}, check = {neg['check']}   "
        f"{'OK' if not (neg['encode'] or neg['pipeline'] or neg['check']) else 'AUDIT VOID'}")
    say(f"    chromosome fold must NOW reach one (loop_real_chromatin built it on measured torsion)")
    say(f"      -> pipeline = {fold['pipeline']}, check = {fold['check']}   "
        f"{'OK' if (fold['pipeline'] and fold['check']) else 'AUDIT VOID'}")
    say(f"    metabolic growth must reach one, with a control (cell_loop closed it today)")
    say(f"      -> pipeline = {grow['pipeline']}, check = {grow['check']}   "
        f"{'OK' if (grow['pipeline'] and grow['check']) else 'AUDIT VOID'}")
    void = bool(neg["encode"] or neg["pipeline"] or neg["check"]
                or not (grow["pipeline"] and grow["check"])
                or not (fold["pipeline"] and fold["check"]))

    real = [r for r in rows if r["question"] not in ("metabolic growth", "functional neighbourhood",
                                                     "NEGATIVE CONTROL")]
    emits = sum(r["emits"] for r in real)
    answers = sum(r["answers"] for r in real)
    well = sum(r["answered_well"] for r in real)
    say("\n" + "=" * 100)
    if void:
        say("  AUDIT VOID. The calibration failed, so every verdict above is unreliable and none of it")
        say("  should be quoted. The instrument is wrong before the repo is.")
        say("  Check which module was credited with the connection and what it actually writes. The")
        say("  criterion is deliberately strict -- one module addressing the slot AND recording a")
        say("  controlled result -- so a failure here is usually a naming collision, not a real path.")
    else:
        say(f"  OF THE {len(real)} QUESTION TYPES THE GOAL NAMES: {emits} can be ASKED, "
            f"{answers} can be got WRONG, and {well} {'is' if well == 1 else 'are'} ANSWERED WELL -- "
            f"every gate in the pipeline that answers it came back positive.")
        say(f"  The THIRD number is the honest one. The second counts questions that can now be got")
        say(f"  wrong, which is real progress and is not the same as knowing the answer.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(CELL), str(OUT / "data_doctrine.json"), str(OUT / "trace_all.json")],
                      available=len(QUESTIONS), used=len(rows), selection="all", seed=SEED,
                      controls=["chromosome-fold must-fail calibration",
                                "metabolic-growth must-pass calibration"],
                      note="entry mapping is a declared judgement, written out per question with counts")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "capability_audit", "manifest": man, "void": void, "rows": rows,
               "n_emits": emits, "n_answers": answers, "n_answered_well": well,
               "n_questions": len(real),
               "controlled_results": checked, "n_modules": len(inv),
               "n_phenotype_modules": len(pheno_mods), "log": log},
              open(OUT / "capability_audit.json", "w"), indent=2)
    say(f"\n  -> {OUT/'capability_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
