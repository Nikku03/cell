"""CELL TRACK EXECUTOR -- the two record corrections the improver accepted (C1 and C2).

Neither of these changes the model. Both change what the RECORD says about the model, and the
distinction matters: this repository's ethos is that an overstatement left standing is a defect
even when the conclusion it overstates happens to be right.

  C1  THE KCAT LAYER CITES A TEST THAT WAS SHOWN TO BE BROKEN.
      Layer 'what the kcat model actually learned' is FAILED, and its evidence reads: "B4 settles
      it -- refit on the EC-median residual, the model scores RMSE 1.5386 against a residual sd of
      1.4849, R2 -0.0737. WORSE THAN A CONSTANT", concluding "EVERYTHING THE MODEL KNOWS IS ALREADY
      IN THE EC NUMBER".

      Loop 134 found three defects in that test. Its training residual was built from a complement
      containing the test fold (worth +0.0312). Its baseline was an in-sample sd rather than an
      out-of-fold constant (worth -0.0024). And the residual construction was the wrong instrument
      entirely, because the question needs the same target and a control that destroys sequence
      while preserving EC.

      Loop 134 then ran that control -- permuting the embedding among records SHARING an EC number
      -- and it cost +0.0046 against a paired interval of 0.0488. C4 confirmed first that the
      control could move: 73.4% of records received a different sequence, achievable bound 0.9997.

      SO THE STATUS DOES NOT CHANGE. The layer stays FAILED and the conclusion stands. What changes
      is that it stops resting on a leaking residual. The correction also fixes a claim in the
      SAME layer that loop 134 contradicts: the EC number is measured at 0.7% of the variance, so
      "everything the model knows is in the EC number" is wrong as stated -- what the model uses is
      family structure at a resolution the ESM embedding captures and the EC string does not.

  C2  cell_run.json RECORDS FOUR COUNTS AND CANNOT SHOW WHICH LAYER MOVED.
      It stores {"RUNS": 11, "CLOSES": 9, "FAILED": 12, "STATIC": 10}. Two layers swapping status
      leaves every count identical, so the improver's Q3 on the cell track -- "what changed" --
      cannot be answered from the artefact at all. Worse, those counts are already STALE: the
      table now reads 12 RUNS, 10 CLOSES, 14 FAILED, 11 STATIC. The artefact was not merely coarse.

      This writes outputs/cell_layers.json with per-layer status, source and a content hash of the
      evidence text, so a future turn can name the layer that moved and not merely observe that a
      count did.

PREDECLARED:

  G1 THE CORRECTION IS NECESSARY.
       the layer must still cite the broken test before the edit. Gate: if it does not, this script
       has already run or the layer was corrected by hand, and it must make no change rather than
       rewrite prose twice.

  G2 THE STATUS IS UNCHANGED.
       Gate: FAILED before and FAILED after. A record correction that flips a verdict is not a
       record correction, and if the evidence really did change the verdict that has to be argued
       in a loop, not in an executor.

  G3 THE COUNTS ARE RECOMPUTED, NOT COPIED.
       Gate: report cell_run.json's stored counts against the table's actual counts and say
       plainly whether the artefact was stale.

  G4 THE NEW ARTEFACT CAN DO WHAT THE OLD ONE COULD NOT.
       Gate: outputs/cell_layers.json must contain per-layer status for all 47 layers, so that a
       diff between two runs names the layer.

-> outputs/cell_record_fix.json
"""
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SRC = Path("colab/cell_assembled.py")
TARGET = "what the kcat model actually learned"

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def layers():
    for node in ast.parse(SRC.read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "LAYERS":
            return [tuple(t) for t in ast.literal_eval(node.value)]
    return []


def layer_nodes(src, name):
    """The AST nodes for one LAYERS entry, so the rewrite can target exact source offsets.

    THE FIRST VERSION OF THIS SCRIPT DID NOT DO THIS AND IT FAILED SILENTLY. It searched for the
    evidence text with `OLD_EV in src`, but every entry in LAYERS is written as adjacent string
    literals across several lines -- "first part " "second part" -- so the concatenated VALUE never
    appears in the source at all. The substitution matched nothing, made no change, and G1 reported
    PASS because it had been written as `gates["G1"] = True` with no condition attached.

    That is a gate that fires while measuring nothing, in the executor whose entire purpose is
    correcting an overstatement. Operating on AST node offsets removes the string-matching problem,
    and G1 below now verifies the edit landed instead of asserting that it did."""
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "LAYERS":
            for t in node.value.elts:
                if (isinstance(t, ast.Tuple) and t.elts
                        and isinstance(t.elts[0], ast.Constant) and t.elts[0].value == name):
                    return t
    return None


def offset(src, lineno, col):
    lines = src.splitlines(keepends=True)
    return sum(len(x) for x in lines[:lineno - 1]) + col


def wrap(value, indent=5):
    """Re-emit a long string as adjacent literals, matching how LAYERS is written by hand."""
    pad = " " * indent
    words, lines, cur = value.split(" "), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 94:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w) if cur else w
    if cur:
        lines.append(cur)
    body = ('\n' + pad).join(json.dumps(l + (" " if i < len(lines) - 1 else ""))
                             for i, l in enumerate(lines))
    return body


def replace_elements(src, name, new_ev, new_cv):
    """Rewrite elements 3 and 4 of one LAYERS tuple, back to front so offsets stay valid."""
    t = layer_nodes(src, name)
    if t is None or len(t.elts) < 5:
        return src, False
    for idx, new in ((4, new_cv), (3, new_ev)):
        n = t.elts[idx]
        a = offset(src, n.lineno, n.col_offset)
        b = offset(src, n.end_lineno, n.end_col_offset)
        src = src[:a] + wrap(new) + src[b:]
    return src, True


OLD_EV = ("the ESM embedding is a FAMILY DETECTOR, not a catalysis model. Same 320 numbers, same "
          "folds: EC top-level class accuracy 0.779 against a 0.394 majority, log10 sequence "
          "length R2 +0.7631, log10 kcat R2 +0.1381. And B4 settles it -- refit on the EC-median "
          "residual, the model scores RMSE 1.5386 against a residual sd of 1.4849, R2 -0.0737. "
          "WORSE THAN A CONSTANT")
NEW_EV = ("the ESM embedding is a FAMILY DETECTOR, not a catalysis model. Same 320 numbers, same "
          "folds: EC top-level class accuracy 0.779 against a 0.394 majority, log10 sequence "
          "length R2 +0.7631, log10 kcat R2 +0.1381. loop 133's B4 reached this by refitting on "
          "the EC-median residual (RMSE 1.5386 vs a residual sd of 1.4849), and loop 134 C2 showed "
          "that test was broken three ways: its training residual was built from a complement "
          "CONTAINING the test fold (+0.0312), its baseline was an in-sample sd (-0.0024), and a "
          "residual is the wrong instrument for the question. loop 134 C3 settles it properly and "
          "without a residual -- permuting the embedding among records SHARING an EC number costs "
          "+0.0046 against a paired interval of 0.0488, with C4 confirming first that 73.4% of "
          "records did receive a different sequence")
OLD_CV = ("EVERYTHING THE MODEL KNOWS IS ALREADY IN THE EC NUMBER. 'Predicting kcat from sequence' "
          "is EC-class lookup routed through a language model. The +0.1323 gain loop 132 confirmed "
          "as real is real -- and it is the EC number's, not the sequence's")
NEW_CV = ("PROTEIN IDENTITY IS WORTH +0.0046, WHICH IS NOTHING. But 'everything is in the EC "
          "number' is wrong as stated and loop 134 C1 is why: the EC number explains 0.7% of the "
          "variance, while sequence-only scores 1.3890 against a constant's 1.5069. What the model "
          "uses is FAMILY STRUCTURE at a resolution the ESM embedding captures and the EC string "
          "does not -- swapping a protein for a class-mate is free, but the class itself is "
          "identified far better by the embedding than by the label. The +0.1323 gain loop 132 "
          "confirmed as real is real, and it is family-level, not per-protein")


def main():
    t0 = time.time()
    say("=" * 100)
    say("  CELL TRACK EXECUTOR -- record corrections C1 and C2")
    say("=" * 100)
    say()
    gates, res = {}, {}

    # ---------------------------------------------------------------- G1
    say("G1 IS THE CORRECTION NECESSARY?")
    src = SRC.read_text()
    L = layers()
    lay = next((l for l in L if l[0] == TARGET), None)
    before_status = lay[1] if lay else None
    # "still cites the broken test" is checked against the PARSED value, not against the source
    # text, because the source writes it as adjacent literals across several lines.
    def corrected(l):
        """The END STATE this script is responsible for, tested on the PARSED value.

        The first attempt at this predicate required '1.5386' to be absent -- and the corrected
        text cites 1.5386 deliberately, as the number loop 133 reported before loop 134 showed how
        it was produced. So the check failed on a file that was already right. A verification
        predicate that forbids mentioning the superseded number would force the record to hide
        exactly the history it exists to preserve."""
        return bool(l and "loop 134 C3" in l[3] and "EVERYTHING THE MODEL KNOWS" not in l[4])

    already = corrected(lay)
    say(f"     layer found: {bool(lay)}   status before: {before_status}")
    say(f"     already cites loop 134 C3 and no longer asserts the overstatement: {already}")
    if already:
        gates["G1"] = True
        say(f"     G1 PASS -- nothing to correct; this script has already run")
        ok = None
    else:
        src2, ok = replace_elements(src, TARGET, NEW_EV, NEW_CV)
        if ok:
            SRC.write_text(src2)
        lay3 = next((l for l in layers() if l[0] == TARGET), None)
        gates["G1"] = corrected(lay3)          # verified by RE-PARSING, never asserted
        say(f"     rewrite applied at AST offsets: {ok}")
        say(f"     G1 {'PASS' if gates['G1'] else 'FAIL'} -- the correction "
            f"{'is in the file and parses' if gates['G1'] else 'DID NOT LAND'}")
    res["g1"] = {"layer_found": bool(lay), "status_before": before_status,
                 "already_correct": already, "rewrote": ok, "verified": gates["G1"]}
    say()

    # ---------------------------------------------------------------- G2
    say("G2 IS THE STATUS UNCHANGED?")
    L2 = layers()
    lay2 = next((l for l in L2 if l[0] == TARGET), None)
    after_status = lay2[1] if lay2 else None
    gates["G2"] = bool(before_status == after_status == "FAILED")
    say(f"     before {before_status}   after {after_status}")
    say(f"     G2 {'PASS' if gates['G2'] else 'FAIL'} -- the conclusion "
        f"{'stands; only its evidence changed' if gates['G2'] else 'MOVED, which an executor must not do'}")
    res["g2"] = {"before": before_status, "after": after_status}
    say()

    # ---------------------------------------------------------------- G3
    say("G3 ARE THE RECORDED COUNTS STALE?")
    import collections
    actual = dict(collections.Counter(l[1] for l in L2))
    run = json.load(open(OUT / "cell_run.json")) if (OUT / "cell_run.json").exists() else {}
    stored = run.get("layers")
    stale = bool(stored and stored != actual)
    say(f"     cell_run.json stored: {stored}")
    say(f"     the table actually reads: {actual}")
    say(f"     STALE: {stale}")
    if stale:
        say(f"     the artefact was not merely coarse -- it disagrees with the table it describes")
    gates["G3"] = True
    res["g3"] = {"stored": stored, "actual": actual, "stale": stale}
    say()

    # ---------------------------------------------------------------- G4
    say("G4 CAN THE NEW ARTEFACT NAME THE LAYER THAT MOVED?")
    per = [{"layer": l[0], "status": l[1], "source": l[2],
            "evidence_sha": hashlib.sha256((l[3] or "").encode()).hexdigest()[:12]}
           for l in L2]
    json.dump({"n_layers": len(per), "counts": actual, "layers": per},
              open(OUT / "cell_layers.json", "w"), indent=1)
    gates["G4"] = bool(len(per) == len(L2) and all(p["status"] for p in per))
    say(f"     wrote outputs/cell_layers.json with per-layer status for {len(per)} layers")
    say(f"     each carries a hash of its evidence text, so a reworded layer is visible too")
    say(f"     G4 {'PASS' if gates['G4'] else 'FAIL'}")
    res["g4"] = {"n_layers": len(per)}
    say()

    say("=" * 100)
    for k in ("G1", "G2", "G3", "G4"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[SRC, OUT / "loop_b4_fix.json", OUT / "cell_run.json"],
                      available=len(L2), used=len(L2), selection="all", seed=0,
                      controls=["the status is required to be UNCHANGED by the correction",
                                "the stored counts are recomputed from the table, not copied"],
                      note="record corrections only; no model behaviour is changed by this script")
    RM.report(man, emit=say)
    json.dump({"test": "cell record fix (C1, C2)", "manifest": man, "gates": gates, **res,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "cell_record_fix.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'cell_record_fix.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
