"""THE IMPROVER, RUN AS A LOOP -- and the first thing the loop must be able to say is "nothing
changed, so I have nothing to propose".

improver.py answers "how do I get a better result than this?" once. Running it ten times raises a
problem that running it once does not, and the problem is fatal if ignored:

    A SELF-IMPROVEMENT LOOP WHOSE INPUTS DO NOT CHANGE BETWEEN TURNS EMITS THE SAME PLAN EVERY
    TURN, AND EMITTING THE SAME PLAN TEN TIMES LOOKS EXACTLY LIKE PROGRESS.

That is this repository's oldest failure mode wearing new clothes. Loop 92 found twelve gates that
fired while measuring nothing; a turn that re-derives an unchanged plan from unchanged JSON is the
same thing at the level of the whole cycle. So the loop is built around three mechanisms whose only
job is to make an empty turn LOOK empty.

  THE FINGERPRINT. Before a turn runs, the track's input artefacts are hashed -- content, not
  mtime. If the fingerprint is unchanged since the last turn AND nothing was executed in between,
  the turn does not emit a plan. It prints STALLED, names the artefacts that would have to change,
  and names what is blocking each. A stalled turn is a real result: it says precisely what the
  project is waiting on.

  THE EXECUTOR. A turn that only plans can never change its own inputs, so the loop would stall at
  turn 2 by construction and the whole exercise would be theatre. Each track therefore registers
  executors: a plan item id mapped to a script that, when run, WRITES A NEW ARTEFACT. That new
  artefact changes the fingerprint, which is what earns the next turn the right to exist. Nothing
  else in this file can change a fingerprint -- in particular, nothing I write by hand can.

  THE DRIFT LEDGER. Across ten turns I both propose the items and write the checks that judge them,
  and the natural gradient of that arrangement is to soften the checks or to shape items to pass
  them. So the source of theory_check is hashed every turn and recorded. If it changes, the turn
  must carry a justification and the change is printed as a DRIFT EVENT rather than absorbed
  silently. The rejection count is recorded per turn for the same reason: a rejection rate that
  falls while item quality is unchanged is the signature of a check being worn down.

WHAT REFLECTION MEANS HERE, and it is not commentary. Each turn compares itself against every turn
before it and answers, from the recordings:

    R1  did the fingerprint move, and which artefact moved it
    R2  did any prediction from an earlier turn get MEASURED, and by how much was it wrong
    R3  did the plan actually change, or is this turn re-emitting its predecessor
    R4  did the checks get weaker -- rejection count, drift events, check-source hash
    R5  what is the loop now waiting on, per track, named as a file or a measurement
    R6  is this track finished, and if so at which turn did it genuinely stop

R2 is the one that gives the whole loop teeth. improver.py predicts a gain for every item; until
something is executed and measured, those predictions are unfalsified opinions. The loop records
the signed error of every prediction that gets measured, so that by turn 10 there is a track record
rather than ten plans.

HONEST EXPECTATION, written before running. Two of the five ml_kcat items are dead on T8 because
their data is not on disk (UniProt active-site annotations were never fetched; no temperature or pH
exists in any form), and a third is BLOCKED behind one of those. The executable surface is small.
This loop is therefore EXPECTED to stall well before turn 10, and the turn at which it stalls, plus
the reason, is the finding. A run that produces ten substantive turns would be the surprise.

-> outputs/improver_loop/turn_<NN>_<track>.json     one per turn per track
-> outputs/improver_loop/history.json               the rolling ledger
-> outputs/improver_loop.json                       the summary
"""
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import improver as IMP  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
LOOPDIR = OUT / "improver_loop"
MAX_TURNS = int(os.environ.get("IMPROVER_TURNS", "10"))
# ALLOW_VACUOUS disables the stall rule. It exists to DEMONSTRATE the rule rather than to be used:
# running with it on produces turns whose recorded plans are byte-identical, which is the evidence
# that the stall rule is not merely a convenience. Never set it for a real run.
ALLOW_VACUOUS = os.environ.get("IMPROVER_ALLOW_VACUOUS") == "1"

# EXECUTORS. A plan item id -> the script that runs it and the artefact it must produce. This is
# the ONLY way a fingerprint can change, which is deliberate: it means the loop cannot talk itself
# into a new turn, it has to earn one by producing a measurement.
EXECUTORS = {
    "ml_kcat": {
        "M1": {"script": "colab/loop_plan_exec.py", "produces": "loop_plan_exec.json",
               "covers": ["M1", "P2", "P4"],
               "note": "the probe that measures the EC channel through three encodings (E7), "
                       "which is what unblocks P6's T7. The same script also measures P2 and P4"},
        "B": {"script": "colab/loop_plan_exec.py", "produces": "loop_plan_exec.json",
              "covers": ["P2", "P4"],
              "note": "the improver's upgrade step said to run the sub-noise items as ONE change"},
        "P2": {"script": "colab/loop_plan_exec.py", "produces": "loop_plan_exec.json",
               "covers": ["P2", "P4"], "note": "same script; P2 is measured inside it as E2"},
    },
    "cell": {
        "C1": {"script": "colab/cell_record_fix.py", "produces": "cell_record_fix.json",
               "covers": ["C1", "C2"],
               "note": "record corrections: the kcat layer's citation, and per-layer status"},
        "C2": {"script": "colab/cell_record_fix.py", "produces": "cell_record_fix.json",
               "covers": ["C1", "C2"], "note": "same script"},
    },
}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def sha(p):
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]


def fingerprint(track):
    """Content hash of every artefact the track reads. Content and not mtime, because a rerun that
    produces identical output has not moved the project and must not be allowed to buy a turn."""
    t = IMP.TRACKS[track]
    d = {f: sha(OUT / f) for f in t["runs"]}
    # Some tracks read SOURCE as data -- the cell track's 47-layer table is a Python literal in
    # cell_assembled.py, and its first item edits it. Hashing outputs/ alone would miss that.
    d.update({f: sha(f) for f in t.get("also_hash", [])})
    blob = json.dumps(d, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16], d


def check_source_hash():
    """The checks judge the items, and I write both. Hashing the judge makes softening visible."""
    src = "".join(inspect.getsource(f) for f in (IMP.theory_check, IMP.plan, IMP.combine))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def executable(track, items, bundle):
    """Items that PASSED every check AND have a registered executor AND whose executor has not
    already produced its artefact. The last clause is what stops a turn re-running a finished job
    to manufacture the appearance of motion."""
    ex = EXECUTORS.get(track, {})
    units = [i for i in items if i["verdict"] == "ACCEPT"]
    if bundle and bundle["verdict"] == "ACCEPT":
        units = units + [bundle]
    out = []
    for u in units:
        e = ex.get(u["id"])
        if not e:
            continue
        if (OUT / e["produces"]).exists():
            continue
        out.append((u, e))
    return out


def run_executor(e):
    t = time.time()
    r = subprocess.run([sys.executable, e["script"]], capture_output=True, text=True, timeout=7200)
    ok = (OUT / e["produces"]).exists()
    return {"script": e["script"], "produces": e["produces"], "ok": bool(ok),
            "returncode": r.returncode, "seconds": time.time() - t,
            "tail": (r.stdout or r.stderr or "")[-1500:]}


# WHERE EACH ITEM'S OUTCOME IS ACTUALLY MEASURED. Without this table the loop scores every item
# against whatever number happens to sit in the executor's output, which for the first draft meant
# scoring the PROBE M1 -- an item that predicts nothing by definition -- against the bundle's
# measured gain, and then reporting it as a prediction M1 had got wrong. A probe has no prediction
# to score, and an item scored against the wrong quantity is worse than an item not scored at all.
MEASURED_AT = {
    "ml_kcat": {
        "P2": ("loop_plan_exec.json", ("e2", "gain"), "mutant flag alone, E2"),
        "P4": ("loop_plan_exec.json", ("e3", "gain"), "explicit EC alone, E3"),
        "B": ("loop_plan_exec.json", ("e4", "gain"), "the bundle tested as one change, E4"),
        "P6": ("loop_plan_exec.json", ("e7", "best_gain"), "best EC encoding, E7"),
    },
    "cell": {},
}


def measured_predictions(track, history):
    """R2. Every prediction an earlier turn made, matched against whatever has since been measured.
    Until this list is non-empty the loop is producing opinions, and it should say so."""
    got = []
    where = MEASURED_AT.get(track, {})
    for h in history:
        if h.get("track") != track:
            continue
        for it in h.get("plan", []):
            if it.get("kind") == "probe":
                continue                      # a probe promises nothing; there is nothing to score
            loc = where.get(it["id"])
            if not loc:
                continue
            fn, path, what = loc
            p = OUT / fn
            if not p.exists():
                continue
            d = json.load(open(p))
            for k in path:
                d = (d or {}).get(k) if isinstance(d, dict) else None
            if d is None:
                continue
            meas = float(d)
            got.append({"turn": h["turn"], "id": it["id"], "measured_as": what,
                        "predicted": it["predicted_gain"], "measured": meas,
                        "signed_error": meas - it["predicted_gain"],
                        "verdict": "OVER-promised" if meas < it["predicted_gain"]
                                   else "UNDER-promised"})
    seen, uniq = set(), []
    for g in got:
        k = (g["turn"], g["id"])
        if k not in seen:
            seen.add(k)
            uniq.append(g)
    return uniq


def reflect(turn, track, rec, history):
    """R1-R6, answered from the recordings and not from my recollection of them."""
    prior = [h for h in history if h.get("track") == track]
    prev = prior[-1] if prior else None
    r = {}

    moved = []
    if prev:
        for f, h in rec["fingerprint_files"].items():
            if prev["fingerprint_files"].get(f) != h:
                moved.append(f)
    r["R1_fingerprint"] = {
        "changed": bool(prev is None or rec["fingerprint"] != prev["fingerprint"]),
        "moved_artefacts": moved,
        "why": ("first turn" if prev is None else
                (f"{len(moved)} artefact(s) changed" if moved else
                 "NOTHING CHANGED -- no new measurement has entered this track"))}

    r["R2_predictions_measured"] = measured_predictions(track, history + [rec])
    r["R2_note"] = ("no prediction has been measured yet; every predicted_gain in this track is "
                    "still an unfalsified opinion"
                    if not r["R2_predictions_measured"] else
                    f"{len(r['R2_predictions_measured'])} prediction(s) now have a measured outcome")

    def sig(rc):
        return sorted((i["id"], i["verdict"], round(i["predicted_gain"], 4))
                      for i in rc.get("plan", []))
    r["R3_plan_changed"] = {
        "changed": bool(prev is None or sig(rec) != sig(prev)),
        "detail": ("first turn" if prev is None else
                   ("the plan differs from the previous turn" if sig(rec) != sig(prev)
                    else "IDENTICAL to the previous turn -- this turn added nothing"))}

    drift = []
    if prev and prev.get("check_source_hash") != rec["check_source_hash"]:
        drift.append(f"theory_check/plan/combine source changed: "
                     f"{prev.get('check_source_hash')} -> {rec['check_source_hash']}")
    if prev and rec["n_rejected"] < prev.get("n_rejected", 0):
        drift.append(f"rejection count FELL {prev.get('n_rejected')} -> {rec['n_rejected']}; "
                     f"if the items did not improve, the checks were softened")
    r["R4_drift"] = {"events": drift, "n_rejected": rec["n_rejected"],
                     "n_blocked": rec["n_blocked"],
                     "check_source_hash": rec["check_source_hash"],
                     "verdict": "no drift detected" if not drift else "DRIFT -- see events"}

    waiting = []
    for it in rec.get("plan", []):
        if it.get("missing_inputs"):
            waiting.append(f"{it['id']}: file absent -- "
                           f"{', '.join(str(m) for m in it['missing_inputs'])}")
        elif it.get("blocked_on"):
            waiting.append(f"{it['id']}: unmeasured -- {'; '.join(it['blocked_on'])}")
        elif it["verdict"] == "REJECT":
            waiting.append(f"{it['id']}: rejected on {', '.join(it['rejected_on'])}")
    r["R5_waiting_on"] = waiting

    r["R6_finished"] = {
        "stalled": rec.get("stalled", False),
        "executable_remaining": rec.get("n_executable", 0),
        "verdict": ("FINISHED -- no executable item and no artefact can change without new data"
                    if rec.get("stalled") else
                    f"{rec.get('n_executable', 0)} executable item(s) remain")}
    return r


def one_turn(turn, track, history):
    fp, fpf = fingerprint(track)
    prior = [h for h in history if h.get("track") == track]
    prev = prior[-1] if prior else None
    executed_since = any(h.get("executed") for h in prior[-1:]) if prior else False

    rec = {"turn": turn, "track": track, "fingerprint": fp, "fingerprint_files": fpf,
           "check_source_hash": check_source_hash(), "stalled": False}

    # THE ANTI-VACUITY RULE, applied before anything is computed.
    if prev and fp == prev["fingerprint"] and not executed_since and not ALLOW_VACUOUS:
        rec.update({"stalled": True, "plan": prev.get("plan", []),
                    "n_rejected": prev.get("n_rejected", 0),
                    "n_blocked": prev.get("n_blocked", 0), "n_executable": 0,
                    "stall_reason": "input fingerprint unchanged since the previous turn and "
                                    "nothing was executed in between, so any plan emitted here "
                                    "would be a copy of the last one"})
        say(f"\n  TURN {turn} [{track}]  STALLED")
        say(f"     fingerprint {fp} unchanged; no artefact moved and nothing ran")
        say(f"     the loop is waiting on data, not on thinking")
        rec["reflection"] = reflect(turn, track, rec, history)
        for w in rec["reflection"]["R5_waiting_on"]:
            say(f"       waiting: {w}")
        return rec

    a = IMP.ask(track)
    items, noise, noise_src = IMP.plan(a)
    items, refuted, bundle = IMP.theory_check(items, a, noise)
    pic = IMP.whole_picture(a)
    ups = IMP.upgrade(items, bundle, pic, noise)
    n_rej = sum(1 for i in items if i["verdict"] == "REJECT")
    n_blk = sum(1 for i in items if i["verdict"] == "BLOCKED")
    ex = executable(track, items, bundle)

    rec.update({"n_measured": sum(1 for i in items if i["verdict"] == "MEASURED"),
                "answers": a, "plan": items, "bundle": bundle, "whole_picture": pic,
                "upgrades": ups, "noise": noise, "noise_source": noise_src,
                "n_rejected": n_rej, "n_blocked": n_blk, "n_executable": len(ex),
                "execute_order": ups[0]["new_order"] if ups else []})

    say(f"\n  TURN {turn} [{track}]  fingerprint {fp}")
    n_meas = sum(1 for i in items if i["verdict"] == "MEASURED")
    ni = f"{noise:.4f}" if noise is not None else "n/a on this track"
    say(f"     plan {len(items)} items: {sum(1 for i in items if i['verdict'].startswith('ACCEPT'))}"
        f" accepted, {n_rej} rejected, {n_blk} blocked, {n_meas} MEASURED   (null {ni})")
    for it in items:
        tag = it["verdict"]
        extra = ""
        if it.get("missing_inputs"):
            extra = f"  MISSING {', '.join(str(m) for m in it['missing_inputs'])}"
        elif it.get("blocked_on"):
            extra = f"  BLOCKED ON {'; '.join(it['blocked_on'])}"
        elif it["rejected_on"]:
            extra = f"  on {', '.join(x.split('_')[0] for x in it['rejected_on'])}"
        if it["verdict"] == "MEASURED":
            say(f"       {it['id']} {tag:<17} {it['change'][:46]}")
            say(f"           -> {it.get('measured_note', '')}")
        else:
            say(f"       {it['id']} {tag:<17} {it['predicted_gain']:+.4f}  "
                f"{it['change'][:46]}{extra}")
    if bundle:
        say(f"       B  {bundle['verdict']:<17} {bundle['combined_gain']:+.4f}  "
            f"bundle of {'+'.join(bundle['members'])}")
    say(f"     executable now: {[u['id'] for u, _ in ex] or 'NOTHING'}")

    # EXECUTE. This is the only thing in the file that can move a fingerprint.
    rec["executed"] = []
    for u, e in ex:
        say(f"     EXECUTING {u['id']} -> {e['script']}")
        out = run_executor(e)
        out["item"] = u["id"]
        rec["executed"].append(out)
        say(f"       {'produced' if out['ok'] else 'FAILED'} {e['produces']} "
            f"[{out['seconds']:.1f}s]")
        if not out["ok"]:
            say(f"       tail: {out['tail'][-300:]}")

    rec["reflection"] = reflect(turn, track, rec, history)
    R = rec["reflection"]
    say(f"     REFLECT")
    say(f"       R1 fingerprint: {R['R1_fingerprint']['why']}")
    say(f"       R2 predictions: {R['R2_note']}")
    for g in R["R2_predictions_measured"]:
        say(f"          turn {g['turn']} {g['id']}: predicted {g['predicted']:+.4f}, measured "
            f"{g['measured']:+.4f} ({g['measured_as']}), error {g['signed_error']:+.4f} ({g['verdict']})")
    say(f"       R3 plan:        {R['R3_plan_changed']['detail']}")
    say(f"       R4 drift:       {R['R4_drift']['verdict']} "
        f"(rejected {R['R4_drift']['n_rejected']}, blocked {R['R4_drift']['n_blocked']})")
    for d in R["R4_drift"]["events"]:
        say(f"          DRIFT: {d}")
    say(f"       R5 waiting on:  {len(R['R5_waiting_on'])} item(s)")
    for w in R["R5_waiting_on"]:
        say(f"          {w}")
    say(f"       R6 {R['R6_finished']['verdict']}")
    return rec


def main():
    t0 = time.time()
    LOOPDIR.mkdir(parents=True, exist_ok=True)
    tracks = sys.argv[1:] or list(IMP.TRACKS.keys())
    say("=" * 100)
    say(f"  THE IMPROVER AS A LOOP -- {MAX_TURNS} turns over {len(tracks)} track(s): "
        f"{', '.join(tracks)}")
    say("=" * 100)
    say(f"  a turn may only exist if an artefact changed or something was executed.")
    say(f"  the loop is EXPECTED to stall before turn {MAX_TURNS}; where it stalls is the finding.")

    history = []
    stalled_streak = {t: 0 for t in tracks}
    for turn in range(1, MAX_TURNS + 1):
        alive = 0
        for track in tracks:
            rec = one_turn(turn, track, history)
            history.append(rec)
            json.dump(rec, open(LOOPDIR / f"turn_{turn:02d}_{track}.json", "w"),
                      indent=1, default=str)
            if rec.get("stalled"):
                stalled_streak[track] += 1
            else:
                stalled_streak[track] = 0
                alive += 1
        json.dump(history, open(LOOPDIR / "history.json", "w"), indent=1, default=str)
        if all(stalled_streak[t] >= 1 for t in tracks) and not ALLOW_VACUOUS:
            say(f"\n  EVERY TRACK STALLED AT TURN {turn}. Continuing would re-emit this turn "
                f"{MAX_TURNS - turn} more times, which is the failure mode this loop exists to "
                f"prevent. Stopping and reporting what it is waiting on.")
            break

    say("\n" + "=" * 100)
    say("  ACROSS THE WHOLE RUN")
    say("=" * 100)
    for track in tracks:
        hs = [h for h in history if h["track"] == track]
        nstall = sum(1 for h in hs if h.get("stalled"))
        nexec = sum(len(h.get("executed", [])) for h in hs)
        preds = measured_predictions(track, hs)
        say(f"  {track}: {len(hs)} turns, {len(hs) - nstall} substantive, {nstall} stalled, "
            f"{nexec} execution(s), {len(preds)} prediction(s) measured")
        if preds:
            err = [abs(p["signed_error"]) for p in preds]
            say(f"     mean |prediction error| {sum(err) / len(err):+.4f}")
        last = hs[-1] if hs else None
        if last:
            for w in last["reflection"]["R5_waiting_on"]:
                say(f"     waiting: {w}")
    say(f"\n  the loop stopped because it ran out of NEW MEASUREMENTS, not out of turns.")

    json.dump({"test": "improver loop", "turns_run": len(set(h['turn'] for h in history)),
               "max_turns": MAX_TURNS, "tracks": tracks, "history": history,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "improver_loop.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'improver_loop.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
