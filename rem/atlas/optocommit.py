"""What does optogenetics measure about holding a decision? The brain target, from raw data.

WHY OPTOGENETICS. Recordings show what neurons DO; optogenetics shows what they CAUSE. Light-gated
channels (channelrhodopsin, Nagel/Hegemann 2003; in neurons, Boyden/Deisseroth 2005; Nobel Prize in
Physiology or Medicine 2026) switch genetically chosen cells on or off with millisecond timing. For a
reasoning network built from brain data, that turns "brain-like" from a resemblance into a CAUSAL test:
perturb the brain, perturb the model the same way, compare what breaks.

THE DATASET. DANDI 000060 (Finkelstein, Fontolan, ..., Romani, Svoboda, Nat Neurosci 2021,
doi 10.1038/s41593-021-00840-6). Mice report whether their vibrissal somatosensory cortex (vS1) was
optogenetically stimulated during a SAMPLE window (light -> lick right, none -> lick left), then hold
the answer through a DELAY until a go cue. On many trials an identical light pulse -- a DISTRACTOR --
arrives at another time. The trial table names the correct response (the sample's), so every
distractor trial asks: did the pulse overwrite a decision the animal should already hold?
Methods text on reward for distractor trials could not be read (paywalled, preprint rate-limited);
the "correct" label used here is the dataset's own trial_instruction field.

=================================================================================================
GATES, PREDECLARED -- written after inspecting ONE session's schema and trial counts (sub-353936,
2017-05-19), before the other 97 sessions were read
=================================================================================================

O0  DATA INTEGRITY, BLOCKING. All 98 files present at their DANDI sizes; every file opens; the trial
    table has outcome, trial_instruction, trial_type_name, photostim_onset, early_lick. Every
    distractor time parsed from trial_type_name must appear in that trial's photostim_onset list.

    AMENDMENT AFTER THE FIRST RUN (commit f3e48c8), RECORDED. O0 FAILED: 59 of 98 files raised in MY
    name parser ('-2.50.75' from names like 'r_-2.5Mini(FullX0.75)'), and 402 parsed distractor times
    were missing from their onset lists. The full name vocabulary was then listed (names and counts
    only, no outcomes). Standard names are  side  or  side_-{t}{Full|Mini}  (Full 2.25 mW, Mini
    0.75 mW). Everything else -- reduced or doubled SAMPLE intensities '(FullX0.75)', '(FullX0.5)',
    'FullX0.5', 'FullX2' (the last at an anomalous onset 4.97 s), and 'r_NoAudCue' -- is a variant
    of the sample or cue, not a distractor, and is COUNTED AND EXCLUDED. 'l_-2.5Mini' (a weak pulse
    in the sample window on a no-sample trial) is kept as its own condition. O0's consistency check
    now applies to standard names. O2 is unchanged: intensity-agnostic, as first written; the
    by-intensity split is reported beside it.

O1  CENSUS. Sessions, mice, task names, trial-type names, distractor times and intensities.

    EXCLUSIONS, fixed now: outcome 'ignore' (no response); any early lick. Left-instruction trials:
    licked right = 'miss'. Right-instruction trials: licked right = 'hit'.

O2  HARNESS, BLOCKING -- the parse must reproduce the paper's published headline, "during the delay,
    distracting stimuli lost influence on behavior over time":
      on no-sample (left) trials, P(lick right | distractor at -1.6 s) > P(lick right | -0.8 s)
      pooled, AND in a majority of mice with >= 10 kept trials in both conditions.
    And trained animals: pooled accuracy on no-distractor trials >= 0.70.
    A failure means OUR READING of the files is wrong, not the paper. Nothing else is reported.

O3  THE TARGET -- THE COMMITMENT CURVE. Per mouse and pooled (mouse-level mean, plus pooled counts):
      FOOLING(t) = P(lick right | no sample, distractor at t) - P(lick right | no sample, none)
    for every distractor time t present, and the same shift on sample trials. Mouse-level spread is
    reported; the mouse, not the trial, is the unit.

O4  WHAT THIS IS AND IS NOT. A behavioural causal target from one task in one lab. It does not yet
    test any model. It is what a reasoning network claiming brain-like commitment must reproduce
    when perturbed the same way.
"""

from __future__ import annotations
import collections
import json
import os
import re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
CACHE = os.path.join(HERE, "_cache", "opto_000060")
OUT = os.path.join(HERE, "RESULTS_optocommit.txt")
ART = R("outputs", "opto_commitment.json")
RULE = "=" * 97
DANDISET = "000060"
API = "https://api.dandiarchive.org/api"
NEED = ("outcome", "trial_instruction", "trial_type_name", "photostim_onset", "early_lick")


def fetch():
    """Download every asset to the gitignored cache (idempotent, size-checked, 4 retries)."""
    import subprocess
    import urllib.request
    os.makedirs(CACHE, exist_ok=True)
    url = f"{API}/dandisets/{DANDISET}/versions/draft/assets/?page_size=200"
    assets = json.load(urllib.request.urlopen(url, timeout=60))["results"]
    man = []
    for a in assets:
        fn = os.path.join(CACHE, a["path"].replace("/", "__"))
        for _ in range(4):
            if os.path.exists(fn) and os.path.getsize(fn) == a["size"]:
                break
            subprocess.run(["curl", "-sSL", "-m", "300", "-o", fn, f"{API}/assets/{a['asset_id']}/download/"])
        man.append(dict(path=a["path"], asset_id=a["asset_id"], size=a["size"],
                        ok=os.path.exists(fn) and os.path.getsize(fn) == a["size"]))
    json.dump(man, open(os.path.join(CACHE, "manifest.json"), "w"), indent=1)
    return man


STD = re.compile(r"^(l|r)(?:_-(\d+\.\d+)(Full|Mini))?$")


def parse_type(name):
    """Standard names only: 'l' -> ('l', None, None); 'l_-1.6Full' -> ('l', 1.6, 'Full').
    Anything else (sample/cue variants) -> None."""
    m = STD.match(name)
    if not m:
        return None
    return m.group(1), (round(float(m.group(2)), 2) if m.group(2) else None), m.group(3)


def load(man):
    import h5py
    dec = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a[()]]
    rows, bad, variants = [], [], collections.Counter()
    for m in man:
        fn = os.path.join(CACHE, m["path"].replace("/", "__"))
        try:
            f = h5py.File(fn, "r")
            t = f["intervals/trials"]
            missing = [k for k in NEED if k not in t]
            if missing:
                bad.append((m["path"], f"missing {missing}")); continue
            cols = {k: dec(t[k]) for k in NEED}
            task = dec(t["task"]) if "task" in t else ["?"] * len(cols["outcome"])
            mouse = m["path"].split("/")[0]
            for i in range(len(cols["outcome"])):
                pt = parse_type(cols["trial_type_name"][i])
                if pt is None:
                    variants[cols["trial_type_name"][i]] += 1
                    continue
                side, dt, inten = pt
                onsets = [round(float(x), 2) for x in cols["photostim_onset"][i].split(",")
                          if x.strip() not in ("N/A", "", "nan", "None")]
                rows.append(dict(mouse=mouse, session=m["path"], task=task[i],
                                 type=cols["trial_type_name"][i], side=side, dt=dt, inten=inten,
                                 instr=cols["trial_instruction"][i], outcome=cols["outcome"][i],
                                 early=cols["early_lick"][i], onsets=onsets))
            f.close()
        except Exception as e:
            bad.append((m["path"], repr(e)))
    return rows, bad, variants


def main():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    P_(RULE); P_("THE BRAIN TARGET: HOW A MOUSE COMMITS TO A DECISION, MEASURED BY OPTOGENETICS"); P_(RULE)
    man = fetch()
    rows, bad, variants = load(man)
    ok_files = sum(m["ok"] for m in man)
    incons = [r for r in rows if r["dt"] is not None and r["dt"] not in r["onsets"]]
    P_(f"  O0 files {ok_files}/{len(man)} at DANDI size; unreadable/incomplete {len(bad)}; "
       f"trials {len(rows):,}; distractor time not in onset list: {len(incons)}")
    for b in bad[:5]:
        P_(f"     {b}")
    if ok_files < len(man) or bad or incons:
        P_("  O0: FAIL. Nothing reported."); open(OUT, "w").write("\n".join(out) + "\n"); return
    P_("  O0: PASS")

    P_("\n  O1 CENSUS")
    mice = sorted({r["mouse"] for r in rows})
    P_(f"    mice {len(mice)}, sessions {len({r['session'] for r in rows})}")
    P_(f"    tasks {dict(collections.Counter(r['task'] for r in rows))}")
    P_(f"    trial types {dict(collections.Counter(r['type'] for r in rows).most_common())}")
    P_(f"    sample/cue VARIANTS excluded (counted): {dict(variants)}  total {sum(variants.values()):,}")
    P_(f"    outcomes {dict(collections.Counter(r['outcome'] for r in rows))}   early licks "
       f"{sum(r['early'] != 'no early' for r in rows)}")
    keep = [r for r in rows if r["outcome"] in ("hit", "miss") and r["early"] == "no early"
            and r["side"] in ("l", "r") and r["instr"] in ("left", "right")]
    for r in keep:
        r["right"] = (r["outcome"] == "miss") if r["instr"] == "left" else (r["outcome"] == "hit")
    P_(f"    kept after predeclared exclusions: {len(keep):,}")
    times = sorted({r["dt"] for r in keep if r["dt"] is not None}, reverse=True)
    sub_conds = sorted({(r["dt"], r["inten"]) for r in keep if r["dt"] is not None}, key=lambda x: (-x[0], x[1]))
    intens = collections.Counter(r["inten"] for r in keep if r["dt"] is not None)
    P_(f"    distractor times (s before go cue) {times}; intensities {dict(intens)}")

    def p_right(rs):
        return (float(np.mean([r["right"] for r in rs])), len(rs)) if rs else (float("nan"), 0)

    conds = [None] + times
    pm = {}
    for mo in mice:
        for instr in ("left", "right"):
            for t in conds:
                rs = [r for r in keep if r["mouse"] == mo and r["instr"] == instr and r["dt"] == t]
                pm[(mo, instr, t)] = p_right(rs)
    pooled = {(i, t): p_right([r for r in keep if r["instr"] == i and r["dt"] == t])
              for i in ("left", "right") for t in conds}

    # ---- O2 -----------------------------------------------------------------------------------
    P_("\n" + RULE); P_("O2  HARNESS: DOES OUR PARSE REPRODUCE THE PUBLISHED HEADLINE? (BLOCKING)"); P_(RULE)
    base = [r for r in keep if r["dt"] is None]
    acc = float(np.mean([r["right"] == (r["instr"] == "right") for r in base]))
    P_(f"  accuracy on no-distractor trials, pooled: {acc:.3f} (n {len(base):,})  -> {'ok' if acc >= 0.70 else 'FAIL'}")
    e, l_ = pooled[("left", 1.6)], pooled[("left", 0.8)]
    P_(f"  pooled P(lick right | no sample): distractor at -1.6 s {e[0]:.3f} (n {e[1]}), at -0.8 s {l_[0]:.3f} (n {l_[1]})")
    elig = [mo for mo in mice if pm[(mo, "left", 1.6)][1] >= 10 and pm[(mo, "left", 0.8)][1] >= 10]
    agree = [mo for mo in elig if pm[(mo, "left", 1.6)][0] > pm[(mo, "left", 0.8)][0]]
    P_(f"  mice with early > late fooling: {len(agree)}/{len(elig)} eligible (>= 10 kept trials each)")
    o2 = acc >= 0.70 and e[0] > l_[0] and len(elig) > 0 and len(agree) > len(elig) / 2
    P_(f"  O2: {'PASS' if o2 else 'FAIL -- the parse is suspect; nothing below is reported'}")
    if not o2:
        open(OUT, "w").write("\n".join(out) + "\n"); return

    # ---- O3 -----------------------------------------------------------------------------------
    P_("\n" + RULE); P_("O3  THE COMMITMENT CURVE"); P_(RULE)
    P_("  P(lick right), pooled over kept trials [mouse-level mean over mice with >= 10 trials]")
    P_("    distractor        no-sample trials (correct = left)        sample trials (correct = right)")
    curve = {}
    for t in conds:
        lab = "none" if t is None else f"-{t:.1f} s"
        cells = []
        for instr in ("left", "right"):
            pr, n = pooled[(instr, t)]
            mm = [pm[(mo, instr, t)][0] for mo in mice if pm[(mo, instr, t)][1] >= 10]
            cells.append(f"{pr:.3f} (n {n:>4}) [{np.mean(mm) if mm else float('nan'):.3f}, {len(mm)} mice]")
            curve[f"{instr}|{lab}"] = dict(pooled=pr, n=n, mouse_mean=float(np.mean(mm)) if mm else None,
                                           mice=len(mm), per_mouse={mo: pm[(mo, instr, t)] for mo in mice})
        P_(f"    {lab:<10} {cells[0]:<42} {cells[1]}")
    P_("\n  BY INTENSITY, pooled P(lick right) [n]:   no-sample trials | sample trials")
    for (t, it) in sub_conds:
        a = p_right([r for r in keep if r["instr"] == "left" and r["dt"] == t and r["inten"] == it])
        b = p_right([r for r in keep if r["instr"] == "right" and r["dt"] == t and r["inten"] == it])
        curve[f"by_intensity|-{t:.1f} s|{it}"] = dict(left=a, right=b)
        P_(f"    -{t:.1f} s {it:<5}   {a[0]:.3f} [{a[1]:>5}]   |   {b[0]:.3f} [{b[1]:>5}]")
    P_("\n  FOOLING(t) on no-sample trials = P(right | distractor t) - P(right | none), per mouse:")
    P_("    mouse        " + "".join(f"{'-' + format(t, '.1f') + ' s':>10}" for t in times))
    fool = {}
    for mo in mice:
        b = pm[(mo, "left", None)]
        vals = []
        for t in times:
            x = pm[(mo, "left", t)]
            v = x[0] - b[0] if (x[1] >= 10 and b[1] >= 10) else None
            vals.append(v); fool[(mo, t)] = v
        P_(f"    {mo:<12} " + "".join(f"{v:>10.3f}" if v is not None else f"{'-':>10}" for v in vals))
    P_("    mean         " + "".join(
        f"{np.mean([fool[(mo, t)] for mo in mice if fool[(mo, t)] is not None]):>10.3f}"
        if any(fool[(mo, t)] is not None for mo in mice) else f"{'-':>10}" for t in times))

    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"dataset": "DANDI:000060", "doi": "10.1038/s41593-021-00840-6",
               "distractor_times_s_before_go": times, "curve": curve,
               "fooling": {f"{mo}|{t}": v for (mo, t), v in fool.items()},
               "kept_trials": len(keep), "mice": mice, "baseline_accuracy": acc},
              open(ART, "w"), indent=1, default=str)
    P_(f"\n  artifact: outputs/opto_commitment.json")
    P_("\n" + RULE); P_("O4  WHAT THIS IS AND IS NOT"); P_(RULE)
    P_("  A causal behavioural target from one task, one lab. It tests no model yet. Any reasoning network")
    P_("  claiming brain-like commitment must reproduce this curve under the same perturbation.")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
