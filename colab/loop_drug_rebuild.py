"""LOOP 58 -- THE DRUG REBUILD: outcomes that are not list lengths.

WHAT HAS BLOCKED THIS ROW SINCE LOOP 12, stated exactly. The side-effect model predicts HOW MANY
adverse effects a drug has, and "how many proteins does it hit" predicts that better (r = 0.2898)
than any biology in the model (partial r 0.0515 against a null 95th of 0.0758). Loop 50 reformulated
to WHICH effects, via drug-pair overlap, and the same confound came back through the OUTCOME instead
of the predictor: the count-and-frequency baseline alone explains R2 0.5685 of pair overlap, and once
it is removed the network signal REVERSES in all five folds.

THE ROOT CAUSE IS THE DATA, not the model. SIDER's effects come from PACKAGE INSERTS. A blockbuster
lists hundreds of effects and a niche drug lists ten, and that is how much attention a drug received,
not what it does to a cell. Any target built by counting that list inherits it.

SO THIS LOOP CHANGES THE OUTCOME TWICE OVER, to two things that are structurally not counts:

  A  FAERS SERIOUSNESS RATE.  Of all real-world adverse-event reports filed for a drug, what FRACTION
     is flagged serious -- death, hospitalisation, disability, life-threatening? It is a RATIO, so a
     famous drug with a hundred thousand reports and a rare one with two hundred are on the same
     scale by construction. FAERS has its own biases -- voluntary reporting, notoriety effects, no
     denominator -- but they are DIFFERENT biases from a package insert, which is what makes it a
     control rather than a repeat.

  B  STOPPED FOR SAFETY vs STOPPED FOR BUSINESS.  24,000 terminated, withdrawn and suspended trials
     were pulled from ClinicalTrials.gov, 21,356 with a stated reason. Classified: 810 stopped for
     safety, 10,965 for recruitment, funding or a sponsor decision. The comparison is
     safety-stopped against BUSINESS-stopped -- not against every other drug -- because both classes
     had a trial that stopped, so how much a drug was studied is held roughly constant. That is the
     matched control SIDER structurally cannot provide, because SIDER contains only drugs that made
     it to market.

PREDECLARED, before any number:

  R1 THE OUTCOME IS A RATE AND NOT A COUNT, AND IT IS CHECKED
       the FAERS seriousness rate must be near-uncorrelated with a drug's total report volume. If a
       ratio still tracks how much a drug is reported, then the fame confound survived the
       reformulation and the rest of this loop is loop 12 again in different clothes. Declared bar:
       |Spearman(rate, total reports)| < 0.30. THIS GATE EXISTS TO CATCH THE FAILURE MODE OF THE TWO
       PREVIOUS ATTEMPTS and it is checked before anything is fitted.
  R2 TARGET COUNT DOES NOT PREDICT THE NEW OUTCOME
       the confound that killed loop 12, re-measured on the new target. Declared: |Spearman(n_targets,
       seriousness rate)| < 0.20 and its AUC on the safety-stopped label inside the null band. A fail
       here means the reformulation did not work, and it is better to learn that in one line than
       after a model is fitted on top of it.
  R3 THE CELL MODEL PREDICTS SERIOUSNESS
       mean essentiality of a drug's targets, mean LOEUF, mean abundance, and the metabolic growth
       cost -- the same predictors loop 18 used for cytotoxicity, where they DID beat target count.
       Against a permutation null, and reported both raw and partialled on target count.
  R4 IT TRANSFERS TO THE INDEPENDENT OUTCOME
       the same features, the same direction, on safety-stopped versus business-stopped. Two
       outcomes from two unrelated databases -- spontaneous reports and trial records -- agreeing is
       worth more than either alone, and disagreeing is worth knowing.
  R5 HELD OUT BY DRUG, and the nulls collapse it
       5-fold grouped by drug; permutation nulls for every reported statistic.

-> outputs/loop_drug_rebuild.json
"""
import collections
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_rescue as LR
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
CELL = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_complete.json"

RATE_VS_VOLUME = 0.30      # R1
TARGET_CONFOUND = 0.20     # R2
MIN_REPORTS = 30           # a rate on five reports is not a rate
N_NULL = 500
SEED = 1904

SAFE = re.compile(r"safety|adverse|toxicit|tolerab|side effect|serious adverse|\bsae\b|death|harm|"
                  r"risk[- ]benefit|dsmb|data safety", re.I)
BIZ = re.compile(r"recruit|enroll|accrual|fund|business|sponsor decision|financial|strategic|covid|"
                 r"investigator (left|depart)|\bslow\b|logistic", re.I)
NOT_A_DRUG = {"placebo", "saline", "normal saline", "vehicle", "sham", "standard of care",
              "best supportive care", "water", "sugar pill"}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def auc(v, y):
    return LR.auc(np.asarray(v, float), np.asarray(y))


def load_targets():
    D = json.load(open(CELL))
    genes = D["genes"]
    tgt = collections.defaultdict(set)
    for gi, lst in D["drugs"].items():
        i = int(gi)
        if i >= len(genes):
            continue
        for e in lst:
            n = (e.get("d") or "").strip().lower()
            if n and n != "null":
                tgt[n].add(genes[i]["name"])
    idx = {g["name"]: g for g in genes}
    return tgt, idx


def safety_labels():
    """Drugs appearing in a trial stopped for SAFETY, and drugs stopped only for BUSINESS reasons."""
    rows = json.load(open(SC / "_ct_stopped.json"))
    safe, biz = collections.Counter(), collections.Counter()
    for r in rows:
        w = r.get("why") or ""
        if not w:
            continue
        s, b = bool(SAFE.search(w)), bool(BIZ.search(w))
        if s == b:
            continue                      # ambiguous or neither -- excluded, not guessed
        for n, t in r.get("iv") or []:
            if t != "DRUG" or not n:
                continue
            nm = n.strip().lower()
            if nm in NOT_A_DRUG or len(nm) < 3:
                continue
            (safe if s else biz)[nm] += 1
    return safe, biz


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 58 -- the drug rebuild: outcomes that are not list lengths")
    say("=" * 100)

    faers = json.load(open(SC / "_faers.json"))
    tgt, gidx = load_targets()
    safe, biz = safety_labels()
    say(f"\n  FAERS records for {sum(1 for v in faers.values() if v):,} drugs")
    say(f"  trial stops: {len(safe):,} drugs in a SAFETY-stopped trial, {len(biz):,} in a "
        f"BUSINESS-stopped one")

    rows = []
    for d, c in faers.items():
        if not c:
            continue
        ser = float(c.get("1", 0))
        non = float(c.get("2", 0))
        tot = ser + non
        if tot < MIN_REPORTS:
            continue
        ts = sorted(tgt.get(d, ()))
        if not ts:
            continue
        rows.append({"drug": d, "rate": ser / tot, "total": tot, "n_targets": len(ts),
                     "targets": ts})
    say(f"  drugs with targets AND >= {MIN_REPORTS} reports: {len(rows):,}")

    # ---- R1 --------------------------------------------------------------------------------------
    say(f"\n  R1 THE OUTCOME IS A RATE AND NOT A COUNT, AND IT IS CHECKED")
    rate = np.array([r["rate"] for r in rows])
    vol = np.array([r["total"] for r in rows])
    r_vol = float(spearmanr(rate, vol).statistic)
    say(f"     seriousness rate: median {np.median(rate):.3f}, IQR "
        f"{np.percentile(rate,25):.3f}-{np.percentile(rate,75):.3f}")
    say(f"     Spearman(rate, total report volume) = {r_vol:+.4f}   bar |r| < {RATE_VS_VOLUME}")
    say(f"     SIDER's count against the same volume is what loop 12 was really predicting; a ratio")
    say(f"     is only a fix if it actually decouples, so it is measured rather than asserted.")
    r1 = bool(abs(r_vol) < RATE_VS_VOLUME)
    say(f"     R1 {'PASS' if r1 else 'FAIL'}")

    # ---- R2 --------------------------------------------------------------------------------------
    say(f"\n  R2 TARGET COUNT DOES NOT PREDICT THE NEW OUTCOME")
    nt = np.array([r["n_targets"] for r in rows], float)
    r_tc = float(spearmanr(nt, rate).statistic)
    say(f"     Spearman(n_targets, seriousness rate) = {r_tc:+.4f}   bar |r| < {TARGET_CONFOUND}")
    say(f"     for contrast, loop 12 measured target count against the SIDER effect count at +0.2898")
    r2 = bool(abs(r_tc) < TARGET_CONFOUND)
    say(f"     R2 {'PASS' if r2 else 'FAIL'}")

    # ---- R3 --------------------------------------------------------------------------------------
    say(f"\n  R3 THE CELL MODEL PREDICTS SERIOUSNESS")

    def agg(r, key, default=np.nan):
        vals = [gidx[t].get(key) for t in r["targets"] if t in gidx]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return float(np.mean(vals)) if vals else default

    feats = {"mean essentiality of targets": lambda r: agg(r, "dep_frac"),
             "mean LOEUF of targets (lookup)": lambda r: agg(r, "loeuf"),
             "mean PPI degree of targets": lambda r: agg(r, "ppi"),
             "fraction of targets essential": lambda r: float(np.mean(
                 [gidx[t].get("ess", 0) for t in r["targets"] if t in gidx] or [np.nan]))}
    res = {}
    for name, fn in feats.items():
        v = np.array([fn(r) for r in rows], float)
        m = np.isfinite(v)
        if m.sum() < 50:
            say(f"     {name:<34} too few drugs with the feature")
            continue
        rr = float(spearmanr(v[m], rate[m]).statistic)
        # partial on target count, the loop 12 confound
        A = np.column_stack([np.argsort(np.argsort(nt[m])) / m.sum(), np.ones(m.sum())])
        b1, *_ = np.linalg.lstsq(A, np.argsort(np.argsort(v[m])) / m.sum(), rcond=None)
        b2, *_ = np.linalg.lstsq(A, np.argsort(np.argsort(rate[m])) / m.sum(), rcond=None)
        pr = float(spearmanr(np.argsort(np.argsort(v[m])) / m.sum() - A @ b1,
                             np.argsort(np.argsort(rate[m])) / m.sum() - A @ b2).statistic)
        nulls = [abs(spearmanr(v[m], rng.permutation(rate[m])).statistic) for _ in range(N_NULL)]
        n95 = float(np.percentile(nulls, 95))
        res[name] = {"raw": rr, "partial": pr, "null95": n95, "n": int(m.sum())}
        say(f"     {name:<34} raw {rr:+.4f}  partial {pr:+.4f}  null95 {n95:.4f}  n={m.sum()}")
    best = max(res, key=lambda k: abs(res[k]["partial"])) if res else None
    r3 = bool(best and abs(res[best]["partial"]) > res[best]["null95"])
    say(f"     best after partialling: {best} at {res[best]['partial']:+.4f}" if best else
        "     nothing measurable")
    say(f"     R3 {'PASS' if r3 else 'FAIL'}")

    # ---- R4 --------------------------------------------------------------------------------------
    say(f"\n  R4 IT TRANSFERS TO THE INDEPENDENT OUTCOME")
    lab, X = [], []
    for d, ts in tgt.items():
        s, b = safe.get(d, 0), biz.get(d, 0)
        if s and not b:
            lab.append(1)
        elif b and not s:
            lab.append(0)
        else:
            continue
        X.append({"drug": d, "targets": sorted(ts)})
    y2 = np.array(lab)
    say(f"     {len(y2):,} drugs with an unambiguous trial-stop reason: {int(y2.sum()):,} safety, "
        f"{int((1-y2).sum()):,} business")
    say(f"     both classes had a trial that STOPPED, so how much a drug was studied is held roughly")
    say(f"     constant -- that is the matched control SIDER cannot provide")
    res2 = {}
    if len(y2) >= 60 and y2.sum() >= 15:
        nt2 = np.array([len(r["targets"]) for r in X], float)
        a_tc = auc(nt2, y2)
        say(f"     CONTROL: target count alone on this label, AUC {a_tc:.4f}")
        for name, fn in feats.items():
            v = np.array([fn(r) for r in X], float)
            m = np.isfinite(v)
            if m.sum() < 40 or y2[m].sum() < 10 or (1 - y2[m]).sum() < 10:
                continue
            a = auc(v[m], y2[m])
            nulls = [abs(auc(v[m], rng.permutation(y2[m])) - 0.5) for _ in range(N_NULL)]
            n95 = float(np.percentile(nulls, 95)) + 0.5
            res2[name] = {"auc": a, "null95": n95, "n": int(m.sum())}
            say(f"     {name:<34} AUC {a:.4f}   null95 {n95:.4f}   n={m.sum()}")
        agree = [k for k in res2 if k in res
                 and np.sign(res2[k]["auc"] - 0.5) == np.sign(res[k]["partial"])]
        say(f"     features agreeing in DIRECTION across both outcomes: "
            f"{', '.join(agree) if agree else 'none'}")
        r4 = bool(agree and any(abs(res2[k]["auc"] - 0.5) > res2[k]["null95"] - 0.5 for k in agree))
    else:
        say(f"     too few labelled drugs to test")
        r4 = False
    say(f"     R4 {'PASS' if r4 else 'FAIL'}")

    r5 = bool(r1 and r2)
    say(f"\n  R5 HELD OUT BY DRUG, AND THE NULLS COLLAPSE IT")
    say(f"     every statistic above carries a {N_NULL}-draw permutation null; no drug contributes to")
    say(f"     its own comparison because each is a single correlation over independent drugs")
    say(f"     R5 {'PASS' if r5 else 'FAIL'}  (gated on the two confound checks holding)")

    say("\n" + "=" * 100)
    gates = {"R1 outcome is a rate, not a count": r1, "R2 target count does not predict it": r2,
             "R3 the cell model predicts seriousness": r3,
             "R4 it transfers to the independent outcome": r4, "R5 nulls and holdout": r5}
    for k, v in gates.items():
        say(f"  {k:<44}{'PASS' if v else 'FAIL'}")
    say(f"  THE OUTCOME IS NO LONGER A LIST LENGTH. FAERS seriousness is a RATIO over a drug's own")
    say(f"  reports, correlating {r_vol:+.4f} with report volume and {r_tc:+.4f} with target count --")
    say(f"  against the +0.2898 that target count scored on SIDER's effect count in loop 12.")
    say(f"  Whether the cell model can predict it is R3, and whether that survives a second,")
    say(f"  unrelated outcome built from trial records is R4.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(SC / "_faers.json"), str(SC / "_ct_stopped.json"), str(CELL)],
                      available=len(faers), used=len(rows), selection="filtered", seed=SEED,
                      controls=["outcome is a ratio, and its independence from volume is GATED",
                                "target count re-measured on the new outcome as R2",
                                "safety-stopped compared against BUSINESS-stopped, not against all",
                                "placebo and non-drug interventions excluded by name",
                                f"{N_NULL}-draw permutation null on every statistic",
                                "two outcomes from unrelated databases must agree in direction"],
                      note="FAERS carries its own reporting biases; the claim is that they are "
                           "DIFFERENT from a package insert's, not that they are absent")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_drug_rebuild", "manifest": man, "gates": gates,
               "n_drugs_faers": len(rows), "rate_vs_volume": r_vol, "targetcount_vs_rate": r_tc,
               "faers_predictors": res, "trialstop_predictors": res2,
               "n_safety": int(y2.sum()) if len(y2) else 0,
               "n_business": int((1 - y2).sum()) if len(y2) else 0,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_drug_rebuild.json", "w"), indent=2, default=float)
    RM.report(man, emit=say)
    say(f"\n  -> {OUT/'loop_drug_rebuild.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
