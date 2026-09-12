"""LOOP 130 -- DO COLLECTRI'S SIGNS CARRY INFORMATION, where this model's do not?

THE QUESTION THE COVERAGE COUNT COULD NOT ANSWER. The gap-2 fetch brought in OmniPath's signed
networks and measured the wrong thing about them: CollecTRI, DoRothEA and tf_target together give
47,613 signed edges with both ends in the model against the 54,128 already here -- fewer, 0.88x --
so by coverage the fetch failed. But coverage was never the binding problem.

Loop 120 put the existing signs into a rate law and found they carry NOTHING:

    real edge signs      AUC 0.5465
    SHUFFLED edge signs  AUC 0.5494 +/- 0.0079     the shuffle scored HIGHER
    regulator count alone AUC 0.5806               the biggest signal in the loop was annotation effort

So the useful question is not whether CollecTRI is bigger, it is whether it is RIGHT. CollecTRI is
literature-curated with explicit mode-of-regulation calls and per-edge references, which is a
different kind of object from an aggregate whose signs came from mixed sources. This loop runs
loop 120's protocol on it, unchanged, so the two are directly comparable.

WHAT IS HELD FIXED, so that only the network varies: the same measured CCD-transcript calls from
Mahdessian via HPA, the same mRNA state from Schwanhausser, the same k_sm drive through
cell_assembled.integrate_tf, the same statistic, the same nulls. Loop 120's numbers are the
control arm and they are quoted, not recomputed differently.

DISCLOSED, measured during the fetch: CollecTRI contributes 45,507 signed edges with both ends in
the model, from 64,515 raw. The union with DoRothEA and tf_target is 47,613 over 1,336 regulators
and 6,799 targets, and reaches 38.1% of state genes against the existing network's 42.9%.

PREDECLARED:

  T1 THE WIRING BUILDS AND REACHES ENOUGH TO TEST                   THE PREREQUISITE.
       CollecTRI through tf_wiring's data structure, and the test set is genes with a measured
       CCD-transcript call, a CollecTRI regulator and an mRNA state. Gate: at least 100 such genes
       with both classes represented. Loop 120 had 273; a much smaller set would make any
       comparison with it meaningless rather than informative.
  T2 DO THE SIGNS DO WORK?                                          LOOP 120's W1, RERUN.
       integrated mRNA amplitude with real CollecTRI signs against the same wiring with signs
       shuffled, 64 shuffles, capability-checked. Gate: real must beat the shuffle mean by more
       than two shuffle standard deviations. Loop 120's existing network FAILED this -- the shuffle
       scored higher. This is the whole point of the loop.
  T3 CCD TRANSCRIPTS HAVE CCD-PROTEIN REGULATORS ABOVE CHANCE       LOOP 120's W2, RERUN.
       null: cell-cycle status shuffled among regulators WITHIN out-degree deciles, so hub status
       is preserved. Gate: |z| >= 2 in the predicted direction. Loop 120 got z = 0.63.
  T4 THE REGULATOR-COUNT CONFOUND                                   THE ONE THAT KILLED LOOP 120.
       regulator count alone scored AUC 0.5806 there, above every network score, and holding it
       fixed by quintile collapsed everything toward 0.5. Gate: the CollecTRI score, stratified by
       regulator-count quintile, must stay above 0.55. A network that only knows how many
       regulators a gene has been assigned is measuring annotation effort.
  T5 FAME                                                           THE CONTROL THAT KEEPS WINNING.
       the gene's own publication count and its regulators' mean, on the same genes. Gate: the
       network must beat both. Loop 120 lost to the gene's own count at 0.5536.
  T6 HEAD TO HEAD ON IDENTICAL GENES                                THE COMPARISON.
       CollecTRI against the existing network on the intersection, same statistic, same nulls, so
       the difference is the network and nothing else. Gate: report both; pass if the comparison is
       made on >= 50 shared genes with both scores computed.

-> outputs/loop_collectri.json
"""
import collections
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(1 << 30)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import gate_guard as GG  # noqa: E402
import cell_assembled as CA  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
CTRI = LR.SC / "omnipath_collectri.tsv"
HPA = LR.SC / "proteinatlas.tsv"
SEED = 13000
NPERM = 2000
NSHUF = 64
LN2 = float(np.log(2.0))
T_DOUBLE_H = 27.5
MU = LN2 / T_DOUBLE_H
T_CYCLE = 24.0
DRIVE_AMP = 0.50

T1_MIN = 100
T2_SD = 2.0
T3_Z = 2.0
T4_AUC = 0.55
T6_MIN = 50

# loop 120's numbers, quoted as the control arm rather than recomputed
L120 = {"auc_real": 0.5465, "auc_shuf": 0.5494, "auc_shuf_sd": 0.0079,
        "w2_real": 0.0986, "w2_null": 0.0486, "w2_sd": 0.0792, "w2_z": 0.63,
        "auc_regcount": 0.5806, "auc_own_pubs": 0.5536, "n": 273}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    sv, r = allv[order], np.empty(len(allv))
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def deciles(v, k=10):
    v = np.asarray(v, float)
    q = np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(q, v, side="right")


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 130 -- do CollecTRI's signs carry information, where this model's do not?")
    say("=" * 100)
    say()

    D = CA.load()
    names = set(D["names"])
    S = D["schwan"]
    pubs = D["pubs"]

    wiring = collections.defaultdict(list)
    r = csv.reader(open(CTRI), delimiter="\t")
    h = next(r)
    iS, iT = h.index("source_genesymbol"), h.index("target_genesymbol")
    iSt, iIn = h.index("is_stimulation"), h.index("is_inhibition")
    raw = 0
    for x in r:
        raw += 1
        s, t = x[iS], x[iT]
        sg = 1 if x[iSt] == "True" else (-1 if x[iIn] == "True" else 0)
        if sg and s in names and t in names:
            wiring[t].append((s, sg))
    regs = sorted({s for v in wiring.values() for s, _ in v})
    nsig = sum(len(v) for v in wiring.values())
    say(f"  CollecTRI: {raw:,} raw edges -> {nsig:,} signed with both ends in the model, "
        f"{len(regs):,} regulators, {len(wiring):,} targets")
    say(f"  the existing network, for comparison: 54,128 signed, 1,451 regulators, 7,485 targets")

    with open(HPA, newline="") as f:
        rr = csv.reader(f, delimiter="\t")
        hh = next(rr)
        jG, jP, jT = hh.index("Gene"), hh.index("CCD Protein"), hh.index("CCD Transcript")
        rows = [(x[jG], x[jP], x[jT]) for x in rr]
    cp = {g: v for g, v, _ in rows if v in ("Yes", "No")}
    ct = {g: v for g, _, v in rows if v in ("Yes", "No")}
    ccdP = {g for g, v in cp.items() if v == "Yes"}

    kbar = {g: S[g]["mrna_copies"] * (LN2 / S[g]["mrna_hl_h"] + MU)
            for g in S if S[g].get("mrna_copies") and S[g].get("mrna_hl_h")}
    a_all = {g: LN2 / S[g]["mrna_hl_h"] + MU for g in kbar}

    gates = {}

    # ---------------------------------------------------------------- T1
    say("T1 THE WIRING BUILDS AND REACHES ENOUGH TO TEST")
    tgt = sorted(g for g in ct if g in wiring and g in kbar)
    y = np.array([ct[g] == "Yes" for g in tgt])
    say(f"     test set: {len(tgt):,} genes with a CCD-transcript call, a CollecTRI regulator and "
        f"an mRNA state")
    say(f"     {int(y.sum())} Yes / {int((~y).sum())} No   (loop 120 had {L120['n']}: 124/149)")
    say(f"     CCD proteins that are CollecTRI regulators: {len(ccdP & set(regs))} of {len(ccdP)}")
    gates["T1"] = bool(len(tgt) >= T1_MIN and y.sum() >= 10 and (~y).sum() >= 10)
    say(f"     T1 {'PASS' if gates['T1'] else 'FAIL'}")
    say()
    if not gates["T1"]:
        say("  test set too small; the remaining gates cannot be compared with loop 120")

    ix = CA.tf_index(wiring, tgt, regs)
    nreg = ix[4]
    kb = np.array([kbar[g] for g in tgt])
    aa = np.array([a_all[g] for g in tgt])
    drive_vec = np.array([g in ccdP for g in regs], float)
    w = 2.0 * np.pi / T_CYCLE

    def dev_sin(vec):
        return lambda t: vec * DRIVE_AMP * np.sin(w * t)

    # ---------------------------------------------------------------- T2
    say("T2 DO THE SIGNS DO WORK?")
    relM, _ = CA.integrate_tf(kb, aa, ix, dev_sin(drive_vec), T_CYCLE)
    a_real = auc(relM[y], relM[~y])
    rows_, cols_, sgn_, nsafe_, nreg_ = ix
    sh, cap = [], None
    for i in range(NSHUF):
        p = rng.permutation(len(sgn_))
        if i == 0:
            cap = GG.null_can_move(sgn_, sgn_[p])
        rs, _ = CA.integrate_tf(kb, aa, (rows_, cols_, sgn_[p], nsafe_, nreg_),
                                dev_sin(drive_vec), T_CYCLE)
        sh.append(auc(rs[y], rs[~y]))
    sh = np.array(sh)
    say(f"     real CollecTRI signs   AUC {a_real:.4f}")
    say(f"     SHUFFLED signs         AUC {sh.mean():.4f} +/- {sh.std():.4f}  ({NSHUF} shuffles)")
    say(f"     null capability: {cap['reason']}")
    say(f"     loop 120, existing network: real {L120['auc_real']:.4f} against shuffled "
        f"{L120['auc_shuf']:.4f} +/- {L120['auc_shuf_sd']:.4f} -- the shuffle scored HIGHER")
    gates["T2"] = bool(cap["capable"]
                       and abs(a_real - 0.5) > abs(sh.mean() - 0.5) + T2_SD * sh.std())
    say(f"     T2 {'PASS' if gates['T2'] else 'FAIL'} -- CollecTRI's signs "
        f"{'carry information the edges alone do not' if gates['T2'] else 'carry NO more than shuffled ones, same as the existing network'}")
    say()

    # ---------------------------------------------------------------- T3
    say("T3 CCD TRANSCRIPTS HAVE CCD-PROTEIN REGULATORS ABOVE CHANCE")
    has = np.array([any(rg in ccdP for rg, _ in wiring[g]) for g in tgt])
    real3 = float(has[y].mean() - has[~y].mean())
    outdeg = collections.Counter(s for v in wiring.values() for s, _ in v)
    strata = deciles([outdeg[rg] for rg in regs])
    isccd = np.array([rg in ccdP for rg in regs])
    nulls, cap3 = [], None
    for i in range(NPERM):
        shf = isccd.copy()
        for s_ in np.unique(strata):
            m = strata == s_
            shf[m] = rng.permutation(shf[m])
        if i == 0:
            cap3 = GG.null_can_move(isccd.astype(int), shf.astype(int))
        ss = {regs[j] for j in np.flatnonzero(shf)}
        hn = np.array([any(rg in ss for rg, _ in wiring[g]) for g in tgt])
        nulls.append(float(hn[y].mean() - hn[~y].mean()))
    sur = GG.survival(real3, nulls, z_min=T3_Z)
    say(f"     CCD transcripts with >=1 CCD-protein regulator {has[y].mean():.1%}, "
        f"non-CCD {has[~y].mean():.1%}")
    GG.report("enrichment difference", sur, emit=say)
    say(f"     loop 120: real {L120['w2_real']:+.4f}, null {L120['w2_null']:+.4f} "
        f"+/- {L120['w2_sd']:.4f}, z {L120['w2_z']:.2f}")
    gates["T3"] = bool(cap3["capable"] and np.isfinite(sur.get("z", np.nan))
                       and abs(sur["z"]) >= T3_Z and real3 > 0)
    say(f"     T3 {'PASS' if gates['T3'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- T4
    say("T4 THE REGULATOR-COUNT CONFOUND")
    a_cnt = auc(nreg[y], nreg[~y])
    say(f"     regulator count alone   AUC {a_cnt:.4f}   (loop 120: {L120['auc_regcount']:.4f})")
    dec = deciles(nreg, k=5)
    num, den = 0.0, 0.0
    for s_ in np.unique(dec):
        m = dec == s_
        if y[m].sum() and (~y[m]).sum():
            wgt = float(m.sum())
            num += wgt * auc(relM[m][y[m]], relM[m][~y[m]])
            den += wgt
    a_strat = num / den if den else float("nan")
    say(f"     CollecTRI integrated    AUC {a_real:.4f} unstratified, {a_strat:.4f} holding "
        f"regulator count fixed by quintile")
    say(f"     loop 120's network collapsed from 0.5465 to 0.5348 under the same stratification")
    gates["T4"] = bool(np.isfinite(a_strat) and a_strat > T4_AUC)
    say(f"     T4 {'PASS' if gates['T4'] else 'FAIL'} -- the network "
        f"{'survives the confound' if gates['T4'] else 'does NOT survive; it is measuring how many regulators a gene has been assigned'}")
    say()

    # ---------------------------------------------------------------- T5
    say("T5 FAME")
    own = np.array([pubs.get(g, 0.0) for g in tgt])
    regf = np.array([np.mean([pubs.get(rg, 0.0) for rg, _ in wiring[g]]) for g in tgt])
    a_own, a_regf = auc(own[y], own[~y]), auc(regf[y], regf[~y])
    say(f"     the gene's own publication count   AUC {a_own:.4f}   "
        f"(loop 120: {L120['auc_own_pubs']:.4f})")
    say(f"     mean count of its regulators       AUC {a_regf:.4f}")
    say(f"     CollecTRI integrated               AUC {a_real:.4f}")
    gates["T5"] = bool(abs(a_real - 0.5) > abs(a_own - 0.5)
                       and abs(a_real - 0.5) > abs(a_regf - 0.5))
    say(f"     T5 {'PASS' if gates['T5'] else 'FAIL'} -- "
        f"{'the network beats fame' if gates['T5'] else 'FAME WINS again'}")
    say()

    # ---------------------------------------------------------------- T6
    say("T6 HEAD TO HEAD ON IDENTICAL GENES")
    old = CA.tf_wiring(D)
    shared = [g for g in tgt if g in old]
    say(f"     {len(shared):,} genes carried by BOTH networks")
    if len(shared) >= T6_MIN:
        ys = np.array([ct[g] == "Yes" for g in shared])
        oregs = sorted({rr2 for v in old.values() for rr2, _ in v})
        ixo = CA.tf_index(old, shared, oregs)
        dvo = np.array([g in ccdP for g in oregs], float)
        kbs = np.array([kbar[g] for g in shared])
        aas = np.array([a_all[g] for g in shared])
        ro, _ = CA.integrate_tf(kbs, aas, ixo, dev_sin(dvo), T_CYCLE)
        ixc = CA.tf_index(wiring, shared, regs)
        rc, _ = CA.integrate_tf(kbs, aas, ixc, dev_sin(drive_vec), T_CYCLE)
        a_o, a_c = auc(ro[ys], ro[~ys]), auc(rc[ys], rc[~ys])
        say(f"     existing network  AUC {a_o:.4f}")
        say(f"     CollecTRI         AUC {a_c:.4f}")
        say(f"     same genes, same statistic, same drive -- the only thing that varies is the "
            f"network")
        gates["T6"] = True
    else:
        a_o = a_c = float("nan")
        gates["T6"] = False
        say(f"     fewer than {T6_MIN} shared genes; not comparable")
    say(f"     T6 {'PASS' if gates['T6'] else 'FAIL'}")
    say()

    say("=" * 100)
    for k in ("T1", "T2", "T3", "T4", "T5", "T6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/6")
    say("=" * 100)

    man = RM.manifest(inputs=[CTRI, HPA, LR.CELL, LR.SC / "_schwan2011.json"],
                      available=raw, used=len(tgt), selection="filtered", seed=SEED,
                      controls=["loop 120's protocol run unchanged so the two are comparable",
                                "edge signs shuffled, capability-checked, integrator re-run",
                                "cell-cycle status shuffled within out-degree deciles",
                                "regulator count held fixed by quintile",
                                "the gene's own publication count and its regulators' mean",
                                "the existing network scored on the identical shared genes"],
                      note="the gap-2 fetch measured coverage and CollecTRI lost; this measures "
                           "whether it is RIGHT, which is the question that mattered")
    RM.report(man, emit=say)
    json.dump({"test": "loop_collectri", "manifest": man, "gates": gates, "loop120": L120,
               "wiring": {"raw": raw, "signed_in_model": nsig, "regulators": len(regs),
                          "targets": len(wiring)},
               "t1": {"n": len(tgt), "n_yes": int(y.sum()), "n_no": int((~y).sum()),
                      "ccd_regulators": len(ccdP & set(regs))},
               "t2": {"auc_real": a_real, "auc_shuf": float(sh.mean()),
                      "auc_shuf_sd": float(sh.std()), "capability": cap},
               "t3": {"frac_yes": float(has[y].mean()), "frac_no": float(has[~y].mean()),
                      "difference": real3, "survival": sur},
               "t4": {"auc_regcount": a_cnt, "auc_stratified": a_strat},
               "t5": {"auc_own_pubs": a_own, "auc_regulator_pubs": a_regf},
               "t6": {"n_shared": len(shared), "auc_existing": a_o, "auc_collectri": a_c},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_collectri.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_collectri.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
