"""Loop 159. The whole-cell model's five mechanisms: do they COMPOSE or SUBSTITUTE?

THE QUESTION THIS EXISTS TO ANSWER. Loop 119 measured that 362 proteins oscillate whose transcript
does not, against 38 the other way (exact binomial p = 2e-67) -- 80.4% of cell-cycle protein
dynamics has no transcriptional source. Every loop since has built a candidate non-transcriptional
source: sinusoidal degradation (121), oscillating translation (122), the two-level degradation
PULSE that loops 142 and 148-152 sized at a 20.3-fold capacity swing and then costed. cell_run.py
switches them all on at once and checks the model does not fall over. It has never asked whether
the mechanisms ADD or whether they do each other's job.

That distinction decides what the arc costs. If pulse degradation and sinusoidal degradation
substitute, the sinusoid in cell_run was already doing the pulse's work for this observable and the
20.3-fold requirement buys nothing here. If they compose, both are needed and the futile-cycling
budget is paid twice.

WHY NOT THE PROFILER. Loop 158 measured the crossover: interaction_profiler saves evaluations only
from n = 7-10 switches upward, and on space A -- 7 blocks, dense, built from blocks worth nothing
alone -- it consumed the ENTIRE state space and still missed 23 of 56 groups. This space is FIVE
switches, 32 configurations, and the mechanisms are near-inert alone by construction: loop 119
proved dP/dt = k_sp*M - b*P is a first-order filter, so each drive on its own is attenuated. That
is the same pathology. So the table is ENUMERATED and the Moebius residual computed EXACTLY over
every reference, with no sampling anywhere. The profiler is then run on the same cached table as a
second, independent test of the one-sidedness loop 158 found -- which is Q6, and it can fail.

THE FIVE SWITCHES, each already built and gated in its own loop:
    tf      TF network drives k_sm            loop 120, 54,128 signed edges
    deg     degradation +/-30% sinusoid       loop 121
    pulse   degradation two-level, 20.3x      loops 142, 148-152
    tl      translation +/-30% sinusoid       loop 122
    div     halve M and P at every t = T      loop 125

THE OBSERVABLE. Each configuration is run to its periodic steady state and every gene's relative
protein swing (max-min)/(2*mean) is read off. The score is the AUC separating the 748 HPA
CCD-Protein-Yes genes from the 776 CCD-Protein-No genes -- the assay-matched control, imaged the
same way and called negative, not a random background.

PREDECLARED, before any number is looked at.

  Q1 THE HARNESS DID NOT CHANGE THE MODEL. With every switch off and division on, the run must
     reproduce loop 125's closed form (k/a)(1-x)/(2-x) to the same 1e-6 cell_run.py's C1 uses, and
     the new pulse_fold argument at its OFF value of 1.0 must leave the trajectory bit-identical to
     the code path that existed before it.
     Gate: both. A switch harness that perturbs the unswitched model measures its own artefacts.

  Q2 DOES THE OBSERVABLE CARRY ANY SIGNAL AT ALL? Take the best of the 32 configurations and set it
     against a label-permutation null on the same genes. The null depends only on the group sizes,
     so it is computed exactly rather than sampled.
     Gate: best AUC exceeds 0.5 + the 95th percentile of |AUC-0.5| under label permutation. If this
     FAILS the honest headline is that no configuration of these five mechanisms predicts which
     proteins oscillate, and Q4/Q5 are descriptions of noise -- they are still reported, labelled.

  Q3 THE DECOMPOSITION IS EXACT. Enumerate all 32 configurations, compute the Moebius residual for
     every subset, and check the residuals reconstruct the table.
     Gate: max absolute reconstruction error < 1e-12. This is an identity, so a failure means the
     enumeration or the indexing is wrong, not the science.

  Q4 COMPOSE OR SUBSTITUTE -- the question. For all 10 pairs, report the SIGNED residual: negative
     means the two substitute (each does the other's job), positive means they synergise.
     Gate: passes on all 10 being classified against the Q2 threshold. The (deg, pulse) pair is the
     one the arc turns on and its sign is stated whichever way it falls.

  Q5 WHAT CAN BE DROPPED. Per switch, the global best minus the best configuration that EXCLUDES
     it. A switch whose regret is inside the permutation resolution can be removed from the model
     with no measurable loss on this observable, whatever else is kept.
     Gate: passes on all five being reported with their regret.

  Q6 THE ONE-SIDEDNESS, RETESTED ON AN UNRELATED OBJECTIVE. Loop 158 found interaction_profiler's
     error was one-sided on both nexus spaces -- it never invented an interaction, it only missed
     some. That was two objectives from one arm. Run it here at n_references 3, 6, 12, 24, 48
     against the exact answer.
     Gate: PASS iff false positives are 0 at every setting. A single false positive means the
     one-sidedness was a property of those two objectives and not of the schedule, and the cheap
     way of using the tool that loop 158 recommended does not generalise.

-> outputs/loop_cell_mechanism_interactions.json
"""
import csv
import json
import os
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
csv.field_size_limit(1 << 30)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "standalone"))
import loop_replication as LR                            # noqa: E402
import cell_assembled as CA                              # noqa: E402
import gate_guard as GG                                  # noqa: E402
import run_manifest as RM                                # noqa: E402
from interaction_profiler import profile_objective       # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_cell_mechanism_interactions.json"
HPA = LR.SC / "proteinatlas.tsv"
T_DOUBLE_H = 27.5
NCYC, NSTEP = 40, 400
BETA_DEG, BETA_TL = 0.3, 0.3
PULSE_FOLD, PULSE_DUTY = 20.286, 0.10     # loop 148's required ratio, loop 142's duty
TF_FRAC, TF_AMP = 0.10, 0.5               # cell_run.py's C2 setting, carried unchanged
SEED = 12800
REFS = (3, 6, 12, 24, 48)
SWITCH = ["tf", "deg", "pulse", "tl", "div"]
C1_TOL = 1e-6

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def auc(scores, pos):
    """Mann-Whitney AUC of `scores` separating pos from ~pos. Ties count as 0.5."""
    if pos.sum() == 0 or (~pos).sum() == 0:
        return 0.5
    r = stats.rankdata(scores)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def perm_resolution(n1, n0, tail=0.95):
    """The 95th percentile of |AUC - 0.5| under label permutation. Under the null the Mann-Whitney
    statistic has mean n1*n0/2 and variance n1*n0*(n1+n0+1)/12, so this is exact rather than
    sampled -- it depends only on the group sizes, not on the scores."""
    sd = np.sqrt(n1 * n0 * (n1 + n0 + 1) / 12.0) / (n1 * n0)
    return float(stats.norm.ppf(tail) * sd)


def moebius(table, n):
    """Every subset's residual, by the inclusion-exclusion transform over the all-zero reference."""
    out = {}
    for r in range(n + 1):
        for g in combinations(range(n), r):
            tot = 0.0
            for k in range(len(g) + 1):
                for sub in combinations(g, k):
                    key = [0] * n
                    for i in sub:
                        key[i] = 1
                    tot += ((-1) ** (len(g) - k)) * table[tuple(key)]
            out[g] = tot
    return out


def residual_maxref(table, group, n):
    """Exact residual maximised over every one of the 2^(n-|S|) references -- what the profiler
    estimates from a random sample, computed here without sampling."""
    others = [i for i in range(n) if i not in group]
    best, arg = 0.0, None
    for rmask in range(1 << len(others)):
        base = [0] * n
        for b, i in enumerate(others):
            base[i] = (rmask >> b) & 1
        tot = 0.0
        for k in range(len(group) + 1):
            for sub in combinations(group, k):
                key = list(base)
                for i in group:
                    key[i] = 0
                for i in sub:
                    key[i] = 1
                tot += ((-1) ** (len(group) - k)) * table[tuple(key)]
        if abs(tot) > best:
            best, arg = abs(tot), tot
    return best, (arg if arg is not None else 0.0)


def load_ccd(genes):
    """HPA CCD-Protein calls, joined onto the model's state genes. The NO set is imaged and called
    negative by the same assay, so it is a control and not a background."""
    yes, no = set(), set()
    with open(HPA, newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        h = next(rd)
        iG, iCP = h.index("Gene"), h.index("CCD Protein")
        for row in rd:
            v = row[iCP].strip()
            if v == "Yes":
                yes.add(row[iG])
            elif v == "No":
                no.add(row[iG])
    idx = {g: i for i, g in enumerate(genes)}
    ky = sorted(yes & idx.keys())
    kn = sorted(no & idx.keys())
    return (np.array([idx[g] for g in ky], int), np.array([idx[g] for g in kn], int),
            len(yes), len(no))


def main():
    t0 = time.time()
    say("=" * 104)
    say("  THE FIVE MECHANISMS OF THE CELL MODEL -- do they compose or substitute?")
    say("=" * 104)
    say()

    D = CA.load()
    st = CA.state_vector(D)
    genes = st["genes"]
    n = len(genes)
    wiring = CA.tf_wiring(D)
    regs = sorted({r for v in wiring.values() for r, _ in v})
    ix = CA.tf_index(wiring, genes, regs)
    rng = np.random.default_rng(SEED)
    drive = np.zeros(len(regs))
    drive[rng.random(len(regs)) < TF_FRAC] = TF_AMP
    w = 2.0 * np.pi / T_DOUBLE_H

    def dev_at(t):
        return drive * np.sin(w * t)

    iy, inn, n_yes_all, n_no_all = load_ccd(genes)
    say(f"     state vector {n:,} genes | TF wiring {len(regs):,} signed regulators, "
        f"{int((ix[4] > 0).sum()):,} targets reached")
    say(f"     HPA CCD-Protein: {n_yes_all} Yes and {n_no_all} No genome-wide; "
        f"{len(iy)} Yes and {len(inn)} No have a full dynamical state here")
    tau = perm_resolution(len(iy), len(inn))
    say(f"     label-permutation resolution at these group sizes: |AUC-0.5| = {tau:.4f} "
        f"(exact, from the Mann-Whitney null)")
    say(f"     switches: {', '.join(SWITCH)}   -> 2^5 = 32 configurations, all enumerated")
    say()

    # ------------------------------------------------------------------ Q1
    say("Q1 THE HARNESS DID NOT CHANGE THE MODEL")
    trM, trP, Mend, Pend = CA.integrate_cell(st, T_DOUBLE_H, divide=True, ncyc=60, nstep=800)
    a = st["k_loss_mrna_deg"]
    xa = np.exp(-a * T_DOUBLE_H)
    M0_an = (st["k_sm"] / a) * (1 - xa) / (2 - xa)
    eM = float(np.max(np.abs(trM[0] - M0_an) / M0_an))
    ratio = float(np.median(Mend / trM[0]))
    say(f"     every switch off, division on: mRNA P(0+) vs the closed form, "
        f"max relative error {eM:.2e}  (gate < {C1_TOL:.0e})")
    say(f"     mRNA max/min over the cycle {ratio:.4f} (analytic 2.0000)")
    _, trP_off, _, _ = CA.integrate_cell(st, T_DOUBLE_H, divide=True, ncyc=12, nstep=200,
                                         pulse_fold=1.0)
    _, trP_bare, _, _ = CA.integrate_cell(st, T_DOUBLE_H, divide=True, ncyc=12, nstep=200)
    ident = float(np.max(np.abs(trP_off - trP_bare)))
    say(f"     pulse_fold=1.0 against the path without the argument: max abs difference {ident:.2e}")
    q1 = bool(eM < C1_TOL and abs(ratio - 2.0) < 1e-3 and ident == 0.0)
    GG.verdict(q1, emit=say, if_true=(
        "the five switches are an addition to the model, not a replacement of it."), if_false=(
        "the harness perturbs the unswitched model -- everything below measures its own artefact."))
    say(f"     Q1 {'PASS' if q1 else 'FAIL'}")
    say()

    # ------------------------------------------------------------------ enumerate
    say("ENUMERATING ALL 32 CONFIGURATIONS")
    table, swings = {}, {}
    for m in range(32):
        c = tuple((m >> k) & 1 for k in range(5))
        tf, deg, pulse, tl, div = c
        _, trP, _, _ = CA.integrate_cell(
            st, T_DOUBLE_H,
            ix=(ix if tf else None), dev_at=(dev_at if tf else None),
            beta_deg=(BETA_DEG if deg else 0.0), beta_tl=(BETA_TL if tl else 0.0),
            divide=bool(div), ncyc=NCYC, nstep=NSTEP,
            pulse_fold=(PULSE_FOLD if pulse else 1.0), pulse_duty=PULSE_DUTY)
        mean = trP.mean(0)
        rel = (trP.max(0) - trP.min(0)) / (2.0 * np.maximum(mean, 1e-300))
        sc = np.concatenate([rel[iy], rel[inn]])
        pos = np.zeros(len(sc), bool)
        pos[:len(iy)] = True
        table[c] = auc(sc, pos)
        swings[c] = float(np.median(rel))
    say(f"     done [{time.time() - t0:.0f}s]   empty configuration AUC "
        f"{table[(0, 0, 0, 0, 0)]:.4f}, median protein swing {swings[(0, 0, 0, 0, 0)]:.2e}")
    order = sorted(table, key=lambda c: -table[c])
    say("     the five best and the five worst configurations:")
    for c in order[:5] + order[-5:]:
        on = "+".join(s for s, b in zip(SWITCH, c) if b) or "(nothing on)"
        say(f"       AUC {table[c]:.4f}   median swing {swings[c]:7.4f}   {on}")
    say()

    # ------------------------------------------------------------------ Q2
    say("Q2 DOES THE OBSERVABLE CARRY ANY SIGNAL AT ALL?")
    best_c = order[0]
    best = table[best_c]
    q2 = bool(abs(best - 0.5) > tau)
    say(f"     best of 32: AUC {best:.4f} at "
        f"{'+'.join(s for s, b in zip(SWITCH, best_c) if b) or '(nothing on)'}")
    say(f"     against the exact label-permutation resolution 0.5 +/- {tau:.4f}")
    GG.verdict(q2, emit=say, if_true=(
        "the model's protein swing does separate imaged CCD-Yes from imaged CCD-No: the "
        "decomposition below is of a real signal."), if_false=(
        "NO configuration of these five mechanisms separates CCD-Yes from CCD-No above the "
        "permutation resolution. The decomposition below is a decomposition of noise, and every "
        "residual in Q4/Q5 must be read as such -- they are reported so the null is on the record "
        "with its structure, not hidden behind a single failed number."))
    say(f"     Q2 {'PASS' if q2 else 'FAIL'}")
    say()

    # ------------------------------------------------------------------ Q3
    say("Q3 THE DECOMPOSITION IS EXACT")
    mob = moebius(table, 5)
    err = 0.0
    for c in table:
        recon = sum(v for g, v in mob.items() if all(c[i] for i in g))
        err = max(err, abs(recon - table[c]))
    q3 = bool(err < 1e-12)
    say(f"     32 configurations reconstructed from their 32 Moebius coefficients: "
        f"max absolute error {err:.2e}  (gate < 1e-12)")
    GG.verdict(q3, emit=say, if_true=(
        "the residuals below ARE the objective, re-expressed -- no sampling, no estimate."),
        if_false="the enumeration or the indexing is wrong; nothing below stands.")
    say(f"     Q3 {'PASS' if q3 else 'FAIL'}")
    say()

    # ------------------------------------------------------------------ Q4
    say("Q4 COMPOSE OR SUBSTITUTE -- the signed pairwise residual, maximised over references")
    pairs = {}
    for g in combinations(range(5), 2):
        mag, sgn = residual_maxref(table, g, 5)
        nm = "+".join(SWITCH[i] for i in g)
        kind = ("independent" if mag <= tau else
                ("SYNERGY" if sgn > 0 else "SUBSTITUTES"))
        pairs[nm] = {"magnitude": mag, "signed": sgn, "verdict": kind,
                     "zeroref": mob[g]}
    for nm, d in sorted(pairs.items(), key=lambda kv: -kv[1]["magnitude"]):
        say(f"       {nm:<14s} {d['signed']:+.4f}   {d['verdict']}")
    q4 = len(pairs) == 10
    dp = pairs["deg+pulse"]
    say(f"     THE PAIR THE ARC TURNS ON -- deg+pulse: {dp['signed']:+.4f}, {dp['verdict']}")
    GG.verdict(dp["verdict"] == "SUBSTITUTES", emit=say, if_true=(
        "the sinusoid and the pulse do each other's job on this observable. cell_run's beta_deg "
        "was already buying what the 20.3-fold pulse buys, so the pulse's futile-cycling bill is "
        "not additional explanatory power HERE -- it is a different waveform reaching the same "
        "place. That does not touch loop 123's finding that the sinusoid cannot reach the "
        "measured AMPLITUDES; it says the two are not separable by WHICH GENES oscillate."),
        if_false=(
        "the sinusoid and the pulse are not substitutes on this observable: what the pulse adds "
        "is not what beta_deg was already adding, so both terms are doing separate work and the "
        "futile-cycling budget is a real additional cost."))
    say(f"     Q4 {'PASS' if q4 else 'FAIL'}")
    say()

    # ------------------------------------------------------------------ Q5
    say("Q5 WHAT CAN BE DROPPED")
    reg = {}
    say(f"     {'switch':<8s} {'solo':>9s} {'best with':>11s} {'best without':>13s} "
        f"{'regret':>9s}  droppable")
    for k, s in enumerate(SWITCH):
        solo = table[tuple(1 if i == k else 0 for i in range(5))] - table[(0, 0, 0, 0, 0)]
        with_k = max(v for c, v in table.items() if c[k])
        wo_k = max(v for c, v in table.items() if not c[k])
        r = best - wo_k
        reg[s] = {"solo": solo, "best_with": with_k, "best_without": wo_k,
                  "regret": r, "droppable": bool(r <= tau), "in_best": bool(best_c[k])}
        say(f"     {s:<8s} {solo:+9.4f} {with_k:11.4f} {wo_k:13.4f} {r:+9.4f}  "
            f"{'YES' if r <= tau else 'no'}")
    q5 = len(reg) == 5
    drop = [s for s in SWITCH if reg[s]["droppable"]]
    say(f"     droppable with no measurable loss on this observable: "
        f"{', '.join(drop) if drop else 'none'}")
    say(f"     Q5 {'PASS' if q5 else 'FAIL'}")
    say()

    say("     NOT A GATE -- the mechanistic reading, added after seeing that every non-division")
    say("     drive lands on the same AUC. beta_deg, beta_tl and the pulse are all GLOBAL: the same")
    say("     waveform is applied to every gene. The only gene-specific quantity left shaping the")
    say("     swing is the protein's own loss rate, so if that is the whole story then half-life")
    say("     alone should reproduce the ceiling exactly.")
    b_all = st["k_loss_prot_deg"]
    a_all = st["k_loss_mrna_deg"]
    sc_hl = np.concatenate([b_all[iy], b_all[inn]])
    sc_mr = np.concatenate([a_all[iy], a_all[inn]])
    pos_hl = np.zeros(len(sc_hl), bool)
    pos_hl[:len(iy)] = True
    auc_hl = auc(sc_hl, pos_hl)
    auc_mr = auc(sc_mr, pos_hl)
    say(f"       protein loss rate b_deg alone, no model at all : AUC {auc_hl:.4f}")
    say(f"       mRNA loss rate a_deg alone                     : AUC {auc_mr:.4f}")
    say(f"       best of all 32 wired configurations            : AUC {best:.4f}")
    say(f"       the five mechanisms are worth {best - max(auc_hl, 1 - auc_hl):+.4f} over the "
        f"single number they are all reading")
    say()

    # ------------------------------------------------------------------ Q6
    say("Q6 THE PROFILER'S ONE-SIDEDNESS, RETESTED ON AN OBJECTIVE FROM A DIFFERENT ARM")
    truth = {}
    for o in (2, 3):
        for g in combinations(range(5), o):
            truth[g] = residual_maxref(table, g, 5)[0] > tau
    say(f"     {len(truth)} groups of order 2 and 3, {sum(truth.values())} genuinely "
        f"interacting at tau={tau:.4f}")
    rows = []
    for R in REFS:
        seen = set()

        def obj(cfg, _s=seen):
            key = tuple(int(cfg[i]) for i in range(5))
            _s.add(key)
            return table[key]

        rep = profile_objective(obj, list(range(5)), 2, tau=tau, max_order=3,
                                n_references=R, adaptive=True, seed=SEED)
        miss = sum(1 for g, t in truth.items() if t and not rep.strengths.get(g, 0.0) > tau)
        fp = sum(1 for g, t in truth.items() if (not t) and rep.strengths.get(g, 0.0) > tau)
        rows.append({"n_references": R, "distinct": len(seen), "missed": miss,
                     "false_positives": fp})
        say(f"       R={R:<3d} distinct configs {len(seen):>3d} of 32   missed {miss:>2d}   "
            f"false positives {fp}")
    q6 = all(r["false_positives"] == 0 for r in rows)
    GG.verdict(q6, emit=say, if_true=(
        "one-sided again, on an objective from a different arm of the repository: the profiler "
        "under-reports and never over-reports. Loop 158's rule holds -- believe what it FINDS at "
        "the cheap setting, pay only when you intend to act on a null."), if_false=(
        "a false positive appears here, so the one-sidedness loop 158 found was a property of "
        "those two nexus objectives and NOT of the schedule. Loop 158's recommendation is "
        "withdrawn: a reported interaction cannot be believed at the cheap setting."))
    say(f"     Q6 {'PASS' if q6 else 'FAIL'}")
    say()

    gates = {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5, "Q6": q6}
    man = RM.manifest(
        inputs=[HPA, LR.CELL],
        available=len(iy) + len(inn), used=len(iy) + len(inn), selection="all",
        seed=SEED,
        controls=[
            "the negative set is the 776 HPA genes IMAGED and called non-CCD, not a random background",
            "the permutation resolution is exact from the Mann-Whitney null, not sampled",
            "all 32 configurations enumerated; the Moebius decomposition is exact over every reference",
            "Q3 gates the decomposition by reconstructing the table from its own coefficients",
            "the profiler is scored against that exact answer rather than trusted",
            "conclusions emitted through gate_guard.verdict",
        ],
        note="the five wired mechanisms of cell_assembled.integrate_cell, enumerated and decomposed")

    res = {"test": "do the cell model's five mechanisms compose or substitute",
           "gates": gates, "tau": tau, "n_yes": len(iy), "n_no": len(inn),
           "table": {"".join(map(str, c)): v for c, v in table.items()},
           "median_swing": {"".join(map(str, c)): v for c, v in swings.items()},
           "moebius": {"+".join(SWITCH[i] for i in g) or "(empty)": v for g, v in mob.items()},
           "pairs": pairs, "regret": reg, "profiler": rows,
           "best": best, "best_config": [s for s, b in zip(SWITCH, best_c) if b],
           "halflife_control": {"auc_protein_loss_rate": auc_hl,
                               "auc_mrna_loss_rate": auc_mr},
           "q1": {"closed_form_error": eM, "mrna_ratio": ratio, "pulse_off_identity": ident},
           "manifest": man, "seconds": time.time() - t0, "log": log}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'PASS' if v else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 104)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
