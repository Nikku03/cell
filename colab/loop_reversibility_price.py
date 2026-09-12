"""LOOP 151 -- CAN WE TEST WHETHER CHAINS STAY REVERSIBLE OUT TO 8? THE PRICE SAYS YES.

Loop 150's R4 left the arc on one sentence: the amplification ceiling is r^c where c is the
COMMITMENT length -- the point past which the proteasome has the substrate and no DUB gets it back
-- and not r^n. At Thrower's minimum c = 4 the ceiling is 5.18x against 20.29x required. So the
median-receptor branch survives only if polyubiquitin chains on pulsed substrates stay
DUB-reversible out to length 8.

Nothing on disk measures a chain-length distribution. Ub-clipping mass spectrometry (Swatek 2019)
does exactly that and is not here; neither is a DUB-inhibition kinetic series. On the direct
evidence the answer is no, we cannot test it, and T6 says so plainly at the end.

BUT REVERSIBILITY IS NOT FREE, AND THE PRICE IS COMPUTABLE FROM CENSUSES THIS REPO ALREADY HAS.
A reversible ladder is a futile cycle. Every ubiquitin a DUB strips off is released as free
ubiquitin and has to be re-activated before it can go back on: E1 + Ub + ATP -> E1~Ub + AMP + PPi,
which is one ATP hydrolysed to AMP, two phosphoanhydride bonds. A chain that fails and restarts
thousands of times before it completes is a chain that burns fuel thousands of times, and BOTH
sides of that are measured -- the E1 census, the ubiquitin census, the proteolytic load and the
cell's translational ATP bill are all in this repo, and three of the four come from the same
Schwanhausser proteome so loop 92's abundance rule holds.

THE ARITHMETIC THAT MAKES IT A TEST. For the constant-rate ladder the expected number of ubiquitin
transfer events before one molecule is finally degraded is exactly lambda*T, and since T =
f_shape(rho, n)/lambda that count is

    transfers per degradation  =  f_shape(rho, n)

the dimensionless MFPT itself -- independent of every rate constant, so it cannot be tuned away.
Loop 149's n = 8, rho = 4.91 solution has f_shape = 1.08e5. That is not a detail. It says the cell
puts a hundred thousand ubiquitins onto a protein, and takes a hundred thousand off, before that
protein is finally destroyed once.

AND THE SAME EXPRESSION IS WHAT MAKES THE MECHANISM WORK. The amplification is r*f(rho,n)/f(rho/r,n)
and the cost is f(rho,n). They are the same function. Amplification is BOUGHT with futile cycling
and there is no configuration that has one without the other, so the frontier in T5 is not a
sweep of a free parameter -- it is the exchange rate.

REPORTED BEFORE ANY GATE, because it bears on the whole arc and is not this loop's to resolve:
loop_proteostasis records a gross load of 2.933e7 molecules/h and a dilution-free load of 3.938e6/h
at a 24 h doubling time. Only the second is true proteolysis and only true proteolysis costs
ubiquitin, so every cost below is computed on BOTH and gated on the SMALLER -- the direction that
makes a FAIL harder to reach. Separately, mu = ln2/24 h = 0.02888/h EXCEEDS the median oscillator's
resting b_lo of 0.02347/h, which means that protein's resting turnover is dilution-dominated and
its true resting proteolysis is at or below zero. That does not change anything computed here, and
it is flagged for a later loop rather than patched silently in this one.

PREDECLARED. Every conclusion is emitted through gate_guard.verdict, which loop 150b added after
the same defect appeared three times, so no sentence below can contradict the gate beneath it.

  T0 CAPABILITY AND REGRESSION.
       (a) UBA1, a ubiquitin-encoding gene and at least three E2s present in Schwanhausser with
           copy numbers, and the translational ATP bill present in the repo record;
       (b) the identity E[transfers] = lambda*T verified against a direct solve for the expected
           number of up-transitions from the absorbing chain, to 1e-8;
       (c) at rho -> 0 the transfer count must equal exactly n -- a ladder with no DUBs climbs once.
       Gate: all three. If (b) fails the cost is not the cost and nothing below is read.

  T1 HOW MANY UBIQUITINS PER PROTEIN DESTROYED?
       f_shape(rho, n) at loop 149's solution and across the grid. Gate: report, and confirm it is
       independent of k_u by recomputing at three different k_u that all satisfy b_lo.

  T2 DOES THE UBIQUITIN POOL TURN OVER FAST ENOUGH?
       total transfer flux divided by the ubiquitin census, as cycles per ubiquitin per second.
       Gate: < 100 /s. A multi-enzyme cascade running every molecule of its own substrate pool more
       than a hundred times a second is not a pool, it is a fiction.
       NOTE PREDECLARED: the Schwanhausser census carries RPS27A but not UBB or UBC, so the pool is
       an UNDERESTIMATE and this gate is therefore harsher than the truth. T3 and T4 do not depend
       on the pool size at all, which is why they are the gates that decide.

  T3 THE E1 BOTTLENECK.                                              THE ONE THAT CAN KILL IT.
       every re-activation passes through E1. Required transfer flux against UBA1 copies times
       k_cat, swept at 0.5, 1 and 5 /s. Gate: required <= capacity at the MOST GENEROUS 5 /s. A
       FAIL at the generous end is a FAIL at every end.

  T4 THE ATP PRICE.                                                  THE OTHER ONE.
       two phosphoanhydride equivalents per futile cycle, against the repo's own translational ATP
       bill of 1.2305e11 /h computed from the same proteome. Gate: < 1x translation. Translation is
       roughly a third to a half of a growing cell's entire budget, so one times translation is
       already a third of everything, and a proofreading mechanism that costs more than the cell
       spends making all of its protein is not a mechanism.

  T5 THE FRONTIER: IS THERE ANY AFFORDABLE LADDER?                   THE ANSWER.
       for n = 2..30 find the minimum rho delivering 20.29x, then its f_shape, flux, E1 load and
       ATP bill. Amplification and cost are the same function, so this is an exchange rate and not
       a free parameter. Gate: at least one n satisfies the cost gates AND the biochemical bound
       n <= 10 that loop 149's M2 gated on. If the affordable ladders are all longer than chains
       are observed to be, the median-receptor branch is squeezed out from both sides at once and
       CDC20 is what is left.

  T6 WHAT WOULD SETTLE IT DIRECTLY.
       state plainly that this is an INDIRECT test through cost, name the two direct measurements
       that would settle it, and record that neither is on disk.

-> outputs/loop_reversibility_price.json
"""
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM        # noqa: E402
import loop_replication as LR    # noqa: E402
import gate_guard as GG          # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
SCHWAN = SC / "_schwan2011.json"

SEED = 15100
E1_KCAT = (0.5, 1.0, 5.0)            # per second; the gate uses the most generous
ATP_PER_CYCLE = 2.0                  # ATP -> AMP + PPi, two phosphoanhydride equivalents
T2_MAX_CYCLES_PER_S = 100.0
T4_MAX_VS_TRANSLATION = 1.0
T5_NMAX = 30
T5_BIOCHEM_NMAX = 10                 # loop 149's M2 bound, from Thrower 2000
UB_GENES = ("RPS27A", "UBA52", "UBB", "UBC")
E2_GENES = ("UBE2D3", "UBE2L3", "UBE2N", "UBE2C", "UBE2S", "UBE2R2", "UBE2G1")

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def f_shape(rho, n):
    """T*lambda for the constant-rate ladder -- and, by T0(b), the transfers per degradation."""
    tot, ti = 0.0, 0.0
    for i in range(n):
        ti = 1.0 + (rho * ti if i > 0 else 0.0)
        tot += ti
    return tot


def amp(rho, n, r):
    return r * f_shape(rho, n) / f_shape(rho / r, n)


def upsteps_direct(lam, mu, n):
    """Expected number of up-transitions before absorption, solved directly."""
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(n):
        tot = lam + (mu if i > 0 else 0.0)
        A[i, i] = 1.0
        b[i] = lam / tot
        if i + 1 < n:
            A[i, i + 1] = -lam / tot
        if i > 0:
            A[i, i - 1] = -mu / tot
    return float(np.linalg.solve(A, b)[0])


def min_rho(n, r, req, grid):
    ok = [x for x in grid if amp(x, n, r) >= req]
    return float(min(ok)) if ok else None


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 151 -- can we test whether chains stay reversible out to 8? The price says yes.")
    say("=" * 100)
    say()

    PEQ = json.load(open(OUT / "loop_pulse_equation.json"))
    PRO = json.load(open(OUT / "loop_proteostasis.json"))
    CAP = json.load(open(OUT / "loop_capacity_ratio.json"))
    MK = json.load(open(OUT / "loop_ubiquitin_markov.json"))
    SIG = json.load(open(OUT / "loop_signalling_cost.json"))
    S = json.load(open(SCHWAN))

    REQ = float(PEQ["x3"]["fold_acceleration"])
    B_LO = float(PEQ["x3"]["b_rest"])
    R_MED = float(CAP["k3"]["receptor_median_fold"])
    N_SOL = int(MK["m4"]["solutions"]["median receptor"]["n"])
    RHO_SOL = float(MK["m4"]["solutions"]["median receptor"]["rho"])
    KU_SOL = float(MK["m4"]["solutions"]["median receptor"]["k_u_per_h"])
    LOAD_GROSS = float(PRO["p2"]["load_molecules_per_h"])
    LOAD_TRUE = float(PRO["p2"]["load_without_dilution_term"])
    ATP_TRANSLATION = float(SIG["y2"]["translation_atp_h"])

    say(f"  REPORTED BEFORE ANY GATE, and not this loop's to resolve:")
    say(f"    gross proteolytic load {LOAD_GROSS:,.0f} /h; dilution-free {LOAD_TRUE:,.0f} /h "
        f"({LOAD_TRUE / LOAD_GROSS:.1%}). Only the second costs ubiquitin.")
    say(f"    Every cost below is computed on BOTH and GATED ON THE SMALLER, which is the "
        f"direction that makes a FAIL harder to reach.")
    mu_dil = math.log(2) / 24.0
    say(f"    mu = ln2/24h = {mu_dil:.5f}/h EXCEEDS the median oscillator's resting b_lo of "
        f"{B_LO:.5f}/h. That protein's resting turnover is dilution-dominated and its true")
    say(f"    resting proteolysis is at or below zero. Flagged for a later loop, not patched here.")
    say()

    gates, res = {}, {}
    RHO_GRID = np.logspace(-3, 6, 9001)

    # ---------------------------------------------------------------- T0
    say("T0 CAPABILITY AND REGRESSION")
    ub = {g: S[g]["prot_copies"] for g in UB_GENES if g in S and S[g].get("prot_copies")}
    e2 = {g: S[g]["prot_copies"] for g in E2_GENES if g in S and S[g].get("prot_copies")}
    e1 = float(S["UBA1"]["prot_copies"]) if "UBA1" in S and S["UBA1"].get("prot_copies") else 0.0
    a_ok = bool(e1 > 0 and ub and len(e2) >= 3 and ATP_TRANSLATION > 0)
    say(f"     (a) UBA1 {e1:,.0f} copies;  ubiquitin genes present "
        f"{ {k: f'{v:,.0f}' for k, v in ub.items()} };  {len(e2)} E2s;  "
        f"translational ATP {ATP_TRANSLATION:.4g} /h   {'ok' if a_ok else 'FAIL'}")
    worst = 0.0
    for n in (2, 4, 8, 12):
        for rho in (0.1, 1.0, 4.91, 20.0):
            direct = upsteps_direct(1.0, rho, n)
            worst = max(worst, abs(f_shape(rho, n) - direct) / direct)
    b_ok = worst < 1e-8
    say(f"     (b) E[transfers] = lambda*T against a direct solve for expected up-transitions: "
        f"worst {worst:.2e}   gate < 1e-08   {'ok' if b_ok else 'FAIL'}")
    zero = max(abs(f_shape(1e-12, n) - n) for n in (1, 4, 8, 12, 20))
    c_ok = zero < 1e-9
    say(f"     (c) at rho -> 0 the count equals n exactly: worst deviation {zero:.2e}   "
        f"{'ok' if c_ok else 'FAIL'}")
    gates["T0"] = bool(a_ok and b_ok and c_ok)
    res["t0"] = {"uba1_copies": e1, "ubiquitin_copies": ub, "e2_copies": e2,
                 "atp_translation_per_h": ATP_TRANSLATION, "worst_upsteps": worst,
                 "rho0_deviation": zero, "pass": gates["T0"]}
    say(f"     T0 {'PASS' if gates['T0'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- T1
    say("T1 HOW MANY UBIQUITINS PER PROTEIN DESTROYED?")
    fs = f_shape(RHO_SOL, N_SOL)
    say(f"     loop 149's solution: n = {N_SOL}, rho = {RHO_SOL:.2f}")
    say(f"     transfers per degradation = f_shape = {fs:,.0f}")
    say(f"     cross-check against k_u/b_lo from the same solution: {KU_SOL / B_LO:,.0f}")
    indep = []
    for scale in (0.5, 1.0, 2.0):
        ku = KU_SOL * scale
        blo = ku / f_shape(RHO_SOL, N_SOL)
        indep.append({"k_u": ku, "b_lo": blo, "transfers": ku / blo})
    spread = max(x["transfers"] for x in indep) / min(x["transfers"] for x in indep)
    say(f"     independent of k_u by construction: three k_u differing 4-fold give transfer counts "
        f"differing by {spread:.6f}x")
    GG.verdict(abs(fs - KU_SOL / B_LO) / fs < 1e-6 and spread < 1 + 1e-9,
               f"the cost is the dimensionless MFPT itself and cannot be tuned away by any choice "
               f"of rate constant.",
               f"the transfer count moves with k_u, so it is not f_shape and the cost argument "
               f"below does not hold.")
    say(f"     IN WORDS: the cell puts {fs:,.0f} ubiquitins onto one protein, and takes "
        f"{fs - N_SOL:,.0f} back off, before that protein is finally destroyed once.")
    gates["T1"] = bool(abs(fs - KU_SOL / B_LO) / fs < 1e-6)
    res["t1"] = {"n": N_SOL, "rho": RHO_SOL, "transfers_per_degradation": fs,
                 "futile_removals": fs - N_SOL, "k_u_independence_spread": spread,
                 "checks": indep, "pass": gates["T1"]}
    say(f"     T1 {'PASS' if gates['T1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- T2
    say("T2 DOES THE UBIQUITIN POOL TURN OVER FAST ENOUGH?")
    ub_total = sum(ub.values())
    flux_true = LOAD_TRUE * fs
    flux_gross = LOAD_GROSS * fs
    cyc_true = flux_true / ub_total / 3600.0
    cyc_gross = flux_gross / ub_total / 3600.0
    say(f"     ubiquitin census {ub_total:,.0f} copies from {list(ub)} -- UBB and UBC are absent "
        f"from the table, so this is an UNDERESTIMATE and the gate is harsher than the truth")
    say(f"     transfer flux  {flux_true:.4g} /h on the dilution-free load  "
        f"({flux_gross:.4g} /h gross)")
    say(f"     cycles per ubiquitin per second: {cyc_true:,.1f} (gross {cyc_gross:,.1f})   "
        f"gate < {T2_MAX_CYCLES_PER_S:.0f}")
    GG.verdict(cyc_true < T2_MAX_CYCLES_PER_S,
               f"the pool can carry it at {cyc_true:,.1f} /s, though that is already a demanding "
               f"number for a three-enzyme cascade.",
               f"every ubiquitin molecule in the cell would have to be added and removed "
               f"{cyc_true:,.0f} times a second. That is not a pool, it is a fiction.")
    gates["T2"] = bool(cyc_true < T2_MAX_CYCLES_PER_S)
    res["t2"] = {"ubiquitin_copies_total": ub_total, "flux_true_per_h": flux_true,
                 "flux_gross_per_h": flux_gross, "cycles_per_ub_per_s_true": cyc_true,
                 "cycles_per_ub_per_s_gross": cyc_gross, "census_is_underestimate": True,
                 "pass": gates["T2"]}
    say(f"     T2 {'PASS' if gates['T2'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- T3
    say("T3 THE E1 BOTTLENECK")
    need_s = flux_true / 3600.0
    caps = {k: e1 * k for k in E1_KCAT}
    say(f"     every re-activation passes through E1. Required {need_s:.4g} activations/s; "
        f"UBA1 {e1:,.0f} copies")
    for k in E1_KCAT:
        say(f"       at k_cat {k:.1f}/s   capacity {caps[k]:.4g}/s   utilisation "
            f"{need_s / caps[k]:.1%}   per-E1 demand {need_s / e1:,.0f}/s")
    best = max(caps.values())
    GG.verdict(need_s <= best,
               f"E1 can supply it even at the low end of the k_cat band.",
               f"at the MOST GENEROUS {max(E1_KCAT):.0f}/s the demand is {need_s / best:,.0f}x "
               f"capacity. Each E1 would have to turn over {need_s / e1:,.0f} times a second "
               f"against a measured k_cat near 1. A FAIL at the generous end is a FAIL at every "
               f"end.")
    gates["T3"] = bool(need_s <= best)
    res["t3"] = {"required_per_s": need_s, "uba1_copies": e1,
                 "capacity_per_s": {str(k): v for k, v in caps.items()},
                 "utilisation": {str(k): need_s / v for k, v in caps.items()},
                 "per_e1_demand_per_s": need_s / e1, "overshoot_at_generous": need_s / best,
                 "pass": gates["T3"]}
    say(f"     T3 {'PASS' if gates['T3'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- T4
    say("T4 THE ATP PRICE")
    atp_true = flux_true * ATP_PER_CYCLE
    atp_gross = flux_gross * ATP_PER_CYCLE
    ratio = atp_true / ATP_TRANSLATION
    say(f"     {ATP_PER_CYCLE:.0f} phosphoanhydride equivalents per re-activation")
    say(f"     futile-cycling bill {atp_true:.4g} ATP/h (gross {atp_gross:.4g})")
    say(f"     the repo's translational bill, same proteome: {ATP_TRANSLATION:.4g} ATP/h")
    say(f"     ratio {ratio:,.1f}x translation   gate < {T4_MAX_VS_TRANSLATION:.0f}x")
    GG.verdict(ratio < T4_MAX_VS_TRANSLATION,
               f"the mechanism costs {ratio:.1%} of what the cell spends making all of its "
               f"protein, which is affordable.",
               f"the ladder would cost {ratio:,.0f} times what the cell spends making ALL of its "
               f"protein. Translation is roughly a third to a half of the whole budget, so this "
               f"is off the scale of the cell's energy economy by orders of magnitude.")
    gates["T4"] = bool(ratio < T4_MAX_VS_TRANSLATION)
    res["t4"] = {"atp_per_h_true": atp_true, "atp_per_h_gross": atp_gross,
                 "atp_translation_per_h": ATP_TRANSLATION, "ratio_vs_translation": ratio,
                 "pass": gates["T4"]}
    say(f"     T4 {'PASS' if gates['T4'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- T5
    say("T5 THE FRONTIER: IS THERE ANY AFFORDABLE LADDER?")
    say(f"     amplification is r*f(rho,n)/f(rho/r,n) and the cost is f(rho,n). Same function. "
        f"This is an exchange rate, not a free parameter.")
    front, affordable = [], []
    for n in range(2, T5_NMAX + 1):
        rho = min_rho(n, R_MED, REQ, RHO_GRID)
        if rho is None:
            front.append({"n": n, "rho": None, "reachable": False})
            continue
        f = f_shape(rho, n)
        fl = LOAD_TRUE * f
        e1u = (fl / 3600.0) / max(caps.values())
        at = fl * ATP_PER_CYCLE / ATP_TRANSLATION
        ok = bool(e1u <= 1.0 and at < T4_MAX_VS_TRANSLATION)
        front.append({"n": n, "rho": rho, "reachable": True, "transfers": f, "flux_per_h": fl,
                      "e1_utilisation": e1u, "atp_vs_translation": at, "affordable": ok})
        if ok:
            affordable.append(n)
    say(f"       n    min rho     transfers/degradation   E1 utilisation   ATP vs translation")
    for row in front:
        if not row["reachable"]:
            say(f"      {row['n']:>3}    unreachable at any rho -- r^n < {REQ:.1f}x")
            continue
        say(f"      {row['n']:>3}  {row['rho']:9.3g}   {row['transfers']:20,.0f}   "
            f"{row['e1_utilisation']:13.2%}   {row['atp_vs_translation']:16.3g}   "
            f"{'AFFORDABLE' if row['affordable'] else ''}")
    within = [n for n in affordable if n <= T5_BIOCHEM_NMAX]
    say(f"     affordable at any length: {affordable if affordable else 'none'}")
    say(f"     affordable AND within the observed chain-length bound n <= {T5_BIOCHEM_NMAX}: "
        f"{within if within else 'NONE'}")
    GG.verdict(bool(within),
               f"there is an affordable ladder inside the observed chain-length range: n = "
               f"{within}. The median-receptor branch survives on cost.",
               f"the two constraints have no overlap. Short ladders reach 20.29x only by cycling "
               f"so hard that E1 and ATP cannot pay for it; ladders cheap enough to run are longer "
               f"than K48 chains are observed to be. The median receptor is squeezed out from both "
               f"sides at once, and CDC20 -- which needs no amplification at all -- is what is "
               f"left standing.")
    gates["T5"] = bool(within)
    res["t5"] = {"frontier": front, "affordable_any_length": affordable,
                 "affordable_within_biochemical_bound": within,
                 "biochemical_nmax": T5_BIOCHEM_NMAX, "pass": gates["T5"]}
    say(f"     T5 {'PASS' if gates['T5'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- T6
    say("T6 WHAT WOULD SETTLE IT DIRECTLY")
    say(f"     everything above is an INDIRECT test. It measures what a reversible ladder would "
        f"COST and compares that to censuses, and it never observes a chain.")
    say(f"     the two measurements that would settle it, neither of which is on disk:")
    say(f"       1. Ub-clipping mass spectrometry (Swatek 2019) on a pulsed substrate through the "
        f"cycle -- it reads chain LENGTH and branching directly, which is the quantity R4 needs.")
    say(f"       2. a DUB-inhibition kinetic series. This model says the resting rate is slow "
        f"because the ladder rarely completes, so removing the back-reaction should collapse the")
    say(f"          half-life. At n = {N_SOL}, rho -> 0 gives b = k_u/n = "
        f"{KU_SOL / N_SOL:.1f}/h, a half-life of {60 * math.log(2) / (KU_SOL / N_SOL):.1f} "
        f"minutes against {math.log(2) / B_LO:.1f} h at rest. That is a "
        f"{(math.log(2) / B_LO) / (math.log(2) / (KU_SOL / N_SOL)):,.0f}-fold prediction and it is "
        f"the kind of thing an experiment either sees or does not.")
    say(f"     what this loop is entitled to say is about the PRICE, not the chain.")
    gates["T6"] = True
    res["t6"] = {"indirect": True, "direct_tests_absent_from_disk":
                 ["Ub-clipping MS for chain length and branching",
                  "DUB-inhibition kinetics on a pulsed substrate"],
                 "dub_inhibition_prediction_halflife_min":
                     60 * math.log(2) / (KU_SOL / N_SOL),
                 "resting_halflife_h": math.log(2) / B_LO}
    say()

    say("=" * 100)
    for k in ("T0", "T1", "T2", "T3", "T4", "T5", "T6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[SCHWAN, OUT / "loop_proteostasis.json",
                              OUT / "loop_pulse_equation.json", OUT / "loop_capacity_ratio.json",
                              OUT / "loop_ubiquitin_markov.json",
                              OUT / "loop_signalling_cost.json"],
                      available=T5_NMAX - 1, used=T5_NMAX - 1, selection="all", seed=SEED,
                      controls=["every cost computed on both loads and gated on the SMALLER, the "
                                "direction that makes a FAIL harder to reach",
                                "the E1 gate taken at the most generous end of the k_cat band",
                                "the transfer count shown independent of every rate constant (T1)",
                                "the identity E[transfers] = lambda*T verified against a direct "
                                "solve before it is used (T0b)",
                                "every conclusion emitted through gate_guard.verdict"],
                      note="reversibility is a futile cycle and a futile cycle has a price. "
                           "Amplification and cost are the SAME function f_shape(rho, n), so they "
                           "cannot be separated: buying 20.29x from a 1.51x receptor change means "
                           "buying the cycling that goes with it. The test is indirect -- it "
                           "measures the price, never the chain.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 151 -- the price of reversibility", "manifest": man, "gates": gates,
               **res, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_reversibility_price.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_reversibility_price.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
