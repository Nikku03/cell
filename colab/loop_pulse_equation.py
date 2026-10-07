"""LOOP 142 -- UPGRADE THE EQUATION: A DESTRUCTION PULSE, NOT A MODULATED RATE.

Loop 122's U6 says it outright: "none of them can, and the equation itself is what has to change."
This is that change, and it is a change of WAVEFORM rather than a new term or a new parameter.

WHERE THE MODEL IS NOW. cell_assembled.integrate_deg solves

    dP/dt = k_sp*Mbar - b(t)*P        with   b(t) = bbar*(1 + beta*sin(wt))

and beta <= 1 is a HARD physical limit, because at beta > 1 the loss rate goes negative for part of
the cycle and protein is created by its own degradation term. Loop 123 computed the beta each
measured oscillation would require: THE MEDIAN IS 2.351. More than half the oscillating proteins
demand a sinusoid that does not exist. That is not a fitting problem, it is the waveform being
wrong.

WHAT THE ARITHMETIC ACTUALLY ALLOWS. For dP/dt = k(t) - b*P with k(t) >= 0 of ANY waveform, the
maximum relative amplitude (max-min)/(max+min) is exactly tanh(b*T/4) -- loop 123's closed layer.
The extremal drive is a square wave, and the bound falls straight out: with x = exp(-b*T/2),
P_max = (2*kbar/b)/(1+x), P_min = P_max*x, and (1-x)/(1+x) = tanh(b*T/4).

Run the same argument with the cycle in the LOSS term instead. Let b(t) sit at b_lo for most of the
period and jump to b_hi for a fraction d of it, with production constant. Neglecting production
DURING the pulse, the protein decays by exp(-b_hi*d*T) and

    relative amplitude  <=  tanh( b_hi * d * T / 2 )                    [an UPPER BOUND]

Same functional form as the old ceiling, and that is the point: bbar and T/4 become the PULSE rate
and d*T/2. Because b_hi has no upper limit -- a cell can raise a protein's degradation a
hundredfold for twenty minutes -- this has no ceiling below 1. The sinusoid was bounded not because
degradation is weak but because a sinusoid spends most of the cycle near its mean. A switch does
not.

X2 CAUGHT ME STATING THAT AS AN EQUALITY. Simulation sits 4-25% below it and the gap grows with
duty cycle, which is exactly what neglected production looks like. The exact periodic solution,
with x = exp(-b_lo*(1-d)T), y = exp(-b_hi*dT), A = k/b_lo and B = k/b_hi, is

    relative amplitude  =  (1-x)(1-y)(A-B) / [ A(1-x)(1+y) + B(1-y)(1+x) ]

which tends to the tanh bound as B/A -> 0. Sizing a required pulse from the BOUND would understate
it -- 17.0x against the exact 20.3x here -- so X3 inverts the exact expression instead. The bound
is still the right thing to quote for what is REACHABLE; it is the wrong thing for what is NEEDED.

THE MECHANISM I EXPECTED AND WHICH THE NUMBERS KILL, recorded because I checked it first. A shared
saturable protease would couple every protein to every other: loss_i = Vmax*(P_i/K_i)/(1 + sum_j
P_j/K_j), which in the linear limit recovers b_i = Vmax/K_i exactly and needs no new per-gene
parameter, since sum_j P_j/K_j is just the total degradation flux over Vmax -- all measured. It
would have explained loop 121's backwards result beautifully: the 362 would be PASSENGERS, crowded
out of a saturated protease by the few proteins that really are timed, and passengers need no
degrons. It is dead. loop_proteostasis measured the load at 2.933e7 molecules/h against 1.382e6
particles, which is 0.6% to 1.8% of capacity depending on the sweep time assumed. The cell runs its
proteasome at under two percent. There is no competition to be had, and X5 records that as a
measured negative rather than an untried idea.

AND THE SAME NUMBER IS WHAT MAKES THE PULSE WORK. 0.6-1.8% utilisation is 55x to 170x of headroom.
Whatever a destruction burst needs, the proteolytic capacity is there. X4 checks it rather than
assuming it.

PREDECLARED:

  X1 DOES THE OLD BOUND FALL OUT OF THE INTEGRATOR?                  THE REGRESSION TEST.
       drive production with a square wave at constant b and compare the simulated amplitude
       against tanh(b*T/4). Gate: agreement to 1%. If the integrator does not reproduce the bound
       the repo already closed on, nothing below can be trusted.

  X2 DOES THE PULSE FORM REPRODUCE ITS OWN DERIVATION?
       drive b with a pulse at constant production and compare against tanh(b_hi*d*T/2). Gate:
       agreement to 1% across a grid of duty cycles and pulse strengths. This is arithmetic
       checking arithmetic and it either holds or the derivation is wrong.

  X3 HOW BIG A PULSE DOES THE MEASUREMENT ACTUALLY DEMAND?
       for the 80 MS-oscillating genes with a measured amplitude and half-life, compute the
       required b_hi*d*T and, at a plausible duty cycle, the fold-acceleration over each protein's
       own resting b. Gate: report the distribution. A requirement of 10^4-fold would refute the
       upgrade as surely as a negative sinusoid did.

  X4 IS THE REQUIRED BURST DELIVERABLE?                              THE PHYSICAL TEST.
       total molecules destroyed per hour during the pulse, against the measured proteasome
       capacity from the closed proteostasis layer. Gate: the burst must fit inside capacity. This
       is unfitted on both sides and it is the test that can kill the upgrade outright.

  X5 THE COMPETITION MECHANISM, KILLED BY MEASUREMENT.
       sum_j P_j*b_j / Vmax from the recorded load and particle count. Gate: report it. If it is
       far below 1 the saturable-protease coupling cannot operate at any duty cycle and the idea
       is retired rather than left lying around as a maybe.

  X6 WHAT THE UPGRADE DOES NOT BUY.
       state plainly that a waveform change makes the amplitudes REACHABLE and says nothing about
       WHICH proteins are pulsed or WHEN. Six mechanisms have been eliminated on that question and
       this loop does not touch it.

-> outputs/loop_pulse_equation.json
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
import run_manifest as RM  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 14200
T_CYCLE = 24.0
DUTY = 0.10                    # a 2.4 h destruction window in a 24 h cycle
NSTEP = 4000
NCYC = 40

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def pulse_amp_exact(k, b_lo, b_hi, d, T):
    """EXACT relative amplitude for the two-phase pulse, from the periodic steady state.

    The docstring above derives tanh(b_hi*d*T/2) by neglecting production DURING the pulse. X2
    caught that: the simulation sits 4-25% BELOW it, and the gap grows with duty cycle, which is
    exactly what neglected production looks like. So that expression is an UPPER BOUND and this is
    the equality.

    With x = exp(-b_lo*(1-d)T), y = exp(-b_hi*dT), A = k/b_lo, B = k/b_hi, the periodic solution is
        P_min = [A*y*(1-x) + B*(1-y)] / (1 - x*y)
        P_max = [A*(1-x) + B*x*(1-y)] / (1 - x*y)
    and the relative amplitude reduces to
        (1-x)(1-y)(A-B) / [A(1-x)(1+y) + B(1-y)(1+x)]
    which tends to tanh(b_hi*d*T/2) as B/A -> 0, recovering the bound."""
    x = math.exp(-b_lo * (1.0 - d) * T)
    y = math.exp(-b_hi * d * T)
    A, B = k / b_lo, k / b_hi
    num = (1 - x) * (1 - y) * (A - B)
    den = A * (1 - x) * (1 + y) + B * (1 - y) * (1 + x)
    return num / den if den > 0 else 0.0


def required_b_hi(Aobs, b_lo, d, T, k=1.0):
    """Invert pulse_amp_exact for b_hi by bisection. Returns None if unreachable."""
    lo, hi = b_lo * 1.0000001, b_lo * 1e6
    if pulse_amp_exact(k, b_lo, hi, d, T) < Aobs:
        return None
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if pulse_amp_exact(k, b_lo, mid, d, T) < Aobs:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def rel_amp(x):
    """(max - min) / (max + min), the convention tanh(b*T/4) is stated in."""
    hi, lo = float(np.max(x)), float(np.min(x))
    return (hi - lo) / (hi + lo) if (hi + lo) > 0 else 0.0


def integrate(k_of_t, b_of_t, T, P0, ncyc=NCYC, nstep=NSTEP):
    """Exponential integrator for dP/dt = k(t) - b(t)P, exact for piecewise-constant coefficients.
    Records the FINAL cycle, so the transient is gone before anything is measured."""
    dt = T / nstep
    P = float(P0)
    tr = np.zeros(nstep)
    for s in range(ncyc * nstep):
        i = s % nstep
        t = i * dt
        if s >= (ncyc - 1) * nstep:
            tr[i] = P
        b = max(b_of_t(t), 1e-12)
        k = max(k_of_t(t), 0.0)
        eb = math.exp(-b * dt)
        P = P * eb + (k / b) * (1.0 - eb)
    return tr


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 142 -- upgrade the equation: a destruction pulse, not a modulated rate")
    say("=" * 100)
    say()
    gates, res = {}, {}

    ms = json.load(open(OUT / "loop_ms_cellcycle.json"))
    pro = json.load(open(OUT / "loop_proteostasis.json"))
    ph = ms["posthoc"]

    # ---------------------------------------------------------------- X1
    say("X1 DOES THE OLD BOUND FALL OUT OF THE INTEGRATOR?")
    say("     dP/dt = k(t) - b*P, k a SQUARE WAVE (the extremal non-negative drive), b constant")
    rows = []
    for hl in (6.0, 12.0, 24.0, 48.0):
        b = math.log(2) / hl
        kbar = 1.0
        tr = integrate(lambda t: 2 * kbar if (t % T_CYCLE) < T_CYCLE / 2 else 0.0,
                       lambda t: b, T_CYCLE, kbar / b)
        sim, thy = rel_amp(tr), math.tanh(b * T_CYCLE / 4.0)
        err = abs(sim - thy) / thy
        rows.append({"half_life_h": hl, "sim": sim, "tanh": thy, "rel_err": err})
        say(f"     t1/2 {hl:>5.1f} h   simulated {sim:.4f}   tanh(bT/4) {thy:.4f}   "
            f"error {err:.2%}")
    gates["X1"] = bool(max(r["rel_err"] for r in rows) < 0.01)
    res["x1"] = rows
    say(f"     X1 {'PASS' if gates['X1'] else 'FAIL'} -- the integrator reproduces the closed bound")
    say()

    # ---------------------------------------------------------------- X2
    say("X2 DOES THE PULSE FORM REPRODUCE ITS OWN DERIVATION?")
    say("     dP/dt = k - b(t)*P, k CONSTANT, b(t) = b_hi on a duty fraction d and b_lo otherwise")
    say("     derivation: relative amplitude = tanh(b_hi*d*T/2)")
    rows2 = []
    for d in (0.05, 0.10, 0.20):
        for bh in (0.2, 0.5, 1.0, 2.0):
            b_lo = math.log(2) / 48.0
            win = d * T_CYCLE
            tr = integrate(lambda t: 1.0,
                           lambda t, bh=bh, win=win: bh if (t % T_CYCLE) < win else b_lo,
                           T_CYCLE, 1.0 / b_lo)
            sim = rel_amp(tr)
            exact = pulse_amp_exact(1.0, b_lo, bh, d, T_CYCLE)
            bound = math.tanh(bh * d * T_CYCLE / 2.0)
            err = abs(sim - exact) / exact
            rows2.append({"duty": d, "b_hi": bh, "sim": sim, "exact": exact,
                          "tanh_bound": bound, "rel_err": err})
            say(f"     d {d:.2f}  b_hi {bh:>4.1f}/h   sim {sim:.4f}   EXACT {exact:.4f}   "
                f"err {err:.2%}   (tanh bound {bound:.4f})")
    worst = max(r["rel_err"] for r in rows2)
    gates["X2"] = bool(worst < 0.01)
    res["x2"] = rows2
    say(f"     worst relative error across the grid: {worst:.2%}")
    say(f"     X2 {'PASS' if gates['X2'] else 'FAIL'} -- the EXACT expression matches simulation")
    say(f"     the tanh(b_hi*d*T/2) form in the docstring is an UPPER BOUND, not an equality: it")
    say(f"     neglects production during the pulse and overstates amplitude by up to "
        f"{max(abs(r['tanh_bound'] - r['exact']) / r['exact'] for r in rows2):.0%} here.")
    say(f"     Using the bound to size a required pulse would UNDER-state it, so X3 inverts the")
    say(f"     exact expression instead.")
    say()

    # ---------------------------------------------------------------- X3
    say("X3 HOW BIG A PULSE DOES THE MEASUREMENT ACTUALLY DEMAND?")
    A = ph["median_rel_obs"]
    hl_osc = ms["v5"]["median_hl_osc"]
    b_rest = math.log(2) / hl_osc
    ceil_old = math.tanh(b_rest * T_CYCLE / 4.0)
    say(f"     {ph['n_tested']} MS-oscillating genes with a measured amplitude and half-life")
    say(f"     median relative amplitude {A:.4f}; median half-life {hl_osc:.2f} h "
        f"-> resting b {b_rest:.4f}/h")
    say(f"     the OLD production ceiling for that protein is tanh(bT/4) = {ceil_old:.4f}")
    say(f"     {ph['over_production_ceiling_fraction']:.1%} of them "
        f"({ph['n_over_ceiling']}/{ph['n_tested']}) exceed it, and the sinusoid the old equation")
    say(f"     would need has median beta {ph['median_beta_sinusoidal']:.3f} -- ABOVE the physical")
    say(f"     limit of 1, i.e. a negative loss rate. That is what has to change.")
    need = 2.0 * math.atanh(min(A, 0.999))
    b_approx = need / (DUTY * T_CYCLE)
    b_hi_req = required_b_hi(A, b_rest, DUTY, T_CYCLE)
    fold = b_hi_req / b_rest
    say(f"     the BOUND would give b_hi*d*T >= 2*artanh(A) = {need:.4f}, i.e. "
        f"{b_approx / b_rest:.1f}x -- an UNDER-estimate, because the bound overstates amplitude")
    say(f"     inverting the EXACT expression at a duty cycle of {DUTY:.0%} "
        f"({DUTY * T_CYCLE:.1f} h window):")
    say(f"       b_hi = {b_hi_req:.4f}/h, i.e. a half-life of {math.log(2) / b_hi_req:.2f} h "
        f"DURING the pulse")
    say(f"       that is {fold:.1f}x the protein's own resting rate")
    by_duty = {}
    for d in (0.05, 0.10, 0.20, 0.33):
        bb = required_b_hi(A, b_rest, d, T_CYCLE)
        by_duty[str(d)] = bb / b_rest if bb else None
        say(f"       duty {d:>5.0%} -> {bb / b_rest:>6.1f}x acceleration"
            if bb else f"       duty {d:>5.0%} -> UNREACHABLE")
    gates["X3"] = bool(fold < 1e3)
    res["x3"] = {"median_A": A, "median_hl_h": hl_osc, "b_rest": b_rest,
                 "old_ceiling": ceil_old, "required_bhi_dT": need,
                 "duty": DUTY, "b_hi_required": b_hi_req, "fold_acceleration": fold,
                 "fold_from_bound_underestimate": b_approx / b_rest, "by_duty": by_duty}
    say(f"     X3 {'PASS' if gates['X3'] else 'FAIL'} -- the demand is "
        f"{'a modest, physiological acceleration -- APC/C and SCF substrates do far more' if gates['X3'] else 'ABSURD and the upgrade is refuted'}")
    say()

    # ---------------------------------------------------------------- X4
    say("X4 IS THE REQUIRED BURST DELIVERABLE?")
    load = pro["p2"]["load_molecules_per_h"]
    part = pro["p3"]["particles"]
    caps = {s: part * 3600.0 / s for s in (1.0, 2.0, 3.0)}
    say(f"     steady proteolytic load {load:.3e} molecules/h (loop_proteostasis, 4,821 genes)")
    say(f"     proteasome particles {part:.3e}")
    # during the pulse the 80 oscillating genes degrade at b_hi instead of b_rest. Their share of
    # the steady load is what scales.
    n_osc, n_tot = ms["v5"]["n_osc"], ms["v5"]["n_osc"] + ms["v5"]["n_flat"]
    share = n_osc / n_tot
    burst_extra = load * share * (fold - 1.0)
    say(f"     the oscillating set is {n_osc}/{n_tot} = {share:.2%} of the measured genes")
    say(f"     accelerating just that set by {fold:.1f}x during the window adds "
        f"{burst_extra:.3e} molecules/h")
    ok = {}
    for s, cap in caps.items():
        tot = load + burst_extra
        ok[s] = tot / cap
        say(f"     at {s:.0f} s/substrate: capacity {cap:.3e}/h   peak load {tot:.3e}/h   "
            f"utilisation {tot / cap:.2%}")
    gates["X4"] = bool(max(ok.values()) < 1.0)
    res["x4"] = {"steady_load": load, "particles": part, "osc_share": share,
                 "burst_extra": burst_extra, "peak_utilisation": ok}
    say(f"     X4 {'PASS' if gates['X4'] else 'FAIL'} -- the burst "
        f"{'fits inside measured capacity with room to spare' if gates['X4'] else 'EXCEEDS capacity and the upgrade is physically refuted'}")
    say()

    # ---------------------------------------------------------------- X5
    say("X5 THE COMPETITION MECHANISM, KILLED BY MEASUREMENT")
    for s, cap in caps.items():
        say(f"     sum_j P_j*b_j / Vmax at {s:.0f} s/substrate = {load / cap:.4f}")
    sat = max(load / c for c in caps.values())
    say(f"     the saturable-protease coupling needs this near 1. It is at most {sat:.4f}.")
    say(f"     loss_i = Vmax*(P_i/K_i)/(1 + sum_j P_j/K_j) therefore sits in its LINEAR limit,")
    say(f"     where it reduces exactly to b_i*P_i and couples nothing to anything.")
    say(f"     I expected this mechanism to be the answer -- it would have explained loop 121's")
    say(f"     backwards result, with the 362 as passengers crowded out of a busy protease and")
    say(f"     therefore needing no degrons of their own. It is retired on measurement, not taste.")
    gates["X5"] = True
    res["x5"] = {"saturation": {str(s): load / c for s, c in caps.items()}, "max": sat,
                 "verdict": "RETIRED -- the proteasome runs at under 2% of capacity"}
    say(f"     X5 PASS -- recorded as a measured negative")
    say()

    # ---------------------------------------------------------------- X6
    say("X6 WHAT THE UPGRADE DOES NOT BUY")
    say(f"     it makes the observed amplitudes REACHABLE. tanh(b_hi*d*T/2) has no ceiling below 1,")
    say(f"     so 'no mechanism inside this equation can produce a 0.45 swing' is no longer true,")
    say(f"     and it stops being true at a {fold:.1f}x acceleration that the proteasome can serve")
    say(f"     {1 / max(ok.values()):.0f}x over.")
    say(f"     IT SAYS NOTHING ABOUT WHICH PROTEINS ARE PULSED OR WHEN. Six mechanisms have been")
    say(f"     eliminated on that question -- transcription, two TF networks, degron motifs,")
    say(f"     translation control, relocalisation and annotated ubiquitin targeting -- and a")
    say(f"     waveform change does not touch it. What this buys is that the equation is no longer")
    say(f"     the thing standing in the way.")
    gates["X6"] = True
    res["x6"] = {"reachable": True, "identifies_which_genes": False}
    say()

    say("=" * 100)
    for k in ("X1", "X2", "X3", "X4", "X5", "X6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[OUT / "loop_ms_cellcycle.json", OUT / "loop_proteostasis.json"],
                      available=ph["n_tested"], used=ph["n_tested"], selection="all", seed=SEED,
                      controls=["the integrator must first reproduce the bound the repo already "
                                "closed on (X1)",
                                "the new bound is checked against simulation across a grid (X2)",
                                "the burst is checked against MEASURED proteasome capacity (X4)",
                                "the competing mechanism I expected to win is killed on "
                                "measurement and recorded (X5)"],
                      note="a change of WAVEFORM, not a new term or a new fitted parameter. The "
                           "requirement is one inequality: b_hi*d*T >= 2*artanh(A).")
    RM.report(man, emit=say)
    json.dump({"test": "loop 142 -- the pulse equation", "manifest": man, "gates": gates, **res,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_pulse_equation.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_pulse_equation.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
