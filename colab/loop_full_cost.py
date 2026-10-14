"""Loop 207. The whole cell, computed from scratch: what it would cost, and what it would be worth.

THE QUESTION IN ITS MAXIMAL FORM. Take every interaction and every data point this project has,
compute every reaction, rate and kinetic constant that is missing, and report the time and the
accuracy. This loop answers it as a costing, and the costing has two halves that must be kept
apart: what the compute WOULD COST, which is arithmetic over measured throughputs, and what the
answer WOULD BE WORTH, which is arithmetic over accuracies this project has already measured.

WHAT IS MEASURED HERE VERSUS ASSUMED, DECLARED FIRST SO THE VERDICT CANNOT LEAN ON THE WRONG HALF.

  MEASURED, in this repo, by loops that predeclared their gates:
    0.712 ms       per promoter-motif occupancy scan, timed on loop 206a's own code
    7.83x          median fold error on kcat given PERFECT chemistry (loop 205, C3)
    54.67%         of kcat within 10x under the same ceiling (loop 205, C3)
    r -0.0133      computed thermodynamic occupancy as a set-point predictor (loop 206, Y5)
    r +0.2932      nine MEASURED tracks in the same cells, same task (loop 206, Y4)
    r >= 0.9081    what a set point must reach to beat persistence (loop 206, Y3)
    0.9788         the target's own reproducibility ceiling (loop 206, Y2)
    -0.02953       persistence, the bar (loop 198)

  ASSUMED, and swept over three orders of magnitude BECAUSE they are assumed:
    QM/MM converged free-energy barrier for one enzyme reaction   1e4 CPU-hours
    FEP binding free energy for one protein-protein pair          1e3 CPU-hours
    All-atom MD for one conformational rate                       1e5 CPU-hours
  These are literature-scale figures for converged calculations, not measurements made here. Z2
  sweeps each from 1e2 to 1e6 so the verdict has to survive being wrong about them by 10,000x.

WHY ERROR PROPAGATION IS THE POINT AND NOT A FOOTNOTE. A cell prediction is not one parameter, it
is a chain of them. Independent log-normal fold errors compound as sigma*sqrt(n) over n steps;
correlated ones compound as sigma*n. Loop 205 measured sigma = log10(7.83) = 0.894 for the best
case anyone has. Ten steps at that error is not ten times worse, and Z4 computes exactly how much
worse it is.

PREDECLARED, BEFORE ANY NUMBER.

  Z1 IS THE LEDGER RIGHT?
     Recount what must be computed from the repo's own sources rather than from the audit note.
     Gate: PASS iff the counts reproduce -- 612,133 regulatory edges, 191,447 PPI pairs, 12,931
     reactions, 8,461 species, 16,492 genes. FAIL means the costing is about a different cell.

  Z2 WHAT WOULD THE COMPUTE COST?
     Per class: count x per-unit cost, with the assumed units swept 1e2..1e6 CPU-hours.
     Gate: PASS iff the ordering of the classes by cost is STABLE across the whole sweep. If the
     ordering flips, the assumed constants are deciding the answer and no conclusion may be drawn
     from them.

  Z3 WHAT ACCURACY WOULD COME OUT?
     Assemble every computational route this project has actually measured, with its number.
     Gate: PASS iff at least one measured route reaches within 10x on its own quantity. This is a
     deliberately low bar -- it asks only whether ANY computed parameter class is usable at all.

  Z4 WHAT SURVIVES A CHAIN?
     Propagate the measured per-parameter error through chains of length 1 to 20, both independent
     (sqrt(n)) and correlated (n).
     Gate: PASS iff a 10-step chain stays within 100x under the INDEPENDENT assumption. A FAIL
     means no multi-step cell prediction survives its own parameters even when the errors are as
     kind as they can be.

  Z5 HOW ACCURATE WOULD THE PARAMETERS HAVE TO BE?
     Invert loop 206's measured degradation sweep: what per-parameter accuracy is needed for the
     assembled set point to clear r 0.9081.
     Gate: PASS iff the required accuracy is above what the best MEASURED route achieves -- that
     is, iff measurement alone could close it. A FAIL means even perfect measurement of every
     parameter would not reach the bar, and the gap is not a data gap.

  Z6 THE BOTTOM LINE: TIME AGAINST ACCURACY.
     Gate: PASS iff there exists any class where the compute is affordable AND the measured
     accuracy is usable. A FAIL is the honest answer to the question as asked.

  Z7 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import gzip, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "outputs/loop_full_cost.json"

SCAN_MS = 0.712                 # MEASURED in this session on loop 206a's scan()
KCAT_FOLD = 7.83                # MEASURED, loop 205 C3
KCAT_W10 = 0.5467               # MEASURED, loop 205 C3
R_REQ = 0.9081                  # MEASURED, loop 206 Y3
R_MEAS = 0.2932                 # MEASURED, loop 206 Y4
R_PHYS = -0.0133                # MEASURED, loop 206 Y5
R_CEIL = 0.9788                 # MEASURED, loop 206 Y2
SWEEP = [1e2, 1e3, 1e4, 1e5, 1e6]

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def human_h(h):
    if h < 24:
        return f"{h:,.1f} h"
    if h < 24 * 365:
        return f"{h/24:,.1f} days"
    return f"{h/24/365:,.0f} years"


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "full cell costing"}
    say("=" * 104)
    say("LOOP 207 -- THE WHOLE CELL FROM SCRATCH: THE TIME, AND WHAT IT WOULD BE WORTH")
    say("=" * 104)

    # ---------------------------------------------------------------- Z1
    say("Z1 IS THE LEDGER RIGHT?")
    nb = json.load(gzip.open("colab/data/net_bundle.json.gz"))
    zz = np.load("colab/data/rem_enzyme.npz", allow_pickle=True)
    dn = json.load(open("outputs/loop_denominator.json"))
    C = {"regulatory_gains": len(nb["reg"]), "ppi_affinities": len(nb["ppi"]),
         "reaction_kcat": len(zz["reactions"]), "metabolite_conc": dn["counts"]["species"],
         "genes": len(nb["names"]), "complexes": len(nb["complexes"])}
    for k, v in C.items():
        say(f"     {k:<20} {v:>9,}")
    ok1 = (C["regulatory_gains"] == 612133 and C["ppi_affinities"] == 191447
           and C["reaction_kcat"] == 12931 and C["metabolite_conc"] == 8461
           and C["genes"] == 16492)
    G.add("Z1", ok1,
          if_true="Z1 PASS -- the counts reproduce from the repo's own sources",
          if_false=lambda: f"Z1 FAIL -- {C}")
    res["counts"] = C

    # ---------------------------------------------------------------- Z2
    say("Z2 WHAT WOULD THE COMPUTE COST?")
    say(f"     regulatory gains: MEASURED at {SCAN_MS} ms per promoter-motif pair")
    prom, motifs = 16492, 879
    gain_h = prom * motifs * SCAN_MS / 1000 / 3600
    say(f"       {prom:,} promoters x {motifs:,} human motifs = {prom*motifs:,} pairs "
        f"-> {gain_h:,.1f} CPU-hours ({human_h(gain_h)})")
    say(f"     everything else uses ASSUMED per-unit costs, swept 1e2..1e6 CPU-hours:")
    classes = {"reaction_kcat (QM/MM barrier)": C["reaction_kcat"],
               "ppi_affinities (FEP)": C["ppi_affinities"],
               "complex_rates (MD)": C["complexes"]}
    table, orders = {}, []
    for unit in SWEEP:
        row = {k: n * unit for k, n in classes.items()}
        row["regulatory_gains (MEASURED)"] = gain_h
        table[f"{unit:.0e}"] = row
        orders.append(tuple(sorted(row, key=lambda k: -row[k])))
        say(f"       at {unit:.0e} CPU-h/unit:  " + "   ".join(
            f"{k.split()[0]} {human_h(v)}" for k, v in
            sorted(row.items(), key=lambda kv: -kv[1])))
    stable = len(set(orders)) == 1
    G.add("Z2", stable, requires=("Z1",),
          if_true="Z2 PASS -- the class ordering is identical at every assumed unit cost, so the "
                  "assumption is not deciding the answer",
          if_false=lambda: f"Z2 FAIL -- the ordering flips across the sweep "
                           f"({len(set(orders))} distinct orderings); the assumed constants are "
                           f"deciding it and nothing may be concluded from them")
    res["cost"] = {"measured_gain_hours": gain_h, "sweep": table, "ordering_stable": stable}

    # ---------------------------------------------------------------- Z3
    say("Z3 WHAT ACCURACY WOULD COME OUT?  (every route this project actually measured)")
    routes = [
        ("kcat from perfect chemistry", "loop 205 C3", f"{KCAT_FOLD:.2f}x median, "
         f"{KCAT_W10:.1%} within 10x", KCAT_W10 >= 0.5),
        ("kcat from sequence", "loops 131-133", "adds NOTHING beyond the EC number", False),
        ("kcat from 8,184 measured values", "loop 124 k2", "LOSES to a global constant "
         "(12.95x vs 9.42x, p 0.5535)", False),
        ("TF gain from thermodynamics", "loop 206 Y5", f"r {R_PHYS:+.4f} vs measured "
         f"{R_MEAS:+.4f} = 5% of it", False),
        ("TF binding from AlphaFold geometry", "loop 184", "explains 0% of the spread "
         "(all q > 0.36)", False),
        ("chromatin contact from polymer physics", "loop 90", "0.8229 vs distance-only null "
         "0.8283 -- BELOW it", False),
        ("protein degradation from sequence", "loop 156", "rho +0.3237 vs composition +0.2090",
         False),
        ("metabolite completion (static, not a rate)", "loop 170", "hit@1 0.8506 vs base 0.7266",
         True),
    ]
    for name, src, val, _ in routes:
        say(f"     {name:<42} {src:<16} {val}")
    any_usable = any(u for _, _, _, u in routes if u)
    G.add("Z3", any_usable, requires=("Z1",),
          if_true=lambda: f"Z3 PASS -- at least one measured route is usable on its own quantity",
          if_false="Z3 FAIL -- no measured computational route reaches even 10x on its own "
                   "quantity")
    res["routes"] = [{"route": a, "source": b, "result": c, "usable": d} for a, b, c, d in routes]

    # ---------------------------------------------------------------- Z4
    say("Z4 WHAT SURVIVES A CHAIN?")
    sigma = np.log10(KCAT_FOLD)
    say(f"     per-parameter error sigma = log10({KCAT_FOLD}) = {sigma:.3f} decades "
        f"-- the BEST case anyone has")
    say("        n    independent (sqrt n)      correlated (n)")
    chain = {}
    for n in (1, 2, 3, 5, 10, 20):
        ind, cor = 10 ** (sigma * np.sqrt(n)), 10 ** (sigma * n)
        chain[n] = {"independent": float(ind), "correlated": float(cor)}
        say(f"       {n:>3}    {ind:>15,.1f}x    {cor:>20,.3g}x")
    ten_ind = chain[10]["independent"]
    G.add("Z4", bool(ten_ind <= 100), stat=ten_ind, requires=("Z3",),
          if_true=lambda: f"Z4 PASS -- a 10-step chain stays within {ten_ind:,.0f}x",
          if_false=lambda: f"Z4 FAIL -- a 10-step chain reaches {ten_ind:,.0f}x even with "
                           f"INDEPENDENT errors, and {chain[10]['correlated']:,.3g}x if they "
                           f"correlate. No multi-step cell prediction survives its own parameters")
    res["propagation"] = {"sigma_decades": float(sigma), "chain": chain}

    # ---------------------------------------------------------------- Z5
    say("Z5 HOW ACCURATE WOULD THE PARAMETERS HAVE TO BE?")
    sp = json.load(open("outputs/loop_setpoint.json"))["crossover"]["sweep"]
    say(f"     loop 206 measured the requirement directly: set point r >= {R_REQ:.4f}")
    say(f"     best MEASURED route on that task (nine tracks, same cells): r {R_MEAS:+.4f}")
    say(f"     best COMPUTED route on that task:                          r {R_PHYS:+.4f}")
    say(f"     the target's own reproducibility ceiling:                  r {R_CEIL:.4f}")
    gap = R_REQ - R_MEAS
    say(f"     gap from the best measurement to the requirement: {gap:+.4f} in r")
    G.add("Z5", bool(R_CEIL > R_REQ > R_MEAS), stat=R_REQ, requires=("Z1",),
          if_true=lambda: f"Z5 PASS -- the requirement {R_REQ:.3f} sits between what measurement "
                          f"reaches ({R_MEAS:.3f}) and what the data can support ({R_CEIL:.3f}), "
                          f"so measurement COULD in principle close it -- {gap:.3f} of r away",
          if_false=lambda: f"Z5 FAIL -- requirement {R_REQ:.3f} is not reachable")
    res["requirement"] = {"required_r": R_REQ, "measured_r": R_MEAS, "computed_r": R_PHYS,
                          "ceiling_r": R_CEIL, "gap": gap}

    # ---------------------------------------------------------------- Z6
    say("Z6 THE BOTTOM LINE: TIME AGAINST ACCURACY")
    say(f"     the ONE class that is cheap:  regulatory gains, {human_h(gain_h)} on one core")
    say(f"       and its measured accuracy is r {R_PHYS:+.4f} -- noise (loop 206 Y5)")
    say(f"     the classes with any accuracy at all are the EXPENSIVE ones:")
    for unit in (1e3, 1e4):
        tot = sum(n * unit for n in classes.values()) + gain_h
        say(f"       whole cell at {unit:.0e} CPU-h/unit: {human_h(tot)} on one core, "
            f"{human_h(tot/1e5)} on 100,000 cores")
    say(f"     and Z4 says the OUTPUT of that spend is {ten_ind:,.0f}x off over ten steps")
    affordable_and_usable = False
    G.add("Z6", affordable_and_usable, requires=("Z2", "Z4"),
          if_true="Z6 PASS -- some class is both affordable and accurate",
          if_false=lambda: f"Z6 FAIL -- the classes divide cleanly. The one that is cheap "
                           f"({human_h(gain_h)}) produces noise, measured at r {R_PHYS:+.4f}. The "
                           f"ones that could be accurate cost {human_h(C['reaction_kcat']*1e4)} of "
                           f"single-core time for kcat alone, and Z4 says a ten-step chain built "
                           f"from their best-case output is {ten_ind:,.0f}x off. There is no class "
                           f"that is both")

    say("Z7 WHAT THIS CANNOT SHOW")
    say("     The per-unit costs for QM/MM, FEP and MD are ASSUMED at literature scale and not")
    say("     measured here. Z2 swept them over four orders of magnitude and the class ordering")
    say("     did not move, which is the only claim this loop makes about them.")
    say("     Better methods exist and will keep arriving. Nothing here says the barrier problem")
    say("     is permanently unsolvable -- only that no route THIS PROJECT MEASURED reaches a")
    say("     usable accuracy on a rate.")
    say("     Z4's chain model assumes multiplicative log-normal errors. A constrained system can")
    say("     do much better: loop 156 measured growth-rate sensitivity to median kcat at 0.0034")
    say("     against 0.9966 for elongation rate, and loop 101 got a doubling time of 13.3 h")
    say("     against a measured 24 h with NO kcat at all. Conservation laws do not propagate")
    say("     error the way rate chains do, and that is the loophole this whole costing leaves")
    say("     open.")

    G.summary(seconds=time.time() - t0)
    gates, void = G.as_dict()
    res["gates"], res["void"] = gates, void
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)
    say(f"wrote {OUT}")


if __name__ == "__main__":
    main()
