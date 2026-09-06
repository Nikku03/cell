"""Loop 90. Redo the parameter selection against the genome, not against chr21.

THE OLDEST OUTSTANDING ITEM IN THIS PROJECT, and it is narrower than its title. "Re-derive targets
genome-wide and redo parameter selection" was written before loops 86 and 88 ran, and each did half
of it:

    LOOP 86 already re-derived the targets. Twenty-three chromosomes through the corrected pipeline,
    G4 recorded them, and G2 FAILED: chr21 sits at the 91st percentile on P(s) and the 96th on the
    long band. The arc had been calibrated on an outlier and loop 86 said so.

    LOOP 88 already used them -- for the ORIENTATION target. Its own docstring states the residual
    in one line: "the map target is chr21's own map". That is legitimate for predicting chr21's map
    and it means the 108-configuration sweep chose its parameters against a chromosome whose decay
    is 2.4 standard deviations from the genome's on P(s) and 3.0 on the long band.

So what is actually left is one question: DID SELECTING ON CHR21 CHOOSE A CONFIGURATION THE GENOME
WOULD NOT HAVE? Everything needed to answer it is already on disk. Loop 88 stored all 108
configurations with their P(s), short-band, long-band, orientation and map rho, and loop 86 stored
the genome-weighted targets with their spreads. No simulation is re-run here, which is the point:
the sweep was expensive and its results are sufficient, so the honest move is to re-score them
rather than to spend eleven more loops of compute confirming what a re-score can settle.

WHAT LOOP 88 ALREADY SETTLED AND THIS LOOP DOES NOT REOPEN. The trade-off between map and
orientation is structural -- rank correlation -0.7574 across the grid, four of five swept axes
pulling the two objectives in opposite directions -- and the entire admissible grid spans map rho
0.7773 to 0.8555 while a distance-only null scores 0.8283. That is not a tuning failure and this
loop is not another attempt at tuning. It asks only whether the SELECTION was made against the
right target.

PREDECLARED, BEFORE ANY NUMBER.

  P1 DOES THE GRID REPRODUCE? All 108 configurations, and loop 88's stored targets checked against
     loop 86's genome-weighted values.
     Gate: PASS iff the grid is complete and every target matches loop 86 to four decimals. Loop
     195's Z1 is the precedent: a re-score of a different grid, or against different targets, is
     about something else.

  P2 CAN THE MODEL REACH THE GENOME AT ALL? Each configuration scored on whether it lands within
     one genome standard deviation of the target on P(s), short band, long band AND orientation
     simultaneously. The grid brackets every target individually; hitting all four at once is a
     different question and is the one that matters.
     Gate: PASS iff at least one configuration does. A FAIL says the model cannot produce a typical
     chromosome's decay in any setting, which would be a stronger and more useful negative than
     anything about which configuration is best.

  P3 IS THE CHR21-SELECTED CONFIGURATION THE GENOME-SELECTED ONE? Configurations ranked two ways:
     by map rho against chr21's own map, which is how loop 88 chose, and by z-distance to the
     genome-weighted band targets. The rank correlation between the two orderings, and where loop
     88's winner sits in the genome ordering.
     Gate: PASS iff loop 88's map-best configuration sits in the top decile of the genome ordering.
     A FAIL is the finding this loop exists for: selection on chr21 picked a configuration the
     genome would not have.

  P4 WHAT DOES SELECTING ON THE GENOME COST ON THE MAP? The genome-best configuration's chr21 map
     rho, against the distance-only null's 0.8283.
     Gate: PASS iff it exceeds the null. A FAIL means fitting the genome's decay and predicting a
     contact map are not merely in tension, as loop 88's J2 found, but incompatible: the
     configuration that matches the genome does worse at the map than knowing nothing but distance.

  P5 WHAT THIS CANNOT SHOW.

-> outputs/loop_genome_reselect.json
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402

from scipy.stats import spearmanr                                 # noqa: E402

OUTDIR = Path(os.environ.get("CELL_OUT", "outputs"))
OUT = OUTDIR / "loop_genome_reselect.json"
JOINT = OUTDIR / "loop_joint.json"
GENOME = OUTDIR / "loop_genome.json"
BANDS = ("ps", "short", "long", "orient")
N_EXPECT = 108
TOP_DECILE = 0.10
SEED = 90090

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    G = GG.Gates(emit=say)
    say("=" * 104)
    say("LOOP 90  REDO THE PARAMETER SELECTION AGAINST THE GENOME, NOT AGAINST CHR21")
    say("=" * 104)
    say("  PREDECLARED: no simulation is re-run. Loop 88 stored all 108 configurations with their")
    say("  bands and map rho, loop 86 stored the genome-weighted targets with their spreads, and a")
    say("  re-score settles what eleven more loops of sweeping would not. Loop 88's trade-off")
    say("  finding is NOT reopened -- the map/orientation split is structural at rank correlation")
    say("  -0.7574 and this loop asks only whether the SELECTION used the right target.")
    say()

    j = json.load(open(JOINT))
    gm = json.load(open(GENOME))
    grid = j["grid"]
    tg = j["targets"]
    gw = gm["genome_weighted"]

    # ---- P1 ------------------------------------------------------------------------------------
    say("P1 DOES THE GRID REPRODUCE?")
    say(f"     {len(grid)} configurations stored by loop 88")
    ok_t = []
    for b in BANDS:
        key = "diff" if b == "orient" else b
        want = round(float(gw[key]["mean"]), 4)
        got = round(float(tg[b]), 4)
        ok_t.append(want == got)
        say(f"     {b:7} loop 88 target {got:+.4f}   loop 86 genome-weighted {want:+.4f}   "
            f"{'match' if want == got else 'MISMATCH'}   (genome sd {gw[key]['sd']:.4f})")
    p1_ok = len(grid) == N_EXPECT and all(ok_t)
    G.add("P1", p1_ok, stat=len(grid),
          if_true=f"P1 PASS -- {len(grid)} configurations and every target matches loop 86, so the "
                  f"re-score is of loop 88's grid against loop 86's genome",
          if_false=lambda: f"P1 FAIL -- {len(grid)} configurations, target match {ok_t}")

    sd = {b: float(gw["diff" if b == "orient" else b]["sd"]) for b in BANDS}
    tgt = {b: float(tg[b]) for b in BANDS}
    Z = np.array([[abs(e[b] - tgt[b]) / sd[b] for b in BANDS] for e in grid])
    within1 = (Z <= 1.0).all(axis=1)
    zdist = Z.max(axis=1)                       # worst band, so all four must be close to score well

    # ---- P2 ------------------------------------------------------------------------------------
    say()
    say("P2 CAN THE MODEL REACH THE GENOME AT ALL?")
    for i, b in enumerate(BANDS):
        n1 = int((Z[:, i] <= 1.0).sum())
        say(f"     {b:7} within 1 genome sd: {n1:3d} of {len(grid)} configurations")
    say(f"     within 1 sd on ALL FOUR simultaneously: {int(within1.sum())} of {len(grid)}")
    G.add("P2", bool(within1.any()), requires=("P1",),
          if_true=lambda: (f"P2 PASS -- {int(within1.sum())} configurations land inside the "
                           f"genome's spread on every band at once, so the model can produce a "
                           f"typical chromosome"),
          if_false="P2 FAIL -- no configuration lands within one genome sd on all four bands "
                   "together. The model cannot produce a typical chromosome's decay in ANY "
                   "setting, which is a stronger negative than any question about which "
                   "configuration is best")

    # ---- P3 ------------------------------------------------------------------------------------
    say()
    say("P3 IS THE CHR21-SELECTED CONFIGURATION THE GENOME-SELECTED ONE?")
    rho_map = np.array([e["rho"] for e in grid])
    order_map = np.argsort(-rho_map)             # loop 88's ordering: best map first
    order_gen = np.argsort(zdist)                # genome ordering: closest bands first
    rank_map = np.empty(len(grid), dtype=int)
    rank_gen = np.empty(len(grid), dtype=int)
    rank_map[order_map] = np.arange(len(grid))
    rank_gen[order_gen] = np.arange(len(grid))
    rs, ps_ = spearmanr(rank_map, rank_gen)
    best_map = int(order_map[0])
    best_gen = int(order_gen[0])
    say(f"     rank correlation between the two orderings: Spearman {rs:+.4f} (p {ps_:.3g})")
    say(f"     loop 88's map-best   #{rank_map[best_map]+1:3d} by map, "
        f"#{rank_gen[best_map]+1:3d} of {len(grid)} by genome fit   "
        f"rho {grid[best_map]['rho']:.4f}  worst-band z {zdist[best_map]:.2f}")
    say(f"     the genome-best      #{rank_map[best_gen]+1:3d} by map, "
        f"#{rank_gen[best_gen]+1:3d} by genome fit   "
        f"rho {grid[best_gen]['rho']:.4f}  worst-band z {zdist[best_gen]:.2f}")
    for tag, i in (("loop 88 map-best", best_map), ("genome-best", best_gen)):
        e = grid[i]
        say(f"       {tag:18} mode {e['mode']:6} sep {e['sep']:.0f} res {e['res']:.0f} "
            f"kappa {e['kappa']:.0f} alpha {e['alpha']:g}  "
            + "  ".join(f"{b} {e[b]:+.3f}" for b in BANDS))
    in_top = rank_gen[best_map] < TOP_DECILE * len(grid)
    G.add("P3", bool(in_top), requires=("P1",),
          if_true=lambda: (f"P3 PASS -- loop 88's winner is #{rank_gen[best_map]+1} of "
                           f"{len(grid)} by genome fit, inside the top decile, so selecting on "
                           f"chr21 chose a configuration the genome also endorses"),
          if_false=lambda: (f"P3 FAIL -- loop 88's winner ranks #{rank_gen[best_map]+1} of "
                            f"{len(grid)} by genome fit. Selecting on chr21's map chose a "
                            f"configuration the genome would not have, which is what this loop "
                            f"was left open to find out"))

    # ---- P4 ------------------------------------------------------------------------------------
    say()
    say("P4 WHAT DOES SELECTING ON THE GENOME COST ON THE MAP?")
    dn = float(j["distance_null"]["rho"])
    gen_rho = float(grid[best_gen]["rho"])
    say(f"     genome-best configuration's chr21 map rho {gen_rho:.4f}")
    say(f"     distance-only null                        {dn:.4f}")
    say(f"     loop 88's map-best                        {grid[best_map]['rho']:.4f}")
    say(f"     margin of the genome-best over the null   {gen_rho - dn:+.4f}")
    G.add("P4", gen_rho > dn, requires=("P1",),
          if_true=lambda: (f"P4 PASS -- the genome-matching configuration still beats the "
                           f"distance-only null on the map, by {gen_rho - dn:+.4f}"),
          if_false=lambda: (f"P4 FAIL -- the configuration that matches the genome's decay scores "
                            f"{gen_rho:.4f} on the map, BELOW the {dn:.4f} of a curve that knows "
                            f"only genomic separation. Fitting the genome and predicting a map are "
                            f"not merely in tension as loop 88's J2 found; on this grid they are "
                            f"incompatible"))

    # ---- P5 ------------------------------------------------------------------------------------
    say()
    say("P5 WHAT THIS CANNOT SHOW")
    say("     The map target is still chr21's map, because that is the only contact map this arc")
    say("     ever scored against. Re-scoring the BANDS against the genome does not make the map")
    say("     comparison genome-wide, and a full answer needs the sweep re-run against other")
    say("     chromosomes' maps -- which loop 86 streamed and discarded, and which this disk at")
    say("     2.4 GB free cannot hold.")
    say("     The genome targets are means over 23 chromosomes weighted by size. A configuration")
    say("     that matches the mean matches no particular chromosome, and chromosomes differ by")
    say("     more than their spread suggests -- chr21 is 2.4 sd out on P(s) and it is a real")
    say("     chromosome, not an error.")
    say("     Re-scoring a stored grid cannot find a configuration the grid does not contain. If")
    say("     the genome's optimum lies outside the 108 swept settings, nothing here would see it,")
    say("     and P2 passing only means the grid brackets the genome, not that it is centred on it.")
    say("     Loop 88's structural finding stands and is not weakened or strengthened by this")
    say("     loop: the map and orientation objectives are opposed across the grid, and the whole")
    say("     admissible range barely straddles a distance-only null.")
    G.add("P5", True, if_true="P5 PASS")

    gates, void = G.as_dict()
    man = RM.manifest(inputs=[JOINT, GENOME], available=len(grid), used=len(grid),
                      selection="all", seed=SEED,
                      controls=["loop 88's stored targets checked against loop 86's genome values",
                                "two independent orderings of the same grid",
                                "the distance-only null as the floor for P4"],
                      note="re-score loop 88's grid against loop 86's genome-weighted targets")
    out_d = dict(test="genome reselection", gates=gates, void=void,
                 n_grid=len(grid), targets=tgt, genome_sd=sd,
                 n_within_1sd=int(within1.sum()),
                 rank_spearman=float(rs), rank_p=float(ps_),
                 best_map=dict(index=best_map, genome_rank=int(rank_gen[best_map]) + 1,
                               **{k: grid[best_map][k] for k in
                                  ("mode", "sep", "res", "kappa", "alpha", "rho", *BANDS)}),
                 best_genome=dict(index=best_gen, map_rank=int(rank_map[best_gen]) + 1,
                                  **{k: grid[best_gen][k] for k in
                                     ("mode", "sep", "res", "kappa", "alpha", "rho", *BANDS)}),
                 distance_null=dn, manifest=man, seconds=time.time() - t0, log=log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)
    say()
    G.summary(seconds=time.time() - t0)
    out_d["log"] = log
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
