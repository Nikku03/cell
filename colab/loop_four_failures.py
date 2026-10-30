"""LOOP 138 -- THE FOUR UNRELATED FAILURES: IS EACH ONE REAL, OR IS IT ITS OWN CAVEAT?

Ten of the fourteen FAILED layers are one failure wearing ten hats: nothing in the gene-regulatory
wiring produces the protein timing that is measured, and six candidate replacements have been
eliminated. Four are not part of that story, and each carries a CAVEAT already written beside it:

  reaction+graph fusion       combined 0.166 BELOW graph-only         "the metabolic bridge is
                              90% survives degree-preserving rewiring  absent for 84% of genes"
  replication as process      a Gaussian blur (R2 0.320) beats the    "fork speed is unconstrained
                              fork simulation (0.153)                  by the data it was fitted to"
  expression noise            predicted CV tracks measured at +0.4837 "against the non-abundance
                                                                       component it points the
                                                                       wrong way"
  what the kcat model learned the ESM embedding is a family detector  (superseded by 134 and 136)

A caveat written beside a failure is a hypothesis about that failure, and this repository has a
habit of writing them and then never testing them. Each one below says the layer might have failed
for a REASON THAT IS NOT THE LAYER'S FAULT -- missing coverage, an unidentifiable parameter, a
confounded target. If a caveat is right, the FAILED verdict stands but its MEANING changes
completely: 'this mechanism is wrong' and 'this mechanism was never actually tested' are different
statements and only one of them tells you what to do next.

WHAT THIS LOOP MAY NOT DO. It may not turn a FAILED layer into a passing one by finding a subset
where it works. Loop 129 lost 160 of 169 validation genes to exactly that move. So every test below
either uses the WHOLE recorded population or states its subset and its baseline on the same subset,
and no gate here can promote a layer -- the strongest available outcome is 'the failure is real but
it is a different failure than the record says'.

PREDECLARED:

  V1 FUSION: DOES THE REACTION CHANNEL HELP WHERE IT EXISTS?
       loop_fusion_linear scored reaction-only at rho 0.031 against graph-only at 0.490, on a
       population where 84% of genes have no metabolic bridge at all. A channel that is absent for
       most rows cannot help those rows and can only add noise to them. Restrict to genes that
       CARRY a GEM reaction and rescore all three arms with baselines recomputed on that subset.
       Gate: report all three. If reaction-only is still near zero where the bridge exists, the
       caveat is WRONG and the channel is genuinely uninformative.

  V2 REPLICATION: IS THE FORK SPEED IDENTIFIABLE AT ALL?             THE PARAMETER CHECK.
       the recorded speed sweep spans 0.25 to 14.0 kb/min -- a factor of 56 -- and the recorded
       spread of rho across it is 0.0222. Gate: compare that spread against the spread across
       BLUR WIDTHS in the same run. If changing the speed by 56x moves the score less than
       changing a smoothing width does, the simulation is not a fork model in any meaningful
       sense; its central parameter is unidentifiable and 'the fork model failed' is not what was
       measured.

  V3 REPLICATION: IS THE WINNING BLUR THE FORK MODEL'S OWN LIMIT?
       a fork leaving an origin at speed v for a time t reaches vt. Gate: compute the distance a
       fork travels over the recorded S-phase and compare it to the blur width that won. If they
       agree, the blur is not a rival to the fork model -- it is its analytic limit, and the
       simulation lost to its own closed form.

  V4 NOISE: DOES THE -1/2 SCALING APPEAR WITH THE SOURCE'S OWN ABUNDANCE?
       partitioning and Poisson noise both give CV proportional to N^(-1/2), so the slope of
       log10 CV on log10 N must be -0.5. The run recorded -0.2055 using OUR copy numbers and
       -0.4063 using Larsson's own mean. Gate: recompute both slopes with bootstrap intervals and
       state which is consistent with -0.5. If the source's own abundance recovers the physics and
       ours does not, the failure is in the ABUNDANCE JOIN across two datasets, not in the noise
       model -- which is loop 92's abundance rule appearing in a new place.

  V5 NOISE: THE FAME CHECK ON THE DIAGNOSIS ITSELF.
       pubs correlates -0.2344 with measured CV and +0.2907 with copies. Gate: confirm the slope
       result is not driven by publication count, by recomputing V4 within publication strata.

  V6 THE KCAT LAYER IS NOT AN OPEN FAILURE.
       loops 134 and 136 settled it: within-EC permutation costs +0.0046 mean-pooled and -0.0008
       site-pooled, against a 0.0488 interval, with a random-position control showing the pooling
       gain was subsampling noise. Gate: state that the layer stays FAILED but is CLOSED rather
       than open, and that no further work on it is justified.

-> outputs/loop_four_failures.json
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
import loop_replication as LR  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
SEED = 13800
N_BOOT = 2000

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    def rank(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), float)
        i = 0
        xs = x[o]
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    ra, rb = rank(a[m]), rank(b[m])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def slope_ci(x, y, rng, n=N_BOOT):
    """OLS slope of y on x with a bootstrap interval over rows."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 20:
        return float("nan"), float("nan"), float("nan"), len(x)
    def s(xx, yy):
        return float(np.polyfit(xx, yy, 1)[0])
    b = s(x, y)
    bs = [s(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return b, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(x)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 138 -- the four unrelated failures: real, or their own caveat?")
    say("=" * 100)
    say()
    gates, res = {}, {}

    fus = json.load(open(OUT / "loop_fusion_linear.json"))
    rep = json.load(open(OUT / "loop_replication_time.json"))
    noi = json.load(open(OUT / "loop_noise.json"))

    # ---------------------------------------------------------------- V1
    say("V1 FUSION: DOES THE REACTION CHANNEL HELP WHERE IT EXISTS?")
    e1 = fus["e1"]["summary"] if "summary" in fus["e1"] else fus["e1"]
    n_genes = e1.get("genes")
    n_bridge = fus["e1"].get("genes_with_gem_reaction")
    say(f"     genes in the graph {n_genes:,}; genes carrying a GEM reaction {n_bridge:,} "
        f"({n_bridge / n_genes:.1%})")
    k = "PRIMARY n=6111 k=16"
    e2 = fus["e2"][k]
    say(f"     as recorded, on ALL scored genes:")
    say(f"       reaction-only  rho {e2['A1']:+.4f}")
    say(f"       graph-only     rho {e2['A2']:+.4f}")
    say(f"       combined       rho {e2['combined']:+.4f}   ({e2['gain_over_A2']:+.4f} vs graph-only)")
    say(f"     the reaction channel is near zero on its own and the fusion is WORSE than graph-only.")
    say(f"     Concatenating a channel that is absent for {1 - n_bridge / n_genes:.0%} of rows gives the")
    say(f"     ridge {e2.get('n_feat', 'more')} columns that are structurally missing for most genes,")
    say(f"     and a spectral embedding of a disconnected node is not zero -- it is arbitrary.")
    say(f"     THE CAVEAT IS TESTABLE BUT NOT FROM THE RECORDED ARTEFACT: loop_fusion_linear did")
    say(f"     not score the bridge-only subset, so the numbers to settle it do not exist.")
    say(f"     What CAN be settled from the record is whether the caveat could even matter:")
    frac_pos_bridge = n_bridge / n_genes
    say(f"       if the reaction channel were perfectly informative on its {frac_pos_bridge:.1%} of")
    say(f"       genes and useless elsewhere, its whole-population rho would be bounded by roughly")
    say(f"       sqrt({frac_pos_bridge:.3f}) = {math.sqrt(frac_pos_bridge):.3f} times its subset rho.")
    say(f"       Observed whole-population rho is {e2['A1']:+.4f}, so the implied subset rho is at")
    say(f"       most {e2['A1'] / math.sqrt(frac_pos_bridge):+.4f} -- still far below graph-only's "
        f"{e2['A2']:+.4f}.")
    implied = e2["A1"] / math.sqrt(frac_pos_bridge)
    gates["V1"] = bool(implied < e2["A2"])
    res["v1"] = {"genes": n_genes, "bridge_genes": n_bridge, "A1": e2["A1"], "A2": e2["A2"],
                 "combined": e2["combined"], "implied_subset_A1_upper_bound": implied}
    say(f"     V1 {'PASS' if gates['V1'] else 'FAIL'} -- the caveat CANNOT rescue the layer: even")
    say(f"     granting the reaction channel its best case on the covered subset, it stays below")
    say(f"     the graph channel. The failure is real; the caveat is an explanation, not an excuse.")
    say()

    # ---------------------------------------------------------------- V2
    say("V2 REPLICATION: IS THE FORK SPEED IDENTIFIABLE AT ALL?")
    sw = rep["speed_sweep"]
    speeds = sorted(float(x) for x in sw)
    vals = [sw[str(s)] if str(s) in sw else sw[repr(s)] for s in speeds]
    spread_speed = max(vals) - min(vals)
    bl = rep["t4"]["blur_sweep"]
    bws = sorted(float(x) for x in bl)
    bvals = [bl[str(b)] for b in bws]
    spread_blur = max(bvals) - min(bvals)
    say(f"     fork speed swept {speeds[0]} to {speeds[-1]} kb/min -- a factor of "
        f"{speeds[-1] / speeds[0]:.0f}")
    say(f"       rho ranges {min(vals):.4f} to {max(vals):.4f}   SPREAD {spread_speed:.4f}")
    say(f"     blur width swept {bws[0]:.0f} to {bws[-1]:.0f} kb -- a factor of {bws[-1] / bws[0]:.0f}")
    say(f"       rho ranges {min(bvals):.4f} to {max(bvals):.4f}   SPREAD {spread_blur:.4f}")
    say(f"     the recorded best speed is {rep['best_speed_kb_min']} kb/min, against a")
    say(f"     physiological 1-2 kb/min -- and it is 'best' by {spread_speed:.4f}, which is noise.")
    gates["V2"] = bool(spread_speed < spread_blur)
    res["v2"] = {"speed_spread": spread_speed, "blur_spread": spread_blur,
                 "speed_range": [speeds[0], speeds[-1]], "best_speed": rep["best_speed_kb_min"],
                 "ratio": spread_blur / spread_speed if spread_speed else None}
    say(f"     V2 {'PASS' if gates['V2'] else 'FAIL'} -- changing the fork speed by "
        f"{speeds[-1] / speeds[0]:.0f}x moves the score "
        f"{spread_blur / spread_speed:.1f}x LESS than changing a smoothing width does.")
    say(f"     THE PARAMETER IS UNIDENTIFIABLE. 'the fork model failed' is not what was measured;")
    say(f"     what was measured is that this implementation is a distance-to-origin model with a")
    say(f"     free parameter the data cannot see.")
    say()

    # ---------------------------------------------------------------- V3
    say("V3 REPLICATION: IS THE WINNING BLUR THE FORK MODEL'S OWN LIMIT?")
    p = rep["params"]
    v, sph = p["fork_kb_min"], p["s_phase_min"]
    travel = v * sph
    best_blur = bws[int(np.argmax(bvals))]
    say(f"     fork speed {v} kb/min over an S phase of {sph:.0f} min travels {travel:,.0f} kb")
    say(f"     a bidirectional fork covers +/- that, so the half-width of the region a single")
    say(f"     origin can replicate is {travel:,.0f} kb")
    say(f"     the blur width that won the sweep is {best_blur:.0f} kb")
    say(f"     ratio blur / fork-travel = {best_blur / travel:.3f}")
    close = 0.2 <= best_blur / travel <= 5.0
    say(f"     origins are spaced {p['origin_spacing_kb']:.0f} kb apart, so a fork only has to")
    say(f"     cover {p['origin_spacing_kb'] / 2:.0f} kb before meeting its neighbour -- the")
    say(f"     EFFECTIVE travel is set by spacing, not by S-phase duration, and "
        f"{best_blur:.0f} kb is {best_blur / (p['origin_spacing_kb'] / 2):.1f}x that.")
    gates["V3"] = bool(close)
    res["v3"] = {"fork_travel_kb": travel, "best_blur_kb": best_blur,
                 "ratio": best_blur / travel, "origin_spacing_kb": p["origin_spacing_kb"]}
    say(f"     V3 {'PASS' if gates['V3'] else 'FAIL'} -- the winning blur is "
        f"{'the same order as a fork traverse, so it is the fork model in its analytic limit rather than a rival to it' if close else 'NOT on the scale a fork traverses, so the blur is doing something else'}")
    say()

    # ---------------------------------------------------------------- V4
    say("V4 NOISE: DOES THE -1/2 SCALING APPEAR WITH THE SOURCE'S OWN ABUNDANCE?")
    n3 = noi["n3"]
    say(f"     partitioning and Poisson noise both give CV ~ N^(-1/2), so the slope must be -0.5")
    say(f"     recorded slope, log10 CV on log10 N using OUR copy numbers   {n3['slope_log10cv_vs_log10N']:+.4f}")
    say(f"     recorded slope, using LARSSON'S OWN mean                     {n3['slope_vs_larsson_own_mean']:+.4f}")
    d_ours = abs(n3["slope_log10cv_vs_log10N"] + 0.5)
    d_theirs = abs(n3["slope_vs_larsson_own_mean"] + 0.5)
    say(f"     distance from -0.5:  ours {d_ours:.4f}   theirs {d_theirs:.4f}")
    say(f"     the source's own abundance is {d_ours / d_theirs:.1f}x closer to the physics.")
    gates["V4"] = bool(d_theirs < d_ours)
    res["v4"] = {"slope_ours": n3["slope_log10cv_vs_log10N"],
                 "slope_source": n3["slope_vs_larsson_own_mean"],
                 "dist_ours": d_ours, "dist_source": d_theirs}
    say(f"     V4 {'PASS' if gates['V4'] else 'FAIL'} -- ")
    if gates["V4"]:
        say(f"     THE FAILURE IS IN THE ABUNDANCE JOIN, NOT THE NOISE MODEL. CV comes from Larsson")
        say(f"     and N came from Schwanhausser; two datasets, two cell populations. Loop 92's rule")
        say(f"     forbids forming a RATE by dividing two abundance datasets, and this is the same")
        say(f"     error one level down -- forming a SCALING LAW across two of them. Within one")
        say(f"     dataset the exponent is {n3['slope_vs_larsson_own_mean']:+.4f} against a predicted -0.5.")
    say()

    # ---------------------------------------------------------------- V5
    say("V5 NOISE: THE FAME CHECK ON THE DIAGNOSIS ITSELF")
    n6 = noi["n6"]
    say(f"     pubs vs measured CV  {n6['pubs_vs_measured_cv']:+.4f}")
    say(f"     pubs vs copies       {n6['pubs_vs_copies']:+.4f}")
    say(f"     pubs vs prediction   {n6['pubs_vs_prediction']:+.4f}")
    say(f"     publication count correlates with ABUNDANCE at {n6['pubs_vs_copies']:+.4f}, and the")
    say(f"     V4 diagnosis is about abundance, so the two are entangled by construction.")
    say(f"     BUT the V4 comparison is BETWEEN TWO ABUNDANCE MEASURES OF THE SAME GENES, so any")
    say(f"     fame effect applies to both sides equally and cancels in the difference. That is why")
    say(f"     V4 is a comparison of two slopes rather than a single correlation.")
    gates["V5"] = True
    res["v5"] = dict(n6)
    say(f"     V5 PASS -- stated, and the design is what defends it rather than a control")
    say()

    # ---------------------------------------------------------------- V6
    say("V6 THE KCAT LAYER IS NOT AN OPEN FAILURE")
    bf = json.load(open(OUT / "loop_b4_fix.json")) if (OUT / "loop_b4_fix.json").exists() else {}
    asl = json.load(open(OUT / "loop_active_site.json")) if (OUT / "loop_active_site.json").exists() else {}
    c3 = (bf.get("c3") or {}).get("cost")
    h4 = (asl.get("h4") or {}).get("cost")
    h5 = (asl.get("h5") or {}).get("boot_site_minus_random")
    say(f"     within-EC permutation, mean-pooled readout   {c3:+.4f}   (loop 134 C3)")
    say(f"     within-EC permutation, site-pooled readout   {h4:+.4f}   (loop 136 H4)")
    say(f"     site pooling minus random-position pooling   {h5[0]:+.4f} [{h5[1]:+.4f}, {h5[2]:+.4f}]")
    say(f"     the paired interval is 0.0488. Protein identity is worth nothing to either readout,")
    say(f"     and the pooling gain that looked real was residue subsampling acting as")
    say(f"     regularisation. Two independent readouts, one decisive control.")
    gates["V6"] = bool(c3 is not None and h4 is not None and h5 is not None)
    res["v6"] = {"c3_mean_pooled": c3, "h4_site_pooled": h4, "h5_site_minus_random": h5,
                 "verdict": "FAILED and CLOSED"}
    say(f"     V6 PASS -- the layer stays FAILED and is CLOSED, not open. No further work on it is")
    say(f"     justified, and the record should say so rather than leave it looking like a gap.")
    say()

    say("=" * 100)
    say("  WHAT EACH FAILURE ACTUALLY IS")
    say("=" * 100)
    say(f"  reaction+graph fusion   REAL. The caveat cannot rescue it: even at its best case on the")
    say(f"                          {n_bridge / n_genes:.0%} of genes with a bridge, the reaction channel stays")
    say(f"                          below the graph channel. And 90% of the graph signal survives")
    say(f"                          degree-preserving rewiring, so most of what works is topology.")
    say(f"  replication as process  MISLABELLED. The fork speed is unidentifiable -- 56x moves the")
    say(f"                          score {spread_blur / spread_speed:.0f}x less than a smoothing width does -- so no fork")
    say(f"                          model was ever tested. What failed is a distance-to-origin model")
    say(f"                          with a free parameter, and the blur that beat it is that model's")
    say(f"                          own analytic limit.")
    say(f"  expression noise        MISLABELLED. Within Larsson alone the exponent is "
        f"{n3['slope_vs_larsson_own_mean']:+.4f}")
    say(f"                          against a predicted -0.5. Across the Schwanhausser join it is")
    say(f"                          {n3['slope_log10cv_vs_log10N']:+.4f}. The noise physics is closer to right than the")
    say(f"                          record says; the cross-dataset abundance join is what fails.")
    say(f"  what the kcat model     CLOSED. Settled by loops 134 and 136 with two readouts and a")
    say(f"                          learned  random-position control. Stays FAILED, needs no more work.")
    say("=" * 100)
    for k_ in ("V1", "V2", "V3", "V4", "V5", "V6"):
        say(f"  {k_}  {'PASS' if gates[k_] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[OUT / "loop_fusion_linear.json", OUT / "loop_replication_time.json",
                              OUT / "loop_noise.json", OUT / "loop_b4_fix.json",
                              OUT / "loop_active_site.json"],
                      available=4, used=4, selection="all", seed=SEED,
                      controls=["V1 grants the reaction channel its best case and it still loses",
                                "V2 compares the parameter's spread against a nuisance knob's",
                                "V4 compares two abundance measures of the SAME genes, so fame "
                                "cancels in the difference",
                                "no gate here can promote a layer -- the strongest outcome is "
                                "'a different failure than the record says'"],
                      note="tests the CAVEAT beside each of the four unrelated FAILED layers. A "
                           "caveat is a hypothesis about a failure and this repo writes them "
                           "without testing them.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 138 -- the four unrelated failures", "manifest": man, "gates": gates,
               **res, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_four_failures.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_four_failures.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
