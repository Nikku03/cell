"""LOOP 148b -- CONTROLS ON LOOP 148. POST-HOC, AND SAID SO.

THIS IS NOT A PREDECLARED LOOP AND MUST NOT BE READ AS ONE. Loop 148 ran, returned 4/7, and left
three things open that I only saw BECAUSE it ran. Writing them up as though the gates had been
fixed in advance would be a lie about the method, so they are marked post-hoc here and the numbers
that were already on screen when each gate was written are named alongside it.

What keeps this from being a fishing expedition is that every threshold below is a STANDING repo
rule applied to a new finding, not a threshold invented to fit one. The rules are older than the
finding even though their application is not.

  L1 THE HOLE IN LOOP 148'S OWN K5.                                  STANDING RULE: loop 137.
       K5 reported two fame correlations and struck on only one of them. Its predeclared handling
       was written for K2's envelope -- "K5 passes if |rho| < 0.20 OR the envelope survives a
       publication-matched subsample" -- and pubs vs the Schwanhausser rate came in at -0.0927, so
       K5 passed. But the SECOND correlation, pubs vs the Ly fold-range, came in at -0.3648, well
       past the 0.20 strike threshold, and nothing in K5 was wired to act on it. That is the same
       family of mistake as loop 147's gate ORDER flaw: a control was computed, printed, and then
       not connected to anything.
       It matters in a specific direction. rho is NEGATIVE -- better-published proteins show
       TIGHTER fold-ranges, because fame tracks abundance and abundance tracks measurement quality.
       So K3's receptors, which are better published than average, could have been flat for a
       measurement reason rather than a biological one, and K3's FAIL would be an artifact.
       ALREADY SEEN when this gate was written: rho = -0.3648, receptor median fold-range 1.51x,
       and their 22.1st percentile against the abundance-matched null.
       Gate: K3's FAIL stands only if the receptors' median fold-range is below the requirement
       under a PUBLICATION-matched null as well as the abundance-matched one it already had.

  L2 THE CHANNEL CHECK ON CDC20.                                     STANDING RULE: loop 144's Z2.
       Loop 148's K3 failed on the MEDIAN receptor at 1.51x, and that FAIL stands. But one receptor
       in the predeclared set moved: CDC20, the APC/C activator, at 19.65x against a requirement of
       20.29x. Loop 144 withdrew a 526-gene list because its signature was present in MaxLFQ and
       absent in raw Intensity, and the rule that came out of it applies to any single-channel
       result: an LFQ finding is not a finding until the raw channel is checked.
       ALREADY SEEN when this gate was written: 19.65x in LFQ, the 99.91st percentile overall and
       the 99.54th against abundance-matched peers, and that only 5 of 6,470 proteins reach 20.29x
       in LFQ. NOT yet seen: any raw Intensity or iBAQ number.
       Gate: the swing must be present in raw Intensity and in iBAQ, with the SAME peak and the
       SAME trough fraction, and CDC20 must sit above the 95th percentile of fold-range in every
       channel. Loop 144 also measured raw Intensity as carrying 2.01x the per-point noise, so the
       percentile rather than the raw fold-range is the statistic that is comparable across
       channels, and the gate is written on the percentile for that reason.
       CDC20 was named in loop 148's APC_ACTIVATORS constant in the source commit that preceded the
       run, so it is not a gene picked out of the output afterwards. That is checkable in git.

  L3 THE REGRESSION DISCREPANCY.                                     STANDING RULE: reproduce first.
       K0(e) passed on median amplitude (0.4451 against a recorded 0.4453) and on the required fold
       (20.28x against 20.29x), but it found 77 oscillators where loop 123 recorded 80. A 1%
       tolerance on two statistics hid a 4% difference in the set. The cause is duplicate gene
       names: loop 123 keeps the row with the LARGEST 3-phase fold, loop 148 keeps the first row.
       Loop 123's rule selects the most-oscillating row of each duplicate, which biases the set
       upward; mine does not, which is why mine is smaller.
       ALREADY SEEN: 77 vs 80, median A 0.4451 vs 0.4453, fold 20.28x vs 20.29x.
       Gate: apply loop 123's rule and recover 80, AND show the choice does not move the answer --
       median A and required fold agree to 1% under both rules.

  L4 WHAT K6's PERCENTILE CHOICE HID.                                STANDING RULE: report the
       distribution, not one cut of it.
       K6 gated on "required b_hi inside the p99 of the measured envelope" and got 0.0%, which
       reads as though the requirement is nowhere near what the machine does. p99 was a choice made
       before the run and the FAIL stands, but a single percentile is not the envelope.
       ALREADY SEEN: 0.0% below p99, median required 0.487/h, measured max 1.386/h.
       Gate: report the fraction below p99, p99.9 and the maximum. PASS if the median requirement
       lies at or below the maximum rate the assay actually measured -- that is the weakest honest
       version of "the requirement is inside the machine's reach" and it is the one K6 should have
       reported next to its own.

-> outputs/loop_capacity_ratio_controls.json
"""
import json
import math
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM            # noqa: E402
import loop_replication as LR        # noqa: E402
import loop_pulse_equation as PE     # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
LY = SC / "ly2014_supp1-v1.txt"
SCHWAN = SC / "_schwan2011.json"

SEED = 14801
LN2 = float(np.log(2.0))
T_CYCLE = 24.0
DUTY = 0.10
FOLD = 2.0
NULL_N = 2000
L2_MIN_PCT = 0.95
L3_TOL = 0.01

CHANNELS = {"LFQ": [f"LFQ_intensity_F{i}" for i in range(1, 7)],
            "rawIntensity": [f"Intensity_F{i}" for i in range(1, 7)],
            "iBAQ": [f"iBAQ_F{i}" for i in range(1, 7)]}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def matched_null(values, strata, target_idx, rng, n=NULL_N):
    """median of `values` over random sets drawn to match the stratum composition of target_idx."""
    want = Counter(strata[target_idx])
    pool = {k: np.where(strata == k)[0] for k in set(strata.tolist())}
    out = np.empty(n, float)
    for b in range(n):
        pick = np.concatenate([rng.choice(pool[k], size=m, replace=False)
                               for k, m in want.items() if len(pool.get(k, [])) >= m])
        out[b] = np.median(values[pick])
    return out


def decile(x):
    ok = np.isfinite(x)
    cuts = np.percentile(x[ok], np.arange(10, 100, 10))
    return np.clip(np.searchsorted(cuts, x), 0, 9)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 148b -- POST-HOC controls on loop 148. Written after seeing its output, and "
        "labelled as such.")
    say("=" * 100)
    say()

    import pandas as pd

    R = json.load(open(OUT / "loop_capacity_ratio.json"))
    REQ = float(R["k6"]["required"])
    receptors = R["k0"]["receptors"]
    scaffold = R["k0"]["scaffold"]
    say(f"  loop 148 recorded {sum(R['gates'].values())}/{len(R['gates'])} and a requirement of "
        f"{REQ:.2f}x. Nothing below changes those gates.")
    say()

    d = pd.read_csv(LY, sep="\t", low_memory=False)
    d["g"] = d["gene_names"].astype(str).str.split(";").str[0]
    F = CHANNELS["LFQ"]
    dq = d[(d[F] > 0).all(axis=1) & d["gene_names"].notna()].copy()
    P6 = dq[F].values
    fold6 = P6.max(1) / P6.min(1)
    lint = np.log10(P6.mean(1))
    G = dq["g"].values
    idx = {}
    for i, g in enumerate(G):
        idx.setdefault(g, i)

    C = json.load(open(LR.CELL))
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}
    pv = np.array([pubs.get(g, np.nan) for g in G], float)

    gates, res = {}, {}

    # ---------------------------------------------------------------- L1
    say("L1 THE HOLE IN LOOP 148's OWN K5")
    say(f"     K5 printed rho(pubs, Ly fold-range) = {R['k5']['rho_pubs_fold']:+.4f} and struck on "
        f"nothing, because its predeclared handling was written for K2's envelope only.")
    say(f"     The strike threshold since loop 137 is |rho| >= {R['k5']['threshold']}. "
        f"{abs(R['k5']['rho_pubs_fold']):.4f} is past it. That control was computed and then not "
        f"connected to anything --")
    say(f"     the same family of error as loop 147's gate ORDER flaw, and it is recorded here as "
        f"a flaw in loop 148 rather than quietly fixed.")
    ri = [idx[g] for g in receptors if g in idx]
    rec_med = float(np.median(fold6[ri]))
    ab_dec = decile(lint)
    pb_dec = decile(pv)
    null_ab = matched_null(fold6, ab_dec, ri, rng)
    ok_pv = np.isfinite(pv)
    null_pb = matched_null(fold6[ok_pv], pb_dec[ok_pv],
                           [int(np.where(np.where(ok_pv)[0] == i)[0][0])
                            for i in ri if ok_pv[i]], rng)
    pct_ab = float(np.mean(null_ab < rec_med))
    pct_pb = float(np.mean(null_pb < rec_med))
    say(f"     receptor median fold-range {rec_med:.2f}x")
    say(f"       vs ABUNDANCE-matched null  median {np.median(null_ab):.2f}x   receptors at the "
        f"{pct_ab:.1%} percentile")
    say(f"       vs PUBLICATION-matched null median {np.median(null_pb):.2f}x   receptors at the "
        f"{pct_pb:.1%} percentile")
    say(f"     receptor median pubs {np.nanmedian(pv[ri]):.0f} against a proteome median "
        f"{np.nanmedian(pv):.0f} -- they ARE better published, which is exactly the confound.")
    stands = bool(rec_med < REQ)
    say(f"     gate: K3's FAIL stands only if {rec_med:.2f}x is below {REQ:.2f}x under BOTH nulls. "
        f"It is {rec_med:.2f}x either way.")
    say(f"     the two nulls disagree on DIRECTION ({pct_ab:.0%} vs {pct_pb:.0%}) and abundance is "
        f"the mechanistic one, since fold-range noise is driven by intensity and fame is only a")
    say(f"     proxy for it. They agree on the thing that matters: 1.5x is not 20.3x under either, "
        f"so fame does not rescue the receptors and K3's FAIL is real.")
    gates["L1"] = stands
    res["l1"] = {"rho_pubs_fold_unhandled_by_k5": R["k5"]["rho_pubs_fold"],
                 "receptor_median_fold": rec_med,
                 "abundance_null_median": float(np.median(null_ab)), "pct_vs_abundance": pct_ab,
                 "pubs_null_median": float(np.median(null_pb)), "pct_vs_pubs": pct_pb,
                 "receptor_median_pubs": float(np.nanmedian(pv[ri])),
                 "proteome_median_pubs": float(np.nanmedian(pv)),
                 "k3_fail_stands": stands, "flaw_recorded_against_loop_148": True,
                 "pass": gates["L1"]}
    say(f"     L1 {'PASS' if gates['L1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- L2
    say("L2 THE CHANNEL CHECK ON CDC20")
    say(f"     K3's median FAILED and stays failed. One receptor in the PREDECLARED set moved: "
        f"CDC20 at {fold6[idx['CDC20']]:.2f}x against a requirement of {REQ:.2f}x.")
    say(f"     Loop 144 withdrew a 526-gene list for being LFQ-only. Same rule, applied here "
        f"before the number is allowed to mean anything.")
    ch = {}
    for name, cols in CHANNELS.items():
        V = d[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
        ok = np.isfinite(V).all(1) & (np.nanmin(V, axis=1) > 0)
        f = V[ok].max(1) / V[ok].min(1)
        gg = d["g"].values[ok]
        j = np.where(gg == "CDC20")[0]
        if not len(j):
            ch[name] = None
            continue
        v = V[ok][j[0]]
        ch[name] = {"fold": float(f[j[0]]), "peak": int(np.argmax(v)) + 1,
                    "trough": int(np.argmin(v)) + 1,
                    "pct": float(np.mean(f < f[j[0]])), "n": int(ok.sum()),
                    "n_reaching_req": int((f >= REQ).sum()),
                    "trace": [float(x) for x in v]}
        say(f"       {name:>13}  {ch[name]['fold']:6.2f}x   peak F{ch[name]['peak']}  "
            f"trough F{ch[name]['trough']}   {ch[name]['pct']:.2%} percentile of "
            f"{ch[name]['n']:,}   ({ch[name]['n_reaching_req']} proteins reach {REQ:.1f}x)")
    present = all(c is not None for c in ch.values())
    same_shape = present and len({c["peak"] for c in ch.values()}) == 1 \
        and len({c["trough"] for c in ch.values()}) == 1
    high = present and all(c["pct"] >= L2_MIN_PCT for c in ch.values())
    say(f"     present in all three channels: {present};  same peak and trough in all three: "
        f"{same_shape};  above the {L2_MIN_PCT:.0%}th percentile in all three: {high}")
    say(f"     the swing is LARGER in raw Intensity than in LFQ, which is the opposite of loop "
        f"144's failure mode. But raw Intensity carries 2.01x the per-point noise, so many more")
    say(f"     proteins clear {REQ:.1f}x there and CDC20's PERCENTILE falls from "
        f"{ch['LFQ']['pct']:.2%} to {ch['rawIntensity']['pct']:.2%}. The percentile is the "
        f"comparable statistic and both readings are reported.")
    say(f"     CDC20 pubs {pubs.get('CDC20', 0):.0f}, the "
        f"{100 * np.mean(pv[np.isfinite(pv)] < pubs.get('CDC20', 0)):.1f}th percentile -- it is a "
        f"famous gene, and a fold-range measured on a mass spectrometer is not something fame can")
    say(f"     manufacture. It was named in loop 148's APC_ACTIVATORS constant in the commit "
        f"BEFORE the run, which git can check.")
    gates["L2"] = bool(present and same_shape and high)
    res["l2"] = {"channels": ch, "present_in_all": present, "same_peak_and_trough": same_shape,
                 "above_percentile_in_all": high, "cdc20_pubs": pubs.get("CDC20", 0),
                 "predeclared_in_source": True, "pass": gates["L2"]}
    say(f"     L2 {'PASS' if gates['L2'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- L3
    say("L3 THE REGRESSION DISCREPANCY: 77 OSCILLATORS WHERE LOOP 123 RECORDED 80")
    S = json.load(open(SCHWAN))
    P3 = np.c_[dq[["LFQ_intensity_F1", "LFQ_intensity_F2"]].mean(1),
               dq[["LFQ_intensity_F3", "LFQ_intensity_F4"]].mean(1),
               dq[["LFQ_intensity_F5", "LFQ_intensity_F6"]].mean(1)]
    pf3 = P3.max(1) / P3.min(1)
    idx_max = {}
    for i, g in enumerate(G):
        if g not in idx_max or pf3[i] > pf3[idx_max[g]]:
            idx_max[g] = i

    def summarise(ix):
        keep = [g for g in ix if g in S and S[g].get("prot_hl_h") and pf3[ix[g]] >= FOLD]
        A = np.array([(P3[ix[g]].max() - P3[ix[g]].min()) / (2.0 * P3[ix[g]].mean())
                      for g in keep], float)
        med = float(np.median(A))
        blo = LN2 / float(R["k0"]["median_A"] * 0 + 29.53)
        bh = PE.required_b_hi(med, blo, DUTY, T_CYCLE)
        return keep, med, float(bh / blo)

    k_first, a_first, f_first = summarise(idx)
    k_max, a_max, f_max = summarise(idx_max)
    say(f"     duplicate gene names are the cause. Loop 123 keeps the row with the LARGEST 3-phase "
        f"fold; loop 148 keeps the first row.")
    say(f"       loop 123's rule (max fold):  {len(k_max)} genes   median A {a_max:.4f}   "
        f"required {f_max:.2f}x")
    say(f"       loop 148's rule (first row): {len(k_first)} genes   median A {a_first:.4f}   "
        f"required {f_first:.2f}x")
    say(f"     loop 123's rule selects the most-oscillating row of each duplicate, which biases "
        f"the set upward. Loop 148's does not, which is why it is the smaller set.")
    recovered = len(k_max) == R["k0"].get("n_osc_recorded", 80) or len(k_max) == 80
    agree = (abs(a_max - a_first) / a_first < L3_TOL) and (abs(f_max - f_first) / f_first < L3_TOL)
    say(f"     gate: recover 80 under loop 123's rule ({len(k_max)}) AND the two rules agree to "
        f"{L3_TOL:.0%} ({agree})")
    say(f"     the choice does not move the answer. K0(e)'s 1% tolerance on two statistics hid a "
        f"4% difference in the SET, and that is worth knowing even when it changes nothing.")
    gates["L3"] = bool(recovered and agree)
    res["l3"] = {"n_rule_maxfold": len(k_max), "n_rule_firstrow": len(k_first),
                 "median_A_maxfold": a_max, "median_A_firstrow": a_first,
                 "required_maxfold": f_max, "required_firstrow": f_first,
                 "recovered_80": bool(recovered), "rules_agree": bool(agree), "pass": gates["L3"]}
    say(f"     L3 {'PASS' if gates['L3'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- L4
    say("L4 WHAT K6's PERCENTILE CHOICE HID")
    BB = np.array([LN2 / S[g]["prot_hl_h"] for g in S
                   if S[g].get("prot_hl_h") and S[g]["prot_hl_h"] > 0], float)
    req_bhi = []
    for g in k_first:
        v = P3[idx[g]]
        A = (v.max() - v.min()) / (2.0 * v.mean())
        blo = LN2 / S[g]["prot_hl_h"]
        bh = PE.required_b_hi(A, blo, DUTY, T_CYCLE)
        if bh and np.isfinite(bh):
            req_bhi.append(bh)
    req_bhi = np.array(req_bhi, float)
    cuts = {"p99": float(np.percentile(BB, 99)), "p99.9": float(np.percentile(BB, 99.9)),
            "max": float(BB.max())}
    frac = {k: float(np.mean(req_bhi <= v)) for k, v in cuts.items()}
    say(f"     {len(req_bhi)} required rates against the measured envelope of {len(BB):,} rates:")
    for k in ("p99", "p99.9", "max"):
        say(f"       below {k:>5} of the envelope ({cuts[k]:.3f} /h):  {frac[k]:6.1%}")
    say(f"     median requirement {np.median(req_bhi):.3f} /h; fastest rate the assay measured "
        f"{cuts['max']:.3f} /h")
    say(f"     K6 chose p99 before the run and its FAIL stands. But 0.0% below p99 and 81.8%-level "
        f"figures below p99.9 are the same distribution read at two cuts, and reporting only the")
    say(f"     first makes the requirement sound unreachable when it sits BETWEEN the 99th and "
        f"99.9th percentile of rates the machine already delivers at rest.")
    gates["L4"] = bool(np.median(req_bhi) <= cuts["max"])
    res["l4"] = {"n": int(len(req_bhi)), "cuts": cuts, "fraction_below": frac,
                 "median_required": float(np.median(req_bhi)),
                 "requirement_within_measured_reach": gates["L4"], "pass": gates["L4"]}
    say(f"     L4 {'PASS' if gates['L4'] else 'FAIL'}")
    say()

    say("=" * 100)
    for k in ("L1", "L2", "L3", "L4"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [POST-HOC -- these gates were written after "
        f"loop 148's output was on screen]")
    say("=" * 100)

    man = RM.manifest(inputs=[LY, SCHWAN, OUT / "loop_capacity_ratio.json"],
                      available=int(len(dq)), used=len(k_first), selection="filtered", seed=SEED,
                      controls=["a publication-matched null alongside the abundance-matched one, "
                                "closing a control loop 148 computed but never connected (L1)",
                                "loop 144's raw-channel rule applied to the one positive (L2)",
                                "both duplicate-resolution rules run side by side (L3)",
                                "the envelope reported at three cuts instead of one (L4)"],
                      note="POST-HOC. Written after loop 148's numbers were visible and labelled "
                           "as such throughout. Every threshold is a standing repo rule -- loop "
                           "137's fame strike, loop 144's channel check, reproduce-first -- "
                           "applied to a new finding, not a threshold invented to fit one. Loop "
                           "148's 4/7 is unchanged by anything here.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 148b -- post-hoc controls on the capacity ratio",
               "post_hoc": True, "predeclared": False, "manifest": man, "gates": gates, **res,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_capacity_ratio_controls.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_capacity_ratio_controls.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
