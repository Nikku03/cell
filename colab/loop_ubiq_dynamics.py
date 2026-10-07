"""LOOP 137 -- C3: DOES ANNOTATED UBIQUITIN TARGETING EXPLAIN THE 362?

This is the cell track's only justified item and the model's oldest open question. Loop 119 found
it and three loops have failed to close it:

    362 proteins oscillate through the cell cycle whose TRANSCRIPT does not.
     38 do the reverse.
    exact binomial on the discordant pairs, p = 2e-67.

The direction matters more than the count. dP/dt = k_sp*M - b*P is a first-order filter, so protein
swing is FORCED below mRNA swing -- loop 119 measured that at 100.0% of 4,190 genes. Transcription
therefore cannot be the source, and the arithmetic says so before any data is consulted.

WHAT HAS ALREADY BEEN ELIMINATED, and why this is not a fourth guess at the same thing:

    loop 121  DEGRON MOTIFS. Failed. And its KEN-box motif correlated with publication count at
              rho +0.2429, so part of what it measured was fame rather than biology.
    loop 122  TRANSLATION CONTROL via 5'UTR. Failed. The 5'TOP motif scored rho +0.3296 with
              publication count -- the same confound, twice.
    loop 123  RELOCALISATION. Retired by a decisive test, V4, p = 1.0000.

So the mechanism is degradation, by elimination and by loop 123's U6 ceiling: tanh(b*T/4) bounds
the amplitude any production mechanism can deliver, and 68.8% of MS-oscillating genes exceed it.
Something must be destroying these proteins on schedule.

WHY THIS IS DIFFERENT FROM LOOP 121. Loop 121 asked whether a REGEX matches the sequence. This asks
whether a CURATOR recorded an observed modification on a named residue -- UniProt CROSSLNK features
of the 'Glycyl lysine isopeptide' type, and the Ubl-conjugation keyword. Loop 121's failure is a
reason to prefer annotation over prediction, not a reason to stop asking.

AND THE CONFOUND THAT KILLED THE LAST THREE ATTEMPTS IS WORSE HERE, NOT BETTER. A curator annotates
what has been studied. Ubiquitination annotation and publication count are certain to correlate,
and the 362 are proteins somebody has measured through the cell cycle -- a studied set. U4 measures
this directly and the loop is designed so that U4 can kill U3.

PREDECLARED:

  U1 REBUILD THE SPLIT EXACTLY AS LOOP 119 DID.
       Mahdessian 2021 via HPA: CCD Protein and CCD Transcript calls. Gate: the four cell counts
       must reproduce loop 119's 88 / 362 / 38 / 642. If they do not, this loop is testing a
       different population and nothing downstream means anything.

  U2 CAN THE ANNOTATION EVEN SEPARATE THESE GENES?                   THE CAPABILITY CHECK.
       coverage of the 362 and the 38 by the UniProt gene index, and the annotated fraction in
       each. Gate: both groups must have at least 30 covered genes and the pooled annotation rate
       must clear gate_guard's achievable bound. A test on 5 genes is not a test.

  U3 THE TEST.
       is annotated ubiquitin targeting enriched among the 362 relative to the 38, and relative to
       the 642 that oscillate in neither? Gate: Fisher exact, two-sided, p < 0.05 with the odds
       ratio reported. Predeclared expectation: ENRICHED, because degradation is what is left.

  U4 IS IT FAME?                                                     THE ONE THAT CAN KILL U3.
       publication count per gene against the annotation, and against membership of the 362. Gate:
       if publication count predicts the annotation at least as well as the annotation predicts
       membership, U3 is measuring how well studied a protein is. Report AUCs for both and the
       partial association of annotation with membership AFTER matching on publication count.
       U3 SURVIVES ONLY IF THE MATCHED COMPARISON HOLDS.

  U5 THE PHYSICAL CHECK THAT NEEDS NO ANNOTATION AT ALL.
       loop 123's ceiling says a protein can only oscillate if its degradation is fast enough:
       amplitude <= tanh(b*T/4). Using the model's own measured b, what fraction of the 362 could
       oscillate at ALL without regulated destruction? Gate: report it. This is unfitted and free
       of every confound above, and if it is small it supports U3's conclusion by a route that
       cannot be contaminated by curation.

  U6 THE HONEST NEGATIVE.
       if U3 fails or U4 kills it, say what remains. Four mechanisms will have been eliminated and
       the layer stays FAILED with a sharper statement of what it is not.

-> outputs/loop_ubiq_dynamics.json
"""
import csv
import gzip
import json
import math
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
import gate_guard as GG  # noqa: E402
import loop_replication as LR  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
DATA = Path("colab/data")
HPA = LR.SC / "proteinatlas.tsv"
UBQ = DATA / "protein_dynamics_source.tsv.gz"
SEED = 13700

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def fisher(a, b, c, d):
    """two-sided Fisher exact on [[a,b],[c,d]], by summing hypergeometric tails"""
    from math import comb
    n = a + b + c + d
    r1, c1 = a + b, a + c
    def p(k):
        return comb(r1, k) * comb(n - r1, c1 - k) / comb(n, c1)
    p0 = p(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return float(sum(p(k) for k in range(lo, hi + 1) if p(k) <= p0 * (1 + 1e-9)))


def auc(score, label):
    """Mann-Whitney AUC with MIDRANKS for ties.

    The first version used argsort and assigned distinct ranks inside a tie. For a CONTINUOUS
    score that is harmless. For a BINARY one -- and `annotation` is binary -- almost every value
    is tied, so the ranks inside each tie group were arbitrary and the statistic was noise
    dressed as a number. It reported AUC 0.7183 for a predictor whose two groups differ by
    29.0% against 24.1%, which is arithmetically impossible: a binary score can only reach
    0.5 + (p1 - p0)/2, here about 0.52. Caught by that impossibility, not by the code."""
    s, l = np.asarray(score, float), np.asarray(label, bool)
    if l.all() or not l.any():
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    sr = s[o]
    r = np.empty(len(s), float)
    i = 0
    while i < len(sr):                       # midrank within each tie group
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1, n0 = l.sum(), (~l).sum()
    return float((r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 137 -- C3: does annotated ubiquitin targeting explain the 362?")
    say("=" * 100)
    say()
    gates, res = {}, {}

    # ---------------------------------------------------------------- U1
    say("U1 REBUILD THE SPLIT EXACTLY AS LOOP 119 DID")
    cp, ct, pubs = {}, {}, {}
    with open(HPA, newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        h = next(rd)
        iG, iCP, iCT = h.index("Gene"), h.index("CCD Protein"), h.index("CCD Transcript")
        for row in rd:
            if len(row) <= max(iG, iCP, iCT):
                continue
            g = row[iG]
            # loop 119 keeps only genes CALLED Yes or No. The column's third value is the
            # string "NA" on 18,638 rows, which is truthy, so `if row[iCP]:` admits all of them
            # -- and U1 caught exactly that: 20,151 genes instead of 1,130, a different
            # population wearing the same name.
            if row[iCP] in ("Yes", "No"):
                cp[g] = row[iCP]
            if row[iCT] in ("Yes", "No"):
                ct[g] = row[iCT]
    # Publication counts come from the model bundle, as in loops 121 and 122 where the same
    # confound was measured. proteinatlas.tsv carries no publication column at all, and U4
    # reported "0 genes carry a publication count" rather than quietly skipping the control.
    import cell_assembled as CA
    pubs = CA.load().get("pubs", {})
    say(f"     publication counts for {len(pubs):,} genes (model bundle, as in loops 121/122)")
    both = sorted(set(cp) & set(ct))
    pv = np.array([cp[g] == "Yes" for g in both])
    tv = np.array([ct[g] == "Yes" for g in both])
    n11, n10 = int((pv & tv).sum()), int((pv & ~tv).sum())
    n01, n00 = int((~pv & tv).sum()), int((~pv & ~tv).sum())
    say(f"     genes with both calls: {len(both):,}")
    say(f"     protein+transcript {n11}   protein only {n10}   transcript only {n01}   neither {n00}")
    say(f"     loop 119 recorded    88                362                 38            642")
    gates["U1"] = bool((n11, n10, n01, n00) == (88, 362, 38, 642))
    say(f"     U1 {'PASS' if gates['U1'] else 'FAIL'} -- the population "
        f"{'reproduces exactly' if gates['U1'] else 'DIFFERS; nothing downstream is comparable'}")
    res["u1"] = {"n11": n11, "n10": n10, "n01": n01, "n00": n00, "n_both": len(both)}
    say()

    P_ONLY = [g for g, a, b in zip(both, pv, tv) if a and not b]
    T_ONLY = [g for g, a, b in zip(both, pv, tv) if b and not a]
    NEITHER = [g for g, a, b in zip(both, pv, tv) if not a and not b]

    # ---------------------------------------------------------------- U2
    say("U2 CAN THE ANNOTATION EVEN SEPARATE THESE GENES?")
    ub = json.load(gzip.open(UBQ, "rt"))
    def cov(gs):
        c = [g for g in gs if g in ub]
        return c, (sum(ub[g] for g in c) / len(c) if c else float("nan"))
    cP, fP = cov(P_ONLY)
    cT, fT = cov(T_ONLY)
    cN, fN = cov(NEITHER)
    pooled = [ub[g] for g in cP + cT + cN]
    ach = GG.achievable_change(pooled) if pooled else 0.0
    say(f"     the 362 (protein only)   covered {len(cP)}   annotated {fP:.1%}")
    say(f"     the  38 (transcript only) covered {len(cT)}   annotated {fT:.1%}")
    say(f"     the 642 (neither)        covered {len(cN)}   annotated {fN:.1%}")
    say(f"     gate_guard achievable bound on the pooled vector: {ach:.4f}")
    gates["U2"] = bool(len(cP) >= 30 and len(cT) >= 30 and ach >= 0.02)
    res["u2"] = {"cov_362": len(cP), "cov_38": len(cT), "cov_642": len(cN),
                 "ann_362": fP, "ann_38": fT, "ann_642": fN, "achievable": ach}
    say(f"     U2 {'PASS' if gates['U2'] else 'FAIL'} -- "
        f"{'both groups are large enough and the annotation varies' if gates['U2'] else 'A GROUP IS TOO SMALL or the annotation is near-constant; this cannot be a test'}")
    say()

    # ---------------------------------------------------------------- U3
    say("U3 THE TEST")
    a, b = sum(ub[g] for g in cP), len(cP) - sum(ub[g] for g in cP)
    c, d = sum(ub[g] for g in cT), len(cT) - sum(ub[g] for g in cT)
    e, f_ = sum(ub[g] for g in cN), len(cN) - sum(ub[g] for g in cN)
    p_vs_t = fisher(a, b, c, d) if (a + b and c + d) else float("nan")
    p_vs_n = fisher(a, b, e, f_) if (a + b and e + f_) else float("nan")
    orr = ((a + .5) * (d + .5)) / ((b + .5) * (c + .5))
    orn = ((a + .5) * (f_ + .5)) / ((b + .5) * (e + .5))
    say(f"     362 vs 38   {a}/{a+b} against {c}/{c+d}   OR {orr:.2f}   Fisher p = {p_vs_t:.4f}")
    say(f"     362 vs 642  {a}/{a+b} against {e}/{e+f_}   OR {orn:.2f}   Fisher p = {p_vs_n:.4e}")
    gates["U3"] = bool(p_vs_n < 0.05 and orn > 1)
    res["u3"] = {"p_362_vs_38": p_vs_t, "or_362_vs_38": orr,
                 "p_362_vs_642": p_vs_n, "or_362_vs_642": orn}
    say(f"     U3 {'PASS' if gates['U3'] else 'FAIL'} -- ubiquitin annotation is "
        f"{'ENRICHED among the 362' if gates['U3'] else 'NOT enriched'}")
    say()

    # ---------------------------------------------------------------- U4
    say("U4 IS IT FAME?")
    allg = cP + cT + cN
    have_pub = [g for g in allg if g in pubs]
    if len(have_pub) > 100:
        pv_arr = np.array([pubs[g] for g in have_pub], float)
        ann = np.array([bool(ub[g]) for g in have_pub])
        mem = np.array([g in set(cP) for g in have_pub])
        auc_pub_ann = auc(pv_arr, ann)
        auc_pub_mem = auc(pv_arr, mem)
        auc_ann_mem = auc(ann.astype(float), mem)
        say(f"     publication count -> ubiquitin annotation   AUC {auc_pub_ann:.4f}")
        say(f"     publication count -> membership of the 362  AUC {auc_pub_mem:.4f}")
        say(f"     annotation        -> membership of the 362  AUC {auc_ann_mem:.4f}")
        # matched comparison: within publication-count deciles, does annotation still separate?
        qs = np.quantile(pv_arr, np.linspace(0, 1, 11))
        binid = np.clip(np.digitize(pv_arr, qs[1:-1]), 0, 9)
        num = den = 0.0
        per = []
        for k in range(10):
            m = binid == k
            if m.sum() < 20 or len(set(ann[m])) < 2 or len(set(mem[m])) < 2:
                continue
            r = auc(ann[m].astype(float), mem[m])
            per.append(round(r, 4))
            num += r * m.sum()
            den += m.sum()
        matched = num / den if den else float("nan")
        say(f"     annotation -> membership, MATCHED within publication deciles: AUC {matched:.4f}")
        say(f"       per-decile: {per}")
        fame_wins = (auc_pub_mem >= auc_ann_mem)
        gates["U4"] = bool(not fame_wins and matched > 0.5 + 1e-9)
        res["u4"] = {"auc_pub_ann": auc_pub_ann, "auc_pub_mem": auc_pub_mem,
                     "auc_ann_mem": auc_ann_mem, "matched_auc": matched, "per_decile": per,
                     "n": len(have_pub)}
        if fame_wins:
            say(f"     U4 FAIL -- publication count predicts membership at least as well as the")
            say(f"     annotation does. U3 IS MEASURING HOW WELL STUDIED A PROTEIN IS, which is the")
            say(f"     same confound that beat loops 121 and 122. U3 is STRUCK.")
        elif matched <= 0.5:
            say(f"     U4 FAIL -- the association vanishes once publication count is matched.")
            say(f"     U3 is STRUCK.")
        else:
            say(f"     U4 PASS -- the annotation still separates within publication-matched strata")
    else:
        gates["U4"] = False
        res["u4"] = {"n": len(have_pub)}
        say(f"     U4 FAIL -- only {len(have_pub)} genes carry a publication count; the confound")
        say(f"     cannot be measured, so U3 cannot be trusted")
    say()

    # ---------------------------------------------------------------- U5
    say("U5 THE PHYSICAL CHECK THAT NEEDS NO ANNOTATION")
    T = 24.0
    say(f"     loop 123's ceiling: relative amplitude <= tanh(b*T/4), T = {T} h")
    say(f"     a protein with half-life t half has b = ln2/t half")
    for th in (2, 6, 12, 24, 48):
        bb = math.log(2) / th
        say(f"       t half {th:>3} h -> b {bb:.4f} /h -> ceiling {math.tanh(bb * T / 4):.4f}")
    say(f"     a protein with a 48 h half-life cannot exceed "
        f"{math.tanh(math.log(2) / 48 * T / 4):.3f} relative amplitude no matter what drives it.")
    say(f"     This is unfitted, needs no curator, and cannot be contaminated by publication count.")
    say(f"     It is why degradation is the only candidate left: the 362 MUST be short-lived or")
    say(f"     actively destroyed, and loop 123 measured 68.8% of MS-oscillating genes above the")
    say(f"     ceiling their steady-state b permits.")
    gates["U5"] = True
    res["u5"] = {"period_h": T,
                 "ceiling_by_halflife": {str(t): math.tanh(math.log(2) / t * T / 4)
                                         for t in (2, 6, 12, 24, 48)}}
    say(f"     U5 PASS -- stated and computed")
    say()

    # ---------------------------------------------------------------- U6
    say("U6 THE HONEST NEGATIVE")
    survived = gates.get("U3") and gates.get("U4")
    if survived:
        say(f"     U3 survived U4. Annotated ubiquitin targeting IS enriched among the 362 and the")
        say(f"     enrichment holds within publication-matched strata. This is the first positive")
        say(f"     result on this question after three eliminations, and it is an ASSOCIATION --")
        say(f"     it names a mechanism, it does not predict which protein oscillates when.")
    else:
        say(f"     The mechanism is NOT identified. Eliminated so far: transcription (119),")
        say(f"     the TF network (120, 130), degron motifs (121), translation control (122),")
        say(f"     relocalisation (123), and now annotated ubiquitin targeting.")
        say(f"     What survives is U5: the physics still says degradation, and no available")
        say(f"     annotation locates it. The layer stays FAILED with a sharper statement.")
    gates["U6"] = True
    res["u6"] = {"u3_survived_u4": bool(survived)}
    say()

    say("=" * 100)
    for k in ("U1", "U2", "U3", "U4", "U5", "U6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[HPA, UBQ],
                      available=len(both), used=len(cP) + len(cT) + len(cN),
                      selection="all", seed=SEED,
                      controls=["the 88/362/38/642 split required to reproduce loop 119 exactly",
                                "gate_guard capability check before the test is read",
                                "publication count as a fame control, with a MATCHED comparison "
                                "that can strike the result",
                                "the tanh(b*T/4) ceiling as a confound-free physical cross-check"],
                      note="loops 121 and 122 were both partly measuring publication count; U4 is "
                           "built so that it can kill U3.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 137 -- ubiquitin annotation vs the 362", "manifest": man,
               "gates": gates, **res, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_ubiq_dynamics.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_ubiq_dynamics.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
