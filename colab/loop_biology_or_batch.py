"""
LOOP 269 -- IS THE LEARNED OPERATOR CELL BIOLOGY, OR IS IT BATCH?

Everything this arc has found rests on one object: W_c, a cell-line-specific operator that
maps a perturbation's average profile to that line's deviation from it. It is worth +0.0752
on Tahoe and +0.0670 on LINCS, it is overwhelmingly off-diagonal, and no annotation predicts
it. The natural reading is that it encodes something about the cell.

There is a rival reading that has never been excluded, and loop 255 measured how dangerous
it is. On LINCS, profiles sharing a PLATE correlated at 0.7788 -- against a construct
ceiling of 0.2487 and a random-pair baseline of 0.2683. Plate structure in that data is
three times larger than the biological ceiling. If cell lines were run on their own plates,
W_c would be a batch effect wearing a cell's name, and every result in the arc would be
about laboratory logistics.

TAHOE CAN SETTLE IT AND LINCS COULD NOT. Measured before writing this loop: 8 plates, 50
cell lines, and every plate carries all 50. Plate and line are FULLY CROSSED, so plate is
available as a control axis rather than being inseparable from the variable under test.

The question becomes concrete: does an operator fitted on one set of plates still describe
the SAME cell line on different plates? A property of the cell survives a change of batch.
A property of the batch does not.

    within-line, across-plate      W_c from plates A vs W_c from plates B
    across-line, same-plate        W_c from plates A vs W_d from plates A

If the operator is cell biology, the first similarity exceeds the second. If it is batch,
the second exceeds the first. Both are computed on identical amounts of data, from the same
matrix, with the same estimator, so the comparison moves exactly one thing.

GATES, ALL DECLARED BEFORE THE NUMBERS:

  V1 ARE PLATE AND LINE ACTUALLY CROSSED?
     Gate: PASS iff every cell line appears on at least 7 of the 8 plates. Without this the
     rest of the loop cannot be interpreted and should not be run.

  V2 HOW LARGE IS THE PLATE EFFECT HERE?                             -- requires V1
     Variance of the response attributable to plate, beside drug and line. Reported and
     gated only as a sanity floor: PASS iff plate is not the single largest term, which
     would mean the assay is dominated by batch before any model is fitted.

  V3 LOAD-BEARING -- DOES THE OPERATOR SURVIVE A CHANGE OF PLATE?    -- requires V1
     W_c fitted on plate-half A, scored on the SAME line's conditions from plate-half B,
     against the additive baseline. Gate: PASS iff at least +0.02.
     A FAIL means the operator does not describe the cell outside the batch it was measured
     in, and the whole arc reduces to a statement about experimental runs.

  V4 IS IT MORE A PROPERTY OF THE LINE THAN OF THE PLATE?            -- requires V3
     Entrywise similarity, within-line-across-plate against across-line-same-plate.
     Gate: PASS iff within-line exceeds across-line by at least 0.05.

  V5 DOES THE OWN-LINE OPERATOR BEAT ANOTHER LINE'S ON HELD-OUT PLATES?  -- requires V3
     Same evaluation as V3, using a different line's operator fitted on the same plates.
     Gate: PASS iff the own-line operator exceeds the other-line operator by at least 0.01.
     This is the predictive form of V4 and does not depend on operator similarity being the
     right way to compare two matrices.

  V6 WHAT THIS CANNOT SHOW
     Stated regardless of outcome.
"""
import json, time, collections
from pathlib import Path
import numpy as np

from gate_guard import Gates
from loop_tahoe import build, decompose, fit_dense, pear

SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
OUT = "outputs/loop_biology_or_batch.json"
SEED = 269269
OBS_PER_PARAM = 2.0
NHOLD = 25
LAM = [1e1, 1e2, 1e3, 1e4]
V1_MIN_PLATES, V3_BAR, V4_BAR, V5_BAR = 7, 0.02, 0.05, 0.01
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def main():
    t0 = time.time()
    G_ = Gates(emit=say)
    res = {"test": "is the cell-line operator a property of the cell or of the batch?"}
    say("=" * 104)
    say("LOOP 269 -- IS THE LEARNED OPERATOR CELL BIOLOGY, OR IS IT BATCH?")
    say("=" * 104)
    say("     Everything in this arc rests on W_c. Loop 255 measured why that is dangerous:")
    say("     on LINCS, profiles sharing a PLATE correlated at 0.7788, against a construct")
    say("     ceiling of 0.2487 and a random baseline of 0.2683 -- batch structure three times")
    say("     larger than the biological ceiling. If lines had their own plates, W_c would be")
    say("     a batch effect wearing a cell's name.")
    say("     Tahoe can settle it and LINCS could not: 8 plates, 50 lines, every plate carries")
    say("     all 50. A property of the CELL survives a change of plate. A property of the")
    say("     BATCH does not.")

    z = np.load(SCR / "tahoe_assembled.npz", allow_pickle=True)
    plate_all = np.array([str(x) for x in z["plate"]])
    M, combo, cell, genes = build()
    keep = np.array([str(x) for x in z["cell"]]) != "NA"
    plate = plate_all[keep]
    lines = sorted(set(cell.tolist())); plates = sorted(set(plate.tolist()))

    say("V1 ARE PLATE AND LINE ACTUALLY CROSSED?")
    per = {l: len(set(plate[cell == l].tolist())) for l in lines}
    mn = min(per.values())
    say(f"     {len(lines)} cell lines, {len(plates)} plates; plates per line min {mn}, "
        f"max {max(per.values())}")
    G_.add("V1", bool(mn >= V1_MIN_PLATES), stat=float(mn),
           if_true=lambda: f"V1 PASS -- every line is on at least {mn} plates, so plate is a "
                           f"control axis and not confounded with the variable under test",
           if_false=lambda: f"V1 FAIL -- some line appears on only {mn} plates; plate and line "
                            f"are too entangled to separate and the rest cannot be read")
    res["V1"] = {"min_plates_per_line": int(mn), "n_plates": len(plates)}

    say("V2 HOW LARGE IS THE PLATE EFFECT HERE?")
    vt = float(M.var())
    def share(key):
        mu = {k: M[key == k].mean(0) for k in np.unique(key)}
        return float(np.stack([mu[k] for k in key]).var()) / vt
    sd_, sl_, sp_ = share(combo), share(cell), share(plate)
    say(f"     drug {sd_:.1%}, line {sl_:.1%}, plate {sp_:.1%}")
    say(f"     LINCS plate correlation was 0.7788 against a 0.2487 construct ceiling")
    G_.add("V2", bool(sp_ < max(sd_, sl_)), stat=float(sp_), requires=("V1",),
           if_true=lambda: f"V2 PASS -- plate accounts for {sp_:.1%}, less than drug or line",
           if_false=lambda: f"V2 FAIL -- plate accounts for {sp_:.1%}, the largest single term; "
                            f"the assay is dominated by batch before any model is fitted")
    res["V2"] = {"drug": sd_, "line": sl_, "plate": sp_}

    pa = set(plates[:len(plates) // 2]); pb = set(plates[len(plates) // 2:])
    inA = np.array([p in pa for p in plate]); inB = ~inA
    say(f"     plate halves: A={sorted(pa)}  B={sorted(pb)}")

    npc = int(np.median([int(((cell == l) & inA).sum()) for l in lines]))
    K = max(50, min(len(genes), int(npc / OBS_PER_PARAM)))
    sel = np.argsort(-M.var(0))[:K]
    Ms = M[:, sel]
    A, R, _, _, _ = decompose(Ms, combo, cell)
    cmean = {k: Ms[combo == k].mean(0) for k in np.unique(combo)}
    grand = Ms.mean(0)
    Xg = np.stack([cmean[k] for k in combo]) - grand
    say(f"     operator on K={K} genes, ~{npc} conditions per line per plate-half "
        f"({npc/K:.1f} obs per parameter)")

    def fit(mask):
        best, be = LAM[0], np.inf
        ix = np.where(mask)[0]
        rng2 = np.random.default_rng(SEED)
        q = rng2.permutation(len(ix)); q1, q2 = ix[q[:len(q)//2]], ix[q[len(q)//2:]]
        for lam in LAM:
            W_ = fit_dense(Xg[q1], R[q1], lam)
            e = float(((Xg[q2] @ W_ - R[q2]) ** 2).mean())
            if e < be: be, best = e, lam
        return fit_dense(Xg[mask], R[mask], best)

    rng = np.random.default_rng(SEED)
    WA = {l: fit((cell == l) & inA) for l in lines}
    WB = {l: fit((cell == l) & inB) for l in lines}
    say(f"     fitted 2 x {len(lines)} operators   [{time.time()-t0:.0f}s]")

    say("V3 LOAD-BEARING -- DOES THE OPERATOR SURVIVE A CHANGE OF PLATE?")
    own, other, base = [], [], []
    hold = list(rng.choice(len(lines), size=min(NHOLD, len(lines)), replace=False))
    for li in hold:
        l = lines[li]
        m = (cell == l) & inB
        ix = np.where(m)[0]
        b0 = float(np.nanmean([pear(A[i], Ms[i]) for i in ix]))
        sc = lambda W: float(np.nanmean(
            [pear(A[i] + Xg[i] @ W, Ms[i]) for i in ix]))
        own.append(sc(WA[l]) - b0)
        d = lines[int(rng.choice([j for j in range(len(lines)) if j != li]))]
        other.append(sc(WA[d]) - b0)
        base.append(b0)
    o_, t_ = float(np.mean(own)), float(np.mean(other))
    say(f"     own-line operator from plates A, scored on the same line's plates B: {o_:+.4f}")
    say(f"     (baseline on those conditions {np.mean(base):.4f})")
    G_.add("V3", bool(o_ >= V3_BAR), stat=o_, requires=("V1",),
           if_true=lambda: f"V3 PASS -- the operator is worth {o_:+.4f} on plates it was never "
                           f"fitted on, so it describes the cell outside its own batch",
           if_false=lambda: f"V3 FAIL -- the operator is worth {o_:+.4f} across a plate "
                            f"boundary; it does not describe the cell outside the batch it was "
                            f"measured in")
    res["V3"] = {"own_across_plate": o_, "baseline": float(np.mean(base))}

    say("V4 IS IT MORE A PROPERTY OF THE LINE THAN OF THE PLATE?")
    wl = [pear(WA[l].ravel(), WB[l].ravel()) for l in lines]
    xl = []
    for _ in range(len(lines)):
        i, j = rng.choice(len(lines), size=2, replace=False)
        xl.append(pear(WA[lines[i]].ravel(), WA[lines[j]].ravel()))
    w_, x_ = float(np.mean(wl)), float(np.mean(xl))
    d4 = w_ - x_
    say(f"     within-line across-plate  {w_:+.4f}")
    say(f"     across-line same-plate    {x_:+.4f}")
    say(f"     difference {d4:+.4f}")
    G_.add("V4", bool(d4 >= V4_BAR), stat=d4, requires=("V3",),
           if_true=lambda: f"V4 PASS -- the same line across different plates is {d4:+.4f} more "
                           f"similar than different lines on the same plate; the operator "
                           f"tracks the CELL",
           if_false=lambda: f"V4 FAIL -- within-line across-plate similarity is only {d4:+.4f} "
                            f"above across-line same-plate; the operator tracks the PLATE")
    res["V4"] = {"within_line_across_plate": w_, "across_line_same_plate": x_, "delta": d4}

    say("V5 DOES THE OWN-LINE OPERATOR BEAT ANOTHER LINE'S ON HELD-OUT PLATES?")
    d5 = o_ - t_
    say(f"     own line {o_:+.4f} vs another line's operator {t_:+.4f}   {d5:+.4f}")
    say(f"     both fitted on plate-half A, both scored on plate-half B, same estimator")
    G_.add("V5", bool(d5 >= V5_BAR), stat=d5, requires=("V3",),
           if_true=lambda: f"V5 PASS -- the own-line operator is worth {d5:+.4f} over another "
                           f"line's on the same held-out plates",
           if_false=lambda: f"V5 FAIL -- the own-line operator is worth {d5:+.4f} over another "
                            f"line's; what transfers across plates is shared, not line-specific")
    res["V5"] = {"own": o_, "other": t_, "delta": d5}

    say("V6 WHAT THIS CANNOT SHOW")
    say("     Plate is the only batch variable in this table. Run date, passage number,")
    say("     confluence and operator are not recorded here, so 'survives a plate change' is")
    say("     not 'survives every technical factor'.")
    say("     A PASS says the operator is a stable property of the cell line as cultured in")
    say("     this laboratory. It does not say the operator is MECHANISM: loops 263 and 264")
    say("     showed it aligns with no pathway, no interaction graph and no annotation tested.")
    say("     Stable and unexplained are compatible.")
    say("     50 cancer lines, one platform, drug perturbations. Nothing here extends to")
    say("     primary cells or to genetic perturbation.")
    say("     Everything is on the double-centred residual, so by loop 260 any arm constant")
    say("     across lines is pinned near zero however expressive it is.")

    res["gates"] = {k: (v == "PASS") for k, v in G_.status.items()}
    res["void"] = [k for k, v in G_.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G_.summary(seconds=res["seconds"])
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
