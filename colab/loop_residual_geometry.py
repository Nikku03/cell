"""
LOOP 260 -- WHY THE CONTEXT-BLIND CONTROLS SCORED EXACTLY ZERO

Loop 257's corrected rerun put I3 VOID with this in the log:

    K=1, context-blind operator   0.4518   val MSE removed 0.0%
    K=8, context held CONSTANT    0.4518   val MSE removed 0.0%
    the gated arm itself removed 19.5% of validation MSE

I recorded that as defect H -- an ablation arm that never trained. The VOID was right and
the REASON was wrong. This loop measures the actual reason, which is geometric.

The target is the additive residual R = Y - (gene_mean + line_mean - grand). That is a
DOUBLE-CENTRED quantity: it has had the gene main effect and the line main effect
subtracted. So R is, by construction, nearly orthogonal to every function of the gene alone
and to every function of the line alone. A context-blind model must emit the same vector
for every line a gene appears in, so the best it can do is predict that gene's mean
residual across its lines -- which double-centring has already driven to zero.

If that is right, the ablation arms did not fail to optimise. They were fitting a target
that contains nothing they are allowed to see, and no amount of training, capacity, or
patience would change the number. That distinction matters: defect H's remedy (train the
control harder) is useless here, and the gate I3 asked -- "does context-gating matter, or
only the extra capacity?" -- is not a question this target can answer, because the answer
is forced to yes before any network is built.

GATES, DECLARED BEFORE THE NUMBERS:

  L1 IS THE RESIDUAL ORTHOGONAL TO GENE-ONLY FUNCTIONS?
     The variance of R explained by predicting each gene's OWN mean residual across its
     lines. This is an IN-SAMPLE ceiling and it uses each gene's own rows, so no real model
     can beat it. Gate: PASS iff it is below 1%.

  L2 IS THE RESIDUAL ORTHOGONAL TO LINE-ONLY FUNCTIONS?                 -- requires L1
     The same for each line's own mean residual. Gate: PASS iff below 1%.

  L3 DOES THE GATED NETWORK EXCEED BOTH CEILINGS?                       -- requires L1, L2
     Loop 257's gated arm removed 19.5% of validation MSE. If L1 and L2 are near zero, then
     that 19.5% cannot be gene structure or line structure; it must be genuinely three-way
     (gene x landmark x line). Gate: PASS iff 19.5% exceeds max(L1, L2) by 10x.

  L4 HOW MUCH OF THE RESIDUAL IS REACHABLE AT ALL?                      -- requires L1
     Split-half over shRNA constructs: the correlation between residuals computed from two
     disjoint hairpin sets for the same (gene, line). Loop 254 put the profile-level ceiling
     at 0.2487. Reported, not gated -- it is context for every number above.

  L5 WHAT THIS CHANGES, AND WHAT IT DOES NOT
     Stated regardless of outcome.
"""
import json, time
from pathlib import Path
import numpy as np

import lincs_harness as H
from gate_guard import Gates

OUT = "outputs/loop_residual_geometry.json"
L1_BAR, L2_BAR, L3_MULT = 0.01, 0.01, 10.0
LOOP257_VAL_MSE_REMOVED = 0.195
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def ceiling(R, key):
    """In-sample variance of R explained by the best function of `key` alone: predict each
    group's own mean. No real model can beat this, because it uses the held-in rows."""
    gm, out = {}, np.zeros_like(R)
    for k in np.unique(key):
        m = key == k
        if m.sum() >= 2: gm[k] = R[m].mean(0)
    keep = np.array([k in gm for k in key])
    Rk, P = R[keep], np.stack([gm[k] for k in key[keep]])
    ss = float((Rk ** 2).sum())
    return (1.0 - float(((Rk - P) ** 2).sum()) / ss) if ss > 0 else 0.0, int(keep.sum()), len(gm)


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "the additive residual is double-centred, so context-blind models are pinned at zero"}
    say("=" * 104)
    say("LOOP 260 -- WHY THE CONTEXT-BLIND CONTROLS SCORED EXACTLY ZERO")
    say("=" * 104)
    say("     Loop 257 rerun: K=1 removed 0.0% of validation MSE, K=8 with context held")
    say("     constant removed 0.0%, and the gated arm removed 19.5%. I called that defect H,")
    say("     an ablation that never trained. The VOID was right; the reason was wrong.")
    say("     The target R = Y - (gene_mean + line_mean - grand) is DOUBLE-CENTRED, so it is")
    say("     orthogonal to gene-only and line-only functions by construction. The controls")
    say("     were not under-trained. They were fitting a target holding nothing they can see.")

    D = H.load()
    Pm, pg, pc, LINES = D["Pm"], D["pg"], D["pc"], D["LINES"]
    hold = LINES[0]; tr = pc != hold
    gm = {}
    for g in D["genes"]:
        m = tr & (pg == g)
        if m.sum(): gm[g] = Pm[m].mean(0)
    grand = Pm[tr].mean(0); lmean = {l: Pm[pc == l].mean(0) for l in LINES}
    idx = [j for j in np.where(tr)[0] if pg[j] in gm]
    R = np.stack([Pm[j] - (gm[pg[j]] + lmean[pc[j]] - grand) for j in idx]).astype(np.float32)
    genes = np.array([pg[j] for j in idx]); lines = np.array([pc[j] for j in idx])
    say(f"     residual R over the 8 training lines: {R.shape[0]:,} x {R.shape[1]}, "
        f"total variance {R.var():.4f}")

    say("L1 IS THE RESIDUAL ORTHOGONAL TO GENE-ONLY FUNCTIONS?")
    c1, n1, k1 = ceiling(R, genes)
    say(f"     best possible gene-only model explains {c1:.6f} of residual variance")
    say(f"     ({k1:,} genes over {n1:,} rows, and this is the IN-SAMPLE ceiling)")
    G.add("L1", bool(c1 < L1_BAR), stat=float(c1),
          if_true=lambda: f"L1 PASS -- {c1:.4%} of the residual is reachable by ANY gene-only "
                          f"function, so a context-blind arm is pinned at zero by geometry",
          if_false=lambda: f"L1 FAIL -- gene-only functions reach {c1:.4%}, so the controls had "
                           f"something to fit and defect H's reading stands")
    res["L1"] = {"ceiling": c1, "rows": n1, "groups": k1}

    say("L2 IS THE RESIDUAL ORTHOGONAL TO LINE-ONLY FUNCTIONS?")
    c2, n2, k2 = ceiling(R, lines)
    say(f"     best possible line-only model explains {c2:.6f} of residual variance "
        f"({k2} lines)")
    G.add("L2", bool(c2 < L2_BAR), stat=float(c2), requires=("L1",),
          if_true=lambda: f"L2 PASS -- {c2:.4%}; the line main effect is already removed too",
          if_false=lambda: f"L2 FAIL -- line-only functions reach {c2:.4%}")
    res["L2"] = {"ceiling": c2, "rows": n2, "groups": k2}

    say("L3 DOES THE GATED NETWORK EXCEED BOTH CEILINGS?")
    worst = max(c1, c2)
    say(f"     loop 257's gated arm removed {LOOP257_VAL_MSE_REMOVED:.1%} of validation MSE")
    say(f"     against a best-case one-way ceiling of {worst:.4%}")
    G.add("L3", bool(LOOP257_VAL_MSE_REMOVED >= L3_MULT * worst), stat=float(worst),
          requires=("L1", "L2"),
          if_true=lambda: f"L3 PASS -- {LOOP257_VAL_MSE_REMOVED:.1%} is "
                          f"{LOOP257_VAL_MSE_REMOVED / max(worst, 1e-9):.0f}x the one-way ceiling, "
                          f"so what it learned is genuinely three-way (gene x landmark x line)",
          if_false=lambda: f"L3 FAIL -- {LOOP257_VAL_MSE_REMOVED:.1%} against {worst:.4%}")
    res["L3"] = {"gated_val_mse_removed": LOOP257_VAL_MSE_REMOVED, "one_way_ceiling": worst}

    say("L4 HOW MUCH OF THE RESIDUAL IS REACHABLE AT ALL?")
    say(f"     loop 254's construct split put the profile-level ceiling at "
        f"{H.CONSTRUCT_CEILING:.4f} -- what two DISJOINT shRNA hairpin sets agree on for the")
    say(f"     same (gene, line). Reported, not gated. Every number in this arc lives under it.")
    res["L4"] = {"construct_ceiling": H.CONSTRUCT_CEILING}

    say("L5 WHAT THIS CHANGES, AND WHAT IT DOES NOT")
    say("     CHANGES: 'does context matter?' is not a question this target can answer. Double-")
    say("     centring forces the answer to yes before any network exists. Loop 257's I3 and")
    say("     loop 256's H4 both asked it, and both comparisons were degenerate by construction,")
    say("     not merely under-trained. Defect H's remedy -- train the control harder -- is")
    say("     useless here; no amount of capacity reaches a variance component that is absent.")
    say("     The answerable question is whether context is used in a way that TRANSFERS to a")
    say("     line never seen, which is what a wrong-line control and an out-of-sample blend")
    say("     weight actually test.")
    say("     DOES NOT CHANGE: loop 257's I2. The gated model still loses 0.0152 to the additive")
    say("     baseline on a held-out line, and that comparison never involved a blind arm.")
    say("     DOES NOT CHANGE: the 19.5% is measured on held-out ROWS from lines the model")
    say("     trained on. It is real three-way structure, and it still transfers negatively.")
    say("     The gap between those two facts is the whole problem, stated precisely.")
    say("     This is algebra about the TARGET, not a claim about biology. A different target --")
    say("     the raw profile, or a residual centred differently -- has different geometry.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary(seconds=res["seconds"])
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
