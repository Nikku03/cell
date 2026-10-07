"""grn_validate — is the RUNNING cell (grn.py) real, or a pretty animation? Four blind tests, honest verdict.

The build tuned exactly one knob (the operating point / ON-fraction). Everything below is emergent — nothing here
was fit to the answer:

  T1  CONVERGENCE   — boot from many random initial states. Do they settle (fixed points), or blow up / oscillate
                      forever? A cell that never settles isn't a cell. Also: how many DISTINCT attractors (a real
                      genome has few stable cell-states, not thousands of noise basins).
  T2  NON-TRIVIALITY— are attractors a real mix of ON/OFF genes, or the degenerate all-on / all-off the activation
                      bias would give without a working threshold?
  T3  ROBUSTNESS    — the acceptance test we EARNED the right to demand: we proved real cells are robust (knockouts
                      rarely cascade, propagation AUC 0.50). So a faithful runtime must ALSO be robust — most single
                      knockouts should barely move the attractor. If ours is brittle, it's wrong.
  T4  ESSENTIALITY  — the payoff, and blind: when we knock a gene out and re-run, does how far the attractor MOVES
                      (dynamical fragility) predict MEASURED essentiality (dep_frac)? And — the confound that killed
                      naive versions before — does it beat the trivial out-degree (hub-ness) baseline? If dynamics
                      add nothing over connectivity, we say so.

-> outputs/orphan/grn_validation.json  + a printed verdict. A null result here is a real result; we report it.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from grn import GRN


def _auc(score, label):
    """rank-AUC: P(score higher for positives). label boolean."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def _partial_spearman(x, y, z):
    """Spearman(x, y) controlling for z — residualize ranks of x and y on ranks of z, then correlate."""
    x, y, z = map(lambda v: np.asarray(v, float), (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    rx, ry, rz = (np.argsort(np.argsort(v[m])).astype(float) for v in (x, y, z))
    def resid(a, b):
        b1 = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    ex, ey = resid(rx, rz), resid(ry, rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    from complete_cell import CompleteCell
    C = CompleteCell()
    g = GRN(C=C)
    R = {"n_nodes": int(g.n), "n_dynamic": int((~g.is_input).sum()), "n_input": int(g.is_input.sum()),
         "theta": float(g.theta), "beta": float(g.beta)}
    print(f"GRN dynamical core: {g.n} genes ({R['n_dynamic']} dynamic, {R['n_input']} inputs), theta={g.theta:.3f}")

    # ---------------------------------------------------------------- T1 convergence + attractor count
    print("\nT1 CONVERGENCE — boot from 40 random initial states")
    atts, steps_to_conv, conv = [], [], 0
    for s in range(40):
        x0 = g._seed(seed=s)
        att, info = g.run(x0, steps=500)
        conv += int(info["converged"]); steps_to_conv.append(info["steps"]); atts.append((att > 0.5).astype(np.int8))
    atts = np.array(atts)
    # cluster attractors by Hamming distance: count how many are >2% of genes apart from all earlier ones
    distinct = []
    for a in atts:
        if all((a != d).mean() > 0.02 for d in distinct):
            distinct.append(a)
    R["T1"] = {"converged_frac": conv / 40, "median_steps": float(np.median(steps_to_conv)),
               "n_distinct_attractors": len(distinct)}
    print(f"   converged: {conv}/40   median steps: {np.median(steps_to_conv):.0f}   "
          f"distinct attractors: {len(distinct)}")

    # ---------------------------------------------------------------- T2 non-triviality
    base, binfo = g.baseline_attractor()
    on = base > 0.5
    R["T2"] = {"on_frac": float(on.mean()), "mean_activity": float(base.mean()),
               "frac_intermediate": float(((base > 0.2) & (base < 0.8)).mean())}
    print(f"\nT2 NON-TRIVIALITY — ON-fraction {on.mean():.1%}  (all-on=100%/all-off=0% would be the failure)")
    trivial = on.mean() < 0.02 or on.mean() > 0.98
    print(f"   {'FAIL: degenerate' if trivial else 'ok: a real mix of ON and OFF genes'}")

    # ---------------------------------------------------------------- T3 robustness + T4 essentiality (shared KO sweep)
    print("\nT3/T4 KNOCKOUT SWEEP — clamp each dynamic gene OFF, re-run, measure attractor displacement")
    dyn = np.where(~g.is_input)[0]
    frag = np.full(g.n, np.nan)
    fr = g.knockout_fragility(base, list(dyn), restart_from_base=True, steps=150)
    for gl, v in fr.items():
        frag[gl] = v
    fvals = frag[dyn]
    # robustness: how many knockouts barely move the cell vs a few that shatter it
    small = float((fvals < 0.5).mean())        # displacement < 0.5 (in a space where a single flip ~= 1.0)
    R["T3"] = {"median_fragility": float(np.median(fvals)), "frac_small_effect": small,
               "max_fragility": float(np.nanmax(fvals)), "p95_fragility": float(np.nanpercentile(fvals, 95))}
    print(f"   median displacement {np.median(fvals):.3f} | {small:.0%} of knockouts barely move the attractor "
          f"| worst {np.nanmax(fvals):.2f}")
    print(f"   {'ok: mostly robust, few critical (matches real biology)' if small > 0.6 else 'brittle — suspicious'}")

    # T4: fragility vs MEASURED essentiality, controlled for out-degree (the hub confound)
    def depfrac(gl):
        gi = int(g.nodes[gl])
        try:
            return float(C.genes[gi].get("dep_frac"))
        except (TypeError, ValueError):
            return np.nan
    dep = np.array([depfrac(gl) for gl in dyn])
    outd = g.outdeg[dyn]
    onb = on[dyn].astype(float)
    ok = np.isfinite(dep) & np.isfinite(fvals)
    ess = (dep > 0.5).astype(bool)             # common-essential: dependency in >50% of DepMap lines
    print(f"\nT4 ESSENTIALITY (blind) — {ok.sum()} genes with measured dep_frac, {ess[ok].sum()} common-essential")
    auc_frag = _auc(fvals[ok], ess[ok])
    auc_out = _auc(outd[ok], ess[ok])
    auc_onb = _auc(onb[ok], ess[ok])
    sp_frag = _spearman(fvals[ok], dep[ok])
    sp_partial = _partial_spearman(fvals[ok], dep[ok], outd[ok])
    R["T4"] = {"n": int(ok.sum()), "n_essential": int(ess[ok].sum()),
               "auc_fragility": auc_frag, "auc_outdegree_baseline": auc_out, "auc_on_in_baseline": auc_onb,
               "spearman_fragility_dep": sp_frag, "partial_spearman_ctrl_outdeg": sp_partial}
    print(f"   AUC  fragility -> essential : {auc_frag:.3f}")
    print(f"   AUC  out-degree (baseline) : {auc_out:.3f}   <- must beat this for 'dynamics add signal'")
    print(f"   AUC  ON-in-baseline        : {auc_onb:.3f}")
    print(f"   Spearman(fragility, dep_frac)              : {sp_frag:+.3f}")
    print(f"   partial Spearman | out-degree (confound)   : {sp_partial:+.3f}   <- dynamics beyond hub-ness")

    # ---------------------------------------------------------------- verdict
    beats_hub = (auc_frag > auc_out + 0.02) and (abs(sp_partial) > 0.05)
    print("\n" + "=" * 66)
    if trivial:
        verdict = "NULL: attractor is degenerate — the runtime doesn't produce a real cell state."
    elif R["T1"]["converged_frac"] < 0.8:
        verdict = "NULL: doesn't reliably settle — not a stable dynamical system."
    elif beats_hub:
        verdict = (f"REAL SIGNAL: the running cell's fragility predicts measured essentiality (AUC {auc_frag:.2f}) "
                   f"and beats the hub baseline ({auc_out:.2f}). Dynamics add causal signal.")
    elif auc_frag > 0.6:
        verdict = (f"PARTIAL: fragility predicts essentiality (AUC {auc_frag:.2f}) but does NOT clearly beat the "
                   f"out-degree baseline ({auc_out:.2f}) — the signal may be connectivity, not dynamics.")
    else:
        verdict = (f"HONEST NULL: the cell runs and is robust, but its dynamical fragility does NOT predict "
                   f"essentiality (AUC {auc_frag:.2f}). Running it doesn't tell us which genes matter.")
    R["verdict"] = verdict
    print("VERDICT:", verdict)
    print("=" * 66)

    json.dump(R, open(f"{os.path.dirname(__file__)}/../outputs/orphan/grn_validation.json", "w"), indent=1)
    return R


if __name__ == "__main__":
    main()
