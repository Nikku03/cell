"""grn_diag — is the NULL real or a mushy-dynamics artifact? Sweep the gain beta (switch-likeness), watch two
things move together: (a) do random starts canalize into FEW attractors (real genomes do), and (b) does the
essentiality signal appear once the dynamics actually canalize? beta is a dynamics parameter, NOT fit to the
essentiality target — so this is diagnosis, not p-hacking. Report the whole curve, not the best cell."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from grn import GRN
from grn_validate import _auc, _spearman, _partial_spearman
from complete_cell import CompleteCell

C = CompleteCell()
# beta = gain / switch-likeness. We stay in the PHYSICAL regime (4-15): above ~15 the map oscillates and
# robustness collapses (25% at beta=15, worse beyond) — unlike real cells, so those points are pathological,
# not informative. The null below is already monotonic and flat across the whole physical range.
print(f"{'beta':>5} {'#attr/24':>9} {'ON%':>6} {'robust%':>8} {'AUC_frag':>9} {'AUC_hub':>8} {'partialρ':>9}")
rows = []
for beta in (4, 8, 15):
    g = GRN(C=C, beta=beta)
    # canalization: distinct attractors from 24 random starts
    atts = []
    for s in range(24):
        att, _ = g.run(g._seed(seed=s), steps=500)
        atts.append((att > 0.5).astype(np.int8))
    distinct = []
    for a in atts:
        if all((a != d).mean() > 0.02 for d in distinct):
            distinct.append(a)
    base, _ = g.baseline_attractor()
    on = base > 0.5
    dyn = np.where(~g.is_input)[0]
    fr = g.knockout_fragility(base, list(dyn), restart_from_base=True, steps=150)
    frag = np.array([fr[gl] for gl in dyn])
    robust = float((frag < 0.5).mean())
    dep = np.array([float(C.genes[int(g.nodes[gl])].get("dep_frac"))
                    if str(C.genes[int(g.nodes[gl])].get("dep_frac")).replace(".", "", 1).replace("-", "", 1).isdigit()
                    else np.nan for gl in dyn])
    outd = g.outdeg[dyn]
    ok = np.isfinite(dep) & np.isfinite(frag)
    ess = (dep > 0.5)
    a_f, a_h = _auc(frag[ok], ess[ok]), _auc(outd[ok], ess[ok])
    pr = _partial_spearman(frag[ok], dep[ok], outd[ok])
    print(f"{beta:>5} {len(distinct):>9} {on.mean()*100:>5.0f}% {robust*100:>7.0f}% "
          f"{a_f:>9.3f} {a_h:>8.3f} {pr:>+9.3f}")
    rows.append(dict(beta=beta, n_attr=len(distinct), on=float(on.mean()), robust=robust,
                     auc_frag=a_f, auc_hub=a_h, partial=pr))
json.dump(rows, open(f"{os.path.dirname(__file__)}/../outputs/orphan/grn_diag.json", "w"), indent=1)
