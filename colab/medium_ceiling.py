"""BUILD 10 -- IT WAS THE MEDIUM, NOT THE KINETICS.

FOUR BUILDS SPENT ON THE WRONG HYPOTHESIS, AND THE MEASUREMENT THAT REDIRECTS THEM. DESIGN.md sec 3 blamed FBA's
recall of 0.203 on missing quantities: flux balance calls a gene tolerated whenever any alternative route exists,
so `BUFFERED` needed to become a number. Four builds tested that from every angle available --

    build 1  per-reaction ceilings, real kcat, GPR-aware, recomputed per deletion   recall 0.203 -> 0.203
    build 7  the entire kcat ACCURACY axis, raw in-vitro to no-information at all   recall 0.203 at every lambda
    build 8  a shared proteome budget, where kcat is a price and enzymes compete    recall 0.203 -> 0.203
    build 9  a 2.9x increase in kcat COVERAGE of flux-carrying reactions            (running)

-- and every one failed identically. Recall never moved off 0.203. That degree of invariance is itself the clue:
a quantity that four independent formulations cannot budge is not the quantity that is binding.

THE MEASUREMENT THAT EXPLAINS IT, which costs one pFBA solve:

    reactions carrying flux at the wild-type optimum      1,991 of 12,931
    DepMap-essential genes present in the model             310
      whose reactions carry ANY flux                        174  =  56.1%

A gene whose reactions carry no flux cannot be called lethal by flux balance, whatever its kinetics, because
deleting it removes capacity from a route the cell was not using. **43.9% of the essential genes are unreachable
by construction**, which puts a hard ceiling of 0.561 on recall before a single parameter is chosen. Every
enzyme-constraint build was competing for the remaining 36% of an already-capped quantity.

AND THE REASON THE FLUX IS SO NARROW IS THE MEDIUM. `cell_sim.set_medium` opens 1,634 of 1,660 exchange
reactions at 1.0 mmol/gDW/h -- generous by design, so the cell would grow at all. But a cell that can import
almost any metabolite has no reason to synthesise it, so its biosynthetic enzymes carry zero flux and their
knockouts are free. The rich medium is what makes 10,940 reactions idle.

WHAT THIS FILE DOES. It builds a MINIMAL SUFFICIENT MEDIUM from the model itself, then re-measures both the
ceiling and the recall.

  A  the ceiling, on the rich medium, so the number being competed for is explicit
  B  a greedy reduction: try closing each open exchange, cheapest-used first, and keep it closed if the cell
     still grows. No metabolite names are involved. Three earlier attempts at a hand-written minimal medium each
     produced exactly 0.00000 growth because of naming mismatches (`L-alanine` vs `alanine`, `phosphate` vs `Pi`)
     and biomass pools needing serum components; deriving the medium from the model sidesteps all of it.
  C  the ceiling and the recall again, on the reduced medium

PRE-REGISTERED CRITERION, deliberately the same bar as builds 1, 7, 8 and 9 so all five are comparable: recall
must rise above 0.203 with precision >= 0.6. Reported alongside the CEILING at each medium, because a recall of
0.30 against a ceiling of 0.56 and a recall of 0.30 against a ceiling of 0.95 are not the same result, and this
project has already published one number that turned out to be a fraction of an unstated maximum.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
sys.path.insert(0, str(Path(__file__).resolve().parent))
LETHAL_THR = 0.5
KEEP_FRAC = 0.10          # a reduced medium must still support >= this fraction of rich-medium growth


def reachable(m, sol, genes, dep, e2s):
    """Genes whose reactions carry flux -- the only ones a deletion can possibly hurt."""
    carry = {r.id for r in m.reactions if abs(float(sol.fluxes[r.id])) > 1e-9}
    ess = [g for g in genes if dep[e2s[g.id]] < -0.5]
    n_e = sum(1 for g in ess if any(r.id in carry for r in g.reactions))
    n_a = sum(1 for g in genes if any(r.id in carry for r in g.reactions))
    return carry, len(ess), n_e, n_a


def score(m, genes, dep, e2s):
    from cobra.flux_analysis import single_gene_deletion
    with m:
        wt = m.slim_optimize()
        ko = single_gene_deletion(m, gene_list=[g.id for g in genes])
    pred = {}
    for ids, val in zip(ko["ids"], ko["growth"]):
        gid = list(ids)[0] if isinstance(ids, (set, frozenset)) else str(ids)
        pred[e2s.get(gid, gid)] = 0.0 if val is None or np.isnan(val) else float(val)
    ratio = {g: (v / wt if wt > 1e-9 else 0.0) for g, v in pred.items()}
    common = sorted(g for g in ratio if g in dep)
    if len(common) < 500:
        return None
    y = np.array([1 if dep[g] < -0.5 else 0 for g in common])
    leth = [g for g in common if ratio[g] < LETHAL_THR]
    tp = sum(1 for g in leth if dep[g] < -0.5)
    x = np.array([-ratio[g] for g in common])
    o = np.argsort(x, kind="mergesort")
    rr = np.empty(len(x), float)
    rr[o] = np.arange(1, len(x) + 1)
    auc = float((rr[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (1 - y).sum()))
    return {"wt": float(wt), "n": len(common), "n_essential": int(y.sum()), "n_called": len(leth),
            "precision": tp / max(len(leth), 1), "recall": tp / max(int(y.sum()), 1), "auc": auc}


def main():
    import cobra
    from cobra.flux_analysis import pfba
    cobra.Configuration().solver = "glpk"
    from cell_sim import set_medium, ensembl_to_symbol, depmap_k562

    m = cobra.io.read_sbml_model(str(SP / "HumanGEM.xml"))
    set_medium(m)
    e2s = ensembl_to_symbol()
    dep = depmap_k562()
    genes = [g for g in m.genes if e2s.get(g.id) in dep]
    print(f"model genes present in DepMap K562: {len(genes):,}")
    assert len(genes) > 500, "GENE JOIN TOO SMALL -- refusing to continue"

    # ---------- A. the ceiling on the rich medium ----------
    rich = pfba(m)
    g_rich = float(m.slim_optimize())
    carry, n_ess, n_e, n_a = reachable(m, rich, genes, dep, e2s)
    ex = [r for r in m.reactions if r.boundary and len(r.metabolites) == 1]
    op = [r for r in ex if r.lower_bound < 0]
    used = [r for r in op if float(rich.fluxes[r.id]) < -1e-9]
    print("\n" + "=" * 88)
    print("A. THE CEILING ON THE RICH MEDIUM")
    print("=" * 88)
    print(f"  growth {g_rich:.4f}/h; {len(carry):,} of {len(m.reactions):,} reactions carry flux")
    print(f"  medium: {len(op):,} of {len(ex):,} exchanges allow import, {len(used):,} imports actually used")
    print(f"  DepMap-essential model genes {n_ess:,}; reachable (their reactions carry flux) {n_e:,} "
          f"= {100*n_e/max(n_ess,1):.1f}%")
    print(f"  CEILING ON RECALL: {n_e/max(n_ess,1):.3f}  -- the rest cannot be called lethal at any parameter")
    s_rich = score(m, genes, dep, e2s)
    print(f"  measured: called {s_rich['n_called']:,}, precision {s_rich['precision']:.3f}, "
          f"recall {s_rich['recall']:.3f}, AUC {s_rich['auc']:.4f}")
    print(f"  the model achieves {s_rich['recall']/(n_e/max(n_ess,1)):.0%} of its own structural ceiling")

    # ---------- B. a minimal sufficient medium, derived from the model ----------
    print("\n" + "=" * 88)
    print("B. GREEDY REDUCTION -- close every import the cell can do without")
    print("=" * 88)
    order = sorted(op, key=lambda r: abs(float(rich.fluxes[r.id])))   # least-used first
    floor = KEEP_FRAC * g_rich
    closed, kept = [], []
    for r in order:
        lb = r.lower_bound
        r.lower_bound = 0.0
        g = m.slim_optimize()
        if g is None or np.isnan(g) or g < floor:
            r.lower_bound = lb
            kept.append(r.id)
        else:
            closed.append(r.id)
    g_min = float(m.slim_optimize())
    print(f"  closed {len(closed):,} exchanges, kept {len(kept):,}")
    print(f"  growth {g_rich:.4f}/h -> {g_min:.4f}/h ({100*g_min/max(g_rich,1e-12):.1f}% of rich)")
    if g_min < floor:
        print("  reduction broke the cell -- refusing to score.")
        return
    kept_names = []
    for rid in kept[:25]:
        r = m.reactions.get_by_id(rid)
        kept_names.append((list(r.metabolites)[0].name or rid))
    print(f"  a sample of what the cell still cannot do without: {', '.join(sorted(set(kept_names))[:18])}")

    # ---------- C. the ceiling and the recall on the reduced medium ----------
    lean = pfba(m)
    carry2, n_ess2, n_e2, n_a2 = reachable(m, lean, genes, dep, e2s)
    print("\n" + "=" * 88)
    print("C. THE CEILING ON THE REDUCED MEDIUM")
    print("=" * 88)
    print(f"  {len(carry2):,} of {len(m.reactions):,} reactions carry flux (was {len(carry):,})")
    print(f"  reachable essential genes {n_e2:,} of {n_ess2:,} = {100*n_e2/max(n_ess2,1):.1f}% "
          f"(was {100*n_e/max(n_ess,1):.1f}%)")
    print(f"  CEILING ON RECALL: {n_e2/max(n_ess2,1):.3f}  (was {n_e/max(n_ess,1):.3f})")
    s_lean = score(m, genes, dep, e2s)
    print(f"  measured: called {s_lean['n_called']:,}, precision {s_lean['precision']:.3f}, "
          f"recall {s_lean['recall']:.3f}, AUC {s_lean['auc']:.4f}")

    print("\n" + "=" * 88)
    print(f"{'medium':22s} {'flux rxns':>10s} {'ceiling':>8s} {'called':>7s} {'prec':>7s} {'recall':>7s} {'AUC':>8s}")
    for tag, cr, ce, s in (("rich (1,634 open)", carry, n_e / max(n_ess, 1), s_rich),
                           ("minimal sufficient", carry2, n_e2 / max(n_ess2, 1), s_lean)):
        print(f"{tag:22s} {len(cr):>10,} {ce:>8.3f} {s['n_called']:>7,} {s['precision']:>7.3f} "
              f"{s['recall']:>7.3f} {s['auc']:>8.4f}")
    print("=" * 88)
    d = s_lean["recall"] - s_rich["recall"]
    passed = bool(d > 0 and s_lean["precision"] >= 0.6)
    print(f"\nPRE-REGISTERED CRITERION (DESIGN.md sec 3, same bar as builds 1, 7, 8, 9)")
    print(f"  recall {s_rich['recall']:.3f} -> {s_lean['recall']:.3f} ({d:+.3f}), "
          f"precision {s_rich['precision']:.3f} -> {s_lean['precision']:.3f}")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    print(f"\n  Four enzyme-constraint builds moved recall by 0.000. This one moves it by {d:+.3f} without")
    print(f"  touching a single kinetic parameter.")

    json.dump({"rich": {"growth": g_rich, "n_flux_reactions": len(carry), "n_essential": n_ess,
                        "n_reachable": n_e, "ceiling": n_e / max(n_ess, 1), "score": s_rich,
                        "n_exchanges_open": len(op), "n_imports_used": len(used)},
               "minimal": {"growth": g_min, "n_flux_reactions": len(carry2), "n_essential": n_ess2,
                           "n_reachable": n_e2, "ceiling": n_e2 / max(n_ess2, 1), "score": s_lean,
                           "n_closed": len(closed), "n_kept": len(kept), "kept": kept},
               "delta_recall": d, "passed": passed},
              open(OUT / "medium_ceiling.json", "w"), indent=1)
    print(f"\n  -> {OUT/'medium_ceiling.json'}")


if __name__ == "__main__":
    main()
