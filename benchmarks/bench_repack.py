"""Algorithm 2 measured on real interfaces: treewidth, exactness, and where greedy breaks.

PAIRED BOUND vs UNBOUND, and that pairing is the point.

The first version of this benchmark ran on BOUND structures only and reported that greedy
matched the exact optimum on 84 of 84 instances. That result is real but it is close to
uninformative, because in a bound structure the deposited side chains ARE the
crystallographic optimum: rotamer offset 0 is already the answer, so both the exact solver
and a greedy one start at the optimum and have nothing to find. The same defect showed up
in Algorithm 3, where exact two-sided repacking contributed 0.0000 kcal/mol on a bound
structure for exactly this reason.

Unbound side chains are in the WRONG rotamers -- that is what makes a docking case medium
or difficult -- so an unbound interface is the only place a packing guarantee can pay.
This script now runs both arms over the SAME cases, the SAME interface sizes and the SAME
rotamer library, so the single thing that differs between them is whether the side chains
came from the complex or from the free component.

Two questions the claim depends on, neither assumed:
  1. What IS the treewidth of a real interface contact graph as it grows? REM is exact in
     d^treewidth, so this is the whole cost story.
  2. Where does a competent greedy packer actually fail? A guaranteed optimum is only worth
     having where the guarantee bites.
"""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np
from rem.docking.data import load_case
from rem.docking.repack import build_from_case
import json as _j

CASES = [c["id"] for c in _j.load(open("benchmarks/db5_classification.json"))["usable"]][:40]
SIZES = [4, 6, 8, 10, 12, 14, 16]
NCHI1, NCHI2 = 3, 2
N_CASES = 12


def sweep(bound: bool) -> list:
    tag = "bound" if bound else "UNBOUND"
    print(f"\n{'='*94}\n  {tag} side chains\n{'='*94}")
    print(f"{'case':6s} {'n_res':>5s} {'edges':>6s} {'dens':>5s} {'tw':>3s} {'configs':>10s} "
          f"{'exact':>11s} {'greedy':>11s} {'gap':>9s} {'ms':>7s}")
    rows = []
    for cid in CASES[:N_CASES]:
        try:
            case = load_case(cid)
        except Exception as e:
            print(f"{cid:6s}  LOAD ERROR {type(e).__name__}")
            continue
        for n in SIZES:
            try:
                p = build_from_case(case, bound=bound, max_residues=n,
                                    n_chi1=NCHI1, n_chi2=NCHI2)
                if len(p.res_keys) < n:
                    continue
                g, edges = p.to_factorgraph()
                tw = g.treewidth()
                npair = len(p.res_keys) * (len(p.res_keys) - 1) / 2
                dens = len(edges) / npair if npair else 0.0
                ncfg = float(np.prod([len(p.rot[r]) for r in p.res_keys]))
                try:
                    ex = p.solve_exact(g)
                except MemoryError:
                    print(f"{cid:6s} {n:5d} {len(edges):6d} {dens:5.2f} {tw:3d} {ncfg:10.2e} "
                          f"{'TREEWIDTH WALL':>33s}")
                    rows.append({"case": cid, "n": n, "tw": tw, "edges": len(edges),
                                 "density": dens, "configs": ncfg, "wall": True,
                                 "bound": bound})
                    continue
                gr = p.solve_greedy(g, restarts=20)
                gap = gr["energy"] - ex["energy"]
                # Is the DEPOSITED conformation (offset 0 everywhere) already optimal? On a
                # bound structure it usually is, and that is precisely why the bound arm
                # cannot discriminate. Measured, not assumed.
                dep = p.energy_of({r: 0 for r in p.res_keys}, g)
                print(f"{cid:6s} {n:5d} {len(edges):6d} {dens:5.2f} {tw:3d} {ncfg:10.2e} "
                      f"{ex['energy']:11.4f} {gr['energy']:11.4f} {gap:+9.4f} "
                      f"{ex['seconds']*1000:7.1f}")
                rows.append({"case": cid, "n": n, "tw": tw, "edges": len(edges),
                             "density": dens, "configs": ncfg, "exact": ex["energy"],
                             "greedy": gr["energy"], "gap": gap, "deposited": dep,
                             "deposited_gap": dep - ex["energy"],
                             "ms": ex["seconds"] * 1000, "wall": False, "bound": bound})
            except Exception as e:
                print(f"{cid:6s} {n:5d}  ERROR {type(e).__name__}: {str(e)[:40]}")
    return rows


def summarize(rows: list, tag: str) -> dict:
    ok = [r for r in rows if not r.get("wall")]
    walls = [r for r in rows if r.get("wall")]
    print(f"\n  --- {tag} ---")
    print(f"  solved exactly: {len(ok)}   hit the treewidth wall: {len(walls)}")
    out = {"n_ok": len(ok), "n_wall": len(walls)}
    if not ok:
        return out
    gaps = np.array([r["gap"] for r in ok])
    miss = (gaps > 1e-6)
    print(f"  greedy MISSED the optimum on {miss.sum()}/{len(gaps)} instances")
    if miss.any():
        print(f"    when it misses: mean {gaps[miss].mean():+.4f}, "
              f"worst {gaps.max():+.4f} kcal/mol")
    dg = np.array([r["deposited_gap"] for r in ok])
    dmiss = (dg > 1e-6)
    print(f"  the DEPOSITED conformation was already optimal on "
          f"{(~dmiss).sum()}/{len(dg)} instances")
    if dmiss.any():
        print(f"    where it was not: exact beat deposited by mean {dg[dmiss].mean():.4f}, "
              f"max {dg.max():.4f} kcal/mol")
    out.update({"greedy_miss": int(miss.sum()), "n": len(gaps),
                "greedy_worst": float(gaps.max()),
                "greedy_mean_when_miss": float(gaps[miss].mean()) if miss.any() else 0.0,
                "deposited_already_optimal": int((~dmiss).sum()),
                "deposited_gap_max": float(dg.max()),
                "deposited_gap_mean": float(dg.mean())})
    print(f"  treewidth by interface size:")
    for n in SIZES:
        t = [r["tw"] for r in ok if r["n"] == n]
        d = [r["density"] for r in ok if r["n"] == n]
        if t:
            print(f"    n={n:2d}  treewidth median {int(np.median(t)):2d} "
                  f"(min {min(t)}, max {max(t)})   contact density {np.mean(d):.2f}")
    return out


rows_b = sweep(bound=True)
rows_u = sweep(bound=False)
sb = summarize(rows_b, "BOUND side chains")
su = summarize(rows_u, "UNBOUND side chains")

print(f"\n{'='*94}\n  PAIRED VERDICT -- the only thing that differs is bound vs unbound "
      f"side chains\n{'='*94}")
print(f"  {'arm':10s} {'instances':>10s} {'greedy missed':>14s} {'worst gap':>11s} "
      f"{'deposited already optimal':>27s}")
for tag, s in (("bound", sb), ("unbound", su)):
    if "n" in s:
        print(f"  {tag:10s} {s['n']:10d} {s['greedy_miss']:14d} "
              f"{s['greedy_worst']:11.4f} "
              f"{s['deposited_already_optimal']:>18d}/{s['n']:<8d}")
print("\n  Reading: if greedy misses on the unbound arm and not the bound one, the "
      "guarantee\n  pays exactly where the side chains are actually wrong -- which is the "
      "only place\n  a docking pipeline needs it. If it misses on NEITHER, the guarantee "
      "does not bite\n  at these interface sizes and this benchmark says so.")

json.dump({"bound": rows_b, "unbound": rows_u,
           "summary": {"bound": sb, "unbound": su}},
          open("benchmarks/repack_results.json", "w"), indent=1)
print("\nwritten benchmarks/repack_results.json")
