"""Algorithm 2 measured on real interfaces: treewidth, exactness, and where greedy breaks.

Two questions the claim depends on, neither assumed:
  1. What IS the treewidth of a real interface contact graph as it grows? REM is exact in
     d^treewidth, so this is the whole cost story.
  2. Where does a competent greedy packer actually fail? A guaranteed optimum is only worth
     having where the guarantee bites, and on tiny interfaces it does not.
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

print(f"{'case':6s} {'n_res':>5s} {'edges':>6s} {'dens':>5s} {'tw':>3s} {'configs':>10s} "
      f"{'exact':>11s} {'greedy':>11s} {'gap':>9s} {'ms':>7s}")
rows = []
for cid in CASES[:12]:
    for n in SIZES:
        try:
            p = build_from_case(load_case(cid), max_residues=n, n_chi1=NCHI1, n_chi2=NCHI2)
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
                             "density": dens, "configs": ncfg, "wall": True})
                continue
            gr = p.solve_greedy(g, restarts=20)
            gap = gr["energy"] - ex["energy"]
            print(f"{cid:6s} {n:5d} {len(edges):6d} {dens:5.2f} {tw:3d} {ncfg:10.2e} "
                  f"{ex['energy']:11.4f} {gr['energy']:11.4f} {gap:+9.4f} "
                  f"{ex['seconds']*1000:7.1f}")
            rows.append({"case": cid, "n": n, "tw": tw, "edges": len(edges),
                         "density": dens, "configs": ncfg, "exact": ex["energy"],
                         "greedy": gr["energy"], "gap": gap,
                         "ms": ex["seconds"] * 1000, "wall": False})
        except Exception as e:
            print(f"{cid:6s} {n:5d}  ERROR {type(e).__name__}: {str(e)[:40]}")
ok = [r for r in rows if not r.get("wall")]
walls = [r for r in rows if r.get("wall")]
print()
print(f"solved exactly: {len(ok)}   hit the treewidth wall: {len(walls)}")
if ok:
    gaps = np.array([r["gap"] for r in ok])
    print(f"greedy gap: {(gaps > 1e-6).sum()}/{len(gaps)} instances where greedy MISSED the "
          f"optimum")
    if (gaps > 1e-6).any():
        print(f"  when it misses: mean {gaps[gaps>1e-6].mean():+.4f}, "
              f"worst {gaps.max():+.4f} kcal/mol")
    print(f"treewidth by interface size:")
    for n in SIZES:
        t = [r["tw"] for r in ok if r["n"] == n]
        d = [r["density"] for r in ok if r["n"] == n]
        if t:
            print(f"    n={n:2d}  treewidth median {int(np.median(t)):2d} "
                  f"(min {min(t)}, max {max(t)})   contact density {np.mean(d):.2f}")
json.dump(rows, open("benchmarks/repack_results.json", "w"), indent=1)
print("\nwritten benchmarks/repack_results.json")
