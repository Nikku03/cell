"""Run the environment-aware reaction demo across a temperature sweep.

Usage:
    python scripts/run_reaction_demo.py [--temperatures 300 1500 3000]

At each T, seeds an atom soup with a small H/C/N/O mix, turns on dynamic
bonding, and runs ``--steps`` MD steps. Reports bonds formed/broken,
resulting molecules, and a stability audit (valence violations,
duplicate bonds, illegal element pairs — all expected to be 0).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cell_sim.atom_engine.atom_soup import SoupSpec
from cell_sim.atom_engine.element import Element
from cell_sim.atom_engine.reaction_demo import ReactionConfig, run_reactions


DEFAULT_COMPOSITION = {
    Element.H: 40,
    Element.C: 10,
    Element.N: 4,
    Element.O: 6,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--temperatures", nargs="+", type=float,
                   default=[300.0, 1500.0, 3000.0])
    p.add_argument("--radius", type=float, default=1.2,
                   help="Confinement sphere radius (nm).")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--report-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None,
                   help="Path to write per-temperature JSON summary.")
    return p.parse_args()


def _run_one(T: float, args: argparse.Namespace, progress) -> dict:
    soup = SoupSpec(
        composition=DEFAULT_COMPOSITION,
        radius_nm=args.radius,
        temperature_K=T,
        seed=args.seed,
    )
    cfg = ReactionConfig(
        soup=soup,
        dt_ps=args.dt,
        target_temperature_K=T,
        steps=args.steps,
        report_every=args.report_every,
    )
    t0 = time.time()
    state, result = run_reactions(cfg, progress=progress)
    elapsed = time.time() - t0
    return {
        "T_target_K": T,
        "n_atoms": result.n_atoms,
        "total_bonds_formed": result.total_bonds_formed,
        "total_bonds_broken": result.total_bonds_broken,
        "net_live_bonds": result.net_live_bonds,
        "final_temperature_K": result.final_temperature_K,
        "valence_violations": result.valence_violations,
        "duplicate_bonds": result.duplicate_bonds,
        "illegal_pairs": result.illegal_pairs,
        "elapsed_s": elapsed,
        "steps_per_s": cfg.steps / max(elapsed, 1e-6),
        "trajectory": {
            "t_ps": result.t_ps,
            "temperature_K": result.temperature_K,
            "live_bonds": result.live_bonds,
            "cumulative_formed": result.cumulative_formed,
            "cumulative_broken": result.cumulative_broken,
            "molecule_count": result.molecule_count,
            "largest_molecule_size": result.largest_molecule_size,
        },
    }


def main() -> int:
    args = _parse_args()
    summaries = []
    for T in args.temperatures:
        def progress(msg: str, _T: float = T) -> None:
            print(f"[T={_T:.0f}K] {msg}", flush=True)
        print(f"\n===== T = {T:.0f} K =====", flush=True)
        s = _run_one(T, args, progress)
        summaries.append(s)
        print(f"[T={T:.0f}K] done: formed={s['total_bonds_formed']} "
              f"broken={s['total_bonds_broken']} "
              f"net_live={s['net_live_bonds']}  "
              f"audit(v,d,i)=({s['valence_violations']},"
              f"{s['duplicate_bonds']},{s['illegal_pairs']}) "
              f"{s['steps_per_s']:.0f} steps/s", flush=True)

    # Summary
    print("\n===== SUMMARY =====")
    print(f"{'T (K)':>8} {'formed':>8} {'broken':>8} {'net':>8} "
          f"{'val_v':>6} {'dup':>4} {'illeg':>6} {'steps/s':>8}")
    for s in summaries:
        print(f"{s['T_target_K']:>8.0f} {s['total_bonds_formed']:>8d} "
              f"{s['total_bonds_broken']:>8d} {s['net_live_bonds']:>8d} "
              f"{s['valence_violations']:>6d} {s['duplicate_bonds']:>4d} "
              f"{s['illegal_pairs']:>6d} {s['steps_per_s']:>8.0f}")

    all_clean = all(s['valence_violations'] == 0 and s['duplicate_bonds'] == 0
                    and s['illegal_pairs'] == 0 for s in summaries)
    print(f"\nstability audit: {'PASS' if all_clean else 'FAIL'} "
          f"across {len(summaries)} temperatures")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "config": vars(args),
            "runs": summaries,
            "audit_passed": all_clean,
        }, indent=2))
        print(f"summary written to {out_path}")

    return 0 if all_clean else 1


if __name__ == "__main__":
    sys.exit(main())
