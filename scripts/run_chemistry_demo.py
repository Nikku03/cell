"""Run the pre-seeded-molecule chemistry demo.

Examples:
    # combustion-ish: H2 + O2 at 3000 K
    python scripts/run_chemistry_demo.py --mix H2=80,O2=40 --temperature 3000

    # Haber-ish: N2 + H2 at 3500 K
    python scripts/run_chemistry_demo.py --mix N2=60,H2=180 --temperature 3500

    # Biology starter soup: H2O + CH4 + NH3 + CO2 at 2500 K
    python scripts/run_chemistry_demo.py --mix H2O=50,CH4=30,NH3=30,CO2=20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cell_sim.atom_engine.chemistry_demo import ChemistryConfig, run_chemistry


def _parse_mix(s: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        name, _, n = token.partition("=")
        out[name.strip()] = int(n)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mix", type=str, default="H2=80,O2=40",
                   help="Comma-separated 'formula=count' (e.g. 'H2=80,O2=40').")
    p.add_argument("--radius", type=float, default=3.0)
    p.add_argument("--temperature", type=float, default=3000.0)
    p.add_argument("--dt", type=float, default=0.0005)
    p.add_argument("--equilibration-steps", type=int, default=500)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--report-every", type=int, default=2000)
    p.add_argument("--bond-k", type=float, default=5.0e4,
                   help="Spring constant for initial and reformed bonds "
                        "(kJ/mol/nm^2). Default 5e4 gives soft toy "
                        "chemistry; 3e5 is realistic covalent stiffness.")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = ChemistryConfig(
        composition=_parse_mix(args.mix),
        radius_nm=args.radius,
        target_temperature_K=args.temperature,
        dt_ps=args.dt,
        equilibration_steps=args.equilibration_steps,
        steps=args.steps,
        report_every=args.report_every,
        bond_form_k_kj_per_nm2=args.bond_k,
        initial_bond_k_kj_per_nm2=args.bond_k,
    )

    def progress(msg: str) -> None:
        print(f"[chem] {msg}", flush=True)

    t0 = time.time()
    state, result = run_chemistry(cfg, progress=progress)
    elapsed = time.time() - t0

    progress(f"done in {elapsed:.1f} s "
             f"({cfg.steps / max(elapsed, 1e-6):.1f} steps/s)")
    progress(f"initial formulas: {result.initial_formulas}")
    progress(f"final   formulas: {result.final_formulas}")

    # Difference (what was produced / consumed)
    all_keys = set(result.initial_formulas) | set(result.final_formulas)
    diff = {k: result.final_formulas.get(k, 0) - result.initial_formulas.get(k, 0)
            for k in all_keys}
    changes = {k: v for k, v in diff.items() if v != 0}
    if changes:
        progress("net population changes (final - initial):")
        for k in sorted(changes, key=lambda x: -abs(changes[x])):
            sign = "+" if changes[k] > 0 else ""
            progress(f"  {k:<12s}: {sign}{changes[k]}")
    else:
        progress("no net population changes (reactions balanced)")

    progress(f"bond events: formed={result.total_bonds_formed} "
             f"broken={result.total_bonds_broken}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "config": vars(args),
            "n_atoms": result.n_atoms,
            "n_bonds_initial": result.n_bonds_initial,
            "initial_formulas": result.initial_formulas,
            "final_formulas": result.final_formulas,
            "diff": changes,
            "total_bonds_formed": result.total_bonds_formed,
            "total_bonds_broken": result.total_bonds_broken,
            "elapsed_s": elapsed,
            "steps_per_s": cfg.steps / max(elapsed, 1e-6),
            "trajectory": {
                "t_ps": result.t_ps,
                "temperature_K": result.temperature_K,
                "formula_snapshots": result.formula_snapshots,
            },
        }, indent=2))
        progress(f"summary written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
