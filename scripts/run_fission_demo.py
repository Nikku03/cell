"""Run the toy vesicle fission demo end-to-end.

Usage:
    python scripts/run_fission_demo.py [--n-per-leaflet 200] [--steps 20000]

Prints step-by-step progress and writes a summary JSON next to the
trajectory log if --out is given.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cell_sim.atom_engine.fission_demo import FissionConfig, run_fission
from cell_sim.atom_engine.vesicle import VesicleSpec


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-per-leaflet", type=int, default=180,
                   help="Lipids per leaflet (total atoms = 4 * this).")
    p.add_argument("--radius", type=float, default=3.0,
                   help="Vesicle outer radius (nm).")
    p.add_argument("--thickness", type=float, default=1.0,
                   help="Bilayer head-to-head thickness (nm).")
    p.add_argument("--equilibration-steps", type=int, default=1000)
    p.add_argument("--steps", type=int, default=20_000,
                   help="Production steps.")
    p.add_argument("--dt", type=float, default=0.005, help="Timestep (ps).")
    p.add_argument("--report-every", type=int, default=500)
    p.add_argument("--constriction-k", type=float, default=800.0,
                   help="Radial constriction spring (kJ/mol/nm^2).")
    p.add_argument("--constriction-width", type=float, default=0.8,
                   help="Gaussian width of the constriction along the axis (nm).")
    p.add_argument("--constriction-ramp-ps", type=float, default=10.0,
                   help="Time over which the constriction ramps from 0 to full.")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None,
                   help="Where to write a summary JSON (optional).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    spec = VesicleSpec(
        n_per_leaflet=args.n_per_leaflet,
        radius_nm=args.radius,
        bilayer_thickness_nm=args.thickness,
        temperature_K=args.temperature,
        seed=args.seed,
    )
    cfg = FissionConfig(
        vesicle=spec,
        dt_ps=args.dt,
        target_temperature_K=args.temperature,
        equilibration_steps=args.equilibration_steps,
        production_steps=args.steps,
        report_every=args.report_every,
        constriction_k_kj_per_nm2=args.constriction_k,
        constriction_width_nm=args.constriction_width,
        constriction_ramp_ps=args.constriction_ramp_ps,
    )

    def progress(msg: str) -> None:
        print(f"[fission] {msg}", flush=True)

    n_atoms = 4 * args.n_per_leaflet
    progress(f"vesicle: {n_atoms} atoms, R={args.radius} nm, t={args.thickness} nm")
    t0 = time.time()
    state, result = run_fission(cfg, progress=progress)
    elapsed = time.time() - t0

    progress(f"done in {elapsed:.1f} s "
             f"({cfg.production_steps / max(elapsed, 1e-6):.0f} steps/s)")
    progress(f"initial_components={result.initial_components} "
             f"final_components={result.final_components}")
    progress(f"bonds_broken={result.bonds_broken} "
             f"bonds_formed={result.bonds_formed}")
    if result.pinch_time_ps is not None:
        progress(f"neck closed at t={result.pinch_time_ps:.2f} ps")
    if result.bimodal_time_ps is not None:
        progress(f"bimodal hemispheres appeared at t={result.bimodal_time_ps:.2f} ps")
    if result.completed_fission:
        progress("FISSION COMPLETED (bimodal + balanced)")
    else:
        progress("fission did NOT complete (no clean bimodal split)")

    summary = {
        "n_atoms": result.n_atoms,
        "n_bonds_initial": result.n_bonds_initial,
        "initial_components": result.initial_components,
        "final_components": result.final_components,
        "pinch_time_ps": result.pinch_time_ps,
        "bimodal_time_ps": result.bimodal_time_ps,
        "completed_fission": result.completed_fission,
        "bonds_broken": result.bonds_broken,
        "bonds_formed": result.bonds_formed,
        "elapsed_s": elapsed,
        "steps_per_s": cfg.production_steps / max(elapsed, 1e-6),
        "trajectory": {
            "t_ps": result.t_ps,
            "temperature_K": result.temperature_K,
            "neck_fraction": result.neck_fraction,
            "n_components": result.n_components,
            "hemisphere_imbalance": result.hemisphere_imbalance,
            "bimodal": result.bimodal,
        },
        "config": {
            "n_per_leaflet": args.n_per_leaflet,
            "radius_nm": args.radius,
            "thickness_nm": args.thickness,
            "dt_ps": args.dt,
            "equilibration_steps": args.equilibration_steps,
            "production_steps": args.steps,
            "constriction_k": args.constriction_k,
            "constriction_width": args.constriction_width,
            "constriction_ramp_ps": args.constriction_ramp_ps,
            "temperature_K": args.temperature,
            "seed": args.seed,
        },
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        progress(f"summary written to {out_path}")

    return 0 if result.completed_fission else 1


if __name__ == "__main__":
    sys.exit(main())
