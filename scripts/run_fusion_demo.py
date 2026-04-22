"""Run the field-only toy vesicle fusion demo.

Two small bilayer vesicles, offset along z, pulled toward each other by
a uniform axial attractor. Success = they end up as one connected
component.

Usage:
    python scripts/run_fusion_demo.py [--n-per-leaflet 80] [--steps 20000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cell_sim.atom_engine.fusion_demo import FusionConfig, run_fusion
from cell_sim.atom_engine.vesicle import VesicleSpec


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-per-leaflet", type=int, default=80)
    p.add_argument("--radius", type=float, default=2.0)
    p.add_argument("--thickness", type=float, default=0.9)
    p.add_argument("--z-offset", type=float, default=3.5)
    p.add_argument("--equilibration-steps", type=int, default=1000)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--report-every", type=int, default=500)
    p.add_argument("--attractor-strength", type=float, default=4.0,
                   help="Per-atom axial pull (kJ/mol/nm).")
    p.add_argument("--attractor-ramp-ps", type=float, default=15.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None)
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
    cfg = FusionConfig(
        vesicle=spec,
        z_offset_nm=args.z_offset,
        dt_ps=args.dt,
        target_temperature_K=args.temperature,
        equilibration_steps=args.equilibration_steps,
        production_steps=args.steps,
        report_every=args.report_every,
        attractor_strength_kj_per_nm=args.attractor_strength,
        attractor_ramp_ps=args.attractor_ramp_ps,
    )

    def progress(msg: str) -> None:
        print(f"[fusion] {msg}", flush=True)

    n_atoms = 2 * 4 * args.n_per_leaflet
    progress(f"two vesicles: {n_atoms} atoms total, "
             f"R={args.radius} nm, t={args.thickness} nm, "
             f"offset={args.z_offset} nm")
    t0 = time.time()
    state, result = run_fusion(cfg, progress=progress)
    elapsed = time.time() - t0

    progress(f"done in {elapsed:.1f} s "
             f"({cfg.production_steps / max(elapsed, 1e-6):.0f} steps/s)")
    progress(f"initial_components={result.initial_components} "
             f"final_components={result.final_components}")
    progress(f"bonds_broken={result.bonds_broken} "
             f"bonds_formed={result.bonds_formed}")
    if result.contact_time_ps is not None:
        progress(f"contact at t={result.contact_time_ps:.2f} ps")
    if result.merge_time_ps is not None:
        progress(f"single-component at t={result.merge_time_ps:.2f} ps")
    if result.completed_fusion:
        progress("FUSION COMPLETED (two vesicles -> one component)")
    else:
        progress("fusion did NOT complete")

    summary = {
        "n_atoms": result.n_atoms,
        "initial_components": result.initial_components,
        "final_components": result.final_components,
        "contact_time_ps": result.contact_time_ps,
        "merge_time_ps": result.merge_time_ps,
        "completed_fusion": result.completed_fusion,
        "bonds_broken": result.bonds_broken,
        "bonds_formed": result.bonds_formed,
        "elapsed_s": elapsed,
        "steps_per_s": cfg.production_steps / max(elapsed, 1e-6),
        "trajectory": {
            "t_ps": result.t_ps,
            "temperature_K": result.temperature_K,
            "n_components": result.n_components,
            "com_separation_nm": result.com_separation_nm,
        },
        "config": {
            "n_per_leaflet": args.n_per_leaflet,
            "radius_nm": args.radius,
            "thickness_nm": args.thickness,
            "z_offset_nm": args.z_offset,
            "dt_ps": args.dt,
            "equilibration_steps": args.equilibration_steps,
            "production_steps": args.steps,
            "attractor_strength": args.attractor_strength,
            "attractor_ramp_ps": args.attractor_ramp_ps,
            "temperature_K": args.temperature,
            "seed": args.seed,
        },
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        progress(f"summary written to {out_path}")

    return 0 if result.completed_fusion else 1


if __name__ == "__main__":
    sys.exit(main())
