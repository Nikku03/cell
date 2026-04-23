"""Water + peptide demos using the upgraded physics.

Runs a small TIP3P-ish water box, then a single glycine zwitterion in
water, both at 300 K with Coulomb + Langevin + angle constraints.
Reports bond-length / bond-angle drift vs equilibrium values.

Usage::

    python scripts/run_water_peptide.py --demo water
    python scripts/run_water_peptide.py --demo glycine
    python scripts/run_water_peptide.py --demo both   # default
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cell_sim.atom_engine.water_peptide_demo import (
    GlycineInWaterConfig,
    WaterBoxConfig,
    run_glycine_in_water,
    run_water_box,
)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", choices=("water", "glycine", "both"),
                   default="both")
    p.add_argument("--n-water", type=int, default=60)
    p.add_argument("--n-glycine", type=int, default=1)
    p.add_argument("--water-steps", type=int, default=4000)
    p.add_argument("--glycine-steps", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out = {}

    if args.demo in ("water", "both"):
        print("\n=== water box ===")
        water_cfg = WaterBoxConfig(
            n_water=args.n_water,
            steps=args.water_steps,
            temperature_K=args.temperature,
        )
        _, water_res = run_water_box(
            water_cfg, progress=lambda m: print(f"[water] {m}"),
        )
        print(f"[water] done in {water_res.elapsed_s:.1f} s")
        out["water"] = {
            "n_atoms": water_res.n_atoms,
            "t_ps": water_res.t_ps,
            "temperature_K": water_res.temperature_K,
            "mean_oh_nm": water_res.mean_oh_nm,
            "mean_hoh_deg": water_res.mean_hoh_deg,
            "mean_nearest_oo_nm": water_res.mean_nearest_oo_nm,
            "hbonds_per_water": water_res.hbonds_per_water,
            "elapsed_s": water_res.elapsed_s,
        }

    if args.demo in ("glycine", "both"):
        print("\n=== glycine in water ===")
        gly_cfg = GlycineInWaterConfig(
            n_water=args.n_water,
            n_glycine=args.n_glycine,
            steps=args.glycine_steps,
            temperature_K=args.temperature,
        )
        _, gly_res = run_glycine_in_water(
            gly_cfg, progress=lambda m: print(f"[gly] {m}"),
        )
        print(f"[gly] done in {gly_res.elapsed_s:.1f} s, "
              f"broken_bonds={gly_res.n_broken_bonds}")
        out["glycine"] = {
            "n_atoms": gly_res.n_atoms,
            "t_ps": gly_res.t_ps,
            "temperature_K": gly_res.temperature_K,
            "backbone_bond_nm": gly_res.n_cal_c_nm,
            "n_ca_c_deg": gly_res.ncac_deg,
            "n_broken_bonds": gly_res.n_broken_bonds,
            "elapsed_s": gly_res.elapsed_s,
        }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nsummary: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
