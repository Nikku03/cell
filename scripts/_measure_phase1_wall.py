"""Phase 1 wall-time measurement.

Runs the production simulator on a 5-gene sample with the gex flag off
then on, and reports the per-knockout wall and event counts. Not a unit
test (would be too slow for the test suite); a one-off measurement that
gets recorded as a fact JSON in the same commit.

Usage:
    python scripts/_measure_phase1_wall.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "cell_sim"))

from cell_sim.layer6_essentiality.real_simulator import (  # noqa: E402
    RealSimulator, RealSimulatorConfig,
)


# 5-gene sample: pgi (essential), ptsG (essential), ftsZ (nonessential
# at 0.5 s), JCVISYN3A_0305 (nonessential), and a wild-type baseline
# included as the empty knockout.
SAMPLE = [
    (),                        # WT baseline
    ("JCVISYN3A_0445",),       # pgi
    ("JCVISYN3A_0779",),       # ptsG
    ("JCVISYN3A_0522",),       # ftsZ
    ("JCVISYN3A_0305",),       # uncharacterized metallopeptidase
]


def time_run(sim, knockout, t_end_s=0.5, dt_s=0.05):
    t0 = time.time()
    traj = sim.run(list(knockout), t_end_s=t_end_s, sample_dt_s=dt_s)
    wall = time.time() - t0
    n_samples = len(traj.samples)
    return wall, n_samples


def main():
    results = {"gex_off": [], "gex_on": []}

    for label, cfg in [
        ("gex_off", RealSimulatorConfig(
            scale_factor=0.05, seed=42, use_rust_backend=False,
            enable_gene_expression=False,
        )),
        ("gex_on", RealSimulatorConfig(
            scale_factor=0.05, seed=42, use_rust_backend=False,
            enable_gene_expression=True,
        )),
    ]:
        print(f"\n=== {label} ===")
        sim = RealSimulator(cfg)
        # Warm setup once so we're measuring the per-run cost, not the
        # one-time SBML parse.
        t0 = time.time()
        sim._ensure_setup()
        setup_wall = time.time() - t0
        print(f"  setup wall: {setup_wall:.2f} s")
        for ko in SAMPLE:
            wall, n_samples = time_run(sim, ko)
            label_ko = "WT" if not ko else ko[0]
            print(f"  {label_ko:18s}  wall={wall:6.2f} s  samples={n_samples}")
            results[label].append({
                "knockout": list(ko),
                "wall_s": wall,
                "n_samples": n_samples,
            })
        results[f"{label}_setup_wall_s"] = setup_wall

    off_walls = [r["wall_s"] for r in results["gex_off"]]
    on_walls = [r["wall_s"] for r in results["gex_on"]]
    off_total = sum(off_walls)
    on_total = sum(on_walls)
    ratio = on_total / off_total if off_total > 0 else float("nan")
    print("\n=== summary ===")
    print(f"  gex_off total: {off_total:.2f} s ({off_total / len(off_walls):.2f} s/run)")
    print(f"  gex_on  total: {on_total:.2f} s ({on_total / len(on_walls):.2f} s/run)")
    print(f"  slowdown ratio: {ratio:.2f}x")

    out = {
        "config": {
            "scale_factor": 0.05,
            "t_end_s": 0.5,
            "dt_s": 0.05,
            "seed": 42,
            "use_rust_backend": False,
            "n_samples_per_knockout": 5,
        },
        "per_run": results,
        "summary": {
            "gex_off_total_wall_s": off_total,
            "gex_on_total_wall_s": on_total,
            "slowdown_ratio": ratio,
            "gex_off_mean_per_run_s": off_total / len(off_walls),
            "gex_on_mean_per_run_s": on_total / len(on_walls),
        },
    }
    out_path = _REPO / "outputs" / "phase1_wall_measurement.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
