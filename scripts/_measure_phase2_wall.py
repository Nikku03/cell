"""Phase 2 wall-time measurement.

Same comparison shape as ``scripts/_measure_phase1_wall.py``, but
re-run after phase 2 (vectorised gex compiled-spec path; commits 9273d57
phase 2 step 2, d1de4fe phase 2 step 3, 2dad436 phase 2 step 4). The
gex_off baseline should be unchanged from phase 1 (gex-off path is
byte-identical, see ``cell_sim/tests/test_phase2_gex_off_bit_identity.py``);
gex_on should drop substantially — the python-closure cache is no
longer thrashed on every gex event.

Phase-1 baseline for reference:
  gex_off mean per run: 14.52 s
  gex_on  mean per run: 52.21 s
  slowdown ratio:       3.60x

Phase-2 success criterion (from FUTURE_WORK.md):
  slowdown ratio < 1.3x

Usage:
    python scripts/_measure_phase2_wall.py

Writes ``outputs/phase2_wall_measurement.json`` with the full per-run
breakdown. Fact JSON written separately to
``memory_bank/facts/measured/phase2_gex_compiled_wall_measurement.json``.
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


# Same 5-gene sample as the phase-1 measurement so the two facts compare
# apples-to-apples: WT + pgi + ptsG + ftsZ + 0305.
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
    print(f"  phase 1 baseline ratio: 3.60x; success criterion: <1.30x")

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
            "phase1_slowdown_ratio_baseline": 3.60,
            "phase2_target_max_slowdown_ratio": 1.30,
        },
    }
    out_path = _REPO / "outputs" / "phase2_wall_measurement.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
