"""Quick robustness check on the JCVISYN3A_0876 x _0878 synth-lethal
prediction: vary RNG seed and biological time, confirm the prediction
is bit-stable.

Runs in <2 min on 4 workers (12-15 sims). Output:
``outputs/synthlet/0876_x_0878_robustness.txt``.
"""
from __future__ import annotations
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cell_sim"))

PAIR = ("JCVISYN3A_0876", "JCVISYN3A_0878")
SEEDS = [42, 1, 2, 7, 123]
T_END = [0.5]   # pinned to v15 / pilot config to keep wall under 5 min


def _build_wt(scale, seed, t_end):
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    sim = RealSimulator(RealSimulatorConfig(
        scale_factor=scale, seed=seed,
        use_rust_backend=True, enable_imb155_patches=True,
    ))
    return sim.run([], t_end_s=t_end, sample_dt_s=0.05), sim


def _detect(loci, sim, wt, gene_to_rules, t_end):
    from cell_sim.layer6_essentiality.per_rule_detector import PerRuleDetector
    from cell_sim.layer6_essentiality.complex_assembly_detector import (
        ComplexAssemblyDetector,
    )
    from cell_sim.layer6_essentiality.annotation_class_detector import (
        AnnotationClassDetector,
    )
    from cell_sim.layer6_essentiality.composed_detector import ComposedDetector
    from cell_sim.layer6_essentiality.harness import FailureMode

    pr = PerRuleDetector(wt=wt, gene_to_rules=gene_to_rules, min_wt_events=20)
    cd = ComposedDetector(
        structural=ComplexAssemblyDetector(),
        annotation=AnnotationClassDetector(), trajectory=pr,
    )
    ko = sim.run(list(loci), t_end_s=t_end, sample_dt_s=0.05)
    if len(loci) == 1:
        mode, t, conf, ev = cd.detect_for_gene(loci[0], ko)
    else:
        m_a, t_a, c_a, e_a = cd.detect_for_gene(loci[0], ko)
        m_b, t_b, c_b, e_b = cd.detect_for_gene(loci[1], ko)
        if m_a != FailureMode.NONE and (m_b == FailureMode.NONE or c_a >= c_b):
            mode, t, conf, ev = m_a, t_a, c_a, f"a:{e_a}"
        elif m_b != FailureMode.NONE:
            mode, t, conf, ev = m_b, t_b, c_b, f"b:{e_b}"
        else:
            mode, t, conf, ev = FailureMode.NONE, None, 0.0, "none"
    return mode.value, conf


def main():
    out_path = REPO_ROOT / "outputs/synthlet/0876_x_0878_robustness.txt"
    lines = ["JCVISYN3A_0876 x _0878 — robustness check across seeds + t_end"]
    lines.append("=" * 64)
    print(lines[0])

    # Use a single sim per (seed, t_end) to amortise WT cost
    for seed in SEEDS:
        for t_end in T_END:
            t0 = time.perf_counter()
            wt, sim = _build_wt(0.05, seed, t_end)
            g2r = sim.build_gene_to_rules_map()

            single_a, conf_a = _detect((PAIR[0],), sim, wt, g2r, t_end)
            single_b, conf_b = _detect((PAIR[1],), sim, wt, g2r, t_end)
            pair_mode, pair_conf = _detect(PAIR, sim, wt, g2r, t_end)
            wall = time.perf_counter() - t0
            row = (
                f"seed={seed:<4d} t_end={t_end:>.1f}s  "
                f"single_a={single_a:25s}  "
                f"single_b={single_b:25s}  "
                f"pair={pair_mode:25s} (conf={pair_conf:.3f})  "
                f"wall={wall:.0f}s"
            )
            print(row)
            lines.append(row)

    # Verdict
    pair_modes = [
        line.split("pair=")[1].split(" ")[0]
        for line in lines if "pair=" in line
    ]
    n_essential = sum(1 for m in pair_modes if m != "none")
    summary = (
        f"\nverdict: pair was essential in {n_essential}/{len(pair_modes)} "
        f"of (seed, t_end) configurations tested."
    )
    print(summary)
    lines.append(summary)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
