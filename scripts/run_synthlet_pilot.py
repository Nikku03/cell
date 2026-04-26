"""Session 26: run the synthetic-lethality pilot screen.

Reads ``outputs/synthlet/pilot_pairs.csv`` (produced by
``scripts/synthlet_pilot_pairs.py``), runs the v15 production stack
(RealSimulator + Rust + ComposedDetector) for:

    1. every unique gene in the pair list (single-knockout reference)
    2. every (a, b) pair (joint knockout)

A pair is flagged ``is_synthetic_lethal`` when the single-A and
single-B knockouts are both NON-essential AND the joint is essential
— per the spec's definition of synthetic lethality.

Output: ``outputs/synthlet/pilot_predictions.csv`` with one row per
pair; the per-pair single columns are populated by joining against
the singles dictionary.

Hard non-negotiables:
    - Reuses the same simulator + detector configuration as the v15
      sweep (scale 0.05, t_end 0.5 s, dt 0.05 s, seed 42, Rust on,
      iMB155 patches enabled). Any deviation would invalidate the
      single-knockout reference numbers.
    - 90-minute wall-time cap (per spec). The script samples a
      timer per task and aborts cleanly if total wall exceeds the cap;
      partial results are still written.

Usage::

    python scripts/run_synthlet_pilot.py \\
        --pairs outputs/synthlet/pilot_pairs.csv \\
        --out   outputs/synthlet/pilot_predictions.csv \\
        --workers 4

For Invariant-3 verification (v15 single-knockout reproducibility on
the 20-gene sample) see ``--verify-only``: that mode runs the singles
on the 20-gene sample from ``scripts/bench_rust_vs_python.py`` and
exits before any pair work.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import random
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cell_sim"))


# ------------------------------------------------------------------
# Config (mirrors v15 production sweep)
# ------------------------------------------------------------------
CFG = {
    "scale": 0.05,
    "seed": 42,
    "t_end_s": 0.5,
    "dt_s": 0.05,
    "use_rust_backend": True,
    "enable_imb155_patches": True,
    "min_wt_events": 20,
}

WT_PICKLE = REPO_ROOT / "outputs/synthlet/.wt_v15.pkl"
WALL_CAP_S = 90 * 60.0   # 90 minutes — hard abort per spec


# ------------------------------------------------------------------
# Per-worker state
# ------------------------------------------------------------------
_worker_sim = None
_worker_detector = None
_worker_cfg: dict = {}


def _worker_init(cfg_dict: dict, wt_pickle_path: str,
                 gene_to_rules: dict | None = None) -> None:
    """Per-process setup. Same shape as run_sweep_parallel.py's worker
    init so the simulator + detector configuration is bit-identical
    to the v15 production sweep."""
    global _worker_sim, _worker_detector, _worker_cfg
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    from cell_sim.layer6_essentiality.per_rule_detector import (
        PerRuleDetector,
    )
    from cell_sim.layer6_essentiality.complex_assembly_detector import (
        ComplexAssemblyDetector,
    )
    from cell_sim.layer6_essentiality.annotation_class_detector import (
        AnnotationClassDetector,
    )
    from cell_sim.layer6_essentiality.composed_detector import (
        ComposedDetector,
    )
    _worker_cfg = cfg_dict
    _worker_sim = RealSimulator(RealSimulatorConfig(
        scale_factor=cfg_dict["scale"],
        seed=cfg_dict["seed"],
        use_rust_backend=cfg_dict.get("use_rust_backend", False),
        enable_imb155_patches=cfg_dict.get("enable_imb155_patches", False),
    ))
    with open(wt_pickle_path, "rb") as fh:
        wt = pickle.load(fh)
    pr = PerRuleDetector(
        wt=wt,
        gene_to_rules=gene_to_rules or {},
        min_wt_events=cfg_dict.get("min_wt_events", 20),
    )
    _worker_detector = ComposedDetector(
        structural=ComplexAssemblyDetector(),
        annotation=AnnotationClassDetector(),
        trajectory=pr,
    )


def _predict_one(task: dict) -> dict:
    """One unit of work. ``task`` is either:

        {"kind": "single", "locus": str}
        {"kind": "pair",   "a": str, "b": str}

    Returns a dict with the prediction + wall time + evidence.
    """
    from cell_sim.layer6_essentiality.harness import FailureMode

    t0 = time.time()
    if task["kind"] == "single":
        locus = task["locus"]
        ko = _worker_sim.run(
            [locus],
            t_end_s=_worker_cfg["t_end_s"],
            sample_dt_s=_worker_cfg["dt_s"],
        )
        mode, t_fail, conf, evidence = _worker_detector.detect_for_gene(
            locus, ko,
        )
        wall = time.time() - t0
        return {
            "kind": "single",
            "locus": locus,
            "essential": int(mode != FailureMode.NONE),
            "failure_mode": mode.value,
            "confidence": float(conf),
            "evidence": evidence,
            "wall_s": wall,
        }
    elif task["kind"] == "pair":
        a, b = task["a"], task["b"]
        ko = _worker_sim.run(
            [a, b],
            t_end_s=_worker_cfg["t_end_s"],
            sample_dt_s=_worker_cfg["dt_s"],
        )
        # Detect against each gene; OR the results. The composed
        # detector decides per-locus essentiality given a trajectory.
        mode_a, t_a, conf_a, ev_a = _worker_detector.detect_for_gene(a, ko)
        mode_b, t_b, conf_b, ev_b = _worker_detector.detect_for_gene(b, ko)
        any_essential = (
            mode_a != FailureMode.NONE or mode_b != FailureMode.NONE
        )
        if mode_a != FailureMode.NONE and mode_b != FailureMode.NONE:
            # both fired — pick higher-confidence one
            if conf_a >= conf_b:
                mode, t_fail, conf, evidence = (
                    mode_a, t_a, conf_a, f"a:{ev_a} | b:{ev_b}"
                )
            else:
                mode, t_fail, conf, evidence = (
                    mode_b, t_b, conf_b, f"a:{ev_a} | b:{ev_b}"
                )
        elif mode_a != FailureMode.NONE:
            mode, t_fail, conf, evidence = (
                mode_a, t_a, conf_a, f"a:{ev_a}"
            )
        elif mode_b != FailureMode.NONE:
            mode, t_fail, conf, evidence = (
                mode_b, t_b, conf_b, f"b:{ev_b}"
            )
        else:
            mode, t_fail, conf, evidence = (
                FailureMode.NONE, None, 0.0,
                f"none[a:{ev_a} | b:{ev_b}]",
            )
        wall = time.time() - t0
        return {
            "kind": "pair",
            "a": a, "b": b,
            "essential": int(any_essential),
            "failure_mode": mode.value,
            "confidence": float(conf),
            "evidence": evidence,
            "wall_s": wall,
        }
    else:
        raise ValueError(f"unknown task kind: {task['kind']}")


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------
def _compute_wt(cfg_dict: dict):
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    sim = RealSimulator(RealSimulatorConfig(
        scale_factor=cfg_dict["scale"], seed=cfg_dict["seed"],
        use_rust_backend=cfg_dict.get("use_rust_backend", False),
        enable_imb155_patches=cfg_dict.get(
            "enable_imb155_patches", False,
        ),
    ))
    return sim.run([], t_end_s=cfg_dict["t_end_s"],
                   sample_dt_s=cfg_dict["dt_s"])


def _build_gene_to_rules(cfg_dict: dict):
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    sim = RealSimulator(RealSimulatorConfig(
        scale_factor=cfg_dict["scale"], seed=cfg_dict["seed"],
        use_rust_backend=cfg_dict.get("use_rust_backend", False),
        enable_imb155_patches=cfg_dict.get(
            "enable_imb155_patches", False,
        ),
    ))
    return sim.build_gene_to_rules_map()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pairs", type=Path,
        default=REPO_ROOT / "outputs/synthlet/pilot_pairs.csv",
    )
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "outputs/synthlet/pilot_predictions.csv",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--verify-only", action="store_true",
        help="Only run the 20-gene Invariant-3 verification pass; "
             "skip the pilot. Output goes to "
             "outputs/synthlet/viability_check.txt.",
    )
    ap.add_argument(
        "--max-pairs", type=int, default=None,
        help="Cap the pilot at the first N pairs (for fast smoke "
             "tests). Default: process all rows in --pairs.",
    )
    args = ap.parse_args()

    cfg_dict = dict(CFG)

    print("[wt] computing wild-type baseline ...")
    t_setup = time.time()
    wt = _compute_wt(cfg_dict)
    print(f"[wt] wall: {time.time() - t_setup:.1f}s")
    WT_PICKLE.parent.mkdir(parents=True, exist_ok=True)
    with WT_PICKLE.open("wb") as fh:
        pickle.dump(wt, fh)

    print("[gene_to_rules] building map ...")
    g2r = _build_gene_to_rules(cfg_dict)
    print(f"[gene_to_rules] {len(g2r)} genes mapped")

    if args.verify_only:
        return _run_invariant3(cfg_dict, g2r, args)
    return _run_pilot(cfg_dict, g2r, args)


def _run_invariant3(cfg_dict: dict, g2r: dict, args) -> int:
    """Verify Session 26 Invariant 3:
       v15 single-knockout reproducibility on the 20-gene sample
       used in scripts/bench_rust_vs_python.py.

    The expected output is the 20-gene predictions, which the user
    can compare against the v15 sweep CSV to confirm bit-identity.
    """
    from cell_sim.layer6_essentiality.labels import load_breuer2019_labels
    labels = load_breuer2019_labels()
    loci = sorted(labels.keys())
    rng = random.Random(cfg_dict["seed"])
    rng.shuffle(loci)
    sample_genes = loci[:20]
    print(f"\n[invariant-3] reproducing v15 on 20-gene sample: "
          f"{sample_genes[:3]}...{sample_genes[-3:]}")

    tasks = [{"kind": "single", "locus": lt} for lt in sample_genes]
    rows = _execute(tasks, cfg_dict, g2r, args.workers)

    pred_csv = REPO_ROOT / (
        "outputs/predictions_parallel_s0.05_t0.5_seed42_thr0.1_w4"
        "_composed_all455_v15_round2_priors.csv"
    )
    if not pred_csv.exists():
        print(f"[warn] {pred_csv} missing; cannot compare to v15 sweep")
        ref_map = {}
    else:
        ref_df = pd.read_csv(pred_csv).set_index("locus_tag")
        ref_map = ref_df["essential"].to_dict()

    out_path = REPO_ROOT / "outputs/synthlet/viability_check.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_match = n_mismatch = 0
    lines = ["Session 26 Invariant 3 — v15 reproducibility on 20-gene sample"]
    lines.append("=" * 60)
    lines.append(f"config: {cfg_dict}")
    lines.append("")
    lines.append(f"{'locus':18s} {'this':>5s} {'v15_ref':>8s} {'mode':30s}")
    for r in rows:
        lt = r["locus"]
        this_pred = r["essential"]
        ref = ref_map.get(lt, None)
        if ref is None:
            status = "(no ref)"
        elif int(ref) == int(this_pred):
            status = "OK"
            n_match += 1
        else:
            status = "MISMATCH"
            n_mismatch += 1
        lines.append(
            f"{lt:18s} {this_pred:>5d} "
            f"{ref if ref is not None else '-':>8} {r['failure_mode']:30s} {status}"
        )
    lines.append("")
    lines.append(f"matches:  {n_match}/{len(rows)}")
    lines.append(f"mismatches: {n_mismatch}/{len(rows)}")
    lines.append(
        "INVARIANT 3: " + ("PASS" if n_mismatch == 0 else "FAIL")
    )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n{out_path.read_text()}")
    print(f"\nwrote {out_path}")
    return 0 if n_mismatch == 0 else 2


def _run_pilot(cfg_dict: dict, g2r: dict, args) -> int:
    df_pairs = pd.read_csv(args.pairs)
    if args.max_pairs:
        df_pairs = df_pairs.head(args.max_pairs)
    print(f"\n[pilot] {len(df_pairs)} pairs from {args.pairs}")

    unique_genes = sorted(set(df_pairs["locus_a"]) | set(df_pairs["locus_b"]))
    print(f"[pilot] {len(unique_genes)} unique genes for single knockouts")
    print(f"[pilot] total tasks: "
          f"{len(unique_genes)} singles + {len(df_pairs)} pairs = "
          f"{len(unique_genes) + len(df_pairs)}")

    tasks: list[dict] = []
    for lt in unique_genes:
        tasks.append({"kind": "single", "locus": lt})
    for _, row in df_pairs.iterrows():
        tasks.append({
            "kind": "pair",
            "a": row["locus_a"],
            "b": row["locus_b"],
        })

    results = _execute(tasks, cfg_dict, g2r, args.workers)

    # Build single-prediction lookup
    single_by_locus = {
        r["locus"]: r for r in results if r["kind"] == "single"
    }
    pair_results = [r for r in results if r["kind"] == "pair"]
    print(f"\n[pilot] {len(single_by_locus)} singles, {len(pair_results)} pairs done")

    # Join with pair metadata
    pred_rows = []
    pair_meta = {
        (row["locus_a"], row["locus_b"]): row
        for _, row in df_pairs.iterrows()
    }
    for r in pair_results:
        a, b = r["a"], r["b"]
        meta = pair_meta.get((a, b))
        if meta is None:
            continue
        s_a = single_by_locus.get(a)
        s_b = single_by_locus.get(b)
        if s_a is None or s_b is None:
            continue
        is_synth = bool(
            s_a["essential"] == 0
            and s_b["essential"] == 0
            and r["essential"] == 1
        )
        pred_rows.append({
            "locus_a": a, "locus_b": b,
            "category": meta["category"],
            "biological_rationale": meta["biological_rationale"],
            "single_a_essential": int(s_a["essential"]),
            "single_b_essential": int(s_b["essential"]),
            "pair_essential": int(r["essential"]),
            "is_synthetic_lethal": int(is_synth),
            "single_a_mode": s_a["failure_mode"],
            "single_b_mode": s_b["failure_mode"],
            "pair_mode": r["failure_mode"],
            "pair_confidence": r["confidence"],
            "mechanism_summary": r["evidence"],
            "pair_wall_s": r["wall_s"],
        })

    out = pd.DataFrame(pred_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n[pilot] wrote {args.out}  rows={len(out)}")

    # Per-category synthetic lethality rates
    print("\n[pilot] per-category synthetic lethality rates:")
    g = (
        out.groupby("category")
           .agg(n=("is_synthetic_lethal", "size"),
                synth=("is_synthetic_lethal", "sum"))
    )
    g["rate"] = g["synth"] / g["n"]
    print(g.to_string())

    # Halt-criteria check
    rate_b = float(g.loc["B_same_pathway", "rate"]) if "B_same_pathway" in g.index else 0.0
    rate_c = float(g.loc["C_random", "rate"]) if "C_random" in g.index else 0.0
    if rate_b > 0.15:
        print(f"\n[HALT] Category B same-pathway rate {rate_b:.1%} > 15% — methodology broken")
        return 3
    if rate_c > 0.10:
        print(f"\n[HALT] Category C random rate {rate_c:.1%} > 10% — false positive rate too high")
        return 4
    return 0


def _execute(tasks: list[dict], cfg_dict: dict, g2r: dict,
             workers: int) -> list[dict]:
    """Run a list of tasks across worker pool. Aborts cleanly on the
    90-min wall cap. Returns the result list (in completion order)."""
    ctx = mp.get_context("spawn")
    t_start = time.time()
    results: list[dict] = []
    with ctx.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(cfg_dict, str(WT_PICKLE), g2r),
    ) as pool:
        n = len(tasks)
        for i, r in enumerate(pool.imap_unordered(_predict_one, tasks), 1):
            results.append(r)
            elapsed = time.time() - t_start
            if r["kind"] == "single":
                print(f"  [{i:4d}/{n}] single {r['locus']:18s} "
                      f"ess={r['essential']} mode={r['failure_mode']:25s} "
                      f"wall={r['wall_s']:.1f}s  total={elapsed:.0f}s")
            else:
                print(f"  [{i:4d}/{n}] pair   {r['a']:18s} {r['b']:18s} "
                      f"ess={r['essential']} mode={r['failure_mode']:25s} "
                      f"wall={r['wall_s']:.1f}s  total={elapsed:.0f}s")
            if elapsed > WALL_CAP_S:
                print(f"\n[ABORT] wall {elapsed:.0f}s > cap {WALL_CAP_S:.0f}s; partial results")
                pool.terminate()
                pool.join()
                break
    print(f"\n[execute] total wall: {time.time() - t_start:.0f}s "
          f"({len(results)}/{len(tasks)} tasks done)")
    return results


if __name__ == "__main__":
    raise SystemExit(main())
