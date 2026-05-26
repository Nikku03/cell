"""Conservation-violation test: does our physics-constrained model refuse
violations that the upstream simulator commits?

This is the first concrete test of the "sense-1 better than the simulator"
claim:  the simulator is not ground truth — physics is.  Where the simulator
commits a conservation-law violation, a properly physics-constrained
surrogate should produce the admissible trajectory instead.

Methodology:
  1. Load the 50 upstream parquet trajectories.
  2. Compute the adenylate pool (ATP + ADP + AMP) over time per trajectory.
  3. Scan 30-step windows.  Flag any window where the pool changes by more
     than VIOLATION_THRESHOLD (default 20%).  This is well above the ~5%
     real fast-growth biology can produce in 30 seconds.
  4. For each flagged window, snapshot the state at the START of the window
     and run our trained model forward 30 steps.
  5. Compare the predicted change-in-pool vs upstream's change-in-pool.
  6. Report aggregate: how many violations did we shrink, by how much.

Usage (Colab cell or local):

    !python test_conservation_violations.py \\
        --parquet-dir /content/drive/MyDrive \\
        --checkpoint /content/drive/MyDrive/cell_emulator_v13.pt

Outputs a per-violation table + aggregate summary + a JSON of all
violations for downstream analysis.
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from pathlib import Path
import numpy as np

# Make the parent module importable for re-using parsers / model classes
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch                                             # noqa: E402
import colab_cell_emulator as cce                        # noqa: E402

ADENYLATE_SPECIES        = ("M_atp_c", "M_adp_c", "M_amp_c")
VIOLATION_THRESHOLD_REL  = 0.20      # 20% pool change in WINDOW_SECONDS = violation
WINDOW_SECONDS           = 30        # detect violations over 30 s biological
TIME_STRIDE              = 30        # model's step in seconds — must match
                                     # colab_cell_emulator.TIME_STRIDE used at training
WINDOW_MODEL_STEPS       = max(1, WINDOW_SECONDS // TIME_STRIDE)
MIN_POOL                 = 1.0       # avoid divide-by-zero on empty pools


def load_parquets(parquet_dir):
    """Yield (path, dataframe) for each parquet trajectory."""
    try:
        import pandas as pd
    except ImportError:
        sys.exit("[err] pandas not available — install pyarrow + pandas")
    # Recursive search — Drive layouts often nest the parquets under
    # MyDrive/<project>/<run>/counts_and_fluxes_*.parquet
    paths = sorted(glob.glob(os.path.join(parquet_dir, "**",
                                          "counts_and_fluxes*.parquet"),
                              recursive=True))
    if not paths:
        sys.exit(f"[err] no counts_and_fluxes*.parquet under {parquet_dir!r} "
                 f"(searched recursively).  Try a more specific --parquet-dir.")
    print(f"[load] {len(paths)} parquet trajectories under {parquet_dir}")
    for p in paths:
        try:
            yield p, pd.read_parquet(p)
        except Exception as e:
            print(f"[warn] couldn't read {p}: {e}")


def find_violations(traj_df, traj_path):
    """Scan one trajectory's adenylate pool for windows exceeding the threshold.

    Parquet layout: rows = species (df.index), columns = time in seconds
    (as float-string labels '0.0', '1.0', ...).  Access via .loc[species].iloc[t].

    We scan only at TIME_STRIDE positions so the snapshot states match the
    distribution the model was trained on (stride=30 → t=0, 30, 60, ...).
    """
    missing = [s for s in ADENYLATE_SPECIES if s not in traj_df.index]
    if missing:
        print(f"[warn] {os.path.basename(traj_path)}: missing {missing}; skipping")
        return []
    arr = np.stack([traj_df.loc[s].values.astype(np.float64)
                    for s in ADENYLATE_SPECIES], axis=0)   # (3, T)
    pool = arr.sum(axis=0)                                 # (T,)
    T = pool.size
    out = []
    for t0 in range(0, T - WINDOW_SECONDS, TIME_STRIDE):
        t1 = t0 + WINDOW_SECONDS
        before = pool[t0]
        after  = pool[t1]
        denom = max(before, MIN_POOL)
        rel = (after - before) / denom
        if abs(rel) > VIOLATION_THRESHOLD_REL:
            out.append({
                "path": traj_path,
                "t0":   int(t0),
                "t1":   int(t1),
                "pool_t0":      float(before),
                "pool_t1":      float(after),
                "rel_change":   float(abs(rel)),
                "signed_change": float(rel),
            })
    return out


def build_model_from_checkpoint(ckpt_path):
    """Reconstruct the trained model from cell_emulator_v13.pt.

    NB: this assumes the checkpoint shape from v13.9 / v14.  If you've
    tweaked DynamicsModel since training, fields may need adjusting.
    """
    print(f"[model] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ckpt.get("config", {})
    species_active = ckpt["species_active"]
    lo, span = ckpt["lo"], ckpt["span"]
    S = len(species_active)

    # The model wants edge_index/edge_weight + species_type_ids;
    # everything else can be defaults that don't matter for forward pass.
    edge_index  = ckpt["edge_index"]
    edge_weight = ckpt["edge_weight"]
    species_type_ids = ckpt["species_type_ids"]

    model = cce.DynamicsModel(
        S=S, hidden=cfg.get("hidden", 32),
        n_layers=cfg.get("n_layers", 3),
        species_type_ids=species_type_ids,
        edge_index=edge_index, edge_weight=edge_weight,
        cfc_tau_min=cfg.get("cfc_tau_min", 0.1),
        n_type_embed=cfg.get("n_type_embed", 4),
        lo_norm=lo, span_norm=span,
        use_stochastic=cfg.get("use_stochastic", True),
    )
    state = ckpt["model"]
    # Strip torch.compile wrapper prefix
    state = {k[10:] if k.startswith("_orig_mod.") else k: v
             for k, v in state.items()}
    # Skip buffers — graph and SBML masks come from current setup
    buf_names = set(dict(model.named_buffers()).keys())
    state = {k: v for k, v in state.items() if k not in buf_names}
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        param_names = set(dict(model.named_parameters()).keys())
        missing_params = [k for k in result.missing_keys if k in param_names]
        if missing_params:
            print(f"[model] WARNING: {len(missing_params)} parameter keys missing — "
                   "predictions may be off")
    model.eval()
    name_to_idx = {n: i for i, n in enumerate(species_active)}
    return model, name_to_idx, lo, span


def roll_model(model, state_norm, n_steps):
    """Roll the model forward n_steps from a normalised state.  Returns (T, S)."""
    out = [state_norm.clone()]
    s = state_norm.clone()
    with torch.no_grad():
        for _ in range(n_steps):
            o = model(s)
            s = o[0] if isinstance(o, tuple) else o
            s = s.clamp(cce.CLAMP_LO, cce.CLAMP_HI)
            out.append(s.clone())
    return torch.stack(out, dim=0)


def evaluate_violation(violation, traj_df, model, name_to_idx, lo, span):
    """For one violation window, snapshot state at t0, roll the model, compare."""
    sp_idx = [name_to_idx.get(s) for s in ADENYLATE_SPECIES]
    if any(i is None for i in sp_idx):
        return None
    # Snapshot state at t0 — parquet rows are species, columns are time
    species_active = list(name_to_idx.keys())
    counts_t0 = np.zeros(len(species_active), dtype=np.float32)
    t0 = violation["t0"]
    in_index = set(traj_df.index)
    for nm, i in name_to_idx.items():
        if nm in in_index:
            counts_t0[i] = float(traj_df.loc[nm].iloc[t0])
    # signed_log + normalise
    sl  = np.sign(counts_t0) * np.log1p(np.abs(counts_t0))
    nrm = np.clip((sl - lo) / span, cce.CLAMP_LO, cce.CLAMP_HI).astype(np.float32)
    state_norm = torch.from_numpy(nrm).unsqueeze(0)  # (1, S)
    # Roll WINDOW_MODEL_STEPS — one model step at stride=30 = 30 s biological
    rolled = roll_model(model, state_norm, WINDOW_MODEL_STEPS)
    final_norm = rolled[-1, 0].numpy()
    # Denormalise → counts
    final_sl    = final_norm * span + lo
    final_counts = np.sign(final_sl) * np.expm1(np.abs(final_sl))
    final_counts = np.maximum(final_counts, 0)
    pred_pool   = float(sum(final_counts[i] for i in sp_idx))
    # Compare
    upstream_signed = violation["signed_change"]
    upstream_abs    = abs(violation["pool_t1"] - violation["pool_t0"])
    model_signed    = (pred_pool - violation["pool_t0"]) / max(violation["pool_t0"], MIN_POOL)
    model_abs       = abs(pred_pool - violation["pool_t0"])
    return {
        "upstream_signed_pct": upstream_signed * 100,
        "upstream_abs":        upstream_abs,
        "model_signed_pct":    model_signed * 100,
        "model_abs":           model_abs,
        "shrink_factor":       1.0 - (model_abs / max(upstream_abs, 1e-9)),
        "pool_t0":             violation["pool_t0"],
        "pool_t1_upstream":    violation["pool_t1"],
        "pool_t1_model":       pred_pool,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", required=True,
                    help="directory containing counts_and_fluxes*.parquet")
    ap.add_argument("--checkpoint", required=True,
                    help="path to cell_emulator_v13.pt")
    ap.add_argument("--max-violations", type=int, default=200,
                    help="cap evaluated violations (the test is O(N) parquet IO)")
    ap.add_argument("--out", default="/tmp/conservation_test.json",
                    help="output JSON path for the full results")
    args = ap.parse_args()

    # Phase 1: scan upstream for violations
    all_violations = []
    for path, df in load_parquets(args.parquet_dir):
        v = find_violations(df, path)
        all_violations.extend(v)
        if v:
            print(f"  {os.path.basename(path):40s}  found {len(v)} violation windows")

    if not all_violations:
        print("\n[result] no adenylate-pool violations found at threshold "
              f"{VIOLATION_THRESHOLD_REL*100:.0f}% over {WINDOW_SECONDS}s windows "
              f"(scanned at TIME_STRIDE={TIME_STRIDE}s steps)")
        print("  → upstream is well-behaved on this metric; try a tighter threshold")
        print("    or a different conservation law (charge balance, element drift)")
        return

    all_violations.sort(key=lambda v: -v["rel_change"])
    cap = min(len(all_violations), args.max_violations)
    print(f"\n[scan] {len(all_violations)} total violation windows, "
          f"evaluating top {cap} by magnitude")

    # Phase 2: load model and evaluate
    model, name_to_idx, lo, span = build_model_from_checkpoint(args.checkpoint)

    import pandas as pd                  # already imported in load_parquets
    cur_path, cur_df = None, None
    results = []
    for v in all_violations[:cap]:
        if v["path"] != cur_path:
            cur_path, cur_df = v["path"], pd.read_parquet(v["path"])
        r = evaluate_violation(v, cur_df, model, name_to_idx, lo, span)
        if r is None:
            continue
        results.append(r)

    # Phase 3: aggregate + print
    if not results:
        print("[result] all violations failed to evaluate (likely species-name mismatch)")
        return

    shrinks = np.array([r["shrink_factor"] for r in results])
    n_fixed_50 = int((shrinks > 0.5).sum())
    n_fixed_any = int((shrinks > 0.0).sum())

    print()
    print("=" * 72)
    print("  CONSERVATION VIOLATION TEST — ADENYLATE POOL")
    print("=" * 72)
    print(f"  violations evaluated   : {len(results)}")
    print(f"  shrunk at all (>0%)    : {n_fixed_any} ({n_fixed_any/len(results)*100:.0f}%)")
    print(f"  fixed (>50% reduction) : {n_fixed_50} ({n_fixed_50/len(results)*100:.0f}%)")
    print(f"  median shrink factor   : {float(np.median(shrinks))*100:+.1f}%")
    print(f"  mean   shrink factor   : {float(shrinks.mean())*100:+.1f}%")
    print(f"  worst case (model bad) : {float(shrinks.min())*100:+.1f}%")
    print(f"  best case  (model good): {float(shrinks.max())*100:+.1f}%")
    print()
    print("  per-violation (top 10 by upstream magnitude):")
    print(f"  {'upstream Δ':>12s}  {'model Δ':>12s}  {'shrink':>8s}")
    for r in results[:10]:
        print(f"  {r['upstream_signed_pct']:>+11.1f}%  "
              f"{r['model_signed_pct']:>+11.1f}%  "
              f"{r['shrink_factor']*100:>+7.1f}%")
    print("=" * 72)

    if shrinks.mean() > 0:
        print("\n→ on average, the constrained surrogate REDUCES the violation.")
        print("  this is the sense-1 'better than the simulator' claim's first data point.")
    else:
        print("\n→ on average, the constrained surrogate REPRODUCES the violation.")
        print("  either the constraint weights are too soft, or the model has memorised the")
        print("  violations.  Try increasing LAMBDA_ATP and retraining, or look at element")
        print("  balance / charge balance as alternative conservation tests.")

    with open(args.out, "w") as f:
        json.dump({
            "config": {
                "violation_threshold_rel": VIOLATION_THRESHOLD_REL,
                "window_seconds":          WINDOW_SECONDS,
                "time_stride":             TIME_STRIDE,
                "window_model_steps":      WINDOW_MODEL_STEPS,
                "adenylate_species":       list(ADENYLATE_SPECIES),
                "max_violations_evaluated": cap,
            },
            "summary": {
                "n_violations_found":  len(all_violations),
                "n_evaluated":         len(results),
                "n_fixed_50pct":       n_fixed_50,
                "n_shrunk":            n_fixed_any,
                "median_shrink_pct":   float(np.median(shrinks)) * 100,
                "mean_shrink_pct":     float(shrinks.mean()) * 100,
            },
            "per_violation": results,
        }, f, indent=2)
    print(f"\n[save] full results -> {args.out}")


if __name__ == "__main__":
    main()
