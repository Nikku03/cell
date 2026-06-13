"""TIER 1 master orchestrator.

Runs every Tier 1 step in order. Each step accepts --smoke (sandbox-safe)
and --real (Colab-required). Prints a final pass/fail matrix.

  1.1a  LOCO evaluation                 sandbox: smoke OK  real: needs Colab+Drive
  1.1b  per-org product output          sandbox: smoke OK  real: needs Drive predictions
  1.1c  calibration                     uses existing calibrate_and_threshold.py
  1.2   OGEE external validation        sandbox: smoke OK  real: needs OGEE on Drive
  1.3   STRING degree kill-gate         sandbox: smoke OK  real: needs STRING on Drive
  1.4   Bacitracin writeup              sandbox: doc check only
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STEPS = [
    ("1.1a", "tier1_loco.py",          "LOCO evaluation"),
    ("1.1b", "tier1_per_org_output.py","per-organism product"),
    ("1.2",  "tier1_ogee_validate.py", "OGEE external validation"),
    ("1.3",  "tier1_string_killgate.py","STRING degree kill-gate"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run every step in smoke mode (sandbox-safe)")
    ap.add_argument("--real", action="store_true",
                    help="run every step in real mode (needs Colab+Drive)")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True

    results = []
    for sid, script, label in STEPS:
        print(f"\n{'#'*70}\n# TIER {sid}  {label}\n{'#'*70}")
        cmd = ["python", str(REPO / "scripts" / script)]
        if args.smoke: cmd.append("--smoke")
        proc = subprocess.run(cmd)
        results.append((sid, label, proc.returncode == 0))

    # 1.4 writeup check
    bdoc = REPO / "outputs" / "tier1_bacitracin_lead.md"
    bok = bdoc.exists() and bdoc.stat().st_size > 4000
    results.append(("1.4", "bacitracin writeup", bok))

    # 1.1c calibration check
    cscript = REPO / "scripts" / "calibrate_and_threshold.py"
    cok = cscript.exists()
    results.append(("1.1c", "calibration available", cok))

    print(f"\n{'='*70}\nTIER 1 SUMMARY\n{'='*70}")
    for sid, label, ok in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}]  Tier {sid:<5s}  {label}")
    fails = sum(1 for _, _, ok in results if not ok)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
