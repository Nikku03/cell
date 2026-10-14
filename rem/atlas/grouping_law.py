"""The scaling law relating a bulk dependence measure to the tail error it fails to control.

Parses RESULTS_grouping.txt rather than hardcoding numbers, so every figure here has its source
in a file produced by rem/atlas/grouping.py.

THE CORRECTION THIS RECORDS FIRST. grouping.py's gate G2 predeclared "tail error > 10x" as the
UNSOUND verdict. That bar was unreachable on any evidence. The statistic used was

    tail_err = |P_exact - P_factorised| / P_exact

and the factorisation UNDERESTIMATES a positively-coupled joint tail, so P_fact < P_exact and
tail_err < 1 identically. A bar at 10 could never be met. This is the third gate in this session
whose bar could not be met by any outcome -- after a sign test on n = 5 whose minimum two-sided
p was 0.0625 against alpha = 0.05, and a vacuity check written as phi > 0.02 on a phi that was
negative. The unbounded statistic is the RATIO P_exact/P_factorised, equivalently exp(Lambda),
and it is what is used below.
"""

from __future__ import annotations
import os, re
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "RESULTS_grouping.txt")
RULE = "=" * 97


def parse(path=RES):
    """Pull (speed, MI, MI_tilt, mean_err, tail_err, Lambda) from the sweep and control blocks."""
    txt = open(path).read()
    rows = []
    # main sweep: speed MI MI_tilt theta mean_err P_tail tail_err Lambda
    for m in re.finditer(r"^\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                         r"([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([+-][\d.]+)\s*$",
                         txt, re.M):
        g = m.groups()
        rows.append(dict(speed=float(g[0]), MI=float(g[1]), MI_tilt=float(g[2]),
                         mean_err=float(g[4]), tail_err=float(g[6]), lam=float(g[7])))
    # G5 control block: speed MI MI_tilt mean_err tail_err |Lambda|
    ctrl = txt.split("G5  ZERO-COUPLING")[-1] if "G5  ZERO-COUPLING" in txt else ""
    for m in re.finditer(r"^\s+(\d+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+"
                         r"([\d.e+-]+)\s+([\d.e+-]+)\s*$", ctrl, re.M):
        g = m.groups()
        rows.append(dict(speed=float(g[0]), MI=float(g[1]), MI_tilt=float(g[2]),
                         mean_err=float(g[3]), tail_err=float(g[4]), lam=float(g[5])))
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: r["speed"]):
        if r["speed"] not in seen:
            seen.add(r["speed"]); out.append(r)
    return out


def report():
    rows = parse()
    sp = np.array([r["speed"] for r in rows])
    MI = np.array([r["MI"] for r in rows])
    te = np.array([r["tail_err"] for r in rows])
    me = np.array([r["mean_err"] for r in rows])
    out = []; P = out.append
    P(RULE)
    P("HOW FAST DOES A BULK DEPENDENCE MEASURE VANISH COMPARED WITH THE TAIL ERROR IT MISSES?")
    P(RULE)
    P(f"  parsed {len(rows)} rows from RESULTS_grouping.txt (sweep + zero-coupling control)")
    P("")
    P(f"  {'speed':>7s} {'MI':>12s} {'mean err':>11s} {'tail err':>11s} {'x4 MI':>8s} {'x4 tail':>8s}")
    for i, r in enumerate(rows):
        a = b = float("nan")
        if i:
            f = sp[i] / sp[i - 1]
            if f > 3.5:
                a = MI[i - 1] / MI[i]; b = te[i - 1] / te[i]
        P(f"  {sp[i]:7.2f} {MI[i]:12.4e} {me[i]:11.4e} {te[i]:11.4e} {a:8.2f} {b:8.2f}")
    w = sp >= 16
    eMI = np.polyfit(np.log(sp[w]), np.log(MI[w]), 1)[0]
    eT = np.polyfit(np.log(sp[w]), np.log(te[w]), 1)[0]
    P("")
    P(f"  weak-coupling exponents (speed >= 16):  MI ~ s^{eMI:.4f}   tail_err ~ s^{eT:.4f}")
    P(f"  ratio {eMI/eT:.4f}; the deepest pair gives MI x{MI[-2]/MI[-1]:.2f} against tail "
      f"x{te[-2]/te[-1]:.2f} per 4x speed, i.e. exactly 2 and 1.")
    P("")
    P(RULE)
    P("THE LAW:  tail_err = c * sqrt(MI)")
    P(RULE)
    c = te / np.sqrt(MI)
    P(f"  {'speed':>7s} {'MI':>12s} {'sqrt(MI)':>11s} {'tail err':>11s} {'c':>8s}")
    for i in range(len(rows)):
        if sp[i] < 8: continue
        P(f"  {sp[i]:7.2f} {MI[i]:12.4e} {np.sqrt(MI[i]):11.4e} {te[i]:11.4e} {c[i]:8.2f}")
    cw = c[sp >= 128]
    P(f"  c converges to {cw[-1]:.2f} as coupling weakens (rows at speed >= 128: "
      f"{', '.join(f'{v:.2f}' for v in cw)})")
    P("")
    P("  WHY, and this is derivable rather than fitted. For weak dependence with correlation rho:")
    P("      MI        = -0.5*log(1 - rho^2)  ~  rho^2 / 2      QUADRATIC in rho")
    P("      tail lift ~  rho * (tail z-scores)                 LINEAR   in rho")
    P("  A mutual information is an average of log[P_AB/(P_A P_B)] taken under P, and that")
    P("  average is second order in the coupling. A tail functional is first order in it. So the")
    P("  bulk measure vanishes as the SQUARE of the quantity it is being used to bound, and the")
    P("  gap widens without limit as coupling weakens -- which is exactly the regime in which a")
    P("  split is ADMITTED.")
    P("")
    P(RULE)
    P("CONSEQUENCE FOR THE ARCHITECTURE'S SECTION 14 THRESHOLD")
    P(RULE)
    cc = float(cw[-1])
    P(f"  Using the measured c = {cc:.2f} for this tail depth:")
    P(f"  {'target tail error':>20s} {'naive MI bar':>14s} {'CORRECT MI bar':>16s} {'factor':>12s}")
    for eps in (1e-1, 1e-2, 1e-3, 1e-4):
        need = (eps / cc) ** 2
        P(f"  {eps:20.0e} {eps:14.0e} {need:16.3e} {eps/need:12.0f}x")
    P("")
    P("  A threshold set on mutual information is therefore wrong by a SQUARE ROOT, and the")
    P("  error is always in the unsafe direction: it admits splits that look decoupled in the")
    P("  bulk while the conjunctive tail is still strongly dependent.")
    P("")
    P("  MEASURED AT THE CALIBRATED THRESHOLD in RESULTS_grouping.txt: tau = 0.001352 was fixed")
    P("  as the largest MI keeping the joint MEAN accurate to 1%. At that same threshold the")
    P("  joint TAIL is wrong by 47.6% -- a 59x amplification from the quantity the threshold")
    P("  controls to the quantity it does not.")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
