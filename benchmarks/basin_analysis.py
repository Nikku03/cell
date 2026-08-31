"""Answers Q1, Q2, Q3 from the per-pose basin data. Bars are in db5_basin.py's docstring."""
from __future__ import annotations
import glob, json, sys
sys.path.insert(0, ".")
import numpy as np

SCORES = ["grid", "pair", "ve", "greedy", "F"]
OK = ("high", "medium", "acceptable")


def _rank(x):
    return np.argsort(np.argsort(np.asarray(x, dtype=float))).astype(float)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    ra, rb = _rank(a[m]), _rank(b[m])
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def spearman_p(rho, n):
    if not np.isfinite(rho) or n < 10:
        return float("nan")
    from math import erfc, sqrt
    z = abs(rho) * sqrt(n - 1)
    return float(erfc(z / sqrt(2.0)))


def load(pattern="benchmarks/basin_w*.json"):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            out += json.load(open(f))
        except Exception:
            pass
    return out


def main():
    data = load(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/basin_w*.json")
    data = [c for c in data if len(c["poses"]) >= 5]
    npose = sum(len(c["poses"]) for c in data)
    print(f"  {len(data)} complexes, {npose} poses\n")

    # ---------------- Q1 ----------------
    TS = np.array([p["TS"] for c in data for p in c["poses"]])
    IR = np.array([p["I_rmsd"] for c in data for p in c["poses"]])
    rho = spearman(TS, IR)
    pv = spearman_p(rho, len(TS))
    per = [spearman([p["TS"] for p in c["poses"]], [p["I_rmsd"] for p in c["poses"]])
           for c in data]
    per = np.array([x for x in per if np.isfinite(x)])
    print("  Q1  basin breadth T*S_conf vs interface RMSD")
    print(f"      pooled Spearman        {rho:+.4f}   (n = {len(TS)}, p = {pv:.2e})")
    print(f"      per-complex median     {np.median(per):+.4f}   "
          f"(negative on {int((per < 0).sum())}/{len(per)} complexes)")
    print(f"      T*S_conf range         {TS.min():.4f} to {TS.max():.4f} kcal/mol")
    q1 = bool(rho <= -0.10 and pv < 1e-3 and np.median(per) < 0)
    print(f"      Q1 {'PASS -- breadth tracks nativeness' if q1 else 'FAIL -- LINE CLOSED'}"
          f"   (bar: pooled <= -0.10, p < 1e-3, median < 0)")

    # ---------------- Q2 ----------------
    print("\n  Q2  rank-1 CAPRI success, same shortlists, different ranking")
    print(f"      {'ranking':>10s} {'successes':>10s} {'median rank-1 I-RMSD':>22s}")
    res = {}
    for s in SCORES:
        wins, r1 = 0, []
        for c in data:
            k = int(np.argmin([p[s] for p in c["poses"]]))
            r1.append(c["poses"][k]["I_rmsd"])
            wins += c["poses"][k]["quality"] in OK
        res[s] = (wins, float(np.median(r1)))
        print(f"      {s:>10s} {wins:6d}/{len(data):<4d} {np.median(r1):22.3f}")
    ceil = sum(any(p["quality"] in OK for p in c["poses"]) for c in data)
    best = float(np.median([min(p["I_rmsd"] for p in c["poses"]) for c in data]))
    print(f"      {'CEILING':>10s} {ceil:6d}/{len(data):<4d} {best:22.3f}   "
          f"(best pose present in the shortlist at all)")
    fwin, bwin = res["F"][0], res["ve"][0]
    print(f"      F vs E_min baseline: {fwin} vs {bwin}")
    q2 = bool(fwin >= 3 and fwin > bwin)
    print(f"      Q2 {'PASS' if q2 else 'NULL -- LINE CLOSED'}   "
          f"(bar: F >= 3 successes and above the E_min baseline)")

    # ---------------- Q3 ----------------
    print("\n  Q3  are the five scorers' errors correlated? (rank error vs true I-RMSD)")
    errs = {s: [] for s in SCORES}
    for c in data:
        tr = _rank([p["I_rmsd"] for p in c["poses"]])
        for s in SCORES:
            errs[s].append(_rank([p[s] for p in c["poses"]]) - tr)
    errs = {s: np.concatenate(v) for s, v in errs.items()}
    print("           " + " ".join(f"{s:>8s}" for s in SCORES))
    off = []
    for i, a in enumerate(SCORES):
        row = []
        for j, b in enumerate(SCORES):
            r = 1.0 if i == j else spearman(errs[a], errs[b])
            row.append(r)
            if i < j:
                off.append(abs(r))
        print(f"      {a:>5s} " + " ".join(f"{v:+8.3f}" for v in row))
    mo = float(np.mean(off))
    verdict = ("SHARED BIAS -- an ensemble treatment of the same energy cannot fix it"
               if mo > 0.5 else
               "INDEPENDENT ROUGHNESS -- averaging or an ensemble can help" if mo < 0.3
               else "INCONCLUSIVE (0.3-0.5)")
    print(f"      mean off-diagonal |rho| = {mo:.3f}  ->  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
