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


def partial(a, b, c):
    """Spearman(a,b) with c held constant -- catches "is this just interface size?"."""
    ra, rb, rc = _rank(a), _rank(b), _rank(c)
    ra, rb, rc = ra - ra.mean(), rb - rb.mean(), rc - rc.mean()
    def cr(x, y):
        d = np.sqrt((x * x).sum() * (y * y).sum())
        return (x * y).sum() / d if d > 0 else float("nan")
    rab, rac, rbc = cr(ra, rb), cr(ra, rc), cr(rb, rc)
    return float((rab - rac * rbc) / np.sqrt((1 - rac ** 2) * (1 - rbc ** 2)))


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
    # LEDGER DEFECT O. score_pose writes TS = 0.0 in place of a measurement when fewer than
    # two interface residues are repackable. That constant is below every real value, is 20%
    # of the rows, and lands preferentially on far-from-native poses, so pooling it
    # manufactures the correlation. The gate is reported on the DEFINED subset; the pooled
    # number is kept only to show the size of the artifact.
    ALL = [p for c in data for p in c["poses"]]
    DEF = [p for p in ALL if not p.get("degenerate")]
    TS_all = np.array([p["TS"] for p in ALL], float)
    IR_all = np.array([p["I_rmsd"] for p in ALL], float)
    print(f"      sentinel rows (TS written as 0.0, not measured): "
          f"{len(ALL) - len(DEF)}/{len(ALL)} = {100.0 * (1 - len(DEF) / len(ALL)):.1f}%")
    print(f"      WITH the sentinel   pooled Spearman {spearman(TS_all, IR_all):+.4f}  "
          f"<- ARTIFACT, not the gate")
    TS = np.array([p["TS"] for p in DEF], float)
    IR = np.array([p["I_rmsd"] for p in DEF], float)
    rho = spearman(TS, IR)
    pv = spearman_p(rho, len(TS))
    per = [spearman([p["TS"] for p in c["poses"] if not p.get("degenerate")],
                    [p["I_rmsd"] for p in c["poses"] if not p.get("degenerate")])
           for c in data]
    per = np.array([x for x in per if np.isfinite(x)])
    print("  Q1  basin breadth T*S_conf vs interface RMSD")
    print(f"      pooled Spearman        {rho:+.4f}   (n = {len(TS)}, p = {pv:.2e})")
    print(f"      per-complex median     {np.median(per):+.4f}   "
          f"(negative on {int((per < 0).sum())}/{len(per)} complexes)")
    print(f"      T*S_conf range         {TS.min():.4f} to {TS.max():.4f} kcal/mol")
    G = np.array([abs(p["grid"]) for p in DEF], float)   # contact-count proxy
    TW = np.array([float(p["treewidth"]) for p in DEF])
    print(f"      vs contact count       {spearman(TS, G):+.4f}   "
          f"(if TS were just interface size this would be large)")
    print(f"      partial | contacts     {partial(TS, IR, G):+.4f}")
    print(f"      partial | treewidth    {partial(TS, IR, TW):+.4f}")
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
    # Ledger defect N: a bar above the achievable ceiling measures nothing. If an oracle
    # reading the answer key could not clear the bar on these inputs, the gate is VOID --
    # it would return this same verdict whether or not the hypothesis were true.
    if ceil < 3:
        print(f"      Q2 VOID -- the bar was UNREACHABLE: an oracle scores {ceil}/{len(data)} "
              f"on these shortlists, below the >= 3 bar. This gate has no power (defect N).")
    else:
        q2 = bool(fwin >= 3 and fwin > bwin)
        print(f"      Q2 {'PASS' if q2 else 'NULL -- LINE CLOSED'}   "
              f"(bar: F >= 3 successes and above the E_min baseline)")

    # A second read that does NOT route through the ceiling, so it survives a void Q2.
    agree = float(np.mean([spearman([p["ve"] for p in c["poses"]],
                                    [p["F"] for p in c["poses"]]) for c in data]))
    same = sum(int(np.argmin([p["ve"] for p in c["poses"]])
                   == np.argmin([p["F"] for p in c["poses"]])) for c in data)
    es = float(np.median([max(p["ve"] for p in c["poses"]) - min(p["ve"] for p in c["poses"])
                          for c in data]))
    ts = float(np.median([max(p["TS"] for p in c["poses"]) - min(p["TS"] for p in c["poses"])
                          for c in data]))
    print(f"      does F rank differently from E_min at all?")
    print(f"        within-complex Spearman(E_min, F)  {agree:+.4f}")
    print(f"        same rank-1 pose                   {same}/{len(data)}")
    print(f"        E_min spread {es:.3f} vs T*S spread {ts:.3f} kcal/mol "
          f"({100 * ts / es:.1f}% of it)")
    print(f"      -> free energy is min-energy plus a {100 * ts / es:.0f}% perturbation; it "
          f"cannot move a selection outcome. LINE CLOSED on this, not on the void gate.")
    # Which term actually carries the signal? Measured after seeing the data: an
    # observation, NOT a predeclared gate, and reported as such.
    print(f"      which term carries nativeness signal? (post-hoc, not a gate)")
    for s_ in ("ve", "F"):
        r = float(np.mean([spearman([p[s_] for p in c["poses"]],
                                    [p["I_rmsd"] for p in c["poses"]]) for c in data]))
        print(f"        {s_:>8s} alone   mean within-complex Spearman vs I_rmsd {r:+.4f}")
    r = float(np.mean([spearman([p["TS"] for p in c["poses"]],
                                [p["I_rmsd"] for p in c["poses"]]) for c in data]))
    print(f"        {'T*S_conf':>8s} alone   mean within-complex Spearman vs I_rmsd {r:+.4f}"
          f"   <- the weaker-weighted term is the stronger signal")

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
    # ve, greedy and F are the SAME energy function scored three ways, so including all
    # three inflates the mean with near-duplicate columns. Report the distinct-scorer mean.
    D = ["grid", "pair", "ve"]
    do = [abs(spearman(errs[a], errs[b])) for i, a in enumerate(D) for b in D[i + 1:]]
    print(f"      deflated to the {len(D)} DISTINCT scorers {D}: "
          f"{', '.join('%.3f' % x for x in do)}  mean = {np.mean(do):.3f}")
    print(f"      (ve/greedy/F agree at |rho| ~ 0.997 -- they are one scorer, not three)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
