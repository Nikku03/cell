"""Answers the gates predeclared in db5_unselected.py. No bar is defined in this file.

Every threshold used here is quoted from that module's docstring, which was committed before
the run. This file is arithmetic on the shards, not a place to decide what counts as a pass.
"""
from __future__ import annotations
import glob, json, sys
sys.path.insert(0, ".")
import numpy as np

OK = ("high", "medium", "acceptable")
TOPK = 20
ALPHA = 0.05                   # one-sided exact p for the retrieval gate
Q1_RHO = -0.10                 # the ORIGINAL Q1 effect-size bar, unchanged
PRIOR_RHO = -0.4498            # what the score-selected shortlist reported, for comparison


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


def partial(a, b, *ctrl):
    """Spearman(a,b) with one or more controls held fixed, by rank-residualising."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    C = [np.asarray(c, float) for c in ctrl]
    m = np.isfinite(a) & np.isfinite(b)
    for c in C:
        m &= np.isfinite(c)
    if m.sum() < 10:
        return float("nan")
    ra, rb = _rank(a[m]), _rank(b[m])
    X = np.column_stack([np.ones(int(m.sum()))] + [_rank(c[m]) for c in C])
    def resid(y):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta
    return spearman(resid(ra), resid(rb))


def sign_test(vals, alt_negative=True):
    """Exact two-sided binomial sign test on per-complex statistics -- the complex is the unit
    of replication, not the pose. Returns (n_neg, n, p)."""
    from math import comb
    v = np.asarray([x for x in vals if np.isfinite(x) and x != 0.0], float)
    n = len(v)
    k = int((v < 0).sum()) if alt_negative else int((v > 0).sum())
    if n == 0:
        return 0, 0, float("nan")
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n
    return k, n, float(min(1.0, 2.0 * tail))


def poisson_binomial_tail(ps, x):
    """Exact P(X >= x) for independent Bernoulli(ps), by dynamic programming.

    A normal approximation misstates the tail of a small-mean skewed discrete distribution,
    which is how one lucky complex gets reported as signal.
    """
    ps = [float(p) for p in ps]
    dist = np.zeros(len(ps) + 1); dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return float(dist[int(x):].sum())


def chance_hit(N, k, top=TOPK):
    """P(a random top-`top` of N contains >= 1 of the k acceptable)."""
    if k <= 0 or N <= 0:
        return 0.0
    q = 1.0
    for j in range(min(top, N)):
        num = N - k - j
        if num <= 0:
            return 1.0
        q *= num / (N - j)
    return float(1.0 - q)


def load(pattern="benchmarks/unsel_w*.json"):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            out += json.load(open(f))
        except Exception:
            pass
    return out


def rank_norm(x, descending=False):
    r = _rank(np.asarray(x, float))
    if descending:
        r = len(r) - 1 - r
    return r / max(1.0, len(r) - 1)


def main():
    data = load(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/unsel_w*.json")
    data = [c for c in data if len(c["poses"]) >= 20]
    if not data:
        print("  no shards")
        return 1
    npose = sum(len(c["poses"]) for c in data)
    nd = sum(1 for c in data for p in c["poses"] if p["degenerate"])
    print(f"  {len(data)} complexes, {npose} sampled poses, {nd} degenerate "
          f"({100.0 * nd / npose:.1f}%) excluded from Q1\n")

    # ---------------- STEP 0: the reachable-set ceiling ----------------
    print("  STEP 0  THE REACHABLE-SET CEILING -- could this search produce an acceptable "
          "pose AT ALL?")
    nacc = np.array([c["ceiling"]["n_acceptable"] for c in data])
    nen = np.array([c["ceiling"]["n_enumerated"] for c in data])
    capped = [c["id"] for c in data if c["ceiling"]["capped"]]
    lmin = np.array([c["ceiling"]["min_L_rmsd_over_rotations"] for c in data])
    ceil = int((nacc > 0).sum())
    print(f"      translations enumerated in closed form  {int(nen.sum())} total, "
          f"{int(np.median(nen))} median/complex")
    print(f"      enumeration cap hit on                  {len(capped)} complexes"
          f"{'  ' + ','.join(capped[:8]) if capped else ''}")
    print(f"      best L_rmsd ANY rotation could reach    min {lmin.min():.2f}, median "
          f"{np.median(lmin):.2f} A   (rotation-set limit, translation optimal)")
    print(f"      CEILING                                 {ceil}/{len(data)} complexes can "
          f"produce >= 1 CAPRI-acceptable pose")
    # the screen behind the I_rmsd branch, validated rather than assumed
    md = [c["screen"]["max_direct_given_I_ok"] for c in data
          if c["screen"]["max_direct_given_I_ok"] is not None]
    nw = sum(c["screen"]["n_within_I_bar"] for c in data)
    if md:
        print(f"      screen validation: {nw} poses had exact I_rmsd <= 4; the largest DIRECT "
              f"interface rmsd among them was {max(md):.2f} A vs a screen at "
              f"{data[0]['screen']['limit']:.0f} A -> "
              f"{'SAFE' if max(md) < data[0]['screen']['limit'] else 'NOT SAFE'}")
    else:
        print(f"      screen validation: no pose anywhere reached exact I_rmsd <= 4, so the "
              f"I-branch screen was never exercised (the L-branch is exact by construction)")

    # ---------------- STEP 2: Q1 off the collider ----------------
    # Runs regardless of the ceiling: a correlation needs variation in I_rmsd, not acceptable
    # poses. Degenerate poses carry TS = nan and drop out of every statistic below.
    print("\n  STEP 2  Q1 RETEST -- does basin breadth track nativeness when the poses were "
          "NOT chosen by a scorer?")
    TS = np.array([p["TS"] for c in data for p in c["poses"]], float)
    IR = np.array([p["I_rmsd"] for c in data for p in c["poses"]], float)
    CN = np.array([p["contacts"] for c in data for p in c["poses"]], float)
    NR = np.array([p["n_repack"] for c in data for p in c["poses"]], float)
    rho = spearman(TS, IR)
    per = np.array([spearman([p["TS"] for p in c["poses"]],
                             [p["I_rmsd"] for p in c["poses"]]) for c in data])
    k, n, psign = sign_test(per)
    print(f"      pooled Spearman(T*S_conf, I_rmsd)     {rho:+.4f}   "
          f"(prior, score-selected: {PRIOR_RHO:+.4f})")
    print(f"      per-complex median                    {np.median(per[np.isfinite(per)]):+.4f}")
    print(f"      sign test over complexes              {k}/{n} negative, p = {psign:.2e}"
          f"   <- the complex is the unit of replication, not the pose")
    print(f"      partial | geometric contact count     "
          f"{partial(TS, IR, CN):+.4f}")
    print(f"      partial | n_repack (extensivity)      {partial(TS, IR, NR):+.4f}")
    print(f"      partial | BOTH                        {partial(TS, IR, CN, NR):+.4f}")
    par = partial(TS, IR, CN, NR)
    q1 = bool(rho <= Q1_RHO and np.median(per[np.isfinite(per)]) < 0
              and psign < ALPHA and np.isfinite(par) and par <= Q1_RHO)
    print(f"      Q1 {'REPLICATES off the collider' if q1 else 'FAILS'}   "
          f"(bar: rho <= {Q1_RHO}, majority negative by sign test, and survives the size "
          f"control)")
    if np.isfinite(rho) and rho < 0 and PRIOR_RHO < 0:
        print(f"      effect retained vs the score-selected estimate: "
              f"{100.0 * rho / PRIOR_RHO:.0f}%")
    fin = np.isfinite(IR)
    print(f"      domain of the claim: I_rmsd spans {IR[fin].min():.2f} to "
          f"{IR[fin].max():.2f} A")

    # THE COLLIDER, MEASURED. Same poses, same complexes; only the selection rule changes.
    sel, whole = [], []
    for c in data:
        ps = c["poses"]
        if len(ps) < 40:
            continue
        g = np.array([p["grid"] for p in ps], float)          # lower = better grid score
        t = np.array([p["TS"] for p in ps], float)
        i_ = np.array([p["I_rmsd"] for p in ps], float)
        top = np.argsort(g)[:TOPK]
        sel.append(spearman(t[top], i_[top]))
        whole.append(spearman(t, i_))
    sel = np.array([x for x in sel if np.isfinite(x)])
    whole = np.array([x for x in whole if np.isfinite(x)])
    print(f"      COLLIDER CHECK -- the prior study's selection rule reapplied to THIS sample:")
    print(f"        top {TOPK} by grid score   median rho {np.median(sel):+.4f}  (n={len(sel)})")
    print(f"        whole sample           median rho {np.median(whole):+.4f}  (n={len(whole)})")
    print(f"        -> a correlation much stronger in the selected slice is the selection "
          f"effect itself.")

    # ---------------- STEP 3: retrieval, power first ----------------
    print(f"\n  STEP 3  RETRIEVAL -- how often does an acceptable pose reach the top {TOPK}?")
    usable = [c for c in data if any(p["quality"] in OK for p in c["poses"])]
    if not usable:
        print(f"      VOID. No sampled pose in ANY of {len(data)} complexes is "
              f"CAPRI-acceptable, so an oracle scores 0 and no ranking -- energy, entropy, "
              f"blend, size, or one reading the answer key -- could score above it.")
        print(f"      Declared VOID and NOT null (ledger defect N): the test has no power, so "
              f"it reports nothing about the rankings.")
        print(f"      The finding is STEP 0's: the SEARCH is the binding constraint.")
        return 0
    ps_chance = []
    for c in usable:
        N = len(c["poses"]); kk = sum(p["quality"] in OK for p in c["poses"])
        ps_chance.append(chance_hit(N, kk))
    oracle = len(usable)
    # THE POWER CHECK, BEFORE ANY VERDICT (ledger defect N).
    p_oracle = poisson_binomial_tail(ps_chance, oracle)
    print(f"      complexes with >= 1 acceptable pose in the sample: {oracle}/{len(data)}")
    print(f"      chance expectation {np.sum(ps_chance):.2f} hits; an ORACLE scores {oracle}, "
          f"exact p = {p_oracle:.3g}")
    if p_oracle >= ALPHA:
        print(f"      VOID. Even a perfect oracle cannot reach p < {ALPHA} against this "
              f"chance baseline, so the gate cannot be passed by any ranking and has no "
              f"power (ledger defect N). No verdict is issued.")
        return 0
    rankings = {
        "energy": lambda q: rank_norm([p["ve"] for p in q]),
        "entropy": lambda q: rank_norm([p["TS"] for p in q], descending=True),
        "50/50": lambda q: 0.5 * rank_norm([p["ve"] for p in q])
                         + 0.5 * rank_norm([p["TS"] for p in q], descending=True),
        "size(ctrl)": lambda q: rank_norm([p["contacts"] for p in q], descending=True),
    }
    hits = {}
    print(f"      {'ranking':>12s} {'hits':>6s} {'exact p':>10s}  verdict")
    for name, fn in rankings.items():
        h, per_c = 0, []
        for c in usable:
            q = c["poses"]
            order = np.argsort(fn(q))[:TOPK]
            got = any(q[int(j)]["quality"] in OK for j in order)
            per_c.append(bool(got)); h += got
        hits[name] = (h, per_c)
        pv = poisson_binomial_tail(ps_chance, h)
        print(f"      {name:>12s} {h:6d} {pv:10.3g}  "
              f"{'SIGNAL' if pv < ALPHA else 'not above chance'}")
    # The actual hypothesis: does entropy ADD to energy? Paired, same complexes.
    a = np.array(hits["energy"][1]); c50 = np.array(hits["50/50"][1])
    gain = int((c50 & ~a).sum()); loss = int((a & ~c50).sum())
    from math import comb
    nb = gain + loss
    pmc = (sum(comb(nb, i) for i in range(gain, nb + 1)) / 2.0 ** nb) if nb else float("nan")
    print(f"      PAIRED, the hypothesis itself: 50/50 vs energy alone -- "
          f"{gain} complexes gained, {loss} lost, exact one-sided p = {pmc:.3g}")
    print(f"      and against the size control: entropy {hits['entropy'][0]} vs "
          f"size {hits['size(ctrl)'][0]} hits "
          f"-- beating chance is not the bar, beating size is.")
    print(f"      lambda sweep of rank(E) + lambda*rank(-T*S) -- DIAGNOSTIC, NOT A GATE:")
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 1e9):
        h = 0
        for c in usable:
            q = c["poses"]
            s = rank_norm([p["ve"] for p in q]) + lam * rank_norm(
                [p["TS"] for p in q], descending=True)
            h += any(q[int(j)]["quality"] in OK for j in np.argsort(s)[:TOPK])
        tag = "entropy only" if lam > 1e8 else ("energy only" if lam == 0 else "")
        print(f"        lambda={lam:<8.2f} hits={h:3d}  {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
